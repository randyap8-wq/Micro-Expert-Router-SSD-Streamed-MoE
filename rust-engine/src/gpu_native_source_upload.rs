//! Bounded real-inference source/upload mechanism. Source I/O lives here;
//! physical installation receives only
//! a one-shot, identity-checked unmapped lease, never a storage handle.
use crate::backend::gpu_native::GpuNativeExecutorContext;
use crate::buffer_pool::PooledBuffer;
use crate::expert_cache::{ExpertResident, GpuExpertCache};
use crate::io_provider::NvmeStorage;
use crate::tensor_header::{TensorHeader, UthDtypeId};
use parking_lot::Mutex;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

pub(crate) const FULL: usize = 2_658_304;
pub(crate) const PAYLOAD: usize = 2_654_208;
pub(crate) const ALIGN: usize = 4096;
pub(crate) const ARENAS: usize = 2;
pub(crate) const ARENA_WIDTH: usize = 8;
pub(crate) const CAPACITY: usize = ARENAS * ARENA_WIDTH;
pub(crate) const UPLOAD_BYTES: usize = ARENA_WIDTH * FULL + ALIGN;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum GpuNativeSourceToUploadCopyElisionQualificationArm {
    Control,
    Treatment,
}
pub(crate) use GpuNativeSourceToUploadCopyElisionQualificationArm as Arm;

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct Metrics {
    pub(crate) acquisition_attempts: u64,
    pub(crate) acquisition_waits: u64,
    pub(crate) acquisition_wait_us: u64,
    pub(crate) high_water: u64,
    pub(crate) map_attempts: u64,
    pub(crate) map_completions: u64,
    pub(crate) map_wait_us: u64,
    pub(crate) unmaps: u64,
    pub(crate) remap_attempts: u64,
    pub(crate) remap_completions: u64,
    pub(crate) remap_failures: u64,
    pub(crate) remap_wait_us: u64,
    // Map counters above still count actual WGPU mapping operations/time.
    // Lease, acquisition and high_water counters retain expert-level units.
    pub(crate) arena_map_attempts: u64,
    pub(crate) arena_map_completions: u64,
    pub(crate) arena_unmaps: u64,
    pub(crate) arena_remap_attempts: u64,
    pub(crate) arena_remap_completions: u64,
    pub(crate) arena_remap_failures: u64,
    pub(crate) arena_map_wait_us: u64,
    pub(crate) arena_high_water: u64,
    pub(crate) source_sets_mapped: u64,
    pub(crate) source_slots_mapped: u64,
    /// Successful original source calls that map two new arena generations.
    pub(crate) source_sets_opening_two_arenas: u64,
    pub(crate) arena_generation_errors: u64,
    pub(crate) alignment_failures: u64,
    pub(crate) leases_created: u64,
    pub(crate) leases_consumed: u64,
    pub(crate) leases_released: u64,
    pub(crate) leases_dropped_unconsumed: u64,
    pub(crate) mapped_direct_io_rejections: u64,
    pub(crate) odirect_observations: u64,
    pub(crate) direct_source_reads: u64,
    pub(crate) direct_source_bytes: u64,
    pub(crate) direct_payload_bytes: u64,
    pub(crate) source_failures: u64,
    pub(crate) source_fallback_reads: u64,
    pub(crate) fused_source_us: u64,
    pub(crate) logical_materialization_operations: u64,
    pub(crate) logical_materialization_bytes: u64,
    pub(crate) logical_materialization_us: u64,
    pub(crate) shared_payload_constructions: u64,
    pub(crate) shared_payload_reuse: u64,
    pub(crate) non_shared_logical_fallbacks: u64,
    pub(crate) non_shared_logical_rejections: u64,
    pub(crate) logical_admissions: u64,
    pub(crate) logical_generation_observations: u64,
    pub(crate) total_payload_bytes_staged: u64,
    pub(crate) physical_cpu_payload_copy_bytes: u64,
    pub(crate) fallback_installs: u64,
    pub(crate) fallback_payload_copy_bytes: u64,
    pub(crate) fallback_payload_copy_us: u64,
    pub(crate) fused_installs: u64,
    pub(crate) fused_gpu_copy_bytes: u64,
    pub(crate) fused_install_sets: u64,
    pub(crate) copy_command_buffers: u64,
    pub(crate) copy_submissions: u64,
    pub(crate) copied_experts: u64,
    pub(crate) copied_bytes: u64,
    pub(crate) copy_encode_us: u64,
    pub(crate) copy_submit_us: u64,
    pub(crate) copy_failures: u64,
    pub(crate) accounting_errors: u64,
}
impl Metrics {
    pub(crate) fn add(&mut self, field: fn(&mut Self) -> &mut u64, n: u64) {
        let value = field(self);
        if let Some(sum) = value.checked_add(n) {
            *value = sum;
        } else {
            self.accounting_errors = self.accounting_errors.saturating_add(1);
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub(crate) struct Snapshot {
    /// Additive diagnostic, sampled from storage at the idle engine boundary.
    pub(crate) source_upload_fd_proof: Option<crate::io_provider::SourceUploadFdProofSnapshot>,
    pub(crate) arm: Arm,
    pub(crate) production_owned: bool,
    pub(crate) ring_capacity: usize,
    pub(crate) active_leases: usize,
    pub(crate) pending_leases: usize,
    pub(crate) ordered_nvme_ids_sha256: String,
    pub(crate) logical_admission_ids_sha256: String,
    pub(crate) logical_generation_ids_sha256: String,
    #[serde(flatten)]
    pub(crate) metrics: Metrics,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ArenaPhase {
    Available,
    Mapping,
    OpenMapped,
    Ready,
    Poisoned,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LeasePhase {
    Free,
    Reserved,
    Pending,
    Ready,
    Encoded,
    Submitted,
    Released,
}

/// One mapping remains live while disjoint source slots accumulate. Reserved
/// slots pin that mapping through all temporary views and I/O. Sealing is per
/// arena, under its state lock; a released prefix never enables append/reuse.
#[derive(Debug)]
struct ArenaGeneration {
    phase: ArenaPhase,
    generation: u64,
    leases: [LeasePhase; ARENA_WIDTH],
    // BUFFER offset established from the first mapped view, not a host pointer.
    aligned_base: Option<usize>,
    ever_mapped: bool,
    map_pending: bool,
    mapping_live: bool,
}
impl Default for ArenaGeneration {
    fn default() -> Self {
        Self {
            phase: ArenaPhase::Available,
            generation: 0,
            leases: [LeasePhase::Free; ARENA_WIDTH],
            aligned_base: None,
            ever_mapped: false,
            map_pending: false,
            mapping_live: false,
        }
    }
}
fn generation_error(m: &mut Metrics) -> String {
    m.add(|m| &mut m.arena_generation_errors, 1);
    m.add(|m| &mut m.accounting_errors, 1);
    "invalid upload arena generation/lifecycle".into()
}
impl ArenaGeneration {
    fn free_slots(&self) -> Vec<usize> {
        if !matches!(self.phase, ArenaPhase::Available | ArenaPhase::OpenMapped)
            || self.leases.contains(&LeasePhase::Reserved)
        {
            return Vec::new();
        }
        self.leases
            .iter()
            .enumerate()
            .filter_map(|(i, p)| (*p == LeasePhase::Free).then_some(i))
            .collect()
    }
    fn reserved_is(&self, r: &SlotReservation) -> bool {
        self.generation == r.generation
            && matches!(self.phase, ArenaPhase::Mapping | ArenaPhase::OpenMapped)
            && !r.slots.is_empty()
            && r.slots.iter().enumerate().all(|(i, &slot)| {
                slot < ARENA_WIDTH
                    && !r.slots[..i].contains(&slot)
                    && self.leases[slot] == LeasePhase::Reserved
            })
    }
    fn mapped(&mut self, generation: u64, base: usize, m: &mut Metrics) -> Result<(), String> {
        if self.generation != generation
            || self.phase != ArenaPhase::Mapping
            || slot_offset(base, ARENA_WIDTH - 1, UPLOAD_BYTES).is_err()
        {
            return Err(generation_error(m));
        }
        self.aligned_base = Some(base);
        self.ever_mapped = true;
        self.map_pending = false;
        self.mapping_live = true;
        self.phase = ArenaPhase::OpenMapped;
        Ok(())
    }
    fn populate(&mut self, r: &SlotReservation, m: &mut Metrics) -> Result<(), String> {
        if !self.reserved_is(r) || self.phase != ArenaPhase::OpenMapped || !self.mapping_live {
            return Err(generation_error(m));
        }
        for &slot in &r.slots {
            self.leases[slot] = LeasePhase::Pending;
        }
        Ok(())
    }
    fn seal(
        &mut self,
        generation: u64,
        m: &mut Metrics,
        unmap: impl FnOnce(),
    ) -> Result<(), String> {
        if self.generation != generation {
            return Err(generation_error(m));
        }
        match self.phase {
            ArenaPhase::Ready => Ok(()),
            ArenaPhase::OpenMapped
                if !self.leases.contains(&LeasePhase::Reserved)
                    && self.aligned_base.is_some()
                    && self.mapping_live =>
            {
                // Reserved slots are held until every BufferViewMut is dropped.
                // The caller holds this arena's lock across the actual unmap.
                unmap();
                self.mapping_live = false;
                m.add(|m| &mut m.unmaps, 1);
                m.add(|m| &mut m.arena_unmaps, 1);
                for phase in &mut self.leases {
                    if *phase == LeasePhase::Pending {
                        *phase = LeasePhase::Ready;
                    }
                }
                self.phase = ArenaPhase::Ready;
                Ok(())
            }
            _ => Err(generation_error(m)),
        }
    }
    fn lease_is(&self, generation: u64, slot: usize, phase: LeasePhase) -> bool {
        self.generation == generation
            && self.phase == ArenaPhase::Ready
            && slot < ARENA_WIDTH
            && self.leases[slot] == phase
    }
    fn lease_transition(
        &mut self,
        generation: u64,
        slot: usize,
        expected: LeasePhase,
        next: LeasePhase,
        m: &mut Metrics,
    ) -> Result<(), String> {
        if !self.lease_is(generation, slot, expected)
            || !matches!(
                (expected, next),
                (LeasePhase::Ready, LeasePhase::Encoded)
                    | (LeasePhase::Encoded, LeasePhase::Submitted)
                    | (LeasePhase::Encoded, LeasePhase::Ready)
            )
        {
            return Err(generation_error(m));
        }
        self.leases[slot] = next;
        Ok(())
    }
    fn release(&mut self, generation: u64, slot: usize, m: &mut Metrics) -> Result<(), String> {
        if self.generation != generation
            || self.phase != ArenaPhase::Ready
            || slot >= ARENA_WIDTH
            || !matches!(
                self.leases[slot],
                LeasePhase::Ready | LeasePhase::Encoded | LeasePhase::Submitted
            )
        {
            return Err(generation_error(m));
        }
        if self.leases[slot] == LeasePhase::Encoded {
            // The command may still exist or submission may have unwound.
            self.phase = ArenaPhase::Poisoned;
            return Err(generation_error(m));
        }
        if self.leases[slot] != LeasePhase::Submitted {
            m.add(|m| &mut m.leases_dropped_unconsumed, 1);
        }
        self.leases[slot] = LeasePhase::Released;
        m.add(|m| &mut m.leases_released, 1);
        if self.active_leases() == 0 {
            self.make_available();
        }
        Ok(())
    }
    fn make_available(&mut self) {
        self.phase = ArenaPhase::Available;
        self.aligned_base = None;
        self.map_pending = false;
        self.mapping_live = false;
        self.leases.fill(LeasePhase::Free);
    }
    fn rollback(
        &mut self,
        r: &SlotReservation,
        m: &mut Metrics,
        unmap: impl FnOnce(),
    ) -> Result<(), String> {
        if !self.reserved_is(r) {
            return Err(generation_error(m));
        }
        for &slot in &r.slots {
            self.leases[slot] = LeasePhase::Free;
        }
        m.add(|m| &mut m.leases_dropped_unconsumed, r.slots.len() as u64);
        m.add(|m| &mut m.leases_released, r.slots.len() as u64);
        if self.active_leases() == 0 {
            // Also cancel any failed/incomplete map_async; count unmap only for
            // a successful live mapping, exactly as in the original metrics.
            if self.map_pending || self.mapping_live {
                unmap();
            }
            if self.mapping_live {
                m.add(|m| &mut m.unmaps, 1);
                m.add(|m| &mut m.arena_unmaps, 1);
            }
            self.make_available();
        }
        Ok(())
    }
    fn active_leases(&self) -> usize {
        self.leases
            .iter()
            .filter(|p| !matches!(p, LeasePhase::Free | LeasePhase::Released))
            .count()
    }
}
#[derive(Debug)]
struct SlotReservation {
    index: usize,
    generation: u64,
    slots: Vec<usize>,
    new_generation: bool,
}

/// Preflight the complete set while both short state locks are held. Allocation
/// prefers one arena with enough unused slots, then splits in arena/slot order
/// when only the combined free capacity fits. No waiting, whole-generation
/// allocation, or I/O.
fn reserve_slots<'a>(
    states: impl Iterator<Item = &'a Mutex<ArenaGeneration>>,
    width: usize,
    m: &mut Metrics,
) -> Result<Vec<SlotReservation>, String> {
    if width == 0 || width > CAPACITY {
        return Err(generation_error(m));
    }
    let mut states = states.map(Mutex::lock).collect::<Vec<_>>();
    let mut remaining = width;
    let mut reservations = Vec::new();
    let whole = states.iter().position(|a| a.free_slots().len() >= width);
    for (index, state) in states.iter().enumerate() {
        if whole.is_some_and(|chosen| chosen != index) {
            continue;
        }
        let slots = state
            .free_slots()
            .into_iter()
            .take(remaining)
            .collect::<Vec<_>>();
        if slots.is_empty() {
            continue;
        }
        let new_generation = state.phase == ArenaPhase::Available;
        let generation = if new_generation {
            state
                .generation
                .checked_add(1)
                .ok_or_else(|| generation_error(m))?
        } else {
            state.generation
        };
        remaining -= slots.len();
        reservations.push(SlotReservation {
            index,
            generation,
            slots,
            new_generation,
        });
        if remaining == 0 {
            break;
        }
    }
    if remaining != 0 {
        // True bounded capacity/busy mapping pressure is not a stale generation.
        return Err("upload expert-slot capacity unavailable".into());
    }
    for r in &reservations {
        let state = &mut states[r.index];
        if r.new_generation {
            state.generation = r.generation;
            state.phase = ArenaPhase::Mapping;
        }
        for &slot in &r.slots {
            state.leases[slot] = LeasePhase::Reserved;
        }
    }
    m.add(|m| &mut m.leases_created, width as u64);
    m.high_water = m
        .high_water
        .max(states.iter().map(|a| a.active_leases() as u64).sum());
    m.arena_high_water = m.arena_high_water.max(
        states
            .iter()
            .filter(|a| a.phase != ArenaPhase::Available)
            .count() as u64,
    );
    Ok(reservations)
}
struct Arena {
    buffer: wgpu::Buffer,
    state: Mutex<ArenaGeneration>,
}
struct Ring {
    executor: Arc<GpuNativeExecutorContext>,
    arenas: [Arena; ARENAS],
}

/// Declared before temporary views. Error/cancellation drops those views before
/// rolling back only this operation's Reserved slots, preserving older Pending
/// slots and their still-live mapping. No Lease escapes a partial source read.
struct SourceReservation {
    state: Arc<State>,
    slots: Vec<SlotReservation>,
    armed: bool,
}
impl SourceReservation {
    fn arena(&self, r: &SlotReservation) -> &Arena {
        &self.state.ring.as_ref().expect("treatment ring").arenas[r.index]
    }
    fn commit(mut self, ids: &[u32], payloads: &[Arc<[u8]>]) -> Result<Vec<Lease>, String> {
        let mut states = self
            .slots
            .iter()
            .map(|r| self.arena(r).state.lock())
            .collect::<Vec<_>>();
        if ids.len() != self.slots.iter().map(|r| r.slots.len()).sum::<usize>()
            || ids.len() != payloads.len()
            || states.iter().zip(&self.slots).any(|(a, r)| {
                !a.reserved_is(r)
                    || a.phase != ArenaPhase::OpenMapped
                    || !a.mapping_live
                    || a.aligned_base.is_none()
            })
        {
            return Err(generation_error(&mut self.state.metrics.lock()));
        }
        let offsets = states
            .iter()
            .zip(&self.slots)
            .flat_map(|(a, r)| {
                r.slots
                    .iter()
                    .map(|&slot| slot_offset(a.aligned_base.unwrap(), slot, UPLOAD_BYTES))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (a, r) in states.iter_mut().zip(&self.slots) {
            a.populate(r, &mut self.state.metrics.lock())?;
        }
        let mut leases = Vec::with_capacity(ids.len());
        for r in &self.slots {
            for &slot in &r.slots {
                let i = leases.len();
                leases.push(Lease {
                    state: self.state.clone(),
                    index: r.index,
                    generation: r.generation,
                    slot,
                    id: ids[i],
                    offset: offsets[i],
                    payload: Some(payloads[i].clone()),
                });
            }
        }
        drop(states);
        self.armed = false;
        self.state.add(|m| &mut m.source_sets_mapped, 1);
        self.state
            .add(|m| &mut m.source_slots_mapped, ids.len() as u64);
        if self.slots.iter().filter(|r| r.new_generation).count() == ARENAS {
            self.state.add(|m| &mut m.source_sets_opening_two_arenas, 1);
        }
        Ok(leases)
    }
}
impl Drop for SourceReservation {
    fn drop(&mut self) {
        if self.armed {
            for r in &self.slots {
                let arena = self.arena(r);
                let _ = arena
                    .state
                    .lock()
                    .rollback(r, &mut self.state.metrics.lock(), || arena.buffer.unmap());
            }
        }
    }
}

/// Safe disjoint slicing shared with the portable source-order regressions.
/// Reservation slot indices are ascending, but may contain rollback holes.
fn source_destinations<'a>(
    view: &'a mut [u8],
    base: usize,
    slots: &[usize],
) -> Result<Vec<&'a mut [u8]>, String> {
    let capacity = view.len();
    let mut tail = view;
    let mut end = 0;
    let mut destinations = Vec::with_capacity(slots.len());
    for &slot in slots {
        let start = slot_offset(base, slot, capacity)?;
        let skip = start.checked_sub(end).ok_or("overlapping source slots")?;
        let (_, rest) = tail.split_at_mut(skip);
        let (full, rest) = rest.split_at_mut(FULL);
        destinations.push(full);
        tail = rest;
        end = start + FULL;
    }
    Ok(destinations)
}

pub(crate) struct State {
    pub(crate) arm: Arm,
    production_owned: bool,
    production_demand_gate: Arc<Semaphore>,
    ring: Option<Ring>,
    pub(crate) metrics: Mutex<Metrics>,
    pending: Mutex<HashMap<u32, Lease>>,
    nvme_ids: Mutex<Sha256>,
    logical_ids: Mutex<Sha256>,
    generations: Mutex<Sha256>,
}

pub(crate) struct ProductionDemandGuard {
    state: Arc<State>,
    _permit: OwnedSemaphorePermit,
}

impl ProductionDemandGuard {
    pub(crate) fn state(&self) -> &Arc<State> {
        &self.state
    }
}

impl Drop for ProductionDemandGuard {
    fn drop(&mut self) {
        // This guard is the sole production owner of the ring while alive, so
        // request failure/cancellation can safely clear its pending source leases.
        self.state.abandon_pending();
    }
}

impl State {
    pub(crate) fn new(
        arm: Arm,
        executor: Arc<GpuNativeExecutorContext>,
    ) -> Result<Arc<Self>, String> {
        Self::new_with_ownership(arm, executor, false)
    }

    pub(crate) fn new_production(
        executor: Arc<GpuNativeExecutorContext>,
    ) -> Result<Arc<Self>, String> {
        Self::new_with_ownership(Arm::Treatment, executor, true)
    }

    fn new_with_ownership(
        arm: Arm,
        executor: Arc<GpuNativeExecutorContext>,
        production_owned: bool,
    ) -> Result<Arc<Self>, String> {
        if production_owned && arm != Arm::Treatment {
            return Err("production source/upload state must use the treatment mechanism".into());
        }
        let ring = if arm == Arm::Treatment {
            let gpu = executor.authoritative_gpu().map_err(|e| e.to_string())?;
            let arenas = std::array::from_fn(|_| Arena {
                buffer: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("source-to-upload bounded eight-slot arena"),
                    size: UPLOAD_BYTES as u64,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                state: Mutex::new(ArenaGeneration::default()),
            });
            Some(Ring { executor, arenas })
        } else {
            None
        };
        Ok(Arc::new(Self {
            arm,
            production_owned,
            production_demand_gate: Arc::new(Semaphore::new(1)),
            ring,
            metrics: Mutex::new(Metrics::default()),
            pending: Mutex::new(HashMap::new()),
            nvme_ids: Mutex::new(Sha256::new()),
            logical_ids: Mutex::new(Sha256::new()),
            generations: Mutex::new(Sha256::new()),
        }))
    }

    #[cfg(test)]
    pub(crate) fn cpu_test_state(arm: Arm) -> Arc<Self> {
        Self::cpu_test_state_with_ownership(arm, false)
    }

    #[cfg(test)]
    pub(crate) fn cpu_test_production_state() -> Arc<Self> {
        Self::cpu_test_state_with_ownership(Arm::Treatment, true)
    }

    #[cfg(test)]
    fn cpu_test_state_with_ownership(arm: Arm, production_owned: bool) -> Arc<Self> {
        Arc::new(Self {
            arm,
            production_owned,
            production_demand_gate: Arc::new(Semaphore::new(1)),
            ring: None,
            metrics: Mutex::new(Metrics::default()),
            pending: Mutex::new(HashMap::new()),
            nvme_ids: Mutex::new(Sha256::new()),
            logical_ids: Mutex::new(Sha256::new()),
            generations: Mutex::new(Sha256::new()),
        })
    }

    pub(crate) fn is_production_owned(&self) -> bool {
        self.production_owned
    }

    pub(crate) fn try_begin_production_demand(
        self: &Arc<Self>,
    ) -> Option<ProductionDemandGuard> {
        if !self.production_owned || self.arm != Arm::Treatment {
            return None;
        }
        let permit = self
            .production_demand_gate
            .clone()
            .try_acquire_owned()
            .ok()?;
        Some(ProductionDemandGuard {
            state: self.clone(),
            _permit: permit,
        })
    }

    pub(crate) fn can_fuse_source_set(
        &self,
        ids: &[u32],
        logical: &GpuExpertCache,
    ) -> bool {
        self.arm == Arm::Treatment
            && !ids.is_empty()
            && ids.len() <= CAPACITY
            && !ids.iter().any(|id| self.pending.lock().contains_key(id))
            && ids.iter().all(|id| {
                logical
                    .current_admission(*id)
                    .is_none_or(|a| a.resident().qualification_shared_payload().is_some())
            })
    }

    pub(crate) fn add(&self, field: fn(&mut Metrics) -> &mut u64, n: u64) {
        self.metrics.lock().add(field, n);
    }
    pub(crate) fn snapshot(&self) -> Snapshot {
        Snapshot {
            source_upload_fd_proof: None,
            arm: self.arm,
            production_owned: self.production_owned,
            ring_capacity: self
                .ring
                .as_ref()
                .map_or(0, |r| r.arenas.len() * ARENA_WIDTH),
            active_leases: self.active_leases(),
            pending_leases: self.pending.lock().len(),
            metrics: self.metrics.lock().clone(),
            ordered_nvme_ids_sha256: format!("{:x}", self.nvme_ids.lock().clone().finalize()),
            logical_admission_ids_sha256: format!(
                "{:x}",
                self.logical_ids.lock().clone().finalize()
            ),
            logical_generation_ids_sha256: format!(
                "{:x}",
                self.generations.lock().clone().finalize()
            ),
        }
    }
    fn active_leases(&self) -> usize {
        self.ring.as_ref().map_or(0, |r| {
            r.arenas
                .iter()
                .map(|a| a.state.lock().active_leases())
                .sum()
        })
    }
    fn arenas_idle(&self) -> bool {
        self.ring.as_ref().is_none_or(|r| {
            r.arenas
                .iter()
                .all(|a| a.state.lock().phase == ArenaPhase::Available)
        })
    }
    pub(crate) fn reset(&self) -> Result<(), String> {
        if !self.arenas_idle()
            || self.active_leases() != 0
            || !self.pending.lock().is_empty()
            || (self.production_owned && self.production_demand_gate.available_permits() != 1)
        {
            return Err("cannot reset upload evidence with outstanding leases or production demand".into());
        }
        *self.metrics.lock() = Metrics::default();
        *self.nvme_ids.lock() = Sha256::new();
        *self.logical_ids.lock() = Sha256::new();
        *self.generations.lock() = Sha256::new();
        Ok(())
    }
    pub(crate) fn record_nvme(&self, ids: &[u32]) {
        let mut h = self.nvme_ids.lock();
        for id in ids {
            h.update(id.to_le_bytes());
        }
    }
    pub(crate) fn record_logical(&self, ids: &[u32], generations: &[u64], new_ids: &[u32]) {
        let mut h = self.generations.lock();
        for (&id, &generation) in ids.iter().zip(generations) {
            h.update(id.to_le_bytes());
            h.update(generation.to_le_bytes());
        }
        let mut a = self.logical_ids.lock();
        for id in new_ids {
            a.update(id.to_le_bytes());
        }
        self.add(|m| &mut m.logical_admissions, new_ids.len() as u64);
        self.add(|m| &mut m.logical_generation_observations, ids.len() as u64);
    }
    fn acquire(self: &Arc<Self>, width: usize) -> Result<SourceReservation, String> {
        self.add(|m| &mut m.acquisition_attempts, width as u64);
        let ring = self
            .ring
            .as_ref()
            .ok_or("control cannot acquire an upload arena")?;
        // Reserve under arena locks before taking metrics, preserving lock order.
        let mut reservation_metrics = Metrics::default();
        let reserved = reserve_slots(
            ring.arenas.iter().map(|a| &a.state),
            width,
            &mut reservation_metrics,
        );
        {
            let mut m = self.metrics.lock();
            m.add(
                |m| &mut m.arena_generation_errors,
                reservation_metrics.arena_generation_errors,
            );
            m.add(
                |m| &mut m.accounting_errors,
                reservation_metrics.accounting_errors,
            );
            m.add(
                |m| &mut m.leases_created,
                reservation_metrics.leases_created,
            );
            m.high_water = m.high_water.max(reservation_metrics.high_water);
            m.arena_high_water = m.arena_high_water.max(reservation_metrics.arena_high_water);
        }
        let reservation = SourceReservation {
            state: self.clone(),
            slots: reserved?,
            armed: true,
        };
        for r in &reservation.slots {
            if r.new_generation {
                self.map_generation(r)?;
            }
        }
        Ok(reservation)
    }
    fn map_generation(&self, r: &SlotReservation) -> Result<(), String> {
        let ring = self.ring.as_ref().expect("treatment ring");
        let arena = &ring.arenas[r.index];
        let remap = arena.state.lock().ever_mapped;
        let gpu = ring
            .executor
            .authoritative_gpu()
            .map_err(|e| e.to_string())?;
        self.add(|m| &mut m.map_attempts, 1);
        self.add(|m| &mut m.arena_map_attempts, 1);
        if remap {
            self.add(|m| &mut m.remap_attempts, 1);
            self.add(|m| &mut m.arena_remap_attempts, 1);
        }
        let start = Instant::now();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        arena.state.lock().map_pending = true;
        arena
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Write, move |r| {
                let _ = tx.send(r);
            });
        gpu.device.poll(wgpu::Maintain::Wait);
        let result = rx
            .recv()
            .map_err(|e| e.to_string())
            .and_then(|r| r.map_err(|e| e.to_string()));
        let wait_us = elapsed(start);
        self.add(|m| &mut m.map_wait_us, wait_us);
        self.add(|m| &mut m.arena_map_wait_us, wait_us);
        if remap {
            self.add(|m| &mut m.remap_wait_us, wait_us);
        }
        if let Err(error) = result {
            if remap {
                self.add(|m| &mut m.remap_failures, 1);
                self.add(|m| &mut m.arena_remap_failures, 1);
            }
            self.add(|m| &mut m.copy_failures, 1);
            return Err(error);
        }
        {
            let mut state = arena.state.lock();
            state.map_pending = false;
            state.mapping_live = true;
            state.ever_mapped = true;
        }
        self.add(|m| &mut m.map_completions, 1);
        self.add(|m| &mut m.arena_map_completions, 1);
        if remap {
            self.add(|m| &mut m.remap_completions, 1);
            self.add(|m| &mut m.arena_remap_completions, 1);
        }
        let view = arena.buffer.slice(..).get_mapped_range_mut();
        let base = aligned_offset(view.as_ptr() as usize, view.len());
        drop(view);
        let base = base.map_err(|e| {
            self.add(|m| &mut m.alignment_failures, 1);
            e
        })?;
        arena
            .state
            .lock()
            .mapped(r.generation, base, &mut self.metrics.lock())?;
        Ok(())
    }
    pub(crate) async fn read_source(
        self: &Arc<Self>,
        storage: &NvmeStorage,
        ids: &[u32],
        buffers: Vec<PooledBuffer>,
        logical: &GpuExpertCache,
    ) -> Result<Vec<Arc<ExpertResident>>, String> {
        if self.arm != Arm::Treatment
            || ids.is_empty()
            || ids.len() != buffers.len()
            || ids.len() > CAPACITY
        {
            return Err("invalid qualification source set".into());
        }
        if ids
            .iter()
            .enumerate()
            .any(|(i, id)| ids[..i].contains(id) || self.pending.lock().contains_key(id))
        {
            self.add(|m| &mut m.accounting_errors, 1);
            return Err("second source read while an upload lease exists".into());
        }
        // A pre-existing ordinary Vec admission cannot share its allocation.
        // Reject before I/O rather than silently adding treatment-only copies.
        for id in ids {
            if logical
                .current_admission(*id)
                .is_some_and(|a| a.resident().qualification_shared_payload().is_none())
            {
                self.add(|m| &mut m.non_shared_logical_rejections, 1);
                return Err("non-shared logical admission prevents source fusion".into());
            }
        }
        // Each original source call owns one transaction and one storage batch.
        // The reservation is declared before every mapped borrow for safe Drop.
        let reservation = self.acquire(ids.len())?;
        let mut views = reservation
            .slots
            .iter()
            .map(|r| reservation.arena(r).buffer.slice(..).get_mapped_range_mut())
            .collect::<Vec<_>>();
        let bases = reservation
            .slots
            .iter()
            .map(|r| {
                reservation
                    .arena(r)
                    .state
                    .lock()
                    .aligned_base
                    .ok_or("missing mapped alignment")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut destinations = Vec::with_capacity(ids.len());
        for ((view, &base), r) in views.iter_mut().zip(&bases).zip(&reservation.slots) {
            destinations.extend(source_destinations(view, base, &r.slots)?);
        }
        let started = Instant::now();
        let result = storage
            .read_experts_batch_into_aligned_slices(ids, &mut destinations)
            .await;
        self.add(|m| &mut m.fused_source_us, elapsed(started));
        drop(destinations);
        let bytes = result.map_err(|e| {
            self.add(|m| &mut m.source_failures, 1);
            self.add(|m| &mut m.mapped_direct_io_rejections, 1);
            e.to_string()
        })?;
        if bytes != ids.len() * FULL {
            self.add(|m| &mut m.accounting_errors, 1);
            return Err("short direct source set".into());
        }
        self.add(|m| &mut m.direct_source_reads, ids.len() as u64);
        self.add(|m| &mut m.direct_source_bytes, bytes as u64);
        self.add(|m| &mut m.odirect_observations, ids.len() as u64);
        self.add(
            |m| &mut m.direct_payload_bytes,
            (ids.len() * PAYLOAD) as u64,
        );
        let mut shared = Vec::with_capacity(ids.len());
        for ((view, &base), r) in views.iter().zip(&bases).zip(&reservation.slots) {
            for &slot in &r.slots {
                let offset = slot_offset(base, slot, view.len())?;
                let payload = checked_payload(&view[offset..offset + FULL])?;
                shared.push(self.materialize_source_payload(
                    ids[shared.len()],
                    payload,
                    logical,
                )?);
            }
        }
        // Reservation pins outlive all mapped borrows. Successful source calls
        // leave their generation mapped; only physical lease consumption seals.
        drop(views);
        let residents = ids
            .iter()
            .zip(&shared)
            .zip(buffers)
            .map(|((&id, payload), capacity_lease)| {
                Arc::new(ExpertResident::new_qualification_shared(
                    id,
                    capacity_lease,
                    payload.clone(),
                ))
            })
            .collect();
        let mut pending = self.pending.lock();
        if ids.iter().any(|id| pending.contains_key(id)) {
            return Err(generation_error(&mut self.metrics.lock()));
        }
        let leases = reservation.commit(ids, &shared)?;
        for lease in leases {
            pending.insert(lease.id, lease);
        }
        Ok(residents)
    }
    fn materialize_source_payload(
        &self,
        id: u32,
        payload: &[u8],
        logical: &GpuExpertCache,
    ) -> Result<Arc<[u8]>, String> {
        let bytes = if let Some(admission) = logical.current_admission(id) {
            let shared = admission
                .resident()
                .qualification_shared_payload()
                .ok_or("logical backing changed during source read")?;
            if shared.as_ref() != payload {
                return Err("immutable source differs from shared logical payload".into());
            }
            self.add(|m| &mut m.shared_payload_reuse, 1);
            shared.clone()
        } else {
            let start = Instant::now();
            let shared: Arc<[u8]> = Arc::from(payload);
            self.add(|m| &mut m.logical_materialization_us, elapsed(start));
            self.add(|m| &mut m.logical_materialization_operations, 1);
            self.add(
                |m| &mut m.logical_materialization_bytes,
                payload.len() as u64,
            );
            self.add(|m| &mut m.shared_payload_constructions, 1);
            shared
        };
        Ok(bytes)
    }

    pub(crate) fn take_lease(
        &self,
        id: u32,
        resident: &ExpertResident,
    ) -> Result<Option<Lease>, String> {
        let lease = self.pending.lock().remove(&id);
        if let Some(lease) = &lease {
            if self.arm != Arm::Treatment
                || !std::ptr::eq(self, lease.state.as_ref())
                || lease.id != resident.id
                || !resident
                    .qualification_shared_payload()
                    .zip(lease.payload.as_ref())
                    .is_some_and(|(a, b)| Arc::ptr_eq(a, b))
            {
                self.add(|m| &mut m.accounting_errors, 1);
                return Err("source/upload resident identity mismatch".into());
            }
            lease.seal()?;
            if !lease
                .arena()
                .state
                .lock()
                .lease_is(lease.generation, lease.slot, LeasePhase::Ready)
            {
                return Err(generation_error(&mut self.metrics.lock()));
            }
        }
        Ok(lease)
    }
    pub(crate) fn has_pending(&self, id: u32) -> bool {
        self.pending.lock().contains_key(&id)
    }
    pub(crate) fn finish_request(&self) -> Result<(), String> {
        if !self.arenas_idle() || !self.pending.lock().is_empty() || self.active_leases() != 0 {
            self.add(|m| &mut m.accounting_errors, 1);
            self.abandon_pending();
            return Err("unconsumed upload lease at demand completion".into());
        }
        Ok(())
    }
    pub(crate) fn abandon_pending(&self) {
        // Drain before dropping: each Lease seals under its own arena lock,
        // then releases normally. Normal open-mapping abandonment is not an
        // uncertain GPU submission; encoded generations still poison in Drop.
        let pending = std::mem::take(&mut *self.pending.lock());
        drop(pending);
    }
}

pub(crate) struct Lease {
    state: Arc<State>,
    index: usize,
    generation: u64,
    slot: usize,
    id: u32,
    offset: usize,
    payload: Option<Arc<[u8]>>,
}

/// One encoder for the physical install set. Concurrent stage jobs append
/// their exact copies under this short lock. No mapping is published until
/// submit returns; upload leases remain owned here through that submission.
pub(crate) struct CopySet<'a> {
    state: &'a State,
    commands: Mutex<CopyCommands>,
    leases: Mutex<Vec<Lease>>,
}
#[derive(Default)]
struct CopyCommands {
    encoder: Option<wgpu::CommandEncoder>,
    closed: bool,
    submission_started: bool,
}
impl<'a> CopySet<'a> {
    pub(crate) fn new(state: &'a State) -> Self {
        Self {
            state,
            commands: Mutex::new(CopyCommands::default()),
            leases: Mutex::new(Vec::new()),
        }
    }
    pub(crate) fn encode(
        &self,
        lease: Lease,
        device: &wgpu::Device,
        destination: &wgpu::Buffer,
        offset: u64,
    ) -> Result<(), String> {
        let start = Instant::now();
        let source = lease.copy_source_offset()?;
        if offset % 4 != 0
            || offset
                .checked_add(PAYLOAD as u64)
                .is_none_or(|end| end > destination.size())
        {
            return Err("invalid physical upload destination".into());
        }
        let mut commands = self.commands.lock();
        if commands.closed || !std::ptr::eq(self.state, lease.state.as_ref()) {
            return Err(generation_error(&mut self.state.metrics.lock()));
        }
        let encoder = commands.encoder.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qualification residency copy set"),
            })
        });
        lease.transition(LeasePhase::Ready, LeasePhase::Encoded)?;
        encoder.copy_buffer_to_buffer(lease.buffer(), source, destination, offset, PAYLOAD as u64);
        self.leases.lock().push(lease);
        self.state.add(|m| &mut m.copy_encode_us, elapsed(start));
        Ok(())
    }
    pub(crate) fn submit(&self, executor: &GpuNativeExecutorContext) -> Result<(), String> {
        let mut commands = self.commands.lock();
        if commands.closed {
            return Err(generation_error(&mut self.state.metrics.lock()));
        }
        commands.closed = true;
        let Some(encoder) = commands.encoder.take() else {
            return Ok(());
        };
        let gpu = executor.authoritative_gpu().map_err(|e| e.to_string())?;
        let mut leases = self.leases.lock();
        if self.state.arm != Arm::Treatment || leases.is_empty() {
            return Err("invalid qualification copy submission".into());
        }
        let command = encoder.finish();
        self.state.add(|m| &mut m.copy_command_buffers, 1);
        let start = Instant::now();
        // If queue submission unwinds, encoded leases poison their arena.
        // No retry or potentially unsafe reuse is permitted for that generation.
        commands.submission_started = true;
        gpu.queue.submit(Some(command));
        self.state.add(|m| &mut m.copy_submit_us, elapsed(start));
        self.state.add(|m| &mut m.copy_submissions, 1);
        self.state
            .add(|m| &mut m.copied_experts, leases.len() as u64);
        self.state
            .add(|m| &mut m.copied_bytes, (leases.len() * PAYLOAD) as u64);
        for lease in leases.iter_mut() {
            lease.submitted()?;
        }
        // map_async on the next acquisition waits for this buffer's submitted
        // copy to finish; no stable virtual address is assumed on remapping.
        leases.clear();
        executor.authoritative_gpu().map_err(|e| e.to_string())?;
        Ok(())
    }
}
impl Drop for CopySet<'_> {
    fn drop(&mut self) {
        let commands = self.commands.get_mut();
        // Destroy every unsubmitted command reference BEFORE cancelling leases.
        drop(commands.encoder.take());
        let leases = self.leases.get_mut();
        if !commands.submission_started {
            for lease in leases.iter() {
                let _ = lease.transition(LeasePhase::Encoded, LeasePhase::Ready);
            }
        }
        // Submitted leases release normally. Encoded leases after an uncertain
        // submission fail closed in Drop and leave their arena poisoned.
        leases.clear();
    }
}
impl Lease {
    fn seal(&self) -> Result<(), String> {
        let arena = self.arena();
        arena
            .state
            .lock()
            .seal(self.generation, &mut self.state.metrics.lock(), || {
                arena.buffer.unmap()
            })
    }
    fn arena(&self) -> &Arena {
        &self.state.ring.as_ref().expect("treatment ring").arenas[self.index]
    }
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.arena().buffer
    }
    pub(crate) fn copy_source_offset(&self) -> Result<u64, String> {
        let arena = self.arena().state.lock();
        if !arena.lease_is(self.generation, self.slot, LeasePhase::Ready) {
            return Err(generation_error(&mut self.state.metrics.lock()));
        }
        let base = arena
            .aligned_base
            .ok_or_else(|| generation_error(&mut self.state.metrics.lock()))?;
        if slot_offset(base, self.slot, UPLOAD_BYTES)? != self.offset {
            return Err(generation_error(&mut self.state.metrics.lock()));
        }
        copy_source_offset(self.offset, UPLOAD_BYTES)
    }
    pub(crate) fn context_id(&self) -> u64 {
        self.state.ring.as_ref().unwrap().executor.context_id()
    }
    fn transition(&self, expected: LeasePhase, next: LeasePhase) -> Result<(), String> {
        self.arena().state.lock().lease_transition(
            self.generation,
            self.slot,
            expected,
            next,
            &mut self.state.metrics.lock(),
        )
    }
    fn submitted(&mut self) -> Result<(), String> {
        self.transition(LeasePhase::Encoded, LeasePhase::Submitted)?;
        self.state.add(|m| &mut m.leases_consumed, 1);
        Ok(())
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        if self.seal().is_err() {
            return;
        }
        let _ = self.arena().state.lock().release(
            self.generation,
            self.slot,
            &mut self.state.metrics.lock(),
        );
    }
}
pub(crate) fn elapsed(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}
pub(crate) fn aligned_offset(base: usize, capacity: usize) -> Result<usize, String> {
    let offset = (ALIGN - base % ALIGN) % ALIGN;
    if offset % 4 != 0
        || offset
            .checked_add(ARENA_WIDTH * FULL)
            .is_none_or(|end| end > capacity)
        || base.checked_add(offset).is_none_or(|p| p % ALIGN != 0)
    {
        return Err("invalid mapped alignment/range".into());
    }
    Ok(offset)
}
pub(crate) fn copy_source_offset(offset: usize, capacity: usize) -> Result<u64, String> {
    if offset / FULL >= ARENA_WIDTH
        || offset % FULL >= ALIGN
        || offset % 4 != 0
        || offset.checked_add(FULL).is_none_or(|end| end > capacity)
    {
        return Err("invalid upload copy range".into());
    }
    Ok((offset + ALIGN) as u64)
}
fn slot_offset(aligned_base: usize, slot: usize, capacity: usize) -> Result<usize, String> {
    if aligned_base >= ALIGN || aligned_base % 4 != 0 || slot >= ARENA_WIDTH {
        return Err("invalid upload arena slot".into());
    }
    let offset = slot
        .checked_mul(FULL)
        .and_then(|n| aligned_base.checked_add(n))
        .ok_or("upload arena slot overflow")?;
    copy_source_offset(offset, capacity)?;
    Ok(offset)
}
fn checked_payload(source: &[u8]) -> Result<&[u8], String> {
    let (header, payload) = TensorHeader::strip(source, ALIGN);
    let h = header.ok_or("missing UTH1")?;
    if source.len() != FULL
        || source.len() - payload.len() != ALIGN
        || payload.len() != PAYLOAD
        || h.dtype != UthDtypeId::Q4_0
        || h.shape_rank != 3
        || h.shape != [768, 2048, 3, 0]
        || h.quant_scale_count != 0
        || h.quant_scale_offset != 0
    {
        return Err("source/upload requires exact full-file Q4 geometry".into());
    }
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_upload_fresh_source_materializes_logical_host_payload_exactly_once() {
        let state = State::cpu_test_state(Arm::Treatment);
        let cache = GpuExpertCache::new(64, 0.0, 0);
        let payload = [4u8; 16];
        let shared = state
            .materialize_source_payload(1, &payload, &cache)
            .unwrap();
        assert_eq!(shared.as_ref(), payload);
        assert_ne!(shared.as_ptr(), payload.as_ptr());
        assert_eq!(
            state.snapshot().metrics.logical_materialization_operations,
            1
        );
        assert_eq!(state.snapshot().metrics.logical_materialization_bytes, 16);
        let pool = crate::buffer_pool::BufferPool::new(1, 4096, 4096);
        let host = ExpertResident::new_qualification_shared(
            1,
            pool.try_acquire().unwrap(),
            shared.clone(),
        );
        let logical = crate::expert_cache::GpuResident::new_qualification_shared(
            1,
            shared,
            crate::inference::WeightDtype::Q4_0,
        );
        assert!(Arc::ptr_eq(
            host.qualification_shared_payload().unwrap(),
            logical.qualification_shared_payload().unwrap()
        ));
    }
    #[test]
    fn source_upload_existing_shared_logical_reuses_allocation_without_materialization() {
        let state = State::cpu_test_state(Arm::Treatment);
        let cache = GpuExpertCache::new(64, 0.0, 0);
        let shared: Arc<[u8]> = Arc::from([3u8; 16].as_slice());
        let resident = Arc::new(crate::expert_cache::GpuResident::new_qualification_shared(
            1,
            shared.clone(),
            crate::inference::WeightDtype::Q4_0,
        ));
        let payloads = HashMap::from([(1, resident)]);
        cache.demand_admit_set(&[1], &payloads).unwrap();
        let generation = cache.current_admission(1).unwrap().generation();
        let reused = state
            .materialize_source_payload(1, &[3; 16], &cache)
            .unwrap();
        assert!(Arc::ptr_eq(&shared, &reused));
        assert_eq!(
            state.snapshot().metrics.logical_materialization_operations,
            0
        );
        assert_eq!(state.snapshot().metrics.shared_payload_reuse, 1);
        assert_eq!(cache.current_admission(1).unwrap().generation(), generation);
        assert!(state
            .materialize_source_payload(1, &[4; 16], &cache)
            .is_err());
    }
    #[test]
    fn source_upload_ordinary_logical_backing_fails_closed_without_extra_copy() {
        let state = State::cpu_test_state(Arm::Treatment);
        let cache = GpuExpertCache::new(64, 0.0, 0);
        let resident = Arc::new(crate::expert_cache::GpuResident::new(1, vec![3u8; 16]));
        cache
            .demand_admit_set(&[1], &HashMap::from([(1, resident)]))
            .unwrap();
        assert!(state
            .materialize_source_payload(1, &[3; 16], &cache)
            .is_err());
        assert_eq!(
            state.snapshot().metrics.logical_materialization_operations,
            0
        );
    }
    #[test]
    fn source_upload_full_source_header_and_payload_geometry_are_exact() {
        let mut source = Vec::new();
        TensorHeader::for_swiglu_expert(crate::inference::WeightDtype::Q4_0, 2048, 768)
            .write_padded(ALIGN, &mut source);
        source.resize(FULL, 0x37);
        let payload = checked_payload(&source).unwrap();
        assert_eq!(payload.len(), PAYLOAD);
        assert!(payload.iter().all(|b| *b == 0x37));
        assert!(checked_payload(payload).is_err());
        assert!(checked_payload(&source[..FULL - 1]).is_err());
        source[0] ^= 1;
        assert!(checked_payload(&source).is_err());
    }
    #[test]
    fn source_upload_alignment_is_recomputed_for_every_mapping() {
        for base in (0x1000..0x4000).step_by(8) {
            let offset = aligned_offset(base, UPLOAD_BYTES).unwrap();
            assert_eq!((base + offset) % ALIGN, 0);
            assert_eq!(
                copy_source_offset(offset, UPLOAD_BYTES).unwrap(),
                (offset + ALIGN) as u64
            );
            assert_eq!(offset + ALIGN + PAYLOAD, offset + FULL);
        }
        assert_ne!(
            aligned_offset(0x1000, UPLOAD_BYTES).unwrap(),
            aligned_offset(0x2008, UPLOAD_BYTES).unwrap()
        );
    }
    #[test]
    fn source_upload_copy_range_rejects_overflow_misalignment_and_short_buffer() {
        for (offset, capacity) in [
            (1, UPLOAD_BYTES),
            (4096, UPLOAD_BYTES),
            (8, FULL),
            (usize::MAX, usize::MAX),
        ] {
            assert!(copy_source_offset(offset, capacity).is_err());
        }
        assert!(aligned_offset(usize::MAX - 4, UPLOAD_BYTES).is_err());
        assert!(aligned_offset(0x1001, UPLOAD_BYTES).is_err());
        assert!(aligned_offset(0x1000, FULL - 1).is_err());
    }
    #[test]
    fn source_upload_hma1b_geometry_alignment_and_all_copy_ranges() {
        assert_eq!((ARENAS, ARENA_WIDTH, CAPACITY), (2, 8, 16));
        assert_eq!(FULL, 649 * ALIGN);
        assert_eq!(UPLOAD_BYTES, 8 * FULL + ALIGN);
        for base in (0x1000..0x4000).step_by(8) {
            let aligned = aligned_offset(base, UPLOAD_BYTES).unwrap();
            let mut previous_end = 0;
            for slot in 0..ARENA_WIDTH {
                let start = slot_offset(aligned, slot, UPLOAD_BYTES).unwrap();
                let copy = copy_source_offset(start, UPLOAD_BYTES).unwrap() as usize;
                assert_eq!((base + start) % ALIGN, 0);
                assert_eq!(start, aligned + slot * FULL);
                assert_eq!(copy - start, ALIGN);
                assert_eq!(start + FULL - copy, PAYLOAD);
                assert!(start >= previous_end);
                assert!(copy + PAYLOAD <= UPLOAD_BYTES);
                previous_end = start + FULL;
            }
        }
        assert!(slot_offset(0, 8, UPLOAD_BYTES).is_err());
        assert!(slot_offset(4096, 0, UPLOAD_BYTES).is_err());
        assert!(slot_offset(8, 7, 8 * FULL).is_err());
    }
    fn test_arenas() -> [Mutex<ArenaGeneration>; ARENAS] {
        std::array::from_fn(|_| Mutex::new(ArenaGeneration::default()))
    }
    fn source_slots(
        arenas: &[Mutex<ArenaGeneration>],
        width: usize,
        m: &mut Metrics,
    ) -> Vec<SlotReservation> {
        let slots = reserve_slots(arenas.iter(), width, m).unwrap();
        for r in &slots {
            let mut a = arenas[r.index].lock();
            if r.new_generation {
                // Portable stand-in for the first mapping's address; all actual
                // state transitions/allocation/slicing use the production code.
                a.mapped(r.generation, ALIGN - 8, m).unwrap();
            }
            a.populate(r, m).unwrap();
        }
        slots
    }
    fn ready_generation(width: usize, m: &mut Metrics) -> (ArenaGeneration, u64) {
        let arenas = test_arenas();
        let slots = source_slots(&arenas, width, m);
        let r = &slots[0];
        let mut a = arenas[0].lock();
        a.seal(r.generation, m, || {}).unwrap();
        (std::mem::take(&mut *a), r.generation)
    }
    #[test]
    fn source_upload_hma1b1_sixteen_singletons_before_consumption_then_true_capacity() {
        let arenas = test_arenas();
        let mut m = Metrics::default();
        for n in 1..=16 {
            source_slots(&arenas, 1, &mut m);
            assert_eq!(
                arenas
                    .iter()
                    .map(|a| a.lock().active_leases())
                    .sum::<usize>(),
                n
            );
            assert_eq!(
                arenas
                    .iter()
                    .map(|a| a
                        .lock()
                        .leases
                        .iter()
                        .filter(|p| **p == LeasePhase::Pending)
                        .count())
                    .sum::<usize>(),
                n
            );
        }
        assert_eq!((m.high_water, m.arena_high_water), (16, 2));
        assert_eq!(m.leases_created, 16);
        assert!(reserve_slots(arenas.iter(), 1, &mut m)
            .unwrap_err()
            .contains("capacity"));
        assert_eq!(m.arena_generation_errors, 0);
        assert_eq!(m.leases_created, 16);
        for a in &arenas {
            let a = a.lock();
            assert_eq!(a.generation, 1);
            assert_eq!(a.phase, ArenaPhase::OpenMapped);
        }
    }
    #[test]
    fn source_upload_hma1b1_width_sequences_do_not_fragment() {
        for widths in [&[5, 5, 6][..], &[8, 8], &[3, 7, 2, 4], &[16]] {
            let arenas = test_arenas();
            let mut m = Metrics::default();
            for &width in widths {
                source_slots(&arenas, width, &mut m);
            }
            assert_eq!(m.high_water, 16, "{widths:?}");
            assert_eq!(m.arena_high_water, 2);
            assert!(reserve_slots(arenas.iter(), 1, &mut m).is_err());
            assert_eq!(m.arena_generation_errors, 0);
        }
    }
    #[test]
    fn source_upload_hma1b1_every_composition_of_sixteen_fits() {
        // Every ordered partition of 16: includes singleton, mixed-width and
        // single wide-call cases rather than only the motivating examples.
        for cuts in 0u32..(1 << 15) {
            let arenas = test_arenas();
            let mut m = Metrics::default();
            let mut width = 1;
            for position in 0..16 {
                if position == 15 || cuts & (1 << position) != 0 {
                    source_slots(&arenas, width, &mut m);
                    width = 1;
                } else {
                    width += 1;
                }
            }
            assert_eq!(m.high_water, 16);
            assert_eq!(m.arena_generation_errors, 0);
        }
    }
    #[test]
    fn source_upload_hma1b1_split_source_keeps_ids_destinations_and_one_batch() {
        let arenas = test_arenas();
        let mut m = Metrics::default();
        source_slots(&arenas, 5, &mut m);
        source_slots(&arenas, 5, &mut m);
        let reserved = reserve_slots(arenas.iter(), 6, &mut m).unwrap();
        assert_eq!(
            reserved
                .iter()
                .map(|r| (r.index, r.slots.clone()))
                .collect::<Vec<_>>(),
            vec![(0, vec![5, 6, 7]), (1, vec![5, 6, 7])]
        );
        let mut views: [Vec<u8>; ARENAS] = std::array::from_fn(|_| vec![0x55; UPLOAD_BYTES]);
        let bases = views
            .iter()
            .map(|v| aligned_offset(v.as_ptr() as usize, v.len()).unwrap())
            .collect::<Vec<_>>();
        let ids = [91u32, 3, 44, 18, 7, 62];
        let mut destinations = Vec::new();
        for ((view, &base), r) in views.iter_mut().zip(&bases).zip(&reserved) {
            destinations.extend(source_destinations(view, base, &r.slots).unwrap());
        }
        let mut batch_calls = 0;
        let mut batch = |ordered_ids: &[u32], dst: &mut [&mut [u8]]| {
            batch_calls += 1;
            assert_eq!(ordered_ids, ids);
            assert_eq!(dst.len(), ordered_ids.len());
            for (&id, slot) in ordered_ids.iter().zip(dst) {
                assert_eq!(slot.len(), FULL);
                assert_eq!(slot.as_ptr() as usize % ALIGN, 0);
                slot[..4].copy_from_slice(&id.to_le_bytes());
                slot[FULL - 1] = id as u8;
            }
        };
        batch(&ids, &mut destinations);
        assert_eq!(batch_calls, 1);
        drop(destinations);
        for (i, &id) in ids.iter().enumerate() {
            let arena = i / 3;
            let offset = slot_offset(bases[arena], 5 + i % 3, UPLOAD_BYTES).unwrap();
            assert_eq!(&views[arena][offset..offset + 4], &id.to_le_bytes());
            assert_eq!(views[arena][offset + FULL - 1], id as u8);
        }
        for (view, &base) in views.iter().zip(&bases) {
            assert!(view[base..base + 5 * FULL].iter().all(|b| *b == 0x55));
        }
    }
    #[test]
    fn source_upload_hma1b1_append_preserves_one_base_and_disjoint_full_ranges() {
        let arenas = test_arenas();
        let mut m = Metrics::default();
        let mut view = vec![0u8; UPLOAD_BYTES];
        let base = aligned_offset(view.as_ptr() as usize, view.len()).unwrap();
        let mut offsets = Vec::new();
        for id in 1u8..=8 {
            let r = reserve_slots(arenas.iter(), 1, &mut m).unwrap().remove(0);
            let mut a = arenas[0].lock();
            if r.new_generation {
                a.mapped(r.generation, base, &mut m).unwrap();
            }
            assert_eq!(a.aligned_base, Some(base));
            let offset = slot_offset(a.aligned_base.unwrap(), r.slots[0], UPLOAD_BYTES).unwrap();
            offsets.push(offset);
            let mut dst =
                source_destinations(&mut view, a.aligned_base.unwrap(), &r.slots).unwrap();
            dst[0].fill(id);
            drop(dst);
            a.populate(&r, &mut m).unwrap();
        }
        assert!(offsets.windows(2).all(|w| w[0] + FULL == w[1]));
        for (i, offset) in offsets.into_iter().enumerate() {
            assert!(view[offset..offset + FULL]
                .iter()
                .all(|b| *b == (i + 1) as u8));
        }
    }
    #[test]
    fn source_upload_hma1b1_concurrent_first_consumers_seal_once_and_forbid_append() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let arenas = test_arenas();
        let mut m = Metrics::default();
        let first = source_slots(&arenas, 1, &mut m);
        source_slots(&arenas, 1, &mut m);
        let gen = first[0].generation;
        let unmaps = AtomicUsize::new(0);
        let metrics = Mutex::new(m);
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            for slot in 0..2 {
                let (arenas, metrics, barrier, unmaps) = (&arenas, &metrics, &barrier, &unmaps);
                scope.spawn(move || {
                    barrier.wait();
                    let mut a = arenas[0].lock();
                    a.seal(gen, &mut metrics.lock(), || {
                        unmaps.fetch_add(1, Ordering::SeqCst);
                    })
                    .unwrap();
                    a.lease_transition(
                        gen,
                        slot,
                        LeasePhase::Ready,
                        LeasePhase::Encoded,
                        &mut metrics.lock(),
                    )
                    .unwrap();
                });
            }
        });
        assert_eq!(unmaps.load(Ordering::SeqCst), 1);
        assert_eq!(metrics.lock().arena_unmaps, 1);
        assert!(arenas[0].lock().free_slots().is_empty());
        assert_eq!(
            reserve_slots(arenas.iter(), 1, &mut metrics.lock()).unwrap()[0].index,
            1
        );
        assert!(reserve_slots(arenas.iter().take(1), 1, &mut metrics.lock()).is_err());
    }
    #[test]
    fn source_upload_hma1b1_reserved_slots_pin_views_against_seal() {
        let arenas = test_arenas();
        let mut m = Metrics::default();
        let first = source_slots(&arenas, 1, &mut m);
        let append = reserve_slots(arenas.iter(), 2, &mut m).unwrap();
        let mut a = arenas[0].lock();
        assert!(a
            .seal(first[0].generation, &mut m, || panic!(
                "unmap with source views"
            ))
            .is_err());
        a.rollback(&append[0], &mut m, || panic!("prior slots still mapped"))
            .unwrap();
        a.seal(first[0].generation, &mut m, || {}).unwrap();
        assert_eq!(m.unmaps, 1);
    }
    #[test]
    fn source_upload_hma1b1_rollback_preserves_committed_slots_and_reuses_holes() {
        let arenas = test_arenas();
        let mut m = Metrics::default();
        let first = source_slots(&arenas, 3, &mut m);
        let failed = reserve_slots(arenas.iter(), 5, &mut m).unwrap();
        let base = arenas[0].lock().aligned_base;
        arenas[0]
            .lock()
            .rollback(&failed[0], &mut m, || panic!("unmap prior data"))
            .unwrap();
        {
            let a = arenas[0].lock();
            assert_eq!(a.phase, ArenaPhase::OpenMapped);
            assert_eq!(a.aligned_base, base);
            assert_eq!(a.generation, first[0].generation);
            assert_eq!(&a.leases[..3], &[LeasePhase::Pending; 3]);
            assert_eq!(a.active_leases(), 3);
        }
        let retry = source_slots(&arenas, 5, &mut m);
        assert_eq!(retry[0].slots, failed[0].slots);
        assert!(!retry[0].new_generation);
        assert_eq!(m.unmaps, 0);
        assert_eq!(m.arena_generation_errors, 0);
        assert_eq!(m.leases_released, 5);
    }
    #[test]
    fn source_upload_hma1b1_split_rollback_and_mapping_cancellation_are_transactional() {
        for mapped in [false, true] {
            let arenas = test_arenas();
            let mut m = Metrics::default();
            source_slots(&arenas, 5, &mut m);
            let failed = reserve_slots(arenas.iter(), 11, &mut m).unwrap();
            assert_eq!(
                failed.iter().map(|r| r.slots.len()).collect::<Vec<_>>(),
                vec![3, 8]
            );
            let mut unmaps = 0;
            for r in &failed {
                let mut a = arenas[r.index].lock();
                if r.new_generation {
                    a.map_pending = true;
                    if mapped {
                        a.mapped(r.generation, 0, &mut m).unwrap();
                    }
                }
                a.rollback(r, &mut m, || {
                    unmaps += 1;
                })
                .unwrap();
            }
            assert_eq!(unmaps, 1);
            assert_eq!(m.unmaps, u64::from(mapped));
            assert_eq!(arenas[0].lock().active_leases(), 5);
            assert_eq!(arenas[1].lock().phase, ArenaPhase::Available);
            assert_eq!(arenas[1].lock().aligned_base, None);
            source_slots(&arenas, 11, &mut m);
            assert_eq!(m.high_water, 16);
            assert_eq!(m.arena_generation_errors, 0);
        }
    }
    #[test]
    fn source_upload_hma1b1_early_and_alignment_error_mapping_cleanup_is_exact() {
        for completed_map in [false, true] {
            let arenas = test_arenas();
            let mut m = Metrics::default();
            let r = reserve_slots(arenas.iter(), 1, &mut m).unwrap().remove(0);
            let mut a = arenas[0].lock();
            // Failure before map_async, or after callback success but before
            // aligned_offset validation can establish OpenMapped.
            a.mapping_live = completed_map;
            a.ever_mapped = completed_map;
            let mut unmaps = 0;
            a.rollback(&r, &mut m, || {
                unmaps += 1;
            })
            .unwrap();
            assert_eq!(unmaps, usize::from(completed_map));
            assert_eq!(m.unmaps, u64::from(completed_map));
            assert_eq!(a.phase, ArenaPhase::Available);
            assert!(!a.mapping_live);
            assert_eq!(a.active_leases(), 0);
            assert_eq!(m.arena_generation_errors, 0);
        }
    }
    #[test]
    fn source_upload_hma1b1_abandon_open_mapping_unmaps_once_without_poison() {
        let arenas = test_arenas();
        let mut m = Metrics::default();
        let slots = source_slots(&arenas, 3, &mut m);
        let mut a = arenas[0].lock();
        for slot in 0..3 {
            // Exact Lease::drop order used by abandon_pending and both guards.
            a.seal(slots[0].generation, &mut m, || {}).unwrap();
            a.release(slots[0].generation, slot, &mut m).unwrap();
            assert_eq!(a.phase == ArenaPhase::Available, slot == 2);
        }
        assert_eq!(m.unmaps, 1);
        assert_eq!(m.leases_dropped_unconsumed, 3);
        assert_eq!(m.leases_created, m.leases_released);
        assert_eq!(m.arena_generation_errors, 0);
        drop(a);
        let fresh = reserve_slots(arenas.iter(), 1, &mut m).unwrap();
        assert_eq!(fresh[0].generation, 2);
        // New virtual mapping => a new buffer alignment, never old pointer reuse.
        arenas[0].lock().mapped(2, 0, &mut m).unwrap();
        assert_eq!(arenas[0].lock().aligned_base, Some(0));
    }
    #[test]
    fn source_upload_hma1b_complete_generations_submit_release_and_remap() {
        let arenas = test_arenas();
        let mut m = Metrics::default();
        for expected in 1..=3 {
            let slots = source_slots(&arenas, 8, &mut m);
            let gen = slots[0].generation;
            assert_eq!(gen, expected);
            let mut a = arenas[0].lock();
            a.seal(gen, &mut m, || {}).unwrap();
            for slot in 0..8 {
                a.lease_transition(gen, slot, LeasePhase::Ready, LeasePhase::Encoded, &mut m)
                    .unwrap();
                a.lease_transition(
                    gen,
                    slot,
                    LeasePhase::Encoded,
                    LeasePhase::Submitted,
                    &mut m,
                )
                .unwrap();
            }
            assert!(a.free_slots().is_empty());
            for (released, slot) in [3, 7, 0, 5, 1, 6, 4, 2].into_iter().enumerate() {
                a.release(gen, slot, &mut m).unwrap();
                assert_eq!(a.active_leases(), 7 - released);
                assert_eq!(a.phase == ArenaPhase::Available, released == 7);
                if released != 7 {
                    assert!(a.free_slots().is_empty());
                }
            }
        }
        assert_eq!(m.leases_released, 24);
        assert_eq!(m.leases_dropped_unconsumed, 0);
        assert_eq!(m.arena_generation_errors, 0);
    }
    #[test]
    fn source_upload_hma1b_stale_double_release_and_submit_fail_closed() {
        let mut m = Metrics::default();
        let (mut a, old) = ready_generation(1, &mut m);
        a.lease_transition(old, 0, LeasePhase::Ready, LeasePhase::Encoded, &mut m)
            .unwrap();
        a.lease_transition(old, 0, LeasePhase::Encoded, LeasePhase::Submitted, &mut m)
            .unwrap();
        assert!(a
            .lease_transition(old, 0, LeasePhase::Encoded, LeasePhase::Submitted, &mut m)
            .is_err());
        a.release(old, 0, &mut m).unwrap();
        assert!(a.release(old, 0, &mut m).is_err());
        let arenas = [Mutex::new(a)];
        let new = reserve_slots(arenas.iter(), 2, &mut m).unwrap().remove(0);
        let mut a = arenas[0].lock();
        assert_ne!(old, new.generation);
        assert!(a.release(old, 0, &mut m).is_err());
        assert!(a.seal(old, &mut m, || panic!("stale unmap")).is_err());
        assert!(a
            .lease_transition(old, 0, LeasePhase::Ready, LeasePhase::Encoded, &mut m)
            .is_err());
        let stale = SlotReservation {
            generation: old,
            ..new
        };
        assert!(a
            .rollback(&stale, &mut m, || panic!("stale rollback"))
            .is_err());
        assert_eq!(a.active_leases(), 2);
        assert_eq!(a.phase, ArenaPhase::Mapping);
        assert_eq!(m.arena_generation_errors, 6);
        assert_eq!(m.accounting_errors, 6);
    }
    #[test]
    fn source_upload_hma1b_mapping_unwind_and_submission_uncertainty() {
        let mut m = Metrics::default();
        let (mut a, gen) = ready_generation(2, &mut m);
        a.lease_transition(gen, 0, LeasePhase::Ready, LeasePhase::Encoded, &mut m)
            .unwrap();
        // Destroy unsubmitted commands before cancelling encoded leases.
        a.lease_transition(gen, 0, LeasePhase::Encoded, LeasePhase::Ready, &mut m)
            .unwrap();
        a.release(gen, 0, &mut m).unwrap();
        a.lease_transition(gen, 1, LeasePhase::Ready, LeasePhase::Encoded, &mut m)
            .unwrap();
        assert!(a.release(gen, 1, &mut m).is_err());
        assert_eq!(a.phase, ArenaPhase::Poisoned);
        let arenas = [Mutex::new(a)];
        assert!(reserve_slots(arenas.iter(), 1, &mut m).is_err());
        assert!(arenas[0]
            .lock()
            .seal(gen, &mut m, || panic!("poison reuse"))
            .is_err());
        assert_eq!(arenas[0].lock().active_leases(), 1);
    }
    #[test]
    fn source_upload_hma1b_generation_overflow_and_invalid_width_fail_closed() {
        let arenas = test_arenas();
        let mut m = Metrics::default();
        assert!(reserve_slots(arenas.iter(), 0, &mut m).is_err());
        assert!(reserve_slots(arenas.iter(), 17, &mut m).is_err());
        arenas[1].lock().generation = u64::MAX;
        assert!(reserve_slots(arenas.iter(), 16, &mut m).is_err());
        // Preflight failure must not reserve even the valid first arena.
        assert_eq!(arenas[0].lock().generation, 0);
        assert_eq!(arenas[0].lock().phase, ArenaPhase::Available);
        assert_eq!(arenas[0].lock().active_leases(), 0);
        assert_eq!(m.leases_created, 0);
        assert_eq!(m.arena_generation_errors, 3);
    }
    #[test]
    fn source_upload_control_has_no_ring_or_lease_path() {
        let state = State::cpu_test_state(Arm::Control);
        assert!(state.acquire(1).is_err());
        assert_eq!(state.snapshot().ring_capacity, 0);
        assert_eq!(state.snapshot().metrics.leases_created, 0);
        assert_eq!(state.snapshot().metrics.direct_source_reads, 0);
    }

    #[test]
    fn source_upload_production_demand_gate_is_exclusive_and_owned() {
        let state = State::cpu_test_production_state();
        assert!(state.snapshot().production_owned);
        let guard = state
            .try_begin_production_demand()
            .expect("first production demand owns the upload ring");
        assert!(state.try_begin_production_demand().is_none());
        drop(guard);
        assert!(state.try_begin_production_demand().is_some());

        let control = State::cpu_test_state(Arm::Control);
        assert!(!control.snapshot().production_owned);
        assert!(control.try_begin_production_demand().is_none());
    }

    #[test]
    fn source_upload_counter_overflow_is_a_visible_accounting_error() {
        let mut metrics = Metrics::default();
        metrics.direct_source_bytes = u64::MAX;
        metrics.add(|m| &mut m.direct_source_bytes, 1);
        assert_eq!(metrics.accounting_errors, 1);
        assert_eq!(metrics.direct_source_bytes, u64::MAX);
    }
    #[test]
    fn source_upload_nvme_and_generation_hashes_preserve_order() {
        let a = State::cpu_test_state(Arm::Control);
        let b = State::cpu_test_state(Arm::Control);
        a.record_nvme(&[3, 1, 2]);
        b.record_nvme(&[2, 1, 3]);
        assert_ne!(
            a.snapshot().ordered_nvme_ids_sha256,
            b.snapshot().ordered_nvme_ids_sha256
        );
        a.record_logical(&[3, 1], &[9, 10], &[3, 1]);
        b.record_logical(&[3, 1], &[10, 9], &[3, 1]);
        assert_ne!(
            a.snapshot().logical_generation_ids_sha256,
            b.snapshot().logical_generation_ids_sha256
        );
        assert_eq!(
            a.snapshot().logical_admission_ids_sha256,
            b.snapshot().logical_admission_ids_sha256
        );
    }
    #[test]
    fn source_upload_hma1b1_production_sequential_source_fallback_is_frozen() {
        let engine = include_str!("engine.rs");
        let start = engine
            .find("    async fn gpu_native_demand_source(")
            .unwrap();
        let end = engine
            .find("    async fn ensure_gpu_native_logical_demand_set(")
            .unwrap();
        // f6cd2ff5: demand source, dispatcher, complete sequential fallback and
        // production scheduler, including all fallback decisions and commits.
        assert_eq!(
            format!("{:x}", Sha256::digest(&engine.as_bytes()[start..end])),
            "e2ae64bc9a1f3126f631855ac007d037695c904214fc17ee4d4e3624219596a3"
        );
        let source = include_str!("gpu_native_source_upload.rs");
        let abandon = source
            .split("pub(crate) fn abandon_pending(&self)")
            .nth(1)
            .unwrap()
            .split("pub(crate) struct Lease")
            .next()
            .unwrap();
        assert!(abandon.contains("std::mem::take"));
        assert!(abandon.contains("drop(pending)"));
        let lease_drop = source
            .split("impl Drop for Lease")
            .nth(1)
            .unwrap()
            .split("pub(crate) fn elapsed")
            .next()
            .unwrap();
        assert!(lease_drop.find("self.seal()").unwrap() < lease_drop.find(".release(").unwrap());
        let take = source
            .split("pub(crate) fn take_lease(")
            .nth(1)
            .unwrap()
            .split("pub(crate) fn has_pending")
            .next()
            .unwrap();
        assert!(take.contains("Arc::ptr_eq"));
        assert!(take.contains("lease.seal()?"));
    }
    #[test]
    fn source_upload_source_owns_the_only_read_and_drops_views_before_unmap() {
        let source = include_str!("gpu_native_source_upload.rs");
        let body = source
            .split("pub(crate) async fn read_source(")
            .nth(1)
            .unwrap()
            .split("pub(crate) fn take_lease")
            .next()
            .unwrap();
        assert_eq!(
            body.matches(".read_experts_batch_into_aligned_slices(")
                .count(),
            1
        );
        assert!(
            body.find("self.acquire(ids.len())").unwrap()
                < body
                    .find(".read_experts_batch_into_aligned_slices(")
                    .unwrap()
        );
        assert!(
            body.find("drop(views)").unwrap()
                < body.find("reservation.commit(ids, &shared)").unwrap()
        );
        for forbidden in [
            ".read_expert(",
            ".read_experts_batch(",
            ".to_vec()",
            "vec![0",
            "Vec::<u8>",
            ".unmap()",
            "aligned_offset(",
        ] {
            assert!(!body.contains(forbidden));
        }
        let physical = include_str!("backend/gpu_native.rs")
            .split("pub(crate) fn stage_q4_expert_source_upload")
            .nth(1)
            .unwrap()
            .split("pub(crate) fn commit_q4_expert_residency_production")
            .next()
            .unwrap();
        for forbidden in [
            "read_expert",
            "NvmeStorage",
            "write_buffer_with",
            ".to_vec()",
        ] {
            assert!(!physical.contains(forbidden));
        }
        let residency = include_str!("gpu_native_residency.rs");
        let submit = residency.find("copies.submit(&self.executor)").unwrap();
        assert!(
            submit
                < residency[submit..]
                    .find(".commit_q4_expert_residency_production_observed(")
                    .unwrap()
                    + submit
        );
    }
}
