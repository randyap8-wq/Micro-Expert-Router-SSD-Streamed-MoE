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
    Mapped,
    Ready,
    Poisoned,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LeasePhase {
    Ready,
    Encoded,
    Submitted,
    Released,
}

/// A generation owns the entire buffer, including unused slots. A released
/// prefix never makes it available. An uncertain encoded/submission lifetime
/// poisons the arena permanently; stale tokens cannot mutate a new generation.
#[derive(Debug)]
struct ArenaGeneration {
    phase: ArenaPhase,
    generation: u64,
    width: usize,
    leases: [LeasePhase; ARENA_WIDTH],
    ever_mapped: bool,
}
impl Default for ArenaGeneration {
    fn default() -> Self {
        Self {
            phase: ArenaPhase::Available,
            generation: 0,
            width: 0,
            leases: [LeasePhase::Released; ARENA_WIDTH],
            ever_mapped: false,
        }
    }
}
fn generation_error(m: &mut Metrics) -> String {
    m.add(|m| &mut m.arena_generation_errors, 1);
    m.add(|m| &mut m.accounting_errors, 1);
    "invalid upload arena generation/lifecycle".into()
}
impl ArenaGeneration {
    fn reserve(&mut self, width: usize, m: &mut Metrics) -> Result<u64, String> {
        if self.phase != ArenaPhase::Available || width == 0 || width > ARENA_WIDTH {
            return Err(generation_error(m));
        }
        let next = self
            .generation
            .checked_add(1)
            .ok_or_else(|| generation_error(m))?;
        self.generation = next;
        self.width = width;
        self.leases = [LeasePhase::Released; ARENA_WIDTH];
        self.leases[..width].fill(LeasePhase::Ready);
        self.phase = ArenaPhase::Mapping;
        Ok(next)
    }
    fn transition(
        &mut self,
        generation: u64,
        expected: ArenaPhase,
        next: ArenaPhase,
        m: &mut Metrics,
    ) -> Result<(), String> {
        if self.generation != generation || self.phase != expected {
            return Err(generation_error(m));
        }
        self.phase = next;
        Ok(())
    }
    fn lease_is(&self, generation: u64, slot: usize, phase: LeasePhase) -> bool {
        self.generation == generation
            && self.phase == ArenaPhase::Ready
            && slot < self.width
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
        if !self.lease_is(generation, slot, expected) {
            return Err(generation_error(m));
        }
        self.leases[slot] = next;
        Ok(())
    }
    fn release(&mut self, generation: u64, slot: usize, m: &mut Metrics) -> Result<(), String> {
        if self.generation != generation
            || self.phase != ArenaPhase::Ready
            || slot >= self.width
            || self.leases[slot] == LeasePhase::Released
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
        if self.leases[..self.width]
            .iter()
            .all(|p| *p == LeasePhase::Released)
        {
            self.phase = ArenaPhase::Available;
        }
        Ok(())
    }
    fn abort_mapping(&mut self, generation: u64, m: &mut Metrics) -> Result<(), String> {
        if self.generation != generation
            || !matches!(self.phase, ArenaPhase::Mapping | ArenaPhase::Mapped)
        {
            return Err(generation_error(m));
        }
        m.add(|m| &mut m.leases_dropped_unconsumed, self.width as u64);
        m.add(|m| &mut m.leases_released, self.width as u64);
        self.leases.fill(LeasePhase::Released);
        self.phase = ArenaPhase::Available;
        Ok(())
    }
    fn active_leases(&self) -> usize {
        self.leases[..self.width]
            .iter()
            .filter(|p| **p != LeasePhase::Released)
            .count()
    }
}
fn reserve_arena<'a>(
    states: impl Iterator<Item = &'a Mutex<ArenaGeneration>>,
    width: usize,
    m: &mut Metrics,
) -> Result<(usize, u64), String> {
    for (i, state) in states.enumerate() {
        let mut state = state.lock();
        if state.phase == ArenaPhase::Available {
            return state.reserve(width, m).map(|generation| (i, generation));
        }
    }
    Err(generation_error(m))
}
struct Arena {
    buffer: wgpu::Buffer,
    state: Mutex<ArenaGeneration>,
}
struct Ring {
    executor: Arc<GpuNativeExecutorContext>,
    arenas: [Arena; ARENAS],
}

/// Sole mapping owner, declared before every view. Drop therefore cancels only
/// after all mapped borrows are destroyed, including async cancellation/errors.
struct ArenaMapping {
    state: Arc<State>,
    index: usize,
    generation: u64,
    width: usize,
    mapped: bool,
    armed: bool,
}
impl ArenaMapping {
    fn arena(&self) -> &Arena {
        &self.state.ring.as_ref().expect("treatment ring").arenas[self.index]
    }
    fn into_leases(mut self, ids: &[u32], aligned_base: usize) -> Result<Vec<Lease>, String> {
        if ids.len() != self.width {
            return Err(generation_error(&mut self.state.metrics.lock()));
        }
        let offsets = (0..self.width)
            .map(|slot| slot_offset(aligned_base, slot, UPLOAD_BYTES))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                self.state.add(|m| &mut m.alignment_failures, 1);
                e
            })?;
        let mut generation = self.arena().state.lock();
        if generation.generation != self.generation || generation.phase != ArenaPhase::Mapped {
            return Err(generation_error(&mut self.state.metrics.lock()));
        }
        self.arena().buffer.unmap();
        generation.transition(
            self.generation,
            ArenaPhase::Mapped,
            ArenaPhase::Ready,
            &mut self.state.metrics.lock(),
        )?;
        drop(generation);
        self.mapped = false;
        self.state.record_unmap();
        self.armed = false;
        Ok(ids
            .iter()
            .zip(offsets)
            .enumerate()
            .map(|(slot, (&id, offset))| Lease {
                state: self.state.clone(),
                index: self.index,
                generation: self.generation,
                slot,
                id,
                offset,
                payload: None,
            })
            .collect())
    }
}
impl Drop for ArenaMapping {
    fn drop(&mut self) {
        if self.armed {
            let mut generation = self.arena().state.lock();
            if generation.generation != self.generation
                || !matches!(generation.phase, ArenaPhase::Mapping | ArenaPhase::Mapped)
            {
                let _ = generation_error(&mut self.state.metrics.lock());
                return;
            }
            // No logical lease escapes until the successful unmap transition.
            // Mapping failure still cancels map_async, but is not a successful unmap.
            self.arena().buffer.unmap();
            if self.mapped {
                self.state.record_unmap();
            }
            let _ = generation.abort_mapping(self.generation, &mut self.state.metrics.lock());
        }
    }
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
    pub(crate) fn reset(&self) -> Result<(), String> {
        if self.active_leases() != 0
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
    fn record_unmap(&self) {
        let mut m = self.metrics.lock();
        m.add(|m| &mut m.unmaps, 1);
        m.add(|m| &mut m.arena_unmaps, 1);
    }
    fn acquire(self: &Arc<Self>, width: usize) -> Result<ArenaMapping, String> {
        self.add(|m| &mut m.acquisition_attempts, width as u64);
        let ring = self
            .ring
            .as_ref()
            .ok_or("control cannot acquire an upload arena")?;
        // Never hold the metrics lock while acquiring an arena lock. Merge
        // reservation errors after releasing the arena lock.
        let mut reservation_metrics = Metrics::default();
        let reserved = reserve_arena(
            ring.arenas.iter().map(|a| &a.state),
            width,
            &mut reservation_metrics,
        );
        self.add(
            |m| &mut m.arena_generation_errors,
            reservation_metrics.arena_generation_errors,
        );
        self.add(
            |m| &mut m.accounting_errors,
            reservation_metrics.accounting_errors,
        );
        let (index, generation) = reserved?;
        let mut mapping = ArenaMapping {
            state: self.clone(),
            index,
            generation,
            width,
            mapped: false,
            armed: true,
        };
        self.add(|m| &mut m.leases_created, width as u64);
        let active = self.active_leases() as u64;
        let arenas = ring
            .arenas
            .iter()
            .filter(|a| a.state.lock().phase != ArenaPhase::Available)
            .count() as u64;
        {
            let mut m = self.metrics.lock();
            m.high_water = m.high_water.max(active);
            m.arena_high_water = m.arena_high_water.max(arenas);
        }
        let arena = &ring.arenas[index];
        let remap = arena.state.lock().ever_mapped;
        // Resolve the GPU before starting a mapping that could be abandoned.
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
        mapping.mapped = true;
        arena.state.lock().transition(
            generation,
            ArenaPhase::Mapping,
            ArenaPhase::Mapped,
            &mut self.metrics.lock(),
        )?;
        self.add(|m| &mut m.map_completions, 1);
        self.add(|m| &mut m.arena_map_completions, 1);
        if remap {
            self.add(|m| &mut m.remap_completions, 1);
            self.add(|m| &mut m.arena_remap_completions, 1);
        }
        arena.state.lock().ever_mapped = true;
        Ok(mapping)
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
        // Keep original ID order and a single storage batch even for a source
        // set wider than eight: its two arena views supply one destination list.
        let mappings = ids
            .chunks(ARENA_WIDTH)
            .map(|chunk| self.acquire(chunk.len()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut views = mappings
            .iter()
            .map(|a| a.arena().buffer.slice(..).get_mapped_range_mut())
            .collect::<Vec<_>>();
        let offsets = views
            .iter()
            .map(|v| aligned_offset(v.as_ptr() as usize, v.len()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                self.add(|m| &mut m.alignment_failures, 1);
                e
            })?;
        let mut destinations = Vec::with_capacity(ids.len());
        for ((view, &offset), mapping) in views.iter_mut().zip(&offsets).zip(&mappings) {
            destinations.extend(view[offset..offset + mapping.width * FULL].chunks_exact_mut(FULL));
        }
        self.add(|m| &mut m.source_sets_mapped, 1);
        self.add(|m| &mut m.source_slots_mapped, ids.len() as u64);
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
        for (i, &id) in ids.iter().enumerate() {
            let view = &views[i / ARENA_WIDTH];
            let offset = slot_offset(offsets[i / ARENA_WIDTH], i % ARENA_WIDTH, view.len())?;
            let payload = checked_payload(&view[offset..offset + FULL])?;
            let bytes = self.materialize_source_payload(id, payload, logical)?;
            shared.push(bytes);
        }
        // Views/slices precede unmap even on error or future cancellation.
        drop(views);
        let mut leases = Vec::with_capacity(ids.len());
        for ((mapping, chunk), offset) in mappings
            .into_iter()
            .zip(ids.chunks(ARENA_WIDTH))
            .zip(offsets)
        {
            leases.extend(mapping.into_leases(chunk, offset)?);
        }
        let mut residents = Vec::with_capacity(ids.len());
        for ((lease, payload), capacity_lease) in leases.iter_mut().zip(shared).zip(buffers) {
            lease.payload = Some(payload.clone());
            residents.push(Arc::new(ExpertResident::new_qualification_shared(
                lease.id,
                capacity_lease,
                payload,
            )));
        }
        let mut pending = self.pending.lock();
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
                || lease.id != resident.id
                || !resident
                    .qualification_shared_payload()
                    .zip(lease.payload.as_ref())
                    .is_some_and(|(a, b)| Arc::ptr_eq(a, b))
                || !lease.arena().state.lock().lease_is(
                    lease.generation,
                    lease.slot,
                    LeasePhase::Ready,
                )
            {
                self.add(|m| &mut m.accounting_errors, 1);
                return Err("source/upload resident identity or unmap mismatch".into());
            }
        }
        Ok(lease)
    }
    pub(crate) fn has_pending(&self, id: u32) -> bool {
        self.pending.lock().contains_key(&id)
    }
    pub(crate) fn finish_request(&self) -> Result<(), String> {
        if !self.pending.lock().is_empty() || self.active_leases() != 0 {
            self.add(|m| &mut m.accounting_errors, 1);
            self.pending.lock().clear();
            return Err("unconsumed upload lease at demand completion".into());
        }
        Ok(())
    }
    pub(crate) fn abandon_pending(&self) {
        self.pending.lock().clear();
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
        let base = self
            .offset
            .checked_sub(self.slot * FULL)
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
    fn ready_generation(width: usize, m: &mut Metrics) -> (ArenaGeneration, u64) {
        let mut a = ArenaGeneration::default();
        let generation = a.reserve(width, m).unwrap();
        a.transition(generation, ArenaPhase::Mapping, ArenaPhase::Mapped, m)
            .unwrap();
        a.transition(generation, ArenaPhase::Mapped, ArenaPhase::Ready, m)
            .unwrap();
        (a, generation)
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
    #[test]
    fn source_upload_hma1b_complete_generations_submit_release_and_remap() {
        let mut m = Metrics::default();
        let mut a = ArenaGeneration::default();
        for expected in 1..=3 {
            let generation = a.reserve(8, &mut m).unwrap();
            assert_eq!(generation, expected);
            a.transition(generation, ArenaPhase::Mapping, ArenaPhase::Mapped, &mut m)
                .unwrap();
            a.transition(generation, ArenaPhase::Mapped, ArenaPhase::Ready, &mut m)
                .unwrap();
            for slot in 0..8 {
                a.lease_transition(
                    generation,
                    slot,
                    LeasePhase::Ready,
                    LeasePhase::Encoded,
                    &mut m,
                )
                .unwrap();
                a.lease_transition(
                    generation,
                    slot,
                    LeasePhase::Encoded,
                    LeasePhase::Submitted,
                    &mut m,
                )
                .unwrap();
            }
            for (released, slot) in [3, 7, 0, 5, 1, 6, 4, 2].into_iter().enumerate() {
                a.release(generation, slot, &mut m).unwrap();
                assert_eq!(a.active_leases(), 7 - released);
                assert_eq!(a.phase == ArenaPhase::Available, released == 7);
            }
        }
        assert_eq!(m.leases_released, 24);
        assert_eq!(m.leases_dropped_unconsumed, 0);
        assert_eq!(m.arena_generation_errors, 0);
    }
    #[test]
    fn source_upload_hma1b_generation_waits_for_last_partial_or_abandoned_lease() {
        let mut m = Metrics::default();
        let (mut a, generation) = ready_generation(8, &mut m);
        for slot in 0..4 {
            a.lease_transition(
                generation,
                slot,
                LeasePhase::Ready,
                LeasePhase::Encoded,
                &mut m,
            )
            .unwrap();
            a.lease_transition(
                generation,
                slot,
                LeasePhase::Encoded,
                LeasePhase::Submitted,
                &mut m,
            )
            .unwrap();
            a.release(generation, slot, &mut m).unwrap();
            assert_eq!(a.phase, ArenaPhase::Ready);
        }
        // Unconsumed pending leases may be abandoned, but only the final one
        // releases the buffer shared with the already-submitted prefix.
        for slot in 4..8 {
            a.release(generation, slot, &mut m).unwrap();
            assert_eq!(a.phase == ArenaPhase::Available, slot == 7);
        }
        assert_eq!(m.leases_released, 8);
        assert_eq!(m.leases_dropped_unconsumed, 4);
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
        let new = a.reserve(2, &mut m).unwrap();
        assert_ne!(old, new);
        assert!(a.release(old, 0, &mut m).is_err());
        assert!(a.abort_mapping(old, &mut m).is_err());
        assert!(a
            .lease_transition(old, 0, LeasePhase::Ready, LeasePhase::Encoded, &mut m)
            .is_err());
        assert_eq!(a.generation, new);
        assert_eq!(a.phase, ArenaPhase::Mapping);
        assert_eq!(a.active_leases(), 2);
        assert_eq!(m.arena_generation_errors, 5);
        assert_eq!(m.accounting_errors, 5);
    }
    #[test]
    fn source_upload_hma1b_two_arenas_are_independent_and_third_fails() {
        let mut m = Metrics::default();
        let arenas: [_; ARENAS] = std::array::from_fn(|_| Mutex::new(ArenaGeneration::default()));
        let (first, gen) = reserve_arena(arenas.iter(), 8, &mut m).unwrap();
        let mut a = arenas[first].lock();
        a.transition(gen, ArenaPhase::Mapping, ArenaPhase::Mapped, &mut m)
            .unwrap();
        a.transition(gen, ArenaPhase::Mapped, ArenaPhase::Ready, &mut m)
            .unwrap();
        a.lease_transition(gen, 0, LeasePhase::Ready, LeasePhase::Encoded, &mut m)
            .unwrap();
        a.lease_transition(gen, 0, LeasePhase::Encoded, LeasePhase::Submitted, &mut m)
            .unwrap();
        drop(a);
        let (second, other) = reserve_arena(arenas.iter(), 8, &mut m).unwrap();
        assert_ne!(first, second);
        assert!(reserve_arena(arenas.iter(), 1, &mut m).is_err());
        arenas[first].lock().release(gen, 0, &mut m).unwrap();
        assert!(reserve_arena(arenas.iter(), 1, &mut m).is_err());
        arenas[second].lock().abort_mapping(other, &mut m).unwrap();
        assert_eq!(reserve_arena(arenas.iter(), 1, &mut m).unwrap().0, second);
        assert_eq!(m.arena_generation_errors, 2);
    }
    #[test]
    fn source_upload_hma1b_mapping_unwind_and_submission_uncertainty() {
        let mut m = Metrics::default();
        for phase in [ArenaPhase::Mapping, ArenaPhase::Mapped] {
            let mut a = ArenaGeneration::default();
            let gen = a.reserve(3, &mut m).unwrap();
            a.phase = phase;
            assert!(a
                .lease_transition(gen, 0, LeasePhase::Ready, LeasePhase::Encoded, &mut m)
                .is_err());
            a.abort_mapping(gen, &mut m).unwrap();
            assert_eq!(a.phase, ArenaPhase::Available);
            assert_eq!(a.active_leases(), 0);
        }
        let (mut a, gen) = ready_generation(2, &mut m);
        a.lease_transition(gen, 0, LeasePhase::Ready, LeasePhase::Encoded, &mut m)
            .unwrap();
        // Unsubmitted encoder is destroyed; only then can the copy owner cancel.
        a.lease_transition(gen, 0, LeasePhase::Encoded, LeasePhase::Ready, &mut m)
            .unwrap();
        a.release(gen, 0, &mut m).unwrap();
        assert_eq!(a.phase, ArenaPhase::Ready);
        a.lease_transition(gen, 1, LeasePhase::Ready, LeasePhase::Encoded, &mut m)
            .unwrap();
        assert!(a.release(gen, 1, &mut m).is_err());
        assert_eq!(a.phase, ArenaPhase::Poisoned);
        assert!(a.reserve(1, &mut m).is_err());
        assert!(a.abort_mapping(gen, &mut m).is_err());
        assert!(m.arena_generation_errors > 0);
    }
    #[test]
    fn source_upload_hma1b_generation_overflow_and_invalid_width_fail_closed() {
        let mut m = Metrics::default();
        let mut a = ArenaGeneration::default();
        assert!(a.reserve(0, &mut m).is_err());
        assert!(a.reserve(9, &mut m).is_err());
        a.generation = u64::MAX;
        assert!(a.reserve(1, &mut m).is_err());
        assert_eq!(a.phase, ArenaPhase::Available);
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
            body.find("self.acquire(chunk.len())").unwrap()
                < body
                    .find(".read_experts_batch_into_aligned_slices(")
                    .unwrap()
        );
        assert!(
            body.find("drop(views)").unwrap()
                < body.find("mapping.into_leases(chunk, offset)").unwrap()
        );
        for forbidden in [
            ".read_expert(",
            ".read_experts_batch(",
            ".to_vec()",
            "vec![0",
            "Vec::<u8>",
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
