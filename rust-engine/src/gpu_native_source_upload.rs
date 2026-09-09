//! Private real-inference source/upload mechanism. No production constructor
//! creates this state. Source I/O lives here; physical installation receives only
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
pub(crate) const CAPACITY: usize = 16;
pub(crate) const UPLOAD_BYTES: usize = FULL + ALIGN;

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
    pub(crate) fd_proof_cache_hits: u64,
    pub(crate) fd_proof_cache_misses: u64,
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
enum SlotState {
    Available,
    Mapping,
    Mapped,
    Ready,
    Submitted,
}
impl SlotState {
    fn transition(&mut self, expected: Self, next: Self) -> Result<(), String> {
        if *self != expected {
            return Err(format!(
                "upload state mismatch: expected {expected:?}, got {self:?}"
            ));
        }
        *self = next;
        Ok(())
    }
}
fn reserve_slot<'a>(states: impl Iterator<Item = &'a Mutex<SlotState>>) -> Option<usize> {
    states.enumerate().find_map(|(i, s)| {
        s.lock()
            .transition(SlotState::Available, SlotState::Mapping)
            .ok()
            .map(|_| i)
    })
}
struct Slot {
    buffer: wgpu::Buffer,
    state: Mutex<SlotState>,
    ever_mapped: Mutex<bool>,
}
struct Ring {
    executor: Arc<GpuNativeExecutorContext>,
    slots: Vec<Slot>,
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
            let slots = (0..CAPACITY)
                .map(|_| Slot {
                    buffer: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("source-to-upload bounded slot"),
                        size: UPLOAD_BYTES as u64,
                        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                    state: Mutex::new(SlotState::Available),
                    ever_mapped: Mutex::new(false),
                })
                .collect();
            Some(Ring { executor, slots })
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
            arm: self.arm,
            production_owned: self.production_owned,
            ring_capacity: self.ring.as_ref().map_or(0, |r| r.slots.len()),
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
            r.slots
                .iter()
                .filter(|s| *s.state.lock() != SlotState::Available)
                .count()
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
    fn acquire(self: &Arc<Self>, id: u32) -> Result<Lease, String> {
        self.add(|m| &mut m.acquisition_attempts, 1);
        let ring = self
            .ring
            .as_ref()
            .ok_or("control cannot acquire an upload lease")?;
        let index = reserve_slot(ring.slots.iter().map(|s| &s.state)).ok_or_else(|| {
            // The frozen single demand stream cannot release an outstanding
            // source lease while waiting here. Fail closed instead of deadlock
            // or creating a seventeenth buffer.
            self.add(|m| &mut m.accounting_errors, 1);
            "bounded upload ring exhausted".to_string()
        })?;
        self.add(|m| &mut m.leases_created, 1);
        let active = self.active_leases() as u64;
        let mut metrics = self.metrics.lock();
        metrics.high_water = metrics.high_water.max(active);
        drop(metrics);
        let lease = Lease {
            state: self.clone(),
            index,
            id,
            offset: 0,
            payload: None,
            consumed: false,
        };
        let slot = &ring.slots[index];
        let remap = *slot.ever_mapped.lock();
        self.add(|m| &mut m.map_attempts, 1);
        if remap {
            self.add(|m| &mut m.remap_attempts, 1);
        }
        let start = Instant::now();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slot.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Write, move |r| {
                let _ = tx.send(r);
            });
        let gpu = ring
            .executor
            .authoritative_gpu()
            .map_err(|e| e.to_string())?;
        gpu.device.poll(wgpu::Maintain::Wait);
        let result = rx
            .recv()
            .map_err(|e| e.to_string())
            .and_then(|r| r.map_err(|e| e.to_string()));
        let wait_us = elapsed(start);
        self.add(|m| &mut m.map_wait_us, wait_us);
        if remap {
            self.add(|m| &mut m.remap_wait_us, wait_us);
        }
        if let Err(error) = result {
            if remap {
                self.add(|m| &mut m.remap_failures, 1);
            }
            self.add(|m| &mut m.copy_failures, 1);
            return Err(error);
        }
        slot.state
            .lock()
            .transition(SlotState::Mapping, SlotState::Mapped)?;
        self.add(|m| &mut m.map_completions, 1);
        if remap {
            self.add(|m| &mut m.remap_completions, 1);
        }
        *slot.ever_mapped.lock() = true;
        Ok(lease)
    }
    pub(crate) async fn read_source(
        self: &Arc<Self>,
        storage: &NvmeStorage,
        ids: &[u32],
        buffers: Vec<PooledBuffer>,
        logical: &GpuExpertCache,
    ) -> Result<Vec<Arc<ExpertResident>>, String> {
        if self.arm != Arm::Treatment || ids.len() != buffers.len() || ids.len() > CAPACITY {
            return Err("invalid qualification source set".into());
        }
        if ids.iter().any(|id| self.pending.lock().contains_key(id)) {
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
        let mut leases = ids
            .iter()
            .map(|&id| self.acquire(id))
            .collect::<Result<Vec<_>, _>>()?;
        let mut views = leases
            .iter()
            .map(|l| l.buffer().slice(..).get_mapped_range_mut())
            .collect::<Vec<_>>();
        let offsets = views
            .iter()
            .map(|v| aligned_offset(v.as_ptr() as usize, v.len()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                self.add(|m| &mut m.alignment_failures, 1);
                e
            })?;
        let mut destinations = views
            .iter_mut()
            .zip(&offsets)
            .map(|(v, &o)| &mut v[o..o + FULL])
            .collect::<Vec<_>>();
        let (proof_hits_before, proof_misses_before) = storage.source_upload_fd_proof_snapshot();
        let started = Instant::now();
        let result = storage
            .read_experts_batch_into_aligned_slices(ids, &mut destinations)
            .await;
        self.add(|m| &mut m.fused_source_us, elapsed(started));
        let (proof_hits_after, proof_misses_after) = storage.source_upload_fd_proof_snapshot();
        self.add(
            |m| &mut m.fd_proof_cache_hits,
            proof_hits_after.saturating_sub(proof_hits_before),
        );
        self.add(
            |m| &mut m.fd_proof_cache_misses,
            proof_misses_after.saturating_sub(proof_misses_before),
        );
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
        for ((&id, view), &offset) in ids.iter().zip(&views).zip(&offsets) {
            let payload = checked_payload(&view[offset..offset + FULL])?;
            let bytes = self.materialize_source_payload(id, payload, logical)?;
            shared.push(bytes);
        }
        // Every view is gone before the first unmap, including error unwinds
        // (views was declared after leases and therefore drops first).
        drop(views);
        let mut residents = Vec::with_capacity(ids.len());
        for (((lease, offset), payload), capacity_lease) in
            leases.iter_mut().zip(offsets).zip(shared).zip(buffers)
        {
            lease.offset = offset;
            lease.payload = Some(payload.clone());
            lease.unmap();
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
                || *lease.slot().state.lock() != SlotState::Ready
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
    id: u32,
    offset: usize,
    payload: Option<Arc<[u8]>>,
    consumed: bool,
}

/// One encoder for the physical install set. Concurrent stage jobs append
/// their exact copies under this short lock. No mapping is published until
/// submit returns; upload leases remain owned here through that submission.
pub(crate) struct CopySet<'a> {
    state: &'a State,
    encoder: Mutex<Option<wgpu::CommandEncoder>>,
    leases: Mutex<Vec<Lease>>,
}
impl<'a> CopySet<'a> {
    pub(crate) fn new(state: &'a State) -> Self {
        Self {
            state,
            encoder: Mutex::new(None),
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
        let mut encoder = self.encoder.lock();
        let encoder = encoder.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qualification residency copy set"),
            })
        });
        encoder.copy_buffer_to_buffer(lease.buffer(), source, destination, offset, PAYLOAD as u64);
        self.leases.lock().push(lease);
        self.state.add(|m| &mut m.copy_encode_us, elapsed(start));
        Ok(())
    }
    pub(crate) fn submit(&self, executor: &GpuNativeExecutorContext) -> Result<(), String> {
        let Some(encoder) = self.encoder.lock().take() else {
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
impl Lease {
    fn slot(&self) -> &Slot {
        &self.state.ring.as_ref().expect("treatment ring").slots[self.index]
    }
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        &self.slot().buffer
    }
    pub(crate) fn copy_source_offset(&self) -> Result<u64, String> {
        copy_source_offset(self.offset, UPLOAD_BYTES)
    }
    pub(crate) fn context_id(&self) -> u64 {
        self.state.ring.as_ref().unwrap().executor.context_id()
    }
    fn unmap(&self) {
        self.buffer().unmap();
        *self.slot().state.lock() = SlotState::Ready;
        self.state.add(|m| &mut m.unmaps, 1);
    }
    pub(crate) fn submitted(&mut self) -> Result<(), String> {
        if self.consumed || *self.slot().state.lock() != SlotState::Ready {
            return Err("upload lease consumed twice or while mapped".into());
        }
        self.consumed = true;
        self.slot()
            .state
            .lock()
            .transition(SlotState::Ready, SlotState::Submitted)?;
        self.state.add(|m| &mut m.leases_consumed, 1);
        Ok(())
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        if matches!(
            *self.slot().state.lock(),
            SlotState::Mapping | SlotState::Mapped
        ) {
            self.unmap();
        }
        if !self.consumed {
            self.state.add(|m| &mut m.leases_dropped_unconsumed, 1);
        }
        *self.slot().state.lock() = SlotState::Available;
        self.state.add(|m| &mut m.leases_released, 1);
    }
}
pub(crate) fn elapsed(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}
pub(crate) fn aligned_offset(base: usize, capacity: usize) -> Result<usize, String> {
    let offset = (ALIGN - base % ALIGN) % ALIGN;
    if offset % 4 != 0
        || offset.checked_add(FULL).is_none_or(|end| end > capacity)
        || base.checked_add(offset).is_none_or(|p| p % ALIGN != 0)
    {
        return Err("invalid mapped alignment/range".into());
    }
    Ok(offset)
}
pub(crate) fn copy_source_offset(offset: usize, capacity: usize) -> Result<u64, String> {
    if offset >= ALIGN
        || offset % 4 != 0
        || offset.checked_add(FULL).is_none_or(|end| end > capacity)
    {
        return Err("invalid upload copy range".into());
    }
    Ok((offset + ALIGN) as u64)
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
    fn source_upload_ring_capacity_is_bounded_and_released_slots_reuse() {
        let slots = (0..CAPACITY)
            .map(|_| Mutex::new(SlotState::Available))
            .collect::<Vec<_>>();
        for i in 0..CAPACITY {
            assert_eq!(reserve_slot(slots.iter()), Some(i));
        }
        assert_eq!(reserve_slot(slots.iter()), None);
        *slots[7].lock() = SlotState::Available;
        assert_eq!(reserve_slot(slots.iter()), Some(7));
        assert_eq!(reserve_slot(slots.iter()), None);
        assert_eq!(slots.len(), 16);
    }
    #[test]
    fn source_upload_lease_map_unmap_submit_and_remap_transitions() {
        let mut state = SlotState::Available;
        for _ in 0..3 {
            state
                .transition(SlotState::Available, SlotState::Mapping)
                .unwrap();
            state
                .transition(SlotState::Mapping, SlotState::Mapped)
                .unwrap();
            state
                .transition(SlotState::Mapped, SlotState::Ready)
                .unwrap();
            state
                .transition(SlotState::Ready, SlotState::Submitted)
                .unwrap();
            state
                .transition(SlotState::Submitted, SlotState::Available)
                .unwrap();
        }
    }
    #[test]
    fn source_upload_lease_consumption_is_exactly_once_and_requires_unmap() {
        let mut state = SlotState::Mapped;
        assert!(state
            .transition(SlotState::Ready, SlotState::Submitted)
            .is_err());
        state
            .transition(SlotState::Mapped, SlotState::Ready)
            .unwrap();
        state
            .transition(SlotState::Ready, SlotState::Submitted)
            .unwrap();
        assert!(state
            .transition(SlotState::Ready, SlotState::Submitted)
            .is_err());
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
            body.find("self.acquire(id)").unwrap()
                < body
                    .find(".read_experts_batch_into_aligned_slices(")
                    .unwrap()
        );
        assert!(body.find("drop(views)").unwrap() < body.find("lease.unmap()").unwrap());
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
