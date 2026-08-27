//! Tier-aware control plane for the future GPU-native token loop.
//!
//! The engine remains responsible for NVMe reads, RAM-cache ownership,
//! speculative admission, and logical [`GpuExpertCache`] generations. This
//! module owns only the preallocated RAM -> VRAM physical plane built from
//! Slice 8's mutable Q4 expert arenas.

#![allow(dead_code)]

use crate::backend::gpu_native::{
    GpuNativeBootstrapError, GpuNativeExecutorContext, GpuNativeQ4ExpertAcquire,
    GpuNativeQ4ExpertArena, GpuNativeQ4ExpertGeometry, GpuNativeQ4ExpertKey,
    GpuNativeQ4ExpertResidency, GpuNativeQ4ExpertRetire, GpuNativeQ4ExpertVramPlan,
};
use crate::expert_cache::{ExpertResident, GpuAdmission, GpuExpertCache};
use lru::LruCache;
use parking_lot::{Mutex, MutexGuard};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeTieredResidencyError {
    InvalidModelLayerCount,
    InvalidExpertsPerLayer,
    ExpertNamespaceOverflow {
        num_layers: usize,
        experts_per_layer: usize,
    },
    GlobalExpertOutOfRange {
        global_id: u32,
        num_layers: usize,
        experts_per_layer: u32,
    },
    LayerOutOfRange {
        layer_index: usize,
        num_layers: usize,
    },
    LocalExpertOutOfRange {
        local_expert_id: u32,
        experts_per_layer: u32,
    },
    ModelBudgetOverflow,
    ModelBudgetTooSmall {
        requested_bytes: u64,
        minimum_bytes: u64,
    },
    DuplicateDemandExpert {
        global_id: u32,
    },
    DemandLayerMismatch {
        requested_layer: usize,
        global_id: u32,
        actual_layer: usize,
    },
    DemandSetExceedsLayerCapacity {
        requested: usize,
        capacity: usize,
    },
    DemandSourceMissing {
        global_id: u32,
    },
    DemandSourceIdentityMismatch {
        global_id: u32,
    },
    LogicalAdmissionStale {
        global_id: u32,
        generation: u64,
    },
    InstallInProgress {
        global_id: u32,
        generation: u64,
    },
    NoPhysicalSlot {
        layer_index: usize,
    },
    NoEvictablePhysicalSlot {
        layer_index: usize,
    },
    StalePhysicalRequester {
        global_id: u32,
        generation: u64,
    },
    PhysicalIdentityCorrupt {
        global_id: u32,
    },
    ResidencyPriorityMismatch,
    Backend(GpuNativeBootstrapError),
}

impl fmt::Display for GpuNativeTieredResidencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModelLayerCount => f.write_str("GPU-native residency requires at least one MoE layer"),
            Self::InvalidExpertsPerLayer => f.write_str("GPU-native residency requires at least one expert per layer"),
            Self::ExpertNamespaceOverflow { num_layers, experts_per_layer } => write!(f, "global expert namespace overflows u32: layers={num_layers} experts_per_layer={experts_per_layer}"),
            Self::GlobalExpertOutOfRange { global_id, num_layers, experts_per_layer } => write!(f, "global expert {global_id} is outside {num_layers} layers x {experts_per_layer} experts"),
            Self::LayerOutOfRange { layer_index, num_layers } => write!(f, "layer {layer_index} is outside {num_layers} GPU-native arenas"),
            Self::LocalExpertOutOfRange { local_expert_id, experts_per_layer } => write!(f, "local expert {local_expert_id} is outside layer width {experts_per_layer}"),
            Self::ModelBudgetOverflow => f.write_str("model-wide GPU-native expert budget arithmetic overflowed"),
            Self::ModelBudgetTooSmall { requested_bytes, minimum_bytes } => write!(f, "model-wide expert budget {requested_bytes} bytes is below the executable minimum {minimum_bytes} bytes"),
            Self::DuplicateDemandExpert { global_id } => write!(f, "demand set repeats global expert {global_id}"),
            Self::DemandLayerMismatch { requested_layer, global_id, actual_layer } => write!(f, "demand for layer {requested_layer} contains global expert {global_id} from layer {actual_layer}"),
            Self::DemandSetExceedsLayerCapacity { requested, capacity } => write!(f, "demand set of {requested} experts exceeds physical layer capacity {capacity}"),
            Self::DemandSourceMissing { global_id } => write!(f, "physical miss for global expert {global_id} has no RAM/logical-admission source"),
            Self::DemandSourceIdentityMismatch { global_id } => write!(f, "RAM resident or logical admission does not match global expert {global_id}"),
            Self::LogicalAdmissionStale { global_id, generation } => write!(f, "logical generation {generation} for global expert {global_id} is no longer current"),
            Self::InstallInProgress { global_id, generation } => write!(f, "physical install is already in progress for global expert {global_id} generation {generation}"),
            Self::NoPhysicalSlot { layer_index } => write!(f, "layer {layer_index} has no free physical expert slot"),
            Self::NoEvictablePhysicalSlot { layer_index } => write!(f, "layer {layer_index} has no physical victim outside the protected demand set"),
            Self::StalePhysicalRequester { global_id, generation } => write!(f, "stale physical requester for global expert {global_id} generation {generation}"),
            Self::PhysicalIdentityCorrupt { global_id } => write!(f, "GPU-native physical metadata disagrees with the authoritative arena for global expert {global_id}"),
            Self::ResidencyPriorityMismatch => f.write_str("residency request used the wrong demand/speculative priority"),
            Self::Backend(error) => write!(f, "GPU-native residency backend error: {error}"),
        }
    }
}

impl std::error::Error for GpuNativeTieredResidencyError {}

impl From<GpuNativeBootstrapError> for GpuNativeTieredResidencyError {
    fn from(value: GpuNativeBootstrapError) -> Self {
        Self::Backend(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GpuNativeLayerExpertId {
    pub(crate) layer_index: usize,
    pub(crate) local_expert_id: u32,
}

pub(crate) fn global_to_layer_local(
    global_id: u32,
    num_layers: usize,
    experts_per_layer: u32,
) -> Result<GpuNativeLayerExpertId, GpuNativeTieredResidencyError> {
    validate_namespace(num_layers, experts_per_layer as usize)?;
    let layer_index = (global_id / experts_per_layer) as usize;
    if layer_index >= num_layers {
        return Err(GpuNativeTieredResidencyError::GlobalExpertOutOfRange {
            global_id,
            num_layers,
            experts_per_layer,
        });
    }
    Ok(GpuNativeLayerExpertId {
        layer_index,
        local_expert_id: global_id % experts_per_layer,
    })
}

pub(crate) fn layer_local_to_global(
    layer_index: usize,
    local_expert_id: u32,
    num_layers: usize,
    experts_per_layer: u32,
) -> Result<u32, GpuNativeTieredResidencyError> {
    validate_namespace(num_layers, experts_per_layer as usize)?;
    if layer_index >= num_layers {
        return Err(GpuNativeTieredResidencyError::LayerOutOfRange {
            layer_index,
            num_layers,
        });
    }
    if local_expert_id >= experts_per_layer {
        return Err(GpuNativeTieredResidencyError::LocalExpertOutOfRange {
            local_expert_id,
            experts_per_layer,
        });
    }
    let global = (layer_index as u64)
        .checked_mul(experts_per_layer as u64)
        .and_then(|base| base.checked_add(local_expert_id as u64))
        .and_then(|id| u32::try_from(id).ok())
        .ok_or(GpuNativeTieredResidencyError::ExpertNamespaceOverflow {
            num_layers,
            experts_per_layer: experts_per_layer as usize,
        })?;
    Ok(global)
}

pub(crate) fn global_to_q4_expert_key(
    global_id: u32,
    logical_generation: u64,
    num_layers: usize,
    experts_per_layer: u32,
) -> Result<GpuNativeQ4ExpertKey, GpuNativeTieredResidencyError> {
    let identity = global_to_layer_local(global_id, num_layers, experts_per_layer)?;
    Ok(GpuNativeQ4ExpertKey::new(
        identity.layer_index,
        identity.local_expert_id,
        logical_generation,
    ))
}

fn validate_namespace(
    num_layers: usize,
    experts_per_layer: usize,
) -> Result<(), GpuNativeTieredResidencyError> {
    if num_layers == 0 {
        return Err(GpuNativeTieredResidencyError::InvalidModelLayerCount);
    }
    if experts_per_layer == 0 {
        return Err(GpuNativeTieredResidencyError::InvalidExpertsPerLayer);
    }
    let count = (num_layers as u128)
        .checked_mul(experts_per_layer as u128)
        .ok_or(GpuNativeTieredResidencyError::ExpertNamespaceOverflow {
            num_layers,
            experts_per_layer,
        })?;
    if count > u32::MAX as u128 + 1 {
        return Err(GpuNativeTieredResidencyError::ExpertNamespaceOverflow {
            num_layers,
            experts_per_layer,
        });
    }
    Ok(())
}

/// One deterministic model-wide post-headroom expert-VRAM plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeModelExpertVramPlan {
    geometry: GpuNativeQ4ExpertGeometry,
    total_expert_budget_bytes: u64,
    minimum_executable_budget_bytes: u64,
    layer_plans: Vec<GpuNativeQ4ExpertVramPlan>,
    total_arena_allocation_bytes: u64,
}

impl GpuNativeModelExpertVramPlan {
    pub(crate) fn try_new(
        num_layers: usize,
        geometry: GpuNativeQ4ExpertGeometry,
        total_expert_budget_bytes: u64,
        limits: &wgpu::Limits,
    ) -> Result<Self, GpuNativeTieredResidencyError> {
        validate_namespace(num_layers, geometry.num_experts())?;
        let minimum_layer =
            GpuNativeQ4ExpertVramPlan::try_for_slot_capacity(geometry, geometry.top_k(), limits)?;
        let minimum_executable_budget_bytes = minimum_layer
            .total_arena_allocation_bytes()
            .checked_mul(num_layers as u64)
            .ok_or(GpuNativeTieredResidencyError::ModelBudgetOverflow)?;
        if total_expert_budget_bytes < minimum_executable_budget_bytes {
            return Err(GpuNativeTieredResidencyError::ModelBudgetTooSmall {
                requested_bytes: total_expert_budget_bytes,
                minimum_bytes: minimum_executable_budget_bytes,
            });
        }

        let mut layer_plans = vec![minimum_layer; num_layers];
        let mut total_arena_allocation_bytes = minimum_executable_budget_bytes;
        loop {
            let remaining = total_expert_budget_bytes - total_arena_allocation_bytes;
            let mut best: Option<(u64, usize, GpuNativeQ4ExpertVramPlan)> = None;
            for (layer_index, current) in layer_plans.iter().copied().enumerate() {
                if current.slot_capacity() >= geometry.num_experts() {
                    continue;
                }
                let candidate = GpuNativeQ4ExpertVramPlan::try_for_slot_capacity(
                    geometry,
                    current.slot_capacity() + 1,
                    limits,
                )?;
                let increment = candidate
                    .total_arena_allocation_bytes()
                    .checked_sub(current.total_arena_allocation_bytes())
                    .ok_or(GpuNativeTieredResidencyError::ModelBudgetOverflow)?;
                if increment > remaining {
                    continue;
                }
                if best.as_ref().is_none_or(|(best_increment, best_layer, _)| {
                    (increment, layer_index) < (*best_increment, *best_layer)
                }) {
                    best = Some((increment, layer_index, candidate));
                }
            }
            let Some((increment, layer_index, candidate)) = best else {
                break;
            };
            layer_plans[layer_index] = candidate;
            total_arena_allocation_bytes = total_arena_allocation_bytes
                .checked_add(increment)
                .ok_or(GpuNativeTieredResidencyError::ModelBudgetOverflow)?;
        }

        debug_assert!(total_arena_allocation_bytes <= total_expert_budget_bytes);
        Ok(Self {
            geometry,
            total_expert_budget_bytes,
            minimum_executable_budget_bytes,
            layer_plans,
            total_arena_allocation_bytes,
        })
    }

    pub(crate) const fn geometry(&self) -> GpuNativeQ4ExpertGeometry {
        self.geometry
    }

    pub(crate) const fn total_expert_budget_bytes(&self) -> u64 {
        self.total_expert_budget_bytes
    }

    pub(crate) const fn minimum_executable_budget_bytes(&self) -> u64 {
        self.minimum_executable_budget_bytes
    }

    pub(crate) fn num_layers(&self) -> usize {
        self.layer_plans.len()
    }

    pub(crate) fn layer_plans(&self) -> &[GpuNativeQ4ExpertVramPlan] {
        &self.layer_plans
    }

    pub(crate) const fn total_arena_allocation_bytes(&self) -> u64 {
        self.total_arena_allocation_bytes
    }

    pub(crate) const fn unused_remainder_bytes(&self) -> u64 {
        self.total_expert_budget_bytes - self.total_arena_allocation_bytes
    }

    pub(crate) fn model_slot_capacity(&self) -> usize {
        self.layer_plans
            .iter()
            .map(|plan| plan.slot_capacity())
            .sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum GpuNativeResidencyPriority {
    Demand,
    Speculative { score: f64 },
}

#[derive(Clone)]
pub(crate) enum GpuNativeDemandExpert {
    Current {
        global_id: u32,
    },
    Install {
        global_id: u32,
        resident: Arc<ExpertResident>,
        admission: GpuAdmission,
    },
}

impl GpuNativeDemandExpert {
    pub(crate) const fn current(global_id: u32) -> Self {
        Self::Current { global_id }
    }

    pub(crate) fn install(
        global_id: u32,
        resident: Arc<ExpertResident>,
        admission: GpuAdmission,
    ) -> Self {
        Self::Install {
            global_id,
            resident,
            admission,
        }
    }

    const fn global_id(&self) -> u32 {
        match self {
            Self::Current { global_id } | Self::Install { global_id, .. } => *global_id,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeSpeculativeProbe {
    Hit(GpuNativeQ4ExpertResidency),
    Miss,
    DroppedPressure,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeSpeculativeInstall {
    Hit(GpuNativeQ4ExpertResidency),
    Installed(GpuNativeQ4ExpertResidency),
    DroppedCapacityOrPressure,
    StaleLogicalGeneration,
}

#[derive(Clone, Copy)]
struct PhysicalRecord {
    key: GpuNativeQ4ExpertKey,
    residency: GpuNativeQ4ExpertResidency,
}

struct LayerResidencyState {
    residents: LruCache<u32, PhysicalRecord>,
    last_installed_generations: HashMap<u32, u64>,
    physical_evictions: u64,
}

impl Default for LayerResidencyState {
    fn default() -> Self {
        Self {
            residents: LruCache::unbounded(),
            last_installed_generations: HashMap::new(),
            physical_evictions: 0,
        }
    }
}

fn touch_physical_record<T: Copy>(residents: &mut LruCache<u32, T>, global_id: u32) -> Option<T> {
    residents.get(&global_id).copied()
}

fn validate_physical_install_source(
    gpu_cache: &GpuExpertCache,
    global_id: u32,
    resident: &Arc<ExpertResident>,
    admission: &GpuAdmission,
) -> Result<(), GpuNativeTieredResidencyError> {
    if resident.id != global_id || admission.resident().id != global_id {
        return Err(GpuNativeTieredResidencyError::DemandSourceIdentityMismatch { global_id });
    }
    if !gpu_cache.contains_generation(global_id, admission.generation()) {
        return Err(GpuNativeTieredResidencyError::LogicalAdmissionStale {
            global_id,
            generation: admission.generation(),
        });
    }
    Ok(())
}

struct LayerResidency {
    arena: Arc<GpuNativeQ4ExpertArena>,
    state: Mutex<LayerResidencyState>,
}

#[derive(Default)]
struct TieredResidencyCounters {
    vram_hits: AtomicU64,
    vram_misses: AtomicU64,
    physical_current_hits: AtomicU64,
    physical_source_acquisitions: AtomicU64,
    logical_admissions_for_physical_misses: AtomicU64,
    ram_to_vram_installs: AtomicU64,
    physical_evictions: AtomicU64,
    physical_reinstalls: AtomicU64,
    stale_generation_rejections: AtomicU64,
    demand_requests: AtomicU64,
    speculative_requests: AtomicU64,
    speculative_vram_hits: AtomicU64,
    speculative_ram_to_vram_installs: AtomicU64,
    speculative_dropped_capacity_or_pressure: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct GpuNativeTieredLayerSnapshot {
    pub(crate) slot_capacity: usize,
    pub(crate) resident_slots: usize,
    pub(crate) free_slots: usize,
    pub(crate) physical_evictions: u64,
}

/// Read-only copy of one layer's physical metadata for shadow analysis.
///
/// `resident_global_ids_mru_to_lru` preserves the production physical LRU
/// ordering but is collected with `LruCache::iter`, which does not promote an
/// entry or update any residency counter.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct GpuNativePhysicalLayerShadowSnapshot {
    pub(crate) layer_index: usize,
    pub(crate) slot_capacity: usize,
    pub(crate) resident_global_ids_mru_to_lru: Vec<u32>,
    pub(crate) free_slots: usize,
}

/// Read-only observer-window evidence for one relevant physical layer.
///
/// The metadata snapshot preserves exact host MRU-to-LRU ordering while the
/// arena snapshot includes physical install, retire, reuse, and mapping
/// counters. Both are copied and released before an observer callback runs.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct GpuNativeObserverLayerGuardSnapshot {
    pub(crate) metadata: GpuNativePhysicalLayerShadowSnapshot,
    pub(crate) arena: crate::backend::gpu_native::GpuNativeQ4ExpertResidencySnapshot,
}

/// Complete read-only production-residency evidence captured on one side of
/// a shadow-observer callback. No lock represented here remains held after
/// construction.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct GpuNativeObserverResidencyGuardSnapshot {
    pub(crate) tiered: GpuNativeTieredResidencySnapshot,
    pub(crate) relevant_layers: Vec<GpuNativeObserverLayerGuardSnapshot>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct GpuNativeTieredResidencySnapshot {
    pub(crate) model_expert_budget_bytes: u64,
    pub(crate) model_arena_allocation_bytes: u64,
    pub(crate) model_slot_capacity: usize,
    pub(crate) resident_physical_slots: usize,
    pub(crate) free_physical_slots: usize,
    pub(crate) vram_hits: u64,
    pub(crate) vram_misses: u64,
    pub(crate) physical_current_hits: u64,
    pub(crate) physical_source_acquisitions: u64,
    pub(crate) logical_admissions_for_physical_misses: u64,
    pub(crate) ram_to_vram_installs: u64,
    pub(crate) physical_evictions: u64,
    pub(crate) physical_reinstalls: u64,
    pub(crate) stale_generation_rejections: u64,
    pub(crate) demand_requests: u64,
    pub(crate) speculative_requests: u64,
    pub(crate) speculative_vram_hits: u64,
    pub(crate) speculative_ram_to_vram_installs: u64,
    pub(crate) speculative_dropped_capacity_or_pressure: u64,
    pub(crate) layers: Vec<GpuNativeTieredLayerSnapshot>,
}

/// Model-scoped owner of the preallocated per-layer Q4 expert arenas.
pub(crate) struct GpuNativeTieredResidencyManager {
    executor: Arc<GpuNativeExecutorContext>,
    gpu_cache: Arc<GpuExpertCache>,
    plan: GpuNativeModelExpertVramPlan,
    layers: Vec<LayerResidency>,
    counters: TieredResidencyCounters,
}

impl GpuNativeTieredResidencyManager {
    pub(crate) fn try_new(
        executor: Arc<GpuNativeExecutorContext>,
        gpu_cache: Arc<GpuExpertCache>,
        num_layers: usize,
        geometry: GpuNativeQ4ExpertGeometry,
        total_expert_budget_bytes: u64,
    ) -> Result<Self, GpuNativeTieredResidencyError> {
        let limits = executor.device_limits()?;
        let plan = GpuNativeModelExpertVramPlan::try_new(
            num_layers,
            geometry,
            total_expert_budget_bytes,
            &limits,
        )?;
        let mut layers = Vec::with_capacity(num_layers);
        for (layer_index, layer_plan) in plan.layer_plans().iter().copied().enumerate() {
            let arena = executor.create_q4_expert_arena(layer_index, layer_plan, &[])?;
            layers.push(LayerResidency {
                arena: Arc::new(arena),
                state: Mutex::new(LayerResidencyState::default()),
            });
        }
        Ok(Self {
            executor,
            gpu_cache,
            plan,
            layers,
            counters: TieredResidencyCounters::default(),
        })
    }

    pub(crate) fn executor(&self) -> &Arc<GpuNativeExecutorContext> {
        &self.executor
    }

    pub(crate) fn gpu_cache(&self) -> &Arc<GpuExpertCache> {
        &self.gpu_cache
    }

    pub(crate) fn plan(&self) -> &GpuNativeModelExpertVramPlan {
        &self.plan
    }

    pub(crate) fn arena(&self, layer_index: usize) -> Option<&Arc<GpuNativeQ4ExpertArena>> {
        self.layers.get(layer_index).map(|layer| &layer.arena)
    }

    fn identity(
        &self,
        global_id: u32,
    ) -> Result<GpuNativeLayerExpertId, GpuNativeTieredResidencyError> {
        global_to_layer_local(
            global_id,
            self.plan.num_layers(),
            self.plan.geometry().num_experts() as u32,
        )
    }

    /// Read-only physical tier-selection probe for the engine's async demand
    /// path. Host logical-admission LRU state is intentionally not consulted:
    /// an exact, internally current arena residency owns immutable executable
    /// bytes for the lifetime of this manager. The `touch=false` lookup does
    /// not promote physical LRU recency or update any residency counter, and
    /// exact arena-identity disagreement propagates as `PhysicalIdentityCorrupt`.
    pub(crate) fn has_current_for_demand(
        &self,
        global_id: u32,
    ) -> Result<bool, GpuNativeTieredResidencyError> {
        let identity = self.identity(global_id)?;
        let layer = &self.layers[identity.layer_index];
        let mut state = layer.state.lock();
        Ok(self
            .current_record_locked(global_id, layer, &mut state, false)?
            .is_some())
    }

    pub(crate) fn record_physical_source_acquisition(&self) {
        self.counters
            .physical_source_acquisitions
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_logical_admissions_for_physical_misses(&self, count: usize) {
        self.counters
            .logical_admissions_for_physical_misses
            .fetch_add(count as u64, Ordering::Relaxed);
    }

    /// VRAM-first speculative probe. It never waits behind a demand mutation;
    /// contention is an immediate best-effort drop.
    pub(crate) fn probe_speculative(
        &self,
        global_id: u32,
        priority: GpuNativeResidencyPriority,
    ) -> Result<GpuNativeSpeculativeProbe, GpuNativeTieredResidencyError> {
        let GpuNativeResidencyPriority::Speculative { score: _ } = priority else {
            return Err(GpuNativeTieredResidencyError::ResidencyPriorityMismatch);
        };
        self.counters
            .speculative_requests
            .fetch_add(1, Ordering::Relaxed);
        let identity = self.identity(global_id)?;
        let layer = &self.layers[identity.layer_index];
        let Some(mut state) = layer.state.try_lock() else {
            self.record_speculative_drop();
            return Ok(GpuNativeSpeculativeProbe::DroppedPressure);
        };
        if let Some(record) = self.current_record_locked(global_id, layer, &mut state, true)? {
            self.counters.vram_hits.fetch_add(1, Ordering::Relaxed);
            self.counters
                .speculative_vram_hits
                .fetch_add(1, Ordering::Relaxed);
            Ok(GpuNativeSpeculativeProbe::Hit(record.residency))
        } else {
            self.counters.vram_misses.fetch_add(1, Ordering::Relaxed);
            Ok(GpuNativeSpeculativeProbe::Miss)
        }
    }

    pub(crate) fn ensure_demand_set(
        &self,
        priority: GpuNativeResidencyPriority,
        layer_index: usize,
        demands: &[GpuNativeDemandExpert],
    ) -> Result<Vec<GpuNativeQ4ExpertResidency>, GpuNativeTieredResidencyError> {
        if !matches!(priority, GpuNativeResidencyPriority::Demand) {
            return Err(GpuNativeTieredResidencyError::ResidencyPriorityMismatch);
        }
        let layer =
            self.layers
                .get(layer_index)
                .ok_or(GpuNativeTieredResidencyError::LayerOutOfRange {
                    layer_index,
                    num_layers: self.layers.len(),
                })?;
        if demands.len() > layer.arena.slot_capacity() {
            return Err(
                GpuNativeTieredResidencyError::DemandSetExceedsLayerCapacity {
                    requested: demands.len(),
                    capacity: layer.arena.slot_capacity(),
                },
            );
        }
        let mut protected = HashSet::with_capacity(demands.len());
        for demand in demands {
            let global_id = demand.global_id();
            if !protected.insert(global_id) {
                return Err(GpuNativeTieredResidencyError::DuplicateDemandExpert { global_id });
            }
            let identity = self.identity(global_id)?;
            if identity.layer_index != layer_index {
                return Err(GpuNativeTieredResidencyError::DemandLayerMismatch {
                    requested_layer: layer_index,
                    global_id,
                    actual_layer: identity.layer_index,
                });
            }
            if let GpuNativeDemandExpert::Install {
                resident,
                admission,
                ..
            } = demand
            {
                self.validate_source(global_id, resident, admission)?;
            }
        }

        self.counters
            .demand_requests
            .fetch_add(demands.len() as u64, Ordering::Relaxed);
        let mut state = layer.state.lock();
        let mut resolved = vec![None; demands.len()];
        let mut misses = Vec::new();
        for (index, demand) in demands.iter().enumerate() {
            let global_id = demand.global_id();
            if let Some(record) = self.current_record_locked(global_id, layer, &mut state, true)? {
                self.counters.vram_hits.fetch_add(1, Ordering::Relaxed);
                self.counters
                    .physical_current_hits
                    .fetch_add(1, Ordering::Relaxed);
                resolved[index] = Some(record.residency);
            } else {
                self.counters.vram_misses.fetch_add(1, Ordering::Relaxed);
                match demand {
                    GpuNativeDemandExpert::Install { .. } => misses.push(index),
                    GpuNativeDemandExpert::Current { .. } => {
                        return Err(GpuNativeTieredResidencyError::DemandSourceMissing {
                            global_id,
                        });
                    }
                }
            }
        }

        while state.residents.len().saturating_add(misses.len()) > layer.arena.slot_capacity() {
            let victim = oldest_unprotected(&state.residents, &protected)
                .ok_or(GpuNativeTieredResidencyError::NoEvictablePhysicalSlot { layer_index })?;
            self.retire_metadata_record_locked(victim, layer, &mut state, true)?;
        }

        for index in misses {
            let GpuNativeDemandExpert::Install {
                global_id,
                resident,
                admission,
            } = &demands[index]
            else {
                unreachable!("miss list contains only install sources")
            };
            let residency =
                self.install_locked(*global_id, resident, admission, layer, &mut state, false)?;
            resolved[index] = Some(residency);
        }
        resolved
            .into_iter()
            .enumerate()
            .map(|(index, residency)| {
                residency.ok_or(GpuNativeTieredResidencyError::DemandSourceMissing {
                    global_id: demands[index].global_id(),
                })
            })
            .collect()
    }

    pub(crate) fn ensure_speculative_resident(
        &self,
        global_id: u32,
        resident: &Arc<ExpertResident>,
        admission: &GpuAdmission,
        priority: GpuNativeResidencyPriority,
    ) -> Result<GpuNativeSpeculativeInstall, GpuNativeTieredResidencyError> {
        let GpuNativeResidencyPriority::Speculative { score: _ } = priority else {
            return Err(GpuNativeTieredResidencyError::ResidencyPriorityMismatch);
        };
        if let Err(error) = self.validate_source(global_id, resident, admission) {
            if matches!(
                error,
                GpuNativeTieredResidencyError::LogicalAdmissionStale { .. }
            ) {
                return Ok(GpuNativeSpeculativeInstall::StaleLogicalGeneration);
            }
            return Err(error);
        }
        let identity = self.identity(global_id)?;
        let layer = &self.layers[identity.layer_index];
        let Some(mut state) = layer.state.try_lock() else {
            self.record_speculative_drop();
            return Ok(GpuNativeSpeculativeInstall::DroppedCapacityOrPressure);
        };
        if let Some(record) = self.current_record_locked(global_id, layer, &mut state, true)? {
            self.counters.vram_hits.fetch_add(1, Ordering::Relaxed);
            self.counters
                .speculative_vram_hits
                .fetch_add(1, Ordering::Relaxed);
            return Ok(GpuNativeSpeculativeInstall::Hit(record.residency));
        }
        if !self
            .gpu_cache
            .contains_generation(global_id, admission.generation())
        {
            self.counters
                .stale_generation_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Ok(GpuNativeSpeculativeInstall::StaleLogicalGeneration);
        }
        if state.residents.len() >= layer.arena.slot_capacity() {
            self.record_speculative_drop();
            return Ok(GpuNativeSpeculativeInstall::DroppedCapacityOrPressure);
        }
        let residency =
            match self.install_locked(global_id, resident, admission, layer, &mut state, true) {
                Ok(residency) => residency,
                Err(GpuNativeTieredResidencyError::NoPhysicalSlot { .. }) => {
                    return Ok(GpuNativeSpeculativeInstall::DroppedCapacityOrPressure);
                }
                Err(error) => return Err(error),
            };
        Ok(GpuNativeSpeculativeInstall::Installed(residency))
    }

    pub(crate) fn snapshot(&self) -> GpuNativeTieredResidencySnapshot {
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let arena = layer.arena.residency_snapshot();
                let state = layer.state.lock();
                GpuNativeTieredLayerSnapshot {
                    slot_capacity: arena.slot_capacity,
                    resident_slots: arena.resident_slots,
                    free_slots: arena.free_slots,
                    physical_evictions: state.physical_evictions,
                }
            })
            .collect::<Vec<_>>();
        GpuNativeTieredResidencySnapshot {
            model_expert_budget_bytes: self.plan.total_expert_budget_bytes(),
            model_arena_allocation_bytes: self.plan.total_arena_allocation_bytes(),
            model_slot_capacity: self.plan.model_slot_capacity(),
            resident_physical_slots: layers.iter().map(|layer| layer.resident_slots).sum(),
            free_physical_slots: layers.iter().map(|layer| layer.free_slots).sum(),
            vram_hits: self.counters.vram_hits.load(Ordering::Relaxed),
            vram_misses: self.counters.vram_misses.load(Ordering::Relaxed),
            physical_current_hits: self.counters.physical_current_hits.load(Ordering::Relaxed),
            physical_source_acquisitions: self
                .counters
                .physical_source_acquisitions
                .load(Ordering::Relaxed),
            logical_admissions_for_physical_misses: self
                .counters
                .logical_admissions_for_physical_misses
                .load(Ordering::Relaxed),
            ram_to_vram_installs: self.counters.ram_to_vram_installs.load(Ordering::Relaxed),
            physical_evictions: self.counters.physical_evictions.load(Ordering::Relaxed),
            physical_reinstalls: self.counters.physical_reinstalls.load(Ordering::Relaxed),
            stale_generation_rejections: self
                .counters
                .stale_generation_rejections
                .load(Ordering::Relaxed),
            demand_requests: self.counters.demand_requests.load(Ordering::Relaxed),
            speculative_requests: self.counters.speculative_requests.load(Ordering::Relaxed),
            speculative_vram_hits: self.counters.speculative_vram_hits.load(Ordering::Relaxed),
            speculative_ram_to_vram_installs: self
                .counters
                .speculative_ram_to_vram_installs
                .load(Ordering::Relaxed),
            speculative_dropped_capacity_or_pressure: self
                .counters
                .speculative_dropped_capacity_or_pressure
                .load(Ordering::Relaxed),
            layers,
        }
    }

    /// Copy one layer's physical capacity, resident ids, and exact LRU order
    /// without acquiring payloads, touching recency, or changing counters.
    pub(crate) fn shadow_layer_snapshot(
        &self,
        layer_index: usize,
    ) -> Result<GpuNativePhysicalLayerShadowSnapshot, GpuNativeTieredResidencyError> {
        let layer =
            self.layers
                .get(layer_index)
                .ok_or(GpuNativeTieredResidencyError::LayerOutOfRange {
                    layer_index,
                    num_layers: self.layers.len(),
                })?;
        let state = layer.state.lock();
        let resident_global_ids_mru_to_lru = state
            .residents
            .iter()
            .map(|(&global_id, _)| global_id)
            .collect::<Vec<_>>();
        let slot_capacity = layer.arena.slot_capacity();
        Ok(GpuNativePhysicalLayerShadowSnapshot {
            layer_index,
            slot_capacity,
            free_slots: slot_capacity.saturating_sub(resident_global_ids_mru_to_lru.len()),
            resident_global_ids_mru_to_lru,
        })
    }

    /// Capture the runtime state that must remain exactly unchanged across a
    /// shadow-observer callback. The range is validated and every lock is
    /// released before this function returns, so callers never hold a
    /// production lock across diagnostic work.
    pub(crate) fn shadow_observer_guard_snapshot(
        &self,
        relevant_layers: &[usize],
    ) -> Result<GpuNativeObserverResidencyGuardSnapshot, GpuNativeTieredResidencyError> {
        let tiered = self.snapshot();
        let relevant_layers = relevant_layers
            .iter()
            .copied()
            .map(|layer_index| {
                let layer = self.layers.get(layer_index).ok_or(
                    GpuNativeTieredResidencyError::LayerOutOfRange {
                        layer_index,
                        num_layers: self.layers.len(),
                    },
                )?;
                Ok(GpuNativeObserverLayerGuardSnapshot {
                    metadata: self.shadow_layer_snapshot(layer_index)?,
                    arena: layer.arena.residency_snapshot(),
                })
            })
            .collect::<Result<Vec<_>, GpuNativeTieredResidencyError>>()?;
        Ok(GpuNativeObserverResidencyGuardSnapshot {
            tiered,
            relevant_layers,
        })
    }

    fn validate_source(
        &self,
        global_id: u32,
        resident: &Arc<ExpertResident>,
        admission: &GpuAdmission,
    ) -> Result<(), GpuNativeTieredResidencyError> {
        let result = validate_physical_install_source(
            self.gpu_cache.as_ref(),
            global_id,
            resident,
            admission,
        );
        if matches!(
            result,
            Err(GpuNativeTieredResidencyError::LogicalAdmissionStale { .. })
        ) {
            self.counters
                .stale_generation_rejections
                .fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    fn current_record_locked(
        &self,
        global_id: u32,
        layer: &LayerResidency,
        state: &mut MutexGuard<'_, LayerResidencyState>,
        touch: bool,
    ) -> Result<Option<PhysicalRecord>, GpuNativeTieredResidencyError> {
        let Some(record) = state.residents.peek(&global_id).copied() else {
            return Ok(None);
        };
        let identity = self.identity(global_id)?;
        if record.key.layer_index() != identity.layer_index
            || record.key.expert_id() != identity.local_expert_id
            || record.residency.key() != record.key
            || layer.arena.layer_index() != identity.layer_index
            || !layer
                .arena
                .contains_exact_residency(self.executor.context_id(), record.residency)
        {
            return Err(GpuNativeTieredResidencyError::PhysicalIdentityCorrupt { global_id });
        }
        if touch {
            let record = touch_physical_record(&mut state.residents, global_id)
                .expect("peeked current physical record remains under layer lock");
            Ok(Some(record))
        } else {
            Ok(Some(record))
        }
    }

    fn retire_metadata_record_locked(
        &self,
        global_id: u32,
        layer: &LayerResidency,
        state: &mut MutexGuard<'_, LayerResidencyState>,
        capacity_eviction: bool,
    ) -> Result<(), GpuNativeTieredResidencyError> {
        let Some(record) = state.residents.peek(&global_id).copied() else {
            return Ok(());
        };
        match self
            .executor
            .retire_q4_expert_residency(&layer.arena, record.key)?
        {
            GpuNativeQ4ExpertRetire::Retired
            | GpuNativeQ4ExpertRetire::CancelledInstall
            | GpuNativeQ4ExpertRetire::NotResident
            | GpuNativeQ4ExpertRetire::StaleRequester => {}
        }
        state.residents.pop(&global_id);
        if capacity_eviction {
            state.physical_evictions = state.physical_evictions.saturating_add(1);
            self.counters
                .physical_evictions
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn install_locked(
        &self,
        global_id: u32,
        resident: &Arc<ExpertResident>,
        admission: &GpuAdmission,
        layer: &LayerResidency,
        state: &mut MutexGuard<'_, LayerResidencyState>,
        speculative: bool,
    ) -> Result<GpuNativeQ4ExpertResidency, GpuNativeTieredResidencyError> {
        let identity = self.identity(global_id)?;
        let key = global_to_q4_expert_key(
            global_id,
            admission.generation(),
            self.plan.num_layers(),
            self.plan.geometry().num_experts() as u32,
        )?;
        let (residency, installed) = match self
            .executor
            .acquire_q4_expert_residency(&layer.arena, key)?
        {
            GpuNativeQ4ExpertAcquire::Hit(hit) => (hit, false),
            GpuNativeQ4ExpertAcquire::Install(permit) => (
                self.executor
                    .install_q4_expert_residency(permit, resident.data())?,
                true,
            ),
            GpuNativeQ4ExpertAcquire::InstallInProgress => {
                return Err(GpuNativeTieredResidencyError::InstallInProgress {
                    global_id,
                    generation: admission.generation(),
                });
            }
            GpuNativeQ4ExpertAcquire::StaleRequester => {
                self.counters
                    .stale_generation_rejections
                    .fetch_add(1, Ordering::Relaxed);
                return Err(GpuNativeTieredResidencyError::StalePhysicalRequester {
                    global_id,
                    generation: admission.generation(),
                });
            }
            GpuNativeQ4ExpertAcquire::NoPhysicalSlot => {
                if speculative {
                    self.record_speculative_drop();
                }
                return Err(GpuNativeTieredResidencyError::NoPhysicalSlot {
                    layer_index: identity.layer_index,
                });
            }
        };
        if !self
            .gpu_cache
            .contains_generation(global_id, admission.generation())
        {
            let _ = self
                .executor
                .retire_q4_expert_residency(&layer.arena, key)?;
            self.counters
                .stale_generation_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Err(GpuNativeTieredResidencyError::LogicalAdmissionStale {
                global_id,
                generation: admission.generation(),
            });
        }
        let reinstall = installed
            && state
                .last_installed_generations
                .insert(global_id, admission.generation())
                == Some(admission.generation());
        state
            .residents
            .put(global_id, PhysicalRecord { key, residency });
        if installed {
            self.counters
                .ram_to_vram_installs
                .fetch_add(1, Ordering::Relaxed);
            if reinstall {
                self.counters
                    .physical_reinstalls
                    .fetch_add(1, Ordering::Relaxed);
            }
            if speculative {
                self.counters
                    .speculative_ram_to_vram_installs
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(residency)
    }

    fn record_speculative_drop(&self) {
        self.counters
            .speculative_dropped_capacity_or_pressure
            .fetch_add(1, Ordering::Relaxed);
    }
}

fn oldest_unprotected<T>(cache: &LruCache<u32, T>, protected: &HashSet<u32>) -> Option<u32> {
    cache
        .iter()
        .rev()
        .find_map(|(&global_id, _)| (!protected.contains(&global_id)).then_some(global_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;
    use crate::expert_cache::GpuResident;

    fn geometry() -> GpuNativeQ4ExpertGeometry {
        GpuNativeQ4ExpertGeometry::try_new(32, 32, 128, 2).unwrap()
    }

    fn limits() -> wgpu::Limits {
        wgpu::Limits {
            max_push_constant_size: 32,
            max_storage_buffers_per_shader_stage: 8,
            max_compute_workgroup_size_x: 64,
            max_compute_invocations_per_workgroup: 64,
            ..wgpu::Limits::default()
        }
    }

    #[test]
    fn global_layer_local_identity_is_checked_without_clamping() {
        assert_eq!(
            global_to_layer_local(0, 2, 128).unwrap(),
            GpuNativeLayerExpertId {
                layer_index: 0,
                local_expert_id: 0,
            }
        );
        assert_eq!(global_to_layer_local(127, 2, 128).unwrap().layer_index, 0);
        assert_eq!(
            global_to_layer_local(128, 2, 128).unwrap(),
            GpuNativeLayerExpertId {
                layer_index: 1,
                local_expert_id: 0,
            }
        );
        assert_eq!(layer_local_to_global(1, 127, 2, 128).unwrap(), 255);
        assert!(matches!(
            global_to_layer_local(256, 2, 128),
            Err(GpuNativeTieredResidencyError::GlobalExpertOutOfRange { .. })
        ));
        assert!(layer_local_to_global(2, 0, 2, 128).is_err());
        assert!(layer_local_to_global(1, 128, 2, 128).is_err());
        let generation = 0xfedc_ba98_7654_3210;
        let key = global_to_q4_expert_key(129, generation, 2, 128).unwrap();
        assert_eq!(key.layer_index(), 1);
        assert_eq!(key.expert_id(), 1);
        assert_eq!(key.logical_generation(), generation);
    }

    #[test]
    fn logical_admission_generation_reaches_physical_key_unchanged() {
        let cache = GpuExpertCache::new(8, 0.0, 0);
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(7, vec![1; 8])))
            .unwrap();
        let first = cache.current_admission(7).unwrap();
        let first_key = global_to_q4_expert_key(7, first.generation(), 2, 128).unwrap();
        assert_eq!(first_key.logical_generation(), first.generation());
        assert!(cache.contains_generation(7, first_key.logical_generation()));

        // Fill the one-entry LRU with another identity, then readmit the same
        // global expert. Only GpuExpertCache advances the logical generation;
        // the physical key is a lossless consumer of that identity.
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(8, vec![2; 8])))
            .unwrap();
        assert!(!cache.contains_generation(7, first.generation()));
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(7, vec![3; 8])))
            .unwrap();
        let newer = cache.current_admission(7).unwrap();
        assert!(newer.generation() > first.generation());
        let newer_key = global_to_q4_expert_key(7, newer.generation(), 2, 128).unwrap();
        assert_eq!(newer_key.logical_generation(), newer.generation());
        assert!(!cache.contains_generation(7, first_key.logical_generation()));
        assert!(cache.contains_generation(7, newer_key.logical_generation()));
    }

    #[test]
    fn physical_install_source_requires_current_matching_logical_admission() {
        let cache = GpuExpertCache::new(8, 0.0, 0);
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(7, vec![1; 8])))
            .unwrap();
        let admission = cache.current_admission(7).unwrap();
        let pool = BufferPool::new(2, 8, 4);
        let resident = Arc::new(ExpertResident::new(7, pool.try_acquire().unwrap()));
        assert_eq!(
            validate_physical_install_source(&cache, 7, &resident, &admission),
            Ok(())
        );

        let wrong_resident = Arc::new(ExpertResident::new(8, pool.try_acquire().unwrap()));
        assert_eq!(
            validate_physical_install_source(&cache, 7, &wrong_resident, &admission),
            Err(GpuNativeTieredResidencyError::DemandSourceIdentityMismatch { global_id: 7 })
        );

        cache
            .demand_admit_lru(Arc::new(GpuResident::new(8, vec![2; 8])))
            .unwrap();
        assert_eq!(
            validate_physical_install_source(&cache, 7, &resident, &admission),
            Err(GpuNativeTieredResidencyError::LogicalAdmissionStale {
                global_id: 7,
                generation: admission.generation(),
            })
        );
    }

    #[test]
    fn pre_fix_foreground_admission_self_evicts_a_current_selected_source() {
        let cache = GpuExpertCache::new(16, 0.0, 0);
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(7, vec![1; 8])))
            .unwrap();
        let selected_a_generation = cache.current_generation(7).unwrap();
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(99, vec![9; 8])))
            .unwrap();

        let selected_a = GpuNativeDemandExpert::current(7);
        assert!(cache.contains_generation(7, selected_a_generation));

        // This is the old engine ordering: A was classified as Current, then
        // selected B was admitted independently. A is the logical LRU even
        // though non-selected 99 is an eligible victim and A+B fit together.
        cache
            .demand_admit_lru(Arc::new(GpuResident::new(8, vec![2; 8])))
            .unwrap();
        assert!(!cache.contains_generation(7, selected_a_generation));
        assert!(cache.contains(99));
        assert!(cache.contains(8));

        let final_resolution = if cache.contains_generation(7, selected_a_generation) {
            Ok(())
        } else {
            match selected_a {
                GpuNativeDemandExpert::Current { global_id } => {
                    Err(GpuNativeTieredResidencyError::DemandSourceMissing { global_id })
                }
                GpuNativeDemandExpert::Install { .. } => unreachable!(),
            }
        };
        assert_eq!(
            final_resolution,
            Err(GpuNativeTieredResidencyError::DemandSourceMissing { global_id: 7 })
        );
    }

    #[test]
    fn model_plan_requires_top_k_slots_per_layer_and_counts_every_arena() {
        let limits = limits();
        let geometry = geometry();
        let layer_min =
            GpuNativeQ4ExpertVramPlan::try_for_slot_capacity(geometry, geometry.top_k(), &limits)
                .unwrap();
        let exact = layer_min.total_arena_allocation_bytes() * 2;
        assert!(matches!(
            GpuNativeModelExpertVramPlan::try_new(2, geometry, exact - 1, &limits),
            Err(GpuNativeTieredResidencyError::ModelBudgetTooSmall { .. })
        ));
        let plan = GpuNativeModelExpertVramPlan::try_new(2, geometry, exact, &limits).unwrap();
        assert_eq!(plan.minimum_executable_budget_bytes(), exact);
        assert_eq!(plan.total_arena_allocation_bytes(), exact);
        assert_eq!(plan.layer_plans().len(), 2);
        assert!(plan
            .layer_plans()
            .iter()
            .all(|layer| layer.slot_capacity() == geometry.top_k()));
        assert!(plan.layer_plans().iter().all(|layer| {
            layer.total_arena_allocation_bytes()
                >= layer.physical_bank_allocation_bytes() + layer.mapping_metadata_bytes()
        }));
        assert!(plan.layer_plans().iter().all(|layer| {
            layer.physical_bank_allocation_bytes() - layer.active_bank_allocation_bytes()
                == 3 * std::mem::size_of::<u32>() as u64
        }));
        assert!(plan
            .layer_plans()
            .iter()
            .all(|layer| layer.total_arena_allocation_bytes() < exact));
    }

    #[test]
    fn model_plan_is_deterministic_monotonic_bounded_and_saturating() {
        let limits = limits();
        let geometry = geometry();
        let minimum = GpuNativeModelExpertVramPlan::try_new(
            3,
            geometry,
            GpuNativeQ4ExpertVramPlan::try_for_slot_capacity(geometry, geometry.top_k(), &limits)
                .unwrap()
                .total_arena_allocation_bytes()
                * 3,
            &limits,
        )
        .unwrap();
        let extra = geometry.slot_stride_bytes() as u64 * 7;
        let larger = GpuNativeModelExpertVramPlan::try_new(
            3,
            geometry,
            minimum.total_expert_budget_bytes() + extra,
            &limits,
        )
        .unwrap();
        let repeat = GpuNativeModelExpertVramPlan::try_new(
            3,
            geometry,
            minimum.total_expert_budget_bytes() + extra,
            &limits,
        )
        .unwrap();
        assert_eq!(larger, repeat);
        assert!(larger.total_arena_allocation_bytes() <= larger.total_expert_budget_bytes());
        for (small, large) in minimum.layer_plans().iter().zip(larger.layer_plans()) {
            assert!(large.slot_capacity() >= small.slot_capacity());
            assert!(large.slot_capacity() <= geometry.num_experts());
        }
        for layer in larger.layer_plans() {
            if layer.slot_capacity() < geometry.num_experts() {
                let next = GpuNativeQ4ExpertVramPlan::try_for_slot_capacity(
                    geometry,
                    layer.slot_capacity() + 1,
                    &limits,
                )
                .unwrap();
                let increment =
                    next.total_arena_allocation_bytes() - layer.total_arena_allocation_bytes();
                assert!(increment > larger.unused_remainder_bytes());
            }
        }
    }

    #[test]
    fn protected_demand_members_are_never_lru_victims() {
        let mut lru = LruCache::unbounded();
        for id in 0..8 {
            lru.put(id, ());
        }
        let protected = HashSet::from([0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(oldest_unprotected(&lru, &protected), Some(7));
        let all = (0..8).collect::<HashSet<_>>();
        assert_eq!(oldest_unprotected(&lru, &all), None);

        let mut other_layer = LruCache::unbounded();
        other_layer.put(128, ());
        other_layer.put(129, ());
        let other_before = other_layer.iter().map(|(&id, _)| id).collect::<Vec<_>>();
        let _ = oldest_unprotected(&lru, &protected);
        assert_eq!(
            other_layer.iter().map(|(&id, _)| id).collect::<Vec<_>>(),
            other_before
        );
    }

    #[test]
    fn normal_physical_demand_hit_promotes_lru_recency() {
        let mut residents = LruCache::unbounded();
        residents.put(7, 70);
        residents.put(8, 80);
        assert_eq!(oldest_unprotected(&residents, &HashSet::new()), Some(7));

        assert_eq!(touch_physical_record(&mut residents, 7), Some(70));
        assert_eq!(oldest_unprotected(&residents, &HashSet::new()), Some(8));
        assert_eq!(residents.peek(&7), Some(&70));
        assert_eq!(residents.peek(&8), Some(&80));
    }

    #[test]
    fn shadow_metadata_iteration_does_not_touch_physical_lru() {
        let mut residents = LruCache::unbounded();
        for id in 128..136 {
            residents.put(id, id * 10);
        }
        let before = residents
            .iter()
            .map(|(&global_id, _)| global_id)
            .collect::<Vec<_>>();

        let copied_mru_to_lru = residents
            .iter()
            .map(|(&global_id, _)| global_id)
            .collect::<Vec<_>>();

        assert_eq!(copied_mru_to_lru, before);
        assert_eq!(
            residents
                .iter()
                .map(|(&global_id, _)| global_id)
                .collect::<Vec<_>>(),
            before
        );
        assert_eq!(oldest_unprotected(&residents, &HashSet::new()), Some(128));
    }
}
