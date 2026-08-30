//! PR1C-B shadow-only comparison of causal GPU-native residency predictors.
//!
//! Every predictor in this module owns private CPU state. The observer reads
//! the ordinary compact boundary report and production residency metadata, but
//! it never performs storage I/O, logical admission, physical installation,
//! eviction, or active prefetch.

use crate::gpu_native_prefetch_shadow::{
    GpuNativePrefetchShadowCallbacks, ObserverRuntimeGuardEvidence, ShadowObserverCallbackKind,
    ShadowObserverError, ShadowPhase,
};
use crate::gpu_native_residency::{
    GpuNativePhysicalLayerShadowSnapshot, GpuNativeTieredResidencyError,
    GpuNativeTieredResidencyManager,
};
use crate::pregate::PerLayerPreGate;
use crate::router::{
    spatial_neighbors, LayeredExpertAffinity, LocalityMonitor, PredictiveLoader,
    SPATIAL_CONFIDENCE_THRESHOLD, W_AFFINITY, W_SPATIAL,
};
use parking_lot::Mutex;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

pub(crate) const SCHEMA: &str = "mer.gpu-native-prefetch-multipredictor-shadow.v2";
pub(crate) const MODE: &str = "gpu-native-prefetch-multipredictor-shadow";
pub(crate) const SOURCE_MAIN_COMMIT: &str = "e8b542110693e74aa8f1013bb16d1bed0bdd8ba7";
pub(crate) const TESTED_PR1BB_COMMIT: &str = "c7006c5c6fbcee74c91526f89a8c2b5b06d8a9c5";
pub(crate) const TESTED_PR1BB_REPORT_SHA256: &str =
    "82317ad85ba29da1401abf3041ebbb7c2ca72c039c1be0c5f2ea53df9e9fc529";
pub(crate) const PR1CA_COMMIT: &str = "5d1e16ad6f2d8c71809a8fb65b1fbb3e4c98972f";
pub(crate) const PR1CA_REPORT_SHA256: &str =
    "cd7d86d9c9ff6691c11320db9d5e44dc26d893a6e14a939a31ed6ab67b270830";
pub(crate) const PR1CA_LOG_SHA256: &str =
    "de20495fdf4cf47db0264e4ad547aa17ee83349dceab7c89f2438487c535c727";

const CONTROL: &str = "per-layer-pregate-transition";
const MARKOV1: &str = "predictive-loader-first-order";
const MARKOV2: &str = "predictive-loader-second-order";
const LOCALITY: &str = "locality-hot-set";
const REDUCED_UNIFIED: &str = "reduced-unified-markov2-locality";
const REDUCED_SPATIAL_AFFINITY: &str = "reduced-unified-markov2-locality-spatial-affinity";
const MAX_FANOUT: usize = 16;
const BASE_FANOUTS: [usize; 4] = [1, 2, 4, 8];
const OPTIONAL_FANOUTS: [usize; 2] = [12, 16];
const FROZEN_ADAPTER_NAME: &str = "NVIDIA L4";

#[derive(Clone, Debug)]
pub(crate) struct MultipredictorObserverConfig {
    pub(crate) num_layers: usize,
    pub(crate) experts_per_layer: usize,
    pub(crate) top_k: usize,
    pub(crate) markov_min_prob: f64,
    pub(crate) locality_window: usize,
    pub(crate) locality_threshold_pct: f32,
    pub(crate) affinity_neighbors_k: usize,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PredictorCandidateAudit {
    pub(crate) source: &'static str,
    pub(crate) source_locations: Vec<&'static str>,
    pub(crate) eligibility: &'static str,
    pub(crate) implementation: &'static str,
    pub(crate) exact_state_consumed: &'static str,
    pub(crate) learning_state_mutated: &'static str,
    pub(crate) id_namespace: &'static str,
    pub(crate) temporal_target: &'static str,
    pub(crate) cpu_visible_at_prediction_freeze: bool,
    pub(crate) unavailable_or_prohibited_requirements: Vec<&'static str>,
    pub(crate) private_shadow_state: bool,
    pub(crate) target_leakage: bool,
    pub(crate) disposition: &'static str,
}

pub(crate) fn predictor_candidate_audit() -> Vec<PredictorCandidateAudit> {
    vec![
        PredictorCandidateAudit {
            source: CONTROL,
            source_locations: vec![
                "rust-engine/src/pregate.rs::PerLayerPreGate::{predict_ranked,observe_transition}",
            ],
            eligibility: "eligible",
            implementation: "exact existing PerLayerPreGate algorithm in private state",
            exact_state_consumed: "completed target-predecessor layer L-1 local top-k plus private transition counts learned from earlier completed L-1 -> L pairs",
            learning_state_mutated: "private per-source-layer local-ID transition counts, after target scoring only",
            id_namespace: "layer-local predictions mapped to target-layer global IDs for scoring",
            temporal_target: "next MoE layer in the same token position",
            cpu_visible_at_prediction_freeze: true,
            unavailable_or_prohibited_requirements: vec![],
            private_shadow_state: true,
            target_leakage: false,
            disposition: "evaluated as the PR1C-A control",
        },
        PredictorCandidateAudit {
            source: MARKOV1,
            source_locations: vec![
                "rust-engine/src/router.rs::PredictiveLoader::{predict_next,observe_step2}",
                "rust-engine/src/router.rs::PredictiveLoader::predict_next_shadow_ranked",
            ],
            eligibility: "eligible",
            implementation: "thin deterministic shadow adapter over exact PredictiveLoader first-order smoothing/count semantics",
            exact_state_consumed: "last-ranked global expert ID from layer L-1 plus private first-order transition rows learned earlier",
            learning_state_mutated: "private PredictiveLoader first-order rows through observe_step2, after target scoring only",
            id_namespace: "global layer-qualified expert IDs",
            temporal_target: "successor route set for the next contiguous MoE layer",
            cpu_visible_at_prediction_freeze: true,
            unavailable_or_prohibited_requirements: vec![],
            private_shadow_state: true,
            target_leakage: false,
            disposition: "evaluated without filtering predictions to the target layer",
        },
        PredictorCandidateAudit {
            source: MARKOV2,
            source_locations: vec![
                "rust-engine/src/router.rs::PredictiveLoader::{predict_next2,observe_step2}",
                "rust-engine/src/router.rs::PredictiveLoader::predict_next2_shadow_ranked",
            ],
            eligibility: "eligible",
            implementation: "thin deterministic shadow adapter over exact PredictiveLoader 50/50 first/second-order blend and fallback",
            exact_state_consumed: "last-ranked global IDs from layers L-2 and L-1 plus private first/second-order rows learned earlier",
            learning_state_mutated: "private PredictiveLoader first/second-order rows through observe_step2, after target scoring only",
            id_namespace: "global layer-qualified expert IDs",
            temporal_target: "successor route set for the next contiguous MoE layer",
            cpu_visible_at_prediction_freeze: true,
            unavailable_or_prohibited_requirements: vec![],
            private_shadow_state: true,
            target_leakage: false,
            disposition: "evaluated; falls back to first-order when no causal L-2 row exists",
        },
        PredictorCandidateAudit {
            source: LOCALITY,
            source_locations: vec![
                "rust-engine/src/router.rs::LocalityMonitor::{hot_set,observe}",
            ],
            eligibility: "eligible",
            implementation: "exact existing LocalityMonitor sliding-window hot_set algorithm in private state",
            exact_state_consumed: "global routed expert activations observed through layer L-1 in the configured layer-scaled window",
            learning_state_mutated: "private sliding window and heat counts, after each route is scored",
            id_namespace: "global layer-qualified expert IDs",
            temporal_target: "target-independent recent-hot ranking scored against next layer L",
            cpu_visible_at_prediction_freeze: true,
            unavailable_or_prohibited_requirements: vec![],
            private_shadow_state: true,
            target_leakage: false,
            disposition: "evaluated as a standalone deterministic ranking without target-layer filtering",
        },
        PredictorCandidateAudit {
            source: "ExpertAffinity",
            source_locations: vec![
                "rust-engine/src/router.rs::ExpertAffinity::{neighbors,observe_layer}",
            ],
            eligibility: "ineligible-standalone",
            implementation: "existing global coactivation table",
            exact_state_consumed: "a seed expert plus historical same-layer coactivation counts",
            learning_state_mutated: "coactivation counts",
            id_namespace: "flat global IDs without target-layer ownership",
            temporal_target: "co-fired neighbor of a supplied seed, not an independent future-layer predictor",
            cpu_visible_at_prediction_freeze: true,
            unavailable_or_prohibited_requirements: vec!["requires another causal predictor to supply a seed"],
            private_shadow_state: false,
            target_leakage: false,
            disposition: "not scored standalone; global ExpertAffinity would mix layer identities",
        },
        PredictorCandidateAudit {
            source: "LayeredExpertAffinity",
            source_locations: vec![
                "rust-engine/src/router.rs::LayeredExpertAffinity::{neighbors,observe_layer}",
            ],
            eligibility: "eligible-fusion-only",
            implementation: "exact existing per-layer coactivation neighbor algorithm in private state",
            exact_state_consumed: "causal fusion seeds plus target-layer-qualified historical coactivation counts learned from earlier completed targets",
            learning_state_mutated: "private target-layer coactivation counts, after target scoring only",
            id_namespace: "layer-local matrix with explicit global mapping",
            temporal_target: "historical same-target-layer neighbor expansion of another causal signal",
            cpu_visible_at_prediction_freeze: true,
            unavailable_or_prohibited_requirements: vec!["cannot rank a future target without causal seeds"],
            private_shadow_state: true,
            target_leakage: false,
            disposition: "evaluated only in the reduced spatial-affinity fusion",
        },
        PredictorCandidateAudit {
            source: "predict_unified",
            source_locations: vec![
                "rust-engine/src/router.rs::PredictiveLoader::{predict_unified,combine_unified_arms}",
                "rust-engine/src/engine.rs::union_prefetch",
            ],
            eligibility: "ineligible-as-full-production-unified",
            implementation: "existing weighted Markov/locality/neural fusion",
            exact_state_consumed: "Markov row, locality hot set, and neural logits from a CPU hidden-state feature",
            learning_state_mutated: "private Markov/locality state would be legal; neural training would require the unavailable hidden feature",
            id_namespace: "global layer-qualified expert IDs",
            temporal_target: "next speculative route candidates",
            cpu_visible_at_prediction_freeze: false,
            unavailable_or_prohibited_requirements: vec!["GPU-native hidden state is not CPU-visible", "a new GPU-to-CPU readback is prohibited"],
            private_shadow_state: false,
            target_leakage: false,
            disposition: "full production unified is not claimed; an explicitly named reduced Markov2+locality fusion is evaluated",
        },
        PredictorCandidateAudit {
            source: "predict_unified_with_spatial",
            source_locations: vec![
                "rust-engine/src/router.rs::PredictiveLoader::{predict_unified_with_spatial,fold_spatial_affinity}",
                "rust-engine/src/engine.rs::{union_prefetch,fold_affinity_spatial}",
            ],
            eligibility: "ineligible-as-full-production-unified",
            implementation: "existing unified fusion plus spatial and affinity expansion",
            exact_state_consumed: "full unified inputs plus affinity seed expansion",
            learning_state_mutated: "same state as full unified plus affinity counts",
            id_namespace: "global candidates with layer-qualified affinity mapping in the engine adapter",
            temporal_target: "next speculative route candidates and their neighbors",
            cpu_visible_at_prediction_freeze: false,
            unavailable_or_prohibited_requirements: vec!["full unified neural hidden-state arm is unavailable", "a new GPU-to-CPU readback is prohibited"],
            private_shadow_state: false,
            target_leakage: false,
            disposition: "full production spatial unified is not claimed; an explicitly named reduced fusion is evaluated",
        },
        PredictorCandidateAudit {
            source: "NeuralSpeculator",
            source_locations: vec![
                "rust-engine/src/router.rs::NeuralSpeculator::{predict_topk,train_step}",
            ],
            eligibility: "ineligible",
            implementation: "existing two-layer online MLP",
            exact_state_consumed: "target residual hidden vector with d_model floats",
            learning_state_mutated: "MLP weights and train-step counter",
            id_namespace: "global layer-qualified expert logits",
            temporal_target: "gate top-k for the hidden vector supplied to the MLP",
            cpu_visible_at_prediction_freeze: false,
            unavailable_or_prohibited_requirements: vec!["the required GPU-native hidden vector is device-resident at prediction freeze", "obtaining it would add a GPU-to-CPU readback and change production execution"],
            private_shadow_state: false,
            target_leakage: false,
            disposition: "refused; no neural prediction or training is performed",
        },
        PredictorCandidateAudit {
            source: REDUCED_UNIFIED,
            source_locations: vec![
                "rust-engine/src/gpu_native_prefetch_multipredictor_shadow.rs::freeze_predictions",
                "rust-engine/src/router.rs::PredictiveLoader::combine_unified_arms",
            ],
            eligibility: "eligible",
            implementation: "thin adapter using exact combine_unified_arms weights with the neural arm explicitly absent",
            exact_state_consumed: "frozen second-order Markov ranking and frozen locality ranking",
            learning_state_mutated: "none during fusion; component private state updates after target scoring",
            id_namespace: "global layer-qualified expert IDs",
            temporal_target: "next contiguous MoE layer",
            cpu_visible_at_prediction_freeze: true,
            unavailable_or_prohibited_requirements: vec![],
            private_shadow_state: true,
            target_leakage: false,
            disposition: "evaluated and explicitly not represented as full production unified",
        },
        PredictorCandidateAudit {
            source: REDUCED_SPATIAL_AFFINITY,
            source_locations: vec![
                "rust-engine/src/gpu_native_prefetch_multipredictor_shadow.rs::freeze_predictions",
                "rust-engine/src/router.rs::{spatial_neighbors,SPATIAL_CONFIDENCE_THRESHOLD,W_AFFINITY,W_SPATIAL}",
                "rust-engine/src/router.rs::LayeredExpertAffinity::neighbors",
            ],
            eligibility: "eligible",
            implementation: "thin global-ID adapter preserving canonical unified, spatial-threshold, spatial-neighbor, and affinity weights",
            exact_state_consumed: "reduced unified candidates plus private historical layer-qualified affinity",
            learning_state_mutated: "none during fusion; affinity updates after target scoring",
            id_namespace: "global candidates with layer-qualified local affinity lookup",
            temporal_target: "next contiguous MoE layer plus neighbors of causally predicted high-confidence seeds",
            cpu_visible_at_prediction_freeze: true,
            unavailable_or_prohibited_requirements: vec![],
            private_shadow_state: true,
            target_leakage: false,
            disposition: "evaluated; with the neural arm absent, canonical 0.80 seed threshold may make the neighbor fold a structural no-op",
        },
    ]
}

#[derive(Clone, Debug)]
struct RankedPrediction {
    global_id: u32,
    score: f64,
}

#[derive(Clone, Debug)]
struct FrozenPredictor {
    source: &'static str,
    score_semantics: &'static str,
    ranked: Vec<RankedPrediction>,
    duplicate_prediction_count: u64,
}

#[derive(Clone, Debug)]
struct PendingTarget {
    phase: ShadowPhase,
    run_index: usize,
    completed_token_position: usize,
    target_layer: usize,
    predictors: Vec<FrozenPredictor>,
    physical: GpuNativePhysicalLayerShadowSnapshot,
    frozen_sequence: u64,
    frozen_at_us: u64,
    target_probe_sequence: Option<u64>,
    target_probe_at_us: Option<u64>,
}

#[derive(Clone, Debug)]
struct RouteHistory {
    completed_token_position: usize,
    layer: usize,
    local_ids: Vec<u32>,
    global_ids: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
struct ActiveRun {
    phase: ShadowPhase,
    run_index: usize,
}

struct PrivatePredictors {
    pregate: PerLayerPreGate,
    markov: PredictiveLoader,
    locality: LocalityMonitor,
    affinity: LayeredExpertAffinity,
}

struct ObserverInner {
    origin: Instant,
    sequence: u64,
    active_run: Option<ActiveRun>,
    predictors: PrivatePredictors,
    last_last_route: Option<RouteHistory>,
    last_route: Option<RouteHistory>,
    pending: Option<PendingTarget>,
    events: Vec<MultipredictorPredictionEvent>,
    runtime_guard_evidence: ObserverRuntimeGuardEvidence,
}

impl ObserverInner {
    fn next_sequence(&mut self) -> u64 {
        self.sequence = self.sequence.saturating_add(1);
        self.sequence
    }

    fn monotonic_us(&self) -> u64 {
        self.origin.elapsed().as_micros().min(u64::MAX as u128) as u64
    }
}

pub(crate) struct GpuNativeMultipredictorShadowObserver {
    config: MultipredictorObserverConfig,
    inner: Mutex<ObserverInner>,
}

impl GpuNativeMultipredictorShadowObserver {
    pub(crate) fn new(
        config: MultipredictorObserverConfig,
    ) -> Result<Arc<Self>, ShadowObserverError> {
        if config.num_layers == 0
            || config.experts_per_layer == 0
            || config.top_k == 0
            || config.top_k > config.experts_per_layer
            || config.locality_window == 0
            || !(config.locality_threshold_pct > 0.0 && config.locality_threshold_pct <= 1.0)
            || config.affinity_neighbors_k == 0
        {
            return Err(ShadowObserverError::new(format!(
                "invalid multipredictor shadow observer geometry/config: {config:?}"
            )));
        }
        let total_experts = config
            .num_layers
            .checked_mul(config.experts_per_layer)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| ShadowObserverError::new("multipredictor expert namespace overflow"))?;
        let window = config
            .locality_window
            .checked_mul(config.num_layers)
            .ok_or_else(|| ShadowObserverError::new("multipredictor locality window overflow"))?;
        Ok(Arc::new(Self {
            inner: Mutex::new(ObserverInner {
                origin: Instant::now(),
                sequence: 0,
                active_run: None,
                predictors: PrivatePredictors {
                    pregate: PerLayerPreGate::new(config.num_layers, MAX_FANOUT),
                    markov: PredictiveLoader::new(
                        total_experts,
                        MAX_FANOUT,
                        config.markov_min_prob,
                        0xC0FFEE,
                    ),
                    locality: LocalityMonitor::new(total_experts, window),
                    affinity: LayeredExpertAffinity::new(
                        config.num_layers,
                        config.experts_per_layer as u32,
                    ),
                },
                last_last_route: None,
                last_route: None,
                pending: None,
                events: Vec::new(),
                runtime_guard_evidence: ObserverRuntimeGuardEvidence::default(),
            }),
            config,
        }))
    }

    pub(crate) fn begin_run(
        &self,
        phase: ShadowPhase,
        run_index: usize,
    ) -> Result<(), ShadowObserverError> {
        let mut inner = self.inner.lock();
        if inner.active_run.is_some() {
            return Err(ShadowObserverError::new(
                "multipredictor shadow begin_run called while a run is active",
            ));
        }
        inner.active_run = Some(ActiveRun { phase, run_index });
        inner.last_last_route = None;
        inner.last_route = None;
        inner.pending = None;
        Ok(())
    }

    pub(crate) fn end_run(&self) -> Result<(), ShadowObserverError> {
        let mut inner = self.inner.lock();
        if inner.active_run.take().is_none() {
            return Err(ShadowObserverError::new(
                "multipredictor shadow end_run called without an active run",
            ));
        }
        inner.last_last_route = None;
        inner.last_route = None;
        inner.pending = None;
        Ok(())
    }

    fn before_segment_inner(
        &self,
        completed_token_position: usize,
        first_new_layer: usize,
    ) -> Result<(), ShadowObserverError> {
        let mut inner = self.inner.lock();
        if inner.active_run.is_none() {
            return Ok(());
        }
        let now = inner.monotonic_us();
        let sequence = inner.next_sequence();
        let Some(pending) = inner.pending.as_mut() else {
            return Ok(());
        };
        if pending.completed_token_position != completed_token_position
            || pending.target_layer != first_new_layer
        {
            return Err(ShadowObserverError::new(format!(
                "pending multipredictor target run={} position={} layer={} did not match segment position={} first_new_layer={}",
                pending.run_index,
                pending.completed_token_position,
                pending.target_layer,
                completed_token_position,
                first_new_layer,
            )));
        }
        if pending.target_probe_sequence.is_some() {
            return Err(ShadowObserverError::new(
                "multipredictor target physical probe was marked more than once",
            ));
        }
        pending.target_probe_sequence = Some(sequence);
        pending.target_probe_at_us = Some(now.max(pending.frozen_at_us));
        Ok(())
    }

    fn observe_boundary_inner(
        &self,
        completed_token_position: usize,
        observed_layers: std::ops::RangeInclusive<usize>,
        selected_ids_by_layer: &[Vec<u32>],
        residency: &GpuNativeTieredResidencyManager,
    ) -> Result<(), ShadowObserverError> {
        let mut inner = self.inner.lock();
        let active = match inner.active_run {
            Some(active) => active,
            None => return Ok(()),
        };
        let start = *observed_layers.start();
        let end = *observed_layers.end();
        if start > end || end >= self.config.num_layers || end >= selected_ids_by_layer.len() {
            return Err(ShadowObserverError::new(format!(
                "invalid multipredictor observed layer range {start}..={end}"
            )));
        }
        if inner.pending.as_ref().is_some_and(|pending| {
            pending.completed_token_position == completed_token_position
                && pending.target_layer < start
        }) {
            return Err(ShadowObserverError::new(
                "multipredictor prediction target was skipped before scoring",
            ));
        }

        for layer in start..=end {
            let local_ids = selected_ids_by_layer.get(layer).ok_or_else(|| {
                ShadowObserverError::new("missing selected route for multipredictor layer")
            })?;
            validate_route(local_ids, self.config.top_k, self.config.experts_per_layer)?;

            if inner.pending.as_ref().is_some_and(|pending| {
                pending.completed_token_position == completed_token_position
                    && pending.target_layer == layer
            }) {
                let pending = inner.pending.take().expect("pending target checked");
                let target_physical = residency.shadow_layer_snapshot(layer)?;
                let actual_physical = classify_authoritative_actual_selected(
                    residency,
                    layer,
                    local_ids,
                    self.config.experts_per_layer,
                )?;
                let scored_sequence = inner.next_sequence();
                let updated_sequence = scored_sequence.saturating_add(1);
                let mut events = score_pending(
                    pending,
                    actual_physical,
                    target_physical,
                    scored_sequence,
                    updated_sequence,
                )?;
                inner.events.append(&mut events);
            }

            let global_ids = local_ids
                .iter()
                .map(|&local_id| global_id(layer, local_id, self.config.experts_per_layer))
                .collect::<Result<Vec<_>, _>>()?;
            let previous = inner.last_route.clone().filter(|route| {
                route.completed_token_position == completed_token_position
                    && route.layer + 1 == layer
            });
            let previous_previous = inner.last_last_route.clone().filter(|route| {
                previous.as_ref().is_some_and(|previous| {
                    route.completed_token_position == completed_token_position
                        && route.layer + 1 == previous.layer
                })
            });

            if let Some(previous) = previous.as_ref() {
                inner.predictors.pregate.observe_transition(
                    previous.layer as u32,
                    &previous.local_ids,
                    local_ids,
                );
                inner.predictors.markov.observe_step2(
                    previous_previous
                        .as_ref()
                        .map(|route| route.global_ids.as_slice())
                        .unwrap_or(&[]),
                    &previous.global_ids,
                    &global_ids,
                );
            }
            inner.predictors.affinity.observe_layer(layer, local_ids);
            inner.predictors.locality.observe(&global_ids);
            let _updated_sequence = inner.next_sequence();

            inner.last_last_route = inner.last_route.take();
            inner.last_route = Some(RouteHistory {
                completed_token_position,
                layer,
                local_ids: local_ids.clone(),
                global_ids,
            });
        }

        if end + 1 < self.config.num_layers {
            if inner.pending.is_some() {
                return Err(ShadowObserverError::new(
                    "multipredictor attempted to overwrite an unscored target",
                ));
            }
            let source = inner.last_route.clone().ok_or_else(|| {
                ShadowObserverError::new("multipredictor lost the last observed route")
            })?;
            let source_previous = inner.last_last_route.clone();
            let physical = residency.shadow_layer_snapshot(end + 1)?;
            inner.pending = Some(freeze_predictions(
                &mut inner,
                &self.config,
                active,
                completed_token_position,
                end + 1,
                &source,
                source_previous.as_ref(),
                physical,
            )?);
        }
        Ok(())
    }

    pub(crate) fn snapshot(&self) -> MultipredictorObserverSnapshot {
        MultipredictorObserverSnapshot::from_events(self.inner.lock().events.clone())
    }

    pub(crate) fn runtime_guard_evidence(&self) -> ObserverRuntimeGuardEvidence {
        self.inner.lock().runtime_guard_evidence
    }

    #[cfg(test)]
    fn predictor_observation_counts(&self) -> (u64, u64, u64) {
        let inner = self.inner.lock();
        (
            inner.predictors.markov.observations(),
            inner.predictors.locality.len() as u64,
            inner.predictors.affinity.total_observations(),
        )
    }
}

impl GpuNativePrefetchShadowCallbacks for GpuNativeMultipredictorShadowObserver {
    fn before_segment(
        &self,
        completed_token_position: usize,
        first_new_layer: usize,
    ) -> Result<(), ShadowObserverError> {
        self.before_segment_inner(completed_token_position, first_new_layer)
    }

    fn observe_boundary(
        &self,
        completed_token_position: usize,
        observed_layers: std::ops::RangeInclusive<usize>,
        selected_ids_by_layer: &[Vec<u32>],
        residency: &GpuNativeTieredResidencyManager,
    ) -> Result<(), ShadowObserverError> {
        self.observe_boundary_inner(
            completed_token_position,
            observed_layers,
            selected_ids_by_layer,
            residency,
        )
    }

    fn record_runtime_guarded_callback(&self, kind: ShadowObserverCallbackKind) {
        let mut inner = self.inner.lock();
        let evidence = &mut inner.runtime_guard_evidence;
        evidence.checked_callback_count = evidence.checked_callback_count.saturating_add(1);
        match kind {
            ShadowObserverCallbackKind::BeforeSegment => {
                evidence.before_segment_callback_count =
                    evidence.before_segment_callback_count.saturating_add(1);
            }
            ShadowObserverCallbackKind::ObserveBoundary => {
                evidence.observe_boundary_callback_count =
                    evidence.observe_boundary_callback_count.saturating_add(1);
            }
        }
        evidence.verified = true;
    }
}

fn freeze_predictions(
    inner: &mut ObserverInner,
    config: &MultipredictorObserverConfig,
    active: ActiveRun,
    completed_token_position: usize,
    target_layer: usize,
    source: &RouteHistory,
    source_previous: Option<&RouteHistory>,
    physical: GpuNativePhysicalLayerShadowSnapshot,
) -> Result<PendingTarget, ShadowObserverError> {
    if source.completed_token_position != completed_token_position
        || source.layer + 1 != target_layer
    {
        return Err(ShadowObserverError::new(
            "multipredictor source was not the target layer's causal predecessor",
        ));
    }

    let pregate = inner
        .predictors
        .pregate
        .predict_ranked(source.layer as u32, &source.local_ids, MAX_FANOUT)
        .into_iter()
        .map(|candidate| {
            Ok(RankedPrediction {
                global_id: global_id(target_layer, candidate.expert_id, config.experts_per_layer)?,
                score: candidate.score as f64,
            })
        })
        .collect::<Result<Vec<_>, ShadowObserverError>>()?;
    let seed = *source
        .global_ids
        .last()
        .ok_or_else(|| ShadowObserverError::new("multipredictor Markov source route was empty"))?;
    let markov1 = inner
        .predictors
        .markov
        .predict_next_shadow_ranked(seed, MAX_FANOUT)
        .into_iter()
        .map(|(global_id, score)| RankedPrediction { global_id, score })
        .collect::<Vec<_>>();
    let markov2 = match source_previous
        .filter(|previous| {
            previous.completed_token_position == completed_token_position
                && previous.layer + 1 == source.layer
        })
        .and_then(|previous| previous.global_ids.last().copied())
    {
        Some(prev_prev) => inner
            .predictors
            .markov
            .predict_next2_shadow_ranked(prev_prev, seed, MAX_FANOUT),
        None => inner
            .predictors
            .markov
            .predict_next_shadow_ranked(seed, MAX_FANOUT),
    }
    .into_iter()
    .map(|(global_id, score)| RankedPrediction { global_id, score })
    .collect::<Vec<_>>();
    let threshold = config.locality_threshold_pct / config.num_layers as f32;
    let locality_all = inner
        .predictors
        .locality
        .hot_set(threshold)
        .into_iter()
        .map(|global_id| RankedPrediction {
            global_id,
            score: inner.predictors.locality.heat(global_id) as f64,
        })
        .collect::<Vec<_>>();
    let locality_ids = locality_all
        .iter()
        .map(|prediction| prediction.global_id)
        .collect::<Vec<_>>();
    let locality = locality_all
        .into_iter()
        .take(MAX_FANOUT)
        .collect::<Vec<_>>();

    let markov2_arm = markov2
        .iter()
        .map(|prediction| (prediction.global_id, prediction.score))
        .collect::<Vec<_>>();
    let mut reduced =
        inner
            .predictors
            .markov
            .combine_unified_arms(&markov2_arm, &locality_ids, &[]);
    reduced.truncate(MAX_FANOUT);
    let reduced = reduced
        .into_iter()
        .map(|(global_id, score)| RankedPrediction {
            global_id,
            score: score as f64,
        })
        .collect::<Vec<_>>();
    let reduced_spatial_affinity =
        fold_reduced_spatial_affinity(&reduced, &inner.predictors.affinity, config);

    let candidates = vec![
        (CONTROL, "summed online transition counts; not a calibrated probability", pregate),
        (MARKOV1, "Laplace-smoothed first-order transition probability", markov1),
        (MARKOV2, "50/50 Laplace-smoothed first/second-order transition blend with first-order fallback", markov2),
        (LOCALITY, "sliding-window heat count with existing descending-heat/ascending-ID tie order", locality),
        (REDUCED_UNIFIED, "canonical 0.33 Markov + 0.25 locality weighted score; neural arm explicitly absent", reduced),
        (REDUCED_SPATIAL_AFFINITY, "reduced unified score plus canonical 0.05 spatial and 0.10 layer-affinity seed expansion at the 0.80 threshold", reduced_spatial_affinity),
    ];
    let predictors = candidates
        .into_iter()
        .map(|(source, score_semantics, raw)| {
            let (ranked, duplicate_prediction_count) = dedupe_ranked(raw);
            FrozenPredictor {
                source,
                score_semantics,
                ranked,
                duplicate_prediction_count,
            }
        })
        .collect();
    let frozen_sequence = inner.next_sequence();
    let frozen_at_us = inner.monotonic_us();
    Ok(PendingTarget {
        phase: active.phase,
        run_index: active.run_index,
        completed_token_position,
        target_layer,
        predictors,
        physical,
        frozen_sequence,
        frozen_at_us,
        target_probe_sequence: None,
        target_probe_at_us: None,
    })
}

fn fold_reduced_spatial_affinity(
    base: &[RankedPrediction],
    affinity: &LayeredExpertAffinity,
    config: &MultipredictorObserverConfig,
) -> Vec<RankedPrediction> {
    let seeds = base
        .iter()
        .filter(|candidate| candidate.score as f32 >= SPATIAL_CONFIDENCE_THRESHOLD)
        .map(|candidate| candidate.global_id)
        .collect::<Vec<_>>();
    let total_experts = config.num_layers.saturating_mul(config.experts_per_layer) as u32;
    let mut combined = base
        .iter()
        .map(|candidate| (candidate.global_id, candidate.score as f32))
        .collect::<HashMap<_, _>>();
    for seed in seeds {
        for neighbor in spatial_neighbors(seed, total_experts, 2) {
            *combined.entry(neighbor).or_insert(0.0) += W_SPATIAL;
        }
        let layer = seed as usize / config.experts_per_layer;
        let local = seed % config.experts_per_layer as u32;
        for local_neighbor in affinity.neighbors(layer, local, config.affinity_neighbors_k) {
            if let Ok(global_neighbor) = global_id(layer, local_neighbor, config.experts_per_layer)
            {
                *combined.entry(global_neighbor).or_insert(0.0) += W_AFFINITY;
            }
        }
    }
    let mut ranked = combined
        .into_iter()
        .filter(|&(_, score)| score > 0.0)
        .map(|(global_id, score)| RankedPrediction {
            global_id,
            score: score as f64,
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.global_id.cmp(&right.global_id))
    });
    ranked.truncate(MAX_FANOUT);
    ranked
}

fn dedupe_ranked(raw: Vec<RankedPrediction>) -> (Vec<RankedPrediction>, u64) {
    let mut seen = HashSet::with_capacity(raw.len());
    let mut duplicate_count = 0u64;
    let ranked = raw
        .into_iter()
        .filter(|candidate| {
            if seen.insert(candidate.global_id) {
                true
            } else {
                duplicate_count = duplicate_count.saturating_add(1);
                false
            }
        })
        .collect();
    (ranked, duplicate_count)
}

fn validate_route(
    local_ids: &[u32],
    top_k: usize,
    experts_per_layer: usize,
) -> Result<(), ShadowObserverError> {
    if local_ids.len() != top_k {
        return Err(ShadowObserverError::new(format!(
            "multipredictor route contained {} experts, expected {top_k}",
            local_ids.len()
        )));
    }
    let mut seen = HashSet::with_capacity(local_ids.len());
    for &local_id in local_ids {
        if local_id as usize >= experts_per_layer {
            return Err(ShadowObserverError::new(format!(
                "multipredictor route expert {local_id} is outside 0..{experts_per_layer}"
            )));
        }
        if !seen.insert(local_id) {
            return Err(ShadowObserverError::new(format!(
                "multipredictor route duplicated expert {local_id}"
            )));
        }
    }
    Ok(())
}

fn global_id(
    layer: usize,
    local_id: u32,
    experts_per_layer: usize,
) -> Result<u32, ShadowObserverError> {
    layer
        .checked_mul(experts_per_layer)
        .and_then(|base| base.checked_add(local_id as usize))
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| ShadowObserverError::new("multipredictor global expert id overflow"))
}

trait AuthoritativePhysicalCurrentness {
    fn authoritative_physical_current(
        &self,
        global_id: u32,
    ) -> Result<bool, GpuNativeTieredResidencyError>;
}

impl AuthoritativePhysicalCurrentness for GpuNativeTieredResidencyManager {
    fn authoritative_physical_current(
        &self,
        global_id: u32,
    ) -> Result<bool, GpuNativeTieredResidencyError> {
        self.has_current_for_demand(global_id)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AuthoritativeActualPhysicalClassification {
    selected_global_ids: Vec<u32>,
    current_selected_global_ids: Vec<u32>,
    missing_selected_global_ids: Vec<u32>,
}

fn classify_authoritative_actual_selected<C: AuthoritativePhysicalCurrentness + ?Sized>(
    currentness: &C,
    target_layer: usize,
    actual_local_ids: &[u32],
    experts_per_layer: usize,
) -> Result<AuthoritativeActualPhysicalClassification, ShadowObserverError> {
    let mut selected_global_ids = Vec::with_capacity(actual_local_ids.len());
    let mut current_selected_global_ids = Vec::with_capacity(actual_local_ids.len());
    let mut missing_selected_global_ids = Vec::with_capacity(actual_local_ids.len());
    for &local_id in actual_local_ids {
        let global_id = global_id(target_layer, local_id, experts_per_layer)?;
        selected_global_ids.push(global_id);
        if currentness.authoritative_physical_current(global_id)? {
            current_selected_global_ids.push(global_id);
        } else {
            missing_selected_global_ids.push(global_id);
        }
    }
    Ok(AuthoritativeActualPhysicalClassification {
        selected_global_ids,
        current_selected_global_ids,
        missing_selected_global_ids,
    })
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct MultipredictorEventOrdering {
    pub(crate) prediction_frozen_sequence: u64,
    pub(crate) target_probe_sequence: u64,
    pub(crate) prediction_scored_sequence: u64,
    pub(crate) predictor_updated_sequence: u64,
    pub(crate) prediction_frozen_at_monotonic_us: u64,
    pub(crate) target_physical_probe_at_monotonic_us: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct MultipredictorPredictionEvent {
    pub(crate) phase: ShadowPhase,
    pub(crate) run_index: usize,
    pub(crate) completed_token_position: usize,
    pub(crate) target_layer: usize,
    pub(crate) predictor_source: &'static str,
    pub(crate) predictor_score_semantics: &'static str,
    pub(crate) requested_fanout: usize,
    pub(crate) emitted_candidate_count: usize,
    pub(crate) emission_limitation_reason: Option<&'static str>,
    pub(crate) ranked_candidate_global_ids: Vec<u32>,
    pub(crate) ranked_candidate_scores: Vec<f64>,
    pub(crate) actual_selected_global_ids: Vec<u32>,
    pub(crate) actual_physical_current_selected_global_ids: Vec<u32>,
    pub(crate) actual_physical_missing_selected_global_ids: Vec<u32>,
    pub(crate) physical_slot_capacity_at_prediction: usize,
    pub(crate) resident_physical_count_at_prediction: usize,
    pub(crate) free_physical_slots_at_prediction: usize,
    pub(crate) resident_global_ids_mru_to_lru_at_prediction: Vec<u32>,
    pub(crate) route_true_positive_count: u64,
    pub(crate) physical_missing_true_positive_count: u64,
    pub(crate) exact_selected_top8_coverage: bool,
    pub(crate) exact_missing_set_coverage: bool,
    pub(crate) miss_boundary: bool,
    pub(crate) coverage_upper_bound_boundary_elimination: bool,
    pub(crate) already_resident_predicted_candidates: u64,
    pub(crate) wrong_predicted_candidates: u64,
    pub(crate) predicted_missing_but_not_selected: u64,
    pub(crate) duplicate_prediction_count: u64,
    pub(crate) prediction_missing_from_physical_at_prediction_time: u64,
    pub(crate) current_policy_installable_count: u64,
    pub(crate) current_policy_full_missing_set_installable: bool,
    pub(crate) current_policy_projected_boundary_elimination: bool,
    pub(crate) counterfactual_replacement_attempts: u64,
    pub(crate) counterfactual_selected_experts_evicted: u64,
    pub(crate) counterfactual_harmful_eviction_event: bool,
    pub(crate) counterfactual_complete_top8_residency: bool,
    pub(crate) counterfactual_projected_boundary_elimination: bool,
    pub(crate) lead_time_us: u64,
    pub(crate) ordering: MultipredictorEventOrdering,
}

fn fanouts_for_capacity(capacity: usize) -> Vec<usize> {
    let mut fanouts = BASE_FANOUTS.to_vec();
    fanouts.extend(
        OPTIONAL_FANOUTS
            .into_iter()
            .filter(|&fanout| fanout <= capacity),
    );
    fanouts
}

fn score_pending(
    pending: PendingTarget,
    actual_physical: AuthoritativeActualPhysicalClassification,
    target_physical: GpuNativePhysicalLayerShadowSnapshot,
    scored_sequence: u64,
    predictor_updated_sequence: u64,
) -> Result<Vec<MultipredictorPredictionEvent>, ShadowObserverError> {
    let target_probe_sequence = pending.target_probe_sequence.ok_or_else(|| {
        ShadowObserverError::new("multipredictor target route preceded its probe marker")
    })?;
    let target_probe_at_us = pending.target_probe_at_us.ok_or_else(|| {
        ShadowObserverError::new("multipredictor target route lacked a probe timestamp")
    })?;
    if !(pending.frozen_sequence < target_probe_sequence
        && target_probe_sequence < scored_sequence
        && scored_sequence < predictor_updated_sequence)
    {
        return Err(ShadowObserverError::new(format!(
            "multipredictor causal ordering failed: frozen={} probe={} scored={} updated={}",
            pending.frozen_sequence,
            target_probe_sequence,
            scored_sequence,
            predictor_updated_sequence,
        )));
    }
    if target_physical.layer_index != pending.target_layer
        || target_physical.slot_capacity != pending.physical.slot_capacity
    {
        return Err(ShadowObserverError::new(format!(
            "multipredictor physical geometry drifted: prediction={:?} target={target_physical:?}",
            pending.physical,
        )));
    }
    let actual = actual_physical.selected_global_ids;
    let actual_set = actual.iter().copied().collect::<HashSet<_>>();
    let actual_current = actual_physical.current_selected_global_ids;
    let missing = actual_physical.missing_selected_global_ids;
    let missing_set = missing.iter().copied().collect::<HashSet<_>>();
    let prediction_residents = pending
        .physical
        .resident_global_ids_mru_to_lru
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let miss_boundary = !missing.is_empty();
    let lead_time_us = target_probe_at_us.saturating_sub(pending.frozen_at_us);
    let mut events = Vec::new();

    for predictor in pending.predictors {
        for requested_fanout in fanouts_for_capacity(pending.physical.slot_capacity) {
            let prefix = predictor
                .ranked
                .iter()
                .take(requested_fanout)
                .cloned()
                .collect::<Vec<_>>();
            let candidate_set = prefix
                .iter()
                .map(|candidate| candidate.global_id)
                .collect::<HashSet<_>>();
            let route_true_positive_count = candidate_set.intersection(&actual_set).count() as u64;
            let physical_missing_true_positive_count =
                candidate_set.intersection(&missing_set).count() as u64;
            let exact_selected_top8_coverage = actual_set.is_subset(&candidate_set);
            let exact_missing_set_coverage = missing_set.is_subset(&candidate_set);
            let already_resident_predicted_candidates =
                candidate_set.intersection(&prediction_residents).count() as u64;
            let wrong_predicted_candidates = candidate_set.difference(&actual_set).count() as u64;
            let predicted_missing_but_not_selected = candidate_set
                .iter()
                .filter(|id| !prediction_residents.contains(id) && !actual_set.contains(id))
                .count() as u64;
            let prediction_missing_from_physical_at_prediction_time =
                candidate_set.difference(&prediction_residents).count() as u64;
            let current_policy = simulate_current_policy(&pending.physical, &prefix);
            let counterfactual =
                simulate_counterfactual_replacement(&pending.physical, &prefix, &actual_set);
            let current_policy_full_missing_set_installable =
                missing_set.is_subset(&current_policy.final_residents);
            let current_policy_projected_boundary_elimination =
                miss_boundary && current_policy_full_missing_set_installable;
            let counterfactual_complete_top8_residency =
                actual_set.is_subset(&counterfactual.final_residents);
            let counterfactual_projected_boundary_elimination =
                miss_boundary && counterfactual_complete_top8_residency;
            events.push(MultipredictorPredictionEvent {
                phase: pending.phase,
                run_index: pending.run_index,
                completed_token_position: pending.completed_token_position,
                target_layer: pending.target_layer,
                predictor_source: predictor.source,
                predictor_score_semantics: predictor.score_semantics,
                requested_fanout,
                emitted_candidate_count: prefix.len(),
                emission_limitation_reason: (prefix.len() < requested_fanout).then_some(
                    match predictor.source {
                        CONTROL => "private pre-gate transition table emitted fewer distinct target-layer candidates than requested",
                        MARKOV1 | MARKOV2 => "the configured global expert namespace, thresholded transition row, and deterministic unseen top-up emitted fewer distinct candidates than requested",
                        LOCALITY => "the thresholded private locality window contained fewer distinct hot experts than requested",
                        REDUCED_UNIFIED => "the eligible Markov2 and locality arms contained fewer distinct fused candidates than requested",
                        REDUCED_SPATIAL_AFFINITY => "the reduced unified, spatial, and historical layer-affinity arms contained fewer distinct fused candidates than requested",
                        _ => "the predictor emitted fewer distinct candidates than requested",
                    },
                ),
                ranked_candidate_global_ids: prefix
                    .iter()
                    .map(|candidate| candidate.global_id)
                    .collect(),
                ranked_candidate_scores: prefix.iter().map(|candidate| candidate.score).collect(),
                actual_selected_global_ids: actual.clone(),
                actual_physical_current_selected_global_ids: actual_current.clone(),
                actual_physical_missing_selected_global_ids: missing.clone(),
                physical_slot_capacity_at_prediction: pending.physical.slot_capacity,
                resident_physical_count_at_prediction: pending
                    .physical
                    .resident_global_ids_mru_to_lru
                    .len(),
                free_physical_slots_at_prediction: pending.physical.free_slots,
                resident_global_ids_mru_to_lru_at_prediction: pending
                    .physical
                    .resident_global_ids_mru_to_lru
                    .clone(),
                route_true_positive_count,
                physical_missing_true_positive_count,
                exact_selected_top8_coverage,
                exact_missing_set_coverage,
                miss_boundary,
                coverage_upper_bound_boundary_elimination: miss_boundary
                    && exact_missing_set_coverage,
                already_resident_predicted_candidates,
                wrong_predicted_candidates,
                predicted_missing_but_not_selected,
                duplicate_prediction_count: predictor.duplicate_prediction_count,
                prediction_missing_from_physical_at_prediction_time,
                current_policy_installable_count: current_policy.installable_count,
                current_policy_full_missing_set_installable,
                current_policy_projected_boundary_elimination,
                counterfactual_replacement_attempts: counterfactual.replacement_attempts,
                counterfactual_selected_experts_evicted: counterfactual.selected_experts_evicted,
                counterfactual_harmful_eviction_event: counterfactual.harmful_eviction_event,
                counterfactual_complete_top8_residency,
                counterfactual_projected_boundary_elimination,
                lead_time_us,
                ordering: MultipredictorEventOrdering {
                    prediction_frozen_sequence: pending.frozen_sequence,
                    target_probe_sequence,
                    prediction_scored_sequence: scored_sequence,
                    predictor_updated_sequence,
                    prediction_frozen_at_monotonic_us: pending.frozen_at_us,
                    target_physical_probe_at_monotonic_us: target_probe_at_us,
                },
            });
        }
    }
    Ok(events)
}

struct CurrentPolicySimulation {
    final_residents: HashSet<u32>,
    installable_count: u64,
}

fn simulate_current_policy(
    physical: &GpuNativePhysicalLayerShadowSnapshot,
    candidates: &[RankedPrediction],
) -> CurrentPolicySimulation {
    let mut final_residents = physical
        .resident_global_ids_mru_to_lru
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut free = physical.free_slots;
    let mut installable_count = 0u64;
    for candidate in candidates {
        if final_residents.contains(&candidate.global_id) || free == 0 {
            continue;
        }
        final_residents.insert(candidate.global_id);
        free -= 1;
        installable_count = installable_count.saturating_add(1);
    }
    CurrentPolicySimulation {
        final_residents,
        installable_count,
    }
}

struct CounterfactualSimulation {
    final_residents: HashSet<u32>,
    replacement_attempts: u64,
    selected_experts_evicted: u64,
    harmful_eviction_event: bool,
}

fn simulate_counterfactual_replacement(
    physical: &GpuNativePhysicalLayerShadowSnapshot,
    candidates: &[RankedPrediction],
    actual_selected: &HashSet<u32>,
) -> CounterfactualSimulation {
    let mut lru = physical.resident_global_ids_mru_to_lru.clone();
    let mut replacement_attempts = 0u64;
    let mut selected_experts_evicted = 0u64;
    for candidate in candidates {
        if let Some(index) = lru.iter().position(|id| *id == candidate.global_id) {
            let id = lru.remove(index);
            lru.insert(0, id);
            continue;
        }
        replacement_attempts = replacement_attempts.saturating_add(1);
        if lru.len() >= physical.slot_capacity && physical.slot_capacity > 0 {
            if let Some(victim) = lru.pop() {
                if actual_selected.contains(&victim) {
                    selected_experts_evicted = selected_experts_evicted.saturating_add(1);
                }
            }
        }
        if physical.slot_capacity > 0 {
            lru.insert(0, candidate.global_id);
        }
    }
    CounterfactualSimulation {
        final_residents: lru.into_iter().collect(),
        replacement_attempts,
        selected_experts_evicted,
        harmful_eviction_event: selected_experts_evicted > 0,
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct LeadTimeStatistics {
    pub(crate) min: u64,
    pub(crate) p50: u64,
    pub(crate) p95: u64,
    pub(crate) p99: u64,
    pub(crate) mean: f64,
    pub(crate) max: u64,
}

impl LeadTimeStatistics {
    fn from_values(values: &[u64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let percentile = |q: f64| {
            let index = ((sorted.len() - 1) as f64 * q).round() as usize;
            sorted[index]
        };
        Self {
            min: sorted[0],
            p50: percentile(0.50),
            p95: percentile(0.95),
            p99: percentile(0.99),
            mean: sorted.iter().map(|&value| value as f64).sum::<f64>() / sorted.len() as f64,
            max: *sorted.last().expect("lead times checked nonempty"),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct FanoutMetrics {
    pub(crate) predictor_source: String,
    pub(crate) requested_fanout: usize,
    pub(crate) target_event_count: u64,
    pub(crate) emitted_candidate_count_distribution: BTreeMap<usize, u64>,
    pub(crate) route_candidate_count: u64,
    pub(crate) route_true_positive_count: u64,
    pub(crate) route_precision: f64,
    pub(crate) route_recall: f64,
    pub(crate) physical_missing_true_positive_count: u64,
    pub(crate) physical_missing_precision: f64,
    pub(crate) physical_missing_recall: f64,
    pub(crate) physical_missing_f1: f64,
    pub(crate) exact_selected_top8_coverage_count: u64,
    pub(crate) exact_selected_top8_coverage_rate: f64,
    pub(crate) miss_boundary_count: u64,
    pub(crate) exact_missing_set_coverage_count: u64,
    pub(crate) exact_missing_set_coverage_rate: f64,
    pub(crate) coverage_upper_bound_boundary_eliminations: u64,
    pub(crate) coverage_upper_bound_boundary_elimination_rate: f64,
    pub(crate) current_policy_projected_boundary_eliminations: u64,
    pub(crate) current_policy_projected_boundary_elimination_rate: f64,
    pub(crate) counterfactual_projected_boundary_eliminations: u64,
    pub(crate) counterfactual_projected_boundary_elimination_rate: f64,
    pub(crate) counterfactual_replacement_attempts: u64,
    pub(crate) counterfactual_selected_experts_evicted: u64,
    pub(crate) counterfactual_harmful_eviction_events: u64,
    pub(crate) harmful_evictions_per_projected_elimination: Option<f64>,
    pub(crate) already_resident_predicted_candidates: u64,
    pub(crate) wrong_predicted_candidates: u64,
    pub(crate) predicted_missing_but_not_selected: u64,
    pub(crate) duplicate_prediction_count: u64,
    pub(crate) prediction_missing_from_physical_at_prediction_time: u64,
    pub(crate) current_policy_installable_count: u64,
    pub(crate) current_policy_full_missing_set_installable_count: u64,
    pub(crate) counterfactual_complete_top8_residency_count: u64,
}

impl FanoutMetrics {
    fn empty(source: &str, requested_fanout: usize) -> Self {
        Self {
            predictor_source: source.to_string(),
            requested_fanout,
            ..Self::default()
        }
    }

    fn add(&mut self, event: &MultipredictorPredictionEvent) {
        self.target_event_count = self.target_event_count.saturating_add(1);
        *self
            .emitted_candidate_count_distribution
            .entry(event.emitted_candidate_count)
            .or_insert(0) += 1;
        self.route_candidate_count += event.emitted_candidate_count as u64;
        self.route_true_positive_count += event.route_true_positive_count;
        self.physical_missing_true_positive_count += event.physical_missing_true_positive_count;
        self.exact_selected_top8_coverage_count += u64::from(event.exact_selected_top8_coverage);
        self.miss_boundary_count += u64::from(event.miss_boundary);
        self.exact_missing_set_coverage_count += u64::from(event.exact_missing_set_coverage);
        self.coverage_upper_bound_boundary_eliminations +=
            u64::from(event.coverage_upper_bound_boundary_elimination);
        self.current_policy_projected_boundary_eliminations +=
            u64::from(event.current_policy_projected_boundary_elimination);
        self.counterfactual_projected_boundary_eliminations +=
            u64::from(event.counterfactual_projected_boundary_elimination);
        self.counterfactual_replacement_attempts += event.counterfactual_replacement_attempts;
        self.counterfactual_selected_experts_evicted +=
            event.counterfactual_selected_experts_evicted;
        self.counterfactual_harmful_eviction_events +=
            u64::from(event.counterfactual_harmful_eviction_event);
        self.already_resident_predicted_candidates += event.already_resident_predicted_candidates;
        self.wrong_predicted_candidates += event.wrong_predicted_candidates;
        self.predicted_missing_but_not_selected += event.predicted_missing_but_not_selected;
        self.duplicate_prediction_count += event.duplicate_prediction_count;
        self.prediction_missing_from_physical_at_prediction_time +=
            event.prediction_missing_from_physical_at_prediction_time;
        self.current_policy_installable_count += event.current_policy_installable_count;
        self.current_policy_full_missing_set_installable_count +=
            u64::from(event.current_policy_full_missing_set_installable);
        self.counterfactual_complete_top8_residency_count +=
            u64::from(event.counterfactual_complete_top8_residency);
    }

    fn finish(&mut self, actual_selected: u64, actual_missing: u64) {
        self.route_precision = ratio(self.route_true_positive_count, self.route_candidate_count);
        self.route_recall = ratio(self.route_true_positive_count, actual_selected);
        self.physical_missing_precision = ratio(
            self.physical_missing_true_positive_count,
            self.route_candidate_count,
        );
        self.physical_missing_recall =
            ratio(self.physical_missing_true_positive_count, actual_missing);
        let f1_denominator = self.physical_missing_precision + self.physical_missing_recall;
        self.physical_missing_f1 = if f1_denominator == 0.0 {
            0.0
        } else {
            2.0 * self.physical_missing_precision * self.physical_missing_recall / f1_denominator
        };
        self.exact_selected_top8_coverage_rate = ratio(
            self.exact_selected_top8_coverage_count,
            self.target_event_count,
        );
        self.exact_missing_set_coverage_rate = ratio(
            self.exact_missing_set_coverage_count,
            self.target_event_count,
        );
        self.coverage_upper_bound_boundary_elimination_rate = ratio(
            self.coverage_upper_bound_boundary_eliminations,
            self.miss_boundary_count,
        );
        self.current_policy_projected_boundary_elimination_rate = ratio(
            self.current_policy_projected_boundary_eliminations,
            self.miss_boundary_count,
        );
        self.counterfactual_projected_boundary_elimination_rate = ratio(
            self.counterfactual_projected_boundary_eliminations,
            self.miss_boundary_count,
        );
        self.harmful_evictions_per_projected_elimination =
            if self.counterfactual_projected_boundary_eliminations == 0 {
                None
            } else {
                Some(
                    self.counterfactual_harmful_eviction_events as f64
                        / self.counterfactual_projected_boundary_eliminations as f64,
                )
            };
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct MissingSetHistogram {
    pub(crate) counts_0_through_8: [u64; 9],
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct LayerMetrics {
    pub(crate) target_layer: usize,
    pub(crate) target_event_count: u64,
    pub(crate) actual_miss_boundary_events: u64,
    pub(crate) missing_count_mean: f64,
    pub(crate) missing_set_histogram: MissingSetHistogram,
    pub(crate) fanouts: Vec<FanoutMetrics>,
    pub(crate) lead_time_us: LeadTimeStatistics,
    pub(crate) free_slot_distribution: BTreeMap<usize, u64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct PhaseAggregate {
    pub(crate) target_event_count: u64,
    pub(crate) missing_set_histogram: MissingSetHistogram,
    pub(crate) lead_time_us: LeadTimeStatistics,
    pub(crate) global_by_predictor_and_fanout: Vec<FanoutMetrics>,
    pub(crate) per_layer: Vec<LayerMetrics>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct MultipredictorObserverSnapshot {
    pub(crate) measured: PhaseAggregate,
    pub(crate) warmup: PhaseAggregate,
    pub(crate) events: Vec<MultipredictorPredictionEvent>,
}

impl MultipredictorObserverSnapshot {
    fn from_events(events: Vec<MultipredictorPredictionEvent>) -> Self {
        Self {
            measured: aggregate_phase(&events, ShadowPhase::Measured),
            warmup: aggregate_phase(&events, ShadowPhase::Warmup),
            events,
        }
    }
}

fn aggregate_phase(events: &[MultipredictorPredictionEvent], phase: ShadowPhase) -> PhaseAggregate {
    let filtered = events
        .iter()
        .filter(|event| event.phase == phase)
        .collect::<Vec<_>>();
    let mut unique_targets =
        BTreeMap::<(usize, usize, usize), &MultipredictorPredictionEvent>::new();
    for event in &filtered {
        unique_targets
            .entry((
                event.run_index,
                event.completed_token_position,
                event.target_layer,
            ))
            .or_insert(*event);
    }
    let mut histogram = MissingSetHistogram::default();
    let mut lead_times = Vec::with_capacity(unique_targets.len());
    for event in unique_targets.values() {
        let missing = event
            .actual_physical_missing_selected_global_ids
            .len()
            .min(8);
        histogram.counts_0_through_8[missing] += 1;
        lead_times.push(event.lead_time_us);
    }

    let mut global = BTreeMap::<(&'static str, usize), FanoutMetrics>::new();
    for event in &filtered {
        global
            .entry((event.predictor_source, event.requested_fanout))
            .or_insert_with(|| FanoutMetrics::empty(event.predictor_source, event.requested_fanout))
            .add(event);
    }
    finish_metric_map(&filtered, &mut global);
    let metric_keys = global.keys().copied().collect::<Vec<_>>();

    let layer_count = unique_targets
        .values()
        .map(|event| event.target_layer + 1)
        .max()
        .unwrap_or(0);
    let mut per_layer = Vec::with_capacity(layer_count);
    for layer in 0..layer_count {
        let targets = unique_targets
            .values()
            .copied()
            .filter(|event| event.target_layer == layer)
            .collect::<Vec<_>>();
        let layer_events = filtered
            .iter()
            .copied()
            .filter(|event| event.target_layer == layer)
            .collect::<Vec<_>>();
        let mut layer_histogram = MissingSetHistogram::default();
        let mut layer_leads = Vec::new();
        let mut free_slots = BTreeMap::new();
        let mut missing_total = 0u64;
        let mut miss_boundaries = 0u64;
        for event in &targets {
            let missing = event
                .actual_physical_missing_selected_global_ids
                .len()
                .min(8);
            layer_histogram.counts_0_through_8[missing] += 1;
            missing_total += missing as u64;
            miss_boundaries += u64::from(missing > 0);
            layer_leads.push(event.lead_time_us);
            *free_slots
                .entry(event.free_physical_slots_at_prediction)
                .or_insert(0) += 1;
        }
        let mut layer_fanouts = metric_keys
            .iter()
            .map(|&(source, fanout)| ((source, fanout), FanoutMetrics::empty(source, fanout)))
            .collect::<BTreeMap<_, _>>();
        for event in &layer_events {
            layer_fanouts
                .entry((event.predictor_source, event.requested_fanout))
                .or_insert_with(|| {
                    FanoutMetrics::empty(event.predictor_source, event.requested_fanout)
                })
                .add(event);
        }
        finish_metric_map(&layer_events, &mut layer_fanouts);
        per_layer.push(LayerMetrics {
            target_layer: layer,
            target_event_count: targets.len() as u64,
            actual_miss_boundary_events: miss_boundaries,
            missing_count_mean: if targets.is_empty() {
                0.0
            } else {
                missing_total as f64 / targets.len() as f64
            },
            missing_set_histogram: layer_histogram,
            fanouts: layer_fanouts.into_values().collect(),
            lead_time_us: LeadTimeStatistics::from_values(&layer_leads),
            free_slot_distribution: free_slots,
        });
    }
    PhaseAggregate {
        target_event_count: unique_targets.len() as u64,
        missing_set_histogram: histogram,
        lead_time_us: LeadTimeStatistics::from_values(&lead_times),
        global_by_predictor_and_fanout: global.into_values().collect(),
        per_layer,
    }
}

fn finish_metric_map<'a>(
    events: &[&'a MultipredictorPredictionEvent],
    metrics: &mut BTreeMap<(&'static str, usize), FanoutMetrics>,
) {
    for ((source, fanout), metric) in metrics {
        let relevant = events
            .iter()
            .filter(|event| event.predictor_source == *source && event.requested_fanout == *fanout);
        let (actual_selected, actual_missing) = relevant.fold((0u64, 0u64), |acc, event| {
            (
                acc.0 + event.actual_selected_global_ids.len() as u64,
                acc.1 + event.actual_physical_missing_selected_global_ids.len() as u64,
            )
        });
        metric.finish(actual_selected, actual_missing);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub(crate) struct RecoveryDiscreteProjection {
    pub(crate) resume_attempts: u64,
    pub(crate) recovery_segments: u64,
    pub(crate) checkpoint_captures: u64,
    pub(crate) checkpoint_restores: u64,
    pub(crate) full_token_replay_attempts: u64,
    pub(crate) layers_encoded: u64,
    pub(crate) attention_layers_reexecuted: u64,
    pub(crate) expert_layers_reexecuted: u64,
    pub(crate) invalid_tail_layers_encoded: u64,
}

impl From<crate::gpu_native_token_loop::GpuNativeRecoverySnapshot> for RecoveryDiscreteProjection {
    fn from(value: crate::gpu_native_token_loop::GpuNativeRecoverySnapshot) -> Self {
        Self {
            resume_attempts: value.resume_attempts,
            recovery_segments: value.recovery_segments,
            checkpoint_captures: value.checkpoint_captures,
            checkpoint_restores: value.checkpoint_restores,
            full_token_replay_attempts: value.full_token_replay_attempts,
            layers_encoded: value.layers_encoded,
            attention_layers_reexecuted: value.attention_layers_reexecuted,
            expert_layers_reexecuted: value.expert_layers_reexecuted,
            invalid_tail_layers_encoded: value.invalid_tail_layers_encoded,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub(crate) struct EngineStorageDiscreteProjection {
    pub(crate) ram_hits: u64,
    pub(crate) ram_misses: u64,
    pub(crate) nvme_read_operations: u64,
    pub(crate) nvme_bytes_read: u64,
    pub(crate) prefetch_completed: u64,
    pub(crate) predictor_observations: u64,
}

impl From<crate::gpu_native_real_benchmark::EngineStorageSnapshot>
    for EngineStorageDiscreteProjection
{
    fn from(value: crate::gpu_native_real_benchmark::EngineStorageSnapshot) -> Self {
        Self {
            ram_hits: value.ram_hits,
            ram_misses: value.ram_misses,
            nvme_read_operations: value.nvme_read_operations,
            nvme_bytes_read: value.nvme_bytes_read,
            prefetch_completed: value.prefetch_completed,
            predictor_observations: value.predictor_observations,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub(crate) struct GpuNativeResidencyDiscreteProjection {
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
}

impl From<crate::gpu_native_real_benchmark::GpuNativeResidencyDelta>
    for GpuNativeResidencyDiscreteProjection
{
    fn from(value: crate::gpu_native_real_benchmark::GpuNativeResidencyDelta) -> Self {
        Self {
            vram_hits: value.vram_hits,
            vram_misses: value.vram_misses,
            physical_current_hits: value.physical_current_hits,
            physical_source_acquisitions: value.physical_source_acquisitions,
            logical_admissions_for_physical_misses: value.logical_admissions_for_physical_misses,
            ram_to_vram_installs: value.ram_to_vram_installs,
            physical_evictions: value.physical_evictions,
            physical_reinstalls: value.physical_reinstalls,
            stale_generation_rejections: value.stale_generation_rejections,
            demand_requests: value.demand_requests,
            speculative_requests: value.speculative_requests,
            speculative_vram_hits: value.speculative_vram_hits,
            speculative_ram_to_vram_installs: value.speculative_ram_to_vram_installs,
            speculative_dropped_capacity_or_pressure: value
                .speculative_dropped_capacity_or_pressure,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct ProductionBehaviorProjection {
    pub(crate) generated_token_ids: Vec<u32>,
    pub(crate) token_loop_before: crate::gpu_native_token_loop::GpuNativeTokenLoopSnapshot,
    pub(crate) token_loop_after: crate::gpu_native_token_loop::GpuNativeTokenLoopSnapshot,
    pub(crate) token_loop_delta: crate::gpu_native_token_loop::GpuNativeTokenLoopSnapshot,
    pub(crate) recovery_before: RecoveryDiscreteProjection,
    pub(crate) recovery_after: RecoveryDiscreteProjection,
    pub(crate) recovery_delta: RecoveryDiscreteProjection,
    pub(crate) routed_execution_before: crate::engine::RoutedExpertExecutionSnapshot,
    pub(crate) routed_execution_after: crate::engine::RoutedExpertExecutionSnapshot,
    pub(crate) routed_execution_delta: crate::engine::RoutedExpertExecutionSnapshot,
    pub(crate) runtime_cache_before: crate::greedy_parity::RuntimeCacheSnapshot,
    pub(crate) runtime_cache_after: crate::greedy_parity::RuntimeCacheSnapshot,
    pub(crate) engine_storage_before: EngineStorageDiscreteProjection,
    pub(crate) engine_storage_after: EngineStorageDiscreteProjection,
    pub(crate) engine_storage_delta: EngineStorageDiscreteProjection,
    pub(crate) gpu_expert_io_before: crate::backend::GpuExpertIoSnapshot,
    pub(crate) gpu_expert_io_after: crate::backend::GpuExpertIoSnapshot,
    pub(crate) gpu_expert_io_delta: crate::backend::GpuExpertIoSnapshot,
    pub(crate) gpu_expert_memory_before: crate::backend::GpuExpertMemorySnapshot,
    pub(crate) gpu_expert_memory_after: crate::backend::GpuExpertMemorySnapshot,
    pub(crate) gpu_native_residency_before:
        crate::gpu_native_residency::GpuNativeTieredResidencySnapshot,
    pub(crate) gpu_native_residency_after:
        crate::gpu_native_residency::GpuNativeTieredResidencySnapshot,
    pub(crate) gpu_native_residency_delta: GpuNativeResidencyDiscreteProjection,
}

impl ProductionBehaviorProjection {
    fn from_request(
        generated_token_ids: &[u32],
        counters: &crate::gpu_native_real_benchmark::RequestSnapshots,
    ) -> Self {
        Self {
            generated_token_ids: generated_token_ids.to_vec(),
            token_loop_before: counters.token_loop_before,
            token_loop_after: counters.token_loop_after,
            token_loop_delta: counters.token_loop_delta,
            recovery_before: counters.recovery_before.into(),
            recovery_after: counters.recovery_after.into(),
            recovery_delta: counters.recovery_delta.into(),
            routed_execution_before: counters.routed_execution_before,
            routed_execution_after: counters.routed_execution_after,
            routed_execution_delta: counters.routed_execution_delta,
            runtime_cache_before: counters.runtime_cache_before,
            runtime_cache_after: counters.runtime_cache_after,
            engine_storage_before: counters.engine_storage_before.into(),
            engine_storage_after: counters.engine_storage_after.into(),
            engine_storage_delta: counters.engine_storage_delta.into(),
            gpu_expert_io_before: counters.gpu_expert_io_before,
            gpu_expert_io_after: counters.gpu_expert_io_after,
            gpu_expert_io_delta: counters.gpu_expert_io_delta,
            gpu_expert_memory_before: counters.gpu_expert_memory_before,
            gpu_expert_memory_after: counters.gpu_expert_memory_after,
            gpu_native_residency_before: counters.gpu_native_residency_before.clone(),
            gpu_native_residency_after: counters.gpu_native_residency_after.clone(),
            gpu_native_residency_delta: counters.gpu_native_residency_delta.into(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BehavioralEquivalenceContract {
    pub(crate) definition: &'static str,
    pub(crate) exact_parity_surface: Vec<&'static str>,
    pub(crate) excluded_timing_telemetry: Vec<&'static str>,
    pub(crate) timing_semantics: &'static str,
    pub(crate) frozen_pr1ca_interpretation: &'static str,
}

impl BehavioralEquivalenceContract {
    fn pr1cb() -> Self {
        Self {
            definition:
                "behavioral_equivalence = generated tokens + deterministic discrete production counters",
            exact_parity_surface: vec![
                "generated token IDs",
                "token-loop before/after/delta counters",
                "recovery before/after/delta discrete counters",
                "routed-execution before/after/delta counters",
                "runtime-cache before/after discrete state and counters",
                "engine/storage before/after/delta discrete counters",
                "GPU expert I/O before/after/delta counters",
                "GPU expert memory before/after discrete state and counters",
                "GPU-native residency before/after state plus delta counters including all speculative counters",
            ],
            excluded_timing_telemetry: vec![
                "recovery_before.residency_service_us",
                "recovery_after.residency_service_us",
                "recovery_delta.residency_service_us",
                "recovery_before.boundary_wait_us",
                "recovery_after.boundary_wait_us",
                "recovery_delta.boundary_wait_us",
                "recovery ratio fields derived from residency_service_us or boundary_wait_us",
                "engine/storage ssd_stall_us duration telemetry",
            ],
            timing_semantics:
                "timing_telemetry = diagnostic, non-qualifying, not exact-parity constrained",
            frozen_pr1ca_interpretation:
                "PASS — exact generated-token + deterministic discrete execution/residency/storage counter parity; serialized full-object mismatch was timing telemetry only.",
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CommandArgs {
    pub(crate) config: std::path::PathBuf,
    pub(crate) prompt: Option<String>,
    pub(crate) request_json: Option<std::path::PathBuf>,
    pub(crate) output_tokens: Option<usize>,
    pub(crate) warmup_runs: usize,
    pub(crate) measured_runs: usize,
    pub(crate) cache_reset: crate::BenchRealCacheReset,
    pub(crate) greedy: bool,
    pub(crate) expected_adapter_name: String,
    pub(crate) report_out: Option<std::path::PathBuf>,
    pub(crate) progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct MultipredictorProductionSemantics {
    pub(crate) inference_math_changed: bool,
    pub(crate) q4_changed: bool,
    pub(crate) router_changed: bool,
    pub(crate) attention_changed: bool,
    pub(crate) rmsnorm_changed: bool,
    pub(crate) lm_head_changed: bool,
    pub(crate) residency_policy_changed: bool,
    pub(crate) replay_policy_changed_relative_to_pr1ca: bool,
    pub(crate) prefetch_policy_changed: bool,
    pub(crate) capacity_changed: bool,
    pub(crate) private_shadow_learning_enabled: bool,
    pub(crate) active_speculative_io: bool,
}

impl MultipredictorProductionSemantics {
    const fn shadow_only() -> Self {
        Self {
            inference_math_changed: false,
            q4_changed: false,
            router_changed: false,
            attention_changed: false,
            rmsnorm_changed: false,
            lm_head_changed: false,
            residency_policy_changed: false,
            replay_policy_changed_relative_to_pr1ca: false,
            prefetch_policy_changed: false,
            capacity_changed: false,
            private_shadow_learning_enabled: true,
            active_speculative_io: false,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PredictorConfigurationEvidence {
    pub(crate) max_ranked_candidates: usize,
    pub(crate) markov_min_prob: f64,
    pub(crate) markov_seed: u64,
    pub(crate) locality_window_per_layer: usize,
    pub(crate) locality_effective_global_window: usize,
    pub(crate) locality_configured_threshold_pct: f32,
    pub(crate) locality_effective_threshold_pct: f32,
    pub(crate) affinity_neighbors_k: usize,
    pub(crate) affinity_decay_enabled: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct MultipredictorRunEvidence {
    pub(crate) phase: ShadowPhase,
    pub(crate) run_index: usize,
    pub(crate) prompt_tokens: usize,
    pub(crate) requested_output_tokens: usize,
    pub(crate) generated_tokens: usize,
    pub(crate) generated_token_ids: Vec<u32>,
    pub(crate) generated_token_ids_sha256: String,
    pub(crate) generated_text_sha256: String,
    pub(crate) behavioral_equivalence_projection: ProductionBehaviorProjection,
    pub(crate) production_counters: crate::gpu_native_real_benchmark::RequestSnapshots,
}

impl MultipredictorRunEvidence {
    fn from_result(
        phase: ShadowPhase,
        result: crate::gpu_native_real_benchmark::PerRunResult,
    ) -> Self {
        let behavioral_equivalence_projection = ProductionBehaviorProjection::from_request(
            &result.generated_token_ids,
            &result.counters,
        );
        Self {
            phase,
            run_index: result.run_index,
            prompt_tokens: result.prompt_tokens,
            requested_output_tokens: result.requested_output_tokens,
            generated_tokens: result.generated_tokens,
            generated_token_ids: result.generated_token_ids,
            generated_token_ids_sha256: result.generated_token_ids_sha256,
            generated_text_sha256: result.generated_text_sha256,
            behavioral_equivalence_projection,
            production_counters: result.counters,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct MultipredictorShadowReport {
    pub(crate) schema: &'static str,
    pub(crate) mode: &'static str,
    pub(crate) source_main_commit: &'static str,
    pub(crate) tested_pr1bb_commit: &'static str,
    pub(crate) tested_pr1bb_report_sha256: &'static str,
    pub(crate) pr1ca_commit: &'static str,
    pub(crate) pr1ca_report_sha256: &'static str,
    pub(crate) pr1ca_log_sha256: &'static str,
    pub(crate) canonical_benchmark_schema_unchanged: &'static str,
    pub(crate) shadow_complete: bool,
    pub(crate) failure: Option<crate::gpu_native_real_benchmark::BenchmarkFailure>,
    pub(crate) qualification_pass: bool,
    pub(crate) performance_claim: bool,
    pub(crate) production_prefetch_enabled: bool,
    pub(crate) production_speculative_runtime_postconditions_verified: bool,
    pub(crate) external_pr1bb_behavioral_equivalence_pending: bool,
    pub(crate) behavioral_equivalence: BehavioralEquivalenceContract,
    pub(crate) production_semantics: MultipredictorProductionSemantics,
    pub(crate) provenance: crate::gpu_native_real_benchmark::BenchmarkProvenance,
    pub(crate) hardware: Option<crate::backend::GpuDeviceIdentity>,
    pub(crate) model_identity: crate::greedy_parity::ModelIdentityEvidence,
    pub(crate) model_load: Option<crate::greedy_parity::ModelLoadEvidence>,
    pub(crate) runtime_contract: Option<crate::gpu_native_real_benchmark::RuntimeContractEvidence>,
    pub(crate) production_configuration: crate::gpu_native_real_benchmark::ProductionConfiguration,
    pub(crate) request: crate::gpu_native_real_benchmark::RequestEvidence,
    pub(crate) cache_reset: crate::BenchRealCacheReset,
    pub(crate) warmup_runs: usize,
    pub(crate) warmup_runs_completed: usize,
    pub(crate) measured_runs: usize,
    pub(crate) measured_runs_completed: usize,
    pub(crate) fanouts: Vec<usize>,
    pub(crate) predictor_configuration: PredictorConfigurationEvidence,
    pub(crate) source_causality_audit: Vec<PredictorCandidateAudit>,
    pub(crate) eligible_predictors_implemented: Vec<&'static str>,
    pub(crate) exact_prediction_freeze_point: &'static str,
    pub(crate) exact_update_timing: &'static str,
    pub(crate) layer_zero_prediction_semantics: &'static str,
    pub(crate) observer_runtime_guarded_classes: Vec<&'static str>,
    pub(crate) observer_runtime_guard_evidence: ObserverRuntimeGuardEvidence,
    pub(crate) lead_time_limitation: &'static str,
    pub(crate) current_policy_model: &'static str,
    pub(crate) counterfactual_model: &'static str,
    pub(crate) counterfactual_limitations: &'static str,
    pub(crate) warmup_learning_semantics: &'static str,
    pub(crate) warmup_run_evidence: Vec<MultipredictorRunEvidence>,
    pub(crate) measured_run_evidence: Vec<MultipredictorRunEvidence>,
    pub(crate) shadow_observations: Option<MultipredictorObserverSnapshot>,
    pub(crate) runtime_shutdown: Option<crate::greedy_parity::BackgroundShutdownEvidence>,
}

fn validate_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn resolve_predict_min_prob(configured: f64, num_experts: u32) -> f64 {
    if configured > 0.0 {
        configured
    } else {
        2.0 / num_experts.max(1) as f64
    }
}

fn validate_runtime_guard_evidence(
    evidence: ObserverRuntimeGuardEvidence,
) -> Result<(), crate::gpu_native_real_benchmark::BenchmarkFailure> {
    if !evidence.verified
        || evidence.checked_callback_count == 0
        || evidence.before_segment_callback_count == 0
        || evidence.observe_boundary_callback_count == 0
        || evidence.checked_callback_count
            != evidence
                .before_segment_callback_count
                .saturating_add(evidence.observe_boundary_callback_count)
    {
        return Err(crate::gpu_native_real_benchmark::BenchmarkFailure::new(
            "postcondition",
            "observer-runtime-guard-incomplete",
            format!(
                "multipredictor callback guard did not verify every callback class: {evidence:?}"
            ),
        ));
    }
    Ok(())
}

async fn execute_shadow_run(
    runtime: &crate::BenchRealRuntime,
    observer: &Arc<GpuNativeMultipredictorShadowObserver>,
    phase: ShadowPhase,
    run_index: usize,
    prompt_ids: &[u32],
    output_tokens: usize,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<MultipredictorRunEvidence, crate::gpu_native_real_benchmark::BenchmarkFailure> {
    observer.begin_run(phase, run_index).map_err(|error| {
        crate::gpu_native_real_benchmark::BenchmarkFailure::new(
            "shadow-observer",
            "begin-run-failed",
            error.to_string(),
        )
    })?;
    let phase_label = match phase {
        ShadowPhase::Warmup => "warmup",
        ShadowPhase::Measured => "measured",
    };
    let execution = crate::with_progress_timeout(
        format!("qualify-gpu-native-prefetch-multipredictor-shadow {phase_label} run {run_index}"),
        watchdog,
        crate::gpu_native_real_benchmark::execute_request(
            runtime,
            prompt_ids,
            output_tokens,
            run_index,
        ),
    )
    .await
    .map_err(|error| {
        crate::gpu_native_real_benchmark::BenchmarkFailure::new(
            "inference",
            "multipredictor-shadow-request-failed",
            error.to_string(),
        )
    });
    let ended = observer.end_run().map_err(|error| {
        crate::gpu_native_real_benchmark::BenchmarkFailure::new(
            "shadow-observer",
            "end-run-failed",
            error.to_string(),
        )
    });
    let result = match (execution, ended) {
        (Ok(result), Ok(())) => result,
        (Err(error), Ok(())) | (Ok(_), Err(error)) => return Err(error),
        (Err(execution), Err(ending)) => {
            return Err(crate::gpu_native_real_benchmark::BenchmarkFailure::new(
                "postcondition",
                "execution-and-observer-finalization-failed",
                format!("{execution}; {ending}"),
            ));
        }
    };
    crate::gpu_native_prefetch_shadow::validate_zero_speculative_work(&result)?;
    Ok(MultipredictorRunEvidence::from_result(phase, result))
}

fn emit_report(
    report: &MultipredictorShadowReport,
    path: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write as _;
    let mut bytes = serde_json::to_vec_pretty(report)?;
    bytes.push(b'\n');
    if let Some(path) = path {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bytes)?;
        eprintln!(
            "GPU-native multipredictor shadow report written to {}",
            path.display()
        );
    } else {
        std::io::stdout().write_all(&bytes)?;
    }
    Ok(())
}

pub(crate) async fn run_command(args: CommandArgs) -> Result<(), Box<dyn std::error::Error>> {
    use crate::gpu_native_real_benchmark::{BenchmarkFailure, BenchmarkProvenance};

    if !args.greedy {
        return Err(BenchmarkFailure::new(
            "preflight",
            "greedy-required",
            "qualify-gpu-native-prefetch-multipredictor-shadow requires the explicit --greedy flag",
        )
        .into());
    }
    if args.measured_runs == 0 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "measured-runs-required",
            "qualify-gpu-native-prefetch-multipredictor-shadow requires --measured-runs > 0",
        )
        .into());
    }
    if args.cache_reset != crate::BenchRealCacheReset::Keep {
        return Err(BenchmarkFailure::new(
            "preflight",
            "cache-reset-contract",
            "multipredictor shadow schema v2 supports only the frozen --cache-reset keep schedule",
        )
        .into());
    }
    if args.expected_adapter_name != FROZEN_ADAPTER_NAME {
        return Err(BenchmarkFailure::new(
            "preflight",
            "frozen-adapter-required",
            format!(
                "multipredictor shadow schema v2 requires --expected-adapter-name {FROZEN_ADAPTER_NAME:?}; observed {:?}",
                args.expected_adapter_name
            ),
        )
        .into());
    }
    let request_input = crate::load_real_cli_request_input(
        "qualify-gpu-native-prefetch-multipredictor-shadow",
        args.prompt.as_ref(),
        args.request_json.as_deref(),
        args.output_tokens,
    )?;
    if request_input.output_tokens < 2 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "insufficient-output-tokens",
            "qualify-gpu-native-prefetch-multipredictor-shadow requires --output-tokens >= 2",
        )
        .into());
    }

    let build = crate::qualification::BuildProvenance::embedded();
    let cfg = crate::config::Config::from_file(&args.config)?;
    crate::gpu_native_real_benchmark::validate_source_config(&cfg)?;
    let (artifacts, artifact_errors) = crate::qualification_artifacts(&args.config, &cfg);
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| {
                BenchmarkFailure::new("preflight", "expert-metadata-unavailable", error)
            })?;
    crate::gpu_native_prefetch_shadow::validate_shadow_preflight(
        &build,
        &artifacts,
        &artifact_errors,
        &expert_metadata,
    )?;
    let total_experts = (cfg.model.num_layers as u32)
        .checked_mul(cfg.model.num_experts)
        .ok_or_else(|| {
            BenchmarkFailure::new(
                "preflight",
                "expert-namespace-overflow",
                "multipredictor total expert namespace overflowed",
            )
        })?;
    let observer_config = MultipredictorObserverConfig {
        num_layers: cfg.model.num_layers,
        experts_per_layer: cfg.model.num_experts as usize,
        top_k: cfg.model.top_k,
        markov_min_prob: resolve_predict_min_prob(cfg.storage.predict_min_prob, total_experts),
        locality_window: cfg.predictive.locality_window,
        locality_threshold_pct: cfg.predictive.locality_threshold_pct,
        affinity_neighbors_k: cfg.predictive.affinity_neighbors_k.max(1),
    };
    let spec = crate::resolve_real_cli_spec_from_config(
        cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
    )?;
    let model_identity = crate::greedy_parity_model_identity(&spec);
    if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "wrong-model-identity",
            format!(
                "requires exact Qwen3-Coder 30B-A3B Q4_0 identity; observed {model_identity:?}"
            ),
        )
        .into());
    }
    let resolved_config_sha256 = crate::resolved_real_cli_spec_sha256(&spec)?;
    let tokenizer = crate::load_real_cli_tokenizer(
        &spec.cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
    )?;
    let prompt_ids = tokenizer.encode(&request_input.prompt)?;
    if prompt_ids.is_empty() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "empty-prompt-tokenization",
            "prompt encoded to zero tokens",
        )
        .into());
    }
    let (executable, executable_sha256) = crate::current_executable_identity()?;
    let executable_canonical_path = std::fs::canonicalize(&executable)?.display().to_string();
    if !validate_hex(&executable_sha256, 64) || !validate_hex(&resolved_config_sha256, 64) {
        return Err(BenchmarkFailure::new(
            "preflight",
            "provenance-unavailable",
            "executable or resolved-config SHA256 was unavailable",
        )
        .into());
    }
    let production_configuration =
        crate::gpu_native_real_benchmark::ProductionConfiguration::from_config(
            &spec.cfg,
            &expert_metadata,
        );
    let request = crate::gpu_native_real_benchmark::RequestEvidence {
        prompt_sha256: crate::greedy_parity::sha256_hex(request_input.prompt.as_bytes()),
        prompt_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&prompt_ids),
        prompt_token_count: prompt_ids.len(),
        requested_output_tokens: request_input.output_tokens,
        greedy: true,
    };
    let predictor_configuration = PredictorConfigurationEvidence {
        max_ranked_candidates: MAX_FANOUT,
        markov_min_prob: observer_config.markov_min_prob,
        markov_seed: 0xC0FFEE,
        locality_window_per_layer: observer_config.locality_window,
        locality_effective_global_window: observer_config
            .locality_window
            .saturating_mul(observer_config.num_layers),
        locality_configured_threshold_pct: observer_config.locality_threshold_pct,
        locality_effective_threshold_pct: observer_config.locality_threshold_pct
            / observer_config.num_layers as f32,
        affinity_neighbors_k: observer_config.affinity_neighbors_k,
        affinity_decay_enabled: false,
    };
    let mut report = MultipredictorShadowReport {
        schema: SCHEMA,
        mode: MODE,
        source_main_commit: SOURCE_MAIN_COMMIT,
        tested_pr1bb_commit: TESTED_PR1BB_COMMIT,
        tested_pr1bb_report_sha256: TESTED_PR1BB_REPORT_SHA256,
        pr1ca_commit: PR1CA_COMMIT,
        pr1ca_report_sha256: PR1CA_REPORT_SHA256,
        pr1ca_log_sha256: PR1CA_LOG_SHA256,
        canonical_benchmark_schema_unchanged: crate::gpu_native_real_benchmark::SCHEMA,
        shadow_complete: false,
        failure: None,
        qualification_pass: false,
        performance_claim: false,
        production_prefetch_enabled: false,
        production_speculative_runtime_postconditions_verified: false,
        external_pr1bb_behavioral_equivalence_pending: true,
        behavioral_equivalence: BehavioralEquivalenceContract::pr1cb(),
        production_semantics: MultipredictorProductionSemantics::shadow_only(),
        provenance: BenchmarkProvenance {
            build,
            executable_canonical_path,
            executable_sha256,
            resolved_config_sha256: resolved_config_sha256.clone(),
            artifacts,
            expert_metadata,
        },
        hardware: None,
        model_identity,
        model_load: None,
        runtime_contract: None,
        production_configuration,
        request,
        cache_reset: args.cache_reset,
        warmup_runs: args.warmup_runs,
        warmup_runs_completed: 0,
        measured_runs: args.measured_runs,
        measured_runs_completed: 0,
        fanouts: BASE_FANOUTS.to_vec(),
        predictor_configuration,
        source_causality_audit: predictor_candidate_audit(),
        eligible_predictors_implemented: vec![
            CONTROL,
            MARKOV1,
            MARKOV2,
            LOCALITY,
            REDUCED_UNIFIED,
            REDUCED_SPATIAL_AFFINITY,
        ],
        exact_prediction_freeze_point: "after ordinary boundary readback makes route L-1 CPU-visible and after all earlier targets are scored/learned, before the next segment submission/probe for target L",
        exact_update_timing: "score every frozen predictor against one authoritative target truth first; only then update private pre-gate, Markov, locality, and layer-affinity state",
        layer_zero_prediction_semantics: "unscored: no candidate is given a target-leaking source for layer 0 and PR1C-A layer-zero semantics remain unchanged",
        observer_runtime_guarded_classes: vec![
            "GPU-native residency counters including every speculative counter",
            "exact relevant-layer physical metadata, resident identities, MRU-to-LRU order, and free slots",
            "arena ownership, install/retire/reuse/mapping/stale-install/cancellation state",
            "RAM cache hits/misses and NVMe operation/byte counters",
            "prefetch_completed and production predictor observations",
            "logical GPU promotions/hits/misses/occupancy/used bytes",
            "production prefetch and governor counters",
        ],
        observer_runtime_guard_evidence: ObserverRuntimeGuardEvidence::default(),
        lead_time_limitation: "monotonic CPU freeze/probe timestamps are diagnostic only; no GPU timestamp or readback was added",
        current_policy_model: "unchanged PR1C-A model: ranked nonresident candidates consume only free physical target-layer slots and never evict",
        counterfactual_model: "unchanged PR1C-A oracle/shadow model: copy prediction-time MRU-to-LRU IDs, promote copied hits, fill free slots, then evict copied LRU tail in predictor order",
        counterfactual_limitations: "oracle harmful-eviction scoring uses later target truth; it models metadata capacity and LRU order, not I/O completion, bandwidth, contention, latency, or generation races",
        warmup_learning_semantics: "one private predictor set spans the keep-cache schedule; warmup trains causally, measured runs continue learning, and no phase resets learned tables/windows/matrices",
        warmup_run_evidence: Vec::new(),
        measured_run_evidence: Vec::new(),
        shadow_observations: None,
        runtime_shutdown: None,
    };

    let runtime = crate::build_isolated_greedy_runtime(
        &spec,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
        tokenizer,
    )
    .await
    .map_err(|error| {
        BenchmarkFailure::new("startup", "runtime-construction-failed", error.to_string())
    })?;
    let validation = crate::gpu_native_prefetch_shadow::validate_shadow_runtime(
        &runtime,
        &resolved_config_sha256,
        &args.expected_adapter_name,
    );
    let (contract, device, model_load) = match validation {
        Ok(evidence) => evidence,
        Err(error) => {
            let shutdown = runtime.shutdown_isolated().await;
            return match shutdown {
                Ok(_) => Err(error.into()),
                Err(shutdown_error) => {
                    Err(format!("{error}; shutdown also failed: {shutdown_error}").into())
                }
            };
        }
    };
    report.runtime_contract = Some(contract);
    report.hardware = Some(device);
    report.model_load = Some(model_load);
    let geometry = runtime
        .gpu_native_token_loop
        .as_ref()
        .expect("validated runtime has token loop")
        .model_geometry();
    if geometry.num_layers != observer_config.num_layers
        || geometry.num_experts != observer_config.experts_per_layer
        || geometry.top_k != observer_config.top_k
    {
        let shutdown = runtime.shutdown_isolated().await;
        return Err(format!(
            "runtime geometry drifted from preflight observer config: runtime={geometry:?} observer={observer_config:?} shutdown={shutdown:?}"
        )
        .into());
    }
    let observer = GpuNativeMultipredictorShadowObserver::new(observer_config)?;
    runtime
        .gpu_native_token_loop
        .as_ref()
        .expect("validated runtime has token loop")
        .install_prefetch_shadow_observer(observer.clone())?;

    let execution = async {
        for run_index in 0..args.warmup_runs {
            let evidence = execute_shadow_run(
                &runtime,
                &observer,
                ShadowPhase::Warmup,
                run_index,
                &prompt_ids,
                request_input.output_tokens,
                args.progress_watchdog,
            )
            .await?;
            report.warmup_run_evidence.push(evidence);
            report.warmup_runs_completed += 1;
        }
        for run_index in 0..args.measured_runs {
            let evidence = execute_shadow_run(
                &runtime,
                &observer,
                ShadowPhase::Measured,
                run_index,
                &prompt_ids,
                request_input.output_tokens,
                args.progress_watchdog,
            )
            .await?;
            report.measured_run_evidence.push(evidence);
            report.measured_runs_completed += 1;
        }
        let snapshot = observer.snapshot();
        if snapshot.measured.target_event_count == 0 {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "no-causal-measured-events",
                "no target layer became eligible for multipredictor shadow scoring",
            ));
        }
        let expected_sources = report
            .eligible_predictors_implemented
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let measured_sources = snapshot
            .events
            .iter()
            .filter(|event| event.phase == ShadowPhase::Measured)
            .map(|event| event.predictor_source)
            .collect::<HashSet<_>>();
        if measured_sources != expected_sources {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "incomplete-predictor-set",
                format!("expected {expected_sources:?}, observed {measured_sources:?}"),
            ));
        }
        report.fanouts = snapshot
            .events
            .iter()
            .map(|event| event.requested_fanout)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let runtime_guard_evidence = observer.runtime_guard_evidence();
        validate_runtime_guard_evidence(runtime_guard_evidence)?;
        report.observer_runtime_guard_evidence = runtime_guard_evidence;
        report.shadow_observations = Some(snapshot);
        Ok::<(), BenchmarkFailure>(())
    }
    .await;
    drop(observer);
    let shutdown = runtime.shutdown_isolated().await;
    match shutdown {
        Ok(evidence) => report.runtime_shutdown = Some(evidence),
        Err(error) => {
            let failure = BenchmarkFailure::new(
                "postcondition",
                "runtime-shutdown-failed",
                error.to_string(),
            );
            report.failure = Some(failure.clone());
            emit_report(&report, args.report_out.as_deref())?;
            return Err(failure.into());
        }
    }

    match execution {
        Ok(()) => {
            if report.warmup_runs_completed != report.warmup_runs
                || report.measured_runs_completed != report.measured_runs
                || report.warmup_run_evidence.len() != report.warmup_runs
                || report.measured_run_evidence.len() != report.measured_runs
            {
                let failure = BenchmarkFailure::new(
                    "postcondition",
                    "incomplete-run-set",
                    "multipredictor shadow qualification did not retain every requested run",
                );
                report.failure = Some(failure.clone());
                emit_report(&report, args.report_out.as_deref())?;
                return Err(failure.into());
            }
            report.production_speculative_runtime_postconditions_verified = true;
            report.shadow_complete = true;
            emit_report(&report, args.report_out.as_deref())
        }
        Err(failure) => {
            let summary = failure.to_string();
            report.failure = Some(failure);
            report.shadow_complete = false;
            emit_report(&report, args.report_out.as_deref())?;
            Err(summary.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;
    use std::cell::RefCell;

    fn config() -> MultipredictorObserverConfig {
        MultipredictorObserverConfig {
            num_layers: 3,
            experts_per_layer: 8,
            top_k: 2,
            markov_min_prob: 0.0,
            locality_window: 8,
            locality_threshold_pct: 0.10,
            affinity_neighbors_k: 2,
        }
    }

    fn physical(
        layer_index: usize,
        residents: &[u32],
        capacity: usize,
    ) -> GpuNativePhysicalLayerShadowSnapshot {
        GpuNativePhysicalLayerShadowSnapshot {
            layer_index,
            slot_capacity: capacity,
            resident_global_ids_mru_to_lru: residents.to_vec(),
            free_slots: capacity.saturating_sub(residents.len()),
        }
    }

    fn ranked(ids: &[u32]) -> Vec<RankedPrediction> {
        ids.iter()
            .enumerate()
            .map(|(rank, &global_id)| RankedPrediction {
                global_id,
                score: (ids.len() - rank) as f64,
            })
            .collect()
    }

    fn pending(predictors: Vec<(&'static str, Vec<u32>)>) -> PendingTarget {
        PendingTarget {
            phase: ShadowPhase::Measured,
            run_index: 0,
            completed_token_position: 9,
            target_layer: 1,
            predictors: predictors
                .into_iter()
                .map(|(source, ids)| FrozenPredictor {
                    source,
                    score_semantics: "test",
                    ranked: ranked(&ids),
                    duplicate_prediction_count: 0,
                })
                .collect(),
            physical: physical(1, &[8, 9], 16),
            frozen_sequence: 1,
            frozen_at_us: 10,
            target_probe_sequence: Some(2),
            target_probe_at_us: Some(20),
        }
    }

    fn actual(
        selected: &[u32],
        current: &[u32],
        missing: &[u32],
    ) -> AuthoritativeActualPhysicalClassification {
        AuthoritativeActualPhysicalClassification {
            selected_global_ids: selected.to_vec(),
            current_selected_global_ids: current.to_vec(),
            missing_selected_global_ids: missing.to_vec(),
        }
    }

    #[test]
    fn schema_mode_and_cli_are_separate_from_v1() {
        assert_eq!(SCHEMA, "mer.gpu-native-prefetch-multipredictor-shadow.v2");
        assert_eq!(MODE, "gpu-native-prefetch-multipredictor-shadow");
        assert_ne!(SCHEMA, crate::gpu_native_prefetch_shadow::SCHEMA);
        let cli = crate::Cli::try_parse_from([
            "mer",
            "qualify-gpu-native-prefetch-multipredictor-shadow",
            "--config",
            "config.toml",
            "--output-tokens",
            "31",
            "--greedy",
            "--expected-adapter-name",
            "NVIDIA L4",
        ])
        .expect("v2 command parses");
        assert!(matches!(
            cli.cmd,
            crate::Cmd::QualifyGpuNativePrefetchMultipredictorShadow { .. }
        ));
    }

    #[test]
    fn source_audit_is_fail_closed_and_neural_is_refused() {
        let audit = predictor_candidate_audit();
        for source in [
            CONTROL,
            MARKOV1,
            MARKOV2,
            LOCALITY,
            REDUCED_UNIFIED,
            REDUCED_SPATIAL_AFFINITY,
        ] {
            let candidate = audit
                .iter()
                .find(|candidate| candidate.source == source)
                .unwrap();
            assert_eq!(candidate.eligibility, "eligible");
            assert!(!candidate.source_locations.is_empty());
            assert!(candidate.cpu_visible_at_prediction_freeze);
            assert!(candidate.private_shadow_state);
            assert!(!candidate.target_leakage);
        }
        assert!(audit.iter().all(|candidate| {
            !candidate.source_locations.is_empty() && !candidate.target_leakage
        }));
        let neural = audit
            .iter()
            .find(|candidate| candidate.source == "NeuralSpeculator")
            .unwrap();
        assert_eq!(neural.eligibility, "ineligible");
        assert!(!neural.cpu_visible_at_prediction_freeze);
        assert!(neural
            .unavailable_or_prohibited_requirements
            .iter()
            .any(|reason| reason.contains("GPU-to-CPU readback")));
        let affinity = audit
            .iter()
            .find(|candidate| candidate.source == "LayeredExpertAffinity")
            .unwrap();
        assert_eq!(affinity.eligibility, "eligible-fusion-only");
    }

    #[test]
    fn control_ranking_reproduces_pr1ca_pregate_semantics() {
        let cfg = config();
        let observer = GpuNativeMultipredictorShadowObserver::new(cfg.clone()).unwrap();
        let reference = PerLayerPreGate::new(cfg.num_layers, MAX_FANOUT);
        reference.observe_transition(0, &[1, 2], &[3, 4]);
        reference.observe_transition(0, &[1, 2], &[3, 5]);
        let expected = reference
            .predict_ranked(0, &[1, 2], MAX_FANOUT)
            .into_iter()
            .map(|candidate| global_id(1, candidate.expert_id, cfg.experts_per_layer).unwrap())
            .collect::<Vec<_>>();

        let mut inner = observer.inner.lock();
        inner
            .predictors
            .pregate
            .observe_transition(0, &[1, 2], &[3, 4]);
        inner
            .predictors
            .pregate
            .observe_transition(0, &[1, 2], &[3, 5]);
        let source = RouteHistory {
            completed_token_position: 0,
            layer: 0,
            local_ids: vec![1, 2],
            global_ids: vec![1, 2],
        };
        let frozen = freeze_predictions(
            &mut inner,
            &cfg,
            ActiveRun {
                phase: ShadowPhase::Warmup,
                run_index: 0,
            },
            0,
            1,
            &source,
            None,
            physical(1, &[], 16),
        )
        .unwrap();
        let control = frozen
            .predictors
            .iter()
            .find(|predictor| predictor.source == CONTROL)
            .unwrap()
            .ranked
            .iter()
            .map(|prediction| prediction.global_id)
            .collect::<Vec<_>>();
        assert_eq!(control, expected);
    }

    #[test]
    fn markov_shadow_adapters_preserve_probabilities_and_break_ties_by_id() {
        let loader = PredictiveLoader::new(16, MAX_FANOUT, 0.0, 7);
        loader.observe_step(&[5], &[9, 7]);
        let first = loader.predict_next_shadow_ranked(5, 4);
        assert_eq!(first[0].0, 7);
        assert_eq!(first[1].0, 9);
        assert_eq!(first[0].1, first[1].1);

        loader.observe_step2(&[3], &[5], &[11, 10]);
        let second = loader.predict_next2_shadow_ranked(3, 5, 4);
        let tied = second
            .windows(2)
            .filter(|pair| pair[0].1 == pair[1].1)
            .all(|pair| pair[0].0 < pair[1].0);
        assert!(
            tied,
            "equal second-order scores must tie-break by ascending ID"
        );
    }

    #[test]
    fn warmup_private_learning_persists_into_measured_phase_without_aliasing() {
        let observer = GpuNativeMultipredictorShadowObserver::new(config()).unwrap();
        let production = PredictiveLoader::new(24, MAX_FANOUT, 0.0, 99);
        observer.begin_run(ShadowPhase::Warmup, 0).unwrap();
        {
            let inner = observer.inner.lock();
            let shadow_address = &inner.predictors.markov as *const PredictiveLoader;
            let production_address = &production as *const PredictiveLoader;
            assert_ne!(shadow_address, production_address);
            inner.predictors.markov.observe_step(&[1], &[9, 10]);
            inner.predictors.locality.observe(&[1, 2]);
            inner.predictors.affinity.observe_layer(1, &[1, 2]);
        }
        observer.end_run().unwrap();
        let after_warmup = observer.predictor_observation_counts();
        observer.begin_run(ShadowPhase::Measured, 0).unwrap();
        assert_eq!(observer.predictor_observation_counts(), after_warmup);
        observer.end_run().unwrap();
        assert_eq!(production.observations(), 0);
    }

    #[test]
    fn every_predictor_scores_the_same_truth_and_updates_follow_scoring() {
        let events = score_pending(
            pending(vec![(CONTROL, vec![10, 11]), (MARKOV1, vec![11, 12])]),
            actual(&[10, 11], &[10], &[11]),
            physical(1, &[8, 9], 16),
            3,
            4,
        )
        .unwrap();
        assert_eq!(events.len(), 12);
        let truth = events[0].actual_selected_global_ids.clone();
        let missing = events[0]
            .actual_physical_missing_selected_global_ids
            .clone();
        assert!(events.iter().all(|event| {
            event.actual_selected_global_ids == truth
                && event.actual_physical_missing_selected_global_ids == missing
                && event.ordering.prediction_frozen_sequence < event.ordering.target_probe_sequence
                && event.ordering.target_probe_sequence < event.ordering.prediction_scored_sequence
                && event.ordering.prediction_scored_sequence
                    < event.ordering.predictor_updated_sequence
        }));
    }

    struct TestCurrentness {
        outcomes: BTreeMap<u32, Result<bool, GpuNativeTieredResidencyError>>,
        calls: RefCell<Vec<u32>>,
    }

    impl AuthoritativePhysicalCurrentness for TestCurrentness {
        fn authoritative_physical_current(
            &self,
            global_id: u32,
        ) -> Result<bool, GpuNativeTieredResidencyError> {
            self.calls.borrow_mut().push(global_id);
            self.outcomes.get(&global_id).cloned().unwrap_or(Ok(false))
        }
    }

    #[test]
    fn authoritative_currentness_is_called_for_every_selected_expert_and_corruption_fails() {
        let currentness = TestCurrentness {
            outcomes: BTreeMap::from([(8, Ok(true)), (9, Ok(false))]),
            calls: RefCell::new(Vec::new()),
        };
        let classified =
            classify_authoritative_actual_selected(&currentness, 1, &[0, 1], 8).unwrap();
        assert_eq!(&*currentness.calls.borrow(), &[8, 9]);
        assert_eq!(classified.current_selected_global_ids, vec![8]);
        assert_eq!(classified.missing_selected_global_ids, vec![9]);

        let corrupt = TestCurrentness {
            outcomes: BTreeMap::from([(
                8,
                Err(GpuNativeTieredResidencyError::PhysicalIdentityCorrupt { global_id: 8 }),
            )]),
            calls: RefCell::new(Vec::new()),
        };
        let error = classify_authoritative_actual_selected(&corrupt, 1, &[0], 8).unwrap_err();
        assert!(error.to_string().contains("metadata disagrees"));
    }

    #[test]
    fn fanouts_duplicates_short_emissions_and_counterfactual_harm_are_explicit() {
        assert_eq!(fanouts_for_capacity(8), vec![1, 2, 4, 8]);
        assert_eq!(fanouts_for_capacity(12), vec![1, 2, 4, 8, 12]);
        assert_eq!(fanouts_for_capacity(16), vec![1, 2, 4, 8, 12, 16]);
        let (deduped, duplicates) = dedupe_ranked(ranked(&[10, 10, 11]));
        assert_eq!(duplicates, 1);
        assert_eq!(deduped.len(), 2);

        let events = score_pending(
            pending(vec![(CONTROL, vec![12])]),
            actual(&[8, 12], &[8], &[12]),
            physical(1, &[8, 9], 16),
            3,
            4,
        )
        .unwrap();
        assert!(events
            .iter()
            .all(|event| event.emitted_candidate_count == 1));
        assert!(events
            .iter()
            .filter(|event| event.requested_fanout > 1)
            .all(|event| event.emission_limitation_reason.is_some()));
        let harmful = simulate_counterfactual_replacement(
            &physical(1, &[8, 9], 2),
            &ranked(&[10]),
            &HashSet::from([9]),
        );
        assert!(harmful.harmful_eviction_event);
        assert_eq!(harmful.selected_experts_evicted, 1);
    }

    #[test]
    fn exact_missing_coverage_and_aggregates_are_per_predictor_and_per_layer() {
        let events = score_pending(
            pending(vec![(CONTROL, vec![11]), (LOCALITY, vec![])]),
            actual(&[10, 11], &[10], &[11]),
            physical(1, &[8, 9], 16),
            3,
            4,
        )
        .unwrap();
        let snapshot = MultipredictorObserverSnapshot::from_events(events);
        assert_eq!(snapshot.measured.target_event_count, 1);
        assert_eq!(snapshot.measured.per_layer.len(), 2);
        assert_eq!(snapshot.measured.per_layer[1].target_event_count, 1);
        let control = snapshot
            .measured
            .global_by_predictor_and_fanout
            .iter()
            .find(|metrics| metrics.predictor_source == CONTROL && metrics.requested_fanout == 1)
            .unwrap();
        assert_eq!(control.exact_missing_set_coverage_count, 1);
        assert_eq!(control.physical_missing_recall, 1.0);
        let locality = snapshot
            .measured
            .global_by_predictor_and_fanout
            .iter()
            .find(|metrics| metrics.predictor_source == LOCALITY && metrics.requested_fanout == 1)
            .unwrap();
        assert_eq!(
            locality.emitted_candidate_count_distribution.get(&0),
            Some(&1)
        );
        assert_eq!(locality.harmful_evictions_per_projected_elimination, None);
    }

    fn empty_snapshots() -> crate::gpu_native_real_benchmark::RequestSnapshots {
        crate::gpu_native_real_benchmark::RequestSnapshots {
            token_loop_before: Default::default(),
            token_loop_after: Default::default(),
            token_loop_delta: Default::default(),
            token_loop_ratios: Default::default(),
            recovery_before: Default::default(),
            recovery_after: Default::default(),
            recovery_delta: Default::default(),
            recovery_ratios: Default::default(),
            routed_execution_before: Default::default(),
            routed_execution_after: Default::default(),
            routed_execution_delta: Default::default(),
            runtime_cache_before: Default::default(),
            runtime_cache_after: Default::default(),
            engine_storage_before: Default::default(),
            engine_storage_after: Default::default(),
            engine_storage_delta: Default::default(),
            gpu_expert_io_before: Default::default(),
            gpu_expert_io_after: Default::default(),
            gpu_expert_io_delta: Default::default(),
            gpu_expert_memory_before: Default::default(),
            gpu_expert_memory_after: Default::default(),
            gpu_native_residency_before: Default::default(),
            gpu_native_residency_after: Default::default(),
            gpu_native_residency_delta: Default::default(),
        }
    }

    fn per_run(
        counters: crate::gpu_native_real_benchmark::RequestSnapshots,
    ) -> crate::gpu_native_real_benchmark::PerRunResult {
        crate::gpu_native_real_benchmark::PerRunResult {
            run_index: 2,
            prompt_tokens: 3,
            requested_output_tokens: 2,
            generated_tokens: 2,
            generated_token_ids: vec![41, 42],
            generated_token_ids_sha256: "token-fingerprint".into(),
            generated_text_sha256: "text-fingerprint".into(),
            timing: crate::gpu_native_real_benchmark::RunTiming::from_measurement(
                2,
                1.0,
                1.0,
                1.0,
                vec![0.1],
            )
            .unwrap(),
            counters,
        }
    }

    #[test]
    fn run_evidence_retains_token_fingerprints_and_rejects_speculative_production_work() {
        let clean = per_run(empty_snapshots());
        assert!(crate::gpu_native_prefetch_shadow::validate_zero_speculative_work(&clean).is_ok());
        let evidence = MultipredictorRunEvidence::from_result(ShadowPhase::Measured, clean);
        assert_eq!(evidence.generated_token_ids, vec![41, 42]);
        assert_eq!(evidence.generated_token_ids_sha256, "token-fingerprint");
        assert_eq!(evidence.generated_text_sha256, "text-fingerprint");
        assert_eq!(
            evidence
                .behavioral_equivalence_projection
                .generated_token_ids,
            vec![41, 42]
        );

        macro_rules! assert_speculative_rejected {
            ($body:expr) => {{
                let mut counters = empty_snapshots();
                $body(&mut counters);
                let result = per_run(counters);
                assert!(
                    crate::gpu_native_prefetch_shadow::validate_zero_speculative_work(&result)
                        .is_err()
                );
            }};
        }
        assert_speculative_rejected!(
            |value: &mut crate::gpu_native_real_benchmark::RequestSnapshots| value
                .gpu_native_residency_delta
                .speculative_requests =
                1
        );
        assert_speculative_rejected!(
            |value: &mut crate::gpu_native_real_benchmark::RequestSnapshots| value
                .gpu_native_residency_delta
                .speculative_vram_hits =
                1
        );
        assert_speculative_rejected!(
            |value: &mut crate::gpu_native_real_benchmark::RequestSnapshots| value
                .gpu_native_residency_delta
                .speculative_ram_to_vram_installs =
                1
        );
        assert_speculative_rejected!(
            |value: &mut crate::gpu_native_real_benchmark::RequestSnapshots| value
                .gpu_native_residency_delta
                .speculative_dropped_capacity_or_pressure =
                1
        );
        assert_speculative_rejected!(
            |value: &mut crate::gpu_native_real_benchmark::RequestSnapshots| value
                .engine_storage_delta
                .prefetch_completed =
                1
        );
    }

    #[test]
    fn behavior_projection_excludes_only_declared_durations_and_retains_tokens() {
        let base = empty_snapshots();
        let expected = ProductionBehaviorProjection::from_request(&[7, 8, 9], &base);
        let mut timing_only = empty_snapshots();
        timing_only.recovery_before.residency_service_us = 1;
        timing_only.recovery_after.residency_service_us = 2;
        timing_only.recovery_delta.residency_service_us = 1;
        timing_only.recovery_before.boundary_wait_us = 3;
        timing_only.recovery_after.boundary_wait_us = 5;
        timing_only.recovery_delta.boundary_wait_us = 2;
        timing_only
            .recovery_ratios
            .residency_service_us_per_completed_position = 12.5;
        timing_only
            .recovery_ratios
            .boundary_wait_us_per_completed_position = 19.0;
        timing_only.engine_storage_delta.ssd_stall_us = 77;
        let observed = ProductionBehaviorProjection::from_request(&[7, 8, 9], &timing_only);
        assert_eq!(expected, observed);
        assert_eq!(observed.generated_token_ids, vec![7, 8, 9]);
    }

    #[test]
    fn behavior_projection_fails_on_meaningful_discrete_counter_changes() {
        let baseline = ProductionBehaviorProjection::from_request(&[1], &empty_snapshots());
        macro_rules! assert_changed {
            ($body:expr) => {{
                let mut changed = baseline.clone();
                $body(&mut changed);
                assert_ne!(baseline, changed);
            }};
        }
        assert_changed!(|value: &mut ProductionBehaviorProjection| value
            .generated_token_ids
            .push(2));
        assert_changed!(|value: &mut ProductionBehaviorProjection| value
            .token_loop_delta
            .token_attempts = 1);
        assert_changed!(|value: &mut ProductionBehaviorProjection| value
            .recovery_delta
            .resume_attempts = 1);
        assert_changed!(|value: &mut ProductionBehaviorProjection| value
            .routed_execution_delta
            .cpu_routed_expert_dispatches =
            1);
        assert_changed!(|value: &mut ProductionBehaviorProjection| value
            .runtime_cache_after
            .predictor_observations = 1);
        assert_changed!(|value: &mut ProductionBehaviorProjection| value
            .engine_storage_delta
            .nvme_bytes_read = 1);
        assert_changed!(|value: &mut ProductionBehaviorProjection| value
            .gpu_expert_io_delta
            .readback_bytes = 1);
        assert_changed!(|value: &mut ProductionBehaviorProjection| value
            .gpu_expert_memory_after
            .physical_installs = 1);
        assert_changed!(|value: &mut ProductionBehaviorProjection| value
            .gpu_native_residency_delta
            .physical_current_hits = 1);
        assert_changed!(|value: &mut ProductionBehaviorProjection| value
            .gpu_native_residency_delta
            .speculative_requests = 1);
    }

    #[test]
    fn runtime_guard_completion_requires_both_callback_classes() {
        assert!(validate_runtime_guard_evidence(ObserverRuntimeGuardEvidence::default()).is_err());
        assert!(
            validate_runtime_guard_evidence(ObserverRuntimeGuardEvidence {
                verified: true,
                checked_callback_count: 1,
                before_segment_callback_count: 1,
                observe_boundary_callback_count: 0,
            })
            .is_err()
        );
        let observer = GpuNativeMultipredictorShadowObserver::new(config()).unwrap();
        GpuNativePrefetchShadowCallbacks::record_runtime_guarded_callback(
            observer.as_ref(),
            ShadowObserverCallbackKind::BeforeSegment,
        );
        GpuNativePrefetchShadowCallbacks::record_runtime_guarded_callback(
            observer.as_ref(),
            ShadowObserverCallbackKind::ObserveBoundary,
        );
        let evidence = observer.runtime_guard_evidence();
        assert!(validate_runtime_guard_evidence(evidence).is_ok());
        assert_eq!(evidence.checked_callback_count, 2);
    }

    #[test]
    fn causal_ordering_fails_closed_before_scoring() {
        let mut invalid = pending(vec![(CONTROL, vec![10])]);
        invalid.target_probe_sequence = Some(1);
        let error = score_pending(
            invalid,
            actual(&[10], &[], &[10]),
            physical(1, &[8, 9], 16),
            3,
            4,
        )
        .unwrap_err();
        assert!(error.to_string().contains("causal ordering failed"));
    }
}
