//! Shadow-only causal predictor and physical-residency scoring for the
//! production GPU-native token loop.
//!
//! The observer owns only CPU metadata. It never calls storage, cache
//! admission, prefetch-governor, or physical-residency mutation APIs.

use crate::gpu_native_residency::{
    GpuNativePhysicalLayerShadowSnapshot, GpuNativeTieredResidencyError,
    GpuNativeTieredResidencyManager,
};
use crate::pregate::PerLayerPreGate;
use parking_lot::Mutex;
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

pub(crate) const SCHEMA: &str = "mer.gpu-native-prefetch-shadow.v1";
pub(crate) const MODE: &str = "gpu-native-prefetch-shadow";
pub(crate) const SOURCE_MAIN_COMMIT: &str = "e8b542110693e74aa8f1013bb16d1bed0bdd8ba7";
pub(crate) const TESTED_PR1BB_COMMIT: &str = "c7006c5c6fbcee74c91526f89a8c2b5b06d8a9c5";
pub(crate) const TESTED_PR1BB_REPORT_SHA256: &str =
    "82317ad85ba29da1401abf3041ebbb7c2ca72c039c1be0c5f2ea53df9e9fc529";
pub(crate) const PR1_COMMIT: &str = "a39c58062ce167773248d3fa5618ac5fd55e54ba";
pub(crate) const ORIGINAL_BASELINE_COMMIT: &str = "db0664159fe4a57e5b630984b9229e233fa21487";
pub(crate) const PR1B_A_EXPERIMENT_COMMIT: &str = "9b1a1da029151dacbb1ad49a562d4eb60e80e260";
pub(crate) const PR1B_A_EXPERIMENT_REPORT_SHA256: &str =
    "ac98c83c52df95967a315452517fb2556e82771aa49c5a5b1ab239fa1b3ec4e4";

const PREDICTOR_SOURCE: &str = "per-layer-pregate-transition";
const FROZEN_ADAPTER_NAME: &str = "NVIDIA L4";
const BASE_FANOUTS: [usize; 4] = [1, 2, 4, 8];
const OPTIONAL_FANOUTS: [usize; 2] = [12, 16];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ShadowPhase {
    Warmup,
    Measured,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ShadowObserverError {
    detail: String,
}

impl ShadowObserverError {
    pub(crate) fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for ShadowObserverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for ShadowObserverError {}

impl From<GpuNativeTieredResidencyError> for ShadowObserverError {
    fn from(error: GpuNativeTieredResidencyError) -> Self {
        Self::new(error.to_string())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ShadowObserverConfig {
    pub(crate) num_layers: usize,
    pub(crate) experts_per_layer: usize,
    pub(crate) top_k: usize,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ShadowEventOrdering {
    pub(crate) prediction_frozen_sequence: u64,
    pub(crate) target_probe_sequence: u64,
    pub(crate) prediction_scored_sequence: u64,
    pub(crate) predictor_updated_sequence: u64,
    pub(crate) prediction_frozen_at_monotonic_us: u64,
    pub(crate) target_physical_probe_at_monotonic_us: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ShadowPredictionEvent {
    pub(crate) phase: ShadowPhase,
    pub(crate) run_index: usize,
    pub(crate) completed_token_position: usize,
    pub(crate) target_layer: usize,
    pub(crate) predictor_source: &'static str,
    pub(crate) predictor_score_semantics: &'static str,
    pub(crate) fanout: usize,
    pub(crate) ranked_candidate_global_ids: Vec<u32>,
    pub(crate) ranked_candidate_scores: Vec<u64>,
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
    pub(crate) ordering: ShadowEventOrdering,
}

#[derive(Clone, Debug)]
struct RankedPrediction {
    global_id: u32,
    score: u64,
}

#[derive(Clone, Debug)]
struct PendingPrediction {
    phase: ShadowPhase,
    run_index: usize,
    completed_token_position: usize,
    target_layer: usize,
    ranked: Vec<RankedPrediction>,
    duplicate_prediction_count: u64,
    physical: GpuNativePhysicalLayerShadowSnapshot,
    frozen_sequence: u64,
    frozen_at_us: u64,
    target_probe_sequence: Option<u64>,
    target_probe_at_us: Option<u64>,
}

#[derive(Clone, Debug)]
struct LastRoute {
    completed_token_position: usize,
    layer: usize,
    local_ids: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
struct ActiveRun {
    phase: ShadowPhase,
    run_index: usize,
}

struct ObserverInner {
    origin: Instant,
    sequence: u64,
    active_run: Option<ActiveRun>,
    predictor: PerLayerPreGate,
    last_route: Option<LastRoute>,
    pending: Option<PendingPrediction>,
    events: Vec<ShadowPredictionEvent>,
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

/// CPU-only observer installed explicitly by the separate shadow CLI.
pub(crate) struct GpuNativePrefetchShadowObserver {
    config: ShadowObserverConfig,
    inner: Mutex<ObserverInner>,
}

impl GpuNativePrefetchShadowObserver {
    pub(crate) fn new(config: ShadowObserverConfig) -> Result<Arc<Self>, ShadowObserverError> {
        if config.num_layers == 0
            || config.experts_per_layer == 0
            || config.top_k == 0
            || config.top_k > config.experts_per_layer
        {
            return Err(ShadowObserverError::new(format!(
                "invalid shadow observer geometry: {config:?}"
            )));
        }
        Ok(Arc::new(Self {
            inner: Mutex::new(ObserverInner {
                origin: Instant::now(),
                sequence: 0,
                active_run: None,
                predictor: PerLayerPreGate::new(config.num_layers, 16),
                last_route: None,
                pending: None,
                events: Vec::new(),
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
                "shadow observer begin_run called while another run is active",
            ));
        }
        inner.active_run = Some(ActiveRun { phase, run_index });
        inner.last_route = None;
        inner.pending = None;
        Ok(())
    }

    pub(crate) fn end_run(&self) -> Result<(), ShadowObserverError> {
        let mut inner = self.inner.lock();
        if inner.active_run.take().is_none() {
            return Err(ShadowObserverError::new(
                "shadow observer end_run called without an active run",
            ));
        }
        inner.last_route = None;
        inner.pending = None;
        Ok(())
    }

    /// Capture a conservative CPU timestamp immediately before the submission
    /// containing the target layer's normal physical arena probe.
    pub(crate) fn before_segment(
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
                "pending shadow target run={} position={} layer={} did not match segment position={} first_new_layer={}",
                pending.run_index,
                pending.completed_token_position,
                pending.target_layer,
                completed_token_position,
                first_new_layer,
            )));
        }
        if pending.target_probe_sequence.is_some() {
            return Err(ShadowObserverError::new(
                "shadow target physical probe was marked more than once",
            ));
        }
        pending.target_probe_sequence = Some(sequence);
        pending.target_probe_at_us = Some(now.max(pending.frozen_at_us));
        Ok(())
    }

    /// Score routes that became CPU-visible at the current production
    /// boundary, update the private predictor only after scoring, and freeze a
    /// prediction for the next not-yet-executed layer.
    pub(crate) fn observe_boundary(
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
                "invalid shadow observed layer range {start}..={end}"
            )));
        }
        if let Some(pending) = inner.pending.as_ref() {
            if pending.completed_token_position == completed_token_position
                && pending.target_layer < start
            {
                return Err(ShadowObserverError::new(
                    "shadow prediction target was skipped before scoring",
                ));
            }
        }

        for layer in start..=end {
            let local_ids = selected_ids_by_layer.get(layer).ok_or_else(|| {
                ShadowObserverError::new("missing selected route for shadow layer")
            })?;
            validate_route(local_ids, self.config.top_k, self.config.experts_per_layer)?;

            if inner.pending.as_ref().is_some_and(|pending| {
                pending.completed_token_position == completed_token_position
                    && pending.target_layer == layer
            }) {
                let pending = inner.pending.take().expect("pending target checked");
                let target_physical = residency.shadow_layer_snapshot(layer)?;
                let scored_sequence = inner.next_sequence();
                let updated_sequence = scored_sequence.saturating_add(1);
                let mut events = score_pending(
                    pending,
                    local_ids,
                    target_physical,
                    self.config.experts_per_layer,
                    scored_sequence,
                    updated_sequence,
                )?;
                inner.events.append(&mut events);
            }

            let previous = inner.last_route.clone();
            if let Some(previous) = previous.as_ref() {
                if previous.completed_token_position == completed_token_position
                    && previous.layer + 1 == layer
                {
                    inner.predictor.observe_transition(
                        previous.layer as u32,
                        &previous.local_ids,
                        local_ids,
                    );
                }
            }
            let _update_sequence = inner.next_sequence();
            inner.last_route = Some(LastRoute {
                completed_token_position,
                layer,
                local_ids: local_ids.clone(),
            });
        }

        if end + 1 < self.config.num_layers {
            let source_route = inner.last_route.clone().ok_or_else(|| {
                ShadowObserverError::new("shadow predictor lost the last observed route")
            })?;
            if inner.pending.is_some() {
                return Err(ShadowObserverError::new(
                    "shadow predictor attempted to overwrite an unscored target",
                ));
            }
            inner.pending = Some(freeze_prediction(
                &mut inner,
                active,
                completed_token_position,
                end + 1,
                &source_route,
                residency.shadow_layer_snapshot(end + 1)?,
                self.config.experts_per_layer,
            )?);
        }
        Ok(())
    }

    pub(crate) fn snapshot(&self) -> ShadowObserverSnapshot {
        let inner = self.inner.lock();
        ShadowObserverSnapshot::from_events(inner.events.clone())
    }

    #[cfg(test)]
    fn inject_frozen_for_test(
        &self,
        phase: ShadowPhase,
        run_index: usize,
        position: usize,
        layer: usize,
        ranked_global_ids: &[u32],
        physical: GpuNativePhysicalLayerShadowSnapshot,
    ) {
        let mut inner = self.inner.lock();
        let frozen_sequence = inner.next_sequence();
        let frozen_at_us = inner.monotonic_us();
        let (ranked, duplicates) = dedupe_ranked(
            ranked_global_ids
                .iter()
                .enumerate()
                .map(|(rank, &global_id)| RankedPrediction {
                    global_id,
                    score: (ranked_global_ids.len() - rank) as u64,
                })
                .collect(),
        );
        inner.pending = Some(PendingPrediction {
            phase,
            run_index,
            completed_token_position: position,
            target_layer: layer,
            ranked,
            duplicate_prediction_count: duplicates,
            physical,
            frozen_sequence,
            frozen_at_us,
            target_probe_sequence: None,
            target_probe_at_us: None,
        });
    }
}

fn validate_route(
    local_ids: &[u32],
    top_k: usize,
    experts_per_layer: usize,
) -> Result<(), ShadowObserverError> {
    if local_ids.len() != top_k {
        return Err(ShadowObserverError::new(format!(
            "shadow route contained {} experts, expected {top_k}",
            local_ids.len()
        )));
    }
    let mut seen = HashSet::with_capacity(local_ids.len());
    for &local_id in local_ids {
        if local_id as usize >= experts_per_layer {
            return Err(ShadowObserverError::new(format!(
                "shadow route expert {local_id} is outside 0..{experts_per_layer}"
            )));
        }
        if !seen.insert(local_id) {
            return Err(ShadowObserverError::new(format!(
                "shadow route duplicated expert {local_id}"
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
        .ok_or_else(|| ShadowObserverError::new("shadow global expert id overflow"))
}

fn freeze_prediction(
    inner: &mut ObserverInner,
    active: ActiveRun,
    completed_token_position: usize,
    target_layer: usize,
    source_route: &LastRoute,
    physical: GpuNativePhysicalLayerShadowSnapshot,
    experts_per_layer: usize,
) -> Result<PendingPrediction, ShadowObserverError> {
    if source_route.completed_token_position != completed_token_position
        || source_route.layer + 1 != target_layer
    {
        return Err(ShadowObserverError::new(
            "shadow pre-gate source was not the target layer's causal predecessor",
        ));
    }
    let raw = inner
        .predictor
        .predict_ranked(source_route.layer as u32, &source_route.local_ids, 16)
        .into_iter()
        .map(|candidate| {
            Ok(RankedPrediction {
                global_id: global_id(target_layer, candidate.expert_id, experts_per_layer)?,
                score: candidate.score,
            })
        })
        .collect::<Result<Vec<_>, ShadowObserverError>>()?;
    let (ranked, duplicate_prediction_count) = dedupe_ranked(raw);
    let frozen_sequence = inner.next_sequence();
    let frozen_at_us = inner.monotonic_us();
    Ok(PendingPrediction {
        phase: active.phase,
        run_index: active.run_index,
        completed_token_position,
        target_layer,
        ranked,
        duplicate_prediction_count,
        physical,
        frozen_sequence,
        frozen_at_us,
        target_probe_sequence: None,
        target_probe_at_us: None,
    })
}

fn dedupe_ranked(raw: Vec<RankedPrediction>) -> (Vec<RankedPrediction>, u64) {
    let mut seen = HashSet::with_capacity(raw.len());
    let mut duplicates = 0u64;
    let ranked = raw
        .into_iter()
        .filter(|candidate| {
            if seen.insert(candidate.global_id) {
                true
            } else {
                duplicates = duplicates.saturating_add(1);
                false
            }
        })
        .collect();
    (ranked, duplicates)
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
    pending: PendingPrediction,
    actual_local_ids: &[u32],
    target_physical: GpuNativePhysicalLayerShadowSnapshot,
    experts_per_layer: usize,
    scored_sequence: u64,
    predictor_updated_sequence: u64,
) -> Result<Vec<ShadowPredictionEvent>, ShadowObserverError> {
    let target_probe_sequence = pending.target_probe_sequence.ok_or_else(|| {
        ShadowObserverError::new("shadow target route was observed before its probe marker")
    })?;
    let target_probe_at_us = pending.target_probe_at_us.ok_or_else(|| {
        ShadowObserverError::new("shadow target route was observed without a monotonic probe time")
    })?;
    if !(pending.frozen_sequence < target_probe_sequence
        && target_probe_sequence < scored_sequence
        && scored_sequence < predictor_updated_sequence)
    {
        return Err(ShadowObserverError::new(format!(
            "shadow causal ordering failed: frozen={} probe={} scored={} updated={}",
            pending.frozen_sequence,
            target_probe_sequence,
            scored_sequence,
            predictor_updated_sequence,
        )));
    }
    let actual = actual_local_ids
        .iter()
        .map(|&local_id| global_id(pending.target_layer, local_id, experts_per_layer))
        .collect::<Result<Vec<_>, _>>()?;
    let actual_set = actual.iter().copied().collect::<HashSet<_>>();
    if target_physical.layer_index != pending.target_layer
        || target_physical.slot_capacity != pending.physical.slot_capacity
    {
        return Err(ShadowObserverError::new(format!(
            "shadow target physical geometry drifted: prediction={:?} target={target_physical:?}",
            pending.physical,
        )));
    }
    let target_residents = target_physical
        .resident_global_ids_mru_to_lru
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let prediction_residents = pending
        .physical
        .resident_global_ids_mru_to_lru
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let actual_current = actual
        .iter()
        .copied()
        .filter(|id| target_residents.contains(id))
        .collect::<Vec<_>>();
    let missing = actual
        .iter()
        .copied()
        .filter(|id| !target_residents.contains(id))
        .collect::<Vec<_>>();
    let missing_set = missing.iter().copied().collect::<HashSet<_>>();
    let miss_boundary = !missing.is_empty();
    let lead_time_us = target_probe_at_us.saturating_sub(pending.frozen_at_us);
    let mut events = Vec::new();

    for fanout in fanouts_for_capacity(pending.physical.slot_capacity) {
        let prefix = pending
            .ranked
            .iter()
            .take(fanout)
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
        events.push(ShadowPredictionEvent {
            phase: pending.phase,
            run_index: pending.run_index,
            completed_token_position: pending.completed_token_position,
            target_layer: pending.target_layer,
            predictor_source: PREDICTOR_SOURCE,
            predictor_score_semantics:
                "summed online transition counts; not a calibrated probability",
            fanout,
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
            coverage_upper_bound_boundary_elimination: miss_boundary && exact_missing_set_coverage,
            already_resident_predicted_candidates,
            wrong_predicted_candidates,
            predicted_missing_but_not_selected,
            duplicate_prediction_count: pending.duplicate_prediction_count,
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
            ordering: ShadowEventOrdering {
                prediction_frozen_sequence: pending.frozen_sequence,
                target_probe_sequence,
                prediction_scored_sequence: scored_sequence,
                predictor_updated_sequence,
                prediction_frozen_at_monotonic_us: pending.frozen_at_us,
                target_physical_probe_at_monotonic_us: target_probe_at_us,
            },
        });
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
        if final_residents.contains(&candidate.global_id) {
            continue;
        }
        if free == 0 {
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
        harmful_eviction_event: selected_experts_evicted > 0,
        selected_experts_evicted,
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
    pub(crate) fanout: usize,
    pub(crate) eligible_prediction_events: u64,
    pub(crate) route_candidate_count: u64,
    pub(crate) route_true_positive_count: u64,
    pub(crate) route_precision: f64,
    pub(crate) route_recall: f64,
    pub(crate) exact_selected_top8_coverage_count: u64,
    pub(crate) physical_missing_true_positive_count: u64,
    pub(crate) physical_missing_precision: f64,
    pub(crate) physical_missing_recall: f64,
    pub(crate) exact_missing_set_coverage_count: u64,
    pub(crate) miss_boundary_count: u64,
    pub(crate) coverage_upper_bound_boundary_eliminations: u64,
    pub(crate) coverage_upper_bound_boundary_elimination_rate: f64,
    pub(crate) already_resident_predicted_candidates: u64,
    pub(crate) wrong_predicted_candidates: u64,
    pub(crate) predicted_missing_but_not_selected: u64,
    pub(crate) duplicate_prediction_count: u64,
    pub(crate) prediction_missing_from_physical_at_prediction_time: u64,
    pub(crate) current_policy_installable_count: u64,
    pub(crate) current_policy_full_missing_set_installable_count: u64,
    pub(crate) current_policy_projected_boundary_eliminations: u64,
    pub(crate) current_policy_projected_boundary_elimination_rate: f64,
    pub(crate) counterfactual_replacement_attempts: u64,
    pub(crate) counterfactual_selected_experts_evicted: u64,
    pub(crate) counterfactual_harmful_eviction_events: u64,
    pub(crate) counterfactual_complete_top8_residency_count: u64,
    pub(crate) counterfactual_projected_boundary_eliminations: u64,
    pub(crate) counterfactual_projected_boundary_elimination_rate: f64,
}

impl FanoutMetrics {
    fn empty(predictor_source: &str, fanout: usize) -> Self {
        Self {
            predictor_source: predictor_source.to_string(),
            fanout,
            ..Self::default()
        }
    }

    fn add(&mut self, event: &ShadowPredictionEvent) {
        self.predictor_source = event.predictor_source.to_string();
        self.fanout = event.fanout;
        self.eligible_prediction_events += 1;
        self.route_candidate_count += event.ranked_candidate_global_ids.len() as u64;
        self.route_true_positive_count += event.route_true_positive_count;
        self.exact_selected_top8_coverage_count += u64::from(event.exact_selected_top8_coverage);
        self.physical_missing_true_positive_count += event.physical_missing_true_positive_count;
        self.exact_missing_set_coverage_count += u64::from(event.exact_missing_set_coverage);
        self.miss_boundary_count += u64::from(event.miss_boundary);
        self.coverage_upper_bound_boundary_eliminations +=
            u64::from(event.coverage_upper_bound_boundary_elimination);
        self.already_resident_predicted_candidates += event.already_resident_predicted_candidates;
        self.wrong_predicted_candidates += event.wrong_predicted_candidates;
        self.predicted_missing_but_not_selected += event.predicted_missing_but_not_selected;
        self.duplicate_prediction_count += event.duplicate_prediction_count;
        self.prediction_missing_from_physical_at_prediction_time +=
            event.prediction_missing_from_physical_at_prediction_time;
        self.current_policy_installable_count += event.current_policy_installable_count;
        self.current_policy_full_missing_set_installable_count +=
            u64::from(event.current_policy_full_missing_set_installable);
        self.current_policy_projected_boundary_eliminations +=
            u64::from(event.current_policy_projected_boundary_elimination);
        self.counterfactual_replacement_attempts += event.counterfactual_replacement_attempts;
        self.counterfactual_selected_experts_evicted +=
            event.counterfactual_selected_experts_evicted;
        self.counterfactual_harmful_eviction_events +=
            u64::from(event.counterfactual_harmful_eviction_event);
        self.counterfactual_complete_top8_residency_count +=
            u64::from(event.counterfactual_complete_top8_residency);
        self.counterfactual_projected_boundary_eliminations +=
            u64::from(event.counterfactual_projected_boundary_elimination);
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
    pub(crate) eligible_prediction_events: u64,
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
pub(crate) struct ShadowObserverSnapshot {
    pub(crate) measured: PhaseAggregate,
    pub(crate) warmup: PhaseAggregate,
    pub(crate) events: Vec<ShadowPredictionEvent>,
}

impl ShadowObserverSnapshot {
    fn from_events(events: Vec<ShadowPredictionEvent>) -> Self {
        Self {
            measured: aggregate_phase(&events, ShadowPhase::Measured),
            warmup: aggregate_phase(&events, ShadowPhase::Warmup),
            events,
        }
    }
}

fn aggregate_phase(events: &[ShadowPredictionEvent], phase: ShadowPhase) -> PhaseAggregate {
    let filtered = events
        .iter()
        .filter(|event| event.phase == phase)
        .collect::<Vec<_>>();
    let mut unique_targets = BTreeMap::<(usize, usize, usize), &ShadowPredictionEvent>::new();
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
            .entry((event.predictor_source, event.fanout))
            .or_insert_with(|| FanoutMetrics::empty(event.predictor_source, event.fanout))
            .add(event);
    }
    for ((source, fanout), metrics) in &mut global {
        let relevant = filtered
            .iter()
            .filter(|event| event.predictor_source == *source && event.fanout == *fanout);
        let (actual_selected, actual_missing) = relevant.fold((0u64, 0u64), |acc, event| {
            (
                acc.0 + event.actual_selected_global_ids.len() as u64,
                acc.1 + event.actual_physical_missing_selected_global_ids.len() as u64,
            )
        });
        metrics.finish(actual_selected, actual_missing);
    }
    let metric_keys = global.keys().copied().collect::<Vec<_>>();

    let mut per_layer = Vec::new();
    for layer in 0..48usize.max(
        unique_targets
            .values()
            .map(|event| event.target_layer + 1)
            .max()
            .unwrap_or(0),
    ) {
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
                .entry((event.predictor_source, event.fanout))
                .or_insert_with(|| FanoutMetrics::empty(event.predictor_source, event.fanout))
                .add(event);
        }
        for ((source, fanout), metrics) in &mut layer_fanouts {
            let relevant = layer_events
                .iter()
                .filter(|event| event.predictor_source == *source && event.fanout == *fanout);
            let (actual_selected, actual_missing) = relevant.fold((0u64, 0u64), |acc, event| {
                (
                    acc.0 + event.actual_selected_global_ids.len() as u64,
                    acc.1 + event.actual_physical_missing_selected_global_ids.len() as u64,
                )
            });
            metrics.finish(actual_selected, actual_missing);
        }
        per_layer.push(LayerMetrics {
            target_layer: layer,
            eligible_prediction_events: targets.len() as u64,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub(crate) struct ShadowSideEffectCounters {
    pub(crate) prefetch_requests: u64,
    pub(crate) source_acquisitions: u64,
    pub(crate) ram_reads: u64,
    pub(crate) nvme_reads: u64,
    pub(crate) logical_admissions: u64,
    pub(crate) physical_installs: u64,
    pub(crate) physical_evictions: u64,
    pub(crate) physical_retires: u64,
    pub(crate) generation_changes: u64,
    pub(crate) lru_touches: u64,
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
pub(crate) struct ShadowProductionSemantics {
    pub(crate) inference_math_changed: bool,
    pub(crate) q4_changed: bool,
    pub(crate) router_changed: bool,
    pub(crate) attention_changed: bool,
    pub(crate) rmsnorm_changed: bool,
    pub(crate) lm_head_changed: bool,
    pub(crate) residency_policy_changed: bool,
    pub(crate) replay_policy_changed_relative_to_merged_main: bool,
    pub(crate) prefetch_policy_changed: bool,
    pub(crate) capacity_changed: bool,
    pub(crate) shadow_diagnostic_enabled: bool,
    pub(crate) active_speculative_io: bool,
}

impl ShadowProductionSemantics {
    const fn shadow_only() -> Self {
        Self {
            inference_math_changed: false,
            q4_changed: false,
            router_changed: false,
            attention_changed: false,
            rmsnorm_changed: false,
            lm_head_changed: false,
            residency_policy_changed: false,
            replay_policy_changed_relative_to_merged_main: false,
            prefetch_policy_changed: false,
            capacity_changed: false,
            shadow_diagnostic_enabled: true,
            active_speculative_io: false,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SourceAuditAnswer {
    pub(crate) question: u8,
    pub(crate) answer: &'static str,
    pub(crate) exact_source_locations: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PredictorSourceAudit {
    pub(crate) source: &'static str,
    pub(crate) evaluated: bool,
    pub(crate) causal_input: &'static str,
    pub(crate) disposition: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ShadowRunEvidence {
    pub(crate) phase: ShadowPhase,
    pub(crate) run_index: usize,
    pub(crate) prompt_tokens: usize,
    pub(crate) requested_output_tokens: usize,
    pub(crate) generated_tokens: usize,
    pub(crate) generated_token_ids: Vec<u32>,
    pub(crate) generated_token_ids_sha256: String,
    pub(crate) generated_text_sha256: String,
    pub(crate) production_counters: crate::gpu_native_real_benchmark::RequestSnapshots,
}

impl ShadowRunEvidence {
    fn from_result(
        phase: ShadowPhase,
        result: crate::gpu_native_real_benchmark::PerRunResult,
    ) -> Self {
        Self {
            phase,
            run_index: result.run_index,
            prompt_tokens: result.prompt_tokens,
            requested_output_tokens: result.requested_output_tokens,
            generated_tokens: result.generated_tokens,
            generated_token_ids: result.generated_token_ids,
            generated_token_ids_sha256: result.generated_token_ids_sha256,
            generated_text_sha256: result.generated_text_sha256,
            production_counters: result.counters,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ShadowReport {
    pub(crate) schema: &'static str,
    pub(crate) mode: &'static str,
    pub(crate) source_main_commit: &'static str,
    pub(crate) tested_pr1bb_commit: &'static str,
    pub(crate) tested_pr1bb_report_sha256: &'static str,
    pub(crate) pr1_commit: &'static str,
    pub(crate) original_baseline_commit: &'static str,
    pub(crate) pr1b_a_experiment_commit: &'static str,
    pub(crate) pr1b_a_experiment_report_sha256: &'static str,
    pub(crate) canonical_benchmark_schema_unchanged: &'static str,
    pub(crate) shadow_complete: bool,
    pub(crate) failure: Option<crate::gpu_native_real_benchmark::BenchmarkFailure>,
    pub(crate) qualification_pass: bool,
    pub(crate) performance_claim: bool,
    pub(crate) production_prefetch_enabled: bool,
    pub(crate) shadow_side_effect_postconditions_verified: bool,
    pub(crate) production_speculative_postconditions_verified: bool,
    pub(crate) production_semantics: ShadowProductionSemantics,
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
    pub(crate) score_calibration_applicable: bool,
    pub(crate) score_semantics: &'static str,
    pub(crate) source_audit: Vec<SourceAuditAnswer>,
    pub(crate) predictor_sources: Vec<PredictorSourceAudit>,
    pub(crate) exact_causal_prediction_point: &'static str,
    pub(crate) exact_target_route_observation_point: &'static str,
    pub(crate) exact_physical_missing_observation_point: &'static str,
    pub(crate) layer_zero_prediction_semantics: &'static str,
    pub(crate) lead_time_limitation: &'static str,
    pub(crate) current_policy_model: &'static str,
    pub(crate) counterfactual_model: &'static str,
    pub(crate) counterfactual_limitations: &'static str,
    pub(crate) warmup_learning_semantics: &'static str,
    pub(crate) shadow_side_effect_counters: ShadowSideEffectCounters,
    pub(crate) warmup_run_evidence: Vec<ShadowRunEvidence>,
    pub(crate) measured_run_evidence: Vec<ShadowRunEvidence>,
    pub(crate) shadow_observations: Option<ShadowObserverSnapshot>,
    pub(crate) runtime_shutdown: Option<crate::greedy_parity::BackgroundShutdownEvidence>,
}

fn source_audit() -> Vec<SourceAuditAnswer> {
    vec![
        SourceAuditAnswer {
            question: 1,
            answer: "MER contains a per-layer pre-gate transition table, first/second-order Markov PredictiveLoader, sliding-window locality, per-layer co-fire affinity plus spatial neighbors, a CPU-hidden-state neural speculator, static route profiling/pinning, and union_prefetch fusion. Only a private pre-gate transition table is scored here.",
            exact_source_locations: vec![
                "rust-engine/src/pregate.rs::PerLayerPreGate",
                "rust-engine/src/router.rs::PredictiveLoader",
                "rust-engine/src/router.rs::LocalityMonitor",
                "rust-engine/src/router.rs::LayeredExpertAffinity",
                "rust-engine/src/router.rs::NeuralSpeculator",
                "rust-engine/src/engine.rs::Engine::maybe_apply_static_residency",
                "rust-engine/src/engine.rs::Engine::union_prefetch",
            ],
        },
        SourceAuditAnswer {
            question: 2,
            answer: "At a CPU-visible recovery boundary, the completed route through layer L-1 and all state learned strictly earlier are available. GPU-native hidden state and target-L route/missing data are not available without readback. Locality, affinity, Markov, and pre-gate state updated only through L-1 are causal; the neural source is unavailable because GPU-native hidden state is device-resident.",
            exact_source_locations: vec![
                "rust-engine/src/gpu_native_token_loop.rs::GpuNativeTokenLoop::step_token_unified_inner",
                "rust-engine/src/config.rs::Config::validate",
            ],
        },
        SourceAuditAnswer {
            question: 3,
            answer: "Per-layer pre-gate and Markov transitions can legally nominate layer L from routes through L-1. Locality and affinity can only be causal if snapshotted before L and given an explicit target-layer mapping. This report evaluates the pre-gate source because it preserves target-layer identity and deterministic rank without activating production prefetch.",
            exact_source_locations: vec![
                "rust-engine/src/pregate.rs::PerLayerPreGate::predict_ranked",
                "rust-engine/src/router.rs::PredictiveLoader::predict_next",
            ],
        },
        SourceAuditAnswer {
            question: 4,
            answer: "The GPU boundary report contains selected_ids for every encoded layer. engine::record_gpu_native_actual_routes is called only after the token finishes and is therefore observation/training input, not proof that a prediction existed before a target layer. The target hidden state used by the CPU neural path is also prohibited target information for GPU-native shadow evidence.",
            exact_source_locations: vec![
                "rust-engine/src/gpu_native_token_loop.rs::GpuNativeBoundaryReport::selected_ids",
                "rust-engine/src/gpu_native_token_loop.rs::GpuNativeTokenLoop::step_token_unified_inner",
                "rust-engine/src/engine.rs::Engine::record_gpu_native_actual_routes",
                "rust-engine/src/engine.rs::Engine::moe_step",
            ],
        },
        SourceAuditAnswer {
            question: 5,
            answer: "Production pre-gate prediction is generated by PerLayerPreGate::observe_and_predict and production union candidates by Engine::union_prefetch. Shadow prediction is generated only by freeze_prediction calling the read-only PerLayerPreGate::predict_ranked after an L-1 boundary and before the next segment submission.",
            exact_source_locations: vec![
                "rust-engine/src/pregate.rs::PerLayerPreGate::observe_and_predict",
                "rust-engine/src/engine.rs::Engine::union_prefetch",
                "rust-engine/src/gpu_native_prefetch_shadow.rs::freeze_prediction",
            ],
        },
        SourceAuditAnswer {
            question: 6,
            answer: "The target route is first produced by the normal GPU router/top-k encoding and becomes CPU-visible in the ordinary boundary report readback. The observer sees it only in observe_boundary after that readback.",
            exact_source_locations: vec![
                "rust-engine/src/backend/wgpu_shaders/gpu_native_router.wgsl::router_topk_main",
                "rust-engine/src/backend/gpu_native.rs::GpuNativeQ4Executor::encode_router",
                "rust-engine/src/gpu_native_token_loop.rs::GpuNativeTokenLoop::execute_token_segment_unified",
                "rust-engine/src/gpu_native_prefetch_shadow.rs::GpuNativePrefetchShadowObserver::observe_boundary",
            ],
        },
        SourceAuditAnswer {
            question: 7,
            answer: "Engine::ensure_gpu_native_demand_residency computes physical_current with GpuNativeTieredResidencyManager::has_current_for_demand and derives physical_missing before any source acquisition. The shadow observer copies the same layer residency metadata at the boundary before that service begins.",
            exact_source_locations: vec![
                "rust-engine/src/engine.rs::Engine::ensure_gpu_native_demand_residency",
                "rust-engine/src/gpu_native_residency.rs::GpuNativeTieredResidencyManager::has_current_for_demand",
                "rust-engine/src/gpu_native_residency.rs::GpuNativeTieredResidencyManager::shadow_layer_snapshot",
            ],
        },
        SourceAuditAnswer {
            question: 8,
            answer: "Speculative physical residency is non-evicting. It uses try_lock and drops on lock pressure, and when resident count reaches slot capacity it returns DroppedCapacityOrPressure. Demand residency alone may retire the oldest unprotected physical resident.",
            exact_source_locations: vec![
                "rust-engine/src/gpu_native_residency.rs::GpuNativeTieredResidencyManager::ensure_speculative_resident",
                "rust-engine/src/gpu_native_residency.rs::GpuNativeTieredResidencyManager::ensure_demand_set",
            ],
        },
        SourceAuditAnswer {
            question: 9,
            answer: "Per-layer slot capacity, resident count, free slots, resident global IDs, and MRU-to-LRU metadata order can be copied under the layer metadata lock with LruCache::iter. No payload is acquired, no LRU get/get_mut is called, and no counter changes.",
            exact_source_locations: vec![
                "rust-engine/src/gpu_native_residency.rs::GpuNativeTieredResidencyManager::shadow_layer_snapshot",
            ],
        },
        SourceAuditAnswer {
            question: 10,
            answer: "Production observe_and_predict records the just-observed transition before predicting the following layer, which is valid for that following target but cannot be reused to score the just-observed route. Shadow ordering is stricter and explicit: freeze target-L prediction, mark the target probe, observe/score L, then update the private predictor. Sequence identities fail closed on any inversion.",
            exact_source_locations: vec![
                "rust-engine/src/pregate.rs::PerLayerPreGate::observe_and_predict",
                "rust-engine/src/gpu_native_prefetch_shadow.rs::GpuNativePrefetchShadowObserver::observe_boundary",
                "rust-engine/src/gpu_native_prefetch_shadow.rs::score_pending",
            ],
        },
    ]
}

fn predictor_source_audit() -> Vec<PredictorSourceAudit> {
    vec![
        PredictorSourceAudit {
            source: PREDICTOR_SOURCE,
            evaluated: true,
            causal_input: "selected route at target layer L-1 plus transition counts learned strictly earlier",
            disposition: "evaluated with one deterministic ranked list and fanout prefixes; the private layer-scoped local-ID table is ranking-equivalent to production's layer-qualified global-ID table",
        },
        PredictorSourceAudit {
            source: "predictive-loader-markov",
            evaluated: false,
            causal_input: "last/last-last routed expert IDs",
            disposition: "audited but excluded from v1 scoring to avoid inventing a target-layer mapping distinct from its production global-ID semantics",
        },
        PredictorSourceAudit {
            source: "locality",
            evaluated: false,
            causal_input: "sliding window through L-1",
            disposition: "audited but excluded as a hot-set signal without a faithful target-layer ranked contract",
        },
        PredictorSourceAudit {
            source: "affinity-spatial",
            evaluated: false,
            causal_input: "co-fire state through L-1 plus static disk adjacency",
            disposition: "audited but excluded because production folds it into union_prefetch rather than exposing an independent target-layer ranking",
        },
        PredictorSourceAudit {
            source: "neural-speculator",
            evaluated: false,
            causal_input: "CPU hidden state",
            disposition: "unavailable before a GPU-native target layer without prohibited hidden-state readback",
        },
        PredictorSourceAudit {
            source: "static-residency-profile",
            evaluated: false,
            causal_input: "offline or accumulated route counts",
            disposition: "not an online future-layer ranking and production use would pin/mutate residency",
        },
    ]
}

fn validate_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_zero_speculative_work(
    result: &crate::gpu_native_real_benchmark::PerRunResult,
) -> Result<(), crate::gpu_native_real_benchmark::BenchmarkFailure> {
    let residency = result.counters.gpu_native_residency_delta;
    if residency.speculative_requests != 0
        || residency.speculative_vram_hits != 0
        || residency.speculative_ram_to_vram_installs != 0
        || residency.speculative_dropped_capacity_or_pressure != 0
        || result.counters.engine_storage_delta.prefetch_completed != 0
    {
        return Err(crate::gpu_native_real_benchmark::BenchmarkFailure::new(
            "postcondition",
            "active-speculative-work-observed",
            format!(
                "shadow qualification requires zero production speculative activity and prefetch completion; residency={residency:?} engine_prefetch_completed={}",
                result.counters.engine_storage_delta.prefetch_completed,
            ),
        ));
    }
    Ok(())
}

fn validate_shadow_preflight(
    build: &crate::qualification::BuildProvenance,
    artifacts: &crate::qualification::QualificationArtifacts,
    artifact_errors: &[String],
    metadata: &crate::qualification::ExpertMetadataEvidence,
) -> Result<(), crate::gpu_native_real_benchmark::BenchmarkFailure> {
    if build.dirty != Some(false)
        || build
            .git_sha
            .as_deref()
            .is_none_or(|sha| !validate_hex(sha, 40))
    {
        return Err(crate::gpu_native_real_benchmark::BenchmarkFailure::new(
            "preflight",
            "provenance-unavailable",
            format!(
                "requires clean embedded full Git SHA; observed git_sha={:?} dirty={:?}",
                build.git_sha, build.dirty
            ),
        ));
    }
    if !artifact_errors.is_empty()
        || artifacts.config.is_none()
        || artifacts.tokenizer.is_none()
        || artifacts.expert_metadata.is_none()
        || artifacts.dense_weights_directory.is_none()
    {
        return Err(crate::gpu_native_real_benchmark::BenchmarkFailure::new(
            "preflight",
            "artifact-provenance-unavailable",
            format!("mandatory artifact identity failed: errors={artifact_errors:?} artifacts={artifacts:?}"),
        ));
    }
    if metadata.dtype.as_deref() != Some("q4_0")
        || metadata.q4_0_layout.as_deref() != Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1)
        || metadata.explicitly_synthetic
    {
        return Err(crate::gpu_native_real_benchmark::BenchmarkFailure::new(
            "preflight",
            "invalid-expert-metadata",
            format!("requires canonical nonsynthetic Q4_0 metadata; observed {metadata:?}"),
        ));
    }
    Ok(())
}

fn validate_shadow_runtime(
    runtime: &crate::BenchRealRuntime,
    resolved_config_sha256: &str,
    expected_adapter_name: &str,
) -> Result<
    (
        crate::gpu_native_real_benchmark::RuntimeContractEvidence,
        crate::backend::GpuDeviceIdentity,
        crate::greedy_parity::ModelLoadEvidence,
    ),
    crate::gpu_native_real_benchmark::BenchmarkFailure,
> {
    let observed_config_sha256 = crate::resolved_real_runtime_identity_sha256(
        &runtime.cfg,
        runtime.model.config.architecture,
        runtime.model.config.first_k_dense_replace,
        &runtime.model.config.advanced,
    )
    .map_err(|error| {
        crate::gpu_native_real_benchmark::BenchmarkFailure::new(
            "startup",
            "runtime-config-identity-unavailable",
            error.to_string(),
        )
    })?;
    if observed_config_sha256 != resolved_config_sha256 {
        return Err(crate::gpu_native_real_benchmark::BenchmarkFailure::new(
            "startup",
            "runtime-config-identity-drift",
            format!(
                "runtime identity {observed_config_sha256} differs from preflight identity {resolved_config_sha256}"
            ),
        ));
    }
    let model_load = crate::greedy_parity_model_load(runtime);
    let input = crate::gpu_native_real_benchmark::RuntimeContractInput {
        real_transformer_enabled: runtime.cfg.real_transformer.enabled,
        real_transformer_gpu_native: runtime.cfg.real_transformer.gpu_native,
        compute_offload: runtime.cfg.real_transformer.compute_offload,
        legacy_execution_plan: runtime.engine.execution_context().plan().into(),
        token_loop_geometry: runtime
            .gpu_native_token_loop
            .as_ref()
            .map(|token_loop| token_loop.model_geometry()),
        authoritative_device: runtime.engine.gpu_device_identity(),
        model_load: model_load.clone(),
        routed_failure_policy: runtime.engine.routed_expert_gpu_failure_policy(),
    };
    let (contract, device) =
        crate::gpu_native_real_benchmark::validate_runtime_contract(&input, expected_adapter_name)?;
    let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
        crate::gpu_native_real_benchmark::BenchmarkFailure::new(
            "startup",
            "missing-gpu-native-token-loop",
            "shadow runtime did not construct the authoritative token loop",
        )
    })?;
    if token_loop.snapshot() != crate::gpu_native_token_loop::GpuNativeTokenLoopSnapshot::default()
        || token_loop.recovery_snapshot()
            != crate::gpu_native_token_loop::GpuNativeRecoverySnapshot::default()
        || runtime.engine.routed_expert_execution_snapshot()
            != crate::engine::RoutedExpertExecutionSnapshot::default()
    {
        return Err(crate::gpu_native_real_benchmark::BenchmarkFailure::new(
            "startup",
            "nonzero-initial-runtime-counters",
            "fresh shadow runtime did not start from zero token/recovery/routed counters",
        ));
    }
    Ok((contract, device, model_load))
}

async fn execute_shadow_run(
    runtime: &crate::BenchRealRuntime,
    observer: &Arc<GpuNativePrefetchShadowObserver>,
    phase: ShadowPhase,
    run_index: usize,
    prompt_ids: &[u32],
    output_tokens: usize,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<ShadowRunEvidence, crate::gpu_native_real_benchmark::BenchmarkFailure> {
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
        format!("qualify-gpu-native-prefetch-shadow {phase_label} run {run_index}"),
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
            "shadow-request-failed",
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
    validate_zero_speculative_work(&result)?;
    Ok(ShadowRunEvidence::from_result(phase, result))
}

fn emit_report(
    report: &ShadowReport,
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
            "GPU-native prefetch shadow report written to {}",
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
            "qualify-gpu-native-prefetch-shadow requires the explicit --greedy flag",
        )
        .into());
    }
    if args.measured_runs == 0 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "measured-runs-required",
            "qualify-gpu-native-prefetch-shadow requires --measured-runs > 0",
        )
        .into());
    }
    if args.cache_reset != crate::BenchRealCacheReset::Keep {
        return Err(BenchmarkFailure::new(
            "preflight",
            "cache-reset-contract",
            "shadow schema v1 supports only the frozen --cache-reset keep online-learning schedule",
        )
        .into());
    }
    if args.expected_adapter_name != FROZEN_ADAPTER_NAME {
        return Err(BenchmarkFailure::new(
            "preflight",
            "frozen-adapter-required",
            format!(
                "shadow schema v1 requires --expected-adapter-name {FROZEN_ADAPTER_NAME:?}; observed {:?}",
                args.expected_adapter_name
            ),
        )
        .into());
    }
    let request_input = crate::load_real_cli_request_input(
        "qualify-gpu-native-prefetch-shadow",
        args.prompt.as_ref(),
        args.request_json.as_deref(),
        args.output_tokens,
    )?;
    if request_input.output_tokens < 2 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "insufficient-output-tokens",
            "qualify-gpu-native-prefetch-shadow requires --output-tokens >= 2",
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
    validate_shadow_preflight(&build, &artifacts, &artifact_errors, &expert_metadata)?;
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
    let mut report = ShadowReport {
        schema: SCHEMA,
        mode: MODE,
        source_main_commit: SOURCE_MAIN_COMMIT,
        tested_pr1bb_commit: TESTED_PR1BB_COMMIT,
        tested_pr1bb_report_sha256: TESTED_PR1BB_REPORT_SHA256,
        pr1_commit: PR1_COMMIT,
        original_baseline_commit: ORIGINAL_BASELINE_COMMIT,
        pr1b_a_experiment_commit: PR1B_A_EXPERIMENT_COMMIT,
        pr1b_a_experiment_report_sha256: PR1B_A_EXPERIMENT_REPORT_SHA256,
        canonical_benchmark_schema_unchanged:
            crate::gpu_native_real_benchmark::SCHEMA,
        shadow_complete: false,
        failure: None,
        qualification_pass: false,
        performance_claim: false,
        production_prefetch_enabled: false,
        shadow_side_effect_postconditions_verified: false,
        production_speculative_postconditions_verified: false,
        production_semantics: ShadowProductionSemantics::shadow_only(),
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
        score_calibration_applicable: false,
        score_semantics: "summed online transition counts; not a calibrated probability",
        source_audit: source_audit(),
        predictor_sources: predictor_source_audit(),
        exact_causal_prediction_point: "GpuNativePrefetchShadowObserver::observe_boundary -> freeze_prediction after route L-1 becomes CPU-visible and before the next segment submission",
        exact_target_route_observation_point: "GpuNativeTokenLoop::step_token_unified_inner after the ordinary boundary-report readback and before demand residency service",
        exact_physical_missing_observation_point: "read-only target-layer physical snapshot immediately before Engine::ensure_gpu_native_demand_residency; equivalent to its has_current_for_demand set computation because no target-layer mutation intervenes",
        layer_zero_prediction_semantics: "unscored: the evaluated per-layer pre-gate transition source has no legal layer predecessor and v1 does not invent a cross-token layer-0 predictor",
        lead_time_limitation: "prediction freeze and conservative target-probe markers are monotonic CPU timestamps; no GPU timestamp or shader instrumentation is added, so timing is diagnostic only",
        current_policy_model: "ranked nonresident candidates consume only free target-layer physical slots; a full arena installs zero because production speculative residency does not evict",
        counterfactual_model: "ORACLE / SHADOW ONLY: copy prediction-time MRU-to-LRU IDs, process candidates in predictor rank order, promote copied hits, fill free slots, then evict the copied LRU tail for nonresident candidates",
        counterfactual_limitations: "the oracle knows the later selected top-8 only when scoring harmful evictions; it models metadata capacity and LRU ordering, not I/O completion, contention, bandwidth, install latency, or generation races",
        warmup_learning_semantics: "one shared private predictor spans the keep-cache schedule; warmup trains it causally, warmup events remain separate, and measured events use only state learned earlier in execution order",
        shadow_side_effect_counters: ShadowSideEffectCounters::default(),
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
    let validation = validate_shadow_runtime(
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
    let observer = GpuNativePrefetchShadowObserver::new(ShadowObserverConfig {
        num_layers: geometry.num_layers,
        experts_per_layer: geometry.num_experts,
        top_k: geometry.top_k,
    })?;
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
                "no target layer became eligible for a causal shadow prediction",
            ));
        }
        report.fanouts = snapshot
            .events
            .iter()
            .map(|event| event.fanout)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
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
                    "shadow qualification did not retain every requested run",
                );
                report.failure = Some(failure.clone());
                emit_report(&report, args.report_out.as_deref())?;
                return Err(failure.into());
            }
            report.shadow_side_effect_postconditions_verified = true;
            report.production_speculative_postconditions_verified = true;
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

    fn physical(residents: &[u32], capacity: usize) -> GpuNativePhysicalLayerShadowSnapshot {
        GpuNativePhysicalLayerShadowSnapshot {
            layer_index: 1,
            slot_capacity: capacity,
            resident_global_ids_mru_to_lru: residents.to_vec(),
            free_slots: capacity.saturating_sub(residents.len()),
        }
    }

    fn observer() -> Arc<GpuNativePrefetchShadowObserver> {
        GpuNativePrefetchShadowObserver::new(ShadowObserverConfig {
            num_layers: 48,
            experts_per_layer: 128,
            top_k: 8,
        })
        .unwrap()
    }

    fn synthetic_event(
        phase: ShadowPhase,
        run_index: usize,
        position: usize,
        layer: usize,
        missing_count: usize,
        fanout: usize,
    ) -> ShadowPredictionEvent {
        let selected = (0..8)
            .map(|local| (layer * 128 + local) as u32)
            .collect::<Vec<_>>();
        let split = 8usize.saturating_sub(missing_count.min(8));
        let current = selected[..split].to_vec();
        let missing = selected[split..].to_vec();
        ShadowPredictionEvent {
            phase,
            run_index,
            completed_token_position: position,
            target_layer: layer,
            predictor_source: PREDICTOR_SOURCE,
            predictor_score_semantics: "counts",
            fanout,
            ranked_candidate_global_ids: selected.iter().copied().take(fanout).collect(),
            ranked_candidate_scores: vec![1; fanout.min(8)],
            actual_selected_global_ids: selected,
            actual_physical_current_selected_global_ids: current,
            actual_physical_missing_selected_global_ids: missing,
            physical_slot_capacity_at_prediction: 8,
            resident_physical_count_at_prediction: split,
            free_physical_slots_at_prediction: 8 - split,
            resident_global_ids_mru_to_lru_at_prediction: vec![],
            route_true_positive_count: fanout.min(8) as u64,
            physical_missing_true_positive_count: 0,
            exact_selected_top8_coverage: fanout >= 8,
            exact_missing_set_coverage: missing_count == 0,
            miss_boundary: missing_count > 0,
            coverage_upper_bound_boundary_elimination: false,
            already_resident_predicted_candidates: 0,
            wrong_predicted_candidates: 0,
            predicted_missing_but_not_selected: 0,
            duplicate_prediction_count: 0,
            prediction_missing_from_physical_at_prediction_time: 0,
            current_policy_installable_count: 0,
            current_policy_full_missing_set_installable: missing_count == 0,
            current_policy_projected_boundary_elimination: false,
            counterfactual_replacement_attempts: 0,
            counterfactual_selected_experts_evicted: 0,
            counterfactual_harmful_eviction_event: false,
            counterfactual_complete_top8_residency: false,
            counterfactual_projected_boundary_elimination: false,
            lead_time_us: (position + layer + 1) as u64,
            ordering: ShadowEventOrdering {
                prediction_frozen_sequence: 1,
                target_probe_sequence: 2,
                prediction_scored_sequence: 3,
                predictor_updated_sequence: 4,
                prediction_frozen_at_monotonic_us: 1,
                target_physical_probe_at_monotonic_us: 2,
            },
        }
    }

    #[test]
    fn inactive_shadow_observer_records_zero_observations() {
        let obs = observer();
        obs.before_segment(0, 0).unwrap();
        let snapshot = obs.snapshot();
        assert!(snapshot.events.is_empty());
        assert_eq!(snapshot.warmup.target_event_count, 0);
        assert_eq!(snapshot.measured.target_event_count, 0);
    }

    #[test]
    fn duplicate_ranking_retains_first_occurrence_deterministically() {
        let (ranked, duplicates) = dedupe_ranked(vec![
            RankedPrediction {
                global_id: 7,
                score: 9,
            },
            RankedPrediction {
                global_id: 2,
                score: 8,
            },
            RankedPrediction {
                global_id: 7,
                score: 1,
            },
        ]);
        assert_eq!(duplicates, 1);
        assert_eq!(
            ranked.iter().map(|x| x.global_id).collect::<Vec<_>>(),
            vec![7, 2]
        );
        assert_eq!(ranked[0].score, 9);
    }

    #[test]
    fn full_arena_current_policy_installs_nothing() {
        let sim = simulate_current_policy(
            &physical(&[128, 129, 130, 131, 132, 133, 134, 135], 8),
            &[RankedPrediction {
                global_id: 136,
                score: 1,
            }],
        );
        assert_eq!(sim.installable_count, 0);
        assert!(!sim.final_residents.contains(&136));
    }

    #[test]
    fn free_slot_current_policy_is_exact() {
        let sim = simulate_current_policy(
            &physical(&[128, 129, 130, 131, 132, 133], 8),
            &[
                RankedPrediction {
                    global_id: 134,
                    score: 2,
                },
                RankedPrediction {
                    global_id: 135,
                    score: 1,
                },
            ],
        );
        assert_eq!(sim.installable_count, 2);
        assert!(sim.final_residents.contains(&134));
        assert!(sim.final_residents.contains(&135));
    }

    #[test]
    fn replacement_simulation_detects_harmful_lru_victim_without_mutating_input() {
        let snapshot = physical(&[128, 129, 130, 131, 132, 133, 134, 135], 8);
        let before = snapshot.clone();
        let actual = [128, 129, 130, 131, 132, 133, 134, 135]
            .into_iter()
            .collect::<HashSet<_>>();
        let sim = simulate_counterfactual_replacement(
            &snapshot,
            &[RankedPrediction {
                global_id: 136,
                score: 1,
            }],
            &actual,
        );
        assert!(sim.harmful_eviction_event);
        assert_eq!(sim.selected_experts_evicted, 1);
        assert_eq!(snapshot, before);
    }

    #[test]
    fn fanout_prefixes_come_from_one_ranking() {
        let ranking = (128..144)
            .rev()
            .map(|global_id| RankedPrediction {
                global_id,
                score: global_id as u64,
            })
            .collect::<Vec<_>>();
        let prefixes = fanouts_for_capacity(8)
            .into_iter()
            .map(|fanout| {
                ranking
                    .iter()
                    .take(fanout)
                    .map(|x| x.global_id)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            prefixes.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![1, 2, 4, 8]
        );
        assert_eq!(&prefixes[3][..4], &prefixes[2]);
    }

    #[test]
    fn scoring_requires_complete_missing_coverage_for_elimination() {
        let obs = observer();
        obs.begin_run(ShadowPhase::Measured, 0).unwrap();
        obs.inject_frozen_for_test(
            ShadowPhase::Measured,
            0,
            0,
            1,
            &[128, 129, 130, 131, 132, 133, 134, 140],
            physical(&[128, 129, 130, 131, 132, 133], 8),
        );
        obs.before_segment(0, 1).unwrap();
        let pending = obs.inner.lock().pending.take().unwrap();
        let scored = score_pending(
            pending,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            physical(&[128, 129, 130, 131, 132, 133], 8),
            128,
            3,
            4,
        )
        .unwrap();
        let fanout8 = scored.iter().find(|event| event.fanout == 8).unwrap();
        assert!(!fanout8.exact_missing_set_coverage);
        assert!(!fanout8.coverage_upper_bound_boundary_elimination);
    }

    #[test]
    fn exact_selected_top8_coverage_requires_all_eight_ids() {
        let score = |ranked: &[u32]| {
            let obs = observer();
            obs.begin_run(ShadowPhase::Measured, 0).unwrap();
            obs.inject_frozen_for_test(ShadowPhase::Measured, 0, 0, 1, ranked, physical(&[], 8));
            obs.before_segment(0, 1).unwrap();
            let pending = obs.inner.lock().pending.take().unwrap();
            score_pending(
                pending,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                physical(&[], 8),
                128,
                3,
                4,
            )
            .unwrap()
        };
        let complete = score(&[128, 129, 130, 131, 132, 133, 134, 135]);
        let partial = score(&[128, 129, 130, 131, 132, 133, 134, 140]);
        assert!(
            complete
                .iter()
                .find(|event| event.fanout == 8)
                .unwrap()
                .exact_selected_top8_coverage
        );
        assert!(
            !partial
                .iter()
                .find(|event| event.fanout == 8)
                .unwrap()
                .exact_selected_top8_coverage
        );
    }

    #[test]
    fn one_missing_expert_requires_that_exact_expert() {
        let make = |ranked: &[u32]| {
            let obs = observer();
            obs.begin_run(ShadowPhase::Measured, 0).unwrap();
            obs.inject_frozen_for_test(
                ShadowPhase::Measured,
                0,
                0,
                1,
                ranked,
                physical(&[128, 129, 130, 131, 132, 133, 134], 8),
            );
            obs.before_segment(0, 1).unwrap();
            let pending = obs.inner.lock().pending.take().unwrap();
            score_pending(
                pending,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                physical(&[128, 129, 130, 131, 132, 133, 134], 8),
                128,
                3,
                4,
            )
            .unwrap()
        };
        let hit = make(&[135]);
        let miss = make(&[140]);
        assert!(hit[0].coverage_upper_bound_boundary_elimination);
        assert!(!miss[0].coverage_upper_bound_boundary_elimination);
    }

    #[test]
    fn zero_missing_target_is_not_a_miss_boundary() {
        let obs = observer();
        obs.begin_run(ShadowPhase::Measured, 0).unwrap();
        obs.inject_frozen_for_test(
            ShadowPhase::Measured,
            0,
            0,
            1,
            &[128],
            physical(&[128, 129, 130, 131, 132, 133, 134, 135], 8),
        );
        obs.before_segment(0, 1).unwrap();
        let pending = obs.inner.lock().pending.take().unwrap();
        let scored = score_pending(
            pending,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            physical(&[128, 129, 130, 131, 132, 133, 134, 135], 8),
            128,
            3,
            4,
        )
        .unwrap();
        assert!(scored.iter().all(|event| !event.miss_boundary));
        assert!(scored
            .iter()
            .all(|event| !event.coverage_upper_bound_boundary_elimination));
    }

    #[test]
    fn monotonic_lead_time_and_ordering_are_nonnegative() {
        let obs = observer();
        obs.begin_run(ShadowPhase::Measured, 0).unwrap();
        obs.inject_frozen_for_test(ShadowPhase::Measured, 0, 0, 1, &[128], physical(&[], 8));
        obs.before_segment(0, 1).unwrap();
        let pending = obs.inner.lock().pending.take().unwrap();
        let scored = score_pending(
            pending,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            physical(&[], 8),
            128,
            3,
            4,
        )
        .unwrap();
        assert!(scored.iter().all(|event| {
            event.ordering.prediction_frozen_sequence < event.ordering.target_probe_sequence
                && event.ordering.target_probe_sequence < event.ordering.prediction_scored_sequence
                && event.ordering.prediction_scored_sequence
                    < event.ordering.predictor_updated_sequence
        }));
    }

    #[test]
    fn warmup_is_excluded_from_measured_aggregate() {
        let mut event = ShadowPredictionEvent {
            phase: ShadowPhase::Warmup,
            run_index: 0,
            completed_token_position: 0,
            target_layer: 1,
            predictor_source: PREDICTOR_SOURCE,
            predictor_score_semantics: "counts",
            fanout: 1,
            ranked_candidate_global_ids: vec![],
            ranked_candidate_scores: vec![],
            actual_selected_global_ids: vec![128; 8],
            actual_physical_current_selected_global_ids: vec![],
            actual_physical_missing_selected_global_ids: vec![128; 8],
            physical_slot_capacity_at_prediction: 8,
            resident_physical_count_at_prediction: 0,
            free_physical_slots_at_prediction: 8,
            resident_global_ids_mru_to_lru_at_prediction: vec![],
            route_true_positive_count: 0,
            physical_missing_true_positive_count: 0,
            exact_selected_top8_coverage: false,
            exact_missing_set_coverage: false,
            miss_boundary: true,
            coverage_upper_bound_boundary_elimination: false,
            already_resident_predicted_candidates: 0,
            wrong_predicted_candidates: 0,
            predicted_missing_but_not_selected: 0,
            duplicate_prediction_count: 0,
            prediction_missing_from_physical_at_prediction_time: 0,
            current_policy_installable_count: 0,
            current_policy_full_missing_set_installable: false,
            current_policy_projected_boundary_elimination: false,
            counterfactual_replacement_attempts: 0,
            counterfactual_selected_experts_evicted: 0,
            counterfactual_harmful_eviction_event: false,
            counterfactual_complete_top8_residency: false,
            counterfactual_projected_boundary_elimination: false,
            lead_time_us: 1,
            ordering: ShadowEventOrdering {
                prediction_frozen_sequence: 1,
                target_probe_sequence: 2,
                prediction_scored_sequence: 3,
                predictor_updated_sequence: 4,
                prediction_frozen_at_monotonic_us: 1,
                target_physical_probe_at_monotonic_us: 2,
            },
        };
        let warmup = event.clone();
        event.phase = ShadowPhase::Measured;
        event.run_index = 1;
        let snapshot = ShadowObserverSnapshot::from_events(vec![warmup, event]);
        assert_eq!(snapshot.warmup.target_event_count, 1);
        assert_eq!(snapshot.measured.target_event_count, 1);
    }

    #[test]
    fn missing_count_histogram_has_exact_zero_through_eight_buckets() {
        let events = (0..=8)
            .map(|missing| synthetic_event(ShadowPhase::Measured, 0, missing, 1, missing, 8))
            .collect::<Vec<_>>();
        let snapshot = ShadowObserverSnapshot::from_events(events);
        assert_eq!(
            snapshot.measured.missing_set_histogram.counts_0_through_8,
            [1; 9]
        );
        assert_eq!(
            snapshot.measured.per_layer[1]
                .missing_set_histogram
                .counts_0_through_8,
            [1; 9]
        );
    }

    #[test]
    fn per_layer_target_counts_sum_to_global() {
        let snapshot = ShadowObserverSnapshot::from_events(vec![
            synthetic_event(ShadowPhase::Measured, 0, 0, 1, 2, 8),
            synthetic_event(ShadowPhase::Measured, 0, 0, 2, 3, 8),
            synthetic_event(ShadowPhase::Measured, 1, 0, 2, 1, 8),
        ]);
        assert_eq!(
            snapshot
                .measured
                .per_layer
                .iter()
                .map(|layer| layer.eligible_prediction_events)
                .sum::<u64>(),
            snapshot.measured.target_event_count
        );
        assert_eq!(snapshot.measured.target_event_count, 3);
        assert_eq!(snapshot.measured.per_layer.len(), 48);
        assert_eq!(snapshot.measured.per_layer[0].eligible_prediction_events, 0);
        assert_eq!(snapshot.measured.per_layer[0].fanouts.len(), 1);
        assert_eq!(snapshot.measured.per_layer[0].fanouts[0].fanout, 8);
    }

    #[test]
    fn shadow_side_effect_counters_are_structurally_zero() {
        assert_eq!(
            ShadowSideEffectCounters::default(),
            ShadowSideEffectCounters {
                prefetch_requests: 0,
                source_acquisitions: 0,
                ram_reads: 0,
                nvme_reads: 0,
                logical_admissions: 0,
                physical_installs: 0,
                physical_evictions: 0,
                physical_retires: 0,
                generation_changes: 0,
                lru_touches: 0,
            }
        );
    }

    #[test]
    fn canonical_benchmark_v4_schema_is_unchanged() {
        assert_eq!(
            crate::gpu_native_real_benchmark::SCHEMA,
            "mer.gpu-native-real-benchmark.v4"
        );
        assert_eq!(SCHEMA, "mer.gpu-native-prefetch-shadow.v1");
    }

    #[test]
    fn cli_parses_frozen_shadow_qualification_surface() {
        use clap::Parser as _;

        let cli = crate::Cli::try_parse_from([
            "micro-expert-router",
            "qualify-gpu-native-prefetch-shadow",
            "--config",
            "config.toml",
            "--prompt",
            "Write a Rust function that adds two i32 values and returns the result.",
            "--output-tokens",
            "16",
            "--warmup-runs",
            "1",
            "--measured-runs",
            "3",
            "--cache-reset",
            "keep",
            "--greedy",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--report-out",
            "shadow.json",
        ])
        .unwrap();
        let crate::Cmd::QualifyGpuNativePrefetchShadow {
            output_tokens,
            warmup_runs,
            measured_runs,
            cache_reset,
            greedy,
            expected_adapter_name,
            report_out,
            ..
        } = cli.cmd
        else {
            panic!("expected qualify-gpu-native-prefetch-shadow command")
        };
        assert_eq!(output_tokens, Some(16));
        assert_eq!(warmup_runs, 1);
        assert_eq!(measured_runs, 3);
        assert_eq!(cache_reset, crate::BenchRealCacheReset::Keep);
        assert!(greedy);
        assert_eq!(expected_adapter_name, "NVIDIA L4");
        assert_eq!(report_out, Some(std::path::PathBuf::from("shadow.json")));
    }
}
