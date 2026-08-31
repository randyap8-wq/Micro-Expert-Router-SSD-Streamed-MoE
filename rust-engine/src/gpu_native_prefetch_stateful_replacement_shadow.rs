//! PR1C-C CPU-only stateful predictive replacement shadow qualification.
//!
//! The production GPU-native runtime remains unchanged. Each policy below
//! owns an independent ID/metadata-only physical-residency model. A complete
//! model-wide production snapshot is cloned exactly once, at the request's
//! first eligible prediction point. Later production snapshots are used only
//! to measure divergence; they are never copied into a running simulation.

use crate::gpu_native_prefetch_shadow::{
    GpuNativePrefetchShadowCallbacks, ObserverRuntimeGuardEvidence, ShadowObserverCallbackKind,
    ShadowObserverError, ShadowPhase,
};
use crate::gpu_native_residency::{
    GpuNativePhysicalLayerShadowSnapshot, GpuNativeTieredResidencyManager,
};
use crate::router::PredictiveLoader;
use parking_lot::Mutex;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

pub(crate) const SCHEMA: &str = "mer.gpu-native-prefetch-stateful-replacement-shadow.v3";
pub(crate) const MODE: &str = "gpu-native-prefetch-stateful-replacement-shadow";
pub(crate) const SOURCE_MAIN_COMMIT: &str = "e8b542110693e74aa8f1013bb16d1bed0bdd8ba7";
pub(crate) const TESTED_PR1BB_COMMIT: &str = "c7006c5c6fbcee74c91526f89a8c2b5b06d8a9c5";
pub(crate) const TESTED_PR1BB_REPORT_SHA256: &str =
    "82317ad85ba29da1401abf3041ebbb7c2ca72c039c1be0c5f2ea53df9e9fc529";
pub(crate) const PR1CA_COMMIT: &str = "5d1e16ad6f2d8c71809a8fb65b1fbb3e4c98972f";
pub(crate) const PR1CA_REPORT_SHA256: &str =
    "cd7d86d9c9ff6691c11320db9d5e44dc26d893a6e14a939a31ed6ab67b270830";
pub(crate) const PR1CA_LOG_SHA256: &str =
    "de20495fdf4cf47db0264e4ad547aa17ee83349dceab7c89f2438487c535c727";
pub(crate) const PR1CB_COMMIT: &str = "7f313a5777b5cdb7394bcade1882abab54f0d797";
pub(crate) const PR1CB_REPORT_SHA256: &str =
    "bca66b00ecb54e815aa5a5c5fdd3f6ae1317ae267f96b39484cb41e11c300811";
pub(crate) const PR1CB_LOG_SHA256: &str =
    "71156ef13632bdc70b2040d27fa203821fa51553f6dbc4ca9e124e52874dffec";

const PRIMARY_PREDICTOR: &str = "predictive-loader-second-order";
const PRIMARY_FANOUT: usize = 8;
const MAX_RANKED_CANDIDATES: usize = 8;
const MARKOV_SEED: u64 = 0xC0FFEE;
const FROZEN_ADAPTER_NAME: &str = "NVIDIA L4";

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum ReplacementPolicy {
    PhysicalLru,
    PhysicalLruPredictionProtected,
    RouteRecency,
    RouteRecencyPredictionProtected,
}

impl ReplacementPolicy {
    const ALL: [Self; 4] = [
        Self::PhysicalLru,
        Self::PhysicalLruPredictionProtected,
        Self::RouteRecency,
        Self::RouteRecencyPredictionProtected,
    ];

    const fn route_recency(self) -> bool {
        matches!(
            self,
            Self::RouteRecency | Self::RouteRecencyPredictionProtected
        )
    }

    const fn prediction_protected(self) -> bool {
        matches!(
            self,
            Self::PhysicalLruPredictionProtected | Self::RouteRecencyPredictionProtected
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct StatefulObserverConfig {
    pub(crate) num_layers: usize,
    pub(crate) experts_per_layer: usize,
    pub(crate) top_k: usize,
    pub(crate) markov_min_prob: f64,
}

#[derive(Clone, Debug)]
struct RankedPrediction {
    global_id: u32,
    score: f64,
}

#[derive(Clone, Debug)]
struct RouteHistory {
    completed_token_position: usize,
    layer: usize,
    global_ids: Vec<u32>,
}

#[derive(Clone, Debug)]
struct ResidentEntry {
    speculative_install_boundary: Option<u64>,
    speculative_install_used: bool,
    last_physical_touch_boundary: Option<u64>,
}

#[derive(Clone, Debug)]
struct HypotheticalLayerState {
    capacity: usize,
    resident_mru_to_lru: Vec<u32>,
    entries: HashMap<u32, ResidentEntry>,
    route_last_seen_clock: HashMap<u32, u64>,
}

impl HypotheticalLayerState {
    fn from_snapshot(
        snapshot: &GpuNativePhysicalLayerShadowSnapshot,
        experts_per_layer: usize,
    ) -> Result<Self, ShadowObserverError> {
        if snapshot.resident_global_ids_mru_to_lru.len() > snapshot.slot_capacity
            || snapshot.free_slots
                != snapshot
                    .slot_capacity
                    .saturating_sub(snapshot.resident_global_ids_mru_to_lru.len())
        {
            return Err(ShadowObserverError::new(format!(
                "stateful initialization capacity mismatch: {snapshot:?}"
            )));
        }
        let mut entries = HashMap::with_capacity(snapshot.resident_global_ids_mru_to_lru.len());
        for &global_id in &snapshot.resident_global_ids_mru_to_lru {
            let actual_layer = global_id as usize / experts_per_layer;
            if actual_layer != snapshot.layer_index {
                return Err(ShadowObserverError::new(format!(
                    "stateful initialization layer {} contains global expert {global_id} from layer {actual_layer}",
                    snapshot.layer_index
                )));
            }
            if entries
                .insert(
                    global_id,
                    ResidentEntry {
                        speculative_install_boundary: None,
                        speculative_install_used: false,
                        last_physical_touch_boundary: None,
                    },
                )
                .is_some()
            {
                return Err(ShadowObserverError::new(format!(
                    "stateful initialization duplicated resident {global_id}"
                )));
            }
        }
        Ok(Self {
            capacity: snapshot.slot_capacity,
            resident_mru_to_lru: snapshot.resident_global_ids_mru_to_lru.clone(),
            entries,
            route_last_seen_clock: HashMap::new(),
        })
    }

    fn contains(&self, global_id: u32) -> bool {
        self.entries.contains_key(&global_id)
    }

    fn touch_physical(&mut self, global_id: u32, boundary: u64) {
        let index = self
            .resident_mru_to_lru
            .iter()
            .position(|&id| id == global_id)
            .expect("resident entry and physical order remain consistent");
        let id = self.resident_mru_to_lru.remove(index);
        self.resident_mru_to_lru.insert(0, id);
        self.entries
            .get_mut(&global_id)
            .expect("resident entry remains present")
            .last_physical_touch_boundary = Some(boundary);
    }

    fn remove(&mut self, global_id: u32) -> ResidentEntry {
        let index = self
            .resident_mru_to_lru
            .iter()
            .position(|&id| id == global_id)
            .expect("victim entry and physical order remain consistent");
        self.resident_mru_to_lru.remove(index);
        self.entries
            .remove(&global_id)
            .expect("victim entry remains present")
    }

    fn install(&mut self, global_id: u32, entry: ResidentEntry) {
        assert!(self.resident_mru_to_lru.len() < self.capacity);
        assert!(!self.entries.contains_key(&global_id));
        self.resident_mru_to_lru.insert(0, global_id);
        self.entries.insert(global_id, entry);
    }

    fn record_route(&mut self, global_ids: &[u32], clock: u64) {
        for &global_id in global_ids {
            self.route_last_seen_clock.insert(global_id, clock);
        }
    }

    fn resident_set(&self) -> HashSet<u32> {
        self.resident_mru_to_lru.iter().copied().collect()
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct VictimAgeDistribution {
    pub(crate) never_observed_or_touched: u64,
    pub(crate) age_0: u64,
    pub(crate) age_1: u64,
    pub(crate) age_2_to_3: u64,
    pub(crate) age_4_to_7: u64,
    pub(crate) age_8_to_15: u64,
    pub(crate) age_16_to_31: u64,
    pub(crate) age_32_plus: u64,
}

impl VictimAgeDistribution {
    fn record(&mut self, age: Option<u64>) {
        match age {
            None => self.never_observed_or_touched += 1,
            Some(0) => self.age_0 += 1,
            Some(1) => self.age_1 += 1,
            Some(2..=3) => self.age_2_to_3 += 1,
            Some(4..=7) => self.age_4_to_7 += 1,
            Some(8..=15) => self.age_8_to_15 += 1,
            Some(16..=31) => self.age_16_to_31 += 1,
            Some(_) => self.age_32_plus += 1,
        }
    }

    fn add(&mut self, other: &Self) {
        self.never_observed_or_touched += other.never_observed_or_touched;
        self.age_0 += other.age_0;
        self.age_1 += other.age_1;
        self.age_2_to_3 += other.age_2_to_3;
        self.age_4_to_7 += other.age_4_to_7;
        self.age_8_to_15 += other.age_8_to_15;
        self.age_16_to_31 += other.age_16_to_31;
        self.age_32_plus += other.age_32_plus;
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct SpeculationDelta {
    speculative_candidate_count: u64,
    speculative_already_resident_hits: u64,
    speculative_installs: u64,
    speculative_replacements: u64,
    speculative_evictions: u64,
    prediction_protected_victim_skips: u64,
    forced_prediction_protected_evictions: u64,
    physical_lru_victim_age: VictimAgeDistribution,
    route_recency_victim_age: VictimAgeDistribution,
    resident_set_divergence_from_actual: u64,
    state_churn: u64,
    installed_global_ids: Vec<u32>,
    evicted_global_ids: Vec<u32>,
}

#[derive(Clone, Debug)]
struct HypotheticalPolicyState {
    policy: ReplacementPolicy,
    layers: Vec<HypotheticalLayerState>,
    route_clock: u64,
    absent_due_to_hypothetical_eviction: HashMap<u32, u64>,
    resynchronization_count: u64,
}

impl HypotheticalPolicyState {
    fn from_snapshots(
        policy: ReplacementPolicy,
        snapshots: &[GpuNativePhysicalLayerShadowSnapshot],
        experts_per_layer: usize,
    ) -> Result<Self, ShadowObserverError> {
        let layers = snapshots
            .iter()
            .enumerate()
            .map(|(expected_layer, snapshot)| {
                if snapshot.layer_index != expected_layer {
                    return Err(ShadowObserverError::new(format!(
                        "stateful initialization expected layer {expected_layer}, observed {}",
                        snapshot.layer_index
                    )));
                }
                HypotheticalLayerState::from_snapshot(snapshot, experts_per_layer)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            policy,
            layers,
            route_clock: 0,
            absent_due_to_hypothetical_eviction: HashMap::new(),
            resynchronization_count: 0,
        })
    }

    fn record_route(&mut self, global_ids: &[u32], experts_per_layer: usize) {
        self.route_clock = self.route_clock.saturating_add(1);
        let mut by_layer = BTreeMap::<usize, Vec<u32>>::new();
        for &global_id in global_ids {
            by_layer
                .entry(global_id as usize / experts_per_layer)
                .or_default()
                .push(global_id);
        }
        for (layer_index, ids) in by_layer {
            if let Some(layer) = self.layers.get_mut(layer_index) {
                layer.record_route(&ids, self.route_clock);
            }
        }
    }

    fn ranked_victims(&self, layer_index: usize) -> Vec<u32> {
        let layer = &self.layers[layer_index];
        if !self.policy.route_recency() {
            return layer.resident_mru_to_lru.iter().rev().copied().collect();
        }
        let physical_rank = layer
            .resident_mru_to_lru
            .iter()
            .enumerate()
            .map(|(index, &global_id)| (global_id, index))
            .collect::<HashMap<_, _>>();
        let mut victims = layer.resident_mru_to_lru.clone();
        victims.sort_by(|left, right| {
            let left_route = layer.route_last_seen_clock.get(left).copied();
            let right_route = layer.route_last_seen_clock.get(right).copied();
            match (left_route, right_route) {
                (None, Some(_)) => std::cmp::Ordering::Less,
                (Some(_), None) => std::cmp::Ordering::Greater,
                (Some(left), Some(right)) => left.cmp(&right),
                (None, None) => std::cmp::Ordering::Equal,
            }
            .then_with(|| {
                physical_rank
                    .get(right)
                    .expect("physical rank covers every resident")
                    .cmp(
                        physical_rank
                            .get(left)
                            .expect("physical rank covers every resident"),
                    )
            })
            .then_with(|| left.cmp(right))
        });
        victims
    }

    fn choose_speculative_victim(
        &self,
        layer_index: usize,
        protected: &HashSet<u32>,
    ) -> Result<(u32, u64, bool), ShadowObserverError> {
        let ranked = self.ranked_victims(layer_index);
        if ranked.is_empty() {
            return Err(ShadowObserverError::new(format!(
                "stateful policy {:?} found no resident victim in full layer {layer_index}",
                self.policy
            )));
        }
        if !self.policy.prediction_protected() {
            return Ok((ranked[0], 0, false));
        }
        let mut skipped = 0u64;
        for &candidate in &ranked {
            if protected.contains(&candidate) {
                skipped = skipped.saturating_add(1);
                continue;
            }
            return Ok((candidate, skipped, false));
        }
        Ok((ranked[0], 0, true))
    }

    fn apply_speculation(
        &mut self,
        candidates: &[RankedPrediction],
        actual_snapshots: &[GpuNativePhysicalLayerShadowSnapshot],
        boundary: u64,
        experts_per_layer: usize,
    ) -> Result<SpeculationDelta, ShadowObserverError> {
        if actual_snapshots.len() != self.layers.len() {
            return Err(ShadowObserverError::new(
                "stateful prediction snapshot did not cover every physical layer",
            ));
        }
        for (layer, actual) in self.layers.iter().zip(actual_snapshots) {
            if layer.capacity != actual.slot_capacity {
                return Err(ShadowObserverError::new(format!(
                    "stateful physical capacity drifted on layer {}: initialized={} actual={}",
                    actual.layer_index, layer.capacity, actual.slot_capacity
                )));
            }
        }

        let protected = candidates
            .iter()
            .map(|candidate| candidate.global_id)
            .collect::<HashSet<_>>();
        let mut delta = SpeculationDelta {
            speculative_candidate_count: candidates.len() as u64,
            ..SpeculationDelta::default()
        };
        for candidate in candidates {
            let layer_index = candidate.global_id as usize / experts_per_layer;
            if layer_index >= self.layers.len() {
                return Err(ShadowObserverError::new(format!(
                    "stateful prediction global expert {} is outside the model namespace",
                    candidate.global_id
                )));
            }
            if self.layers[layer_index].contains(candidate.global_id) {
                self.layers[layer_index].touch_physical(candidate.global_id, boundary);
                delta.speculative_already_resident_hits += 1;
                continue;
            }
            if self.layers[layer_index].capacity == 0 {
                return Err(ShadowObserverError::new(format!(
                    "stateful prediction targeted zero-capacity layer {layer_index}"
                )));
            }
            if self.layers[layer_index].resident_mru_to_lru.len()
                >= self.layers[layer_index].capacity
            {
                let (victim, skipped, forced) =
                    self.choose_speculative_victim(layer_index, &protected)?;
                let physical_age = self.layers[layer_index]
                    .entries
                    .get(&victim)
                    .and_then(|entry| entry.last_physical_touch_boundary)
                    .map(|last| boundary.saturating_sub(last));
                let route_age = self.layers[layer_index]
                    .route_last_seen_clock
                    .get(&victim)
                    .copied()
                    .map(|last| self.route_clock.saturating_sub(last));
                self.layers[layer_index].remove(victim);
                self.absent_due_to_hypothetical_eviction
                    .insert(victim, boundary);
                delta.evicted_global_ids.push(victim);
                delta.speculative_replacements += 1;
                delta.speculative_evictions += 1;
                delta.prediction_protected_victim_skips += skipped;
                delta.forced_prediction_protected_evictions += u64::from(forced);
                delta.physical_lru_victim_age.record(physical_age);
                delta.route_recency_victim_age.record(route_age);
            }
            self.layers[layer_index].install(
                candidate.global_id,
                ResidentEntry {
                    speculative_install_boundary: Some(boundary),
                    speculative_install_used: false,
                    last_physical_touch_boundary: Some(boundary),
                },
            );
            self.absent_due_to_hypothetical_eviction
                .remove(&candidate.global_id);
            delta.installed_global_ids.push(candidate.global_id);
            delta.speculative_installs += 1;
        }

        delta.state_churn = delta
            .speculative_installs
            .saturating_add(delta.speculative_evictions);
        delta.resident_set_divergence_from_actual = self
            .layers
            .iter()
            .zip(actual_snapshots)
            .map(|(hypothetical, actual)| {
                let actual = actual
                    .resident_global_ids_mru_to_lru
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>();
                hypothetical
                    .resident_set()
                    .symmetric_difference(&actual)
                    .count() as u64
            })
            .sum();
        Ok(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;
    use std::ffi::OsString;

    fn layer_snapshot(
        layer_index: usize,
        residents_mru_to_lru: &[u32],
        capacity: usize,
    ) -> GpuNativePhysicalLayerShadowSnapshot {
        GpuNativePhysicalLayerShadowSnapshot {
            layer_index,
            slot_capacity: capacity,
            resident_global_ids_mru_to_lru: residents_mru_to_lru.to_vec(),
            free_slots: capacity.saturating_sub(residents_mru_to_lru.len()),
        }
    }

    fn predictions(ids: &[u32]) -> Vec<RankedPrediction> {
        ids.iter()
            .enumerate()
            .map(|(rank, &global_id)| RankedPrediction {
                global_id,
                score: 1.0 / (rank + 1) as f64,
            })
            .collect()
    }

    fn policy(
        policy: ReplacementPolicy,
        snapshots: &[GpuNativePhysicalLayerShadowSnapshot],
        experts_per_layer: usize,
    ) -> HypotheticalPolicyState {
        HypotheticalPolicyState::from_snapshots(policy, snapshots, experts_per_layer).unwrap()
    }

    #[test]
    fn state_persists_without_resynchronization_and_capacity_is_exact() {
        let actual = vec![layer_snapshot(0, &[0, 1], 2), layer_snapshot(1, &[4, 5], 2)];
        let mut state = policy(ReplacementPolicy::PhysicalLru, &actual, 4);
        let first = state
            .apply_speculation(&predictions(&[2]), &actual, 0, 4)
            .unwrap();
        assert_eq!(first.evicted_global_ids, vec![1]);
        assert_eq!(state.layers[0].resident_mru_to_lru, vec![2, 0]);
        assert_eq!(state.layers[0].resident_mru_to_lru.len(), 2);

        let second = state.apply_speculation(&[], &actual, 1, 4).unwrap();
        assert_eq!(state.layers[0].resident_mru_to_lru, vec![2, 0]);
        assert_eq!(second.resident_set_divergence_from_actual, 2);
        assert_eq!(state.resynchronization_count, 0);
    }

    #[test]
    fn initialization_is_deterministic_and_rejects_inexact_capacity_metadata() {
        let actual = vec![layer_snapshot(0, &[2, 0], 3), layer_snapshot(1, &[5], 2)];
        let first = policy(ReplacementPolicy::PhysicalLru, &actual, 4);
        let second = policy(ReplacementPolicy::PhysicalLru, &actual, 4);
        for (first_layer, second_layer) in first.layers.iter().zip(&second.layers) {
            assert_eq!(first_layer.capacity, second_layer.capacity);
            assert_eq!(
                first_layer.resident_mru_to_lru,
                second_layer.resident_mru_to_lru
            );
            assert_eq!(
                first_layer.entries.keys().copied().collect::<BTreeSet<_>>(),
                second_layer
                    .entries
                    .keys()
                    .copied()
                    .collect::<BTreeSet<_>>()
            );
        }

        let invalid = vec![GpuNativePhysicalLayerShadowSnapshot {
            layer_index: 0,
            slot_capacity: 2,
            resident_global_ids_mru_to_lru: vec![0],
            free_slots: 0,
        }];
        assert!(HypotheticalPolicyState::from_snapshots(
            ReplacementPolicy::PhysicalLru,
            &invalid,
            4
        )
        .is_err());
    }

    #[test]
    fn target_truth_cannot_influence_speculative_victim_selection() {
        let actual = vec![layer_snapshot(0, &[0, 1], 2)];
        let mut first_truth = policy(ReplacementPolicy::PhysicalLru, &actual, 4);
        let mut second_truth = policy(ReplacementPolicy::PhysicalLru, &actual, 4);
        let first = first_truth
            .apply_speculation(&predictions(&[2]), &actual, 7, 4)
            .unwrap();
        let second = second_truth
            .apply_speculation(&predictions(&[2]), &actual, 7, 4)
            .unwrap();
        assert_eq!(first.evicted_global_ids, second.evicted_global_ids);
        assert_eq!(
            first_truth.layers[0].resident_mru_to_lru,
            second_truth.layers[0].resident_mru_to_lru
        );

        let first_score = first_truth
            .score_and_service_target(0, &[0], &HashSet::new(), 7, first)
            .unwrap();
        let second_score = second_truth
            .score_and_service_target(0, &[1], &HashSet::new(), 7, second)
            .unwrap();
        assert_ne!(
            first_score.same_boundary_selected_expert_evictions,
            second_score.same_boundary_selected_expert_evictions
        );
    }

    #[test]
    fn physical_lru_and_route_recency_choose_their_frozen_victims_deterministically() {
        let actual = vec![layer_snapshot(0, &[0, 1], 2)];
        let mut lru = policy(ReplacementPolicy::PhysicalLru, &actual, 4);
        let lru_delta = lru
            .apply_speculation(&predictions(&[2]), &actual, 0, 4)
            .unwrap();
        assert_eq!(lru_delta.evicted_global_ids, vec![1]);

        let mut route = policy(ReplacementPolicy::RouteRecency, &actual, 4);
        route.record_route(&[1], 4);
        let route_delta = route
            .apply_speculation(&predictions(&[2]), &actual, 0, 4)
            .unwrap();
        assert_eq!(route_delta.evicted_global_ids, vec![0]);

        let mut tied_again = policy(ReplacementPolicy::RouteRecency, &actual, 4);
        let tied_delta = tied_again
            .apply_speculation(&predictions(&[2]), &actual, 0, 4)
            .unwrap();
        assert_eq!(tied_delta.evicted_global_ids, vec![1]);
    }

    #[test]
    fn prediction_protection_skips_and_forced_protected_eviction_are_distinct() {
        let actual = vec![layer_snapshot(0, &[0, 1], 2)];
        let mut protected = policy(
            ReplacementPolicy::PhysicalLruPredictionProtected,
            &actual,
            4,
        );
        let delta = protected
            .apply_speculation(&predictions(&[2, 1]), &actual, 0, 4)
            .unwrap();
        assert_eq!(delta.evicted_global_ids, vec![0]);
        assert_eq!(delta.prediction_protected_victim_skips, 1);
        assert_eq!(delta.forced_prediction_protected_evictions, 0);
        assert_eq!(delta.speculative_already_resident_hits, 1);

        let forced_actual = vec![layer_snapshot(0, &[0], 1)];
        let mut forced = policy(
            ReplacementPolicy::PhysicalLruPredictionProtected,
            &forced_actual,
            4,
        );
        let forced_delta = forced
            .apply_speculation(&predictions(&[0, 1]), &forced_actual, 0, 4)
            .unwrap();
        assert_eq!(forced_delta.evicted_global_ids, vec![0]);
        assert_eq!(forced_delta.forced_prediction_protected_evictions, 1);
    }

    #[test]
    fn wrong_install_can_become_useful_later_without_midrun_reset() {
        let actual = vec![layer_snapshot(0, &[0], 2)];
        let mut state = policy(ReplacementPolicy::PhysicalLru, &actual, 4);
        let first_spec = state
            .apply_speculation(&predictions(&[2]), &actual, 0, 4)
            .unwrap();
        let first = state
            .score_and_service_target(0, &[0], &HashSet::new(), 0, first_spec)
            .unwrap();
        assert_eq!(first.wrong_speculative_installs, 1);
        assert_eq!(first.useful_speculative_installs, 0);

        let second_spec = state.apply_speculation(&[], &actual, 1, 4).unwrap();
        let second = state
            .score_and_service_target(0, &[2], &HashSet::from([2]), 1, second_spec)
            .unwrap();
        assert_eq!(second.useful_speculative_installs, 1);
        assert_eq!(second.speculative_installs_useful_later, 1);
        assert!(second.miss_boundary_eliminated_vs_actual);
        assert_eq!(second.demand_installs_avoided, 1);
    }

    #[test]
    fn downstream_speculative_harm_introduces_a_boundary_and_demand_repairs_it() {
        let actual = vec![layer_snapshot(0, &[0, 1], 2)];
        let mut state = policy(ReplacementPolicy::PhysicalLru, &actual, 4);
        let first_spec = state
            .apply_speculation(&predictions(&[2]), &actual, 0, 4)
            .unwrap();
        let first = state
            .score_and_service_target(0, &[2], &HashSet::from([2]), 0, first_spec)
            .unwrap();
        assert!(first.miss_boundary_eliminated_vs_actual);
        assert_eq!(first.useful_speculative_installs, 1);

        let second_spec = state.apply_speculation(&[], &actual, 1, 4).unwrap();
        let second = state
            .score_and_service_target(0, &[1], &HashSet::new(), 1, second_spec)
            .unwrap();
        assert!(second.miss_boundary_introduced_vs_actual);
        assert_eq!(second.downstream_selected_expert_evictions, 1);
        assert!(second.harmful_eviction_event);
        assert_eq!(second.demand_installs, 1);
        assert_eq!(second.extra_demand_installs_induced, 1);
        assert!(state.layers[0].contains(1));
    }

    #[test]
    fn downstream_harm_includes_demand_eviction_cascades() {
        let actual = vec![layer_snapshot(0, &[0, 1], 2)];
        let mut state = policy(ReplacementPolicy::PhysicalLru, &actual, 4);
        let first_spec = state
            .apply_speculation(&predictions(&[2]), &actual, 0, 4)
            .unwrap();
        let first = state
            .score_and_service_target(0, &[1], &HashSet::new(), 0, first_spec)
            .unwrap();
        assert_eq!(first.same_boundary_selected_expert_evictions, 1);
        assert_eq!(first.demand_evictions, 1);

        let second_spec = state.apply_speculation(&[], &actual, 1, 4).unwrap();
        let second = state
            .score_and_service_target(0, &[0], &HashSet::new(), 1, second_spec)
            .unwrap();
        assert_eq!(second.downstream_selected_expert_evictions, 1);
        assert!(second.miss_boundary_introduced_vs_actual);
    }

    #[test]
    fn demand_service_matches_pr1bb_hit_touch_protection_eviction_and_install_order() {
        let actual = vec![layer_snapshot(0, &[0, 1], 2)];
        let mut state = policy(ReplacementPolicy::PhysicalLru, &actual, 4);
        let metrics = state
            .score_and_service_target(
                0,
                &[1, 2],
                &HashSet::from([2]),
                0,
                SpeculationDelta::default(),
            )
            .unwrap();
        assert_eq!(metrics.selected_resident_hits, 1);
        assert_eq!(metrics.selected_misses, 1);
        assert_eq!(metrics.demand_evictions, 1);
        assert_eq!(metrics.demand_installs, 1);
        assert_eq!(state.layers[0].resident_mru_to_lru, vec![2, 1]);
    }

    fn event(
        target_layer: usize,
        actual_missing: &[u32],
        policy_metrics: PolicyBoundaryMetrics,
    ) -> StatefulBoundaryEvent {
        StatefulBoundaryEvent {
            phase: ShadowPhase::Measured,
            run_index: 0,
            completed_token_position: 0,
            target_layer,
            run_boundary_index: 0,
            predictor_source: PRIMARY_PREDICTOR,
            predictor_score_semantics: "test",
            requested_fanout: PRIMARY_FANOUT,
            emitted_candidate_count: 0,
            ranked_candidate_global_ids: Vec::new(),
            ranked_candidate_scores: Vec::new(),
            actual_selected_global_ids: actual_missing.to_vec(),
            actual_physical_current_selected_global_ids: Vec::new(),
            actual_physical_missing_selected_global_ids: actual_missing.to_vec(),
            baseline_actual_miss_boundary: !actual_missing.is_empty(),
            actual_model_wide_snapshot_at_prediction_sha256: "a".repeat(64),
            policy_metrics: vec![policy_metrics],
            ordering: StatefulEventOrdering {
                prediction_frozen_sequence: 1,
                speculation_applied_sequence: 2,
                target_probe_sequence: 3,
                prediction_scored_sequence: 4,
                predictor_updated_sequence: 5,
                prediction_frozen_at_monotonic_us: 1,
                target_probe_at_monotonic_us: 2,
            },
        }
    }

    #[test]
    fn global_and_per_layer_aggregation_compute_signed_net_benefits() {
        let actual = vec![layer_snapshot(0, &[0, 1], 2)];
        let mut state = policy(ReplacementPolicy::PhysicalLru, &actual, 4);
        let eliminated_spec = state
            .apply_speculation(&predictions(&[2]), &actual, 0, 4)
            .unwrap();
        let eliminated = state
            .score_and_service_target(0, &[2], &HashSet::from([2]), 0, eliminated_spec)
            .unwrap();
        let introduced_spec = state.apply_speculation(&[], &actual, 1, 4).unwrap();
        let introduced = state
            .score_and_service_target(0, &[1], &HashSet::new(), 1, introduced_spec)
            .unwrap();
        let events = vec![event(0, &[2], eliminated), event(0, &[], introduced)];
        let aggregate = aggregate_phase(&events, ShadowPhase::Measured);
        let metrics = aggregate
            .global_policies
            .iter()
            .find(|metrics| metrics.policy == ReplacementPolicy::PhysicalLru)
            .unwrap();
        assert_eq!(metrics.target_boundaries, 2);
        assert_eq!(metrics.miss_boundaries_eliminated_vs_actual, 1);
        assert_eq!(metrics.miss_boundaries_introduced_vs_actual, 1);
        assert_eq!(metrics.net_boundary_benefit, 0);
        assert_eq!(metrics.net_miss_boundary_reduction, 0);
        assert_eq!(aggregate.per_layer[0].policies[0].target_boundaries, 2);
    }

    #[test]
    fn request_state_resets_but_private_predictor_learning_persists_across_phases() {
        let actual = vec![layer_snapshot(0, &[0, 1], 2)];
        let mut first = policy(ReplacementPolicy::PhysicalLru, &actual, 4);
        first
            .apply_speculation(&predictions(&[2]), &actual, 0, 4)
            .unwrap();
        let second = policy(ReplacementPolicy::PhysicalLru, &actual, 4);
        assert_ne!(
            first.layers[0].resident_mru_to_lru,
            second.layers[0].resident_mru_to_lru
        );

        let observer = GpuNativeStatefulReplacementShadowObserver::new(StatefulObserverConfig {
            num_layers: 2,
            experts_per_layer: 4,
            top_k: 2,
            markov_min_prob: 0.0,
        })
        .unwrap();
        observer.begin_run(ShadowPhase::Warmup, 0).unwrap();
        let after_warmup = {
            let mut inner = observer.inner.lock();
            inner.predictor.observe_step(&[0], &[4, 5]);
            let observations = inner.predictor.observations();
            inner.active_run = None;
            observations
        };
        observer.begin_run(ShadowPhase::Measured, 0).unwrap();
        let inner = observer.inner.lock();
        assert_eq!(inner.predictor.observations(), after_warmup);
        assert!(inner
            .active_run
            .as_ref()
            .is_some_and(|run| run.policies.is_none() && run.next_boundary_index == 0));
    }

    #[test]
    fn runtime_guard_requires_both_callback_classes_and_model_wide_scope() {
        let observer = GpuNativeStatefulReplacementShadowObserver::new(StatefulObserverConfig {
            num_layers: 2,
            experts_per_layer: 4,
            top_k: 2,
            markov_min_prob: 0.0,
        })
        .unwrap();
        assert!(GpuNativePrefetchShadowCallbacks::guard_all_residency_layers(observer.as_ref()));
        assert!(validate_runtime_guard_evidence(ObserverRuntimeGuardEvidence::default()).is_err());
        GpuNativePrefetchShadowCallbacks::record_runtime_guarded_callback(
            observer.as_ref(),
            ShadowObserverCallbackKind::BeforeSegment,
        );
        GpuNativePrefetchShadowCallbacks::record_runtime_guarded_callback(
            observer.as_ref(),
            ShadowObserverCallbackKind::ObserveBoundary,
        );
        assert!(validate_runtime_guard_evidence(observer.runtime_guard_evidence()).is_ok());
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
            run_index: 0,
            prompt_tokens: 1,
            requested_output_tokens: 2,
            generated_tokens: 2,
            generated_token_ids: vec![7, 8],
            generated_token_ids_sha256: "tokens".into(),
            generated_text_sha256: "text".into(),
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
    fn zero_production_speculation_and_behavior_projection_contract_are_retained() {
        let clean = per_run(empty_snapshots());
        assert!(crate::gpu_native_prefetch_shadow::validate_zero_speculative_work(&clean).is_ok());
        let projection = crate::gpu_native_prefetch_multipredictor_shadow::ProductionBehaviorProjection::from_request(
            &clean.generated_token_ids,
            &clean.counters,
        );
        assert_eq!(projection.generated_token_ids, vec![7, 8]);
        let contract =
            crate::gpu_native_prefetch_multipredictor_shadow::BehavioralEquivalenceContract::pr1cb(
            );
        assert!(contract.definition.contains("generated tokens"));

        let mut speculative = empty_snapshots();
        speculative.gpu_native_residency_delta.speculative_requests = 1;
        assert!(
            crate::gpu_native_prefetch_shadow::validate_zero_speculative_work(&per_run(
                speculative
            ))
            .is_err()
        );
    }

    #[test]
    fn report_completion_is_fail_closed_and_cli_is_separate() {
        let empty = StatefulObserverSnapshot::from_data(Vec::new(), Vec::new());
        assert!(validate_snapshot_completion(&empty, 0, 1).is_err());
        assert_eq!(
            SCHEMA,
            "mer.gpu-native-prefetch-stateful-replacement-shadow.v3"
        );
        assert_eq!(MODE, "gpu-native-prefetch-stateful-replacement-shadow");
        assert_eq!(PR1CB_COMMIT, "7f313a5777b5cdb7394bcade1882abab54f0d797");
        assert_eq!(FrozenPr1cbResult::authoritative().miss_boundaries, 3_996);

        let raw = [
            "mer",
            "qualify-gpu-native-prefetch-stateful-replacement-shadow",
            "--config",
            "config.toml",
            "--prompt",
            "hello",
            "--output-tokens",
            "2",
            "--measured-runs",
            "1",
            "--greedy",
            "--expected-adapter-name",
            "NVIDIA L4",
        ]
        .into_iter()
        .map(OsString::from)
        .collect::<Vec<_>>();
        let (normalized, stateful_requested) =
            crate::normalize_stateful_replacement_shadow_command(&raw);
        assert!(stateful_requested);
        let cli = crate::Cli::try_parse_from(normalized).unwrap();
        assert!(matches!(
            cli.cmd,
            crate::Cmd::QualifyGpuNativePrefetchMultipredictorShadow { .. }
        ));
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct StatefulRunInitializationEvidence {
    pub(crate) phase: ShadowPhase,
    pub(crate) run_index: usize,
    pub(crate) completed_token_position: usize,
    pub(crate) first_prediction_target_layer: usize,
    pub(crate) actual_model_wide_snapshot_sha256: String,
    pub(crate) actual_layers: Vec<GpuNativePhysicalLayerShadowSnapshot>,
    pub(crate) policy_clone_count: usize,
    pub(crate) resynchronization_count_after_initialization: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct StatefulEventOrdering {
    pub(crate) prediction_frozen_sequence: u64,
    pub(crate) speculation_applied_sequence: u64,
    pub(crate) target_probe_sequence: u64,
    pub(crate) prediction_scored_sequence: u64,
    pub(crate) predictor_updated_sequence: u64,
    pub(crate) prediction_frozen_at_monotonic_us: u64,
    pub(crate) target_probe_at_monotonic_us: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct StatefulBoundaryEvent {
    pub(crate) phase: ShadowPhase,
    pub(crate) run_index: usize,
    pub(crate) completed_token_position: usize,
    pub(crate) target_layer: usize,
    pub(crate) run_boundary_index: u64,
    pub(crate) predictor_source: &'static str,
    pub(crate) predictor_score_semantics: &'static str,
    pub(crate) requested_fanout: usize,
    pub(crate) emitted_candidate_count: usize,
    pub(crate) ranked_candidate_global_ids: Vec<u32>,
    pub(crate) ranked_candidate_scores: Vec<f64>,
    pub(crate) actual_selected_global_ids: Vec<u32>,
    pub(crate) actual_physical_current_selected_global_ids: Vec<u32>,
    pub(crate) actual_physical_missing_selected_global_ids: Vec<u32>,
    pub(crate) baseline_actual_miss_boundary: bool,
    pub(crate) actual_model_wide_snapshot_at_prediction_sha256: String,
    pub(crate) policy_metrics: Vec<PolicyBoundaryMetrics>,
    pub(crate) ordering: StatefulEventOrdering,
}

#[derive(Clone, Copy, Debug)]
struct ActiveRunIdentity {
    phase: ShadowPhase,
    run_index: usize,
}

struct ActiveRunState {
    identity: ActiveRunIdentity,
    policies: Option<BTreeMap<ReplacementPolicy, HypotheticalPolicyState>>,
    initialization: Option<StatefulRunInitializationEvidence>,
    observed_routes: Vec<Vec<u32>>,
    next_boundary_index: u64,
    pending: Option<PendingTarget>,
}

struct PendingTarget {
    phase: ShadowPhase,
    run_index: usize,
    completed_token_position: usize,
    target_layer: usize,
    run_boundary_index: u64,
    candidates: Vec<RankedPrediction>,
    actual_model_wide_snapshot_at_prediction_sha256: String,
    policy_speculation: BTreeMap<ReplacementPolicy, SpeculationDelta>,
    frozen_sequence: u64,
    speculation_applied_sequence: u64,
    frozen_at_us: u64,
    target_probe_sequence: Option<u64>,
    target_probe_at_us: Option<u64>,
}

struct ScoredTarget {
    pending: PendingTarget,
    actual_selected: Vec<u32>,
    actual_current: Vec<u32>,
    actual_missing: Vec<u32>,
    policy_metrics: Vec<PolicyBoundaryMetrics>,
    scored_sequence: u64,
}

struct ObserverInner {
    origin: Instant,
    sequence: u64,
    active_run: Option<ActiveRunState>,
    predictor: PredictiveLoader,
    last_last_route: Option<RouteHistory>,
    last_route: Option<RouteHistory>,
    events: Vec<StatefulBoundaryEvent>,
    initializations: Vec<StatefulRunInitializationEvidence>,
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

pub(crate) struct GpuNativeStatefulReplacementShadowObserver {
    config: StatefulObserverConfig,
    inner: Mutex<ObserverInner>,
}

impl GpuNativeStatefulReplacementShadowObserver {
    pub(crate) fn new(config: StatefulObserverConfig) -> Result<Arc<Self>, ShadowObserverError> {
        if config.num_layers < 2
            || config.experts_per_layer == 0
            || config.top_k == 0
            || config.top_k > config.experts_per_layer
        {
            return Err(ShadowObserverError::new(format!(
                "invalid stateful replacement observer geometry/config: {config:?}"
            )));
        }
        let total_experts = config
            .num_layers
            .checked_mul(config.experts_per_layer)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| ShadowObserverError::new("stateful expert namespace overflow"))?;
        Ok(Arc::new(Self {
            inner: Mutex::new(ObserverInner {
                origin: Instant::now(),
                sequence: 0,
                active_run: None,
                predictor: PredictiveLoader::new(
                    total_experts,
                    MAX_RANKED_CANDIDATES,
                    config.markov_min_prob,
                    MARKOV_SEED,
                ),
                last_last_route: None,
                last_route: None,
                events: Vec::new(),
                initializations: Vec::new(),
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
                "stateful replacement begin_run called while a run is active",
            ));
        }
        inner.active_run = Some(ActiveRunState {
            identity: ActiveRunIdentity { phase, run_index },
            policies: None,
            initialization: None,
            observed_routes: Vec::new(),
            next_boundary_index: 0,
            pending: None,
        });
        inner.last_last_route = None;
        inner.last_route = None;
        Ok(())
    }

    pub(crate) fn end_run(&self) -> Result<(), ShadowObserverError> {
        let mut inner = self.inner.lock();
        let run = inner.active_run.take().ok_or_else(|| {
            ShadowObserverError::new("stateful replacement end_run called without an active run")
        })?;
        if run.pending.is_some() {
            return Err(ShadowObserverError::new(
                "stateful replacement run ended with an unscored prediction target",
            ));
        }
        let initialization = run.initialization.ok_or_else(|| {
            ShadowObserverError::new("stateful replacement run never initialized policy state")
        })?;
        if run.policies.as_ref().map(BTreeMap::len) != Some(ReplacementPolicy::ALL.len()) {
            return Err(ShadowObserverError::new(
                "stateful replacement run did not retain all four policy states",
            ));
        }
        inner.initializations.push(initialization);
        inner.last_last_route = None;
        inner.last_route = None;
        Ok(())
    }

    fn all_layer_snapshots(
        &self,
        residency: &GpuNativeTieredResidencyManager,
    ) -> Result<Vec<GpuNativePhysicalLayerShadowSnapshot>, ShadowObserverError> {
        (0..self.config.num_layers)
            .map(|layer| residency.shadow_layer_snapshot(layer).map_err(Into::into))
            .collect()
    }

    fn snapshot_sha256(
        snapshots: &[GpuNativePhysicalLayerShadowSnapshot],
    ) -> Result<String, ShadowObserverError> {
        let bytes = serde_json::to_vec(snapshots).map_err(|error| {
            ShadowObserverError::new(format!(
                "failed to serialize stateful model-wide snapshot: {error}"
            ))
        })?;
        Ok(crate::greedy_parity::sha256_hex(&bytes))
    }

    fn initialize_policies(
        &self,
        run: &mut ActiveRunState,
        completed_token_position: usize,
        target_layer: usize,
        snapshots: &[GpuNativePhysicalLayerShadowSnapshot],
        snapshot_sha256: &str,
    ) -> Result<(), ShadowObserverError> {
        if run.policies.is_some() || run.initialization.is_some() {
            return Ok(());
        }
        let mut policies = BTreeMap::new();
        for policy in ReplacementPolicy::ALL {
            let mut state = HypotheticalPolicyState::from_snapshots(
                policy,
                snapshots,
                self.config.experts_per_layer,
            )?;
            for route in &run.observed_routes {
                state.record_route(route, self.config.experts_per_layer);
            }
            policies.insert(policy, state);
        }
        run.initialization = Some(StatefulRunInitializationEvidence {
            phase: run.identity.phase,
            run_index: run.identity.run_index,
            completed_token_position,
            first_prediction_target_layer: target_layer,
            actual_model_wide_snapshot_sha256: snapshot_sha256.to_string(),
            actual_layers: snapshots.to_vec(),
            policy_clone_count: policies.len(),
            resynchronization_count_after_initialization: 0,
        });
        run.policies = Some(policies);
        Ok(())
    }

    fn freeze_next_target(
        &self,
        inner: &mut ObserverInner,
        run: &mut ActiveRunState,
        completed_token_position: usize,
        target_layer: usize,
        source: &RouteHistory,
        source_previous: Option<&RouteHistory>,
        residency: &GpuNativeTieredResidencyManager,
    ) -> Result<PendingTarget, ShadowObserverError> {
        if source.completed_token_position != completed_token_position
            || source.layer + 1 != target_layer
        {
            return Err(ShadowObserverError::new(
                "stateful second-order source was not the target layer's causal predecessor",
            ));
        }
        let seed = *source.global_ids.last().ok_or_else(|| {
            ShadowObserverError::new("stateful second-order source route was empty")
        })?;
        let ranked = match source_previous
            .filter(|previous| {
                previous.completed_token_position == completed_token_position
                    && previous.layer + 1 == source.layer
            })
            .and_then(|previous| previous.global_ids.last().copied())
        {
            Some(previous_seed) => inner.predictor.predict_next2_shadow_ranked(
                previous_seed,
                seed,
                MAX_RANKED_CANDIDATES,
            ),
            None => inner
                .predictor
                .predict_next_shadow_ranked(seed, MAX_RANKED_CANDIDATES),
        };
        let mut seen = HashSet::with_capacity(ranked.len());
        let candidates = ranked
            .into_iter()
            .filter_map(|(global_id, score)| {
                seen.insert(global_id)
                    .then_some(RankedPrediction { global_id, score })
            })
            .take(PRIMARY_FANOUT)
            .collect::<Vec<_>>();
        let actual_snapshots = self.all_layer_snapshots(residency)?;
        let snapshot_sha256 = Self::snapshot_sha256(&actual_snapshots)?;
        self.initialize_policies(
            run,
            completed_token_position,
            target_layer,
            &actual_snapshots,
            &snapshot_sha256,
        )?;
        let frozen_sequence = inner.next_sequence();
        let frozen_at_us = inner.monotonic_us();
        let run_boundary_index = run.next_boundary_index;
        run.next_boundary_index = run.next_boundary_index.saturating_add(1);
        let mut policy_speculation = BTreeMap::new();
        for (policy, state) in run
            .policies
            .as_mut()
            .expect("stateful policies initialized before speculation")
        {
            let delta = state.apply_speculation(
                &candidates,
                &actual_snapshots,
                run_boundary_index,
                self.config.experts_per_layer,
            )?;
            policy_speculation.insert(*policy, delta);
        }
        let speculation_applied_sequence = inner.next_sequence();
        Ok(PendingTarget {
            phase: run.identity.phase,
            run_index: run.identity.run_index,
            completed_token_position,
            target_layer,
            run_boundary_index,
            candidates,
            actual_model_wide_snapshot_at_prediction_sha256: snapshot_sha256,
            policy_speculation,
            frozen_sequence,
            speculation_applied_sequence,
            frozen_at_us,
            target_probe_sequence: None,
            target_probe_at_us: None,
        })
    }

    fn before_segment_inner(
        &self,
        completed_token_position: usize,
        first_new_layer: usize,
    ) -> Result<(), ShadowObserverError> {
        let mut inner = self.inner.lock();
        let now = inner.monotonic_us();
        let sequence = inner.next_sequence();
        let Some(run) = inner.active_run.as_mut() else {
            return Ok(());
        };
        let Some(pending) = run.pending.as_mut() else {
            return Ok(());
        };
        if pending.completed_token_position != completed_token_position
            || pending.target_layer != first_new_layer
        {
            return Err(ShadowObserverError::new(format!(
                "pending stateful target run={} position={} layer={} did not match segment position={} first_new_layer={}",
                pending.run_index,
                pending.completed_token_position,
                pending.target_layer,
                completed_token_position,
                first_new_layer,
            )));
        }
        if pending.target_probe_sequence.is_some() {
            return Err(ShadowObserverError::new(
                "stateful target probe was marked more than once",
            ));
        }
        pending.target_probe_sequence = Some(sequence);
        pending.target_probe_at_us = Some(now.max(pending.frozen_at_us));
        Ok(())
    }

    fn classify_actual_selected(
        &self,
        target_layer: usize,
        local_ids: &[u32],
        residency: &GpuNativeTieredResidencyManager,
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), ShadowObserverError> {
        let mut selected = Vec::with_capacity(local_ids.len());
        let mut current = Vec::new();
        let mut missing = Vec::new();
        for &local_id in local_ids {
            let global_id = global_id(target_layer, local_id, self.config.experts_per_layer)?;
            selected.push(global_id);
            if residency.has_current_for_demand(global_id)? {
                current.push(global_id);
            } else {
                missing.push(global_id);
            }
        }
        Ok((selected, current, missing))
    }

    fn score_pending(
        &self,
        inner: &mut ObserverInner,
        run: &mut ActiveRunState,
        pending: PendingTarget,
        actual_local_ids: &[u32],
        residency: &GpuNativeTieredResidencyManager,
    ) -> Result<ScoredTarget, ShadowObserverError> {
        let target_probe_sequence = pending.target_probe_sequence.ok_or_else(|| {
            ShadowObserverError::new("stateful target truth preceded its probe marker")
        })?;
        if !(pending.frozen_sequence < pending.speculation_applied_sequence
            && pending.speculation_applied_sequence < target_probe_sequence)
        {
            return Err(ShadowObserverError::new(format!(
                "stateful pre-target causal ordering failed: frozen={} applied={} probe={target_probe_sequence}",
                pending.frozen_sequence, pending.speculation_applied_sequence
            )));
        }
        let (actual_selected, actual_current, actual_missing) =
            self.classify_actual_selected(pending.target_layer, actual_local_ids, residency)?;
        let actual_missing_set = actual_missing.iter().copied().collect::<HashSet<_>>();
        let scored_sequence = inner.next_sequence();
        let mut policy_metrics = Vec::with_capacity(ReplacementPolicy::ALL.len());
        let mut speculation = pending.policy_speculation.clone();
        for (policy, state) in run.policies.as_mut().ok_or_else(|| {
            ShadowObserverError::new("stateful target scored before initialization")
        })? {
            let delta = speculation.remove(policy).ok_or_else(|| {
                ShadowObserverError::new(format!(
                    "stateful target lacked frozen speculation for policy {policy:?}"
                ))
            })?;
            policy_metrics.push(state.score_and_service_target(
                pending.target_layer,
                &actual_selected,
                &actual_missing_set,
                pending.run_boundary_index,
                delta,
            )?);
        }
        if !speculation.is_empty() {
            return Err(ShadowObserverError::new(
                "stateful target retained unknown policy speculation",
            ));
        }
        Ok(ScoredTarget {
            pending,
            actual_selected,
            actual_current,
            actual_missing,
            policy_metrics,
            scored_sequence,
        })
    }

    fn observe_boundary_inner(
        &self,
        completed_token_position: usize,
        observed_layers: std::ops::RangeInclusive<usize>,
        selected_ids_by_layer: &[Vec<u32>],
        residency: &GpuNativeTieredResidencyManager,
    ) -> Result<(), ShadowObserverError> {
        let mut inner = self.inner.lock();
        let Some(mut run) = inner.active_run.take() else {
            return Ok(());
        };
        let result = (|| {
            let start = *observed_layers.start();
            let end = *observed_layers.end();
            if start > end || end >= self.config.num_layers || end >= selected_ids_by_layer.len() {
                return Err(ShadowObserverError::new(format!(
                    "invalid stateful observed layer range {start}..={end}"
                )));
            }
            if run.pending.as_ref().is_some_and(|pending| {
                pending.completed_token_position == completed_token_position
                    && pending.target_layer < start
            }) {
                return Err(ShadowObserverError::new(
                    "stateful prediction target was skipped before scoring",
                ));
            }

            for layer in start..=end {
                let local_ids = selected_ids_by_layer.get(layer).ok_or_else(|| {
                    ShadowObserverError::new("missing selected route for stateful layer")
                })?;
                validate_route(local_ids, self.config.top_k, self.config.experts_per_layer)?;

                let scored = if run.pending.as_ref().is_some_and(|pending| {
                    pending.completed_token_position == completed_token_position
                        && pending.target_layer == layer
                }) {
                    let pending = run.pending.take().expect("pending target checked");
                    Some(self.score_pending(&mut inner, &mut run, pending, local_ids, residency)?)
                } else {
                    None
                };

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
                    inner.predictor.observe_step2(
                        previous_previous
                            .as_ref()
                            .map(|route| route.global_ids.as_slice())
                            .unwrap_or(&[]),
                        &previous.global_ids,
                        &global_ids,
                    );
                }
                let predictor_updated_sequence = inner.next_sequence();

                if let Some(scored) = scored {
                    let target_probe_sequence = scored
                        .pending
                        .target_probe_sequence
                        .expect("scored target had a probe sequence");
                    if !(target_probe_sequence < scored.scored_sequence
                        && scored.scored_sequence < predictor_updated_sequence)
                    {
                        return Err(ShadowObserverError::new(format!(
                            "stateful post-target causal ordering failed: probe={target_probe_sequence} scored={} updated={predictor_updated_sequence}",
                            scored.scored_sequence
                        )));
                    }
                    inner.events.push(StatefulBoundaryEvent {
                        phase: scored.pending.phase,
                        run_index: scored.pending.run_index,
                        completed_token_position: scored.pending.completed_token_position,
                        target_layer: scored.pending.target_layer,
                        run_boundary_index: scored.pending.run_boundary_index,
                        predictor_source: PRIMARY_PREDICTOR,
                        predictor_score_semantics: "50/50 Laplace-smoothed first/second-order transition blend with first-order fallback; exact frozen PR1C-B semantics",
                        requested_fanout: PRIMARY_FANOUT,
                        emitted_candidate_count: scored.pending.candidates.len(),
                        ranked_candidate_global_ids: scored
                            .pending
                            .candidates
                            .iter()
                            .map(|candidate| candidate.global_id)
                            .collect(),
                        ranked_candidate_scores: scored
                            .pending
                            .candidates
                            .iter()
                            .map(|candidate| candidate.score)
                            .collect(),
                        actual_selected_global_ids: scored.actual_selected,
                        actual_physical_current_selected_global_ids: scored.actual_current,
                        actual_physical_missing_selected_global_ids: scored.actual_missing.clone(),
                        baseline_actual_miss_boundary: !scored.actual_missing.is_empty(),
                        actual_model_wide_snapshot_at_prediction_sha256: scored
                            .pending
                            .actual_model_wide_snapshot_at_prediction_sha256,
                        policy_metrics: scored.policy_metrics,
                        ordering: StatefulEventOrdering {
                            prediction_frozen_sequence: scored.pending.frozen_sequence,
                            speculation_applied_sequence: scored
                                .pending
                                .speculation_applied_sequence,
                            target_probe_sequence,
                            prediction_scored_sequence: scored.scored_sequence,
                            predictor_updated_sequence,
                            prediction_frozen_at_monotonic_us: scored.pending.frozen_at_us,
                            target_probe_at_monotonic_us: scored
                                .pending
                                .target_probe_at_us
                                .expect("scored target had a probe timestamp"),
                        },
                    });
                }

                if let Some(policies) = run.policies.as_mut() {
                    for state in policies.values_mut() {
                        state.record_route(&global_ids, self.config.experts_per_layer);
                    }
                }
                run.observed_routes.push(global_ids.clone());
                inner.last_last_route = inner.last_route.take();
                inner.last_route = Some(RouteHistory {
                    completed_token_position,
                    layer,
                    global_ids,
                });
            }

            if end + 1 < self.config.num_layers {
                if run.pending.is_some() {
                    return Err(ShadowObserverError::new(
                        "stateful observer attempted to overwrite an unscored target",
                    ));
                }
                let source = inner.last_route.clone().ok_or_else(|| {
                    ShadowObserverError::new("stateful observer lost the last observed route")
                })?;
                let source_previous = inner.last_last_route.clone();
                run.pending = Some(self.freeze_next_target(
                    &mut inner,
                    &mut run,
                    completed_token_position,
                    end + 1,
                    &source,
                    source_previous.as_ref(),
                    residency,
                )?);
            }
            Ok(())
        })();
        inner.active_run = Some(run);
        result
    }

    pub(crate) fn snapshot(&self) -> StatefulObserverSnapshot {
        let inner = self.inner.lock();
        StatefulObserverSnapshot::from_data(inner.events.clone(), inner.initializations.clone())
    }

    pub(crate) fn runtime_guard_evidence(&self) -> ObserverRuntimeGuardEvidence {
        self.inner.lock().runtime_guard_evidence
    }
}

impl GpuNativePrefetchShadowCallbacks for GpuNativeStatefulReplacementShadowObserver {
    fn guard_all_residency_layers(&self) -> bool {
        true
    }

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

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PolicyAggregateMetrics {
    pub(crate) policy: ReplacementPolicy,
    pub(crate) target_boundaries: u64,
    pub(crate) baseline_actual_miss_boundaries: u64,
    pub(crate) hypothetical_miss_boundaries: u64,
    pub(crate) miss_boundaries_eliminated_vs_actual: u64,
    pub(crate) miss_boundaries_introduced_vs_actual: u64,
    pub(crate) net_miss_boundary_reduction: i64,
    pub(crate) net_boundary_benefit: i64,
    pub(crate) net_miss_boundary_reduction_rate: f64,
    pub(crate) actual_selected_expert_misses: u64,
    pub(crate) hypothetical_selected_expert_misses: u64,
    pub(crate) expert_misses_avoided: u64,
    pub(crate) expert_misses_introduced: u64,
    pub(crate) net_expert_miss_reduction: i64,
    pub(crate) net_expert_miss_benefit: i64,
    pub(crate) speculative_candidate_count: u64,
    pub(crate) speculative_already_resident_hits: u64,
    pub(crate) speculative_installs: u64,
    pub(crate) speculative_replacements: u64,
    pub(crate) wrong_speculative_installs: u64,
    pub(crate) useful_speculative_installs: u64,
    pub(crate) speculative_installs_useful_later: u64,
    pub(crate) demand_installs: u64,
    pub(crate) demand_installs_avoided: u64,
    pub(crate) extra_demand_installs_induced: u64,
    pub(crate) total_hypothetical_evictions: u64,
    pub(crate) speculative_evictions: u64,
    pub(crate) demand_evictions: u64,
    pub(crate) same_boundary_selected_expert_evictions: u64,
    pub(crate) downstream_selected_expert_evictions: u64,
    pub(crate) harmful_eviction_events: u64,
    pub(crate) prediction_protected_victim_skips: u64,
    pub(crate) forced_prediction_protected_evictions: u64,
    pub(crate) physical_lru_victim_age: VictimAgeDistribution,
    pub(crate) route_recency_victim_age: VictimAgeDistribution,
    pub(crate) state_churn_total: u64,
    pub(crate) state_churn_mean_per_target: f64,
    pub(crate) state_churn_max_per_target: u64,
    pub(crate) resident_set_divergence_total: u64,
    pub(crate) resident_set_divergence_mean_per_target: f64,
    pub(crate) resident_set_divergence_max_per_target: u64,
    pub(crate) resynchronization_count: u64,
    pub(crate) projected_nvme_reads_avoided: Option<i64>,
    pub(crate) projected_nvme_bytes_avoided: Option<i128>,
}

impl PolicyAggregateMetrics {
    fn empty(policy: ReplacementPolicy) -> Self {
        Self {
            policy,
            target_boundaries: 0,
            baseline_actual_miss_boundaries: 0,
            hypothetical_miss_boundaries: 0,
            miss_boundaries_eliminated_vs_actual: 0,
            miss_boundaries_introduced_vs_actual: 0,
            net_miss_boundary_reduction: 0,
            net_boundary_benefit: 0,
            net_miss_boundary_reduction_rate: 0.0,
            actual_selected_expert_misses: 0,
            hypothetical_selected_expert_misses: 0,
            expert_misses_avoided: 0,
            expert_misses_introduced: 0,
            net_expert_miss_reduction: 0,
            net_expert_miss_benefit: 0,
            speculative_candidate_count: 0,
            speculative_already_resident_hits: 0,
            speculative_installs: 0,
            speculative_replacements: 0,
            wrong_speculative_installs: 0,
            useful_speculative_installs: 0,
            speculative_installs_useful_later: 0,
            demand_installs: 0,
            demand_installs_avoided: 0,
            extra_demand_installs_induced: 0,
            total_hypothetical_evictions: 0,
            speculative_evictions: 0,
            demand_evictions: 0,
            same_boundary_selected_expert_evictions: 0,
            downstream_selected_expert_evictions: 0,
            harmful_eviction_events: 0,
            prediction_protected_victim_skips: 0,
            forced_prediction_protected_evictions: 0,
            physical_lru_victim_age: VictimAgeDistribution::default(),
            route_recency_victim_age: VictimAgeDistribution::default(),
            state_churn_total: 0,
            state_churn_mean_per_target: 0.0,
            state_churn_max_per_target: 0,
            resident_set_divergence_total: 0,
            resident_set_divergence_mean_per_target: 0.0,
            resident_set_divergence_max_per_target: 0,
            resynchronization_count: 0,
            projected_nvme_reads_avoided: None,
            projected_nvme_bytes_avoided: None,
        }
    }

    fn add(&mut self, event: &StatefulBoundaryEvent, policy: &PolicyBoundaryMetrics) {
        self.target_boundaries += 1;
        self.baseline_actual_miss_boundaries += u64::from(event.baseline_actual_miss_boundary);
        self.hypothetical_miss_boundaries += u64::from(policy.hypothetical_miss_boundary);
        self.miss_boundaries_eliminated_vs_actual +=
            u64::from(policy.miss_boundary_eliminated_vs_actual);
        self.miss_boundaries_introduced_vs_actual +=
            u64::from(policy.miss_boundary_introduced_vs_actual);
        self.actual_selected_expert_misses +=
            event.actual_physical_missing_selected_global_ids.len() as u64;
        self.hypothetical_selected_expert_misses += policy.selected_misses;
        self.expert_misses_avoided += policy.expert_misses_avoided;
        self.expert_misses_introduced += policy.expert_misses_introduced;
        self.speculative_candidate_count += policy.speculative_candidate_count;
        self.speculative_already_resident_hits += policy.speculative_already_resident_hits;
        self.speculative_installs += policy.speculative_installs;
        self.speculative_replacements += policy.speculative_replacements;
        self.wrong_speculative_installs += policy.wrong_speculative_installs;
        self.useful_speculative_installs += policy.useful_speculative_installs;
        self.speculative_installs_useful_later += policy.speculative_installs_useful_later;
        self.demand_installs += policy.demand_installs;
        self.demand_installs_avoided += policy.demand_installs_avoided;
        self.extra_demand_installs_induced += policy.extra_demand_installs_induced;
        self.total_hypothetical_evictions += policy.total_hypothetical_evictions;
        self.speculative_evictions += policy.speculative_evictions;
        self.demand_evictions += policy.demand_evictions;
        self.same_boundary_selected_expert_evictions +=
            policy.same_boundary_selected_expert_evictions;
        self.downstream_selected_expert_evictions += policy.downstream_selected_expert_evictions;
        self.harmful_eviction_events += u64::from(policy.harmful_eviction_event);
        self.prediction_protected_victim_skips += policy.prediction_protected_victim_skips;
        self.forced_prediction_protected_evictions += policy.forced_prediction_protected_evictions;
        self.physical_lru_victim_age
            .add(&policy.physical_lru_victim_age);
        self.route_recency_victim_age
            .add(&policy.route_recency_victim_age);
        self.state_churn_total += policy.state_churn;
        self.state_churn_max_per_target = self.state_churn_max_per_target.max(policy.state_churn);
        self.resident_set_divergence_total += policy.resident_set_divergence_from_actual;
        self.resident_set_divergence_max_per_target = self
            .resident_set_divergence_max_per_target
            .max(policy.resident_set_divergence_from_actual);
        self.resynchronization_count += policy.resynchronization_count;
    }

    fn finish(&mut self) {
        self.net_boundary_benefit = self.miss_boundaries_eliminated_vs_actual as i64
            - self.miss_boundaries_introduced_vs_actual as i64;
        self.net_miss_boundary_reduction = self.net_boundary_benefit;
        self.net_miss_boundary_reduction_rate = signed_ratio(
            self.net_boundary_benefit,
            self.baseline_actual_miss_boundaries,
        );
        self.net_expert_miss_benefit =
            self.expert_misses_avoided as i64 - self.expert_misses_introduced as i64;
        self.net_expert_miss_reduction = self.net_expert_miss_benefit;
        if self.target_boundaries > 0 {
            self.state_churn_mean_per_target =
                self.state_churn_total as f64 / self.target_boundaries as f64;
            self.resident_set_divergence_mean_per_target =
                self.resident_set_divergence_total as f64 / self.target_boundaries as f64;
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PerLayerPolicyAggregate {
    pub(crate) target_layer: usize,
    pub(crate) policies: Vec<PolicyAggregateMetrics>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct StatefulPhaseAggregate {
    pub(crate) target_boundaries: u64,
    pub(crate) global_policies: Vec<PolicyAggregateMetrics>,
    pub(crate) per_layer: Vec<PerLayerPolicyAggregate>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct StatefulObserverSnapshot {
    pub(crate) warmup: StatefulPhaseAggregate,
    pub(crate) measured: StatefulPhaseAggregate,
    pub(crate) initializations: Vec<StatefulRunInitializationEvidence>,
    pub(crate) events: Vec<StatefulBoundaryEvent>,
}

impl StatefulObserverSnapshot {
    fn from_data(
        events: Vec<StatefulBoundaryEvent>,
        initializations: Vec<StatefulRunInitializationEvidence>,
    ) -> Self {
        Self {
            warmup: aggregate_phase(&events, ShadowPhase::Warmup),
            measured: aggregate_phase(&events, ShadowPhase::Measured),
            initializations,
            events,
        }
    }
}

fn aggregate_phase(events: &[StatefulBoundaryEvent], phase: ShadowPhase) -> StatefulPhaseAggregate {
    let filtered = events
        .iter()
        .filter(|event| event.phase == phase)
        .collect::<Vec<_>>();
    let mut global = ReplacementPolicy::ALL
        .into_iter()
        .map(|policy| (policy, PolicyAggregateMetrics::empty(policy)))
        .collect::<BTreeMap<_, _>>();
    for event in &filtered {
        for policy in &event.policy_metrics {
            global
                .get_mut(&policy.policy)
                .expect("all policies preallocated")
                .add(event, policy);
        }
    }
    for metrics in global.values_mut() {
        metrics.finish();
    }
    let layer_count = filtered
        .iter()
        .map(|event| event.target_layer + 1)
        .max()
        .unwrap_or(0);
    let mut per_layer = Vec::with_capacity(layer_count);
    for target_layer in 0..layer_count {
        let mut policies = ReplacementPolicy::ALL
            .into_iter()
            .map(|policy| (policy, PolicyAggregateMetrics::empty(policy)))
            .collect::<BTreeMap<_, _>>();
        for event in filtered
            .iter()
            .copied()
            .filter(|event| event.target_layer == target_layer)
        {
            for policy in &event.policy_metrics {
                policies
                    .get_mut(&policy.policy)
                    .expect("all policies preallocated")
                    .add(event, policy);
            }
        }
        for metrics in policies.values_mut() {
            metrics.finish();
        }
        per_layer.push(PerLayerPolicyAggregate {
            target_layer,
            policies: policies.into_values().collect(),
        });
    }
    StatefulPhaseAggregate {
        target_boundaries: filtered.len() as u64,
        global_policies: global.into_values().collect(),
        per_layer,
    }
}

fn signed_ratio(numerator: i64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn validate_route(
    local_ids: &[u32],
    top_k: usize,
    experts_per_layer: usize,
) -> Result<(), ShadowObserverError> {
    if local_ids.len() != top_k {
        return Err(ShadowObserverError::new(format!(
            "stateful route contained {} experts, expected {top_k}",
            local_ids.len()
        )));
    }
    let mut seen = HashSet::with_capacity(local_ids.len());
    for &local_id in local_ids {
        if local_id as usize >= experts_per_layer {
            return Err(ShadowObserverError::new(format!(
                "stateful route expert {local_id} is outside 0..{experts_per_layer}"
            )));
        }
        if !seen.insert(local_id) {
            return Err(ShadowObserverError::new(format!(
                "stateful route duplicated expert {local_id}"
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
        .ok_or_else(|| ShadowObserverError::new("stateful global expert id overflow"))
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

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FrozenPr1cbResult {
    pub(crate) shadow_exit: i32,
    pub(crate) report_contract: bool,
    pub(crate) token_parity_pr1bb: bool,
    pub(crate) token_parity_pr1ca: bool,
    pub(crate) behavior_parity_pr1bb: bool,
    pub(crate) behavior_parity_pr1ca: bool,
    pub(crate) final_exit: i32,
    pub(crate) observer_guarded_callbacks: u64,
    pub(crate) production_speculative_activity: u64,
    pub(crate) measured_targets: u64,
    pub(crate) best_predictor: &'static str,
    pub(crate) best_fanout: usize,
    pub(crate) route_precision: f64,
    pub(crate) route_recall: f64,
    pub(crate) physical_missing_precision: f64,
    pub(crate) physical_missing_recall: f64,
    pub(crate) exact_missing_set_coverage_count: u64,
    pub(crate) miss_boundaries: u64,
    pub(crate) coverage_upper_bound_eliminations: u64,
    pub(crate) coverage_upper_bound_elimination_rate: f64,
    pub(crate) current_policy_projected_eliminations: u64,
    pub(crate) isolated_counterfactual_lru_projected_eliminations: u64,
    pub(crate) isolated_counterfactual_lru_projected_elimination_rate: f64,
    pub(crate) isolated_counterfactual_harmful_eviction_events: u64,
    pub(crate) isolated_counterfactual_harmful_evictions_per_projected_elimination: f64,
}

impl FrozenPr1cbResult {
    fn authoritative() -> Self {
        Self {
            shadow_exit: 0,
            report_contract: true,
            token_parity_pr1bb: true,
            token_parity_pr1ca: true,
            behavior_parity_pr1bb: true,
            behavior_parity_pr1ca: true,
            final_exit: 0,
            observer_guarded_callbacks: 11_230,
            production_speculative_activity: 0,
            measured_targets: 3_999,
            best_predictor: PRIMARY_PREDICTOR,
            best_fanout: PRIMARY_FANOUT,
            route_precision: 0.6890838535515438,
            route_recall: 0.6383158289572393,
            physical_missing_precision: 0.3120296946178505,
            physical_missing_recall: 0.5191735444388299,
            exact_missing_set_coverage_count: 908,
            miss_boundaries: 3_996,
            coverage_upper_bound_eliminations: 905,
            coverage_upper_bound_elimination_rate: 0.22647647647647648,
            current_policy_projected_eliminations: 0,
            isolated_counterfactual_lru_projected_eliminations: 720,
            isolated_counterfactual_lru_projected_elimination_rate: 0.18018018018018017,
            isolated_counterfactual_harmful_eviction_events: 2_499,
            isolated_counterfactual_harmful_evictions_per_projected_elimination: 3.470833333333333,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct StatefulPredictorContract {
    pub(crate) predictor: &'static str,
    pub(crate) fanout: usize,
    pub(crate) retuned: bool,
    pub(crate) global_id_predictions_target_layer_filtered: bool,
    pub(crate) exact_causal_state: &'static str,
    pub(crate) exact_freeze_point: &'static str,
    pub(crate) exact_update_timing: &'static str,
    pub(crate) target_truth_available_to_speculative_victim_selection: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct StatefulDemandServiceModel {
    pub(crate) source_locations: Vec<&'static str>,
    pub(crate) equivalence: Vec<&'static str>,
    pub(crate) simplifications: Vec<&'static str>,
    pub(crate) physical_lru_execution_touch_semantics: &'static str,
    pub(crate) route_recency_semantics: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct NvmeProjectionContract {
    pub(crate) fixed_expert_payload_bytes: usize,
    pub(crate) nvme_reads_or_bytes_derivable_from_residency_ids_alone: bool,
    pub(crate) reason: &'static str,
    pub(crate) performance_measurement: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct StatefulReplacementShadowReport {
    pub(crate) schema: &'static str,
    pub(crate) mode: &'static str,
    pub(crate) source_main_commit: &'static str,
    pub(crate) tested_pr1bb_commit: &'static str,
    pub(crate) tested_pr1bb_report_sha256: &'static str,
    pub(crate) pr1ca_commit: &'static str,
    pub(crate) pr1ca_report_sha256: &'static str,
    pub(crate) pr1ca_log_sha256: &'static str,
    pub(crate) pr1cb_commit: &'static str,
    pub(crate) pr1cb_first_authoritative_report_sha256: &'static str,
    pub(crate) pr1cb_first_authoritative_log_sha256: &'static str,
    pub(crate) frozen_pr1cb_result: FrozenPr1cbResult,
    pub(crate) canonical_benchmark_schema_unchanged: &'static str,
    pub(crate) shadow_complete: bool,
    pub(crate) failure: Option<crate::gpu_native_real_benchmark::BenchmarkFailure>,
    pub(crate) qualification_pass: bool,
    pub(crate) performance_claim: bool,
    pub(crate) production_prefetch_enabled: bool,
    pub(crate) production_speculative_runtime_postconditions_verified: bool,
    pub(crate) external_pr1bb_pr1ca_pr1cb_behavioral_equivalence_pending: bool,
    pub(crate) behavioral_equivalence:
        crate::gpu_native_prefetch_multipredictor_shadow::BehavioralEquivalenceContract,
    pub(crate) production_semantics:
        crate::gpu_native_prefetch_multipredictor_shadow::MultipredictorProductionSemantics,
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
    pub(crate) policies: Vec<ReplacementPolicy>,
    pub(crate) predictor_contract: StatefulPredictorContract,
    pub(crate) initialization_semantics: &'static str,
    pub(crate) resynchronization_semantics: &'static str,
    pub(crate) candidate_layer_semantics: &'static str,
    pub(crate) aggregation_layer_semantics: &'static str,
    pub(crate) demand_service_model: StatefulDemandServiceModel,
    pub(crate) nvme_projection_contract: NvmeProjectionContract,
    pub(crate) observer_runtime_guarded_classes: Vec<&'static str>,
    pub(crate) observer_runtime_guard_evidence: ObserverRuntimeGuardEvidence,
    pub(crate) warmup_learning_semantics: &'static str,
    pub(crate) warmup_run_evidence:
        Vec<crate::gpu_native_prefetch_multipredictor_shadow::MultipredictorRunEvidence>,
    pub(crate) measured_run_evidence:
        Vec<crate::gpu_native_prefetch_multipredictor_shadow::MultipredictorRunEvidence>,
    pub(crate) shadow_observations: Option<StatefulObserverSnapshot>,
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
                "stateful replacement callback guard did not verify every callback class: {evidence:?}"
            ),
        ));
    }
    Ok(())
}

fn validate_snapshot_completion(
    snapshot: &StatefulObserverSnapshot,
    warmup_runs: usize,
    measured_runs: usize,
) -> Result<(), crate::gpu_native_real_benchmark::BenchmarkFailure> {
    use crate::gpu_native_real_benchmark::BenchmarkFailure;
    if snapshot.measured.target_boundaries == 0 {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "no-causal-measured-events",
            "no target layer became eligible for stateful replacement scoring",
        ));
    }
    if snapshot.initializations.len() != warmup_runs.saturating_add(measured_runs) {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "incomplete-stateful-initialization-set",
            format!(
                "expected {} request initializations, observed {}",
                warmup_runs.saturating_add(measured_runs),
                snapshot.initializations.len()
            ),
        ));
    }
    let expected = ReplacementPolicy::ALL.into_iter().collect::<BTreeSet<_>>();
    for initialization in &snapshot.initializations {
        if initialization.policy_clone_count != expected.len()
            || initialization.resynchronization_count_after_initialization != 0
            || !validate_hex(&initialization.actual_model_wide_snapshot_sha256, 64)
        {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "invalid-stateful-initialization",
                format!("invalid stateful initialization: {initialization:?}"),
            ));
        }
    }
    for event in &snapshot.events {
        let policies = event
            .policy_metrics
            .iter()
            .map(|metrics| metrics.policy)
            .collect::<BTreeSet<_>>();
        if policies != expected
            || event.policy_metrics.len() != expected.len()
            || event.predictor_source != PRIMARY_PREDICTOR
            || event.requested_fanout != PRIMARY_FANOUT
            || !validate_hex(&event.actual_model_wide_snapshot_at_prediction_sha256, 64)
            || event
                .policy_metrics
                .iter()
                .any(|metrics| metrics.resynchronization_count != 0)
            || !(event.ordering.prediction_frozen_sequence
                < event.ordering.speculation_applied_sequence
                && event.ordering.speculation_applied_sequence
                    < event.ordering.target_probe_sequence
                && event.ordering.target_probe_sequence < event.ordering.prediction_scored_sequence
                && event.ordering.prediction_scored_sequence
                    < event.ordering.predictor_updated_sequence)
        {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "invalid-stateful-event-contract",
                format!(
                    "stateful event failed policy/causality/completion validation: run={} position={} layer={}",
                    event.run_index, event.completed_token_position, event.target_layer
                ),
            ));
        }
    }
    Ok(())
}

async fn execute_shadow_run(
    runtime: &crate::BenchRealRuntime,
    observer: &Arc<GpuNativeStatefulReplacementShadowObserver>,
    phase: ShadowPhase,
    run_index: usize,
    prompt_ids: &[u32],
    output_tokens: usize,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<
    crate::gpu_native_prefetch_multipredictor_shadow::MultipredictorRunEvidence,
    crate::gpu_native_real_benchmark::BenchmarkFailure,
> {
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
        format!(
            "qualify-gpu-native-prefetch-stateful-replacement-shadow {phase_label} run {run_index}"
        ),
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
            "stateful-replacement-shadow-request-failed",
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
    Ok(
        crate::gpu_native_prefetch_multipredictor_shadow::MultipredictorRunEvidence::from_result(
            phase, result,
        ),
    )
}

fn emit_report(
    report: &StatefulReplacementShadowReport,
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
            "GPU-native stateful replacement shadow report written to {}",
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
            "qualify-gpu-native-prefetch-stateful-replacement-shadow requires the explicit --greedy flag",
        )
        .into());
    }
    if args.measured_runs == 0 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "measured-runs-required",
            "qualify-gpu-native-prefetch-stateful-replacement-shadow requires --measured-runs > 0",
        )
        .into());
    }
    if args.cache_reset != crate::BenchRealCacheReset::Keep {
        return Err(BenchmarkFailure::new(
            "preflight",
            "cache-reset-contract",
            "stateful replacement shadow schema v3 supports only the frozen --cache-reset keep schedule",
        )
        .into());
    }
    if args.expected_adapter_name != FROZEN_ADAPTER_NAME {
        return Err(BenchmarkFailure::new(
            "preflight",
            "frozen-adapter-required",
            format!(
                "stateful replacement shadow schema v3 requires --expected-adapter-name {FROZEN_ADAPTER_NAME:?}; observed {:?}",
                args.expected_adapter_name
            ),
        )
        .into());
    }
    let request_input = crate::load_real_cli_request_input(
        "qualify-gpu-native-prefetch-stateful-replacement-shadow",
        args.prompt.as_ref(),
        args.request_json.as_deref(),
        args.output_tokens,
    )?;
    if request_input.output_tokens < 2 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "insufficient-output-tokens",
            "qualify-gpu-native-prefetch-stateful-replacement-shadow requires --output-tokens >= 2",
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
                "stateful replacement total expert namespace overflowed",
            )
        })?;
    let observer_config = StatefulObserverConfig {
        num_layers: cfg.model.num_layers,
        experts_per_layer: cfg.model.num_experts as usize,
        top_k: cfg.model.top_k,
        markov_min_prob: resolve_predict_min_prob(cfg.storage.predict_min_prob, total_experts),
    };
    let fixed_expert_payload_bytes = cfg.model.expert_size;
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
    let mut report = StatefulReplacementShadowReport {
        schema: SCHEMA,
        mode: MODE,
        source_main_commit: SOURCE_MAIN_COMMIT,
        tested_pr1bb_commit: TESTED_PR1BB_COMMIT,
        tested_pr1bb_report_sha256: TESTED_PR1BB_REPORT_SHA256,
        pr1ca_commit: PR1CA_COMMIT,
        pr1ca_report_sha256: PR1CA_REPORT_SHA256,
        pr1ca_log_sha256: PR1CA_LOG_SHA256,
        pr1cb_commit: PR1CB_COMMIT,
        pr1cb_first_authoritative_report_sha256: PR1CB_REPORT_SHA256,
        pr1cb_first_authoritative_log_sha256: PR1CB_LOG_SHA256,
        frozen_pr1cb_result: FrozenPr1cbResult::authoritative(),
        canonical_benchmark_schema_unchanged: crate::gpu_native_real_benchmark::SCHEMA,
        shadow_complete: false,
        failure: None,
        qualification_pass: false,
        performance_claim: false,
        production_prefetch_enabled: false,
        production_speculative_runtime_postconditions_verified: false,
        external_pr1bb_pr1ca_pr1cb_behavioral_equivalence_pending: true,
        behavioral_equivalence:
            crate::gpu_native_prefetch_multipredictor_shadow::BehavioralEquivalenceContract::pr1cb(
            ),
        production_semantics:
            crate::gpu_native_prefetch_multipredictor_shadow::MultipredictorProductionSemantics::shadow_only(),
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
        policies: ReplacementPolicy::ALL.to_vec(),
        predictor_contract: StatefulPredictorContract {
            predictor: PRIMARY_PREDICTOR,
            fanout: PRIMARY_FANOUT,
            retuned: false,
            global_id_predictions_target_layer_filtered: false,
            exact_causal_state: "private PredictiveLoader first/second-order rows retained across warmup and measured requests; the last-ranked global IDs from causal contiguous layers L-2 and L-1 seed the exact PR1C-B 50/50 blend with first-order fallback",
            exact_freeze_point: "after route L-1 becomes CPU-visible and every earlier target has been scored/learned, before target L segment submission and before target truth is visible",
            exact_update_timing: "all four policy speculative mutations complete from the frozen candidates first; target truth is then scored; only afterward are private PredictiveLoader rows and route-recency metadata updated",
            target_truth_available_to_speculative_victim_selection: false,
        },
        initialization_semantics: "at the first eligible prediction point of each request, one read-only model-wide production physical snapshot is cloned into each of four private policy simulators; the clone preserves exact per-layer capacity, free slots, resident global identities, and MRU-to-LRU order",
        resynchronization_semantics: "after request initialization, later model-wide production snapshots are checksummed and used only for capacity validation and resident-set divergence; hypothetical residents and ordering are never copied from production until the next request reset",
        candidate_layer_semantics: "the frozen PR1C-B second-order predictor emits unfiltered global expert IDs; each speculative candidate mutates the hypothetical physical layer named by its global ID rather than being incorrectly forced into the immediate target layer",
        aggregation_layer_semantics: "global metrics include all candidate-layer mutations; per-layer metrics are grouped by the causal target boundary layer at which the prediction was frozen and scored",
        demand_service_model: StatefulDemandServiceModel {
            source_locations: vec![
                "rust-engine/src/engine.rs::Engine::ensure_gpu_native_demand_residency",
                "rust-engine/src/gpu_native_residency.rs::GpuNativeTieredResidencyManager::ensure_demand_set",
                "rust-engine/src/gpu_native_residency.rs::oldest_unprotected",
                "rust-engine/src/gpu_native_residency.rs::touch_physical_record",
            ],
            equivalence: vec![
                "score the complete selected set against pre-service physical residency",
                "when any selected miss exists, touch current selected hits in selected order",
                "protect every selected global ID while evicting ordinary physical LRU non-selected victims until the entire missing set fits",
                "install missing selected experts in selected order, making the last installed miss most recent",
                "retain speculative and demand victims in one causal absence ledger so later selected-expert harm includes demand-repair cascades",
                "when no selected miss exists, do not invent a demand transaction or physical-LRU touch",
            ],
            simplifications: vec![
                "IDs and metadata only: no payload acquisition, logical admission, arena operation, generation race, retry race, or storage I/O is simulated",
                "the model assumes the source-atomic PR1B-B demand transaction succeeds after the already-observed authoritative miss classification",
                "fixed expert payload size does not reveal whether a physical miss would be satisfied by RAM or NVMe",
            ],
            physical_lru_execution_touch_semantics: "PR1B-B source has no separate physical execution-route touch: physical MRU-to-LRU changes only through speculative probes/installs and the faithful demand transaction described above",
            route_recency_semantics: "a separate causal shadow clock records every actual routed set only after target scoring; route-recency victims are least recently observed, ties use existing physical LRU order and then ascending global expert ID",
        },
        nvme_projection_contract: NvmeProjectionContract {
            fixed_expert_payload_bytes,
            nvme_reads_or_bytes_derivable_from_residency_ids_alone: false,
            reason: "net physical demand misses determine projected source acquisitions, but IDs and fixed payload size do not determine RAM hit versus NVMe read; policy aggregates therefore serialize projected NVMe reads/bytes as null instead of making a performance claim",
            performance_measurement: false,
        },
        observer_runtime_guarded_classes: vec![
            "GPU-native residency counters including every speculative counter",
            "model-wide physical metadata, identities, MRU-to-LRU order, free slots, and arena state for every callback",
            "RAM cache hits/misses and NVMe operation/byte counters",
            "prefetch_completed and production predictor observations",
            "logical GPU promotions/hits/misses/occupancy/used bytes",
            "production prefetch and governor counters",
        ],
        observer_runtime_guard_evidence: ObserverRuntimeGuardEvidence::default(),
        warmup_learning_semantics: "one private second-order predictor spans the frozen keep-cache schedule; predictor learning persists causally across warmup and measured requests, while every policy residency simulator and route-recency clock resets from exactly one production snapshot at each request",
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
            "runtime geometry drifted from preflight stateful observer config: runtime={geometry:?} observer={observer_config:?} shutdown={shutdown:?}"
        )
        .into());
    }
    let observer = GpuNativeStatefulReplacementShadowObserver::new(observer_config)?;
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
        validate_snapshot_completion(&snapshot, args.warmup_runs, args.measured_runs)?;
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
                    "stateful replacement shadow qualification did not retain every requested run",
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

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PolicyBoundaryMetrics {
    pub(crate) policy: ReplacementPolicy,
    pub(crate) selected_resident_hits: u64,
    pub(crate) selected_misses: u64,
    pub(crate) hypothetical_miss_boundary: bool,
    pub(crate) miss_boundary_eliminated_vs_actual: bool,
    pub(crate) miss_boundary_introduced_vs_actual: bool,
    pub(crate) expert_misses_avoided: u64,
    pub(crate) expert_misses_introduced: u64,
    pub(crate) speculative_candidate_count: u64,
    pub(crate) speculative_already_resident_hits: u64,
    pub(crate) speculative_installs: u64,
    pub(crate) speculative_replacements: u64,
    pub(crate) wrong_speculative_installs: u64,
    pub(crate) useful_speculative_installs: u64,
    pub(crate) speculative_installs_useful_later: u64,
    pub(crate) demand_installs: u64,
    pub(crate) demand_installs_avoided: u64,
    pub(crate) extra_demand_installs_induced: u64,
    pub(crate) total_hypothetical_evictions: u64,
    pub(crate) speculative_evictions: u64,
    pub(crate) demand_evictions: u64,
    pub(crate) same_boundary_selected_expert_evictions: u64,
    pub(crate) downstream_selected_expert_evictions: u64,
    pub(crate) harmful_eviction_event: bool,
    pub(crate) prediction_protected_victim_skips: u64,
    pub(crate) forced_prediction_protected_evictions: u64,
    pub(crate) physical_lru_victim_age: VictimAgeDistribution,
    pub(crate) route_recency_victim_age: VictimAgeDistribution,
    pub(crate) state_churn: u64,
    pub(crate) resident_set_divergence_from_actual: u64,
    pub(crate) resynchronization_count: u64,
}

impl HypotheticalPolicyState {
    fn score_and_service_target(
        &mut self,
        target_layer: usize,
        selected: &[u32],
        actual_missing: &HashSet<u32>,
        boundary: u64,
        speculation: SpeculationDelta,
    ) -> Result<PolicyBoundaryMetrics, ShadowObserverError> {
        let selected_set = selected.iter().copied().collect::<HashSet<_>>();
        let hypothetical_missing = selected
            .iter()
            .copied()
            .filter(|global_id| !self.layers[target_layer].contains(*global_id))
            .collect::<Vec<_>>();
        let hypothetical_missing_set = hypothetical_missing.iter().copied().collect::<HashSet<_>>();
        let mut useful_speculative_installs = 0u64;
        let mut useful_later = 0u64;
        for &global_id in selected {
            let Some(entry) = self.layers[target_layer].entries.get_mut(&global_id) else {
                continue;
            };
            let Some(origin) = entry.speculative_install_boundary else {
                continue;
            };
            if !entry.speculative_install_used {
                entry.speculative_install_used = true;
                useful_speculative_installs += 1;
                useful_later += u64::from(origin < boundary);
            }
        }
        let wrong_speculative_installs = speculation
            .installed_global_ids
            .iter()
            .filter(|global_id| !selected_set.contains(global_id))
            .count() as u64;

        let mut same_boundary_selected_expert_evictions = 0u64;
        let mut downstream_selected_expert_evictions = 0u64;
        for &global_id in &hypothetical_missing {
            match self
                .absent_due_to_hypothetical_eviction
                .get(&global_id)
                .copied()
            {
                Some(origin) if origin == boundary => same_boundary_selected_expert_evictions += 1,
                Some(_) => downstream_selected_expert_evictions += 1,
                None => {}
            }
        }

        let mut demand_evictions = 0u64;
        let demand_installs = hypothetical_missing.len() as u64;
        if !hypothetical_missing.is_empty() {
            for &global_id in selected {
                if self.layers[target_layer].contains(global_id) {
                    self.layers[target_layer].touch_physical(global_id, boundary);
                }
            }
            while self.layers[target_layer]
                .resident_mru_to_lru
                .len()
                .saturating_add(hypothetical_missing.len())
                > self.layers[target_layer].capacity
            {
                let victim = self.layers[target_layer]
                    .resident_mru_to_lru
                    .iter()
                    .rev()
                    .copied()
                    .find(|victim| !selected_set.contains(victim))
                    .ok_or_else(|| {
                        ShadowObserverError::new(format!(
                            "stateful PR1B-B demand service found no unprotected victim on layer {target_layer}"
                        ))
                })?;
                self.layers[target_layer].remove(victim);
                self.absent_due_to_hypothetical_eviction
                    .insert(victim, boundary);
                demand_evictions += 1;
            }
            for &global_id in &hypothetical_missing {
                self.layers[target_layer].install(
                    global_id,
                    ResidentEntry {
                        speculative_install_boundary: None,
                        speculative_install_used: false,
                        last_physical_touch_boundary: Some(boundary),
                    },
                );
                self.absent_due_to_hypothetical_eviction.remove(&global_id);
            }
        }

        let expert_misses_avoided =
            actual_missing.difference(&hypothetical_missing_set).count() as u64;
        let expert_misses_introduced =
            hypothetical_missing_set.difference(actual_missing).count() as u64;
        let actual_miss_boundary = !actual_missing.is_empty();
        let hypothetical_miss_boundary = !hypothetical_missing.is_empty();
        let harmful_eviction_event =
            same_boundary_selected_expert_evictions > 0 || downstream_selected_expert_evictions > 0;
        let total_hypothetical_evictions = speculation
            .speculative_evictions
            .saturating_add(demand_evictions);
        let state_churn = speculation
            .state_churn
            .saturating_add(demand_installs)
            .saturating_add(demand_evictions);

        Ok(PolicyBoundaryMetrics {
            policy: self.policy,
            selected_resident_hits: selected.len() as u64 - hypothetical_missing.len() as u64,
            selected_misses: hypothetical_missing.len() as u64,
            hypothetical_miss_boundary,
            miss_boundary_eliminated_vs_actual: actual_miss_boundary && !hypothetical_miss_boundary,
            miss_boundary_introduced_vs_actual: !actual_miss_boundary && hypothetical_miss_boundary,
            expert_misses_avoided,
            expert_misses_introduced,
            speculative_candidate_count: speculation.speculative_candidate_count,
            speculative_already_resident_hits: speculation.speculative_already_resident_hits,
            speculative_installs: speculation.speculative_installs,
            speculative_replacements: speculation.speculative_replacements,
            wrong_speculative_installs,
            useful_speculative_installs,
            speculative_installs_useful_later: useful_later,
            demand_installs,
            demand_installs_avoided: expert_misses_avoided,
            extra_demand_installs_induced: expert_misses_introduced,
            total_hypothetical_evictions,
            speculative_evictions: speculation.speculative_evictions,
            demand_evictions,
            same_boundary_selected_expert_evictions,
            downstream_selected_expert_evictions,
            harmful_eviction_event,
            prediction_protected_victim_skips: speculation.prediction_protected_victim_skips,
            forced_prediction_protected_evictions: speculation
                .forced_prediction_protected_evictions,
            physical_lru_victim_age: speculation.physical_lru_victim_age,
            route_recency_victim_age: speculation.route_recency_victim_age,
            state_churn,
            resident_set_divergence_from_actual: speculation.resident_set_divergence_from_actual,
            resynchronization_count: self.resynchronization_count,
        })
    }
}
