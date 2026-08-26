//! Tier 3 — per-layer **pre-gate** predictor.
//!
//! ## Idea
//!
//! In a real transformer the engine sees one MoE layer's routed expert
//! set strictly *before* the next layer's — `forward` walks layers
//! `0, 1, 2, …` in order, calling [`crate::engine::Engine::moe_step`]
//! once per MoE layer. The experts a layer routes to are highly
//! predictive of the *next* layer's routed set (consecutive layers
//! operate on the same residual stream), so we can learn an online
//! conditional map
//!
//! ```text
//!   P(expert b routed in layer L+1 | expert a routed in layer L)
//! ```
//!
//! and, the moment layer `L` routes, prefetch layer `L+1`'s likely
//! experts from the SSD so their bytes land in RAM *while layer `L` is
//! still computing*. Unlike the speculative arms that fire on a single
//! hidden state, this is a **high-precision** signal: it conditions on
//! the actual routing decision of the immediately preceding layer.
//!
//! ## Mechanism
//!
//! A per-source-expert frequency table is maintained for every layer
//! transition `L → L+1`. Each `moe_step` call:
//!   1. records the transition from the previously-seen layer's routed
//!      set to the current one (when they are consecutive), and
//!   2. predicts the *next* layer's experts from the current routed set,
//!      returning them for the engine to prefetch.
//!
//! The whole feature is gated behind `pregate_enabled`; when off the
//! engine never constructs a [`PerLayerPreGate`] and behaves exactly as
//! before.
//!
//! ## Concurrency
//!
//! The "previous layer" link uses a single slot, which is exact for
//! sequential decoding (the benchmark and single-stream serving). Under
//! heavily-batched concurrent decoding the slot can occasionally link
//! layers from different in-flight positions; because every prediction
//! only drives *speculative* prefetch (wrong guesses waste bandwidth the
//! Tier 4 governor can throttle, never correctness), this is acceptable
//! and self-correcting as the frequency tables converge.

use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Predicted-confidence score attached to pre-gate prefetches. High by
/// design — the prediction conditions on the previous layer's actual
/// routing — so the Tier 4 governor admits it ahead of the low-value
/// single-hidden-state speculation.
pub const PREGATE_PREFETCH_PROB: f64 = 0.9;

/// Per-source frequency table over next-layer targets.
type TargetTable = Mutex<HashMap<u32, u64>>;

/// One read-only pre-gate candidate. `score` is an accumulated transition
/// count, not a calibrated probability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreGatePrediction {
    pub expert_id: u32,
    pub score: u64,
}

/// Online layer-to-layer conditional expert predictor. See the module
/// docs for the model.
#[derive(Debug)]
pub struct PerLayerPreGate {
    /// `transitions[L]` maps a source expert routed in layer `L` to a
    /// frequency table over the experts routed in layer `L+1`. Indexed
    /// by source layer; the final layer has no successor so its slot is
    /// never written.
    transitions: Vec<DashMap<u32, TargetTable>>,
    /// Number of next-layer experts to predict / prefetch per step.
    top_n: usize,
    /// Previous `moe_step`'s `(layer, routed_set, predicted_next)`. Used
    /// to (a) link consecutive layers for transition recording and (b)
    /// score the previous prediction against the now-known actual set.
    last: Mutex<Option<LastStep>>,
    /// Predictions that intersected the actually-routed next set.
    hits: AtomicU64,
    /// Predictions that missed.
    misses: AtomicU64,
}

#[derive(Debug, Clone)]
struct LastStep {
    layer: u32,
    routed: Vec<u32>,
    predicted_next: Vec<u32>,
}

impl PerLayerPreGate {
    /// Build a pre-gate over `num_layers` MoE layers predicting `top_n`
    /// experts per transition. `top_n` is clamped to at least 1.
    pub fn new(num_layers: usize, top_n: usize) -> Self {
        let layers = num_layers.max(1);
        let transitions = (0..layers).map(|_| DashMap::new()).collect();
        Self {
            transitions,
            top_n: top_n.max(1),
            last: Mutex::new(None),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Record the transition into `routed` at `layer` (from the
    /// previously-observed layer, when consecutive), score the prior
    /// prediction, and return the predicted expert set for layer
    /// `layer + 1` so the caller can prefetch it. Returns an empty
    /// vector until the relevant transition table has been seen at least
    /// once.
    pub fn observe_and_predict(&self, layer: u32, routed: &[u32]) -> Vec<u32> {
        {
            let last = self.last.lock();
            if let Some(prev) = last.as_ref() {
                // Only link strictly-consecutive layers. Guard against the
                // `u32::MAX` sentinel wrapping back to `0` and falsely
                // linking an unrelated first layer onto a stale final-layer
                // observation.
                if prev.layer != u32::MAX && layer == prev.layer.wrapping_add(1) {
                    self.observe_transition(prev.layer, &prev.routed, routed);
                    self.score_prediction(&prev.predicted_next, routed);
                }
            }
            // Defer storing `last` until the prediction is computed so the
            // borrow of `routed` is unambiguous; we update it below.
            drop(last);
        }

        let predicted = self
            .predict_ranked(layer, routed, self.top_n)
            .into_iter()
            .map(|candidate| candidate.expert_id)
            .collect::<Vec<_>>();

        let mut last = self.last.lock();
        *last = Some(LastStep {
            layer,
            routed: routed.to_vec(),
            predicted_next: predicted.clone(),
        });
        predicted
    }

    /// Record one causally observed `src_layer -> src_layer + 1`
    /// transition. This method performs no prediction, scoring, prefetch, or
    /// external state change, which lets diagnostic callers enforce an
    /// explicit predict -> score -> update order.
    pub fn observe_transition(&self, src_layer: u32, source_set: &[u32], target_set: &[u32]) {
        let Some(layer_map) = self.transitions.get(src_layer as usize) else {
            return;
        };
        for &a in source_set {
            let entry = layer_map.entry(a).or_default();
            let mut table = entry.lock();
            for &b in target_set {
                *table.entry(b).or_insert(0) += 1;
            }
        }
    }

    /// Predict layer `layer + 1`'s experts from the current routed set by
    /// summing each source expert's learned target frequencies and
    /// taking the `top_n` highest.
    /// Return a deterministic read-only ranking for `layer + 1` from the
    /// already-observed routed set at `layer`. Ties break by ascending expert
    /// id. The returned score is the summed transition count and therefore is
    /// deliberately not represented as a probability.
    pub fn predict_ranked(
        &self,
        layer: u32,
        routed: &[u32],
        limit: usize,
    ) -> Vec<PreGatePrediction> {
        if limit == 0 {
            return Vec::new();
        }
        let Some(layer_map) = self.transitions.get(layer as usize) else {
            return Vec::new();
        };
        let mut scores: HashMap<u32, u64> = HashMap::new();
        for &a in routed {
            if let Some(entry) = layer_map.get(&a) {
                let table = entry.lock();
                for (&b, &c) in table.iter() {
                    *scores.entry(b).or_insert(0) += c;
                }
            }
        }
        if scores.is_empty() {
            return Vec::new();
        }
        let mut ranked: Vec<(u32, u64)> = scores.into_iter().collect();
        // Highest score first; ties break by ascending id for determinism.
        ranked.sort_unstable_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
        ranked.truncate(limit);
        ranked
            .into_iter()
            .map(|(expert_id, score)| PreGatePrediction { expert_id, score })
            .collect()
    }

    /// Update hit/miss telemetry for a prediction against the actual set.
    fn score_prediction(&self, predicted: &[u32], actual: &[u32]) {
        if predicted.is_empty() {
            return;
        }
        let intersects = predicted.iter().any(|p| actual.contains(p));
        if intersects {
            self.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// `(hits, misses)` for the predictions scored so far.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits.load(Ordering::Relaxed), self.misses.load(Ordering::Relaxed))
    }

    /// Fraction of scored predictions that intersected the actual
    /// next-layer set, in `[0, 1]`. `0.0` when nothing has been scored.
    pub fn accuracy(&self) -> f64 {
        let (h, m) = self.stats();
        let total = h + m;
        if total == 0 {
            0.0
        } else {
            h as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_and_predicts_consecutive_layer_transition() {
        // 3 MoE layers, predict up to 2 next-layer experts.
        let pg = PerLayerPreGate::new(3, 2);

        // First forward pass: layer0 {0,1} -> layer1 {10,11} -> layer2 {20}.
        // No predictions yet (tables empty), but transitions are recorded.
        assert!(pg.observe_and_predict(0, &[0, 1]).is_empty());
        assert!(pg.observe_and_predict(1, &[10, 11]).is_empty());
        assert!(pg.observe_and_predict(2, &[20]).is_empty());

        // Reset the consecutive-layer link by starting a fresh pass at
        // layer 0 — the second pass routes the same way.
        let p0 = pg.observe_and_predict(0, &[0, 1]);
        // Now layer 0's routed set {0,1} should predict layer 1's {10,11}.
        let mut got = p0.clone();
        got.sort_unstable();
        assert_eq!(got, vec![10, 11], "layer0 set predicts layer1 experts");

        let p1 = pg.observe_and_predict(1, &[10, 11]);
        assert_eq!(p1, vec![20], "layer1 set predicts layer2 experts");
    }

    #[test]
    fn non_consecutive_calls_do_not_record() {
        let pg = PerLayerPreGate::new(4, 2);
        // Jump from layer 0 to layer 2 (a non-consecutive link): nothing
        // should be learned for transition 0->1.
        pg.observe_and_predict(0, &[1, 2]);
        pg.observe_and_predict(2, &[5, 6]);
        // Re-routing layer 0 the same way yields no prediction.
        assert!(pg.observe_and_predict(0, &[1, 2]).is_empty());
    }

    #[test]
    fn accuracy_tracks_prediction_quality() {
        let pg = PerLayerPreGate::new(2, 2);
        // Train 0->1: {0} -> {9}.
        pg.observe_and_predict(0, &[0]);
        pg.observe_and_predict(1, &[9]);
        // Second pass: predict {9} from {0}; actual next is {9} ⇒ hit.
        pg.observe_and_predict(0, &[0]);
        pg.observe_and_predict(1, &[9]);
        let (hits, misses) = pg.stats();
        assert_eq!((hits, misses), (1, 0), "one correctly-scored prediction");
        assert!((pg.accuracy() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ranked_prediction_is_deterministic_and_breaks_equal_scores_by_id() {
        let pg = PerLayerPreGate::new(2, 8);
        pg.observe_transition(0, &[3], &[9, 2, 7]);
        pg.observe_transition(0, &[4], &[7, 2, 9]);

        let first = pg.predict_ranked(0, &[4, 3], 8);
        let second = pg.predict_ranked(0, &[3, 4], 8);
        assert_eq!(first, second);
        assert_eq!(
            first
                .iter()
                .map(|prediction| prediction.expert_id)
                .collect::<Vec<_>>(),
            vec![2, 7, 9]
        );
        assert!(first.iter().all(|prediction| prediction.score == 2));
    }

    #[test]
    fn target_route_cannot_enter_its_own_frozen_prediction() {
        let pg = PerLayerPreGate::new(2, 8);
        pg.observe_transition(0, &[1], &[9]);

        let frozen = pg.predict_ranked(0, &[1], 8);
        assert_eq!(frozen[0].expert_id, 9);
        assert!(!frozen.iter().any(|prediction| prediction.expert_id == 4));

        // Scoring uses `frozen`; only afterward may the just-observed target
        // update the transition table used by later predictions.
        let scored_target = [4];
        let frozen_hit = frozen
            .iter()
            .any(|prediction| scored_target.contains(&prediction.expert_id));
        assert!(!frozen_hit);
        pg.observe_transition(0, &[1], &scored_target);

        let later = pg.predict_ranked(0, &[1], 8);
        assert!(later.iter().any(|prediction| prediction.expert_id == 4));
    }
}
