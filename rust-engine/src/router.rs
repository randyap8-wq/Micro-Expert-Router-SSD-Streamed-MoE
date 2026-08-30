//! Deterministic Markov-chain router and Markov predictive prefetcher.
//!
//! For an MoE engine whose interesting axis is **I/O**, a router that picks
//! experts uniformly at random is actively misleading: it has no temporal
//! correlation, so any prefetch policy looks worthless. Real MoE traces show
//! strong locality — once a stream wanders into a "topic", it tends to keep
//! activating the same handful of experts for many tokens. We therefore
//! model the router as a **first-order Markov chain over expert ids**.
//!
//! Two components live here:
//!
//! 1. [`TopKRouter`] picks `k` distinct expert ids per token by sampling
//!    from `P(next | last_expert)`. The transition matrix is either:
//!    * **Generated** with structured locality — experts are split into
//!      `cluster_count` groups and stay inside their group with a high
//!      probability ([`TopKRouter::clustered`]).
//!    * **Loaded** from a file containing a row-stochastic `N x N` matrix
//!      (whitespace-separated `f64` values, row-major — easy to feed real
//!      routing traces; see [`TopKRouter::from_matrix_file`]).
//!    The router is fully deterministic given a seed.
//!
//! 2. [`PredictiveLoader`] is a first-order Markov *predictor* built
//!    online from observed activations and used by the engine to issue
//!    speculative prefetches. The router is fixed; the predictor learns.

use parking_lot::RwLock;
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::{HashMap, VecDeque};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::fs;
use std::io;
use std::path::Path;

/// Picks the top-K experts for a given token from a first-order Markov
/// chain over expert ids. Distinct ids guaranteed.
pub struct TopKRouter {
    num_experts: u32,
    k: usize,
    /// Row-stochastic `N x N` transition matrix: `transition[from][to]`
    /// is `P(next = to | last = from)`.
    transition: Vec<Vec<f64>>,
    /// Initial distribution used when no expert has been activated yet.
    initial: Vec<f64>,
    rng: parking_lot::Mutex<StdRng>,
    /// Last expert id this router emitted; drives the next sample. The
    /// router is therefore *stateful*; given a seed and a stream of
    /// `route()` calls, the output sequence is fully reproducible.
    last_expert: parking_lot::Mutex<Option<u32>>,
}

impl TopKRouter {
    /// Build a router from a precomputed transition matrix. Each row must
    /// have exactly `num_experts` entries and sum to a positive value
    /// (it is normalised to a probability distribution internally).
    pub fn from_matrix(num_experts: u32, k: usize, transition: Vec<Vec<f64>>, seed: u64) -> Self {
        assert!(k as u32 <= num_experts, "k must be <= num_experts");
        assert!(k > 0, "k must be > 0");
        let n = num_experts as usize;
        assert_eq!(
            transition.len(),
            n,
            "transition matrix must have num_experts rows"
        );
        let normalised: Vec<Vec<f64>> = transition
            .into_iter()
            .enumerate()
            .map(|(row_idx, row)| {
                assert_eq!(
                    row.len(),
                    n,
                    "row {row_idx} must have num_experts columns"
                );
                normalise_row(&row, n)
            })
            .collect();
        // Initial distribution = uniform over all experts. (Equivalently
        // we could use the stationary distribution of the chain, but
        // uniform is just as deterministic and obviously cluster-agnostic.)
        let initial = vec![1.0 / n as f64; n];
        Self {
            num_experts,
            k,
            transition: normalised,
            initial,
            rng: parking_lot::Mutex::new(StdRng::seed_from_u64(seed)),
            last_expert: parking_lot::Mutex::new(None),
        }
    }

    /// Build a router whose transition matrix has **structured cluster
    /// locality**: experts are partitioned into `cluster_count` groups
    /// (by `id % cluster_count`), and the chain stays inside the
    /// current cluster with probability `intra_cluster_p` (uniform
    /// across in-cluster experts) and jumps out of cluster with the
    /// remaining mass (uniform across the rest of the model).
    ///
    /// This is the synthetic stand-in used when no real routing trace is
    /// supplied. Defaults: `cluster_count = 4`, `intra_cluster_p = 0.9`,
    /// matching the gist's "4 clusters, mostly stay in cluster" example.
    pub fn clustered(
        num_experts: u32,
        k: usize,
        cluster_count: usize,
        intra_cluster_p: f64,
        seed: u64,
    ) -> Self {
        assert!(num_experts >= 1);
        assert!(cluster_count >= 1);
        assert!(
            (0.0..=1.0).contains(&intra_cluster_p),
            "intra_cluster_p must be in [0, 1], got {intra_cluster_p}"
        );
        let n = num_experts as usize;
        let cluster_count = cluster_count.min(n);
        // Assign each expert id to a cluster by `id % cluster_count` so
        // even small N (e.g. 8 experts in 4 clusters) lands two members
        // per cluster and cluster locality is meaningful.
        let cluster_of = |id: usize| id % cluster_count;
        let mut transition = vec![vec![0.0_f64; n]; n];
        for from in 0..n {
            let from_c = cluster_of(from);
            let in_cluster: Vec<usize> = (0..n).filter(|&j| cluster_of(j) == from_c).collect();
            let out_cluster: Vec<usize> = (0..n).filter(|&j| cluster_of(j) != from_c).collect();
            // Distribute `intra_cluster_p` uniformly across in-cluster
            // experts and `1 - intra_cluster_p` across out-of-cluster
            // experts. Edge case: only one cluster — all mass stays in.
            let p_in = if in_cluster.is_empty() {
                0.0
            } else {
                intra_cluster_p / in_cluster.len() as f64
            };
            let p_out = if out_cluster.is_empty() {
                0.0
            } else {
                (1.0 - intra_cluster_p) / out_cluster.len() as f64
            };
            for &j in &in_cluster {
                transition[from][j] = p_in;
            }
            for &j in &out_cluster {
                transition[from][j] = p_out;
            }
            // If single cluster (out_cluster empty) and intra_cluster_p < 1,
            // fold the remaining mass back into the in-cluster row so the
            // row still sums to 1. `in_cluster` is non-empty here:
            // every `from` is in some cluster, so its own cluster
            // contains at least itself. The `.max(1)` is belt-and-suspenders
            // against `cluster_count > num_experts` configurations.
            if out_cluster.is_empty() {
                let extra = (1.0 - intra_cluster_p) / in_cluster.len().max(1) as f64;
                for &j in &in_cluster {
                    transition[from][j] += extra;
                }
            }
        }
        Self::from_matrix(num_experts, k, transition, seed)
    }

    /// Load a row-stochastic transition matrix from a text file. The
    /// file must contain `num_experts * num_experts` whitespace-separated
    /// `f64` values laid out row-major. Rows are normalised so each sums
    /// to 1; degenerate (all-zero) rows are replaced with the uniform
    /// distribution. Use this to feed a real Mixtral routing trace
    /// without rebuilding.
    pub fn from_matrix_file(
        path: &Path,
        num_experts: u32,
        k: usize,
        seed: u64,
    ) -> io::Result<Self> {
        let n = num_experts as usize;
        let text = fs::read_to_string(path)?;
        let values: Vec<f64> = text
            .split_ascii_whitespace()
            .map(|tok| tok.parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        if values.len() != n * n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "router matrix file {} has {} values; expected {} (num_experts^2)",
                    path.display(),
                    values.len(),
                    n * n
                ),
            ));
        }
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        for r in 0..n {
            rows.push(values[r * n..(r + 1) * n].to_vec());
        }
        Ok(Self::from_matrix(num_experts, k, rows, seed))
    }

    /// Convenience constructor for tests / call sites that don't care
    /// about the matrix shape and just want a router with the gist's
    /// default cluster locality (4 clusters, 90% intra-cluster).
    pub fn new(num_experts: u32, k: usize, seed: u64) -> Self {
        Self::clustered(num_experts, k, 4, 0.9, seed)
    }

    /// Force a specific routing decision (used by the CLI to reproduce
    /// the "Router selects Expert ID 3 and 7" example from the spec).
    /// Updates the chain state to the last forced id so the next
    /// `route()` continues from there.
    pub fn fixed(&self, ids: &[u32]) -> Vec<u32> {
        let chosen: Vec<u32> = ids
            .iter()
            .filter(|i| **i < self.num_experts)
            .copied()
            .collect();
        if let Some(&last) = chosen.last() {
            *self.last_expert.lock() = Some(last);
        }
        chosen
    }

    /// Sample `k` distinct expert ids from `P(next | last_expert)`. The
    /// `_token_idx` argument is preserved so the engine API doesn't
    /// change, but the routing is now driven by the Markov chain
    /// internal state, not the token index. The RNG is seeded once at
    /// construction, so a full run is reproducible.
    pub fn route(&self, _token_idx: u64) -> Vec<u32> {
        let n = self.num_experts as usize;
        let mut last_guard = self.last_expert.lock();
        let row: &[f64] = match *last_guard {
            Some(prev) if (prev as usize) < n => &self.transition[prev as usize],
            _ => &self.initial,
        };
        // Materialise weights so we can zero-out picked experts to
        // guarantee `k` distinct results without re-rolling.
        let mut weights: Vec<f64> = row.to_vec();
        let mut chosen = Vec::with_capacity(self.k);
        let mut rng = self.rng.lock();
        for _ in 0..self.k {
            let dist = match WeightedIndex::new(&weights) {
                Ok(d) => d,
                Err(_) => break,
            };
            let idx = dist.sample(&mut *rng);
            chosen.push(idx as u32);
            weights[idx] = 0.0;
        }
        // Advance chain state. We use the *first* (highest-probability
        // sample on average) chosen id as the next "last expert" so the
        // chain reflects the dominant per-token activation; this keeps
        // cluster locality visible even when k > 1.
        if let Some(&first) = chosen.first() {
            *last_guard = Some(first);
        }
        chosen
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn num_experts(&self) -> u32 {
        self.num_experts
    }
}

/// Normalise a row to a probability distribution. All-zero or negative-sum
/// rows fall back to a uniform distribution so the router never panics on
/// a malformed input matrix.
fn normalise_row(row: &[f64], n: usize) -> Vec<f64> {
    let sum: f64 = row.iter().filter(|&&v| v.is_finite() && v > 0.0).sum();
    if sum > 0.0 && sum.is_finite() {
        row.iter()
            .map(|&v| if v.is_finite() && v > 0.0 { v / sum } else { 0.0 })
            .collect()
    } else {
        vec![1.0 / n as f64; n]
    }
}

/// First-order Markov predictor: P(next | last_expert).
///
/// Transitions are stored as a **sparse** per-row map of observation counts:
/// `rows[from]` holds only the successor ids that have been observed at
/// least once, plus a cached row total. A flat `[N x N]` dense matrix would
/// allocate `O(N²)` u32 cells up front (~1 MiB at N=512, ~64 MiB at
/// N=4096) regardless of how many distinct transitions we actually
/// observe — sparse-by-row scales with the number of *visited* (from, to)
/// pairs instead, which is the natural footprint of an MoE routing trace
/// (each token activates only `k` experts so per-token only `k²` pairs
/// are observed).
///
/// Probabilities are still smoothed with a Laplace prior `prior` per
/// transition so unseen successors have non-zero probability and the
/// predictor doesn't need a warm-up phase. The prior is applied
/// implicitly: an absent map entry counts as `prior`, an entry with
/// stored count `c` counts as `c + prior`, and the row's effective total
/// is `total_observed[from] + num_experts * prior`.
struct Row {
    /// Observed counts beyond the prior; absent keys mean "0 observed".
    counts: HashMap<u32, u32>,
    /// Sum of `counts.values()` (i.e. observations beyond the prior).
    total_observed: u64,
}

impl Row {
    fn new() -> Self {
        Self { counts: HashMap::new(), total_observed: 0 }
    }
}

pub struct PredictiveLoader {
    num_experts: u32,
    /// `rows[from]` — sparse counts of successors of `from` (1st-order).
    rows: RwLock<Vec<Row>>,
    /// `rows2[(prev_prev, prev)]` — sparse counts of successors of the
    /// pair `(prev_prev -> prev)`. Stored as a `HashMap` keyed by the
    /// flat index `prev_prev * num_experts + prev` to keep the memory
    /// footprint sparse — only pairs we actually observed are
    /// allocated. Used by [`Self::predict_next`] to blend a 2nd-order
    /// signal with the 1st-order baseline.
    rows2: RwLock<HashMap<u64, Row>>,
    /// Number of successors to suggest per call.
    fanout: usize,
    /// Probability threshold below which we don't bother prefetching.
    min_prob: f64,
    /// Prior added to every transition to smooth the empty-state case.
    prior: u32,
    rng: parking_lot::Mutex<StdRng>,
}

impl PredictiveLoader {
    pub fn new(num_experts: u32, fanout: usize, min_prob: f64, seed: u64) -> Self {
        let n = num_experts as usize;
        let mut rows = Vec::with_capacity(n);
        for _ in 0..n {
            rows.push(Row::new());
        }
        Self {
            num_experts,
            rows: RwLock::new(rows),
            rows2: RwLock::new(HashMap::new()),
            fanout,
            min_prob,
            prior: 1,
            rng: parking_lot::Mutex::new(StdRng::seed_from_u64(seed.wrapping_add(0xDEAD))),
        }
    }

    /// Record that `to` was activated immediately after `from`.
    pub fn observe(&self, from: u32, to: u32) {
        if from >= self.num_experts || to >= self.num_experts {
            return;
        }
        let mut rows = self.rows.write();
        let row = &mut rows[from as usize];
        let entry = row.counts.entry(to).or_insert(0);
        // Saturate at u32::MAX rather than wrapping. Once a cell
        // saturates, further observations stop counting toward the row
        // total too — otherwise the implied probability
        // `(*entry + prior) / total` would drift away from the true
        // frequency as `total` grew past `*entry`.
        if *entry < u32::MAX {
            *entry += 1;
            row.total_observed = row.total_observed.saturating_add(1);
        }
    }

    /// Record a 2nd-order observation: `(prev_prev -> prev -> to)` was
    /// the actual sequence over three consecutive tokens.
    pub fn observe2(&self, prev_prev: u32, prev: u32, to: u32) {
        let n = self.num_experts;
        if prev_prev >= n || prev >= n || to >= n {
            return;
        }
        let key = (prev_prev as u64) * (n as u64) + prev as u64;
        let mut rows2 = self.rows2.write();
        let row = rows2.entry(key).or_insert_with(Row::new);
        let entry = row.counts.entry(to).or_insert(0);
        if *entry < u32::MAX {
            *entry += 1;
            row.total_observed = row.total_observed.saturating_add(1);
        }
    }

    /// Record a whole batch of activations from a single token's expert set.
    /// Each previous-token expert transitions to each current-token expert.
    pub fn observe_step(&self, prev_set: &[u32], next_set: &[u32]) {
        // Take the write lock once for the whole batch (top-K² updates
        // per call) rather than re-acquiring it inside every iteration.
        // With top-K=8 the inner loop fires 64 updates per token, and
        // the previous per-update lock acquisition dominated the
        // routing tail under continuous batching.
        if prev_set.is_empty() || next_set.is_empty() {
            return;
        }
        let n = self.num_experts;
        let mut rows = self.rows.write();
        for &from in prev_set {
            if from >= n {
                continue;
            }
            let row = &mut rows[from as usize];
            for &to in next_set {
                if to >= n {
                    continue;
                }
                let entry = row.counts.entry(to).or_insert(0);
                if *entry < u32::MAX {
                    *entry += 1;
                    row.total_observed = row.total_observed.saturating_add(1);
                }
            }
        }
    }

    /// Record both the 1st-order `(prev -> next)` and the 2nd-order
    /// `(prev_prev -> prev -> next)` transitions. `prev_prev_set` may
    /// be empty (e.g. very first token after a cold start), in which
    /// case only the 1st-order observations are recorded.
    pub fn observe_step2(&self, prev_prev_set: &[u32], prev_set: &[u32], next_set: &[u32]) {
        self.observe_step(prev_set, next_set);
        for &pp in prev_prev_set {
            for &p in prev_set {
                for &n in next_set {
                    self.observe2(pp, p, n);
                }
            }
        }
    }

    /// Effective total for a row including the smoothing prior over all `n`
    /// possible successors.
    #[inline]
    fn effective_total(&self, total_observed: u64) -> u64 {
        total_observed + (self.num_experts as u64) * (self.prior as u64)
    }

    /// Predict up to `fanout` successors for `from`, weighted by transition
    /// probability. Returns `(expert_id, p)` pairs sorted by descending `p`,
    /// filtered by `min_prob`.
    pub fn predict_next(&self, from: u32) -> Vec<(u32, f64)> {
        if from >= self.num_experts || self.fanout == 0 {
            return Vec::new();
        }
        let n = self.num_experts as usize;
        let rows = self.rows.read();
        let row = &rows[from as usize];
        let total = self.effective_total(row.total_observed);
        if total == 0 {
            return Vec::new();
        }
        let total_f = total as f64;
        let prior_f = self.prior as f64;

        // Observed successors get their (count + prior) probability.
        let mut probs: Vec<(u32, f64)> = row
            .counts
            .iter()
            .map(|(&id, &c)| (id, (c as f64 + prior_f) / total_f))
            .filter(|&(_, p)| p >= self.min_prob)
            .collect();

        // Top up the result with unseen successors at the smoothed
        // prior probability when there is still room in the fanout. The
        // fanout cap is the only bound we need here — previously this
        // path was also gated by `prior_p >= self.min_prob`, but with
        // `prior_p = 1 / (total_observed + num_experts)` that guard is
        // mathematically impossible to clear once `num_experts` is even
        // modestly large (e.g. `min_prob = 0.05`, `num_experts = 64`
        // requires negative observations), which silently killed all
        // speculative prefetches in production-sized configurations.
        // Iterate in id order for determinism.
        let prior_p = prior_f / total_f;
        if probs.len() < self.fanout {
            for id in 0..n as u32 {
                if probs.len() >= self.fanout {
                    break;
                }
                if !row.counts.contains_key(&id) {
                    probs.push((id, prior_p));
                }
            }
        }

        probs.sort_by(|a, b| b.1.total_cmp(&a.1));
        probs.truncate(self.fanout);
        probs
    }

    /// Deterministic, read-only adapter for shadow qualification. This keeps
    /// the production first-order probability, smoothing, threshold, and
    /// unseen-candidate semantics, but applies an explicit ascending-id tie
    /// break before the caller-supplied limit. Ordinary production prediction
    /// continues to use [`Self::predict_next`] unchanged.
    pub(crate) fn predict_next_shadow_ranked(&self, from: u32, limit: usize) -> Vec<(u32, f64)> {
        if from >= self.num_experts || self.fanout == 0 || limit == 0 {
            return Vec::new();
        }
        let n = self.num_experts as usize;
        let rows = self.rows.read();
        let row = &rows[from as usize];
        let total_f = self.effective_total(row.total_observed) as f64;
        let prior_f = self.prior as f64;
        let mut ranked = row
            .counts
            .iter()
            .map(|(&id, &count)| (id, (count as f64 + prior_f) / total_f))
            .filter(|&(_, probability)| probability >= self.min_prob)
            .collect::<Vec<_>>();
        let prior_probability = prior_f / total_f;
        if ranked.len() < self.fanout {
            for id in 0..n as u32 {
                if ranked.len() >= self.fanout {
                    break;
                }
                if !row.counts.contains_key(&id) {
                    ranked.push((id, prior_probability));
                }
            }
        }
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        ranked.truncate(limit.min(self.fanout));
        ranked
    }

    /// 2nd-order variant of [`Self::predict_next`]. When a `(prev_prev,
    /// prev)` row has been observed, blends its distribution with the
    /// 1st-order distribution from `prev` (50/50). Falls back to pure
    /// 1st-order when no 2nd-order data exists for the pair, so the
    /// caller can use this unconditionally without warm-up.
    pub fn predict_next2(&self, prev_prev: u32, prev: u32) -> Vec<(u32, f64)> {
        let baseline = self.predict_next(prev);
        if prev_prev >= self.num_experts || prev >= self.num_experts || self.fanout == 0 {
            return baseline;
        }
        let key = (prev_prev as u64) * (self.num_experts as u64) + prev as u64;
        let rows2 = self.rows2.read();
        let Some(row2) = rows2.get(&key) else {
            return baseline;
        };
        if row2.total_observed == 0 {
            return baseline;
        }
        let total = self.effective_total(row2.total_observed);
        let total_f = total as f64;
        let prior_f = self.prior as f64;
        // Build a combined probability map: blend 1st-order and 2nd-order.
        let mut combined: HashMap<u32, f64> = HashMap::new();
        for (id, p) in &baseline {
            *combined.entry(*id).or_insert(0.0) += 0.5 * *p;
        }
        for (&id, &c) in &row2.counts {
            let p = (c as f64 + prior_f) / total_f;
            *combined.entry(id).or_insert(0.0) += 0.5 * p;
        }
        let mut out: Vec<(u32, f64)> = combined
            .into_iter()
            .filter(|&(_, p)| p >= self.min_prob)
            .collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out.truncate(self.fanout);
        out
    }

    /// Deterministic, read-only second-order adapter for shadow
    /// qualification. It preserves the production 50/50 blend and fallback
    /// semantics while making equal-score ordering explicit. Production uses
    /// [`Self::predict_next2`] unchanged.
    pub(crate) fn predict_next2_shadow_ranked(
        &self,
        prev_prev: u32,
        prev: u32,
        limit: usize,
    ) -> Vec<(u32, f64)> {
        let baseline = self.predict_next_shadow_ranked(prev, self.fanout);
        if prev_prev >= self.num_experts
            || prev >= self.num_experts
            || self.fanout == 0
            || limit == 0
        {
            return baseline.into_iter().take(limit).collect();
        }
        let key = (prev_prev as u64) * (self.num_experts as u64) + prev as u64;
        let rows2 = self.rows2.read();
        let Some(row2) = rows2.get(&key) else {
            return baseline.into_iter().take(limit).collect();
        };
        if row2.total_observed == 0 {
            return baseline.into_iter().take(limit).collect();
        }
        let total_f = self.effective_total(row2.total_observed) as f64;
        let prior_f = self.prior as f64;
        let mut combined = HashMap::<u32, f64>::new();
        for (id, probability) in baseline {
            *combined.entry(id).or_insert(0.0) += 0.5 * probability;
        }
        for (&id, &count) in &row2.counts {
            let probability = (count as f64 + prior_f) / total_f;
            *combined.entry(id).or_insert(0.0) += 0.5 * probability;
        }
        let mut ranked = combined
            .into_iter()
            .filter(|&(_, probability)| probability >= self.min_prob)
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        ranked.truncate(limit.min(self.fanout));
        ranked
    }

    /// **Unified prediction.** Combines the Markov-chain row predictions
    /// with the [`LocalityMonitor`]'s current hot set and (optionally)
    /// the [`NeuralSpeculator`]'s top-K to produce a single ranked list
    /// of `(expert_id, priority_score)` pairs. This is the API called
    /// out in the design spec under Task 1: *"the predictor should now
    /// be able to return a unified set of (expert_id, priority_score)"*.
    ///
    /// Scoring (in `[0, 1]`):
    /// * Markov contribution: probability from
    ///   [`Self::predict_next2`] (or [`Self::predict_next`] when
    ///   `prev_prev` is `None`), weighted **W_MARKOV**.
    /// * Locality contribution: **W_LOCALITY** (rank-decayed, see
    ///   [`RANK_TIEBREAK_DECAY`]) for every expert in
    ///   `monitor.get_hot_experts(threshold_pct)` — these are
    ///   temporally stable hot experts and we want them in the
    ///   prefetch set even when the Markov chain is uncertain.
    /// * Speculator contribution: **W_SPECULATOR** (rank-decayed) for
    ///   every expert in `speculator.predict_topk(hidden, speculator_k)` —
    ///   semantic intent is the strongest single signal in the design.
    ///
    /// The three constants sum to **1.0**, so an expert hit by all
    /// three arms with maximal Markov probability tops out at exactly
    /// `1.0` rather than overshooting (the previous design summed to
    /// 1.2, which made the score ill-defined as a probability and
    /// made cross-arm comparisons against `[0, 1]` thresholds
    /// awkward). Their relative weighting still encodes the design
    /// intent: speculator > Markov > locality.
    ///
    /// The three contributions sum, then the result is sorted by
    /// descending score (ties broken by ascending id for determinism)
    /// and truncated to `self.fanout`. An expert that lights up in
    /// every arm is therefore prioritised over one that lights up in
    /// only one.
    ///
    /// Hidden state is borrowed (not cloned) so this is safe to call
    /// on the hot path; the speculator forward is its own internal
    /// allocation.
    pub fn predict_unified(
        &self,
        prev_prev: Option<u32>,
        prev: u32,
        monitor: Option<&LocalityMonitor>,
        threshold_pct: f32,
        speculator: Option<&NeuralSpeculator>,
        hidden: &[f32],
        speculator_k: usize,
    ) -> Vec<(u32, f32)> {
        // Gather the three predictive arms, then fuse them with the
        // canonical weights via [`Self::combine_unified_arms`]. Keeping
        // the gather/fuse split lets the engine's hot path reuse the
        // exact same weighting with a speculator top-K it already has in
        // hand (so it never re-runs the forward), while this offline API
        // stays a one-call convenience.
        let markov = match prev_prev {
            Some(pp) => self.predict_next2(pp, prev),
            None => self.predict_next(prev),
        };
        let locality_ids: Vec<u32> = monitor
            .map(|m| m.get_hot_experts(threshold_pct))
            .unwrap_or_default();
        let speculator_ids: Vec<u32> = match speculator {
            Some(s) if speculator_k > 0 => s.predict_topk(hidden, speculator_k),
            _ => Vec::new(),
        };
        let mut out = self.combine_unified_arms(&markov, &locality_ids, &speculator_ids);
        // Honour the loader's fanout when set; otherwise return all.
        if self.fanout > 0 {
            out.truncate(self.fanout);
        }
        out
    }

    /// Fuse three **precomputed** predictive arms into a single ranked
    /// `(expert_id, score)` list using the canonical unified weights
    /// ([`W_MARKOV`] / [`W_LOCALITY`] / [`W_SPECULATOR`]). Each arm's
    /// contribution sums, so an id hit by multiple arms outranks one hit
    /// by a single arm; the result is sorted by descending score (ties
    /// broken by ascending id for determinism).
    ///
    /// Unlike [`Self::predict_unified`] this takes the arms as inputs
    /// rather than computing them, so a caller that already evaluated
    /// the neural speculator (e.g. the engine, which runs and trains it
    /// once per token) can apply the documented weighting without a
    /// second forward pass. The result is **not** truncated to the
    /// loader's fanout — callers that want that apply it themselves.
    pub fn combine_unified_arms(
        &self,
        markov: &[(u32, f64)],
        locality_ids: &[u32],
        speculator_ids: &[u32],
    ) -> Vec<(u32, f32)> {
        // Compile-time invariant: the headline weights sum to 1.0 (within
        // f32 epsilon). Kept as a debug_assert so a future tweak that
        // breaks the contract trips a test rather than silently producing
        // >1 scores.
        debug_assert!((W_SPECULATOR + W_MARKOV + W_LOCALITY - 1.0).abs() < 1e-6);

        let mut combined: HashMap<u32, f32> = HashMap::new();

        // 1) Markov contribution.
        for &(id, p) in markov {
            *combined.entry(id).or_insert(0.0) += W_MARKOV * (p as f32);
        }
        // 2) Locality contribution. The hot set arrives heat-sorted;
        //    apply the tiny per-rank decay so the heat ordering
        //    survives the fuse instead of collapsing into an all-ties
        //    ascending-id sort (see [`RANK_TIEBREAK_DECAY`]). Clamped
        //    at half weight so a pathologically long list can never
        //    invert cross-arm ordering.
        for (rank, &id) in locality_ids.iter().enumerate() {
            let f = (1.0 - RANK_TIEBREAK_DECAY * rank as f32).max(0.5);
            *combined.entry(id).or_insert(0.0) += W_LOCALITY * f;
        }
        // 3) Speculator contribution (logit-ranked top-K; same
        //    rank-decay rationale as the locality arm).
        for (rank, &id) in speculator_ids.iter().enumerate() {
            let f = (1.0 - RANK_TIEBREAK_DECAY * rank as f32).max(0.5);
            *combined.entry(id).or_insert(0.0) += W_SPECULATOR * f;
        }

        let mut out: Vec<(u32, f32)> = combined
            .into_iter()
            .filter(|&(_, p)| p > 0.0)
            .collect();
        out.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.cmp(&b.0))
        });
        out
    }

    /// Extended unified prediction with **spatial prefetching** and
    /// **expert-affinity** contributions on top of the existing
    /// Markov / locality / speculator arms.
    ///
    /// * `affinity` — optional [`ExpertAffinity`] co-occurrence matrix.
    ///   For each expert already in the candidate set with
    ///   `score >= SPATIAL_CONFIDENCE_THRESHOLD`, the top-K affinity
    ///   neighbours are folded in with a flat weight so experts that
    ///   habitually fire together in the same layer get prefetched
    ///   alongside the seed.
    /// * Spatial prefetching: for the same high-confidence seeds, the
    ///   immediate UTH neighbours (`id ± 1`, clipped to
    ///   `[0, num_experts)`) are also enqueued. Pulling them from the
    ///   SSD piggy-backs on the drive's sequential-read locality, so
    ///   their cost is close to free.
    ///
    /// Returns the same `Vec<(expert_id, score)>` shape as
    /// [`Self::predict_unified`] with scores still bounded in
    /// `[0, ~1.2]` (the spatial / affinity weights add a small tail
    /// that nudges co-fired neighbours into the fanout without
    /// overwhelming the headline arms). Sorted by descending score.
    pub fn predict_unified_with_spatial(
        &self,
        prev_prev: Option<u32>,
        prev: u32,
        monitor: Option<&LocalityMonitor>,
        threshold_pct: f32,
        speculator: Option<&NeuralSpeculator>,
        hidden: &[f32],
        speculator_k: usize,
        affinity: Option<&ExpertAffinity>,
        affinity_k: usize,
    ) -> Vec<(u32, f32)> {
        let base = self.predict_unified(
            prev_prev,
            prev,
            monitor,
            threshold_pct,
            speculator,
            hidden,
            speculator_k,
        );
        let mut out = self.fold_spatial_affinity(base, affinity, affinity_k);
        if self.fanout > 0 {
            out.truncate(self.fanout);
        }
        out
    }

    /// Fold the auxiliary **spatial** (UTH disk-adjacency) and
    /// **affinity** (observed co-occurrence) neighbour arms onto an
    /// already-scored candidate list. For every seed scoring
    /// `>= SPATIAL_CONFIDENCE_THRESHOLD`, its immediate UTH neighbours
    /// gain [`W_SPATIAL`] and its top-`affinity_k` co-fired neighbours
    /// gain [`W_AFFINITY`]. Returns the re-ranked list (descending
    /// score); the caller applies any fanout truncation.
    ///
    /// Shared by [`Self::predict_unified_with_spatial`] and the engine's
    /// hot-path prefetch so both apply identical neighbour weights.
    pub fn fold_spatial_affinity(
        &self,
        base: Vec<(u32, f32)>,
        affinity: Option<&ExpertAffinity>,
        affinity_k: usize,
    ) -> Vec<(u32, f32)> {
        // Identify high-confidence seeds (scores >= the spatial
        // threshold). Collected into a Vec so we can mutate the combined
        // map without invalidating the iterator.
        let seeds: Vec<u32> = base
            .iter()
            .filter(|(_, s)| *s >= SPATIAL_CONFIDENCE_THRESHOLD)
            .map(|(id, _)| *id)
            .collect();

        let mut combined: HashMap<u32, f32> = base.into_iter().collect();

        // Spatial neighbour contribution.
        if !seeds.is_empty() {
            let n = self.num_experts;
            for &seed in &seeds {
                for nbr in spatial_neighbors(seed, n, 2) {
                    *combined.entry(nbr).or_insert(0.0) += W_SPATIAL;
                }
            }
        }

        // Affinity-matrix contribution. Same seeds, but pulling from
        // observed co-occurrence rather than UTH adjacency.
        if let Some(aff) = affinity {
            if affinity_k > 0 {
                let n = self.num_experts;
                for &seed in &seeds {
                    for nbr in aff.neighbors(seed, affinity_k) {
                        if nbr < n {
                            *combined.entry(nbr).or_insert(0.0) += W_AFFINITY;
                        }
                    }
                }
            }
        }

        let mut out: Vec<(u32, f32)> = combined
            .into_iter()
            .filter(|&(_, p)| p > 0.0)
            .collect();
        out.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.cmp(&b.0))
        });
        out
    }

    #[allow(dead_code)]
    pub fn sample_next(&self, from: u32) -> Vec<u32> {
        if from >= self.num_experts || self.fanout == 0 {
            return Vec::new();
        }
        let n = self.num_experts as usize;
        let rows = self.rows.read();
        let row = &rows[from as usize];

        let prior_f = self.prior as f64;
        let mut weights: Vec<f64> = vec![prior_f; n];
        for (&id, &c) in &row.counts {
            weights[id as usize] = c as f64 + prior_f;
        }
        drop(rows);

        let mut chosen = Vec::with_capacity(self.fanout);
        let mut rng = self.rng.lock();
        for _ in 0..self.fanout {
            let dist = match WeightedIndex::new(&weights) {
                Ok(d) => d,
                Err(_) => break,
            };
            let idx = dist.sample(&mut *rng);
            chosen.push(idx as u32);
            weights[idx] = 0.0;
        }
        chosen
    }

    #[allow(dead_code)]
    pub fn fanout(&self) -> usize {
        self.fanout
    }

    #[allow(dead_code)]
    pub fn min_prob(&self) -> f64 {
        self.min_prob
    }

    /// Total number of transition observations recorded (each `observe`
    /// call counts as one). Note this counts *every* observation,
    /// including repeated `(from, to)` pairs — it is **not** the number
    /// of distinct `(from, to)` pairs visited.
    pub fn observations(&self) -> u64 {
        let rows = self.rows.read();
        rows.iter().map(|r| r.total_observed).sum()
    }
}

/// Sliding-window **Locality Monitor** — the temporal-stability arm of the
/// dual-path predictive controller.
///
/// The monitor observes every routed expert id and maintains a heat map
/// (frequency count) over a sliding window of the most recent `window`
/// observations. The intent is orthogonal to the Markov predictor: where
/// the Markov chain answers "given the last expert id, what tends to
/// fire next?", the locality monitor answers "regardless of *which*
/// expert just fired, which experts have been disproportionately busy
/// over the last few hundred tokens?". An expert that crosses the
/// `threshold_pct` heat ratio is "hot" and can be **pinned** in the
/// expert cache so it is protected from LRU eviction even when the
/// Markov chain wanders elsewhere.
///
/// Internally:
/// * `window` is a `VecDeque<u32>` of expert ids in arrival order; once
///   it reaches capacity each new observation evicts the oldest entry.
/// * `counts` is a flat `Vec<u32>` indexed by expert id holding the
///   number of times that id appears in the window. The flat layout is
///   `O(num_experts)` memory but reads/writes are cache-line tight.
/// * `total` mirrors `window.len()` so `is_hot` doesn't have to lock the
///   deque just to compute the denominator.
///
/// All accessors lock a single `RwLock`; the monitor is small, the
/// critical section is short, and contention is bounded by the rate at
/// which the engine routes tokens.
pub struct LocalityMonitor {
    num_experts: u32,
    capacity: usize,
    inner: RwLock<LocalityInner>,
}

struct LocalityInner {
    window: VecDeque<u32>,
    counts: Vec<u32>,
    total: u64,
}

impl LocalityMonitor {
    /// Default heat threshold (10% of the window) for declaring an
    /// expert "hot". Mirrors the spec's example.
    pub const DEFAULT_THRESHOLD_PCT: f32 = 0.10;

    pub fn new(num_experts: u32, window: usize) -> Self {
        let capacity = window.max(1);
        Self {
            num_experts,
            capacity,
            inner: RwLock::new(LocalityInner {
                window: VecDeque::with_capacity(capacity),
                counts: vec![0u32; num_experts as usize],
                total: 0,
            }),
        }
    }

    /// Observe a single expert activation. Drops the oldest observation
    /// if the window is full.
    pub fn observe_one(&self, id: u32) {
        if id >= self.num_experts {
            return;
        }
        let mut inner = self.inner.write();
        if inner.window.len() == self.capacity {
            if let Some(old) = inner.window.pop_front() {
                let slot = &mut inner.counts[old as usize];
                if *slot > 0 {
                    *slot -= 1;
                }
            }
        }
        // `total` is the cumulative observation counter; count every
        // observation, not just the ones that grew the window.
        inner.total = inner.total.saturating_add(1);
        inner.window.push_back(id);
        inner.counts[id as usize] = inner.counts[id as usize].saturating_add(1);
    }

    /// Observe a batch of expert activations (e.g. one token's top-K).
    pub fn observe(&self, ids: &[u32]) {
        for &id in ids {
            self.observe_one(id);
        }
    }

    /// Number of observations currently in the window.
    pub fn len(&self) -> usize {
        self.inner.read().window.len()
    }

    /// Whether `id` currently exceeds the heat threshold. `threshold_pct`
    /// is a fraction in `[0, 1]`; e.g. `0.1` means "≥10% of the window".
    /// Always returns `false` when the window is empty (no signal yet).
    pub fn is_hot(&self, id: u32, threshold_pct: f32) -> bool {
        if id >= self.num_experts {
            return false;
        }
        let inner = self.inner.read();
        let len = inner.window.len();
        if len == 0 {
            return false;
        }
        let count = inner.counts[id as usize] as f32;
        let cutoff = threshold_pct.max(0.0) * len as f32;
        // `>=` so a threshold of 0.0 still requires at least one
        // observation (since `count == 0` for absent ids).
        count > 0.0 && count >= cutoff
    }

    /// Snapshot of the current hot set: every id whose count meets the
    /// `threshold_pct` ratio. Sorted descending by heat count, ties
    /// broken by ascending id for determinism.
    pub fn hot_set(&self, threshold_pct: f32) -> Vec<u32> {
        let inner = self.inner.read();
        let len = inner.window.len();
        if len == 0 {
            return Vec::new();
        }
        let cutoff = (threshold_pct.max(0.0) * len as f32).max(1.0);
        let mut out: Vec<(u32, u32)> = inner
            .counts
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| {
                if (c as f32) >= cutoff {
                    Some((i as u32, c))
                } else {
                    None
                }
            })
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out.into_iter().map(|(id, _)| id).collect()
    }

    /// Spec-compliant alias of [`Self::hot_set`]. The "Omniscient
    /// Predictive Architecture" design document calls this method
    /// `get_hot_experts(threshold)`; the original implementation
    /// shipped under the shorter `hot_set` name. Both are kept so
    /// callers can use whichever they prefer.
    #[inline]
    pub fn get_hot_experts(&self, threshold_pct: f32) -> Vec<u32> {
        self.hot_set(threshold_pct)
    }

    /// Heat count for a specific id (0 if `id` is out of range).
    pub fn heat(&self, id: u32) -> u32 {
        if id >= self.num_experts {
            return 0;
        }
        self.inner.read().counts[id as usize]
    }

    /// Reset the monitor back to its empty state. Useful for tests and
    /// for re-warming the monitor between unrelated request streams.
    #[allow(dead_code)]
    pub fn clear(&self) {
        let mut inner = self.inner.write();
        inner.window.clear();
        for c in inner.counts.iter_mut() {
            *c = 0;
        }
        inner.total = 0;
    }
}

/// Scale `grad` in place so its L2 norm is at most `max_norm`. Skips
/// when the norm is already within bounds or when it is zero / not
/// finite (the per-element NaN guard above already replaced
/// non-finite entries with `0.0`, but we keep the defensive check so
/// future callers can't trip a divide-by-zero or `NaN / NaN`).
fn clip_gradient_norm(grad: &mut [f32], max_norm: f32) {
    let mut sumsq = 0.0f32;
    for &g in grad.iter() {
        sumsq += g * g;
    }
    if !sumsq.is_finite() || sumsq <= max_norm * max_norm {
        return;
    }
    let norm = sumsq.sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return;
    }
    let scale = max_norm / norm;
    for v in grad.iter_mut() {
        *v *= scale;
    }
}

/// Tiny **Neural Speculator** — the semantic-intent arm of the dual-path
/// predictive controller.
///
/// A two-layer MLP (`d_model -> hidden -> num_experts`) trained online
/// against the gating network's actual top-K decisions. The network is
/// kept intentionally small (default hidden = 128) so it fits in L1/L2
/// cache and so a single SGD step per token is cheap; the goal is not
/// to *replace* the gate but to provide a fast, stateless prediction of
/// "what experts will likely fire next given this hidden state" that
/// can be unioned with the Markov chain's hint to drive prefetch.
///
/// Architecture details:
/// * Layer 1: `W1` of shape `[hidden, d_model]`, bias `b1` of `hidden`,
///   ReLU activation.
/// * Layer 2: `W2` of shape `[num_experts, hidden]`, bias `b2` of
///   `num_experts`. The output is interpreted as a logit vector;
///   `predict_topk` returns the top-K ids by logit, and `train_step`
///   takes a softmax + cross-entropy step against a soft target spread
///   uniformly over the actual top-K ids selected by the real gate.
///
/// Initialisation uses He-uniform scaling (the standard ReLU init) with
/// a deterministic seed so identical runs produce identical predictions.
/// All numerical paths are `f32`. Stability is enforced by three
/// stacked safeguards: a **global L2-norm gradient cap** at
/// [`Self::MAX_GRAD_NORM`] on both `dlogits` and the back-propagated
/// `dh`, a per-element `±MAX_GRAD_NORM` clamp as a belt-and-braces
/// guard, and a `clamp_finite` check on every weight write. The
/// effective learning rate decays inverse-time against the
/// cumulative step counter (see [`Self::LR_DECAY_RATE`]) so a sudden
/// topic shift late in a long run can't dominate the model. The
/// predictor is a *prefetch hint*; correctness still flows through
/// the real gate downstream, so even a pathological speculator can
/// only degrade prefetch accuracy, never engine correctness.
pub struct NeuralSpeculator {
    d_model: usize,
    hidden: usize,
    num_experts: u32,
    /// Locked together so a `predict()` always sees a consistent `(W1,
    /// b1, W2, b2)` snapshot — `train_step` writes all four and we
    /// don't want a half-updated set of weights to be read mid-update.
    ///
    /// Read access is **prioritised** in the off-path training worker:
    /// the worker calls [`parking_lot::RwLock::try_write_for`] with a
    /// short timeout and drops the sample if it can't acquire the
    /// lock, so the hot-path `predict_topk` is never blocked by
    /// background SGD compute.
    weights: RwLock<SpeculatorWeights>,
    /// Cumulative count of SGD updates applied to `weights`. Drives
    /// the inverse-time learning-rate schedule (see [`Self::effective_lr`])
    /// so the speculator stops chasing fresh routing distributions
    /// aggressively after a long adaptation window — a sudden topic
    /// shift in a long document can otherwise spike the gradient and
    /// degrade prefetch accuracy until the new distribution stabilises.
    train_steps: AtomicU64,
    /// Asynchronous training queue. Set on first call to
    /// [`Self::spawn_training_worker`] (idempotent) and consumed by a
    /// dedicated background thread. The queue is bounded so a
    /// runaway producer can't pin unbounded memory: when full,
    /// [`Self::queue_train`] drops the newest sample (training is a
    /// best-effort signal — the *correct* routing still flows
    /// through the gate downstream).
    train_queue: OnceLock<std::sync::mpsc::SyncSender<TrainSample>>,
}

/// A single hidden-state / actual-routing pair queued for off-path
/// SGD. Fields are owned so the background thread doesn't borrow
/// anything from the hot path.
struct TrainSample {
    x: Vec<f32>,
    actual_top_k: Vec<u32>,
    lr: f32,
}

#[cfg(test)]
#[derive(Debug, PartialEq)]
pub(crate) struct QueuedTrainSampleForTest {
    pub x: Vec<f32>,
    pub actual_top_k: Vec<u32>,
    pub lr: f32,
}

struct SpeculatorWeights {
    /// `[hidden, d_model]` row-major.
    w1: Vec<f32>,
    /// `[hidden]`.
    b1: Vec<f32>,
    /// `[num_experts, hidden]` row-major.
    w2: Vec<f32>,
    /// `[num_experts]`.
    b2: Vec<f32>,
}

impl NeuralSpeculator {
    /// Default learning rate for the online SGD step. Small by design:
    /// the speculator is trained continuously over every token and a
    /// large lr would let occasional outlier routings dominate.
    pub const DEFAULT_LR: f32 = 1e-3;

    /// L2 weight-decay coefficient applied per SGD step. Small so it
    /// only nudges idle weights toward zero (preventing unbounded
    /// drift when the same expert id never re-appears) without
    /// fighting the gradient signal on actively-routed experts. The
    /// per-step update applies `w := w * (1 - lr * WEIGHT_DECAY)`
    /// before the gradient step (the standard "decoupled weight
    /// decay" / AdamW-style ordering) so the decay is independent of
    /// the gradient magnitude.
    pub const WEIGHT_DECAY: f32 = 1e-4;

    /// Global L2-norm cap on the per-step gradient (the reviewer's
    /// `if grad_norm > threshold { grad /= grad_norm }` suggestion).
    /// Applied to both `dlogits` and the back-propagated `dh` so a
    /// single noisy sample — typical during a topic shift in a long
    /// document — can't blow up the weight update beyond the same
    /// per-element clip the spec already enforces. Picked to match
    /// the existing per-element clamp magnitude, so well-behaved
    /// gradients are unaffected and only the tail is scaled down.
    pub const MAX_GRAD_NORM: f32 = 1.0;

    /// Inverse-time learning-rate decay: `lr_eff = lr / (1 + steps *
    /// LR_DECAY_RATE)`. With `1e-5` the effective LR is halved after
    /// roughly 100k SGD updates, ten-times-down after ~1M — slow
    /// enough that the speculator still tracks routing-distribution
    /// drift, but fast enough that an outlier sample late in a long
    /// run can't dominate. Set to zero to disable decay entirely.
    pub const LR_DECAY_RATE: f32 = 1e-5;

    /// Build a fresh speculator with He-uniform initialisation.
    pub fn new(d_model: usize, hidden: usize, num_experts: u32, seed: u64) -> Self {
        assert!(d_model > 0 && hidden > 0 && num_experts > 0);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0xC0DE_FEED));
        // He uniform: U(-sqrt(6/fan_in), +sqrt(6/fan_in)).
        let bound1 = (6.0f32 / d_model as f32).sqrt();
        let bound2 = (6.0f32 / hidden as f32).sqrt();
        use rand::Rng;
        let w1: Vec<f32> = (0..hidden * d_model)
            .map(|_| rng.gen_range(-bound1..bound1))
            .collect();
        let w2: Vec<f32> = (0..(num_experts as usize) * hidden)
            .map(|_| rng.gen_range(-bound2..bound2))
            .collect();
        let b1 = vec![0.0f32; hidden];
        let b2 = vec![0.0f32; num_experts as usize];
        Self {
            d_model,
            hidden,
            num_experts,
            weights: RwLock::new(SpeculatorWeights { w1, b1, w2, b2 }),
            train_steps: AtomicU64::new(0),
            train_queue: OnceLock::new(),
        }
    }

    /// Effective per-step learning rate: the caller-supplied `lr`
    /// scaled by an inverse-time decay against
    /// [`Self::train_steps`]. Made `pub(crate)` so tests can assert
    /// the schedule without going through `train_step`.
    pub(crate) fn effective_lr(&self, lr: f32) -> f32 {
        let steps = self.train_steps.load(Ordering::Relaxed) as f32;
        let denom = 1.0 + steps * Self::LR_DECAY_RATE;
        if denom > 0.0 && denom.is_finite() {
            lr / denom
        } else {
            lr
        }
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Forward pass returning the full logit vector of length `num_experts`.
    /// Used internally by `predict_topk` and `train_step` and also by
    /// tests that want to inspect the raw logits.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.d_model);
        let w = self.weights.read();
        let mut h = vec![0.0f32; self.hidden];
        for i in 0..self.hidden {
            let row = &w.w1[i * self.d_model..(i + 1) * self.d_model];
            let mut acc = w.b1[i];
            for j in 0..self.d_model {
                acc += row[j] * x[j];
            }
            // ReLU
            h[i] = if acc > 0.0 { acc } else { 0.0 };
        }
        let n = self.num_experts as usize;
        let mut logits = vec![0.0f32; n];
        for i in 0..n {
            let row = &w.w2[i * self.hidden..(i + 1) * self.hidden];
            let mut acc = w.b2[i];
            for j in 0..self.hidden {
                acc += row[j] * h[j];
            }
            logits[i] = acc;
        }
        logits
    }

    /// Predict the top-`k` expert ids by logit. Output is in descending
    /// logit order; ties broken by ascending id so the choice is fully
    /// deterministic. Empty input or `k==0` yields an empty vector.
    pub fn predict_topk(&self, x: &[f32], k: usize) -> Vec<u32> {
        if k == 0 || x.len() != self.d_model {
            return Vec::new();
        }
        let logits = self.forward(x);
        let mut idx: Vec<(u32, f32)> = logits
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as u32, v))
            .collect();
        idx.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.cmp(&b.0))
        });
        idx.truncate(k.min(self.num_experts as usize));
        idx.into_iter().map(|(i, _)| i).collect()
    }

    /// Predict the top-`k` expert ids together with their softmax
    /// probabilities. Useful for callers (telemetry, prefetch
    /// budgeting) that want a confidence signal alongside the ranking.
    pub fn predict_topk_with_probs(&self, x: &[f32], k: usize) -> Vec<(u32, f32)> {
        if k == 0 || x.len() != self.d_model {
            return Vec::new();
        }
        let mut logits = self.forward(x);
        // Numerically-stable softmax in place.
        let mut max = f32::NEG_INFINITY;
        for &v in &logits {
            if v > max {
                max = v;
            }
        }
        if !max.is_finite() {
            return Vec::new();
        }
        let mut sum = 0.0f32;
        for v in logits.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in logits.iter_mut() {
                *v /= sum;
            }
        }
        let mut idx: Vec<(u32, f32)> = logits
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as u32, v))
            .collect();
        idx.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.cmp(&b.0))
        });
        idx.truncate(k.min(self.num_experts as usize));
        idx
    }

    /// Predict the top-`k` expert ids **restricted to a single layer's
    /// slice of the global id space** (Part 1, layer-ahead prefetch).
    ///
    /// Expert ids in this engine are layer-qualified global ids:
    /// `global = layer * per_layer + local`. The speculator's output
    /// head spans every global id, so to predict the experts that will
    /// fire in a *specific* layer we score the full logit vector and
    /// then keep only the ids falling in that layer's window
    /// `[layer * per_layer, (layer + 1) * per_layer)`. This lets
    /// `moe_step` ask "what will layer L+1 want?" while it is still
    /// computing layer L, so the io_uring reads for L+1 are in flight a
    /// full layer-compute early.
    ///
    /// Returns global ids in descending-logit order (ties broken by
    /// ascending id, matching [`predict_topk`]). An out-of-range layer,
    /// `per_layer == 0`, `k == 0`, or a mismatched feature width yields
    /// an empty vector. Non-finite logits (only possible if the weights
    /// diverge) sort as equal, matching [`predict_topk`] /
    /// [`predict_topk_with_probs`]; a wrong but bounded ranking is
    /// acceptable for a prefetch hint.
    pub fn predict_topk_for_layer(
        &self,
        x: &[f32],
        layer: u32,
        per_layer: u32,
        k: usize,
    ) -> Vec<u32> {
        if k == 0 || per_layer == 0 || x.len() != self.d_model {
            return Vec::new();
        }
        let n = self.num_experts;
        let base = match layer.checked_mul(per_layer) {
            Some(b) if b < n => b,
            _ => return Vec::new(),
        };
        let end = base.saturating_add(per_layer).min(n);
        let logits = self.forward(x);
        let mut idx: Vec<(u32, f32)> = (base..end)
            .map(|i| (i, logits[i as usize]))
            .collect();
        idx.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.0.cmp(&b.0))
        });
        idx.truncate(k.min((end - base) as usize));
        idx.into_iter().map(|(i, _)| i).collect()
    }

    /// One step of online SGD against a soft-target distribution that
    /// places mass `1/|actual|` on each id in `actual_top_k` and 0
    /// elsewhere. The loss is cross-entropy on top of softmax; the
    /// gradient w.r.t. the logits is `softmax(logits) - target`.
    /// Gradient values are clipped to `±1.0` per element and weight
    /// updates are NaN/Inf-guarded.
    ///
    /// Returns the cross-entropy loss for diagnostics. Out-of-range
    /// ids in `actual_top_k` are silently ignored; if every id is
    /// out of range the weights are not updated and the returned loss
    /// is `f32::NAN`.
    pub fn train_step(&self, x: &[f32], actual_top_k: &[u32], lr: f32) -> f32 {
        if x.len() != self.d_model || actual_top_k.is_empty() {
            return f32::NAN;
        }
        let mut w = self.weights.write();
        self.train_step_locked(&mut w, x, actual_top_k, lr)
    }

    /// SGD body with the weight lock held by the caller. Separated
    /// out so the off-path training worker can attempt the write
    /// using [`parking_lot::RwLock::try_write_for`] and bail (rather
    /// than block readers) when the hot path is busy.
    fn train_step_locked(
        &self,
        w: &mut SpeculatorWeights,
        x: &[f32],
        actual_top_k: &[u32],
        lr: f32,
    ) -> f32 {
        // Inverse-time LR decay against the cumulative step counter.
        // Callers pass a *base* lr (typically `DEFAULT_LR`); the
        // effective rate falls off so the speculator stops chasing
        // outlier samples late in a long run. `lr == 0.0` (the
        // "measure loss only" idiom used in tests) still produces
        // `lr_eff == 0.0`.
        let lr_eff = self.effective_lr(lr);

        // Build the target distribution.
        let n = self.num_experts as usize;
        let mut target = vec![0.0f32; n];
        let mut hits = 0usize;
        for &id in actual_top_k {
            if (id as usize) < n {
                target[id as usize] += 1.0;
                hits += 1;
            }
        }
        if hits == 0 {
            return f32::NAN;
        }
        let inv = 1.0f32 / hits as f32;
        for v in target.iter_mut() {
            *v *= inv;
        }

        // Forward (re-do here so we can capture the hidden activation).
        let mut h_pre = vec![0.0f32; self.hidden];
        let mut h = vec![0.0f32; self.hidden];
        for i in 0..self.hidden {
            let row = &w.w1[i * self.d_model..(i + 1) * self.d_model];
            let mut acc = w.b1[i];
            for j in 0..self.d_model {
                acc += row[j] * x[j];
            }
            h_pre[i] = acc;
            h[i] = if acc > 0.0 { acc } else { 0.0 };
        }
        let mut logits = vec![0.0f32; n];
        for i in 0..n {
            let row = &w.w2[i * self.hidden..(i + 1) * self.hidden];
            let mut acc = w.b2[i];
            for j in 0..self.hidden {
                acc += row[j] * h[j];
            }
            logits[i] = acc;
        }
        // Softmax + cross-entropy loss.
        let mut max = f32::NEG_INFINITY;
        for &v in &logits {
            if v > max {
                max = v;
            }
        }
        if !max.is_finite() {
            return f32::NAN;
        }
        let mut probs = logits.clone();
        let mut sum = 0.0f32;
        for v in probs.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if !(sum > 0.0 && sum.is_finite()) {
            return f32::NAN;
        }
        let mut loss = 0.0f32;
        for (i, v) in probs.iter_mut().enumerate() {
            *v /= sum;
            if target[i] > 0.0 {
                let p = v.max(1e-12);
                loss -= target[i] * p.ln();
            }
        }

        // Gradient w.r.t. logits = probs - target.
        let mut dlogits = probs;
        for i in 0..n {
            dlogits[i] -= target[i];
            // Replace non-finite entries early so the norm below is
            // well-defined; the per-element clip below would otherwise
            // be poisoned by a single NaN.
            if !dlogits[i].is_finite() {
                dlogits[i] = 0.0;
            }
        }
        // Global gradient-norm cap: `if ||g|| > τ { g *= τ / ||g|| }`.
        // This is the reviewer's requested protection against a
        // topic shift spiking the per-step update — distinct from
        // (and applied *before*) the per-element clamp, which only
        // bounds individual coordinates.
        clip_gradient_norm(&mut dlogits, Self::MAX_GRAD_NORM);
        // Per-element clamp as a belt-and-braces guard. After the
        // norm cap above, well-behaved samples are unaffected; only
        // pathological coordinates (e.g. residue of an out-of-range
        // gradient that survived the norm scaling because the rest
        // of the vector was near-zero) are pinned to ±1.
        for v in dlogits.iter_mut() {
            if *v > 1.0 {
                *v = 1.0;
            } else if *v < -1.0 {
                *v = -1.0;
            }
        }

        // Backprop into hidden: dh = W2^T · dlogits, masked by ReLU
        // derivative (1 where pre-activation > 0, else 0).
        let mut dh = vec![0.0f32; self.hidden];
        for i in 0..n {
            let g = dlogits[i];
            if g == 0.0 {
                continue;
            }
            let row = &w.w2[i * self.hidden..(i + 1) * self.hidden];
            for j in 0..self.hidden {
                dh[j] += row[j] * g;
            }
        }
        for j in 0..self.hidden {
            if h_pre[j] <= 0.0 {
                dh[j] = 0.0;
            }
            if !dh[j].is_finite() {
                dh[j] = 0.0;
            }
        }
        // Global norm cap on the hidden-layer gradient too — the
        // multiplication by `W2` can blow up the magnitude even when
        // `dlogits` itself was bounded.
        clip_gradient_norm(&mut dh, Self::MAX_GRAD_NORM);
        for v in dh.iter_mut() {
            if *v > 1.0 {
                *v = 1.0;
            } else if *v < -1.0 {
                *v = -1.0;
            }
        }

        // Decoupled weight decay (AdamW-style): shrink every weight by
        // `(1 - lr * WEIGHT_DECAY)` *before* the gradient step. This
        // bounds the magnitude of weights for experts/features that
        // never appear in `actual_top_k` (whose gradient through this
        // step is zero), so the speculator can't drift toward
        // arbitrarily large logits over an infinite training run.
        // Biases are intentionally not decayed (standard practice;
        // they don't suffer from the same scale-blowup pathology).
        let decay = 1.0 - lr_eff * Self::WEIGHT_DECAY;
        if decay > 0.0 && decay < 1.0 {
            for v in w.w2.iter_mut() {
                let new = *v * decay;
                if new.is_finite() {
                    *v = new;
                }
            }
            for v in w.w1.iter_mut() {
                let new = *v * decay;
                if new.is_finite() {
                    *v = new;
                }
            }
        }

        // Update W2 / b2.
        for i in 0..n {
            let g = dlogits[i];
            if g == 0.0 {
                continue;
            }
            let row = &mut w.w2[i * self.hidden..(i + 1) * self.hidden];
            for j in 0..self.hidden {
                let upd = lr_eff * g * h[j];
                let new = row[j] - upd;
                row[j] = if new.is_finite() { new } else { row[j] };
            }
            let new_b = w.b2[i] - lr_eff * g;
            w.b2[i] = if new_b.is_finite() { new_b } else { w.b2[i] };
        }
        // Update W1 / b1.
        for i in 0..self.hidden {
            let g = dh[i];
            if g == 0.0 {
                continue;
            }
            let row = &mut w.w1[i * self.d_model..(i + 1) * self.d_model];
            for j in 0..self.d_model {
                let upd = lr_eff * g * x[j];
                let new = row[j] - upd;
                row[j] = if new.is_finite() { new } else { row[j] };
            }
            let new_b = w.b1[i] - lr_eff * g;
            w.b1[i] = if new_b.is_finite() { new_b } else { w.b1[i] };
        }

        // Count this step *after* a successful update so the LR
        // schedule advances monotonically. `train_step_locked` is the
        // single point of entry for weight writes (both inline
        // `train_step` and the off-path worker funnel through here),
        // so the counter cannot double-bump.
        self.train_steps.fetch_add(1, Ordering::Relaxed);

        loss
    }

    /// Idempotently spawn the off-path training worker. Subsequent
    /// calls are no-ops. After this returns, [`Self::queue_train`]
    /// will drop samples onto the worker's bounded channel; the
    /// worker pulls them in FIFO order and applies SGD updates,
    /// preferring readers via [`parking_lot::RwLock::try_write_for`].
    ///
    /// The worker is a plain OS thread (not a tokio task) so it
    /// keeps making progress even if the tokio runtime is
    /// momentarily saturated; it terminates when the [`Arc`] count
    /// of the speculator drops to one (its own ref) — i.e. when no
    /// engine still holds it.
    pub fn spawn_training_worker(self: &std::sync::Arc<Self>) {
        // OnceLock::get_or_init guarantees a single worker even
        // under concurrent installs.
        self.train_queue.get_or_init(|| {
            // Bounded queue so a runaway producer can't pin memory.
            // A few hundred slots is plenty: at typical decoder
            // throughput the worker drains the queue between
            // tokens; the bound only matters during pathological
            // bursts (and in that case we'd rather drop old
            // samples than stall the producer).
            let (tx, rx) = std::sync::mpsc::sync_channel::<TrainSample>(256);
            let weak = std::sync::Arc::downgrade(self);
            std::thread::Builder::new()
                .name("mer-speculator-train".to_string())
                .spawn(move || speculator_training_loop(weak, rx))
                .ok(); // Failing to spawn is non-fatal; queued samples will drop.
            tx
        });
    }

    /// Push one `(hidden_state, actual_top_k)` sample onto the
    /// off-path training queue. Drops the sample silently when the
    /// queue is full or when [`Self::spawn_training_worker`] was
    /// never called — training is a prefetch-hint signal, not a
    /// correctness-critical path.
    pub fn queue_train(&self, x: &[f32], actual_top_k: &[u32], lr: f32) {
        if x.len() != self.d_model || actual_top_k.is_empty() {
            return;
        }
        if let Some(tx) = self.train_queue.get() {
            let sample = TrainSample {
                x: x.to_vec(),
                actual_top_k: actual_top_k.to_vec(),
                lr,
            };
            // `try_send` is non-blocking: when the worker is
            // behind, the newest sample is dropped (a deliberate
            // back-pressure choice — old samples in the queue still
            // contain useful gradient signal).
            let _ = tx.try_send(sample);
        }
    }

    #[cfg(test)]
    pub(crate) fn train_steps_for_test(&self) -> u64 {
        self.train_steps.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn capture_queued_train_for_test<R>(
        spec: &std::sync::Arc<Self>,
        f: impl FnOnce() -> R,
    ) -> (R, QueuedTrainSampleForTest) {
        let (tx, rx) = std::sync::mpsc::sync_channel::<TrainSample>(1);
        if spec.train_queue.set(tx).is_err() {
            panic!("test training queue must be installed before spawning the worker");
        }

        let result = f();
        let sample = rx
            .try_recv()
            .expect("current sample should be submitted to the training queue");
        (
            result,
            QueuedTrainSampleForTest {
                x: sample.x,
                actual_top_k: sample.actual_top_k,
                lr: sample.lr,
            },
        )
    }
}

/// Background training loop. Pulls samples from the queue and
/// applies SGD updates, preferring readers by using
/// [`parking_lot::RwLock::try_write_for`] — if a hot-path
/// `predict_topk` is currently reading, the worker waits at most a
/// few hundred microseconds and then drops the sample rather than
/// starve the inference critical path.
fn speculator_training_loop(
    weak: std::sync::Weak<NeuralSpeculator>,
    rx: std::sync::mpsc::Receiver<TrainSample>,
) {
    use std::time::Duration;
    // Cap the per-sample wait. The hot path reads in the low-µs
    // range, so 500 µs is generous: if we can't get the write
    // lock in that window, readers are bursty and we'd rather
    // skip this update.
    const TRY_WRITE_BUDGET: Duration = Duration::from_micros(500);

    while let Ok(sample) = rx.recv() {
        let Some(spec) = weak.upgrade() else {
            return; // engine dropped the speculator; quit cleanly.
        };
        if let Some(mut w) = spec.weights.try_write_for(TRY_WRITE_BUDGET) {
            let _loss = spec.train_step_locked(&mut w, &sample.x, &sample.actual_top_k, sample.lr);
            // Loss is intentionally discarded — telemetry covers
            // accuracy via the engine's predict-and-train path.
        } else {
            tracing::trace!(
                "speculator training worker: skipped one sample (reader contention)"
            );
        }
        drop(spec);
    }
}

// =====================================================================
// Advanced predictive prefetching primitives (gist Part 2).
//
// * `ExpertAffinity` — co-occurrence heat-map tracking which expert
//   ids fire together inside the same MoE layer (orthogonal to the
//   *sequential* Markov chain modelled by `PredictiveLoader`).
// * `spatial_neighbors` — UTH-layout neighbour lookup used by spatial
//   prefetching when the predictor is highly confident in a hit.
// * `SpeculationController` — latency-aware speculation window
//   controller that bumps the prefetch depth when `ssd_stall_us`
//   telemetry rises and clamps it under critical memory pressure.
// =====================================================================

/// Confidence threshold (a score in `[0, 1]`) above which the
/// predictor should also issue a spatial prefetch for an expert's
/// immediate UTH neighbours. 0.80 matches the gist's example.
pub const SPATIAL_CONFIDENCE_THRESHOLD: f32 = 0.80;

// Unified-ranking arm weights. Hoisted to module scope (out of the
// individual `predict_unified*` method bodies) so the engine's hot-path
// `union_prefetch` consumes the *same* constants as the offline
// `predict_unified` API instead of re-deriving its own ad-hoc scheme.
// The three headline weights sum to 1.0 so a unanimously-predicted
// expert tops out at exactly 1.0; relative ordering encodes the design
// intent "speculator > Markov > locality".
pub const W_SPECULATOR: f32 = 0.42;
pub const W_MARKOV: f32 = 0.33;
pub const W_LOCALITY: f32 = 0.25;
// Auxiliary neighbour-arm weights for `predict_unified_with_spatial` /
// the engine's affinity+spatial fold. Small relative to the headline
// arms: they should pull a co-fired / disk-adjacent neighbour into the
// prefetch set when a seed is already confident, never drive the
// fanout on their own.
pub const W_AFFINITY: f32 = 0.10;
pub const W_SPATIAL: f32 = 0.05;

/// Per-rank decay applied to the flat locality / speculator
/// contributions in [`PredictiveLoader::combine_unified_arms`]. Both
/// arms hand the combiner a *ranked* list (locality is heat-sorted,
/// the speculator logit-sorted), but a flat weight discards that
/// ordering: every id ties at exactly `W_LOCALITY` / `W_SPECULATOR`
/// and the deterministic ascending-id tie-break then systematically
/// favours low (low-layer) ids when the candidate list is truncated
/// to the shadow budget. Scaling the contribution by
/// `1 − RANK_TIEBREAK_DECAY × rank` preserves each arm's internal
/// ranking while staying small enough (1e-4) that cross-arm ordering
/// is unaffected for any realistic list length.
pub const RANK_TIEBREAK_DECAY: f32 = 1e-4;

/// Default cap on the speculation window after a latency bump. The
/// controller never grows the window beyond `base_depth +
/// MAX_LATENCY_BUMP` tokens, even under sustained SSD stall.
pub const MAX_LATENCY_BUMP: usize = 2;

/// Lock-free co-occurrence matrix tracking which expert ids fire
/// together inside the same MoE layer. The hot path (`observe_layer`)
/// only performs atomic updates on pre-allocated `AtomicU32` cells;
/// saturating increments may use a compare-exchange retry loop, but
/// the path never grabs a lock and never allocates.
///
/// The matrix is **symmetric**: `affinity(i, j) == affinity(j, i)`,
/// because co-occurrence is undirected (two experts active in the
/// same layer-step are equally "neighbours" of each other regardless
/// of which one the gate scored higher). We store the full N×N table
/// (rather than a triangular half) so `neighbors(id, k)` can walk a
/// contiguous row, which is cache-friendly and avoids per-call index
/// arithmetic.
///
/// Sizing: at N=64 experts the matrix is 64×64×4 B = 16 KiB; at
/// N=4096 it is 64 MiB. The pre-allocation cost is identical to a
/// `Vec<AtomicU32>` of the same size; with `bitvec` we could halve
/// the per-pair byte count by capping at 65k observations and packing
/// counters at 16 bits, but real Mixtral traces don't approach that
/// ceiling within a single conversation and the simpler u32 layout
/// stays clearly correct under any saturating-add semantics.
pub struct ExpertAffinity {
    num_experts: u32,
    /// Flat N×N matrix of u32 counters, indexed as `[i * n + j]`.
    /// Allocated once at construction; never resized. Diagonal cells
    /// (`[i*n + i]`) are kept at zero — self-affinity is meaningless.
    counts: Box<[AtomicU32]>,
    /// Cumulative number of `observe_layer` calls. Used by
    /// [`Self::total_observations`] for diagnostics and to size the
    /// expected per-pair denominator.
    observations: AtomicU64,
}

impl ExpertAffinity {
    /// Pre-allocate the N×N counter matrix. `num_experts` must be
    /// non-zero. Construction touches every cell once, which is the
    /// only allocation this type ever performs.
    pub fn new(num_experts: u32) -> Self {
        assert!(num_experts > 0, "ExpertAffinity num_experts must be > 0");
        let n = num_experts as usize;
        // `Vec::from_iter` + `into_boxed_slice` so the buffer lives in
        // a single contiguous allocation that never grows.
        let counts: Vec<AtomicU32> = (0..n * n).map(|_| AtomicU32::new(0)).collect();
        Self {
            num_experts,
            counts: counts.into_boxed_slice(),
            observations: AtomicU64::new(0),
        }
    }

    /// Total `observe_layer` calls recorded since construction.
    pub fn total_observations(&self) -> u64 {
        self.observations.load(Ordering::Relaxed)
    }

    /// Record that every pair `(i, j)` in `experts` was activated
    /// together by the same MoE layer for the same token. Hot-path
    /// safe: each pair becomes two saturating `fetch_add(1)` calls
    /// (one per ordering) — no allocations, no locks.
    pub fn observe_layer(&self, experts: &[u32]) {
        if experts.len() < 2 {
            return;
        }
        let n = self.num_experts as usize;
        for (idx, &a) in experts.iter().enumerate() {
            if (a as usize) >= n {
                continue;
            }
            for &b in &experts[idx + 1..] {
                if (b as usize) >= n || a == b {
                    continue;
                }
                let ab = (a as usize) * n + b as usize;
                let ba = (b as usize) * n + a as usize;
                // Saturating add — pin at u32::MAX so a long-running
                // session can't wrap the counter back to zero.
                Self::sat_add(&self.counts[ab]);
                Self::sat_add(&self.counts[ba]);
            }
        }
        self.observations.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    fn sat_add(cell: &AtomicU32) {
        // CAS loop saturating at u32::MAX. Almost always succeeds on
        // first try; the loop only runs under contention.
        let mut cur = cell.load(Ordering::Relaxed);
        loop {
            if cur == u32::MAX {
                return;
            }
            match cell.compare_exchange_weak(cur, cur + 1, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return,
                Err(actual) => cur = actual,
            }
        }
    }

    /// Co-occurrence count of `(a, b)`. Returns `0` for out-of-range
    /// ids or for `a == b` (diagonal is always zero).
    pub fn affinity(&self, a: u32, b: u32) -> u32 {
        if a == b {
            return 0;
        }
        let n = self.num_experts as usize;
        if (a as usize) >= n || (b as usize) >= n {
            return 0;
        }
        self.counts[(a as usize) * n + b as usize].load(Ordering::Relaxed)
    }

    /// Top-`k` experts most frequently co-fired with `id`, in
    /// descending affinity order (ties broken by ascending id for
    /// determinism). Zero-count neighbours are filtered out. Allocates
    /// a single `Vec<(u32, u32)>` for the sort; the matrix itself is
    /// untouched.
    pub fn neighbors(&self, id: u32, k: usize) -> Vec<u32> {
        let n = self.num_experts as usize;
        if (id as usize) >= n || k == 0 {
            return Vec::new();
        }
        let row_start = (id as usize) * n;
        let mut scored: Vec<(u32, u32)> = (0..n)
            .filter_map(|j| {
                if j == id as usize {
                    return None;
                }
                let c = self.counts[row_start + j].load(Ordering::Relaxed);
                if c == 0 {
                    None
                } else {
                    Some((j as u32, c))
                }
            })
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(k);
        scored.into_iter().map(|(id, _)| id).collect()
    }

    /// Reset the matrix to its empty state. Used by tests and by the
    /// scheduler when a session ends and the prior co-occurrence
    /// distribution is no longer representative.
    #[allow(dead_code)]
    pub fn clear(&self) {
        for cell in self.counts.iter() {
            cell.store(0, Ordering::Relaxed);
        }
        self.observations.store(0, Ordering::Relaxed);
    }

    /// **Exponential bit-shift decay (gist Part 2, fix #7).** Right-
    /// shift every counter by `bits` so older co-occurrences age out
    /// of the matrix instead of accumulating indefinitely until they
    /// saturate at `u32::MAX`. Cheap: one atomic load + atomic store
    /// per cell, no allocation, no lock. Called periodically by the
    /// background decay worker spawned by
    /// [`LayeredExpertAffinity::spawn_decay_worker`]; `bits = 1` halves
    /// every counter, which keeps the heat map responsive to
    /// distribution shifts without losing all signal in one epoch.
    pub fn decay(&self, bits: u32) {
        if bits == 0 {
            return;
        }
        for cell in self.counts.iter() {
            // Relaxed is correct: the decay is *advisory* — concurrent
            // `observe_layer` increments may race with the shift, but
            // we only ever lose at most `bits` of magnitude on a
            // racing cell, which is precisely the intent.
            let cur = cell.load(Ordering::Relaxed);
            cell.store(cur >> bits, Ordering::Relaxed);
        }
        // Observation counter is *not* decayed — it's used for
        // "total samples seen" diagnostics and decaying it would
        // make the "Per ~100k observations" trigger of the worker
        // race against itself.
    }
}

/// **Per-layer expert-affinity matrix (gist Part 1, fix #2).** Owns
/// one [`ExpertAffinity`] per MoE layer so co-occurrences observed in
/// layer $L_0$ never get folded into the "experts that fire together"
/// signal for layer $L_5$. This matches the way real Mixtral-class
/// gates behave: each layer learns its own routing pattern, and
/// fusing them produces ghost neighbours that are not actually
/// co-fired in any single layer.
///
/// Hot-path API ([`Self::observe_layer`]) takes the originating layer
/// index, so the existing single-namespace flat global-id scheme used
/// by the cache and the storage layer still passes through unchanged
/// — only the affinity accounting becomes layer-aware.
pub struct LayeredExpertAffinity {
    layers: Box<[ExpertAffinity]>,
    num_experts: u32,
}

impl LayeredExpertAffinity {
    /// Pre-allocate one `N × N` matrix per layer. `num_layers` and
    /// `num_experts` must both be non-zero. Memory is laid out
    /// layer-major (one `Box<[AtomicU32]>` per layer), so a hot path
    /// that only touches a single layer never thrashes adjacent
    /// layers' cache lines.
    pub fn new(num_layers: usize, num_experts: u32) -> Self {
        assert!(num_layers > 0, "LayeredExpertAffinity num_layers must be > 0");
        assert!(num_experts > 0, "LayeredExpertAffinity num_experts must be > 0");
        let layers: Vec<ExpertAffinity> =
            (0..num_layers).map(|_| ExpertAffinity::new(num_experts)).collect();
        Self {
            layers: layers.into_boxed_slice(),
            num_experts,
        }
    }

    /// Record that every pair `(i, j)` in `experts` was activated
    /// together by the same MoE step inside `layer_idx`. Hot-path
    /// safe — delegates to the single-layer
    /// [`ExpertAffinity::observe_layer`] for `layer_idx`. Out-of-range
    /// layer indices are silently dropped (matches the rest of the
    /// router's "best-effort instrumentation" stance — telemetry
    /// must never panic the inference path).
    pub fn observe_layer(&self, layer_idx: usize, experts: &[u32]) {
        if let Some(layer) = self.layers.get(layer_idx) {
            layer.observe_layer(experts);
        }
    }

    /// Top-`k` experts most frequently co-fired with `id` **inside
    /// `layer_idx`**, in descending affinity order.
    pub fn neighbors(&self, layer_idx: usize, id: u32, k: usize) -> Vec<u32> {
        self.layers
            .get(layer_idx)
            .map(|l| l.neighbors(id, k))
            .unwrap_or_default()
    }

    /// Co-occurrence count of `(a, b)` within `layer_idx`. Returns 0
    /// for out-of-range layer / expert ids or `a == b`.
    pub fn affinity(&self, layer_idx: usize, a: u32, b: u32) -> u32 {
        self.layers
            .get(layer_idx)
            .map(|l| l.affinity(a, b))
            .unwrap_or(0)
    }

    /// Right-shift every counter in **every** layer by `bits` (gist
    /// Part 2, fix #7). One pass over all `num_layers × N × N` cells;
    /// still O(1) per cell so the total decay cost is linear in the
    /// matrix size — fine on the background epoch worker.
    pub fn decay(&self, bits: u32) {
        for layer in self.layers.iter() {
            layer.decay(bits);
        }
    }

    /// Cumulative `observe_layer` calls summed across every layer.
    /// Drives the decay worker's "shift after ~`epoch_threshold`
    /// observations" trigger.
    pub fn total_observations(&self) -> u64 {
        self.layers.iter().map(|l| l.total_observations()).sum()
    }

    /// Spawn a background worker that calls [`Self::decay`] once
    /// every `epoch_threshold` cumulative observations across all
    /// layers. The worker is **opt-in** (the engine starts it only
    /// when the predictive arm is configured) and parks itself
    /// efficiently between epochs — the poll interval below is the
    /// upper bound on how long a saturating counter could remain at
    /// `u32::MAX` before being shifted down.
    ///
    /// `bits = 1` halves every counter per epoch, which gives the
    /// matrix an effective sliding window of roughly `epoch_threshold
    /// × log2(u32::MAX)` observations before residual signal decays
    /// below noise — well beyond the conversation lengths a single
    /// session realistically generates.
    ///
    /// Returns a [`DecayWorkerHandle`] (gist Part 5, fix #11) — an
    /// owned supervisor that the engine retains for the lifetime of
    /// the affinity tracker. Dropping the handle (e.g. on engine
    /// shutdown) signals the worker to exit at its next poll
    /// boundary; [`DecayWorkerHandle::shutdown`] / [`DecayWorkerHandle::abort`]
    /// are exposed for callers that want to retire the worker
    /// explicitly before drop. The handle owns the shutdown flag so
    /// the worker's lifetime is tied to it by construction — no
    /// `#[must_use]` lint is required, because dropping the handle
    /// *does* something meaningful (clean shutdown).
    pub fn spawn_decay_worker(
        self: std::sync::Arc<Self>,
        epoch_threshold: u64,
        bits: u32,
        poll_interval: std::time::Duration,
    ) -> DecayWorkerHandle {
        let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let shutdown_clone = shutdown.clone();
        let aff = self;
        // Capture the baseline *before* spawning so the worker
        // observes a deterministic starting point — otherwise a race
        // between thread startup and concurrent `observe_layer` calls
        // would let the first epoch's delta drop to zero.
        let baseline = aff.total_observations();
        let join = std::thread::Builder::new()
            .name("affinity-decay".to_string())
            .spawn(move || {
                let mut last_seen = baseline;
                while shutdown_clone.load(Ordering::Relaxed) {
                    std::thread::sleep(poll_interval);
                    let now = aff.total_observations();
                    if now.saturating_sub(last_seen) >= epoch_threshold {
                        aff.decay(bits);
                        last_seen = now;
                    }
                }
            })
            .expect("affinity-decay worker thread failed to spawn");
        DecayWorkerHandle {
            shutdown,
            join: Some(join),
        }
    }
}

/// Owned supervisor returned by [`LayeredExpertAffinity::spawn_decay_worker`]
/// (gist Part 5, fix #11).
///
/// The handle wraps the shutdown flag *and* the background thread's
/// `JoinHandle`. Dropping the handle:
///
/// 1. clears the flag so the worker exits at the next poll boundary;
/// 2. joins the worker so the engine has a deterministic teardown
///    order (no orphaned threads running after the affinity tracker
///    has been dropped).
///
/// Because dropping is a meaningful operation, the type does **not**
/// carry a `#[must_use]` attribute — letting the value bind to `_` or
/// fall out of scope at the end of an engine's lifetime is exactly
/// the intended shutdown path. Callers that want to retire the worker
/// earlier can call [`Self::shutdown`] cooperatively.
pub struct DecayWorkerHandle {
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// `None` after the handle has been joined explicitly via
    /// [`Self::shutdown`]; the `Drop` impl then has nothing to do.
    join: Option<std::thread::JoinHandle<()>>,
}

impl DecayWorkerHandle {
    /// Cooperative shutdown: clear the flag and join the worker.
    /// Subsequent calls / `Drop` are no-ops.
    pub fn shutdown(mut self) {
        self.shutdown.store(false, std::sync::atomic::Ordering::Relaxed);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }

}

impl Drop for DecayWorkerHandle {
    fn drop(&mut self) {
        self.shutdown.store(false, std::sync::atomic::Ordering::Relaxed);
        if let Some(j) = self.join.take() {
            // Best-effort join — a panicked worker shouldn't crash
            // the engine on shutdown.
            let _ = j.join();
        }
    }
}

/// UTH-layout spatial neighbours of `id` — the experts whose tensor
/// records sit immediately adjacent to `id` on disk. Returns up to
/// `k` ids, prioritising `id-1` then `id+1` then expanding outwards.
/// Clipped to `[0, num_experts)`. Used by spatial prefetching to
/// piggy-back on the NVMe drive's sequential-read efficiency: pulling
/// expert N also brings N±1 along for almost free.
pub fn spatial_neighbors(id: u32, num_experts: u32, k: usize) -> Vec<u32> {
    if k == 0 || num_experts == 0 || id >= num_experts {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(k);
    let mut left = (id as i64) - 1;
    let mut right = (id as i64) + 1;
    while out.len() < k && (left >= 0 || right < num_experts as i64) {
        if left >= 0 {
            out.push(left as u32);
            left -= 1;
            if out.len() >= k {
                break;
            }
        }
        if right < num_experts as i64 {
            out.push(right as u32);
            right += 1;
        }
    }
    out
}

/// Latency-aware speculation window controller.
///
/// Reads cumulative `ssd_stall_us` telemetry on each token and adjusts
/// the speculation depth (how many tokens ahead the predictor
/// pre-fetches experts for) to hide rising disk wait times. The
/// controller is **lock-free** — all state is in atomics so the
/// inference thread can call [`Self::update_from_stall`] and
/// [`Self::current_depth`] on the hot path without contention.
///
/// Policy:
/// * If `Δssd_stall_us` since the previous call exceeds
///   [`Self::STALL_RISING_THRESHOLD_US`], grow the window by one
///   token (clipped to `base_depth + MAX_LATENCY_BUMP`).
/// * If `Δssd_stall_us` is below half that threshold for two
///   consecutive updates, shrink the window by one (clipped to
///   `base_depth`).
/// * [`Self::suspend`] forces depth to zero (called by the scheduler
///   under [`crate::block_pool::PressureLevel::Critical`]);
///   [`Self::resume`] restores the most recent computed depth.
pub struct SpeculationController {
    base_depth: usize,
    /// Currently active window, in tokens. `0` while suspended.
    current_depth: AtomicUsize,
    /// Snapshot of `current_depth` taken at the moment of suspension
    /// so [`Self::resume`] can restore the same value. Sentinel
    /// `usize::MAX` means "not currently suspended".
    saved_depth: AtomicUsize,
    /// Last cumulative `ssd_stall_us` observed via
    /// [`Self::update_from_stall`].
    last_stall_us: AtomicU64,
    /// Consecutive update calls with `Δstall` below the calm
    /// threshold. Counted so a single quiet sample doesn't
    /// immediately back off after a sustained stall burst.
    calm_streak: AtomicU64,
}

impl SpeculationController {
    /// Minimum Δstall (microseconds) between two consecutive
    /// `update_from_stall` calls that counts as "I/O latency is
    /// rising". 1 ms is roughly one NVMe round-trip on a healthy
    /// drive — anything noticeably larger is interpreted as the SSD
    /// queue building up.
    pub const STALL_RISING_THRESHOLD_US: u64 = 1_000;

    /// Build a controller with the given baseline depth. The active
    /// window starts at `base_depth` and ranges over `[base_depth,
    /// base_depth + MAX_LATENCY_BUMP]` during normal operation; under
    /// suspension it temporarily reads `0`.
    pub fn new(base_depth: usize) -> Self {
        Self {
            base_depth,
            current_depth: AtomicUsize::new(base_depth),
            saved_depth: AtomicUsize::new(usize::MAX),
            last_stall_us: AtomicU64::new(0),
            calm_streak: AtomicU64::new(0),
        }
    }

    /// Currently active speculation depth, in tokens. Returns `0`
    /// while [`Self::suspend`] is in effect.
    pub fn current_depth(&self) -> usize {
        self.current_depth.load(Ordering::Relaxed)
    }

    /// Feed the engine's cumulative `ssd_stall_us` telemetry into the
    /// controller. Computes the delta since the previous call and
    /// adjusts the speculation window. Returns the new
    /// [`Self::current_depth`].
    ///
    /// Safe to call on the per-token hot path: in the common case
    /// this is a single `AtomicU64::swap` (to atomically pivot the
    /// `last_stall_us` baseline) plus a small handful of relaxed
    /// loads/stores on the depth / streak atomics — no
    /// `compare_exchange` retry loop, no lock, no allocation.
    pub fn update_from_stall(&self, cumulative_stall_us: u64) -> usize {
        // Don't fight a manual suspension from the scheduler. Keep
        // the cumulative-stall baseline fresh behind the scenes so
        // resume() does not immediately react to stall that accrued
        // while speculation was intentionally disabled, but
        // current_depth stays at zero until then.
        if self.saved_depth.load(Ordering::Relaxed) != usize::MAX {
            self.last_stall_us.store(cumulative_stall_us, Ordering::Relaxed);
            return self.current_depth.load(Ordering::Relaxed);
        }
        let prev = self.last_stall_us.swap(cumulative_stall_us, Ordering::Relaxed);
        let delta = cumulative_stall_us.saturating_sub(prev);
        let mut depth = self.current_depth.load(Ordering::Relaxed);
        let max_depth = self.base_depth + MAX_LATENCY_BUMP;
        if delta >= Self::STALL_RISING_THRESHOLD_US {
            // Stall rising → widen the window.
            self.calm_streak.store(0, Ordering::Relaxed);
            if depth < max_depth {
                depth += 1;
            }
        } else if delta * 2 <= Self::STALL_RISING_THRESHOLD_US {
            let streak = self.calm_streak.fetch_add(1, Ordering::Relaxed) + 1;
            if streak >= 2 && depth > self.base_depth {
                depth -= 1;
                self.calm_streak.store(0, Ordering::Relaxed);
            }
        }
        self.current_depth.store(depth, Ordering::Relaxed);
        depth
    }

    /// Force the depth to zero. Called by the scheduler when
    /// [`crate::block_pool::PressureLevel::Critical`] is reached.
    /// Idempotent — calling twice is a no-op.
    pub fn suspend(&self) {
        let cur = self.current_depth.load(Ordering::Relaxed);
        // Only stash the pre-suspend value the *first* time so a
        // double-suspend doesn't overwrite the saved depth with 0.
        let _ = self.saved_depth.compare_exchange(
            usize::MAX,
            cur,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        self.current_depth.store(0, Ordering::Relaxed);
    }

    /// Restore the depth that was active before the most recent
    /// [`Self::suspend`]. No-op when not suspended.
    pub fn resume(&self) {
        let saved = self.saved_depth.swap(usize::MAX, Ordering::Relaxed);
        if saved != usize::MAX {
            self.current_depth.store(saved, Ordering::Relaxed);
        }
    }

    /// `true` while [`Self::suspend`] is in effect.
    pub fn is_suspended(&self) -> bool {
        self.saved_depth.load(Ordering::Relaxed) != usize::MAX
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_returns_k_distinct_experts() {
        let r = TopKRouter::new(64, 2, 42);
        for t in 0..1000 {
            let ids = r.route(t);
            assert_eq!(ids.len(), 2);
            assert_ne!(ids[0], ids[1]);
            for id in &ids {
                assert!(*id < 64);
            }
        }
    }

    #[test]
    fn clustered_router_prefers_in_cluster_transitions() {
        // 8 experts, 4 clusters → cluster membership is `id % 4`, so
        // experts {0,4} share cluster 0. With a 0.95 intra-cluster
        // probability, the chain has stationary distribution uniform
        // across clusters (~25% each) but the per-*step* property we
        // care about is locality: consecutive routes should land in
        // the same cluster ~95% of the time. That is the actual signal
        // a prefetcher learns from.
        let r = TopKRouter::clustered(8, 1, 4, 0.95, 7);
        r.fixed(&[0]);
        let trials = 4000;
        let mut last_cluster = 0u32;
        let mut same_cluster = 0;
        for t in 0..trials {
            let ids = r.route(t as u64);
            let c = ids[0] % 4;
            if t > 0 && c == last_cluster {
                same_cluster += 1;
            }
            last_cluster = c;
        }
        let frac = same_cluster as f64 / (trials - 1) as f64;
        assert!(
            frac > 0.85,
            "expected >85% step-to-step in-cluster transitions, got {:.3} ({}/{})",
            frac,
            same_cluster,
            trials - 1
        );
    }

    #[test]
    fn router_is_deterministic_given_seed() {
        // Two routers built with the same seed must emit identical
        // sequences. This is the property that makes runs reproducible.
        let a = TopKRouter::new(16, 2, 0xC0FFEE);
        let b = TopKRouter::new(16, 2, 0xC0FFEE);
        for t in 0..200 {
            assert_eq!(a.route(t), b.route(t), "diverged at token {t}");
        }
    }

    #[test]
    fn matrix_file_loads_and_normalises() {
        // 4 experts, "shift-by-one" matrix (each row picks (r+1) mod 4):
        // emit a temp file, load it, verify the chain cycles deterministically.
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "micro-expert-router-matrix-{}-{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let body = "\
            0 1 0 0\n\
            0 0 1 0\n\
            0 0 0 1\n\
            1 0 0 0\n";
        std::fs::write(&path, body).unwrap();
        let r = TopKRouter::from_matrix_file(&path, 4, 1, 1).unwrap();
        let _ = std::fs::remove_file(&path);
        r.fixed(&[0]);
        let expected = [1u32, 2, 3, 0, 1, 2, 3, 0];
        for (t, want) in expected.iter().enumerate() {
            let got = r.route(t as u64);
            assert_eq!(got, vec![*want], "step {t}: want {want:?}, got {got:?}");
        }
    }

    #[test]
    fn predictor_learns_simple_transition() {
        let p = PredictiveLoader::new(8, 2, 0.0, 1);
        for _ in 0..100 {
            p.observe(0, 3);
        }
        let preds = p.predict_next(0);
        assert!(!preds.is_empty());
        // The most likely successor of 0 must be 3.
        assert_eq!(preds[0].0, 3);
        assert!(preds[0].1 > 0.5);
    }

    #[test]
    fn predictor_respects_min_prob_threshold() {
        // The `min_prob` filter applies to *observed* successors: a row
        // with one observation that doesn't clear the threshold must
        // not surface that observed id. The unseen-successor prior-fill
        // is intentionally not subject to this filter (the fanout cap
        // is the only bound there), so we verify the observed id is
        // absent from the result rather than asserting an empty set.
        let p = PredictiveLoader::new(64, 4, 0.5, 1);
        p.observe(0, 7);
        let preds = p.predict_next(0);
        assert!(
            preds.iter().all(|&(id, _)| id != 7),
            "observed successor below min_prob must be filtered out: got {preds:?}"
        );
    }

    #[test]
    fn predictor_falls_back_to_prior_when_threshold_is_low() {
        let p = PredictiveLoader::new(8, 3, 0.05, 1);
        let preds = p.predict_next(0);
        assert_eq!(preds.len(), 3);
        for (_, prob) in &preds {
            assert!((prob - 1.0 / 8.0).abs() < 1e-9);
        }
    }

    #[test]
    fn predictor_observations_counts_only_real_transitions() {
        let p = PredictiveLoader::new(16, 2, 0.0, 1);
        assert_eq!(p.observations(), 0);
        for _ in 0..5 {
            p.observe(2, 7);
        }
        assert_eq!(p.observations(), 5);
    }

    #[test]
    fn predictor_handles_zero_fanout() {
        let p = PredictiveLoader::new(16, 0, 0.0, 1);
        p.observe(0, 1);
        assert!(p.predict_next(0).is_empty());
        assert!(p.sample_next(0).is_empty());
    }

    #[test]
    fn second_order_predictor_outranks_first_order_when_signal_disagrees() {
        // 1st-order: from `prev=2` we mostly see `5`.
        // 2nd-order: from `(prev_prev=7, prev=2)` we mostly see `9`.
        // `predict_next2(7, 2)` should rank `9` higher than `predict_next(2)` does.
        let p = PredictiveLoader::new(16, 4, 0.0, 1);
        for _ in 0..50 {
            p.observe(2, 5);
        }
        for _ in 0..50 {
            p.observe2(7, 2, 9);
            // observe2 also implies observe(2, 9) is *not* called; the
            // engine wires both via observe_step2. Test only the rows2
            // contribution here.
        }
        let preds1 = p.predict_next(2);
        let preds2 = p.predict_next2(7, 2);
        let rank1 = |id: u32, list: &[(u32, f64)]| {
            list.iter().position(|(e, _)| *e == id)
        };
        let r9_first = rank1(9, &preds1).unwrap_or(usize::MAX);
        let r9_second = rank1(9, &preds2).unwrap_or(usize::MAX);
        assert!(
            r9_second < r9_first || (r9_second == 0),
            "2nd-order should rank 9 at least as high as 1st-order: \
             1st={preds1:?} 2nd={preds2:?}"
        );
    }

    #[test]
    fn observe_step2_falls_back_to_first_order_when_prev_prev_empty() {
        let p = PredictiveLoader::new(8, 2, 0.0, 1);
        // Empty prev_prev means we only get 1st-order observations.
        p.observe_step2(&[], &[1], &[3]);
        let preds = p.predict_next(1);
        assert_eq!(preds[0].0, 3);
    }

    // ---------------------- LocalityMonitor tests ---------------------

    #[test]
    fn locality_monitor_tracks_window_and_heat_counts() {
        let m = LocalityMonitor::new(8, 4);
        m.observe(&[1, 2, 1, 3]);
        assert_eq!(m.len(), 4);
        assert_eq!(m.heat(1), 2);
        assert_eq!(m.heat(2), 1);
        assert_eq!(m.heat(3), 1);
        // Adding a 5th observation evicts the oldest (id=1).
        m.observe_one(5);
        assert_eq!(m.len(), 4);
        assert_eq!(m.heat(1), 1);
        assert_eq!(m.heat(5), 1);
    }

    #[test]
    fn locality_monitor_hot_set_above_threshold() {
        let m = LocalityMonitor::new(8, 10);
        // Expert 4 is hammered; others get one observation each.
        for _ in 0..7 {
            m.observe_one(4);
        }
        m.observe(&[0, 1, 2]);
        // 70% > 10% -> expert 4 is hot.
        assert!(m.is_hot(4, 0.10));
        // 10% threshold: only ids whose count >= 1 (== 10%*10) qualify
        // — i.e. every observed id (0,1,2,4).
        let hot = m.hot_set(0.10);
        assert!(hot.contains(&4), "expected 4 in hot set: {hot:?}");
        // 50% threshold: only expert 4 (>=5 obs) qualifies.
        let hot_strict = m.hot_set(0.50);
        assert_eq!(hot_strict, vec![4]);
    }

    #[test]
    fn locality_monitor_ignores_out_of_range_ids() {
        let m = LocalityMonitor::new(4, 8);
        m.observe(&[0, 1, 99, 2, 100]);
        assert_eq!(m.len(), 3);
        assert!(!m.is_hot(99, 0.0));
    }

    #[test]
    fn locality_monitor_clear_resets_state() {
        let m = LocalityMonitor::new(4, 8);
        m.observe(&[0, 1, 2, 3, 0]);
        assert!(m.len() > 0);
        m.clear();
        assert_eq!(m.len(), 0);
        assert_eq!(m.heat(0), 0);
        assert!(m.hot_set(0.0).is_empty());
    }

    // ---------------------- NeuralSpeculator tests --------------------

    #[test]
    fn speculator_predict_topk_returns_distinct_sorted_ids() {
        let s = NeuralSpeculator::new(16, 32, 8, 1);
        let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.5).collect();
        let top = s.predict_topk(&x, 3);
        assert_eq!(top.len(), 3);
        // distinct
        let mut sorted = top.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
        for id in top {
            assert!(id < 8);
        }
    }

    #[test]
    fn speculator_initial_distribution_is_deterministic() {
        let a = NeuralSpeculator::new(8, 16, 4, 42);
        let b = NeuralSpeculator::new(8, 16, 4, 42);
        let x: Vec<f32> = (0..8).map(|i| (i as f32).sin()).collect();
        assert_eq!(a.forward(&x), b.forward(&x));
    }

    #[test]
    fn speculator_train_step_reduces_loss_on_fixed_target() {
        // Simple test: repeatedly train the speculator against a single
        // (x, target) pair and verify the cross-entropy loss decreases.
        let s = NeuralSpeculator::new(8, 16, 4, 7);
        let x: Vec<f32> = vec![0.5; 8];
        let target = [2u32, 3];
        let initial = s.train_step(&x, &target, 0.1);
        for _ in 0..200 {
            s.train_step(&x, &target, 0.1);
        }
        let after = s.train_step(&x, &target, 0.0); // 0 lr -> just measure loss
        assert!(initial.is_finite());
        assert!(after.is_finite());
        assert!(
            after < initial * 0.9,
            "expected loss to decrease materially: initial={initial} after={after}"
        );
        // After training, the top-2 prediction must include both target ids.
        let preds = s.predict_topk(&x, 2);
        assert!(preds.contains(&2), "preds={preds:?}");
        assert!(preds.contains(&3), "preds={preds:?}");
    }

    #[test]
    fn speculator_learns_hidden_derived_labels_above_random() {
        let d_model = 8usize;
        let num_experts = 4u32;
        let s = NeuralSpeculator::new(d_model, 32, num_experts, 0x1EAF);
        let samples: Vec<(Vec<f32>, u32)> = (0..num_experts)
            .map(|id| {
                let mut x = vec![-0.5f32; d_model];
                x[id as usize] = 1.0;
                x[id as usize + num_experts as usize] =
                    if id % 2 == 0 { 0.25 } else { -0.25 };
                (x, id)
            })
            .collect();

        for _ in 0..500 {
            for (x, target) in &samples {
                s.train_step(x, &[*target], 0.05);
            }
        }

        let correct = samples
            .iter()
            .filter(|(x, target)| s.predict_topk(x, 1).first() == Some(target))
            .count();
        let accuracy = correct as f32 / samples.len() as f32;
        assert!(
            accuracy >= 0.75,
            "hidden-derived labels should be learnable well above random: {correct}/{}",
            samples.len()
        );
    }

    #[test]
    fn speculator_uncorrelated_labels_remain_near_random() {
        use rand::Rng;

        let d_model = 8usize;
        let num_experts = 8u32;
        let train_len = 2_048usize;
        let heldout_len = 2_048usize;
        let s = NeuralSpeculator::new(d_model, 16, num_experts, 0xC011_7E);

        let mut train_x_rng = StdRng::seed_from_u64(0xA11C_E5EED);
        let mut train_label_rng = StdRng::seed_from_u64(0x1A8E_15EED);
        let train: Vec<(Vec<f32>, u32)> = (0..train_len)
            .map(|_| {
                let x = (0..d_model)
                    .map(|_| train_x_rng.gen_range(-1.0f32..1.0))
                    .collect();
                let label = train_label_rng.gen_range(0..num_experts);
                (x, label)
            })
            .collect();

        for _ in 0..3 {
            for (x, target) in &train {
                s.train_step(x, &[*target], 0.05);
            }
        }

        let mut heldout_x_rng = StdRng::seed_from_u64(0x5E1F_1E55);
        let mut heldout_label_rng = StdRng::seed_from_u64(0xBADC_0FFE);
        let correct = (0..heldout_len)
            .filter(|_| {
                let x: Vec<f32> = (0..d_model)
                    .map(|_| heldout_x_rng.gen_range(-1.0f32..1.0))
                    .collect();
                let label = heldout_label_rng.gen_range(0..num_experts);
                s.predict_topk(&x, 1).first() == Some(&label)
            })
            .count();
        let accuracy = correct as f32 / heldout_len as f32;
        let random = 1.0 / num_experts as f32;
        let tolerance = 0.06;
        assert!(
            (random - tolerance..=random + tolerance).contains(&accuracy),
            "held-out labels are independent of hidden states; accuracy should stay near random ({random}), got {accuracy} ({correct}/{heldout_len})"
        );
    }

    #[test]
    fn speculator_training_worker_consumes_samples_and_advances_steps() {
        let s = std::sync::Arc::new(NeuralSpeculator::new(4, 8, 4, 0x57E9));
        s.spawn_training_worker();
        let x = vec![0.25f32, -0.5, 0.75, 1.0];
        for _ in 0..16 {
            s.queue_train(&x, &[2], 0.1);
        }

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while s.train_steps.load(Ordering::Relaxed) == 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert!(
            s.train_steps.load(Ordering::Relaxed) > 0,
            "background worker should consume queued samples"
        );
    }

    #[test]
    fn speculator_handles_empty_or_invalid_input() {
        let s = NeuralSpeculator::new(4, 8, 4, 1);
        assert!(s.predict_topk(&[1.0; 4], 0).is_empty());
        assert!(s.predict_topk(&[], 2).is_empty());
        // Wrong-length input: predict returns empty rather than panicking.
        assert!(s.predict_topk(&[1.0; 8], 2).is_empty());
        // Train with empty actual set -> no-op (NaN sentinel).
        assert!(s.train_step(&[1.0; 4], &[], 0.1).is_nan());
        // Train with all out-of-range ids -> no-op.
        assert!(s.train_step(&[1.0; 4], &[99, 100], 0.1).is_nan());
    }

    #[test]
    fn speculator_predict_topk_with_probs_is_normalised() {
        let s = NeuralSpeculator::new(4, 8, 8, 11);
        let x: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
        let preds = s.predict_topk_with_probs(&x, 8);
        assert_eq!(preds.len(), 8);
        let total: f32 = preds.iter().map(|(_, p)| *p).sum();
        assert!((total - 1.0).abs() < 1e-4, "softmax sums to {total}");
        // Sorted descending by probability.
        for w in preds.windows(2) {
            assert!(w[0].1 + 1e-6 >= w[1].1);
        }
    }

    #[test]
    fn speculator_predict_topk_for_layer_restricts_to_window() {
        // 3 layers * 4 experts/layer = 12 global ids.
        let s = NeuralSpeculator::new(8, 16, 12, 5);
        let x: Vec<f32> = (0..8).map(|i| (i as f32).cos()).collect();
        let per_layer = 4u32;
        for layer in 0..3u32 {
            let preds = s.predict_topk_for_layer(&x, layer, per_layer, 2);
            assert!(preds.len() <= 2);
            let base = layer * per_layer;
            for id in preds {
                assert!(
                    (base..base + per_layer).contains(&id),
                    "id {id} outside layer {layer} window [{base},{})",
                    base + per_layer
                );
            }
        }
    }

    #[test]
    fn speculator_predict_topk_for_layer_descending_and_deterministic() {
        let s = NeuralSpeculator::new(8, 16, 12, 9);
        let x: Vec<f32> = vec![0.2; 8];
        // Full window, ranked: must be descending by the slice's logits.
        let preds = s.predict_topk_for_layer(&x, 1, 4, 4);
        assert_eq!(preds.len(), 4);
        let logits = s.forward(&x);
        for w in preds.windows(2) {
            assert!(
                logits[w[0] as usize] >= logits[w[1] as usize],
                "not descending by logit"
            );
        }
        // Deterministic.
        assert_eq!(preds, s.predict_topk_for_layer(&x, 1, 4, 4));
    }

    #[test]
    fn speculator_predict_topk_for_layer_guards_invalid_input() {
        let s = NeuralSpeculator::new(4, 8, 8, 1); // 8 global ids
        // per_layer == 0 -> empty.
        assert!(s.predict_topk_for_layer(&[0.1; 4], 0, 0, 2).is_empty());
        // k == 0 -> empty.
        assert!(s.predict_topk_for_layer(&[0.1; 4], 0, 4, 0).is_empty());
        // Wrong feature width -> empty (no panic).
        assert!(s.predict_topk_for_layer(&[0.1; 8], 0, 4, 2).is_empty());
        // Layer past the id space -> empty (base = 4*4 = 16 >= 8).
        assert!(s.predict_topk_for_layer(&[0.1; 4], 4, 4, 2).is_empty());
        // Final partial window is clamped to the id count (base 4, end min(8,8)).
        let last = s.predict_topk_for_layer(&[0.1; 4], 1, 4, 9);
        assert!(last.len() <= 4);
        for id in last {
            assert!((4..8).contains(&id));
        }
    }

    #[test]
    fn locality_monitor_get_hot_experts_aliases_hot_set() {
        // Spec-named alias must be byte-equivalent to `hot_set`.
        let m = LocalityMonitor::new(4, 16);
        for _ in 0..6 { m.observe_one(2); }
        for _ in 0..3 { m.observe_one(0); }
        let from_alias = m.get_hot_experts(0.10);
        let from_orig = m.hot_set(0.10);
        assert_eq!(from_alias, from_orig);
        assert!(from_alias.contains(&2));
    }

    #[test]
    fn unified_headline_weights_sum_to_one() {
        // The three headline arms (speculator + markov + locality) must
        // partition a probability of 1.0 so `combine_unified_arms` never
        // produces a score > 1 for a single-source id. Asserting it here
        // makes the contract a test-time guarantee independent of the
        // per-call `debug_assert` in `combine_unified_arms`.
        assert!((W_SPECULATOR + W_MARKOV + W_LOCALITY - 1.0).abs() < 1e-6);
    }

    #[test]
    fn combine_unified_arms_applies_documented_weights() {
        // The engine's hot path consumes `combine_unified_arms` with
        // precomputed arms; assert it applies the canonical weights
        // (0.33 markov, 0.25 locality, 0.42 speculator) and that an id
        // hit by multiple arms outranks one hit by a single arm.
        let p = PredictiveLoader::new(8, 8, 0.0, 1);
        // Markov arm: id=1 at p=1.0; locality arm: id=2; speculator arm:
        // id=3 and id=1 (so id=1 is hit by markov + speculator).
        let markov = vec![(1u32, 1.0f64)];
        let locality = vec![2u32];
        let speculator = vec![3u32, 1u32];
        let out = p.combine_unified_arms(&markov, &locality, &speculator);
        let score = |id: u32| out.iter().find(|(i, _)| *i == id).map(|(_, s)| *s);
        // id=1: W_MARKOV*1.0 + W_SPECULATOR at rank 1 (rank-decayed).
        let spec_rank1 = W_SPECULATOR * (1.0 - RANK_TIEBREAK_DECAY);
        assert!((score(1).unwrap() - (W_MARKOV + spec_rank1)).abs() < 1e-6);
        // id=3: W_SPECULATOR at rank 0 (full weight).
        assert!((score(3).unwrap() - W_SPECULATOR).abs() < 1e-6);
        // id=2: W_LOCALITY at rank 0 (full weight).
        assert!((score(2).unwrap() - W_LOCALITY).abs() < 1e-6);
        // Ranking: multi-arm id=1 first, then speculator id=3, then
        // locality id=2.
        assert_eq!(out.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![1, 3, 2]);
        // Result is not truncated to fanout (caller's responsibility).
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn combine_unified_arms_preserves_arm_internal_ranking() {
        // The locality hot set arrives heat-sorted and the speculator
        // top-K logit-sorted. A flat per-arm weight used to collapse
        // each arm into an all-ties group whose deterministic
        // ascending-id tie-break systematically favoured low (low-
        // layer) ids under budget truncation. The per-rank decay must
        // keep each arm's ids in their *given* order — here
        // deliberately descending-id so an ascending-id tie-break
        // would invert it.
        let p = PredictiveLoader::new(16, 16, 0.0, 7);
        let locality = vec![9u32, 5, 2];
        let speculator = vec![14u32, 11, 3];
        let out = p.combine_unified_arms(&[], &locality, &speculator);
        let ids: Vec<u32> = out.iter().map(|(i, _)| *i).collect();
        // Speculator arm (0.42) outranks locality (0.25) throughout,
        // and within each arm the caller-supplied order survives.
        assert_eq!(ids, vec![14, 11, 3, 9, 5, 2]);
        // Decay is tiny: rank-2 locality keeps >99.9% of its weight.
        let score_2 = out.iter().find(|(i, _)| *i == 2).unwrap().1;
        assert!(score_2 > W_LOCALITY * 0.999 && score_2 < W_LOCALITY);
    }

    #[test]
    fn predict_unified_blends_three_arms() {
        // 8 experts, fanout 8 so we get every contributing arm in
        // the output. Markov hand-trained to predict 0->1, locality
        // pre-warmed with id=2, speculator predicts id=3.
        let p = PredictiveLoader::new(8, 8, 0.0, 42);
        for _ in 0..20 { p.observe(0, 1); }
        let monitor = LocalityMonitor::new(8, 16);
        for _ in 0..6 { monitor.observe_one(2); }
        let speculator = NeuralSpeculator::new(4, 8, 8, 7);
        let hidden = vec![0.5f32, -0.2, 0.3, 0.1];
        // Force speculator's top-1 to be id=3 by training on id=3
        // many times until that becomes its top output for `hidden`.
        for _ in 0..200 {
            speculator.train_step(&hidden, &[3], 0.1);
        }
        let unified = p.predict_unified(
            None,
            0,
            Some(&monitor),
            0.10,
            Some(&speculator),
            &hidden,
            1,
        );
        // Result is ordered by descending priority.
        for w in unified.windows(2) {
            assert!(
                w[0].1 + 1e-6 >= w[1].1,
                "priority must be sorted descending, got {:?}",
                unified
            );
        }
        // Each of the three contributing ids appears at least once.
        let ids: Vec<u32> = unified.iter().map(|&(id, _)| id).collect();
        for expected in [1u32, 2u32, 3u32] {
            assert!(ids.contains(&expected),
                "predict_unified missing id {expected}: {unified:?}");
        }
        // No score is negative.
        for &(_, s) in &unified {
            assert!(s > 0.0);
        }
    }

    #[test]
    fn predict_unified_handles_no_optional_arms() {
        // With monitor=None and speculator=None the unified ranking
        // should preserve the Markov-only `predict_next` ordering. We
        // use *distinct* observation counts so there are no
        // probability ties — `predict_next` and `predict_unified`
        // break ties slightly differently (the latter uses ascending
        // id as a deterministic tiebreaker, the former is stable in
        // input order), which would otherwise be a benign reordering
        // that this test isn't trying to assert against.
        let p = PredictiveLoader::new(4, 4, 0.0, 42);
        for _ in 0..15 { p.observe(0, 1); }
        for _ in 0..7 { p.observe(0, 2); }
        for _ in 0..3 { p.observe(0, 3); }
        let unified = p.predict_unified(
            None,
            0,
            None,
            0.0,
            None,
            &[],
            0,
        );
        let markov = p.predict_next(0);
        assert_eq!(unified.len(), markov.len());
        for ((id_u, _), (id_m, _)) in unified.iter().zip(markov.iter()) {
            assert_eq!(id_u, id_m);
        }
        // And — the headline guarantee of the normalised weights —
        // a 100%-Markov hit (no arms missing, p≈1) cannot exceed
        // 1.0. Here the top expert has p≈15/25=0.6 and the only
        // contribution is W_MARKOV * p = 0.33 * 0.6 ≈ 0.198, so
        // every score is comfortably below 1.
        for &(_, s) in &unified {
            assert!(s >= 0.0 && s <= 1.0, "score {s} outside [0,1]");
        }
    }

    #[test]
    fn predict_unified_score_is_bounded_by_one() {
        // A unanimous expert (p=1 Markov, hot in locality, top of the
        // speculator) must score ≤ 1.0 under the normalised weights —
        // this is the headline guarantee of the new design.
        let p = PredictiveLoader::new(4, 4, 0.0, 42);
        // Make Markov certain that 0 -> 1 (every observation is the
        // same transition).
        for _ in 0..200 { p.observe(0, 1); }
        let monitor = LocalityMonitor::new(4, 8);
        // Saturate locality so id=1 is well above the 10% threshold.
        for _ in 0..16 { monitor.observe_one(1); }
        let speculator = NeuralSpeculator::new(4, 8, 4, 7);
        let hidden = vec![0.5f32, -0.2, 0.3, 0.1];
        // Train the speculator hard so id=1 dominates the top-K for `hidden`.
        for _ in 0..400 { speculator.train_step(&hidden, &[1], 0.1); }
        let unified = p.predict_unified(
            None,
            0,
            Some(&monitor),
            0.10,
            Some(&speculator),
            &hidden,
            1,
        );
        // id=1 should be the top entry and its score must respect the
        // [0, 1] bound.
        let top = unified.first().expect("non-empty result");
        assert_eq!(top.0, 1);
        assert!(top.1 <= 1.0 + 1e-6, "top score {} exceeded 1.0", top.1);
    }

    // ------------ ExpertAffinity / spatial / speculation tests ------------

    #[test]
    fn expert_affinity_records_symmetric_co_occurrence() {
        let aff = ExpertAffinity::new(8);
        aff.observe_layer(&[1, 3, 5]);
        // Every unordered pair (1,3),(1,5),(3,5) should be counted once
        // in each direction (matrix is symmetric).
        assert_eq!(aff.affinity(1, 3), 1);
        assert_eq!(aff.affinity(3, 1), 1);
        assert_eq!(aff.affinity(1, 5), 1);
        assert_eq!(aff.affinity(5, 1), 1);
        assert_eq!(aff.affinity(3, 5), 1);
        // Diagonal stays zero.
        assert_eq!(aff.affinity(1, 1), 0);
        // Out-of-range ids return zero rather than panicking.
        assert_eq!(aff.affinity(1, 99), 0);
        // total_observations counts calls, not pairs.
        assert_eq!(aff.total_observations(), 1);

        // Repeated layer observation accumulates.
        for _ in 0..4 {
            aff.observe_layer(&[1, 3]);
        }
        assert_eq!(aff.affinity(1, 3), 5);
        assert_eq!(aff.total_observations(), 5);
    }

    #[test]
    fn expert_affinity_neighbors_ranked_by_count() {
        let aff = ExpertAffinity::new(16);
        // Co-fire 2 + {5, 7, 9} a lot, 2 + {1} once.
        for _ in 0..10 { aff.observe_layer(&[2, 5]); }
        for _ in 0..7  { aff.observe_layer(&[2, 7]); }
        for _ in 0..3  { aff.observe_layer(&[2, 9]); }
        aff.observe_layer(&[2, 1]);
        let nbrs = aff.neighbors(2, 3);
        assert_eq!(nbrs, vec![5, 7, 9]);
        // k=0 → empty.
        assert!(aff.neighbors(2, 0).is_empty());
        // Out-of-range id → empty.
        assert!(aff.neighbors(99, 4).is_empty());
    }

    #[test]
    fn layered_affinity_isolates_co_occurrences_per_layer() {
        // gist Part 1, fix #2: a co-firing of (4, 6) in layer 0
        // must not appear as a neighbour signal for layer 1.
        let l = LayeredExpertAffinity::new(/*num_layers=*/ 2, /*num_experts=*/ 8);
        for _ in 0..5 {
            l.observe_layer(0, &[4, 6]);
        }
        // Layer 0: 4 → 6 (and 6 → 4) is the strongest neighbour.
        assert_eq!(l.neighbors(0, 4, 1), vec![6]);
        assert_eq!(l.neighbors(0, 6, 1), vec![4]);
        assert_eq!(l.affinity(0, 4, 6), 5);
        // Layer 1: never observed → all zero.
        assert_eq!(l.affinity(1, 4, 6), 0);
        assert!(l.neighbors(1, 4, 1).is_empty());
        // Out-of-range layer index is silently dropped (telemetry
        // must never panic the inference path).
        l.observe_layer(99, &[0, 1]);
        assert_eq!(l.affinity(99, 0, 1), 0);
        assert!(l.neighbors(99, 0, 1).is_empty());
    }

    #[test]
    fn expert_affinity_decay_halves_counts_without_data_loss() {
        // gist Part 2, fix #7: bit-shift decay ages co-occurrences
        // out of the matrix without resetting it to zero.
        let aff = ExpertAffinity::new(8);
        for _ in 0..16 {
            aff.observe_layer(&[3, 5]);
        }
        let before = aff.affinity(3, 5);
        assert!(before >= 16);
        aff.decay(1);
        let after = aff.affinity(3, 5);
        // One right-shift ≈ halving (within ±1 due to integer
        // truncation on the symmetric pair update).
        assert!(after <= before / 2 + 1, "after={after} vs before={before}");
        assert!(after > 0, "decay must not wipe the signal");
        // observation counter is preserved across decay.
        assert!(aff.total_observations() >= 16);
    }

    #[test]
    fn layered_affinity_decay_worker_shifts_counters() {
        // gist Part 2, fix #7 — wired through the LayeredExpertAffinity
        // background worker. Spawn the worker first (so it captures
        // the "zero observations" baseline), then drive enough new
        // co-firings to cross `epoch_threshold` and confirm the
        // worker right-shifts.
        let l = std::sync::Arc::new(LayeredExpertAffinity::new(1, 8));
        let handle = l
            .clone()
            .spawn_decay_worker(
                /*epoch_threshold=*/ 1,
                /*bits=*/ 1,
                /*poll_interval=*/ std::time::Duration::from_millis(5),
            );
        for _ in 0..32 {
            l.observe_layer(0, &[2, 4]);
        }
        let pre = l.affinity(0, 2, 4);
        // Wait long enough for the worker to observe one epoch.
        // Multiple poll intervals (≥10 polls of 5 ms) covers the
        // OS-scheduler tail under load while still keeping the
        // test latency bounded.
        std::thread::sleep(std::time::Duration::from_millis(300));
        let post = l.affinity(0, 2, 4);
        handle.shutdown();
        assert!(
            post < pre,
            "decay worker should have shifted counters down (pre={pre} post={post})"
        );
    }

    #[test]
    fn spatial_neighbors_clips_to_bounds() {
        assert_eq!(spatial_neighbors(0, 8, 2), vec![1, 2]);
        assert_eq!(spatial_neighbors(7, 8, 2), vec![6, 5]);
        // Middle: returns id-1 then id+1.
        let mid = spatial_neighbors(4, 8, 4);
        assert_eq!(mid, vec![3, 5, 2, 6]);
        // k=0 → empty.
        assert!(spatial_neighbors(4, 8, 0).is_empty());
        // Out-of-range id → empty.
        assert!(spatial_neighbors(8, 8, 2).is_empty());
    }

    #[test]
    fn speculation_controller_grows_under_rising_stall() {
        let ctl = SpeculationController::new(2);
        assert_eq!(ctl.current_depth(), 2);
        // First call sets the baseline; no delta yet.
        ctl.update_from_stall(0);
        assert_eq!(ctl.current_depth(), 2);
        // Big jump in cumulative stall → +1.
        ctl.update_from_stall(5_000);
        assert_eq!(ctl.current_depth(), 3);
        // Another big jump → +1 (capped at base + MAX_LATENCY_BUMP = 4).
        ctl.update_from_stall(15_000);
        assert_eq!(ctl.current_depth(), 4);
        // Saturates at the cap.
        ctl.update_from_stall(99_000);
        assert_eq!(ctl.current_depth(), 4);
    }

    #[test]
    fn speculation_controller_backs_off_when_stall_calms() {
        let ctl = SpeculationController::new(2);
        ctl.update_from_stall(0);
        ctl.update_from_stall(10_000); // depth 3
        assert_eq!(ctl.current_depth(), 3);
        // Two consecutive calm updates → -1.
        ctl.update_from_stall(10_000); // delta 0
        ctl.update_from_stall(10_000); // delta 0 — streak hits 2
        assert_eq!(ctl.current_depth(), 2);
        // Never goes below base.
        ctl.update_from_stall(10_000);
        ctl.update_from_stall(10_000);
        assert_eq!(ctl.current_depth(), 2);
    }

    #[test]
    fn speculation_controller_suspend_resume_zeroes_depth() {
        let ctl = SpeculationController::new(3);
        ctl.update_from_stall(0);
        ctl.update_from_stall(5_000); // depth 4
        ctl.suspend();
        assert_eq!(ctl.current_depth(), 0);
        assert!(ctl.is_suspended());
        // Updates while suspended do not change the depth.
        ctl.update_from_stall(50_000);
        assert_eq!(ctl.current_depth(), 0);
        ctl.resume();
        assert!(!ctl.is_suspended());
        // The depth at the moment of suspension is restored.
        assert_eq!(ctl.current_depth(), 4);
    }

    #[test]
    fn predict_unified_with_spatial_adds_neighbours() {
        // Train the Markov chain so prev=0 → next=4 with near-certainty.
        let p = PredictiveLoader::new(8, 8, 0.0, 1);
        for _ in 0..200 { p.observe(0, 4); }
        let monitor = LocalityMonitor::new(8, 16);
        for _ in 0..16 { monitor.observe_one(4); }
        let speculator = NeuralSpeculator::new(4, 8, 8, 7);
        let hidden = vec![0.5f32, -0.2, 0.3, 0.1];
        for _ in 0..400 { speculator.train_step(&hidden, &[4], 0.1); }
        // Affinity: 4 strongly co-fires with 6.
        let aff = ExpertAffinity::new(8);
        for _ in 0..50 { aff.observe_layer(&[4, 6]); }

        let with_spatial = p.predict_unified_with_spatial(
            None,
            0,
            Some(&monitor),
            0.10,
            Some(&speculator),
            &hidden,
            1,
            Some(&aff),
            2,
        );
        // Seed (4) is high-confidence → spatial neighbours 3 and 5 +
        // affinity neighbour 6 must appear in the candidate set.
        let ids: Vec<u32> = with_spatial.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&4));
        assert!(ids.contains(&3) || ids.contains(&5),
            "expected at least one UTH neighbour of seed 4 in {ids:?}");
        assert!(ids.contains(&6),
            "expected affinity neighbour 6 in {ids:?}");
    }
}
