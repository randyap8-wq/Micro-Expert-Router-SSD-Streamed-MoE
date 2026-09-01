//! Per-layer expert cache for multi-layer MoE models (gist Phase 5,
//! "Option B: per-layer caches").
//!
//! Mixtral has 32 layers, each with its own pool of 8 experts. A flat
//! `expert_id` namespace `0..N-1` cannot represent that without forcing
//! every layer's experts onto a single shared LRU — which would let layer
//! 5's prefetched experts evict layer 0's, defeating the cache.
//!
//! [`MultiLayerExpertCache`] owns one [`crate::expert_cache::ExpertCache`]
//! per layer plus the `experts_per_layer` stride used to derive
//! `(layer, local_id)` from the *global* expert id encoded in
//! [`ExpertResident::id`]. The rest of the engine still threads a single
//! id-space through router, predictor and cache APIs — the wrapper just
//! dispatches each call to the per-layer LRU that owns it. For single-
//! layer models (the in-tree `serve` path) use [`Self::single_layer`],
//! which gives the same observable behaviour as the original flat
//! `ExpertCache`.
//!
//! The on-disk file naming convention is `expert_<layer>_<id>.bin` for
//! multi-layer models (single-layer models continue to use
//! `expert_<id>.bin`, written by the existing extractor).

use crate::expert_cache::{ExpertCache, ExpertResident};
use std::sync::Arc;

/// Fixed `(layer, expert)` key for a multi-layer expert lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub fn new(layer: u32, expert: u32) -> Self {
        Self { layer, expert }
    }
}

/// One [`ExpertCache`] per layer. Capacities can be set per-layer (e.g.
/// to give "hot" early layers more residency budget) or uniformly via
/// [`MultiLayerExpertCache::with_uniform_capacity`].
///
/// `experts_per_layer` is the stride used to decode a global expert id
/// into `(layer, local_id)` — `layer = id / experts_per_layer`,
/// `local_id = id % experts_per_layer`. The engine builds resident
/// experts with global ids (see `model.rs`'s layer-qualified id space),
/// so this stride must match the model's layout. Single-layer models
/// can use [`Self::single_layer`], which sets the stride to `u32::MAX`
/// so every id maps to layer 0.
pub struct MultiLayerExpertCache {
    caches: Vec<Arc<ExpertCache>>,
    experts_per_layer: u32,
}

impl MultiLayerExpertCache {
    /// Build a cache with `num_layers` per-layer caches, each of
    /// capacity `cap_per_layer`. `experts_per_layer` is the stride
    /// used to decode global expert ids into `(layer, local)`.
    pub fn with_uniform_capacity(
        num_layers: usize,
        cap_per_layer: usize,
        experts_per_layer: u32,
    ) -> Self {
        assert!(num_layers > 0, "num_layers must be > 0");
        assert!(experts_per_layer > 0, "experts_per_layer must be > 0");
        let caches = (0..num_layers)
            .map(|_| Arc::new(ExpertCache::new(cap_per_layer)))
            .collect();
        Self {
            caches,
            experts_per_layer,
        }
    }

    /// Build a cache from explicit per-layer capacities.
    pub fn with_capacities(per_layer_caps: Vec<usize>, experts_per_layer: u32) -> Self {
        assert!(!per_layer_caps.is_empty(), "must have at least one layer");
        assert!(experts_per_layer > 0, "experts_per_layer must be > 0");
        let caches = per_layer_caps
            .into_iter()
            .map(|c| Arc::new(ExpertCache::new(c)))
            .collect();
        Self {
            caches,
            experts_per_layer,
        }
    }

    /// Single-layer convenience: one underlying `ExpertCache` of
    /// `capacity`, with the stride set to `u32::MAX` so every global
    /// id maps to layer 0. Used by the in-tree `serve` path and tests
    /// that haven't been ported to a real multi-layer model yet.
    pub fn single_layer(capacity: usize) -> Self {
        Self {
            caches: vec![Arc::new(ExpertCache::new(capacity))],
            experts_per_layer: u32::MAX,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.caches.len()
    }

    /// Decode a global expert id into the index of its per-layer
    /// cache. Returns `None` when `id` is out of the validated
    /// namespace (hardening pass, Part A6): a malformed id must never
    /// be clamped into the last layer's cache — lookups miss, inserts
    /// are rejected, and pin/unpin are no-ops for such ids.
    fn try_layer_idx(&self, id: u32) -> Option<usize> {
        let layer = (id / self.experts_per_layer) as usize;
        (layer < self.caches.len()).then_some(layer)
    }

    /// The per-layer cache index that owns the global expert id `id`
    /// (clamped to the last layer for out-of-range ids). Budgeting /
    /// diagnostics only — the id-keyed cache operations use the
    /// non-clamping [`Self::try_layer_idx`] instead.
    pub fn layer_of(&self, id: u32) -> usize {
        let layer = (id / self.experts_per_layer) as usize;
        layer.min(self.caches.len().saturating_sub(1))
    }

    /// Residency capacity of one per-layer cache. Returns 0 for an
    /// out-of-range layer index so callers can treat "unknown layer"
    /// as "no budget".
    pub fn capacity_of_layer(&self, layer: usize) -> usize {
        self.caches.get(layer).map(|c| c.capacity()).unwrap_or(0)
    }

    /// Borrow the [`ExpertCache`] for one layer (so existing engine code
    /// that takes an `Arc<ExpertCache>` keeps working). Panics if
    /// `layer` is out of range — call sites that may receive an
    /// untrusted layer index should pre-validate against
    /// [`Self::num_layers`].
    pub fn cache_for_layer(&self, layer: u32) -> Arc<ExpertCache> {
        let idx = layer as usize;
        assert!(
            idx < self.caches.len(),
            "MultiLayerExpertCache::cache_for_layer: layer {} out of range (num_layers = {})",
            layer,
            self.caches.len()
        );
        self.caches[idx].clone()
    }

    // --- ExpertCache-mirroring API on global expert ids ------------------
    //
    // The engine hot path operates on global ids; these methods route each
    // call to the per-layer LRU that owns it. Aggregate getters
    // (`len`/`capacity`/`pinned_count`/`resident_ids`) sum across layers
    // so existing diagnostics keep reporting whole-engine totals.

    pub fn get(&self, id: u32) -> Option<Arc<ExpertResident>> {
        self.caches[self.try_layer_idx(id)?].get(id)
    }

    pub fn contains(&self, id: u32) -> bool {
        self.try_layer_idx(id)
            .is_some_and(|idx| self.caches[idx].contains(id))
    }

    pub fn insert(
        &self,
        resident: Arc<ExpertResident>,
    ) -> Result<Option<Arc<ExpertResident>>, Arc<ExpertResident>> {
        // Out-of-namespace ids are rejected, never clamped into the
        // last layer's cache (Part A6).
        let Some(idx) = self.try_layer_idx(resident.id) else {
            return Err(resident);
        };
        self.caches[idx].insert(resident)
    }

    pub fn pin(&self, id: u32) {
        if let Some(idx) = self.try_layer_idx(id) {
            self.caches[idx].pin(id);
        }
    }

    pub fn unpin(&self, id: u32) {
        if let Some(idx) = self.try_layer_idx(id) {
            self.caches[idx].unpin(id);
        }
    }

    /// Whether an expert is already pinned in its owning per-layer cache.
    /// Qualification-only callers use this to install temporary demand-set
    /// pins without accidentally removing a pre-existing production pin when
    /// the temporary guard is dropped.
    pub(crate) fn is_pinned(&self, id: u32) -> bool {
        self.try_layer_idx(id)
            .is_some_and(|idx| self.caches[idx].is_pinned(id))
    }

    /// Snapshot the current resident count of every per-layer cache.
    ///
    /// The exact-demand source-concurrency qualifier uses this as the start
    /// of a private, deterministic simulation of the existing sequential
    /// `evict -> read -> insert` schedule. It never changes cache capacity or
    /// policy.
    pub(crate) fn qualification_layer_lengths(&self) -> Vec<usize> {
        self.caches.iter().map(|cache| cache.len()).collect()
    }

    /// Evict from the layer the ordinary global victim selector would choose
    /// after accounting for qualification-only virtual ordered inserts.
    /// Ties retain the existing lowest-layer-index rule.
    pub(crate) fn qualification_evict_lru_with_virtual_lengths(
        &self,
        virtual_lengths: &[usize],
    ) -> Option<Arc<ExpertResident>> {
        if virtual_lengths.len() != self.caches.len() {
            return None;
        }
        let mut best: Option<(usize, usize)> = None;
        for (idx, &len) in virtual_lengths.iter().enumerate() {
            if len == 0 {
                continue;
            }
            match best {
                Some((_, best_len)) if len <= best_len => {}
                _ => best = Some((idx, len)),
            }
        }
        let (start, _) = best?;
        let n = self.caches.len();
        for offset in 0..n {
            let idx = (start + offset) % n;
            if virtual_lengths[idx] == 0 {
                continue;
            }
            if let Some(resident) = self.caches[idx].evict_lru() {
                return Some(resident);
            }
        }
        None
    }

    /// Evict the ordinary non-pinned LRU victim from one exact layer.
    /// This mirrors the victim selection performed by `ExpertCache::insert`
    /// when that layer is already at capacity.
    pub(crate) fn qualification_evict_lru_from_layer(
        &self,
        layer: usize,
    ) -> Option<Arc<ExpertResident>> {
        self.caches.get(layer)?.evict_lru()
    }

    /// **Tier 4 — cost-aware eviction.** Enable or disable the
    /// lowest-heat eviction policy across every per-layer cache. No-op
    /// effect until at least one layer fills; off by default so the
    /// engine preserves pure-LRU behaviour unless asked.
    pub fn set_cost_aware(&self, on: bool) {
        for c in &self.caches {
            c.set_cost_aware(on);
        }
    }

    /// Pop a least-recently-used non-pinned entry. With multiple
    /// layers, evicts from the layer whose per-layer LRU has the most
    /// residents (so we relieve the most-pressured layer first); ties
    /// go to the lowest layer index. Returns `None` only when every
    /// resident across every layer is pinned.
    pub fn evict_lru(&self) -> Option<Arc<ExpertResident>> {
        let mut best: Option<(usize, usize)> = None;
        for (idx, cache) in self.caches.iter().enumerate() {
            let len = cache.len();
            if len == 0 {
                continue;
            }
            match best {
                Some((_, best_len)) if len <= best_len => {}
                _ => best = Some((idx, len)),
            }
        }
        let (start, _) = best?;
        // Try the heaviest layer first, then fall back to others in
        // case every entry there is pinned.
        let n = self.caches.len();
        for offset in 0..n {
            let idx = (start + offset) % n;
            if let Some(r) = self.caches[idx].evict_lru() {
                return Some(r);
            }
        }
        None
    }

    pub fn len(&self) -> usize {
        self.caches.iter().map(|c| c.len()).sum()
    }

    /// Pop a least-recently-used, non-pinned, **shadow-backed** entry
    /// (see [`ExpertCache::evict_lru_shadow_backed`]). Walks layers
    /// heaviest-first so Buffer B recycling relieves the most-pressured
    /// layer's LRU first. Returns `None` when no unpinned shadow-backed
    /// resident exists anywhere.
    pub fn evict_lru_shadow_backed(&self) -> Option<Arc<ExpertResident>> {
        // Snapshot each layer's length *once* before sorting. `len()`
        // takes a lock and reads live state, so calling it from inside
        // the comparator (as `sort_by_key` does, repeatedly per element)
        // lets a concurrent mutation change a key mid-sort. That makes
        // the ordering non-total and trips the `sort` total-order
        // assertion under load. Sorting over a stable snapshot keeps the
        // key fixed for the duration of the sort.
        let lens: Vec<usize> = self.caches.iter().map(|c| c.len()).collect();
        let mut order: Vec<usize> = (0..self.caches.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(lens[i]));
        for idx in order {
            if let Some(r) = self.caches[idx].evict_lru_shadow_backed() {
                return Some(r);
            }
        }
        None
    }

    pub fn capacity(&self) -> usize {
        self.caches.iter().map(|c| c.capacity()).sum()
    }

    pub fn pinned_count(&self) -> usize {
        self.caches.iter().map(|c| c.pinned_count()).sum()
    }

    /// Snapshot of all pinned ids across every per-layer cache, sorted
    /// ascending (diagnostics / tests).
    pub fn pinned_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.caches.iter().flat_map(|c| c.pinned_ids()).collect();
        ids.sort_unstable();
        ids
    }

    pub fn resident_ids(&self) -> Vec<u32> {
        let mut ids = Vec::with_capacity(self.len());
        for c in &self.caches {
            ids.extend(c.resident_ids());
        }
        ids
    }

    /// Total number of cached experts across all layers.
    pub fn total_resident(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;

    #[test]
    fn per_layer_caches_are_independent() {
        let pool = BufferPool::new(4, 4096, 4096);
        // experts_per_layer = 8 -> id 0 = (layer 0, local 0); id 8 = (layer 1, local 0)
        let mlc = MultiLayerExpertCache::with_uniform_capacity(2, 2, 8);

        // Insert expert 0 into layer 0 via the global-id API.
        let resident = Arc::new(ExpertResident::new(
            0,
            pool.try_acquire().unwrap(),
        ));
        let _ = mlc.insert(resident);

        assert!(mlc.contains(0));
        assert!(!mlc.contains(8));
        assert_eq!(mlc.total_resident(), 1);
        assert!(mlc.contains_at(ExpertKey::new(0, 0)));
        assert!(!mlc.contains_at(ExpertKey::new(1, 0)));
    }

    #[test]
    fn cache_for_layer_returns_clones_of_same_arc() {
        let mlc = MultiLayerExpertCache::with_uniform_capacity(3, 1, 4);
        let a = mlc.cache_for_layer(0);
        let b = mlc.cache_for_layer(0);
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn layer_of_and_capacity_of_layer_decode_global_ids() {
        let mlc = MultiLayerExpertCache::with_capacities(vec![3, 2], 8);
        assert_eq!(mlc.layer_of(0), 0);
        assert_eq!(mlc.layer_of(7), 0);
        assert_eq!(mlc.layer_of(8), 1);
        // Out-of-range ids clamp to the last layer (mirrors layer_idx).
        assert_eq!(mlc.layer_of(99), 1);
        assert_eq!(mlc.capacity_of_layer(0), 3);
        assert_eq!(mlc.capacity_of_layer(1), 2);
        assert_eq!(mlc.capacity_of_layer(5), 0);
    }

    /// Part A6 (checked expert namespace): a global id outside the
    /// validated `num_layers * experts_per_layer` namespace must never
    /// be clamped into the last layer's cache — lookups miss, inserts
    /// are rejected, and pin/unpin are no-ops.
    #[test]
    fn out_of_namespace_ids_are_rejected_not_clamped() {
        let pool = BufferPool::new(4, 4096, 4096);
        // 2 layers × 8 experts per layer → valid global ids are 0..16.
        let mlc = MultiLayerExpertCache::with_capacities(vec![2, 2], 8);

        // Boundary: last valid id (15) works normally.
        let _ = mlc.insert(Arc::new(ExpertResident::new(
            15,
            pool.try_acquire().unwrap(),
        )));
        assert!(mlc.contains(15));
        assert!(mlc.get(15).is_some());

        // First out-of-range id (16) and u32::MAX: miss + rejected insert.
        for bad in [16u32, u32::MAX] {
            assert!(!mlc.contains(bad));
            assert!(mlc.get(bad).is_none());
            let rejected = mlc.insert(Arc::new(ExpertResident::new(
                bad,
                pool.try_acquire().unwrap(),
            )));
            assert!(rejected.is_err(), "insert of id {bad} must be rejected");
            // The last layer's cache is untouched by the malformed id.
            assert!(!mlc.contains(bad));
            // pin/unpin must be no-ops, not panics or last-layer pins.
            mlc.pin(bad);
            mlc.unpin(bad);
            assert_eq!(mlc.pinned_count(), 0);
        }
    }

    #[test]
    fn evict_lru_shadow_backed_skips_primary_and_pinned() {
        // 2 primary + 2 shadow buffers; experts_per_layer=8, 1 layer.
        let pool = BufferPool::new_with_shadow(2, 2, 4096, 4096);
        let mlc = MultiLayerExpertCache::single_layer(4);

        // id 0: primary-backed; ids 1, 2: shadow-backed (1 is LRU).
        let _ = mlc.insert(Arc::new(ExpertResident::new(0, pool.try_acquire().unwrap())));
        let _ = mlc.insert(Arc::new(ExpertResident::new(
            1,
            pool.try_acquire_shadow().unwrap(),
        )));
        let _ = mlc.insert(Arc::new(ExpertResident::new(
            2,
            pool.try_acquire_shadow().unwrap(),
        )));

        // Pin the LRU shadow-backed resident: eviction must skip it and
        // take the *next* shadow-backed one (id 2), never primary id 0.
        mlc.pin(1);
        let evicted = mlc.evict_lru_shadow_backed().expect("one candidate left");
        assert_eq!(evicted.id, 2);
        assert!(evicted.is_shadow_backed());
        drop(evicted);
        // Its buffer must return to the SHADOW free list.
        assert!(pool.try_acquire_shadow().is_some());
        // No unpinned shadow-backed residents remain.
        assert!(mlc.evict_lru_shadow_backed().is_none());
        assert!(mlc.contains(0), "primary-backed resident untouched");
        assert!(mlc.contains(1), "pinned shadow-backed resident untouched");
    }

    #[test]
    fn single_layer_acts_like_flat_cache() {
        let pool = BufferPool::new(4, 4096, 4096);
        let mlc = MultiLayerExpertCache::single_layer(2);
        for id in [3u32, 7u32, 42u32] {
            let r = Arc::new(ExpertResident::new(id, pool.try_acquire().unwrap()));
            let _ = mlc.insert(r);
        }
        // Capacity is 2 so the oldest insertion (3) should have been
        // evicted on the third insert.
        assert_eq!(mlc.len(), 2);
        assert!(mlc.contains(7));
        assert!(mlc.contains(42));
        assert!(!mlc.contains(3));
        let mut ids = mlc.resident_ids();
        ids.sort();
        assert_eq!(ids, vec![7, 42]);
    }

    #[test]
    fn evict_lru_targets_most_loaded_layer() {
        let pool = BufferPool::new(8, 4096, 4096);
        let mlc = MultiLayerExpertCache::with_uniform_capacity(2, 4, 8);
        // Layer 0 gets 3 residents, layer 1 gets 1.
        for id in [0u32, 1, 2] {
            let r = Arc::new(ExpertResident::new(id, pool.try_acquire().unwrap()));
            let _ = mlc.insert(r);
        }
        let r = Arc::new(ExpertResident::new(8, pool.try_acquire().unwrap()));
        let _ = mlc.insert(r);
        assert_eq!(mlc.len(), 4);

        let evicted = mlc.evict_lru().expect("an eviction");
        // Evicts from layer 0 (heaviest) — LRU there is id 0.
        assert_eq!(evicted.id, 0);
        assert!(!mlc.contains(0));
        assert!(mlc.contains(8), "layer 1's expert untouched");
    }
}

impl MultiLayerExpertCache {
    /// `(layer, local)` → encoded global expert id, in the canonical
    /// stride-based encoding the engine emits everywhere.
    #[inline]
    fn global_id(&self, key: ExpertKey) -> u32 {
        key.layer
            .saturating_mul(self.experts_per_layer)
            .saturating_add(key.expert)
    }

    /// `(layer, local)` membership check — kept for tests/diagnostics
    /// that already use the explicit `ExpertKey` form.
    pub fn contains_at(&self, key: ExpertKey) -> bool {
        self.caches
            .get(key.layer as usize)
            .map(|c| c.contains(self.global_id(key)))
            .unwrap_or(false)
    }

}
