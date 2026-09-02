//! In-RAM LRU cache of resident experts.
//!
//! Each cache entry is an `Arc<ExpertResident>` whose buffer is owned by the
//! [`BufferPool`](crate::buffer_pool::BufferPool). Eviction simply drops the
//! `Arc`; once any in-flight inference also drops its handle, the underlying
//! `PooledBuffer` returns to the pool's free list automatically.
//!
//! When the on-disk expert file was produced by `gguf-convert` (its default
//! mode), the buffer starts with a 64-byte Unified Tensor Header padded out
//! to one block. [`ExpertResident::data`] transparently strips that prefix
//! so every consumer downstream sees only the bare weight payload —
//! existing code paths (the SwiGLU kernels, the cache verifier, the
//! synthetic-expert fixtures) don't need to learn about UTH.

use crate::buffer_pool::PooledBuffer;
use crate::gguf_loader::DEFAULT_BLOCK_ALIGN;
use crate::tensor_header::{MixedExpertHeader, TensorHeader};
use lru::LruCache;
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

static RESIDENT_EXPERT_BUFFER_BYTES: AtomicU64 = AtomicU64::new(0);

/// Bytes owned by all live CPU expert residents. This includes residents
/// temporarily held by an in-flight activation after LRU eviction, which is
/// the useful definition for process memory accounting.
pub fn resident_expert_buffer_bytes() -> u64 {
    RESIDENT_EXPERT_BUFFER_BYTES.load(Ordering::Relaxed)
}

/// Pack/unpack an `f64` heat score into the bits of an `AtomicU64` so it
/// can be read and updated without a lock. The cost-aware eviction
/// scorer tolerates the occasional torn update from a racing reader —
/// the score is a heuristic, not an invariant.
#[inline]
fn load_heat_f64(a: &AtomicU64) -> f64 {
    f64::from_bits(a.load(Ordering::Relaxed))
}
#[inline]
fn store_heat_f64(a: &AtomicU64, v: f64) {
    a.store(v.to_bits(), Ordering::Relaxed);
}

/// One resident expert: id + the bytes loaded from the SSD.
///
/// The optional Unified Tensor Header prefix is parsed **once** at
/// construction time (see [`ExpertResident::new`]); the resulting
/// `payload_offset` is cached so that [`Self::data`] — which sits on
/// the inference + `--io-only` hot paths — is a cheap subslice
/// operation with no re-parsing.
pub struct ExpertResident {
    pub id: u32,
    pub buffer: PooledBuffer,
    /// Byte offset within `buffer` at which the bare weight payload
    /// begins. `0` for legacy blobs and synthetic fixtures (no UTH);
    /// `UTH_BYTES + page padding` for `gguf-convert` blobs.
    payload_offset: usize,
    /// Parsed mixed-projection header, present only for UTH2 experts.
    mixed_layout: Option<MixedExpertHeader>,
    /// Monotonic hit counter (Phase 2 — three-tier memory hierarchy).
    ///
    /// Bumped by [`GpuExpertCache::observe_ram_hit`] / engine routing
    /// every time a RAM lookup resolves to this resident. Read by the
    /// promotion controller — once `hits >= promote_after_hits`, the
    /// expert becomes a candidate for logical GPU admission.
    ///
    /// Stored as an `AtomicU64` so the engine's lock-free routing hot
    /// path can update it with a single relaxed atomic increment.
    hits: AtomicU64,
    /// Cached once-per-resident Q4_0 zero-padded payload used when the
    /// on-disk bytes are slightly short (≤ one block/page) of the
    /// derived expected size.
    q4_0_padded: OnceCell<(usize, Arc<[u8]>)>,
    /// Test/benchmark-only Candle Q8_0 reference preparation. The default
    /// native path executes directly over `data()` and retains no duplicate.
    #[cfg(feature = "q8-candle-reference")]
    q8_0_qmm: OnceCell<
        Result<crate::inference::PreparedQ8_0Expert, crate::inference::ExpertWeightsError>,
    >,
    /// **Tier 4 cost-aware eviction.** Decaying heat score: bumped by
    /// `+1` on every cache hit and exponentially decayed by the number
    /// of intervening insertions (cache-pressure events). Only
    /// maintained when the owning [`ExpertCache`] has cost-aware
    /// eviction enabled; otherwise it stays at its initial value and is
    /// never read. Stored as `f64` bits behind an `AtomicU64` so the
    /// lock-free hit path can update it.
    heat_bits: AtomicU64,
    /// Insertion epoch (a logical cache-pressure clock) at which this
    /// resident's heat was last refreshed. Paired with `heat_bits` to
    /// apply lazy exponential decay.
    heat_last_epoch: AtomicU64,
}

impl ExpertResident {
    /// Construct a resident expert, computing and caching the UTH
    /// payload offset once. Subsequent calls to [`Self::data`] do not
    /// re-probe the header.
    pub fn new(id: u32, buffer: PooledBuffer) -> Self {
        Self::new_with_block_align(id, buffer, DEFAULT_BLOCK_ALIGN)
    }

    /// Construct a resident expert using the block alignment declared
    /// by the storage metadata. This is required for UTH1/UTH2 payload
    /// offsets to be parsed correctly when a dataset was converted with
    /// an alignment other than 4096 bytes.
    pub fn new_with_block_align(id: u32, buffer: PooledBuffer, block_align: usize) -> Self {
        let (payload_offset, mixed_layout) = {
            let raw = buffer.as_slice();
            let (payload, mixed) =
                if let Some((h, payload)) = MixedExpertHeader::strip(raw, block_align) {
                    (payload, Some(h))
                } else {
                    let (_, payload) = TensorHeader::strip(raw, block_align);
                    (payload, None)
                };
            // `payload` is either `raw` unchanged (offset 0) or a suffix
            // subslice of it; derive the offset directly from the slice
            // lengths rather than via pointer arithmetic.
            let payload_offset = raw.len() - payload.len();
            debug_assert!(payload_offset <= raw.len());
            (payload_offset, mixed)
        };
        RESIDENT_EXPERT_BUFFER_BYTES.fetch_add(buffer.as_slice().len() as u64, Ordering::Relaxed);
        Self {
            id,
            buffer,
            payload_offset,
            mixed_layout,
            hits: AtomicU64::new(0),
            q4_0_padded: OnceCell::new(),
            #[cfg(feature = "q8-candle-reference")]
            q8_0_qmm: OnceCell::new(),
            heat_bits: AtomicU64::new(0.0f64.to_bits()),
            heat_last_epoch: AtomicU64::new(0),
        }
    }

    /// Parsed UTH2 layout for mixed-projection experts.
    #[inline]
    pub fn mixed_layout(&self) -> Option<MixedExpertHeader> {
        self.mixed_layout
    }

    /// Increment the resident's monotonic hit counter and return the
    /// new value. Used by the engine on every RAM hit to drive
    /// [`GpuExpertCache`] promotion decisions (Phase 2). Cheap: a
    /// single relaxed atomic FAA — safe to call from the lock-free
    /// inference hot path.
    #[inline]
    pub fn record_hit(&self) -> u64 {
        self.hits.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// **Tier 4 cost-aware eviction.** Refresh this resident's decaying
    /// heat score for an access at logical insertion-`epoch`: decay the
    /// stored heat by `decay^(epoch − last_epoch)`, add `1.0` for this
    /// access, and stamp `epoch`. `decay ∈ (0, 1]`; `epoch` is the
    /// owning cache's monotonic insertion counter, so an expert reused
    /// every few insertions keeps a high score while one untouched
    /// across many insertions fades toward zero. Approximate under
    /// concurrent access (heat is a heuristic), never a correctness
    /// hazard.
    #[inline]
    pub fn bump_heat(&self, epoch: u64, decay: f64) {
        let prev = self.heat_last_epoch.swap(epoch, Ordering::Relaxed);
        let dt = epoch.saturating_sub(prev).min(4096) as i32;
        let decayed = load_heat_f64(&self.heat_bits) * decay.powi(dt);
        store_heat_f64(&self.heat_bits, decayed + 1.0);
    }

    /// Current heat score decayed forward to `epoch` (read-only; does
    /// not mutate the stored score). Used by the cost-aware victim
    /// scorer to compare residents at a single point in logical time.
    #[inline]
    pub fn decayed_heat(&self, epoch: u64, decay: f64) -> f64 {
        let last = self.heat_last_epoch.load(Ordering::Relaxed);
        let dt = epoch.saturating_sub(last).min(4096) as i32;
        load_heat_f64(&self.heat_bits) * decay.powi(dt)
    }

    /// Whether this resident's bytes live in a **shadow** (Buffer B)
    /// pool buffer — i.e. it entered the cache via a speculative
    /// prefetch (`Engine::spawn_prefetch`) rather than a foreground
    /// miss. Used by [`ExpertCache::evict_lru_shadow_backed`] to
    /// recycle Buffer B capacity when every shadow slot is parked
    /// inside long-lived residents.
    #[inline]
    pub fn is_shadow_backed(&self) -> bool {
        self.buffer.is_shadow()
    }

    /// Bare weight bytes — i.e. the buffer with any leading Unified
    /// Tensor Header stripped. The vast majority of callers want this.
    ///
    /// O(1): uses the cached `payload_offset` computed in [`Self::new`],
    /// so the UTH is **not** reparsed on each call.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.buffer.as_slice()[self.payload_offset..]
    }

    /// Return a cached zero-padded Q4_0 payload when the resident is at
    /// most `tolerance` bytes short of `need`. The padded allocation is
    /// created at most once for this resident.
    pub fn q4_0_padded_payload(&self, need: usize, tolerance: usize) -> Option<Arc<[u8]>> {
        let data = self.data();
        if data.len() >= need {
            return None;
        }
        let shortfall = need - data.len();
        if need <= tolerance || shortfall > tolerance {
            return None;
        }

        let (cached_need, cached) = self.q4_0_padded.get_or_init(|| {
            let mut padded = Vec::with_capacity(need);
            padded.extend_from_slice(data);
            padded.resize(need, 0);
            (need, Arc::from(padded.into_boxed_slice()))
        });
        (*cached_need == need).then(|| cached.clone())
    }

    #[cfg(feature = "q8-candle-reference")]
    pub(crate) fn prepared_q8_0_qmm<F>(
        &self,
        prepare: F,
    ) -> (
        Result<&crate::inference::PreparedQ8_0Expert, crate::inference::ExpertWeightsError>,
        bool,
    )
    where
        F: FnOnce(&[u8]) -> Result<
            crate::inference::PreparedQ8_0Expert,
            crate::inference::ExpertWeightsError,
        >,
    {
        let mut prepared_now = false;
        let result = self.q8_0_qmm.get_or_init(|| {
            prepared_now = true;
            prepare(self.data())
        });
        let result = match result {
            Ok(prepared) => Ok(prepared),
            Err(err) => Err(err.clone()),
        };
        (result, prepared_now)
    }
}

impl Drop for ExpertResident {
    fn drop(&mut self) {
        RESIDENT_EXPERT_BUFFER_BYTES.fetch_sub(
            self.buffer.as_slice().len() as u64,
            Ordering::Relaxed,
        );
    }
}

/// Thread-safe fixed-capacity LRU cache of resident experts.
pub struct ExpertCache {
    inner: Mutex<LruCache<u32, Arc<ExpertResident>>>,
    /// Expert ids that are pinned and must never be returned by
    /// [`Self::evict_lru`]. Pinning is set by the engine after an
    /// expert has been observed enough times to be considered "hot"
    /// (see [`crate::engine::Engine`] / `pin_after_observations`).
    pinned: Mutex<HashSet<u32>>,
    capacity: usize,
    /// **Tier 4 — cost-aware eviction.** When `true`, [`Self::insert`]'s
    /// pre-eviction and [`Self::evict_lru`] choose the lowest decaying
    /// **heat** resident rather than the strict LRU victim, and
    /// [`Self::get`] maintains each resident's heat score on hit. When
    /// `false` (the default) the cache is a pure LRU and the heat
    /// machinery is completely inert, so legacy behaviour is preserved
    /// bit-for-bit. Interior-mutable so the engine can flip it on a
    /// shared `Arc<ExpertCache>` after construction.
    cost_aware: AtomicBool,
    /// Logical cache-pressure clock: incremented once per insertion.
    /// Drives the exponential decay of resident heat scores.
    epoch: AtomicU64,
    /// Cache positions owned by in-flight exact-demand production batches.
    /// Reserved positions count against `capacity` immediately, so ordinary
    /// inserts cannot consume them while the owning batch performs NVMe I/O.
    /// Every mutation is serialized with `inner`; the atomic representation
    /// also permits a cheap process-wide leak snapshot.
    reserved_slots: std::sync::atomic::AtomicUsize,
}

/// Why a production exact-demand cache reservation could not be created.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ExpertCacheReservationError {
    Capacity,
    VictimUnavailable,
    CostAwarePolicy,
    InvalidExpertId,
    DuplicateExpertId,
    ConcurrentReservationConflict,
}

/// RAII ownership of exact positions in one per-layer [`ExpertCache`].
///
/// The reservation evicts all required strict-LRU victims atomically at
/// creation time, but holds no mutex while device reads run. Successful
/// ordered commits consume one position each. Dropping the guard releases
/// every unused position, including cancellation and panic unwind.
pub(crate) struct ExpertCacheSlotReservation {
    cache: Arc<ExpertCache>,
    remaining: usize,
}

impl ExpertCacheSlotReservation {
    /// Create ownership for positions whose exact victims were already
    /// removed while the multi-layer transaction held every authoritative
    /// cache lock. Callers must have included existing reservations in the
    /// locked capacity proof before constructing this guard.
    pub(crate) fn new(cache: Arc<ExpertCache>, count: usize) -> Self {
        cache.reserved_slots.fetch_add(count, Ordering::Relaxed);
        Self {
            cache,
            remaining: count,
        }
    }

    pub(crate) fn remaining(&self) -> usize {
        self.remaining
    }

    /// Consume one reserved position without invoking the ordinary eviction
    /// path. An existing id or inconsistent occupancy is an invariant
    /// violation and leaves the reservation intact for RAII cleanup.
    pub(crate) fn commit(
        &mut self,
        resident: Arc<ExpertResident>,
    ) -> Result<(), Arc<ExpertResident>> {
        if self.remaining == 0 {
            return Err(resident);
        }
        let id = resident.id;
        let mut guard = self.cache.inner.lock();
        let reserved = self.cache.reserved_slots.load(Ordering::Relaxed);
        if reserved == 0
            || guard.peek(&id).is_some()
            || guard.len().saturating_add(reserved) > self.cache.capacity
        {
            return Err(resident);
        }
        self.cache.reserved_slots.fetch_sub(1, Ordering::Relaxed);
        self.remaining -= 1;
        let evicted = guard.push(id, resident);
        debug_assert!(
            evicted.is_none(),
            "reserved cache commit must not evict or replace a resident"
        );
        Ok(())
    }
}

impl Drop for ExpertCacheSlotReservation {
    fn drop(&mut self) {
        if self.remaining == 0 {
            return;
        }
        // Serialize release with ordinary insertion so no insert can observe
        // a partially released reservation or steal a still-owned position.
        let _guard = self.cache.inner.lock();
        let previous = self
            .cache
            .reserved_slots
            .fetch_sub(self.remaining, Ordering::Relaxed);
        debug_assert!(previous >= self.remaining, "cache reservation underflow");
        self.remaining = 0;
    }
}

/// Per-insertion decay factor applied to resident heat scores in
/// cost-aware mode. `0.98` gives heat a half-life of ~34 insertions, so
/// an expert needs to keep getting hit to stay resident — long enough to
/// ride out a brief lull, short enough to release genuinely cold experts.
const COST_AWARE_HEAT_DECAY: f64 = 0.98;

impl ExpertCache {
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity).expect("cache capacity must be > 0");
        Self {
            inner: Mutex::new(LruCache::new(cap)),
            pinned: Mutex::new(HashSet::new()),
            capacity,
            cost_aware: AtomicBool::new(false),
            epoch: AtomicU64::new(0),
            reserved_slots: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Enable or disable **Tier 4 cost-aware eviction** on this cache.
    /// Cheap and idempotent; safe to call on a shared `Arc<ExpertCache>`
    /// at startup. No-op effect on the hot path while `false`.
    pub fn set_cost_aware(&self, on: bool) {
        // Serialize policy changes with cache-wide exact-demand planning.
        // Startup is the normal call site, so these uncontended locks do not
        // add work to the serving hot path.
        let _pinned = self.pinned.lock();
        let _inner = self.inner.lock();
        self.cost_aware.store(on, Ordering::Relaxed);
    }

    /// Whether cost-aware eviction is currently enabled.
    #[inline]
    pub fn is_cost_aware(&self) -> bool {
        self.cost_aware.load(Ordering::Relaxed)
    }

    /// Look up an expert. Updates LRU recency on hit, and (in cost-aware
    /// mode) refreshes the resident's decaying heat score.
    pub fn get(&self, id: u32) -> Option<Arc<ExpertResident>> {
        let resident = self.inner.lock().get(&id).cloned();
        if let Some(r) = resident.as_ref() {
            if self.is_cost_aware() {
                let epoch = self.epoch.load(Ordering::Relaxed);
                r.bump_heat(epoch, COST_AWARE_HEAT_DECAY);
            }
        }
        resident
    }

    /// Peek without changing recency. Useful for the predictive loader to
    /// check residency without polluting the LRU order.
    pub fn contains(&self, id: u32) -> bool {
        self.inner.lock().peek(&id).is_some()
    }

    /// Insert a resident expert.
    ///
    /// Returns `Ok(Some(evicted))` when an entry was evicted to make
    /// room (so the caller can observe / log the eviction), `Ok(None)`
    /// when the entry was inserted without displacing anything, and
    /// `Err(resident)` when the cache is full and **every** resident
    /// expert is pinned. The error case hands the original `Arc` back
    /// to the caller so its `PooledBuffer` can return to the pool —
    /// the alternative (silently calling `LruCache::push`, which
    /// would evict a pinned entry) would break the pinning contract.
    pub fn insert(
        &self,
        resident: Arc<ExpertResident>,
    ) -> Result<Option<Arc<ExpertResident>>, Arc<ExpertResident>> {
        let id = resident.id;
        // Lock order: `pinned` before `inner` (matches `evict_lru`).
        // The capacity check, pre-eviction and `push` must form a
        // single critical section: releasing the lock in between
        // would let another thread fill the cache and `push` would
        // then silently evict the LRU entry — which may be pinned.
        let pinned = self.pinned.lock();
        let mut guard = self.inner.lock();
        // Tier 4: every insertion is one tick of the cache-pressure
        // clock that ages resident heat scores. Cheap; only meaningful
        // when cost-aware mode is enabled.
        let cost_aware = self.is_cost_aware();
        let epoch = if cost_aware {
            self.epoch.fetch_add(1, Ordering::Relaxed) + 1
        } else {
            0
        };
        let mut pre_evicted = None;
        let effective_len = guard.len()
            .saturating_add(self.reserved_slots.load(Ordering::Relaxed));
        if effective_len >= self.capacity && guard.peek(&id).is_none() {
            // Pick the victim under the active policy: strict LRU by
            // default, or the lowest decaying-heat resident in
            // cost-aware mode.
            match self.select_victim_id(&guard, &pinned) {
                Some(victim) => pre_evicted = guard.pop(&victim),
                None => {
                    // Cache is full *and* every resident expert is
                    // pinned. We must refuse the insert: calling `push`
                    // here would evict a pinned id (LruCache has no
                    // pinning concept).
                    return Err(resident);
                }
            }
        }
        // Tier 4: seed the freshly-loaded resident's heat for *this*
        // access so it isn't the instant next eviction victim (which
        // would thrash the very expert we just paid an SSD read for).
        if cost_aware {
            resident.bump_heat(epoch, COST_AWARE_HEAT_DECAY);
        }
        // `LruCache::push` returns the (k, v) pair that was evicted, if any.
        // With the pre-eviction above we never hit a second eviction
        // path here, but `push` on an existing key returns the old
        // value — which is fine to surface as "evicted" too.
        let push_evicted = guard.push(id, resident).map(|(_, v)| v);
        Ok(push_evicted.or(pre_evicted))
    }

    pub(crate) fn reserved_slots(&self) -> usize {
        self.reserved_slots.load(Ordering::Relaxed)
    }

    /// Cache-wide reservation planning locks every layer's pin set first and
    /// then every LRU in ascending layer order. These accessors keep the
    /// underlying fields private while allowing that single transaction to
    /// hold all authoritative state at once.
    pub(crate) fn lock_pinned(&self) -> parking_lot::MutexGuard<'_, HashSet<u32>> {
        self.pinned.lock()
    }

    pub(crate) fn lock_inner(
        &self,
    ) -> parking_lot::MutexGuard<'_, LruCache<u32, Arc<ExpertResident>>> {
        self.inner.lock()
    }

    /// Choose the id to evict from `guard` under the active policy.
    /// Returns the least-recently-used non-pinned id by default, or the
    /// resident with the lowest decaying **heat** in cost-aware mode.
    /// Ties resolve toward the more-LRU candidate. `None` when every
    /// resident is pinned.
    fn select_victim_id(
        &self,
        guard: &LruCache<u32, Arc<ExpertResident>>,
        pinned: &HashSet<u32>,
    ) -> Option<u32> {
        if self.is_cost_aware() {
            let epoch = self.epoch.load(Ordering::Relaxed);
            // `iter()` yields most-recently-used first, so iterating in
            // order and replacing on `score <= best` makes the more-LRU
            // candidate win heat ties.
            let mut best: Option<(u32, f64)> = None;
            for (k, v) in guard.iter() {
                if pinned.contains(k) {
                    continue;
                }
                let score = v.decayed_heat(epoch, COST_AWARE_HEAT_DECAY);
                let replace = match best {
                    None => true,
                    Some((_, bs)) => score <= bs,
                };
                if replace {
                    best = Some((*k, score));
                }
            }
            best.map(|(k, _)| k)
        } else {
            // Strict LRU: `iter()` is MRU-first, so the last non-pinned
            // id is the least-recently-used non-pinned victim.
            guard
                .iter()
                .map(|(k, _)| *k)
                .filter(|k| !pinned.contains(k))
                .last()
        }
    }

    /// Number of resident experts currently in the cache.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Pop the least-recently-used **non-pinned** entry. Returns the
    /// removed `Arc` so callers can observe (and log) what was evicted;
    /// once the `Arc` is dropped its `PooledBuffer` returns to the
    /// pool's free list. Pinned experts (see [`Self::pin`]) are
    /// skipped — if every resident expert is pinned this returns
    /// `None`, meaning there is no room to evict.
    pub fn evict_lru(&self) -> Option<Arc<ExpertResident>> {
        let pinned = self.pinned.lock();
        if pinned.is_empty() && !self.is_cost_aware() {
            // Fast path: no pinning and pure LRU, just pop LRU.
            return self.inner.lock().pop_lru().map(|(_, v)| v);
        }
        // Otherwise defer to the policy-aware victim selector (strict
        // LRU, or lowest decaying heat in cost-aware mode), skipping
        // pinned residents.
        let mut guard = self.inner.lock();
        let victim = self.select_victim_id(&guard, &pinned)?;
        guard.pop(&victim)
    }

    /// Pop the least-recently-used entry that is **shadow-backed**
    /// (see [`ExpertResident::is_shadow_backed`]) and not pinned.
    /// Returns `None` when no such resident exists.
    ///
    /// Used by the engine when the shadow (Buffer B) free list is
    /// empty: prefetched residents keep their shadow tag for the life
    /// of their residency, so once `shadow_slots` of them accumulate
    /// in the LRU every further speculative prefetch would be dropped
    /// ("shadow pool busy") until ordinary eviction happens to recycle
    /// one. Evicting the LRU shadow-backed resident hands its buffer
    /// back to Buffer B so the look-ahead pipeline keeps running.
    pub fn evict_lru_shadow_backed(&self) -> Option<Arc<ExpertResident>> {
        let pinned = self.pinned.lock();
        let mut guard = self.inner.lock();
        // `LruCache::iter` yields most-recently-used first, so walk the
        // collected order in reverse to test the LRU end first.
        let id_order: Vec<u32> = guard
            .iter()
            .filter_map(|(k, v)| if v.is_shadow_backed() { Some(*k) } else { None })
            .collect();
        for &id in id_order.iter().rev() {
            if !pinned.contains(&id) {
                if let Some(v) = guard.pop(&id) {
                    return Some(v);
                }
            }
        }
        None
    }

    /// Pin an expert id so it is never returned by [`Self::evict_lru`].
    /// If the id isn't currently resident this still records the pin —
    /// when the expert is later loaded it will be protected from
    /// eviction.
    pub fn pin(&self, id: u32) {
        self.pinned.lock().insert(id);
    }

    /// Remove a pin previously installed by [`Self::pin`].
    #[allow(dead_code)]
    pub fn unpin(&self, id: u32) {
        self.pinned.lock().remove(&id);
    }

    /// Whether `id` is currently pinned.
    #[allow(dead_code)]
    pub fn is_pinned(&self, id: u32) -> bool {
        self.pinned.lock().contains(&id)
    }

    /// Snapshot of currently-pinned ids (for diagnostics / metrics).
    #[allow(dead_code)]
    pub fn pinned_ids(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.pinned.lock().iter().copied().collect();
        v.sort_unstable();
        v
    }

    /// Number of currently-pinned ids.
    pub fn pinned_count(&self) -> usize {
        self.pinned.lock().len()
    }

    /// Snapshot of current residency (for logs/diagnostics).
    pub fn resident_ids(&self) -> Vec<u32> {
        self.inner.lock().iter().map(|(k, _)| *k).collect()
    }
}

// =====================================================================
// Phase 2 — logical GPU expert admission: Segmented Hybrid Policy.
// =====================================================================

/// One host-side logical GPU admission. The payload remains a `Vec<u8>` in
/// process memory; physical routed-expert device buffers are owned and
/// accounted by `GpuBackend`'s physical registry.
pub struct GpuResident {
    pub id: u32,
    /// Host payload retained by the logical admission policy.
    bytes: Vec<u8>,
    /// On-disk encoding of `bytes`. `F32` residents feed the dense
    /// matmul pipeline; `Q4_0` residents stay in native GGUF blocks
    /// and feed the inline-dequant pipeline (`matmul_q4_0.wgsl`) —
    /// see `GpuBackend::expert_matmul`.
    dtype: crate::inference::WeightDtype,
}

impl GpuResident {
    pub fn new(id: u32, bytes: Vec<u8>) -> Self {
        Self {
            id,
            bytes,
            dtype: crate::inference::WeightDtype::F32,
        }
    }

    /// Like [`GpuResident::new`] but tagging the bytes with their
    /// native on-disk dtype, so the GPU backend can pick the matching
    /// matmul pipeline (e.g. Q4_0 inline dequant) without guessing
    /// from the byte length.
    pub fn new_with_dtype(id: u32, bytes: Vec<u8>, dtype: crate::inference::WeightDtype) -> Self {
        Self { id, bytes, dtype }
    }

    /// Bare weight bytes ready for `run_inference_*`.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.bytes
    }

    /// Native encoding of [`GpuResident::data`].
    #[inline]
    pub fn dtype(&self) -> crate::inference::WeightDtype {
        self.dtype
    }

    /// Size of the logical host payload admitted for future GPU upload.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
}

impl crate::backend::GpuStorage for GpuResident {
    fn byte_len(&self) -> usize {
        self.bytes.len()
    }
    fn as_wgpu_buffer(&self) -> Option<&wgpu::Buffer> {
        None // GpuResident is host-side only; VRAM lives in VramExpertEntry
    }
}

/// One installed logical GPU admission. Generations are allocated by
/// [`GpuExpertCache`] and change only when an id is evicted and admitted
/// again; ordinary hits preserve the generation.
#[derive(Clone)]
pub struct GpuAdmission {
    resident: Arc<GpuResident>,
    generation: u64,
}

impl GpuAdmission {
    #[inline]
    pub fn resident(&self) -> &Arc<GpuResident> {
        &self.resident
    }

    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        self.resident.byte_len()
    }
}

/// Outcome of a logical GPU-admission lookup. The variants double as the
/// instrumentation discriminator for `mer_gpu_cache_hits_total` and
/// the engine's three-tier reporting in `/v1/admin/health/experts`.
pub enum GpuLookup {
    /// Hit on the **Anchor Core** — high-frequency, permanently
    /// pinned expert. No LRU recency update.
    AnchorHit(GpuAdmission),
    /// Hit on the **LRU Edge** — temporal locality. Recency updated.
    LruHit(GpuAdmission),
    /// Miss. Caller falls through to the RAM tier.
    Miss,
}

impl GpuLookup {
    pub fn is_hit(&self) -> bool {
        !matches!(self, GpuLookup::Miss)
    }
}

/// A selected foreground expert could not be installed in the logical GPU
/// admission LRU. The engine maps this to its typed routed-GPU failure policy;
/// this cache never performs a CPU fallback itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuDemandAdmissionError {
    EmptyPayload,
    PayloadExceedsLruCapacity {
        bytes: usize,
        capacity: usize,
    },
    DuplicateDemandExpert {
        expert_id: u32,
    },
    DemandSourceIdentityMismatch {
        selected_id: u32,
        source_id: u32,
    },
    DemandSetExceedsLruCapacity {
        required_bytes: usize,
        capacity: usize,
    },
    ProtectedDemandCapacity {
        required_bytes: usize,
        protected_bytes: usize,
        capacity: usize,
    },
    GenerationExhausted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuDemandAdmissionPreflight {
    AlreadyAdmitted,
    NeedsPayload,
}

/// Atomic result of resolving one complete foreground selected-expert set.
pub enum GpuDemandSetAdmission {
    /// Every selected id has a current logical admission. Existing identities
    /// are retained and `newly_admitted` counts only genuinely new identities.
    Ready {
        admissions: Vec<GpuAdmission>,
        newly_admitted: usize,
    },
    /// The cache needs owned payloads for these currently absent selected ids.
    /// No logical state was changed.
    PayloadRequired(Vec<u32>),
}

/// Precise result of one threshold-driven logical GPU promotion attempt.
///
/// `MovedLruToAnchor`, `InstalledAnchor`, and `InstalledLru` are the only
/// outcomes that change logical cache state and therefore the only outcomes
/// counted by `GpuExpertCache::promotions` and `mer_promotions_total`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuHotPromotionOutcome {
    /// The existing admission was moved intact from LRU Edge to Anchor Core.
    MovedLruToAnchor,
    /// A previously absent expert was installed directly in Anchor Core.
    InstalledAnchor,
    /// A previously absent expert was installed in LRU Edge because Anchor
    /// Core lacked room.
    InstalledLru,
    /// The expert was already in Anchor Core; no state changed.
    AlreadyAnchor,
    /// The expert remains in LRU Edge because Anchor Core lacks room.
    AlreadyLruAnchorFull,
    /// No logical admission exists yet, so the caller must obtain an owned
    /// host payload before completing the hot promotion.
    PayloadRequired,
    /// The payload cannot fit either eligible logical region.
    NoCapacity,
    /// A new admission could not allocate a unique logical generation.
    GenerationExhausted,
}

impl GpuHotPromotionOutcome {
    /// Whether this outcome performed exactly one logical cache transition.
    pub fn is_transition(self) -> bool {
        matches!(
            self,
            Self::MovedLruToAnchor | Self::InstalledAnchor | Self::InstalledLru
        )
    }
}

impl std::fmt::Display for GpuDemandAdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPayload => f.write_str("expert payload is empty"),
            Self::PayloadExceedsLruCapacity { bytes, capacity } => write!(
                f,
                "expert payload {bytes} bytes exceeds logical GPU demand LRU capacity {capacity} bytes"
            ),
            Self::DuplicateDemandExpert { expert_id } => {
                write!(f, "logical GPU demand set repeats expert {expert_id}")
            }
            Self::DemandSourceIdentityMismatch {
                selected_id,
                source_id,
            } => write!(
                f,
                "logical GPU demand source expert {source_id} does not match selected expert {selected_id}"
            ),
            Self::DemandSetExceedsLruCapacity {
                required_bytes,
                capacity,
            } => write!(
                f,
                "selected logical GPU demand set requires {required_bytes} LRU bytes but capacity is {capacity} bytes"
            ),
            Self::ProtectedDemandCapacity {
                required_bytes,
                protected_bytes,
                capacity,
            } => write!(
                f,
                "logical GPU demand requires {required_bytes} bytes but {protected_bytes} bytes are protected within {capacity}-byte LRU capacity"
            ),
            Self::GenerationExhausted => {
                f.write_str("logical GPU admission generation space exhausted")
            }
        }
    }
}

impl std::error::Error for GpuDemandAdmissionError {}

/// Thread-safe logical GPU-admission cache implementing the **Segmented
/// Hybrid Policy** from the Phase 2 spec:
///
/// * **Anchor Core** — `HashMap<u32, GpuAdmission>` for experts
///   that have crossed `promote_after_hits`. Pinned, never evicted.
///   Sized by `anchor_ratio * capacity_bytes`.
/// * **LRU Edge** — `LruCache<u32, GpuAdmission>` for temporal
///   topic shifts. O(1) recency tracking, byte-budgeted evictions.
///
/// Concurrency contract (gist "Zero-Contention" critical constraint):
///
/// * All cache-state updates go through a single `parking_lot::Mutex`
///   wrapping the `Inner` struct. The critical section is just the
///   HashMap / LRU manipulation — never any I/O, never any compute.
/// * Hit counters on individual `ExpertResident`s are
///   [`AtomicU64`](std::sync::atomic::AtomicU64); the inference hot
///   path bumps them lock-free.
/// * The compatibility `mer_vram_used_bytes` gauge is logical admitted host
///   payload bytes, not physical wgpu allocation bytes.
pub struct GpuExpertCache {
    inner: Mutex<GpuExpertCacheInner>,
    /// Logical host-payload capacity of the **Anchor Core**, in bytes.
    anchor_capacity_bytes: usize,
    /// Capacity of the **LRU Edge**, in bytes.
    lru_capacity_bytes: usize,
    /// Promotion threshold copied out of `[gpu_cache].promote_after_hits`.
    /// `0` disables Anchor Core promotions (everything routes to the
    /// LRU Edge).
    promote_after_hits: u64,
    /// Total successful logical promotion transitions since startup: new
    /// Anchor/LRU admissions plus LRU-to-Anchor graduation. Mirror of the
    /// `mer_promotions_total` Prometheus counter; exposed here too so the
    /// admin health endpoint can render the value without going through the
    /// Prometheus registry.
    promotions: AtomicU64,
    /// Logical host payload bytes admitted across Anchor + LRU.
    logical_admitted_bytes: AtomicU64,
    /// Cumulative logical GPU-admission hits — mirrors the
    /// `mer_gpu_cache_hits_total` Prometheus counter.
    hits: AtomicU64,
    /// Cumulative logical GPU-admission misses — mirrors
    /// `mer_gpu_cache_misses_total`.
    misses: AtomicU64,
}

/// RAII protection for one complete foreground selected-expert set.
///
/// The guard records ids in cache metadata but never retains the cache mutex,
/// so it is safe to hold across async RAM/NVMe acquisition. Every logical LRU
/// eviction path consults this registry. Dropping the guard releases only this
/// transaction's references, allowing overlapping demand sets to share ids.
pub struct GpuDemandSetProtection {
    cache: Arc<GpuExpertCache>,
    expert_ids: Vec<u32>,
}

impl Drop for GpuDemandSetProtection {
    fn drop(&mut self) {
        let mut g = self.cache.inner.lock();
        for expert_id in &self.expert_ids {
            let remove = {
                let count = g
                    .demand_protections
                    .get_mut(expert_id)
                    .expect("foreground demand protection must remain registered");
                *count -= 1;
                *count == 0
            };
            if remove {
                g.demand_protections.remove(expert_id);
            }
        }
    }
}

struct GpuExpertCacheInner {
    /// **Anchor Core** — permanently pinned high-frequency experts.
    anchor: HashMap<u32, GpuAdmission>,
    anchor_used_bytes: usize,
    /// **LRU Edge** — temporal locality region.
    lru: LruCache<u32, GpuAdmission>,
    lru_used_bytes: usize,
    /// Next logical admission generation. Exhaustion cleanly refuses a new
    /// admission rather than reusing an identity.
    next_generation: u64,
    /// Ids with one threshold-triggered promotion request in flight. This is
    /// cleared when promotion completes or enqueueing fails, so eviction can
    /// retrigger exactly one request even when the RAM hit count is already
    /// above the threshold.
    promotion_pending: HashSet<u32>,
    /// Logically evicted ids that may claim one new promotion attempt without
    /// a fresh hit-count edge. Consuming this marker prevents a failed or
    /// oversized promotion from retrying on every later RAM hit.
    promotion_rearm: HashSet<u32>,
    /// Reference-counted selected ids protected by in-flight foreground
    /// demand-set transactions. This is metadata only; no lock is retained
    /// while a transaction performs async source acquisition.
    demand_protections: HashMap<u32, usize>,
}

impl GpuExpertCacheInner {
    fn allocate_generation(&mut self) -> Option<u64> {
        let generation = self.next_generation;
        self.next_generation = generation.checked_add(1)?;
        Some(generation)
    }
}

fn logical_lru_is_protected(
    inner: &GpuExpertCacheInner,
    expert_id: u32,
    additionally_protected: &HashSet<u32>,
) -> bool {
    additionally_protected.contains(&expert_id)
        || inner
            .demand_protections
            .get(&expert_id)
            .is_some_and(|count| *count > 0)
}

fn oldest_unprotected_logical_lru(
    inner: &GpuExpertCacheInner,
    additionally_protected: &HashSet<u32>,
) -> Option<u32> {
    inner.lru.iter().rev().find_map(|(&expert_id, _)| {
        (!logical_lru_is_protected(inner, expert_id, additionally_protected)).then_some(expert_id)
    })
}

fn unprotected_logical_lru_bytes(
    inner: &GpuExpertCacheInner,
    additionally_protected: &HashSet<u32>,
) -> usize {
    inner
        .lru
        .iter()
        .filter_map(|(&expert_id, admission)| {
            (!logical_lru_is_protected(inner, expert_id, additionally_protected))
                .then_some(admission.byte_len())
        })
        .sum()
}

impl GpuExpertCache {
    /// Construct a new logical GPU-admission cache.
    ///
    /// * `capacity_bytes` — logical host-admission budget (anchor + LRU),
    ///   also passed to the backend as its physical expert-weight cap.
    /// * `anchor_ratio` — fraction of `capacity_bytes` reserved for
    ///   the Anchor Core. Clamped to `[0.0, 1.0]`.
    /// * `promote_after_hits` — threshold for RAM → GPU admission.
    ///   `0` disables Anchor Core promotion.
    pub fn new(capacity_bytes: usize, anchor_ratio: f32, promote_after_hits: u64) -> Self {
        let ratio = anchor_ratio.clamp(0.0, 1.0);
        let anchor_capacity_bytes = ((capacity_bytes as f32) * ratio) as usize;
        let lru_capacity_bytes = capacity_bytes.saturating_sub(anchor_capacity_bytes);
        // `LruCache` requires a non-zero entry count even when the
        // bytes budget would naturally allow zero. Use `unbounded()`
        // so eviction is driven solely by the byte-budget check
        // below — passing a sentinel like `usize::MAX` to `new()`
        // makes the underlying hashbrown allocator overflow.
        Self {
            inner: Mutex::new(GpuExpertCacheInner {
                anchor: HashMap::new(),
                anchor_used_bytes: 0,
                lru: LruCache::unbounded(),
                lru_used_bytes: 0,
                next_generation: 1,
                promotion_pending: HashSet::new(),
                promotion_rearm: HashSet::new(),
                demand_protections: HashMap::new(),
            }),
            anchor_capacity_bytes,
            lru_capacity_bytes,
            promote_after_hits,
            promotions: AtomicU64::new(0),
            logical_admitted_bytes: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Logical admission budget (anchor + LRU host payloads), in bytes. The
    /// same configured value is also the physical expert-weight capacity.
    #[inline]
    pub fn capacity_bytes(&self) -> usize {
        self.anchor_capacity_bytes + self.lru_capacity_bytes
    }

    /// Logical admitted host payload bytes (anchor + LRU). Retained for API
    /// compatibility; physical wgpu bytes come from the backend ledger.
    #[inline]
    pub fn used_bytes(&self) -> u64 {
        self.logical_admitted_bytes.load(Ordering::Relaxed)
    }

    /// Cumulative RAM → logical GPU-admission promotions.
    #[inline]
    pub fn promotions(&self) -> u64 {
        self.promotions.load(Ordering::Relaxed)
    }

    /// Cumulative logical GPU-admission hits.
    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Cumulative logical GPU-admission misses.
    #[inline]
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Look up a logical GPU admission. Returns the [`GpuLookup`] discriminator
    /// (anchor / LRU / miss) plus the resident handle on hit.
    ///
    /// **LRU Edge** hits update recency; **Anchor Core** hits do not
    /// (anchored experts are permanently hot by definition).
    pub fn get(&self, id: u32) -> GpuLookup {
        let mut g = self.inner.lock();
        if let Some(r) = g.anchor.get(&id).cloned() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return GpuLookup::AnchorHit(r);
        }
        // `LruCache::get` updates recency; that's what we want for
        // the LRU Edge.
        if let Some(r) = g.lru.get(&id).cloned() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return GpuLookup::LruHit(r);
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        GpuLookup::Miss
    }

    /// Clone the current logical admission without recording a routing probe
    /// or changing logical LRU recency. Physical backends use this to validate
    /// identity after routing has already performed the authoritative
    /// telemetry/recency lookup.
    pub fn current_admission(&self, id: u32) -> Option<GpuAdmission> {
        let g = self.inner.lock();
        g.anchor
            .get(&id)
            .or_else(|| g.lru.peek(&id))
            .cloned()
    }

    /// Check whether an expert is currently resident in either the
    /// anchor or LRU regions, without mutating recency or counters.
    pub fn contains(&self, id: u32) -> bool {
        let g = self.inner.lock();
        g.anchor.contains_key(&id) || g.lru.peek(&id).is_some()
    }

    /// Current admission generation without changing recency or counters.
    pub fn current_generation(&self, id: u32) -> Option<u64> {
        let g = self.inner.lock();
        g.anchor
            .get(&id)
            .or_else(|| g.lru.peek(&id))
            .map(GpuAdmission::generation)
    }

    /// O(1) identity validation used by the physical registry after upload.
    pub fn contains_generation(&self, id: u32, generation: u64) -> bool {
        self.current_generation(id) == Some(generation)
    }

    /// Should the resident's current hit count promote it to the
    /// Anchor Core? Cheap relaxed-atomic compare against the
    /// configured threshold; safe to call from the hot path before
    /// kicking off an async promotion.
    #[inline]
    pub fn should_promote(&self, ram_hits: u64) -> bool {
        self.promote_after_hits > 0 && ram_hits >= self.promote_after_hits
    }

    /// Claim the single outstanding promotion request for an id. Unlike a
    /// hit-count edge alone, this becomes claimable again after logical
    /// eviction, even when the RAM resident's monotonic hit count is already
    /// above the threshold.
    pub fn claim_promotion(&self, id: u32, ram_hits: u64) -> bool {
        if !self.should_promote(ram_hits) {
            return false;
        }
        let mut g = self.inner.lock();
        // An LRU-resident expert may still need to graduate into Anchor Core.
        // LRU residency is deliberately not a rejection condition: the
        // ordinary threshold / rearm rules below still ensure at most one
        // queued request, without touching LRU recency.
        if g.anchor.contains_key(&id) || g.promotion_pending.contains(&id) {
            return false;
        }
        let threshold_crossed = ram_hits == self.promote_after_hits;
        let rearmed_after_eviction = g.promotion_rearm.remove(&id);
        if !threshold_crossed && !rearmed_after_eviction {
            return false;
        }
        g.promotion_pending.insert(id)
    }

    /// Release a promotion claim when channel enqueueing fails.
    pub fn cancel_promotion(&self, id: u32) {
        let mut g = self.inner.lock();
        if g.promotion_pending.remove(&id) {
            g.promotion_rearm.insert(id);
        }
    }

    /// Resolve an already-admitted threshold-hot expert without copying a RAM
    /// payload. An LRU admission is moved atomically into Anchor Core when
    /// capacity permits; the exact [`GpuAdmission`] (including generation and
    /// `Arc<GpuResident>`) is retained. No unrelated LRU recency is touched.
    ///
    /// [`GpuHotPromotionOutcome::PayloadRequired`] means the id is absent and
    /// the caller may copy the RAM payload outside the cache mutex before
    /// calling [`Self::promote_hot_sync`].
    pub fn promote_hot_existing(&self, id: u32) -> GpuHotPromotionOutcome {
        let mut g = self.inner.lock();
        let outcome = match self.resolve_existing_hot_locked(&mut g, id) {
            Some(outcome) => outcome,
            None => return GpuHotPromotionOutcome::PayloadRequired,
        };
        drop(g);
        self.record_hot_transition(outcome);
        outcome
    }

    /// Complete a threshold-driven hot promotion after the caller obtained an
    /// owned payload. The locked existing-admission recheck makes the method
    /// race-safe: if foreground demand installed the id in LRU after the
    /// copy-free probe, that exact admission is moved to Anchor and this
    /// redundant `resident` is dropped without replacing its generation.
    pub fn promote_hot_sync(&self, resident: Arc<GpuResident>) -> GpuHotPromotionOutcome {
        let id = resident.id;
        let bytes = resident.byte_len();
        let mut g = self.inner.lock();

        if let Some(outcome) = self.resolve_existing_hot_locked(&mut g, id) {
            drop(g);
            self.record_hot_transition(outcome);
            return outcome;
        }

        if bytes == 0 {
            g.promotion_pending.remove(&id);
            g.promotion_rearm.remove(&id);
            return GpuHotPromotionOutcome::NoCapacity;
        }

        let use_anchor = bytes <= self.anchor_capacity_bytes
            && g.anchor_used_bytes
                .checked_add(bytes)
                .is_some_and(|used| used <= self.anchor_capacity_bytes);
        if !use_anchor && bytes > self.lru_capacity_bytes {
            g.promotion_pending.remove(&id);
            g.promotion_rearm.remove(&id);
            return GpuHotPromotionOutcome::NoCapacity;
        }
        if !use_anchor {
            let additionally_protected = HashSet::new();
            let required_bytes = g.lru_used_bytes.saturating_add(bytes);
            let bytes_to_evict = required_bytes.saturating_sub(self.lru_capacity_bytes);
            if bytes_to_evict > unprotected_logical_lru_bytes(&g, &additionally_protected) {
                g.promotion_pending.remove(&id);
                g.promotion_rearm.remove(&id);
                return GpuHotPromotionOutcome::NoCapacity;
            }
        }
        let Some(generation) = g.allocate_generation() else {
            g.promotion_pending.remove(&id);
            g.promotion_rearm.remove(&id);
            return GpuHotPromotionOutcome::GenerationExhausted;
        };
        let admission = GpuAdmission { resident, generation };
        g.promotion_pending.remove(&id);
        g.promotion_rearm.remove(&id);

        let outcome = if use_anchor {
            g.anchor.insert(id, admission);
            g.anchor_used_bytes = g
                .anchor_used_bytes
                .checked_add(bytes)
                .expect("logical GPU anchor byte overflow after capacity check");
            GpuHotPromotionOutcome::InstalledAnchor
        } else {
            let additionally_protected = HashSet::new();
            while g
                .lru_used_bytes
                .checked_add(bytes)
                .is_none_or(|used| used > self.lru_capacity_bytes)
            {
                let evicted_id = oldest_unprotected_logical_lru(&g, &additionally_protected)
                    .expect("hot capacity preflight proved an unprotected logical victim");
                let victim = g
                    .lru
                    .pop(&evicted_id)
                    .expect("selected hot logical victim remained under cache lock");
                g.lru_used_bytes = g
                    .lru_used_bytes
                    .checked_sub(victim.byte_len())
                    .expect("logical GPU LRU byte underflow during hot admission");
                g.promotion_rearm.insert(evicted_id);
            }
            let previous = g.lru.put(id, admission);
            debug_assert!(previous.is_none(), "hot admission rechecked under lock");
            g.lru_used_bytes = g
                .lru_used_bytes
                .checked_add(bytes)
                .expect("logical GPU LRU byte overflow after capacity check");
            GpuHotPromotionOutcome::InstalledLru
        };
        drop(g);

        self.record_hot_transition(outcome);
        self.refresh_used_bytes();
        outcome
    }

    /// Resolve Anchor/LRU state under the caller's cache lock. `None` means
    /// the id is absent and a payload is required. Completed attempts always
    /// clear their pending/rearm markers; an Anchor-full LRU entry is not
    /// rearmed, preventing retry traffic on every later hit.
    fn resolve_existing_hot_locked(
        &self,
        g: &mut GpuExpertCacheInner,
        id: u32,
    ) -> Option<GpuHotPromotionOutcome> {
        if g.anchor.contains_key(&id) {
            g.promotion_pending.remove(&id);
            g.promotion_rearm.remove(&id);
            return Some(GpuHotPromotionOutcome::AlreadyAnchor);
        }

        let bytes = g.lru.peek(&id).map(GpuAdmission::byte_len)?;
        let anchor_has_room = bytes <= self.anchor_capacity_bytes
            && g.anchor_used_bytes
                .checked_add(bytes)
                .is_some_and(|used| used <= self.anchor_capacity_bytes);
        if !anchor_has_room {
            g.promotion_pending.remove(&id);
            g.promotion_rearm.remove(&id);
            return Some(GpuHotPromotionOutcome::AlreadyLruAnchorFull);
        }

        let admission = g
            .lru
            .pop(&id)
            .expect("peeked logical GPU LRU admission must remain under lock");
        g.lru_used_bytes = g
            .lru_used_bytes
            .checked_sub(bytes)
            .expect("logical GPU LRU byte underflow during Anchor graduation");
        let previous = g.anchor.insert(id, admission);
        debug_assert!(previous.is_none(), "Anchor presence checked under lock");
        g.anchor_used_bytes = g
            .anchor_used_bytes
            .checked_add(bytes)
            .expect("logical GPU anchor byte overflow after capacity check");
        g.promotion_pending.remove(&id);
        g.promotion_rearm.remove(&id);
        Some(GpuHotPromotionOutcome::MovedLruToAnchor)
    }

    fn record_hot_transition(&self, outcome: GpuHotPromotionOutcome) {
        if outcome.is_transition() {
            self.promotions.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Synchronous promotion entry point — copy a RAM resident's bytes into
    /// the host-side logical admission cache and place it in the Anchor Core
    /// if budget allows, otherwise in the LRU Edge. Physical upload remains
    /// lazy and is owned by `GpuBackend`.
    ///
    /// **Hot-path callers must not invoke this directly** — instead
    /// hand the resident off to the engine's background promotion
    /// task (see [`crate::engine::Engine`]). The synchronous path
    /// exists for the warm-up sequence (where blocking is the
    /// expected behaviour) and for tests.
    ///
    /// Returns `true` when the expert was admitted, `false` if it
    /// could not fit even after eviction (e.g. payload exceeds the
    /// LRU budget entirely).
    pub fn promote_sync(&self, resident: Arc<GpuResident>) -> bool {
        let bytes = resident.byte_len();
        let mut g = self.inner.lock();
        g.promotion_pending.remove(&resident.id);
        g.promotion_rearm.remove(&resident.id);
        if bytes == 0 {
            return false;
        }
        // Already resident: nothing to promote. Touch the LRU entry so
        // it becomes MRU, but don't count this as a new promotion nor
        // re-account bytes (the existing entry already owns them).
        if g.anchor.contains_key(&resident.id) {
            return true;
        }
        if g.lru.get(&resident.id).is_some() {
            return true;
        }
        // Anchor first: if it fits in the anchor budget *and* the
        // engine flagged this expert as hot, install there. We treat
        // any explicit promote_sync as "hot" (the engine only calls
        // this after threshold), but still prefer Anchor only when
        // there's room without evicting another anchor entry.
        let use_anchor = bytes <= self.anchor_capacity_bytes
            && g.anchor_used_bytes
                .checked_add(bytes)
                .is_some_and(|used| used <= self.anchor_capacity_bytes);
        if !use_anchor && bytes > self.lru_capacity_bytes {
            return false;
        }
        if !use_anchor {
            let additionally_protected = HashSet::new();
            let required_bytes = g.lru_used_bytes.saturating_add(bytes);
            let bytes_to_evict = required_bytes.saturating_sub(self.lru_capacity_bytes);
            if bytes_to_evict > unprotected_logical_lru_bytes(&g, &additionally_protected) {
                return false;
            }
        }
        let Some(generation) = g.allocate_generation() else {
            return false;
        };
        let admission = GpuAdmission { resident, generation };
        if use_anchor {
            g.anchor.insert(admission.resident.id, admission);
            g.anchor_used_bytes = g
                .anchor_used_bytes
                .checked_add(bytes)
                .expect("logical GPU anchor byte overflow after capacity check");
            drop(g);
            self.promotions.fetch_add(1, Ordering::Relaxed);
            self.refresh_used_bytes();
            return true;
        }
        // Evict LRU entries until there is room. `LruCache::pop_lru`
        // returns the least-recently-used (k, v).
        let additionally_protected = HashSet::new();
        while g
            .lru_used_bytes
            .checked_add(bytes)
            .is_none_or(|used| used > self.lru_capacity_bytes)
        {
            let evicted_id = oldest_unprotected_logical_lru(&g, &additionally_protected)
                .expect("promotion capacity preflight proved an unprotected logical victim");
            let victim = g
                .lru
                .pop(&evicted_id)
                .expect("selected promotion victim remained under cache lock");
            g.lru_used_bytes = g.lru_used_bytes.saturating_sub(victim.byte_len());
            g.promotion_rearm.insert(evicted_id);
        }
        let already = g.lru.put(admission.resident.id, admission);
        if let Some(prev) = already {
            // Replacing an existing entry — subtract the old footprint.
            g.lru_used_bytes = g.lru_used_bytes.saturating_sub(prev.byte_len());
        }
        g.lru_used_bytes = g
            .lru_used_bytes
            .checked_add(bytes)
            .expect("logical GPU LRU byte overflow after capacity check");
        drop(g);
        self.promotions.fetch_add(1, Ordering::Relaxed);
        self.refresh_used_bytes();
        true
    }

    /// Non-evicting LRU-only promotion — install `resident` into the
    /// **LRU Edge** if and only if it fits in the remaining LRU byte
    /// budget without evicting any existing entry, and never place it
    /// in the Anchor Core.
    ///
    /// This is the warm-up counterpart to [`Self::promote_sync`]: the
    /// synchronous NVMe-miss path in the engine uses it to pin a freshly
    /// loaded expert in the logical admission LRU without (a) evicting threshold-promoted
    /// hot experts already resident in the LRU Edge, or (b) consuming
    /// Anchor Core slots that the hit-threshold policy reserves for
    /// genuinely hot experts. Anchor Core promotion remains the
    /// exclusive responsibility of the threshold-driven background
    /// promotion task wired up in
    /// [`crate::engine::Engine::install_gpu_cache`].
    ///
    /// Returns `true` when the expert was installed in the LRU Edge,
    /// `false` if it was already resident or would not fit without
    /// eviction.
    pub fn try_promote_lru_no_evict(&self, resident: Arc<GpuResident>) -> bool {
        let bytes = resident.byte_len();
        let mut g = self.inner.lock();
        if bytes == 0 {
            return false;
        }
        // Already resident anywhere: nothing to do. Don't touch LRU
        // recency — the caller is a warm-up path, not a real access. Any
        // promotion markers for this already-satisfied id are stale.
        if g.anchor.contains_key(&resident.id) || g.lru.peek(&resident.id).is_some() {
            g.promotion_pending.remove(&resident.id);
            g.promotion_rearm.remove(&resident.id);
            return false;
        }
        // Strictly non-evicting: must fit in whatever LRU budget is
        // currently free.
        if g
            .lru_used_bytes
            .checked_add(bytes)
            .is_none_or(|used| used > self.lru_capacity_bytes)
        {
            return false;
        }
        let Some(generation) = g.allocate_generation() else {
            return false;
        };
        let admission = GpuAdmission { resident, generation };
        g.promotion_pending.remove(&admission.resident.id);
        g.promotion_rearm.remove(&admission.resident.id);
        g.lru.put(admission.resident.id, admission);
        g.lru_used_bytes = g
            .lru_used_bytes
            .checked_add(bytes)
            .expect("logical GPU LRU byte overflow after capacity check");
        drop(g);
        self.promotions.fetch_add(1, Ordering::Relaxed);
        self.refresh_used_bytes();
        true
    }

    /// Protect one complete foreground selected-expert set from logical LRU
    /// eviction until the returned guard is dropped. Registration is atomic,
    /// but the guard holds no mutex and may cross async RAM/NVMe awaits.
    pub fn protect_demand_set(
        self: &Arc<Self>,
        expert_ids: &[u32],
    ) -> Result<GpuDemandSetProtection, GpuDemandAdmissionError> {
        let mut seen = HashSet::with_capacity(expert_ids.len());
        for &expert_id in expert_ids {
            if !seen.insert(expert_id) {
                return Err(GpuDemandAdmissionError::DuplicateDemandExpert { expert_id });
            }
        }

        let mut g = self.inner.lock();
        for &expert_id in expert_ids {
            let count = g.demand_protections.entry(expert_id).or_insert(0);
            *count = count
                .checked_add(1)
                .expect("foreground demand protection reference count overflowed");
        }
        drop(g);
        Ok(GpuDemandSetProtection {
            cache: self.clone(),
            expert_ids: expert_ids.to_vec(),
        })
    }

    /// Resolve a complete foreground selected-expert set as one logical
    /// transaction. Existing admissions retain their exact generation and
    /// payload. Missing admissions are installed together after evicting only
    /// non-selected, non-protected LRU entries. Every validation and capacity
    /// check completes before the first eviction, so failures are atomic.
    ///
    /// Callers first pass no `sources`. `PayloadRequired` identifies exactly
    /// which multi-megabyte payload copies are needed outside the cache mutex;
    /// a second call supplies only those copies and revalidates atomically.
    pub fn demand_admit_set(
        &self,
        expert_ids: &[u32],
        sources: &HashMap<u32, Arc<GpuResident>>,
    ) -> Result<GpuDemandSetAdmission, GpuDemandAdmissionError> {
        let mut selected = HashSet::with_capacity(expert_ids.len());
        for &expert_id in expert_ids {
            if !selected.insert(expert_id) {
                return Err(GpuDemandAdmissionError::DuplicateDemandExpert { expert_id });
            }
        }

        let mut g = self.inner.lock();
        let mut payload_required = Vec::new();
        let mut new_ids = Vec::new();
        let mut selected_lru_bytes = 0usize;
        let mut new_bytes = 0usize;

        for &expert_id in expert_ids {
            if g.anchor.contains_key(&expert_id) {
                continue;
            }
            if let Some(admission) = g.lru.peek(&expert_id) {
                selected_lru_bytes = selected_lru_bytes.checked_add(admission.byte_len()).ok_or(
                    GpuDemandAdmissionError::DemandSetExceedsLruCapacity {
                        required_bytes: usize::MAX,
                        capacity: self.lru_capacity_bytes,
                    },
                )?;
                continue;
            }

            let Some(source) = sources.get(&expert_id) else {
                payload_required.push(expert_id);
                continue;
            };
            if source.id != expert_id {
                return Err(GpuDemandAdmissionError::DemandSourceIdentityMismatch {
                    selected_id: expert_id,
                    source_id: source.id,
                });
            }
            let bytes = source.byte_len();
            if bytes == 0 {
                return Err(GpuDemandAdmissionError::EmptyPayload);
            }
            if bytes > self.lru_capacity_bytes {
                return Err(GpuDemandAdmissionError::PayloadExceedsLruCapacity {
                    bytes,
                    capacity: self.lru_capacity_bytes,
                });
            }
            selected_lru_bytes = selected_lru_bytes.checked_add(bytes).ok_or(
                GpuDemandAdmissionError::DemandSetExceedsLruCapacity {
                    required_bytes: usize::MAX,
                    capacity: self.lru_capacity_bytes,
                },
            )?;
            new_bytes = new_bytes.checked_add(bytes).ok_or(
                GpuDemandAdmissionError::DemandSetExceedsLruCapacity {
                    required_bytes: usize::MAX,
                    capacity: self.lru_capacity_bytes,
                },
            )?;
            new_ids.push(expert_id);
        }

        if !payload_required.is_empty() {
            return Ok(GpuDemandSetAdmission::PayloadRequired(payload_required));
        }
        if selected_lru_bytes > self.lru_capacity_bytes {
            return Err(GpuDemandAdmissionError::DemandSetExceedsLruCapacity {
                required_bytes: selected_lru_bytes,
                capacity: self.lru_capacity_bytes,
            });
        }

        let required_bytes = g.lru_used_bytes.checked_add(new_bytes).ok_or(
            GpuDemandAdmissionError::ProtectedDemandCapacity {
                required_bytes: usize::MAX,
                protected_bytes: g.lru_used_bytes,
                capacity: self.lru_capacity_bytes,
            },
        )?;
        let bytes_to_evict = required_bytes.saturating_sub(self.lru_capacity_bytes);
        let evictable_bytes = unprotected_logical_lru_bytes(&g, &selected);
        if bytes_to_evict > evictable_bytes {
            return Err(GpuDemandAdmissionError::ProtectedDemandCapacity {
                required_bytes,
                protected_bytes: g.lru_used_bytes.saturating_sub(evictable_bytes),
                capacity: self.lru_capacity_bytes,
            });
        }

        let generation_count = u64::try_from(new_ids.len())
            .map_err(|_| GpuDemandAdmissionError::GenerationExhausted)?;
        let next_generation = g
            .next_generation
            .checked_add(generation_count)
            .ok_or(GpuDemandAdmissionError::GenerationExhausted)?;

        while g
            .lru_used_bytes
            .checked_add(new_bytes)
            .is_none_or(|used| used > self.lru_capacity_bytes)
        {
            let victim_id = oldest_unprotected_logical_lru(&g, &selected)
                .expect("set capacity preflight proved an unprotected logical victim");
            let victim = g
                .lru
                .pop(&victim_id)
                .expect("selected logical victim remained under cache lock");
            g.lru_used_bytes = g
                .lru_used_bytes
                .checked_sub(victim.byte_len())
                .expect("logical GPU LRU byte underflow during demand-set admission");
            g.promotion_rearm.insert(victim_id);
        }

        let first_generation = g.next_generation;
        g.next_generation = next_generation;
        for (offset, expert_id) in new_ids.iter().copied().enumerate() {
            let source = sources
                .get(&expert_id)
                .expect("demand-set source validated before mutation")
                .clone();
            let generation = first_generation
                .checked_add(offset as u64)
                .expect("generation range preflight covered every new admission");
            let bytes = source.byte_len();
            g.promotion_pending.remove(&expert_id);
            g.promotion_rearm.remove(&expert_id);
            let previous = g.lru.put(
                expert_id,
                GpuAdmission {
                    resident: source,
                    generation,
                },
            );
            debug_assert!(previous.is_none(), "demand set rechecked under one lock");
            g.lru_used_bytes = g
                .lru_used_bytes
                .checked_add(bytes)
                .expect("logical GPU demand-set LRU byte overflow after capacity check");
        }

        let admissions = expert_ids
            .iter()
            .map(|expert_id| {
                g.anchor
                    .get(expert_id)
                    .or_else(|| g.lru.peek(expert_id))
                    .cloned()
                    .expect("every selected expert is admitted by the atomic transaction")
            })
            .collect();
        let newly_admitted = new_ids.len();
        drop(g);

        if newly_admitted > 0 {
            self.promotions
                .fetch_add(newly_admitted as u64, Ordering::Relaxed);
            self.refresh_used_bytes();
        }
        Ok(GpuDemandSetAdmission::Ready {
            admissions,
            newly_admitted,
        })
    }

    /// Check whether a selected foreground expert needs an owned host payload
    /// before attempting logical LRU admission. This probe never changes
    /// routing telemetry, LRU recency, generations, or byte accounting.
    pub fn demand_admission_preflight(
        &self,
        expert_id: u32,
        payload_bytes: usize,
    ) -> Result<GpuDemandAdmissionPreflight, GpuDemandAdmissionError> {
        let g = self.inner.lock();
        if g.anchor.contains_key(&expert_id) || g.lru.peek(&expert_id).is_some() {
            return Ok(GpuDemandAdmissionPreflight::AlreadyAdmitted);
        }
        if payload_bytes == 0 {
            return Err(GpuDemandAdmissionError::EmptyPayload);
        }
        if payload_bytes > self.lru_capacity_bytes {
            return Err(GpuDemandAdmissionError::PayloadExceedsLruCapacity {
                bytes: payload_bytes,
                capacity: self.lru_capacity_bytes,
            });
        }
        Ok(GpuDemandAdmissionPreflight::NeedsPayload)
    }

    /// Ensure that one selected foreground expert has a current logical GPU
    /// admission. New demand admissions always enter the evictable LRU Edge;
    /// they never consume or evict Anchor Core entries. Physical upload stays
    /// lazy and exclusively owned by `GpuBackend`.
    ///
    /// Existing admissions return `Ok(false)` without changing generation,
    /// byte accounting, hit/miss telemetry, or LRU recency. A new admission
    /// returns `Ok(true)` after evicting least-recently-used dynamic entries as
    /// needed. Failures leave the existing admission state unchanged.
    pub fn demand_admit_lru(
        &self,
        resident: Arc<GpuResident>,
    ) -> Result<bool, GpuDemandAdmissionError> {
        let bytes = resident.byte_len();
        if bytes == 0 {
            return Err(GpuDemandAdmissionError::EmptyPayload);
        }

        let mut g = self.inner.lock();
        if g.anchor.contains_key(&resident.id) || g.lru.peek(&resident.id).is_some() {
            return Ok(false);
        }
        if bytes > self.lru_capacity_bytes {
            return Err(GpuDemandAdmissionError::PayloadExceedsLruCapacity {
                bytes,
                capacity: self.lru_capacity_bytes,
            });
        }
        let additionally_protected = HashSet::new();
        let required_bytes = g.lru_used_bytes.checked_add(bytes).ok_or(
            GpuDemandAdmissionError::ProtectedDemandCapacity {
                required_bytes: usize::MAX,
                protected_bytes: g.lru_used_bytes,
                capacity: self.lru_capacity_bytes,
            },
        )?;
        let evictable_bytes = unprotected_logical_lru_bytes(&g, &additionally_protected);
        if required_bytes.saturating_sub(self.lru_capacity_bytes) > evictable_bytes {
            return Err(GpuDemandAdmissionError::ProtectedDemandCapacity {
                required_bytes,
                protected_bytes: g.lru_used_bytes.saturating_sub(evictable_bytes),
                capacity: self.lru_capacity_bytes,
            });
        }
        // Allocate identity before eviction so generation exhaustion cannot
        // disturb any existing admission or byte accounting.
        let generation = g
            .allocate_generation()
            .ok_or(GpuDemandAdmissionError::GenerationExhausted)?;

        while g
            .lru_used_bytes
            .checked_add(bytes)
            .is_none_or(|used| used > self.lru_capacity_bytes)
        {
            let evicted_id = oldest_unprotected_logical_lru(&g, &additionally_protected)
                .expect("demand capacity preflight proved an unprotected logical victim");
            let victim = g
                .lru
                .pop(&evicted_id)
                .expect("selected demand victim remained under cache lock");
            g.lru_used_bytes = g.lru_used_bytes.saturating_sub(victim.byte_len());
            g.promotion_rearm.insert(evicted_id);
        }

        let id = resident.id;
        g.promotion_pending.remove(&id);
        g.promotion_rearm.remove(&id);
        let previous = g.lru.put(
            id,
            GpuAdmission {
                resident,
                generation,
            },
        );
        debug_assert!(previous.is_none(), "demand admission rechecked under lock");
        g.lru_used_bytes = g
            .lru_used_bytes
            .checked_add(bytes)
            .expect("logical GPU demand LRU byte overflow after capacity check");
        drop(g);

        self.promotions.fetch_add(1, Ordering::Relaxed);
        self.refresh_used_bytes();
        Ok(true)
    }

    /// Number of Anchor Core entries.
    pub fn anchor_len(&self) -> usize {
        self.inner.lock().anchor.len()
    }

    /// Number of LRU Edge entries.
    pub fn lru_len(&self) -> usize {
        self.inner.lock().lru.len()
    }

    fn refresh_used_bytes(&self) {
        let g = self.inner.lock();
        let total = g
            .anchor_used_bytes
            .checked_add(g.lru_used_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .expect("logical GPU admitted-byte total overflow");
        drop(g);
        self.logical_admitted_bytes.store(total, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;

    fn make(id: u32, pool: &BufferPool) -> Arc<ExpertResident> {
        let buffer = pool.try_acquire().unwrap();
        Arc::new(ExpertResident::new(id, buffer))
    }

    #[test]
    fn resident_uses_configured_block_align_for_uth_payload() {
        let block_align = 8192usize;
        let mut file = Vec::new();
        crate::tensor_header::TensorHeader::for_swiglu_expert(
            crate::inference::WeightDtype::F32,
            8,
            16,
        )
        .write_padded(block_align, &mut file);
        file.extend_from_slice(&[0xA5u8; 64]);
        file.resize(block_align * 2, 0);

        let pool = BufferPool::new(1, file.len(), block_align);
        let mut buffer = pool.try_acquire().unwrap();
        buffer.as_mut_slice().copy_from_slice(&file);
        let resident = ExpertResident::new_with_block_align(0, buffer, block_align);
        assert_eq!(resident.data().len(), block_align);
        assert!(resident.data()[..64].iter().all(|&b| b == 0xA5));
        assert!(resident.data()[64..].iter().all(|&b| b == 0));

        let range = |offset| crate::tensor_header::ProjectionRange {
            dtype: crate::tensor_header::UthDtypeId::F32,
            offset,
            len: 16,
            weights: 4,
        };
        let mixed =
            crate::tensor_header::MixedExpertHeader::new(2, 2, range(0), range(16), range(32));
        let mut mixed_file = Vec::new();
        mixed.write_padded(block_align, &mut mixed_file);
        mixed_file.extend_from_slice(&[0x5Au8; 64]);
        mixed_file.resize(block_align * 2, 0);

        let pool = BufferPool::new(1, mixed_file.len(), block_align);
        let mut buffer = pool.try_acquire().unwrap();
        buffer.as_mut_slice().copy_from_slice(&mixed_file);
        let resident = ExpertResident::new_with_block_align(1, buffer, block_align);
        assert!(resident.mixed_layout().is_some());
        assert_eq!(resident.data().len(), block_align);
        assert!(resident.data()[..64].iter().all(|&b| b == 0x5A));
        assert!(resident.data()[64..].iter().all(|&b| b == 0));
    }

    #[test]
    fn lru_eviction_returns_buffer_to_pool() {
        let pool = BufferPool::new(3, 4096, 4096);
        let cache = ExpertCache::new(2);

        let _ = cache
            .insert(make(0, &pool))
            .map_err(|_| panic!("insert failed"));
        let _ = cache
            .insert(make(1, &pool))
            .map_err(|_| panic!("insert failed"));
        // 2 of 3 slots are occupied by cache entries; 1 is free.
        let scratch = pool.try_acquire().expect("third slot free");
        assert!(pool.try_acquire().is_none());
        drop(scratch);

        // Inserting a third entry evicts expert 0 (the LRU). The evicted
        // Arc is returned and the cache no longer references its buffer.
        let evicted = match cache.insert(make(2, &pool)) {
            Ok(Some(e)) => e,
            other => panic!("expected Ok(Some(_)), got {:?}", other.is_ok()),
        };
        assert_eq!(evicted.id, 0);

        // Pool is fully occupied (cache holds 1 + 2, plus the evicted Arc
        // still holds expert 0's buffer).
        assert!(pool.try_acquire().is_none());
        // Once the evicted Arc is dropped, its buffer returns to the pool.
        drop(evicted);
        assert!(pool.try_acquire().is_some());
    }

    #[test]
    fn hit_updates_recency() {
        let pool = BufferPool::new(3, 4096, 4096);
        let cache = ExpertCache::new(2);
        let _ = cache
            .insert(make(0, &pool))
            .map_err(|_| panic!("insert failed"));
        let _ = cache
            .insert(make(1, &pool))
            .map_err(|_| panic!("insert failed"));
        // Touch expert 0 -> it is now most-recently used.
        let _ = cache.get(0);
        // Inserting expert 2 should evict 1, not 0.
        let _ = cache
            .insert(make(2, &pool))
            .map_err(|_| panic!("insert failed"));
        assert!(cache.contains(0));
        assert!(!cache.contains(1));
        assert!(cache.contains(2));
    }

    #[test]
    fn pinned_entry_is_protected_from_eviction() {
        let pool = BufferPool::new(4, 4096, 4096);
        let cache = ExpertCache::new(2);
        let _ = cache
            .insert(make(0, &pool))
            .map_err(|_| panic!("insert failed"));
        let _ = cache
            .insert(make(1, &pool))
            .map_err(|_| panic!("insert failed"));
        // Pin expert 0. Even though it's the LRU, expert 1 must be
        // evicted instead when expert 2 is inserted.
        cache.pin(0);
        let evicted = match cache.insert(make(2, &pool)) {
            Ok(Some(e)) => e,
            other => panic!("expected Ok(Some(_)), got {:?}", other.is_ok()),
        };
        assert_eq!(evicted.id, 1);
        assert!(cache.contains(0));
        assert!(!cache.contains(1));
        assert!(cache.contains(2));
        assert!(cache.is_pinned(0));
        assert_eq!(cache.pinned_count(), 1);
    }

    #[test]
    fn evict_lru_returns_none_when_all_pinned() {
        let pool = BufferPool::new(4, 4096, 4096);
        let cache = ExpertCache::new(2);
        let _ = cache
            .insert(make(0, &pool))
            .map_err(|_| panic!("insert failed"));
        let _ = cache
            .insert(make(1, &pool))
            .map_err(|_| panic!("insert failed"));
        cache.pin(0);
        cache.pin(1);
        assert!(cache.evict_lru().is_none());
    }

    #[test]
    fn cost_aware_evicts_coldest_not_lru() {
        // Scenario where strict LRU and cost-aware eviction diverge: the
        // *least-recently-used* resident is also the *hottest*. Cost-aware
        // mode must keep the hot expert and evict the cold newcomer.
        let pool = BufferPool::new(4, 4096, 4096);
        let cache = ExpertCache::new(2);
        cache.set_cost_aware(true);
        // A (id 0) is loaded and then hit many times → high heat. The
        // hits make it most-recently-used for now.
        let _ = cache
            .insert(make(0, &pool))
            .map_err(|_| panic!("insert failed"));
        for _ in 0..10 {
            let _ = cache.get(0);
        }
        // B (id 1) is loaded cold. Now A is the LRU (B is MRU) but A is
        // far hotter than B.
        let _ = cache
            .insert(make(1, &pool))
            .map_err(|_| panic!("insert failed"));
        // Inserting C evicts a victim: cost-aware keeps hot A, drops cold B.
        let evicted = match cache.insert(make(2, &pool)) {
            Ok(Some(e)) => e,
            other => panic!("expected an eviction, got ok={}", other.is_ok()),
        };
        assert_eq!(
            evicted.id, 1,
            "cost-aware eviction should drop the cold expert (1), not the hot LRU (0)"
        );
        assert!(cache.contains(0));
        assert!(!cache.contains(1));
        assert!(cache.contains(2));
    }

    #[test]
    fn cost_aware_disabled_is_pure_lru() {
        // The same access pattern as `cost_aware_evicts_coldest_not_lru`,
        // but with cost-aware mode OFF (the default), must reproduce the
        // legacy strict-LRU outcome: the hot-but-LRU expert is evicted.
        let pool = BufferPool::new(4, 4096, 4096);
        let cache = ExpertCache::new(2);
        let _ = cache
            .insert(make(0, &pool))
            .map_err(|_| panic!("insert failed"));
        for _ in 0..10 {
            let _ = cache.get(0);
        }
        let _ = cache
            .insert(make(1, &pool))
            .map_err(|_| panic!("insert failed"));
        // 0 is MRU after the hits, then 1 is inserted → 1 MRU, 0 LRU.
        let evicted = match cache.insert(make(2, &pool)) {
            Ok(Some(e)) => e,
            other => panic!("expected an eviction, got ok={}", other.is_ok()),
        };
        assert_eq!(
            evicted.id, 0,
            "pure LRU should evict the least-recently-used expert (0)"
        );
        assert!(!cache.contains(0));
        assert!(cache.contains(1));
        assert!(cache.contains(2));
    }

    #[test]
    fn cost_aware_heat_decays_releasing_stale_hot() {
        // An expert that was hot long ago but has gone cold must
        // eventually become the eviction victim as its heat decays under
        // sustained churn from other experts.
        let pool = BufferPool::new(8, 4096, 4096);
        let cache = ExpertCache::new(2);
        cache.set_cost_aware(true);
        // Make expert 0 very hot, then never touch it again.
        let _ = cache
            .insert(make(0, &pool))
            .map_err(|_| panic!("insert failed"));
        for _ in 0..20 {
            let _ = cache.get(0);
        }
        // Churn many distinct cold experts through the other slot. Each
        // insertion ages expert 0's heat; after enough churn its decayed
        // heat falls below a freshly-loaded expert's seed and it is
        // finally evicted.
        let mut zero_evicted = false;
        for id in 1..400u32 {
            if let Ok(Some(ev)) = cache.insert(make(id, &pool)) {
                if ev.id == 0 {
                    zero_evicted = true;
                    break;
                }
            }
        }
        assert!(
            zero_evicted,
            "stale-hot expert 0 should eventually be released once its heat decays"
        );
    }

    #[test]
    fn insert_returns_err_when_all_pinned() {
        // Cache full of pinned entries must reject a new insert with
        // `Err(resident)` rather than silently evicting a pinned slot.
        let pool = BufferPool::new(4, 4096, 4096);
        let cache = ExpertCache::new(2);
        let _ = cache
            .insert(make(0, &pool))
            .map_err(|_| panic!("insert failed"));
        let _ = cache
            .insert(make(1, &pool))
            .map_err(|_| panic!("insert failed"));
        cache.pin(0);
        cache.pin(1);
        let new_resident = make(2, &pool);
        let new_id = new_resident.id;
        let err = match cache.insert(new_resident) {
            Err(rejected) => rejected,
            Ok(_) => panic!("expected Err, got Ok"),
        };
        assert_eq!(err.id, new_id);
        // Both pinned entries are still resident.
        assert!(cache.contains(0));
        assert!(cache.contains(1));
        assert!(!cache.contains(2));
        // The rejected resident's buffer returns to the pool when
        // dropped — i.e. the contract that a rejected insert hands the
        // Arc back so its PooledBuffer can be reclaimed.
        drop(err);
        // After dropping the rejected resident *and* the scratch
        // buffer that `make(2, ...)` consumed, the pool should have
        // strictly more free slots than it did at the rejection.
        assert!(pool.try_acquire().is_some());
    }

    fn gpu_res(id: u32, bytes: usize) -> Arc<GpuResident> {
        Arc::new(GpuResident::new(id, vec![0u8; bytes]))
    }

    fn ready_demand_set(outcome: GpuDemandSetAdmission) -> (Vec<GpuAdmission>, usize) {
        match outcome {
            GpuDemandSetAdmission::Ready {
                admissions,
                newly_admitted,
            } => (admissions, newly_admitted),
            GpuDemandSetAdmission::PayloadRequired(missing) => {
                panic!("unexpected missing demand payloads: {missing:?}")
            }
        }
    }

    #[test]
    fn all_current_demand_set_is_hit_only_and_preserves_identity() {
        let cache = Arc::new(GpuExpertCache::new(16, 0.0, 0));
        assert_eq!(cache.demand_admit_lru(gpu_res(1, 8)), Ok(true));
        assert_eq!(cache.demand_admit_lru(gpu_res(2, 8)), Ok(true));
        let generations = [
            cache.current_generation(1).unwrap(),
            cache.current_generation(2).unwrap(),
        ];
        let promotions = cache.promotions();
        let used = cache.used_bytes();
        let _protection = cache.protect_demand_set(&[1, 2]).unwrap();

        let (admissions, newly_admitted) =
            ready_demand_set(cache.demand_admit_set(&[1, 2], &HashMap::new()).unwrap());

        assert_eq!(newly_admitted, 0);
        assert_eq!(admissions[0].generation(), generations[0]);
        assert_eq!(admissions[1].generation(), generations[1]);
        assert_eq!(cache.promotions(), promotions);
        assert_eq!(cache.used_bytes(), used);
    }

    #[test]
    fn all_missing_demand_set_is_admitted_atomically() {
        let cache = Arc::new(GpuExpertCache::new(16, 0.0, 0));
        let _protection = cache.protect_demand_set(&[1, 2]).unwrap();
        assert!(matches!(
            cache.demand_admit_set(&[1, 2], &HashMap::new()).unwrap(),
            GpuDemandSetAdmission::PayloadRequired(ids) if ids == vec![1, 2]
        ));
        let sources = HashMap::from([(1, gpu_res(1, 8)), (2, gpu_res(2, 8))]);

        let (admissions, newly_admitted) =
            ready_demand_set(cache.demand_admit_set(&[1, 2], &sources).unwrap());

        assert_eq!(newly_admitted, 2);
        assert_eq!(admissions.len(), 2);
        assert!(cache.contains_generation(1, admissions[0].generation()));
        assert!(cache.contains_generation(2, admissions[1].generation()));
        assert_eq!(cache.used_bytes(), 16);
    }

    #[test]
    fn mixed_demand_set_evicts_non_selected_lru_not_current_member() {
        let cache = Arc::new(GpuExpertCache::new(24, 0.0, 0));
        assert_eq!(cache.demand_admit_lru(gpu_res(1, 8)), Ok(true));
        assert_eq!(cache.demand_admit_lru(gpu_res(90, 8)), Ok(true));
        assert_eq!(cache.demand_admit_lru(gpu_res(91, 8)), Ok(true));
        let generation = cache.current_generation(1).unwrap();
        let _protection = cache.protect_demand_set(&[1, 2]).unwrap();
        let sources = HashMap::from([(2, gpu_res(2, 8))]);

        let (admissions, newly_admitted) =
            ready_demand_set(cache.demand_admit_set(&[1, 2], &sources).unwrap());

        assert_eq!(newly_admitted, 1);
        assert_eq!(admissions[0].generation(), generation);
        assert!(cache.contains_generation(1, generation));
        assert!(cache.contains(2));
        assert!(
            !cache.contains(90),
            "oldest non-selected entry must be evicted"
        );
        assert!(cache.contains(91));
    }

    #[test]
    fn insufficient_selected_set_capacity_fails_atomically() {
        let cache = Arc::new(GpuExpertCache::new(12, 0.0, 0));
        assert_eq!(cache.demand_admit_lru(gpu_res(90, 4)), Ok(true));
        let generation = cache.current_generation(90).unwrap();
        let used = cache.used_bytes();
        let promotions = cache.promotions();
        let next_generation = cache.inner.lock().next_generation;
        let _protection = cache.protect_demand_set(&[1, 2]).unwrap();
        let sources = HashMap::from([(1, gpu_res(1, 8)), (2, gpu_res(2, 8))]);

        assert!(matches!(
            cache.demand_admit_set(&[1, 2], &sources),
            Err(GpuDemandAdmissionError::DemandSetExceedsLruCapacity {
                required_bytes: 16,
                capacity: 12,
            })
        ));
        assert!(cache.contains_generation(90, generation));
        assert!(!cache.contains(1));
        assert!(!cache.contains(2));
        assert_eq!(cache.used_bytes(), used);
        assert_eq!(cache.promotions(), promotions);
        assert_eq!(cache.inner.lock().next_generation, next_generation);
    }

    #[test]
    fn foreground_protection_blocks_individual_self_eviction() {
        let cache = Arc::new(GpuExpertCache::new(16, 0.0, 0));
        assert_eq!(cache.demand_admit_lru(gpu_res(1, 8)), Ok(true));
        assert_eq!(cache.demand_admit_lru(gpu_res(90, 8)), Ok(true));
        let generation = cache.current_generation(1).unwrap();
        let protection = cache.protect_demand_set(&[1, 2]).unwrap();

        assert_eq!(cache.demand_admit_lru(gpu_res(2, 8)), Ok(true));
        assert!(cache.contains_generation(1, generation));
        assert!(cache.contains(2));
        assert!(!cache.contains(90));

        drop(protection);
        assert_eq!(cache.demand_admit_lru(gpu_res(3, 8)), Ok(true));
        assert!(
            !cache.contains(1),
            "released protection restores ordinary LRU eviction"
        );
    }

    #[test]
    fn protected_capacity_conflict_fails_without_oscillation() {
        let cache = Arc::new(GpuExpertCache::new(16, 0.0, 0));
        assert_eq!(cache.demand_admit_lru(gpu_res(1, 8)), Ok(true));
        assert_eq!(cache.demand_admit_lru(gpu_res(2, 8)), Ok(true));
        let _other_foreground = cache.protect_demand_set(&[1, 2]).unwrap();
        let sources = HashMap::from([(3, gpu_res(3, 8))]);

        assert!(matches!(
            cache.demand_admit_set(&[3], &sources),
            Err(GpuDemandAdmissionError::ProtectedDemandCapacity {
                required_bytes: 24,
                protected_bytes: 16,
                capacity: 16,
            })
        ));
        assert!(cache.contains(1));
        assert!(cache.contains(2));
        assert!(!cache.contains(3));
    }

    #[test]
    fn stale_probe_generation_is_rejected_and_recovery_allocates_fresh_identity() {
        let cache = Arc::new(GpuExpertCache::new(8, 0.0, 0));
        assert_eq!(cache.demand_admit_lru(gpu_res(1, 8)), Ok(true));
        let stale_generation = cache.current_generation(1).unwrap();
        assert_eq!(cache.demand_admit_lru(gpu_res(90, 8)), Ok(true));
        assert!(!cache.contains_generation(1, stale_generation));
        let _protection = cache.protect_demand_set(&[1]).unwrap();
        let sources = HashMap::from([(1, gpu_res(1, 8))]);

        let (admissions, newly_admitted) =
            ready_demand_set(cache.demand_admit_set(&[1], &sources).unwrap());

        assert_eq!(newly_admitted, 1);
        assert!(admissions[0].generation() > stale_generation);
        assert!(!cache.contains_generation(1, stale_generation));
        assert!(cache.contains_generation(1, admissions[0].generation()));
    }

    #[test]
    fn speculative_hot_promotion_cannot_evict_foreground_set() {
        let cache = Arc::new(GpuExpertCache::new(16, 0.0, 3));
        assert_eq!(cache.demand_admit_lru(gpu_res(1, 8)), Ok(true));
        assert_eq!(cache.demand_admit_lru(gpu_res(2, 8)), Ok(true));
        let generations = [
            cache.current_generation(1).unwrap(),
            cache.current_generation(2).unwrap(),
        ];
        assert!(cache.claim_promotion(3, 3));
        let _protection = cache.protect_demand_set(&[1, 2]).unwrap();

        assert_eq!(
            cache.promote_hot_sync(gpu_res(3, 8)),
            GpuHotPromotionOutcome::NoCapacity
        );
        assert!(cache.contains_generation(1, generations[0]));
        assert!(cache.contains_generation(2, generations[1]));
        assert!(!cache.contains(3));
    }

    #[test]
    fn duplicate_demand_protection_is_fail_closed() {
        let cache = Arc::new(GpuExpertCache::new(16, 0.0, 0));
        assert!(matches!(
            cache.protect_demand_set(&[1, 1]),
            Err(GpuDemandAdmissionError::DuplicateDemandExpert { expert_id: 1 })
        ));
        assert!(cache.inner.lock().demand_protections.is_empty());
    }

    #[test]
    fn try_promote_lru_no_evict_skips_when_lru_full() {
        // anchor_ratio = 0.0 → entire budget is LRU; capacity = 100B.
        let cache = GpuExpertCache::new(100, 0.0, 0);
        // Fill the LRU Edge exactly.
        assert!(cache.try_promote_lru_no_evict(gpu_res(1, 60)));
        assert!(cache.try_promote_lru_no_evict(gpu_res(2, 40)));
        assert_eq!(cache.lru_len(), 2);
        // No room left, and the helper must NOT evict.
        assert!(!cache.try_promote_lru_no_evict(gpu_res(3, 1)));
        assert!(cache.contains(1));
        assert!(cache.contains(2));
        assert!(!cache.contains(3));
        // Promotion counter only advanced for the two successful installs.
        assert_eq!(cache.promotions(), 2);
    }

    #[test]
    fn try_promote_lru_no_evict_never_uses_anchor_core() {
        // Anchor gets 50B; LRU gets 50B. A 40B entry would *fit* the
        // anchor budget (and `promote_sync` would place it there), but
        // the no-evict helper must keep it in the LRU Edge so the
        // threshold-driven background task is the only thing that ever
        // promotes into the Anchor Core.
        let cache = GpuExpertCache::new(100, 0.5, 0);
        assert!(cache.try_promote_lru_no_evict(gpu_res(7, 40)));
        assert_eq!(cache.anchor_len(), 0);
        assert_eq!(cache.lru_len(), 1);
    }

    #[test]
    fn try_promote_lru_no_evict_is_idempotent() {
        let cache = GpuExpertCache::new(100, 0.0, 0);
        assert!(cache.try_promote_lru_no_evict(gpu_res(9, 32)));
        // Second call for the same id is a no-op (already resident),
        // and must not double-count the promotion counter or bytes.
        assert!(!cache.try_promote_lru_no_evict(gpu_res(9, 32)));
        assert_eq!(cache.promotions(), 1);
        assert_eq!(cache.used_bytes(), 32);
    }

    #[test]
    fn failed_no_evict_promotion_preserves_eviction_rearm() {
        let cache = GpuExpertCache::new(20, 0.0, 3);
        assert!(cache.promote_sync(gpu_res(1, 8)));
        assert!(cache.promote_sync(gpu_res(2, 20)));
        assert!(!cache.contains(1), "id 1 must be rearmed by logical eviction");

        assert!(!cache.try_promote_lru_no_evict(gpu_res(1, 8)));
        assert!(
            cache.claim_promotion(1, 99),
            "failed opportunistic admission must leave one rearm claim"
        );
        assert!(!cache.claim_promotion(1, 100), "rearm remains one-shot");
    }

    #[test]
    fn failed_no_evict_promotion_preserves_pending_claim() {
        let cache = GpuExpertCache::new(20, 0.0, 3);
        assert!(cache.promote_sync(gpu_res(1, 8)));
        assert!(cache.promote_sync(gpu_res(2, 20)));
        assert!(!cache.contains(1));
        assert!(cache.claim_promotion(1, 3));

        assert!(!cache.try_promote_lru_no_evict(gpu_res(1, 8)));
        assert!(
            !cache.claim_promotion(1, 3),
            "failed opportunistic admission must keep the existing claim pending"
        );
    }

    #[test]
    fn current_admission_anchor_does_not_change_counters() {
        let cache = GpuExpertCache::new(100, 1.0, 0);
        let resident = gpu_res(1, 32);
        assert!(cache.promote_sync(resident.clone()));
        assert_eq!(cache.anchor_len(), 1);

        let admission = cache.current_admission(1).expect("anchor admission");
        assert!(Arc::ptr_eq(admission.resident(), &resident));
        assert_eq!(admission.generation(), cache.current_generation(1).unwrap());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn current_admission_lru_does_not_change_counters() {
        let cache = GpuExpertCache::new(100, 0.0, 0);
        let resident = gpu_res(2, 32);
        assert!(cache.promote_sync(resident.clone()));
        assert_eq!(cache.lru_len(), 1);

        let admission = cache.current_admission(2).expect("LRU admission");
        assert!(Arc::ptr_eq(admission.resident(), &resident));
        assert_eq!(admission.generation(), cache.current_generation(2).unwrap());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn current_admission_lru_does_not_update_recency() {
        let cache = GpuExpertCache::new(20, 0.0, 0);
        assert!(cache.promote_sync(gpu_res(1, 10)));
        assert!(cache.promote_sync(gpu_res(2, 10)));

        assert!(cache.current_admission(1).is_some());
        assert!(cache.promote_sync(gpu_res(3, 10)));
        assert!(!cache.contains(1), "non-mutating lookup must leave id 1 LRU");
        assert!(cache.contains(2));
        assert!(cache.contains(3));
    }

    #[test]
    fn current_admission_miss_does_not_increment_misses() {
        let cache = GpuExpertCache::new(20, 0.0, 0);
        assert!(cache.current_admission(99).is_none());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn demand_preflight_detects_anchor_without_state_changes() {
        let cache = GpuExpertCache::new(100, 0.5, 0);
        assert!(cache.promote_sync(gpu_res(99, 40)));
        let generation = cache.current_generation(99).unwrap();
        let used = cache.used_bytes();
        let promotions = cache.promotions();
        let hits = cache.hits();
        let misses = cache.misses();

        assert_eq!(
            cache.demand_admission_preflight(99, 40),
            Ok(GpuDemandAdmissionPreflight::AlreadyAdmitted)
        );
        assert!(cache.contains_generation(99, generation));
        assert_eq!(cache.used_bytes(), used);
        assert_eq!(cache.promotions(), promotions);
        assert_eq!(cache.hits(), hits);
        assert_eq!(cache.misses(), misses);
        assert_eq!(cache.anchor_len(), 1);
        assert_eq!(cache.lru_len(), 0);
    }

    #[test]
    fn demand_preflight_detects_lru_without_touching_recency() {
        let cache = GpuExpertCache::new(20, 0.0, 0);
        assert_eq!(cache.demand_admit_lru(gpu_res(1, 10)), Ok(true));
        assert_eq!(cache.demand_admit_lru(gpu_res(2, 10)), Ok(true));
        let hits = cache.hits();
        let misses = cache.misses();

        assert_eq!(
            cache.demand_admission_preflight(1, 10),
            Ok(GpuDemandAdmissionPreflight::AlreadyAdmitted)
        );
        assert_eq!(cache.hits(), hits);
        assert_eq!(cache.misses(), misses);
        assert_eq!(cache.demand_admit_lru(gpu_res(3, 10)), Ok(true));
        assert!(!cache.contains(1), "preflight must leave id 1 as LRU");
        assert!(cache.contains(2));
        assert!(cache.contains(3));
    }

    #[test]
    fn demand_preflight_absent_fit_needs_payload_and_zero_is_typed() {
        let cache = GpuExpertCache::new(100, 0.5, 0);
        assert_eq!(
            cache.demand_admission_preflight(7, 50),
            Ok(GpuDemandAdmissionPreflight::NeedsPayload)
        );
        assert_eq!(
            cache.demand_admission_preflight(7, 0),
            Err(GpuDemandAdmissionError::EmptyPayload)
        );
    }

    #[test]
    fn oversized_demand_preflight_preserves_all_logical_state() {
        let cache = GpuExpertCache::new(100, 0.5, 0);
        assert!(cache.promote_sync(gpu_res(99, 40)));
        assert_eq!(cache.demand_admit_lru(gpu_res(1, 25)), Ok(true));
        let anchor_generation = cache.current_generation(99).unwrap();
        let lru_generation = cache.current_generation(1).unwrap();
        let used = cache.used_bytes();
        let promotions = cache.promotions();
        let next_generation = cache.inner.lock().next_generation;

        assert_eq!(
            cache.demand_admission_preflight(2, 51),
            Err(GpuDemandAdmissionError::PayloadExceedsLruCapacity {
                bytes: 51,
                capacity: 50,
            })
        );
        assert!(cache.contains_generation(99, anchor_generation));
        assert!(cache.contains_generation(1, lru_generation));
        assert!(!cache.contains(2));
        assert_eq!(cache.used_bytes(), used);
        assert_eq!(cache.promotions(), promotions);
        assert_eq!(cache.inner.lock().next_generation, next_generation);
        assert_eq!(cache.anchor_len(), 1);
        assert_eq!(cache.lru_len(), 1);
    }

    #[test]
    fn demand_install_rechecks_existing_admission_after_preflight() {
        let cache = GpuExpertCache::new(100, 0.0, 0);
        assert_eq!(
            cache.demand_admission_preflight(7, 32),
            Ok(GpuDemandAdmissionPreflight::NeedsPayload)
        );
        assert_eq!(cache.demand_admit_lru(gpu_res(7, 32)), Ok(true));
        let generation = cache.current_generation(7).unwrap();
        let used = cache.used_bytes();
        let promotions = cache.promotions();
        let next_generation = cache.inner.lock().next_generation;

        assert_eq!(cache.demand_admit_lru(gpu_res(7, 64)), Ok(false));
        assert!(cache.contains_generation(7, generation));
        assert_eq!(cache.used_bytes(), used);
        assert_eq!(cache.promotions(), promotions);
        assert_eq!(cache.inner.lock().next_generation, next_generation);
        assert_eq!(cache.lru_len(), 1);
    }

    #[test]
    fn cold_demand_admits_to_lru_without_consuming_anchor() {
        let cache = GpuExpertCache::new(100, 0.5, 0);
        assert_eq!(cache.demand_admit_lru(gpu_res(1, 40)), Ok(true));
        assert!(cache.current_admission(1).is_some());
        assert_eq!(cache.anchor_len(), 0);
        assert_eq!(cache.lru_len(), 1);
        assert_eq!(cache.used_bytes(), 40);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    #[test]
    fn demand_admission_evicts_lru_and_preserves_anchor() {
        let cache = GpuExpertCache::new(100, 0.5, 0);
        assert!(cache.promote_sync(gpu_res(99, 40)));
        let anchor_generation = cache.current_generation(99).unwrap();
        assert_eq!(cache.demand_admit_lru(gpu_res(1, 25)), Ok(true));
        assert_eq!(cache.demand_admit_lru(gpu_res(2, 25)), Ok(true));

        assert_eq!(cache.demand_admit_lru(gpu_res(3, 25)), Ok(true));
        assert!(!cache.contains(1), "oldest dynamic entry must be evicted");
        assert!(cache.contains(2));
        assert!(cache.contains(3));
        assert!(cache.contains_generation(99, anchor_generation));
        assert_eq!(cache.anchor_len(), 1);
        assert_eq!(cache.lru_len(), 2);
    }

    #[test]
    fn existing_demand_preserves_generation_bytes_and_counters() {
        let cache = GpuExpertCache::new(96, 0.0, 0);
        assert_eq!(cache.demand_admit_lru(gpu_res(7, 32)), Ok(true));
        assert_eq!(cache.demand_admit_lru(gpu_res(8, 32)), Ok(true));
        let generation = cache.current_generation(7).unwrap();
        let used = cache.used_bytes();
        let promotions = cache.promotions();
        let hits = cache.hits();
        let misses = cache.misses();

        assert_eq!(cache.demand_admit_lru(gpu_res(7, 48)), Ok(false));
        assert!(cache.contains_generation(7, generation));
        assert_eq!(cache.used_bytes(), used);
        assert_eq!(cache.promotions(), promotions);
        assert_eq!(cache.hits(), hits);
        assert_eq!(cache.misses(), misses);
        assert_eq!(cache.lru_len(), 2);

        assert_eq!(cache.demand_admit_lru(gpu_res(9, 40)), Ok(true));
        assert!(!cache.contains(7), "existing demand lookup must not touch LRU");
        assert!(cache.contains(8));
        assert!(cache.contains(9));
    }

    #[test]
    fn demand_admitted_threshold_hot_lru_moves_to_anchor_with_exact_identity() {
        let cache = GpuExpertCache::new(100, 0.5, 3);
        let resident = gpu_res(7, 40);
        assert_eq!(cache.demand_admit_lru(resident.clone()), Ok(true));
        let admission_before = cache.current_admission(7).expect("LRU admission");
        let generation = admission_before.generation();
        let used = cache.used_bytes();
        let next_generation = cache.inner.lock().next_generation;

        assert!(!cache.claim_promotion(7, 2));
        assert!(cache.claim_promotion(7, 3));
        assert!(
            !cache.claim_promotion(7, 3),
            "threshold-hot LRU admission must have one pending request"
        );
        assert_eq!(
            cache.promote_hot_existing(7),
            GpuHotPromotionOutcome::MovedLruToAnchor
        );

        let admission_after = cache.current_admission(7).expect("Anchor admission");
        assert_eq!(admission_after.generation(), generation);
        assert!(Arc::ptr_eq(admission_after.resident(), &resident));
        assert!(Arc::ptr_eq(
            admission_after.resident(),
            admission_before.resident()
        ));
        assert_eq!(cache.anchor_len(), 1);
        assert_eq!(cache.lru_len(), 0);
        assert_eq!(cache.used_bytes(), used);
        assert_eq!(cache.promotions(), 2, "demand install + tier graduation");
        let inner = cache.inner.lock();
        assert_eq!(inner.anchor_used_bytes, 40);
        assert_eq!(inner.lru_used_bytes, 0);
        assert_eq!(inner.next_generation, next_generation);
        assert!(!inner.promotion_pending.contains(&7));
        assert!(!inner.promotion_rearm.contains(&7));
    }

    #[test]
    fn hot_lru_move_preserves_unrelated_lru_recency() {
        let cache = GpuExpertCache::new(120, 0.5, 3);
        assert_eq!(cache.demand_admit_lru(gpu_res(1, 20)), Ok(true));
        assert_eq!(cache.demand_admit_lru(gpu_res(2, 20)), Ok(true));
        assert_eq!(cache.demand_admit_lru(gpu_res(3, 20)), Ok(true));

        assert!(cache.claim_promotion(2, 3));
        assert_eq!(
            cache.promote_hot_existing(2),
            GpuHotPromotionOutcome::MovedLruToAnchor
        );
        assert_eq!(cache.demand_admit_lru(gpu_res(4, 40)), Ok(true));

        assert!(!cache.contains(1), "oldest unrelated LRU entry must remain victim");
        assert!(cache.contains(2));
        assert!(cache.contains(3));
        assert!(cache.contains(4));
        assert_eq!(cache.anchor_len(), 1);
        assert_eq!(cache.lru_len(), 2);
    }

    #[test]
    fn anchor_full_hot_lru_is_unchanged_and_does_not_requeue() {
        let cache = GpuExpertCache::new(100, 0.5, 3);
        assert_eq!(
            cache.promote_hot_sync(gpu_res(99, 50)),
            GpuHotPromotionOutcome::InstalledAnchor
        );
        let resident = gpu_res(7, 25);
        assert_eq!(cache.demand_admit_lru(resident.clone()), Ok(true));
        assert_eq!(cache.demand_admit_lru(gpu_res(8, 25)), Ok(true));
        let generation = cache.current_generation(7).unwrap();
        let used = cache.used_bytes();
        let promotions = cache.promotions();
        let next_generation = cache.inner.lock().next_generation;

        assert!(cache.claim_promotion(7, 3));
        assert_eq!(
            cache.promote_hot_existing(7),
            GpuHotPromotionOutcome::AlreadyLruAnchorFull
        );
        assert_eq!(cache.current_generation(7), Some(generation));
        assert!(Arc::ptr_eq(
            cache.current_admission(7).unwrap().resident(),
            &resident
        ));
        assert_eq!(cache.used_bytes(), used);
        assert_eq!(cache.promotions(), promotions);
        assert_eq!(cache.inner.lock().next_generation, next_generation);
        assert_eq!(cache.anchor_len(), 1);
        assert_eq!(cache.lru_len(), 2);
        assert!(!cache.claim_promotion(7, 4));
        assert!(!cache.claim_promotion(7, 99));

        assert_eq!(cache.demand_admit_lru(gpu_res(9, 25)), Ok(true));
        assert!(!cache.contains(7), "Anchor-full attempt must not touch LRU recency");
        assert!(cache.contains(8));
        assert!(cache.contains(9));
    }

    #[test]
    fn already_anchor_hot_promotion_is_uncounted_noop() {
        let cache = GpuExpertCache::new(100, 0.5, 3);
        let resident = gpu_res(7, 40);
        assert_eq!(
            cache.promote_hot_sync(resident.clone()),
            GpuHotPromotionOutcome::InstalledAnchor
        );
        let generation = cache.current_generation(7).unwrap();
        let used = cache.used_bytes();
        let promotions = cache.promotions();

        assert!(!cache.claim_promotion(7, 3));
        assert_eq!(
            cache.promote_hot_existing(7),
            GpuHotPromotionOutcome::AlreadyAnchor
        );
        assert_eq!(cache.current_generation(7), Some(generation));
        assert!(Arc::ptr_eq(
            cache.current_admission(7).unwrap().resident(),
            &resident
        ));
        assert_eq!(cache.used_bytes(), used);
        assert_eq!(cache.promotions(), promotions);
    }

    #[test]
    fn demand_wins_threshold_race_then_existing_admission_moves_intact() {
        let cache = GpuExpertCache::new(100, 0.5, 3);
        assert!(cache.claim_promotion(7, 3));
        assert_eq!(
            cache.promote_hot_existing(7),
            GpuHotPromotionOutcome::PayloadRequired
        );
        let demand_resident = gpu_res(7, 40);
        assert_eq!(cache.demand_admit_lru(demand_resident.clone()), Ok(true));
        let generation = cache.current_generation(7).unwrap();

        assert_eq!(
            cache.promote_hot_sync(gpu_res(7, 40)),
            GpuHotPromotionOutcome::MovedLruToAnchor
        );
        let current = cache.current_admission(7).unwrap();
        assert_eq!(current.generation(), generation);
        assert!(Arc::ptr_eq(current.resident(), &demand_resident));
        assert_eq!(cache.anchor_len(), 1);
        assert_eq!(cache.lru_len(), 0);
        assert_eq!(cache.promotions(), 2);
    }

    #[test]
    fn background_wins_threshold_race_then_demand_recheck_is_noop() {
        let cache = GpuExpertCache::new(100, 0.5, 3);
        assert_eq!(
            cache.demand_admission_preflight(7, 40),
            Ok(GpuDemandAdmissionPreflight::NeedsPayload)
        );
        assert!(cache.claim_promotion(7, 3));
        assert_eq!(
            cache.promote_hot_existing(7),
            GpuHotPromotionOutcome::PayloadRequired
        );
        let background_resident = gpu_res(7, 40);
        assert_eq!(
            cache.promote_hot_sync(background_resident.clone()),
            GpuHotPromotionOutcome::InstalledAnchor
        );
        let generation = cache.current_generation(7).unwrap();
        let next_generation = cache.inner.lock().next_generation;

        assert_eq!(cache.demand_admit_lru(gpu_res(7, 40)), Ok(false));
        let current = cache.current_admission(7).unwrap();
        assert_eq!(current.generation(), generation);
        assert!(Arc::ptr_eq(current.resident(), &background_resident));
        assert_eq!(cache.inner.lock().next_generation, next_generation);
        assert_eq!(cache.promotions(), 1);
        assert_eq!(cache.anchor_len(), 1);
        assert_eq!(cache.lru_len(), 0);
    }

    #[test]
    fn demand_readmission_allocates_a_newer_generation() {
        let cache = GpuExpertCache::new(20, 0.0, 0);
        assert_eq!(cache.demand_admit_lru(gpu_res(7, 10)), Ok(true));
        let first = cache.current_generation(7).unwrap();
        assert_eq!(cache.demand_admit_lru(gpu_res(8, 20)), Ok(true));
        assert!(!cache.contains(7));
        assert_eq!(cache.demand_admit_lru(gpu_res(7, 10)), Ok(true));
        let second = cache.current_generation(7).unwrap();
        assert!(second > first);
    }

    #[test]
    fn oversized_demand_fails_without_mutating_existing_state() {
        let cache = GpuExpertCache::new(100, 0.5, 0);
        assert_eq!(cache.demand_admit_lru(gpu_res(1, 25)), Ok(true));
        let generation = cache.current_generation(1).unwrap();
        let used = cache.used_bytes();
        let promotions = cache.promotions();

        assert_eq!(
            cache.demand_admit_lru(gpu_res(2, 51)),
            Err(GpuDemandAdmissionError::PayloadExceedsLruCapacity {
                bytes: 51,
                capacity: 50,
            })
        );
        assert!(cache.contains_generation(1, generation));
        assert!(!cache.contains(2));
        assert_eq!(cache.used_bytes(), used);
        assert_eq!(cache.promotions(), promotions);
    }

    #[test]
    fn generation_exhaustion_does_not_evict_existing_demand() {
        let cache = GpuExpertCache::new(20, 0.0, 0);
        assert_eq!(cache.demand_admit_lru(gpu_res(1, 10)), Ok(true));
        let generation = cache.current_generation(1).unwrap();
        cache.inner.lock().next_generation = u64::MAX;

        assert_eq!(
            cache.demand_admit_lru(gpu_res(2, 20)),
            Err(GpuDemandAdmissionError::GenerationExhausted)
        );
        assert!(cache.contains_generation(1, generation));
        assert!(!cache.contains(2));
        assert_eq!(cache.used_bytes(), 10);
    }

    #[test]
    fn get_retains_counter_and_lru_recency_behavior() {
        let cache = GpuExpertCache::new(20, 0.0, 0);
        assert!(cache.promote_sync(gpu_res(1, 10)));
        assert!(cache.promote_sync(gpu_res(2, 10)));

        assert!(matches!(cache.get(1), GpuLookup::LruHit(_)));
        assert!(matches!(cache.get(99), GpuLookup::Miss));
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);

        assert!(cache.promote_sync(gpu_res(3, 10)));
        assert!(cache.contains(1), "mutating get must make id 1 MRU");
        assert!(!cache.contains(2), "id 2 must remain the LRU victim");
        assert!(cache.contains(3));
    }

    #[test]
    fn logical_admission_generation_changes_only_after_eviction_and_readmission() {
        let cache = GpuExpertCache::new(16, 0.0, 0);
        assert!(cache.promote_sync(gpu_res(7, 8)));
        let g1 = cache.current_generation(7).expect("first admission generation");
        let hit_generation = match cache.get(7) {
            GpuLookup::LruHit(admission) => admission.generation(),
            _ => panic!("expected logical LRU hit"),
        };
        assert_eq!(hit_generation, g1, "ordinary hits preserve generation");

        assert!(cache.promote_sync(gpu_res(8, 16)));
        assert!(!cache.contains(7), "larger admission evicts old generation");
        assert!(cache.promote_sync(gpu_res(7, 8)));
        let g2 = cache.current_generation(7).expect("readmission generation");
        assert_ne!(g2, g1, "readmission must never reuse physical identity");
    }

    #[test]
    fn promotion_claim_rearms_after_logical_eviction_above_hit_threshold() {
        let cache = GpuExpertCache::new(16, 0.0, 3);
        assert!(cache.promote_sync(gpu_res(7, 8)));
        assert!(!cache.claim_promotion(7, 99), "admitted id needs no request");
        assert!(cache.promote_sync(gpu_res(8, 16)));
        assert!(!cache.contains(7));

        assert!(
            cache.claim_promotion(7, 99),
            "evicted id can claim again without a fresh RAM-hit edge"
        );
        assert!(!cache.claim_promotion(7, 100), "only one request may be pending");
        assert!(cache.promote_sync(gpu_res(7, 8)));
        assert!(cache.contains(7));
    }
}
