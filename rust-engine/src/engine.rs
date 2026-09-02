//! Top-level engine that wires the router, cache, buffer pool, predictive
//! loader, storage, and inference placeholder together.
//!
//! Responsibilities of [`Engine::generate`]:
//!
//! 1. Ask the router which experts a given token needs.
//! 2. For each required expert, look up the cache:
//!    * **Hit** — clone the `Arc<ExpertResident>` and pass it to inference.
//!    * **Miss** — acquire a buffer from the pool, dispatch a (real)
//!      io_uring read, install the resident in the cache, then run inference.
//! 3. Run the placeholder inference function on the bytes.
//! 4. Update the predictive Markov model with the observed transition.
//! 5. Speculatively kick off prefetches for the most likely next experts.
//! 6. Record per-token latency and emit structured tracing events.

use crate::aligned_buffer::AlignedBuffer;
use crate::backend::Backend as _;
use crate::buffer_pool::BufferPool;
use crate::expert_cache::{
    ExpertResident, GpuAdmission, GpuDemandAdmissionError, GpuDemandSetAdmission, GpuExpertCache,
    GpuHotPromotionOutcome, GpuResident,
};
use crate::gating::Router;
use crate::gpu_native_residency::{
    global_to_layer_local as gpu_native_global_to_layer_local, GpuNativeDemandExpert,
    GpuNativeResidencyPriority, GpuNativeSpeculativeInstall, GpuNativeSpeculativeProbe,
    GpuNativeTieredResidencyError, GpuNativeTieredResidencyManager,
    GpuNativeTieredResidencySnapshot,
};
use crate::inference::{
    combine_outputs, run_inference_bf16, run_inference_f16, run_inference_int8,
    run_inference_mixed_quant, run_inference_mxfp4, run_inference_q4_0, run_inference_q4_0_qmm,
    run_inference_q4k, run_inference_q4k_qmm, run_inference_q5k, run_inference_q6k,
    run_inference_q8_0_direct_with_timing, synth_hidden_state, uniform_scores,
    ExpertWeightsError, HiddenState, InferenceOutput, WeightDtype, Q4K_BLOCK_ELEMS, Q4_0_BLOCK_ELEMS,
    Q8_0_BLOCK_ELEMS,
};
use crate::io_provider::NvmeStorage;
use crate::metrics::Metrics;
use crate::multi_layer_cache::{MultiLayerCacheReservation, MultiLayerExpertCache};
use crate::router::{
    DecayWorkerHandle, LayeredExpertAffinity, LocalityMonitor, NeuralSpeculator, PredictiveLoader,
};
use dashmap::DashMap;
use hdrhistogram::Histogram;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tracing::{debug, info, warn};

// =====================================================================
// Persistent, page-aligned KV cache (Industrial Upgrade Task 2).
// =====================================================================

/// Default block alignment for [`AlignedKvCache`] backing storage. The
/// engine's NVMe path uses the same 4 KiB constant; sharing it here
/// keeps the KV bytes cheap to splice into a future `O_DIRECT`
/// snapshot path without re-allocating into a new aligned region.
pub const KV_CACHE_BLOCK_ALIGN: usize = 4096;

/// Decompose a **global** expert id into its `(layer, layer-local)`
/// pair given a layer-qualified geometry of `per_layer` experts each
/// (`global = layer * per_layer + local`). The inverse of
/// [`layer_local_to_global`]. Callers must ensure `per_layer > 0`.
#[inline]
fn global_to_layer_local(global: u32, per_layer: u32) -> (u32, u32) {
    (global / per_layer, global % per_layer)
}

/// Recompose a `(layer, layer-local)` pair into its **global** expert
/// id. The inverse of [`global_to_layer_local`].
#[inline]
fn layer_local_to_global(layer: u32, local: u32, per_layer: u32) -> u32 {
    layer * per_layer + local
}

/// **Persistent, page-aligned KV cache** complementing the per-layer
/// paged KV cache in `transformer.rs`. The transformer module's
/// `KvCache` is a `Vec`-backed paged cache used inside one model
/// forward pass; this is a *session-scoped*, contiguous, page-aligned
/// cache that survives across [`Engine::generate`] calls so a single
/// chat / completion request can decode many tokens without
/// recomputing the prefix on every call.
///
/// **Why page-aligned?** Backing the cache with [`AlignedBuffer`]
/// means a future "warm-restart" path can `pwrite(2)` the cache
/// straight to an `O_DIRECT` snapshot file without bouncing through
/// the kernel page cache. It also makes the K/V bytes cheap to share
/// with `io_uring`'s registered fixed buffers if the engine ever
/// pushes attention compute to a device queue.
///
/// **Rolling window.** When the cache fills its `window_tokens`
/// budget, `append` shifts the tail down by one slot and writes the
/// new K/V at the end. This bounds memory at
/// `2 * window_tokens * kv_dim * 4` bytes (per-instance) and
/// implements the same sliding-window attention pattern Mistral / the
/// real transformer use.
///
/// **Memory safety.** The underlying `AlignedBuffer` is owned and
/// `Drop`'s deallocate the page-aligned region. `zeroize()` overwrites
/// the bytes via a trivial `fill(0)` before `reset` — sufficient for
/// the engine's session-deletion path because the buffer is
/// immediately re-allocated on the next session.
pub struct AlignedKvCache {
    keys: AlignedBuffer,
    values: AlignedBuffer,
    /// Number of tokens currently resident.
    seq_len: usize,
    /// Token capacity of the rolling window. `0` means unbounded
    /// (the cache will refuse `append` once it's full instead of
    /// shifting).
    window_tokens: usize,
    /// Hidden dimension per K/V row.
    kv_dim: usize,
    /// **Dtype hint** describing the K/V row layout in memory. The
    /// cache itself always stores `f32` rows (which is what the
    /// candle-core attention path consumes), but the engine records
    /// this so the upstream attention block can confirm it matches
    /// the model's hidden-layer dtype and skip an unnecessary cast
    /// before the K·Vᵀ dot products. Defaults to
    /// [`WeightDtype::F32`] for backwards compatibility.
    kv_dtype: WeightDtype,
}

impl AlignedKvCache {
    /// Allocate a fresh cache that holds up to `window_tokens` K/V
    /// rows of `kv_dim` floats each, page-aligned to
    /// [`KV_CACHE_BLOCK_ALIGN`].
    ///
    /// Panics if `window_tokens == 0` or `kv_dim == 0`.
    pub fn new(window_tokens: usize, kv_dim: usize) -> Self {
        Self::with_dtype(window_tokens, kv_dim, WeightDtype::F32)
    }

    /// Allocate a fresh cache and tag it with the dtype the rest of
    /// the model's hidden-layer pipeline expects K/V rows to use.
    /// The storage layout is identical to [`Self::new`] (always
    /// `f32` on disk / in DRAM); the `dtype` is recorded so callers
    /// in the attention block can avoid redundant casts when the
    /// model is also `F32` and the dtype hint matches.
    pub fn with_dtype(window_tokens: usize, kv_dim: usize, dtype: WeightDtype) -> Self {
        assert!(window_tokens > 0, "window_tokens must be > 0");
        assert!(kv_dim > 0, "kv_dim must be > 0");
        let row_bytes = kv_dim * std::mem::size_of::<f32>();
        let raw = window_tokens * row_bytes;
        // Round up to the page alignment so AlignedBuffer's invariant
        // (size % align == 0) holds. The trailing pad bytes are
        // unused and never read; `seq_len` bounds every iteration.
        let padded = raw.div_ceil(KV_CACHE_BLOCK_ALIGN) * KV_CACHE_BLOCK_ALIGN;
        Self {
            keys: AlignedBuffer::new(padded, KV_CACHE_BLOCK_ALIGN),
            values: AlignedBuffer::new(padded, KV_CACHE_BLOCK_ALIGN),
            seq_len: 0,
            window_tokens,
            kv_dim,
            kv_dtype: dtype,
        }
    }

    /// Number of tokens currently resident.
    #[inline]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Window capacity in tokens.
    #[inline]
    pub fn window_tokens(&self) -> usize {
        self.window_tokens
    }

    /// Page-aligned base address of the key buffer (for `O_DIRECT`
    /// snapshot use). Always a multiple of [`KV_CACHE_BLOCK_ALIGN`].
    pub fn keys_ptr(&self) -> *const u8 {
        self.keys.as_slice().as_ptr()
    }

    /// Page-aligned base address of the value buffer.
    pub fn values_ptr(&self) -> *const u8 {
        self.values.as_slice().as_ptr()
    }

    /// Append one (k, v) row. If the cache is at capacity, the
    /// oldest token is evicted (rolling window) and the new row
    /// replaces it at the tail.
    ///
    /// Returns `true` when an eviction actually happened, `false`
    /// when the new row simply extended the resident window.
    ///
    /// Panics if either slice's length differs from `kv_dim`.
    pub fn append(&mut self, k: &[f32], v: &[f32]) -> bool {
        assert_eq!(
            k.len(),
            self.kv_dim,
            "AlignedKvCache::append: kv_dim mismatch"
        );
        assert_eq!(
            v.len(),
            self.kv_dim,
            "AlignedKvCache::append: kv_dim mismatch"
        );
        let evicted = if self.seq_len == self.window_tokens {
            self.shift_one_left();
            true
        } else {
            false
        };
        let pos = self.seq_len;
        self.write_row(pos, k, v);
        self.seq_len += 1;
        evicted
    }

    /// Read the i-th cached key (`i < seq_len`). Returns a slice of
    /// length `kv_dim` borrowed from the page-aligned backing store.
    pub fn key(&self, i: usize) -> &[f32] {
        assert!(i < self.seq_len, "AlignedKvCache::key: index out of bounds");
        let row = self.row_floats(self.keys.as_slice(), i);
        row
    }

    /// Read the i-th cached value.
    pub fn value(&self, i: usize) -> &[f32] {
        assert!(
            i < self.seq_len,
            "AlignedKvCache::value: index out of bounds"
        );
        self.row_floats(self.values.as_slice(), i)
    }

    /// Drop every resident token. The backing allocation is kept so
    /// the next `append` doesn't pay for a fresh page-aligned alloc.
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Overwrite every resident K/V byte with zero before [`Self::reset`]
    /// — the engine calls this before tearing down a session so the
    /// next allocation that lands in the same heap region cannot
    /// observe the previous tenant's attention state.
    pub fn zeroize(&mut self) {
        self.keys.as_mut_slice().fill(0);
        self.values.as_mut_slice().fill(0);
        self.reset();
    }

    /// Resident bytes (keys + values), useful for telemetry.
    pub fn resident_bytes(&self) -> usize {
        self.seq_len * self.kv_dim * std::mem::size_of::<f32>() * 2
    }

    fn write_row(&mut self, pos: usize, k: &[f32], v: &[f32]) {
        let row_bytes = self.kv_dim * std::mem::size_of::<f32>();
        let start = pos * row_bytes;
        let end = start + row_bytes;
        debug_assert_eq!(k.len(), self.kv_dim);
        debug_assert_eq!(v.len(), self.kv_dim);
        // SAFETY: writing bytes — the underlying AlignedBuffer is
        // initialised and we slice within bounds (pos < window_tokens
        // is guaranteed by append's eviction logic).
        let kb = &mut self.keys.as_mut_slice()[start..end];
        let vb = &mut self.values.as_mut_slice()[start..end];
        // SAFETY: this crate only supports little-endian targets, so the
        // in-memory representation of `[f32]` already matches the desired
        // serialized layout. `[f32]` is contiguous, and the produced byte
        // slices cover exactly `row_bytes`.
        let k_bytes = unsafe { std::slice::from_raw_parts(k.as_ptr() as *const u8, row_bytes) };
        let v_bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, row_bytes) };
        kb.copy_from_slice(k_bytes);
        vb.copy_from_slice(v_bytes);
    }

    fn row_floats<'a>(&'a self, buf: &'a [u8], pos: usize) -> &'a [f32] {
        let row_bytes = self.kv_dim * std::mem::size_of::<f32>();
        let start = pos * row_bytes;
        let bytes = &buf[start..start + row_bytes];
        // SAFETY: AlignedBuffer is allocated with `KV_CACHE_BLOCK_ALIGN`
        // (4096-byte) alignment, so every per-row offset is a multiple
        // of `4 = align_of::<f32>()`. The byte length is exactly
        // `kv_dim * 4`, and `f32` has no validity invariants beyond
        // alignment. The two `debug_assert!`s below check those
        // invariants in debug builds (gist feedback #1.7) so a future
        // refactor that violates them fails loudly rather than
        // returning a misaligned / mis-sized slice.
        debug_assert_eq!(
            bytes.len(),
            self.kv_dim * std::mem::size_of::<f32>(),
            "row_floats: byte slice length must be exactly kv_dim * 4"
        );
        debug_assert_eq!(
            (bytes.as_ptr() as usize) % std::mem::align_of::<f32>(),
            0,
            "row_floats: byte slice pointer must be 4-byte aligned for f32"
        );
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.kv_dim) }
    }

    /// Shift the K/V rows one slot toward index 0. Called by `append`
    /// when the rolling window is full. `O(seq_len * kv_dim)` byte
    /// moves; cheap relative to one attention sweep.
    fn shift_one_left(&mut self) {
        if self.seq_len == 0 {
            return;
        }
        let row_bytes = self.kv_dim * std::mem::size_of::<f32>();
        let live_bytes = (self.seq_len - 1) * row_bytes;
        if live_bytes > 0 {
            let kb = self.keys.as_mut_slice();
            let vb = self.values.as_mut_slice();
            kb.copy_within(row_bytes..row_bytes + live_bytes, 0);
            vb.copy_within(row_bytes..row_bytes + live_bytes, 0);
        }
        // Always reflect the eviction in `seq_len`. The early-return
        // branch above only skips the memcpy when there's nothing
        // live to keep (window_tokens == 1); the slot count still has
        // to decrement so the next `append` writes at row 0 and the
        // window cap is never exceeded.
        self.seq_len -= 1;
    }
}

/// Internal: outcome of a single fetch attempt.
enum FetchOnceError {
    /// The buffer pool was exhausted for so long that we hit the
    /// MAX_FETCH_YIELDS cap. Surface to the caller so it can return
    /// 503 / NotReady rather than degrade into an unbounded busy-loop.
    PoolStarved,
    /// The storage layer returned an I/O error. The retry loop in
    /// [`Engine::fetch_with_retry`] may choose to try again.
    Io(String),
}

/// Public error type for [`Engine::fetch_with_retry`].
///
/// The legacy [`Engine::fetch`] keeps its prior crashing semantics —
/// the synthetic benchmark / `Engine::generate` path has no upstream
/// "skip this expert" path. The real-transformer path uses
/// `fetch_with_retry` (via [`Engine::moe_step`]) so a single corrupt
/// expert downgrades gracefully into a missing top-K member rather
/// than killing the server.
#[derive(Clone, Debug, PartialEq)]
pub enum ExpertReadError {
    /// Storage returned a (possibly transient) I/O error every attempt.
    Io {
        id: u32,
        attempts: usize,
        source: String,
    },
    /// Buffer pool starved for too long — likely a configuration bug
    /// (more pinned experts than the pool can keep resident, or way
    /// more concurrent requests than expected).
    PoolStarved { id: u32 },
}

impl std::fmt::Display for ExpertReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExpertReadError::Io {
                id,
                attempts,
                source,
            } => write!(
                f,
                "expert {id} read failed after {attempts} attempts: {source}"
            ),
            ExpertReadError::PoolStarved { id } => write!(
                f,
                "expert {id} fetch starved: buffer pool exhausted with cache pinned",
            ),
        }
    }
}

impl std::error::Error for ExpertReadError {}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum GpuNativeDemandResidencyError {
    ManagerNotInstalled,
    QualificationIsolationViolation,
    QualificationCompletedReadSetMismatch {
        requested_global_ids: Vec<u32>,
        completed_global_ids: Vec<u32>,
    },
    ProductionBatchCommitViolation {
        global_id: u32,
    },
    ProductionBatchPoolUnavailableAfterReservation {
        requested: usize,
        acquired: usize,
    },
    ProductionBatchReadFailedAfterReservation {
        global_ids: Vec<u32>,
        source: String,
    },
    ExpertRead(ExpertReadError),
    LogicalAdmission(crate::backend::GpuExpertDispatchError),
    LogicalDemandSet(GpuDemandAdmissionError),
    LogicalDemandSetRecoveryExhausted {
        global_ids: Vec<u32>,
        attempts: usize,
    },
    PhysicalDemandRecoveryExhausted {
        global_id: u32,
        recovery_attempts: usize,
    },
    Tiered(GpuNativeTieredResidencyError),
}

impl std::fmt::Display for GpuNativeDemandResidencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ManagerNotInstalled => {
                f.write_str("GPU-native tiered residency manager is not installed")
            }
            Self::QualificationIsolationViolation => f.write_str(
                "qualification-only exact-demand source observed overlapping foreground demand sets",
            ),
            Self::QualificationCompletedReadSetMismatch {
                requested_global_ids,
                completed_global_ids,
            } => write!(
                f,
                "qualification-only exact-demand completed-read set {completed_global_ids:?} did not match requested set {requested_global_ids:?}"
            ),
            Self::ProductionBatchCommitViolation { global_id } => write!(
                f,
                "production exact-demand reserved cache commit invariant failed for expert {global_id}"
            ),
            Self::ProductionBatchPoolUnavailableAfterReservation {
                requested,
                acquired,
            } => write!(
                f,
                "production exact-demand batch acquired only {acquired} of {requested} primary buffers after committing its cache victim schedule; sequential fallback is unsafe"
            ),
            Self::ProductionBatchReadFailedAfterReservation { global_ids, source } => write!(
                f,
                "production exact-demand batch read for experts {global_ids:?} failed after committing its cache victim schedule; sequential fallback is unsafe: {source}"
            ),
            Self::ExpertRead(error) => write!(f, "tiered residency fetch failed: {error}"),
            Self::LogicalAdmission(error) => {
                write!(f, "tiered residency logical admission failed: {error}")
            }
            Self::LogicalDemandSet(error) => {
                write!(f, "tiered residency logical demand-set admission failed: {error}")
            }
            Self::LogicalDemandSetRecoveryExhausted {
                global_ids,
                attempts,
            } => write!(
                f,
                "tiered residency logical demand-set source resolution remained incomplete for experts {global_ids:?} after {attempts} bounded attempts"
            ),
            Self::PhysicalDemandRecoveryExhausted {
                global_id,
                recovery_attempts,
            } => write!(
                f,
                "tiered residency physical demand-set recovery remained stale/missing for expert {global_id} after {recovery_attempts} bounded recovery attempt(s)"
            ),
            Self::Tiered(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for GpuNativeDemandResidencyError {}

impl From<ExpertReadError> for GpuNativeDemandResidencyError {
    fn from(value: ExpertReadError) -> Self {
        Self::ExpertRead(value)
    }
}

impl From<GpuNativeTieredResidencyError> for GpuNativeDemandResidencyError {
    fn from(value: GpuNativeTieredResidencyError) -> Self {
        Self::Tiered(value)
    }
}

impl From<GpuDemandAdmissionError> for GpuNativeDemandResidencyError {
    fn from(value: GpuDemandAdmissionError) -> Self {
        Self::LogicalDemandSet(value)
    }
}

/// Explicit arm of the PR2-A exact-demand source qualification. This is not
/// part of [`EngineOptions`] or TOML configuration: only the dedicated
/// qualifier may install it on a fresh isolated runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GpuNativeDemandSourceQualificationArm {
    Control,
    Treatment,
}

/// Cumulative production-path evidence. These atomics are always present and
/// add no per-token logging; the dedicated v2 qualifier resets them between
/// warmup and measured arms and records exact snapshots.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProductionDemandSourceSnapshot {
    pub(crate) ordinary_production_path_exercised: bool,
    pub(crate) production_source_sets: u64,
    pub(crate) production_batch_eligible_sets: u64,
    pub(crate) production_batch_attempts: u64,
    pub(crate) production_batch_successes: u64,
    pub(crate) production_batch_experts: u64,
    pub(crate) production_sequential_fallback_mixed_ram: u64,
    pub(crate) production_sequential_fallback_single_item: u64,
    pub(crate) production_sequential_fallback_singleflight_contention: u64,
    pub(crate) production_sequential_fallback_reservation: u64,
    pub(crate) production_sequential_fallback_pool: u64,
    pub(crate) production_sequential_fallback_batch_read_error: u64,
    pub(crate) production_batch_width_min: u64,
    pub(crate) production_batch_width_max: u64,
    pub(crate) production_batch_width_mean: f64,
    pub(crate) production_singleflight_ids_claimed: u64,
    pub(crate) production_singleflight_claim_rollbacks: u64,
    pub(crate) production_singleflight_followers_observed: u64,
    pub(crate) production_cache_slots_reserved: u64,
    pub(crate) production_cache_reservations_consumed: u64,
    pub(crate) production_cache_reservations_released: u64,
    pub(crate) production_cache_reservation_leaks: u64,
    pub(crate) production_batch_commit_violations: u64,
    pub(crate) stale_singleflight_entries: u64,
}

#[derive(Default)]
struct ProductionDemandSourceTelemetry {
    source_sets: AtomicU64,
    batch_eligible_sets: AtomicU64,
    batch_attempts: AtomicU64,
    batch_successes: AtomicU64,
    batch_experts: AtomicU64,
    fallback_mixed_ram: AtomicU64,
    fallback_single_item: AtomicU64,
    fallback_singleflight_contention: AtomicU64,
    fallback_reservation: AtomicU64,
    batch_width_min: AtomicU64,
    batch_width_max: AtomicU64,
    batch_width_sum: AtomicU64,
    singleflight_ids_claimed: AtomicU64,
    singleflight_claim_rollbacks: AtomicU64,
    singleflight_followers_observed: AtomicU64,
    cache_slots_reserved: AtomicU64,
    cache_reservations_consumed: AtomicU64,
    cache_reservations_released: AtomicU64,
    batch_commit_violations: AtomicU64,
}

impl ProductionDemandSourceTelemetry {
    fn reset(&self) {
        self.source_sets.store(0, Ordering::Relaxed);
        self.batch_eligible_sets.store(0, Ordering::Relaxed);
        self.batch_attempts.store(0, Ordering::Relaxed);
        self.batch_successes.store(0, Ordering::Relaxed);
        self.batch_experts.store(0, Ordering::Relaxed);
        self.fallback_mixed_ram.store(0, Ordering::Relaxed);
        self.fallback_single_item.store(0, Ordering::Relaxed);
        self.fallback_singleflight_contention
            .store(0, Ordering::Relaxed);
        self.fallback_reservation.store(0, Ordering::Relaxed);
        self.batch_width_min.store(u64::MAX, Ordering::Relaxed);
        self.batch_width_max.store(0, Ordering::Relaxed);
        self.batch_width_sum.store(0, Ordering::Relaxed);
        self.singleflight_ids_claimed.store(0, Ordering::Relaxed);
        self.singleflight_claim_rollbacks
            .store(0, Ordering::Relaxed);
        self.singleflight_followers_observed
            .store(0, Ordering::Relaxed);
        self.cache_slots_reserved.store(0, Ordering::Relaxed);
        self.cache_reservations_consumed.store(0, Ordering::Relaxed);
        self.cache_reservations_released.store(0, Ordering::Relaxed);
        self.batch_commit_violations.store(0, Ordering::Relaxed);
    }

    fn record_success(&self, width: usize) {
        let width = width as u64;
        self.batch_successes.fetch_add(1, Ordering::Relaxed);
        self.batch_experts.fetch_add(width, Ordering::Relaxed);
        self.batch_width_min.fetch_min(width, Ordering::Relaxed);
        self.batch_width_max.fetch_max(width, Ordering::Relaxed);
        self.batch_width_sum.fetch_add(width, Ordering::Relaxed);
    }

    fn snapshot(
        &self,
        cache_reservation_leaks: usize,
        stale_singleflight_entries: usize,
    ) -> ProductionDemandSourceSnapshot {
        let successes = self.batch_successes.load(Ordering::Relaxed);
        let width_min = self.batch_width_min.load(Ordering::Relaxed);
        ProductionDemandSourceSnapshot {
            ordinary_production_path_exercised: self.source_sets.load(Ordering::Relaxed) > 0,
            production_source_sets: self.source_sets.load(Ordering::Relaxed),
            production_batch_eligible_sets: self.batch_eligible_sets.load(Ordering::Relaxed),
            production_batch_attempts: self.batch_attempts.load(Ordering::Relaxed),
            production_batch_successes: successes,
            production_batch_experts: self.batch_experts.load(Ordering::Relaxed),
            production_sequential_fallback_mixed_ram: self
                .fallback_mixed_ram
                .load(Ordering::Relaxed),
            production_sequential_fallback_single_item: self
                .fallback_single_item
                .load(Ordering::Relaxed),
            production_sequential_fallback_singleflight_contention: self
                .fallback_singleflight_contention
                .load(Ordering::Relaxed),
            production_sequential_fallback_reservation: self
                .fallback_reservation
                .load(Ordering::Relaxed),
            // Retained in schema v2, but fail-closed post-reservation failures
            // never take a sequential fallback.
            production_sequential_fallback_pool: 0,
            production_sequential_fallback_batch_read_error: 0,
            production_batch_width_min: if successes == 0 || width_min == u64::MAX {
                0
            } else {
                width_min
            },
            production_batch_width_max: self.batch_width_max.load(Ordering::Relaxed),
            production_batch_width_mean: if successes == 0 {
                0.0
            } else {
                self.batch_width_sum.load(Ordering::Relaxed) as f64 / successes as f64
            },
            production_singleflight_ids_claimed: self
                .singleflight_ids_claimed
                .load(Ordering::Relaxed),
            production_singleflight_claim_rollbacks: self
                .singleflight_claim_rollbacks
                .load(Ordering::Relaxed),
            production_singleflight_followers_observed: self
                .singleflight_followers_observed
                .load(Ordering::Relaxed),
            production_cache_slots_reserved: self.cache_slots_reserved.load(Ordering::Relaxed),
            production_cache_reservations_consumed: self
                .cache_reservations_consumed
                .load(Ordering::Relaxed),
            production_cache_reservations_released: self
                .cache_reservations_released
                .load(Ordering::Relaxed),
            production_cache_reservation_leaks: cache_reservation_leaks as u64,
            production_batch_commit_violations: self
                .batch_commit_violations
                .load(Ordering::Relaxed),
            stale_singleflight_entries: stale_singleflight_entries as u64,
        }
    }
}

/// Low-overhead, qualification-only source and cache evidence. All counters
/// are cumulative since the most recent explicit qualification reset.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct GpuNativeDemandSourceQualificationSnapshot {
    pub(crate) arm: GpuNativeDemandSourceQualificationArm,
    pub(crate) qualification_only: bool,
    pub(crate) production_demand_source_changed: bool,
    pub(crate) single_request_stream: bool,
    pub(crate) overlapping_demand_sets: u64,
    pub(crate) primary_pool_capacity: usize,
    pub(crate) shadow_pool_capacity: usize,
    pub(crate) demand_sets: u64,
    pub(crate) physical_missing_experts: u64,
    pub(crate) physical_probe_us: u64,
    pub(crate) source_sets: u64,
    pub(crate) source_experts: u64,
    pub(crate) demand_source_requests: u64,
    pub(crate) source_ram_hits: u64,
    pub(crate) source_ram_misses: u64,
    pub(crate) source_nvme_reads: u64,
    pub(crate) source_nvme_bytes: u64,
    pub(crate) source_set_width_min: u64,
    pub(crate) source_set_width_max: u64,
    pub(crate) source_set_width_mean: f64,
    pub(crate) source_acquisition_wall_us: u64,
    pub(crate) sum_individual_source_service_us: Option<u64>,
    pub(crate) logical_demand_admission_us: u64,
    pub(crate) physical_demand_install_us: u64,
    pub(crate) total_residency_service_us: u64,
    pub(crate) batch_eligible_all_ram_miss_source_sets: u64,
    pub(crate) sequential_fallback_source_sets_due_ram_residency: u64,
    pub(crate) batch_eligible_source_experts: u64,
    pub(crate) sequential_fallback_source_experts: u64,
    pub(crate) batch_path_exercises: u64,
    pub(crate) concurrent_source_reads: u64,
    pub(crate) actual_batch_nvme_width_min: u64,
    pub(crate) actual_batch_nvme_width_max: u64,
    pub(crate) actual_batch_nvme_width_mean: f64,
    pub(crate) mixed_ram_state_batch_attempts: u64,
    pub(crate) pool_capacity_failures: u64,
    pub(crate) batch_read_failures: u64,
    pub(crate) ordered_commit_violations: u64,
    pub(crate) ram_cache_source_hits: u64,
    pub(crate) ram_cache_source_misses: u64,
    pub(crate) ram_cache_inserts: u64,
    pub(crate) ram_cache_evictions: u64,
    pub(crate) selected_route_ids_sha256: String,
    pub(crate) physical_missing_ids_sha256: String,
    pub(crate) demand_source_request_ids_sha256: String,
    pub(crate) demand_ram_insert_ids_sha256: String,
    pub(crate) demand_ram_eviction_ids_sha256: String,
    pub(crate) deterministic_cache_commit_reconciliation: bool,
}

#[derive(Clone)]
struct QualificationOrderedHasher {
    inner: Sha256,
}

impl Default for QualificationOrderedHasher {
    fn default() -> Self {
        Self {
            inner: Sha256::new(),
        }
    }
}

impl QualificationOrderedHasher {
    fn record_set(&mut self, ids: &[u32]) {
        self.inner.update([0x53]);
        self.inner.update((ids.len() as u64).to_le_bytes());
        for &id in ids {
            self.inner.update(id.to_le_bytes());
        }
    }

    fn record_id(&mut self, id: u32) {
        self.inner.update([0x49]);
        self.inner.update(id.to_le_bytes());
    }

    fn hex(&self) -> String {
        format!("{:x}", self.inner.clone().finalize())
    }
}

/// Normalize completed source reads into original request order before any
/// cache mutation. The completed map is deliberately unordered, so neither
/// storage-worker completion order nor map iteration order can become LRU
/// insertion order.
fn qualification_order_completed_residents(
    request_order: &[u32],
    mut completed: HashMap<u32, Arc<ExpertResident>>,
) -> Result<Vec<(u32, Arc<ExpertResident>)>, GpuNativeDemandResidencyError> {
    fn completed_set(completed: &HashMap<u32, Arc<ExpertResident>>) -> Vec<u32> {
        let mut ids = completed.keys().copied().collect::<Vec<_>>();
        ids.sort_unstable();
        ids
    }
    if completed.len() != request_order.len()
        || request_order.iter().any(|id| !completed.contains_key(id))
    {
        return Err(
            GpuNativeDemandResidencyError::QualificationCompletedReadSetMismatch {
                requested_global_ids: request_order.to_vec(),
                completed_global_ids: completed_set(&completed),
            },
        );
    }
    let mut ordered = Vec::with_capacity(request_order.len());
    for &global_id in request_order {
        let resident = completed.remove(&global_id).ok_or_else(|| {
            GpuNativeDemandResidencyError::QualificationCompletedReadSetMismatch {
                requested_global_ids: request_order.to_vec(),
                completed_global_ids: completed_set(&completed),
            }
        })?;
        ordered.push((global_id, resident));
    }
    if !completed.is_empty() {
        return Err(
            GpuNativeDemandResidencyError::QualificationCompletedReadSetMismatch {
                requested_global_ids: request_order.to_vec(),
                completed_global_ids: completed_set(&completed),
            },
        );
    }
    Ok(ordered)
}

struct GpuNativeDemandSourceQualification {
    arm: GpuNativeDemandSourceQualificationArm,
    primary_pool_capacity: usize,
    shadow_pool_capacity: usize,
    active_demand_set: AtomicBool,
    overlapping_demand_sets: AtomicU64,
    demand_sets: AtomicU64,
    physical_missing_experts: AtomicU64,
    physical_probe_us: AtomicU64,
    source_sets: AtomicU64,
    source_experts: AtomicU64,
    demand_source_requests: AtomicU64,
    source_ram_hits: AtomicU64,
    source_ram_misses: AtomicU64,
    source_nvme_reads: AtomicU64,
    source_nvme_bytes: AtomicU64,
    source_set_width_min: AtomicU64,
    source_set_width_max: AtomicU64,
    source_set_width_sum: AtomicU64,
    source_acquisition_wall_us: AtomicU64,
    individual_source_service_us: AtomicU64,
    logical_demand_admission_us: AtomicU64,
    physical_demand_install_us: AtomicU64,
    total_residency_service_us: AtomicU64,
    ram_cache_inserts: AtomicU64,
    ram_cache_evictions: AtomicU64,
    selected_route_ids: parking_lot::Mutex<QualificationOrderedHasher>,
    physical_missing_ids: parking_lot::Mutex<QualificationOrderedHasher>,
    demand_source_request_ids: parking_lot::Mutex<QualificationOrderedHasher>,
    demand_ram_insert_ids: parking_lot::Mutex<QualificationOrderedHasher>,
    demand_ram_eviction_ids: parking_lot::Mutex<QualificationOrderedHasher>,
}

impl GpuNativeDemandSourceQualification {
    fn new(
        arm: GpuNativeDemandSourceQualificationArm,
        primary_pool_capacity: usize,
        shadow_pool_capacity: usize,
    ) -> Self {
        Self {
            arm,
            primary_pool_capacity,
            shadow_pool_capacity,
            active_demand_set: AtomicBool::new(false),
            overlapping_demand_sets: AtomicU64::new(0),
            demand_sets: AtomicU64::new(0),
            physical_missing_experts: AtomicU64::new(0),
            physical_probe_us: AtomicU64::new(0),
            source_sets: AtomicU64::new(0),
            source_experts: AtomicU64::new(0),
            demand_source_requests: AtomicU64::new(0),
            source_ram_hits: AtomicU64::new(0),
            source_ram_misses: AtomicU64::new(0),
            source_nvme_reads: AtomicU64::new(0),
            source_nvme_bytes: AtomicU64::new(0),
            source_set_width_min: AtomicU64::new(u64::MAX),
            source_set_width_max: AtomicU64::new(0),
            source_set_width_sum: AtomicU64::new(0),
            source_acquisition_wall_us: AtomicU64::new(0),
            individual_source_service_us: AtomicU64::new(0),
            logical_demand_admission_us: AtomicU64::new(0),
            physical_demand_install_us: AtomicU64::new(0),
            total_residency_service_us: AtomicU64::new(0),
            ram_cache_inserts: AtomicU64::new(0),
            ram_cache_evictions: AtomicU64::new(0),
            selected_route_ids: parking_lot::Mutex::new(QualificationOrderedHasher::default()),
            physical_missing_ids: parking_lot::Mutex::new(QualificationOrderedHasher::default()),
            demand_source_request_ids: parking_lot::Mutex::new(
                QualificationOrderedHasher::default(),
            ),
            demand_ram_insert_ids: parking_lot::Mutex::new(
                QualificationOrderedHasher::default(),
            ),
            demand_ram_eviction_ids: parking_lot::Mutex::new(
                QualificationOrderedHasher::default(),
            ),
        }
    }

    fn snapshot(&self) -> GpuNativeDemandSourceQualificationSnapshot {
        let source_sets = self.source_sets.load(Ordering::Relaxed);
        let width_min = self.source_set_width_min.load(Ordering::Relaxed);
        GpuNativeDemandSourceQualificationSnapshot {
            arm: self.arm,
            qualification_only: false,
            production_demand_source_changed: true,
            single_request_stream: self.overlapping_demand_sets.load(Ordering::Relaxed) == 0,
            overlapping_demand_sets: self.overlapping_demand_sets.load(Ordering::Relaxed),
            primary_pool_capacity: self.primary_pool_capacity,
            shadow_pool_capacity: self.shadow_pool_capacity,
            demand_sets: self.demand_sets.load(Ordering::Relaxed),
            physical_missing_experts: self.physical_missing_experts.load(Ordering::Relaxed),
            physical_probe_us: self.physical_probe_us.load(Ordering::Relaxed),
            source_sets,
            source_experts: self.source_experts.load(Ordering::Relaxed),
            demand_source_requests: self.demand_source_requests.load(Ordering::Relaxed),
            source_ram_hits: self.source_ram_hits.load(Ordering::Relaxed),
            source_ram_misses: self.source_ram_misses.load(Ordering::Relaxed),
            source_nvme_reads: self.source_nvme_reads.load(Ordering::Relaxed),
            source_nvme_bytes: self.source_nvme_bytes.load(Ordering::Relaxed),
            source_set_width_min: if source_sets == 0 || width_min == u64::MAX {
                0
            } else {
                width_min
            },
            source_set_width_max: self.source_set_width_max.load(Ordering::Relaxed),
            source_set_width_mean: if source_sets == 0 {
                0.0
            } else {
                self.source_set_width_sum.load(Ordering::Relaxed) as f64 / source_sets as f64
            },
            source_acquisition_wall_us: self
                .source_acquisition_wall_us
                .load(Ordering::Relaxed),
            sum_individual_source_service_us: (self.arm
                == GpuNativeDemandSourceQualificationArm::Control)
                .then(|| self.individual_source_service_us.load(Ordering::Relaxed)),
            logical_demand_admission_us: self
                .logical_demand_admission_us
                .load(Ordering::Relaxed),
            physical_demand_install_us: self.physical_demand_install_us.load(Ordering::Relaxed),
            total_residency_service_us: self.total_residency_service_us.load(Ordering::Relaxed),
            // Schema-v2 compatibility fields from the retired v1 algorithm.
            // The production qualifier never populated them.
            batch_eligible_all_ram_miss_source_sets: 0,
            sequential_fallback_source_sets_due_ram_residency: 0,
            batch_eligible_source_experts: 0,
            sequential_fallback_source_experts: 0,
            batch_path_exercises: 0,
            concurrent_source_reads: 0,
            actual_batch_nvme_width_min: 0,
            actual_batch_nvme_width_max: 0,
            actual_batch_nvme_width_mean: 0.0,
            mixed_ram_state_batch_attempts: 0,
            pool_capacity_failures: 0,
            batch_read_failures: 0,
            ordered_commit_violations: 0,
            ram_cache_source_hits: self.source_ram_hits.load(Ordering::Relaxed),
            ram_cache_source_misses: self.source_ram_misses.load(Ordering::Relaxed),
            ram_cache_inserts: self.ram_cache_inserts.load(Ordering::Relaxed),
            ram_cache_evictions: self.ram_cache_evictions.load(Ordering::Relaxed),
            selected_route_ids_sha256: self.selected_route_ids.lock().hex(),
            physical_missing_ids_sha256: self.physical_missing_ids.lock().hex(),
            demand_source_request_ids_sha256: self.demand_source_request_ids.lock().hex(),
            demand_ram_insert_ids_sha256: self.demand_ram_insert_ids.lock().hex(),
            demand_ram_eviction_ids_sha256: self.demand_ram_eviction_ids.lock().hex(),
            deterministic_cache_commit_reconciliation: true,
        }
    }

    fn record_source_set(&self, ids: &[u32]) {
        let width = ids.len() as u64;
        self.source_sets.fetch_add(1, Ordering::Relaxed);
        self.source_experts.fetch_add(width, Ordering::Relaxed);
        self.source_set_width_min.fetch_min(width, Ordering::Relaxed);
        self.source_set_width_max.fetch_max(width, Ordering::Relaxed);
        self.source_set_width_sum.fetch_add(width, Ordering::Relaxed);
    }

    fn record_source_request(&self, id: u32) {
        self.demand_source_requests.fetch_add(1, Ordering::Relaxed);
        self.demand_source_request_ids.lock().record_id(id);
    }

    fn record_cache_insert(&self, id: u32) {
        self.ram_cache_inserts.fetch_add(1, Ordering::Relaxed);
        self.demand_ram_insert_ids.lock().record_id(id);
    }

    fn record_cache_eviction(&self, id: u32) {
        self.ram_cache_evictions.fetch_add(1, Ordering::Relaxed);
        self.demand_ram_eviction_ids.lock().record_id(id);
    }

}

fn qualification_elapsed_us(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

struct QualificationDemandServiceGuard {
    state: Arc<GpuNativeDemandSourceQualification>,
    started: Instant,
}

impl QualificationDemandServiceGuard {
    fn enter(
        state: Arc<GpuNativeDemandSourceQualification>,
    ) -> Result<Self, GpuNativeDemandResidencyError> {
        if state
            .active_demand_set
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            state
                .overlapping_demand_sets
                .fetch_add(1, Ordering::Relaxed);
            return Err(GpuNativeDemandResidencyError::QualificationIsolationViolation);
        }
        state.demand_sets.fetch_add(1, Ordering::Relaxed);
        Ok(Self {
            state,
            started: Instant::now(),
        })
    }
}

impl Drop for QualificationDemandServiceGuard {
    fn drop(&mut self) {
        self.state
            .total_residency_service_us
            .fetch_add(qualification_elapsed_us(self.started), Ordering::Relaxed);
        self.state
            .active_demand_set
            .store(false, Ordering::Release);
    }
}

const GPU_NATIVE_LOGICAL_DEMAND_SET_ATTEMPTS: usize = 2;
const GPU_NATIVE_PHYSICAL_DEMAND_RECOVERY_LIMIT: usize = 1;

#[inline]
const fn gpu_native_physical_demand_recovery_allowed(completed_attempts: usize) -> bool {
    completed_attempts < GPU_NATIVE_PHYSICAL_DEMAND_RECOVERY_LIMIT
}

fn gpu_native_physical_missing_ids(global_ids: &[u32], physical_current: &[bool]) -> Vec<u32> {
    assert_eq!(
        global_ids.len(),
        physical_current.len(),
        "physical probe results must cover the complete selected set"
    );
    global_ids
        .iter()
        .copied()
        .zip(physical_current.iter().copied())
        .filter_map(|(global_id, current)| (!current).then_some(global_id))
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeResidencyInstallError {
    LegacyPhysicalExpertPlaneActive,
    LogicalCacheMismatch,
    AlreadyInstalled,
}

impl std::fmt::Display for GpuNativeResidencyInstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LegacyPhysicalExpertPlaneActive => f.write_str(
                "GPU-native residency cannot be installed while the legacy routed-expert GPU plane is active",
            ),
            Self::LogicalCacheMismatch => f.write_str(
                "GPU-native residency manager does not share the execution context's logical GpuExpertCache",
            ),
            Self::AlreadyInstalled => {
                f.write_str("GPU-native tiered residency manager is already installed")
            }
        }
    }
}

impl std::error::Error for GpuNativeResidencyInstallError {}

/// RAII guard that ensures the in-flight singleflight slot for an
/// expert id is freed (and any waiters notified) when the leader's
/// fetch attempt finishes — success, failure, or panic. See
/// [`Engine::fetch_with_retry`] for the algorithm; this guard keeps
/// the cleanup logic on every exit path so a panicking I/O task
/// cannot wedge a stale entry in `Engine::in_flight`.
struct SingleflightLeaderGuard {
    map: Arc<DashMap<u32, Arc<Notify>>>,
    id: u32,
    notify: Arc<Notify>,
    /// When `false` the guard is a no-op; constructing it on the
    /// follower path keeps the call site identical between leaders
    /// and followers without spurious notifications.
    armed: bool,
}

impl SingleflightLeaderGuard {
    fn try_claim(map: Arc<DashMap<u32, Arc<Notify>>>, id: u32) -> Result<Self, Arc<Notify>> {
        match map.entry(id) {
            dashmap::mapref::entry::Entry::Occupied(occupied) => Err(occupied.get().clone()),
            dashmap::mapref::entry::Entry::Vacant(vacant) => {
                let notify = Arc::new(Notify::new());
                vacant.insert(notify.clone());
                Ok(Self {
                    map: map.clone(),
                    id,
                    notify,
                    armed: true,
                })
            }
        }
    }
}

impl Drop for SingleflightLeaderGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // Remove the entry first so any caller landing *after* the
        // notify_waiters() call below sees a fresh slot to fill.
        // Remove only our own notification identity. This guards against a
        // future protocol refactor replacing an entry before an older guard
        // drops; an old guard must never delete a newer leader's claim.
        if let dashmap::mapref::entry::Entry::Occupied(occupied) = self.map.entry(self.id) {
            if Arc::ptr_eq(occupied.get(), &self.notify) {
                occupied.remove();
            }
        }
        // Wake every follower that parked on this id. They will
        // re-check the cache and either return a hit (the common
        // case) or fall through to their own fetch.
        self.notify.notify_waiters();
    }
}

/// All-or-nothing leadership over one exact expert-id set. Partially acquired
/// claims are ordinary [`SingleflightLeaderGuard`]s, so contention,
/// cancellation, error, and unwind all remove entries and notify followers.
struct MultiIdSingleflightLeadership {
    guards: Vec<SingleflightLeaderGuard>,
    telemetry: Arc<ProductionDemandSourceTelemetry>,
    successful: bool,
}

impl MultiIdSingleflightLeadership {
    fn finish(mut self) {
        self.successful = true;
    }
}

impl Drop for MultiIdSingleflightLeadership {
    fn drop(&mut self) {
        if !self.successful {
            self.telemetry
                .singleflight_claim_rollbacks
                .fetch_add(self.guards.len() as u64, Ordering::Relaxed);
        }
        // Fields drop after this body; each guard performs the authoritative
        // map removal followed by follower notification.
    }
}

/// Telemetry-aware wrapper around the per-layer cache reservation. The inner
/// guard owns the actual positions; this wrapper only accounts consumption
/// and unused release without changing its safety protocol.
struct ProductionCacheReservation {
    inner: MultiLayerCacheReservation,
    telemetry: Arc<ProductionDemandSourceTelemetry>,
}

impl ProductionCacheReservation {
    fn commit(&mut self, resident: Arc<ExpertResident>) -> Result<bool, Arc<ExpertResident>> {
        let cached = self.inner.commit(resident)?;
        if cached {
            self.telemetry
                .cache_reservations_consumed
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(cached)
    }
}

impl Drop for ProductionCacheReservation {
    fn drop(&mut self) {
        let unused = self.inner.remaining();
        if unused > 0 {
            self.telemetry
                .cache_reservations_released
                .fetch_add(unused as u64, Ordering::Relaxed);
        }
        // `inner` drops next and releases the actual positions.
    }
}

/// Boot-time engine error reserved for startup-only invariant checks
/// the synchronous `Engine::new` constructor doesn't perform.
#[derive(Debug)]
pub enum EngineError {
    /// The cold-start manifest observed at least two experts whose
    /// Unified Tensor Header declared **different** weight dtypes.
    /// Surfaces [`crate::io_provider::IncompatibleExpertTypes`] —
    /// the engine refuses to dispatch against a heterogeneous set
    /// of experts because a single quant scheme is wired into the
    /// per-token math kernel.
    IncompatibleExpertTypes(crate::io_provider::IncompatibleExpertTypes),
    /// The manifest indexed experts whose unique on-disk dtype does
    /// not match the engine's configured `WeightDtype`.
    ManifestDtypeMismatch {
        expected: WeightDtype,
        found: WeightDtype,
    },
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::IncompatibleExpertTypes(e) => write!(f, "{e}"),
            EngineError::ManifestDtypeMismatch { expected, found } => write!(
                f,
                "manifest dtype mismatch: engine configured for {expected:?} \
                 but every indexed expert declares {found:?}"
            ),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<crate::io_provider::IncompatibleExpertTypes> for EngineError {
    fn from(e: crate::io_provider::IncompatibleExpertTypes) -> Self {
        EngineError::IncompatibleExpertTypes(e)
    }
}

/// Optional JSONL trace sink — one record per `Engine::generate` call.
///
/// When the engine is constructed with a trace path, every token's
/// `{token, layer, experts, cache_hit}` is appended as a line. Used by
/// `scripts/compute_transition_matrix.py` and the
/// `validate-predictor` subcommand to evaluate the predictor offline
/// against real routing distributions. See gist Phase 6.
///
/// **I/O off the hot path.** `write_record` is a non-blocking enqueue
/// onto a bounded `std::sync::mpsc::sync_channel`. A dedicated worker
/// thread drains the channel and does the actual `BufWriter::write_all`
/// + `flush` against the file. When the channel is full (writer can't
/// keep up — slow disk, full FS, etc.), the newest record is dropped
/// rather than stalling the engine on a blocking write. This makes
/// the trace strictly best-effort and decouples disk latency from
/// per-token decode latency.
pub struct TraceWriter {
    tx: parking_lot::Mutex<Option<std::sync::mpsc::SyncSender<TraceRecord>>>,
    /// Shared with the worker so `flush` can synchronise on a
    /// definite "everything queued so far has been written" point
    /// (used in shutdown paths and tests).
    flush_signal: Arc<(
        parking_lot::Mutex<u64>,
        parking_lot::Condvar,
        std::sync::atomic::AtomicU64,
    )>,
    /// Producer-side high-water mark: the largest sequence number
    /// successfully enqueued onto the channel. Updated *after* a
    /// successful `try_send` so `flush()` never waits on a record the
    /// channel rejected (queue full / disconnected). When `try_send`
    /// fails the record is silently dropped and the HWM is left
    /// unchanged — that matches the documented "best-effort" trace
    /// contract.
    producer_hwm: std::sync::atomic::AtomicU64,
    /// Monotonic sequence counter for outgoing records. Per-instance
    /// (was previously a `static` inside `write_record`, which shared
    /// the counter across every `TraceWriter` ever created in the
    /// process and prevented `flush` from synchronising correctly
    /// when multiple writers existed in tests).
    seq: std::sync::atomic::AtomicU64,
    /// Set to `true` the first time a write fails so subsequent failures
    /// stay silent. Without this guard a sticky I/O error (full disk,
    /// unwritable path) would emit a `warn!` on *every* record and
    /// drown the rest of the logs.
    write_failed_once: std::sync::atomic::AtomicBool,
}

/// One serialised record handed across the channel. Kept as a small
/// owned struct (rather than a pre-formatted `String`) so the worker
/// thread does the `format!` work, not the producer.
struct TraceRecord {
    token: u64,
    layer: u32,
    experts: Vec<u32>,
    cache_hit: Vec<bool>,
    /// The predictive controller's guess for this token's experts (the
    /// neural speculator's top-K when installed; empty when no
    /// speculator is wired). Logged alongside the gate's actual
    /// `experts` so offline analysis can diff *Predicted vs. Actual*
    /// per layer — isolating "wrong layer" from "wrong expert within
    /// the correct layer".
    predicted: Vec<u32>,
    /// Monotonic sequence number assigned at enqueue time so `flush`
    /// can wait for the worker to catch up to a specific point.
    seq: u64,
}

impl TraceWriter {
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        // Append semantics: documented as "appends one record per
        // token", so existing trace files must be preserved across
        // invocations. `create(true)` still creates the file if it
        // doesn't already exist.
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let mut writer = std::io::BufWriter::new(f);
        // Bounded channel: at sustained 100k tokens/s the worker
        // drains in a few ms. The bound only matters if the disk
        // stalls — in which case dropping the *newest* record is
        // the right back-pressure (old records still contain useful
        // signal; the loss is bounded and visible in the warn log).
        let (tx, rx) = std::sync::mpsc::sync_channel::<TraceRecord>(4096);
        let flush_signal = Arc::new((
            parking_lot::Mutex::new(0u64),
            parking_lot::Condvar::new(),
            std::sync::atomic::AtomicU64::new(0),
        ));
        let flush_signal_w = flush_signal.clone();
        std::thread::Builder::new()
            .name("mer-trace-writer".to_string())
            .spawn(move || {
                use std::io::Write;
                use std::sync::mpsc::TryRecvError;
                let mut latched_failure = false;
                // Outer loop: block until at least one record arrives.
                'outer: while let Ok(first) = rx.recv() {
                    // `latest_seq` is assigned inside the inner loop before
                    // it's read after the loop, so no initial value is
                    // needed here.
                    let mut latest_seq;
                    let mut rec = first;
                    // Inner loop: drain anything already queued before
                    // touching the BufWriter::flush. This lets sustained
                    // writes batch in the BufWriter (which is the whole
                    // point of buffering); we only flush when the
                    // channel momentarily empties, so quiet periods see
                    // bytes hit the kernel quickly enough for `flush()`
                    // callers to make progress.
                    let drained_cleanly = loop {
                        let mut s = String::with_capacity(64 + rec.experts.len() * 8);
                        s.push_str(&format!(
                            "{{\"token\":{},\"layer\":{},\"experts\":[",
                            rec.token, rec.layer
                        ));
                        for (i, e) in rec.experts.iter().enumerate() {
                            if i > 0 { s.push(','); }
                            s.push_str(&e.to_string());
                        }
                        s.push_str("],\"cache_hit\":[");
                        for (i, h) in rec.cache_hit.iter().enumerate() {
                            if i > 0 { s.push(','); }
                            s.push_str(if *h { "true" } else { "false" });
                        }
                        s.push_str("],\"predicted\":[");
                        for (i, e) in rec.predicted.iter().enumerate() {
                            if i > 0 { s.push(','); }
                            s.push_str(&e.to_string());
                        }
                        s.push_str("]}\n");
                        if !latched_failure {
                            if let Err(e) = writer.write_all(s.as_bytes()) {
                                warn!(error = %e, "trace writer failed; subsequent records may be lost (further failures suppressed)");
                                latched_failure = true;
                            }
                        }
                        latest_seq = rec.seq;
                        match rx.try_recv() {
                            Ok(next) => { rec = next; }
                            Err(TryRecvError::Empty) => break true,
                            Err(TryRecvError::Disconnected) => break false,
                        }
                    };
                    // Flush the BufWriter so any `TraceWriter::flush()`
                    // caller waiting on `latest_seq` sees the bytes hit
                    // the file descriptor (not just BufWriter's
                    // in-memory buffer). Without this flush, advancing
                    // the worker HWM before flushing would let
                    // `flush()` return while the JSONL bytes were still
                    // stuck inside the BufWriter — exactly the bug the
                    // reviewer flagged.
                    if !latched_failure {
                        if let Err(e) = writer.flush() {
                            warn!(error = %e, "trace writer flush failed; subsequent records may be lost (further failures suppressed)");
                            latched_failure = true;
                        }
                    }
                    // Only *now* publish the high-water mark so flushers
                    // unblock with the bytes already durable in the file
                    // descriptor.
                    flush_signal_w.2.store(latest_seq, std::sync::atomic::Ordering::Release);
                    {
                        let mut g = flush_signal_w.0.lock();
                        *g = latest_seq;
                        flush_signal_w.1.notify_all();
                    }
                    if !drained_cleanly {
                        // Sender side dropped after the drain — exit
                        // the outer loop and run the shutdown flush.
                        break 'outer;
                    }
                }
                // Channel closed: final flush so partial records hit the disk.
                let _ = writer.flush();
            })
            .ok();
        Ok(Self {
            tx: parking_lot::Mutex::new(Some(tx)),
            flush_signal,
            producer_hwm: std::sync::atomic::AtomicU64::new(0),
            seq: std::sync::atomic::AtomicU64::new(0),
            write_failed_once: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub fn write_record(
        &self,
        token: u64,
        layer: u32,
        experts: &[u32],
        cache_hit: &[bool],
        predicted: &[u32],
    ) {
        // Assign a monotonic per-writer sequence so flush() has
        // something to wait on.
        let seq = self.seq.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let rec = TraceRecord {
            token,
            layer,
            experts: experts.to_vec(),
            cache_hit: cache_hit.to_vec(),
            predicted: predicted.to_vec(),
            seq,
        };
        let guard = self.tx.lock();
        let Some(tx) = guard.as_ref() else { return };
        // `try_send` is non-blocking; on overflow the newest record is
        // dropped (back-pressure to bound memory).
        match tx.try_send(rec) {
            Ok(()) => {
                // Only publish the producer HWM *after* the channel has
                // accepted the record. If we advanced it before
                // `try_send` and the send then failed (queue full or
                // disconnected), every subsequent `flush()` would stall
                // until the 500 ms timeout waiting for a seq the
                // worker can never observe. Release pairs with
                // flush()'s Acquire load.
                //
                // The store uses a CAS-style max so out-of-order
                // success notifications (rare under contention) can't
                // walk the HWM backwards.
                let mut current = self.producer_hwm.load(std::sync::atomic::Ordering::Acquire);
                while seq > current {
                    match self.producer_hwm.compare_exchange_weak(
                        current,
                        seq,
                        std::sync::atomic::Ordering::AcqRel,
                        std::sync::atomic::Ordering::Acquire,
                    ) {
                        Ok(_) => break,
                        Err(observed) => current = observed,
                    }
                }
            }
            Err(e) => {
                if !self
                    .write_failed_once
                    .swap(true, std::sync::atomic::Ordering::Relaxed)
                {
                    warn!(reason = %e, "trace writer queue full; dropping records (further drops suppressed)");
                }
            }
        }
    }

    pub fn flush(&self) {
        // Block until the worker has caught up to the highest seq the
        // producer side ever successfully enqueued *and* the worker's
        // BufWriter has been flushed so those bytes are visible to
        // file readers. Bounded wait so a stuck worker can't deadlock
        // the caller.
        //
        // Two-part invariant the worker now upholds: after every
        // queue-drained iteration it (1) calls `BufWriter::flush()`
        // and *then* (2) publishes `latest_seq` on `flush_signal.2`
        // and `flush_signal.0`. So observing `*guard >= snapshot`
        // implies the bytes for every seq ≤ snapshot have hit the
        // file descriptor.
        //
        // Snapshot the *producer* HWM (not the worker's HWM); the
        // worker's HWM lags the producer and would let `flush` return
        // before the worker has actually drained the latest records.
        let snapshot = self.producer_hwm.load(std::sync::atomic::Ordering::Acquire);
        if snapshot == 0 {
            return; // nothing ever queued (or every send dropped)
        }
        let mut guard = self.flush_signal.0.lock();
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
        while *guard < snapshot {
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            let _ = self.flush_signal.1.wait_for(&mut guard, deadline - now);
        }
    }
}

impl Drop for TraceWriter {
    fn drop(&mut self) {
        // Closing the sender drops the worker's channel rx, which
        // exits the loop and flushes the BufWriter as part of the
        // worker's `let _ = writer.flush()` shutdown step. The
        // OS-level fsync/close happens when the file goes out of
        // scope on the worker thread.
        let mut guard = self.tx.lock();
        guard.take();
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CycleStats {
    pub hits: u64,
    pub misses: u64,
    pub prefetch_hits: u64,
    pub bytes_read: u64,
}

/// Snapshot of the engine's predictive-architecture telemetry: the
/// running accuracy of the [`NeuralSpeculator`] (M arm), the running
/// hit rate of the [`LocalityMonitor`] (L arm), and the cumulative
/// SSD-stall time on the inference critical path. Returned by
/// [`Engine::predictive_telemetry`].
#[derive(Debug, Default, Clone, Copy)]
pub struct PredictiveTelemetry {
    pub speculator_hits: u64,
    pub speculator_misses: u64,
    /// `speculator_hits / (speculator_hits + speculator_misses)`, or
    /// `0.0` when neither has fired. Hits are predicted expert IDs
    /// contained in the gate's actual top-K; misses are predicted
    /// expert IDs not contained in the gate's actual top-K. The ratio
    /// is prediction precision@K. The field name is preserved for
    /// backwards compatibility — the design spec's "speculator
    /// accuracy" is the top-1 metric below.
    pub speculator_accuracy: f64,
    /// Cumulative count of tokens for which the speculator's **top-1**
    /// prediction matched the gate's actual top-1 routed expert.
    /// Mirrors the `mer_speculator_accuracy_total` Prometheus counter.
    pub speculator_top1_matches: u64,
    /// Total tokens for which the speculator was invoked (the
    /// denominator of the top-1 accuracy ratio).
    pub speculator_top1_total: u64,
    /// `speculator_top1_matches / speculator_top1_total`, or `0.0`
    /// when the speculator has not been invoked yet.
    pub speculator_top1_accuracy: f64,
    pub locality_hits: u64,
    pub locality_misses: u64,
    /// `locality_hits / (locality_hits + locality_misses)`, or `0.0`
    /// when neither has fired.
    pub locality_hit_rate: f64,
    /// Cumulative SSD critical-path stall, in microseconds.
    pub ssd_stall_us: u64,
}

#[derive(Default)]
pub(crate) struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
    prefetch_completed: AtomicU64,
    prefetch_used: AtomicU64,
    /// Cumulative bytes pulled from the storage layer. **Single
    /// source of truth: incremented exactly once per disk read, by
    /// the leader inside [`Engine::fetch_once`] (and by the
    /// background prefetch task, which is also a leader path).**
    /// Critical-path callers (`generate`, `moe_step`) never bump
    /// this counter — followers parked on the in-flight singleflight
    /// notify don't issue I/O, so adding to `bytes_read` from
    /// `generate` post-SSD-dedup (gist Phase 1) would double-count
    /// every miss. The per-call `FetchStats::bytes_read` is a
    /// separate, *logical* accumulator: it tracks how many bytes a
    /// given token consumed, including bytes that were served from
    /// the cache without touching disk (so it sums to the working
    /// set, not the I/O traffic). The invariant
    /// `EngineReport::bytes_read >= sum(FetchStats::bytes_read)` may
    /// fail because the per-call stat counts cache hits while the
    /// counter does not, but `EngineReport::bytes_read >=
    /// sum(critical-path miss bytes)` always holds — see the test
    /// `assert_singleflight_dedupes_concurrent_misses`.
    bytes_read: AtomicU64,
    /// Cumulative experts dropped from a `moe_step` mixture because
    /// their fetch failed after all retry attempts. Surfaced via
    /// `EngineReport::expert_read_failures` so operators can alert on
    /// it from /metrics + /health.
    expert_read_failures: AtomicU64,
    /// Cumulative routed/shared experts substituted with a zero
    /// contribution because the development-only
    /// `allow_degraded_experts` mode is active. Always zero in strict
    /// production mode. Non-zero values mark every metric and
    /// benchmark figure of the run as degraded / non-authoritative.
    degraded_expert_substitutions: AtomicU64,
    /// Number of times a `fetch_with_retry` caller piggy-backed on a
    /// concurrent leader's in-flight read instead of issuing its own
    /// (gist Phase 1 — SSD Read De-Duplication). Each increment maps
    /// directly to one disk read that was *not* performed.
    singleflight_followers: AtomicU64,
    /// Speculative prefetches dropped because the concurrent-prefetch
    /// semaphore was exhausted (gist Phase 3 — bounded prefetch).
    /// Surfaced via `EngineReport::prefetch_dropped_concurrency`.
    prefetch_dropped_concurrency: AtomicU64,
    /// Speculative prefetches dropped because no buffer could be
    /// acquired — the shadow (Buffer B) half was starved even after
    /// recycling the LRU shadow-backed resident, or (legacy
    /// single-pool configs) the primary pool was busy. Previously this
    /// was only a `debug!`, making shadow-pool starvation invisible in
    /// production; surfaced via
    /// `EngineReport::prefetch_dropped_pool_starved` and
    /// `mer_prefetch_dropped_pool_starved_total`.
    prefetch_dropped_pool_starved: AtomicU64,
    /// **Tier 4.** Speculative prefetches the adaptive
    /// [`crate::prefetch_governor::PrefetchGovernor`] declined to admit
    /// because their expected value did not clear the contention-scaled
    /// bar. Always `0` when the governor is disabled (the default), so
    /// legacy telemetry is unchanged. Surfaced via
    /// `EngineReport::prefetch_dropped_governor`.
    prefetch_dropped_governor: AtomicU64,
    /// Tokens for which the neural speculator (M arm) was silently
    /// disabled because the hidden-state width didn't match the
    /// speculator's `d_model`. A persistent non-zero rate means the
    /// predictive arm is misconfigured and contributing nothing.
    speculator_dmodel_mismatch: AtomicU64,
    /// Expert activations that fell back from the GPU fast path to the
    /// CPU path because physical routed-expert dispatch errored. Invisible
    /// mixed GPU/CPU execution is a major source of
    /// inconsistent token latency, so make it countable.
    gpu_cpu_fallbacks: AtomicU64,
    /// Routed expert activations selected by the real MoE routing path.
    selected_routed_experts: AtomicU64,
    /// Routed-expert activations entering the GPU dispatch boundary. This
    /// includes activations rejected by pre-backend runtime-invariant guards,
    /// so GPU-plan attempts stay equal to `selected_routed_experts`.
    gpu_dispatch_attempts: AtomicU64,
    /// Physical GPU routed-expert dispatches that returned output.
    gpu_dispatch_successes: AtomicU64,
    /// Physical GPU routed-expert dispatches that returned a typed failure.
    gpu_dispatch_failures: AtomicU64,
    /// Routed expert activations executed by the CPU backend. This includes
    /// CPU plans and explicit serving-mode fallbacks, but never strict GPU
    /// failures.
    cpu_routed_expert_dispatches: AtomicU64,
    /// Hardware-independent CPU routed-expert forward spy. Test-only so the
    /// public fallback metric keeps its production schema and meaning.
    #[cfg(test)]
    cpu_expert_forward_calls: AtomicU64,
}

/// Monotonic execution counters for real routed-expert activations.
///
/// These counters deliberately exclude the synthetic [`Engine::generate`]
/// path. They are a qualification seam rather than a second Prometheus
/// telemetry surface.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RoutedExpertExecutionSnapshot {
    pub selected_routed_experts: u64,
    pub gpu_dispatch_attempts: u64,
    pub gpu_dispatch_successes: u64,
    pub gpu_dispatch_failures: u64,
    pub cpu_routed_expert_dispatches: u64,
    pub gpu_cpu_fallbacks: u64,
    pub degraded_expert_substitutions: u64,
}

/// Shape parameters of the SwiGLU expert FFN executed by the engine.
///
/// Each on-disk expert file is a flat blob of `f32` weights laid out as
/// `gate_proj || up_proj || down_proj` (see [`crate::inference`]).
#[derive(Clone, Copy, Debug)]
pub struct ModelShape {
    pub d_model: usize,
    pub d_ff: usize,
    /// Seed used to derive per-token hidden states. In a real model this
    /// would come from the previous transformer layer; here it lets us
    /// produce reproducible activations for the synthetic stream.
    pub hidden_seed: u64,
}

/// Explicit parallelism level for routed expert execution.
///
/// The production-safe default keeps the selected experts sequential and
/// lets each expert use the row-parallel inner kernels. Expert-parallel
/// mode fans the selected experts out across the shared Rayon pool, so
/// those inner kernels see `in_rayon_worker()` and run inline instead of
/// recursively occupying the whole pool.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Eq,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "kebab-case")]
pub enum ExpertExecutionPolicy {
    #[default]
    Auto,
    SequentialExpertsRowParallel,
    ParallelExpertsSingleThread,
}

/// Runtime recovery contract after the authoritative plan selects GPU routed
/// experts. PR3 keeps this engine-scoped: current entry points retain the
/// compatibility default, while qualification can opt in before inference.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RoutedExpertGpuFailurePolicy {
    StrictFailClosed,
    #[default]
    ServingCpuFallback,
}

/// Run-time options that affect how `Engine::generate` executes a token.
///
/// The defaults model a normal end-to-end run (router → I/O → SwiGLU
/// FFN); `io_only` flips off the FFN compute so the same instrumentation
/// can be used to measure pure I/O cost.
#[derive(Clone, Copy, Debug)]
pub struct EngineOptions {
    /// When `true`, skip [`run_inference`] and instead XOR every byte of
    /// the resident buffer to force the read to fully materialise. This
    /// isolates the SSD-streaming cost from FFN compute and is what
    /// `--io-only` on the CLI maps to.
    pub io_only: bool,
    /// On-disk weight dtype. Selects which of the `run_inference*`
    /// variants is dispatched per cache hit.
    pub dtype: WeightDtype,
    /// Fraction of `d_model` columns to load when the partial-load path
    /// is enabled (`(0.1..1.0)`). `1.0` disables partial loading and
    /// the engine reads the full expert as before.
    pub partial_load_fraction: f64,
    /// After an expert has been observed this many times in routing
    /// targets, pin it permanently in the LRU cache. `0` disables
    /// frequency-based pinning entirely.
    pub pin_after_observations: u64,
    /// **QMatMul fast path for 4-bit dtypes (Industrial Upgrade Task 1).**
    /// When `true` (default) and `dtype` is `Q4_0` or `Q4K`, the
    /// engine dispatches per-expert SwiGLU through candle-core's
    /// `QMatMul` directly over the on-disk quantised blocks — no F32
    /// dequant of the weights happens. Falls back to the legacy
    /// dequant path automatically when `QMatMul` returns an error
    /// (e.g. block-alignment mismatch on a corrupt blob), so this is
    /// a strict-superset behaviour switch.
    pub use_qmm_for_q4: bool,
    /// Explicit expert-execution parallelism policy for the routed top-K.
    /// `Auto` selects by dtype, selected-expert count, and available compute
    /// workers while preserving the invariant that the engine never nests
    /// full-pool expert parallelism inside full-pool row parallelism.
    pub expert_execution_policy: ExpertExecutionPolicy,
    /// Upper bound on speculative prefetches in flight at any one
    /// time. Each call to `spawn_prefetch` must acquire a semaphore
    /// permit before issuing the I/O — when the bound is reached the
    /// prefetch is dropped (it's speculative, missing one is fine)
    /// and `prefetch_dropped_concurrency` is incremented. Values less
    /// than `1` are clamped to `1`; the default `64` matches typical
    /// io_uring queue depths.
    pub max_concurrent_prefetches: usize,
    /// Upper bound on yield iterations [`Engine::fetch_once`] spins
    /// through while waiting for a free [`PooledBuffer`] when the
    /// expert cache is full of pinned residents. Once the limit is
    /// reached the call returns [`FetchOnceError::PoolStarved`]
    /// instead of yielding indefinitely. Defaults to
    /// [`DEFAULT_MAX_FETCH_YIELDS`] (`128`) — low enough to surface a
    /// pool-misconfiguration as a fast error in latency-sensitive
    /// scenarios, but high enough to absorb a transient burst of
    /// concurrent prefetches under steady-state load (gist feedback
    /// #1.3). Values less than `1` are clamped to `1` at use.
    pub max_fetch_yields: usize,
    /// **Tier 4 — adaptive prefetch governor.** When `true`, every
    /// speculative prefetch is admitted by
    /// [`crate::prefetch_governor::PrefetchGovernor`], which throttles
    /// speculation as measured prefetch precision falls or as foreground
    /// (token-blocking) reads queue up for the device. `false` (the
    /// default) preserves the legacy unbounded-admission behaviour
    /// exactly. This is the highest-leverage knob on a bandwidth-bound
    /// SSD: it stops low-precision speculation from inflating the latency
    /// of the foreground misses that actually block token generation.
    pub prefetch_governor: bool,
    /// Precision floor (and optimistic EWMA seed lower bound) for the
    /// prefetch governor, in `[0, 1]`. Only consulted when
    /// `prefetch_governor` is set.
    pub prefetch_precision_floor: f64,
    /// Per-outstanding-foreground-read multiplier the prefetch governor
    /// applies to its admission threshold. Higher values make
    /// speculation back off harder while real misses are in flight.
    pub prefetch_contention_weight: f64,
    /// **Tier 4 — cost-aware eviction.** When `true`, the RAM expert
    /// cache evicts the non-pinned resident with the lowest *decaying
    /// heat score* (frequency of use, aged over time) rather than the
    /// strict LRU victim, so a genuinely hot expert that briefly fell to
    /// the LRU tail is not dumped ahead of a one-shot cold expert.
    /// `false` (the default) keeps pure-LRU + binary pin-set eviction.
    pub cost_aware_eviction: bool,
    /// **Tier 3 — per-layer pre-gate predictor.** When `true`, the
    /// engine trains an online conditional map from one layer's routed
    /// expert set to the *next* layer's experts and uses it to drive
    /// high-precision next-layer prefetch on the real-transformer /
    /// trace-replay path. `false` (the default) leaves the existing
    /// speculator/Markov look-ahead untouched.
    pub pregate_enabled: bool,
    /// **Tier 1 — route-profile collection.** When `true`, the engine
    /// always bumps per-expert `route_observations` (even when neither
    /// frequency pinning nor online static residency would otherwise
    /// need them) so a run can emit a popularity profile via
    /// `--profile-out`. `false` (the default) keeps the per-token bump
    /// free when no consumer needs the counts.
    pub collect_route_profile: bool,
    /// **Fail-closed real inference policies (hardening pass).**
    /// Independent development-only fail-open switches; all default to
    /// `false` (strict). See [`crate::inference::RealInferencePolicy`]:
    /// `allow_degraded_experts` preserves the legacy drop-from-mixture
    /// behaviour for failed routed/shared/dense experts (counted via
    /// `degraded_expert_substitutions`), `allow_nonfinite_attention_fallback`
    /// preserves the legacy uniform-softmax fallback for non-finite
    /// attention rows, and `allow_truncated_expert_payloads` preserves
    /// the legacy one-page zero-fill tolerance for short quantised
    /// payloads. Enabling one never enables another; `bench-real`
    /// rejects all three.
    pub policy: crate::inference::RealInferencePolicy,
}

/// Default semaphore ceiling for `Engine::spawn_prefetch`. Matches a
/// typical io_uring submission-queue depth and is the source of truth
/// for both `EngineOptions::default()` and the TOML default of
/// `[real_transformer].max_concurrent_prefetches`.
pub const DEFAULT_MAX_CONCURRENT_PREFETCHES: usize = 64;

/// Default cap on yield iterations [`Engine::fetch_once`] waits
/// before declaring the buffer pool starved and returning
/// [`FetchOnceError::PoolStarved`]. Lowered from `1024` in gist
/// feedback #1.3 — `1024` yields under heavy load corresponds to
/// many milliseconds of soft-stall before the engine surfaces the
/// underlying pool-sizing bug to the caller, which is too forgiving
/// for latency-sensitive scenarios. `128` still absorbs a normal
/// burst of concurrent prefetches without spurious failures.
pub const DEFAULT_MAX_FETCH_YIELDS: usize = 128;

/// Default look-ahead **pipeline depth** for [`Engine::speculate_layer_ahead`]:
/// how many MoE layers of compute the engine tries to keep the SSD reads
/// running ahead of. Set to roughly `ceil(io_latency / compute_latency)`
/// — three layers of ~77 ms SwiGLU compute (≈ 231 ms) is enough to fully
/// hide a ~206 ms cold expert read, so the data lands resident before the
/// execution thread reaches that layer. Tunable per-deployment via the
/// `[storage] pipeline_depth` TOML key (serve) or `--pipeline-depth` (run);
/// `1` reproduces the legacy single-layer look-ahead.
pub const DEFAULT_PIPELINE_DEPTH: u32 = 3;

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            io_only: false,
            dtype: WeightDtype::F32,
            partial_load_fraction: 1.0,
            pin_after_observations: 0,
            use_qmm_for_q4: true,
            expert_execution_policy: ExpertExecutionPolicy::Auto,
            max_concurrent_prefetches: DEFAULT_MAX_CONCURRENT_PREFETCHES,
            max_fetch_yields: DEFAULT_MAX_FETCH_YIELDS,
            prefetch_governor: false,
            prefetch_precision_floor: 0.05,
            prefetch_contention_weight: 1.0,
            cost_aware_eviction: false,
            pregate_enabled: false,
            collect_route_profile: false,
            policy: crate::inference::RealInferencePolicy::STRICT,
        }
    }
}

/// Core MoE / I/O wiring: cache, buffer pool, storage, router,
/// predictor, model shape, and run-time options.
///
/// Owns the actual MoE expert-streaming machinery — everything needed
/// to turn a routing decision into resident weights and back. It is
/// deliberately free of telemetry and predictive-routing state so that
/// future feature work (e.g. additional cache tiers, scheduler swaps)
/// can be added without churning the observability layer alongside.
pub(crate) struct EngineCore {
    /// The engine's expert cache. Wrapped in [`MultiLayerExpertCache`]
    /// so the per-layer LRU dispatch is on the hot path: single-layer
    /// models use [`MultiLayerExpertCache::single_layer`] (observably
    /// identical to the previous flat `ExpertCache`), while multi-layer
    /// `serve` paths construct it with `with_uniform_capacity` /
    /// `with_capacities` so layer N's prefetched experts can never
    /// evict layer M's residents.
    pub(super) cache: Arc<MultiLayerExpertCache>,
    pub(super) pool: BufferPool,
    pub(super) storage: Arc<NvmeStorage>,
    /// Routing strategy. `Router::Linear` runs the production
    /// `LinearGate` (`softmax(W_gate · x) → top-K`) and is the path
    /// `cmd_serve` wires up when `[real_transformer].enabled = true`
    /// and the loaded model exposes per-layer gate weights;
    /// `Router::Markov` runs the legacy deterministic `TopKRouter`
    /// over expert ids and is the benchmark / `--io-only` fallback.
    /// Both are exercised by the engine through the same call site,
    /// so swapping them does not change cache / I/O / hit-rate
    /// telemetry shape, only which expert ids are selected.
    pub(super) router: Router,
    pub(super) predictor: Arc<PredictiveLoader>,
    pub(super) shape: ModelShape,
    pub(super) options: EngineOptions,
    /// Optional logical GPU-admission cache. `None` (default) leaves
    /// the engine in its legacy 2-tier posture. When `Some`, every
    /// cache lookup in [`Engine::generate`] / [`Engine::moe_step`]
    /// first probes logical admission; misses fall through to the RAM
    /// `MultiLayerExpertCache` and then to NVMe.
    pub(super) gpu_cache: Option<Arc<GpuExpertCache>>,
    /// Optional future-token-loop physical residency plane. It is never
    /// installed from the legacy GPU mode. When absent, every existing
    /// demand and speculative path remains unchanged.
    pub(super) gpu_native_residency: Option<Arc<GpuNativeTieredResidencyManager>>,
    /// One-shot sender that feeds background RAM → logical-GPU-admission
    /// promotion task. The receiver lives on a dedicated Tokio task
    /// spawned by [`Engine::install_gpu_cache`]; the inference hot
    /// path never blocks on this channel — promotions are pure
    /// fire-and-forget.
    pub(super) gpu_promotion_tx:
        Option<tokio::sync::mpsc::UnboundedSender<(u32, Arc<ExpertResident>)>>,
    /// In-flight read singleflight (gist Phase 1 — SSD Read
    /// De-Duplication). Lives on `EngineCore` (gist feedback #2.4)
    /// because it is part of the I/O-runtime infrastructure that
    /// the cache + pool + storage triple already lives on. When N
    /// concurrent tasks all miss the cache on the same expert id,
    /// only the first task issues a disk read; the rest park on the
    /// shared [`Notify`] and re-check the cache once the leader's
    /// read completes. With this in place, `BatchScheduler` pre-pass
    /// `engine.warm_with(&unique_ids)` truly maps to "one read per
    /// unique id across the batch", and even without a pre-pass
    /// concurrent `moe_step` invocations no longer duplicate I/O.
    pub(super) in_flight: Arc<DashMap<u32, Arc<Notify>>>,
    /// Bound on concurrent speculative prefetches. Sized from
    /// [`EngineOptions::max_concurrent_prefetches`]. Lives on
    /// `EngineCore` (gist feedback #2.4) alongside the other
    /// I/O-runtime infrastructure. Each `spawn_prefetch` call must
    /// obtain an owned permit *before* spawning the async task;
    /// failure to acquire drops the prefetch and increments
    /// `EngineMetrics::counters::prefetch_dropped_concurrency`.
    pub(super) prefetch_semaphore: Arc<tokio::sync::Semaphore>,
    /// Immutable execution plan plus the sole backend context used by this
    /// engine and its associated real model. Routed-expert dispatch reads the
    /// backend selected by this context; it never reconstructs one privately.
    pub(super) execution_context: Arc<crate::backend::ExecutionContext>,
    /// Immutable once the engine is shared for inference.
    pub(super) routed_expert_gpu_failure_policy: RoutedExpertGpuFailurePolicy,
    /// **Tier 4 — adaptive prefetch admission controller.** Always
    /// present; constructed in the transparent pass-through state unless
    /// [`EngineOptions::prefetch_governor`] is set, in which case it
    /// gates every `spawn_prefetch` admission on measured prefetch
    /// precision and foreground-read contention. Lock-free, so the
    /// per-prefetch `admit` check and the per-hit precision feedback add
    /// only a handful of relaxed atomic ops to the hot path.
    pub(super) governor: Arc<crate::prefetch_governor::PrefetchGovernor>,
}

/// Predictive-routing state: aliasing & frequency-based pinning,
/// locality monitor (the **L** arm of `S ∪ L ∪ M`), and the neural
/// speculator (the **M** arm). All three live together because they
/// share the same "observe routing decision → predict / pin" code
/// path called from `generate` / `moe_step`.
///
/// Each arm is independently optional — a fresh `Engine::new(...)`
/// disables all three, which preserves the legacy benchmark path
/// bit-for-bit.
/// One step of Markov routing history: the expert set the gate chose,
/// tagged with the MoE layer it was chosen for (`None` on the
/// layer-less `generate` benchmark path). The layer tag lets
/// `moe_step` verify that consecutive history entries actually came
/// from consecutive layers of the *same* token stream before learning
/// or predicting from them — concurrent batched requests interleave
/// their `moe_step` calls in this engine-global ring, and an
/// uncontiguous pair is cross-stream noise the predictor must not
/// train on (Finding 5).
#[derive(Default, Clone)]
pub(crate) struct MarkovHistory {
    pub(crate) ids: Vec<u32>,
    pub(crate) layer: Option<u32>,
}

/// The two-deep Markov history ring (`prev` + `prev_prev`) behind a
/// **single** mutex. The legacy layout used one mutex per entry, which
/// forced every history update to take two locks back-to-back (a
/// nested-acquisition pattern that both doubles the lock traffic on
/// the per-token hot path and bakes in an implicit lock-ordering
/// invariant). Collapsing the ring into one critical section makes the
/// shift (`last → last_last`, `target → last`) atomic by construction:
/// no interleaving can ever observe a half-shifted ring.
#[derive(Default)]
pub(crate) struct MarkovRing {
    /// Expert set the gate chose on the previous step.
    pub(crate) last: MarkovHistory,
    /// Expert set active two steps ago — feeds the predictor's
    /// 2nd-order rows.
    pub(crate) last_last: MarkovHistory,
}

pub(crate) struct EngineSpeculation {
    /// Optional alias map: when present, any routed/predicted expert id
    /// is remapped to its canonical id before the cache is consulted.
    /// Used for **expert deduplication** — pairs of experts that the
    /// offline analyser flagged as numerically near-identical share a
    /// single resident copy. `None` means no aliasing.
    pub(super) alias_map: Option<Arc<HashMap<u32, u32>>>,
    /// Number of times an alias redirect actually changed an expert id
    /// during routing/prefetch (for diagnostics).
    pub(super) alias_redirects: AtomicU64,
    /// Per-expert routing-observation counts used by frequency-based
    /// pinning. Once an expert's count crosses
    /// `options.pin_after_observations`, the engine asks the cache to
    /// pin it. Sharded `DashMap` of atomics instead of a global
    /// `RwLock<HashMap>`: the per-token bump is a shard *read* lock +
    /// `fetch_add` in steady state, so concurrent `moe_step` calls
    /// from batched requests no longer serialize on one writer lock.
    pub(super) route_observations: DashMap<u32, AtomicU64>,
    /// Engine-owned monotonic token counter for route-observation bumps.
    /// Online static-residency warmup gates on this instead of the
    /// caller-supplied `token_idx`, which can be per-stream or reused on
    /// `moe_step`.
    pub(super) route_observation_tokens: AtomicU64,
    /// Two-deep Markov history (`last` + `last_last`) behind a single
    /// mutex — see [`MarkovRing`] for why the entries share one lock.
    pub(super) markov_ring: parking_lot::Mutex<MarkovRing>,
    /// Locality monitor — sliding-window heat map over recently-routed
    /// experts. When configured, the engine reconciles its hot set
    /// against the expert cache after every token: ids in the hot set
    /// are pinned (cannot be LRU-evicted) and ids that just dropped
    /// out are unpinned. Forms the **L** arm of the speculative I/O
    /// union `E = S ∪ L ∪ M`.
    pub(super) locality: Option<Arc<LocalityMonitor>>,
    /// Set of expert ids the locality monitor pinned on the previous
    /// reconciliation. Diff'd against the current hot set so we only
    /// `pin`/`unpin` ids that actually changed status.
    pub(super) locality_pinned: parking_lot::Mutex<HashSet<u32>>,
    /// Expert ids pinned by static residency. Locality reconciliation must
    /// not unpin these ids when they leave the sliding-window hot set.
    pub(super) static_pinned: parking_lot::Mutex<HashSet<u32>>,
    /// Heat threshold for [`Self::locality`]. Mirrors
    /// [`LocalityMonitor::DEFAULT_THRESHOLD_PCT`] when not overridden.
    pub(super) locality_threshold_pct: f32,
    /// Cumulative locality-hit count (target experts that were already
    /// in the locality monitor's hot set at routing time).
    pub(super) locality_hits: AtomicU64,
    /// Cumulative locality-miss count.
    pub(super) locality_misses: AtomicU64,
    /// Neural speculator — a tiny 2-layer MLP that predicts the gate's
    /// top-K from the hidden state. Forms the **M** arm of the union
    /// `E = S ∪ L ∪ M` and is trained online against the actual gate
    /// decision. Wrapped in an `Arc` for cheap cloning into spawned
    /// prefetch tasks; internal weights are guarded by an `RwLock`
    /// owned by the speculator itself.
    pub(super) speculator: Option<Arc<NeuralSpeculator>>,
    /// Number of speculator predictions pulled per token (top-K size
    /// for the M arm). Defaults to the router's `top_k`.
    pub(super) speculator_topk: usize,
    /// **Look-ahead pipeline depth** for [`Engine::speculate_layer_ahead`]:
    /// the engine prefetches the experts of the sliding window of layers
    /// `current_layer + 1 ..= current_layer + pipeline_depth`, so the SSD
    /// reads for the next several layers are already in flight while the
    /// current layer computes. Deeper look-ahead hides more of the SSD
    /// read latency behind compute (see [`DEFAULT_PIPELINE_DEPTH`]); `1`
    /// reproduces the legacy single-layer look-ahead. Predictions further
    /// out are staler, so the per-layer fanout is tapered with distance to
    /// keep low-confidence far-layer reads from flooding the SSD.
    pub(super) pipeline_depth: u32,
    /// Cumulative speculator hit count: predicted expert IDs contained
    /// in the gate's actual top-K.
    pub(super) spec_hits: AtomicU64,
    /// Cumulative speculator miss count: predicted expert IDs not
    /// contained in the gate's actual top-K.
    pub(super) spec_misses: AtomicU64,
    /// Cumulative count of tokens for which the speculator's **top-1**
    /// prediction matched the gate's actual top-1 routed expert.
    /// Mirrors the `mer_speculator_accuracy_total` Prometheus counter.
    pub(super) spec_top1_matches: AtomicU64,
    /// Cumulative count of tokens for which the speculator was
    /// invoked. Denominator of the top-1 accuracy ratio.
    pub(super) spec_tokens: AtomicU64,
    /// Per-layer expert co-occurrence matrix — the **affinity** arm.
    /// When present (and the model exposes a layer-qualified id
    /// geometry), `moe_step` records the layer's routed set into the
    /// matrix and `union_prefetch` folds each high-confidence seed's
    /// top co-fired neighbours into the prefetch union. `None` keeps
    /// the engine's behaviour identical to a deployment without the
    /// affinity arm.
    pub(super) affinity: Option<Arc<LayeredExpertAffinity>>,
    /// Number of co-fired neighbours pulled per high-confidence seed
    /// when [`Self::affinity`] is set.
    pub(super) affinity_neighbors_k: usize,
    /// Owned supervisor for the background exponential-decay worker
    /// that ages the affinity matrix. Retained for the engine's
    /// lifetime; dropping it stops the worker. `None` when the
    /// affinity arm is disabled.
    pub(super) affinity_decay: Option<DecayWorkerHandle>,
    /// Tier 1 — **static residency** controller. When present, the
    /// engine pins the hottest `fraction` of experts permanently in the
    /// RAM cache (from an offline profile at startup, or an online hot
    /// set derived from [`Self::route_observations`] after a warmup
    /// window) so a skewed routing distribution can exceed the bare
    /// cache-capacity hit-rate ceiling. `None` disables the feature.
    pub(super) static_residency: Option<crate::residency::StaticResidencyState>,
    /// Tier 3 — **per-layer pre-gate** predictor. When present, every
    /// `moe_step` records the layer-to-layer routing transition and
    /// prefetches the predicted next-layer experts (a high-precision
    /// signal conditioned on the previous layer's actual routing).
    /// `None` disables the feature.
    pub(super) pregate: Option<Arc<crate::pregate::PerLayerPreGate>>,
}

/// Observability: latency histograms, cumulative timing atomics,
/// hit/miss/byte counters, the optional Prometheus sink, and the
/// optional JSONL routing trace writer.
///
/// Lives in its own struct so the locality / speculator code paths
/// can borrow `&EngineMetrics` to record telemetry without grabbing
/// the whole `Engine`, and so future observability work (extra
/// histograms, additional exporters) lands in one cohesive place.
pub(crate) struct EngineMetrics {
    pub(super) counters: Arc<Counters>,
    /// Latency histogram of per-token cycle time, in microseconds.
    pub(super) cycle_hist: parking_lot::Mutex<Histogram<u64>>,
    /// Latency histogram of cache-miss I/O reads, in microseconds.
    pub(super) io_hist: parking_lot::Mutex<Histogram<u64>>,
    /// Latency histogram of per-token compute (FFN forward), in microseconds.
    pub(super) compute_hist: parking_lot::Mutex<Histogram<u64>>,
    /// Aggregate microseconds spent on I/O wait across all tokens (i.e.
    /// the sum of per-token critical-path miss latencies). Lets us
    /// report `avg_io_wait_us` and "% of token time on I/O" without
    /// re-deriving them from the histogram.
    pub(super) total_io_wait_us: AtomicU64,
    /// Aggregate microseconds spent on per-token compute across all tokens.
    pub(super) total_compute_us: AtomicU64,
    /// Aggregate microseconds spent on per-token cycle (compute + I/O wait
    /// + scheduling overhead) across all tokens.
    pub(super) total_cycle_us: AtomicU64,
    /// Cumulative microseconds spent on the SSD critical-path stall —
    /// the wall-clock window during which the engine was blocked
    /// waiting for cache-miss reads to land. Distinct from
    /// `total_io_wait_us` only in that it's exported as its own
    /// Prometheus histogram (`mer_ssd_stall_seconds`).
    pub(super) total_ssd_stall_us: AtomicU64,
    /// Number of tokens processed (i.e. `Engine::generate` calls).
    pub(super) tokens_processed: AtomicU64,
    /// Optional Prometheus metrics sink. When present, the locality
    /// hit / miss counters and speculator hit / miss counters are
    /// updated alongside the per-Engine atomics.
    pub(super) prom: Option<Metrics>,
    /// Optional JSONL trace sink. When set, every `generate` call
    /// appends one record. See [`TraceWriter`] and gist Phase 6.
    pub(super) trace_writer: parking_lot::RwLock<Option<Arc<TraceWriter>>>,
}

impl EngineCore {
    fn new(
        cache: Arc<MultiLayerExpertCache>,
        pool: BufferPool,
        storage: Arc<NvmeStorage>,
        router: Router,
        predictor: Arc<PredictiveLoader>,
        shape: ModelShape,
        options: EngineOptions,
        execution_context: Arc<crate::backend::ExecutionContext>,
    ) -> Self {
        // Bound the speculative-prefetch semaphore by the buffer
        // pool's *actual* headroom (`pool_slots − cache_slots`), not
        // just by the operator-facing `max_concurrent_prefetches`
        // ceiling. The pool is sized as `cache_slots + headroom` (see
        // `cmd_run` / `cmd_serve` in `main.rs`), so allowing more than
        // `headroom` prefetches in flight at once is a contract
        // violation: every in-flight prefetch holds a `PooledBuffer`
        // for the duration of its I/O, and when the cache is fully
        // pinned a foreground fetch has nowhere to land — surfacing as
        // the `expert fetch starved: buffer pool exhausted with cache
        // pinned` panic at `Engine::fetch` even though
        // `max_concurrent_prefetches=64` looks innocuous on paper.
        //
        // Take the min of the user ceiling and the pool headroom, then
        // *reserve one headroom slot exclusively for the critical
        // path*. Clamping only to the full headroom is not enough: when
        // the cache is fully pinned and prefetch is running at full
        // concurrency, every headroom buffer is held by an in-flight
        // prefetch for the duration of its (multi-millisecond, 672 MB)
        // read, so a foreground miss has nowhere to land and the engine
        // panics at `Engine::fetch`. Subtracting one guarantees there is
        // always at least one buffer a prefetch can never take, so a
        // foreground fetch is assured a slot even under worst-case
        // pinning + saturated speculation. If the reserved-slot
        // subtraction leaves zero permits (headroom ≤ 1) prefetch is
        // disabled entirely for this configuration rather than starving
        // the critical path.
        let pool_headroom = pool.capacity().saturating_sub(cache.capacity());
        let prefetch_permits = if pool.shadow_capacity() > 0 {
            // Double-buffered layout: speculative look-ahead prefetches
            // draw exclusively from the **shadow** (Buffer B) half of the
            // pool, which is fully reserved for them. The primary (Buffer
            // A) half backs the resident LRU and the foreground miss
            // path, so speculation can never starve a real cache miss no
            // matter how many prefetches are in flight. Bound concurrency
            // by the shadow capacity directly — there is no need to
            // reserve a primary headroom slot because the two halves no
            // longer share buffers.
            options
                .max_concurrent_prefetches
                .min(pool.shadow_capacity())
        } else {
            // Legacy single-pool layout: speculation shares the primary
            // pool with the resident LRU, so reserve one headroom slot
            // exclusively for the critical path (see the long-form
            // rationale above) by subtracting one from the headroom.
            options
                .max_concurrent_prefetches
                .min(pool_headroom.saturating_sub(1))
        };
        let governor = Arc::new(crate::prefetch_governor::PrefetchGovernor::new(
            options.prefetch_governor,
            crate::prefetch_governor::GovernorConfig {
                precision_floor: options.prefetch_precision_floor,
                contention_weight: options.prefetch_contention_weight,
                ..crate::prefetch_governor::GovernorConfig::default()
            },
        ));
        // Tier 4: flip the RAM cache into cost-aware (lowest-heat)
        // eviction when requested. Off by default ⇒ pure LRU.
        cache.set_cost_aware(options.cost_aware_eviction);
        Self {
            cache,
            pool,
            storage,
            router,
            predictor,
            shape,
            options,
            gpu_cache: None,
            gpu_native_residency: None,
            gpu_promotion_tx: None,
            in_flight: Arc::new(DashMap::new()),
            prefetch_semaphore: Arc::new(tokio::sync::Semaphore::new(prefetch_permits)),
            execution_context,
            routed_expert_gpu_failure_policy: RoutedExpertGpuFailurePolicy::default(),
            governor,
        }
    }
}

impl EngineSpeculation {
    fn new(speculator_topk_default: usize) -> Self {
        Self {
            alias_map: None,
            alias_redirects: AtomicU64::new(0),
            route_observations: DashMap::new(),
            route_observation_tokens: AtomicU64::new(0),
            markov_ring: parking_lot::Mutex::new(MarkovRing::default()),
            locality: None,
            locality_pinned: parking_lot::Mutex::new(HashSet::new()),
            static_pinned: parking_lot::Mutex::new(HashSet::new()),
            locality_threshold_pct: LocalityMonitor::DEFAULT_THRESHOLD_PCT,
            locality_hits: AtomicU64::new(0),
            locality_misses: AtomicU64::new(0),
            speculator: None,
            speculator_topk: speculator_topk_default,
            pipeline_depth: DEFAULT_PIPELINE_DEPTH,
            spec_hits: AtomicU64::new(0),
            spec_misses: AtomicU64::new(0),
            spec_top1_matches: AtomicU64::new(0),
            spec_tokens: AtomicU64::new(0),
            affinity: None,
            affinity_neighbors_k: 0,
            affinity_decay: None,
            static_residency: None,
            pregate: None,
        }
    }
}

impl EngineMetrics {
    fn new() -> Self {
        // 1us..60s, 3 sig figs — wide enough for cache hits (sub-ms)
        // and slow SSD stalls (multi-second worst case) alike.
        let mk_hist = || {
            parking_lot::Mutex::new(
                Histogram::new_with_bounds(1, 60_000_000, 3)
                    .expect("hdr histogram bounds (1us..60s, 3 sig figs) are valid"),
            )
        };
        Self {
            counters: Arc::new(Counters::default()),
            cycle_hist: mk_hist(),
            io_hist: mk_hist(),
            compute_hist: mk_hist(),
            total_io_wait_us: AtomicU64::new(0),
            total_compute_us: AtomicU64::new(0),
            total_cycle_us: AtomicU64::new(0),
            total_ssd_stall_us: AtomicU64::new(0),
            tokens_processed: AtomicU64::new(0),
            prom: None,
            trace_writer: parking_lot::RwLock::new(None),
        }
    }
}

/// Explicit lifecycle for asynchronous work owned by an [`Engine`].
///
/// Background promotion and speculative-prefetch tasks used to be detached
/// `tokio::spawn` calls. Several of those futures own an `Arc<Engine>`, so
/// dropping the caller's last runtime handle was not sufficient to retire the
/// engine or any resource family reachable through it. This controller stops
/// new admissions, cooperatively cancels every registered future, and retains
/// abort handles only as a bounded fallback for work that does not yield
/// promptly. It never owns a task future or an `Arc<Engine>` itself.
struct EngineBackgroundTasks {
    shutdown_requested: AtomicBool,
    active: AtomicUsize,
    state: parking_lot::Mutex<EngineBackgroundTaskState>,
    shutdown_notify: Notify,
    idle_notify: Notify,
}

#[derive(Default)]
struct EngineBackgroundTaskState {
    abort_handles: Vec<tokio::task::AbortHandle>,
}

struct EngineBackgroundTaskGuard {
    owner: Arc<EngineBackgroundTasks>,
}

impl Drop for EngineBackgroundTaskGuard {
    fn drop(&mut self) {
        if self.owner.active.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.owner.idle_notify.notify_waiters();
        }
    }
}

impl EngineBackgroundTasks {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            shutdown_requested: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            state: parking_lot::Mutex::new(EngineBackgroundTaskState::default()),
            shutdown_notify: Notify::new(),
            idle_notify: Notify::new(),
        })
    }

    #[inline]
    fn accepts_work(&self) -> bool {
        !self.shutdown_requested.load(Ordering::Acquire)
    }

    /// Register and detach one runtime-owned background future.
    ///
    /// The state lock makes the admission check and handle registration atomic
    /// with respect to `shutdown`: once shutdown flips the flag while holding
    /// the same lock, no later task can escape registration.
    fn spawn<F>(self: &Arc<Self>, future: F) -> bool
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let mut state = self.state.lock();
        if !self.accepts_work() {
            return false;
        }
        state.abort_handles.retain(|handle| !handle.is_finished());
        self.active.fetch_add(1, Ordering::AcqRel);
        let owner = self.clone();
        let cancellation = self.clone();
        let handle = tokio::spawn(async move {
            let _guard = EngineBackgroundTaskGuard { owner };
            tokio::select! {
                biased;
                _ = cancellation.cancelled() => {}
                _ = future => {}
            }
        });
        state.abort_handles.push(handle.abort_handle());
        true
    }

    async fn cancelled(&self) {
        if self.shutdown_requested.load(Ordering::Acquire) {
            return;
        }
        let notified = self.shutdown_notify.notified();
        if self.shutdown_requested.load(Ordering::Acquire) {
            return;
        }
        notified.await;
    }

    async fn wait_for_idle(&self, timeout: Duration) -> bool {
        let wait = async {
            loop {
                let notified = self.idle_notify.notified();
                if self.active.load(Ordering::Acquire) == 0 {
                    return;
                }
                notified.await;
            }
        };
        tokio::time::timeout(timeout, wait).await.is_ok()
    }

    async fn shutdown(&self) -> Result<(), String> {
        const COOPERATIVE_GRACE: Duration = Duration::from_secs(1);
        const ABORT_GRACE: Duration = Duration::from_secs(4);

        let abort_handles = {
            let mut state = self.state.lock();
            self.shutdown_requested.store(true, Ordering::Release);
            state.abort_handles.retain(|handle| !handle.is_finished());
            state.abort_handles.clone()
        };
        self.shutdown_notify.notify_waiters();

        if self.wait_for_idle(COOPERATIVE_GRACE).await {
            self.state.lock().abort_handles.clear();
            return Ok(());
        }

        // Speculation and logical-GPU promotion are best-effort background
        // work. Aborting their futures is resource-safe: buffer, semaphore,
        // singleflight, and promotion claims all have drop/error cleanup.
        for handle in abort_handles {
            handle.abort();
        }
        if self.wait_for_idle(ABORT_GRACE).await {
            self.state.lock().abort_handles.clear();
            Ok(())
        } else {
            Err(format!(
                "engine background tasks remained active after controlled shutdown: active={}",
                self.active.load(Ordering::Acquire)
            ))
        }
    }
}

/// Top-level façade: composes [`EngineCore`] (MoE/IO), [`EngineSpeculation`]
/// (aliasing, locality, neural speculator) and [`EngineMetrics`]
/// (histograms, counters, Prometheus / trace sinks).
///
/// All public methods stay on `Engine` so callers see the same API
/// surface they always did; `generate` and `moe_step` are the
/// cross-cutting flows that orchestrate across all three sub-objects.
pub struct Engine {
    pub(crate) core: EngineCore,
    pub(crate) speculation: EngineSpeculation,
    pub(crate) metrics: EngineMetrics,
    background_tasks: Arc<EngineBackgroundTasks>,
    diagnostic_cpu_q4_boundary_emulation: std::sync::atomic::AtomicBool,
    diagnostic_cpu_q4_boundary_emulated_dispatches: AtomicU64,
    diagnostic_route_capture_armed: std::sync::atomic::AtomicBool,
    diagnostic_route_capture: parking_lot::Mutex<Option<DiagnosticRouteCaptureArm>>,
    gpu_native_demand_source_qualification:
        parking_lot::RwLock<Option<Arc<GpuNativeDemandSourceQualification>>>,
    production_demand_source: Arc<ProductionDemandSourceTelemetry>,
    #[cfg(test)]
    production_batch_test_hooks: parking_lot::Mutex<ProductionBatchTestHooks>,
}

#[cfg(test)]
#[derive(Default)]
struct ProductionBatchTestHooks {
    after_claims: Option<Arc<tokio::sync::Barrier>>,
    after_claim_rollback: Option<Arc<tokio::sync::Barrier>>,
    after_buffers: Option<Arc<tokio::sync::Barrier>>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuQ4BoundaryEmulationSnapshot {
    pub enabled: bool,
    pub routed_expert_dispatches: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct RoutedFfnDiagnosticCapture {
    pub token_idx: u64,
    pub layer: u32,
    pub input_bits: Vec<u32>,
    pub expert_ids: Vec<u32>,
    pub routing_weight_bits: Vec<u32>,
}

struct DiagnosticRouteCaptureArm {
    target_token_idx: u64,
    captured: Option<RoutedFfnDiagnosticCapture>,
}

/// Run a CPU/GPU-blocking expert compute closure while *donating* the
/// current tokio worker thread to it.
///
/// The per-expert FFN forward (and the synchronous wgpu dispatch +
/// readback behind `Backend::expert_matmul`) takes ~10ms+ per layer.
/// Running it inline on a tokio worker pins that worker for the whole
/// slice, and with a fused decode batch every worker can be pinned at
/// once — starving the `tokio::spawn`-ed speculative prefetch tasks
/// exactly when they need to run to bury the next layer's SSD reads
/// under compute. `block_in_place` flags the worker as blocked so the
/// scheduler migrates other ready tasks (the prefetches) to sibling
/// workers, the same discipline `io_provider` already applies to its
/// `pread(2)` calls.
///
/// `block_in_place` panics on a `current_thread` runtime (used by
/// plain `#[tokio::test]`), so fall back to running inline there —
/// a single-threaded runtime has no sibling workers to protect anyway.
fn run_compute_donated<R>(f: impl FnOnce() -> R) -> R {
    use tokio::runtime::{Handle, RuntimeFlavor};
    match Handle::try_current() {
        Ok(h) if h.runtime_flavor() == RuntimeFlavor::MultiThread => tokio::task::block_in_place(f),
        _ => f(),
    }
}

fn accumulate_ordered_f16_outputs(
    outputs: &[half::f16],
    weights: &[f32],
    d_model: usize,
    out: &mut Vec<f32>,
) -> bool {
    let Some(expected) = weights.len().checked_mul(d_model) else {
        return false;
    };
    if outputs.len() != expected {
        return false;
    }
    out.clear();
    out.resize(d_model, 0.0);
    for (weight, values) in weights.iter().zip(outputs.chunks_exact(d_model)) {
        if *weight != 0.0 {
            for (dst, value) in out.iter_mut().zip(values) {
                *dst += *weight * value.to_f32();
            }
        }
    }
    true
}

enum MoeStepOutputMode<'a> {
    PerExpert,
    WeightedInto {
        weights: &'a [f32],
        out: &'a mut Vec<f32>,
    },
}

enum MoeStepResult {
    PerExpert(Vec<HiddenState>),
    WeightedInto,
}

/// Error taxonomy for a failed real-model MoE step (hardening pass,
/// Part A1). Under strict production mode (the default,
/// `EngineOptions::policy.allow_degraded_experts == false`) any required
/// routed expert that fails to load or execute fails the whole step
/// with one of these variants instead of silently degrading the
/// mixture. These are recoverable *request-level* failures: callers
/// map them to an HTTP 500 (server) or a CLI error (bench) — the
/// process never panics for them.
#[derive(Debug)]
pub enum MoeStepError {
    /// A routed expert could not be read from storage even after
    /// retries (I/O failure, checksum failure, truncated payload, …).
    ExpertFetch {
        layer: u32,
        expert: u32,
        source: ExpertReadError,
    },
    /// A resident expert failed weight-layout validation, quantized
    /// preparation, or FFN compute.
    ExpertCompute {
        layer: u32,
        expert: u32,
        source: ExpertWeightsError,
    },
    /// Strict GPU routed-expert dispatch failure, kept distinct from storage,
    /// CPU weight, attention, degraded-substitution, and startup errors.
    GpuExpertDispatch {
        source: crate::backend::GpuExpertDispatchError,
    },
}

impl std::fmt::Display for MoeStepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeStepError::ExpertFetch {
                layer,
                expert,
                source,
            } => write!(
                f,
                "routed expert {expert} (layer {layer}) failed to load: {source}"
            ),
            MoeStepError::ExpertCompute {
                layer,
                expert,
                source,
            } => write!(
                f,
                "routed expert {expert} (layer {layer}) failed to execute: {source}"
            ),
            MoeStepError::GpuExpertDispatch { source } => write!(f, "{source}"),
        }
    }
}

impl std::error::Error for MoeStepError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MoeStepError::ExpertFetch { source, .. } => Some(source),
            MoeStepError::ExpertCompute { source, .. } => Some(source),
            MoeStepError::GpuExpertDispatch { source } => Some(source),
        }
    }
}

/// Dispatch a single per-expert SwiGLU forward pass according to
/// `dtype`. For `Q4_0` / `Q4K` and `use_qmm = true` the
/// `QMatMul`-based path is tried first and the dequant path is used
/// as a fallback when QMM returns an error (this can happen on a
/// corrupt block stream where dequant has more lenient bounds
/// checks). Q8_0 always uses the validated native resident-byte path;
/// every other dtype calls its normal entry point directly.
#[allow(clippy::too_many_arguments)]
fn dispatch_expert_forward(
    dtype: WeightDtype,
    use_qmm: bool,
    token_idx: u64,
    r: &ExpertResident,
    x: &[f32],
    d_model: usize,
    d_ff: usize,
    // Engine-scoped truncated-payload tolerance in bytes
    // (`RealInferencePolicy::expert_size_tolerance`); `0` = strict.
    q4_tolerance: usize,
    timings: Option<&crate::stage_timing::StageTimings>,
) -> Result<(InferenceOutput, HiddenState), ExpertWeightsError> {
    // One-time diagnostic: log the actual resident buffer size against
    // the size the engine expects for this dtype/shape on the very
    // first expert forward pass (gist Fix 3). This surfaces on-disk
    // size mismatches immediately at INFO, instead of only becoming
    // visible once an expert is skipped with a "buffer too small"
    // warning.
    static LOG_FIRST_EXPERT_SIZE_ONCE: std::sync::Once = std::sync::Once::new();
    LOG_FIRST_EXPERT_SIZE_ONCE.call_once(|| {
        let actual = r.data().len();
        let expected = if dtype == WeightDtype::Mixed {
            r.buffer.as_slice().len()
        } else {
            crate::inference::expert_weight_bytes_for(d_model, d_ff, dtype)
        };
        info!(
            expert = r.id,
            dtype = dtype.as_str(),
            payload_bytes = actual,
            physical_slot_bytes = r.buffer.as_slice().len(),
            expected_bytes = expected,
            d_model,
            d_ff,
            "first expert load: actual vs expected buffer size"
        );
    });
    match dtype {
        // Phase 3 compute plane: when built with `--features cuda`,
        // route the F32 SwiGLU through candle-core's CUDA backend via
        // `run_inference_gpu`, which transparently falls back to the CPU
        // `run_inference` kernel at runtime when no device is present.
        // Without the feature this is a direct call to the CPU path, so
        // default builds are byte-for-byte unchanged.
        #[cfg(feature = "cuda")]
        WeightDtype::F32 => crate::inference::run_inference_gpu(token_idx, r, x, d_model, d_ff),
        #[cfg(not(feature = "cuda"))]
        WeightDtype::F32 => crate::inference::run_inference(token_idx, r, x, d_model, d_ff),
        WeightDtype::F16 => run_inference_f16(token_idx, r, x, d_model, d_ff),
        WeightDtype::Int8 => run_inference_int8(token_idx, r, x, d_model, d_ff),
        WeightDtype::Q4K
            if use_qmm && d_model % Q4K_BLOCK_ELEMS == 0 && d_ff % Q4K_BLOCK_ELEMS == 0 =>
        {
            match run_inference_q4k_qmm(token_idx, r, x, d_model, d_ff) {
                Ok(v) => Ok(v),
                Err(e) => {
                    debug!(error = %e, "QMatMul Q4_K path failed; falling back to dequant");
                    run_inference_q4k(token_idx, r, x, d_model, d_ff)
                }
            }
        }
        WeightDtype::Q4K => run_inference_q4k(token_idx, r, x, d_model, d_ff),
        WeightDtype::Q4_0
            if use_qmm && d_model % Q4_0_BLOCK_ELEMS == 0 && d_ff % Q4_0_BLOCK_ELEMS == 0 =>
        {
            match run_inference_q4_0_qmm(token_idx, r, x, d_model, d_ff, q4_tolerance) {
                Ok(v) => Ok(v),
                Err(e) => {
                    debug!(error = %e, "QMatMul Q4_0 path failed; falling back to dequant");
                    run_inference_q4_0(token_idx, r, x, d_model, d_ff, q4_tolerance)
                }
            }
        }
        WeightDtype::Q4_0 => run_inference_q4_0(token_idx, r, x, d_model, d_ff, q4_tolerance),
        // Native Q8_0 is always the production path. It validates the full
        // payload before dispatch and handles non-row-aligned shapes with a
        // scalar block-stream reference; malformed residents remain errors.
        WeightDtype::Q8_0 => {
            run_inference_q8_0_direct_with_timing(token_idx, r, x, d_model, d_ff, timings)
        }
        WeightDtype::Q5K => run_inference_q5k(token_idx, r, x, d_model, d_ff),
        WeightDtype::Q6K => run_inference_q6k(token_idx, r, x, d_model, d_ff),
        WeightDtype::BF16 => run_inference_bf16(token_idx, r, x, d_model, d_ff),
        WeightDtype::MXFP4 => run_inference_mxfp4(token_idx, r, x, d_model, d_ff),
        WeightDtype::Mixed => run_inference_mixed_quant(token_idx, r, x, d_model, d_ff),
    }
}

fn summarise_output_like_cpu(token_idx: u64, expert_id: u32, y: &[f32]) -> InferenceOutput {
    let mut sum_sq = 0.0f64;
    for &v in y {
        sum_sq += (v as f64) * (v as f64);
    }
    let out_norm = sum_sq.sqrt() as f32;
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    let mut digest = FNV_OFFSET ^ token_idx ^ (expert_id as u64);
    for &v in y {
        digest ^= v.to_bits() as u64;
        digest = digest.wrapping_mul(FNV_PRIME);
    }
    InferenceOutput {
        expert_id,
        digest,
        out_norm,
    }
}

impl Engine {
    pub fn new(
        cache: Arc<MultiLayerExpertCache>,
        pool: BufferPool,
        storage: Arc<NvmeStorage>,
        router: Router,
        predictor: Arc<PredictiveLoader>,
        shape: ModelShape,
    ) -> Self {
        Self::with_options(
            cache,
            pool,
            storage,
            router,
            predictor,
            shape,
            EngineOptions::default(),
        )
    }

    pub fn with_options(
        cache: Arc<MultiLayerExpertCache>,
        pool: BufferPool,
        storage: Arc<NvmeStorage>,
        router: Router,
        predictor: Arc<PredictiveLoader>,
        shape: ModelShape,
        options: EngineOptions,
    ) -> Self {
        Self::with_options_and_execution_context(
            cache,
            pool,
            storage,
            router,
            predictor,
            shape,
            options,
            crate::backend::cpu_execution_context(),
        )
    }

    /// Construct an engine that consumes an already-resolved authoritative
    /// execution context. Production startup uses this path so planning,
    /// reporting, real-model attention, and routed-expert dispatch all share
    /// the same context identity and backend instances.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options_and_execution_context(
        cache: Arc<MultiLayerExpertCache>,
        pool: BufferPool,
        storage: Arc<NvmeStorage>,
        router: Router,
        predictor: Arc<PredictiveLoader>,
        shape: ModelShape,
        options: EngineOptions,
        execution_context: Arc<crate::backend::ExecutionContext>,
    ) -> Self {
        let engine_expert_spec = crate::backend::RoutedExpertGpuSpec {
            dtype: options.dtype,
            d_model: shape.d_model,
            d_ff: shape.d_ff,
        };
        if execution_context.plan().routed_experts() == crate::backend::ExecutionPlane::Gpu {
            assert_eq!(
                execution_context.plan().routed_expert_gpu_spec(),
                engine_expert_spec,
                "resolved GPU routed-expert plan does not match Engine dtype/geometry"
            );
            assert!(
                crate::backend::routed_expert_gpu_compatibility(engine_expert_spec).is_ok(),
                "resolved GPU routed-expert plan is incompatible with Engine dtype/geometry"
            );
        }
        let speculator_topk_default = router.top_k();
        let production_demand_source = Arc::new(ProductionDemandSourceTelemetry::default());
        production_demand_source.reset();
        Self {
            core: EngineCore::new(
                cache,
                pool,
                storage,
                router,
                predictor,
                shape,
                options,
                execution_context,
            ),
            speculation: EngineSpeculation::new(speculator_topk_default),
            metrics: EngineMetrics::new(),
            background_tasks: EngineBackgroundTasks::new(),
            diagnostic_cpu_q4_boundary_emulation: std::sync::atomic::AtomicBool::new(false),
            diagnostic_cpu_q4_boundary_emulated_dispatches: AtomicU64::new(0),
            diagnostic_route_capture_armed: std::sync::atomic::AtomicBool::new(false),
            diagnostic_route_capture: parking_lot::Mutex::new(None),
            gpu_native_demand_source_qualification: parking_lot::RwLock::new(None),
            production_demand_source,
            #[cfg(test)]
            production_batch_test_hooks: parking_lot::Mutex::new(
                ProductionBatchTestHooks::default(),
            ),
        }
    }

    /// Enable CPU-only emulation of the production routed-expert f16 input
    /// and output boundaries. Only isolated numerical-diagnostic workers call
    /// this; ordinary CPU serving retains its existing Q4 execution path.
    pub(crate) fn enable_cpu_q4_boundary_emulation(&self) -> Result<(), String> {
        if self.execution_context().plan().routed_experts()
            != crate::backend::ExecutionPlane::Cpu
            || self.core.options.dtype != WeightDtype::Q4_0
            || self.core.options.policy != crate::inference::RealInferencePolicy::STRICT
        {
            return Err(
                "Q4 boundary emulation requires a strict CPU Q4_0 routed-expert plan"
                    .to_string(),
            );
        }
        self.diagnostic_cpu_q4_boundary_emulation
            .store(true, Ordering::Release);
        Ok(())
    }

    pub(crate) fn cpu_q4_boundary_emulation_snapshot(
        &self,
    ) -> CpuQ4BoundaryEmulationSnapshot {
        CpuQ4BoundaryEmulationSnapshot {
            enabled: self
                .diagnostic_cpu_q4_boundary_emulation
                .load(Ordering::Acquire),
            routed_expert_dispatches: self
                .diagnostic_cpu_q4_boundary_emulated_dispatches
                .load(Ordering::Acquire),
        }
    }

    /// Arm a one-shot, diagnostic-only capture at the routed-FFN boundary.
    /// Normal serving never arms this slot, so its hot path performs only one
    /// relaxed atomic load and never locks or clones activations.
    pub(crate) fn arm_layer0_route_capture(&self, target_token_idx: u64) -> Result<(), String> {
        let mut slot = self.diagnostic_route_capture.lock();
        if slot.is_some() {
            return Err("routed-FFN diagnostic capture is already armed".to_string());
        }
        *slot = Some(DiagnosticRouteCaptureArm {
            target_token_idx,
            captured: None,
        });
        self.diagnostic_route_capture_armed
            .store(true, Ordering::Relaxed);
        Ok(())
    }

    pub(crate) fn take_layer0_route_capture(
        &self,
    ) -> Option<RoutedFfnDiagnosticCapture> {
        self.diagnostic_route_capture_armed
            .store(false, Ordering::Relaxed);
        self.diagnostic_route_capture
            .lock()
            .take()
            .and_then(|arm| arm.captured)
    }

    fn capture_layer0_route_if_armed(
        &self,
        token_idx: u64,
        layer: u32,
        x: &[f32],
        experts: &[u32],
        weights: &[f32],
    ) {
        if !self
            .diagnostic_route_capture_armed
            .load(Ordering::Relaxed)
        {
            return;
        }
        let mut slot = self.diagnostic_route_capture.lock();
        let Some(arm) = slot.as_mut() else {
            return;
        };
        if layer == 0 && token_idx == arm.target_token_idx && arm.captured.is_none() {
            arm.captured = Some(RoutedFfnDiagnosticCapture {
                token_idx,
                layer,
                input_bits: x.iter().copied().map(f32::to_bits).collect(),
                expert_ids: experts.to_vec(),
                routing_weight_bits: weights.iter().copied().map(f32::to_bits).collect(),
            });
        }
    }

    pub fn execution_context(&self) -> &Arc<crate::backend::ExecutionContext> {
        &self.core.execution_context
    }

    fn routed_expert_backend(&self) -> &Arc<crate::backend::BackendBox> {
        self.core.execution_context.routed_expert_backend()
    }

    /// Whether the configured expert dtype and shape are eligible for the GPU
    /// `Backend::expert_matmul` fast path. F32 and Q4_0 must fit the fixed GPU
    /// expert workspace. Q4_0 additionally requires both `d_model` and `d_ff`
    /// to be block-aligned so every matrix row starts on a 32-element block
    /// boundary, as assumed by `matmul_q4_0.wgsl`. All other dtypes stay on
    /// the CPU path.
    fn gpu_eligible_dtype(&self) -> bool {
        crate::backend::routed_expert_gpu_compatibility(
            crate::backend::RoutedExpertGpuSpec {
                dtype: self.core.options.dtype,
                d_model: self.core.shape.d_model,
                d_ff: self.core.shape.d_ff,
            },
        )
        .is_ok()
    }

    /// Select the policy before placing the engine behind an `Arc`; there is
    /// intentionally no runtime setter.
    pub fn with_routed_expert_gpu_failure_policy(
        mut self,
        policy: RoutedExpertGpuFailurePolicy,
    ) -> Self {
        self.core.routed_expert_gpu_failure_policy = policy;
        self
    }

    pub fn routed_expert_gpu_failure_policy(&self) -> RoutedExpertGpuFailurePolicy {
        self.core.routed_expert_gpu_failure_policy
    }

    /// Exact MER-owned routed-expert GPU weight/workspace ledger for PR5.
    /// Returns `None` when the resolved execution context has no GPU backend.
    pub fn gpu_expert_memory_snapshot(
        &self,
    ) -> Option<crate::backend::GpuExpertMemorySnapshot> {
        self.core.execution_context.gpu_expert_memory_snapshot()
    }

    /// Identity of the adapter already selected by the authoritative
    /// execution context. This never performs adapter rediscovery.
    pub fn gpu_device_identity(&self) -> Option<crate::backend::GpuDeviceIdentity> {
        self.core.execution_context.gpu_device_identity()
    }

    /// Monotonic routed-expert GPU transfer/submission counters.
    pub fn gpu_expert_io_snapshot(&self) -> Option<crate::backend::GpuExpertIoSnapshot> {
        self.core.execution_context.gpu_expert_io_snapshot()
    }

    fn q4_parity_dispatch_snapshot(
        &self,
        expert_id: u32,
    ) -> Result<crate::q4_parity::CompleteDispatchSnapshot, String> {
        let backend = self.routed_expert_backend();
        let logical_generation = self
            .execution_context()
            .gpu_expert_cache()
            .current_admission(expert_id)
            .map(|admission| admission.generation());
        let memory = self
            .gpu_expert_memory_snapshot()
            .ok_or_else(|| "authoritative GPU backend has no physical-memory snapshot".to_string())?;
        let gpu_io = self
            .gpu_expert_io_snapshot()
            .ok_or_else(|| "authoritative GPU backend has no routed-expert I/O snapshot".to_string())?;
        Ok(crate::q4_parity::CompleteDispatchSnapshot {
            logical_generation,
            physical: backend.gpu_physical_expert_residency(expert_id),
            memory,
            gpu_io,
            routed: self.routed_expert_execution_snapshot(),
        })
    }

    /// Qualification-only complete Q4_0 expert execution. It fetches the
    /// selected global expert through production storage, computes the
    /// fail-loud authoritative CPU reference from that exact stripped payload,
    /// and sends each vector through the normal strict routed-expert boundary.
    /// Normal serving never calls this method.
    pub(crate) async fn qualify_q4_0_complete_expert(
        self: &Arc<Self>,
        global_expert_id: u32,
        layer_index: u32,
        inputs: &[Vec<f32>],
    ) -> Result<crate::q4_parity::CompleteExpertExecution, String> {
        use sha2::{Digest, Sha256};

        if self.core.options.dtype != WeightDtype::Q4_0 {
            return Err(format!(
                "complete-expert parity requires Q4_0, got {}",
                self.core.options.dtype.as_str()
            ));
        }
        if self.execution_context().plan().routed_experts()
            != crate::backend::ExecutionPlane::Gpu
            || self.routed_expert_gpu_failure_policy()
                != RoutedExpertGpuFailurePolicy::StrictFailClosed
            || self.core.options.policy != crate::inference::RealInferencePolicy::STRICT
        {
            return Err(
                "complete-expert parity requires a strict fail-closed GPU routed-expert plan"
                    .to_string(),
            );
        }
        if inputs.len() < 2 {
            return Err("complete-expert parity requires at least two input vectors".to_string());
        }
        if inputs.iter().any(|input| {
            input.len() != self.core.shape.d_model
                || input.iter().any(|value| {
                    !value.is_finite() || half::f16::from_f32(*value).to_f32() != *value
                })
        }) {
            return Err(format!(
                "complete-expert inputs must contain exactly {} finite f16-exact values",
                self.core.shape.d_model
            ));
        }

        let block_align = self.core.storage.config().block_align;
        crate::q4_parity::validate_checkpoint_block_align(block_align)?;

        self.core
            .storage
            .validate_expert_file_layout(std::iter::once(global_expert_id))
            .map_err(|error| {
                format!(
                    "global expert {global_expert_id} does not have the exact configured checkpoint file size: {error}"
                )
            })?;

        let expected_payload = crate::inference::expert_weight_bytes_for(
            self.core.shape.d_model,
            self.core.shape.d_ff,
            WeightDtype::Q4_0,
        );
        let checkpoint_payload_bytes =
            crate::q4_parity::aligned_checkpoint_payload_bytes(expected_payload, block_align)?;
        let resident = self
            .fetch_with_retry(global_expert_id)
            .await
            .map_err(|error| {
                format!("failed to fetch global expert {global_expert_id}: {error}")
            })?;
        if resident.data().len() != checkpoint_payload_bytes {
            return Err(format!(
                "global expert {global_expert_id} checkpoint payload has {} bytes, expected exactly {checkpoint_payload_bytes} ({expected_payload} canonical Q4_0 bytes plus alignment padding)",
                resident.data().len(),
            ));
        }
        let (canonical_payload, alignment_padding) = resident.data().split_at(expected_payload);
        let alignment_padding_bytes = alignment_padding.len();
        if alignment_padding.iter().any(|byte| *byte != 0) {
            return Err(format!(
                "global expert {global_expert_id} has non-zero bytes in its {}-byte checkpoint alignment pad",
                alignment_padding.len()
            ));
        }
        let payload_sha256 = format!("{:x}", Sha256::digest(canonical_payload));
        let checkpoint_payload_sha256 = format!("{:x}", Sha256::digest(resident.data()));

        // Compute every CPU oracle before the first GPU snapshot. This keeps
        // each recorded before/after pair immediately adjacent to exactly one
        // production GPU dispatch.
        let cpu_outputs: Result<Vec<Vec<f32>>, String> = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                crate::inference::q4_0_cpu_reference_forward(
                    canonical_payload,
                    input,
                    self.core.shape.d_model,
                    self.core.shape.d_ff,
                )
                .map_err(|error| format!("CPU reference vector {index} failed: {error}"))
            })
            .collect();
        let cpu_outputs = cpu_outputs?;

        let mut dispatches = Vec::with_capacity(inputs.len());
        for (input, cpu_f32) in inputs.iter().cloned().zip(cpu_outputs) {
            // Mirror the authoritative production routing probe exactly once.
            // This owns logical admission hit/miss telemetry and LRU recency;
            // the backend remains non-mutating with respect to logical LRU.
            let _ = self
                .execution_context()
                .gpu_expert_cache()
                .get(global_expert_id);
            let before = self.q4_parity_dispatch_snapshot(global_expert_id)?;
            let gpu_f16 = run_compute_donated(|| {
                self.forward_moe_resident(0, layer_index, resident.as_ref(), &input, None)
            })
            .map_err(|error| {
                format!(
                    "global expert {global_expert_id} layer {layer_index} GPU dispatch failed: {error}"
                )
            })?;
            let after = self.q4_parity_dispatch_snapshot(global_expert_id)?;
            dispatches.push(crate::q4_parity::CompleteDispatchExecution {
                input,
                cpu_f32,
                gpu_f16,
                before,
                after,
            });
        }

        Ok(crate::q4_parity::CompleteExpertExecution {
            d_model: self.core.shape.d_model,
            d_ff: self.core.shape.d_ff,
            checkpoint_block_align: block_align,
            payload_bytes: expected_payload,
            checkpoint_payload_bytes,
            alignment_padding_bytes,
            payload_sha256,
            checkpoint_payload_sha256,
            dispatches,
        })
    }

    /// Diagnostic-only replay of one frozen exact-f32 GPU expert input through
    /// the old boundary-emulated CPU reference, the ordinary production CPU
    /// Q4 reference, and the existing current-GPU arithmetic emulator. The
    /// canonical payload hash is checked before any arithmetic is performed.
    pub(crate) async fn audit_q4_0_f32_reference_boundary(
        self: &Arc<Self>,
        global_expert_id: u32,
        exact_gpu_f32_input: &[f32],
        expected_canonical_payload_sha256: &str,
    ) -> Result<crate::gpu_native_f32_reference_boundary_audit::Q4F32ReferenceReplayBundle, String>
    {
        use sha2::{Digest, Sha256};

        if self.core.options.dtype != WeightDtype::Q4_0
            || self.execution_context().plan().routed_experts()
                != crate::backend::ExecutionPlane::Cpu
            || self.core.options.policy != crate::inference::RealInferencePolicy::STRICT
        {
            return Err(
                "f32 reference-boundary audit requires a strict CPU Q4_0 routed-expert plan"
                    .to_string(),
            );
        }
        if exact_gpu_f32_input.len() != self.core.shape.d_model
            || exact_gpu_f32_input.iter().any(|value| !value.is_finite())
            || expected_canonical_payload_sha256.len() != 64
            || !expected_canonical_payload_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(
                "f32 reference-boundary audit received malformed input or payload identity"
                    .to_string(),
            );
        }
        self.core
            .storage
            .validate_expert_file_layout(std::iter::once(global_expert_id))
            .map_err(|error| {
                format!(
                    "global expert {global_expert_id} does not have the exact configured checkpoint file size: {error}"
                )
            })?;
        let expected_payload = crate::inference::expert_weight_bytes_for(
            self.core.shape.d_model,
            self.core.shape.d_ff,
            WeightDtype::Q4_0,
        );
        let checkpoint_payload_bytes = crate::q4_parity::aligned_checkpoint_payload_bytes(
            expected_payload,
            self.core.storage.config().block_align,
        )?;
        let resident = self
            .fetch_with_retry(global_expert_id)
            .await
            .map_err(|error| format!("failed to fetch global expert {global_expert_id}: {error}"))?;
        if resident.data().len() != checkpoint_payload_bytes {
            return Err(format!(
                "global expert {global_expert_id} checkpoint payload has {} bytes, expected {checkpoint_payload_bytes}",
                resident.data().len()
            ));
        }
        let (payload, padding) = resident.data().split_at(expected_payload);
        if padding.iter().any(|byte| *byte != 0) {
            return Err(format!(
                "global expert {global_expert_id} has non-zero checkpoint alignment padding"
            ));
        }
        let canonical_payload_sha256 = format!("{:x}", Sha256::digest(payload));
        if !canonical_payload_sha256.eq_ignore_ascii_case(expected_canonical_payload_sha256) {
            return Err(format!(
                "global expert {global_expert_id} canonical Q4 payload SHA differs: observed {canonical_payload_sha256} expected {expected_canonical_payload_sha256}"
            ));
        }
        crate::gpu_native_f32_reference_boundary_audit::audit_q4_reference_paths(
            payload,
            exact_gpu_f32_input,
            self.core.shape.d_model,
            self.core.shape.d_ff,
            canonical_payload_sha256,
        )
    }

    /// Diagnostic-only internal-stage attribution for one exact Q4_0 expert.
    /// It fetches the canonical production payload, observes the unchanged
    /// Candle reference sequence, and constructs host-only mixed replays. No
    /// result from this method is admitted back into production execution.
    pub(crate) async fn diagnose_q4_0_expert_stages(
        self: &Arc<Self>,
        global_expert_id: u32,
        gpu_effective_input: &[f32],
        actual_gpu_gate: &[f32],
        actual_gpu_up: &[f32],
        actual_gpu_gated: &[f32],
    ) -> Result<crate::inference::Q4ExpertStageDiagnosticBundle, String> {
        use sha2::{Digest, Sha256};

        if self.core.options.dtype != WeightDtype::Q4_0
            || self.execution_context().plan().routed_experts()
                != crate::backend::ExecutionPlane::Cpu
            || self.core.options.policy != crate::inference::RealInferencePolicy::STRICT
        {
            return Err(
                "Q4 stage attribution requires a strict CPU Q4_0 routed-expert plan".to_string(),
            );
        }
        if gpu_effective_input.len() != self.core.shape.d_model
            || actual_gpu_gate.len() != self.core.shape.d_ff
            || actual_gpu_up.len() != self.core.shape.d_ff
            || actual_gpu_gated.len() != self.core.shape.d_ff
            || gpu_effective_input
                .iter()
                .chain(actual_gpu_gate)
                .chain(actual_gpu_up)
                .chain(actual_gpu_gated)
                .any(|value| !value.is_finite())
        {
            return Err("Q4 stage attribution received nonfinite or malformed vectors".to_string());
        }

        self.core
            .storage
            .validate_expert_file_layout(std::iter::once(global_expert_id))
            .map_err(|error| {
                format!(
                    "global expert {global_expert_id} does not have the exact configured checkpoint file size: {error}"
                )
            })?;
        let expected_payload = crate::inference::expert_weight_bytes_for(
            self.core.shape.d_model,
            self.core.shape.d_ff,
            WeightDtype::Q4_0,
        );
        let block_align = self.core.storage.config().block_align;
        let checkpoint_payload_bytes =
            crate::q4_parity::aligned_checkpoint_payload_bytes(expected_payload, block_align)?;
        let resident = self
            .fetch_with_retry(global_expert_id)
            .await
            .map_err(|error| {
                format!("failed to fetch global expert {global_expert_id}: {error}")
            })?;
        if resident.data().len() != checkpoint_payload_bytes {
            return Err(format!(
                "global expert {global_expert_id} checkpoint payload has {} bytes, expected {checkpoint_payload_bytes}",
                resident.data().len()
            ));
        }
        let (payload, padding) = resident.data().split_at(expected_payload);
        if padding.iter().any(|byte| *byte != 0) {
            return Err(format!(
                "global expert {global_expert_id} has non-zero checkpoint alignment padding"
            ));
        }
        let weights = crate::inference::OwnedExpertWeights::from_bytes_q4_0_with_tolerance(
            payload,
            self.core.shape.d_model,
            self.core.shape.d_ff,
            0,
        )
        .map_err(|error| format!("CPU Q4 stage weights failed: {error}"))?;
        let cpu_boundary_input =
            crate::numerical_diagnostics::round_trip_f16_values(gpu_effective_input)
                .map_err(|error| format!("CPU input boundary emulation failed: {error}"))?;
        let cpu_production = weights
            .diagnostic_forward_stage_trace(&cpu_boundary_input)
            .map_err(|error| format!("CPU production stage trace failed: {error}"))?;
        let ordinary_cpu = crate::inference::q4_0_cpu_reference_forward(
            payload,
            &cpu_boundary_input,
            self.core.shape.d_model,
            self.core.shape.d_ff,
        )
        .map_err(|error| format!("ordinary CPU production expert failed: {error}"))?;
        if ordinary_cpu
            .iter()
            .map(|value| value.to_bits())
            .ne(cpu_production.down.iter().map(|value| value.to_bits()))
        {
            return Err(
                "CPU diagnostic stage final output differs from ordinary production output"
                    .to_string(),
            );
        }
        let cpu_effective_output =
            crate::numerical_diagnostics::round_trip_f16_values(&cpu_production.down)
                .map_err(|error| format!("CPU output boundary emulation failed: {error}"))?;
        let current_gpu_emulation = crate::inference::diagnostic_q4_0_expert_arithmetic(
            payload,
            gpu_effective_input,
            self.core.shape.d_model,
            self.core.shape.d_ff,
            crate::inference::DiagnosticQ4DotArithmetic::CurrentGpu,
        )
        .map_err(|error| format!("current-GPU arithmetic emulation failed: {error}"))?;
        let rejected_logical_dequant_emulation =
            crate::inference::diagnostic_q4_0_expert_arithmetic(
                payload,
                gpu_effective_input,
                self.core.shape.d_model,
                self.core.shape.d_ff,
                crate::inference::DiagnosticQ4DotArithmetic::RejectedLogicalDequant,
            )
            .map_err(|error| format!("rejected logical-dequant emulation failed: {error}"))?;
        let gpu_gate_up_through_cpu_swiglu_down = weights
            .diagnostic_replay_gate_up(actual_gpu_gate, actual_gpu_up)
            .map_err(|error| format!("GPU gate/up CPU replay failed: {error}"))?;
        let gpu_gated_through_cpu_down = weights
            .diagnostic_replay_down(actual_gpu_gated)
            .map_err(|error| format!("GPU gated CPU-down replay failed: {error}"))?;
        let cpu_gated_through_current_gpu_down = crate::inference::diagnostic_q4_0_down_arithmetic(
            payload,
            &cpu_production.gated,
            self.core.shape.d_model,
            self.core.shape.d_ff,
            crate::inference::DiagnosticQ4DotArithmetic::CurrentGpu,
        )
        .map_err(|error| format!("CPU gated current-GPU down replay failed: {error}"))?;
        let gpu_gated_through_current_gpu_down = crate::inference::diagnostic_q4_0_down_arithmetic(
            payload,
            actual_gpu_gated,
            self.core.shape.d_model,
            self.core.shape.d_ff,
            crate::inference::DiagnosticQ4DotArithmetic::CurrentGpu,
        )
        .map_err(|error| format!("GPU gated current-GPU down replay failed: {error}"))?;

        Ok(crate::inference::Q4ExpertStageDiagnosticBundle {
            canonical_payload_sha256: format!("{:x}", Sha256::digest(payload)),
            cpu_boundary_input,
            cpu_production,
            cpu_effective_output,
            current_gpu_emulation,
            rejected_logical_dequant_emulation,
            gpu_gate_up_through_cpu_swiglu_down,
            gpu_gated_through_cpu_down,
            cpu_gated_through_current_gpu_down,
            gpu_gated_through_current_gpu_down,
        })
    }

    /// Snapshot the qualification-only real routed-expert execution counters.
    pub fn routed_expert_execution_snapshot(&self) -> RoutedExpertExecutionSnapshot {
        RoutedExpertExecutionSnapshot {
            selected_routed_experts: self
                .metrics
                .counters
                .selected_routed_experts
                .load(Ordering::Relaxed),
            gpu_dispatch_attempts: self
                .metrics
                .counters
                .gpu_dispatch_attempts
                .load(Ordering::Relaxed),
            gpu_dispatch_successes: self
                .metrics
                .counters
                .gpu_dispatch_successes
                .load(Ordering::Relaxed),
            gpu_dispatch_failures: self
                .metrics
                .counters
                .gpu_dispatch_failures
                .load(Ordering::Relaxed),
            cpu_routed_expert_dispatches: self
                .metrics
                .counters
                .cpu_routed_expert_dispatches
                .load(Ordering::Relaxed),
            gpu_cpu_fallbacks: self
                .metrics
                .counters
                .gpu_cpu_fallbacks
                .load(Ordering::Relaxed),
            degraded_expert_substitutions: self
                .metrics
                .counters
                .degraded_expert_substitutions
                .load(Ordering::Relaxed),
        }
    }

    /// Synchronously copy a freshly-loaded RAM resident into the logical GPU
    /// admission cache, if installed and GPU-compatible. Device upload remains
    /// lazy and authoritative in `GpuBackend`'s physical registry.
    ///
    /// This is the warm-up counterpart to the background promotion
    /// task wired up in [`Engine::install_gpu_cache`]: the background
    /// task only fires after an expert crosses the RAM-hit promotion
    /// threshold. Calling this after an NVMe load makes the payload eligible
    /// for a lazy physical upload on the next routed GPU dispatch.
    ///
    /// The byte handling mirrors the background task exactly: both
    /// F32 and Q4_0 experts are promoted byte-for-byte — Q4_0 bytes
    /// stay in native GGUF blocks (~8× smaller than F32) and are
    /// dequantised *inline on the GPU* by the `matmul_q4_0.wgsl`
    /// pipeline; the resident is dtype-tagged so the backend picks
    /// the right pipeline. The synchronous path
    /// uses [`GpuExpertCache::try_promote_lru_no_evict`] so it never
    /// evicts already-resident hot experts and never consumes Anchor
    /// Core slots — anchor promotion stays the exclusive job of the
    /// threshold-driven background task in
    /// [`Engine::install_gpu_cache`]. When the LRU Edge is already
    /// full, this call is a no-op and the expert stays on the CPU
    /// path until the background task anchors it.
    fn try_promote_resident_to_gpu(&self, resident: &Arc<ExpertResident>) {
        // Only meaningful when a GPU backend is live and the dtype is
        // one the GPU kernels can actually consume; otherwise the
        // promotion would waste logical admission budget on bytes the fast
        // path can never use.
        if !self.routed_expert_backend().is_gpu() || !self.gpu_eligible_dtype() {
            return;
        }
        let Some(gpu) = self.core.gpu_cache.as_ref() else {
            return;
        };
        let id = resident.id;
        // Already logically admitted: nothing to do (and the LRU helper
        // would short-circuit anyway). Skip the byte copy entirely.
        if gpu.contains(id) {
            return;
        }
        // Bytes are promoted verbatim and dtype-tagged: Q4_0 experts
        // stay in native GGUF blocks (~8× fewer bytes across PCIe and
        // as a physical buffer than a dequantised F32 stream) and are unpacked
        // inline by the GPU's `matmul_q4_0.wgsl` pipeline.
        let gpu_res = Arc::new(GpuResident::new_with_dtype(
            id,
            resident.data().to_vec(),
            self.core.options.dtype,
        ));
        if gpu.try_promote_lru_no_evict(gpu_res) {
            if let Some(p) = self.metrics.prom.as_ref() {
                p.record_promotions(1);
                p.set_vram_used_bytes(gpu.used_bytes() as u64);
            }
        }
    }

    /// Establish the current selected expert's logical GPU admission at the
    /// real routed-compute boundary. Routing remains authoritative for normal
    /// logical hit/miss telemetry and LRU touches; this demand path only fills
    /// a missing admission and never uses Anchor Core. Physical upload remains
    /// lazy in `GpuBackend`.
    fn demand_admit_resident_to_gpu(
        &self,
        layer: u32,
        resident: &ExpertResident,
    ) -> Result<(), crate::backend::GpuExpertDispatchError> {
        use crate::expert_cache::GpuDemandAdmissionPreflight;

        let gpu = self.execution_context().gpu_expert_cache();
        let admission_error = |error| {
            crate::backend::GpuExpertDispatchError::new(
                layer,
                resident.id,
                crate::backend::GpuExpertDispatchErrorKind::ResidencyMiss,
                format!("foreground demand admission failed: {error}"),
            )
        };
        match gpu.demand_admission_preflight(resident.id, resident.data().len()) {
            Ok(GpuDemandAdmissionPreflight::AlreadyAdmitted) => return Ok(()),
            Ok(GpuDemandAdmissionPreflight::NeedsPayload) => {}
            Err(error) => return Err(admission_error(error)),
        }

        // The preflight eliminates copies for admissions already visible at
        // the probe and for deterministically oversized payloads, then drops
        // the cache lock before this memcpy. `demand_admit_lru` rechecks under
        // its own lock for identity/accounting correctness. A concurrent
        // admission between the probe and install may therefore cause one
        // redundant host copy followed by `Ok(false)`; avoiding that bounded
        // race would require reservation/singleflight machinery and is
        // intentionally deferred unless profiling demonstrates a need.
        let gpu_resident = Arc::new(GpuResident::new_with_dtype(
            resident.id,
            resident.data().to_vec(),
            self.core.options.dtype,
        ));
        match gpu.demand_admit_lru(gpu_resident) {
            Ok(newly_admitted) => {
                if newly_admitted {
                    if let Some(prom) = self.metrics.prom.as_ref() {
                        prom.record_promotions(1);
                        prom.set_vram_used_bytes(gpu.used_bytes());
                    }
                }
                Ok(())
            }
            Err(error) => Err(admission_error(error)),
        }
    }

    /// Process one threshold-driven promotion request. Existing logical
    /// admissions are resolved first so an LRU entry can move to Anchor Core
    /// without copying its multi-megabyte RAM payload. Only an absent id
    /// requires a payload copy, followed by a locked recheck that preserves a
    /// foreground demand admission if it won the race.
    fn complete_background_gpu_promotion(
        gpu: &GpuExpertCache,
        prom: Option<&Metrics>,
        id: u32,
        resident: &ExpertResident,
        dtype: WeightDtype,
    ) -> GpuHotPromotionOutcome {
        let initial = gpu.promote_hot_existing(id);
        let outcome = if initial == GpuHotPromotionOutcome::PayloadRequired {
            // The cache lock was released by `promote_hot_existing` before
            // this copy. `promote_hot_sync` rechecks under its own lock, so a
            // concurrent demand admission is moved intact rather than
            // replaced by this redundant payload.
            let gpu_resident = Arc::new(GpuResident::new_with_dtype(
                id,
                resident.data().to_vec(),
                dtype,
            ));
            gpu.promote_hot_sync(gpu_resident)
        } else {
            initial
        };

        if outcome.is_transition() {
            if let Some(prom) = prom {
                prom.record_promotions(1);
                prom.set_vram_used_bytes(gpu.used_bytes());
            }
        }
        outcome
    }

    /// Apply the existing non-evicting speculative logical-admission policy
    /// and return its current generation. The historical `GpuResident`
    /// payload copy remains compatibility debt; physical upload still reads
    /// directly from `ExpertResident::data()` and this helper adds no further
    /// persistent host copy.
    fn ensure_speculative_gpu_admission(
        &self,
        resident: &Arc<ExpertResident>,
    ) -> Option<crate::expert_cache::GpuAdmission> {
        let gpu = self.core.execution_context.gpu_expert_cache();
        if let Some(admission) = gpu.current_admission(resident.id) {
            return Some(admission);
        }
        let bytes = resident.data().len();
        if (gpu.used_bytes() as usize).saturating_add(bytes) > gpu.capacity_bytes() {
            return None;
        }
        let gpu_resident = Arc::new(GpuResident::new_with_dtype(
            resident.id,
            resident.data().to_vec(),
            self.core.options.dtype,
        ));
        if gpu.try_promote_lru_no_evict(gpu_resident) {
            if let Some(prom) = self.metrics.prom.as_ref() {
                prom.record_promotions(1);
                prom.set_vram_used_bytes(gpu.used_bytes());
            }
        }
        gpu.current_admission(resident.id)
    }

    /// Explicit future bootstrap seam for the GPU-native physical plane.
    ///
    /// The manager must retain the exact logical cache owned by this engine's
    /// authoritative execution context. Slice 9 also rejects simultaneous
    /// activation of the legacy routed-expert GPU registry: Slice 10 may
    /// compose the GPU-native token loop, but it must never double-consume one
    /// request's configured expert capacity across both physical planes.
    pub(crate) fn install_gpu_native_residency_manager(
        &mut self,
        manager: Arc<GpuNativeTieredResidencyManager>,
    ) -> Result<(), GpuNativeResidencyInstallError> {
        if self.core.gpu_native_residency.is_some() {
            return Err(GpuNativeResidencyInstallError::AlreadyInstalled);
        }
        if self.core.execution_context.plan().routed_experts()
            == crate::backend::ExecutionPlane::Gpu
        {
            return Err(GpuNativeResidencyInstallError::LegacyPhysicalExpertPlaneActive);
        }
        if !Arc::ptr_eq(
            manager.gpu_cache(),
            self.core.execution_context.gpu_expert_cache(),
        ) {
            return Err(GpuNativeResidencyInstallError::LogicalCacheMismatch);
        }
        self.core.gpu_cache = Some(manager.gpu_cache().clone());
        self.core.gpu_native_residency = Some(manager);
        Ok(())
    }

    pub(crate) fn gpu_native_residency_snapshot(&self) -> Option<GpuNativeTieredResidencySnapshot> {
        self.core
            .gpu_native_residency
            .as_ref()
            .map(|manager| manager.snapshot())
    }

    /// Install v2 evidence. Control explicitly forces the legacy sequential
    /// helper; treatment exercises the same ordinary production path used
    /// when no qualifier is active. The dedicated command calls this only on a
    /// fresh isolated runtime; normal construction leaves the slot `None`.
    pub(crate) fn enable_gpu_native_demand_source_production_qualification(
        &self,
        arm: GpuNativeDemandSourceQualificationArm,
    ) -> Result<(), String> {
        if !self.core.in_flight.is_empty() || self.core.cache.reserved_slots() != 0 {
            return Err(
                "cannot enable exact-demand source qualification with active singleflight entries or cache reservations"
                    .into(),
            );
        }
        let mut slot = self.gpu_native_demand_source_qualification.write();
        if slot.is_some() {
            return Err("exact-demand source qualification is already enabled".into());
        }
        *slot = Some(Arc::new(GpuNativeDemandSourceQualification::new(
            arm,
            self.core.pool.capacity(),
            self.core.pool.shadow_capacity(),
        )));
        self.production_demand_source.reset();
        Ok(())
    }

    /// Reset qualification-only counters between warmup and measurement.
    /// Cache, pool, residency, router, and all production state are retained.
    pub(crate) fn reset_gpu_native_demand_source_qualification(&self) -> Result<(), String> {
        if !self.core.in_flight.is_empty() || self.core.cache.reserved_slots() != 0 {
            return Err(
                "cannot reset exact-demand source qualification with active singleflight entries or cache reservations"
                    .into(),
            );
        }
        let mut slot = self.gpu_native_demand_source_qualification.write();
        let current = slot
            .as_ref()
            .ok_or("exact-demand source qualification is not enabled")?;
        if current.active_demand_set.load(Ordering::Acquire) {
            return Err("cannot reset exact-demand source qualification during demand service".into());
        }
        let arm = current.arm;
        let primary_pool_capacity = current.primary_pool_capacity;
        let shadow_pool_capacity = current.shadow_pool_capacity;
        *slot = Some(Arc::new(GpuNativeDemandSourceQualification::new(
            arm,
            primary_pool_capacity,
            shadow_pool_capacity,
        )));
        self.production_demand_source.reset();
        Ok(())
    }

    pub(crate) fn gpu_native_demand_source_qualification_snapshot(
        &self,
    ) -> Option<GpuNativeDemandSourceQualificationSnapshot> {
        self.gpu_native_demand_source_qualification
            .read()
            .as_ref()
            .map(|state| state.snapshot())
    }

    pub(crate) fn production_demand_source_snapshot(&self) -> ProductionDemandSourceSnapshot {
        self.production_demand_source
            .snapshot(self.core.cache.reserved_slots(), self.core.in_flight.len())
    }

    pub(crate) fn gpu_native_demand_source_qualification_ram_cache_state_sha256(
        &self,
    ) -> Option<String> {
        self.gpu_native_demand_source_qualification
            .read()
            .as_ref()
            .map(|_| self.core.cache.qualification_state_sha256())
    }

    fn gpu_native_demand_source_qualification(
        &self,
    ) -> Option<Arc<GpuNativeDemandSourceQualification>> {
        self.gpu_native_demand_source_qualification
            .read()
            .as_ref()
            .cloned()
    }

    /// Record actual GPU-native route selections across layers into the engine's
    /// route observation and prefetch infrastructure without requiring CPU hidden state.
    pub(crate) fn record_gpu_native_actual_routes(
        self: &Arc<Self>,
        selected_ids_by_layer: &[Vec<u32>],
    ) {
        self.core.governor.refresh();
        let per_layer_opt = self.core.storage.config().num_experts_per_layer;

        if let Some(qualification) = self.gpu_native_demand_source_qualification() {
            let per_layer = per_layer_opt.unwrap_or(0);
            let mut hasher = qualification.selected_route_ids.lock();
            for (layer_idx, local_ids) in selected_ids_by_layer.iter().enumerate() {
                let global_ids = if per_layer > 0 {
                    local_ids
                        .iter()
                        .map(|&local_id| layer_idx as u32 * per_layer + local_id)
                        .collect::<Vec<_>>()
                } else {
                    local_ids.clone()
                };
                hasher.record_set(&global_ids);
            }
        }

        for (layer_idx, local_ids) in selected_ids_by_layer.iter().enumerate() {
            if local_ids.is_empty() {
                continue;
            }
            let per_layer = per_layer_opt.unwrap_or(0);
            let target: Vec<u32> = if per_layer > 0 {
                local_ids
                    .iter()
                    .map(|&loc| self.resolve_alias(layer_idx as u32 * per_layer + loc))
                    .collect()
            } else {
                local_ids.iter().map(|&id| self.resolve_alias(id)).collect()
            };

            self.metrics
                .counters
                .selected_routed_experts
                .fetch_add(target.len() as u64, Ordering::Relaxed);

            self.locality_observe_and_reconcile(&target);

            if let Some(affinity) = self.speculation.affinity.as_ref() {
                affinity.observe_layer(layer_idx, local_ids);
            }

            if self.core.options.pin_after_observations > 0
                || self.core.options.collect_route_profile
                || self.static_residency_needs_counts()
            {
                self.bump_route_observations(&target);
            }
            self.maybe_apply_static_residency();

            if let Some(&seed) = target.last() {
                let ring = self.speculation.markov_ring.lock();
                let contiguous = self.markov_layers_contiguous(ring.last.layer, Some(layer_idx as u32));
                let s_markov = match ring.last.ids.last() {
                    Some(&pp) if contiguous => self.core.predictor.predict_next2(pp, seed),
                    _ => self.core.predictor.predict_next(seed),
                };
                drop(ring);
                let in_flight: HashSet<u32> = target.iter().copied().collect();
                self.union_prefetch(&s_markov, &[], &in_flight, Some(layer_idx as u32));
            }

            if !target.is_empty() {
                let mut ring = self.speculation.markov_ring.lock();
                if !ring.last.ids.is_empty()
                    && self.markov_layers_contiguous(ring.last.layer, Some(layer_idx as u32))
                {
                    let pp: &[u32] =
                        if self.markov_layers_contiguous(ring.last_last.layer, ring.last.layer) {
                            &ring.last_last.ids
                        } else {
                            &[]
                        };
                    self.core
                        .predictor
                        .observe_step2(pp, &ring.last.ids, &target);
                }
                ring.last_last = ring.last.clone();
                ring.last = MarkovHistory {
                    ids: target.clone(),
                    layer: Some(layer_idx as u32),
                };
            }
        }
    }

    /// Attach the logical GPU-admission cache — Phase 2 hierarchy policy.
    ///
    /// Spawns a background Tokio task that drains an MPSC channel of
    /// `(expert_id, ram_resident)` promotion requests fed by the
    /// inference hot path. The hot path itself never blocks on the
    /// promotion — it `send`s and moves on — so installing this cache
    /// has no impact on per-token latency. When the channel is
    /// disconnected (engine drop) the background task exits.
    ///
    /// Updates the compatibility `mer_vram_used_bytes` gauge with logical
    /// admitted host payload bytes after every successful promotion.
    pub fn install_gpu_cache(&mut self) {
        let gpu = self.core.execution_context.gpu_expert_cache().clone();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(u32, Arc<ExpertResident>)>();
        let gpu_for_task = gpu.clone();
        let prom_for_task = self.metrics.prom.clone();
        // Snapshot the (immutable) expert dtype so each promoted
        // resident is dtype-tagged. Q4_0 logical admissions retain raw GGUF
        // blocks; the backend later uploads and unpacks them inline via
        // `matmul_q4_0.wgsl`. The F32 path remains byte-for-byte unchanged.
        let promote_dtype = self.core.options.dtype;
        // Capacity is constant for the lifetime of the cache; publish
        // it once so `mer_vram_capacity_bytes` is available on the
        // very first `/metrics` scrape (dashboards compute
        // utilisation as `mer_vram_used_bytes / mer_vram_capacity_bytes`).
        if let Some(p) = prom_for_task.as_ref() {
            p.set_vram_capacity_bytes(gpu.capacity_bytes() as u64);
        }
        let spawned = self.background_tasks.spawn(async move {
            while let Some((id, resident)) = rx.recv().await {
                // Existing LRU admissions graduate atomically without a host
                // payload copy. Absent ids copy outside the cache mutex and
                // are rechecked under lock before final hot admission.
                Self::complete_background_gpu_promotion(
                    gpu_for_task.as_ref(),
                    prom_for_task.as_ref(),
                    id,
                    resident.as_ref(),
                    promote_dtype,
                );
            }
        });
        debug_assert!(
            spawned,
            "a newly built engine must accept GPU promotion work"
        );
        self.core.gpu_cache = Some(gpu.clone());
        self.core.gpu_promotion_tx = Some(tx);
    }

    /// Stop admission and deterministically retire every anonymous async task
    /// owned by this engine. Isolated qualification runtimes call this before
    /// dropping their strong engine handle; ordinary steady-state serving is
    /// unchanged until a caller explicitly requests shutdown.
    pub(crate) async fn shutdown_background_tasks(&self) -> Result<(), String> {
        self.background_tasks.shutdown().await
    }

    /// **Test-only** wiring of the GPU promotion channel without
    /// spawning the background consumer task. Used by the regression
    /// test in [`tests`] (gist Task 1, "GPU Promotion Regression
    /// Test") to inspect the `gpu_promotion_tx` MPSC sender side
    /// directly and assert that after `promote_after_hits` RAM hits
    /// on the same expert *exactly one* promotion message is emitted.
    ///
    /// Returning the `UnboundedReceiver` to the test lets the
    /// assertion be performed on the raw mpsc traffic — i.e. before
    /// `promote_sync` consumes it — which is the contract the gist
    /// asks for ("verify the message count directly, do not use the
    /// report API").
    #[cfg(test)]
    pub(crate) fn install_gpu_cache_for_test(
        &mut self,
        gpu: Arc<GpuExpertCache>,
    ) -> tokio::sync::mpsc::UnboundedReceiver<(u32, Arc<ExpertResident>)> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<(u32, Arc<ExpertResident>)>();
        self.core.gpu_cache = Some(gpu);
        self.core.gpu_promotion_tx = Some(tx);
        rx
    }

    /// Install a JSONL routing trace sink. Every subsequent
    /// `generate` call appends `{token, layer, experts, cache_hit}` to
    /// the underlying file. Passing `None` disables tracing.
    pub fn set_trace_writer(&self, writer: Option<Arc<TraceWriter>>) {
        *self.metrics.trace_writer.write() = writer;
    }

    /// Install an alias map. Calls to [`Self::generate`] / prefetch will
    /// remap ids through it before consulting the cache, so multiple
    /// near-identical experts share a single resident copy.
    pub fn with_alias_map(mut self, map: HashMap<u32, u32>) -> Self {
        // Keep only entries that actually move ids. Self-aliases are noise.
        let cleaned: HashMap<u32, u32> = map.into_iter().filter(|(k, v)| k != v).collect();
        self.speculation.alias_map = if cleaned.is_empty() {
            None
        } else {
            Some(Arc::new(cleaned))
        };
        self
    }

    /// Install a sliding-window [`LocalityMonitor`]. The engine will
    /// observe every routed expert and, after each `generate` /
    /// `moe_step`, reconcile the monitor's hot set with the cache's pin
    /// state — newly hot ids are pinned, ids that fell below the heat
    /// threshold are unpinned.
    pub fn with_locality_monitor(
        mut self,
        monitor: Arc<LocalityMonitor>,
        threshold_pct: f32,
    ) -> Self {
        self.speculation.locality = Some(monitor);
        // Clamp into a sane range; values outside `[0,1]` make no
        // semantic sense for a "fraction of the window" threshold.
        self.speculation.locality_threshold_pct = threshold_pct.clamp(0.0, 1.0);
        self
    }

    /// Install a [`NeuralSpeculator`]. When set, the engine will (a)
    /// query the speculator for its top-K prediction at every routed
    /// hidden state, (b) compare it against the actual gate decision
    /// to update speculator-accuracy telemetry, (c) feed that decision
    /// back into a single online SGD step, and (d) union the
    /// speculator's prediction with the predictor's Markov chain hint
    /// when issuing speculative prefetches.
    pub fn with_speculator(mut self, spec: Arc<NeuralSpeculator>, top_k: usize) -> Self {
        // Spawn the off-path training worker (idempotent: no-op if
        // already running). Background SGD updates then flow through
        // `NeuralSpeculator::queue_train` without blocking the
        // engine's per-token critical path.
        spec.spawn_training_worker();
        self.speculation.speculator = Some(spec);
        self.speculation.speculator_topk = top_k.max(1);
        self
    }

    /// Set the look-ahead **pipeline depth** — how many MoE layers ahead
    /// [`Self::speculate_layer_ahead`] prefetches (the sliding window
    /// `current_layer + 1 ..= current_layer + depth`). Sized to roughly
    /// `ceil(io_latency / compute_latency)` so the SSD reads for the next
    /// several layers complete behind the current layer's compute; see
    /// [`DEFAULT_PIPELINE_DEPTH`]. Clamped to at least `1` (a value of `1`
    /// reproduces the legacy single-layer look-ahead).
    pub fn with_pipeline_depth(mut self, depth: u32) -> Self {
        self.speculation.pipeline_depth = depth.max(1);
        self
    }

    /// Install a per-layer [`LayeredExpertAffinity`] co-occurrence
    /// matrix — the **affinity** arm. When set, `moe_step` records each
    /// layer's routed set into the matrix and `union_prefetch` folds the
    /// top-`neighbors_k` co-fired neighbours (plus UTH disk-adjacent
    /// neighbours) of every high-confidence prediction into the
    /// speculative prefetch union. A background exponential-decay worker
    /// is spawned to age the counters every `decay_epoch` cumulative
    /// observations; its handle is retained for the engine's lifetime
    /// (dropping the engine stops the worker).
    pub fn with_affinity(
        mut self,
        affinity: Arc<LayeredExpertAffinity>,
        neighbors_k: usize,
        decay_epoch: u64,
    ) -> Self {
        // `bits = 1` halves every counter per epoch; the 250 ms poll is
        // the upper bound on how long a saturated counter lingers before
        // the next shift. Both mirror the defaults documented on
        // `LayeredExpertAffinity::spawn_decay_worker`.
        let handle = affinity.clone().spawn_decay_worker(
            decay_epoch.max(1),
            1,
            std::time::Duration::from_millis(250),
        );
        self.speculation.affinity = Some(affinity);
        self.speculation.affinity_neighbors_k = neighbors_k.max(1);
        self.speculation.affinity_decay = Some(handle);
        self
    }

    /// Tier 1 — install the **static residency** controller. `fraction`
    /// is the share of the global expert namespace to pin permanently
    /// (`<= 0.0` disables the feature and leaves the engine unchanged).
    /// When `profile` is `Some`, its hot set is pinned at the first
    /// token with no warmup; when `None`, the engine derives the hot set
    /// online from [`Self::route_observations`] after `warmup_tokens`
    /// tokens. The pin budget is sized against the router's expert
    /// namespace so it is a true fraction of *all* experts.
    pub fn with_static_residency(
        mut self,
        fraction: f64,
        warmup_tokens: u64,
        profile: Option<crate::residency::ResidencyProfile>,
    ) -> Self {
        if fraction <= 0.0 {
            return self;
        }
        let namespace = self.core.router.num_experts() as usize;
        self.speculation.static_residency = Some(crate::residency::StaticResidencyState::new(
            fraction.min(1.0),
            warmup_tokens,
            namespace,
            profile,
        ));
        self
    }

    /// Tier 3 — install the **per-layer pre-gate** predictor. Once set,
    /// every `moe_step` records the layer-to-layer routing transition
    /// and prefetches the predicted next-layer experts. `top_n` bounds
    /// how many next-layer experts are prefetched per step.
    pub fn with_pregate(mut self, pregate: Arc<crate::pregate::PerLayerPreGate>) -> Self {
        self.speculation.pregate = Some(pregate);
        self
    }

    /// Whether the static-residency controller still needs
    /// `route_observations` populated this token — true only while an
    /// *online* (profile-less) controller is configured and has not yet
    /// pinned its hot set. Keeps the per-token observation bump free
    /// when the feature is off or already applied.
    fn static_residency_needs_counts(&self) -> bool {
        self.speculation
            .static_residency
            .as_ref()
            .map(|sr| sr.is_online() && !sr.applied.load(Ordering::Acquire))
            .unwrap_or(false)
    }

    /// Build a [`ResidencyProfile`] snapshot from the live
    /// `route_observations` counters. Cheap relaxed reads — the counts
    /// are advisory, so a torn-but-consistent snapshot is acceptable.
    fn snapshot_route_profile(&self) -> crate::residency::ResidencyProfile {
        let mut counts = HashMap::new();
        for entry in self.speculation.route_observations.iter() {
            counts.insert(*entry.key(), entry.value().load(Ordering::Relaxed));
        }
        crate::residency::ResidencyProfile::from_counts(counts)
    }

    /// Dump the engine's live route-observation profile to `path` as
    /// JSON (`{ "<id>": <count> }`). Used by the `--profile-out` flag so
    /// a run can emit its own hot-set profile for a later
    /// `--static-residency-profile` warm start.
    pub fn dump_route_profile(&self, path: &std::path::Path) -> std::io::Result<()> {
        self.snapshot_route_profile().dump_json(path)
    }

    /// Apply the static-residency hot set exactly once, when ready. For
    /// a profile-seeded controller this fires on the first token; for an
    /// online controller it waits until the engine has bumped route
    /// observations for `warmup_tokens` tokens, then derives the hot set
    /// from the accumulated route observations.
    /// The pin is latched behind a compare-and-swap so concurrent
    /// per-token calls apply it a single time.
    fn maybe_apply_static_residency(&self) {
        let Some(sr) = self.speculation.static_residency.as_ref() else {
            return;
        };
        if sr.applied.load(Ordering::Acquire) {
            return;
        }
        let hot = if let Some(profile) = sr.profile.as_ref() {
            profile.hot_set(sr.fraction, sr.namespace)
        } else {
            if self
                .speculation
                .route_observation_tokens
                .load(Ordering::Acquire)
                < sr.warmup_tokens
            {
                return;
            }
            self.snapshot_route_profile()
                .hot_set(sr.fraction, sr.namespace)
        };
        if hot.is_empty() {
            return;
        }
        // Latch: exactly one caller transitions false -> true and pins.
        if sr
            .applied
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        for id in &hot {
            self.core.cache.pin(*id);
        }
        {
            let mut static_pinned = self.speculation.static_pinned.lock();
            static_pinned.extend(hot.iter().copied());
        }
        info!(
            pinned = hot.len(),
            fraction = sr.fraction,
            source = if sr.profile.is_some() {
                "profile"
            } else {
                "online"
            },
            warmup_tokens = sr.warmup_tokens,
            "static residency: pinned hot expert set"
        );
    }

    /// Wire a Prometheus metrics sink. The engine will mirror its
    /// telemetry counters (locality / speculator hits & misses, SSD
    /// stall) into the metrics registry alongside its own atomics.
    pub fn with_metrics(mut self, m: Metrics) -> Self {
        self.metrics.prom = Some(m);
        self
    }

    /// Resolve an id through the alias map (if any), bumping the
    /// redirect counter on a hit. Pure function on `&self`; safe to
    /// call from any context.
    fn resolve_alias(&self, id: u32) -> u32 {
        if let Some(m) = &self.speculation.alias_map {
            if let Some(&canon) = m.get(&id) {
                if canon != id {
                    self.speculation
                        .alias_redirects
                        .fetch_add(1, Ordering::Relaxed);
                    return canon;
                }
            }
        }
        id
    }

    #[inline]
    fn credit_prefetch_use(&self, resident: &Arc<ExpertResident>, new_hits: u64) {
        if new_hits == 1 && resident.is_shadow_backed() {
            self.metrics
                .counters
                .prefetch_used
                .fetch_add(1, Ordering::Relaxed);
            self.core.governor.record_used();
        }
    }

    /// Total number of distinct experts the engine's router can
    /// address. Exposed so warm-up / diagnostic paths can size
    /// their work to the global expert namespace without reaching
    /// into the router enum.
    pub fn num_experts(&self) -> u32 {
        self.core.router.num_experts()
    }

    /// Process a single token: route, fetch missing experts, run inference,
    /// update predictor, and kick off prefetches. Returns one [`CycleStats`].
    ///
    /// Returns `Err(ExpertReadError)` when a routed expert cannot be
    /// fetched even after retries (corrupt file, persistent I/O error,
    /// or a starved buffer pool). This path has no per-expert "skip"
    /// option (unlike [`Self::moe_step`], which drops a failed expert
    /// from the top-K mixture), so the only safe degradation is to
    /// surface the error to the caller — the HTTP serving path maps it
    /// to a 500 instead of crashing the process.
    pub async fn generate(self: &Arc<Self>, token_idx: u64) -> Result<CycleStats, ExpertReadError> {
        let cycle_start = Instant::now();
        // Tier 4: fold the previous token's prefetch precision window
        // into the governor's EWMA before this token issues any new
        // speculative reads. No-op when the governor is disabled.
        self.core.governor.refresh();
        // Compute the residual-stream hidden state up front. The
        // production `Router::Linear` path needs it to compute the
        // gate's softmax logits; the legacy `Router::Markov` path
        // ignores it. Either way the value is re-used by the FFN
        // forward pass below, so this is at worst the same single
        // `synth_hidden_state` call the legacy path always made.
        let hidden: HiddenState = synth_hidden_state(
            token_idx,
            self.core.shape.d_model,
            self.core.shape.hidden_seed,
        );
        let decision = self.core.router.route(&hidden, token_idx);
        let raw_target = decision.experts;
        // Resolve aliases up front so the cache + predictor only ever
        // see canonical expert ids. This is what makes deduplicated
        // experts share one resident copy.
        let target: Vec<u32> = raw_target
            .iter()
            .map(|&id| self.resolve_alias(id))
            .collect();
        let mut stats = CycleStats::default();

        // Locality monitor: observe the chosen experts and reconcile
        // pin state. When no monitor is configured this is a no-op
        // and we fall back to the legacy frequency-based pinning
        // below. (The two are intentionally orthogonal — frequency
        // pinning is monotonic and global; locality pinning is
        // sliding-window and topical.)
        self.locality_observe_and_reconcile(&target);

        // Frequency-based pinning: bump observation counts and ask the
        // cache to pin any id that crossed the threshold this token.
        // Also bump (without pinning) when an online static-residency
        // controller still needs route counts to derive its hot set, or
        // when the run is collecting a popularity profile for export.
        if self.core.options.pin_after_observations > 0
            || self.core.options.collect_route_profile
            || self.static_residency_needs_counts()
        {
            self.bump_route_observations(&target);
        }
        // Tier 1: pin the static-residency hot set once it is ready
        // (immediately for a profile seed, post-warmup for online).
        self.maybe_apply_static_residency();

        // 1) Make sure every required expert is resident.
        //
        // Cache-miss reads are issued concurrently. Two routed experts
        // that both miss kick off two `pread(2)` calls in parallel via
        // `tokio::spawn`, so the NVMe queue actually sees the queue depth
        // the routing decision implies; sequentially `await`-ing each
        // fetch would serialise an opportunity the device can already
        // satisfy concurrently. Hits are resolved inline.
        let io_wait_start = Instant::now();
        let mut residents: Vec<Option<Arc<ExpertResident>>> = vec![None; target.len()];
        let mut cache_hits_per_expert: Vec<bool> = vec![false; target.len()];
        let mut miss_handles: Vec<(
            usize,
            tokio::task::JoinHandle<Result<Arc<ExpertResident>, ExpertReadError>>,
        )> = Vec::new();
        // Logical GPU-admission tier — aggregate hits/misses across this routing
        // decision and record once, rather than incrementing Prometheus
        // counters per activation on the hot path.
        let mut gpu_hits_acc: u64 = 0;
        let mut gpu_misses_acc: u64 = 0;
        for (i, &id) in target.iter().enumerate() {
            // Logical GPU-admission tier. The cache
            // shadows RAM; on hit we still resolve the authoritative
            // `ExpertResident` from RAM below, but the counter reflects
            // the promotion-policy decision.
            if let Some(gpu) = self.core.gpu_cache.as_ref() {
                let lookup = gpu.get(id);
                if lookup.is_hit() {
                    gpu_hits_acc += 1;
                } else {
                    gpu_misses_acc += 1;
                }
            }
            if let Some(r) = self.core.cache.get(id) {
                self.metrics.counters.hits.fetch_add(1, Ordering::Relaxed);
                stats.hits += 1;
                debug!(expert = id, "cache hit");
                cache_hits_per_expert[i] = true;
                // RAM hit: bump the per-expert hit counter and, if we
                // have GPU admission configured, claim one fire-and-forget
                // promotion request after the threshold.
                let new_hits = r.record_hit();
                // Tier 4 precision feedback: the first hit on a
                // shadow-backed resident means a speculative prefetch
                // paid off (it was consumed before eviction). Credit the
                // governor so its precision EWMA tracks reality, and bump
                // the (previously dead) `prefetch_used` counter so the
                // run summary can report end-to-end prefetch precision.
                self.credit_prefetch_use(&r, new_hits);
                if self.background_tasks.accepts_work() {
                    if let (Some(gpu), Some(tx)) = (
                        self.core.gpu_cache.as_ref(),
                        self.core.gpu_promotion_tx.as_ref(),
                    ) {
                        if gpu.claim_promotion(id, new_hits)
                            && tx.send((id, r.clone())).is_err()
                        {
                            gpu.cancel_promotion(id);
                        }
                    }
                }
                residents[i] = Some(r);
            } else {
                self.metrics.counters.misses.fetch_add(1, Ordering::Relaxed);
                stats.misses += 1;
                debug!(expert = id, "cache miss, fetching from NVMe");
                let me = self.clone();
                miss_handles.push((i, tokio::spawn(async move { me.fetch(id).await })));
            }
        }
        // Aggregate logical GPU-admission outcome for this routing decision.
        if let Some(p) = self.metrics.prom.as_ref() {
            if gpu_hits_acc > 0 || gpu_misses_acc > 0 {
                p.record_gpu_cache(gpu_hits_acc, gpu_misses_acc);
            }
        }
        // Emit a trace record after we know which experts were chosen
        // and which were already resident. Layer is `0` for the
        // single-namespace flat router path; the multi-layer path
        // (`moe_step`) emits its own record with the caller-supplied
        // layer id. The `predicted` set is the neural speculator's
        // top-K guess for this token (empty when no speculator is
        // wired), logged so offline tooling can diff Predicted vs.
        // Actual without a second engine pass.
        if let Some(tw) = self.metrics.trace_writer.read().as_ref() {
            let predicted = self.trace_prediction(&hidden);
            tw.write_record(token_idx, 0, &target, &cache_hits_per_expert, &predicted);
        }

        // 2) Update the predictor with the observed transition and fire
        //    the speculative union prefetch (S ∪ L ∪ M) *now* — before
        //    awaiting the miss fetches and before the FFN compute — so
        //    the speculative reads overlap both the foreground SSD
        //    stall and this token's compute instead of running
        //    sequentially after them (this mirrors `moe_step`, which
        //    has always issued its union prefetch ahead of the
        //    miss-await).
        //
        //    Use the 2nd-order helper when we have a `prev_prev` set
        //    (anything from token_idx >= 2), so the predictor learns
        //    `(prev_prev -> prev -> next)` triples in addition to the
        //    `(prev -> next)` baseline.
        {
            let mut ring = self.speculation.markov_ring.lock();
            if !ring.last.ids.is_empty() {
                self.core
                    .predictor
                    .observe_step2(&ring.last_last.ids, &ring.last.ids, &target);
            }
            ring.last_last = ring.last.clone();
            ring.last = MarkovHistory {
                ids: target.clone(),
                layer: None,
            };
        }
        // Kick off speculative prefetches for the most-recent expert,
        // using the 2nd-order predictor when a prev_prev is available
        // (which gives sharper distributions than 1st-order alone and
        // therefore wastes less prefetch bandwidth). When a neural
        // speculator is configured, also union its top-K (the **M**
        // arm) and the locality monitor's hot set (the **L** arm)
        // into the prefetch set — see [`Engine::union_prefetch`].
        if let Some(&seed) = target.last() {
            let ring = self.speculation.markov_ring.lock();
            let s_markov = match ring.last_last.ids.last() {
                Some(&pp) => self.core.predictor.predict_next2(pp, seed),
                None => self.core.predictor.predict_next(seed),
            };
            drop(ring);
            // Speculator: predict + train on the residual-stream
            // hidden state computed at the top of `generate` (when
            // the speculator's d_model matches; otherwise this is
            // a no-op — see `speculator_predict_and_train`).
            let m_speculator = self.speculator_predict_and_train(&hidden, &target, None);
            // The gate's own targets are being fetched into primary
            // (Buffer A) buffers by the miss tasks spawned above —
            // pass them as `already_in_flight` so the union prefetch
            // doesn't re-fetch them into scarce shadow slots or steal
            // their singleflight leadership (same rationale as
            // `moe_step`). Synthetic single-layer benchmark path: no
            // layer-qualified id geometry, so no affinity fold.
            let in_flight: HashSet<u32> = target.iter().copied().collect();
            self.union_prefetch(&s_markov, &m_speculator, &in_flight, None);
        }

        let had_misses = !miss_handles.is_empty();
        for (i, h) in miss_handles {
            // `fetch` reports a fatal read error as `Err` (the engine
            // cannot make progress without the requested expert);
            // propagate it to the caller instead of crashing the
            // process. The outer `expect` only covers a *panicked*
            // fetch task (a bug, not an I/O failure), preserving the
            // pre-concurrency panic-propagation semantics for that case.
            let r = h.await.expect("expert fetch task panicked")?;
            let new_hits = r.record_hit();
            self.credit_prefetch_use(&r, new_hits);
            // We still account the per-call `stats.bytes_read` here
            // for the synthetic-benchmark accumulator (it tracks
            // logical bytes consumed, not bytes actually pulled
            // from disk), but the engine-wide `bytes_read` counter
            // is now bumped inside `fetch_once`, so we don't bump
            // it again — that would double-count every miss after
            // SSD-read dedup (gist Phase 1) was introduced.
            stats.bytes_read += r.buffer.len() as u64;
            // Synchronous logical admission after the NVMe load; the next GPU
            // routed dispatch performs authoritative physical lookup/upload.
            self.try_promote_resident_to_gpu(&r);
            residents[i] = Some(r);
        }
        let io_wait_us = if had_misses {
            io_wait_start.elapsed().as_micros() as u64
        } else {
            0
        };
        // The *SSD stall* is the slice of the critical path we were
        // actually blocked on reads. With concurrent miss fetches it's
        // bounded by `io_wait_us`; we report them as the same value
        // here, since a mock-storage benchmark has no separate "in
        // flight, but not blocking" component. The Prometheus sink
        // exports it as its own histogram so future overlapped-fetch
        // refactors can decouple the two without breaking dashboards.
        if io_wait_us > 0 {
            self.metrics
                .total_ssd_stall_us
                .fetch_add(io_wait_us, Ordering::Relaxed);
            if let Some(m) = &self.metrics.prom {
                m.record_ssd_stall(io_wait_us as f64 / 1_000_000.0);
            }
        }
        // Benchmark-path-only invariant (`Engine::generate`; the real
        // serving path is `moe_step`, which uses `try_fetch_with_skip`
        // and never reaches this): every slot was populated above by
        // either a cache hit or a successfully-joined miss fetch — a
        // failed fetch already returned `Err` before this point, so an
        // empty slot here is a control-flow bug, not an I/O failure.
        let residents: Vec<Arc<ExpertResident>> = residents
            .into_iter()
            .map(|r| r.expect("internal invariant (benchmark path): every routed expert slot must be populated by either a hit or a completed miss fetch"))
            .collect();

        // 3) Either run the real SwiGLU FFN, or — under `--io-only` —
        //    just touch every byte of the resident buffer with a cheap
        //    XOR checksum so the kernel actually delivers the page data
        //    and we can isolate the SSD-streaming cost from FFN compute.
        //    Either way this is a multi-millisecond blocking slice, so
        //    donate the worker thread (`block_in_place`) for its
        //    duration — otherwise the speculative prefetch tasks fired
        //    above can't get a worker to overlap this compute.
        let compute_start = Instant::now();
        let compute_us = run_compute_donated(|| {
            if self.core.options.io_only {
                let mut digest: u64 = 0;
                let mut total_bytes: u64 = 0;
                for r in &residents {
                    let bytes = r.data();
                    total_bytes += bytes.len() as u64;
                    // XOR every byte. The accumulator is 64 bits wide so we
                    // also rotate per chunk; this prevents a smart compiler
                    // from folding the loop and guarantees every read byte
                    // is observed, the whole point of `--io-only`.
                    let mut acc: u64 = 0;
                    for chunk in bytes.chunks(8) {
                        // Final chunk may be < 8 bytes; the remaining slots
                        // in `buf` stay zero. XOR with zero is a no-op, so
                        // the digest is still deterministic and every
                        // actually-read byte still contributes.
                        let mut buf = [0u8; 8];
                        buf[..chunk.len()].copy_from_slice(chunk);
                        acc ^= u64::from_le_bytes(buf);
                    }
                    // `% 63` (deliberately not 64): `rotate_left(0)` and
                    // `rotate_left(64)` are both no-ops on `u64`. Using 63
                    // keeps the rotation amount in `0..63` so adjacent
                    // expert ids actually pick different rotations and
                    // the per-expert contributions don't collapse.
                    digest ^= acc.rotate_left((r.id % 63) as u32);
                }
                let us = compute_start.elapsed().as_micros() as u64;
                debug!(
                    token = token_idx,
                    bytes_touched = total_bytes,
                    io_only_digest = digest,
                    "io-only mode: skipped FFN, touched buffer bytes"
                );
                us
            } else {
                // Real expert FFN forward pass over weights streamed from SSD.
                // `hidden` is the residual-stream activation already
                // computed at the top of `generate`; under
                // `Router::Linear` it is *the same* tensor that drove the
                // routing decision, so the FFN sees the exact gate
                // input (the production path), and under `Router::Markov`
                // it stays the synthetic placeholder the benchmark path
                // has always used.
                let x: &HiddenState = &hidden;
                let mut per_expert_y: Vec<HiddenState> = Vec::with_capacity(residents.len());
                let mut outputs: Vec<InferenceOutput> = Vec::with_capacity(residents.len());
                for r in &residents {
                    // ── Phase 3: GPU fast path ────────────────────────────────
                    // This synthetic path is compatibility/diagnostic
                    // infrastructure, not strict-hybrid qualification. PR3's
                    // policy applies only to real-model `moe_step`.
                    // CandleBackend::expert_matmul bails unconditionally, so we
                    // always guard behind is_gpu(). A logical/physical miss returns Err
                    // and we fall through to the CPU path below. Both F32 and
                    // (block-aligned) Q4_0 experts are eligible — see
                    // `Engine::gpu_eligible_dtype`.
                    debug!(
                        expert = r.id,
                        is_gpu = self.routed_expert_backend().is_gpu(),
                        gpu_eligible_dtype = self.gpu_eligible_dtype(),
                        "generate GPU fast-path guard"
                    );
                    let use_gpu =
                        self.routed_expert_backend().is_gpu() && self.gpu_eligible_dtype();
                    let gpu_result = if use_gpu {
                        let mut out_f16 = vec![half::f16::ZERO; self.core.shape.d_model];
                        let x_f16: Vec<half::f16> =
                            x.iter().map(|&f| half::f16::from_f32(f)).collect();
                        let x_view = crate::backend::TensorView {
                            data: &x_f16,
                            rows: 1,
                            cols: self.core.shape.d_model,
                        };
                        let mut out_view = crate::backend::TensorViewMut {
                            data: &mut out_f16,
                            rows: 1,
                            cols: self.core.shape.d_model,
                        };
                        // `generate` is the synthetic-benchmark path; it has no
                        // per-layer iteration, so we route everything through
                        // layer 0 — `expert_matmul` ignores `layer_idx` anyway
                        // (the trait API takes it only for future logging).
                        debug!(
                            expert = r.id,
                            is_gpu = self.routed_expert_backend().is_gpu(),
                            dtype = ?self.core.options.dtype,
                            "calling backend.expert_matmul"
                        );
                        let matmul_res = self.routed_expert_backend().expert_matmul(
                            0,
                            r.id,
                            x_view,
                            self.core.shape.d_model,
                            self.core.shape.d_ff,
                            &mut out_view,
                        );
                        debug!(
                            expert = r.id,
                            is_gpu = self.routed_expert_backend().is_gpu(),
                            dtype = ?self.core.options.dtype,
                            ok = matmul_res.is_ok(),
                            "returned from backend.expert_matmul"
                        );
                        match matmul_res {
                            Ok(()) => {
                                Some(out_f16.iter().map(|h| h.to_f32()).collect::<Vec<f32>>())
                            }
                            Err(_) => None,
                        }
                    } else {
                        None
                    };

                    let res = if let Some(gpu_out) = gpu_result {
                        Ok((
                            summarise_output_like_cpu(token_idx, r.id, &gpu_out),
                            gpu_out,
                        ))
                    } else {
                        dispatch_expert_forward(
                            self.core.options.dtype,
                            self.core.options.use_qmm_for_q4,
                            token_idx,
                            r,
                            x,
                            self.core.shape.d_model,
                            self.core.shape.d_ff,
                            self.core.options.policy.expert_size_tolerance(),
                            None,
                        )
                    };
                    match res {
                        Ok((out, y)) => {
                            outputs.push(out);
                            per_expert_y.push(y);
                        }
                        Err(e) => {
                            warn!(
                                token = token_idx,
                                expert = r.id,
                                error = %e,
                                "skipping expert: failed to reinterpret buffer as SwiGLU weights"
                            );
                        }
                    }
                }
                // Synthetic / benchmark path has no real gating network, so
                // weight every routed expert uniformly (`1/k`) — that matches
                // the legacy averaging behaviour bit-for-bit while flowing
                // through the new softmax-gated combiner signature.
                let scores = uniform_scores(per_expert_y.len());
                let combined = combine_outputs(&per_expert_y, &scores);
                let us = compute_start.elapsed().as_micros() as u64;
                debug!(
                    token = token_idx,
                    d_model = self.core.shape.d_model,
                    d_ff = self.core.shape.d_ff,
                    ?outputs,
                    combined_norm = combined.iter().map(|v| v * v).sum::<f32>().sqrt(),
                    "FFN forward complete"
                );
                us
            }
        });
        let _ = self.metrics.compute_hist.lock().record(compute_us.max(1));
        self.metrics
            .total_compute_us
            .fetch_add(compute_us, Ordering::Relaxed);
        self.metrics
            .total_io_wait_us
            .fetch_add(io_wait_us, Ordering::Relaxed);

        let cycle_us = cycle_start.elapsed().as_micros() as u64;
        let _ = self.metrics.cycle_hist.lock().record(cycle_us.max(1));
        self.metrics
            .total_cycle_us
            .fetch_add(cycle_us, Ordering::Relaxed);
        self.metrics
            .tokens_processed
            .fetch_add(1, Ordering::Relaxed);

        Ok(stats)
    }

    async fn fetch(self: &Arc<Self>, id: u32) -> Result<Arc<ExpertResident>, ExpertReadError> {
        match self.fetch_with_retry(id).await {
            Ok(r) => Ok(r),
            Err(e) => {
                // Critical-path miss could not be satisfied even after
                // retries. Surface the error to the caller —
                // `Engine::generate` propagates it as
                // `Err(ExpertReadError)` so the HTTP serving path can
                // return a 500 instead of crashing the process. The
                // real-transformer path uses [`Self::moe_step`] which
                // calls [`Self::try_fetch_with_skip`] instead, so a
                // single corrupt expert never kills the process either.
                warn!(expert = id, error = %e, "fatal: expert fetch failed after retries");
                Err(e)
            }
        }
    }

    /// Try to fetch an expert with exponential-backoff retry on
    /// transient I/O errors. Returns `Err(ExpertReadError::*)` when
    /// the request cannot be satisfied (corrupt file, persistent I/O
    /// error, or saturated buffer pool with every cache slot pinned).
    ///
    /// This is the production entry point: prefer it over the
    /// panicking [`Self::fetch`] when the caller has a way to
    /// degrade — e.g. the multi-expert `moe_step` can drop a single
    /// failed expert from the top-K mixture and continue.
    ///
    /// **SSD Read De-Duplication (gist Phase 1).** This method
    /// participates in a process-wide in-flight singleflight: when N
    /// concurrent callers all miss the cache on the same id, only
    /// the first issues a disk read; the rest park on a shared
    /// [`Notify`] and re-check the cache once the leader is done.
    /// This guarantees one SSD read per unique expert id across an
    /// entire continuous-batching wave, with no risk of deadlock if
    /// the [`BufferPool`] is saturated (the leader may still return
    /// [`ExpertReadError::PoolStarved`] and the waiters retry
    /// through their own [`Self::fetch_once`] path).
    pub async fn fetch_with_retry(
        self: &Arc<Self>,
        id: u32,
    ) -> Result<Arc<ExpertResident>, ExpertReadError> {
        // Fast path: already cached — no singleflight needed. We
        // deliberately do *not* bump the `hits` counter here: the
        // upstream `moe_step` path already increments hits/misses
        // before deciding to call us, so doing it again would
        // double-count.
        if let Some(r) = self.core.cache.get(id) {
            return Ok(r);
        }

        // Loop so that a follower whose leader failed re-contends
        // for the singleflight slot rather than barrelling into its
        // own disk read (which would be the thundering-herd case
        // F1.4 documents). Bound the contention loop so a stream of
        // failing leaders can still surface an error.
        const MAX_LEADER_ELECTIONS: usize = 4;
        for _election in 0..MAX_LEADER_ELECTIONS {
            // Singleflight: try to install a fresh Notify. If we win
            // the race we are the "leader" and will drive the actual
            // read. Otherwise we clone the existing Notify and wait
            // for the leader, then re-check the cache. We use
            // DashMap's `Entry::Occupied/Vacant` distinction so the
            // leader bit is unambiguous (Arc strong-count is racy
            // under TSO).
            let _guard = match SingleflightLeaderGuard::try_claim(
                self.core.in_flight.clone(),
                id,
            ) {
                Ok(guard) => guard,
                Err(notify) => {
                    // Pre-register as a waiter *before* re-checking the
                    // cache and the in_flight map, so we cannot miss the
                    // leader's `notify_waiters()` call if it lands
                    // between our entry lookup and our await. This is
                    // the standard `tokio::sync::Notify` race-free pattern.
                    let fut = notify.notified();
                    tokio::pin!(fut);
                    fut.as_mut().enable();
                    if let Some(r) = self.core.cache.get(id) {
                        self.metrics
                            .counters
                            .singleflight_followers
                            .fetch_add(1, Ordering::Relaxed);
                        return Ok(r);
                    }
                    if self.core.in_flight.contains_key(&id) {
                        fut.await;
                        if let Some(r) = self.core.cache.get(id) {
                            self.metrics
                                .counters
                                .singleflight_followers
                                .fetch_add(1, Ordering::Relaxed);
                            return Ok(r);
                        }
                        // Leader failed. Loop back and contend for the
                        // singleflight slot again. Exactly one woken follower
                        // becomes the new leader; the rest park on its Notify.
                    }
                    continue;
                }
            };

            // Re-check the cache now that we hold leadership. The
            // line-`2264` fast-path miss happened *before* we won the
            // `in_flight` election; in the window between the two, a
            // prior leader may have finished its read, inserted the
            // resident, and dropped its guard — and that guard's
            // `in_flight.remove` is exactly what freed the slot we
            // just won. Because `fetch_once` inserts into the cache
            // *before* the guard removes the `in_flight` slot, and our
            // winning `entry()` observed that removal, the insert is
            // guaranteed visible to this `get()`. Without this check
            // two callers that miss the fast path in quick succession
            // each become a leader and issue a redundant disk read,
            // breaking the SSD-dedup invariant asserted by
            // `fetch_with_retry_deduplicates_concurrent_reads`. The
            // guard still runs on this early return, freeing the slot
            // we installed and waking any followers parked on us.
            if let Some(r) = self.core.cache.get(id) {
                self.metrics
                    .counters
                    .singleflight_followers
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(r);
            }

            const MAX_ATTEMPTS: usize = 3;
            let mut last_err: Option<String> = None;
            for attempt in 0..MAX_ATTEMPTS {
                match self.fetch_once(id).await {
                    Ok(r) => {
                        if attempt > 0 {
                            info!(expert = id, attempt, "expert fetch recovered after retry");
                        }
                        return Ok(r);
                    }
                    Err(FetchOnceError::PoolStarved) => {
                        return Err(ExpertReadError::PoolStarved { id });
                    }
                    Err(FetchOnceError::Io(msg)) => {
                        last_err = Some(msg.clone());
                        if attempt + 1 < MAX_ATTEMPTS {
                            // Exponential backoff: 10ms, 40ms, 160ms.
                            // Cap at 500ms to keep request latency
                            // bounded — the real-transformer path can
                            // skip failed experts so a long retry
                            // storm is worse than a quick degraded
                            // response.
                            let backoff_ms = (10u64 << (attempt * 2)).min(500);
                            warn!(
                                expert = id,
                                attempt,
                                backoff_ms,
                                error = %msg,
                                "expert fetch failed; will retry"
                            );
                            tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                        }
                    }
                }
            }
            return Err(ExpertReadError::Io {
                id,
                attempts: MAX_ATTEMPTS,
                source: last_err.unwrap_or_else(|| "unknown".into()),
            });
        }
        // Exhausted the leader-election budget without a successful
        // fetch and without ourselves becoming leader. Treat this
        // as an I/O failure so callers can surface a 503.
        Err(ExpertReadError::Io {
            id,
            attempts: MAX_LEADER_ELECTIONS,
            source: "exhausted singleflight leader-election budget".into(),
        })
    }

    async fn gpu_native_demand_source(
        self: &Arc<Self>,
        global_id: u32,
        residents: &mut HashMap<u32, Arc<ExpertResident>>,
    ) -> Result<Arc<ExpertResident>, GpuNativeDemandResidencyError> {
        let qualification = self.gpu_native_demand_source_qualification();
        if let Some(state) = qualification.as_ref() {
            state.record_source_request(global_id);
        }
        if let Some(resident) = residents.get(&global_id) {
            return Ok(resident.clone());
        }
        let resident = match self.core.cache.get(global_id) {
            Some(resident) => {
                if let Some(state) = qualification.as_ref() {
                    state.source_ram_hits.fetch_add(1, Ordering::Relaxed);
                }
                resident
            }
            None => {
                if let Some(state) = qualification.as_ref() {
                    state.source_ram_misses.fetch_add(1, Ordering::Relaxed);
                }
                self.fetch_with_retry(global_id).await?
            }
        };
        residents.insert(global_id, resident.clone());
        Ok(resident)
    }

    async fn gpu_native_source_physical_missing_set(
        self: &Arc<Self>,
        global_ids: &[u32],
        residents: &mut HashMap<u32, Arc<ExpertResident>>,
    ) -> Result<(), GpuNativeDemandResidencyError> {
        let qualification = self.gpu_native_demand_source_qualification();
        if let Some(state) = qualification.as_ref() {
            state.record_source_set(global_ids);
        }
        let started = Instant::now();
        let result = match qualification.as_ref().map(|state| state.arm) {
            Some(GpuNativeDemandSourceQualificationArm::Control) => {
                self.gpu_native_sequential_source_physical_missing_set(global_ids, residents)
                    .await
            }
            Some(GpuNativeDemandSourceQualificationArm::Treatment) | None => {
                self.gpu_native_production_source_physical_missing_set(global_ids, residents)
                    .await
            }
        };
        if let Some(state) = qualification.as_ref() {
            state
                .source_acquisition_wall_us
                .fetch_add(qualification_elapsed_us(started), Ordering::Relaxed);
        }
        result
    }

    /// The exact ordinary foreground source loop shared by control and the
    /// treatment's conservative RAM-residency fallback. It owns every real
    /// recency mutation, hit/miss classification, retry, eviction, insertion,
    /// and request-order side effect.
    async fn gpu_native_sequential_source_physical_missing_set(
        self: &Arc<Self>,
        global_ids: &[u32],
        residents: &mut HashMap<u32, Arc<ExpertResident>>,
    ) -> Result<(), GpuNativeDemandResidencyError> {
        for &global_id in global_ids {
            if residents.contains_key(&global_id) {
                continue;
            }
            let individual_started = Instant::now();
            let source_result = self
                .gpu_native_demand_source(global_id, residents)
                .await
                .map(|_| ());
            if let Some(state) = self.gpu_native_demand_source_qualification() {
                state.individual_source_service_us.fetch_add(
                    qualification_elapsed_us(individual_started),
                    Ordering::Relaxed,
                );
            }
            source_result?;
        }
        Ok(())
    }

    /// Ordinary production source for an exact physical-missing set. Setup or
    /// contention failures before cache reservation remain opportunistic and
    /// use the complete legacy sequential helper. Once the atomic victim
    /// schedule has removed residents, pool/read/commit failures are surfaced
    /// fail-closed: replaying sequentially could stop on an early expert after
    /// later experts' victims were already removed, so it would falsely claim
    /// exact failure-state semantics.
    async fn gpu_native_production_source_physical_missing_set(
        self: &Arc<Self>,
        global_ids: &[u32],
        residents: &mut HashMap<u32, Arc<ExpertResident>>,
    ) -> Result<(), GpuNativeDemandResidencyError> {
        let telemetry = self.production_demand_source.clone();
        telemetry.source_sets.fetch_add(1, Ordering::Relaxed);
        let unresolved = global_ids
            .iter()
            .copied()
            .filter(|global_id| !residents.contains_key(global_id))
            .collect::<Vec<_>>();

        if unresolved.len() <= 1 {
            telemetry
                .fallback_single_item
                .fetch_add(1, Ordering::Relaxed);
            return self
                .gpu_native_sequential_source_physical_missing_set(global_ids, residents)
                .await;
        }

        // Conservative PR2-A eligibility probe: `contains` is explicitly
        // non-recency-mutating. One hit makes the entire original-order set
        // sequential; mixed RAM state is never batched.
        if unresolved
            .iter()
            .any(|global_id| self.core.cache.contains(*global_id))
        {
            telemetry.fallback_mixed_ram.fetch_add(1, Ordering::Relaxed);
            return self
                .gpu_native_sequential_source_physical_missing_set(global_ids, residents)
                .await;
        }
        telemetry
            .batch_eligible_sets
            .fetch_add(1, Ordering::Relaxed);
        telemetry.batch_attempts.fetch_add(1, Ordering::Relaxed);

        // Reserve process-wide leadership in original request order using the
        // exact same map/guard protocol as `fetch_with_retry`.
        let mut leadership = MultiIdSingleflightLeadership {
            guards: Vec::with_capacity(unresolved.len()),
            telemetry: telemetry.clone(),
            successful: false,
        };
        for &global_id in &unresolved {
            match SingleflightLeaderGuard::try_claim(self.core.in_flight.clone(), global_id) {
                Ok(guard) => {
                    telemetry
                        .singleflight_ids_claimed
                        .fetch_add(1, Ordering::Relaxed);
                    leadership.guards.push(guard);
                }
                Err(_leader_notify) => {
                    telemetry
                        .singleflight_followers_observed
                        .fetch_add(1, Ordering::Relaxed);
                    telemetry
                        .fallback_singleflight_contention
                        .fetch_add(1, Ordering::Relaxed);
                    drop(leadership);
                    #[cfg(test)]
                    {
                        let barrier = {
                            self.production_batch_test_hooks
                                .lock()
                                .after_claim_rollback
                                .clone()
                        };
                        if let Some(barrier) = barrier {
                            barrier.wait().await;
                            barrier.wait().await;
                        }
                    }
                    return self
                        .gpu_native_sequential_source_physical_missing_set(global_ids, residents)
                        .await;
                }
            }
        }

        #[cfg(test)]
        {
            let barrier = { self.production_batch_test_hooks.lock().after_claims.clone() };
            if let Some(barrier) = barrier {
                barrier.wait().await;
                barrier.wait().await;
            }
        }

        // A cache insert can win between the eligibility probe and the last
        // leadership claim. Recheck without recency mutation while all ids are
        // owned; any newly resident id aborts the complete batch.
        if unresolved
            .iter()
            .any(|global_id| self.core.cache.contains(*global_id))
        {
            telemetry.fallback_mixed_ram.fetch_add(1, Ordering::Relaxed);
            drop(leadership);
            return self
                .gpu_native_sequential_source_physical_missing_set(global_ids, residents)
                .await;
        }

        let reservation_outcome =
            match self.core.cache.try_reserve_exact_demand(&unresolved) {
                Ok(reservation) => reservation,
                Err(_) => {
                    telemetry
                        .fallback_reservation
                        .fetch_add(1, Ordering::Relaxed);
                    drop(leadership);
                    return self
                        .gpu_native_sequential_source_physical_missing_set(global_ids, residents)
                        .await;
                }
            };
        let reserved_slots = reservation_outcome.reservation.remaining();
        telemetry
            .cache_slots_reserved
            .fetch_add(reserved_slots as u64, Ordering::Relaxed);
        let mut cache_reservation = ProductionCacheReservation {
            inner: reservation_outcome.reservation,
            telemetry: telemetry.clone(),
        };
        if let Some(qualification) = self.gpu_native_demand_source_qualification() {
            for &victim_id in &reservation_outcome.eviction_ids {
                qualification.record_cache_eviction(victim_id);
            }
        }
        drop(reservation_outcome.victims);

        let mut buffers = Vec::with_capacity(unresolved.len());
        for _ in 0..unresolved.len() {
            match self.core.pool.try_acquire() {
                Some(buffer) => buffers.push(buffer),
                None => {
                    let acquired = buffers.len();
                    drop(buffers);
                    drop(cache_reservation);
                    drop(leadership);
                    return Err(
                        GpuNativeDemandResidencyError::ProductionBatchPoolUnavailableAfterReservation {
                            requested: unresolved.len(),
                            acquired,
                        },
                    );
                }
            }
        }

        #[cfg(test)]
        {
            let barrier = {
                self.production_batch_test_hooks
                    .lock()
                    .after_buffers
                    .clone()
            };
            if let Some(barrier) = barrier {
                barrier.wait().await;
                barrier.wait().await;
            }
        }

        let batch_started = Instant::now();
        let mut refs = buffers.iter_mut().collect::<Vec<_>>();
        let _foreground = self.core.governor.foreground_guard();
        let read_result = self
            .core
            .storage
            .read_experts_batch(&unresolved, &mut refs)
            .await;
        drop(_foreground);
        let batch_wall_us = qualification_elapsed_us(batch_started);
        let expected_bytes = buffers.iter().map(|buffer| buffer.len()).sum::<usize>();
        let read_bytes = match read_result {
            Ok(read_bytes) if read_bytes == expected_bytes => read_bytes,
            result => {
                let source = match result {
                    Ok(read_bytes) => {
                        format!("batch returned {read_bytes} bytes, expected {expected_bytes}")
                    }
                    Err(error) => error.to_string(),
                };
                drop(buffers);
                drop(cache_reservation);
                drop(leadership);
                return Err(
                    GpuNativeDemandResidencyError::ProductionBatchReadFailedAfterReservation {
                        global_ids: unresolved,
                        source,
                    },
                );
            }
        };

        if let Some(qualification) = self.gpu_native_demand_source_qualification() {
            for &global_id in &unresolved {
                qualification.record_source_request(global_id);
                qualification
                    .source_ram_misses
                    .fetch_add(1, Ordering::Relaxed);
            }
            qualification
                .source_nvme_reads
                .fetch_add(unresolved.len() as u64, Ordering::Relaxed);
            qualification
                .source_nvme_bytes
                .fetch_add(read_bytes as u64, Ordering::Relaxed);
        }
        self.metrics
            .counters
            .bytes_read
            .fetch_add(read_bytes as u64, Ordering::Relaxed);
        let _ = self.metrics.io_hist.lock().record(batch_wall_us.max(1));

        let block_align = self.core.storage.config().block_align;
        let completed = unresolved
            .iter()
            .copied()
            .zip(buffers)
            .map(|(global_id, buffer)| {
                (
                    global_id,
                    Arc::new(ExpertResident::new_with_block_align(
                        global_id,
                        buffer,
                        block_align,
                    )),
                )
            })
            .collect::<HashMap<_, _>>();
        let staged = qualification_order_completed_residents(&unresolved, completed)?;
        for (global_id, resident) in staged {
            if cache_reservation.commit(resident.clone()).is_err() {
                telemetry
                    .batch_commit_violations
                    .fetch_add(1, Ordering::Relaxed);
                return Err(
                    GpuNativeDemandResidencyError::ProductionBatchCommitViolation { global_id },
                );
            }
            if let Some(qualification) = self.gpu_native_demand_source_qualification() {
                qualification.record_cache_insert(global_id);
            }
            residents.insert(global_id, resident);
        }
        telemetry.record_success(unresolved.len());
        leadership.finish();
        Ok(())
    }

    async fn ensure_gpu_native_logical_demand_set(
        self: &Arc<Self>,
        global_ids: &[u32],
        residents: &mut HashMap<u32, Arc<ExpertResident>>,
    ) -> Result<(Vec<GpuAdmission>, usize), GpuNativeDemandResidencyError> {
        let gpu = self.execution_context().gpu_expert_cache();
        let mut payloads = HashMap::with_capacity(global_ids.len());
        for attempt in 1..=GPU_NATIVE_LOGICAL_DEMAND_SET_ATTEMPTS {
            match gpu.demand_admit_set(global_ids, &payloads)? {
                GpuDemandSetAdmission::Ready {
                    admissions,
                    newly_admitted,
                } => {
                    if newly_admitted > 0 {
                        if let Some(prom) = self.metrics.prom.as_ref() {
                            prom.record_promotions(newly_admitted as u64);
                            prom.set_vram_used_bytes(gpu.used_bytes());
                        }
                    }
                    return Ok((admissions, newly_admitted));
                }
                GpuDemandSetAdmission::PayloadRequired(missing)
                    if attempt == GPU_NATIVE_LOGICAL_DEMAND_SET_ATTEMPTS =>
                {
                    return Err(
                        GpuNativeDemandResidencyError::LogicalDemandSetRecoveryExhausted {
                            global_ids: missing,
                            attempts: GPU_NATIVE_LOGICAL_DEMAND_SET_ATTEMPTS,
                        },
                    );
                }
                GpuDemandSetAdmission::PayloadRequired(missing) => {
                    for global_id in missing {
                        let resident = self.gpu_native_demand_source(global_id, residents).await?;
                        payloads.insert(
                            global_id,
                            Arc::new(GpuResident::new_with_dtype(
                                global_id,
                                resident.data().to_vec(),
                                self.core.options.dtype,
                            )),
                        );
                    }
                }
            }
        }
        unreachable!("logical demand-set loop has a fixed positive attempt count")
    }

    /// GPU-native foreground demand service. Exact physical hits are
    /// authoritative and never touch the logical cache, RAM, or storage.
    /// Only physical misses use the existing RAM -> `fetch_with_retry`
    /// hierarchy, protect/admit their complete missing subset atomically, and
    /// authorize a physical install. Demand never consults the speculative
    /// governor or semaphore.
    pub(crate) async fn ensure_gpu_native_demand_residency(
        self: &Arc<Self>,
        layer_index: usize,
        global_ids: &[u32],
    ) -> Result<
        Vec<crate::backend::gpu_native::GpuNativeQ4ExpertResidency>,
        GpuNativeDemandResidencyError,
    > {
        let manager = self
            .core
            .gpu_native_residency
            .as_ref()
            .cloned()
            .ok_or(GpuNativeDemandResidencyError::ManagerNotInstalled)?;
        let qualification = self.gpu_native_demand_source_qualification();
        let _qualification_guard = qualification
            .as_ref()
            .cloned()
            .map(QualificationDemandServiceGuard::enter)
            .transpose()?;
        let num_layers = manager.plan().num_layers();
        let experts_per_layer = manager.plan().geometry().num_experts() as u32;
        let mut seen = HashSet::with_capacity(global_ids.len());
        for &global_id in global_ids {
            if !seen.insert(global_id) {
                return Err(
                    GpuNativeTieredResidencyError::DuplicateDemandExpert { global_id }.into(),
                );
            }
            let identity =
                gpu_native_global_to_layer_local(global_id, num_layers, experts_per_layer)?;
            if identity.layer_index != layer_index {
                return Err(GpuNativeTieredResidencyError::DemandLayerMismatch {
                    requested_layer: layer_index,
                    global_id,
                    actual_layer: identity.layer_index,
                }
                .into());
            }
        }

        let mut residents = HashMap::with_capacity(global_ids.len());
        let mut recovery_attempts = 0usize;
        loop {
            // No logical-cache lock is held while probing the physical layer.
            // A probe result may race with a later physical eviction; the
            // fixed one-retry recovery below re-probes the complete set.
            let physical_probe_started = Instant::now();
            let mut physical_current = Vec::with_capacity(global_ids.len());
            for &global_id in global_ids {
                let current = manager.has_current_for_demand(global_id)?;
                physical_current.push(current);
            }
            if let Some(state) = qualification.as_ref() {
                state
                    .physical_probe_us
                    .fetch_add(qualification_elapsed_us(physical_probe_started), Ordering::Relaxed);
            }
            let physical_missing = gpu_native_physical_missing_ids(global_ids, &physical_current);
            if let Some(state) = qualification.as_ref() {
                state
                    .physical_missing_experts
                    .fetch_add(physical_missing.len() as u64, Ordering::Relaxed);
                state
                    .physical_missing_ids
                    .lock()
                    .record_set(&physical_missing);
            }

            // Physical hits never reach this logical/source block. Protection
            // is metadata-only and releases the logical mutex before every
            // RAM/NVMe await, so there is no logical -> physical nesting.
            let mut admissions_by_id = HashMap::with_capacity(physical_missing.len());
            let _logical_protection = if physical_missing.is_empty() {
                None
            } else {
                let gpu = self.execution_context().gpu_expert_cache().clone();
                let protection = gpu.protect_demand_set(&physical_missing)?;
                for &global_id in &physical_missing {
                    if !residents.contains_key(&global_id) {
                        manager.record_physical_source_acquisition();
                    }
                }
                self.gpu_native_source_physical_missing_set(&physical_missing, &mut residents)
                    .await?;
                let logical_admission_started = Instant::now();
                let (admissions, newly_admitted) = self
                    .ensure_gpu_native_logical_demand_set(&physical_missing, &mut residents)
                    .await?;
                if let Some(state) = qualification.as_ref() {
                    state.logical_demand_admission_us.fetch_add(
                        qualification_elapsed_us(logical_admission_started),
                        Ordering::Relaxed,
                    );
                }
                manager.record_logical_admissions_for_physical_misses(newly_admitted);
                admissions_by_id
                    .extend(physical_missing.iter().copied().zip(admissions.into_iter()));
                Some(protection)
            };

            let demands = global_ids
                .iter()
                .copied()
                .enumerate()
                .map(|(index, global_id)| {
                    if physical_current[index] {
                        GpuNativeDemandExpert::current(global_id)
                    } else {
                        GpuNativeDemandExpert::install(
                            global_id,
                            residents
                                .get(&global_id)
                                .expect("physical miss acquired an authoritative source")
                                .clone(),
                            admissions_by_id
                                .get(&global_id)
                                .expect("physical miss established a logical admission")
                                .clone(),
                        )
                    }
                })
                .collect::<Vec<_>>();

            let physical_install_started = Instant::now();
            let physical_install_result = manager.ensure_demand_set(
                GpuNativeResidencyPriority::Demand,
                layer_index,
                &demands,
            );
            if let Some(state) = qualification.as_ref() {
                state.physical_demand_install_us.fetch_add(
                    qualification_elapsed_us(physical_install_started),
                    Ordering::Relaxed,
                );
            }
            match physical_install_result {
                Ok(residencies) => return Ok(residencies),
                Err(GpuNativeTieredResidencyError::DemandSourceMissing { global_id: _ })
                    if gpu_native_physical_demand_recovery_allowed(recovery_attempts) =>
                {
                    recovery_attempts += 1;
                    // A Current record changed before the physical transaction.
                    // The loop re-probes the whole selected set, but only the
                    // now-missing members may acquire source/admission state.
                }
                Err(GpuNativeTieredResidencyError::DemandSourceMissing { global_id }) => {
                    return Err(
                        GpuNativeDemandResidencyError::PhysicalDemandRecoveryExhausted {
                            global_id,
                            recovery_attempts,
                        },
                    );
                }
                Err(error) => return Err(error.into()),
            }
        }
    }

    /// One single fetch attempt. Acquires a buffer (yielding briefly
    /// if the pool is under pressure), issues the read, and either
    /// installs the resident in the cache or surfaces the I/O error
    /// to the retry loop.
    async fn fetch_once(self: &Arc<Self>, id: u32) -> Result<Arc<ExpertResident>, FetchOnceError> {
        let io_start = Instant::now();
        // Acquire-with-eviction: evict an LRU entry if the cache is at
        // capacity (which releases its `PooledBuffer` on `Arc` drop),
        // then wait for a free buffer.
        //
        // The critical-path fetch must outwait a multi-millisecond
        // prefetch read — an in-flight prefetch holds a pool buffer for
        // the *entire* duration of its 672 MB read, so a bounded
        // `yield_now()` spin (which completes in microseconds) gives up
        // long before any prefetch can release its buffer and surfaces a
        // spurious `PoolStarved` panic. Instead we park on the pool's
        // async `acquire()`, which registers on the pool's `Notify` and
        // wakes the instant a buffer is released. `EngineCore::new`
        // reserves one headroom slot that prefetch can never take, so a
        // foreground fetch is always guaranteed a buffer becomes
        // available — `acquire()` therefore cannot block forever even
        // when the cache is fully pinned and speculation is saturated.
        if self.core.cache.len() >= self.core.cache.capacity() {
            if let Some(evicted) = self.core.cache.evict_lru() {
                debug!(evicted = evicted.id, "evicted LRU to make room");
                if let Some(qualification) = self.gpu_native_demand_source_qualification() {
                    qualification.record_cache_eviction(evicted.id);
                }
                drop(evicted);
            }
        }
        let mut buf = self.core.pool.acquire().await;
        // Tier 4: mark this as a foreground (token-blocking) read for the
        // duration of the device I/O so the governor throttles
        // speculation that would otherwise queue ahead of it. No-op when
        // the governor is disabled.
        let _fg = self.core.governor.foreground_guard();
        let read_result = self.core.storage.read_expert(id, &mut buf).await;
        match read_result {
            Ok(_) => {
                let io_us = io_start.elapsed().as_micros() as u64;
                let _ = self.metrics.io_hist.lock().record(io_us.max(1));
                if let Some(qualification) = self.gpu_native_demand_source_qualification() {
                    qualification
                        .source_nvme_reads
                        .fetch_add(1, Ordering::Relaxed);
                    qualification
                        .source_nvme_bytes
                        .fetch_add(buf.len() as u64, Ordering::Relaxed);
                }
                // Track every byte the engine actually pulls off the
                // SSD — including `fetch_with_retry`'s leader path,
                // not just the `moe_step` critical path. This is
                // what makes the SSD-read-dedup invariant in
                // `fetch_with_retry_deduplicates_concurrent_reads`
                // (and any future observability) directly checkable:
                // a deduplicated batch of N concurrent fetches must
                // increase `bytes_read` by exactly one expert's
                // worth, regardless of which call site issued them.
                self.metrics
                    .counters
                    .bytes_read
                    .fetch_add(buf.len() as u64, Ordering::Relaxed);
                let resident = Arc::new(ExpertResident::new_with_block_align(
                    id,
                    buf,
                    self.core.storage.config().block_align,
                ));
                match self.core.cache.insert(resident.clone()) {
                    Ok(Some(evicted)) => {
                        if let Some(qualification) =
                            self.gpu_native_demand_source_qualification()
                        {
                            qualification.record_cache_insert(id);
                            qualification.record_cache_eviction(evicted.id);
                        }
                        debug!(expert = id, "inserted (with eviction)")
                    }
                    Ok(None) => {
                        if let Some(qualification) =
                            self.gpu_native_demand_source_qualification()
                        {
                            qualification.record_cache_insert(id);
                        }
                        debug!(expert = id, "inserted")
                    }
                    Err(rejected) => {
                        // Cache is full of pinned entries — surface this
                        // explicitly. The caller still gets a usable
                        // `Arc<ExpertResident>` (the bytes are loaded);
                        // it just won't be cached, so the next access
                        // will re-fetch. This degrades gracefully
                        // rather than violating the pin contract.
                        warn!(
                            expert = id,
                            "expert loaded but cache rejected insert (every slot pinned); \
                             returning resident without caching"
                        );
                        return Ok(rejected);
                    }
                }
                Ok(resident)
            }
            Err(e) => {
                // The buffer is returned to the pool when `buf` is dropped.
                Err(FetchOnceError::Io(e.to_string()))
            }
        }
    }

    fn spawn_gpu_native_ram_to_vram_prefetch(
        self: &Arc<Self>,
        manager: Arc<GpuNativeTieredResidencyManager>,
        resident: Arc<ExpertResident>,
        p: f64,
    ) {
        if !self.background_tasks.accepts_work() {
            return;
        }
        let id = resident.id;
        // This is the prediction's one and only governor decision. A PCIe
        // write does not call `record_completed`; that denominator remains
        // tied to speculative RAM/NVMe prefetch usefulness.
        if !self.core.governor.admit(p) {
            self.metrics
                .counters
                .prefetch_dropped_governor
                .fetch_add(1, Ordering::Relaxed);
            return;
        }
        let permit = match self.core.prefetch_semaphore.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                self.metrics
                    .counters
                    .prefetch_dropped_concurrency
                    .fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        let me = self.clone();
        self.background_tasks.spawn(async move {
            let _permit = permit;
            let Some(admission) = me.ensure_speculative_gpu_admission(&resident) else {
                return;
            };
            match manager.ensure_speculative_resident(
                id,
                &resident,
                &admission,
                GpuNativeResidencyPriority::Speculative { score: p },
            ) {
                Ok(
                    GpuNativeSpeculativeInstall::Hit(_)
                    | GpuNativeSpeculativeInstall::Installed(_)
                    | GpuNativeSpeculativeInstall::DroppedCapacityOrPressure
                    | GpuNativeSpeculativeInstall::StaleLogicalGeneration,
                ) => {}
                Err(error) => debug!(
                    expert = id,
                    error = %error,
                    "RAM-to-VRAM speculative residency was dropped"
                ),
            }
        });
    }

    fn spawn_prefetch(self: &Arc<Self>, id: u32, p: f64) {
        if !self.background_tasks.accepts_work() {
            return;
        }
        if let Some(manager) = self.core.gpu_native_residency.as_ref().cloned() {
            match manager.probe_speculative(
                id,
                GpuNativeResidencyPriority::Speculative { score: p },
            ) {
                Ok(GpuNativeSpeculativeProbe::Hit(_))
                | Ok(GpuNativeSpeculativeProbe::DroppedPressure) => return,
                Ok(GpuNativeSpeculativeProbe::Miss) => {
                    if let Some(resident) = self.core.cache.get(id) {
                        self.spawn_gpu_native_ram_to_vram_prefetch(manager, resident, p);
                        return;
                    }
                }
                Err(error) => {
                    debug!(
                        expert = id,
                        error = %error,
                        "dropping invalid GPU-native speculative residency request"
                    );
                    return;
                }
            }
        }
        // **Dedup before spending a permit.** `union_prefetch` and
        // `speculate_layer_ahead` can both nominate the same id for one
        // token; without this pre-check each duplicate consumed a
        // semaphore permit (and a spawned task) before the singleflight
        // map rejected it — under a tight `max_concurrent_prefetches`
        // budget, two predictions of the same expert could crowd out a
        // genuinely new one. Racing with a concurrent insert/landing is
        // fine: the post-spawn `contains` re-check and the singleflight
        // entry below stay authoritative.
        if self.core.cache.contains(id) || self.core.in_flight.contains_key(&id) {
            return;
        }
        // **Tier 4 admission gate.** Before spending a semaphore permit
        // (or any device bandwidth), ask the adaptive governor whether
        // this speculative read is worth issuing *right now*, given its
        // predicted score `p`, the recently-measured prefetch precision,
        // and how many foreground (token-blocking) misses are currently
        // competing for the SSD. A disabled governor always admits, so
        // the legacy unbounded behaviour is preserved bit-for-bit.
        if !self.core.governor.admit(p) {
            self.metrics
                .counters
                .prefetch_dropped_governor
                .fetch_add(1, Ordering::Relaxed);
            debug!(
                expert = id,
                score = p,
                "governor throttled speculative prefetch"
            );
            return;
        }
        // Speculative prefetches are *bounded*: each spawn must hold
        // an owned permit from `prefetch_semaphore` for the duration
        // of the I/O. When the configured ceiling
        // (`EngineOptions::max_concurrent_prefetches`) is saturated
        // we drop the request rather than queue it — speculative
        // loads are valuable only if they complete before the real
        // miss, and queuing them defeats that. The drop is observable
        // via the `prefetch_dropped_concurrency` counter.
        let permit = match self.core.prefetch_semaphore.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                self.metrics
                    .counters
                    .prefetch_dropped_concurrency
                    .fetch_add(1, Ordering::Relaxed);
                debug!(
                    expert = id,
                    "skipping prefetch: concurrency ceiling reached"
                );
                return;
            }
        };
        let me = self.clone();
        self.background_tasks.spawn(async move {
            // Permit released on task completion (drop). Holding it
            // across the I/O is what enforces the bound.
            let _permit = permit;
            // Re-check (could have been loaded by another task in the meantime).
            if me.core.cache.contains(id) {
                return;
            }
            // **Get-or-wait join (Part 3).** Register this prefetch in
            // the singleflight `in_flight` map *before* issuing the
            // read, using the exact same leader/follower protocol as
            // `fetch_with_retry`. The payoff: a foreground cache miss
            // for the same id (the gate reaching a layer whose experts
            // we predicted a layer ahead) becomes a *follower* that
            // parks on this prefetch's `Notify` and re-checks the cache
            // when it lands — turning what used to be a duplicate
            // blocking SSD read into a sub-millisecond wait on an
            // already-in-flight speculation. If someone else (a
            // foreground leader, or another prefetch) already owns the
            // in-flight slot, there is nothing useful to do: they are
            // already fetching this id, so drop.
            // The guard removes the in-flight slot and notifies every
            // parked follower on *every* exit path below (buffer-starved
            // early return, read error, or success). Followers then
            // re-check the cache: a hit on success, or a re-contention
            // for leadership on failure — never a wedged stale entry.
            let _guard = match SingleflightLeaderGuard::try_claim(
                me.core.in_flight.clone(),
                id,
            ) {
                Ok(guard) => guard,
                Err(_) => return,
            };
            // **Double-buffered acquire (Part 2).** Speculation draws
            // from the **shadow** (Buffer B) half of the pool, never the
            // primary (Buffer A) half that backs the resident LRU and
            // the foreground miss path. This is the invariant that
            // protects compute on Buffer A: a speculative look-ahead can
            // never steal the buffer a real cache miss needs. When the
            // shadow half is disabled (legacy single-pool configs that
            // call `BufferPool::new`), fall back to the previous
            // non-evicting primary `try_acquire` so those deployments
            // keep prefetching exactly as before. Either way we *never*
            // block or evict: a busy pool simply drops the speculative
            // load.
            let mut buf = match me.core.pool.try_acquire_shadow() {
                Some(b) => b,
                None if me.core.pool.shadow_capacity() == 0 => match me.core.pool.try_acquire() {
                    Some(b) => b,
                    None => {
                        me.note_prefetch_dropped_pool_starved(id);
                        return;
                    }
                },
                None => {
                    // **Shadow recycling (Finding 3).** Buffer B is
                    // starved — but its capacity may be parked inside
                    // long-lived shadow-backed residents rather than
                    // genuinely in flight. Evict the LRU unpinned
                    // shadow-backed resident; dropping its (typically
                    // sole) `Arc` returns the buffer to the shadow
                    // free-list, so the retry below usually succeeds.
                    // If a clone of the resident is still referenced
                    // elsewhere the buffer comes back later — drop
                    // this speculative load, exactly like before.
                    match me.core.cache.evict_lru_shadow_backed().and_then(|victim| {
                        debug!(
                            expert = id,
                            recycled = victim.id,
                            "shadow pool starved: recycled LRU shadow-backed resident"
                        );
                        drop(victim);
                        me.core.pool.try_acquire_shadow()
                    }) {
                        Some(b) => b,
                        None => {
                            me.note_prefetch_dropped_pool_starved(id);
                            return;
                        }
                    }
                }
            };
            let started = Instant::now();
            match me.core.storage.read_expert(id, &mut buf).await {
                Ok(_) => {
                    me.metrics
                        .counters
                        .prefetch_completed
                        .fetch_add(1, Ordering::Relaxed);
                    // Tier 4: a speculative read landed in the cache. This
                    // is the precision *denominator*; the matching
                    // numerator (`record_used`) fires if/when a hit later
                    // consumes this shadow-backed resident. No-op when the
                    // governor is disabled.
                    me.core.governor.record_completed();
                    me.metrics
                        .counters
                        .bytes_read
                        .fetch_add(buf.len() as u64, Ordering::Relaxed);
                    // **Self-balancing shadow accounting.** We insert the
                    // resident still holding its *shadow*-tagged buffer
                    // rather than calling `BufferPool::promote_shadow`
                    // first. Promotion permanently re-tags the slot as
                    // primary, so on eviction it would return to the
                    // primary free-list — over a long-running serve every
                    // confirmed-then-evicted prefetch would migrate one
                    // buffer from shadow to primary, draining Buffer B to
                    // zero and silently disabling look-ahead. Keeping the
                    // buffer shadow-tagged means it returns to the shadow
                    // free-list on eviction, holding Buffer B's capacity
                    // constant for the life of the process. The bytes are
                    // identical either way (promotion only changes the
                    // drop destination), and a shadow-backed resident
                    // serves cache hits exactly like a primary-backed one.
                    let resident = Arc::new(ExpertResident::new_with_block_align(
                        id,
                        buf,
                        me.core.storage.config().block_align,
                    ));
                    // Prefetches are best-effort: if the cache rejects
                    // the insert (every slot pinned), the resident drops
                    // here and its buffer returns to the shadow pool —
                    // exactly the right behaviour for a speculative load.
                    if let Err(_rejected) = me.core.cache.insert(resident.clone()) {
                        debug!(
                            expert = id,
                            "prefetch dropped: cache full of pinned entries"
                        );
                        return;
                    }
                    // With the optional Slice 9 plane, the same prediction and
                    // semaphore permit continue through logical admission and
                    // RAM -> VRAM. There is no second governor decision and no
                    // second prefetch scheduler.
                    if let Some(manager) = me.core.gpu_native_residency.as_ref().cloned() {
                        if let Some(admission) = me.ensure_speculative_gpu_admission(&resident) {
                            if let Err(error) = manager.ensure_speculative_resident(
                                id,
                                &resident,
                                &admission,
                                GpuNativeResidencyPriority::Speculative { score: p },
                            ) {
                                debug!(
                                    expert = id,
                                    error = %error,
                                    "cold prefetch landed in RAM but GPU-native speculation was dropped"
                                );
                            }
                        }
                    } else if let Some(gpu) = me.core.gpu_cache.as_ref() {
                        // **Eager logical GPU admission.** A speculative prefetch is
                        // a strong "about to be routed" signal, so try to
                        // stage the host bytes into the admission LRU *now* instead
                        // of waiting for `promote_after_hits` RAM hits — the
                        // lazy path can leave a predicted expert on the CPU
                        // fallback path for its first few activations.
                        // Only attempt eager admission when the logical cache has room;
                        // otherwise we would copy ~expert_size bytes into a Vec only to have the
                        // non-evicting promotion path immediately reject it.
                        let bytes = resident.data().len();
                        if (gpu.used_bytes() as usize).saturating_add(bytes) <= gpu.capacity_bytes()
                        {
                            me.try_promote_resident_to_gpu(&resident);
                        }
                    }
                    debug!(
                        expert = id,
                        prob = p,
                        elapsed_us = started.elapsed().as_micros() as u64,
                        "prefetch complete"
                    );
                    // `_guard` drops here: the in-flight slot is removed
                    // and any foreground follower waiting on this id is
                    // woken to re-check the cache — where it now hits.
                }
                Err(e) => warn!(expert = id, error = %e, "prefetch failed"),
            }
        });
    }

    /// A speculative prefetch was dropped because no pool buffer could
    /// be acquired (shadow starved even after recycling, or legacy
    /// primary pool busy). Counts it, mirrors to Prometheus, and warns
    /// on the first occurrence so starvation is never silent.
    fn note_prefetch_dropped_pool_starved(&self, id: u32) {
        let prev = self
            .metrics
            .counters
            .prefetch_dropped_pool_starved
            .fetch_add(1, Ordering::Relaxed);
        if let Some(p) = self.metrics.prom.as_ref() {
            p.record_prefetch_dropped_pool_starved(1);
        }
        if prev == 0 {
            warn!(
                expert = id,
                "prefetch dropped: buffer pool starved (further drops counted in \
                 mer_prefetch_dropped_pool_starved_total, logged at debug)"
            );
        } else {
            debug!(expert = id, "skipping prefetch: pool starved");
        }
    }

    /// The predictive controller's expert guess for the current token,
    /// used purely as the `predicted` column of the routing trace
    /// (`--trace-out`). Returns the neural speculator's top-K over the
    /// supplied hidden state — alias-resolved so it lines up with the
    /// `experts` column — or an empty vec when no speculator is wired
    /// (or its `d_model` disagrees with the hidden width). This is a
    /// *read-only* prediction: it never trains the speculator, so
    /// logging the trace cannot perturb the online-SGD accuracy
    /// telemetry.
    fn trace_prediction(&self, hidden: &[f32]) -> Vec<u32> {
        let Some(spec) = self.speculation.speculator.as_ref() else {
            return Vec::new();
        };
        if hidden.len() != spec.d_model() {
            return Vec::new();
        }
        // `speculator_topk` is `>= 1` whenever a speculator is installed
        // (`with_speculator` clamps it), so this yields a non-empty guess
        // on the live path; an empty `predicted` column therefore signals
        // "no speculator wired" rather than "speculator predicted nothing".
        spec.predict_topk(hidden, self.speculation.speculator_topk)
            .into_iter()
            .map(|id| self.resolve_alias(id))
            .collect()
    }

    // -----------------------------------------------------------------
    // Locality / speculator integration helpers.
    //
    // These are called from `generate` and `moe_step` after the gating
    // decision (`target`) is known. They are no-ops when neither
    // monitor is configured, which preserves the legacy code path
    // bit-for-bit.
    // -----------------------------------------------------------------

    /// Effective locality heat threshold for the current id geometry.
    ///
    /// The configured `locality_threshold_pct` ("hot once it appears in
    /// X% of the window") was designed for a *flat* expert namespace.
    /// With layer-qualified global ids the window interleaves every
    /// layer's activations, so a single expert's achievable share of the
    /// window is diluted by the layer count: at 32 layers × top-2 even a
    /// *always-chosen* expert caps out at ~3% of the window and a 10%
    /// threshold is mathematically unreachable — the hot set stays empty
    /// forever (the `hit_rate=0.04%` symptom). Dividing the threshold by
    /// the number of layers restores the intended per-layer semantics:
    /// "hot once it appears in X% of the tokens its layer routed".
    fn effective_locality_threshold(&self) -> f32 {
        let pct = self.speculation.locality_threshold_pct;
        if let Some(per_layer) = self.core.storage.config().num_experts_per_layer {
            if per_layer > 0 {
                let layers = self.core.router.num_experts().div_ceil(per_layer).max(1);
                return pct / layers as f32;
            }
        }
        pct
    }

    /// Whether a Markov-history entry recorded for layer `prev` is a
    /// valid predecessor of the current step at layer `cur` (Finding 5).
    ///
    /// The history ring ([`MarkovRing`], `last` / `last_last`) is
    /// engine-global, but `moe_step` may be driven by several
    /// concurrently-batched token streams. On the layer-qualified
    /// geometry consecutive steps of one stream always advance the
    /// layer by exactly one (wrapping from the last layer back to 0 at
    /// the token boundary), so any entry that *doesn't* satisfy that
    /// contiguity came from a different stream — training on it would
    /// teach the predictor cross-stream noise, and predicting from it
    /// keys the 2nd-order lookup on a junk pair. Layer-less callers
    /// (`cur == None`, the `generate` path) and flat namespaces skip
    /// the check entirely, preserving legacy behaviour bit-for-bit.
    fn markov_layers_contiguous(&self, prev: Option<u32>, cur: Option<u32>) -> bool {
        let Some(per_layer) = self.core.storage.config().num_experts_per_layer else {
            return true;
        };
        if per_layer == 0 {
            return true;
        }
        let Some(cur) = cur else {
            return true;
        };
        let Some(prev) = prev else {
            return false;
        };
        let layers = self.core.router.num_experts().div_ceil(per_layer).max(1);
        cur == prev.wrapping_add(1) || (prev == layers.saturating_sub(1) && cur == 0)
    }

    /// Frequency-based pinning: bump per-expert routing-observation
    /// counts and pin any id that crosses
    /// `options.pin_after_observations` exactly once.
    ///
    /// Lock structure (this is the `route_observations` restructuring
    /// flagged as follow-up in PR #101): the counts live in a sharded
    /// `DashMap<u32, AtomicU64>`, so the steady-state bump for an
    /// already-seen expert takes only a shard **read** lock plus a
    /// relaxed `fetch_add` — concurrent `generate`/`moe_step` calls
    /// from batched requests touch disjoint shards instead of
    /// serializing on one `RwLock<HashMap>` writer guard. The shard
    /// write lock is only taken on the first observation of a given
    /// expert id (entry insertion). `fetch_add`'s returned
    /// previous value makes the threshold crossing exact: precisely
    /// one caller observes `prev + 1 == threshold` and issues the pin.
    fn bump_route_observations(&self, target: &[u32]) {
        self.speculation
            .route_observation_tokens
            .fetch_add(1, Ordering::Release);
        let threshold = self.core.options.pin_after_observations;
        for &id in target {
            let prev = if let Some(counter) = self.speculation.route_observations.get(&id) {
                counter.fetch_add(1, Ordering::Relaxed)
            } else {
                self.speculation
                    .route_observations
                    .entry(id)
                    .or_insert_with(|| AtomicU64::new(0))
                    .fetch_add(1, Ordering::Relaxed)
            };
            if prev + 1 == threshold {
                debug!(expert = id, count = threshold, "pinning hot expert");
                self.core.cache.pin(id);
            }
        }
    }

    /// Observe the chosen expert ids in the locality monitor and
    /// reconcile pinning with the expert cache: ids that just entered
    /// the hot set are pinned (LRU-eviction-protected), ids that just
    /// dropped out are unpinned.
    ///
    /// Also records per-token locality hit/miss telemetry: a chosen
    /// expert is a "locality hit" if it was *already* in the hot set
    /// at the time of routing (i.e. before this token's observation
    /// pushed it in or out). Returns the size of the hot set, useful
    /// for tests.
    fn locality_observe_and_reconcile(&self, target: &[u32]) -> usize {
        let Some(monitor) = self.speculation.locality.as_ref() else {
            return 0;
        };
        // Snapshot pre-observation hit/miss against the *current* hot set.
        let threshold = self.effective_locality_threshold();
        let mut hits: u64 = 0;
        let mut misses: u64 = 0;
        for &id in target {
            if monitor.is_hot(id, threshold) {
                hits += 1;
            } else {
                misses += 1;
            }
        }
        if hits > 0 {
            self.speculation
                .locality_hits
                .fetch_add(hits, Ordering::Relaxed);
        }
        if misses > 0 {
            self.speculation
                .locality_misses
                .fetch_add(misses, Ordering::Relaxed);
        }
        if let Some(m) = &self.metrics.prom {
            m.record_locality(hits, misses);
        }

        // Update the monitor's window with this token's activations.
        monitor.observe(target);

        // Reconcile pin set against the post-observation hot set.
        //
        // **Pin budget (Finding 1).** Pinning every hot id is unsafe:
        // with a low effective threshold the hot set can cover the
        // entire recent working set, and pinning it all saturates the
        // cache — `insert` then rejects every new resident ("every
        // slot pinned"), `evict_lru` returns `None`, and foreground
        // misses serialize on the single reserved pool buffer (the
        // multi-second SSD-stall spikes). Cap pins so every per-layer
        // cache always keeps at least one evictable slot.
        // `hot_set` is sorted hottest-first, so the cap keeps the
        // most valuable ids.
        let ranked = monitor.hot_set(threshold);
        let mut pins_per_layer: HashMap<usize, usize> =
            HashMap::with_capacity(self.core.cache.num_layers());
        let mut new_hot: HashSet<u32> = HashSet::with_capacity(ranked.len());
        for id in ranked {
            let layer = self.core.cache.layer_of(id);
            let budget = self.core.cache.capacity_of_layer(layer).saturating_sub(1);
            let used = pins_per_layer.entry(layer).or_insert(0);
            if *used < budget {
                *used += 1;
                new_hot.insert(id);
            }
        }
        let mut prev = self.speculation.locality_pinned.lock();
        for &id in new_hot.iter() {
            if !prev.contains(&id) {
                self.core.cache.pin(id);
            }
        }
        for &id in prev.iter() {
            if !new_hot.contains(&id) {
                let is_static_pinned = self.speculation.static_pinned.lock().contains(&id);
                if !is_static_pinned {
                    self.core.cache.unpin(id);
                }
            }
        }
        let len = new_hot.len();
        *prev = new_hot;
        len
    }

    /// Run the speculator forward over `x`, compare its top-K to the
    /// gate's actual `target`, record accuracy telemetry, and take one
    /// online SGD step against the actual decision. Returns the
    /// speculator's prediction so the caller can union it into the
    /// prefetch set.
    ///
    /// `layer` is the current MoE layer when known (the `moe_step`
    /// path). With a layer-qualified id geometry the gate's decision is
    /// confined to that layer's slice of the global namespace, so the
    /// prediction is taken from [`NeuralSpeculator::predict_topk_for_layer`]
    /// over the *same slice* — a global arg-max would spread the top-K
    /// across every layer's logits and almost never land in the current
    /// layer (the `accuracy=0.82%` symptom), while also feeding
    /// wrong-layer ids into the union prefetch where they waste shadow
    /// slots. Pass `None` on the layer-less `generate` path.
    fn speculator_predict_and_train(
        &self,
        x: &[f32],
        target: &[u32],
        layer: Option<u32>,
    ) -> Vec<u32> {
        let Some(spec) = self.speculation.speculator.as_ref() else {
            return Vec::new();
        };
        if x.len() != spec.d_model() {
            // Hidden state shape mismatch — nothing useful we can
            // predict against, so disable the M arm for this token.
            // This keeps the speculator graceful in the synthetic
            // benchmark where d_model can disagree with the real
            // model, but the disablement must not be invisible: a
            // persistently mismatched speculator silently zeroes the
            // M arm (and, through the unified score ceiling, the
            // affinity/spatial fold too). Warn once and count every
            // occurrence so operators can see it in /metrics.
            let prev = self
                .metrics
                .counters
                .speculator_dmodel_mismatch
                .fetch_add(1, Ordering::Relaxed);
            if let Some(p) = self.metrics.prom.as_ref() {
                p.record_speculator_disabled(1);
            }
            if prev == 0 {
                warn!(
                    hidden_len = x.len(),
                    speculator_d_model = spec.d_model(),
                    "speculator disabled: hidden-state width != speculator d_model \
                     (M arm contributes nothing; counted in mer_speculator_disabled_total)"
                );
            }
            return Vec::new();
        }
        let preds = match (layer, self.core.storage.config().num_experts_per_layer) {
            (Some(l), Some(per_layer)) if per_layer > 0 => {
                spec.predict_topk_for_layer(x, l, per_layer, self.speculation.speculator_topk)
            }
            _ => spec.predict_topk(x, self.speculation.speculator_topk),
        };
        // Prediction precision@K components: hits are predicted expert
        // IDs contained in the gate's actual top-K, misses are
        // predicted expert IDs outside that top-K.
        let target_set: HashSet<u32> = target.iter().copied().collect();
        let mut hits: u64 = 0;
        for &p in &preds {
            if target_set.contains(&p) {
                hits += 1;
            }
        }
        let misses = preds.len() as u64 - hits;
        if hits > 0 {
            self.speculation
                .spec_hits
                .fetch_add(hits, Ordering::Relaxed);
        }
        if misses > 0 {
            self.speculation
                .spec_misses
                .fetch_add(misses, Ordering::Relaxed);
        }
        // Top-1 accuracy: 1 if the speculator's #1 expert matches the
        // gate's #1 expert for this token, 0 otherwise. This is the
        // counter the design spec calls `mer_speculator_accuracy_total`.
        let top1_match: u64 = match (preds.first(), target.first()) {
            (Some(&p), Some(&t)) if p == t => 1,
            _ => 0,
        };
        if top1_match > 0 {
            self.speculation
                .spec_top1_matches
                .fetch_add(1, Ordering::Relaxed);
        }
        // One token observed by the speculator (regardless of match).
        self.speculation.spec_tokens.fetch_add(1, Ordering::Relaxed);
        if let Some(m) = &self.metrics.prom {
            m.record_speculator(hits, misses);
            m.record_speculator_top1(top1_match);
        }
        // Off-path SGD: queue the (hidden_state, actual_top_k)
        // sample to the speculator's background training worker
        // instead of running the update inline. This keeps the
        // per-token engine step free of model-weight write locks;
        // see `NeuralSpeculator::spawn_training_worker` for the
        // worker's reader-preferring lock policy.
        spec.queue_train(x, target, NeuralSpeculator::DEFAULT_LR);
        preds
    }

    /// **Layer-ahead speculation (Part 1).** While layer `current_layer`
    /// is about to run, ask the neural speculator which experts the
    /// *upcoming* layers in the sliding window
    /// `current_layer + 1 ..= current_layer + pipeline_depth` will most
    /// likely activate and kick off their prefetches now, so the io_uring
    /// reads for those layers are in flight during `L`'s compute. By the
    /// time the router reaches `L+d`, that layer's predicted experts have
    /// had up to `d` layer-computes of head start — enough, at the default
    /// `pipeline_depth = 3`, to bury a ~206 ms cold expert read under
    /// ~231 ms of overlapping SwiGLU compute, turning a blocking SSD stall
    /// into a sub-millisecond cache hit. A windowed (rather than single
    /// farthest-layer) look-ahead is robust to dropped speculative
    /// prefetches: every layer in the pipeline is kept primed, so one
    /// dropped read cannot leave a hole that stalls a later layer.
    ///
    /// The feature fed to M is the residual stream *entering* `L` (the
    /// `x` already on hand), which is increasingly stale for layers
    /// further out in the window. That is acceptable: the speculator is
    /// only a prefetch hint, exactly the staleness
    /// `speculator_predict_and_train` already tolerates. Because deeper
    /// predictions are staler (and therefore lower-confidence), the
    /// per-layer fanout is **tapered with distance** — full
    /// `speculator_topk` at `L+1`, narrower further out — so low-value
    /// far-layer guesses don't flood the SSD bandwidth the near layers
    /// depend on. Nearer layers are also issued first, so they win the
    /// shadow buffers under contention.
    ///
    /// No-op (and zero added latency) when the speculator is absent, the
    /// hidden width disagrees, or the layer-count geometry is unknown
    /// (`num_experts_per_layer` not configured). Layers past the last one
    /// yield no predictions (`predict_topk_for_layer` returns empty) and
    /// are skipped. Predicted ids draw from the shadow (Buffer B) pool
    /// like every other speculative prefetch, so a wrong guess can never
    /// steal a buffer from a real miss.
    fn speculate_layer_ahead(self: &Arc<Self>, x: &[f32], current_layer: u32) {
        let Some(spec) = self.speculation.speculator.as_ref() else {
            return;
        };
        if x.len() != spec.d_model() {
            return;
        }
        let Some(per_layer) = self.core.storage.config().num_experts_per_layer else {
            // Without a layer-qualified id geometry we cannot restrict
            // the speculator's global output head to the next layer's
            // slice, so layer-ahead prediction is disabled.
            return;
        };
        let depth = self.speculation.pipeline_depth.max(1);
        let base_k = self.speculation.speculator_topk;
        // Walk the look-ahead window nearest-first so the most valuable
        // (least stale) layers acquire shadow buffers before the deeper,
        // lower-confidence ones under contention.
        for distance in 1..=depth {
            let Some(next_layer) = current_layer.checked_add(distance) else {
                break;
            };
            // Taper the fanout with distance: full `speculator_topk` at
            // `L+1`, then `topk / distance` (at least 1) further out. This
            // keeps the SSD bandwidth focused on the high-confidence near
            // layers rather than flooding it with stale far-layer guesses.
            let k = (base_k / distance as usize).max(1);
            let preds = spec.predict_topk_for_layer(x, next_layer, per_layer, k);
            // A past-the-last-layer index yields no predictions; the
            // remaining (even deeper) layers can only be emptier, so stop.
            if preds.is_empty() {
                break;
            }
            // Confidence tag decays with distance, mirroring the taper —
            // surfaced in the prefetch-complete debug log.
            let prob = 0.5 / distance as f64;
            for id in preds {
                let canon = self.resolve_alias(id);
                if !self.core.cache.contains(canon) {
                    // The shadow-pool bound and prefetch semaphore keep
                    // this windowed look-ahead from over-committing.
                    self.spawn_prefetch(canon, prob);
                }
            }
        }
    }

    /// Tier 3 — drive the per-layer pre-gate. Records this layer's
    /// routing transition into the conditional map and prefetches the
    /// experts it predicts for the *next* layer. A disabled pre-gate
    /// (`None`) makes this a no-op, preserving legacy behaviour.
    fn pregate_prefetch(self: &Arc<Self>, layer: u32, target: &[u32]) {
        let Some(pregate) = self.speculation.pregate.as_ref() else {
            return;
        };
        let predicted = pregate.observe_and_predict(layer, target);
        for id in predicted {
            // Resolve aliases defensively (idempotent) so deduplicated
            // experts share the canonical resident copy, then issue the
            // speculative read with the pre-gate's high confidence tag.
            let id = self.resolve_alias(id);
            self.spawn_prefetch(id, crate::pregate::PREGATE_PREFETCH_PROB);
        }
    }

    /// Prefetch every id in the union `S ∪ L ∪ M` (plus the optional
    /// affinity/spatial neighbour fold) that isn't already resident —
    /// the **speculative I/O union-fetch** described in the design spec.
    /// `s_markov` is the predictor's Markov-chain top-K (already
    /// prob-ranked), `m_speculator` is the neural speculator's top-K,
    /// `already_in_flight` dedupes against ids the caller already kicked
    /// off via the regular cache-miss path, and `layer` is the current
    /// MoE layer (when known) used to scope the per-layer affinity fold.
    ///
    /// The three headline arms are fused with the **canonical unified
    /// weights** (`0.33·markov + 0.25·locality + 0.42·speculator`) via
    /// [`PredictiveLoader::combine_unified_arms`] — the same scoring the
    /// offline [`PredictiveLoader::predict_unified`] API exposes — so the
    /// documented prioritisation (speculator > Markov > locality) drives
    /// the prefetch ranking and the truncation to the shadow budget,
    /// rather than the previous flat `p = 0.5` tag. The speculator top-K
    /// is passed in precomputed (the engine already ran and trained the
    /// speculator once this token), so no second forward pass is issued.
    fn union_prefetch(
        self: &Arc<Self>,
        s_markov: &[(u32, f64)],
        m_speculator: &[u32],
        already_in_flight: &HashSet<u32>,
        layer: Option<u32>,
    ) {
        // Locality (L) arm — the monitor's current hot set, or empty.
        let locality_ids: Vec<u32> = self
            .speculation
            .locality
            .as_ref()
            .map(|m| m.hot_set(self.effective_locality_threshold()))
            .unwrap_or_default();

        // If expert aliasing is enabled, canonicalize ids *before* scoring so
        // evidence isn't split across aliases and neighbour folds operate on
        // the same ids the cache ultimately uses.
        let mut scored = if self.speculation.alias_map.is_some() {
            // Canonicalize + dedupe flat-weight arms after alias
            // resolution, **preserving each arm's ranking** (heat order
            // for locality, logit order for the speculator) — the
            // combiner's per-rank tie-break decay depends on it. On an
            // alias collision the first (higher-ranked) id wins.
            let mut seen_loc: HashSet<u32> = HashSet::with_capacity(locality_ids.len());
            let locality_ids: Vec<u32> = locality_ids
                .iter()
                .map(|&id| self.resolve_alias(id))
                .filter(|&id| seen_loc.insert(id))
                .collect();

            let mut seen_spec: HashSet<u32> = HashSet::with_capacity(m_speculator.len());
            let speculator_ids: Vec<u32> = m_speculator
                .iter()
                .map(|&id| self.resolve_alias(id))
                .filter(|&id| seen_spec.insert(id))
                .collect();

            // Canonicalize Markov ids, keeping the max probability when multiple ids
            // map to the same canonical expert.
            let mut markov: HashMap<u32, f64> = HashMap::new();
            for &(id, p) in s_markov {
                let canon = self.resolve_alias(id);
                markov
                    .entry(canon)
                    .and_modify(|cur| *cur = cur.max(p))
                    .or_insert(p);
            }
            let markov: Vec<(u32, f64)> = markov.into_iter().collect();
            self.core
                .predictor
                .combine_unified_arms(&markov, &locality_ids, &speculator_ids)
        } else {
            self.core
                .predictor
                .combine_unified_arms(s_markov, &locality_ids, m_speculator)
        };
        // Optional affinity + spatial neighbour fold: for every
        // high-confidence seed, pull its top co-fired (per-layer
        // affinity) and disk-adjacent (UTH spatial) neighbours into the
        // prefetch set. Gated on the affinity arm being installed *and*
        // a layer-qualified id geometry being available.
        if let Some(affinity) = self.speculation.affinity.as_ref() {
            // `layer` is only `Some` on the `moe_step` path, where the
            // current MoE layer is known — the affinity fold is scoped
            // per-layer, so skip it on the layer-less `generate` path.
            if layer.is_some() {
                if let Some(per_layer) = self.core.storage.config().num_experts_per_layer {
                    if per_layer > 0 {
                        scored = self.fold_affinity_spatial(scored, affinity, per_layer);
                    }
                }
            }
        }
        // Resolve aliases, drop residents, and dedupe against ids
        // already in flight — preserving the descending-score order from
        // the fuse/fold above (we only ever skip ids, never reorder). On
        // an alias collision the first (higher-scored) id wins.
        let mut seen: HashSet<u32> = already_in_flight.clone();
        let mut candidates: Vec<(u32, f64)> = Vec::with_capacity(scored.len());
        for (id, score) in scored {
            let canon = self.resolve_alias(id);
            if self.core.cache.contains(canon) {
                continue;
            }
            if seen.insert(canon) {
                candidates.push((canon, score as f64));
            }
        }
        // Truncate to the shadow-slot budget: in-flight speculation can
        // never exceed Buffer B's capacity, so anything past that would
        // be dropped by `spawn_prefetch`'s `try_acquire_shadow` anyway.
        // Truncating here keeps the *best* ids instead of letting
        // arbitrary spawn ordering decide which survive. A zero shadow
        // capacity means the legacy single-pool layout, where the
        // semaphore alone bounds concurrency — leave the list intact.
        let budget = self.core.pool.shadow_capacity();
        if budget > 0 && candidates.len() > budget {
            candidates.truncate(budget);
        }
        for (canon, p) in candidates {
            self.spawn_prefetch(canon, p);
        }
    }

    /// Fold the **affinity** (per-layer co-occurrence) and **spatial**
    /// (UTH disk-adjacency) neighbour arms onto an already-scored
    /// candidate list, mirroring
    /// [`PredictiveLoader::fold_spatial_affinity`] but in the engine's
    /// *global* id namespace.
    ///
    /// Spatial neighbours use the global namespace directly (expert
    /// `g ± 1` is the disk-adjacent record). Affinity is per-layer, so a
    /// seed is split into `(seed_layer, local)`, its co-fired neighbours
    /// are looked up in `seed_layer`'s matrix, and each local neighbour
    /// is mapped back to its global id. Only seeds scoring at least
    /// [`crate::router::SPATIAL_CONFIDENCE_THRESHOLD`] contribute.
    fn fold_affinity_spatial(
        &self,
        base: Vec<(u32, f32)>,
        affinity: &LayeredExpertAffinity,
        per_layer: u32,
    ) -> Vec<(u32, f32)> {
        use crate::router::{
            spatial_neighbors, SPATIAL_CONFIDENCE_THRESHOLD, W_AFFINITY, W_SPATIAL,
        };
        let seeds: Vec<u32> = base
            .iter()
            .filter(|(_, s)| *s >= SPATIAL_CONFIDENCE_THRESHOLD)
            .map(|(id, _)| *id)
            .collect();
        if seeds.is_empty() {
            return base;
        }
        let global_n = self.core.router.num_experts();
        let k = self.speculation.affinity_neighbors_k.max(1);
        let mut combined: HashMap<u32, f32> = base.into_iter().collect();
        for &seed in &seeds {
            // Spatial: global disk adjacency.
            for nbr in spatial_neighbors(seed, global_n, 2) {
                *combined.entry(nbr).or_insert(0.0) += W_SPATIAL;
            }
            // Affinity: co-occurrence within the seed's own layer.
            let (seed_layer, local) = global_to_layer_local(seed, per_layer);
            for local_nbr in affinity.neighbors(seed_layer as usize, local, k) {
                let global_nbr = layer_local_to_global(seed_layer, local_nbr, per_layer);
                if global_nbr < global_n {
                    *combined.entry(global_nbr).or_insert(0.0) += W_AFFINITY;
                }
            }
        }
        let mut out: Vec<(u32, f32)> = combined.into_iter().filter(|&(_, p)| p > 0.0).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// Snapshot of the engine's predictive-architecture telemetry. The
    /// returned ratios are in `[0, 1]`; both fall back to `0.0` when no
    /// observations have been recorded yet (the safer default for a
    /// freshly-warmed engine).
    pub fn predictive_telemetry(&self) -> PredictiveTelemetry {
        let s_hits = self.speculation.spec_hits.load(Ordering::Relaxed);
        let s_misses = self.speculation.spec_misses.load(Ordering::Relaxed);
        let s_top1 = self.speculation.spec_top1_matches.load(Ordering::Relaxed);
        let s_top1_total = self.speculation.spec_tokens.load(Ordering::Relaxed);
        let l_hits = self.speculation.locality_hits.load(Ordering::Relaxed);
        let l_misses = self.speculation.locality_misses.load(Ordering::Relaxed);
        let s_total = s_hits + s_misses;
        let l_total = l_hits + l_misses;
        PredictiveTelemetry {
            speculator_hits: s_hits,
            speculator_misses: s_misses,
            speculator_accuracy: if s_total == 0 {
                0.0
            } else {
                s_hits as f64 / s_total as f64
            },
            speculator_top1_matches: s_top1,
            speculator_top1_total: s_top1_total,
            speculator_top1_accuracy: if s_top1_total == 0 {
                0.0
            } else {
                s_top1 as f64 / s_top1_total as f64
            },
            locality_hits: l_hits,
            locality_misses: l_misses,
            locality_hit_rate: if l_total == 0 {
                0.0
            } else {
                l_hits as f64 / l_total as f64
            },
            ssd_stall_us: self.metrics.total_ssd_stall_us.load(Ordering::Relaxed),
        }
    }

    /// **Real-transformer MoE step.** Given a hidden state `x` and the
    /// expert ids the gating network selected for it, ensure every chosen
    /// expert is resident in the SSD-streaming cache (concurrent
    /// `pread(2)` for the misses, exactly as `generate` does), run each
    /// expert's SwiGLU FFN over `x`, and return the per-expert output
    /// vectors aligned with the input `experts` slice.
    ///
    /// This is the bridge from the dense `TransformerLayer` code (which
    /// produces a routing decision) to the MoE compute (which the
    /// SSD-streaming substrate makes interesting). The caller — typically
    /// `crate::model::RealModel::step` — then folds the returned vectors
    /// back into the residual stream via `TransformerLayer::moe_combine`.
    ///
    /// The same hits / misses / bytes / latency counters that
    /// `Engine::generate` updates are bumped here too, so
    /// `engine.print_summary()` shows the same shape regardless of
    /// whether the engine is driving the benchmark Markov path or a real
    /// transformer.
    ///
    /// `token_idx` is used only as a digest seed for `InferenceOutput`;
    /// it has no effect on the activation produced.
    /// Whether the development-only degraded mode
    /// (`[real_transformer] allow_degraded_experts`) is active. The
    /// real-model forward pass consults this for its fail-closed
    /// numerical checks (Part A3) in addition to the engine's own
    /// expert-failure handling (Part A1).
    pub fn allow_degraded_experts(&self) -> bool {
        self.core.options.policy.allow_degraded_experts
    }

    /// Whether the development-only uniform fallback for non-finite
    /// attention rows (`[real_transformer]
    /// allow_nonfinite_attention_fallback`) is active. Independent of
    /// [`Self::allow_degraded_experts`] (hardening pass, policy
    /// separation): degraded experts never enable fabricated attention.
    pub fn allow_nonfinite_attention_fallback(&self) -> bool {
        self.core.options.policy.allow_nonfinite_attention_fallback
    }

    /// The engine-scoped fail-open policy set (all default `false`).
    pub fn inference_policy(&self) -> crate::inference::RealInferencePolicy {
        self.core.options.policy
    }

    /// Record one degraded-mode zero substitution of a required expert
    /// contribution (routed, shared, or dense FFN). Only ever called
    /// when `allow_degraded_experts` is active; the counter marks every
    /// metric and benchmark figure of the run as non-authoritative.
    pub fn record_degraded_expert_substitution(&self) {
        self.metrics
            .counters
            .degraded_expert_substitutions
            .fetch_add(1, Ordering::Relaxed);
    }

    pub async fn moe_step(
        self: &Arc<Self>,
        token_idx: u64,
        layer: u32,
        x: &HiddenState,
        experts: &[u32],
    ) -> Result<Vec<HiddenState>, MoeStepError> {
        self.moe_step_with_timing(token_idx, layer, x, experts, None)
            .await
    }

    fn resolved_expert_execution_policy(&self, selected_experts: usize) -> ExpertExecutionPolicy {
        match self.core.options.expert_execution_policy {
            ExpertExecutionPolicy::Auto => {
                let threads = crate::parallel::num_threads();
                if self.core.options.dtype == WeightDtype::Q8_0
                    && selected_experts >= 4
                    && threads >= selected_experts
                    && self.core.shape.d_model.is_multiple_of(Q8_0_BLOCK_ELEMS)
                    && self.core.shape.d_ff.is_multiple_of(Q8_0_BLOCK_ELEMS)
                    && !crate::parallel::in_rayon_worker()
                {
                    ExpertExecutionPolicy::ParallelExpertsSingleThread
                } else {
                    ExpertExecutionPolicy::SequentialExpertsRowParallel
                }
            }
            policy => policy,
        }
    }

    fn forward_moe_resident(
        &self,
        token_idx: u64,
        layer: u32,
        r: &ExpertResident,
        x: &HiddenState,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> Result<HiddenState, MoeStepError> {
        // The authoritative plan decides the execution plane. CPU plans never
        // invoke the GPU boundary; GPU plans must produce GPU output or take
        // the explicitly configured strict-vs-serving branch.
        if self.execution_context().plan().routed_experts()
            == crate::backend::ExecutionPlane::Gpu
        {
            self.metrics
                .counters
                .gpu_dispatch_attempts
                .fetch_add(1, Ordering::Relaxed);
            let backend = self.routed_expert_backend();
            let mut out_f16 = vec![half::f16::ZERO; self.core.shape.d_model];
            let x_f16: Vec<half::f16> = x.iter().map(|&f| half::f16::from_f32(f)).collect();
            let x_view = crate::backend::TensorView {
                data: &x_f16,
                rows: 1,
                cols: self.core.shape.d_model,
            };
            let mut out_view = crate::backend::TensorViewMut {
                data: &mut out_f16,
                rows: 1,
                cols: self.core.shape.d_model,
            };
            let matmul_res = if !backend.is_gpu() {
                Err(crate::backend::GpuExpertDispatchError::new(
                    layer,
                    r.id,
                    crate::backend::GpuExpertDispatchErrorKind::RuntimeInvariant,
                    "authoritative GPU routed-expert plan resolved to a non-GPU backend",
                ))
            } else if !self.gpu_eligible_dtype() {
                Err(crate::backend::GpuExpertDispatchError::new(
                    layer,
                    r.id,
                    crate::backend::GpuExpertDispatchErrorKind::RuntimeInvariant,
                    format!(
                        "authoritative GPU plan is incompatible with dtype {:?}, d_model={}, d_ff={}",
                        self.core.options.dtype, self.core.shape.d_model, self.core.shape.d_ff
                    ),
                ))
            } else if let Err(source) = self.demand_admit_resident_to_gpu(layer, r) {
                Err(source)
            } else {
                backend.routed_expert_matmul(
                    layer,
                    r.id,
                    x_view,
                    self.core.shape.d_model,
                    self.core.shape.d_ff,
                    &mut out_view,
                )
            };
            match matmul_res {
                Ok(()) => {
                    self.metrics
                        .counters
                        .gpu_dispatch_successes
                        .fetch_add(1, Ordering::Relaxed);
                    return Ok(out_f16.iter().map(|h| h.to_f32()).collect::<Vec<f32>>());
                }
                Err(source)
                    if self.core.routed_expert_gpu_failure_policy
                        == RoutedExpertGpuFailurePolicy::StrictFailClosed =>
                {
                    self.metrics
                        .counters
                        .gpu_dispatch_failures
                        .fetch_add(1, Ordering::Relaxed);
                    return Err(MoeStepError::GpuExpertDispatch { source });
                }
                Err(source) => {
                    self.metrics
                        .counters
                        .gpu_dispatch_failures
                        .fetch_add(1, Ordering::Relaxed);
                    // Only an actual serving-mode CPU recovery increments the
                    // fallback counter.
                    let prev = self
                        .metrics
                        .counters
                        .gpu_cpu_fallbacks
                        .fetch_add(1, Ordering::Relaxed);
                    if let Some(p) = self.metrics.prom.as_ref() {
                        p.record_gpu_cpu_fallback(1);
                    }
                    if prev == 0 {
                        warn!(
                            expert = r.id,
                            error = %source,
                            "GPU expert dispatch fell back to CPU \
                             (further fallbacks counted in mer_gpu_cpu_fallbacks_total)"
                        );
                    }
                }
            }
        }

        self.forward_moe_resident_cpu(token_idx, layer, r, x, timings)
    }

    fn forward_moe_resident_cpu(
        &self,
        token_idx: u64,
        layer: u32,
        r: &ExpertResident,
        x: &HiddenState,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> Result<HiddenState, MoeStepError> {
        self.metrics
            .counters
            .cpu_routed_expert_dispatches
            .fetch_add(1, Ordering::Relaxed);

        #[cfg(test)]
        self.metrics
            .counters
            .cpu_expert_forward_calls
            .fetch_add(1, Ordering::Relaxed);

        if self
            .diagnostic_cpu_q4_boundary_emulation
            .load(Ordering::Acquire)
        {
            let expected = crate::inference::expert_weight_bytes_for(
                self.core.shape.d_model,
                self.core.shape.d_ff,
                WeightDtype::Q4_0,
            );
            if r.data().len() < expected || r.data()[expected..].iter().any(|byte| *byte != 0) {
                return Err(MoeStepError::ExpertCompute {
                    layer,
                    expert: r.id,
                    source: ExpertWeightsError::InvalidLayout(format!(
                        "diagnostic Q4 payload has {} bytes; expected {expected} canonical bytes followed only by zero alignment padding",
                        r.data().len()
                    )),
                });
            }
            let effective_input = crate::numerical_diagnostics::round_trip_f16_values(x)
                .map_err(|source| MoeStepError::ExpertCompute {
                    layer,
                    expert: r.id,
                    source,
                })?;
            let cpu = crate::inference::q4_0_cpu_reference_forward(
                &r.data()[..expected],
                &effective_input,
                self.core.shape.d_model,
                self.core.shape.d_ff,
            )
            .map_err(|source| MoeStepError::ExpertCompute {
                layer,
                expert: r.id,
                source,
            })?;
            let output = crate::numerical_diagnostics::round_trip_f16_values(&cpu).map_err(
                |source| MoeStepError::ExpertCompute {
                    layer,
                    expert: r.id,
                    source,
                },
            )?;
            self.diagnostic_cpu_q4_boundary_emulated_dispatches
                .fetch_add(1, Ordering::Release);
            return Ok(output);
        }

        dispatch_expert_forward(
            self.core.options.dtype,
            self.core.options.use_qmm_for_q4,
            token_idx,
            r,
            x,
            self.core.shape.d_model,
            self.core.shape.d_ff,
            self.core.options.policy.expert_size_tolerance(),
            timings,
        )
        .map(|(_out, y)| y)
        .map_err(|source| MoeStepError::ExpertCompute {
            layer,
            expert: r.id,
            source,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn try_forward_moe_q4_layer_batch(
        &self,
        token_idx: u64,
        layer: u32,
        expert_ids: &[u32],
        residents: &[Option<Arc<ExpertResident>>],
        x: &HiddenState,
        weights: &[f32],
        out: &mut Vec<f32>,
        timings: Option<&crate::stage_timing::StageTimings>,
        allow_degraded: bool,
    ) -> Result<bool, MoeStepError> {
        if self.core.options.dtype != WeightDtype::Q4_0
            || expert_ids.len() != 8
            || expert_ids.len() != residents.len()
            || expert_ids.len() != weights.len()
            || self.execution_context().plan().routed_experts()
                != crate::backend::ExecutionPlane::Gpu
            || !self.routed_expert_backend().is_gpu()
            || !self.gpu_eligible_dtype()
            || residents.iter().any(Option::is_none)
        {
            return Ok(false);
        }
        for resident in residents {
            let resident = resident.as_deref().expect("batch eligibility checked residents");
            if self.demand_admit_resident_to_gpu(layer, resident).is_err() {
                return Ok(false);
            }
        }
        let Some(output_len) = expert_ids.len().checked_mul(self.core.shape.d_model) else {
            return Ok(false);
        };
        let x_f16: Vec<half::f16> = x.iter().map(|value| half::f16::from_f32(*value)).collect();
        let x_view = crate::backend::TensorView {
            data: &x_f16,
            rows: 1,
            cols: self.core.shape.d_model,
        };
        let mut outputs = vec![half::f16::ZERO; output_len];
        let outcome = self.routed_expert_backend().routed_expert_matmul_batch_q4(
            layer,
            expert_ids,
            x_view,
            self.core.shape.d_model,
            self.core.shape.d_ff,
            &mut outputs,
        );
        let logical_dispatches = expert_ids.len() as u64;
        if matches!(&outcome, crate::backend::GpuExpertBatchDispatchOutcome::Ineligible) {
            return Ok(false);
        }
        self.metrics
            .counters
            .gpu_dispatch_attempts
            .fetch_add(logical_dispatches, Ordering::Relaxed);
        let source = match outcome {
            crate::backend::GpuExpertBatchDispatchOutcome::Ineligible => unreachable!(),
            crate::backend::GpuExpertBatchDispatchOutcome::Completed => {
                if accumulate_ordered_f16_outputs(
                    &outputs,
                    weights,
                    self.core.shape.d_model,
                    out,
                ) {
                    self.metrics
                        .counters
                        .gpu_dispatch_successes
                        .fetch_add(logical_dispatches, Ordering::Relaxed);
                    return Ok(true);
                }
                crate::backend::GpuExpertDispatchError::new(
                    layer,
                    expert_ids[0],
                    crate::backend::GpuExpertDispatchErrorKind::RuntimeInvariant,
                    "completed Q4 layer batch returned malformed output geometry",
                )
            }
            crate::backend::GpuExpertBatchDispatchOutcome::Failed(source) => source,
        };
        self.metrics
            .counters
            .gpu_dispatch_failures
            .fetch_add(logical_dispatches, Ordering::Relaxed);
        if self.core.routed_expert_gpu_failure_policy
            == RoutedExpertGpuFailurePolicy::StrictFailClosed
        {
            return Err(MoeStepError::GpuExpertDispatch { source });
        }

        let previous_fallbacks = self
            .metrics
            .counters
            .gpu_cpu_fallbacks
            .fetch_add(logical_dispatches, Ordering::Relaxed);
        if let Some(prom) = self.metrics.prom.as_ref() {
            prom.record_gpu_cpu_fallback(logical_dispatches);
        }
        if previous_fallbacks == 0 {
            warn!(
                layer,
                experts = expert_ids.len(),
                error = %source,
                "GPU Q4 layer batch fell back to ordered CPU expert execution"
            );
        }
        out.clear();
        out.resize(self.core.shape.d_model, 0.0);
        for (slot, resident) in residents.iter().enumerate() {
            let resident = resident.as_deref().expect("batch eligibility checked residents");
            match self.forward_moe_resident_cpu(token_idx, layer, resident, x, timings) {
                Ok(values) => {
                    let weight = weights[slot];
                    if weight != 0.0 {
                        for (dst, value) in out.iter_mut().zip(values) {
                            *dst += weight * value;
                        }
                    }
                }
                Err(error) if allow_degraded => {
                    self.metrics
                        .counters
                        .degraded_expert_substitutions
                        .fetch_add(1, Ordering::Relaxed);
                    warn!(
                        token = token_idx,
                        layer,
                        expert = resident.id,
                        error = %error,
                        "DEGRADED MODE: batch CPU fallback failed; dropping contribution"
                    );
                }
                Err(error) => return Err(error),
            }
        }
        Ok(true)
    }

    pub async fn moe_step_with_timing(
        self: &Arc<Self>,
        token_idx: u64,
        layer: u32,
        x: &HiddenState,
        experts: &[u32],
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> Result<Vec<HiddenState>, MoeStepError> {
        match self
            .moe_step_inner(
                token_idx,
                layer,
                x,
                experts,
                MoeStepOutputMode::PerExpert,
                timings,
            )
            .await?
        {
            MoeStepResult::PerExpert(outputs) => Ok(outputs),
            MoeStepResult::WeightedInto => unreachable!("per-expert moe_step requested"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn moe_step_weighted_into_with_timing(
        self: &Arc<Self>,
        token_idx: u64,
        layer: u32,
        x: &HiddenState,
        experts: &[u32],
        weights: &[f32],
        out: &mut Vec<f32>,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> Result<(), MoeStepError> {
        assert_eq!(
            experts.len(),
            weights.len(),
            "moe_step_weighted_into_with_timing: experts and weights must align"
        );
        self.capture_layer0_route_if_armed(token_idx, layer, x, experts, weights);
        match self
            .moe_step_inner(
                token_idx,
                layer,
                x,
                experts,
                MoeStepOutputMode::WeightedInto { weights, out },
                timings,
            )
            .await?
        {
            MoeStepResult::WeightedInto => Ok(()),
            MoeStepResult::PerExpert(_) => unreachable!("weighted moe_step requested"),
        }
    }

    async fn moe_step_inner<'a>(
        self: &Arc<Self>,
        token_idx: u64,
        layer: u32,
        x: &HiddenState,
        experts: &[u32],
        output_mode: MoeStepOutputMode<'a>,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> Result<MoeStepResult, MoeStepError> {
        let cycle_start = Instant::now();
        // Tier 4: fold the previous step's prefetch precision window into
        // the governor's EWMA before issuing this step's speculation.
        // No-op when the governor is disabled.
        self.core.governor.refresh();
        // Resolve aliases up front so the cache + predictor only ever
        // see canonical expert ids (mirrors `generate`).
        let target: Vec<u32> = experts.iter().map(|&id| self.resolve_alias(id)).collect();
        self.metrics
            .counters
            .selected_routed_experts
            .fetch_add(target.len() as u64, Ordering::Relaxed);

        // Locality monitor: observe and reconcile pinning. Same
        // semantics as in `generate`.
        self.locality_observe_and_reconcile(&target);

        // Affinity arm: record which experts the gate co-activated in
        // *this* layer. The matrix is per-layer in the local id
        // namespace, so map the global ids back to their layer-local
        // index before observing. No-op unless the affinity arm is
        // installed and the model exposes a layer-qualified geometry.
        if let Some(affinity) = self.speculation.affinity.as_ref() {
            if let Some(per_layer) = self.core.storage.config().num_experts_per_layer {
                if per_layer > 0 {
                    let locals: Vec<u32> = target
                        .iter()
                        .map(|&g| global_to_layer_local(g, per_layer).1)
                        .collect();
                    affinity.observe_layer(layer as usize, &locals);
                }
            }
        }

        // Speculator: predict against the *real* hidden state (this is
        // the path where d_model matches by construction) and train
        // online against the gate's actual top-K decision.
        let m_speculator = self.speculator_predict_and_train(x, &target, Some(layer));

        // Frequency-based pinning: same logic as `generate`. Also bump
        // (without pinning) for an online static-residency controller or
        // when collecting a popularity profile for export.
        if self.core.options.pin_after_observations > 0
            || self.core.options.collect_route_profile
            || self.static_residency_needs_counts()
        {
            self.bump_route_observations(&target);
        }
        // Tier 1: pin the static-residency hot set once it is ready.
        self.maybe_apply_static_residency();

        // **Speculative I/O union-fetch (S ∪ L ∪ M), issued
        // concurrently with the target-miss fetches.** Fire the
        // predictor's 2nd-order Markov-chain hint and union it with
        // the locality hot set and the speculator's top-K so all
        // three arms compete for cache slots while the SSD is *also*
        // pulling the experts the gate just chose. The prefetch
        // tasks are spawned here (they only depend on the cache /
        // storage Arcs) and *do not block* the await on
        // `miss_handles` below — the OS / io_uring queue interleaves
        // both sets of reads. This is the change called out in the
        // design spec under Task 2: the predictive-controller's union
        // prefetch must overlap the critical-path SSD stall, not run
        // sequentially after it.
        //
        // We read `last_experts` here — which still holds the *previous*
        // step's target set, because the history ring-buffer update
        // happens after compute below — so the 2nd-order lookup key is
        // `(prev, current)`, matching the `(prev_prev, prev) -> next`
        // transitions the predictor was trained on via `observe_step2`
        // (and matching `generate`, which performs the same lookup
        // *after* shifting the ring buffer).
        if let Some(&seed) = target.last() {
            let ring = self.speculation.markov_ring.lock();
            // 2nd-order lookup only when the history entry really is
            // the previous layer of this stream (see
            // `markov_layers_contiguous`); otherwise fall back to the
            // 1st-order row keyed on the current seed alone.
            let contiguous = self.markov_layers_contiguous(ring.last.layer, Some(layer));
            let s_markov = match ring.last.ids.last() {
                Some(&pp) if contiguous => self.core.predictor.predict_next2(pp, seed),
                _ => self.core.predictor.predict_next(seed),
            };
            drop(ring);
            // The gate's own targets are *not* speculative: the miss
            // loop below fetches them into primary (Buffer A) buffers
            // microseconds from now. Passing them as
            // `already_in_flight` keeps the (heavily overlapping,
            // layer-scoped) speculator arm from re-fetching them into
            // scarce shadow slots — and from winning the singleflight
            // slot so the foreground miss lands in a shadow buffer.
            let in_flight: HashSet<u32> = target.iter().copied().collect();
            self.union_prefetch(&s_markov, &m_speculator, &in_flight, Some(layer));
        }

        // **Layer-ahead look-ahead (Part 1).** Independently of the
        // current layer's union prefetch above, predict the *next*
        // layer's experts from the residual entering this layer and
        // submit their reads now, so they overlap this layer's compute
        // and the next layer finds them already resident.
        self.speculate_layer_ahead(x, layer);

        // **Tier 3 pre-gate look-ahead.** Record this layer's routing
        // transition and prefetch the predicted next-layer experts — a
        // high-precision signal conditioned on the previous layer's
        // *actual* routing, complementing the hidden-state speculation
        // above. No-op when the pre-gate is disabled.
        self.pregate_prefetch(layer, &target);

        // Concurrent miss fetches; hits resolved inline.
        let io_wait_start = Instant::now();
        let mut residents: Vec<Option<Arc<ExpertResident>>> = vec![None; target.len()];
        let mut miss_handles: Vec<(
            usize,
            tokio::task::JoinHandle<Result<Arc<ExpertResident>, ExpertReadError>>,
        )> = Vec::new();
        let mut cache_hits_per_expert: Vec<bool> = Vec::with_capacity(target.len());
        // Logical GPU-admission tier — aggregate hits/misses across this routing
        // decision and record once, rather than incrementing Prometheus
        // counters per activation on the hot path.
        let mut gpu_hits_acc: u64 = 0;
        let mut gpu_misses_acc: u64 = 0;
        let cache_lookup_start = Instant::now();
        for (i, &id) in target.iter().enumerate() {
            if let Some(gpu) = self.core.gpu_cache.as_ref() {
                let lookup = gpu.get(id);
                if lookup.is_hit() {
                    gpu_hits_acc += 1;
                } else {
                    gpu_misses_acc += 1;
                }
            }
            if let Some(r) = self.core.cache.get(id) {
                self.metrics.counters.hits.fetch_add(1, Ordering::Relaxed);
                let new_hits = r.record_hit();
                // Tier 4 precision feedback (mirrors `generate`): a
                // first hit on a shadow-backed resident is a prefetch
                // that paid off. No-op when the governor is disabled.
                self.credit_prefetch_use(&r, new_hits);
                if self.background_tasks.accepts_work() {
                    if let (Some(gpu), Some(tx)) = (
                        self.core.gpu_cache.as_ref(),
                        self.core.gpu_promotion_tx.as_ref(),
                    ) {
                        // One outstanding claim prevents queue flooding while
                        // allowing a new request after logical eviction, even
                        // though the RAM resident's hit count no longer has a
                        // fresh threshold-crossing edge.
                        if gpu.claim_promotion(id, new_hits)
                            && tx.send((id, r.clone())).is_err()
                        {
                            gpu.cancel_promotion(id);
                        }
                    }
                }
                residents[i] = Some(r);
                cache_hits_per_expert.push(true);
            } else {
                self.metrics.counters.misses.fetch_add(1, Ordering::Relaxed);
                let me = self.clone();
                miss_handles.push((
                    i,
                    tokio::spawn(async move { me.fetch_with_retry(id).await }),
                ));
                cache_hits_per_expert.push(false);
            }
        }
        crate::stage_timing::record_optional(
            timings,
            crate::stage_timing::EXPERT_CACHE_LOOKUP,
            cache_lookup_start.elapsed(),
        );
        // Aggregate logical GPU-admission outcome for this routing decision.
        if let Some(p) = self.metrics.prom.as_ref() {
            if gpu_hits_acc > 0 || gpu_misses_acc > 0 {
                p.record_gpu_cache(gpu_hits_acc, gpu_misses_acc);
            }
        }
        // Emit one routing-trace record per `moe_step` call — same
        // contract as `generate`, but with the real per-layer index
        // supplied by the caller. This is what makes `--trace-out`
        // useful for the `--gate-weights` and real-transformer paths
        // (which go through `moe_step`, not `generate`). `m_speculator`
        // is the speculator's top-K prediction already computed (and
        // trained) above this token, so we reuse it as the `predicted`
        // column rather than running a second forward.
        if let Some(tw) = self.metrics.trace_writer.read().as_ref() {
            let predicted: Vec<u32> = m_speculator
                .iter()
                .map(|&id| self.resolve_alias(id))
                .collect();
            tw.write_record(
                token_idx,
                layer,
                &target,
                &cache_hits_per_expert,
                &predicted,
            );
        }
        let had_misses = !miss_handles.is_empty();
        // Strict production mode (the default): a routed expert whose
        // fetch failed after retries fails the whole step — the caller
        // maps this to a request-level error. Development-only
        // degraded mode (`allow_degraded_experts = true`) preserves the
        // legacy behaviour: the failed slot is dropped from the mixture
        // (a zero contribution below) with a prominent warning and the
        // `degraded_expert_substitutions` counter marks the run
        // non-authoritative.
        let allow_degraded = self.core.options.policy.allow_degraded_experts;
        let mut first_fetch_error: Option<MoeStepError> = None;
        for (i, h) in miss_handles {
            // `fetch_with_retry` already retried with backoff. A join
            // error means the task itself panicked, which is fatal —
            // re-raise so the supervising scheduler can restart us.
            match h.await.expect("expert fetch task panicked") {
                Ok(r) => {
                    let new_hits = r.record_hit();
                    self.credit_prefetch_use(&r, new_hits);
                    // `bytes_read` is already bumped inside
                    // `fetch_once` on the actual leader path, so we
                    // don't double-count here. Followers that
                    // joined the singleflight (or that found the
                    // expert already resident by the time their
                    // task ran) contribute zero bytes, which is
                    // the correct accounting now that the engine
                    // dedups SSD reads (gist Phase 1).
                    residents[i] = Some(r);
                }
                Err(e) => {
                    let id = target[i];
                    self.metrics
                        .counters
                        .expert_read_failures
                        .fetch_add(1, Ordering::Relaxed);
                    if allow_degraded {
                        self.metrics
                            .counters
                            .degraded_expert_substitutions
                            .fetch_add(1, Ordering::Relaxed);
                        warn!(token = token_idx, layer, expert = id, error = %e,
                            "DEGRADED MODE: expert fetch failed after retries; \
                             dropping from mixture (allow_degraded_experts = true, \
                             output is non-authoritative)");
                    } else if first_fetch_error.is_none() {
                        first_fetch_error = Some(MoeStepError::ExpertFetch {
                            layer,
                            expert: id,
                            source: e,
                        });
                    }
                }
            }
        }
        let io_wait_elapsed = if had_misses {
            io_wait_start.elapsed()
        } else {
            std::time::Duration::ZERO
        };
        crate::stage_timing::record_optional(
            timings,
            crate::stage_timing::FOREGROUND_EXPERT_IO_WAIT,
            io_wait_elapsed,
        );
        let io_wait_us = io_wait_elapsed.as_micros() as u64;
        // Strict mode: surface the first fetch failure now that every
        // handle has been drained (so no fetch task is left detached
        // mid-flight holding a pool buffer).
        if let Some(err) = first_fetch_error {
            return Err(err);
        }
        // Degraded mode only: `None` entries correspond to failed
        // fetches and are dropped from the mixture below (a zero
        // contribution keeps the caller's weights[] alignment valid).
        let residents: Vec<Option<Arc<ExpertResident>>> = residents;

        // Run the SwiGLU FFN per expert against the hidden state.
        // Donate this worker thread for the duration: per-expert FFN
        // compute (CPU QMatMul or the synchronous wgpu dispatch +
        // readback) is a multi-millisecond blocking slice, and running
        // it inline would starve the speculative prefetch tasks spawned
        // above of a worker right when they must overlap this compute.
        let compute_start = Instant::now();
        let expert_policy = self.resolved_expert_execution_policy(residents.len());
        let step_result: Result<MoeStepResult, MoeStepError> = crate::stage_timing::time_optional(
            timings,
            crate::stage_timing::EXPERT_COMPUTE,
            || {
                run_compute_donated(|| {
                    match output_mode {
                        MoeStepOutputMode::PerExpert => {
                            let run_one = |r_opt: &Option<Arc<ExpertResident>>| -> Result<HiddenState, MoeStepError> {
                                let Some(r) = r_opt else {
                                    // Degraded mode only (strict mode returned above):
                                    // push a zero vector so the caller's weights[]
                                    // alignment stays valid (combining with weight
                                    // `w_i * 0 = 0` is equivalent to dropping this
                                    // expert from the mixture).
                                    return Ok(vec![0.0f32; self.core.shape.d_model]);
                                };
                                match self.forward_moe_resident(token_idx, layer, r, x, timings) {
                                    Ok(y) => Ok(y),
                                    Err(e @ MoeStepError::GpuExpertDispatch { .. }) => Err(e),
                                    Err(e) if allow_degraded => {
                                        self.metrics
                                            .counters
                                            .degraded_expert_substitutions
                                            .fetch_add(1, Ordering::Relaxed);
                                        warn!(
                                            token = token_idx,
                                            expert = r.id,
                                            error = %e,
                                            "DEGRADED MODE: expert compute failed; substituting zero \
                                             contribution (allow_degraded_experts = true, output is \
                                             non-authoritative)"
                                        );
                                        // Preserve the legacy aligned-vector contract for callers
                                        // that combine outside the engine.
                                        Ok(vec![0.0f32; self.core.shape.d_model])
                                    }
                                    Err(e) => Err(e),
                                }
                            };
                            let per_expert_y: Result<Vec<HiddenState>, MoeStepError> =
                                match expert_policy {
                                    ExpertExecutionPolicy::ParallelExpertsSingleThread => {
                                        use rayon::prelude::*;
                                        residents.par_iter().map(run_one).collect()
                                    }
                                    ExpertExecutionPolicy::Auto
                                    | ExpertExecutionPolicy::SequentialExpertsRowParallel => {
                                        residents.iter().map(run_one).collect()
                                    }
                                };
                            Ok(MoeStepResult::PerExpert(per_expert_y?))
                        }
                        MoeStepOutputMode::WeightedInto { weights, out } => {
                            debug_assert_eq!(residents.len(), weights.len());
                            out.clear();
                            out.resize(self.core.shape.d_model, 0.0);
                            if self.try_forward_moe_q4_layer_batch(
                                token_idx,
                                layer,
                                &target,
                                &residents,
                                x,
                                weights,
                                out,
                                timings,
                                allow_degraded,
                            )? {
                                return Ok(MoeStepResult::WeightedInto);
                            }
                            let accumulate_one = |slot: usize,
                                                  r_opt: &Option<Arc<ExpertResident>>,
                                                  acc: &mut [f32]|
                             -> Result<(), MoeStepError> {
                                let Some(r) = r_opt else {
                                    // Degraded mode only: failed fetch drops out of the
                                    // mixture (strict mode returned above).
                                    return Ok(());
                                };
                                match self.forward_moe_resident(token_idx, layer, r, x, timings) {
                                    Ok(y) => {
                                        let weight = weights[slot];
                                        if weight != 0.0 {
                                            for (dst, v) in acc.iter_mut().zip(y.iter()) {
                                                *dst += weight * *v;
                                            }
                                        }
                                        Ok(())
                                    }
                                    Err(e @ MoeStepError::GpuExpertDispatch { .. }) => Err(e),
                                    Err(e) if allow_degraded => {
                                        self.metrics
                                            .counters
                                            .degraded_expert_substitutions
                                            .fetch_add(1, Ordering::Relaxed);
                                        warn!(
                                            token = token_idx,
                                            expert = r.id,
                                            error = %e,
                                            "DEGRADED MODE: expert compute failed; dropping from \
                                             mixture (allow_degraded_experts = true, output is \
                                             non-authoritative)"
                                        );
                                        Ok(())
                                    }
                                    Err(e) => Err(e),
                                }
                            };
                            match expert_policy {
                                ExpertExecutionPolicy::ParallelExpertsSingleThread => {
                                    use rayon::prelude::*;
                                    let chunks =
                                        residents.len().min(crate::parallel::num_threads()).max(1);
                                    let chunk_len = residents.len().div_ceil(chunks).max(1);
                                    let d_model = self.core.shape.d_model;
                                    let partials: Result<Vec<Vec<f32>>, MoeStepError> = (0
                                        ..chunks)
                                        .into_par_iter()
                                        .map(|chunk| {
                                            let start = chunk * chunk_len;
                                            let end = (start + chunk_len).min(residents.len());
                                            let mut local = vec![0.0f32; d_model];
                                            for (offset, r_opt) in
                                                residents[start..end].iter().enumerate()
                                            {
                                                accumulate_one(start + offset, r_opt, &mut local)?;
                                            }
                                            Ok(local)
                                        })
                                        .collect();
                                    for partial in partials? {
                                        for (dst, v) in out.iter_mut().zip(partial.iter()) {
                                            *dst += *v;
                                        }
                                    }
                                }
                                ExpertExecutionPolicy::Auto
                                | ExpertExecutionPolicy::SequentialExpertsRowParallel => {
                                    for (slot, r_opt) in residents.iter().enumerate() {
                                        accumulate_one(slot, r_opt, out)?;
                                    }
                                }
                            }
                            Ok(MoeStepResult::WeightedInto)
                        }
                    }
                })
            },
        );
        let step_result = step_result?;
        let compute_us = compute_start.elapsed().as_micros() as u64;
        let _ = self.metrics.compute_hist.lock().record(compute_us.max(1));
        self.metrics
            .total_compute_us
            .fetch_add(compute_us, Ordering::Relaxed);
        self.metrics
            .total_io_wait_us
            .fetch_add(io_wait_us, Ordering::Relaxed);
        if io_wait_us > 0 {
            self.metrics
                .total_ssd_stall_us
                .fetch_add(io_wait_us, Ordering::Relaxed);
            if let Some(m) = &self.metrics.prom {
                m.record_ssd_stall(io_wait_us as f64 / 1_000_000.0);
            }
        }

        // Update predictor history (mirrors `generate`). The actual
        // union prefetch was already fired above, before the
        // target-miss await — this block only carries forward the
        // 2nd-order ring buffer for the *next* step's prefetch.
        //
        // **Layer-continuity guard (Finding 5).** The ring is
        // engine-global, so with concurrently-batched streams the
        // previous entry may belong to a different request. Only train
        // on `(last -> target)` when `last` really is this stream's
        // previous layer, and only feed the 2nd-order triple when
        // `last_last -> last` is contiguous too; otherwise skip the
        // observation rather than teach the predictor cross-stream
        // transitions. Single-stream behaviour is unchanged.
        if !target.is_empty() {
            let mut ring = self.speculation.markov_ring.lock();
            if !ring.last.ids.is_empty()
                && self.markov_layers_contiguous(ring.last.layer, Some(layer))
            {
                let pp: &[u32] =
                    if self.markov_layers_contiguous(ring.last_last.layer, ring.last.layer) {
                        &ring.last_last.ids
                    } else {
                        &[]
                    };
                self.core
                    .predictor
                    .observe_step2(pp, &ring.last.ids, &target);
            }
            ring.last_last = ring.last.clone();
            ring.last = MarkovHistory {
                ids: target.clone(),
                layer: Some(layer),
            };
        }

        let cycle_us = cycle_start.elapsed().as_micros() as u64;
        let _ = self.metrics.cycle_hist.lock().record(cycle_us.max(1));
        self.metrics
            .total_cycle_us
            .fetch_add(cycle_us, Ordering::Relaxed);
        self.metrics
            .tokens_processed
            .fetch_add(1, Ordering::Relaxed);

        Ok(step_result)
    }

    /// Whether `id` is currently resident in the expert cache. Cheap
    /// (one sharded-LRU lookup, no recency mutation) — used by the
    /// batch scheduler's pre-pass profitability gate to count how many
    /// peeked experts a warm pass would actually fetch.
    pub fn is_expert_cached(&self, id: u32) -> bool {
        self.core.cache.contains(id)
    }

    /// Force-fetch a specific set of experts and load them into the cache.
    /// Mirrors the spec example "the router selects Expert ID 3 and 7".
    ///
    /// **SSD Read De-Duplication (gist Phase 1).** The set is
    /// deduplicated (so accidental repeats in the caller's slice
    /// never trigger duplicate I/O), then every uncached id is
    /// fetched **concurrently** on the tokio runtime. Combined with
    /// the in-flight singleflight inside
    /// [`Self::fetch_with_retry`], `BatchScheduler` can call this
    /// once per batch with the union of every request's predicted
    /// experts and get exactly one disk read per unique id — the
    /// "single, unified" read the gist asks for.
    pub async fn warm_with(self: &Arc<Self>, ids: &[u32]) -> std::io::Result<()> {
        // Deduplicate up front: callers may pass overlapping
        // per-request prediction sets without thinking about it.
        let mut unique: HashSet<u32> = HashSet::with_capacity(ids.len());
        for &id in ids {
            if id >= self.core.router.num_experts() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("expert id {id} >= num_experts"),
                ));
            }
            // Skip ids that are already resident — we still record
            // them in `unique` so subsequent overlapping calls don't
            // re-issue, but no fetch task is spawned.
            if self.core.cache.contains(id) {
                continue;
            }
            unique.insert(id);
        }
        if unique.is_empty() {
            return Ok(());
        }

        // Spawn one fetch task per unique uncached id. All of them
        // funnel through `fetch_with_retry` (singleflight'd), so the
        // SSD sees at most one read per id even when this method is
        // called concurrently from multiple call sites (e.g. the
        // BatchScheduler pre-pass and a parallel speculative-decode
        // verification).
        let mut handles = Vec::with_capacity(unique.len());
        for id in unique {
            let me = self.clone();
            handles.push(tokio::spawn(
                async move { (id, me.fetch_with_retry(id).await) },
            ));
        }
        for h in handles {
            match h.await {
                Ok((id, Err(e))) => {
                    warn!(expert = id, error = %e, "warm_with: fetch failed");
                    // We swallow the error here: warm_with is a
                    // best-effort prefetch, and `moe_step`'s own
                    // retry / skip path will handle the same id
                    // again if it really is critical.
                }
                Ok((_, Ok(_))) => {}
                Err(e) => {
                    warn!(error = %e, "warm_with: fetch task panicked");
                }
            }
        }
        Ok(())
    }

    pub fn report(&self) -> EngineReport {
        let cycle = self.metrics.cycle_hist.lock();
        let io = self.metrics.io_hist.lock();
        let compute = self.metrics.compute_hist.lock();
        let tokens = self.metrics.tokens_processed.load(Ordering::Relaxed);
        let total_io_wait_us = self.metrics.total_io_wait_us.load(Ordering::Relaxed);
        let total_compute_us = self.metrics.total_compute_us.load(Ordering::Relaxed);
        let total_cycle_us = self.metrics.total_cycle_us.load(Ordering::Relaxed);
        let avg_io_wait_us = if tokens == 0 {
            0.0
        } else {
            total_io_wait_us as f64 / tokens as f64
        };
        let avg_compute_us = if tokens == 0 {
            0.0
        } else {
            total_compute_us as f64 / tokens as f64
        };
        let pct_time_io = if total_cycle_us == 0 {
            0.0
        } else {
            (total_io_wait_us as f64 / total_cycle_us as f64) * 100.0
        };
        EngineReport {
            hits: self.metrics.counters.hits.load(Ordering::Relaxed),
            misses: self.metrics.counters.misses.load(Ordering::Relaxed),
            prefetch_completed: self
                .metrics
                .counters
                .prefetch_completed
                .load(Ordering::Relaxed),
            bytes_read: self.metrics.counters.bytes_read.load(Ordering::Relaxed),
            cycle_p50_us: cycle.value_at_quantile(0.50),
            cycle_p95_us: cycle.value_at_quantile(0.95),
            cycle_p99_us: cycle.value_at_quantile(0.99),
            cycle_max_us: cycle.max(),
            io_p50_us: io.value_at_quantile(0.50),
            io_p95_us: io.value_at_quantile(0.95),
            io_p99_us: io.value_at_quantile(0.99),
            io_count: io.len(),
            compute_p50_us: compute.value_at_quantile(0.50),
            compute_p95_us: compute.value_at_quantile(0.95),
            compute_p99_us: compute.value_at_quantile(0.99),
            cache_capacity: self.core.cache.capacity(),
            pool_capacity: self.core.pool.capacity(),
            num_experts: self.core.router.num_experts(),
            top_k: self.core.router.top_k(),
            d_model: self.core.shape.d_model,
            d_ff: self.core.shape.d_ff,
            expert_size: self.core.storage.config().expert_size,
            predictor_observations: self.core.predictor.observations(),
            tokens_processed: tokens,
            avg_io_wait_us,
            avg_compute_us,
            pct_time_io,
            io_only: self.core.options.io_only,
            pinned_count: self.core.cache.pinned_count(),
            alias_redirects: self.speculation.alias_redirects.load(Ordering::Relaxed),
            dtype: self.core.options.dtype,
            partial_load_fraction: self.core.options.partial_load_fraction,
            predictive: self.predictive_telemetry(),
            locality_enabled: self.speculation.locality.is_some(),
            speculator_enabled: self.speculation.speculator.is_some(),
            expert_read_failures: self
                .metrics
                .counters
                .expert_read_failures
                .load(Ordering::Relaxed),
            degraded_expert_substitutions: self
                .metrics
                .counters
                .degraded_expert_substitutions
                .load(Ordering::Relaxed),
            inference_policy: self.core.options.policy,
            prefetch_dropped_concurrency: self
                .metrics
                .counters
                .prefetch_dropped_concurrency
                .load(Ordering::Relaxed),
            prefetch_dropped_pool_starved: self
                .metrics
                .counters
                .prefetch_dropped_pool_starved
                .load(Ordering::Relaxed),
            speculator_dmodel_mismatch: self
                .metrics
                .counters
                .speculator_dmodel_mismatch
                .load(Ordering::Relaxed),
            gpu_cpu_fallbacks: self
                .metrics
                .counters
                .gpu_cpu_fallbacks
                .load(Ordering::Relaxed),
            gpu_cache_enabled: self.core.gpu_cache.is_some(),
            vram_used_bytes: self
                .core
                .gpu_cache
                .as_ref()
                .map(|g| g.used_bytes() as u64)
                .unwrap_or(0),
            vram_capacity_bytes: self
                .core
                .gpu_cache
                .as_ref()
                .map(|g| g.capacity_bytes() as u64)
                .unwrap_or(0),
            gpu_promotions: self
                .core
                .gpu_cache
                .as_ref()
                .map(|g| g.promotions())
                .unwrap_or(0),
            gpu_cache_hits: self.core.gpu_cache.as_ref().map(|g| g.hits()).unwrap_or(0),
            gpu_cache_misses: self
                .core
                .gpu_cache
                .as_ref()
                .map(|g| g.misses())
                .unwrap_or(0),
            gpu_anchor_count: self
                .core
                .gpu_cache
                .as_ref()
                .map(|g| g.anchor_len())
                .unwrap_or(0),
            gpu_lru_count: self
                .core
                .gpu_cache
                .as_ref()
                .map(|g| g.lru_len())
                .unwrap_or(0),
            prefetch_used: self.metrics.counters.prefetch_used.load(Ordering::Relaxed),
            prefetch_dropped_governor: self
                .metrics
                .counters
                .prefetch_dropped_governor
                .load(Ordering::Relaxed),
            governor_enabled: self.core.governor.is_enabled(),
            governor_precision: self.core.governor.precision(),
            governor_admitted: self.core.governor.decisions().0,
            governor_throttled: self.core.governor.decisions().1,
            pregate_enabled: self.speculation.pregate.is_some(),
            pregate_accuracy: self
                .speculation
                .pregate
                .as_ref()
                .map(|pg| pg.accuracy())
                .unwrap_or(0.0),
            pregate_hits: self
                .speculation
                .pregate
                .as_ref()
                .map(|pg| pg.stats().0)
                .unwrap_or(0),
            pregate_misses: self
                .speculation
                .pregate
                .as_ref()
                .map(|pg| pg.stats().1)
                .unwrap_or(0),
            singleflight_followers: self
                .metrics
                .counters
                .singleflight_followers
                .load(Ordering::Relaxed),
            resident_expert_buffer_bytes: crate::expert_cache::resident_expert_buffer_bytes(),
            expert_buffer_pool_allocated_bytes: self.core.pool.allocated_bytes() as u64,
            expert_buffer_pool_primary_bytes: self.core.pool.primary_allocated_bytes() as u64,
            expert_buffer_pool_shadow_bytes: self.core.pool.shadow_allocated_bytes() as u64,
            prepared_duplicate_expert_bytes:
                crate::inference::prepared_duplicate_expert_bytes(),
            q8_direct_kernel_dispatches: crate::inference::q8_direct_kernel_dispatches(),
            q8_scalar_layout_fallbacks: crate::inference::q8_scalar_layout_fallbacks(),
            q8_preparation_seconds: crate::inference::q8_preparation_seconds(),
            q8_gate_up_kernel_seconds: crate::inference::q8_gate_up_kernel_seconds(),
            q8_down_kernel_seconds: crate::inference::q8_down_kernel_seconds(),
        }
    }

    pub fn print_summary(&self) {
        let r = self.report();
        let total = r.hits + r.misses;
        let hit_rate = if total == 0 {
            0.0
        } else {
            r.hits as f64 / total as f64 * 100.0
        };
        info!("===================== run summary =====================");
        info!(
            "experts:       {} (top-{}), cache={} slots, pool={} slots",
            r.num_experts, r.top_k, r.cache_capacity, r.pool_capacity
        );
        info!(
            "ffn shape:     d_model={}  d_ff={}  expert slot={} bytes  payload={} bytes (dtype={})",
            r.d_model,
            r.d_ff,
            r.expert_size,
            if r.dtype == WeightDtype::Mixed {
                r.expert_size
                    .saturating_sub(self.core.storage.config().block_align)
            } else {
                crate::inference::expert_weight_bytes_for(r.d_model, r.d_ff, r.dtype)
            },
            r.dtype.as_str()
        );
        if r.dtype == WeightDtype::Mixed {
            info!(
                "mixed quant:   experts={}  projections={}  q4k_opt={}  q5k_opt={}  q6k_opt={}  scalar_fallbacks={}  full_dequant_fallbacks={}  unsupported_dispatches={}",
                crate::inference::mixed_expert_dispatches(),
                crate::inference::quantized_projection_dispatches(),
                crate::inference::mixed_q4k_optimized_projection_dispatches(),
                crate::inference::mixed_q5k_optimized_projection_dispatches(),
                crate::inference::mixed_q6k_optimized_projection_dispatches(),
                crate::inference::mixed_scalar_fallbacks(),
                crate::inference::mixed_dequant_fallbacks(),
                crate::inference::unsupported_quant_dispatches()
            );
        }
        if r.dtype == WeightDtype::Q8_0 || r.q8_direct_kernel_dispatches > 0 {
            info!(
                "q8 memory:    resident_buffers={} bytes  pool={} bytes (primary={} shadow={})  prepared_duplicates={} bytes",
                r.resident_expert_buffer_bytes,
                r.expert_buffer_pool_allocated_bytes,
                r.expert_buffer_pool_primary_bytes,
                r.expert_buffer_pool_shadow_bytes,
                r.prepared_duplicate_expert_bytes,
            );
            info!(
                "q8 kernels:   production_auto_backend={}  direct_dispatches={}  scalar_layout_fallbacks={}  preparation={:.6}s  gate_up={:.6}s  down={:.6}s",
                crate::inference::q8_direct_kernel_backend(),
                r.q8_direct_kernel_dispatches,
                r.q8_scalar_layout_fallbacks,
                r.q8_preparation_seconds,
                r.q8_gate_up_kernel_seconds,
                r.q8_down_kernel_seconds,
            );
        }
        info!(
            "lookups:       hits={}  misses={}  hit_rate={:.2}%",
            r.hits, r.misses, hit_rate
        );
        info!(
            "prefetches:    completed={}  predictor_observations={}",
            r.prefetch_completed, r.predictor_observations
        );
        info!(
            "i/o:           reads={}  bytes={:.2} MiB",
            r.io_count,
            r.bytes_read as f64 / (1024.0 * 1024.0)
        );
        // SSD read de-duplication win — only emitted when singleflight
        // coalescing actually saved a read, so the legacy summary shape is
        // preserved for runs/tests that never contend on the same expert.
        if r.singleflight_followers > 0 {
            info!(
                "dedup:         {} concurrent reads coalesced onto in-flight leaders",
                r.singleflight_followers
            );
        }
        info!(
            "i/o latency:   p50={}us  p95={}us  p99={}us",
            r.io_p50_us, r.io_p95_us, r.io_p99_us
        );
        info!(
            "compute:       p50={}us  p95={}us  p99={}us  ({})",
            r.compute_p50_us,
            r.compute_p95_us,
            r.compute_p99_us,
            if r.io_only {
                "io-only XOR digest, FFN skipped"
            } else {
                "SwiGLU FFN per token"
            }
        );
        info!(
            "cycle latency: p50={}us  p95={}us  p99={}us  max={}us",
            r.cycle_p50_us, r.cycle_p95_us, r.cycle_p99_us, r.cycle_max_us
        );
        info!(
            "per-token avg: io_wait={:.1}us  compute={:.1}us  (over {} tokens)",
            r.avg_io_wait_us, r.avg_compute_us, r.tokens_processed
        );
        info!(
            "I/O share:     {:.2}% of token cycle time spent waiting on SSD reads",
            r.pct_time_io
        );
        info!(
            "energy knobs:  dtype={}  partial_load_fraction={:.2}  pinned={}  alias_redirects={}",
            r.dtype.as_str(),
            r.partial_load_fraction,
            r.pinned_count,
            r.alias_redirects
        );
        // Only emit the predictive line when either L or M is wired in;
        // the legacy benchmark path (everything off) keeps its existing
        // summary shape so older diff-on-output tests stay valid.
        if r.locality_enabled || r.speculator_enabled {
            info!(
                "predictive:    locality={} (hit_rate={:.2}%)  speculator={} (top1={:.2}% precision_at_k={:.2}%)  ssd_stall={:.1}ms",
                if r.locality_enabled { "on" } else { "off" },
                r.predictive.locality_hit_rate * 100.0,
                if r.speculator_enabled { "on" } else { "off" },
                r.predictive.speculator_top1_accuracy * 100.0,
                r.predictive.speculator_accuracy * 100.0,
                r.predictive.ssd_stall_us as f64 / 1000.0,
            );
        }
        // Tier 4 governor line — only emitted when the adaptive
        // prefetch governor is enabled, so the legacy summary shape is
        // untouched for existing runs/tests.
        if r.governor_enabled {
            let precision_pct = r.governor_precision * 100.0;
            let admit_total = r.governor_admitted + r.governor_throttled;
            let admit_rate = if admit_total == 0 {
                0.0
            } else {
                r.governor_admitted as f64 / admit_total as f64 * 100.0
            };
            info!(
                "governor:      on  precision={:.2}%  prefetch_used={}/{}  admitted={} throttled={} ({:.1}% admit)",
                precision_pct,
                r.prefetch_used,
                r.prefetch_completed,
                r.governor_admitted,
                r.governor_throttled,
                admit_rate,
            );
        }
        // Tier 3 pre-gate line — only when the per-layer pre-gate is
        // enabled, keeping the legacy summary shape otherwise.
        if r.pregate_enabled {
            info!(
                "pregate:       on  accuracy={:.2}%  predictions={}/{} (hit/total)",
                r.pregate_accuracy * 100.0,
                r.pregate_hits,
                r.pregate_hits + r.pregate_misses,
            );
        }
        // Health diagnostics — only emitted when something actually
        // degraded, so the legacy summary shape is preserved for clean
        // runs. These counters are otherwise only legible via the
        // Prometheus exporter; surfacing them here makes a problem run
        // self-explanatory from the CLI summary alone.
        let prefetch_dropped = r.prefetch_dropped_concurrency
            + r.prefetch_dropped_pool_starved
            + r.prefetch_dropped_governor;
        if prefetch_dropped > 0
            || r.gpu_cpu_fallbacks > 0
            || r.speculator_dmodel_mismatch > 0
            || r.expert_read_failures > 0
        {
            info!(
                "diagnostics:   prefetch_dropped={} (concurrency={} pool_starved={} governor={})  gpu_cpu_fallbacks={}  speculator_dmodel_mismatch={}  expert_read_failures={}",
                prefetch_dropped,
                r.prefetch_dropped_concurrency,
                r.prefetch_dropped_pool_starved,
                r.prefetch_dropped_governor,
                r.gpu_cpu_fallbacks,
                r.speculator_dmodel_mismatch,
                r.expert_read_failures,
            );
        }
        info!("=======================================================");
    }
}

#[derive(Debug, Clone)]
pub struct EngineReport {
    pub hits: u64,
    pub misses: u64,
    pub prefetch_completed: u64,
    pub bytes_read: u64,
    pub cycle_p50_us: u64,
    pub cycle_p95_us: u64,
    pub cycle_p99_us: u64,
    pub cycle_max_us: u64,
    pub io_p50_us: u64,
    pub io_p95_us: u64,
    pub io_p99_us: u64,
    pub io_count: u64,
    pub compute_p50_us: u64,
    pub compute_p95_us: u64,
    pub compute_p99_us: u64,
    pub cache_capacity: usize,
    pub pool_capacity: usize,
    pub num_experts: u32,
    pub top_k: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub expert_size: usize,
    pub predictor_observations: u64,
    /// Number of `Engine::generate` calls completed.
    pub tokens_processed: u64,
    /// Mean per-token critical-path I/O wait, in microseconds. Tokens that
    /// were entirely served from cache contribute 0 to this average.
    pub avg_io_wait_us: f64,
    /// Mean per-token compute (FFN forward, or XOR-digest under
    /// `--io-only`), in microseconds.
    pub avg_compute_us: f64,
    /// Total per-token critical-path I/O wait as a percentage of total
    /// token cycle time — the headline "what fraction of token time was
    /// the engine waiting on SSD?" number the run summary prints.
    pub pct_time_io: f64,
    /// Whether this run was executed in `--io-only` mode (FFN skipped).
    pub io_only: bool,
    /// Number of experts currently pinned in the LRU cache (Change 5:
    /// frequency-based pinning).
    pub pinned_count: usize,
    /// Number of times an alias map redirected an expert id to a
    /// canonical id (Change 6: expert deduplication). Each redirect is
    /// one cache lookup that targeted a deduplicated copy.
    pub alias_redirects: u64,
    /// On-disk weight dtype used by this engine instance (Change 1).
    pub dtype: WeightDtype,
    /// Partial-load fraction used by this engine instance (Change 3).
    pub partial_load_fraction: f64,
    /// Snapshot of the predictive-architecture telemetry: locality
    /// hit-rate, speculator accuracy, and cumulative SSD critical-path
    /// stall. Populated regardless of whether the L/M arms are wired
    /// in (the counters stay at zero when disabled, which still
    /// produces the correct `0.0` ratios).
    pub predictive: PredictiveTelemetry,
    /// Whether the [`LocalityMonitor`] (the **L** arm of the
    /// predictive `S ∪ L ∪ M` union-fetch) was configured on this run.
    pub locality_enabled: bool,
    /// Whether the [`NeuralSpeculator`] (the **M** arm of the
    /// predictive `S ∪ L ∪ M` union-fetch) was configured on this run.
    pub speculator_enabled: bool,
    /// Cumulative number of routed experts dropped from a mixture
    /// because their fetch (after retries) failed. Non-zero values
    /// indicate corrupt weight files or persistent SSD I/O errors;
    /// alert on a non-zero rate from the Prometheus exporter.
    pub expert_read_failures: u64,
    /// Cumulative zero-vector substitutions performed by the
    /// development-only `allow_degraded_experts` mode. Always zero in
    /// strict production mode; any non-zero value marks the run's
    /// output and benchmark figures as degraded / non-authoritative.
    pub degraded_expert_substitutions: u64,
    /// The engine-scoped fail-open policy set active for this run
    /// (hardening pass, policy separation). Any `true` field marks the
    /// run's output and every benchmark figure as potentially
    /// non-authoritative — surfaced here so request/benchmark metadata
    /// can report whether a degraded policy was used.
    pub inference_policy: crate::inference::RealInferencePolicy,
    /// Speculative prefetches dropped because
    /// `EngineOptions::max_concurrent_prefetches` was already saturated.
    pub prefetch_dropped_concurrency: u64,
    /// Speculative prefetches dropped because no pool buffer could be
    /// acquired (shadow half starved even after recycling an LRU
    /// shadow-backed resident, or legacy primary pool busy). A high
    /// rate means look-ahead is being silently disabled by buffer
    /// starvation — grow the shadow half or reduce prefetch fanout.
    pub prefetch_dropped_pool_starved: u64,
    /// Tokens for which the neural speculator was disabled by a
    /// hidden-state / `d_model` mismatch. Persistent non-zero values
    /// mean the M predictive arm is misconfigured and contributing
    /// nothing.
    pub speculator_dmodel_mismatch: u64,
    /// Expert activations that fell back from the GPU fast path to the
    /// CPU path because physical GPU expert dispatch errored. Non-zero rates
    /// explain mixed GPU/CPU token latency in serving-fallback mode.
    pub gpu_cpu_fallbacks: u64,
    /// Whether the engine has a logical GPU-admission cache attached.
    pub gpu_cache_enabled: bool,
    /// Compatibility field: logical host payload bytes admitted across the
    /// GPU Anchor + LRU. Not physical wgpu allocation bytes.
    pub vram_used_bytes: u64,
    /// Compatibility field: logical admission budget. The same configured
    /// value caps physical expert weights; fixed workspaces are separate.
    pub vram_capacity_bytes: u64,
    /// Cumulative successful logical GPU-promotion transitions: new
    /// Anchor/LRU admissions plus LRU-to-Anchor graduation. No-op outcomes are
    /// excluded; this mirrors `mer_promotions_total` when metrics are enabled.
    pub gpu_promotions: u64,
    /// Logical GPU-admission hit count (anchor + LRU).
    pub gpu_cache_hits: u64,
    /// Logical GPU-admission miss count.
    pub gpu_cache_misses: u64,
    /// Number of experts in the logical admission anchor region.
    pub gpu_anchor_count: usize,
    /// Number of experts in the logical admission LRU region.
    pub gpu_lru_count: usize,
    /// **Tier 4.** Speculative prefetches that landed in cache and were
    /// then consumed by a hit before eviction (precision numerator).
    /// Previously always `0` (the counter was dead); now wired on both
    /// the `generate` and `moe_step` hit paths.
    pub prefetch_used: u64,
    /// **Tier 4.** Speculative prefetches the adaptive governor declined
    /// to admit. `0` when the governor is disabled (the default).
    pub prefetch_dropped_governor: u64,
    /// **Tier 4.** Whether the adaptive prefetch governor is enabled.
    pub governor_enabled: bool,
    /// **Tier 4.** Current governor precision EWMA (consumed / completed),
    /// in `[0, 1]`. Meaningless when `governor_enabled` is `false`.
    pub governor_precision: f64,
    /// **Tier 4.** Cumulative speculative prefetches the governor
    /// admitted.
    pub governor_admitted: u64,
    /// **Tier 4.** Cumulative speculative prefetches the governor
    /// throttled (declined).
    pub governor_throttled: u64,
    /// **Tier 3.** Whether the per-layer pre-gate predictor is enabled.
    pub pregate_enabled: bool,
    /// **Tier 3.** Fraction of scored pre-gate predictions that
    /// intersected the actually-routed next-layer set, in `[0, 1]`.
    /// `0.0` when the pre-gate is disabled or nothing was scored yet.
    pub pregate_accuracy: f64,
    /// **Tier 3.** Pre-gate predictions that intersected the actual
    /// next-layer routed set.
    pub pregate_hits: u64,
    /// **Tier 3.** Pre-gate predictions that missed.
    pub pregate_misses: u64,
    /// Concurrent `fetch_with_retry` callers that piggy-backed on an
    /// in-flight leader's read (or a just-published resident) instead of
    /// issuing their own (Phase 1 — SSD read de-duplication). Each one
    /// maps directly to one disk read that was avoided.
    pub singleflight_followers: u64,
    /// Bytes held by live CPU resident expert buffers (occupancy, not pool allocation).
    pub resident_expert_buffer_bytes: u64,
    /// Bytes preallocated by all primary and shadow expert-buffer slots.
    pub expert_buffer_pool_allocated_bytes: u64,
    /// Bytes preallocated by primary expert-buffer slots.
    pub expert_buffer_pool_primary_bytes: u64,
    /// Bytes preallocated by speculative shadow expert-buffer slots.
    pub expert_buffer_pool_shadow_bytes: u64,
    /// Bytes retained by prepared duplicate expert representations. Zero for
    /// the production native Q8_0 path.
    pub prepared_duplicate_expert_bytes: u64,
    pub q8_direct_kernel_dispatches: u64,
    pub q8_scalar_layout_fallbacks: u64,
    pub q8_preparation_seconds: f64,
    pub q8_gate_up_kernel_seconds: f64,
    pub q8_down_kernel_seconds: f64,
}

#[cfg(test)]
mod tests {
    //! Integration test for the full `Engine::generate` loop.
    //!
    //! Wires the real `NvmeStorage` (with `O_DIRECT` disabled — required on
    //! tmpfs/CI), real `BufferPool`, real `ExpertCache`, real `TopKRouter`
    //! and `PredictiveLoader` against on-disk synthetic experts written by
    //! `generate_synthetic_experts`, and runs many tokens through
    //! `Engine::generate`. This is the "no integration tests for the full
    //! Engine::generate loop" gap closed.
    use super::*;
    use crate::buffer_pool::BufferPool;
    use crate::io_provider::{generate_synthetic_experts, NvmeStorage, StorageConfig};
    use crate::router::{PredictiveLoader, TopKRouter};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

    /// Self-cleaning unique temp directory for test fixtures.
    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(label: &str) -> Self {
            // Combine pid + monotonic counter + nanos for uniqueness across
            // parallel test runs without pulling in a tempfile dependency.
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir().join(format!(
                "micro-expert-router-{label}-{}-{n}-{ts}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).expect("create temp dir");
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn gpu_native_current_to_missing_recovery_is_bounded_fail_closed() {
        assert!(gpu_native_physical_demand_recovery_allowed(0));
        assert!(!gpu_native_physical_demand_recovery_allowed(1));
        assert!(!gpu_native_physical_demand_recovery_allowed(usize::MAX));
        assert_eq!(GPU_NATIVE_PHYSICAL_DEMAND_RECOVERY_LIMIT, 1);
        assert_eq!(GPU_NATIVE_LOGICAL_DEMAND_SET_ATTEMPTS, 2);

        let error = GpuNativeDemandResidencyError::PhysicalDemandRecoveryExhausted {
            global_id: 6065,
            recovery_attempts: GPU_NATIVE_PHYSICAL_DEMAND_RECOVERY_LIMIT,
        };
        assert_eq!(
            error.to_string(),
            "tiered residency physical demand-set recovery remained stale/missing for expert 6065 after 1 bounded recovery attempt(s)"
        );
    }

    #[test]
    fn gpu_native_physical_partition_skips_hits_and_preserves_miss_order() {
        let selected = [7, 3, 11, 5];
        assert!(gpu_native_physical_missing_ids(&selected, &[true; 4]).is_empty());
        assert_eq!(
            gpu_native_physical_missing_ids(&selected, &[true, false, true, false]),
            vec![3, 5]
        );
    }

    fn build_engine(
        data_dir: &std::path::Path,
        num_experts: u32,
        d_model: usize,
        d_ff: usize,
        cache_slots: usize,
        top_k: usize,
        predict_fanout: usize,
        seed: u64,
    ) -> Arc<Engine> {
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block_align = 4096usize;
        // Round expert_size up to a multiple of block_align (an O_DIRECT
        // invariant the storage layer asserts even when --no-direct is set).
        let expert_size = weight_bytes.div_ceil(block_align) * block_align;

        generate_synthetic_experts(data_dir, num_experts, expert_size, d_model, d_ff)
            .expect("generate synthetic experts");

        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: data_dir.to_path_buf(),
                expert_size,
                block_align,
                // tmpfs / overlayfs (typical for CI) doesn't support O_DIRECT.
                use_direct_io: false,
                num_experts_per_layer: None,
            })
            .expect("storage init"),
        );
        storage
            .warmup_fds(0..num_experts)
            .expect("pre-open expert fds");

        let pool_slots = cache_slots + predict_fanout.max(1);
        let pool = BufferPool::new(pool_slots, expert_size, block_align);
        let cache = Arc::new(MultiLayerExpertCache::single_layer(cache_slots));
        let router = Router::Markov(Arc::new(TopKRouter::new(num_experts, top_k, seed)));
        let predictor = Arc::new(PredictiveLoader::new(
            num_experts,
            predict_fanout,
            0.05,
            seed,
        ));

        Arc::new(Engine::new(
            cache,
            pool,
            storage,
            router,
            predictor,
            ModelShape {
                d_model,
                d_ff,
                hidden_seed: seed,
            },
        ))
    }

    fn build_multi_layer_source_engine(
        data_dir: &std::path::Path,
        per_layer_caps: Vec<usize>,
        experts_per_layer: u32,
        top_k: usize,
        seed: u64,
    ) -> Arc<Engine> {
        let d_model = 8usize;
        let d_ff = 8usize;
        let num_experts = experts_per_layer * per_layer_caps.len() as u32;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block_align = 4096usize;
        let expert_size = weight_bytes.div_ceil(block_align) * block_align;
        generate_synthetic_experts(data_dir, num_experts, expert_size, d_model, d_ff)
            .expect("generate multi-layer synthetic experts");
        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: data_dir.to_path_buf(),
                expert_size,
                block_align,
                use_direct_io: false,
                num_experts_per_layer: Some(experts_per_layer),
            })
            .expect("multi-layer storage init"),
        );
        storage
            .warmup_fds(0..num_experts)
            .expect("pre-open multi-layer expert fds");
        let cache = Arc::new(MultiLayerExpertCache::with_capacities(
            per_layer_caps,
            experts_per_layer,
        ));
        let pool = BufferPool::new(
            cache.capacity() + top_k.max(1),
            expert_size,
            block_align,
        );
        let router = Router::Markov(Arc::new(TopKRouter::new(num_experts, top_k, seed)));
        let predictor = Arc::new(PredictiveLoader::new(num_experts, 0, 0.05, seed));
        Arc::new(Engine::new(
            cache,
            pool,
            storage,
            router,
            predictor,
            ModelShape {
                d_model,
                d_ff,
                hidden_seed: seed,
            },
        ))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn gpu_native_demand_source_reuses_ram_without_duplicate_nvme_read() {
        let dir = TempDir::new("gpu-native-demand-source");
        let engine = build_engine(&dir.path, 4, 8, 8, 2, 1, 1, 17);
        assert_eq!(engine.report().bytes_read, 0);

        let mut first_request = HashMap::new();
        let first = engine
            .gpu_native_demand_source(0, &mut first_request)
            .await
            .unwrap();
        let after_nvme = engine.report().bytes_read;
        assert_eq!(after_nvme, engine.core.storage.config().expert_size as u64);

        let mut second_request = HashMap::new();
        let ram_hit = engine
            .gpu_native_demand_source(0, &mut second_request)
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&first, &ram_hit));
        assert_eq!(engine.report().bytes_read, after_nvme);

        let same_request = engine
            .gpu_native_demand_source(0, &mut second_request)
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&ram_hit, &same_request));
        assert_eq!(engine.report().bytes_read, after_nvme);
    }

    #[test]
    fn production_commit_order_ignores_later_request_completing_first() {
        let pool = BufferPool::new(3, 4096, 4096);
        let request_order = [3, 1, 2];
        let mut completed = HashMap::new();
        for &id in request_order.iter().rev() {
            completed.insert(id, Arc::new(ExpertResident::new(id, pool.try_acquire().unwrap())));
        }

        let ordered = qualification_order_completed_residents(&request_order, completed).unwrap();
        assert_eq!(
            ordered.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            request_order
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn production_multilayer_warmup_shape_reconciles_ordered_hashes_and_full_state() {
        let control_dir = TempDir::new("production-multilayer-warmup-control");
        let treatment_dir = TempDir::new("production-multilayer-warmup-treatment");
        let control =
            build_multi_layer_source_engine(&control_dir.path, vec![3, 2], 8, 2, 0xA21);
        let treatment =
            build_multi_layer_source_engine(&treatment_dir.path, vec![3, 2], 8, 2, 0xA21);
        control
            .enable_gpu_native_demand_source_production_qualification(
                GpuNativeDemandSourceQualificationArm::Control,
            )
            .unwrap();
        treatment
            .enable_gpu_native_demand_source_production_qualification(
                GpuNativeDemandSourceQualificationArm::Treatment,
            )
            .unwrap();

        // Cold start, then alternate layers until both aggregate and
        // per-layer limits have been crossed repeatedly. Every set is an
        // exact all-RAM-miss pair and treatment uses ordinary production.
        let source_sets = [
            [0, 1],
            [8, 9],
            [2, 3],
            [10, 11],
            [4, 5],
            [12, 13],
            [6, 7],
            [14, 15],
        ];
        for ids in source_sets {
            run_production_source_set(control.clone(), ids.to_vec())
                .await
                .unwrap();
            run_production_source_set(treatment.clone(), ids.to_vec())
                .await
                .unwrap();
        }

        let control_source = control
            .gpu_native_demand_source_qualification_snapshot()
            .unwrap();
        let treatment_source = treatment
            .gpu_native_demand_source_qualification_snapshot()
            .unwrap();
        assert_eq!(
            control_source.demand_ram_eviction_ids_sha256,
            treatment_source.demand_ram_eviction_ids_sha256,
            "authoritative warmup-shaped ordered eviction stream"
        );
        assert_eq!(
            control_source.demand_ram_insert_ids_sha256,
            treatment_source.demand_ram_insert_ids_sha256,
            "authoritative warmup-shaped ordered insertion stream"
        );
        assert_eq!(
            control.core.cache.qualification_state_sha256(),
            treatment.core.cache.qualification_state_sha256(),
            "complete layer-indexed MRU-to-LRU cache state"
        );
        assert_eq!(control.core.cache.len(), treatment.core.cache.len());
        for layer in 0..control.core.cache.num_layers() {
            assert_eq!(
                control
                    .core
                    .cache
                    .cache_for_layer(layer as u32)
                    .resident_ids(),
                treatment
                    .core
                    .cache
                    .cache_for_layer(layer as u32)
                    .resident_ids()
            );
        }
        assert!(control_source.ram_cache_evictions > 0);
        assert_eq!(
            control_source.ram_cache_evictions,
            treatment_source.ram_cache_evictions
        );
        let production = treatment.production_demand_source_snapshot();
        assert_eq!(production.production_batch_successes, source_sets.len() as u64);
        assert_eq!(production.production_cache_reservation_leaks, 0);
        assert_eq!(production.production_batch_commit_violations, 0);
        assert_eq!(production.stale_singleflight_entries, 0);
    }

    async fn wait_for_production_snapshot<F>(engine: &Arc<Engine>, condition: F)
    where
        F: Fn(&ProductionDemandSourceSnapshot) -> bool,
    {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let snapshot = engine.production_demand_source_snapshot();
                if condition(&snapshot) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("production telemetry condition timed out");
    }

    async fn run_production_source_set(
        engine: Arc<Engine>,
        ids: Vec<u32>,
    ) -> Result<HashMap<u32, Arc<ExpertResident>>, GpuNativeDemandResidencyError> {
        let mut residents = HashMap::new();
        engine
            .gpu_native_source_physical_missing_set(&ids, &mut residents)
            .await?;
        Ok(residents)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn production_same_all_miss_set_singleflights_one_batch_per_unique_expert() {
        let dir = TempDir::new("production-same-set");
        let engine = build_engine(&dir.path, 8, 8, 8, 6, 3, 0, 17);
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        engine.production_batch_test_hooks.lock().after_claims = Some(barrier.clone());

        let first = tokio::spawn(run_production_source_set(engine.clone(), vec![0, 1, 2]));
        barrier.wait().await;
        let second = tokio::spawn(run_production_source_set(engine.clone(), vec![0, 1, 2]));
        wait_for_production_snapshot(&engine, |snapshot| {
            snapshot.production_sequential_fallback_singleflight_contention == 1
        })
        .await;
        barrier.wait().await;

        let first = first.await.unwrap().unwrap();
        let second = second.await.unwrap().unwrap();
        engine.production_batch_test_hooks.lock().after_claims = None;
        for id in [0, 1, 2] {
            assert!(Arc::ptr_eq(&first[&id], &second[&id]));
        }
        assert_eq!(
            engine.report().bytes_read,
            3 * engine.core.storage.config().expert_size as u64
        );
        let snapshot = engine.production_demand_source_snapshot();
        assert_eq!(snapshot.production_batch_successes, 1);
        assert_eq!(snapshot.production_batch_experts, 3);
        assert!(snapshot.production_singleflight_followers_observed > 0);
        assert_eq!(snapshot.production_cache_reservation_leaks, 0);
        assert_eq!(snapshot.stale_singleflight_entries, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn production_partial_overlap_never_reads_shared_ids_twice() {
        let dir = TempDir::new("production-overlap");
        let engine = build_engine(&dir.path, 8, 8, 8, 8, 4, 0, 18);
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        engine.production_batch_test_hooks.lock().after_claims = Some(barrier.clone());

        let first = tokio::spawn(run_production_source_set(engine.clone(), vec![1, 2, 3, 4]));
        barrier.wait().await;
        let second = tokio::spawn(run_production_source_set(engine.clone(), vec![3, 4, 5, 6]));
        wait_for_production_snapshot(&engine, |snapshot| {
            snapshot.production_sequential_fallback_singleflight_contention == 1
        })
        .await;
        barrier.wait().await;
        let first = first.await.unwrap().unwrap();
        let second = second.await.unwrap().unwrap();
        engine.production_batch_test_hooks.lock().after_claims = None;
        assert!(Arc::ptr_eq(&first[&3], &second[&3]));
        assert!(Arc::ptr_eq(&first[&4], &second[&4]));
        assert_eq!(
            engine.report().bytes_read,
            6 * engine.core.storage.config().expert_size as u64
        );
        assert_eq!(
            engine
                .production_demand_source_snapshot()
                .stale_singleflight_entries,
            0
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn production_disjoint_same_layer_reservations_never_steal_or_evict_pins() {
        let dir = TempDir::new("production-disjoint-reservations");
        let engine = build_engine(&dir.path, 10, 8, 8, 6, 2, 0, 19);
        let mut priming = HashMap::new();
        for id in 0..6 {
            engine
                .gpu_native_demand_source(id, &mut priming)
                .await
                .unwrap();
        }
        drop(priming);
        engine.core.cache.pin(0);
        let barrier = Arc::new(tokio::sync::Barrier::new(3));
        engine.production_batch_test_hooks.lock().after_buffers = Some(barrier.clone());

        let first = tokio::spawn(run_production_source_set(engine.clone(), vec![6, 7]));
        let second = tokio::spawn(run_production_source_set(engine.clone(), vec![8, 9]));
        barrier.wait().await;
        assert_eq!(engine.core.cache.reserved_slots(), 4);
        assert!(
            engine.core.cache.len() + engine.core.cache.reserved_slots()
                <= engine.core.cache.capacity()
        );
        assert!(engine.core.cache.contains(0));
        assert!(engine.core.cache.is_pinned(0));
        barrier.wait().await;
        first.await.unwrap().unwrap();
        second.await.unwrap().unwrap();
        engine.production_batch_test_hooks.lock().after_buffers = None;

        assert_eq!(engine.core.cache.len(), engine.core.cache.capacity());
        assert!(engine.core.cache.contains(0));
        assert!(engine.core.cache.is_pinned(0));
        let snapshot = engine.production_demand_source_snapshot();
        assert_eq!(snapshot.production_batch_successes, 2);
        assert_eq!(snapshot.production_cache_slots_reserved, 4);
        assert_eq!(snapshot.production_cache_reservations_consumed, 4);
        assert_eq!(snapshot.production_cache_reservations_released, 0);
        assert_eq!(snapshot.production_cache_reservation_leaks, 0);
        assert_eq!(snapshot.production_batch_commit_violations, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn production_partial_claim_race_rolls_back_and_falls_back_without_deadlock() {
        let dir = TempDir::new("production-claim-race");
        let engine = build_engine(&dir.path, 4, 8, 8, 4, 2, 0, 20);
        let occupied_notify = Arc::new(Notify::new());
        assert!(engine
            .core
            .in_flight
            .insert(2, occupied_notify.clone())
            .is_none());
        let rollback_barrier = Arc::new(tokio::sync::Barrier::new(2));
        engine
            .production_batch_test_hooks
            .lock()
            .after_claim_rollback = Some(rollback_barrier.clone());

        let request = tokio::spawn(run_production_source_set(engine.clone(), vec![1, 2]));
        rollback_barrier.wait().await;
        assert!(!engine.core.in_flight.contains_key(&1));
        assert!(engine.core.in_flight.contains_key(&2));
        rollback_barrier.wait().await;
        engine.core.in_flight.remove(&2);
        occupied_notify.notify_waiters();
        let residents = tokio::time::timeout(Duration::from_secs(2), request)
            .await
            .expect("claim-race fallback deadlocked")
            .unwrap()
            .unwrap();
        engine
            .production_batch_test_hooks
            .lock()
            .after_claim_rollback = None;
        assert_eq!(residents.len(), 2);
        let snapshot = engine.production_demand_source_snapshot();
        assert_eq!(snapshot.production_singleflight_claim_rollbacks, 1);
        assert_eq!(
            snapshot.production_sequential_fallback_singleflight_contention,
            1
        );
        assert_eq!(snapshot.stale_singleflight_entries, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn production_post_claim_cache_insertion_aborts_batch_without_duplicate_read() {
        let dir = TempDir::new("production-cache-insertion-race");
        let engine = build_engine(&dir.path, 4, 8, 8, 4, 2, 0, 21);
        let mut priming = HashMap::new();
        let retained = engine
            .gpu_native_demand_source(0, &mut priming)
            .await
            .unwrap();
        drop(priming);
        let evicted = engine.core.cache.evict_lru().unwrap();
        assert!(Arc::ptr_eq(&retained, &evicted));
        let before = engine.report().bytes_read;
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        engine.production_batch_test_hooks.lock().after_claims = Some(barrier.clone());

        let request = tokio::spawn(run_production_source_set(engine.clone(), vec![0, 1]));
        barrier.wait().await;
        assert!(engine.core.cache.insert(evicted).is_ok());
        barrier.wait().await;
        let residents = request.await.unwrap().unwrap();
        engine.production_batch_test_hooks.lock().after_claims = None;
        assert!(Arc::ptr_eq(&retained, &residents[&0]));
        assert_eq!(
            engine.report().bytes_read - before,
            engine.core.storage.config().expert_size as u64
        );
        let snapshot = engine.production_demand_source_snapshot();
        assert_eq!(snapshot.production_batch_successes, 0);
        assert_eq!(snapshot.production_sequential_fallback_mixed_ram, 1);
        assert_eq!(snapshot.production_singleflight_claim_rollbacks, 2);
        assert_eq!(snapshot.stale_singleflight_entries, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn production_batch_device_failure_after_victims_fails_closed_without_late_fallback() {
        let dir = TempDir::new("production-batch-read-error");
        let engine = build_engine(&dir.path, 6, 8, 8, 4, 2, 0, 22);
        let mut priming = HashMap::new();
        for id in 0..4 {
            engine
                .gpu_native_demand_source(id, &mut priming)
                .await
                .unwrap();
        }
        drop(priming);
        std::fs::OpenOptions::new()
            .write(true)
            .open(dir.path.join("expert_4.bin"))
            .unwrap()
            .set_len(0)
            .unwrap();
        let mut residents = HashMap::new();
        let error = engine
            .gpu_native_source_physical_missing_set(&[4, 5], &mut residents)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            GpuNativeDemandResidencyError::ProductionBatchReadFailedAfterReservation { .. }
        ));
        assert!(residents.is_empty());
        assert!(!engine.core.cache.contains(0));
        assert!(!engine.core.cache.contains(1));
        assert!(engine.core.cache.contains(2));
        assert!(engine.core.cache.contains(3));
        assert!(!engine.core.cache.contains(4));
        assert!(!engine.core.cache.contains(5));
        if let Err(error) = engine.fetch_with_retry(5).await {
            panic!("sequential recovery for healthy expert failed: {error}");
        }
        assert!(!engine.core.storage.is_drive_unavailable(5));
        let snapshot = engine.production_demand_source_snapshot();
        assert_eq!(snapshot.production_sequential_fallback_batch_read_error, 0);
        assert_eq!(snapshot.production_batch_successes, 0);
        assert_eq!(snapshot.production_cache_reservations_released, 2);
        assert_eq!(snapshot.production_singleflight_claim_rollbacks, 2);
        assert_eq!(snapshot.production_cache_reservation_leaks, 0);
        assert_eq!(snapshot.stale_singleflight_entries, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn production_pool_failure_after_victims_fails_closed_and_releases_transaction() {
        let dir = TempDir::new("production-pool-fail-closed");
        let engine = build_engine(&dir.path, 4, 8, 8, 2, 2, 0, 122);
        let mut retained = HashMap::new();
        for id in [0, 1] {
            engine
                .gpu_native_demand_source(id, &mut retained)
                .await
                .unwrap();
        }

        // Eviction removes both cache references, but request-local Arcs keep
        // those victim buffers unavailable. Only the single headroom buffer
        // can be acquired, forcing the post-reservation fail-closed path.
        let mut residents = HashMap::new();
        let error = engine
            .gpu_native_source_physical_missing_set(&[2, 3], &mut residents)
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            GpuNativeDemandResidencyError::ProductionBatchPoolUnavailableAfterReservation {
                requested: 2,
                acquired: 1,
            }
        ));
        assert!(residents.is_empty());
        assert_eq!(engine.core.cache.len(), 0);
        assert_eq!(engine.core.cache.reserved_slots(), 0);
        let snapshot = engine.production_demand_source_snapshot();
        assert_eq!(snapshot.production_sequential_fallback_pool, 0);
        assert_eq!(snapshot.production_cache_reservations_released, 2);
        assert_eq!(snapshot.production_singleflight_claim_rollbacks, 2);
        assert_eq!(snapshot.production_cache_reservation_leaks, 0);
        assert_eq!(snapshot.stale_singleflight_entries, 0);
        drop(retained);
        assert_eq!(
            engine.core.pool.primary_available(),
            engine.core.pool.capacity()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn production_batch_cancellation_releases_claims_reservations_and_buffers() {
        let dir = TempDir::new("production-batch-cancel");
        let engine = build_engine(&dir.path, 4, 8, 8, 4, 2, 0, 23);
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        engine.production_batch_test_hooks.lock().after_buffers = Some(barrier.clone());
        let request = tokio::spawn(run_production_source_set(engine.clone(), vec![0, 1]));
        barrier.wait().await;
        assert_eq!(engine.core.cache.reserved_slots(), 2);
        assert_eq!(engine.core.in_flight.len(), 2);
        assert_eq!(
            engine.core.pool.primary_available(),
            engine.core.pool.capacity() - 2
        );
        request.abort();
        match request.await {
            Err(error) => assert!(error.is_cancelled()),
            Ok(_) => panic!("cancelled production batch unexpectedly completed"),
        }
        engine.production_batch_test_hooks.lock().after_buffers = None;

        let snapshot = engine.production_demand_source_snapshot();
        assert_eq!(snapshot.production_cache_reservation_leaks, 0);
        assert_eq!(snapshot.stale_singleflight_entries, 0);
        assert_eq!(snapshot.production_cache_reservations_released, 2);
        assert_eq!(snapshot.production_singleflight_claim_rollbacks, 2);
        assert_eq!(
            engine.core.pool.primary_available(),
            engine.core.pool.capacity()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn production_path_batches_without_qualifier_and_v2_control_remains_sequential() {
        let normal_dir = TempDir::new("production-normal");
        let control_dir = TempDir::new("production-v2-control");
        let treatment_dir = TempDir::new("production-v2-treatment");
        let normal = build_engine(&normal_dir.path, 4, 8, 8, 4, 2, 0, 24);
        let control = build_engine(&control_dir.path, 4, 8, 8, 4, 2, 0, 24);
        let treatment = build_engine(&treatment_dir.path, 4, 8, 8, 4, 2, 0, 24);
        control
            .enable_gpu_native_demand_source_production_qualification(
                GpuNativeDemandSourceQualificationArm::Control,
            )
            .unwrap();
        treatment
            .enable_gpu_native_demand_source_production_qualification(
                GpuNativeDemandSourceQualificationArm::Treatment,
            )
            .unwrap();

        let normal_residents = run_production_source_set(normal.clone(), vec![2, 0])
            .await
            .unwrap();
        let control_residents = run_production_source_set(control.clone(), vec![2, 0])
            .await
            .unwrap();
        let treatment_residents = run_production_source_set(treatment.clone(), vec![2, 0])
            .await
            .unwrap();
        for id in [2, 0] {
            assert_eq!(normal_residents[&id].data(), control_residents[&id].data());
            assert_eq!(
                normal_residents[&id].data(),
                treatment_residents[&id].data()
            );
        }
        assert_eq!(
            normal
                .production_demand_source_snapshot()
                .production_batch_successes,
            1
        );
        assert_eq!(
            control
                .production_demand_source_snapshot()
                .production_batch_successes,
            0
        );
        assert_eq!(
            treatment
                .production_demand_source_snapshot()
                .production_batch_successes,
            1
        );
        let source = treatment
            .gpu_native_demand_source_qualification_snapshot()
            .unwrap();
        assert!(!source.qualification_only);
        assert!(source.production_demand_source_changed);
        assert!(
            treatment
                .production_demand_source_snapshot()
                .ordinary_production_path_exercised
        );
        assert_eq!(
            normal.core.cache.resident_ids(),
            treatment.core.cache.resident_ids()
        );
        assert_eq!(normal.report().prefetch_completed, 0);
        assert_eq!(treatment.report().prefetch_completed, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn production_all_hit_mixed_one_miss_and_cost_aware_sets_stay_sequential() {
        let dir = TempDir::new("production-sequential-fallbacks");
        let engine = build_engine(&dir.path, 6, 8, 8, 6, 2, 0, 25);
        let mut priming = HashMap::new();
        for id in [0, 1] {
            engine
                .gpu_native_demand_source(id, &mut priming)
                .await
                .unwrap();
        }
        drop(priming);
        let before = engine.report().bytes_read;
        run_production_source_set(engine.clone(), vec![0, 1])
            .await
            .unwrap();
        run_production_source_set(engine.clone(), vec![1, 2])
            .await
            .unwrap();
        run_production_source_set(engine.clone(), vec![3])
            .await
            .unwrap();
        engine.core.cache.set_cost_aware(true);
        run_production_source_set(engine.clone(), vec![4, 5])
            .await
            .unwrap();
        assert_eq!(
            engine.report().bytes_read - before,
            4 * engine.core.storage.config().expert_size as u64
        );
        let snapshot = engine.production_demand_source_snapshot();
        assert_eq!(snapshot.production_batch_successes, 0);
        assert_eq!(snapshot.production_sequential_fallback_mixed_ram, 2);
        assert_eq!(snapshot.production_sequential_fallback_single_item, 1);
        assert_eq!(snapshot.production_sequential_fallback_reservation, 1);
        assert_eq!(snapshot.production_cache_reservation_leaks, 0);
        assert_eq!(snapshot.stale_singleflight_entries, 0);
    }

    #[test]
    fn gpu_native_tiered_residency_is_strictly_opt_in() {
        let dir = TempDir::new("gpu-native-residency-opt-in");
        let engine = build_engine(&dir.path, 4, 8, 8, 2, 1, 1, 17);
        assert!(engine.core.gpu_native_residency.is_none());
        assert!(engine.gpu_native_residency_snapshot().is_none());
    }

    fn rebuild_with_speculator(
        base: &Arc<Engine>,
        spec: Arc<NeuralSpeculator>,
        top_k: usize,
    ) -> Arc<Engine> {
        Arc::new(
            Engine::new(
                base.core.cache.clone(),
                base.core.pool.clone(),
                base.core.storage.clone(),
                base.core.router.clone(),
                base.core.predictor.clone(),
                base.core.shape,
            )
            .with_speculator(spec, top_k),
        )
    }

    fn rebuild_with_test_gpu(
        base: &Arc<Engine>,
        test_gpu: crate::backend::TestGpuBackend,
        failure_policy: Option<RoutedExpertGpuFailurePolicy>,
        expert_execution_policy: ExpertExecutionPolicy,
        allow_degraded_experts: bool,
    ) -> Arc<Engine> {
        rebuild_with_test_gpu_capacity(
            base,
            test_gpu,
            failure_policy,
            expert_execution_policy,
            allow_degraded_experts,
            64 * 1024,
            base.core.options.dtype,
        )
    }

    fn rebuild_with_test_gpu_capacity(
        base: &Arc<Engine>,
        test_gpu: crate::backend::TestGpuBackend,
        failure_policy: Option<RoutedExpertGpuFailurePolicy>,
        expert_execution_policy: ExpertExecutionPolicy,
        allow_degraded_experts: bool,
        gpu_capacity_bytes: usize,
        dtype: WeightDtype,
    ) -> Arc<Engine> {
        let gpu_expert_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(
            gpu_capacity_bytes,
            0.5,
            16,
        ));
        let backend = Arc::new(crate::backend::BackendBox::TestGpu(test_gpu));
        let context = crate::backend::resolve_execution_context_with(
            crate::backend::ComputeOffload::Hybrid,
            true,
            crate::backend::GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: 16,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: base.core.shape.d_model,
                v_head_dim: base.core.shape.d_model,
                q4_truncation_tolerance: 0,
            },
            crate::backend::RoutedExpertGpuSpec {
                dtype,
                d_model: base.core.shape.d_model,
                d_ff: base.core.shape.d_ff,
            },
            gpu_expert_cache,
            |_| Ok(backend),
        )
        .expect("test hybrid execution context");

        let mut options = base.core.options;
        options.dtype = dtype;
        options.expert_execution_policy = expert_execution_policy;
        options.policy.allow_degraded_experts = allow_degraded_experts;
        let engine = Engine::with_options_and_execution_context(
            base.core.cache.clone(),
            base.core.pool.clone(),
            base.core.storage.clone(),
            base.core.router.clone(),
            base.core.predictor.clone(),
            base.core.shape,
            options,
            context,
        );
        Arc::new(match failure_policy {
            Some(policy) => engine.with_routed_expert_gpu_failure_policy(policy),
            None => engine,
        })
    }

    fn cpu_expert_forward_calls(engine: &Engine) -> u64 {
        engine
            .metrics
            .counters
            .cpu_expert_forward_calls
            .load(Ordering::Relaxed)
    }

    fn rebuild_with_q4_test_gpu(
        base: &Arc<Engine>,
        test_gpu: crate::backend::TestGpuBackend,
    ) -> Arc<Engine> {
        rebuild_with_test_gpu_capacity(
            base,
            test_gpu,
            Some(RoutedExpertGpuFailurePolicy::StrictFailClosed),
            ExpertExecutionPolicy::SequentialExpertsRowParallel,
            false,
            1024 * 1024,
            WeightDtype::Q4_0,
        )
    }

    fn rendered_counter(metrics: &Metrics, name: &str) -> u64 {
        let body = String::from_utf8(metrics.render().expect("render Prometheus metrics"))
            .expect("metrics are UTF-8");
        body.lines()
            .find_map(|line| {
                let (metric, value) = line.split_once(' ')?;
                (metric == name)
                    .then(|| value.parse::<f64>().expect("Prometheus counter value") as u64)
            })
            .unwrap_or_else(|| panic!("missing Prometheus counter {name} in:\n{body}"))
    }

    fn assert_gpu_dispatch_error(
        error: MoeStepError,
        expected_kind: crate::backend::GpuExpertDispatchErrorKind,
        expected_layer: u32,
        expected_expert: u32,
    ) {
        match error {
            MoeStepError::GpuExpertDispatch { source } => {
                assert_eq!(source.kind, expected_kind);
                assert_eq!(source.layer, expected_layer);
                assert_eq!(source.expert_id, expected_expert);
                assert!(!source.detail.is_empty());
            }
            other => panic!("expected typed GPU expert dispatch error, got: {other}"),
        }
    }

    #[test]
    fn ordered_batch_aggregation_preserves_f16_values_f32_weights_and_zero_slots() {
        let outputs: Vec<half::f16> = (0..16)
            .map(|value| half::f16::from_f32(value as f32 / 4.0))
            .collect();
        let weights = [0.5, 0.0, -0.25, 1.0, 0.125, -0.5, 0.25, 0.75];
        let mut actual = Vec::new();
        assert!(accumulate_ordered_f16_outputs(&outputs, &weights, 2, &mut actual));
        assert_eq!(actual, [3.625, 4.09375]);
        assert!(!accumulate_ordered_f16_outputs(&outputs[..15], &weights, 2, &mut actual));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn warm_q4_top8_batch_preserves_duplicates_zero_weights_and_accounting() {
        let dir = TempDir::new("q4-layer-batch");
        let base = build_engine(&dir.path, 8, 32, 32, 8, 8, 0, 0xB47C);
        let gpu = crate::backend::TestGpuBackend::batch_success(0.25);
        let engine = rebuild_with_q4_test_gpu(&base, gpu.clone());
        let experts = [1, 1, 2, 3, 4, 5, 6, 7];
        let weights = [0.5, 0.0, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.015625];
        let mut out = Vec::new();
        engine.moe_step_weighted_into_with_timing(
            0, 0, &vec![1.0; 32], &experts, &weights, &mut out, None,
        ).await.unwrap();
        assert_eq!(out, vec![0.25; 32]);
        assert_eq!((gpu.batch_calls(), gpu.expert_calls()), (1, 8));
        assert_eq!(engine.routed_expert_execution_snapshot(), RoutedExpertExecutionSnapshot {
            selected_routed_experts: 8, gpu_dispatch_attempts: 8,
            gpu_dispatch_successes: 8, ..RoutedExpertExecutionSnapshot::default()
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ineligible_q4_batch_uses_unchanged_scalar_dispatch() {
        let dir = TempDir::new("q4-layer-batch-scalar");
        let base = build_engine(&dir.path, 8, 32, 32, 8, 8, 0, 0x5CA1A2);
        let gpu = crate::backend::TestGpuBackend::success(0.5);
        let engine = rebuild_with_q4_test_gpu(&base, gpu.clone());
        let mut out = Vec::new();
        engine.moe_step_weighted_into_with_timing(
            0, 0, &vec![1.0; 32], &[0, 1, 2, 3, 4, 5, 6, 7], &[0.125; 8], &mut out, None,
        ).await.unwrap();
        assert_eq!(out, vec![0.5; 32]);
        assert_eq!((gpu.batch_calls(), gpu.expert_calls()), (0, 8));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn started_strict_q4_batch_failure_is_counted_once_without_scalar_replay() {
        let dir = TempDir::new("q4-layer-batch-failure");
        let base = build_engine(&dir.path, 8, 32, 32, 8, 8, 0, 0xFA17);
        let gpu = crate::backend::TestGpuBackend::batch_failure(
            crate::backend::GpuExpertDispatchErrorKind::ReadbackMap,
        );
        let engine = rebuild_with_q4_test_gpu(&base, gpu.clone());
        let mut out = Vec::new();
        let error = engine.moe_step_weighted_into_with_timing(
            0, 4, &vec![1.0; 32], &[0, 1, 2, 3, 4, 5, 6, 7], &[0.125; 8], &mut out, None,
        ).await.unwrap_err();
        assert_gpu_dispatch_error(error, crate::backend::GpuExpertDispatchErrorKind::ReadbackMap, 4, 0);
        assert_eq!((gpu.batch_calls(), gpu.expert_calls()), (1, 8));
        assert_eq!(engine.routed_expert_execution_snapshot(), RoutedExpertExecutionSnapshot {
            selected_routed_experts: 8, gpu_dispatch_attempts: 8, gpu_dispatch_failures: 8,
            ..RoutedExpertExecutionSnapshot::default()
        });
    }

    #[tokio::test]
    async fn engine_consumes_the_exact_resolved_hybrid_context() {
        let dir = TempDir::new("execution-context-identity");
        let base = build_engine(&dir.path, 4, 8, 16, 2, 2, 0, 7);
        let gpu_backend = Arc::new(crate::backend::BackendBox::TestGpu(
            crate::backend::TestGpuBackend::success(1.0),
        ));
        let expected_backend = gpu_backend.clone();
        let gpu_expert_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(1024, 0.5, 16));
        let expected_gpu_expert_cache = gpu_expert_cache.clone();
        let context = crate::backend::resolve_execution_context_with(
            crate::backend::ComputeOffload::Hybrid,
            true,
            crate::backend::GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: 16,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 8,
                v_head_dim: 8,
                q4_truncation_tolerance: 0,
            },
            crate::backend::RoutedExpertGpuSpec {
                dtype: WeightDtype::F32,
                d_model: 8,
                d_ff: 16,
            },
            gpu_expert_cache,
            |_| Ok(gpu_backend),
        )
        .unwrap();

        let mut engine = Engine::with_options_and_execution_context(
            base.core.cache.clone(),
            base.core.pool.clone(),
            base.core.storage.clone(),
            base.core.router.clone(),
            base.core.predictor.clone(),
            base.core.shape,
            base.core.options.clone(),
            context.clone(),
        );

        assert!(Arc::ptr_eq(engine.execution_context(), &context));
        assert_eq!(engine.execution_context().id(), context.plan().context_id());
        assert!(Arc::ptr_eq(
            engine.routed_expert_backend(),
            &expected_backend
        ));
        assert!(Arc::ptr_eq(
            engine.execution_context().gpu_expert_cache(),
            &expected_gpu_expert_cache
        ));
        engine.install_gpu_cache();
        assert!(Arc::ptr_eq(
            engine.core.gpu_cache.as_ref().unwrap(),
            &expected_gpu_expert_cache
        ));

        let other = crate::backend::cpu_execution_context();
        assert_ne!(engine.execution_context().id(), other.id());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn strict_gpu_expert_success_never_enters_cpu_or_fallback_accounting() {
        let dir = TempDir::new("strict-gpu-success");
        let base = build_engine(&dir.path, 8, 16, 32, 4, 2, 1, 0x301);
        let test_gpu = crate::backend::TestGpuBackend::success(0.25);
        let engine = rebuild_with_test_gpu(
            &base,
            test_gpu.clone(),
            Some(RoutedExpertGpuFailurePolicy::StrictFailClosed),
            ExpertExecutionPolicy::SequentialExpertsRowParallel,
            false,
        );
        let hidden = crate::inference::synth_hidden_state(0, 16, 0x301);
        let output = engine
            .moe_step(0, 3, &hidden, &[2])
            .await
            .expect("injected GPU success must complete");
        let expected = half::f16::from_f32(0.25).to_f32();
        assert_eq!(output.len(), 1);
        assert!(output[0].iter().all(|&value| value == expected));
        assert_eq!(test_gpu.expert_calls(), 1);
        assert_eq!(cpu_expert_forward_calls(&engine), 0);
        assert_eq!(engine.report().gpu_cpu_fallbacks, 0);
        let gpu_cache = engine.execution_context().gpu_expert_cache();
        assert!(gpu_cache.current_admission(2).is_some());
        assert_eq!(gpu_cache.anchor_len(), 0);
        assert_eq!(gpu_cache.lru_len(), 1);
        assert_eq!(
            engine.routed_expert_execution_snapshot(),
            RoutedExpertExecutionSnapshot {
                selected_routed_experts: 1,
                gpu_dispatch_attempts: 1,
                gpu_dispatch_successes: 1,
                gpu_dispatch_failures: 0,
                cpu_routed_expert_dispatches: 0,
                gpu_cpu_fallbacks: 0,
                degraded_expert_substitutions: 0,
            }
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn demand_wins_hot_promotion_race_keeps_internal_and_prometheus_equal() {
        let dir = TempDir::new("hot-promotion-demand-wins");
        let base = build_engine(&dir.path, 4, 8, 16, 4, 1, 0, 0x30A);
        let id = 2;
        base.warm_with(&[id]).await.expect("warm RAM resident");
        let test_gpu = crate::backend::TestGpuBackend::success(0.25);
        let rebuilt = rebuild_with_test_gpu(
            &base,
            test_gpu,
            Some(RoutedExpertGpuFailurePolicy::StrictFailClosed),
            ExpertExecutionPolicy::SequentialExpertsRowParallel,
            false,
        );
        let metrics = Metrics::new();
        let engine = match Arc::try_unwrap(rebuilt) {
            Ok(engine) => engine.with_metrics(metrics.clone()),
            Err(_) => panic!("test owns the sole rebuilt engine Arc"),
        };
        let resident = engine.core.cache.get(id).expect("RAM resident");
        let gpu = engine.execution_context().gpu_expert_cache();

        assert!(gpu.claim_promotion(id, 16));
        engine
            .demand_admit_resident_to_gpu(0, resident.as_ref())
            .expect("foreground demand admission");
        let demand_admission = gpu.current_admission(id).expect("demand LRU admission");
        let generation = demand_admission.generation();

        assert_eq!(
            Engine::complete_background_gpu_promotion(
                gpu,
                Some(&metrics),
                id,
                resident.as_ref(),
                engine.core.options.dtype,
            ),
            GpuHotPromotionOutcome::MovedLruToAnchor
        );
        let current = gpu.current_admission(id).expect("graduated admission");
        assert_eq!(current.generation(), generation);
        assert!(Arc::ptr_eq(current.resident(), demand_admission.resident()));
        assert_eq!(gpu.promotions(), 2, "demand install + Anchor graduation");
        assert_eq!(rendered_counter(&metrics, "mer_promotions_total"), 2);
        assert_eq!(gpu.anchor_len(), 1);
        assert_eq!(gpu.lru_len(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn background_wins_hot_promotion_race_keeps_demand_recheck_noop() {
        let dir = TempDir::new("hot-promotion-background-wins");
        let base = build_engine(&dir.path, 4, 8, 16, 4, 1, 0, 0x30B);
        let id = 2;
        base.warm_with(&[id]).await.expect("warm RAM resident");
        let test_gpu = crate::backend::TestGpuBackend::success(0.25);
        let rebuilt = rebuild_with_test_gpu(
            &base,
            test_gpu,
            Some(RoutedExpertGpuFailurePolicy::StrictFailClosed),
            ExpertExecutionPolicy::SequentialExpertsRowParallel,
            false,
        );
        let metrics = Metrics::new();
        let engine = match Arc::try_unwrap(rebuilt) {
            Ok(engine) => engine.with_metrics(metrics.clone()),
            Err(_) => panic!("test owns the sole rebuilt engine Arc"),
        };
        let resident = engine.core.cache.get(id).expect("RAM resident");
        let gpu = engine.execution_context().gpu_expert_cache();

        assert_eq!(
            gpu.demand_admission_preflight(id, resident.data().len()),
            Ok(crate::expert_cache::GpuDemandAdmissionPreflight::NeedsPayload)
        );
        assert!(gpu.claim_promotion(id, 16));
        assert_eq!(
            Engine::complete_background_gpu_promotion(
                gpu,
                Some(&metrics),
                id,
                resident.as_ref(),
                engine.core.options.dtype,
            ),
            GpuHotPromotionOutcome::InstalledAnchor
        );
        let background_admission = gpu.current_admission(id).expect("Anchor admission");
        let generation = background_admission.generation();

        engine
            .demand_admit_resident_to_gpu(0, resident.as_ref())
            .expect("demand recheck must observe background admission");
        let current = gpu.current_admission(id).expect("current admission");
        assert_eq!(current.generation(), generation);
        assert!(Arc::ptr_eq(current.resident(), background_admission.resident()));
        assert_eq!(gpu.promotions(), 1);
        assert_eq!(rendered_counter(&metrics, "mer_promotions_total"), 1);
        assert_eq!(gpu.anchor_len(), 1);
        assert_eq!(gpu.lru_len(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn serving_demand_admitted_hot_expert_graduates_to_anchor() {
        let dir = TempDir::new("serving-demand-hot-graduation");
        let base = build_engine(&dir.path, 4, 8, 16, 4, 1, 0, 0x30C);
        let test_gpu = crate::backend::TestGpuBackend::success(0.25);
        let gpu = Arc::new(crate::expert_cache::GpuExpertCache::new(64 * 1024, 0.5, 3));
        let backend = Arc::new(crate::backend::BackendBox::TestGpu(test_gpu.clone()));
        let context = crate::backend::resolve_execution_context_with(
            crate::backend::ComputeOffload::Hybrid,
            true,
            crate::backend::GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: 16,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: base.core.shape.d_model,
                v_head_dim: base.core.shape.d_model,
                q4_truncation_tolerance: 0,
            },
            crate::backend::RoutedExpertGpuSpec {
                dtype: base.core.options.dtype,
                d_model: base.core.shape.d_model,
                d_ff: base.core.shape.d_ff,
            },
            gpu.clone(),
            |_| Ok(backend),
        )
        .expect("test Hybrid context");
        let mut engine = Engine::with_options_and_execution_context(
            base.core.cache.clone(),
            base.core.pool.clone(),
            base.core.storage.clone(),
            base.core.router.clone(),
            base.core.predictor.clone(),
            base.core.shape,
            base.core.options,
            context,
        );
        assert_eq!(
            engine.routed_expert_gpu_failure_policy(),
            RoutedExpertGpuFailurePolicy::ServingCpuFallback
        );
        let mut rx = engine.install_gpu_cache_for_test(gpu.clone());
        let engine = Arc::new(engine);
        let id = 2;
        let hidden = crate::inference::synth_hidden_state(0, 8, 0x30C);

        for token in 0..3 {
            engine
                .moe_step(token, 0, &hidden, &[id])
                .await
                .expect("serving demand GPU execution");
        }
        let (queued_id, resident) = rx.try_recv().expect("threshold promotion request");
        assert_eq!(queued_id, id);
        assert!(rx.try_recv().is_err(), "only one threshold request is queued");
        let generation = gpu.current_generation(id).expect("demand LRU generation");
        assert_eq!(gpu.anchor_len(), 0);
        assert_eq!(gpu.lru_len(), 1);

        assert_eq!(
            Engine::complete_background_gpu_promotion(
                gpu.as_ref(),
                None,
                id,
                resident.as_ref(),
                engine.core.options.dtype,
            ),
            GpuHotPromotionOutcome::MovedLruToAnchor
        );
        assert_eq!(gpu.current_generation(id), Some(generation));
        assert_eq!(gpu.anchor_len(), 1);
        assert_eq!(gpu.lru_len(), 0);
        assert_eq!(test_gpu.expert_calls(), 3);
        assert_eq!(cpu_expert_forward_calls(&engine), 0);
        assert_eq!(engine.report().gpu_cpu_fallbacks, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn strict_gpu_demand_admission_failure_never_reaches_backend_or_cpu() {
        let dir = TempDir::new("strict-gpu-demand-admission-failure");
        let base = build_engine(&dir.path, 8, 16, 32, 4, 2, 1, 0x309);
        let test_gpu = crate::backend::TestGpuBackend::success(0.25);
        let engine = rebuild_with_test_gpu_capacity(
            &base,
            test_gpu.clone(),
            Some(RoutedExpertGpuFailurePolicy::StrictFailClosed),
            ExpertExecutionPolicy::SequentialExpertsRowParallel,
            false,
            4 * 1024,
            base.core.options.dtype,
        );
        let hidden = crate::inference::synth_hidden_state(0, 16, 0x309);
        let error = engine
            .moe_step(0, 5, &hidden, &[2])
            .await
            .expect_err("oversized foreground demand must fail closed");
        assert_gpu_dispatch_error(
            error,
            crate::backend::GpuExpertDispatchErrorKind::ResidencyMiss,
            5,
            2,
        );
        assert_eq!(test_gpu.expert_calls(), 0);
        assert_eq!(cpu_expert_forward_calls(&engine), 0);
        assert_eq!(engine.report().gpu_cpu_fallbacks, 0);
        assert!(engine
            .execution_context()
            .gpu_expert_cache()
            .current_admission(2)
            .is_none());
        assert_eq!(
            engine.routed_expert_execution_snapshot(),
            RoutedExpertExecutionSnapshot {
                selected_routed_experts: 1,
                gpu_dispatch_attempts: 1,
                gpu_dispatch_successes: 0,
                gpu_dispatch_failures: 1,
                cpu_routed_expert_dispatches: 0,
                gpu_cpu_fallbacks: 0,
                degraded_expert_substitutions: 0,
            }
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn strict_gpu_expert_failures_are_typed_and_never_recover_on_cpu() {
        use crate::backend::GpuExpertDispatchErrorKind as Kind;
        for (label, kind) in [
            ("residency", Kind::ResidencyMiss),
            ("physical-capacity", Kind::PhysicalCapacity),
            ("upload", Kind::Upload),
            ("validation", Kind::ValidationDispatch),
            ("submission", Kind::Submission),
            ("readback-channel", Kind::ReadbackChannel),
            ("readback-map", Kind::ReadbackMap),
            ("device-loss", Kind::DeviceLost),
        ] {
            let dir = TempDir::new(label);
            let base = build_engine(&dir.path, 8, 16, 32, 4, 2, 1, 0x302);
            let test_gpu = crate::backend::TestGpuBackend::failure(kind);
            let engine = rebuild_with_test_gpu(
                &base,
                test_gpu.clone(),
                Some(RoutedExpertGpuFailurePolicy::StrictFailClosed),
                ExpertExecutionPolicy::SequentialExpertsRowParallel,
                false,
            );
            let hidden = crate::inference::synth_hidden_state(0, 16, 0x302);
            let error = engine
                .moe_step(0, 7, &hidden, &[3])
                .await
                .expect_err("strict injected GPU failure must fail the MoE step");
            assert_gpu_dispatch_error(error, kind, 7, 3);
            assert_eq!(test_gpu.expert_calls(), 1, "{label}");
            assert_eq!(cpu_expert_forward_calls(&engine), 0, "{label}");
            let report = engine.report();
            assert_eq!(report.gpu_cpu_fallbacks, 0, "{label}");
            assert_eq!(report.degraded_expert_substitutions, 0, "{label}");
            assert_eq!(
                engine.routed_expert_execution_snapshot(),
                RoutedExpertExecutionSnapshot {
                    selected_routed_experts: 1,
                    gpu_dispatch_attempts: 1,
                    gpu_dispatch_successes: 0,
                    gpu_dispatch_failures: 1,
                    cpu_routed_expert_dispatches: 0,
                    gpu_cpu_fallbacks: 0,
                    degraded_expert_substitutions: 0,
                },
                "{label}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn serving_gpu_failure_falls_back_once_and_returns_cpu_result() {
        let dir = TempDir::new("serving-gpu-fallback");
        let base = build_engine(&dir.path, 8, 16, 32, 4, 2, 1, 0x303);
        let test_gpu = crate::backend::TestGpuBackend::failure(
            crate::backend::GpuExpertDispatchErrorKind::ValidationDispatch,
        );
        let engine = rebuild_with_test_gpu(
            &base,
            test_gpu.clone(),
            None,
            ExpertExecutionPolicy::SequentialExpertsRowParallel,
            false,
        );
        assert_eq!(
            engine.routed_expert_gpu_failure_policy(),
            RoutedExpertGpuFailurePolicy::ServingCpuFallback
        );
        let hidden = crate::inference::synth_hidden_state(0, 16, 0x303);
        let output = engine
            .moe_step(0, 0, &hidden, &[3])
            .await
            .expect("serving mode must recover on CPU");
        let resident = engine.core.cache.get(3).expect("expert resident after step");
        let expected = dispatch_expert_forward(
            engine.core.options.dtype,
            engine.core.options.use_qmm_for_q4,
            0,
            &resident,
            &hidden,
            engine.core.shape.d_model,
            engine.core.shape.d_ff,
            engine.core.options.policy.expert_size_tolerance(),
            None,
        )
        .expect("reference CPU expert forward")
        .1;
        assert_eq!(output, vec![expected]);
        assert_eq!(test_gpu.expert_calls(), 1);
        assert_eq!(cpu_expert_forward_calls(&engine), 1);
        assert_eq!(engine.report().gpu_cpu_fallbacks, 1);
        assert_eq!(
            engine.routed_expert_execution_snapshot(),
            RoutedExpertExecutionSnapshot {
                selected_routed_experts: 1,
                gpu_dispatch_attempts: 1,
                gpu_dispatch_successes: 0,
                gpu_dispatch_failures: 1,
                cpu_routed_expert_dispatches: 1,
                gpu_cpu_fallbacks: 1,
                degraded_expert_substitutions: 0,
            }
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cpu_plan_never_invokes_gpu_routed_expert_dispatch() {
        let dir = TempDir::new("cpu-plan-no-gpu-dispatch");
        let base = build_engine(&dir.path, 8, 16, 32, 4, 2, 1, 0x304);
        let test_gpu = crate::backend::TestGpuBackend::success(9.0);
        let backend = Arc::new(crate::backend::BackendBox::TestGpu(test_gpu.clone()));
        let gpu_init_calls = Arc::new(AtomicU64::new(0));
        let init_calls = gpu_init_calls.clone();
        let context = crate::backend::resolve_execution_context_with(
            crate::backend::ComputeOffload::Auto,
            true,
            crate::backend::GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: 16,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 16,
                v_head_dim: 16,
                q4_truncation_tolerance: 0,
            },
            crate::backend::RoutedExpertGpuSpec {
                dtype: WeightDtype::F32,
                d_model: 16,
                d_ff: 32,
            },
            Arc::new(crate::expert_cache::GpuExpertCache::new(1024, 0.5, 16)),
            move |_| {
                init_calls.fetch_add(1, Ordering::Relaxed);
                Ok(backend)
            },
        )
        .expect("Auto + strict attention resolves to CPU");
        assert_eq!(context.plan().routed_experts(), crate::backend::ExecutionPlane::Cpu);
        assert_eq!(gpu_init_calls.load(Ordering::Relaxed), 0);
        let engine = Arc::new(Engine::with_options_and_execution_context(
            base.core.cache.clone(),
            base.core.pool.clone(),
            base.core.storage.clone(),
            base.core.router.clone(),
            base.core.predictor.clone(),
            base.core.shape,
            base.core.options,
            context,
        ));
        let hidden = crate::inference::synth_hidden_state(0, 16, 0x304);
        engine
            .moe_step(0, 0, &hidden, &[1])
            .await
            .expect("CPU plan remains unchanged");
        assert_eq!(test_gpu.expert_calls(), 0);
        assert_eq!(cpu_expert_forward_calls(&engine), 1);
        assert_eq!(engine.report().gpu_cpu_fallbacks, 0);
        assert_eq!(
            engine.routed_expert_execution_snapshot(),
            RoutedExpertExecutionSnapshot {
                selected_routed_experts: 1,
                gpu_dispatch_attempts: 0,
                gpu_dispatch_successes: 0,
                gpu_dispatch_failures: 0,
                cpu_routed_expert_dispatches: 1,
                gpu_cpu_fallbacks: 0,
                degraded_expert_substitutions: 0,
            }
        );
        assert!(
            engine.gpu_expert_memory_snapshot().is_none(),
            "CPU plans must not depend on a physical GPU expert registry"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn strict_gpu_plan_with_cpu_runtime_backend_fails_closed() {
        let dir = TempDir::new("strict-plan-runtime-invariant");
        let base = build_engine(&dir.path, 8, 16, 32, 4, 2, 1, 0x305);
        let context = crate::backend::test_gpu_execution_context_unchecked(
            Arc::new(crate::backend::BackendBox::Cpu(
                crate::backend::CandleBackend::new(),
            )),
            crate::backend::RoutedExpertGpuSpec {
                dtype: WeightDtype::F32,
                d_model: 16,
                d_ff: 32,
            },
        );
        let engine = Arc::new(
            Engine::with_options_and_execution_context(
                base.core.cache.clone(),
                base.core.pool.clone(),
                base.core.storage.clone(),
                base.core.router.clone(),
                base.core.predictor.clone(),
                base.core.shape,
                base.core.options,
                context,
            )
            .with_routed_expert_gpu_failure_policy(
                RoutedExpertGpuFailurePolicy::StrictFailClosed,
            ),
        );
        let hidden = crate::inference::synth_hidden_state(0, 16, 0x305);
        let error = engine
            .moe_step(0, 4, &hidden, &[2])
            .await
            .expect_err("runtime/backend mismatch must fail closed");
        assert_gpu_dispatch_error(
            error,
            crate::backend::GpuExpertDispatchErrorKind::RuntimeInvariant,
            4,
            2,
        );
        assert_eq!(cpu_expert_forward_calls(&engine), 0);
        assert_eq!(engine.report().gpu_cpu_fallbacks, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn degraded_expert_policy_cannot_swallow_strict_gpu_failure() {
        let dir = TempDir::new("strict-gpu-beats-degraded");
        let base = build_engine(&dir.path, 8, 16, 32, 4, 2, 1, 0x306);
        let test_gpu = crate::backend::TestGpuBackend::failure(
            crate::backend::GpuExpertDispatchErrorKind::Upload,
        );
        let engine = rebuild_with_test_gpu(
            &base,
            test_gpu,
            Some(RoutedExpertGpuFailurePolicy::StrictFailClosed),
            ExpertExecutionPolicy::SequentialExpertsRowParallel,
            true,
        );
        let hidden = crate::inference::synth_hidden_state(0, 16, 0x306);
        let error = engine
            .moe_step(0, 5, &hidden, &[3])
            .await
            .expect_err("allow_degraded_experts must not swallow strict GPU failure");
        assert_gpu_dispatch_error(
            error,
            crate::backend::GpuExpertDispatchErrorKind::Upload,
            5,
            3,
        );
        assert_eq!(cpu_expert_forward_calls(&engine), 0);
        assert_eq!(engine.report().gpu_cpu_fallbacks, 0);
        assert_eq!(engine.report().degraded_expert_substitutions, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn parallel_top_k_strict_gpu_failure_has_no_cpu_recovery() {
        let dir = TempDir::new("strict-gpu-parallel-top-k");
        let base = build_engine(&dir.path, 8, 16, 32, 4, 2, 1, 0x307);
        let test_gpu = crate::backend::TestGpuBackend::failure(
            crate::backend::GpuExpertDispatchErrorKind::ReadbackMap,
        );
        let engine = rebuild_with_test_gpu(
            &base,
            test_gpu.clone(),
            Some(RoutedExpertGpuFailurePolicy::StrictFailClosed),
            ExpertExecutionPolicy::ParallelExpertsSingleThread,
            false,
        );
        let hidden = crate::inference::synth_hidden_state(0, 16, 0x307);
        let error = engine
            .moe_step(0, 6, &hidden, &[1, 2])
            .await
            .expect_err("one parallel GPU failure must fail the whole step");
        match error {
            MoeStepError::GpuExpertDispatch { source } => {
                assert_eq!(source.kind, crate::backend::GpuExpertDispatchErrorKind::ReadbackMap);
                assert_eq!(source.layer, 6);
                assert!([1, 2].contains(&source.expert_id));
            }
            other => panic!("expected typed parallel GPU failure, got: {other}"),
        }
        assert!(test_gpu.expert_calls() >= 1);
        assert_eq!(cpu_expert_forward_calls(&engine), 0);
        assert_eq!(engine.report().gpu_cpu_fallbacks, 0);
    }

    #[test]
    fn real_inference_error_preserves_typed_gpu_dispatch_source() {
        let gpu_error = crate::backend::GpuExpertDispatchError::new(
            9,
            17,
            crate::backend::GpuExpertDispatchErrorKind::DeviceLost,
            "injected request-boundary device loss",
        );
        let real_error: crate::model::RealInferenceError =
            MoeStepError::GpuExpertDispatch { source: gpu_error.clone() }.into();
        match real_error {
            crate::model::RealInferenceError::MoeStep(
                MoeStepError::GpuExpertDispatch { source },
            ) => assert_eq!(source, gpu_error),
            other => panic!("typed GPU source was flattened at real inference boundary: {other}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn generate_loop_routes_fetches_and_runs_inference() {
        let dir = TempDir::new("gen-integration");
        let num_experts: u32 = 16;
        let top_k = 2;
        let d_model = 32;
        let d_ff = 64;
        let cache_slots = 8;
        let predict_fanout = 2;
        let tokens: u64 = 64;

        let engine = build_engine(
            &dir.path,
            num_experts,
            d_model,
            d_ff,
            cache_slots,
            top_k,
            predict_fanout,
            0xC0FFEE,
        );

        let mut total_hits = 0u64;
        let mut total_misses = 0u64;
        let mut total_bytes = 0u64;
        for t in 0..tokens {
            let s = engine.generate(t).await.expect("generate should succeed");
            total_hits += s.hits;
            total_misses += s.misses;
            total_bytes += s.bytes_read;
        }

        // Every token routes to exactly `top_k` experts, so the cumulative
        // hit + miss count must be exactly `tokens * top_k`.
        assert_eq!(
            total_hits + total_misses,
            tokens * top_k as u64,
            "every routed expert must produce exactly one cache lookup"
        );

        // The first token always misses (cold cache); after that the
        // cache + prefetcher should eventually start serving experts
        // from RAM rather than disk.
        assert!(
            total_hits > 0,
            "expected at least some cache hits across {tokens} tokens"
        );
        assert!(
            total_misses > 0,
            "expected at least some cache misses across {tokens} tokens"
        );
        assert!(
            total_bytes > 0,
            "expected the engine to read bytes from the SSD"
        );

        // The aggregate report mirrors the per-cycle totals on the
        // critical path. `r.bytes_read` may exceed `total_bytes` because
        // background prefetch tasks also contribute to the counter
        // without being part of any single token's stats.
        let r = engine.report();
        assert_eq!(r.hits, total_hits);
        assert_eq!(r.misses, total_misses);
        assert!(
            r.bytes_read >= total_bytes,
            "report bytes_read ({}) must include at least the critical-path bytes ({total_bytes})",
            r.bytes_read
        );
        // The I/O histogram records one sample per *physical* SSD read
        // (a singleflight leader inside `fetch_once`), which is neither an
        // upper nor a lower bound on `total_misses`:
        //   * a foreground miss that joins an in-flight prefetch/leader
        //     becomes a singleflight *follower* — it counts as a miss but
        //     issues no read, so it records no histogram sample; and
        //   * a speculative prefetch leader records a sample for an expert
        //     that was never a foreground miss.
        // The only guaranteed invariant is that the cold-start miss forces
        // at least one physical read, so the histogram is non-empty.
        assert!(
            r.io_count > 0,
            "io histogram must record at least the cold-start read"
        );
        // Latency histograms must have observed at least one sample of each
        // category (compute always, I/O at least once because a cold start
        // forces a miss).
        assert!(r.cycle_p50_us > 0);
        assert!(r.compute_p50_us > 0);
        assert!(r.io_p50_us > 0);

        // Predictor learned something (transitions other than the very first
        // were observed).
        assert!(
            r.predictor_observations > 0,
            "predictor should have logged at least one transition"
        );
        assert_eq!(
            engine.routed_expert_execution_snapshot(),
            RoutedExpertExecutionSnapshot::default(),
            "synthetic Engine::generate traffic must not qualify as real routed execution"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn moe_step_weighted_into_matches_per_expert_combiner() {
        let dir = TempDir::new("moe-weighted-into");
        let num_experts: u32 = 8;
        let d_model = 16;
        let d_ff = 32;
        let seed = 0x574E_ADED_u64;
        let engine = build_engine(
            &dir.path,
            num_experts,
            d_model,
            d_ff,
            /*cache_slots=*/ 8,
            /*top_k=*/ 3,
            /*predict_fanout=*/ 1,
            seed,
        );
        let mut engine_owned = match Arc::try_unwrap(engine) {
            Ok(engine) => engine,
            Err(_) => panic!("test owns the sole engine Arc"),
        };
        engine_owned.core.options.expert_execution_policy =
            ExpertExecutionPolicy::ParallelExpertsSingleThread;
        let engine = Arc::new(engine_owned);

        let hidden = crate::inference::synth_hidden_state(3, d_model, seed);
        let experts = [1, 3, 5];
        let weights = [0.2, 0.3, 0.5];
        let per_expert = engine
            .moe_step(0, /*layer=*/ 0, &hidden, &experts)
            .await
            .expect("per-expert moe_step");
        let expected = combine_outputs(&per_expert, &weights);

        let mut direct = Vec::new();
        engine
            .moe_step_weighted_into_with_timing(
                1,
                /*layer=*/ 0,
                &hidden,
                &experts,
                &weights,
                &mut direct,
                None,
            )
            .await
            .expect("weighted moe_step");

        assert_eq!(direct.len(), expected.len());
        for (i, (&a, &b)) in direct.iter().zip(expected.iter()).enumerate() {
            let tol = 1e-5 * b.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "weighted moe output diverged at {i}: direct={a} expected={b} tol={tol}"
            );
        }
    }

    /// Part A1 (fail-closed inference): under strict production mode
    /// (`allow_degraded_experts = false`, the default), a routed expert
    /// whose SSD read fails after retries fails the whole `moe_step`
    /// with `MoeStepError::ExpertFetch` instead of silently substituting
    /// a zero contribution.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn strict_mode_fails_step_on_expert_fetch_failure() {
        let dir = TempDir::new("strict-fetch-fail");
        let d_model = 16;
        let engine = build_engine(&dir.path, 8, d_model, 32, 4, 2, 1, 0xDEAD);
        // Truncate expert 3's backing file: the pre-opened fd now reads
        // zero bytes, so the fetch fails deterministically after retries.
        let path = engine.core.storage.expert_path(3);
        std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&path)
            .expect("truncate expert file");
        let hidden = crate::inference::synth_hidden_state(0, d_model, 0xDEAD);
        let err = engine
            .moe_step(0, /*layer=*/ 0, &hidden, &[3])
            .await
            .expect_err("strict mode must fail the step on a fetch failure");
        match err {
            MoeStepError::ExpertFetch { expert, layer, .. } => {
                assert_eq!(expert, 3);
                assert_eq!(layer, 0);
            }
            other => panic!("expected ExpertFetch, got: {other}"),
        }
        let report = engine.report();
        assert!(report.expert_read_failures >= 1);
        assert_eq!(
            report.degraded_expert_substitutions, 0,
            "strict mode must never substitute"
        );
    }

    /// Part A1: the development-only degraded mode
    /// (`allow_degraded_experts = true`) preserves the legacy behaviour —
    /// the failed expert is dropped from the mixture (zero contribution)
    /// and every substitution is counted so the run can be marked
    /// non-authoritative.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn degraded_mode_substitutes_zero_and_counts() {
        let dir = TempDir::new("degraded-fetch-fail");
        let d_model = 16;
        let engine = build_engine(&dir.path, 8, d_model, 32, 4, 2, 1, 0xBEEF);
        let mut engine_owned = match Arc::try_unwrap(engine) {
            Ok(engine) => engine,
            Err(_) => panic!("test owns the sole engine Arc"),
        };
        engine_owned.core.options.policy.allow_degraded_experts = true;
        let engine = Arc::new(engine_owned);
        let path = engine.core.storage.expert_path(3);
        std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&path)
            .expect("truncate expert file");
        let hidden = crate::inference::synth_hidden_state(0, d_model, 0xBEEF);
        let outputs = engine
            .moe_step(0, /*layer=*/ 0, &hidden, &[3])
            .await
            .expect("degraded mode must keep the step alive");
        assert_eq!(outputs.len(), 1);
        assert!(
            outputs[0].iter().all(|&v| v == 0.0),
            "failed expert must contribute a zero vector in degraded mode"
        );
        let report = engine.report();
        assert!(report.expert_read_failures >= 1);
        assert!(
            report.degraded_expert_substitutions >= 1,
            "degraded substitutions must be counted"
        );
    }

    /// A6 audit (validation closure, item 3): a routed expert id
    /// outside the storage namespace fails the step with a typed
    /// error in strict mode — no panic, no clamp, no silent miss —
    /// and does not insert anything into the cache.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn strict_mode_fails_step_on_out_of_namespace_expert_id() {
        let dir = TempDir::new("strict-bad-expert-id");
        let d_model = 16;
        let num_experts = 8u32;
        let engine = build_engine(&dir.path, num_experts, d_model, 32, 4, 2, 1, 0xA16);
        let hidden = crate::inference::synth_hidden_state(0, d_model, 0xA16);
        for bad in [num_experts, u32::MAX] {
            let err = engine
                .moe_step(0, /*layer=*/ 0, &hidden, &[bad])
                .await
                .expect_err("out-of-namespace expert id must fail the step");
            assert!(
                matches!(err, MoeStepError::ExpertFetch { expert, .. } if expert == bad),
                "expected ExpertFetch for expert {bad}, got: {err}"
            );
            assert!(
                !engine.core.cache.contains(bad),
                "a failed out-of-namespace fetch must not mutate the cache"
            );
        }
        assert_eq!(engine.report().degraded_expert_substitutions, 0);
    }

    /// Policy audit (validation closure, item 4): the *attention*
    /// fallback policy must not enable expert degradation — an expert
    /// fetch failure still fails the step.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn attention_fallback_policy_does_not_enable_expert_degradation() {
        let dir = TempDir::new("attnfb-no-expert-degrade");
        let d_model = 16;
        let engine = build_engine(&dir.path, 8, d_model, 32, 4, 2, 1, 0xA11);
        let mut engine_owned = match Arc::try_unwrap(engine) {
            Ok(engine) => engine,
            Err(_) => panic!("test owns the sole engine Arc"),
        };
        engine_owned.core.options.policy = crate::inference::RealInferencePolicy {
            allow_nonfinite_attention_fallback: true,
            ..crate::inference::RealInferencePolicy::STRICT
        };
        let engine = Arc::new(engine_owned);
        let path = engine.core.storage.expert_path(3);
        std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&path)
            .expect("truncate expert file");
        let hidden = crate::inference::synth_hidden_state(0, d_model, 0xA11);
        let err = engine
            .moe_step(0, /*layer=*/ 0, &hidden, &[3])
            .await
            .expect_err("attention fallback policy must not tolerate expert failures");
        assert!(matches!(err, MoeStepError::ExpertFetch { .. }));
        assert_eq!(
            engine.report().degraded_expert_substitutions,
            0,
            "no expert substitution may occur under the attention-only policy"
        );
    }

    /// Policy audit (item 4): the truncated-payload policy must not
    /// enable expert degradation either — a failed expert fetch still
    /// fails the step.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn truncated_payload_policy_does_not_enable_expert_degradation() {
        let dir = TempDir::new("trunc-no-expert-degrade");
        let d_model = 16;
        let engine = build_engine(&dir.path, 8, d_model, 32, 4, 2, 1, 0xA12);
        let mut engine_owned = match Arc::try_unwrap(engine) {
            Ok(engine) => engine,
            Err(_) => panic!("test owns the sole engine Arc"),
        };
        engine_owned.core.options.policy = crate::inference::RealInferencePolicy {
            allow_truncated_expert_payloads: true,
            ..crate::inference::RealInferencePolicy::STRICT
        };
        let engine = Arc::new(engine_owned);
        let path = engine.core.storage.expert_path(3);
        std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&path)
            .expect("truncate expert file");
        let hidden = crate::inference::synth_hidden_state(0, d_model, 0xA12);
        let err = engine
            .moe_step(0, /*layer=*/ 0, &hidden, &[3])
            .await
            .expect_err("truncated-payload policy must not tolerate expert fetch failures");
        assert!(matches!(err, MoeStepError::ExpertFetch { .. }));
        assert_eq!(engine.report().degraded_expert_substitutions, 0);
    }

    /// Policy audit (item 4): `EngineReport` must reflect the exact
    /// engine-scoped policy, and defaults must be strict (all false).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn engine_report_reflects_configured_policy_and_defaults_strict() {
        let dir = TempDir::new("policy-report");
        let engine = build_engine(&dir.path, 8, 16, 32, 4, 2, 1, 0xA13);
        let report = engine.report();
        assert_eq!(
            report.inference_policy,
            crate::inference::RealInferencePolicy::STRICT,
            "default engine policy must be strict"
        );
        assert!(!report.inference_policy.any_degraded());

        let mut engine_owned = match Arc::try_unwrap(engine) {
            Ok(engine) => engine,
            Err(_) => panic!("test owns the sole engine Arc"),
        };
        let custom = crate::inference::RealInferencePolicy {
            allow_degraded_experts: true,
            allow_nonfinite_attention_fallback: false,
            allow_truncated_expert_payloads: true,
        };
        engine_owned.core.options.policy = custom;
        let engine = Arc::new(engine_owned);
        assert_eq!(
            engine.report().inference_policy,
            custom,
            "EngineReport must carry the exact configured policy"
        );
        assert!(engine.report().inference_policy.any_degraded());
    }

    /// Policy audit (item 4): the fail-open policy is engine-scoped,
    /// never process-global — a strict engine must fail closed even
    /// while a degraded engine coexists in the same process.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn strict_engine_does_not_inherit_policy_from_degraded_engine() {
        let d_model = 16;
        let dir_degraded = TempDir::new("policy-scope-degraded");
        let engine_degraded = build_engine(&dir_degraded.path, 8, d_model, 32, 4, 2, 1, 0xA14);
        let mut owned = match Arc::try_unwrap(engine_degraded) {
            Ok(engine) => engine,
            Err(_) => panic!("test owns the sole engine Arc"),
        };
        owned.core.options.policy.allow_degraded_experts = true;
        let engine_degraded = Arc::new(owned);

        let dir_strict = TempDir::new("policy-scope-strict");
        let engine_strict = build_engine(&dir_strict.path, 8, d_model, 32, 4, 2, 1, 0xA15);

        // Break expert 3 in both data dirs.
        for engine in [&engine_degraded, &engine_strict] {
            let path = engine.core.storage.expert_path(3);
            std::fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&path)
                .expect("truncate expert file");
        }
        let hidden = crate::inference::synth_hidden_state(0, d_model, 0xA14);

        // The degraded engine substitutes; the strict engine fails —
        // in either construction/execution order.
        engine_degraded
            .moe_step(0, 0, &hidden, &[3])
            .await
            .expect("degraded engine substitutes");
        let err = engine_strict
            .moe_step(0, 0, &hidden, &[3])
            .await
            .expect_err("strict engine must fail closed regardless of the degraded engine");
        assert!(matches!(err, MoeStepError::ExpertFetch { .. }));
        assert!(engine_degraded.report().degraded_expert_substitutions >= 1);
        assert_eq!(
            engine_strict.report().degraded_expert_substitutions,
            0,
            "the strict engine's telemetry must be unaffected by the degraded engine"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn warm_with_preloads_experts_into_cache() {
        // Mirrors the spec's "router selects Expert ID 3 and 7" warm-up.
        let dir = TempDir::new("gen-warm");
        let num_experts: u32 = 8;
        let engine = build_engine(&dir.path, num_experts, 16, 32, 4, 2, 1, 0xBEEF);

        engine.warm_with(&[3, 7]).await.expect("warm fetch");

        // `warm_with` reads through `fetch`, which doesn't bump the
        // hit/miss/bytes counters (those track router-driven `generate`
        // traffic only). The observable side-effect is that both warmed
        // experts are now resident in the cache.
        let r = engine.report();
        assert_eq!(r.hits, 0);
        assert_eq!(r.misses, 0);
        assert!(engine.core.cache.contains(3));
        assert!(engine.core.cache.contains(7));

        // Subsequent generate calls now have warmed slots to hit.
        let _ = engine.generate(0).await.expect("generate should succeed");
        // After at least one token, the per-token cycle histogram must
        // have recorded a sample.
        let r = engine.report();
        assert!(r.cycle_p50_us > 0);
    }

    /// Gist Phase 1 — SSD Read De-Duplication.
    ///
    /// Drive many concurrent `fetch_with_retry` calls against the same
    /// uncached expert id and assert that the engine performed
    /// **exactly one** disk read — directly observable as the
    /// `bytes_read` counter equalling one expert's worth of bytes
    /// (instead of N × that). Both the in-flight singleflight *and*
    /// the cache-hit fast path satisfy this property: a follower
    /// either parks on the leader's Notify or, if the leader has
    /// already finished, returns from the cache check before
    /// touching the storage layer. Either way the disk is read
    /// once. With synthetic local files the leader's read completes
    /// in microseconds, but the `bytes_read` invariant holds for
    /// any I/O latency.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetch_with_retry_deduplicates_concurrent_reads() {
        let dir = TempDir::new("gen-singleflight");
        let num_experts: u32 = 8;
        let engine = build_engine(&dir.path, num_experts, 16, 32, 8, 2, 1, 0xF11F);
        // Sanity: nothing resident yet, no bytes read.
        assert!(!engine.core.cache.contains(5));
        assert_eq!(engine.report().bytes_read, 0);
        let expert_size = engine.core.pool.buffer_size() as u64;

        const N: usize = 32;
        let barrier = Arc::new(tokio::sync::Barrier::new(N));
        let mut handles = Vec::with_capacity(N);
        for _ in 0..N {
            let e = engine.clone();
            let b = barrier.clone();
            handles.push(tokio::spawn(async move {
                b.wait().await;
                e.fetch_with_retry(5).await.expect("fetch")
            }));
        }
        for h in handles {
            let _ = h.await.unwrap();
        }

        // The decisive invariant: even with 32 concurrent callers,
        // the SSD must have served exactly one expert's worth of
        // bytes. Without the singleflight + cache fast-path
        // combination, this would be N × expert_size.
        let r = engine.report();
        assert_eq!(
            r.bytes_read, expert_size,
            "expected exactly one disk read of {expert_size} bytes; got {}",
            r.bytes_read,
        );
        assert!(engine.core.cache.contains(5));
    }

    /// Regression for F1.4: a caller that starts as a follower must
    /// re-contend and succeed after being notified by a failed leader
    /// that did not populate the cache.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fetch_with_retry_follower_recontends_after_leader_failure() {
        let dir = TempDir::new("gen-singleflight-recontend");
        let num_experts: u32 = 8;
        let engine = build_engine(&dir.path, num_experts, 16, 32, 8, 2, 1, 0xF11E);
        let expert_size = engine.core.pool.buffer_size() as u64;
        assert!(!engine.core.cache.contains(5));

        // Seed an in-flight entry so this call must enter the follower
        // path first. We then simulate a failing leader by removing
        // the entry and notifying waiters without filling the cache.
        let notify = Arc::new(Notify::new());
        engine.core.in_flight.insert(5, notify.clone());

        let e = engine.clone();
        let follower = tokio::spawn(async move { e.fetch_with_retry(5).await });
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        engine.core.in_flight.remove(&5);
        notify.notify_waiters();

        let _ = follower
            .await
            .expect("join")
            .expect("follower should re-contend and fetch after failed leader");
        assert!(engine.core.cache.contains(5));
        assert_eq!(engine.report().bytes_read, expert_size);
        assert!(!engine.core.in_flight.contains_key(&5));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cache_cap_bounds_residency_under_load() {
        // The engine must never let more than `cache_slots` experts be
        // resident at once, even under heavy churn. Pick num_experts >>
        // cache_slots to force eviction on most tokens.
        let dir = TempDir::new("gen-evict");
        let num_experts: u32 = 32;
        let cache_slots = 4;
        let engine = build_engine(&dir.path, num_experts, 16, 32, cache_slots, 2, 2, 7);

        for t in 0..50 {
            let _ = engine.generate(t).await.expect("generate should succeed");
            // Residency must NEVER exceed the configured cache capacity,
            // even mid-stream — this is the actual invariant the test
            // name promises. Asserting after every token catches a class
            // of regressions where the cache temporarily holds N+1
            // entries in between an insert and an eviction.
            assert!(
                engine.core.cache.resident_ids().len() <= cache_slots,
                "cache residency {} exceeded capacity {} at token {t}",
                engine.core.cache.resident_ids().len(),
                cache_slots
            );
            assert!(
                engine.core.cache.len() <= cache_slots,
                "cache.len() {} exceeded capacity {} at token {t}",
                engine.core.cache.len(),
                cache_slots
            );
        }
        let r = engine.report();
        assert_eq!(r.cache_capacity, cache_slots);
        assert!(
            engine.core.cache.resident_ids().len() <= cache_slots,
            "post-stream residency {} exceeded capacity {}",
            engine.core.cache.resident_ids().len(),
            cache_slots
        );
        // Misses dominate when cache_slots is small relative to working set.
        assert!(
            r.misses > r.hits / 2,
            "expected eviction churn to produce many misses"
        );
    }

    // ----------- Locality / Speculator / Union-Fetch tests ----------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn engine_with_locality_monitor_pins_hot_experts() {
        // Build an engine with a tight hot threshold so any expert
        // routed twice in the recent window enters the hot set, and
        // verify that those experts get pinned in the cache.
        let dir = TempDir::new("locality-pin");
        let num_experts: u32 = 8;
        let top_k = 2;
        let d_model = 16;
        let d_ff = 32;
        let cache_slots = 6;
        let predict_fanout = 1;

        let engine = build_engine(
            &dir.path,
            num_experts,
            d_model,
            d_ff,
            cache_slots,
            top_k,
            predict_fanout,
            0x10CA117F,
        );
        // Re-wrap with a locality monitor. We drop the previous Arc
        // and rebuild via the same helpers; the cleanest way is to
        // unwrap and rebuild — the helper returns `Arc<Engine>` so
        // we mutate via a fresh constructor instead.
        let engine = {
            // SAFETY: tests own the only Arc reference at this point.
            let cache = engine.core.cache.clone();
            let pool = engine.core.pool.clone();
            let storage = engine.core.storage.clone();
            let router = engine.core.router.clone();
            let predictor = engine.core.predictor.clone();
            let shape = engine.core.shape;
            let monitor = Arc::new(LocalityMonitor::new(num_experts, /*window=*/ 16));
            // Threshold of 0.05 ⇒ any id observed at least once in
            // the 16-slot window is "hot" — easy to trip.
            Arc::new(
                Engine::new(cache, pool, storage, router, predictor, shape)
                    .with_locality_monitor(monitor.clone(), 0.05),
            )
        };
        // Drive a few tokens; the synthetic router routes deterministically,
        // so after several tokens the locality monitor will see repeated
        // ids and start pinning them.
        for t in 0..32u64 {
            let _ = engine.generate(t).await.expect("generate should succeed");
        }
        let pinned = engine.core.cache.pinned_count();
        assert!(
            pinned > 0,
            "locality monitor should have pinned at least one hot expert; got {pinned}"
        );
        // Telemetry must show non-zero locality observations.
        let tele = engine.predictive_telemetry();
        assert!(
            tele.locality_hits + tele.locality_misses > 0,
            "expected locality counters to fire; got {:?}",
            tele
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn engine_with_speculator_records_accuracy_telemetry() {
        let dir = TempDir::new("spec-accuracy");
        let num_experts: u32 = 8;
        let top_k = 2;
        let d_model = 16;
        let d_ff = 32;
        let cache_slots = 6;
        let predict_fanout = 1;

        let engine = build_engine(
            &dir.path,
            num_experts,
            d_model,
            d_ff,
            cache_slots,
            top_k,
            predict_fanout,
            0x5EEEEDED,
        );
        let engine = {
            let cache = engine.core.cache.clone();
            let pool = engine.core.pool.clone();
            let storage = engine.core.storage.clone();
            let router = engine.core.router.clone();
            let predictor = engine.core.predictor.clone();
            let shape = engine.core.shape;
            let spec = Arc::new(NeuralSpeculator::new(d_model, 32, num_experts, 0xABCD));
            Arc::new(
                Engine::new(cache, pool, storage, router, predictor, shape)
                    .with_speculator(spec, top_k),
            )
        };
        for t in 0..50u64 {
            let _ = engine.generate(t).await.expect("generate should succeed");
        }
        let tele = engine.predictive_telemetry();
        assert!(
            tele.speculator_hits + tele.speculator_misses > 0,
            "speculator counters should be non-zero after 50 tokens; got {:?}",
            tele
        );
        assert!(tele.speculator_accuracy >= 0.0 && tele.speculator_accuracy <= 1.0);
    }

    #[test]
    fn speculator_prediction_is_recorded_before_current_sample_training() {
        let dir = TempDir::new("spec-before-train");
        let num_experts: u32 = 4;
        let d_model = 4usize;
        let d_ff = 8usize;
        let seed = 0xB4E0_1ABE;
        let base = build_engine(&dir.path, num_experts, d_model, d_ff, 4, 1, 1, seed);
        let spec = Arc::new(NeuralSpeculator::new(d_model, 16, num_experts, seed));
        let hidden = vec![1.0f32, -0.5, 0.25, -0.125];

        for _ in 0..400 {
            spec.train_step(&hidden, &[0], 0.1);
        }
        assert_eq!(spec.predict_topk(&hidden, 1), vec![0]);
        let train_steps_before = spec.train_steps_for_test();

        let ((engine, preds), queued) =
            NeuralSpeculator::capture_queued_train_for_test(&spec, || {
                let engine = rebuild_with_speculator(&base, spec.clone(), 1);
                let preds = engine.speculator_predict_and_train(&hidden, &[1], None);
                (engine, preds)
            });
        assert_eq!(
            preds,
            vec![0],
            "prediction should use weights from before the current target is queued"
        );
        assert_eq!(
            spec.train_steps_for_test(),
            train_steps_before,
            "current sample must not be trained synchronously on the caller thread"
        );
        assert_eq!(spec.predict_topk(&hidden, 1), vec![0]);
        assert_eq!(queued.x, hidden);
        assert_eq!(queued.actual_top_k, vec![1]);
        assert_eq!(queued.lr, NeuralSpeculator::DEFAULT_LR);

        let tele = engine.predictive_telemetry();
        assert_eq!(tele.speculator_top1_matches, 0);
        assert_eq!(tele.speculator_top1_total, 1);
        assert_eq!(tele.speculator_hits, 0);
        assert_eq!(tele.speculator_misses, 1);
    }

    #[test]
    fn speculator_dmodel_mismatch_returns_empty_and_counts_disabled() {
        let dir = TempDir::new("spec-dmodel-mismatch");
        let num_experts: u32 = 4;
        let d_model = 4usize;
        let d_ff = 8usize;
        let seed = 0xD15A_B1ED;
        let base = build_engine(&dir.path, num_experts, d_model, d_ff, 4, 1, 1, seed);
        let spec = Arc::new(NeuralSpeculator::new(d_model + 1, 16, num_experts, seed));
        let metrics = Metrics::new();
        let engine = Arc::new(
            Engine::new(
                base.core.cache.clone(),
                base.core.pool.clone(),
                base.core.storage.clone(),
                base.core.router.clone(),
                base.core.predictor.clone(),
                base.core.shape,
            )
            .with_metrics(metrics.clone())
            .with_speculator(spec, 1),
        );

        let preds = engine.speculator_predict_and_train(&[0.0f32; 4], &[0], None);
        assert!(preds.is_empty());

        let report = engine.report();
        assert_eq!(report.speculator_dmodel_mismatch, 1);
        let tele = engine.predictive_telemetry();
        assert_eq!(tele.speculator_top1_total, 0);
        assert_eq!(tele.speculator_hits + tele.speculator_misses, 0);
        let prom = String::from_utf8(metrics.render().unwrap()).unwrap();
        assert!(
            prom.contains("mer_speculator_evaluations_total 0"),
            "d_model mismatch must not count as a valid top-1 evaluation:\n{prom}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn engine_predictive_telemetry_reports_ssd_stall() {
        let dir = TempDir::new("ssd-stall");
        let num_experts: u32 = 8;
        let top_k = 2;
        let d_model = 16;
        let d_ff = 32;
        // Tiny cache so we must take SSD misses.
        let cache_slots = 2;
        let predict_fanout = 1;
        let engine = build_engine(
            &dir.path,
            num_experts,
            d_model,
            d_ff,
            cache_slots,
            top_k,
            predict_fanout,
            0xDEADBEEF,
        );
        for t in 0..16u64 {
            let _ = engine.generate(t).await.expect("generate should succeed");
        }
        let tele = engine.predictive_telemetry();
        // With a 2-slot cache and 8 experts at top-k=2, we expect to
        // pay for at least *some* SSD stall.
        assert!(
            tele.ssd_stall_us > 0,
            "expected non-zero ssd stall; got {tele:?}"
        );
    }

    /// End-to-end smoke test (the gist's "e2e integration test"
    /// production-readiness item). Builds the full SSD-streamed
    /// expert pipeline against synthetic weights, runs N tokens
    /// through `Engine::generate`, and checks the deterministic
    /// conservation laws that any healthy run must satisfy:
    ///
    ///   * total expert fetches = `top_k * num_tokens` (no router
    ///     drop or double-fetch)
    ///   * prefetch hits never exceed total fetches
    ///   * no expert read failures on synthetic data
    ///
    /// We deliberately do **not** hash per-token `hits` vs `misses` —
    /// that ratio depends on background-prefetcher timing relative
    /// to the synchronous fetch loop and is non-deterministic by
    /// design. For decoded-token-stream determinism see
    /// `batch_scheduler::tests::step_registered_matches_direct_step`,
    /// which exercises the real `RealModel.step` path.
    ///
    /// Marked `#[ignore]` so it doesn't run in the default `cargo
    /// test` invocation. Invoke with
    /// `cargo test --release -- --ignored e2e` to exercise.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "e2e — run with `cargo test --release -- --ignored e2e`"]
    async fn e2e_engine_runs_a_full_token_stream() {
        let dir = TempDir::new("e2e");
        const TOP_K: usize = 2;
        const N: u64 = 32;
        let engine = build_engine(
            &dir.path, /*num_experts=*/ 8, /*d_model=*/ 16, /*d_ff=*/ 32,
            /*cache_slots=*/ 4, TOP_K, /*predict_fanout=*/ 2, /*seed=*/ 0xE2E5EED1,
        );
        let mut total_fetches: u64 = 0;
        let mut total_prefetch: u64 = 0;
        for t in 0..N {
            let s = engine.generate(t).await.expect("generate should succeed");
            let per_token = s.hits + s.misses;
            assert_eq!(
                per_token, TOP_K as u64,
                "token {t}: expected {TOP_K} fetches, got {per_token} ({s:?})"
            );
            total_fetches += per_token;
            total_prefetch += s.prefetch_hits;
        }
        assert_eq!(total_fetches, N * TOP_K as u64);
        assert!(
            total_prefetch <= total_fetches,
            "prefetch hits ({total_prefetch}) cannot exceed total fetches ({total_fetches})",
        );
        let report = engine.report();
        assert_eq!(report.hits + report.misses, total_fetches);
        assert_eq!(report.expert_read_failures, 0);
    }

    /// Stress test for gist Part 1, fix #3: when many
    /// `spawn_prefetch` calls race past the semaphore ceiling, the
    /// excess prefetches must be **dropped** (not queued, not
    /// crashed) and the `prefetch_dropped_concurrency` counter must
    /// reflect that. We construct an engine with a deliberately tiny
    /// semaphore (cap=1), fire a burst of prefetches, and assert
    /// both the counter and that no panics escape the spawned tasks.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn spawn_prefetch_is_bounded_by_semaphore_under_load() {
        let dir = TempDir::new("prefetch-stress");
        let num_experts: u32 = 32;
        let d_model = 16;
        let d_ff = 32;
        let cache_slots = 4;
        let predict_fanout = 4;
        let seed = 0xDEADBEEFu64;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block_align = 4096usize;
        let expert_size = weight_bytes.div_ceil(block_align) * block_align;
        crate::io_provider::generate_synthetic_experts(
            &dir.path,
            num_experts,
            expert_size,
            d_model,
            d_ff,
        )
        .expect("generate experts");
        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.path.clone(),
                expert_size,
                block_align,
                use_direct_io: false,
                num_experts_per_layer: None,
            })
            .unwrap(),
        );
        storage.warmup_fds(0..num_experts).expect("warmup");
        let pool_slots = cache_slots + predict_fanout + 16;
        let pool = BufferPool::new(pool_slots, expert_size, block_align);
        let cache = Arc::new(MultiLayerExpertCache::single_layer(cache_slots));
        let router = Router::Markov(Arc::new(TopKRouter::new(num_experts, 2, seed)));
        let predictor = Arc::new(PredictiveLoader::new(
            num_experts,
            predict_fanout,
            0.05,
            seed,
        ));
        let mut opts = EngineOptions::default();
        opts.max_concurrent_prefetches = 1; // adversarial ceiling
        let engine = Arc::new(Engine::with_options(
            cache,
            pool,
            storage,
            router,
            predictor,
            ModelShape {
                d_model,
                d_ff,
                hidden_seed: seed,
            },
            opts,
        ));
        // Fire a burst of prefetches well past the ceiling. With a
        // semaphore cap of 1 the vast majority must be refused.
        let burst = 256u32;
        for id in 0..burst {
            engine.spawn_prefetch(id % num_experts, 0.5);
        }
        // Give in-flight prefetches a moment to settle.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let dropped = engine
            .metrics
            .counters
            .prefetch_dropped_concurrency
            .load(Ordering::Relaxed);
        assert!(
            dropped > 0,
            "expected some prefetches to be dropped under a 1-permit semaphore; got 0"
        );
        // Sanity: the report surface mirrors the counter.
        assert_eq!(engine.report().prefetch_dropped_concurrency, dropped);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn controlled_shutdown_cancels_in_flight_engine_owned_background_work() {
        let dir = TempDir::new("controlled-background-shutdown");
        let engine = build_engine(&dir.path, 4, 16, 32, 2, 1, 1, 0x5A17);
        let weak_engine = Arc::downgrade(&engine);

        // Test seam with the exact ownership shape of spawn_prefetch: the
        // detached future owns Arc<Engine> and otherwise cannot finish. The
        // old untracked spawn design would keep the full engine graph alive.
        let task_engine = engine.clone();
        assert!(engine.background_tasks.spawn(async move {
            std::future::pending::<()>().await;
            drop(task_engine);
        }));
        assert_eq!(engine.background_tasks.active.load(Ordering::Acquire), 1);
        assert!(weak_engine.strong_count() >= 2);

        engine.shutdown_background_tasks().await.unwrap();
        assert!(!engine.background_tasks.accepts_work());
        assert_eq!(engine.background_tasks.active.load(Ordering::Acquire), 0);

        // Shutdown is an admission boundary: neither direct tracker users nor
        // the real speculative-prefetch producer may create later work.
        assert!(!engine.background_tasks.spawn(std::future::pending::<()>()));
        engine.spawn_prefetch(0, 0.5);
        assert_eq!(engine.background_tasks.active.load(Ordering::Acquire), 0);

        drop(engine);
        assert!(weak_engine.upgrade().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn controlled_shutdown_prevents_cross_engine_cache_and_counter_contamination() {
        let first_dir = TempDir::new("isolated-runtime-a");
        let first = build_engine(&first_dir.path, 4, 16, 32, 2, 1, 1, 0xA11CE);
        first.warm_with(&[0]).await.unwrap();
        let _ = first.generate(0).await.unwrap();
        assert!(first.core.cache.len() > 0);
        assert!(first.report().tokens_processed > 0);

        // Include work in flight so this is not merely a clean empty-engine
        // drop. It retains the same Arc<Engine> graph as speculation.
        let task_engine = first.clone();
        assert!(first.background_tasks.spawn(async move {
            std::future::pending::<()>().await;
            drop(task_engine);
        }));
        first.shutdown_background_tasks().await.unwrap();
        drop(first);

        let second_dir = TempDir::new("isolated-runtime-b");
        let second = build_engine(&second_dir.path, 4, 16, 32, 2, 1, 1, 0xB0B);
        let report = second.report();
        assert_eq!(second.core.cache.len(), 0);
        assert_eq!(report.tokens_processed, 0);
        assert_eq!(report.hits, 0);
        assert_eq!(report.misses, 0);
        assert_eq!(report.bytes_read, 0);
        assert_eq!(
            second.routed_expert_execution_snapshot(),
            RoutedExpertExecutionSnapshot::default()
        );
        assert!(second.gpu_expert_io_snapshot().is_none());
        second.shutdown_background_tasks().await.unwrap();
    }

    /// Regression test for the post-`predict_min_prob` panic:
    /// `expert fetch starved: buffer pool exhausted with cache pinned`.
    ///
    /// The buffer pool is sized as `cache_slots + headroom`. If the
    /// prefetch semaphore is set to the operator-facing default
    /// (`DEFAULT_MAX_CONCURRENT_PREFETCHES = 64`) without being
    /// clamped, every in-flight prefetch holds a `PooledBuffer` for
    /// the duration of its I/O — so when the cache is fully pinned a
    /// foreground fetch has nowhere to land. The fix is to clamp the
    /// semaphore at construction time to the pool's actual headroom
    /// (`pool_slots − cache_slots`) **minus one slot reserved for the
    /// critical path**, keeping `max_concurrent_prefetches` as an
    /// additional user ceiling.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn prefetch_semaphore_is_clamped_to_pool_headroom() {
        let dir = TempDir::new("prefetch-clamp");
        let num_experts: u32 = 8;
        let d_model = 16;
        let d_ff = 32;
        let cache_slots = 4;
        let predict_fanout = 2;
        let top_k = 2;
        let seed = 0xBADC0DEu64;

        // Build via the shared `build_engine` fixture so the pool/cache/
        // router/predictor wiring stays aligned with the other tests. It
        // sizes the pool as `cache_slots + predict_fanout.max(1)` (same as
        // `cmd_run` / `cmd_serve`) and leaves the operator ceiling at the
        // runaway default (`DEFAULT_MAX_CONCURRENT_PREFETCHES = 64`) — so
        // without the clamp the semaphore would allow 64 concurrent
        // prefetches even though only 2 pool buffers are available beyond
        // the pinned cache slots.
        let engine = build_engine(
            &dir.path,
            num_experts,
            d_model,
            d_ff,
            cache_slots,
            top_k,
            predict_fanout,
            seed,
        );

        // Same pool sizing as `cmd_run` / `cmd_serve` and `build_engine`:
        // cache_slots + headroom. The clamp now *reserves one headroom
        // slot for the critical path*, so the prefetch semaphore is sized
        // at `headroom - 1`, not the full headroom.
        let pool_slots = cache_slots + predict_fanout.max(1); // headroom = predict_fanout = 2
        let headroom = pool_slots - cache_slots;
        let expected_permits = headroom - 1; // one slot reserved for the foreground fetch
        let available = engine.core.prefetch_semaphore.available_permits();
        assert_eq!(
            available, expected_permits,
            "prefetch semaphore must be clamped to pool headroom minus one reserved \
             critical-path slot (pool_slots={pool_slots} - cache_slots={cache_slots} - 1 \
             = {expected_permits}); got {available}"
        );
        assert!(
            available < DEFAULT_MAX_CONCURRENT_PREFETCHES,
            "clamp must strictly tighten the user ceiling when pool headroom is smaller"
        );
    }

    /// **Gist Task 1 — GPU Promotion Regression Test.**
    ///
    /// After `promote_after_hits` cache hits on the same expert via
    /// `moe_step`, *exactly one* `(expert_id, resident)` message must
    /// be emitted on the `gpu_promotion_tx` MPSC. The assertion runs
    /// directly against the receiver side of the channel — i.e. the
    /// raw mpsc traffic, before any consumer task processes it — per
    /// the gist's "verify the message count directly; do not use the
    /// report API" requirement.
    ///
    /// The test installs a custom channel via
    /// [`Engine::install_gpu_cache_for_test`] so it can observe the
    /// sender without the background promotion task draining it.
    /// `promote_after_hits` is chosen small (= 3) to keep the test
    /// fast; the same logic applies for any positive threshold.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn moe_step_emits_exactly_one_gpu_promotion_after_threshold_hits() {
        use crate::expert_cache::GpuExpertCache;

        let dir = TempDir::new("gpu-promotion");
        let num_experts: u32 = 4;
        let top_k = 1;
        let d_model = 16;
        let d_ff = 32;
        let cache_slots = 4;
        let predict_fanout = 1;
        let seed: u64 = 0xBADC0FFEE0DDF00D_u64;

        // Build a plain engine (no GPU cache yet).
        let engine = build_engine(
            &dir.path,
            num_experts,
            d_model,
            d_ff,
            cache_slots,
            top_k,
            predict_fanout,
            seed,
        );

        // Warm the RAM cache with one expert so every moe_step is a
        // RAM hit (the path that drives promotion).
        let target_id: u32 = 0;
        engine
            .warm_with(&[target_id])
            .await
            .expect("warm RAM cache");
        assert!(
            engine.core.cache.get(target_id).is_some(),
            "warm_with must leave the expert resident"
        );

        // Install a GPU cache + custom mpsc channel (no background
        // consumer). `promote_after_hits = 3` means the *3rd* RAM hit
        // is the crossing event; the 1st and 2nd RAM hits must not
        // emit, and the 4th, 5th, … must not emit either (edge
        // trigger).
        let promote_after: u64 = 3;
        let total_hits: u64 = 6;
        // Capacity must be large enough to fit the expert resident.
        // expert_weight_bytes() gives the f32 weight footprint.
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let gpu = Arc::new(GpuExpertCache::new(
            weight_bytes * 4,
            /*anchor_ratio=*/ 0.5,
            promote_after,
        ));
        // We need `&mut Engine` to install — engine was wrapped in
        // Arc<Engine> by build_engine, so unwrap it back.
        let mut engine_owned = Arc::try_unwrap(engine)
            .map_err(|_| ())
            .expect("test owns the sole Arc");
        let mut rx = engine_owned.install_gpu_cache_for_test(gpu.clone());
        let engine = Arc::new(engine_owned);

        // Drive `moe_step` total_hits times against the same expert.
        // We bypass the gate by constructing a hidden state directly
        // and passing the target expert id.
        let hidden = crate::inference::synth_hidden_state(0, d_model, seed);
        for t in 0..total_hits {
            let _ = engine
                .moe_step(t, /*layer=*/ 0, &hidden, &[target_id])
                .await;
        }

        // Drain the channel — must contain exactly one message,
        // emitted on the threshold-crossing hit.
        let mut received: Vec<(u32, Arc<ExpertResident>)> = Vec::new();
        // `try_recv` lets us inspect without blocking — the sender
        // half is still alive (kept in `engine.core.gpu_promotion_tx`)
        // so a naive `recv()` would hang indefinitely.
        while let Ok(msg) = rx.try_recv() {
            received.push(msg);
        }
        assert_eq!(
            received.len(),
            1,
            "exactly one promotion must be enqueued for {total_hits} RAM hits with \
             promote_after_hits = {promote_after} (got {})",
            received.len()
        );
        assert_eq!(
            received[0].0, target_id,
            "the promotion message must carry the target expert id"
        );
    }

    /// **Affinity arm wiring (end-to-end).** With a layer-qualified id
    /// geometry and an installed [`LayeredExpertAffinity`], driving
    /// `moe_step` for a layer must record the layer's co-fired experts
    /// into *that layer's* matrix (in the layer-local id namespace),
    /// and must not leak co-firings into other layers' matrices. This
    /// exercises the `observe_layer` call wired into `moe_step` and the
    /// global→local id mapping.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn moe_step_records_layer_affinity_co_firings() {
        let dir = TempDir::new("affinity-wiring");
        let per_layer: u32 = 4;
        let num_layers: usize = 2;
        let total_experts: u32 = per_layer * num_layers as u32; // 8 global ids
        let d_model = 16usize;
        let d_ff = 32usize;
        let seed: u64 = 0xA5A5_F00D_u64;

        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block_align = 4096usize;
        let expert_size = weight_bytes.div_ceil(block_align) * block_align;
        generate_synthetic_experts(&dir.path, total_experts, expert_size, d_model, d_ff)
            .expect("generate synthetic experts");

        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.path.clone(),
                expert_size,
                block_align,
                use_direct_io: false,
                // Layer-qualified geometry: global id = layer*per_layer + local.
                num_experts_per_layer: Some(per_layer),
            })
            .expect("storage init"),
        );
        storage.warmup_fds(0..total_experts).expect("pre-open fds");

        let pool = BufferPool::new(total_experts as usize + 2, expert_size, block_align);
        let cache = Arc::new(MultiLayerExpertCache::single_layer(total_experts as usize));
        let router = Router::Markov(Arc::new(TopKRouter::new(total_experts, 2, seed)));
        let predictor = Arc::new(PredictiveLoader::new(total_experts, 2, 0.05, seed));

        // Keep a clone of the affinity matrix so we can assert on it
        // after it is moved into the engine.
        let affinity = Arc::new(LayeredExpertAffinity::new(num_layers, per_layer));
        let engine = Arc::new(
            Engine::new(
                cache,
                pool,
                storage,
                router,
                predictor,
                ModelShape {
                    d_model,
                    d_ff,
                    hidden_seed: seed,
                },
            )
            .with_affinity(
                affinity.clone(),
                /*neighbors_k=*/ 2,
                /*decay_epoch=*/ 1_000_000,
            ),
        );

        // Co-fire global experts {4,5} in layer 1 (local {0,1}) a few
        // times so the pair's co-occurrence is unambiguous.
        let hidden = crate::inference::synth_hidden_state(0, d_model, seed);
        for t in 0..3u64 {
            let _ = engine.moe_step(t, /*layer=*/ 1, &hidden, &[4, 5]).await;
        }

        // Layer 1's matrix (local namespace) must show 0 and 1 as mutual
        // neighbours.
        assert_eq!(
            affinity.neighbors(1, 0, 2),
            vec![1],
            "local 0's neighbour in layer 1"
        );
        assert_eq!(
            affinity.neighbors(1, 1, 2),
            vec![0],
            "local 1's neighbour in layer 1"
        );
        assert_eq!(affinity.affinity(1, 0, 1), 3, "co-fired three times");
        // No leakage into layer 0's matrix.
        assert!(
            affinity.neighbors(0, 0, 2).is_empty(),
            "layer 0 must be untouched"
        );
    }

    /// **Tier 1 — online static residency.** With no seed profile the
    /// engine must derive its hot set from the live route observations
    /// after the warmup window and pin exactly `ceil(fraction × N)`
    /// experts — the most-frequently routed ones — exactly once.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn static_residency_online_pins_hottest_after_warmup() {
        let dir = TempDir::new("static-residency-online");
        let total_experts: u32 = 8;
        let d_model = 16usize;
        let d_ff = 32usize;
        let seed: u64 = 0x5EED_1234_u64;

        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block_align = 4096usize;
        let expert_size = weight_bytes.div_ceil(block_align) * block_align;
        generate_synthetic_experts(&dir.path, total_experts, expert_size, d_model, d_ff)
            .expect("generate synthetic experts");

        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.path.clone(),
                expert_size,
                block_align,
                use_direct_io: false,
                num_experts_per_layer: None,
            })
            .expect("storage init"),
        );
        storage.warmup_fds(0..total_experts).expect("pre-open fds");

        let pool = BufferPool::new(total_experts as usize + 2, expert_size, block_align);
        let cache = Arc::new(MultiLayerExpertCache::single_layer(total_experts as usize));
        let router = Router::Markov(Arc::new(TopKRouter::new(total_experts, 2, seed)));
        let predictor = Arc::new(PredictiveLoader::new(total_experts, 2, 0.05, seed));

        // fraction 0.25 of 8 experts ⇒ pin the 2 hottest. Warm up 4 tokens.
        let engine = Arc::new(
            Engine::new(
                cache.clone(),
                pool,
                storage,
                router,
                predictor,
                ModelShape {
                    d_model,
                    d_ff,
                    hidden_seed: seed,
                },
            )
            .with_static_residency(0.25, /*warmup_tokens=*/ 4, /*profile=*/ None),
        );

        let hidden = crate::inference::synth_hidden_state(0, d_model, seed);
        // Skewed stream: experts {2,5} fire every token; a rotating cold
        // expert fires once each so {2,5} are unambiguously hottest.
        // Reuse token_idx=0 on every call to prove online warmup is driven
        // by the engine-owned observation counter, not the caller seed.
        for t in 0..3u64 {
            let cold = 6 + (t % 2) as u32; // 6 or 7, never as hot as {2,5}
            let _ = engine
                .moe_step(0, /*layer=*/ 0, &hidden, &[2, 5, cold])
                .await;
        }
        assert_eq!(
            cache.pinned_count(),
            0,
            "warmup should not fire before 4 bumps"
        );
        for t in 3..8u64 {
            let cold = 6 + (t % 2) as u32;
            let _ = engine
                .moe_step(0, /*layer=*/ 0, &hidden, &[2, 5, cold])
                .await;
        }

        // After warmup the hot set must be pinned exactly once.
        assert_eq!(cache.pinned_count(), 2, "ceil(0.25*8)=2 experts pinned");
        assert_eq!(cache.pinned_ids(), vec![2, 5], "the two hottest experts");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn static_residency_pin_survives_locality_unpin() {
        let dir = TempDir::new("static-locality-pin");
        let total_experts: u32 = 4;
        let d_model = 16usize;
        let d_ff = 32usize;
        let seed: u64 = 0x51A7_1C_u64;
        let engine = build_engine(
            &dir.path,
            total_experts,
            d_model,
            d_ff,
            /*cache_slots=*/ 4,
            /*top_k=*/ 1,
            /*predict_fanout=*/ 1,
            seed,
        );
        let profile = crate::residency::ResidencyProfile::from_counts(HashMap::from([
            (0, 10),
            (1, 1),
            (2, 1),
            (3, 1),
        ]));
        let engine = {
            let cache = engine.core.cache.clone();
            let pool = engine.core.pool.clone();
            let storage = engine.core.storage.clone();
            let router = engine.core.router.clone();
            let predictor = engine.core.predictor.clone();
            let shape = engine.core.shape;
            let monitor = Arc::new(LocalityMonitor::new(total_experts, /*window=*/ 2));
            Arc::new(
                Engine::new(cache, pool, storage, router, predictor, shape)
                    .with_static_residency(0.25, /*warmup_tokens=*/ 100, Some(profile))
                    .with_locality_monitor(monitor, 0.5),
            )
        };

        engine.maybe_apply_static_residency();
        assert!(
            engine.core.cache.pinned_ids().contains(&0),
            "static residency pins expert 0"
        );

        engine.locality_observe_and_reconcile(&[0]);
        assert!(
            engine.core.cache.pinned_ids().contains(&0),
            "locality also sees expert 0 hot"
        );
        engine.locality_observe_and_reconcile(&[1]);
        engine.locality_observe_and_reconcile(&[1]);

        assert!(
            engine.core.cache.pinned_ids().contains(&0),
            "locality must not unpin an expert pinned by static residency"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn credit_prefetch_use_counts_shadow_resident_once() {
        let dir = TempDir::new("prefetch-use-credit");
        let engine = build_engine(
            &dir.path,
            /*num_experts=*/ 2,
            /*d_model=*/ 16,
            /*d_ff=*/ 32,
            /*cache_slots=*/ 2,
            /*top_k=*/ 1,
            /*predict_fanout=*/ 1,
            0xC0FF_EE_u64,
        );
        let shadow_pool = BufferPool::new_with_shadow(1, 1, 4096, 4096);
        let buf = shadow_pool.try_acquire_shadow().expect("shadow buffer");
        let resident = Arc::new(ExpertResident::new(0, buf));
        assert!(resident.is_shadow_backed());

        let first = resident.record_hit();
        engine.credit_prefetch_use(&resident, first);
        let second = resident.record_hit();
        engine.credit_prefetch_use(&resident, second);

        assert_eq!(
            engine
                .metrics
                .counters
                .prefetch_used
                .load(Ordering::Relaxed),
            1,
            "only the first consumption of a shadow-backed resident is credited"
        );
    }

    /// **Tier 1 — profile-seeded static residency.** With an offline
    /// popularity profile the hot set is pinned at the first token with
    /// no warmup, independent of the live routing stream.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn static_residency_profile_pins_immediately() {
        let dir = TempDir::new("static-residency-profile");
        let total_experts: u32 = 8;
        let d_model = 16usize;
        let d_ff = 32usize;
        let seed: u64 = 0x5EED_ABCD_u64;

        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block_align = 4096usize;
        let expert_size = weight_bytes.div_ceil(block_align) * block_align;
        generate_synthetic_experts(&dir.path, total_experts, expert_size, d_model, d_ff)
            .expect("generate synthetic experts");

        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.path.clone(),
                expert_size,
                block_align,
                use_direct_io: false,
                num_experts_per_layer: None,
            })
            .expect("storage init"),
        );
        storage.warmup_fds(0..total_experts).expect("pre-open fds");

        let pool = BufferPool::new(total_experts as usize + 2, expert_size, block_align);
        let cache = Arc::new(MultiLayerExpertCache::single_layer(total_experts as usize));
        let router = Router::Markov(Arc::new(TopKRouter::new(total_experts, 2, seed)));
        let predictor = Arc::new(PredictiveLoader::new(total_experts, 2, 0.05, seed));

        // Profile makes experts 0 and 3 hottest regardless of the stream.
        let mut counts = std::collections::HashMap::new();
        counts.insert(0u32, 1000u64);
        counts.insert(3u32, 900u64);
        counts.insert(1u32, 5u64);
        let profile = crate::residency::ResidencyProfile::from_counts(counts);

        let engine = Arc::new(
            Engine::new(
                cache.clone(),
                pool,
                storage,
                router,
                predictor,
                ModelShape {
                    d_model,
                    d_ff,
                    hidden_seed: seed,
                },
            )
            .with_static_residency(0.25, /*warmup_tokens=*/ 100, Some(profile)),
        );

        let hidden = crate::inference::synth_hidden_state(0, d_model, seed);
        // A single token suffices — the seed profile ignores warmup.
        let _ = engine.moe_step(0, /*layer=*/ 0, &hidden, &[5, 6]).await;

        assert_eq!(cache.pinned_count(), 2, "ceil(0.25*8)=2 experts pinned");
        assert_eq!(
            cache.pinned_ids(),
            vec![0, 3],
            "profile's two hottest experts"
        );
    }

    /// **Layer-scoped speculator accuracy.** On the `moe_step` path the
    /// gate's decision is confined to the current layer's slice of the
    /// layer-qualified global namespace, so
    /// `speculator_predict_and_train` must draw its prediction from
    /// that same slice. A global arg-max spreads the top-K across every
    /// layer's logits and almost never lands in the current layer —
    /// the production `accuracy=0.82%` symptom.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn speculator_prediction_is_scoped_to_current_layer() {
        let dir = TempDir::new("spec-layer-scope");
        let per_layer: u32 = 4;
        let num_layers: usize = 4;
        let total_experts: u32 = per_layer * num_layers as u32;
        let d_model = 16usize;
        let d_ff = 32usize;
        let seed: u64 = 0xBEEF_CAFE_u64;

        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block_align = 4096usize;
        let expert_size = weight_bytes.div_ceil(block_align) * block_align;
        generate_synthetic_experts(&dir.path, total_experts, expert_size, d_model, d_ff)
            .expect("generate synthetic experts");

        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.path.clone(),
                expert_size,
                block_align,
                use_direct_io: false,
                num_experts_per_layer: Some(per_layer),
            })
            .expect("storage init"),
        );
        storage.warmup_fds(0..total_experts).expect("pre-open fds");

        let pool = BufferPool::new(total_experts as usize + 2, expert_size, block_align);
        let cache = Arc::new(MultiLayerExpertCache::single_layer(total_experts as usize));
        let router = Router::Markov(Arc::new(TopKRouter::new(total_experts, 2, seed)));
        let predictor = Arc::new(PredictiveLoader::new(total_experts, 2, 0.05, seed));
        let spec = Arc::new(NeuralSpeculator::new(d_model, 8, total_experts, seed));

        let engine = Arc::new(
            Engine::new(
                cache,
                pool,
                storage,
                router,
                predictor,
                ModelShape {
                    d_model,
                    d_ff,
                    hidden_seed: seed,
                },
            )
            .with_speculator(spec, /*top_k=*/ 2),
        );

        let hidden = crate::inference::synth_hidden_state(0, d_model, seed);
        for layer in 0..num_layers as u32 {
            let base = layer * per_layer;
            let target = [base, base + 1];
            let preds = engine.speculator_predict_and_train(&hidden, &target, Some(layer));
            assert!(
                !preds.is_empty(),
                "speculator must predict for layer {layer}"
            );
            for &p in &preds {
                assert!(
                    p >= base && p < base + per_layer,
                    "layer {layer}: predicted id {p} is outside slice {base}..{}",
                    base + per_layer
                );
            }
        }
    }

    /// **Locality pin budget (Finding 1).** Pinning the entire hot set
    /// can saturate the cache (every slot pinned ⇒ `insert` rejects all
    /// new residents and `evict_lru` returns `None`, serializing every
    /// foreground miss on the one reserved pool buffer). The reconcile
    /// step must cap pins so at least one slot per layer cache stays
    /// evictable, keeping the hottest ids (hot_set is heat-sorted).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn locality_pinning_leaves_an_evictable_slot() {
        let dir = TempDir::new("locality-pin-cap");
        let num_experts: u32 = 8;
        let d_model = 16usize;
        let d_ff = 32usize;
        let cache_slots = 3usize;
        let seed = 0x71D_CAFEu64;

        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block_align = 4096usize;
        let expert_size = weight_bytes.div_ceil(block_align) * block_align;
        generate_synthetic_experts(&dir.path, num_experts, expert_size, d_model, d_ff)
            .expect("generate synthetic experts");
        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.path.clone(),
                expert_size,
                block_align,
                use_direct_io: false,
                num_experts_per_layer: None,
            })
            .expect("storage init"),
        );
        storage.warmup_fds(0..num_experts).expect("pre-open fds");
        let pool = BufferPool::new(cache_slots + 2, expert_size, block_align);
        let cache = Arc::new(MultiLayerExpertCache::single_layer(cache_slots));
        let router = Router::Markov(Arc::new(TopKRouter::new(num_experts, 2, seed)));
        let predictor = Arc::new(PredictiveLoader::new(num_experts, 2, 0.05, seed));
        // Threshold 0.0 + tiny window ⇒ *every* observed id is hot, so
        // without the cap the whole working set would be pinned.
        let monitor = Arc::new(LocalityMonitor::new(num_experts, 64));
        let engine = Arc::new(
            Engine::new(
                cache,
                pool,
                storage,
                router,
                predictor,
                ModelShape {
                    d_model,
                    d_ff,
                    hidden_seed: seed,
                },
            )
            .with_locality_monitor(monitor, 0.0),
        );

        // Observe every expert repeatedly: all 8 ids meet the 0.0
        // threshold, but pins must stay below the cache capacity.
        for round in 0..16u32 {
            for id in 0..num_experts {
                engine.locality_observe_and_reconcile(&[id, (id + round) % num_experts]);
            }
        }
        let pinned = engine.core.cache.pinned_count();
        assert!(
            pinned <= cache_slots - 1,
            "locality pinning must leave >=1 evictable slot: pinned={pinned}, cap={cache_slots}"
        );
        assert!(pinned > 0, "the hottest ids should still be pinned");
    }

    /// **Markov layer-continuity guard (Finding 5).** The engine-global
    /// history ring may interleave concurrently-batched streams; an
    /// entry is a valid 2nd-order predecessor only if its layer is
    /// exactly one before the current step's (wrapping at the token
    /// boundary). Layer-less callers and flat namespaces bypass the
    /// check (legacy behaviour).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn markov_history_requires_layer_contiguity() {
        let dir = TempDir::new("markov-contiguity");
        let per_layer: u32 = 4;
        let num_layers: u32 = 4;
        let total = per_layer * num_layers;
        let d_model = 16usize;
        let d_ff = 32usize;
        let seed = 0x5EEDu64;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block_align = 4096usize;
        let expert_size = weight_bytes.div_ceil(block_align) * block_align;
        generate_synthetic_experts(&dir.path, total, expert_size, d_model, d_ff)
            .expect("generate synthetic experts");
        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.path.clone(),
                expert_size,
                block_align,
                use_direct_io: false,
                num_experts_per_layer: Some(per_layer),
            })
            .expect("storage init"),
        );
        let pool = BufferPool::new(4, expert_size, block_align);
        let cache = Arc::new(MultiLayerExpertCache::single_layer(2));
        let router = Router::Markov(Arc::new(TopKRouter::new(total, 2, seed)));
        let predictor = Arc::new(PredictiveLoader::new(total, 2, 0.05, seed));
        let engine = Arc::new(Engine::new(
            cache,
            pool,
            storage,
            router,
            predictor,
            ModelShape {
                d_model,
                d_ff,
                hidden_seed: seed,
            },
        ));

        // Contiguous: L -> L+1, and last-layer -> 0 (token boundary).
        assert!(engine.markov_layers_contiguous(Some(0), Some(1)));
        assert!(engine.markov_layers_contiguous(Some(2), Some(3)));
        assert!(engine.markov_layers_contiguous(Some(num_layers - 1), Some(0)));
        // Non-contiguous: skips, repeats, backwards, unknown prev.
        assert!(!engine.markov_layers_contiguous(Some(0), Some(2)));
        assert!(!engine.markov_layers_contiguous(Some(1), Some(1)));
        assert!(!engine.markov_layers_contiguous(Some(3), Some(2)));
        assert!(!engine.markov_layers_contiguous(None, Some(1)));
        // Layer-less current step (generate path) bypasses the check.
        assert!(engine.markov_layers_contiguous(Some(3), None));
        assert!(engine.markov_layers_contiguous(None, None));
    }

    /// **Shadow-pool recycling (Finding 3).** Prefetched residents keep
    /// their shadow (Buffer B) buffer for the life of their residency,
    /// so once `shadow_slots` of them accumulate every further
    /// speculative prefetch used to be dropped ("shadow pool busy")
    /// until an unrelated eviction happened to recycle one. The fix:
    /// when Buffer B is starved, `spawn_prefetch` evicts the LRU
    /// unpinned shadow-backed resident and retries.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn spawn_prefetch_recycles_shadow_backed_residents_when_starved() {
        let dir = TempDir::new("shadow-recycle");
        let num_experts: u32 = 8;
        let d_model = 16usize;
        let d_ff = 32usize;
        let seed = 0xB0FFu64;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block_align = 4096usize;
        let expert_size = weight_bytes.div_ceil(block_align) * block_align;
        generate_synthetic_experts(&dir.path, num_experts, expert_size, d_model, d_ff)
            .expect("generate synthetic experts");
        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.path.clone(),
                expert_size,
                block_align,
                use_direct_io: false,
                num_experts_per_layer: None,
            })
            .expect("storage init"),
        );
        storage.warmup_fds(0..num_experts).expect("pre-open fds");
        // ONE shadow slot: the first prefetched resident parks it.
        let pool = BufferPool::new_with_shadow(4, 1, expert_size, block_align);
        let cache = Arc::new(MultiLayerExpertCache::single_layer(3));
        let router = Router::Markov(Arc::new(TopKRouter::new(num_experts, 2, seed)));
        let predictor = Arc::new(PredictiveLoader::new(num_experts, 2, 0.05, seed));
        let engine = Arc::new(Engine::new(
            cache,
            pool,
            storage,
            router,
            predictor,
            ModelShape {
                d_model,
                d_ff,
                hidden_seed: seed,
            },
        ));

        // Helper: spawn a prefetch and wait until the id is resident.
        async fn prefetch_and_wait(engine: &Arc<Engine>, id: u32) {
            engine.spawn_prefetch(id, 0.5);
            for _ in 0..200 {
                if engine.core.cache.contains(id) {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
            panic!("prefetch of expert {id} never landed");
        }

        // First prefetch parks the only shadow buffer inside resident 0.
        prefetch_and_wait(&engine, 0).await;
        assert!(engine.core.cache.get(0).unwrap().is_shadow_backed());

        // Second prefetch finds Buffer B starved; it must recycle the
        // LRU shadow-backed resident (id 0) and still complete.
        prefetch_and_wait(&engine, 1).await;
        assert!(
            engine.core.cache.contains(1),
            "starved prefetch must complete by recycling a shadow-backed resident"
        );
        assert!(
            !engine.core.cache.contains(0),
            "the parked shadow-backed resident must have been evicted to free Buffer B"
        );
        // A *pinned* shadow-backed resident must never be recycled.
        engine.core.cache.pin(1);
        engine.spawn_prefetch(2, 0.5);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(
            engine.core.cache.contains(1),
            "pinned shadow-backed resident must survive shadow starvation"
        );
    }

    /// **Windowed depth-N look-ahead (`speculate_layer_ahead`).** With a
    /// layer-qualified geometry, an installed speculator, and
    /// `pipeline_depth = 3`, driving `moe_step` for layer 0 must submit
    /// speculative prefetches for the sliding window of layers
    /// `1 ..= 3` — not just the next layer. We detect the depth by
    /// asserting that at least one expert from a layer `>= 2` (global id
    /// `>= 2 * per_layer`, unreachable with the legacy single-layer
    /// look-ahead) becomes resident after the look-ahead fires. The
    /// `pipeline_depth = 1` control must never reach those layers.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn speculate_layer_ahead_primes_window_of_upcoming_layers() {
        async fn deepest_prefetched_layer(pipeline_depth: u32, per_layer: u32) -> u32 {
            let dir = TempDir::new("layer-ahead-window");
            let num_layers: usize = 4;
            let total_experts: u32 = per_layer * num_layers as u32;
            let d_model = 16usize;
            let d_ff = 32usize;
            let seed: u64 = 0x1234_5678_9ABC_DEF0;

            let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
            let block_align = 4096usize;
            let expert_size = weight_bytes.div_ceil(block_align) * block_align;
            generate_synthetic_experts(&dir.path, total_experts, expert_size, d_model, d_ff)
                .expect("generate synthetic experts");

            let storage = Arc::new(
                NvmeStorage::new(StorageConfig {
                    base_path: dir.path.clone(),
                    expert_size,
                    block_align,
                    use_direct_io: false,
                    num_experts_per_layer: Some(per_layer),
                })
                .expect("storage init"),
            );
            storage.warmup_fds(0..total_experts).expect("pre-open fds");

            // Generous pool so neither the foreground misses nor the
            // speculative window starve for buffers in this test.
            let pool = BufferPool::new(total_experts as usize + 8, expert_size, block_align);
            let cache = Arc::new(MultiLayerExpertCache::single_layer(total_experts as usize));
            let router = Router::Markov(Arc::new(TopKRouter::new(total_experts, 2, seed)));
            let predictor = Arc::new(PredictiveLoader::new(total_experts, 2, 0.05, seed));
            let spec = Arc::new(NeuralSpeculator::new(d_model, 8, total_experts, seed));

            let engine = Arc::new(
                Engine::new(
                    cache,
                    pool,
                    storage,
                    router,
                    predictor,
                    ModelShape {
                        d_model,
                        d_ff,
                        hidden_seed: seed,
                    },
                )
                .with_speculator(spec, /*top_k=*/ 2)
                .with_pipeline_depth(pipeline_depth),
            );

            // Fire the windowed look-ahead in isolation (calling
            // `speculate_layer_ahead` directly rather than `moe_step`, so
            // the global `union_prefetch` arm — which primes arbitrary
            // layers regardless of `pipeline_depth` — doesn't confound the
            // depth measurement). The window spans layers
            // `1 ..= pipeline_depth` off the residual entering layer 0.
            let hidden = crate::inference::synth_hidden_state(0, d_model, seed);
            engine.speculate_layer_ahead(&hidden, /*current_layer=*/ 0);

            // The look-ahead spawns one prefetch per predicted id with the
            // distance-tapered fanout (full `top_k` at L+1, `top_k/distance`
            // further out). Against an initially-empty cache none are deduped,
            // so this is exactly how many background reads must complete.
            let top_k = 2usize;
            let expected_spawns: u64 = (1..=pipeline_depth)
                .map(|distance| (top_k / distance as usize).max(1) as u64)
                .sum();

            // Speculative prefetches run on background tasks; wait for the
            // expected number to complete (bounded), then measure how many
            // layers deep the primed experts reach.
            let mut deepest = 0u32;
            for _ in 0..300 {
                if engine.report().prefetch_completed >= expected_spawns {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            for id in 0..total_experts {
                if engine.core.cache.contains(id) {
                    deepest = deepest.max(id / per_layer);
                }
            }
            deepest
        }

        let per_layer: u32 = 4;
        // Depth 3 must reach at least layer 2 (ids >= 8) — only possible
        // because the look-ahead window spans multiple upcoming layers.
        let deep = deepest_prefetched_layer(3, per_layer).await;
        assert!(
            deep >= 2,
            "pipeline_depth=3 must prime experts at least 2 layers ahead (got deepest layer {deep})"
        );
        // Depth 1 is the legacy single-layer look-ahead: it can only ever
        // reach layer 1, never layer 2+.
        let shallow = deepest_prefetched_layer(1, per_layer).await;
        assert!(
            shallow <= 1,
            "pipeline_depth=1 must never prime beyond the next layer (got deepest layer {shallow})"
        );
    }

    /// `AlignedKvCache::append` extends the resident window until
    /// capacity, after which it slides the tail down by one and
    /// overwrites the freed slot with the new row. The first
    /// `seq_len` indices always read back the most recent K/V rows.
    #[test]
    fn aligned_kv_cache_rolls_window_and_keeps_recent_rows() {
        let kv_dim = 8usize;
        let window = 4usize;
        let mut cache = AlignedKvCache::new(window, kv_dim);
        // The buffer must be page-aligned (4 KiB) — that's the whole
        // point of using AlignedBuffer here.
        assert_eq!(cache.keys_ptr() as usize % KV_CACHE_BLOCK_ALIGN, 0);
        assert_eq!(cache.values_ptr() as usize % KV_CACHE_BLOCK_ALIGN, 0);

        for i in 0..window {
            let k: Vec<f32> = (0..kv_dim).map(|j| (i * 10 + j) as f32).collect();
            let v: Vec<f32> = (0..kv_dim).map(|j| (i * 10 + j) as f32 + 0.5).collect();
            assert_eq!(cache.append(&k, &v), false, "no eviction before full");
        }
        assert_eq!(cache.seq_len(), window);
        // Read back: token 0 has values starting at 0, token 3 at 30.
        assert_eq!(cache.key(0)[0], 0.0);
        assert_eq!(cache.key(3)[0], 30.0);

        // Filling one more row evicts the oldest. After the shift,
        // index 0 is what used to be index 1 (values 10..), index 3
        // is the *new* row (values 40..).
        let k: Vec<f32> = (0..kv_dim).map(|j| (4 * 10 + j) as f32).collect();
        let v: Vec<f32> = (0..kv_dim).map(|j| (4 * 10 + j) as f32 + 0.5).collect();
        assert_eq!(cache.append(&k, &v), true, "eviction expected at capacity");
        assert_eq!(cache.seq_len(), window);
        assert_eq!(cache.key(0)[0], 10.0, "oldest token shifted out");
        assert_eq!(cache.key(window - 1)[0], 40.0, "new token at tail");
        assert_eq!(cache.value(window - 1)[0], 40.5);

        // Resident bytes accounting matches: 4 tokens * 8 floats * 2 (k+v) * 4 bytes.
        assert_eq!(cache.resident_bytes(), 4 * 8 * 2 * 4);

        // Reset clears seq_len but keeps the page-aligned allocation.
        let ptr_before = cache.keys_ptr();
        cache.zeroize();
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.keys_ptr(), ptr_before, "allocation must be reused");
    }

    /// **Gist Task 2 — proptest for `AlignedKvCache` and the
    /// `row_floats` slice arithmetic that backs `.key()` / `.value()`.**
    ///
    /// Two invariants we want to fuzz:
    ///   1. `seq_len()` never exceeds `window_tokens()` regardless
    ///      of how many `append()` calls have been made.
    ///   2. After any number of appends, `key(i)` / `value(i)` for
    ///      every `i < seq_len()` returns a slice of length exactly
    ///      `kv_dim` that lies fully inside the backing
    ///      `AlignedBuffer`. The row content must equal what was
    ///      written for the *most recent* `seq_len` appends (i.e.
    ///      the rolling-window contract).
    mod aligned_kv_cache_proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig {
                cases: 48,
                ..ProptestConfig::default()
            })]

            #[test]
            fn append_respects_window_and_row_slices_are_valid(
                window_tokens in 1usize..16,
                kv_dim in 1usize..24,
                num_appends in 0usize..200,
            ) {
                let mut cache = AlignedKvCache::new(window_tokens, kv_dim);
                // Keep a log of every row written so we can verify
                // the rolling-window contract against the most
                // recent `min(num_appends, window_tokens)` entries.
                let mut history: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(num_appends);
                for t in 0..num_appends {
                    let k: Vec<f32> = (0..kv_dim).map(|j| (t * 1000 + j) as f32).collect();
                    let v: Vec<f32> = (0..kv_dim).map(|j| (t * 1000 + j) as f32 + 0.25).collect();
                    cache.append(&k, &v);
                    history.push((k, v));
                    // Invariant 1: window cap.
                    prop_assert!(
                        cache.seq_len() <= cache.window_tokens(),
                        "seq_len {} exceeded window {} after {} appends",
                        cache.seq_len(), cache.window_tokens(), t + 1,
                    );
                }
                // Invariant 2: all live rows have correct length and
                // hold the most recent values.
                let live = cache.seq_len();
                prop_assert_eq!(live, num_appends.min(window_tokens));
                let history_tail = &history[history.len().saturating_sub(live)..];
                for i in 0..live {
                    let k_slice = cache.key(i);
                    let v_slice = cache.value(i);
                    prop_assert_eq!(k_slice.len(), kv_dim);
                    prop_assert_eq!(v_slice.len(), kv_dim);
                    let (expected_k, expected_v) = &history_tail[i];
                    for j in 0..kv_dim {
                        prop_assert_eq!(k_slice[j], expected_k[j]);
                        prop_assert_eq!(v_slice[j], expected_v[j]);
                    }
                }
            }
        }
    }
}
// end mod engine::tests
