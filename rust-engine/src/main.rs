//! Micro-Expert-Router — MoE execution engine that hot-swaps experts from a
//! PCIe-attached NVMe drive into pre-allocated, page-aligned RAM via
//! `O_DIRECT` `pread(2)` (dispatched off the Tokio runtime with
//! `block_in_place`).
//!
//! See README.md at the repository root for architecture and design notes.

// Gist Task 3 — "Nightly AMX feature gating". See the matching
// comment in `lib.rs`. Off by default; opt in with
// `--features nightly-amx` and a nightly toolchain to unlock the
// real Intel AMX tile intrinsic surface. When this feature is not
// enabled, the AMX dispatch path falls back to AVX-512.
#![cfg_attr(feature = "nightly-amx", feature(stdarch_x86_amx))]

#[cfg(feature = "alloc-count")]
mod alloc_count {
    use super::AllocationSnapshot;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicU64, Ordering};

    pub struct CountingAllocator;

    static ALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
    static DEALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
    static REALLOCATION_CALLS: AtomicU64 = AtomicU64::new(0);
    static BYTES_ALLOCATED: AtomicU64 = AtomicU64::new(0);
    static BYTES_DEALLOCATED: AtomicU64 = AtomicU64::new(0);
    static CURRENT_BYTES: AtomicU64 = AtomicU64::new(0);
    static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: this allocator only observes the operation, then forwards
            // the exact layout to the platform allocator.
            let ptr = unsafe { System.alloc(layout) };
            if !ptr.is_null() {
                ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
                record_allocation(layout.size() as u64);
            }
            ptr
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            DEALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
            record_deallocation(layout.size() as u64);
            // SAFETY: `ptr`/`layout` are the exact pair passed to this global
            // allocator by the standard library, so forwarding them is valid.
            unsafe { System.dealloc(ptr, layout) };
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            // SAFETY: forwards the original pointer/layout and requested size
            // unchanged to the platform allocator.
            let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
            if !new_ptr.is_null() {
                REALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
                let old_size = layout.size() as u64;
                let new_size = new_size as u64;
                if new_size >= old_size {
                    record_allocation(new_size - old_size);
                } else {
                    record_deallocation(old_size - new_size);
                }
            }
            new_ptr
        }
    }

    pub fn reset() {
        ALLOCATION_CALLS.store(0, Ordering::Relaxed);
        DEALLOCATION_CALLS.store(0, Ordering::Relaxed);
        REALLOCATION_CALLS.store(0, Ordering::Relaxed);
        BYTES_ALLOCATED.store(0, Ordering::Relaxed);
        BYTES_DEALLOCATED.store(0, Ordering::Relaxed);
        CURRENT_BYTES.store(0, Ordering::Relaxed);
        PEAK_BYTES.store(0, Ordering::Relaxed);
    }

    pub fn snapshot() -> AllocationSnapshot {
        AllocationSnapshot {
            allocation_calls: ALLOCATION_CALLS.load(Ordering::Relaxed),
            deallocation_calls: DEALLOCATION_CALLS.load(Ordering::Relaxed),
            reallocation_calls: REALLOCATION_CALLS.load(Ordering::Relaxed),
            bytes_allocated: BYTES_ALLOCATED.load(Ordering::Relaxed),
            bytes_deallocated: BYTES_DEALLOCATED.load(Ordering::Relaxed),
            current_bytes: CURRENT_BYTES.load(Ordering::Relaxed),
            peak_bytes: PEAK_BYTES.load(Ordering::Relaxed),
        }
    }

    fn record_allocation(bytes: u64) {
        BYTES_ALLOCATED.fetch_add(bytes, Ordering::Relaxed);
        let current = CURRENT_BYTES
            .fetch_add(bytes, Ordering::Relaxed)
            .saturating_add(bytes);
        update_peak(current);
    }

    fn record_deallocation(bytes: u64) {
        BYTES_DEALLOCATED.fetch_add(bytes, Ordering::Relaxed);
        let _ = CURRENT_BYTES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            Some(current.saturating_sub(bytes))
        });
    }

    fn update_peak(current: u64) {
        let mut peak = PEAK_BYTES.load(Ordering::Relaxed);
        while current > peak {
            match PEAK_BYTES.compare_exchange_weak(
                peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => peak = observed,
            }
        }
    }
}

#[cfg(feature = "alloc-count")]
#[global_allocator]
static GLOBAL_ALLOCATOR: alloc_count::CountingAllocator = alloc_count::CountingAllocator;

mod aligned_buffer;
mod architecture;
mod backend;
mod batch_scheduler;
mod block_pool;
mod buffer_pool;
mod config;
mod dense_tensor;
mod dequant;
mod distributed;
mod draft;
mod engine;
mod expert_cache;
mod gating;
mod gguf;
mod gguf_loader;
mod gpu_native_residency;
pub(crate) mod gpu_native_diagnostics;
pub(crate) mod gpu_native_expert_permutation_semantic_parity;
pub(crate) mod gpu_native_f32_reference_boundary_audit;
pub(crate) mod gpu_native_greedy_parity;
pub(crate) mod gpu_native_layer0_diagnostics;
pub(crate) mod gpu_native_q4_expert_stage_attribution;
pub(crate) mod gpu_native_demand_source_concurrency;
pub(crate) mod gpu_native_real_benchmark;
pub(crate) mod gpu_native_router_rank_diagnostics;
pub(crate) mod gpu_native_semantic_parity_corpus;
pub(crate) mod gpu_native_semantic_parity_v2;
pub(crate) mod gpu_native_spanish_first_token_attribution;
pub(crate) mod gpu_native_v2_holdout_failure_attribution;

pub(crate) mod gpu_native_token_loop;
#[cfg(feature = "grpc")]
mod grpc;
#[cfg(feature = "grpc")]
mod grpc_gen;
mod inference;
mod io_provider;
mod io_reactor;
#[cfg(all(feature = "io_uring", target_os = "linux"))]
mod io_uring_storage;
mod kernels;
mod metrics;
mod middleware;
mod mla;
mod model;
mod multi_layer_cache;
mod numa;
mod packed_storage;
mod parallel;
mod prefetch_governor;
mod pregate;
mod qualification;
mod greedy_parity;
mod numerical_diagnostics;
mod q4_parity;
mod rayon_autotune;
mod residency;
mod router;
mod rpc;
mod sampling;
mod server;
mod session;
mod stage_timing;
mod tensor_header;
mod tokenizer;
mod transformer;
#[cfg(feature = "tui")]
mod tui;
mod workload;

use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use crate::backend::Backend;
use crate::buffer_pool::BufferPool;
use crate::engine::{Engine, EngineOptions, ModelShape};
use crate::inference::expert_weight_bytes_for;
use crate::io_provider::{NvmeStorage, StorageConfig};
use crate::multi_layer_cache::MultiLayerExpertCache;
use crate::router::{
    LayeredExpertAffinity, LocalityMonitor, NeuralSpeculator, PredictiveLoader, TopKRouter,
};

const SUPPORTED_SYNTHETIC_DTYPES: &str = "f32, f16, bf16, int8, q4k, q4_0, q8_0, mxfp4";
const SUPPORTED_RUNTIME_DTYPES: &str =
    "f32, f16, bf16, int8, q4k, q4_0, q5k, q6k, q8_0, mxfp4, mixed";

/// MoE execution engine that streams experts from NVMe via O_DIRECT pread(2).
#[derive(Parser, Debug)]
#[command(name = "micro-expert-router", version, about)]
struct Cli {
    /// Logging filter (e.g. `info`, `debug`, `micro_expert_router=debug`).
    #[arg(long, global = true, default_value = "info", env = "RUST_LOG")]
    log: String,

    /// Worker count for MER's process-wide CPU Rayon compute pool.
    ///
    /// Precedence: RAYON_NUM_THREADS > CLI > performance.rayon_threads >
    /// autotune/profile > MER's startup default.
    #[arg(long, global = true, value_name = "N", value_parser = parse_positive_rayon_threads)]
    rayon_threads: Option<usize>,

    /// Optional CPU placement mask in Linux cpulist syntax, e.g. `0-24`.
    ///
    /// Leave unset for portable default behavior: MER does not apply an
    /// artificial CPU mask. Precedence: CLI > performance.cpu_mask >
    /// legacy MER_PIN_CORES.
    #[arg(long, global = true, value_name = "CPULIST")]
    cpu_mask: Option<String>,

    /// Progress watchdog timeout in seconds. `0` disables the watchdog.
    #[arg(long, global = true, value_name = "SECS")]
    progress_timeout_secs: Option<u64>,

    /// Reuse a saved low-confidence Rayon autotune profile.
    ///
    /// By default low-confidence profiles are kept for auditability but are
    /// not applied to normal runs.
    #[arg(long, global = true)]
    reuse_low_confidence_rayon_profile: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum BenchRealCacheReset {
    /// Keep the same engine/cache across warmup and measured runs.
    Keep,
    /// Rebuild the runtime before every run, giving each run a cold cache.
    FreshRuntime,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum BenchRealOutputFormat {
    Human,
    Json,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Q8BenchKernel {
    Auto,
    Scalar,
    Avx2,
    Avx512,
    All,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Generate synthetic expert files for testing on local disk.
    GenData {
        /// Directory to write `expert_<id>.bin` files into.
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
        /// Number of experts to create.
        #[arg(long, default_value_t = 64)]
        num_experts: u32,
        /// Bytes per expert. Must be a multiple of 4096 for O_DIRECT and
        /// at least `3 * d_model * d_ff * 4` bytes (the SwiGLU weights);
        /// any extra bytes are zero-padded.
        ///
        /// Default 16 MiB pairs cleanly with `d_model=512 d_ff=2048`
        /// (12 MiB of weights + 4 MiB of padding).
        #[arg(long, default_value_t = 16 * 1024 * 1024)]
        expert_size: usize,
        /// Hidden / residual-stream dimension of the FFN (Mixtral: 4096,
        /// DeepSeek-V3: 7168). Default 512 keeps the synthetic compute
        /// cheap so I/O remains observable.
        #[arg(long, default_value_t = 512)]
        d_model: usize,
        /// Intermediate FFN dimension (Mixtral: 14336, Llama-3-MoE: 14336).
        /// Default 2048.
        #[arg(long, default_value_t = 2048)]
        d_ff: usize,
        /// Block alignment for `O_DIRECT` (4096 on most NVMe). The
        /// generated file size (`expert_size`) must be a multiple of
        /// this so the run path can read each expert with `O_DIRECT`
        /// without `EINVAL`. Must match what `run` is invoked with.
        #[arg(long, default_value_t = 4096)]
        block_align: usize,
        /// On-disk weight dtype for synthetic files: f32, f16, bf16, int8,
        /// q4k, q4_0, q8_0, or mxfp4. q5k, q6k, and mixed are GGUF/runtime
        /// formats and are not synthesized by this generator.
        #[arg(long, default_value = "f32")]
        dtype: String,
    },

    /// **Tier 2.** Repack a directory of `expert_<id>.bin` files into a
    /// single packed blob + JSON manifest for the packed storage layout.
    /// Experts are written back-to-back (one block-aligned `expert_size`
    /// slot each) in an order chosen by `--profile` / `--order` so the
    /// engine can coalesce co-fetched experts into single `preadv`s.
    Repack {
        /// Source directory containing `expert_<id>.bin` (or
        /// `expert_<layer>_<local>.bin` with `--num-experts-per-layer`).
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
        /// Output blob path (all expert payloads concatenated).
        #[arg(long)]
        out_blob: PathBuf,
        /// Output manifest path. Defaults to `<out_blob>.manifest.json`.
        #[arg(long)]
        out_manifest: Option<PathBuf>,
        /// Number of experts to pack (ids `0..num_experts`, unless
        /// `--order` restricts the set).
        #[arg(long, default_value_t = 64)]
        num_experts: u32,
        /// Bytes per expert. Must equal the source files' `expert_size`.
        #[arg(long, default_value_t = 16 * 1024 * 1024)]
        expert_size: usize,
        /// Block alignment (must match `gen-data` / the source files).
        #[arg(long, default_value_t = 4096)]
        block_align: usize,
        /// Disable `O_DIRECT` when reading the source files (needed on
        /// tmpfs / macOS / CI).
        #[arg(long)]
        no_direct: bool,
        /// Experts per layer for layer-qualified source naming.
        #[arg(long)]
        num_experts_per_layer: Option<u32>,
        /// Order experts hottest-first using a routing-frequency profile
        /// JSON (as produced by `run --profile-out`). Unobserved experts
        /// are appended in numeric order. Ignored if `--order` is set.
        #[arg(long)]
        profile: Option<PathBuf>,
        /// Explicit physical layout: a file listing expert ids (one per
        /// line, or a JSON array). Overrides `--profile`. Only the listed
        /// experts are packed.
        #[arg(long)]
        order: Option<PathBuf>,
    },

    /// Run the token-generation simulation against the on-disk experts.
    Run {
        /// Directory with `expert_<id>.bin` files. May also contain a
        /// `metadata.json` written by `scripts/extract_mixtral_experts.py`,
        /// in which case `num_experts`, `d_model`, `d_ff`, `top_k`, and
        /// `expert_size` are auto-loaded (CLI flags still override).
        #[arg(long, default_value = "./data")]
        data_dir: PathBuf,
        /// Total number of experts in the model.
        #[arg(long, default_value_t = 64)]
        num_experts: u32,
        /// Bytes per expert. Must equal what was used in `gen-data`.
        #[arg(long, default_value_t = 16 * 1024 * 1024)]
        expert_size: usize,
        /// Hidden / residual-stream dimension. Must match `gen-data`.
        #[arg(long, default_value_t = 512)]
        d_model: usize,
        /// Intermediate FFN dimension. Must match `gen-data`.
        #[arg(long, default_value_t = 2048)]
        d_ff: usize,
        /// LRU cache + buffer pool capacity (resident experts at once).
        ///
        /// The whole point of this engine is that experts stream from
        /// SSD; making the cache big defeats that. The default is 4
        /// slots and the engine warns if more than 16 are requested.
        #[arg(long, default_value_t = 4)]
        cache_slots: usize,
        /// Top-K experts the router activates per token.
        #[arg(long, default_value_t = 2)]
        top_k: usize,
        /// Number of tokens to simulate.
        #[arg(long, default_value_t = 200)]
        tokens: u64,
        /// Probe Rayon worker counts on this machine/model/backend before
        /// initializing the process-wide pool, then run with the winner.
        #[arg(long)]
        autotune_rayon: bool,
        /// Number of tokens each Rayon autotune probe runs.
        #[arg(
            long,
            default_value_t = crate::rayon_autotune::DEFAULT_AUTOTUNE_TOKENS,
            value_name = "N",
            value_parser = parse_positive_u64_value
        )]
        autotune_tokens: u64,
        /// Number of repeated fine probes for each finalist candidate.
        #[arg(
            long,
            default_value_t = crate::rayon_autotune::DEFAULT_AUTOTUNE_REPEATS,
            value_name = "N",
            value_parser = parse_positive_usize_value
        )]
        autotune_repeats: usize,
        /// Number of tokens for the cheap coarse autotune pass.
        #[arg(
            long,
            default_value_t = crate::rayon_autotune::DEFAULT_AUTOTUNE_COARSE_TOKENS,
            value_name = "N",
            value_parser = parse_positive_u64_value
        )]
        autotune_coarse_tokens: u64,
        /// Number of coarse winners promoted to repeated fine probes.
        #[arg(
            long,
            default_value_t = crate::rayon_autotune::DEFAULT_AUTOTUNE_TOP_CANDIDATES,
            value_name = "N",
            value_parser = parse_positive_usize_value
        )]
        autotune_top_candidates: usize,
        /// Print a concise coarse/fine Rayon autotune result table.
        #[arg(long)]
        autotune_print_table: bool,
        /// Slow-regime cutoff for the worst fine-probe p95.
        #[arg(
            long,
            default_value_t = crate::rayon_autotune::DEFAULT_SLOW_P95_THRESHOLD_MS,
            value_name = "MS",
            value_parser = parse_positive_f64_value
        )]
        autotune_slow_p95_ms: f64,
        /// Slow-regime cutoff for the worst fine-probe p99.
        #[arg(
            long,
            default_value_t = crate::rayon_autotune::DEFAULT_SLOW_P99_THRESHOLD_MS,
            value_name = "MS",
            value_parser = parse_positive_f64_value
        )]
        autotune_slow_p99_ms: f64,
        /// Use a low-confidence autotune result for this run.
        ///
        /// Without this flag, low-confidence results are treated as failed
        /// selection and the startup thread resolver falls back to a saved
        /// profile or MER's default sizing.
        #[arg(long)]
        allow_low_confidence_rayon_autotune: bool,
        /// Predictive prefetch fanout (how many candidates to issue per token).
        #[arg(long, default_value_t = 2)]
        predict_fanout: usize,
        /// **Look-ahead pipeline depth.** In serve mode, controls how many MoE
        /// layers ahead the speculator prefetches (the sliding window
        /// `layer + 1 ..= layer + pipeline_depth`), hiding cold SSD reads behind
        /// compute. `1` reproduces the legacy single-layer look-ahead.
        ///
        /// In `run`, this currently only scales speculative prefetch headroom
        /// (shadow buffer budget = `predict_fanout * pipeline_depth`).
        #[arg(long, default_value_t = crate::engine::DEFAULT_PIPELINE_DEPTH)]
        pipeline_depth: u32,
        /// Don't prefetch below this transition probability. The default
        /// (`0.0`) auto-scales the threshold to `2 / num_experts` so
        /// it remains achievable as the expert pool grows; pass an
        /// explicit positive value to override (e.g. `--predict-min-prob 0.05`).
        #[arg(long, default_value_t = 0.0)]
        predict_min_prob: f64,
        /// Disable O_DIRECT (use buffered reads). Required on tmpfs/overlay/CI
        /// and on macOS, where O_DIRECT is not supported. When set, the run
        /// summary's I/O latency includes any page-cache effects — see the
        /// startup warning emitted in this case.
        #[arg(long)]
        no_direct: bool,
        /// Block alignment for O_DIRECT (4096 on most NVMe).
        #[arg(long, default_value_t = 4096)]
        block_align: usize,
        /// PRNG seed for reproducible runs.
        #[arg(long, default_value_t = 0xC0FFEE)]
        seed: u64,
        /// On-disk weight dtype. Accepts f32, f16, bf16, int8, q4k, q4_0,
        /// q5k, q6k, q8_0, mxfp4, or mixed. Must match the generated or
        /// converted expert dataset.
        #[arg(long, default_value = "f32")]
        dtype: String,
        /// Fraction (`0.1..=1.0`) of input dimensions loaded per expert
        /// when partial column loading is enabled. `1.0` (default)
        /// disables partial loading. The forward pass still produces
        /// finite, correct-shape outputs for any value in range; lower
        /// fractions trade a small amount of accuracy for proportionally
        /// less compute / dequant energy.
        #[arg(long, default_value_t = 1.0)]
        partial_load_fraction: f64,
        /// After an expert has been observed in routing this many times,
        /// pin it permanently in the LRU cache. `0` (default) disables
        /// frequency-based pinning. Pinned experts are never reloaded
        /// from SSD, eliminating their I/O energy.
        #[arg(long, default_value_t = 0)]
        pin_after_observations: u64,
        /// Optional alias map JSON: `{ "src_id": canonical_id, ... }`.
        /// Pairs of experts the offline analyser flagged as numerically
        /// near-identical share a single resident copy at runtime,
        /// eliminating duplicate SSD reads.
        #[arg(long)]
        alias_map: Option<PathBuf>,
        /// Use the Linux `io_uring` storage backend with registered
        /// fixed buffers (one syscall to enqueue many reads, kernel
        /// reads directly into pre-pinned pool buffers). Requires the
        /// `io_uring` cargo feature; without it this flag logs a
        /// warning and the engine falls back to the default `pread(2)`
        /// path.
        #[arg(long)]
        io_uring: bool,
        /// Sleep this many micros between tokens (0 = as fast as possible).
        #[arg(long, default_value_t = 0)]
        token_pause_us: u64,
        /// Force-route the first token to these expert ids (comma-separated).
        /// Spec example: `--first-token 3,7`.
        #[arg(long, value_delimiter = ',')]
        first_token: Vec<u32>,
        /// Disable predictive prefetching entirely (for ablation).
        #[arg(long)]
        no_prefetch: bool,
        /// **I/O-only benchmarking mode**: skip the SwiGLU FFN forward
        /// pass entirely; still read every expert from SSD and XOR every
        /// byte to force the read to fully materialise. Use this to
        /// isolate the SSD-streaming cost from FFN compute.
        #[arg(long)]
        io_only: bool,
        /// **Force SSD reads.** Refuse to run with optimisations that
        /// would let the OS serve experts from RAM (page cache) instead
        /// of the device. Concretely: requires `O_DIRECT` (i.e. the run
        /// fails if `--no-direct` is also set on Linux). On macOS, where
        /// O_DIRECT is unavailable, this flag prints a warning and runs
        /// in best-effort mode.
        #[arg(long)]
        force_ssd: bool,
        /// Number of cluster groups for the router's first-order Markov
        /// chain (default 4: matches the gist's example). Each cluster
        /// is a group of experts that the router prefers to keep
        /// activating consecutively.
        #[arg(long, default_value_t = 4)]
        router_clusters: usize,
        /// Probability the Markov router stays inside its current
        /// expert cluster on each step. Higher = stronger temporal
        /// locality = more prefetch signal.
        #[arg(long, default_value_t = 0.9)]
        router_intra_p: f64,
        /// Optional path to a precomputed transition matrix. Whitespace-
        /// separated `f64` values, row-major, `num_experts^2` entries.
        /// Overrides `--router-clusters` / `--router-intra-p` when set.
        #[arg(long)]
        router_matrix: Option<PathBuf>,
        /// Optional path to a real **gating-network** weight matrix
        /// (`f32` little-endian, row-major, shape `[num_experts × d_model]`).
        ///
        /// When set, the run loop bypasses the deterministic Markov
        /// `TopKRouter` and instead computes per-token routing the way
        /// production Mixtral does: `softmax(W_gate · x) → top-K`.
        /// Each routed expert is still streamed from the SSD via the
        /// LRU cache (`Engine::moe_step`), so the SSD-bandwidth /
        /// cache-hit metrics reported at the end are directly
        /// comparable to the legacy Markov path.
        ///
        /// File format: bare little-endian `f32`s, no header. Generate
        /// one with `numpy.tofile` from a real Mixtral checkpoint, or
        /// use the seeded synthetic fallback if you only want to
        /// exercise the path (omit this flag to keep the legacy
        /// Markov router).
        ///
        /// May also point at a **directory** of per-layer `gate_<L>.bin`
        /// files (the same naming the model loader uses): they are
        /// auto-discovered, sorted by layer index, and concatenated in
        /// order, so you don't have to `cat` them into one file first.
        #[arg(long)]
        gate_weights: Option<PathBuf>,
        /// Optional path to write a JSONL **routing trace** to. Each
        /// line records one token's `{token, layer, experts,
        /// cache_hit}`, suitable for offline analysis with
        /// `scripts/compute_transition_matrix.py` and the
        /// `validate-predictor` subcommand.
        #[arg(long)]
        trace_out: Option<PathBuf>,
        /// Initialise the GPU compute backend before running the
        /// benchmark so the FFN forward pass uses GPU matmul where
        /// available. The run path also installs bounded logical admission
        /// plus physical expert-weight residency. Falls back to the default CPU backend
        /// with a warning if GPU init fails.
        #[arg(long)]
        gpu: bool,
        /// Physical expert-weight cap and logical admission budget, in MiB
        /// (only with `--gpu`). The 4 GiB default fits ~40
        /// Mixtral-8x7B Q4 experts (~99 MiB each — 512 MiB would hold
        /// barely 5); lower it on cards with less free VRAM.
        #[arg(long, default_value_t = 4096)]
        gpu_cache_mb: usize,
        /// Enable the **neural speculator** (arm `M`): a 2-layer MLP
        /// trained online against the gate's actual top-K. Predicts
        /// from the residual stream — the same feature the gate sees —
        /// so it is the strongest single prefetch signal and the one
        /// that actually drives `speculate_layer_ahead`. Off by default
        /// so the legacy Markov-only path is unchanged; turn it on to
        /// measure whether the predictive arms move the hit rate.
        #[arg(long)]
        speculator: bool,
        /// Hidden width of the speculator MLP (only when `--speculator`).
        #[arg(long, default_value_t = 128)]
        speculator_hidden_dim: usize,
        /// Top-K experts pulled from the speculator each step. `0`
        /// inherits `--top-k`.
        #[arg(long, default_value_t = 0)]
        speculator_top_k: usize,
        /// Enable the **locality monitor** (arm `L`): a sliding window
        /// over recent activations whose hot set is unioned into the
        /// prefetch set *and* pinned in the LRU so genuinely hot experts
        /// stop being evicted by cold ones — a frequency-aware upgrade
        /// over plain LRU eviction.
        #[arg(long)]
        locality: bool,
        /// Locality sliding-window size, in routing observations.
        #[arg(long, default_value_t = 256)]
        locality_window: usize,
        /// Heat threshold: an expert is "hot" once it appears in this
        /// fraction of the locality window. `0.10` ≈ 10% of recent
        /// activations.
        #[arg(long, default_value_t = 0.10)]
        locality_threshold_pct: f32,
        /// Enable the **per-layer expert-affinity** arm: folds co-fired
        /// and disk-adjacent neighbours of high-confidence predictions
        /// into the prefetch union. Only effective on multi-layer runs
        /// (`--num-experts-per-layer` set).
        #[arg(long)]
        affinity: bool,
        /// Number of co-fired neighbours pulled per seed (with `--affinity`).
        #[arg(long, default_value_t = 4)]
        affinity_neighbors_k: usize,
        /// Exponential-decay epoch for the affinity counters, in
        /// cumulative observations (with `--affinity`).
        #[arg(long, default_value_t = 10_000)]
        affinity_decay_epoch: u64,
        /// **Tier 4 — adaptive prefetch governor.** Throttle speculative
        /// prefetches by measured precision (consumed / completed) and
        /// foreground-read contention, so low-value speculation can't
        /// inflate the latency of the foreground misses that actually
        /// block token generation. Off by default (legacy unbounded
        /// admission). This is the highest-leverage knob on a
        /// bandwidth-bound SSD.
        #[arg(long)]
        prefetch_governor: bool,
        /// Precision floor / optimistic EWMA seed for the governor, in
        /// `[0, 1]` (with `--prefetch-governor`).
        #[arg(long, default_value_t = 0.05)]
        prefetch_precision_floor: f64,
        /// Per-outstanding-foreground-read multiplier on the governor's
        /// admission threshold (with `--prefetch-governor`). Higher ⇒
        /// speculation backs off harder while real misses are in flight.
        #[arg(long, default_value_t = 1.0)]
        prefetch_contention_weight: f64,
        /// **Tier 4 — cost-aware eviction.** Evict the coldest resident
        /// by decaying heat score instead of strict LRU, so a hot expert
        /// that briefly fell to the LRU tail isn't dumped ahead of a
        /// one-shot cold expert. Off by default (pure LRU).
        #[arg(long)]
        cost_aware_eviction: bool,
        /// **Tier 3 — per-layer pre-gate predictor.** Train an online
        /// layer-L→L+1 conditional map and drive high-precision
        /// next-layer prefetch from it. Off by default.
        #[arg(long)]
        pregate: bool,
        /// **Tier 1 — static residency.** Fraction of the global expert
        /// namespace to pin permanently in RAM (the hottest experts), in
        /// `(0, 1]`. `0.0` (default) disables it. Lifts the hit-rate
        /// ceiling above the bare cache fraction on a *skewed* workload.
        #[arg(long, default_value_t = 0.0)]
        static_residency_fraction: f64,
        /// Tokens to observe before deriving the online static-residency
        /// hot set (ignored when `--static-residency-profile` is given).
        #[arg(long, default_value_t = 0)]
        static_residency_warmup_tokens: u64,
        /// Path to an offline expert-popularity profile JSON
        /// (`{ "<id>": <count> }`) to seed static residency at startup.
        /// When omitted, the hot set is derived online.
        #[arg(long)]
        static_residency_profile: Option<String>,
        /// Write the run's accumulated route-observation profile to this
        /// JSON path at shutdown (consumable by
        /// `--static-residency-profile` on a later run).
        #[arg(long)]
        profile_out: Option<String>,
        /// **Benchmark workload.** `synthetic` (default) keeps the legacy
        /// uniform-i.i.d. stream (the engine/gate routes its own hidden
        /// state); `skewed` drives `moe_step` from a Zipf-popular,
        /// Markov-correlated expert generator (so static residency and
        /// the predictors are exercisable); `replay` replays a recorded
        /// JSONL routing trace via `--replay-trace`.
        #[arg(long, default_value = "synthetic")]
        workload: String,
        /// Zipf exponent for `--workload skewed` (larger ⇒ more skew;
        /// `1.0` ≈ classic Zipf, `0.0` ≈ uniform).
        #[arg(long, default_value_t = 1.1)]
        zipf_s: f64,
        /// Markov stay-probability for `--workload skewed`, in `[0, 1]`:
        /// the chance a token reuses the previous token's expert set
        /// (temporal correlation the predictors can exploit).
        #[arg(long, default_value_t = 0.0)]
        workload_correlation: f64,
        /// JSONL routing trace to replay with `--workload replay` (the
        /// `--trace-out` format).
        #[arg(long)]
        replay_trace: Option<String>,
        /// Number of transformer layers, used to size the affinity
        /// matrix. `1` (default) is the single-namespace synthetic
        /// benchmark.
        #[arg(long, default_value_t = 1)]
        num_layers: u32,
        /// Experts **per layer** for a layer-qualified id geometry. When
        /// set, `speculate_layer_ahead` restricts the speculator's
        /// global head to the next layer's slice and actually prefetches
        /// `layer + 1 ..= layer + pipeline_depth` ahead — the mechanism
        /// that hides SSD latency behind compute. Leave unset for the
        /// flat single-namespace benchmark (no layer-ahead).
        #[arg(long)]
        num_experts_per_layer: Option<u32>,
        /// **Tier 2 — packed storage.** Read every expert from this single
        /// packed blob (produced by the `repack` subcommand) instead of
        /// one file per expert. Requires `--packed-manifest`. Adjacent
        /// experts are fetched with coalesced `preadv` syscalls.
        #[arg(long)]
        packed_blob: Option<PathBuf>,
        /// **Tier 2.** JSON manifest (`id -> offset,len`) accompanying
        /// `--packed-blob`. Required when `--packed-blob` is set.
        #[arg(long)]
        packed_manifest: Option<PathBuf>,
    },

    /// Convert a GGUF checkpoint (Mixtral-style) into the engine's
    /// per-expert binary format plus a `metadata.json` and the dense
    /// weight files [`RealModel::from_dir`] consumes. Phase 2.
    GgufConvert {
        /// Path to a normal `*.gguf` file or any file in a standard
        /// `*-00001-of-00005.gguf` shard set.
        #[arg(long)]
        gguf_path: PathBuf,
        /// Output directory. Created if it doesn't exist.
        #[arg(long)]
        out_dir: PathBuf,
        /// Override the number of layers (defaults to
        /// `llama.block_count` from the GGUF metadata).
        #[arg(long, default_value_t = 0)]
        num_layers: usize,
        /// Override the experts-per-layer (defaults to
        /// `llama.expert_count` from the GGUF metadata).
        #[arg(long, default_value_t = 0)]
        num_experts: usize,
        /// Skip the Unified Tensor Header (U.T.H.) prefix on every
        /// `expert_<id>.bin`. By default the converter emits a 64-byte
        /// page-padded UTH so the loader knows the dtype + shape +
        /// tile-hint before reading any weight bytes; pass this flag
        /// to produce legacy bare-payload files for compatibility
        /// with consumers that pre-date UTH support.
        #[arg(long, default_value_t = false)]
        no_uth: bool,
        /// Use the legacy eager GGUF reader (slurps the entire file
        /// into RAM before slicing tensors out). The default is the
        /// streaming reader which keeps only the header + tensor
        /// info table resident — a strict win for ≥ 100 GB
        /// checkpoints. The eager path is still useful in tests and
        /// for small fixtures.
        #[arg(long, default_value_t = false)]
        legacy_eager: bool,
        /// **Native quantised pass-through.** When set and the
        /// source GGUF stores its expert tensors as `Q4_0`, `Q4_K`,
        /// `Q5_K`, `Q6_K`, or `Q8_0`, write the raw quantised block stream to disk
        /// instead of dequantising to F32 first. The output
        /// `expert_<id>.bin` stays quantized. Mixed projection triples
        /// are written with UTH2 headers. If quantized output is not
        /// possible, conversion fails before writing expert files.
        #[arg(long, default_value_t = false)]
        native_quant: bool,
        /// Convert only routed expert blobs and metadata. Dense
        /// transformer tensors are skipped deliberately; the output is
        /// valid for expert-streaming benchmarks, not full-model runs.
        #[arg(long, default_value_t = false)]
        experts_only: bool,
    },

    /// Validate a converted expert dataset before running inference.
    ValidateData {
        /// Directory containing `expert_<id>.bin` files and metadata.json.
        #[arg(long)]
        data_dir: PathBuf,
    },

    /// Replay a routing trace through the predictive prefetcher and
    /// print per-K hit-rate statistics. Phase 6.
    ValidatePredictor {
        /// Path to a JSONL routing trace (produced by `run --trace-out`).
        #[arg(long)]
        trace: PathBuf,
        /// LRU cache size to simulate. Repeat the flag to evaluate
        /// multiple sizes in one run (e.g. `--cache-slots 4
        /// --cache-slots 8 --cache-slots 16`). Defaults to a sweep of
        /// 2, 4, 8, 16.
        #[arg(long)]
        cache_slots: Vec<usize>,
    },

    /// Start the OpenAI-compatible HTTP server (Phase 6 / 8 / 9).
    ///
    /// Reads server, model, storage, and tokenizer settings from a TOML
    /// config file. The engine is built exactly as in `run`, but instead
    /// of streaming a fixed token count it stays up serving requests on
    /// `POST /v1/completions`, `POST /v1/chat/completions`, and exports
    /// Prometheus metrics on `GET /metrics`.
    Serve {
        /// Path to the TOML config file. See `config.toml` at the
        /// repository root for an example.
        #[arg(long)]
        config: PathBuf,
    },

    /// Benchmark the real transformer path without starting HTTP.
    ///
    /// Uses the same TOML config surface as `serve`, requires
    /// `[real_transformer] enabled = true`, and reports prompt/decode
    /// timing separately from the legacy synthetic `run` sustained_tps
    /// metric.
    BenchReal {
        /// Path to the TOML config file.
        #[arg(long)]
        config: PathBuf,
        /// Prompt text to encode and benchmark.
        #[arg(long, conflicts_with = "request_json")]
        prompt: Option<String>,
        /// OpenAI-style request JSON containing `prompt`, or chat
        /// `messages`, and optionally `max_tokens`.
        #[arg(long, conflicts_with = "prompt")]
        request_json: Option<PathBuf>,
        /// Number of completion tokens to generate. Overrides
        /// `max_tokens` from `--request-json` when both are supplied.
        #[arg(long)]
        output_tokens: Option<usize>,
        /// Warmup runs to execute before measurements.
        #[arg(long, default_value_t = 1)]
        warmup_runs: usize,
        /// Measured runs to report.
        #[arg(long, default_value_t = 1)]
        measured_runs: usize,
        /// Cache reset policy between runs.
        #[arg(long, value_enum, default_value_t = BenchRealCacheReset::Keep)]
        cache_reset: BenchRealCacheReset,
        /// Force deterministic greedy decoding for benchmark parity.
        #[arg(long)]
        greedy: bool,
        /// Output format.
        #[arg(long, value_enum, default_value_t = BenchRealOutputFormat::Human)]
        format: BenchRealOutputFormat,
    },

    /// Benchmark the ordinary production GPU-native real-model token loop.
    ///
    /// This is throughput evidence only. It requires strict greedy execution,
    /// exact hardware identity, and a complete real checkpoint, and it never
    /// invokes diagnostic token-step variants.
    BenchGpuNativeReal {
        /// Path to the strict production GPU-native TOML config.
        #[arg(long)]
        config: PathBuf,
        /// Prompt text to tokenize once and benchmark.
        #[arg(long, conflicts_with = "request_json")]
        prompt: Option<String>,
        /// OpenAI-style request JSON containing `prompt` or chat `messages`.
        #[arg(long, conflicts_with = "prompt")]
        request_json: Option<PathBuf>,
        /// Exact number of generated tokens; must be at least two.
        #[arg(long)]
        output_tokens: Option<usize>,
        /// Warmup requests, excluded from measured aggregates.
        #[arg(long, default_value_t = 1)]
        warmup_runs: usize,
        /// Measured requests retained individually in the report.
        #[arg(long, default_value_t = 1)]
        measured_runs: usize,
        /// Cache/runtime reset policy.
        #[arg(long, value_enum, default_value_t = BenchRealCacheReset::Keep)]
        cache_reset: BenchRealCacheReset,
        /// Required first-baseline sampling contract.
        #[arg(long, required = true)]
        greedy: bool,
        /// Exact authoritative adapter name expected at runtime.
        #[arg(long)]
        expected_adapter_name: String,
        /// Write the typed JSON benchmark report here instead of stdout.
        #[arg(long)]
        report_out: Option<PathBuf>,
    },

    /// Production-path PR2-A.2 control/treatment qualification. Control
    /// forces the legacy exact sequential source helper; treatment exercises
    /// the same ordinary production batching seam used by normal serving.
    #[command(name = "qualify-gpu-native-demand-source-concurrency-production")]
    QualifyGpuNativeDemandSourceConcurrencyProduction {
        /// Path to the frozen strict production GPU-native TOML config.
        #[arg(long)]
        config: PathBuf,
        /// Exact authoritative adapter name required for both isolated arms.
        #[arg(long)]
        expected_adapter_name: String,
        /// Required destination for the typed v2 qualification report.
        #[arg(long)]
        report_out: PathBuf,
    },

    /// Qualify strict real-checkpoint inference with CPU dense/attention/KV/
    /// router/head planes and native-Q4_0 routed experts on a hardware GPU.
    QualifyHybridQ4 {
        /// Path to the TOML config file.
        #[arg(long)]
        config: PathBuf,
        /// Prompt text to encode and qualify.
        #[arg(long, conflicts_with = "request_json")]
        prompt: Option<String>,
        /// OpenAI-style request JSON containing `prompt` or chat `messages`.
        #[arg(long, conflicts_with = "prompt")]
        request_json: Option<PathBuf>,
        /// Completion-token count; overrides request JSON `max_tokens`.
        #[arg(long)]
        output_tokens: Option<usize>,
        /// Strict greedy warmups before the single measured request.
        #[arg(long, default_value_t = 0)]
        warmup_runs: usize,
        /// Write the typed JSON report here instead of stdout.
        #[arg(long)]
        report_out: Option<PathBuf>,
        /// Opaque reference to separately collected external GPU-memory evidence.
        #[arg(long)]
        external_gpu_memory_artifact: Option<String>,
    },

    /// Qualify canonical Q4_0 WGSL math and one complete extracted routed
    /// expert against MER's authoritative CPU implementation.
    QualifyHybridQ4Parity {
        /// Path to the strict Hybrid native-Q4_0 TOML config.
        #[arg(long)]
        config: PathBuf,
        /// Global expert id: layer * num_experts_per_layer + layer-local id.
        #[arg(long)]
        expert_id: u32,
        /// Exact wgpu adapter name required for this qualification run.
        #[arg(long)]
        expected_adapter_name: String,
        /// Write the typed JSON report here instead of stdout.
        #[arg(long)]
        report_out: Option<PathBuf>,
    },

    /// Compare fixed-corpus greedy token IDs between isolated strict CPU and
    /// strict Hybrid Q4_0 real-checkpoint execution.
    QualifyHybridQ4GreedyParity {
        /// Path to the strict Hybrid native-Q4_0 TOML config.
        #[arg(long)]
        config: PathBuf,
        /// Exact wgpu adapter name required for every Hybrid corpus case.
        #[arg(long)]
        expected_adapter_name: String,
        /// Write the typed JSON report here instead of stdout.
        #[arg(long)]
        report_out: Option<PathBuf>,
    },

    /// Strictly compare 16 greedy tokens across the fixed four-case corpus
    /// between the authoritative Hybrid-boundary CPU reference and the full
    /// production GPU-native Q4 token loop.
    QualifyGpuNativeQ4GreedyParity {
        /// Path to the strict GPU-native Q4 TOML config.
        #[arg(long)]
        config: PathBuf,
        /// Exact wgpu adapter name required for every GPU-native corpus case.
        #[arg(long)]
        expected_adapter_name: String,
        /// Write the versioned typed JSON report here instead of stdout.
        #[arg(long)]
        report_out: Option<PathBuf>,
    },

    /// Qualify GPU-native semantic correctness over the independent frozen v2 holdout corpus.
    #[command(name = "qualify-gpu-native-semantic-parity-v2")]
    QualifyGpuNativeSemanticParityV2 {
        /// Path to the strict GPU-native Q4 TOML config.
        #[arg(long)]
        config: PathBuf,
        /// Exact wgpu adapter name required for both isolated runtimes.
        #[arg(long)]
        expected_adapter_name: String,
        /// Required versioned typed JSON qualification destination.
        #[arg(long)]
        report_out: PathBuf,
    },

    /// Attribute the frozen first GPU-native semantic-parity v2 holdout failures.
    #[command(name = "diagnose-gpu-native-v2-holdout-failures")]
    DiagnoseGpuNativeV2HoldoutFailures {
        /// Path to the strict GPU-native Q4 TOML config.
        #[arg(long)]
        config: PathBuf,
        /// Exact wgpu adapter name required for the isolated GPU runtime.
        #[arg(long)]
        expected_adapter_name: String,
        /// Exact frozen v2 qualification report to consume as immutable evidence.
        #[arg(long)]
        v2_report: PathBuf,
        /// Required frozen SHA256 of the supplied v2 report.
        #[arg(long)]
        expected_v2_report_sha256: String,
        /// Required typed diagnostic JSON destination.
        #[arg(long)]
        report_out: PathBuf,
    },

    /// Attribute internal stages of the frozen GPU-native Q4 expert failures.
    #[command(name = "diagnose-gpu-native-q4-expert-stages")]
    DiagnoseGpuNativeQ4ExpertStages {
        /// Path to the strict GPU-native Q4 TOML config.
        #[arg(long)]
        config: PathBuf,
        /// Exact wgpu adapter name required for the isolated GPU runtime.
        #[arg(long)]
        expected_adapter_name: String,
        /// Exact frozen bea attribution report consumed as immutable evidence.
        #[arg(long)]
        bea_report: PathBuf,
        /// Required frozen SHA256 of the supplied bea report.
        #[arg(long)]
        expected_bea_report_sha256: String,
        /// Required typed diagnostic JSON destination.
        #[arg(long)]
        report_out: PathBuf,
    },

    /// Audit the frozen GPU-native Q4 evidence against ordinary exact-f32 CPU references.
    #[command(name = "diagnose-gpu-native-f32-reference-boundary")]
    DiagnoseGpuNativeF32ReferenceBoundary {
        /// Path to the strict GPU-native Q4 TOML config used to build a CPU-only reference.
        #[arg(long)]
        config: PathBuf,
        /// Exact frozen GPU-native semantic-parity v2 report.
        #[arg(long)]
        v2_report: PathBuf,
        /// Required frozen SHA256 of the supplied v2 report.
        #[arg(long)]
        expected_v2_report_sha256: String,
        /// Exact frozen GPU-native Q4 expert-stage attribution report.
        #[arg(long)]
        stage_report: PathBuf,
        /// Required frozen SHA256 of the supplied stage report.
        #[arg(long)]
        expected_stage_report_sha256: String,
        /// Required typed diagnostic JSON destination.
        #[arg(long)]
        report_out: PathBuf,
    },

    /// Attribute the frozen Spanish first generated-token divergence across isolated exact-f32 CPU, historical-f16 CPU, and GPU-native planes.
    #[command(name = "diagnose-gpu-native-spanish-first-token-attribution")]
    DiagnoseGpuNativeSpanishFirstTokenAttribution {
        /// Path to the strict GPU-native Q4 TOML configuration.
        #[arg(long)]
        config: PathBuf,
        /// Exact frozen GPU-native semantic-parity v2 report.
        #[arg(long)]
        v2_report: PathBuf,
        /// Required frozen SHA256 of the supplied v2 report.
        #[arg(long)]
        expected_v2_report_sha256: String,
        /// Exact frozen GPU-native Q4 expert-stage attribution report.
        #[arg(long)]
        stage_report: PathBuf,
        /// Required frozen SHA256 of the supplied stage report.
        #[arg(long)]
        expected_stage_report_sha256: String,
        /// Exact frozen f32 reference-boundary audit report.
        #[arg(long)]
        boundary_audit_report: PathBuf,
        /// Required frozen SHA256 of the supplied boundary-audit report.
        #[arg(long)]
        expected_boundary_audit_report_sha256: String,
        /// Exact production GPU adapter name; this diagnostic requires NVIDIA L4.
        #[arg(long)]
        expected_adapter_name: String,
        /// Required typed diagnostic JSON destination.
        #[arg(long)]
        report_out: PathBuf,
    },

    /// Collect reproducibility and complete first-token CPU/Hybrid logit
    /// evidence for the fixed json-transformation case.
    DiagnoseHybridQ4GreedyDivergence {
        /// Path to the strict Hybrid native-Q4_0 TOML config.
        #[arg(long)]
        config: PathBuf,
        /// Exact wgpu adapter name required for Hybrid diagnostic workers.
        #[arg(long)]
        expected_adapter_name: String,
        /// Required typed diagnostic JSON destination.
        #[arg(long)]
        report_out: PathBuf,
    },
    /// Diagnose first mathematical divergence between production Hybrid Q4 reference and full GPU-native execution.
    DiagnoseGpuNativeQ4FirstDivergence {
        /// Strict GPU-native Q4 configuration.
        #[arg(long)]
        config: PathBuf,
        /// Expected GPU adapter name (e.g. "NVIDIA L4").
        #[arg(long)]
        expected_adapter_name: String,
        /// Target case (e.g. "json-transformation").
        #[arg(long)]
        case: String,
        /// Optional typed diagnostic report output path.
        #[arg(long)]
        report_out: Option<PathBuf>,
    },
    /// Attribute a target GPU-native router rank permutation without changing inference or qualification semantics.
    DiagnoseGpuNativeRouterRankDivergence {
        /// Strict GPU-native Q4 configuration.
        #[arg(long)]
        config: PathBuf,
        /// Exact required GPU adapter name (for example, "NVIDIA L4").
        #[arg(long)]
        expected_adapter_name: String,
        /// Fixed-corpus case name.
        #[arg(long)]
        case: String,
        /// Zero-based generated-token position to capture.
        #[arg(long)]
        generated_position: usize,
        /// Zero-based transformer layer to capture.
        #[arg(long)]
        layer: usize,
        /// Required versioned diagnostic JSON destination.
        #[arg(long)]
        report_out: PathBuf,
    },
    /// Measure the semantic effect of a target GPU-native expert-rank permutation without changing inference or qualification semantics.
    DiagnoseGpuNativeExpertPermutationSemanticParity {
        /// Strict GPU-native Q4 configuration.
        #[arg(long)]
        config: PathBuf,
        /// Exact required GPU adapter name (for example, "NVIDIA L4").
        #[arg(long)]
        expected_adapter_name: String,
        /// Fixed-corpus case name.
        #[arg(long)]
        case: String,
        /// Zero-based generated-token position to capture.
        #[arg(long)]
        generated_position: usize,
        /// Zero-based transformer layer to capture.
        #[arg(long)]
        layer: usize,
        /// Required versioned diagnostic JSON destination.
        #[arg(long)]
        report_out: PathBuf,
    },
    /// Survey semantic parity across every routing event in the frozen greedy corpus without producing a qualification PASS.
    DiagnoseGpuNativeSemanticParityCorpus {
        /// Strict GPU-native Q4 configuration.
        #[arg(long)]
        config: PathBuf,
        /// Exact required GPU adapter name (for example, "NVIDIA L4").
        #[arg(long)]
        expected_adapter_name: String,
        /// Required versioned diagnostic JSON destination.
        #[arg(long)]
        report_out: PathBuf,
    },
    /// Diagnose first internal mathematical divergence inside layer-0 attention between CPU reference and GPU-native execution.
    #[command(name = "diagnose-gpu-native-layer0-attention-first-divergence")]
    DiagnoseGpuNativeLayer0AttentionFirstDivergence {
        /// Strict GPU-native Q4 configuration.
        #[arg(long)]
        config: PathBuf,
        /// Expected GPU adapter name (e.g. "NVIDIA L4").
        #[arg(long)]
        expected_adapter_name: String,
        /// Target case (e.g. "json-transformation").
        #[arg(long, default_value = "json-transformation")]
        case: String,
        /// Optional typed diagnostic report output path.
        #[arg(long)]
        report_out: Option<PathBuf>,
        /// Optional Slice 11B report path to cross-check final-position layer-0 post-attention evidence.
        #[arg(long)]
        slice11b_report: Option<PathBuf>,
    },



    /// Private same-binary worker for one strict-Hybrid greedy-parity plane.
    #[command(name = "greedy-parity-hybrid-worker-internal", hide = true)]
    GreedyParityHybridWorkerInternal {
        /// The same strict Hybrid config parsed by the parent orchestrator.
        #[arg(long)]
        config: PathBuf,
    },

    /// Private same-binary CPU/Hybrid first-token logit worker.
    #[command(name = "greedy-parity-logit-worker-internal", hide = true)]
    GreedyParityLogitWorkerInternal {
        #[arg(long)]
        config: PathBuf,
    },

    /// Benchmark dense matvec backends on the target Qwen3-Coder CPU shapes.
    ///
    /// This is intentionally separate from `bench-real`: it isolates Q/K/V/O,
    /// router gate, and LM-head projection kernels so operators can choose a
    /// production `[real_transformer].dense_matvec_backend` without running a
    /// full checkpoint. The LM-head shape allocates about 1.2 GiB of weights;
    /// use `--skip-lm-head` for quick local smoke runs.
    MatvecMicrobench {
        /// Backend(s) to benchmark. Repeat or pass comma-separated values.
        /// Defaults to matrixmultiply, rayon, and rayon-matrixmultiply.
        #[arg(long = "backend", value_delimiter = ',')]
        backend: Vec<crate::parallel::DenseMatvecBackend>,
        /// Warmup iterations per shape/backend.
        #[arg(long, default_value_t = 1)]
        warmup_runs: usize,
        /// Measured iterations per shape/backend.
        #[arg(long, default_value_t = 3)]
        measured_runs: usize,
        /// Skip the 151936 × 2048 LM-head shape.
        #[arg(long)]
        skip_lm_head: bool,
        /// Emit JSON instead of human-readable rows.
        #[arg(long)]
        json: bool,
    },

    /// Compare native zero-copy Q8_0 experts with the retained Candle reference.
    /// Build with `--features q8-candle-reference` to include both paths.
    Q8ExpertMicrobench {
        /// Expert hidden width (Qwen3-Coder-30B-A3B: 2048).
        #[arg(long, default_value_t = 2048)]
        d_model: usize,
        /// Expert intermediate width (Qwen3-Coder-30B-A3B: 768).
        #[arg(long, default_value_t = 768)]
        d_ff: usize,
        /// Warmup resident-hit iterations per path.
        #[arg(long, default_value_t = 1)]
        warmup_runs: usize,
        /// Measured resident-hit iterations per path.
        #[arg(long, default_value_t = 5)]
        measured_runs: usize,
        /// Native backend to exercise. `auto` mirrors the conservative
        /// production policy; `all` includes every supported native backend
        /// plus the Candle reference.
        #[arg(long, value_enum, default_value_t = Q8BenchKernel::Auto)]
        kernel: Q8BenchKernel,
    },

    /// Count heap allocations for the old transformer wrappers vs request scratch buffers.
    ///
    /// Requires `--features alloc-count`. The synthetic layer keeps this
    /// benchmark independent of checkpoint files while exercising the same
    /// attention, RMSNorm, router, residual, and MoE-combine methods used by
    /// the real decode loop.
    ScratchAllocMicrobench {
        /// Hidden size for the synthetic transformer layer. Must be a multiple of 4.
        #[arg(long, default_value_t = 32)]
        d_model: usize,
        /// Number of routed experts in the synthetic gate.
        #[arg(long, default_value_t = 8)]
        num_experts: usize,
        /// Experts selected per token.
        #[arg(long, default_value_t = 2)]
        top_k: usize,
        /// Tokens to run before resetting allocation counters.
        #[arg(long, default_value_t = 1)]
        warmup_tokens: usize,
        /// Tokens measured after warmup. Defaults below the 16-token KV page size.
        #[arg(long, default_value_t = 8)]
        measured_tokens: usize,
        /// Emit JSON instead of human-readable rows.
        #[arg(long)]
        json: bool,
    },
    /// Native terminal dashboard. Polls a running `serve` instance and renders
    /// SSD/RAM/logical-GPU-admission hits, cache state, admission utilisation,
    /// and I/O reactor activity. Pure
    /// observability; the dashboard does not mutate engine state.
    ///
    /// Requires the binary to be built with the `tui` cargo feature
    /// (on by default). With `--no-default-features` this subcommand
    /// exits with a helpful error message.
    Monitor {
        /// Base URL of the `serve` HTTP endpoint to poll. Defaults to
        /// `http://127.0.0.1:8080` to match the example config.
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        url: String,
        /// How often to refresh the dashboard, in milliseconds.
        #[arg(long, default_value_t = 500)]
        refresh_ms: u64,
    },
}

/// Resolve the effective `predict_min_prob` for a given expert-pool size.
///
/// A configured value of `0.0` (or negative — treated identically) is the
/// "auto" sentinel and scales the threshold to `2 / num_experts`, so
/// the Laplace-smoothed posteriors in [`PredictiveLoader::predict_next`]
/// can actually clear the gate as the pool grows (a fixed `0.05` becomes
/// mathematically unreachable past ~20 experts). Any positive value is
/// passed through unchanged, preserving operator overrides.
fn resolve_predict_min_prob(configured: f64, num_experts: u32) -> f64 {
    if configured > 0.0 {
        configured
    } else {
        let n = num_experts.max(1) as f64;
        2.0 / n
    }
}

fn parse_positive_rayon_threads(s: &str) -> Result<usize, String> {
    parse_positive_usize_value(s).map_err(|e| {
        if e == "value must be > 0" {
            "--rayon-threads must be > 0".to_string()
        } else {
            e
        }
    })
}

fn parse_positive_usize_value(s: &str) -> Result<usize, String> {
    let n = s
        .parse::<usize>()
        .map_err(|_| format!("expected a positive integer, got {s:?}"))?;
    if n == 0 {
        Err("value must be > 0".to_string())
    } else {
        Ok(n)
    }
}

fn parse_positive_u64_value(s: &str) -> Result<u64, String> {
    let n = s
        .parse::<u64>()
        .map_err(|_| format!("expected a positive integer, got {s:?}"))?;
    if n == 0 {
        Err("value must be > 0".to_string())
    } else {
        Ok(n)
    }
}

fn parse_positive_f64_value(s: &str) -> Result<f64, String> {
    let n = s
        .parse::<f64>()
        .map_err(|_| format!("expected a positive number, got {s:?}"))?;
    if n.is_finite() && n > 0.0 {
        Ok(n)
    } else {
        Err("value must be a finite number > 0".to_string())
    }
}

fn init_logging(filter: &str, worker_protocol_stdout: bool) {
    let env_filter = EnvFilter::try_new(filter).unwrap_or_else(|_| EnvFilter::new("info"));
    if worker_protocol_stdout {
        // The private worker reserves stdout for exactly one typed JSON value.
        // All inherited diagnostics must stay on stderr or the parent rejects
        // the protocol as corrupted.
        let _ = tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_target(false)
            .with_level(true)
            .with_writer(std::io::stderr)
            .try_init();
    } else {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_target(false)
            .with_level(true)
            .try_init();
    }
}

fn rayon_env_override_present() -> bool {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .is_some_and(|n| n > 0)
}

fn rayon_hard_override_present(cli: Option<usize>, config: Option<usize>) -> bool {
    rayon_env_override_present() || cli.is_some() || config.is_some()
}

fn rayon_autotune_key_for_cli(
    cli: &Cli,
    affinity: &crate::numa::EffectiveCpuAffinity,
) -> Option<crate::rayon_autotune::CpuAutotuneKey> {
    match &cli.cmd {
        Cmd::Run {
            data_dir,
            num_experts,
            expert_size,
            d_model,
            d_ff,
            top_k,
            dtype,
            workload,
            gate_weights,
            gpu,
            pipeline_depth,
            ..
        } => Some(crate::rayon_autotune::CpuAutotuneKey {
            machine_fingerprint: crate::rayon_autotune::machine_fingerprint_from_affinity(
                affinity,
            ),
            model_fingerprint: format!(
                "run;data_dir={};experts={};expert_size={};d_model={};d_ff={};top_k={};dtype={};workload={};gate={};pipeline_depth={}",
                data_dir.display(),
                num_experts,
                expert_size,
                d_model,
                d_ff,
                top_k,
                dtype,
                workload,
                gate_weights
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "none".to_string()),
                pipeline_depth,
            ),
            backend_fingerprint: format!(
                "dense={};gpu={};features={}",
                crate::parallel::default_dense_matvec_backend(),
                gpu,
                build_features().join(",")
            ),
        }),
        _ => None,
    }
}

fn run_profile_path(cli: &Cli) -> Option<PathBuf> {
    match &cli.cmd {
        Cmd::Run { data_dir, .. } => Some(crate::rayon_autotune::default_profile_path(data_dir)),
        _ => None,
    }
}

fn load_profiled_rayon_threads(
    cli: &Cli,
    key: Option<&crate::rayon_autotune::CpuAutotuneKey>,
) -> Option<usize> {
    let key = key?;
    let path = run_profile_path(cli)?;
    let profile = crate::rayon_autotune::load_profile(&path, key)?;
    if !profile.reusable_by_default() && !cli.reuse_low_confidence_rayon_profile {
        warn!(
            path = %path.display(),
            threads = profile.threads,
            confidence = profile.confidence.as_str(),
            reason = %profile.selection_reason,
            "skipping low-confidence Rayon autotune profile; pass --reuse-low-confidence-rayon-profile to opt in"
        );
        return None;
    }
    info!(
        path = %path.display(),
        threads = profile.threads,
        confidence = profile.confidence.as_str(),
        median_p50_ms = profile.median_p50_ms,
        worst_p95_ms = profile.worst_p95_ms,
        worst_p99_ms = ?profile.worst_p99_ms,
        median_sustained_tps = profile.median_sustained_tps,
        "loaded placement-aware Rayon autotune profile"
    );
    Some(profile.threads)
}

fn load_high_confidence_profile_threads(
    cli: &Cli,
    key: Option<&crate::rayon_autotune::CpuAutotuneKey>,
) -> Option<usize> {
    let key = key?;
    let path = run_profile_path(cli)?;
    let profile = crate::rayon_autotune::load_profile(&path, key)?;
    (profile.confidence == crate::rayon_autotune::RayonAutotuneConfidence::High)
        .then_some(profile.threads)
}

fn maybe_run_startup_rayon_autotune(
    cli: &Cli,
    raw_args: &[OsString],
    affinity: &crate::numa::EffectiveCpuAffinity,
    key: Option<&crate::rayon_autotune::CpuAutotuneKey>,
    config_threads: Option<usize>,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    let Cmd::Run {
        autotune_rayon,
        autotune_tokens,
        autotune_repeats,
        autotune_coarse_tokens,
        autotune_top_candidates,
        autotune_print_table,
        autotune_slow_p95_ms,
        autotune_slow_p99_ms,
        allow_low_confidence_rayon_autotune,
        ..
    } = &cli.cmd
    else {
        return Ok(None);
    };
    if !*autotune_rayon {
        return Ok(None);
    }
    if std::env::var_os(crate::rayon_autotune::AUTOTUNE_PROBE_ENV).is_some() {
        return Ok(None);
    }
    if rayon_hard_override_present(cli.rayon_threads, config_threads) {
        info!(
            "Rayon autotune requested, but an explicit thread-count override is present; skipping probes"
        );
        return Ok(None);
    }

    let candidates = crate::rayon_autotune::default_thread_candidates(affinity.logical_cores);
    if candidates.is_empty() {
        return Ok(None);
    }
    info!(
        effective_cpu_mask = %affinity.display,
        logical_cores = affinity.logical_cores,
        candidates = ?candidates,
        coarse_tokens = autotune_coarse_tokens,
        fine_tokens = autotune_tokens,
        repeats = autotune_repeats,
        top_candidates = autotune_top_candidates,
        slow_p95_threshold_ms = autotune_slow_p95_ms,
        slow_p99_threshold_ms = autotune_slow_p99_ms,
        "starting Rayon autotune probes"
    );

    let mut probes = Vec::new();
    for threads in candidates.iter().copied() {
        let probe = run_rayon_autotune_probe_observation(
            raw_args,
            *autotune_coarse_tokens,
            threads,
            crate::rayon_autotune::RayonAutotuneProbeStage::Coarse,
            1,
        );
        log_rayon_autotune_probe_observation(&probe);
        probes.push(probe);
    }

    let slow_p95_threshold_ms = *autotune_slow_p95_ms;
    let slow_p99_threshold_ms = *autotune_slow_p99_ms;
    let coarse_summaries = crate::rayon_autotune::summarize_candidate_results(
        &candidates,
        1,
        &probes,
        slow_p95_threshold_ms,
        slow_p99_threshold_ms,
    );
    let previous_profile_threads = load_high_confidence_profile_threads(cli, key);
    let fine_candidates = crate::rayon_autotune::fine_thread_candidates(
        affinity.logical_cores,
        *autotune_top_candidates,
        previous_profile_threads,
        &coarse_summaries,
    );
    if fine_candidates.is_empty() {
        if *autotune_print_table {
            let table = crate::rayon_autotune::format_autotune_table(&probes, &coarse_summaries);
            info!("Rayon autotune table\n{table}");
        }
        return Err("Rayon autotune coarse pass did not produce any successful probe".into());
    }
    info!(
        fine_candidates = ?fine_candidates,
        previous_high_confidence_profile_threads = ?previous_profile_threads,
        repeats = autotune_repeats,
        fine_tokens = autotune_tokens,
        "Rayon autotune coarse pass selected fine candidates"
    );

    for threads in fine_candidates.iter().copied() {
        for repeat in 1..=*autotune_repeats {
            let probe = run_rayon_autotune_probe_observation(
                raw_args,
                *autotune_tokens,
                threads,
                crate::rayon_autotune::RayonAutotuneProbeStage::Fine,
                repeat,
            );
            log_rayon_autotune_probe_observation(&probe);
            probes.push(probe);
        }
    }

    let fine_probes: Vec<_> = probes
        .iter()
        .filter(|p| p.stage == crate::rayon_autotune::RayonAutotuneProbeStage::Fine)
        .cloned()
        .collect();
    let candidate_summaries = crate::rayon_autotune::summarize_candidate_results(
        &fine_candidates,
        *autotune_repeats,
        &fine_probes,
        slow_p95_threshold_ms,
        slow_p99_threshold_ms,
    );
    let selection = crate::rayon_autotune::select_best_candidate(&candidate_summaries)
        .ok_or("Rayon autotune did not produce any successful fine probe")?;
    if *autotune_print_table {
        let table = crate::rayon_autotune::format_autotune_table(&probes, &candidate_summaries);
        info!("Rayon autotune table\n{table}");
    }
    let best = selection.selected.clone();
    info!(
        threads = best.threads,
        confidence = selection.confidence.as_str(),
        median_p50_ms = ?best.median_p50_ms,
        worst_p95_ms = ?best.worst_p95_ms,
        worst_p99_ms = ?best.worst_p99_ms,
        median_sustained_tps = ?best.median_sustained_tps,
        slow_p95_threshold_ms,
        slow_p99_threshold_ms,
        reason = %selection.reason,
        "Rayon autotune selected worker count"
    );

    let use_selected = selected_autotune_threads(
        &selection,
        *allow_low_confidence_rayon_autotune,
    );
    if use_selected.is_none() {
        warn!(
            threads = best.threads,
            confidence = selection.confidence.as_str(),
            "Rayon autotune result is low-confidence; falling back to saved profile/default threads unless --allow-low-confidence-rayon-autotune is passed"
        );
    }

    if use_selected.is_some() {
        if let (Some(path), Some(key)) = (run_profile_path(cli), key) {
            let median_p50_ms = best.median_p50_ms.unwrap_or(0.0);
            let worst_p95_ms = best.worst_p95_ms.unwrap_or(0.0);
            let median_sustained_tps = best.median_sustained_tps.unwrap_or(0.0);
            let profile = crate::rayon_autotune::RayonAutotuneProfile {
                threads: best.threads,
                effective_cpu_mask: affinity.cpus.clone(),
                effective_cpu_mask_display: Some(affinity.display.clone()),
                logical_cores: affinity.logical_cores,
                repeats: *autotune_repeats,
                p50_ms: median_p50_ms,
                p95_ms: worst_p95_ms,
                p99_ms: best.worst_p99_ms,
                sustained_tps: median_sustained_tps,
                median_p50_ms,
                worst_p95_ms,
                worst_p99_ms: best.worst_p99_ms,
                median_sustained_tps,
                confidence: selection.confidence,
                selection_reason: selection.reason.clone(),
                candidate_results: candidate_summaries.clone(),
                probe_results: probes.clone(),
            };
            if let Err(e) = crate::rayon_autotune::save_profile(&path, key, profile) {
                warn!(path = %path.display(), error = %e, "failed to save Rayon autotune profile");
            } else if selection.confidence == crate::rayon_autotune::RayonAutotuneConfidence::Low {
                warn!(
                    path = %path.display(),
                    confidence = selection.confidence.as_str(),
                    "saved low-confidence Rayon autotune profile; normal runs will not reuse it without --reuse-low-confidence-rayon-profile"
                );
            } else {
                info!(
                    path = %path.display(),
                    confidence = selection.confidence.as_str(),
                    "saved placement-aware Rayon autotune profile"
                );
            }
        }
    } else if let Some(path) = run_profile_path(cli) {
        warn!(
            path = %path.display(),
            confidence = selection.confidence.as_str(),
            "not saving low-confidence Rayon autotune result as the default profile"
        );
    }

    Ok(use_selected)
}

fn selected_autotune_threads(
    selection: &crate::rayon_autotune::RayonAutotuneSelection,
    allow_low_confidence: bool,
) -> Option<usize> {
    if selection.confidence == crate::rayon_autotune::RayonAutotuneConfidence::Low
        && !allow_low_confidence
    {
        None
    } else {
        Some(selection.selected.threads)
    }
}

fn run_rayon_autotune_probe_observation(
    raw_args: &[OsString],
    autotune_tokens: u64,
    threads: usize,
    stage: crate::rayon_autotune::RayonAutotuneProbeStage,
    repeat: usize,
) -> crate::rayon_autotune::RayonAutotuneProbeObservation {
    match run_rayon_autotune_probe(raw_args, autotune_tokens, threads) {
        Ok(Some(result)) => {
            crate::rayon_autotune::RayonAutotuneProbeObservation::from_probe_result(
                stage,
                repeat,
                autotune_tokens,
                result,
            )
        }
        Ok(None) => crate::rayon_autotune::RayonAutotuneProbeObservation::invalid(
            threads,
            stage,
            repeat,
            autotune_tokens,
            "probe produced no parseable result",
        ),
        Err(e) => crate::rayon_autotune::RayonAutotuneProbeObservation::invalid(
            threads,
            stage,
            repeat,
            autotune_tokens,
            e.to_string(),
        ),
    }
}

fn log_rayon_autotune_probe_observation(
    probe: &crate::rayon_autotune::RayonAutotuneProbeObservation,
) {
    if probe.valid {
        info!(
            stage = probe.stage.as_str(),
            threads = probe.threads,
            repeat = probe.repeat,
            tokens = probe.tokens,
            p50_ms = ?probe.p50_ms,
            p95_ms = ?probe.p95_ms,
            p99_ms = ?probe.p99_ms,
            sustained_tps = ?probe.sustained_tps,
            "Rayon autotune probe complete"
        );
    } else {
        warn!(
            stage = probe.stage.as_str(),
            threads = probe.threads,
            repeat = probe.repeat,
            tokens = probe.tokens,
            reason = %probe.reason.as_deref().unwrap_or("invalid"),
            "Rayon autotune probe failed"
        );
    }
}

fn run_rayon_autotune_probe(
    raw_args: &[OsString],
    autotune_tokens: u64,
    threads: usize,
) -> Result<Option<crate::rayon_autotune::RayonAutotuneProbeResult>, Box<dyn std::error::Error>> {
    let exe = raw_args
        .first()
        .ok_or("missing argv[0] for Rayon autotune probe")?;
    let child_args = autotune_child_args(raw_args, autotune_tokens, threads);
    let output = Command::new(exe)
        .args(&child_args)
        .env(crate::rayon_autotune::AUTOTUNE_PROBE_ENV, "1")
        .env("RAYON_NUM_THREADS", threads.to_string())
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "probe exited with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(parse_autotune_probe_output(&output.stdout))
}

fn autotune_child_args(
    raw_args: &[OsString],
    autotune_tokens: u64,
    threads: usize,
) -> Vec<OsString> {
    let mut out = Vec::new();
    let mut iter = raw_args.iter().skip(1).peekable();
    while let Some(arg) = iter.next() {
        let s = arg.to_string_lossy();
        if s == "--autotune-rayon"
            || s.starts_with("--autotune-rayon=")
            || s.starts_with("--autotune-tokens=")
            || s.starts_with("--autotune-repeats=")
            || s.starts_with("--autotune-coarse-tokens=")
            || s.starts_with("--autotune-top-candidates=")
            || s.starts_with("--autotune-slow-p95-ms=")
            || s.starts_with("--autotune-slow-p99-ms=")
            || s == "--autotune-print-table"
            || s == "--allow-low-confidence-rayon-autotune"
            || s.starts_with("--tokens=")
            || s.starts_with("--rayon-threads=")
        {
            continue;
        }
        if s == "--autotune-tokens"
            || s == "--autotune-repeats"
            || s == "--autotune-coarse-tokens"
            || s == "--autotune-top-candidates"
            || s == "--autotune-slow-p95-ms"
            || s == "--autotune-slow-p99-ms"
            || s == "--tokens"
            || s == "--rayon-threads"
        {
            let _ = iter.next();
            continue;
        }
        out.push(arg.clone());
    }
    out.push(OsString::from("--tokens"));
    out.push(OsString::from(autotune_tokens.to_string()));
    out.push(OsString::from("--rayon-threads"));
    out.push(OsString::from(threads.to_string()));
    out
}

fn parse_autotune_probe_output(
    stdout: &[u8],
) -> Option<crate::rayon_autotune::RayonAutotuneProbeResult> {
    let body = String::from_utf8_lossy(stdout);
    for line in body.lines() {
        let Some(json) = line.strip_prefix("MER_RAYON_AUTOTUNE_RESULT ") else {
            continue;
        };
        if let Ok(result) = serde_json::from_str(json) {
            return Some(result);
        }
    }
    None
}

fn startup_config_path(cmd: &Cmd) -> Option<&Path> {
    match cmd {
        Cmd::Serve { config }
        | Cmd::BenchReal { config, .. }
        | Cmd::BenchGpuNativeReal { config, .. }
        | Cmd::QualifyGpuNativeDemandSourceConcurrencyProduction { config, .. }
        | Cmd::QualifyHybridQ4 { config, .. }
        | Cmd::QualifyHybridQ4Parity { config, .. }
        | Cmd::QualifyHybridQ4GreedyParity { config, .. }
        | Cmd::QualifyGpuNativeQ4GreedyParity { config, .. }
        | Cmd::QualifyGpuNativeSemanticParityV2 { config, .. }
        | Cmd::DiagnoseGpuNativeV2HoldoutFailures { config, .. }
        | Cmd::DiagnoseGpuNativeQ4ExpertStages { config, .. }
        | Cmd::DiagnoseGpuNativeF32ReferenceBoundary { config, .. }
        | Cmd::DiagnoseGpuNativeSpanishFirstTokenAttribution { config, .. }
        | Cmd::DiagnoseHybridQ4GreedyDivergence { config, .. }
        | Cmd::DiagnoseGpuNativeQ4FirstDivergence { config, .. }
        | Cmd::DiagnoseGpuNativeRouterRankDivergence { config, .. }
        | Cmd::DiagnoseGpuNativeExpertPermutationSemanticParity { config, .. }
        | Cmd::DiagnoseGpuNativeSemanticParityCorpus { config, .. }
        | Cmd::DiagnoseGpuNativeLayer0AttentionFirstDivergence { config, .. }
        | Cmd::GreedyParityHybridWorkerInternal { config }
        | Cmd::GreedyParityLogitWorkerInternal { config } => Some(config.as_path()),
        _ => None,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let raw_args: Vec<OsString> = std::env::args_os().collect();
    let cli = Cli::parse();
    let worker_protocol_stdout = matches!(
        cli.cmd,
        Cmd::GreedyParityHybridWorkerInternal { .. }
            | Cmd::GreedyParityLogitWorkerInternal { .. }
    );
    init_logging(&cli.log, worker_protocol_stdout);

    // Load only enough startup configuration to resolve process-level
    // runtime controls before CPU placement / Rayon exist. The command
    // handlers still own their normal config reconciliation and runtime
    // construction below.
    let mut startup_config = match startup_config_path(&cli.cmd) {
        Some(path) => Some(crate::config::Config::from_file(path)?),
        None => None,
    };

    let config_cpu_mask = startup_config
        .as_ref()
        .and_then(|cfg| cfg.performance.cpu_mask.as_deref());
    let legacy_mer_pin_cores = std::env::var(crate::numa::MER_PIN_CORES_ENV).ok();
    let cpu_mask_request = crate::numa::resolve_cpu_mask_request(
        cli.cpu_mask.as_deref(),
        config_cpu_mask,
        legacy_mer_pin_cores.as_deref(),
    )
    .map_err(|e| format!("CPU mask selection: {e}"))?;

    // Apply explicit placement before any Tokio runtime or Rayon worker is
    // spawned, then read the effective affinity that the OS actually gave us.
    let pin = crate::numa::apply_cpu_mask_request(cpu_mask_request.as_ref());
    let startup_pinned = matches!(pin, crate::numa::PinResult::Pinned { .. });
    let effective_affinity = crate::numa::effective_cpu_affinity();
    info!(
        requested_cpu_mask = cpu_mask_request
            .as_ref()
            .map(|r| r.display.as_str())
            .unwrap_or("none"),
        requested_cpu_mask_source = cpu_mask_request
            .as_ref()
            .map(|r| r.source.as_str())
            .unwrap_or("none"),
        effective_cpu_mask = %effective_affinity.display,
        logical_cores = effective_affinity.logical_cores,
        pinning_result = %pin.as_log_line(),
        "startup CPU placement"
    );

    // `MER_PIN_CORES` is now consumed centrally at process start via the
    // `numa` module. Clear it so later code cannot re-apply affinity and
    // drift from this startup contract.
    // SAFETY: this runs during single-threaded startup, before the Tokio
    // runtime or Rayon worker pool is created and before this process
    // introduces any concurrent environment access.
    unsafe {
        std::env::remove_var(crate::numa::MER_PIN_CORES_ENV);
    }

    let progress_config_secs = startup_config
        .as_ref()
        .and_then(|cfg| cfg.performance.progress_timeout_secs);
    let progress_watchdog = crate::rayon_autotune::normalize_progress_timeout_secs(
        cli.progress_timeout_secs,
        progress_config_secs,
    );
    match progress_watchdog.timeout {
        Some(timeout) => info!(
            progress_timeout_secs = timeout.as_secs(),
            "progress watchdog enabled"
        ),
        None => info!("progress watchdog disabled"),
    }

    // Size the shared compute (`rayon`) pool now: after affinity pinning so
    // its workers inherit the startup mask, after CLI/config/env precedence
    // is resolved, and before any matmul touches it. By default it preserves
    // MER's existing logical-core-minus-headroom policy under the effective
    // affinity mask; `RAYON_NUM_THREADS` remains the hard reproduction
    // override above autotune/profile/default selection.
    let rayon_config_threads = startup_config
        .as_ref()
        .and_then(|cfg| cfg.performance.rayon_threads);
    let run_autotune_key = rayon_autotune_key_for_cli(&cli, &effective_affinity);
    let profile_threads = if rayon_hard_override_present(cli.rayon_threads, rayon_config_threads) {
        None
    } else {
        load_profiled_rayon_threads(&cli, run_autotune_key.as_ref())
    };
    let autotuned_threads = maybe_run_startup_rayon_autotune(
        &cli,
        &raw_args,
        &effective_affinity,
        run_autotune_key.as_ref(),
        rayon_config_threads,
    )?;
    let rayon_selection = crate::parallel::resolve_rayon_threads_from_env(
        cli.rayon_threads,
        rayon_config_threads,
        autotuned_threads,
        profile_threads,
    )
    .map_err(|e| format!("rayon thread selection: {e}"))?;
    crate::parallel::init_global_pool(rayon_selection, effective_affinity.logical_cores);

    // Log the selected math kernel backend once. The dispatcher itself
    // is lazy, but emitting this at startup gives ops a single line in
    // the journal that tells them "you're running the scalar path"
    // before they go looking for missing AVX-512 perf.
    crate::kernels::log_backend();

    // Install the default plugin-system execution context (gist Task 2).
    // Logged on the same boot line so ops can see both the low-level
    // CPU-feature dispatch and the high-level backend in one place.
    //
    // For the `serve` and `qualify-hybrid-q4` subcommands we **defer** the
    // default install until their handlers have loaded the TOML config — the
    // hybrid compute offload (`[real_transformer].compute_offload`, gist
    // Part 2 fix #5) is resolved there, and it must run
    // *before* `install_default` claims the OnceLock. Other
    // subcommands keep the immediate install so their math path is
    // ready as soon as `main` returns into them.
    //
    // The `run` subcommand grows one extra wrinkle: when invoked with
    // `--gpu` it must leave the execution-context `OnceLock` free until
    // `cmd_run` has applied any dataset metadata and can resolve against
    // the effective expert dtype/geometry. GPU initialization failure is
    // fatal; legacy deterministic dtype/geometry bypass remains visible in
    // the resolved component plan.
    let run_gpu_requested = matches!(cli.cmd, Cmd::Run { gpu: true, .. });
    if !matches!(
        cli.cmd,
        Cmd::Serve { .. }
            | Cmd::QualifyHybridQ4 { .. }
            | Cmd::QualifyHybridQ4Parity { .. }
            | Cmd::QualifyHybridQ4GreedyParity { .. }
            | Cmd::QualifyGpuNativeQ4GreedyParity { .. }
            | Cmd::DiagnoseHybridQ4GreedyDivergence { .. }
            | Cmd::DiagnoseGpuNativeRouterRankDivergence { .. }
            | Cmd::GreedyParityHybridWorkerInternal { .. }
            | Cmd::GreedyParityLogitWorkerInternal { .. }
    ) && !run_gpu_requested
    {
        crate::backend::install_default();
        let b = crate::backend::current();
        info!(
            backend = b.device_name(),
            compute_plane = b.compute_plane(),
            "math backend installed"
        );
    }

    match cli.cmd {
        Cmd::GenData {
            data_dir,
            num_experts,
            expert_size,
            d_model,
            d_ff,
            block_align,
            dtype,
        } => cmd_gen_data(
            &data_dir,
            num_experts,
            expert_size,
            d_model,
            d_ff,
            block_align,
            &dtype,
        ),
        Cmd::Repack {
            data_dir,
            out_blob,
            out_manifest,
            num_experts,
            expert_size,
            block_align,
            no_direct,
            num_experts_per_layer,
            profile,
            order,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_repack(RepackArgs {
                data_dir,
                out_blob,
                out_manifest,
                num_experts,
                expert_size,
                block_align,
                no_direct,
                num_experts_per_layer,
                profile,
                order,
            }))
        }
        Cmd::Run { .. } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(async move {
                if let Cmd::Run {
                    data_dir,
                    num_experts,
                    expert_size,
                    d_model,
                    d_ff,
                    cache_slots,
                    top_k,
                    tokens,
                    autotune_rayon: _,
                    autotune_tokens: _,
                    predict_fanout,
                    predict_min_prob,
                    no_direct,
                    block_align,
                    seed,
                    dtype,
                    partial_load_fraction,
                    pin_after_observations,
                    alias_map,
                    io_uring,
                    token_pause_us,
                    first_token,
                    no_prefetch,
                    io_only,
                    force_ssd,
                    router_clusters,
                    router_intra_p,
                    router_matrix,
                    gate_weights,
                    trace_out,
                    gpu,
                    gpu_cache_mb,
                    pipeline_depth,
                    speculator,
                    speculator_hidden_dim,
                    speculator_top_k,
                    locality,
                    locality_window,
                    locality_threshold_pct,
                    affinity,
                    affinity_neighbors_k,
                    affinity_decay_epoch,
                    prefetch_governor,
                    prefetch_precision_floor,
                    prefetch_contention_weight,
                    cost_aware_eviction,
                    pregate,
                    static_residency_fraction,
                    static_residency_warmup_tokens,
                    static_residency_profile,
                    profile_out,
                    workload,
                    zipf_s,
                    workload_correlation,
                    replay_trace,
                    num_layers,
                    num_experts_per_layer,
                    packed_blob,
                    packed_manifest,
                    ..
                } = cli.cmd
                {
                    let dtype =
                        crate::inference::WeightDtype::from_str_opt(&dtype).ok_or_else(|| {
                            format!(
                                "--dtype: unknown value {dtype:?} (supported: {SUPPORTED_RUNTIME_DTYPES})"
                            )
                        })?;
                    cmd_run(
                        RunArgs {
                            data_dir,
                            num_experts,
                            expert_size,
                            d_model,
                            d_ff,
                            cache_slots,
                            top_k,
                            tokens,
                            predict_fanout,
                            predict_min_prob,
                            no_direct,
                            block_align,
                            seed,
                            dtype,
                            partial_load_fraction,
                            pin_after_observations,
                            alias_map_path: alias_map,
                            io_uring,
                            token_pause_us,
                            first_token,
                            no_prefetch,
                            io_only,
                            force_ssd,
                            router_clusters,
                            router_intra_p,
                            router_matrix,
                            gate_weights,
                            trace_out,
                            gpu_cache_mb: gpu.then_some(gpu_cache_mb),
                            pipeline_depth,
                            speculator,
                            speculator_hidden_dim,
                            speculator_top_k,
                            locality,
                            locality_window,
                            locality_threshold_pct,
                            affinity,
                            affinity_neighbors_k,
                            affinity_decay_epoch,
                            prefetch_governor,
                            prefetch_precision_floor,
                            prefetch_contention_weight,
                            cost_aware_eviction,
                            pregate,
                            static_residency_fraction,
                            static_residency_warmup_tokens,
                            static_residency_profile,
                            profile_out,
                            workload,
                            zipf_s,
                            workload_correlation,
                            replay_trace,
                            num_layers,
                            num_experts_per_layer,
                            packed_blob,
                            packed_manifest,
                        },
                        startup_pinned,
                        progress_watchdog,
                    )
                    .await
                } else {
                    unreachable!()
                }
            })
        }
        Cmd::Serve { config } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_serve(config))
        }
        Cmd::BenchReal {
            config,
            prompt,
            request_json,
            output_tokens,
            warmup_runs,
            measured_runs,
            cache_reset,
            greedy,
            format,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_bench_real(BenchRealArgs {
                config,
                prompt,
                request_json,
                output_tokens,
                warmup_runs,
                measured_runs,
                cache_reset,
                greedy,
                format,
                progress_watchdog,
            }))
        }
        Cmd::BenchGpuNativeReal {
            config,
            prompt,
            request_json,
            output_tokens,
            warmup_runs,
            measured_runs,
            cache_reset,
            greedy,
            expected_adapter_name,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(crate::gpu_native_real_benchmark::run_command(
                crate::gpu_native_real_benchmark::CommandArgs {
                    config,
                    prompt,
                    request_json,
                    output_tokens,
                    warmup_runs,
                    measured_runs,
                    cache_reset,
                    greedy,
                    expected_adapter_name,
                    report_out,
                    progress_watchdog,
                },
            ))
        }
        Cmd::QualifyGpuNativeDemandSourceConcurrencyProduction {
            config,
            expected_adapter_name,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(
                crate::gpu_native_demand_source_concurrency::run_production_command(
                    crate::gpu_native_demand_source_concurrency::CommandArgs {
                        config,
                        expected_adapter_name,
                        report_out,
                        progress_watchdog,
                    },
                ),
            )
        }
        Cmd::QualifyHybridQ4 {
            config,
            prompt,
            request_json,
            output_tokens,
            warmup_runs,
            report_out,
            external_gpu_memory_artifact,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_qualify_hybrid_q4(QualifyHybridQ4Args {
                config,
                prompt,
                request_json,
                output_tokens,
                warmup_runs,
                report_out,
                external_gpu_memory_artifact,
                progress_watchdog,
            }))
        }
        Cmd::QualifyHybridQ4Parity {
            config,
            expert_id,
            expected_adapter_name,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_qualify_hybrid_q4_parity(
                QualifyHybridQ4ParityArgs {
                    config,
                    expert_id,
                    expected_adapter_name,
                    report_out,
                    progress_watchdog,
                },
            ))
        }
        Cmd::QualifyHybridQ4GreedyParity {
            config,
            expected_adapter_name,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_qualify_hybrid_q4_greedy_parity(
                QualifyHybridQ4GreedyParityArgs {
                    config,
                    parsed_config: startup_config
                        .take()
                        .ok_or("greedy parity startup config was not parsed")?,
                    expected_adapter_name,
                    report_out,
                    progress_watchdog,
                },
            ))
        }
        Cmd::QualifyGpuNativeQ4GreedyParity {
            config,
            expected_adapter_name,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_qualify_gpu_native_q4_greedy_parity(
                QualifyGpuNativeQ4GreedyParityArgs {
                    config,
                    parsed_config: startup_config
                        .take()
                        .ok_or("GPU-native greedy parity startup config was not parsed")?,
                    expected_adapter_name,
                    report_out,
                    progress_watchdog,
                },
            ))
        }
        Cmd::QualifyGpuNativeSemanticParityV2 {
            config,
            expected_adapter_name,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(crate::gpu_native_semantic_parity_v2::run_qualification(
                config,
                startup_config
                    .take()
                    .ok_or("qualify-gpu-native-semantic-parity-v2 startup config was not parsed")?,
                expected_adapter_name,
                report_out,
                progress_watchdog,
            ))
        }
        Cmd::DiagnoseGpuNativeV2HoldoutFailures {
            config,
            expected_adapter_name,
            v2_report,
            expected_v2_report_sha256,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(
                crate::gpu_native_v2_holdout_failure_attribution::run_diagnostic(
                    config,
                    startup_config.take().ok_or(
                        "diagnose-gpu-native-v2-holdout-failures startup config was not parsed",
                    )?,
                    expected_adapter_name,
                    v2_report,
                    expected_v2_report_sha256,
                    report_out,
                    progress_watchdog,
                ),
            )
        }
        Cmd::DiagnoseGpuNativeQ4ExpertStages {
            config,
            expected_adapter_name,
            bea_report,
            expected_bea_report_sha256,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(
                crate::gpu_native_q4_expert_stage_attribution::run_diagnostic(
                    config,
                    startup_config.take().ok_or(
                        "diagnose-gpu-native-q4-expert-stages startup config was not parsed",
                    )?,
                    expected_adapter_name,
                    bea_report,
                    expected_bea_report_sha256,
                    report_out,
                    progress_watchdog,
                ),
            )
        }
        Cmd::DiagnoseGpuNativeF32ReferenceBoundary {
            config,
            v2_report,
            expected_v2_report_sha256,
            stage_report,
            expected_stage_report_sha256,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(
                crate::gpu_native_f32_reference_boundary_audit::run_diagnostic(
                    config,
                    startup_config.take().ok_or(
                        "diagnose-gpu-native-f32-reference-boundary startup config was not parsed",
                    )?,
                    v2_report,
                    expected_v2_report_sha256,
                    stage_report,
                    expected_stage_report_sha256,
                    report_out,
                    progress_watchdog,
                ),
            )
        }
        Cmd::DiagnoseGpuNativeSpanishFirstTokenAttribution {
            config,
            v2_report,
            expected_v2_report_sha256,
            stage_report,
            expected_stage_report_sha256,
            boundary_audit_report,
            expected_boundary_audit_report_sha256,
            expected_adapter_name,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(
                crate::gpu_native_spanish_first_token_attribution::run_diagnostic(
                    config,
                    startup_config.take().ok_or(
                        "diagnose-gpu-native-spanish-first-token-attribution startup config was not parsed",
                    )?,
                    v2_report,
                    expected_v2_report_sha256,
                    stage_report,
                    expected_stage_report_sha256,
                    boundary_audit_report,
                    expected_boundary_audit_report_sha256,
                    expected_adapter_name,
                    report_out,
                    progress_watchdog,
                ),
            )
        }
        Cmd::DiagnoseHybridQ4GreedyDivergence {
            config,
            expected_adapter_name,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_diagnose_hybrid_q4_greedy_divergence(
                DiagnoseHybridQ4GreedyDivergenceArgs {
                    config,
                    parsed_config: startup_config
                        .take()
                        .ok_or("logit diagnostic startup config was not parsed")?,
                    expected_adapter_name,
                    report_out,
                    progress_watchdog,
                },
            ))
        }
        Cmd::GreedyParityHybridWorkerInternal { config: _ } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_greedy_parity_hybrid_worker_internal(
                GreedyParityHybridWorkerArgs {
                    parsed_config: startup_config
                        .take()
                        .ok_or("greedy parity worker startup config was not parsed")?,
                    progress_watchdog,
                },
            ))
        }
        Cmd::GreedyParityLogitWorkerInternal { config: _ } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_greedy_parity_logit_worker_internal(
                GreedyParityHybridWorkerArgs {
                    parsed_config: startup_config
                        .take()
                        .ok_or("logit diagnostic worker startup config was not parsed")?,
                    progress_watchdog,
                },
            ))
        }
        Cmd::DiagnoseGpuNativeQ4FirstDivergence {
            config: _,
            expected_adapter_name,
            case,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_diagnose_gpu_native_q4_first_divergence(
                DiagnoseGpuNativeQ4FirstDivergenceArgs {
                    parsed_config: startup_config
                        .take()
                        .ok_or("diagnose-gpu-native-q4-first-divergence startup config was not parsed")?,
                    expected_adapter_name,
                    case,
                    report_out,
                    progress_watchdog,
                },
            ))
        }
        Cmd::DiagnoseGpuNativeRouterRankDivergence {
            config,
            expected_adapter_name,
            case,
            generated_position,
            layer,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_diagnose_gpu_native_router_rank_divergence(
                DiagnoseGpuNativeRouterRankDivergenceArgs {
                    config,
                    parsed_config: startup_config.take().ok_or(
                        "diagnose-gpu-native-router-rank-divergence startup config was not parsed",
                    )?,
                    expected_adapter_name,
                    case,
                    generated_position,
                    layer,
                    report_out,
                    progress_watchdog,
                },
            ))
        }
        Cmd::DiagnoseGpuNativeExpertPermutationSemanticParity {
            config,
            expected_adapter_name,
            case,
            generated_position,
            layer,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_diagnose_gpu_native_expert_permutation_semantic_parity(
                DiagnoseGpuNativeExpertPermutationSemanticParityArgs {
                    config,
                    parsed_config: startup_config.take().ok_or(
                        "diagnose-gpu-native-expert-permutation-semantic-parity startup config was not parsed",
                    )?,
                    expected_adapter_name,
                    case,
                    generated_position,
                    layer,
                    report_out,
                    progress_watchdog,
                },
            ))
        }
        Cmd::DiagnoseGpuNativeSemanticParityCorpus {
            config,
            expected_adapter_name,
            report_out,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(
                crate::gpu_native_semantic_parity_corpus::run_diagnostic(
                    config,
                    startup_config.take().ok_or(
                        "diagnose-gpu-native-semantic-parity-corpus startup config was not parsed",
                    )?,
                    expected_adapter_name,
                    report_out,
                    progress_watchdog,
                ),
            )
        }
        Cmd::DiagnoseGpuNativeLayer0AttentionFirstDivergence {
            config: _,
            expected_adapter_name,
            case,
            report_out,
            slice11b_report,
        } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_diagnose_gpu_native_layer0_attention_first_divergence(
                DiagnoseGpuNativeLayer0AttentionFirstDivergenceArgs {
                    parsed_config: startup_config
                        .take()
                        .ok_or("diagnose-gpu-native-layer0-attention-first-divergence startup config was not parsed")?,
                    expected_adapter_name,
                    case,
                    report_out,
                    slice11b_report,
                    progress_watchdog,
                },
            ))
        }

        Cmd::MatvecMicrobench {
            backend,
            warmup_runs,
            measured_runs,
            skip_lm_head,
            json,
        } => cmd_matvec_microbench(MatvecMicrobenchArgs {
            backends: backend,
            warmup_runs,
            measured_runs,
            skip_lm_head,
            json,
        }),
        Cmd::Q8ExpertMicrobench {
            d_model,
            d_ff,
            warmup_runs,
            measured_runs,
            kernel,
        } => cmd_q8_expert_microbench(d_model, d_ff, warmup_runs, measured_runs, kernel),
        Cmd::ScratchAllocMicrobench {
            d_model,
            num_experts,
            top_k,
            warmup_tokens,
            measured_tokens,
            json,
        } => cmd_scratch_alloc_microbench(ScratchAllocMicrobenchArgs {
            d_model,
            num_experts,
            top_k,
            warmup_tokens,
            measured_tokens,
            json,
        }),
        Cmd::GgufConvert {
            gguf_path,
            out_dir,
            num_layers,
            num_experts,
            no_uth,
            legacy_eager,
            native_quant,
            experts_only,
        } => cmd_gguf_convert(
            &gguf_path,
            &out_dir,
            num_layers,
            num_experts,
            !no_uth,
            legacy_eager,
            native_quant,
            experts_only,
        ),
        Cmd::ValidateData { data_dir } => cmd_validate_data(&data_dir),
        Cmd::ValidatePredictor { trace, cache_slots } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(cmd_validate_predictor(&trace, &cache_slots))
        }
        Cmd::Monitor { url, refresh_ms } => {
            #[cfg(feature = "tui")]
            {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()?;
                rt.block_on(crate::tui::run_monitor(&url, refresh_ms))
            }
            #[cfg(not(feature = "tui"))]
            {
                let _ = (url, refresh_ms);
                Err("monitor subcommand requires the `tui` cargo feature; \
                     rebuild without `--no-default-features` to enable it"
                    .into())
            }
        }
    }
}

/// Install the GPU compute backend for the `run` subcommand (gist
/// Fix 2).
///
/// Builds a bounded [`GpuExpertCache`], resolves one execution context, and
/// installs that same context for the engine to consume. `run --gpu` is an
/// explicit request, so initialization or installation failure is fatal.
///
/// Finding 5 (fail-closed GPU initialization): `--gpu` is an explicit request,
/// so both GPU initialisation failure and backend-installation failure are
/// fatal. Legacy deterministic expert compatibility bypass remains represented
/// by the resolved component plan. Operators who want best-effort GPU
/// initialization with CPU fallback use serving `compute_offload = "auto"`.
fn install_run_gpu_backend(
    gpu_cache_mb: usize,
    routed_expert_gpu_spec: crate::backend::RoutedExpertGpuSpec,
) -> Result<Arc<crate::backend::ExecutionContext>, Box<dyn std::error::Error>> {
    if gpu_cache_mb == 0 {
        return Err(
            "explicit run --gpu requires --gpu-cache-mb > 0 so routed experts can execute on GPU"
                .into(),
        );
    }
    // The KV-cache geometry below only sizes the dense-backbone cache,
    // which the synthetic `run` benchmark does not exercise — it routes
    // everything through `expert_matmul`.
    //
    // Give run-mode GPU promotion a bounded (but non-zero) budget so
    // repeated experts can become logically admitted and then physically
    // uploaded by the backend on demand. Sized by
    // `--gpu-cache-mb` (default 4 GiB — a single Mixtral-8x7B Q4
    // expert is ~99 MiB, so anything much smaller thrashes).
    let gpu_expert_cache = std::sync::Arc::new(crate::expert_cache::GpuExpertCache::new(
        gpu_cache_mb.saturating_mul(1024 * 1024),
        0.5,
        16,
    ));
    let execution_context = crate::backend::resolve_execution_context(
        crate::backend::ComputeOffload::Gpu,
        false,
        crate::backend::GpuBackendGeometry {
            num_layers: 1,
            max_seq_len: 1,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            v_head_dim: 1,
            q4_truncation_tolerance: 0,
        },
        routed_expert_gpu_spec,
        gpu_expert_cache.clone(),
    )
    .map_err(|e| format!("explicit --gpu request failed: {e}"))?;
    let backend = execution_context.primary_backend();
    let device_name = backend.device_name().to_string();
    let compute_plane = backend.compute_plane().to_string();
    crate::backend::set_execution_context(execution_context.clone())
        .map_err(|e| format!("explicit --gpu request failed: context installation failed ({e})"))?;
    if execution_context.plan().routed_experts() != crate::backend::ExecutionPlane::Gpu {
        warn!(
            dtype = routed_expert_gpu_spec.dtype.as_str(),
            d_model = routed_expert_gpu_spec.d_model,
            d_ff = routed_expert_gpu_spec.d_ff,
            routed_expert_plane = execution_context.plan().routed_experts().as_str(),
            "run --gpu resolved routed experts to CPU; expert compute is not GPU-offloaded"
        );
    }
    info!(
        device = device_name,
        compute_plane,
        routed_expert_plane = execution_context.plan().routed_experts().as_str(),
        context_id = %execution_context.id(),
        vram_capacity_mb = gpu_cache_mb,
        "GPU execution context installed for run benchmark"
    );
    Ok(execution_context)
}

/// **Tier 2.** Attach a packed expert blob to `storage` when both the blob
/// and its manifest are configured, after validating the manifest's slot
/// size against the engine's `expert_size`. Returns the storage unchanged
/// when no packed layout is configured (the default). Shared by the
/// `serve` and `run` engine-build paths.
fn maybe_attach_packed_blob(
    storage: NvmeStorage,
    packed_blob: Option<&std::path::Path>,
    packed_manifest: Option<&std::path::Path>,
    use_direct_io: bool,
    expert_size: usize,
) -> Result<NvmeStorage, Box<dyn std::error::Error>> {
    match (packed_blob, packed_manifest) {
        (Some(blob_path), Some(manifest_path)) => {
            let blob =
                crate::packed_storage::PackedBlob::open(blob_path, manifest_path, use_direct_io)?;
            blob.validate()
                .map_err(|e| format!("packed blob validation failed: {e}"))?;
            let slot = blob.manifest().expert_size;
            if slot != expert_size as u64 {
                return Err(format!(
                    "packed manifest expert_size ({slot}) != expert_size ({expert_size}); \
                     re-run `repack` with the matching --expert-size"
                )
                .into());
            }
            info!(
                experts = blob.len(),
                blob = %blob_path.display(),
                "Tier 2: packed expert blob attached (single-fd reads + coalesced preadv)"
            );
            Ok(storage.with_packed_blob(Arc::new(blob)))
        }
        (Some(_), None) | (None, Some(_)) => Err(
            "both packed_blob and packed_manifest must be set to enable the packed layout".into(),
        ),
        (None, None) => Ok(storage),
    }
}

struct BenchRealArgs {
    config: PathBuf,
    prompt: Option<String>,
    request_json: Option<PathBuf>,
    output_tokens: Option<usize>,
    warmup_runs: usize,
    measured_runs: usize,
    cache_reset: BenchRealCacheReset,
    greedy: bool,
    format: BenchRealOutputFormat,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

struct QualifyHybridQ4Args {
    config: PathBuf,
    prompt: Option<String>,
    request_json: Option<PathBuf>,
    output_tokens: Option<usize>,
    warmup_runs: usize,
    report_out: Option<PathBuf>,
    external_gpu_memory_artifact: Option<String>,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

struct QualifyHybridQ4ParityArgs {
    config: PathBuf,
    expert_id: u32,
    expected_adapter_name: String,
    report_out: Option<PathBuf>,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

struct QualifyHybridQ4GreedyParityArgs {
    config: PathBuf,
    parsed_config: crate::config::Config,
    expected_adapter_name: String,
    report_out: Option<PathBuf>,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

struct QualifyGpuNativeQ4GreedyParityArgs {
    config: PathBuf,
    parsed_config: crate::config::Config,
    expected_adapter_name: String,
    report_out: Option<PathBuf>,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

struct GreedyParityHybridWorkerArgs {
    parsed_config: crate::config::Config,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

struct DiagnoseHybridQ4GreedyDivergenceArgs {
    config: PathBuf,
    parsed_config: crate::config::Config,
    expected_adapter_name: String,
    report_out: PathBuf,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

struct DiagnoseGpuNativeQ4FirstDivergenceArgs {
    parsed_config: crate::config::Config,
    expected_adapter_name: String,
    case: String,
    report_out: Option<PathBuf>,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

struct DiagnoseGpuNativeRouterRankDivergenceArgs {
    config: PathBuf,
    parsed_config: crate::config::Config,
    expected_adapter_name: String,
    case: String,
    generated_position: usize,
    layer: usize,
    report_out: PathBuf,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

struct DiagnoseGpuNativeExpertPermutationSemanticParityArgs {
    config: PathBuf,
    parsed_config: crate::config::Config,
    expected_adapter_name: String,
    case: String,
    generated_position: usize,
    layer: usize,
    report_out: PathBuf,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

struct DiagnoseGpuNativeLayer0AttentionFirstDivergenceArgs {
    parsed_config: crate::config::Config,
    expected_adapter_name: String,
    case: String,
    report_out: Option<PathBuf>,
    slice11b_report: Option<PathBuf>,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}


struct MatvecMicrobenchArgs {
    backends: Vec<crate::parallel::DenseMatvecBackend>,
    warmup_runs: usize,
    measured_runs: usize,
    skip_lm_head: bool,
    json: bool,
}

#[cfg_attr(not(feature = "alloc-count"), allow(dead_code))]
struct ScratchAllocMicrobenchArgs {
    d_model: usize,
    num_experts: usize,
    top_k: usize,
    warmup_tokens: usize,
    measured_tokens: usize,
    json: bool,
}

#[derive(Serialize)]
struct MatvecMicrobenchReport {
    benchmark: &'static str,
    model: &'static str,
    d_model: usize,
    d_ff: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    warmup_runs: usize,
    measured_runs: usize,
    build: BenchRealBuildInfo,
    results: Vec<MatvecMicrobenchResult>,
}

#[derive(Serialize)]
struct MatvecMicrobenchResult {
    shape: &'static str,
    backend: String,
    rows: usize,
    cols: usize,
    multiply_accumulates: usize,
    best_ms: f64,
    mean_ms: f64,
    checksum: u64,
}

#[cfg(feature = "alloc-count")]
#[derive(Clone, Copy, Debug, Default, Serialize)]
struct AllocationSnapshot {
    allocation_calls: u64,
    deallocation_calls: u64,
    reallocation_calls: u64,
    bytes_allocated: u64,
    bytes_deallocated: u64,
    current_bytes: u64,
    peak_bytes: u64,
}

#[cfg(feature = "alloc-count")]
#[derive(Serialize)]
struct ScratchAllocMicrobenchReport {
    benchmark: &'static str,
    model: &'static str,
    d_model: usize,
    num_experts: usize,
    top_k: usize,
    warmup_tokens: usize,
    measured_tokens: usize,
    build: BenchRealBuildInfo,
    results: Vec<ScratchAllocMicrobenchResult>,
}

#[cfg(feature = "alloc-count")]
#[derive(Serialize)]
struct ScratchAllocMicrobenchResult {
    variant: &'static str,
    elapsed_ms: f64,
    allocations: AllocationSnapshot,
    allocation_calls_per_token: f64,
    bytes_allocated_per_token: f64,
    checksum: u64,
}

#[cfg(feature = "alloc-count")]
#[derive(Clone, Copy)]
enum ScratchAllocVariant {
    CompatibilityWrappers,
    ScratchBuffers,
}

#[cfg(feature = "alloc-count")]
impl ScratchAllocVariant {
    fn label(self) -> &'static str {
        match self {
            Self::CompatibilityWrappers => "compatibility-wrappers",
            Self::ScratchBuffers => "scratch-buffers",
        }
    }
}

#[derive(Clone, Copy)]
struct MatvecShape {
    name: &'static str,
    rows: usize,
    cols: usize,
}

struct BenchRealInput {
    prompt: String,
    output_tokens: usize,
}

struct BenchRealRuntime {
    cfg: crate::config::Config,
    engine: Arc<Engine>,
    model: Arc<crate::model::RealModel>,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    gpu_native_token_loop: Option<Arc<crate::gpu_native_token_loop::GpuNativeTokenLoop>>,

    isolated_cache: Option<Arc<MultiLayerExpertCache>>,
    isolated_shutdown: Option<IsolatedRuntimeShutdownWitness>,
}

/// Type-specific weak references to every mutable resource family that an
/// isolated qualification runtime can hand to background work. Controlled
/// shutdown consumes the runtime, closes its producer channels, and waits for
/// these references to become un-upgradeable before the next plane is built.
struct IsolatedRuntimeShutdownWitness {
    engine: std::sync::Weak<Engine>,
    model: std::sync::Weak<crate::model::RealModel>,
    cache: std::sync::Weak<MultiLayerExpertCache>,
    storage: std::sync::Weak<NvmeStorage>,
    predictor: std::sync::Weak<PredictiveLoader>,
    execution_context: std::sync::Weak<crate::backend::ExecutionContext>,
    gpu_cache: std::sync::Weak<crate::expert_cache::GpuExpertCache>,
    speculator: Option<std::sync::Weak<NeuralSpeculator>>,
    affinity: Option<std::sync::Weak<LayeredExpertAffinity>>,
}

/// Point-in-time strong-owner counts for every isolated mutable resource
/// family. `Weak::strong_count` observes ownership without upgrading the weak
/// reference, so polling this diagnostic never keeps a resource alive.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct IsolatedRuntimeLiveResources {
    engine: usize,
    model: usize,
    cache: usize,
    storage: usize,
    predictor: usize,
    execution_context: usize,
    gpu_cache: usize,
    speculator: Option<usize>,
    affinity: Option<usize>,
}

#[derive(Debug)]
struct IsolatedRuntimeShutdownError {
    detail: String,
}

impl IsolatedRuntimeShutdownError {
    fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for IsolatedRuntimeShutdownError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.detail)
    }
}

impl std::error::Error for IsolatedRuntimeShutdownError {}

impl IsolatedRuntimeLiveResources {
    fn all_released(self) -> bool {
        self.engine == 0
            && self.model == 0
            && self.cache == 0
            && self.storage == 0
            && self.predictor == 0
            && self.execution_context == 0
            && self.gpu_cache == 0
            && self.speculator.is_none_or(|count| count == 0)
            && self.affinity.is_none_or(|count| count == 0)
    }
}

impl std::fmt::Display for IsolatedRuntimeLiveResources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn optional_count(value: Option<usize>) -> String {
            value.map_or_else(|| "disabled".to_string(), |count| count.to_string())
        }
        write!(
            f,
            "engine={} model={} cache={} storage={} predictor={} execution_context={} gpu_cache={} speculator={} affinity={}",
            self.engine,
            self.model,
            self.cache,
            self.storage,
            self.predictor,
            self.execution_context,
            self.gpu_cache,
            optional_count(self.speculator),
            optional_count(self.affinity),
        )
    }
}

impl IsolatedRuntimeShutdownWitness {
    fn live_resources(&self) -> IsolatedRuntimeLiveResources {
        IsolatedRuntimeLiveResources {
            engine: self.engine.strong_count(),
            model: self.model.strong_count(),
            cache: self.cache.strong_count(),
            storage: self.storage.strong_count(),
            predictor: self.predictor.strong_count(),
            execution_context: self.execution_context.strong_count(),
            gpu_cache: self.gpu_cache.strong_count(),
            speculator: self.speculator.as_ref().map(std::sync::Weak::strong_count),
            affinity: self.affinity.as_ref().map(std::sync::Weak::strong_count),
        }
    }

    async fn wait_for_release(
        self,
    ) -> Result<crate::greedy_parity::BackgroundShutdownEvidence, String> {
        const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(10);
        const SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
        wait_for_isolated_release(|| self.live_resources(), POLL_INTERVAL, SHUTDOWN_TIMEOUT).await
    }
}

async fn wait_for_isolated_release<F>(
    mut live_resources: F,
    poll_interval: std::time::Duration,
    shutdown_timeout: std::time::Duration,
) -> Result<crate::greedy_parity::BackgroundShutdownEvidence, String>
where
    F: FnMut() -> IsolatedRuntimeLiveResources,
{
    let started = Instant::now();
    let mut poll_iterations = 0u32;
    loop {
        poll_iterations = poll_iterations.saturating_add(1);
        let live = live_resources();
        if live.all_released() {
            return Ok(crate::greedy_parity::BackgroundShutdownEvidence {
                controlled_shutdown_requested: true,
                all_runtime_resources_released: true,
                poll_iterations,
            });
        }
        if started.elapsed() >= shutdown_timeout {
            return Err(format!(
                "isolated runtime background resources remained live after {}s: {live}",
                shutdown_timeout.as_secs_f64(),
            ));
        }
        tokio::time::sleep(poll_interval).await;
    }
}

impl BenchRealRuntime {
    async fn shutdown_isolated(
        mut self,
    ) -> Result<crate::greedy_parity::BackgroundShutdownEvidence, IsolatedRuntimeShutdownError>
    {
        let witness = self.isolated_shutdown.take().ok_or_else(|| {
            IsolatedRuntimeShutdownError::new(
                "runtime was not constructed by the isolated qualification factory",
            )
        })?;
        // Stop producers and cancel/drain every engine-owned async background
        // task before dropping the runtime's strong references. Affinity's
        // owned thread handle is joined in Engine drop; the neural-speculator
        // worker owns only a Weak and exits when its producer disappears. The
        // witness remains an independent postcondition proving every mutable
        // resource family is gone before another isolated runtime is built.
        let controlled_shutdown = self.engine.shutdown_background_tasks().await;
        drop(self);
        let released = witness.wait_for_release().await;
        match (controlled_shutdown, released) {
            (Ok(()), Ok(evidence)) => Ok(evidence),
            (Err(error), Ok(_)) => Err(IsolatedRuntimeShutdownError::new(error)),
            (Ok(()), Err(error)) => Err(IsolatedRuntimeShutdownError::new(error)),
            (Err(controlled_error), Err(release_error)) => {
                Err(IsolatedRuntimeShutdownError::new(format!(
                    "{controlled_error}; weak release postcondition also failed: {release_error}"
                )))
            }
        }
    }
}

#[derive(Serialize)]
struct BenchRealSuiteReport {
    benchmark: &'static str,
    config: String,
    prompt: String,
    warmup_runs: usize,
    measured_runs: usize,
    cache_reset: BenchRealCacheReset,
    greedy: bool,
    build: BenchRealBuildInfo,
    aggregate: BenchRealAggregate,
    runs: Vec<BenchRealRunReport>,
}

#[derive(Serialize)]
struct BenchRealBuildInfo {
    git_commit: String,
    build_features: Vec<&'static str>,
    threads: usize,
    dense_matvec_backend: String,
}

#[derive(Clone, Serialize)]
struct BenchRealRunReport {
    run_index: usize,
    prompt_tokens: usize,
    completion_tokens: usize,
    total_api_tokens: usize,
    model_forward_evaluations: usize,
    lm_head_evaluations: usize,
    prompt_seconds: f64,
    prompt_tps: f64,
    decode_seconds: f64,
    decode_tps: f64,
    time_to_first_token_seconds: f64,
    total_seconds: f64,
    decode_token_latency_p50_ms: f64,
    decode_token_latency_p95_ms: f64,
    decode_token_latency_p99_ms: f64,
    decode_token_latency_max_ms: f64,
    cache_hits: u64,
    cache_misses: u64,
    hit_rate: f64,
    ssd_bytes: u64,
    ssd_stall_seconds: f64,
    rss_bytes: Option<u64>,
    output_token_ids: Vec<u32>,
    output_text: String,
    stage_timings: std::collections::BTreeMap<String, crate::stage_timing::StageTimingSnapshot>,
}

#[derive(Serialize)]
struct BenchRealAggregate {
    prompt_seconds_mean: f64,
    prompt_tps_mean: f64,
    decode_seconds_mean: f64,
    decode_tps_mean: f64,
    time_to_first_token_p50_seconds: f64,
    total_seconds_mean: f64,
    cache_hits_total: u64,
    cache_misses_total: u64,
    hit_rate: f64,
    ssd_bytes_total: u64,
    output_token_parity: bool,
}

async fn cmd_bench_real(args: BenchRealArgs) -> Result<(), Box<dyn std::error::Error>> {
    let input = load_bench_real_input(&args)?;
    if args.measured_runs == 0 {
        return Err("bench-real requires --measured-runs > 0".into());
    }

    if args.cache_reset == BenchRealCacheReset::Keep {
        let runtime = build_bench_real_runtime(&args.config).await?;
        let params = bench_sampling_params(&runtime.cfg, args.greedy);
        for i in 0..args.warmup_runs {
            let _ = with_progress_timeout(
                format!("bench-real warmup run {i}"),
                args.progress_watchdog,
                run_bench_real_once(&runtime, &input.prompt, input.output_tokens, params, i),
            )
            .await?;
        }
        let softmax_before = crate::transformer::nonfinite_softmax_fallbacks();
        let mut runs = Vec::with_capacity(args.measured_runs);
        for i in 0..args.measured_runs {
            runs.push(
                with_progress_timeout(
                    format!("bench-real measured run {i}"),
                    args.progress_watchdog,
                    run_bench_real_once(&runtime, &input.prompt, input.output_tokens, params, i),
                )
                .await?,
            );
        }
        assert_no_softmax_fallbacks(softmax_before)?;
        emit_bench_real_report(&args, input, runs)?;
    } else {
        for i in 0..args.warmup_runs {
            let runtime = build_bench_real_runtime(&args.config).await?;
            let params = bench_sampling_params(&runtime.cfg, args.greedy);
            let _ = with_progress_timeout(
                format!("bench-real warmup run {i}"),
                args.progress_watchdog,
                run_bench_real_once(&runtime, &input.prompt, input.output_tokens, params, i),
            )
            .await?;
        }
        let softmax_before = crate::transformer::nonfinite_softmax_fallbacks();
        let mut runs = Vec::with_capacity(args.measured_runs);
        for i in 0..args.measured_runs {
            let runtime = build_bench_real_runtime(&args.config).await?;
            let params = bench_sampling_params(&runtime.cfg, args.greedy);
            runs.push(
                with_progress_timeout(
                    format!("bench-real measured run {i}"),
                    args.progress_watchdog,
                    run_bench_real_once(&runtime, &input.prompt, input.output_tokens, params, i),
                )
                .await?,
            );
        }
        assert_no_softmax_fallbacks(softmax_before)?;
        emit_bench_real_report(&args, input, runs)?;
    }
    Ok(())
}

struct RealCliRequestInput {
    prompt: String,
    output_tokens: usize,
    input_kind: &'static str,
}

fn load_real_cli_request_input(
    command: &'static str,
    prompt_arg: Option<&String>,
    request_json: Option<&Path>,
    output_tokens_arg: Option<usize>,
) -> Result<RealCliRequestInput, Box<dyn std::error::Error>> {
    let mut json_max_tokens = None;
    let (prompt, input_kind) = if let Some(prompt) = prompt_arg {
        (prompt.clone(), "prompt")
    } else if let Some(path) = request_json {
        let body = std::fs::read_to_string(path)?;
        let value: serde_json::Value = serde_json::from_str(&body)?;
        json_max_tokens = value
            .get("max_tokens")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);
        let prompt = if let Some(prompt) = value.get("prompt").and_then(|value| value.as_str()) {
            prompt.to_string()
        } else if let Some(messages) = value.get("messages").and_then(|value| value.as_array()) {
            flatten_bench_messages(messages)
        } else {
            return Err(
                "--request-json must contain a string `prompt` or chat `messages` array".into(),
            );
        };
        (prompt, "request-json")
    } else {
        return Err(format!("{command} requires either --prompt or --request-json").into());
    };
    if prompt.is_empty() {
        return Err(format!("{command} prompt must be non-empty").into());
    }
    let output_tokens = output_tokens_arg.or(json_max_tokens).unwrap_or(16);
    if output_tokens == 0 {
        return Err(format!("{command} requires output token count > 0").into());
    }
    Ok(RealCliRequestInput {
        prompt,
        output_tokens,
        input_kind,
    })
}

fn load_qualification_input(
    args: &QualifyHybridQ4Args,
) -> Result<RealCliRequestInput, Box<dyn std::error::Error>> {
    load_real_cli_request_input(
        "qualify-hybrid-q4",
        args.prompt.as_ref(),
        args.request_json.as_deref(),
        args.output_tokens,
    )
}

fn qualification_artifacts(
    config_path: &Path,
    cfg: &crate::config::Config,
) -> (crate::qualification::QualificationArtifacts, Vec<String>) {
    use crate::qualification::{hash_optional_small_file, hash_small_file};

    let mut errors = Vec::new();
    let config_digest = match hash_small_file(config_path) {
        Ok(digest) => Some(digest),
        Err(error) => {
            errors.push(format!(
                "failed to hash qualification config {}: {error}",
                config_path.display()
            ));
            None
        }
    };
    let mut hash_optional = |path: Option<&Path>, label: &str| match hash_optional_small_file(path)
    {
        Ok(digest) => digest,
        Err(error) => {
            let display = path
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<none>".to_string());
            errors.push(format!(
                "failed to hash optional {label} {display}: {error}"
            ));
            None
        }
    };

    let metadata_path = cfg.model.data_dir.join("metadata.json");
    let weights_config = cfg
        .real_transformer
        .weights_dir
        .as_ref()
        .map(|dir| dir.join("config.json"));
    let artifacts = crate::qualification::QualificationArtifacts {
        config: config_digest,
        tokenizer: hash_optional(cfg.tokenizer.path.as_deref(), "tokenizer"),
        expert_metadata: hash_optional(Some(&metadata_path), "expert metadata"),
        packed_manifest: hash_optional(
            cfg.storage.packed_manifest.as_deref(),
            "packed expert manifest",
        ),
        weights_config: hash_optional(weights_config.as_deref(), "weights config"),
        dense_weights_directory: cfg
            .real_transformer
            .weights_dir
            .as_deref()
            .map(crate::qualification::canonical_or_configured),
        expert_data_directory: crate::qualification::canonical_or_configured(&cfg.model.data_dir),
        packed_expert_blob: cfg
            .storage
            .packed_blob
            .as_deref()
            .map(crate::qualification::canonical_or_configured),
        large_artifacts_recursively_hashed: false,
    };
    (artifacts, errors)
}

fn emit_qualification_report(
    report: &crate::qualification::QualificationReport,
    report_out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    if let Some(path) = report_out {
        std::fs::write(path, json)?;
        eprintln!("qualification report written to {}", path.display());
    } else {
        use std::io::Write as _;
        std::io::stdout().write_all(&json)?;
    }
    Ok(())
}

fn fail_qualification(
    mut report: crate::qualification::QualificationReport,
    failure: crate::qualification::QualificationFailure,
    report_out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let summary = format!("{}: {}", failure.code, failure.detail);
    report.fail(failure);
    emit_qualification_report(&report, report_out)?;
    Err(summary.into())
}

fn emit_q4_parity_report(
    report: &crate::q4_parity::Q4ParityReport,
    report_out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    if let Some(path) = report_out {
        std::fs::write(path, json)?;
        eprintln!("Q4_0 parity report written to {}", path.display());
    } else {
        use std::io::Write as _;
        std::io::stdout().write_all(&json)?;
    }
    Ok(())
}

fn fail_q4_parity(
    mut report: crate::q4_parity::Q4ParityReport,
    failure: crate::qualification::QualificationFailure,
    report_out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let summary = format!("{}: {}", failure.code, failure.detail);
    report.fail(failure);
    match emit_q4_parity_report(&report, report_out) {
        Ok(()) => Err(summary.into()),
        Err(emit_error) => Err(format!(
            "{summary}; additionally failed to emit Q4_0 parity report: {emit_error}"
        )
        .into()),
    }
}

fn q4_parity_readback_timeout(
    args: &QualifyHybridQ4ParityArgs,
) -> Result<std::time::Duration, crate::qualification::QualificationFailure> {
    args.progress_watchdog.timeout.ok_or_else(|| {
        crate::qualification::QualificationFailure::new(
            crate::qualification::FailureStage::Preflight,
            "progress-watchdog-required",
            "strict Q4_0 parity requires a positive performance.progress_timeout_secs or --progress-timeout-secs so raw GPU readback is bounded",
        )
    })
}

fn qualification_artifact_failure(
    errors: &[String],
) -> crate::qualification::QualificationFailure {
    crate::qualification::QualificationFailure::new(
        crate::qualification::FailureStage::Preflight,
        "artifact-identity-unavailable",
        errors.join("; "),
    )
}

fn qualification_metadata_failure(
    detail: impl Into<String>,
) -> crate::qualification::QualificationFailure {
    crate::qualification::QualificationFailure::new(
        crate::qualification::FailureStage::Preflight,
        "expert-metadata-unreadable",
        detail,
    )
}

fn qualification_inference_failure(
    error: &(dyn std::error::Error + 'static),
) -> crate::qualification::QualificationFailure {
    use crate::engine::MoeStepError;
    use crate::model::RealInferenceError;
    use crate::qualification::{FailureStage, GpuFailureEvidence, QualificationFailure};

    if let Some(RealInferenceError::MoeStep(MoeStepError::GpuExpertDispatch { source })) =
        error.downcast_ref::<RealInferenceError>()
    {
        let mut failure = QualificationFailure::new(
            FailureStage::Inference,
            "routed-gpu-dispatch-failed",
            source.to_string(),
        );
        failure.gpu_dispatch = Some(GpuFailureEvidence {
            layer: source.layer,
            expert_id: source.expert_id,
            kind: source.kind.to_string(),
        });
        failure
    } else {
        QualificationFailure::new(
            FailureStage::Inference,
            "inference-failed",
            error.to_string(),
        )
    }
}

async fn cmd_qualify_hybrid_q4(
    args: QualifyHybridQ4Args,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::{
        gpu_io_delta, request_evidence, routed_execution_delta, validate_device,
        validate_execution_plan, validate_gpu_failure_policy, validate_gpu_io,
        validate_memory, validate_postconditions, validate_preflight, BuildProvenance,
        FailureStage, PreflightEvidence, QualificationFailure, QualificationReport,
        QualificationTiming,
    };

    // Config/request parse errors retain normal CLI error behavior. Once both
    // are parsed, every expected qualification failure below emits the typed
    // report before returning a non-zero status.
    let cfg = crate::config::Config::from_file(&args.config)?;
    let input = load_qualification_input(&args)?;
    let provenance = BuildProvenance::embedded();
    let (artifacts, artifact_errors) = qualification_artifacts(&args.config, &cfg);
    let metadata_path = cfg.model.data_dir.join("metadata.json");
    let metadata_result = crate::qualification::read_expert_metadata(&metadata_path);
    let metadata = metadata_result.clone().ok();
    let request = request_evidence(
        input.input_kind,
        &input.prompt,
        input.output_tokens,
        args.warmup_runs,
    );
    let mut report = QualificationReport::new(
        provenance.clone(),
        artifacts,
        metadata.clone(),
        request,
        args.external_gpu_memory_artifact.clone(),
    );

    if !artifact_errors.is_empty() {
        return fail_qualification(
            report,
            qualification_artifact_failure(&artifact_errors),
            args.report_out.as_deref(),
        );
    }
    let metadata = match metadata_result {
        Ok(metadata) => metadata,
        Err(detail) => {
            return fail_qualification(
                report,
                qualification_metadata_failure(detail),
                args.report_out.as_deref(),
            );
        }
    };
    let weight_policy = cfg.real_transformer.resolve_weight_policy();
    if !matches!(
        weight_policy,
        Ok(crate::config::RealWeightPolicy::StrictReal)
    ) {
        return fail_qualification(
            report,
            QualificationFailure::new(
                FailureStage::Preflight,
                "non-strict-weight-policy",
                weight_policy
                    .err()
                    .unwrap_or_else(|| "resolved weight policy is SeededDev".to_string()),
            ),
            args.report_out.as_deref(),
        );
    }
    let capacity_bytes = cfg
        .gpu_cache
        .vram_capacity_mb
        .checked_mul(1024 * 1024)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .unwrap_or(0);
    let preflight = PreflightEvidence {
        provenance,
        real_transformer_enabled: cfg.real_transformer.enabled,
        weights_dir_configured: cfg.real_transformer.weights_dir.is_some(),
        strict_weights: cfg.real_transformer.strict_weights,
        allow_seeded_fallback: cfg.real_transformer.allow_seeded_fallback,
        allow_degraded_experts: cfg.real_transformer.allow_degraded_experts,
        allow_attention_fallback: cfg.real_transformer.allow_nonfinite_attention_fallback,
        allow_truncated_expert_payloads: cfg.real_transformer.allow_truncated_expert_payloads,
        distributed_enabled: cfg.distributed.enabled,
        gpu_cache_enabled: cfg.gpu_cache.enabled,
        gpu_expert_capacity_bytes: capacity_bytes,
        requested_mode: cfg.real_transformer.compute_offload,
        routed_expert_dtype: cfg.model.dtype,
        metadata,
    };
    if let Err(failure) = validate_preflight(&preflight, &mut report.qualification_checks) {
        return fail_qualification(report, failure, args.report_out.as_deref());
    }

    let runtime =
        match build_real_cli_runtime(&args.config, RealCliRuntimeMode::StrictHybridQualification)
            .await
        {
            Ok(runtime) => runtime,
            Err(error) => {
                return fail_qualification(
                    report,
                    QualificationFailure::new(
                        FailureStage::Startup,
                        "startup-failed",
                        error.to_string(),
                    ),
                    args.report_out.as_deref(),
                );
            }
        };
    if !runtime.model.load_status.strict
        || runtime.model.load_status.seeded_fallback_remained
        || runtime.model.load_status.loaded_tensors != runtime.model.load_status.required_tensors
    {
        return fail_qualification(
            report,
            QualificationFailure::new(
                FailureStage::Startup,
                "seeded-model-loaded",
                format!(
                    "strict={} loaded={}/{} seeded_fallback_remained={}",
                    runtime.model.load_status.strict,
                    runtime.model.load_status.loaded_tensors,
                    runtime.model.load_status.required_tensors,
                    runtime.model.load_status.seeded_fallback_remained
                ),
            ),
            args.report_out.as_deref(),
        );
    }

    let context = runtime.engine.execution_context();
    report.execution_plan = Some(context.plan().into());
    report.device = runtime.engine.gpu_device_identity();
    if let Err(failure) = validate_execution_plan(context.plan(), &mut report.qualification_checks)
    {
        return fail_qualification(report, failure, args.report_out.as_deref());
    }
    if let Err(failure) = validate_device(report.device.as_ref(), &mut report.qualification_checks)
    {
        return fail_qualification(report, failure, args.report_out.as_deref());
    }
    if let Err(failure) = validate_gpu_failure_policy(
        runtime.engine.routed_expert_gpu_failure_policy(),
        &mut report.qualification_checks,
    ) {
        return fail_qualification(report, failure, args.report_out.as_deref());
    }

    let greedy = crate::sampling::SamplingParams::greedy();
    for index in 0..args.warmup_runs {
        if let Err(error) = with_progress_timeout(
            format!("qualify-hybrid-q4 warmup run {index}"),
            args.progress_watchdog,
            run_bench_real_once(&runtime, &input.prompt, input.output_tokens, greedy, index),
        )
        .await
        {
            return fail_qualification(
                report,
                qualification_inference_failure(error.as_ref()),
                args.report_out.as_deref(),
            );
        }
    }

    let routed_before = runtime.engine.routed_expert_execution_snapshot();
    let io_before = match runtime.engine.gpu_expert_io_snapshot() {
        Some(snapshot) => snapshot,
        None => {
            return fail_qualification(
                report,
                QualificationFailure::new(
                    FailureStage::Startup,
                    "gpu-device-unavailable",
                    "authoritative GPU backend has no routed-expert I/O snapshot",
                ),
                args.report_out.as_deref(),
            );
        }
    };
    let memory_before = match runtime.engine.gpu_expert_memory_snapshot() {
        Some(snapshot) => snapshot,
        None => {
            return fail_qualification(
                report,
                QualificationFailure::new(
                    FailureStage::Startup,
                    "gpu-device-unavailable",
                    "authoritative GPU backend has no PR4 memory snapshot",
                ),
                args.report_out.as_deref(),
            );
        }
    };
    report.gpu_memory_before = Some(memory_before);

    let measured = with_progress_timeout(
        "qualify-hybrid-q4 measured request".to_string(),
        args.progress_watchdog,
        run_bench_real_once(&runtime, &input.prompt, input.output_tokens, greedy, 0),
    )
    .await;

    let routed_after = runtime.engine.routed_expert_execution_snapshot();
    let io_after = runtime.engine.gpu_expert_io_snapshot();
    let memory_after = runtime.engine.gpu_expert_memory_snapshot();
    report.gpu_memory_after = memory_after;
    report.routed_experts = routed_execution_delta(routed_before, routed_after).ok();
    report.gpu_io = io_after.and_then(|after| gpu_io_delta(io_before, after).ok());

    let measured = match measured {
        Ok(measured) => measured,
        Err(error) => {
            return fail_qualification(
                report,
                qualification_inference_failure(error.as_ref()),
                args.report_out.as_deref(),
            );
        }
    };
    let routed_delta = match routed_execution_delta(routed_before, routed_after) {
        Ok(delta) => delta,
        Err(failure) => {
            return fail_qualification(report, failure, args.report_out.as_deref());
        }
    };
    let io_after = match io_after {
        Some(snapshot) => snapshot,
        None => {
            return fail_qualification(
                report,
                QualificationFailure::new(
                    FailureStage::Postcondition,
                    "gpu-device-unavailable",
                    "authoritative GPU I/O snapshot disappeared during measurement",
                ),
                args.report_out.as_deref(),
            );
        }
    };
    let memory_after = match memory_after {
        Some(snapshot) => snapshot,
        None => {
            return fail_qualification(
                report,
                QualificationFailure::new(
                    FailureStage::Postcondition,
                    "gpu-device-unavailable",
                    "authoritative PR4 memory snapshot disappeared during measurement",
                ),
                args.report_out.as_deref(),
            );
        }
    };
    let io_delta = match gpu_io_delta(io_before, io_after) {
        Ok(delta) => delta,
        Err(failure) => {
            return fail_qualification(report, failure, args.report_out.as_deref());
        }
    };
    report.routed_experts = Some(routed_delta);
    report.gpu_io = Some(io_delta);
    if let Err(failure) = validate_gpu_io(io_delta, &mut report.qualification_checks) {
        return fail_qualification(report, failure, args.report_out.as_deref());
    }
    for snapshot in [memory_before, memory_after] {
        if let Err(failure) = validate_memory(snapshot) {
            return fail_qualification(report, failure, args.report_out.as_deref());
        }
    }
    report.timing = Some(QualificationTiming::from_measurement(
        measured.prompt_tokens,
        input.output_tokens,
        measured.completion_tokens,
        measured.prompt_seconds,
        measured.decode_seconds,
        measured.total_seconds,
    ));
    if let Err(failure) = validate_postconditions(
        measured.completion_tokens,
        routed_delta,
        &mut report.qualification_checks,
    ) {
        return fail_qualification(report, failure, args.report_out.as_deref());
    }
    if let Err(failure) = report.finish() {
        return fail_qualification(report, failure, args.report_out.as_deref());
    }
    emit_qualification_report(&report, args.report_out.as_deref())
}

async fn cmd_qualify_hybrid_q4_parity(
    args: QualifyHybridQ4ParityArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::{
        validate_device, validate_execution_plan, validate_gpu_failure_policy,
        validate_memory, validate_preflight, BuildProvenance, FailureStage, PreflightEvidence,
        QualificationChecks, QualificationFailure,
    };

    let cfg = crate::config::Config::from_file(&args.config)?;
    let provenance = BuildProvenance::embedded();
    let (artifacts, artifact_errors) = qualification_artifacts(&args.config, &cfg);
    let metadata_path = cfg.model.data_dir.join("metadata.json");
    let metadata_result = crate::qualification::read_expert_metadata(&metadata_path);
    let metadata = metadata_result.clone().ok();
    let mut report = crate::q4_parity::Q4ParityReport::new(
        provenance.clone(),
        artifacts,
        metadata.clone(),
        args.expected_adapter_name.clone(),
    );
    let raw_readback_timeout = match q4_parity_readback_timeout(&args) {
        Ok(timeout) => timeout,
        Err(failure) => {
            return fail_q4_parity(report, failure, args.report_out.as_deref());
        }
    };

    if !artifact_errors.is_empty() {
        return fail_q4_parity(
            report,
            qualification_artifact_failure(&artifact_errors),
            args.report_out.as_deref(),
        );
    }
    let metadata = match metadata_result {
        Ok(metadata) => metadata,
        Err(detail) => {
            return fail_q4_parity(
                report,
                qualification_metadata_failure(detail),
                args.report_out.as_deref(),
            );
        }
    };
    let weight_policy = cfg.real_transformer.resolve_weight_policy();
    if !matches!(weight_policy, Ok(crate::config::RealWeightPolicy::StrictReal)) {
        return fail_q4_parity(
            report,
            QualificationFailure::new(
                FailureStage::Preflight,
                "non-strict-weight-policy",
                weight_policy
                    .err()
                    .unwrap_or_else(|| "resolved weight policy is SeededDev".to_string()),
            ),
            args.report_out.as_deref(),
        );
    }
    let capacity_bytes = cfg
        .gpu_cache
        .vram_capacity_mb
        .checked_mul(1024 * 1024)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .unwrap_or(0);
    let preflight = PreflightEvidence {
        provenance,
        real_transformer_enabled: cfg.real_transformer.enabled,
        weights_dir_configured: cfg.real_transformer.weights_dir.is_some(),
        strict_weights: cfg.real_transformer.strict_weights,
        allow_seeded_fallback: cfg.real_transformer.allow_seeded_fallback,
        allow_degraded_experts: cfg.real_transformer.allow_degraded_experts,
        allow_attention_fallback: cfg.real_transformer.allow_nonfinite_attention_fallback,
        allow_truncated_expert_payloads: cfg.real_transformer.allow_truncated_expert_payloads,
        distributed_enabled: cfg.distributed.enabled,
        gpu_cache_enabled: cfg.gpu_cache.enabled,
        gpu_expert_capacity_bytes: capacity_bytes,
        requested_mode: cfg.real_transformer.compute_offload,
        routed_expert_dtype: cfg.model.dtype,
        metadata,
    };
    let mut base_checks = QualificationChecks::default();
    if let Err(failure) = validate_preflight(&preflight, &mut base_checks) {
        return fail_q4_parity(report, failure, args.report_out.as_deref());
    }
    report.checks.clean_build = base_checks.clean_build;
    report.checks.strict_hybrid_preflight = base_checks.strict_real_checkpoint
        && base_checks.requested_hybrid
        && base_checks.native_q4_0_routed_experts;
    report.checks.canonical_q4_0_layout = base_checks.canonical_q4_layout;

    let runtime = match build_real_cli_runtime(
        &args.config,
        RealCliRuntimeMode::StrictHybridQualification,
    )
    .await
    {
        Ok(runtime) => runtime,
        Err(error) => {
            return fail_q4_parity(
                report,
                QualificationFailure::new(
                    FailureStage::Startup,
                    "startup-failed",
                    error.to_string(),
                ),
                args.report_out.as_deref(),
            );
        }
    };
    if !runtime.model.load_status.strict
        || runtime.model.load_status.seeded_fallback_remained
        || runtime.model.load_status.loaded_tensors != runtime.model.load_status.required_tensors
    {
        return fail_q4_parity(
            report,
            QualificationFailure::new(
                FailureStage::Startup,
                "seeded-model-loaded",
                format!(
                    "strict={} loaded={}/{} seeded_fallback_remained={}",
                    runtime.model.load_status.strict,
                    runtime.model.load_status.loaded_tensors,
                    runtime.model.load_status.required_tensors,
                    runtime.model.load_status.seeded_fallback_remained
                ),
            ),
            args.report_out.as_deref(),
        );
    }

    let context = runtime.engine.execution_context();
    report.execution_plan = Some(context.plan().into());
    if let Err(failure) = validate_execution_plan(context.plan(), &mut base_checks) {
        return fail_q4_parity(report, failure, args.report_out.as_deref());
    }
    report.checks.exact_execution_plan = base_checks.resolved_planes_match_contract;

    let device = runtime.engine.gpu_device_identity();
    report.device = device.clone();
    if let Err(failure) = validate_device(device.as_ref(), &mut base_checks) {
        return fail_q4_parity(report, failure, args.report_out.as_deref());
    }
    report.checks.hardware_gpu_adapter = base_checks.hardware_gpu_adapter;
    let Some(device) = device else {
        return fail_q4_parity(
            report,
            QualificationFailure::new(
                FailureStage::Startup,
                "gpu-device-unavailable",
                "validated authoritative GPU identity unexpectedly disappeared",
            ),
            args.report_out.as_deref(),
        );
    };
    if device.name != args.expected_adapter_name {
        return fail_q4_parity(
            report,
            QualificationFailure::new(
                FailureStage::Startup,
                "unexpected-gpu-adapter",
                format!(
                    "selected adapter {:?}, expected exact name {:?}",
                    device.name, args.expected_adapter_name
                ),
            ),
            args.report_out.as_deref(),
        );
    }
    report.checks.expected_adapter_exact_match = true;
    if let Err(failure) = validate_gpu_failure_policy(
        runtime.engine.routed_expert_gpu_failure_policy(),
        &mut base_checks,
    ) {
        return fail_q4_parity(report, failure, args.report_out.as_deref());
    }
    report.checks.strict_gpu_failure_policy = base_checks.strict_gpu_failure_policy;

    let identity = match crate::q4_parity::expert_identity(
        args.expert_id,
        runtime.cfg.model.num_layers,
        runtime.cfg.model.num_experts,
    ) {
        Ok(identity) => identity,
        Err(failure) => {
            return fail_q4_parity(report, failure, args.report_out.as_deref());
        }
    };
    report.checks.global_expert_identity_valid = true;

    // Raw canonical blocks use only ephemeral qualification buffers. Prove
    // they leave logical/physical expert state and production I/O counters
    // unchanged before beginning the complete-expert evidence window.
    let backend = context.routed_expert_backend();
    let raw_memory_before = runtime.engine.gpu_expert_memory_snapshot();
    let raw_io_before = runtime.engine.gpu_expert_io_snapshot();
    let raw_physical_before = backend.gpu_physical_expert_residency(args.expert_id);
    let Some(raw_memory_before) = raw_memory_before else {
        return fail_q4_parity(
            report,
            QualificationFailure::new(
                FailureStage::Startup,
                "gpu-device-unavailable",
                "raw Q4_0 qualification has no physical-memory snapshot",
            ),
            args.report_out.as_deref(),
        );
    };
    let Some(raw_io_before) = raw_io_before else {
        return fail_q4_parity(
            report,
            QualificationFailure::new(
                FailureStage::Startup,
                "gpu-device-unavailable",
                "raw Q4_0 qualification has no routed-expert I/O snapshot",
            ),
            args.report_out.as_deref(),
        );
    };
    if let Err(failure) = validate_memory(raw_memory_before) {
        return fail_q4_parity(report, failure, args.report_out.as_deref());
    }
    if raw_memory_before.logical_admitted_bytes != 0
        || raw_memory_before.expert_live_bytes != 0
        || raw_memory_before.expert_registry_bytes != 0
        || raw_memory_before.physical_entries != 0
        || raw_memory_before.physical_installs != 0
        || raw_memory_before.physical_evictions != 0
        || raw_memory_before.stale_retirements != 0
        || raw_io_before != crate::backend::GpuExpertIoSnapshot::default()
        || raw_physical_before.is_some()
    {
        return fail_q4_parity(
            report,
            QualificationFailure::new(
                FailureStage::Startup,
                "physical-registry-not-cold",
                "complete-expert parity requires a fresh logical/physical expert registry and zero routed GPU-I/O counters",
            ),
            args.report_out.as_deref(),
        );
    }
    let raw_validation = match crate::q4_parity::run_raw_shader_cases(
        backend,
        raw_readback_timeout,
    ) {
        Ok(validation) => validation,
        Err(failure) => {
            return fail_q4_parity(report, failure, args.report_out.as_deref());
        }
    };
    report.checks.raw_shader_cases_passed = raw_validation.all_cases_passed();
    let mut raw_failure = raw_validation.tolerance_failure;
    report.raw_cases = raw_validation.reports;
    let raw_memory_after = runtime.engine.gpu_expert_memory_snapshot();
    let raw_io_after = runtime.engine.gpu_expert_io_snapshot();
    let raw_physical_after = backend.gpu_physical_expert_residency(args.expert_id);
    let Some(raw_memory_after) = raw_memory_after else {
        let missing = QualificationFailure::new(
            FailureStage::Postcondition,
            "gpu-device-unavailable",
            "raw Q4_0 qualification lost its physical-memory snapshot",
        );
        return fail_q4_parity(
            report,
            raw_failure.unwrap_or(missing),
            args.report_out.as_deref(),
        );
    };
    let Some(raw_io_after) = raw_io_after else {
        let missing = QualificationFailure::new(
            FailureStage::Postcondition,
            "gpu-device-unavailable",
            "raw Q4_0 qualification lost its routed-expert I/O snapshot",
        );
        return fail_q4_parity(
            report,
            raw_failure.unwrap_or(missing),
            args.report_out.as_deref(),
        );
    };
    report.raw_isolation = Some(crate::q4_parity::RawIsolationEvidence {
        memory_before: raw_memory_before,
        memory_after: raw_memory_after,
        gpu_io_before: raw_io_before,
        gpu_io_after: raw_io_after,
        selected_physical_before: raw_physical_before,
        selected_physical_after: raw_physical_after,
    });
    let memory_after_valid = match validate_memory(raw_memory_after) {
        Ok(()) => true,
        Err(failure) => {
            raw_failure.get_or_insert(failure);
            false
        }
    };
    let raw_state_unchanged = raw_memory_after == raw_memory_before
        && raw_io_after == raw_io_before
        && raw_physical_after == raw_physical_before;
    if !raw_state_unchanged {
        raw_failure.get_or_insert_with(|| QualificationFailure::new(
            FailureStage::Postcondition,
            "raw-q4-contaminated-expert-residency",
            "raw Q4_0 dispatch changed production expert residency, memory, or I/O evidence",
        ));
    }
    report.checks.raw_dispatch_isolated_from_expert_registry =
        memory_after_valid && raw_state_unchanged;
    if let Some(failure) = raw_failure {
        return fail_q4_parity(report, failure, args.report_out.as_deref());
    }

    let inputs = crate::q4_parity::deterministic_complete_inputs(runtime.cfg.model.d_model);
    let execution = match runtime
        .engine
        .qualify_q4_0_complete_expert(args.expert_id, identity.layer_index, &inputs)
        .await
    {
        Ok(execution) => execution,
        Err(detail) => {
            return fail_q4_parity(
                report,
                QualificationFailure::new(
                    FailureStage::Inference,
                    "complete-expert-dispatch-failed",
                    detail,
                ),
                args.report_out.as_deref(),
            );
        }
    };
    let expected_payload = crate::inference::expert_weight_bytes_for(
        runtime.cfg.model.d_model,
        runtime.cfg.model.d_ff,
        crate::inference::WeightDtype::Q4_0,
    );
    if execution.payload_bytes != expected_payload {
        return fail_q4_parity(
            report,
            QualificationFailure::new(
                FailureStage::Postcondition,
                "expert-payload-size-mismatch",
                format!(
                    "complete expert has {} logical bytes, expected exactly {expected_payload}",
                    execution.payload_bytes
                ),
            ),
            args.report_out.as_deref(),
        );
    }
    report.checks.exact_expert_payload_size = true;
    let complete_validation = match crate::q4_parity::validate_complete_expert(
        identity,
        execution,
        runtime.cfg.model.d_model,
    ) {
        Ok(complete) => complete,
        Err(failure) => {
            return fail_q4_parity(report, failure, args.report_out.as_deref());
        }
    };
    let crate::q4_parity::CompleteExpertValidation {
        report: complete_report,
        invariants: outcomes,
        tolerance_failure,
    } = complete_validation;
    report.complete_expert = Some(complete_report);
    report.checks.initial_physical_install_exactly_once =
        outcomes.initial_physical_install_exactly_once;
    report.checks.subsequent_dispatches_reused_generation =
        outcomes.subsequent_dispatches_reused_generation;
    report.checks.subsequent_dispatches_uploaded_zero_weight_bytes =
        outcomes.subsequent_dispatches_uploaded_zero_weight_bytes;
    report.checks.every_dispatch_completed_gpu_io = outcomes.every_dispatch_completed_gpu_io;
    report.checks.zero_evictions_or_stale_retirements =
        outcomes.zero_evictions_or_stale_retirements;
    report.checks.zero_cpu_fallback_or_degraded_execution =
        outcomes.zero_cpu_fallback_or_degraded_execution;
    report.checks.complete_expert_vectors_passed = outcomes.complete_expert_vectors_passed;
    if let Some(failure) = tolerance_failure {
        return fail_q4_parity(report, failure, args.report_out.as_deref());
    }
    if let Err(failure) = report.finish() {
        return fail_q4_parity(report, failure, args.report_out.as_deref());
    }
    emit_q4_parity_report(&report, args.report_out.as_deref())
}

fn emit_greedy_parity_report(
    report: &crate::greedy_parity::GreedyParityReport,
    report_out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    if let Some(path) = report_out {
        std::fs::write(path, json)?;
        eprintln!("greedy-token parity report written to {}", path.display());
    } else {
        use std::io::Write as _;
        std::io::stdout().write_all(&json)?;
    }
    Ok(())
}

fn fail_greedy_parity(
    mut report: crate::greedy_parity::GreedyParityReport,
    failure: crate::qualification::QualificationFailure,
    report_out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let summary = format!("{}: {}", failure.code, failure.detail);
    report.fail(failure);
    match emit_greedy_parity_report(&report, report_out) {
        Ok(()) => Err(summary.into()),
        Err(emit_error) => Err(format!(
            "{summary}; additionally failed to emit greedy-token parity report: {emit_error}"
        )
        .into()),
    }
}

fn greedy_parity_model_identity(
    spec: &ResolvedRealCliSpec,
) -> crate::greedy_parity::ModelIdentityEvidence {
    crate::greedy_parity::ModelIdentityEvidence {
        architecture: spec.architecture.model_type().to_string(),
        num_layers: spec.cfg.model.num_layers,
        num_experts_per_layer: spec.cfg.model.num_experts,
        total_experts: (spec.cfg.model.num_layers as u64)
            .saturating_mul(spec.cfg.model.num_experts as u64),
        top_k: spec.cfg.model.top_k,
        d_model: spec.cfg.model.d_model,
        d_ff: spec.cfg.model.d_ff,
        routed_expert_dtype: spec.cfg.model.dtype.as_str().to_string(),
    }
}

fn greedy_parity_model_load(
    runtime: &BenchRealRuntime,
) -> crate::greedy_parity::ModelLoadEvidence {
    let load = &runtime.model.load_status;
    crate::greedy_parity::ModelLoadEvidence {
        strict: load.strict,
        loader: load.loader.to_string(),
        loaded_tensors: load.loaded_tensors,
        required_tensors: load.required_tensors,
        optional_probed: load.optional_probed,
        optional_loaded: load.optional_loaded,
        seeded_fallback_remained: load.seeded_fallback_remained,
    }
}

fn greedy_parity_runtime_cache_snapshot(
    runtime: &BenchRealRuntime,
) -> crate::greedy_parity::RuntimeCacheSnapshot {
    let report = runtime.engine.report();
    let gpu_cache = runtime.engine.execution_context().gpu_expert_cache();
    let memory = runtime.engine.gpu_expert_memory_snapshot();
    crate::greedy_parity::RuntimeCacheSnapshot {
        ram_entries: runtime
            .isolated_cache
            .as_ref()
            .map_or(0, |cache| cache.len()),
        ram_hits: report.hits,
        ram_misses: report.misses,
        bytes_read: report.bytes_read,
        prefetch_completed: report.prefetch_completed,
        predictor_observations: report.predictor_observations,
        logical_gpu_hits: gpu_cache.hits(),
        logical_gpu_misses: gpu_cache.misses(),
        logical_gpu_promotions: gpu_cache.promotions(),
        logical_admitted_bytes: gpu_cache.used_bytes(),
        logical_anchor_entries: gpu_cache.anchor_len(),
        logical_lru_entries: gpu_cache.lru_len(),
        physical_entries: memory.map_or(0, |snapshot| snapshot.physical_entries),
        physical_installs: memory.map_or(0, |snapshot| snapshot.physical_installs),
        physical_evictions: memory.map_or(0, |snapshot| snapshot.physical_evictions),
        stale_retirements: memory.map_or(0, |snapshot| snapshot.stale_retirements),
    }
}

fn greedy_parity_failure_policy_name(
    policy: crate::engine::RoutedExpertGpuFailurePolicy,
) -> &'static str {
    match policy {
        crate::engine::RoutedExpertGpuFailurePolicy::StrictFailClosed => "strict-fail-closed",
        crate::engine::RoutedExpertGpuFailurePolicy::ServingCpuFallback => {
            "serving-cpu-fallback"
        }
    }
}

// Qualification call sites deliberately name every frozen/shared input; this
// makes configuration and token-ID drift reviewable at the boundary.
#[allow(clippy::too_many_arguments)]
async fn execute_greedy_parity_plane(
    spec: &ResolvedRealCliSpec,
    mode: RealCliRuntimeMode,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    prompt_token_ids: &[u32],
    prompt_token_ids_sha256: &str,
    resolved_config_sha256: &str,
    expected_adapter_name: &str,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
    case_name: &str,
) -> Result<crate::greedy_parity::PlaneRunEvidence, Box<dyn std::error::Error>> {
    execute_greedy_parity_plane_internal(
        spec,
        mode,
        tokenizer,
        prompt_token_ids,
        prompt_token_ids_sha256,
        resolved_config_sha256,
        expected_adapter_name,
        watchdog,
        case_name,
        None,
        None,
        false,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn execute_greedy_parity_boundary_reference_plane(
    spec: &ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    prompt_token_ids: &[u32],
    prompt_token_ids_sha256: &str,
    resolved_config_sha256: &str,
    expected_adapter_name: &str,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
    case_name: &str,
) -> Result<crate::greedy_parity::PlaneRunEvidence, Box<dyn std::error::Error>> {
    execute_greedy_parity_plane_internal(
        spec,
        RealCliRuntimeMode::IsolatedGreedyParityCpu,
        tokenizer,
        prompt_token_ids,
        prompt_token_ids_sha256,
        resolved_config_sha256,
        expected_adapter_name,
        watchdog,
        case_name,
        None,
        None,
        true,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn execute_greedy_parity_plane_with_logits(
    spec: &ResolvedRealCliSpec,
    mode: RealCliRuntimeMode,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    prompt_token_ids: &[u32],
    prompt_token_ids_sha256: &str,
    resolved_config_sha256: &str,
    expected_adapter_name: &str,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
    case_name: &str,
    first_token_logit_bits: &mut Vec<u32>,
    route_capture: &mut Option<crate::engine::RoutedFfnDiagnosticCapture>,
    cpu_q4_boundary_emulation: bool,
) -> Result<crate::greedy_parity::PlaneRunEvidence, Box<dyn std::error::Error>> {
    execute_greedy_parity_plane_internal(
        spec,
        mode,
        tokenizer,
        prompt_token_ids,
        prompt_token_ids_sha256,
        resolved_config_sha256,
        expected_adapter_name,
        watchdog,
        case_name,
        Some(first_token_logit_bits),
        Some(route_capture),
        cpu_q4_boundary_emulation,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn execute_greedy_parity_plane_internal(
    spec: &ResolvedRealCliSpec,
    mode: RealCliRuntimeMode,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    prompt_token_ids: &[u32],
    prompt_token_ids_sha256: &str,
    resolved_config_sha256: &str,
    expected_adapter_name: &str,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
    case_name: &str,
    first_token_logit_bits: Option<&mut Vec<u32>>,
    route_capture: Option<&mut Option<crate::engine::RoutedFfnDiagnosticCapture>>,
    cpu_q4_boundary_emulation: bool,
) -> Result<crate::greedy_parity::PlaneRunEvidence, Box<dyn std::error::Error>> {
    use crate::qualification::{gpu_io_delta, routed_execution_delta, validate_memory};

    let runtime = build_isolated_greedy_runtime(spec, mode, tokenizer).await?;
    let attempt = async {
        if cpu_q4_boundary_emulation {
            runtime.engine.enable_cpu_q4_boundary_emulation()?;
        }
        let boundary_emulation_before =
            runtime.engine.cpu_q4_boundary_emulation_snapshot();
        if boundary_emulation_before.enabled != cpu_q4_boundary_emulation
            || boundary_emulation_before.routed_expert_dispatches != 0
        {
            return Err(format!(
                "{case_name} runtime Q4 boundary-emulation state did not start clean or match the requested execution mode"
            )
            .into());
        }
        let context = runtime.engine.execution_context();
        let execution_plan: crate::qualification::ExecutionPlanEvidence = context.plan().into();
        let plane = match mode {
            RealCliRuntimeMode::IsolatedGreedyParityCpu => "cpu",
            RealCliRuntimeMode::IsolatedGreedyParityHybrid => "hybrid",
            _ => return Err("non-isolated mode reached greedy parity execution".into()),
        };
        let device = runtime.engine.gpu_device_identity();
        match mode {
            RealCliRuntimeMode::IsolatedGreedyParityCpu => {
                if !crate::greedy_parity::cpu_plan_exact(&execution_plan) {
                    return Err(format!(
                        "CPU control did not resolve every component to CPU: {execution_plan:?}"
                    )
                    .into());
                }
                if device.is_some() {
                    return Err("CPU control unexpectedly exposes a GPU device identity".into());
                }
            }
            RealCliRuntimeMode::IsolatedGreedyParityHybrid => {
                if !crate::greedy_parity::hybrid_plan_exact(&execution_plan) {
                    return Err(format!(
                        "Hybrid run did not resolve the strict component plan: {execution_plan:?}"
                    )
                    .into());
                }
                let selected = device.as_ref().ok_or(
                    "strict Hybrid run has no authoritative GPU device identity",
                )?;
                if selected.software_adapter
                    || selected.device_type.eq_ignore_ascii_case("cpu")
                {
                    return Err(format!(
                        "strict Hybrid selected software adapter {:?}",
                        selected.name
                    )
                    .into());
                }
                if selected.name != expected_adapter_name {
                    return Err(format!(
                        "strict Hybrid selected adapter {:?}, expected exact name {:?}",
                        selected.name, expected_adapter_name
                    )
                    .into());
                }
                if runtime.engine.routed_expert_gpu_failure_policy()
                    != crate::engine::RoutedExpertGpuFailurePolicy::StrictFailClosed
                {
                    return Err("strict Hybrid engine does not use StrictFailClosed".into());
                }
            }
            _ => unreachable!("isolated modes checked above"),
        }

        let cache_before = greedy_parity_runtime_cache_snapshot(&runtime);
        let observed_config_sha256 = resolved_real_runtime_identity_sha256(
            &runtime.cfg,
            runtime.model.config.architecture,
            runtime.model.config.first_k_dense_replace,
            &runtime.model.config.advanced,
        )?;
        if observed_config_sha256 != resolved_config_sha256 {
            return Err(format!(
                "{plane} runtime identity {observed_config_sha256} drifted from frozen specification {resolved_config_sha256}"
            )
            .into());
        }
        let routed_before = runtime.engine.routed_expert_execution_snapshot();
        let io_before_option = runtime.engine.gpu_expert_io_snapshot();
        let io_before = io_before_option.unwrap_or_default();
        let memory_before = runtime.engine.gpu_expert_memory_snapshot();
        let attention_softmax_before = crate::transformer::nonfinite_softmax_fallbacks();
        if cache_before != crate::greedy_parity::RuntimeCacheSnapshot::default()
            || routed_before != crate::engine::RoutedExpertExecutionSnapshot::default()
            || io_before != crate::backend::GpuExpertIoSnapshot::default()
        {
            return Err(format!(
                "{plane} isolated runtime did not start from clean cache/counter state"
            )
            .into());
        }
        if let Some(snapshot) = memory_before {
            validate_memory(snapshot).map_err(|failure| failure.detail)?;
        }
        if route_capture.is_some() {
            let target_token_idx = prompt_token_ids
                .len()
                .checked_sub(1)
                .and_then(|position| position.checked_mul(runtime.model.config.num_layers))
                .and_then(|token_idx| u64::try_from(token_idx).ok())
                .ok_or("layer-0 route capture token index overflowed")?;
            runtime
                .engine
                .arm_layer0_route_capture(target_token_idx)?;
        }
        let measured = with_progress_timeout(
            format!("greedy parity {case_name} {plane} inference"),
            watchdog,
            run_real_once_from_token_ids_internal(
                &runtime,
                prompt_token_ids,
                crate::greedy_parity::OUTPUT_TOKEN_LIMIT,
                crate::sampling::SamplingParams::greedy(),
                0,
                first_token_logit_bits,
            ),
        )
        .await?;
        if let Some(destination) = route_capture {
            *destination = Some(
                runtime
                    .engine
                    .take_layer0_route_capture()
                    .ok_or("final-prompt layer-0 routed-FFN input was not captured")?,
            );
        }
        let routed_after = runtime.engine.routed_expert_execution_snapshot();
        let cpu_q4_boundary_emulation =
            runtime.engine.cpu_q4_boundary_emulation_snapshot();
        let io_after_option = runtime.engine.gpu_expert_io_snapshot();
        let io_after = io_after_option.unwrap_or_default();
        let memory_after = runtime.engine.gpu_expert_memory_snapshot();
        let attention_softmax_nonfinite_fallbacks =
            crate::transformer::nonfinite_softmax_fallbacks()
                .saturating_sub(attention_softmax_before);
        if let Some(snapshot) = memory_after {
            validate_memory(snapshot).map_err(|failure| failure.detail)?;
        }
        if mode == RealCliRuntimeMode::IsolatedGreedyParityHybrid
            && (io_before_option.is_none()
                || io_after_option.is_none()
                || memory_before.is_none()
                || memory_after.is_none())
        {
            return Err("strict Hybrid GPU evidence disappeared during inference".into());
        }
        if mode == RealCliRuntimeMode::IsolatedGreedyParityCpu
            && (io_before_option.is_some()
                || io_after_option.is_some()
                || memory_before.is_some()
                || memory_after.is_some())
        {
            return Err("CPU control unexpectedly exposed GPU I/O or memory evidence".into());
        }

        let routed_delta = routed_execution_delta(routed_before, routed_after)
            .map_err(|failure| failure.detail)?;
        let gpu_io_delta = gpu_io_delta(io_before, io_after).map_err(|failure| failure.detail)?;
        let generated_token_ids = measured.report.output_token_ids;
        let generated_text = measured.report.output_text;
        let generation = crate::greedy_parity::GenerationEvidence {
            prompt_token_ids_sha256: prompt_token_ids_sha256.to_string(),
            generated_token_ids_sha256: crate::greedy_parity::token_ids_sha256(
                &generated_token_ids,
            ),
            generated_text_sha256: crate::greedy_parity::sha256_hex(generated_text.as_bytes()),
            generated_token_count: generated_token_ids.len(),
            generated_token_ids,
            termination_reason: crate::greedy_parity::TerminationReason::LengthLimit,
        };
        let initial_kv_sequence_lengths = measured.initial_kv_sequence_lengths;
        let initial_state = crate::greedy_parity::InitialStateEvidence {
            context_id: execution_plan.context_id.clone(),
            resolved_config_sha256: observed_config_sha256,
            kv_cache_count: initial_kv_sequence_lengths.len(),
            all_kv_empty: initial_kv_sequence_lengths.iter().all(|&length| length == 0),
            kv_sequence_lengths: initial_kv_sequence_lengths,
            cache: cache_before,
            routed: routed_before,
            gpu_io_available: io_before_option.is_some(),
            gpu_io: io_before,
        };
        Ok(crate::greedy_parity::PlaneRunEvidence {
            plane: plane.to_string(),
            model_load: greedy_parity_model_load(&runtime),
            execution_plan,
            routed_expert_gpu_failure_policy: greedy_parity_failure_policy_name(
                runtime.engine.routed_expert_gpu_failure_policy(),
            )
            .to_string(),
            cpu_q4_boundary_emulation,
            device,
            initial_state,
            generation,
            routed_execution_delta: routed_delta,
            gpu_io_delta,
            attention_softmax_nonfinite_fallbacks,
            gpu_memory_before: memory_before,
            gpu_memory_after: memory_after,
            background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence::default(),
            worker_process: None,
        })
    }
    .await;

    // Always consume and shut down a successfully constructed isolated runtime,
    // including when inference or evidence validation failed.
    let shutdown = runtime.shutdown_isolated().await;
    match (attempt, shutdown) {
        (Ok(mut evidence), Ok(shutdown)) => {
            evidence.background_shutdown = shutdown;
            Ok(evidence)
        }
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error.into()),
        (Err(error), Err(shutdown_error)) => Err(IsolatedRuntimeShutdownError::new(format!(
            "{error}; isolated shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

#[derive(Debug)]
struct BoundedChildOutput {
    bytes: Vec<u8>,
    truncated: bool,
}

#[derive(Debug)]
struct GreedyParityWorkerCapture {
    process_id: u32,
    status: std::process::ExitStatus,
    timed_out: bool,
    reaped: bool,
    stdout: BoundedChildOutput,
    stderr: BoundedChildOutput,
    transport_error: Option<String>,
}

fn read_child_output_bounded(
    mut reader: impl std::io::Read,
    limit: usize,
) -> std::io::Result<BoundedChildOutput> {
    let mut retained = Vec::with_capacity(limit.min(8192));
    let mut buffer = [0_u8; 8192];
    let mut truncated = false;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let remaining = limit.saturating_sub(retained.len());
        let keep = remaining.min(read);
        retained.extend_from_slice(&buffer[..keep]);
        truncated |= keep != read;
    }
    Ok(BoundedChildOutput {
        bytes: retained,
        truncated,
    })
}

#[cfg(unix)]
fn child_exit_signal(status: &std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt as _;
    status.signal()
}

#[cfg(not(unix))]
fn child_exit_signal(_status: &std::process::ExitStatus) -> Option<i32> {
    None
}

/// Spawn, feed, drain, bound, and reap one worker. Both pipes are drained on
/// dedicated threads so verbose diagnostics cannot deadlock the child. A
/// timeout always kills and then waits for the exact child before returning.
async fn run_greedy_parity_worker_process(
    command: std::process::Command,
    request_json: Vec<u8>,
    timeout: Duration,
) -> Result<GreedyParityWorkerCapture, String> {
    run_greedy_parity_worker_process_with_limits(
        command,
        request_json,
        timeout,
        crate::greedy_parity::MAX_WORKER_STDOUT_BYTES,
        crate::greedy_parity::MAX_WORKER_STDERR_BYTES,
    )
    .await
}

async fn run_greedy_parity_worker_process_with_limits(
    mut command: std::process::Command,
    request_json: Vec<u8>,
    timeout: Duration,
    stdout_limit: usize,
    stderr_limit: usize,
) -> Result<GreedyParityWorkerCapture, String> {
    use std::io::Write as _;
    use std::process::Stdio;

    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| format!("failed to spawn Hybrid worker: {error}"))?;
    let process_id = child.id();
    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            return Err("spawned Hybrid worker has no stdout pipe".to_string());
        }
    };
    let stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => {
            let _ = child.kill();
            let _ = child.wait();
            return Err("spawned Hybrid worker has no stderr pipe".to_string());
        }
    };
    let stdout_thread =
        std::thread::spawn(move || read_child_output_bounded(stdout, stdout_limit));
    let stderr_thread =
        std::thread::spawn(move || read_child_output_bounded(stderr, stderr_limit));

    let mut transport_error = None;
    match child.stdin.take() {
        Some(mut stdin) => {
            if let Err(error) = stdin.write_all(&request_json) {
                transport_error = Some(format!("failed to write Hybrid worker request: {error}"));
                let _ = child.kill();
            }
        }
        None => {
            transport_error = Some("spawned Hybrid worker has no stdin pipe".to_string());
            let _ = child.kill();
        }
    }

    let started = Instant::now();
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() < timeout && transport_error.is_none() => {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Ok(None) => {
                timed_out = transport_error.is_none();
                if let Err(error) = child.kill() {
                    transport_error.get_or_insert_with(|| {
                        format!("failed to kill Hybrid worker after timeout: {error}")
                    });
                }
                break child
                    .wait()
                    .map_err(|error| format!("failed to reap Hybrid worker: {error}"))?;
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("failed to query Hybrid worker status: {error}"));
            }
        }
    };

    let stdout = stdout_thread
        .join()
        .map_err(|_| "Hybrid worker stdout reader thread panicked".to_string())?
        .map_err(|error| format!("failed to read Hybrid worker stdout: {error}"))?;
    let stderr = stderr_thread
        .join()
        .map_err(|_| "Hybrid worker stderr reader thread panicked".to_string())?
        .map_err(|error| format!("failed to read Hybrid worker stderr: {error}"))?;
    Ok(GreedyParityWorkerCapture {
        process_id,
        status,
        timed_out,
        reaped: true,
        stdout,
        stderr,
        transport_error,
    })
}

fn current_executable_identity() -> Result<(PathBuf, String), Box<dyn std::error::Error>> {
    let executable = std::env::current_exe()?;
    let digest = crate::qualification::hash_small_file(&executable)?;
    Ok((executable, digest.sha256))
}

fn emit_hybrid_worker_response(
    response: &crate::greedy_parity::HybridWorkerResponse,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write as _;
    let stdout = std::io::stdout();
    let mut locked = stdout.lock();
    serde_json::to_writer(&mut locked, response)?;
    locked.write_all(b"\n")?;
    locked.flush()?;
    Ok(())
}

/// Hidden same-binary entry point. It never tokenizes: the only prompt input
/// accepted is the parent's typed token-ID vector on stdin.
async fn cmd_greedy_parity_hybrid_worker_internal(
    args: GreedyParityHybridWorkerArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Read as _;

    let mut input = Vec::new();
    std::io::stdin()
        .take((crate::greedy_parity::MAX_WORKER_STDOUT_BYTES + 1) as u64)
        .read_to_end(&mut input)?;
    if input.len() > crate::greedy_parity::MAX_WORKER_STDOUT_BYTES {
        return Err("Hybrid worker request exceeds the private protocol limit".into());
    }
    let request = crate::greedy_parity::parse_hybrid_worker_request_exact(&input)?;
    let spec = resolve_real_cli_spec_from_config(
        args.parsed_config,
        RealCliRuntimeMode::IsolatedGreedyParityHybrid,
    )?;
    let observed_config_sha256 = resolved_real_cli_spec_sha256(&spec)?;
    let (_, observed_executable_sha256) = current_executable_identity()?;
    let provenance = crate::qualification::BuildProvenance::embedded();
    let identity = crate::greedy_parity::validate_hybrid_worker_request(
        &request,
        &observed_config_sha256,
        &observed_executable_sha256,
        provenance.git_sha.as_deref(),
    );
    if !identity.all_verified() {
        let response = crate::greedy_parity::HybridWorkerResponse::from_request(
            &request,
            &observed_config_sha256,
            &observed_executable_sha256,
            provenance.git_sha.as_deref(),
            identity,
            None,
            Some("Hybrid worker request identity validation failed".to_string()),
        );
        emit_hybrid_worker_response(&response)?;
        return Err("Hybrid worker request identity validation failed".into());
    }

    let tokenizer = load_real_cli_tokenizer(
        &spec.cfg,
        RealCliRuntimeMode::IsolatedGreedyParityHybrid,
    )?;
    let attempt = execute_greedy_parity_plane(
        &spec,
        RealCliRuntimeMode::IsolatedGreedyParityHybrid,
        tokenizer,
        &request.prompt_token_ids,
        &request.prompt_token_ids_sha256,
        &request.resolved_config_sha256,
        &request.expected_adapter_name,
        args.progress_watchdog,
        &request.case_name,
    )
    .await;
    match attempt {
        Ok(plane) => {
            let response = crate::greedy_parity::HybridWorkerResponse::from_request(
                &request,
                &observed_config_sha256,
                &observed_executable_sha256,
                provenance.git_sha.as_deref(),
                identity,
                Some(plane),
                None,
            );
            emit_hybrid_worker_response(&response)
        }
        Err(error) => {
            let detail = error.to_string();
            let response = crate::greedy_parity::HybridWorkerResponse::from_request(
                &request,
                &observed_config_sha256,
                &observed_executable_sha256,
                provenance.git_sha.as_deref(),
                identity,
                None,
                Some(detail.clone()),
            );
            emit_hybrid_worker_response(&response)?;
            Err(detail.into())
        }
    }
}

fn emit_logit_worker_response(
    response: &crate::numerical_diagnostics::DiagnosticWorkerResponse,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write as _;
    let stdout = std::io::stdout();
    let mut locked = stdout.lock();
    serde_json::to_writer(&mut locked, response)?;
    locked.write_all(b"\n")?;
    locked.flush()?;
    Ok(())
}

/// Hidden same-binary worker. The request embeds the existing greedy-parity
/// identity contract; the only new controls are plane and repeated-run index.
async fn cmd_greedy_parity_logit_worker_internal(
    args: GreedyParityHybridWorkerArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Read as _;

    let mut input = Vec::new();
    std::io::stdin()
        .take((crate::greedy_parity::MAX_WORKER_STDOUT_BYTES + 1) as u64)
        .read_to_end(&mut input)?;
    if input.len() > crate::greedy_parity::MAX_WORKER_STDOUT_BYTES {
        return Err("logit worker request exceeds the private protocol limit".into());
    }
    let request = crate::numerical_diagnostics::parse_worker_request_exact(&input)?;
    request.validate_static()?;
    let spec = resolve_real_cli_spec_from_config(
        args.parsed_config,
        RealCliRuntimeMode::IsolatedGreedyParityHybrid,
    )?;
    let observed_config_sha256 = resolved_real_cli_spec_sha256(&spec)?;
    let (_, observed_executable_sha256) = current_executable_identity()?;
    let provenance = crate::qualification::BuildProvenance::embedded();
    let identity = crate::greedy_parity::validate_hybrid_worker_request(
        &request.base,
        &observed_config_sha256,
        &observed_executable_sha256,
        provenance.git_sha.as_deref(),
    );
    if !identity.all_verified() {
        let detail = "logit worker request identity validation failed".to_string();
        let response = crate::numerical_diagnostics::DiagnosticWorkerResponse {
            protocol_version: crate::numerical_diagnostics::WORKER_PROTOCOL_VERSION.to_string(),
            plane: request.plane,
            run_index: request.run_index,
            base: crate::greedy_parity::HybridWorkerResponse::from_request(
                &request.base,
                &observed_config_sha256,
                &observed_executable_sha256,
                provenance.git_sha.as_deref(),
                identity,
                None,
                Some(detail.clone()),
            ),
            chosen_token_id: None,
            first_token_logit_bits: None,
            first_token_logit_bits_sha256: None,
            route_capture: None,
            failure: Some(detail.clone()),
        };
        emit_logit_worker_response(&response)?;
        return Err(detail.into());
    }
    let mode = match request.plane {
        crate::numerical_diagnostics::DiagnosticPlane::Cpu => {
            RealCliRuntimeMode::IsolatedGreedyParityCpu
        }
        crate::numerical_diagnostics::DiagnosticPlane::CpuBoundaryEmulation => {
            RealCliRuntimeMode::IsolatedGreedyParityCpu
        }
        crate::numerical_diagnostics::DiagnosticPlane::Hybrid => {
            RealCliRuntimeMode::IsolatedGreedyParityHybrid
        }
    };
    let tokenizer = load_real_cli_tokenizer(
        &spec.cfg,
        RealCliRuntimeMode::IsolatedGreedyParityHybrid,
    )?;
    let mut logit_bits = Vec::new();
    let mut route_capture = None;
    let attempt = execute_greedy_parity_plane_with_logits(
        &spec,
        mode,
        tokenizer,
        &request.base.prompt_token_ids,
        &request.base.prompt_token_ids_sha256,
        &request.base.resolved_config_sha256,
        &request.base.expected_adapter_name,
        args.progress_watchdog,
        &request.base.case_name,
        &mut logit_bits,
        &mut route_capture,
        request.plane
            == crate::numerical_diagnostics::DiagnosticPlane::CpuBoundaryEmulation,
    )
    .await;
    match attempt {
        Ok(plane) => {
            if logit_bits.is_empty()
                || logit_bits.len() > crate::numerical_diagnostics::MAX_VOCAB_SIZE
            {
                return Err("logit worker captured an invalid vocabulary vector".into());
            }
            let chosen_token_id = plane
                .generation
                .generated_token_ids
                .first()
                .copied()
                .ok_or("logit worker generated no first token")?;
            let logit_sha256 = crate::numerical_diagnostics::f32_bits_sha256(&logit_bits);
            let response = crate::numerical_diagnostics::DiagnosticWorkerResponse {
                protocol_version: crate::numerical_diagnostics::WORKER_PROTOCOL_VERSION
                    .to_string(),
                plane: request.plane,
                run_index: request.run_index,
                base: crate::greedy_parity::HybridWorkerResponse::from_request(
                    &request.base,
                    &observed_config_sha256,
                    &observed_executable_sha256,
                    provenance.git_sha.as_deref(),
                    identity,
                    Some(plane),
                    None,
                ),
                chosen_token_id: Some(chosen_token_id),
                first_token_logit_bits: Some(logit_bits),
                first_token_logit_bits_sha256: Some(logit_sha256),
                route_capture,
                failure: None,
            };
            emit_logit_worker_response(&response)
        }
        Err(error) => {
            let detail = error.to_string();
            let response = crate::numerical_diagnostics::DiagnosticWorkerResponse {
                protocol_version: crate::numerical_diagnostics::WORKER_PROTOCOL_VERSION
                    .to_string(),
                plane: request.plane,
                run_index: request.run_index,
                base: crate::greedy_parity::HybridWorkerResponse::from_request(
                    &request.base,
                    &observed_config_sha256,
                    &observed_executable_sha256,
                    provenance.git_sha.as_deref(),
                    identity,
                    None,
                    Some(detail.clone()),
                ),
                chosen_token_id: None,
                first_token_logit_bits: None,
                first_token_logit_bits_sha256: None,
                route_capture: None,
                failure: Some(detail.clone()),
            };
            emit_logit_worker_response(&response)?;
            Err(detail.into())
        }
    }
}

#[derive(Debug)]
struct GreedyParityWorkerCaseFailure {
    detail: String,
    evidence: crate::greedy_parity::HybridWorkerFailureEvidence,
}

#[allow(clippy::too_many_arguments)]
async fn execute_greedy_parity_hybrid_worker(
    executable: &Path,
    config: &Path,
    request: &crate::greedy_parity::HybridWorkerRequest,
    timeout: Duration,
) -> Result<crate::greedy_parity::PlaneRunEvidence, GreedyParityWorkerCaseFailure> {
    let request_json = serde_json::to_vec(request).map_err(|error| {
        GreedyParityWorkerCaseFailure {
            detail: format!("failed to serialize Hybrid worker request: {error}"),
            evidence: crate::greedy_parity::HybridWorkerFailureEvidence {
                worker_id: request.worker_id.clone(),
                case_name: request.case_name.clone(),
                child_process_spawned: false,
                process_id: None,
                exit_code: None,
                signal: None,
                timed_out: false,
                process_reaped: false,
                evidence_emitted: false,
                identity_validation_succeeded: false,
                identity_validation: None,
                stderr: String::new(),
                stderr_truncated: false,
            },
        }
    })?;
    let mut command = Command::new(executable);
    command
        .arg("--progress-timeout-secs")
        .arg(timeout.as_secs().to_string())
        .arg("greedy-parity-hybrid-worker-internal")
        .arg("--config")
        .arg(config);
    let capture = run_greedy_parity_worker_process(command, request_json, timeout)
        .await
        .map_err(|detail| GreedyParityWorkerCaseFailure {
            detail,
            evidence: crate::greedy_parity::HybridWorkerFailureEvidence {
                worker_id: request.worker_id.clone(),
                case_name: request.case_name.clone(),
                child_process_spawned: false,
                process_id: None,
                exit_code: None,
                signal: None,
                timed_out: false,
                process_reaped: false,
                evidence_emitted: false,
                identity_validation_succeeded: false,
                identity_validation: None,
                stderr: String::new(),
                stderr_truncated: false,
            },
        })?;

    let stderr = String::from_utf8_lossy(&capture.stderr.bytes).into_owned();
    if !stderr.is_empty() {
        eprintln!(
            "greedy parity Hybrid worker {} stderr{}:\n{}",
            request.case_name,
            if capture.stderr.truncated {
                " (bounded/truncated)"
            } else {
                ""
            },
            stderr
        );
    }
    let response_result = if capture.stdout.truncated {
        Err(format!(
            "Hybrid worker stdout exceeded {} bytes",
            crate::greedy_parity::MAX_WORKER_STDOUT_BYTES
        ))
    } else {
        crate::greedy_parity::parse_hybrid_worker_response_exact(&capture.stdout.bytes)
    };
    let response = response_result.as_ref().ok();
    let validation = response.map(|response| {
        crate::greedy_parity::validate_hybrid_worker_response(request, response)
    });
    let exit_code = capture.status.code();
    let signal = child_exit_signal(&capture.status);
    let normal_zero_exit = capture.status.success()
        && exit_code == Some(0)
        && signal.is_none()
        && !capture.timed_out
        && capture.transport_error.is_none();
    let process = crate::greedy_parity::HybridWorkerProcessEvidence {
        worker_id: request.worker_id.clone(),
        child_process_spawned: true,
        process_id: Some(capture.process_id),
        executable_sha256: request.executable_sha256.clone(),
        build_git_sha: request.build_git_sha.clone(),
        executable_identity_verified: validation
            .is_some_and(|value| value.executable_identity_verified),
        build_sha_identity_verified: validation
            .is_some_and(|value| value.build_sha_identity_verified),
        case_identity_verified: validation.is_some_and(|value| value.case_identity_verified),
        config_identity_verified: validation
            .is_some_and(|value| value.config_identity_verified),
        expected_adapter_identity_verified: validation
            .is_some_and(|value| value.expected_adapter_identity_verified),
        prompt_token_identity_verified: validation
            .is_some_and(|value| value.prompt_token_identity_verified),
        output_token_limit_verified: validation
            .is_some_and(|value| value.output_token_limit_verified),
        greedy_sampling_identity_verified: validation
            .is_some_and(|value| value.greedy_sampling_identity_verified),
        normal_zero_exit,
        exit_code,
        signal,
        process_reaped: capture.reaped,
        timed_out: capture.timed_out,
        evidence_emitted: response.is_some(),
    };
    let response_failure = response.and_then(|response| response.failure.as_deref());
    let success = process.normal_zero_exit
        && process.process_reaped
        && validation.is_some_and(|value| value.all_verified())
        && response.is_some_and(|response| {
            response.failure.is_none()
                && response.plane.is_some()
                && response
                    .plane
                    .as_ref()
                    .is_some_and(|plane| plane.worker_process.is_none())
        });
    if success {
        let mut plane = response.unwrap().plane.clone().unwrap();
        plane.worker_process = Some(process);
        return Ok(plane);
    }

    let parse_detail = response_result.as_ref().err().cloned();
    let detail = [
        capture.transport_error.as_deref(),
        parse_detail.as_deref(),
        response_failure,
        (!normal_zero_exit).then_some("Hybrid worker did not exit normally with status zero"),
        (!capture.reaped).then_some("Hybrid worker was not reaped"),
        validation
            .is_some_and(|value| !value.all_verified())
            .then_some("Hybrid worker response identity validation failed"),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join("; ");
    Err(GreedyParityWorkerCaseFailure {
        detail: if detail.is_empty() {
            "Hybrid worker emitted no successful plane evidence".to_string()
        } else {
            detail
        },
        evidence: crate::greedy_parity::HybridWorkerFailureEvidence {
            worker_id: request.worker_id.clone(),
            case_name: request.case_name.clone(),
            child_process_spawned: true,
            process_id: Some(capture.process_id),
            exit_code,
            signal,
            timed_out: capture.timed_out,
            process_reaped: capture.reaped,
            evidence_emitted: response.is_some(),
            identity_validation_succeeded: validation
                .is_some_and(|value| value.all_verified()),
            identity_validation: validation,
            stderr,
            stderr_truncated: capture.stderr.truncated,
        },
    })
}

struct CompletedLogitWorker {
    run: crate::numerical_diagnostics::RepeatedRunEvidence,
    first_token_logit_bits: Vec<u32>,
    chosen_token_id: u32,
    route_capture: crate::engine::RoutedFfnDiagnosticCapture,
}

async fn execute_logit_diagnostic_worker(
    executable: &Path,
    config: &Path,
    request: &crate::numerical_diagnostics::DiagnosticWorkerRequest,
    timeout: Duration,
) -> Result<
    CompletedLogitWorker,
    crate::numerical_diagnostics::DiagnosticWorkerFailureEvidence,
> {
    let empty_process = || crate::greedy_parity::HybridWorkerProcessEvidence {
        worker_id: request.base.worker_id.clone(),
        executable_sha256: request.base.executable_sha256.clone(),
        build_git_sha: request.base.build_git_sha.clone(),
        ..Default::default()
    };
    let request_json = serde_json::to_vec(request).map_err(|error| {
        crate::numerical_diagnostics::DiagnosticWorkerFailureEvidence {
            worker_id: request.base.worker_id.clone(),
            plane: request.plane,
            run_index: request.run_index,
            detail: format!("failed to serialize logit worker request: {error}"),
            process: empty_process(),
            stderr: String::new(),
            stderr_truncated: false,
        }
    })?;
    let mut command = Command::new(executable);
    command
        .arg("--progress-timeout-secs")
        .arg(timeout.as_secs().to_string())
        .arg("greedy-parity-logit-worker-internal")
        .arg("--config")
        .arg(config);
    let capture = run_greedy_parity_worker_process_with_limits(
        command,
        request_json,
        timeout,
        crate::numerical_diagnostics::MAX_WORKER_STDOUT_BYTES,
        crate::greedy_parity::MAX_WORKER_STDERR_BYTES,
    )
    .await
    .map_err(|detail| crate::numerical_diagnostics::DiagnosticWorkerFailureEvidence {
        worker_id: request.base.worker_id.clone(),
        plane: request.plane,
        run_index: request.run_index,
        detail,
        process: empty_process(),
        stderr: String::new(),
        stderr_truncated: false,
    })?;
    let stderr = String::from_utf8_lossy(&capture.stderr.bytes).into_owned();
    let response_result = if capture.stdout.truncated {
        Err(format!(
            "logit worker stdout exceeded {} bytes",
            crate::numerical_diagnostics::MAX_WORKER_STDOUT_BYTES
        ))
    } else {
        crate::numerical_diagnostics::parse_worker_response_exact(&capture.stdout.bytes)
    };
    let parse_detail = response_result.as_ref().err().cloned();
    let response = response_result.as_ref().ok();
    let validation = response.map(|response| {
        crate::greedy_parity::validate_hybrid_worker_response(&request.base, &response.base)
    });
    let response_identity_exact = response.is_some_and(|response| {
        response.protocol_version == crate::numerical_diagnostics::WORKER_PROTOCOL_VERSION
            && response.plane == request.plane
            && response.run_index == request.run_index
            && response.base.worker_id == request.base.worker_id
    });
    let identity_attested = response_identity_exact
        && validation.is_some_and(|value| value.all_verified())
        && response.is_some_and(|response| {
            response.failure.is_none()
                && response.base.failure.is_none()
                && response.base.plane.is_some()
                && response.chosen_token_id.is_some()
                && response.first_token_logit_bits.is_some()
                && response.first_token_logit_bits_sha256.is_some()
                && response.route_capture.is_some()
        });
    let exit_code = capture.status.code();
    let signal = child_exit_signal(&capture.status);
    let normal_zero_exit = capture.status.success()
        && exit_code == Some(0)
        && signal.is_none()
        && !capture.timed_out
        && capture.transport_error.is_none();
    let process = crate::greedy_parity::HybridWorkerProcessEvidence {
        worker_id: request.base.worker_id.clone(),
        child_process_spawned: true,
        process_id: Some(capture.process_id),
        executable_sha256: request.base.executable_sha256.clone(),
        build_git_sha: request.base.build_git_sha.clone(),
        executable_identity_verified: identity_attested,
        build_sha_identity_verified: identity_attested,
        case_identity_verified: identity_attested,
        config_identity_verified: identity_attested,
        expected_adapter_identity_verified: identity_attested,
        prompt_token_identity_verified: identity_attested,
        output_token_limit_verified: identity_attested,
        greedy_sampling_identity_verified: identity_attested,
        normal_zero_exit,
        exit_code,
        signal,
        process_reaped: capture.reaped,
        timed_out: capture.timed_out,
        evidence_emitted: response.is_some(),
    };
    let completed = (|| -> Result<CompletedLogitWorker, String> {
        if !normal_zero_exit || !capture.reaped || !identity_attested {
            return Err("logit worker did not exit/reap with exact identity".to_string());
        }
        let response = response.ok_or("logit worker emitted no response")?;
        let logit_bits = response
            .first_token_logit_bits
            .clone()
            .ok_or("logit worker omitted complete logits")?;
        let logit_sha256 = response
            .first_token_logit_bits_sha256
            .clone()
            .ok_or("logit worker omitted logit hash")?;
        if logit_bits.is_empty()
            || logit_bits.len() > crate::numerical_diagnostics::MAX_VOCAB_SIZE
            || crate::numerical_diagnostics::f32_bits_sha256(&logit_bits) != logit_sha256
        {
            return Err("logit worker vector is malformed or hash-mismatched".to_string());
        }
        let chosen_token_id = response
            .chosen_token_id
            .ok_or("logit worker omitted chosen token")?;
        let route_capture = response
            .route_capture
            .clone()
            .ok_or("logit worker omitted layer-0 route capture")?;
        let logits: Vec<f32> = logit_bits.iter().copied().map(f32::from_bits).collect();
        if crate::numerical_diagnostics::top_logits(&logits, 1)
            .first()
            .map(|item| item.token_id)
            != Some(chosen_token_id)
        {
            return Err("complete logits disagree with production greedy token".to_string());
        }
        let mut plane = response
            .base
            .plane
            .clone()
            .ok_or("logit worker omitted plane evidence")?;
        if plane.worker_process.is_some()
            || plane.generation.generated_token_ids.first().copied() != Some(chosen_token_id)
        {
            return Err("logit worker plane evidence is malformed".to_string());
        }
        plane.worker_process = Some(process.clone());
        let generated_token_ids = plane.generation.generated_token_ids.clone();
        Ok(CompletedLogitWorker {
            run: crate::numerical_diagnostics::RepeatedRunEvidence {
                plane: request.plane,
                run_index: request.run_index,
                worker_id: request.base.worker_id.clone(),
                generated_token_ids_sha256: crate::greedy_parity::token_ids_sha256(
                    &generated_token_ids,
                ),
                generated_token_ids,
                first_token_logit_bits_sha256: logit_sha256,
                process: process.clone(),
                plane_evidence: plane,
            },
            first_token_logit_bits: logit_bits,
            chosen_token_id,
            route_capture,
        })
    })();
    completed.map_err(|detail| {
        let response_failure = response.and_then(|response| {
            response
                .failure
                .clone()
                .or_else(|| response.base.failure.clone())
        });
        crate::numerical_diagnostics::DiagnosticWorkerFailureEvidence {
            worker_id: request.base.worker_id.clone(),
            plane: request.plane,
            run_index: request.run_index,
            detail: [
                Some(detail),
                capture.transport_error,
                parse_detail,
                response_failure,
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join("; "),
            process,
            stderr,
            stderr_truncated: capture.stderr.truncated,
        }
    })
}

fn emit_numerical_diagnostic_report(
    report: &crate::numerical_diagnostics::DiagnosticReport,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut bytes = serde_json::to_vec_pretty(report)?;
    bytes.push(b'\n');
    std::fs::write(path, bytes)?;
    eprintln!("logit diagnostic report written to {}", path.display());
    Ok(())
}

fn fail_numerical_diagnostic(
    mut report: crate::numerical_diagnostics::DiagnosticReport,
    code: &str,
    detail: impl Into<String>,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let detail = detail.into();
    report.fail(code, detail.clone());
    match emit_numerical_diagnostic_report(&report, path) {
        Ok(()) => Err(format!("{code}: {detail}").into()),
        Err(error) => Err(format!("{code}: {detail}; report emission failed: {error}").into()),
    }
}

async fn execute_actual_input_q4_shadow(
    spec: &ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    expected_adapter_name: &str,
    capture: &crate::engine::RoutedFfnDiagnosticCapture,
) -> Result<crate::numerical_diagnostics::ActualInputQ4ShadowEvidence, String> {
    let runtime = build_isolated_greedy_runtime(
        spec,
        RealCliRuntimeMode::IsolatedGreedyParityHybrid,
        tokenizer,
    )
    .await
    .map_err(|error| error.to_string())?;
    let attempt = async {
        let plan: crate::qualification::ExecutionPlanEvidence =
            runtime.engine.execution_context().plan().into();
        if !crate::greedy_parity::hybrid_plan_exact(&plan)
            || runtime.engine.routed_expert_gpu_failure_policy()
                != crate::engine::RoutedExpertGpuFailurePolicy::StrictFailClosed
        {
            return Err("Q4 shadow runtime did not retain the exact strict Hybrid plan".to_string());
        }
        let device = runtime
            .engine
            .gpu_device_identity()
            .ok_or("Q4 shadow runtime has no authoritative GPU identity")?;
        if device.software_adapter
            || device.device_type.eq_ignore_ascii_case("cpu")
            || device.name != expected_adapter_name
        {
            return Err(format!(
                "Q4 shadow selected adapter {:?}, expected exact hardware adapter {:?}",
                device.name, expected_adapter_name
            ));
        }
        let input: Vec<f32> = capture
            .input_bits
            .iter()
            .map(|bits| half::f16::from_f32(f32::from_bits(*bits)).to_f32())
            .collect();
        let inputs = vec![input.clone(), input];
        let mut outputs = Vec::with_capacity(capture.expert_ids.len());
        for &global_expert_id in &capture.expert_ids {
            let execution = runtime
                .engine
                .qualify_q4_0_complete_expert(global_expert_id, 0, &inputs)
                .await?;
            let dispatch = execution
                .dispatches
                .into_iter()
                .next()
                .ok_or("Q4 shadow expert returned no production dispatch")?;
            outputs.push(crate::numerical_diagnostics::Q4ShadowExpertOutput {
                global_expert_id,
                cpu_f32: dispatch.cpu_f32,
                gpu_f16: dispatch.gpu_f16,
            });
        }
        crate::numerical_diagnostics::build_actual_input_q4_shadow(capture, outputs)
    }
    .await;
    let shutdown = runtime.shutdown_isolated().await.map_err(|error| error.to_string());
    match (attempt, shutdown) {
        (Ok(evidence), Ok(_)) => Ok(evidence),
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(shutdown_error)) => {
            Err(format!("{error}; isolated shutdown also failed: {shutdown_error}"))
        }
    }
}

async fn cmd_diagnose_hybrid_q4_greedy_divergence(
    args: DiagnoseHybridQ4GreedyDivergenceArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::{
        validate_preflight, BuildProvenance, PreflightEvidence, QualificationChecks,
    };

    let cfg = args.parsed_config;
    let provenance = BuildProvenance::embedded();
    let build_git_sha = provenance
        .git_sha
        .clone()
        .filter(|sha| sha.len() == 40 && sha.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .ok_or("logit diagnostic requires embedded immutable build SHA")?;
    if provenance.dirty != Some(false) {
        return Err("logit diagnostic requires clean build provenance".into());
    }
    if args.expected_adapter_name.is_empty() {
        return Err("--expected-adapter-name must be non-empty".into());
    }
    let worker_timeout = args
        .progress_watchdog
        .timeout
        .ok_or("logit diagnostic requires a positive progress timeout")?;
    let (_, artifact_errors) = qualification_artifacts(&args.config, &cfg);
    if !artifact_errors.is_empty() {
        return Err(format!(
            "logit diagnostic artifact preflight failed: {}",
            artifact_errors.join("; ")
        )
        .into());
    }
    let metadata = crate::qualification::read_expert_metadata(
        &cfg.model.data_dir.join("metadata.json"),
    )
    .map_err(|error| format!("logit diagnostic metadata preflight failed: {error}"))?;
    if !matches!(
        cfg.real_transformer.resolve_weight_policy(),
        Ok(crate::config::RealWeightPolicy::StrictReal)
    ) {
        return Err("logit diagnostic requires the strict real-weight policy".into());
    }
    let capacity_bytes = cfg
        .gpu_cache
        .vram_capacity_mb
        .checked_mul(1024 * 1024)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .unwrap_or(0);
    let preflight = PreflightEvidence {
        provenance: provenance.clone(),
        real_transformer_enabled: cfg.real_transformer.enabled,
        weights_dir_configured: cfg.real_transformer.weights_dir.is_some(),
        strict_weights: cfg.real_transformer.strict_weights,
        allow_seeded_fallback: cfg.real_transformer.allow_seeded_fallback,
        allow_degraded_experts: cfg.real_transformer.allow_degraded_experts,
        allow_attention_fallback: cfg.real_transformer.allow_nonfinite_attention_fallback,
        allow_truncated_expert_payloads: cfg.real_transformer.allow_truncated_expert_payloads,
        distributed_enabled: cfg.distributed.enabled,
        gpu_cache_enabled: cfg.gpu_cache.enabled,
        gpu_expert_capacity_bytes: capacity_bytes,
        requested_mode: cfg.real_transformer.compute_offload,
        routed_expert_dtype: cfg.model.dtype,
        metadata,
    };
    validate_preflight(&preflight, &mut QualificationChecks::default())
        .map_err(|failure| format!("{}: {}", failure.code, failure.detail))?;
    let spec = resolve_real_cli_spec_from_config(
        cfg,
        RealCliRuntimeMode::IsolatedGreedyParityHybrid,
    )?;
    if !greedy_parity_model_identity(&spec).is_qwen3_coder_30b_a3b_q4_0() {
        return Err("logit diagnostic requires exact Qwen3-Coder 30B-A3B Q4_0 geometry".into());
    }
    let resolved_config_sha256 = resolved_real_cli_spec_sha256(&spec)?;
    let (worker_executable, executable_sha256) = current_executable_identity()?;
    let tokenizer = load_real_cli_tokenizer(
        &spec.cfg,
        RealCliRuntimeMode::IsolatedGreedyParityHybrid,
    )?;
    let fixed = crate::greedy_parity::fixed_case(
        crate::numerical_diagnostics::TARGET_CASE,
    )
    .ok_or("json-transformation fixed corpus case is unavailable")?;
    // Parent-only tokenization: every worker receives this identical vector.
    let prompt_token_ids = tokenizer.encode(fixed.prompt)?;
    if prompt_token_ids.is_empty() {
        return Err("json-transformation prompt encoded to zero tokens".into());
    }
    let prompt_sha256 = crate::greedy_parity::sha256_hex(fixed.prompt.as_bytes());
    let prompt_token_ids_sha256 = crate::greedy_parity::token_ids_sha256(&prompt_token_ids);
    let mut report = crate::numerical_diagnostics::DiagnosticReport::new(
        provenance,
        build_git_sha.clone(),
        executable_sha256.clone(),
        resolved_config_sha256.clone(),
        args.expected_adapter_name.clone(),
        prompt_sha256,
        prompt_token_ids_sha256,
        prompt_token_ids.len(),
    );
    let mut completed = Vec::with_capacity(6);
    for plane in [
        crate::numerical_diagnostics::DiagnosticPlane::Cpu,
        crate::numerical_diagnostics::DiagnosticPlane::CpuBoundaryEmulation,
        crate::numerical_diagnostics::DiagnosticPlane::Hybrid,
    ] {
        for run_index in 0..crate::numerical_diagnostics::REPEATED_RUNS_PER_PLANE {
            let worker_id = crate::numerical_diagnostics::diagnostic_worker_id(
                &build_git_sha,
                &executable_sha256,
                plane,
                run_index,
            );
            let base = crate::greedy_parity::HybridWorkerRequest::new(
                worker_id,
                fixed,
                resolved_config_sha256.clone(),
                args.expected_adapter_name.clone(),
                prompt_token_ids.clone(),
                executable_sha256.clone(),
                build_git_sha.clone(),
            );
            let request = crate::numerical_diagnostics::DiagnosticWorkerRequest::new(
                plane,
                run_index,
                base,
            );
            match execute_logit_diagnostic_worker(
                &worker_executable,
                &args.config,
                &request,
                worker_timeout,
            )
            .await
            {
                Ok(worker) => completed.push(worker),
                Err(failure) => {
                    let detail = failure.detail.clone();
                    report.runs = completed.iter().map(|worker| worker.run.clone()).collect();
                    report.reproducibility = Some(
                        crate::numerical_diagnostics::validate_repeated_run_identity(&report.runs),
                    );
                    report.worker_failures.push(failure);
                    return fail_numerical_diagnostic(
                        report,
                        "logit-diagnostic-worker-failed",
                        detail,
                        &args.report_out,
                    );
                }
            }
        }
    }
    report.runs = completed.iter().map(|worker| worker.run.clone()).collect();
    let reproducibility =
        crate::numerical_diagnostics::validate_repeated_run_identity(&report.runs);
    report.reproducibility = Some(reproducibility.clone());
    if !reproducibility.cpu_bitwise_reproducible
        || !reproducibility.cpu_boundary_emulation_bitwise_reproducible
        || !reproducibility.hybrid_bitwise_reproducible
        || !reproducibility.all_worker_ids_unique
        || !reproducibility.all_process_ids_unique
        || !reproducibility.every_worker_exited_zero_and_reaped
        || !reproducibility.no_retries
    {
        return fail_numerical_diagnostic(
            report,
            "logit-diagnostic-reproducibility-failed",
            "fresh CPU/boundary-emulation/Hybrid runs were nondeterministic or process evidence was incomplete",
            &args.report_out,
        );
    }
    let cpu = completed
        .iter()
        .find(|worker| worker.run.plane == crate::numerical_diagnostics::DiagnosticPlane::Cpu)
        .ok_or("missing CPU logit diagnostic run")?;
    let hybrid = completed
        .iter()
        .find(|worker| worker.run.plane == crate::numerical_diagnostics::DiagnosticPlane::Hybrid)
        .ok_or("missing Hybrid logit diagnostic run")?;
    let boundary = completed
        .iter()
        .find(|worker| {
            worker.run.plane
                == crate::numerical_diagnostics::DiagnosticPlane::CpuBoundaryEmulation
        })
        .ok_or("missing CPU boundary-emulation logit diagnostic run")?;
    let cpu_logits: Vec<f32> = cpu
        .first_token_logit_bits
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect();
    let hybrid_logits: Vec<f32> = hybrid
        .first_token_logit_bits
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect();
    let boundary_logits: Vec<f32> = boundary
        .first_token_logit_bits
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect();
    report.first_token_logits = Some(match
        crate::numerical_diagnostics::build_first_token_logit_evidence(
            &cpu_logits,
            &hybrid_logits,
            cpu.chosen_token_id,
            hybrid.chosen_token_id,
        )
    {
        Ok(evidence) => evidence,
        Err(error) => {
            return fail_numerical_diagnostic(
                report,
                "first-token-logit-comparison-failed",
                error,
                &args.report_out,
            )
        }
    });
    report.cpu_boundary_emulation = Some(match
        crate::numerical_diagnostics::build_boundary_plane_evidence(
            &cpu_logits,
            &hybrid_logits,
            &boundary_logits,
            cpu.chosen_token_id,
            hybrid.chosen_token_id,
            boundary.chosen_token_id,
            boundary.run.generated_token_ids_sha256.clone(),
        )
    {
        Ok(evidence) => evidence,
        Err(error) => {
            return fail_numerical_diagnostic(
                report,
                "cpu-boundary-emulation-comparison-failed",
                error,
                &args.report_out,
            )
        }
    });
    let route_captures: Vec<_> = completed
        .iter()
        .filter(|worker| {
            matches!(
                worker.run.plane,
                crate::numerical_diagnostics::DiagnosticPlane::Cpu
                    | crate::numerical_diagnostics::DiagnosticPlane::Hybrid
            )
        })
        .map(|worker| (worker.run.plane, worker.route_capture.clone()))
        .collect();
    let expected_token_idx = prompt_token_ids
        .len()
        .checked_sub(1)
        .and_then(|position| position.checked_mul(spec.cfg.model.num_layers))
        .and_then(|token_idx| u64::try_from(token_idx).ok())
        .ok_or("layer-0 route capture token index overflowed")?;
    let route_evidence = match crate::numerical_diagnostics::reconcile_route_captures(
        &route_captures,
        expected_token_idx,
    ) {
        Ok(evidence) => evidence,
        Err(error) => {
            return fail_numerical_diagnostic(
                report,
                "layer0-route-capture-invalid",
                error,
                &args.report_out,
            )
        }
    };
    let route_matches = route_evidence.exact_capture_match;
    report.route_capture = Some(route_evidence);
    if !route_matches {
        return fail_numerical_diagnostic(
            report,
            "cpu-hybrid-layer0-route-mismatch",
            "CPU/Hybrid layer-0 input, selected experts, or routing-weight bits differ; Q4 shadow dispatch was not performed",
            &args.report_out,
        );
    }
    let shadow_capture = &route_captures[0].1;
    report.actual_input_q4_shadow = Some(
        match execute_actual_input_q4_shadow(
            &spec,
            tokenizer,
            &args.expected_adapter_name,
            shadow_capture,
        )
        .await
        {
            Ok(evidence) => evidence,
            Err(error) => {
                return fail_numerical_diagnostic(
                    report,
                    "actual-input-q4-shadow-failed",
                    error,
                    &args.report_out,
                )
            }
        },
    );
    if let Err(error) = report.finish() {
        return fail_numerical_diagnostic(
            report,
            "logit-diagnostic-evidence-incomplete",
            error,
            &args.report_out,
        );
    }
    emit_numerical_diagnostic_report(&report, &args.report_out)
}

fn emit_gpu_native_first_divergence_report(
    report: &crate::gpu_native_diagnostics::GpuNativeFirstDivergenceReport,
    out_path: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(report)?;
    if let Some(path) = out_path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, json)?;
        info!(
            report = %path.display(),
            "GPU-native first-divergence diagnostic report written"
        );
    } else {
        println!("{json}");
    }
    if let Some(ref failure) = report.failure {
        Err(failure.clone().into())
    } else {
        Ok(())
    }
}

fn fail_gpu_native_first_divergence(
    mut report: crate::gpu_native_diagnostics::GpuNativeFirstDivergenceReport,
    reason: String,
    out_path: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    report.fail(reason);
    emit_gpu_native_first_divergence_report(&report, out_path)
}

async fn cmd_diagnose_gpu_native_q4_first_divergence(
    args: DiagnoseGpuNativeQ4FirstDivergenceArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::BuildProvenance;

    let provenance = BuildProvenance::embedded();
    let build_git_sha = provenance
        .git_sha
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let (_, executable_sha256) = current_executable_identity()?;

    let cfg = args.parsed_config.clone();
    let metadata_path = cfg.model.data_dir.join("metadata.json");
    let expert_metadata = crate::qualification::read_expert_metadata(&metadata_path)
        .map_err(|e| format!("failed to read expert metadata: {e}"))?;

    let fixed_case = crate::greedy_parity::fixed_case(&args.case).ok_or_else(|| {
        format!(
            "unknown corpus case: {}; expected one of: rust-generation, rust-debugging, json-transformation, multilingual-spanish",
            args.case
        )
    })?;

    let prompt_sha256 = crate::greedy_parity::sha256_hex(fixed_case.prompt.as_bytes());

    let mut gpu_spec = resolve_real_cli_spec_from_config(
        cfg.clone(),
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
    )?;
    gpu_spec.cfg.real_transformer.gpu_native = true;
    gpu_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Gpu;
    gpu_spec.cfg.real_transformer.strict_weights = true;
    gpu_spec.cfg.real_transformer.allow_degraded_experts = false;
    gpu_spec.cfg.real_transformer.allow_seeded_fallback = false;

    let resolved_config_sha256 = resolved_real_cli_spec_sha256(&gpu_spec)?;

    // 1. Reference Phase (CPU with Hybrid boundary emulation)
    let mut ref_spec = resolve_real_cli_spec_from_config(
        cfg.clone(),
        RealCliRuntimeMode::IsolatedGreedyParityCpu,
    )?;
    ref_spec.cfg.real_transformer.gpu_native = false;
    ref_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Cpu;
    ref_spec.cfg.real_transformer.strict_weights = true;
    ref_spec.cfg.real_transformer.allow_degraded_experts = false;
    ref_spec.cfg.real_transformer.allow_seeded_fallback = false;

    let ref_context =
        resolve_isolated_real_cli_context(&ref_spec, crate::backend::ComputeOffload::Cpu)?;

    let ref_runtime = build_real_cli_runtime_from_spec(
        &ref_spec,
        RealCliRuntimeMode::IsolatedGreedyParityCpu,
        ref_context,
        None,
    )
    .await?;

    ref_runtime.engine.enable_cpu_q4_boundary_emulation()?;

    let prompt_token_ids = ref_runtime.tokenizer.encode(fixed_case.prompt)?;
    if prompt_token_ids.is_empty() {
        return Err("prompt tokenized to zero tokens".into());
    }

    let mut ref_kv = ref_runtime.model.fresh_kv_caches();
    let final_pos = prompt_token_ids.len() - 1;
    let final_tid = prompt_token_ids[final_pos];

    for (pos, &tid) in prompt_token_ids[..prompt_token_ids.len() - 1]
        .iter()
        .enumerate()
    {
        ref_runtime
            .model
            .forward_token_hidden(&ref_runtime.engine, tid, pos, &mut ref_kv)
            .await?;
    }

    let ref_trace = ref_runtime
        .model
        .forward_token_diagnostic_trace(&ref_runtime.engine, final_tid, final_pos, &mut ref_kv, None)
        .await?;

    let _ = ref_runtime.shutdown_isolated().await;

    // 2. GPU-Native Phase
    let gpu_context =
        resolve_isolated_real_cli_context(&gpu_spec, crate::backend::ComputeOffload::Gpu)?;

    let gpu_runtime = build_real_cli_runtime_from_spec(
        &gpu_spec,
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
        gpu_context,
        None,
    )
    .await?;

    let device = gpu_runtime
        .engine
        .execution_context()
        .gpu_device_identity()
        .ok_or("GPU execution context has no active device identity")?;

    let gpu_native_token_loop = gpu_runtime
        .gpu_native_token_loop
        .clone()
        .ok_or("GPU-native token loop was not initialized on GPU runtime")?;

    let model_geometry = gpu_native_token_loop.model_geometry();

    let case_evidence = crate::gpu_native_diagnostics::CaseEvidence {
        case_name: fixed_case.name.to_string(),
        prompt_sha256,
        prompt_token_count: prompt_token_ids.len(),
    };

    let mut report = crate::gpu_native_diagnostics::GpuNativeFirstDivergenceReport::new(
        provenance,
        build_git_sha,
        executable_sha256,
        resolved_config_sha256.clone(),
        model_geometry,
        expert_metadata,
        case_evidence,
        crate::greedy_parity::token_ids_sha256(&prompt_token_ids),
        args.expected_adapter_name.clone(),
    );

    if device.name != args.expected_adapter_name {
        return fail_gpu_native_first_divergence(
            report,
            format!(
                "adapter mismatch: runtime has {:?}, expected {:?}",
                device.name, args.expected_adapter_name
            ),
            args.report_out.as_deref(),
        );
    }
    report.actual_adapter = Some(device);

    let mut req_state = match gpu_native_token_loop.create_request_state() {
        Ok(state) => state,
        Err(e) => {
            return fail_gpu_native_first_divergence(
                report,
                format!("failed to create request state: {e}"),
                args.report_out.as_deref(),
            );
        }
    };

    for (pos, &tid) in prompt_token_ids[..prompt_token_ids.len() - 1]
        .iter()
        .enumerate()
    {
        if let Err(e) = gpu_native_token_loop
            .step_token(&gpu_runtime.engine, &mut req_state, tid, pos, false)
            .await
        {
            return fail_gpu_native_first_divergence(
                report,
                format!("prefix prompt step failed at position {pos} (token {tid}): {e}"),
                args.report_out.as_deref(),
            );
        }
    }

    let trace_layout = match crate::gpu_native_diagnostics::GpuNativeDiagnosticTraceLayout::try_new(
        model_geometry.num_layers,
        model_geometry.d_model,
        model_geometry.top_k,
        model_geometry.vocab_size,
    ) {
        Ok(layout) => layout,
        Err(e) => {
            return fail_gpu_native_first_divergence(
                report,
                format!("failed to construct trace layout: {e}"),
                args.report_out.as_deref(),
            );
        }
    };

    let staging_buffer =
        match gpu_native_token_loop.create_diagnostic_staging_buffer(&trace_layout) {
            Ok(buf) => buf,
            Err(e) => {
                return fail_gpu_native_first_divergence(
                    report,
                    format!("failed to create diagnostic staging buffer: {e}"),
                    args.report_out.as_deref(),
                );
            }
        };

    let (gpu_trace, attempts) = match gpu_native_token_loop
        .step_token_diagnostic(
            &gpu_runtime.engine,
            &mut req_state,
            final_tid,
            final_pos,
            true,
            &trace_layout,
            &staging_buffer,
        )
        .await
    {
        Ok(res) => res,
        Err(e) => {
            return fail_gpu_native_first_divergence(
                report,
                format!(
                    "diagnostic step failed at final position {final_pos} (token {final_tid}): {e}"
                ),
                args.report_out.as_deref(),
            );
        }
    };

    let counters_delta = gpu_native_token_loop.snapshot();
    drop(req_state);
    drop(staging_buffer);
    let _ = gpu_runtime.shutdown_isolated().await;

    // 3. Comparison
    let (first_exact_divergence, first_divergence, stages) =
        match crate::gpu_native_diagnostics::compare_diagnostic_traces(&ref_trace, &gpu_trace) {
            Ok(res) => res,
            Err(e) => {
                return fail_gpu_native_first_divergence(
                    report,
                    format!("failed to compare diagnostic traces: {e}"),
                    args.report_out.as_deref(),
                );
            }
        };

    report.reference_sampled_token_id = Some(ref_trace.sampled_token);
    report.gpu_native_sampled_token_id = Some(gpu_trace.sampled_token);
    report.token_match = Some(ref_trace.sampled_token == gpu_trace.sampled_token);
    report.first_exact_divergence = first_exact_divergence;
    report.first_divergence = first_divergence;
    report.stages = stages;
    report.token_loop_counters_delta = counters_delta;
    report.attempt_count = attempts;
    report.diagnostic_complete = true;
    report.qualification_pass = false;

    emit_gpu_native_first_divergence_report(&report, args.report_out.as_deref())
}

fn emit_layer0_attention_first_divergence_report(
    report: &crate::gpu_native_layer0_diagnostics::Layer0AttentionFirstDivergenceReport,
    out_path: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(report)?;
    if let Some(path) = out_path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, json)?;
        info!(
            report = %path.display(),
            "GPU-native layer-0 attention first-divergence diagnostic report written"
        );
    } else {
        println!("{json}");
    }
    if let Some(ref failure) = report.failure {
        Err(failure.clone().into())
    } else {
        Ok(())
    }
}

fn fail_layer0_attention_first_divergence(
    mut report: crate::gpu_native_layer0_diagnostics::Layer0AttentionFirstDivergenceReport,
    reason: String,
    out_path: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    report.fail(reason);
    emit_layer0_attention_first_divergence_report(&report, out_path)
}

async fn cmd_diagnose_gpu_native_layer0_attention_first_divergence(
    args: DiagnoseGpuNativeLayer0AttentionFirstDivergenceArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::BuildProvenance;

    let provenance = BuildProvenance::embedded();
    let build_git_sha = provenance
        .git_sha
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let (_, executable_sha256) = current_executable_identity()?;

    let cfg = args.parsed_config.clone();
    let metadata_path = cfg.model.data_dir.join("metadata.json");
    let expert_metadata = crate::qualification::read_expert_metadata(&metadata_path)
        .map_err(|e| format!("failed to read expert metadata: {e}"))?;

    let fixed_case = crate::greedy_parity::fixed_case(&args.case).ok_or_else(|| {
        format!(
            "unknown corpus case: {}; expected one of: rust-generation, rust-debugging, json-transformation, multilingual-spanish",
            args.case
        )
    })?;

    let prompt_sha256 = crate::greedy_parity::sha256_hex(fixed_case.prompt.as_bytes());

    let mut gpu_spec = resolve_real_cli_spec_from_config(
        cfg.clone(),
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
    )?;
    gpu_spec.cfg.real_transformer.gpu_native = true;
    gpu_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Gpu;
    gpu_spec.cfg.real_transformer.strict_weights = true;
    gpu_spec.cfg.real_transformer.allow_degraded_experts = false;
    gpu_spec.cfg.real_transformer.allow_seeded_fallback = false;

    let resolved_config_sha256 = resolved_real_cli_spec_sha256(&gpu_spec)?;

    // 1. Reference Phase (CPU with strict weights)
    let mut ref_spec = resolve_real_cli_spec_from_config(
        cfg.clone(),
        RealCliRuntimeMode::IsolatedGreedyParityCpu,
    )?;
    ref_spec.cfg.real_transformer.gpu_native = false;
    ref_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Cpu;
    ref_spec.cfg.real_transformer.strict_weights = true;
    ref_spec.cfg.real_transformer.allow_degraded_experts = false;
    ref_spec.cfg.real_transformer.allow_seeded_fallback = false;

    let ref_context =
        resolve_isolated_real_cli_context(&ref_spec, crate::backend::ComputeOffload::Cpu)?;

    let ref_runtime = build_real_cli_runtime_from_spec(
        &ref_spec,
        RealCliRuntimeMode::IsolatedGreedyParityCpu,
        ref_context,
        None,
    )
    .await?;

    let prompt_token_ids = ref_runtime.tokenizer.encode(fixed_case.prompt)?;
    if prompt_token_ids.is_empty() {
        return Err("prompt tokenized to zero tokens".into());
    }

    // 2. GPU-Native Phase
    let gpu_context =
        resolve_isolated_real_cli_context(&gpu_spec, crate::backend::ComputeOffload::Gpu)?;

    let gpu_runtime = build_real_cli_runtime_from_spec(
        &gpu_spec,
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
        gpu_context,
        None,
    )
    .await?;

    let device = gpu_runtime
        .engine
        .execution_context()
        .gpu_device_identity()
        .ok_or("GPU execution context has no active device identity")?;

    let gpu_native_token_loop = gpu_runtime
        .gpu_native_token_loop
        .clone()
        .ok_or("GPU-native token loop was not initialized on GPU runtime")?;

    let model_geometry = gpu_native_token_loop.model_geometry();

    let case_evidence = crate::gpu_native_diagnostics::CaseEvidence {
        case_name: fixed_case.name.to_string(),
        prompt_sha256,
        prompt_token_count: prompt_token_ids.len(),
    };

    let mut report = crate::gpu_native_layer0_diagnostics::Layer0AttentionFirstDivergenceReport::new(
        provenance,
        build_git_sha,
        executable_sha256,
        resolved_config_sha256.clone(),
        model_geometry,
        expert_metadata,
        case_evidence,
        crate::greedy_parity::token_ids_sha256(&prompt_token_ids),
        prompt_token_ids.clone(),
        args.expected_adapter_name.clone(),
    );

    if device.name != args.expected_adapter_name {
        return fail_layer0_attention_first_divergence(
            report,
            format!(
                "adapter mismatch: runtime has {:?}, expected {:?}",
                device.name, args.expected_adapter_name
            ),
            args.report_out.as_deref(),
        );
    }
    report.actual_adapter = Some(device);

    let q_width = model_geometry.num_heads * model_geometry.head_dim;
    let kv_width = model_geometry.num_kv_heads * model_geometry.head_dim;

    let trace_layout = match crate::gpu_native_layer0_diagnostics::Layer0AttentionDiagnosticTraceLayout::try_new(
        model_geometry.d_model,
        q_width,
        kv_width,
    ) {
        Ok(layout) => layout,
        Err(e) => {
            return fail_layer0_attention_first_divergence(
                report,
                format!("failed to construct Layer0 trace layout: {e}"),
                args.report_out.as_deref(),
            );
        }
    };

    let staging_buffer = match gpu_native_token_loop
        .create_layer0_diagnostic_staging_buffer(&trace_layout)
    {
        Ok(buf) => buf,
        Err(e) => {
            return fail_layer0_attention_first_divergence(
                report,
                format!("failed to create Layer0 diagnostic staging buffer: {e}"),
                args.report_out.as_deref(),
            );
        }
    };

    let mut req_state = match gpu_native_token_loop.create_request_state() {
        Ok(state) => state,
        Err(e) => {
            return fail_layer0_attention_first_divergence(
                report,
                format!("failed to create request state: {e}"),
                args.report_out.as_deref(),
            );
        }
    };

    if ref_runtime.model.layers.is_empty() {
        return fail_layer0_attention_first_divergence(
            report,
            "reference model has zero layers".to_string(),
            args.report_out.as_deref(),
        );
    }

    let ref_layer0 = &ref_runtime.model.layers[0];
    let mut ref_kv = crate::transformer::KvCache::new(ref_layer0.attn.kv_dim());
    let ref_backend = ref_runtime.engine.execution_context().attention_backend();

    let mut position_comparisons = Vec::with_capacity(prompt_token_ids.len());
    let mut final_position_post_attention: Option<crate::gpu_native_layer0_diagnostics::Layer0AttentionStageComparison> = None;

    let mut ref_scratch = crate::transformer::TransformerLayerScratch::new();
    let mut ref_out = Vec::with_capacity(model_geometry.d_model);

    for (pos, &tid) in prompt_token_ids.iter().enumerate() {
        // 1. CPU Reference Execution for Layer 0
        let emb = match ref_runtime.model.try_embed(tid) {
            Ok(v) => v,
            Err(e) => {
                return fail_layer0_attention_first_divergence(
                    report,
                    format!("CPU reference embedding failed at pos {pos} (token {tid}): {e}"),
                    args.report_out.as_deref(),
                );
            }
        };

        let mut cpu_sink = crate::transformer::Layer0AttentionCpuDiagnosticSink {
            embedding: Some(emb.clone()),
            ..Default::default()
        };

        ref_out.clear();
        if let Err(e) = ref_layer0.try_attn_block_into_layer0_diagnostic(
            &emb,
            pos,
            0,
            &mut ref_kv,
            &**ref_backend,
            &mut ref_scratch,
            &mut ref_out,
            None,
            &mut cpu_sink,
        ) {
            return fail_layer0_attention_first_divergence(
                report,
                format!("CPU reference layer-0 attention failed at pos {pos} (token {tid}): {e:?}"),
                args.report_out.as_deref(),
            );
        }

        // 2. GPU-Native Execution for Layer 0
        let gpu_trace = match gpu_native_token_loop
            .step_layer0_attention_diagnostic(
                &gpu_runtime.engine,
                &mut req_state,
                tid,
                pos,
                &trace_layout,
                &staging_buffer,
            )
            .await
        {
            Ok(t) => t,
            Err(e) => {
                return fail_layer0_attention_first_divergence(
                    report,
                    format!("GPU layer-0 diagnostic step failed at pos {pos} (token {tid}): {e}"),
                    args.report_out.as_deref(),
                );
            }
        };

        // 3. Comparison
        let pos_comp = match crate::gpu_native_layer0_diagnostics::compare_layer0_attention_traces(
            pos,
            tid,
            &cpu_sink,
            &gpu_trace,
        ) {
            Ok(c) => c,
            Err(e) => {
                return fail_layer0_attention_first_divergence(
                    report,
                    format!("comparison failed at pos {pos} (token {tid}): {e}"),
                    args.report_out.as_deref(),
                );
            }
        };

        if pos == prompt_token_ids.len() - 1 {
            final_position_post_attention = pos_comp
                .stages
                .iter()
                .find(|s| s.stage == crate::gpu_native_layer0_diagnostics::Layer0AttentionStage::PostAttentionResidual)
                .cloned();
        }

        position_comparisons.push(pos_comp);
    }

    drop(req_state);
    drop(staging_buffer);
    let _ = ref_runtime.shutdown_isolated().await;
    let _ = gpu_runtime.shutdown_isolated().await;

    report.record_completed_position_evidence(position_comparisons, final_position_post_attention);

    // 4. Optional Slice 11B Cross-Check
    let slice11b_cross_check = if let Some(ref path) = args.slice11b_report {
        if report.final_position_post_attention.is_none() {
            return fail_layer0_attention_first_divergence(
                report,
                "no final position post-attention evidence recorded".to_string(),
                args.report_out.as_deref(),
            );
        }
        let final_stage = report
            .final_position_post_attention
            .as_ref()
            .expect("final position post-attention evidence checked above");
        let check = match crate::gpu_native_layer0_diagnostics::cross_check_slice11b_report(
            path,
            &report,
            final_stage,
        ) {
            Ok(c) => c,
            Err(e) => {
                return fail_layer0_attention_first_divergence(
                    report,
                    format!("Slice 11B report cross-check failed: {e}"),
                    args.report_out.as_deref(),
                );
            }
        };
        if check.cross_check_status != "verified_match" {
            let disc = check.discrepancy.clone().unwrap_or_default();
            report.slice11b_cross_check = Some(check);
            return fail_layer0_attention_first_divergence(
                report,
                format!("Slice 11B layer-0 replay cross-check mismatch: {disc}"),
                args.report_out.as_deref(),
            );
        }
        Some(check)
    } else {
        None
    };

    report.slice11b_cross_check = slice11b_cross_check;
    report.diagnostic_complete = true;
    report.qualification_pass = false;

    emit_layer0_attention_first_divergence_report(&report, args.report_out.as_deref())
}

struct ReferenceRoutingCapture {
    generation: crate::greedy_parity::GenerationEvidence,
    routes: Vec<Vec<Vec<u32>>>,
    background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
}

struct GpuNativeQualificationCapture {
    evidence: crate::gpu_native_greedy_parity::GpuCandidateCaseEvidence,
    device: crate::backend::GpuDeviceIdentity,
    model_geometry: crate::gpu_native_token_loop::GpuNativeModelGeometry,
}

struct RouterRankReferenceCapture {
    generated_token_ids: Vec<u32>,
    target_trace: crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    gate: crate::gating::LinearGate,
    gate_identity: crate::gpu_native_router_rank_diagnostics::GateTensorIdentity,
    same_input_routed_moe:
        Option<crate::gpu_native_expert_permutation_semantic_parity::CpuSameInputRoutedMoeCapture>,
    model_load: crate::greedy_parity::ModelLoadEvidence,
    background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
}

struct RouterRankGpuCapture {
    generated_token_ids: Vec<u32>,
    target_trace: crate::gpu_native_router_rank_diagnostics::RouterRankGpuTrace,
    gate_identity: crate::gpu_native_router_rank_diagnostics::GateTensorIdentity,
    model_load: crate::greedy_parity::ModelLoadEvidence,
    device: crate::backend::GpuDeviceIdentity,
    model_geometry: crate::gpu_native_token_loop::GpuNativeModelGeometry,
    background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
}

struct SemanticGpuCapture {
    generated_token_ids: Vec<u32>,
    target_trace: crate::gpu_native_expert_permutation_semantic_parity::SemanticGpuTrace,
    gate_identity: crate::gpu_native_router_rank_diagnostics::GateTensorIdentity,
    model_load: crate::greedy_parity::ModelLoadEvidence,
    device: crate::backend::GpuDeviceIdentity,
    model_geometry: crate::gpu_native_token_loop::GpuNativeModelGeometry,
    background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
}

fn emit_gpu_native_greedy_parity_report(
    report: &crate::gpu_native_greedy_parity::GpuNativeGreedyParityReport,
    report_out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    if let Some(path) = report_out {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, json)?;
        eprintln!(
            "GPU-native greedy-token parity report written to {}",
            path.display()
        );
    } else {
        use std::io::Write as _;
        std::io::stdout().write_all(&json)?;
    }
    Ok(())
}

fn fail_gpu_native_greedy_parity(
    mut report: crate::gpu_native_greedy_parity::GpuNativeGreedyParityReport,
    failure: crate::qualification::QualificationFailure,
    report_out: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let summary = format!("{}: {}", failure.code, failure.detail);
    report.fail(failure);
    match emit_gpu_native_greedy_parity_report(&report, report_out) {
        Ok(()) => Err(summary.into()),
        Err(emit_error) => Err(format!(
            "{summary}; GPU-native greedy-parity report emission failed: {emit_error}"
        )
        .into()),
    }
}

fn gpu_native_case_execution_failure(
    default_code: &'static str,
    case_name: &str,
    error: &(dyn std::error::Error + 'static),
) -> crate::qualification::QualificationFailure {
    let isolated_shutdown = error.is::<IsolatedRuntimeShutdownError>();
    crate::qualification::QualificationFailure::new(
        if isolated_shutdown {
            crate::qualification::FailureStage::Postcondition
        } else {
            crate::qualification::FailureStage::Inference
        },
        if isolated_shutdown {
            "isolated-runtime-shutdown-failed"
        } else {
            default_code
        },
        format!("fixed case {case_name}: {error}"),
    )
}

fn gpu_native_generation_evidence(
    tokenizer: &crate::tokenizer::Tokenizer,
    prompt_token_ids_sha256: &str,
    generated_token_ids: Vec<u32>,
) -> Result<crate::greedy_parity::GenerationEvidence, Box<dyn std::error::Error>> {
    let generated_text = tokenizer.decode(&generated_token_ids)?;
    Ok(crate::greedy_parity::GenerationEvidence {
        prompt_token_ids_sha256: prompt_token_ids_sha256.to_string(),
        generated_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&generated_token_ids),
        generated_text_sha256: crate::greedy_parity::sha256_hex(generated_text.as_bytes()),
        generated_token_count: generated_token_ids.len(),
        generated_token_ids,
        termination_reason: crate::greedy_parity::TerminationReason::LengthLimit,
    })
}

async fn execute_gpu_native_reference_routing_replay(
    spec: &ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    prompt_token_ids: &[u32],
    prompt_token_ids_sha256: &str,
    resolved_config_sha256: &str,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
    case_name: &str,
) -> Result<ReferenceRoutingCapture, Box<dyn std::error::Error>> {
    let runtime = build_isolated_greedy_runtime(
        spec,
        RealCliRuntimeMode::IsolatedGreedyParityCpu,
        tokenizer.clone(),
    )
    .await?;
    let attempt = async {
        runtime.engine.enable_cpu_q4_boundary_emulation()?;
        let observed_config_sha256 = resolved_real_runtime_identity_sha256(
            &runtime.cfg,
            runtime.model.config.architecture,
            runtime.model.config.first_k_dense_replace,
            &runtime.model.config.advanced,
        )?;
        if observed_config_sha256 != resolved_config_sha256 {
            return Err(format!(
                "reference routing replay identity {observed_config_sha256} drifted from {resolved_config_sha256}"
            )
            .into());
        }
        let boundary_before = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        if !boundary_before.enabled || boundary_before.routed_expert_dispatches != 0 {
            return Err("reference routing replay boundary emulation did not start clean".into());
        }
        let routed_before = runtime.engine.routed_expert_execution_snapshot();
        let attention_fallbacks_before = crate::transformer::nonfinite_softmax_fallbacks();
        let (generated_token_ids, routes) = with_progress_timeout(
            format!("GPU-native greedy parity {case_name} reference routing replay"),
            watchdog,
            async {
                let mut kv = runtime.model.fresh_kv_caches();
                let prefix_count = prompt_token_ids.len().saturating_sub(1);
                for (position, &token_id) in prompt_token_ids[..prefix_count].iter().enumerate() {
                    runtime
                        .model
                        .forward_token_hidden(&runtime.engine, token_id, position, &mut kv)
                        .await?;
                }
                let mut input_token = *prompt_token_ids
                    .last()
                    .ok_or("reference routing replay prompt is empty")?;
                let mut position = prefix_count;
                let mut generated_token_ids =
                    Vec::with_capacity(crate::greedy_parity::OUTPUT_TOKEN_LIMIT);
                let mut routes = Vec::with_capacity(crate::greedy_parity::OUTPUT_TOKEN_LIMIT);
                while generated_token_ids.len() < crate::greedy_parity::OUTPUT_TOKEN_LIMIT {
                    let trace = runtime
                        .model
                        .forward_token_diagnostic_trace(
                            &runtime.engine,
                            input_token,
                            position,
                            &mut kv,
                            None,
                        )
                        .await?;
                    input_token = trace.sampled_token;
                    generated_token_ids.push(trace.sampled_token);
                    routes.push(trace.layer_selected_ids);
                    position = position
                        .checked_add(1)
                        .ok_or("reference routing replay position overflowed")?;
                }
                Ok::<_, Box<dyn std::error::Error>>((generated_token_ids, routes))
            },
        )
        .await?;
        let boundary_after = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        if !boundary_after.enabled || boundary_after.routed_expert_dispatches == 0 {
            return Err("reference routing replay did not exercise the Hybrid F16 boundary".into());
        }
        let routed_delta = crate::qualification::routed_execution_delta(
            routed_before,
            runtime.engine.routed_expert_execution_snapshot(),
        )
        .map_err(|failure| failure.detail)?;
        if routed_delta.degraded_expert_substitutions != 0 {
            return Err("reference routing replay recorded degraded expert execution".into());
        }
        let attention_fallbacks = crate::transformer::nonfinite_softmax_fallbacks()
            .saturating_sub(attention_fallbacks_before);
        if attention_fallbacks != 0 {
            return Err(format!(
                "reference routing replay recorded {attention_fallbacks} nonfinite attention fallbacks"
            )
            .into());
        }
        let generation = gpu_native_generation_evidence(
            &tokenizer,
            prompt_token_ids_sha256,
            generated_token_ids,
        )?;
        Ok::<_, Box<dyn std::error::Error>>(ReferenceRoutingCapture {
            generation,
            routes,
            background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence::default(),
        })
    }
    .await;
    let shutdown = runtime.shutdown_isolated().await;
    match (attempt, shutdown) {
        (Ok(mut capture), Ok(shutdown)) => {
            capture.background_shutdown = shutdown;
            Ok(capture)
        }
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error.into()),
        (Err(error), Err(shutdown_error)) => Err(IsolatedRuntimeShutdownError::new(format!(
            "{error}; reference routing replay shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

async fn execute_gpu_native_qualification_case(
    spec: &ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    prompt_token_ids: &[u32],
    prompt_token_ids_sha256: &str,
    resolved_config_sha256: &str,
    expected_adapter_name: &str,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
    case_name: &str,
) -> Result<GpuNativeQualificationCapture, Box<dyn std::error::Error>> {
    let runtime = build_isolated_greedy_runtime(
        spec,
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
        tokenizer.clone(),
    )
    .await?;
    let attempt = async {
        let observed_config_sha256 = resolved_real_runtime_identity_sha256(
            &runtime.cfg,
            runtime.model.config.architecture,
            runtime.model.config.first_k_dense_replace,
            &runtime.model.config.advanced,
        )?;
        if observed_config_sha256 != resolved_config_sha256 {
            return Err(format!(
                "GPU-native runtime identity {observed_config_sha256} drifted from {resolved_config_sha256}"
            )
            .into());
        }
        let device = runtime
            .engine
            .gpu_device_identity()
            .ok_or("GPU-native runtime has no authoritative adapter identity")?;
        if device.name != expected_adapter_name {
            return Err(format!(
                "GPU-native runtime selected adapter {:?}, expected exact name {:?}",
                device.name, expected_adapter_name
            )
            .into());
        }
        if device.software_adapter || device.device_type.eq_ignore_ascii_case("cpu") {
            return Err(format!(
                "GPU-native runtime selected software adapter {:?}",
                device.name
            )
            .into());
        }
        let token_loop = runtime
            .gpu_native_token_loop
            .as_ref()
            .ok_or("GPU-native token loop was not initialized")?;
        let model_geometry = token_loop.model_geometry();
        let counters_before = token_loop.snapshot();
        if counters_before != crate::gpu_native_token_loop::GpuNativeTokenLoopSnapshot::default() {
            return Err("GPU-native token-loop counters did not start at zero".into());
        }
        let routed_before = runtime.engine.routed_expert_execution_snapshot();
        let attention_fallbacks_before = crate::transformer::nonfinite_softmax_fallbacks();
        let mut request = token_loop.create_request_state()?;
        let trace_layout =
            crate::gpu_native_diagnostics::GpuNativeDiagnosticTraceLayout::try_new(
                model_geometry.num_layers,
                model_geometry.d_model,
                model_geometry.top_k,
                model_geometry.vocab_size,
            )?;
        let staging = token_loop.create_diagnostic_staging_buffer(&trace_layout)?;
        let (generated_token_ids, routes) = with_progress_timeout(
            format!("GPU-native greedy parity {case_name} candidate"),
            watchdog,
            async {
                let prefix_count = prompt_token_ids.len().saturating_sub(1);
                for (position, &token_id) in prompt_token_ids[..prefix_count].iter().enumerate() {
                    token_loop
                        .step_token(&runtime.engine, &mut request, token_id, position, false)
                        .await?;
                }
                let mut input_token = *prompt_token_ids
                    .last()
                    .ok_or("GPU-native qualification prompt is empty")?;
                let mut position = prefix_count;
                let mut generated_token_ids =
                    Vec::with_capacity(crate::greedy_parity::OUTPUT_TOKEN_LIMIT);
                let mut routes = Vec::with_capacity(crate::greedy_parity::OUTPUT_TOKEN_LIMIT);
                while generated_token_ids.len() < crate::greedy_parity::OUTPUT_TOKEN_LIMIT {
                    let (trace, _) = token_loop
                        .step_token_diagnostic(
                            &runtime.engine,
                            &mut request,
                            input_token,
                            position,
                            true,
                            &trace_layout,
                            &staging,
                        )
                        .await?;
                    if trace.final_status != 0
                        || trace.layer_statuses.iter().any(|&status| status != 0)
                    {
                        return Err(format!(
                            "GPU-native diagnostic returned nonzero numerical status at generated position {}",
                            generated_token_ids.len()
                        )
                        .into());
                    }
                    input_token = trace.sampled_token;
                    generated_token_ids.push(trace.sampled_token);
                    routes.push(trace.layer_selected_ids);
                    position = position
                        .checked_add(1)
                        .ok_or("GPU-native qualification position overflowed")?;
                }
                Ok::<_, Box<dyn std::error::Error>>((generated_token_ids, routes))
            },
        )
        .await?;
        let expected_completed = prompt_token_ids
            .len()
            .saturating_sub(1)
            .checked_add(crate::greedy_parity::OUTPUT_TOKEN_LIMIT)
            .ok_or("GPU-native expected completion count overflowed")?;
        let request_completed_expected_tokens = request.committed_position() == expected_completed;
        if !request_completed_expected_tokens {
            return Err(format!(
                "GPU-native request retired early at committed position {}, expected {expected_completed}",
                request.committed_position()
            )
            .into());
        }
        let token_loop_counters_delta = token_loop.snapshot();
        if token_loop_counters_delta.tokens_completed != expected_completed as u64 {
            return Err(format!(
                "GPU-native token counter recorded {} completions, expected {expected_completed}",
                token_loop_counters_delta.tokens_completed
            )
            .into());
        }
        if token_loop_counters_delta.fatal_failures != 0
            || token_loop_counters_delta.no_progress_failures != 0
        {
            return Err("GPU-native token loop recorded fatal or NoProgress failures".into());
        }
        let routed_execution_delta = crate::qualification::routed_execution_delta(
            routed_before,
            runtime.engine.routed_expert_execution_snapshot(),
        )
        .map_err(|failure| failure.detail)?;
        if routed_execution_delta.degraded_expert_substitutions != 0
            || routed_execution_delta.gpu_cpu_fallbacks != 0
        {
            return Err("GPU-native runtime recorded degraded or CPU-fallback expert execution".into());
        }
        let attention_softmax_nonfinite_fallbacks =
            crate::transformer::nonfinite_softmax_fallbacks()
                .saturating_sub(attention_fallbacks_before);
        if attention_softmax_nonfinite_fallbacks != 0 {
            return Err(format!(
                "GPU-native runtime recorded {attention_softmax_nonfinite_fallbacks} nonfinite attention fallbacks"
            )
            .into());
        }
        let generation = gpu_native_generation_evidence(
            &tokenizer,
            prompt_token_ids_sha256,
            generated_token_ids,
        )?;
        drop(request);
        drop(staging);
        Ok::<_, Box<dyn std::error::Error>>(GpuNativeQualificationCapture {
            evidence: crate::gpu_native_greedy_parity::GpuCandidateCaseEvidence {
                generation,
                model_load: greedy_parity_model_load(&runtime),
                selected_expert_ids_by_generated_position_layer: routes,
                token_loop_counters_delta,
                routed_execution_delta,
                attention_softmax_nonfinite_fallbacks,
                request_completed_expected_tokens,
                background_shutdown:
                    crate::greedy_parity::BackgroundShutdownEvidence::default(),
            },
            device,
            model_geometry,
        })
    }
    .await;
    let shutdown = runtime.shutdown_isolated().await;
    match (attempt, shutdown) {
        (Ok(mut capture), Ok(shutdown)) => {
            capture.evidence.background_shutdown = shutdown;
            Ok(capture)
        }
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error.into()),
        (Err(error), Err(shutdown_error)) => Err(IsolatedRuntimeShutdownError::new(format!(
            "{error}; GPU-native qualification shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

async fn execute_router_rank_reference(
    spec: &ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    prompt_token_ids: &[u32],
    resolved_config_sha256: &str,
    generated_position: usize,
    layer: usize,
    same_input_router_input: Option<&[f32]>,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<RouterRankReferenceCapture, Box<dyn std::error::Error>> {
    let runtime =
        build_isolated_greedy_runtime(spec, RealCliRuntimeMode::IsolatedGreedyParityCpu, tokenizer)
            .await?;
    let attempt = async {
        runtime.engine.enable_cpu_q4_boundary_emulation()?;
        let observed_config_sha256 = resolved_real_runtime_identity_sha256(
            &runtime.cfg,
            runtime.model.config.architecture,
            runtime.model.config.first_k_dense_replace,
            &runtime.model.config.advanced,
        )?;
        if observed_config_sha256 != resolved_config_sha256 {
            return Err(format!(
                "router-rank reference identity {observed_config_sha256} drifted from {resolved_config_sha256}"
            )
            .into());
        }
        if layer >= runtime.model.layers.len() {
            return Err(format!("target layer {layer} is outside loaded reference model").into());
        }
        let boundary_before = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        if !boundary_before.enabled || boundary_before.routed_expert_dispatches != 0 {
            return Err("router-rank reference boundary emulation did not start clean".into());
        }
        let routed_before = runtime.engine.routed_expert_execution_snapshot();
        let attention_before = crate::transformer::nonfinite_softmax_fallbacks();
        let gate = runtime.model.layers[layer].gate.clone();
        let gate_identity =
            crate::gpu_native_router_rank_diagnostics::GateTensorIdentity::from_gate(layer, &gate);
        let model_load = greedy_parity_model_load(&runtime);
        let (generated_token_ids, target_trace) = with_progress_timeout(
            "GPU-native router-rank authoritative reference".to_string(),
            watchdog,
            async {
                let mut kv = runtime.model.fresh_kv_caches();
                let prefix_count = prompt_token_ids.len().saturating_sub(1);
                for (position, &token_id) in prompt_token_ids[..prefix_count].iter().enumerate() {
                    runtime
                        .model
                        .forward_token_hidden(&runtime.engine, token_id, position, &mut kv)
                        .await?;
                }
                let mut input_token = *prompt_token_ids
                    .last()
                    .ok_or("router-rank reference prompt is empty")?;
                let mut position = prefix_count;
                let mut generated_token_ids = Vec::with_capacity(generated_position + 1);
                let mut target_trace = None;
                for index in 0..=generated_position {
                    let trace = runtime
                        .model
                        .forward_token_diagnostic_trace(
                            &runtime.engine,
                            input_token,
                            position,
                            &mut kv,
                            None,
                        )
                        .await?;
                    input_token = trace.sampled_token;
                    generated_token_ids.push(trace.sampled_token);
                    if index == generated_position {
                        target_trace = Some(trace);
                    }
                    position = position
                        .checked_add(1)
                        .ok_or("router-rank reference position overflowed")?;
                }
                Ok::<_, Box<dyn std::error::Error>>((
                    generated_token_ids,
                    target_trace.ok_or("router-rank reference target trace was not captured")?,
                ))
            },
        )
        .await?;
        let same_input_routed_moe = if let Some(input) = same_input_router_input {
            if input.len() != runtime.model.config.d_model {
                return Err("semantic same-input router/MoE input has invalid width".into());
            }
            let input = input.to_vec();
            let routing = gate.route(&input);
            let global_ids = routing
                .experts
                .iter()
                .map(|&expert| runtime.model.global_expert_id(layer, expert))
                .collect::<Vec<_>>();
            let expert_outputs = runtime
                .engine
                .moe_step_with_timing(
                    (generated_position as u64)
                        .wrapping_mul(runtime.model.config.num_layers as u64)
                        .wrapping_add(layer as u64),
                    layer as u32,
                    &input,
                    &global_ids,
                    None,
                )
                .await?;
            let routed_moe_output =
                crate::inference::combine_outputs(&expert_outputs, &routing.weights);
            Some(
                crate::gpu_native_expert_permutation_semantic_parity::CpuSameInputRoutedMoeCapture {
                    selected_ids: routing.experts,
                    selected_weights: routing.weights,
                    expert_outputs,
                    routed_moe_output,
                },
            )
        } else {
            None
        };
        let boundary_after = runtime.engine.cpu_q4_boundary_emulation_snapshot();
        if !boundary_after.enabled || boundary_after.routed_expert_dispatches == 0 {
            return Err("router-rank reference did not exercise the Hybrid F16 boundary".into());
        }
        let routed_delta = crate::qualification::routed_execution_delta(
            routed_before,
            runtime.engine.routed_expert_execution_snapshot(),
        )
        .map_err(|failure| failure.detail)?;
        if routed_delta.degraded_expert_substitutions != 0
            || crate::transformer::nonfinite_softmax_fallbacks().saturating_sub(attention_before)
                != 0
        {
            return Err("router-rank reference recorded degraded or nonfinite fallback".into());
        }
        Ok::<_, Box<dyn std::error::Error>>(RouterRankReferenceCapture {
            generated_token_ids,
            target_trace,
            gate,
            gate_identity,
            same_input_routed_moe,
            model_load,
            background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence::default(),
        })
    }
    .await;
    let shutdown = runtime.shutdown_isolated().await;
    match (attempt, shutdown) {
        (Ok(mut capture), Ok(shutdown)) => {
            capture.background_shutdown = shutdown;
            Ok(capture)
        }
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error.into()),
        (Err(error), Err(shutdown_error)) => Err(IsolatedRuntimeShutdownError::new(format!(
            "{error}; router-rank reference shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

#[allow(clippy::too_many_arguments)]
async fn execute_router_rank_gpu(
    spec: &ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    prompt_token_ids: &[u32],
    resolved_config_sha256: &str,
    expected_adapter_name: &str,
    case: &str,
    generated_position: usize,
    layer: usize,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<RouterRankGpuCapture, Box<dyn std::error::Error>> {
    let runtime = build_isolated_greedy_runtime(
        spec,
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
        tokenizer,
    )
    .await?;
    let attempt = async {
        let observed_config_sha256 = resolved_real_runtime_identity_sha256(
            &runtime.cfg,
            runtime.model.config.architecture,
            runtime.model.config.first_k_dense_replace,
            &runtime.model.config.advanced,
        )?;
        if observed_config_sha256 != resolved_config_sha256 {
            return Err(format!(
                "router-rank GPU identity {observed_config_sha256} drifted from {resolved_config_sha256}"
            )
            .into());
        }
        let device = runtime
            .engine
            .gpu_device_identity()
            .ok_or("router-rank GPU runtime has no authoritative adapter identity")?;
        if device.name != expected_adapter_name {
            return Err(format!(
                "router-rank GPU runtime selected adapter {:?}, expected {:?}",
                device.name, expected_adapter_name
            )
            .into());
        }
        if device.software_adapter || device.device_type.eq_ignore_ascii_case("cpu") {
            return Err(format!(
                "router-rank GPU runtime selected software adapter {:?}",
                device.name
            )
            .into());
        }
        let token_loop = runtime
            .gpu_native_token_loop
            .as_ref()
            .ok_or("GPU-native token loop was not initialized")?;
        let model_geometry = token_loop.model_geometry();
        crate::gpu_native_router_rank_diagnostics::validate_target(
            case,
            generated_position,
            layer,
            expected_adapter_name,
            model_geometry,
        )?;
        if layer >= runtime.model.layers.len() {
            return Err(format!("target layer {layer} is outside loaded GPU model").into());
        }
        let gate_identity =
            crate::gpu_native_router_rank_diagnostics::GateTensorIdentity::from_gate(
                layer,
                &runtime.model.layers[layer].gate,
            );
        let model_load = greedy_parity_model_load(&runtime);
        let counters_before = token_loop.snapshot();
        if counters_before != crate::gpu_native_token_loop::GpuNativeTokenLoopSnapshot::default() {
            return Err("router-rank GPU token-loop counters did not start at zero".into());
        }
        let routed_before = runtime.engine.routed_expert_execution_snapshot();
        let attention_before = crate::transformer::nonfinite_softmax_fallbacks();
        let mut request = token_loop.create_router_rank_diagnostic_request_state()?;
        let trace_layout =
            crate::gpu_native_router_rank_diagnostics::RouterRankTraceLayout::try_new(
                model_geometry,
                layer,
            )?;
        let staging = token_loop.create_router_rank_diagnostic_staging_buffer(&trace_layout)?;
        let (generated_token_ids, target_trace) = with_progress_timeout(
            "GPU-native router-rank candidate".to_string(),
            watchdog,
            async {
                let prefix_count = prompt_token_ids.len().saturating_sub(1);
                for (position, &token_id) in prompt_token_ids[..prefix_count].iter().enumerate() {
                    token_loop
                        .step_token(&runtime.engine, &mut request, token_id, position, false)
                        .await?;
                }
                let mut input_token = *prompt_token_ids
                    .last()
                    .ok_or("router-rank GPU prompt is empty")?;
                let mut position = prefix_count;
                let mut generated_token_ids = Vec::with_capacity(generated_position + 1);
                let mut target_trace = None;
                for index in 0..=generated_position {
                    if index == generated_position {
                        let (trace, sampled_token, _) = token_loop
                            .step_token_router_rank_diagnostic(
                                &runtime.engine,
                                &mut request,
                                input_token,
                                position,
                                &trace_layout,
                                &staging,
                            )
                            .await?;
                        generated_token_ids.push(sampled_token);
                        target_trace = Some(trace);
                    } else {
                        let sampled_token = token_loop
                            .step_token(
                                &runtime.engine,
                                &mut request,
                                input_token,
                                position,
                                true,
                            )
                            .await?
                            .ok_or("router-rank GPU generation produced no token")?;
                        input_token = sampled_token;
                        generated_token_ids.push(sampled_token);
                    }
                    position = position
                        .checked_add(1)
                        .ok_or("router-rank GPU position overflowed")?;
                }
                Ok::<_, Box<dyn std::error::Error>>((
                    generated_token_ids,
                    target_trace.ok_or("router-rank GPU target trace was not captured")?,
                ))
            },
        )
        .await?;
        let expected_completed = prompt_token_ids
            .len()
            .saturating_sub(1)
            .checked_add(generated_position + 1)
            .ok_or("router-rank expected completion count overflowed")?;
        let counters_after = token_loop.snapshot();
        if request.committed_position() != expected_completed
            || counters_after.tokens_completed != expected_completed as u64
            || counters_after.fatal_failures != 0
            || counters_after.no_progress_failures != 0
        {
            return Err("router-rank GPU request completion/counter evidence is invalid".into());
        }
        let routed_delta = crate::qualification::routed_execution_delta(
            routed_before,
            runtime.engine.routed_expert_execution_snapshot(),
        )
        .map_err(|failure| failure.detail)?;
        if routed_delta.degraded_expert_substitutions != 0
            || routed_delta.gpu_cpu_fallbacks != 0
            || routed_delta.gpu_dispatch_failures != 0
            || crate::transformer::nonfinite_softmax_fallbacks().saturating_sub(attention_before)
                != 0
        {
            return Err("router-rank GPU run recorded a fallback or dispatch failure".into());
        }
        drop(request);
        drop(staging);
        Ok::<_, Box<dyn std::error::Error>>(RouterRankGpuCapture {
            generated_token_ids,
            target_trace,
            gate_identity,
            model_load,
            device,
            model_geometry,
            background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence::default(),
        })
    }
    .await;
    let shutdown = runtime.shutdown_isolated().await;
    match (attempt, shutdown) {
        (Ok(mut capture), Ok(shutdown)) => {
            capture.background_shutdown = shutdown;
            Ok(capture)
        }
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error.into()),
        (Err(error), Err(shutdown_error)) => Err(IsolatedRuntimeShutdownError::new(format!(
            "{error}; router-rank GPU shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

#[allow(clippy::too_many_arguments)]
async fn execute_expert_permutation_semantic_gpu(
    spec: &ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    prompt_token_ids: &[u32],
    resolved_config_sha256: &str,
    expected_adapter_name: &str,
    case: &str,
    generated_position: usize,
    layer: usize,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<SemanticGpuCapture, Box<dyn std::error::Error>> {
    let runtime = build_isolated_greedy_runtime(
        spec,
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
        tokenizer,
    )
    .await?;
    let attempt = async {
        let observed_config_sha256 = resolved_real_runtime_identity_sha256(
            &runtime.cfg,
            runtime.model.config.architecture,
            runtime.model.config.first_k_dense_replace,
            &runtime.model.config.advanced,
        )?;
        if observed_config_sha256 != resolved_config_sha256 {
            return Err(format!(
                "expert-permutation semantic GPU identity {observed_config_sha256} drifted from {resolved_config_sha256}"
            )
            .into());
        }
        let device = runtime
            .engine
            .gpu_device_identity()
            .ok_or("expert-permutation semantic GPU runtime has no authoritative adapter identity")?;
        if device.name != expected_adapter_name {
            return Err(format!(
                "expert-permutation semantic GPU runtime selected adapter {:?}, expected {:?}",
                device.name, expected_adapter_name
            )
            .into());
        }
        if device.software_adapter || device.device_type.eq_ignore_ascii_case("cpu") {
            return Err(format!(
                "expert-permutation semantic GPU runtime selected software adapter {:?}",
                device.name
            )
            .into());
        }
        let token_loop = runtime
            .gpu_native_token_loop
            .as_ref()
            .ok_or("GPU-native token loop was not initialized")?;
        let model_geometry = token_loop.model_geometry();
        crate::gpu_native_router_rank_diagnostics::validate_target(
            case,
            generated_position,
            layer,
            expected_adapter_name,
            model_geometry,
        )?;
        if layer >= runtime.model.layers.len() {
            return Err(format!("target layer {layer} is outside loaded GPU model").into());
        }
        let gate_identity =
            crate::gpu_native_router_rank_diagnostics::GateTensorIdentity::from_gate(
                layer,
                &runtime.model.layers[layer].gate,
            );
        let model_load = greedy_parity_model_load(&runtime);
        let counters_before = token_loop.snapshot();
        if counters_before != crate::gpu_native_token_loop::GpuNativeTokenLoopSnapshot::default() {
            return Err("expert-permutation semantic GPU token-loop counters did not start at zero"
                .into());
        }
        let routed_before = runtime.engine.routed_expert_execution_snapshot();
        let attention_before = crate::transformer::nonfinite_softmax_fallbacks();
        let mut request = token_loop
            .create_expert_permutation_semantic_diagnostic_request_state()?;
        let trace_layout =
            crate::gpu_native_expert_permutation_semantic_parity::SemanticTraceLayout::try_new(
                model_geometry,
                layer,
            )?;
        let staging = token_loop
            .create_expert_permutation_semantic_diagnostic_staging_buffer(&trace_layout)?;
        let (generated_token_ids, target_trace) = with_progress_timeout(
            "GPU-native expert-permutation semantic candidate".to_string(),
            watchdog,
            async {
                let prefix_count = prompt_token_ids.len().saturating_sub(1);
                for (position, &token_id) in prompt_token_ids[..prefix_count].iter().enumerate() {
                    token_loop
                        .step_token(&runtime.engine, &mut request, token_id, position, false)
                        .await?;
                }
                let mut input_token = *prompt_token_ids
                    .last()
                    .ok_or("expert-permutation semantic GPU prompt is empty")?;
                let mut position = prefix_count;
                let mut generated_token_ids = Vec::with_capacity(generated_position + 1);
                let mut target_trace = None;
                for index in 0..=generated_position {
                    if index == generated_position {
                        let (trace, sampled_token, _) = token_loop
                            .step_token_expert_permutation_semantic_diagnostic(
                                &runtime.engine,
                                &mut request,
                                input_token,
                                position,
                                &trace_layout,
                                &staging,
                            )
                            .await?;
                        generated_token_ids.push(sampled_token);
                        target_trace = Some(trace);
                    } else {
                        let sampled_token = token_loop
                            .step_token(
                                &runtime.engine,
                                &mut request,
                                input_token,
                                position,
                                true,
                            )
                            .await?
                            .ok_or("expert-permutation semantic GPU generation produced no token")?;
                        input_token = sampled_token;
                        generated_token_ids.push(sampled_token);
                    }
                    position = position
                        .checked_add(1)
                        .ok_or("expert-permutation semantic GPU position overflowed")?;
                }
                Ok::<_, Box<dyn std::error::Error>>((
                    generated_token_ids,
                    target_trace
                        .ok_or("expert-permutation semantic GPU target trace was not captured")?,
                ))
            },
        )
        .await?;
        let expected_completed = prompt_token_ids
            .len()
            .saturating_sub(1)
            .checked_add(generated_position + 1)
            .ok_or("expert-permutation semantic expected completion count overflowed")?;
        let counters_after = token_loop.snapshot();
        if request.committed_position() != expected_completed
            || counters_after.tokens_completed != expected_completed as u64
            || counters_after.fatal_failures != 0
            || counters_after.no_progress_failures != 0
        {
            return Err(
                "expert-permutation semantic GPU request completion/counter evidence is invalid"
                    .into(),
            );
        }
        let routed_delta = crate::qualification::routed_execution_delta(
            routed_before,
            runtime.engine.routed_expert_execution_snapshot(),
        )
        .map_err(|failure| failure.detail)?;
        if routed_delta.degraded_expert_substitutions != 0
            || routed_delta.gpu_cpu_fallbacks != 0
            || routed_delta.gpu_dispatch_failures != 0
            || crate::transformer::nonfinite_softmax_fallbacks().saturating_sub(attention_before)
                != 0
        {
            return Err(
                "expert-permutation semantic GPU run recorded a fallback or dispatch failure"
                    .into(),
            );
        }
        drop(request);
        drop(staging);
        Ok::<_, Box<dyn std::error::Error>>(SemanticGpuCapture {
            generated_token_ids,
            target_trace,
            gate_identity,
            model_load,
            device,
            model_geometry,
            background_shutdown: crate::greedy_parity::BackgroundShutdownEvidence::default(),
        })
    }
    .await;
    let shutdown = runtime.shutdown_isolated().await;
    match (attempt, shutdown) {
        (Ok(mut capture), Ok(shutdown)) => {
            capture.background_shutdown = shutdown;
            Ok(capture)
        }
        (Err(error), Ok(_)) => Err(error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error.into()),
        (Err(error), Err(shutdown_error)) => Err(IsolatedRuntimeShutdownError::new(format!(
            "{error}; expert-permutation semantic GPU shutdown also failed: {shutdown_error}"
        ))
        .into()),
    }
}

fn emit_router_rank_divergence_report(
    report: &crate::gpu_native_router_rank_diagnostics::RouterRankDivergenceReport,
    report_out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = report_out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    std::fs::write(report_out, json)?;
    eprintln!(
        "GPU-native router-rank diagnostic report written to {}",
        report_out.display()
    );
    Ok(())
}

async fn cmd_diagnose_gpu_native_router_rank_divergence(
    args: DiagnoseGpuNativeRouterRankDivergenceArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::BuildProvenance;

    let cfg = args.parsed_config;
    let provenance = BuildProvenance::embedded();
    let (artifacts, artifact_errors) = qualification_artifacts(&args.config, &cfg);
    if !artifact_errors.is_empty() {
        return Err(format!(
            "router-rank diagnostic artifact preflight failed: {}",
            artifact_errors.join("; ")
        )
        .into());
    }
    if args.progress_watchdog.timeout.is_none() {
        return Err("router-rank diagnostic requires a positive progress timeout".into());
    }
    if provenance.dirty != Some(false)
        || provenance
            .git_sha
            .as_deref()
            .is_none_or(|sha| sha.len() != 40 || !sha.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return Err("router-rank diagnostic requires clean embedded 40-hex Git provenance".into());
    }
    let capacity_bytes = cfg
        .gpu_cache
        .vram_capacity_mb
        .checked_mul(1024 * 1024)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .unwrap_or(0);
    let source_config = crate::gpu_native_greedy_parity::SourceConfigEvidence {
        real_transformer_enabled: cfg.real_transformer.enabled,
        gpu_native_enabled: cfg.real_transformer.gpu_native,
        weights_dir_configured: cfg.real_transformer.weights_dir.is_some(),
        strict_weights: cfg.real_transformer.strict_weights,
        allow_seeded_fallback: cfg.real_transformer.allow_seeded_fallback,
        allow_degraded_experts: cfg.real_transformer.allow_degraded_experts,
        allow_nonfinite_attention_fallback: cfg.real_transformer.allow_nonfinite_attention_fallback,
        allow_truncated_expert_payloads: cfg.real_transformer.allow_truncated_expert_payloads,
        distributed_enabled: cfg.distributed.enabled,
        gpu_cache_enabled: cfg.gpu_cache.enabled,
        gpu_expert_capacity_bytes: capacity_bytes,
        routed_expert_dtype: cfg.model.dtype.as_str().to_string(),
    };
    if !source_config.is_strict() {
        return Err(
            "router-rank diagnostic requires the strict GPU-native Q4 qualifier configuration"
                .into(),
        );
    }
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| format!("router-rank expert metadata preflight failed: {error}"))?;
    if expert_metadata.q4_0_layout.as_deref() != Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1)
        || expert_metadata.explicitly_synthetic
    {
        return Err("router-rank diagnostic requires canonical nonsynthetic Q4_0 metadata".into());
    }
    let fixed_case = crate::greedy_parity::fixed_case(&args.case)
        .ok_or_else(|| format!("unknown fixed corpus case {:?}", args.case))?;
    if args.generated_position >= crate::greedy_parity::OUTPUT_TOKEN_LIMIT
        || args.layer >= cfg.model.num_layers
        || cfg.model.num_experts <= crate::gpu_native_router_rank_diagnostics::HIGHER_EXPERT_ID
        || cfg.model.top_k != crate::gpu_native_router_rank_diagnostics::REQUIRED_TOP_K
        || args.expected_adapter_name.trim().is_empty()
    {
        return Err("router-rank target, expert geometry, or adapter is invalid".into());
    }

    let mut gpu_spec =
        resolve_real_cli_spec_from_config(cfg, RealCliRuntimeMode::IsolatedGpuNativeDiagnostic)?;
    gpu_spec.cfg.real_transformer.gpu_native = true;
    gpu_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Gpu;
    let model_identity = greedy_parity_model_identity(&gpu_spec);
    if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
        return Err(
            "router-rank diagnostic requires exact Qwen3-Coder 30B-A3B Q4_0 geometry".into(),
        );
    }
    let gpu_resolved_config_sha256 = resolved_real_cli_spec_sha256(&gpu_spec)?;
    let mut reference_spec = gpu_spec.clone();
    reference_spec.cfg.real_transformer.gpu_native = false;
    reference_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Cpu;
    let reference_resolved_config_sha256 = resolved_real_cli_spec_sha256(&reference_spec)?;
    let tokenizer = load_real_cli_tokenizer(
        &gpu_spec.cfg,
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
    )?;
    let prompt_token_ids = tokenizer.encode(fixed_case.prompt)?;
    if prompt_token_ids.is_empty() {
        return Err("router-rank fixed prompt encoded to zero tokens".into());
    }

    let reference = execute_router_rank_reference(
        &reference_spec,
        tokenizer.clone(),
        &prompt_token_ids,
        &reference_resolved_config_sha256,
        args.generated_position,
        args.layer,
        None,
        args.progress_watchdog,
    )
    .await?;
    let gpu = execute_router_rank_gpu(
        &gpu_spec,
        tokenizer,
        &prompt_token_ids,
        &gpu_resolved_config_sha256,
        &args.expected_adapter_name,
        fixed_case.name,
        args.generated_position,
        args.layer,
        args.progress_watchdog,
    )
    .await?;
    crate::gpu_native_router_rank_diagnostics::validate_target(
        fixed_case.name,
        args.generated_position,
        args.layer,
        &args.expected_adapter_name,
        gpu.model_geometry,
    )?;

    let reference_hidden = reference
        .target_trace
        .layer_router_input
        .get(args.layer)
        .ok_or("reference target router input is missing")?;
    let reference_router = crate::gpu_native_router_rank_diagnostics::evaluate_cpu_router(
        "reference-hidden-to-cpu-production-router",
        &reference.gate,
        reference_hidden,
    )?;
    let captured_reference_ids = reference
        .target_trace
        .layer_selected_ids
        .get(args.layer)
        .ok_or("reference target selected IDs are missing")?;
    let captured_reference_weights = reference
        .target_trace
        .layer_selected_weights
        .get(args.layer)
        .ok_or("reference target selected weights are missing")?;
    if captured_reference_ids != &reference_router.top_8_ids
        || captured_reference_weights
            .iter()
            .map(|value| value.to_bits())
            .ne(reference_router
                .top_8_weights
                .iter()
                .map(|value| value.bits))
    {
        return Err(
            "recomputed reference router evidence differs from the captured production decision"
                .into(),
        );
    }
    let cpu_on_gpu_input = crate::gpu_native_router_rank_diagnostics::evaluate_cpu_router(
        "gpu-hidden-to-cpu-production-router",
        &reference.gate,
        &gpu.target_trace.router_input,
    )?;
    let actual_gpu_router = crate::gpu_native_router_rank_diagnostics::evaluate_actual_gpu_router(
        gpu.target_trace.raw_logits,
        gpu.target_trace.selected_ids,
        gpu.target_trace.selected_weights,
    )?;
    let gate_identity = crate::gpu_native_router_rank_diagnostics::GateIdentityEvidence::new(
        reference.gate_identity,
        gpu.gate_identity,
    );
    let (_, executable_sha256) = current_executable_identity()?;
    let report = crate::gpu_native_router_rank_diagnostics::RouterRankDivergenceReport::build(
        crate::gpu_native_router_rank_diagnostics::DiagnosticProvenance {
            build: provenance,
            executable_sha256,
            artifacts,
            gpu_resolved_config_sha256,
            reference_resolved_config_sha256,
            model_identity,
            reference_model_load: reference.model_load,
            gpu_native_model_load: gpu.model_load,
            reference_background_shutdown: reference.background_shutdown,
            gpu_native_background_shutdown: gpu.background_shutdown,
            expert_metadata,
            device: gpu.device,
        },
        fixed_case.name,
        args.generated_position,
        args.layer,
        prompt_token_ids,
        reference.generated_token_ids,
        gpu.generated_token_ids,
        reference_hidden,
        &gpu.target_trace.router_input,
        gate_identity,
        reference_router,
        cpu_on_gpu_input,
        actual_gpu_router,
    )?;
    let failure = report.failure.clone();
    emit_router_rank_divergence_report(&report, &args.report_out)?;
    if let Some(failure) = failure {
        return Err(failure.into());
    }
    Ok(())
}

fn emit_expert_permutation_semantic_parity_report(
    report: &crate::gpu_native_expert_permutation_semantic_parity::ExpertPermutationSemanticParityReport,
    report_out: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if report.qualification_pass() {
        return Err(
            "expert-permutation semantic diagnostic cannot claim qualification PASS".into(),
        );
    }
    if let Some(parent) = report_out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    std::fs::write(report_out, json)?;
    eprintln!(
        "GPU-native expert-permutation semantic diagnostic report written to {}",
        report_out.display()
    );
    Ok(())
}

async fn cmd_diagnose_gpu_native_expert_permutation_semantic_parity(
    args: DiagnoseGpuNativeExpertPermutationSemanticParityArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::BuildProvenance;

    let cfg = args.parsed_config;
    let provenance = BuildProvenance::embedded();
    let (artifacts, artifact_errors) = qualification_artifacts(&args.config, &cfg);
    if !artifact_errors.is_empty() {
        return Err(format!(
            "expert-permutation semantic diagnostic artifact preflight failed: {}",
            artifact_errors.join("; ")
        )
        .into());
    }
    if args.progress_watchdog.timeout.is_none() {
        return Err(
            "expert-permutation semantic diagnostic requires a positive progress timeout".into(),
        );
    }
    if provenance.dirty != Some(false)
        || provenance
            .git_sha
            .as_deref()
            .is_none_or(|sha| sha.len() != 40 || !sha.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        return Err(
            "expert-permutation semantic diagnostic requires clean embedded 40-hex Git provenance"
                .into(),
        );
    }
    let capacity_bytes = cfg
        .gpu_cache
        .vram_capacity_mb
        .checked_mul(1024 * 1024)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .unwrap_or(0);
    let source_config = crate::gpu_native_greedy_parity::SourceConfigEvidence {
        real_transformer_enabled: cfg.real_transformer.enabled,
        gpu_native_enabled: cfg.real_transformer.gpu_native,
        weights_dir_configured: cfg.real_transformer.weights_dir.is_some(),
        strict_weights: cfg.real_transformer.strict_weights,
        allow_seeded_fallback: cfg.real_transformer.allow_seeded_fallback,
        allow_degraded_experts: cfg.real_transformer.allow_degraded_experts,
        allow_nonfinite_attention_fallback: cfg.real_transformer.allow_nonfinite_attention_fallback,
        allow_truncated_expert_payloads: cfg.real_transformer.allow_truncated_expert_payloads,
        distributed_enabled: cfg.distributed.enabled,
        gpu_cache_enabled: cfg.gpu_cache.enabled,
        gpu_expert_capacity_bytes: capacity_bytes,
        routed_expert_dtype: cfg.model.dtype.as_str().to_string(),
    };
    if !source_config.is_strict() {
        return Err(
            "expert-permutation semantic diagnostic requires the strict GPU-native Q4 qualifier configuration"
                .into(),
        );
    }
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| {
                format!("expert-permutation semantic expert metadata preflight failed: {error}")
            })?;
    if expert_metadata.q4_0_layout.as_deref() != Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1)
        || expert_metadata.explicitly_synthetic
    {
        return Err(
            "expert-permutation semantic diagnostic requires canonical nonsynthetic Q4_0 metadata"
                .into(),
        );
    }
    let fixed_case = crate::greedy_parity::fixed_case(&args.case)
        .ok_or_else(|| format!("unknown fixed corpus case {:?}", args.case))?;
    if args.generated_position >= crate::greedy_parity::OUTPUT_TOKEN_LIMIT
        || args.layer >= cfg.model.num_layers
        || cfg.model.num_experts <= crate::gpu_native_router_rank_diagnostics::HIGHER_EXPERT_ID
        || cfg.model.top_k != crate::gpu_native_router_rank_diagnostics::REQUIRED_TOP_K
        || args.expected_adapter_name.trim().is_empty()
    {
        return Err(
            "expert-permutation semantic target, expert geometry, or adapter is invalid".into(),
        );
    }

    let mut gpu_spec =
        resolve_real_cli_spec_from_config(cfg, RealCliRuntimeMode::IsolatedGpuNativeDiagnostic)?;
    gpu_spec.cfg.real_transformer.gpu_native = true;
    gpu_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Gpu;
    let model_identity = greedy_parity_model_identity(&gpu_spec);
    if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
        return Err(
            "expert-permutation semantic diagnostic requires exact Qwen3-Coder 30B-A3B Q4_0 geometry"
                .into(),
        );
    }
    let gpu_resolved_config_sha256 = resolved_real_cli_spec_sha256(&gpu_spec)?;
    let mut reference_spec = gpu_spec.clone();
    reference_spec.cfg.real_transformer.gpu_native = false;
    reference_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Cpu;
    let reference_resolved_config_sha256 = resolved_real_cli_spec_sha256(&reference_spec)?;
    let tokenizer = load_real_cli_tokenizer(
        &gpu_spec.cfg,
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
    )?;
    let prompt_token_ids = tokenizer.encode(fixed_case.prompt)?;
    if prompt_token_ids.is_empty() {
        return Err("expert-permutation semantic fixed prompt encoded to zero tokens".into());
    }

    // The exact GPU-produced target input is required before the isolated CPU
    // reference runtime can execute the same-input routed-MoE witness.
    let gpu = execute_expert_permutation_semantic_gpu(
        &gpu_spec,
        tokenizer.clone(),
        &prompt_token_ids,
        &gpu_resolved_config_sha256,
        &args.expected_adapter_name,
        fixed_case.name,
        args.generated_position,
        args.layer,
        args.progress_watchdog,
    )
    .await?;
    let reference = execute_router_rank_reference(
        &reference_spec,
        tokenizer,
        &prompt_token_ids,
        &reference_resolved_config_sha256,
        args.generated_position,
        args.layer,
        Some(&gpu.target_trace.router_input),
        args.progress_watchdog,
    )
    .await?;
    crate::gpu_native_router_rank_diagnostics::validate_target(
        fixed_case.name,
        args.generated_position,
        args.layer,
        &args.expected_adapter_name,
        gpu.model_geometry,
    )?;

    let reference_router_input = reference
        .target_trace
        .layer_router_input
        .get(args.layer)
        .ok_or("reference target router input is missing")?;
    let reference_routed_moe_output = reference
        .target_trace
        .layer_routed_moe_output
        .get(args.layer)
        .ok_or("reference target routed-MoE output is missing")?;
    let reference_router = crate::gpu_native_router_rank_diagnostics::evaluate_cpu_router(
        "reference-hidden-to-cpu-production-router",
        &reference.gate,
        reference_router_input,
    )?;
    let captured_reference_ids = reference
        .target_trace
        .layer_selected_ids
        .get(args.layer)
        .ok_or("reference target selected IDs are missing")?;
    let captured_reference_weights = reference
        .target_trace
        .layer_selected_weights
        .get(args.layer)
        .ok_or("reference target selected weights are missing")?;
    if captured_reference_ids != &reference_router.top_8_ids
        || captured_reference_weights
            .iter()
            .map(|value| value.to_bits())
            .ne(reference_router
                .top_8_weights
                .iter()
                .map(|value| value.bits))
    {
        return Err(
            "recomputed reference router evidence differs from the captured production decision"
                .into(),
        );
    }
    let cpu_on_gpu_input = crate::gpu_native_router_rank_diagnostics::evaluate_cpu_router(
        "gpu-hidden-to-cpu-production-router",
        &reference.gate,
        &gpu.target_trace.router_input,
    )?;
    let same_input = reference
        .same_input_routed_moe
        .as_ref()
        .ok_or("same-input CPU routed-MoE witness was not captured")?;
    if same_input.selected_ids != cpu_on_gpu_input.top_8_ids
        || same_input
            .selected_weights
            .iter()
            .map(|value| value.to_bits())
            .ne(cpu_on_gpu_input
                .top_8_weights
                .iter()
                .map(|value| value.bits))
        || same_input.expert_outputs.len() != same_input.selected_ids.len()
    {
        return Err(
            "same-input CPU routed-MoE capture differs from CPU production routing evidence".into(),
        );
    }
    let actual_gpu_router = crate::gpu_native_router_rank_diagnostics::evaluate_actual_gpu_router(
        gpu.target_trace.raw_logits.clone(),
        gpu.target_trace.selected_ids.clone(),
        gpu.target_trace.selected_weights.clone(),
    )?;
    let gate_identity = crate::gpu_native_router_rank_diagnostics::GateIdentityEvidence::new(
        reference.gate_identity,
        gpu.gate_identity,
    );
    let (_, executable_sha256) = current_executable_identity()?;
    let report = crate::gpu_native_expert_permutation_semantic_parity::ExpertPermutationSemanticParityReport::build(
        crate::gpu_native_router_rank_diagnostics::DiagnosticProvenance {
            build: provenance,
            executable_sha256,
            artifacts,
            gpu_resolved_config_sha256,
            reference_resolved_config_sha256,
            model_identity,
            reference_model_load: reference.model_load,
            gpu_native_model_load: gpu.model_load,
            reference_background_shutdown: reference.background_shutdown,
            gpu_native_background_shutdown: gpu.background_shutdown,
            expert_metadata,
            device: gpu.device,
        },
        fixed_case.name,
        args.generated_position,
        args.layer,
        prompt_token_ids,
        reference.generated_token_ids,
        gpu.generated_token_ids,
        reference_router_input,
        &gpu.target_trace.router_input,
        gate_identity,
        reference_router,
        cpu_on_gpu_input,
        actual_gpu_router,
        reference_routed_moe_output,
        &same_input.routed_moe_output,
        &gpu.target_trace.routed_moe_output,
        &gpu.target_trace.route_outputs,
    )?;
    let failure = report.failure.clone();
    emit_expert_permutation_semantic_parity_report(&report, &args.report_out)?;
    if let Some(failure) = failure {
        return Err(failure.into());
    }
    Ok(())
}

async fn cmd_qualify_gpu_native_q4_greedy_parity(
    args: QualifyGpuNativeQ4GreedyParityArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::{BuildProvenance, FailureStage, QualificationFailure};

    let cfg = args.parsed_config;
    let provenance = BuildProvenance::embedded();
    let (artifacts, artifact_errors) = qualification_artifacts(&args.config, &cfg);
    let metadata_path = cfg.model.data_dir.join("metadata.json");
    let metadata_result = crate::qualification::read_expert_metadata(&metadata_path);
    let metadata = metadata_result.clone().ok();
    let capacity_bytes = cfg
        .gpu_cache
        .vram_capacity_mb
        .checked_mul(1024 * 1024)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .unwrap_or(0);
    let source_config = crate::gpu_native_greedy_parity::SourceConfigEvidence {
        real_transformer_enabled: cfg.real_transformer.enabled,
        gpu_native_enabled: cfg.real_transformer.gpu_native,
        weights_dir_configured: cfg.real_transformer.weights_dir.is_some(),
        strict_weights: cfg.real_transformer.strict_weights,
        allow_seeded_fallback: cfg.real_transformer.allow_seeded_fallback,
        allow_degraded_experts: cfg.real_transformer.allow_degraded_experts,
        allow_nonfinite_attention_fallback: cfg.real_transformer.allow_nonfinite_attention_fallback,
        allow_truncated_expert_payloads: cfg.real_transformer.allow_truncated_expert_payloads,
        distributed_enabled: cfg.distributed.enabled,
        gpu_cache_enabled: cfg.gpu_cache.enabled,
        gpu_expert_capacity_bytes: capacity_bytes,
        routed_expert_dtype: cfg.model.dtype.as_str().to_string(),
    };
    let mut report = crate::gpu_native_greedy_parity::GpuNativeGreedyParityReport::new(
        provenance.clone(),
        artifacts,
        metadata.clone(),
        args.expected_adapter_name.clone(),
        source_config,
    );

    let preflight_failure = if args.expected_adapter_name.is_empty() {
        Some(QualificationFailure::new(
            FailureStage::Preflight,
            "expected-adapter-name-empty",
            "--expected-adapter-name must be a non-empty exact adapter name",
        ))
    } else if args.progress_watchdog.timeout.is_none() {
        Some(QualificationFailure::new(
            FailureStage::Preflight,
            "progress-watchdog-required",
            "GPU-native greedy parity requires a positive progress timeout",
        ))
    } else if !artifact_errors.is_empty() {
        Some(qualification_artifact_failure(&artifact_errors))
    } else if metadata_result.is_err() {
        Some(qualification_metadata_failure(
            metadata_result.err().unwrap(),
        ))
    } else if !report.source_config.is_strict() {
        Some(QualificationFailure::new(
            FailureStage::Preflight,
            "non-strict-gpu-native-config",
            "qualification requires GPU-native Q4, strict real weights, no seeded/degraded/nonfinite/truncated fallback, no distributed mode, and a positive GPU cache budget",
        ))
    } else if provenance.dirty != Some(false)
        || provenance
            .git_sha
            .as_deref()
            .is_none_or(|sha| sha.len() != 40 || !sha.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        Some(QualificationFailure::new(
            FailureStage::Preflight,
            "clean-build-provenance-required",
            "qualification requires an embedded clean 40-hex Git build identity",
        ))
    } else if metadata.as_ref().is_none_or(|metadata| {
        metadata.q4_0_layout.as_deref() != Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1)
            || metadata.explicitly_synthetic
    }) {
        Some(QualificationFailure::new(
            FailureStage::Preflight,
            "noncanonical-expert-metadata",
            "qualification requires canonical, nonsynthetic Q4_0 expert metadata",
        ))
    } else {
        None
    };
    if let Some(failure) = preflight_failure {
        return fail_gpu_native_greedy_parity(report, failure, args.report_out.as_deref());
    }

    let mut gpu_spec = match resolve_real_cli_spec_from_config(
        cfg,
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
    ) {
        Ok(spec) => spec,
        Err(error) => {
            return fail_gpu_native_greedy_parity(
                report,
                QualificationFailure::new(
                    FailureStage::Preflight,
                    "configuration-reconciliation-failed",
                    error.to_string(),
                ),
                args.report_out.as_deref(),
            );
        }
    };
    gpu_spec.cfg.real_transformer.gpu_native = true;
    gpu_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Gpu;
    let identity = greedy_parity_model_identity(&gpu_spec);
    if !identity.is_qwen3_coder_30b_a3b_q4_0() {
        return fail_gpu_native_greedy_parity(
            report,
            QualificationFailure::new(
                FailureStage::Preflight,
                "unexpected-model-identity",
                "fixed corpus is qualified only for Qwen3-Coder 30B-A3B Q4_0 geometry",
            ),
            args.report_out.as_deref(),
        );
    }
    let resolved_config_sha256 = match resolved_real_cli_spec_sha256(&gpu_spec) {
        Ok(hash) => hash,
        Err(error) => {
            return fail_gpu_native_greedy_parity(
                report,
                QualificationFailure::new(
                    FailureStage::Preflight,
                    "resolved-config-hash-failed",
                    error.to_string(),
                ),
                args.report_out.as_deref(),
            );
        }
    };
    report.resolved_config_sha256 = Some(resolved_config_sha256.clone());
    let mut reference_spec = gpu_spec.clone();
    reference_spec.cfg.real_transformer.gpu_native = false;
    reference_spec.cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Cpu;
    let reference_resolved_config_sha256 = match resolved_real_cli_spec_sha256(&reference_spec) {
        Ok(hash) => hash,
        Err(error) => {
            return fail_gpu_native_greedy_parity(
                report,
                QualificationFailure::new(
                    FailureStage::Preflight,
                    "reference-config-hash-failed",
                    error.to_string(),
                ),
                args.report_out.as_deref(),
            );
        }
    };
    report.reference_resolved_config_sha256 = Some(reference_resolved_config_sha256.clone());
    let tokenizer = match load_real_cli_tokenizer(
        &gpu_spec.cfg,
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic,
    ) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            return fail_gpu_native_greedy_parity(
                report,
                QualificationFailure::new(
                    FailureStage::Startup,
                    "tokenizer-load-failed",
                    error.to_string(),
                ),
                args.report_out.as_deref(),
            );
        }
    };

    for fixed in crate::greedy_parity::FIXED_CORPUS {
        let prompt_token_ids = match tokenizer.encode(fixed.prompt) {
            Ok(ids) if !ids.is_empty() => ids,
            Ok(_) => {
                return fail_gpu_native_greedy_parity(
                    report,
                    QualificationFailure::new(
                        FailureStage::Inference,
                        "prompt-tokenization-empty",
                        format!("fixed case {} encoded to zero tokens", fixed.name),
                    ),
                    args.report_out.as_deref(),
                );
            }
            Err(error) => {
                return fail_gpu_native_greedy_parity(
                    report,
                    QualificationFailure::new(
                        FailureStage::Inference,
                        "prompt-tokenization-failed",
                        format!("fixed case {}: {error}", fixed.name),
                    ),
                    args.report_out.as_deref(),
                );
            }
        };
        let prompt_token_ids_sha256 = crate::greedy_parity::token_ids_sha256(&prompt_token_ids);
        let authoritative_reference = match execute_greedy_parity_boundary_reference_plane(
            &reference_spec,
            tokenizer.clone(),
            &prompt_token_ids,
            &prompt_token_ids_sha256,
            &reference_resolved_config_sha256,
            &args.expected_adapter_name,
            args.progress_watchdog,
            fixed.name,
        )
        .await
        {
            Ok(reference) => reference,
            Err(error) => {
                return fail_gpu_native_greedy_parity(
                    report,
                    gpu_native_case_execution_failure(
                        "authoritative-reference-failed",
                        fixed.name,
                        error.as_ref(),
                    ),
                    args.report_out.as_deref(),
                );
            }
        };
        if !authoritative_reference.cpu_q4_boundary_emulation.enabled
            || authoritative_reference
                .cpu_q4_boundary_emulation
                .routed_expert_dispatches
                == 0
        {
            return fail_gpu_native_greedy_parity(
                report,
                QualificationFailure::new(
                    FailureStage::Postcondition,
                    "authoritative-reference-boundary-missing",
                    format!(
                        "fixed case {} did not exercise the production Hybrid F16 boundary",
                        fixed.name
                    ),
                ),
                args.report_out.as_deref(),
            );
        }
        let reference_routing = match execute_gpu_native_reference_routing_replay(
            &reference_spec,
            tokenizer.clone(),
            &prompt_token_ids,
            &prompt_token_ids_sha256,
            &reference_resolved_config_sha256,
            args.progress_watchdog,
            fixed.name,
        )
        .await
        {
            Ok(capture) => capture,
            Err(error) => {
                return fail_gpu_native_greedy_parity(
                    report,
                    gpu_native_case_execution_failure(
                        "reference-routing-replay-failed",
                        fixed.name,
                        error.as_ref(),
                    ),
                    args.report_out.as_deref(),
                );
            }
        };
        let reference_routing_matches = reference_routing.generation.generated_token_ids
            == authoritative_reference.generation.generated_token_ids;
        if !reference_routing_matches {
            return fail_gpu_native_greedy_parity(
                report,
                QualificationFailure::new(
                    FailureStage::Postcondition,
                    "reference-routing-replay-token-mismatch",
                    format!(
                        "fixed case {} diagnostic routing replay drifted from the authoritative reference",
                        fixed.name
                    ),
                ),
                args.report_out.as_deref(),
            );
        }
        let gpu_native = match execute_gpu_native_qualification_case(
            &gpu_spec,
            tokenizer.clone(),
            &prompt_token_ids,
            &prompt_token_ids_sha256,
            &resolved_config_sha256,
            &args.expected_adapter_name,
            args.progress_watchdog,
            fixed.name,
        )
        .await
        {
            Ok(capture) => capture,
            Err(error) => {
                return fail_gpu_native_greedy_parity(
                    report,
                    gpu_native_case_execution_failure(
                        "gpu-native-candidate-failed",
                        fixed.name,
                        error.as_ref(),
                    ),
                    args.report_out.as_deref(),
                );
            }
        };
        if let Some(observed) = &report.actual_adapter {
            if observed != &gpu_native.device {
                return fail_gpu_native_greedy_parity(
                    report,
                    QualificationFailure::new(
                        FailureStage::Postcondition,
                        "adapter-identity-drift",
                        format!("fixed case {} selected a different adapter", fixed.name),
                    ),
                    args.report_out.as_deref(),
                );
            }
        } else {
            report.actual_adapter = Some(gpu_native.device.clone());
        }
        if let Some(observed) = report.model_geometry {
            if observed != gpu_native.model_geometry {
                return fail_gpu_native_greedy_parity(
                    report,
                    QualificationFailure::new(
                        FailureStage::Postcondition,
                        "model-geometry-drift",
                        format!(
                            "fixed case {} observed different model geometry",
                            fixed.name
                        ),
                    ),
                    args.report_out.as_deref(),
                );
            }
        } else {
            report.model_geometry = Some(gpu_native.model_geometry);
        }
        let comparison = match crate::gpu_native_greedy_parity::compare_case_outputs(
            fixed.name,
            &prompt_token_ids,
            &authoritative_reference.generation.generated_token_ids,
            &gpu_native.evidence.generation.generated_token_ids,
            &reference_routing.routes,
            &gpu_native
                .evidence
                .selected_expert_ids_by_generated_position_layer,
            crate::greedy_parity::OUTPUT_TOKEN_LIMIT,
            gpu_native.model_geometry.num_layers,
            gpu_native.model_geometry.top_k,
        ) {
            Ok(comparison) => comparison,
            Err(detail) => {
                return fail_gpu_native_greedy_parity(
                    report,
                    QualificationFailure::new(
                        FailureStage::Postcondition,
                        "comparison-evidence-invalid",
                        detail,
                    ),
                    args.report_out.as_deref(),
                );
            }
        };
        let case = crate::gpu_native_greedy_parity::CaseReport {
            name: fixed.name.to_string(),
            behavior: fixed.behavior.to_string(),
            prompt: fixed.prompt.to_string(),
            prompt_sha256: crate::greedy_parity::sha256_hex(fixed.prompt.as_bytes()),
            prompt_token_ids,
            prompt_token_ids_sha256,
            tokens_per_case: crate::greedy_parity::OUTPUT_TOKEN_LIMIT,
            authoritative_reference,
            reference_routing_replay:
                crate::gpu_native_greedy_parity::ReferenceRoutingReplayEvidence {
                    generation: reference_routing.generation,
                    matches_authoritative_reference: reference_routing_matches,
                    routing_positions_captured: reference_routing.routes.len(),
                    routed_layers_per_position: gpu_native.model_geometry.num_layers,
                    selected_expert_ids_by_generated_position_layer: reference_routing.routes,
                    background_shutdown: reference_routing.background_shutdown,
                },
            gpu_native: gpu_native.evidence,
            comparison,
        };
        if let Err(detail) = report.record_case(case) {
            return fail_gpu_native_greedy_parity(
                report,
                QualificationFailure::new(
                    FailureStage::Postcondition,
                    "report-accounting-failed",
                    detail,
                ),
                args.report_out.as_deref(),
            );
        }
        if let Some(failure) = report.semantic_failure() {
            return fail_gpu_native_greedy_parity(report, failure, args.report_out.as_deref());
        }
    }
    if let Err(failure) = report.finalize() {
        return fail_gpu_native_greedy_parity(report, failure, args.report_out.as_deref());
    }
    emit_gpu_native_greedy_parity_report(&report, args.report_out.as_deref())
}

async fn cmd_qualify_hybrid_q4_greedy_parity(
    args: QualifyHybridQ4GreedyParityArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::qualification::{
        validate_preflight, BuildProvenance, FailureStage, PreflightEvidence,
        QualificationChecks, QualificationFailure,
    };

    // The supplied Hybrid config is parsed exactly once. Reconciliation below
    // produces one frozen spec from which both isolated planes are derived.
    let cfg = args.parsed_config;
    let provenance = BuildProvenance::embedded();
    let (artifacts, artifact_errors) = qualification_artifacts(&args.config, &cfg);
    let metadata_path = cfg.model.data_dir.join("metadata.json");
    let metadata_result = crate::qualification::read_expert_metadata(&metadata_path);
    let metadata = metadata_result.clone().ok();
    let mut report = crate::greedy_parity::GreedyParityReport::new(
        provenance.clone(),
        artifacts,
        metadata.clone(),
        args.expected_adapter_name.clone(),
    );

    if args.expected_adapter_name.is_empty() {
        return fail_greedy_parity(
            report,
            QualificationFailure::new(
                FailureStage::Preflight,
                "expected-adapter-name-empty",
                "--expected-adapter-name must be a non-empty exact adapter name",
            ),
            args.report_out.as_deref(),
        );
    }
    if args.progress_watchdog.timeout.is_none() {
        return fail_greedy_parity(
            report,
            QualificationFailure::new(
                FailureStage::Preflight,
                "progress-watchdog-required",
                "greedy parity requires a positive performance.progress_timeout_secs or --progress-timeout-secs",
            ),
            args.report_out.as_deref(),
        );
    }
    if !artifact_errors.is_empty() {
        return fail_greedy_parity(
            report,
            qualification_artifact_failure(&artifact_errors),
            args.report_out.as_deref(),
        );
    }
    let metadata = match metadata_result {
        Ok(metadata) => metadata,
        Err(detail) => {
            return fail_greedy_parity(
                report,
                qualification_metadata_failure(detail),
                args.report_out.as_deref(),
            );
        }
    };
    let weight_policy = cfg.real_transformer.resolve_weight_policy();
    if !matches!(weight_policy, Ok(crate::config::RealWeightPolicy::StrictReal)) {
        return fail_greedy_parity(
            report,
            QualificationFailure::new(
                FailureStage::Preflight,
                "non-strict-weight-policy",
                weight_policy
                    .err()
                    .unwrap_or_else(|| "resolved weight policy is SeededDev".to_string()),
            ),
            args.report_out.as_deref(),
        );
    }
    let capacity_bytes = cfg
        .gpu_cache
        .vram_capacity_mb
        .checked_mul(1024 * 1024)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .unwrap_or(0);
    let preflight = PreflightEvidence {
        provenance,
        real_transformer_enabled: cfg.real_transformer.enabled,
        weights_dir_configured: cfg.real_transformer.weights_dir.is_some(),
        strict_weights: cfg.real_transformer.strict_weights,
        allow_seeded_fallback: cfg.real_transformer.allow_seeded_fallback,
        allow_degraded_experts: cfg.real_transformer.allow_degraded_experts,
        allow_attention_fallback: cfg.real_transformer.allow_nonfinite_attention_fallback,
        allow_truncated_expert_payloads: cfg.real_transformer.allow_truncated_expert_payloads,
        distributed_enabled: cfg.distributed.enabled,
        gpu_cache_enabled: cfg.gpu_cache.enabled,
        gpu_expert_capacity_bytes: capacity_bytes,
        requested_mode: cfg.real_transformer.compute_offload,
        routed_expert_dtype: cfg.model.dtype,
        metadata,
    };
    report.source_preflight = Some(crate::greedy_parity::StrictHybridPreflightEvidence {
        real_transformer_enabled: preflight.real_transformer_enabled,
        weights_dir_configured: preflight.weights_dir_configured,
        strict_weights: preflight.strict_weights,
        allow_seeded_fallback: preflight.allow_seeded_fallback,
        allow_degraded_experts: preflight.allow_degraded_experts,
        allow_attention_fallback: preflight.allow_attention_fallback,
        allow_truncated_expert_payloads: preflight.allow_truncated_expert_payloads,
        distributed_enabled: preflight.distributed_enabled,
        gpu_cache_enabled: preflight.gpu_cache_enabled,
        gpu_expert_capacity_bytes: preflight.gpu_expert_capacity_bytes,
        requested_mode: match preflight.requested_mode {
            crate::backend::ComputeOffload::Cpu => "cpu",
            crate::backend::ComputeOffload::Gpu => "gpu",
            crate::backend::ComputeOffload::Auto => "auto",
            crate::backend::ComputeOffload::Hybrid => "hybrid",
        }
        .to_string(),
        routed_expert_dtype: preflight.routed_expert_dtype.as_str().to_string(),
    });
    let mut base_checks = QualificationChecks::default();
    if let Err(failure) = validate_preflight(&preflight, &mut base_checks) {
        return fail_greedy_parity(report, failure, args.report_out.as_deref());
    }
    let spec = match resolve_real_cli_spec_from_config(
        cfg,
        RealCliRuntimeMode::IsolatedGreedyParityHybrid,
    ) {
        Ok(spec) => spec,
        Err(error) => {
            return fail_greedy_parity(
                report,
                QualificationFailure::new(
                    FailureStage::Preflight,
                    "configuration-reconciliation-failed",
                    error.to_string(),
                ),
                args.report_out.as_deref(),
            );
        }
    };
    let identity = greedy_parity_model_identity(&spec);
    if !identity.is_qwen3_coder_30b_a3b_q4_0() {
        report.model_identity = Some(identity);
        return fail_greedy_parity(
            report,
            QualificationFailure::new(
                FailureStage::Preflight,
                "unexpected-model-identity",
                "fixed corpus is qualified only for Qwen3-Coder 30B-A3B Q4_0 geometry",
            ),
            args.report_out.as_deref(),
        );
    }
    report.model_identity = Some(identity);
    let resolved_config_sha256 = match resolved_real_cli_spec_sha256(&spec) {
        Ok(hash) => hash,
        Err(error) => {
            return fail_greedy_parity(
                report,
                QualificationFailure::new(
                    FailureStage::Preflight,
                    "resolved-config-hash-failed",
                    error.to_string(),
                ),
                args.report_out.as_deref(),
            );
        }
    };
    report.resolved_config_sha256 = Some(resolved_config_sha256.clone());
    let worker_timeout = args
        .progress_watchdog
        .timeout
        .expect("positive progress timeout checked above");
    let (worker_executable, executable_sha256) = match current_executable_identity() {
        Ok(identity) => identity,
        Err(error) => {
            return fail_greedy_parity(
                report,
                QualificationFailure::new(
                    FailureStage::Startup,
                    "worker-executable-identity-failed",
                    error.to_string(),
                ),
                args.report_out.as_deref(),
            );
        }
    };
    report.orchestrator_executable_sha256 = Some(executable_sha256.clone());
    let build_git_sha = match report.provenance.git_sha.clone() {
        Some(sha)
            if sha.len() == 40 && sha.bytes().all(|byte| byte.is_ascii_hexdigit()) =>
        {
            sha
        }
        _ => {
            return fail_greedy_parity(
                report,
                QualificationFailure::new(
                    FailureStage::Preflight,
                    "worker-build-identity-unavailable",
                    "same-binary worker isolation requires an embedded immutable build git SHA",
                ),
                args.report_out.as_deref(),
            );
        }
    };
    let tokenizer = match load_real_cli_tokenizer(
        &spec.cfg,
        RealCliRuntimeMode::IsolatedGreedyParityHybrid,
    ) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            return fail_greedy_parity(
                report,
                QualificationFailure::new(
                    FailureStage::Startup,
                    "tokenizer-load-failed",
                    error.to_string(),
                ),
                args.report_out.as_deref(),
            );
        }
    };

    for (case_index, fixed) in crate::greedy_parity::FIXED_CORPUS
        .into_iter()
        .enumerate()
    {
        // The sole encode call for this case happens here. The CPU runtime
        // receives this immutable ID slice directly; the same IDs and frozen
        // tokenizer artifact identity cross the typed worker protocol.
        let prompt_token_ids = match tokenizer.encode(fixed.prompt) {
            Ok(ids) if !ids.is_empty() => ids,
            Ok(_) => {
                return fail_greedy_parity(
                    report,
                    QualificationFailure::new(
                        FailureStage::Inference,
                        "prompt-tokenization-empty",
                        format!("fixed case {} encoded to zero tokens", fixed.name),
                    ),
                    args.report_out.as_deref(),
                );
            }
            Err(error) => {
                return fail_greedy_parity(
                    report,
                    QualificationFailure::new(
                        FailureStage::Inference,
                        "prompt-tokenization-failed",
                        format!("fixed case {}: {error}", fixed.name),
                    ),
                    args.report_out.as_deref(),
                );
            }
        };
        let mut case = crate::greedy_parity::CaseReport::new(fixed, prompt_token_ids);
        let cpu = execute_greedy_parity_plane(
            &spec,
            RealCliRuntimeMode::IsolatedGreedyParityCpu,
            tokenizer.clone(),
            &case.prompt_token_ids,
            &case.prompt_token_ids_sha256,
            &resolved_config_sha256,
            &args.expected_adapter_name,
            args.progress_watchdog,
            fixed.name,
        )
        .await;
        let cpu = match cpu {
            Ok(cpu) => cpu,
            Err(error) => {
                let mut failure = qualification_inference_failure(error.as_ref());
                failure.detail = format!("fixed case {} CPU control: {error}", fixed.name);
                case.failure = Some(failure.clone());
                report.cases.push(case);
                return fail_greedy_parity(report, failure, args.report_out.as_deref());
            }
        };
        case.cpu = Some(cpu);

        let boundary_reference = execute_greedy_parity_boundary_reference_plane(
            &spec,
            tokenizer.clone(),
            &case.prompt_token_ids,
            &case.prompt_token_ids_sha256,
            &resolved_config_sha256,
            &args.expected_adapter_name,
            args.progress_watchdog,
            fixed.name,
        )
        .await;
        let boundary_reference = match boundary_reference {
            Ok(reference) => reference,
            Err(error) => {
                let mut failure = qualification_inference_failure(error.as_ref());
                failure.detail = format!(
                    "fixed case {} CPU Hybrid-boundary reference: {error}",
                    fixed.name
                );
                case.failure = Some(failure.clone());
                report.cases.push(case);
                return fail_greedy_parity(report, failure, args.report_out.as_deref());
            }
        };
        case.boundary_reference = Some(boundary_reference);

        let worker_request = crate::greedy_parity::HybridWorkerRequest::new(
            crate::greedy_parity::hybrid_worker_id(
                &build_git_sha,
                &executable_sha256,
                case_index,
                fixed.name,
            ),
            fixed,
            resolved_config_sha256.clone(),
            args.expected_adapter_name.clone(),
            case.prompt_token_ids.clone(),
            executable_sha256.clone(),
            build_git_sha.clone(),
        );
        let hybrid = execute_greedy_parity_hybrid_worker(
            &worker_executable,
            &args.config,
            &worker_request,
            worker_timeout,
        )
        .await;
        let hybrid = match hybrid {
            Ok(hybrid) => hybrid,
            Err(worker_error) => {
                let mut failure = QualificationFailure::new(
                    FailureStage::Inference,
                    "hybrid-worker-failed",
                    format!(
                        "fixed case {} strict Hybrid worker: {}",
                        fixed.name, worker_error.detail
                    ),
                );
                if worker_error.evidence.timed_out {
                    failure.code = "hybrid-worker-timeout".to_string();
                }
                case.worker_failure = Some(worker_error.evidence);
                case.failure = Some(failure.clone());
                report.cases.push(case);
                return fail_greedy_parity(report, failure, args.report_out.as_deref());
            }
        };
        case.hybrid = Some(hybrid);
        let (cpu_generation, boundary_generation, hybrid_generation) =
            match (&case.cpu, &case.boundary_reference, &case.hybrid) {
            (Some(cpu), Some(boundary), Some(hybrid)) => (
                &cpu.generation,
                &boundary.generation,
                &hybrid.generation,
            ),
            _ => {
                let failure = QualificationFailure::new(
                    FailureStage::Postcondition,
                    "plane-evidence-incomplete",
                    format!(
                        "fixed case {} lost successful CPU, boundary-reference, or Hybrid evidence",
                        fixed.name
                    ),
                );
                case.failure = Some(failure.clone());
                report.cases.push(case);
                return fail_greedy_parity(report, failure, args.report_out.as_deref());
            }
        };
        let ordinary_comparison = crate::greedy_parity::compare_generations(
            cpu_generation,
            hybrid_generation,
            |ids| tokenizer.decode(ids).map_err(|error| error.to_string()),
        );
        let ordinary_comparison = match ordinary_comparison {
            Ok(comparison) => comparison,
            Err(error) => {
                let failure = QualificationFailure::new(
                    FailureStage::Postcondition,
                    "divergence-evidence-decode-failed",
                    format!("fixed case {} ordinary CPU comparison: {error}", fixed.name),
                );
                case.failure = Some(failure.clone());
                report.cases.push(case);
                return fail_greedy_parity(report, failure, args.report_out.as_deref());
            }
        };
        case.ordinary_cpu_vs_hybrid = Some(ordinary_comparison);

        let boundary_comparison = crate::greedy_parity::compare_generations(
            boundary_generation,
            hybrid_generation,
            |ids| tokenizer.decode(ids).map_err(|error| error.to_string()),
        );
        let boundary_comparison = match boundary_comparison {
            Ok(comparison) => comparison,
            Err(error) => {
                let failure = QualificationFailure::new(
                    FailureStage::Postcondition,
                    "divergence-evidence-decode-failed",
                    format!(
                        "fixed case {} CPU Hybrid-boundary comparison: {error}",
                        fixed.name
                    ),
                );
                case.failure = Some(failure.clone());
                report.cases.push(case);
                return fail_greedy_parity(report, failure, args.report_out.as_deref());
            }
        };
        let diverged = !boundary_comparison.exact_token_ids
            || !boundary_comparison.equal_generated_count
            || !boundary_comparison.equal_termination_reason
            || !boundary_comparison.equal_generated_text_hash;
        let divergence_position = boundary_comparison
            .first_divergence
            .as_ref()
            .map(|evidence| evidence.position);
        case.boundary_reference_vs_hybrid = Some(boundary_comparison);
        if diverged {
            let failure = QualificationFailure::new(
                FailureStage::Postcondition,
                "boundary-reference-greedy-token-divergence",
                format!(
                    "fixed case {} Hybrid diverged from the CPU Hybrid-boundary reference at position {:?}",
                    fixed.name, divergence_position
                ),
            );
            case.failure = Some(failure.clone());
            report.cases.push(case);
            return fail_greedy_parity(report, failure, args.report_out.as_deref());
        }
        report.cases.push(case);
    }
    if let Err(failure) = report.finish() {
        return fail_greedy_parity(report, failure, args.report_out.as_deref());
    }
    emit_greedy_parity_report(&report, args.report_out.as_deref())
}

async fn with_progress_timeout<T, F>(
    label: String,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
    fut: F,
) -> Result<T, Box<dyn std::error::Error>>
where
    F: std::future::Future<Output = Result<T, Box<dyn std::error::Error>>>,
{
    if let Some(timeout) = watchdog.timeout {
        match tokio::time::timeout(timeout, fut).await {
            Ok(result) => result,
            Err(_) => Err(format!(
                "progress watchdog fired after {}s while waiting for {label}",
                timeout.as_secs()
            )
            .into()),
        }
    } else {
        fut.await
    }
}

/// Strict real-model benchmarks must not emit attention-softmax non-finite
/// fallbacks: any increase over the measured window signals a numerically
/// invalid run (a `NaN`/`inf` propagated into attention), so the benchmark is
/// rejected rather than reporting a valid-looking throughput (gist Finding 9).
fn assert_no_softmax_fallbacks(before: u64) -> Result<(), Box<dyn std::error::Error>> {
    let delta = crate::transformer::nonfinite_softmax_fallbacks().saturating_sub(before);
    if delta > 0 {
        return Err(format!(
            "bench-real INVALID: {delta} attention-softmax non-finite fallback(s) occurred during \
             the measured window; the run produced NaN/inf attention scores and is not a valid \
             measurement"
        )
        .into());
    }
    Ok(())
}

fn cmd_q8_expert_microbench(
    d_model: usize,
    d_ff: usize,
    warmup_runs: usize,
    measured_runs: usize,
    kernel: Q8BenchKernel,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "q8-candle-reference"))]
    {
        let _ = (d_model, d_ff, warmup_runs, measured_runs, kernel);
        return Err(
            "q8-expert-microbench compares against Candle; on the AVX-512 baseline rebuild with \
             `cargo run --release --features avx512,q8-candle-reference -- q8-expert-microbench`"
                .into(),
        );
    }

    #[cfg(feature = "q8-candle-reference")]
    {
        if d_model == 0
            || d_ff == 0
            || !d_model.is_multiple_of(crate::inference::Q8_0_BLOCK_ELEMS)
            || !d_ff.is_multiple_of(crate::inference::Q8_0_BLOCK_ELEMS)
        {
            return Err("Q8 Candle comparison requires positive dimensions divisible by 32".into());
        }
        if measured_runs == 0 {
            return Err("q8-expert-microbench requires --measured-runs > 0".into());
        }
        let weights = d_model.checked_mul(d_ff).ok_or("Q8 benchmark shape overflow")?;
        let blocks = weights.div_ceil(crate::inference::Q8_0_BLOCK_ELEMS);
        let logical_bytes = blocks
            .checked_mul(crate::inference::Q8_0_BLOCK_BYTES)
            .and_then(|n| n.checked_mul(3))
            .ok_or("Q8 benchmark byte size overflow")?;
        let mut bytes = Vec::with_capacity(logical_bytes);
        let mut state = 0x9182_7364_55aa_f00du64;
        for _ in 0..blocks * 3 {
            let mut values = [0.0f32; crate::inference::Q8_0_BLOCK_ELEMS];
            for value in &mut values {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *value = (((state >> 40) as u32 as f32) / ((1u32 << 24) - 1) as f32 - 0.5)
                    * 0.08;
            }
            let mut block = [0u8; crate::inference::Q8_0_BLOCK_BYTES];
            crate::inference::quantize_q8_0_block(&values, &mut block);
            bytes.extend_from_slice(&block);
        }
        let x = crate::inference::synth_hidden_state(0, d_model, 0x51);
        let slot = logical_bytes.div_ceil(4096) * 4096;

        #[derive(Clone, Copy)]
        enum BenchPath {
            Candle,
            Direct(crate::inference::Q8DirectKernelChoice),
        }

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        struct MemorySnapshot {
            resident: u64,
            pool_allocated: u64,
            pool_primary: u64,
            pool_shadow: u64,
            prepared_duplicate: u64,
        }

        impl MemorySnapshot {
            fn capture() -> Self {
                Self {
                    resident: crate::expert_cache::resident_expert_buffer_bytes(),
                    pool_allocated: crate::buffer_pool::expert_buffer_pool_allocated_bytes(),
                    pool_primary: crate::buffer_pool::expert_buffer_pool_primary_bytes(),
                    pool_shadow: crate::buffer_pool::expert_buffer_pool_shadow_bytes(),
                    prepared_duplicate: crate::inference::prepared_duplicate_expert_bytes(),
                }
            }

            fn delta(self, baseline: Self) -> Self {
                Self {
                    resident: self.resident.saturating_sub(baseline.resident),
                    pool_allocated: self.pool_allocated.saturating_sub(baseline.pool_allocated),
                    pool_primary: self.pool_primary.saturating_sub(baseline.pool_primary),
                    pool_shadow: self.pool_shadow.saturating_sub(baseline.pool_shadow),
                    prepared_duplicate: self
                        .prepared_duplicate
                        .saturating_sub(baseline.prepared_duplicate),
                }
            }
        }

        struct LifecycleResult {
            first_exec_ms: f64,
            repeated_ms: f64,
            after_construction: MemorySnapshot,
            after_first: MemorySnapshot,
            after_repeated: MemorySnapshot,
            after_drop: MemorySnapshot,
        }

        fn residents(
            count: usize,
            slot: usize,
            bytes: &[u8],
        ) -> (
            crate::buffer_pool::BufferPool,
            Vec<crate::expert_cache::ExpertResident>,
        ) {
            let pool = crate::buffer_pool::BufferPool::new(count, slot, 4096);
            let experts = (0..count)
                .map(|id| {
                    let mut buffer = pool.try_acquire().expect("benchmark buffer slot");
                    buffer.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
                    crate::expert_cache::ExpertResident::new(id as u32, buffer)
                })
                .collect();
            (pool, experts)
        }

        fn execute(
            path: BenchPath,
            token_idx: u64,
            resident: &crate::expert_cache::ExpertResident,
            x: &[f32],
            d_model: usize,
            d_ff: usize,
        ) -> Result<
            (crate::inference::InferenceOutput, crate::inference::HiddenState),
            crate::inference::ExpertWeightsError,
        > {
            match path {
                BenchPath::Candle => crate::inference::run_inference_q8_0_qmm_with_timing(
                    token_idx, resident, x, d_model, d_ff, None,
                ),
                BenchPath::Direct(choice) => {
                    crate::inference::run_inference_q8_0_direct_with_timing_and_kernel(
                        token_idx, resident, x, d_model, d_ff, None, choice,
                    )
                }
            }
        }

        fn lifecycle(
            path: BenchPath,
            count: usize,
            slot: usize,
            bytes: &[u8],
            x: &[f32],
            d_model: usize,
            d_ff: usize,
            warmup: usize,
            measured: usize,
            logical_bytes: usize,
        ) -> Result<LifecycleResult, Box<dyn std::error::Error>> {
            let baseline = MemorySnapshot::capture();
            let (pool, experts) = residents(count, slot, bytes);
            let after_construction = MemorySnapshot::capture();
            let started = Instant::now();
            for resident in &experts {
                let output = execute(path, 0, resident, x, d_model, d_ff)?;
                std::hint::black_box(output);
            }
            let first_exec_ms = started.elapsed().as_secs_f64() * 1000.0;
            let after_first = MemorySnapshot::capture();
            let expected_prepared = match path {
                BenchPath::Candle => (count as u64).saturating_mul(logical_bytes as u64),
                BenchPath::Direct(_) => 0,
            };
            let prepared_delta = after_first
                .prepared_duplicate
                .saturating_sub(baseline.prepared_duplicate);
            if prepared_delta != expected_prepared {
                return Err(format!(
                    "prepared duplicate bytes after first execution: got {prepared_delta}, expected {expected_prepared}"
                )
                .into());
            }
            for iteration in 0..warmup {
                for resident in &experts {
                    std::hint::black_box(execute(
                        path,
                        iteration as u64,
                        resident,
                        x,
                        d_model,
                        d_ff,
                    )?);
                }
            }
            let started = Instant::now();
            for iteration in 0..measured {
                for resident in &experts {
                    std::hint::black_box(execute(
                        path,
                        iteration as u64,
                        resident,
                        x,
                        d_model,
                        d_ff,
                    )?);
                }
            }
            let repeated_ms = started.elapsed().as_secs_f64() * 1000.0 / measured as f64;
            let after_repeated = MemorySnapshot::capture();
            drop(experts);
            drop(pool);
            let after_drop = MemorySnapshot::capture();
            if after_drop != baseline {
                return Err(format!(
                    "Q8 benchmark memory counters did not return to baseline: baseline={baseline:?}, after_drop={after_drop:?}"
                )
                .into());
            }
            Ok(LifecycleResult {
                first_exec_ms,
                repeated_ms,
                after_construction: after_construction.delta(baseline),
                after_first: after_first.delta(baseline),
                after_repeated: after_repeated.delta(baseline),
                after_drop: after_drop.delta(baseline),
            })
        }

        let selected = match kernel {
            Q8BenchKernel::Auto => vec![crate::inference::Q8DirectKernelChoice::Auto],
            Q8BenchKernel::Scalar => vec![crate::inference::Q8DirectKernelChoice::Scalar],
            Q8BenchKernel::Avx2 => vec![crate::inference::Q8DirectKernelChoice::Avx2],
            Q8BenchKernel::Avx512 => vec![crate::inference::Q8DirectKernelChoice::Avx512],
            Q8BenchKernel::All => vec![
                crate::inference::Q8DirectKernelChoice::Scalar,
                crate::inference::Q8DirectKernelChoice::Avx2,
                crate::inference::Q8DirectKernelChoice::Avx512,
            ],
        };
        let mut paths = vec![("candle-qstorage", "candle-qmatmul", BenchPath::Candle)];
        for choice in selected {
            match crate::inference::q8_direct_kernel_backend_for(choice) {
                Ok(backend) => paths.push(("direct-native", backend, BenchPath::Direct(choice))),
                Err(error) if kernel == Q8BenchKernel::All => {
                    println!("SKIPPED backend={choice:?}: {error}");
                }
                Err(error) => return Err(error.into()),
            }
        }
        println!(
            "q8-expert-microbench d_model={d_model} d_ff={d_ff} expert_bytes={logical_bytes} production_auto_backend={}",
            crate::inference::q8_direct_kernel_backend()
        );
        println!(
            "{:<18} {:<16} {:>22} {:>20} {:>28} {:>26}",
            "path",
            "backend",
            "single-first-exec-ms",
            "single-repeated-ms",
            "top8-serial-first-exec-ms",
            "top8-serial-repeated-ms"
        );
        println!(
            "memory rows are deltas from the baseline captured before resident/pool construction"
        );
        for (name, backend, path) in paths {
            let single = lifecycle(
                path,
                1,
                slot,
                &bytes,
                &x,
                d_model,
                d_ff,
                warmup_runs,
                measured_runs,
                logical_bytes,
            )?;
            let top8 = lifecycle(
                path,
                8,
                slot,
                &bytes,
                &x,
                d_model,
                d_ff,
                warmup_runs,
                measured_runs,
                logical_bytes,
            )?;
            println!(
                "{name:<18} {backend:<16} {:>22.3} {:>20.3} {:>28.3} {:>26.3}",
                single.first_exec_ms,
                single.repeated_ms,
                top8.first_exec_ms,
                top8.repeated_ms,
            );
            for (count, result) in [(1, &single), (8, &top8)] {
                for (stage, memory) in [
                    ("after-construction", result.after_construction),
                    ("after-first-exec", result.after_first),
                    ("after-repeated", result.after_repeated),
                    ("after-explicit-drop", result.after_drop),
                ] {
                    println!(
                        "memory path={name} backend={backend} experts={count} stage={stage} resident_expert_buffer_bytes={} expert_buffer_pool_allocated_bytes={} expert_buffer_pool_primary_bytes={} expert_buffer_pool_shadow_bytes={} prepared_duplicate_expert_bytes={}",
                        memory.resident,
                        memory.pool_allocated,
                        memory.pool_primary,
                        memory.pool_shadow,
                        memory.prepared_duplicate,
                    );
                }
            }
        }
        Ok(())
    }
}

fn cmd_matvec_microbench(args: MatvecMicrobenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.measured_runs == 0 {
        return Err("matvec-microbench requires --measured-runs > 0".into());
    }
    let backends = if args.backends.is_empty() {
        vec![
            crate::parallel::DenseMatvecBackend::Matrixmultiply,
            crate::parallel::DenseMatvecBackend::Rayon,
            crate::parallel::DenseMatvecBackend::RayonMatrixmultiply,
        ]
    } else {
        args.backends.clone()
    };
    let mut shapes = vec![
        MatvecShape {
            name: "q-projection",
            rows: 32 * 128,
            cols: 2048,
        },
        MatvecShape {
            name: "k-projection",
            rows: 4 * 128,
            cols: 2048,
        },
        MatvecShape {
            name: "v-projection",
            rows: 4 * 128,
            cols: 2048,
        },
        MatvecShape {
            name: "o-projection",
            rows: 2048,
            cols: 32 * 128,
        },
        MatvecShape {
            name: "router-gate",
            rows: 128,
            cols: 2048,
        },
    ];
    if !args.skip_lm_head {
        shapes.push(MatvecShape {
            name: "lm-head",
            rows: 151_936,
            cols: 2048,
        });
    }

    let mut results = Vec::with_capacity(shapes.len() * backends.len());
    for (shape_idx, shape) in shapes.iter().enumerate() {
        let x = deterministic_f32_vec(shape.cols, 0x9e37_79b9 ^ shape_idx as u64);
        let w = deterministic_f32_vec(
            shape.rows * shape.cols,
            0xd1b5_4a32_d192_ed03u64 ^ shape_idx as u64,
        );
        for &backend in &backends {
            for _ in 0..args.warmup_runs {
                let y = crate::transformer::matmul_row_major_with_backend(
                    std::hint::black_box(&w),
                    std::hint::black_box(&x),
                    shape.rows,
                    shape.cols,
                    backend,
                );
                std::hint::black_box(&y);
            }
            let mut total = Duration::ZERO;
            let mut best = Duration::MAX;
            let mut checksum = 0u64;
            for _ in 0..args.measured_runs {
                let started = Instant::now();
                let y = crate::transformer::matmul_row_major_with_backend(
                    std::hint::black_box(&w),
                    std::hint::black_box(&x),
                    shape.rows,
                    shape.cols,
                    backend,
                );
                let elapsed = started.elapsed();
                std::hint::black_box(&y);
                checksum = checksum_f32_bits(&y);
                total += elapsed;
                best = best.min(elapsed);
            }
            results.push(MatvecMicrobenchResult {
                shape: shape.name,
                backend: backend.to_string(),
                rows: shape.rows,
                cols: shape.cols,
                multiply_accumulates: shape.rows.saturating_mul(shape.cols),
                best_ms: best.as_secs_f64() * 1_000.0,
                mean_ms: (total.as_secs_f64() * 1_000.0) / args.measured_runs as f64,
                checksum,
            });
        }
    }

    let report = MatvecMicrobenchReport {
        benchmark: "matvec-microbench",
        model: "Qwen3-Coder-30B-A3B-Instruct Q8_0",
        d_model: 2048,
        d_ff: 768,
        num_heads: 32,
        num_kv_heads: 4,
        head_dim: 128,
        vocab_size: 151_936,
        warmup_runs: args.warmup_runs,
        measured_runs: args.measured_runs,
        build: BenchRealBuildInfo {
            git_commit: git_commit_short(),
            build_features: build_features(),
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            dense_matvec_backend: crate::parallel::dense_matvec_backend().to_string(),
        },
        results,
    };
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_matvec_microbench_human(&report);
    }
    Ok(())
}

fn cmd_scratch_alloc_microbench(
    args: ScratchAllocMicrobenchArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "alloc-count"))]
    {
        let _ = args;
        Err(
            "scratch-alloc-microbench requires `--features alloc-count` so the binary installs the counting allocator"
                .into(),
        )
    }

    #[cfg(feature = "alloc-count")]
    {
        if args.measured_tokens == 0 {
            return Err("scratch-alloc-microbench requires --measured-tokens > 0".into());
        }
        if args.d_model == 0 || !args.d_model.is_multiple_of(4) {
            return Err(
                "scratch-alloc-microbench requires --d-model to be a positive multiple of 4".into(),
            );
        }
        if args.num_experts == 0 {
            return Err("scratch-alloc-microbench requires --num-experts > 0".into());
        }
        if args.top_k == 0 || args.top_k > args.num_experts {
            return Err("scratch-alloc-microbench requires 0 < --top-k <= --num-experts".into());
        }

        let backend = crate::backend::current();
        let results = vec![
            run_scratch_alloc_variant(&args, ScratchAllocVariant::CompatibilityWrappers, &backend),
            run_scratch_alloc_variant(&args, ScratchAllocVariant::ScratchBuffers, &backend),
        ];

        let report = ScratchAllocMicrobenchReport {
            benchmark: "scratch-alloc-microbench",
            model: "synthetic-transformer-layer",
            d_model: args.d_model,
            num_experts: args.num_experts,
            top_k: args.top_k,
            warmup_tokens: args.warmup_tokens,
            measured_tokens: args.measured_tokens,
            build: BenchRealBuildInfo {
                git_commit: git_commit_short(),
                build_features: build_features(),
                threads: std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
                dense_matvec_backend: crate::parallel::dense_matvec_backend().to_string(),
            },
            results,
        };
        if args.json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            print_scratch_alloc_microbench_human(&report);
        }
        Ok(())
    }
}

#[cfg(feature = "alloc-count")]
fn run_scratch_alloc_variant(
    args: &ScratchAllocMicrobenchArgs,
    variant: ScratchAllocVariant,
    backend: &crate::backend::BackendBox,
) -> ScratchAllocMicrobenchResult {
    let layer = make_synthetic_transformer_layer(args.d_model, args.num_experts, args.top_k);
    let expert_bank = make_synthetic_expert_bank(args.d_model, args.num_experts);
    let mut selected_outputs = vec![vec![0.0f32; args.d_model]; args.top_k];
    let mut hidden = deterministic_f32_vec(args.d_model, 0x7a11_0ca7_ed15_ea5e);
    let mut kv = crate::transformer::KvCache::new_kv(layer.kv_dim(), layer.v_dim());
    let mut pos = 0usize;

    let mut scratch = crate::transformer::TransformerLayerScratch::new();
    let mut next_hidden = Vec::with_capacity(args.d_model);

    for _ in 0..args.warmup_tokens {
        run_synthetic_decode_step(
            &layer,
            variant,
            &mut hidden,
            pos,
            &mut kv,
            backend,
            &expert_bank,
            &mut selected_outputs,
            &mut scratch,
            &mut next_hidden,
        );
        pos += 1;
    }

    alloc_count::reset();
    let started = Instant::now();
    for _ in 0..args.measured_tokens {
        run_synthetic_decode_step(
            &layer,
            variant,
            &mut hidden,
            pos,
            &mut kv,
            backend,
            &expert_bank,
            &mut selected_outputs,
            &mut scratch,
            &mut next_hidden,
        );
        pos += 1;
        std::hint::black_box(&hidden);
    }
    let elapsed = started.elapsed();
    let allocations = alloc_count::snapshot();
    ScratchAllocMicrobenchResult {
        variant: variant.label(),
        elapsed_ms: elapsed.as_secs_f64() * 1_000.0,
        allocations,
        allocation_calls_per_token: allocations.allocation_calls as f64
            / args.measured_tokens as f64,
        bytes_allocated_per_token: allocations.bytes_allocated as f64 / args.measured_tokens as f64,
        checksum: checksum_f32_bits(&hidden),
    }
}

#[cfg(feature = "alloc-count")]
#[allow(clippy::too_many_arguments)]
fn run_synthetic_decode_step(
    layer: &crate::transformer::TransformerLayer,
    variant: ScratchAllocVariant,
    hidden: &mut Vec<f32>,
    pos: usize,
    kv: &mut crate::transformer::KvCache,
    backend: &crate::backend::BackendBox,
    expert_bank: &[Vec<f32>],
    selected_outputs: &mut [Vec<f32>],
    scratch: &mut crate::transformer::TransformerLayerScratch,
    next_hidden: &mut Vec<f32>,
) {
    match variant {
        ScratchAllocVariant::CompatibilityWrappers => {
            let after_attn = layer.attn_block_with_timing(hidden, pos, 0, kv, backend, None);
            let (_normed, routing) = layer.moe_pre_with_timing(&after_attn, None);
            copy_selected_expert_outputs(&routing, expert_bank, selected_outputs);
            *hidden = layer.moe_combine_with_timing(
                &after_attn,
                &selected_outputs[..routing.experts.len()],
                &routing.weights,
                None,
            );
        }
        ScratchAllocVariant::ScratchBuffers => {
            layer.attn_block_into_with_timing(
                hidden,
                pos,
                0,
                kv,
                backend,
                scratch,
                next_hidden,
                None,
            );
            std::mem::swap(hidden, next_hidden);
            next_hidden.clear();

            let routing = layer.moe_pre_into_with_timing(hidden, scratch, None);
            copy_selected_expert_outputs(&routing, expert_bank, selected_outputs);
            let mut moe_accum = std::mem::take(&mut scratch.moe_accum);
            layer.moe_combine_into_with_timing(
                hidden,
                &selected_outputs[..routing.experts.len()],
                &routing.weights,
                &mut moe_accum,
                next_hidden,
                None,
            );
            scratch.moe_accum = moe_accum;
            std::mem::swap(hidden, next_hidden);
            next_hidden.clear();
            scratch.routing.recycle_decision(routing);
        }
    }
}

#[cfg(feature = "alloc-count")]
fn make_synthetic_transformer_layer(
    d_model: usize,
    num_experts: usize,
    top_k: usize,
) -> crate::transformer::TransformerLayer {
    let head_dim = 4;
    let num_heads = d_model / head_dim;
    let num_kv_heads = num_heads;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let mk = |rows: usize, cols: usize, scale: f32| {
        (0..rows * cols)
            .map(|i| ((i % 7) as f32 - 3.0) * scale)
            .collect::<Vec<f32>>()
    };
    crate::transformer::TransformerLayer {
        rms_attn: crate::transformer::RmsNorm::new(vec![1.0; d_model], 1e-6),
        attn: crate::transformer::MultiHeadSelfAttention {
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim: head_dim,
            v_head_dim: head_dim,
            attention_value_scale: None,
            rope_base: 10000.0,
            wq: crate::dense_tensor::DenseWeight::from_f32(
                mk(q_dim, d_model, 0.05),
                q_dim,
                d_model,
            ),
            wk: crate::dense_tensor::DenseWeight::from_f32(
                mk(kv_dim, d_model, 0.05),
                kv_dim,
                d_model,
            ),
            wv: crate::dense_tensor::DenseWeight::from_f32(
                mk(kv_dim, d_model, 0.05),
                kv_dim,
                d_model,
            ),
            wo: crate::dense_tensor::DenseWeight::from_f32(
                mk(d_model, q_dim, 0.05),
                d_model,
                q_dim,
            ),
            window_size: None,
            q_norm: None,
            k_norm: None,
            rope_yarn: None,
            rope_cache: None,
            bq: None,
            bk: None,
            bv: None,
            bo: None,
            sink_bias: None,
        },
        mla: None,
        rms_moe: crate::transformer::RmsNorm::new(vec![1.0; d_model], 1e-6),
        gate: crate::gating::LinearGate::new(
            mk(num_experts, d_model, 0.1),
            num_experts,
            d_model,
            top_k,
        ),
        shared_expert: None,
        dense_ffn: None,
    }
}

#[cfg(feature = "alloc-count")]
fn make_synthetic_expert_bank(d_model: usize, num_experts: usize) -> Vec<Vec<f32>> {
    (0..num_experts)
        .map(|expert| deterministic_f32_vec(d_model, 0x5eed_0000_u64 | expert as u64))
        .collect()
}

#[cfg(feature = "alloc-count")]
fn copy_selected_expert_outputs(
    routing: &crate::gating::RoutingDecision,
    expert_bank: &[Vec<f32>],
    selected_outputs: &mut [Vec<f32>],
) {
    debug_assert!(routing.experts.len() <= selected_outputs.len());
    for (idx, &expert_id) in routing.experts.iter().enumerate() {
        let source = &expert_bank[expert_id as usize % expert_bank.len()];
        selected_outputs[idx].clear();
        selected_outputs[idx].extend_from_slice(source);
    }
}

#[cfg(feature = "alloc-count")]
fn print_scratch_alloc_microbench_human(report: &ScratchAllocMicrobenchReport) {
    println!("scratch-alloc-microbench");
    println!(
        "  model={} d_model={} num_experts={} top_k={}",
        report.model, report.d_model, report.num_experts, report.top_k
    );
    println!(
        "  warmup_tokens={} measured_tokens={} git={} threads={} features={}",
        report.warmup_tokens,
        report.measured_tokens,
        report.build.git_commit,
        report.build.threads,
        report.build.build_features.join(",")
    );
    for result in &report.results {
        println!(
            "  {:<23} tokens={:<4} elapsed={:>8.3}ms allocs={:<7} reallocs={:<5} bytes={:<10} peak={:<10} allocs/token={:>7.2} bytes/token={:>9.1} checksum={:#016x}",
            result.variant,
            report.measured_tokens,
            result.elapsed_ms,
            result.allocations.allocation_calls,
            result.allocations.reallocation_calls,
            result.allocations.bytes_allocated,
            result.allocations.peak_bytes,
            result.allocation_calls_per_token,
            result.bytes_allocated_per_token,
            result.checksum
        );
    }
    if let [baseline, scratch] = report.results.as_slice() {
        println!(
            "  reduction vs wrappers: alloc_calls={:>6.2}% bytes_allocated={:>6.2}%",
            percent_reduction(
                baseline.allocations.allocation_calls,
                scratch.allocations.allocation_calls
            ),
            percent_reduction(
                baseline.allocations.bytes_allocated,
                scratch.allocations.bytes_allocated
            )
        );
    }
}

#[cfg(feature = "alloc-count")]
fn percent_reduction(baseline: u64, measured: u64) -> f64 {
    if baseline == 0 {
        return 0.0;
    }
    ((baseline as f64 - measured as f64) / baseline as f64) * 100.0
}

fn deterministic_f32_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
        let v = ((bits >> 40) as u32 & 0xffff) as f32 / 32768.0 - 1.0;
        out.push(v);
    }
    out
}

fn checksum_f32_bits(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
        (h ^ v.to_bits() as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn print_matvec_microbench_human(report: &MatvecMicrobenchReport) {
    println!("matvec-microbench");
    println!(
        "  model={} d_model={} d_ff={} heads={} kv_heads={} head_dim={} vocab={}",
        report.model,
        report.d_model,
        report.d_ff,
        report.num_heads,
        report.num_kv_heads,
        report.head_dim,
        report.vocab_size
    );
    println!(
        "  warmup_runs={} measured_runs={} git={} threads={} features={}",
        report.warmup_runs,
        report.measured_runs,
        report.build.git_commit,
        report.build.threads,
        report.build.build_features.join(",")
    );
    for result in &report.results {
        println!(
            "  {:<13} {:<22} rows={:<6} cols={:<5} best={:>9.3}ms mean={:>9.3}ms checksum={:#016x}",
            result.shape,
            result.backend,
            result.rows,
            result.cols,
            result.best_ms,
            result.mean_ms,
            result.checksum
        );
    }
}

fn load_bench_real_input(
    args: &BenchRealArgs,
) -> Result<BenchRealInput, Box<dyn std::error::Error>> {
    let input = load_real_cli_request_input(
        "bench-real",
        args.prompt.as_ref(),
        args.request_json.as_deref(),
        args.output_tokens,
    )?;
    Ok(BenchRealInput {
        prompt: input.prompt,
        output_tokens: input.output_tokens,
    })
}

fn flatten_bench_messages(messages: &[serde_json::Value]) -> String {
    let mut out = String::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("user");
        let content = message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        out.push_str(role);
        out.push_str(": ");
        out.push_str(content);
        out.push('\n');
    }
    out
}

fn bench_sampling_params(
    cfg: &crate::config::Config,
    greedy: bool,
) -> crate::sampling::SamplingParams {
    if greedy {
        crate::sampling::SamplingParams::greedy()
    } else {
        cfg.sampling.to_params()
    }
}

/// Fail-closed policy gate for `bench-real` (hardening pass, item 4):
/// every development-only fail-open flag is rejected — independently
/// and in combination — because a benchmark taken under any degraded
/// policy is not a production measurement. Factored out of
/// [`build_bench_real_runtime`] so the rejection matrix is directly
/// unit-testable.
fn validate_bench_real_policies(cfg: &crate::config::Config) -> Result<(), String> {
    if !cfg.real_transformer.enabled {
        return Err("bench-real requires [real_transformer] enabled = true".into());
    }
    if matches!(
        cfg.real_transformer.compute_offload,
        crate::backend::ComputeOffload::Gpu
            | crate::backend::ComputeOffload::Auto
            | crate::backend::ComputeOffload::Hybrid
    ) || cfg.gpu_cache.enabled
    {
        return Err(
            "bench-real is CPU-only for this sprint; set real_transformer.compute_offload = \"cpu\" (\"gpu\", \"auto\" and \"hybrid\" are rejected) and disable [gpu_cache].enabled"
                .into(),
        );
    }
    if cfg.distributed.enabled {
        return Err(
            "bench-real runs the local direct real-model path; disable distributed.enabled".into(),
        );
    }
    if cfg.real_transformer.weights_dir.is_none() {
        return Err("bench-real requires real_transformer.weights_dir; seeded fallback benchmarks are not production measurements".into());
    }
    if cfg.real_transformer.allow_seeded_fallback {
        return Err(
            "bench-real rejects real_transformer.allow_seeded_fallback = true; seeded fallback \
             benchmarks are not production measurements"
                .into(),
        );
    }
    if cfg.real_transformer.allow_degraded_experts {
        return Err(
            "bench-real rejects real_transformer.allow_degraded_experts = true; degraded-mode \
             benchmarks are not production measurements"
                .into(),
        );
    }
    if cfg.real_transformer.allow_nonfinite_attention_fallback {
        return Err(
            "bench-real rejects real_transformer.allow_nonfinite_attention_fallback = true; \
             uniform-attention fallback benchmarks are not production measurements"
                .into(),
        );
    }
    if cfg.real_transformer.allow_truncated_expert_payloads {
        return Err(
            "bench-real rejects real_transformer.allow_truncated_expert_payloads = true; \
             zero-filled truncated expert payloads are not production measurements"
                .into(),
        );
    }
    if !cfg.real_transformer.strict_weights {
        return Err(
            "bench-real requires real_transformer.strict_weights = true; a benchmark that may \
             include seeded fallback tensors is not a production measurement"
                .into(),
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RealCliRuntimeMode {
    BenchReal,
    StrictHybridQualification,
    IsolatedGreedyParityCpu,
    IsolatedGreedyParityHybrid,
    IsolatedGpuNativeDiagnostic,
    IsolatedGpuNativeBenchmark,
}

impl RealCliRuntimeMode {
    fn is_isolated(self) -> bool {
        matches!(
            self,
            Self::IsolatedGreedyParityCpu
                | Self::IsolatedGreedyParityHybrid
                | Self::IsolatedGpuNativeDiagnostic
                | Self::IsolatedGpuNativeBenchmark
        )
    }

    fn installs_logical_gpu_cache(self) -> bool {
        matches!(
            self,
            Self::StrictHybridQualification
                | Self::IsolatedGreedyParityHybrid
                | Self::IsolatedGpuNativeDiagnostic
                | Self::IsolatedGpuNativeBenchmark
        )
    }

    fn tokenizer_command(self) -> &'static str {
        match self {
            Self::BenchReal => "bench-real",
            Self::StrictHybridQualification => "qualify-hybrid-q4",
            Self::IsolatedGreedyParityCpu | Self::IsolatedGreedyParityHybrid => {
                "qualify-hybrid-q4-greedy-parity"
            }
            Self::IsolatedGpuNativeDiagnostic => "diagnose-gpu-native-q4-first-divergence",
            Self::IsolatedGpuNativeBenchmark => "bench-gpu-native-real",
        }
    }
}

#[derive(Clone)]
struct ResolvedRealCliSpec {
    cfg: crate::config::Config,
    architecture: crate::architecture::Architecture,
    first_k_dense_replace: usize,
    advanced: crate::model::AdvancedConfig,
}

fn strict_hybrid_gpu_geometry(
    cfg: &crate::config::Config,
    resolved_advanced: &crate::model::AdvancedConfig,
) -> crate::backend::GpuBackendGeometry {
    let rt = &cfg.real_transformer;
    let head_dim = if rt.head_dim == 0 {
        cfg.model.d_model / rt.num_heads
    } else {
        rt.head_dim
    };
    crate::backend::GpuBackendGeometry {
        num_layers: cfg.model.num_layers,
        max_seq_len: if rt.window_size == 0 {
            4096
        } else {
            rt.window_size
        },
        num_heads: rt.num_heads,
        num_kv_heads: rt.num_kv_heads,
        head_dim,
        // Zero preserves GpuBackendGeometry's symmetric fallback to head_dim.
        v_head_dim: resolved_advanced.v_head_dim.unwrap_or(0),
        q4_truncation_tolerance: 0,
    }
}

async fn build_bench_real_runtime(
    config_path: &Path,
) -> Result<BenchRealRuntime, Box<dyn std::error::Error>> {
    build_real_cli_runtime(config_path, RealCliRuntimeMode::BenchReal).await
}

async fn build_real_cli_runtime(
    config_path: &Path,
    mode: RealCliRuntimeMode,
) -> Result<BenchRealRuntime, Box<dyn std::error::Error>> {
    use crate::config::Config;

    let cfg = Config::from_file(config_path)?;
    let spec = resolve_real_cli_spec_from_config(cfg, mode)?;
    let execution_context = match mode {
        RealCliRuntimeMode::BenchReal => {
            crate::backend::install_default();
            let context = crate::backend::current_execution_context();
            let b = crate::backend::current();
            info!(
                backend = b.device_name(),
                compute_plane = b.compute_plane(),
                "bench-real math backend installed"
            );
            context
        }
        RealCliRuntimeMode::StrictHybridQualification => {
            let context = resolve_isolated_real_cli_context(
                &spec,
                spec.cfg.real_transformer.compute_offload,
            )?;
            crate::backend::set_execution_context(context.clone())
                .map_err(|e| format!("failed to install qualification execution context: {e}"))?;
            context
        }
        RealCliRuntimeMode::IsolatedGreedyParityCpu
        | RealCliRuntimeMode::IsolatedGreedyParityHybrid
        | RealCliRuntimeMode::IsolatedGpuNativeDiagnostic
        | RealCliRuntimeMode::IsolatedGpuNativeBenchmark => {
            return Err("isolated qualification runtimes must use the private isolated factory".into());
        }
    };
    build_real_cli_runtime_from_spec(&spec, mode, execution_context, None).await
}

fn resolve_real_cli_spec_from_config(
    mut cfg: crate::config::Config,
    mode: RealCliRuntimeMode,
) -> Result<ResolvedRealCliSpec, Box<dyn std::error::Error>> {
    crate::parallel::set_dense_matvec_backend(cfg.real_transformer.dense_matvec_backend);
    if mode == RealCliRuntimeMode::BenchReal {
        validate_bench_real_policies(&cfg)?;
    } else if mode == RealCliRuntimeMode::IsolatedGpuNativeBenchmark {
        crate::gpu_native_real_benchmark::validate_source_config(&cfg)?;
    }

    let (resolved_architecture, resolved_first_k_dense_replace, resolved_advanced) =
        reconcile_real_model_config(&mut cfg)?;
    // Finding 7: validate the fully-resolved configuration before any weights
    // are loaded or backends installed.
    validate_resolved_real_model_config(
        &cfg,
        resolved_architecture,
        resolved_first_k_dense_replace,
        &resolved_advanced,
    )?;
    Ok(ResolvedRealCliSpec {
        cfg,
        architecture: resolved_architecture,
        first_k_dense_replace: resolved_first_k_dense_replace,
        advanced: resolved_advanced,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IsolatedExecutionContextContract {
    CpuReference,
    HybridRoutedExperts,
    GpuNativeAuthoritative,
}

fn isolated_execution_context_contract(
    spec: &ResolvedRealCliSpec,
    requested: crate::backend::ComputeOffload,
) -> IsolatedExecutionContextContract {
    if spec.cfg.real_transformer.gpu_native {
        IsolatedExecutionContextContract::GpuNativeAuthoritative
    } else if requested == crate::backend::ComputeOffload::Hybrid {
        IsolatedExecutionContextContract::HybridRoutedExperts
    } else {
        IsolatedExecutionContextContract::CpuReference
    }
}

fn resolve_isolated_real_cli_context(
    spec: &ResolvedRealCliSpec,
    requested: crate::backend::ComputeOffload,
) -> Result<Arc<crate::backend::ExecutionContext>, Box<dyn std::error::Error>> {
    let cfg = &spec.cfg;
    let contract = isolated_execution_context_contract(spec, requested);
    let capacity_bytes = match contract {
        IsolatedExecutionContextContract::HybridRoutedExperts
        | IsolatedExecutionContextContract::GpuNativeAuthoritative => cfg
            .gpu_cache
            .vram_capacity_mb
            .checked_mul(1024 * 1024)
            .ok_or("gpu_cache.vram_capacity_mb overflows usize")?,
        IsolatedExecutionContextContract::CpuReference => 0,
    };
    let gpu_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(
        capacity_bytes,
        cfg.gpu_cache.vram_anchor_ratio,
        cfg.gpu_cache.promote_after_hits,
    ));
    match contract {
        IsolatedExecutionContextContract::GpuNativeAuthoritative => {
            let mut geometry = strict_hybrid_gpu_geometry(cfg, &spec.advanced);
            if cfg.real_transformer.gpu_native_max_seq_len > 0 {
                geometry.max_seq_len = cfg.real_transformer.gpu_native_max_seq_len;
            }
            Ok(crate::backend::resolve_execution_context_for_gpu_native(
                geometry,
                gpu_cache,
            )?)
        }
        IsolatedExecutionContextContract::HybridRoutedExperts
        | IsolatedExecutionContextContract::CpuReference => {
            Ok(crate::backend::resolve_execution_context(
                requested,
                true,
                strict_hybrid_gpu_geometry(cfg, &spec.advanced),
                crate::backend::RoutedExpertGpuSpec {
                    dtype: cfg.model.dtype,
                    d_model: cfg.model.d_model,
                    d_ff: cfg.model.d_ff,
                },
                gpu_cache,
            )?)
        }
    }
}

fn resolved_real_cli_spec_sha256(
    spec: &ResolvedRealCliSpec,
) -> Result<String, Box<dyn std::error::Error>> {
    resolved_real_runtime_identity_sha256(
        &spec.cfg,
        spec.architecture,
        spec.first_k_dense_replace,
        &spec.advanced,
    )
}

fn resolved_real_runtime_identity_sha256(
    cfg: &crate::config::Config,
    architecture: crate::architecture::Architecture,
    first_k_dense_replace: usize,
    advanced: &crate::model::AdvancedConfig,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut bytes = serde_json::to_vec(cfg)?;
    bytes.extend_from_slice(architecture.model_type().as_bytes());
    bytes.extend_from_slice(&(first_k_dense_replace as u64).to_le_bytes());
    bytes.extend_from_slice(format!("{advanced:?}").as_bytes());
    Ok(crate::greedy_parity::sha256_hex(&bytes))
}

/// Private qualification-only factory. It deliberately constructs an
/// execution context without installing it in the process-global OnceLock.
/// Every invocation builds a new storage/cache/predictor/model/engine family.
async fn build_isolated_greedy_runtime(
    spec: &ResolvedRealCliSpec,
    mode: RealCliRuntimeMode,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
) -> Result<BenchRealRuntime, Box<dyn std::error::Error>> {
    let requested = match mode {
        RealCliRuntimeMode::IsolatedGreedyParityCpu => crate::backend::ComputeOffload::Cpu,
        RealCliRuntimeMode::IsolatedGreedyParityHybrid => {
            crate::backend::ComputeOffload::Hybrid
        }
        RealCliRuntimeMode::IsolatedGpuNativeDiagnostic => {
            crate::backend::ComputeOffload::Gpu
        }
        RealCliRuntimeMode::IsolatedGpuNativeBenchmark => crate::backend::ComputeOffload::Gpu,
        _ => return Err("non-isolated runtime mode passed to isolated factory".into()),
    };
    let context = resolve_isolated_real_cli_context(spec, requested)?;
    build_real_cli_runtime_from_spec(spec, mode, context, Some(tokenizer)).await
}

fn load_real_cli_tokenizer(
    cfg: &crate::config::Config,
    mode: RealCliRuntimeMode,
) -> Result<Arc<crate::tokenizer::Tokenizer>, Box<dyn std::error::Error>> {
    let command = mode.tokenizer_command();
    match cfg.tokenizer.path.as_ref() {
        Some(path) => crate::tokenizer::Tokenizer::from_file(path)
            .map(Arc::new)
            .map_err(|error| {
                format!(
                    "{command} failed to load tokenizer {}: {error}",
                    path.display()
                )
                .into()
            }),
        None if mode == RealCliRuntimeMode::BenchReal => Err(
            "bench-real requires tokenizer.path; byte tokenizer fallback is not a production benchmark"
                .into(),
        ),
        None => Err(format!(
            "{command} requires tokenizer.path; byte tokenizer fallback is not permitted"
        )
        .into()),
    }
}

async fn build_real_cli_runtime_from_spec(
    spec: &ResolvedRealCliSpec,
    mode: RealCliRuntimeMode,
    execution_context: Arc<crate::backend::ExecutionContext>,
    tokenizer_override: Option<Arc<crate::tokenizer::Tokenizer>>,
) -> Result<BenchRealRuntime, Box<dyn std::error::Error>> {
    use crate::metrics::Metrics;

    let cfg = spec.cfg.clone();
    let resolved_architecture = spec.architecture;
    let resolved_first_k_dense_replace = spec.first_k_dense_replace;
    let resolved_advanced = spec.advanced.clone();

    if !cfg.model.data_dir.is_dir() {
        return Err(format!(
            "data dir {} does not exist; run `gen-data` or the extractor first",
            cfg.model.data_dir.display()
        )
        .into());
    }
    crate::gguf_loader::validate_q4_0_dataset_layout(&cfg.model.data_dir, cfg.model.dtype)?;

    let storage = NvmeStorage::new(StorageConfig {
        base_path: cfg.model.data_dir.clone(),
        expert_size: cfg.model.expert_size,
        block_align: cfg.storage.block_align,
        use_direct_io: !cfg.storage.no_direct,
        num_experts_per_layer: if cfg.model.num_layers > 1 {
            Some(cfg.model.num_experts)
        } else {
            None
        },
    })?;
    let storage = maybe_attach_packed_blob(
        storage,
        cfg.storage.packed_blob.as_deref(),
        cfg.storage.packed_manifest.as_deref(),
        !cfg.storage.no_direct,
        cfg.model.expert_size,
    )?;
    let storage = Arc::new(storage);
    let storage_shutdown = Arc::downgrade(&storage);
    let total_experts_for_files =
        (cfg.model.num_layers as u32).saturating_mul(cfg.model.num_experts);
    if !storage.is_packed() {
        storage.warmup_fds(0..total_experts_for_files)?;
    }
    // Root-cause guard (hardening pass, Part A2): reject an
    // `expert_size` that disagrees with the on-disk file layout before
    // any truncated payload can be read.
    storage.validate_expert_file_layout(0..total_experts_for_files.min(8))?;

    let pipeline_depth = cfg.storage.pipeline_depth.max(1) as usize;
    let shadow_slots = cfg.storage.predict_fanout.saturating_mul(pipeline_depth);
    let primary_slots = cfg.storage.cache_slots + 1;
    let pool = if shadow_slots > 0 {
        BufferPool::new_with_shadow(
            primary_slots,
            shadow_slots,
            cfg.model.expert_size,
            cfg.storage.block_align,
        )
    } else {
        BufferPool::new(
            primary_slots,
            cfg.model.expert_size,
            cfg.storage.block_align,
        )
    };
    let cache = {
        let num_layers = cfg.model.num_layers.max(1);
        let per_layer = cfg.model.num_experts.max(1);
        let total = cfg.storage.cache_slots.max(1);
        let base = total / num_layers;
        let extra = total % num_layers;
        let caps: Vec<usize> = (0..num_layers)
            .map(|i| base + if i < extra { 1 } else { 0 })
            .collect();
        if num_layers == 1 {
            Arc::new(MultiLayerExpertCache::single_layer(total))
        } else {
            Arc::new(MultiLayerExpertCache::with_capacities(caps, per_layer))
        }
    };
    let cache_shutdown = Arc::downgrade(&cache);
    let total_experts: u32 = (cfg.model.num_layers as u32)
        .saturating_mul(cfg.model.num_experts)
        .max(cfg.model.num_experts);
    let predictor = Arc::new(PredictiveLoader::new(
        total_experts,
        cfg.storage.predict_fanout,
        resolve_predict_min_prob(cfg.storage.predict_min_prob, total_experts),
        0xC0FFEE,
    ));
    let predictor_shutdown = Arc::downgrade(&predictor);

    let rt = &cfg.real_transformer;
    let head_dim = if rt.head_dim == 0 {
        cfg.model.d_model / rt.num_heads
    } else {
        rt.head_dim
    };
    let num_kv_heads = if rt.num_kv_heads == 0 {
        rt.num_heads
    } else {
        rt.num_kv_heads
    };
    let model_cfg = crate::model::RealModelConfig {
        d_model: cfg.model.d_model,
        d_ff: cfg.model.d_ff,
        num_heads: rt.num_heads,
        num_kv_heads,
        head_dim,
        vocab_size: rt.vocab_size,
        num_layers: cfg.model.num_layers,
        num_experts: cfg.model.num_experts as usize,
        top_k: cfg.model.top_k,
        rope_base: rt.rope_base,
        rms_eps: rt.rms_eps,
        window_size: if rt.window_size == 0 {
            None
        } else {
            Some(rt.window_size)
        },
        architecture: resolved_architecture,
        first_k_dense_replace: resolved_first_k_dense_replace,
        advanced: resolved_advanced,
    };
    let model = Arc::new(crate::model::RealModel::from_dir_auto_with_options(
        model_cfg,
        rt.weights_dir.as_ref().expect("weights_dir checked above"),
        rt.seed,
        crate::model::RealModelLoadOptions {
            strict_weights: rt.strict_weights,
        },
    )?);
    if model.load_status.seeded_fallback_remained {
        return Err(format!(
            "bench-real loaded an incomplete checkpoint: {} of {} required tensors loaded, \
             seeded fallback remained; not a production measurement",
            model.load_status.loaded_tensors, model.load_status.required_tensors
        )
        .into());
    }
    info!(
        strict = model.load_status.strict,
        loader = model.load_status.loader,
        loaded_tensors = model.load_status.loaded_tensors,
        required_tensors = model.load_status.required_tensors,
        seeded_fallback_remained = model.load_status.seeded_fallback_remained,
        "bench-real model-loading status"
    );

    info!(
        num_experts = model.layers[0].gate.num_experts,
        d_model = model.layers[0].gate.d_model,
        top_k = model.layers[0].gate.top_k,
        "bench-real routing: LinearGate wired from real model"
    );
    let router = crate::gating::Router::Linear(Arc::new(model.layers[0].gate.clone()));
    let metrics = Metrics::new();
    let execution_context_shutdown = Arc::downgrade(&execution_context);
    let gpu_cache_shutdown = Arc::downgrade(execution_context.gpu_expert_cache());
    let mut speculator_shutdown = None;
    let mut affinity_shutdown = None;
    let mut engine_builder = Engine::with_options_and_execution_context(
        cache.clone(),
        pool,
        storage,
        router,
        predictor,
        ModelShape {
            d_model: cfg.model.d_model,
            d_ff: cfg.model.d_ff,
            hidden_seed: 0xC0FFEE,
        },
        EngineOptions {
            io_only: false,
            dtype: cfg.model.dtype,
            partial_load_fraction: cfg.storage.partial_load_fraction,
            pin_after_observations: cfg.storage.pin_after_observations,
            use_qmm_for_q4: true,
            expert_execution_policy: cfg.real_transformer.expert_execution_policy,
            max_concurrent_prefetches: cfg.real_transformer.max_concurrent_prefetches,
            max_fetch_yields: cfg.real_transformer.max_fetch_yields,
            prefetch_governor: cfg.predictive.prefetch_governor,
            prefetch_precision_floor: cfg.predictive.prefetch_precision_floor,
            prefetch_contention_weight: cfg.predictive.prefetch_contention_weight,
            cost_aware_eviction: cfg.predictive.cost_aware_eviction,
            pregate_enabled: cfg.predictive.pregate_enabled,
            collect_route_profile: false,
            policy: cfg.real_transformer.inference_policy(),
        },
        execution_context.clone(),
    );
    engine_builder = engine_builder.with_pipeline_depth(cfg.storage.pipeline_depth);
    if cfg.predictive.locality_enabled {
        let window = cfg
            .predictive
            .locality_window
            .saturating_mul(cfg.model.num_layers.max(1));
        let monitor = Arc::new(LocalityMonitor::new(total_experts, window));
        engine_builder =
            engine_builder.with_locality_monitor(monitor, cfg.predictive.locality_threshold_pct);
    }
    if cfg.predictive.speculator_enabled {
        let top_k = if cfg.predictive.speculator_top_k == 0 {
            cfg.model.top_k
        } else {
            cfg.predictive.speculator_top_k
        };
        let spec = Arc::new(NeuralSpeculator::new(
            cfg.model.d_model,
            cfg.predictive.speculator_hidden_dim,
            total_experts,
            0xC0FFEE,
        ));
        speculator_shutdown = Some(Arc::downgrade(&spec));
        engine_builder = engine_builder.with_speculator(spec, top_k);
    }
    if cfg.predictive.affinity_enabled {
        let affinity = Arc::new(LayeredExpertAffinity::new(
            cfg.model.num_layers.max(1),
            cfg.model.num_experts,
        ));
        affinity_shutdown = Some(Arc::downgrade(&affinity));
        engine_builder = engine_builder.with_affinity(
            affinity,
            cfg.predictive.affinity_neighbors_k,
            cfg.predictive.affinity_decay_epoch,
        );
    }
    if cfg.predictive.static_residency_fraction > 0.0 {
        let profile = match cfg.predictive.static_residency_profile.as_ref() {
            Some(path) => Some(crate::residency::ResidencyProfile::load_json(
                std::path::Path::new(path),
            )?),
            None => None,
        };
        engine_builder = engine_builder.with_static_residency(
            cfg.predictive.static_residency_fraction,
            cfg.predictive.static_residency_warmup_tokens,
            profile,
        );
    }
    if cfg.predictive.pregate_enabled {
        let pregate = Arc::new(crate::pregate::PerLayerPreGate::new(
            cfg.model.num_layers.max(1),
            cfg.model.top_k,
        ));
        engine_builder = engine_builder.with_pregate(pregate);
    }
    if mode.installs_logical_gpu_cache() {
        engine_builder.install_gpu_cache();
        engine_builder = engine_builder.with_routed_expert_gpu_failure_policy(
            crate::engine::RoutedExpertGpuFailurePolicy::StrictFailClosed,
        );
    }
    let (engine, gpu_native_token_loop) = if cfg.real_transformer.gpu_native {
        let executor = Arc::new(
            execution_context
                .create_gpu_native_executor_context(cfg.model.d_model)
                .map_err(|e| format!("failed to create GPU-native executor context: {e}"))?,
        );
        let total_expert_budget = (cfg.gpu_cache.vram_capacity_mb as u64) * 1024 * 1024;
        let expert_geom = crate::backend::gpu_native::GpuNativeQ4ExpertGeometry::try_new(
            cfg.model.d_model,
            cfg.model.d_ff,
            cfg.model.num_experts as usize,
            cfg.model.top_k,
        )
        .map_err(|e| format!("invalid GPU-native Q4 expert geometry: {e}"))?;
        let residency_manager = Arc::new(
            crate::gpu_native_residency::GpuNativeTieredResidencyManager::try_new(
                executor.clone(),
                execution_context.gpu_expert_cache().clone(),
                cfg.model.num_layers,
                expert_geom,
                total_expert_budget,
            )
            .map_err(|e| format!("failed to initialize tiered residency manager: {e}"))?,
        );
        engine_builder
            .install_gpu_native_residency_manager(residency_manager.clone())
            .map_err(|e| format!("failed to install tiered residency manager on engine: {e}"))?;
        let token_loop = crate::gpu_native_token_loop::GpuNativeTokenLoop::try_new(
            executor,
            residency_manager,
            &model,
            cfg.real_transformer.gpu_native_max_seq_len,
        )
        .map_err(|e| format!("failed to initialize GPU-native token loop: {e}"))?;
        let engine = Arc::new(engine_builder.with_metrics(metrics));
        (engine, Some(token_loop))
    } else {
        let engine = Arc::new(engine_builder.with_metrics(metrics));
        (engine, None)
    };

    let tokenizer = match tokenizer_override {
        Some(tokenizer) => tokenizer,
        None => load_real_cli_tokenizer(&cfg, mode)?,
    };
    // The tokenizer must be addressable by the reconciled model vocabulary
    // (Finding 4): every emittable token id must be < model.vocab_size.
    tokenizer
        .validate_vocab_compat(model.config.vocab_size)
        .map_err(|e| -> Box<dyn std::error::Error> {
            format!(
                "{} tokenizer is incompatible with the model: {e}",
                mode.tokenizer_command()
            )
            .into()
        })?;

    let isolated_shutdown = mode.is_isolated().then(|| IsolatedRuntimeShutdownWitness {
        engine: Arc::downgrade(&engine),
        model: Arc::downgrade(&model),
        cache: cache_shutdown,
        storage: storage_shutdown,
        predictor: predictor_shutdown,
        execution_context: execution_context_shutdown,
        gpu_cache: gpu_cache_shutdown,
        speculator: speculator_shutdown,
        affinity: affinity_shutdown,
    });

    Ok(BenchRealRuntime {
        cfg,
        engine,
        model,
        tokenizer,
        gpu_native_token_loop,
        isolated_cache: mode.is_isolated().then_some(cache),
        isolated_shutdown,
    })
}

/// Reconcile the resolved real-model configuration from the TOML config and,
/// when present, the checkpoint `config.json`. Shared by `build_bench_real_runtime`
/// and `cmd_serve` (Finding 2) so serving and bench-real resolve identically.
///
/// The checkpoint `config.json` is always parsed when `weights_dir` points at a
/// directory containing one, regardless of whether the TOML pins an
/// architecture. An explicit TOML architecture never suppresses checkpoint
/// reconciliation; if both are present they must agree or reconciliation fails
/// naming both.
fn reconcile_real_model_config(
    cfg: &mut crate::config::Config,
) -> Result<
    (
        crate::architecture::Architecture,
        usize,
        crate::model::AdvancedConfig,
    ),
    Box<dyn std::error::Error>,
> {
    let mut resolved_architecture = crate::architecture::Architecture::Mixtral;
    let mut resolved_first_k_dense_replace = 0usize;
    let mut resolved_advanced = crate::model::AdvancedConfig::default();

    // Parse the TOML-declared architecture (if any) up front. It is used to
    // seed the default resolution and to cross-check against a checkpoint
    // `config.json` — it must never suppress checkpoint reconciliation.
    let toml_architecture = match cfg.real_transformer.architecture.clone() {
        Some(arch_str) => Some(
            crate::architecture::Architecture::from_model_type(&arch_str).ok_or_else(|| {
                format!(
                    "[real_transformer] architecture = \"{arch_str}\" is not a recognised model_type"
                )
            })?,
        ),
        None => None,
    };
    if let Some(arch) = toml_architecture {
        resolved_architecture = arch;
    }

    // Always reconcile from the checkpoint's `config.json` when it exists,
    // regardless of whether the TOML pins an architecture. This keeps
    // checkpoint-specific advanced routing semantics (norm_topk_prob,
    // scoring_func, routed_scaling_factor, n_group, topk_group,
    // num_shared_experts, rope_scaling, …) authoritative even when the user
    // points `weights_dir` at original safetensors.
    if let Some(dir) = cfg.real_transformer.weights_dir.clone() {
        match crate::architecture::HfConfig::from_dir(&dir) {
            Ok(Some(hf)) => {
                info!(
                    architecture = ?hf.architecture,
                    "config.json detected; reconciling bench-real hyperparameters from checkpoint"
                );
                // If TOML also pinned an architecture, it must agree with the
                // checkpoint. Silently preferring one over the other risks
                // loading a checkpoint under the wrong routing/attention path.
                if let Some(toml_arch) = toml_architecture {
                    if toml_arch != hf.architecture {
                        return Err(format!(
                            "[real_transformer] architecture = \"{}\" conflicts with checkpoint \
                             config.json architecture \"{}\" in {}; remove the TOML override or \
                             point weights_dir at a matching checkpoint",
                            toml_arch.model_type(),
                            hf.architecture.model_type(),
                            dir.display()
                        )
                        .into());
                    }
                }
                resolved_architecture = hf.architecture;
                resolved_first_k_dense_replace = hf.first_k_dense_replace.unwrap_or(0);
                resolved_advanced = crate::model::RealModelConfig::from_hf_config(&hf).advanced;
                crate::inference::set_swiglu_limit(resolved_advanced.swiglu_limit);
                cfg.model.d_model = hf.hidden_size;
                cfg.model.d_ff = hf.resolved_d_ff();
                cfg.model.num_layers = hf.num_hidden_layers;
                cfg.model.num_experts = hf.num_routed_experts.unwrap_or(1).max(1) as u32;
                cfg.model.top_k = hf
                    .num_experts_per_tok
                    .unwrap_or(1)
                    .clamp(1, cfg.model.num_experts.max(1) as usize);
                cfg.real_transformer.vocab_size = hf.vocab_size;
                cfg.real_transformer.num_heads = hf.num_attention_heads;
                cfg.real_transformer.num_kv_heads = if hf.num_key_value_heads == 0 {
                    hf.num_attention_heads
                } else {
                    hf.num_key_value_heads
                };
                cfg.real_transformer.head_dim = hf.resolved_head_dim();
                cfg.real_transformer.rope_base = hf.rope_theta;
                cfg.real_transformer.rms_eps = hf.rms_norm_eps;
                cfg.real_transformer.window_size = hf.sliding_window.unwrap_or(0);
            }
            // No `config.json` present: preserve TOML-only behaviour, using the
            // TOML architecture already resolved above (or the Mixtral default).
            Ok(None) => {}
            Err(e) => {
                return Err(
                    format!("failed to read config.json from {}: {e}", dir.display()).into(),
                );
            }
        }
    }
    Ok((
        resolved_architecture,
        resolved_first_k_dense_replace,
        resolved_advanced,
    ))
}

/// Finding 7: post-reconciliation, architecture-aware validation of the
/// fully-resolved real-model configuration. Shared by `build_bench_real_runtime`
/// and `cmd_serve` (Finding 2) so both paths reject the same invalid configs.
///
/// This runs *after* [`reconcile_real_model_config`] has merged the TOML and
/// checkpoint `config.json` into `cfg`. It validates the final resolved
/// architecture, model shape, routing configuration, storage/cache
/// relationships, and the integer conversions the loader will subsequently
/// perform.
///
/// It deliberately does **not** reuse the generic [`crate::config::Config`]
/// validation's universal `num_heads * head_dim == d_model` rule: that
/// identity is invalid for several supported architectures (MLA decomposes
/// Q/K/V through low-rank latents, and MiMo-V2-Flash uses an asymmetric
/// `v_head_dim != head_dim` with a partial-rotary Q/K width), so applying it
/// would reject valid checkpoints.
fn validate_resolved_real_model_config(
    cfg: &crate::config::Config,
    arch: crate::architecture::Architecture,
    first_k_dense_replace: usize,
    advanced: &crate::model::AdvancedConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut errs: Vec<String> = Vec::new();

    let d_model = cfg.model.d_model;
    let d_ff = cfg.model.d_ff;
    let num_layers = cfg.model.num_layers;
    let num_experts = cfg.model.num_experts as usize;
    let top_k = cfg.model.top_k;
    let vocab_size = cfg.real_transformer.vocab_size;
    let num_heads = cfg.real_transformer.num_heads;
    let num_kv_heads = cfg.real_transformer.num_kv_heads;
    let head_dim = cfg.real_transformer.head_dim;
    let v_head_dim = advanced.v_head_dim.unwrap_or(head_dim);

    // ---- Resolved model shape ----
    if d_model == 0 {
        errs.push("resolved d_model is 0".to_string());
    }
    if d_ff == 0 {
        errs.push("resolved d_ff is 0".to_string());
    }
    if num_layers == 0 {
        errs.push("resolved num_layers is 0".to_string());
    }
    if vocab_size == 0 {
        errs.push("resolved vocab_size is 0".to_string());
    }
    if num_heads == 0 {
        errs.push("resolved num_heads is 0".to_string());
    }
    if head_dim == 0 {
        errs.push("resolved head_dim is 0".to_string());
    }
    if v_head_dim == 0 {
        errs.push("resolved v_head_dim is 0".to_string());
    }

    // ---- Architecture-aware attention head geometry ----
    // MLA carries its own low-rank projection dims and does not obey the
    // dense GQA head arithmetic; validate the GQA invariants only for the
    // standard dense-attention families.
    if advanced.mla.is_none() {
        if num_kv_heads == 0 {
            errs.push("resolved num_kv_heads is 0".to_string());
        } else if num_heads % num_kv_heads != 0 {
            errs.push(format!(
                "grouped-query attention requires num_heads ({num_heads}) to be a multiple of \
                 num_kv_heads ({num_kv_heads})"
            ));
        }
        // A separate SWA KV-head count (MiMo-V2-Flash) must divide num_heads too.
        if let Some(swa_kv) = advanced.swa_num_key_value_heads {
            if swa_kv == 0 || num_heads % swa_kv != 0 {
                errs.push(format!(
                    "swa_num_key_value_heads ({swa_kv}) must be a non-zero divisor of num_heads \
                     ({num_heads})"
                ));
            }
        }
    }

    // ---- Routing configuration ----
    if num_experts == 0 {
        errs.push("resolved num_experts is 0".to_string());
    }
    if top_k == 0 {
        errs.push("resolved top_k is 0".to_string());
    }
    if top_k > num_experts {
        errs.push(format!(
            "top_k ({top_k}) exceeds num_experts ({num_experts})"
        ));
    }
    if first_k_dense_replace > num_layers {
        errs.push(format!(
            "first_k_dense_replace ({first_k_dense_replace}) exceeds num_layers ({num_layers})"
        ));
    }
    // Group-limited routing (DeepSeek-V3 `n_group`/`topk_group`).
    if advanced.n_group > 1 {
        if num_experts % advanced.n_group != 0 {
            errs.push(format!(
                "group-limited routing requires num_experts ({num_experts}) to be a multiple of \
                 n_group ({})",
                advanced.n_group
            ));
        }
        if advanced.topk_group == 0 || advanced.topk_group > advanced.n_group {
            errs.push(format!(
                "topk_group ({}) must be in 1..=n_group ({})",
                advanced.topk_group, advanced.n_group
            ));
        }
    }

    // ---- Storage / cache relationships ----
    let cache_slots = cfg.storage.cache_slots;
    let block_align = cfg.storage.block_align;
    let expert_size = cfg.model.expert_size;
    // The per-layer routed-expert LRU must be able to hold every *routed*
    // expert activated for a single token (the top-K). Finding 7: always-on
    // shared experts are loaded resident in the model (`layers[l].shared_expert`)
    // and computed directly — they are never streamed through the routed
    // SSD/expert cache, so they must NOT be added to this working-set
    // requirement. Doing so would reject an otherwise-adequate `cache_slots`.
    if cache_slots < top_k {
        errs.push(format!(
            "storage.cache_slots ({cache_slots}) is smaller than the routed experts activated \
             per layer (top_k {top_k}); the cache cannot hold one layer's routed working set"
        ));
    }
    if expert_size == 0 {
        errs.push("model.expert_size is 0".to_string());
    }
    if !cfg.storage.no_direct {
        if block_align == 0 {
            errs.push("storage.block_align is 0 while O_DIRECT is enabled".to_string());
        } else if expert_size % block_align != 0 {
            errs.push(format!(
                "model.expert_size ({expert_size}) must be a multiple of storage.block_align \
                 ({block_align}) for O_DIRECT reads (set storage.no_direct = true to relax)"
            ));
        }
    }

    // ---- Integer conversions the loader will perform ----
    // These products index/allocate resident tensors; a usize overflow here
    // would silently wrap during loading. Validate they are representable.
    let q_dim = num_heads.checked_mul(head_dim);
    if q_dim.is_none() {
        errs.push(format!(
            "num_heads ({num_heads}) * head_dim ({head_dim}) overflows usize"
        ));
    }
    if d_model.checked_mul(vocab_size).is_none() {
        errs.push(format!(
            "d_model ({d_model}) * vocab_size ({vocab_size}) overflows usize (embedding size)"
        ));
    }
    if let Some(q) = q_dim {
        if q.checked_mul(d_model).is_none() {
            errs.push("q_proj element count overflows usize".to_string());
        }
    }
    if d_model.checked_mul(d_ff).is_none() {
        errs.push(format!(
            "d_model ({d_model}) * d_ff ({d_ff}) overflows usize (dense FFN size)"
        ));
    }
    if num_experts > u32::MAX as usize {
        errs.push(format!("num_experts ({num_experts}) does not fit in u32"));
    }

    if errs.is_empty() {
        info!(
            architecture = %arch.model_type(),
            d_model,
            num_layers,
            num_experts,
            top_k,
            "bench-real resolved configuration passed architecture-aware validation"
        );
        Ok(())
    } else {
        Err(format!(
            "resolved bench-real configuration for architecture \"{}\" failed validation:\n  - {}",
            arch.model_type(),
            errs.join("\n  - ")
        )
        .into())
    }
}

async fn run_bench_real_once(
    runtime: &BenchRealRuntime,
    prompt: &str,
    output_tokens: usize,
    params: crate::sampling::SamplingParams,
    run_index: usize,
) -> Result<BenchRealRunReport, Box<dyn std::error::Error>> {
    let prompt_ids = runtime.tokenizer.encode(prompt)?;
    if prompt_ids.is_empty() {
        return Err("bench-real prompt encoded to zero tokens".into());
    }
    Ok(
        run_real_once_from_token_ids(runtime, &prompt_ids, output_tokens, params, run_index)
            .await?
            .report,
    )
}

struct PreTokenizedRealRun {
    report: BenchRealRunReport,
    initial_kv_sequence_lengths: Vec<usize>,
}

/// Execute the existing production-authentic real-model path from caller-owned
/// token IDs. The fixed-corpus qualifier uses this seam to prove both planes
/// receive the exact same single tokenization result.
async fn run_real_once_from_token_ids(
    runtime: &BenchRealRuntime,
    prompt_ids: &[u32],
    output_tokens: usize,
    params: crate::sampling::SamplingParams,
    run_index: usize,
) -> Result<PreTokenizedRealRun, Box<dyn std::error::Error>> {
    run_real_once_from_token_ids_internal(
        runtime,
        prompt_ids,
        output_tokens,
        params,
        run_index,
        None,
    )
    .await
}

/// Sole pre-tokenized inference implementation. The private logit diagnostic
/// may copy the exact first-token row-dot results immediately before the
/// unchanged production greedy selector; all normal callers pass `None`.
async fn run_real_once_from_token_ids_internal(
    runtime: &BenchRealRuntime,
    prompt_ids: &[u32],
    output_tokens: usize,
    params: crate::sampling::SamplingParams,
    run_index: usize,
    first_token_logit_bits: Option<&mut Vec<u32>>,
) -> Result<PreTokenizedRealRun, Box<dyn std::error::Error>> {
    if prompt_ids.is_empty() {
        return Err("pre-tokenized real-model prompt contains zero tokens".into());
    }
    let stage_timings = crate::stage_timing::StageTimings::default();
    let mut kv = runtime.model.fresh_kv_caches();
    let initial_kv_sequence_lengths = kv.iter().map(|cache| cache.seq_len).collect();
    let pre = runtime.engine.report();
    let total_started = Instant::now();
    let prompt_started = Instant::now();
    let mut pos = 0usize;
    let mut forward_evaluations = 0usize;
    let mut lm_head_evaluations = 0usize;
    let mut completion_ids = Vec::with_capacity(output_tokens);
    let mut decode_latencies_us = Vec::with_capacity(output_tokens.saturating_sub(1));

    for &tid in &prompt_ids[..prompt_ids.len().saturating_sub(1)] {
        runtime
            .model
            .forward_token_hidden_with_timing(
                &runtime.engine,
                tid,
                pos,
                &mut kv,
                Some(&stage_timings),
            )
            .await?;
        forward_evaluations += 1;
        pos += 1;
    }

    let final_prompt = *prompt_ids.last().expect("prompt_ids checked non-empty");
    let final_prompt_pos = pos;
    let first_started = Instant::now();
    let final_hidden = runtime
        .model
        .forward_token_hidden_with_timing(
            &runtime.engine,
            final_prompt,
            final_prompt_pos,
            &mut kv,
            Some(&stage_timings),
        )
        .await?;
    forward_evaluations += 1;
    pos += 1;
    let prompt_elapsed = prompt_started.elapsed();
    stage_timings.record(crate::stage_timing::TOTAL_PROMPT, prompt_elapsed);
    let prompt_seconds = prompt_elapsed.as_secs_f64();
    if let Some(bits) = first_token_logit_bits {
        if !bits.is_empty() {
            return Err("first-token logit capture destination was not empty".into());
        }
        bits.extend(
            runtime
                .model
                .lm_head
                .diagnostic_greedy_logits(&final_hidden)
                .into_iter()
                .map(f32::to_bits),
        );
    }
    let first = runtime.model.sample_hidden_with_timing(
        &final_hidden,
        &params,
        final_prompt_pos,
        Some(&stage_timings),
    );
    lm_head_evaluations += 1;
    let _first_token_latency_us = first_started.elapsed().as_micros() as u64;
    let time_to_first_token_seconds = total_started.elapsed().as_secs_f64();
    completion_ids.push(first);

    let decode_started = Instant::now();
    let mut last = first;
    while completion_ids.len() < output_tokens {
        let step_started = Instant::now();
        let next = runtime
            .model
            .decode_step_with_timing(
                &runtime.engine,
                last,
                pos,
                &mut kv,
                &params,
                Some(&stage_timings),
            )
            .await?;
        forward_evaluations += 1;
        lm_head_evaluations += 1;
        decode_latencies_us.push(step_started.elapsed().as_micros() as u64);
        completion_ids.push(next);
        last = next;
        pos += 1;
    }
    let decode_elapsed = decode_started.elapsed();
    stage_timings.record(crate::stage_timing::TOTAL_DECODE, decode_elapsed);
    let decode_seconds = decode_elapsed.as_secs_f64();
    let total_seconds = total_started.elapsed().as_secs_f64();
    debug_assert_eq!(
        forward_evaluations,
        bench_real_expected_forward_evaluations(prompt_ids.len(), output_tokens)
    );
    debug_assert_eq!(lm_head_evaluations, output_tokens);

    let post = runtime.engine.report();
    let cache_hits = post.hits.saturating_sub(pre.hits);
    let cache_misses = post.misses.saturating_sub(pre.misses);
    let total_lookups = cache_hits + cache_misses;
    let hit_rate = if total_lookups == 0 {
        0.0
    } else {
        cache_hits as f64 / total_lookups as f64
    };
    let ssd_bytes = post.bytes_read.saturating_sub(pre.bytes_read);
    let ssd_stall_us = post
        .predictive
        .ssd_stall_us
        .saturating_sub(pre.predictive.ssd_stall_us);
    decode_latencies_us.sort_unstable();
    let output_text = runtime.tokenizer.decode(&completion_ids)?;
    let stage_timings = stage_timings.snapshot();

    let report = BenchRealRunReport {
        run_index,
        prompt_tokens: prompt_ids.len(),
        completion_tokens: output_tokens,
        total_api_tokens: prompt_ids.len() + output_tokens,
        model_forward_evaluations: forward_evaluations,
        lm_head_evaluations,
        prompt_seconds,
        prompt_tps: rate_per_second(prompt_ids.len(), prompt_seconds),
        decode_seconds,
        decode_tps: rate_per_second(output_tokens.saturating_sub(1), decode_seconds),
        time_to_first_token_seconds,
        total_seconds,
        decode_token_latency_p50_ms: percentile_us_to_ms(&decode_latencies_us, 0.50),
        decode_token_latency_p95_ms: percentile_us_to_ms(&decode_latencies_us, 0.95),
        decode_token_latency_p99_ms: percentile_us_to_ms(&decode_latencies_us, 0.99),
        decode_token_latency_max_ms: decode_latencies_us.last().copied().unwrap_or(0) as f64
            / 1000.0,
        cache_hits,
        cache_misses,
        hit_rate,
        ssd_bytes,
        ssd_stall_seconds: ssd_stall_us as f64 / 1_000_000.0,
        rss_bytes: current_rss_bytes(),
        output_token_ids: completion_ids,
        output_text,
        stage_timings,
    };
    Ok(PreTokenizedRealRun {
        report,
        initial_kv_sequence_lengths,
    })
}

fn emit_bench_real_report(
    args: &BenchRealArgs,
    input: BenchRealInput,
    runs: Vec<BenchRealRunReport>,
) -> Result<(), Box<dyn std::error::Error>> {
    let suite = BenchRealSuiteReport {
        benchmark: "bench-real",
        config: args.config.display().to_string(),
        prompt: input.prompt,
        warmup_runs: args.warmup_runs,
        measured_runs: args.measured_runs,
        cache_reset: args.cache_reset,
        greedy: args.greedy,
        build: BenchRealBuildInfo {
            git_commit: git_commit_short(),
            build_features: build_features(),
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            dense_matvec_backend: crate::parallel::dense_matvec_backend().to_string(),
        },
        aggregate: aggregate_bench_real(&runs),
        runs,
    };
    match args.format {
        BenchRealOutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&suite)?);
        }
        BenchRealOutputFormat::Human => print_bench_real_human(&suite),
    }
    Ok(())
}

fn print_bench_real_human(suite: &BenchRealSuiteReport) {
    println!("bench-real");
    println!("  config: {}", suite.config);
    println!(
        "  warmup_runs={} measured_runs={} cache_reset={:?} greedy={}",
        suite.warmup_runs, suite.measured_runs, suite.cache_reset, suite.greedy
    );
    println!(
        "  build: git={} threads={} dense_matvec_backend={} features={}",
        suite.build.git_commit,
        suite.build.threads,
        suite.build.dense_matvec_backend,
        suite.build.build_features.join(",")
    );
    for run in &suite.runs {
        println!(
            "  run {}: prompt_tokens={} completion_tokens={} forwards={} lm_heads={}",
            run.run_index,
            run.prompt_tokens,
            run.completion_tokens,
            run.model_forward_evaluations,
            run.lm_head_evaluations
        );
        println!(
            "    prompt={:.3}s ({:.3} tok/s) ttft={:.3}s decode={:.3}s ({:.3} tok/s) total={:.3}s",
            run.prompt_seconds,
            run.prompt_tps,
            run.time_to_first_token_seconds,
            run.decode_seconds,
            run.decode_tps,
            run.total_seconds
        );
        println!(
            "    decode latency: p50={:.3}ms p95={:.3}ms p99={:.3}ms max={:.3}ms",
            run.decode_token_latency_p50_ms,
            run.decode_token_latency_p95_ms,
            run.decode_token_latency_p99_ms,
            run.decode_token_latency_max_ms
        );
        println!(
            "    cache: hits={} misses={} hit_rate={:.2}% ssd_bytes={} ssd_stall={:.3}s rss={}",
            run.cache_hits,
            run.cache_misses,
            run.hit_rate * 100.0,
            run.ssd_bytes,
            run.ssd_stall_seconds,
            run.rss_bytes
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );
        if run.stage_timings.is_empty() {
            println!("    stage timing: unavailable until stage-level timers are enabled");
        } else {
            println!("    stage timing:");
            for (stage, timing) in &run.stage_timings {
                println!(
                    "      {:<32} total={:.6}s count={} mean={:.6}s max={:.6}s",
                    stage,
                    timing.total_seconds,
                    timing.count,
                    timing.mean_seconds,
                    timing.max_seconds
                );
            }
        }
    }
    println!(
        "  aggregate: prompt_tps_mean={:.3} decode_tps_mean={:.3} ttft_p50={:.3}s total_mean={:.3}s parity={}",
        suite.aggregate.prompt_tps_mean,
        suite.aggregate.decode_tps_mean,
        suite.aggregate.time_to_first_token_p50_seconds,
        suite.aggregate.total_seconds_mean,
        suite.aggregate.output_token_parity
    );
}

fn aggregate_bench_real(runs: &[BenchRealRunReport]) -> BenchRealAggregate {
    let n = runs.len().max(1) as f64;
    let cache_hits_total = runs.iter().map(|r| r.cache_hits).sum();
    let cache_misses_total = runs.iter().map(|r| r.cache_misses).sum();
    let total_lookups = cache_hits_total + cache_misses_total;
    let hit_rate = if total_lookups == 0 {
        0.0
    } else {
        cache_hits_total as f64 / total_lookups as f64
    };
    let mut ttft_us: Vec<u64> = runs
        .iter()
        .map(|r| (r.time_to_first_token_seconds * 1_000_000.0).round() as u64)
        .collect();
    ttft_us.sort_unstable();
    let output_token_parity = runs
        .windows(2)
        .all(|pair| pair[0].output_token_ids == pair[1].output_token_ids);
    BenchRealAggregate {
        prompt_seconds_mean: runs.iter().map(|r| r.prompt_seconds).sum::<f64>() / n,
        prompt_tps_mean: runs.iter().map(|r| r.prompt_tps).sum::<f64>() / n,
        decode_seconds_mean: runs.iter().map(|r| r.decode_seconds).sum::<f64>() / n,
        decode_tps_mean: runs.iter().map(|r| r.decode_tps).sum::<f64>() / n,
        time_to_first_token_p50_seconds: percentile_us(&ttft_us, 0.50) as f64 / 1_000_000.0,
        total_seconds_mean: runs.iter().map(|r| r.total_seconds).sum::<f64>() / n,
        cache_hits_total,
        cache_misses_total,
        hit_rate,
        ssd_bytes_total: runs.iter().map(|r| r.ssd_bytes).sum(),
        output_token_parity,
    }
}

fn bench_real_expected_forward_evaluations(
    prompt_tokens: usize,
    completion_tokens: usize,
) -> usize {
    if prompt_tokens == 0 || completion_tokens == 0 {
        0
    } else {
        prompt_tokens + completion_tokens - 1
    }
}

fn rate_per_second(count: usize, seconds: f64) -> f64 {
    if count == 0 || seconds <= 0.0 {
        0.0
    } else {
        count as f64 / seconds
    }
}

fn percentile_us_to_ms(sorted_us: &[u64], q: f64) -> f64 {
    percentile_us(sorted_us, q) as f64 / 1000.0
}

fn percentile_us(sorted_us: &[u64], q: f64) -> u64 {
    if sorted_us.is_empty() {
        return 0;
    }
    let q = q.clamp(0.0, 1.0);
    let idx = ((sorted_us.len() - 1) as f64 * q).round() as usize;
    sorted_us[idx]
}

fn build_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    if cfg!(feature = "tokenizer") {
        features.push("tokenizer");
    }
    if cfg!(feature = "io_uring") {
        features.push("io_uring");
    }
    if cfg!(feature = "blas") {
        features.push("blas");
    }
    if cfg!(feature = "alloc-count") {
        features.push("alloc-count");
    }
    if cfg!(feature = "avx512") {
        features.push("avx512");
    }
    if cfg!(feature = "q8-candle-reference") {
        features.push("q8-candle-reference");
    }
    if cfg!(feature = "amx") {
        features.push("amx");
    }
    if cfg!(feature = "nightly-amx") {
        features.push("nightly-amx");
    }
    if cfg!(feature = "cuda") {
        features.push("cuda");
    }
    if cfg!(feature = "tui") {
        features.push("tui");
    }
    if cfg!(feature = "grpc") {
        features.push("grpc");
    }
    features
}

fn git_commit_short() -> String {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output();
    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

fn current_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let body = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in body.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kib.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        let output = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let kib: u64 = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .ok()?;
        Some(kib.saturating_mul(1024))
    }
}

/// Resolve the serving tokenizer, enforcing the real-vs-synthetic policy
/// (Finding 4).
///
/// * `real_enabled` — `real_transformer.enabled`.
/// * `tokenizer_path` — configured `tokenizer.path`, if any.
/// * `model_vocab_size` — reconciled model vocabulary, when a real model
///   is present, used to validate tokenizer/model compatibility.
///
/// When a real transformer is enabled, a real tokenizer is mandatory: a
/// missing path or a load failure is fatal, the byte fallback is never
/// used, and every emittable token id must be `< model_vocab_size`. When
/// the real transformer is disabled, the byte-level fallback is preserved
/// for the synthetic/legacy path.
fn resolve_serving_tokenizer(
    real_enabled: bool,
    tokenizer_path: Option<&std::path::Path>,
    model_vocab_size: Option<usize>,
) -> Result<Arc<crate::tokenizer::Tokenizer>, crate::config::ConfigError> {
    use crate::config::ConfigError;
    use crate::tokenizer::Tokenizer;

    if real_enabled {
        let path = tokenizer_path.ok_or_else(|| {
            ConfigError::Invalid(
                "real_transformer.enabled requires tokenizer.path; the byte-level fallback \
                 tokenizer is not valid for real-checkpoint inference"
                    .to_string(),
            )
        })?;
        let tok = Tokenizer::from_file(path).map_err(|e| {
            ConfigError::Invalid(format!(
                "failed to load tokenizer {} for real-transformer serving: {e}",
                path.display()
            ))
        })?;
        if let Some(vocab) = model_vocab_size {
            tok.validate_vocab_compat(vocab).map_err(|e| {
                ConfigError::Invalid(format!(
                    "tokenizer {} is incompatible with the model: {e}",
                    path.display()
                ))
            })?;
        }
        Ok(Arc::new(tok))
    } else {
        match tokenizer_path {
            Some(p) => match Tokenizer::from_file(p) {
                Ok(t) => Ok(Arc::new(t)),
                Err(e) => {
                    warn!(path = %p.display(), error = %e, "tokenizer load failed; falling back to byte tokenizer");
                    Ok(Arc::new(Tokenizer::bytes()))
                }
            },
            None => {
                info!("no tokenizer.json configured; using byte-level fallback tokenizer");
                Ok(Arc::new(Tokenizer::bytes()))
            }
        }
    }
}

async fn cmd_serve(config_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    use crate::config::Config;
    use crate::metrics::Metrics;
    use crate::server::{serve, AppState};

    // NUMA-local pinning via `MER_PIN_CORES` is consumed centrally
    // at process start in `main()` (see `numa::apply_mer_pin_cores_env`)
    // and the env var is then cleared. We deliberately do **not**
    // re-read it here — every subcommand goes through that single
    // startup contract, so a per-subcommand re-read would be dead
    // code (gist feedback #2.3).

    let mut cfg = Config::from_file(&config_path)?;
    crate::parallel::set_dense_matvec_backend(cfg.real_transformer.dense_matvec_backend);

    // ---- Architecture resolution & hyperparameter reconciliation ----
    //
    // Resolve the model family and, when a Hugging Face `config.json` is
    // present in `weights_dir`, remap its hyperparameters onto the
    // engine-visible `[model]` / `[real_transformer]` config *before* we
    // size the expert cache, the layer-qualified expert namespace, or the
    // routing tables below. Doing this here (rather than only when the
    // `RealModel` is built) keeps a single source of truth: the engine and
    // the real model always agree on `num_layers` / `num_experts` / dims,
    // so a checkpoint never streams against a mismatched namespace.
    //
    // Precedence: explicit `[real_transformer] architecture = "…"` override
    // (exact HF `model_type`) wins; otherwise auto-detect from
    // `config.json`; otherwise default to Mixtral. An unrecognised
    // architecture is a hard error — we never silently mislabel a model.
    // Precedence: the checkpoint `config.json` (when present under
    // `weights_dir`) is always parsed and reconciled; an explicit
    // `[real_transformer] architecture = "…"` override never suppresses that
    // reconciliation and must agree with the checkpoint if both are present
    // (Finding 2). Serving and bench-real share the exact same reconciliation
    // and resolved-validation helpers so they can never disagree on the
    // resolved architecture/dims or accept a config the other rejects.
    let mut resolved_architecture = crate::architecture::Architecture::Mixtral;
    let mut resolved_first_k_dense_replace = 0usize;
    let mut resolved_advanced = crate::model::AdvancedConfig::default();
    if cfg.real_transformer.enabled {
        let (arch, first_k, advanced) = reconcile_real_model_config(&mut cfg)?;
        resolved_architecture = arch;
        resolved_first_k_dense_replace = first_k;
        resolved_advanced = advanced;
        // GPT-OSS SwiGLU gate clamp: install the process-global limit so the
        // expert-FFN hot path applies it (no-op when `swiglu_limit` is unset).
        crate::inference::set_swiglu_limit(resolved_advanced.swiglu_limit);
        // Architecture-aware validation of the fully-resolved config, identical
        // to the bench-real path (Finding 2/7).
        validate_resolved_real_model_config(
            &cfg,
            resolved_architecture,
            resolved_first_k_dense_replace,
            &resolved_advanced,
        )?;
    }

    info!(
        bind = %cfg.server.bind,
        data_dir = %cfg.model.data_dir.display(),
        num_experts = cfg.model.num_experts,
        num_layers = cfg.model.num_layers,
        top_k = cfg.model.top_k,
        d_model = cfg.model.d_model,
        d_ff = cfg.model.d_ff,
        "loaded server config"
    );

    // Resolve requested mode once into the authoritative component plan and
    // backend context before constructing the model or engine.
    //
    // The logical GPU-admission cache is constructed up-front so the same
    // `Arc` can be threaded into both `GpuBackend` (which validates admission
    // generations before physical lookup/upload) and
    // `Engine::install_gpu_cache` further below. When `[gpu_cache].enabled =
    // false` the zero-capacity cache never promotes anything.
    let gpu_expert_cache = {
        let capacity_bytes = if cfg.gpu_cache.enabled {
            (cfg.gpu_cache.vram_capacity_mb as usize) * 1024 * 1024
        } else {
            0
        };
        std::sync::Arc::new(crate::expert_cache::GpuExpertCache::new(
            capacity_bytes,
            cfg.gpu_cache.vram_anchor_ratio,
            cfg.gpu_cache.promote_after_hits,
        ))
    };
    // Strict attention numerics (the production default) can only be
    // validated on the checked CPU softmax path, so backend resolution is
    // strictness-aware (hardening pass, strict GPU behaviour): an explicit
    // `gpu` request fails startup, `auto` resolves to CPU with a recorded
    // reason, and the explicitly named `hybrid` mode runs checked CPU
    // attention with GPU expert offload.
    let strict_attention = cfg.real_transformer.enabled
        && !cfg
            .real_transformer
            .inference_policy()
            .allow_nonfinite_attention_fallback;
    let num_heads = cfg.real_transformer.num_heads;
    let head_dim = if cfg.real_transformer.head_dim == 0 {
        if num_heads > 0 {
            cfg.model.d_model / num_heads
        } else {
            64
        }
    } else {
        cfg.real_transformer.head_dim
    };
    let gpu_geometry = crate::backend::GpuBackendGeometry {
        num_layers: cfg.model.num_layers,
        max_seq_len: if cfg.real_transformer.gpu_native {
            cfg.real_transformer.gpu_native_max_seq_len
        } else if cfg.real_transformer.window_size == 0 {
            4096
        } else {
            cfg.real_transformer.window_size
        },
        num_heads,
        num_kv_heads: cfg.real_transformer.num_kv_heads,
        head_dim,
        // Finding 12: asymmetric-V models keep the CPU attention path;
        // zero retains the existing symmetric GPU sizing contract.
        v_head_dim: 0,
        q4_truncation_tolerance: cfg
            .real_transformer
            .inference_policy()
            .expert_size_tolerance(),
    };
    let execution_context = if cfg.real_transformer.gpu_native {
        crate::backend::resolve_execution_context_for_gpu_native(
            gpu_geometry,
            gpu_expert_cache.clone(),
        )?
    } else {
        crate::backend::resolve_execution_context(
            cfg.real_transformer.compute_offload,
            strict_attention,
            gpu_geometry,
            crate::backend::RoutedExpertGpuSpec {
                dtype: cfg.model.dtype,
                d_model: cfg.model.d_model,
                d_ff: cfg.model.d_ff,
            },
            gpu_expert_cache.clone(),
        )?
    };
    crate::backend::set_execution_context(execution_context.clone())
        .map_err(|e| format!("failed to install resolved execution context: {e}"))?;
    {
        let plan = execution_context.plan();
        let backend = execution_context.primary_backend();
        if plan.fallback_occurred() {
            warn!(
                requested = ?plan.requested(),
                resolved = ?plan.resolved(),
                reason = plan.reason().unwrap_or(""),
                "compute_offload = \"auto\": GPU initialization failed; resolved to CPU"
            );
        }
        info!(
            context_id = %execution_context.id(),
            backend = backend.device_name(),
            compute_plane = backend.compute_plane(),
            requested = ?plan.requested(),
            resolved = ?plan.resolved(),
            embeddings_plane = plan.embeddings().as_str(),
            lm_head_plane = plan.lm_head().as_str(),
            dense_projections_plane = plan.dense_projections().as_str(),
            attention_plane = plan.attention().as_str(),
            kv_plane = plan.kv().as_str(),
            router_plane = plan.router().as_str(),
            expert_plane = plan.routed_experts().as_str(),
            reason = plan.reason().unwrap_or(""),
            "resolved execution context installed"
        );
    }

    if !cfg.model.data_dir.is_dir() {
        return Err(format!(
            "data dir {} does not exist; run `gen-data` or the extractor first",
            cfg.model.data_dir.display()
        )
        .into());
    }
    crate::gguf_loader::validate_q4_0_dataset_layout(&cfg.model.data_dir, cfg.model.dtype)?;

    // Wire the multi-layer extractor naming when num_layers > 1, so
    // either `expert_<id>.bin` or `expert_<layer>_<local>.bin` works.
    let storage = NvmeStorage::new(StorageConfig {
        base_path: cfg.model.data_dir.clone(),
        expert_size: cfg.model.expert_size,
        block_align: cfg.storage.block_align,
        use_direct_io: !cfg.storage.no_direct,
        num_experts_per_layer: if cfg.model.num_layers > 1 {
            Some(cfg.model.num_experts)
        } else {
            None
        },
    })?;
    // Tier 2: attach the packed blob if configured (defaults: no-op).
    let storage = maybe_attach_packed_blob(
        storage,
        cfg.storage.packed_blob.as_deref(),
        cfg.storage.packed_manifest.as_deref(),
        !cfg.storage.no_direct,
        cfg.model.expert_size,
    )?;
    let storage = Arc::new(storage);
    // Warm fds across the whole multi-layer namespace (one global id
    // per (layer, local_expert) pair) so the steady-state path never
    // pays the open() cost. Skipped in packed mode: every expert is
    // served from the single already-open blob fd, and the per-expert
    // files may not even exist on disk.
    let total_experts = (cfg.model.num_layers as u32).saturating_mul(cfg.model.num_experts);
    if !storage.is_packed() {
        storage.warmup_fds(0..total_experts)?;
    }
    // Root-cause guard (hardening pass, Part A2): reject an
    // `expert_size` that disagrees with the on-disk file layout before
    // any truncated payload can be read.
    storage.validate_expert_file_layout(0..total_experts.min(8))?;

    // Double-buffered pool (Parts 1–2): split the RAM buffers into a
    // **primary** (Buffer A) half that backs the resident LRU plus one
    // reserved slot the foreground miss path is always guaranteed, and a
    // **shadow** (Buffer B) half that backs speculative look-ahead
    // prefetches. The shadow half is sized to the prefetch fanout scaled
    // by the look-ahead `pipeline_depth` (`predict_fanout * pipeline_depth`)
    // so a depth-N windowed look-ahead (`speculate_layer_ahead` priming
    // `layer + 1 ..= layer + pipeline_depth`) has a buffer per in-flight
    // layer and can never steal the buffer a real cache miss needs. The
    // prefetch semaphore is derived from this shadow capacity in
    // `Engine::with_options`, so it scales automatically. A fanout of 0
    // disables Buffer B and the engine falls back to the legacy
    // single-pool prefetch path.
    let pipeline_depth = cfg.storage.pipeline_depth.max(1) as usize;
    let shadow_slots = cfg.storage.predict_fanout.saturating_mul(pipeline_depth);
    let primary_slots = cfg.storage.cache_slots + 1;
    let pool = if shadow_slots > 0 {
        BufferPool::new_with_shadow(
            primary_slots,
            shadow_slots,
            cfg.model.expert_size,
            cfg.storage.block_align,
        )
    } else {
        BufferPool::new(
            primary_slots,
            cfg.model.expert_size,
            cfg.storage.block_align,
        )
    };
    let cache = {
        let num_layers = cfg.model.num_layers.max(1);
        let per_layer = cfg.model.num_experts.max(1) as u32;
        // Split the configured residency budget across layers so the
        // *aggregate* capacity matches the operator's `cache_slots`
        // setting. Layers each get a fair share with the remainder
        // distributed to the lower-indexed layers (which tend to be
        // hotter in MoE workloads).
        let total = cfg.storage.cache_slots.max(1);
        let base = total / num_layers;
        let extra = total % num_layers;
        let caps: Vec<usize> = (0..num_layers)
            .map(|i| base + if i < extra { 1 } else { 0 })
            .collect();
        if num_layers == 1 {
            Arc::new(MultiLayerExpertCache::single_layer(total))
        } else {
            Arc::new(MultiLayerExpertCache::with_capacities(caps, per_layer))
        }
    };

    // Multi-layer addressing: the engine's expert cache uses a single
    // global namespace `(layer * num_experts_per_layer) + local`, so
    // the router / predictor / locality monitor / speculator must all
    // be sized against the *total* expert count, not the per-layer
    // count. Otherwise layer-≥1 ids silently fall outside the
    // predictor's row table and the locality monitor's `is_hot` always
    // returns false for them.
    let total_experts: u32 = (cfg.model.num_layers as u32)
        .saturating_mul(cfg.model.num_experts)
        .max(cfg.model.num_experts);

    let predictor = Arc::new(PredictiveLoader::new(
        total_experts,
        cfg.storage.predict_fanout,
        resolve_predict_min_prob(cfg.storage.predict_min_prob, total_experts),
        0xC0FFEE,
    ));

    // Build the real transformer (if enabled) *before* the engine so
    // its per-layer `LinearGate` can be wired into the engine as the
    // production routing path. When `[real_transformer].enabled =
    // true`, the engine holds `Router::Linear` from the loaded
    // model's first-layer gate — that is the path
    // `Engine::generate` will exercise on the benchmark / warmup
    // surfaces. The per-token `RealModel::step` loop in serve mode
    // routes each MoE layer through *its own* layer-local gate
    // (`TransformerLayer::moe_pre`) and calls `engine.moe_step` with
    // the already-routed ids, so the engine's router does not
    // override per-layer routing — but it does mean the engine's
    // self-reported `num_experts` / `top_k` now reflect the actual
    // gate shape rather than the legacy Markov stand-in, which is
    // the gist's "wire `LinearGate` into `serve`" ask.
    let real_model: Option<Arc<crate::model::RealModel>> = if cfg.real_transformer.enabled {
        let rt = &cfg.real_transformer;
        if rt.allow_degraded_experts {
            warn!(
                "DEGRADED MODE ENABLED: real_transformer.allow_degraded_experts = true — a \
                 failed routed/shared expert is substituted with a zero contribution instead \
                 of failing the request. Output and all benchmark figures are \
                 NON-AUTHORITATIVE (see degraded_expert_substitutions)."
            );
        }
        if rt.allow_nonfinite_attention_fallback {
            warn!(
                "NON-FINITE-ATTENTION FALLBACK ENABLED: \
                 real_transformer.allow_nonfinite_attention_fallback = true — a non-finite \
                 attention softmax row is replaced by a uniform distribution instead of \
                 failing the request. Output is NON-AUTHORITATIVE."
            );
        }
        if rt.allow_truncated_expert_payloads {
            warn!(
                "TRUNCATED-PAYLOAD TOLERANCE ENABLED: \
                 real_transformer.allow_truncated_expert_payloads = true — quantised expert \
                 payloads up to one page short are zero-filled instead of failing the \
                 request. Output is NON-AUTHORITATIVE."
            );
        }
        // Log each resolved fail-open policy explicitly, so the effective
        // strict/degraded posture of the run is always visible at startup
        // (hardening pass, policy separation). The policy is engine-scoped
        // (threaded via `EngineOptions::policy`), not a process global.
        let policy = rt.inference_policy();
        info!(
            allow_degraded_experts = policy.allow_degraded_experts,
            allow_nonfinite_attention_fallback = policy.allow_nonfinite_attention_fallback,
            allow_truncated_expert_payloads = policy.allow_truncated_expert_payloads,
            degraded = policy.any_degraded(),
            "resolved real-inference fail-open policies"
        );
        let head_dim = if rt.head_dim == 0 {
            cfg.model.d_model / rt.num_heads
        } else {
            rt.head_dim
        };
        let num_kv_heads = if rt.num_kv_heads == 0 {
            rt.num_heads
        } else {
            rt.num_kv_heads
        };
        // Hyperparameters were already reconciled from `config.json` (when
        // present) at the top of `cmd_serve`, so `cfg.model` /
        // `cfg.real_transformer` are the single source of truth here. We
        // just stamp the resolved architecture + dense/MoE boundary onto
        // the `RealModelConfig`. Recognised-but-unrunnable families
        // (DeepSeek-V3: MLA + FP8) fail loud inside `from_safetensors`.
        let model_cfg = crate::model::RealModelConfig {
            d_model: cfg.model.d_model,
            d_ff: cfg.model.d_ff,
            num_heads: rt.num_heads,
            num_kv_heads,
            head_dim,
            vocab_size: rt.vocab_size,
            num_layers: cfg.model.num_layers,
            num_experts: cfg.model.num_experts as usize,
            top_k: cfg.model.top_k,
            rope_base: rt.rope_base,
            rms_eps: rt.rms_eps,
            window_size: if rt.window_size == 0 {
                None
            } else {
                Some(rt.window_size)
            },
            architecture: resolved_architecture,
            first_k_dense_replace: resolved_first_k_dense_replace,
            advanced: resolved_advanced,
        };
        let load_options = crate::model::RealModelLoadOptions {
            strict_weights: rt.strict_weights,
        };
        // Fail-closed real-model weight policy (gist Finding 1). A missing
        // `weights_dir` or `strict_weights = false` is rejected unless the
        // operator explicitly opts into the development seeded fallback.
        let weight_policy = rt.resolve_weight_policy()?;
        let m = match (rt.weights_dir.as_ref(), weight_policy) {
            (Some(dir), _) => {
                let loaded = crate::model::RealModel::from_dir_auto_with_options(
                    model_cfg,
                    dir,
                    rt.seed,
                    load_options,
                )?;
                if loaded.load_status.seeded_fallback_remained {
                    warn!(
                        loader = loaded.load_status.loader,
                        loaded_tensors = loaded.load_status.loaded_tensors,
                        required_tensors = loaded.load_status.required_tensors,
                        "DEVELOPMENT FALLBACK: real_transformer served with seeded weights \
                         remaining — output is NOT real-checkpoint inference"
                    );
                }
                loaded
            }
            (None, crate::config::RealWeightPolicy::SeededDev) => {
                warn!(
                    "DEVELOPMENT FALLBACK: real_transformer.enabled with no weights_dir and \
                     allow_seeded_fallback = true — serving deterministic SEEDED weights, \
                     output is NOT real-checkpoint inference"
                );
                crate::model::RealModel::new_seeded(model_cfg, rt.seed)
            }
            // resolve_weight_policy already rejects (None, StrictReal).
            (None, crate::config::RealWeightPolicy::StrictReal) => unreachable!(
                "resolve_weight_policy rejects a missing weights_dir without seeded fallback"
            ),
        };
        info!(
            strict = m.load_status.strict,
            loader = m.load_status.loader,
            loaded_tensors = m.load_status.loaded_tensors,
            required_tensors = m.load_status.required_tensors,
            seeded_fallback_remained = m.load_status.seeded_fallback_remained,
            "real transformer model-loading status"
        );
        Some(Arc::new(m))
    } else {
        None
    };

    // Build draft engine for speculative decoding when the speculator is
    // enabled and a real model is available. `DraftEngine::from_main`
    // avoids loading any extra weights from disk, but it currently
    // **clones** the main model's embedding into a fresh `Arc<Vec<f32>>`
    // rather than sharing the `RealModel`'s allocation, so enabling this
    // path costs one additional `vocab_size * d_model * 4` bytes of
    // resident memory. See `draft::DraftEngine::from_main` for the exact
    // allocation site.
    let draft_engine: Option<Arc<crate::draft::DraftEngine>> = if cfg.predictive.speculator_enabled
    {
        real_model.as_ref().map(|m| {
            let d = crate::draft::DraftEngine::from_main(m);
            tracing::info!(
                vocab_size = m.config.vocab_size,
                d_model = m.config.d_model,
                "draft engine built for speculative decoding"
            );
            Arc::new(d)
        })
    } else {
        None
    };

    let speculation_k = cfg.real_transformer.speculation_base_depth.max(1);

    let router = if let Some(ref m) = real_model {
        // Production routing path: the engine's `route()` runs the
        // first layer's `softmax(W_gate · x) → top-K` (Mixtral-style)
        // instead of the legacy deterministic Markov chain. Per-layer
        // gates still drive per-layer routing inside `RealModel::step`
        // — this engine-level gate is what `Engine::generate` and
        // anything else that asks the engine for a routing decision
        // sees.
        info!(
            num_experts = m.layers[0].gate.num_experts,
            d_model = m.layers[0].gate.d_model,
            top_k = m.layers[0].gate.top_k,
            "engine routing: LinearGate (production softmax-gated path) wired from real model"
        );
        crate::gating::Router::Linear(Arc::new(m.layers[0].gate.clone()))
    } else {
        info!(
            total_experts,
            clusters = 4,
            "engine routing: clustered Markov chain (no real model loaded)"
        );
        crate::gating::Router::Markov(Arc::new(TopKRouter::clustered(
            total_experts,
            cfg.model.top_k,
            4,
            0.9,
            0xC0FFEE,
        )))
    };

    let metrics = Metrics::new();
    metrics.set_backend_component_planes(execution_context.plan());
    let mut engine_builder = Engine::with_options_and_execution_context(
        cache,
        pool,
        storage,
        router,
        predictor,
        ModelShape {
            d_model: cfg.model.d_model,
            d_ff: cfg.model.d_ff,
            hidden_seed: 0xC0FFEE,
        },
        EngineOptions {
            io_only: false,
            dtype: cfg.model.dtype,
            partial_load_fraction: cfg.storage.partial_load_fraction,
            pin_after_observations: cfg.storage.pin_after_observations,
            use_qmm_for_q4: true,
            expert_execution_policy: cfg.real_transformer.expert_execution_policy,
            max_concurrent_prefetches: cfg.real_transformer.max_concurrent_prefetches,
            max_fetch_yields: cfg.real_transformer.max_fetch_yields,
            prefetch_governor: cfg.predictive.prefetch_governor,
            prefetch_precision_floor: cfg.predictive.prefetch_precision_floor,
            prefetch_contention_weight: cfg.predictive.prefetch_contention_weight,
            cost_aware_eviction: cfg.predictive.cost_aware_eviction,
            pregate_enabled: cfg.predictive.pregate_enabled,
            collect_route_profile: false,
            policy: cfg.real_transformer.inference_policy(),
        },
        execution_context.clone(),
    );
    // Apply the configured look-ahead pipeline depth (`[storage]
    // pipeline_depth`). Controls how many layers ahead
    // `speculate_layer_ahead` primes; sized in tandem with the shadow
    // buffer-pool budget above.
    engine_builder = engine_builder.with_pipeline_depth(cfg.storage.pipeline_depth);
    // Attach the speculative-architecture components requested via
    // the `[predictive]` config section. Sized against the global
    // expert namespace (see `total_experts` above) so multi-layer
    // models don't silently drop layer-≥1 ids on the floor.
    if cfg.predictive.locality_enabled {
        // Scale the sliding window by the layer count: with a
        // layer-qualified namespace every token contributes
        // `num_layers × top_k` activations, so a flat 256-deep window
        // only holds ~8 activations *per layer* — far too few for the
        // per-layer heat threshold (`effective_locality_threshold`,
        // which divides by the layer count) to discriminate anything.
        // Multiplying the window by the layer count keeps the
        // *per-layer* history depth equal to what the operator
        // configured for a flat namespace.
        let window = cfg
            .predictive
            .locality_window
            .saturating_mul(cfg.model.num_layers.max(1));
        let monitor = Arc::new(LocalityMonitor::new(total_experts, window));
        engine_builder =
            engine_builder.with_locality_monitor(monitor, cfg.predictive.locality_threshold_pct);
    }
    if cfg.predictive.speculator_enabled {
        let top_k = if cfg.predictive.speculator_top_k == 0 {
            cfg.model.top_k
        } else {
            cfg.predictive.speculator_top_k
        };
        let spec = Arc::new(NeuralSpeculator::new(
            cfg.model.d_model,
            cfg.predictive.speculator_hidden_dim,
            total_experts,
            0xC0FFEE,
        ));
        engine_builder = engine_builder.with_speculator(spec, top_k);
    }
    // Per-layer expert-affinity arm: tracks which experts co-fire inside
    // the same MoE layer and folds their co-fired + disk-adjacent
    // neighbours into the prefetch union. Sized in the *layer-local* id
    // namespace (one `num_experts × num_experts` matrix per layer); the
    // engine maps global ids ↔ layer-local before observing / looking up
    // neighbours. Only effective when the model exposes a
    // layer-qualified geometry (`num_experts_per_layer`).
    if cfg.predictive.affinity_enabled {
        let num_layers = cfg.model.num_layers.max(1);
        let affinity = Arc::new(LayeredExpertAffinity::new(
            num_layers,
            cfg.model.num_experts,
        ));
        engine_builder = engine_builder.with_affinity(
            affinity,
            cfg.predictive.affinity_neighbors_k,
            cfg.predictive.affinity_decay_epoch,
        );
    }
    // Tier 1 — static residency. Pin the hottest `fraction` of experts
    // permanently (from an offline profile when `static_residency_profile`
    // is set, else online after the warmup window).
    if cfg.predictive.static_residency_fraction > 0.0 {
        let profile = match cfg.predictive.static_residency_profile.as_ref() {
            Some(path) => {
                let p = crate::residency::ResidencyProfile::load_json(std::path::Path::new(path))?;
                info!(
                    path = %path,
                    experts = p.len(),
                    "loaded static-residency popularity profile"
                );
                Some(p)
            }
            None => None,
        };
        engine_builder = engine_builder.with_static_residency(
            cfg.predictive.static_residency_fraction,
            cfg.predictive.static_residency_warmup_tokens,
            profile,
        );
    }
    // Tier 3 — per-layer pre-gate. Predict + prefetch the next layer's
    // experts from the current layer's routed set on the multi-layer
    // `moe_step` path.
    if cfg.predictive.pregate_enabled {
        let pregate = Arc::new(crate::pregate::PerLayerPreGate::new(
            cfg.model.num_layers.max(1),
            cfg.model.top_k,
        ));
        engine_builder = engine_builder.with_pregate(pregate);
    }
    // Phase 2: optional logical GPU-admission cache. Physical wgpu expert
    // residency is owned separately by the backend registry. When disabled,
    // the engine retains its historical 2-tier posture.
    if cfg.gpu_cache.enabled {
        // `gpu_cache.dtype` is currently advisory — it is validated by
        // `AppConfig::validate` (so typos fail fast) and surfaced here
        // for observability, but the promotion path copies on-disk
        // bytes into logical host admission without conversion or repacking. Parse it
        // here purely so the startup log records the operator's
        // declared intent.
        let dtype_for_logging = crate::inference::WeightDtype::from_str_opt(&cfg.gpu_cache.dtype)
            .unwrap_or(crate::inference::WeightDtype::F16);
        info!(
            vram_capacity_mb = cfg.gpu_cache.vram_capacity_mb,
            anchor_ratio = cfg.gpu_cache.vram_anchor_ratio,
            promote_after_hits = cfg.gpu_cache.promote_after_hits,
            dtype_advisory = %dtype_for_logging.as_str(),
            "logical GPU expert admission enabled (dtype is advisory; physical upload is lazy and backend-owned)"
        );
        engine_builder.install_gpu_cache();
    }

    let (real_model, batch_scheduler, gpu_native_token_loop, engine) = if let Some(model_arc) = real_model {
        if cfg.real_transformer.gpu_native {
            let executor = Arc::new(
                execution_context
                    .create_gpu_native_executor_context(cfg.model.d_model)
                    .map_err(|e| format!("failed to create GPU-native executor context: {e}"))?,
            );
            let total_expert_budget = (cfg.gpu_cache.vram_capacity_mb as u64) * 1024 * 1024;
            let expert_geom = crate::backend::gpu_native::GpuNativeQ4ExpertGeometry::try_new(
                cfg.model.d_model,
                cfg.model.d_ff,
                cfg.model.num_experts as usize,
                cfg.model.top_k,
            )
            .map_err(|e| format!("invalid GPU-native Q4 expert geometry: {e}"))?;
            let residency_manager = Arc::new(
                crate::gpu_native_residency::GpuNativeTieredResidencyManager::try_new(
                    executor.clone(),
                    gpu_expert_cache.clone(),
                    cfg.model.num_layers,
                    expert_geom,
                    total_expert_budget,
                )
                .map_err(|e| format!("failed to construct tiered residency manager: {e}"))?,
            );
            engine_builder
                .install_gpu_native_residency_manager(residency_manager.clone())
                .map_err(|e| format!("failed to install tiered residency manager on engine: {e}"))?;
            let token_loop = crate::gpu_native_token_loop::GpuNativeTokenLoop::try_new(
                executor,
                residency_manager,
                &model_arc,
                cfg.real_transformer.gpu_native_max_seq_len,
            )
            .map_err(|e| format!("failed to initialize GPU-native token loop: {e}"))?;
            let engine = Arc::new(engine_builder.with_metrics(metrics.clone()));
            info!(
                num_layers = cfg.model.num_layers,
                d_model = cfg.model.d_model,
                num_experts = cfg.model.num_experts,
                top_k = cfg.model.top_k,
                max_seq_len = cfg.real_transformer.gpu_native_max_seq_len,
                "GPU-native token loop initialized (authoritative GPU-owned execution)"
            );
            (Some(model_arc), None, Some(token_loop), engine)
        } else {
            let engine = Arc::new(engine_builder.with_metrics(metrics.clone()));
            let rt = &cfg.real_transformer;
        let head_dim = if rt.head_dim == 0 {
            cfg.model.d_model / rt.num_heads
        } else {
            rt.head_dim
        };
        let num_kv_heads = if rt.num_kv_heads == 0 {
            rt.num_heads
        } else {
            rt.num_kv_heads
        };
        let batch_cfg = crate::batch_scheduler::BatchConfig {
            max_batch_size: rt.max_batch_size,
            batch_timeout: std::time::Duration::from_millis(rt.batch_timeout_ms),
            idle_eviction_threshold: std::time::Duration::from_millis(
                rt.idle_eviction_threshold_ms,
            ),
            speculation_base_depth: rt.speculation_base_depth,
            // Pool back-pressure ladder is now config-driven
            // (gist Part 1, fix #4). Validation in `Config::validate`
            // already enforces 0 < high <= critical <= 1.
            pressure_thresholds: crate::block_pool::PressureThresholds::try_new(
                rt.pressure_high_threshold,
                rt.pressure_critical_threshold,
            )
            .expect("pressure thresholds validated by Config::validate")
            .with_max_overflow_capacity(rt.max_overflow_capacity),
            ..Default::default()
        };
        // Expert-placement layer: single-node default (every id
        // local), or the `[distributed]` `id % num_nodes` hash
        // partitioning over the configured mesh when enabled.
        let shard_router: std::sync::Arc<dyn crate::distributed::ShardRouter> =
            if cfg.distributed.enabled {
                let router = crate::distributed::RpcShardRouter::from_modulo_placement(
                    &cfg.distributed.nodes,
                    cfg.distributed.self_index,
                    total_experts,
                    std::time::Duration::from_millis(cfg.distributed.remote_fetch_timeout_ms),
                );
                info!(
                    nodes = cfg.distributed.nodes.len(),
                    self_index = cfg.distributed.self_index,
                    total_experts,
                    remote_fetch_timeout_ms = cfg.distributed.remote_fetch_timeout_ms,
                    "distributed expert partitioning enabled (id % num_nodes)"
                );
                std::sync::Arc::new(router)
            } else {
                std::sync::Arc::new(crate::distributed::LocalShardRouter)
            };
        let scheduler = crate::batch_scheduler::BatchScheduler::spawn_with_shard_router(
            model_arc.clone(),
            engine.clone(),
            batch_cfg,
            shard_router,
        );
        info!(
            num_layers = cfg.model.num_layers,
            num_heads = rt.num_heads,
            num_kv_heads,
            head_dim,
            vocab_size = rt.vocab_size,
            max_batch_size = rt.max_batch_size,
            batch_timeout_ms = rt.batch_timeout_ms,
            idle_eviction_threshold_ms = rt.idle_eviction_threshold_ms,
            speculation_base_depth = rt.speculation_base_depth,
            "real transformer pipeline enabled (with continuous batching)"
        );
        (Some(model_arc), Some(Arc::new(scheduler)), None, engine)
    }
} else {
    let engine = Arc::new(engine_builder.with_metrics(metrics.clone()));
    info!("real_transformer disabled; using legacy benchmark generator");
    (None, None, None, engine)
};

let tokenizer = resolve_serving_tokenizer(
    cfg.real_transformer.enabled,
    cfg.tokenizer.path.as_deref(),
    real_model.as_ref().map(|m| m.config.vocab_size),
)?;

    let sessions = if cfg.server.session_ttl_secs > 0 {
        let store = crate::session::SessionStore::new(std::time::Duration::from_secs(
            cfg.server.session_ttl_secs,
        ));
        // Sweep every TTL/2 (or once a minute, whichever is shorter) so
        // peak memory stays bounded but the evictor doesn't dominate
        // the wakeup budget.
        let sweep =
            std::time::Duration::from_secs((cfg.server.session_ttl_secs / 2).max(1).min(60));
        store.spawn_evictor(sweep);
        Some(store)
    } else {
        None
    };

    // Background overflow-slab reclaimer: every 60s, ask the paged-KV
    // pool to return any heap-backed overflow blocks that are no
    // longer in use. Cheap when there's nothing to reclaim (single
    // mutex check + early return), so safe to run unconditionally.
    if let Some(pool) = batch_scheduler.as_ref().and_then(|s| s.block_pool()) {
        let pool = pool.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(std::time::Duration::from_secs(60));
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let reclaimed = pool.shrink_overflow_to_fit();
                if reclaimed > 0 {
                    tracing::info!(
                        reclaimed,
                        "background sweep: reclaimed paged-KV overflow blocks"
                    );
                }
            }
        });
    }

    // Build the production-readiness middleware bundle:
    //  - API-key gate (optional, off by default)
    //  - in-process token-bucket rate limit (optional, off by default)
    //  - admission controller (concurrency cap + paged-KV free-block watermark)
    use crate::middleware::{Admission, ApiKeyGate, MiddlewareState, RateLimiter};
    let api_keys = ApiKeyGate::new(&cfg.security.api_keys);
    let rate_limit = RateLimiter::new(cfg.security.rate_limit_rps, cfg.security.rate_limit_burst);
    let free_probe: Option<std::sync::Arc<dyn Fn() -> usize + Send + Sync>> =
        match batch_scheduler.as_ref().and_then(|s| s.block_pool()) {
            Some(p) => {
                let p = p.clone();
                Some(std::sync::Arc::new(move || p.free_blocks()))
            }
            None => None,
        };
    let admission = Admission::new(
        cfg.server.max_concurrent_requests,
        cfg.server.admission_min_free_blocks,
        free_probe,
    );
    let middleware_state = MiddlewareState {
        api_keys,
        rate_limit,
        admission,
    };

    // Live, atomically-swappable runtime configuration. The hot
    // token-evaluation path reads sampling defaults and the
    // `max_tokens` cap through `runtime.snapshot()` (a single relaxed
    // atomic load — see `LiveConfig` in `crate::config`). SIGHUP
    // refreshes this in place.
    let runtime = crate::config::LiveConfig::from_config(&cfg);

    let state = AppState {
        engine,
        tokenizer,
        metrics,
        real_model,
        batch_scheduler,
        draft_engine,
        gpu_native_token_loop,
        speculation_k,
        runtime: runtime.clone(),
        sessions,
        middleware: middleware_state,
    };
    // SIGHUP-triggered config reload.
    //
    // For fields covered by [`crate::config::RuntimeConfig`] (sampling
    // defaults, max-tokens cap, telemetry flags) we apply the reload
    // live via `runtime.try_reload(&new)` — an atomic `ArcSwap` store.
    // In-flight requests holding a `runtime.snapshot()` keep observing
    // their previous `Arc<RuntimeConfig>` until they drop it; concurrent
    // SIGHUPs never block on each other and never block readers.
    //
    // For restart-required fields (storage prefetch settings, batch
    // scheduler timing, etc.) we still emit a structured diff at WARN
    // level so operators know a restart is needed to fully apply the
    // file. If parsing or validation fails the in-memory runtime is
    // left **pristine** and a single `tracing::warn!` line documents
    // the rejection.
    #[cfg(unix)]
    {
        let path = config_path.clone();
        let baseline = cfg.clone();
        let runtime = runtime;
        tokio::spawn(async move {
            use tokio::signal::unix::{signal, SignalKind};
            let mut sig = match signal(SignalKind::hangup()) {
                Ok(s) => s,
                Err(e) => {
                    warn!(error = %e, "SIGHUP handler install failed; config reload disabled");
                    return;
                }
            };
            let mut prev = baseline;
            while sig.recv().await.is_some() {
                info!("SIGHUP received; reloading config from {}", path.display());
                let new = match Config::from_file(&path) {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(
                            error = %e,
                            "config reload rejected; existing runtime configuration left un-mutated",
                        );
                        continue;
                    }
                };
                // Apply the safe-to-reload subset live. `try_reload`
                // re-validates internally; on rejection it logs a
                // structured `tracing::warn!` and leaves the live
                // `ArcSwap<RuntimeConfig>` un-mutated. The atomic store
                // is contention-free with request-path readers
                // (`runtime.snapshot()` is a single relaxed atomic load).
                match runtime.try_reload(&new) {
                    Ok(rc) => info!(
                        sampling_temperature = rc.sampling.temperature,
                        sampling_top_p = rc.sampling.top_p,
                        sampling_top_k = rc.sampling.top_k,
                        max_tokens_cap = rc.max_tokens_cap,
                        "live runtime configuration swapped atomically",
                    ),
                    Err(_) => {
                        // try_reload already emitted a structured warn;
                        // skip applying restart-key diffs against an
                        // invalid file.
                        continue;
                    }
                }
                if prev.real_transformer.dense_matvec_backend
                    != new.real_transformer.dense_matvec_backend
                {
                    crate::parallel::set_dense_matvec_backend(
                        new.real_transformer.dense_matvec_backend,
                    );
                }
                // Restart-required diff: surface changes that the live
                // swap does **not** cover so operators know which fields
                // still demand a process restart.
                let restart_keys: &[(&str, String, String)] = &[
                    (
                        "storage.predict_fanout",
                        prev.storage.predict_fanout.to_string(),
                        new.storage.predict_fanout.to_string(),
                    ),
                    (
                        "real_transformer.batch_timeout_ms",
                        prev.real_transformer.batch_timeout_ms.to_string(),
                        new.real_transformer.batch_timeout_ms.to_string(),
                    ),
                    (
                        "real_transformer.idle_eviction_threshold_ms",
                        prev.real_transformer.idle_eviction_threshold_ms.to_string(),
                        new.real_transformer.idle_eviction_threshold_ms.to_string(),
                    ),
                    (
                        "real_transformer.speculation_base_depth",
                        prev.real_transformer.speculation_base_depth.to_string(),
                        new.real_transformer.speculation_base_depth.to_string(),
                    ),
                    (
                        "real_transformer.expert_execution_policy",
                        format!("{:?}", prev.real_transformer.expert_execution_policy),
                        format!("{:?}", new.real_transformer.expert_execution_policy),
                    ),
                    (
                        "storage.predict_min_prob",
                        prev.storage.predict_min_prob.to_string(),
                        new.storage.predict_min_prob.to_string(),
                    ),
                    (
                        "storage.partial_load_fraction",
                        prev.storage.partial_load_fraction.to_string(),
                        new.storage.partial_load_fraction.to_string(),
                    ),
                ];
                for (k, before, after) in restart_keys {
                    if before != after {
                        warn!(key = k, before = %before, after = %after,
                            "config changed but requires restart to take effect");
                    }
                }
                prev = new;
            }
        });
    }

    serve(state, &cfg.server.bind).await
}

fn cmd_gen_data(
    data_dir: &std::path::Path,
    num_experts: u32,
    expert_size: usize,
    d_model: usize,
    d_ff: usize,
    block_align: usize,
    dtype_str: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::inference::WeightDtype;
    let dtype = WeightDtype::from_str_opt(dtype_str).ok_or_else(|| {
        format!(
            "--dtype: unknown value {dtype_str:?} (supported for gen-data: {SUPPORTED_SYNTHETIC_DTYPES})"
        )
    })?;
    if matches!(
        dtype,
        WeightDtype::Q5K | WeightDtype::Q6K | WeightDtype::Mixed
    ) {
        return Err(format!(
            "gen-data does not synthesize dtype {}; use gguf-convert --native-quant or an offline extractor for this layout",
            dtype.as_str()
        )
        .into());
    }
    if block_align == 0 || !block_align.is_power_of_two() {
        return Err(format!(
            "--block-align ({block_align}) must be a positive power of two \
             (4096 on most NVMe)."
        )
        .into());
    }
    if expert_size % block_align != 0 {
        return Err(format!(
            "--expert-size ({expert_size}) must be a multiple of --block-align \
             ({block_align}) so the run path can read each expert with O_DIRECT \
             without EINVAL."
        )
        .into());
    }
    let weight_bytes = crate::inference::expert_weight_bytes_for(d_model, d_ff, dtype);
    if weight_bytes > expert_size {
        return Err(format!(
            "expert_size ({expert_size}) is too small for the SwiGLU weights of \
             d_model={d_model}, d_ff={d_ff} dtype={} ({weight_bytes} bytes). Increase \
             --expert-size or shrink --d-model / --d-ff.",
            dtype.as_str()
        )
        .into());
    }
    info!(
        path = %data_dir.display(),
        num_experts,
        expert_size_mib = expert_size as f64 / (1024.0 * 1024.0),
        d_model,
        d_ff,
        block_align,
        dtype = dtype.as_str(),
        weight_mib = weight_bytes as f64 / (1024.0 * 1024.0),
        "generating synthetic SwiGLU expert weights"
    );
    let started = Instant::now();
    crate::io_provider::generate_synthetic_experts_with_dtype(
        data_dir,
        num_experts,
        expert_size,
        d_model,
        d_ff,
        dtype,
    )?;
    if dtype == WeightDtype::Q4_0 {
        let metadata = serde_json::json!({
            "format_version": 2,
            "conversion_mode": "synthetic",
            "num_experts": num_experts,
            "d_model": d_model,
            "d_ff": d_ff,
            "expert_size": expert_size,
            "maximum_payload_bytes": weight_bytes,
            "block_align": block_align,
            "dtype": dtype.as_str(),
            "weight_layout": "gate_proj || up_proj || down_proj (row-major)",
            "q4_0_layout": crate::inference::Q4_0_LAYOUT_STANDARD_V1,
            "experts_written": num_experts,
        });
        let mut body = serde_json::to_vec_pretty(&metadata)?;
        body.push(b'\n');
        std::fs::write(data_dir.join("metadata.json"), body)?;
    }
    let total_bytes = num_experts as u64 * expert_size as u64;
    info!(
        elapsed_s = started.elapsed().as_secs_f64(),
        total_mib = total_bytes as f64 / (1024.0 * 1024.0),
        "expert files written"
    );
    Ok(())
}

struct RepackArgs {
    data_dir: PathBuf,
    out_blob: PathBuf,
    out_manifest: Option<PathBuf>,
    num_experts: u32,
    expert_size: usize,
    block_align: usize,
    no_direct: bool,
    num_experts_per_layer: Option<u32>,
    profile: Option<PathBuf>,
    order: Option<PathBuf>,
}

/// Parse an explicit `--order` file: either a JSON array of ids or a
/// newline / whitespace-separated list (blank lines and `#` comments
/// ignored).
fn parse_order_file(path: &std::path::Path) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    let trimmed = raw.trim_start();
    if trimmed.starts_with('[') {
        let ids: Vec<u32> = serde_json::from_str(trimmed)?;
        return Ok(ids);
    }
    let mut ids = Vec::new();
    for line in raw.lines() {
        let without_comment = line.split_once('#').map_or(line, |(body, _)| body);
        for tok in without_comment.split(|c: char| c.is_whitespace() || c == ',') {
            let t = tok.trim();
            if t.is_empty() {
                continue;
            }
            ids.push(t.parse::<u32>()?);
        }
    }
    Ok(ids)
}

fn validate_order(ids: &[u32], num_experts: u32) -> Result<(), String> {
    let mut seen = std::collections::HashSet::with_capacity(ids.len());
    for &id in ids {
        if id >= num_experts {
            return Err(format!(
                "--order id {id} is out of range for --num-experts {num_experts}"
            ));
        }
        if !seen.insert(id) {
            return Err(format!("--order contains duplicate expert id {id}"));
        }
    }
    Ok(())
}

/// **Tier 2.** Build a packed blob + manifest from a per-expert directory.
async fn cmd_repack(args: RepackArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.block_align == 0 || !args.block_align.is_power_of_two() {
        return Err(format!(
            "--block-align ({}) must be a positive power of two",
            args.block_align
        )
        .into());
    }
    if args.expert_size % args.block_align != 0 {
        return Err(format!(
            "--expert-size ({}) must be a multiple of --block-align ({})",
            args.expert_size, args.block_align
        )
        .into());
    }
    if !args.data_dir.is_dir() {
        return Err(format!("data dir {} does not exist", args.data_dir.display()).into());
    }

    // Resolve the physical layout order.
    let order: Vec<u32> = if let Some(order_path) = &args.order {
        let ids = parse_order_file(order_path)?;
        if ids.is_empty() {
            return Err(format!("--order file {} listed no ids", order_path.display()).into());
        }
        validate_order(&ids, args.num_experts)?;
        let missing = args.num_experts as usize - ids.len();
        if missing > 0 {
            warn!(
                missing,
                "repack: explicit order omits experts; running/serving in packed mode will hard-error with NotFound if an omitted expert is routed"
            );
        }
        info!(
            count = ids.len(),
            path = %order_path.display(),
            "repack: using explicit expert order"
        );
        ids
    } else if let Some(profile_path) = &args.profile {
        let profile = crate::residency::ResidencyProfile::load_json(profile_path)?;
        // Hottest-first over the whole namespace, then append any expert
        // the profile never observed so the blob still covers 0..N.
        let mut ranked = profile.hot_set(1.0, args.num_experts as usize);
        let seen: std::collections::HashSet<u32> = ranked.iter().copied().collect();
        for id in 0..args.num_experts {
            if !seen.contains(&id) {
                ranked.push(id);
            }
        }
        info!(
            observed = seen.len(),
            total = ranked.len(),
            path = %profile_path.display(),
            "repack: ordering experts hottest-first from profile"
        );
        ranked
    } else {
        info!(
            num_experts = args.num_experts,
            "repack: using numeric expert order"
        );
        (0..args.num_experts).collect()
    };

    let manifest_path = args.out_manifest.clone().unwrap_or_else(|| {
        let mut p = args.out_blob.clone().into_os_string();
        p.push(".manifest.json");
        PathBuf::from(p)
    });

    let storage = NvmeStorage::new(StorageConfig {
        base_path: args.data_dir.clone(),
        expert_size: args.expert_size,
        block_align: args.block_align,
        use_direct_io: !args.no_direct,
        num_experts_per_layer: args.num_experts_per_layer,
    })?;
    // One reusable buffer is enough (we read sequentially), but a tiny
    // pool keeps the acquire/release ergonomics and alignment.
    let pool = BufferPool::new(2, args.expert_size, args.block_align);

    info!(
        experts = order.len(),
        out_blob = %args.out_blob.display(),
        out_manifest = %manifest_path.display(),
        "repack: writing packed blob"
    );
    let started = Instant::now();
    let manifest =
        crate::io_provider::pack_experts(&storage, &pool, &order, &args.out_blob, &manifest_path)
            .await?;
    info!(
        elapsed_s = started.elapsed().as_secs_f64(),
        blob_mib = manifest.blob_len() as f64 / (1024.0 * 1024.0),
        experts = manifest.len(),
        "repack complete — point [storage] packed_blob / packed_manifest at these files"
    );
    Ok(())
}

struct RunArgs {
    data_dir: PathBuf,
    num_experts: u32,
    expert_size: usize,
    d_model: usize,
    d_ff: usize,
    cache_slots: usize,
    top_k: usize,
    tokens: u64,
    predict_fanout: usize,
    predict_min_prob: f64,
    no_direct: bool,
    block_align: usize,
    seed: u64,
    dtype: crate::inference::WeightDtype,
    partial_load_fraction: f64,
    pin_after_observations: u64,
    alias_map_path: Option<PathBuf>,
    io_uring: bool,
    token_pause_us: u64,
    first_token: Vec<u32>,
    no_prefetch: bool,
    io_only: bool,
    force_ssd: bool,
    router_clusters: usize,
    router_intra_p: f64,
    router_matrix: Option<PathBuf>,
    gate_weights: Option<PathBuf>,
    trace_out: Option<PathBuf>,
    gpu_cache_mb: Option<usize>,
    pipeline_depth: u32,
    speculator: bool,
    speculator_hidden_dim: usize,
    speculator_top_k: usize,
    locality: bool,
    locality_window: usize,
    locality_threshold_pct: f32,
    affinity: bool,
    affinity_neighbors_k: usize,
    affinity_decay_epoch: u64,
    prefetch_governor: bool,
    prefetch_precision_floor: f64,
    prefetch_contention_weight: f64,
    cost_aware_eviction: bool,
    pregate: bool,
    static_residency_fraction: f64,
    static_residency_warmup_tokens: u64,
    static_residency_profile: Option<String>,
    profile_out: Option<String>,
    workload: String,
    zipf_s: f64,
    workload_correlation: f64,
    replay_trace: Option<String>,
    num_layers: u32,
    num_experts_per_layer: Option<u32>,
    packed_blob: Option<PathBuf>,
    packed_manifest: Option<PathBuf>,
}

async fn cmd_run(
    mut args: RunArgs,
    startup_pinned: bool,
    progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    // 0) If `metadata.json` exists alongside the expert blobs (e.g. as
    //    written by `scripts/extract_mixtral_experts.py`), use it to fill
    //    in any args the user didn't override on the command line. We
    //    detect "user didn't override" by comparing against clap defaults
    //    — anyone who actually passes a flag overrides the metadata.
    apply_metadata_if_present(&mut args);

    let execution_context = if let Some(gpu_cache_mb) = args.gpu_cache_mb {
        install_run_gpu_backend(
            gpu_cache_mb,
            crate::backend::RoutedExpertGpuSpec {
                dtype: args.dtype,
                d_model: args.d_model,
                d_ff: args.d_ff,
            },
        )?
    } else {
        crate::backend::current_execution_context()
    };

    let weight_bytes = expert_weight_bytes_for(args.d_model, args.d_ff, args.dtype);
    let reported_weight_bytes = if args.dtype == crate::inference::WeightDtype::Mixed {
        args.expert_size
    } else {
        weight_bytes
    };
    info!(
        num_experts = args.num_experts,
        top_k = args.top_k,
        cache_slots = args.cache_slots,
        expert_mib = args.expert_size as f64 / (1024.0 * 1024.0),
        d_model = args.d_model,
        d_ff = args.d_ff,
        weight_mib = reported_weight_bytes as f64 / (1024.0 * 1024.0),
        direct_io = !args.no_direct,
        block_align = args.block_align,
        io_only = args.io_only,
        force_ssd = args.force_ssd,
        "starting engine"
    );

    if args.cache_slots > 16 {
        warn!(
            cache_slots = args.cache_slots,
            "--cache-slots is larger than 16. The whole point of this engine is to \
             stream experts from SSD; a large in-RAM cache hides exactly the metric \
             you're trying to measure. Consider 4-8."
        );
    }

    // macOS / non-Linux: O_DIRECT is not available. Force the user (or the
    // run config) into buffered reads and explain what that means for the
    // measurements.
    //
    // Note: the if-else branch selection is decided once at line entry —
    // the `args.no_direct = true` mutation inside the `if` body does NOT
    // retroactively flip the condition. The `else` branch fires when the
    // user *already* passed `--no-direct` on the command line.
    #[cfg(not(target_os = "linux"))]
    {
        if !args.no_direct {
            warn!(
                "O_DIRECT is not supported on this OS (Linux-only). Falling back \
                 to buffered reads (`--no-direct`); measured I/O latency therefore \
                 includes OS page-cache effects and will under-report cold-read \
                 latency on a real NVMe device."
            );
            args.no_direct = true;
        } else {
            warn!(
                "Running with `--no-direct` (buffered reads). Measured I/O latency \
                 includes OS page-cache effects."
            );
        }
        if args.force_ssd {
            warn!(
                "`--force-ssd` was requested but O_DIRECT is unavailable on this OS. \
                 Running in best-effort mode: the OS may still serve some reads from \
                 the page cache. Use a Linux host on a real NVMe device for a clean \
                 measurement."
            );
        }
    }

    #[cfg(target_os = "linux")]
    {
        if args.force_ssd && args.no_direct {
            return Err(
                "--force-ssd requires O_DIRECT (do not pass --no-direct alongside it). \
                 With buffered reads the OS page cache can serve repeats from RAM, \
                 which defeats the SSD-bandwidth measurement."
                    .into(),
            );
        }
        if args.no_direct {
            warn!(
                "Running with `--no-direct` (buffered reads). I/O latency in the \
                 summary includes OS page-cache effects."
            );
        }
    }

    if args.expert_size % args.block_align != 0 {
        return Err(format!(
            "expert_size ({}) must be a multiple of block_align ({}) for O_DIRECT",
            args.expert_size, args.block_align
        )
        .into());
    }
    if weight_bytes > 0 && weight_bytes > args.expert_size {
        return Err(format!(
            "expert_size ({}) is too small for the SwiGLU weights of d_model={}, \
             d_ff={} ({} bytes). Increase --expert-size or shrink --d-model / --d-ff \
             so it matches what gen-data wrote.",
            args.expert_size, args.d_model, args.d_ff, weight_bytes
        )
        .into());
    }
    // Multi-drive striping (gist Phase 4). If `--data-dir` contains
    // commas (e.g. `--data-dir /mnt/nvme0,/mnt/nvme1`), we shard
    // experts across the listed directories by `id % n_drives`. The
    // single-dir path is unchanged. Done early because the io_uring
    // NUMA probe below also takes the (canonical) data dir.
    let data_dirs: Vec<PathBuf> = parse_striped_data_dir(&args.data_dir)?;
    let primary_dir = data_dirs
        .first()
        .cloned()
        .unwrap_or_else(|| args.data_dir.clone());
    for d in &data_dirs {
        if !d.is_dir() {
            return Err(format!(
                "data dir {} does not exist; run `gen-data` first",
                d.display()
            )
            .into());
        }
    }
    if data_dirs.len() > 1 {
        info!(
            drives = data_dirs.len(),
            dirs = ?data_dirs,
            "multi-drive striping enabled (experts sharded by id % n_drives)"
        );
    }
    // Treat the first dir as the canonical metadata source for any
    // `metadata.json` / `alias-map` lookups downstream. The other
    // directories only need to contain `expert_<id>.bin`.
    args.data_dir = primary_dir.clone();
    crate::gguf_loader::validate_q4_0_dataset_layout(&args.data_dir, args.dtype)?;

    if args.io_uring {
        // CPU placement is a startup-level decision now. If the operator
        // supplied `--cpu-mask` / `[performance].cpu_mask` / legacy
        // `MER_PIN_CORES`, Rayon, Tokio, and io_uring setup all inherit that
        // same mask. With no startup mask, do not repin here: a late repin
        // would make the already-created Rayon pool disagree with the main
        // thread's placement.
        if startup_pinned {
            info!("startup CPU placement already applied; io_uring inherits it");
        } else {
            info!("no startup CPU mask requested; io_uring will not apply a late process repin");
        }
        #[cfg(all(target_os = "linux", feature = "io_uring"))]
        {
            // Best-effort: detect which NUMA node the data dir's
            // backing block device sits on, and ask the io_uring
            // backend to pin its constructing thread there. The
            // detection function is a no-op on systems where the
            // sysfs entries are missing — it just leaves
            // `numa_node = None` and `IoUringStorage::new` skips
            // pinning entirely.
            let numa_node = detect_data_dir_numa_node(&args.data_dir);
            if let Some(n) = numa_node {
                info!(
                    numa_node = n,
                    "detected NUMA node for data dir; will pin io_uring"
                );
            }
            // Build the backend from the same pool we'll hand the
            // engine; the registration happens inside ::new(). We log
            // the result and then continue with the portable backend
            // for the actual generate() loop — `IoUringStorage` is a
            // drop-in alternative read API (`read_expert_fixed` /
            // `read_experts_batch_fixed`) that callers can wire into
            // their own `Storage` impl. Validating it here gives users
            // a clear error path on misconfigured kernels without
            // reaching the hot path.
            let probe_pool = crate::buffer_pool::BufferPool::new(
                args.cache_slots.max(1),
                args.expert_size,
                args.block_align,
            );
            match crate::io_uring_storage::IoUringStorage::new(
                crate::io_uring_storage::IoUringConfig {
                    base_path: args.data_dir.clone(),
                    expert_size: args.expert_size,
                    block_align: args.block_align,
                    queue_depth: 64,
                    numa_node,
                },
                &probe_pool,
            ) {
                Ok(s) => info!(
                    registered_buffers = s.registered_buffers(),
                    "io_uring backend initialised: registered fixed buffers + ring ready. \
                     The engine still drives reads through the portable pread path; \
                     IoUringStorage::read_experts_batch_fixed is available for \
                     custom integrations."
                ),
                Err(e) => warn!(
                    error = %e,
                    "io_uring backend probe failed (kernel may not support it); \
                     continuing with the portable pread backend."
                ),
            }
        }
        #[cfg(not(all(target_os = "linux", feature = "io_uring")))]
        {
            warn!(
                "--io-uring was passed but this binary was built without the \
                 `io_uring` cargo feature (or is not on Linux). Falling back \
                 to the default `pread(2)` storage backend. Rebuild on Linux \
                 with `--features io_uring` to enable."
            );
        }
    }

    let storage_cfg = StorageConfig {
        base_path: primary_dir.clone(),
        expert_size: args.expert_size,
        block_align: args.block_align,
        use_direct_io: !args.no_direct,
        // The CLI `generate` path is a single-namespace benchmark
        // (`gen-data` produces `expert_<id>.bin`); the multi-layer
        // fallback is only relevant to the `serve` HF-extractor path.
        // `--num-experts-per-layer` opts a `run` into the same
        // layer-qualified geometry so `speculate_layer_ahead` can
        // restrict the speculator head per layer and prefetch ahead.
        num_experts_per_layer: args.num_experts_per_layer,
    };
    let storage = if data_dirs.len() > 1 {
        NvmeStorage::striped(storage_cfg, data_dirs.clone())?
    } else {
        NvmeStorage::new(storage_cfg)?
    };
    // Tier 2: attach the packed blob if configured (defaults: no-op).
    let storage = maybe_attach_packed_blob(
        storage,
        args.packed_blob.as_deref(),
        args.packed_manifest.as_deref(),
        !args.no_direct,
        args.expert_size,
    )?;
    let storage = Arc::new(storage);
    if !storage.is_packed() {
        storage.warmup_fds(0..args.num_experts)?;
    }

    let pipeline_depth = args.pipeline_depth.max(1) as usize;
    let prefetch_headroom = if args.no_prefetch || args.predict_fanout == 0 {
        0
    } else {
        // Scale the speculative headroom by the look-ahead pipeline depth:
        // a depth-N windowed look-ahead (`speculate_layer_ahead` priming
        // `layer + 1 ..= layer + pipeline_depth`) needs a shadow buffer per
        // in-flight layer. The prefetch semaphore is derived from this
        // shadow capacity in `Engine::with_options`, so it scales with it.
        args.predict_fanout.saturating_mul(pipeline_depth)
    };
    // Double-buffered pool: primary (Buffer A) = resident LRU + one
    // reserved foreground slot; shadow (Buffer B) = speculative
    // look-ahead prefetches (sized to `predict_fanout * pipeline_depth`).
    // See `cmd_serve` for the full rationale. `--no-prefetch` (headroom 0)
    // disables Buffer B and keeps the legacy single-pool layout.
    let shadow_slots = prefetch_headroom;
    let primary_slots = args.cache_slots + 1;
    let pool_slots = primary_slots + shadow_slots;

    // Rough RAM heuristic: we don't want to pin more than ~1/4 of total
    // RAM in the buffer pool. This is *advisory* — we warn rather than
    // hard-fail because the user may know their environment better than
    // our /proc/meminfo guess. Skip silently if we can't read RAM.
    if let Some(total_ram) = total_ram_bytes() {
        let pool_bytes = pool_slots as u64 * args.expert_size as u64;
        let budget = total_ram / 4;
        if pool_bytes > budget {
            warn!(
                pool_mib = pool_bytes / (1024 * 1024),
                budget_mib = budget / (1024 * 1024),
                total_ram_mib = total_ram / (1024 * 1024),
                "buffer pool ({} slots × {:.1} MiB/expert) exceeds 1/4 of total RAM. \
                 Lower --cache-slots / --predict-fanout / --pipeline-depth or risk OOM / heavy swapping.",
                pool_slots,
                args.expert_size as f64 / (1024.0 * 1024.0)
            );
        }
    }

    info!(
        cache_slots = args.cache_slots,
        pool_slots = pool_slots,
        prefetch_headroom = prefetch_headroom,
        pipeline_depth = pipeline_depth,
        "buffer pool sized with prefetch headroom (shadow = predict_fanout × pipeline_depth)"
    );
    let pool = if shadow_slots > 0 {
        BufferPool::new_with_shadow(
            primary_slots,
            shadow_slots,
            args.expert_size,
            args.block_align,
        )
    } else {
        BufferPool::new(primary_slots, args.expert_size, args.block_align)
    };
    let cache = Arc::new(MultiLayerExpertCache::single_layer(args.cache_slots));

    // Build the Markov router. If the user supplied a precomputed matrix
    // (e.g. derived from a real Mixtral routing trace), prefer that;
    // otherwise generate a clustered transition matrix.
    let router = if let Some(path) = args.router_matrix.as_ref() {
        info!(matrix = %path.display(), "loading router transition matrix from file");
        Arc::new(TopKRouter::from_matrix_file(
            path,
            args.num_experts,
            args.top_k,
            args.seed,
        )?)
    } else {
        info!(
            clusters = args.router_clusters,
            intra_cluster_p = args.router_intra_p,
            "router: deterministic Markov chain with structured cluster locality"
        );
        Arc::new(TopKRouter::clustered(
            args.num_experts,
            args.top_k,
            args.router_clusters,
            args.router_intra_p,
            args.seed,
        ))
    };
    let predictor = Arc::new(PredictiveLoader::new(
        args.num_experts,
        if args.no_prefetch {
            0
        } else {
            args.predict_fanout
        },
        resolve_predict_min_prob(args.predict_min_prob, args.num_experts),
        args.seed,
    ));

    let engine = Arc::new({
        let mut base = Engine::with_options_and_execution_context(
            cache.clone(),
            pool.clone(),
            storage.clone(),
            crate::gating::Router::Markov(router.clone()),
            predictor.clone(),
            ModelShape {
                d_model: args.d_model,
                d_ff: args.d_ff,
                hidden_seed: args.seed,
            },
            EngineOptions {
                io_only: args.io_only,
                dtype: args.dtype,
                partial_load_fraction: args.partial_load_fraction,
                pin_after_observations: args.pin_after_observations,
                use_qmm_for_q4: true,
                expert_execution_policy: crate::engine::ExpertExecutionPolicy::Auto,
                max_concurrent_prefetches: 64,
                max_fetch_yields: crate::engine::DEFAULT_MAX_FETCH_YIELDS,
                prefetch_governor: args.prefetch_governor,
                prefetch_precision_floor: args.prefetch_precision_floor,
                prefetch_contention_weight: args.prefetch_contention_weight,
                cost_aware_eviction: args.cost_aware_eviction,
                pregate_enabled: args.pregate,
                collect_route_profile: args.profile_out.is_some(),
                // The synthetic `run` benchmark path keeps the legacy
                // drop-from-mixture behaviour: a corrupt synthetic
                // expert must not abort a long streaming benchmark.
                policy: crate::inference::RealInferencePolicy {
                    allow_degraded_experts: true,
                    ..crate::inference::RealInferencePolicy::STRICT
                },
            },
            execution_context.clone(),
        );
        if execution_context.plan().routed_experts() == crate::backend::ExecutionPlane::Gpu {
            base.install_gpu_cache();
        }
        // Apply the configured look-ahead pipeline depth (sized in tandem
        // with the shadow buffer-pool budget above). No-op for the legacy
        // Markov path (no speculator installed); takes effect when a
        // speculator drives `speculate_layer_ahead`.
        let mut base = base.with_pipeline_depth(args.pipeline_depth);
        // Predictive arms (opt-in, mirroring `cmd_serve`'s `[predictive]`
        // wiring). These are what turn the speculative-I/O union-fetch
        // `E = S ∪ L ∪ M` from "Markov-only" into the full predictor:
        //   * M — neural speculator over the residual stream (also the
        //     only arm that drives `speculate_layer_ahead` look-ahead),
        //   * L — sliding-window locality monitor whose hot set is pinned
        //     (frequency-aware eviction on top of plain LRU),
        //   * affinity — per-layer co-occurrence + disk-adjacency fold.
        // All off by default so the legacy benchmark is bit-for-bit; turn
        // them on to measure whether they move the hit rate / I/O share.
        if args.locality {
            // Mirror `cmd_serve`: when the run uses a layer-qualified
            // namespace, scale the window by the layer count so the
            // per-layer history depth matches the configured value
            // (see `effective_locality_threshold` in engine.rs).
            let num_layers = args
                .num_experts_per_layer
                .filter(|&p| p > 0)
                .map(|p| args.num_experts.div_ceil(p).max(1) as usize)
                .unwrap_or(1);
            let monitor = Arc::new(LocalityMonitor::new(
                args.num_experts,
                args.locality_window.saturating_mul(num_layers),
            ));
            base = base.with_locality_monitor(monitor, args.locality_threshold_pct);
        }
        if args.speculator {
            let top_k = if args.speculator_top_k == 0 {
                args.top_k
            } else {
                args.speculator_top_k
            };
            let spec = Arc::new(NeuralSpeculator::new(
                args.d_model,
                args.speculator_hidden_dim,
                args.num_experts,
                args.seed,
            ));
            base = base.with_speculator(spec, top_k);
        }
        if args.affinity {
            // The affinity arm is only consulted on the layer-qualified
            // `moe_step` path (the `--gate-weights` / multi-layer route);
            // the flat single-namespace `generate` benchmark never folds
            // it in. Warn rather than silently no-op when the user asks
            // for affinity without a layer geometry.
            if args.num_experts_per_layer.is_none() {
                warn!(
                    "--affinity has no effect without --num-experts-per-layer: the \
                     affinity fold only runs on the layer-qualified moe_step path. \
                     Pass --num-experts-per-layer (and typically --gate-weights) to \
                     exercise it."
                );
            }
            let per_layer = args.num_experts_per_layer.unwrap_or(args.num_experts);
            let affinity = Arc::new(LayeredExpertAffinity::new(
                args.num_layers.max(1) as usize,
                per_layer,
            ));
            base = base.with_affinity(
                affinity,
                args.affinity_neighbors_k,
                args.affinity_decay_epoch,
            );
        }
        // Tier 1 — static residency. Pin the hottest `fraction` of the
        // expert namespace permanently. With `--static-residency-profile`
        // the hot set comes from an offline popularity profile (warm at
        // startup); otherwise it is derived online from route counts after
        // `--static-residency-warmup-tokens`.
        if args.static_residency_fraction > 0.0 {
            let profile = match args.static_residency_profile.as_ref() {
                Some(path) => {
                    let p =
                        crate::residency::ResidencyProfile::load_json(std::path::Path::new(path))?;
                    info!(
                        path = %path,
                        experts = p.len(),
                        "loaded static-residency popularity profile"
                    );
                    Some(p)
                }
                None => None,
            };
            base = base.with_static_residency(
                args.static_residency_fraction,
                args.static_residency_warmup_tokens,
                profile,
            );
        }
        // Tier 3 — per-layer pre-gate. Predict (and prefetch) the next
        // layer's experts from the current layer's routed set. Only
        // effective on the multi-layer `moe_step` path; warn when no
        // layer geometry is configured so it can't actually fire.
        if args.pregate {
            if args.num_layers <= 1 {
                warn!(
                    "--pregate has no effect with --num-layers 1: the pre-gate predicts \
                     the *next* layer's experts, so it needs a multi-layer geometry \
                     (set --num-layers > 1, typically with --gate-weights / a real model)."
                );
            }
            let pregate = Arc::new(crate::pregate::PerLayerPreGate::new(
                args.num_layers.max(1) as usize,
                args.top_k,
            ));
            base = base.with_pregate(pregate);
        }
        // Optional alias map (Change 6: expert deduplication).
        match args.alias_map_path.as_ref() {
            Some(path) => {
                let map = load_alias_map(path)?;
                info!(
                    path = %path.display(),
                    entries = map.len(),
                    "loaded expert alias map (deduplicated experts share resident copies)"
                );
                base.with_alias_map(map)
            }
            None => base,
        }
    });

    // Optional warm-up to mirror the spec example ("the router selects
    // Expert ID 3 and 7"): fetch those experts up front so the first real
    // token routes against an already-warm cache.
    if !args.first_token.is_empty() {
        let target = router.fixed(&args.first_token);
        info!(experts = ?target, "warm-up fetch (mirrors spec example)");
        engine.warm_with(&target).await?;
    }

    // Optional JSONL routing trace (gist Phase 6). When set, every
    // call to `engine.generate` appends one record. Wired up *after*
    // the warm-up so warm-fetched experts don't pollute the trace
    // with synthetic tokens (`Engine::warm_with` doesn't go through
    // `generate`).
    let trace_writer = match args.trace_out.as_ref() {
        Some(path) => {
            info!(path = %path.display(), "writing routing trace");
            let w = Arc::new(crate::engine::TraceWriter::open(path)?);
            engine.set_trace_writer(Some(w.clone()));
            Some(w)
        }
        None => None,
    };

    let stream_started = Instant::now();
    info!(
        tokens = args.tokens,
        "streaming tokens (latency / throughput logs follow)"
    );
    let autotune_probe = std::env::var_os(crate::rayon_autotune::AUTOTUNE_PROBE_ENV).is_some();
    let mut token_cycle_us =
        autotune_probe.then(|| Vec::with_capacity(args.tokens.min(1_000_000) as usize));

    // Optional production gating network. When present, every token's
    // expert ids come from `softmax(W_gate · x) → top-K` (real Mixtral
    // routing) instead of the deterministic Markov `TopKRouter`. The
    // SSD-streaming substrate is identical either way — only the *id
    // selection* changes — so the cycle / I/O / hit-rate metrics are
    // directly comparable across the two paths.
    let gate: Option<crate::gating::LinearGate> = match args.gate_weights.as_ref() {
        Some(path) => {
            info!(
                gate_weights = %path.display(),
                num_experts = args.num_experts,
                d_model = args.d_model,
                top_k = args.top_k,
                "loading gating-network weight matrix"
            );
            Some(load_gate_weights(
                path,
                args.num_experts as usize,
                args.d_model,
                args.top_k,
            )?)
        }
        None => None,
    };

    // Benchmark workload selection (Tier 1/3 falsifiability). `synthetic`
    // keeps the legacy uniform-i.i.d. stream; `skewed`/`replay` drive
    // `moe_step` with an explicit, structured expert set so the
    // skew-aware and correlation-aware machinery is exercisable.
    let workload = crate::workload::Workload::from_str_opt(&args.workload).ok_or_else(|| {
        format!(
            "--workload: unknown value {:?} (use 'synthetic', 'skewed', or 'replay')",
            args.workload
        )
    })?;
    let mut skewed_stream = if workload == crate::workload::Workload::Skewed {
        info!(
            zipf_s = args.zipf_s,
            correlation = args.workload_correlation,
            top_k = args.top_k,
            "workload: skewed (Zipf popularity + Markov correlation)"
        );
        Some(crate::workload::SkewedStream::new(
            args.num_experts,
            args.top_k,
            args.zipf_s,
            args.workload_correlation,
            args.seed,
        ))
    } else {
        None
    };
    let mut replay_stream = if workload == crate::workload::Workload::Replay {
        let path = args
            .replay_trace
            .as_ref()
            .ok_or("--workload replay requires --replay-trace <path>")?;
        let stream = crate::workload::ReplayStream::load(std::path::Path::new(path))?;
        if stream.is_empty() {
            return Err(format!("--replay-trace {path}: no usable routing records").into());
        }
        info!(path = %path, records = stream.len(), "workload: replay JSONL routing trace");
        Some(stream)
    } else {
        None
    };

    for t in 0..args.tokens {
        let start = Instant::now();
        let stats = with_progress_timeout(format!("run token {t}"), progress_watchdog, async {
            let stats = match workload {
                // Structured workloads: drive `moe_step` with the harness's
                // explicit expert set and measure the engine-counter delta.
                crate::workload::Workload::Skewed | crate::workload::Workload::Replay => {
                    let (tok_idx, layer_idx, experts): (u64, u32, Vec<u32>) = match workload {
                        crate::workload::Workload::Skewed => (
                            t,
                            0,
                            skewed_stream
                                .as_mut()
                                .expect("skewed stream")
                                .next_experts(),
                        ),
                        _ => {
                            let record = replay_stream
                                .as_mut()
                                .expect("replay stream")
                                .next_record()
                                .expect("replay stream non-empty");
                            let layer = u32::try_from(record.layer).map_err(|_| {
                                format!("replay layer {} does not fit in u32", record.layer)
                            })?;
                            (record.token, layer, record.experts)
                        }
                    };
                    let hidden =
                        crate::inference::synth_hidden_state(tok_idx, args.d_model, args.seed);
                    let pre = engine.report();
                    let _ = engine.moe_step(tok_idx, layer_idx, &hidden, &experts).await;
                    let post = engine.report();
                    crate::engine::CycleStats {
                        hits: post.hits.saturating_sub(pre.hits),
                        misses: post.misses.saturating_sub(pre.misses),
                        prefetch_hits: 0,
                        bytes_read: post.bytes_read.saturating_sub(pre.bytes_read),
                    }
                }
                crate::workload::Workload::Synthetic => {
                    if let Some(gate) = gate.as_ref() {
                        // Real gating-network path. Hidden state is the same
                        // synthetic activation `Engine::generate` would have
                        // used, so the only difference relative to the legacy
                        // path is *which* experts are selected.
                        let hidden =
                            crate::inference::synth_hidden_state(t, args.d_model, args.seed);
                        let dec = gate.route(&hidden);
                        let pre = engine.report();
                        let _ = engine.moe_step(t, 0, &hidden, &dec.experts).await;
                        let post = engine.report();
                        crate::engine::CycleStats {
                            hits: post.hits.saturating_sub(pre.hits),
                            misses: post.misses.saturating_sub(pre.misses),
                            prefetch_hits: 0,
                            bytes_read: post.bytes_read.saturating_sub(pre.bytes_read),
                        }
                    } else {
                        engine.generate(t).await?
                    }
                }
            };
            Ok::<crate::engine::CycleStats, Box<dyn std::error::Error>>(stats)
        })
        .await?;
        let elapsed = start.elapsed();
        if let Some(samples) = token_cycle_us.as_mut() {
            samples.push(elapsed.as_micros() as u64);
        }
        let throughput = if elapsed.as_secs_f64() > 0.0 {
            1.0 / elapsed.as_secs_f64()
        } else {
            f64::INFINITY
        };
        if !autotune_probe {
            info!(
                token = t,
                cycle_us = elapsed.as_micros() as u64,
                tps = format!("{throughput:.1}"),
                hits = stats.hits,
                misses = stats.misses,
                kib = stats.bytes_read / 1024,
                resident = ?cache.resident_ids(),
                "tick"
            );
        }
        if args.token_pause_us > 0 {
            tokio::time::sleep(Duration::from_micros(args.token_pause_us)).await;
        }
    }

    let wall = stream_started.elapsed();
    let r = engine.report();
    let total_lookups = (r.hits + r.misses).max(1);
    info!(
        wall_s = wall.as_secs_f64(),
        sustained_tps = args.tokens as f64 / wall.as_secs_f64(),
        avg_throughput_mibps = (r.bytes_read as f64 / (1024.0 * 1024.0)) / wall.as_secs_f64(),
        hit_rate_pct = (r.hits as f64 / total_lookups as f64) * 100.0,
        "stream complete"
    );
    if autotune_probe {
        let token_cycle_us = token_cycle_us
            .as_mut()
            .expect("autotune probe samples initialized");
        token_cycle_us.sort_unstable();
        let report = crate::rayon_autotune::RayonAutotuneProbeResult {
            threads: crate::parallel::num_threads(),
            valid: true,
            p50_ms: crate::rayon_autotune::percentile_ms(&token_cycle_us, 0.50),
            p95_ms: crate::rayon_autotune::percentile_ms(&token_cycle_us, 0.95),
            p99_ms: Some(crate::rayon_autotune::percentile_ms(&token_cycle_us, 0.99)),
            sustained_tps: args.tokens as f64 / wall.as_secs_f64(),
        };
        println!(
            "MER_RAYON_AUTOTUNE_RESULT {}",
            serde_json::to_string(&report)?
        );
    }
    engine.print_summary();

    if r.misses > 0 && r.io_p50_us == 0 {
        warn!(
            "I/O latency histogram is empty despite cache misses; check that \
             tracing is enabled and runs are long enough to produce samples."
        );
    }

    // Flush the trace before returning so the JSONL file is complete.
    if let Some(tw) = trace_writer.as_ref() {
        tw.flush();
    }

    // Tier 1 — emit the accumulated expert-popularity profile so a later
    // run can warm-start static residency with `--static-residency-profile`.
    if let Some(path) = args.profile_out.as_ref() {
        engine
            .dump_route_profile(std::path::Path::new(path))
            .map_err(|e| format!("failed to write route profile {}: {e}", path))?;
        info!(path = %path, "wrote route-observation profile");
    }

    Ok(())
}

/// Per-CLI defaults. We compare an `args` value to its default to detect
/// whether the user actually passed the flag, so `metadata.json` can fill
/// in just the values the user *didn't* override.
mod cli_defaults {
    pub const NUM_EXPERTS: u32 = 64;
    pub const EXPERT_SIZE: usize = 16 * 1024 * 1024;
    pub const D_MODEL: usize = 512;
    pub const D_FF: usize = 2048;
    pub const TOP_K: usize = 2;
    pub const BLOCK_ALIGN: usize = 4096;
}

/// Hand-rolled `metadata.json` parser. The only fields we care about are
/// numeric scalars (`num_experts`, `d_model`, `d_ff`, `top_k`,
/// `expert_size`); pulling in `serde_json` for that would add a heavy
/// dependency the rest of the engine doesn't need.
fn apply_metadata_if_present(args: &mut RunArgs) {
    let path = args.data_dir.join("metadata.json");
    let body = match std::fs::read_to_string(&path) {
        Ok(b) => b,
        Err(_) => return,
    };
    info!(path = %path.display(), "found metadata.json — auto-filling unspecified args");
    let mut overrode_anything = false;
    let mut set_if_default = |key: &str, current: u64, default: u64, sink: &mut dyn FnMut(u64)| {
        if let Some(v) = parse_json_number(&body, key) {
            // Only fill in values the user didn't override on the CLI.
            if current == default {
                sink(v);
                overrode_anything = true;
            } else if v != current {
                warn!(
                    key,
                    metadata = v,
                    cli = current,
                    "CLI value overrides metadata.json"
                );
            }
        }
    };
    set_if_default(
        "num_experts",
        args.num_experts as u64,
        cli_defaults::NUM_EXPERTS as u64,
        &mut |v| args.num_experts = v as u32,
    );
    set_if_default(
        "d_model",
        args.d_model as u64,
        cli_defaults::D_MODEL as u64,
        &mut |v| args.d_model = v as usize,
    );
    set_if_default(
        "d_ff",
        args.d_ff as u64,
        cli_defaults::D_FF as u64,
        &mut |v| args.d_ff = v as usize,
    );
    set_if_default(
        "top_k",
        args.top_k as u64,
        cli_defaults::TOP_K as u64,
        &mut |v| args.top_k = v as usize,
    );
    set_if_default(
        "expert_size",
        args.expert_size as u64,
        cli_defaults::EXPERT_SIZE as u64,
        &mut |v| args.expert_size = v as usize,
    );
    set_if_default(
        "block_align",
        args.block_align as u64,
        cli_defaults::BLOCK_ALIGN as u64,
        &mut |v| args.block_align = v as usize,
    );
    if args.dtype == crate::inference::WeightDtype::F32 {
        if let Some(dtype_str) = parse_json_string(&body, "dtype") {
            if let Some(dtype) = crate::inference::WeightDtype::from_str_opt(&dtype_str) {
                args.dtype = dtype;
                overrode_anything = true;
            }
        }
    }
    if overrode_anything {
        info!(
            num_experts = args.num_experts,
            d_model = args.d_model,
            d_ff = args.d_ff,
            top_k = args.top_k,
            expert_mib = args.expert_size as f64 / (1024.0 * 1024.0),
            "engine parameters after metadata.json"
        );
    }
}

/// Look up `"key": <number>` in a JSON document and return the integer.
/// Tolerates whitespace and surrounding quotes; returns `None` if the
/// key is missing or the value is non-integer / negative. Good enough
/// for the small handful of scalars in `metadata.json`.
fn parse_json_number(body: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\"");
    let pos = body.find(&needle)?;
    let after = &body[pos + needle.len()..];
    let after = after.trim_start();
    let after = after.strip_prefix(':')?;
    let after = after.trim_start();
    let mut end = 0;
    for (i, c) in after.char_indices() {
        if c.is_ascii_digit() {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    after[..end].parse::<u64>().ok()
}

fn parse_json_string(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let start = body.find(&needle)? + needle.len();
    let rest = body[start..].trim_start();
    let rest = rest.strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Parse a tiny JSON object of the form `{ "src_id": canonical_id, ... }`
/// into a `HashMap<u32, u32>`. Hand-rolled to keep `serde_json` out of
/// the engine's dep tree (the rest of the engine uses our smaller
/// `parse_json_number`-style helpers). Returns an error if the file
/// can't be read or contains a malformed entry.
fn load_alias_map(
    path: &std::path::Path,
) -> Result<std::collections::HashMap<u32, u32>, Box<dyn std::error::Error>> {
    let body = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read alias map {}: {e}", path.display()))?;
    let body = body.trim();
    let body = body
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| format!("alias map {} must be a JSON object", path.display()))?;
    let mut map = std::collections::HashMap::new();
    for raw in body.split(',') {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let (k, v) = raw
            .split_once(':')
            .ok_or_else(|| format!("alias map entry {raw:?} missing ':'"))?;
        // Strip optional whitespace + surrounding quotes around the key.
        let k = k.trim().trim_matches('"');
        let v = v.trim();
        let key: u32 = k
            .parse()
            .map_err(|_| format!("alias map key {k:?} must be a non-negative integer"))?;
        let val: u32 = v
            .parse()
            .map_err(|_| format!("alias map value {v:?} must be a non-negative integer"))?;
        map.insert(key, val);
    }
    Ok(map)
}

/// Load a real gating-network weight matrix from disk.
///
/// File format: bare little-endian `f32`s, no header, row-major,
/// `[num_experts × d_model]`. This is the layout `numpy.tofile` writes
/// for `block_sparse_moe.gate.weight` after `astype(np.float32)`. A
/// future PR can teach this to read `safetensors` directly so the user
/// can point it at a HuggingFace shard without a conversion step.
///
/// **Directory input.** When `path` is a directory rather than a file,
/// the per-layer `gate_<L>.bin` files inside it (the same naming the
/// real-model loader writes/reads in `model.rs`) are auto-discovered,
/// sorted ascending by layer index, and concatenated in layer order.
/// This is the in-memory equivalent of
/// `cat gate_0.bin gate_1.bin … gate_N.bin > real_gate.bin`, so users
/// can point `--gate-weights` straight at a model directory instead of
/// hand-concatenating a non-standard monolithic file.
fn load_gate_weights(
    path: &std::path::Path,
    num_experts: usize,
    d_model: usize,
    top_k: usize,
) -> Result<crate::gating::LinearGate, Box<dyn std::error::Error>> {
    let bytes = if path.is_dir() {
        read_gate_dir_concatenated(path)?
    } else {
        std::fs::read(path)
            .map_err(|e| format!("failed to read gate weights {}: {e}", path.display()))?
    };
    let expected = num_experts
        .checked_mul(d_model)
        .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "num_experts * d_model overflowed".to_string())?;
    if bytes.len() != expected {
        return Err(format!(
            "gate weights {} have {} bytes, expected {} ({} experts × {} d_model × 4 bytes/f32)",
            path.display(),
            bytes.len(),
            expected,
            num_experts,
            d_model
        )
        .into());
    }
    let mut weights = Vec::<f32>::with_capacity(num_experts * d_model);
    for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(chunk);
        weights.push(f32::from_le_bytes(buf));
    }
    Ok(crate::gating::LinearGate::new(
        weights,
        num_experts,
        d_model,
        top_k,
    ))
}

/// Discover and concatenate the per-layer `gate_<L>.bin` files in `dir`,
/// sorted ascending by layer index. Returns the concatenated raw bytes,
/// which [`load_gate_weights`] then validates against the expected
/// `num_experts × d_model × 4` total — exactly as if the caller had run
/// `cat gate_0.bin gate_1.bin … > real_gate.bin` first.
fn read_gate_dir_concatenated(
    dir: &std::path::Path,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut entries: Vec<(u32, PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| {
        format!(
            "failed to scan gate-weights directory {}: {e}",
            dir.display()
        )
    })? {
        let entry = entry
            .map_err(|e| format!("failed to read a directory entry in {}: {e}", dir.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if let Some(idx) = parse_gate_layer_index(name) {
            entries.push((idx, path));
        }
    }
    if entries.is_empty() {
        return Err(format!(
            "no gate_<layer>.bin files found in directory {}; expected per-layer files \
             named like gate_0.bin, gate_1.bin, … (each file is a little-endian f32 shard; concatenation must total [num_experts × d_model])",
            dir.display()
        )
        .into());
    }
    entries.sort_by_key(|(idx, _)| *idx);
    // Reject duplicate layer indices: the concatenation order would be
    // ambiguous and almost certainly indicates a stray file.
    for w in entries.windows(2) {
        if w[0].0 == w[1].0 {
            return Err(format!(
                "duplicate gate layer index {} in directory {} ({} and {})",
                w[0].0,
                dir.display(),
                w[0].1.display(),
                w[1].1.display()
            )
            .into());
        }
    }
    // `entries` is guaranteed non-empty here (early return above), so the
    // first/last layer indices are always present.
    let first_layer = entries.first().map(|(i, _)| *i).expect("entries non-empty");
    let last_layer = entries.last().map(|(i, _)| *i).expect("entries non-empty");
    info!(
        dir = %dir.display(),
        files = entries.len(),
        first_layer,
        last_layer,
        "discovered per-layer gate files; concatenating in ascending layer order"
    );
    let mut bytes = Vec::new();
    for (idx, p) in &entries {
        let mut chunk = std::fs::read(p).map_err(|e| {
            format!(
                "failed to read gate file {} (layer {idx}): {e}",
                p.display()
            )
        })?;
        bytes.append(&mut chunk);
    }
    Ok(bytes)
}

/// Parse the layer index `N` out of a `gate_<N>.bin` filename. Returns
/// `None` for any name that doesn't match that exact pattern (so
/// unrelated files in the directory are simply ignored).
fn parse_gate_layer_index(name: &str) -> Option<u32> {
    let idx = name.strip_prefix("gate_")?.strip_suffix(".bin")?;
    if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    idx.parse::<u32>().ok()
}

/// Best-effort total-RAM probe. Returns `None` (heuristic disabled) on
/// platforms or filesystems we don't recognise. We intentionally avoid
/// pulling in a `sysinfo`-style dependency for one number.
fn total_ram_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let body = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in body.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kib: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kib.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Detect the NUMA node of the block device backing `data_dir`.
///
/// Returns `Some(node)` when both probes succeed:
///   1. `stat(2)` on `data_dir` yields a device id whose major/minor
///      we map to `/sys/dev/block/MAJ:MIN/device/numa_node`.
///   2. The contents of that sysfs entry parse to a non-negative integer.
///
/// Any failure (non-Linux build, sysfs entry missing, NUMA disabled in
/// the kernel which reports `-1`, permission errors) returns `None`
/// and lets the caller continue without NUMA pinning. This is a
/// *hint*; it must never block startup.
pub fn detect_data_dir_numa_node(data_dir: &std::path::Path) -> Option<i32> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::MetadataExt;
        let md = std::fs::metadata(data_dir).ok()?;
        // st_dev is encoded as major:minor; major = (dev >> 8) & 0xfff
        // for the legacy layout but Linux uses a more flexible
        // encoding. libc::major()/minor() handle both.
        let dev = md.dev();
        // `libc::major` / `libc::minor` are safe `const fn`s in libc ≥ 0.2.156;
        // no `unsafe` block is required.
        let major = libc::major(dev) as u32;
        let minor = libc::minor(dev) as u32;
        let sys_path = format!("/sys/dev/block/{}:{}/device/numa_node", major, minor);
        let body = std::fs::read_to_string(&sys_path).ok()?;
        let node: i32 = body.trim().parse().ok()?;
        // Kernel reports `-1` when NUMA is disabled or unknown.
        if node < 0 {
            None
        } else {
            Some(node)
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = data_dir;
        None
    }
}

/// Parse `--data-dir` into a list of directories. If the path
/// stringifies to a comma-separated list, split it; otherwise return a
/// single-element vec. Used by gist Phase 4 (multi-drive striping).
fn parse_striped_data_dir(p: &std::path::Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let s = p.to_string_lossy();
    if s.contains(',') {
        let dirs: Vec<PathBuf> = s
            .split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(PathBuf::from)
            .collect();
        if dirs.is_empty() {
            return Err(format!(
                "invalid --data-dir '{}': comma-separated list must contain at least one \
                 non-empty directory path",
                p.display()
            )
            .into());
        }
        Ok(dirs)
    } else {
        Ok(vec![p.to_path_buf()])
    }
}

fn cmd_gguf_convert(
    gguf_path: &PathBuf,
    out_dir: &PathBuf,
    num_layers: usize,
    num_experts: usize,
    emit_uth: bool,
    legacy_eager: bool,
    native_quant: bool,
    experts_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    info!(
        path = %gguf_path.display(),
        emit_uth,
        legacy_eager,
        native_quant,
        experts_only,
        "opening GGUF file"
    );
    let opts = crate::gguf_loader::ExtractOptions {
        emit_uth,
        native_quant,
        experts_only,
        arch_override: None,
    };
    let source = crate::gguf::open_gguf_source(gguf_path, legacy_eager)?;
    if let Some(arch) = source.architecture() {
        info!(architecture = arch, "GGUF source opened");
    }
    let report = crate::gguf_loader::extract_experts_from_source(
        &*source,
        out_dir,
        num_layers,
        num_experts,
        opts,
    )?;
    let total_gib = report.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let read_time_at_7gbps = report.total_bytes as f64 / (7.0 * 1024.0 * 1024.0 * 1024.0);
    info!(
        experts_written = report.experts_written,
        dense_written = report.dense_written,
        skipped = report.skipped,
        total_bytes = report.total_bytes,
        total_gib,
        expected_read_seconds_at_7gbps = read_time_at_7gbps,
        d_model = report.d_model,
        d_ff = report.d_ff,
        num_layers = report.num_layers,
        num_experts_per_layer = report.num_experts_per_layer,
        "gguf-convert complete"
    );
    println!(
        "gguf-convert: wrote {} expert files + {} dense tensors ({:.2} GiB total). \
         At 7 GB/s aggregate SSD read bandwidth, a full warm-up scan would take ~{:.2}s.",
        report.experts_written, report.dense_written, total_gib, read_time_at_7gbps
    );
    Ok(())
}

fn cmd_validate_data(data_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let report = crate::gguf_loader::validate_data_dir(data_dir)?;
    println!(
        "validate-data: ok (experts={}, expert_size={} bytes, block_align={}, dtype={}, mixed_experts={})",
        report.num_experts,
        report.expert_size,
        report.block_align,
        report.dtype.as_str(),
        report.mixed_experts
    );
    Ok(())
}

async fn cmd_validate_predictor(
    trace_path: &PathBuf,
    cache_slots: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(trace_path)?;
    let text = String::from_utf8_lossy(&bytes);
    // Parse JSONL records {"token": .., "layer": .., "experts": [..], "cache_hit": [..]}.
    // We extract just the per-token expert id sequence; the predictor
    // validation replays them through a fresh LRU and prints per-K
    // hit rates plus per-layer breakdown and top-1 / top-2 accuracy.
    #[derive(Default)]
    struct LayerStats {
        tokens: u64,
        // for top-1 / top-2 accuracy we compare the predicted set of
        // size K against the actual top-1 / top-2 routed experts.
        top1_hits: u64,
        top2_hits: u64,
    }

    // Flat list of (token, layer, experts) records in the order
    // they were observed in the JSONL file. We rely on a stable
    // sort over the global `token` field to reconstruct the engine's
    // per-token, per-layer interleaving — even if a multi-layer
    // trace's records were appended in any order. Pre gist
    // feedback #2.2 we instead grouped by layer first and then
    // flattened, which produced "all of layer 0, then all of
    // layer 1, …" — meaningless on real multi-layer (e.g. Mixtral's
    // 32 layers) traces because the per-layer caches saw an entirely
    // synthetic recent-history.
    let mut records: Vec<(u64, u32, Vec<u32>)> = Vec::new();
    let mut by_layer: std::collections::BTreeMap<u32, LayerStats> = Default::default();
    for (file_idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Fall back to the file-line index when a record lacks
        // an explicit `token` so legacy traces still order
        // monotonically by appearance.
        let token = json_get_u64(line, "token").unwrap_or(file_idx as u64);
        let layer = json_get_u64(line, "layer").unwrap_or(0) as u32;
        let experts = json_get_u32_array(line, "experts");
        if experts.is_empty() {
            continue;
        }
        by_layer.entry(layer).or_default().tokens += 1;
        records.push((token, layer, experts));
    }
    // Stable sort by (token, layer) reconstructs the original
    // interleaved order the engine produced — for token T the
    // entries for layer 0, 1, 2, … appear in order, then the same
    // for token T+1, etc. — which is what the LRU saw in production.
    records.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

    // Per-cache-size simulation: maintain a single LRU shared across
    // *all* layers in the trace and count hits. This matches
    // `scripts/compute_transition_matrix.py::simulate_lru`, which
    // replays the trace through one global LRU rather than per-layer
    // caches — having both lets the Rust and Python paths produce
    // identical hit-rate numbers for the same trace.
    let ks: Vec<usize> = if cache_slots.is_empty() {
        vec![2, 4, 8, 16]
    } else {
        cache_slots.to_vec()
    };
    println!("validate-predictor: trace={}", trace_path.display());
    for k in &ks {
        let mut hits = 0u64;
        let mut total = 0u64;
        // Maintain order in a VecDeque and membership in a HashSet
        // so the per-token hit check is O(1) instead of O(N). The
        // VecDeque carries the LRU ordering (front = oldest); the
        // HashSet mirrors the same id set for fast `contains`. This
        // is the same hit-rate as before, just without the O(N·M)
        // walk over `lru.iter().any(...)` that the prior version
        // performed (gist feedback #2.5 — keeps `validate-predictor`
        // workable on long real-engine traces).
        let mut lru: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
        let mut lru_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for (_, _, experts) in &records {
            for &e in experts.iter() {
                if lru_set.contains(&e) {
                    hits += 1;
                    // Move-to-back: O(N) here but only on a hit
                    // (cheap relative to the surrounding miss-path).
                    if let Some(pos) = lru.iter().position(|x| *x == e) {
                        lru.remove(pos);
                    }
                } else if lru.len() == *k {
                    if let Some(evicted) = lru.pop_front() {
                        lru_set.remove(&evicted);
                    }
                }
                lru.push_back(e);
                lru_set.insert(e);
                total += 1;
            }
        }
        let rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        println!("  cache_slots={k:>3}  hit_rate={rate:>6.3}  hits={hits}/{total}");
    }

    // Group sorted records into per-layer buckets *after* the LRU
    // replay so we can consume `records` without cloning each `experts`
    // vector.
    let mut tokens_per_layer: std::collections::BTreeMap<u32, Vec<Vec<u32>>> = Default::default();
    for (_, layer, experts) in records.into_iter() {
        tokens_per_layer.entry(layer).or_default().push(experts);
    }

    // Top-1 / Top-2 predictor accuracy: replay one-step-ahead via a
    // simple last-expert Markov predictor (the cheapest baseline the
    // engine has). For each (prev, curr) pair we predict `prev` and
    // count it as a top-1 hit if it appears in `curr`, top-2 if any
    // of {prev, second-most-recent} appears in `curr`.
    for (layer, seq) in tokens_per_layer.iter() {
        let stats = by_layer.entry(*layer).or_default();
        let mut prev: Option<u32> = None;
        let mut prev2: Option<u32> = None;
        for experts in seq {
            if let Some(p) = prev {
                if experts.iter().any(|&x| x == p) {
                    stats.top1_hits += 1;
                }
                let predict2: std::collections::HashSet<u32> =
                    [Some(p), prev2].iter().filter_map(|x| *x).collect();
                if experts.iter().any(|x| predict2.contains(x)) {
                    stats.top2_hits += 1;
                }
            }
            prev2 = prev;
            prev = experts.first().copied();
        }
    }
    println!("\nper-layer Markov predictor accuracy:");
    for (layer, st) in &by_layer {
        let denom = st.tokens.saturating_sub(1).max(1);
        let top1 = st.top1_hits as f64 / denom as f64;
        let top2 = st.top2_hits as f64 / denom as f64;
        println!(
            "  layer={layer:>3}  tokens={:>6}  top1={top1:>6.3}  top2={top2:>6.3}",
            st.tokens
        );
    }
    Ok(())
}

/// Pull a numeric field and a `[..]` u32 array out of one JSONL line.
/// The trace records have a fixed schema (`{token, layer, experts,
/// cache_hit}`), so we route through `serde_json::Value` for safety
/// without paying the cost of deriving a full type.
fn json_get_u64(line: &str, key: &str) -> Option<u64> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    v.get(key).and_then(|x| x.as_u64())
}

fn json_get_u32_array(line: &str, key: &str) -> Vec<u32> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
        return Vec::new();
    };
    v.get(key)
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| e.as_u64().map(|n| n as u32))
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen_data_q4_0_records_standard_layout_metadata() {
        let dir = tempdir_unique("q4-standard-metadata");
        cmd_gen_data(&dir, 2, 4096, 32, 32, 4096, "q4_0").expect("generate Q4_0");
        let metadata: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("metadata.json")).expect("metadata.json"),
        )
        .expect("metadata JSON");
        assert_eq!(metadata["dtype"], "q4_0");
        assert_eq!(
            metadata["q4_0_layout"],
            crate::inference::Q4_0_LAYOUT_STANDARD_V1
        );
        crate::gguf_loader::validate_data_dir(&dir).expect("generated dataset validates");
        crate::gguf_loader::validate_q4_0_dataset_layout(&dir, crate::inference::WeightDtype::Q4_0)
            .expect("runtime accepts generated dataset");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cli_parses_rayon_threads_after_run_subcommand() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "run",
            "--rayon-threads",
            "30",
        ])
        .expect("global --rayon-threads should parse after subcommand");
        assert_eq!(cli.rayon_threads, Some(30));
    }

    #[test]
    fn cli_parses_cpu_mask_after_run_subcommand() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "run",
            "--cpu-mask",
            "0-24",
        ])
        .expect("global --cpu-mask should parse after subcommand");
        assert_eq!(cli.cpu_mask.as_deref(), Some("0-24"));
    }

    #[test]
    fn autotune_child_args_preserve_cpu_mask_and_strip_recursive_flags() {
        let raw = vec![
            OsString::from("micro-expert-router"),
            OsString::from("--log"),
            OsString::from("info"),
            OsString::from("run"),
            OsString::from("--cpu-mask"),
            OsString::from("0-24"),
            OsString::from("--autotune-rayon"),
            OsString::from("--autotune-tokens"),
            OsString::from("2000"),
            OsString::from("--autotune-repeats"),
            OsString::from("3"),
            OsString::from("--autotune-coarse-tokens"),
            OsString::from("512"),
            OsString::from("--autotune-top-candidates"),
            OsString::from("2"),
            OsString::from("--autotune-slow-p95-ms"),
            OsString::from("110"),
            OsString::from("--autotune-slow-p99-ms"),
            OsString::from("150"),
            OsString::from("--autotune-print-table"),
            OsString::from("--allow-low-confidence-rayon-autotune"),
            OsString::from("--tokens"),
            OsString::from("10000"),
        ];
        let args = autotune_child_args(&raw, 123, 25);
        let rendered: Vec<String> = args
            .iter()
            .map(|s| s.to_string_lossy().to_string())
            .collect();
        assert!(rendered
            .windows(2)
            .any(|w| w[0] == "--cpu-mask" && w[1] == "0-24"));
        assert!(!rendered.iter().any(|s| s == "--autotune-rayon"));
        assert!(!rendered.iter().any(|s| s == "--autotune-tokens"));
        assert!(!rendered.iter().any(|s| s == "--autotune-repeats"));
        assert!(!rendered.iter().any(|s| s == "--autotune-coarse-tokens"));
        assert!(!rendered.iter().any(|s| s == "--autotune-top-candidates"));
        assert!(!rendered.iter().any(|s| s == "--autotune-slow-p95-ms"));
        assert!(!rendered.iter().any(|s| s == "--autotune-slow-p99-ms"));
        assert!(!rendered.iter().any(|s| s == "--autotune-print-table"));
        assert!(!rendered
            .iter()
            .any(|s| s == "--allow-low-confidence-rayon-autotune"));
        assert!(rendered
            .windows(2)
            .any(|w| w[0] == "--tokens" && w[1] == "123"));
        assert!(rendered
            .windows(2)
            .any(|w| w[0] == "--rayon-threads" && w[1] == "25"));
    }

    #[test]
    fn cli_parses_autotune_stability_flags() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "run",
            "--autotune-rayon",
            "--autotune-tokens",
            "2000",
            "--autotune-repeats",
            "3",
            "--autotune-coarse-tokens",
            "512",
            "--autotune-top-candidates",
            "2",
            "--autotune-slow-p95-ms",
            "110",
            "--autotune-slow-p99-ms",
            "150",
            "--autotune-print-table",
            "--allow-low-confidence-rayon-autotune",
        ])
        .expect("autotune stability flags should parse");
        let Cmd::Run {
            autotune_rayon,
            autotune_tokens,
            autotune_repeats,
            autotune_coarse_tokens,
            autotune_top_candidates,
            autotune_slow_p95_ms,
            autotune_slow_p99_ms,
            autotune_print_table,
            allow_low_confidence_rayon_autotune,
            ..
        } = cli.cmd
        else {
            panic!("expected run command");
        };
        assert!(autotune_rayon);
        assert_eq!(autotune_tokens, 2000);
        assert_eq!(autotune_repeats, 3);
        assert_eq!(autotune_coarse_tokens, 512);
        assert_eq!(autotune_top_candidates, 2);
        assert_eq!(autotune_slow_p95_ms, 110.0);
        assert_eq!(autotune_slow_p99_ms, 150.0);
        assert!(autotune_print_table);
        assert!(allow_low_confidence_rayon_autotune);
    }

    #[test]
    fn cli_rejects_zero_rayon_threads() {
        let err = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "run",
            "--rayon-threads",
            "0",
        ])
        .expect_err("zero rayon threads must be rejected");
        assert!(
            err.to_string().contains("--rayon-threads must be > 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn low_confidence_rayon_profile_is_not_reused_without_opt_in() {
        let dir = tempdir_unique("rayon-low-confidence-profile");
        std::fs::create_dir_all(&dir).unwrap();
        let key = crate::rayon_autotune::CpuAutotuneKey {
            machine_fingerprint: "machine".to_string(),
            model_fingerprint: "model".to_string(),
            backend_fingerprint: "backend".to_string(),
        };
        let profile = crate::rayon_autotune::RayonAutotuneProfile {
            threads: 25,
            effective_cpu_mask: Some((0..25).collect()),
            effective_cpu_mask_display: Some("0-24".to_string()),
            logical_cores: 25,
            repeats: 1,
            p50_ms: 99.0,
            p95_ms: 101.0,
            p99_ms: Some(120.0),
            sustained_tps: 10.0,
            median_p50_ms: 99.0,
            worst_p95_ms: 101.0,
            worst_p99_ms: Some(120.0),
            median_sustained_tps: 10.0,
            confidence: crate::rayon_autotune::RayonAutotuneConfidence::Low,
            selection_reason: "slow-regime profile".to_string(),
            candidate_results: Vec::new(),
            probe_results: Vec::new(),
        };
        let path = crate::rayon_autotune::default_profile_path(&dir);
        crate::rayon_autotune::save_profile(&path, &key, profile).unwrap();

        let dir_arg = dir.to_string_lossy().to_string();
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "run",
            "--data-dir",
            dir_arg.as_str(),
        ])
        .unwrap();
        assert_eq!(load_profiled_rayon_threads(&cli, Some(&key)), None);

        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "--reuse-low-confidence-rayon-profile",
            "run",
            "--data-dir",
            dir_arg.as_str(),
        ])
        .unwrap();
        assert_eq!(load_profiled_rayon_threads(&cli, Some(&key)), Some(25));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn low_confidence_autotune_selection_requires_current_run_opt_in() {
        let selection = crate::rayon_autotune::RayonAutotuneSelection {
            selected: crate::rayon_autotune::RayonAutotuneCandidateSummary {
                threads: 25,
                requested_repeats: 2,
                successful_repeats: 2,
                all_repeats_successful: true,
                p95_below_slow_threshold: false,
                p99_below_slow_threshold: true,
                median_p50_ms: Some(99.0),
                worst_p95_ms: Some(101.0),
                worst_p99_ms: Some(120.0),
                median_sustained_tps: Some(10.0),
                p50_cv: Some(0.0),
                confidence: crate::rayon_autotune::RayonAutotuneConfidence::Low,
                rejection_reason: Some("worst p95 exceeds threshold".to_string()),
            },
            confidence: crate::rayon_autotune::RayonAutotuneConfidence::Low,
            reason: "low confidence".to_string(),
        };
        assert_eq!(selected_autotune_threads(&selection, false), None);
        assert_eq!(selected_autotune_threads(&selection, true), Some(25));
    }

    // ---- Item 4: bench-real fail-open policy rejection matrix ----

    /// A minimal config that passes every `bench-real` policy gate:
    /// real transformer enabled, CPU offload, a weights dir, strict
    /// weights, and every fail-open flag disabled.
    fn bench_real_ok_cfg() -> crate::config::Config {
        use std::path::PathBuf;
        crate::config::Config {
            server: crate::config::ServerConfig {
                bind: "127.0.0.1:0".into(),
                max_tokens: 32,
                session_ttl_secs: 0,
                max_concurrent_requests: 0,
                admission_min_free_blocks: 0,
            },
            performance: crate::config::PerformanceConfig::default(),
            model: crate::config::ModelConfig {
                data_dir: PathBuf::from("./data"),
                num_experts: 8,
                top_k: 2,
                d_model: 8,
                d_ff: 16,
                expert_size: 4096,
                num_layers: 1,
                dtype: crate::inference::WeightDtype::F32,
            },
            storage: crate::config::StorageConfigToml {
                cache_slots: 4,
                block_align: 4096,
                no_direct: true,
                predict_fanout: 2,
                pipeline_depth: crate::engine::DEFAULT_PIPELINE_DEPTH,
                predict_min_prob: 0.05,
                partial_load_fraction: 1.0,
                pin_after_observations: 0,
                packed_blob: None,
                packed_manifest: None,
            },
            tokenizer: crate::config::TokenizerConfig::default(),
            real_transformer: {
                let mut rt = crate::config::RealTransformerConfig::default();
                rt.enabled = true;
                rt.weights_dir = Some(PathBuf::from("./weights"));
                rt
            },
            sampling: crate::config::SamplingConfig::default(),
            predictive: crate::config::PredictiveConfig::default(),
            security: crate::config::SecurityConfig::default(),
            gpu_cache: crate::config::GpuCacheConfig::default(),
            distributed: crate::config::DistributedConfig::default(),
        }
    }

    /// The baseline (strict) config passes the gate; the serde
    /// defaults keep every fail-open flag disabled.
    #[test]
    fn bench_real_accepts_strict_baseline_and_flags_default_false() {
        let cfg = bench_real_ok_cfg();
        assert!(!cfg.real_transformer.allow_degraded_experts);
        assert!(!cfg.real_transformer.allow_nonfinite_attention_fallback);
        assert!(!cfg.real_transformer.allow_truncated_expert_payloads);
        assert_eq!(
            cfg.real_transformer.inference_policy(),
            crate::inference::RealInferencePolicy::STRICT
        );
        validate_bench_real_policies(&cfg).expect("strict baseline must pass");
    }

    /// Each fail-open flag is rejected independently, with an error
    /// that names the offending flag.
    #[test]
    fn bench_real_rejects_each_fail_open_flag_independently() {
        let cases: [(&str, fn(&mut crate::config::Config)); 3] = [
            ("allow_degraded_experts", |c| {
                c.real_transformer.allow_degraded_experts = true
            }),
            ("allow_nonfinite_attention_fallback", |c| {
                c.real_transformer.allow_nonfinite_attention_fallback = true
            }),
            ("allow_truncated_expert_payloads", |c| {
                c.real_transformer.allow_truncated_expert_payloads = true
            }),
        ];
        for (name, set) in cases {
            let mut cfg = bench_real_ok_cfg();
            set(&mut cfg);
            let err = validate_bench_real_policies(&cfg)
                .expect_err("bench-real must reject a fail-open policy");
            assert!(
                err.contains(name),
                "rejection for {name} must name the flag; got: {err}"
            );
        }
    }

    /// Combinations of fail-open flags are rejected too (the gate
    /// fails on the first offending flag; no combination slips
    /// through).
    #[test]
    fn bench_real_rejects_fail_open_flag_combinations() {
        for mask in 1u8..8 {
            let mut cfg = bench_real_ok_cfg();
            cfg.real_transformer.allow_degraded_experts = mask & 1 != 0;
            cfg.real_transformer.allow_nonfinite_attention_fallback = mask & 2 != 0;
            cfg.real_transformer.allow_truncated_expert_payloads = mask & 4 != 0;
            assert!(
                validate_bench_real_policies(&cfg).is_err(),
                "flag combination {mask:03b} must be rejected"
            );
        }
    }

    /// The pre-existing gates still hold alongside the policy flags.
    #[test]
    fn bench_real_rejects_seeded_fallback_non_strict_and_non_cpu() {
        let mut cfg = bench_real_ok_cfg();
        cfg.real_transformer.allow_seeded_fallback = true;
        assert!(validate_bench_real_policies(&cfg)
            .expect_err("seeded fallback rejected")
            .contains("allow_seeded_fallback"));

        let mut cfg = bench_real_ok_cfg();
        cfg.real_transformer.strict_weights = false;
        assert!(validate_bench_real_policies(&cfg)
            .expect_err("non-strict weights rejected")
            .contains("strict_weights"));

        for offload in [
            crate::backend::ComputeOffload::Gpu,
            crate::backend::ComputeOffload::Auto,
            crate::backend::ComputeOffload::Hybrid,
        ] {
            let mut cfg = bench_real_ok_cfg();
            cfg.real_transformer.compute_offload = offload;
            assert!(
                validate_bench_real_policies(&cfg).is_err(),
                "{offload:?} must be rejected"
            );
        }
    }

    #[test]
    fn bench_real_validity_detects_softmax_fallback_delta() {
        // Snapshot then force a fallback: the current count only grows, so a
        // check against the pre-snapshot baseline must report the run invalid.
        // Share the transformer softmax-fallback test lock so this mutation of
        // the process-wide counter cannot race with the transformer tests that
        // assert exact before/after deltas.
        let _g = crate::transformer::SOFTMAX_FALLBACK_TEST_LOCK
            .lock()
            .unwrap();
        let snapshot = crate::transformer::nonfinite_softmax_fallbacks();
        crate::transformer::record_nonfinite_softmax_fallback();
        assert!(
            assert_no_softmax_fallbacks(snapshot).is_err(),
            "a nonzero softmax-fallback delta must invalidate the benchmark"
        );
    }

    // ---- Finding 4: real serving requires a real tokenizer ----

    #[test]
    fn real_serving_without_tokenizer_path_fails() {
        match resolve_serving_tokenizer(true, None, Some(1000)) {
            Err(crate::config::ConfigError::Invalid(_)) => {}
            _ => panic!("real serving must reject a missing tokenizer path"),
        }
    }

    #[test]
    fn real_serving_with_unloadable_tokenizer_fails() {
        let path = std::path::PathBuf::from("/nonexistent/does-not-exist/tokenizer.json");
        match resolve_serving_tokenizer(true, Some(&path), Some(1000)) {
            Err(crate::config::ConfigError::Invalid(_)) => {}
            _ => panic!("real serving must reject an unloadable tokenizer"),
        }
    }

    #[test]
    fn synthetic_serving_without_tokenizer_uses_byte_fallback() {
        let tok = match resolve_serving_tokenizer(false, None, None) {
            Ok(t) => t,
            Err(_) => panic!("synthetic mode must keep the byte-tokenizer fallback"),
        };
        assert_eq!(
            tok.vocab_size(),
            256,
            "byte tokenizer expected in synthetic mode"
        );
    }

    #[test]
    fn parses_basic_json_numbers() {
        let body = r#"{ "num_experts": 8, "d_model": 4096, "d_ff": 14336, "top_k": 2, "expert_size": 92274688 }"#;
        assert_eq!(parse_json_number(body, "num_experts"), Some(8));
        assert_eq!(parse_json_number(body, "d_model"), Some(4096));
        assert_eq!(parse_json_number(body, "d_ff"), Some(14336));
        assert_eq!(parse_json_number(body, "top_k"), Some(2));
        assert_eq!(parse_json_number(body, "expert_size"), Some(92274688));
        assert_eq!(parse_json_number(body, "missing"), None);
    }

    #[test]
    fn parses_pretty_printed_json() {
        let body = "{\n  \"num_experts\" : 16,\n  \"d_model\" : 512\n}";
        assert_eq!(parse_json_number(body, "num_experts"), Some(16));
        assert_eq!(parse_json_number(body, "d_model"), Some(512));
    }

    #[test]
    fn parses_gate_layer_index_only_for_exact_pattern() {
        assert_eq!(parse_gate_layer_index("gate_0.bin"), Some(0));
        assert_eq!(parse_gate_layer_index("gate_31.bin"), Some(31));
        // Anything that isn't exactly `gate_<digits>.bin` is ignored.
        assert_eq!(parse_gate_layer_index("gate_.bin"), None);
        assert_eq!(parse_gate_layer_index("gate_1x.bin"), None);
        assert_eq!(parse_gate_layer_index("gate_1.bin.bak"), None);
        assert_eq!(parse_gate_layer_index("rms_moe_1.bin"), None);
        assert_eq!(parse_gate_layer_index("gate.bin"), None);
    }

    #[test]
    fn parse_order_file_strips_inline_comments() {
        let path = tempdir_unique("order-inline-comments.txt");
        std::fs::write(&path, "# full-line comment\n12  # hot expert\n3,4\n8 9\n").unwrap();
        let ids = parse_order_file(&path).unwrap();
        assert_eq!(ids, vec![12, 3, 4, 8, 9]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn repack_order_validation_rejects_duplicate_ids() {
        let err = validate_order(&[0, 1, 1], 4).unwrap_err();
        assert!(
            err.contains("duplicate expert id 1"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn repack_order_validation_rejects_out_of_range_ids() {
        let err = validate_order(&[0, 4], 4).unwrap_err();
        assert!(err.contains("out of range"), "unexpected error: {err}");
        assert!(err.contains("4"), "unexpected error: {err}");
    }

    #[test]
    fn repack_order_validation_allows_subsets() {
        assert!(validate_order(&[0, 2], 4).is_ok());
    }

    #[test]
    fn load_gate_weights_concatenates_directory_in_layer_order() {
        // Global num_experts=2 spread over 2 layers (1 expert/layer) with
        // d_model=2, so each per-layer file holds [1 expert × 2 d_model]
        // = 2 f32s, and the concatenation is 2 × 2 = 4 f32s = the expected
        // [num_experts × d_model] matrix. Written out of order on disk to
        // prove discovery sorts by layer index.
        let dir = tempdir_unique("gate-dir");
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, vals: &[f32]| {
            let mut bytes = Vec::new();
            for v in vals {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(dir.join(name), bytes).unwrap();
        };
        // Intentionally write layer 1 before layer 0 and add a decoy.
        write("gate_1.bin", &[3.0, 4.0]);
        write("gate_0.bin", &[1.0, 2.0]);
        write("notes.txt", &[]);

        let gate = load_gate_weights(
            &dir, /*num_experts=*/ 2, /*d_model=*/ 2, /*top_k=*/ 1,
        )
        .expect("directory gate load should succeed");
        // Concatenation must be layer 0 then layer 1.
        assert_eq!(gate.weights, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(gate.num_experts, 2);
        assert_eq!(gate.d_model, 2);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn load_gate_weights_errors_on_empty_directory() {
        let dir = tempdir_unique("gate-empty");
        std::fs::create_dir_all(&dir).unwrap();
        let err = load_gate_weights(&dir, 2, 2, 1).unwrap_err();
        assert!(
            err.to_string().contains("no gate_<layer>.bin files"),
            "unexpected error: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bench_real_forward_count_matches_prompt_decode_contract() {
        assert_eq!(bench_real_expected_forward_evaluations(15, 16), 30);
        assert_eq!(bench_real_expected_forward_evaluations(1, 1), 1);
        assert_eq!(bench_real_expected_forward_evaluations(1, 4), 4);
        assert_eq!(bench_real_expected_forward_evaluations(4, 1), 4);
        assert_eq!(bench_real_expected_forward_evaluations(0, 4), 0);
        assert_eq!(bench_real_expected_forward_evaluations(4, 0), 0);
    }

    #[test]
    fn bench_real_percentile_reads_sorted_microseconds() {
        let values = vec![100, 200, 300, 400, 500];
        assert_eq!(percentile_us(&values, 0.0), 100);
        assert_eq!(percentile_us(&values, 0.50), 300);
        assert_eq!(percentile_us(&values, 0.95), 500);
        assert_eq!(percentile_us(&values, 1.0), 500);
        assert_eq!(percentile_us_to_ms(&values, 0.50), 0.3);
    }

    #[test]
    fn bench_real_request_json_supports_chat_messages_and_max_tokens() {
        let path = tempdir_unique("bench-real-request.json");
        std::fs::write(
            &path,
            r#"{
                "messages": [
                    { "role": "system", "content": "Be brief." },
                    { "role": "user", "content": "Explain caches." }
                ],
                "max_tokens": 7
            }"#,
        )
        .unwrap();
        let args = BenchRealArgs {
            config: PathBuf::from("config.toml"),
            prompt: None,
            request_json: Some(path.clone()),
            output_tokens: None,
            warmup_runs: 0,
            measured_runs: 1,
            cache_reset: BenchRealCacheReset::Keep,
            greedy: true,
            format: BenchRealOutputFormat::Json,
            progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig::disabled(),
        };
        let input = load_bench_real_input(&args).unwrap();
        assert_eq!(input.output_tokens, 7);
        assert!(input.prompt.contains("system: Be brief."));
        assert!(input.prompt.contains("user: Explain caches."));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bench_real_cli_output_tokens_override_request_json() {
        let path = tempdir_unique("bench-real-request-override.json");
        std::fs::write(&path, r#"{ "prompt": "hello", "max_tokens": 7 }"#).unwrap();
        let args = BenchRealArgs {
            config: PathBuf::from("config.toml"),
            prompt: None,
            request_json: Some(path.clone()),
            output_tokens: Some(3),
            warmup_runs: 0,
            measured_runs: 1,
            cache_reset: BenchRealCacheReset::Keep,
            greedy: true,
            format: BenchRealOutputFormat::Json,
            progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig::disabled(),
        };
        let input = load_bench_real_input(&args).unwrap();
        assert_eq!(input.prompt, "hello");
        assert_eq!(input.output_tokens, 3);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn bench_real_cli_defaults_remain_unchanged() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "bench-real",
            "--config",
            "config.toml",
            "--prompt",
            "hello",
        ])
        .unwrap();
        let Cmd::BenchReal {
            warmup_runs,
            measured_runs,
            cache_reset,
            greedy,
            format,
            ..
        } = cli.cmd
        else {
            panic!("expected bench-real command");
        };
        assert_eq!(warmup_runs, 1);
        assert_eq!(measured_runs, 1);
        assert_eq!(cache_reset, BenchRealCacheReset::Keep);
        assert!(!greedy);
        assert_eq!(format, BenchRealOutputFormat::Human);
    }

    #[test]
    fn qualification_cli_has_strict_defaults_and_external_evidence_slot() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "qualify-hybrid-q4",
            "--config",
            "config.toml",
            "--prompt",
            "hello",
            "--external-gpu-memory-artifact",
            "pr6://sample",
        ])
        .unwrap();
        let Cmd::QualifyHybridQ4 {
            warmup_runs,
            output_tokens,
            report_out,
            external_gpu_memory_artifact,
            ..
        } = cli.cmd
        else {
            panic!("expected qualify-hybrid-q4 command");
        };
        assert_eq!(warmup_runs, 0);
        assert_eq!(output_tokens, None);
        assert_eq!(report_out, None);
        assert_eq!(
            external_gpu_memory_artifact.as_deref(),
            Some("pr6://sample")
        );
    }

    #[test]
    fn qualification_cli_rejects_both_request_inputs() {
        let error = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "qualify-hybrid-q4",
            "--config",
            "config.toml",
            "--prompt",
            "hello",
            "--request-json",
            "request.json",
        ])
        .unwrap_err();
        assert!(error.to_string().contains("cannot be used with"));
    }

    #[test]
    fn q4_parity_cli_requires_and_preserves_global_expert_and_adapter() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "qualify-hybrid-q4-parity",
            "--config",
            "config.toml",
            "--expert-id",
            "257",
            "--expected-adapter-name",
            "NVIDIA L4",
        ])
        .unwrap();
        let Cmd::QualifyHybridQ4Parity {
            expert_id,
            expected_adapter_name,
            report_out,
            ..
        } = cli.cmd
        else {
            panic!("expected qualify-hybrid-q4-parity command");
        };
        assert_eq!(expert_id, 257);
        assert_eq!(expected_adapter_name, "NVIDIA L4");
        assert_eq!(report_out, None);
    }

    #[test]
    fn greedy_parity_cli_requires_one_config_and_exact_adapter() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "qualify-hybrid-q4-greedy-parity",
            "--config",
            "strict-hybrid.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--report-out",
            "report.json",
        ])
        .unwrap();
        let Cmd::QualifyHybridQ4GreedyParity {
            config,
            expected_adapter_name,
            report_out,
        } = cli.cmd
        else {
            panic!("expected qualify-hybrid-q4-greedy-parity command");
        };
        assert_eq!(config, PathBuf::from("strict-hybrid.toml"));
        assert_eq!(expected_adapter_name, "NVIDIA L4");
        assert_eq!(report_out, Some(PathBuf::from("report.json")));

        let error = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "qualify-hybrid-q4-greedy-parity",
            "--config",
            "strict-hybrid.toml",
            "--cpu-config",
            "cpu.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
        ])
        .unwrap_err();
        assert!(error.to_string().contains("--cpu-config"));
    }

    #[test]
    fn gpu_native_greedy_parity_cli_has_the_strict_minimum_surface() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "qualify-gpu-native-q4-greedy-parity",
            "--config",
            "strict-gpu-native.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--report-out",
            "gpu-native-report.json",
        ])
        .unwrap();
        let Cmd::QualifyGpuNativeQ4GreedyParity {
            config,
            expected_adapter_name,
            report_out,
        } = cli.cmd
        else {
            panic!("expected qualify-gpu-native-q4-greedy-parity command");
        };
        assert_eq!(config, PathBuf::from("strict-gpu-native.toml"));
        assert_eq!(expected_adapter_name, "NVIDIA L4");
        assert_eq!(report_out, Some(PathBuf::from("gpu-native-report.json")));
    }

    #[test]
    fn gpu_native_router_rank_diagnostic_cli_has_explicit_target_surface() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "diagnose-gpu-native-router-rank-divergence",
            "--config",
            "strict-gpu-native.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--case",
            "rust-generation",
            "--generated-position",
            "5",
            "--layer",
            "44",
            "--report-out",
            "router-rank.json",
        ])
        .unwrap();
        let Cmd::DiagnoseGpuNativeRouterRankDivergence {
            config,
            expected_adapter_name,
            case,
            generated_position,
            layer,
            report_out,
        } = cli.cmd
        else {
            panic!("expected diagnose-gpu-native-router-rank-divergence command");
        };
        assert_eq!(config, PathBuf::from("strict-gpu-native.toml"));
        assert_eq!(expected_adapter_name, "NVIDIA L4");
        assert_eq!(case, "rust-generation");
        assert_eq!(generated_position, 5);
        assert_eq!(layer, 44);
        assert_eq!(report_out, PathBuf::from("router-rank.json"));
    }

    #[test]
    fn expert_permutation_semantic_diagnostic_cli_has_required_target_surface() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "diagnose-gpu-native-expert-permutation-semantic-parity",
            "--config",
            "strict-gpu-native.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--case",
            "rust-generation",
            "--generated-position",
            "5",
            "--layer",
            "44",
            "--report-out",
            "semantic-witness.json",
            "--progress-timeout-secs",
            "300",
        ])
        .unwrap();
        assert_eq!(cli.progress_timeout_secs, Some(300));
        let Cmd::DiagnoseGpuNativeExpertPermutationSemanticParity {
            config,
            expected_adapter_name,
            case,
            generated_position,
            layer,
            report_out,
        } = &cli.cmd
        else {
            panic!("expected diagnose-gpu-native-expert-permutation-semantic-parity command");
        };
        assert_eq!(config, &PathBuf::from("strict-gpu-native.toml"));
        assert_eq!(expected_adapter_name, "NVIDIA L4");
        assert_eq!(case, "rust-generation");
        assert_eq!(*generated_position, 5);
        assert_eq!(*layer, 44);
        assert_eq!(report_out, &PathBuf::from("semantic-witness.json"));
        assert_eq!(
            super::startup_config_path(&cli.cmd),
            Some(Path::new("strict-gpu-native.toml"))
        );
    }

    #[test]
    fn semantic_parity_corpus_diagnostic_cli_has_frozen_corpus_surface() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "diagnose-gpu-native-semantic-parity-corpus",
            "--config",
            "strict-gpu-native.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--report-out",
            "semantic-corpus.json",
            "--progress-timeout-secs",
            "300",
        ])
        .unwrap();
        assert_eq!(cli.progress_timeout_secs, Some(300));
        let Cmd::DiagnoseGpuNativeSemanticParityCorpus {
            config,
            expected_adapter_name,
            report_out,
        } = &cli.cmd
        else {
            panic!("expected diagnose-gpu-native-semantic-parity-corpus command");
        };
        assert_eq!(config, &PathBuf::from("strict-gpu-native.toml"));
        assert_eq!(expected_adapter_name, "NVIDIA L4");
        assert_eq!(report_out, &PathBuf::from("semantic-corpus.json"));
        assert_eq!(
            super::startup_config_path(&cli.cmd),
            Some(Path::new("strict-gpu-native.toml"))
        );
    }

    #[test]
    fn semantic_parity_v2_qualifier_cli_has_versioned_strict_surface() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "qualify-gpu-native-semantic-parity-v2",
            "--config",
            "strict-gpu-native.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--report-out",
            "semantic-parity-v2.json",
            "--progress-timeout-secs",
            "300",
        ])
        .unwrap();
        assert_eq!(cli.progress_timeout_secs, Some(300));
        let Cmd::QualifyGpuNativeSemanticParityV2 {
            config,
            expected_adapter_name,
            report_out,
        } = &cli.cmd
        else {
            panic!("expected qualify-gpu-native-semantic-parity-v2 command");
        };
        assert_eq!(config, &PathBuf::from("strict-gpu-native.toml"));
        assert_eq!(expected_adapter_name, "NVIDIA L4");
        assert_eq!(report_out, &PathBuf::from("semantic-parity-v2.json"));
        assert_eq!(
            super::startup_config_path(&cli.cmd),
            Some(Path::new("strict-gpu-native.toml"))
        );
    }

    #[test]
    fn v2_holdout_failure_attribution_cli_has_frozen_input_surface() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "diagnose-gpu-native-v2-holdout-failures",
            "--config",
            "strict-gpu-native.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--v2-report",
            "gpu-native-semantic-parity-v2-300448ac.json",
            "--expected-v2-report-sha256",
            crate::gpu_native_v2_holdout_failure_attribution::FROZEN_V2_REPORT_SHA256,
            "--report-out",
            "v2-holdout-attribution.json",
            "--progress-timeout-secs",
            "300",
        ])
        .unwrap();
        assert_eq!(cli.progress_timeout_secs, Some(300));
        let Cmd::DiagnoseGpuNativeV2HoldoutFailures {
            config,
            expected_adapter_name,
            v2_report,
            expected_v2_report_sha256,
            report_out,
        } = &cli.cmd
        else {
            panic!("expected diagnose-gpu-native-v2-holdout-failures command");
        };
        assert_eq!(config, &PathBuf::from("strict-gpu-native.toml"));
        assert_eq!(expected_adapter_name, "NVIDIA L4");
        assert_eq!(
            v2_report,
            &PathBuf::from("gpu-native-semantic-parity-v2-300448ac.json")
        );
        assert_eq!(
            expected_v2_report_sha256,
            crate::gpu_native_v2_holdout_failure_attribution::FROZEN_V2_REPORT_SHA256
        );
        assert_eq!(report_out, &PathBuf::from("v2-holdout-attribution.json"));
        assert_eq!(
            super::startup_config_path(&cli.cmd),
            Some(Path::new("strict-gpu-native.toml"))
        );
    }

    #[test]
    fn q4_expert_stage_attribution_cli_has_versioned_frozen_surface() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "diagnose-gpu-native-q4-expert-stages",
            "--config",
            "strict-gpu-native.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--bea-report",
            "bea-attribution.json",
            "--expected-bea-report-sha256",
            crate::gpu_native_q4_expert_stage_attribution::FROZEN_BEA_REPORT_SHA256,
            "--report-out",
            "q4-expert-stages.json",
            "--progress-timeout-secs",
            "300",
        ])
        .unwrap();
        assert_eq!(cli.progress_timeout_secs, Some(300));
        let Cmd::DiagnoseGpuNativeQ4ExpertStages {
            config,
            expected_adapter_name,
            bea_report,
            expected_bea_report_sha256,
            report_out,
        } = &cli.cmd
        else {
            panic!("expected diagnose-gpu-native-q4-expert-stages command");
        };
        assert_eq!(config, &PathBuf::from("strict-gpu-native.toml"));
        assert_eq!(expected_adapter_name, "NVIDIA L4");
        assert_eq!(bea_report, &PathBuf::from("bea-attribution.json"));
        assert_eq!(
            expected_bea_report_sha256,
            crate::gpu_native_q4_expert_stage_attribution::FROZEN_BEA_REPORT_SHA256
        );
        assert_eq!(report_out, &PathBuf::from("q4-expert-stages.json"));
        assert_eq!(
            crate::gpu_native_q4_expert_stage_attribution::SCHEMA_VERSION,
            "mer.gpu-native-q4-expert-stage-attribution.v1"
        );
        assert_eq!(
            super::startup_config_path(&cli.cmd),
            Some(Path::new("strict-gpu-native.toml"))
        );
    }

    #[test]
    fn f32_reference_boundary_audit_cli_has_versioned_cpu_only_frozen_surface() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "diagnose-gpu-native-f32-reference-boundary",
            "--config",
            "strict-gpu-native.toml",
            "--v2-report",
            "gpu-native-semantic-parity-v2-300448ac.json",
            "--expected-v2-report-sha256",
            crate::gpu_native_f32_reference_boundary_audit::FROZEN_V2_REPORT_SHA256,
            "--stage-report",
            "gpu-native-q4-expert-stage-attribution-987e7e4b.json",
            "--expected-stage-report-sha256",
            crate::gpu_native_f32_reference_boundary_audit::FROZEN_STAGE_REPORT_SHA256,
            "--report-out",
            "f32-reference-boundary-audit.json",
            "--progress-timeout-secs",
            "300",
        ])
        .unwrap();
        assert_eq!(cli.progress_timeout_secs, Some(300));
        let Cmd::DiagnoseGpuNativeF32ReferenceBoundary {
            config,
            v2_report,
            expected_v2_report_sha256,
            stage_report,
            expected_stage_report_sha256,
            report_out,
        } = &cli.cmd
        else {
            panic!("expected diagnose-gpu-native-f32-reference-boundary command");
        };
        assert_eq!(config, &PathBuf::from("strict-gpu-native.toml"));
        assert_eq!(
            v2_report,
            &PathBuf::from("gpu-native-semantic-parity-v2-300448ac.json")
        );
        assert_eq!(
            expected_v2_report_sha256,
            crate::gpu_native_f32_reference_boundary_audit::FROZEN_V2_REPORT_SHA256
        );
        assert_eq!(
            stage_report,
            &PathBuf::from("gpu-native-q4-expert-stage-attribution-987e7e4b.json")
        );
        assert_eq!(
            expected_stage_report_sha256,
            crate::gpu_native_f32_reference_boundary_audit::FROZEN_STAGE_REPORT_SHA256
        );
        assert_eq!(
            report_out,
            &PathBuf::from("f32-reference-boundary-audit.json")
        );
        assert_eq!(
            crate::gpu_native_f32_reference_boundary_audit::SCHEMA_VERSION,
            "mer.gpu-native-f32-reference-boundary-audit.v1"
        );
        assert_eq!(
            super::startup_config_path(&cli.cmd),
            Some(Path::new("strict-gpu-native.toml"))
        );
    }

    #[test]
    fn spanish_first_token_attribution_cli_has_three_frozen_reports_and_l4_surface() {
        let cli = Cli::try_parse_from([
            "micro-expert-router",
            "--progress-timeout-secs",
            "300",
            "diagnose-gpu-native-spanish-first-token-attribution",
            "--config",
            "config.toml",
            "--v2-report",
            "v2.json",
            "--expected-v2-report-sha256",
            crate::gpu_native_spanish_first_token_attribution::FROZEN_V2_REPORT_SHA256,
            "--stage-report",
            "stage.json",
            "--expected-stage-report-sha256",
            crate::gpu_native_spanish_first_token_attribution::FROZEN_STAGE_REPORT_SHA256,
            "--boundary-audit-report",
            "boundary.json",
            "--expected-boundary-audit-report-sha256",
            crate::gpu_native_spanish_first_token_attribution::FROZEN_BOUNDARY_AUDIT_REPORT_SHA256,
            "--expected-adapter-name",
            "NVIDIA L4",
            "--report-out",
            "spanish-first-token.json",
        ])
        .unwrap();
        assert_eq!(cli.progress_timeout_secs, Some(300));
        let Cmd::DiagnoseGpuNativeSpanishFirstTokenAttribution {
            config,
            v2_report,
            expected_v2_report_sha256,
            stage_report,
            expected_stage_report_sha256,
            boundary_audit_report,
            expected_boundary_audit_report_sha256,
            expected_adapter_name,
            report_out,
        } = &cli.cmd
        else {
            panic!("expected Spanish first-token attribution command");
        };
        assert_eq!(config, &PathBuf::from("config.toml"));
        assert_eq!(v2_report, &PathBuf::from("v2.json"));
        assert_eq!(
            expected_v2_report_sha256,
            crate::gpu_native_spanish_first_token_attribution::FROZEN_V2_REPORT_SHA256
        );
        assert_eq!(stage_report, &PathBuf::from("stage.json"));
        assert_eq!(
            expected_stage_report_sha256,
            crate::gpu_native_spanish_first_token_attribution::FROZEN_STAGE_REPORT_SHA256
        );
        assert_eq!(boundary_audit_report, &PathBuf::from("boundary.json"));
        assert_eq!(
            expected_boundary_audit_report_sha256,
            crate::gpu_native_spanish_first_token_attribution::FROZEN_BOUNDARY_AUDIT_REPORT_SHA256
        );
        assert_eq!(expected_adapter_name, "NVIDIA L4");
        assert_eq!(report_out, &PathBuf::from("spanish-first-token.json"));
        assert_eq!(
            super::startup_config_path(&cli.cmd),
            Some(Path::new("config.toml"))
        );
    }

    #[test]
    fn logit_diagnostic_cli_requires_report_and_exact_adapter() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "diagnose-hybrid-q4-greedy-divergence",
            "--config",
            "strict-hybrid.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--report-out",
            "diagnostic.json",
        ])
        .unwrap();
        let Cmd::DiagnoseHybridQ4GreedyDivergence {
            config,
            expected_adapter_name,
            report_out,
        } = cli.cmd
        else {
            panic!("expected logit diagnostic command");
        };
        assert_eq!(config, PathBuf::from("strict-hybrid.toml"));
        assert_eq!(expected_adapter_name, "NVIDIA L4");
        assert_eq!(report_out, PathBuf::from("diagnostic.json"));

        let missing_report = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "diagnose-hybrid-q4-greedy-divergence",
            "--config",
            "strict-hybrid.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
        ]);
        assert!(missing_report.is_err());
    }

    #[test]
    fn greedy_parity_failure_report_is_written_and_returns_error() {
        let dir = tempdir_unique("greedy-parity-failure");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.json");
        let report = crate::greedy_parity::GreedyParityReport::new(
            crate::qualification::BuildProvenance {
                git_sha: Some("0".repeat(40)),
                dirty: Some(false),
                package_version: "test".to_string(),
            },
            crate::qualification::QualificationArtifacts::default(),
            None,
            "NVIDIA L4".to_string(),
        );
        let failure = crate::qualification::QualificationFailure::new(
            crate::qualification::FailureStage::Postcondition,
            "test-failure",
            "typed failure must produce a nonzero command result",
        );
        assert!(fail_greedy_parity(report, failure, Some(&path)).is_err());
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(json["status"], "fail");
        assert_eq!(json["failure"]["code"], "test-failure");
        let _ = std::fs::remove_dir_all(dir);
    }

    fn gpu_native_greedy_parity_test_report(
    ) -> crate::gpu_native_greedy_parity::GpuNativeGreedyParityReport {
        crate::gpu_native_greedy_parity::GpuNativeGreedyParityReport::new(
            crate::qualification::BuildProvenance {
                git_sha: Some("0".repeat(40)),
                dirty: Some(false),
                package_version: "test".to_string(),
            },
            crate::qualification::QualificationArtifacts::default(),
            None,
            "NVIDIA L4".to_string(),
            crate::gpu_native_greedy_parity::SourceConfigEvidence {
                real_transformer_enabled: true,
                gpu_native_enabled: true,
                weights_dir_configured: true,
                strict_weights: true,
                allow_seeded_fallback: false,
                allow_degraded_experts: false,
                allow_nonfinite_attention_fallback: false,
                allow_truncated_expert_payloads: false,
                distributed_enabled: false,
                gpu_cache_enabled: true,
                gpu_expert_capacity_bytes: 1,
                routed_expert_dtype: "q4_0".to_string(),
            },
        )
    }

    #[test]
    fn gpu_native_isolated_shutdown_failure_report_is_durable_and_nonzero() {
        let dir = tempdir_unique("gpu-native-shutdown-failure");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("report.json");
        let teardown_error = IsolatedRuntimeShutdownError::new(
            "isolated runtime background resources remained live after 30s: engine=1 storage=1",
        );
        let failure = gpu_native_case_execution_failure(
            "gpu-native-candidate-failed",
            "test-case",
            &teardown_error,
        );
        let error = fail_gpu_native_greedy_parity(
            gpu_native_greedy_parity_test_report(),
            failure,
            Some(&path),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("isolated-runtime-shutdown-failed"));
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(json["qualification_pass"], false);
        assert_eq!(json["failure"]["code"], "isolated-runtime-shutdown-failed");
        assert!(json["failure"]["detail"]
            .as_str()
            .unwrap()
            .contains("engine=1 storage=1"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn gpu_native_report_write_failure_remains_nonzero_and_explicit() {
        let dir = tempdir_unique("gpu-native-report-write-failure");
        std::fs::create_dir_all(&dir).unwrap();
        let failure = crate::qualification::QualificationFailure::new(
            crate::qualification::FailureStage::Postcondition,
            "isolated-runtime-shutdown-failed",
            "forced teardown failure",
        );
        let error = fail_gpu_native_greedy_parity(
            gpu_native_greedy_parity_test_report(),
            failure,
            Some(&dir),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("report emission failed"), "{error}");
        assert!(
            error.contains("isolated-runtime-shutdown-failed"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn greedy_parity_isolated_contexts_are_distinct_and_cpu_exact() {
        let global_before = crate::backend::current_execution_context();
        let mut cfg = bench_real_ok_cfg();
        cfg.model.d_model = 32;
        cfg.model.d_ff = 32;
        cfg.model.dtype = crate::inference::WeightDtype::Q4_0;
        cfg.real_transformer.num_heads = 1;
        cfg.real_transformer.num_kv_heads = 1;
        cfg.real_transformer.head_dim = 32;
        let spec = ResolvedRealCliSpec {
            cfg,
            architecture: crate::architecture::Architecture::Qwen3Moe,
            first_k_dense_replace: 0,
            advanced: crate::model::AdvancedConfig::default(),
        };
        let first = resolve_isolated_real_cli_context(
            &spec,
            crate::backend::ComputeOffload::Cpu,
        )
        .unwrap();
        let second = resolve_isolated_real_cli_context(
            &spec,
            crate::backend::ComputeOffload::Cpu,
        )
        .unwrap();
        assert_ne!(first.id(), second.id());
        assert!(!Arc::ptr_eq(
            first.gpu_expert_cache(),
            second.gpu_expert_cache()
        ));
        assert!(crate::greedy_parity::cpu_plan_exact(&first.plan().into()));
        assert!(crate::greedy_parity::cpu_plan_exact(&second.plan().into()));
        let global_after = crate::backend::current_execution_context();
        assert!(Arc::ptr_eq(&global_before, &global_after));
        assert_ne!(first.id(), global_after.id());
        assert_ne!(second.id(), global_after.id());
        assert!(RealCliRuntimeMode::IsolatedGreedyParityCpu.is_isolated());
        assert!(RealCliRuntimeMode::IsolatedGreedyParityHybrid.is_isolated());
        assert!(!RealCliRuntimeMode::BenchReal.is_isolated());
    }

    #[tokio::test]
    async fn greedy_parity_shutdown_waits_for_background_reference_release() {
        let owner = Arc::new(());
        let weak = Arc::downgrade(&owner);
        let background = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            drop(owner);
        });
        let evidence = wait_for_isolated_release(
            || IsolatedRuntimeLiveResources {
                engine: weak.strong_count(),
                ..IsolatedRuntimeLiveResources::default()
            },
            std::time::Duration::from_millis(1),
            std::time::Duration::from_secs(1),
        )
        .await
        .unwrap();
        background.await.unwrap();
        assert!(evidence.controlled_shutdown_requested);
        assert!(evidence.all_runtime_resources_released);
        assert!(evidence.poll_iterations > 1);
    }

    #[tokio::test]
    async fn greedy_parity_shutdown_timeout_fails_closed() {
        let retained = Arc::new(());
        let weak = Arc::downgrade(&retained);
        let error = wait_for_isolated_release(
            || IsolatedRuntimeLiveResources {
                storage: weak.strong_count(),
                ..IsolatedRuntimeLiveResources::default()
            },
            std::time::Duration::from_millis(1),
            std::time::Duration::from_millis(5),
        )
        .await
        .unwrap_err();
        assert!(error.contains("background resources remained live"));
        assert!(error.contains("storage=1"), "{error}");
        assert!(error.contains("engine=0"), "{error}");
        assert!(error.contains("gpu_cache=0"), "{error}");
        drop(retained);
    }

    #[test]
    fn greedy_parity_internal_worker_command_is_hidden_and_non_recursive() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "greedy-parity-hybrid-worker-internal",
            "--config",
            "strict-hybrid.toml",
        ])
        .unwrap();
        assert!(matches!(
            cli.cmd,
            Cmd::GreedyParityHybridWorkerInternal { config }
                if config == Path::new("strict-hybrid.toml")
        ));
        let help = <Cli as clap::CommandFactory>::command()
            .render_long_help()
            .to_string();
        assert!(!help.contains("greedy-parity-hybrid-worker-internal"));
        let diagnostic = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "greedy-parity-logit-worker-internal",
            "--config",
            "strict-hybrid.toml",
        ])
        .unwrap();
        assert!(matches!(
            diagnostic.cmd,
            Cmd::GreedyParityLogitWorkerInternal { config }
                if config == Path::new("strict-hybrid.toml")
        ));
        assert!(!help.contains("greedy-parity-logit-worker-internal"));
    }

    #[test]
    fn greedy_parity_worker_stderr_is_bounded_while_input_is_fully_drained() {
        let input = vec![b'x'; crate::greedy_parity::MAX_WORKER_STDERR_BYTES + 4096];
        let bounded = read_child_output_bounded(
            std::io::Cursor::new(input),
            crate::greedy_parity::MAX_WORKER_STDERR_BYTES,
        )
        .unwrap();
        assert_eq!(
            bounded.bytes.len(),
            crate::greedy_parity::MAX_WORKER_STDERR_BYTES
        );
        assert!(bounded.truncated);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn greedy_parity_worker_supervisor_observes_success_and_zero_exit() {
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg("cat >/dev/null; printf '{\"ok\":true}'; printf diagnostic >&2");
        let capture = run_greedy_parity_worker_process(
            command,
            br#"{"request":true}"#.to_vec(),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
        assert_eq!(capture.status.code(), Some(0));
        assert_eq!(child_exit_signal(&capture.status), None);
        assert!(capture.reaped);
        assert!(!capture.timed_out);
        assert_eq!(capture.stdout.bytes, br#"{"ok":true}"#);
        assert_eq!(capture.stderr.bytes, b"diagnostic");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn greedy_parity_worker_supervisor_preserves_nonzero_exit() {
        let mut command = Command::new("sh");
        command.arg("-c").arg("cat >/dev/null; exit 7");
        let capture = run_greedy_parity_worker_process(
            command,
            b"request".to_vec(),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
        assert_eq!(capture.status.code(), Some(7));
        assert!(capture.reaped);
        assert!(!capture.timed_out);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn greedy_parity_worker_supervisor_preserves_signal_termination() {
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg("cat >/dev/null; kill -TERM $$");
        let capture = run_greedy_parity_worker_process(
            command,
            b"request".to_vec(),
            Duration::from_secs(2),
        )
        .await
        .unwrap();
        assert_eq!(capture.status.code(), None);
        assert_eq!(child_exit_signal(&capture.status), Some(15));
        assert!(capture.reaped);
        assert!(!capture.timed_out);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn greedy_parity_worker_timeout_kills_and_reaps_exact_child() {
        let mut command = Command::new("sleep");
        command.arg("2");
        let started = Instant::now();
        let capture = run_greedy_parity_worker_process(
            command,
            b"request".to_vec(),
            Duration::from_millis(20),
        )
        .await
        .unwrap();
        assert!(capture.timed_out);
        assert!(capture.reaped);
        assert!(child_exit_signal(&capture.status).is_some());
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn q4_parity_cli_rejects_missing_explicit_identity_inputs() {
        for missing in ["--expert-id", "--expected-adapter-name"] {
            let mut args = vec![
                "micro-expert-router",
                "qualify-hybrid-q4-parity",
                "--config",
                "config.toml",
                "--expert-id",
                "257",
                "--expected-adapter-name",
                "NVIDIA L4",
            ];
            let index = args.iter().position(|value| *value == missing).unwrap();
            args.drain(index..=index + 1);
            let error = <Cli as clap::Parser>::try_parse_from(args).unwrap_err();
            assert!(error.to_string().contains(missing), "{error}");
        }
    }

    #[test]
    fn q4_parity_command_receives_configured_progress_watchdog() {
        let args = QualifyHybridQ4ParityArgs {
            config: PathBuf::from("config.toml"),
            expert_id: 0,
            expected_adapter_name: "NVIDIA L4".to_string(),
            report_out: None,
            progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig {
                timeout: Some(std::time::Duration::from_secs(17)),
            },
        };
        assert_eq!(
            q4_parity_readback_timeout(&args).unwrap(),
            std::time::Duration::from_secs(17)
        );

        let disabled = QualifyHybridQ4ParityArgs {
            progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig::disabled(),
            ..args
        };
        assert_eq!(
            q4_parity_readback_timeout(&disabled).unwrap_err().code,
            "progress-watchdog-required"
        );
    }

    #[test]
    fn q4_parity_report_emission_error_preserves_primary_failure_first() {
        let report = crate::q4_parity::Q4ParityReport::new(
            crate::qualification::BuildProvenance::embedded(),
            crate::qualification::QualificationArtifacts::default(),
            None,
            "NVIDIA L4".to_string(),
        );
        let failure = crate::qualification::QualificationFailure::new(
            crate::qualification::FailureStage::Postcondition,
            "primary-q4-failure",
            "primary detail",
        );
        let directory = tempdir_unique("q4-parity-report-is-directory");
        std::fs::create_dir_all(&directory).unwrap();
        let error = fail_q4_parity(report, failure, Some(&directory))
            .unwrap_err()
            .to_string();
        assert!(error.starts_with("primary-q4-failure: primary detail;"), "{error}");
        assert!(
            error.contains("additionally failed to emit Q4_0 parity report"),
            "{error}"
        );
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn qualification_request_json_uses_bench_semantics_and_cli_override() {
        let path = tempdir_unique("qualification-request.json");
        std::fs::write(
            &path,
            r#"{ "prompt": "hello", "max_tokens": 7, "temperature": 0.9 }"#,
        )
        .unwrap();
        let args = QualifyHybridQ4Args {
            config: PathBuf::from("config.toml"),
            prompt: None,
            request_json: Some(path.clone()),
            output_tokens: Some(3),
            warmup_runs: 0,
            report_out: None,
            external_gpu_memory_artifact: None,
            progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig::disabled(),
        };
        let input = load_qualification_input(&args).unwrap();
        assert_eq!(input.prompt, "hello");
        assert_eq!(input.output_tokens, 3);
        assert_eq!(input.input_kind, "request-json");
        assert!(crate::qualification::request_evidence(
            input.input_kind,
            &input.prompt,
            input.output_tokens,
            0
        )
        .greedy);
        let _ = std::fs::remove_file(path);
    }

    /// Tiny unique temp-dir helper (avoids pulling a dev-dependency for
    /// these filesystem tests).
    fn tempdir_unique(prefix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()))
    }

    /// Minimal `Config` for exercising [`reconcile_real_model_config`]. The
    /// non-`real_transformer` sections carry placeholder values; the tests
    /// only inspect fields the reconciler overwrites.
    fn minimal_bench_cfg() -> crate::config::Config {
        use crate::config::*;
        Config {
            server: ServerConfig {
                bind: "127.0.0.1:8080".into(),
                max_tokens: 64,
                session_ttl_secs: 0,
                max_concurrent_requests: 0,
                admission_min_free_blocks: 0,
            },
            performance: PerformanceConfig::default(),
            model: ModelConfig {
                data_dir: PathBuf::from("./data"),
                num_experts: 8,
                top_k: 2,
                d_model: 64,
                d_ff: 256,
                expert_size: 4096,
                num_layers: 1,
                dtype: WeightDtype::F32,
            },
            storage: StorageConfigToml {
                cache_slots: 4,
                block_align: 4096,
                no_direct: false,
                predict_fanout: 2,
                pipeline_depth: crate::engine::DEFAULT_PIPELINE_DEPTH,
                predict_min_prob: 0.0,
                partial_load_fraction: 1.0,
                pin_after_observations: 0,
                packed_blob: None,
                packed_manifest: None,
            },
            tokenizer: TokenizerConfig::default(),
            real_transformer: RealTransformerConfig::default(),
            sampling: SamplingConfig::default(),
            predictive: PredictiveConfig::default(),
            security: SecurityConfig::default(),
            gpu_cache: GpuCacheConfig::default(),
            distributed: DistributedConfig::default(),
        }
    }

    #[test]
    fn qualification_metadata_read_failure_has_distinct_typed_code() {
        let failure = qualification_metadata_failure("malformed metadata JSON");
        assert_eq!(
            failure.stage,
            crate::qualification::FailureStage::Preflight
        );
        assert_eq!(failure.code, "expert-metadata-unreadable");
        assert_eq!(failure.detail, "malformed metadata JSON");
    }

    #[test]
    fn configured_missing_artifact_produces_typed_identity_failure() {
        let dir = tempdir_unique("qualification-missing-artifact");
        std::fs::create_dir_all(&dir).unwrap();
        let config_path = dir.join("config.toml");
        std::fs::write(&config_path, b"# qualification fixture\n").unwrap();
        std::fs::write(dir.join("metadata.json"), b"{}").unwrap();

        let missing_tokenizer = dir.join("missing-tokenizer.json");
        let mut cfg = minimal_bench_cfg();
        cfg.model.data_dir = dir.clone();
        cfg.tokenizer.path = Some(missing_tokenizer.clone());
        let (artifacts, errors) = qualification_artifacts(&config_path, &cfg);

        assert!(artifacts.tokenizer.is_none());
        assert_eq!(errors.len(), 1);
        let missing_tokenizer = missing_tokenizer.display().to_string();
        assert!(errors[0].contains(&missing_tokenizer));
        let failure = qualification_artifact_failure(&errors);
        assert_eq!(
            failure.stage,
            crate::qualification::FailureStage::Preflight
        );
        assert_eq!(failure.code, "artifact-identity-unavailable");
        assert!(failure.detail.contains("missing-tokenizer.json"));

        let _ = std::fs::remove_dir_all(dir);
    }

    /// A distinctive tiny Qwen3-MoE `config.json` (checkpoint dims differ from
    /// [`minimal_bench_cfg`]) with `norm_topk_prob = true`.
    fn write_qwen3_moe_config_json(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        let body = serde_json::json!({
            "model_type": "qwen3_moe",
            "hidden_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 100,
            "intermediate_size": 128,
            "moe_intermediate_size": 64,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "rope_theta": 5_000_000.0,
            "rms_norm_eps": 1e-5
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(&body).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn explicit_architecture_still_reconciles_config_json() {
        // D. Explicit TOML architecture must not suppress config.json
        // reconciliation of advanced fields and dimensions.
        let dir = tempdir_unique("bench-real-qwen3moe-cfg");
        write_qwen3_moe_config_json(&dir);
        let mut cfg = minimal_bench_cfg();
        cfg.real_transformer.weights_dir = Some(dir.clone());
        cfg.real_transformer.architecture = Some("qwen3_moe".to_string());

        let (arch, _first_k, advanced) =
            reconcile_real_model_config(&mut cfg).expect("reconcile should succeed");

        assert_eq!(arch, crate::architecture::Architecture::Qwen3Moe);
        assert!(
            advanced.norm_topk_prob,
            "checkpoint norm_topk_prob=true must be reconciled even with explicit architecture"
        );
        // Dimensions come from config.json, not the placeholder TOML.
        assert_eq!(cfg.model.d_model, 32);
        assert_eq!(cfg.model.num_layers, 3);
        assert_eq!(cfg.model.num_experts, 4);
        assert_eq!(cfg.real_transformer.num_heads, 4);
        assert_eq!(cfg.real_transformer.num_kv_heads, 2);
        assert_eq!(cfg.real_transformer.head_dim, 8);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn explicit_architecture_mismatch_is_rejected() {
        // E. A TOML architecture that disagrees with config.json is a hard
        // configuration error naming both architectures.
        let dir = tempdir_unique("bench-real-arch-mismatch");
        write_qwen3_moe_config_json(&dir);
        let mut cfg = minimal_bench_cfg();
        cfg.real_transformer.weights_dir = Some(dir.clone());
        cfg.real_transformer.architecture = Some("mixtral".to_string());

        let err = reconcile_real_model_config(&mut cfg)
            .expect_err("mismatched architecture must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("mixtral"), "error must name TOML arch: {msg}");
        assert!(
            msg.contains("qwen3_moe"),
            "error must name checkpoint arch: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- Finding 7: resolved-config architecture-aware validation ----

    #[test]
    fn resolved_config_validation_accepts_reconciled_checkpoint() {
        let dir = tempdir_unique("bench-real-f7-ok");
        write_qwen3_moe_config_json(&dir);
        let mut cfg = minimal_bench_cfg();
        cfg.real_transformer.weights_dir = Some(dir.clone());
        let (arch, first_k, advanced) =
            reconcile_real_model_config(&mut cfg).expect("reconcile should succeed");
        validate_resolved_real_model_config(&cfg, arch, first_k, &advanced)
            .expect("a well-formed reconciled checkpoint must pass validation");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolved_config_validation_rejects_top_k_over_num_experts() {
        let mut cfg = minimal_bench_cfg();
        cfg.model.num_experts = 4;
        cfg.model.top_k = 8; // impossible: more activated than available
        let advanced = crate::model::AdvancedConfig::default();
        let err = validate_resolved_real_model_config(
            &cfg,
            crate::architecture::Architecture::Mixtral,
            0,
            &advanced,
        )
        .expect_err("top_k > num_experts must fail");
        assert!(err.to_string().contains("top_k"), "{err}");
    }

    #[test]
    fn resolved_config_validation_rejects_gqa_head_indivisibility() {
        let mut cfg = minimal_bench_cfg();
        cfg.real_transformer.num_heads = 6;
        cfg.real_transformer.num_kv_heads = 4; // 6 % 4 != 0
        cfg.real_transformer.head_dim = 8;
        let advanced = crate::model::AdvancedConfig::default();
        let err = validate_resolved_real_model_config(
            &cfg,
            crate::architecture::Architecture::Mixtral,
            0,
            &advanced,
        )
        .expect_err("non-divisible GQA head counts must fail");
        assert!(err.to_string().contains("num_kv_heads"), "{err}");
    }

    #[test]
    fn resolved_config_validation_allows_asymmetric_v_head_dim() {
        // MiMo-V2-Flash style: v_head_dim != head_dim must NOT be rejected,
        // and the invalid universal num_heads*head_dim==d_model rule must not
        // be applied.
        let mut cfg = minimal_bench_cfg();
        cfg.model.d_model = 64;
        cfg.real_transformer.num_heads = 4;
        cfg.real_transformer.num_kv_heads = 2;
        cfg.real_transformer.head_dim = 24; // 4*24 = 96 != d_model 64
        let mut advanced = crate::model::AdvancedConfig::default();
        advanced.v_head_dim = Some(16);
        validate_resolved_real_model_config(
            &cfg,
            crate::architecture::Architecture::MiMoV2,
            0,
            &advanced,
        )
        .expect("asymmetric V geometry must be accepted");
    }

    #[test]
    fn strict_hybrid_geometry_uses_resolved_asymmetric_v_head_dim() {
        let mut cfg = minimal_bench_cfg();
        cfg.real_transformer.num_heads = 4;
        cfg.real_transformer.num_kv_heads = 2;
        cfg.real_transformer.head_dim = 24;
        let advanced = crate::model::AdvancedConfig {
            v_head_dim: Some(16),
            ..Default::default()
        };

        let geometry = strict_hybrid_gpu_geometry(&cfg, &advanced);
        assert_eq!(geometry.head_dim, 24);
        assert_eq!(geometry.v_head_dim, 16);

        let symmetric = crate::model::AdvancedConfig::default();
        assert_eq!(strict_hybrid_gpu_geometry(&cfg, &symmetric).v_head_dim, 0);
    }

    #[test]
    fn resolved_config_validation_rejects_undersized_cache() {
        let mut cfg = minimal_bench_cfg();
        cfg.model.top_k = 4;
        cfg.storage.cache_slots = 2; // smaller than activated per layer
        let advanced = crate::model::AdvancedConfig::default();
        let err = validate_resolved_real_model_config(
            &cfg,
            crate::architecture::Architecture::Mixtral,
            0,
            &advanced,
        )
        .expect_err("cache smaller than the per-layer working set must fail");
        assert!(err.to_string().contains("cache_slots"), "{err}");
    }

    #[test]
    fn resolved_config_validation_rejects_odirect_misaligned_expert_size() {
        let mut cfg = minimal_bench_cfg();
        cfg.storage.no_direct = false;
        cfg.storage.block_align = 4096;
        cfg.model.expert_size = 4097; // not a multiple of block_align
        let advanced = crate::model::AdvancedConfig::default();
        let err = validate_resolved_real_model_config(
            &cfg,
            crate::architecture::Architecture::Mixtral,
            0,
            &advanced,
        )
        .expect_err("O_DIRECT misaligned expert_size must fail");
        assert!(err.to_string().contains("block_align"), "{err}");
    }

    /// Finding 2: serving and bench-real share a single reconcile + validation
    /// path, so both resolve an identical config from the same checkpoint and
    /// reject the same invalid config. Guards against the two call sites
    /// drifting apart (e.g. one reconciling advanced routing fields the other
    /// misses, or the two applying different validation rules).
    #[test]
    fn serve_and_bench_real_share_reconcile_and_validation() {
        let dir = tempdir_unique("f2-shared-reconcile");
        write_qwen3_moe_config_json(&dir);

        // Two independently-constructed configs standing in for the serving and
        // bench-real entry points; both point at the same checkpoint.
        let mut serve_cfg = minimal_bench_cfg();
        serve_cfg.real_transformer.enabled = true;
        serve_cfg.real_transformer.weights_dir = Some(dir.clone());
        let mut bench_cfg = minimal_bench_cfg();
        bench_cfg.real_transformer.enabled = true;
        bench_cfg.real_transformer.weights_dir = Some(dir.clone());

        let (serve_arch, serve_fk, serve_adv) =
            reconcile_real_model_config(&mut serve_cfg).expect("serve reconcile");
        let (bench_arch, bench_fk, bench_adv) =
            reconcile_real_model_config(&mut bench_cfg).expect("bench reconcile");

        // Identical architecture / dims / routing resolution across both paths.
        assert_eq!(serve_arch, bench_arch);
        assert_eq!(serve_fk, bench_fk);
        assert_eq!(serve_adv.norm_topk_prob, bench_adv.norm_topk_prob);
        assert_eq!(serve_cfg.model.d_model, bench_cfg.model.d_model);
        assert_eq!(serve_cfg.model.num_layers, bench_cfg.model.num_layers);
        assert_eq!(serve_cfg.model.num_experts, bench_cfg.model.num_experts);
        assert_eq!(serve_cfg.model.top_k, bench_cfg.model.top_k);
        assert_eq!(
            serve_cfg.real_transformer.num_heads,
            bench_cfg.real_transformer.num_heads
        );
        assert_eq!(
            serve_cfg.real_transformer.num_kv_heads,
            bench_cfg.real_transformer.num_kv_heads
        );
        assert_eq!(
            serve_cfg.real_transformer.head_dim,
            bench_cfg.real_transformer.head_dim
        );
        assert_eq!(
            serve_cfg.real_transformer.vocab_size,
            bench_cfg.real_transformer.vocab_size
        );

        // Both resolved configs pass the shared validation.
        validate_resolved_real_model_config(&serve_cfg, serve_arch, serve_fk, &serve_adv)
            .expect("serve validation");
        validate_resolved_real_model_config(&bench_cfg, bench_arch, bench_fk, &bench_adv)
            .expect("bench validation");

        // An identical invalid mutation is rejected identically by both paths.
        let mut serve_bad = serve_cfg.clone();
        serve_bad.storage.cache_slots = 1;
        serve_bad.model.top_k = serve_bad.model.num_experts as usize;
        let mut bench_bad = bench_cfg.clone();
        bench_bad.storage.cache_slots = 1;
        bench_bad.model.top_k = bench_bad.model.num_experts as usize;
        let serve_err =
            validate_resolved_real_model_config(&serve_bad, serve_arch, serve_fk, &serve_adv)
                .expect_err("serve must reject undersized cache");
        let bench_err =
            validate_resolved_real_model_config(&bench_bad, bench_arch, bench_fk, &bench_adv)
                .expect_err("bench must reject undersized cache");
        assert_eq!(serve_err.to_string(), bench_err.to_string());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gpu_native_first_divergence_cli_parses_expected_flags() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "diagnose-gpu-native-q4-first-divergence",
            "--config",
            "config.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--case",
            "json-transformation",
            "--report-out",
            "report.json",
        ])
        .unwrap();

        match &cli.cmd {
            Cmd::DiagnoseGpuNativeQ4FirstDivergence {
                config,
                expected_adapter_name,
                case,
                report_out,
            } => {
                assert_eq!(config, &PathBuf::from("config.toml"));
                assert_eq!(expected_adapter_name, "NVIDIA L4");
                assert_eq!(case, "json-transformation");
                assert_eq!(report_out, &Some(PathBuf::from("report.json")));
            }
            _ => panic!("unexpected command variant"),
        }
        assert_eq!(
            super::startup_config_path(&cli.cmd),
            Some(Path::new("config.toml"))
        );
    }

    #[test]
    fn startup_config_path_resolves_gpu_native_first_divergence() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "diagnose-gpu-native-q4-first-divergence",
            "--config",
            "config.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--case",
            "json-transformation",
        ])
        .unwrap();

        assert_eq!(
            super::startup_config_path(&cli.cmd),
            Some(Path::new("config.toml"))
        );
    }

    #[test]
    fn gpu_native_layer0_attention_first_divergence_cli_parses_expected_flags() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "diagnose-gpu-native-layer0-attention-first-divergence",
            "--config",
            "config.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--case",
            "json-transformation",
            "--report-out",
            "layer0_report.json",
            "--slice11b-report",
            "slice11b.json",
        ])
        .unwrap();

        match &cli.cmd {
            Cmd::DiagnoseGpuNativeLayer0AttentionFirstDivergence {
                config,
                expected_adapter_name,
                case,
                report_out,
                slice11b_report,
            } => {
                assert_eq!(config, &PathBuf::from("config.toml"));
                assert_eq!(expected_adapter_name, "NVIDIA L4");
                assert_eq!(case, "json-transformation");
                assert_eq!(report_out, &Some(PathBuf::from("layer0_report.json")));
                assert_eq!(slice11b_report, &Some(PathBuf::from("slice11b.json")));
            }
            _ => panic!("unexpected command variant"),
        }
        assert_eq!(
            super::startup_config_path(&cli.cmd),
            Some(Path::new("config.toml"))
        );
    }

    #[test]
    fn startup_config_path_resolves_gpu_native_layer0_attention_first_divergence() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "diagnose-gpu-native-layer0-attention-first-divergence",
            "--config",
            "config.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--case",
            "json-transformation",
        ])
        .unwrap();

        assert_eq!(
            super::startup_config_path(&cli.cmd),
            Some(Path::new("config.toml"))
        );
    }

    #[test]
    fn gpu_native_demand_source_concurrency_cli_parses_production_command() {
        let cli = <Cli as clap::Parser>::try_parse_from([
            "micro-expert-router",
            "qualify-gpu-native-demand-source-concurrency-production",
            "--config",
            "config.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--report-out",
            "pr2a2-production-report.json",
        ])
        .unwrap();

        match &cli.cmd {
            Cmd::QualifyGpuNativeDemandSourceConcurrencyProduction {
                config,
                expected_adapter_name,
                report_out,
            } => {
                assert_eq!(config, &PathBuf::from("config.toml"));
                assert_eq!(expected_adapter_name, "NVIDIA L4");
                assert_eq!(
                    report_out,
                    &PathBuf::from("pr2a2-production-report.json")
                );
            }
            _ => panic!("unexpected command variant"),
        }
        assert_eq!(
            super::startup_config_path(&cli.cmd),
            Some(Path::new("config.toml"))
        );
    }

    #[test]
    fn isolated_execution_context_contract_selects_expected_planes() {
        let mut cfg = minimal_bench_cfg();
        cfg.real_transformer.enabled = true;
        cfg.real_transformer.gpu_native = true;
        cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Gpu;
        let gpu_spec = ResolvedRealCliSpec {
            cfg: cfg.clone(),
            architecture: crate::architecture::Architecture::Qwen3Moe,
            first_k_dense_replace: 0,
            advanced: crate::model::AdvancedConfig::default(),
        };
        assert_eq!(
            isolated_execution_context_contract(&gpu_spec, crate::backend::ComputeOffload::Gpu),
            IsolatedExecutionContextContract::GpuNativeAuthoritative
        );

        let mut cpu_cfg = cfg.clone();
        cpu_cfg.real_transformer.gpu_native = false;
        cpu_cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Cpu;
        let ref_spec = ResolvedRealCliSpec {
            cfg: cpu_cfg,
            architecture: crate::architecture::Architecture::Qwen3Moe,
            first_k_dense_replace: 0,
            advanced: crate::model::AdvancedConfig::default(),
        };
        assert_eq!(
            isolated_execution_context_contract(&ref_spec, crate::backend::ComputeOffload::Cpu),
            IsolatedExecutionContextContract::CpuReference
        );

        let mut hybrid_cfg = cfg.clone();
        hybrid_cfg.real_transformer.gpu_native = false;
        hybrid_cfg.real_transformer.compute_offload = crate::backend::ComputeOffload::Hybrid;
        let hybrid_spec = ResolvedRealCliSpec {
            cfg: hybrid_cfg,
            architecture: crate::architecture::Architecture::Qwen3Moe,
            first_k_dense_replace: 0,
            advanced: crate::model::AdvancedConfig::default(),
        };
        assert_eq!(
            isolated_execution_context_contract(&hybrid_spec, crate::backend::ComputeOffload::Hybrid),
            IsolatedExecutionContextContract::HybridRoutedExperts
        );

        assert!(RealCliRuntimeMode::IsolatedGpuNativeDiagnostic.is_isolated());
        assert!(RealCliRuntimeMode::IsolatedGpuNativeDiagnostic.installs_logical_gpu_cache());
        assert_eq!(
            RealCliRuntimeMode::IsolatedGpuNativeDiagnostic.tokenizer_command(),
            "diagnose-gpu-native-q4-first-divergence"
        );
    }
}
