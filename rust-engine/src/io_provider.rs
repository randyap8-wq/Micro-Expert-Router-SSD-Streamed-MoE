//! NVMe storage provider.
//!
//! Opens each expert as its own file (optionally with `O_DIRECT` on Linux),
//! keeps the file descriptors resident in an fd cache, and reads experts
//! into a [`PooledBuffer`] owned by the caller via positional reads
//! (`pread(2)`).
//!
//! ## Why this layout?
//!
//! * **One fd per expert, kept open.** Removes per-read `open()` syscalls
//!   from the steady-state latency budget — the same property an io_uring
//!   path would want with registered files.
//! * **`O_DIRECT`.** Bypasses the page cache so we measure (and consume)
//!   raw NVMe bandwidth. Required by the spec; the buffer pool guarantees
//!   the alignment invariants the kernel checks.
//! * **`block_in_place` + `pread`.** Each cache miss runs the synchronous
//!   `pread(2)` on the current Tokio worker via
//!   [`tokio::task::block_in_place`]. The worker is donated to blocking
//!   work and other ready tasks are picked up by sibling workers, so the
//!   runtime stays responsive. Using `pread` (positional) instead of
//!   `read_exact` means we do not touch the file offset and reads are
//!   safe to issue concurrently against the same fd.
//! * **Optional `--no-direct` fallback.** Useful on tmpfs / overlayfs /
//!   non-Linux CI where `O_DIRECT` returns `EINVAL`. The engine still
//!   exercises the same prefetch + LRU + alignment logic.
//!
//! ## Why not `rio`?
//!
//! Earlier drafts used the [`rio`] io_uring crate directly. `rio 0.9.4` has
//! an unfixed use-after-free advisory and the crate is unmaintained, so we
//! removed the dependency. The intended production replacement is
//! [`tokio-uring`] (or the raw `io-uring` crate with registered fixed
//! buffers); both require restructuring `main` around their own runtime
//! entry points (`tokio_uring::start`), which is left as future work. The
//! `io_provider` module is the only place that needs to change to swap
//! backends, so this is a self-contained migration.

use crate::buffer_pool::PooledBuffer;
use crate::inference::WeightDtype;
use crate::tensor_header::{TensorHeader, UTH_BYTES};
use lru::LruCache;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::num::NonZeroUsize;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;

// =====================================================================
// Fault-tolerant I/O knobs (gist Task 3 — "Hardened" MER).
// =====================================================================

/// Maximum number of times the storage layer will retry a transient
/// I/O failure (EIO / EINTR / `Interrupted` / `WouldBlock`) before
/// surfacing the error.
///
/// Three attempts is the "industry-standard 3-tier" retry the gist
/// asks for; further attempts are pointless on real drive failures
/// and only add latency.
pub const STORAGE_RETRY_ATTEMPTS: u32 = 3;

/// Initial backoff between retries. Doubles on every retry (exponential
/// backoff), capped by [`STORAGE_RETRY_MAX_BACKOFF`].
pub const STORAGE_RETRY_BACKOFF: Duration = Duration::from_millis(5);

/// Upper bound on the backoff between retries — keeps a hung-drive
/// situation from stretching a single token's latency beyond what the
/// HTTP server's per-request timeout will tolerate.
pub const STORAGE_RETRY_MAX_BACKOFF: Duration = Duration::from_millis(50);

/// Consecutive-failure threshold at which the per-expert circuit
/// breaker trips and subsequent reads short-circuit to
/// `HardwareFailure` instead of touching the drive. A tripped breaker
/// is reset by a successful read.
///
/// **Gist Phase 3 ("Hardware Failure Recovery").** Per the spec, the
/// threshold is **3** consecutive read operations returning a
/// timeout / EIO / checksum-style error — short enough to react to a
/// failing NVMe shard inside one inference cycle, long enough to
/// tolerate the kind of single-millisecond firmware hiccup that
/// otherwise produces a false-positive trip.
pub const STORAGE_BREAKER_THRESHOLD: u32 = 3;

/// Consecutive-failure threshold at which the **per-drive** circuit
/// breaker trips. When this many reads against *any* expert routed
/// to the same drive fail consecutively, subsequent reads against
/// that drive short-circuit to `HardwareFailure::DriveUnavailable`
/// and the higher-level scheduler stops routing to it (gist Phase 3
/// — "stop attempting to route to that specific drive").
///
/// In the single-drive layout `NvmeStorage::new` exposes one virtual
/// drive, so a tripped per-drive breaker is equivalent to a global
/// engine pause for new SSD reads — the cache + in-flight in-memory
/// responses can still serve traffic until they drain.
pub const STORAGE_DRIVE_BREAKER_THRESHOLD: u32 = 3;

/// How long after a circuit breaker trips before a single probe read
/// is allowed through to test recovery. The breaker stays "open" to
/// every other read during this window, then transitions to
/// "half-open" — exactly one in-flight probe is admitted, and:
///
/// * if the probe **succeeds**, [`NvmeStorage::note_read_success`]
///   clears the `tripped` bit and the breaker returns to "closed"
///   (full traffic);
/// * if the probe **fails**, [`NvmeStorage::note_read_failure`]
///   re-stamps `tripped_at_ms` to "now" so the next probe is held
///   off for another `STORAGE_BREAKER_PROBE_INTERVAL`.
///
/// Without this half-open path a tripped breaker would short-circuit
/// every subsequent read forever, making `note_read_success`
/// unreachable through the public `read_*` entry points and rendering
/// the documented "first successful read clears the breaker"
/// semantics unimplementable. The probe gate is a process-wide
/// rate-limit on hopeful retries: the engine is allowed to *try*
/// once every `STORAGE_BREAKER_PROBE_INTERVAL`, no more, no less.
///
/// 500 ms is chosen as a compromise between two goals: fast enough
/// that a recovered drive returns to service within a single
/// inference cycle, slow enough that a genuinely failed drive does
/// not generate continuous I/O pressure.
pub const STORAGE_BREAKER_PROBE_INTERVAL: Duration = Duration::from_millis(500);

/// Process-wide monotonic clock used as the reference point for
/// [`BreakerState::tripped_at_ms`] / [`DriveBreakerState::tripped_at_ms`].
/// Storing elapsed milliseconds in an `AtomicU64` keeps the breaker
/// state lock-free; the `Instant` itself can't be packed into one.
fn breaker_clock_origin() -> Instant {
    use std::sync::OnceLock;
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

/// Milliseconds elapsed since [`breaker_clock_origin`].
#[inline]
fn breaker_now_ms() -> u64 {
    // Clamp to at least 1 so the value never collides with the
    // "no trip on record" sentinel `0` used by
    // [`BreakerState::tripped_at_ms`] / [`DriveBreakerState::tripped_at_ms`].
    // Without this clamp a trip recorded in the very first
    // millisecond of the process (common in tests) would be
    // indistinguishable from a never-tripped breaker, and
    // `try_admit_probe` would admit traffic immediately.
    (breaker_clock_origin().elapsed().as_millis() as u64).max(1)
}

/// Default upper bound on the number of expert file descriptors kept
/// open in [`NvmeStorage`]'s bounded fd cache, derived from the
/// process's `RLIMIT_NOFILE` soft limit.
///
/// The legacy fd cache was an unbounded `HashMap<u32, Arc<File>>` —
/// one fd per *distinct* expert, never evicted. With per-expert files
/// and a low cache-hit rate (the SSD-streamed regime this engine
/// targets) that map grows without bound and the process eventually
/// hits `EMFILE` ("Too many open files"). Bounding the cache to a
/// fraction of the soft limit keeps the steady-state fd count flat
/// regardless of how many experts stream past.
///
/// We reserve headroom for the rest of the engine's fds (listening
/// sockets, the cold-start manifest, log files, in-flight read fds
/// whose `Arc<File>` outlived their cache slot) and clamp the result
/// to a sane `[64, 65536]` range so a tiny `ulimit -n` cannot disable
/// caching entirely and an effectively-unlimited rlimit does not try
/// to keep millions of descriptors resident.
fn default_fd_cache_cap() -> usize {
    const RESERVED_FDS: usize = 128;
    const MIN_CAP: usize = 64;
    const MAX_CAP: usize = 65_536;
    let soft = unsafe {
        let mut rl: libc::rlimit = std::mem::zeroed();
        if libc::getrlimit(libc::RLIMIT_NOFILE, &mut rl) == 0 {
            // `rlim_cur` is a 64-bit `rlim_t`; on a 32-bit `usize`
            // platform an effectively-unlimited limit (`RLIM_INFINITY`)
            // would truncate, so fall back to the conservative default
            // and let the clamp below cap it.
            usize::try_from(rl.rlim_cur).unwrap_or(1024)
        } else {
            // Conservative default if the rlimit query fails.
            1024
        }
    };
    soft.saturating_sub(RESERVED_FDS).clamp(MIN_CAP, MAX_CAP)
}

/// Classify an `io::Error` as transient (worth retrying) vs. fatal
/// (worth surfacing immediately to the circuit breaker).
///
/// `UnexpectedEof` is **not** treated as transient even though
/// `read_at_with_retries` synthesises it for short reads. A genuinely
/// truncated `expert_<id>.bin` is not going to grow back during the
/// next 50 ms of backoff, so retrying it three times is pointless —
/// we count the failure against the breaker on the first read and
/// surface the error immediately.
fn is_transient_io_error(e: &io::Error) -> bool {
    use io::ErrorKind::*;
    matches!(e.kind(), Interrupted | WouldBlock | TimedOut)
        || e.raw_os_error() == Some(libc::EIO)
        || e.raw_os_error() == Some(libc::EAGAIN)
        || e.raw_os_error() == Some(libc::EBUSY)
}

/// Saturating `fetch_add(1)` for an `AtomicU32`. Returns the post-
/// increment value, clamped at `u32::MAX`. The breaker thresholds
/// fit comfortably in `u32` (single digits), so a naive `fetch_add`
/// can theoretically wrap to 0 after 2^32 consecutive failures and
/// silently un-trip the breaker. Implemented as a CAS loop on the
/// load — at the rate the failure path runs in practice this loop
/// completes in a single iteration. (F3.2 in the audit.)
#[inline]
fn saturating_increment_u32(c: &AtomicU32) -> u32 {
    let mut cur = c.load(Ordering::Acquire);
    loop {
        if cur == u32::MAX {
            return u32::MAX;
        }
        match c.compare_exchange_weak(cur, cur + 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return cur + 1,
            Err(observed) => cur = observed,
        }
    }
}

/// Configuration for the storage layer.
#[derive(Clone, Debug)]
pub struct StorageConfig {
    /// Directory containing `expert_<id>.bin` files (or
    /// `expert_<layer>_<local_id>.bin` when [`Self::num_experts_per_layer`]
    /// is set; see also [`NvmeStorage::expert_path`]).
    pub base_path: PathBuf,
    /// Size (bytes) of every expert file. Must be a multiple of `block_align`.
    pub expert_size: usize,
    /// Logical block size to use for `O_DIRECT` alignment (typically 4096).
    pub block_align: usize,
    /// Whether to open files with `O_DIRECT` and bypass the page cache.
    pub use_direct_io: bool,
    /// Optional: number of experts per MoE layer in a multi-layer
    /// model. When `Some(n)` and `n > 0`, [`NvmeStorage::expert_path`]
    /// resolves the global expert id `g` to
    /// `expert_<g / n>_<g % n>.bin` whenever the legacy single-namespace
    /// `expert_<g>.bin` file does not exist. This makes the storage
    /// layer compatible with both the legacy GGUF-converter naming
    /// (single global namespace) **and** the multi-layer HF extractor
    /// naming (per-layer namespace) without requiring a second copy
    /// of the weight files. `None` (default) preserves the original
    /// single-namespace behaviour exactly.
    pub num_experts_per_layer: Option<u32>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::new(),
            expert_size: 0,
            block_align: 4096,
            use_direct_io: false,
            num_experts_per_layer: None,
        }
    }
}

// =====================================================================
// Per-expert circuit-breaker state (gist Task 3 — "Hardened" MER).
// =====================================================================

/// Sticky per-expert health state. Allocated lazily the first time an
/// expert id is touched and kept resident in [`NvmeStorage::breakers`]
/// for the lifetime of the engine.
#[derive(Debug)]
pub struct BreakerState {
    /// Number of consecutive failed reads since the last success.
    /// Reset to 0 on the first successful `read_at`.
    pub consecutive_failures: AtomicU32,
    /// Sticky "permanently failed" bit. Once set, every subsequent
    /// read against this expert short-circuits to
    /// [`HardwareFailure::ExpertUnavailable`] without touching the
    /// drive *until the half-open probe gate admits one read* (see
    /// [`STORAGE_BREAKER_PROBE_INTERVAL`]). Cleared automatically
    /// the moment the breaker observes a healthy read again (a
    /// transient firmware hiccup must not permanently take an expert
    /// out of rotation).
    pub tripped: AtomicBool,
    /// Reference timestamp in milliseconds since
    /// [`breaker_clock_origin`], updated each time the breaker (a)
    /// first trips or (b) admits a probe read. `0` while the breaker
    /// is closed. Used by [`NvmeStorage::try_admit_probe`] to
    /// rate-limit probe attempts to one per
    /// [`STORAGE_BREAKER_PROBE_INTERVAL`].
    pub tripped_at_ms: AtomicU64,
}

impl Default for BreakerState {
    fn default() -> Self {
        Self {
            consecutive_failures: AtomicU32::new(0),
            tripped: AtomicBool::new(false),
            tripped_at_ms: AtomicU64::new(0),
        }
    }
}

/// Sticky per-drive health state — the aggregate cousin of
/// [`BreakerState`] that lets the storage layer stop routing to a
/// failed shard before *every* expert on that shard has independently
/// tripped its own breaker. Gist Phase 3 ("Hardware Failure
/// Recovery"): "if three consecutive read operations against the
/// same drive return a timeout or checksum error, the controller
/// must stop attempting to route to that specific drive".
///
/// One [`DriveBreakerState`] is kept per shard in [`NvmeStorage`]'s
/// striped layout, indexed by `id % n_drives`. The single-drive
/// layout has exactly one entry, in which case the drive breaker is
/// the engine-wide fail-safe.
#[derive(Debug)]
pub struct DriveBreakerState {
    /// Number of consecutive failed reads across **all** experts
    /// routed to this drive since the last successful read.
    pub consecutive_failures: AtomicU32,
    /// Sticky "drive unavailable" flag. Reads short-circuit until
    /// the next successful read on this drive clears it. The
    /// half-open probe gate (see [`STORAGE_BREAKER_PROBE_INTERVAL`])
    /// admits one probe per interval so a recovered drive can
    /// actually reach the read path that calls `note_read_success`.
    pub tripped: AtomicBool,
    /// Reference timestamp (ms since [`breaker_clock_origin`]) for
    /// the probe gate. See [`BreakerState::tripped_at_ms`].
    pub tripped_at_ms: AtomicU64,
}

impl Default for DriveBreakerState {
    fn default() -> Self {
        Self {
            consecutive_failures: AtomicU32::new(0),
            tripped: AtomicBool::new(false),
            tripped_at_ms: AtomicU64::new(0),
        }
    }
}

/// Outcome of the storage layer's fault-tolerant read path.
///
/// The variants distinguish:
///
/// * `Transient` — every retry returned a transient `io::Error`. The
///   upstream caller may choose to retry the whole request (a higher-
///   level loop in `Engine::moe_step` does this for prefetches).
/// * `ExpertUnavailable` — the per-expert circuit breaker has tripped
///   after `STORAGE_BREAKER_THRESHOLD` consecutive failures. Future
///   reads against this id short-circuit without touching the drive
///   until the breaker resets.
///
/// Surfaced to the HTTP server as a 503 by `server.rs`.
#[derive(Debug)]
pub enum HardwareFailure {
    /// Every retry returned an `io::Error` we classified as transient.
    /// The inner error is the most recent one.
    Transient {
        expert_id: u32,
        attempts: u32,
        last_error: io::Error,
    },
    /// Per-expert circuit breaker has tripped.
    ExpertUnavailable {
        expert_id: u32,
        consecutive_failures: u32,
    },
    /// Per-drive circuit breaker has tripped — the storage layer
    /// should stop routing **any** new reads to this shard until a
    /// successful read clears the breaker. Surfaces the failing
    /// drive index (`id % n_drives` for striped layouts) so
    /// dashboards and pagers can correlate the trip with a specific
    /// device. Gist Phase 3.
    DriveUnavailable {
        drive_index: usize,
        consecutive_failures: u32,
    },
}

impl std::fmt::Display for HardwareFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HardwareFailure::Transient { expert_id, attempts, last_error } => write!(
                f,
                "hardware failure reading expert {expert_id} after {attempts} attempts: {last_error}"
            ),
            HardwareFailure::ExpertUnavailable { expert_id, consecutive_failures } => write!(
                f,
                "expert {expert_id} marked unavailable by the circuit breaker \
                 ({consecutive_failures} consecutive failures)"
            ),
            HardwareFailure::DriveUnavailable { drive_index, consecutive_failures } => write!(
                f,
                "drive {drive_index} marked unavailable by the circuit breaker \
                 ({consecutive_failures} consecutive failures across all experts)"
            ),
        }
    }
}

impl std::error::Error for HardwareFailure {}

impl From<HardwareFailure> for io::Error {
    /// Convert a `HardwareFailure` to an `io::Error` so call sites
    /// that already plumb `io::Result` (the entire I/O layer) can
    /// surface it without an enum re-wrapping. The original
    /// `HardwareFailure` is preserved as the inner error so
    /// `downcast_ref::<HardwareFailure>()` recovers the full
    /// information.
    fn from(e: HardwareFailure) -> io::Error {
        io::Error::new(io::ErrorKind::Other, e)
    }
}

// =====================================================================

/// Source-upload proof activity only. Read/reset at an idle qualification boundary;
/// relaxed atomic loads are not a coherent snapshot while proofs are in flight.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub(crate) struct SourceUploadFdProofSnapshot {
    pub(crate) source_upload_fd_proof_requests: u64,
    pub(crate) source_upload_fd_proof_hits: u64,
    pub(crate) source_upload_fd_proof_misses: u64,
    pub(crate) source_upload_fd_proof_failures: u64,
}

#[derive(Default)]
struct SourceUploadFdProofCache {
    // One weak identity per expert, never an additional owner of an open fd.
    proven: Mutex<HashMap<u32, Weak<File>>>,
    requests: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    failures: AtomicU64,
}

/// Keep the real validation shared with controlled syscall-failure tests.
#[cfg(any(target_os = "linux", test))]
fn validate_source_upload_fd(
    expected_len: u64,
    direct: impl FnOnce() -> io::Result<bool>,
    length: impl FnOnce() -> io::Result<u64>,
) -> io::Result<()> {
    if !direct()? || length()? != expected_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "source/upload source fd must be O_DIRECT and exactly one full expert",
        ));
    }
    Ok(())
}

pub struct NvmeStorage {
    cfg: StorageConfig,
    /// **Bounded** LRU cache of open expert file descriptors.
    ///
    /// Replaces the former unbounded `HashMap<u32, Arc<File>>`, which
    /// kept one fd per distinct expert forever and made the engine
    /// crash with `EMFILE` ("Too many open files") once enough experts
    /// streamed past in a low-cache-hit-rate run. The cache holds at
    /// most [`Self::max_open_files`] descriptors; opening a new expert
    /// once the cache is full evicts the least-recently-used fd. An
    /// evicted fd whose `Arc<File>` is still referenced by an in-flight
    /// read stays open until that read drops its clone, so eviction is
    /// always safe. Guarded by a [`Mutex`] (not an `RwLock`) because
    /// `LruCache::get` mutates recency order.
    files: Mutex<LruCache<u32, Arc<File>>>,
    source_upload_fd_proofs: SourceUploadFdProofCache,
    /// Optional multi-drive layout. When non-empty, expert `id` lives at
    /// `extra_paths[id as usize % extra_paths.len()] / expert_<id>.bin`
    /// (with `cfg.base_path` *included* as `extra_paths[0]`). When
    /// empty, only `cfg.base_path` is consulted — the legacy single-
    /// drive layout. Gist Phase 4 (multi-drive striping).
    striped_paths: Vec<PathBuf>,
    /// Optional cold-start manifest. When attached via
    /// [`NvmeStorage::with_manifest`], `expert_path(id)` consults it
    /// first and avoids re-walking the multi-namespace fallback +
    /// the per-call `metadata()` syscall.  Built once at engine boot
    /// (see [`Manifest::scan`]) and shared across the steady-state
    /// fetch path. Wrapped in an [`Arc`] for cheap cloning into
    /// background prefetch tasks.
    manifest: Option<Arc<Manifest>>,
    /// Per-expert circuit-breaker state (gist Task 3). Tracks
    /// consecutive read failures and a "tripped" sticky flag. Reads
    /// against a tripped breaker short-circuit to
    /// [`HardwareFailure`] instead of touching the drive. The first
    /// successful read clears the failure counter and resets the
    /// breaker — drives that recover after a brief glitch (a common
    /// pattern on hot NVMe enclosures) keep serving traffic without
    /// operator intervention.
    breakers: RwLock<HashMap<u32, Arc<BreakerState>>>,
    /// Per-drive circuit-breaker state (gist Phase 3 — "stop
    /// attempting to route to that specific drive"). Indexed by
    /// `id % n_drives` for striped layouts; the single-drive layout
    /// keeps a one-element vector. Eagerly sized at construction so
    /// the hot path is a plain index load, no `RwLock` access.
    drive_breakers: Vec<Arc<DriveBreakerState>>,
    /// **Tier 2.** Optional packed-blob layout. When attached via
    /// [`NvmeStorage::with_packed_blob`], every expert is read from a
    /// single shared blob fd at the manifest-recorded offset, and
    /// [`Self::read_experts_batch`] coalesces physically-adjacent
    /// experts into one vectored `preadv`. `None` (default) keeps the
    /// original one-file-per-expert path bit-for-bit.
    packed: Option<Arc<crate::packed_storage::PackedBlob>>,
}

impl NvmeStorage {
    pub fn new(cfg: StorageConfig) -> io::Result<Self> {
        assert!(cfg.block_align.is_power_of_two());
        assert!(
            cfg.expert_size % cfg.block_align == 0,
            "expert_size {} must be a multiple of block_align {}",
            cfg.expert_size,
            cfg.block_align
        );

        Ok(Self {
            cfg,
            files: Mutex::new(LruCache::new(
                NonZeroUsize::new(default_fd_cache_cap())
                    .expect("default_fd_cache_cap() is clamped to >= 64"),
            )),
            source_upload_fd_proofs: SourceUploadFdProofCache::default(),
            striped_paths: Vec::new(),
            manifest: None,
            breakers: RwLock::new(HashMap::new()),
            drive_breakers: vec![Arc::new(DriveBreakerState::default())],
            packed: None,
        })
    }

    /// Override the bounded fd-cache capacity (see [`Self::files`]).
    ///
    /// Builder-style for chaining at construction
    /// (`NvmeStorage::new(cfg)?.with_max_open_files(4096)`). The default
    /// is derived from `RLIMIT_NOFILE` via [`default_fd_cache_cap`];
    /// callers that know the exact working set (or want a tighter bound
    /// in tests) can pin it here. `cap` is clamped to at least 1.
    ///
    /// Call this **before** any fd-opening operation (e.g. before
    /// `warmup_fds`): it replaces the cache wholesale, dropping any
    /// descriptors already opened. In practice it is only used at
    /// construction, immediately after `new`, so the cache is empty.
    pub fn with_max_open_files(self, cap: usize) -> Self {
        let cap = NonZeroUsize::new(cap.max(1)).expect("cap.max(1) >= 1");
        *self.files.lock() = LruCache::new(cap);
        self
    }

    /// Current bound on the number of simultaneously-open expert fds.
    pub fn max_open_files(&self) -> usize {
        self.files.lock().cap().get()
    }

    /// Construct a multi-drive striped storage. Experts are distributed
    /// across `dirs` by `id % dirs.len()`, so cache-miss reads issued
    /// concurrently for distinct expert ids hit *different* NVMe
    /// devices in the common case — the queue-depth advantage scales
    /// linearly with `dirs.len()` until the host PCIe link saturates.
    ///
    /// Layout invariants the rest of the engine relies on are
    /// preserved: every expert file is still `expert_size` bytes
    /// aligned to `block_align`, and `read_expert` returns the same
    /// `PooledBuffer` shape. The fd cache is shared across all drives.
    ///
    /// Compatible single-drive behaviour: when `dirs.len() == 1`, this
    /// is equivalent to [`Self::new`] with `cfg.base_path =
    /// dirs[0]`.
    pub fn striped(cfg: StorageConfig, dirs: Vec<PathBuf>) -> io::Result<Self> {
        if dirs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "NvmeStorage::striped requires at least one directory",
            ));
        }
        let mut cfg = cfg;
        cfg.base_path = dirs[0].clone();
        let mut s = Self::new(cfg)?;
        s.drive_breakers = (0..dirs.len())
            .map(|_| Arc::new(DriveBreakerState::default()))
            .collect();
        s.striped_paths = dirs;
        Ok(s)
    }

    /// Attach a pre-built [`Manifest`] (typically produced by
    /// [`Manifest::scan`] at engine boot) to this storage. The
    /// manifest's `payload_offset` / `dtype` lookups are then
    /// available to the rest of the engine via [`Self::manifest`],
    /// and [`Self::expert_path`] short-circuits the multi-namespace
    /// resolution + `metadata()` syscall on every call.
    ///
    /// Builder-style: returns `self` for chaining at construction time
    /// (`NvmeStorage::new(cfg)?.with_manifest(m)`).
    pub fn with_manifest(mut self, manifest: Arc<Manifest>) -> Self {
        self.manifest = Some(manifest);
        self
    }

    /// The cold-start manifest attached via [`Self::with_manifest`],
    /// if any. `None` means the storage was constructed without
    /// indexing — the legacy per-fetch resolution path is used.
    pub fn manifest(&self) -> Option<&Arc<Manifest>> {
        self.manifest.as_ref()
    }

    /// **Tier 2.** Attach a packed blob (single-file layout + manifest)
    /// produced by the `repack` CLI subcommand. Once attached, every
    /// `read_expert*` call sources bytes from the shared blob fd at the
    /// expert's recorded offset, and [`Self::read_experts_batch`]
    /// coalesces adjacent experts into one vectored `preadv`.
    ///
    /// Builder-style: returns `self` for chaining at construction
    /// (`NvmeStorage::new(cfg)?.with_packed_blob(blob)`).
    pub fn with_packed_blob(mut self, blob: Arc<crate::packed_storage::PackedBlob>) -> Self {
        self.packed = Some(blob);
        self
    }

    /// The packed blob attached via [`Self::with_packed_blob`], if any.
    #[allow(dead_code)] // public accessor; engine uses `is_packed()` on the hot path.
    pub fn packed(&self) -> Option<&Arc<crate::packed_storage::PackedBlob>> {
        self.packed.as_ref()
    }

    /// Whether this storage is serving experts from a packed blob.
    pub fn is_packed(&self) -> bool {
        self.packed.is_some()
    }

    /// Number of drives this storage is striped across. `1` for the
    /// legacy single-drive layout.
    pub fn num_drives(&self) -> usize {
        self.striped_paths.len().max(1)
    }

    pub fn config(&self) -> &StorageConfig {
        &self.cfg
    }

    /// Path of the file backing a given expert id.
    ///
    /// Resolution order:
    ///
    /// 1. The legacy single-namespace path
    ///    `<dir>/expert_<id>.bin` (compatible with the GGUF
    ///    converter and the engine's synthetic generators).
    /// 2. If [`StorageConfig::num_experts_per_layer`] is set and the
    ///    legacy file does not exist, the multi-layer extractor path
    ///    `<dir>/expert_<id / n>_<id % n>.bin` (matching
    ///    `scripts/extract_mixtral_experts.py`'s multi-layer dump).
    ///
    /// `<dir>` is `cfg.base_path` for the single-drive layout, or
    /// `striped_paths[id % n_drives]` for striped multi-drive layouts.
    pub fn expert_path(&self, id: u32) -> PathBuf {
        // Zero-latency seek when a manifest is present: the path was
        // already resolved at scan time.
        if let Some(m) = &self.manifest {
            if let Some(entry) = m.lookup(id) {
                return entry.path.clone();
            }
        }
        let dir = if self.striped_paths.is_empty() {
            self.cfg.base_path.clone()
        } else {
            let n = self.striped_paths.len();
            self.striped_paths[(id as usize) % n].clone()
        };
        let primary = dir.join(format!("expert_{id}.bin"));
        // Fast path: the legacy single-namespace file exists. Use
        // metadata() rather than try_exists() so we behave the same
        // as `open_one` would on permission / I/O errors (those still
        // bubble up to the caller).
        if std::fs::metadata(&primary).is_ok() {
            return primary;
        }
        // Multi-layer fallback: resolve to `expert_<layer>_<local>.bin`
        // when the operator told us the per-layer count.
        if let Some(n) = self.cfg.num_experts_per_layer {
            if n > 0 {
                let layer = id / n;
                let local = id % n;
                return dir.join(format!("expert_{layer}_{local}.bin"));
            }
        }
        primary
    }

    fn open_one(&self, id: u32) -> io::Result<Arc<File>> {
        let path = self.expert_path(id);
        let file = open_expert_file(&path, self.cfg.use_direct_io)?;
        Ok(Arc::new(file))
    }

    /// Startup validation: the configured `expert_size` must match the
    /// actual on-disk expert file layout.
    ///
    /// The engine reads exactly `expert_size` bytes from offset 0 of
    /// each expert file and then strips any Unified Tensor Header page.
    /// If the on-disk file is *larger* than `expert_size`, the read
    /// silently drops the file's tail — historically this manifested as
    /// real Mixtral Q4_0 payloads arriving exactly one page (4,096
    /// bytes) short, because `expert_size` had been configured as the
    /// bare weight-payload size while the converter prepends a
    /// page-padded UTH. Those missing bytes were then zero-filled by
    /// the legacy tolerance path, silently corrupting the expert's last
    /// Q4 blocks. This check turns that misconfiguration into a precise
    /// startup error instead.
    ///
    /// Probes the first expert id in `sample_ids` whose backing file
    /// exists (ids whose files are absent are skipped — sparse layouts
    /// are legal). No-ops when a packed blob is attached (the packed
    /// manifest slot size is validated separately) or when no sampled
    /// file exists.
    pub fn validate_expert_file_layout(
        &self,
        sample_ids: impl IntoIterator<Item = u32>,
    ) -> io::Result<()> {
        if self.packed.is_some() {
            return Ok(());
        }
        let expert_size = self.cfg.expert_size as u64;
        let block_align = self.cfg.block_align as u64;
        for id in sample_ids {
            let path = self.expert_path(id);
            let meta = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let file_size = meta.len();
            if file_size == expert_size {
                return Ok(());
            }
            // Distinguish the classic misconfiguration (expert_size
            // excludes the converter's page-padded UTH prefix) from a
            // generic size mismatch by probing the file head for a UTH.
            let mut head = vec![0u8; UTH_BYTES];
            let has_uth = File::open(&path)
                .and_then(|f| f.read_at(&mut head, 0))
                .map(|n| n >= UTH_BYTES && TensorHeader::probe(&head[..n]).is_some())
                .unwrap_or(false);
            let hint = if has_uth && file_size == expert_size + block_align {
                format!(
                    " The file carries a Unified Tensor Header page: model.expert_size \
                     appears to be the bare weight-payload size and must include the \
                     {block_align}-byte UTH page (set expert_size = {file_size}, i.e. the \
                     converter metadata.json value)."
                )
            } else if has_uth {
                " The file carries a Unified Tensor Header page; expert_size must equal \
                 the full file size written by the converter (header page + payload)."
                    .to_string()
            } else {
                String::new()
            };
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "expert file {} is {file_size} bytes but model.expert_size is \
                     {expert_size}; the engine reads exactly expert_size bytes per expert, \
                     so a mismatch truncates or over-reads every expert payload.{hint}",
                    path.display()
                ),
            ));
        }
        Ok(())
    }

    /// Get (and cache) the file handle for an expert id, bounded by the
    /// LRU fd cache (see [`Self::files`]).
    fn fd_for(&self, id: u32) -> io::Result<Arc<File>> {
        // Fast path: already resident. `get` mutates LRU recency, so
        // this needs the write-capable `Mutex` guard.
        {
            let mut guard = self.files.lock();
            if let Some(f) = guard.get(&id) {
                return Ok(f.clone());
            }
        }
        // Open *outside* the lock so a slow `open()` syscall (or an
        // O_DIRECT EINVAL probe) never serialises every other reader.
        let f = self.open_one(id)?;
        let mut guard = self.files.lock();
        // A concurrent caller may have opened the same id while we were
        // unlocked. Prefer the entry already in the cache so every
        // reader shares a single fd; our freshly-opened `File` drops.
        if let Some(existing) = guard.get(&id) {
            return Ok(existing.clone());
        }
        // `put` evicts the least-recently-used fd when the cache is at
        // capacity. The evicted `Arc<File>` is dropped here only if no
        // in-flight read still holds a clone, so the fd stays valid for
        // any read already in progress against it.
        guard.put(id, f.clone());
        Ok(f)
    }

    /// Per-expert circuit-breaker state (gist Task 3). Lazily
    /// initialised on first access; never freed for the lifetime of
    /// the engine. The returned `Arc` is cheap to clone into a
    /// background prefetch task.
    pub fn breaker(&self, id: u32) -> Arc<BreakerState> {
        if let Some(b) = self.breakers.read().get(&id) {
            return b.clone();
        }
        let mut guard = self.breakers.write();
        guard
            .entry(id)
            .or_insert_with(|| Arc::new(BreakerState::default()))
            .clone()
    }

    /// `true` iff the per-expert circuit breaker has tripped after
    /// `STORAGE_BREAKER_THRESHOLD` consecutive failures and reads
    /// against this id will short-circuit.
    pub fn is_expert_unavailable(&self, id: u32) -> bool {
        self.breakers
            .read()
            .get(&id)
            .map(|b| b.tripped.load(Ordering::Acquire))
            .unwrap_or(false)
    }

    /// Record one successful read: reset the failure counter and
    /// clear the tripped flag if it was set. Called from the
    /// fault-tolerant read paths after a confirmed `expert_size`
    /// pread.
    fn note_read_success(&self, id: u32) {
        if let Some(b) = self.breakers.read().get(&id) {
            b.consecutive_failures.store(0, Ordering::Release);
            if b.tripped.swap(false, Ordering::AcqRel) {
                b.tripped_at_ms.store(0, Ordering::Release);
                tracing::info!(
                    expert_id = id,
                    "circuit breaker reset after successful probe read"
                );
            }
        }
        // A successful read also clears the drive-level breaker:
        // whatever transient outage was happening is over.
        let drive_idx = self.drive_index_for(id);
        let db = &self.drive_breakers[drive_idx];
        db.consecutive_failures.store(0, Ordering::Release);
        if db.tripped.swap(false, Ordering::AcqRel) {
            db.tripped_at_ms.store(0, Ordering::Release);
            tracing::info!(
                drive_index = drive_idx,
                "drive circuit breaker reset after successful probe read"
            );
        }
    }

    /// Index of the drive serving `id` under the current layout.
    /// `0` for the single-drive (`new`) layout.
    #[inline]
    fn drive_index_for(&self, id: u32) -> usize {
        if self.striped_paths.is_empty() {
            0
        } else {
            (id as usize) % self.striped_paths.len()
        }
    }

    /// `true` iff the per-drive circuit breaker for the drive that
    /// would serve `id` has tripped. Gist Phase 3 — exposed publicly
    /// so the engine's higher-level scheduling layer can stop
    /// dispatching to a known-bad drive without having to issue an
    /// I/O.
    pub fn is_drive_unavailable(&self, id: u32) -> bool {
        self.drive_breakers[self.drive_index_for(id)]
            .tripped
            .load(Ordering::Acquire)
    }

    /// Record one failed read. Returns `(just_tripped, count)`:
    /// `just_tripped` is `true` only on the call that flips the
    /// breaker from closed to open (so callers can log once);
    /// `count` is the post-increment consecutive-failure tally,
    /// which the caller propagates into
    /// `HardwareFailure::ExpertUnavailable.consecutive_failures` so
    /// the value surfaced to the HTTP 503 reflects the actual
    /// observed failure count rather than a placeholder.
    ///
    /// Also bumps the **per-drive** breaker so a shard that sees
    /// failures spread across many experts trips before any one
    /// expert independently does.
    fn note_read_failure(&self, id: u32) -> (bool, u32, bool, u32) {
        let b = self.breaker(id);
        let n = saturating_increment_u32(&b.consecutive_failures);
        let just_tripped = if n >= STORAGE_BREAKER_THRESHOLD {
            // F3.1: stamp `tripped_at_ms` BEFORE publishing the open
            // state. Otherwise a concurrent reader observing
            // `tripped == true` from the swap below can race and
            // load `tripped_at_ms == 0` (the closed-state sentinel),
            // which `try_admit_probe` would treat as "infinitely
            // long ago" and admit a probe at trip-instant. The
            // Release on the timestamp synchronises-with the
            // Acquire load inside `try_admit_probe` through the
            // `tripped` swap's AcqRel ordering.
            let now = breaker_now_ms();
            b.tripped_at_ms.store(now, Ordering::Release);
            let became_open = !b.tripped.swap(true, Ordering::AcqRel);
            if became_open {
                tracing::error!(
                    expert_id = id,
                    consecutive_failures = n,
                    "circuit breaker tripped — expert unavailable until probe succeeds"
                );
            } else {
                // Already open; this is effectively a failed probe.
                // The store above restamped the gate so the next
                // probe is held off another full interval.
            }
            became_open
        } else {
            false
        };
        let drive_idx = self.drive_index_for(id);
        let db = &self.drive_breakers[drive_idx];
        let dn = saturating_increment_u32(&db.consecutive_failures);
        let drive_just_tripped = if dn >= STORAGE_DRIVE_BREAKER_THRESHOLD {
            // Same publish-ordering invariant as above.
            let now = breaker_now_ms();
            db.tripped_at_ms.store(now, Ordering::Release);
            let became_open = !db.tripped.swap(true, Ordering::AcqRel);
            if became_open {
                tracing::error!(
                    drive_index = drive_idx,
                    consecutive_failures = dn,
                    "drive circuit breaker tripped — no further reads will be routed to this shard"
                );
            }
            became_open
        } else {
            false
        };
        (just_tripped, n, drive_just_tripped, dn)
    }

    /// Half-open probe gate. Returns `true` iff the caller is allowed
    /// to attempt one read against a tripped breaker — i.e. at least
    /// [`STORAGE_BREAKER_PROBE_INTERVAL`] has elapsed since the last
    /// probe (or the initial trip) **and** the current task wins the
    /// `compare_exchange` race against any concurrent probe.
    ///
    /// Exactly one probe is admitted per interval per breaker; losers
    /// of the CAS see the short-circuit return as usual. This is the
    /// "half-open" state from the classic circuit-breaker pattern:
    /// the breaker has not yet decided whether the underlying drive
    /// is healthy, so it lets one read through to find out.
    #[inline]
    fn try_admit_probe(tripped_at_ms: &AtomicU64) -> bool {
        let now = breaker_now_ms();
        let last = tripped_at_ms.load(Ordering::Acquire);
        // `last == 0` is the "force-admit" sentinel used by tests and
        // by operators to manually reset the rate-limit. The race
        // that previously made this sentinel observable from a
        // genuinely-tripped breaker — concurrent reader sees
        // `tripped == true` from `swap` but `tripped_at_ms == 0`
        // because the failing thread had not yet stored its
        // timestamp — is fixed in `note_read_failure`, which now
        // stamps `tripped_at_ms` (Release) *before* publishing
        // `tripped = true` via swap (AcqRel). Any caller that
        // arrived here because they observed `tripped == true`
        // synchronises-with that Release and therefore reads the
        // real, non-zero stamp.
        if last != 0 && now.saturating_sub(last) < STORAGE_BREAKER_PROBE_INTERVAL.as_millis() as u64
        {
            return false;
        }
        // CAS so concurrent callers race for the single probe slot.
        // The winner advances `tripped_at_ms` to `now`; losers fall
        // back to the short-circuit.
        tripped_at_ms
            .compare_exchange(last, now, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Fault-tolerant `pread(2)` wrapper with three-tier retry +
    /// circuit breaker. Used by every public `read_expert*` entry
    /// point. The returned `io::Error` wraps a [`HardwareFailure`]
    /// (recoverable via `e.into_inner().downcast::<HardwareFailure>()`)
    /// when the retry budget is exhausted or the breaker has tripped.
    fn read_at_with_retries(
        &self,
        file: &File,
        expert_id: u32,
        offset: u64,
        dst: &mut [u8],
    ) -> io::Result<usize> {
        // Fast path: per-drive breaker is tripped. Drive failures
        // are sticky and affect every expert sharded onto this
        // drive, so we short-circuit before even consulting
        // `is_expert_unavailable` — *unless* the half-open probe
        // gate admits exactly one recovery read. Without that gate
        // the breaker could never reach the read path that calls
        // `note_read_success`, leaving a recovered drive offline
        // forever. Gist Phase 3.
        let drive_index = self.drive_index_for(expert_id);
        if self.is_drive_unavailable(expert_id) {
            let db = &self.drive_breakers[drive_index];
            if !Self::try_admit_probe(&db.tripped_at_ms) {
                let cf = db.consecutive_failures.load(Ordering::Acquire);
                return Err(HardwareFailure::DriveUnavailable {
                    drive_index,
                    consecutive_failures: cf,
                }
                .into());
            }
            tracing::info!(
                drive_index,
                "drive circuit breaker half-open — admitting one probe read"
            );
        }
        // Fast path: per-expert breaker is tripped. Same half-open
        // probe gate applies so a recovered expert can clear it.
        if self.is_expert_unavailable(expert_id) {
            let b = self.breaker(expert_id);
            if !Self::try_admit_probe(&b.tripped_at_ms) {
                let cf = b.consecutive_failures.load(Ordering::Acquire);
                return Err(HardwareFailure::ExpertUnavailable {
                    expert_id,
                    consecutive_failures: cf,
                }
                .into());
            }
            tracing::info!(
                expert_id,
                "expert circuit breaker half-open — admitting one probe read"
            );
        }
        let mut last_err: Option<io::Error> = None;
        let mut backoff = STORAGE_RETRY_BACKOFF;
        for attempt in 0..STORAGE_RETRY_ATTEMPTS {
            match file.read_at(dst, offset) {
                Ok(n) if n == dst.len() => {
                    self.note_read_success(expert_id);
                    return Ok(n);
                }
                Ok(n) => {
                    // Short read — almost certainly a permanently
                    // truncated file. Not worth retrying (the file
                    // isn't going to grow back in 50 ms); count it
                    // against the breaker and surface fast.
                    let (tripped, cf, drive_tripped, dcf) = self.note_read_failure(expert_id);
                    let err = io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        format!(
                            "short read on expert {expert_id}: got {n} bytes, expected {}",
                            dst.len()
                        ),
                    );
                    if drive_tripped {
                        return Err(HardwareFailure::DriveUnavailable {
                            drive_index: self.drive_index_for(expert_id),
                            consecutive_failures: dcf,
                        }
                        .into());
                    }
                    if tripped {
                        return Err(HardwareFailure::ExpertUnavailable {
                            expert_id,
                            consecutive_failures: cf,
                        }
                        .into());
                    }
                    return Err(err);
                }
                Err(e) if is_transient_io_error(&e) => {
                    tracing::warn!(
                        expert_id,
                        attempt = attempt + 1,
                        error = %e,
                        "transient I/O error; retrying"
                    );
                    last_err = Some(e);
                }
                Err(e) => {
                    // Fatal error (permission, broken pipe, etc.): no
                    // point retrying. We still record one logical
                    // failure per `read_at_with_retries` *call* (not
                    // per retry within a call), so a hot fd that
                    // keeps returning the same fatal error trips the
                    // breaker after `STORAGE_BREAKER_THRESHOLD`
                    // *invocations* — i.e. once the layer above has
                    // attempted the expert that many times.
                    let (tripped, cf, drive_tripped, dcf) = self.note_read_failure(expert_id);
                    if drive_tripped {
                        return Err(HardwareFailure::DriveUnavailable {
                            drive_index: self.drive_index_for(expert_id),
                            consecutive_failures: dcf,
                        }
                        .into());
                    }
                    if tripped {
                        return Err(HardwareFailure::ExpertUnavailable {
                            expert_id,
                            consecutive_failures: cf,
                        }
                        .into());
                    }
                    return Err(e);
                }
            }
            // Exponential backoff between retries. We don't sleep
            // after the last attempt — the error is about to bubble
            // up anyway.
            if attempt + 1 < STORAGE_RETRY_ATTEMPTS {
                std::thread::sleep(backoff);
                backoff = (backoff * 2).min(STORAGE_RETRY_MAX_BACKOFF);
            }
        }
        let (tripped, cf, drive_tripped, dcf) = self.note_read_failure(expert_id);
        // `last_err` is always Some here: the only way to fall out of
        // the loop without returning is via the transient-error branch
        // (which sets `last_err`). The `unwrap_or_else` is a
        // defence-in-depth fallback for an impossible state.
        let err = last_err.unwrap_or_else(|| {
            io::Error::other("retry loop exited without recording a transient error (bug)")
        });
        if drive_tripped {
            Err(HardwareFailure::DriveUnavailable {
                drive_index: self.drive_index_for(expert_id),
                consecutive_failures: dcf,
            }
            .into())
        } else if tripped {
            Err(HardwareFailure::ExpertUnavailable {
                expert_id,
                consecutive_failures: cf,
            }
            .into())
        } else {
            Err(HardwareFailure::Transient {
                expert_id,
                attempts: STORAGE_RETRY_ATTEMPTS,
                last_error: err,
            }
            .into())
        }
    }

    /// Warm the fd cache by pre-opening expert fds, taking the `open()`
    /// cost out of the steady-state path.
    ///
    /// Bounded by the LRU fd cache (see [`Self::files`]): warming more
    /// than [`Self::max_open_files`] ids leaves only the most-recently
    /// warmed `max_open_files` fds resident, which is exactly the
    /// behaviour that keeps the engine under its `RLIMIT_NOFILE` budget
    /// for models with more experts than the cache can hold.
    pub fn warmup_fds(&self, ids: impl IntoIterator<Item = u32>) -> io::Result<()> {
        for id in ids {
            self.fd_for(id)?;
        }
        Ok(())
    }

    /// Read the full bytes of `expert_id` into `buf`.
    ///
    /// `buf` must be exactly `expert_size` bytes long and aligned to
    /// `block_align`; the [`BufferPool`](crate::buffer_pool::BufferPool) takes
    /// care of both invariants.
    ///
    /// Returns the number of bytes actually read (which equals `expert_size`
    /// on success — short reads are surfaced as an `UnexpectedEof` error).
    pub async fn read_expert(&self, expert_id: u32, buf: &mut PooledBuffer) -> io::Result<usize> {
        debug_assert_eq!(buf.len(), self.cfg.expert_size);
        // Tier 2 packed path: source the bytes from the shared blob fd at
        // the expert's recorded offset. The fault-tolerant retry +
        // circuit-breaker wrapper is reused unchanged — only the (file,
        // offset) pair differs from the per-file path.
        if let Some(packed) = &self.packed {
            let entry = packed.entry(expert_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("expert {expert_id} is not present in the packed blob manifest"),
                )
            })?;
            let file = packed.file().clone();
            let dst_len = buf.len();
            let n = tokio::task::block_in_place(|| {
                self.read_at_with_retries(
                    &file,
                    expert_id,
                    entry.offset,
                    &mut buf.as_mut_slice()[..dst_len],
                )
            })?;
            return Ok(n);
        }
        // Note: the breaker fast-fail used to live here for symmetry
        // with `read_at_with_retries`, but that bypassed the
        // half-open probe gate (a tripped breaker could never reach
        // the read path that calls `note_read_success`). The
        // short-circuit + probe-gate logic now lives only in
        // `read_at_with_retries`, which we always go through below.
        let file = self.fd_for(expert_id)?;
        // `read_at_with_retries` already enforces `dst.len()` and surfaces
        // short reads as transient errors that retry, so no extra
        // length check is needed here. Each expert is stored as its
        // own file, so the read always starts at byte 0.
        let dst_len = buf.len();
        let n = tokio::task::block_in_place(|| {
            self.read_at_with_retries(&file, expert_id, 0, &mut buf.as_mut_slice()[..dst_len])
        })?;
        Ok(n)
    }

    /// Diagnostic-only arbitrary destination seam. Validate before resolving
    /// any fd or entering I/O; ordinary `read_expert` remains unchanged.
    pub(crate) async fn read_expert_into_aligned_slice(
        &self,
        expert_id: u32,
        dst: &mut [u8],
    ) -> io::Result<usize> {
        if dst.len() != self.cfg.expert_size
            || dst.is_empty()
            || !self.cfg.block_align.is_power_of_two()
            || dst.len() % self.cfg.block_align != 0
            || dst.as_ptr() as usize % self.cfg.block_align != 0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "diagnostic destination must be one full expert and block aligned",
            ));
        }
        let (file, offset) = if let Some(packed) = &self.packed {
            let entry = packed.entry(expert_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("expert {expert_id} is not present in the packed blob manifest"),
                )
            })?;
            (packed.file().clone(), entry.offset)
        } else {
            (self.fd_for(expert_id)?, 0)
        };
        tokio::task::block_in_place(|| self.read_at_with_retries(&file, expert_id, offset, dst))
    }

    /// Inspect the actual cached per-file fd used by the standalone diagnostic.
    /// Its exclusively owned storage has no concurrent readers or fd eviction
    /// between this probe and the read of the same expert. No fallback opens.
    /// Returns (F_GETFL flags, O_DIRECT observed, actual full-file length).
    pub(crate) fn source_to_upload_fd_evidence(
        &self,
        expert_id: u32,
    ) -> io::Result<(i32, bool, u64)> {
        if self.is_packed() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "source-to-upload diagnostic requires one file per expert",
            ));
        }
        #[cfg(target_os = "linux")]
        {
            let file = self.fd_for(expert_id)?;
            // SAFETY: fd_for owns a live File, and F_GETFL has no third argument.
            let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
            if flags == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok((flags, flags & libc::O_DIRECT != 0, file.metadata()?.len()))
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = expert_id;
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "actual O_DIRECT fd evidence requires Linux",
            ))
        }
    }

    pub(crate) fn source_upload_fd_proof_snapshot(&self) -> SourceUploadFdProofSnapshot {
        let c = &self.source_upload_fd_proofs;
        SourceUploadFdProofSnapshot {
            source_upload_fd_proof_requests: c.requests.load(Ordering::Relaxed),
            source_upload_fd_proof_hits: c.hits.load(Ordering::Relaxed),
            source_upload_fd_proof_misses: c.misses.load(Ordering::Relaxed),
            source_upload_fd_proof_failures: c.failures.load(Ordering::Relaxed),
        }
    }

    /// Caller must ensure no source reads are active. Retain both fd LRU and
    /// proofs: a warmup-proven live Arc remains proven in the measured interval.
    pub(crate) fn reset_source_upload_fd_proof_telemetry(&self) {
        let c = &self.source_upload_fd_proofs;
        c.requests.store(0, Ordering::Relaxed);
        c.hits.store(0, Ordering::Relaxed);
        c.misses.store(0, Ordering::Relaxed);
        c.failures.store(0, Ordering::Relaxed);
    }

    /// Diagnostic idle-boundary preproof only: no source data read. Callers
    /// must retain enough fd-cache capacity for their complete ID universe.
    /// The normal batch helper still performs every normal proof lookup.
    pub(crate) fn preprove_source_upload_fds(&self, ids: &[u32]) -> io::Result<()> {
        for &id in ids {
            let file = self.fd_for(id)?;
            self.prove_source_upload_fd(id, &file)?;
        }
        Ok(())
    }

    fn prove_source_upload_fd(&self, id: u32, file: &Arc<File>) -> io::Result<()> {
        self.prove_source_upload_fd_with(id, file, |file| {
            #[cfg(target_os = "linux")]
            {
                validate_source_upload_fd(
                    self.cfg.expert_size as u64,
                    || {
                        // SAFETY: the caller owns this live File throughout proof
                        // and read. F_GETFL has no third argument.
                        let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
                        if flags < 0 {
                            return Err(io::Error::last_os_error());
                        }
                        Ok(flags & libc::O_DIRECT != 0)
                    },
                    || Ok(file.metadata()?.len()),
                )
            }
            #[cfg(not(target_os = "linux"))]
            {
                let _ = file;
                Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "source/upload direct source requires Linux O_DIRECT",
                ))
            }
        })
    }

    // Production always supplies the real validator above. Tests inject only
    // the proof operation, so identity/concurrency/accounting use the same path.
    fn prove_source_upload_fd_with(
        &self,
        id: u32,
        file: &Arc<File>,
        prove: impl FnOnce(&File) -> io::Result<()>,
    ) -> io::Result<()> {
        let c = &self.source_upload_fd_proofs;
        c.requests.fetch_add(1, Ordering::Relaxed);
        let proven = c
            .proven
            .lock()
            .get(&id)
            .and_then(Weak::upgrade)
            .is_some_and(|proven| Arc::ptr_eq(&proven, file));
        if proven {
            c.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        c.misses.fetch_add(1, Ordering::Relaxed);
        // Neither the proof-cache lock nor the fd-LRU lock spans syscalls.
        // Concurrent misses may duplicate proofs, but publish only on success.
        if let Err(error) = prove(file) {
            c.failures.fetch_add(1, Ordering::Relaxed);
            return Err(error);
        }
        c.proven.lock().insert(id, Arc::downgrade(file));
        Ok(())
    }

    /// Batched read: fill `bufs[i]` with the bytes of `ids[i]`, all in
    /// one blocking-donation. The two slices must have the same length.
    ///
    /// **Why this exists.** When a single token misses on `K > 1` experts,
    /// the engine wants to push all `K` reads into the device's queue
    /// before doing any per-buffer post-processing. The default per-fetch
    /// path runs each `pread(2)` inside its own
    /// [`tokio::task::block_in_place`] call, which means the runtime has
    /// to round-trip between scheduler decisions for every expert. This
    /// helper hoists all `K` reads into one `block_in_place` block and
    /// issues them **concurrently** on scoped OS threads, so the NVMe
    /// genuinely sees parallel queue depth `K` — the same property an
    /// `io_uring` `submit_and_wait(K)` provides on the high-throughput
    /// path.
    ///
    /// On Linux with the `io_uring` cargo feature this method also has
    /// a sibling, `crate::io_uring_storage::IoUringStorage::read_experts_batch_fixed`,
    /// that pushes all `K` reads as `READ_FIXED` SQEs and submits once.
    pub async fn read_experts_batch(
        &self,
        ids: &[u32],
        bufs: &mut [&mut PooledBuffer],
    ) -> io::Result<usize> {
        assert_eq!(
            ids.len(),
            bufs.len(),
            "read_experts_batch: ids and bufs must have the same length"
        );
        if ids.is_empty() {
            return Ok(0);
        }
        // Tier 2 packed path: coalesce physically-adjacent experts into
        // single vectored `preadv` syscalls. Distinct ids ⇒ disjoint
        // destination buffers, so the scatter is sound.
        if self.packed.is_some() {
            return self.read_experts_batch_packed(ids, bufs).await;
        }
        // Resolve all fds before donating the worker — `fd_for` takes a
        // (rare) write lock the first time it sees an id, and we don't
        // want to hold that lock across `block_in_place`.
        let mut files: Vec<Arc<File>> = Vec::with_capacity(ids.len());
        for &id in ids {
            files.push(self.fd_for(id)?);
        }
        let expert_size = self.cfg.expert_size;
        for buf in bufs.iter() {
            debug_assert_eq!(buf.len(), expert_size);
        }

        // Single donation: all K reads dispatched **concurrently** via
        // scoped threads, so the NVMe actually sees queue depth K
        // instead of an effective queue depth of 1 (sequential preads
        // never let the device overlap commands — prefetch reads then
        // queue behind foreground reads). Spawning K-1 short-lived
        // threads costs tens of microseconds; a serialized NVMe read
        // costs hundreds, so the break-even is immediate for K > 1.
        // Each read still passes through the per-expert
        // fault-tolerant path so a single bad drive can't wedge the
        // whole batch — it surfaces as a `HardwareFailure` for that
        // expert id and the engine's higher-level cache code routes
        // around it. The first error (by slot order) is returned.
        let id_vec: Vec<u32> = ids.to_vec();
        let total = tokio::task::block_in_place(|| -> io::Result<usize> {
            if id_vec.len() == 1 {
                // Fast path: no thread spawn for the common single-miss case.
                let buf = &mut *bufs[0];
                return self.read_at_with_retries(
                    &files[0],
                    id_vec[0],
                    0,
                    &mut buf.as_mut_slice()[..expert_size],
                );
            }
            let results: Vec<io::Result<usize>> = std::thread::scope(|scope| {
                let handles: Vec<_> = files
                    .iter()
                    .zip(bufs.iter_mut())
                    .zip(id_vec.iter())
                    .map(|((file, buf), &id)| {
                        let buf: &mut PooledBuffer = &mut *buf;
                        scope.spawn(move || {
                            debug_assert_eq!(buf.len(), expert_size);
                            self.read_at_with_retries(
                                file,
                                id,
                                0,
                                &mut buf.as_mut_slice()[..expert_size],
                            )
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .zip(id_vec.iter())
                    .map(|(h, &id)| {
                        h.join().unwrap_or_else(|_| {
                            Err(io::Error::other(format!(
                                "read_experts_batch worker panicked reading expert {id}"
                            )))
                        })
                    })
                    .collect()
            });
            let mut total = 0usize;
            for r in results {
                total += r?;
            }
            Ok(total)
        })?;
        Ok(total)
    }

    /// Source/upload external destinations with the production per-file scheduler:
    /// resolve every fd, one blocking donation, scoped OS threads, existing
    /// per-expert retries, and errors collected in original slot order. There is
    /// no packed, buffered, or allocation fallback.
    pub(crate) async fn read_experts_batch_into_aligned_slices(
        &self,
        ids: &[u32],
        destinations: &mut [&mut [u8]],
    ) -> io::Result<usize> {
        if self.is_packed()
            || ids.len() != destinations.len()
            || self.cfg.expert_size != 2_658_304
            || self.cfg.block_align != 4096
            || destinations.iter().any(|dst| {
                dst.len() != self.cfg.expert_size
                    || dst.len() % self.cfg.block_align != 0
                    || dst.as_ptr() as usize % self.cfg.block_align != 0
            })
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "source/upload requires unpacked full-file 4096-aligned external destinations",
            ));
        }
        if ids.is_empty() {
            return Ok(0);
        }
        let mut files: Vec<Arc<File>> = Vec::with_capacity(ids.len());
        for &id in ids {
            files.push(self.fd_for(id)?);
        }
        // Prove the exact resolved Arc once; retain resolution and error order.
        for (&id, file) in ids.iter().zip(&files) {
            self.prove_source_upload_fd(id, file)?;
        }
        let id_vec: Vec<u32> = ids.to_vec();
        tokio::task::block_in_place(|| -> io::Result<usize> {
            if id_vec.len() == 1 {
                return self.read_at_with_retries(&files[0], id_vec[0], 0, destinations[0]);
            }
            let results: Vec<io::Result<usize>> = std::thread::scope(|scope| {
                let handles: Vec<_> = files
                    .iter()
                    .zip(destinations.iter_mut())
                    .zip(id_vec.iter())
                    .map(|((file, dst), &id)| {
                        let dst: &mut [u8] = dst;
                        scope.spawn(move || self.read_at_with_retries(file, id, 0, dst))
                    })
                    .collect();
                handles
                    .into_iter()
                    .zip(id_vec.iter())
                    .map(|(h, &id)| {
                        h.join().unwrap_or_else(|_| {
                            Err(io::Error::other(format!(
                                "read_experts_batch worker panicked reading expert {id}"
                            )))
                        })
                    })
                    .collect()
            });
            let mut total = 0usize;
            for result in results {
                total += result?;
            }
            Ok(total)
        })
    }

    /// **Tier 2.** Packed-blob sibling of [`Self::read_experts_batch`].
    ///
    /// Resolves every requested expert to its `(offset, len)` slot in the
    /// shared blob, groups physically-adjacent slots into contiguous runs
    /// (see [`crate::packed_storage::coalesce_runs`]), and issues **one
    /// `preadv(2)` per run** that scatters the bytes straight into the
    /// per-expert [`PooledBuffer`]s. A run of `K` adjacent experts thus
    /// costs a single vectored syscall at full device queue depth instead
    /// of `K` independent `pread`s.
    ///
    /// Robustness: a singleton run, an oversized iovec batch, or any
    /// `preadv` error / short read falls back to the ordinary
    /// per-expert [`Self::read_at_with_retries`] path, so the circuit
    /// breaker, retry budget and short-read accounting behave exactly as
    /// on the one-file-per-expert path.
    #[allow(dead_code)] // reached via `read_experts_batch` (batched-miss path).
    async fn read_experts_batch_packed(
        &self,
        ids: &[u32],
        bufs: &mut [&mut PooledBuffer],
    ) -> io::Result<usize> {
        let packed = self
            .packed
            .as_ref()
            .expect("read_experts_batch_packed called without a packed blob");
        let expert_size = self.cfg.expert_size;

        // Resolve slots and capture the (disjoint) destination pointers.
        // Distinct ids ⇒ distinct slots ⇒ disjoint buffers, so the raw
        // pointers never alias and reconstructing `&mut [u8]` from them
        // inside the blocking closure is sound.
        let mut requested: Vec<(u32, usize, crate::packed_storage::PackedEntry)> =
            Vec::with_capacity(ids.len());
        for (idx, &id) in ids.iter().enumerate() {
            let entry = packed.entry(id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("expert {id} is not present in the packed blob manifest"),
                )
            })?;
            requested.push((id, idx, entry));
        }
        let mut dst: Vec<(*mut u8, usize)> = Vec::with_capacity(bufs.len());
        for buf in bufs.iter_mut() {
            let b: &mut PooledBuffer = &mut *buf;
            debug_assert_eq!(b.len(), expert_size);
            let s = b.as_mut_slice();
            dst.push((s.as_mut_ptr(), s.len()));
        }

        let runs = crate::packed_storage::coalesce_runs(requested);
        let file = packed.file().clone();

        let total = tokio::task::block_in_place(|| -> io::Result<usize> {
            let mut total = 0usize;
            for run in &runs {
                total += self.read_run(&file, run, &dst)?;
            }
            Ok(total)
        })?;
        Ok(total)
    }

    /// Read a single coalesced run into its destination buffers.
    ///
    /// Multi-member runs go through one vectored `preadv`; singletons and
    /// any failed/short `preadv` fall back to the per-expert
    /// fault-tolerant path. `dst[i]` is the `(ptr, len)` of the caller's
    /// `i`-th request buffer.
    #[allow(dead_code)] // reached via `read_experts_batch_packed`.
    fn read_run(
        &self,
        file: &File,
        run: &crate::packed_storage::ContiguousRun,
        dst: &[(*mut u8, usize)],
    ) -> io::Result<usize> {
        // Singleton (or, defensively, an oversized iovec batch): per-expert
        // path. `IOV_MAX` is 1024 on Linux / macOS; top-k miss sets are
        // orders of magnitude smaller, so the cap is never hit in practice.
        const IOV_MAX_SAFE: usize = 512;
        if run.members.len() == 1 || run.members.len() > IOV_MAX_SAFE {
            let mut total = 0usize;
            for &(id, idx) in &run.members {
                let (ptr, len) = dst[idx];
                // SAFETY: disjoint per-expert buffer, valid for `len`
                // bytes for the duration of this call.
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
                let off = self.packed_offset(id);
                total += self.read_at_with_retries(file, id, off, slice)?;
            }
            return Ok(total);
        }

        // Multi-member contiguous run: one vectored read.
        let iovecs: Vec<libc::iovec> = run
            .members
            .iter()
            .map(|&(_, idx)| {
                let (ptr, len) = dst[idx];
                libc::iovec {
                    iov_base: ptr as *mut libc::c_void,
                    iov_len: len,
                }
            })
            .collect();
        let want = run.total_len as usize;
        // SAFETY: `iovecs` point into the caller's disjoint, live
        // destination buffers; `file` is a valid open fd; the run is
        // physically contiguous starting at `start_offset`.
        let n = unsafe {
            libc::preadv(
                file.as_raw_fd(),
                iovecs.as_ptr(),
                iovecs.len() as libc::c_int,
                run.start_offset as libc::off_t,
            )
        };
        if n >= 0 && n as usize == want {
            // Full coalesced read: credit each member's breaker.
            for &(id, _) in &run.members {
                self.note_read_success(id);
            }
            return Ok(want);
        }
        // Short or failed preadv — fall back to independent, fully
        // fault-tolerant per-expert reads so a single bad slot can't wedge
        // the whole run and the retry/breaker accounting is preserved.
        if n < 0 {
            let err = io::Error::last_os_error();
            tracing::warn!(
                error = %err,
                run_start = run.start_offset,
                members = run.members.len(),
                "coalesced preadv failed; falling back to per-expert reads"
            );
        } else {
            tracing::warn!(
                got = n,
                want,
                run_start = run.start_offset,
                members = run.members.len(),
                "coalesced preadv short read; falling back to per-expert reads"
            );
        }
        let mut total = 0usize;
        for &(id, idx) in &run.members {
            let (ptr, len) = dst[idx];
            // SAFETY: see the singleton branch above.
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            let off = self.packed_offset(id);
            total += self.read_at_with_retries(file, id, off, slice)?;
        }
        Ok(total)
    }

    /// Blob offset of a packed expert id. Panics only if called without a
    /// packed blob attached (an internal invariant of the packed read
    /// paths). For an id missing from the manifest — which the callers
    /// already rule out by resolving every entry up front in
    /// `read_experts_batch_packed` — it returns `u64::MAX` so the
    /// subsequent `pread` fails loudly instead of silently aliasing the
    /// first expert's bytes at offset `0`.
    fn packed_offset(&self, id: u32) -> u64 {
        self.packed
            .as_ref()
            .and_then(|p| p.entry(id))
            .map(|e| e.offset)
            .unwrap_or(u64::MAX)
    }
    /// of an expert's `gate_proj` and `up_proj` plus the full `down_proj`,
    /// packed into `buf` in the layout consumed by
    /// [`crate::inference::OwnedExpertWeights::from_bytes_partial`]:
    ///
    /// ```text
    ///   gate_packed [d_ff x M]  ||  up_packed [d_ff x M]  ||  down [d_model x d_ff]
    /// ```
    ///
    /// `M = col_indices.len()` and `dtype` selects the on-disk byte width
    /// (2 for f16, 4 for f32). `buf.len()` must be at least the packed
    /// blob size (`(2*d_ff*M + d_model*d_ff) * bytes_per_weight`).
    ///
    /// **Implementation note (energy):** the row-major on-disk layout
    /// stores all columns of every row contiguously, so a strict
    /// "read only M columns" path would still need to touch every row.
    /// Today this function reads the full expert file once and packs
    /// the requested columns into `buf` in-process; that gives the
    /// **compute / dequantise** energy saving (proportional to M/d_model)
    /// without (yet) the **SSD bandwidth** saving. Switching to a
    /// column-major on-disk layout — written by the offline extractor —
    /// is the follow-up that turns this into a true bandwidth reduction.
    /// The engine API and the `from_bytes_partial` consumer are stable
    /// across that change.
    #[allow(dead_code)]
    pub async fn read_expert_columns(
        &self,
        expert_id: u32,
        col_indices: &[usize],
        dtype: WeightDtype,
        d_model: usize,
        d_ff: usize,
        buf: &mut PooledBuffer,
    ) -> io::Result<usize> {
        let bpw = dtype.bytes_per_weight();
        let m = col_indices.len();
        for &c in col_indices {
            if c >= d_model {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("col index {c} out of range for d_model={d_model}"),
                ));
            }
        }
        let packed_bytes = (2 * d_ff * m + d_model * d_ff) * bpw;
        if buf.len() < packed_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "destination buffer too small for partial load: have {}, need {}",
                    buf.len(),
                    packed_bytes
                ),
            ));
        }

        // Stage the full file into a scratch Vec, then pack out the
        // requested columns. `block_in_place` keeps the runtime
        // responsive while pread runs.
        let file = self.fd_for(expert_id)?;
        let expert_size = self.cfg.expert_size;
        let mut scratch = vec![0u8; expert_size];
        tokio::task::block_in_place(|| file.read_at(&mut scratch, 0))?;

        let row_bytes = d_model * bpw;
        let gate_off = 0;
        let up_off = d_ff * row_bytes;
        let down_off = 2 * d_ff * row_bytes;

        let mut pos = 0usize;
        // gate_packed
        for i in 0..d_ff {
            let row_start = gate_off + i * row_bytes;
            for &c in col_indices {
                let src = row_start + c * bpw;
                buf.as_mut_slice()[pos..pos + bpw].copy_from_slice(&scratch[src..src + bpw]);
                pos += bpw;
            }
        }
        // up_packed
        for i in 0..d_ff {
            let row_start = up_off + i * row_bytes;
            for &c in col_indices {
                let src = row_start + c * bpw;
                buf.as_mut_slice()[pos..pos + bpw].copy_from_slice(&scratch[src..src + bpw]);
                pos += bpw;
            }
        }
        // down_proj copied verbatim
        let down_size = d_model * d_ff * bpw;
        buf.as_mut_slice()[pos..pos + down_size]
            .copy_from_slice(&scratch[down_off..down_off + down_size]);
        pos += down_size;

        Ok(pos)
    }
}

/// **Tier 2.** Build a packed blob + manifest from a source storage.
///
/// Reads each expert in `order` from `source` (which may be a plain
/// one-file-per-expert [`NvmeStorage`], honouring its path resolution and
/// fault-tolerant retries) and writes the payloads back-to-back into
/// `blob_path`, one uniform `expert_size` slot each. Emits the JSON
/// [`crate::packed_storage::PackedManifest`] to `manifest_path`.
///
/// `order` defines the **physical layout**: experts that the engine tends to
/// fetch together (a routing-frequency hot set, or a co-firing affinity
/// ordering) should be adjacent so the steady-state read path can coalesce
/// them into a single `preadv`. Returns the manifest it wrote.
pub async fn pack_experts(
    source: &NvmeStorage,
    pool: &crate::buffer_pool::BufferPool,
    order: &[u32],
    blob_path: &Path,
    manifest_path: &Path,
) -> io::Result<crate::packed_storage::PackedManifest> {
    use std::io::Write;
    let expert_size = source.cfg.expert_size;
    let block_align = source.cfg.block_align;
    let blob = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(blob_path)?;
    let mut writer = std::io::BufWriter::with_capacity(8 * 1024 * 1024, blob);
    // Reject duplicate ids: the manifest collapses entries by key while the
    // blob writes every slot, so a repeated id yields a malformed layout.
    let mut seen = std::collections::HashSet::with_capacity(order.len());
    for &id in order {
        if !seen.insert(id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("duplicate expert id {id} in packed order"),
            ));
        }
        let mut buf = pool.acquire().await;
        debug_assert_eq!(buf.len(), expert_size);
        source.read_expert(id, &mut buf).await?;
        writer.write_all(buf.as_slice())?;
    }
    writer.flush()?;
    writer
        .into_inner()
        .map_err(|e| e.into_error())?
        .sync_all()?;
    let manifest = crate::packed_storage::PackedManifest::uniform(
        order.to_vec(),
        expert_size as u64,
        block_align as u64,
    );
    manifest.write_to(manifest_path)?;
    Ok(manifest)
}

#[cfg(target_os = "linux")]
pub(crate) fn open_expert_file(path: &Path, direct: bool) -> io::Result<File> {
    let mut opts = OpenOptions::new();
    opts.read(true);
    if direct {
        opts.custom_flags(libc::O_DIRECT);
    }
    match opts.open(path) {
        Ok(f) => Ok(f),
        Err(e) if direct && e.raw_os_error() == Some(libc::EINVAL) => {
            // Filesystem doesn't support O_DIRECT (tmpfs, overlayfs, some
            // FUSE mounts). Tell the user to either move the data dir to a
            // real block device or disable direct I/O.
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "open({}) with O_DIRECT failed (EINVAL): the underlying \
                     filesystem does not support direct I/O. Re-run with \
                     --no-direct, or place the data directory on an ext4/xfs \
                     mount on a real NVMe device.",
                    path.display()
                ),
            ))
        }
        Err(e) => Err(e),
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn open_expert_file(path: &Path, _direct: bool) -> io::Result<File> {
    OpenOptions::new().read(true).open(path)
}

/// Generate `num_experts` deterministic test files in `dir` with f32
/// weights. See [`generate_synthetic_experts_with_dtype`] for the f16
/// variant.
pub fn generate_synthetic_experts(
    dir: &Path,
    num_experts: u32,
    expert_size: usize,
    d_model: usize,
    d_ff: usize,
) -> io::Result<()> {
    generate_synthetic_experts_with_dtype(
        dir,
        num_experts,
        expert_size,
        d_model,
        d_ff,
        WeightDtype::F32,
    )
}

/// Generate `num_experts` deterministic test files in `dir`. Each file
/// contains real `f32` *or* `f16` SwiGLU weights laid out as
/// `gate_proj || up_proj || down_proj` (row-major; see
/// [`crate::inference`]).
///
/// `weight_bytes` (= [`crate::inference::expert_weight_bytes_for`])
/// is the number of bytes the engine will actually consume. `expert_size`
/// is the size on disk; if it is larger than `weight_bytes` the trailing
/// region is zero padded so the file size stays a multiple of
/// `block_align` (an `O_DIRECT` requirement on Linux).
///
/// Weights are drawn from a small bounded uniform distribution
/// (`U(-scale, +scale)` with `scale ≈ 1 / sqrt(d_model)`) using a
/// per-expert deterministic xorshift, so the SwiGLU forward pass remains
/// numerically stable for any `d_model`/`d_ff` and runs are reproducible.
pub fn generate_synthetic_experts_with_dtype(
    dir: &Path,
    num_experts: u32,
    expert_size: usize,
    d_model: usize,
    d_ff: usize,
    dtype: WeightDtype,
) -> io::Result<()> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;

    let weight_bytes = crate::inference::expert_weight_bytes_for(d_model, d_ff, dtype);
    if weight_bytes > expert_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "expert_size {expert_size} too small for d_model={d_model} d_ff={d_ff} \
                 dtype={:?} (need at least {weight_bytes} bytes for the SwiGLU weights)",
                dtype
            ),
        ));
    }

    let scale = 1.0f32 / (d_model.max(1) as f32).sqrt();
    let pad_bytes = expert_size - weight_bytes;
    let zero_pad = vec![0u8; (1 << 20).min(pad_bytes.max(1))];
    let chunk_floats = 16 * 1024;

    if matches!(
        dtype,
        WeightDtype::Q5K | WeightDtype::Q6K | WeightDtype::Mixed
    ) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "synthetic expert generation does not support dtype {}",
                dtype.as_str()
            ),
        ));
    }

    for id in 0..num_experts {
        let path = dir.join(format!("expert_{id}.bin"));
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        let mut state: u64 =
            0x9E37_79B9_7F4A_7C15u64.wrapping_add((id as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));

        // For INT8, write the 12-byte per-tensor scale header first.
        // Synthetic weights are drawn from `U(-scale, +scale)`, so the
        // per-tensor max-abs is bounded by `scale` and the symmetric
        // quantizer divisor is `scale / 127.0`. Real INT8-quantised
        // weights would compute these from the actual tensor maxima
        // during conversion (see `python/quantize_int8.py`).
        if matches!(dtype, WeightDtype::Int8) {
            let q = scale / 127.0;
            let meta = crate::inference::Int8ExpertMeta {
                gate_scale: q,
                up_scale: q,
                down_scale: q,
            };
            f.write_all(&meta.to_bytes())?;
        }

        let mut floats_remaining = crate::inference::expert_weight_count(d_model, d_ff);
        // Q4K writes per-block (144 bytes for 256 weights). For other
        // dtypes we write per-weight. Keep the per-weight RNG fed
        // through `state` so synthetic experts remain deterministic
        // across dtype choices for tests / golden runs.
        if matches!(dtype, WeightDtype::Q4K) {
            use crate::inference::{Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS};
            let mut block_floats = vec![0.0f32; Q4K_BLOCK_ELEMS];
            let mut block_bytes = vec![0u8; Q4K_BLOCK_BYTES];
            while floats_remaining > 0 {
                let n = floats_remaining.min(Q4K_BLOCK_ELEMS);
                for slot in block_floats.iter_mut().take(n) {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = (state >> 40) as u32;
                    let unit = (u as f32) / ((1u32 << 23) as f32) - 1.0;
                    *slot = unit * scale;
                }
                // Pad the tail of the last block with zeros so its
                // dequant produces zeros for the unused tail slots.
                for slot in block_floats.iter_mut().skip(n) {
                    *slot = 0.0;
                }
                quantize_q4k_block_min_max(&block_floats, &mut block_bytes);
                f.write_all(&block_bytes)?;
                floats_remaining = floats_remaining.saturating_sub(n);
            }
        } else if matches!(dtype, WeightDtype::Q4_0 | WeightDtype::Q8_0) {
            // Q4_0 / Q8_0 write per-block, per-tensor (gate / up / down
            // independently rounded up to a block boundary).
            use crate::inference::{
                quantize_q4_0_block, quantize_q8_0_block, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS,
                Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS,
            };
            let (block_elems, block_bytes) = match dtype {
                WeightDtype::Q4_0 => (Q4_0_BLOCK_ELEMS, Q4_0_BLOCK_BYTES),
                WeightDtype::Q8_0 => (Q8_0_BLOCK_ELEMS, Q8_0_BLOCK_BYTES),
                _ => unreachable!(),
            };
            let mut block_floats = vec![0.0f32; block_elems];
            let mut block_bytes_buf = vec![0u8; block_bytes];
            let one = d_model.saturating_mul(d_ff);
            for _tensor in 0..3 {
                let mut t_remaining = one;
                while t_remaining > 0 {
                    let n = t_remaining.min(block_elems);
                    for slot in block_floats.iter_mut().take(n) {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        let u = (state >> 40) as u32;
                        let unit = (u as f32) / ((1u32 << 23) as f32) - 1.0;
                        *slot = unit * scale;
                    }
                    for slot in block_floats.iter_mut().skip(n) {
                        *slot = 0.0;
                    }
                    match dtype {
                        WeightDtype::Q4_0 => {
                            quantize_q4_0_block(&block_floats[..block_elems], &mut block_bytes_buf)
                        }
                        WeightDtype::Q8_0 => {
                            quantize_q8_0_block(&block_floats[..block_elems], &mut block_bytes_buf)
                        }
                        _ => unreachable!(),
                    }
                    f.write_all(&block_bytes_buf)?;
                    t_remaining = t_remaining.saturating_sub(n);
                }
            }
            // The per-weight loop below writes nothing because
            // floats_remaining was set assuming the per-weight format;
            // null it out so we don't double-write.
            floats_remaining = 0;
        } else if matches!(dtype, WeightDtype::MXFP4) {
            // MXFP4 writes three projections (gate, up, down) back to
            // back. Each projection is its packed E2M1 weight bytes
            // (`rows * ceil(cols/2)`, low-nibble first) immediately
            // followed by its E8M0 block scales (`rows * ceil(cols/32)`).
            // Synthetic weights use random nibbles and a fixed unit
            // block scale (E8M0 byte 127 = 2^0), so dequant reproduces
            // the canonical E2M1 magnitudes deterministically.
            use crate::inference::MXFP4_SCALE_BLOCK;
            for &(rows, cols) in &[(d_ff, d_model), (d_ff, d_model), (d_model, d_ff)] {
                let wbytes = rows.saturating_mul(cols.div_ceil(2));
                let mut buf = vec![0u8; wbytes];
                for b in buf.iter_mut() {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    *b = (state >> 40) as u8;
                }
                f.write_all(&buf)?;
                let sbytes = rows.saturating_mul(cols.div_ceil(MXFP4_SCALE_BLOCK));
                // E8M0 byte 127 => 2^0 = 1.0 (non-zero, unit scale).
                let sbuf = vec![127u8; sbytes];
                f.write_all(&sbuf)?;
            }
            floats_remaining = 0;
        } else {
            let bpw = dtype.bytes_per_weight();
            let mut buf = Vec::<u8>::with_capacity(chunk_floats * bpw);
            while floats_remaining > 0 {
                let n = floats_remaining.min(chunk_floats);
                buf.clear();
                for _ in 0..n {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = (state >> 40) as u32;
                    let unit = (u as f32) / ((1u32 << 23) as f32) - 1.0;
                    let v = unit * scale;
                    match dtype {
                        WeightDtype::F32 => buf.extend_from_slice(&v.to_le_bytes()),
                        WeightDtype::F16 => {
                            let h = half::f16::from_f32(v);
                            buf.extend_from_slice(&h.to_bits().to_le_bytes());
                        }
                        WeightDtype::BF16 => {
                            let h = half::bf16::from_f32(v);
                            buf.extend_from_slice(&h.to_bits().to_le_bytes());
                        }
                        WeightDtype::Int8 => {
                            // Per-tensor symmetric quant. With the synthetic
                            // distribution and `q = scale/127.0`, `v / q`
                            // is in `[-127, +127]` so no clamp loss occurs;
                            // we still clamp defensively for robustness.
                            let q = scale / 127.0;
                            let qv = (v / q).round().clamp(-127.0, 127.0) as i8;
                            buf.push(qv as u8);
                        }
                        WeightDtype::Q4K => unreachable!("Q4K handled above"),
                        WeightDtype::Q4_0 => unreachable!("Q4_0 handled above"),
                        WeightDtype::Q8_0 => unreachable!("Q8_0 handled above"),
                        WeightDtype::Q5K => unreachable!("Q5K rejected above"),
                        WeightDtype::Q6K => unreachable!("Q6K rejected above"),
                        WeightDtype::MXFP4 => unreachable!("MXFP4 handled above"),
                        WeightDtype::Mixed => unreachable!("mixed rejected above"),
                    }
                }
                f.write_all(&buf)?;
                floats_remaining -= n;
            }
        }

        let mut remaining_pad = pad_bytes;
        while remaining_pad > 0 {
            let n = remaining_pad.min(zero_pad.len());
            f.write_all(&zero_pad[..n])?;
            remaining_pad -= n;
        }
        f.flush()?;
    }
    Ok(())
}

/// Quantise a 256-element block to GGUF Q4_K_M layout, writing 144
/// bytes into `dst`. The encoder uses simple per-sub-block min/max
/// clipping followed by 4-bit linear quantisation:
///
/// * super-block range `(lo, hi) = (min(x), max(x))`;
/// * `d = (hi - lo) / 63 / 15`,  `dmin = -lo / 63`;
/// * each sub-block's `scale6` is the 6-bit value that minimises the
///   sub-block's quantisation error against `d` (here we use the
///   block-wide max of |x - lo| -> 63 mapping for simplicity, which
///   is what the reference `ggml_quantize_q4_K_reference` does for a
///   minimum-effort encoder when no statistics are available);
/// * each sub-block's `min6` is `0` (no per-sub-block offset beyond
///   the global `dmin`).
///
/// This is a faithful inverse of [`crate::inference::dequantize_q4k_block`]
/// for the synthetic-weight regime: every weight produced by the
/// generator is bounded in `[-scale, +scale]`, so the simple
/// per-block fitting suffices and no per-sub-block bias correction
/// is needed for tests / golden runs to round-trip cleanly. A
/// production encoder (e.g. `python/quantize_q4k.py`) would solve
/// the per-sub-block 2-D least-squares problem; we don't do that
/// here because the synthetic distribution is uniform.
fn quantize_q4k_block_min_max(src: &[f32], dst: &mut [u8]) {
    use crate::inference::{Q4K_BLOCK_BYTES, Q4K_BLOCK_ELEMS, Q4K_SUBBLOCKS, Q4K_SUBBLOCK_ELEMS};
    debug_assert_eq!(src.len(), Q4K_BLOCK_ELEMS);
    debug_assert_eq!(dst.len(), Q4K_BLOCK_BYTES);

    // Find the super-block range.
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in src {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        // Degenerate block: emit all zeros (which dequantises to 0).
        for b in dst.iter_mut() {
            *b = 0;
        }
        return;
    }
    // d * 63 * 15 = (hi - lo)  =>  d = (hi - lo) / 945.
    let denom = (hi - lo) / 945.0f32;
    let d_f16 = half::f16::from_f32(denom);
    // dmin * 63 = -lo  =>  dmin = -lo / 63.
    let dmin_f16 = half::f16::from_f32(-lo / 63.0);

    dst[0..2].copy_from_slice(&d_f16.to_bits().to_le_bytes());
    dst[2..4].copy_from_slice(&dmin_f16.to_bits().to_le_bytes());

    // Use a constant per-sub-block scale6 = 63 and min6 = 63 so the
    // dequant maps q4 in 0..15 to lo + (q4 / 15) * (hi - lo) — i.e.
    // the standard 4-bit linear quantiser scaled across the whole
    // super-block. (sub_scale=63 ensures q4=15 -> hi; min6=63 with
    // dmin = -lo/63 contributes -dmin * 63 = lo.)
    //
    // Pack the 8 (scale6, min6) = (63, 63) pairs.
    let pairs = [(63u8, 63u8); Q4K_SUBBLOCKS];
    let s = q4k_pack_scales_local(&pairs);
    dst[4..16].copy_from_slice(&s);

    // Quantise each weight: q4 = round((v - lo) / (hi - lo) * 15).
    // (Using the block-wide range, since every sub_scale equals 63
    // and dmin*min6 == lo.)
    let inv_range = 15.0f32 / (hi - lo);
    let qs = &mut dst[16..16 + 128];
    for j in 0..Q4K_SUBBLOCKS {
        let qs_off = j * (Q4K_SUBBLOCK_ELEMS / 2);
        for i in 0..Q4K_SUBBLOCK_ELEMS {
            let v = src[j * Q4K_SUBBLOCK_ELEMS + i];
            let q = ((v - lo) * inv_range).round().clamp(0.0, 15.0) as u8;
            let byte_idx = qs_off + (i >> 1);
            if i & 1 == 0 {
                qs[byte_idx] = (qs[byte_idx] & 0xF0) | (q & 0x0F);
            } else {
                qs[byte_idx] = (qs[byte_idx] & 0x0F) | ((q & 0x0F) << 4);
            }
        }
    }
}

/// Local copy of the inference module's q4k scale packer; kept here to
/// avoid making the inference helper `pub`. Mirrors the bit layout
/// described in [`crate::inference::dequantize_q4k_block`].
fn q4k_pack_scales_local(pairs: &[(u8, u8); 8]) -> [u8; 12] {
    let mut s = [0u8; 12];
    for j in 0..4 {
        s[j] = pairs[j].0 & 0x3F;
        s[j + 4] = pairs[j].1 & 0x3F;
    }
    for j in 4..8 {
        let (scale_j, min_j) = pairs[j];
        let scale_j = scale_j & 0x3F;
        let min_j = min_j & 0x3F;
        s[j + 4] = (scale_j & 0x0F) | ((min_j & 0x0F) << 4);
        s[j - 4] = (s[j - 4] & 0x3F) | (((scale_j >> 4) & 0x03) << 6);
        s[j] = (s[j] & 0x3F) | (((min_j >> 4) & 0x03) << 6);
    }
    s
}

// =====================================================================
// Cold-start expert manifest (Task 3 of the Industrial Upgrade).
// =====================================================================

/// One row of [`Manifest`]: everything we learned about a single
/// `expert_<id>.bin` file at startup that the steady-state path would
/// otherwise have to re-derive on every fetch.
///
/// In particular `payload_offset` is the byte offset where the *weight
/// payload* starts inside the file — `0` for legacy bare-payload
/// blobs, [`crate::tensor_header::UTH_BYTES`] padded to `block_align`
/// (typically 4096) when the file carries a Unified Tensor Header.
/// With this number cached, a fetch handler can skip the per-call
/// `TensorHeader::probe` over the head of every resident buffer and
/// jump directly at the payload bytes — the "Zero-Latency Seek"
/// guarantee the spec asks for.
#[derive(Debug, Clone)]
pub struct ManifestEntry {
    /// Filesystem path resolved by `NvmeStorage::expert_path` at scan
    /// time. Cached so subsequent file-open calls don't re-walk the
    /// striped-drive lookup or hit the multi-layer fallback again.
    pub path: PathBuf,
    /// `metadata().len()` of the file. Always a multiple of
    /// `block_align`; reads of size `expert_size` bounded against this
    /// value catch a truncated / partially-written conversion.
    pub file_size: u64,
    /// Byte offset where the weight payload begins. Equal to the
    /// page-padded UTH size when a header is present, `0` otherwise.
    pub payload_offset: usize,
    /// Number of payload bytes (= `file_size - payload_offset`,
    /// rounded *down* to the alignment block — defensive against a
    /// short tail page).
    pub payload_size: usize,
    /// Weight dtype declared by the header. `None` for legacy
    /// (bare-payload) files where the dtype must be supplied
    /// out-of-band by the engine config / `metadata.json`.
    pub dtype: Option<WeightDtype>,
    /// Parsed UTH, when present. Callers that need the AMX tile hints
    /// or quant-scale-offset metadata go through this field.
    pub header: Option<TensorHeader>,
}

/// Cold-start index over every `expert_<id>.bin` in a data directory.
///
/// Built once at engine boot by [`Manifest::scan`] and then kept
/// resident (`Arc<Manifest>`) so that the inference loop's
/// `read_expert(id)` call resolves to a `(path, payload_offset,
/// payload_size, dtype)` tuple in `O(1)` time without touching the
/// filesystem header again. This is the "cold-start manifest"
/// requirement of the Industrial Upgrade spec — it eliminates the
/// per-fetch UTH probe and gives `ExpertLoader` a zero-latency seek
/// into the weight bytes.
///
/// Memory cost is tiny: one [`ManifestEntry`] per expert file,
/// dominated by the `PathBuf` (a few hundred bytes for a Mixtral-scale
/// 8 × 32 expert model is on the order of 64 KiB).
#[derive(Debug, Clone, Default)]
pub struct Manifest {
    entries: HashMap<u32, ManifestEntry>,
    block_align: usize,
}

impl Manifest {
    /// Walk `dirs` and probe the head of every `expert_<id>.bin` for
    /// `id` in `ids`, returning a populated [`Manifest`].
    ///
    /// For each id we:
    /// * resolve the path via the same single-namespace / multi-layer
    ///   logic [`NvmeStorage::expert_path`] uses (callers that want
    ///   striped multi-drive layouts pass the full `striped_paths`
    ///   list and `num_experts_per_layer`),
    /// * `pread` the first `block_align` bytes (cheap — at most one
    ///   page per file),
    /// * call [`TensorHeader::probe`] over those bytes.
    ///
    /// Missing files are tolerated: they're recorded as a `None`
    /// entry-set member and the engine's existing "fall back to
    /// synthetic init" code path handles the gap.  Files that exist
    /// but fail the probe are recorded with `header = None` and
    /// `dtype = None` (legacy bare-payload layout).
    pub fn scan(
        dirs: &[PathBuf],
        ids: impl IntoIterator<Item = u32>,
        block_align: usize,
        num_experts_per_layer: Option<u32>,
    ) -> io::Result<Self> {
        assert!(
            block_align.is_power_of_two() && block_align > 0,
            "Manifest::scan: block_align must be a power of two"
        );
        let dirs: Vec<PathBuf> = if dirs.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Manifest::scan requires at least one directory",
            ));
        } else {
            dirs.to_vec()
        };
        let n_drives = dirs.len();

        let mut entries = HashMap::new();
        let mut head = vec![0u8; block_align];
        for id in ids {
            let dir = &dirs[(id as usize) % n_drives];
            let primary = dir.join(format!("expert_{id}.bin"));
            let path = if std::fs::metadata(&primary).is_ok() {
                primary
            } else if let Some(n) = num_experts_per_layer {
                if n > 0 {
                    dir.join(format!("expert_{}_{}.bin", id / n, id % n))
                } else {
                    primary
                }
            } else {
                primary
            };

            let meta = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let file_size = meta.len();
            // Probe the first block of the file for a UTH. We use
            // `read_at` (positional) so the cached fd table — which
            // doesn't exist yet at scan time — is not perturbed; this
            // is the only sync filesystem hit the Manifest performs.
            let probed = match File::open(&path) {
                Ok(f) => match f.read_at(&mut head, 0) {
                    Ok(n) => {
                        if n >= UTH_BYTES {
                            TensorHeader::probe(&head[..n.min(block_align)])
                        } else {
                            None
                        }
                    }
                    Err(err) => {
                        tracing::warn!(
                            "failed to read header probe from {}: {}",
                            path.display(),
                            err
                        );
                        None
                    }
                },
                Err(err) => {
                    tracing::warn!(
                        "failed to open {} for header probe: {}",
                        path.display(),
                        err
                    );
                    None
                }
            };

            let (payload_offset, dtype) = match probed.as_ref() {
                Some(h) => (block_align, Some(h.dtype.to_weight())),
                None => (0usize, None),
            };
            let payload_size =
                (file_size as usize).saturating_sub(payload_offset) & !(block_align - 1);

            entries.insert(
                id,
                ManifestEntry {
                    path,
                    file_size,
                    payload_offset,
                    payload_size,
                    dtype,
                    header: probed,
                },
            );
        }

        Ok(Self {
            entries,
            block_align,
        })
    }

    /// Look up a manifest entry by expert id. `None` if the file
    /// wasn't present at scan time.
    #[inline]
    pub fn lookup(&self, id: u32) -> Option<&ManifestEntry> {
        self.entries.get(&id)
    }

    /// Number of indexed experts.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Convenience: zero-latency lookup of the byte offset where the
    /// weight payload starts inside `expert_<id>.bin`. Returns `0`
    /// when the file has no header and `None` when the file wasn't
    /// indexed.
    #[inline]
    pub fn payload_offset(&self, id: u32) -> Option<usize> {
        self.entries.get(&id).map(|e| e.payload_offset)
    }

    /// Convenience: dtype declared by the file's UTH. Returns `None`
    /// in **two** distinct cases:
    ///
    /// * `id` was not seen at scan time (no `expert_<id>.bin` in any
    ///   of the manifest's data dirs), or
    /// * the file *was* indexed but has no UTH (legacy bare-payload
    ///   layout written before the `--no-uth` flag was introduced).
    ///
    /// Callers that need to distinguish these cases should use
    /// [`Manifest::lookup`] (which returns `None` only in the first
    /// case; an indexed legacy file still yields `Some(entry)` with
    /// `entry.dtype == None`).
    #[inline]
    pub fn dtype(&self, id: u32) -> Option<WeightDtype> {
        self.entries.get(&id).and_then(|e| e.dtype)
    }

    /// Verify that every expert recorded in the manifest carries the
    /// **same** on-disk dtype.
    ///
    /// Returns:
    ///
    /// * `Ok(Some(dtype))` — every manifest entry that advertised a
    ///   UTH dtype agreed on `dtype`; entries with `dtype = None`
    ///   (legacy bare-payload files) are ignored.
    /// * `Ok(None)` — the manifest is empty, or every entry is a
    ///   legacy bare-payload file with no UTH (so no dtype could be
    ///   read). The engine falls back to the config's declared dtype.
    /// * `Err(IncompatibleExpertTypes { found })` — at least two
    ///   entries declared *different* dtypes. The engine refuses to
    ///   boot rather than silently driving heterogeneous experts
    ///   through a single dispatch arm.
    ///
    /// This is the runtime cross-check that backs the
    /// `[EngineError::IncompatibleExpertTypes]` startup error: a
    /// compute kernel built for one quant scheme produces silently
    /// wrong activations against another, so we surface the
    /// inconsistency before the first `pread`.
    pub fn verify_uniform_dtype(&self) -> Result<Option<WeightDtype>, IncompatibleExpertTypes> {
        let mut chosen: Option<WeightDtype> = None;
        let mut all_seen: Vec<WeightDtype> = Vec::new();
        for entry in self.entries.values() {
            let Some(d) = entry.dtype else { continue };
            if !all_seen.contains(&d) {
                all_seen.push(d);
            }
            match chosen {
                None => chosen = Some(d),
                Some(c) if c == d => {}
                Some(_) => {
                    return Err(IncompatibleExpertTypes { found: all_seen });
                }
            }
        }
        Ok(chosen)
    }
}

/// Returned by [`Manifest::verify_uniform_dtype`] when the manifest
/// indexed at least two experts whose on-disk Unified Tensor Header
/// declares **different** weight dtypes. Surfaced upstream by the
/// engine as `EngineError::IncompatibleExpertTypes`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncompatibleExpertTypes {
    /// The set of distinct dtypes the manifest observed, in the
    /// order they were first encountered. Always has at least two
    /// entries (otherwise the verifier would have returned `Ok`).
    pub found: Vec<WeightDtype>,
}

impl std::fmt::Display for IncompatibleExpertTypes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "manifest indexed experts with incompatible weight dtypes: {:?}",
            self.found
        )
    }
}

impl std::error::Error for IncompatibleExpertTypes {}

// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;

    #[test]
    fn manifest_verify_uniform_dtype_agrees() {
        let mut m = Manifest::default();
        m.block_align = 4096;
        m.entries.insert(
            0,
            ManifestEntry {
                path: PathBuf::from("/dev/null"),
                file_size: 0,
                payload_offset: 0,
                payload_size: 0,
                dtype: Some(WeightDtype::Q8_0),
                header: None,
            },
        );
        m.entries.insert(
            1,
            ManifestEntry {
                path: PathBuf::from("/dev/null"),
                file_size: 0,
                payload_offset: 0,
                payload_size: 0,
                dtype: Some(WeightDtype::Q8_0),
                header: None,
            },
        );
        assert_eq!(m.verify_uniform_dtype(), Ok(Some(WeightDtype::Q8_0)));
    }

    #[test]
    fn manifest_verify_uniform_dtype_rejects_mismatch() {
        let mut m = Manifest::default();
        m.block_align = 4096;
        m.entries.insert(
            0,
            ManifestEntry {
                path: PathBuf::from("/dev/null"),
                file_size: 0,
                payload_offset: 0,
                payload_size: 0,
                dtype: Some(WeightDtype::Q4_0),
                header: None,
            },
        );
        m.entries.insert(
            1,
            ManifestEntry {
                path: PathBuf::from("/dev/null"),
                file_size: 0,
                payload_offset: 0,
                payload_size: 0,
                dtype: Some(WeightDtype::Q8_0),
                header: None,
            },
        );
        let err = m.verify_uniform_dtype().unwrap_err();
        // Both dtypes are surfaced — the engine logs them on refusal.
        assert!(err.found.contains(&WeightDtype::Q4_0));
        assert!(err.found.contains(&WeightDtype::Q8_0));
    }

    #[test]
    fn manifest_verify_uniform_dtype_ignores_legacy_entries() {
        let mut m = Manifest::default();
        m.block_align = 4096;
        m.entries.insert(
            0,
            ManifestEntry {
                path: PathBuf::from("/dev/null"),
                file_size: 0,
                payload_offset: 0,
                payload_size: 0,
                dtype: None,
                header: None,
            },
        );
        // Pure-legacy manifest → no dtype to verify against, so the
        // engine falls back to the config-declared dtype.
        assert_eq!(m.verify_uniform_dtype(), Ok(None));
    }

    /// Internal helper: a unique tempdir under `std::env::temp_dir()`.
    fn tempdir(tag: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        path.push(format!("mer-io-test-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    /// Striping smoke test: when `NvmeStorage::striped` is constructed
    /// with N directories, `expert_path(id)` selects directory
    /// `id % N`, and `read_expert` returns the same bytes the file
    /// behind that path contains.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn striped_storage_shards_by_id_modulo_drives() {
        let d0 = tempdir("stripe-a");
        let d1 = tempdir("stripe-b");
        let num_experts = 4u32;
        let d_model = 4usize;
        let d_ff = 8usize;
        let block = 4096usize;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let expert_size = weight_bytes.div_ceil(block) * block;
        // Even ids -> d0; odd ids -> d1.
        for id in 0..num_experts {
            let dir = if id % 2 == 0 { &d0 } else { &d1 };
            let path = dir.join(format!("expert_{id}.bin"));
            // Distinct fill byte per expert so reads can be verified.
            let mut blob = vec![0u8; expert_size];
            for b in blob.iter_mut() {
                *b = (id as u8).wrapping_add(0x10);
            }
            std::fs::write(&path, &blob).unwrap();
        }
        let storage = NvmeStorage::striped(
            StorageConfig {
                base_path: d0.clone(),
                expert_size,
                block_align: block,
                use_direct_io: false,
                num_experts_per_layer: None,
            },
            vec![d0.clone(), d1.clone()],
        )
        .unwrap();
        assert_eq!(storage.num_drives(), 2);
        // expert_0 / expert_2 must resolve to d0; expert_1 / expert_3 to d1.
        assert_eq!(storage.expert_path(0), d0.join("expert_0.bin"));
        assert_eq!(storage.expert_path(1), d1.join("expert_1.bin"));
        assert_eq!(storage.expert_path(2), d0.join("expert_2.bin"));
        assert_eq!(storage.expert_path(3), d1.join("expert_3.bin"));

        let pool = BufferPool::new(num_experts as usize, expert_size, block);
        for id in 0..num_experts {
            let mut buf = pool.acquire().await;
            storage.read_expert(id, &mut buf).await.unwrap();
            assert_eq!(buf.as_slice()[0], (id as u8).wrapping_add(0x10));
        }
        let _ = std::fs::remove_dir_all(&d0);
        let _ = std::fs::remove_dir_all(&d1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_experts_batch_returns_same_bytes_as_single_reads() {
        let dir = tempdir("batch");
        let num_experts = 4u32;
        let d_model = 8usize;
        let d_ff = 16usize;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block = 4096usize;
        let expert_size = ((weight_bytes + block - 1) / block) * block;
        generate_synthetic_experts(&dir, num_experts, expert_size, d_model, d_ff).unwrap();
        let storage = NvmeStorage::new(StorageConfig {
            base_path: dir.clone(),
            expert_size,
            block_align: block,
            use_direct_io: false,
            num_experts_per_layer: None,
        })
        .unwrap();

        let pool = BufferPool::new(num_experts as usize * 2 + 2, expert_size, block);

        // Reference: read each expert one-by-one.
        let mut ref_bufs: Vec<Vec<u8>> = Vec::with_capacity(num_experts as usize);
        for id in 0..num_experts {
            let mut b = pool.acquire().await;
            storage.read_expert(id, &mut b).await.unwrap();
            ref_bufs.push(b.as_slice().to_vec());
        }

        // Batched read into fresh buffers.
        let mut bufs: Vec<_> = Vec::with_capacity(num_experts as usize);
        for _ in 0..num_experts {
            bufs.push(pool.acquire().await);
        }
        let ids: Vec<u32> = (0..num_experts).collect();
        let mut buf_refs: Vec<&mut crate::buffer_pool::PooledBuffer> = bufs.iter_mut().collect();
        let total = storage
            .read_experts_batch(&ids, &mut buf_refs)
            .await
            .unwrap();
        assert_eq!(total, expert_size * num_experts as usize);
        for (i, b) in bufs.iter().enumerate() {
            assert_eq!(
                b.as_slice(),
                ref_bufs[i].as_slice(),
                "mismatch on expert {i}"
            );
        }

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// **Tier 2.** A packed blob must return byte-identical payloads to the
    /// one-file-per-expert path for both single (`read_expert`) and
    /// coalesced (`read_experts_batch`) reads — including a shuffled
    /// request that exercises the offset sort, a contiguous run serviced
    /// by one `preadv`, and an isolated slot serviced by the singleton
    /// fallback.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn packed_blob_reads_match_per_file_including_coalesce() {
        let dir = tempdir("packed");
        let num_experts = 6u32;
        let d_model = 8usize;
        let d_ff = 16usize;
        let block = 4096usize;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let expert_size = weight_bytes.div_ceil(block) * block;
        generate_synthetic_experts(&dir, num_experts, expert_size, d_model, d_ff).unwrap();

        let cfg = StorageConfig {
            base_path: dir.clone(),
            expert_size,
            block_align: block,
            use_direct_io: false,
            num_experts_per_layer: None,
        };
        let source = NvmeStorage::new(cfg.clone()).unwrap();
        let pool = BufferPool::new(num_experts as usize * 2 + 4, expert_size, block);

        // Reference bytes via the per-file path.
        let mut ref_bufs: Vec<Vec<u8>> = Vec::with_capacity(num_experts as usize);
        for id in 0..num_experts {
            let mut b = pool.acquire().await;
            source.read_expert(id, &mut b).await.unwrap();
            ref_bufs.push(b.as_slice().to_vec());
        }

        // Pack experts in id order, then open a packed storage over the blob.
        let order: Vec<u32> = (0..num_experts).collect();
        let blob_path = dir.join("experts.blob");
        let manifest_path = dir.join("experts.manifest.json");
        let manifest = pack_experts(&source, &pool, &order, &blob_path, &manifest_path)
            .await
            .unwrap();
        assert_eq!(manifest.len(), num_experts as usize);
        assert_eq!(manifest.entries[&3].offset, 3 * expert_size as u64);

        let blob =
            crate::packed_storage::PackedBlob::open(&blob_path, &manifest_path, false).unwrap();
        let packed = NvmeStorage::new(cfg)
            .unwrap()
            .with_packed_blob(Arc::new(blob));
        assert!(packed.is_packed());

        // Single packed reads match the per-file bytes.
        for id in 0..num_experts {
            let mut b = pool.acquire().await;
            packed.read_expert(id, &mut b).await.unwrap();
            assert_eq!(
                b.as_slice(),
                ref_bufs[id as usize].as_slice(),
                "single packed read mismatch on expert {id}"
            );
        }

        // Coalesced batch: {0,1,2} is a contiguous run (one preadv) and {5}
        // is isolated (singleton fallback). The request is shuffled so the
        // offset sort is exercised; results must land in request order.
        let ids = vec![2u32, 0, 1, 5];
        let mut bufs: Vec<_> = Vec::with_capacity(ids.len());
        for _ in &ids {
            bufs.push(pool.acquire().await);
        }
        let mut buf_refs: Vec<&mut crate::buffer_pool::PooledBuffer> = bufs.iter_mut().collect();
        let total = packed
            .read_experts_batch(&ids, &mut buf_refs)
            .await
            .unwrap();
        assert_eq!(total, expert_size * ids.len());
        for (slot, &id) in ids.iter().enumerate() {
            assert_eq!(
                bufs[slot].as_slice(),
                ref_bufs[id as usize].as_slice(),
                "coalesced read mismatch: expert {id} in request slot {slot}"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// descriptors resident, evicting the least-recently-used when a
    /// new expert is opened, and (b) still return correct bytes for an
    /// expert whose fd was evicted (it is simply re-opened on demand).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bounded_fd_cache_evicts_lru_and_rereads() {
        let dir = tempdir("fdcache");
        let num_experts = 8u32;
        let d_model = 4usize;
        let d_ff = 8usize;
        let block = 4096usize;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let expert_size = weight_bytes.div_ceil(block) * block;
        for id in 0..num_experts {
            let path = dir.join(format!("expert_{id}.bin"));
            let mut blob = vec![0u8; expert_size];
            for b in blob.iter_mut() {
                *b = (id as u8).wrapping_add(0x20);
            }
            std::fs::write(&path, &blob).unwrap();
        }
        // Cap the fd cache to 2 — far below the 8-expert working set.
        let storage = NvmeStorage::new(StorageConfig {
            base_path: dir.clone(),
            expert_size,
            block_align: block,
            use_direct_io: false,
            num_experts_per_layer: None,
        })
        .unwrap()
        .with_max_open_files(2);
        assert_eq!(storage.max_open_files(), 2);

        let pool = BufferPool::new(2, expert_size, block);
        // Read every expert; each read is correct even though most of
        // them were evicted from the 2-slot fd cache before being read
        // again.
        for round in 0..2 {
            for id in 0..num_experts {
                let mut buf = pool.acquire().await;
                storage.read_expert(id, &mut buf).await.unwrap();
                assert_eq!(
                    buf.as_slice()[0],
                    (id as u8).wrapping_add(0x20),
                    "wrong bytes for expert {id} on round {round}"
                );
                // The cache never exceeds its bound.
                assert!(storage.files.lock().len() <= 2);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Part A2 root-cause guard: `validate_expert_file_layout` rejects
    /// an `expert_size` that disagrees with the on-disk file size, with
    /// a targeted diagnosis when the file carries a page-padded UTH and
    /// `expert_size` is exactly the bare payload size (the historical
    /// one-page-short truncation), and accepts a matching layout.
    #[test]
    fn validate_expert_file_layout_detects_expert_size_uth_mismatch() {
        let dir = tempdir("layoutval");
        let block = 4096usize;
        let payload = 2 * block;
        // Write expert_0.bin = page-padded UTH prefix + payload, the
        // converter's emit_uth layout.
        let mut blob = Vec::with_capacity(block + payload);
        crate::tensor_header::TensorHeader::for_swiglu_expert(
            crate::inference::WeightDtype::Q4_0,
            64,
            128,
        )
        .write_padded(block, &mut blob);
        blob.resize(block + payload, 0xAB);
        std::fs::write(dir.join("expert_0.bin"), &blob).unwrap();

        let mk = |expert_size: usize| {
            NvmeStorage::new(StorageConfig {
                base_path: dir.clone(),
                expert_size,
                block_align: block,
                use_direct_io: false,
                num_experts_per_layer: None,
            })
            .unwrap()
        };

        // Misconfigured: expert_size excludes the UTH page → precise error.
        let err = mk(payload)
            .validate_expert_file_layout(0..1)
            .expect_err("payload-only expert_size must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("Unified Tensor Header"),
            "error must diagnose the UTH root cause: {msg}"
        );

        // Correct: expert_size covers header page + payload → Ok.
        mk(block + payload)
            .validate_expert_file_layout(0..1)
            .expect("full file size must validate");

        // Corrupt header magic: probe fails, generic mismatch error
        // (no UTH diagnosis) is still surfaced for a size mismatch.
        let mut corrupt = blob.clone();
        corrupt[0..4].copy_from_slice(b"XXXX");
        std::fs::write(dir.join("expert_0.bin"), &corrupt).unwrap();
        let err = mk(payload)
            .validate_expert_file_layout(0..1)
            .expect_err("size mismatch must still be rejected with a corrupt header");
        assert!(err.to_string().contains("model.expert_size"));

        // Missing files are skipped (sparse layouts are legal).
        mk(payload)
            .validate_expert_file_layout(5..6)
            .expect("absent sample files are skipped");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn multi_layer_naming_fallback_resolves_layered_files() {
        // The multi-layer HF extractor writes `expert_<layer>_<id>.bin`;
        // the storage layer must transparently resolve a global expert
        // id to the corresponding `(layer, local)` file when the
        // legacy single-namespace file is missing.
        let dir = tempdir("multilayer");
        let d_model = 4usize;
        let d_ff = 8usize;
        let block = 4096usize;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let expert_size = ((weight_bytes + block - 1) / block) * block;
        // 2 layers × 3 experts. Write only the layered names — the
        // legacy `expert_<global>.bin` files do *not* exist.
        let n = 3u32;
        for layer in 0..2u32 {
            for local in 0..n {
                let path = dir.join(format!("expert_{layer}_{local}.bin"));
                let mut blob = vec![0u8; expert_size];
                for b in blob.iter_mut() {
                    *b = ((layer * n + local) as u8).wrapping_add(0x40);
                }
                std::fs::write(&path, &blob).unwrap();
            }
        }
        let storage = NvmeStorage::new(StorageConfig {
            base_path: dir.clone(),
            expert_size,
            block_align: block,
            use_direct_io: false,
            num_experts_per_layer: Some(n),
        })
        .unwrap();
        // Global id → (id/n, id%n).
        assert_eq!(storage.expert_path(0), dir.join("expert_0_0.bin"));
        assert_eq!(storage.expert_path(2), dir.join("expert_0_2.bin"));
        assert_eq!(storage.expert_path(3), dir.join("expert_1_0.bin"));
        assert_eq!(storage.expert_path(5), dir.join("expert_1_2.bin"));
        // And the bytes round-trip through `read_expert`, which is
        // what the engine's miss path actually calls.
        let pool = BufferPool::new(2, expert_size, block);
        let mut buf = pool.acquire().await;
        let bytes = storage.read_expert(4, &mut buf).await.unwrap();
        assert_eq!(bytes, expert_size);
        // Layer=1, local=1 → fill byte = (1*3 + 1) + 0x40 = 0x44.
        assert_eq!(buf.as_slice()[0], 0x44);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `Manifest::scan` indexes every file under the directory, parses
    /// the UTH where present, and exposes the payload offset / dtype
    /// in `O(1)`. Files without a UTH are still indexed (legacy
    /// bare-payload layout) but report `header = None` and
    /// `payload_offset = 0`.
    #[test]
    fn manifest_scan_indexes_uth_and_legacy_files() {
        let dir = tempdir("manifest");
        let block = 4096usize;
        let d_model = 4usize;
        let d_ff = 8usize;
        let payload = crate::inference::expert_weight_bytes(d_model, d_ff);
        let payload_pad = payload.div_ceil(block) * block;

        // expert_0.bin: legacy bare payload, no UTH.
        std::fs::write(dir.join("expert_0.bin"), vec![0xAAu8; payload_pad]).unwrap();
        // expert_1.bin: F32 SwiGLU header + page-padded payload.
        let mut blob1 = Vec::with_capacity(block + payload_pad);
        TensorHeader::for_swiglu_expert(WeightDtype::F32, d_model, d_ff)
            .write_padded(block, &mut blob1);
        blob1.resize(block + payload_pad, 0xBB);
        std::fs::write(dir.join("expert_1.bin"), &blob1).unwrap();

        let m = Manifest::scan(
            &[dir.clone()],
            [0u32, 1u32, 7u32], // 7 doesn't exist → silently skipped
            block,
            None,
        )
        .expect("scan");
        assert_eq!(m.len(), 2, "missing files are tolerated");
        assert!(m.lookup(7).is_none());

        let e0 = m.lookup(0).expect("expert_0 indexed");
        assert_eq!(e0.payload_offset, 0, "no header → offset 0");
        assert!(e0.header.is_none());
        assert!(e0.dtype.is_none());
        assert_eq!(e0.file_size as usize, payload_pad);

        let e1 = m.lookup(1).expect("expert_1 indexed");
        assert_eq!(e1.payload_offset, block, "UTH page-padded to block");
        assert_eq!(e1.dtype, Some(WeightDtype::F32));
        let h = e1.header.as_ref().expect("UTH parsed");
        assert_eq!(h.shape[0] as usize, d_ff);
        assert_eq!(m.payload_offset(1), Some(block));
        assert_eq!(m.dtype(1), Some(WeightDtype::F32));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// When a `Manifest` is attached, `expert_path` short-circuits
    /// the multi-namespace fallback and returns the path the
    /// manifest already cached.
    #[test]
    fn nvme_storage_with_manifest_short_circuits_path_resolution() {
        let dir = tempdir("storage-manifest");
        let block = 4096usize;
        let d_model = 4usize;
        let d_ff = 8usize;
        let payload = crate::inference::expert_weight_bytes(d_model, d_ff);
        let expert_size = payload.div_ceil(block) * block;
        std::fs::write(dir.join("expert_0.bin"), vec![0u8; expert_size]).unwrap();
        std::fs::write(dir.join("expert_1.bin"), vec![0u8; expert_size]).unwrap();

        let manifest = Arc::new(Manifest::scan(&[dir.clone()], [0u32, 1u32], block, None).unwrap());
        let storage = NvmeStorage::new(StorageConfig {
            base_path: dir.clone(),
            expert_size,
            block_align: block,
            use_direct_io: false,
            num_experts_per_layer: None,
        })
        .unwrap()
        .with_manifest(manifest.clone());

        assert!(storage.manifest().is_some());
        assert_eq!(storage.expert_path(0), dir.join("expert_0.bin"));
        // Unindexed id → falls through to the legacy resolution.
        assert_eq!(storage.expert_path(99), dir.join("expert_99.bin"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Per-expert circuit breaker trips after
    /// [`STORAGE_BREAKER_THRESHOLD`] consecutive failures, and resets
    /// on the first success — gist Task 3 ("Hardened" MER).
    #[test]
    fn breaker_trips_and_resets() {
        let dir = tempdir("breaker");
        let block = 4096usize;
        let expert_size = block;
        // Don't write any expert files — every read will fail.
        let storage = NvmeStorage::new(StorageConfig {
            base_path: dir.clone(),
            expert_size,
            block_align: block,
            use_direct_io: false,
            num_experts_per_layer: None,
        })
        .unwrap();

        // Initially healthy.
        assert!(!storage.is_expert_unavailable(0));
        // Drive the breaker over the threshold.
        for _ in 0..STORAGE_BREAKER_THRESHOLD {
            storage.note_read_failure(0);
        }
        assert!(storage.is_expert_unavailable(0));
        // Reset on the first success.
        storage.note_read_success(0);
        assert!(!storage.is_expert_unavailable(0));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Gist Phase 3 — per-drive circuit breaker.
    ///
    /// Failures *spread across multiple experts on the same drive*
    /// must trip the drive-level breaker even though no single
    /// expert has hit its own threshold. Once tripped, any read
    /// against that drive short-circuits to
    /// `HardwareFailure::DriveUnavailable` without ever touching
    /// the device — that's the "stop attempting to route to that
    /// specific drive" behaviour from the gist.
    #[test]
    fn drive_breaker_trips_after_three_consecutive_failures() {
        let dir = tempdir("drive-breaker");
        let block = 4096usize;
        let expert_size = block;
        let storage = NvmeStorage::new(StorageConfig {
            base_path: dir.clone(),
            expert_size,
            block_align: block,
            use_direct_io: false,
            num_experts_per_layer: None,
        })
        .unwrap();

        // Three failures spread across distinct expert ids: the
        // per-expert breakers each only see one failure (well below
        // their `STORAGE_BREAKER_THRESHOLD`), but the drive-level
        // counter sums them and trips at 3.
        assert!(!storage.is_drive_unavailable(0));
        assert!(!storage.is_drive_unavailable(1));
        assert!(!storage.is_drive_unavailable(2));
        storage.note_read_failure(0);
        storage.note_read_failure(1);
        storage.note_read_failure(2);
        // All three experts now route through the same (single)
        // drive, so any of them reports it unavailable.
        assert!(storage.is_drive_unavailable(0));
        assert!(storage.is_drive_unavailable(7));
        // Per-expert breakers are *not* tripped — only one failure each.
        assert!(!storage.is_expert_unavailable(0));
        assert!(!storage.is_expert_unavailable(1));
        assert!(!storage.is_expert_unavailable(2));

        // A successful read on **any** expert on this drive clears
        // the drive-level breaker.
        storage.note_read_success(0);
        assert!(!storage.is_drive_unavailable(0));
        assert!(!storage.is_drive_unavailable(1));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// In a striped layout each drive maintains an independent
    /// breaker. A failed shard must not take its siblings out of
    /// rotation.
    #[test]
    fn drive_breaker_is_independent_per_shard() {
        let block = 4096usize;
        let expert_size = block;
        let dir_a = tempdir("drive-striped-a");
        let dir_b = tempdir("drive-striped-b");
        let storage = NvmeStorage::striped(
            StorageConfig {
                base_path: dir_a.clone(),
                expert_size,
                block_align: block,
                use_direct_io: false,
                num_experts_per_layer: None,
            },
            vec![dir_a.clone(), dir_b.clone()],
        )
        .unwrap();

        // Drive 0 serves even ids, drive 1 serves odd ids. Three
        // failures on even ids trip drive 0 only.
        storage.note_read_failure(0);
        storage.note_read_failure(2);
        storage.note_read_failure(4);
        assert!(storage.is_drive_unavailable(0));
        assert!(storage.is_drive_unavailable(8));
        // Odd ids (drive 1) are still healthy.
        assert!(!storage.is_drive_unavailable(1));
        assert!(!storage.is_drive_unavailable(3));

        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    /// Gist Phase 3 — half-open recovery.
    ///
    /// Without a probe gate, a tripped breaker would short-circuit
    /// every read forever, so `note_read_success` is unreachable and
    /// the documented "first successful read clears the breaker"
    /// semantics is unimplementable. This test exercises the
    /// half-open path end-to-end:
    ///
    /// 1. Trip both breakers via `note_read_failure`.
    /// 2. Confirm an immediate `read_expert` returns
    ///    `DriveUnavailable` (probe gate is closed for the first
    ///    `STORAGE_BREAKER_PROBE_INTERVAL`).
    /// 3. Backdate the trip timestamps to simulate elapsed probe
    ///    interval (so the test doesn't sleep for 500 ms).
    /// 4. Confirm the next `read_expert` succeeds and clears both
    ///    breakers.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn breaker_half_open_probe_recovers_after_interval() {
        let dir = tempdir("breaker-probe");
        let d_model = 4usize;
        let d_ff = 8usize;
        let block = 4096usize;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let expert_size = ((weight_bytes + block - 1) / block) * block;
        // One real expert file so the underlying `pread` succeeds.
        generate_synthetic_experts(&dir, 1, expert_size, d_model, d_ff).unwrap();
        let storage = NvmeStorage::new(StorageConfig {
            base_path: dir.clone(),
            expert_size,
            block_align: block,
            use_direct_io: false,
            num_experts_per_layer: None,
        })
        .unwrap();

        // Trip both breakers. We synthesise 3 logical failures so the
        // per-expert breaker also trips (per-drive trips at 3 as well).
        for _ in 0..STORAGE_BREAKER_THRESHOLD {
            storage.note_read_failure(0);
        }
        assert!(storage.is_expert_unavailable(0));
        assert!(storage.is_drive_unavailable(0));

        // Step 2: immediate read short-circuits — probe gate is closed.
        let pool = BufferPool::new(2, expert_size, block);
        let mut buf = pool.acquire().await;
        let err = storage.read_expert(0, &mut buf).await.unwrap_err();
        let inner = err
            .into_inner()
            .and_then(|e| e.downcast::<HardwareFailure>().ok())
            .expect("expected a structured HardwareFailure");
        assert!(
            matches!(*inner, HardwareFailure::DriveUnavailable { .. }),
            "expected DriveUnavailable while probe gate is closed, got {:?}",
            inner
        );
        // Breakers remain tripped.
        assert!(storage.is_drive_unavailable(0));
        assert!(storage.is_expert_unavailable(0));

        // Step 3: backdate both `tripped_at_ms` so the probe gate
        // admits the next attempt. `0` is "long ago" by definition
        // (it precedes any `breaker_now_ms()` value).
        storage.breaker(0).tripped_at_ms.store(0, Ordering::Release);
        storage.drive_breakers[0]
            .tripped_at_ms
            .store(0, Ordering::Release);

        // Step 4: probe is admitted, real read succeeds, both
        // breakers clear.
        let mut buf2 = pool.acquire().await;
        let n = storage
            .read_expert(0, &mut buf2)
            .await
            .expect("half-open probe should succeed against a healthy file");
        assert_eq!(n, expert_size);
        assert!(!storage.is_drive_unavailable(0));
        assert!(!storage.is_expert_unavailable(0));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `try_admit_probe` rate-limits recovery probes to one per
    /// `STORAGE_BREAKER_PROBE_INTERVAL`. Concurrent attempts race on
    /// a CAS and only one wins; the rest fall through to the
    /// short-circuit return so we don't dogpile a struggling drive.
    #[test]
    fn try_admit_probe_admits_at_most_one_per_interval() {
        let stamp = AtomicU64::new(0); // last probe was "long ago"
                                       // First call wins the CAS and advances `stamp` to now.
        assert!(NvmeStorage::try_admit_probe(&stamp));
        let first = stamp.load(Ordering::Acquire);
        assert!(first >= 1, "stamp should advance past zero");
        // Immediate follow-up is rejected — we're well inside
        // `STORAGE_BREAKER_PROBE_INTERVAL`.
        assert!(!NvmeStorage::try_admit_probe(&stamp));
        assert!(!NvmeStorage::try_admit_probe(&stamp));
        // Backdate the stamp past the interval — next probe admitted.
        stamp.store(0, Ordering::Release);
        assert!(NvmeStorage::try_admit_probe(&stamp));
    }

    /// `is_transient_io_error` classifies the well-known set of
    /// retryable kinds correctly.
    #[test]
    fn transient_io_error_classification() {
        use std::io::{Error, ErrorKind};
        assert!(is_transient_io_error(&Error::new(
            ErrorKind::Interrupted,
            ""
        )));
        assert!(is_transient_io_error(&Error::new(
            ErrorKind::WouldBlock,
            ""
        )));
        assert!(is_transient_io_error(&Error::new(ErrorKind::TimedOut, "")));
        // EIO maps to a raw-OS retryable error too.
        let eio = Error::from_raw_os_error(libc::EIO);
        assert!(is_transient_io_error(&eio));
        // UnexpectedEof is NOT transient: a permanently truncated
        // file won't grow back during the retry budget, so we surface
        // the short read on the first observation rather than wasting
        // 3×backoff. Short reads are handled by a dedicated branch in
        // `read_at_with_retries` that fails fast.
        assert!(!is_transient_io_error(&Error::new(
            ErrorKind::UnexpectedEof,
            ""
        )));
        // Permission denied is NOT transient — surfacing it lets the
        // operator fix the mount rather than retry forever.
        assert!(!is_transient_io_error(&Error::new(
            ErrorKind::PermissionDenied,
            ""
        )));
    }

    /// `HardwareFailure` converts to an `io::Error` and round-trips
    /// out through `into_inner().downcast` so call sites can recover
    /// the structured error.
    #[test]
    fn hardware_failure_roundtrips_through_io_error() {
        let hf = HardwareFailure::ExpertUnavailable {
            expert_id: 7,
            consecutive_failures: STORAGE_BREAKER_THRESHOLD,
        };
        let ioerr: io::Error = hf.into();
        let inner = ioerr.into_inner().expect("inner");
        let hf = inner.downcast::<HardwareFailure>().expect("downcast");
        match *hf {
            HardwareFailure::ExpertUnavailable { expert_id, .. } => assert_eq!(expert_id, 7),
            _ => panic!("wrong variant"),
        }
    }

    /// `pack_experts` rejects a duplicate id in the requested order rather
    /// than writing a malformed blob whose manifest collapses by key.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pack_experts_rejects_duplicate_order() {
        let dir = tempdir("packed-dup");
        let num_experts = 3u32;
        let d_model = 8usize;
        let d_ff = 16usize;
        let block = 4096usize;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let expert_size = weight_bytes.div_ceil(block) * block;
        generate_synthetic_experts(&dir, num_experts, expert_size, d_model, d_ff).unwrap();

        let cfg = StorageConfig {
            base_path: dir.clone(),
            expert_size,
            block_align: block,
            use_direct_io: false,
            num_experts_per_layer: None,
        };
        let source = NvmeStorage::new(cfg).unwrap();
        let pool = BufferPool::new(num_experts as usize * 2 + 4, expert_size, block);
        let blob_path = dir.join("dup.blob");
        let manifest_path = dir.join("dup.manifest.json");

        let err = pack_experts(&source, &pool, &[0, 1, 1], &blob_path, &manifest_path)
            .await
            .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod source_to_upload_tests {
    use super::*;
    use crate::aligned_buffer::AlignedBuffer;
    use crate::buffer_pool::BufferPool;

    fn directory() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "mer-source-to-upload-storage-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn storage(path: &Path) -> NvmeStorage {
        NvmeStorage::new(StorageConfig {
            base_path: path.to_path_buf(),
            expert_size: 4096,
            block_align: 4096,
            use_direct_io: false,
            num_experts_per_layer: Some(8),
        })
        .unwrap()
    }

    // Current-thread runtime also proves these checks precede block_in_place.
    #[tokio::test]
    async fn source_to_upload_wrong_length_rejected_before_fd_or_io() {
        let path = directory();
        let s = storage(&path);
        let mut dst = AlignedBuffer::new(8192, 4096);
        for len in [0, 4095, 4097, 8192] {
            let e = s
                .read_expert_into_aligned_slice(17, &mut dst.as_mut_slice()[..len])
                .await
                .unwrap_err();
            assert_eq!(e.kind(), io::ErrorKind::InvalidInput);
            assert!(s.files.lock().is_empty());
            assert!(s.breakers.read().is_empty());
        }
        std::fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test]
    async fn source_to_upload_misaligned_slice_rejected_before_fd_or_io() {
        let path = directory();
        let s = storage(&path);
        let mut dst = AlignedBuffer::new(8192, 4096);
        let e = s
            .read_expert_into_aligned_slice(17, &mut dst.as_mut_slice()[1..4097])
            .await
            .unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::InvalidInput);
        assert!(s.files.lock().is_empty());
        assert!(s.breakers.read().is_empty());
        assert!(!s.is_drive_unavailable(17));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn source_to_upload_arbitrary_slice_matches_production_resolution() {
        let path = directory();
        std::fs::write(path.join("expert_2_1.bin"), vec![91; 4096]).unwrap();
        let mut dst = AlignedBuffer::new(8192, 4096);
        let pool = BufferPool::new(1, 4096, 4096);
        let mut reference = pool.acquire().await;
        for expected in [91, 37] {
            if expected == 37 {
                std::fs::write(path.join("expert_17.bin"), vec![37; 4096]).unwrap();
            }
            let s = storage(&path);
            assert_eq!(s.read_expert(17, &mut reference).await.unwrap(), 4096);
            let subrange = &mut dst.as_mut_slice()[4096..8192];
            assert_eq!(
                s.read_expert_into_aligned_slice(17, subrange)
                    .await
                    .unwrap(),
                4096
            );
            assert_eq!(subrange, reference.as_slice());
            assert!(subrange.iter().all(|b| *b == expected));
            assert!(dst.as_slice()[..4096].iter().all(|b| *b == 0));
        }
        std::fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn source_to_upload_slice_reuses_packed_offset_but_fd_evidence_rejects_packed() {
        let path = directory();
        let blob_path = path.join("experts.blob");
        let manifest_path = path.join("manifest.json");
        std::fs::write(&blob_path, [vec![19; 4096], vec![83; 4096]].concat()).unwrap();
        crate::packed_storage::PackedManifest::uniform(vec![9, 17], 4096, 4096)
            .write_to(&manifest_path)
            .unwrap();
        let packed =
            crate::packed_storage::PackedBlob::open(&blob_path, &manifest_path, false).unwrap();
        let s = storage(&path).with_packed_blob(Arc::new(packed));
        let mut dst = AlignedBuffer::new(4096, 4096);
        assert_eq!(
            s.read_expert_into_aligned_slice(17, dst.as_mut_slice())
                .await
                .unwrap(),
            4096
        );
        assert!(dst.as_slice().iter().all(|b| *b == 83));
        assert_eq!(
            s.source_to_upload_fd_evidence(17).unwrap_err().kind(),
            io::ErrorKind::InvalidInput
        );
        assert_eq!(
            s.read_expert_into_aligned_slice(99, dst.as_mut_slice())
                .await
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
        std::fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn source_to_upload_short_reads_use_existing_breakers_without_fallback() {
        let path = directory();
        std::fs::write(path.join("expert_17.bin"), [1]).unwrap();
        let s = storage(&path);
        let mut dst = AlignedBuffer::new(4096, 4096);
        for _ in 0..STORAGE_BREAKER_THRESHOLD {
            assert!(s
                .read_expert_into_aligned_slice(17, dst.as_mut_slice())
                .await
                .is_err());
        }
        assert!(s.is_expert_unavailable(17));
        assert!(s.is_drive_unavailable(17));
        let pool = BufferPool::new(1, 4096, 4096);
        let mut buf = pool.acquire().await;
        assert!(s.read_expert(17, &mut buf).await.is_err());
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn source_to_upload_fd_evidence_is_platform_honest() {
        let path = directory();
        std::fs::write(path.join("expert_17.bin"), vec![0; 4096]).unwrap();
        let s = storage(&path);
        #[cfg(target_os = "linux")]
        {
            let (flags, direct, len) = s.source_to_upload_fd_evidence(17).unwrap();
            assert!(!direct);
            assert_eq!(flags & libc::O_DIRECT, 0);
            assert_eq!(len, 4096);
        }
        #[cfg(not(target_os = "linux"))]
        {
            assert_eq!(
                s.source_to_upload_fd_evidence(17).unwrap_err().kind(),
                io::ErrorKind::Unsupported
            );
            assert!(s.files.lock().is_empty());
        }
        std::fs::remove_dir_all(path).unwrap();
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn source_upload_external_batch_rejects_bad_destinations_before_fd_or_io() {
        let dir = directory();
        let storage = NvmeStorage::new(StorageConfig {
            base_path: dir.clone(),
            expert_size: 2_658_304,
            block_align: 4096,
            use_direct_io: false,
            num_experts_per_layer: None,
        })
        .unwrap();
        let pool = crate::buffer_pool::BufferPool::new(1, 2_658_304 + 4096, 4096);
        let mut buffer = pool.try_acquire().unwrap();
        for (offset, len) in [(0, 2_658_303), (1, 2_658_304), (0, 2_654_208)] {
            let mut destinations = vec![&mut buffer.as_mut_slice()[offset..offset + len]];
            let error = storage
                .read_experts_batch_into_aligned_slices(&[999], &mut destinations)
                .await
                .unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        }
        let error = storage
            .read_experts_batch_into_aligned_slices(&[1], &mut [])
            .await
            .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
    }
    #[test]
    fn source_upload_batch_preserves_production_scoped_thread_and_retry_shape() {
        let source = include_str!("io_provider.rs");
        let production = source
            .split("pub async fn read_experts_batch(")
            .nth(1)
            .unwrap()
            .split("pub(crate) async fn read_experts_batch_into_aligned_slices")
            .next()
            .unwrap();
        let treatment = source
            .split("pub(crate) async fn read_experts_batch_into_aligned_slices(")
            .nth(1)
            .unwrap()
            .split("/// **Tier 2.**")
            .next()
            .unwrap();
        for body in [production, treatment] {
            assert_eq!(body.matches("tokio::task::block_in_place(").count(), 1);
            assert_eq!(body.matches("std::thread::scope(").count(), 1);
            assert!(body.contains("scope.spawn("));
            assert!(body.contains("read_at_with_retries("));
            assert!(
                body.find("files.push(self.fd_for(id)?)").unwrap()
                    < body.find("tokio::task::block_in_place(").unwrap()
            );
            assert!(body.contains(".zip(id_vec.iter())"));
            assert!(body.contains("h.join()"));
        }
        assert!(!treatment.contains("read_expert("));
        assert!(!treatment.contains("spawn_blocking"));
        assert!(treatment.contains("self.prove_source_upload_fd(id, file)?"));
        assert!(!treatment.contains("F_GETFL"));
        assert!(!treatment.contains("metadata()"));
        assert!(!production.contains("prove_source_upload_fd"));
    }

    fn proof_counts(s: &NvmeStorage, requests: u64, hits: u64, misses: u64, failures: u64) {
        assert_eq!(
            s.source_upload_fd_proof_snapshot(),
            SourceUploadFdProofSnapshot {
                source_upload_fd_proof_requests: requests,
                source_upload_fd_proof_hits: hits,
                source_upload_fd_proof_misses: misses,
                source_upload_fd_proof_failures: failures,
            }
        );
        assert_eq!(hits + misses, requests);
        assert!(failures <= misses);
    }

    // Portable seam supplies the direct flag only; real File metadata and the
    // production length validator still run. No production platform fallback.
    fn portable_proof(file: &File) -> io::Result<()> {
        validate_source_upload_fd(4096, || Ok(true), || Ok(file.metadata()?.len()))
    }

    #[test]
    fn source_upload_fd_proof_first_miss_then_same_arc_hit_without_reproof() {
        let path = directory();
        std::fs::write(path.join("expert_17.bin"), vec![0; 4096]).unwrap();
        let s = storage(&path);
        let first = s.fd_for(17).unwrap();
        let attempts = AtomicU64::new(0);
        s.prove_source_upload_fd_with(17, &first, |file| {
            attempts.fetch_add(1, Ordering::Relaxed);
            assert!(s.source_upload_fd_proofs.proven.try_lock().is_some());
            assert!(s.files.try_lock().is_some());
            portable_proof(file)
        })
        .unwrap();
        proof_counts(&s, 1, 0, 1, 0);
        let second = s.fd_for(17).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        s.prove_source_upload_fd_with(17, &second, |_| panic!("hit ran real proof"))
            .unwrap();
        proof_counts(&s, 2, 1, 1, 0);
        assert_eq!(attempts.load(Ordering::Relaxed), 1);
        s.reset_source_upload_fd_proof_telemetry();
        proof_counts(&s, 0, 0, 0, 0);
        s.prove_source_upload_fd_with(17, &second, |_| panic!("reset discarded proof"))
            .unwrap();
        proof_counts(&s, 1, 1, 0, 0);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn source_upload_fd_proof_eviction_reopen_rejects_dead_and_live_stale_weak() {
        let path = directory();
        for id in [17, 18] {
            std::fs::write(path.join(format!("expert_{id}.bin")), vec![0; 4096]).unwrap();
        }
        let s = storage(&path).with_max_open_files(1);
        let first = s.fd_for(17).unwrap();
        s.prove_source_upload_fd_with(17, &first, portable_proof)
            .unwrap();
        let old = Arc::downgrade(&first);
        drop(s.fd_for(18).unwrap());
        assert_eq!(s.files.lock().len(), 1);
        drop(first);
        assert!(old.upgrade().is_none(), "proof must not keep the fd alive");
        let second = s.fd_for(17).unwrap();
        s.prove_source_upload_fd_with(17, &second, portable_proof)
            .unwrap();
        proof_counts(&s, 2, 0, 2, 0);
        drop(s.fd_for(18).unwrap());
        let third = s.fd_for(17).unwrap();
        assert!(!Arc::ptr_eq(&second, &third));
        assert!(s.source_upload_fd_proofs.proven.lock()[&17]
            .upgrade()
            .is_some());
        s.prove_source_upload_fd_with(17, &third, portable_proof)
            .unwrap();
        proof_counts(&s, 3, 0, 3, 0);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn source_upload_fd_proof_wrong_file_sizes_fail_closed_and_are_not_cached() {
        let path = directory();
        for len in [4095, 4097, 8192] {
            std::fs::write(path.join("expert_17.bin"), vec![0; len]).unwrap();
            let s = storage(&path);
            let file = s.fd_for(17).unwrap();
            for _ in 0..2 {
                assert_eq!(
                    s.prove_source_upload_fd_with(17, &file, portable_proof)
                        .unwrap_err()
                        .kind(),
                    io::ErrorKind::InvalidInput
                );
            }
            proof_counts(&s, 2, 0, 2, 2);
            assert!(s.source_upload_fd_proofs.proven.lock().is_empty());
        }
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn source_upload_fd_proof_syscall_failures_and_missing_direct_are_not_cached() {
        let path = directory();
        std::fs::write(path.join("expert_17.bin"), vec![0; 4096]).unwrap();
        let s = storage(&path);
        let file = s.fd_for(17).unwrap();
        for case in 0..3 {
            let error = s
                .prove_source_upload_fd_with(17, &file, |_| {
                    validate_source_upload_fd(
                        4096,
                        || match case {
                            0 => Err(io::Error::from_raw_os_error(libc::EBADF)),
                            1 => Ok(false),
                            _ => Ok(true),
                        },
                        || {
                            assert_eq!(case, 2, "failed F_GETFL/direct check must short circuit");
                            Err(io::Error::from_raw_os_error(libc::EIO))
                        },
                    )
                })
                .unwrap_err();
            match case {
                0 => assert_eq!(error.raw_os_error(), Some(libc::EBADF)),
                1 => assert_eq!(error.kind(), io::ErrorKind::InvalidInput),
                _ => assert_eq!(error.raw_os_error(), Some(libc::EIO)),
            }
        }
        proof_counts(&s, 3, 0, 3, 3);
        assert!(s.source_upload_fd_proofs.proven.lock().is_empty());
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn source_upload_fd_proof_real_buffered_fd_is_rejected_on_every_platform() {
        let path = directory();
        std::fs::write(path.join("expert_17.bin"), vec![0; 4096]).unwrap();
        let s = storage(&path);
        let file = s.fd_for(17).unwrap();
        let error = s.prove_source_upload_fd(17, &file).unwrap_err();
        #[cfg(target_os = "linux")]
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        #[cfg(not(target_os = "linux"))]
        assert_eq!(error.kind(), io::ErrorKind::Unsupported);
        proof_counts(&s, 1, 0, 1, 1);
        assert!(s.source_upload_fd_proofs.proven.lock().is_empty());
        std::fs::remove_dir_all(path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn source_upload_fd_proof_ordinary_batch_has_zero_activity() {
        let path = directory();
        for id in [17, 18] {
            std::fs::write(path.join(format!("expert_{id}.bin")), vec![id as u8; 4096]).unwrap();
        }
        let s = storage(&path);
        let pool = BufferPool::new(2, 4096, 4096);
        let mut a = pool.try_acquire().unwrap();
        let mut b = pool.try_acquire().unwrap();
        assert_eq!(
            s.read_experts_batch(&[17], &mut [&mut a]).await.unwrap(),
            4096
        );
        assert_eq!(
            s.read_experts_batch(&[18, 17], &mut [&mut a, &mut b])
                .await
                .unwrap(),
            8192
        );
        assert!(a.as_slice().iter().all(|v| *v == 18));
        assert!(b.as_slice().iter().all(|v| *v == 17));
        proof_counts(&s, 0, 0, 0, 0);
        assert!(s.source_upload_fd_proofs.proven.lock().is_empty());
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn source_upload_fd_proof_concurrent_first_use_publishes_only_success() {
        let path = directory();
        std::fs::write(path.join("expert_17.bin"), vec![0; 4096]).unwrap();
        for succeed in [false, true] {
            let s = storage(&path);
            let file = s.fd_for(17).unwrap();
            let barrier = std::sync::Barrier::new(4);
            let attempts = AtomicU64::new(0);
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..4)
                    .map(|_| {
                        scope.spawn(|| {
                            s.prove_source_upload_fd_with(17, &file, |file| {
                                attempts.fetch_add(1, Ordering::Relaxed);
                                // All four must reach proof, with neither cache lock held,
                                // before any may publish; no sleeps or scheduling guesses.
                                barrier.wait();
                                if succeed {
                                    portable_proof(file)
                                } else {
                                    Err(io::Error::other("proof failed"))
                                }
                            })
                        })
                    })
                    .collect();
                for handle in handles {
                    assert_eq!(handle.join().unwrap().is_ok(), succeed);
                }
            });
            assert_eq!(attempts.load(Ordering::Relaxed), 4);
            proof_counts(&s, 4, 0, 4, if succeed { 0 } else { 4 });
            assert_eq!(s.source_upload_fd_proofs.proven.lock().is_empty(), !succeed);
            if succeed {
                s.prove_source_upload_fd_with(17, &file, |_| panic!("completed proof missing"))
                    .unwrap();
                proof_counts(&s, 5, 1, 4, 0);
            }
        }
        std::fs::remove_dir_all(path).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn source_upload_fd_proof_real_direct_batch_miss_hit_and_wrong_size() {
        let path = directory();
        let full = 2_658_304;
        std::fs::write(path.join("expert_17.bin"), vec![17; full]).unwrap();
        let s = NvmeStorage::new(StorageConfig {
            base_path: path.clone(),
            expert_size: full,
            block_align: 4096,
            use_direct_io: true,
            num_experts_per_layer: None,
        })
        .unwrap();
        if let Err(error) = s.fd_for(17) {
            if error.kind() == io::ErrorKind::InvalidInput
                || error.raw_os_error() == Some(libc::EOPNOTSUPP)
            {
                eprintln!("O_DIRECT unavailable on test filesystem: {error}; portable proof seam covers identity and failures");
                std::fs::remove_dir_all(path).unwrap();
                return;
            }
            panic!("unexpected direct open failure: {error}");
        }
        let mut destination = AlignedBuffer::new(full, 4096);
        for _ in 0..2 {
            assert_eq!(
                s.read_experts_batch_into_aligned_slices(&[17], &mut [destination.as_mut_slice()])
                    .await
                    .unwrap(),
                full
            );
            assert!(destination.as_slice().iter().all(|v| *v == 17));
        }
        proof_counts(&s, 2, 1, 1, 0);
        std::fs::write(path.join("expert_18.bin"), vec![0; full + 4096]).unwrap();
        assert_eq!(
            s.read_experts_batch_into_aligned_slices(&[18], &mut [destination.as_mut_slice()])
                .await
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidInput
        );
        proof_counts(&s, 3, 1, 2, 1);
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn source_to_upload_copy_elision_b_preproof_has_no_source_read() {
        let source = include_str!("io_provider.rs");
        let body = source
            .split("pub(crate) fn preprove_source_upload_fds(")
            .nth(1)
            .unwrap()
            .split("fn prove_source_upload_fd(")
            .next()
            .unwrap();
        let compact: String = body.split_whitespace().collect();
        assert_eq!(compact, "&self,ids:&[u32])->io::Result<()>{for&idinids{letfile=self.fd_for(id)?;self.prove_source_upload_fd(id,&file)?;}Ok(())}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn source_to_upload_copy_elision_b_preproof_reset_retains_ids_and_batch_hits() {
        let path = directory();
        let full = 2_658_304;
        let s = NvmeStorage::new(StorageConfig {
            base_path: path.clone(),
            expert_size: full,
            block_align: 4096,
            use_direct_io: false,
            num_experts_per_layer: None,
        })
        .unwrap()
        .with_max_open_files(8);
        let ids: Vec<u32> = (0..8).collect();
        let mut files = Vec::new();
        // Portable fixture injects only the existing proof operation. This is
        // not Linux O_DIRECT authority; actual helper/scheduler and bytes run.
        for &id in &ids {
            std::fs::write(path.join(format!("expert_{id}.bin")), vec![id as u8; full]).unwrap();
            let file = s.fd_for(id).unwrap();
            s.prove_source_upload_fd_with(id, &file, |file| {
                validate_source_upload_fd(full as u64, || Ok(true), || Ok(file.metadata()?.len()))
            })
            .unwrap();
            files.push(file);
        }
        s.reset_source_upload_fd_proof_telemetry();
        s.preprove_source_upload_fds(&ids).unwrap();
        proof_counts(&s, 8, 8, 0, 0);
        assert!(s.breakers.read().is_empty());
        s.reset_source_upload_fd_proof_telemetry();
        proof_counts(&s, 0, 0, 0, 0);
        let mut host = AlignedBuffer::new(full * 8, 4096);
        for _phase in 0..2 {
            for k in 1..=8 {
                for _arm in 0..2 {
                    let mut destinations: Vec<_> = host.as_mut_slice()[..k * full]
                        .chunks_exact_mut(full)
                        .collect();
                    assert_eq!(
                        s.read_experts_batch_into_aligned_slices(&ids[..k], &mut destinations)
                            .await
                            .unwrap(),
                        k * full
                    );
                    for (j, dst) in destinations.iter().enumerate() {
                        assert!(dst.iter().all(|b| *b == j as u8));
                    }
                }
            }
            proof_counts(&s, 72, 72, 0, 0);
            for (&id, file) in ids.iter().zip(&files) {
                assert!(Arc::ptr_eq(file, &s.fd_for(id).unwrap()));
            }
            s.reset_source_upload_fd_proof_telemetry();
            proof_counts(&s, 0, 0, 0, 0);
        }
        std::fs::remove_dir_all(path).unwrap();
    }
}
