//! Qualification-only source throughput. No Engine, model, cache, GPU or predictor.
//! The storage implementations and their retry semantics are used unchanged.

use crate::buffer_pool::{BufferPool, PooledBuffer};
use crate::gpu_native_oracle_routes::OracleRouteTrace;
use crate::io_provider::{NvmeStorage, StorageConfig};
use parking_lot::Mutex;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

const SCHEMA: &str = "mer.expert-source-throughput.v1";
const PARENT: &str = "4eb143b34b00826e2031b313f3734bace18a7ecc";
const TRACE_SHA: &str = "7f5f07489a1f2eb755ca0ae8af1ac96f3d0b5ab14aff7c1c957337d83707fc1b";
const DATA_PATH: &str = "/mnt/localssd/data/qwen3-coder-q4";
const EXPERT_SIZE: usize = 2_658_304;
const ALIGN: usize = 4096;
const EXPERTS_PER_LAYER: u32 = 128;
const TOTAL_EXPERTS: u32 = 6144;
const MAX_QD: usize = 8;
const MAX_READS: usize = 1_048_576;
const WARMUPS: usize = 1;
const REPETITIONS: usize = 3;
const IO_UNSUPPORTED: &str = "Existing IoUringStorage::linux_impl::Ring::fd_for uses File::open (no O_DIRECT), and resolves only expert_<global>.bin or expert_<global:04>.bin; IoUringConfig has neither use_direct_io nor num_experts_per_layer. It cannot satisfy the authoritative O_DIRECT/layer-qualified data contract. No ring is constructed, no fixed read is issued, and no pread fallback is substituted.";

#[derive(clap::Args, Debug)]
pub(crate) struct CommandArgs {
    /// Immutable ORACLE-0A report; exact frozen artifact SHA256 is required.
    #[arg(long)]
    oracle_trace: PathBuf,
    /// Prefix length of the flattened routes, cycling only if longer than the trace.
    #[arg(long, default_value_t = 4096)]
    read_count: usize,
    /// New JSON report destination. Existing files are never overwritten.
    #[arg(long)]
    report_out: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Mode {
    SequentialPread,
    IndependentPread,
    BatchPread,
    IoUringFixed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct Arm {
    mode: Mode,
    configured_qd: usize,
}

fn arms() -> Vec<Arm> {
    let mut result = vec![Arm {
        mode: Mode::SequentialPread,
        configured_qd: 1,
    }];
    for mode in [Mode::IndependentPread, Mode::BatchPread, Mode::IoUringFixed] {
        result.extend([2, 4, 8].map(|configured_qd| Arm {
            mode,
            configured_qd,
        }));
    }
    result
}

fn run_order(repetition: usize) -> Vec<Arm> {
    let mut result = arms();
    match repetition % 3 {
        0 => {}
        1 => result.reverse(),
        _ => result.rotate_left(3),
    }
    result
}

#[derive(Serialize)]
struct Configuration {
    data_path: &'static str,
    expert_size: usize,
    block_align: usize,
    use_direct_io: bool,
    num_experts_per_layer: u32,
    total_experts: u32,
    reads_per_repetition: usize,
    warmup_repetitions: usize,
    measured_repetitions: usize,
    pool_capacity: usize,
    tokio_worker_threads: usize,
    arms: Vec<Arm>,
    measured_orders: Vec<Vec<Arm>>,
    sampling: &'static str,
    checksum_contract: &'static str,
    wall_time_contract: &'static str,
    batch_latency_contract: &'static str,
    completion_accounting: &'static str,
    qd_contract: &'static str,
    fd_cache_contract: &'static str,
}

impl Configuration {
    fn new(read_count: usize) -> Self {
        Self {
            data_path: DATA_PATH,
            expert_size: EXPERT_SIZE,
            block_align: ALIGN,
            use_direct_io: true,
            num_experts_per_layer: EXPERTS_PER_LAYER,
            total_experts: TOTAL_EXPERTS,
            reads_per_repetition: read_count,
            warmup_repetitions: WARMUPS,
            measured_repetitions: REPETITIONS,
            pool_capacity: MAX_QD,
            tokio_worker_threads: MAX_QD,
            arms: arms(),
            measured_orders: (0..REPETITIONS).map(run_order).collect(),
            sampling: "trace.records in stored position/layer/rank order; global = layer*128 + local; retain repeats; prefix, cycling whole stream if needed; SHA256 of consecutive u32 little-endian IDs",
            checksum_contract: "touch first and last u64 of every 4096-byte block (including short final block); per-read wrapping FNV-1a over those bytes; SHA256 domain mer.expert-source-touch.v1 NUL, then in logical order index u64-le, id u32-le, bytes u64-le, touch u64-le; sampled touch is not a full-file integrity hash",
            wall_time_contract: "timer includes buffer acquisition, scheduling, all reads/retries, sampled touch, receipt collection and buffer release; excludes fd preopen/audit, pool allocation, final checksum reduction and report serialization",
            batch_latency_contract: "API entry-to-return wall time in microseconds, excluding acquisition/touch; one read_expert call is a batch of 1 for sequential/independent; one read_experts_batch call for batch-pread; percentiles nearest-rank",
            completion_accounting: "only API-confirmed complete experts/bytes are counted; batch error exposes no partial completion, so all members are failed-or-unconfirmed; no physical byte estimate on errors",
            qd_contract: "total outstanding expert requests bounded by QD; independent tasks are refilled after completion; batch arms have exactly one outstanding batch; actual kernel/device inflight is unobservable without changing storage, reported null",
            fd_cache_contract: "benchmark-local NvmeStorage with fd cache sized to sampled unique IDs, preopened once before timing; no expert data cache; all cached data fds audited via Linux /proc/self/fd and F_GETFL before and after every repetition",
        }
    }
}

#[derive(Default, Serialize)]
struct Provenance {
    build_git_sha: String,
    build_git_dirty: String,
    git_sha: Option<String>,
    tree_sha: Option<String>,
    parent_sha: Option<String>,
    executable_sha256: Option<String>,
}

fn git(args: &[&str]) -> io::Result<String> {
    let output = std::process::Command::new("git")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn provenance(out: &mut Provenance) -> io::Result<()> {
    out.build_git_sha = env!("MER_BUILD_GIT_SHA").to_string();
    out.build_git_dirty = env!("MER_BUILD_GIT_DIRTY").to_string();
    out.git_sha = Some(git(&["rev-parse", "HEAD"])?);
    out.tree_sha = Some(git(&["rev-parse", "HEAD^{tree}"])?);
    out.parent_sha = Some(git(&["rev-parse", "HEAD^"])?);
    out.executable_sha256 = Some(hex_sha(&std::fs::read(std::env::current_exe()?)?));
    ensure(
        out.git_sha.as_deref() == Some(out.build_git_sha.as_str()),
        "build and checkout SHA differ",
    )?;
    ensure(
        out.parent_sha.as_deref() == Some(PARENT),
        "qualifier is not a direct child of the frozen parent",
    )?;
    ensure(
        out.build_git_dirty == "false",
        "binary was built from dirty or unavailable source provenance",
    )?;
    ensure(
        git(&["status", "--porcelain", "--untracked-files=no"])?.is_empty(),
        "tracked checkout is dirty",
    )
}

#[derive(Default, Serialize)]
struct InputEvidence {
    trace_path: PathBuf,
    expected_trace_sha256: &'static str,
    trace_sha256: Option<String>,
    sampled_id_sequence_sha256: Option<String>,
    source_route_ids: usize,
    sampled_unique_ids: usize,
    sampled_layers: usize,
}

fn hex_sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sequence_sha(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn ensure(ok: bool, message: &str) -> io::Result<()> {
    if ok {
        Ok(())
    } else {
        Err(io::Error::other(message))
    }
}

fn flatten_sample(trace: &OracleRouteTrace, count: usize) -> io::Result<Vec<u32>> {
    trace.validate().map_err(io::Error::other)?;
    ensure(
        trace.geometry.layers == 48 && trace.geometry.experts == 128 && trace.geometry.top_k == 8,
        "ORACLE route geometry must be 48 layers, 128 local experts, top-k 8",
    )?;
    ensure(
        (1..=MAX_READS).contains(&count),
        "read-count must be in 1..=1048576",
    )?;
    let flattened: Vec<u32> = trace
        .records
        .iter()
        .flat_map(|r| {
            r.ordered_selected_expert_ids
                .iter()
                .map(move |id| r.layer_index as u32 * EXPERTS_PER_LAYER + id)
        })
        .collect();
    ensure(
        !flattened.is_empty() && flattened.iter().all(|id| *id < TOTAL_EXPERTS),
        "empty or invalid global route sequence",
    )?;
    Ok(flattened.iter().copied().cycle().take(count).collect())
}

fn decode_trace(
    bytes: &[u8],
    expected_sha: &str,
    count: usize,
) -> io::Result<(OracleRouteTrace, Vec<u32>)> {
    ensure(
        hex_sha(bytes) == expected_sha,
        "frozen ORACLE artifact SHA256 mismatch",
    )?;
    let root: serde_json::Value = serde_json::from_slice(bytes)?;
    ensure(
        root["schema"] == crate::gpu_native_oracle_routes::SCHEMA
            && root["complete"] == true
            && root["failure"].is_null(),
        "ORACLE report schema/completion/failure mismatch",
    )?;
    let trace: OracleRouteTrace = serde_json::from_value(root["trace"].clone())?;
    let ids = flatten_sample(&trace, count)?;
    Ok((trace, ids))
}

fn load_input(args: &CommandArgs, evidence: &mut InputEvidence) -> io::Result<Arc<Vec<u32>>> {
    // Hash and parse the exact same immutable snapshot, never reopen the trace.
    let bytes = std::fs::read(&args.oracle_trace)?;
    evidence.trace_sha256 = Some(hex_sha(&bytes));
    let (trace, ids) = decode_trace(&bytes, TRACE_SHA, args.read_count)?;
    evidence.source_route_ids = trace.total_selected_expert_ids;
    evidence.sampled_id_sequence_sha256 = Some(sequence_sha(&ids));
    evidence.sampled_unique_ids = ids.iter().copied().collect::<BTreeSet<_>>().len();
    evidence.sampled_layers = ids
        .iter()
        .map(|id| id / EXPERTS_PER_LAYER)
        .collect::<BTreeSet<_>>()
        .len();
    Ok(Arc::new(ids))
}

#[derive(Default)]
struct PoolCounts {
    current: usize,
    peak: usize,
}

struct BenchPool {
    pool: BufferPool,
    counts: Mutex<PoolCounts>,
}

impl BenchPool {
    fn new(size: usize) -> Arc<Self> {
        Arc::new(Self {
            pool: BufferPool::new_qualification_oracle_future_source(MAX_QD, size, ALIGN),
            counts: Mutex::new(PoolCounts::default()),
        })
    }
    fn reset(&self) -> io::Result<()> {
        let mut counts = self.counts.lock();
        ensure(
            counts.current == 0 && self.final_slots() == 0,
            "pool has outstanding leases before repetition",
        )?;
        *counts = PoolCounts::default();
        Ok(())
    }
    fn acquire(self: &Arc<Self>) -> io::Result<Lease> {
        let mut counts = self.counts.lock();
        let buffer = self
            .pool
            .try_acquire()
            .ok_or_else(|| io::Error::other("benchmark pool exhausted"))?;
        counts.current += 1;
        counts.peak = counts.peak.max(counts.current);
        Ok(Lease {
            buffer: Some(buffer),
            owner: self.clone(),
        })
    }
    fn final_slots(&self) -> usize {
        self.pool.capacity() - self.pool.primary_available()
    }
}

struct Lease {
    buffer: Option<PooledBuffer>,
    owner: Arc<BenchPool>,
}
impl Lease {
    fn buffer(&mut self) -> &mut PooledBuffer {
        self.buffer.as_mut().expect("live lease")
    }
}
impl Drop for Lease {
    fn drop(&mut self) {
        let mut counts = self.owner.counts.lock();
        drop(self.buffer.take());
        counts.current -= 1;
    }
}

#[derive(Default)]
struct Inflight {
    current: AtomicUsize,
    peak: AtomicUsize,
}
struct ReadGuard<'a> {
    counter: &'a Inflight,
    reads: usize,
}
impl Inflight {
    fn enter(&self, reads: usize) -> ReadGuard<'_> {
        let now = self.current.fetch_add(reads, Ordering::SeqCst) + reads;
        self.peak.fetch_max(now, Ordering::SeqCst);
        ReadGuard {
            counter: self,
            reads,
        }
    }
}
impl Drop for ReadGuard<'_> {
    fn drop(&mut self) {
        self.counter.current.fetch_sub(self.reads, Ordering::SeqCst);
    }
}

// Touch only a small fixed portion of every block so the diagnostic does not
// become a full-file CPU hashing benchmark. This is deliberately documented.
fn touch(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for block in bytes.chunks(ALIGN) {
        for byte in block.iter().take(8).chain(block.iter().rev().take(8).rev()) {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
    }
    std::hint::black_box(hash)
}

#[derive(Clone, Copy)]
struct Receipt {
    bytes: usize,
    touch: u64,
}
fn checksum(ids: &[u32], receipts: &[Option<Receipt>]) -> Option<String> {
    if ids.len() != receipts.len() || receipts.iter().any(Option::is_none) {
        return None;
    }
    let mut h = Sha256::new();
    h.update(b"mer.expert-source-touch.v1\0");
    for (index, (&id, receipt)) in ids.iter().zip(receipts).enumerate() {
        let receipt = receipt.as_ref()?;
        h.update((index as u64).to_le_bytes());
        h.update(id.to_le_bytes());
        h.update((receipt.bytes as u64).to_le_bytes());
        h.update(receipt.touch.to_le_bytes());
    }
    Some(format!("{:x}", h.finalize()))
}

struct ReadResult {
    start: usize,
    count: usize,
    latency_us: Option<f64>,
    result: io::Result<Vec<Receipt>>,
}

async fn single_read(
    storage: Arc<NvmeStorage>,
    pool: Arc<BenchPool>,
    inflight: Arc<Inflight>,
    index: usize,
    id: u32,
) -> ReadResult {
    let mut latency_us = None;
    let result = async {
        let mut lease = pool.acquire()?;
        let guard = inflight.enter(1);
        let start = Instant::now();
        let result = storage.read_expert(id, lease.buffer()).await;
        latency_us = Some(start.elapsed().as_secs_f64() * 1e6);
        drop(guard);
        let bytes = result?;
        ensure(
            bytes == storage.config().expert_size,
            "read_expert byte reconciliation failed",
        )?;
        let receipt = Receipt {
            bytes,
            touch: touch(&lease.buffer().as_slice()[..bytes]),
        };
        Ok(vec![receipt])
    }
    .await;
    ReadResult {
        start: index,
        count: 1,
        latency_us,
        result,
    }
}

async fn batch_read(
    storage: &NvmeStorage,
    pool: &Arc<BenchPool>,
    inflight: &Inflight,
    start_index: usize,
    ids: &[u32],
) -> ReadResult {
    let mut latency_us = None;
    let result = async {
        let mut leases = (0..ids.len())
            .map(|_| pool.acquire())
            .collect::<io::Result<Vec<_>>>()?;
        let mut buffers: Vec<_> = leases.iter_mut().map(Lease::buffer).collect();
        let guard = inflight.enter(ids.len());
        let start = Instant::now();
        let result = storage.read_experts_batch(ids, &mut buffers).await;
        latency_us = Some(start.elapsed().as_secs_f64() * 1e6);
        drop(guard);
        let bytes = result?;
        ensure(
            bytes == ids.len() * storage.config().expert_size,
            "read_experts_batch byte reconciliation failed",
        )?;
        Ok(buffers
            .iter()
            .map(|buffer| Receipt {
                bytes: buffer.len(),
                touch: touch(buffer.as_slice()),
            })
            .collect())
    }
    .await;
    ReadResult {
        start: start_index,
        count: ids.len(),
        latency_us,
        result,
    }
}

#[derive(Default, Serialize)]
struct Latency {
    p50_us: Option<f64>,
    p95_us: Option<f64>,
    p99_us: Option<f64>,
    mean_us: Option<f64>,
}
fn latency(values: &mut [f64]) -> Latency {
    if values.is_empty() {
        return Latency::default();
    }
    values.sort_by(f64::total_cmp);
    let percentile =
        |p: f64| Some(values[((values.len() as f64 * p).ceil() as usize).saturating_sub(1)]);
    Latency {
        p50_us: percentile(0.50),
        p95_us: percentile(0.95),
        p99_us: percentile(0.99),
        mean_us: Some(values.iter().sum::<f64>() / values.len() as f64),
    }
}

#[derive(Serialize)]
struct RunResult {
    #[serde(flatten)]
    arm: Arm,
    warmup: bool,
    repetition: usize,
    order_index: usize,
    status: &'static str,
    unsupported_reason: Option<&'static str>,
    backend_used: Option<&'static str>,
    requested_reads: usize,
    completed_reads: usize,
    failures: usize,
    failure_events: usize,
    failure_examples: Vec<String>,
    bytes_completed: u64,
    wall_us: Option<f64>,
    decimal_gb_s: Option<f64>,
    gib_s: Option<f64>,
    reads_s: Option<f64>,
    batch_count: usize,
    batch_wall_latency: Latency,
    checksum: Option<String>,
    sampled_id_sequence_sha256: String,
    pool_peak_slots: usize,
    pool_final_slots: usize,
    tracked_pool_final_slots: usize,
    peak_outstanding_expert_requests: usize,
    final_outstanding_expert_requests: usize,
    actual_peak_inflight_reads: Option<usize>,
    direct_io_verified: bool,
    direct_fd_audits: Vec<DirectAudit>,
    gates: Vec<Gate>,
}

impl RunResult {
    fn new(arm: Arm, warmup: bool, repetition: usize, order_index: usize, ids: &[u32]) -> Self {
        Self {
            arm,
            warmup,
            repetition,
            order_index,
            status: "failed",
            unsupported_reason: None,
            backend_used: None,
            requested_reads: ids.len(),
            completed_reads: 0,
            failures: 0,
            failure_events: 0,
            failure_examples: vec![],
            bytes_completed: 0,
            wall_us: None,
            decimal_gb_s: None,
            gib_s: None,
            reads_s: None,
            batch_count: 0,
            batch_wall_latency: Latency::default(),
            checksum: None,
            sampled_id_sequence_sha256: sequence_sha(ids),
            pool_peak_slots: 0,
            pool_final_slots: 0,
            tracked_pool_final_slots: 0,
            peak_outstanding_expert_requests: 0,
            final_outstanding_expert_requests: 0,
            actual_peak_inflight_reads: None,
            direct_io_verified: false,
            direct_fd_audits: vec![],
            gates: vec![],
        }
    }
    fn error(&mut self, detail: String) {
        self.failure_events += 1;
        if self.failure_examples.len() < 16 {
            self.failure_examples.push(detail);
        }
    }
    fn collect(
        &mut self,
        outcome: ReadResult,
        receipts: &mut [Option<Receipt>],
        latencies: &mut Vec<f64>,
    ) {
        if let Some(us) = outcome.latency_us {
            self.batch_count += 1;
            latencies.push(us);
        }
        match outcome.result {
            Ok(reads) if reads.len() == outcome.count => {
                for (offset, receipt) in reads.into_iter().enumerate() {
                    let index = outcome.start + offset;
                    if let Some(slot) = receipts.get_mut(index).filter(|slot| slot.is_none()) {
                        self.completed_reads += 1;
                        self.bytes_completed += receipt.bytes as u64;
                        *slot = Some(receipt);
                    } else {
                        self.error(format!(
                            "duplicate or out-of-range completion index {index}"
                        ));
                    }
                }
            }
            Ok(_) => self.error("wrong number of batch receipts".into()),
            Err(error) => self.error(format!(
                "sequence index {} count {}: {error}",
                outcome.start, outcome.count
            )),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct Gate {
    name: &'static str,
    pass: bool,
}
fn gate(name: &'static str, pass: bool) -> Gate {
    Gate { name, pass }
}

fn run_gates(run: &RunResult, size: usize, expected_checksum: Option<&str>) -> Vec<Gate> {
    vec![
        gate(
            "all_requested_reads_completed",
            run.completed_reads == run.requested_reads,
        ),
        gate(
            "no_failed_or_unconfirmed_reads",
            run.failures == 0 && run.failure_events == 0,
        ),
        gate(
            "byte_reconciliation",
            run.bytes_completed == run.completed_reads as u64 * size as u64,
        ),
        gate(
            "checksum_reconciliation",
            run.checksum.is_some() && run.checksum.as_deref() == expected_checksum,
        ),
        gate(
            "pool_released",
            run.pool_final_slots == 0 && run.tracked_pool_final_slots == 0,
        ),
        gate(
            "qd_bound",
            run.peak_outstanding_expert_requests <= run.arm.configured_qd
                && run.pool_peak_slots <= run.arm.configured_qd
                && run.final_outstanding_expert_requests == 0
                && run
                    .actual_peak_inflight_reads
                    .is_none_or(|peak| peak <= run.arm.configured_qd),
        ),
        gate("o_direct_verified", run.direct_io_verified),
        gate(
            "no_backend_fallback",
            run.backend_used == Some("NvmeStorage/pread") && run.arm.mode != Mode::IoUringFixed,
        ),
    ]
}

async fn measure(
    arm: Arm,
    storage: Arc<NvmeStorage>,
    pool: Arc<BenchPool>,
    ids: Arc<Vec<u32>>,
    warmup: bool,
    repetition: usize,
    order_index: usize,
) -> RunResult {
    let mut out = RunResult::new(arm, warmup, repetition, order_index, &ids);
    if arm.mode == Mode::IoUringFixed {
        out.status = "unsupported";
        out.unsupported_reason = Some(IO_UNSUPPORTED);
        return out;
    }
    if let Err(error) = pool.reset() {
        out.error(error.to_string());
        out.failures = ids.len();
        out.pool_final_slots = pool.final_slots();
        out.tracked_pool_final_slots = pool.counts.lock().current;
        return out;
    }
    out.backend_used = Some("NvmeStorage/pread");
    let inflight = Arc::new(Inflight::default());
    let mut receipts = vec![None; ids.len()];
    let mut latencies = vec![];
    let start = Instant::now();
    if arm.mode == Mode::IndependentPread {
        // JoinSet length is the TOTAL bound. A task owns at most one lease
        // and one read; refill only after joining a completed task. All
        // spawned tasks are drained even on I/O/join failures.
        let mut tasks = tokio::task::JoinSet::new();
        let mut next = 0;
        loop {
            while next < ids.len() && tasks.len() < arm.configured_qd {
                tasks.spawn(single_read(
                    storage.clone(),
                    pool.clone(),
                    inflight.clone(),
                    next,
                    ids[next],
                ));
                next += 1;
            }
            match tasks.join_next().await {
                Some(Ok(outcome)) => out.collect(outcome, &mut receipts, &mut latencies),
                Some(Err(error)) => out.error(format!("independent read task: {error}")),
                None => break,
            }
        }
    } else if arm.mode == Mode::SequentialPread {
        for (index, &id) in ids.iter().enumerate() {
            let outcome =
                single_read(storage.clone(), pool.clone(), inflight.clone(), index, id).await;
            out.collect(outcome, &mut receipts, &mut latencies);
        }
    } else {
        for (batch_index, chunk) in ids.chunks(arm.configured_qd).enumerate() {
            let outcome = batch_read(
                &storage,
                &pool,
                &inflight,
                batch_index * arm.configured_qd,
                chunk,
            )
            .await;
            out.collect(outcome, &mut receipts, &mut latencies);
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    out.wall_us = Some(seconds * 1e6);
    out.decimal_gb_s = Some(out.bytes_completed as f64 / seconds / 1e9);
    out.gib_s = Some(out.bytes_completed as f64 / seconds / (1u64 << 30) as f64);
    out.reads_s = Some(out.completed_reads as f64 / seconds);
    out.failures = out.requested_reads - out.completed_reads;
    out.batch_wall_latency = latency(&mut latencies);
    out.checksum = checksum(&ids, &receipts);
    let counts = pool.counts.lock();
    out.pool_peak_slots = counts.peak;
    out.tracked_pool_final_slots = counts.current;
    out.pool_final_slots = pool.final_slots();
    out.peak_outstanding_expert_requests = inflight.peak.load(Ordering::SeqCst);
    out.final_outstanding_expert_requests = inflight.current.load(Ordering::SeqCst);
    out
}

#[derive(Serialize)]
struct DirectAudit {
    sampled_files: usize,
    verified_files: usize,
    verified_fds: usize,
    method: &'static str,
}

// Audit the descriptors actually held in NvmeStorage's warmed, non-evicting
// benchmark fd cache, not newly opened probe descriptors. F_GETFL neither
// takes ownership nor changes flags. This diagnostic is single-purpose and
// performs no fd activity concurrently with the audits.
#[cfg(target_os = "linux")]
fn audit_direct(storage: &NvmeStorage, ids: &[u32]) -> io::Result<DirectAudit> {
    use std::os::unix::fs::MetadataExt;
    let mut expected = BTreeSet::new();
    for id in ids.iter().copied().collect::<BTreeSet<_>>() {
        let meta = std::fs::metadata(storage.expert_path(id))?;
        ensure(
            meta.is_file() && meta.len() == storage.config().expert_size as u64,
            "sampled file size/type mismatch",
        )?;
        expected.insert((meta.dev(), meta.ino()));
    }
    let mut verified = BTreeSet::new();
    let mut count = 0;
    for entry in std::fs::read_dir("/proc/self/fd")? {
        let entry = entry?;
        let Some(fd) = entry
            .file_name()
            .to_str()
            .and_then(|s| s.parse::<i32>().ok())
        else {
            continue;
        };
        let Ok(meta) = std::fs::metadata(entry.path()) else {
            continue;
        };
        let identity = (meta.dev(), meta.ino());
        if expected.contains(&identity) {
            // SAFETY: F_GETFL reads flags of this observed process fd. There
            // is no ownership transfer or fd close, even on error.
            let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
            ensure(
                flags >= 0 && flags & libc::O_DIRECT != 0,
                "actual sampled data fd does not have O_DIRECT",
            )?;
            verified.insert(identity);
            count += 1;
        }
    }
    ensure(
        storage.config().use_direct_io
            && !storage.is_packed()
            && !expected.is_empty()
            && verified == expected,
        "cached O_DIRECT descriptor coverage incomplete",
    )?;
    Ok(DirectAudit { sampled_files: expected.len(), verified_files: verified.len(), verified_fds: count, method: "Linux /proc/self/fd device/inode coverage + fcntl(F_GETFL) O_DIRECT on actual cached data descriptors" })
}

#[cfg(not(target_os = "linux"))]
fn audit_direct(_storage: &NvmeStorage, _ids: &[u32]) -> io::Result<DirectAudit> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "authoritative O_DIRECT pread qualification requires Linux; portable buffered reads are tests only"))
}

#[derive(Serialize)]
struct Aggregate {
    #[serde(flatten)]
    arm: Arm,
    status: &'static str,
    unsupported_reason: Option<&'static str>,
    measured_repetitions: usize,
    mean_gb_s: Option<f64>,
    median_gb_s: Option<f64>,
    min_gb_s: Option<f64>,
    max_gb_s: Option<f64>,
}

fn aggregates(runs: &[RunResult]) -> Vec<Aggregate> {
    arms()
        .into_iter()
        .map(|arm| {
            let measured: Vec<_> = runs.iter().filter(|r| r.arm == arm && !r.warmup).collect();
            let valid =
                measured.len() == REPETITIONS && measured.iter().all(|r| r.status == "passed");
            let mut values: Vec<_> = measured.iter().filter_map(|r| r.decimal_gb_s).collect();
            values.sort_by(f64::total_cmp);
            Aggregate {
                arm,
                status: if arm.mode == Mode::IoUringFixed {
                    "unsupported"
                } else if valid {
                    "passed"
                } else {
                    "failed"
                },
                unsupported_reason: (arm.mode == Mode::IoUringFixed).then_some(IO_UNSUPPORTED),
                measured_repetitions: measured
                    .iter()
                    .filter(|r| r.status != "unsupported")
                    .count(),
                mean_gb_s: valid.then(|| values.iter().sum::<f64>() / values.len() as f64),
                median_gb_s: valid.then(|| values[values.len() / 2]),
                min_gb_s: valid.then(|| values[0]),
                max_gb_s: valid.then(|| values[values.len() - 1]),
            }
        })
        .collect()
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    diagnostic_only: bool,
    complete: bool,
    qualification_pass: bool,
    failure: Option<String>,
    provenance: Provenance,
    configuration: Configuration,
    input: InputEvidence,
    direct_io_verified: bool,
    runs: Vec<RunResult>,
    aggregates: Vec<Aggregate>,
    gates: Vec<Gate>,
}

fn finalize(report: &mut Report, execution_ok: bool) {
    let reference = report
        .runs
        .iter()
        .find(|r| {
            r.arm.mode == Mode::SequentialPread
                && !r.warmup
                && r.completed_reads == r.requested_reads
                && r.failure_events == 0
        })
        .and_then(|r| r.checksum.clone());
    for run in &mut report.runs {
        if run.status == "unsupported" {
            continue;
        }
        run.gates = run_gates(run, EXPERT_SIZE, reference.as_deref());
        run.status = if run.gates.iter().all(|g| g.pass) {
            "passed"
        } else {
            "failed"
        };
    }
    let active: Vec<_> = report
        .runs
        .iter()
        .filter(|r| r.status != "unsupported")
        .collect();
    report.direct_io_verified = !active.is_empty() && active.iter().all(|r| r.direct_io_verified);
    report.gates = vec![
        gate("preflight_and_execution", execution_ok),
        gate(
            "all_repetitions_present",
            report.runs.len() == arms().len() * (WARMUPS + REPETITIONS),
        ),
        gate(
            "all_supported_arms_pass",
            active.len() == 7 * (WARMUPS + REPETITIONS)
                && active.iter().all(|r| r.status == "passed"),
        ),
        gate(
            "identical_input_sequence",
            !report.runs.is_empty()
                && report.runs.iter().all(|r| {
                    Some(&r.sampled_id_sequence_sha256)
                        == report.input.sampled_id_sequence_sha256.as_ref()
                }),
        ),
        gate(
            "unsupported_arms_explicit_no_fallback",
            report
                .runs
                .iter()
                .filter(|r| r.status == "unsupported")
                .all(|r| {
                    r.arm.mode == Mode::IoUringFixed
                        && r.unsupported_reason == Some(IO_UNSUPPORTED)
                        && r.backend_used.is_none()
                        && r.completed_reads == 0
                        && r.batch_count == 0
                }),
        ),
    ];
    report.qualification_pass = report.gates.iter().all(|g| g.pass);
    report.complete = report.qualification_pass;
    if !report.qualification_pass && report.failure.is_none() {
        report.failure = Some("source throughput qualification gates failed; inspect per-run gates and failure_examples".into());
    }
    report.aggregates = aggregates(&report.runs);
}

async fn execute(args: &CommandArgs, report: &mut Report) -> io::Result<()> {
    ensure(
        (1..=MAX_READS).contains(&args.read_count),
        "read-count must be in 1..=1048576",
    )?;
    provenance(&mut report.provenance)?;
    let ids = load_input(args, &mut report.input)?;
    ensure(
        cfg!(target_os = "linux"),
        "authoritative O_DIRECT pread qualification requires Linux; no portable fallback",
    )?;
    let storage = Arc::new(
        NvmeStorage::new(StorageConfig {
            base_path: PathBuf::from(DATA_PATH),
            expert_size: EXPERT_SIZE,
            block_align: ALIGN,
            use_direct_io: true,
            num_experts_per_layer: Some(EXPERTS_PER_LAYER),
        })?
        .with_max_open_files(report.input.sampled_unique_ids),
    );
    // Check every sampled file without reading or interpreting a weight header.
    for id in ids.iter().copied().collect::<BTreeSet<_>>() {
        let path = storage.expert_path(id);
        let meta = std::fs::metadata(&path)?;
        ensure(
            meta.is_file() && meta.len() == EXPERT_SIZE as u64,
            &format!(
                "{} must be a regular file of exactly {EXPERT_SIZE} bytes",
                path.display()
            ),
        )?;
    }
    storage.warmup_fds(ids.iter().copied().collect::<BTreeSet<_>>())?;
    audit_direct(&storage, &ids)?;
    let pool = BenchPool::new(EXPERT_SIZE);
    for (warmup, repetition) in (0..WARMUPS)
        .map(|r| (true, r))
        .chain((0..REPETITIONS).map(|r| (false, r)))
    {
        let order = if warmup {
            arms()
        } else {
            run_order(repetition)
        };
        for (order_index, arm) in order.into_iter().enumerate() {
            eprintln!(
                "source-throughput warmup={warmup} rep={repetition} mode={:?} qd={}",
                arm.mode, arm.configured_qd
            );
            let before = if arm.mode != Mode::IoUringFixed {
                Some(audit_direct(&storage, &ids)?)
            } else {
                None
            };
            let mut result = measure(
                arm,
                storage.clone(),
                pool.clone(),
                ids.clone(),
                warmup,
                repetition,
                order_index,
            )
            .await;
            if let Some(before) = before {
                result.direct_fd_audits.push(before);
                match audit_direct(&storage, &ids) {
                    Ok(after) => {
                        result.direct_fd_audits.push(after);
                        result.direct_io_verified = true;
                    }
                    Err(error) => result.error(format!("post-read direct fd audit: {error}")),
                }
            }
            let released = result.pool_final_slots == 0 && result.tracked_pool_final_slots == 0;
            report.runs.push(result);
            ensure(
                released,
                "pool did not drain; refusing to start another arm",
            )?;
        }
    }
    Ok(())
}

pub(crate) fn run_command(args: &CommandArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Reserve output exclusively, so even an invalid input/output path pairing
    // cannot overwrite frozen evidence. Failure reports also use this handle.
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.report_out)?;
    let mut report = Report {
        schema: SCHEMA,
        diagnostic_only: true,
        complete: false,
        qualification_pass: false,
        failure: None,
        provenance: Provenance::default(),
        configuration: Configuration::new(args.read_count),
        input: InputEvidence {
            trace_path: args.oracle_trace.clone(),
            expected_trace_sha256: TRACE_SHA,
            ..InputEvidence::default()
        },
        direct_io_verified: false,
        runs: vec![],
        aggregates: vec![],
        gates: vec![],
    };
    let execution = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(MAX_QD)
        .enable_all()
        .build()
        .and_then(|rt| rt.block_on(execute(args, &mut report)));
    if let Err(error) = &execution {
        report.failure = Some(error.to_string());
    }
    finalize(&mut report, execution.is_ok());
    serde_json::to_writer_pretty(&mut output, &report)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    ensure(
        report.qualification_pass,
        report.failure.as_deref().unwrap_or("qualification failed"),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_native_oracle_routes::{OracleGeometry, OracleRouteRecord};

    fn trace() -> OracleRouteTrace {
        OracleRouteTrace::try_new(
            OracleGeometry {
                layers: 48,
                experts: 128,
                top_k: 8,
                d_model: 2048,
                d_ff: 768,
            },
            (0..2)
                .flat_map(|position| {
                    (0..48).map(move |layer_index| OracleRouteRecord {
                        position,
                        layer_index,
                        ordered_selected_expert_ids: vec![127, 0, 5, 3, 2, 1, 9, 8],
                    })
                })
                .collect(),
        )
        .unwrap()
    }

    struct Fixture {
        path: PathBuf,
        storage: Arc<NvmeStorage>,
    }
    impl Fixture {
        fn new() -> Self {
            static SERIAL: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "mer-source-throughput-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                SERIAL.fetch_add(1, Ordering::SeqCst)
            ));
            std::fs::create_dir(&path).unwrap();
            for id in [0u32, 1, 2, 128, 130, 6143] {
                let bytes: Vec<_> = (0..ALIGN)
                    .map(|offset| {
                        (u64::from(id) * 29 + offset as u64 * 17 + u64::from(id >> 8)) as u8
                    })
                    .collect();
                std::fs::write(
                    path.join(format!("expert_{}_{}.bin", id / 128, id % 128)),
                    bytes,
                )
                .unwrap();
            }
            let storage = Arc::new(
                NvmeStorage::new(StorageConfig {
                    base_path: path.clone(),
                    expert_size: ALIGN,
                    block_align: ALIGN,
                    use_direct_io: false,
                    num_experts_per_layer: Some(128),
                })
                .unwrap(),
            );
            Self { path, storage }
        }
        fn ids(&self) -> Arc<Vec<u32>> {
            Arc::new(
                [130, 0, 6143, 128, 130, 2, 1]
                    .into_iter()
                    .cycle()
                    .take(19)
                    .collect(),
            )
        }
        fn expected_checksum(&self, ids: &[u32]) -> String {
            let receipts: Vec<_> = ids
                .iter()
                .map(|id| {
                    let bytes = std::fs::read(self.storage.expert_path(*id)).unwrap();
                    Some(Receipt {
                        bytes: bytes.len(),
                        touch: touch(&bytes),
                    })
                })
                .collect();
            checksum(ids, &receipts).unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.path).unwrap();
        }
    }

    #[test]
    fn global_ids_preserve_layer_rank_repeats_and_cycle() {
        let trace = trace();
        let ids = flatten_sample(&trace, 800).unwrap();
        assert_eq!(&ids[..8], &[127, 0, 5, 3, 2, 1, 9, 8]);
        assert_eq!(&ids[8..16], &[255, 128, 133, 131, 130, 129, 137, 136]);
        assert_eq!(ids[47 * 8], 6143);
        assert_eq!(&ids[..384], &ids[384..768]);
        assert_eq!(&ids[..32], &ids[768..]);
        assert_eq!(
            ids.iter().map(|id| id / 128).collect::<BTreeSet<_>>().len(),
            48
        );
        assert!(flatten_sample(&trace, 0).is_err());
        assert!(flatten_sample(&trace, MAX_READS + 1).is_err());
    }

    #[test]
    fn trace_hash_and_parse_use_one_snapshot_and_reject_corruption() {
        let root = serde_json::json!({"schema": crate::gpu_native_oracle_routes::SCHEMA, "complete": true, "failure": null, "trace": trace()});
        let bytes = serde_json::to_vec(&root).unwrap();
        let hash = hex_sha(&bytes);
        assert_eq!(decode_trace(&bytes, &hash, 4096).unwrap().1.len(), 4096);
        assert!(decode_trace(&bytes, TRACE_SHA, 4096).is_err());
        let mut corrupt = root;
        corrupt["trace"]["records"][0]["ordered_selected_expert_ids"][0] = 128.into();
        let bytes = serde_json::to_vec(&corrupt).unwrap();
        assert!(decode_trace(&bytes, &hex_sha(&bytes), 4096).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn same_exact_sequence_bytes_and_checksum_for_every_pread_arm() {
        let fixture = Fixture::new();
        let ids = fixture.ids();
        let expected = fixture.expected_checksum(&ids);
        let pool = BenchPool::new(ALIGN);
        for arm in arms()
            .into_iter()
            .filter(|arm| arm.mode != Mode::IoUringFixed)
        {
            let run = measure(
                arm,
                fixture.storage.clone(),
                pool.clone(),
                ids.clone(),
                false,
                0,
                0,
            )
            .await;
            assert_eq!(run.completed_reads, ids.len());
            assert_eq!(run.failures, 0);
            assert_eq!(run.failure_events, 0, "{:?}", run.failure_examples);
            assert_eq!(run.bytes_completed, ids.len() as u64 * ALIGN as u64);
            assert_eq!(run.checksum.as_deref(), Some(expected.as_str()));
            assert_eq!(run.sampled_id_sequence_sha256, sequence_sha(&ids));
            assert_eq!(run.pool_final_slots, 0);
            assert_eq!(run.tracked_pool_final_slots, 0);
            assert!(run.pool_peak_slots <= arm.configured_qd);
            assert!(run.peak_outstanding_expert_requests <= arm.configured_qd);
            assert_eq!(run.final_outstanding_expert_requests, 0);
            assert!(run.actual_peak_inflight_reads.is_none());
            let batch_count = if arm.mode == Mode::BatchPread {
                ids.len().div_ceil(arm.configured_qd)
            } else {
                ids.len()
            };
            assert_eq!(run.batch_count, batch_count);
            assert!(
                run.batch_wall_latency.p99_us.unwrap() >= run.batch_wall_latency.p50_us.unwrap()
            );
            // Portable test reads must never pass the authoritative direct gate.
            assert!(!run.direct_io_verified);
            assert!(
                !run_gates(&run, ALIGN, Some(&expected))
                    .iter()
                    .find(|g| g.name == "o_direct_verified")
                    .unwrap()
                    .pass
            );
        }
        assert_eq!(pool.pool.primary_available(), MAX_QD);
    }

    #[test]
    fn batch_chunking_widths_have_exact_stream_and_bounded_tail() {
        let ids: Vec<u32> = (0..19).collect();
        for width in [2, 4, 8] {
            let chunks: Vec<_> = ids.chunks(width).collect();
            assert_eq!(chunks.len(), ids.len().div_ceil(width));
            assert!(chunks
                .iter()
                .all(|chunk| !chunk.is_empty() && chunk.len() <= width));
            assert_eq!(
                chunks
                    .iter()
                    .flat_map(|chunk| chunk.iter().copied())
                    .collect::<Vec<_>>(),
                ids
            );
            assert_eq!(chunks.last().unwrap().len(), if width == 2 { 1 } else { 3 });
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn read_failures_propagate_and_every_arm_releases_pool() {
        for arm in arms()
            .into_iter()
            .filter(|arm| arm.mode != Mode::IoUringFixed)
        {
            let fixture = Fixture::new();
            let mut ids = (*fixture.ids()).clone();
            ids[1] = 6000; // Missing file in the middle of a multi-file batch.
            let pool = BenchPool::new(ALIGN);
            let run = measure(
                arm,
                fixture.storage.clone(),
                pool.clone(),
                Arc::new(ids),
                false,
                0,
                0,
            )
            .await;
            assert!(run.failure_events > 0);
            assert!(run.failures > 0);
            assert_eq!(run.completed_reads + run.failures, run.requested_reads);
            assert!(run.completed_reads < run.requested_reads);
            assert_eq!(
                run.bytes_completed,
                run.completed_reads as u64 * ALIGN as u64
            );
            assert!(run.checksum.is_none());
            assert_eq!(run.pool_final_slots, 0);
            assert_eq!(run.tracked_pool_final_slots, 0);
            assert_eq!(pool.pool.primary_available(), MAX_QD);
            assert!(!run_gates(&run, ALIGN, None).iter().all(|g| g.pass));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn partial_batch_acquisition_error_returns_every_new_lease() {
        let fixture = Fixture::new();
        let pool = BenchPool::new(ALIGN);
        let held: Vec<_> = (0..MAX_QD - 1).map(|_| pool.acquire().unwrap()).collect();
        let inflight = Inflight::default();
        let outcome = batch_read(&fixture.storage, &pool, &inflight, 0, &[0, 1]).await;
        assert!(outcome.result.is_err());
        assert!(outcome.latency_us.is_none());
        assert_eq!(pool.final_slots(), MAX_QD - 1);
        assert_eq!(pool.counts.lock().current, MAX_QD - 1);
        assert_eq!(inflight.peak.load(Ordering::SeqCst), 0);
        drop(held);
        pool.reset().unwrap();
        assert_eq!(pool.final_slots(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn truncated_batch_is_failed_unconfirmed_and_drained() {
        let fixture = Fixture::new();
        std::fs::write(fixture.storage.expert_path(2), [1u8; 16]).unwrap();
        let pool = BenchPool::new(ALIGN);
        let outcome = batch_read(
            &fixture.storage,
            &pool,
            &Inflight::default(),
            0,
            &[0, 2, 1, 130],
        )
        .await;
        assert!(outcome.result.is_err());
        assert_eq!(pool.final_slots(), 0);
        assert_eq!(pool.counts.lock().current, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn reconciliation_rejects_bytes_checksum_qd_pool_and_backend_drift() {
        let fixture = Fixture::new();
        let ids = fixture.ids();
        let reference = fixture.expected_checksum(&ids);
        let mut run = measure(
            arms()[0],
            fixture.storage.clone(),
            BenchPool::new(ALIGN),
            ids,
            false,
            0,
            0,
        )
        .await;
        run.direct_io_verified = true; // Pure gate fault injection, not a qualification report.
        assert!(run_gates(&run, ALIGN, Some(&reference))
            .iter()
            .all(|g| g.pass));
        run.bytes_completed -= 1;
        assert!(!run_gates(&run, ALIGN, Some(&reference))[2].pass);
        run.bytes_completed += 1;
        assert!(!run_gates(&run, ALIGN, Some("wrong checksum"))[3].pass);
        run.pool_final_slots = 1;
        assert!(!run_gates(&run, ALIGN, Some(&reference))[4].pass);
        run.pool_final_slots = 0;
        run.peak_outstanding_expert_requests = 2;
        assert!(!run_gates(&run, ALIGN, Some(&reference))[5].pass);
        run.peak_outstanding_expert_requests = 1;
        run.backend_used = Some("io_uring");
        assert!(!run_gates(&run, ALIGN, Some(&reference))[7].pass);
    }

    #[test]
    fn checksum_is_logical_order_sensitive_and_includes_duplicate_accesses() {
        let receipt = Some(Receipt {
            bytes: ALIGN,
            touch: 42,
        });
        assert_ne!(
            checksum(&[0, 1, 0], &[receipt; 3]),
            checksum(&[1, 0, 0], &[receipt; 3])
        );
        assert_ne!(checksum(&[0, 0], &[receipt; 2]), checksum(&[0], &[receipt]));
        assert!(checksum(&[0, 1], &[receipt, None]).is_none());
        assert!(checksum(&[0], &[receipt; 2]).is_none());
        let mut bytes = vec![0u8; ALIGN * 2];
        let original = touch(&bytes);
        bytes[ALIGN] = 1;
        assert_ne!(original, touch(&bytes));
        bytes[ALIGN] = 0;
        bytes[ALIGN * 2 - 1] = 1;
        assert_ne!(original, touch(&bytes));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unsupported_fixed_arms_never_read_or_lease_or_fallback() {
        let fixture = Fixture::new();
        let pool = BenchPool::new(ALIGN);
        for arm in arms()
            .into_iter()
            .filter(|arm| arm.mode == Mode::IoUringFixed)
        {
            let run = measure(
                arm,
                fixture.storage.clone(),
                pool.clone(),
                Arc::new(vec![6000]),
                false,
                0,
                0,
            )
            .await;
            assert_eq!(run.status, "unsupported");
            assert_eq!(run.unsupported_reason, Some(IO_UNSUPPORTED));
            assert!(run.backend_used.is_none());
            assert_eq!(run.completed_reads, 0);
            assert_eq!(run.batch_count, 0);
            assert_eq!(run.pool_peak_slots, 0);
            assert!(run.wall_us.is_none());
            let json = serde_json::to_value(&run).unwrap();
            assert_eq!(json["mode"], "io-uring-fixed");
            assert_eq!(json["status"], "unsupported");
            assert!(json["unsupported_reason"]
                .as_str()
                .unwrap()
                .contains("no O_DIRECT"));
        }
        assert_eq!(pool.pool.primary_available(), MAX_QD);
    }

    #[test]
    fn run_order_is_deterministic_complete_reversed_then_rotated() {
        let original = arms();
        assert_eq!(run_order(0), original);
        assert_eq!(
            run_order(1),
            original.iter().copied().rev().collect::<Vec<_>>()
        );
        let mut rotated = original.clone();
        rotated.rotate_left(3);
        assert_eq!(run_order(2), rotated);
        for rep in 0..12 {
            assert_eq!(run_order(rep), run_order(rep % 3));
            for arm in &original {
                assert_eq!(run_order(rep).iter().filter(|a| *a == arm).count(), 1);
            }
        }
    }

    #[test]
    fn aggregates_exclude_warmup_and_failed_repetitions() {
        let arm = arms()[0];
        let mut runs: Vec<_> = [1000.0, 2.0, 4.0, 3.0]
            .into_iter()
            .enumerate()
            .map(|(index, speed)| {
                let mut r = RunResult::new(arm, index == 0, index.saturating_sub(1), 0, &[0]);
                r.status = "passed";
                r.decimal_gb_s = Some(speed);
                r
            })
            .collect();
        let a = &aggregates(&runs)[0];
        assert_eq!(a.measured_repetitions, 3);
        assert_eq!(
            (a.mean_gb_s, a.median_gb_s, a.min_gb_s, a.max_gb_s),
            (Some(3.0), Some(3.0), Some(2.0), Some(4.0))
        );
        runs[2].status = "failed";
        assert!(aggregates(&runs)[0].mean_gb_s.is_none());
        assert_eq!(aggregates(&runs)[7].status, "unsupported");
    }

    #[test]
    fn nearest_rank_latencies_are_deterministic() {
        let mut values: Vec<_> = (1..=100).rev().map(f64::from).collect();
        let stats = latency(&mut values);
        assert_eq!(
            (stats.p50_us, stats.p95_us, stats.p99_us, stats.mean_us),
            (Some(50.0), Some(95.0), Some(99.0), Some(50.5))
        );
        assert!(latency(&mut []).mean_us.is_none());
    }

    #[test]
    fn command_persists_failure_and_never_overwrites_existing_evidence() {
        let fixture = Fixture::new();
        let args = CommandArgs {
            oracle_trace: fixture.path.join("not-read.json"),
            read_count: 0,
            report_out: fixture.path.join("failure.json"),
        };
        assert!(run_command(&args).is_err());
        let bytes = std::fs::read(&args.report_out).unwrap();
        let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(report["schema"], SCHEMA);
        assert_eq!(report["complete"], false);
        assert_eq!(report["qualification_pass"], false);
        assert!(report["failure"].as_str().unwrap().contains("read-count"));
        assert_eq!(report["runs"], serde_json::json!([]));
        assert!(run_command(&args).is_err());
        assert_eq!(std::fs::read(&args.report_out).unwrap(), bytes);
    }

    #[test]
    fn cli_default_and_storage_only_dispatch_parse() {
        use clap::Parser;
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let cli = crate::Cli::try_parse_from([
                    "mer",
                    "qualify-expert-source-throughput",
                    "--oracle-trace",
                    "frozen.json",
                    "--report-out",
                    "new.json",
                ])
                .unwrap();
                let crate::Cmd::QualifyExpertSourceThroughput(args) = cli.cmd else {
                    panic!("wrong CLI command")
                };
                assert_eq!(args.read_count, 4096);
                assert_eq!(args.oracle_trace, PathBuf::from("frozen.json"));
            })
            .unwrap()
            .join()
            .unwrap();
    }
}
