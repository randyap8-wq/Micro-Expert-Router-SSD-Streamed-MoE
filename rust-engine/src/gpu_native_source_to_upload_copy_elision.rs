//! Standalone diagnostic: full expert pread into an aligned pool allocation or
//! directly into a mapped MAP_WRITE | COPY_SRC allocation. No inference runtime.
//! Hashes/readback/fd evidence are excluded from transfer-cycle and source timers.
use crate::buffer_pool::{BufferPool, PooledBuffer};
use crate::config::Config;
use crate::inference::WeightDtype;
use crate::io_provider::{NvmeStorage, StorageConfig};
use crate::tensor_header::{TensorHeader, UthDtypeId};
use futures::FutureExt;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::num::NonZeroU64;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

const SCHEMA: &str = "mer.gpu-native-mapped-memory-odirect-discriminator.v1";
const SOURCE_API: &str = "read_expert_into_aligned_slice";
const FULL: usize = 2_658_304;
const ALIGN: usize = 4096;
const PREFIX: usize = 4096;
const PAYLOAD: usize = 2_654_208;
const EPOCH_OFFSET: usize = 4;
const SLOT: usize = 2_654_212;
const UPLOAD: usize = FULL + ALIGN;
const NAMESPACE: u32 = 48 * 128;
const EPOCH: u32 = 0x1234_5678;
const MAX_ITERATIONS: usize = 65_536;
const GPU_TIMEOUT: Duration = Duration::from_secs(30);

type Result<T> = std::result::Result<T, Failure>;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Args {
    pub(crate) config: PathBuf,
    pub(crate) expected_adapter_name: String,
    pub(crate) warmup_iterations: usize,
    pub(crate) iterations: usize,
    pub(crate) report_out: PathBuf,
}

#[derive(Debug)]
struct Failure {
    classification: &'static str,
    detail: String,
    complete: bool,
}
impl Failure {
    fn runtime(classification: &'static str, detail: impl ToString) -> Self {
        Self {
            classification,
            detail: detail.to_string(),
            complete: false,
        }
    }
    fn authority(detail: impl ToString) -> Self {
        Self {
            classification: "authority-failed",
            detail: detail.to_string(),
            complete: true,
        }
    }
    fn accounting(detail: impl ToString) -> Self {
        Self::runtime("accounting-failed", detail)
    }
}

fn add(dst: &mut u64, n: u64) -> Result<()> {
    *dst = dst
        .checked_add(n)
        .ok_or_else(|| Failure::accounting("counter overflow"))?;
    Ok(())
}
fn elapsed(start: Instant) -> Result<u64> {
    u64::try_from(start.elapsed().as_nanos()).map_err(Failure::accounting)
}
fn timed(dst: &mut u64, start: Instant) -> Result<()> {
    add(dst, elapsed(start)?)
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn finish_sha(hash: &Sha256) -> String {
    format!("{:x}", hash.clone().finalize())
}
fn gbps(bytes: u64, ns: u64) -> Option<f64> {
    (ns > 0).then(|| bytes as f64 / ns as f64)
}

/// Address and offset arithmetic is checked independently of real GPU memory.
/// A page-aligned CPU address alone does not prove GPU copy-offset alignment.
fn aligned_subrange(
    base: usize,
    capacity: usize,
    len: usize,
    align: usize,
) -> std::result::Result<usize, String> {
    if base == 0 || !align.is_power_of_two() || len == 0 || len % align != 0 {
        return Err("invalid direct-read address, alignment, or length".into());
    }
    let offset = (align - base % align) % align;
    let pointer = base.checked_add(offset).ok_or("aligned address overflow")?;
    let end = offset.checked_add(len).ok_or("aligned range overflow")?;
    base.checked_add(end)
        .ok_or("mapped address range overflow")?;
    if pointer % align != 0 || end > capacity {
        return Err("aligned direct-read range is outside mapping".into());
    }
    Ok(offset)
}
fn copy_offsets(
    offset: usize,
    prefix: usize,
    payload: usize,
    capacity: usize,
) -> std::result::Result<u64, String> {
    let source = offset
        .checked_add(prefix)
        .ok_or("GPU source offset overflow")?;
    let end = source
        .checked_add(payload)
        .ok_or("GPU source end overflow")?;
    let dest_end = EPOCH_OFFSET
        .checked_add(payload)
        .ok_or("GPU destination end overflow")?;
    let alignment = wgpu::COPY_BUFFER_ALIGNMENT as usize;
    if source % alignment != 0
        || EPOCH_OFFSET % alignment != 0
        || payload % alignment != 0
        || payload == 0
        || end > capacity
        || dest_end > SLOT
    {
        return Err("GPU copy offset, length, or bounds contract unavailable".into());
    }
    u64::try_from(source).map_err(|e| e.to_string())
}
fn validate_geometry(cfg: &Config) -> Result<()> {
    let m = &cfg.model;
    let payload = m
        .d_model
        .checked_mul(m.d_ff)
        .and_then(|n| n.checked_mul(3))
        .filter(|n| n % 32 == 0)
        .and_then(|n| (n / 32).checked_mul(18));
    if (m.d_model, m.d_ff, m.num_layers, m.num_experts, m.top_k) != (2048, 768, 48, 128, 8)
        || m.dtype != WeightDtype::Q4_0
        || payload != Some(PAYLOAD)
        || m.expert_size != FULL
        || cfg.storage.block_align != ALIGN
        || PREFIX.checked_add(PAYLOAD) != Some(FULL)
        || EPOCH_OFFSET.checked_add(PAYLOAD) != Some(SLOT)
        || FULL % ALIGN != 0
    {
        return Err(Failure::authority("requires exact Qwen3-Coder Q4 geometry: 48x128 experts, 2048x768, full file 2658304, UTH prefix 4096, payload 2654208, slot 2654212"));
    }
    Ok(())
}
fn payload_range(source: &[u8]) -> std::result::Result<(usize, &[u8]), String> {
    let (header, payload) = TensorHeader::strip(source, ALIGN);
    let h = header.ok_or("missing or invalid UTH1 header")?;
    let prefix = source
        .len()
        .checked_sub(payload.len())
        .ok_or("payload offset underflow")?;
    if source.len() != FULL
        || prefix != PREFIX
        || payload.len() != PAYLOAD
        || h.dtype != UthDtypeId::Q4_0
        || h.shape_rank != 3
        || h.shape != [768, 2048, 3, 0]
        || h.quant_scale_count != 0
        || h.quant_scale_offset != 0
    {
        return Err("UTH/payload does not match authoritative full-file Q4 geometry".into());
    }
    Ok((prefix, payload))
}

/// Evenly spaced inclusive namespace samples; after one full namespace, repeat.
/// Encoding for the witness is concatenated u32 little-endian IDs, no framing.
fn expert_sequence(count: usize, total: u32) -> std::result::Result<Vec<u32>, String> {
    if total < 2 || count > MAX_ITERATIONS {
        return Err("invalid sequence bounds".into());
    }
    let span = count.min(total as usize);
    Ok((0..count)
        .map(|i| {
            if span == 1 {
                total / 2
            } else {
                ((i % span) as u64 * (total - 1) as u64 / (span - 1) as u64) as u32
            }
        })
        .collect())
}
fn sequence_sha(ids: &[u32]) -> String {
    let mut hash = Sha256::new();
    for id in ids {
        hash.update(id.to_le_bytes());
    }
    finish_sha(&hash)
}

#[derive(Default, Debug, Serialize)]
struct Times {
    map_wait_ns: u64,
    source_direct_read_ns: u64,
    alignment_setup_ns: u64,
    control_view_acquisition_ns: u64,
    control_cpu_payload_copy_ns: u64,
    control_staging_drop_scheduling_ns: u64,
    treatment_unmap_ns: u64,
    treatment_gpu_copy_encoding_ns: u64,
    epoch_write_ns: u64,
    submit_drain_ns: u64,
    transfer_cycle_ns: u64,
    verification_readback_ns: u64,
}
#[derive(Default, Debug, Serialize)]
struct PointerEvidence {
    observations: u64,
    aligned: u64,
    invalid: u64,
    mapping_base_mod_4096_counts: BTreeMap<usize, u64>,
    aligned_offset_min: Option<usize>,
    aligned_offset_max: Option<usize>,
    gpu_offset_checks: u64,
    gpu_offset_failures: u64,
}
impl PointerEvidence {
    fn observe(&mut self, base: usize, offset: usize, len: usize) -> Result<()> {
        add(&mut self.observations, 1)?;
        if (base + offset) % ALIGN == 0 && len == FULL && len % ALIGN == 0 {
            add(&mut self.aligned, 1)?;
        } else {
            add(&mut self.invalid, 1)?;
        }
        add(
            self.mapping_base_mod_4096_counts
                .entry(base % ALIGN)
                .or_default(),
            1,
        )?;
        self.aligned_offset_min = Some(self.aligned_offset_min.map_or(offset, |v| v.min(offset)));
        self.aligned_offset_max = Some(self.aligned_offset_max.map_or(offset, |v| v.max(offset)));
        Ok(())
    }
}
#[derive(Default, Debug, Serialize)]
struct FdEvidence {
    checks: u64,
    direct_observed: u64,
    full_file_length_observed: u64,
    flags_counts: BTreeMap<i32, u64>,
    failures: u64,
}
#[derive(Default, Debug, Serialize)]
struct Arm {
    ops_attempted: u64,
    source_read_attempts: u64,
    source_read_ops: u64,
    full_source_bytes: u64,
    payload_ops: u64,
    payload_bytes: u64,
    cpu_payload_copy_bytes: u64,
    upload_ops: u64,
    gpu_copied_bytes: u64,
    explicit_copy_buffer_bytes: u64,
    epoch_bytes: u64,
    gpu_completed_ops: u64,
    verified_ops: u64,
    verification_readback_bytes: u64,
    verification_destination_reset_ops: u64,
    verification_destination_reset_bytes: u64,
    map_attempts: u64,
    maps_completed: u64,
    unmaps: u64,
    alignment_failures: u64,
    mapped_direct_io_rejections: u64,
    rejection_errno_counts: BTreeMap<i32, u64>,
    map_failures: u64,
    source_failures: u64,
    gpu_failures: u64,
    accounting_failures: u64,
    exact_read_length_failures: u64,
    fallback_reads: u64,
    pointers: PointerEvidence,
    fd_evidence: FdEvidence,
    times: Times,
    source_gbps: Option<f64>,
    cpu_payload_copy_gbps: Option<f64>,
    submit_drain_payload_gbps: Option<f64>,
    transfer_cycle_payload_gbps: Option<f64>,
}
impl Arm {
    fn rates(&mut self) {
        self.source_gbps = gbps(self.full_source_bytes, self.times.source_direct_read_ns);
        self.cpu_payload_copy_gbps = gbps(
            self.cpu_payload_copy_bytes,
            self.times.control_cpu_payload_copy_ns,
        );
        self.submit_drain_payload_gbps = gbps(self.gpu_copied_bytes, self.times.submit_drain_ns);
        self.transfer_cycle_payload_gbps =
            gbps(self.gpu_copied_bytes, self.times.transfer_cycle_ns);
    }
    fn reconcile(&self, treatment: bool) -> bool {
        let expected = |ops: u64, bytes: usize| ops.checked_mul(bytes as u64);
        expected(self.source_read_ops, FULL) == Some(self.full_source_bytes)
            && expected(self.payload_ops, PAYLOAD) == Some(self.payload_bytes)
            && expected(self.upload_ops, PAYLOAD) == Some(self.gpu_copied_bytes)
            && expected(self.upload_ops, EPOCH_OFFSET) == Some(self.epoch_bytes)
            && expected(self.verified_ops, SLOT) == Some(self.verification_readback_bytes)
            && expected(self.verification_destination_reset_ops, SLOT)
                == Some(self.verification_destination_reset_bytes)
            && self.cpu_payload_copy_bytes == if treatment { 0 } else { self.gpu_copied_bytes }
            && self.explicit_copy_buffer_bytes == if treatment { self.gpu_copied_bytes } else { 0 }
            && self.source_read_ops <= self.source_read_attempts
            && self.source_read_attempts <= self.ops_attempted
            && self.payload_ops <= self.source_read_ops
            && self.upload_ops <= self.payload_ops
            && self.gpu_completed_ops <= self.upload_ops
            && self.verified_ops <= self.gpu_completed_ops
            && self.pointers.aligned == self.source_read_attempts
            && self.pointers.invalid == 0
            && self.exact_read_length_failures == 0
            && self.fallback_reads == 0
            && self.accounting_failures == 0
    }
    fn success(&self, expected: u64, treatment: bool) -> bool {
        self.reconcile(treatment)
            && self.ops_attempted == expected
            && self.verified_ops == expected
            && self.fd_evidence.checks == expected
            && self.fd_evidence.direct_observed == expected
            && self.fd_evidence.full_file_length_observed == expected
            && self.fd_evidence.failures == 0
            && self.verification_destination_reset_ops == expected
            && self.source_read_ops == expected
            && self.source_read_attempts == expected
            && self.source_failures == 0
            && self.gpu_failures == 0
            && self.map_failures == 0
            && self.alignment_failures == 0
            && self.mapped_direct_io_rejections == 0
            && (!treatment
                || (self.maps_completed == expected
                    && self.map_attempts == expected
                    && self.unmaps == expected
                    && self.pointers.gpu_offset_checks == expected
                    && self.pointers.gpu_offset_failures == 0))
    }
}

#[derive(Default)]
struct Streams {
    source: Sha256,
    payload: Sha256,
    gpu: Sha256,
}
#[derive(Default, Debug, Serialize)]
struct Witnesses {
    full_source_sha256: String,
    bare_payload_sha256: String,
    gpu_destination_payload_sha256: String,
}
impl Streams {
    fn snapshot(&self) -> Witnesses {
        Witnesses {
            full_source_sha256: finish_sha(&self.source),
            bare_payload_sha256: finish_sha(&self.payload),
            gpu_destination_payload_sha256: finish_sha(&self.gpu),
        }
    }
    fn source(&mut self, bytes: &[u8]) -> std::result::Result<(usize, Hashes), String> {
        let (offset, payload) = payload_range(bytes)?;
        self.source.update(bytes);
        self.payload.update(payload);
        Ok((
            offset,
            Hashes {
                source: sha(bytes),
                payload: sha(payload),
                gpu: String::new(),
                epoch: false,
            },
        ))
    }
}
#[derive(Debug)]
struct Hashes {
    source: String,
    payload: String,
    gpu: String,
    epoch: bool,
}
#[derive(Debug)]
enum Outcome {
    Verified(Hashes),
    MappedRejected,
    AlignmentUnavailable,
}
#[derive(Debug, Serialize)]
struct Mismatch {
    phase: &'static str,
    pair: usize,
    expert_id: u32,
    kind: &'static str,
    control: String,
    treatment: String,
}
/// Raw source durations are captured outside both arm calls, in pair order.
#[derive(Clone, Debug, Serialize)]
struct SourceReadPair {
    pair: usize,
    expert_id: u32,
    control_ns: u64,
    treatment_ns: u64,
}

#[derive(Default, Debug, PartialEq, Serialize)]
struct PairStats {
    paired_source_read_samples: u64,
    treatment_slower_pairs: u64,
    treatment_faster_pairs: u64,
    equal_pairs: u64,
    paired_control_source_read_ns: u64,
    paired_treatment_source_read_ns: u64,
    aggregate_treatment_minus_control_ns: i128,
    mean_treatment_minus_control_ns: Option<f64>,
    median_treatment_minus_control_ns: Option<f64>,
    median_treatment_over_control_ratio: Option<f64>,
}

fn source_duration(before: u64, after: u64) -> Result<u64> {
    after
        .checked_sub(before)
        .filter(|n| *n > 0)
        .ok_or_else(|| Failure::accounting("source duration underflow or zero"))
}

fn pair_stats(samples: &[SourceReadPair]) -> Result<PairStats> {
    if samples.len() > MAX_ITERATIONS {
        return Err(Failure::accounting("paired sample limit exceeded"));
    }
    let mut stats = PairStats::default();
    let mut deltas = Vec::with_capacity(samples.len());
    for sample in samples {
        if sample.control_ns == 0 || sample.treatment_ns == 0 {
            return Err(Failure::accounting("zero paired source duration"));
        }
        let delta = i128::from(sample.treatment_ns)
            .checked_sub(i128::from(sample.control_ns))
            .ok_or_else(|| Failure::accounting("paired delta overflow"))?;
        stats.aggregate_treatment_minus_control_ns = stats
            .aggregate_treatment_minus_control_ns
            .checked_add(delta)
            .ok_or_else(|| Failure::accounting("paired aggregate overflow"))?;
        add(&mut stats.paired_source_read_samples, 1)?;
        add(&mut stats.paired_control_source_read_ns, sample.control_ns)?;
        add(
            &mut stats.paired_treatment_source_read_ns,
            sample.treatment_ns,
        )?;
        add(
            if delta > 0 {
                &mut stats.treatment_slower_pairs
            } else if delta < 0 {
                &mut stats.treatment_faster_pairs
            } else {
                &mut stats.equal_pairs
            },
            1,
        )?;
        deltas.push(delta);
    }
    if stats
        .treatment_slower_pairs
        .checked_add(stats.treatment_faster_pairs)
        .and_then(|n| n.checked_add(stats.equal_pairs))
        != Some(stats.paired_source_read_samples)
        || i128::from(stats.paired_treatment_source_read_ns)
            .checked_sub(i128::from(stats.paired_control_source_read_ns))
            != Some(stats.aggregate_treatment_minus_control_ns)
    {
        return Err(Failure::accounting("paired counters do not reconcile"));
    }
    if samples.is_empty() {
        return Ok(stats);
    }
    stats.mean_treatment_minus_control_ns = Some(
        stats.aggregate_treatment_minus_control_ns as f64 / stats.paired_source_read_samples as f64,
    );
    deltas.sort_unstable();
    let hi = samples.len() / 2;
    let lo = (samples.len() - 1) / 2;
    // For odd counts lo==hi; doubling and halving yields the middle value.
    stats.median_treatment_minus_control_ns = Some(
        deltas[lo]
            .checked_add(deltas[hi])
            .ok_or_else(|| Failure::accounting("paired median overflow"))? as f64
            / 2.0,
    );
    let mut ratios: Vec<_> = samples.iter().collect();
    ratios.sort_unstable_by(|a, b| {
        // A product of two u64 values fits u128 exactly, including u64::MAX.
        (u128::from(a.treatment_ns) * u128::from(b.control_ns))
            .cmp(&(u128::from(b.treatment_ns) * u128::from(a.control_ns)))
            // Equal fractions can round differently when large integer operands
            // convert to f64. Canonicalize their representation before selection.
            .then_with(|| a.control_ns.cmp(&b.control_ns))
    });
    let ratio = |i: usize| ratios[i].treatment_ns as f64 / ratios[i].control_ns as f64;
    stats.median_treatment_over_control_ratio = Some(ratio(lo) / 2.0 + ratio(hi) / 2.0);
    Ok(stats)
}

#[derive(Debug, Serialize)]
struct Phase {
    name: &'static str,
    ordered_expert_ids: Vec<u32>,
    expert_id_sequence_sha256: String,
    attempted_expert_id_sequence_sha256: String,
    pairs_attempted: u64,
    pairs_completed: u64,
    successful_pairs_completed: u64,
    source_read_pairs: Vec<SourceReadPair>,
    #[serde(flatten)]
    paired_source_read_stats: PairStats,
    control: Arm,
    treatment: Arm,
    control_witnesses: Witnesses,
    treatment_witnesses: Witnesses,
    mismatch_count: u64,
    first_mismatch: Option<Mismatch>,
    first_mechanism_rejection: Option<String>,
}
impl Phase {
    fn new(name: &'static str, ids: Vec<u32>) -> Self {
        Self {
            name,
            expert_id_sequence_sha256: sequence_sha(&ids),
            ordered_expert_ids: ids,
            attempted_expert_id_sequence_sha256: sequence_sha(&[]),
            pairs_attempted: 0,
            pairs_completed: 0,
            successful_pairs_completed: 0,
            source_read_pairs: Vec::new(),
            paired_source_read_stats: PairStats::default(),
            control: Arm::default(),
            treatment: Arm::default(),
            control_witnesses: Witnesses::default(),
            treatment_witnesses: Witnesses::default(),
            mismatch_count: 0,
            first_mismatch: None,
            first_mechanism_rejection: None,
        }
    }
    fn mismatch(
        &mut self,
        pair: usize,
        id: u32,
        kind: &'static str,
        control: &str,
        treatment: &str,
    ) -> Result<()> {
        if control != treatment {
            add(&mut self.mismatch_count, 1)?;
            self.first_mismatch.get_or_insert_with(|| Mismatch {
                phase: self.name,
                pair,
                expert_id: id,
                kind,
                control: control.into(),
                treatment: treatment.into(),
            });
        }
        Ok(())
    }
    fn compare(
        &mut self,
        pair: usize,
        id: u32,
        control: &Outcome,
        treatment: &Outcome,
    ) -> Result<()> {
        if let Outcome::Verified(c) = control {
            self.mismatch(pair, id, "control-gpu-payload", &c.payload, &c.gpu)?;
            self.mismatch(
                pair,
                id,
                "control-epoch",
                "true",
                if c.epoch { "true" } else { "false" },
            )?;
        }
        if let Outcome::Verified(t) = treatment {
            self.mismatch(pair, id, "treatment-gpu-payload", &t.payload, &t.gpu)?;
            self.mismatch(
                pair,
                id,
                "treatment-epoch",
                "true",
                if t.epoch { "true" } else { "false" },
            )?;
        }
        if let (Outcome::Verified(c), Outcome::Verified(t)) = (control, treatment) {
            self.mismatch(pair, id, "full-source", &c.source, &t.source)?;
            self.mismatch(pair, id, "bare-payload", &c.payload, &t.payload)?;
            self.mismatch(pair, id, "gpu-destination-payload", &c.gpu, &t.gpu)?;
        }
        Ok(())
    }
    fn parity(&self) -> bool {
        let c = &self.control_witnesses;
        let t = &self.treatment_witnesses;
        self.mismatch_count == 0
            && c.full_source_sha256 == t.full_source_sha256
            && c.bare_payload_sha256 == t.bare_payload_sha256
            && c.bare_payload_sha256 == c.gpu_destination_payload_sha256
            && c.bare_payload_sha256 == t.gpu_destination_payload_sha256
    }
    fn finish_pair(
        &mut self,
        pair: usize,
        id: u32,
        c: &Outcome,
        t: &Outcome,
        control_before: u64,
        treatment_before: u64,
    ) -> Result<()> {
        let mismatches_before = self.mismatch_count;
        self.compare(pair, id, c, t)?;
        if matches!((c, t), (Outcome::Verified(_), Outcome::Verified(_)))
            && self.mismatch_count == mismatches_before
        {
            let sample = SourceReadPair {
                pair,
                expert_id: id,
                control_ns: source_duration(
                    control_before,
                    self.control.times.source_direct_read_ns,
                )?,
                treatment_ns: source_duration(
                    treatment_before,
                    self.treatment.times.source_direct_read_ns,
                )?,
            };
            add(&mut self.successful_pairs_completed, 1)?;
            self.source_read_pairs.push(sample);
        }
        add(&mut self.pairs_completed, 1)?;
        Ok(())
    }
    fn pairs_accounted(&self) -> bool {
        let stats = &self.paired_source_read_stats;
        pair_stats(&self.source_read_pairs).is_ok_and(|expected| expected == *stats)
            && stats.paired_source_read_samples == self.successful_pairs_completed
            && self.successful_pairs_completed <= self.pairs_completed
            && self.successful_pairs_completed <= self.control.verified_ops
            && self.successful_pairs_completed <= self.treatment.verified_ops
            && stats.paired_control_source_read_ns <= self.control.times.source_direct_read_ns
            && stats.paired_treatment_source_read_ns <= self.treatment.times.source_direct_read_ns
            && self.source_read_pairs.iter().all(|s| {
                u64::try_from(s.pair).is_ok_and(|i| i < self.pairs_completed)
                    && self.ordered_expert_ids.get(s.pair) == Some(&s.expert_id)
            })
            && self
                .source_read_pairs
                .windows(2)
                .all(|p| p[0].pair < p[1].pair)
    }
    fn accounted(&self) -> bool {
        self.pairs_accounted()
            && self.pairs_completed == self.ordered_expert_ids.len() as u64
            && self.pairs_attempted == self.pairs_completed
            && self.attempted_expert_id_sequence_sha256 == self.expert_id_sequence_sha256
            && self.control.reconcile(false)
            && self.treatment.reconcile(true)
    }
    fn successful(&self) -> bool {
        let n = self.ordered_expert_ids.len() as u64;
        self.accounted()
            && self.successful_pairs_completed == n
            && self.paired_source_read_stats.paired_control_source_read_ns
                == self.control.times.source_direct_read_ns
            && self
                .paired_source_read_stats
                .paired_treatment_source_read_ns
                == self.treatment.times.source_direct_read_ns
            && self.control.success(n, false)
            && self.treatment.success(n, true)
            && self.parity()
    }
}

#[derive(Default, Debug, Serialize)]
struct Authority {
    control_source_api: &'static str,
    treatment_source_api: &'static str,
    same_source_api: bool,
    control_destination: &'static str,
    treatment_destination: &'static str,
    source_timer_excludes_allocation: bool,
    source_timer_excludes_map_async_device_poll: bool,
    source_timer_excludes_alignment_setup: bool,
    source_timer_excludes_hashes_readback_fd_evidence: bool,
    source_timer_excludes_gpu_copy_unmap: bool,
    linux: bool,
    expected_adapter_name: String,
    adapter_name: Option<String>,
    adapter_backend: Option<String>,
    adapter_device_type: Option<String>,
    adapter_vendor: Option<u32>,
    adapter_device: Option<u32>,
    driver: Option<String>,
    driver_info: Option<String>,
    adapter_authoritative: bool,
    direct_io_requested: bool,
    packed_storage: Option<bool>,
    source_data_dir: Option<PathBuf>,
    exact_geometry: bool,
    full_source_bytes: usize,
    block_alignment: usize,
    uth_prefix_bytes: usize,
    bare_payload_bytes: usize,
    physical_slot_bytes: usize,
    upload_capacity_bytes: usize,
}
#[derive(Debug, Serialize)]
struct Report {
    schema: &'static str,
    args: Args,
    config_sha256: Option<String>,
    complete: bool,
    correctness_pass: bool,
    classification: String,
    failure: Option<String>,
    runtime_failures: u64,
    accounting_failures: u64,
    authority: Authority,
    warmup: Phase,
    measured: Phase,
    source_throughput_ratio_treatment_over_control: Option<f64>,
    source_read_slowdown_percent_treatment_vs_control: Option<f64>,
    performance_required_for_correctness: bool,
    total_cycle_superiority_required: bool,
    sequence_contract: &'static str,
    timing_contract: &'static str,
    paired_timing_contract: &'static str,
    interpretation_contract: &'static str,
    odirect_evidence_contract: &'static str,
    cpu_copy_accounting_contract: &'static str,
}
impl Report {
    fn new(args: Args) -> Self {
        Self { schema: SCHEMA, authority: Authority {
                control_source_api: SOURCE_API, treatment_source_api: SOURCE_API,
                same_source_api: true, control_destination: "aligned-host-pool",
                treatment_destination: "wgpu-map-write",
                source_timer_excludes_allocation: true,
                source_timer_excludes_map_async_device_poll: true,
                source_timer_excludes_alignment_setup: true,
                source_timer_excludes_hashes_readback_fd_evidence: true,
                source_timer_excludes_gpu_copy_unmap: true,
                linux: cfg!(target_os = "linux"), expected_adapter_name: args.expected_adapter_name.clone(),
                full_source_bytes: FULL, block_alignment: ALIGN, uth_prefix_bytes: PREFIX,
                bare_payload_bytes: PAYLOAD, physical_slot_bytes: SLOT, upload_capacity_bytes: UPLOAD,
                ..Authority::default()
            }, args, config_sha256: None, complete: false, correctness_pass: false,
            classification: "not-completed".into(), failure: None, runtime_failures: 0, accounting_failures: 0,
            warmup: Phase::new("warmup", vec![]), measured: Phase::new("measured", vec![]),
            source_throughput_ratio_treatment_over_control: None, source_read_slowdown_percent_treatment_vs_control: None,
            performance_required_for_correctness: false,
            total_cycle_superiority_required: false,
            sequence_contract: "global ID = layer*128+local; span=min(count,6144); ID[i]=floor((i%span)*6143/(span-1)); singleton warmup=3072; SHA256(concatenated u32 LE IDs). Even pair CONTROL,TREATMENT; odd pair TREATMENT,CONTROL; parity hashes concatenate exact source/payload bytes in pair order, without framing. Warmup is separate and excluded.",
            timing_contract: "nanoseconds, decimal GB/s=bytes/ns. Both source timers cover only read_expert_into_aligned_slice (cached fd lookup, block_in_place, pread and unchanged retries/breakers). Each arm's total cycle is its wall time minus fd evidence, source/header hashing and GPU readback verification. Each standalone destination is GPU-cleared before its arm, outside transfer timers, to prevent stale paired data from satisfying parity. Source timers exclude allocation, map_async/device.poll, alignment setup, hashes/readback/fd evidence, GPU copy and unmap. Performance does not determine correctness. submit/drain includes queued epoch and payload; it is descriptive, not isolated DMA bandwidth.",
            paired_timing_contract: "Each arm runs exactly once per pair. Take checked deltas of its existing source_direct_read_ns accumulator outside the arm calls; only pairs with two verified, matching outcomes enter source_read_pairs. pairs_completed retains completed negative outcomes; successful_pairs_completed counts matching verified pairs. Warmup statistics are separate. Signed deltas and sums use checked i128 arithmetic, arm totals/counts use checked u64 arithmetic. Mean=sum/count. Odd median=middle; even median=arithmetic mean of the middle two. Ratios are sorted by exact u128 cross products before conversion. Reported floating means/medians/ratios are rounded f64 summaries; raw integer durations and aggregate delta retain exact evidence. Zero durations fail accounting. No performance decision is made here.",
            interpretation_contract: ">=5% treatment slowdown + median delta >0 + majority slower pairs: strong mapped-substrate evidence. >=3% slowdown + median delta >0 + majority slower pairs: material mapped-substrate evidence. Within +/-1% with median near zero: evidence against singleton substrate penalty; next HMA-1C-B batch test. 1-3%, or aggregate/paired direction disagreement: ambiguous; next HMA-1C-B. Treatment speedup: evidence against raw singleton WGPU backing as cause; next batch/surrounding-helper discriminator. Apply after authoritative FIRST; slowdown is 100*(treatment source ns/control source ns-1), not throughput loss. No near-zero tolerance or FIRST iteration count is selected by this diagnostic.",
            odirect_evidence_contract: "Before each arm: Linux fcntl(F_GETFL) on the actual cached expert fd plus fstat file length. Exclusively owned sequential storage retains that same fd through the read; no packed storage, no fallback reads, no dense tensors or inference.",
            cpu_copy_accounting_contract: "CONTROL counts exact bytes passed to QueueWriteBufferView::copy_from_slice; TREATMENT reads the entire source file directly into BufferViewMut, performs no CPU payload copy, then unmaps and encodes the bare payload GPU copy. Hashing/readback are excluded verification work. gpu_copied_bytes excludes the separately counted 4-byte epoch and readback bytes.",
        }
    }
    fn authoritative(&self) -> bool {
        let a = &self.authority;
        a.control_source_api == SOURCE_API
            && a.treatment_source_api == SOURCE_API
            && a.same_source_api
            && a.control_destination == "aligned-host-pool"
            && a.treatment_destination == "wgpu-map-write"
            && a.source_timer_excludes_allocation
            && a.source_timer_excludes_map_async_device_poll
            && a.source_timer_excludes_alignment_setup
            && a.source_timer_excludes_hashes_readback_fd_evidence
            && a.source_timer_excludes_gpu_copy_unmap
            && a.linux
            && a.expected_adapter_name == "NVIDIA L4"
            && a.adapter_authoritative
            && a.direct_io_requested
            && a.packed_storage == Some(false)
            && a.exact_geometry
    }
    fn fail(&mut self, failure: Failure) {
        self.complete = failure.complete;
        self.correctness_pass = false;
        self.classification = failure.classification.into();
        self.failure = Some(failure.detail);
        if !failure.complete {
            self.runtime_failures += 1;
        }
        if failure.classification == "accounting-failed" {
            self.accounting_failures += 1;
        }
    }
    fn classify(&mut self) {
        self.warmup.control.rates();
        self.warmup.treatment.rates();
        self.measured.control.rates();
        self.measured.treatment.rates();
        self.complete = true;
        self.correctness_pass = false;
        if !self.authoritative() {
            self.classification = "authority-failed".into();
            return;
        }
        if !self.warmup.accounted() || !self.measured.accounted() || self.accounting_failures != 0 {
            self.fail(Failure::accounting(
                "phase/byte/operation/sequence reconciliation failed",
            ));
            return;
        }
        let phases = [&self.warmup, &self.measured];
        if phases.iter().any(|p| p.mismatch_count > 0) {
            self.classification = if phases.iter().any(|p| {
                p.first_mismatch
                    .as_ref()
                    .is_some_and(|m| matches!(m.kind, "full-source" | "bare-payload"))
            }) {
                "source-parity-failed"
            } else {
                "gpu-copy-parity-failed"
            }
            .into();
            return;
        }
        let controls_ok = phases
            .iter()
            .all(|p| p.control.success(p.ordered_expert_ids.len() as u64, false));
        let rejected = phases.iter().all(|p| {
            let t = &p.treatment;
            let n = p.ordered_expert_ids.len() as u64;
            t.ops_attempted == n
                && t.source_read_attempts == n
                && t.mapped_direct_io_rejections == n
                && t.source_failures == n
                && t.source_read_ops == 0
                && t.upload_ops == 0
                && t.map_attempts == n
                && t.maps_completed == n
                && t.unmaps == n
                && t.verification_destination_reset_ops == n
                && t.fd_evidence.direct_observed == n
                && t.fd_evidence.full_file_length_observed == n
                && t.map_failures == 0
                && t.gpu_failures == 0
                && t.alignment_failures == 0
                && t.rejection_errno_counts.values().sum::<u64>() == n
                && t.rejection_errno_counts
                    .keys()
                    .all(|e| mapped_rejection(Some(*e)))
        });
        if controls_ok && rejected && self.measured.treatment.mapped_direct_io_rejections >= 2 {
            self.classification = "mapped-upload-direct-io-rejected".into();
            return;
        }
        if controls_ok
            && phases.iter().any(|p| p.treatment.alignment_failures > 0)
            && phases.iter().all(|p| {
                p.treatment.source_failures == 0
                    && p.treatment.map_failures == 0
                    && p.treatment.gpu_failures == 0
            })
        {
            self.classification = "alignment-contract-unavailable".into();
            return;
        }
        if !self.warmup.successful() || !self.measured.successful() || self.runtime_failures != 0 {
            self.fail(Failure::runtime("runtime-failed", "incomplete or inconsistent arm evidence; mapped I/O rejection requires every attempted treatment read to reject and every CONTROL to work"));
            return;
        }
        let c = &self.measured.control;
        let t = &self.measured.treatment;
        let ratio = match (c.source_gbps, t.source_gbps) {
            (Some(c), Some(t)) if c > 0.0 && c.is_finite() && t.is_finite() => t / c,
            _ => {
                self.fail(Failure::accounting(
                    "missing or invalid source-only throughput",
                ));
                return;
            }
        };
        self.source_throughput_ratio_treatment_over_control = Some(ratio);
        self.source_read_slowdown_percent_treatment_vs_control = Some(
            self.measured
                .paired_source_read_stats
                .aggregate_treatment_minus_control_ns as f64
                / c.times.source_direct_read_ns as f64
                * 100.0,
        );
        self.correctness_pass = true;
        self.classification = "mapped-memory-discriminator-complete".into();
    }
}
fn mapped_rejection(errno: Option<i32>) -> bool {
    matches!(errno, Some(libc::EINVAL) | Some(libc::EFAULT))
}

/// All buffers and callbacks belong exclusively to this diagnostic.
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    upload: wgpu::Buffer,
    destination: wgpu::Buffer,
    readback: wgpu::Buffer,
    errors: Arc<Mutex<Option<String>>>,
}
impl Gpu {
    async fn new(a: &mut Authority) -> Result<Self> {
        Self::with_upload_capacity(a, UPLOAD).await
    }
    async fn with_upload_capacity(a: &mut Authority, upload_capacity: usize) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .into_iter()
            .find(|adapter| {
                let info = adapter.get_info();
                info.name == "NVIDIA L4"
                    && info.backend == wgpu::Backend::Vulkan
                    && info.device_type == wgpu::DeviceType::DiscreteGpu
            })
            .ok_or_else(|| {
                Failure::authority("exact discrete NVIDIA L4 Vulkan adapter unavailable")
            })?;
        let info = adapter.get_info();
        a.adapter_name = Some(info.name.clone());
        a.adapter_backend = Some(format!("{:?}", info.backend));
        a.adapter_device_type = Some(format!("{:?}", info.device_type));
        a.adapter_vendor = Some(info.vendor);
        a.adapter_device = Some(info.device);
        a.driver = Some(info.driver);
        a.driver_info = Some(info.driver_info);
        a.adapter_authoritative = a.linux
            && info.name == a.expected_adapter_name
            && info.name == "NVIDIA L4"
            && info.backend == wgpu::Backend::Vulkan
            && info.device_type == wgpu::DeviceType::DiscreteGpu;
        if !a.adapter_authoritative {
            return Err(Failure::authority("adapter authority mismatch"));
        }
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("source-to-upload-diagnostic"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .map_err(|e| Failure::runtime("gpu-failed", e))?;
        let errors = Arc::new(Mutex::new(None));
        let uncaptured = errors.clone();
        device.on_uncaptured_error(Box::new(move |error| {
            uncaptured
                .lock()
                .unwrap()
                .get_or_insert_with(|| format!("wgpu: {error}"));
        }));
        let lost = errors.clone();
        device.set_device_lost_callback(move |reason, message| {
            lost.lock()
                .unwrap()
                .get_or_insert_with(|| format!("device lost {reason:?}: {message}"));
        });
        let upload = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mapped-direct-source-upload"),
            size: upload_capacity as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let destination = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("standalone-physical-slot"),
            size: SLOT as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("verification-only-slot-readback"),
            size: SLOT as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gpu = Self {
            device,
            queue,
            upload,
            destination,
            readback,
            errors,
        };
        gpu.check()?;
        Ok(gpu)
    }
    fn check(&self) -> Result<()> {
        match self
            .errors
            .lock()
            .map_err(|e| Failure::runtime("gpu-failed", e))?
            .as_ref()
        {
            Some(e) => Err(Failure::runtime("gpu-failed", e)),
            None => Ok(()),
        }
    }
    fn wait<T>(&self, rx: &mpsc::Receiver<T>) -> Result<T> {
        let start = Instant::now();
        loop {
            self.device.poll(wgpu::Maintain::Poll);
            self.check()?;
            match rx.recv_timeout(Duration::from_millis(1)) {
                Ok(value) => return Ok(value),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(Failure::runtime(
                        "gpu-failed",
                        "GPU callback channel disconnected",
                    ))
                }
                Err(mpsc::RecvTimeoutError::Timeout) if start.elapsed() >= GPU_TIMEOUT => {
                    return Err(Failure::runtime(
                        "gpu-failed",
                        "GPU callback timed out after 30 seconds",
                    ))
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }
    fn map(&self, buffer: &wgpu::Buffer, mode: wgpu::MapMode) -> Result<()> {
        let (tx, rx) = mpsc::channel();
        buffer.slice(..).map_async(mode, move |r| {
            let _ = tx.send(r);
        });
        match self
            .wait(&rx)
            .and_then(|r| r.map_err(|e| Failure::runtime("map-failed", e)))
        {
            Ok(()) => Ok(()),
            Err(e) => {
                buffer.unmap();
                Err(e)
            }
        }
    }
    fn drain(&self, command: Option<wgpu::CommandBuffer>) -> Result<()> {
        self.queue.submit(command);
        let (tx, rx) = mpsc::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = tx.send(());
        });
        self.wait(&rx)
    }
    fn verify(&self, hashes: &mut Hashes, stream: &mut Streams) -> Result<()> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("verification-only-readback"),
            });
        encoder.copy_buffer_to_buffer(&self.destination, 0, &self.readback, 0, SLOT as u64);
        self.drain(Some(encoder.finish()))?;
        self.map(&self.readback, wgpu::MapMode::Read)?;
        {
            let view = self.readback.slice(..).get_mapped_range();
            hashes.epoch = view[..EPOCH_OFFSET] == EPOCH.to_le_bytes();
            hashes.gpu = sha(&view[EPOCH_OFFSET..]);
            stream.gpu.update(&view[EPOCH_OFFSET..]);
        }
        self.readback.unmap();
        self.check()
    }

    fn reset_destination(&self) -> Result<()> {
        // Verification preparation prevents an even pair's TREATMENT from
        // passing via the preceding CONTROL payload left in the shared slot.
        // This private GPU clear is outside transfer timers and CPU-copy bytes.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("verification-only-destination-reset"),
            });
        encoder.clear_buffer(&self.destination, 0, None);
        self.drain(Some(encoder.finish()))
    }
}

fn prepare_destination(gpu: &Gpu, arm: &mut Arm) -> Result<()> {
    let start = Instant::now();
    let result = gpu.reset_destination();
    timed(&mut arm.times.verification_readback_ns, start)?;
    if result.is_err() {
        add(&mut arm.gpu_failures, 1)?;
    }
    result?;
    add(&mut arm.verification_destination_reset_ops, 1)?;
    add(&mut arm.verification_destination_reset_bytes, SLOT as u64)
}

fn fd_evidence(storage: &NvmeStorage, id: u32, arm: &mut Arm) -> Result<()> {
    let start = Instant::now();
    let evidence = storage.source_to_upload_fd_evidence(id);
    timed(&mut arm.times.verification_readback_ns, start)?;
    add(&mut arm.fd_evidence.checks, 1)?;
    let (flags, direct, len) = evidence.map_err(|e| {
        arm.fd_evidence.failures += 1;
        Failure::runtime("source-failed", format!("fd evidence for expert {id}: {e}"))
    })?;
    add(arm.fd_evidence.flags_counts.entry(flags).or_default(), 1)?;
    if direct {
        add(&mut arm.fd_evidence.direct_observed, 1)?;
    }
    if len == FULL as u64 {
        add(&mut arm.fd_evidence.full_file_length_observed, 1)?;
    }
    if !direct || len != FULL as u64 {
        return Err(Failure::authority(format!(
            "expert {id}: actual fd O_DIRECT={direct}, full file length={len}, flags={flags}"
        )));
    }
    Ok(())
}
fn read_result(
    result: io::Result<usize>,
    treatment: bool,
    arm: &mut Arm,
    id: u32,
    rejection: &mut Option<String>,
) -> Result<bool> {
    match result {
        Ok(n) if n == FULL => {
            add(&mut arm.source_read_ops, 1)?;
            add(&mut arm.full_source_bytes, n as u64)?;
            Ok(true)
        }
        Ok(n) => {
            add(&mut arm.exact_read_length_failures, 1)?;
            Err(Failure::accounting(format!(
                "expert {id}: read length {n}, expected {FULL}"
            )))
        }
        Err(e) => {
            add(&mut arm.source_failures, 1)?;
            if e.kind() == io::ErrorKind::UnexpectedEof {
                add(&mut arm.exact_read_length_failures, 1)?;
            }
            if treatment && mapped_rejection(e.raw_os_error()) {
                add(&mut arm.mapped_direct_io_rejections, 1)?;
                add(
                    arm.rejection_errno_counts
                        .entry(e.raw_os_error().unwrap())
                        .or_default(),
                    1,
                )?;
                rejection.get_or_insert_with(|| {
                    format!(
                        "expert {id}: mapped full-file pread rejected: {e}; errno={:?}",
                        e.raw_os_error()
                    )
                });
                Ok(false)
            } else {
                Err(Failure::runtime(
                    "source-failed",
                    format!("expert {id}: {e}; errno={:?}", e.raw_os_error()),
                ))
            }
        }
    }
}
fn note_payload(arm: &mut Arm) -> Result<()> {
    add(&mut arm.payload_ops, 1)?;
    add(&mut arm.payload_bytes, PAYLOAD as u64)
}
fn note_upload(arm: &mut Arm, treatment: bool) -> Result<()> {
    add(&mut arm.upload_ops, 1)?;
    add(&mut arm.gpu_copied_bytes, PAYLOAD as u64)?;
    add(&mut arm.epoch_bytes, EPOCH_OFFSET as u64)?;
    if treatment {
        add(&mut arm.explicit_copy_buffer_bytes, PAYLOAD as u64)?;
    }
    Ok(())
}
fn verify(gpu: &Gpu, arm: &mut Arm, hashes: &mut Hashes, streams: &mut Streams) -> Result<()> {
    let start = Instant::now();
    let result = gpu.verify(hashes, streams);
    timed(&mut arm.times.verification_readback_ns, start)?;
    if result.is_err() {
        add(&mut arm.gpu_failures, 1)?;
    }
    result?;
    add(&mut arm.verified_ops, 1)?;
    add(&mut arm.verification_readback_bytes, SLOT as u64)
}

async fn control(
    gpu: &Gpu,
    storage: &NvmeStorage,
    buf: &mut PooledBuffer,
    id: u32,
    arm: &mut Arm,
    streams: &mut Streams,
) -> Result<Outcome> {
    add(&mut arm.ops_attempted, 1)?;
    fd_evidence(storage, id, arm)?;
    prepare_destination(gpu, arm)?;
    let cycle = Instant::now();
    let verification_before = arm.times.verification_readback_ns;
    let result: Result<Outcome> = async {
        let start = Instant::now();
        let base = buf.as_slice().as_ptr() as usize;
        let offset = aligned_subrange(base, buf.len(), FULL, ALIGN).map_err(Failure::accounting)?;
        if buf.len() != FULL || offset != 0 {
            return Err(Failure::accounting(
                "CONTROL pool buffer must be exactly FULL bytes and page aligned",
            ));
        }
        arm.pointers.observe(base, 0, buf.len())?;
        timed(&mut arm.times.alignment_setup_ns, start)?;
        add(&mut arm.source_read_attempts, 1)?;
        let start = Instant::now();
        let read = storage
            .read_expert_into_aligned_slice(id, buf.as_mut_slice())
            .await;
        timed(&mut arm.times.source_direct_read_ns, start)?;
        read_result(read, false, arm, id, &mut None)?;
        let start = Instant::now();
        let parsed = streams.source(buf.as_slice());
        timed(&mut arm.times.verification_readback_ns, start)?;
        let (offset, mut hashes) =
            parsed.map_err(|e| Failure::authority(format!("expert {id}: {e}")))?;
        note_payload(arm)?;
        let start = Instant::now();
        gpu.queue
            .write_buffer(&gpu.destination, 0, &EPOCH.to_le_bytes());
        timed(&mut arm.times.epoch_write_ns, start)?;
        let start = Instant::now();
        let view = gpu.queue.write_buffer_with(
            &gpu.destination,
            EPOCH_OFFSET as u64,
            NonZeroU64::new(PAYLOAD as u64).unwrap(),
        );
        timed(&mut arm.times.control_view_acquisition_ns, start)?;
        let mut view = view.ok_or_else(|| {
            Failure::runtime("gpu-failed", "Queue::write_buffer_with returned None")
        })?;
        let start = Instant::now();
        view.copy_from_slice(&buf.as_slice()[offset..]);
        timed(&mut arm.times.control_cpu_payload_copy_ns, start)?;
        add(&mut arm.cpu_payload_copy_bytes, PAYLOAD as u64)?;
        let start = Instant::now();
        drop(view);
        timed(&mut arm.times.control_staging_drop_scheduling_ns, start)?;
        gpu.check()?;
        note_upload(arm, false)?;
        let start = Instant::now();
        let result = gpu.drain(None);
        timed(&mut arm.times.submit_drain_ns, start)?;
        result?;
        add(&mut arm.gpu_completed_ops, 1)?;
        verify(gpu, arm, &mut hashes, streams)?;
        Ok(Outcome::Verified(hashes))
    }
    .await;
    let verification = arm
        .times
        .verification_readback_ns
        .checked_sub(verification_before)
        .ok_or_else(|| Failure::accounting("verification timer underflow"))?;
    let transfer = elapsed(cycle)?
        .checked_sub(verification)
        .ok_or_else(|| Failure::accounting("transfer timer underflow"))?;
    add(&mut arm.times.transfer_cycle_ns, transfer)?;
    if result
        .as_ref()
        .is_err_and(|e| e.classification == "gpu-failed")
        && arm.gpu_failures == 0
    {
        add(&mut arm.gpu_failures, 1)?;
    }
    result
}

async fn treatment(
    gpu: &Gpu,
    storage: &NvmeStorage,
    id: u32,
    arm: &mut Arm,
    streams: &mut Streams,
    rejection: &mut Option<String>,
) -> Result<Outcome> {
    add(&mut arm.ops_attempted, 1)?;
    fd_evidence(storage, id, arm)?;
    prepare_destination(gpu, arm)?;
    let cycle = Instant::now();
    let verification_before = arm.times.verification_readback_ns;
    let result: Result<Outcome> = async {
        add(&mut arm.map_attempts, 1)?;
        let start = Instant::now();
        let mapped = gpu.map(&gpu.upload, wgpu::MapMode::Write);
        timed(&mut arm.times.map_wait_ns, start)?;
        if mapped.is_err() {
            add(&mut arm.map_failures, 1)?;
        }
        mapped?;
        add(&mut arm.maps_completed, 1)?;
        // This inner future borrows the view only. It completes and drops every
        // BufferViewMut before the outer code unmaps or submits any GPU work.
        let source = async {
            let start = Instant::now();
            let mut view = gpu.upload.slice(..).get_mapped_range_mut();
            let base = view.as_ptr() as usize;
            // Recomputed for EVERY map, including every warmup and measured op.
            let offset = aligned_subrange(base, view.len(), FULL, ALIGN).and_then(|offset| {
                copy_offsets(offset, PREFIX, PAYLOAD, view.len()).map(|_| offset)
            });
            timed(&mut arm.times.alignment_setup_ns, start)?;
            let offset = match offset {
                Ok(offset) => offset,
                Err(e) => {
                    add(&mut arm.alignment_failures, 1)?;
                    add(
                        arm.pointers
                            .mapping_base_mod_4096_counts
                            .entry(base % ALIGN)
                            .or_default(),
                        1,
                    )?;
                    rejection.get_or_insert_with(|| {
                        format!("expert {id}: {e}; mapped base modulo 4096={}", base % ALIGN)
                    });
                    return Ok(None);
                }
            };
            arm.pointers.observe(base, offset, FULL)?;
            add(&mut arm.source_read_attempts, 1)?;
            let start = Instant::now();
            let read = storage
                .read_expert_into_aligned_slice(id, &mut view[offset..offset + FULL])
                .await;
            timed(&mut arm.times.source_direct_read_ns, start)?;
            if !read_result(read, true, arm, id, rejection)? {
                return Ok(Some(Err(Outcome::MappedRejected)));
            }
            let start = Instant::now();
            let parsed = streams.source(&view[offset..offset + FULL]);
            timed(&mut arm.times.verification_readback_ns, start)?;
            let (prefix, hashes) =
                parsed.map_err(|e| Failure::authority(format!("expert {id}: {e}")))?;
            note_payload(arm)?;
            let start = Instant::now();
            let gpu_offset = copy_offsets(offset, prefix, PAYLOAD, view.len());
            timed(&mut arm.times.alignment_setup_ns, start)?;
            add(&mut arm.pointers.gpu_offset_checks, 1)?;
            let gpu_offset = gpu_offset.map_err(|e| {
                arm.pointers.gpu_offset_failures += 1;
                Failure::accounting(e)
            })?;
            Ok(Some(Ok((gpu_offset, hashes))))
        }
        .await;
        let start = Instant::now();
        gpu.upload.unmap();
        timed(&mut arm.times.treatment_unmap_ns, start)?;
        add(&mut arm.unmaps, 1)?;
        gpu.check()?;
        let (gpu_offset, mut hashes) = match source? {
            None => return Ok(Outcome::AlignmentUnavailable),
            Some(Err(outcome)) => return Ok(outcome),
            Some(Ok(data)) => data,
        };
        let start = Instant::now();
        gpu.queue
            .write_buffer(&gpu.destination, 0, &EPOCH.to_le_bytes());
        timed(&mut arm.times.epoch_write_ns, start)?;
        let start = Instant::now();
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("source-to-upload-payload-copy"),
            });
        encoder.copy_buffer_to_buffer(
            &gpu.upload,
            gpu_offset,
            &gpu.destination,
            EPOCH_OFFSET as u64,
            PAYLOAD as u64,
        );
        let command = encoder.finish();
        timed(&mut arm.times.treatment_gpu_copy_encoding_ns, start)?;
        gpu.check()?;
        note_upload(arm, true)?;
        let start = Instant::now();
        let result = gpu.drain(Some(command));
        timed(&mut arm.times.submit_drain_ns, start)?;
        result?;
        add(&mut arm.gpu_completed_ops, 1)?;
        verify(gpu, arm, &mut hashes, streams)?;
        Ok(Outcome::Verified(hashes))
    }
    .await;
    let verification = arm
        .times
        .verification_readback_ns
        .checked_sub(verification_before)
        .ok_or_else(|| Failure::accounting("verification timer underflow"))?;
    let transfer = elapsed(cycle)?
        .checked_sub(verification)
        .ok_or_else(|| Failure::accounting("transfer timer underflow"))?;
    add(&mut arm.times.transfer_cycle_ns, transfer)?;
    if result
        .as_ref()
        .is_err_and(|e| e.classification == "gpu-failed")
        && arm.gpu_failures == 0
    {
        add(&mut arm.gpu_failures, 1)?;
    }
    result
}

async fn run_phase(
    phase: &mut Phase,
    gpu: &Gpu,
    storage: &NvmeStorage,
    buf: &mut PooledBuffer,
) -> Result<()> {
    let mut control_streams = Streams::default();
    let mut treatment_streams = Streams::default();
    let mut attempted_ids = Sha256::new();
    let result = async {
        for pair in 0..phase.ordered_expert_ids.len() {
            let id = phase.ordered_expert_ids[pair];
            add(&mut phase.pairs_attempted, 1)?;
            attempted_ids.update(id.to_le_bytes());
            // Each accumulator changes only in its one source call per pair.
            // Sampling here keeps all new work outside both existing timers.
            let control_before = phase.control.times.source_direct_read_ns;
            let treatment_before = phase.treatment.times.source_direct_read_ns;
            let (c, t) = if pair % 2 == 0 {
                let c = control(
                    gpu,
                    storage,
                    buf,
                    id,
                    &mut phase.control,
                    &mut control_streams,
                )
                .await?;
                let t = treatment(
                    gpu,
                    storage,
                    id,
                    &mut phase.treatment,
                    &mut treatment_streams,
                    &mut phase.first_mechanism_rejection,
                )
                .await?;
                (c, t)
            } else {
                let t = treatment(
                    gpu,
                    storage,
                    id,
                    &mut phase.treatment,
                    &mut treatment_streams,
                    &mut phase.first_mechanism_rejection,
                )
                .await?;
                let c = control(
                    gpu,
                    storage,
                    buf,
                    id,
                    &mut phase.control,
                    &mut control_streams,
                )
                .await?;
                (c, t)
            };
            phase.finish_pair(pair, id, &c, &t, control_before, treatment_before)?;
        }
        Ok(())
    }
    .await;
    phase.control_witnesses = control_streams.snapshot();
    phase.treatment_witnesses = treatment_streams.snapshot();
    phase.attempted_expert_id_sequence_sha256 = finish_sha(&attempted_ids);
    phase.control.rates();
    phase.treatment.rates();
    phase.paired_source_read_stats = pair_stats(&phase.source_read_pairs)?;
    result
}

async fn execute(report: &mut Report) -> Result<()> {
    if !(2..=MAX_ITERATIONS).contains(&report.args.iterations)
        || report.args.warmup_iterations > MAX_ITERATIONS
    {
        return Err(Failure::runtime(
            "invalid-arguments",
            "iterations must be 2..=65536; warmup iterations 0..=65536",
        ));
    }
    report.warmup = Phase::new(
        "warmup",
        expert_sequence(report.args.warmup_iterations, NAMESPACE).map_err(Failure::accounting)?,
    );
    report.measured = Phase::new(
        "measured",
        expert_sequence(report.args.iterations, NAMESPACE).map_err(Failure::accounting)?,
    );
    // Read/hash/parse one identical config snapshot; never construct an Engine,
    // RealModel, residency manager, dense tensor loader or production backend.
    let bytes =
        std::fs::read(&report.args.config).map_err(|e| Failure::runtime("config-failed", e))?;
    report.config_sha256 = Some(sha(&bytes));
    let text = std::str::from_utf8(&bytes).map_err(|e| Failure::runtime("config-failed", e))?;
    let config: Config = toml::from_str(text).map_err(|e| Failure::runtime("config-failed", e))?;
    config.validate().map_err(Failure::authority)?;
    report.authority.direct_io_requested = !config.storage.no_direct;
    report.authority.packed_storage =
        Some(config.storage.packed_blob.is_some() || config.storage.packed_manifest.is_some());
    report.authority.source_data_dir = Some(config.model.data_dir.clone());
    validate_geometry(&config)?;
    report.authority.exact_geometry = true;
    if !report.authority.linux
        || report.args.expected_adapter_name != "NVIDIA L4"
        || !report.authority.direct_io_requested
        || report.authority.packed_storage != Some(false)
    {
        return Err(Failure::authority(
            "requires Linux, exact NVIDIA L4, direct I/O enabled, and non-packed per-expert layout",
        ));
    }
    let storage = NvmeStorage::new(StorageConfig {
        base_path: config.model.data_dir,
        expert_size: FULL,
        block_align: ALIGN,
        use_direct_io: true,
        num_experts_per_layer: Some(128),
    })
    .map_err(|e| Failure::runtime("source-failed", e))?;
    if storage.is_packed() {
        return Err(Failure::authority("packed storage is forbidden"));
    }
    let gpu = Gpu::new(&mut report.authority).await?;
    let pool = BufferPool::new(1, FULL, ALIGN);
    let mut buf = pool
        .try_acquire()
        .ok_or_else(|| Failure::runtime("runtime-failed", "CONTROL pool allocation unavailable"))?;
    run_phase(&mut report.warmup, &gpu, &storage, &mut buf).await?;
    run_phase(&mut report.measured, &gpu, &storage, &mut buf).await?;
    gpu.check()?;
    report.classify();
    Ok(())
}

pub(crate) async fn run_command(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    hma1c_b::hma1c_c::hma1c_d::run_command(args).await
}

// A's schema and runner remain version-separated test evidence. The CLI runs B.
#[cfg(test)]
async fn run_command_hma1c_a(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Exclusive creation prevents accidentally replacing an immutable experiment.
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.report_out)?;
    let mut report = Report::new(args);
    let execution = std::panic::AssertUnwindSafe(execute(&mut report))
        .catch_unwind()
        .await;
    match execution {
        Ok(Ok(())) => {}
        Ok(Err(failure)) => report.fail(failure),
        Err(payload) => {
            let detail = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "non-string panic".into());
            report.fail(Failure::runtime(
                "runtime-failed",
                format!("diagnostic panic: {detail}"),
            ));
        }
    }
    serde_json::to_writer_pretty(&mut output, &report)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    if report.complete {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "{}: {}",
            report.classification,
            report.failure.as_deref().unwrap_or("diagnostic incomplete")
        ))
        .into())
    }
}

#[cfg(test)]
mod tests {
    use super::run_command_hma1c_a as run_command;
    use super::*;
    use clap::Parser;

    fn args() -> Args {
        Args {
            config: "unused.toml".into(),
            expected_adapter_name: "NVIDIA L4".into(),
            warmup_iterations: 0,
            iterations: 2,
            report_out: "unused.json".into(),
        }
    }
    fn config() -> Config {
        toml::from_str(
            r#"
[server]
[model]
data_dir = "/nonexistent/source-to-upload-test"
num_experts = 128
top_k = 8
d_model = 2048
d_ff = 768
expert_size = 2658304
num_layers = 48
dtype = "q4_0"
[storage]
cache_slots = 48
block_align = 4096
no_direct = false
"#,
        )
        .unwrap()
    }
    fn source(tag: u8) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(FULL);
        TensorHeader::for_swiglu_expert(WeightDtype::Q4_0, 2048, 768)
            .write_padded(ALIGN, &mut bytes);
        bytes.resize(FULL, tag);
        bytes
    }
    fn good_arm(n: u64, treatment: bool) -> Arm {
        let mut a = Arm {
            ops_attempted: n,
            source_read_attempts: n,
            source_read_ops: n,
            full_source_bytes: n * FULL as u64,
            payload_ops: n,
            payload_bytes: n * PAYLOAD as u64,
            cpu_payload_copy_bytes: if treatment { 0 } else { n * PAYLOAD as u64 },
            upload_ops: n,
            gpu_copied_bytes: n * PAYLOAD as u64,
            explicit_copy_buffer_bytes: if treatment { n * PAYLOAD as u64 } else { 0 },
            epoch_bytes: n * 4,
            gpu_completed_ops: n,
            verified_ops: n,
            verification_readback_bytes: n * SLOT as u64,
            verification_destination_reset_ops: n,
            verification_destination_reset_bytes: n * SLOT as u64,
            ..Arm::default()
        };
        a.fd_evidence = FdEvidence {
            checks: n,
            direct_observed: n,
            full_file_length_observed: n,
            ..FdEvidence::default()
        };
        a.pointers.observations = n;
        a.pointers.aligned = n;
        a.times.source_direct_read_ns = n * FULL as u64;
        a.times.transfer_cycle_ns = n * FULL as u64;
        if treatment {
            a.maps_completed = n;
            a.map_attempts = n;
            a.unmaps = n;
            a.pointers.gpu_offset_checks = n;
        }
        a
    }
    fn good_phase(name: &'static str, n: usize) -> Phase {
        let mut p = Phase::new(name, expert_sequence(n, NAMESPACE).unwrap());
        p.pairs_attempted = n as u64;
        p.pairs_completed = n as u64;
        p.attempted_expert_id_sequence_sha256 = p.expert_id_sequence_sha256.clone();
        p.control = good_arm(n as u64, false);
        p.treatment = good_arm(n as u64, true);
        set_pair_times(&mut p, &vec![(FULL as u64, FULL as u64); n]);
        p.control_witnesses = Witnesses {
            full_source_sha256: sha(b"source"),
            bare_payload_sha256: sha(b"payload"),
            gpu_destination_payload_sha256: sha(b"payload"),
        };
        p.treatment_witnesses = Witnesses {
            full_source_sha256: sha(b"source"),
            bare_payload_sha256: sha(b"payload"),
            gpu_destination_payload_sha256: sha(b"payload"),
        };
        p
    }
    fn set_pair_times(p: &mut Phase, times: &[(u64, u64)]) {
        p.source_read_pairs = times
            .iter()
            .enumerate()
            .map(|(pair, &(control_ns, treatment_ns))| SourceReadPair {
                pair,
                expert_id: p.ordered_expert_ids[pair],
                control_ns,
                treatment_ns,
            })
            .collect();
        p.paired_source_read_stats = pair_stats(&p.source_read_pairs).unwrap();
        p.successful_pairs_completed = p.paired_source_read_stats.paired_source_read_samples;
        p.control.times.source_direct_read_ns =
            p.paired_source_read_stats.paired_control_source_read_ns;
        p.treatment.times.source_direct_read_ns =
            p.paired_source_read_stats.paired_treatment_source_read_ns;
    }
    fn clear_pair_times(p: &mut Phase) {
        p.source_read_pairs.clear();
        p.paired_source_read_stats = PairStats::default();
        p.successful_pairs_completed = 0;
    }
    fn good_report() -> Report {
        let mut r = Report::new(args());
        r.authority.linux = true;
        r.authority.adapter_authoritative = true;
        r.authority.direct_io_requested = true;
        r.authority.packed_storage = Some(false);
        r.authority.exact_geometry = true;
        r.warmup = good_phase("warmup", 0);
        r.measured = good_phase("measured", 2);
        r
    }
    fn rejected_arm(n: u64, errno: i32) -> Arm {
        let mut a = good_arm(n, true);
        a.source_read_ops = 0;
        a.full_source_bytes = 0;
        a.payload_ops = 0;
        a.payload_bytes = 0;
        a.upload_ops = 0;
        a.gpu_copied_bytes = 0;
        a.explicit_copy_buffer_bytes = 0;
        a.epoch_bytes = 0;
        a.gpu_completed_ops = 0;
        a.verified_ops = 0;
        a.verification_readback_bytes = 0;
        a.source_failures = n;
        a.mapped_direct_io_rejections = n;
        a.rejection_errno_counts.insert(errno, n);
        a
    }

    fn samples(times: &[(u64, u64)]) -> Vec<SourceReadPair> {
        times
            .iter()
            .enumerate()
            .map(|(pair, &(control_ns, treatment_ns))| SourceReadPair {
                pair,
                expert_id: pair as u32,
                control_ns,
                treatment_ns,
            })
            .collect()
    }
    #[test]
    fn source_to_upload_copy_elision_pair_medians_odd_even_signed_and_equal() {
        let odd = pair_stats(&samples(&[(10, 15), (10, 5), (10, 10)])).unwrap();
        assert_eq!(odd.paired_source_read_samples, 3);
        assert_eq!(
            (
                odd.treatment_slower_pairs,
                odd.treatment_faster_pairs,
                odd.equal_pairs
            ),
            (1, 1, 1)
        );
        assert_eq!(odd.aggregate_treatment_minus_control_ns, 0);
        assert_eq!(odd.mean_treatment_minus_control_ns, Some(0.0));
        assert_eq!(odd.median_treatment_minus_control_ns, Some(0.0));
        assert_eq!(odd.median_treatment_over_control_ratio, Some(1.0));
        let input = samples(&[(20, 10), (10, 20), (10, 9), (10, 14)]);
        let even = pair_stats(&input).unwrap();
        assert_eq!(even.aggregate_treatment_minus_control_ns, 3);
        assert_eq!(even.mean_treatment_minus_control_ns, Some(0.75));
        assert_eq!(even.median_treatment_minus_control_ns, Some(1.5));
        assert_eq!(even.median_treatment_over_control_ratio, Some(1.15));
        let reversed: Vec<_> = input.iter().cloned().rev().collect();
        assert_eq!(pair_stats(&reversed).unwrap(), even);
        let negative = pair_stats(&samples(&[(10, 8), (10, 9)])).unwrap();
        assert_eq!(negative.median_treatment_minus_control_ns, Some(-1.5));
        assert_eq!(negative.mean_treatment_minus_control_ns, Some(-1.5));
        assert_eq!(pair_stats(&[]).unwrap(), PairStats::default());
        let single = pair_stats(&samples(&[(4, 7)])).unwrap();
        assert_eq!(single.median_treatment_minus_control_ns, Some(3.0));
        assert_eq!(single.median_treatment_over_control_ratio, Some(1.75));
    }
    #[test]
    fn source_to_upload_copy_elision_pair_arithmetic_is_checked() {
        assert_eq!(source_duration(17, 27).unwrap(), 10);
        for (before, after) in [(9, 8), (0, 0), (u64::MAX, 0)] {
            assert_eq!(
                source_duration(before, after).unwrap_err().classification,
                "accounting-failed"
            );
        }
        for times in [
            vec![(0, 1)],
            vec![(1, 0)],
            vec![(u64::MAX, 1), (1, 1)],
            vec![(1, u64::MAX), (1, 1)],
        ] {
            assert_eq!(
                pair_stats(&samples(&times)).unwrap_err().classification,
                "accounting-failed"
            );
        }
        assert!(pair_stats(&samples(&vec![(1, 1); MAX_ITERATIONS + 1])).is_err());
        for (c, t) in [(u64::MAX, 1), (1, u64::MAX), (u64::MAX, u64::MAX)] {
            let stats = pair_stats(&samples(&[(c, t)])).unwrap();
            assert_eq!(
                stats.aggregate_treatment_minus_control_ns,
                i128::from(t) - i128::from(c)
            );
            assert!(stats
                .median_treatment_over_control_ratio
                .unwrap()
                .is_finite());
        }
        let mut counter = u64::MAX;
        assert!(add(&mut counter, 1).is_err());
        assert_eq!(counter, u64::MAX);
    }
    #[test]
    fn source_to_upload_copy_elision_equal_ratio_median_is_order_independent() {
        // All ratios are exactly 1/3, but the large operands round upward in f64.
        let k = 9_007_199_254_740_894;
        assert_ne!(k as f64 / (3 * k) as f64, 1.0 / 3.0);
        let a = samples(&[(3, 1), (3 * k, k), (6, 2)]);
        let b = samples(&[(3 * k, k), (3, 1), (6, 2)]);
        let expected = pair_stats(&a).unwrap();
        assert_eq!(pair_stats(&b).unwrap(), expected);
        assert_eq!(
            expected.median_treatment_over_control_ratio,
            Some(1.0 / 3.0)
        );
    }
    #[test]
    fn source_to_upload_copy_elision_pair_accounting_fails_closed() {
        let mutations: Vec<fn(&mut Phase)> = vec![
            |p| p.paired_source_read_stats.paired_source_read_samples += 1,
            |p| p.paired_source_read_stats.treatment_slower_pairs = u64::MAX,
            |p| p.paired_source_read_stats.treatment_faster_pairs += 1,
            |p| p.paired_source_read_stats.equal_pairs -= 1,
            |p| {
                p.paired_source_read_stats
                    .aggregate_treatment_minus_control_ns += 1
            },
            |p| p.paired_source_read_stats.mean_treatment_minus_control_ns = Some(f64::NAN),
            |p| p.paired_source_read_stats.median_treatment_minus_control_ns = Some(1.0),
            |p| {
                p.paired_source_read_stats
                    .median_treatment_over_control_ratio = None
            },
            |p| p.paired_source_read_stats.paired_control_source_read_ns += 1,
            |p| p.paired_source_read_stats.paired_treatment_source_read_ns += 1,
            |p| p.successful_pairs_completed -= 1,
            |p| p.pairs_completed -= 1,
            |p| p.source_read_pairs.pop().map(|_| ()).unwrap(),
            |p| p.source_read_pairs[0].control_ns = 0,
            |p| p.source_read_pairs[0].expert_id = 99,
            |p| p.source_read_pairs[1].pair = 0,
            |p| p.source_read_pairs.swap(0, 1),
        ];
        for (i, mutate) in mutations.into_iter().enumerate() {
            let mut r = good_report();
            mutate(&mut r.measured);
            r.classify();
            assert!(!r.complete && !r.correctness_pass, "mutation {i}");
            assert_eq!(r.classification, "accounting-failed", "mutation {i}");
        }
    }
    fn verified() -> Outcome {
        Outcome::Verified(Hashes {
            source: "s".into(),
            payload: "p".into(),
            gpu: "p".into(),
            epoch: true,
        })
    }
    #[test]
    fn source_to_upload_copy_elision_samples_only_successful_completed_pairs() {
        let mut p = good_phase("measured", 2);
        clear_pair_times(&mut p);
        p.pairs_completed = 0;
        p.control.times.source_direct_read_ns = 23;
        p.treatment.times.source_direct_read_ns = 29;
        p.finish_pair(0, 0, &verified(), &verified(), 3, 4).unwrap();
        assert_eq!(
            (
                p.source_read_pairs[0].control_ns,
                p.source_read_pairs[0].treatment_ns
            ),
            (20, 25)
        );
        p.finish_pair(1, 6143, &verified(), &Outcome::MappedRejected, 23, 29)
            .unwrap();
        p.paired_source_read_stats = pair_stats(&p.source_read_pairs).unwrap();
        assert_eq!((p.pairs_completed, p.successful_pairs_completed), (2, 1));
        assert_eq!(p.paired_source_read_stats.paired_source_read_samples, 1);
        assert!(p.accounted());
        assert!(!p.successful());
        for bad in [
            Outcome::AlignmentUnavailable,
            Outcome::MappedRejected,
            Outcome::Verified(Hashes {
                source: "bad".into(),
                payload: "p".into(),
                gpu: "p".into(),
                epoch: true,
            }),
        ] {
            let mut p = Phase::new("measured", vec![0]);
            p.finish_pair(0, 0, &verified(), &bad, 0, 0).unwrap();
            assert_eq!(p.pairs_completed, 1);
            assert_eq!(p.successful_pairs_completed, 0);
            assert!(p.source_read_pairs.is_empty());
        }
        let mut p = Phase::new("measured", vec![0]);
        assert!(p.finish_pair(0, 0, &verified(), &verified(), 1, 0).is_err());
        assert_eq!(p.pairs_completed, 0);
        assert_eq!(p.successful_pairs_completed, 0);
        assert!(p.source_read_pairs.is_empty());
    }
    #[test]
    fn source_to_upload_copy_elision_warmup_cannot_change_measured_evidence() {
        let mut r = good_report();
        set_pair_times(&mut r.measured, &[(100, 106), (100, 110)]);
        r.classify();
        let measured = serde_json::to_value(&r.measured).unwrap();
        let ratio = r.source_throughput_ratio_treatment_over_control;
        let slowdown = r.source_read_slowdown_percent_treatment_vs_control;
        r.warmup = good_phase("warmup", 3);
        set_pair_times(&mut r.warmup, &[(10000, 1), (20000, 1), (30000, 1)]);
        r.classify();
        assert!(r.correctness_pass);
        assert_eq!(serde_json::to_value(&r.measured).unwrap(), measured);
        assert_eq!(r.source_throughput_ratio_treatment_over_control, ratio);
        assert_eq!(
            r.source_read_slowdown_percent_treatment_vs_control,
            slowdown
        );
    }
    #[test]
    fn source_to_upload_copy_elision_schema_and_authority_are_explicit() {
        let mut r = good_report();
        r.classify();
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(
            json["schema"],
            "mer.gpu-native-mapped-memory-odirect-discriminator.v1"
        );
        assert!(json.get("feasibility_pass").is_none());
        assert!(json.get("source_throughput_minimum_ratio").is_none());
        assert_eq!(json["correctness_pass"], true);
        assert_eq!(json["performance_required_for_correctness"], false);
        let a = &json["authority"];
        assert_eq!(a["control_source_api"], "read_expert_into_aligned_slice");
        assert_eq!(a["treatment_source_api"], "read_expert_into_aligned_slice");
        assert_eq!(a["control_destination"], "aligned-host-pool");
        assert_eq!(a["treatment_destination"], "wgpu-map-write");
        for key in [
            "same_source_api",
            "source_timer_excludes_allocation",
            "source_timer_excludes_map_async_device_poll",
            "source_timer_excludes_alignment_setup",
            "source_timer_excludes_hashes_readback_fd_evidence",
            "source_timer_excludes_gpu_copy_unmap",
        ] {
            assert_eq!(a[key], true, "{key}");
        }
        for key in [
            "paired_source_read_samples",
            "treatment_slower_pairs",
            "treatment_faster_pairs",
            "equal_pairs",
            "aggregate_treatment_minus_control_ns",
            "mean_treatment_minus_control_ns",
            "median_treatment_minus_control_ns",
            "median_treatment_over_control_ratio",
        ] {
            assert!(json["measured"].get(key).is_some(), "{key}");
        }
        assert_eq!(json["measured"]["paired_source_read_samples"], 2);
        assert_eq!(json["warmup"]["paired_source_read_samples"], 0);
    }
    #[test]
    fn source_to_upload_copy_elision_same_helper_and_source_timer_contract() {
        let source = include_str!("gpu_native_source_to_upload_copy_elision.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let control = source
            .split("async fn control(")
            .nth(1)
            .unwrap()
            .split("async fn treatment(")
            .next()
            .unwrap();
        let treatment = source
            .split("async fn treatment(")
            .nth(1)
            .unwrap()
            .split("async fn run_phase(")
            .next()
            .unwrap();
        assert_eq!(
            source
                .matches("timed(&mut arm.times.source_direct_read_ns, start)?;")
                .count(),
            2
        );
        for (arm, destination) in [
            (control, "buf.as_mut_slice()"),
            (treatment, "&mutview[offset..offset+FULL]"),
        ] {
            let marker = "timed(&mut arm.times.source_direct_read_ns, start)?;";
            let before = arm.split(marker).next().unwrap();
            let timed_body = before.rsplit("let start = Instant::now();").next().unwrap();
            let compact: String = timed_body.split_whitespace().collect();
            assert_eq!(
                compact,
                format!("letread=storage.read_expert_into_aligned_slice(id,{destination}).await;")
            );
            assert!(!arm.contains(".read_expert("));
            assert!(before.contains("fd_evidence(storage, id, arm)?;"));
            assert!(before.contains("aligned_subrange("));
            let after = arm.split(marker).nth(1).unwrap();
            assert!(after.contains("streams.source("));
            assert!(after.contains("verify(gpu, arm, &mut hashes, streams)?;"));
        }
        assert!(control.contains("if buf.len() != FULL || offset != 0"));
        let before_treatment_timer = treatment.split("let read = storage").next().unwrap();
        assert!(before_treatment_timer.contains("gpu.map(&gpu.upload, wgpu::MapMode::Write)"));
        assert!(
            before_treatment_timer.contains("copy_offsets(offset, PREFIX, PAYLOAD, view.len())")
        );
        let after_treatment_timer = treatment
            .split("timed(&mut arm.times.source_direct_read_ns, start)?;")
            .nth(1)
            .unwrap();
        assert!(after_treatment_timer.contains("gpu.upload.unmap();"));
        assert!(after_treatment_timer.contains("encoder.copy_buffer_to_buffer("));
        let execute = source.split("async fn execute(").nth(1).unwrap();
        assert!(
            execute.find("BufferPool::new(1, FULL, ALIGN)").unwrap()
                < execute.find("run_phase(").unwrap()
        );
    }
    #[test]
    fn source_to_upload_copy_elision_alternating_order_and_identical_id_contract() {
        let source = include_str!("gpu_native_source_to_upload_copy_elision.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        let run = source
            .split("async fn run_phase(")
            .nth(1)
            .unwrap()
            .split("async fn execute(")
            .next()
            .unwrap();
        assert!(run.contains("let id = phase.ordered_expert_ids[pair];"));
        let (even, odd) = run
            .split("let (c, t) = if pair % 2 == 0 {")
            .nth(1)
            .unwrap()
            .split_once("} else {")
            .unwrap();
        let odd = odd.split("phase.finish_pair(").next().unwrap();
        assert!(even.find("let c = control(").unwrap() < even.find("let t = treatment(").unwrap());
        assert!(odd.find("let t = treatment(").unwrap() < odd.find("let c = control(").unwrap());
        for arm_order in [even, odd] {
            let compact: String = arm_order.split_whitespace().collect();
            assert!(compact.contains(
                "control(gpu,storage,buf,id,&mutphase.control,&mutcontrol_streams,).await?;"
            ));
            assert!(compact.contains("treatment(gpu,storage,id,&mutphase.treatment,&muttreatment_streams,&mutphase.first_mechanism_rejection,).await?;"));
        }
        assert!(
            run.find("let control_before = phase.control.times.source_direct_read_ns;")
                .unwrap()
                < run.find("let (c, t)").unwrap()
        );
        assert!(
            run.find("let treatment_before = phase.treatment.times.source_direct_read_ns;")
                .unwrap()
                < run.find("let (c, t)").unwrap()
        );
        assert!(run.find("phase.finish_pair(").unwrap() > run.rfind("(c, t)").unwrap());
    }
    #[tokio::test]
    async fn source_to_upload_copy_elision_control_pool_has_exact_full_aligned_slice() {
        let pool = BufferPool::new(1, FULL, ALIGN);
        let mut buf = pool.try_acquire().unwrap();
        let ptr = buf.as_slice().as_ptr();
        assert_eq!(buf.len(), 2_658_304);
        assert_eq!(ptr as usize % 4096, 0);
        let slice = buf.as_mut_slice();
        assert_eq!(slice.len(), 2_658_304);
        assert_eq!(slice.as_ptr(), ptr);
        assert_eq!(
            aligned_subrange(ptr as usize, slice.len(), FULL, ALIGN).unwrap(),
            0
        );
    }

    #[test]
    fn source_to_upload_alignment_smallest_offset_for_every_page_residue() {
        for residue in 0..ALIGN {
            let base = ALIGN * 16 + residue;
            let offset = aligned_subrange(base, UPLOAD, FULL, ALIGN).unwrap();
            assert_eq!((base + offset) % ALIGN, 0);
            assert!(offset < ALIGN && offset + FULL <= UPLOAD);
            if offset > 0 {
                assert_ne!((base + offset - 1) % ALIGN, 0);
            }
            assert_eq!(
                copy_offsets(offset, PREFIX, PAYLOAD, UPLOAD).is_ok(),
                residue % 4 == 0
            );
        }
    }
    #[test]
    fn source_to_upload_alignment_recomputed_after_remap() {
        assert_eq!(aligned_subrange(0x1000, UPLOAD, FULL, ALIGN).unwrap(), 0);
        assert_eq!(aligned_subrange(0x2008, UPLOAD, FULL, ALIGN).unwrap(), 4088);
        assert_eq!(copy_offsets(4088, PREFIX, PAYLOAD, UPLOAD).unwrap(), 8184);
    }
    #[test]
    fn source_to_upload_alignment_rejects_invalid_impossible_and_overflow() {
        for (base, capacity, len, align) in [
            (0, UPLOAD, FULL, ALIGN),
            (4096, UPLOAD, FULL, 0),
            (4096, UPLOAD, FULL, 3),
            (4096, UPLOAD, 0, ALIGN),
            (4096, UPLOAD, FULL - 1, ALIGN),
            (4097, FULL, FULL, ALIGN),
            (usize::MAX, UPLOAD, FULL, ALIGN),
        ] {
            assert!(aligned_subrange(base, capacity, len, align).is_err());
        }
        assert!(copy_offsets(1, PREFIX, PAYLOAD, UPLOAD).is_err());
        assert!(copy_offsets(0, PREFIX, PAYLOAD - 1, UPLOAD).is_err());
        assert!(copy_offsets(usize::MAX, PREFIX, PAYLOAD, UPLOAD).is_err());
        assert!(copy_offsets(ALIGN, PREFIX, PAYLOAD, FULL).is_err());
    }
    #[test]
    fn source_to_upload_exact_authoritative_geometry() {
        let mut c = config();
        c.validate().unwrap();
        validate_geometry(&c).unwrap();
        assert_eq!(3 * 2048 * 768 / 32 * 18, PAYLOAD);
        assert_eq!(FULL % ALIGN, 0);
        // Bare payload is also block aligned; only FULL satisfies the source contract.
        assert_eq!(PAYLOAD % ALIGN, 0);
        assert_eq!(SLOT, EPOCH_OFFSET + PAYLOAD);
        for size in [PAYLOAD, FULL - ALIGN, FULL + ALIGN] {
            c.model.expert_size = size;
            assert!(validate_geometry(&c).is_err());
        }
        c = config();
        c.model.d_ff += 32;
        assert!(validate_geometry(&c).is_err());
        c = config();
        c.model.num_layers -= 1;
        assert!(validate_geometry(&c).is_err());
    }
    #[test]
    fn source_to_upload_header_offsets_and_validation() {
        let bytes = source(7);
        let (offset, payload) = payload_range(&bytes).unwrap();
        assert_eq!(offset, PREFIX);
        assert_eq!(payload.len(), PAYLOAD);
        assert_eq!(payload.as_ptr() as usize - bytes.as_ptr() as usize, PREFIX);
        for index in [0, 4, 6, 7, 8, 12, 44] {
            let mut bad = bytes.clone();
            bad[index] ^= 0xff;
            assert!(payload_range(&bad).is_err(), "header byte {index}");
        }
        assert!(payload_range(&bytes[..FULL - 1]).is_err());
    }
    #[test]
    fn source_to_upload_sequence_spans_namespace_and_is_deterministic() {
        let ids = expert_sequence(128, NAMESPACE).unwrap();
        assert_eq!(ids.first(), Some(&0));
        assert_eq!(ids.last(), Some(&6143));
        assert_eq!(ids, expert_sequence(128, NAMESPACE).unwrap());
        assert!(ids.windows(2).all(|p| p[0] < p[1]));
        let all = expert_sequence(6144, NAMESPACE).unwrap();
        assert_eq!(all, (0..NAMESPACE).collect::<Vec<_>>());
        let repeated = expert_sequence(6146, NAMESPACE).unwrap();
        assert_eq!(&repeated[6144..], &[0, 1]);
        assert_eq!(expert_sequence(1, NAMESPACE).unwrap(), vec![3072]);
        assert!(expert_sequence(2, 1).is_err());
        assert!(expert_sequence(MAX_ITERATIONS + 1, NAMESPACE).is_err());
    }
    #[test]
    fn source_to_upload_sequence_hash_frozen_little_endian() {
        let ids = expert_sequence(4, NAMESPACE).unwrap();
        assert_eq!(ids, vec![0, 2047, 4095, 6143]);
        // Independent known-byte encoding, including IDs exceeding one byte.
        assert_eq!(
            sequence_sha(&ids),
            sha(&[0, 0, 0, 0, 255, 7, 0, 0, 255, 15, 0, 0, 255, 23, 0, 0])
        );
        let mut reversed = ids.clone();
        reversed.reverse();
        assert_ne!(sequence_sha(&ids), sequence_sha(&reversed));
        assert_ne!(sequence_sha(&[1, 1]), sequence_sha(&[1]));
    }
    #[test]
    fn source_to_upload_witnesses_hash_exact_ordered_bytes_without_payload_copies() {
        let a = source(7);
        let b = source(23);
        let mut streams = Streams::default();
        streams.source(&a).unwrap();
        streams.source(&b).unwrap();
        streams.gpu.update(&a[PREFIX..]);
        streams.gpu.update(&b[PREFIX..]);
        let w = streams.snapshot();
        assert_eq!(
            w.full_source_sha256,
            sha(&[a.as_slice(), b.as_slice()].concat())
        );
        assert_eq!(
            w.bare_payload_sha256,
            sha(&[&a[PREFIX..], &b[PREFIX..]].concat())
        );
        assert_eq!(w.bare_payload_sha256, w.gpu_destination_payload_sha256);
        let mut reversed = Streams::default();
        reversed.source(&b).unwrap();
        reversed.source(&a).unwrap();
        assert_ne!(w.full_source_sha256, reversed.snapshot().full_source_sha256);
    }
    #[test]
    fn source_to_upload_pair_mismatch_records_first_context() {
        let mut p = good_phase("measured", 2);
        let c = Outcome::Verified(Hashes {
            source: "source-a".into(),
            payload: "p".into(),
            gpu: "p".into(),
            epoch: true,
        });
        let t = Outcome::Verified(Hashes {
            source: "source-b".into(),
            payload: "p".into(),
            gpu: "p".into(),
            epoch: true,
        });
        p.compare(1, 6143, &c, &t).unwrap();
        assert_eq!(p.mismatch_count, 1);
        let first = p.first_mismatch.as_ref().unwrap();
        assert_eq!(
            (first.pair, first.expert_id, first.kind),
            (1, 6143, "full-source")
        );
        p.compare(2, 0, &c, &t).unwrap();
        assert_eq!(p.mismatch_count, 2);
        assert_eq!(p.first_mismatch.as_ref().unwrap().expert_id, 6143);
    }
    #[test]
    fn source_to_upload_pass_does_not_require_total_cycle_superiority() {
        let mut r = good_report();
        r.measured.treatment.times.transfer_cycle_ns *= 10;
        r.classify();
        assert!(r.complete && r.correctness_pass);
        assert_eq!(r.classification, "mapped-memory-discriminator-complete");
        assert!(
            r.measured.treatment.transfer_cycle_payload_gbps
                < r.measured.control.transfer_cycle_payload_gbps
        );
    }
    #[test]
    fn source_to_upload_copy_elision_performance_never_controls_correctness() {
        for treatment in [10, 97, 99, 100, 101, 103, 105, 1000] {
            let mut r = good_report();
            set_pair_times(&mut r.measured, &[(100, treatment), (100, treatment)]);
            r.measured.treatment.times.verification_readback_ns = u64::MAX / 2;
            r.classify();
            assert!(r.complete && r.correctness_pass);
            assert_eq!(r.classification, "mapped-memory-discriminator-complete");
            assert_eq!(
                r.source_read_slowdown_percent_treatment_vs_control,
                Some((treatment as f64 - 100.0) / 100.0 * 100.0)
            );
            assert!(!r.performance_required_for_correctness);
        }
    }
    #[test]
    fn source_to_upload_gate_fails_closed_on_authority_or_evidence_mutation() {
        let mutations: Vec<fn(&mut Report)> = vec![
            |r| r.authority.control_source_api = "read_expert",
            |r| r.authority.treatment_source_api = "read_expert",
            |r| r.authority.same_source_api = false,
            |r| r.authority.control_destination = "wrong",
            |r| r.authority.treatment_destination = "wrong",
            |r| r.authority.source_timer_excludes_allocation = false,
            |r| r.authority.source_timer_excludes_map_async_device_poll = false,
            |r| r.authority.source_timer_excludes_alignment_setup = false,
            |r| {
                r.authority
                    .source_timer_excludes_hashes_readback_fd_evidence = false
            },
            |r| r.authority.source_timer_excludes_gpu_copy_unmap = false,
            |r| r.authority.linux = false,
            |r| r.authority.adapter_authoritative = false,
            |r| r.authority.expected_adapter_name = "other".into(),
            |r| r.authority.direct_io_requested = false,
            |r| r.authority.packed_storage = Some(true),
            |r| r.authority.exact_geometry = false,
            |r| r.measured.control.fd_evidence.direct_observed -= 1,
            |r| r.measured.treatment.fd_evidence.full_file_length_observed -= 1,
            |r| r.measured.treatment.pointers.aligned -= 1,
            |r| r.measured.treatment.fallback_reads = 1,
            |r| r.measured.control.cpu_payload_copy_bytes -= 1,
            |r| r.measured.treatment.cpu_payload_copy_bytes = 1,
            |r| r.measured.treatment.explicit_copy_buffer_bytes -= 4,
            |r| r.measured.treatment.exact_read_length_failures = 1,
            |r| r.measured.treatment.pointers.gpu_offset_failures = 1,
            |r| r.measured.treatment.source_failures = 1,
            |r| r.measured.treatment.gpu_failures = 1,
            |r| r.measured.treatment.map_failures = 1,
            |r| r.measured.treatment.accounting_failures = 1,
            |r| r.measured.treatment.unmaps -= 1,
            |r| r.measured.treatment.verification_destination_reset_ops -= 1,
            |r| r.measured.control.verified_ops -= 1,
            |r| r.measured.attempted_expert_id_sequence_sha256 = "wrong".into(),
            |r| r.measured.treatment_witnesses.bare_payload_sha256 = "wrong".into(),
            |r| r.measured.control.times.source_direct_read_ns = 0,
            |r| r.runtime_failures = 1,
        ];
        for (i, mutate) in mutations.into_iter().enumerate() {
            let mut r = good_report();
            mutate(&mut r);
            r.classify();
            assert!(!r.correctness_pass, "mutation {i}");
        }
    }
    #[test]
    fn source_to_upload_treatment_cpu_copy_accounting_is_exactly_zero() {
        let mut arm = good_arm(2, true);
        assert!(arm.reconcile(true));
        arm.cpu_payload_copy_bytes = 4;
        assert!(!arm.reconcile(true));
        let mut control = good_arm(2, false);
        control.cpu_payload_copy_bytes += 1;
        assert!(!control.reconcile(false));
        let mut overflow = good_arm(2, false);
        overflow.source_read_ops = u64::MAX;
        assert!(!overflow.reconcile(false));
    }
    #[test]
    fn source_to_upload_negative_direct_io_rejection_is_complete() {
        for errno in [libc::EINVAL, libc::EFAULT] {
            let mut r = good_report();
            r.measured.treatment = rejected_arm(2, errno);
            clear_pair_times(&mut r.measured);
            r.classify();
            assert_eq!(r.classification, "mapped-upload-direct-io-rejected");
            assert!(r.complete && !r.correctness_pass);
        }
        assert!(!mapped_rejection(Some(libc::EIO)));
        assert!(!mapped_rejection(None));
    }
    #[test]
    fn source_to_upload_negative_rejection_requires_working_control_and_consistency() {
        let mut r = good_report();
        r.measured.treatment = rejected_arm(2, libc::EINVAL);
        clear_pair_times(&mut r.measured);
        r.measured.control.source_failures = 1;
        r.classify();
        assert!(!r.complete && !r.correctness_pass);
        let mut r = good_report();
        r.measured.treatment = rejected_arm(2, libc::EIO);
        clear_pair_times(&mut r.measured);
        r.classify();
        assert!(!r.complete && !r.correctness_pass);
        let mut r = good_report();
        r.measured.treatment = rejected_arm(2, libc::EINVAL);
        clear_pair_times(&mut r.measured);
        r.measured.treatment.mapped_direct_io_rejections = 1;
        r.classify();
        assert!(!r.complete);
        let mut r = good_report();
        r.warmup = good_phase("warmup", 1);
        r.measured.treatment = rejected_arm(2, libc::EINVAL);
        clear_pair_times(&mut r.measured);
        r.classify();
        assert!(!r.complete);
    }
    #[test]
    fn source_to_upload_alignment_unavailable_is_scientific_negative() {
        let mut r = good_report();
        let mut t = rejected_arm(2, libc::EINVAL);
        t.source_read_attempts = 0;
        t.pointers.aligned = 0;
        t.pointers.observations = 0;
        t.source_failures = 0;
        t.mapped_direct_io_rejections = 0;
        t.rejection_errno_counts.clear();
        t.alignment_failures = 2;
        r.measured.treatment = t;
        clear_pair_times(&mut r.measured);
        r.classify();
        assert_eq!(r.classification, "alignment-contract-unavailable");
        assert!(r.complete && !r.correctness_pass);
    }
    #[test]
    fn source_to_upload_parity_classifications() {
        for (kind, expected) in [
            ("full-source", "source-parity-failed"),
            ("bare-payload", "source-parity-failed"),
            ("treatment-gpu-payload", "gpu-copy-parity-failed"),
        ] {
            let mut r = good_report();
            r.measured.mismatch(0, 0, kind, "a", "b").unwrap();
            r.classify();
            assert_eq!(r.classification, expected);
            assert!(r.complete && !r.correctness_pass);
        }
    }
    #[test]
    fn source_to_upload_read_errors_preserve_errno_and_reject_short_reads() {
        let mut a = Arm::default();
        let mut context = None;
        assert!(!read_result(
            Err(io::Error::from_raw_os_error(libc::EINVAL)),
            true,
            &mut a,
            17,
            &mut context
        )
        .unwrap());
        assert_eq!(a.source_failures, 1);
        assert_eq!(a.rejection_errno_counts[&libc::EINVAL], 1);
        assert!(context.unwrap().contains("expert 17"));
        assert!(read_result(
            Err(io::Error::from_raw_os_error(libc::EINVAL)),
            false,
            &mut a,
            17,
            &mut None
        )
        .is_err());
        assert!(read_result(Ok(FULL - 1), true, &mut a, 17, &mut None).is_err());
        assert_eq!(a.exact_read_length_failures, 1);
    }
    #[test]
    fn source_to_upload_cli_parses_explicit_and_defaults_without_startup_model_loading() {
        let cli = crate::Cli::try_parse_from([
            "mer",
            "diagnose-gpu-native-source-to-upload-copy-elision",
            "--config",
            "frozen.toml",
            "--report-out",
            "new.json",
        ])
        .unwrap();
        assert!(crate::startup_config_path(&cli.cmd).is_none());
        assert!(matches!(
            cli.cmd,
            crate::Cmd::DiagnoseGpuNativeSourceToUploadCopyElision {
                iterations: 128,
                warmup_iterations: 3,
                ..
            }
        ));
        let cli = crate::Cli::try_parse_from([
            "mer",
            "diagnose-gpu-native-source-to-upload-copy-elision",
            "--config",
            "frozen.toml",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--warmup-iterations",
            "0",
            "--iterations",
            "16",
            "--report-out",
            "new.json",
        ])
        .unwrap();
        assert!(matches!(
            cli.cmd,
            crate::Cmd::DiagnoseGpuNativeSourceToUploadCopyElision {
                iterations: 16,
                warmup_iterations: 0,
                ..
            }
        ));
        assert!(crate::Cli::try_parse_from([
            "mer",
            "diagnose-gpu-native-source-to-upload-copy-elision"
        ])
        .is_err());
        assert!(crate::Cli::try_parse_from([
            "mer",
            "diagnose-gpu-native-source-to-upload-copy-elision",
            "--config",
            "frozen.toml",
            "--report-out",
            "new.json",
            "--no-direct"
        ])
        .is_err());
    }
    #[test]
    fn source_to_upload_report_completion_semantics() {
        let mut r = good_report();
        r.fail(Failure::authority("wrong adapter"));
        assert!(r.complete && !r.correctness_pass);
        assert_eq!(r.runtime_failures, 0);
        r.fail(Failure::runtime("gpu-failed", "unexpected failure"));
        assert!(!r.complete && !r.correctness_pass);
        assert_eq!(r.runtime_failures, 1);
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["schema"], SCHEMA);
        assert_eq!(json["complete"], false);
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn source_to_upload_runner_writes_incomplete_failure_and_protects_report() {
        let dir = std::env::temp_dir().join(format!(
            "mer-source-to-upload-runner-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let mut a = args();
        a.config = dir.join("missing.toml");
        a.report_out = dir.join("failure.json");
        assert!(run_command(a.clone()).await.is_err());
        let bytes = std::fs::read(&a.report_out).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["complete"], false);
        assert_eq!(json["classification"], "config-failed");
        assert!(run_command(a.clone()).await.is_err());
        assert_eq!(std::fs::read(&a.report_out).unwrap(), bytes);
        a.config = dir.join("valid.toml");
        a.report_out = dir.join("authority.json");
        a.expected_adapter_name = "forbidden-test-adapter".into();
        std::fs::write(&a.config, toml::to_string(&config()).unwrap()).unwrap();
        run_command(a.clone()).await.unwrap();
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&a.report_out).unwrap()).unwrap();
        assert_eq!(json["complete"], true);
        assert_eq!(json["correctness_pass"], false);
        assert_eq!(json["classification"], "authority-failed");
        assert!(json["authority"]["adapter_name"].is_null());
        std::fs::remove_dir_all(dir).unwrap();
    }
}

/// HMA-1C-B is a separate diagnostic contract. A above remains frozen evidence;
/// only this module is reachable from the diagnostic CLI on the B branch.
mod hma1c_b {
    use super::*;
    use crate::aligned_buffer::AlignedBuffer;
    use crate::io_provider::SourceUploadFdProofSnapshot;
    use std::collections::BTreeSet;

    const B_SCHEMA: &str = "mer.gpu-native-mapped-memory-odirect-batch-discriminator.v1";
    const B_API: &str = "read_experts_batch_into_aligned_slices";
    const MAX_WIDTH: usize = 8;
    const UNIVERSE_SIZE: usize = 128;
    const HOST_ARENA: usize = MAX_WIDTH * FULL;
    const MAPPED_ARENA: usize = HOST_ARENA + ALIGN;
    const SCHEDULE_CONTRACT: &str = "Width-stratified deterministic sweep, not a production-weighted replay: no durable ordered production width histogram exists. Each phase starts p=0. K=p%8+1; round=p/8; start=(p*17)%128; slot j uses universe[(start+j*13)%128]. Universe[i]=floor(i*6143/127), i=0..127. Even round CONTROL then TREATMENT; odd round TREATMENT then CONTROL. Ordered source-set ID SHA256 encodes, per set, K as u32 LE followed by K ordered IDs as u32 LE. Ordered width SHA256 encodes each K as u32 LE. Raw samples retain p, round, K, IDs and order.";
    const INTERPRETATION_CONTRACT: &str = "PRIMARY = K=2..8 aggregate; K=1 is internal comparison, all-K descriptive. STRONG: primary slowdown >=5%, primary median treatment/control ratio >1, majority primary sets treatment-slower, >=5 of 7 multi-expert widths positive, and both primary execution-order strata positive. MATERIAL: same requirements at >=3%. Evidence AGAINST a material batch substrate cause: primary slowdown within +/-1%, primary median ratio within +/-1% of 1.0, and no coherent positive width trend. AMBIGUOUS / next surrounding-helper discriminator: >1% but <3%, aggregate/median/majority disagreement, order strata reversing sign, or fewer than 5/7 multi-expert widths agreeing in direction. Primary treatment speedup is evidence against raw mapped backing as the production source-gap cause. Do not round a sub-3% result upward. No unprovided numerical definition of coherent width trend is invented; apply this frozen interpretation to exact evidence after authoritative hardware. A is frozen AMBIGUOUS at +2.8884733837131744%; K=1 cross-run differences are contextual, not causal. Correctness is independent of performance.";
    const TIMER_CONTRACT: &str = "Only the one awaited NvmeStorage::read_experts_batch_into_aligned_slices call per arm per source set is timed. Arena allocation, map_async/device.poll, mapped-view acquisition, checked alignment and slice construction, preproof, fd evidence, hashing, GPU copy, readback and unmap are outside that interval. The helper retains its normal fd resolution, proof-cache lookup, scoped-thread scheduler, retries and breaker semantics inside the interval.";

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
    enum ExecutionOrder {
        #[serde(rename = "CONTROL-then-TREATMENT")]
        ControlFirst,
        #[serde(rename = "TREATMENT-then-CONTROL")]
        TreatmentFirst,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    struct SourceSet {
        set_index: usize,
        round: usize,
        width: usize,
        ordered_expert_ids: Vec<u32>,
        execution_order: ExecutionOrder,
    }
    fn schedule(count: usize) -> Result<Vec<SourceSet>> {
        if count > MAX_ITERATIONS {
            return Err(Failure::accounting("source-set count exceeds 65536"));
        }
        let universe = expert_sequence(UNIVERSE_SIZE, NAMESPACE).map_err(Failure::accounting)?;
        (0..count)
            .map(|p| {
                let width = p % MAX_WIDTH + 1;
                let round = p / MAX_WIDTH;
                let start = p
                    .checked_mul(17)
                    .ok_or_else(|| Failure::accounting("schedule start overflow"))?
                    % UNIVERSE_SIZE;
                let ids: Vec<_> = (0..width)
                    .map(|j| universe[(start + j * 13) % UNIVERSE_SIZE])
                    .collect();
                if ids.iter().any(|id| *id >= NAMESPACE)
                    || ids.iter().copied().collect::<BTreeSet<_>>().len() != width
                {
                    return Err(Failure::accounting(
                        "non-distinct or out-of-range source set",
                    ));
                }
                Ok(SourceSet {
                    set_index: p,
                    round,
                    width,
                    ordered_expert_ids: ids,
                    execution_order: if round % 2 == 0 {
                        ExecutionOrder::ControlFirst
                    } else {
                        ExecutionOrder::TreatmentFirst
                    },
                })
            })
            .collect()
    }
    fn source_bytes(slots: u64) -> Result<u64> {
        slots
            .checked_mul(FULL as u64)
            .ok_or_else(|| Failure::accounting("source byte overflow"))
    }
    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    struct ScheduleEvidence {
        source_set_count: u64,
        expert_slot_count: u64,
        source_bytes: u64,
        width_histogram: BTreeMap<usize, u64>,
        ordered_source_set_ids_sha256: String,
        ordered_width_sha256: String,
    }
    fn schedule_evidence(sets: &[SourceSet]) -> Result<ScheduleEvidence> {
        let mut e = ScheduleEvidence {
            source_set_count: 0,
            expert_slot_count: 0,
            source_bytes: 0,
            width_histogram: (1..=MAX_WIDTH).map(|k| (k, 0)).collect(),
            ordered_source_set_ids_sha256: String::new(),
            ordered_width_sha256: String::new(),
        };
        let mut ids = Sha256::new();
        let mut widths = Sha256::new();
        for set in sets {
            if !(1..=MAX_WIDTH).contains(&set.width)
                || set.ordered_expert_ids.len() != set.width
                || set.ordered_expert_ids.iter().any(|id| *id >= NAMESPACE)
                || set
                    .ordered_expert_ids
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .len()
                    != set.width
            {
                return Err(Failure::accounting("invalid source-set geometry"));
            }
            add(&mut e.source_set_count, 1)?;
            add(&mut e.expert_slot_count, set.width as u64)?;
            add(e.width_histogram.get_mut(&set.width).unwrap(), 1)?;
            widths.update((set.width as u32).to_le_bytes());
            ids.update((set.width as u32).to_le_bytes());
            for id in &set.ordered_expert_ids {
                ids.update(id.to_le_bytes());
            }
        }
        e.source_bytes = source_bytes(e.expert_slot_count)?;
        e.ordered_source_set_ids_sha256 = finish_sha(&ids);
        e.ordered_width_sha256 = finish_sha(&widths);
        Ok(e)
    }

    /// Split one arena with safe exclusive chunks. Check the whole eight-slot
    /// backing range on every map, even when only the first K slices are used.
    fn arena_offset(base: usize, capacity: usize, width: usize, treatment: bool) -> Result<usize> {
        if !(1..=MAX_WIDTH).contains(&width)
            || capacity != if treatment { MAPPED_ARENA } else { HOST_ARENA }
        {
            return Err(Failure::accounting("invalid batch arena size or width"));
        }
        let offset =
            aligned_subrange(base, capacity, HOST_ARENA, ALIGN).map_err(Failure::accounting)?;
        if !treatment && offset != 0 {
            return Err(Failure::accounting("host arena must start page aligned"));
        }
        for j in 0..width {
            let slot = j
                .checked_mul(FULL)
                .and_then(|n| offset.checked_add(n))
                .ok_or_else(|| Failure::accounting("arena slot overflow"))?;
            // This also checks COPY_BUFFER_ALIGNMENT for every mapped payload.
            copy_offsets(slot, PREFIX, PAYLOAD, capacity).map_err(Failure::accounting)?;
        }
        Ok(offset)
    }
    fn arena_slices(
        arena: &mut [u8],
        width: usize,
        treatment: bool,
    ) -> Result<(usize, Vec<&mut [u8]>)> {
        let offset = arena_offset(arena.as_ptr() as usize, arena.len(), width, treatment)?;
        let len = width
            .checked_mul(FULL)
            .ok_or_else(|| Failure::accounting("arena length overflow"))?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Failure::accounting("arena end overflow"))?;
        let slices = arena
            .get_mut(offset..end)
            .ok_or_else(|| Failure::accounting("arena out of range"))?
            .chunks_exact_mut(FULL)
            .collect();
        Ok((offset, slices))
    }

    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    struct RawSample {
        #[serde(flatten)]
        source_set: SourceSet,
        control_ns: u64,
        treatment_ns: u64,
        delta_ns: i128,
        // The exact ratio components are control_ns and treatment_ns. No f64
        // representation participates in correctness or threshold decisions.
    }
    impl RawSample {
        fn new(source_set: SourceSet, control_ns: u64, treatment_ns: u64) -> Result<Self> {
            if control_ns == 0 || treatment_ns == 0 {
                return Err(Failure::accounting("zero batch source duration"));
            }
            let delta_ns = i128::from(treatment_ns)
                .checked_sub(i128::from(control_ns))
                .ok_or_else(|| Failure::accounting("batch delta overflow"))?;
            Ok(Self {
                source_set,
                control_ns,
                treatment_ns,
                delta_ns,
            })
        }
    }
    #[derive(Default, Debug, PartialEq, Serialize)]
    struct StratumStats {
        samples: u64,
        control_total_ns: u64,
        treatment_total_ns: u64,
        delta_ns: i128,
        slowdown_percent: Option<f64>,
    }
    impl StratumStats {
        fn observe(&mut self, s: &RawSample) -> Result<()> {
            add(&mut self.samples, 1)?;
            add(&mut self.control_total_ns, s.control_ns)?;
            add(&mut self.treatment_total_ns, s.treatment_ns)?;
            self.delta_ns = self
                .delta_ns
                .checked_add(s.delta_ns)
                .ok_or_else(|| Failure::accounting("stratum delta overflow"))?;
            Ok(())
        }
        fn finish(&mut self) -> Result<()> {
            if i128::from(self.treatment_total_ns).checked_sub(i128::from(self.control_total_ns))
                != Some(self.delta_ns)
            {
                return Err(Failure::accounting("stratum sums disagree"));
            }
            self.slowdown_percent = (self.control_total_ns > 0)
                .then(|| self.delta_ns as f64 / self.control_total_ns as f64 * 100.0);
            Ok(())
        }
    }
    #[derive(Default, Debug, PartialEq, Serialize)]
    struct BatchStats {
        #[serde(flatten)]
        paired: PairStats,
        aggregate_slowdown_percent: Option<f64>,
        control_first: StratumStats,
        treatment_first: StratumStats,
    }
    fn batch_stats(samples: &[RawSample]) -> Result<BatchStats> {
        let mut stats = BatchStats::default();
        let mut pairs = Vec::with_capacity(samples.len());
        for s in samples {
            if *s != RawSample::new(s.source_set.clone(), s.control_ns, s.treatment_ns)? {
                return Err(Failure::accounting(
                    "raw delta differs from exact components",
                ));
            }
            match s.source_set.execution_order {
                ExecutionOrder::ControlFirst => stats.control_first.observe(s)?,
                ExecutionOrder::TreatmentFirst => stats.treatment_first.observe(s)?,
            }
            pairs.push(SourceReadPair {
                pair: s.source_set.set_index,
                expert_id: 0,
                control_ns: s.control_ns,
                treatment_ns: s.treatment_ns,
            });
        }
        stats.paired = pair_stats(&pairs)?;
        stats.control_first.finish()?;
        stats.treatment_first.finish()?;
        let p = &stats.paired;
        if stats
            .control_first
            .samples
            .checked_add(stats.treatment_first.samples)
            != Some(p.paired_source_read_samples)
            || stats
                .control_first
                .control_total_ns
                .checked_add(stats.treatment_first.control_total_ns)
                != Some(p.paired_control_source_read_ns)
            || stats
                .control_first
                .treatment_total_ns
                .checked_add(stats.treatment_first.treatment_total_ns)
                != Some(p.paired_treatment_source_read_ns)
        {
            return Err(Failure::accounting("order strata do not reconcile"));
        }
        stats.aggregate_slowdown_percent = (p.paired_control_source_read_ns > 0).then(|| {
            p.aggregate_treatment_minus_control_ns as f64 / p.paired_control_source_read_ns as f64
                * 100.0
        });
        Ok(stats)
    }
    #[derive(Default, Debug, PartialEq, Serialize)]
    struct Statistics {
        per_width: BTreeMap<usize, BatchStats>,
        primary_k2_through_k8: BatchStats,
        descriptive_all_k: BatchStats,
        positive_multi_expert_widths: u64,
        negative_multi_expert_widths: u64,
        equal_multi_expert_widths: u64,
    }
    fn statistics(samples: &[RawSample]) -> Result<Statistics> {
        let mut stats = Statistics::default();
        for k in 1..=MAX_WIDTH {
            let selected: Vec<_> = samples
                .iter()
                .filter(|s| s.source_set.width == k)
                .cloned()
                .collect();
            let width = batch_stats(&selected)?;
            if k >= 2 && width.paired.paired_source_read_samples > 0 {
                add(
                    if width.paired.aggregate_treatment_minus_control_ns > 0 {
                        &mut stats.positive_multi_expert_widths
                    } else if width.paired.aggregate_treatment_minus_control_ns < 0 {
                        &mut stats.negative_multi_expert_widths
                    } else {
                        &mut stats.equal_multi_expert_widths
                    },
                    1,
                )?;
            }
            stats.per_width.insert(k, width);
        }
        let primary: Vec<_> = samples
            .iter()
            .filter(|s| (2..=MAX_WIDTH).contains(&s.source_set.width))
            .cloned()
            .collect();
        stats.primary_k2_through_k8 = batch_stats(&primary)?;
        stats.descriptive_all_k = batch_stats(samples)?;
        for (min, aggregate) in [
            (1, &stats.descriptive_all_k),
            (2, &stats.primary_k2_through_k8),
        ] {
            let mut count = 0;
            let mut control = 0;
            let mut treatment = 0;
            for (_, width) in stats.per_width.range(min..) {
                add(&mut count, width.paired.paired_source_read_samples)?;
                add(&mut control, width.paired.paired_control_source_read_ns)?;
                add(&mut treatment, width.paired.paired_treatment_source_read_ns)?;
            }
            if (count, control, treatment)
                != (
                    aggregate.paired.paired_source_read_samples,
                    aggregate.paired.paired_control_source_read_ns,
                    aggregate.paired.paired_treatment_source_read_ns,
                )
            {
                return Err(Failure::accounting(
                    "width and aggregate statistics do not reconcile",
                ));
            }
        }
        Ok(stats)
    }

    fn proof_hits_only(proof: &SourceUploadFdProofSnapshot, slots: u64) -> bool {
        slots.checked_mul(2).is_some_and(|requests| {
            proof.source_upload_fd_proof_requests == requests
                && proof.source_upload_fd_proof_hits == requests
                && proof.source_upload_fd_proof_misses == 0
                && proof.source_upload_fd_proof_failures == 0
        })
    }
    #[derive(Debug, Serialize)]
    struct BatchArm {
        #[serde(flatten)]
        evidence: Arm,
        source_sets_attempted: u64,
        batch_helper_calls: u64,
        completed_source_sets: Vec<SourceSet>,
        source_schedule: ScheduleEvidence,
    }
    impl BatchArm {
        fn new() -> Result<Self> {
            Ok(Self {
                evidence: Arm::default(),
                source_sets_attempted: 0,
                batch_helper_calls: 0,
                completed_source_sets: Vec::new(),
                source_schedule: schedule_evidence(&[])?,
            })
        }
        fn successful(&self, expected: &ScheduleEvidence, treatment: bool) -> bool {
            let a = &self.evidence;
            let n = expected.expert_slot_count;
            let sets = expected.source_set_count;
            self.source_schedule == *expected
                && schedule_evidence(&self.completed_source_sets).is_ok_and(|e| e == *expected)
                && self.source_sets_attempted == sets
                && self.batch_helper_calls == sets
                && a.reconcile(treatment)
                && a.ops_attempted == n
                && a.source_read_attempts == n
                && a.source_read_ops == n
                && a.full_source_bytes == expected.source_bytes
                && a.payload_ops == n
                && a.upload_ops == n
                && a.gpu_completed_ops == n
                && a.verified_ops == n
                && a.fd_evidence.checks == n
                && a.fd_evidence.direct_observed == n
                && a.fd_evidence.full_file_length_observed == n
                && a.fd_evidence.failures == 0
                && a.pointers.observations == n
                && a.verification_destination_reset_ops == n
                && a.source_failures == 0
                && a.gpu_failures == 0
                && a.map_failures == 0
                && a.alignment_failures == 0
                && a.mapped_direct_io_rejections == 0
                && a.rejection_errno_counts.is_empty()
                && a.pointers.gpu_offset_failures == 0
                && if treatment {
                    a.map_attempts == sets
                        && a.maps_completed == sets
                        && a.unmaps == sets
                        && a.pointers.gpu_offset_checks == n
                } else {
                    a.map_attempts == 0 && a.maps_completed == 0 && a.unmaps == 0
                }
        }
    }
    #[derive(Debug, Serialize)]
    struct BatchPhase {
        name: &'static str,
        schedule: Vec<SourceSet>,
        expected: ScheduleEvidence,
        raw_samples: Vec<RawSample>,
        statistics: Statistics,
        control: BatchArm,
        treatment: BatchArm,
        control_witnesses: Witnesses,
        treatment_witnesses: Witnesses,
        fd_proof: SourceUploadFdProofSnapshot,
        fd_proof_hits_only: bool,
        mismatch_count: u64,
        first_mismatch: Option<Mismatch>,
    }
    impl BatchPhase {
        fn new(name: &'static str, count: usize) -> Result<Self> {
            let schedule = schedule(count)?;
            Ok(Self {
                name,
                expected: schedule_evidence(&schedule)?,
                schedule,
                raw_samples: Vec::new(),
                statistics: statistics(&[])?,
                control: BatchArm::new()?,
                treatment: BatchArm::new()?,
                control_witnesses: Witnesses::default(),
                treatment_witnesses: Witnesses::default(),
                fd_proof: SourceUploadFdProofSnapshot::default(),
                fd_proof_hits_only: false,
                mismatch_count: 0,
                first_mismatch: None,
            })
        }
        fn mismatch(
            &mut self,
            set: &SourceSet,
            id: u32,
            kind: &'static str,
            c: &str,
            t: &str,
        ) -> Result<()> {
            if c != t {
                add(&mut self.mismatch_count, 1)?;
                self.first_mismatch.get_or_insert_with(|| Mismatch {
                    phase: self.name,
                    pair: set.set_index,
                    expert_id: id,
                    kind,
                    control: c.into(),
                    treatment: t.into(),
                });
            }
            Ok(())
        }
        fn compare(&mut self, set: &SourceSet, c: &[Hashes], t: &[Hashes]) -> Result<()> {
            if c.len() != set.width || t.len() != set.width {
                return Err(Failure::accounting("batch hash count mismatch"));
            }
            for ((&id, c), t) in set.ordered_expert_ids.iter().zip(c).zip(t) {
                self.mismatch(set, id, "full-source", &c.source, &t.source)?;
                self.mismatch(set, id, "bare-payload", &c.payload, &t.payload)?;
                self.mismatch(set, id, "control-gpu-payload", &c.payload, &c.gpu)?;
                self.mismatch(set, id, "treatment-gpu-payload", &t.payload, &t.gpu)?;
                self.mismatch(
                    set,
                    id,
                    "control-epoch",
                    "true",
                    if c.epoch { "true" } else { "false" },
                )?;
                self.mismatch(
                    set,
                    id,
                    "treatment-epoch",
                    "true",
                    if t.epoch { "true" } else { "false" },
                )?;
            }
            Ok(())
        }
        fn successful(&self) -> bool {
            let c = &self.control_witnesses;
            let t = &self.treatment_witnesses;
            self.fd_proof_hits_only
                && proof_hits_only(&self.fd_proof, self.expected.expert_slot_count)
                && schedule(self.schedule.len()).is_ok_and(|s| s == self.schedule)
                && schedule_evidence(&self.schedule).is_ok_and(|e| e == self.expected)
                && self.control.completed_source_sets == self.schedule
                && self.treatment.completed_source_sets == self.schedule
                && self.control.successful(&self.expected, false)
                && self.treatment.successful(&self.expected, true)
                && self.raw_samples.len() == self.schedule.len()
                && self
                    .raw_samples
                    .iter()
                    .zip(&self.schedule)
                    .all(|(s, set)| s.source_set == *set)
                && statistics(&self.raw_samples).is_ok_and(|s| s == self.statistics)
                && self
                    .statistics
                    .descriptive_all_k
                    .paired
                    .paired_control_source_read_ns
                    == self.control.evidence.times.source_direct_read_ns
                && self
                    .statistics
                    .descriptive_all_k
                    .paired
                    .paired_treatment_source_read_ns
                    == self.treatment.evidence.times.source_direct_read_ns
                && self.mismatch_count == 0
                && self.first_mismatch.is_none()
                && c.full_source_sha256 == t.full_source_sha256
                && c.bare_payload_sha256 == t.bare_payload_sha256
                && c.bare_payload_sha256 == c.gpu_destination_payload_sha256
                && c.bare_payload_sha256 == t.gpu_destination_payload_sha256
        }
    }

    #[derive(Debug, Serialize)]
    struct BatchAuthority {
        #[serde(flatten)]
        base: Authority,
        same_batch_source_api: bool,
        max_batch_width: usize,
        control_arena_capacity_bytes: usize,
        fd_preproof_completed: bool,
        fd_cache_capacity: usize,
        preproof_universe_size: usize,
        preproof_ordered_expert_ids: Vec<u32>,
        preproof: SourceUploadFdProofSnapshot,
        after_preproof_telemetry_reset: SourceUploadFdProofSnapshot,
        after_warmup_telemetry_reset: SourceUploadFdProofSnapshot,
        source_timer_excludes_fd_preproof: bool,
        source_timer_excludes_slice_setup: bool,
    }
    #[derive(Debug, Serialize)]
    struct BatchReport {
        schema: &'static str,
        args: Args,
        config_sha256: Option<String>,
        complete: bool,
        correctness_pass: bool,
        authoritative: bool,
        classification: String,
        failure: Option<String>,
        authority: BatchAuthority,
        warmup: BatchPhase,
        measured: BatchPhase,
        performance_required_for_correctness: bool,
        primary_endpoint: &'static str,
        schedule_contract: &'static str,
        timing_contract: &'static str,
        interpretation_contract: &'static str,
        retry_evidence_contract: &'static str,
        source_byte_evidence_contract: &'static str,
    }
    impl BatchReport {
        fn new(args: Args) -> Result<Self> {
            let mut base = Report::new(args.clone()).authority;
            base.control_source_api = B_API;
            base.treatment_source_api = B_API;
            base.control_destination = "aligned-host-arena";
            base.treatment_destination = "wgpu-map-write-arena";
            base.upload_capacity_bytes = MAPPED_ARENA;
            Ok(Self { schema: B_SCHEMA, args, config_sha256: None, complete: false, correctness_pass: false,
                authoritative: false, classification: "not-run".into(), failure: None,
                authority: BatchAuthority { base, same_batch_source_api: true, max_batch_width: MAX_WIDTH,
                    control_arena_capacity_bytes: HOST_ARENA, fd_preproof_completed: false, fd_cache_capacity: 0,
                    preproof_universe_size: UNIVERSE_SIZE, preproof_ordered_expert_ids: expert_sequence(UNIVERSE_SIZE, NAMESPACE).map_err(Failure::accounting)?,
                    preproof: SourceUploadFdProofSnapshot::default(), after_preproof_telemetry_reset: SourceUploadFdProofSnapshot::default(),
                    after_warmup_telemetry_reset: SourceUploadFdProofSnapshot::default(), source_timer_excludes_fd_preproof: true, source_timer_excludes_slice_setup: true },
                warmup: BatchPhase::new("warmup", 0)?, measured: BatchPhase::new("measured", 0)?, performance_required_for_correctness: false,
                primary_endpoint: "K=2..8 aggregate", schedule_contract: SCHEDULE_CONTRACT, timing_contract: TIMER_CONTRACT,
                interpretation_contract: INTERPRETATION_CONTRACT,
                retry_evidence_contract: "No diagnostic retry or fallback. The unmodified batch helper uses read_at_with_retries and the existing breaker. Transient retry attempts are emitted by that helper to tracing logs; the existing helper exposes no exact retry counter. A successful helper result alone must not be reported as proof of zero internal retries. Source failures below count failed helper calls; preserve the run log for retry evidence.",
                source_byte_evidence_contract: "Source bytes count exact successful FULL reads only, K*FULL per successful batch result. Partial bytes from a failed concurrent helper are unavailable, never inferred as zero physical I/O. Failure is non-authoritative. Hash streams concatenate exact full files, bare payloads and verified GPU payloads in source-set/slot order independently per arm. Destination reset, copies and readback are verification outside source timers." })
        }
        fn authority_valid(&self) -> bool {
            let a = &self.authority;
            let b = &a.base;
            self.schema == B_SCHEMA
                && !self.performance_required_for_correctness
                && a.same_batch_source_api
                && b.same_source_api
                && b.control_source_api == B_API
                && b.treatment_source_api == B_API
                && b.control_destination == "aligned-host-arena"
                && b.treatment_destination == "wgpu-map-write-arena"
                && a.max_batch_width == MAX_WIDTH
                && a.control_arena_capacity_bytes == HOST_ARENA
                && b.upload_capacity_bytes == MAPPED_ARENA
                && b.full_source_bytes == FULL
                && b.block_alignment == ALIGN
                && b.uth_prefix_bytes == PREFIX
                && b.bare_payload_bytes == PAYLOAD
                && b.physical_slot_bytes == SLOT
                && b.source_timer_excludes_allocation
                && b.source_timer_excludes_map_async_device_poll
                && b.source_timer_excludes_alignment_setup
                && b.source_timer_excludes_hashes_readback_fd_evidence
                && b.source_timer_excludes_gpu_copy_unmap
                && a.source_timer_excludes_fd_preproof
                && a.source_timer_excludes_slice_setup
                && b.linux
                && b.expected_adapter_name == "NVIDIA L4"
                && b.adapter_authoritative
                && b.direct_io_requested
                && b.packed_storage == Some(false)
                && b.exact_geometry
                && a.fd_preproof_completed
                && a.fd_cache_capacity >= a.preproof_universe_size
                && a.preproof_universe_size == UNIVERSE_SIZE
                && expert_sequence(UNIVERSE_SIZE, NAMESPACE)
                    .is_ok_and(|ids| ids == a.preproof_ordered_expert_ids)
                && a.preproof.source_upload_fd_proof_requests == UNIVERSE_SIZE as u64
                && a.preproof.source_upload_fd_proof_hits == 0
                && a.preproof.source_upload_fd_proof_misses == UNIVERSE_SIZE as u64
                && a.preproof.source_upload_fd_proof_failures == 0
                && a.after_preproof_telemetry_reset == SourceUploadFdProofSnapshot::default()
                && a.after_warmup_telemetry_reset == SourceUploadFdProofSnapshot::default()
                && self.warmup.fd_proof_hits_only
                && self.measured.fd_proof_hits_only
                && proof_hits_only(
                    &self.warmup.fd_proof,
                    self.warmup.expected.expert_slot_count,
                )
                && proof_hits_only(
                    &self.measured.fd_proof,
                    self.measured.expected.expert_slot_count,
                )
        }
        fn classify(&mut self) {
            self.complete = true;
            self.correctness_pass = false;
            self.authoritative = self.authority_valid();
            if !self.authoritative {
                self.classification = "authority-failed".into();
            } else if self.warmup.schedule.len() != self.args.warmup_iterations
                || self.measured.schedule.len() != self.args.iterations
                || self.measured.schedule.is_empty()
                || !self.warmup.successful()
                || !self.measured.successful()
            {
                self.authoritative = false;
                self.classification = "evidence-reconciliation-failed".into();
            } else {
                self.correctness_pass = true;
                self.classification = "batch-mapped-memory-discriminator-complete".into();
            }
        }
        fn fail(&mut self, failure: Failure) {
            self.complete = failure.complete;
            self.authoritative = false;
            self.correctness_pass = false;
            self.classification = failure.classification.into();
            self.failure = Some(failure.detail);
        }
    }

    fn begin_set(storage: &NvmeStorage, set: &SourceSet, arm: &mut BatchArm) -> Result<()> {
        add(&mut arm.source_sets_attempted, 1)?;
        add(&mut arm.evidence.ops_attempted, set.width as u64)?;
        for &id in &set.ordered_expert_ids {
            fd_evidence(storage, id, &mut arm.evidence)?;
        }
        Ok(())
    }
    fn observe_slices(
        arm: &mut BatchArm,
        base: usize,
        offset: usize,
        width: usize,
        treatment: bool,
    ) -> Result<()> {
        for j in 0..width {
            let slot = offset
                .checked_add(
                    j.checked_mul(FULL)
                        .ok_or_else(|| Failure::accounting("pointer slot overflow"))?,
                )
                .ok_or_else(|| Failure::accounting("pointer offset overflow"))?;
            arm.evidence.pointers.observe(base, slot, FULL)?;
            if treatment {
                add(&mut arm.evidence.pointers.gpu_offset_checks, 1)?;
            }
        }
        Ok(())
    }
    /// Both arms reach this single source call exactly once per set. All caller
    /// setup precedes entry; all accounting and verification follow the timer.
    async fn batch_source(
        storage: &NvmeStorage,
        set: &SourceSet,
        destinations: &mut [&mut [u8]],
        arm: &mut BatchArm,
        treatment: bool,
    ) -> Result<u64> {
        add(&mut arm.batch_helper_calls, 1)?;
        add(&mut arm.evidence.source_read_attempts, set.width as u64)?;
        let ids = set.ordered_expert_ids.as_slice();
        let start = Instant::now();
        let read = storage
            .read_experts_batch_into_aligned_slices(ids, destinations)
            .await;
        let ns = elapsed(start)?;
        add(&mut arm.evidence.times.source_direct_read_ns, ns)?;
        let expected = source_bytes(set.width as u64)?;
        match read {
            Ok(n) if u64::try_from(n).ok() == Some(expected) => {
                add(&mut arm.evidence.source_read_ops, set.width as u64)?;
                add(&mut arm.evidence.full_source_bytes, expected)?;
                arm.completed_source_sets.push(set.clone());
            }
            Ok(n) => {
                add(&mut arm.evidence.exact_read_length_failures, 1)?;
                return Err(Failure::accounting(format!(
                    "set {}: batch returned {n} bytes, expected {expected}",
                    set.set_index
                )));
            }
            Err(e) => {
                add(&mut arm.evidence.source_failures, 1)?;
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    add(&mut arm.evidence.exact_read_length_failures, 1)?;
                }
                let rejected = treatment && mapped_rejection(e.raw_os_error());
                if rejected {
                    add(&mut arm.evidence.mapped_direct_io_rejections, 1)?;
                    add(
                        arm.evidence
                            .rejection_errno_counts
                            .entry(e.raw_os_error().unwrap())
                            .or_default(),
                        1,
                    )?;
                }
                return Err(Failure::runtime(
                    if rejected {
                        "mapped-upload-direct-io-rejected"
                    } else {
                        "source-failed"
                    },
                    format!(
                        "set {} {:?}: {e}; errno={:?}; no fallback or diagnostic retry",
                        set.set_index,
                        ids,
                        e.raw_os_error()
                    ),
                ));
            }
        }
        if ns == 0 {
            return Err(Failure::accounting("zero source timer"));
        }
        Ok(ns)
    }
    async fn control_batch(
        gpu: &Gpu,
        storage: &NvmeStorage,
        host: &mut AlignedBuffer,
        set: &SourceSet,
        arm: &mut BatchArm,
        streams: &mut Streams,
    ) -> Result<(u64, Vec<Hashes>)> {
        begin_set(storage, set, arm)?;
        let base = host.as_slice().as_ptr() as usize;
        let (offset, mut destinations) = arena_slices(host.as_mut_slice(), set.width, false)
            .map_err(|e| {
                arm.evidence.alignment_failures += 1;
                e
            })?;
        observe_slices(arm, base, offset, set.width, false)?;
        let ns = batch_source(storage, set, &mut destinations, arm, false).await?;
        let mut hashes = Vec::with_capacity(set.width);
        for source in destinations {
            let (_, mut h) = streams.source(source).map_err(Failure::authority)?;
            note_payload(&mut arm.evidence)?;
            prepare_destination(gpu, &mut arm.evidence)?;
            gpu.queue
                .write_buffer(&gpu.destination, 0, &EPOCH.to_le_bytes());
            let mut view = gpu
                .queue
                .write_buffer_with(
                    &gpu.destination,
                    EPOCH_OFFSET as u64,
                    NonZeroU64::new(PAYLOAD as u64).unwrap(),
                )
                .ok_or_else(|| {
                    Failure::runtime(
                        "gpu-failed",
                        "CONTROL verification staging view unavailable",
                    )
                })?;
            view.copy_from_slice(&source[PREFIX..]);
            add(&mut arm.evidence.cpu_payload_copy_bytes, PAYLOAD as u64)?;
            drop(view);
            note_upload(&mut arm.evidence, false)?;
            gpu.drain(None)?;
            add(&mut arm.evidence.gpu_completed_ops, 1)?;
            verify(gpu, &mut arm.evidence, &mut h, streams)?;
            hashes.push(h);
        }
        gpu.check()?;
        Ok((ns, hashes))
    }
    async fn treatment_batch(
        gpu: &Gpu,
        storage: &NvmeStorage,
        set: &SourceSet,
        arm: &mut BatchArm,
        streams: &mut Streams,
    ) -> Result<(u64, Vec<Hashes>)> {
        begin_set(storage, set, arm)?;
        add(&mut arm.evidence.map_attempts, 1)?;
        let start = Instant::now();
        let mapped = gpu.map(&gpu.upload, wgpu::MapMode::Write);
        timed(&mut arm.evidence.times.map_wait_ns, start)?;
        if mapped.is_err() {
            add(&mut arm.evidence.map_failures, 1)?;
        }
        mapped?;
        add(&mut arm.evidence.maps_completed, 1)?;
        // Catch only to guarantee unmap after the view future is dropped. A
        // panic is then propagated to the report boundary, never retried.
        let mapped_source = std::panic::AssertUnwindSafe(async {
            let mut view = gpu.upload.slice(..).get_mapped_range_mut();
            let base = view.as_ptr() as usize;
            let capacity = view.len();
            let (offset, mut destinations) =
                arena_slices(&mut view, set.width, true).map_err(|e| {
                    arm.evidence.alignment_failures += 1;
                    e
                })?;
            observe_slices(arm, base, offset, set.width, true)?;
            let ns = batch_source(storage, set, &mut destinations, arm, true).await?;
            let mut payloads = Vec::with_capacity(set.width);
            for (j, source) in destinations.into_iter().enumerate() {
                let (prefix, h) = streams.source(source).map_err(Failure::authority)?;
                note_payload(&mut arm.evidence)?;
                let source_offset = offset
                    .checked_add(
                        j.checked_mul(FULL)
                            .ok_or_else(|| Failure::accounting("copy slot overflow"))?,
                    )
                    .ok_or_else(|| Failure::accounting("copy base overflow"))?;
                let gpu_offset = copy_offsets(source_offset, prefix, PAYLOAD, capacity)
                    .map_err(Failure::accounting)?;
                payloads.push((gpu_offset, h));
            }
            Ok::<_, Failure>((ns, payloads))
        })
        .catch_unwind()
        .await;
        let start = Instant::now();
        gpu.upload.unmap();
        timed(&mut arm.evidence.times.treatment_unmap_ns, start)?;
        add(&mut arm.evidence.unmaps, 1)?;
        let source = match mapped_source {
            Ok(result) => result,
            Err(panic) => std::panic::resume_unwind(panic),
        };
        let (ns, payloads) = source?;
        gpu.check()?;
        let mut hashes = Vec::with_capacity(set.width);
        for (gpu_offset, mut h) in payloads {
            prepare_destination(gpu, &mut arm.evidence)?;
            gpu.queue
                .write_buffer(&gpu.destination, 0, &EPOCH.to_le_bytes());
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("batch-source-verification-copy"),
                });
            encoder.copy_buffer_to_buffer(
                &gpu.upload,
                gpu_offset,
                &gpu.destination,
                EPOCH_OFFSET as u64,
                PAYLOAD as u64,
            );
            note_upload(&mut arm.evidence, true)?;
            gpu.drain(Some(encoder.finish()))?;
            add(&mut arm.evidence.gpu_completed_ops, 1)?;
            verify(gpu, &mut arm.evidence, &mut h, streams)?;
            hashes.push(h);
        }
        gpu.check()?;
        Ok((ns, hashes))
    }
    fn note_arm_error(arm: &mut BatchArm, failure: &Failure) -> Result<()> {
        if failure.classification == "gpu-failed" && arm.evidence.gpu_failures == 0 {
            add(&mut arm.evidence.gpu_failures, 1)?;
        }
        if failure.classification == "accounting-failed" && arm.evidence.accounting_failures == 0 {
            add(&mut arm.evidence.accounting_failures, 1)?;
        }
        Ok(())
    }
    async fn phase(
        phase: &mut BatchPhase,
        gpu: &Gpu,
        storage: &NvmeStorage,
        host: &mut AlignedBuffer,
    ) -> Result<()> {
        let mut c_streams = Streams::default();
        let mut t_streams = Streams::default();
        let result = async {
            for set in phase.schedule.clone() {
                let (c, t) = match set.execution_order {
                    ExecutionOrder::ControlFirst => {
                        let c = control_batch(
                            gpu,
                            storage,
                            host,
                            &set,
                            &mut phase.control,
                            &mut c_streams,
                        )
                        .await;
                        if let Err(e) = &c {
                            note_arm_error(&mut phase.control, e)?;
                        }
                        let c = c?;
                        let t = treatment_batch(
                            gpu,
                            storage,
                            &set,
                            &mut phase.treatment,
                            &mut t_streams,
                        )
                        .await;
                        if let Err(e) = &t {
                            note_arm_error(&mut phase.treatment, e)?;
                        }
                        (c, t?)
                    }
                    ExecutionOrder::TreatmentFirst => {
                        let t = treatment_batch(
                            gpu,
                            storage,
                            &set,
                            &mut phase.treatment,
                            &mut t_streams,
                        )
                        .await;
                        if let Err(e) = &t {
                            note_arm_error(&mut phase.treatment, e)?;
                        }
                        let t = t?;
                        let c = control_batch(
                            gpu,
                            storage,
                            host,
                            &set,
                            &mut phase.control,
                            &mut c_streams,
                        )
                        .await;
                        if let Err(e) = &c {
                            note_arm_error(&mut phase.control, e)?;
                        }
                        (c?, t)
                    }
                };
                phase
                    .raw_samples
                    .push(RawSample::new(set.clone(), c.0, t.0)?);
                phase.compare(&set, &c.1, &t.1)?;
                if phase.mismatch_count != 0 {
                    return Err(Failure::runtime(
                        "hash-parity-failed",
                        "source/payload/GPU/epoch mismatch",
                    ));
                }
            }
            Ok(())
        }
        .await;
        // The helper joins every worker before returning, including on errors.
        // This is an idle snapshot, with no diagnostic storage shared elsewhere.
        phase.fd_proof = storage.source_upload_fd_proof_snapshot();
        phase.fd_proof_hits_only =
            proof_hits_only(&phase.fd_proof, phase.expected.expert_slot_count);
        phase.control.source_schedule = schedule_evidence(&phase.control.completed_source_sets)?;
        phase.treatment.source_schedule =
            schedule_evidence(&phase.treatment.completed_source_sets)?;
        phase.control_witnesses = c_streams.snapshot();
        phase.treatment_witnesses = t_streams.snapshot();
        phase.statistics = statistics(&phase.raw_samples)?;
        phase.control.evidence.rates();
        phase.treatment.evidence.rates();
        result?;
        if !phase.fd_proof_hits_only {
            return Err(Failure::authority(format!(
                "{} proof state is not all hits: {:?}",
                phase.name, phase.fd_proof
            )));
        }
        Ok(())
    }
    async fn execute(report: &mut BatchReport) -> Result<()> {
        if !(2..=MAX_ITERATIONS).contains(&report.args.iterations)
            || report.args.warmup_iterations > MAX_ITERATIONS
        {
            return Err(Failure::runtime("invalid-arguments", "measured sets must be 2..=65536; warmup sets 0..=65536; FIRST counts must be explicitly frozen by reviewer"));
        }
        report.warmup = BatchPhase::new("warmup", report.args.warmup_iterations)?;
        report.measured = BatchPhase::new("measured", report.args.iterations)?;
        let bytes =
            std::fs::read(&report.args.config).map_err(|e| Failure::runtime("config-failed", e))?;
        report.config_sha256 = Some(sha(&bytes));
        let text = std::str::from_utf8(&bytes).map_err(|e| Failure::runtime("config-failed", e))?;
        let config: Config =
            toml::from_str(text).map_err(|e| Failure::runtime("config-failed", e))?;
        config.validate().map_err(Failure::authority)?;
        let a = &mut report.authority;
        a.base.direct_io_requested = !config.storage.no_direct;
        a.base.packed_storage =
            Some(config.storage.packed_blob.is_some() || config.storage.packed_manifest.is_some());
        a.base.source_data_dir = Some(config.model.data_dir.clone());
        validate_geometry(&config)?;
        a.base.exact_geometry = true;
        if !a.base.linux
            || report.args.expected_adapter_name != "NVIDIA L4"
            || !a.base.direct_io_requested
            || a.base.packed_storage != Some(false)
        {
            return Err(Failure::authority("requires Linux, exact NVIDIA L4 Vulkan, O_DIRECT, unpacked full-file Qwen geometry"));
        }
        let storage = NvmeStorage::new(StorageConfig {
            base_path: config.model.data_dir,
            expert_size: FULL,
            block_align: ALIGN,
            use_direct_io: true,
            num_experts_per_layer: Some(128),
        })
        .map_err(|e| Failure::runtime("source-failed", e))?;
        if storage.is_packed() {
            return Err(Failure::authority("packed storage forbidden"));
        }
        a.fd_cache_capacity = storage.max_open_files();
        if a.fd_cache_capacity < a.preproof_universe_size {
            return Err(Failure::authority(
                "fd cache cannot retain the complete deterministic universe",
            ));
        }
        let preproof = storage.preprove_source_upload_fds(&a.preproof_ordered_expert_ids);
        a.preproof = storage.source_upload_fd_proof_snapshot();
        preproof.map_err(|e| Failure::authority(format!("fd preproof: {e}")))?;
        if a.preproof.source_upload_fd_proof_requests != a.preproof_universe_size as u64
            || a.preproof.source_upload_fd_proof_misses != a.preproof_universe_size as u64
            || a.preproof.source_upload_fd_proof_hits != 0
            || a.preproof.source_upload_fd_proof_failures != 0
        {
            return Err(Failure::authority(
                "fresh universe preproof counters do not reconcile",
            ));
        }
        a.fd_preproof_completed = true;
        storage.reset_source_upload_fd_proof_telemetry();
        a.after_preproof_telemetry_reset = storage.source_upload_fd_proof_snapshot();
        if a.after_preproof_telemetry_reset != SourceUploadFdProofSnapshot::default() {
            return Err(Failure::authority("preproof telemetry reset failed"));
        }
        let gpu = Gpu::with_upload_capacity(&mut a.base, MAPPED_ARENA).await?;
        let mut host = AlignedBuffer::new(HOST_ARENA, ALIGN);
        phase(&mut report.warmup, &gpu, &storage, &mut host).await?;
        if !report.warmup.successful() {
            return Err(Failure::authority("warmup evidence did not reconcile"));
        }
        storage.reset_source_upload_fd_proof_telemetry();
        report.authority.after_warmup_telemetry_reset = storage.source_upload_fd_proof_snapshot();
        if report.authority.after_warmup_telemetry_reset != SourceUploadFdProofSnapshot::default() {
            return Err(Failure::authority("warmup telemetry reset failed"));
        }
        phase(&mut report.measured, &gpu, &storage, &mut host).await?;
        gpu.check()?;
        report.classify();
        Ok(())
    }
    pub(super) async fn run_command(
        args: Args,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args.report_out)?;
        let mut report = BatchReport::new(args).map_err(|e| io::Error::other(e.detail))?;
        match std::panic::AssertUnwindSafe(execute(&mut report))
            .catch_unwind()
            .await
        {
            Ok(Ok(())) => {}
            Ok(Err(e)) => report.fail(e),
            Err(p) => {
                let detail = p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "non-string panic".into());
                report.fail(Failure::runtime(
                    "runtime-failed",
                    format!("B diagnostic panic: {detail}"),
                ));
            }
        }
        serde_json::to_writer_pretty(&mut output, &report)?;
        output.write_all(b"\n")?;
        output.sync_all()?;
        if report.complete {
            Ok(())
        } else {
            Err(io::Error::other(format!(
                "{}: {}",
                report.classification,
                report.failure.as_deref().unwrap_or("incomplete")
            ))
            .into())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        fn args() -> Args {
            Args {
                config: "unused.toml".into(),
                expected_adapter_name: "NVIDIA L4".into(),
                warmup_iterations: 8,
                iterations: 128,
                report_out: "unused.json".into(),
            }
        }
        fn fixture_phase(name: &'static str, count: usize, treatment_ns: u64) -> BatchPhase {
            let mut p = BatchPhase::new(name, count).unwrap();
            p.raw_samples = p
                .schedule
                .iter()
                .cloned()
                .map(|s| RawSample::new(s, 1000, treatment_ns).unwrap())
                .collect();
            p.statistics = statistics(&p.raw_samples).unwrap();
            let n = p.expected.expert_slot_count;
            let sets = p.expected.source_set_count;
            for (t, arm) in [(false, &mut p.control), (true, &mut p.treatment)] {
                arm.source_sets_attempted = sets;
                arm.batch_helper_calls = sets;
                arm.completed_source_sets = p.schedule.clone();
                arm.source_schedule = p.expected.clone();
                let a = &mut arm.evidence;
                a.ops_attempted = n;
                a.source_read_attempts = n;
                a.source_read_ops = n;
                a.full_source_bytes = n * FULL as u64;
                a.payload_ops = n;
                a.payload_bytes = n * PAYLOAD as u64;
                a.upload_ops = n;
                a.gpu_copied_bytes = n * PAYLOAD as u64;
                a.epoch_bytes = n * 4;
                a.gpu_completed_ops = n;
                a.verified_ops = n;
                a.verification_readback_bytes = n * SLOT as u64;
                a.verification_destination_reset_ops = n;
                a.verification_destination_reset_bytes = n * SLOT as u64;
                a.pointers.observations = n;
                a.pointers.aligned = n;
                a.fd_evidence = FdEvidence {
                    checks: n,
                    direct_observed: n,
                    full_file_length_observed: n,
                    ..FdEvidence::default()
                };
                a.times.source_direct_read_ns = sets * if t { treatment_ns } else { 1000 };
                if t {
                    a.map_attempts = sets;
                    a.maps_completed = sets;
                    a.unmaps = sets;
                    a.pointers.gpu_offset_checks = n;
                    a.explicit_copy_buffer_bytes = n * PAYLOAD as u64;
                } else {
                    a.cpu_payload_copy_bytes = n * PAYLOAD as u64;
                }
            }
            p.control_witnesses = Witnesses {
                full_source_sha256: sha(b"source"),
                bare_payload_sha256: sha(b"payload"),
                gpu_destination_payload_sha256: sha(b"payload"),
            };
            p.treatment_witnesses = Witnesses {
                full_source_sha256: sha(b"source"),
                bare_payload_sha256: sha(b"payload"),
                gpu_destination_payload_sha256: sha(b"payload"),
            };
            p.fd_proof = SourceUploadFdProofSnapshot {
                source_upload_fd_proof_requests: n * 2,
                source_upload_fd_proof_hits: n * 2,
                ..SourceUploadFdProofSnapshot::default()
            };
            p.fd_proof_hits_only = true;
            assert!(p.successful());
            p
        }
        fn fixture(treatment_ns: u64) -> BatchReport {
            let mut r = BatchReport::new(args()).unwrap();
            let a = &mut r.authority;
            a.base.linux = true;
            a.base.adapter_authoritative = true;
            a.base.direct_io_requested = true;
            a.base.packed_storage = Some(false);
            a.base.exact_geometry = true;
            a.fd_cache_capacity = 256;
            a.fd_preproof_completed = true;
            a.preproof = SourceUploadFdProofSnapshot {
                source_upload_fd_proof_requests: 128,
                source_upload_fd_proof_misses: 128,
                ..SourceUploadFdProofSnapshot::default()
            };
            r.warmup = fixture_phase("warmup", 8, 1000);
            r.measured = fixture_phase("measured", 128, treatment_ns);
            r
        }
        #[test]
        fn source_to_upload_copy_elision_b_schedule_and_hashes_pinned() {
            let s = schedule(128).unwrap();
            let first = vec![
                vec![0],
                vec![822, 1451],
                vec![1644, 2273, 2902],
                vec![2466, 3095, 3724, 4353],
                vec![3289, 3917, 4546, 5175, 5804],
                vec![4111, 4740, 5369, 5997, 435, 1064],
                vec![4933, 5562, 0, 628, 1257, 1886, 2515],
                vec![5756, 193, 822, 1451, 2079, 2708, 3337, 3966],
                vec![386],
            ];
            assert_eq!(
                s[..9]
                    .iter()
                    .map(|s| s.ordered_expert_ids.clone())
                    .collect::<Vec<_>>(),
                first
            );
            assert_eq!(
                s[127].ordered_expert_ids,
                vec![5369, 5997, 435, 1064, 1692, 2321, 2950, 3579]
            );
            let e = schedule_evidence(&s).unwrap();
            assert_eq!(
                (e.source_set_count, e.expert_slot_count, e.source_bytes),
                (128, 576, 1_531_183_104)
            );
            assert_eq!(
                e.ordered_source_set_ids_sha256,
                "e916156e4e83f0b34f2e66cf106dc3cf80ad42ca59dd6945ea79edd245f14321"
            );
            assert_eq!(
                e.ordered_width_sha256,
                "2a4d07514866ea5a3d597284e6c2799e60a5726bf1704c0ce80058a66dfcfecc"
            );
            assert_eq!(e.width_histogram, (1..=8).map(|k| (k, 16)).collect());
            assert_eq!(s, schedule(128).unwrap());
            for k in 1..=8 {
                for order in [ExecutionOrder::ControlFirst, ExecutionOrder::TreatmentFirst] {
                    assert_eq!(
                        s.iter()
                            .filter(|s| s.width == k && s.execution_order == order)
                            .count(),
                        8
                    );
                }
            }
            let all = schedule(MAX_ITERATIONS).unwrap();
            for (p, s) in all.iter().enumerate() {
                assert_eq!((s.set_index, s.round, s.width), (p, p / 8, p % 8 + 1));
                assert_eq!(
                    s.ordered_expert_ids
                        .iter()
                        .copied()
                        .collect::<BTreeSet<_>>()
                        .len(),
                    s.width
                );
                assert!(s.ordered_expert_ids.iter().all(|id| *id < NAMESPACE));
            }
            let mut reordered = s.clone();
            reordered.swap(0, 1);
            assert_ne!(schedule_evidence(&reordered).unwrap(), e);
            let mut ids_reversed = s;
            ids_reversed[7].ordered_expert_ids.reverse();
            assert_ne!(
                schedule_evidence(&ids_reversed)
                    .unwrap()
                    .ordered_source_set_ids_sha256,
                e.ordered_source_set_ids_sha256
            );
            assert_eq!(
                schedule_evidence(&schedule(8).unwrap())
                    .unwrap()
                    .expert_slot_count,
                36
            );
            assert!(schedule(MAX_ITERATIONS + 1).is_err());
        }
        #[test]
        fn source_to_upload_copy_elision_b_arenas_exact_nonoverlapping_checked() {
            assert_eq!(HOST_ARENA, 21_266_432);
            assert_eq!(MAPPED_ARENA, 21_270_528);
            let mut host = AlignedBuffer::new(HOST_ARENA, ALIGN);
            let address = host.as_slice().as_ptr() as usize;
            assert_eq!(address % ALIGN, 0);
            for k in 1..=8 {
                let (offset, mut slices) = arena_slices(host.as_mut_slice(), k, false).unwrap();
                assert_eq!(offset, 0);
                assert_eq!(slices.len(), k);
                for (j, s) in slices.iter_mut().enumerate() {
                    assert_eq!(s.len(), FULL);
                    assert_eq!(s.as_ptr() as usize, address + j * FULL);
                    s[0] = j as u8;
                }
            }
            // Model every address residue, including WGPU copy misalignment.
            for residue in 0..ALIGN {
                for width in 1..=8 {
                    let offset = arena_offset(ALIGN * 16 + residue, MAPPED_ARENA, width, true);
                    if residue % 4 == 0 {
                        let offset = offset.unwrap();
                        assert_eq!((residue + offset) % ALIGN, 0);
                        assert!(offset + 8 * FULL <= MAPPED_ARENA);
                    } else {
                        assert!(offset.is_err());
                    }
                }
            }
            // Exercise real safe mutable slices into a simulated mapped arena.
            let mut backing = AlignedBuffer::new(MAPPED_ARENA + ALIGN, ALIGN);
            let start = 8;
            let base = backing.as_slice().as_ptr() as usize + start;
            let (offset, slices) = arena_slices(
                &mut backing.as_mut_slice()[start..start + MAPPED_ARENA],
                8,
                true,
            )
            .unwrap();
            assert_eq!(offset, 4088);
            for (j, s) in slices.into_iter().enumerate() {
                assert_eq!(s.as_ptr() as usize, base + offset + j * FULL);
                assert_eq!(s.len(), FULL);
                s.fill(j as u8);
            }
            assert!(backing.as_slice()[..ALIGN].iter().all(|b| *b == 0));
            assert!(backing.as_slice()[ALIGN + HOST_ARENA..]
                .iter()
                .all(|b| *b == 0));
            for (base, capacity, width, t) in [
                (0, MAPPED_ARENA, 8, true),
                (usize::MAX, MAPPED_ARENA, 8, true),
                (ALIGN, MAPPED_ARENA - 1, 8, true),
                (ALIGN, MAPPED_ARENA, 0, true),
                (ALIGN, MAPPED_ARENA, 9, true),
                (ALIGN + 4, HOST_ARENA, 8, false),
            ] {
                assert!(arena_offset(base, capacity, width, t).is_err());
            }
        }
        #[test]
        fn source_to_upload_copy_elision_b_stats_and_primary_reconcile() {
            let r = fixture(1029);
            let s = &r.measured.statistics;
            assert_eq!(
                s.primary_k2_through_k8.paired.paired_source_read_samples,
                112
            );
            assert_eq!(s.primary_k2_through_k8.control_first.samples, 56);
            assert_eq!(s.primary_k2_through_k8.treatment_first.samples, 56);
            assert_eq!(
                s.primary_k2_through_k8.paired.paired_control_source_read_ns,
                112_000
            );
            assert_eq!(
                s.primary_k2_through_k8
                    .paired
                    .paired_treatment_source_read_ns,
                115_248
            );
            assert_eq!(s.primary_k2_through_k8.paired.treatment_slower_pairs, 112);
            assert_eq!(
                s.primary_k2_through_k8.aggregate_slowdown_percent,
                Some(3248.0 / 112000.0 * 100.0)
            );
            assert!(s.primary_k2_through_k8.aggregate_slowdown_percent.unwrap() < 3.0);
            assert_eq!(s.positive_multi_expert_widths, 7);
            for k in 1..=8 {
                let w = &s.per_width[&k];
                assert_eq!(w.paired.paired_source_read_samples, 16);
                assert_eq!((w.control_first.samples, w.treatment_first.samples), (8, 8));
                assert_eq!(w.paired.mean_treatment_minus_control_ns, Some(29.0));
                assert_eq!(w.paired.median_treatment_minus_control_ns, Some(29.0));
                assert_eq!(w.paired.median_treatment_over_control_ratio, Some(1.029));
            }
            let mut samples = r.measured.raw_samples.clone();
            for s in &mut samples {
                if s.source_set.width == 1 {
                    *s = RawSample::new(s.source_set.clone(), 1, 999999).unwrap();
                }
            }
            assert_eq!(
                statistics(&samples).unwrap().primary_k2_through_k8,
                s.primary_k2_through_k8
            );
            let mut s = schedule(3)
                .unwrap()
                .into_iter()
                .zip([(10, 9), (10, 10), (10, 12)])
                .map(|(s, (c, t))| RawSample::new(s, c, t).unwrap())
                .collect::<Vec<_>>();
            let stats = batch_stats(&s).unwrap();
            assert_eq!(
                (
                    stats.paired.treatment_slower_pairs,
                    stats.paired.treatment_faster_pairs,
                    stats.paired.equal_pairs
                ),
                (1, 1, 1)
            );
            assert_eq!(stats.paired.median_treatment_minus_control_ns, Some(0.0));
            s.reverse();
            assert_eq!(batch_stats(&s).unwrap(), stats);
        }
        #[test]
        fn source_to_upload_copy_elision_b_schema_correctness_independent_of_performance() {
            for t in [1, 970, 990, 1000, 1010, 1029, 1030, 1050, 10000] {
                let mut r = fixture(t);
                r.classify();
                assert!(r.complete && r.correctness_pass && r.authoritative);
                let j = serde_json::to_value(&r).unwrap();
                assert_eq!(j["schema"], B_SCHEMA);
                assert_ne!(B_SCHEMA, SCHEMA);
                assert_eq!(j["performance_required_for_correctness"], false);
                let a = &j["authority"];
                assert_eq!(a["same_batch_source_api"], true);
                assert_eq!(a["control_source_api"], B_API);
                assert_eq!(a["treatment_source_api"], B_API);
                assert_eq!(a["control_destination"], "aligned-host-arena");
                assert_eq!(a["treatment_destination"], "wgpu-map-write-arena");
                let sample = &j["measured"]["raw_samples"][8];
                for key in [
                    "set_index",
                    "round",
                    "width",
                    "ordered_expert_ids",
                    "execution_order",
                    "control_ns",
                    "treatment_ns",
                    "delta_ns",
                ] {
                    assert!(sample.get(key).is_some(), "{key}");
                }
                assert_eq!(sample["execution_order"], "TREATMENT-then-CONTROL");
                assert!(r
                    .interpretation_contract
                    .contains("Do not round a sub-3% result upward"));
            }
            let mut r = fixture(1000);
            r.args.warmup_iterations = 0;
            r.args.iterations = 19;
            r.warmup = fixture_phase("warmup", 0, 1000);
            r.measured = fixture_phase("measured", 19, 1050);
            r.classify();
            assert!(r.correctness_pass); // FIRST counts are not a correctness predicate.
        }
        #[test]
        fn source_to_upload_copy_elision_b_proof_miss_and_evidence_tamper_fail_closed() {
            let mutations: Vec<fn(&mut BatchReport)> = vec![
                |r| r.authority.fd_preproof_completed = false,
                |r| r.authority.fd_cache_capacity = 127,
                |r| r.authority.preproof.source_upload_fd_proof_failures = 1,
                |r| {
                    r.authority
                        .after_preproof_telemetry_reset
                        .source_upload_fd_proof_requests = 1
                },
                |r| {
                    r.authority
                        .after_warmup_telemetry_reset
                        .source_upload_fd_proof_requests = 1
                },
                |r| r.authority.preproof_ordered_expert_ids[0] = 1,
                |r| r.authority.same_batch_source_api = false,
                |r| r.authority.base.control_source_api = SOURCE_API,
                |r| r.authority.base.treatment_source_api = SOURCE_API,
                |r| r.authority.base.control_destination = "pool",
                |r| r.authority.max_batch_width = 7,
                |r| r.authority.source_timer_excludes_fd_preproof = false,
                |r| r.authority.source_timer_excludes_slice_setup = false,
                |r| r.measured.fd_proof.source_upload_fd_proof_misses = 1,
                |r| r.measured.fd_proof.source_upload_fd_proof_failures = 1,
                |r| r.measured.fd_proof.source_upload_fd_proof_hits -= 1,
                |r| r.warmup.fd_proof.source_upload_fd_proof_misses = 1,
                |r| r.measured.control.batch_helper_calls += 1,
                |r| r.measured.treatment.evidence.source_read_ops -= 1,
                |r| r.measured.control.source_schedule.source_bytes -= 1,
                |r| r.measured.control.source_schedule.expert_slot_count -= 1,
                |r| r.measured.treatment.source_schedule.ordered_width_sha256 = "bad".into(),
                |r| r.measured.control.evidence.fd_evidence.direct_observed -= 1,
                |r| r.measured.treatment.evidence.pointers.aligned -= 1,
                |r| r.measured.treatment.evidence.unmaps -= 1,
                |r| r.measured.treatment.evidence.source_failures = 1,
                |r| r.measured.treatment.evidence.fallback_reads = 1,
                |r| r.measured.treatment.evidence.mapped_direct_io_rejections = 1,
                |r| r.measured.treatment.evidence.gpu_failures = 1,
                |r| r.measured.treatment.evidence.map_failures = 1,
                |r| r.measured.treatment.evidence.cpu_payload_copy_bytes = 1,
                |r| r.measured.treatment_witnesses.full_source_sha256 = "bad".into(),
                |r| r.measured.mismatch_count = 1,
                |r| {
                    r.measured.raw_samples[0].source_set.execution_order =
                        ExecutionOrder::TreatmentFirst
                },
                |r| r.measured.raw_samples[0].delta_ns = 1,
                |r| r.measured.raw_samples.swap(0, 1),
                |r| {
                    r.measured
                        .statistics
                        .per_width
                        .get_mut(&2)
                        .unwrap()
                        .paired
                        .paired_control_source_read_ns += 1
                },
                |r| {
                    r.measured
                        .statistics
                        .primary_k2_through_k8
                        .paired
                        .equal_pairs += 1
                },
                |r| {
                    r.measured
                        .statistics
                        .descriptive_all_k
                        .control_first
                        .samples += 1
                },
            ];
            for (i, mutate) in mutations.into_iter().enumerate() {
                let mut r = fixture(1000);
                mutate(&mut r);
                r.classify();
                assert!(!r.correctness_pass && !r.authoritative, "mutation {i}");
            }
            assert!(!proof_hits_only(
                &SourceUploadFdProofSnapshot::default(),
                u64::MAX
            ));
        }
        #[test]
        fn source_to_upload_copy_elision_b_overflow_zero_and_invalid_schedules() {
            assert!(source_bytes(u64::MAX).is_err());
            let set = schedule(1).unwrap().remove(0);
            assert!(RawSample::new(set.clone(), 0, 1).is_err());
            assert!(RawSample::new(set.clone(), 1, 0).is_err());
            for (c, t) in [(u64::MAX, 1), (1, u64::MAX)] {
                let s = RawSample::new(set.clone(), c, t).unwrap();
                assert!(batch_stats(&[s.clone(), s]).is_err());
            }
            let mut s = RawSample::new(set.clone(), 1, 2).unwrap();
            s.delta_ns = 0;
            assert!(batch_stats(&[s]).is_err());
            let mut bad = set;
            bad.width = 2;
            bad.ordered_expert_ids = vec![0, 0];
            assert!(schedule_evidence(&[bad.clone()]).is_err());
            bad.ordered_expert_ids = vec![0, NAMESPACE];
            assert!(schedule_evidence(&[bad]).is_err());
        }
        #[test]
        fn source_to_upload_copy_elision_b_one_batch_call_and_timer_scope() {
            let whole = include_str!("gpu_native_source_to_upload_copy_elision.rs");
            let b = whole
                .split("mod hma1c_b {")
                .nth(1)
                .unwrap()
                .split("#[cfg(test)]")
                .next()
                .unwrap();
            assert_eq!(
                b.matches(".read_experts_batch_into_aligned_slices(")
                    .count(),
                1
            );
            assert!(!b.contains(".read_expert("));
            assert!(!b.contains(".read_expert_into_aligned_slice("));
            let source = b
                .split("async fn batch_source(")
                .nth(1)
                .unwrap()
                .split("async fn control_batch(")
                .next()
                .unwrap();
            let timed = source
                .split("let start = Instant::now();")
                .nth(1)
                .unwrap()
                .split("let ns = elapsed(start)?;")
                .next()
                .unwrap();
            let compact: String = timed.split_whitespace().collect();
            assert_eq!(
                compact,
                "letread=storage.read_experts_batch_into_aligned_slices(ids,destinations).await;"
            );
            let control = b
                .split("async fn control_batch(")
                .nth(1)
                .unwrap()
                .split("async fn treatment_batch(")
                .next()
                .unwrap();
            let treatment = b
                .split("async fn treatment_batch(")
                .nth(1)
                .unwrap()
                .split("fn note_arm_error(")
                .next()
                .unwrap();
            for arm in [control, treatment] {
                assert_eq!(arm.matches("batch_source(").count(), 1);
                assert!(arm.find("arena_slices(").unwrap() < arm.find("batch_source(").unwrap());
                assert!(arm.find("batch_source(").unwrap() < arm.find("streams.source(").unwrap());
                assert!(arm.find("batch_source(").unwrap() < arm.find("verify(").unwrap());
            }
            assert!(treatment.find("gpu.map(").unwrap() < treatment.find("batch_source(").unwrap());
            assert!(
                treatment.find("get_mapped_range_mut()").unwrap()
                    < treatment.find("batch_source(").unwrap()
            );
            assert!(
                treatment.find("batch_source(").unwrap()
                    < treatment.find("gpu.upload.unmap()").unwrap()
            );
            assert!(
                treatment.find("gpu.upload.unmap()").unwrap()
                    < treatment.find("encoder.copy_buffer_to_buffer(").unwrap()
            );
            let run = b
                .split("async fn phase(")
                .nth(1)
                .unwrap()
                .split("async fn execute(")
                .next()
                .unwrap();
            assert!(run.contains("match set.execution_order"));
            assert!(!run.contains("% 2"));
            let first = run
                .split("ExecutionOrder::ControlFirst =>")
                .nth(1)
                .unwrap()
                .split("ExecutionOrder::TreatmentFirst =>")
                .next()
                .unwrap();
            assert!(
                first.find("control_batch(").unwrap() < first.find("treatment_batch(").unwrap()
            );
            let second = run
                .split("ExecutionOrder::TreatmentFirst =>")
                .nth(1)
                .unwrap();
            assert!(
                second.find("treatment_batch(").unwrap() < second.find("control_batch(").unwrap()
            );
            let exec_raw = b.split("async fn execute(").nth(1).unwrap();
            let exec: String = exec_raw.split_whitespace().collect();
            assert_eq!(exec.matches("AlignedBuffer::new(").count(), 1);
            assert_eq!(exec.matches("Gpu::with_upload_capacity(").count(), 1);
            assert!(
                exec.find("preprove_source_upload_fds(").unwrap()
                    < exec.find("phase(&mutreport.warmup").unwrap()
            );
            assert_eq!(
                exec.matches("reset_source_upload_fd_proof_telemetry()")
                    .count(),
                2
            );
            assert!(
                exec.find("AlignedBuffer::new(").unwrap()
                    < exec.find("phase(&mutreport.warmup").unwrap()
            );
        }
        #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
        async fn source_to_upload_copy_elision_b_runner_preserves_report_without_gpu() {
            let dir = std::env::temp_dir().join(format!(
                "mer-hma1cb-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let mut a = args();
            a.config = dir.join("missing.toml");
            a.report_out = dir.join("report.json");
            assert!(run_command(a.clone()).await.is_err());
            let bytes = std::fs::read(&a.report_out).unwrap();
            let j: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(j["schema"], B_SCHEMA);
            assert_eq!(j["classification"], "config-failed");
            assert_eq!(j["correctness_pass"], false);
            assert_eq!(j["measured"]["expected"]["expert_slot_count"], 576);
            assert!(run_command(a.clone()).await.is_err());
            assert_eq!(std::fs::read(&a.report_out).unwrap(), bytes);
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
    /// HMA-1C-C is nested only to reuse B's private, frozen schedule/geometry.
    /// A and B runners, evidence contracts and tests remain version-separated.
    pub(super) mod hma1c_c {
        use super::*;
        use std::cmp::Ordering;

        const C_SCHEMA: &str = "mer.gpu-native-mapped-memory-odirect-concurrency-interaction.v1";
        const SERIAL_API: &str = "read_experts_serial_into_aligned_slices";
        const MEASURED_B_COUNT: usize = 128;
        const WARMUP_B_COUNT: usize = 32;
        const C_SCHEDULE_CONTRACT: &str = "Exactly B schedule(128) filtered K>=2 measured; B schedule(32) filtered K>=2 warmup. Original B p, round and ordered IDs retained. Williams class = zero-based occurrence within width modulo 4; A=HS B=HC C=MS D=MC: ABDC, BCAD, CDBA, DACB. Source IDs/width hashes use B encoding. Execution SHA256 per set: p, round, K, width occurrence, class as u64 LE, then four cell codes as u8 (HS=0 HC=1 MS=2 MC=3). Complete schedule SHA256 per set: same five u64 LE fields, K IDs as u32 LE, then four u8 cell codes. Cross-run C-vs-B timings are descriptive only.";
        const C_TIMER_CONTRACT: &str = "Each cell times ONLY one awaited source helper call. HS/MS: read_experts_serial_into_aligned_slices; HC/MC: unchanged read_experts_batch_into_aligned_slices. Dispatch, counters, allocation, map_async, device.poll, mapped-view acquisition, checked alignment/slices, fd preproof/evidence, hashing, GPU copy/readback/verification and unmap are outside the source timer. Normal fd resolution, proof hits, retries and breaker behavior remain inside the helper.";
        const C_INTERPRETATION: &str = "Primary=sum((MC-HC)-(MS-HS)); normalize by sum(HC)*100. STRONG requires >=5%, median exact interaction>0, majority positive sets, >=5/7 positive widths and all four positive Williams classes. MATERIAL uses the same conditions at >=3%. AGAINST requires abs(interaction)<=1%, exact median multiplicative ratio-of-ratios in [0.99,1.01], and fewer than five positive widths. <=-3% with analogous negative consistency means concurrency REDUCES the mapped penalty. Other results are AMBIGUOUS, including >1% but <3%, aggregate/median/majority disagreement, insufficient width support or any order-class reversal. No rounding or threshold adjustment after hardware. MC>HC alone is not interaction evidence.";
        const C_SECONDARY: &str = "Serial and concurrent mapped-vs-host are secondary: report aggregate slowdown, exact median delta/ratio, majority, per-width and Williams order evidence. Weak serial + material concurrent + material positive interaction supports concurrency-specific amplification; material serial and concurrent + weak interaction supports raw destination main effect; neither coherent gives evidence against mapped backing/concurrency as dominant source-gap cause; mixed consistency remains ambiguous / inspect next surrounding helper.";
        const C_RETRY: &str = "Performance authority additionally requires zero exact occurrences of 'transient I/O error; retrying' in the complete external FIRST log. The report cannot prove zero internal transient retries. One or more occurrences means RETRY_CONTAMINATED / INCONCLUSIVE for performance. No diagnostic retry or fallback. Interpretation fields are conditional on the external zero-occurrence audit.";

        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
        enum Cell {
            HS,
            HC,
            MS,
            MC,
        }
        impl Cell {
            const ALL: [Self; 4] = [Self::HS, Self::HC, Self::MS, Self::MC];
            fn index(self) -> usize {
                self as usize
            }
            fn mapped(self) -> bool {
                matches!(self, Self::MS | Self::MC)
            }
            fn serial(self) -> bool {
                matches!(self, Self::HS | Self::MS)
            }
            fn api(self) -> &'static str {
                if self.serial() {
                    SERIAL_API
                } else {
                    B_API
                }
            }
        }
        const WILLIAMS: [[Cell; 4]; 4] = [
            [Cell::HS, Cell::HC, Cell::MC, Cell::MS],
            [Cell::HC, Cell::MS, Cell::HS, Cell::MC],
            [Cell::MS, Cell::MC, Cell::HC, Cell::HS],
            [Cell::MC, Cell::HS, Cell::MS, Cell::HC],
        ];
        #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
        struct CSet {
            set_index: usize,
            round: usize,
            width: usize,
            ordered_expert_ids: Vec<u32>,
            width_occurrence: usize,
            williams_order_class: usize,
            execution_sequence: [Cell; 4],
        }
        impl CSet {
            fn b_set(&self) -> SourceSet {
                SourceSet {
                    set_index: self.set_index,
                    round: self.round,
                    width: self.width,
                    ordered_expert_ids: self.ordered_expert_ids.clone(),
                    execution_order: if self.round % 2 == 0 {
                        ExecutionOrder::ControlFirst
                    } else {
                        ExecutionOrder::TreatmentFirst
                    },
                }
            }
        }
        fn c_schedule(b_count: usize) -> Result<Vec<CSet>> {
            if ![0, WARMUP_B_COUNT, MEASURED_B_COUNT].contains(&b_count) {
                return Err(Failure::accounting(
                    "C requires frozen B schedule counts 32/128",
                ));
            }
            let mut occurrences = [0usize; MAX_WIDTH + 1];
            schedule(b_count)?
                .into_iter()
                .filter(|s| s.width >= 2)
                .map(|s| {
                    let r = occurrences[s.width];
                    occurrences[s.width] = r
                        .checked_add(1)
                        .ok_or_else(|| Failure::accounting("width occurrence overflow"))?;
                    Ok(CSet {
                        set_index: s.set_index,
                        round: s.round,
                        width: s.width,
                        ordered_expert_ids: s.ordered_expert_ids,
                        width_occurrence: r,
                        williams_order_class: r % 4,
                        execution_sequence: WILLIAMS[r % 4],
                    })
                })
                .collect()
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
        struct CPlan {
            #[serde(flatten)]
            source: ScheduleEvidence,
            execution_order_sha256: String,
            complete_schedule_sha256: String,
        }
        fn plan(sets: &[CSet]) -> Result<CPlan> {
            let mut order = Sha256::new();
            let mut complete = Sha256::new();
            let mut occurrences = [0usize; MAX_WIDTH + 1];
            for s in sets {
                if !(2..=MAX_WIDTH).contains(&s.width)
                    || s.width_occurrence != occurrences[s.width]
                    || s.williams_order_class != s.width_occurrence % 4
                    || s.execution_sequence != WILLIAMS[s.williams_order_class]
                {
                    return Err(Failure::accounting("invalid Williams source set"));
                }
                occurrences[s.width] = occurrences[s.width]
                    .checked_add(1)
                    .ok_or_else(|| Failure::accounting("order count overflow"))?;
                for value in [
                    s.set_index,
                    s.round,
                    s.width,
                    s.width_occurrence,
                    s.williams_order_class,
                ] {
                    let bytes = u64::try_from(value)
                        .map_err(Failure::accounting)?
                        .to_le_bytes();
                    order.update(bytes);
                    complete.update(bytes);
                }
                for id in &s.ordered_expert_ids {
                    complete.update(id.to_le_bytes());
                }
                for cell in s.execution_sequence {
                    order.update([cell as u8]);
                    complete.update([cell as u8]);
                }
            }
            Ok(CPlan {
                source: schedule_evidence(&sets.iter().map(CSet::b_set).collect::<Vec<_>>())?,
                execution_order_sha256: finish_sha(&order),
                complete_schedule_sha256: finish_sha(&complete),
            })
        }
        fn signed_add(a: i128, b: i128) -> Result<i128> {
            a.checked_add(b)
                .ok_or_else(|| Failure::accounting("signed sum overflow"))
        }
        fn signed_sub(a: i128, b: i128) -> Result<i128> {
            a.checked_sub(b)
                .ok_or_else(|| Failure::accounting("signed delta overflow"))
        }
        fn product(a: u128, b: u128) -> Result<u128> {
            a.checked_mul(b)
                .ok_or_else(|| Failure::accounting("interaction product overflow"))
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
        struct CSample {
            #[serde(flatten)]
            source_set: CSet,
            hs_ns: u64,
            ms_ns: u64,
            hc_ns: u64,
            mc_ns: u64,
            serial_destination_delta_ns: i128,
            concurrent_destination_delta_ns: i128,
            interaction_delta_ns: i128,
            mc_times_hs: u128,
            hc_times_ms: u128,
            multiplicative_interaction_direction: i8,
        }
        impl CSample {
            fn new(
                source_set: CSet,
                hs_ns: u64,
                ms_ns: u64,
                hc_ns: u64,
                mc_ns: u64,
            ) -> Result<Self> {
                if [hs_ns, ms_ns, hc_ns, mc_ns].contains(&0) {
                    return Err(Failure::accounting("zero C source duration"));
                }
                let serial_destination_delta_ns = signed_sub(ms_ns.into(), hs_ns.into())?;
                let concurrent_destination_delta_ns = signed_sub(mc_ns.into(), hc_ns.into())?;
                let interaction_delta_ns =
                    signed_sub(concurrent_destination_delta_ns, serial_destination_delta_ns)?;
                let mc_times_hs = product(mc_ns.into(), hs_ns.into())?;
                let hc_times_ms = product(hc_ns.into(), ms_ns.into())?;
                let multiplicative_interaction_direction = match mc_times_hs.cmp(&hc_times_ms) {
                    Ordering::Less => -1,
                    Ordering::Equal => 0,
                    Ordering::Greater => 1,
                };
                Ok(Self {
                    source_set,
                    hs_ns,
                    ms_ns,
                    hc_ns,
                    mc_ns,
                    serial_destination_delta_ns,
                    concurrent_destination_delta_ns,
                    interaction_delta_ns,
                    mc_times_hs,
                    hc_times_ms,
                    multiplicative_interaction_direction,
                })
            }
            fn valid(&self) -> bool {
                Self::new(
                    self.source_set.clone(),
                    self.hs_ns,
                    self.ms_ns,
                    self.hc_ns,
                    self.mc_ns,
                )
                .is_ok_and(|s| s == *self)
            }
            fn times(&self) -> [u64; 4] {
                [self.hs_ns, self.hc_ns, self.ms_ns, self.mc_ns]
            }
            fn ratio(&self) -> Ratio {
                Ratio {
                    numerator: self.mc_times_hs,
                    denominator: self.hc_times_ms,
                }
            }
        }
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
        struct Ratio {
            numerator: u128,
            denominator: u128,
        }
        impl Ratio {
            // Euclidean comparison of positive rationals: no overflowing cross
            // products and no floating-point ordering, even near u128::MAX.
            fn compare(self, other: Self) -> Ordering {
                let (mut a, mut b, mut c, mut d) = (
                    self.numerator,
                    self.denominator,
                    other.numerator,
                    other.denominator,
                );
                let mut reversed = false;
                loop {
                    let cmp = (a / b).cmp(&(c / d));
                    if cmp != Ordering::Equal {
                        return if reversed { cmp.reverse() } else { cmp };
                    }
                    let (r, s) = (a % b, c % d);
                    if r == 0 || s == 0 {
                        let cmp = r.cmp(&s);
                        return if reversed { cmp.reverse() } else { cmp };
                    }
                    (a, b, c, d) = (b, r, d, s);
                    reversed = !reversed;
                }
            }
            fn descriptive(self) -> f64 {
                self.numerator as f64 / self.denominator as f64
            }
        }
        // Five limbs retain exact 128x128 products scaled by the small integer
        // threshold, including the sum needed for an even-sample median.
        fn scaled_wide_product(a: u128, b: u128, scale: u64) -> Result<[u64; 5]> {
            let mut out = [0u64; 5];
            for (i, x) in [a as u64, (a >> 64) as u64].into_iter().enumerate() {
                let mut carry = 0u128;
                for (j, y) in [b as u64, (b >> 64) as u64].into_iter().enumerate() {
                    let n = u128::from(x)
                        .checked_mul(y.into())
                        .and_then(|n| n.checked_add(out[i + j].into()))
                        .and_then(|n| n.checked_add(carry))
                        .ok_or_else(|| Failure::accounting("wide product overflow"))?;
                    out[i + j] = n as u64;
                    carry = n >> 64;
                }
                out[i + 2] = u64::try_from(carry).map_err(Failure::accounting)?;
            }
            let mut carry = 0u128;
            for limb in &mut out {
                let n = u128::from(*limb)
                    .checked_mul(scale.into())
                    .and_then(|n| n.checked_add(carry))
                    .ok_or_else(|| Failure::accounting("wide scale overflow"))?;
                *limb = n as u64;
                carry = n >> 64;
            }
            if carry != 0 {
                return Err(Failure::accounting("wide capacity overflow"));
            }
            Ok(out)
        }
        fn wide_add(a: [u64; 5], b: [u64; 5]) -> Result<[u64; 5]> {
            let mut out = [0u64; 5];
            let mut carry = 0u128;
            for i in 0..5 {
                let n = u128::from(a[i])
                    .checked_add(b[i].into())
                    .and_then(|n| n.checked_add(carry))
                    .ok_or_else(|| Failure::accounting("wide addition overflow"))?;
                out[i] = n as u64;
                carry = n >> 64;
            }
            if carry != 0 {
                return Err(Failure::accounting("wide sum overflow"));
            }
            Ok(out)
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
        struct RatioMedian {
            lower: Ratio,
            upper: Ratio,
        }
        impl RatioMedian {
            fn new(mut ratios: Vec<(usize, Ratio)>) -> Option<Self> {
                if ratios.is_empty() {
                    return None;
                }
                ratios.sort_by(|a, b| a.1.compare(b.1).then(a.0.cmp(&b.0)));
                Some(Self {
                    lower: ratios[(ratios.len() - 1) / 2].1,
                    upper: ratios[ratios.len() / 2].1,
                })
            }
            fn compare_hundredths(&self, threshold: u64) -> Result<Ordering> {
                // (lo+hi)/2 ? threshold/100, exactly, without reducing ratios.
                let lhs = wide_add(
                    scaled_wide_product(self.lower.numerator, self.upper.denominator, 100)?,
                    scaled_wide_product(self.upper.numerator, self.lower.denominator, 100)?,
                )?;
                let rhs = scaled_wide_product(
                    self.lower.denominator,
                    self.upper.denominator,
                    threshold
                        .checked_mul(2)
                        .ok_or_else(|| Failure::accounting("median threshold overflow"))?,
                )?;
                Ok(lhs.iter().rev().cmp(rhs.iter().rev()))
            }
            fn descriptive(&self) -> f64 {
                self.lower.descriptive() / 2.0 + self.upper.descriptive() / 2.0
            }
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
        struct SignedFraction {
            numerator: i128,
            denominator: u64,
        }
        fn median_signed(mut values: Vec<i128>) -> Result<Option<SignedFraction>> {
            if values.is_empty() {
                return Ok(None);
            }
            values.sort();
            let (lo, hi) = (values[(values.len() - 1) / 2], values[values.len() / 2]);
            if values.len() % 2 == 1 {
                Ok(Some(SignedFraction {
                    numerator: hi,
                    denominator: 1,
                }))
            } else {
                Ok(Some(SignedFraction {
                    numerator: signed_add(lo, hi)?,
                    denominator: 2,
                }))
            }
        }
        #[derive(Clone, Default, Debug, PartialEq, Eq, Serialize)]
        struct Directions {
            positive: u64,
            negative: u64,
            equal: u64,
        }
        impl Directions {
            fn observe(&mut self, delta: i128) -> Result<()> {
                add(
                    if delta > 0 {
                        &mut self.positive
                    } else if delta < 0 {
                        &mut self.negative
                    } else {
                        &mut self.equal
                    },
                    1,
                )
            }
            fn count(&self) -> Result<u64> {
                self.positive
                    .checked_add(self.negative)
                    .and_then(|n| n.checked_add(self.equal))
                    .ok_or_else(|| Failure::accounting("direction count overflow"))
            }
        }
        #[derive(Clone, Default, Debug, PartialEq, Eq, Serialize)]
        struct Totals {
            samples: u64,
            hs_ns: u64,
            ms_ns: u64,
            hc_ns: u64,
            mc_ns: u64,
            serial_destination_delta_ns: i128,
            concurrent_destination_delta_ns: i128,
            interaction_delta_ns: i128,
            interaction_samples: Directions,
            serial_destination_samples: Directions,
            concurrent_destination_samples: Directions,
            multiplicative_interaction_samples: Directions,
        }
        impl Totals {
            fn observe(&mut self, s: &CSample) -> Result<()> {
                if !s.valid() {
                    return Err(Failure::accounting("C raw arithmetic does not reconcile"));
                }
                add(&mut self.samples, 1)?;
                add(&mut self.hs_ns, s.hs_ns)?;
                add(&mut self.ms_ns, s.ms_ns)?;
                add(&mut self.hc_ns, s.hc_ns)?;
                add(&mut self.mc_ns, s.mc_ns)?;
                self.serial_destination_delta_ns = signed_add(
                    self.serial_destination_delta_ns,
                    s.serial_destination_delta_ns,
                )?;
                self.concurrent_destination_delta_ns = signed_add(
                    self.concurrent_destination_delta_ns,
                    s.concurrent_destination_delta_ns,
                )?;
                self.interaction_delta_ns =
                    signed_add(self.interaction_delta_ns, s.interaction_delta_ns)?;
                self.interaction_samples.observe(s.interaction_delta_ns)?;
                self.serial_destination_samples
                    .observe(s.serial_destination_delta_ns)?;
                self.concurrent_destination_samples
                    .observe(s.concurrent_destination_delta_ns)?;
                self.multiplicative_interaction_samples
                    .observe(s.multiplicative_interaction_direction.into())?;
                self.reconcile()
            }
            fn reconcile(&self) -> Result<()> {
                if self.serial_destination_delta_ns
                    != signed_sub(self.ms_ns.into(), self.hs_ns.into())?
                    || self.concurrent_destination_delta_ns
                        != signed_sub(self.mc_ns.into(), self.hc_ns.into())?
                    || self.interaction_delta_ns
                        != signed_sub(
                            self.concurrent_destination_delta_ns,
                            self.serial_destination_delta_ns,
                        )?
                    || [
                        &self.interaction_samples,
                        &self.serial_destination_samples,
                        &self.concurrent_destination_samples,
                        &self.multiplicative_interaction_samples,
                    ]
                    .into_iter()
                    .any(|d| d.count().ok() != Some(self.samples))
                {
                    return Err(Failure::accounting("C totals do not reconcile"));
                }
                Ok(())
            }
            fn times(&self) -> [u64; 4] {
                [self.hs_ns, self.hc_ns, self.ms_ns, self.mc_ns]
            }
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
        struct ExactStats {
            #[serde(flatten)]
            totals: Totals,
            mean_signed_interaction_delta_ns: Option<SignedFraction>,
            median_signed_interaction_delta_ns: Option<SignedFraction>,
            median_ratio_of_ratios_exact: Option<RatioMedian>,
            serial_median_delta_ns: Option<SignedFraction>,
            concurrent_median_delta_ns: Option<SignedFraction>,
            serial_median_mapped_over_host_ratio_exact: Option<RatioMedian>,
            concurrent_median_mapped_over_host_ratio_exact: Option<RatioMedian>,
        }
        fn exact_stats(samples: &[CSample]) -> Result<ExactStats> {
            let mut totals = Totals::default();
            for s in samples {
                totals.observe(s)?;
            }
            let median = |f: fn(&CSample) -> i128| median_signed(samples.iter().map(f).collect());
            let ratios = |f: fn(&CSample) -> Ratio| {
                RatioMedian::new(
                    samples
                        .iter()
                        .map(|s| (s.source_set.set_index, f(s)))
                        .collect(),
                )
            };
            Ok(ExactStats {
                mean_signed_interaction_delta_ns: (totals.samples != 0).then(|| SignedFraction {
                    numerator: totals.interaction_delta_ns,
                    denominator: totals.samples,
                }),
                median_signed_interaction_delta_ns: median(|s| s.interaction_delta_ns)?,
                median_ratio_of_ratios_exact: ratios(CSample::ratio),
                serial_median_delta_ns: median(|s| s.serial_destination_delta_ns)?,
                concurrent_median_delta_ns: median(|s| s.concurrent_destination_delta_ns)?,
                serial_median_mapped_over_host_ratio_exact: ratios(|s| Ratio {
                    numerator: s.ms_ns.into(),
                    denominator: s.hs_ns.into(),
                }),
                concurrent_median_mapped_over_host_ratio_exact: ratios(|s| Ratio {
                    numerator: s.mc_ns.into(),
                    denominator: s.hc_ns.into(),
                }),
                totals,
            })
        }
        #[derive(Debug, Serialize)]
        struct DescriptiveStats {
            interaction_percent_of_hc: Option<f64>,
            serial_mapped_slowdown_percent: Option<f64>,
            concurrent_mapped_slowdown_percent: Option<f64>,
            slowdown_percentage_point_difference: Option<f64>,
            mean_signed_interaction_delta_ns: Option<f64>,
            median_signed_interaction_delta_ns: Option<f64>,
            median_ratio_of_ratios: Option<f64>,
            serial_median_delta_ns: Option<f64>,
            concurrent_median_delta_ns: Option<f64>,
            serial_median_mapped_over_host_ratio: Option<f64>,
            concurrent_median_mapped_over_host_ratio: Option<f64>,
        }
        fn descriptive(e: &ExactStats) -> DescriptiveStats {
            let t = &e.totals;
            let pct = |d: i128, n: u64| (n != 0).then(|| d as f64 / n as f64 * 100.0);
            let frac = |f: &Option<SignedFraction>| {
                f.as_ref()
                    .map(|f| f.numerator as f64 / f.denominator as f64)
            };
            let serial = pct(t.serial_destination_delta_ns, t.hs_ns);
            let concurrent = pct(t.concurrent_destination_delta_ns, t.hc_ns);
            DescriptiveStats {
                interaction_percent_of_hc: pct(t.interaction_delta_ns, t.hc_ns),
                serial_mapped_slowdown_percent: serial,
                concurrent_mapped_slowdown_percent: concurrent,
                slowdown_percentage_point_difference: concurrent.zip(serial).map(|(c, s)| c - s),
                mean_signed_interaction_delta_ns: frac(&e.mean_signed_interaction_delta_ns),
                median_signed_interaction_delta_ns: frac(&e.median_signed_interaction_delta_ns),
                median_ratio_of_ratios: e
                    .median_ratio_of_ratios_exact
                    .as_ref()
                    .map(RatioMedian::descriptive),
                serial_median_delta_ns: frac(&e.serial_median_delta_ns),
                concurrent_median_delta_ns: frac(&e.concurrent_median_delta_ns),
                serial_median_mapped_over_host_ratio: e
                    .serial_median_mapped_over_host_ratio_exact
                    .as_ref()
                    .map(RatioMedian::descriptive),
                concurrent_median_mapped_over_host_ratio: e
                    .concurrent_median_mapped_over_host_ratio_exact
                    .as_ref()
                    .map(RatioMedian::descriptive),
            }
        }
        #[derive(Debug, Serialize)]
        struct CStats {
            exact: ExactStats,
            descriptive: DescriptiveStats,
        }
        impl CStats {
            fn new(samples: &[CSample]) -> Result<Self> {
                let exact = exact_stats(samples)?;
                Ok(Self {
                    descriptive: descriptive(&exact),
                    exact,
                })
            }
        }
        #[derive(Debug, Serialize)]
        struct WidthStats {
            #[serde(flatten)]
            aggregate: CStats,
            williams_order_classes: BTreeMap<usize, CStats>,
        }
        #[derive(Debug, Serialize)]
        struct CStatistics {
            primary_k2_through_k8: CStats,
            per_width: BTreeMap<usize, WidthStats>,
            williams_order_classes: BTreeMap<usize, CStats>,
            interaction_widths: Directions,
            serial_destination_widths: Directions,
            concurrent_destination_widths: Directions,
        }
        fn reconcile_partitions<'a>(
            parts: impl Iterator<Item = &'a ExactStats>,
            primary: &ExactStats,
        ) -> Result<()> {
            let mut totals = Totals::default();
            for e in parts {
                let t = &e.totals;
                add(&mut totals.samples, t.samples)?;
                add(&mut totals.hs_ns, t.hs_ns)?;
                add(&mut totals.ms_ns, t.ms_ns)?;
                add(&mut totals.hc_ns, t.hc_ns)?;
                add(&mut totals.mc_ns, t.mc_ns)?;
                totals.serial_destination_delta_ns = signed_add(
                    totals.serial_destination_delta_ns,
                    t.serial_destination_delta_ns,
                )?;
                totals.concurrent_destination_delta_ns = signed_add(
                    totals.concurrent_destination_delta_ns,
                    t.concurrent_destination_delta_ns,
                )?;
                totals.interaction_delta_ns =
                    signed_add(totals.interaction_delta_ns, t.interaction_delta_ns)?;
                for (a, b) in [
                    (&mut totals.interaction_samples, &t.interaction_samples),
                    (
                        &mut totals.serial_destination_samples,
                        &t.serial_destination_samples,
                    ),
                    (
                        &mut totals.concurrent_destination_samples,
                        &t.concurrent_destination_samples,
                    ),
                    (
                        &mut totals.multiplicative_interaction_samples,
                        &t.multiplicative_interaction_samples,
                    ),
                ] {
                    add(&mut a.positive, b.positive)?;
                    add(&mut a.negative, b.negative)?;
                    add(&mut a.equal, b.equal)?;
                }
            }
            totals.reconcile()?;
            if totals != primary.totals {
                return Err(Failure::accounting("width/order partition mismatch"));
            }
            Ok(())
        }
        fn c_statistics(samples: &[CSample]) -> Result<CStatistics> {
            if samples.iter().any(|s| {
                !(2..=8).contains(&s.source_set.width) || s.source_set.williams_order_class >= 4
            }) {
                return Err(Failure::accounting("invalid C statistical stratum"));
            }
            let mut out = CStatistics {
                primary_k2_through_k8: CStats::new(samples)?,
                per_width: BTreeMap::new(),
                williams_order_classes: BTreeMap::new(),
                interaction_widths: Directions::default(),
                serial_destination_widths: Directions::default(),
                concurrent_destination_widths: Directions::default(),
            };
            let order_stats = |subset: &[CSample]| -> Result<BTreeMap<usize, CStats>> {
                (0..4)
                    .map(|class| {
                        Ok((
                            class,
                            CStats::new(
                                &subset
                                    .iter()
                                    .filter(|s| s.source_set.williams_order_class == class)
                                    .cloned()
                                    .collect::<Vec<_>>(),
                            )?,
                        ))
                    })
                    .collect()
            };
            for k in 2..=8 {
                let subset: Vec<_> = samples
                    .iter()
                    .filter(|s| s.source_set.width == k)
                    .cloned()
                    .collect();
                let aggregate = CStats::new(&subset)?;
                let williams_order_classes = order_stats(&subset)?;
                reconcile_partitions(
                    williams_order_classes.values().map(|s| &s.exact),
                    &aggregate.exact,
                )?;
                if aggregate.exact.totals.samples > 0 {
                    out.interaction_widths
                        .observe(aggregate.exact.totals.interaction_delta_ns)?;
                    out.serial_destination_widths
                        .observe(aggregate.exact.totals.serial_destination_delta_ns)?;
                    out.concurrent_destination_widths
                        .observe(aggregate.exact.totals.concurrent_destination_delta_ns)?;
                }
                out.per_width.insert(
                    k,
                    WidthStats {
                        aggregate,
                        williams_order_classes,
                    },
                );
            }
            out.williams_order_classes = order_stats(samples)?;
            reconcile_partitions(
                out.per_width.values().map(|s| &s.aggregate.exact),
                &out.primary_k2_through_k8.exact,
            )?;
            reconcile_partitions(
                out.williams_order_classes.values().map(|s| &s.exact),
                &out.primary_k2_through_k8.exact,
            )?;
            Ok(out)
        }
        impl CStatistics {
            // Deliberately compare only exact fields. Descriptive f64 values
            // never participate in correctness, authority or interpretation.
            fn exact_matches(&self, other: &Self) -> bool {
                self.primary_k2_through_k8.exact == other.primary_k2_through_k8.exact
                    && self.interaction_widths == other.interaction_widths
                    && self.serial_destination_widths == other.serial_destination_widths
                    && self.concurrent_destination_widths == other.concurrent_destination_widths
                    && self.per_width.keys().eq(other.per_width.keys())
                    && self
                        .williams_order_classes
                        .keys()
                        .eq(other.williams_order_classes.keys())
                    && self.per_width.iter().all(|(k, w)| {
                        let o = &other.per_width[k];
                        w.aggregate.exact == o.aggregate.exact
                            && w.williams_order_classes
                                .keys()
                                .eq(o.williams_order_classes.keys())
                            && w.williams_order_classes
                                .iter()
                                .all(|(c, s)| s.exact == o.williams_order_classes[c].exact)
                    })
                    && self
                        .williams_order_classes
                        .iter()
                        .all(|(c, s)| s.exact == other.williams_order_classes[c].exact)
            }
        }
        fn interaction_interpretation(s: &CStatistics) -> Result<&'static str> {
            let e = &s.primary_k2_through_k8.exact;
            let t = &e.totals;
            if t.samples != 112 || t.hc_ns == 0 {
                return Ok("INCONCLUSIVE");
            }
            let scaled = t
                .interaction_delta_ns
                .checked_mul(100)
                .ok_or_else(|| Failure::accounting("threshold overflow"))?;
            let threshold = |percent: i128| {
                i128::from(t.hc_ns)
                    .checked_mul(percent)
                    .ok_or_else(|| Failure::accounting("threshold product overflow"))
            };
            let median = e
                .median_signed_interaction_delta_ns
                .as_ref()
                .ok_or_else(|| Failure::accounting("missing median"))?
                .numerator;
            let positive = median > 0
                && t.interaction_samples.positive > t.samples / 2
                && s.interaction_widths.positive >= 5
                && s.williams_order_classes
                    .values()
                    .all(|c| c.exact.totals.interaction_delta_ns > 0);
            let negative = median < 0
                && t.interaction_samples.negative > t.samples / 2
                && s.interaction_widths.negative >= 5
                && s.williams_order_classes
                    .values()
                    .all(|c| c.exact.totals.interaction_delta_ns < 0);
            if positive && scaled >= threshold(5)? {
                return Ok("STRONG_AMPLIFICATION");
            }
            if positive && scaled >= threshold(3)? {
                return Ok("MATERIAL_AMPLIFICATION");
            }
            if negative && scaled <= threshold(-3)? {
                return Ok("CONCURRENCY_REDUCES_MAPPED_PENALTY");
            }
            let direction = t.interaction_delta_ns.signum();
            let majority_disagrees = if direction > 0 {
                t.interaction_samples.positive <= t.samples / 2
            } else if direction < 0 {
                t.interaction_samples.negative <= t.samples / 2
            } else {
                t.interaction_samples.positive > t.samples / 2
                    || t.interaction_samples.negative > t.samples / 2
            };
            if median.signum() != direction
                || majority_disagrees
                || (direction > 0
                    && s.williams_order_classes
                        .values()
                        .any(|c| c.exact.totals.interaction_delta_ns < 0))
            {
                return Ok("AMBIGUOUS");
            }
            let ratio = e
                .median_ratio_of_ratios_exact
                .as_ref()
                .ok_or_else(|| Failure::accounting("missing multiplicative median"))?;
            if scaled
                .checked_abs()
                .ok_or_else(|| Failure::accounting("absolute interaction overflow"))?
                <= threshold(1)?
                && ratio.compare_hundredths(99)? != Ordering::Less
                && ratio.compare_hundredths(101)? != Ordering::Greater
                && s.interaction_widths.positive < 5
            {
                return Ok("AGAINST_MATERIAL_INTERACTION");
            }
            Ok("AMBIGUOUS")
        }
        #[derive(Debug, Serialize)]
        struct CellArm {
            #[serde(flatten)]
            evidence: Arm,
            source_sets_attempted: u64,
            serial_helper_calls: u64,
            concurrent_helper_calls: u64,
            cell: Cell,
            completed_source_sets: Vec<SourceSet>,
            source_schedule: ScheduleEvidence,
        }
        impl CellArm {
            fn new(cell: Cell) -> Result<Self> {
                Ok(Self {
                    evidence: Arm::default(),
                    source_sets_attempted: 0,
                    serial_helper_calls: 0,
                    concurrent_helper_calls: 0,
                    cell,
                    completed_source_sets: Vec::new(),
                    source_schedule: schedule_evidence(&[])?,
                })
            }
            fn successful(&self, expected: &ScheduleEvidence, treatment: bool) -> bool {
                let a = &self.evidence;
                let n = expected.expert_slot_count;
                let sets = expected.source_set_count;
                self.source_schedule == *expected
                    && schedule_evidence(&self.completed_source_sets).is_ok_and(|e| e == *expected)
                    && self.source_sets_attempted == sets
                    && self.serial_helper_calls == if self.cell.serial() { sets } else { 0 }
                    && self.concurrent_helper_calls == if self.cell.serial() { 0 } else { sets }
                    && self.cell.mapped() == treatment
                    && a.reconcile(treatment)
                    && a.ops_attempted == n
                    && a.source_read_attempts == n
                    && a.source_read_ops == n
                    && a.full_source_bytes == expected.source_bytes
                    && a.payload_ops == n
                    && a.upload_ops == n
                    && a.gpu_completed_ops == n
                    && a.verified_ops == n
                    && a.fd_evidence.checks == n
                    && a.fd_evidence.direct_observed == n
                    && a.fd_evidence.full_file_length_observed == n
                    && a.fd_evidence.failures == 0
                    && a.pointers.observations == n
                    && a.verification_destination_reset_ops == n
                    && a.source_failures == 0
                    && a.gpu_failures == 0
                    && a.map_failures == 0
                    && a.alignment_failures == 0
                    && a.mapped_direct_io_rejections == 0
                    && a.rejection_errno_counts.is_empty()
                    && a.pointers.gpu_offset_failures == 0
                    && if treatment {
                        a.map_attempts == sets
                            && a.maps_completed == sets
                            && a.unmaps == sets
                            && a.pointers.gpu_offset_checks == n
                    } else {
                        a.map_attempts == 0 && a.maps_completed == 0 && a.unmaps == 0
                    }
            }
        }
        fn begin_set(storage: &NvmeStorage, set: &SourceSet, arm: &mut CellArm) -> Result<()> {
            add(&mut arm.source_sets_attempted, 1)?;
            add(&mut arm.evidence.ops_attempted, set.width as u64)?;
            for &id in &set.ordered_expert_ids {
                fd_evidence(storage, id, &mut arm.evidence)?;
            }
            Ok(())
        }
        fn observe_slices(
            arm: &mut CellArm,
            base: usize,
            offset: usize,
            width: usize,
            treatment: bool,
        ) -> Result<()> {
            for j in 0..width {
                let slot = offset
                    .checked_add(
                        j.checked_mul(FULL)
                            .ok_or_else(|| Failure::accounting("pointer slot overflow"))?,
                    )
                    .ok_or_else(|| Failure::accounting("pointer offset overflow"))?;
                arm.evidence.pointers.observe(base, slot, FULL)?;
                if treatment {
                    add(&mut arm.evidence.pointers.gpu_offset_checks, 1)?;
                }
            }
            Ok(())
        }
        /// Each cell reaches its selected source helper exactly once per set. All caller
        /// setup precedes entry; all accounting and verification follow the timer.
        async fn cell_source(
            storage: &NvmeStorage,
            set: &SourceSet,
            destinations: &mut [&mut [u8]],
            arm: &mut CellArm,
            treatment: bool,
        ) -> Result<u64> {
            add(
                if arm.cell.serial() {
                    &mut arm.serial_helper_calls
                } else {
                    &mut arm.concurrent_helper_calls
                },
                1,
            )?;
            add(&mut arm.evidence.source_read_attempts, set.width as u64)?;
            let ids = set.ordered_expert_ids.as_slice();
            let (read, ns) = if arm.cell.serial() {
                let start = Instant::now();
                let read = storage
                    .read_experts_serial_into_aligned_slices(ids, destinations)
                    .await;
                let ns = elapsed(start)?;
                (read, ns)
            } else {
                let start = Instant::now();
                let read = storage
                    .read_experts_batch_into_aligned_slices(ids, destinations)
                    .await;
                let ns = elapsed(start)?;
                (read, ns)
            };
            add(&mut arm.evidence.times.source_direct_read_ns, ns)?;
            let expected = source_bytes(set.width as u64)?;
            match read {
                Ok(n) if u64::try_from(n).ok() == Some(expected) => {
                    add(&mut arm.evidence.source_read_ops, set.width as u64)?;
                    add(&mut arm.evidence.full_source_bytes, expected)?;
                    arm.completed_source_sets.push(set.clone());
                }
                Ok(n) => {
                    add(&mut arm.evidence.exact_read_length_failures, 1)?;
                    return Err(Failure::accounting(format!(
                        "set {}: batch returned {n} bytes, expected {expected}",
                        set.set_index
                    )));
                }
                Err(e) => {
                    add(&mut arm.evidence.source_failures, 1)?;
                    if e.kind() == io::ErrorKind::UnexpectedEof {
                        add(&mut arm.evidence.exact_read_length_failures, 1)?;
                    }
                    let rejected = treatment && mapped_rejection(e.raw_os_error());
                    if rejected {
                        add(&mut arm.evidence.mapped_direct_io_rejections, 1)?;
                        add(
                            arm.evidence
                                .rejection_errno_counts
                                .entry(e.raw_os_error().unwrap())
                                .or_default(),
                            1,
                        )?;
                    }
                    return Err(Failure::runtime(
                        if rejected {
                            "mapped-upload-direct-io-rejected"
                        } else {
                            "source-failed"
                        },
                        format!(
                            "set {} {:?}: {e}; errno={:?}; no fallback or diagnostic retry",
                            set.set_index,
                            ids,
                            e.raw_os_error()
                        ),
                    ));
                }
            }
            if ns == 0 {
                return Err(Failure::accounting("zero source timer"));
            }
            Ok(ns)
        }
        async fn host_cell(
            gpu: &Gpu,
            storage: &NvmeStorage,
            host: &mut AlignedBuffer,
            set: &SourceSet,
            arm: &mut CellArm,
            streams: &mut Streams,
        ) -> Result<(u64, Vec<Hashes>)> {
            begin_set(storage, set, arm)?;
            let base = host.as_slice().as_ptr() as usize;
            let slices = arena_slices(host.as_mut_slice(), set.width, false);
            if slices.is_err() {
                add(&mut arm.evidence.alignment_failures, 1)?;
            }
            let (offset, mut destinations) = slices?;
            observe_slices(arm, base, offset, set.width, false)?;
            let ns = cell_source(storage, set, &mut destinations, arm, false).await?;
            let mut hashes = Vec::with_capacity(set.width);
            for source in destinations {
                let (_, mut h) = streams.source(source).map_err(Failure::authority)?;
                note_payload(&mut arm.evidence)?;
                prepare_destination(gpu, &mut arm.evidence)?;
                gpu.queue
                    .write_buffer(&gpu.destination, 0, &EPOCH.to_le_bytes());
                let mut view = gpu
                    .queue
                    .write_buffer_with(
                        &gpu.destination,
                        EPOCH_OFFSET as u64,
                        NonZeroU64::new(PAYLOAD as u64).unwrap(),
                    )
                    .ok_or_else(|| {
                        Failure::runtime(
                            "gpu-failed",
                            "CONTROL verification staging view unavailable",
                        )
                    })?;
                view.copy_from_slice(&source[PREFIX..]);
                add(&mut arm.evidence.cpu_payload_copy_bytes, PAYLOAD as u64)?;
                drop(view);
                note_upload(&mut arm.evidence, false)?;
                gpu.drain(None)?;
                add(&mut arm.evidence.gpu_completed_ops, 1)?;
                verify(gpu, &mut arm.evidence, &mut h, streams)?;
                hashes.push(h);
            }
            gpu.check()?;
            Ok((ns, hashes))
        }
        async fn mapped_cell(
            gpu: &Gpu,
            storage: &NvmeStorage,
            set: &SourceSet,
            arm: &mut CellArm,
            streams: &mut Streams,
        ) -> Result<(u64, Vec<Hashes>)> {
            begin_set(storage, set, arm)?;
            add(&mut arm.evidence.map_attempts, 1)?;
            let start = Instant::now();
            let mapped = gpu.map(&gpu.upload, wgpu::MapMode::Write);
            timed(&mut arm.evidence.times.map_wait_ns, start)?;
            if mapped.is_err() {
                add(&mut arm.evidence.map_failures, 1)?;
            }
            mapped?;
            add(&mut arm.evidence.maps_completed, 1)?;
            // Catch only to guarantee unmap after the view future is dropped. A
            // panic is then propagated to the report boundary, never retried.
            let mapped_source = std::panic::AssertUnwindSafe(async {
                let mut view = gpu.upload.slice(..).get_mapped_range_mut();
                let base = view.as_ptr() as usize;
                let capacity = view.len();
                let slices = arena_slices(&mut view, set.width, true);
                if slices.is_err() {
                    add(&mut arm.evidence.alignment_failures, 1)?;
                }
                let (offset, mut destinations) = slices?;
                observe_slices(arm, base, offset, set.width, true)?;
                let ns = cell_source(storage, set, &mut destinations, arm, true).await?;
                let mut payloads = Vec::with_capacity(set.width);
                for (j, source) in destinations.into_iter().enumerate() {
                    let (prefix, h) = streams.source(source).map_err(Failure::authority)?;
                    note_payload(&mut arm.evidence)?;
                    let source_offset = offset
                        .checked_add(
                            j.checked_mul(FULL)
                                .ok_or_else(|| Failure::accounting("copy slot overflow"))?,
                        )
                        .ok_or_else(|| Failure::accounting("copy base overflow"))?;
                    let gpu_offset = copy_offsets(source_offset, prefix, PAYLOAD, capacity)
                        .map_err(Failure::accounting)?;
                    payloads.push((gpu_offset, h));
                }
                Ok::<_, Failure>((ns, payloads))
            })
            .catch_unwind()
            .await;
            let start = Instant::now();
            gpu.upload.unmap();
            timed(&mut arm.evidence.times.treatment_unmap_ns, start)?;
            add(&mut arm.evidence.unmaps, 1)?;
            let source = match mapped_source {
                Ok(result) => result,
                Err(panic) => std::panic::resume_unwind(panic),
            };
            let (ns, payloads) = source?;
            gpu.check()?;
            let mut hashes = Vec::with_capacity(set.width);
            for (gpu_offset, mut h) in payloads {
                prepare_destination(gpu, &mut arm.evidence)?;
                gpu.queue
                    .write_buffer(&gpu.destination, 0, &EPOCH.to_le_bytes());
                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("batch-source-verification-copy"),
                        });
                encoder.copy_buffer_to_buffer(
                    &gpu.upload,
                    gpu_offset,
                    &gpu.destination,
                    EPOCH_OFFSET as u64,
                    PAYLOAD as u64,
                );
                note_upload(&mut arm.evidence, true)?;
                gpu.drain(Some(encoder.finish()))?;
                add(&mut arm.evidence.gpu_completed_ops, 1)?;
                verify(gpu, &mut arm.evidence, &mut h, streams)?;
                hashes.push(h);
            }
            gpu.check()?;
            Ok((ns, hashes))
        }
        fn note_arm_error(arm: &mut CellArm, failure: &Failure) -> Result<()> {
            if failure.classification == "gpu-failed" && arm.evidence.gpu_failures == 0 {
                add(&mut arm.evidence.gpu_failures, 1)?;
            }
            if failure.classification == "accounting-failed"
                && arm.evidence.accounting_failures == 0
            {
                add(&mut arm.evidence.accounting_failures, 1)?;
            }
            Ok(())
        }
        fn c_proof_hits_only(proof: &SourceUploadFdProofSnapshot, slots: u64) -> bool {
            slots.checked_mul(4).is_some_and(|n| {
                proof.source_upload_fd_proof_requests == n
                    && proof.source_upload_fd_proof_hits == n
                    && proof.source_upload_fd_proof_misses == 0
                    && proof.source_upload_fd_proof_failures == 0
            })
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
        struct ExecutedCell {
            set_index: usize,
            ordinal: usize,
            cell: Cell,
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
        struct CellHash {
            source: String,
            payload: String,
            gpu: String,
            epoch: bool,
        }
        impl From<Hashes> for CellHash {
            fn from(h: Hashes) -> Self {
                Self {
                    source: h.source,
                    payload: h.payload,
                    gpu: h.gpu,
                    epoch: h.epoch,
                }
            }
        }
        #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
        struct VerificationSet {
            source_set: CSet,
            cells: BTreeMap<Cell, Vec<CellHash>>,
        }
        impl VerificationSet {
            fn valid(&self) -> bool {
                self.cells.len() == 4
                    && Cell::ALL.into_iter().all(|cell| {
                        self.cells.get(&cell).is_some_and(|hashes| {
                            hashes.len() == self.source_set.width
                                && hashes.iter().enumerate().all(|(j, h)| {
                                    h.epoch
                                        && h.source.len() == 64
                                        && h.payload.len() == 64
                                        && h.payload == h.gpu
                                        && self
                                            .cells
                                            .get(&Cell::HS)
                                            .and_then(|v| v.get(j))
                                            .is_some_and(|host| {
                                                h.source == host.source
                                                    && h.payload == host.payload
                                                    && h.gpu == host.gpu
                                            })
                                })
                        })
                    })
            }
        }
        #[derive(Debug, Serialize)]
        struct CPhase {
            name: &'static str,
            b_schedule_count: usize,
            schedule: Vec<CSet>,
            expected: CPlan,
            execution_trace: Vec<ExecutedCell>,
            raw_samples: Vec<CSample>,
            raw_verification: Vec<VerificationSet>,
            statistics: CStatistics,
            cells: BTreeMap<Cell, CellArm>,
            witnesses: BTreeMap<Cell, Witnesses>,
            fd_proof: SourceUploadFdProofSnapshot,
            fd_proof_hits_only: bool,
            mismatch_count: u64,
        }
        impl CPhase {
            fn new(name: &'static str, b_schedule_count: usize) -> Result<Self> {
                let schedule = c_schedule(b_schedule_count)?;
                Ok(Self {
                    name,
                    b_schedule_count,
                    expected: plan(&schedule)?,
                    schedule,
                    execution_trace: Vec::new(),
                    raw_samples: Vec::new(),
                    raw_verification: Vec::new(),
                    statistics: c_statistics(&[])?,
                    cells: Cell::ALL
                        .into_iter()
                        .map(|c| Ok((c, CellArm::new(c)?)))
                        .collect::<Result<_>>()?,
                    witnesses: Cell::ALL
                        .into_iter()
                        .map(|c| (c, Witnesses::default()))
                        .collect(),
                    fd_proof: SourceUploadFdProofSnapshot::default(),
                    fd_proof_hits_only: false,
                    mismatch_count: 0,
                })
            }
            fn successful(&self) -> bool {
                let Some(host) = self.witnesses.get(&Cell::HS) else {
                    return false;
                };
                let expected_trace: Vec<_> = self
                    .schedule
                    .iter()
                    .flat_map(|s| {
                        s.execution_sequence
                            .iter()
                            .enumerate()
                            .map(|(ordinal, &cell)| ExecutedCell {
                                set_index: s.set_index,
                                ordinal,
                                cell,
                            })
                    })
                    .collect();
                let b_sets: Vec<_> = self.schedule.iter().map(CSet::b_set).collect();
                self.fd_proof_hits_only
                    && c_proof_hits_only(&self.fd_proof, self.expected.source.expert_slot_count)
                    && c_schedule(self.b_schedule_count).is_ok_and(|s| s == self.schedule)
                    && plan(&self.schedule).is_ok_and(|e| e == self.expected)
                    && self.execution_trace == expected_trace
                    && self.cells.len() == 4
                    && self.witnesses.len() == 4
                    && self.raw_samples.len() == self.schedule.len()
                    && self.raw_verification.len() == self.schedule.len()
                    && self
                        .raw_samples
                        .iter()
                        .zip(&self.schedule)
                        .all(|(s, set)| s.source_set == *set && s.valid())
                    && self
                        .raw_verification
                        .iter()
                        .zip(&self.schedule)
                        .all(|(s, set)| s.source_set == *set && s.valid())
                    && c_statistics(&self.raw_samples)
                        .is_ok_and(|s| s.exact_matches(&self.statistics))
                    && self.mismatch_count == 0
                    && Cell::ALL.into_iter().all(|cell| {
                        self.cells.get(&cell).is_some_and(|arm| {
                            arm.cell == cell
                                && arm.completed_source_sets == b_sets
                                && arm.successful(&self.expected.source, cell.mapped())
                                && arm.evidence.times.source_direct_read_ns
                                    == self.statistics.primary_k2_through_k8.exact.totals.times()
                                        [cell.index()]
                        }) && self.witnesses.get(&cell).is_some_and(|w| {
                            w.full_source_sha256 == host.full_source_sha256
                                && w.bare_payload_sha256 == host.bare_payload_sha256
                                && w.bare_payload_sha256 == w.gpu_destination_payload_sha256
                                && w.full_source_sha256.len() == 64
                                && w.bare_payload_sha256.len() == 64
                        })
                    })
            }
        }
        async fn c_phase(
            p: &mut CPhase,
            gpu: &Gpu,
            storage: &NvmeStorage,
            host: &mut AlignedBuffer,
        ) -> Result<()> {
            let mut streams: [Streams; 4] = std::array::from_fn(|_| Streams::default());
            let result=async {
                for set in p.schedule.clone() {
                    let b_set=set.b_set(); let mut times=[0u64;4];
                    let mut verification=VerificationSet{source_set:set.clone(),cells:BTreeMap::new()};
                    for (ordinal,cell) in set.execution_sequence.into_iter().enumerate() {
                        p.execution_trace.push(ExecutedCell{set_index:set.set_index,ordinal,cell});
                        let arm=p.cells.get_mut(&cell).ok_or_else(||Failure::accounting("missing cell"))?;
                        let result=if cell.mapped() {
                            mapped_cell(gpu,storage,&b_set,arm,&mut streams[cell.index()]).await
                        } else {
                            host_cell(gpu,storage,host,&b_set,arm,&mut streams[cell.index()]).await
                        };
                        if let Err(e)=&result {note_arm_error(arm,e)?;}
                        let (ns,hashes)=result?; times[cell.index()]=ns;
                        verification.cells.insert(cell,hashes.into_iter().map(CellHash::from).collect());
                    }
                    p.raw_samples.push(CSample::new(set,times[Cell::HS.index()],times[Cell::MS.index()],times[Cell::HC.index()],times[Cell::MC.index()])?);
                    let valid=verification.valid(); p.raw_verification.push(verification);
                    if !valid { add(&mut p.mismatch_count,1)?; return Err(Failure::runtime("hash-parity-failed","four-cell source/payload/GPU/epoch mismatch; raw verification retained")); }
                }
                Ok(())
            }.await;
            // Both helpers return only after every attempted read has finished.
            p.fd_proof = storage.source_upload_fd_proof_snapshot();
            p.fd_proof_hits_only =
                c_proof_hits_only(&p.fd_proof, p.expected.source.expert_slot_count);
            for cell in Cell::ALL {
                let arm = p
                    .cells
                    .get_mut(&cell)
                    .ok_or_else(|| Failure::accounting("missing cell"))?;
                arm.source_schedule = schedule_evidence(&arm.completed_source_sets)?;
                arm.evidence.rates();
                p.witnesses.insert(cell, streams[cell.index()].snapshot());
            }
            p.statistics = c_statistics(&p.raw_samples)?;
            result?;
            if !p.fd_proof_hits_only {
                return Err(Failure::authority(format!(
                    "{} expected only proof hits: {:?}",
                    p.name, p.fd_proof
                )));
            }
            Ok(())
        }
        #[derive(Debug, Serialize)]
        struct CAuthority {
            // Common hardware/geometry fields retain their existing meaning;
            // control/treatment denote the concurrent HC/MC contrast here.
            #[serde(flatten)]
            base: Authority,
            cell_source_apis: BTreeMap<Cell, &'static str>,
            host_arena_capacity_bytes: usize,
            max_width: usize,
            fd_preproof_completed: bool,
            fd_cache_capacity: usize,
            preproof_universe_size: usize,
            preproof_ordered_expert_ids: Vec<u32>,
            preproof: SourceUploadFdProofSnapshot,
            after_preproof_telemetry_reset: SourceUploadFdProofSnapshot,
            after_warmup_telemetry_reset: SourceUploadFdProofSnapshot,
        }
        #[derive(Debug, Serialize)]
        struct CReport {
            schema: &'static str,
            args: Args,
            config_sha256: Option<String>,
            complete: bool,
            correctness_pass: bool,
            authoritative: bool,
            classification: String,
            failure: Option<String>,
            authority: CAuthority,
            warmup: CPhase,
            measured: CPhase,
            performance_required_for_correctness: bool,
            performance_authority: &'static str,
            interaction_interpretation_conditional_on_zero_retry_log: &'static str,
            primary_endpoint: &'static str,
            schedule_contract: &'static str,
            timing_contract: &'static str,
            interpretation_contract: &'static str,
            secondary_destination_endpoints: &'static str,
            retry_evidence_contract: &'static str,
            source_byte_evidence_contract: &'static str,
        }
        impl CReport {
            fn new(args: Args) -> Result<Self> {
                let mut base = Report::new(args.clone()).authority;
                base.control_source_api = B_API;
                base.treatment_source_api = B_API;
                base.control_destination = "aligned-host-arena";
                base.treatment_destination = "wgpu-map-write-arena";
                base.upload_capacity_bytes = MAPPED_ARENA;
                Ok(Self{schema:C_SCHEMA,args,config_sha256:None,complete:false,correctness_pass:false,authoritative:false,classification:"not-run".into(),failure:None,
                    authority:CAuthority{base,cell_source_apis:Cell::ALL.into_iter().map(|c|(c,c.api())).collect(),host_arena_capacity_bytes:HOST_ARENA,max_width:MAX_WIDTH,
                        fd_preproof_completed:false,fd_cache_capacity:0,preproof_universe_size:UNIVERSE_SIZE,
                        preproof_ordered_expert_ids:expert_sequence(UNIVERSE_SIZE,NAMESPACE).map_err(Failure::accounting)?,preproof:SourceUploadFdProofSnapshot::default(),after_preproof_telemetry_reset:SourceUploadFdProofSnapshot::default(),after_warmup_telemetry_reset:SourceUploadFdProofSnapshot::default()},
                    warmup:CPhase::new("warmup",WARMUP_B_COUNT)?,measured:CPhase::new("measured",MEASURED_B_COUNT)?,performance_required_for_correctness:false,
                    performance_authority:"PENDING_EXTERNAL_RETRY_LOG_AUDIT",interaction_interpretation_conditional_on_zero_retry_log:"INCONCLUSIVE",
                    primary_endpoint:"112 K=2..8 sets: sum((MC-HC)-(MS-HS)); percent denominator=sum(HC)",schedule_contract:C_SCHEDULE_CONTRACT,timing_contract:C_TIMER_CONTRACT,
                    interpretation_contract:C_INTERPRETATION,secondary_destination_endpoints:C_SECONDARY,retry_evidence_contract:C_RETRY,
                    source_byte_evidence_contract:"Each successful helper must return exactly K*FULL bytes. Failed-helper partial physical I/O is unavailable, never inferred as zero. Each cell retains ordered per-set full-source/payload/verified-GPU hashes and concatenated byte-stream witnesses. Verification/reset/copy/readback are outside source timers."})
            }
            fn authority_valid(&self) -> bool {
                let a = &self.authority;
                let b = &a.base;
                self.schema == C_SCHEMA
                    && !self.performance_required_for_correctness
                    && self.args.iterations == MEASURED_B_COUNT
                    && self.args.warmup_iterations == WARMUP_B_COUNT
                    && self.warmup.b_schedule_count == WARMUP_B_COUNT
                    && self.measured.b_schedule_count == MEASURED_B_COUNT
                    && a.cell_source_apis == Cell::ALL.into_iter().map(|c| (c, c.api())).collect()
                    && a.host_arena_capacity_bytes == HOST_ARENA
                    && a.max_width == MAX_WIDTH
                    && b.same_source_api
                    && b.control_source_api == B_API
                    && b.treatment_source_api == B_API
                    && b.control_destination == "aligned-host-arena"
                    && b.treatment_destination == "wgpu-map-write-arena"
                    && b.upload_capacity_bytes == MAPPED_ARENA
                    && b.full_source_bytes == FULL
                    && b.block_alignment == ALIGN
                    && b.uth_prefix_bytes == PREFIX
                    && b.bare_payload_bytes == PAYLOAD
                    && b.physical_slot_bytes == SLOT
                    && b.source_timer_excludes_allocation
                    && b.source_timer_excludes_map_async_device_poll
                    && b.source_timer_excludes_alignment_setup
                    && b.source_timer_excludes_hashes_readback_fd_evidence
                    && b.source_timer_excludes_gpu_copy_unmap
                    && b.linux
                    && b.expected_adapter_name == "NVIDIA L4"
                    && self.args.expected_adapter_name == "NVIDIA L4"
                    && b.adapter_authoritative
                    && b.direct_io_requested
                    && b.packed_storage == Some(false)
                    && b.exact_geometry
                    && a.fd_preproof_completed
                    && a.fd_cache_capacity >= 128
                    && a.preproof_universe_size == 128
                    && expert_sequence(128, NAMESPACE)
                        .is_ok_and(|ids| ids == a.preproof_ordered_expert_ids)
                    && a.preproof
                        == SourceUploadFdProofSnapshot {
                            source_upload_fd_proof_requests: 128,
                            source_upload_fd_proof_misses: 128,
                            ..SourceUploadFdProofSnapshot::default()
                        }
                    && a.after_preproof_telemetry_reset == SourceUploadFdProofSnapshot::default()
                    && a.after_warmup_telemetry_reset == SourceUploadFdProofSnapshot::default()
                    && c_proof_hits_only(&self.warmup.fd_proof, 140)
                    && c_proof_hits_only(&self.measured.fd_proof, 560)
            }
            fn classify(&mut self) -> Result<()> {
                self.complete = true;
                self.correctness_pass = false;
                self.authoritative = false;
                self.interaction_interpretation_conditional_on_zero_retry_log = "INCONCLUSIVE";
                if !self.authority_valid() {
                    self.classification = "authority-failed".into();
                } else if !self.warmup.successful() || !self.measured.successful() {
                    self.classification = "evidence-reconciliation-failed".into();
                } else {
                    self.correctness_pass = true;
                    self.authoritative = true;
                    self.classification = "destination-concurrency-interaction-complete".into();
                    self.interaction_interpretation_conditional_on_zero_retry_log =
                        interaction_interpretation(&self.measured.statistics)?;
                }
                Ok(())
            }
            fn fail(&mut self, failure: Failure) {
                self.complete = failure.complete;
                self.correctness_pass = false;
                self.authoritative = false;
                self.classification = failure.classification.into();
                self.failure = Some(failure.detail);
                self.interaction_interpretation_conditional_on_zero_retry_log = "INCONCLUSIVE";
            }
        }
        async fn execute(report: &mut CReport) -> Result<()> {
            if report.args.iterations != MEASURED_B_COUNT
                || report.args.warmup_iterations != WARMUP_B_COUNT
            {
                return Err(Failure::runtime("invalid-arguments", "C requires --iterations 128 --warmup-iterations 32 as B schedule inputs; actual filtered counts are 112 measured and 28 warmup source sets"));
            }
            let bytes = std::fs::read(&report.args.config)
                .map_err(|e| Failure::runtime("config-failed", e))?;
            report.config_sha256 = Some(sha(&bytes));
            let text =
                std::str::from_utf8(&bytes).map_err(|e| Failure::runtime("config-failed", e))?;
            let config: Config =
                toml::from_str(text).map_err(|e| Failure::runtime("config-failed", e))?;
            config.validate().map_err(Failure::authority)?;
            let a = &mut report.authority;
            a.base.direct_io_requested = !config.storage.no_direct;
            a.base.packed_storage = Some(
                config.storage.packed_blob.is_some() || config.storage.packed_manifest.is_some(),
            );
            a.base.source_data_dir = Some(config.model.data_dir.clone());
            validate_geometry(&config)?;
            a.base.exact_geometry = true;
            if !a.base.linux
                || report.args.expected_adapter_name != "NVIDIA L4"
                || !a.base.direct_io_requested
                || a.base.packed_storage != Some(false)
            {
                return Err(Failure::authority("requires Linux, exact NVIDIA L4 Vulkan, O_DIRECT, unpacked full-file Qwen geometry"));
            }
            let storage = NvmeStorage::new(StorageConfig {
                base_path: config.model.data_dir,
                expert_size: FULL,
                block_align: ALIGN,
                use_direct_io: true,
                num_experts_per_layer: Some(128),
            })
            .map_err(|e| Failure::runtime("source-failed", e))?;
            if storage.is_packed() {
                return Err(Failure::authority("packed storage forbidden"));
            }
            a.fd_cache_capacity = storage.max_open_files();
            if a.fd_cache_capacity < a.preproof_universe_size {
                return Err(Failure::authority(
                    "fd cache cannot retain the complete deterministic universe",
                ));
            }
            let preproof = storage.preprove_source_upload_fds(&a.preproof_ordered_expert_ids);
            a.preproof = storage.source_upload_fd_proof_snapshot();
            preproof.map_err(|e| Failure::authority(format!("fd preproof: {e}")))?;
            if a.preproof.source_upload_fd_proof_requests != a.preproof_universe_size as u64
                || a.preproof.source_upload_fd_proof_misses != a.preproof_universe_size as u64
                || a.preproof.source_upload_fd_proof_hits != 0
                || a.preproof.source_upload_fd_proof_failures != 0
            {
                return Err(Failure::authority(
                    "fresh universe preproof counters do not reconcile",
                ));
            }
            a.fd_preproof_completed = true;
            storage.reset_source_upload_fd_proof_telemetry();
            a.after_preproof_telemetry_reset = storage.source_upload_fd_proof_snapshot();
            if a.after_preproof_telemetry_reset != SourceUploadFdProofSnapshot::default() {
                return Err(Failure::authority("preproof telemetry reset failed"));
            }
            let gpu = Gpu::with_upload_capacity(&mut a.base, MAPPED_ARENA).await?;
            let mut host = AlignedBuffer::new(HOST_ARENA, ALIGN);
            c_phase(&mut report.warmup, &gpu, &storage, &mut host).await?;
            if !report.warmup.successful() {
                return Err(Failure::authority("warmup evidence did not reconcile"));
            }
            storage.reset_source_upload_fd_proof_telemetry();
            report.authority.after_warmup_telemetry_reset =
                storage.source_upload_fd_proof_snapshot();
            if report.authority.after_warmup_telemetry_reset
                != SourceUploadFdProofSnapshot::default()
            {
                return Err(Failure::authority("warmup telemetry reset failed"));
            }
            c_phase(&mut report.measured, &gpu, &storage, &mut host).await?;
            gpu.check()?;
            report.classify()?;
            Ok(())
        }
        pub(crate) async fn run_command(
            args: Args,
        ) -> std::result::Result<(), Box<dyn std::error::Error>> {
            let mut output = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&args.report_out)?;
            let mut report = CReport::new(args).map_err(|e| io::Error::other(e.detail))?;
            match std::panic::AssertUnwindSafe(execute(&mut report))
                .catch_unwind()
                .await
            {
                Ok(Ok(())) => {}
                Ok(Err(e)) => report.fail(e),
                Err(p) => {
                    let detail = p
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "non-string panic".into());
                    report.fail(Failure::runtime(
                        "runtime-failed",
                        format!("C diagnostic panic: {detail}"),
                    ));
                }
            }
            serde_json::to_writer_pretty(&mut output, &report)?;
            output.write_all(b"\n")?;
            output.sync_all()?;
            if report.complete {
                Ok(())
            } else {
                Err(io::Error::other(format!(
                    "{}: {}",
                    report.classification,
                    report.failure.as_deref().unwrap_or("incomplete")
                ))
                .into())
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;
            fn args() -> Args {
                Args {
                    config: "unused.toml".into(),
                    expected_adapter_name: "NVIDIA L4".into(),
                    warmup_iterations: 32,
                    iterations: 128,
                    report_out: "unused.json".into(),
                }
            }
            fn samples(times: [u64; 4]) -> Vec<CSample> {
                c_schedule(128)
                    .unwrap()
                    .into_iter()
                    .map(|s| CSample::new(s, times[0], times[2], times[1], times[3]).unwrap())
                    .collect()
            }
            fn fixture_phase(name: &'static str, count: usize, times: [u64; 4]) -> CPhase {
                let mut p = CPhase::new(name, count).unwrap();
                let n = p.expected.source.expert_slot_count;
                let sets = p.expected.source.source_set_count;
                p.raw_samples = p
                    .schedule
                    .iter()
                    .cloned()
                    .map(|s| CSample::new(s, times[0], times[2], times[1], times[3]).unwrap())
                    .collect();
                p.statistics = c_statistics(&p.raw_samples).unwrap();
                p.execution_trace = p
                    .schedule
                    .iter()
                    .flat_map(|s| {
                        s.execution_sequence
                            .iter()
                            .enumerate()
                            .map(|(ordinal, &cell)| ExecutedCell {
                                set_index: s.set_index,
                                ordinal,
                                cell,
                            })
                    })
                    .collect();
                p.raw_verification = p
                    .schedule
                    .iter()
                    .map(|s| VerificationSet {
                        source_set: s.clone(),
                        cells: Cell::ALL
                            .into_iter()
                            .map(|c| {
                                (
                                    c,
                                    (0..s.width)
                                        .map(|_| CellHash {
                                            source: sha(b"source"),
                                            payload: sha(b"payload"),
                                            gpu: sha(b"payload"),
                                            epoch: true,
                                        })
                                        .collect(),
                                )
                            })
                            .collect(),
                    })
                    .collect();
                for cell in Cell::ALL {
                    let arm = p.cells.get_mut(&cell).unwrap();
                    arm.source_sets_attempted = sets;
                    arm.serial_helper_calls = if cell.serial() { sets } else { 0 };
                    arm.concurrent_helper_calls = if cell.serial() { 0 } else { sets };
                    arm.completed_source_sets = p.schedule.iter().map(CSet::b_set).collect();
                    arm.source_schedule = p.expected.source.clone();
                    let a = &mut arm.evidence;
                    a.ops_attempted = n;
                    a.source_read_attempts = n;
                    a.source_read_ops = n;
                    a.full_source_bytes = n * FULL as u64;
                    a.payload_ops = n;
                    a.payload_bytes = n * PAYLOAD as u64;
                    a.upload_ops = n;
                    a.gpu_copied_bytes = n * PAYLOAD as u64;
                    a.epoch_bytes = n * 4;
                    a.gpu_completed_ops = n;
                    a.verified_ops = n;
                    a.verification_readback_bytes = n * SLOT as u64;
                    a.verification_destination_reset_ops = n;
                    a.verification_destination_reset_bytes = n * SLOT as u64;
                    a.pointers.observations = n;
                    a.pointers.aligned = n;
                    a.fd_evidence = FdEvidence {
                        checks: n,
                        direct_observed: n,
                        full_file_length_observed: n,
                        ..FdEvidence::default()
                    };
                    a.times.source_direct_read_ns = sets * times[cell.index()];
                    if cell.mapped() {
                        a.map_attempts = sets;
                        a.maps_completed = sets;
                        a.unmaps = sets;
                        a.pointers.gpu_offset_checks = n;
                        a.explicit_copy_buffer_bytes = n * PAYLOAD as u64;
                    } else {
                        a.cpu_payload_copy_bytes = n * PAYLOAD as u64;
                    }
                    p.witnesses.insert(
                        cell,
                        Witnesses {
                            full_source_sha256: sha(b"source"),
                            bare_payload_sha256: sha(b"payload"),
                            gpu_destination_payload_sha256: sha(b"payload"),
                        },
                    );
                }
                p.fd_proof = SourceUploadFdProofSnapshot {
                    source_upload_fd_proof_requests: n * 4,
                    source_upload_fd_proof_hits: n * 4,
                    ..SourceUploadFdProofSnapshot::default()
                };
                p.fd_proof_hits_only = true;
                assert!(p.successful());
                p
            }
            fn fixture(times: [u64; 4]) -> CReport {
                let mut r = CReport::new(args()).unwrap();
                let a = &mut r.authority;
                a.base.linux = true;
                a.base.adapter_authoritative = true;
                a.base.direct_io_requested = true;
                a.base.packed_storage = Some(false);
                a.base.exact_geometry = true;
                a.fd_cache_capacity = 128;
                a.fd_preproof_completed = true;
                a.preproof = SourceUploadFdProofSnapshot {
                    source_upload_fd_proof_requests: 128,
                    source_upload_fd_proof_misses: 128,
                    ..SourceUploadFdProofSnapshot::default()
                };
                r.warmup = fixture_phase("warmup", 32, times);
                r.measured = fixture_phase("measured", 128, times);
                r
            }
            #[test]
            fn source_to_upload_copy_elision_c_keeps_b_implementation_and_tests_byte_identical() {
                let whole = include_str!("gpu_native_source_to_upload_copy_elision.rs");
                let b = whole
                    .split_once("mod hma1c_b {")
                    .unwrap()
                    .1
                    .split_once("    /// HMA-1C-C is nested only to reuse B")
                    .unwrap()
                    .0;
                assert_eq!(
                    sha(format!("mod hma1c_b {{{b}").as_bytes()),
                    "f5730a13c2e3788072782f1cddc875be83e956b32c8f7de8bf012a72a6b0a5b3"
                );
            }
            #[test]
            fn source_to_upload_copy_elision_c_b_schedule_and_order_hashes_pinned() {
                let pins = [
                    (
                        32,
                        28,
                        140,
                        "92f6d734fbf5f2060259a9886498ff27abbce48f961997eac8015fd3ab555fd9",
                        "a69c3ddcb54774564b07945b1b685079c3936c2adaee0b97a2a03936f826df3b",
                        "38ae77cc897e75ed247db0b940c42f9a58a897457ca851dd07ac83c7d792adee",
                        "c649854776be748e6eda77f33c2a89ebf491c3524a6e20a89bb9081fd1787b35",
                    ),
                    (
                        128,
                        112,
                        560,
                        "58ceb173ba4bb64fecac985b67dd326e5f11995c8a3dd8a5c55f77850faf522b",
                        "a66e722e5f7375fbb1a3283383b9cb46e2395f10a71a8bad59e5210ecdfdfe36",
                        "51ab91a5058224956c5f3e9080f690abc4ecb3a8228f4463edb262b81ff84049",
                        "d0e2777bcf22bcb6a065cb70a079b2567204543cd73c3a78aaa87a6e3de5de8d",
                    ),
                ];
                for (count, sets, slots, ids, widths, order, complete) in pins {
                    let s = c_schedule(count).unwrap();
                    let e = plan(&s).unwrap();
                    assert_eq!(
                        s.iter().map(CSet::b_set).collect::<Vec<_>>(),
                        schedule(count)
                            .unwrap()
                            .into_iter()
                            .filter(|s| s.width >= 2)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(
                        (
                            e.source.source_set_count,
                            e.source.expert_slot_count,
                            e.source.source_bytes
                        ),
                        (sets, slots, slots * FULL as u64)
                    );
                    assert_eq!(e.source.ordered_source_set_ids_sha256, ids);
                    assert_eq!(e.source.ordered_width_sha256, widths);
                    assert_eq!(e.execution_order_sha256, order);
                    assert_eq!(e.complete_schedule_sha256, complete);
                    assert_eq!(s[0].set_index, 1);
                    assert_eq!(s[0].ordered_expert_ids, vec![822, 1451]);
                }
                assert!(c_schedule(112).is_err());
                assert!(c_schedule(28).is_err());
                let mut bad = c_schedule(128).unwrap();
                bad[0].execution_sequence.swap(0, 1);
                assert!(plan(&bad).is_err());
                let mut bad = c_schedule(128).unwrap();
                bad[0].width_occurrence = 4;
                assert!(plan(&bad).is_err());
                let mut bad = c_schedule(128).unwrap();
                bad[0].williams_order_class = usize::MAX;
                assert!(plan(&bad).is_err());
            }
            #[test]
            fn source_to_upload_copy_elision_c_williams_width_position_predecessor_balance() {
                for (count, repetitions) in [(32, 1usize), (128, 4usize)] {
                    let schedule = c_schedule(count).unwrap();
                    for width in 2..=8 {
                        let sets: Vec<_> = schedule.iter().filter(|s| s.width == width).collect();
                        assert_eq!(sets.len(), 4 * repetitions);
                        let mut classes = [0; 4];
                        let mut positions = [[0; 4]; 4];
                        let mut predecessor = [[0; 4]; 4];
                        for (r, s) in sets.iter().enumerate() {
                            assert_eq!(s.width_occurrence, r);
                            assert_eq!(s.williams_order_class, r % 4);
                            classes[s.williams_order_class] += 1;
                            for (position, cell) in s.execution_sequence.iter().enumerate() {
                                positions[cell.index()][position] += 1;
                            }
                            for pair in s.execution_sequence.windows(2) {
                                predecessor[pair[0].index()][pair[1].index()] += 1;
                            }
                        }
                        assert_eq!(classes, [repetitions; 4]);
                        assert_eq!(positions, [[repetitions; 4]; 4]);
                        for (a, row) in predecessor.iter().enumerate() {
                            for (b, &n) in row.iter().enumerate() {
                                assert_eq!(n, if a == b { 0 } else { repetitions });
                            }
                        }
                    }
                }
            }
            #[test]
            fn source_to_upload_copy_elision_c_exact_deltas_products_medians_and_ratios() {
                let set = c_schedule(128).unwrap().remove(0);
                let s = CSample::new(set.clone(), 100, 103, 50, 60).unwrap();
                assert_eq!(
                    (
                        s.serial_destination_delta_ns,
                        s.concurrent_destination_delta_ns,
                        s.interaction_delta_ns
                    ),
                    (3, 10, 7)
                );
                assert_eq!(
                    (
                        s.mc_times_hs,
                        s.hc_times_ms,
                        s.multiplicative_interaction_direction
                    ),
                    (6000, 5150, 1)
                );
                let negative = CSample::new(set.clone(), 100, 130, 50, 60).unwrap();
                assert_eq!(negative.interaction_delta_ns, -20);
                assert_eq!(negative.multiplicative_interaction_direction, -1);
                // Additive and multiplicative interaction directions can differ.
                let disagreement = CSample::new(set, 100, 110, 1000, 1015).unwrap();
                assert_eq!(disagreement.interaction_delta_ns, 5);
                assert_eq!(disagreement.multiplicative_interaction_direction, -1);
                assert_eq!(
                    median_signed(vec![-3, 2]).unwrap(),
                    Some(SignedFraction {
                        numerator: -1,
                        denominator: 2
                    })
                );
                assert_eq!(
                    median_signed(vec![7, 1, 3]).unwrap(),
                    Some(SignedFraction {
                        numerator: 3,
                        denominator: 1
                    })
                );
                let m = u128::MAX;
                let lo = Ratio {
                    numerator: m - 1,
                    denominator: m,
                };
                let hi = Ratio {
                    numerator: m,
                    denominator: m - 1,
                };
                assert_eq!(lo.descriptive(), hi.descriptive());
                assert_eq!(lo.compare(hi), Ordering::Less);
                let med = RatioMedian::new(vec![(2, hi), (1, lo)]).unwrap();
                assert_eq!(med.lower, lo);
                assert_eq!(med.upper, hi);
                assert_eq!(med.compare_hundredths(99).unwrap(), Ordering::Greater);
                assert_eq!(med.compare_hundredths(101).unwrap(), Ordering::Less);
                for n in 1..30u128 {
                    for d in 1..30u128 {
                        for c in 1..10u128 {
                            let a = Ratio {
                                numerator: n,
                                denominator: d,
                            };
                            let b = Ratio {
                                numerator: c,
                                denominator: 7,
                            };
                            assert_eq!(a.compare(b), (n * 7).cmp(&(c * d)));
                            let med = RatioMedian { lower: a, upper: b };
                            for threshold in [99, 100, 101] {
                                assert_eq!(
                                    med.compare_hundredths(threshold).unwrap(),
                                    ((n * 7 + c * d) * 100).cmp(&(d * 7 * 2 * threshold as u128))
                                );
                            }
                        }
                    }
                }
                let same = Ratio {
                    numerator: m,
                    denominator: m,
                };
                assert_eq!(
                    RatioMedian {
                        lower: same,
                        upper: same
                    }
                    .compare_hundredths(100)
                    .unwrap(),
                    Ordering::Equal
                );
            }
            #[test]
            fn source_to_upload_copy_elision_c_statistics_width_order_and_primary_reconcile() {
                let raw: Vec<_> = c_schedule(128)
                    .unwrap()
                    .into_iter()
                    .map(|s| {
                        let d = (s.width * s.width_occurrence) as u64;
                        CSample::new(s, 1000 + d, 1010 + d, 800 + d, 850 + d).unwrap()
                    })
                    .collect();
                let s = c_statistics(&raw).unwrap();
                let p = &s.primary_k2_through_k8.exact;
                assert_eq!(p.totals.samples, 112);
                assert_eq!(p.totals.interaction_delta_ns, 40 * 112);
                assert_eq!(
                    s.interaction_widths,
                    Directions {
                        positive: 7,
                        negative: 0,
                        equal: 0
                    }
                );
                assert_eq!(
                    p.median_signed_interaction_delta_ns,
                    Some(SignedFraction {
                        numerator: 80,
                        denominator: 2
                    })
                );
                assert_eq!(
                    p.mean_signed_interaction_delta_ns,
                    Some(SignedFraction {
                        numerator: 4480,
                        denominator: 112
                    })
                );
                for w in s.per_width.values() {
                    assert_eq!(w.aggregate.exact.totals.samples, 16);
                    for c in w.williams_order_classes.values() {
                        assert_eq!(c.exact.totals.samples, 4);
                    }
                }
                for c in s.williams_order_classes.values() {
                    assert_eq!(c.exact.totals.samples, 28);
                }
                reconcile_partitions(s.per_width.values().map(|w| &w.aggregate.exact), p).unwrap();
                reconcile_partitions(s.williams_order_classes.values().map(|c| &c.exact), p)
                    .unwrap();
                let mut reversed = raw;
                reversed.reverse();
                assert!(s.exact_matches(&c_statistics(&reversed).unwrap()));
                let mut bad = p.clone();
                bad.totals.interaction_delta_ns += 1;
                assert!(reconcile_partitions(
                    s.per_width.values().map(|w| &w.aggregate.exact),
                    &bad
                )
                .is_err());
                let d = &s.primary_k2_through_k8.descriptive;
                assert_eq!(
                    d.interaction_percent_of_hc,
                    Some(4480.0 / p.totals.hc_ns as f64 * 100.0)
                );
                assert_eq!(
                    d.slowdown_percentage_point_difference,
                    Some(
                        d.concurrent_mapped_slowdown_percent.unwrap()
                            - d.serial_mapped_slowdown_percent.unwrap()
                    )
                );
                assert_eq!(p.serial_median_delta_ns.as_ref().unwrap().numerator, 20);
                assert_eq!(
                    p.concurrent_median_delta_ns.as_ref().unwrap().numerator,
                    100
                );
            }
            #[test]
            fn source_to_upload_copy_elision_c_frozen_exact_interpretation_thresholds() {
                for (mc, expected) in [
                    (1050, "STRONG_AMPLIFICATION"),
                    (1049, "MATERIAL_AMPLIFICATION"),
                    (1030, "MATERIAL_AMPLIFICATION"),
                    (1029, "AMBIGUOUS"),
                    (1011, "AMBIGUOUS"),
                    (1000, "AGAINST_MATERIAL_INTERACTION"),
                    (990, "AGAINST_MATERIAL_INTERACTION"),
                    (970, "CONCURRENCY_REDUCES_MAPPED_PENALTY"),
                ] {
                    assert_eq!(
                        interaction_interpretation(
                            &c_statistics(&samples([1000, 1000, 1000, mc])).unwrap()
                        )
                        .unwrap(),
                        expected
                    );
                }
                // Exact sub-three threshold far below f64 precision at u64 scale.
                let h = 1_000_000_000_000_000u64;
                assert_eq!(
                    interaction_interpretation(
                        &c_statistics(&samples([h, h, h, h + h / 100 * 3 - 1])).unwrap()
                    )
                    .unwrap(),
                    "AMBIGUOUS"
                );
                let base = samples([1000, 1000, 1000, 1100]);
                for mode in 0..3 {
                    let raw: Vec<_> = base
                        .iter()
                        .map(|s| {
                            let reverse = match mode {
                                0 => s.source_set.williams_order_class == 0,
                                1 => s.source_set.width < 5,
                                _ => s.source_set.set_index < 70,
                            };
                            CSample::new(
                                s.source_set.clone(),
                                1000,
                                1000,
                                1000,
                                if reverse { 999 } else { 1300 },
                            )
                            .unwrap()
                        })
                        .collect();
                    assert_eq!(
                        interaction_interpretation(&c_statistics(&raw).unwrap()).unwrap(),
                        "AMBIGUOUS"
                    );
                }
                let mixed: Vec<_> = c_schedule(128)
                    .unwrap()
                    .into_iter()
                    .map(|s| {
                        let mc = if s.width <= 4 { 1003 } else { 998 };
                        CSample::new(s, 1000, 1000, 1000, mc).unwrap()
                    })
                    .collect();
                assert_eq!(
                    interaction_interpretation(&c_statistics(&mixed).unwrap()).unwrap(),
                    "AMBIGUOUS"
                );
                // A raw destination main effect with no interaction is not a win.
                assert_eq!(
                    interaction_interpretation(
                        &c_statistics(&samples([1000, 1000, 1100, 1100])).unwrap()
                    )
                    .unwrap(),
                    "AGAINST_MATERIAL_INTERACTION"
                );
            }
            #[test]
            fn source_to_upload_copy_elision_c_schema_and_correctness_independent_of_performance() {
                for times in [
                    [1000, 1000, 1000, 1],
                    [1000, 1000, 1000, 970],
                    [1000; 4],
                    [1000, 1000, 1000, 1029],
                    [1000, 1000, 1000, 1050],
                    [1000, 1000, 1100, 1100],
                ] {
                    let mut r = fixture(times);
                    r.classify().unwrap();
                    assert!(r.complete && r.correctness_pass && r.authoritative);
                    let j = serde_json::to_value(&r).unwrap();
                    assert_eq!(
                        j["schema"],
                        "mer.gpu-native-mapped-memory-odirect-concurrency-interaction.v1"
                    );
                    assert_ne!(C_SCHEMA, B_SCHEMA);
                    assert_ne!(C_SCHEMA, SCHEMA);
                    assert_eq!(j["performance_required_for_correctness"], false);
                    assert_eq!(
                        j["performance_authority"],
                        "PENDING_EXTERNAL_RETRY_LOG_AUDIT"
                    );
                    for field in [
                        "set_index",
                        "round",
                        "width",
                        "ordered_expert_ids",
                        "width_occurrence",
                        "williams_order_class",
                        "execution_sequence",
                        "hs_ns",
                        "ms_ns",
                        "hc_ns",
                        "mc_ns",
                        "serial_destination_delta_ns",
                        "concurrent_destination_delta_ns",
                        "interaction_delta_ns",
                        "mc_times_hs",
                        "hc_times_ms",
                        "multiplicative_interaction_direction",
                    ] {
                        assert!(
                            j["measured"]["raw_samples"][0].get(field).is_some(),
                            "{field}"
                        );
                    }
                    for cell in Cell::ALL {
                        assert_eq!(r.authority.cell_source_apis[&cell], cell.api());
                    }
                    assert!(r
                        .retry_evidence_contract
                        .contains("RETRY_CONTAMINATED / INCONCLUSIVE"));
                    // Deliberately corrupt every descriptive value; exact evidence
                    // alone determines correctness and the frozen interpretation.
                    let before = r.interaction_interpretation_conditional_on_zero_retry_log;
                    r.measured
                        .statistics
                        .primary_k2_through_k8
                        .descriptive
                        .interaction_percent_of_hc = Some(-999.0);
                    r.classify().unwrap();
                    assert!(r.correctness_pass);
                    assert_eq!(
                        r.interaction_interpretation_conditional_on_zero_retry_log,
                        before
                    );
                }
            }
            #[test]
            fn source_to_upload_copy_elision_c_four_cell_parity_and_fail_closed_counters() {
                let mut p = fixture_phase("measured", 128, [1000; 4]);
                let reject: Vec<Box<dyn Fn(&mut CPhase)>> = vec![
                    Box::new(|p| p.cells.get_mut(&Cell::HS).unwrap().serial_helper_calls -= 1),
                    Box::new(|p| p.cells.get_mut(&Cell::HC).unwrap().serial_helper_calls = 1),
                    Box::new(|p| p.cells.get_mut(&Cell::MS).unwrap().evidence.unmaps -= 1),
                    Box::new(|p| p.cells.get_mut(&Cell::MC).unwrap().evidence.maps_completed -= 1),
                    Box::new(|p| p.cells.get_mut(&Cell::MC).unwrap().evidence.source_failures = 1),
                    Box::new(|p| p.cells.get_mut(&Cell::MS).unwrap().evidence.gpu_failures = 1),
                    Box::new(|p| p.cells.get_mut(&Cell::HS).unwrap().evidence.fallback_reads = 1),
                    Box::new(|p| {
                        p.cells
                            .get_mut(&Cell::MC)
                            .unwrap()
                            .evidence
                            .mapped_direct_io_rejections = 1
                    }),
                    Box::new(|p| {
                        p.cells
                            .get_mut(&Cell::MS)
                            .unwrap()
                            .evidence
                            .pointers
                            .invalid = 1
                    }),
                    Box::new(|p| {
                        p.cells
                            .get_mut(&Cell::MC)
                            .unwrap()
                            .evidence
                            .exact_read_length_failures = 1
                    }),
                    Box::new(|p| {
                        p.cells
                            .get_mut(&Cell::HC)
                            .unwrap()
                            .evidence
                            .full_source_bytes -= 1
                    }),
                    Box::new(|p| {
                        p.cells.get_mut(&Cell::MS).unwrap().completed_source_sets[0]
                            .ordered_expert_ids
                            .reverse()
                    }),
                    Box::new(|p| {
                        p.execution_trace.swap(0, 1);
                    }),
                    Box::new(|p| {
                        p.raw_verification[0].cells.get_mut(&Cell::MC).unwrap()[0].gpu =
                            sha(b"wrong")
                    }),
                    Box::new(|p| {
                        p.raw_verification[0].cells.get_mut(&Cell::MS).unwrap()[0].source =
                            sha(b"wrong")
                    }),
                    Box::new(|p| {
                        p.raw_verification[0].cells.get_mut(&Cell::HC).unwrap()[0].epoch = false
                    }),
                    Box::new(|p| {
                        p.witnesses.get_mut(&Cell::HS).unwrap().bare_payload_sha256 = sha(b"wrong")
                    }),
                    Box::new(|p| p.raw_samples[0].interaction_delta_ns += 1),
                    Box::new(|p| {
                        p.statistics
                            .per_width
                            .get_mut(&2)
                            .unwrap()
                            .aggregate
                            .exact
                            .totals
                            .hs_ns += 1
                    }),
                    Box::new(|p| {
                        p.statistics
                            .williams_order_classes
                            .get_mut(&0)
                            .unwrap()
                            .exact
                            .totals
                            .hs_ns += 1
                    }),
                    Box::new(|p| p.mismatch_count = 1),
                ];
                for (i, f) in reject.into_iter().enumerate() {
                    f(&mut p);
                    assert!(!p.successful(), "mutation {i}");
                    p = fixture_phase("measured", 128, [1000; 4]);
                }
                let mut v = p.raw_verification.remove(0);
                assert!(v.valid());
                v.cells.remove(&Cell::HS);
                assert!(!v.valid());
            }
            #[test]
            fn source_to_upload_copy_elision_c_exact_preproof_and_phase_authority() {
                let r = fixture([1000; 4]);
                assert!(r.authority_valid());
                assert_eq!(r.warmup.fd_proof.source_upload_fd_proof_hits, 560);
                assert_eq!(r.measured.fd_proof.source_upload_fd_proof_hits, 2240);
                for mode in 0..10 {
                    let mut r = fixture([1000; 4]);
                    match mode {
                        0 => r.authority.fd_cache_capacity = 127,
                        1 => r.authority.preproof.source_upload_fd_proof_hits = 1,
                        2 => r.authority.preproof.source_upload_fd_proof_misses -= 1,
                        3 => {
                            r.authority
                                .after_warmup_telemetry_reset
                                .source_upload_fd_proof_hits = 1
                        }
                        4 => r.authority.preproof_ordered_expert_ids.reverse(),
                        5 => r.warmup.fd_proof.source_upload_fd_proof_misses = 1,
                        6 => r.measured.fd_proof.source_upload_fd_proof_failures = 1,
                        7 => r.measured.fd_proof.source_upload_fd_proof_hits -= 1,
                        8 => r.measured.fd_proof.source_upload_fd_proof_requests += 1,
                        _ => r.args.iterations = 112,
                    }
                    r.classify().unwrap();
                    assert!(!r.authoritative && !r.correctness_pass, "{mode}");
                }
                assert!(!c_proof_hits_only(
                    &SourceUploadFdProofSnapshot::default(),
                    u64::MAX
                ));
            }
            #[test]
            fn source_to_upload_copy_elision_c_overflow_and_exact_large_json() {
                assert!(signed_add(i128::MAX, 1).is_err());
                assert!(signed_sub(i128::MIN, 1).is_err());
                assert!(product(u128::MAX, 2).is_err());
                assert!(median_signed(vec![i128::MAX, i128::MAX]).is_err());
                assert!(wide_add([u64::MAX; 5], [1; 5]).is_err());
                let set = c_schedule(128).unwrap().remove(0);
                for times in [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]] {
                    assert!(
                        CSample::new(set.clone(), times[0], times[1], times[2], times[3]).is_err()
                    );
                }
                let s = CSample::new(set, u64::MAX, u64::MAX, u64::MAX, u64::MAX).unwrap();
                assert_eq!(s.mc_times_hs, u128::from(u64::MAX) * u128::from(u64::MAX));
                let json = serde_json::to_string(&s).unwrap();
                assert!(json.contains(&s.mc_times_hs.to_string()));
                assert!(exact_stats(&[s.clone(), s]).is_err());
                let mut s = samples([1000; 4]);
                s[0].hc_times_ms += 1;
                assert!(exact_stats(&s).is_err());
                assert!(Directions {
                    positive: u64::MAX,
                    negative: 1,
                    equal: 0
                }
                .count()
                .is_err());
            }
            #[test]
            fn source_to_upload_copy_elision_c_call_sites_and_source_only_timer() {
                let whole = include_str!("gpu_native_source_to_upload_copy_elision.rs");
                let c = whole
                    .split("pub(super) mod hma1c_c {")
                    .nth(1)
                    .unwrap()
                    .split("#[cfg(test)]")
                    .next()
                    .unwrap();
                let timed = c
                    .split("async fn cell_source(")
                    .nth(1)
                    .unwrap()
                    .split("async fn host_cell(")
                    .next()
                    .unwrap();
                assert_eq!(timed.matches("let start = Instant::now();").count(), 2);
                let intervals: Vec<String> = timed
                    .split("let start = Instant::now();")
                    .skip(1)
                    .map(|s| {
                        s.split("let ns = elapsed(start)?;")
                            .next()
                            .unwrap()
                            .split_whitespace()
                            .collect()
                    })
                    .collect();
                assert_eq!(intervals,vec!["letread=storage.read_experts_serial_into_aligned_slices(ids,destinations).await;","letread=storage.read_experts_batch_into_aligned_slices(ids,destinations).await;"]);
                assert_eq!(
                    c.matches(".read_experts_serial_into_aligned_slices(")
                        .count(),
                    1
                );
                assert_eq!(
                    c.matches(".read_experts_batch_into_aligned_slices(")
                        .count(),
                    1
                );
                assert!(!c.contains(".read_expert_into_aligned_slice("));
                let host = c
                    .split("async fn host_cell(")
                    .nth(1)
                    .unwrap()
                    .split("async fn mapped_cell(")
                    .next()
                    .unwrap();
                let mapped = c
                    .split("async fn mapped_cell(")
                    .nth(1)
                    .unwrap()
                    .split("fn note_arm_error(")
                    .next()
                    .unwrap();
                for body in [host, mapped] {
                    assert_eq!(body.matches("cell_source(").count(), 1);
                    for setup in ["begin_set(", "arena_slices(", "observe_slices("] {
                        assert!(body.find(setup).unwrap() < body.find("cell_source(").unwrap());
                    }
                    for after in ["streams.source(", "verify(", "prepare_destination("] {
                        assert!(body.find(after).unwrap() > body.find("cell_source(").unwrap());
                    }
                }
                for setup in ["gpu.map(", "get_mapped_range_mut()"] {
                    assert!(mapped.find(setup).unwrap() < mapped.find("cell_source(").unwrap());
                }
                assert!(
                    mapped.find("gpu.upload.unmap()").unwrap()
                        > mapped.find("cell_source(").unwrap()
                );
                let phase = c
                    .split("async fn c_phase(")
                    .nth(1)
                    .unwrap()
                    .split("struct CAuthority")
                    .next()
                    .unwrap();
                assert!(phase.contains("set.execution_sequence.into_iter().enumerate()"));
                assert!(!phase.contains("% 2"));
                assert_eq!(phase.matches("mapped_cell(").count(), 1);
                assert_eq!(phase.matches("host_cell(").count(), 1);
                let exec = c.split("async fn execute(").nth(1).unwrap();
                assert_eq!(exec.matches("AlignedBuffer::new(").count(), 1);
                assert_eq!(exec.matches("Gpu::with_upload_capacity(").count(), 1);
                assert_eq!(
                    exec.matches("reset_source_upload_fd_proof_telemetry()")
                        .count(),
                    2
                );
                assert!(
                    exec.find("preprove_source_upload_fds(").unwrap()
                        < exec.find("c_phase(&mut report.warmup").unwrap()
                );
                assert!(whole.contains("hma1c_b::hma1c_c::hma1c_d::run_command(args).await"));
            }
            #[test]
            fn source_to_upload_copy_elision_c_exact_arena_geometry_and_remap_alignment() {
                assert_eq!(
                    (FULL, ALIGN, MAX_WIDTH, HOST_ARENA, MAPPED_ARENA),
                    (2_658_304, 4096, 8, 21_266_432, 21_270_528)
                );
                let mut host = AlignedBuffer::new(HOST_ARENA, ALIGN);
                let host_base = host.as_slice().as_ptr() as usize;
                let mut mapped = AlignedBuffer::new(MAPPED_ARENA + ALIGN, ALIGN);
                for k in 2..=8 {
                    let (_, slices) = arena_slices(host.as_mut_slice(), k, false).unwrap();
                    assert_eq!(slices.len(), k);
                    for (j, s) in slices.into_iter().enumerate() {
                        assert_eq!(s.len(), FULL);
                        assert_eq!(s.as_ptr() as usize, host_base + j * FULL);
                    }
                    for residue in (0..ALIGN).step_by(4) {
                        let base = ALIGN * 16 + residue;
                        let offset = arena_offset(base, MAPPED_ARENA, k, true).unwrap();
                        assert_eq!((base + offset) % ALIGN, 0);
                    }
                    for start in [0, 4, 8, 4092] {
                        let base = mapped.as_slice().as_ptr() as usize + start;
                        let (offset, slices) = arena_slices(
                            &mut mapped.as_mut_slice()[start..start + MAPPED_ARENA],
                            k,
                            true,
                        )
                        .unwrap();
                        assert_eq!((base + offset) % ALIGN, 0);
                        assert_eq!(slices.len(), k);
                        for (j, s) in slices.into_iter().enumerate() {
                            assert_eq!(s.len(), FULL);
                            assert_eq!(s.as_ptr() as usize, base + offset + j * FULL);
                        }
                    }
                }
                assert!(arena_offset(usize::MAX, MAPPED_ARENA, 8, true).is_err());
                let gpu = include_str!("gpu_native_source_to_upload_copy_elision.rs")
                    .split("async fn with_upload_capacity(")
                    .nth(1)
                    .unwrap()
                    .split("fn check(")
                    .next()
                    .unwrap();
                assert!(
                    gpu.contains("wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC")
                );
            }
            #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
            async fn source_to_upload_copy_elision_c_create_new_and_invalid_args_without_gpu() {
                let dir = std::env::temp_dir().join(format!(
                    "mer-hma1cc-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ));
                std::fs::create_dir_all(&dir).unwrap();
                let mut a = args();
                a.config = dir.join("missing.toml");
                a.report_out = dir.join("report.json");
                assert!(run_command(a.clone()).await.is_err());
                let bytes = std::fs::read(&a.report_out).unwrap();
                let j: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(j["schema"], C_SCHEMA);
                assert_eq!(j["classification"], "config-failed");
                assert_eq!(j["measured"]["expected"]["expert_slot_count"], 560);
                assert_eq!(j["warmup"]["expected"]["expert_slot_count"], 140);
                assert!(run_command(a.clone()).await.is_err());
                assert_eq!(std::fs::read(&a.report_out).unwrap(), bytes);
                a.iterations = 112;
                a.report_out = dir.join("invalid.json");
                assert!(run_command(a.clone()).await.is_err());
                let j: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&a.report_out).unwrap()).unwrap();
                assert_eq!(j["classification"], "invalid-arguments");
                std::fs::remove_dir_all(dir).unwrap();
            }
        }

        /// D owns its split-pair design and interpretation; C's cell runners,
        /// source timer, GPU mechanics and exact rational primitives are reused.
        pub(crate) mod hma1c_d {
            use super::*;
            const D_SCHEMA: &str = "mer.gpu-native-mapped-memory-odirect-split-pair-interaction.v1";
            const MEASURED_COUNT: usize = 112;
            const WARMUP_COUNT: usize = 28;
            const D_UNIVERSE: usize = 256;
            const D_SCHEDULE_CONTRACT: &str = "Universe[i]=floor(i*6143/255), i=0..255. Measured r=0..15, warmup r=16..19; K=2..8 in ascending order within each round. w=K-2; block_index=r*7+w; class=(r+5*w)%16. Bits 0/1 reverse serial/concurrent destination order; bit2 concurrent pair first; bit3 B serial/A concurrent. start_A=(block_index*17)%256; start_B=(start_A+128)%256; A/B[j]=universe[(start_A/B+13*j)%256]. Each disjoint set is read exactly twice by its assigned pair. CLI retains iterations=128/warmup_iterations=32; actual D blocks=112/28. Hash encoding: metadata [block_index,r,K,class] u64 LE, then four decoded bit values u8; IDs encode K u32 LE followed by A then B IDs u32 LE; order adds serial role, concurrent role (A=0 B=1), then four cell codes (HS=0 HC=1 MS=2 MC=3). Complete hashes metadata+IDs+roles+cells per block. Temporal quartile=measured block_index/28.";
            const D_INTERPRETATION: &str = "Primary=sum((MC-HC)-(MS-HS)); percent denominator=sum(HC). STRONG >=5%, MATERIAL >=3%, both with exact median>0, >56 positive blocks, >=5/7 positive widths, both levels of all four binary marginal factors positive and all four temporal quartiles positive. <=-3% with analogous negative consistency is NEGATIVE_MATERIAL. AGAINST: abs aggregate<=1%, exact median ratio-of-ratios in [0.99,1.01], fewer than 5 positive widths; disagreement/reversal gates still apply. All other outcomes AMBIGUOUS, including any aggregate/median/majority disagreement, binary marginal reversal, temporal reversal, or >1% and <3%. All 16 individual design classes are descriptive only. No float or rounded percentage participates in classification.";
            const D_SECONDARY: &str = "Serial mapped slowdown, concurrent mapped slowdown and their percentage-point difference are secondary. Retain exact raw deltas/products, exact median interaction and ratio-of-ratios, sign counts, widths, binary marginal factors, temporal quartiles and all 16 descriptive classes.";
            const D_RETRY: &str = "Performance interpretation is conditional on an external audit of the complete FIRST log finding zero exact occurrences of the retry marker defined by the unchanged read_at_with_retries helper. Any occurrence or any measured fd-proof miss/failure makes performance non-authoritative and AMBIGUOUS. No diagnostic retry/fallback. This report does not claim to have audited its external log.";
            #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
            enum SourceRole {
                A,
                B,
            }
            #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
            struct DSet {
                set_index: usize,
                round: usize,
                width: usize,
                design_class: usize,
                serial_order_reversed: bool,
                concurrent_order_reversed: bool,
                concurrent_pair_first: bool,
                b_serial_a_concurrent: bool,
                a_ordered_expert_ids: Vec<u32>,
                b_ordered_expert_ids: Vec<u32>,
                serial_source_set: SourceRole,
                concurrent_source_set: SourceRole,
                execution_sequence: [Cell; 4],
            }
            impl DSet {
                fn role(&self, cell: Cell) -> SourceRole {
                    if cell.serial() {
                        self.serial_source_set
                    } else {
                        self.concurrent_source_set
                    }
                }
                fn b_set(&self, cell: Cell) -> SourceSet {
                    SourceSet {
                        set_index: self.set_index,
                        round: self.round,
                        width: self.width,
                        ordered_expert_ids: match self.role(cell) {
                            SourceRole::A => self.a_ordered_expert_ids.clone(),
                            SourceRole::B => self.b_ordered_expert_ids.clone(),
                        },
                        execution_order: if if cell.serial() {
                            self.serial_order_reversed
                        } else {
                            self.concurrent_order_reversed
                        } {
                            ExecutionOrder::TreatmentFirst
                        } else {
                            ExecutionOrder::ControlFirst
                        },
                    }
                }
                fn factors(&self) -> [bool; 4] {
                    [
                        self.serial_order_reversed,
                        self.concurrent_order_reversed,
                        self.concurrent_pair_first,
                        self.b_serial_a_concurrent,
                    ]
                }
            }
            fn d_set(round: usize, width: usize, universe: &[u32]) -> Result<DSet> {
                if round >= 20
                    || !(2..=8).contains(&width)
                    || universe.len() != D_UNIVERSE
                    || universe.iter().copied().collect::<BTreeSet<_>>().len() != D_UNIVERSE
                {
                    return Err(Failure::accounting("invalid D schedule geometry"));
                }
                let w = width - 2;
                let set_index = round
                    .checked_mul(7)
                    .and_then(|v| v.checked_add(w))
                    .ok_or_else(|| Failure::accounting("D block overflow"))?;
                let design_class = (round + 5 * w) % 16;
                let bits = std::array::from_fn::<_, 4, _>(|i| design_class & (1 << i) != 0);
                let start_a = (set_index * 17) % D_UNIVERSE;
                let start_b = (start_a + 128) % D_UNIVERSE;
                let a: Vec<_> = (0..width)
                    .map(|j| universe[(start_a + 13 * j) % D_UNIVERSE])
                    .collect();
                let b: Vec<_> = (0..width)
                    .map(|j| universe[(start_b + 13 * j) % D_UNIVERSE])
                    .collect();
                if a.iter().chain(&b).copied().collect::<BTreeSet<_>>().len() != 2 * width {
                    return Err(Failure::accounting("D source sets overlap or repeat"));
                }
                let serial = if bits[0] {
                    [Cell::MS, Cell::HS]
                } else {
                    [Cell::HS, Cell::MS]
                };
                let concurrent = if bits[1] {
                    [Cell::MC, Cell::HC]
                } else {
                    [Cell::HC, Cell::MC]
                };
                let (first, last) = if bits[2] {
                    (concurrent, serial)
                } else {
                    (serial, concurrent)
                };
                Ok(DSet {
                    set_index,
                    round,
                    width,
                    design_class,
                    serial_order_reversed: bits[0],
                    concurrent_order_reversed: bits[1],
                    concurrent_pair_first: bits[2],
                    b_serial_a_concurrent: bits[3],
                    a_ordered_expert_ids: a,
                    b_ordered_expert_ids: b,
                    serial_source_set: if bits[3] {
                        SourceRole::B
                    } else {
                        SourceRole::A
                    },
                    concurrent_source_set: if bits[3] {
                        SourceRole::A
                    } else {
                        SourceRole::B
                    },
                    execution_sequence: [first[0], first[1], last[0], last[1]],
                })
            }
            fn d_schedule(count: usize) -> Result<Vec<DSet>> {
                let rounds = match count {
                    0 => 0..0,
                    WARMUP_COUNT => 16..20,
                    MEASURED_COUNT => 0..16,
                    _ => {
                        return Err(Failure::accounting(
                            "D requires 28 warmup or 112 measured blocks",
                        ))
                    }
                };
                let universe =
                    expert_sequence(D_UNIVERSE, NAMESPACE).map_err(Failure::accounting)?;
                rounds
                    .flat_map(|r| (2..=8).map(move |k| (r, k)))
                    .map(|(r, k)| d_set(r, k, &universe))
                    .collect()
            }
            #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
            struct DPlan {
                block_count: u64,
                expert_slots_per_cell: u64,
                source_bytes_per_cell: u64,
                a_source_schedule: ScheduleEvidence,
                b_source_schedule: ScheduleEvidence,
                cell_source_schedules: BTreeMap<Cell, ScheduleEvidence>,
                universe_sha256: String,
                ordered_ab_ids_sha256: String,
                execution_order_sha256: String,
                complete_schedule_sha256: String,
            }
            fn plan(sets: &[DSet]) -> Result<DPlan> {
                let universe =
                    expert_sequence(D_UNIVERSE, NAMESPACE).map_err(Failure::accounting)?;
                if d_schedule(sets.len())? != sets {
                    return Err(Failure::accounting("D schedule differs from frozen plan"));
                }
                let mut ids = Sha256::new();
                let mut order = Sha256::new();
                let mut complete = Sha256::new();
                for s in sets {
                    let mut metadata = Vec::new();
                    for n in [s.set_index, s.round, s.width, s.design_class] {
                        metadata.extend_from_slice(
                            &u64::try_from(n).map_err(Failure::accounting)?.to_le_bytes(),
                        );
                    }
                    metadata.extend(s.factors().map(u8::from));
                    order.update(&metadata);
                    complete.update(&metadata);
                    let mut source = Vec::new();
                    source.extend_from_slice(&(s.width as u32).to_le_bytes());
                    for id in s.a_ordered_expert_ids.iter().chain(&s.b_ordered_expert_ids) {
                        source.extend_from_slice(&id.to_le_bytes());
                    }
                    ids.update(&source);
                    complete.update(&source);
                    let execution = [
                        s.serial_source_set as u8,
                        s.concurrent_source_set as u8,
                        s.execution_sequence[0] as u8,
                        s.execution_sequence[1] as u8,
                        s.execution_sequence[2] as u8,
                        s.execution_sequence[3] as u8,
                    ];
                    order.update(execution);
                    complete.update(execution);
                }
                let raw_sets = |a: bool| {
                    sets.iter()
                        .map(|s| SourceSet {
                            set_index: s.set_index,
                            round: s.round,
                            width: s.width,
                            ordered_expert_ids: if a {
                                s.a_ordered_expert_ids.clone()
                            } else {
                                s.b_ordered_expert_ids.clone()
                            },
                            execution_order: ExecutionOrder::ControlFirst,
                        })
                        .collect::<Vec<_>>()
                };
                let a_source_schedule = schedule_evidence(&raw_sets(true))?;
                Ok(DPlan {
                    block_count: sets.len() as u64,
                    expert_slots_per_cell: a_source_schedule.expert_slot_count,
                    source_bytes_per_cell: a_source_schedule.source_bytes,
                    a_source_schedule,
                    b_source_schedule: schedule_evidence(&raw_sets(false))?,
                    cell_source_schedules: Cell::ALL
                        .into_iter()
                        .map(|c| {
                            Ok((
                                c,
                                schedule_evidence(
                                    &sets.iter().map(|s| s.b_set(c)).collect::<Vec<_>>(),
                                )?,
                            ))
                        })
                        .collect::<Result<_>>()?,
                    universe_sha256: sequence_sha(&universe),
                    ordered_ab_ids_sha256: finish_sha(&ids),
                    execution_order_sha256: finish_sha(&order),
                    complete_schedule_sha256: finish_sha(&complete),
                })
            }
            #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
            struct DSample {
                #[serde(flatten)]
                source_set: DSet,
                hs_ns: u64,
                ms_ns: u64,
                hc_ns: u64,
                mc_ns: u64,
                serial_destination_delta_ns: i128,
                concurrent_destination_delta_ns: i128,
                interaction_delta_ns: i128,
                mc_times_hs: u128,
                hc_times_ms: u128,
                multiplicative_interaction_direction: i8,
            }
            impl DSample {
                fn new(
                    source_set: DSet,
                    hs_ns: u64,
                    ms_ns: u64,
                    hc_ns: u64,
                    mc_ns: u64,
                ) -> Result<Self> {
                    if [hs_ns, ms_ns, hc_ns, mc_ns].contains(&0) {
                        return Err(Failure::accounting("zero C source duration"));
                    }
                    let serial_destination_delta_ns = signed_sub(ms_ns.into(), hs_ns.into())?;
                    let concurrent_destination_delta_ns = signed_sub(mc_ns.into(), hc_ns.into())?;
                    let interaction_delta_ns =
                        signed_sub(concurrent_destination_delta_ns, serial_destination_delta_ns)?;
                    let mc_times_hs = product(mc_ns.into(), hs_ns.into())?;
                    let hc_times_ms = product(hc_ns.into(), ms_ns.into())?;
                    let multiplicative_interaction_direction = match mc_times_hs.cmp(&hc_times_ms) {
                        Ordering::Less => -1,
                        Ordering::Equal => 0,
                        Ordering::Greater => 1,
                    };
                    Ok(Self {
                        source_set,
                        hs_ns,
                        ms_ns,
                        hc_ns,
                        mc_ns,
                        serial_destination_delta_ns,
                        concurrent_destination_delta_ns,
                        interaction_delta_ns,
                        mc_times_hs,
                        hc_times_ms,
                        multiplicative_interaction_direction,
                    })
                }
                fn valid(&self) -> bool {
                    Self::new(
                        self.source_set.clone(),
                        self.hs_ns,
                        self.ms_ns,
                        self.hc_ns,
                        self.mc_ns,
                    )
                    .is_ok_and(|s| s == *self)
                }
                fn times(&self) -> [u64; 4] {
                    [self.hs_ns, self.hc_ns, self.ms_ns, self.mc_ns]
                }
                fn ratio(&self) -> Ratio {
                    Ratio {
                        numerator: self.mc_times_hs,
                        denominator: self.hc_times_ms,
                    }
                }
            }
            #[derive(Clone, Default, Debug, PartialEq, Eq, Serialize)]
            struct Totals {
                samples: u64,
                hs_ns: u64,
                ms_ns: u64,
                hc_ns: u64,
                mc_ns: u64,
                serial_destination_delta_ns: i128,
                concurrent_destination_delta_ns: i128,
                interaction_delta_ns: i128,
                interaction_samples: Directions,
                serial_destination_samples: Directions,
                concurrent_destination_samples: Directions,
                multiplicative_interaction_samples: Directions,
            }
            impl Totals {
                fn observe(&mut self, s: &DSample) -> Result<()> {
                    if !s.valid() {
                        return Err(Failure::accounting("C raw arithmetic does not reconcile"));
                    }
                    add(&mut self.samples, 1)?;
                    add(&mut self.hs_ns, s.hs_ns)?;
                    add(&mut self.ms_ns, s.ms_ns)?;
                    add(&mut self.hc_ns, s.hc_ns)?;
                    add(&mut self.mc_ns, s.mc_ns)?;
                    self.serial_destination_delta_ns = signed_add(
                        self.serial_destination_delta_ns,
                        s.serial_destination_delta_ns,
                    )?;
                    self.concurrent_destination_delta_ns = signed_add(
                        self.concurrent_destination_delta_ns,
                        s.concurrent_destination_delta_ns,
                    )?;
                    self.interaction_delta_ns =
                        signed_add(self.interaction_delta_ns, s.interaction_delta_ns)?;
                    self.interaction_samples.observe(s.interaction_delta_ns)?;
                    self.serial_destination_samples
                        .observe(s.serial_destination_delta_ns)?;
                    self.concurrent_destination_samples
                        .observe(s.concurrent_destination_delta_ns)?;
                    self.multiplicative_interaction_samples
                        .observe(s.multiplicative_interaction_direction.into())?;
                    self.reconcile()
                }
                fn reconcile(&self) -> Result<()> {
                    if self.serial_destination_delta_ns
                        != signed_sub(self.ms_ns.into(), self.hs_ns.into())?
                        || self.concurrent_destination_delta_ns
                            != signed_sub(self.mc_ns.into(), self.hc_ns.into())?
                        || self.interaction_delta_ns
                            != signed_sub(
                                self.concurrent_destination_delta_ns,
                                self.serial_destination_delta_ns,
                            )?
                        || [
                            &self.interaction_samples,
                            &self.serial_destination_samples,
                            &self.concurrent_destination_samples,
                            &self.multiplicative_interaction_samples,
                        ]
                        .into_iter()
                        .any(|d| d.count().ok() != Some(self.samples))
                    {
                        return Err(Failure::accounting("C totals do not reconcile"));
                    }
                    Ok(())
                }
                fn times(&self) -> [u64; 4] {
                    [self.hs_ns, self.hc_ns, self.ms_ns, self.mc_ns]
                }
            }
            #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
            struct ExactStats {
                #[serde(flatten)]
                totals: Totals,
                mean_signed_interaction_delta_ns: Option<SignedFraction>,
                median_signed_interaction_delta_ns: Option<SignedFraction>,
                median_ratio_of_ratios_exact: Option<RatioMedian>,
                serial_median_delta_ns: Option<SignedFraction>,
                concurrent_median_delta_ns: Option<SignedFraction>,
                serial_median_mapped_over_host_ratio_exact: Option<RatioMedian>,
                concurrent_median_mapped_over_host_ratio_exact: Option<RatioMedian>,
            }
            fn exact_stats(samples: &[DSample]) -> Result<ExactStats> {
                let mut totals = Totals::default();
                for s in samples {
                    totals.observe(s)?;
                }
                let median =
                    |f: fn(&DSample) -> i128| median_signed(samples.iter().map(f).collect());
                let ratios = |f: fn(&DSample) -> Ratio| {
                    RatioMedian::new(
                        samples
                            .iter()
                            .map(|s| (s.source_set.set_index, f(s)))
                            .collect(),
                    )
                };
                Ok(ExactStats {
                    mean_signed_interaction_delta_ns: (totals.samples != 0).then(|| {
                        SignedFraction {
                            numerator: totals.interaction_delta_ns,
                            denominator: totals.samples,
                        }
                    }),
                    median_signed_interaction_delta_ns: median(|s| s.interaction_delta_ns)?,
                    median_ratio_of_ratios_exact: ratios(DSample::ratio),
                    serial_median_delta_ns: median(|s| s.serial_destination_delta_ns)?,
                    concurrent_median_delta_ns: median(|s| s.concurrent_destination_delta_ns)?,
                    serial_median_mapped_over_host_ratio_exact: ratios(|s| Ratio {
                        numerator: s.ms_ns.into(),
                        denominator: s.hs_ns.into(),
                    }),
                    concurrent_median_mapped_over_host_ratio_exact: ratios(|s| Ratio {
                        numerator: s.mc_ns.into(),
                        denominator: s.hc_ns.into(),
                    }),
                    totals,
                })
            }
            #[derive(Debug, Serialize)]
            struct DescriptiveStats {
                interaction_percent_of_hc: Option<f64>,
                serial_mapped_slowdown_percent: Option<f64>,
                concurrent_mapped_slowdown_percent: Option<f64>,
                slowdown_percentage_point_difference: Option<f64>,
                mean_signed_interaction_delta_ns: Option<f64>,
                median_signed_interaction_delta_ns: Option<f64>,
                median_ratio_of_ratios: Option<f64>,
                serial_median_delta_ns: Option<f64>,
                concurrent_median_delta_ns: Option<f64>,
                serial_median_mapped_over_host_ratio: Option<f64>,
                concurrent_median_mapped_over_host_ratio: Option<f64>,
            }
            fn descriptive(e: &ExactStats) -> DescriptiveStats {
                let t = &e.totals;
                let pct = |d: i128, n: u64| (n != 0).then(|| d as f64 / n as f64 * 100.0);
                let frac = |f: &Option<SignedFraction>| {
                    f.as_ref()
                        .map(|f| f.numerator as f64 / f.denominator as f64)
                };
                let serial = pct(t.serial_destination_delta_ns, t.hs_ns);
                let concurrent = pct(t.concurrent_destination_delta_ns, t.hc_ns);
                DescriptiveStats {
                    interaction_percent_of_hc: pct(t.interaction_delta_ns, t.hc_ns),
                    serial_mapped_slowdown_percent: serial,
                    concurrent_mapped_slowdown_percent: concurrent,
                    slowdown_percentage_point_difference: concurrent
                        .zip(serial)
                        .map(|(c, s)| c - s),
                    mean_signed_interaction_delta_ns: frac(&e.mean_signed_interaction_delta_ns),
                    median_signed_interaction_delta_ns: frac(&e.median_signed_interaction_delta_ns),
                    median_ratio_of_ratios: e
                        .median_ratio_of_ratios_exact
                        .as_ref()
                        .map(RatioMedian::descriptive),
                    serial_median_delta_ns: frac(&e.serial_median_delta_ns),
                    concurrent_median_delta_ns: frac(&e.concurrent_median_delta_ns),
                    serial_median_mapped_over_host_ratio: e
                        .serial_median_mapped_over_host_ratio_exact
                        .as_ref()
                        .map(RatioMedian::descriptive),
                    concurrent_median_mapped_over_host_ratio: e
                        .concurrent_median_mapped_over_host_ratio_exact
                        .as_ref()
                        .map(RatioMedian::descriptive),
                }
            }
            #[derive(Debug, Serialize)]
            struct DStats {
                exact: ExactStats,
                descriptive: DescriptiveStats,
            }
            impl DStats {
                fn new(samples: &[DSample]) -> Result<Self> {
                    let exact = exact_stats(samples)?;
                    Ok(Self {
                        descriptive: descriptive(&exact),
                        exact,
                    })
                }
            }
            fn reconcile_partitions<'a>(
                parts: impl Iterator<Item = &'a ExactStats>,
                primary: &ExactStats,
            ) -> Result<()> {
                let mut totals = Totals::default();
                for e in parts {
                    let t = &e.totals;
                    add(&mut totals.samples, t.samples)?;
                    add(&mut totals.hs_ns, t.hs_ns)?;
                    add(&mut totals.ms_ns, t.ms_ns)?;
                    add(&mut totals.hc_ns, t.hc_ns)?;
                    add(&mut totals.mc_ns, t.mc_ns)?;
                    totals.serial_destination_delta_ns = signed_add(
                        totals.serial_destination_delta_ns,
                        t.serial_destination_delta_ns,
                    )?;
                    totals.concurrent_destination_delta_ns = signed_add(
                        totals.concurrent_destination_delta_ns,
                        t.concurrent_destination_delta_ns,
                    )?;
                    totals.interaction_delta_ns =
                        signed_add(totals.interaction_delta_ns, t.interaction_delta_ns)?;
                    for (a, b) in [
                        (&mut totals.interaction_samples, &t.interaction_samples),
                        (
                            &mut totals.serial_destination_samples,
                            &t.serial_destination_samples,
                        ),
                        (
                            &mut totals.concurrent_destination_samples,
                            &t.concurrent_destination_samples,
                        ),
                        (
                            &mut totals.multiplicative_interaction_samples,
                            &t.multiplicative_interaction_samples,
                        ),
                    ] {
                        add(&mut a.positive, b.positive)?;
                        add(&mut a.negative, b.negative)?;
                        add(&mut a.equal, b.equal)?;
                    }
                }
                totals.reconcile()?;
                if totals != primary.totals {
                    return Err(Failure::accounting("width/order partition mismatch"));
                }
                Ok(())
            }

            const FACTORS: [&str; 4] = [
                "serial_destination_order",
                "concurrent_destination_order",
                "pair_execution_order",
                "source_set_role",
            ];
            #[derive(Debug, Serialize)]
            struct DStatistics {
                primary_k2_through_k8: DStats,
                per_width: BTreeMap<usize, DStats>,
                binary_marginal_factors: BTreeMap<&'static str, BTreeMap<usize, DStats>>,
                temporal_quartiles: BTreeMap<usize, DStats>,
                descriptive_design_classes: BTreeMap<usize, DStats>,
                interaction_widths: Directions,
            }
            fn d_statistics(samples: &[DSample]) -> Result<DStatistics> {
                // Partial prefixes are retained on failure; only successful full
                // phases may supply classification and exact schedule authority.
                let mut seen = BTreeSet::new();
                let universe =
                    expert_sequence(D_UNIVERSE, NAMESPACE).map_err(Failure::accounting)?;
                for s in samples {
                    let b = &s.source_set;
                    if !seen.insert(b.set_index)
                        || d_set(b.round, b.width, &universe)? != *b
                        || !s.valid()
                    {
                        return Err(Failure::accounting("invalid D raw block"));
                    }
                }
                let partition = |n: usize,
                                 key: &dyn Fn(&DSample) -> usize|
                 -> Result<BTreeMap<usize, DStats>> {
                    (0..n)
                        .map(|i| {
                            Ok((
                                i,
                                DStats::new(
                                    &samples
                                        .iter()
                                        .filter(|s| key(s) == i)
                                        .cloned()
                                        .collect::<Vec<_>>(),
                                )?,
                            ))
                        })
                        .collect()
                };
                let primary_k2_through_k8 = DStats::new(samples)?;
                let per_width = partition(7, &|s| s.source_set.width - 2)?
                    .into_iter()
                    .map(|(i, s)| (i + 2, s))
                    .collect::<BTreeMap<_, _>>();
                let mut binary_marginal_factors = BTreeMap::new();
                for (bit, name) in FACTORS.iter().enumerate() {
                    binary_marginal_factors.insert(
                        *name,
                        partition(2, &|s| usize::from(s.source_set.factors()[bit]))?,
                    );
                }
                let temporal_quartiles = partition(4, &|s| (s.source_set.set_index % 112) / 28)?;
                let descriptive_design_classes = partition(16, &|s| s.source_set.design_class)?;
                for p in [&per_width, &temporal_quartiles, &descriptive_design_classes]
                    .into_iter()
                    .chain(binary_marginal_factors.values())
                {
                    reconcile_partitions(
                        p.values().map(|s| &s.exact),
                        &primary_k2_through_k8.exact,
                    )?;
                }
                let mut interaction_widths = Directions::default();
                for w in per_width.values().filter(|w| w.exact.totals.samples > 0) {
                    interaction_widths.observe(w.exact.totals.interaction_delta_ns)?;
                }
                Ok(DStatistics {
                    primary_k2_through_k8,
                    per_width,
                    binary_marginal_factors,
                    temporal_quartiles,
                    descriptive_design_classes,
                    interaction_widths,
                })
            }
            impl DStatistics {
                fn exact_matches(&self, other: &Self) -> bool {
                    let eq = |a: &BTreeMap<usize, DStats>, b: &BTreeMap<usize, DStats>| {
                        a.keys().eq(b.keys()) && a.iter().all(|(k, s)| s.exact == b[k].exact)
                    };
                    self.primary_k2_through_k8.exact == other.primary_k2_through_k8.exact
                        && self.interaction_widths == other.interaction_widths
                        && eq(&self.per_width, &other.per_width)
                        && eq(&self.temporal_quartiles, &other.temporal_quartiles)
                        && eq(
                            &self.descriptive_design_classes,
                            &other.descriptive_design_classes,
                        )
                        && self
                            .binary_marginal_factors
                            .keys()
                            .eq(other.binary_marginal_factors.keys())
                        && self
                            .binary_marginal_factors
                            .iter()
                            .all(|(k, s)| eq(s, &other.binary_marginal_factors[k]))
                }
            }
            fn interaction_interpretation(s: &DStatistics) -> Result<&'static str> {
                let e = &s.primary_k2_through_k8.exact;
                let t = &e.totals;
                if t.samples != 112
                    || t.hc_ns == 0
                    || s.per_width.len() != 7
                    || s.per_width.values().any(|s| s.exact.totals.samples != 16)
                    || s.binary_marginal_factors.len() != 4
                    || FACTORS.iter().any(|name| {
                        s.binary_marginal_factors.get(name).is_none_or(|p| {
                            p.len() != 2
                                || (0..2)
                                    .any(|i| p.get(&i).is_none_or(|s| s.exact.totals.samples != 56))
                        })
                    })
                    || s.temporal_quartiles.len() != 4
                    || (0..4).any(|i| {
                        s.temporal_quartiles
                            .get(&i)
                            .is_none_or(|s| s.exact.totals.samples != 28)
                    })
                {
                    return Ok("INCONCLUSIVE");
                }
                let scaled = t
                    .interaction_delta_ns
                    .checked_mul(100)
                    .ok_or_else(|| Failure::accounting("D threshold overflow"))?;
                let threshold = |p: i128| {
                    i128::from(t.hc_ns)
                        .checked_mul(p)
                        .ok_or_else(|| Failure::accounting("D threshold product overflow"))
                };
                let median = e
                    .median_signed_interaction_delta_ns
                    .as_ref()
                    .ok_or_else(|| Failure::accounting("D missing median"))?
                    .numerator;
                let strata: Vec<_> = s
                    .binary_marginal_factors
                    .values()
                    .flat_map(|p| p.values())
                    .chain(s.temporal_quartiles.values())
                    .map(|s| s.exact.totals.interaction_delta_ns)
                    .collect();
                let positive = median > 0
                    && t.interaction_samples.positive > 56
                    && s.interaction_widths.positive >= 5
                    && strata.iter().all(|v| *v > 0);
                let negative = median < 0
                    && t.interaction_samples.negative > 56
                    && s.interaction_widths.negative >= 5
                    && strata.iter().all(|v| *v < 0);
                if positive && scaled >= threshold(5)? {
                    return Ok("STRONG_POSITIVE_INTERACTION");
                }
                if positive && scaled >= threshold(3)? {
                    return Ok("MATERIAL_POSITIVE_INTERACTION");
                }
                if negative && scaled <= threshold(-3)? {
                    return Ok("NEGATIVE_MATERIAL_INTERACTION");
                }
                let direction = t.interaction_delta_ns.signum();
                let majority_disagrees = match direction {
                    1 => t.interaction_samples.positive <= 56,
                    -1 => t.interaction_samples.negative <= 56,
                    _ => t.interaction_samples.positive > 56 || t.interaction_samples.negative > 56,
                };
                if median.signum() != direction
                    || majority_disagrees
                    || strata.iter().any(|v| v.signum() == -direction && *v != 0)
                {
                    return Ok("AMBIGUOUS");
                }
                let ratio = e
                    .median_ratio_of_ratios_exact
                    .as_ref()
                    .ok_or_else(|| Failure::accounting("D missing ratio median"))?;
                if scaled
                    .checked_abs()
                    .ok_or_else(|| Failure::accounting("D absolute overflow"))?
                    <= threshold(1)?
                    && ratio.compare_hundredths(99)? != Ordering::Less
                    && ratio.compare_hundredths(101)? != Ordering::Greater
                    && s.interaction_widths.positive < 5
                {
                    return Ok("EVIDENCE_AGAINST_MATERIAL_INTERACTION");
                }
                Ok("AMBIGUOUS")
            }
            #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
            struct ExecutedCell {
                source_role: SourceRole,
                ordered_expert_ids: Vec<u32>,
                set_index: usize,
                ordinal: usize,
                cell: Cell,
            }
            #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
            struct CellHash {
                source: String,
                payload: String,
                gpu: String,
                epoch: bool,
            }
            impl From<Hashes> for CellHash {
                fn from(h: Hashes) -> Self {
                    Self {
                        source: h.source,
                        payload: h.payload,
                        gpu: h.gpu,
                        epoch: h.epoch,
                    }
                }
            }
            #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
            struct VerificationSet {
                source_set: DSet,
                cells: BTreeMap<Cell, Vec<CellHash>>,
            }
            impl VerificationSet {
                fn valid(&self) -> bool {
                    self.cells.len() == 4
                        && Cell::ALL.into_iter().all(|cell| {
                            self.cells.get(&cell).is_some_and(|hashes| {
                                hashes.len() == self.source_set.width
                                    && hashes.iter().enumerate().all(|(j, h)| {
                                        h.epoch
                                            && h.source.len() == 64
                                            && h.payload.len() == 64
                                            && h.payload == h.gpu
                                            && self
                                                .cells
                                                .get(&if cell.serial() {
                                                    Cell::HS
                                                } else {
                                                    Cell::HC
                                                })
                                                .and_then(|v| v.get(j))
                                                .is_some_and(|host| {
                                                    h.source == host.source
                                                        && h.payload == host.payload
                                                        && h.gpu == host.gpu
                                                })
                                    })
                            })
                        })
                }
            }
            #[derive(Debug, Serialize)]
            struct DPhase {
                name: &'static str,
                schedule_count: usize,
                schedule: Vec<DSet>,
                expected: DPlan,
                execution_trace: Vec<ExecutedCell>,
                raw_samples: Vec<DSample>,
                raw_verification: Vec<VerificationSet>,
                statistics: DStatistics,
                cells: BTreeMap<Cell, CellArm>,
                witnesses: BTreeMap<Cell, Witnesses>,
                fd_proof: SourceUploadFdProofSnapshot,
                fd_proof_hits_only: bool,
                mismatch_count: u64,
            }
            impl DPhase {
                fn new(name: &'static str, schedule_count: usize) -> Result<Self> {
                    let schedule = d_schedule(schedule_count)?;
                    Ok(Self {
                        name,
                        schedule_count,
                        expected: plan(&schedule)?,
                        schedule,
                        execution_trace: Vec::new(),
                        raw_samples: Vec::new(),
                        raw_verification: Vec::new(),
                        statistics: d_statistics(&[])?,
                        cells: Cell::ALL
                            .into_iter()
                            .map(|c| Ok((c, CellArm::new(c)?)))
                            .collect::<Result<_>>()?,
                        witnesses: Cell::ALL
                            .into_iter()
                            .map(|c| (c, Witnesses::default()))
                            .collect(),
                        fd_proof: SourceUploadFdProofSnapshot::default(),
                        fd_proof_hits_only: false,
                        mismatch_count: 0,
                    })
                }
                fn successful(&self) -> bool {
                    let expected_trace: Vec<_> = self
                        .schedule
                        .iter()
                        .flat_map(|s| {
                            s.execution_sequence
                                .iter()
                                .enumerate()
                                .map(|(ordinal, &cell)| ExecutedCell {
                                    set_index: s.set_index,
                                    source_role: s.role(cell),
                                    ordered_expert_ids: s.b_set(cell).ordered_expert_ids,
                                    ordinal,
                                    cell,
                                })
                        })
                        .collect();
                    self.fd_proof_hits_only
                        && c_proof_hits_only(&self.fd_proof, self.expected.expert_slots_per_cell)
                        && d_schedule(self.schedule_count).is_ok_and(|s| s == self.schedule)
                        && plan(&self.schedule).is_ok_and(|e| e == self.expected)
                        && self.execution_trace == expected_trace
                        && self.cells.len() == 4
                        && self.witnesses.len() == 4
                        && self.raw_samples.len() == self.schedule.len()
                        && self.raw_verification.len() == self.schedule.len()
                        && self
                            .raw_samples
                            .iter()
                            .zip(&self.schedule)
                            .all(|(s, set)| s.source_set == *set && s.valid())
                        && self
                            .raw_verification
                            .iter()
                            .zip(&self.schedule)
                            .all(|(s, set)| s.source_set == *set && s.valid())
                        && d_statistics(&self.raw_samples)
                            .is_ok_and(|s| s.exact_matches(&self.statistics))
                        && self.mismatch_count == 0
                        && Cell::ALL.into_iter().all(|cell| {
                            self.cells.get(&cell).is_some_and(|arm| {
                                arm.cell == cell
                                    && arm.completed_source_sets
                                        == self
                                            .schedule
                                            .iter()
                                            .map(|s| s.b_set(cell))
                                            .collect::<Vec<_>>()
                                    && arm.successful(
                                        &self.expected.cell_source_schedules[&cell],
                                        cell.mapped(),
                                    )
                                    && arm.evidence.times.source_direct_read_ns
                                        == self
                                            .statistics
                                            .primary_k2_through_k8
                                            .exact
                                            .totals
                                            .times()[cell.index()]
                            }) && self.witnesses.get(&cell).is_some_and(|w| {
                                let Some(host) = self.witnesses.get(&if cell.serial() {
                                    Cell::HS
                                } else {
                                    Cell::HC
                                }) else {
                                    return false;
                                };
                                w.full_source_sha256 == host.full_source_sha256
                                    && w.bare_payload_sha256 == host.bare_payload_sha256
                                    && w.bare_payload_sha256 == w.gpu_destination_payload_sha256
                                    && w.full_source_sha256.len() == 64
                                    && w.bare_payload_sha256.len() == 64
                            })
                        })
                }
            }
            async fn d_phase(
                p: &mut DPhase,
                gpu: &Gpu,
                storage: &NvmeStorage,
                host: &mut AlignedBuffer,
            ) -> Result<()> {
                let mut streams: [Streams; 4] = std::array::from_fn(|_| Streams::default());
                let result = async {
                    for set in p.schedule.clone() {
                        let mut times = [0u64; 4];
                        let mut verification = VerificationSet {
                            source_set: set.clone(),
                            cells: BTreeMap::new(),
                        };
                        for (ordinal, cell) in set.execution_sequence.into_iter().enumerate() {
                            let b_set = set.b_set(cell);
                            p.execution_trace.push(ExecutedCell {
                                set_index: set.set_index,
                                ordinal,
                                cell,
                                source_role: set.role(cell),
                                ordered_expert_ids: b_set.ordered_expert_ids.clone(),
                            });
                            let arm = p.cells.get_mut(&cell)
                                .ok_or_else(|| Failure::accounting("missing cell"))?;
                            let result = if cell.mapped() {
                                mapped_cell(gpu, storage, &b_set, arm, &mut streams[cell.index()]).await
                            } else {
                                host_cell(gpu, storage, host, &b_set, arm, &mut streams[cell.index()]).await
                            };
                            if let Err(e) = &result {
                                note_arm_error(arm, e)?;
                            }
                            let (ns, hashes) = result?;
                            times[cell.index()] = ns;
                            verification.cells.insert(cell, hashes.into_iter().map(CellHash::from).collect());
                        }
                        p.raw_samples.push(DSample::new(
                            set, times[Cell::HS.index()], times[Cell::MS.index()],
                            times[Cell::HC.index()], times[Cell::MC.index()],
                        )?);
                        let valid = verification.valid();
                        p.raw_verification.push(verification);
                        if !valid {
                            add(&mut p.mismatch_count, 1)?;
                            return Err(Failure::runtime("hash-parity-failed",
                                "split-pair source/payload/GPU/epoch mismatch; raw verification retained"));
                        }
                    }
                    Ok(())
                }.await;
                // Both helpers return only after every attempted read has finished.
                p.fd_proof = storage.source_upload_fd_proof_snapshot();
                p.fd_proof_hits_only =
                    c_proof_hits_only(&p.fd_proof, p.expected.expert_slots_per_cell);
                for cell in Cell::ALL {
                    let arm = p
                        .cells
                        .get_mut(&cell)
                        .ok_or_else(|| Failure::accounting("missing cell"))?;
                    arm.source_schedule = schedule_evidence(&arm.completed_source_sets)?;
                    arm.evidence.rates();
                    p.witnesses.insert(cell, streams[cell.index()].snapshot());
                }
                p.statistics = d_statistics(&p.raw_samples)?;
                result?;
                if !p.fd_proof_hits_only {
                    return Err(Failure::authority(format!(
                        "{} expected only proof hits: {:?}",
                        p.name, p.fd_proof,
                    )));
                }
                Ok(())
            }
            #[derive(Debug, Serialize)]
            struct DReport {
                schema: &'static str,
                args: Args,
                config_sha256: Option<String>,
                complete: bool,
                correctness_pass: bool,
                authoritative: bool,
                classification: String,
                failure: Option<String>,
                authority: CAuthority,
                warmup: DPhase,
                measured: DPhase,
                performance_required_for_correctness: bool,
                performance_authority: &'static str,
                interaction_interpretation_conditional_on_zero_retry_log: &'static str,
                primary_endpoint: &'static str,
                schedule_contract: &'static str,
                timing_contract: &'static str,
                interpretation_contract: &'static str,
                secondary_destination_endpoints: &'static str,
                retry_evidence_contract: &'static str,
                source_byte_evidence_contract: &'static str,
            }
            impl DReport {
                fn new(args: Args) -> Result<Self> {
                    let mut base = Report::new(args.clone()).authority;
                    base.control_source_api = B_API;
                    base.treatment_source_api = B_API;
                    base.control_destination = "aligned-host-arena";
                    base.treatment_destination = "wgpu-map-write-arena";
                    base.upload_capacity_bytes = MAPPED_ARENA;
                    Ok(Self {
                        schema: D_SCHEMA,
                        args,
                        config_sha256: None,
                        complete: false,
                        correctness_pass: false,
                        authoritative: false,
                        classification: "not-run".into(),
                        failure: None,
                        authority: CAuthority {
                            base,
                            cell_source_apis: Cell::ALL.into_iter().map(|c| (c, c.api())).collect(),
                            host_arena_capacity_bytes: HOST_ARENA,
                            max_width: MAX_WIDTH,
                            fd_preproof_completed: false,
                            fd_cache_capacity: 0,
                            preproof_universe_size: D_UNIVERSE,
                            preproof_ordered_expert_ids: expert_sequence(D_UNIVERSE, NAMESPACE).map_err(Failure::accounting)?,
                            preproof: SourceUploadFdProofSnapshot::default(),
                            after_preproof_telemetry_reset: SourceUploadFdProofSnapshot::default(),
                            after_warmup_telemetry_reset: SourceUploadFdProofSnapshot::default(),
                        },
                        warmup: DPhase::new("warmup", WARMUP_COUNT)?,
                        measured: DPhase::new("measured", MEASURED_COUNT)?,
                        performance_required_for_correctness: false,
                        performance_authority: "PENDING_EXTERNAL_RETRY_LOG_AUDIT",
                        interaction_interpretation_conditional_on_zero_retry_log: "INCONCLUSIVE",
                        primary_endpoint: "112 K=2..8 split-pair blocks: sum((MC-HC)-(MS-HS)); percent denominator=sum(HC)",
                        schedule_contract: D_SCHEDULE_CONTRACT,
                        timing_contract: C_TIMER_CONTRACT,
                        interpretation_contract: D_INTERPRETATION,
                        secondary_destination_endpoints: D_SECONDARY,
                        retry_evidence_contract: D_RETRY,
                        source_byte_evidence_contract: "Each successful helper must return exactly K*FULL bytes. Failed-helper partial physical I/O is unavailable, never inferred as zero. Each cell retains ordered per-set full-source/payload/verified-GPU hashes and concatenated byte-stream witnesses. Equality is required within HS/MS and HC/MC separately; A and B are disjoint and are not compared for byte equality. Verification/reset/copy/readback are outside source timers.",
                    })
                }
                fn authority_valid(&self) -> bool {
                    let a = &self.authority;
                    let b = &a.base;
                    self.schema == D_SCHEMA
                        && !self.performance_required_for_correctness
                        && self.args.iterations == 128
                        && self.args.warmup_iterations == 32
                        && self.warmup.schedule_count == WARMUP_COUNT
                        && self.measured.schedule_count == MEASURED_COUNT
                        && a.cell_source_apis
                            == Cell::ALL.into_iter().map(|c| (c, c.api())).collect()
                        && a.host_arena_capacity_bytes == HOST_ARENA
                        && a.max_width == MAX_WIDTH
                        && b.same_source_api
                        && b.control_source_api == B_API
                        && b.treatment_source_api == B_API
                        && b.control_destination == "aligned-host-arena"
                        && b.treatment_destination == "wgpu-map-write-arena"
                        && b.upload_capacity_bytes == MAPPED_ARENA
                        && b.full_source_bytes == FULL
                        && b.block_alignment == ALIGN
                        && b.uth_prefix_bytes == PREFIX
                        && b.bare_payload_bytes == PAYLOAD
                        && b.physical_slot_bytes == SLOT
                        && b.source_timer_excludes_allocation
                        && b.source_timer_excludes_map_async_device_poll
                        && b.source_timer_excludes_alignment_setup
                        && b.source_timer_excludes_hashes_readback_fd_evidence
                        && b.source_timer_excludes_gpu_copy_unmap
                        && b.linux
                        && b.expected_adapter_name == "NVIDIA L4"
                        && self.args.expected_adapter_name == "NVIDIA L4"
                        && b.adapter_authoritative
                        && b.direct_io_requested
                        && b.packed_storage == Some(false)
                        && b.exact_geometry
                        && a.fd_preproof_completed
                        && a.fd_cache_capacity >= 256
                        && a.preproof_universe_size == 256
                        && expert_sequence(256, NAMESPACE)
                            .is_ok_and(|ids| ids == a.preproof_ordered_expert_ids)
                        && a.preproof
                            == SourceUploadFdProofSnapshot {
                                source_upload_fd_proof_requests: 256,
                                source_upload_fd_proof_misses: 256,
                                ..SourceUploadFdProofSnapshot::default()
                            }
                        && a.after_preproof_telemetry_reset
                            == SourceUploadFdProofSnapshot::default()
                        && a.after_warmup_telemetry_reset == SourceUploadFdProofSnapshot::default()
                        && c_proof_hits_only(&self.warmup.fd_proof, 140)
                        && c_proof_hits_only(&self.measured.fd_proof, 560)
                }
                fn classify(&mut self) -> Result<()> {
                    self.complete = true;
                    self.correctness_pass = false;
                    self.authoritative = false;
                    self.interaction_interpretation_conditional_on_zero_retry_log = "AMBIGUOUS";
                    if !self.authority_valid() {
                        self.classification = "authority-failed".into();
                    } else if !self.warmup.successful() || !self.measured.successful() {
                        self.classification = "evidence-reconciliation-failed".into();
                    } else {
                        self.correctness_pass = true;
                        self.authoritative = true;
                        self.classification = "split-pair-interaction-complete".into();
                        self.interaction_interpretation_conditional_on_zero_retry_log =
                            interaction_interpretation(&self.measured.statistics)?;
                    }
                    Ok(())
                }
                fn fail(&mut self, failure: Failure) {
                    self.complete = failure.complete;
                    self.correctness_pass = false;
                    self.authoritative = false;
                    self.classification = failure.classification.into();
                    self.failure = Some(failure.detail);
                    self.interaction_interpretation_conditional_on_zero_retry_log = "AMBIGUOUS";
                }
            }
            async fn execute(report: &mut DReport) -> Result<()> {
                if report.args.iterations != 128 || report.args.warmup_iterations != 32 {
                    return Err(Failure::runtime("invalid-arguments", "D requires --iterations 128 --warmup-iterations 32 for CLI compatibility; D generates 112 measured split-pair blocks (r=0..15) and 28 warmup blocks (r=16..19)"));
                }
                let bytes = std::fs::read(&report.args.config)
                    .map_err(|e| Failure::runtime("config-failed", e))?;
                report.config_sha256 = Some(sha(&bytes));
                let text = std::str::from_utf8(&bytes)
                    .map_err(|e| Failure::runtime("config-failed", e))?;
                let config: Config =
                    toml::from_str(text).map_err(|e| Failure::runtime("config-failed", e))?;
                config.validate().map_err(Failure::authority)?;
                let a = &mut report.authority;
                a.base.direct_io_requested = !config.storage.no_direct;
                a.base.packed_storage = Some(
                    config.storage.packed_blob.is_some()
                        || config.storage.packed_manifest.is_some(),
                );
                a.base.source_data_dir = Some(config.model.data_dir.clone());
                validate_geometry(&config)?;
                a.base.exact_geometry = true;
                if !a.base.linux
                    || report.args.expected_adapter_name != "NVIDIA L4"
                    || !a.base.direct_io_requested
                    || a.base.packed_storage != Some(false)
                {
                    return Err(Failure::authority("requires Linux, exact NVIDIA L4 Vulkan, O_DIRECT, unpacked full-file Qwen geometry"));
                }
                let storage = NvmeStorage::new(StorageConfig {
                    base_path: config.model.data_dir,
                    expert_size: FULL,
                    block_align: ALIGN,
                    use_direct_io: true,
                    num_experts_per_layer: Some(128),
                })
                .map_err(|e| Failure::runtime("source-failed", e))?;
                if storage.is_packed() {
                    return Err(Failure::authority("packed storage forbidden"));
                }
                a.fd_cache_capacity = storage.max_open_files();
                if a.fd_cache_capacity < a.preproof_universe_size {
                    return Err(Failure::authority(
                        "fd cache cannot retain the complete deterministic universe",
                    ));
                }
                let preproof = storage.preprove_source_upload_fds(&a.preproof_ordered_expert_ids);
                a.preproof = storage.source_upload_fd_proof_snapshot();
                preproof.map_err(|e| Failure::authority(format!("fd preproof: {e}")))?;
                if a.preproof.source_upload_fd_proof_requests != a.preproof_universe_size as u64
                    || a.preproof.source_upload_fd_proof_misses != a.preproof_universe_size as u64
                    || a.preproof.source_upload_fd_proof_hits != 0
                    || a.preproof.source_upload_fd_proof_failures != 0
                {
                    return Err(Failure::authority(
                        "fresh universe preproof counters do not reconcile",
                    ));
                }
                a.fd_preproof_completed = true;
                storage.reset_source_upload_fd_proof_telemetry();
                a.after_preproof_telemetry_reset = storage.source_upload_fd_proof_snapshot();
                if a.after_preproof_telemetry_reset != SourceUploadFdProofSnapshot::default() {
                    return Err(Failure::authority("preproof telemetry reset failed"));
                }
                let gpu = Gpu::with_upload_capacity(&mut a.base, MAPPED_ARENA).await?;
                let mut host = AlignedBuffer::new(HOST_ARENA, ALIGN);
                d_phase(&mut report.warmup, &gpu, &storage, &mut host).await?;
                if !report.warmup.successful() {
                    return Err(Failure::authority("warmup evidence did not reconcile"));
                }
                storage.reset_source_upload_fd_proof_telemetry();
                report.authority.after_warmup_telemetry_reset =
                    storage.source_upload_fd_proof_snapshot();
                if report.authority.after_warmup_telemetry_reset
                    != SourceUploadFdProofSnapshot::default()
                {
                    return Err(Failure::authority("warmup telemetry reset failed"));
                }
                d_phase(&mut report.measured, &gpu, &storage, &mut host).await?;
                gpu.check()?;
                report.classify()?;
                Ok(())
            }
            pub(crate) async fn run_command(
                args: Args,
            ) -> std::result::Result<(), Box<dyn std::error::Error>> {
                let mut output = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&args.report_out)?;
                let mut report = DReport::new(args).map_err(|e| io::Error::other(e.detail))?;
                match std::panic::AssertUnwindSafe(execute(&mut report))
                    .catch_unwind()
                    .await
                {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => report.fail(e),
                    Err(p) => {
                        let detail = p
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_else(|| "non-string panic".into());
                        report.fail(Failure::runtime(
                            "runtime-failed",
                            format!("D diagnostic panic: {detail}"),
                        ));
                    }
                }
                serde_json::to_writer_pretty(&mut output, &report)?;
                output.write_all(b"\n")?;
                output.sync_all()?;
                if report.complete {
                    Ok(())
                } else {
                    Err(io::Error::other(format!(
                        "{}: {}",
                        report.classification,
                        report.failure.as_deref().unwrap_or("incomplete")
                    ))
                    .into())
                }
            }

            #[cfg(test)]
            mod tests {
                use super::*;
                fn args() -> Args {
                    Args {
                        config: "unused.toml".into(),
                        expected_adapter_name: "NVIDIA L4".into(),
                        warmup_iterations: 32,
                        iterations: 128,
                        report_out: "unused.json".into(),
                    }
                }
                fn samples(times: [u64; 4]) -> Vec<DSample> {
                    d_schedule(112)
                        .unwrap()
                        .into_iter()
                        .map(|s| DSample::new(s, times[0], times[2], times[1], times[3]).unwrap())
                        .collect()
                }
                fn fixture_phase(name: &'static str, count: usize, times: [u64; 4]) -> DPhase {
                    let mut p = DPhase::new(name, count).unwrap();
                    let n = p.expected.expert_slots_per_cell;
                    let sets = p.expected.block_count;
                    p.raw_samples = p
                        .schedule
                        .iter()
                        .cloned()
                        .map(|s| DSample::new(s, times[0], times[2], times[1], times[3]).unwrap())
                        .collect();
                    p.statistics = d_statistics(&p.raw_samples).unwrap();
                    p.execution_trace = p
                        .schedule
                        .iter()
                        .flat_map(|s| {
                            s.execution_sequence
                                .iter()
                                .enumerate()
                                .map(|(ordinal, &cell)| ExecutedCell {
                                    set_index: s.set_index,
                                    source_role: s.role(cell),
                                    ordered_expert_ids: s.b_set(cell).ordered_expert_ids,
                                    ordinal,
                                    cell,
                                })
                        })
                        .collect();
                    p.raw_verification = p
                        .schedule
                        .iter()
                        .map(|s| VerificationSet {
                            source_set: s.clone(),
                            cells: Cell::ALL
                                .into_iter()
                                .map(|c| {
                                    (
                                        c,
                                        (0..s.width)
                                            .map(|_| CellHash {
                                                source: sha(if c.serial() {
                                                    b"serial"
                                                } else {
                                                    b"concurrent"
                                                }),
                                                payload: sha(b"payload"),
                                                gpu: sha(b"payload"),
                                                epoch: true,
                                            })
                                            .collect(),
                                    )
                                })
                                .collect(),
                        })
                        .collect();
                    for cell in Cell::ALL {
                        let arm = p.cells.get_mut(&cell).unwrap();
                        arm.source_sets_attempted = sets;
                        arm.serial_helper_calls = if cell.serial() { sets } else { 0 };
                        arm.concurrent_helper_calls = if cell.serial() { 0 } else { sets };
                        arm.completed_source_sets =
                            p.schedule.iter().map(|s| s.b_set(cell)).collect();
                        arm.source_schedule = p.expected.cell_source_schedules[&cell].clone();
                        let a = &mut arm.evidence;
                        a.ops_attempted = n;
                        a.source_read_attempts = n;
                        a.source_read_ops = n;
                        a.full_source_bytes = n * FULL as u64;
                        a.payload_ops = n;
                        a.payload_bytes = n * PAYLOAD as u64;
                        a.upload_ops = n;
                        a.gpu_copied_bytes = n * PAYLOAD as u64;
                        a.epoch_bytes = n * 4;
                        a.gpu_completed_ops = n;
                        a.verified_ops = n;
                        a.verification_readback_bytes = n * SLOT as u64;
                        a.verification_destination_reset_ops = n;
                        a.verification_destination_reset_bytes = n * SLOT as u64;
                        a.pointers.observations = n;
                        a.pointers.aligned = n;
                        a.fd_evidence = FdEvidence {
                            checks: n,
                            direct_observed: n,
                            full_file_length_observed: n,
                            ..FdEvidence::default()
                        };
                        a.times.source_direct_read_ns = sets * times[cell.index()];
                        if cell.mapped() {
                            a.map_attempts = sets;
                            a.maps_completed = sets;
                            a.unmaps = sets;
                            a.pointers.gpu_offset_checks = n;
                            a.explicit_copy_buffer_bytes = n * PAYLOAD as u64;
                        } else {
                            a.cpu_payload_copy_bytes = n * PAYLOAD as u64;
                        }
                        p.witnesses.insert(
                            cell,
                            Witnesses {
                                full_source_sha256: sha(if cell.serial() {
                                    b"serial"
                                } else {
                                    b"concurrent"
                                }),
                                bare_payload_sha256: sha(b"payload"),
                                gpu_destination_payload_sha256: sha(b"payload"),
                            },
                        );
                    }
                    p.fd_proof = SourceUploadFdProofSnapshot {
                        source_upload_fd_proof_requests: n * 4,
                        source_upload_fd_proof_hits: n * 4,
                        ..SourceUploadFdProofSnapshot::default()
                    };
                    p.fd_proof_hits_only = true;
                    assert!(p.successful());
                    p
                }
                fn fixture(times: [u64; 4]) -> DReport {
                    let mut r = DReport::new(args()).unwrap();
                    let a = &mut r.authority;
                    a.base.linux = true;
                    a.base.adapter_authoritative = true;
                    a.base.direct_io_requested = true;
                    a.base.packed_storage = Some(false);
                    a.base.exact_geometry = true;
                    a.fd_cache_capacity = 256;
                    a.fd_preproof_completed = true;
                    a.preproof = SourceUploadFdProofSnapshot {
                        source_upload_fd_proof_requests: 256,
                        source_upload_fd_proof_misses: 256,
                        ..SourceUploadFdProofSnapshot::default()
                    };
                    r.warmup = fixture_phase("warmup", 28, times);
                    r.measured = fixture_phase("measured", 112, times);
                    r
                }

                #[test]
                fn source_to_upload_copy_elision_d_hash_pins_and_unchanged_runners() {
                    let whole = include_str!("gpu_native_source_to_upload_copy_elision.rs");
                    let c = whole
                        .split("    pub(super) mod hma1c_c {")
                        .nth(1)
                        .unwrap()
                        .split("        #[cfg(test)]")
                        .next()
                        .unwrap();
                    assert_eq!(
                        sha(c.as_bytes()),
                        "77ed07c20c3a66f395d6c470dde18dbd9d18574f176f79466ca1a33ed3016422"
                    );
                    let io = include_str!("io_provider.rs")
                        .split("\n// Diagnostic-only full-schedule fixture.")
                        .next()
                        .unwrap();
                    assert_eq!(
                        sha(io.as_bytes()),
                        "e6c9148935d47e01b8ab4f498250a22d9ca0144bd899233bbfc2bfde2f9501cd"
                    );
                    let d = whole
                        .split("        pub(crate) mod hma1c_d {")
                        .nth(1)
                        .unwrap()
                        .split("            #[cfg(test)]")
                        .next()
                        .unwrap();
                    assert!(!d.contains(".read_experts_serial_into_aligned_slices("));
                    assert!(!d.contains(".read_experts_batch_into_aligned_slices("));
                    let phase = d
                        .split("async fn d_phase(")
                        .nth(1)
                        .unwrap()
                        .split("struct DReport")
                        .next()
                        .unwrap();
                    assert_eq!(phase.matches("host_cell(").count(), 1);
                    assert_eq!(phase.matches("mapped_cell(").count(), 1);
                    assert!(phase
                        .split_whitespace()
                        .collect::<String>()
                        .contains("letb_set=set.b_set(cell);"));
                    let exec = d.split("async fn execute(").nth(1).unwrap();
                    assert_eq!(exec.matches("preprove_source_upload_fds(").count(), 1);
                    assert_eq!(
                        exec.matches("reset_source_upload_fd_proof_telemetry()")
                            .count(),
                        2
                    );
                    assert!(
                        exec.find("preprove_source_upload_fds(").unwrap()
                            < exec
                                .find("reset_source_upload_fd_proof_telemetry()")
                                .unwrap()
                    );
                    for (count, ids, order, complete) in [
                        (
                            28,
                            "a9a5e7636c44f3bb6070a63e63b5532f22809262f0fc9d7c05f1886ce442a292",
                            "57fae1360cc13bfa62c13af6e141668cb388712dbb3f5c19501707e7a5c870bb",
                            "e0caded6746d9ccc65d416cd761c3c0c8de23a5fe130b660c0f024439d1b8e05",
                        ),
                        (
                            112,
                            "aa60ee0e4bfe50874b18d70857da221ec920e6f8f137363ebac568d83b72a802",
                            "40e1dc2fc03f4cae91ab359e5620d960260970c0fce8f0f2bd9e1d8a20135421",
                            "c347a06c53df847b2d7409ccbd605939625c81a899b15180eb23d7468da022dd",
                        ),
                    ] {
                        let p = plan(&d_schedule(count).unwrap()).unwrap();
                        assert_eq!(
                            p.universe_sha256,
                            "7a993987f13a94c6b3cc3f75a38dde55919a820d68b57e28c5f9ac599c1565b3"
                        );
                        assert_eq!(p.ordered_ab_ids_sha256, ids);
                        assert_eq!(p.execution_order_sha256, order);
                        assert_eq!(p.complete_schedule_sha256, complete);
                    }
                }
                #[test]
                fn source_to_upload_copy_elision_d_synthetic_json_for_independent_audit() {
                    let mut r = fixture([1000, 1000, 1000, 1100]);
                    r.classify().unwrap();
                    let json = serde_json::to_string_pretty(&r).unwrap();
                    assert!(json.contains("split-pair-interaction-complete"));
                    // Optional test-only export lets the independent Python tool
                    // verify Rust's serialization and exact stats. Never hardware evidence.
                    if let Some(path) = std::env::var_os("MER_HMA1CD_TEST_REPORT_OUT") {
                        std::fs::write(path, json).unwrap();
                    }
                }
                #[test]
                fn source_to_upload_copy_elision_d_exact_schedule_independent_recomputation() {
                    for (count, rounds) in [(112, 0..16), (28, 16..20)] {
                        let s = d_schedule(count).unwrap();
                        let universe: Vec<u32> = (0..256).map(|i| i * 6143 / 255).collect();
                        let mut expected = Vec::new();
                        for r in rounds {
                            for k in 2..=8 {
                                let i = r * 7 + k - 2;
                                let class = (r + 5 * (k - 2)) % 16;
                                let a: Vec<_> =
                                    (0..k).map(|j| universe[(i * 17 + j * 13) % 256]).collect();
                                let b: Vec<_> = (0..k)
                                    .map(|j| universe[(i * 17 + 128 + j * 13) % 256])
                                    .collect();
                                expected.push((i, r, k, class, a, b));
                            }
                        }
                        for (block, (i, r, k, class, a, b)) in s.iter().zip(expected) {
                            assert_eq!(
                                (
                                    block.set_index,
                                    block.round,
                                    block.width,
                                    block.design_class
                                ),
                                (i, r, k, class)
                            );
                            assert_eq!(block.a_ordered_expert_ids, a);
                            assert_eq!(block.b_ordered_expert_ids, b);
                            assert!(a.iter().all(|id| !b.contains(id)));
                            let mut reads = BTreeMap::new();
                            for cell in block.execution_sequence {
                                for id in block.b_set(cell).ordered_expert_ids {
                                    *reads.entry(id).or_insert(0) += 1;
                                }
                            }
                            assert_eq!(reads.len(), 2 * k);
                            assert!(reads.values().all(|n| *n == 2));
                            let serial = if class & 1 == 0 {
                                [Cell::HS, Cell::MS]
                            } else {
                                [Cell::MS, Cell::HS]
                            };
                            let concurrent = if class & 2 == 0 {
                                [Cell::HC, Cell::MC]
                            } else {
                                [Cell::MC, Cell::HC]
                            };
                            let expected = if class & 4 == 0 {
                                [serial[0], serial[1], concurrent[0], concurrent[1]]
                            } else {
                                [concurrent[0], concurrent[1], serial[0], serial[1]]
                            };
                            assert_eq!(block.execution_sequence, expected);
                            assert_eq!(
                                block.b_set(Cell::HS).ordered_expert_ids,
                                if class & 8 == 0 { a.clone() } else { b.clone() }
                            );
                            assert_eq!(
                                block.b_set(Cell::HC).ordered_expert_ids,
                                if class & 8 == 0 { b } else { a }
                            );
                        }
                        let e = plan(&s).unwrap();
                        assert_eq!(e.block_count, count as u64);
                        assert_eq!(e.expert_slots_per_cell, count as u64 * 5);
                        assert_eq!(e.source_bytes_per_cell, count as u64 * 5 * FULL as u64);
                    }
                    let s = d_schedule(112).unwrap();
                    for k in 2..=8 {
                        let classes: BTreeSet<_> = s
                            .iter()
                            .filter(|s| s.width == k)
                            .map(|s| s.design_class)
                            .collect();
                        assert_eq!(classes, (0..16).collect());
                        for bit in 0..4 {
                            assert_eq!(
                                s.iter()
                                    .filter(|s| s.width == k && s.factors()[bit])
                                    .count(),
                                8
                            );
                        }
                    }
                    for count in [1, 32, 128, 111, 113] {
                        assert!(d_schedule(count).is_err());
                    }
                    for mode in 0..8 {
                        let mut bad = s.clone();
                        match mode {
                            0 => bad[0].a_ordered_expert_ids[0] = bad[0].b_ordered_expert_ids[0],
                            1 => bad[0].design_class ^= 1,
                            2 => bad[0].execution_sequence.swap(0, 1),
                            3 => bad[0].serial_source_set = SourceRole::B,
                            4 => bad[0].concurrent_order_reversed = true,
                            5 => bad[0].width = 9,
                            6 => bad[0].set_index = 1,
                            _ => bad.swap(0, 1),
                        }
                        assert!(plan(&bad).is_err(), "{mode}");
                    }
                }
                fn varying(mut f: impl FnMut(&DSet) -> i64) -> Vec<DSample> {
                    d_schedule(112)
                        .unwrap()
                        .into_iter()
                        .map(|s| {
                            let d = f(&s);
                            DSample::new(s, 10000, 10000, 10000, (10000 + d) as u64).unwrap()
                        })
                        .collect()
                }
                fn interpretation(v: &[DSample]) -> &'static str {
                    interaction_interpretation(&d_statistics(v).unwrap()).unwrap()
                }
                #[test]
                fn source_to_upload_copy_elision_d_exact_thresholds_and_descriptive_classes() {
                    for (delta, expected) in [
                        (500, "STRONG_POSITIVE_INTERACTION"),
                        (499, "MATERIAL_POSITIVE_INTERACTION"),
                        (300, "MATERIAL_POSITIVE_INTERACTION"),
                        (299, "AMBIGUOUS"),
                        (101, "AMBIGUOUS"),
                        (0, "EVIDENCE_AGAINST_MATERIAL_INTERACTION"),
                        (-299, "AMBIGUOUS"),
                        (-300, "NEGATIVE_MATERIAL_INTERACTION"),
                    ] {
                        assert_eq!(interpretation(&varying(|_| delta)), expected, "{delta}");
                    }
                    // One class reverses, but every frozen interpretation stratum
                    // stays positive: individual classes must remain descriptive.
                    let v = varying(|s| if s.design_class == 0 { -100 } else { 1000 });
                    assert_eq!(interpretation(&v), "STRONG_POSITIVE_INTERACTION");
                    let huge = 1_000_000_000_000_000_000u64;
                    let v: Vec<_> = d_schedule(112)
                        .unwrap()
                        .into_iter()
                        .map(|s| {
                            DSample::new(s, huge, huge, huge, huge + huge / 100 * 3 - 1).unwrap()
                        })
                        .collect();
                    // Aggregate u64 totals intentionally fail closed on overflow.
                    assert!(d_statistics(&v).is_err());
                    let base = 100_000_000_000_000_000u64;
                    let v: Vec<_> = d_schedule(112)
                        .unwrap()
                        .into_iter()
                        .map(|s| {
                            DSample::new(s, base, base, base, base + base / 100 * 3 - 1).unwrap()
                        })
                        .collect();
                    assert_eq!(interpretation(&v), "AMBIGUOUS");
                }

                #[test]
                fn source_to_upload_copy_elision_d_against_exact_bounds_and_width_count() {
                    for sign in [-1, 1] {
                        // Four widths support the majority, all marginal and
                        // temporal signs agree, and both exact ratio bounds are inclusive.
                        assert_eq!(
                            interpretation(&varying(
                                |s| sign * if s.width <= 5 { 100 } else { -100 }
                            )),
                            "EVIDENCE_AGAINST_MATERIAL_INTERACTION"
                        );
                        assert_eq!(
                            interpretation(&varying(
                                |s| sign * if s.width <= 5 { 101 } else { -101 }
                            )),
                            "AMBIGUOUS"
                        );
                    }
                    let endpoint = |extra: i64| {
                        varying(|s| {
                            if s.set_index == 0 {
                                9700 + extra
                            } else if s.width <= 5 {
                                100
                            } else {
                                -100
                            }
                        })
                    };
                    let v = endpoint(0);
                    let stats = d_statistics(&v).unwrap();
                    assert_eq!(
                        stats
                            .primary_k2_through_k8
                            .exact
                            .totals
                            .interaction_delta_ns
                            * 100,
                        i128::from(stats.primary_k2_through_k8.exact.totals.hc_ns)
                    );
                    assert_eq!(interpretation(&v), "EVIDENCE_AGAINST_MATERIAL_INTERACTION");
                    assert_eq!(interpretation(&endpoint(1)), "AMBIGUOUS");
                    // Five positive widths exclude AGAINST even at tiny aggregate.
                    assert_eq!(
                        interpretation(&varying(|s| if s.width <= 6 { 1 } else { -1 })),
                        "AMBIGUOUS"
                    );
                    let mut v = varying(|_| 0);
                    v.pop();
                    assert_eq!(interpretation(&v), "INCONCLUSIVE");
                }
                #[test]
                fn source_to_upload_copy_elision_d_all_marginal_and_temporal_reversals_gate() {
                    for bit in 0..4 {
                        for level in [false, true] {
                            for sign in [-1, 1] {
                                // Reversed stratum has 29 small positive and 27 negative
                                // samples: majority/median stay positive while its sum reverses.
                                let mut seen = 0;
                                let v = varying(|s| {
                                    if s.factors()[bit] == level {
                                        seen += 1;
                                        if seen <= 29 {
                                            sign * 100
                                        } else {
                                            -sign * 200
                                        }
                                    } else {
                                        sign * 2000
                                    }
                                });
                                assert_eq!(
                                    interpretation(&v),
                                    "AMBIGUOUS",
                                    "bit={bit} level={level} sign={sign}"
                                );
                            }
                        }
                    }
                    for q in 0..4 {
                        for sign in [-1, 1] {
                            let v = varying(|s| {
                                if s.set_index / 28 == q {
                                    -sign * 100
                                } else {
                                    sign * 1000
                                }
                            });
                            assert_eq!(interpretation(&v), "AMBIGUOUS", "quartile {q} sign {sign}");
                        }
                    }
                    // Only four widths support a large aggregate.
                    assert_eq!(
                        interpretation(&varying(|s| if s.width <= 5 { 2000 } else { -100 })),
                        "AMBIGUOUS"
                    );
                    // Aggregate positive, median and majority negative.
                    assert_eq!(
                        interpretation(&varying(|s| if s.set_index < 55 { 2000 } else { -100 })),
                        "AMBIGUOUS"
                    );
                    // Exactly 56 positives is not a majority.
                    assert_eq!(
                        interpretation(&varying(|s| if s.set_index < 56 { 2000 } else { -100 })),
                        "AMBIGUOUS"
                    );
                }
                #[test]
                fn source_to_upload_copy_elision_d_exact_partitions_arithmetic_and_json() {
                    let v = varying(|s| (s.set_index as i64 - 30) * 10);
                    let stats = d_statistics(&v).unwrap();
                    assert_eq!(
                        stats
                            .primary_k2_through_k8
                            .exact
                            .totals
                            .interaction_delta_ns,
                        v.iter().map(|s| s.interaction_delta_ns).sum::<i128>()
                    );
                    assert!(stats.exact_matches(&d_statistics(&v).unwrap()));
                    for p in [
                        &stats.per_width,
                        &stats.temporal_quartiles,
                        &stats.descriptive_design_classes,
                    ]
                    .into_iter()
                    .chain(stats.binary_marginal_factors.values())
                    {
                        reconcile_partitions(
                            p.values().map(|s| &s.exact),
                            &stats.primary_k2_through_k8.exact,
                        )
                        .unwrap();
                    }
                    let mut bad = v.clone();
                    bad[0].interaction_delta_ns += 1;
                    assert!(d_statistics(&bad).is_err());
                    let mut bad = v.clone();
                    bad[0].mc_times_hs += 1;
                    assert!(d_statistics(&bad).is_err());
                    let mut bad = v.clone();
                    bad[0].source_set.set_index = 1;
                    assert!(d_statistics(&bad).is_err());
                    assert!(signed_add(i128::MAX, 1).is_err());
                    assert!(signed_sub(i128::MIN, 1).is_err());
                    assert!(product(u128::MAX, 2).is_err());
                    let large = DSample::new(
                        v[0].source_set.clone(),
                        u64::MAX,
                        u64::MAX,
                        u64::MAX,
                        u64::MAX,
                    )
                    .unwrap();
                    let json = serde_json::to_string(&large).unwrap();
                    assert!(json.contains(&large.mc_times_hs.to_string()));
                    assert!(exact_stats(&[large.clone(), large]).is_err());
                    let s = DSample::new(v[0].source_set.clone(), 100, 105, 200, 212).unwrap();
                    assert_eq!(
                        (
                            s.serial_destination_delta_ns,
                            s.concurrent_destination_delta_ns,
                            s.interaction_delta_ns,
                            s.mc_times_hs,
                            s.hc_times_ms,
                            s.multiplicative_interaction_direction
                        ),
                        (5, 12, 7, 21200, 21000, 1)
                    );
                    let mut stats = d_statistics(&v).unwrap();
                    stats
                        .primary_k2_through_k8
                        .descriptive
                        .interaction_percent_of_hc = Some(f64::NAN);
                    assert!(stats.exact_matches(&d_statistics(&v).unwrap()));
                }
                #[test]
                fn source_to_upload_copy_elision_d_correctness_independent_and_fail_closed() {
                    for times in [[1000, 1000, 1000, 1200], [1000; 4], [1000, 1000, 1000, 800]] {
                        let mut r = fixture(times);
                        r.classify().unwrap();
                        assert!(r.correctness_pass && r.authoritative);
                        assert_eq!(r.performance_authority, "PENDING_EXTERNAL_RETRY_LOG_AUDIT");
                    }
                    for mode in 0..27 {
                        let mut r = fixture([1000; 4]);
                        match mode {
                            0 => r.authority.fd_cache_capacity = 255,
                            1 => r.authority.preproof.source_upload_fd_proof_requests = 255,
                            2 => r.authority.preproof.source_upload_fd_proof_hits = 1,
                            3 => r.authority.preproof.source_upload_fd_proof_misses = 255,
                            4 => r.authority.preproof.source_upload_fd_proof_failures = 1,
                            5 => r.measured.fd_proof.source_upload_fd_proof_misses = 1,
                            6 => r.measured.fd_proof.source_upload_fd_proof_failures = 1,
                            7 => r.measured.fd_proof.source_upload_fd_proof_hits -= 1,
                            8 => r.measured.fd_proof.source_upload_fd_proof_requests += 1,
                            9 => {
                                r.measured
                                    .cells
                                    .get_mut(&Cell::MS)
                                    .unwrap()
                                    .serial_helper_calls += 1
                            }
                            10 => r.measured.cells.get_mut(&Cell::MC).unwrap().evidence.unmaps += 1,
                            11 => {
                                r.measured
                                    .cells
                                    .get_mut(&Cell::HS)
                                    .unwrap()
                                    .evidence
                                    .map_attempts = 1
                            }
                            12 => {
                                r.measured
                                    .cells
                                    .get_mut(&Cell::MC)
                                    .unwrap()
                                    .evidence
                                    .mapped_direct_io_rejections = 1
                            }
                            13 => {
                                r.measured
                                    .cells
                                    .get_mut(&Cell::HC)
                                    .unwrap()
                                    .evidence
                                    .full_source_bytes -= 1
                            }
                            14 => {
                                r.measured
                                    .cells
                                    .get_mut(&Cell::HS)
                                    .unwrap()
                                    .evidence
                                    .exact_read_length_failures = 1
                            }
                            15 => r
                                .measured
                                .cells
                                .get_mut(&Cell::MS)
                                .unwrap()
                                .completed_source_sets[0]
                                .ordered_expert_ids
                                .reverse(),
                            16 => r.measured.execution_trace.swap(0, 1),
                            17 => {
                                r.measured.raw_verification[0]
                                    .cells
                                    .get_mut(&Cell::MS)
                                    .unwrap()[0]
                                    .source = sha(b"bad")
                            }
                            18 => {
                                r.measured.raw_verification[0]
                                    .cells
                                    .get_mut(&Cell::MC)
                                    .unwrap()[0]
                                    .gpu = sha(b"bad")
                            }
                            19 => {
                                r.measured.raw_verification[0]
                                    .cells
                                    .get_mut(&Cell::HC)
                                    .unwrap()[0]
                                    .epoch = false
                            }
                            20 => {
                                r.measured
                                    .witnesses
                                    .get_mut(&Cell::MS)
                                    .unwrap()
                                    .full_source_sha256 = sha(b"bad")
                            }
                            21 => r.measured.raw_samples[0].interaction_delta_ns += 1,
                            22 => {
                                r.measured
                                    .statistics
                                    .binary_marginal_factors
                                    .get_mut(FACTORS[0])
                                    .unwrap()
                                    .get_mut(&0)
                                    .unwrap()
                                    .exact
                                    .totals
                                    .hs_ns += 1
                            }
                            23 => {
                                r.measured
                                    .statistics
                                    .temporal_quartiles
                                    .get_mut(&0)
                                    .unwrap()
                                    .exact
                                    .totals
                                    .samples += 1
                            }
                            24 => {
                                r.measured
                                    .statistics
                                    .descriptive_design_classes
                                    .get_mut(&0)
                                    .unwrap()
                                    .exact
                                    .totals
                                    .hs_ns += 1
                            }
                            25 => {
                                r.measured
                                    .cells
                                    .get_mut(&Cell::MC)
                                    .unwrap()
                                    .evidence
                                    .fallback_reads = 1
                            }
                            _ => r.measured.raw_samples[0]
                                .source_set
                                .a_ordered_expert_ids
                                .reverse(),
                        }
                        r.classify().unwrap();
                        assert!(!r.correctness_pass && !r.authoritative, "mutation {mode}");
                        assert_eq!(
                            r.interaction_interpretation_conditional_on_zero_retry_log,
                            "AMBIGUOUS"
                        );
                    }
                }
                #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
                async fn source_to_upload_copy_elision_d_exclusive_output_and_invalid_args_without_gpu(
                ) {
                    let dir = std::env::temp_dir().join(format!(
                        "mer-hma1cd-{}-{}",
                        std::process::id(),
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_nanos()
                    ));
                    std::fs::create_dir_all(&dir).unwrap();
                    let mut a = args();
                    a.config = dir.join("missing.toml");
                    a.report_out = dir.join("out.json");
                    assert!(run_command(a.clone()).await.is_err());
                    let bytes = std::fs::read(&a.report_out).unwrap();
                    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                    assert_eq!(json["schema"], D_SCHEMA);
                    assert_eq!(json["classification"], "config-failed");
                    assert_eq!(json["measured"]["expected"]["block_count"], 112);
                    assert!(run_command(a.clone()).await.is_err());
                    assert_eq!(std::fs::read(&a.report_out).unwrap(), bytes);
                    a.iterations = 112;
                    a.report_out = dir.join("invalid.json");
                    assert!(run_command(a.clone()).await.is_err());
                    let json: serde_json::Value =
                        serde_json::from_slice(&std::fs::read(&a.report_out).unwrap()).unwrap();
                    assert_eq!(json["classification"], "invalid-arguments");
                    std::fs::remove_dir_all(dir).unwrap();
                }
            }
        }
    }
}
