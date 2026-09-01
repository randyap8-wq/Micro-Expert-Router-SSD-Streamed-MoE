//! Benchmark-only reporting and orchestration for the production GPU-native
//! real-model token loop.
//!
//! This module does not define an inference variant. The command path creates
//! an ordinary [`crate::gpu_native_token_loop::GpuNativeRequestState`] and
//! calls only [`crate::gpu_native_token_loop::GpuNativeTokenLoop::step_token`].

use crate::backend::{GpuDeviceIdentity, GpuExpertIoSnapshot, GpuExpertMemorySnapshot};
use crate::engine::RoutedExpertExecutionSnapshot;
use crate::gpu_native_residency::GpuNativeTieredResidencySnapshot;
use crate::gpu_native_token_loop::{
    GpuNativeModelGeometry, GpuNativeRecoverySnapshot, GpuNativeTokenLoopSnapshot,
};
use crate::greedy_parity::{
    BackgroundShutdownEvidence, ModelIdentityEvidence, ModelLoadEvidence, RuntimeCacheSnapshot,
};
use crate::qualification::{
    BuildProvenance, ExecutionPlanEvidence, ExpertMetadataEvidence, QualificationArtifacts,
};
use serde::Serialize;
use std::path::PathBuf;

pub(crate) const SCHEMA: &str = "mer.gpu-native-real-benchmark.v4";
pub(crate) const MODE: &str = "gpu-native-real-benchmark";
pub(crate) const OPTIMIZATION: &str = "gpu-native-physical-source-of-truth-pr1bb";
pub(crate) const IMMEDIATE_COMPARISON_COMMIT: &str = "a39c58062ce167773248d3fa5618ac5fd55e54ba";
pub(crate) const PR1B_A_EXPERIMENT_COMMIT: &str = "9b1a1da029151dacbb1ad49a562d4eb60e80e260";
pub(crate) const PR1B_A_EXPERIMENT_REPORT_SHA256: &str =
    "ac98c83c52df95967a315452517fb2556e82771aa49c5a5b1ab239fa1b3ec4e4";
pub(crate) const ORIGINAL_BASELINE_COMMIT: &str = "db0664159fe4a57e5b630984b9229e233fa21487";

const EXPECTED_NUM_LAYERS: usize = 48;
const EXPECTED_NUM_EXPERTS: usize = 128;
const EXPECTED_TOP_K: usize = 8;
const EXPECTED_D_MODEL: usize = 2048;
const EXPECTED_D_FF: usize = 768;

#[derive(Clone, Debug)]
pub(crate) struct CommandArgs {
    pub(crate) config: PathBuf,
    pub(crate) prompt: Option<String>,
    pub(crate) request_json: Option<PathBuf>,
    pub(crate) output_tokens: Option<usize>,
    pub(crate) warmup_runs: usize,
    pub(crate) measured_runs: usize,
    pub(crate) cache_reset: crate::BenchRealCacheReset,
    pub(crate) greedy: bool,
    pub(crate) expected_adapter_name: String,
    pub(crate) report_out: Option<PathBuf>,
    pub(crate) progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BenchmarkFailure {
    pub(crate) stage: String,
    pub(crate) code: String,
    pub(crate) detail: String,
}

impl BenchmarkFailure {
    pub(crate) fn new(stage: &str, code: &str, detail: impl Into<String>) -> Self {
        Self {
            stage: stage.to_string(),
            code: code.to_string(),
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for BenchmarkFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for BenchmarkFailure {}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct ProductionSemantics {
    pub(crate) production_inference_math_changed: bool,
    pub(crate) production_q4_changed: bool,
    pub(crate) production_router_changed: bool,
    pub(crate) production_attention_changed: bool,
    pub(crate) production_rmsnorm_changed: bool,
    pub(crate) production_lm_head_changed: bool,
    pub(crate) production_residency_policy_changed: bool,
    pub(crate) production_replay_policy_changed: bool,
    pub(crate) production_prefetch_policy_changed: bool,
    pub(crate) diagnostic_trace_enabled: bool,
}

impl ProductionSemantics {
    pub(crate) const fn physical_source_of_truth_pr1bb() -> Self {
        Self {
            production_inference_math_changed: false,
            production_q4_changed: false,
            production_router_changed: false,
            production_attention_changed: false,
            production_rmsnorm_changed: false,
            production_lm_head_changed: false,
            production_residency_policy_changed: true,
            production_replay_policy_changed: true,
            production_prefetch_policy_changed: false,
            diagnostic_trace_enabled: false,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BenchmarkProvenance {
    pub(crate) build: BuildProvenance,
    pub(crate) executable_canonical_path: String,
    pub(crate) executable_sha256: String,
    pub(crate) resolved_config_sha256: String,
    pub(crate) artifacts: QualificationArtifacts,
    pub(crate) expert_metadata: ExpertMetadataEvidence,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RequestEvidence {
    pub(crate) prompt_sha256: String,
    pub(crate) prompt_token_ids_sha256: String,
    pub(crate) prompt_token_count: usize,
    pub(crate) requested_output_tokens: usize,
    pub(crate) greedy: bool,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct CacheResidencyConfiguration {
    pub(crate) ram_cache_slots: usize,
    pub(crate) block_align: usize,
    pub(crate) direct_io: bool,
    pub(crate) pipeline_depth: u32,
    pub(crate) partial_load_fraction: f64,
    pub(crate) pin_after_observations: u64,
    pub(crate) packed_blob: Option<String>,
    pub(crate) packed_manifest: Option<String>,
    pub(crate) gpu_cache_enabled: bool,
    pub(crate) gpu_vram_capacity_mb: usize,
    pub(crate) gpu_vram_anchor_ratio: f32,
    pub(crate) gpu_promote_after_hits: u64,
    pub(crate) gpu_cache_dtype: String,
    pub(crate) gpu_native_max_seq_len: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct PredictorPrefetchConfiguration {
    pub(crate) predict_fanout: usize,
    pub(crate) predict_min_prob: f64,
    pub(crate) max_concurrent_prefetches: usize,
    pub(crate) max_fetch_yields: usize,
    pub(crate) locality_enabled: bool,
    pub(crate) locality_window: usize,
    pub(crate) locality_threshold_pct: f32,
    pub(crate) speculator_enabled: bool,
    pub(crate) speculator_hidden_dim: usize,
    pub(crate) speculator_top_k: usize,
    pub(crate) affinity_enabled: bool,
    pub(crate) affinity_neighbors_k: usize,
    pub(crate) affinity_decay_epoch: u64,
    pub(crate) prefetch_governor: bool,
    pub(crate) prefetch_precision_floor: f64,
    pub(crate) prefetch_contention_weight: f64,
    pub(crate) cost_aware_eviction: bool,
    pub(crate) pregate_enabled: bool,
    pub(crate) static_residency_fraction: f64,
    pub(crate) static_residency_warmup_tokens: u64,
    pub(crate) static_residency_profile: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct ProductionConfiguration {
    pub(crate) q4_dtype: String,
    pub(crate) q4_layout: Option<String>,
    pub(crate) cache_residency: CacheResidencyConfiguration,
    pub(crate) predictor_prefetch: PredictorPrefetchConfiguration,
}

impl ProductionConfiguration {
    pub(crate) fn from_config(
        cfg: &crate::config::Config,
        metadata: &ExpertMetadataEvidence,
    ) -> Self {
        Self {
            q4_dtype: cfg.model.dtype.as_str().to_string(),
            q4_layout: metadata.q4_0_layout.clone(),
            cache_residency: CacheResidencyConfiguration {
                ram_cache_slots: cfg.storage.cache_slots,
                block_align: cfg.storage.block_align,
                direct_io: !cfg.storage.no_direct,
                pipeline_depth: cfg.storage.pipeline_depth,
                partial_load_fraction: cfg.storage.partial_load_fraction,
                pin_after_observations: cfg.storage.pin_after_observations,
                packed_blob: cfg
                    .storage
                    .packed_blob
                    .as_ref()
                    .map(|path| path.display().to_string()),
                packed_manifest: cfg
                    .storage
                    .packed_manifest
                    .as_ref()
                    .map(|path| path.display().to_string()),
                gpu_cache_enabled: cfg.gpu_cache.enabled,
                gpu_vram_capacity_mb: cfg.gpu_cache.vram_capacity_mb,
                gpu_vram_anchor_ratio: cfg.gpu_cache.vram_anchor_ratio,
                gpu_promote_after_hits: cfg.gpu_cache.promote_after_hits,
                gpu_cache_dtype: cfg.gpu_cache.dtype.clone(),
                gpu_native_max_seq_len: cfg.real_transformer.gpu_native_max_seq_len,
            },
            predictor_prefetch: PredictorPrefetchConfiguration {
                predict_fanout: cfg.storage.predict_fanout,
                predict_min_prob: cfg.storage.predict_min_prob,
                max_concurrent_prefetches: cfg.real_transformer.max_concurrent_prefetches,
                max_fetch_yields: cfg.real_transformer.max_fetch_yields,
                locality_enabled: cfg.predictive.locality_enabled,
                locality_window: cfg.predictive.locality_window,
                locality_threshold_pct: cfg.predictive.locality_threshold_pct,
                speculator_enabled: cfg.predictive.speculator_enabled,
                speculator_hidden_dim: cfg.predictive.speculator_hidden_dim,
                speculator_top_k: cfg.predictive.speculator_top_k,
                affinity_enabled: cfg.predictive.affinity_enabled,
                affinity_neighbors_k: cfg.predictive.affinity_neighbors_k,
                affinity_decay_epoch: cfg.predictive.affinity_decay_epoch,
                prefetch_governor: cfg.predictive.prefetch_governor,
                prefetch_precision_floor: cfg.predictive.prefetch_precision_floor,
                prefetch_contention_weight: cfg.predictive.prefetch_contention_weight,
                cost_aware_eviction: cfg.predictive.cost_aware_eviction,
                pregate_enabled: cfg.predictive.pregate_enabled,
                static_residency_fraction: cfg.predictive.static_residency_fraction,
                static_residency_warmup_tokens: cfg.predictive.static_residency_warmup_tokens,
                static_residency_profile: cfg.predictive.static_residency_profile.clone(),
            },
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RuntimeContractInput {
    pub(crate) real_transformer_enabled: bool,
    pub(crate) real_transformer_gpu_native: bool,
    pub(crate) compute_offload: crate::backend::ComputeOffload,
    pub(crate) legacy_execution_plan: ExecutionPlanEvidence,
    pub(crate) token_loop_geometry: Option<GpuNativeModelGeometry>,
    pub(crate) authoritative_device: Option<GpuDeviceIdentity>,
    pub(crate) model_load: ModelLoadEvidence,
    pub(crate) routed_failure_policy: crate::engine::RoutedExpertGpuFailurePolicy,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeContractEvidence {
    pub(crate) real_transformer_enabled: bool,
    pub(crate) real_transformer_gpu_native: bool,
    pub(crate) compute_offload: String,
    pub(crate) ordinary_step_token_only: bool,
    pub(crate) legacy_execution_plan: ExecutionPlanEvidence,
    pub(crate) token_loop_geometry: GpuNativeModelGeometry,
    pub(crate) strict_fail_closed_routed_experts: bool,
}

pub(crate) fn validate_runtime_contract(
    input: &RuntimeContractInput,
    expected_adapter_name: &str,
) -> Result<(RuntimeContractEvidence, GpuDeviceIdentity), BenchmarkFailure> {
    if expected_adapter_name.trim().is_empty() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "missing-expected-adapter",
            "bench-gpu-native-real requires a nonempty exact adapter name",
        ));
    }
    if !input.real_transformer_enabled
        || !input.real_transformer_gpu_native
        || input.compute_offload != crate::backend::ComputeOffload::Gpu
    {
        return Err(BenchmarkFailure::new(
            "startup",
            "gpu-native-runtime-contract",
            format!(
                "requires real_transformer.enabled=true, gpu_native=true, and compute_offload=gpu; observed enabled={} gpu_native={} compute_offload={:?}",
                input.real_transformer_enabled,
                input.real_transformer_gpu_native,
                input.compute_offload,
            ),
        ));
    }

    // ExecutionPlanEvidence is retained as legacy context evidence. Its CPU
    // markers are not interpreted as placement of the separate authoritative
    // GPU-native full-token loop.
    let plan = &input.legacy_execution_plan;
    if plan.requested != "gpu"
        || plan.resolved != "gpu"
        || plan.embeddings != "cpu"
        || plan.lm_head != "cpu"
        || plan.dense_projections != "cpu"
        || plan.attention != "gpu"
        || plan.kv != "gpu"
        || plan.router != "cpu"
        || plan.routed_experts != "cpu"
        || plan.routed_expert_dtype != "q4_0"
        || plan.fallback_occurred
    {
        return Err(BenchmarkFailure::new(
            "startup",
            "legacy-execution-context-contract",
            format!(
                "legacy execution-context evidence drifted from the corrected GPU-native contract: {plan:?}"
            ),
        ));
    }

    let geometry = input.token_loop_geometry.ok_or_else(|| {
        BenchmarkFailure::new(
            "startup",
            "missing-gpu-native-token-loop",
            "authoritative runtime did not construct gpu_native_token_loop",
        )
    })?;
    if geometry.num_layers != EXPECTED_NUM_LAYERS
        || geometry.num_experts != EXPECTED_NUM_EXPERTS
        || geometry.top_k != EXPECTED_TOP_K
        || geometry.d_model != EXPECTED_D_MODEL
        || geometry.d_ff != EXPECTED_D_FF
    {
        return Err(BenchmarkFailure::new(
            "startup",
            "wrong-model-geometry",
            format!(
                "expected Qwen geometry layers={EXPECTED_NUM_LAYERS} experts={EXPECTED_NUM_EXPERTS} top_k={EXPECTED_TOP_K} d_model={EXPECTED_D_MODEL} d_ff={EXPECTED_D_FF}; observed {geometry:?}"
            ),
        ));
    }

    let load = &input.model_load;
    if !load.strict
        || load.required_tensors == 0
        || load.loaded_tensors != load.required_tensors
        || load.seeded_fallback_remained
        || load.loader == "seeded"
    {
        return Err(BenchmarkFailure::new(
            "startup",
            "incomplete-model-load",
            format!("strict real checkpoint load was incomplete: {load:?}"),
        ));
    }
    if input.routed_failure_policy != crate::engine::RoutedExpertGpuFailurePolicy::StrictFailClosed
    {
        return Err(BenchmarkFailure::new(
            "startup",
            "cpu-fallback-policy-enabled",
            "routed-expert failure policy was not strict-fail-closed",
        ));
    }

    let device = input.authoritative_device.clone().ok_or_else(|| {
        BenchmarkFailure::new(
            "startup",
            "missing-authoritative-adapter",
            "authoritative GPU device identity was unavailable",
        )
    })?;
    if device.name != expected_adapter_name {
        return Err(BenchmarkFailure::new(
            "startup",
            "wrong-adapter",
            format!(
                "selected adapter {:?} did not equal expected adapter {:?}",
                device.name, expected_adapter_name
            ),
        ));
    }
    if device.software_adapter || device.device_type.eq_ignore_ascii_case("cpu") {
        return Err(BenchmarkFailure::new(
            "startup",
            "software-adapter",
            format!("selected adapter is not authoritative hardware: {device:?}"),
        ));
    }
    if device.wgpu_backend != "vulkan"
        || device.compute_plane != "wgpu-vulkan"
        || device.driver.trim().is_empty()
    {
        return Err(BenchmarkFailure::new(
            "startup",
            "wrong-gpu-backend",
            format!(
                "requires wgpu_backend=vulkan, compute_plane=wgpu-vulkan, and nonempty driver identity; observed {device:?}"
            ),
        ));
    }
    if expected_adapter_name == "NVIDIA L4"
        && (device.name != "NVIDIA L4"
            || device.vendor_id != 0x10de
            || device.device_type != "DiscreteGpu")
    {
        return Err(BenchmarkFailure::new(
            "startup",
            "wrong-l4-hardware",
            format!("expected an NVIDIA L4 discrete GPU; observed {device:?}"),
        ));
    }

    Ok((
        RuntimeContractEvidence {
            real_transformer_enabled: input.real_transformer_enabled,
            real_transformer_gpu_native: input.real_transformer_gpu_native,
            compute_offload: "gpu".to_string(),
            ordinary_step_token_only: true,
            legacy_execution_plan: input.legacy_execution_plan.clone(),
            token_loop_geometry: geometry,
            strict_fail_closed_routed_experts: true,
        },
        device,
    ))
}

pub(crate) fn validate_source_config(cfg: &crate::config::Config) -> Result<(), BenchmarkFailure> {
    let rt = &cfg.real_transformer;
    if !rt.enabled
        || !rt.gpu_native
        || rt.compute_offload != crate::backend::ComputeOffload::Gpu
        || rt.weights_dir.is_none()
        || !rt.strict_weights
        || rt.allow_seeded_fallback
        || rt.allow_degraded_experts
        || rt.allow_nonfinite_attention_fallback
        || rt.allow_truncated_expert_payloads
        || cfg.distributed.enabled
        || !cfg.gpu_cache.enabled
        || cfg.gpu_cache.vram_capacity_mb == 0
        || cfg.model.dtype != crate::inference::WeightDtype::Q4_0
    {
        return Err(BenchmarkFailure::new(
            "preflight",
            "strict-gpu-native-config-required",
            "requires enabled strict real checkpoint inference, gpu_native=true, compute_offload=gpu, Q4_0 routed experts, enabled nonzero GPU cache, no distributed mode, and every fail-open policy disabled",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct CounterRatios {
    pub(crate) attempts_per_completed_position: f64,
    pub(crate) misses_per_completed_position: f64,
    pub(crate) replays_per_completed_position: f64,
    pub(crate) submissions_per_completed_position: f64,
}

impl CounterRatios {
    fn from_delta(delta: GpuNativeTokenLoopSnapshot) -> Result<Self, BenchmarkFailure> {
        if delta.tokens_completed == 0 {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "zero-completed-positions",
                "token-loop counter delta completed zero positions",
            ));
        }
        let completed = delta.tokens_completed as f64;
        Ok(Self {
            attempts_per_completed_position: delta.token_attempts as f64 / completed,
            misses_per_completed_position: delta.residency_miss_attempts as f64 / completed,
            replays_per_completed_position: delta.replay_attempts as f64 / completed,
            submissions_per_completed_position: delta.queue_submissions as f64 / completed,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct RecoveryRatios {
    pub(crate) resume_attempts_per_completed_position: f64,
    pub(crate) recovery_segments_per_completed_position: f64,
    pub(crate) checkpoint_captures_per_completed_position: f64,
    pub(crate) checkpoint_restores_per_completed_position: f64,
    pub(crate) full_token_replays_per_completed_position: f64,
    pub(crate) layers_encoded_per_completed_position: f64,
    pub(crate) attention_layers_reexecuted_per_completed_position: f64,
    pub(crate) expert_layers_reexecuted_per_completed_position: f64,
    pub(crate) invalid_tail_layers_encoded_per_completed_position: f64,
    pub(crate) residency_service_us_per_completed_position: f64,
    pub(crate) boundary_wait_us_per_completed_position: f64,
}

impl RecoveryRatios {
    fn from_delta(
        delta: GpuNativeRecoverySnapshot,
        completed_positions: u64,
    ) -> Result<Self, BenchmarkFailure> {
        if completed_positions == 0 {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "zero-completed-positions",
                "recovery ratios require at least one completed position",
            ));
        }
        let completed = completed_positions as f64;
        Ok(Self {
            resume_attempts_per_completed_position: delta.resume_attempts as f64 / completed,
            recovery_segments_per_completed_position: delta.recovery_segments as f64 / completed,
            checkpoint_captures_per_completed_position: delta.checkpoint_captures as f64
                / completed,
            checkpoint_restores_per_completed_position: delta.checkpoint_restores as f64
                / completed,
            full_token_replays_per_completed_position: delta.full_token_replay_attempts as f64
                / completed,
            layers_encoded_per_completed_position: delta.layers_encoded as f64 / completed,
            attention_layers_reexecuted_per_completed_position: delta.attention_layers_reexecuted
                as f64
                / completed,
            expert_layers_reexecuted_per_completed_position: delta.expert_layers_reexecuted as f64
                / completed,
            invalid_tail_layers_encoded_per_completed_position: delta.invalid_tail_layers_encoded
                as f64
                / completed,
            residency_service_us_per_completed_position: delta.residency_service_us as f64
                / completed,
            boundary_wait_us_per_completed_position: delta.boundary_wait_us as f64 / completed,
        })
    }
}

macro_rules! checked_delta {
    ($after:expr, $before:expr, $field:ident, $label:expr) => {
        $after.$field.checked_sub($before.$field).ok_or_else(|| {
            BenchmarkFailure::new(
                "postcondition",
                "counter-regression",
                format!("{} counter {} regressed", $label, stringify!($field)),
            )
        })?
    };
}

pub(crate) fn token_loop_delta(
    before: GpuNativeTokenLoopSnapshot,
    after: GpuNativeTokenLoopSnapshot,
) -> Result<GpuNativeTokenLoopSnapshot, BenchmarkFailure> {
    Ok(GpuNativeTokenLoopSnapshot {
        token_attempts: checked_delta!(after, before, token_attempts, "token-loop"),
        tokens_completed: checked_delta!(after, before, tokens_completed, "token-loop"),
        warm_tokens_completed: checked_delta!(after, before, warm_tokens_completed, "token-loop"),
        residency_miss_attempts: checked_delta!(
            after,
            before,
            residency_miss_attempts,
            "token-loop"
        ),
        replay_attempts: checked_delta!(after, before, replay_attempts, "token-loop"),
        residency_services: checked_delta!(after, before, residency_services, "token-loop"),
        fatal_failures: checked_delta!(after, before, fatal_failures, "token-loop"),
        no_progress_failures: checked_delta!(after, before, no_progress_failures, "token-loop"),
        queue_submissions: checked_delta!(after, before, queue_submissions, "token-loop"),
        boundary_maps: checked_delta!(after, before, boundary_maps, "token-loop"),
        boundary_readbacks: checked_delta!(after, before, boundary_readbacks, "token-loop"),
    })
}

pub(crate) fn recovery_delta(
    before: GpuNativeRecoverySnapshot,
    after: GpuNativeRecoverySnapshot,
) -> Result<GpuNativeRecoverySnapshot, BenchmarkFailure> {
    Ok(GpuNativeRecoverySnapshot {
        resume_attempts: checked_delta!(after, before, resume_attempts, "recovery"),
        recovery_segments: checked_delta!(after, before, recovery_segments, "recovery"),
        checkpoint_captures: checked_delta!(after, before, checkpoint_captures, "recovery"),
        checkpoint_restores: checked_delta!(after, before, checkpoint_restores, "recovery"),
        full_token_replay_attempts: checked_delta!(
            after,
            before,
            full_token_replay_attempts,
            "recovery"
        ),
        layers_encoded: checked_delta!(after, before, layers_encoded, "recovery"),
        attention_layers_reexecuted: checked_delta!(
            after,
            before,
            attention_layers_reexecuted,
            "recovery"
        ),
        expert_layers_reexecuted: checked_delta!(
            after,
            before,
            expert_layers_reexecuted,
            "recovery"
        ),
        invalid_tail_layers_encoded: checked_delta!(
            after,
            before,
            invalid_tail_layers_encoded,
            "recovery"
        ),
        residency_service_us: checked_delta!(after, before, residency_service_us, "recovery"),
        boundary_wait_us: checked_delta!(after, before, boundary_wait_us, "recovery"),
    })
}

pub(crate) fn routed_delta(
    before: RoutedExpertExecutionSnapshot,
    after: RoutedExpertExecutionSnapshot,
) -> Result<RoutedExpertExecutionSnapshot, BenchmarkFailure> {
    Ok(RoutedExpertExecutionSnapshot {
        selected_routed_experts: checked_delta!(
            after,
            before,
            selected_routed_experts,
            "routed-execution"
        ),
        gpu_dispatch_attempts: checked_delta!(
            after,
            before,
            gpu_dispatch_attempts,
            "routed-execution"
        ),
        gpu_dispatch_successes: checked_delta!(
            after,
            before,
            gpu_dispatch_successes,
            "routed-execution"
        ),
        gpu_dispatch_failures: checked_delta!(
            after,
            before,
            gpu_dispatch_failures,
            "routed-execution"
        ),
        cpu_routed_expert_dispatches: checked_delta!(
            after,
            before,
            cpu_routed_expert_dispatches,
            "routed-execution"
        ),
        gpu_cpu_fallbacks: checked_delta!(after, before, gpu_cpu_fallbacks, "routed-execution"),
        degraded_expert_substitutions: checked_delta!(
            after,
            before,
            degraded_expert_substitutions,
            "routed-execution"
        ),
    })
}

pub(crate) fn gpu_io_delta(
    before: GpuExpertIoSnapshot,
    after: GpuExpertIoSnapshot,
) -> Result<GpuExpertIoSnapshot, BenchmarkFailure> {
    Ok(GpuExpertIoSnapshot {
        expert_weight_uploads: checked_delta!(
            after,
            before,
            expert_weight_uploads,
            "gpu-expert-io"
        ),
        expert_weight_upload_bytes: checked_delta!(
            after,
            before,
            expert_weight_upload_bytes,
            "gpu-expert-io"
        ),
        hidden_state_uploads: checked_delta!(after, before, hidden_state_uploads, "gpu-expert-io"),
        hidden_state_upload_bytes: checked_delta!(
            after,
            before,
            hidden_state_upload_bytes,
            "gpu-expert-io"
        ),
        queue_submissions: checked_delta!(after, before, queue_submissions, "gpu-expert-io"),
        map_requests: checked_delta!(after, before, map_requests, "gpu-expert-io"),
        readback_completions: checked_delta!(after, before, readback_completions, "gpu-expert-io"),
        readback_bytes: checked_delta!(after, before, readback_bytes, "gpu-expert-io"),
    })
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct EngineStorageSnapshot {
    pub(crate) ram_hits: u64,
    pub(crate) ram_misses: u64,
    pub(crate) nvme_read_operations: u64,
    pub(crate) nvme_bytes_read: u64,
    pub(crate) prefetch_completed: u64,
    pub(crate) predictor_observations: u64,
    pub(crate) ssd_stall_us: u64,
}

impl EngineStorageSnapshot {
    pub(crate) fn from_runtime(runtime: &crate::BenchRealRuntime) -> Self {
        let report = runtime.engine.report();
        Self {
            ram_hits: report.hits,
            ram_misses: report.misses,
            nvme_read_operations: report.io_count,
            nvme_bytes_read: report.bytes_read,
            prefetch_completed: report.prefetch_completed,
            predictor_observations: report.predictor_observations,
            ssd_stall_us: report.predictive.ssd_stall_us,
        }
    }

    pub(crate) fn checked_delta(self, before: Self) -> Result<Self, BenchmarkFailure> {
        Ok(Self {
            ram_hits: checked_delta!(self, before, ram_hits, "engine-storage"),
            ram_misses: checked_delta!(self, before, ram_misses, "engine-storage"),
            nvme_read_operations: checked_delta!(
                self,
                before,
                nvme_read_operations,
                "engine-storage"
            ),
            nvme_bytes_read: checked_delta!(self, before, nvme_bytes_read, "engine-storage"),
            prefetch_completed: checked_delta!(self, before, prefetch_completed, "engine-storage"),
            predictor_observations: checked_delta!(
                self,
                before,
                predictor_observations,
                "engine-storage"
            ),
            ssd_stall_us: checked_delta!(self, before, ssd_stall_us, "engine-storage"),
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct GpuNativeResidencyDelta {
    pub(crate) vram_hits: u64,
    pub(crate) vram_misses: u64,
    pub(crate) physical_current_hits: u64,
    pub(crate) physical_source_acquisitions: u64,
    pub(crate) logical_admissions_for_physical_misses: u64,
    pub(crate) ram_to_vram_installs: u64,
    pub(crate) physical_evictions: u64,
    pub(crate) physical_reinstalls: u64,
    pub(crate) stale_generation_rejections: u64,
    pub(crate) demand_requests: u64,
    pub(crate) speculative_requests: u64,
    pub(crate) speculative_vram_hits: u64,
    pub(crate) speculative_ram_to_vram_installs: u64,
    pub(crate) speculative_dropped_capacity_or_pressure: u64,
}

pub(crate) fn gpu_native_residency_delta(
    before: &GpuNativeTieredResidencySnapshot,
    after: &GpuNativeTieredResidencySnapshot,
) -> Result<GpuNativeResidencyDelta, BenchmarkFailure> {
    Ok(GpuNativeResidencyDelta {
        vram_hits: checked_delta!(after, before, vram_hits, "gpu-native-residency"),
        vram_misses: checked_delta!(after, before, vram_misses, "gpu-native-residency"),
        physical_current_hits: checked_delta!(
            after,
            before,
            physical_current_hits,
            "gpu-native-residency"
        ),
        physical_source_acquisitions: checked_delta!(
            after,
            before,
            physical_source_acquisitions,
            "gpu-native-residency"
        ),
        logical_admissions_for_physical_misses: checked_delta!(
            after,
            before,
            logical_admissions_for_physical_misses,
            "gpu-native-residency"
        ),
        ram_to_vram_installs: checked_delta!(
            after,
            before,
            ram_to_vram_installs,
            "gpu-native-residency"
        ),
        physical_evictions: checked_delta!(
            after,
            before,
            physical_evictions,
            "gpu-native-residency"
        ),
        physical_reinstalls: checked_delta!(
            after,
            before,
            physical_reinstalls,
            "gpu-native-residency"
        ),
        stale_generation_rejections: checked_delta!(
            after,
            before,
            stale_generation_rejections,
            "gpu-native-residency"
        ),
        demand_requests: checked_delta!(after, before, demand_requests, "gpu-native-residency"),
        speculative_requests: checked_delta!(
            after,
            before,
            speculative_requests,
            "gpu-native-residency"
        ),
        speculative_vram_hits: checked_delta!(
            after,
            before,
            speculative_vram_hits,
            "gpu-native-residency"
        ),
        speculative_ram_to_vram_installs: checked_delta!(
            after,
            before,
            speculative_ram_to_vram_installs,
            "gpu-native-residency"
        ),
        speculative_dropped_capacity_or_pressure: checked_delta!(
            after,
            before,
            speculative_dropped_capacity_or_pressure,
            "gpu-native-residency"
        ),
    })
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RequestSnapshots {
    pub(crate) token_loop_before: GpuNativeTokenLoopSnapshot,
    pub(crate) token_loop_after: GpuNativeTokenLoopSnapshot,
    pub(crate) token_loop_delta: GpuNativeTokenLoopSnapshot,
    pub(crate) token_loop_ratios: CounterRatios,
    pub(crate) recovery_before: GpuNativeRecoverySnapshot,
    pub(crate) recovery_after: GpuNativeRecoverySnapshot,
    pub(crate) recovery_delta: GpuNativeRecoverySnapshot,
    pub(crate) recovery_ratios: RecoveryRatios,
    pub(crate) routed_execution_before: RoutedExpertExecutionSnapshot,
    pub(crate) routed_execution_after: RoutedExpertExecutionSnapshot,
    pub(crate) routed_execution_delta: RoutedExpertExecutionSnapshot,
    pub(crate) runtime_cache_before: RuntimeCacheSnapshot,
    pub(crate) runtime_cache_after: RuntimeCacheSnapshot,
    pub(crate) engine_storage_before: EngineStorageSnapshot,
    pub(crate) engine_storage_after: EngineStorageSnapshot,
    pub(crate) engine_storage_delta: EngineStorageSnapshot,
    pub(crate) gpu_expert_io_before: GpuExpertIoSnapshot,
    pub(crate) gpu_expert_io_after: GpuExpertIoSnapshot,
    pub(crate) gpu_expert_io_delta: GpuExpertIoSnapshot,
    pub(crate) gpu_expert_memory_before: GpuExpertMemorySnapshot,
    pub(crate) gpu_expert_memory_after: GpuExpertMemorySnapshot,
    pub(crate) gpu_native_residency_before: GpuNativeTieredResidencySnapshot,
    pub(crate) gpu_native_residency_after: GpuNativeTieredResidencySnapshot,
    pub(crate) gpu_native_residency_delta: GpuNativeResidencyDelta,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct LatencyStatistics {
    pub(crate) p50_seconds: f64,
    pub(crate) p95_seconds: f64,
    pub(crate) p99_seconds: f64,
    pub(crate) mean_seconds: f64,
    pub(crate) min_seconds: f64,
    pub(crate) max_seconds: f64,
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

impl LatencyStatistics {
    fn from_values(values: &[f64]) -> Result<Self, BenchmarkFailure> {
        if values.is_empty() || values.iter().any(|value| !finite_positive(*value)) {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "invalid-decode-latency",
                "post-TTFT decode latencies must be nonempty, finite, and positive",
            ));
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        Ok(Self {
            p50_seconds: percentile(&sorted, 0.50),
            p95_seconds: percentile(&sorted, 0.95),
            p99_seconds: percentile(&sorted, 0.99),
            mean_seconds: sorted.iter().sum::<f64>() / sorted.len() as f64,
            min_seconds: sorted[0],
            max_seconds: *sorted.last().expect("latencies checked nonempty"),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RunTiming {
    pub(crate) prompt_seconds: f64,
    pub(crate) time_to_first_token_seconds: f64,
    pub(crate) decode_generated_tokens: usize,
    pub(crate) decode_seconds: f64,
    pub(crate) decode_tps: f64,
    pub(crate) end_to_end_seconds: f64,
    pub(crate) end_to_end_generated_tps: f64,
    pub(crate) post_ttft_decode_token_latencies_seconds: Vec<f64>,
    pub(crate) post_ttft_decode_token_latency: LatencyStatistics,
}

impl RunTiming {
    pub(crate) fn from_measurement(
        generated_tokens: usize,
        prompt_seconds: f64,
        time_to_first_token_seconds: f64,
        decode_seconds: f64,
        decode_latencies: Vec<f64>,
    ) -> Result<Self, BenchmarkFailure> {
        let decode_generated_tokens = generated_tokens.saturating_sub(1);
        if generated_tokens < 2
            || !finite_positive(prompt_seconds)
            || !finite_positive(time_to_first_token_seconds)
            || !finite_positive(decode_seconds)
            || decode_latencies.len() != decode_generated_tokens
        {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "invalid-measured-duration",
                format!(
                    "requires at least two generated tokens and finite positive prompt/TTFT/decode durations with one latency per post-TTFT token; generated={generated_tokens} prompt={prompt_seconds} ttft={time_to_first_token_seconds} decode={decode_seconds} latencies={}",
                    decode_latencies.len(),
                ),
            ));
        }
        let end_to_end_seconds = prompt_seconds + decode_seconds;
        if !finite_positive(end_to_end_seconds) {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "invalid-end-to-end-duration",
                "end-to-end duration was nonfinite or nonpositive",
            ));
        }
        Ok(Self {
            prompt_seconds,
            time_to_first_token_seconds,
            decode_generated_tokens,
            decode_seconds,
            decode_tps: decode_generated_tokens as f64 / decode_seconds,
            end_to_end_seconds,
            end_to_end_generated_tps: generated_tokens as f64 / end_to_end_seconds,
            post_ttft_decode_token_latency: LatencyStatistics::from_values(&decode_latencies)?,
            post_ttft_decode_token_latencies_seconds: decode_latencies,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PerRunResult {
    pub(crate) run_index: usize,
    pub(crate) prompt_tokens: usize,
    pub(crate) requested_output_tokens: usize,
    pub(crate) generated_tokens: usize,
    pub(crate) generated_token_ids: Vec<u32>,
    pub(crate) generated_token_ids_sha256: String,
    pub(crate) generated_text_sha256: String,
    pub(crate) timing: RunTiming,
    pub(crate) counters: RequestSnapshots,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct DistributionSummary {
    pub(crate) mean: f64,
    pub(crate) median: f64,
    pub(crate) min: f64,
    pub(crate) max: f64,
}

fn distribution(values: &[f64]) -> Result<DistributionSummary, BenchmarkFailure> {
    if values.is_empty() || values.iter().any(|value| !finite_positive(*value)) {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "invalid-aggregate-input",
            "aggregate inputs must be nonempty, finite, and positive",
        ));
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(DistributionSummary {
        mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
        median: percentile(&sorted, 0.50),
        min: sorted[0],
        max: *sorted.last().expect("distribution checked nonempty"),
    })
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct TtftAggregate {
    pub(crate) mean_seconds: f64,
    pub(crate) median_seconds: f64,
    pub(crate) p95_seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Aggregate {
    pub(crate) decode_tps: DistributionSummary,
    pub(crate) end_to_end_generated_tps: DistributionSummary,
    pub(crate) time_to_first_token: TtftAggregate,
    pub(crate) pooled_post_ttft_decode_token_latency: LatencyStatistics,
    pub(crate) counter_totals: GpuNativeTokenLoopSnapshot,
    pub(crate) counter_ratios: CounterRatios,
    pub(crate) recovery_totals: GpuNativeRecoverySnapshot,
    pub(crate) recovery_ratios: RecoveryRatios,
}

fn add_counter_totals(
    total: &mut GpuNativeTokenLoopSnapshot,
    value: GpuNativeTokenLoopSnapshot,
) -> Result<(), BenchmarkFailure> {
    macro_rules! add {
        ($field:ident) => {
            total.$field = total.$field.checked_add(value.$field).ok_or_else(|| {
                BenchmarkFailure::new(
                    "postcondition",
                    "counter-overflow",
                    format!("aggregate counter {} overflowed", stringify!($field)),
                )
            })?;
        };
    }
    add!(token_attempts);
    add!(tokens_completed);
    add!(warm_tokens_completed);
    add!(residency_miss_attempts);
    add!(replay_attempts);
    add!(residency_services);
    add!(fatal_failures);
    add!(no_progress_failures);
    add!(queue_submissions);
    add!(boundary_maps);
    add!(boundary_readbacks);
    Ok(())
}

fn add_recovery_totals(
    total: &mut GpuNativeRecoverySnapshot,
    value: GpuNativeRecoverySnapshot,
) -> Result<(), BenchmarkFailure> {
    macro_rules! add {
        ($field:ident) => {
            total.$field = total.$field.checked_add(value.$field).ok_or_else(|| {
                BenchmarkFailure::new(
                    "postcondition",
                    "counter-overflow",
                    format!(
                        "aggregate recovery counter {} overflowed",
                        stringify!($field)
                    ),
                )
            })?;
        };
    }
    add!(resume_attempts);
    add!(recovery_segments);
    add!(checkpoint_captures);
    add!(checkpoint_restores);
    add!(full_token_replay_attempts);
    add!(layers_encoded);
    add!(attention_layers_reexecuted);
    add!(expert_layers_reexecuted);
    add!(invalid_tail_layers_encoded);
    add!(residency_service_us);
    add!(boundary_wait_us);
    Ok(())
}

pub(crate) fn aggregate(runs: &[PerRunResult]) -> Result<Aggregate, BenchmarkFailure> {
    if runs.is_empty() {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "missing-measured-runs",
            "no measured run results were retained",
        ));
    }
    let decode_tps = runs
        .iter()
        .map(|run| run.timing.decode_tps)
        .collect::<Vec<_>>();
    let end_to_end = runs
        .iter()
        .map(|run| run.timing.end_to_end_generated_tps)
        .collect::<Vec<_>>();
    let mut ttft = runs
        .iter()
        .map(|run| run.timing.time_to_first_token_seconds)
        .collect::<Vec<_>>();
    ttft.sort_by(f64::total_cmp);
    let mut pooled_latencies = Vec::new();
    let mut counter_totals = GpuNativeTokenLoopSnapshot::default();
    let mut recovery_totals = GpuNativeRecoverySnapshot::default();
    for run in runs {
        pooled_latencies.extend_from_slice(&run.timing.post_ttft_decode_token_latencies_seconds);
        add_counter_totals(&mut counter_totals, run.counters.token_loop_delta)?;
        add_recovery_totals(&mut recovery_totals, run.counters.recovery_delta)?;
    }
    Ok(Aggregate {
        decode_tps: distribution(&decode_tps)?,
        end_to_end_generated_tps: distribution(&end_to_end)?,
        time_to_first_token: TtftAggregate {
            mean_seconds: ttft.iter().sum::<f64>() / ttft.len() as f64,
            median_seconds: percentile(&ttft, 0.50),
            p95_seconds: percentile(&ttft, 0.95),
        },
        pooled_post_ttft_decode_token_latency: LatencyStatistics::from_values(&pooled_latencies)?,
        counter_ratios: CounterRatios::from_delta(counter_totals)?,
        recovery_ratios: RecoveryRatios::from_delta(
            recovery_totals,
            counter_totals.tokens_completed,
        )?,
        recovery_totals,
        counter_totals,
    })
}

fn retain_measured_result(
    retained: &mut Vec<PerRunResult>,
    result: Result<PerRunResult, BenchmarkFailure>,
) -> Result<(), BenchmarkFailure> {
    match result {
        Ok(run) => {
            retained.push(run);
            Ok(())
        }
        Err(failure) => Err(failure),
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeConstructionTiming {
    pub(crate) phase: String,
    pub(crate) run_index: Option<usize>,
    pub(crate) seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeShutdown {
    pub(crate) phase: String,
    pub(crate) run_index: Option<usize>,
    pub(crate) evidence: BackgroundShutdownEvidence,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BenchmarkReport {
    pub(crate) schema: &'static str,
    pub(crate) mode: &'static str,
    pub(crate) optimization: &'static str,
    pub(crate) immediate_comparison_commit: &'static str,
    pub(crate) pr1b_a_experiment_commit: &'static str,
    pub(crate) pr1b_a_experiment_report_sha256: &'static str,
    pub(crate) original_baseline_commit: &'static str,
    pub(crate) benchmark_complete: bool,
    pub(crate) failure: Option<BenchmarkFailure>,
    pub(crate) qualification_pass: bool,
    pub(crate) correctness_qualification_pending: bool,
    pub(crate) provenance: BenchmarkProvenance,
    pub(crate) hardware: Option<GpuDeviceIdentity>,
    pub(crate) model_identity: ModelIdentityEvidence,
    pub(crate) model_load: Option<ModelLoadEvidence>,
    pub(crate) request: RequestEvidence,
    pub(crate) cache_reset: crate::BenchRealCacheReset,
    pub(crate) warmup_runs: usize,
    pub(crate) warmup_runs_completed: usize,
    pub(crate) measured_runs: usize,
    pub(crate) runtime_constructions: Vec<RuntimeConstructionTiming>,
    pub(crate) runtime_shutdowns: Vec<RuntimeShutdown>,
    pub(crate) per_run_results: Vec<PerRunResult>,
    pub(crate) aggregate: Option<Aggregate>,
    pub(crate) runtime_contract: Option<RuntimeContractEvidence>,
    pub(crate) production_configuration: ProductionConfiguration,
    pub(crate) production_semantics: ProductionSemantics,
}

impl BenchmarkReport {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        provenance: BenchmarkProvenance,
        model_identity: ModelIdentityEvidence,
        request: RequestEvidence,
        cache_reset: crate::BenchRealCacheReset,
        warmup_runs: usize,
        measured_runs: usize,
        production_configuration: ProductionConfiguration,
    ) -> Self {
        Self {
            schema: SCHEMA,
            mode: MODE,
            optimization: OPTIMIZATION,
            immediate_comparison_commit: IMMEDIATE_COMPARISON_COMMIT,
            pr1b_a_experiment_commit: PR1B_A_EXPERIMENT_COMMIT,
            pr1b_a_experiment_report_sha256: PR1B_A_EXPERIMENT_REPORT_SHA256,
            original_baseline_commit: ORIGINAL_BASELINE_COMMIT,
            benchmark_complete: false,
            failure: None,
            qualification_pass: false,
            correctness_qualification_pending: true,
            provenance,
            hardware: None,
            model_identity,
            model_load: None,
            request,
            cache_reset,
            warmup_runs,
            warmup_runs_completed: 0,
            measured_runs,
            runtime_constructions: Vec::new(),
            runtime_shutdowns: Vec::new(),
            per_run_results: Vec::new(),
            aggregate: None,
            runtime_contract: None,
            production_configuration,
            production_semantics: ProductionSemantics::physical_source_of_truth_pr1bb(),
        }
    }

    pub(crate) fn finish(&mut self) -> Result<(), BenchmarkFailure> {
        if self.per_run_results.len() != self.measured_runs {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "incomplete-measured-run-set",
                format!(
                    "retained {} of {} requested measured runs",
                    self.per_run_results.len(),
                    self.measured_runs
                ),
            ));
        }
        if self.warmup_runs_completed != self.warmup_runs {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "incomplete-warmup-set",
                format!(
                    "completed {} of {} requested warmups",
                    self.warmup_runs_completed, self.warmup_runs
                ),
            ));
        }
        if self.hardware.is_none() || self.model_load.is_none() || self.runtime_contract.is_none() {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "runtime-evidence-incomplete",
                "hardware, model-load, or authoritative runtime-contract evidence is missing",
            ));
        }
        self.aggregate = Some(aggregate(&self.per_run_results)?);
        self.benchmark_complete = true;
        self.failure = None;
        Ok(())
    }

    pub(crate) fn fail(&mut self, failure: BenchmarkFailure) {
        self.benchmark_complete = false;
        self.aggregate = None;
        self.failure = Some(failure);
    }
}

pub(crate) fn expected_runtime_constructions(
    cache_reset: crate::BenchRealCacheReset,
    warmup_runs: usize,
    measured_runs: usize,
) -> usize {
    match cache_reset {
        crate::BenchRealCacheReset::Keep => 1,
        crate::BenchRealCacheReset::FreshRuntime => warmup_runs.saturating_add(measured_runs),
    }
}

pub(crate) fn validate_request_postconditions(
    prompt_tokens: usize,
    requested_output_tokens: usize,
    generated_tokens: usize,
    token_loop: GpuNativeTokenLoopSnapshot,
    recovery: GpuNativeRecoverySnapshot,
    routed: RoutedExpertExecutionSnapshot,
) -> Result<(), BenchmarkFailure> {
    let expected_completed = prompt_tokens
        .checked_add(requested_output_tokens)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| {
            BenchmarkFailure::new(
                "postcondition",
                "completed-position-overflow",
                "expected completed-position count overflowed",
            )
        })?;
    if generated_tokens != requested_output_tokens {
        return Err(BenchmarkFailure::new(
            "inference",
            "incomplete-generation",
            format!("generated {generated_tokens} of {requested_output_tokens} requested tokens"),
        ));
    }
    if token_loop.tokens_completed != expected_completed as u64
        || token_loop.fatal_failures != 0
        || token_loop.no_progress_failures != 0
    {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "token-loop-postcondition",
            format!(
                "expected {expected_completed} completed positions and zero fatal/no-progress failures; observed {token_loop:?}"
            ),
        ));
    }
    if token_loop.replay_attempts != 0 || recovery.full_token_replay_attempts != 0 {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "full-token-replay-observed",
            format!(
                "PR1 requires zero full-token restarts; token_loop.replay_attempts={} recovery.full_token_replay_attempts={}",
                token_loop.replay_attempts, recovery.full_token_replay_attempts,
            ),
        ));
    }
    if token_loop.residency_miss_attempts != 0 && recovery.resume_attempts == 0 {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "missing-resume-evidence",
            format!(
                "recoverable misses require resumable execution evidence; misses={} recovery={recovery:?}",
                token_loop.residency_miss_attempts,
            ),
        ));
    }
    if routed.cpu_routed_expert_dispatches != 0
        || routed.gpu_cpu_fallbacks != 0
        || routed.degraded_expert_substitutions != 0
    {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "fallback-or-degradation",
            format!("CPU fallback or degraded expert substitution occurred: {routed:?}"),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OrdinaryStep {
    token_id: u32,
    position: usize,
    sample: bool,
}

fn next_ordinary_step(
    prompt_ids: &[u32],
    generated_ids: &[u32],
    completed_positions: usize,
) -> Option<OrdinaryStep> {
    if completed_positions < prompt_ids.len() {
        return Some(OrdinaryStep {
            token_id: prompt_ids[completed_positions],
            position: completed_positions,
            sample: completed_positions + 1 == prompt_ids.len(),
        });
    }
    generated_ids.last().copied().map(|token_id| OrdinaryStep {
        token_id,
        position: completed_positions,
        sample: true,
    })
}

#[derive(Clone, Debug)]
struct RequestSnapshotStart {
    token_loop: GpuNativeTokenLoopSnapshot,
    recovery: GpuNativeRecoverySnapshot,
    routed: RoutedExpertExecutionSnapshot,
    runtime_cache: RuntimeCacheSnapshot,
    engine_storage: EngineStorageSnapshot,
    gpu_expert_io: GpuExpertIoSnapshot,
    gpu_expert_memory: GpuExpertMemorySnapshot,
    gpu_native_residency: GpuNativeTieredResidencySnapshot,
}

impl RequestSnapshotStart {
    fn capture(runtime: &crate::BenchRealRuntime) -> Result<Self, BenchmarkFailure> {
        let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
            BenchmarkFailure::new(
                "startup",
                "missing-gpu-native-token-loop",
                "request snapshot could not find gpu_native_token_loop",
            )
        })?;
        Ok(Self {
            token_loop: token_loop.snapshot(),
            recovery: token_loop.recovery_snapshot(),
            routed: runtime.engine.routed_expert_execution_snapshot(),
            runtime_cache: crate::greedy_parity_runtime_cache_snapshot(runtime),
            engine_storage: EngineStorageSnapshot::from_runtime(runtime),
            gpu_expert_io: runtime.engine.gpu_expert_io_snapshot().ok_or_else(|| {
                BenchmarkFailure::new(
                    "startup",
                    "missing-gpu-io-snapshot",
                    "authoritative runtime did not expose GPU expert I/O snapshot",
                )
            })?,
            gpu_expert_memory: runtime.engine.gpu_expert_memory_snapshot().ok_or_else(|| {
                BenchmarkFailure::new(
                    "startup",
                    "missing-gpu-memory-snapshot",
                    "authoritative runtime did not expose GPU expert memory snapshot",
                )
            })?,
            gpu_native_residency: runtime.engine.gpu_native_residency_snapshot().ok_or_else(
                || {
                    BenchmarkFailure::new(
                        "startup",
                        "missing-gpu-native-residency-snapshot",
                        "authoritative runtime did not expose GPU-native residency snapshot",
                    )
                },
            )?,
        })
    }

    fn finish(
        self,
        runtime: &crate::BenchRealRuntime,
    ) -> Result<RequestSnapshots, BenchmarkFailure> {
        let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
            BenchmarkFailure::new(
                "postcondition",
                "missing-gpu-native-token-loop",
                "gpu_native_token_loop disappeared after request",
            )
        })?;
        let token_loop_after = token_loop.snapshot();
        let recovery_after = token_loop.recovery_snapshot();
        let routed_after = runtime.engine.routed_expert_execution_snapshot();
        let engine_storage_after = EngineStorageSnapshot::from_runtime(runtime);
        let gpu_expert_io_after = runtime.engine.gpu_expert_io_snapshot().ok_or_else(|| {
            BenchmarkFailure::new(
                "postcondition",
                "missing-gpu-io-snapshot",
                "GPU expert I/O snapshot disappeared after request",
            )
        })?;
        let token_loop_delta = token_loop_delta(self.token_loop, token_loop_after)?;
        let recovery_delta = recovery_delta(self.recovery, recovery_after)?;
        let routed_execution_delta = routed_delta(self.routed, routed_after)?;
        let gpu_native_residency_after = runtime
            .engine
            .gpu_native_residency_snapshot()
            .ok_or_else(|| {
                BenchmarkFailure::new(
                    "postcondition",
                    "missing-gpu-native-residency-snapshot",
                    "GPU-native residency snapshot disappeared after request",
                )
            })?;
        let gpu_native_residency_delta =
            gpu_native_residency_delta(&self.gpu_native_residency, &gpu_native_residency_after)?;
        Ok(RequestSnapshots {
            token_loop_before: self.token_loop,
            token_loop_after,
            token_loop_delta,
            token_loop_ratios: CounterRatios::from_delta(token_loop_delta)?,
            recovery_before: self.recovery,
            recovery_after,
            recovery_delta,
            recovery_ratios: RecoveryRatios::from_delta(
                recovery_delta,
                token_loop_delta.tokens_completed,
            )?,
            routed_execution_before: self.routed,
            routed_execution_after: routed_after,
            routed_execution_delta,
            runtime_cache_before: self.runtime_cache,
            runtime_cache_after: crate::greedy_parity_runtime_cache_snapshot(runtime),
            engine_storage_before: self.engine_storage,
            engine_storage_after,
            engine_storage_delta: engine_storage_after.checked_delta(self.engine_storage)?,
            gpu_expert_io_before: self.gpu_expert_io,
            gpu_expert_io_after,
            gpu_expert_io_delta: gpu_io_delta(self.gpu_expert_io, gpu_expert_io_after)?,
            gpu_expert_memory_before: self.gpu_expert_memory,
            gpu_expert_memory_after: runtime.engine.gpu_expert_memory_snapshot().ok_or_else(
                || {
                    BenchmarkFailure::new(
                        "postcondition",
                        "missing-gpu-memory-snapshot",
                        "GPU expert memory snapshot disappeared after request",
                    )
                },
            )?,
            gpu_native_residency_before: self.gpu_native_residency,
            gpu_native_residency_after,
            gpu_native_residency_delta,
        })
    }
}

pub(crate) async fn execute_request(
    runtime: &crate::BenchRealRuntime,
    prompt_ids: &[u32],
    requested_output_tokens: usize,
    run_index: usize,
) -> Result<PerRunResult, Box<dyn std::error::Error>> {
    if prompt_ids.is_empty() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "empty-prompt-tokenization",
            "prompt encoded to zero tokens",
        )
        .into());
    }
    if requested_output_tokens < 2 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "insufficient-output-tokens",
            "decode throughput requires --output-tokens >= 2",
        )
        .into());
    }
    let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
        BenchmarkFailure::new(
            "startup",
            "missing-gpu-native-token-loop",
            "runtime.gpu_native_token_loop is None",
        )
    })?;
    let required =
        crate::gpu_native_token_loop::GpuNativeTokenLoop::calculate_required_context_len(
            0,
            prompt_ids.len(),
            requested_output_tokens,
        )
        .ok_or_else(|| {
            BenchmarkFailure::new(
                "preflight",
                "context-length-overflow",
                "prompt plus output token count overflowed context arithmetic",
            )
        })?;
    if required > token_loop.max_seq_len() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "context-limit-exceeded",
            format!(
                "request requires {required} positions but gpu_native_max_seq_len is {}",
                token_loop.max_seq_len()
            ),
        )
        .into());
    }

    let mut request = token_loop.create_request_state()?;
    let snapshots = RequestSnapshotStart::capture(runtime)?;
    let prompt_started = std::time::Instant::now();
    let mut completed_positions = 0usize;
    let mut generated_ids = Vec::with_capacity(requested_output_tokens);
    while completed_positions < prompt_ids.len() {
        let step = next_ordinary_step(prompt_ids, &generated_ids, completed_positions)
            .expect("prompt step must exist");
        let sampled = token_loop
            .step_token(
                &runtime.engine,
                &mut request,
                step.token_id,
                step.position,
                step.sample,
            )
            .await?;
        completed_positions += 1;
        if step.sample {
            generated_ids.push(sampled.ok_or_else(|| {
                BenchmarkFailure::new(
                    "inference",
                    "missing-first-generated-token",
                    "final prompt position produced no sampled token",
                )
            })?);
        } else if sampled.is_some() {
            return Err(BenchmarkFailure::new(
                "inference",
                "unexpected-prefix-sample",
                "prompt prefix position unexpectedly produced a sampled token",
            )
            .into());
        }
    }
    let prompt_seconds = prompt_started.elapsed().as_secs_f64();
    let time_to_first_token_seconds = prompt_seconds;

    let decode_started = std::time::Instant::now();
    let mut decode_latencies = Vec::with_capacity(requested_output_tokens - 1);
    while generated_ids.len() < requested_output_tokens {
        let step = next_ordinary_step(prompt_ids, &generated_ids, completed_positions)
            .expect("decode step must exist after first sample");
        let step_started = std::time::Instant::now();
        let sampled = token_loop
            .step_token(
                &runtime.engine,
                &mut request,
                step.token_id,
                step.position,
                step.sample,
            )
            .await?
            .ok_or_else(|| {
                BenchmarkFailure::new(
                    "inference",
                    "missing-decode-token",
                    format!(
                        "decode position {} produced no sampled token",
                        step.position
                    ),
                )
            })?;
        decode_latencies.push(step_started.elapsed().as_secs_f64());
        generated_ids.push(sampled);
        completed_positions += 1;
    }
    let decode_seconds = decode_started.elapsed().as_secs_f64();
    let counters = snapshots.finish(runtime)?;
    validate_request_postconditions(
        prompt_ids.len(),
        requested_output_tokens,
        generated_ids.len(),
        counters.token_loop_delta,
        counters.recovery_delta,
        counters.routed_execution_delta,
    )?;
    let output_text = runtime.tokenizer.decode(&generated_ids)?;
    Ok(PerRunResult {
        run_index,
        prompt_tokens: prompt_ids.len(),
        requested_output_tokens,
        generated_tokens: generated_ids.len(),
        generated_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&generated_ids),
        generated_text_sha256: crate::greedy_parity::sha256_hex(output_text.as_bytes()),
        generated_token_ids: generated_ids,
        timing: RunTiming::from_measurement(
            requested_output_tokens,
            prompt_seconds,
            time_to_first_token_seconds,
            decode_seconds,
            decode_latencies,
        )?,
        counters,
    })
}

pub(crate) fn is_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub(crate) fn validate_preflight_provenance(
    build: &BuildProvenance,
) -> Result<(), BenchmarkFailure> {
    if build.dirty != Some(false) || build.git_sha.as_deref().is_none_or(|sha| !is_hex(sha, 40)) {
        return Err(BenchmarkFailure::new(
            "preflight",
            "provenance-unavailable",
            format!(
                "requires clean embedded full Git SHA and dirty=false; observed git_sha={:?} dirty={:?}",
                build.git_sha, build.dirty
            ),
        ));
    }
    Ok(())
}

pub(crate) fn validate_artifacts(
    artifacts: &QualificationArtifacts,
    errors: &[String],
) -> Result<(), BenchmarkFailure> {
    if !errors.is_empty()
        || artifacts.config.is_none()
        || artifacts.tokenizer.is_none()
        || artifacts.expert_metadata.is_none()
        || artifacts.dense_weights_directory.is_none()
    {
        return Err(BenchmarkFailure::new(
            "preflight",
            "artifact-provenance-unavailable",
            format!(
                "config, tokenizer, expert metadata, and dense weights directory identities are mandatory; errors={errors:?} artifacts={artifacts:?}"
            ),
        ));
    }
    Ok(())
}

pub(crate) fn validate_expert_metadata(
    metadata: &ExpertMetadataEvidence,
) -> Result<(), BenchmarkFailure> {
    if metadata.dtype.as_deref() != Some("q4_0")
        || metadata.q4_0_layout.as_deref() != Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1)
        || metadata.explicitly_synthetic
    {
        return Err(BenchmarkFailure::new(
            "preflight",
            "invalid-expert-metadata",
            format!("requires canonical nonsynthetic Q4_0 expert metadata; observed {metadata:?}"),
        ));
    }
    Ok(())
}

fn emit_report(
    report: &BenchmarkReport,
    report_out: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write as _;

    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    if let Some(path) = report_out {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, json)?;
        eprintln!(
            "GPU-native real-model benchmark report written to {}",
            path.display()
        );
    } else {
        std::io::stdout().write_all(&json)?;
    }
    Ok(())
}

pub(crate) async fn construct_runtime(
    spec: &crate::ResolvedRealCliSpec,
    tokenizer: std::sync::Arc<crate::tokenizer::Tokenizer>,
    phase: &str,
    run_index: Option<usize>,
    report: &mut BenchmarkReport,
) -> Result<crate::BenchRealRuntime, BenchmarkFailure> {
    let started = std::time::Instant::now();
    let runtime = crate::build_isolated_greedy_runtime(
        spec,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
        tokenizer,
    )
    .await
    .map_err(|error| {
        BenchmarkFailure::new("startup", "runtime-construction-failed", error.to_string())
    })?;
    report
        .runtime_constructions
        .push(RuntimeConstructionTiming {
            phase: phase.to_string(),
            run_index,
            seconds: started.elapsed().as_secs_f64(),
        });
    Ok(runtime)
}

pub(crate) fn validate_and_record_runtime(
    runtime: &crate::BenchRealRuntime,
    resolved_config_sha256: &str,
    expected_adapter_name: &str,
    report: &mut BenchmarkReport,
) -> Result<(), BenchmarkFailure> {
    let observed_config_sha256 = crate::resolved_real_runtime_identity_sha256(
        &runtime.cfg,
        runtime.model.config.architecture,
        runtime.model.config.first_k_dense_replace,
        &runtime.model.config.advanced,
    )
    .map_err(|error| {
        BenchmarkFailure::new(
            "startup",
            "runtime-config-identity-unavailable",
            error.to_string(),
        )
    })?;
    if observed_config_sha256 != resolved_config_sha256 {
        return Err(BenchmarkFailure::new(
            "startup",
            "runtime-config-identity-drift",
            format!(
                "runtime identity {observed_config_sha256} differs from preflight identity {resolved_config_sha256}"
            ),
        ));
    }
    let model_load = crate::greedy_parity_model_load(runtime);
    let input = RuntimeContractInput {
        real_transformer_enabled: runtime.cfg.real_transformer.enabled,
        real_transformer_gpu_native: runtime.cfg.real_transformer.gpu_native,
        compute_offload: runtime.cfg.real_transformer.compute_offload,
        legacy_execution_plan: runtime.engine.execution_context().plan().into(),
        token_loop_geometry: runtime
            .gpu_native_token_loop
            .as_ref()
            .map(|token_loop| token_loop.model_geometry()),
        authoritative_device: runtime.engine.gpu_device_identity(),
        model_load: model_load.clone(),
        routed_failure_policy: runtime.engine.routed_expert_gpu_failure_policy(),
    };
    let (contract, device) = validate_runtime_contract(&input, expected_adapter_name)?;
    let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
        BenchmarkFailure::new(
            "startup",
            "missing-gpu-native-token-loop",
            "runtime contract passed without gpu_native_token_loop",
        )
    })?;
    if token_loop.snapshot() != GpuNativeTokenLoopSnapshot::default() {
        return Err(BenchmarkFailure::new(
            "startup",
            "nonzero-initial-token-loop-counters",
            format!(
                "fresh isolated runtime token-loop counters were not zero: {:?}",
                token_loop.snapshot()
            ),
        ));
    }
    if token_loop.recovery_snapshot() != GpuNativeRecoverySnapshot::default() {
        return Err(BenchmarkFailure::new(
            "startup",
            "nonzero-initial-recovery-counters",
            format!(
                "fresh isolated runtime recovery counters were not zero: {:?}",
                token_loop.recovery_snapshot()
            ),
        ));
    }
    if runtime.engine.routed_expert_execution_snapshot() != RoutedExpertExecutionSnapshot::default()
    {
        return Err(BenchmarkFailure::new(
            "startup",
            "nonzero-initial-routed-counters",
            "fresh isolated runtime routed-execution counters were not zero",
        ));
    }
    if let Some(previous) = &report.hardware {
        if previous != &device {
            return Err(BenchmarkFailure::new(
                "startup",
                "adapter-identity-drift",
                format!(
                    "fresh runtime adapter identity drifted: before={previous:?} after={device:?}"
                ),
            ));
        }
    } else {
        report.hardware = Some(device);
    }
    if let Some(previous) = &report.model_load {
        if previous != &model_load {
            return Err(BenchmarkFailure::new(
                "startup",
                "model-load-identity-drift",
                format!(
                    "fresh runtime model load evidence drifted: before={previous:?} after={model_load:?}"
                ),
            ));
        }
    } else {
        report.model_load = Some(model_load);
    }
    if report.runtime_contract.is_none() {
        report.runtime_contract = Some(contract);
    }
    Ok(())
}

pub(crate) async fn shutdown_runtime(
    runtime: crate::BenchRealRuntime,
    phase: &str,
    run_index: Option<usize>,
    report: &mut BenchmarkReport,
) -> Result<(), BenchmarkFailure> {
    let evidence = runtime.shutdown_isolated().await.map_err(|error| {
        BenchmarkFailure::new(
            "postcondition",
            "runtime-shutdown-failed",
            error.to_string(),
        )
    })?;
    if !evidence.controlled_shutdown_requested || !evidence.all_runtime_resources_released {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "runtime-shutdown-incomplete",
            format!("controlled isolated runtime shutdown was incomplete: {evidence:?}"),
        ));
    }
    report.runtime_shutdowns.push(RuntimeShutdown {
        phase: phase.to_string(),
        run_index,
        evidence,
    });
    Ok(())
}

async fn execute_with_fresh_runtime(
    spec: &crate::ResolvedRealCliSpec,
    tokenizer: std::sync::Arc<crate::tokenizer::Tokenizer>,
    prompt_ids: &[u32],
    output_tokens: usize,
    phase: &str,
    run_index: usize,
    expected_adapter_name: &str,
    resolved_config_sha256: &str,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
    report: &mut BenchmarkReport,
) -> Result<PerRunResult, BenchmarkFailure> {
    let runtime = construct_runtime(spec, tokenizer, phase, Some(run_index), report).await?;
    let validation = validate_and_record_runtime(
        &runtime,
        resolved_config_sha256,
        expected_adapter_name,
        report,
    );
    if let Err(validation_error) = validation {
        let shutdown = shutdown_runtime(runtime, phase, Some(run_index), report).await;
        return match shutdown {
            Ok(()) => Err(validation_error),
            Err(shutdown_error) => Err(BenchmarkFailure::new(
                "postcondition",
                "runtime-validation-and-shutdown-failed",
                format!("{validation_error}; {shutdown_error}"),
            )),
        };
    }
    let request_result = crate::with_progress_timeout(
        format!("bench-gpu-native-real {phase} run {run_index}"),
        watchdog,
        execute_request(&runtime, prompt_ids, output_tokens, run_index),
    )
    .await
    .map_err(|error| {
        BenchmarkFailure::new(
            "inference",
            if phase == "measured" {
                "measured-request-failed"
            } else {
                "warmup-request-failed"
            },
            error.to_string(),
        )
    });
    let shutdown_result = shutdown_runtime(runtime, phase, Some(run_index), report).await;
    match (request_result, shutdown_result) {
        (Ok(result), Ok(())) => Ok(result),
        (Err(request_error), Ok(())) => Err(request_error),
        (Ok(_), Err(shutdown_error)) => Err(shutdown_error),
        (Err(request_error), Err(shutdown_error)) => Err(BenchmarkFailure::new(
            "postcondition",
            "request-and-shutdown-failed",
            format!("{request_error}; {shutdown_error}"),
        )),
    }
}

async fn execute_keep_schedule(
    args: &CommandArgs,
    spec: &crate::ResolvedRealCliSpec,
    tokenizer: std::sync::Arc<crate::tokenizer::Tokenizer>,
    prompt_ids: &[u32],
    output_tokens: usize,
    resolved_config_sha256: &str,
    report: &mut BenchmarkReport,
) -> Result<(), BenchmarkFailure> {
    let runtime = construct_runtime(spec, tokenizer, "shared", None, report).await?;
    let validation = validate_and_record_runtime(
        &runtime,
        resolved_config_sha256,
        &args.expected_adapter_name,
        report,
    );
    if let Err(validation_error) = validation {
        let shutdown = shutdown_runtime(runtime, "shared", None, report).await;
        return match shutdown {
            Ok(()) => Err(validation_error),
            Err(shutdown_error) => Err(BenchmarkFailure::new(
                "postcondition",
                "runtime-validation-and-shutdown-failed",
                format!("{validation_error}; {shutdown_error}"),
            )),
        };
    }

    let execution = async {
        for index in 0..args.warmup_runs {
            crate::with_progress_timeout(
                format!("bench-gpu-native-real warmup run {index}"),
                args.progress_watchdog,
                execute_request(&runtime, prompt_ids, output_tokens, index),
            )
            .await
            .map_err(|error| {
                BenchmarkFailure::new("inference", "warmup-request-failed", error.to_string())
            })?;
            report.warmup_runs_completed += 1;
        }
        for index in 0..args.measured_runs {
            let measured = crate::with_progress_timeout(
                format!("bench-gpu-native-real measured run {index}"),
                args.progress_watchdog,
                execute_request(&runtime, prompt_ids, output_tokens, index),
            )
            .await
            .map_err(|error| {
                BenchmarkFailure::new("inference", "measured-request-failed", error.to_string())
            });
            retain_measured_result(&mut report.per_run_results, measured)?;
        }
        Ok::<(), BenchmarkFailure>(())
    }
    .await;
    let shutdown = shutdown_runtime(runtime, "shared", None, report).await;
    match (execution, shutdown) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(execution_error), Ok(())) => Err(execution_error),
        (Ok(()), Err(shutdown_error)) => Err(shutdown_error),
        (Err(execution_error), Err(shutdown_error)) => Err(BenchmarkFailure::new(
            "postcondition",
            "execution-and-shutdown-failed",
            format!("{execution_error}; {shutdown_error}"),
        )),
    }
}

async fn execute_fresh_schedule(
    args: &CommandArgs,
    spec: &crate::ResolvedRealCliSpec,
    tokenizer: std::sync::Arc<crate::tokenizer::Tokenizer>,
    prompt_ids: &[u32],
    output_tokens: usize,
    resolved_config_sha256: &str,
    report: &mut BenchmarkReport,
) -> Result<(), BenchmarkFailure> {
    for index in 0..args.warmup_runs {
        let _ = execute_with_fresh_runtime(
            spec,
            tokenizer.clone(),
            prompt_ids,
            output_tokens,
            "warmup",
            index,
            &args.expected_adapter_name,
            resolved_config_sha256,
            args.progress_watchdog,
            report,
        )
        .await?;
        report.warmup_runs_completed += 1;
    }
    for index in 0..args.measured_runs {
        let measured = execute_with_fresh_runtime(
            spec,
            tokenizer.clone(),
            prompt_ids,
            output_tokens,
            "measured",
            index,
            &args.expected_adapter_name,
            resolved_config_sha256,
            args.progress_watchdog,
            report,
        )
        .await;
        retain_measured_result(&mut report.per_run_results, measured)?;
    }
    Ok(())
}

pub(crate) async fn run_command(args: CommandArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.greedy {
        return Err(BenchmarkFailure::new(
            "preflight",
            "greedy-required",
            "bench-gpu-native-real requires the explicit --greedy flag",
        )
        .into());
    }
    if args.measured_runs == 0 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "measured-runs-required",
            "bench-gpu-native-real requires --measured-runs > 0",
        )
        .into());
    }
    if args.expected_adapter_name.trim().is_empty() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "missing-expected-adapter",
            "bench-gpu-native-real requires --expected-adapter-name",
        )
        .into());
    }
    let input = crate::load_real_cli_request_input(
        "bench-gpu-native-real",
        args.prompt.as_ref(),
        args.request_json.as_deref(),
        args.output_tokens,
    )?;
    if input.output_tokens < 2 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "insufficient-output-tokens",
            "bench-gpu-native-real requires --output-tokens >= 2",
        )
        .into());
    }

    let build = BuildProvenance::embedded();
    validate_preflight_provenance(&build)?;
    let cfg = crate::config::Config::from_file(&args.config)?;
    validate_source_config(&cfg)?;
    let (artifacts, artifact_errors) = crate::qualification_artifacts(&args.config, &cfg);
    validate_artifacts(&artifacts, &artifact_errors)?;
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| {
                BenchmarkFailure::new("preflight", "expert-metadata-unavailable", error)
            })?;
    validate_expert_metadata(&expert_metadata)?;

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
    // Tokenize once and retain this exact caller-owned vector for every
    // warmup and measured request across either cache policy.
    let prompt_ids = tokenizer.encode(&input.prompt)?;
    if prompt_ids.is_empty() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "empty-prompt-tokenization",
            "prompt encoded to zero tokens",
        )
        .into());
    }
    let (executable, executable_sha256) = crate::current_executable_identity()?;
    let executable_canonical_path = std::fs::canonicalize(&executable)
        .map_err(|error| {
            BenchmarkFailure::new(
                "preflight",
                "executable-provenance-unavailable",
                format!("failed to canonicalize {}: {error}", executable.display()),
            )
        })?
        .display()
        .to_string();
    if !is_hex(&executable_sha256, 64) || !is_hex(&resolved_config_sha256, 64) {
        return Err(BenchmarkFailure::new(
            "preflight",
            "provenance-unavailable",
            "executable or resolved-config SHA256 was unavailable",
        )
        .into());
    }

    let production_configuration =
        ProductionConfiguration::from_config(&spec.cfg, &expert_metadata);
    let mut report = BenchmarkReport::new(
        BenchmarkProvenance {
            build,
            executable_canonical_path,
            executable_sha256,
            resolved_config_sha256: resolved_config_sha256.clone(),
            artifacts,
            expert_metadata,
        },
        model_identity,
        RequestEvidence {
            prompt_sha256: crate::greedy_parity::sha256_hex(input.prompt.as_bytes()),
            prompt_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&prompt_ids),
            prompt_token_count: prompt_ids.len(),
            requested_output_tokens: input.output_tokens,
            greedy: true,
        },
        args.cache_reset,
        args.warmup_runs,
        args.measured_runs,
        production_configuration,
    );

    let execution = match args.cache_reset {
        crate::BenchRealCacheReset::Keep => {
            execute_keep_schedule(
                &args,
                &spec,
                tokenizer,
                &prompt_ids,
                input.output_tokens,
                &resolved_config_sha256,
                &mut report,
            )
            .await
        }
        crate::BenchRealCacheReset::FreshRuntime => {
            execute_fresh_schedule(
                &args,
                &spec,
                tokenizer,
                &prompt_ids,
                input.output_tokens,
                &resolved_config_sha256,
                &mut report,
            )
            .await
        }
    };

    let completion = execution.and_then(|()| {
        let expected =
            expected_runtime_constructions(args.cache_reset, args.warmup_runs, args.measured_runs);
        if report.runtime_constructions.len() != expected {
            return Err(BenchmarkFailure::new(
                "postcondition",
                "cache-reset-semantics-drift",
                format!(
                    "cache mode {:?} constructed {} runtimes, expected {expected}",
                    args.cache_reset,
                    report.runtime_constructions.len()
                ),
            ));
        }
        report.finish()
    });

    match completion {
        Ok(()) => emit_report(&report, args.report_out.as_deref()),
        Err(failure) => {
            let summary = failure.to_string();
            report.fail(failure);
            emit_report(&report, args.report_out.as_deref()).map_err(|emit_error| {
                format!(
                    "{summary}; additionally failed to emit benchmark failure report: {emit_error}"
                )
            })?;
            Err(summary.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l4_device() -> GpuDeviceIdentity {
        GpuDeviceIdentity {
            name: "NVIDIA L4".into(),
            vendor_id: 0x10de,
            device_id: 0x27b8,
            device_type: "DiscreteGpu".into(),
            wgpu_backend: "vulkan".into(),
            driver: "test".into(),
            driver_info: "test".into(),
            compute_plane: "wgpu-vulkan".into(),
            software_adapter: false,
        }
    }

    fn qwen_geometry() -> GpuNativeModelGeometry {
        GpuNativeModelGeometry {
            num_layers: 48,
            d_model: 2048,
            d_ff: 768,
            num_experts: 128,
            top_k: 8,
            num_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            rope_dim: 128,
            vocab_size: 151_936,
            max_seq_len: 4096,
            rms_eps: 1e-6,
            rope_base: 10_000.0,
        }
    }

    fn legacy_plan() -> ExecutionPlanEvidence {
        ExecutionPlanEvidence {
            context_id: "test".into(),
            requested: "gpu".into(),
            resolved: "gpu".into(),
            embeddings: "cpu".into(),
            lm_head: "cpu".into(),
            dense_projections: "cpu".into(),
            attention: "gpu".into(),
            kv: "gpu".into(),
            router: "cpu".into(),
            routed_experts: "cpu".into(),
            routed_expert_dtype: "q4_0".into(),
            fallback_occurred: false,
            reason: None,
        }
    }

    fn strict_load() -> ModelLoadEvidence {
        ModelLoadEvidence {
            strict: true,
            loader: "safetensors".into(),
            loaded_tensors: 10,
            required_tensors: 10,
            optional_probed: 0,
            optional_loaded: 0,
            seeded_fallback_remained: false,
        }
    }

    fn runtime_contract() -> RuntimeContractInput {
        RuntimeContractInput {
            real_transformer_enabled: true,
            real_transformer_gpu_native: true,
            compute_offload: crate::backend::ComputeOffload::Gpu,
            legacy_execution_plan: legacy_plan(),
            token_loop_geometry: Some(qwen_geometry()),
            authoritative_device: Some(l4_device()),
            model_load: strict_load(),
            routed_failure_policy: crate::engine::RoutedExpertGpuFailurePolicy::StrictFailClosed,
        }
    }

    fn empty_snapshots(counter: GpuNativeTokenLoopSnapshot) -> RequestSnapshots {
        RequestSnapshots {
            token_loop_before: GpuNativeTokenLoopSnapshot::default(),
            token_loop_after: counter,
            token_loop_delta: counter,
            token_loop_ratios: CounterRatios::from_delta(counter).unwrap(),
            recovery_before: GpuNativeRecoverySnapshot::default(),
            recovery_after: GpuNativeRecoverySnapshot::default(),
            recovery_delta: GpuNativeRecoverySnapshot::default(),
            recovery_ratios: RecoveryRatios::from_delta(
                GpuNativeRecoverySnapshot::default(),
                counter.tokens_completed,
            )
            .unwrap(),
            routed_execution_before: RoutedExpertExecutionSnapshot::default(),
            routed_execution_after: RoutedExpertExecutionSnapshot::default(),
            routed_execution_delta: RoutedExpertExecutionSnapshot::default(),
            runtime_cache_before: RuntimeCacheSnapshot::default(),
            runtime_cache_after: RuntimeCacheSnapshot::default(),
            engine_storage_before: EngineStorageSnapshot::default(),
            engine_storage_after: EngineStorageSnapshot::default(),
            engine_storage_delta: EngineStorageSnapshot::default(),
            gpu_expert_io_before: GpuExpertIoSnapshot::default(),
            gpu_expert_io_after: GpuExpertIoSnapshot::default(),
            gpu_expert_io_delta: GpuExpertIoSnapshot::default(),
            gpu_expert_memory_before: GpuExpertMemorySnapshot::default(),
            gpu_expert_memory_after: GpuExpertMemorySnapshot::default(),
            gpu_native_residency_before: GpuNativeTieredResidencySnapshot::default(),
            gpu_native_residency_after: GpuNativeTieredResidencySnapshot::default(),
            gpu_native_residency_delta: GpuNativeResidencyDelta::default(),
        }
    }

    fn run(index: usize, decode_tps_seconds: f64) -> PerRunResult {
        let counter = GpuNativeTokenLoopSnapshot {
            token_attempts: 5,
            tokens_completed: 5,
            warm_tokens_completed: 5,
            queue_submissions: 5,
            boundary_maps: 5,
            boundary_readbacks: 5,
            ..GpuNativeTokenLoopSnapshot::default()
        };
        PerRunResult {
            run_index: index,
            prompt_tokens: 4,
            requested_output_tokens: 2,
            generated_tokens: 2,
            generated_token_ids: vec![7, 8],
            generated_token_ids_sha256: "a".repeat(64),
            generated_text_sha256: "b".repeat(64),
            timing: RunTiming::from_measurement(
                2,
                2.0,
                2.0,
                decode_tps_seconds,
                vec![decode_tps_seconds],
            )
            .unwrap(),
            counters: empty_snapshots(counter),
        }
    }

    #[test]
    fn timing_arithmetic_excludes_first_decode_and_includes_all_end_to_end() {
        let timing = RunTiming::from_measurement(5, 2.0, 2.0, 4.0, vec![1.0; 4]).unwrap();
        assert_eq!(timing.decode_generated_tokens, 4);
        assert_eq!(timing.decode_tps, 1.0);
        assert_eq!(timing.end_to_end_seconds, 6.0);
        assert_eq!(timing.end_to_end_generated_tps, 5.0 / 6.0);
    }

    #[test]
    fn percentile_and_latency_statistics_are_deterministic() {
        let stats = LatencyStatistics::from_values(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(stats.p50_seconds, 3.0);
        assert_eq!(stats.p95_seconds, 5.0);
        assert_eq!(stats.p99_seconds, 5.0);
        assert_eq!(stats.mean_seconds, 3.0);
        assert_eq!(stats.min_seconds, 1.0);
        assert_eq!(stats.max_seconds, 5.0);
    }

    #[test]
    fn counter_delta_and_ratios_are_exact() {
        let before = GpuNativeTokenLoopSnapshot {
            token_attempts: 10,
            tokens_completed: 5,
            queue_submissions: 10,
            ..GpuNativeTokenLoopSnapshot::default()
        };
        let after = GpuNativeTokenLoopSnapshot {
            token_attempts: 18,
            tokens_completed: 9,
            residency_miss_attempts: 2,
            replay_attempts: 2,
            queue_submissions: 18,
            ..GpuNativeTokenLoopSnapshot::default()
        };
        let delta = token_loop_delta(before, after).unwrap();
        let ratios = CounterRatios::from_delta(delta).unwrap();
        assert_eq!(delta.token_attempts, 8);
        assert_eq!(delta.tokens_completed, 4);
        assert_eq!(ratios.attempts_per_completed_position, 2.0);
        assert_eq!(ratios.misses_per_completed_position, 0.5);
        assert_eq!(ratios.replays_per_completed_position, 0.5);
        assert_eq!(ratios.submissions_per_completed_position, 2.0);
    }

    #[test]
    fn recovery_delta_and_ratios_are_exact() {
        let before = GpuNativeRecoverySnapshot {
            resume_attempts: 2,
            checkpoint_captures: 10,
            layers_encoded: 10,
            residency_service_us: 40,
            boundary_wait_us: 50,
            ..GpuNativeRecoverySnapshot::default()
        };
        let after = GpuNativeRecoverySnapshot {
            resume_attempts: 4,
            recovery_segments: 3,
            checkpoint_captures: 18,
            checkpoint_restores: 2,
            layers_encoded: 20,
            attention_layers_reexecuted: 2,
            expert_layers_reexecuted: 4,
            invalid_tail_layers_encoded: 6,
            residency_service_us: 140,
            boundary_wait_us: 250,
            ..GpuNativeRecoverySnapshot::default()
        };
        let delta = recovery_delta(before, after).unwrap();
        let ratios = RecoveryRatios::from_delta(delta, 2).unwrap();
        assert_eq!(delta.resume_attempts, 2);
        assert_eq!(delta.checkpoint_captures, 8);
        assert_eq!(delta.full_token_replay_attempts, 0);
        assert_eq!(ratios.resume_attempts_per_completed_position, 1.0);
        assert_eq!(ratios.checkpoint_captures_per_completed_position, 4.0);
        assert_eq!(ratios.residency_service_us_per_completed_position, 50.0);
        assert_eq!(ratios.boundary_wait_us_per_completed_position, 100.0);
    }

    #[test]
    fn physical_source_of_truth_counters_are_reported_without_speculation() {
        let before = GpuNativeTieredResidencySnapshot {
            vram_hits: 10,
            vram_misses: 4,
            physical_current_hits: 10,
            physical_source_acquisitions: 4,
            logical_admissions_for_physical_misses: 3,
            ram_to_vram_installs: 4,
            ..GpuNativeTieredResidencySnapshot::default()
        };
        let after = GpuNativeTieredResidencySnapshot {
            vram_hits: 17,
            vram_misses: 6,
            physical_current_hits: 17,
            physical_source_acquisitions: 6,
            logical_admissions_for_physical_misses: 5,
            ram_to_vram_installs: 6,
            ..before.clone()
        };
        let delta = gpu_native_residency_delta(&before, &after).unwrap();
        assert_eq!(delta.vram_hits, 7);
        assert_eq!(delta.vram_misses, 2);
        assert_eq!(delta.physical_current_hits, 7);
        assert_eq!(delta.physical_source_acquisitions, 2);
        assert_eq!(delta.logical_admissions_for_physical_misses, 2);
        assert_eq!(delta.ram_to_vram_installs, 2);
        assert_eq!(delta.speculative_requests, 0);
        assert_eq!(delta.speculative_vram_hits, 0);
        assert_eq!(delta.speculative_ram_to_vram_installs, 0);
        assert_eq!(delta.speculative_dropped_capacity_or_pressure, 0);
    }

    #[test]
    fn cache_reset_semantics_define_runtime_construction_count() {
        assert_eq!(
            expected_runtime_constructions(crate::BenchRealCacheReset::Keep, 3, 4),
            1
        );
        assert_eq!(
            expected_runtime_constructions(crate::BenchRealCacheReset::FreshRuntime, 3, 4),
            7
        );
    }

    #[test]
    fn aggregate_excludes_unpassed_warmup_and_retains_each_measured_run() {
        let warmup = run(99, 100.0);
        let mut measured = vec![run(0, 1.0), run(1, 2.0)];
        let recovery = GpuNativeRecoverySnapshot {
            resume_attempts: 2,
            recovery_segments: 3,
            checkpoint_restores: 2,
            ..GpuNativeRecoverySnapshot::default()
        };
        measured[0].counters.recovery_after = recovery;
        measured[0].counters.recovery_delta = recovery;
        measured[0].counters.recovery_ratios = RecoveryRatios::from_delta(recovery, 5).unwrap();
        let aggregate = aggregate(&measured).unwrap();
        assert_eq!(measured.len(), 2);
        assert_eq!(aggregate.decode_tps.mean, 0.75);
        assert_ne!(aggregate.decode_tps.mean, warmup.timing.decode_tps);
        assert_eq!(aggregate.counter_totals.tokens_completed, 10);
        assert_eq!(aggregate.recovery_totals, recovery);
        assert_eq!(
            aggregate
                .recovery_ratios
                .resume_attempts_per_completed_position,
            0.2
        );
    }

    #[test]
    fn zero_or_nonfinite_duration_fails_closed() {
        assert!(RunTiming::from_measurement(2, 1.0, 1.0, 0.0, vec![0.0]).is_err());
        assert!(RunTiming::from_measurement(2, f64::NAN, 1.0, 1.0, vec![1.0]).is_err());
    }

    #[test]
    fn failed_measured_run_prevents_aggregate_publication() {
        let mut retained = vec![run(0, 1.0)];
        let failed = retain_measured_result(
            &mut retained,
            Err(BenchmarkFailure::new(
                "inference",
                "failed-measured-run",
                "fixture failure",
            )),
        )
        .unwrap_err();
        assert_eq!(failed.code, "failed-measured-run");
        assert_eq!(
            retained.len(),
            1,
            "failed run must not replace or hide prior evidence"
        );
    }

    #[test]
    fn strict_runtime_contract_accepts_corrected_legacy_context() {
        let (evidence, device) =
            validate_runtime_contract(&runtime_contract(), "NVIDIA L4").unwrap();
        assert!(evidence.ordinary_step_token_only);
        assert_eq!(evidence.legacy_execution_plan.embeddings, "cpu");
        assert_eq!(device.name, "NVIDIA L4");
    }

    #[test]
    fn missing_token_loop_is_rejected() {
        let mut input = runtime_contract();
        input.token_loop_geometry = None;
        assert_eq!(
            validate_runtime_contract(&input, "NVIDIA L4")
                .unwrap_err()
                .code,
            "missing-gpu-native-token-loop"
        );
    }

    #[test]
    fn wrong_and_software_adapters_are_rejected() {
        let mut wrong = runtime_contract();
        wrong.authoritative_device.as_mut().unwrap().name = "other".into();
        assert_eq!(
            validate_runtime_contract(&wrong, "NVIDIA L4")
                .unwrap_err()
                .code,
            "wrong-adapter"
        );
        let mut software = runtime_contract();
        software
            .authoritative_device
            .as_mut()
            .unwrap()
            .software_adapter = true;
        assert_eq!(
            validate_runtime_contract(&software, "NVIDIA L4")
                .unwrap_err()
                .code,
            "software-adapter"
        );
    }

    #[test]
    fn wrong_backend_and_incomplete_strict_load_are_rejected() {
        let mut wrong_backend = runtime_contract();
        wrong_backend
            .authoritative_device
            .as_mut()
            .unwrap()
            .wgpu_backend = "metal".into();
        assert_eq!(
            validate_runtime_contract(&wrong_backend, "NVIDIA L4")
                .unwrap_err()
                .code,
            "wrong-gpu-backend"
        );

        let mut incomplete = runtime_contract();
        incomplete.model_load.loaded_tensors -= 1;
        assert_eq!(
            validate_runtime_contract(&incomplete, "NVIDIA L4")
                .unwrap_err()
                .code,
            "incomplete-model-load"
        );
    }

    #[test]
    fn fallback_and_degradation_are_rejected() {
        let token_loop = GpuNativeTokenLoopSnapshot {
            tokens_completed: 5,
            ..GpuNativeTokenLoopSnapshot::default()
        };
        let routed = RoutedExpertExecutionSnapshot {
            gpu_cpu_fallbacks: 1,
            ..RoutedExpertExecutionSnapshot::default()
        };
        assert_eq!(
            validate_request_postconditions(
                4,
                2,
                2,
                token_loop,
                GpuNativeRecoverySnapshot::default(),
                routed,
            )
            .unwrap_err()
            .code,
            "fallback-or-degradation"
        );

        let cpu_dispatch = RoutedExpertExecutionSnapshot {
            cpu_routed_expert_dispatches: 1,
            ..RoutedExpertExecutionSnapshot::default()
        };
        assert_eq!(
            validate_request_postconditions(
                4,
                2,
                2,
                token_loop,
                GpuNativeRecoverySnapshot::default(),
                cpu_dispatch,
            )
            .unwrap_err()
            .code,
            "fallback-or-degradation"
        );

        let degraded = RoutedExpertExecutionSnapshot {
            degraded_expert_substitutions: 1,
            ..RoutedExpertExecutionSnapshot::default()
        };
        assert_eq!(
            validate_request_postconditions(
                4,
                2,
                2,
                token_loop,
                GpuNativeRecoverySnapshot::default(),
                degraded,
            )
            .unwrap_err()
            .code,
            "fallback-or-degradation"
        );
    }

    #[test]
    fn fatal_no_progress_and_incomplete_generation_are_rejected() {
        let fatal = GpuNativeTokenLoopSnapshot {
            tokens_completed: 5,
            fatal_failures: 1,
            ..GpuNativeTokenLoopSnapshot::default()
        };
        assert_eq!(
            validate_request_postconditions(
                4,
                2,
                2,
                fatal,
                GpuNativeRecoverySnapshot::default(),
                RoutedExpertExecutionSnapshot::default(),
            )
            .unwrap_err()
            .code,
            "token-loop-postcondition"
        );

        let no_progress = GpuNativeTokenLoopSnapshot {
            tokens_completed: 5,
            no_progress_failures: 1,
            ..GpuNativeTokenLoopSnapshot::default()
        };
        assert_eq!(
            validate_request_postconditions(
                4,
                2,
                2,
                no_progress,
                GpuNativeRecoverySnapshot::default(),
                RoutedExpertExecutionSnapshot::default(),
            )
            .unwrap_err()
            .code,
            "token-loop-postcondition"
        );

        let otherwise_valid = GpuNativeTokenLoopSnapshot {
            tokens_completed: 5,
            ..GpuNativeTokenLoopSnapshot::default()
        };
        assert_eq!(
            validate_request_postconditions(
                4,
                2,
                1,
                otherwise_valid,
                GpuNativeRecoverySnapshot::default(),
                RoutedExpertExecutionSnapshot::default(),
            )
            .unwrap_err()
            .code,
            "incomplete-generation"
        );
    }

    #[test]
    fn full_token_replay_is_rejected_but_resumable_miss_is_accepted() {
        let token_loop = GpuNativeTokenLoopSnapshot {
            tokens_completed: 5,
            residency_miss_attempts: 1,
            replay_attempts: 1,
            ..GpuNativeTokenLoopSnapshot::default()
        };
        assert_eq!(
            validate_request_postconditions(
                4,
                2,
                2,
                token_loop,
                GpuNativeRecoverySnapshot {
                    resume_attempts: 1,
                    full_token_replay_attempts: 1,
                    ..GpuNativeRecoverySnapshot::default()
                },
                RoutedExpertExecutionSnapshot::default(),
            )
            .unwrap_err()
            .code,
            "full-token-replay-observed"
        );

        let resumable = GpuNativeTokenLoopSnapshot {
            replay_attempts: 0,
            ..token_loop
        };
        assert!(validate_request_postconditions(
            4,
            2,
            2,
            resumable,
            GpuNativeRecoverySnapshot {
                resume_attempts: 1,
                recovery_segments: 1,
                checkpoint_restores: 1,
                ..GpuNativeRecoverySnapshot::default()
            },
            RoutedExpertExecutionSnapshot::default(),
        )
        .is_ok());
    }

    #[test]
    fn report_flags_are_permanently_nonqualification_and_nondiagnostic() {
        let report = BenchmarkReport::new(
            BenchmarkProvenance {
                build: BuildProvenance {
                    git_sha: Some("a".repeat(40)),
                    dirty: Some(false),
                    package_version: "test".into(),
                },
                executable_canonical_path: "/test/bin".into(),
                executable_sha256: "b".repeat(64),
                resolved_config_sha256: "c".repeat(64),
                artifacts: QualificationArtifacts::default(),
                expert_metadata: ExpertMetadataEvidence {
                    dtype: Some("q4_0".into()),
                    q4_0_layout: Some(crate::inference::Q4_0_LAYOUT_STANDARD_V1.into()),
                    conversion_mode: None,
                    source: None,
                    explicitly_synthetic: false,
                },
            },
            ModelIdentityEvidence {
                architecture: "qwen3_moe".into(),
                num_layers: 48,
                num_experts_per_layer: 128,
                total_experts: 6_144,
                top_k: 8,
                d_model: 2_048,
                d_ff: 768,
                routed_expert_dtype: "q4_0".into(),
            },
            RequestEvidence {
                prompt_sha256: "d".repeat(64),
                prompt_token_ids_sha256: "e".repeat(64),
                prompt_token_count: 2,
                requested_output_tokens: 2,
                greedy: true,
            },
            crate::BenchRealCacheReset::Keep,
            0,
            1,
            ProductionConfiguration::default(),
        );
        assert!(!report.production_semantics.diagnostic_trace_enabled);
        assert!(
            !report
                .production_semantics
                .production_inference_math_changed
        );
        assert!(!report.production_semantics.production_q4_changed);
        assert!(!report.production_semantics.production_router_changed);
        assert!(!report.production_semantics.production_attention_changed);
        assert!(!report.production_semantics.production_rmsnorm_changed);
        assert!(!report.production_semantics.production_lm_head_changed);
        assert!(
            report
                .production_semantics
                .production_residency_policy_changed
        );
        assert!(report.production_semantics.production_replay_policy_changed);
        assert!(
            !report
                .production_semantics
                .production_prefetch_policy_changed
        );
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(json["qualification_pass"], false);
        assert_eq!(json["correctness_qualification_pending"], true);
        assert_eq!(json["schema"], SCHEMA);
        assert_eq!(json["optimization"], OPTIMIZATION);
        assert_eq!(
            json["immediate_comparison_commit"],
            IMMEDIATE_COMPARISON_COMMIT
        );
        assert_eq!(json["pr1b_a_experiment_commit"], PR1B_A_EXPERIMENT_COMMIT);
        assert_eq!(
            json["pr1b_a_experiment_report_sha256"],
            PR1B_A_EXPERIMENT_REPORT_SHA256
        );
        assert_eq!(json["original_baseline_commit"], ORIGINAL_BASELINE_COMMIT);
        assert_eq!(
            json["production_semantics"]["production_residency_policy_changed"],
            true
        );
        assert_eq!(
            json["production_semantics"]["production_replay_policy_changed"],
            true
        );
        assert_eq!(
            json["production_semantics"]["diagnostic_trace_enabled"],
            false
        );
    }

    #[test]
    fn cli_parses_required_gpu_native_benchmark_surface() {
        use clap::Parser as _;

        let cli = crate::Cli::try_parse_from([
            "micro-expert-router",
            "bench-gpu-native-real",
            "--config",
            "config.toml",
            "--prompt",
            "hello",
            "--output-tokens",
            "8",
            "--warmup-runs",
            "2",
            "--measured-runs",
            "3",
            "--cache-reset",
            "fresh-runtime",
            "--greedy",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--report-out",
            "report.json",
        ])
        .unwrap();
        let crate::Cmd::BenchGpuNativeReal {
            output_tokens,
            warmup_runs,
            measured_runs,
            cache_reset,
            greedy,
            expected_adapter_name,
            report_out,
            ..
        } = cli.cmd
        else {
            panic!("expected bench-gpu-native-real command")
        };
        assert_eq!(output_tokens, Some(8));
        assert_eq!(warmup_runs, 2);
        assert_eq!(measured_runs, 3);
        assert_eq!(cache_reset, crate::BenchRealCacheReset::FreshRuntime);
        assert!(greedy);
        assert_eq!(expected_adapter_name, "NVIDIA L4");
        assert_eq!(report_out, Some(PathBuf::from("report.json")));
    }

    #[test]
    fn cli_rejects_missing_required_greedy_flag() {
        use clap::Parser as _;

        let error = crate::Cli::try_parse_from([
            "micro-expert-router",
            "bench-gpu-native-real",
            "--config",
            "config.toml",
            "--prompt",
            "hello",
            "--expected-adapter-name",
            "NVIDIA L4",
        ])
        .unwrap_err();
        assert!(error.to_string().contains("--greedy"));
    }

    #[test]
    fn ordinary_step_orchestration_matches_ingest_contract_fixture() {
        fn sample(token: u32, position: usize) -> u32 {
            token.wrapping_add(position as u32).wrapping_add(17) % 101
        }
        fn ordinary(prompt: &[u32], output_tokens: usize) -> Vec<u32> {
            let mut generated = Vec::new();
            let mut completed = 0usize;
            while generated.len() < output_tokens {
                let step = next_ordinary_step(prompt, &generated, completed).unwrap();
                completed += 1;
                if step.sample {
                    generated.push(sample(step.token_id, step.position));
                }
            }
            generated
        }
        fn ingest_contract(prompt: &[u32], output_tokens: usize) -> Vec<u32> {
            let mut generated = vec![sample(*prompt.last().unwrap(), prompt.len() - 1)];
            while generated.len() < output_tokens {
                let position = prompt.len() + generated.len() - 1;
                generated.push(sample(*generated.last().unwrap(), position));
            }
            generated
        }
        let prompt = [3, 5, 8];
        assert_eq!(ordinary(&prompt, 6), ingest_contract(&prompt, 6));
    }
}
