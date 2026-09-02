//! PR2-A production qualification for exact-demand concurrent source
//! acquisition. The command compares an explicit sequential control with the
//! ordinary production source path used by normal serving, constructing a
//! fresh isolated runtime for each arm.

use crate::backend::{GpuExpertIoSnapshot, GpuExpertMemorySnapshot};
use crate::engine::{
    GpuNativeDemandSourceQualificationArm, GpuNativeDemandSourceQualificationSnapshot,
    ProductionDemandSourceSnapshot,
    RoutedExpertExecutionSnapshot,
};
use crate::gpu_native_real_benchmark::{
    BenchmarkFailure, BenchmarkProvenance, BenchmarkReport, EngineStorageSnapshot,
    GpuNativeResidencyDelta, PerRunResult, ProductionConfiguration, RequestEvidence,
};
use crate::gpu_native_residency::GpuNativeTieredResidencySnapshot;
use crate::gpu_native_token_loop::{GpuNativeRecoverySnapshot, GpuNativeTokenLoopSnapshot};
use crate::qualification::{BuildProvenance, QualificationArtifacts};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub(crate) const PRODUCTION_SCHEMA: &str = "mer.gpu-native-demand-source-concurrency.v2";
pub(crate) const PRODUCTION_MODE: &str = "qualify-gpu-native-demand-source-concurrency-production";
pub(crate) const FROZEN_PROMPT: &str =
    "Write a Rust function that adds two i32 values and returns the result.";
pub(crate) const FROZEN_OUTPUT_TOKENS: usize = 128;
pub(crate) const FROZEN_WARMUP_RUNS: usize = 1;
pub(crate) const FROZEN_MEASURED_RUNS: usize = 3;

#[derive(Clone, Debug)]
pub(crate) struct CommandArgs {
    pub(crate) config: PathBuf,
    pub(crate) expected_adapter_name: String,
    pub(crate) report_out: PathBuf,
    pub(crate) progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FrozenWorkload {
    model: &'static str,
    quantization: &'static str,
    prompt: &'static str,
    output_tokens: usize,
    warmup_runs: usize,
    measured_runs: usize,
    cache_reset: &'static str,
    sampling: &'static str,
    expected_adapter_name: String,
    backend: &'static str,
    datadog: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct WarmupEvidence {
    run_index: usize,
    generated_tokens: usize,
    generated_token_ids_sha256: String,
    generated_text_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ArmWorkEvidence {
    token_loop: GpuNativeTokenLoopSnapshot,
    recovery: GpuNativeRecoverySnapshot,
    routed_execution: RoutedExpertExecutionSnapshot,
    engine_storage: EngineStorageSnapshot,
    gpu_expert_io: GpuExpertIoSnapshot,
    gpu_expert_memory_before: GpuExpertMemorySnapshot,
    gpu_expert_memory_after: GpuExpertMemorySnapshot,
    gpu_native_residency: GpuNativeResidencyDelta,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ArmReport {
    arm: GpuNativeDemandSourceQualificationArm,
    complete: bool,
    failure: Option<BenchmarkFailure>,
    isolated_runtime: bool,
    warmup_results: Vec<WarmupEvidence>,
    warmup_source: Option<GpuNativeDemandSourceQualificationSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warmup_production: Option<ProductionDemandSourceSnapshot>,
    warmup_ram_cache_state_sha256: Option<String>,
    warmup_work: Option<ArmWorkEvidence>,
    source: Option<GpuNativeDemandSourceQualificationSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    production: Option<ProductionDemandSourceSnapshot>,
    work: Option<ArmWorkEvidence>,
    benchmark: BenchmarkReport,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct Reconciliation {
    generated_tokens_exact: bool,
    generated_token_hashes_exact: bool,
    warmup_token_hashes_exact: bool,
    warmup_source_requests_exact: bool,
    warmup_ram_source_hits_exact: bool,
    warmup_ram_source_misses_exact: bool,
    warmup_nvme_reads_exact: bool,
    warmup_nvme_bytes_exact: bool,
    warmup_ram_cache_inserts_exact: bool,
    warmup_ram_cache_evictions_exact: bool,
    warmup_ordered_ram_insert_ids_exact: bool,
    warmup_ordered_ram_eviction_ids_exact: bool,
    warmup_physical_missing_sequence_exact: bool,
    warmup_demand_source_sequence_exact: bool,
    warmup_ram_cache_state_exact: bool,
    warmup_all_speculative_work_zero: bool,
    selected_route_sequence_exact: bool,
    selected_route_counts_exact: bool,
    physical_missing_sequence_exact: bool,
    physical_missing_counts_exact: bool,
    demand_source_request_sequence_exact: bool,
    demand_source_requests_exact: bool,
    ram_source_hits_exact: bool,
    ram_source_misses_exact: bool,
    demand_nvme_reads_exact: bool,
    demand_nvme_bytes_exact: bool,
    logical_admissions_exact: bool,
    ram_to_vram_installs_exact: bool,
    ram_to_vram_bytes_exact: bool,
    physical_evictions_exact: bool,
    vram_hits_exact: bool,
    vram_misses_exact: bool,
    residency_miss_attempts_exact: bool,
    residency_services_exact: bool,
    recovery_segments_exact: bool,
    miss_boundaries_exact: bool,
    recovery_semantics_exact: bool,
    full_token_replay_zero: bool,
    fatal_and_no_progress_zero: bool,
    ram_cache_inserts_exact: bool,
    ram_cache_evictions_exact: bool,
    ordered_ram_insert_ids_exact: bool,
    ordered_ram_eviction_ids_exact: bool,
    all_speculative_work_zero: bool,
    primary_pool_capacity_exact: bool,
    all_invariants_pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BehavioralGate {
    generated_token_parity_exact: bool,
    route_parity_exact: bool,
    physical_missing_sequence_exact: bool,
    full_token_replay_zero: bool,
    fatal_and_no_progress_zero: bool,
    speculation_zero: bool,
    passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct WorkEquivalenceGate {
    demand_nvme_bytes_exact: bool,
    demand_h2d_installs_exact: bool,
    demand_h2d_bytes_exact: bool,
    physical_evictions_exact: bool,
    recovery_and_miss_boundaries_exact: bool,
    deterministic_ram_cache_order_exact: bool,
    passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct MetricComparison {
    control: f64,
    treatment: f64,
    delta_percent: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProductionReconciliation {
    common: Reconciliation,
    warmup_production_cache_reservation_leaks_zero: bool,
    warmup_production_batch_commit_violations_zero: bool,
    warmup_stale_singleflight_entries_zero: bool,
    production_cache_reservation_leaks_zero: bool,
    production_batch_commit_violations_zero: bool,
    stale_singleflight_entries_zero: bool,
    all_invariants_pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProductionMechanismGate {
    control_production_batch_successes_zero: bool,
    treatment_production_batch_successes_gt_zero: bool,
    treatment_production_batch_width_max_gt_one: bool,
    treatment_ordinary_production_path_exercised: bool,
    treatment_cache_slots_exactly_reconciled: bool,
    treatment_singleflight_claims_exactly_reconciled: bool,
    passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProductionGates {
    behavioral: BehavioralGate,
    work_equivalence: WorkEquivalenceGate,
    mechanism: ProductionMechanismGate,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProductionArmPerformance {
    decode_tps: f64,
    end_to_end_generated_tps: f64,
    mean_request_wall_seconds: f64,
    source_acquisition_wall_us: u64,
    logical_demand_admission_us: u64,
    physical_demand_install_us: u64,
    total_residency_service_us: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProductionPerformanceComparison {
    control: ProductionArmPerformance,
    treatment: ProductionArmPerformance,
    decode_tps: MetricComparison,
    end_to_end_generated_tps: MetricComparison,
    mean_request_wall_seconds: MetricComparison,
    source_acquisition_wall_us: MetricComparison,
    logical_demand_admission_us: MetricComparison,
    physical_demand_install_us: MetricComparison,
    total_residency_service_us: MetricComparison,
    performance_result: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ProductionQualificationReport {
    schema: &'static str,
    mode: &'static str,
    production_demand_source_changed: bool,
    control_forces_legacy_sequential: bool,
    treatment_uses_ordinary_production_path: bool,
    benchmark_complete: bool,
    qualification_pass: bool,
    performance_result: &'static str,
    failure: Option<BenchmarkFailure>,
    frozen_workload: FrozenWorkload,
    provenance: BenchmarkProvenance,
    control: Option<ArmReport>,
    treatment: Option<ArmReport>,
    reconciliation: Option<ProductionReconciliation>,
    gates: Option<ProductionGates>,
    performance: Option<ProductionPerformanceComparison>,
}

struct Prepared {
    spec: crate::ResolvedRealCliSpec,
    tokenizer: Arc<crate::tokenizer::Tokenizer>,
    prompt_ids: Vec<u32>,
    resolved_config_sha256: String,
    provenance: BenchmarkProvenance,
    model_identity: crate::greedy_parity::ModelIdentityEvidence,
    request: RequestEvidence,
    production_configuration: ProductionConfiguration,
}

#[derive(Clone)]
struct ArmStart {
    token_loop: GpuNativeTokenLoopSnapshot,
    recovery: GpuNativeRecoverySnapshot,
    routed: RoutedExpertExecutionSnapshot,
    engine_storage: EngineStorageSnapshot,
    gpu_io: GpuExpertIoSnapshot,
    gpu_memory: GpuExpertMemorySnapshot,
    residency: GpuNativeTieredResidencySnapshot,
}

impl ArmStart {
    fn capture(runtime: &crate::BenchRealRuntime) -> Result<Self, BenchmarkFailure> {
        let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
            BenchmarkFailure::new(
                "startup",
                "missing-gpu-native-token-loop",
                "PR2-A arm did not construct the authoritative GPU-native token loop",
            )
        })?;
        Ok(Self {
            token_loop: token_loop.snapshot(),
            recovery: token_loop.recovery_snapshot(),
            routed: runtime.engine.routed_expert_execution_snapshot(),
            engine_storage: EngineStorageSnapshot::from_runtime(runtime),
            gpu_io: runtime.engine.gpu_expert_io_snapshot().ok_or_else(|| {
                BenchmarkFailure::new(
                    "startup",
                    "missing-gpu-io-snapshot",
                    "PR2-A arm did not expose GPU expert I/O counters",
                )
            })?,
            gpu_memory: runtime.engine.gpu_expert_memory_snapshot().ok_or_else(|| {
                BenchmarkFailure::new(
                    "startup",
                    "missing-gpu-memory-snapshot",
                    "PR2-A arm did not expose GPU expert memory counters",
                )
            })?,
            residency: runtime
                .engine
                .gpu_native_residency_snapshot()
                .ok_or_else(|| {
                    BenchmarkFailure::new(
                        "startup",
                        "missing-gpu-native-residency-snapshot",
                        "PR2-A arm did not expose GPU-native residency counters",
                    )
                })?,
        })
    }

    fn finish(
        self,
        runtime: &crate::BenchRealRuntime,
    ) -> Result<ArmWorkEvidence, BenchmarkFailure> {
        let token_loop = runtime.gpu_native_token_loop.as_ref().ok_or_else(|| {
            BenchmarkFailure::new(
                "postcondition",
                "missing-gpu-native-token-loop",
                "PR2-A token loop disappeared after measurement",
            )
        })?;
        let engine_storage_after = EngineStorageSnapshot::from_runtime(runtime);
        let gpu_io_after = runtime.engine.gpu_expert_io_snapshot().ok_or_else(|| {
            BenchmarkFailure::new(
                "postcondition",
                "missing-gpu-io-snapshot",
                "PR2-A GPU expert I/O counters disappeared",
            )
        })?;
        let gpu_memory_after = runtime.engine.gpu_expert_memory_snapshot().ok_or_else(|| {
            BenchmarkFailure::new(
                "postcondition",
                "missing-gpu-memory-snapshot",
                "PR2-A GPU expert memory counters disappeared",
            )
        })?;
        let residency_after = runtime
            .engine
            .gpu_native_residency_snapshot()
            .ok_or_else(|| {
                BenchmarkFailure::new(
                    "postcondition",
                    "missing-gpu-native-residency-snapshot",
                    "PR2-A GPU-native residency counters disappeared",
                )
            })?;
        Ok(ArmWorkEvidence {
            token_loop: crate::gpu_native_real_benchmark::token_loop_delta(
                self.token_loop,
                token_loop.snapshot(),
            )?,
            recovery: crate::gpu_native_real_benchmark::recovery_delta(
                self.recovery,
                token_loop.recovery_snapshot(),
            )?,
            routed_execution: crate::gpu_native_real_benchmark::routed_delta(
                self.routed,
                runtime.engine.routed_expert_execution_snapshot(),
            )?,
            engine_storage: engine_storage_after.checked_delta(self.engine_storage)?,
            gpu_expert_io: crate::gpu_native_real_benchmark::gpu_io_delta(
                self.gpu_io,
                gpu_io_after,
            )?,
            gpu_expert_memory_before: self.gpu_memory,
            gpu_expert_memory_after: gpu_memory_after,
            gpu_native_residency: crate::gpu_native_real_benchmark::gpu_native_residency_delta(
                &self.residency,
                &residency_after,
            )?,
        })
    }
}

fn frozen_workload(expected_adapter_name: String) -> FrozenWorkload {
    FrozenWorkload {
        model: "Qwen3-Coder-30B-A3B-Instruct",
        quantization: "pure Q4_0",
        prompt: FROZEN_PROMPT,
        output_tokens: FROZEN_OUTPUT_TOKENS,
        warmup_runs: FROZEN_WARMUP_RUNS,
        measured_runs: FROZEN_MEASURED_RUNS,
        cache_reset: "keep",
        sampling: "greedy",
        expected_adapter_name,
        backend: "WGPU/Vulkan",
        datadog: "off",
    }
}

fn validate_isolation_config(cfg: &crate::config::Config) -> Result<(), BenchmarkFailure> {
    let predictive = &cfg.predictive;
    if cfg.storage.predict_fanout != 0
        || predictive.locality_enabled
        || predictive.speculator_enabled
        || predictive.affinity_enabled
        || predictive.pregate_enabled
        || predictive.static_residency_fraction != 0.0
        || predictive.static_residency_profile.is_some()
    {
        return Err(BenchmarkFailure::new(
            "preflight",
            "speculation-must-be-disabled",
            "PR2-A requires storage.predict_fanout=0 and every locality/speculator/affinity/pregate/static-residency arm disabled",
        ));
    }
    if cfg.storage.pin_after_observations != 0 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "observation-pinning-must-be-disabled",
            format!(
                "PR2-A requires storage.pin_after_observations=0; observed {}",
                cfg.storage.pin_after_observations
            ),
        ));
    }
    if predictive.cost_aware_eviction {
        return Err(BenchmarkFailure::new(
            "preflight",
            "cost-aware-eviction-must-be-disabled",
            "PR2-A requires predictive.cost_aware_eviction=false so victim ordering is strict LRU",
        ));
    }
    if cfg.storage.packed_blob.is_some() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "packed-blob-must-be-disabled",
            "PR2-A requires storage.packed_blob to be unset so the treatment measures ordinary per-file reads",
        ));
    }
    if cfg.storage.packed_manifest.is_some() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "packed-manifest-must-be-disabled",
            "PR2-A requires storage.packed_manifest to be unset so the treatment measures ordinary per-file reads",
        ));
    }
    Ok(())
}

fn prepare(args: &CommandArgs) -> Result<Prepared, Box<dyn std::error::Error>> {
    if args.expected_adapter_name.trim().is_empty() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "missing-expected-adapter",
            "PR2-A requires an exact nonempty adapter name",
        )
        .into());
    }
    let build = BuildProvenance::embedded();
    crate::gpu_native_real_benchmark::validate_preflight_provenance(&build)?;
    let cfg = crate::config::Config::from_file(&args.config)?;
    crate::gpu_native_real_benchmark::validate_source_config(&cfg)?;
    validate_isolation_config(&cfg)?;
    let (artifacts, artifact_errors): (QualificationArtifacts, Vec<String>) =
        crate::qualification_artifacts(&args.config, &cfg);
    crate::gpu_native_real_benchmark::validate_artifacts(&artifacts, &artifact_errors)?;
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| {
                BenchmarkFailure::new("preflight", "expert-metadata-unavailable", error)
            })?;
    crate::gpu_native_real_benchmark::validate_expert_metadata(&expert_metadata)?;
    let spec = crate::resolve_real_cli_spec_from_config(
        cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
    )?;
    let model_identity = crate::greedy_parity_model_identity(&spec);
    if !model_identity.is_qwen3_coder_30b_a3b_q4_0() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "wrong-model-identity",
            format!("PR2-A requires exact Qwen3-Coder 30B-A3B Q4_0; observed {model_identity:?}"),
        )
        .into());
    }
    let resolved_config_sha256 = crate::resolved_real_cli_spec_sha256(&spec)?;
    let tokenizer = crate::load_real_cli_tokenizer(
        &spec.cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBenchmark,
    )?;
    let prompt_ids = tokenizer.encode(FROZEN_PROMPT)?;
    if prompt_ids.is_empty() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "empty-prompt-tokenization",
            "the frozen PR2-A prompt encoded to zero tokens",
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
    if !crate::gpu_native_real_benchmark::is_hex(&executable_sha256, 64)
        || !crate::gpu_native_real_benchmark::is_hex(&resolved_config_sha256, 64)
    {
        return Err(BenchmarkFailure::new(
            "preflight",
            "provenance-unavailable",
            "PR2-A executable or resolved-config SHA256 was unavailable",
        )
        .into());
    }
    let production_configuration =
        ProductionConfiguration::from_config(&spec.cfg, &expert_metadata);
    Ok(Prepared {
        spec,
        tokenizer,
        prompt_ids: prompt_ids.clone(),
        resolved_config_sha256: resolved_config_sha256.clone(),
        provenance: BenchmarkProvenance {
            build,
            executable_canonical_path,
            executable_sha256,
            resolved_config_sha256,
            artifacts,
            expert_metadata,
        },
        model_identity,
        request: RequestEvidence {
            prompt_sha256: crate::greedy_parity::sha256_hex(FROZEN_PROMPT.as_bytes()),
            prompt_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&prompt_ids),
            prompt_token_count: prompt_ids.len(),
            requested_output_tokens: FROZEN_OUTPUT_TOKENS,
            greedy: true,
        },
        production_configuration,
    })
}

fn benchmark_report(prepared: &Prepared) -> BenchmarkReport {
    BenchmarkReport::new(
        prepared.provenance.clone(),
        prepared.model_identity.clone(),
        prepared.request.clone(),
        crate::BenchRealCacheReset::Keep,
        FROZEN_WARMUP_RUNS,
        FROZEN_MEASURED_RUNS,
        prepared.production_configuration.clone(),
    )
}

async fn run_arm(
    prepared: &Prepared,
    args: &CommandArgs,
    arm: GpuNativeDemandSourceQualificationArm,
) -> Result<ArmReport, BenchmarkFailure> {
    let arm_name = match arm {
        GpuNativeDemandSourceQualificationArm::Control => "control",
        GpuNativeDemandSourceQualificationArm::Treatment => "treatment",
    };
    let mut benchmark = benchmark_report(prepared);
    let runtime = crate::gpu_native_real_benchmark::construct_runtime(
        &prepared.spec,
        prepared.tokenizer.clone(),
        arm_name,
        None,
        &mut benchmark,
    )
    .await?;
    let enable_result = runtime
        .engine
        .enable_gpu_native_demand_source_production_qualification(arm);
    if let Err(error) = enable_result {
        let failure = BenchmarkFailure::new("startup", "qualification-arm-enable-failed", error);
        let _ = crate::gpu_native_real_benchmark::shutdown_runtime(
            runtime,
            arm_name,
            None,
            &mut benchmark,
        )
        .await;
        return Err(failure);
    }
    if let Err(validation_error) = crate::gpu_native_real_benchmark::validate_and_record_runtime(
        &runtime,
        &prepared.resolved_config_sha256,
        &args.expected_adapter_name,
        &mut benchmark,
    ) {
        let shutdown = crate::gpu_native_real_benchmark::shutdown_runtime(
            runtime,
            arm_name,
            None,
            &mut benchmark,
        )
        .await;
        return match shutdown {
            Ok(()) => Err(validation_error),
            Err(shutdown_error) => Err(BenchmarkFailure::new(
                "postcondition",
                "runtime-validation-and-shutdown-failed",
                format!("{validation_error}; {shutdown_error}"),
            )),
        };
    }

    let mut warmup_results = Vec::with_capacity(FROZEN_WARMUP_RUNS);
    let mut execution_failure = None;
    let mut warmup_start = match ArmStart::capture(&runtime) {
        Ok(captured) => Some(captured),
        Err(error) => {
            execution_failure = Some(error);
            None
        }
    };
    if execution_failure.is_none() {
        for index in 0..FROZEN_WARMUP_RUNS {
            let result = crate::with_progress_timeout(
                format!("{PRODUCTION_MODE} {arm_name} warmup {index}"),
                args.progress_watchdog,
                crate::gpu_native_real_benchmark::execute_request(
                    &runtime,
                    &prepared.prompt_ids,
                    FROZEN_OUTPUT_TOKENS,
                    index,
                ),
            )
            .await;
            match result {
                Ok(run) => {
                    warmup_results.push(WarmupEvidence {
                        run_index: index,
                        generated_tokens: run.generated_tokens,
                        generated_token_ids_sha256: run.generated_token_ids_sha256,
                        generated_text_sha256: run.generated_text_sha256,
                    });
                    benchmark.warmup_runs_completed += 1;
                }
                Err(error) => {
                    execution_failure = Some(BenchmarkFailure::new(
                        "inference",
                        "warmup-request-failed",
                        error.to_string(),
                    ));
                    break;
                }
            }
        }
    }

    let mut warmup_source = None;
    let mut warmup_production = None;
    let mut warmup_ram_cache_state_sha256 = None;
    let mut warmup_work = None;
    if execution_failure.is_none() {
        warmup_source = runtime
            .engine
            .gpu_native_demand_source_qualification_snapshot();
        if warmup_source.is_none() {
            execution_failure = Some(BenchmarkFailure::new(
                "postcondition",
                "missing-warmup-source-snapshot",
                "PR2-A qualification source counters disappeared after warmup",
            ));
        }
    }
    if execution_failure.is_none() {
        warmup_production = Some(runtime.engine.production_demand_source_snapshot());
    }
    if execution_failure.is_none() {
        warmup_ram_cache_state_sha256 = runtime
            .engine
            .gpu_native_demand_source_qualification_ram_cache_state_sha256();
        if warmup_ram_cache_state_sha256.is_none() {
            execution_failure = Some(BenchmarkFailure::new(
                "postcondition",
                "missing-warmup-ram-cache-state",
                "PR2-A RAM-cache state could not be hashed before the warmup counter reset",
            ));
        }
    }
    if execution_failure.is_none() {
        match warmup_start
            .take()
            .expect("warmup start captured")
            .finish(&runtime)
        {
            Ok(work) => warmup_work = Some(work),
            Err(error) => execution_failure = Some(error),
        }
    }

    let mut start = None;
    if execution_failure.is_none() {
        if let Err(error) = runtime
            .engine
            .reset_gpu_native_demand_source_qualification()
        {
            execution_failure = Some(BenchmarkFailure::new(
                "postcondition",
                "qualification-counter-reset-failed",
                error,
            ));
        } else {
            match ArmStart::capture(&runtime) {
                Ok(captured) => start = Some(captured),
                Err(error) => execution_failure = Some(error),
            }
        }
    }

    if execution_failure.is_none() {
        for index in 0..FROZEN_MEASURED_RUNS {
            let result = crate::with_progress_timeout(
                format!("{PRODUCTION_MODE} {arm_name} measured {index}"),
                args.progress_watchdog,
                crate::gpu_native_real_benchmark::execute_request(
                    &runtime,
                    &prepared.prompt_ids,
                    FROZEN_OUTPUT_TOKENS,
                    index,
                ),
            )
            .await;
            match result {
                Ok(run) => benchmark.per_run_results.push(run),
                Err(error) => {
                    execution_failure = Some(BenchmarkFailure::new(
                        "inference",
                        "measured-request-failed",
                        error.to_string(),
                    ));
                    break;
                }
            }
        }
    }

    let source = runtime
        .engine
        .gpu_native_demand_source_qualification_snapshot();
    let production = Some(runtime.engine.production_demand_source_snapshot());
    let work = if execution_failure.is_none() {
        match start.expect("measurement start captured").finish(&runtime) {
            Ok(work) => Some(work),
            Err(error) => {
                execution_failure = Some(error);
                None
            }
        }
    } else {
        None
    };

    let shutdown =
        crate::gpu_native_real_benchmark::shutdown_runtime(runtime, arm_name, None, &mut benchmark)
            .await;
    if let Err(error) = shutdown {
        execution_failure = Some(match execution_failure {
            Some(previous) => BenchmarkFailure::new(
                "postcondition",
                "execution-and-shutdown-failed",
                format!("{previous}; {error}"),
            ),
            None => error,
        });
    }

    if execution_failure.is_none() {
        if let Err(error) = benchmark.finish() {
            execution_failure = Some(error);
        }
    }
    if let Some(failure) = execution_failure.clone() {
        benchmark.fail(failure.clone());
    }
    Ok(ArmReport {
        arm,
        complete: execution_failure.is_none(),
        failure: execution_failure,
        isolated_runtime: true,
        warmup_results,
        warmup_source,
        warmup_production,
        warmup_ram_cache_state_sha256,
        warmup_work,
        source,
        production,
        work,
        benchmark,
    })
}

fn generated_results(arm: &ArmReport) -> &[PerRunResult] {
    &arm.benchmark.per_run_results
}

fn recovery_semantics_equal(a: GpuNativeRecoverySnapshot, b: GpuNativeRecoverySnapshot) -> bool {
    a.resume_attempts == b.resume_attempts
        && a.recovery_segments == b.recovery_segments
        && a.checkpoint_captures == b.checkpoint_captures
        && a.checkpoint_restores == b.checkpoint_restores
        && a.full_token_replay_attempts == b.full_token_replay_attempts
        && a.layers_encoded == b.layers_encoded
        && a.attention_layers_reexecuted == b.attention_layers_reexecuted
        && a.expert_layers_reexecuted == b.expert_layers_reexecuted
        && a.invalid_tail_layers_encoded == b.invalid_tail_layers_encoded
}

fn arm_all_speculative_work_zero(work: &ArmWorkEvidence) -> bool {
    work.gpu_native_residency.speculative_requests == 0
        && work.gpu_native_residency.speculative_vram_hits == 0
        && work.gpu_native_residency.speculative_ram_to_vram_installs == 0
        && work
            .gpu_native_residency
            .speculative_dropped_capacity_or_pressure
            == 0
        && work.engine_storage.prefetch_completed == 0
}

fn reconcile(control: &ArmReport, treatment: &ArmReport) -> Reconciliation {
    let c = control.work.as_ref().expect("complete control work");
    let t = treatment.work.as_ref().expect("complete treatment work");
    let cs = control.source.as_ref().expect("complete control source");
    let ts = treatment
        .source
        .as_ref()
        .expect("complete treatment source");
    let cws = control
        .warmup_source
        .as_ref()
        .expect("complete control warmup source");
    let tws = treatment
        .warmup_source
        .as_ref()
        .expect("complete treatment warmup source");
    let cww = control
        .warmup_work
        .as_ref()
        .expect("complete control warmup work");
    let tww = treatment
        .warmup_work
        .as_ref()
        .expect("complete treatment warmup work");
    let generated_tokens_exact = generated_results(control)
        .iter()
        .map(|run| run.generated_tokens)
        .eq(generated_results(treatment)
            .iter()
            .map(|run| run.generated_tokens));
    let generated_token_hashes_exact = generated_results(control)
        .iter()
        .map(|run| &run.generated_token_ids_sha256)
        .eq(generated_results(treatment)
            .iter()
            .map(|run| &run.generated_token_ids_sha256));
    let warmup_token_hashes_exact = control
        .warmup_results
        .iter()
        .map(|run| &run.generated_token_ids_sha256)
        .eq(treatment
            .warmup_results
            .iter()
            .map(|run| &run.generated_token_ids_sha256));
    let selected_route_sequence_exact =
        cs.selected_route_ids_sha256 == ts.selected_route_ids_sha256;
    let selected_route_counts_exact =
        c.routed_execution.selected_routed_experts == t.routed_execution.selected_routed_experts;
    let physical_missing_sequence_exact =
        cs.physical_missing_ids_sha256 == ts.physical_missing_ids_sha256;
    let physical_missing_counts_exact = cs.physical_missing_experts == ts.physical_missing_experts;
    let demand_source_request_sequence_exact =
        cs.demand_source_request_ids_sha256 == ts.demand_source_request_ids_sha256;
    let full_token_replay_zero = c.token_loop.replay_attempts == 0
        && t.token_loop.replay_attempts == 0
        && c.recovery.full_token_replay_attempts == 0
        && t.recovery.full_token_replay_attempts == 0;
    let fatal_and_no_progress_zero = c.token_loop.fatal_failures == 0
        && t.token_loop.fatal_failures == 0
        && c.token_loop.no_progress_failures == 0
        && t.token_loop.no_progress_failures == 0;
    let all_speculative_work_zero =
        arm_all_speculative_work_zero(c) && arm_all_speculative_work_zero(t);
    let warmup_all_speculative_work_zero =
        arm_all_speculative_work_zero(cww) && arm_all_speculative_work_zero(tww);
    let mut result = Reconciliation {
        generated_tokens_exact,
        generated_token_hashes_exact,
        warmup_token_hashes_exact,
        warmup_source_requests_exact: cws.demand_source_requests == tws.demand_source_requests,
        warmup_ram_source_hits_exact: cws.source_ram_hits == tws.source_ram_hits,
        warmup_ram_source_misses_exact: cws.source_ram_misses == tws.source_ram_misses,
        warmup_nvme_reads_exact: cws.source_nvme_reads == tws.source_nvme_reads,
        warmup_nvme_bytes_exact: cws.source_nvme_bytes == tws.source_nvme_bytes,
        warmup_ram_cache_inserts_exact: cws.ram_cache_inserts == tws.ram_cache_inserts,
        warmup_ram_cache_evictions_exact: cws.ram_cache_evictions == tws.ram_cache_evictions,
        warmup_ordered_ram_insert_ids_exact: cws.demand_ram_insert_ids_sha256
            == tws.demand_ram_insert_ids_sha256,
        warmup_ordered_ram_eviction_ids_exact: cws.demand_ram_eviction_ids_sha256
            == tws.demand_ram_eviction_ids_sha256,
        warmup_physical_missing_sequence_exact: cws.physical_missing_ids_sha256
            == tws.physical_missing_ids_sha256,
        warmup_demand_source_sequence_exact: cws.demand_source_request_ids_sha256
            == tws.demand_source_request_ids_sha256,
        warmup_ram_cache_state_exact: control.warmup_ram_cache_state_sha256
            == treatment.warmup_ram_cache_state_sha256,
        warmup_all_speculative_work_zero,
        selected_route_sequence_exact,
        selected_route_counts_exact,
        physical_missing_sequence_exact,
        physical_missing_counts_exact,
        demand_source_request_sequence_exact,
        demand_source_requests_exact: cs.demand_source_requests == ts.demand_source_requests,
        ram_source_hits_exact: cs.source_ram_hits == ts.source_ram_hits,
        ram_source_misses_exact: cs.source_ram_misses == ts.source_ram_misses,
        demand_nvme_reads_exact: cs.source_nvme_reads == ts.source_nvme_reads,
        demand_nvme_bytes_exact: cs.source_nvme_bytes == ts.source_nvme_bytes,
        logical_admissions_exact: c
            .gpu_native_residency
            .logical_admissions_for_physical_misses
            == t.gpu_native_residency
                .logical_admissions_for_physical_misses,
        ram_to_vram_installs_exact: c.gpu_native_residency.ram_to_vram_installs
            == t.gpu_native_residency.ram_to_vram_installs,
        ram_to_vram_bytes_exact: c.gpu_expert_io.expert_weight_upload_bytes
            == t.gpu_expert_io.expert_weight_upload_bytes,
        physical_evictions_exact: c.gpu_native_residency.physical_evictions
            == t.gpu_native_residency.physical_evictions,
        vram_hits_exact: c.gpu_native_residency.vram_hits == t.gpu_native_residency.vram_hits,
        vram_misses_exact: c.gpu_native_residency.vram_misses == t.gpu_native_residency.vram_misses,
        residency_miss_attempts_exact: c.token_loop.residency_miss_attempts
            == t.token_loop.residency_miss_attempts,
        residency_services_exact: c.token_loop.residency_services
            == t.token_loop.residency_services,
        recovery_segments_exact: c.recovery.recovery_segments == t.recovery.recovery_segments,
        miss_boundaries_exact: c.token_loop.residency_miss_attempts
            == t.token_loop.residency_miss_attempts
            && c.token_loop.residency_services == t.token_loop.residency_services,
        recovery_semantics_exact: recovery_semantics_equal(c.recovery, t.recovery),
        full_token_replay_zero,
        fatal_and_no_progress_zero,
        ram_cache_inserts_exact: cs.ram_cache_inserts == ts.ram_cache_inserts,
        ram_cache_evictions_exact: cs.ram_cache_evictions == ts.ram_cache_evictions,
        ordered_ram_insert_ids_exact: cs.demand_ram_insert_ids_sha256
            == ts.demand_ram_insert_ids_sha256,
        ordered_ram_eviction_ids_exact: cs.demand_ram_eviction_ids_sha256
            == ts.demand_ram_eviction_ids_sha256,
        all_speculative_work_zero,
        primary_pool_capacity_exact: cs.primary_pool_capacity == ts.primary_pool_capacity,
        all_invariants_pass: false,
    };
    result.all_invariants_pass = result.generated_tokens_exact
        && result.generated_token_hashes_exact
        && result.warmup_token_hashes_exact
        && result.warmup_source_requests_exact
        && result.warmup_ram_source_hits_exact
        && result.warmup_ram_source_misses_exact
        && result.warmup_nvme_reads_exact
        && result.warmup_nvme_bytes_exact
        && result.warmup_ram_cache_inserts_exact
        && result.warmup_ram_cache_evictions_exact
        && result.warmup_ordered_ram_insert_ids_exact
        && result.warmup_ordered_ram_eviction_ids_exact
        && result.warmup_physical_missing_sequence_exact
        && result.warmup_demand_source_sequence_exact
        && result.warmup_ram_cache_state_exact
        && result.warmup_all_speculative_work_zero
        && result.selected_route_sequence_exact
        && result.selected_route_counts_exact
        && result.physical_missing_sequence_exact
        && result.physical_missing_counts_exact
        && result.demand_source_request_sequence_exact
        && result.demand_source_requests_exact
        && result.ram_source_hits_exact
        && result.ram_source_misses_exact
        && result.demand_nvme_reads_exact
        && result.demand_nvme_bytes_exact
        && result.logical_admissions_exact
        && result.ram_to_vram_installs_exact
        && result.ram_to_vram_bytes_exact
        && result.physical_evictions_exact
        && result.vram_hits_exact
        && result.vram_misses_exact
        && result.residency_miss_attempts_exact
        && result.residency_services_exact
        && result.recovery_segments_exact
        && result.miss_boundaries_exact
        && result.recovery_semantics_exact
        && result.full_token_replay_zero
        && result.fatal_and_no_progress_zero
        && result.ram_cache_inserts_exact
        && result.ram_cache_evictions_exact
        && result.ordered_ram_insert_ids_exact
        && result.ordered_ram_eviction_ids_exact
        && result.all_speculative_work_zero
        && result.primary_pool_capacity_exact;
    result
}

fn common_gates(reconciliation: &Reconciliation) -> (BehavioralGate, WorkEquivalenceGate) {
    let behavioral_pass = reconciliation.generated_tokens_exact
        && reconciliation.generated_token_hashes_exact
        && reconciliation.selected_route_sequence_exact
        && reconciliation.selected_route_counts_exact
        && reconciliation.physical_missing_sequence_exact
        && reconciliation.physical_missing_counts_exact
        && reconciliation.full_token_replay_zero
        && reconciliation.fatal_and_no_progress_zero
        && reconciliation.all_speculative_work_zero;
    let work_pass = reconciliation.demand_nvme_bytes_exact
        && reconciliation.ram_to_vram_installs_exact
        && reconciliation.ram_to_vram_bytes_exact
        && reconciliation.physical_evictions_exact
        && reconciliation.miss_boundaries_exact
        && reconciliation.recovery_semantics_exact
        && reconciliation.ordered_ram_insert_ids_exact
        && reconciliation.ordered_ram_eviction_ids_exact;
    (
        BehavioralGate {
            generated_token_parity_exact: reconciliation.generated_tokens_exact
                && reconciliation.generated_token_hashes_exact,
            route_parity_exact: reconciliation.selected_route_sequence_exact
                && reconciliation.selected_route_counts_exact,
            physical_missing_sequence_exact: reconciliation.physical_missing_sequence_exact
                && reconciliation.physical_missing_counts_exact,
            full_token_replay_zero: reconciliation.full_token_replay_zero,
            fatal_and_no_progress_zero: reconciliation.fatal_and_no_progress_zero,
            speculation_zero: reconciliation.all_speculative_work_zero,
            passed: behavioral_pass,
        },
        WorkEquivalenceGate {
            demand_nvme_bytes_exact: reconciliation.demand_nvme_bytes_exact,
            demand_h2d_installs_exact: reconciliation.ram_to_vram_installs_exact,
            demand_h2d_bytes_exact: reconciliation.ram_to_vram_bytes_exact,
            physical_evictions_exact: reconciliation.physical_evictions_exact,
            recovery_and_miss_boundaries_exact: reconciliation.miss_boundaries_exact
                && reconciliation.recovery_semantics_exact,
            deterministic_ram_cache_order_exact: reconciliation.ordered_ram_insert_ids_exact
                && reconciliation.ordered_ram_eviction_ids_exact,
            passed: work_pass,
        },
    )
}

fn comparison(control: f64, treatment: f64) -> MetricComparison {
    MetricComparison {
        control,
        treatment,
        delta_percent: if control == 0.0 {
            0.0
        } else {
            (treatment - control) / control * 100.0
        },
    }
}

fn production_safety_zero(snapshot: &ProductionDemandSourceSnapshot) -> bool {
    snapshot.production_cache_reservation_leaks == 0
        && snapshot.production_batch_commit_violations == 0
        && snapshot.stale_singleflight_entries == 0
}

fn production_reconciliation(
    common: Reconciliation,
    control: &ArmReport,
    treatment: &ArmReport,
) -> ProductionReconciliation {
    let cwp = control
        .warmup_production
        .as_ref()
        .expect("complete v2 control warmup production telemetry");
    let twp = treatment
        .warmup_production
        .as_ref()
        .expect("complete v2 treatment warmup production telemetry");
    let cp = control
        .production
        .as_ref()
        .expect("complete v2 control production telemetry");
    let tp = treatment
        .production
        .as_ref()
        .expect("complete v2 treatment production telemetry");
    let warmup_production_cache_reservation_leaks_zero =
        cwp.production_cache_reservation_leaks == 0 && twp.production_cache_reservation_leaks == 0;
    let warmup_production_batch_commit_violations_zero =
        cwp.production_batch_commit_violations == 0 && twp.production_batch_commit_violations == 0;
    let warmup_stale_singleflight_entries_zero =
        cwp.stale_singleflight_entries == 0 && twp.stale_singleflight_entries == 0;
    let production_cache_reservation_leaks_zero =
        cp.production_cache_reservation_leaks == 0 && tp.production_cache_reservation_leaks == 0;
    let production_batch_commit_violations_zero =
        cp.production_batch_commit_violations == 0 && tp.production_batch_commit_violations == 0;
    let stale_singleflight_entries_zero =
        cp.stale_singleflight_entries == 0 && tp.stale_singleflight_entries == 0;
    let all_invariants_pass = common.all_invariants_pass
        && warmup_production_cache_reservation_leaks_zero
        && warmup_production_batch_commit_violations_zero
        && warmup_stale_singleflight_entries_zero
        && production_cache_reservation_leaks_zero
        && production_batch_commit_violations_zero
        && stale_singleflight_entries_zero;
    ProductionReconciliation {
        common,
        warmup_production_cache_reservation_leaks_zero,
        warmup_production_batch_commit_violations_zero,
        warmup_stale_singleflight_entries_zero,
        production_cache_reservation_leaks_zero,
        production_batch_commit_violations_zero,
        stale_singleflight_entries_zero,
        all_invariants_pass,
    }
}

fn production_gates(
    reconciliation: &ProductionReconciliation,
    control: &ArmReport,
    treatment: &ArmReport,
) -> ProductionGates {
    let (behavioral, work_equivalence) = common_gates(&reconciliation.common);
    let cp = control
        .production
        .as_ref()
        .expect("complete v2 control production telemetry");
    let tp = treatment
        .production
        .as_ref()
        .expect("complete v2 treatment production telemetry");
    let control_production_batch_successes_zero = cp.production_batch_successes == 0;
    let treatment_production_batch_successes_gt_zero = tp.production_batch_successes > 0;
    let treatment_production_batch_width_max_gt_one = tp.production_batch_width_max > 1;
    let treatment_ordinary_production_path_exercised = tp.ordinary_production_path_exercised;
    let treatment_cache_slots_exactly_reconciled = tp.production_cache_slots_reserved
        == tp
            .production_cache_reservations_consumed
            .saturating_add(tp.production_cache_reservations_released);
    let treatment_singleflight_claims_exactly_reconciled = tp.production_singleflight_ids_claimed
        == tp
            .production_batch_experts
            .saturating_add(tp.production_singleflight_claim_rollbacks);
    let passed = control_production_batch_successes_zero
        && treatment_production_batch_successes_gt_zero
        && treatment_production_batch_width_max_gt_one
        && treatment_ordinary_production_path_exercised
        && treatment_cache_slots_exactly_reconciled
        && treatment_singleflight_claims_exactly_reconciled
        && production_safety_zero(cp)
        && production_safety_zero(tp);
    ProductionGates {
        behavioral,
        work_equivalence,
        mechanism: ProductionMechanismGate {
            control_production_batch_successes_zero,
            treatment_production_batch_successes_gt_zero,
            treatment_production_batch_width_max_gt_one,
            treatment_ordinary_production_path_exercised,
            treatment_cache_slots_exactly_reconciled,
            treatment_singleflight_claims_exactly_reconciled,
            passed,
        },
    }
}

fn production_arm_performance(
    arm: &ArmReport,
) -> Result<ProductionArmPerformance, BenchmarkFailure> {
    let aggregate = arm.benchmark.aggregate.as_ref().ok_or_else(|| {
        BenchmarkFailure::new(
            "postcondition",
            "missing-arm-aggregate",
            "complete PR2-A.2 arm did not produce a benchmark aggregate",
        )
    })?;
    let runs = generated_results(arm);
    let mean_request_wall_seconds = runs
        .iter()
        .map(|run| run.timing.end_to_end_seconds)
        .sum::<f64>()
        / runs.len() as f64;
    let source = arm.source.as_ref().expect("complete arm source");
    Ok(ProductionArmPerformance {
        decode_tps: aggregate.decode_tps.mean,
        end_to_end_generated_tps: aggregate.end_to_end_generated_tps.mean,
        mean_request_wall_seconds,
        source_acquisition_wall_us: source.source_acquisition_wall_us,
        logical_demand_admission_us: source.logical_demand_admission_us,
        physical_demand_install_us: source.physical_demand_install_us,
        total_residency_service_us: source.total_residency_service_us,
    })
}

fn production_performance(
    control: &ArmReport,
    treatment: &ArmReport,
) -> Result<ProductionPerformanceComparison, BenchmarkFailure> {
    let control = production_arm_performance(control)?;
    let treatment = production_arm_performance(treatment)?;
    let performance_result = if treatment.decode_tps > control.decode_tps
        && treatment.source_acquisition_wall_us < control.source_acquisition_wall_us
        && treatment.total_residency_service_us < control.total_residency_service_us
    {
        "improved"
    } else if treatment.decode_tps < control.decode_tps
        && treatment.source_acquisition_wall_us > control.source_acquisition_wall_us
    {
        "regressed"
    } else {
        "mixed_or_no_improvement"
    };
    Ok(ProductionPerformanceComparison {
        decode_tps: comparison(control.decode_tps, treatment.decode_tps),
        end_to_end_generated_tps: comparison(
            control.end_to_end_generated_tps,
            treatment.end_to_end_generated_tps,
        ),
        mean_request_wall_seconds: comparison(
            control.mean_request_wall_seconds,
            treatment.mean_request_wall_seconds,
        ),
        source_acquisition_wall_us: comparison(
            control.source_acquisition_wall_us as f64,
            treatment.source_acquisition_wall_us as f64,
        ),
        logical_demand_admission_us: comparison(
            control.logical_demand_admission_us as f64,
            treatment.logical_demand_admission_us as f64,
        ),
        physical_demand_install_us: comparison(
            control.physical_demand_install_us as f64,
            treatment.physical_demand_install_us as f64,
        ),
        total_residency_service_us: comparison(
            control.total_residency_service_us as f64,
            treatment.total_residency_service_us as f64,
        ),
        control,
        treatment,
        performance_result,
    })
}

fn emit_report<T: Serialize>(
    report: &T,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut json = serde_json::to_vec_pretty(report)?;
    json.push(b'\n');
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, json)?;
    eprintln!("PR2-A qualification report written to {}", path.display());
    Ok(())
}

pub(crate) async fn run_production_command(
    args: CommandArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let prepared = prepare(&args)?;
    let mut report = ProductionQualificationReport {
        schema: PRODUCTION_SCHEMA,
        mode: PRODUCTION_MODE,
        production_demand_source_changed: true,
        control_forces_legacy_sequential: true,
        treatment_uses_ordinary_production_path: true,
        benchmark_complete: false,
        qualification_pass: false,
        performance_result: "not_measured",
        failure: None,
        frozen_workload: frozen_workload(args.expected_adapter_name.clone()),
        provenance: prepared.provenance.clone(),
        control: None,
        treatment: None,
        reconciliation: None,
        gates: None,
        performance: None,
    };

    let control = match run_arm(
        &prepared,
        &args,
        GpuNativeDemandSourceQualificationArm::Control,
    )
    .await
    {
        Ok(control) => control,
        Err(failure) => {
            report.failure = Some(failure.clone());
            emit_report(&report, &args.report_out)?;
            return Err(failure.to_string().into());
        }
    };
    let control_failure = control.failure.clone();
    report.control = Some(control);
    if let Some(failure) = control_failure {
        report.failure = Some(failure.clone());
        emit_report(&report, &args.report_out)?;
        return Err(failure.to_string().into());
    }

    let treatment = match run_arm(
        &prepared,
        &args,
        GpuNativeDemandSourceQualificationArm::Treatment,
    )
    .await
    {
        Ok(treatment) => treatment,
        Err(failure) => {
            report.failure = Some(failure.clone());
            emit_report(&report, &args.report_out)?;
            return Err(failure.to_string().into());
        }
    };
    let treatment_failure = treatment.failure.clone();
    report.treatment = Some(treatment);
    if let Some(failure) = treatment_failure {
        report.failure = Some(failure.clone());
        emit_report(&report, &args.report_out)?;
        return Err(failure.to_string().into());
    }

    let control = report.control.as_ref().expect("control stored");
    let treatment = report.treatment.as_ref().expect("treatment stored");
    let reconciliation =
        production_reconciliation(reconcile(control, treatment), control, treatment);
    let gates = production_gates(&reconciliation, control, treatment);
    let performance = match production_performance(control, treatment) {
        Ok(performance) => performance,
        Err(failure) => {
            report.failure = Some(failure.clone());
            emit_report(&report, &args.report_out)?;
            return Err(failure.to_string().into());
        }
    };
    let qualification_pass = reconciliation.all_invariants_pass
        && gates.behavioral.passed
        && gates.work_equivalence.passed
        && gates.mechanism.passed;
    report.benchmark_complete = true;
    report.qualification_pass = qualification_pass;
    report.performance_result = performance.performance_result;
    report.reconciliation = Some(reconciliation);
    report.gates = Some(gates);
    report.performance = Some(performance);
    emit_report(&report, &args.report_out)?;
    if qualification_pass {
        Ok(())
    } else {
        Err("PR2-A.2 production qualification gates did not all pass; see emitted report".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn isolation_config() -> crate::config::Config {
        crate::config::Config {
            server: crate::config::ServerConfig {
                bind: "127.0.0.1:8080".into(),
                max_tokens: 128,
                session_ttl_secs: 0,
                max_concurrent_requests: 0,
                admission_min_free_blocks: 0,
            },
            performance: crate::config::PerformanceConfig::default(),
            model: crate::config::ModelConfig {
                data_dir: PathBuf::from("./data"),
                num_experts: 8,
                top_k: 2,
                d_model: 64,
                d_ff: 256,
                expert_size: 4096,
                num_layers: 1,
                dtype: crate::inference::WeightDtype::F32,
            },
            storage: crate::config::StorageConfigToml {
                cache_slots: 4,
                block_align: 4096,
                no_direct: false,
                predict_fanout: 0,
                pipeline_depth: crate::engine::DEFAULT_PIPELINE_DEPTH,
                predict_min_prob: 0.0,
                partial_load_fraction: 1.0,
                pin_after_observations: 0,
                packed_blob: None,
                packed_manifest: None,
            },
            tokenizer: crate::config::TokenizerConfig::default(),
            real_transformer: crate::config::RealTransformerConfig::default(),
            sampling: crate::config::SamplingConfig::default(),
            predictive: crate::config::PredictiveConfig::default(),
            security: crate::config::SecurityConfig::default(),
            gpu_cache: crate::config::GpuCacheConfig::default(),
            distributed: crate::config::DistributedConfig::default(),
        }
    }

    #[test]
    fn frozen_workload_contract_is_literal() {
        assert_eq!(
            PRODUCTION_SCHEMA,
            "mer.gpu-native-demand-source-concurrency.v2"
        );
        assert_eq!(
            PRODUCTION_MODE,
            "qualify-gpu-native-demand-source-concurrency-production"
        );
        assert_eq!(FROZEN_OUTPUT_TOKENS, 128);
        assert_eq!(FROZEN_WARMUP_RUNS, 1);
        assert_eq!(FROZEN_MEASURED_RUNS, 3);
        assert_eq!(
            FROZEN_PROMPT,
            "Write a Rust function that adds two i32 values and returns the result."
        );
    }

    #[test]
    fn delta_percent_preserves_direction() {
        assert_eq!(comparison(10.0, 12.0).delta_percent, 20.0);
        assert_eq!(comparison(10.0, 8.0).delta_percent, -20.0);
        assert_eq!(comparison(0.0, 8.0).delta_percent, 0.0);
    }

    #[test]
    fn preflight_rejects_source_mechanisms_that_confound_pr2a() {
        let cfg = isolation_config();
        validate_isolation_config(&cfg).expect("literal PR2-A isolation config");

        let mut observation_pinning = cfg.clone();
        observation_pinning.storage.pin_after_observations = 1;
        assert_eq!(
            validate_isolation_config(&observation_pinning)
                .unwrap_err()
                .code,
            "observation-pinning-must-be-disabled"
        );

        let mut cost_aware = cfg.clone();
        cost_aware.predictive.cost_aware_eviction = true;
        assert_eq!(
            validate_isolation_config(&cost_aware).unwrap_err().code,
            "cost-aware-eviction-must-be-disabled"
        );

        let mut packed_blob = cfg.clone();
        packed_blob.storage.packed_blob = Some(PathBuf::from("experts.bin"));
        assert_eq!(
            validate_isolation_config(&packed_blob).unwrap_err().code,
            "packed-blob-must-be-disabled"
        );

        let mut packed_manifest = cfg;
        packed_manifest.storage.packed_manifest = Some(PathBuf::from("experts.json"));
        assert_eq!(
            validate_isolation_config(&packed_manifest)
                .unwrap_err()
                .code,
            "packed-manifest-must-be-disabled"
        );
    }
}
