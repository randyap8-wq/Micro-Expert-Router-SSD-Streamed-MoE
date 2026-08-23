//! Fail-closed accounting for full GPU-native Q4 greedy-token qualification.
//!
//! Runtime construction remains in `main.rs`. This module owns the versioned
//! report, fixed semantic contracts, exact token/routing comparisons, and
//! hardware-independent PASS derivation.

use crate::backend::GpuDeviceIdentity;
use crate::engine::RoutedExpertExecutionSnapshot;
use crate::gpu_native_token_loop::{GpuNativeModelGeometry, GpuNativeTokenLoopSnapshot};
use crate::greedy_parity::{
    BackgroundShutdownEvidence, CorpusEvidence, GenerationEvidence, GreedySamplingEvidence,
    ModelLoadEvidence, PlaneRunEvidence,
};
use crate::qualification::{
    BuildProvenance, ExpertMetadataEvidence, FailureStage, QualificationArtifacts,
    QualificationFailure,
};
use serde::Serialize;

pub const SCHEMA_VERSION: &str = "mer.gpu-native-q4-greedy-parity.v2";
pub const MODE: &str = "gpu-native-q4-greedy-parity-qualification";

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReferenceContractEvidence {
    pub execution: &'static str,
    pub routed_expert_boundary: &'static str,
    pub routing_and_aggregation: &'static str,
    pub strict_real_checkpoint: bool,
    pub seeded_fallback_allowed: bool,
    pub degraded_experts_allowed: bool,
}

impl ReferenceContractEvidence {
    pub const fn authoritative() -> Self {
        Self {
            execution: "cpu",
            routed_expert_boundary: "production-hybrid-f16-input-output",
            routing_and_aggregation: "f32",
            strict_real_checkpoint: true,
            seeded_fallback_allowed: false,
            degraded_experts_allowed: false,
        }
    }

    fn is_exact(&self) -> bool {
        self == &Self::authoritative()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GpuCandidatePathEvidence {
    pub implementation: &'static str,
    pub real_dense_weights: bool,
    pub gpu_rmsnorm: bool,
    pub gpu_attention: bool,
    pub gpu_kv: bool,
    pub gpu_router: bool,
    pub q4_gpu_experts: bool,
    pub gpu_combine: bool,
    pub final_rmsnorm: bool,
    pub lm_head: bool,
    pub gpu_greedy_argmax: bool,
    pub tiered_residency_retry: bool,
}

impl GpuCandidatePathEvidence {
    pub const fn production() -> Self {
        Self {
            implementation: "GpuNativeTokenLoop",
            real_dense_weights: true,
            gpu_rmsnorm: true,
            gpu_attention: true,
            gpu_kv: true,
            gpu_router: true,
            q4_gpu_experts: true,
            gpu_combine: true,
            final_rmsnorm: true,
            lm_head: true,
            gpu_greedy_argmax: true,
            tiered_residency_retry: true,
        }
    }

    fn is_exact(&self) -> bool {
        self == &Self::production()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SourceConfigEvidence {
    pub real_transformer_enabled: bool,
    pub gpu_native_enabled: bool,
    pub weights_dir_configured: bool,
    pub strict_weights: bool,
    pub allow_seeded_fallback: bool,
    pub allow_degraded_experts: bool,
    pub allow_nonfinite_attention_fallback: bool,
    pub allow_truncated_expert_payloads: bool,
    pub distributed_enabled: bool,
    pub gpu_cache_enabled: bool,
    pub gpu_expert_capacity_bytes: u64,
    pub routed_expert_dtype: String,
}

impl SourceConfigEvidence {
    pub fn is_strict(&self) -> bool {
        self.real_transformer_enabled
            && self.gpu_native_enabled
            && self.weights_dir_configured
            && self.strict_weights
            && !self.allow_seeded_fallback
            && !self.allow_degraded_experts
            && !self.allow_nonfinite_attention_fallback
            && !self.allow_truncated_expert_payloads
            && !self.distributed_enabled
            && self.gpu_cache_enabled
            && self.gpu_expert_capacity_bytes > 0
            && self.routed_expert_dtype == "q4_0"
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct TokenMismatchEvidence {
    pub case: String,
    pub generated_position: usize,
    pub reference_token_id: Option<u32>,
    pub gpu_native_token_id: Option<u32>,
    pub prompt_token_ids: Vec<u32>,
    pub preceding_generated_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExpertIdSetMismatchEvidence {
    pub case: String,
    pub generated_position: usize,
    pub layer: usize,
    pub reference_selected_expert_ids: Vec<u32>,
    pub gpu_native_selected_expert_ids: Vec<u32>,
    pub canonical_reference_expert_ids: Vec<u32>,
    pub canonical_gpu_native_expert_ids: Vec<u32>,
    pub reference_only_expert_ids: Vec<u32>,
    pub gpu_native_only_expert_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExpertIdRankOrderDriftEvidence {
    pub case: String,
    pub generated_position: usize,
    pub layer: usize,
    pub first_differing_rank_slot: usize,
    pub reference_selected_expert_ids: Vec<u32>,
    pub gpu_native_selected_expert_ids: Vec<u32>,
    pub canonical_reference_expert_ids: Vec<u32>,
    pub canonical_gpu_native_expert_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CaseComparisonEvidence {
    pub exact_token_matches: usize,
    pub token_ids_match: bool,
    pub first_token_mismatch: Option<TokenMismatchEvidence>,
    pub expert_id_selection_set_match: bool,
    pub first_expert_id_set_mismatch: Option<ExpertIdSetMismatchEvidence>,
    pub expert_id_rank_order_match: bool,
    pub rank_order_drift_count: usize,
    pub first_expert_id_rank_order_drift: Option<ExpertIdRankOrderDriftEvidence>,
    pub routing_positions_compared: usize,
    pub routing_layer_comparisons: usize,
}

pub fn compare_case_outputs(
    case: &str,
    prompt_token_ids: &[u32],
    reference_token_ids: &[u32],
    gpu_native_token_ids: &[u32],
    reference_routes: &[Vec<Vec<u32>>],
    gpu_native_routes: &[Vec<Vec<u32>>],
    tokens_per_case: usize,
    num_layers: usize,
    top_k: usize,
    num_experts: Option<usize>,
) -> Result<CaseComparisonEvidence, String> {
    if tokens_per_case == 0 {
        return Err("tokens_per_case must be greater than zero".to_string());
    }
    if reference_token_ids.len() != tokens_per_case || gpu_native_token_ids.len() != tokens_per_case
    {
        return Err(format!(
            "case {case} expected {tokens_per_case} tokens per plane, got reference={} gpu_native={}",
            reference_token_ids.len(),
            gpu_native_token_ids.len()
        ));
    }
    validate_route_shape(
        case,
        "reference",
        reference_routes,
        tokens_per_case,
        num_layers,
        top_k,
        num_experts,
    )?;
    validate_route_shape(
        case,
        "gpu_native",
        gpu_native_routes,
        tokens_per_case,
        num_layers,
        top_k,
        num_experts,
    )?;

    let exact_token_matches = reference_token_ids
        .iter()
        .zip(gpu_native_token_ids)
        .filter(|(reference, gpu)| reference == gpu)
        .count();
    let first_token_mismatch = reference_token_ids
        .iter()
        .zip(gpu_native_token_ids)
        .position(|(reference, gpu)| reference != gpu)
        .map(|generated_position| TokenMismatchEvidence {
            case: case.to_string(),
            generated_position,
            reference_token_id: reference_token_ids.get(generated_position).copied(),
            gpu_native_token_id: gpu_native_token_ids.get(generated_position).copied(),
            prompt_token_ids: prompt_token_ids.to_vec(),
            preceding_generated_ids: reference_token_ids[..generated_position].to_vec(),
        });

    let mut first_expert_id_set_mismatch = None;
    let mut first_expert_id_rank_order_drift = None;
    let mut rank_order_drift_count = 0usize;
    let mut expert_id_rank_order_match = true;
    for generated_position in 0..tokens_per_case {
        for layer in 0..num_layers {
            let reference = &reference_routes[generated_position][layer];
            let gpu_native = &gpu_native_routes[generated_position][layer];
            if reference == gpu_native {
                continue;
            }
            expert_id_rank_order_match = false;
            let mut canonical_reference = reference.clone();
            canonical_reference.sort_unstable();
            let mut canonical_gpu_native = gpu_native.clone();
            canonical_gpu_native.sort_unstable();
            if canonical_reference == canonical_gpu_native {
                rank_order_drift_count = rank_order_drift_count
                    .checked_add(1)
                    .ok_or("rank_order_drift_count overflowed")?;
                if first_expert_id_rank_order_drift.is_none() {
                    let first_differing_rank_slot = reference
                        .iter()
                        .zip(gpu_native)
                        .position(|(reference_id, gpu_id)| reference_id != gpu_id)
                        .expect("different equal-length route lists have a differing slot");
                    first_expert_id_rank_order_drift = Some(ExpertIdRankOrderDriftEvidence {
                        case: case.to_string(),
                        generated_position,
                        layer,
                        first_differing_rank_slot,
                        reference_selected_expert_ids: reference.clone(),
                        gpu_native_selected_expert_ids: gpu_native.clone(),
                        canonical_reference_expert_ids: canonical_reference,
                        canonical_gpu_native_expert_ids: canonical_gpu_native,
                    });
                }
            } else if first_expert_id_set_mismatch.is_none() {
                let reference_only_expert_ids = canonical_reference
                    .iter()
                    .copied()
                    .filter(|id| canonical_gpu_native.binary_search(id).is_err())
                    .collect();
                let gpu_native_only_expert_ids = canonical_gpu_native
                    .iter()
                    .copied()
                    .filter(|id| canonical_reference.binary_search(id).is_err())
                    .collect();
                first_expert_id_set_mismatch = Some(ExpertIdSetMismatchEvidence {
                    case: case.to_string(),
                    generated_position,
                    layer,
                    reference_selected_expert_ids: reference.clone(),
                    gpu_native_selected_expert_ids: gpu_native.clone(),
                    canonical_reference_expert_ids: canonical_reference,
                    canonical_gpu_native_expert_ids: canonical_gpu_native,
                    reference_only_expert_ids,
                    gpu_native_only_expert_ids,
                });
            }
        }
    }

    Ok(CaseComparisonEvidence {
        exact_token_matches,
        token_ids_match: first_token_mismatch.is_none(),
        first_token_mismatch,
        expert_id_selection_set_match: first_expert_id_set_mismatch.is_none(),
        first_expert_id_set_mismatch,
        expert_id_rank_order_match,
        rank_order_drift_count,
        first_expert_id_rank_order_drift,
        routing_positions_compared: tokens_per_case,
        routing_layer_comparisons: tokens_per_case
            .checked_mul(num_layers)
            .ok_or("routing comparison count overflowed")?,
    })
}

fn validate_route_shape(
    case: &str,
    plane: &str,
    routes: &[Vec<Vec<u32>>],
    tokens_per_case: usize,
    num_layers: usize,
    top_k: usize,
    num_experts: Option<usize>,
) -> Result<(), String> {
    if routes.len() != tokens_per_case {
        return Err(format!(
            "case {case} {plane} routing evidence has {} positions, expected {tokens_per_case}",
            routes.len()
        ));
    }
    for (position, layers) in routes.iter().enumerate() {
        if layers.len() != num_layers {
            return Err(format!(
                "case {case} {plane} position {position} has {} routed layers, expected {num_layers}",
                layers.len()
            ));
        }
        for (layer, selected) in layers.iter().enumerate() {
            if selected.len() != top_k {
                return Err(format!(
                    "case {case} {plane} position {position} layer {layer} selected {} experts, expected {top_k}",
                    selected.len()
                ));
            }
            let mut canonical_selected = selected.clone();
            canonical_selected.sort_unstable();
            if let Some(duplicate) = canonical_selected
                .windows(2)
                .find(|pair| pair[0] == pair[1])
                .map(|pair| pair[0])
            {
                return Err(format!(
                    "case {case} {plane} position {position} layer {layer} selected duplicate expert ID {duplicate}"
                ));
            }
            if let Some(num_experts) = num_experts {
                if let Some(invalid) = canonical_selected
                    .iter()
                    .copied()
                    .find(|&id| id as usize >= num_experts)
                {
                    return Err(format!(
                        "case {case} {plane} position {position} layer {layer} selected expert ID {invalid}, outside num_experts={num_experts}"
                    ));
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ReferenceRoutingReplayEvidence {
    pub generation: GenerationEvidence,
    pub matches_authoritative_reference: bool,
    pub routing_positions_captured: usize,
    pub routed_layers_per_position: usize,
    pub selected_expert_ids_by_generated_position_layer: Vec<Vec<Vec<u32>>>,
    pub background_shutdown: BackgroundShutdownEvidence,
}

#[derive(Clone, Debug, Serialize)]
pub struct GpuCandidateCaseEvidence {
    pub generation: GenerationEvidence,
    pub model_load: ModelLoadEvidence,
    pub selected_expert_ids_by_generated_position_layer: Vec<Vec<Vec<u32>>>,
    pub token_loop_counters_delta: GpuNativeTokenLoopSnapshot,
    pub routed_execution_delta: RoutedExpertExecutionSnapshot,
    pub attention_softmax_nonfinite_fallbacks: u64,
    pub request_completed_expected_tokens: bool,
    pub background_shutdown: BackgroundShutdownEvidence,
}

#[derive(Clone, Debug, Serialize)]
pub struct CaseReport {
    pub name: String,
    pub behavior: String,
    pub prompt: String,
    pub prompt_sha256: String,
    pub prompt_token_ids: Vec<u32>,
    pub prompt_token_ids_sha256: String,
    pub tokens_per_case: usize,
    pub authoritative_reference: PlaneRunEvidence,
    pub reference_routing_replay: ReferenceRoutingReplayEvidence,
    pub gpu_native: GpuCandidateCaseEvidence,
    pub comparison: CaseComparisonEvidence,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct FallbackDegradedEvidence {
    pub all_model_loads_strict_complete: bool,
    pub seeded_fallback_remained: bool,
    pub degraded_expert_substitutions: u64,
    pub gpu_dispatch_failures: u64,
    pub gpu_cpu_fallbacks: u64,
    pub nonfinite_attention_fallbacks: u64,
    pub fatal_numerical_failures: u64,
    pub no_progress_failures: u64,
    pub unexpected_request_retirements: u64,
    pub all_requests_completed_expected_tokens: bool,
}

impl FallbackDegradedEvidence {
    fn passes(&self, source: &SourceConfigEvidence) -> bool {
        source.is_strict()
            && self.all_model_loads_strict_complete
            && !self.seeded_fallback_remained
            && self.degraded_expert_substitutions == 0
            && self.gpu_dispatch_failures == 0
            && self.gpu_cpu_fallbacks == 0
            && self.nonfinite_attention_fallbacks == 0
            && self.fatal_numerical_failures == 0
            && self.no_progress_failures == 0
            && self.unexpected_request_retirements == 0
            && self.all_requests_completed_expected_tokens
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ResidencyEvidence {
    pub miss_attempts: u64,
    pub replay_attempts: u64,
    pub services: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct QualificationChecks {
    pub clean_build_provenance: bool,
    pub source_config_strict: bool,
    pub fixed_corpus_identity: bool,
    pub fixed_greedy_sampling: bool,
    pub authoritative_reference_contract: bool,
    pub production_gpu_candidate_path: bool,
    pub model_geometry_recorded: bool,
    pub adapter_exact_match: bool,
    pub hardware_adapter: bool,
    pub exact_case_and_token_totals: bool,
    pub exact_generated_token_ids: bool,
    pub expert_id_selection_set_match: bool,
    pub reference_routing_replays_match: bool,
    pub residency_misses_fully_serviced: bool,
    pub zero_fallback_degraded_or_fatal_evidence: bool,
}

impl QualificationChecks {
    fn passes(&self) -> bool {
        self.clean_build_provenance
            && self.source_config_strict
            && self.fixed_corpus_identity
            && self.fixed_greedy_sampling
            && self.authoritative_reference_contract
            && self.production_gpu_candidate_path
            && self.model_geometry_recorded
            && self.adapter_exact_match
            && self.hardware_adapter
            && self.exact_case_and_token_totals
            && self.exact_generated_token_ids
            && self.expert_id_selection_set_match
            && self.reference_routing_replays_match
            && self.residency_misses_fully_serviced
            && self.zero_fallback_degraded_or_fatal_evidence
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct GpuNativeGreedyParityReport {
    pub schema_version: &'static str,
    pub mode: &'static str,
    pub qualification_pass: bool,
    pub failure: Option<QualificationFailure>,
    pub build_provenance: BuildProvenance,
    pub artifacts: QualificationArtifacts,
    pub expert_metadata: Option<ExpertMetadataEvidence>,
    pub resolved_config_sha256: Option<String>,
    pub reference_resolved_config_sha256: Option<String>,
    pub model_geometry: Option<GpuNativeModelGeometry>,
    pub expected_adapter_name: String,
    pub actual_adapter: Option<GpuDeviceIdentity>,
    pub reference_contract: ReferenceContractEvidence,
    pub gpu_candidate_path: GpuCandidatePathEvidence,
    pub source_config: SourceConfigEvidence,
    pub corpus: CorpusEvidence,
    pub sampling: GreedySamplingEvidence,
    pub cases: Vec<CaseReport>,
    pub tokens_per_case: usize,
    pub total_tokens_expected: usize,
    pub total_tokens_compared: usize,
    pub exact_token_matches: usize,
    pub first_token_mismatch: Option<TokenMismatchEvidence>,
    pub expert_id_selection_set_match: bool,
    pub first_expert_id_set_mismatch: Option<ExpertIdSetMismatchEvidence>,
    pub expert_id_rank_order_match: bool,
    pub rank_order_drift_count: usize,
    pub first_expert_id_rank_order_drift: Option<ExpertIdRankOrderDriftEvidence>,
    pub routing_positions_compared: usize,
    pub routing_layer_comparisons: usize,
    pub fallback_degraded_evidence: FallbackDegradedEvidence,
    pub token_loop_counter_deltas: GpuNativeTokenLoopSnapshot,
    pub residency: ResidencyEvidence,
    pub checks: QualificationChecks,
}

impl GpuNativeGreedyParityReport {
    pub fn new(
        build_provenance: BuildProvenance,
        artifacts: QualificationArtifacts,
        expert_metadata: Option<ExpertMetadataEvidence>,
        expected_adapter_name: String,
        source_config: SourceConfigEvidence,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            mode: MODE,
            qualification_pass: false,
            failure: None,
            build_provenance,
            artifacts,
            expert_metadata,
            resolved_config_sha256: None,
            reference_resolved_config_sha256: None,
            model_geometry: None,
            expected_adapter_name,
            actual_adapter: None,
            reference_contract: ReferenceContractEvidence::authoritative(),
            gpu_candidate_path: GpuCandidatePathEvidence::production(),
            source_config,
            corpus: CorpusEvidence::fixed(),
            sampling: GreedySamplingEvidence::fixed(),
            cases: Vec::new(),
            tokens_per_case: crate::greedy_parity::OUTPUT_TOKEN_LIMIT,
            total_tokens_expected: crate::greedy_parity::CORPUS_CASE_COUNT
                * crate::greedy_parity::OUTPUT_TOKEN_LIMIT,
            total_tokens_compared: 0,
            exact_token_matches: 0,
            first_token_mismatch: None,
            expert_id_selection_set_match: true,
            first_expert_id_set_mismatch: None,
            expert_id_rank_order_match: true,
            rank_order_drift_count: 0,
            first_expert_id_rank_order_drift: None,
            routing_positions_compared: 0,
            routing_layer_comparisons: 0,
            fallback_degraded_evidence: FallbackDegradedEvidence {
                all_model_loads_strict_complete: true,
                all_requests_completed_expected_tokens: true,
                ..FallbackDegradedEvidence::default()
            },
            token_loop_counter_deltas: GpuNativeTokenLoopSnapshot::default(),
            residency: ResidencyEvidence::default(),
            checks: QualificationChecks::default(),
        }
    }

    pub fn fail(&mut self, failure: QualificationFailure) {
        self.derive_checks();
        self.qualification_pass = false;
        self.failure = Some(failure);
    }

    pub fn record_case(&mut self, case: CaseReport) -> Result<(), String> {
        let expected = crate::greedy_parity::FIXED_CORPUS
            .get(self.cases.len())
            .ok_or("qualification received more cases than the fixed corpus")?;
        if case.name != expected.name {
            return Err(format!(
                "fixed corpus case order mismatch: expected {}, got {}",
                expected.name, case.name
            ));
        }
        let geometry = self
            .model_geometry
            .ok_or("model geometry must be recorded before case evidence")?;
        if case.behavior != expected.behavior
            || case.prompt != expected.prompt
            || case.prompt_sha256 != crate::greedy_parity::sha256_hex(expected.prompt.as_bytes())
            || case.prompt_token_ids.is_empty()
            || case.prompt_token_ids_sha256
                != crate::greedy_parity::token_ids_sha256(&case.prompt_token_ids)
            || case.tokens_per_case != self.tokens_per_case
            || case
                .authoritative_reference
                .generation
                .generated_token_count
                != self.tokens_per_case
            || case.gpu_native.generation.generated_token_count != self.tokens_per_case
            || case
                .reference_routing_replay
                .generation
                .generated_token_count
                != self.tokens_per_case
            || case.reference_routing_replay.routing_positions_captured != self.tokens_per_case
        {
            return Err(format!(
                "fixed corpus case {} has invalid identity, token, or routing evidence",
                case.name
            ));
        }
        let recomputed_comparison = compare_case_outputs(
            &case.name,
            &case.prompt_token_ids,
            &case.authoritative_reference.generation.generated_token_ids,
            &case.gpu_native.generation.generated_token_ids,
            &case
                .reference_routing_replay
                .selected_expert_ids_by_generated_position_layer,
            &case
                .gpu_native
                .selected_expert_ids_by_generated_position_layer,
            self.tokens_per_case,
            geometry.num_layers,
            geometry.top_k,
            Some(geometry.num_experts),
        )?;
        if recomputed_comparison != case.comparison {
            return Err(format!(
                "fixed corpus case {} comparison does not reconcile with its exact token and expert-ID evidence",
                case.name
            ));
        }
        self.total_tokens_compared = self
            .total_tokens_compared
            .checked_add(case.tokens_per_case)
            .ok_or("total_tokens_compared overflowed")?;
        self.exact_token_matches = self
            .exact_token_matches
            .checked_add(case.comparison.exact_token_matches)
            .ok_or("exact_token_matches overflowed")?;
        self.routing_positions_compared = self
            .routing_positions_compared
            .checked_add(case.comparison.routing_positions_compared)
            .ok_or("routing_positions_compared overflowed")?;
        self.routing_layer_comparisons = self
            .routing_layer_comparisons
            .checked_add(case.comparison.routing_layer_comparisons)
            .ok_or("routing_layer_comparisons overflowed")?;
        if self.first_token_mismatch.is_none() {
            self.first_token_mismatch = case.comparison.first_token_mismatch.clone();
        }
        if self.first_expert_id_set_mismatch.is_none() {
            self.first_expert_id_set_mismatch =
                case.comparison.first_expert_id_set_mismatch.clone();
        }
        self.expert_id_selection_set_match &= case.comparison.expert_id_selection_set_match;
        if self.first_expert_id_rank_order_drift.is_none() {
            self.first_expert_id_rank_order_drift =
                case.comparison.first_expert_id_rank_order_drift.clone();
        }
        self.expert_id_rank_order_match &= case.comparison.expert_id_rank_order_match;
        self.rank_order_drift_count = self
            .rank_order_drift_count
            .checked_add(case.comparison.rank_order_drift_count)
            .ok_or("rank_order_drift_count overflowed")?;
        self.accumulate_fallback_evidence(&case)?;
        self.token_loop_counter_deltas = checked_add_snapshots(
            self.token_loop_counter_deltas,
            case.gpu_native.token_loop_counters_delta,
        )?;
        self.residency = ResidencyEvidence {
            miss_attempts: self.token_loop_counter_deltas.residency_miss_attempts,
            replay_attempts: self.token_loop_counter_deltas.replay_attempts,
            services: self.token_loop_counter_deltas.residency_services,
        };
        self.cases.push(case);
        Ok(())
    }

    fn accumulate_fallback_evidence(&mut self, case: &CaseReport) -> Result<(), String> {
        let reference = &case.authoritative_reference;
        let gpu = &case.gpu_native;
        self.fallback_degraded_evidence
            .all_model_loads_strict_complete &= model_load_strict_complete(&reference.model_load)
            && model_load_strict_complete(&gpu.model_load);
        self.fallback_degraded_evidence.seeded_fallback_remained |=
            reference.model_load.seeded_fallback_remained
                || gpu.model_load.seeded_fallback_remained;
        self.fallback_degraded_evidence
            .degraded_expert_substitutions = checked_add(
            self.fallback_degraded_evidence
                .degraded_expert_substitutions,
            reference
                .routed_execution_delta
                .degraded_expert_substitutions,
            "reference degraded expert substitutions",
        )?;
        self.fallback_degraded_evidence.gpu_dispatch_failures = checked_add(
            self.fallback_degraded_evidence.gpu_dispatch_failures,
            reference.routed_execution_delta.gpu_dispatch_failures,
            "reference GPU dispatch failures",
        )?;
        self.fallback_degraded_evidence.gpu_dispatch_failures = checked_add(
            self.fallback_degraded_evidence.gpu_dispatch_failures,
            gpu.routed_execution_delta.gpu_dispatch_failures,
            "GPU-native GPU dispatch failures",
        )?;
        self.fallback_degraded_evidence.gpu_cpu_fallbacks = checked_add(
            self.fallback_degraded_evidence.gpu_cpu_fallbacks,
            reference.routed_execution_delta.gpu_cpu_fallbacks,
            "reference GPU-to-CPU fallbacks",
        )?;
        self.fallback_degraded_evidence.gpu_cpu_fallbacks = checked_add(
            self.fallback_degraded_evidence.gpu_cpu_fallbacks,
            gpu.routed_execution_delta.gpu_cpu_fallbacks,
            "GPU-native GPU-to-CPU fallbacks",
        )?;
        self.fallback_degraded_evidence
            .degraded_expert_substitutions = checked_add(
            self.fallback_degraded_evidence
                .degraded_expert_substitutions,
            gpu.routed_execution_delta.degraded_expert_substitutions,
            "GPU-native degraded expert substitutions",
        )?;
        self.fallback_degraded_evidence
            .nonfinite_attention_fallbacks = checked_add(
            self.fallback_degraded_evidence
                .nonfinite_attention_fallbacks,
            reference.attention_softmax_nonfinite_fallbacks,
            "reference attention fallbacks",
        )?;
        self.fallback_degraded_evidence
            .nonfinite_attention_fallbacks = checked_add(
            self.fallback_degraded_evidence
                .nonfinite_attention_fallbacks,
            gpu.attention_softmax_nonfinite_fallbacks,
            "GPU-native attention fallbacks",
        )?;
        self.fallback_degraded_evidence.fatal_numerical_failures = checked_add(
            self.fallback_degraded_evidence.fatal_numerical_failures,
            gpu.token_loop_counters_delta.fatal_failures,
            "GPU-native fatal numerical failures",
        )?;
        self.fallback_degraded_evidence.no_progress_failures = checked_add(
            self.fallback_degraded_evidence.no_progress_failures,
            gpu.token_loop_counters_delta.no_progress_failures,
            "GPU-native no-progress failures",
        )?;
        self.fallback_degraded_evidence
            .all_requests_completed_expected_tokens &= gpu.request_completed_expected_tokens;
        Ok(())
    }

    pub fn semantic_failure(&self) -> Option<QualificationFailure> {
        if let Some(mismatch) = &self.first_token_mismatch {
            return Some(QualificationFailure::new(
                FailureStage::Postcondition,
                "generated-token-mismatch",
                format!(
                    "case {} generated position {}: reference={:?} gpu_native={:?}",
                    mismatch.case,
                    mismatch.generated_position,
                    mismatch.reference_token_id,
                    mismatch.gpu_native_token_id
                ),
            ));
        }
        if let Some(mismatch) = &self.first_expert_id_set_mismatch {
            return Some(QualificationFailure::new(
                FailureStage::Postcondition,
                "expert-id-selection-set-mismatch",
                format!(
                    "case {} generated position {} layer {}: reference_only={:?} gpu_native_only={:?}",
                    mismatch.case,
                    mismatch.generated_position,
                    mismatch.layer,
                    mismatch.reference_only_expert_ids,
                    mismatch.gpu_native_only_expert_ids
                ),
            ));
        }
        None
    }

    fn derive_checks(&mut self) {
        let clean_build_provenance = self
            .build_provenance
            .git_sha
            .as_deref()
            .is_some_and(valid_git_sha)
            && self.build_provenance.dirty == Some(false);
        let exact_case_and_token_totals = self.cases.len()
            == crate::greedy_parity::CORPUS_CASE_COUNT
            && self.tokens_per_case == crate::greedy_parity::OUTPUT_TOKEN_LIMIT
            && self.total_tokens_expected
                == crate::greedy_parity::CORPUS_CASE_COUNT
                    * crate::greedy_parity::OUTPUT_TOKEN_LIMIT
            && self.total_tokens_compared == self.total_tokens_expected;
        self.checks = QualificationChecks {
            clean_build_provenance,
            source_config_strict: self.source_config.is_strict(),
            fixed_corpus_identity: self.corpus == CorpusEvidence::fixed(),
            fixed_greedy_sampling: self.sampling == GreedySamplingEvidence::fixed(),
            authoritative_reference_contract: self.reference_contract.is_exact(),
            production_gpu_candidate_path: self.gpu_candidate_path.is_exact(),
            model_geometry_recorded: self.model_geometry.is_some(),
            adapter_exact_match: self
                .actual_adapter
                .as_ref()
                .is_some_and(|adapter| adapter.name == self.expected_adapter_name),
            hardware_adapter: self.actual_adapter.as_ref().is_some_and(|adapter| {
                !adapter.software_adapter && !adapter.device_type.eq_ignore_ascii_case("cpu")
            }),
            exact_case_and_token_totals,
            exact_generated_token_ids: self.exact_token_matches == self.total_tokens_expected
                && self.first_token_mismatch.is_none(),
            expert_id_selection_set_match: self.expert_id_selection_set_match
                && self.first_expert_id_set_mismatch.is_none()
                && self.routing_positions_compared == self.total_tokens_expected
                && self.model_geometry.is_some_and(|geometry| {
                    self.routing_layer_comparisons
                        == self
                            .total_tokens_expected
                            .checked_mul(geometry.num_layers)
                            .unwrap_or(usize::MAX)
                }),
            reference_routing_replays_match: !self.cases.is_empty()
                && self.cases.iter().all(|case| {
                    case.reference_routing_replay
                        .matches_authoritative_reference
                }),
            residency_misses_fully_serviced: !self.cases.is_empty()
                && self.residency.miss_attempts == self.residency.services
                && self.residency.replay_attempts == self.residency.miss_attempts,
            zero_fallback_degraded_or_fatal_evidence: !self.cases.is_empty()
                && self.fallback_degraded_evidence.passes(&self.source_config),
        };
    }

    pub fn finalize(&mut self) -> Result<(), QualificationFailure> {
        self.derive_checks();

        if let Some(failure) = self.semantic_failure() {
            self.fail(failure.clone());
            return Err(failure);
        }
        if !self.checks.passes() {
            let failure = QualificationFailure::new(
                FailureStage::Postcondition,
                "qualification-checks-failed",
                "one or more strict GPU-native greedy-parity qualification checks failed",
            );
            self.fail(failure.clone());
            return Err(failure);
        }
        self.failure = None;
        self.qualification_pass = true;
        Ok(())
    }
}

fn valid_git_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn model_load_strict_complete(load: &ModelLoadEvidence) -> bool {
    load.strict
        && load.loaded_tensors == load.required_tensors
        && load.required_tensors > 0
        && !load.seeded_fallback_remained
        && load.loader != "seeded"
}

fn checked_add(left: u64, right: u64, field: &str) -> Result<u64, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("{field} overflowed"))
}

fn checked_add_snapshots(
    left: GpuNativeTokenLoopSnapshot,
    right: GpuNativeTokenLoopSnapshot,
) -> Result<GpuNativeTokenLoopSnapshot, String> {
    Ok(GpuNativeTokenLoopSnapshot {
        token_attempts: checked_add(left.token_attempts, right.token_attempts, "token_attempts")?,
        tokens_completed: checked_add(
            left.tokens_completed,
            right.tokens_completed,
            "tokens_completed",
        )?,
        warm_tokens_completed: checked_add(
            left.warm_tokens_completed,
            right.warm_tokens_completed,
            "warm_tokens_completed",
        )?,
        residency_miss_attempts: checked_add(
            left.residency_miss_attempts,
            right.residency_miss_attempts,
            "residency_miss_attempts",
        )?,
        replay_attempts: checked_add(
            left.replay_attempts,
            right.replay_attempts,
            "replay_attempts",
        )?,
        residency_services: checked_add(
            left.residency_services,
            right.residency_services,
            "residency_services",
        )?,
        fatal_failures: checked_add(left.fatal_failures, right.fatal_failures, "fatal_failures")?,
        no_progress_failures: checked_add(
            left.no_progress_failures,
            right.no_progress_failures,
            "no_progress_failures",
        )?,
        queue_submissions: checked_add(
            left.queue_submissions,
            right.queue_submissions,
            "queue_submissions",
        )?,
        boundary_maps: checked_add(left.boundary_maps, right.boundary_maps, "boundary_maps")?,
        boundary_readbacks: checked_add(
            left.boundary_readbacks,
            right.boundary_readbacks,
            "boundary_readbacks",
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn routes(tokens: usize, layers: usize, top_k: usize) -> Vec<Vec<Vec<u32>>> {
        (0..tokens)
            .map(|position| {
                (0..layers)
                    .map(|layer| {
                        (0..top_k)
                            .map(|slot| (position * 100 + layer * 10 + slot) as u32)
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn report_schema_and_fixed_totals_are_versioned() {
        let report = test_report();
        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["schema_version"], SCHEMA_VERSION);
        assert_eq!(SCHEMA_VERSION, "mer.gpu-native-q4-greedy-parity.v2");
        assert_eq!(value["mode"], MODE);
        assert_eq!(value["tokens_per_case"], 16);
        assert_eq!(value["total_tokens_expected"], 64);
        assert_eq!(value["qualification_pass"], false);
    }

    #[test]
    fn explicit_failure_keeps_qualification_fail_closed() {
        let mut report = test_report();
        report.fail(QualificationFailure::new(
            FailureStage::Inference,
            "candidate-failed",
            "fatal numerical status",
        ));
        assert!(!report.qualification_pass);
        assert_eq!(
            report.failure.as_ref().map(|failure| failure.code.as_str()),
            Some("candidate-failed")
        );
    }

    #[test]
    fn failed_report_derives_known_checks_without_passing_incomplete_totals() {
        let mut report = test_report();
        report.actual_adapter = Some(GpuDeviceIdentity {
            name: "NVIDIA L4".to_string(),
            vendor_id: 0x10de,
            device_id: 0,
            device_type: "DiscreteGpu".to_string(),
            wgpu_backend: "Vulkan".to_string(),
            driver: "NVIDIA".to_string(),
            driver_info: "test".to_string(),
            compute_plane: "gpu-native".to_string(),
            software_adapter: false,
        });
        report.fail(QualificationFailure::new(
            FailureStage::Postcondition,
            "candidate-failed",
            "test failure after adapter selection",
        ));

        let value = serde_json::to_value(&report).unwrap();
        assert_eq!(value["checks"]["clean_build_provenance"], true);
        assert_eq!(value["checks"]["source_config_strict"], true);
        assert_eq!(value["checks"]["fixed_corpus_identity"], true);
        assert_eq!(value["checks"]["fixed_greedy_sampling"], true);
        assert_eq!(value["checks"]["adapter_exact_match"], true);
        assert_eq!(value["checks"]["hardware_adapter"], true);
        assert_eq!(value["checks"]["exact_case_and_token_totals"], false);
        assert_eq!(value["checks"]["exact_generated_token_ids"], false);
        assert_eq!(value["checks"]["expert_id_selection_set_match"], false);
        assert_eq!(value["qualification_pass"], false);
    }

    fn test_report() -> GpuNativeGreedyParityReport {
        GpuNativeGreedyParityReport::new(
            BuildProvenance {
                git_sha: Some("1".repeat(40)),
                dirty: Some(false),
                package_version: "test".to_string(),
            },
            QualificationArtifacts::default(),
            None,
            "NVIDIA L4".to_string(),
            strict_source_config(),
        )
    }

    #[test]
    fn exact_token_match_accounting_requires_all_tokens() {
        let reference: Vec<u32> = (0..16).collect();
        let comparison = compare_case_outputs(
            "case",
            &[10, 11],
            &reference,
            &reference,
            &routes(16, 2, 2),
            &routes(16, 2, 2),
            16,
            2,
            2,
            Some(10_000),
        )
        .unwrap();
        assert_eq!(comparison.exact_token_matches, 16);
        assert!(comparison.token_ids_match);
        assert!(comparison.expert_id_selection_set_match);
        assert!(comparison.expert_id_rank_order_match);
        assert_eq!(comparison.rank_order_drift_count, 0);
        assert_eq!(comparison.routing_layer_comparisons, 32);
    }

    #[test]
    fn first_token_mismatch_preserves_failure_context() {
        let reference: Vec<u32> = (100..116).collect();
        let mut gpu = reference.clone();
        gpu[5] = 999;
        gpu[9] = 998;
        let comparison = compare_case_outputs(
            "rust-generation",
            &[7, 8, 9],
            &reference,
            &gpu,
            &routes(16, 1, 2),
            &routes(16, 1, 2),
            16,
            1,
            2,
            Some(10_000),
        )
        .unwrap();
        let mismatch = comparison.first_token_mismatch.unwrap();
        assert_eq!(mismatch.generated_position, 5);
        assert_eq!(mismatch.reference_token_id, Some(105));
        assert_eq!(mismatch.gpu_native_token_id, Some(999));
        assert_eq!(mismatch.prompt_token_ids, vec![7, 8, 9]);
        assert_eq!(mismatch.preceding_generated_ids, reference[..5]);
        assert_eq!(comparison.exact_token_matches, 14);
    }

    #[test]
    fn pure_rank_permutation_is_informational_and_not_a_semantic_failure() {
        let tokens = vec![42];
        let reference_routes = vec![vec![vec![30, 37, 114, 86, 29, 68, 113, 35]]];
        let gpu_routes = vec![vec![vec![30, 37, 114, 86, 29, 113, 68, 35]]];
        let comparison = compare_case_outputs(
            "rust-generation",
            &[1],
            &tokens,
            &tokens,
            &reference_routes,
            &gpu_routes,
            1,
            1,
            8,
            Some(128),
        )
        .unwrap();
        assert!(comparison.token_ids_match);
        assert!(comparison.expert_id_selection_set_match);
        assert!(comparison.first_expert_id_set_mismatch.is_none());
        assert!(!comparison.expert_id_rank_order_match);
        assert_eq!(comparison.rank_order_drift_count, 1);
        let drift = comparison
            .first_expert_id_rank_order_drift
            .as_ref()
            .unwrap();
        assert_eq!(drift.generated_position, 0);
        assert_eq!(drift.layer, 0);
        assert_eq!(drift.first_differing_rank_slot, 5);
        assert_eq!(
            drift.canonical_reference_expert_ids,
            vec![29, 30, 35, 37, 68, 86, 113, 114]
        );
        assert_eq!(
            drift.canonical_gpu_native_expert_ids,
            drift.canonical_reference_expert_ids
        );

        let mut report = test_report();
        report.first_token_mismatch = comparison.first_token_mismatch.clone();
        report.first_expert_id_set_mismatch = comparison.first_expert_id_set_mismatch.clone();
        assert!(report.semantic_failure().is_none());
    }

    #[test]
    fn true_membership_change_fails_closed_with_set_difference_evidence() {
        let tokens = vec![42];
        let reference_routes = vec![vec![vec![30, 37, 114, 86, 29, 68, 113, 35]]];
        let gpu_routes = vec![vec![vec![30, 37, 114, 86, 29, 64, 113, 35]]];
        let comparison = compare_case_outputs(
            "rust-generation",
            &[1],
            &tokens,
            &tokens,
            &reference_routes,
            &gpu_routes,
            1,
            1,
            8,
            Some(128),
        )
        .unwrap();
        assert!(!comparison.expert_id_selection_set_match);
        assert_eq!(comparison.rank_order_drift_count, 0);
        let mismatch = comparison.first_expert_id_set_mismatch.as_ref().unwrap();
        assert_eq!(mismatch.reference_only_expert_ids, vec![68]);
        assert_eq!(mismatch.gpu_native_only_expert_ids, vec![64]);

        let mut report = test_report();
        report.first_expert_id_set_mismatch = comparison.first_expert_id_set_mismatch.clone();
        let failure = report.semantic_failure().unwrap();
        assert_eq!(failure.code, "expert-id-selection-set-mismatch");
    }

    #[test]
    fn duplicate_and_out_of_range_expert_ids_are_rejected() {
        let tokens = vec![42];
        let duplicate_routes = vec![vec![vec![30, 30]]];
        let duplicate_error = compare_case_outputs(
            "case",
            &[1],
            &tokens,
            &tokens,
            &duplicate_routes,
            &duplicate_routes,
            1,
            1,
            2,
            Some(128),
        )
        .unwrap_err();
        assert!(duplicate_error.contains("duplicate expert ID 30"));

        let out_of_range_routes = vec![vec![vec![30, 128]]];
        let range_error = compare_case_outputs(
            "case",
            &[1],
            &tokens,
            &tokens,
            &out_of_range_routes,
            &out_of_range_routes,
            1,
            1,
            2,
            Some(128),
        )
        .unwrap_err();
        assert!(range_error.contains("outside num_experts=128"));
    }

    #[test]
    fn token_mismatch_fails_even_when_expert_membership_matches() {
        let reference_routes = vec![vec![vec![30, 37]]];
        let gpu_routes = vec![vec![vec![37, 30]]];
        let comparison = compare_case_outputs(
            "case",
            &[1],
            &[42],
            &[43],
            &reference_routes,
            &gpu_routes,
            1,
            1,
            2,
            Some(128),
        )
        .unwrap();
        assert!(comparison.expert_id_selection_set_match);

        let mut report = test_report();
        report.first_token_mismatch = comparison.first_token_mismatch.clone();
        report.first_expert_id_set_mismatch = comparison.first_expert_id_set_mismatch.clone();
        let failure = report.semantic_failure().unwrap();
        assert_eq!(failure.code, "generated-token-mismatch");
    }

    #[test]
    fn serialized_routing_evidence_preserves_exact_expert_ids() {
        let selected_expert_ids = routes(2, 3, 2);
        let evidence = ReferenceRoutingReplayEvidence {
            generation: GenerationEvidence {
                prompt_token_ids_sha256: "prompt".to_string(),
                generated_token_ids: vec![1, 2],
                generated_token_ids_sha256: "tokens".to_string(),
                generated_text_sha256: "text".to_string(),
                generated_token_count: 2,
                termination_reason: crate::greedy_parity::TerminationReason::LengthLimit,
            },
            matches_authoritative_reference: true,
            routing_positions_captured: 2,
            routed_layers_per_position: 3,
            selected_expert_ids_by_generated_position_layer: selected_expert_ids.clone(),
            background_shutdown: BackgroundShutdownEvidence::default(),
        };
        let value = serde_json::to_value(evidence).unwrap();
        assert_eq!(
            value["selected_expert_ids_by_generated_position_layer"],
            serde_json::json!(selected_expert_ids)
        );
        assert!(value.get("selected_expert_weights").is_none());
    }

    #[test]
    fn zero_or_incomplete_token_configuration_fails_closed() {
        let route = routes(16, 1, 1);
        assert!(compare_case_outputs("case", &[1], &[], &[], &[], &[], 0, 1, 1, None).is_err());
        assert!(compare_case_outputs(
            "case",
            &[1],
            &[1; 15],
            &[1; 16],
            &route,
            &route,
            16,
            1,
            1,
            None,
        )
        .is_err());
    }

    #[test]
    fn fallback_evidence_fails_closed() {
        let source = strict_source_config();
        let mut evidence = FallbackDegradedEvidence {
            all_model_loads_strict_complete: true,
            all_requests_completed_expected_tokens: true,
            ..FallbackDegradedEvidence::default()
        };
        assert!(evidence.passes(&source));
        evidence.fatal_numerical_failures = 1;
        assert!(!evidence.passes(&source));
        evidence.fatal_numerical_failures = 0;
        evidence.degraded_expert_substitutions = 1;
        assert!(!evidence.passes(&source));
    }

    fn strict_source_config() -> SourceConfigEvidence {
        SourceConfigEvidence {
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
        }
    }
}
