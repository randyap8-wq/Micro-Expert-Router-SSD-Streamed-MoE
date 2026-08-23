//! Focused GPU-native router-cutoff divergence diagnostics.
//!
//! This module is diagnostic-only. It owns report construction, ranking,
//! cutoff margins, layerwise drift summaries, and evidence-based causal
//! classification. Production routing and qualification semantics are not
//! changed here.

use serde::Serialize;
use std::collections::BTreeSet;

use crate::backend::GpuDeviceIdentity;
use crate::engine::{CpuQ4BoundaryEmulationSnapshot, RoutedExpertExecutionSnapshot};
use crate::gpu_native_token_loop::{GpuNativeModelGeometry, GpuNativeTokenLoopSnapshot};
use crate::numerical_diagnostics::{compare_vectors, VectorComparisonEvidence};
use crate::qualification::{BuildProvenance, ExpertMetadataEvidence, QualificationArtifacts};

pub const SCHEMA_VERSION: &str = "mer.gpu-native-router-cutoff-diagnostic.v1";
pub const MODE: &str = "diagnose-gpu-native-router-cutoff";
pub const DEFAULT_TOP_N: usize = 12;
pub const CUTOFF_EXPERT_A: u32 = 56;
pub const CUTOFF_EXPERT_B: u32 = 108;

pub fn validate_target(
    case: &str,
    generated_position: usize,
    focus_layer: usize,
    num_layers: usize,
    generated_position_limit: usize,
) -> Result<(), String> {
    if case.trim().is_empty() {
        return Err("--case must be non-empty".to_string());
    }
    if generated_position >= generated_position_limit {
        return Err(format!(
            "generated position {generated_position} is out of range 0..{generated_position_limit}"
        ));
    }
    if num_layers == 0 || focus_layer >= num_layers {
        return Err(format!(
            "focus layer {focus_layer} is out of range for {num_layers} layers"
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VectorDriftEvidence {
    pub comparison: VectorComparisonEvidence,
    pub max_symmetric_relative_error: Option<f32>,
    pub rms_symmetric_relative_error: Option<f64>,
}

pub fn compare_vector_drift(
    reference: &[f32],
    actual: &[f32],
) -> Result<VectorDriftEvidence, String> {
    let comparison = compare_vectors(reference, actual)?;
    let mut max_relative = 0.0f32;
    let mut squared_relative = 0.0f64;
    let mut finite_count = 0usize;
    for (&left, &right) in reference.iter().zip(actual) {
        if !left.is_finite() || !right.is_finite() {
            continue;
        }
        let scale = left.abs().max(right.abs());
        let relative = if scale == 0.0 {
            0.0
        } else {
            (right - left).abs() / scale
        };
        max_relative = max_relative.max(relative);
        squared_relative += f64::from(relative) * f64::from(relative);
        finite_count += 1;
    }
    Ok(VectorDriftEvidence {
        comparison,
        max_symmetric_relative_error: (finite_count > 0).then_some(max_relative),
        rms_symmetric_relative_error: (finite_count > 0)
            .then_some((squared_relative / finite_count as f64).sqrt()),
    })
}

fn strict_softmax(raw_logits: &[f32]) -> Result<Vec<f32>, String> {
    if raw_logits.is_empty() || raw_logits.iter().any(|value| !value.is_finite()) {
        return Err("router logits must be a non-empty finite vector".to_string());
    }
    let max_logit = raw_logits
        .iter()
        .copied()
        .max_by(f32::total_cmp)
        .ok_or("router logits are empty")?;
    let mut scores = Vec::with_capacity(raw_logits.len());
    let mut denominator = 0.0f32;
    for &logit in raw_logits {
        let exponent = (logit - max_logit).exp();
        if !exponent.is_finite() || exponent < 0.0 {
            return Err("router softmax reconstruction produced a nonfinite exponent".to_string());
        }
        denominator += exponent;
        scores.push(exponent);
    }
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err("router softmax reconstruction produced an invalid denominator".to_string());
    }
    for score in &mut scores {
        *score /= denominator;
    }
    Ok(scores)
}

fn ranked_expert_ids(raw_logits: &[f32]) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..raw_logits.len() as u32).collect();
    ids.sort_by(|&left, &right| {
        raw_logits[right as usize]
            .total_cmp(&raw_logits[left as usize])
            .then_with(|| left.cmp(&right))
    });
    ids
}

fn canonical_set(ids: &[u32]) -> Vec<u32> {
    ids.iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn validate_selected(ids: &[u32], num_experts: usize) -> Result<(), String> {
    if ids.is_empty() {
        return Err("selected expert ids must be non-empty".to_string());
    }
    if ids.iter().any(|&id| id as usize >= num_experts) {
        return Err("selected expert id is out of range".to_string());
    }
    if canonical_set(ids).len() != ids.len() {
        return Err("selected expert ids contain duplicates".to_string());
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RankedRouterExpertEvidence {
    pub rank: usize,
    pub expert_id: u32,
    pub raw_logit: f32,
    pub softmax_score: f32,
    pub selected: bool,
    pub selected_weight: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RouterEvaluationEvidence {
    pub raw_logit_source: String,
    pub softmax_score_source: String,
    pub selection_source: String,
    pub raw_logits_all_experts: Vec<f32>,
    pub softmax_scores_all_experts: Vec<f32>,
    pub selected_expert_ids: Vec<u32>,
    pub selected_weights: Option<Vec<f32>>,
    pub ranked_top_n: Vec<RankedRouterExpertEvidence>,
}

pub fn build_router_evaluation(
    raw_logits: &[f32],
    selected_ids: &[u32],
    selected_weights: Option<&[f32]>,
    raw_logit_source: &str,
    softmax_score_source: &str,
    selection_source: &str,
    top_n: usize,
) -> Result<RouterEvaluationEvidence, String> {
    validate_selected(selected_ids, raw_logits.len())?;
    if top_n < 10 || top_n > raw_logits.len() {
        return Err(format!(
            "top_n must be in 10..={}, got {top_n}",
            raw_logits.len()
        ));
    }
    if selected_weights.is_some_and(|weights| weights.len() != selected_ids.len()) {
        return Err("selected weights length differs from selected ids".to_string());
    }
    let scores = strict_softmax(raw_logits)?;
    let ranked_ids = ranked_expert_ids(raw_logits);
    let ranked_top_n = ranked_ids
        .iter()
        .take(top_n)
        .enumerate()
        .map(|(index, &expert_id)| {
            let selected_slot = selected_ids.iter().position(|&id| id == expert_id);
            RankedRouterExpertEvidence {
                rank: index + 1,
                expert_id,
                raw_logit: raw_logits[expert_id as usize],
                softmax_score: scores[expert_id as usize],
                selected: selected_slot.is_some(),
                selected_weight: selected_slot.and_then(|slot| {
                    selected_weights.and_then(|weights| weights.get(slot).copied())
                }),
            }
        })
        .collect();
    Ok(RouterEvaluationEvidence {
        raw_logit_source: raw_logit_source.to_string(),
        softmax_score_source: softmax_score_source.to_string(),
        selection_source: selection_source.to_string(),
        raw_logits_all_experts: raw_logits.to_vec(),
        softmax_scores_all_experts: scores,
        selected_expert_ids: selected_ids.to_vec(),
        selected_weights: selected_weights.map(ToOwned::to_owned),
        ranked_top_n,
    })
}

fn ranked_expert(
    evaluation: &RouterEvaluationEvidence,
    expert_id: u32,
) -> Result<RankedRouterExpertEvidence, String> {
    let raw_logits = &evaluation.raw_logits_all_experts;
    if expert_id as usize >= raw_logits.len() {
        return Err(format!("expert {expert_id} is out of range"));
    }
    let ids = ranked_expert_ids(raw_logits);
    let rank = ids
        .iter()
        .position(|&id| id == expert_id)
        .ok_or_else(|| format!("expert {expert_id} is absent from ranking"))?
        + 1;
    let selected_slot = evaluation
        .selected_expert_ids
        .iter()
        .position(|&id| id == expert_id);
    Ok(RankedRouterExpertEvidence {
        rank,
        expert_id,
        raw_logit: raw_logits[expert_id as usize],
        softmax_score: evaluation.softmax_scores_all_experts[expert_id as usize],
        selected: selected_slot.is_some(),
        selected_weight: selected_slot.and_then(|slot| {
            evaluation
                .selected_weights
                .as_ref()
                .and_then(|weights| weights.get(slot).copied())
        }),
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExpertPairEvidence {
    pub expert_56: RankedRouterExpertEvidence,
    pub expert_108: RankedRouterExpertEvidence,
    pub raw_logit_56_minus_108: f32,
    pub softmax_score_56_minus_108: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CutoffMarginEvidence {
    pub rank_7: RankedRouterExpertEvidence,
    pub rank_8: RankedRouterExpertEvidence,
    pub rank_9: RankedRouterExpertEvidence,
    pub rank_10: RankedRouterExpertEvidence,
    pub score_rank_8_minus_rank_9: f32,
    pub raw_logit_rank_8_minus_rank_9: f32,
    pub expert_pair: ExpertPairEvidence,
}

pub fn cutoff_margin(
    evaluation: &RouterEvaluationEvidence,
) -> Result<CutoffMarginEvidence, String> {
    let ids = ranked_expert_ids(&evaluation.raw_logits_all_experts);
    if ids.len() < 10 || evaluation.selected_expert_ids.len() != 8 {
        return Err("cutoff evidence requires at least 10 experts and top_k=8".to_string());
    }
    let at_rank = |rank: usize| ranked_expert(evaluation, ids[rank - 1]);
    let rank_7 = at_rank(7)?;
    let rank_8 = at_rank(8)?;
    let rank_9 = at_rank(9)?;
    let rank_10 = at_rank(10)?;
    let expert_56 = ranked_expert(evaluation, CUTOFF_EXPERT_A)?;
    let expert_108 = ranked_expert(evaluation, CUTOFF_EXPERT_B)?;
    Ok(CutoffMarginEvidence {
        score_rank_8_minus_rank_9: rank_8.softmax_score - rank_9.softmax_score,
        raw_logit_rank_8_minus_rank_9: rank_8.raw_logit - rank_9.raw_logit,
        rank_7,
        rank_8,
        rank_9,
        rank_10,
        expert_pair: ExpertPairEvidence {
            raw_logit_56_minus_108: expert_56.raw_logit - expert_108.raw_logit,
            softmax_score_56_minus_108: expert_56.softmax_score - expert_108.softmax_score,
            expert_56,
            expert_108,
        },
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LayerwiseDriftEvidence {
    pub layer: usize,
    pub post_attention: VectorDriftEvidence,
    pub router_input: VectorDriftEvidence,
    pub reference_selected_expert_ids: Vec<u32>,
    pub gpu_selected_expert_ids: Vec<u32>,
    pub selected_set_membership_equal: bool,
    pub selected_rank_order_equal: bool,
    pub post_moe: VectorDriftEvidence,
    pub cpu_shadow_on_gpu_input_selected_expert_ids: Vec<u32>,
    pub cpu_shadow_set_matches_reference: bool,
    pub cpu_shadow_set_matches_gpu: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LayerwiseDriftSummary {
    pub layers_through_focus: Vec<LayerwiseDriftEvidence>,
    pub first_rank_order_drift_layer: Option<usize>,
    pub first_set_membership_drift_layer: Option<usize>,
    pub focus_layer_is_first_set_membership_drift: bool,
    pub first_cpu_shadow_set_change_layer: Option<usize>,
}

pub fn build_layerwise_drift(
    reference: &crate::gpu_native_diagnostics::ModelDiagnosticTrace,
    gpu: &crate::gpu_native_diagnostics::GpuNativeDiagnosticTrace,
    cpu_shadow_ids: &[Vec<u32>],
    focus_layer: usize,
) -> Result<LayerwiseDriftSummary, String> {
    let required = focus_layer.checked_add(1).ok_or("focus layer overflowed")?;
    for len in [
        reference.layer_post_attn.len(),
        reference.layer_router_input.len(),
        reference.layer_selected_ids.len(),
        reference.layer_post_moe.len(),
        gpu.layer_post_attn.len(),
        gpu.layer_router_input.len(),
        gpu.layer_selected_ids.len(),
        gpu.layer_post_moe.len(),
        cpu_shadow_ids.len(),
    ] {
        if len < required {
            return Err("layerwise diagnostic evidence is incomplete".to_string());
        }
    }
    let mut layers = Vec::with_capacity(required);
    for layer in 0..required {
        let reference_ids = &reference.layer_selected_ids[layer];
        let gpu_ids = &gpu.layer_selected_ids[layer];
        let shadow_ids = &cpu_shadow_ids[layer];
        validate_selected(reference_ids, usize::MAX)?;
        validate_selected(gpu_ids, usize::MAX)?;
        validate_selected(shadow_ids, usize::MAX)?;
        let reference_set = canonical_set(reference_ids);
        let gpu_set = canonical_set(gpu_ids);
        let shadow_set = canonical_set(shadow_ids);
        layers.push(LayerwiseDriftEvidence {
            layer,
            post_attention: compare_vector_drift(
                &reference.layer_post_attn[layer],
                &gpu.layer_post_attn[layer],
            )?,
            router_input: compare_vector_drift(
                &reference.layer_router_input[layer],
                &gpu.layer_router_input[layer],
            )?,
            reference_selected_expert_ids: reference_ids.clone(),
            gpu_selected_expert_ids: gpu_ids.clone(),
            selected_set_membership_equal: reference_set == gpu_set,
            selected_rank_order_equal: reference_ids == gpu_ids,
            post_moe: compare_vector_drift(
                &reference.layer_post_moe[layer],
                &gpu.layer_post_moe[layer],
            )?,
            cpu_shadow_on_gpu_input_selected_expert_ids: shadow_ids.clone(),
            cpu_shadow_set_matches_reference: shadow_set == reference_set,
            cpu_shadow_set_matches_gpu: shadow_set == gpu_set,
        });
    }
    let first_rank_order_drift_layer = layers
        .iter()
        .find(|evidence| !evidence.selected_rank_order_equal)
        .map(|evidence| evidence.layer);
    let first_set_membership_drift_layer = layers
        .iter()
        .find(|evidence| !evidence.selected_set_membership_equal)
        .map(|evidence| evidence.layer);
    Ok(LayerwiseDriftSummary {
        first_rank_order_drift_layer,
        first_set_membership_drift_layer,
        focus_layer_is_first_set_membership_drift: first_set_membership_drift_layer
            == Some(focus_layer),
        first_cpu_shadow_set_change_layer: layers
            .iter()
            .find(|evidence| !evidence.cpu_shadow_set_matches_reference)
            .map(|evidence| evidence.layer),
        layers_through_focus: layers,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NearTieEvidence {
    pub pair_experts: [u32; 2],
    pub pair_straddles_observed_membership: bool,
    pub reference_pair_raw_gap_abs: f32,
    pub gpu_pair_raw_gap_abs: f32,
    pub cpu_shadow_pair_raw_gap_abs: f32,
    pub upstream_logit_max_abs_error: f32,
    pub same_input_router_logit_max_abs_error: f32,
    pub established_pairwise_logit_drift_bound: f32,
    pub within_observed_drift_bound: bool,
    pub rule: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ClassificationEvidence {
    pub upstream_state_drift: bool,
    pub gpu_router_math_drift: bool,
    pub cutoff_near_tie: bool,
    pub mixed_or_ambiguous: bool,
    pub near_tie_evidence: NearTieEvidence,
    pub reason: String,
}

fn max_abs(comparison: &VectorDriftEvidence) -> f32 {
    comparison
        .comparison
        .max_absolute_error
        .unwrap_or(f32::INFINITY)
}

pub fn classify_divergence(
    reference: &RouterEvaluationEvidence,
    gpu: &RouterEvaluationEvidence,
    cpu_shadow_on_gpu_input: &RouterEvaluationEvidence,
    reference_vs_shadow_logits: &VectorDriftEvidence,
    shadow_vs_gpu_logits: &VectorDriftEvidence,
) -> Result<ClassificationEvidence, String> {
    let reference_set = canonical_set(&reference.selected_expert_ids);
    let gpu_set = canonical_set(&gpu.selected_expert_ids);
    let shadow_set = canonical_set(&cpu_shadow_on_gpu_input.selected_expert_ids);
    let upstream_state_drift = reference_set != gpu_set && shadow_set == gpu_set;
    let gpu_router_math_drift = reference_set != gpu_set && shadow_set == reference_set;

    let reference_pair = cutoff_margin(reference)?.expert_pair;
    let gpu_pair = cutoff_margin(gpu)?.expert_pair;
    let shadow_pair = cutoff_margin(cpu_shadow_on_gpu_input)?.expert_pair;
    let pair_straddles_observed_membership = reference_set.contains(&CUTOFF_EXPERT_A)
        && !reference_set.contains(&CUTOFF_EXPERT_B)
        && !gpu_set.contains(&CUTOFF_EXPERT_A)
        && gpu_set.contains(&CUTOFF_EXPERT_B);
    let upstream_error = max_abs(reference_vs_shadow_logits);
    let router_error = max_abs(shadow_vs_gpu_logits);
    // If each logit may move by at most `upstream_error + router_error`,
    // the difference between a pair may move by at most twice that value.
    let observed_pair_bound = 2.0 * (upstream_error + router_error);
    let reference_gap = reference_pair.raw_logit_56_minus_108.abs();
    let gpu_gap = gpu_pair.raw_logit_56_minus_108.abs();
    let shadow_gap = shadow_pair.raw_logit_56_minus_108.abs();
    let within_observed_drift_bound = pair_straddles_observed_membership
        && observed_pair_bound.is_finite()
        && reference_gap <= observed_pair_bound
        && gpu_gap <= observed_pair_bound
        && shadow_gap <= observed_pair_bound;
    let mixed_or_ambiguous = reference_set == gpu_set
        || (!upstream_state_drift && !gpu_router_math_drift)
        || (upstream_state_drift && gpu_router_math_drift);
    let reason = if upstream_state_drift {
        "CPU shadow on the actual GPU router input selects the GPU membership set; the set change is already present in upstream state".to_string()
    } else if gpu_router_math_drift {
        "CPU shadow on the actual GPU router input retains the reference membership set while the production GPU router selects a different set; the divergence is inside same-input GPU router arithmetic".to_string()
    } else if reference_set == gpu_set {
        "the requested focus layer has no selected-expert membership divergence".to_string()
    } else {
        "CPU shadow selects neither observed membership set, so the evidence is mixed or ambiguous"
            .to_string()
    };
    Ok(ClassificationEvidence {
        upstream_state_drift,
        gpu_router_math_drift,
        cutoff_near_tie: within_observed_drift_bound,
        mixed_or_ambiguous,
        near_tie_evidence: NearTieEvidence {
            pair_experts: [CUTOFF_EXPERT_A, CUTOFF_EXPERT_B],
            pair_straddles_observed_membership,
            reference_pair_raw_gap_abs: reference_gap,
            gpu_pair_raw_gap_abs: gpu_gap,
            cpu_shadow_pair_raw_gap_abs: shadow_gap,
            upstream_logit_max_abs_error: upstream_error,
            same_input_router_logit_max_abs_error: router_error,
            established_pairwise_logit_drift_bound: observed_pair_bound,
            within_observed_drift_bound,
            rule: "near tie iff experts 56/108 straddle the observed sets and the reference, GPU, and CPU-shadow raw-logit gaps are all within 2 * (max upstream same-router logit drift + max same-input CPU/GPU router logit drift)".to_string(),
        },
        reason,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PositionAccountingEvidence {
    pub prompt_token_ids: Vec<u32>,
    pub prompt_token_count: usize,
    pub reference_preceding_generated_ids: Vec<u32>,
    pub gpu_preceding_generated_ids: Vec<u32>,
    pub reference_input_token_id: u32,
    pub gpu_input_token_id: u32,
    pub generated_position: usize,
    pub diagnostic_position: usize,
    pub gpu_committed_position_before_diagnostic: usize,
    pub gpu_committed_position_after_diagnostic: usize,
    pub reference_kv_sequence_lengths_after_diagnostic: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FocusStateEvidence {
    pub reference_router_input: Vec<f32>,
    pub gpu_router_input: Vec<f32>,
    pub router_input_comparison: VectorDriftEvidence,
    pub reference_post_moe: Vec<f32>,
    pub gpu_post_moe: Vec<f32>,
    pub post_moe_comparison: VectorDriftEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GreedyLogitMarginEvidence {
    pub top_1_token_id: u32,
    pub top_1_logit: f32,
    pub top_2_token_id: u32,
    pub top_2_logit: f32,
    pub top_1_minus_top_2_margin: f32,
}

pub fn greedy_logit_margin(logits: &[f32]) -> Result<GreedyLogitMarginEvidence, String> {
    if logits.len() < 2 || logits.iter().any(|value| !value.is_finite()) {
        return Err("greedy logit margin requires at least two finite logits".to_string());
    }
    let ids = ranked_expert_ids(logits);
    let top_1 = ids[0];
    let top_2 = ids[1];
    Ok(GreedyLogitMarginEvidence {
        top_1_token_id: top_1,
        top_1_logit: logits[top_1 as usize],
        top_2_token_id: top_2,
        top_2_logit: logits[top_2 as usize],
        top_1_minus_top_2_margin: logits[top_1 as usize] - logits[top_2 as usize],
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FinalEffectEvidence {
    pub final_norm_comparison: VectorDriftEvidence,
    pub final_logits_comparison: VectorDriftEvidence,
    pub reference_greedy_token_id: u32,
    pub gpu_greedy_token_id: u32,
    pub token_equal: bool,
    pub reference_lm_head_margin: GreedyLogitMarginEvidence,
    pub gpu_lm_head_margin: GreedyLogitMarginEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FocusRouterEvidence {
    pub reference: RouterEvaluationEvidence,
    pub gpu_native: RouterEvaluationEvidence,
    pub cpu_shadow_on_gpu_input: RouterEvaluationEvidence,
    pub gpu_shadow_on_reference_input: Option<RouterEvaluationEvidence>,
    pub reference_cutoff_margin: CutoffMarginEvidence,
    pub gpu_cutoff_margin: CutoffMarginEvidence,
    pub cpu_shadow_cutoff_margin: CutoffMarginEvidence,
    pub gpu_shadow_cutoff_margin: Option<CutoffMarginEvidence>,
    pub reference_vs_gpu_raw_logits: VectorDriftEvidence,
    pub reference_vs_cpu_shadow_raw_logits: VectorDriftEvidence,
    pub cpu_shadow_vs_gpu_raw_logits: VectorDriftEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SafetyEvidence {
    pub reference_cpu_q4_boundary_emulation: CpuQ4BoundaryEmulationSnapshot,
    pub reference_routed_execution_delta: RoutedExpertExecutionSnapshot,
    pub gpu_routed_execution_delta: RoutedExpertExecutionSnapshot,
    pub gpu_layer_statuses: Vec<u32>,
    pub gpu_final_status: u32,
    pub attention_nonfinite_fallbacks: u64,
    pub no_fatal_failure: bool,
    pub no_gpu_cpu_fallback: bool,
    pub no_degraded_expert_substitution: bool,
    pub no_unserviced_residency_failure: bool,
}

impl SafetyEvidence {
    fn is_clean(&self) -> bool {
        self.reference_cpu_q4_boundary_emulation.enabled
            && self
                .reference_cpu_q4_boundary_emulation
                .routed_expert_dispatches
                > 0
            && self.gpu_layer_statuses.iter().all(|&status| status == 0)
            && self.gpu_final_status == 0
            && self.attention_nonfinite_fallbacks == 0
            && self.no_fatal_failure
            && self.no_gpu_cpu_fallback
            && self.no_degraded_expert_substitution
            && self.no_unserviced_residency_failure
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DiagnosticCaseEvidence {
    pub name: String,
    pub prompt: String,
    pub prompt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuNativeRouterCutoffReport {
    pub schema_version: String,
    pub mode: String,
    pub diagnostic_complete: bool,
    pub qualification_pass: bool,
    pub failure: Option<String>,
    pub build_provenance: BuildProvenance,
    pub build_git_sha: String,
    pub executable_sha256: String,
    pub resolved_config_sha256: String,
    pub artifacts: QualificationArtifacts,
    pub expert_metadata: ExpertMetadataEvidence,
    pub model_geometry: GpuNativeModelGeometry,
    pub expected_adapter_name: String,
    pub actual_adapter: GpuDeviceIdentity,
    pub case: DiagnosticCaseEvidence,
    pub generated_position: usize,
    pub focus_layer: usize,
    pub position_accounting: PositionAccountingEvidence,
    pub focus_state: FocusStateEvidence,
    pub layerwise_drift: LayerwiseDriftSummary,
    pub focus_router: FocusRouterEvidence,
    pub classification: ClassificationEvidence,
    pub final_effect: FinalEffectEvidence,
    pub safety: SafetyEvidence,
    pub token_loop_counters_delta: GpuNativeTokenLoopSnapshot,
    pub diagnostic_attempt_count: usize,
}

impl GpuNativeRouterCutoffReport {
    pub fn finish(&mut self) -> Result<(), String> {
        self.diagnostic_complete = false;
        self.qualification_pass = false;
        if self.failure.is_some() {
            return Err("cannot finish a failed diagnostic report".to_string());
        }
        if self.focus_layer >= self.model_geometry.num_layers
            || self.layerwise_drift.layers_through_focus.len() != self.focus_layer + 1
        {
            return Err("layerwise evidence does not cover every layer through focus".to_string());
        }
        if self.focus_router.reference.raw_logits_all_experts.len()
            != self.model_geometry.num_experts
            || self.focus_router.gpu_native.raw_logits_all_experts.len()
                != self.model_geometry.num_experts
            || self
                .focus_router
                .cpu_shadow_on_gpu_input
                .raw_logits_all_experts
                .len()
                != self.model_geometry.num_experts
        {
            return Err("focus router evidence does not cover every expert".to_string());
        }
        if self.actual_adapter.name != self.expected_adapter_name {
            return Err("actual adapter does not match expected adapter".to_string());
        }
        if !self.safety.is_clean() {
            return Err("fallback/failure evidence is not clean".to_string());
        }
        if self.diagnostic_attempt_count == 0 {
            return Err("diagnostic attempt count must be positive".to_string());
        }
        self.diagnostic_complete = true;
        self.qualification_pass = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_with_pair(value_56: f32, value_108: f32) -> Vec<f32> {
        let mut logits = (0..128).map(|id| -(id as f32)).collect::<Vec<_>>();
        for (rank, id) in [9usize, 82, 46, 74, 15, 43, 113].into_iter().enumerate() {
            logits[id] = 20.0 - rank as f32;
        }
        logits[56] = value_56;
        logits[108] = value_108;
        logits
    }

    fn evaluation(raw: &[f32], selected: &[u32]) -> RouterEvaluationEvidence {
        build_router_evaluation(
            raw,
            selected,
            None,
            "test",
            "host stable softmax reconstructed from raw logits",
            "test selection",
            12,
        )
        .unwrap()
    }

    #[test]
    fn target_validation_rejects_malformed_and_out_of_range_values() {
        assert!(validate_target("rust-debugging", 0, 24, 48, 16).is_ok());
        assert!(validate_target("", 0, 24, 48, 16).is_err());
        assert!(validate_target("rust-debugging", 16, 24, 48, 16).is_err());
        assert!(validate_target("rust-debugging", 0, 48, 48, 16).is_err());
    }

    #[test]
    fn top_n_ranking_and_cutoff_margin_are_deterministic() {
        let raw = raw_with_pair(12.0, 11.5);
        let selected = [9, 82, 46, 74, 15, 43, 113, 56];
        let evidence = evaluation(&raw, &selected);
        assert_eq!(evidence.ranked_top_n.len(), 12);
        assert_eq!(evidence.ranked_top_n[7].expert_id, 56);
        assert_eq!(evidence.ranked_top_n[8].expert_id, 108);
        let margin = cutoff_margin(&evidence).unwrap();
        assert_eq!(margin.rank_8.expert_id, 56);
        assert_eq!(margin.rank_9.expert_id, 108);
        assert_eq!(margin.raw_logit_rank_8_minus_rank_9, 0.5);
        assert_eq!(margin.expert_pair.expert_56.expert_id, 56);
        assert_eq!(margin.expert_pair.expert_108.expert_id, 108);
    }

    #[test]
    fn cpu_shadow_classifies_upstream_state_drift() {
        let reference_raw = raw_with_pair(12.0, 11.9);
        let gpu_raw = raw_with_pair(11.9, 12.0);
        let reference = evaluation(&reference_raw, &[9, 82, 46, 74, 15, 43, 113, 56]);
        let gpu = evaluation(&gpu_raw, &[9, 82, 46, 74, 15, 43, 113, 108]);
        let shadow = evaluation(&gpu_raw, &[9, 82, 46, 74, 15, 43, 113, 108]);
        let upstream = compare_vector_drift(&reference_raw, &gpu_raw).unwrap();
        let router = compare_vector_drift(&gpu_raw, &gpu_raw).unwrap();
        let classification =
            classify_divergence(&reference, &gpu, &shadow, &upstream, &router).unwrap();
        assert!(classification.upstream_state_drift);
        assert!(!classification.gpu_router_math_drift);
        assert!(!classification.mixed_or_ambiguous);
    }

    #[test]
    fn cpu_shadow_classifies_same_input_gpu_router_math_drift() {
        let reference_raw = raw_with_pair(12.0, 11.9);
        let gpu_raw = raw_with_pair(11.9, 12.0);
        let reference = evaluation(&reference_raw, &[9, 82, 46, 74, 15, 43, 113, 56]);
        let gpu = evaluation(&gpu_raw, &[9, 82, 46, 74, 15, 43, 113, 108]);
        let shadow = evaluation(&reference_raw, &[9, 82, 46, 74, 15, 43, 113, 56]);
        let upstream = compare_vector_drift(&reference_raw, &reference_raw).unwrap();
        let router = compare_vector_drift(&reference_raw, &gpu_raw).unwrap();
        let classification =
            classify_divergence(&reference, &gpu, &shadow, &upstream, &router).unwrap();
        assert!(!classification.upstream_state_drift);
        assert!(classification.gpu_router_math_drift);
        assert!(!classification.mixed_or_ambiguous);
    }

    #[test]
    fn near_tie_is_derived_from_observed_logit_drift_not_an_epsilon() {
        let reference_raw = raw_with_pair(12.0, 11.999);
        let shadow_raw = raw_with_pair(11.9995, 11.9995);
        let gpu_raw = raw_with_pair(11.999, 12.0);
        let reference = evaluation(&reference_raw, &[9, 82, 46, 74, 15, 43, 113, 56]);
        let gpu = evaluation(&gpu_raw, &[9, 82, 46, 74, 15, 43, 113, 108]);
        let shadow = evaluation(&shadow_raw, &[9, 82, 46, 74, 15, 43, 113, 56]);
        let upstream = compare_vector_drift(&reference_raw, &shadow_raw).unwrap();
        let router = compare_vector_drift(&shadow_raw, &gpu_raw).unwrap();
        let classification =
            classify_divergence(&reference, &gpu, &shadow, &upstream, &router).unwrap();
        assert!(classification.cutoff_near_tie);
        assert!(
            classification
                .near_tie_evidence
                .established_pairwise_logit_drift_bound
                >= 0.001
        );

        let far_reference_raw = raw_with_pair(15.0, 10.0);
        let far_gpu_raw = raw_with_pair(10.0, 15.0);
        let far_reference = evaluation(&far_reference_raw, &[9, 82, 46, 74, 15, 43, 113, 56]);
        let far_gpu = evaluation(&far_gpu_raw, &[9, 82, 46, 74, 15, 43, 113, 108]);
        let no_drift = compare_vector_drift(&far_reference_raw, &far_reference_raw).unwrap();
        let classification = classify_divergence(
            &far_reference,
            &far_gpu,
            &far_reference,
            &no_drift,
            &no_drift,
        )
        .unwrap();
        assert!(!classification.cutoff_near_tie);
    }

    #[test]
    fn malformed_router_evidence_fails_closed() {
        let raw = raw_with_pair(12.0, 11.9);
        assert!(build_router_evaluation(
            &raw,
            &[9, 9, 46, 74, 15, 43, 113, 56],
            None,
            "test",
            "test",
            "test",
            12,
        )
        .is_err());
        assert!(build_router_evaluation(
            &raw,
            &[9, 82, 46, 74, 15, 43, 113, 56],
            None,
            "test",
            "test",
            "test",
            9,
        )
        .is_err());
    }

    #[test]
    fn report_schema_is_diagnostic_and_never_claims_qualification() {
        assert_eq!(SCHEMA_VERSION, "mer.gpu-native-router-cutoff-diagnostic.v1");
        assert_eq!(MODE, "diagnose-gpu-native-router-cutoff");
        let value = serde_json::json!({
            "schema_version": SCHEMA_VERSION,
            "mode": MODE,
            "diagnostic_complete": true,
            "qualification_pass": false,
            "expert_56": CUTOFF_EXPERT_A,
            "expert_108": CUTOFF_EXPERT_B,
        });
        let encoded = serde_json::to_string(&value).unwrap();
        assert!(encoded.contains("\"qualification_pass\":false"));
        assert!(encoded.contains("\"expert_56\":56"));
        assert!(encoded.contains("\"expert_108\":108"));
    }

    #[test]
    fn strict_completion_and_failure_accounting_round_trips() {
        let raw = raw_with_pair(12.0, 11.9);
        let selected = [9, 82, 46, 74, 15, 43, 113, 56];
        let router = evaluation(&raw, &selected);
        let router_cutoff = cutoff_margin(&router).unwrap();
        let vector_drift = compare_vector_drift(&[1.0], &[1.0]).unwrap();
        let raw_drift = compare_vector_drift(&raw, &raw).unwrap();
        let classification =
            classify_divergence(&router, &router, &router, &raw_drift, &raw_drift).unwrap();
        let layer = LayerwiseDriftEvidence {
            layer: 0,
            post_attention: vector_drift.clone(),
            router_input: vector_drift.clone(),
            reference_selected_expert_ids: selected.to_vec(),
            gpu_selected_expert_ids: selected.to_vec(),
            selected_set_membership_equal: true,
            selected_rank_order_equal: true,
            post_moe: vector_drift.clone(),
            cpu_shadow_on_gpu_input_selected_expert_ids: selected.to_vec(),
            cpu_shadow_set_matches_reference: true,
            cpu_shadow_set_matches_gpu: true,
        };
        let lm_margin = greedy_logit_margin(&[2.0, 1.0]).unwrap();
        let mut report = GpuNativeRouterCutoffReport {
            schema_version: SCHEMA_VERSION.to_string(),
            mode: MODE.to_string(),
            diagnostic_complete: false,
            qualification_pass: false,
            failure: None,
            build_provenance: BuildProvenance {
                git_sha: Some("test-sha".to_string()),
                dirty: Some(true),
                package_version: "test".to_string(),
            },
            build_git_sha: "test-sha".to_string(),
            executable_sha256: "test-executable".to_string(),
            resolved_config_sha256: "test-config".to_string(),
            artifacts: QualificationArtifacts::default(),
            expert_metadata: ExpertMetadataEvidence {
                dtype: Some("q4_0".to_string()),
                q4_0_layout: Some("test".to_string()),
                conversion_mode: None,
                source: None,
                explicitly_synthetic: false,
            },
            model_geometry: GpuNativeModelGeometry {
                num_layers: 1,
                d_model: 1,
                d_ff: 1,
                num_experts: 128,
                top_k: 8,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 1,
                rope_dim: 1,
                vocab_size: 2,
                max_seq_len: 2,
                rms_eps: 1.0e-6,
                rope_base: 10_000.0,
            },
            expected_adapter_name: "NVIDIA L4".to_string(),
            actual_adapter: GpuDeviceIdentity {
                name: "NVIDIA L4".to_string(),
                vendor_id: 0,
                device_id: 0,
                device_type: "DiscreteGpu".to_string(),
                wgpu_backend: "Vulkan".to_string(),
                driver: "test".to_string(),
                driver_info: "test".to_string(),
                compute_plane: "gpu-native".to_string(),
                software_adapter: false,
            },
            case: DiagnosticCaseEvidence {
                name: "rust-debugging".to_string(),
                prompt: "test prompt".to_string(),
                prompt_sha256: "test-prompt".to_string(),
            },
            generated_position: 0,
            focus_layer: 0,
            position_accounting: PositionAccountingEvidence {
                prompt_token_ids: vec![1],
                prompt_token_count: 1,
                reference_preceding_generated_ids: Vec::new(),
                gpu_preceding_generated_ids: Vec::new(),
                reference_input_token_id: 1,
                gpu_input_token_id: 1,
                generated_position: 0,
                diagnostic_position: 0,
                gpu_committed_position_before_diagnostic: 0,
                gpu_committed_position_after_diagnostic: 1,
                reference_kv_sequence_lengths_after_diagnostic: vec![1],
            },
            focus_state: FocusStateEvidence {
                reference_router_input: vec![1.0],
                gpu_router_input: vec![1.0],
                router_input_comparison: vector_drift.clone(),
                reference_post_moe: vec![1.0],
                gpu_post_moe: vec![1.0],
                post_moe_comparison: vector_drift.clone(),
            },
            layerwise_drift: LayerwiseDriftSummary {
                layers_through_focus: vec![layer],
                first_rank_order_drift_layer: None,
                first_set_membership_drift_layer: None,
                focus_layer_is_first_set_membership_drift: false,
                first_cpu_shadow_set_change_layer: None,
            },
            focus_router: FocusRouterEvidence {
                reference: router.clone(),
                gpu_native: router.clone(),
                cpu_shadow_on_gpu_input: router.clone(),
                gpu_shadow_on_reference_input: Some(router),
                reference_cutoff_margin: router_cutoff.clone(),
                gpu_cutoff_margin: router_cutoff.clone(),
                cpu_shadow_cutoff_margin: router_cutoff.clone(),
                gpu_shadow_cutoff_margin: Some(router_cutoff),
                reference_vs_gpu_raw_logits: raw_drift.clone(),
                reference_vs_cpu_shadow_raw_logits: raw_drift.clone(),
                cpu_shadow_vs_gpu_raw_logits: raw_drift,
            },
            classification,
            final_effect: FinalEffectEvidence {
                final_norm_comparison: vector_drift.clone(),
                final_logits_comparison: compare_vector_drift(&[2.0, 1.0], &[2.0, 1.0]).unwrap(),
                reference_greedy_token_id: 0,
                gpu_greedy_token_id: 0,
                token_equal: true,
                reference_lm_head_margin: lm_margin.clone(),
                gpu_lm_head_margin: lm_margin,
            },
            safety: SafetyEvidence {
                reference_cpu_q4_boundary_emulation: CpuQ4BoundaryEmulationSnapshot {
                    enabled: true,
                    routed_expert_dispatches: 1,
                },
                reference_routed_execution_delta: RoutedExpertExecutionSnapshot::default(),
                gpu_routed_execution_delta: RoutedExpertExecutionSnapshot::default(),
                gpu_layer_statuses: vec![0],
                gpu_final_status: 0,
                attention_nonfinite_fallbacks: 0,
                no_fatal_failure: true,
                no_gpu_cpu_fallback: true,
                no_degraded_expert_substitution: true,
                no_unserviced_residency_failure: true,
            },
            token_loop_counters_delta: GpuNativeTokenLoopSnapshot::default(),
            diagnostic_attempt_count: 1,
        };

        report.finish().unwrap();
        assert!(report.diagnostic_complete);
        assert!(!report.qualification_pass);
        let encoded = serde_json::to_string(&report).unwrap();
        assert!(encoded.contains("\"diagnostic_complete\":true"));
        assert!(encoded.contains("\"raw_logits_all_experts\":["));

        let mut failed = report;
        failed.failure = Some("diagnostic capture failed".to_string());
        assert!(failed.finish().is_err());
        assert!(!failed.diagnostic_complete);
        assert!(!failed.qualification_pass);

        failed.failure = None;
        failed.safety.no_fatal_failure = false;
        assert!(failed.finish().is_err());
        assert!(!failed.diagnostic_complete);
        assert!(!failed.qualification_pass);
    }
}
