//! Phase-B GPU-native first-token mathematical divergence diagnostics.
//!
//! Compares the authoritative CPU reference (emulating the production Hybrid
//! f16 expert input/output boundary with f32 routing aggregation) against the
//! full GPU-native transformer execution path stage-by-stage for token 0.
//!
//! Diagnostic-only: normal serving and GPU token loop execution paths remain
//! untouched. Qualification pass is always false.

use serde::{Deserialize, Serialize};

use crate::backend::gpu_native::MAX_GPU_NATIVE_ROUTER_TOP_K;
use crate::backend::GpuDeviceIdentity;
use crate::gpu_native_token_loop::{
    GpuNativeModelGeometry, GpuNativeTokenLoopError, GpuNativeTokenLoopSnapshot,
};
use crate::greedy_parity::GreedySamplingEvidence;
use crate::numerical_diagnostics::VectorComparisonEvidence;
use crate::qualification::{BuildProvenance, ExpertMetadataEvidence};

pub const SCHEMA_VERSION: &str = "mer.gpu-native-q4-first-divergence-diagnostic.v1";
pub const MODE: &str = "diagnose-gpu-native-q4-first-divergence";
pub const TARGET_CASE: &str = "json-transformation";

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErrorTolerance {
    pub absolute: f32,
    pub relative: f32,
    pub formula: &'static str,
}

impl ErrorTolerance {
    pub fn evaluates(&self, reference: f32, actual: f32) -> bool {
        if !reference.is_finite() || !actual.is_finite() {
            return false;
        }
        let abs_error = (actual - reference).abs();
        let allowed = self.absolute + self.relative * reference.abs();
        abs_error <= allowed
    }
}

/// Reference tolerances derived from established MER qualification contracts.
///
/// Provenance:
/// - `RAW_PROJECTION_TOLERANCE`: derived from `q4_parity::RAW_TOLERANCE` (absolute 1.0e-5, relative 1.0e-4)
///   for raw shader execution against scalar float dot product.
/// - `Q4_EXPERT_F16_BOUNDARY_TOLERANCE`: derived from `q4_parity::COMPLETE_TOLERANCE` (absolute 2.0e-3, relative 5.0e-3)
///   for complete single-expert execution at the production F16 input/output boundary.
///
/// Note: Accumulated multi-layer hidden states (post-attention across layers, full 48-layer post-MoE state,
/// final RMSNorm, and vocabulary logits) do not have a closed-form single-operation contract in MER.
/// The diagnostic reports exact bitwise match, max absolute error, and RMS error without pretending
/// unestablished thresholds prove end-to-end qualification.
pub const RAW_PROJECTION_TOLERANCE: ErrorTolerance = ErrorTolerance {
    absolute: 1.0e-5,
    relative: 1.0e-4,
    formula: "abs_error <= absolute + relative * abs(reference)",
};

pub const Q4_EXPERT_F16_BOUNDARY_TOLERANCE: ErrorTolerance = ErrorTolerance {
    absolute: 2.0e-3,
    relative: 5.0e-3,
    formula: "abs_error <= absolute + relative * abs(reference)",
};

/// Fixed byte layout for the single contiguous diagnostic staging buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuNativeDiagnosticTraceLayout {
    pub num_layers: usize,
    pub d_model: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub vocab_size: usize,
    pub embedding_offset: usize,
    pub embedding_bytes: usize,
    pub layer_post_attn_offset: usize,
    pub layer_post_attn_bytes: usize,
    pub layer_router_input_offset: usize,
    pub layer_router_input_bytes: usize,
    pub layer_router_logits_offset: usize,
    pub layer_router_logits_bytes: usize,
    pub layer_selected_ids_offset: usize,
    pub layer_selected_ids_bytes: usize,
    pub layer_selected_weights_offset: usize,
    pub layer_selected_weights_bytes: usize,
    pub layer_post_moe_offset: usize,
    pub layer_post_moe_bytes: usize,
    pub layer_status_offset: usize,
    pub layer_status_bytes: usize,
    pub final_norm_offset: usize,
    pub final_norm_bytes: usize,
    pub logits_offset: usize,
    pub logits_bytes: usize,
    pub final_status_offset: usize,
    pub final_status_bytes: usize,
    pub sampled_token_offset: usize,
    pub sampled_token_bytes: usize,
    pub total_bytes: u64,
}

impl GpuNativeDiagnosticTraceLayout {
    pub fn try_new(
        num_layers: usize,
        d_model: usize,
        top_k: usize,
        vocab_size: usize,
    ) -> Result<Self, GpuNativeTokenLoopError> {
        Self::try_new_internal(num_layers, d_model, 0, top_k, vocab_size)
    }

    pub fn try_new_with_router_logits(
        num_layers: usize,
        d_model: usize,
        num_experts: usize,
        top_k: usize,
        vocab_size: usize,
    ) -> Result<Self, GpuNativeTokenLoopError> {
        if num_experts == 0 || top_k > num_experts {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "num_experts must be > 0 and >= top_k".into(),
            });
        }
        Self::try_new_internal(num_layers, d_model, num_experts, top_k, vocab_size)
    }

    fn try_new_internal(
        num_layers: usize,
        d_model: usize,
        num_experts: usize,
        top_k: usize,
        vocab_size: usize,
    ) -> Result<Self, GpuNativeTokenLoopError> {
        if num_layers == 0 {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "num_layers must be > 0".into(),
            });
        }
        if d_model == 0 {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "d_model must be > 0".into(),
            });
        }
        if top_k == 0 || top_k > MAX_GPU_NATIVE_ROUTER_TOP_K {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: format!("top_k must be in 1..={MAX_GPU_NATIVE_ROUTER_TOP_K}"),
            });
        }
        if vocab_size == 0 {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "vocab_size must be > 0".into(),
            });
        }

        let f32_bytes = std::mem::size_of::<f32>();
        let u32_bytes = std::mem::size_of::<u32>();

        let embedding_offset = 0usize;
        let embedding_bytes = d_model.checked_mul(f32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "embedding bytes overflow".into(),
            }
        })?;

        let layer_post_attn_offset =
            embedding_offset
                .checked_add(embedding_bytes)
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "layer_post_attn_offset overflow".into(),
                })?;
        let single_vector_bytes = embedding_bytes;
        let layer_post_attn_bytes =
            num_layers.checked_mul(single_vector_bytes).ok_or_else(|| {
                GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "layer_post_attn_bytes overflow".into(),
                }
            })?;

        let layer_router_input_offset =
            layer_post_attn_offset
                .checked_add(layer_post_attn_bytes)
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "layer_router_input_offset overflow".into(),
                })?;
        let layer_router_input_bytes = layer_post_attn_bytes;

        let layer_router_logits_offset = layer_router_input_offset
            .checked_add(layer_router_input_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "layer_router_logits_offset overflow".into(),
            })?;
        let single_router_logits_bytes = num_experts.checked_mul(f32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "single router logits bytes overflow".into(),
            }
        })?;
        let layer_router_logits_bytes = num_layers
            .checked_mul(single_router_logits_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "layer_router_logits_bytes overflow".into(),
            })?;

        let layer_selected_ids_offset = layer_router_logits_offset
            .checked_add(layer_router_logits_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "layer_selected_ids_offset overflow".into(),
            })?;
        let single_topk_u32_bytes = top_k.checked_mul(u32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "single topk u32 bytes overflow".into(),
            }
        })?;
        let layer_selected_ids_bytes =
            num_layers
                .checked_mul(single_topk_u32_bytes)
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "layer_selected_ids_bytes overflow".into(),
                })?;

        let layer_selected_weights_offset = layer_selected_ids_offset
            .checked_add(layer_selected_ids_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "layer_selected_weights_offset overflow".into(),
            })?;
        let single_topk_f32_bytes = top_k.checked_mul(f32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "single topk f32 bytes overflow".into(),
            }
        })?;
        let layer_selected_weights_bytes = num_layers
            .checked_mul(single_topk_f32_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "layer_selected_weights_bytes overflow".into(),
            })?;

        let layer_post_moe_offset = layer_selected_weights_offset
            .checked_add(layer_selected_weights_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "layer_post_moe_offset overflow".into(),
            })?;
        let layer_post_moe_bytes = layer_post_attn_bytes;

        let layer_status_offset = layer_post_moe_offset
            .checked_add(layer_post_moe_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "layer_status_offset overflow".into(),
            })?;
        let layer_status_bytes = num_layers.checked_mul(u32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "layer_status_bytes overflow".into(),
            }
        })?;

        let final_norm_offset = layer_status_offset
            .checked_add(layer_status_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "final_norm_offset overflow".into(),
            })?;
        let final_norm_bytes = single_vector_bytes;

        let logits_offset = final_norm_offset
            .checked_add(final_norm_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "logits_offset overflow".into(),
            })?;
        let logits_bytes = vocab_size.checked_mul(f32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "logits_bytes overflow".into(),
            }
        })?;

        let final_status_offset = logits_offset.checked_add(logits_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "final_status_offset overflow".into(),
            }
        })?;
        let final_status_bytes = u32_bytes;

        let sampled_token_offset = final_status_offset
            .checked_add(final_status_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "sampled_token_offset overflow".into(),
            })?;
        let sampled_token_bytes = u32_bytes;

        let total_size = sampled_token_offset
            .checked_add(sampled_token_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "total size overflow".into(),
            })?;

        let total_bytes = u64::try_from(total_size).map_err(|_| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "total size exceeds u64".into(),
            }
        })?;

        Ok(Self {
            num_layers,
            d_model,
            num_experts,
            top_k,
            vocab_size,
            embedding_offset,
            embedding_bytes,
            layer_post_attn_offset,
            layer_post_attn_bytes,
            layer_router_input_offset,
            layer_router_input_bytes,
            layer_router_logits_offset,
            layer_router_logits_bytes,
            layer_selected_ids_offset,
            layer_selected_ids_bytes,
            layer_selected_weights_offset,
            layer_selected_weights_bytes,
            layer_post_moe_offset,
            layer_post_moe_bytes,
            layer_status_offset,
            layer_status_bytes,
            final_norm_offset,
            final_norm_bytes,
            logits_offset,
            logits_bytes,
            final_status_offset,
            final_status_bytes,
            sampled_token_offset,
            sampled_token_bytes,
            total_bytes,
        })
    }

    #[inline]
    pub fn layer_post_attn_offset(&self, layer: usize) -> u64 {
        (self.layer_post_attn_offset + layer * self.d_model * 4) as u64
    }

    #[inline]
    pub fn layer_router_input_offset(&self, layer: usize) -> u64 {
        (self.layer_router_input_offset + layer * self.d_model * 4) as u64
    }

    #[inline]
    pub fn layer_router_logits_offset(&self, layer: usize) -> u64 {
        (self.layer_router_logits_offset + layer * self.num_experts * 4) as u64
    }

    #[inline]
    pub fn layer_selected_ids_offset(&self, layer: usize) -> u64 {
        (self.layer_selected_ids_offset + layer * self.top_k * 4) as u64
    }

    #[inline]
    pub fn layer_selected_weights_offset(&self, layer: usize) -> u64 {
        (self.layer_selected_weights_offset + layer * self.top_k * 4) as u64
    }

    #[inline]
    pub fn layer_post_moe_offset(&self, layer: usize) -> u64 {
        (self.layer_post_moe_offset + layer * self.d_model * 4) as u64
    }

    #[inline]
    pub fn layer_status_offset(&self, layer: usize) -> u64 {
        (self.layer_status_offset + layer * 4) as u64
    }

    pub fn parse(&self, bytes: &[u8]) -> Result<GpuNativeDiagnosticTrace, GpuNativeTokenLoopError> {
        if (bytes.len() as u64) < self.total_bytes {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: format!(
                    "diagnostic trace byte buffer too small: expected {}, got {}",
                    self.total_bytes,
                    bytes.len()
                ),
            });
        }

        let parse_f32_vec = |offset: usize, count: usize| -> Vec<f32> {
            let slice = &bytes[offset..offset + count * 4];
            slice
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        };

        let parse_u32_vec = |offset: usize, count: usize| -> Vec<u32> {
            let slice = &bytes[offset..offset + count * 4];
            slice
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        };

        let embedding = parse_f32_vec(self.embedding_offset, self.d_model);

        let mut layer_post_attn = Vec::with_capacity(self.num_layers);
        let mut layer_router_input = Vec::with_capacity(self.num_layers);
        let mut layer_router_logits = Vec::with_capacity(self.num_layers);
        let mut layer_selected_ids = Vec::with_capacity(self.num_layers);
        let mut layer_selected_weights = Vec::with_capacity(self.num_layers);
        let mut layer_post_moe = Vec::with_capacity(self.num_layers);
        let mut layer_statuses = Vec::with_capacity(self.num_layers);

        for l in 0..self.num_layers {
            layer_post_attn.push(parse_f32_vec(
                self.layer_post_attn_offset(l) as usize,
                self.d_model,
            ));
            layer_router_input.push(parse_f32_vec(
                self.layer_router_input_offset(l) as usize,
                self.d_model,
            ));
            layer_router_logits.push(parse_f32_vec(
                self.layer_router_logits_offset(l) as usize,
                self.num_experts,
            ));
            layer_selected_ids.push(parse_u32_vec(
                self.layer_selected_ids_offset(l) as usize,
                self.top_k,
            ));
            layer_selected_weights.push(parse_f32_vec(
                self.layer_selected_weights_offset(l) as usize,
                self.top_k,
            ));
            layer_post_moe.push(parse_f32_vec(
                self.layer_post_moe_offset(l) as usize,
                self.d_model,
            ));
            let status_off = self.layer_status_offset(l) as usize;
            layer_statuses.push(u32::from_le_bytes(
                bytes[status_off..status_off + 4].try_into().unwrap(),
            ));
        }

        let final_norm = parse_f32_vec(self.final_norm_offset, self.d_model);
        let logits = parse_f32_vec(self.logits_offset, self.vocab_size);
        let final_status = u32::from_le_bytes(
            bytes[self.final_status_offset..self.final_status_offset + 4]
                .try_into()
                .unwrap(),
        );
        let sampled_token = u32::from_le_bytes(
            bytes[self.sampled_token_offset..self.sampled_token_offset + 4]
                .try_into()
                .unwrap(),
        );

        Ok(GpuNativeDiagnosticTrace {
            embedding,
            layer_post_attn,
            layer_router_input,
            layer_router_logits,
            layer_selected_ids,
            layer_selected_weights,
            layer_post_moe,
            layer_statuses,
            final_norm,
            logits,
            final_status,
            sampled_token,
        })
    }
}

/// Parsed results of one full GPU-native diagnostic execution step.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuNativeDiagnosticTrace {
    pub embedding: Vec<f32>,
    pub layer_post_attn: Vec<Vec<f32>>,
    pub layer_router_input: Vec<Vec<f32>>,
    pub layer_router_logits: Vec<Vec<f32>>,
    pub layer_selected_ids: Vec<Vec<u32>>,
    pub layer_selected_weights: Vec<Vec<f32>>,
    pub layer_post_moe: Vec<Vec<f32>>,
    pub layer_statuses: Vec<u32>,
    pub final_norm: Vec<f32>,
    pub logits: Vec<f32>,
    pub final_status: u32,
    pub sampled_token: u32,
}

/// Diagnostic-only output from running the unmodified production GPU router
/// against a caller-supplied router input.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuNativeRouterShadowTrace {
    pub raw_logits: Vec<f32>,
    pub selected_ids: Vec<u32>,
    pub selected_weights: Vec<f32>,
    pub status: u32,
}

/// Captured boundaries from the authoritative reference model forward.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelDiagnosticTrace {
    pub embedding: Vec<f32>,
    pub layer_post_attn: Vec<Vec<f32>>,
    pub layer_router_input: Vec<Vec<f32>>,
    pub layer_router_logits: Vec<Vec<f32>>,
    pub layer_selected_ids: Vec<Vec<u32>>,
    pub layer_selected_weights: Vec<Vec<f32>>,
    pub layer_post_moe: Vec<Vec<f32>>,
    pub final_norm: Vec<f32>,
    pub logits: Vec<f32>,
    pub sampled_token: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum DiagnosticStage {
    Embedding,
    LayerPostAttention { layer: usize },
    LayerRouterInput { layer: usize },
    LayerSelectedExpertIds { layer: usize },
    LayerSelectedExpertWeights { layer: usize },
    LayerPostMoe { layer: usize },
    FinalNorm,
    Logits,
    SampledToken,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DiscreteComparisonEvidence {
    pub reference: Vec<u32>,
    pub gpu_native: Vec<u32>,
    pub matches: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StageComparisonEvidence {
    pub stage: DiagnosticStage,
    pub exact_match: bool,
    pub tolerance_pass: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_comparison: Option<VectorComparisonEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discrete_comparison: Option<DiscreteComparisonEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FirstDivergenceEvidence {
    pub stage: DiagnosticStage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_absolute_error: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rms_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worst_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worst_reference_value: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worst_gpu_native_value: Option<f32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CaseEvidence {
    pub case_name: String,
    pub prompt_sha256: String,
    pub prompt_token_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuNativeFirstDivergenceReport {
    pub schema_version: String,
    pub mode: String,
    pub diagnostic_complete: bool,
    pub qualification_pass: bool,
    pub failure: Option<String>,

    pub provenance: BuildProvenance,
    pub build_git_sha: String,
    pub executable_sha256: String,
    pub resolved_config_sha256: String,
    pub model_geometry: GpuNativeModelGeometry,
    pub expert_metadata: ExpertMetadataEvidence,
    pub case: CaseEvidence,
    pub prompt_token_ids_sha256: String,
    pub sampling: GreedySamplingEvidence,
    pub expected_adapter_name: String,
    pub actual_adapter: Option<GpuDeviceIdentity>,

    pub reference_sampled_token_id: Option<u32>,
    pub gpu_native_sampled_token_id: Option<u32>,
    pub token_match: Option<bool>,

    pub first_exact_divergence: Option<FirstDivergenceEvidence>,
    pub first_divergence: Option<FirstDivergenceEvidence>,
    pub stages: Vec<StageComparisonEvidence>,

    pub token_loop_counters_delta: GpuNativeTokenLoopSnapshot,
    pub attempt_count: usize,
}

impl GpuNativeFirstDivergenceReport {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provenance: BuildProvenance,
        build_git_sha: String,
        executable_sha256: String,
        resolved_config_sha256: String,
        model_geometry: GpuNativeModelGeometry,
        expert_metadata: ExpertMetadataEvidence,
        case: CaseEvidence,
        prompt_token_ids_sha256: String,
        expected_adapter_name: String,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION.to_string(),
            mode: MODE.to_string(),
            diagnostic_complete: false,
            qualification_pass: false,
            failure: None,
            provenance,
            build_git_sha,
            executable_sha256,
            resolved_config_sha256,
            model_geometry,
            expert_metadata,
            case,
            prompt_token_ids_sha256,
            sampling: GreedySamplingEvidence::fixed(),
            expected_adapter_name,
            actual_adapter: None,
            reference_sampled_token_id: None,
            gpu_native_sampled_token_id: None,
            token_match: None,
            first_exact_divergence: None,
            first_divergence: None,
            stages: Vec::new(),
            token_loop_counters_delta: GpuNativeTokenLoopSnapshot::default(),
            attempt_count: 0,
        }
    }

    pub fn fail(&mut self, reason: String) {
        self.diagnostic_complete = true;
        self.qualification_pass = false;
        self.failure = Some(reason);
    }
}

struct DiagnosticTraceComparator {
    stages: Vec<StageComparisonEvidence>,
    first_exact_divergence: Option<FirstDivergenceEvidence>,
    first_divergence: Option<FirstDivergenceEvidence>,
}

impl DiagnosticTraceComparator {
    fn new() -> Self {
        Self {
            stages: Vec::new(),
            first_exact_divergence: None,
            first_divergence: None,
        }
    }

    fn record_vector_stage(
        &mut self,
        stage: DiagnosticStage,
        layer: Option<usize>,
        ref_vec: &[f32],
        gpu_vec: &[f32],
        tolerance: Option<ErrorTolerance>,
    ) -> Result<(), String> {
        let comp = crate::numerical_diagnostics::compare_vectors(ref_vec, gpu_vec)?;
        let nonfinite = comp.cpu_nonfinite_count > 0 || comp.hybrid_nonfinite_count > 0;

        let tolerance_pass = if nonfinite {
            false
        } else if let Some(tol) = tolerance {
            let mut pass = true;
            for (&r, &g) in ref_vec.iter().zip(gpu_vec) {
                if !tol.evaluates(r, g) {
                    pass = false;
                    break;
                }
            }
            pass
        } else {
            comp.exact_f32_bits
        };

        let stage_evidence = StageComparisonEvidence {
            stage: stage.clone(),
            exact_match: comp.exact_f32_bits,
            tolerance_pass,
            vector_comparison: Some(comp.clone()),
            discrete_comparison: None,
        };

        if !comp.exact_f32_bits && self.first_exact_divergence.is_none() {
            self.first_exact_divergence = Some(FirstDivergenceEvidence {
                stage: stage.clone(),
                layer,
                reason: if nonfinite {
                    format!(
                        "nonfinite value: reference nonfinite={}, gpu_native nonfinite={}",
                        comp.cpu_nonfinite_count, comp.hybrid_nonfinite_count
                    )
                } else {
                    format!(
                        "exact f32 bits differ (max absolute error {:?}, RMS error {:?})",
                        comp.max_absolute_error, comp.rms_error
                    )
                },
                max_absolute_error: comp.max_absolute_error,
                rms_error: comp.rms_error,
                worst_index: comp.max_error.as_ref().map(|e| e.index),
                worst_reference_value: comp.max_error.as_ref().and_then(|e| e.cpu.value),
                worst_gpu_native_value: comp.max_error.as_ref().and_then(|e| e.hybrid.value),
            });
        }

        if !tolerance_pass && self.first_divergence.is_none() {
            let reason = if nonfinite {
                format!(
                    "nonfinite value detected: reference nonfinite={}, gpu_native nonfinite={}",
                    comp.cpu_nonfinite_count, comp.hybrid_nonfinite_count
                )
            } else if let Some(tol) = tolerance {
                if let Some(ref max_err) = comp.max_error {
                    format!(
                        "max absolute error {:?} exceeds reference tolerance (abs={}, rel={}) at index {}: ref={:?}, gpu={:?}",
                        comp.max_absolute_error, tol.absolute, tol.relative, max_err.index, max_err.cpu, max_err.hybrid
                    )
                } else {
                    "tolerance check failed".to_string()
                }
            } else {
                format!(
                    "exact f32 bit mismatch at stage with no formal multi-layer tolerance contract (max abs error {:?})",
                    comp.max_absolute_error
                )
            };

            self.first_divergence = Some(FirstDivergenceEvidence {
                stage,
                layer,
                reason,
                max_absolute_error: comp.max_absolute_error,
                rms_error: comp.rms_error,
                worst_index: comp.max_error.as_ref().map(|e| e.index),
                worst_reference_value: comp.max_error.as_ref().and_then(|e| e.cpu.value),
                worst_gpu_native_value: comp.max_error.as_ref().and_then(|e| e.hybrid.value),
            });
        }

        self.stages.push(stage_evidence);
        Ok(())
    }

    fn record_discrete_stage(
        &mut self,
        stage: DiagnosticStage,
        layer: Option<usize>,
        ref_vals: &[u32],
        gpu_vals: &[u32],
    ) -> Result<(), String> {
        if ref_vals.len() != gpu_vals.len() {
            return Err(format!(
                "discrete stage length mismatch: ref={}, gpu={}",
                ref_vals.len(),
                gpu_vals.len()
            ));
        }
        let matches = ref_vals == gpu_vals;
        let stage_evidence = StageComparisonEvidence {
            stage: stage.clone(),
            exact_match: matches,
            tolerance_pass: matches,
            vector_comparison: None,
            discrete_comparison: Some(DiscreteComparisonEvidence {
                reference: ref_vals.to_vec(),
                gpu_native: gpu_vals.to_vec(),
                matches,
            }),
        };

        if !matches && self.first_exact_divergence.is_none() {
            self.first_exact_divergence = Some(FirstDivergenceEvidence {
                stage: stage.clone(),
                layer,
                reason: format!(
                    "discrete values mismatch: reference {:?}, gpu_native {:?}",
                    ref_vals, gpu_vals
                ),
                max_absolute_error: None,
                rms_error: None,
                worst_index: None,
                worst_reference_value: None,
                worst_gpu_native_value: None,
            });
        }

        if !matches && self.first_divergence.is_none() {
            self.first_divergence = Some(FirstDivergenceEvidence {
                stage,
                layer,
                reason: format!(
                    "discrete values mismatch: reference {:?}, gpu_native {:?}",
                    ref_vals, gpu_vals
                ),
                max_absolute_error: None,
                rms_error: None,
                worst_index: None,
                worst_reference_value: None,
                worst_gpu_native_value: None,
            });
        }

        self.stages.push(stage_evidence);
        Ok(())
    }
}

pub fn compare_diagnostic_traces(
    reference: &ModelDiagnosticTrace,
    gpu_native: &GpuNativeDiagnosticTrace,
) -> Result<
    (
        Option<FirstDivergenceEvidence>,
        Option<FirstDivergenceEvidence>,
        Vec<StageComparisonEvidence>,
    ),
    String,
> {
    let mut comparator = DiagnosticTraceComparator::new();

    // 1. Embedding: expected exact f32 weights
    comparator.record_vector_stage(
        DiagnosticStage::Embedding,
        None,
        &reference.embedding,
        &gpu_native.embedding,
        None,
    )?;

    // 2. Per-layer stages
    let num_layers = reference.layer_post_attn.len();
    if gpu_native.layer_post_attn.len() != num_layers
        || reference.layer_router_input.len() != num_layers
        || gpu_native.layer_router_input.len() != num_layers
        || reference.layer_selected_ids.len() != num_layers
        || gpu_native.layer_selected_ids.len() != num_layers
        || reference.layer_selected_weights.len() != num_layers
        || gpu_native.layer_selected_weights.len() != num_layers
        || reference.layer_post_moe.len() != num_layers
        || gpu_native.layer_post_moe.len() != num_layers
    {
        return Err("layer count mismatch between reference and GPU-native traces".to_string());
    }

    for l in 0..num_layers {
        comparator.record_vector_stage(
            DiagnosticStage::LayerPostAttention { layer: l },
            Some(l),
            &reference.layer_post_attn[l],
            &gpu_native.layer_post_attn[l],
            None,
        )?;

        comparator.record_vector_stage(
            DiagnosticStage::LayerRouterInput { layer: l },
            Some(l),
            &reference.layer_router_input[l],
            &gpu_native.layer_router_input[l],
            None,
        )?;

        comparator.record_discrete_stage(
            DiagnosticStage::LayerSelectedExpertIds { layer: l },
            Some(l),
            &reference.layer_selected_ids[l],
            &gpu_native.layer_selected_ids[l],
        )?;

        comparator.record_vector_stage(
            DiagnosticStage::LayerSelectedExpertWeights { layer: l },
            Some(l),
            &reference.layer_selected_weights[l],
            &gpu_native.layer_selected_weights[l],
            None,
        )?;

        comparator.record_vector_stage(
            DiagnosticStage::LayerPostMoe { layer: l },
            Some(l),
            &reference.layer_post_moe[l],
            &gpu_native.layer_post_moe[l],
            None,
        )?;
    }

    // 3. Final Norm
    comparator.record_vector_stage(
        DiagnosticStage::FinalNorm,
        None,
        &reference.final_norm,
        &gpu_native.final_norm,
        None,
    )?;

    // 4. Logits
    comparator.record_vector_stage(
        DiagnosticStage::Logits,
        None,
        &reference.logits,
        &gpu_native.logits,
        None,
    )?;

    // 5. Sampled Token: exact discrete match
    comparator.record_discrete_stage(
        DiagnosticStage::SampledToken,
        None,
        &[reference.sampled_token],
        &[gpu_native.sampled_token],
    )?;

    Ok((
        comparator.first_exact_divergence,
        comparator.first_divergence,
        comparator.stages,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_layout_offset_arithmetic_is_exact_and_contiguous() {
        let layout = GpuNativeDiagnosticTraceLayout::try_new(2, 4, 2, 8).unwrap();
        // embedding: 0..16
        assert_eq!(layout.embedding_offset, 0);
        assert_eq!(layout.embedding_bytes, 16);
        // layer_post_attn: 16..48 (2 * 16 = 32)
        assert_eq!(layout.layer_post_attn_offset, 16);
        assert_eq!(layout.layer_post_attn_bytes, 32);
        assert_eq!(layout.layer_post_attn_offset(0), 16);
        assert_eq!(layout.layer_post_attn_offset(1), 32);
        // layer_router_input: 48..80 (32)
        assert_eq!(layout.layer_router_input_offset, 48);
        assert_eq!(layout.layer_router_input_bytes, 32);
        assert_eq!(layout.layer_router_input_offset(0), 48);
        assert_eq!(layout.layer_router_input_offset(1), 64);
        // Frozen trace layout does not include raw router logits.
        assert_eq!(layout.layer_router_logits_offset, 80);
        assert_eq!(layout.layer_router_logits_bytes, 0);
        // layer_selected_ids: 80..96 (2 * 2 * 4 = 16)
        assert_eq!(layout.layer_selected_ids_offset, 80);
        assert_eq!(layout.layer_selected_ids_bytes, 16);
        assert_eq!(layout.layer_selected_ids_offset(0), 80);
        assert_eq!(layout.layer_selected_ids_offset(1), 88);
        // layer_selected_weights: 96..112 (16)
        assert_eq!(layout.layer_selected_weights_offset, 96);
        assert_eq!(layout.layer_selected_weights_bytes, 16);
        assert_eq!(layout.layer_selected_weights_offset(0), 96);
        assert_eq!(layout.layer_selected_weights_offset(1), 104);
        // layer_post_moe: 112..144 (32)
        assert_eq!(layout.layer_post_moe_offset, 112);
        assert_eq!(layout.layer_post_moe_bytes, 32);
        assert_eq!(layout.layer_post_moe_offset(0), 112);
        assert_eq!(layout.layer_post_moe_offset(1), 128);
        // layer_status: 144..152 (2 * 4 = 8)
        assert_eq!(layout.layer_status_offset, 144);
        assert_eq!(layout.layer_status_bytes, 8);
        assert_eq!(layout.layer_status_offset(0), 144);
        assert_eq!(layout.layer_status_offset(1), 148);
        // final_norm: 152..168 (16)
        assert_eq!(layout.final_norm_offset, 152);
        assert_eq!(layout.final_norm_bytes, 16);
        // logits: 168..200 (8 * 4 = 32)
        assert_eq!(layout.logits_offset, 168);
        assert_eq!(layout.logits_bytes, 32);
        // final_status: 200..204 (4)
        assert_eq!(layout.final_status_offset, 200);
        assert_eq!(layout.final_status_bytes, 4);
        // sampled_token: 204..208 (4)
        assert_eq!(layout.sampled_token_offset, 204);
        assert_eq!(layout.sampled_token_bytes, 4);
        // total: 208
        assert_eq!(layout.total_bytes, 208);
    }

    #[test]
    fn router_logit_extension_is_explicit_and_contiguous() {
        let layout =
            GpuNativeDiagnosticTraceLayout::try_new_with_router_logits(2, 4, 4, 2, 8).unwrap();
        assert_eq!(layout.layer_router_logits_offset, 80);
        assert_eq!(layout.layer_router_logits_bytes, 32);
        assert_eq!(layout.layer_router_logits_offset(0), 80);
        assert_eq!(layout.layer_router_logits_offset(1), 96);
        assert_eq!(layout.layer_selected_ids_offset, 112);
        assert_eq!(layout.total_bytes, 240);

        let mut bytes = vec![0u8; layout.total_bytes as usize];
        let captured = [1.0f32, 2.0, 3.0, 4.0];
        let offset = layout.layer_router_logits_offset(1) as usize;
        for (index, value) in captured.iter().enumerate() {
            bytes[offset + index * 4..offset + index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        let trace = layout.parse(&bytes).unwrap();
        assert_eq!(trace.layer_router_logits[1], captured);
    }

    #[test]
    fn trace_layout_overflow_rejections() {
        assert!(GpuNativeDiagnosticTraceLayout::try_new(0, 4, 2, 8).is_err());
        assert!(GpuNativeDiagnosticTraceLayout::try_new(2, 0, 2, 8).is_err());
        assert!(GpuNativeDiagnosticTraceLayout::try_new(2, 4, 0, 8).is_err());
        assert!(GpuNativeDiagnosticTraceLayout::try_new(2, 4, 9999, 8).is_err());
        assert!(GpuNativeDiagnosticTraceLayout::try_new(2, 4, 2, 0).is_err());
        assert!(GpuNativeDiagnosticTraceLayout::try_new(usize::MAX, 4, 2, 8).is_err());
        assert!(GpuNativeDiagnosticTraceLayout::try_new_with_router_logits(2, 4, 0, 2, 8).is_err());
        assert!(GpuNativeDiagnosticTraceLayout::try_new_with_router_logits(2, 4, 1, 2, 8).is_err());
    }

    #[test]
    fn trace_layout_parse_exact_round_trip() {
        let layout = GpuNativeDiagnosticTraceLayout::try_new(2, 4, 2, 4).unwrap();
        let mut bytes = vec![0u8; layout.total_bytes as usize];

        // embedding: [1.0, 2.0, 3.0, 4.0]
        let emb = [1.0f32, 2.0, 3.0, 4.0];
        for (i, v) in emb.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }

        // layer 0 post_attn
        let l0_attn = [10.0f32, 11.0, 12.0, 13.0];
        let off = layout.layer_post_attn_offset(0) as usize;
        for (i, v) in l0_attn.iter().enumerate() {
            bytes[off + i * 4..off + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }

        // layer 1 selected ids
        let l1_ids = [5u32, 7u32];
        let off = layout.layer_selected_ids_offset(1) as usize;
        for (i, v) in l1_ids.iter().enumerate() {
            bytes[off + i * 4..off + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }

        // final_norm
        let fnorm = [50.0f32, 51.0, 52.0, 53.0];
        let off = layout.final_norm_offset;
        for (i, v) in fnorm.iter().enumerate() {
            bytes[off + i * 4..off + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }

        // sampled_token
        bytes[layout.sampled_token_offset..layout.sampled_token_offset + 4]
            .copy_from_slice(&42u32.to_le_bytes());

        let trace = layout.parse(&bytes).unwrap();
        assert_eq!(trace.embedding, emb);
        assert_eq!(trace.layer_post_attn[0], l0_attn);
        assert_eq!(trace.layer_selected_ids[1], l1_ids);
        assert_eq!(trace.final_norm, fnorm);
        assert_eq!(trace.sampled_token, 42);

        // Under-sized buffer fails
        assert!(layout
            .parse(&bytes[..layout.total_bytes as usize - 1])
            .is_err());
    }

    fn sample_trace(
        layers: usize,
        d_model: usize,
        top_k: usize,
        vocab: usize,
    ) -> ModelDiagnosticTrace {
        ModelDiagnosticTrace {
            embedding: vec![1.0; d_model],
            layer_post_attn: vec![vec![2.0; d_model]; layers],
            layer_router_input: vec![vec![3.0; d_model]; layers],
            layer_router_logits: vec![vec![0.25; top_k]; layers],
            layer_selected_ids: vec![vec![0; top_k]; layers],
            layer_selected_weights: vec![vec![0.5; top_k]; layers],
            layer_post_moe: vec![vec![4.0; d_model]; layers],
            final_norm: vec![5.0; d_model],
            logits: vec![6.0; vocab],
            sampled_token: 5212,
        }
    }

    fn model_to_gpu_trace(m: &ModelDiagnosticTrace) -> GpuNativeDiagnosticTrace {
        GpuNativeDiagnosticTrace {
            embedding: m.embedding.clone(),
            layer_post_attn: m.layer_post_attn.clone(),
            layer_router_input: m.layer_router_input.clone(),
            layer_router_logits: m.layer_router_logits.clone(),
            layer_selected_ids: m.layer_selected_ids.clone(),
            layer_selected_weights: m.layer_selected_weights.clone(),
            layer_post_moe: m.layer_post_moe.clone(),
            layer_statuses: vec![0; m.layer_post_attn.len()],
            final_norm: m.final_norm.clone(),
            logits: m.logits.clone(),
            final_status: 0,
            sampled_token: m.sampled_token,
        }
    }

    #[test]
    fn no_divergence_when_identical() {
        let reference = sample_trace(2, 4, 2, 8);
        let gpu = model_to_gpu_trace(&reference);

        let (first_exact, first_div, stages) = compare_diagnostic_traces(&reference, &gpu).unwrap();
        assert!(first_exact.is_none());
        assert!(first_div.is_none());
        assert!(stages.iter().all(|s| s.tolerance_pass && s.exact_match));
    }

    #[test]
    fn first_divergence_embedding() {
        let reference = sample_trace(2, 4, 2, 8);
        let mut gpu = model_to_gpu_trace(&reference);
        gpu.embedding[0] = 100.0;

        let (_first_exact, first_div, _) = compare_diagnostic_traces(&reference, &gpu).unwrap();
        assert!(first_div.is_some());
        let div = first_div.unwrap();
        assert_eq!(div.stage, DiagnosticStage::Embedding);
        assert_eq!(div.layer, None);
        assert_eq!(div.worst_index, Some(0));
    }

    #[test]
    fn first_divergence_layer_post_attention() {
        let reference = sample_trace(2, 4, 2, 8);
        let mut gpu = model_to_gpu_trace(&reference);
        gpu.layer_post_attn[1][2] = 99.0;

        let (_first_exact, first_div, _) = compare_diagnostic_traces(&reference, &gpu).unwrap();
        assert!(first_div.is_some());
        let div = first_div.unwrap();
        assert_eq!(div.stage, DiagnosticStage::LayerPostAttention { layer: 1 });
        assert_eq!(div.layer, Some(1));
    }

    #[test]
    fn first_divergence_layer_router_input() {
        let reference = sample_trace(2, 4, 2, 8);
        let mut gpu = model_to_gpu_trace(&reference);
        gpu.layer_router_input[0][1] = 99.0;

        let (_first_exact, first_div, _) = compare_diagnostic_traces(&reference, &gpu).unwrap();
        assert!(first_div.is_some());
        let div = first_div.unwrap();
        assert_eq!(div.stage, DiagnosticStage::LayerRouterInput { layer: 0 });
        assert_eq!(div.layer, Some(0));
    }

    #[test]
    fn first_divergence_layer_routing_ids() {
        let reference = sample_trace(2, 4, 2, 8);
        let mut gpu = model_to_gpu_trace(&reference);
        gpu.layer_selected_ids[0][1] = 99;

        let (_first_exact, first_div, _) = compare_diagnostic_traces(&reference, &gpu).unwrap();
        assert!(first_div.is_some());
        let div = first_div.unwrap();
        assert_eq!(
            div.stage,
            DiagnosticStage::LayerSelectedExpertIds { layer: 0 }
        );
        assert_eq!(div.layer, Some(0));
    }

    #[test]
    fn first_divergence_layer_routing_weights() {
        let reference = sample_trace(2, 4, 2, 8);
        let mut gpu = model_to_gpu_trace(&reference);
        gpu.layer_selected_weights[1][0] = 0.99;

        let (_first_exact, first_div, _) = compare_diagnostic_traces(&reference, &gpu).unwrap();
        assert!(first_div.is_some());
        let div = first_div.unwrap();
        assert_eq!(
            div.stage,
            DiagnosticStage::LayerSelectedExpertWeights { layer: 1 }
        );
        assert_eq!(div.layer, Some(1));
    }

    #[test]
    fn first_divergence_layer_post_moe() {
        let reference = sample_trace(2, 4, 2, 8);
        let mut gpu = model_to_gpu_trace(&reference);
        gpu.layer_post_moe[1][0] = 99.0;

        let (_first_exact, first_div, _) = compare_diagnostic_traces(&reference, &gpu).unwrap();
        assert!(first_div.is_some());
        let div = first_div.unwrap();
        assert_eq!(div.stage, DiagnosticStage::LayerPostMoe { layer: 1 });
        assert_eq!(div.layer, Some(1));
    }

    #[test]
    fn first_divergence_final_norm() {
        let reference = sample_trace(2, 4, 2, 8);
        let mut gpu = model_to_gpu_trace(&reference);
        gpu.final_norm[3] = 99.0;

        let (_first_exact, first_div, _) = compare_diagnostic_traces(&reference, &gpu).unwrap();
        assert!(first_div.is_some());
        let div = first_div.unwrap();
        assert_eq!(div.stage, DiagnosticStage::FinalNorm);
    }

    #[test]
    fn first_divergence_logits() {
        let reference = sample_trace(2, 4, 2, 8);
        let mut gpu = model_to_gpu_trace(&reference);
        gpu.logits[5] = 99.0;

        let (_first_exact, first_div, _) = compare_diagnostic_traces(&reference, &gpu).unwrap();
        assert!(first_div.is_some());
        let div = first_div.unwrap();
        assert_eq!(div.stage, DiagnosticStage::Logits);
    }

    #[test]
    fn first_divergence_sampled_token() {
        let reference = sample_trace(2, 4, 2, 8);
        let mut gpu = model_to_gpu_trace(&reference);
        gpu.sampled_token = 715;

        let (_first_exact, first_div, _) = compare_diagnostic_traces(&reference, &gpu).unwrap();
        assert!(first_div.is_some());
        let div = first_div.unwrap();
        assert_eq!(div.stage, DiagnosticStage::SampledToken);
    }

    #[test]
    fn nonfinite_handling_fails_tolerance_and_flags_divergence() {
        let reference = sample_trace(2, 4, 2, 8);
        let mut gpu = model_to_gpu_trace(&reference);
        gpu.embedding[0] = f32::NAN;

        let (_first_exact, first_div, stages) =
            compare_diagnostic_traces(&reference, &gpu).unwrap();
        assert!(first_div.is_some());
        assert_eq!(first_div.unwrap().stage, DiagnosticStage::Embedding);
        assert!(!stages[0].tolerance_pass);
    }

    #[test]
    fn length_mismatch_fails_closed() {
        let mut reference = sample_trace(2, 4, 2, 8);
        let gpu = model_to_gpu_trace(&reference);
        reference.layer_post_attn.pop();

        assert!(compare_diagnostic_traces(&reference, &gpu).is_err());
    }

    #[test]
    fn report_cannot_claim_qualification_pass() {
        let report = GpuNativeFirstDivergenceReport::new(
            BuildProvenance::embedded(),
            "0123456789012345678901234567890123456789".to_string(),
            "a".repeat(64),
            "b".repeat(64),
            GpuNativeModelGeometry {
                num_layers: 48,
                d_model: 2048,
                d_ff: 1024,
                num_experts: 64,
                top_k: 8,
                num_heads: 16,
                num_kv_heads: 2,
                head_dim: 128,
                rope_dim: 64,
                vocab_size: 151936,
                max_seq_len: 2048,
                rms_eps: 1e-6,
                rope_base: 10000.0,
            },
            ExpertMetadataEvidence {
                dtype: Some("q4_0".to_string()),
                q4_0_layout: Some("canonical".to_string()),
                conversion_mode: Some("native".to_string()),
                source: Some("checkpoint".to_string()),
                explicitly_synthetic: false,
            },
            CaseEvidence {
                case_name: TARGET_CASE.to_string(),
                prompt_sha256: "c".repeat(64),
                prompt_token_count: 32,
            },
            "d".repeat(64),
            "NVIDIA L4".to_string(),
        );

        assert!(!report.qualification_pass);
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"qualification_pass\":false"));
    }

    #[test]
    fn small_delta_on_cumulative_stage_flags_first_divergence_with_no_tolerance() {
        // A small delta (1e-4) on layer_post_moe would have satisfied the old
        // Q4_EXPERT_F16_BOUNDARY_TOLERANCE (abs=2e-3, rel=5e-3). With tolerance=None,
        // it must now immediately flag first divergence and exact divergence.
        let reference = sample_trace(2, 4, 2, 8);
        let mut gpu = model_to_gpu_trace(&reference);
        gpu.layer_post_moe[0][0] = 4.0 + 1e-4;

        let (first_exact, first_div, stages) = compare_diagnostic_traces(&reference, &gpu).unwrap();

        assert!(first_exact.is_some());
        assert_eq!(
            first_exact.unwrap().stage,
            DiagnosticStage::LayerPostMoe { layer: 0 }
        );

        assert!(first_div.is_some());
        let div = first_div.unwrap();
        assert_eq!(div.stage, DiagnosticStage::LayerPostMoe { layer: 0 });
        assert_eq!(div.layer, Some(0));

        let moe_stage = stages
            .iter()
            .find(|s| s.stage == DiagnosticStage::LayerPostMoe { layer: 0 })
            .unwrap();
        assert!(!moe_stage.exact_match);
        assert!(!moe_stage.tolerance_pass);

        // Same for small delta on Logits (1e-4)
        let mut gpu_logits = model_to_gpu_trace(&reference);
        gpu_logits.logits[0] = 6.0 + 1e-4;
        let (first_exact_l, first_div_l, stages_l) =
            compare_diagnostic_traces(&reference, &gpu_logits).unwrap();
        assert_eq!(first_exact_l.unwrap().stage, DiagnosticStage::Logits);
        assert_eq!(first_div_l.unwrap().stage, DiagnosticStage::Logits);
        let logits_stage = stages_l
            .iter()
            .find(|s| s.stage == DiagnosticStage::Logits)
            .unwrap();
        assert!(!logits_stage.exact_match);
        assert!(!logits_stage.tolerance_pass);

        // Same for small delta on LayerPostAttention (1e-6, which satisfied RAW_PROJECTION_TOLERANCE)
        let mut gpu_attn = model_to_gpu_trace(&reference);
        gpu_attn.layer_post_attn[0][0] = 2.0 + 1e-6;
        let (first_exact_a, first_div_a, stages_a) =
            compare_diagnostic_traces(&reference, &gpu_attn).unwrap();
        assert_eq!(
            first_exact_a.unwrap().stage,
            DiagnosticStage::LayerPostAttention { layer: 0 }
        );
        assert_eq!(
            first_div_a.unwrap().stage,
            DiagnosticStage::LayerPostAttention { layer: 0 }
        );
        let attn_stage = stages_a
            .iter()
            .find(|s| s.stage == DiagnosticStage::LayerPostAttention { layer: 0 })
            .unwrap();
        assert!(!attn_stage.exact_match);
        assert!(!attn_stage.tolerance_pass);
    }
}
