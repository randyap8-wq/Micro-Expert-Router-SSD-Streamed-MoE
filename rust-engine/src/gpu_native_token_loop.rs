//! GPU-native, GPU-owned autoregressive token loop.
//!
//! Owns the execution of the entire forward transformer pass on GPU:
//! Embedding lookup -> Layer (Attention RMSNorm -> QKV/RoPE/KV -> Causal Attention/O -> MoE RMSNorm -> Router -> Q4 Expert Combine) -> Final RMSNorm -> LM Head -> GPU Greedy Argmax.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex as TokioMutex;

use crate::architecture::Architecture;
use crate::backend::gpu_native::{
    GpuNativeAttentionGeometry, GpuNativeAttentionNorm, GpuNativeAttentionPlan,
    GpuNativeAttentionScratch, GpuNativeBootstrapError, GpuNativeDenseWeightHandle,
    GpuNativeDenseWeightKey, GpuNativeExecutorContext, GpuNativeKvState,
    GpuNativePreExpertCheckpointLayout, GpuNativePreExpertCheckpoints, GpuNativeQ4ExpertGeometry,
    GpuNativeQ4ExpertScratch, GpuNativeRmsNormHandle, GpuNativeRouterGeometry, GpuNativeRouterPlan,
    GpuNativeRouterScratch, GpuNativeScratch, GpuNativeTokenState, GPU_NATIVE_STATUS_FATAL_MASK,
    GPU_NATIVE_STATUS_RETRYABLE_MASK, MAX_GPU_NATIVE_ROUTER_EXPERTS, MAX_GPU_NATIVE_ROUTER_TOP_K,
};
use crate::dense_tensor::DenseDType;
use crate::engine::{Engine, GpuNativeDemandResidencyError};
use crate::gating::ScoringFunc;
use crate::gpu_native_residency::GpuNativeTieredResidencyManager;
use crate::model::RealModel;
use crate::sampling::SamplingParams;

/// Structured summary of runtime counters across the GPU-native token loop.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuNativeTokenLoopSnapshot {
    pub token_attempts: u64,
    pub tokens_completed: u64,
    pub warm_tokens_completed: u64,
    pub residency_miss_attempts: u64,
    pub replay_attempts: u64,
    pub residency_services: u64,
    pub fatal_failures: u64,
    pub no_progress_failures: u64,
    pub queue_submissions: u64,
    pub boundary_maps: u64,
    pub boundary_readbacks: u64,
}

/// Separate PR1 recovery accounting so frozen token-loop schemas retain their
/// literal historical meaning.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GpuNativeRecoverySnapshot {
    pub resume_attempts: u64,
    pub recovery_segments: u64,
    pub checkpoint_captures: u64,
    pub checkpoint_restores: u64,
    pub full_token_replay_attempts: u64,
    pub layers_encoded: u64,
    pub attention_layers_reexecuted: u64,
    pub expert_layers_reexecuted: u64,
    pub invalid_tail_layers_encoded: u64,
    pub residency_service_us: u64,
    pub boundary_wait_us: u64,
}

#[derive(Default)]
struct GpuNativeTokenLoopCounters {
    token_attempts: AtomicU64,
    tokens_completed: AtomicU64,
    warm_tokens_completed: AtomicU64,
    residency_miss_attempts: AtomicU64,
    replay_attempts: AtomicU64,
    residency_services: AtomicU64,
    fatal_failures: AtomicU64,
    no_progress_failures: AtomicU64,
    queue_submissions: AtomicU64,
    boundary_maps: AtomicU64,
    boundary_readbacks: AtomicU64,
}

#[derive(Default)]
struct GpuNativeRecoveryCounters {
    resume_attempts: AtomicU64,
    recovery_segments: AtomicU64,
    checkpoint_captures: AtomicU64,
    checkpoint_restores: AtomicU64,
    full_token_replay_attempts: AtomicU64,
    layers_encoded: AtomicU64,
    attention_layers_reexecuted: AtomicU64,
    expert_layers_reexecuted: AtomicU64,
    invalid_tail_layers_encoded: AtomicU64,
    residency_service_us: AtomicU64,
    boundary_wait_us: AtomicU64,
}

impl GpuNativeRecoveryCounters {
    fn snapshot(&self) -> GpuNativeRecoverySnapshot {
        GpuNativeRecoverySnapshot {
            resume_attempts: self.resume_attempts.load(Ordering::Relaxed),
            recovery_segments: self.recovery_segments.load(Ordering::Relaxed),
            checkpoint_captures: self.checkpoint_captures.load(Ordering::Relaxed),
            checkpoint_restores: self.checkpoint_restores.load(Ordering::Relaxed),
            full_token_replay_attempts: self.full_token_replay_attempts.load(Ordering::Relaxed),
            layers_encoded: self.layers_encoded.load(Ordering::Relaxed),
            attention_layers_reexecuted: self.attention_layers_reexecuted.load(Ordering::Relaxed),
            expert_layers_reexecuted: self.expert_layers_reexecuted.load(Ordering::Relaxed),
            invalid_tail_layers_encoded: self.invalid_tail_layers_encoded.load(Ordering::Relaxed),
            residency_service_us: self.residency_service_us.load(Ordering::Relaxed),
            boundary_wait_us: self.boundary_wait_us.load(Ordering::Relaxed),
        }
    }
}

fn saturating_micros(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_micros()).unwrap_or(u64::MAX)
}

impl GpuNativeTokenLoopCounters {
    fn snapshot(&self) -> GpuNativeTokenLoopSnapshot {
        GpuNativeTokenLoopSnapshot {
            token_attempts: self.token_attempts.load(Ordering::Relaxed),
            tokens_completed: self.tokens_completed.load(Ordering::Relaxed),
            warm_tokens_completed: self.warm_tokens_completed.load(Ordering::Relaxed),
            residency_miss_attempts: self.residency_miss_attempts.load(Ordering::Relaxed),
            replay_attempts: self.replay_attempts.load(Ordering::Relaxed),
            residency_services: self.residency_services.load(Ordering::Relaxed),
            fatal_failures: self.fatal_failures.load(Ordering::Relaxed),
            no_progress_failures: self.no_progress_failures.load(Ordering::Relaxed),
            queue_submissions: self.queue_submissions.load(Ordering::Relaxed),
            boundary_maps: self.boundary_maps.load(Ordering::Relaxed),
            boundary_readbacks: self.boundary_readbacks.load(Ordering::Relaxed),
        }
    }
}

/// Errors originating in the GPU-native token loop.
#[derive(Debug, Clone, PartialEq)]
pub enum GpuNativeTokenLoopError {
    IncompatibleModel(GpuNativeModelCompatibilityError),
    ContextLimitExceeded {
        requested_position: usize,
        max_seq_len: usize,
    },
    PositionMismatch {
        requested_position: usize,
        committed_position: usize,
    },
    AttemptBoundExceeded {
        attempts: usize,
        max_attempts: usize,
    },
    NoProgress {
        layer_index: usize,
        selected_ids: Vec<u32>,
    },
    FatalNumericalFailure {
        layer_index: Option<usize>,
        status_bits: u32,
    },
    UnknownStatusBits {
        layer_index: Option<usize>,
        status_bits: u32,
        unknown_bits: u32,
    },
    ResidencyServiceFailed(GpuNativeDemandResidencyError),
    UnsupportedSampling {
        reason: String,
    },
    Bootstrap(GpuNativeBootstrapError),
    InvalidBoundaryReport {
        detail: String,
    },
    InvalidSelectedExpertId {
        layer_index: usize,
        expert_id: u32,
    },
    DuplicateSelectedExpertId {
        layer_index: usize,
        expert_id: u32,
    },
    InvalidTopKCount {
        expected: usize,
        actual: usize,
    },
    MapFailed(String),
}

impl fmt::Display for GpuNativeTokenLoopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleModel(err) => write!(f, "incompatible model: {err}"),
            Self::ContextLimitExceeded {
                requested_position,
                max_seq_len,
            } => write!(
                f,
                "requested position {requested_position} exceeds gpu_native_max_seq_len {max_seq_len}"
            ),
            Self::PositionMismatch {
                requested_position,
                committed_position,
            } => write!(
                f,
                "position mismatch: requested position {requested_position} does not match committed position {committed_position}"
            ),
            Self::AttemptBoundExceeded {
                attempts,
                max_attempts,
            } => write!(
                f,
                "token attempt bound exceeded: {attempts} attempts >= {max_attempts}"
            ),
            Self::NoProgress {
                layer_index,
                selected_ids,
            } => write!(
                f,
                "no progress after residency service on layer {layer_index} with experts {selected_ids:?}"
            ),
            Self::FatalNumericalFailure {
                layer_index,
                status_bits,
            } => write!(
                f,
                "fatal numerical failure on {:?} with status bits 0x{status_bits:08x}",
                layer_index
            ),
            Self::UnknownStatusBits {
                layer_index,
                status_bits,
                unknown_bits,
            } => write!(
                f,
                "unknown GPU-native status bits 0x{unknown_bits:08x} on {layer_index:?} in status 0x{status_bits:08x}"
            ),
            Self::ResidencyServiceFailed(err) => write!(f, "residency service failed: {err}"),
            Self::UnsupportedSampling { reason } => write!(f, "unsupported sampling: {reason}"),
            Self::Bootstrap(err) => write!(f, "GPU-native bootstrap error: {err}"),
            Self::InvalidBoundaryReport { detail } => {
                write!(f, "invalid boundary report: {detail}")
            }
            Self::InvalidSelectedExpertId {
                layer_index,
                expert_id,
            } => write!(
                f,
                "invalid selected expert id {expert_id} on layer {layer_index}"
            ),
            Self::DuplicateSelectedExpertId {
                layer_index,
                expert_id,
            } => write!(
                f,
                "duplicate selected expert id {expert_id} on layer {layer_index}"
            ),
            Self::InvalidTopKCount { expected, actual } => write!(
                f,
                "selected expert count mismatch: expected {expected}, got {actual}"
            ),
            Self::MapFailed(detail) => write!(f, "staging buffer map failed: {detail}"),
        }
    }
}

impl std::error::Error for GpuNativeTokenLoopError {}

impl From<GpuNativeModelCompatibilityError> for GpuNativeTokenLoopError {
    fn from(err: GpuNativeModelCompatibilityError) -> Self {
        Self::IncompatibleModel(err)
    }
}

impl From<GpuNativeBootstrapError> for GpuNativeTokenLoopError {
    fn from(err: GpuNativeBootstrapError) -> Self {
        Self::Bootstrap(err)
    }
}

/// Errors raised when a model checkpoint violates the supported GPU-native model contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuNativeModelCompatibilityError {
    UnsupportedArchitecture {
        architecture: String,
    },
    DenseLayerUnsupported {
        layer_index: usize,
    },
    SharedExpertUnsupported {
        layer_index: usize,
    },
    MlaUnsupported {
        layer_index: usize,
    },
    NonQ4ExpertDtype {
        dtype: String,
    },
    TooManyExperts {
        num_experts: usize,
        max: usize,
    },
    InvalidTopK {
        top_k: usize,
        max: usize,
    },
    IncompatibleGeometry {
        detail: String,
    },
    AsymmetricVHeadDim {
        head_dim: usize,
        v_head_dim: usize,
    },
    AttentionSinkUnsupported {
        layer_index: usize,
    },
    AttentionBiasesUnsupported {
        layer_index: usize,
    },
    ValueScaleUnsupported {
        layer_index: usize,
    },
    SlidingWindowUnsupported {
        layer_index: usize,
    },
    GroupedRoutingUnsupported {
        layer_index: usize,
    },
    NonSoftmaxRouter {
        layer_index: usize,
    },
    RouterCorrectionBiasUnsupported {
        layer_index: usize,
    },
    RoutedScalingFactorUnsupported {
        layer_index: usize,
        factor_bits: u32,
    },
    NonNormalisedTopK {
        layer_index: usize,
    },
    UnsupportedDenseDtype {
        tensor: String,
        dtype: String,
    },
    InconsistentRopeDimension {
        layer_index: usize,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for GpuNativeModelCompatibilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedArchitecture { architecture } => write!(
                f,
                "GPU-native execution supports only Qwen3Moe in this slice, got {architecture}"
            ),
            Self::DenseLayerUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} contains a dense FFN; all layers must be sparse MoE"
            ),
            Self::SharedExpertUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} contains a shared expert; shared experts are unsupported"
            ),
            Self::MlaUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} contains MLA; MLA is unsupported"
            ),
            Self::NonQ4ExpertDtype { dtype } => write!(
                f,
                "routed expert dtype must be Q4_0, got {dtype}"
            ),
            Self::TooManyExperts { num_experts, max } => write!(
                f,
                "num_experts ({num_experts}) exceeds GPU router maximum ({max})"
            ),
            Self::InvalidTopK { top_k, max } => write!(
                f,
                "top_k ({top_k}) must be in 1..={max}"
            ),
            Self::IncompatibleGeometry { detail } => write!(
                f,
                "incompatible model geometry: {detail}"
            ),
            Self::AsymmetricVHeadDim { head_dim, v_head_dim } => write!(
                f,
                "asymmetric V head dimension unsupported: head_dim={head_dim}, v_head_dim={v_head_dim}"
            ),
            Self::AttentionSinkUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} uses attention sink bias; sinks are unsupported"
            ),
            Self::AttentionBiasesUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} uses attention projection biases; biases are unsupported"
            ),
            Self::ValueScaleUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} uses attention value scaling; value scaling is unsupported"
            ),
            Self::SlidingWindowUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} uses sliding-window attention; sliding window is unsupported"
            ),
            Self::GroupedRoutingUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} uses grouped expert routing; grouped routing is unsupported"
            ),
            Self::NonSoftmaxRouter { layer_index } => write!(
                f,
                "layer {layer_index} uses non-softmax router scoring; only Softmax is supported"
            ),
            Self::RouterCorrectionBiasUnsupported { layer_index } => write!(
                f,
                "layer {layer_index} uses router correction bias; correction bias is unsupported"
            ),
            Self::RoutedScalingFactorUnsupported { layer_index, factor_bits } => write!(
                f,
                "layer {layer_index} uses routed scaling factor 0x{factor_bits:08x} != 1.0"
            ),
            Self::NonNormalisedTopK { layer_index } => write!(
                f,
                "layer {layer_index} does not normalise top-K weights; top-K normalisation is required"
            ),
            Self::UnsupportedDenseDtype { tensor, dtype } => write!(
                f,
                "tensor {tensor} has unsupported dense dtype {dtype}; only F32 and Q8_0 are supported"
            ),
            Self::InconsistentRopeDimension {
                layer_index,
                expected,
                actual,
            } => write!(
                f,
                "layer {layer_index} has rope_dim {actual}, expected uniform rope_dim {expected}"
            ),
        }
    }
}

impl std::error::Error for GpuNativeModelCompatibilityError {}

/// Fixed byte layout for one token-boundary compact report staging buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuNativeBoundaryReportLayout {
    pub num_layers: usize,
    pub top_k: usize,
    pub layer_status_offset: usize,
    pub layer_status_bytes: usize,
    pub selected_ids_offset: usize,
    pub selected_ids_bytes: usize,
    pub final_status_offset: usize,
    pub sampled_token_offset: usize,
    pub total_bytes: u64,
}

impl GpuNativeBoundaryReportLayout {
    pub fn try_new(num_layers: usize, top_k: usize) -> Result<Self, GpuNativeTokenLoopError> {
        if num_layers == 0 {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "num_layers must be > 0".into(),
            });
        }
        if top_k == 0 || top_k > MAX_GPU_NATIVE_ROUTER_TOP_K {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: format!("top_k must be in 1..={MAX_GPU_NATIVE_ROUTER_TOP_K}"),
            });
        }
        let u32_bytes = std::mem::size_of::<u32>();

        let layer_status_offset = 0usize;
        let layer_status_bytes = num_layers.checked_mul(u32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "layer status bytes overflow".into(),
            }
        })?;

        let selected_ids_offset = layer_status_bytes;
        let selected_ids_per_layer = top_k.checked_mul(u32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "selected ids per layer overflow".into(),
            }
        })?;
        let selected_ids_bytes =
            num_layers
                .checked_mul(selected_ids_per_layer)
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "total selected ids bytes overflow".into(),
                })?;

        let final_status_offset = selected_ids_offset
            .checked_add(selected_ids_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "final status offset overflow".into(),
            })?;

        let sampled_token_offset = final_status_offset.checked_add(u32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "sampled token offset overflow".into(),
            }
        })?;

        let total_size = sampled_token_offset.checked_add(u32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "total report bytes overflow".into(),
            }
        })?;

        let total_bytes = u64::try_from(total_size).map_err(|_| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "total report bytes exceed u64".into(),
            }
        })?;

        Ok(Self {
            num_layers,
            top_k,
            layer_status_offset,
            layer_status_bytes,
            selected_ids_offset,
            selected_ids_bytes,
            final_status_offset,
            sampled_token_offset,
            total_bytes,
        })
    }

    pub fn parse(&self, bytes: &[u8]) -> Result<GpuNativeBoundaryReport, GpuNativeTokenLoopError> {
        if (bytes.len() as u64) < self.total_bytes {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: format!(
                    "readback bytes length {} less than required {}",
                    bytes.len(),
                    self.total_bytes
                ),
            });
        }

        let mut layer_statuses = Vec::with_capacity(self.num_layers);
        for l in 0..self.num_layers {
            let start = self.layer_status_offset + l * 4;
            let status = u32::from_le_bytes(bytes[start..start + 4].try_into().map_err(|_| {
                GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "failed to decode layer status".into(),
                }
            })?);
            layer_statuses.push(status);
        }

        let mut selected_ids = Vec::with_capacity(self.num_layers);
        for l in 0..self.num_layers {
            let mut layer_ids = Vec::with_capacity(self.top_k);
            let layer_start = self.selected_ids_offset + l * self.top_k * 4;
            for k in 0..self.top_k {
                let start = layer_start + k * 4;
                let id = u32::from_le_bytes(bytes[start..start + 4].try_into().map_err(|_| {
                    GpuNativeTokenLoopError::InvalidBoundaryReport {
                        detail: "failed to decode selected id".into(),
                    }
                })?);
                layer_ids.push(id);
            }
            selected_ids.push(layer_ids);
        }

        let final_status = u32::from_le_bytes(
            bytes[self.final_status_offset..self.final_status_offset + 4]
                .try_into()
                .map_err(|_| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "failed to decode final status".into(),
                })?,
        );

        let sampled_token = u32::from_le_bytes(
            bytes[self.sampled_token_offset..self.sampled_token_offset + 4]
                .try_into()
                .map_err(|_| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "failed to decode sampled token".into(),
                })?,
        );

        Ok(GpuNativeBoundaryReport {
            layer_statuses,
            selected_ids,
            final_status,
            sampled_token,
        })
    }
}

/// Parsed results of one token attempt's boundary readback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuNativeBoundaryReport {
    pub layer_statuses: Vec<u32>,
    pub selected_ids: Vec<Vec<u32>>,
    pub final_status: u32,
    pub sampled_token: u32,
}

impl GpuNativeBoundaryReport {
    /// Returns the index of the first layer whose latched status transitioned to non-zero.
    pub fn first_failure_layer(&self) -> Option<usize> {
        self.layer_statuses.iter().position(|&status| status != 0)
    }

    /// Inspect only the layer interval encoded by the current boundary.
    pub fn first_failure_layer_in(
        &self,
        attempted_layers: Range<usize>,
    ) -> Result<Option<usize>, GpuNativeTokenLoopError> {
        if attempted_layers.start >= attempted_layers.end
            || attempted_layers.end > self.layer_statuses.len()
        {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: format!(
                    "attempted layer range {:?} is invalid for {} layer statuses",
                    attempted_layers,
                    self.layer_statuses.len()
                ),
            });
        }
        Ok(self.layer_statuses[attempted_layers.clone()]
            .iter()
            .position(|&status| status != 0)
            .map(|relative| attempted_layers.start + relative))
    }
}

const GPU_NATIVE_RECOVERY_INITIAL_WINDOW: usize = 1;
const GPU_NATIVE_RECOVERY_MAX_WINDOW: usize = 8;

fn gpu_native_attempt_bound(num_layers: usize) -> Result<usize, GpuNativeTokenLoopError> {
    num_layers
        .checked_mul(2)
        .and_then(|bound| bound.checked_add(2))
        .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
            detail: "GPU-native recovery attempt bound overflow".into(),
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GpuNativeAttemptStart {
    Fresh,
    ResumeExpert { layer_index: usize },
    Continue { layer_index: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GpuNativeMissSignature {
    layer_index: usize,
    selected_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GpuNativeExecutionSegment {
    attempt_start: GpuNativeAttemptStart,
    ordinary_layers: Range<usize>,
    attempted_layers: Range<usize>,
    completes_token: bool,
}

impl GpuNativeExecutionSegment {
    fn fresh(num_layers: usize) -> Result<Self, GpuNativeTokenLoopError> {
        if num_layers == 0 {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "cannot encode a token with zero layers".into(),
            });
        }
        Ok(Self {
            attempt_start: GpuNativeAttemptStart::Fresh,
            ordinary_layers: 0..num_layers,
            attempted_layers: 0..num_layers,
            completes_token: true,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GpuNativeRecoveryCursor {
    next_layer: usize,
    window_layers: usize,
    last_miss: Option<GpuNativeMissSignature>,
    residency_services: usize,
    clean_segments: usize,
    pending_resume_layer: Option<usize>,
}

impl GpuNativeRecoveryCursor {
    fn after_serviced_miss(
        num_layers: usize,
        miss: GpuNativeMissSignature,
    ) -> Result<Self, GpuNativeTokenLoopError> {
        if miss.layer_index >= num_layers {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: format!(
                    "recovery miss layer {} is outside 0..{num_layers}",
                    miss.layer_index
                ),
            });
        }
        Ok(Self {
            next_layer: miss.layer_index + 1,
            window_layers: GPU_NATIVE_RECOVERY_INITIAL_WINDOW,
            last_miss: Some(miss.clone()),
            residency_services: 1,
            clean_segments: 0,
            pending_resume_layer: Some(miss.layer_index),
        })
    }

    fn plan(
        &self,
        num_layers: usize,
    ) -> Result<GpuNativeExecutionSegment, GpuNativeTokenLoopError> {
        if num_layers == 0
            || self.window_layers == 0
            || self.window_layers > GPU_NATIVE_RECOVERY_MAX_WINDOW
            || self.next_layer > num_layers
        {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: format!("invalid GPU-native recovery cursor: {self:?}"),
            });
        }

        let (attempt_start, attempted_start, ordinary_start) =
            if let Some(layer_index) = self.pending_resume_layer {
                if layer_index >= num_layers || self.next_layer != layer_index + 1 {
                    return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                        detail: format!("invalid pending resume cursor: {self:?}"),
                    });
                }
                (
                    GpuNativeAttemptStart::ResumeExpert { layer_index },
                    layer_index,
                    self.next_layer,
                )
            } else {
                if self.next_layer >= num_layers {
                    return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                        detail: "completed recovery cursor cannot plan another segment".into(),
                    });
                }
                (
                    GpuNativeAttemptStart::Continue {
                        layer_index: self.next_layer,
                    },
                    self.next_layer,
                    self.next_layer,
                )
            };
        let ordinary_end = ordinary_start
            .saturating_add(self.window_layers)
            .min(num_layers);
        Ok(GpuNativeExecutionSegment {
            attempt_start,
            ordinary_layers: ordinary_start..ordinary_end,
            attempted_layers: attempted_start..ordinary_end,
            completes_token: ordinary_end == num_layers,
        })
    }

    fn record_clean_segment(
        &mut self,
        segment: &GpuNativeExecutionSegment,
        num_layers: usize,
    ) -> Result<bool, GpuNativeTokenLoopError> {
        if self.plan(num_layers)? != *segment {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "recovery segment does not match current cursor".into(),
            });
        }
        let previous_next = self.next_layer;
        self.next_layer = segment.ordinary_layers.end;
        self.pending_resume_layer = None;
        self.clean_segments = self.clean_segments.saturating_add(1);
        self.window_layers = self
            .window_layers
            .saturating_mul(2)
            .min(GPU_NATIVE_RECOVERY_MAX_WINDOW);
        if self.next_layer <= previous_next && !segment.completes_token {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "clean recovery segment made no forward progress".into(),
            });
        }
        Ok(segment.completes_token)
    }

    fn record_serviced_miss(
        &mut self,
        segment: &GpuNativeExecutionSegment,
        miss: GpuNativeMissSignature,
    ) -> Result<(), GpuNativeTokenLoopError> {
        if !segment.attempted_layers.contains(&miss.layer_index) {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: format!(
                    "miss layer {} is outside attempted interval {:?}",
                    miss.layer_index, segment.attempted_layers
                ),
            });
        }
        if self.last_miss.as_ref() == Some(&miss) {
            return Err(GpuNativeTokenLoopError::NoProgress {
                layer_index: miss.layer_index,
                selected_ids: miss.selected_ids,
            });
        }
        self.next_layer = miss.layer_index + 1;
        self.window_layers = GPU_NATIVE_RECOVERY_INITIAL_WINDOW;
        self.pending_resume_layer = Some(miss.layer_index);
        self.last_miss = Some(miss);
        self.residency_services = self.residency_services.saturating_add(1);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GpuNativeStatusDisposition {
    Clean,
    RetryableResidencyMiss,
}

fn classify_gpu_native_status(
    status_bits: u32,
    layer_index: Option<usize>,
) -> Result<GpuNativeStatusDisposition, GpuNativeTokenLoopError> {
    if (status_bits & GPU_NATIVE_STATUS_FATAL_MASK) != 0 {
        return Err(GpuNativeTokenLoopError::FatalNumericalFailure {
            layer_index,
            status_bits,
        });
    }
    let known_status_mask = GPU_NATIVE_STATUS_FATAL_MASK | GPU_NATIVE_STATUS_RETRYABLE_MASK;
    let unknown_bits = status_bits & !known_status_mask;
    if unknown_bits != 0 {
        return Err(GpuNativeTokenLoopError::UnknownStatusBits {
            layer_index,
            status_bits,
            unknown_bits,
        });
    }
    if status_bits == GPU_NATIVE_STATUS_RETRYABLE_MASK {
        Ok(GpuNativeStatusDisposition::RetryableResidencyMiss)
    } else {
        Ok(GpuNativeStatusDisposition::Clean)
    }
}

fn authoritative_actual_routes<'a>(
    report: &'a GpuNativeBoundaryReport,
    segment: &GpuNativeExecutionSegment,
) -> Result<Option<&'a [Vec<u32>]>, GpuNativeTokenLoopError> {
    if report
        .first_failure_layer_in(segment.attempted_layers.clone())?
        .is_some()
        || !segment.completes_token
    {
        return Ok(None);
    }

    match classify_gpu_native_status(report.final_status, None) {
        Ok(GpuNativeStatusDisposition::Clean) => Ok(Some(&report.selected_ids)),
        Ok(GpuNativeStatusDisposition::RetryableResidencyMiss) => {
            Err(GpuNativeTokenLoopError::FatalNumericalFailure {
                layer_index: None,
                status_bits: report.final_status,
            })
        }
        Err(error) => Err(error),
    }
}

/// Persistent per-layer plans and handles for one Qwen3-MoE transformer layer.
pub struct GpuNativeLayerPlan {
    pub layer_index: usize,
    pub rms_attn_handle: GpuNativeRmsNormHandle,
    pub rms_moe_handle: GpuNativeRmsNormHandle,
    pub attn_plan: GpuNativeAttentionPlan,
    pub router_plan: GpuNativeRouterPlan,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuNativeModelGeometry {
    pub num_layers: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
}

/// Diagnostic trace sink for capturing intermediate layer states during an attempt.
pub struct GpuNativeDiagnosticSink<'a> {
    pub layout: &'a crate::gpu_native_diagnostics::GpuNativeDiagnosticTraceLayout,
    pub staging_buffer: &'a wgpu::Buffer,
}

/// Target-layer-only sink used exclusively by the router-rank diagnostic.
pub struct GpuNativeRouterRankDiagnosticSink<'a> {
    pub layout: &'a crate::gpu_native_router_rank_diagnostics::RouterRankTraceLayout,
    pub staging_buffer: &'a wgpu::Buffer,
}

#[derive(Clone, Copy)]
enum GpuNativeSemanticDiagnosticLayout<'a> {
    Target(&'a crate::gpu_native_expert_permutation_semantic_parity::SemanticTraceLayout),
    Corpus(&'a crate::gpu_native_semantic_parity_corpus::SemanticCorpusTraceLayout),
    Q4ExpertStages(&'a crate::gpu_native_q4_expert_stage_attribution::Q4ExpertStageTraceLayout),
}

/// Copy-only observation sink used exclusively by semantic diagnostics.
pub struct GpuNativeExpertPermutationSemanticSink<'a> {
    layout: GpuNativeSemanticDiagnosticLayout<'a>,
    pub staging_buffer: &'a wgpu::Buffer,
    stage_scratch: Option<&'a GpuNativeScratch>,
}

impl GpuNativeExpertPermutationSemanticSink<'_> {
    fn target_layout_for_layer(
        &self,
        layer: usize,
    ) -> Option<&crate::gpu_native_expert_permutation_semantic_parity::SemanticTraceLayout> {
        match self.layout {
            GpuNativeSemanticDiagnosticLayout::Target(layout) if layout.target_layer == layer => {
                Some(layout)
            }
            GpuNativeSemanticDiagnosticLayout::Target(_)
            | GpuNativeSemanticDiagnosticLayout::Corpus(_)
            | GpuNativeSemanticDiagnosticLayout::Q4ExpertStages(_) => None,
        }
    }

    fn corpus_layout(
        &self,
    ) -> Option<&crate::gpu_native_semantic_parity_corpus::SemanticCorpusTraceLayout> {
        match self.layout {
            GpuNativeSemanticDiagnosticLayout::Corpus(layout) => Some(layout),
            GpuNativeSemanticDiagnosticLayout::Target(_)
            | GpuNativeSemanticDiagnosticLayout::Q4ExpertStages(_) => None,
        }
    }

    fn q4_expert_stage_layout_for_layer(
        &self,
        layer: usize,
    ) -> Option<&crate::gpu_native_q4_expert_stage_attribution::Q4ExpertStageTargetLayout> {
        match self.layout {
            GpuNativeSemanticDiagnosticLayout::Q4ExpertStages(layout) => {
                layout.target_for_layer(layer)
            }
            GpuNativeSemanticDiagnosticLayout::Target(_)
            | GpuNativeSemanticDiagnosticLayout::Corpus(_) => None,
        }
    }

    fn total_bytes(&self) -> u64 {
        match self.layout {
            GpuNativeSemanticDiagnosticLayout::Target(layout) => layout.total_bytes,
            GpuNativeSemanticDiagnosticLayout::Corpus(layout) => layout.total_bytes,
            GpuNativeSemanticDiagnosticLayout::Q4ExpertStages(layout) => layout.total_bytes,
        }
    }
}

struct GpuNativeAttemptOutput {
    boundary_report: GpuNativeBoundaryReport,
    diagnostic_trace: Option<crate::gpu_native_diagnostics::GpuNativeDiagnosticTrace>,
    router_rank_trace: Option<crate::gpu_native_router_rank_diagnostics::RouterRankGpuTrace>,
    semantic_trace: Option<crate::gpu_native_expert_permutation_semantic_parity::SemanticGpuTrace>,
    semantic_corpus_trace: Option<crate::gpu_native_semantic_parity_corpus::SemanticCorpusGpuTrace>,
    q4_expert_stage_trace:
        Option<crate::gpu_native_q4_expert_stage_attribution::Q4ExpertStageGpuTrace>,
}

struct GpuNativeStepOutput {
    sampled_token: Option<u32>,
    attempts: usize,
    diagnostic_trace: Option<crate::gpu_native_diagnostics::GpuNativeDiagnosticTrace>,
    router_rank_trace: Option<crate::gpu_native_router_rank_diagnostics::RouterRankGpuTrace>,
    semantic_trace: Option<crate::gpu_native_expert_permutation_semantic_parity::SemanticGpuTrace>,
    semantic_corpus_trace: Option<crate::gpu_native_semantic_parity_corpus::SemanticCorpusGpuTrace>,
    q4_expert_stage_trace:
        Option<crate::gpu_native_q4_expert_stage_attribution::Q4ExpertStageGpuTrace>,
}

/// Persistent, model-scoped owner of the GPU-native token loop.
pub struct GpuNativeTokenLoop {
    executor: Arc<GpuNativeExecutorContext>,
    residency_manager: Arc<GpuNativeTieredResidencyManager>,
    model_geometry: GpuNativeModelGeometry,
    embedding_handle: GpuNativeDenseWeightHandle,
    final_norm_handle: GpuNativeRmsNormHandle,
    lm_head_handle: GpuNativeDenseWeightHandle,
    layers: Vec<GpuNativeLayerPlan>,
    report_layout: GpuNativeBoundaryReportLayout,
    counters: GpuNativeTokenLoopCounters,
    recovery_counters: GpuNativeRecoveryCounters,
    execution_guard: TokioMutex<()>,
}

impl GpuNativeTokenLoop {
    /// Validate that `model` conforms strictly to the supported Qwen3Moe GPU-native contract.
    pub fn validate_model_compatibility(
        model: &RealModel,
    ) -> Result<(), GpuNativeModelCompatibilityError> {
        if model.config.architecture != Architecture::Qwen3Moe {
            return Err(GpuNativeModelCompatibilityError::UnsupportedArchitecture {
                architecture: format!("{:?}", model.config.architecture),
            });
        }
        if model.config.top_k == 0 || model.config.top_k > MAX_GPU_NATIVE_ROUTER_TOP_K {
            return Err(GpuNativeModelCompatibilityError::InvalidTopK {
                top_k: model.config.top_k,
                max: MAX_GPU_NATIVE_ROUTER_TOP_K,
            });
        }
        if model.config.num_experts > MAX_GPU_NATIVE_ROUTER_EXPERTS {
            return Err(GpuNativeModelCompatibilityError::TooManyExperts {
                num_experts: model.config.num_experts,
                max: MAX_GPU_NATIVE_ROUTER_EXPERTS,
            });
        }
        if model.config.window_size.is_some() {
            return Err(GpuNativeModelCompatibilityError::SlidingWindowUnsupported {
                layer_index: 0,
            });
        }

        let is_valid_dense_dtype =
            |dtype: DenseDType| -> bool { matches!(dtype, DenseDType::F32 | DenseDType::Q8_0) };

        if !is_valid_dense_dtype(model.embedding.dtype()) {
            return Err(GpuNativeModelCompatibilityError::UnsupportedDenseDtype {
                tensor: "embed.weight".into(),
                dtype: model.embedding.dtype_name().into(),
            });
        }
        if !is_valid_dense_dtype(model.lm_head.weights.dtype()) {
            return Err(GpuNativeModelCompatibilityError::UnsupportedDenseDtype {
                tensor: "lm_head.weight".into(),
                dtype: model.lm_head.weights.dtype_name().into(),
            });
        }
        let expected_rope_dim = model.layers.first().map(|l| l.attn.rope_dim).unwrap_or(0);

        for (layer_idx, layer) in model.layers.iter().enumerate() {
            if layer.attn.rope_dim != expected_rope_dim {
                return Err(
                    GpuNativeModelCompatibilityError::InconsistentRopeDimension {
                        layer_index: layer_idx,
                        expected: expected_rope_dim,
                        actual: layer.attn.rope_dim,
                    },
                );
            }
            if layer.dense_ffn.is_some() {
                return Err(GpuNativeModelCompatibilityError::DenseLayerUnsupported {
                    layer_index: layer_idx,
                });
            }
            if layer.shared_expert.is_some() {
                return Err(GpuNativeModelCompatibilityError::SharedExpertUnsupported {
                    layer_index: layer_idx,
                });
            }
            if layer.mla.is_some() {
                return Err(GpuNativeModelCompatibilityError::MlaUnsupported {
                    layer_index: layer_idx,
                });
            }
            if layer.attn.v_head_dim != layer.attn.head_dim {
                return Err(GpuNativeModelCompatibilityError::AsymmetricVHeadDim {
                    head_dim: layer.attn.head_dim,
                    v_head_dim: layer.attn.v_head_dim,
                });
            }
            if layer.attn.sink_bias.is_some() {
                return Err(GpuNativeModelCompatibilityError::AttentionSinkUnsupported {
                    layer_index: layer_idx,
                });
            }
            if layer.attn.bq.is_some()
                || layer.attn.bk.is_some()
                || layer.attn.bv.is_some()
                || layer.attn.bo.is_some()
            {
                return Err(
                    GpuNativeModelCompatibilityError::AttentionBiasesUnsupported {
                        layer_index: layer_idx,
                    },
                );
            }
            if layer.attn.attention_value_scale.is_some() {
                return Err(GpuNativeModelCompatibilityError::ValueScaleUnsupported {
                    layer_index: layer_idx,
                });
            }
            if layer.attn.window_size.is_some() {
                return Err(GpuNativeModelCompatibilityError::SlidingWindowUnsupported {
                    layer_index: layer_idx,
                });
            }
            if layer.gate.scoring_func != ScoringFunc::Softmax {
                return Err(GpuNativeModelCompatibilityError::NonSoftmaxRouter {
                    layer_index: layer_idx,
                });
            }
            if layer.gate.correction_bias.is_some() {
                return Err(
                    GpuNativeModelCompatibilityError::RouterCorrectionBiasUnsupported {
                        layer_index: layer_idx,
                    },
                );
            }
            if layer.gate.n_group > 1 || layer.gate.topk_group > 1 {
                return Err(
                    GpuNativeModelCompatibilityError::GroupedRoutingUnsupported {
                        layer_index: layer_idx,
                    },
                );
            }
            if (layer.gate.routed_scaling_factor - 1.0).abs() > 1e-5 {
                return Err(
                    GpuNativeModelCompatibilityError::RoutedScalingFactorUnsupported {
                        layer_index: layer_idx,
                        factor_bits: layer.gate.routed_scaling_factor.to_bits(),
                    },
                );
            }
            if !layer.gate.normalise_topk {
                return Err(GpuNativeModelCompatibilityError::NonNormalisedTopK {
                    layer_index: layer_idx,
                });
            }

            for (name, weight) in [
                ("wq", &layer.attn.wq),
                ("wk", &layer.attn.wk),
                ("wv", &layer.attn.wv),
                ("wo", &layer.attn.wo),
                ("gate", &layer.gate.weights),
            ] {
                if !is_valid_dense_dtype(weight.dtype()) {
                    return Err(GpuNativeModelCompatibilityError::UnsupportedDenseDtype {
                        tensor: format!("layer_{layer_idx}_{name}"),
                        dtype: weight.dtype_name().into(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Construct and initialize the persistent GPU-native token loop.
    pub fn try_new(
        executor: Arc<GpuNativeExecutorContext>,
        residency_manager: Arc<GpuNativeTieredResidencyManager>,
        model: &RealModel,
        max_seq_len: usize,
    ) -> Result<Arc<Self>, GpuNativeTokenLoopError> {
        Self::validate_model_compatibility(model)?;

        let num_layers = model.layers.len();
        let top_k = model.config.top_k;
        let d_model = model.config.d_model;
        let d_ff = model.config.d_ff;
        let num_experts = model.config.num_experts;
        let num_heads = model.config.num_heads;
        let num_kv_heads = model.config.num_kv_heads;
        let head_dim = model.config.head_dim;
        let rope_dim = model
            .layers
            .first()
            .map(|l| l.attn.rope_dim)
            .unwrap_or(head_dim);
        let vocab_size = model.config.vocab_size;
        let rms_eps = model.config.rms_eps;
        let rope_base = model.config.rope_base;

        if max_seq_len == 0 {
            return Err(GpuNativeTokenLoopError::ContextLimitExceeded {
                requested_position: 0,
                max_seq_len,
            });
        }

        let model_geometry = GpuNativeModelGeometry {
            num_layers,
            d_model,
            d_ff,
            num_experts,
            top_k,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim,
            vocab_size,
            max_seq_len,
            rms_eps,
            rope_base,
        };

        let report_layout = GpuNativeBoundaryReportLayout::try_new(num_layers, top_k)?;

        let embedding_handle = executor.register_dense_weight(
            GpuNativeDenseWeightKey::try_new("model.embed")?,
            &model.embedding,
        )?;

        let mut layers = Vec::with_capacity(num_layers);
        let router_geometry = GpuNativeRouterGeometry::try_new(d_model, num_experts, top_k)?;
        let attention_geometry = GpuNativeAttentionGeometry::try_new(
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim,
        )?;

        for (l, layer) in model.layers.iter().enumerate() {
            let rms_attn_handle = executor.register_rms_norm(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.rms_attn"))?,
                layer.rms_attn.weight.as_slice(),
            )?;
            let rms_moe_handle = executor.register_rms_norm(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.rms_moe"))?,
                layer.rms_moe.weight.as_slice(),
            )?;

            let q_handle = executor.register_dense_weight(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.attn.q"))?,
                &layer.attn.wq,
            )?;
            let k_handle = executor.register_dense_weight(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.attn.k"))?,
                &layer.attn.wk,
            )?;
            let v_handle = executor.register_dense_weight(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.attn.v"))?,
                &layer.attn.wv,
            )?;
            let o_handle = executor.register_dense_weight(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.attn.o"))?,
                &layer.attn.wo,
            )?;

            let q_norm_handle = if let Some(ref q_norm) = layer.attn.q_norm {
                let handle = executor.register_rms_norm(
                    GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.attn.q_norm"))?,
                    q_norm.weight.as_slice(),
                )?;
                Some(GpuNativeAttentionNorm::try_new(handle, q_norm.eps)?)
            } else {
                None
            };

            let k_norm_handle = if let Some(ref k_norm) = layer.attn.k_norm {
                let handle = executor.register_rms_norm(
                    GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.attn.k_norm"))?,
                    k_norm.weight.as_slice(),
                )?;
                Some(GpuNativeAttentionNorm::try_new(handle, k_norm.eps)?)
            } else {
                None
            };

            let rope_handle = executor.register_standard_rope(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.attn.rope"))?,
                layer.attn.rope_dim,
                layer.attn.rope_base,
            )?;

            let attn_plan = executor.create_attention_plan(
                l,
                attention_geometry,
                q_handle,
                k_handle,
                v_handle,
                o_handle,
                q_norm_handle,
                k_norm_handle,
                rope_handle,
            )?;

            let gate_handle = executor.register_dense_weight(
                GpuNativeDenseWeightKey::try_new(format!("model.layers.{l}.router"))?,
                &layer.gate.weights,
            )?;
            let router_plan = executor.create_router_plan(l, router_geometry, gate_handle)?;

            layers.push(GpuNativeLayerPlan {
                layer_index: l,
                rms_attn_handle,
                rms_moe_handle,
                attn_plan,
                router_plan,
            });
        }

        let final_norm_handle = executor.register_rms_norm(
            GpuNativeDenseWeightKey::try_new("model.final_norm")?,
            model.final_rms.weight.as_slice(),
        )?;
        let lm_head_handle = executor.register_dense_weight(
            GpuNativeDenseWeightKey::try_new("model.lm_head")?,
            &model.lm_head.weights,
        )?;

        Ok(Arc::new(Self {
            executor,
            residency_manager,
            model_geometry,
            embedding_handle,
            final_norm_handle,
            lm_head_handle,
            layers,
            report_layout,
            counters: GpuNativeTokenLoopCounters::default(),
            recovery_counters: GpuNativeRecoveryCounters::default(),
            execution_guard: TokioMutex::new(()),
        }))
    }

    pub fn model_geometry(&self) -> GpuNativeModelGeometry {
        self.model_geometry
    }

    pub fn rope_dim(&self) -> usize {
        self.model_geometry.rope_dim
    }

    pub fn max_seq_len(&self) -> usize {
        self.model_geometry.max_seq_len
    }

    pub fn snapshot(&self) -> GpuNativeTokenLoopSnapshot {
        self.counters.snapshot()
    }

    pub fn recovery_snapshot(&self) -> GpuNativeRecoverySnapshot {
        self.recovery_counters.snapshot()
    }

    /// Allocate request-local device resources for one GPU-native sequence.
    pub fn create_request_state(&self) -> Result<GpuNativeRequestState, GpuNativeTokenLoopError> {
        self.create_request_state_inner(false, false)
    }

    /// Allocate request-local resources with copyable raw router logits for
    /// the explicit router-rank diagnostic only.
    pub fn create_router_rank_diagnostic_request_state(
        &self,
    ) -> Result<GpuNativeRequestState, GpuNativeTokenLoopError> {
        self.create_request_state_inner(true, false)
    }

    /// Allocate request-local copy-capable router/expert scratch only for the
    /// explicit expert-permutation semantic witness.
    pub fn create_expert_permutation_semantic_diagnostic_request_state(
        &self,
    ) -> Result<GpuNativeRequestState, GpuNativeTokenLoopError> {
        self.create_request_state_inner(true, true)
    }

    /// Allocate request-local copy-capable router/expert scratch for the
    /// diagnostic-only full-corpus semantic survey.
    pub fn create_semantic_parity_corpus_diagnostic_request_state(
        &self,
    ) -> Result<GpuNativeRequestState, GpuNativeTokenLoopError> {
        self.create_request_state_inner(true, true)
    }

    /// Allocate request-local copy-capable router/expert scratch for the
    /// diagnostic-only Q4 expert internal-stage observer.
    pub fn create_q4_expert_stage_diagnostic_request_state(
        &self,
    ) -> Result<GpuNativeRequestState, GpuNativeTokenLoopError> {
        self.create_request_state_inner(true, true)
    }

    fn create_request_state_inner(
        &self,
        router_rank_diagnostic: bool,
        semantic_diagnostic: bool,
    ) -> Result<GpuNativeRequestState, GpuNativeTokenLoopError> {
        let token_state = self.executor.create_token_state()?;
        let kv_width = self.model_geometry.num_kv_heads * self.model_geometry.head_dim;
        let kv_state = self.executor.create_kv_state(
            self.model_geometry.num_layers,
            self.model_geometry.max_seq_len,
            kv_width,
        )?;

        let attn_geom = GpuNativeAttentionGeometry::try_new(
            self.model_geometry.d_model,
            self.model_geometry.num_heads,
            self.model_geometry.num_kv_heads,
            self.model_geometry.head_dim,
            self.model_geometry.rope_dim,
        )?;
        let attn_scratch = self.executor.create_attention_scratch(attn_geom)?;

        let router_geom = GpuNativeRouterGeometry::try_new(
            self.model_geometry.d_model,
            self.model_geometry.num_experts,
            self.model_geometry.top_k,
        )?;
        let router_scratch = if router_rank_diagnostic {
            self.executor
                .create_router_diagnostic_scratch(router_geom)?
        } else {
            self.executor.create_router_scratch(router_geom)?
        };

        let expert_geom = GpuNativeQ4ExpertGeometry::try_new(
            self.model_geometry.d_model,
            self.model_geometry.d_ff,
            self.model_geometry.num_experts,
            self.model_geometry.top_k,
        )?;
        let expert_scratch = if semantic_diagnostic {
            self.executor
                .create_q4_expert_semantic_diagnostic_scratch(expert_geom)?
        } else {
            self.executor.create_q4_expert_scratch(expert_geom)?
        };

        let logits_scratch = self
            .executor
            .create_scratch(self.model_geometry.vocab_size)?;
        let sampled_token_buf = self.executor.create_boundary_result_scratch(1)?;
        let checkpoint_layout = GpuNativePreExpertCheckpointLayout::try_new(
            self.model_geometry.num_layers,
            self.model_geometry.d_model,
            self.model_geometry.top_k,
        )?;
        let pre_expert_checkpoints = self
            .executor
            .create_pre_expert_checkpoints(checkpoint_layout)?;

        let gpu = self.executor.authoritative_gpu()?;
        let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_boundary_staging"),
            size: self.report_layout.total_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(GpuNativeRequestState {
            token_state,
            kv_state,
            attn_scratch,
            router_scratch,
            expert_scratch,
            logits_scratch,
            sampled_token_buf,
            pre_expert_checkpoints,
            staging_buffer,
            committed_position: 0,
            max_seq_len: self.model_geometry.max_seq_len,
        })
    }

    /// Calculate the required context capacity for a prompt of length `prompt_len`
    /// and `max_tokens` completion tokens, starting from `committed_position`.
    ///
    /// For `max_tokens == 0`, zero completion tokens are generated and zero additional positions are consumed.
    /// For `max_tokens > 0`, the full autoregressive forward count (and final committed position) is:
    /// `committed_position + prompt_len + max_tokens - 1`
    /// because the forward evaluation of the final prompt token produces completion token 0.
    pub fn calculate_required_context_len(
        committed_position: usize,
        prompt_len: usize,
        max_tokens: usize,
    ) -> Option<usize> {
        if max_tokens == 0 {
            return Some(committed_position);
        }
        committed_position
            .checked_add(prompt_len)
            .and_then(|sum| sum.checked_add(max_tokens))
            .and_then(|sum| sum.checked_sub(1))
    }

    /// Ingest a multi-token prompt, commit KV positions, and generate up to `max_tokens` completion tokens.
    pub async fn ingest_prompt_and_generate(
        self: &Arc<Self>,
        engine: &Arc<Engine>,
        request: &mut GpuNativeRequestState,
        prompt_ids: &[u32],
        max_tokens: usize,
        params: &SamplingParams,
    ) -> Result<Vec<u32>, GpuNativeTokenLoopError> {
        if prompt_ids.is_empty() {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "prompt_ids must be non-empty".into(),
            });
        }
        if !params.is_greedy() {
            return Err(GpuNativeTokenLoopError::UnsupportedSampling {
                reason: "only greedy sampling (temperature=0.0) is supported in gpu_native mode in this slice".into(),
            });
        }
        if max_tokens == 0 {
            return Ok(Vec::new());
        }

        let total_required_len = Self::calculate_required_context_len(
            request.committed_position,
            prompt_ids.len(),
            max_tokens,
        )
        .ok_or_else(|| GpuNativeTokenLoopError::ContextLimitExceeded {
            requested_position: usize::MAX,
            max_seq_len: request.max_seq_len,
        })?;

        if total_required_len > request.max_seq_len {
            return Err(GpuNativeTokenLoopError::ContextLimitExceeded {
                requested_position: total_required_len,
                max_seq_len: request.max_seq_len,
            });
        }

        let _guard = self.execution_guard.lock().await;

        // Ingest prefix prompt tokens without evaluating LM-head
        let prefix_count = prompt_ids.len().saturating_sub(1);
        for &token_id in &prompt_ids[..prefix_count] {
            let pos = request.committed_position;
            self.step_token_unified_inner(engine, request, token_id, pos, false, None, None, None)
                .await?;
        }

        // Final prompt token: evaluate LM-head and sample first completion token
        let final_prompt = *prompt_ids.last().expect("checked non-empty");
        let final_prompt_pos = request.committed_position;
        let first_completion = self
            .step_token_unified_inner(
                engine,
                request,
                final_prompt,
                final_prompt_pos,
                true,
                None,
                None,
                None,
            )
            .await?
            .sampled_token
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "final prompt step produced no sampled token".into(),
            })?;

        let mut completion_ids = Vec::with_capacity(max_tokens);
        completion_ids.push(first_completion);

        let mut last_token = first_completion;
        while completion_ids.len() < max_tokens {
            let pos = request.committed_position;
            if pos >= request.max_seq_len {
                break;
            }
            let next_token = self
                .step_token_unified_inner(engine, request, last_token, pos, true, None, None, None)
                .await?
                .sampled_token
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "decode step produced no sampled token".into(),
                })?;
            completion_ids.push(next_token);
            last_token = next_token;
        }

        Ok(completion_ids)
    }

    /// Execute one single token step with bounded retries and residency demand service.
    pub async fn step_token(
        &self,
        engine: &Arc<Engine>,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        sample: bool,
    ) -> Result<Option<u32>, GpuNativeTokenLoopError> {
        let _guard = self.execution_guard.lock().await;
        let out = self
            .step_token_unified_inner(
                engine, request, token_id, position, sample, None, None, None,
            )
            .await?;
        Ok(out.sampled_token)
    }

    /// Diagnostic variant of step_token that captures intermediate activation traces on the final attempt.
    pub async fn step_token_diagnostic(
        &self,
        engine: &Arc<Engine>,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        sample: bool,
        trace_layout: &crate::gpu_native_diagnostics::GpuNativeDiagnosticTraceLayout,
        diagnostic_staging_buffer: &wgpu::Buffer,
    ) -> Result<
        (
            crate::gpu_native_diagnostics::GpuNativeDiagnosticTrace,
            usize,
        ),
        GpuNativeTokenLoopError,
    > {
        let _guard = self.execution_guard.lock().await;
        let out = self
            .step_token_unified_inner(
                engine,
                request,
                token_id,
                position,
                sample,
                Some((trace_layout, diagnostic_staging_buffer)),
                None,
                None,
            )
            .await?;
        let trace =
            out.diagnostic_trace
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "diagnostic trace was not collected".into(),
                })?;
        Ok((trace, out.attempts))
    }

    /// Diagnostic-only token step that captures the exact production router
    /// input, dense-GEMV logits, and selected outputs at one target layer.
    pub async fn step_token_router_rank_diagnostic(
        &self,
        engine: &Arc<Engine>,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        trace_layout: &crate::gpu_native_router_rank_diagnostics::RouterRankTraceLayout,
        diagnostic_staging_buffer: &wgpu::Buffer,
    ) -> Result<
        (
            crate::gpu_native_router_rank_diagnostics::RouterRankGpuTrace,
            u32,
            usize,
        ),
        GpuNativeTokenLoopError,
    > {
        let _guard = self.execution_guard.lock().await;
        let out = self
            .step_token_unified_inner(
                engine,
                request,
                token_id,
                position,
                true,
                None,
                Some((trace_layout, diagnostic_staging_buffer)),
                None,
            )
            .await?;
        let trace = out.router_rank_trace.ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "router-rank diagnostic trace was not collected".into(),
            }
        })?;
        let sampled_token =
            out.sampled_token
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "router-rank diagnostic step produced no sampled token".into(),
                })?;
        Ok((trace, sampled_token, out.attempts))
    }

    /// Diagnostic-only token step that captures the exact target-layer router
    /// evidence, every per-route GPU expert output, and the production GPU
    /// routed-MoE combined vector without changing the encoded math.
    pub async fn step_token_expert_permutation_semantic_diagnostic(
        &self,
        engine: &Arc<Engine>,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        trace_layout: &crate::gpu_native_expert_permutation_semantic_parity::SemanticTraceLayout,
        diagnostic_staging_buffer: &wgpu::Buffer,
    ) -> Result<
        (
            crate::gpu_native_expert_permutation_semantic_parity::SemanticGpuTrace,
            u32,
            usize,
        ),
        GpuNativeTokenLoopError,
    > {
        let _guard = self.execution_guard.lock().await;
        let out = self
            .step_token_unified_inner(
                engine,
                request,
                token_id,
                position,
                true,
                None,
                None,
                Some((
                    GpuNativeSemanticDiagnosticLayout::Target(trace_layout),
                    diagnostic_staging_buffer,
                    None,
                )),
            )
            .await?;
        let trace =
            out.semantic_trace
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "expert-permutation semantic trace was not collected".into(),
                })?;
        let sampled_token =
            out.sampled_token
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "expert-permutation semantic step produced no sampled token".into(),
                })?;
        Ok((trace, sampled_token, out.attempts))
    }

    /// Diagnostic-only token step that copies the exact production router
    /// evidence, per-route expert outputs, and routed-MoE result for every
    /// layer in one frozen traversal step.
    pub async fn step_token_semantic_parity_corpus_diagnostic(
        &self,
        engine: &Arc<Engine>,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        trace_layout: &crate::gpu_native_semantic_parity_corpus::SemanticCorpusTraceLayout,
        diagnostic_staging_buffer: &wgpu::Buffer,
    ) -> Result<
        (
            crate::gpu_native_semantic_parity_corpus::SemanticCorpusGpuTrace,
            u32,
            usize,
        ),
        GpuNativeTokenLoopError,
    > {
        let _guard = self.execution_guard.lock().await;
        let out = self
            .step_token_unified_inner(
                engine,
                request,
                token_id,
                position,
                true,
                None,
                None,
                Some((
                    GpuNativeSemanticDiagnosticLayout::Corpus(trace_layout),
                    diagnostic_staging_buffer,
                    None,
                )),
            )
            .await?;
        let trace = out.semantic_corpus_trace.ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "semantic corpus trace was not collected".into(),
            }
        })?;
        let sampled_token = out.sampled_token.ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "semantic corpus diagnostic step produced no sampled token".into(),
            }
        })?;
        Ok((trace, sampled_token, out.attempts))
    }

    /// Diagnostic-only final-prompt step that captures both the established
    /// full activation trace and the established semantic-corpus router/expert
    /// trace from one unchanged production traversal. Ordinary token steps
    /// never allocate either sink.
    pub async fn step_token_full_and_semantic_corpus_diagnostic(
        &self,
        engine: &Arc<Engine>,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        full_trace_layout: &crate::gpu_native_diagnostics::GpuNativeDiagnosticTraceLayout,
        full_trace_staging_buffer: &wgpu::Buffer,
        semantic_trace_layout: &crate::gpu_native_semantic_parity_corpus::SemanticCorpusTraceLayout,
        semantic_trace_staging_buffer: &wgpu::Buffer,
    ) -> Result<
        (
            crate::gpu_native_diagnostics::GpuNativeDiagnosticTrace,
            crate::gpu_native_semantic_parity_corpus::SemanticCorpusGpuTrace,
            u32,
            usize,
        ),
        GpuNativeTokenLoopError,
    > {
        let _guard = self.execution_guard.lock().await;
        let out = self
            .step_token_unified_inner(
                engine,
                request,
                token_id,
                position,
                true,
                Some((full_trace_layout, full_trace_staging_buffer)),
                None,
                Some((
                    GpuNativeSemanticDiagnosticLayout::Corpus(semantic_trace_layout),
                    semantic_trace_staging_buffer,
                    None,
                )),
            )
            .await?;
        let full_trace =
            out.diagnostic_trace
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "combined diagnostic full trace was not collected".into(),
                })?;
        let semantic_trace = out.semantic_corpus_trace.ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "combined diagnostic semantic trace was not collected".into(),
            }
        })?;
        let sampled_token =
            out.sampled_token
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "combined diagnostic final-prompt step produced no sampled token"
                        .into(),
                })?;
        Ok((full_trace, semantic_trace, sampled_token, out.attempts))
    }

    /// Diagnostic-only token step that observes raw gate, raw up,
    /// post-SwiGLU gated activation, diagnostic down, and the unchanged
    /// production down output at one or more frozen layers.
    pub async fn step_token_q4_expert_stage_diagnostic(
        &self,
        engine: &Arc<Engine>,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        trace_layout: &crate::gpu_native_q4_expert_stage_attribution::Q4ExpertStageTraceLayout,
        diagnostic_staging_buffer: &wgpu::Buffer,
    ) -> Result<
        (
            crate::gpu_native_q4_expert_stage_attribution::Q4ExpertStageGpuTrace,
            u32,
            usize,
        ),
        GpuNativeTokenLoopError,
    > {
        let expert_geometry = GpuNativeQ4ExpertGeometry::try_new(
            self.model_geometry.d_model,
            self.model_geometry.d_ff,
            self.model_geometry.num_experts,
            self.model_geometry.top_k,
        )?;
        if trace_layout.d_model != self.model_geometry.d_model
            || trace_layout.d_ff != self.model_geometry.d_ff
            || trace_layout.top_k != self.model_geometry.top_k
        {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "Q4 expert stage layout does not match token-loop geometry".into(),
            });
        }
        let stage_scratch = self
            .executor
            .create_q4_expert_stage_diagnostic_scratch(expert_geometry)?;
        let _guard = self.execution_guard.lock().await;
        let out = self
            .step_token_unified_inner(
                engine,
                request,
                token_id,
                position,
                true,
                None,
                None,
                Some((
                    GpuNativeSemanticDiagnosticLayout::Q4ExpertStages(trace_layout),
                    diagnostic_staging_buffer,
                    Some(&stage_scratch),
                )),
            )
            .await?;
        let trace = out.q4_expert_stage_trace.ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "Q4 expert stage trace was not collected".into(),
            }
        })?;
        let sampled_token =
            out.sampled_token
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "Q4 expert stage diagnostic produced no sampled token".into(),
                })?;
        Ok((trace, sampled_token, out.attempts))
    }

    /// Allocate a device-resident staging buffer for diagnostic trace readbacks.
    pub fn create_diagnostic_staging_buffer(
        &self,
        trace_layout: &crate::gpu_native_diagnostics::GpuNativeDiagnosticTraceLayout,
    ) -> Result<wgpu::Buffer, GpuNativeTokenLoopError> {
        let gpu = self.executor.authoritative_gpu()?;
        let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_diagnostic_staging"),
            size: trace_layout.total_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Ok(staging_buffer)
    }

    /// Allocate the target-layer-only router-rank readback buffer.
    pub fn create_router_rank_diagnostic_staging_buffer(
        &self,
        trace_layout: &crate::gpu_native_router_rank_diagnostics::RouterRankTraceLayout,
    ) -> Result<wgpu::Buffer, GpuNativeTokenLoopError> {
        let gpu = self.executor.authoritative_gpu()?;
        Ok(gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_router_rank_diagnostic_staging"),
            size: trace_layout.total_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }))
    }

    /// Allocate the target-layer-only expert-permutation semantic readback buffer.
    pub fn create_expert_permutation_semantic_diagnostic_staging_buffer(
        &self,
        trace_layout: &crate::gpu_native_expert_permutation_semantic_parity::SemanticTraceLayout,
    ) -> Result<wgpu::Buffer, GpuNativeTokenLoopError> {
        let gpu = self.executor.authoritative_gpu()?;
        Ok(gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_expert_permutation_semantic_diagnostic_staging"),
            size: trace_layout.total_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }))
    }

    /// Allocate the full-layer semantic-corpus readback buffer.
    pub fn create_semantic_parity_corpus_diagnostic_staging_buffer(
        &self,
        trace_layout: &crate::gpu_native_semantic_parity_corpus::SemanticCorpusTraceLayout,
    ) -> Result<wgpu::Buffer, GpuNativeTokenLoopError> {
        let gpu = self.executor.authoritative_gpu()?;
        Ok(gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_semantic_parity_corpus_diagnostic_staging"),
            size: trace_layout.total_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }))
    }

    /// Allocate the frozen-layer Q4 expert stage readback buffer.
    pub fn create_q4_expert_stage_diagnostic_staging_buffer(
        &self,
        trace_layout: &crate::gpu_native_q4_expert_stage_attribution::Q4ExpertStageTraceLayout,
    ) -> Result<wgpu::Buffer, GpuNativeTokenLoopError> {
        let gpu = self.executor.authoritative_gpu()?;
        Ok(gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_q4_expert_stage_diagnostic_staging"),
            size: trace_layout.total_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }))
    }
    /// Diagnostic variant for layer-0 attention only. Bypasses router, MoE, and later layers.
    /// Executes one prompt position and captures all layer-0 attention intermediates.
    pub async fn step_layer0_attention_diagnostic(
        &self,
        _engine: &Arc<Engine>,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        trace_layout: &crate::gpu_native_layer0_diagnostics::Layer0AttentionDiagnosticTraceLayout,
        diagnostic_staging_buffer: &wgpu::Buffer,
    ) -> Result<
        crate::gpu_native_layer0_diagnostics::Layer0AttentionDiagnosticTrace,
        GpuNativeTokenLoopError,
    > {
        let _guard = self.execution_guard.lock().await;

        if position != request.committed_position {
            return Err(GpuNativeTokenLoopError::PositionMismatch {
                requested_position: position,
                committed_position: request.committed_position,
            });
        }
        if position >= request.max_seq_len {
            return Err(GpuNativeTokenLoopError::ContextLimitExceeded {
                requested_position: position,
                max_seq_len: request.max_seq_len,
            });
        }

        if self.layers.is_empty() {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "no layers available in token loop".into(),
            });
        }

        let gpu = self.executor.authoritative_gpu()?;
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_native_layer0_attention_diagnostic"),
            });

        let layer_0 = &self.layers[0];
        let sink = crate::backend::gpu_native::Layer0AttentionGpuDiagnosticSink {
            layout: trace_layout,
            staging_buffer: diagnostic_staging_buffer,
        };

        // 1. Embedding lookup
        self.executor.encode_embedding_lookup(
            &mut encoder,
            &self.embedding_handle,
            token_id,
            &request.token_state,
        )?;
        encoder.copy_buffer_to_buffer(
            request.token_state.hidden_buffer(),
            0,
            diagnostic_staging_buffer,
            trace_layout.embedding_offset as u64,
            trace_layout.embedding_bytes as u64,
        );

        // 2. Attention Pre-Norm
        self.executor.encode_rms_norm_state_in_place(
            &mut encoder,
            &layer_0.rms_attn_handle,
            self.model_geometry.rms_eps,
            &request.token_state,
        )?;
        encoder.copy_buffer_to_buffer(
            request.token_state.hidden_buffer(),
            0,
            diagnostic_staging_buffer,
            trace_layout.attention_pre_norm_offset as u64,
            trace_layout.attention_pre_norm_bytes as u64,
        );

        // 3. Attention Prepare (Q, K, V, QK-Norm, RoPE, KV Append)
        self.executor.encode_attention_prepare_layer0_diagnostic(
            &mut encoder,
            &layer_0.attn_plan,
            &request.token_state,
            &request.attn_scratch,
            &request.kv_state,
            position,
            &sink,
        )?;

        // 4. Attention Complete (Causal Attention, O Projection, Residual Add)
        // The backend expects the absolute current position and derives seq_len internally.
        self.executor.encode_attention_complete_layer0_diagnostic(
            &mut encoder,
            &layer_0.attn_plan,
            &request.token_state,
            &request.attn_scratch,
            &request.kv_state,
            position,
            &sink,
        )?;

        // 5. Post-Attention Residual & Status copy
        encoder.copy_buffer_to_buffer(
            request.token_state.hidden_buffer(),
            0,
            diagnostic_staging_buffer,
            trace_layout.post_attention_residual_offset as u64,
            trace_layout.post_attention_residual_bytes as u64,
        );
        encoder.copy_buffer_to_buffer(
            request.token_state.status_buffer(),
            0,
            diagnostic_staging_buffer,
            trace_layout.status_offset as u64,
            4,
        );

        gpu.queue.submit(Some(encoder.finish()));

        // Map and parse diagnostic trace
        let slice = diagnostic_staging_buffer.slice(..trace_layout.total_bytes);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuNativeTokenLoopError::MapFailed(e.to_string()))?
            .map_err(|e| GpuNativeTokenLoopError::MapFailed(format!("{e:?}")))?;

        let mapped = slice.get_mapped_range();
        let trace = trace_layout
            .parse(&mapped)
            .map_err(|e| GpuNativeTokenLoopError::InvalidBoundaryReport { detail: e })?;
        drop(mapped);
        diagnostic_staging_buffer.unmap();

        if (trace.status & GPU_NATIVE_STATUS_FATAL_MASK) != 0 {
            return Err(GpuNativeTokenLoopError::FatalNumericalFailure {
                layer_index: Some(0),
                status_bits: trace.status,
            });
        }

        request.committed_position += 1;
        Ok(trace)
    }

    /// Allocate a device-resident staging buffer for Layer-0 diagnostic trace readbacks.
    pub fn create_layer0_diagnostic_staging_buffer(
        &self,
        trace_layout: &crate::gpu_native_layer0_diagnostics::Layer0AttentionDiagnosticTraceLayout,
    ) -> Result<wgpu::Buffer, GpuNativeTokenLoopError> {
        let gpu = self.executor.authoritative_gpu()?;
        let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_layer0_diagnostic_staging"),
            size: trace_layout.total_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Ok(staging_buffer)
    }

    /// Authoritative internal loop for stepping a token with bounded retries and demand residency service.
    /// Caller MUST hold self.execution_guard for the duration of the step or generation request.
    async fn step_token_unified_inner(
        &self,
        engine: &Arc<Engine>,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        sample: bool,
        diagnostic_sink: Option<(
            &crate::gpu_native_diagnostics::GpuNativeDiagnosticTraceLayout,
            &wgpu::Buffer,
        )>,
        router_rank_sink: Option<(
            &crate::gpu_native_router_rank_diagnostics::RouterRankTraceLayout,
            &wgpu::Buffer,
        )>,
        semantic_sink: Option<(
            GpuNativeSemanticDiagnosticLayout<'_>,
            &wgpu::Buffer,
            Option<&GpuNativeScratch>,
        )>,
    ) -> Result<GpuNativeStepOutput, GpuNativeTokenLoopError> {
        let max_attempts = gpu_native_attempt_bound(self.layers.len())?;
        let mut attempts = 0usize;
        let mut recovery: Option<GpuNativeRecoveryCursor> = None;
        let mut is_warm = true;

        loop {
            if attempts >= max_attempts {
                self.counters.fatal_failures.fetch_add(1, Ordering::Relaxed);
                return Err(GpuNativeTokenLoopError::AttemptBoundExceeded {
                    attempts,
                    max_attempts,
                });
            }

            let segment = match recovery.as_ref() {
                Some(cursor) => cursor.plan(self.layers.len())?,
                None => GpuNativeExecutionSegment::fresh(self.layers.len())?,
            };
            let sink = diagnostic_sink.map(|(layout, buf)| GpuNativeDiagnosticSink {
                layout,
                staging_buffer: buf,
            });
            let router_rank_sink =
                router_rank_sink.map(|(layout, buf)| GpuNativeRouterRankDiagnosticSink {
                    layout,
                    staging_buffer: buf,
                });
            let semantic_sink = semantic_sink.map(|(layout, buf, stage_scratch)| {
                GpuNativeExpertPermutationSemanticSink {
                    layout,
                    staging_buffer: buf,
                    stage_scratch,
                }
            });
            let output = if matches!(segment.attempt_start, GpuNativeAttemptStart::Fresh) {
                self.execute_token_attempt_unified(
                    request,
                    token_id,
                    position,
                    sample,
                    false,
                    sink.as_ref(),
                    router_rank_sink.as_ref(),
                    semantic_sink.as_ref(),
                )?
            } else {
                self.execute_token_segment_unified(
                    request,
                    token_id,
                    position,
                    sample,
                    false,
                    &segment,
                    sink.as_ref(),
                    router_rank_sink.as_ref(),
                    semantic_sink.as_ref(),
                )?
            };
            attempts += 1;
            let report = &output.boundary_report;

            if let Some(fail_layer) =
                report.first_failure_layer_in(segment.attempted_layers.clone())?
            {
                let layer_status = report.layer_statuses[fail_layer];
                match classify_gpu_native_status(layer_status, Some(fail_layer)) {
                    Ok(GpuNativeStatusDisposition::RetryableResidencyMiss) => {}
                    Ok(GpuNativeStatusDisposition::Clean) => {
                        self.counters.fatal_failures.fetch_add(1, Ordering::Relaxed);
                        return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                            detail: format!(
                                "failure layer {fail_layer} classified as clean status"
                            ),
                        });
                    }
                    Err(error) => {
                        self.counters.fatal_failures.fetch_add(1, Ordering::Relaxed);
                        return Err(error);
                    }
                }

                is_warm = false;
                self.counters
                    .residency_miss_attempts
                    .fetch_add(1, Ordering::Relaxed);
                self.recovery_counters
                    .invalid_tail_layers_encoded
                    .fetch_add(
                        segment
                            .attempted_layers
                            .end
                            .saturating_sub(fail_layer.saturating_add(1))
                            as u64,
                        Ordering::Relaxed,
                    );
                let local_ids = &report.selected_ids[fail_layer];
                if local_ids.len() != self.model_geometry.top_k {
                    return Err(GpuNativeTokenLoopError::InvalidTopKCount {
                        expected: self.model_geometry.top_k,
                        actual: local_ids.len(),
                    });
                }
                for &id in local_ids {
                    if id as usize >= self.model_geometry.num_experts {
                        return Err(GpuNativeTokenLoopError::InvalidSelectedExpertId {
                            layer_index: fail_layer,
                            expert_id: id,
                        });
                    }
                }
                let mut seen = HashSet::with_capacity(local_ids.len());
                for &id in local_ids {
                    if !seen.insert(id) {
                        return Err(GpuNativeTokenLoopError::DuplicateSelectedExpertId {
                            layer_index: fail_layer,
                            expert_id: id,
                        });
                    }
                }

                let miss = GpuNativeMissSignature {
                    layer_index: fail_layer,
                    selected_ids: local_ids.clone(),
                };
                if recovery
                    .as_ref()
                    .and_then(|cursor| cursor.last_miss.as_ref())
                    == Some(&miss)
                {
                    self.counters
                        .no_progress_failures
                        .fetch_add(1, Ordering::Relaxed);
                    return Err(GpuNativeTokenLoopError::NoProgress {
                        layer_index: fail_layer,
                        selected_ids: local_ids.clone(),
                    });
                }

                let layer_base = fail_layer
                    .checked_mul(self.model_geometry.num_experts)
                    .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                        detail: "global expert layer offset overflow".into(),
                    })?;
                let global_ids = local_ids
                    .iter()
                    .map(|&local_id| {
                        layer_base
                            .checked_add(local_id as usize)
                            .and_then(|global_id| u32::try_from(global_id).ok())
                            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                                detail: "global selected expert id overflow".into(),
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let residency_started = Instant::now();
                engine
                    .ensure_gpu_native_demand_residency(fail_layer, &global_ids)
                    .await
                    .map_err(GpuNativeTokenLoopError::ResidencyServiceFailed)?;
                self.recovery_counters
                    .residency_service_us
                    .fetch_add(saturating_micros(residency_started), Ordering::Relaxed);
                self.counters
                    .residency_services
                    .fetch_add(1, Ordering::Relaxed);

                match recovery.as_mut() {
                    Some(cursor) => cursor.record_serviced_miss(&segment, miss)?,
                    None => {
                        recovery = Some(GpuNativeRecoveryCursor::after_serviced_miss(
                            self.layers.len(),
                            miss,
                        )?);
                    }
                }
                continue;
            }

            if let Some(cursor) = recovery.as_mut() {
                let completed = cursor.record_clean_segment(&segment, self.layers.len())?;
                if completed != segment.completes_token {
                    return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                        detail: "recovery completion disagrees with encoded segment".into(),
                    });
                }
            }

            let selected_ids = match authoritative_actual_routes(report, &segment) {
                Ok(Some(selected_ids)) => selected_ids,
                Ok(None) => continue,
                Err(error) => {
                    self.counters.fatal_failures.fetch_add(1, Ordering::Relaxed);
                    return Err(error);
                }
            };

            engine
                .record_gpu_native_actual_routes(selected_ids)
                .map_err(GpuNativeTokenLoopError::ResidencyServiceFailed)?;

            request.committed_position += 1;
            self.counters
                .tokens_completed
                .fetch_add(1, Ordering::Relaxed);
            if is_warm {
                self.counters
                    .warm_tokens_completed
                    .fetch_add(1, Ordering::Relaxed);
            }

            return Ok(GpuNativeStepOutput {
                sampled_token: if sample {
                    Some(report.sampled_token)
                } else {
                    None
                },
                attempts,
                diagnostic_trace: output.diagnostic_trace,
                router_rank_trace: output.router_rank_trace,
                semantic_trace: output.semantic_trace,
                semantic_corpus_trace: output.semantic_corpus_trace,
                q4_expert_stage_trace: output.q4_expert_stage_trace,
            });
        }
    }

    /// Encode and execute one single attempt on device and read back the compact boundary report.
    pub fn execute_token_attempt(
        &self,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        sample: bool,
        replay: bool,
    ) -> Result<GpuNativeBoundaryReport, GpuNativeTokenLoopError> {
        let output = self.execute_token_attempt_unified(
            request, token_id, position, sample, replay, None, None, None,
        )?;
        Ok(output.boundary_report)
    }

    /// Diagnostic variant of execute_token_attempt that additionally records full activation traces.
    pub fn execute_token_attempt_diagnostic(
        &self,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        sample: bool,
        replay: bool,
        trace_layout: &crate::gpu_native_diagnostics::GpuNativeDiagnosticTraceLayout,
        diagnostic_staging_buffer: &wgpu::Buffer,
    ) -> Result<crate::gpu_native_diagnostics::GpuNativeDiagnosticTrace, GpuNativeTokenLoopError>
    {
        let sink = GpuNativeDiagnosticSink {
            layout: trace_layout,
            staging_buffer: diagnostic_staging_buffer,
        };
        let output = self.execute_token_attempt_unified(
            request,
            token_id,
            position,
            sample,
            replay,
            Some(&sink),
            None,
            None,
        )?;
        output
            .diagnostic_trace
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "diagnostic trace was not collected".into(),
            })
    }

    /// Encode the routed expert stage for one layer from the exact current
    /// hidden/residual/router state. This is shared by ordinary layers and
    /// checkpoint-restored recovery.
    fn encode_expert_layer_unified(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        request: &GpuNativeRequestState,
        layer_idx: usize,
        diagnostic_sink: Option<&GpuNativeDiagnosticSink<'_>>,
        semantic_sink: Option<&GpuNativeExpertPermutationSemanticSink<'_>>,
    ) -> Result<(), GpuNativeTokenLoopError> {
        let layer_plan = self.layers.get(layer_idx).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: format!("missing GPU-native layer plan {layer_idx}"),
            }
        })?;
        let arena = self.residency_manager.arena(layer_idx).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: format!("missing expert arena for layer {layer_idx}"),
            }
        })?;
        if let Some(sink) = semantic_sink {
            if let Some(layout) = sink.q4_expert_stage_layout_for_layer(layer_idx) {
                let stage_scratch = sink.stage_scratch.ok_or_else(|| {
                    GpuNativeTokenLoopError::InvalidBoundaryReport {
                        detail: "Q4 expert stage sink is missing diagnostic scratch".into(),
                    }
                })?;
                self.executor.encode_q4_expert_stage_diagnostic(
                    encoder,
                    &layer_plan.router_plan,
                    &request.router_scratch,
                    arena,
                    &request.token_state,
                    &request.expert_scratch,
                    stage_scratch,
                )?;
                encoder.copy_buffer_to_buffer(
                    stage_scratch.buffer(),
                    0,
                    sink.staging_buffer,
                    layout.stages_offset,
                    layout.stages_bytes,
                );
            }
        }
        self.executor.encode_q4_expert_arena_combine(
            encoder,
            &layer_plan.router_plan,
            &request.router_scratch,
            arena,
            &request.token_state,
            &request.expert_scratch,
        )?;

        if let Some(sink) = semantic_sink {
            if let Some(layout) = sink.target_layout_for_layer(layer_idx) {
                encoder.copy_buffer_to_buffer(
                    request.expert_scratch.route_outputs_buffer(),
                    0,
                    sink.staging_buffer,
                    layout.route_outputs_offset,
                    layout.route_outputs_bytes,
                );
                encoder.copy_buffer_to_buffer(
                    request.expert_scratch.combined_buffer(),
                    0,
                    sink.staging_buffer,
                    layout.routed_moe_output_offset,
                    layout.routed_moe_output_bytes,
                );
            }
            if let Some(layout) = sink.corpus_layout() {
                encoder.copy_buffer_to_buffer(
                    request.expert_scratch.route_outputs_buffer(),
                    0,
                    sink.staging_buffer,
                    layout.route_outputs_offset(layer_idx),
                    layout.route_outputs_layer_bytes,
                );
                encoder.copy_buffer_to_buffer(
                    request.expert_scratch.combined_buffer(),
                    0,
                    sink.staging_buffer,
                    layout.routed_moe_output_offset(layer_idx),
                    layout.routed_moe_output_layer_bytes,
                );
            }
            if let Some(layout) = sink.q4_expert_stage_layout_for_layer(layer_idx) {
                encoder.copy_buffer_to_buffer(
                    request.expert_scratch.route_outputs_buffer(),
                    0,
                    sink.staging_buffer,
                    layout.production_down_offset,
                    layout.production_down_bytes,
                );
            }
        }

        if let Some(sink) = diagnostic_sink {
            encoder.copy_buffer_to_buffer(
                request.token_state.hidden_buffer(),
                0,
                sink.staging_buffer,
                sink.layout.layer_post_moe_offset(layer_idx),
                (self.model_geometry.d_model * 4) as u64,
            );
            encoder.copy_buffer_to_buffer(
                request.token_state.status_buffer(),
                0,
                sink.staging_buffer,
                sink.layout.layer_status_offset(layer_idx),
                4,
            );
        }

        let status_offset = (layer_idx * 4) as u64;
        encoder.copy_buffer_to_buffer(
            request.token_state.status_buffer(),
            0,
            &request.staging_buffer,
            status_offset,
            4,
        );
        let ids_offset = (self.layers.len() * 4
            + layer_idx * self.model_geometry.top_k * std::mem::size_of::<u32>())
            as u64;
        encoder.copy_buffer_to_buffer(
            request.router_scratch.selected_ids_buffer(),
            0,
            &request.staging_buffer,
            ids_offset,
            (self.model_geometry.top_k * std::mem::size_of::<u32>()) as u64,
        );
        Ok(())
    }

    /// Single authoritative internal forward attempt encoder and executor.
    fn execute_token_attempt_unified(
        &self,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        sample: bool,
        replay: bool,
        diagnostic_sink: Option<&GpuNativeDiagnosticSink<'_>>,
        router_rank_sink: Option<&GpuNativeRouterRankDiagnosticSink<'_>>,
        semantic_sink: Option<&GpuNativeExpertPermutationSemanticSink<'_>>,
    ) -> Result<GpuNativeAttemptOutput, GpuNativeTokenLoopError> {
        let segment = GpuNativeExecutionSegment::fresh(self.layers.len())?;
        self.execute_token_segment_unified(
            request,
            token_id,
            position,
            sample,
            replay,
            &segment,
            diagnostic_sink,
            router_rank_sink,
            semantic_sink,
        )
    }

    /// Shared monolithic/recovery segment encoder and boundary executor.
    #[allow(clippy::too_many_arguments)]
    fn execute_token_segment_unified(
        &self,
        request: &mut GpuNativeRequestState,
        token_id: u32,
        position: usize,
        sample: bool,
        full_token_replay: bool,
        segment: &GpuNativeExecutionSegment,
        diagnostic_sink: Option<&GpuNativeDiagnosticSink<'_>>,
        router_rank_sink: Option<&GpuNativeRouterRankDiagnosticSink<'_>>,
        semantic_sink: Option<&GpuNativeExpertPermutationSemanticSink<'_>>,
    ) -> Result<GpuNativeAttemptOutput, GpuNativeTokenLoopError> {
        if position != request.committed_position {
            return Err(GpuNativeTokenLoopError::PositionMismatch {
                requested_position: position,
                committed_position: request.committed_position,
            });
        }
        if position >= request.max_seq_len {
            return Err(GpuNativeTokenLoopError::ContextLimitExceeded {
                requested_position: position,
                max_seq_len: request.max_seq_len,
            });
        }

        self.counters.token_attempts.fetch_add(1, Ordering::Relaxed);
        if full_token_replay {
            if segment.attempt_start != GpuNativeAttemptStart::Fresh {
                return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "full-token replay requires a fresh segment".into(),
                });
            }
            self.counters
                .replay_attempts
                .fetch_add(1, Ordering::Relaxed);
            self.recovery_counters
                .full_token_replay_attempts
                .fetch_add(1, Ordering::Relaxed);
        }

        let gpu = self.executor.authoritative_gpu()?;
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(if diagnostic_sink.is_some() {
                    "gpu_native_token_attempt_diagnostic"
                } else if router_rank_sink.is_some() {
                    "gpu_native_router_rank_diagnostic"
                } else if semantic_sink.is_some() {
                    "gpu_native_expert_permutation_semantic_diagnostic"
                } else {
                    "gpu_native_token_attempt"
                }),
            });

        let num_layers = self.layers.len();
        let top_k = self.model_geometry.top_k;
        let rms_eps = self.model_geometry.rms_eps;
        if segment.attempted_layers.start >= segment.attempted_layers.end
            || segment.attempted_layers.end > num_layers
            || segment.ordinary_layers.start > segment.ordinary_layers.end
            || segment.ordinary_layers.end > num_layers
        {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: format!("invalid GPU-native execution segment: {segment:?}"),
            });
        }

        match segment.attempt_start {
            GpuNativeAttemptStart::Fresh => {
                if full_token_replay {
                    self.executor
                        .encode_clear_retryable_status(&mut encoder, &request.token_state)?;
                }
                self.executor.encode_embedding_lookup(
                    &mut encoder,
                    &self.embedding_handle,
                    token_id,
                    &request.token_state,
                )?;
                if let Some(sink) = diagnostic_sink {
                    encoder.copy_buffer_to_buffer(
                        request.token_state.hidden_buffer(),
                        0,
                        sink.staging_buffer,
                        sink.layout.embedding_offset as u64,
                        sink.layout.embedding_bytes as u64,
                    );
                }
            }
            GpuNativeAttemptStart::ResumeExpert { layer_index } => {
                self.executor
                    .encode_clear_retryable_status(&mut encoder, &request.token_state)?;
                self.executor.encode_restore_pre_expert_checkpoint(
                    &mut encoder,
                    &request.pre_expert_checkpoints,
                    layer_index,
                    &request.token_state,
                    &request.router_scratch,
                )?;
                self.recovery_counters
                    .checkpoint_restores
                    .fetch_add(1, Ordering::Relaxed);
                self.encode_expert_layer_unified(
                    &mut encoder,
                    request,
                    layer_index,
                    diagnostic_sink,
                    semantic_sink,
                )?;
            }
            GpuNativeAttemptStart::Continue { layer_index } => {
                if segment.ordinary_layers.start != layer_index {
                    return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                        detail: format!("continuation does not start at layer {layer_index}"),
                    });
                }
            }
        }

        for layer_idx in segment.ordinary_layers.clone() {
            let layer_plan = &self.layers[layer_idx];
            // 1. Attention Pre-Norm
            self.executor.encode_rms_norm_state_in_place(
                &mut encoder,
                &layer_plan.rms_attn_handle,
                rms_eps,
                &request.token_state,
            )?;

            // 2. Attention Prepare
            self.executor.encode_attention_prepare(
                &mut encoder,
                &layer_plan.attn_plan,
                &request.token_state,
                &request.attn_scratch,
                &request.kv_state,
                position,
            )?;

            // 3. Attention Complete
            // The backend expects the absolute current position and derives seq_len internally.
            self.executor.encode_attention_complete(
                &mut encoder,
                &layer_plan.attn_plan,
                &request.token_state,
                &request.attn_scratch,
                &request.kv_state,
                position,
            )?;

            if let Some(sink) = diagnostic_sink {
                encoder.copy_buffer_to_buffer(
                    request.token_state.hidden_buffer(),
                    0,
                    sink.staging_buffer,
                    sink.layout.layer_post_attn_offset(layer_idx),
                    (self.model_geometry.d_model * 4) as u64,
                );
            }

            // 4. MoE Pre-Norm
            self.executor.encode_rms_norm_state_in_place(
                &mut encoder,
                &layer_plan.rms_moe_handle,
                rms_eps,
                &request.token_state,
            )?;

            if let Some(sink) = diagnostic_sink {
                encoder.copy_buffer_to_buffer(
                    request.token_state.hidden_buffer(),
                    0,
                    sink.staging_buffer,
                    sink.layout.layer_router_input_offset(layer_idx),
                    (self.model_geometry.d_model * 4) as u64,
                );
            }
            if let Some(sink) =
                router_rank_sink.filter(|sink| sink.layout.target_layer == layer_idx)
            {
                encoder.copy_buffer_to_buffer(
                    request.token_state.hidden_buffer(),
                    0,
                    sink.staging_buffer,
                    sink.layout.router_input_offset,
                    sink.layout.router_input_bytes,
                );
            }

            // 5. Router
            self.executor.encode_router(
                &mut encoder,
                &layer_plan.router_plan,
                &request.token_state,
                &request.router_scratch,
            )?;

            if let Some(sink) = diagnostic_sink {
                encoder.copy_buffer_to_buffer(
                    request.router_scratch.selected_ids_buffer(),
                    0,
                    sink.staging_buffer,
                    sink.layout.layer_selected_ids_offset(layer_idx),
                    (top_k * 4) as u64,
                );
                encoder.copy_buffer_to_buffer(
                    request.router_scratch.selected_weights_buffer(),
                    0,
                    sink.staging_buffer,
                    sink.layout.layer_selected_weights_offset(layer_idx),
                    (top_k * 4) as u64,
                );
            }
            if let Some(sink) =
                router_rank_sink.filter(|sink| sink.layout.target_layer == layer_idx)
            {
                encoder.copy_buffer_to_buffer(
                    request.router_scratch.logits_buffer(),
                    0,
                    sink.staging_buffer,
                    sink.layout.raw_logits_offset,
                    sink.layout.raw_logits_bytes,
                );
                encoder.copy_buffer_to_buffer(
                    request.router_scratch.selected_ids_buffer(),
                    0,
                    sink.staging_buffer,
                    sink.layout.selected_ids_offset,
                    sink.layout.selected_ids_bytes,
                );
                encoder.copy_buffer_to_buffer(
                    request.router_scratch.selected_weights_buffer(),
                    0,
                    sink.staging_buffer,
                    sink.layout.selected_weights_offset,
                    sink.layout.selected_weights_bytes,
                );
            }
            if let Some(sink) = semantic_sink {
                if let Some(layout) = sink.target_layout_for_layer(layer_idx) {
                    encoder.copy_buffer_to_buffer(
                        request.token_state.hidden_buffer(),
                        0,
                        sink.staging_buffer,
                        layout.router_input_offset,
                        layout.router_input_bytes,
                    );
                    encoder.copy_buffer_to_buffer(
                        request.router_scratch.logits_buffer(),
                        0,
                        sink.staging_buffer,
                        layout.raw_logits_offset,
                        layout.raw_logits_bytes,
                    );
                    encoder.copy_buffer_to_buffer(
                        request.router_scratch.selected_ids_buffer(),
                        0,
                        sink.staging_buffer,
                        layout.selected_ids_offset,
                        layout.selected_ids_bytes,
                    );
                    encoder.copy_buffer_to_buffer(
                        request.router_scratch.selected_weights_buffer(),
                        0,
                        sink.staging_buffer,
                        layout.selected_weights_offset,
                        layout.selected_weights_bytes,
                    );
                }
                if let Some(layout) = sink.corpus_layout() {
                    encoder.copy_buffer_to_buffer(
                        request.token_state.hidden_buffer(),
                        0,
                        sink.staging_buffer,
                        layout.router_input_offset(layer_idx),
                        layout.router_input_layer_bytes,
                    );
                    encoder.copy_buffer_to_buffer(
                        request.router_scratch.logits_buffer(),
                        0,
                        sink.staging_buffer,
                        layout.raw_logits_offset(layer_idx),
                        layout.raw_logits_layer_bytes,
                    );
                    encoder.copy_buffer_to_buffer(
                        request.router_scratch.selected_ids_buffer(),
                        0,
                        sink.staging_buffer,
                        layout.selected_ids_offset(layer_idx),
                        layout.selected_ids_layer_bytes,
                    );
                    encoder.copy_buffer_to_buffer(
                        request.router_scratch.selected_weights_buffer(),
                        0,
                        sink.staging_buffer,
                        layout.selected_weights_offset(layer_idx),
                        layout.selected_weights_layer_bytes,
                    );
                }
                if let Some(layout) = sink.q4_expert_stage_layout_for_layer(layer_idx) {
                    encoder.copy_buffer_to_buffer(
                        request.token_state.hidden_buffer(),
                        0,
                        sink.staging_buffer,
                        layout.router_input_offset,
                        layout.router_input_bytes,
                    );
                    encoder.copy_buffer_to_buffer(
                        request.router_scratch.selected_ids_buffer(),
                        0,
                        sink.staging_buffer,
                        layout.selected_ids_offset,
                        layout.selected_ids_bytes,
                    );
                    encoder.copy_buffer_to_buffer(
                        request.router_scratch.selected_weights_buffer(),
                        0,
                        sink.staging_buffer,
                        layout.selected_weights_offset,
                        layout.selected_weights_bytes,
                    );
                }
            }

            self.executor.encode_capture_pre_expert_checkpoint(
                &mut encoder,
                &request.pre_expert_checkpoints,
                layer_idx,
                &request.token_state,
                &request.router_scratch,
            )?;
            self.recovery_counters
                .checkpoint_captures
                .fetch_add(1, Ordering::Relaxed);

            self.encode_expert_layer_unified(
                &mut encoder,
                request,
                layer_idx,
                diagnostic_sink,
                semantic_sink,
            )?;
        }

        if segment.completes_token && sample {
            // Final RMSNorm
            self.executor.encode_rms_norm_hidden_in_place(
                &mut encoder,
                &self.final_norm_handle,
                rms_eps,
                &request.token_state,
            )?;

            if let Some(sink) = diagnostic_sink {
                encoder.copy_buffer_to_buffer(
                    request.token_state.hidden_buffer(),
                    0,
                    sink.staging_buffer,
                    sink.layout.final_norm_offset as u64,
                    sink.layout.final_norm_bytes as u64,
                );
            }

            // LM Head GEMV
            self.executor.encode_dense_gemv_hidden_to_scratch(
                &mut encoder,
                &self.lm_head_handle,
                &request.token_state,
                &request.logits_scratch,
            )?;

            if let Some(sink) = diagnostic_sink {
                encoder.copy_buffer_to_buffer(
                    request.logits_scratch.buffer(),
                    0,
                    sink.staging_buffer,
                    sink.layout.logits_offset as u64,
                    sink.layout.logits_bytes as u64,
                );
            }

            // GPU Greedy Argmax
            self.executor.encode_greedy_argmax(
                &mut encoder,
                &request.logits_scratch,
                &request.sampled_token_buf,
                &request.token_state,
                self.model_geometry.vocab_size,
            )?;

            if let Some(sink) = diagnostic_sink {
                encoder.copy_buffer_to_buffer(
                    request.sampled_token_buf.buffer(),
                    0,
                    sink.staging_buffer,
                    sink.layout.sampled_token_offset as u64,
                    4,
                );
                encoder.copy_buffer_to_buffer(
                    request.token_state.status_buffer(),
                    0,
                    sink.staging_buffer,
                    sink.layout.final_status_offset as u64,
                    4,
                );
            }

            // Copy final status and sampled token to production staging
            let final_status_offset = (num_layers * 4 + num_layers * top_k * 4) as u64;
            encoder.copy_buffer_to_buffer(
                request.token_state.status_buffer(),
                0,
                &request.staging_buffer,
                final_status_offset,
                4,
            );
            let token_offset = final_status_offset + 4;
            encoder.copy_buffer_to_buffer(
                request.sampled_token_buf.buffer(),
                0,
                &request.staging_buffer,
                token_offset,
                4,
            );
        } else if segment.completes_token {
            // Ingest-only: copy final status to production staging
            let final_status_offset = (num_layers * 4 + num_layers * top_k * 4) as u64;
            encoder.copy_buffer_to_buffer(
                request.token_state.status_buffer(),
                0,
                &request.staging_buffer,
                final_status_offset,
                4,
            );
            if let Some(sink) = diagnostic_sink {
                encoder.copy_buffer_to_buffer(
                    request.token_state.status_buffer(),
                    0,
                    sink.staging_buffer,
                    sink.layout.final_status_offset as u64,
                    4,
                );
            }
        }

        let resume_layers = usize::from(matches!(
            segment.attempt_start,
            GpuNativeAttemptStart::ResumeExpert { .. }
        ));
        self.recovery_counters.layers_encoded.fetch_add(
            (segment.ordinary_layers.len() + resume_layers) as u64,
            Ordering::Relaxed,
        );
        if !matches!(segment.attempt_start, GpuNativeAttemptStart::Fresh) {
            self.recovery_counters
                .recovery_segments
                .fetch_add(1, Ordering::Relaxed);
            self.recovery_counters
                .attention_layers_reexecuted
                .fetch_add(segment.ordinary_layers.len() as u64, Ordering::Relaxed);
            self.recovery_counters.expert_layers_reexecuted.fetch_add(
                (segment.ordinary_layers.len() + resume_layers) as u64,
                Ordering::Relaxed,
            );
        }
        if resume_layers == 1 {
            self.recovery_counters
                .resume_attempts
                .fetch_add(1, Ordering::Relaxed);
        }

        // ONE submission
        gpu.queue.submit(Some(encoder.finish()));
        self.counters
            .queue_submissions
            .fetch_add(1, Ordering::Relaxed);

        // ONE map/readback for production boundary report
        let slice = request
            .staging_buffer
            .slice(..self.report_layout.total_bytes);
        let boundary_wait_started = Instant::now();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.counters.boundary_maps.fetch_add(1, Ordering::Relaxed);

        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| GpuNativeTokenLoopError::MapFailed(e.to_string()))?
            .map_err(|e| GpuNativeTokenLoopError::MapFailed(format!("{e:?}")))?;

        let mapped = slice.get_mapped_range();
        let report = self.report_layout.parse(&mapped)?;
        drop(mapped);
        request.staging_buffer.unmap();

        self.counters
            .boundary_readbacks
            .fetch_add(1, Ordering::Relaxed);
        self.recovery_counters
            .boundary_wait_us
            .fetch_add(saturating_micros(boundary_wait_started), Ordering::Relaxed);

        let diagnostic_trace = if let Some(sink) = diagnostic_sink {
            let diag_slice = sink.staging_buffer.slice(..sink.layout.total_bytes);
            let (tx_d, rx_d) = std::sync::mpsc::sync_channel(1);
            diag_slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx_d.send(res);
            });
            gpu.device.poll(wgpu::Maintain::Wait);
            rx_d.recv()
                .map_err(|e| GpuNativeTokenLoopError::MapFailed(e.to_string()))?
                .map_err(|e| GpuNativeTokenLoopError::MapFailed(format!("{e:?}")))?;
            let diag_mapped = diag_slice.get_mapped_range();
            let trace = sink.layout.parse(&diag_mapped)?;
            drop(diag_mapped);
            sink.staging_buffer.unmap();
            Some(trace)
        } else {
            None
        };

        let router_rank_trace = if let Some(sink) = router_rank_sink {
            let rank_slice = sink.staging_buffer.slice(..sink.layout.total_bytes);
            let (tx_rank, rx_rank) = std::sync::mpsc::sync_channel(1);
            rank_slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx_rank.send(res);
            });
            gpu.device.poll(wgpu::Maintain::Wait);
            rx_rank
                .recv()
                .map_err(|e| GpuNativeTokenLoopError::MapFailed(e.to_string()))?
                .map_err(|e| GpuNativeTokenLoopError::MapFailed(format!("{e:?}")))?;
            let rank_mapped = rank_slice.get_mapped_range();
            let trace = sink
                .layout
                .parse(&rank_mapped)
                .map_err(|detail| GpuNativeTokenLoopError::InvalidBoundaryReport { detail })?;
            drop(rank_mapped);
            sink.staging_buffer.unmap();
            Some(trace)
        } else {
            None
        };

        let (semantic_trace, semantic_corpus_trace, q4_expert_stage_trace) =
            if let Some(sink) = semantic_sink {
                let semantic_slice = sink.staging_buffer.slice(..sink.total_bytes());
                let (tx_semantic, rx_semantic) = std::sync::mpsc::sync_channel(1);
                semantic_slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx_semantic.send(result);
                });
                gpu.device.poll(wgpu::Maintain::Wait);
                rx_semantic
                    .recv()
                    .map_err(|error| GpuNativeTokenLoopError::MapFailed(error.to_string()))?
                    .map_err(|error| GpuNativeTokenLoopError::MapFailed(format!("{error:?}")))?;
                let mapped = semantic_slice.get_mapped_range();
                let parsed = match sink.layout {
                    GpuNativeSemanticDiagnosticLayout::Target(layout) => (
                        Some(layout.parse(&mapped).map_err(|detail| {
                            GpuNativeTokenLoopError::InvalidBoundaryReport { detail }
                        })?),
                        None,
                        None,
                    ),
                    GpuNativeSemanticDiagnosticLayout::Corpus(layout) => (
                        None,
                        Some(layout.parse(&mapped).map_err(|detail| {
                            GpuNativeTokenLoopError::InvalidBoundaryReport { detail }
                        })?),
                        None,
                    ),
                    GpuNativeSemanticDiagnosticLayout::Q4ExpertStages(layout) => (
                        None,
                        None,
                        Some(layout.parse(&mapped).map_err(|detail| {
                            GpuNativeTokenLoopError::InvalidBoundaryReport { detail }
                        })?),
                    ),
                };
                drop(mapped);
                sink.staging_buffer.unmap();
                parsed
            } else {
                (None, None, None)
            };

        Ok(GpuNativeAttemptOutput {
            boundary_report: report,
            diagnostic_trace,
            router_rank_trace,
            semantic_trace,
            semantic_corpus_trace,
            q4_expert_stage_trace,
        })
    }
}

/// Request-local device resources for one GPU-native generation sequence.
pub struct GpuNativeRequestState {
    pub token_state: GpuNativeTokenState,
    pub kv_state: GpuNativeKvState,
    pub attn_scratch: GpuNativeAttentionScratch,
    pub router_scratch: GpuNativeRouterScratch,
    pub expert_scratch: GpuNativeQ4ExpertScratch,
    pub logits_scratch: GpuNativeScratch,
    pub sampled_token_buf: GpuNativeScratch,
    pub pre_expert_checkpoints: GpuNativePreExpertCheckpoints,
    pub staging_buffer: wgpu::Buffer,
    pub committed_position: usize,
    pub max_seq_len: usize,
}

impl GpuNativeRequestState {
    pub fn committed_position(&self) -> usize {
        self.committed_position
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::architecture::Architecture;
    use crate::backend::gpu_native::{
        GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE, GPU_NATIVE_STATUS_EXPERT_NUMERICAL_FAILURE,
        GPU_NATIVE_STATUS_LM_HEAD_NUMERICAL_FAILURE, GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE,
    };
    use crate::config::Config;
    use crate::dense_tensor::DenseWeight;
    use crate::gating::{LinearGate, ScoringFunc};
    use crate::model::RealModelConfig;
    use crate::transformer::{LMHead, MultiHeadSelfAttention, RmsNorm, TransformerLayer};

    pub(crate) fn make_test_qwen3_moe_config() -> RealModelConfig {
        RealModelConfig {
            d_model: 32,
            d_ff: 32,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: 32,
            num_layers: 2,
            num_experts: 4,
            top_k: 2,
            rope_base: 10_000.0,
            rms_eps: 1e-5,
            window_size: None,
            architecture: Architecture::Qwen3Moe,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        }
    }

    pub(crate) fn make_test_qwen3_moe_model(cfg: RealModelConfig) -> RealModel {
        let embedding = DenseWeight::from_f32(
            vec![0.1; cfg.vocab_size * cfg.d_model],
            cfg.vocab_size,
            cfg.d_model,
        );
        let lm_head = LMHead::new(
            vec![0.1; cfg.vocab_size * cfg.d_model],
            cfg.vocab_size,
            cfg.d_model,
        );
        let final_rms = RmsNorm::new(vec![1.0; cfg.d_model], cfg.rms_eps);

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            let attn = MultiHeadSelfAttention {
                d_model: cfg.d_model,
                num_heads: cfg.num_heads,
                num_kv_heads: cfg.num_kv_heads,
                head_dim: cfg.head_dim,
                rope_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                attention_value_scale: None,
                rope_base: cfg.rope_base,
                wq: DenseWeight::from_f32(
                    vec![0.01; cfg.num_heads * cfg.head_dim * cfg.d_model],
                    cfg.num_heads * cfg.head_dim,
                    cfg.d_model,
                ),
                wk: DenseWeight::from_f32(
                    vec![0.01; cfg.num_kv_heads * cfg.head_dim * cfg.d_model],
                    cfg.num_kv_heads * cfg.head_dim,
                    cfg.d_model,
                ),
                wv: DenseWeight::from_f32(
                    vec![0.01; cfg.num_kv_heads * cfg.head_dim * cfg.d_model],
                    cfg.num_kv_heads * cfg.head_dim,
                    cfg.d_model,
                ),
                wo: DenseWeight::from_f32(
                    vec![0.01; cfg.d_model * cfg.num_heads * cfg.head_dim],
                    cfg.d_model,
                    cfg.num_heads * cfg.head_dim,
                ),
                window_size: None,
                q_norm: Some(RmsNorm::new(vec![1.0; cfg.head_dim], cfg.rms_eps)),
                k_norm: Some(RmsNorm::new(vec![1.0; cfg.head_dim], cfg.rms_eps)),
                rope_yarn: None,
                rope_cache: None,
                bq: None,
                bk: None,
                bv: None,
                bo: None,
                sink_bias: None,
            };
            let gate = LinearGate::new(
                vec![0.1; cfg.num_experts * cfg.d_model],
                cfg.num_experts,
                cfg.d_model,
                cfg.top_k,
            );
            let layer = TransformerLayer {
                rms_attn: RmsNorm::new(vec![1.0; cfg.d_model], cfg.rms_eps),
                attn,
                mla: None,
                rms_moe: RmsNorm::new(vec![1.0; cfg.d_model], cfg.rms_eps),
                gate,
                shared_expert: None,
                dense_ffn: None,
            };
            layers.push(layer);
        }

        RealModel {
            config: cfg,
            embedding,
            layers,
            final_rms,
            lm_head,
            load_status: crate::model::WeightLoadStatus::default(),
        }
    }

    #[test]
    fn compatibility_validation_accepts_supported_qwen3_moe() {
        let cfg = make_test_qwen3_moe_config();
        let model = make_test_qwen3_moe_model(cfg);
        assert!(GpuNativeTokenLoop::validate_model_compatibility(&model).is_ok());
    }

    #[test]
    fn compatibility_validation_rejects_wrong_architecture() {
        let mut cfg = make_test_qwen3_moe_config();
        cfg.architecture = Architecture::Mixtral;
        let model = make_test_qwen3_moe_model(cfg);
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model),
            Err(GpuNativeModelCompatibilityError::UnsupportedArchitecture { .. })
        ));
    }

    #[test]
    fn compatibility_validation_rejects_dense_and_shared_experts() {
        let cfg = make_test_qwen3_moe_config();
        let mut model = make_test_qwen3_moe_model(cfg);
        model.layers[0].dense_ffn = Some(
            crate::transformer::SharedExpert::from_projections(
                32,
                32,
                &vec![0.1; 32 * 32],
                &vec![0.1; 32 * 32],
                &vec![0.1; 32 * 32],
                None,
            )
            .unwrap(),
        );
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model),
            Err(GpuNativeModelCompatibilityError::DenseLayerUnsupported { .. })
        ));

        let cfg2 = make_test_qwen3_moe_config();
        let mut model2 = make_test_qwen3_moe_model(cfg2);
        model2.layers[0].shared_expert = Some(
            crate::transformer::SharedExpert::from_projections(
                32,
                32,
                &vec![0.1; 32 * 32],
                &vec![0.1; 32 * 32],
                &vec![0.1; 32 * 32],
                None,
            )
            .unwrap(),
        );
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model2),
            Err(GpuNativeModelCompatibilityError::SharedExpertUnsupported { .. })
        ));
    }

    #[test]
    fn compatibility_validation_rejects_asymmetric_v_and_sinks() {
        let cfg = make_test_qwen3_moe_config();
        let mut model = make_test_qwen3_moe_model(cfg);
        model.layers[0].attn.v_head_dim = 32;
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model),
            Err(GpuNativeModelCompatibilityError::AsymmetricVHeadDim { .. })
        ));

        let cfg2 = make_test_qwen3_moe_config();
        let mut model2 = make_test_qwen3_moe_model(cfg2);
        model2.layers[0].attn.sink_bias = Some(vec![0.0; 2]);
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model2),
            Err(GpuNativeModelCompatibilityError::AttentionSinkUnsupported { .. })
        ));
    }

    #[test]
    fn compatibility_validation_rejects_biases_and_sliding_window() {
        let cfg = make_test_qwen3_moe_config();
        let mut model = make_test_qwen3_moe_model(cfg);
        model.layers[0].attn.bq = Some(vec![0.0; 32]);
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model),
            Err(GpuNativeModelCompatibilityError::AttentionBiasesUnsupported { .. })
        ));

        let mut cfg2 = make_test_qwen3_moe_config();
        cfg2.window_size = Some(4096);
        let model2 = make_test_qwen3_moe_model(cfg2);
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model2),
            Err(GpuNativeModelCompatibilityError::SlidingWindowUnsupported { .. })
        ));
    }

    #[test]
    fn compatibility_validation_rejects_non_softmax_and_router_features() {
        let cfg = make_test_qwen3_moe_config();
        let mut model = make_test_qwen3_moe_model(cfg);
        model.layers[0].gate.scoring_func = ScoringFunc::Sigmoid;
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model),
            Err(GpuNativeModelCompatibilityError::NonSoftmaxRouter { .. })
        ));

        let cfg2 = make_test_qwen3_moe_config();
        let mut model2 = make_test_qwen3_moe_model(cfg2);
        model2.layers[0].gate.correction_bias = Some(vec![0.0; 4]);
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model2),
            Err(GpuNativeModelCompatibilityError::RouterCorrectionBiasUnsupported { .. })
        ));

        let cfg3 = make_test_qwen3_moe_config();
        let mut model3 = make_test_qwen3_moe_model(cfg3);
        model3.layers[0].gate.n_group = 2;
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model3),
            Err(GpuNativeModelCompatibilityError::GroupedRoutingUnsupported { .. })
        ));

        let cfg4 = make_test_qwen3_moe_config();
        let mut model4 = make_test_qwen3_moe_model(cfg4);
        model4.layers[0].gate.routed_scaling_factor = 2.0;
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model4),
            Err(GpuNativeModelCompatibilityError::RoutedScalingFactorUnsupported { .. })
        ));
    }

    #[test]
    fn compatibility_validation_validates_rope_dim_uniformity() {
        let cfg = make_test_qwen3_moe_config();
        let mut model = make_test_qwen3_moe_model(cfg);
        // Valid uniform rope_dim
        assert_eq!(model.layers[0].attn.rope_dim, 16);
        assert_eq!(model.layers[1].attn.rope_dim, 16);
        assert!(GpuNativeTokenLoop::validate_model_compatibility(&model).is_ok());

        // Inconsistent rope_dim across layers fails closed
        model.layers[1].attn.rope_dim = 8;
        let err = GpuNativeTokenLoop::validate_model_compatibility(&model).unwrap_err();
        assert_eq!(
            err,
            GpuNativeModelCompatibilityError::InconsistentRopeDimension {
                layer_index: 1,
                expected: 16,
                actual: 8,
            }
        );
        let msg = err.to_string();
        assert!(msg.contains("layer 1 has rope_dim 8, expected uniform rope_dim 16"));
    }

    #[test]
    fn model_rope_dim_is_derived_independently_from_max_seq_len() {
        let cfg = make_test_qwen3_moe_config();
        let model = make_test_qwen3_moe_model(cfg);
        assert_eq!(model.layers[0].attn.rope_dim, 16);

        // max_seq_len (e.g. 8) and rope_dim (16) must remain distinct and independent
        let max_seq_len = 8usize;
        let rope_dim = model.layers[0].attn.rope_dim;
        assert_ne!(rope_dim, max_seq_len);

        let geom = GpuNativeModelGeometry {
            num_layers: model.config.num_layers,
            d_model: model.config.d_model,
            d_ff: model.config.d_ff,
            num_experts: model.config.num_experts,
            top_k: model.config.top_k,
            num_heads: model.config.num_heads,
            num_kv_heads: model.config.num_kv_heads,
            head_dim: model.config.head_dim,
            rope_dim,
            vocab_size: model.config.vocab_size,
            max_seq_len,
            rms_eps: model.config.rms_eps,
            rope_base: model.config.rope_base,
        };
        assert_eq!(geom.rope_dim, 16);
        assert_eq!(geom.max_seq_len, 8);

        // Attention geometry construction must accept rope_dim=16, not max_seq_len=8
        let attn_geom = GpuNativeAttentionGeometry::try_new(
            geom.d_model,
            geom.num_heads,
            geom.num_kv_heads,
            geom.head_dim,
            geom.rope_dim,
        )
        .unwrap();
        assert_eq!(attn_geom.rope_dim(), 16);
    }

    #[test]
    fn boundary_report_layout_and_parser_work_correctly() {
        let layout = GpuNativeBoundaryReportLayout::try_new(2, 2).unwrap();
        assert_eq!(layout.total_bytes, 32);
        assert_eq!(layout.layer_status_offset, 0);
        assert_eq!(layout.layer_status_bytes, 8);
        assert_eq!(layout.selected_ids_offset, 8);
        assert_eq!(layout.selected_ids_bytes, 16);
        assert_eq!(layout.final_status_offset, 24);
        assert_eq!(layout.sampled_token_offset, 28);

        let mut bytes = vec![0u8; 32];
        bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&3u32.to_le_bytes());
        bytes[16..20].copy_from_slice(&0u32.to_le_bytes());
        bytes[20..24].copy_from_slice(&1u32.to_le_bytes());
        bytes[24..28].copy_from_slice(&0u32.to_le_bytes());
        bytes[28..32].copy_from_slice(&17u32.to_le_bytes());

        let report = layout.parse(&bytes).unwrap();
        assert_eq!(report.layer_statuses, vec![0, 0]);
        assert_eq!(report.selected_ids, vec![vec![2, 3], vec![0, 1]]);
        assert_eq!(report.final_status, 0);
        assert_eq!(report.sampled_token, 17);
        assert_eq!(report.first_failure_layer(), None);
    }

    #[test]
    fn first_failure_layer_detects_first_nonzero_transition() {
        let layout = GpuNativeBoundaryReportLayout::try_new(4, 2).unwrap();
        let mut bytes = vec![0u8; layout.total_bytes as usize];
        bytes[8..12].copy_from_slice(&4u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&4u32.to_le_bytes());

        let report = layout.parse(&bytes).unwrap();
        assert_eq!(report.first_failure_layer(), Some(2));
    }

    fn recovery_miss(layer_index: usize) -> GpuNativeMissSignature {
        GpuNativeMissSignature {
            layer_index,
            selected_ids: vec![7, 2, 5, 1],
        }
    }

    #[test]
    fn zero_miss_warm_plan_is_one_monolithic_boundary() {
        let segment = GpuNativeExecutionSegment::fresh(48).unwrap();
        assert_eq!(segment.attempt_start, GpuNativeAttemptStart::Fresh);
        assert_eq!(segment.ordinary_layers, 0..48);
        assert_eq!(segment.attempted_layers, 0..48);
        assert!(segment.completes_token);
        assert_eq!(gpu_native_attempt_bound(48).unwrap(), 98);
    }

    #[test]
    fn recovery_resumes_layer_zero_middle_and_last_without_prior_layers() {
        for layer_index in [0, 24, 47] {
            let cursor =
                GpuNativeRecoveryCursor::after_serviced_miss(48, recovery_miss(layer_index))
                    .unwrap();
            let segment = cursor.plan(48).unwrap();
            assert_eq!(
                segment.attempt_start,
                GpuNativeAttemptStart::ResumeExpert { layer_index }
            );
            assert_eq!(segment.attempted_layers.start, layer_index);
            assert_eq!(segment.ordinary_layers.start, layer_index + 1);
            assert_eq!(segment.ordinary_layers.end, (layer_index + 2).min(48));
            assert_eq!(segment.completes_token, layer_index == 47);
            assert!(segment.ordinary_layers.start > layer_index);
        }
    }

    #[test]
    fn recovery_window_doubles_one_two_four_eight_and_caps() {
        let mut cursor =
            GpuNativeRecoveryCursor::after_serviced_miss(48, recovery_miss(10)).unwrap();
        let expected = [
            (1, 11..12),
            (2, 12..14),
            (4, 14..18),
            (8, 18..26),
            (8, 26..34),
        ];
        for (window, ordinary_layers) in expected {
            assert_eq!(cursor.window_layers, window);
            let segment = cursor.plan(48).unwrap();
            assert_eq!(segment.ordinary_layers, ordinary_layers);
            assert!(!cursor.record_clean_segment(&segment, 48).unwrap());
        }
        assert_eq!(cursor.window_layers, 8);
        assert_eq!(cursor.clean_segments, 5);
    }

    #[test]
    fn new_miss_resets_window_and_strictly_increasing_misses_resume() {
        let mut cursor =
            GpuNativeRecoveryCursor::after_serviced_miss(48, recovery_miss(0)).unwrap();
        let first = cursor.plan(48).unwrap();
        cursor
            .record_serviced_miss(&first, recovery_miss(1))
            .unwrap();
        assert_eq!(cursor.window_layers, 1);
        assert_eq!(cursor.pending_resume_layer, Some(1));
        let second = cursor.plan(48).unwrap();
        assert_eq!(second.attempted_layers, 1..3);
        cursor
            .record_serviced_miss(&second, recovery_miss(2))
            .unwrap();
        assert_eq!(cursor.window_layers, 1);
        assert_eq!(cursor.residency_services, 3);

        let third = cursor.plan(48).unwrap();
        assert!(cursor.record_clean_segment(&third, 48).is_ok());
        let fourth = cursor.plan(48).unwrap();
        assert_eq!(cursor.window_layers, 2);
        cursor
            .record_serviced_miss(&fourth, recovery_miss(fourth.ordinary_layers.start))
            .unwrap();
        assert_eq!(cursor.window_layers, 1);
    }

    #[test]
    fn repeated_same_ordered_miss_is_no_progress() {
        let mut cursor =
            GpuNativeRecoveryCursor::after_serviced_miss(48, recovery_miss(10)).unwrap();
        let segment = cursor.plan(48).unwrap();
        assert_eq!(
            cursor.record_serviced_miss(&segment, recovery_miss(10)),
            Err(GpuNativeTokenLoopError::NoProgress {
                layer_index: 10,
                selected_ids: vec![7, 2, 5, 1],
            })
        );
        let reordered = GpuNativeMissSignature {
            layer_index: 10,
            selected_ids: vec![2, 7, 5, 1],
        };
        assert!(cursor.record_serviced_miss(&segment, reordered).is_ok());
    }

    #[test]
    fn recovery_cursor_fails_closed_without_forward_progress() {
        let mut cursor =
            GpuNativeRecoveryCursor::after_serviced_miss(48, recovery_miss(10)).unwrap();
        cursor.window_layers = 0;
        assert!(matches!(
            cursor.plan(48),
            Err(GpuNativeTokenLoopError::InvalidBoundaryReport { .. })
        ));

        let mut last = GpuNativeRecoveryCursor::after_serviced_miss(48, recovery_miss(47)).unwrap();
        let final_segment = last.plan(48).unwrap();
        assert!(last.record_clean_segment(&final_segment, 48).unwrap());
        assert!(matches!(
            last.plan(48),
            Err(GpuNativeTokenLoopError::InvalidBoundaryReport { .. })
        ));

        assert!(matches!(
            gpu_native_attempt_bound(usize::MAX),
            Err(GpuNativeTokenLoopError::InvalidBoundaryReport { .. })
        ));
    }

    #[test]
    fn attempted_range_ignores_stale_statuses_outside_segment() {
        let report = GpuNativeBoundaryReport {
            layer_statuses: vec![GPU_NATIVE_STATUS_RETRYABLE_MASK, 0, 0, 1 << 31],
            selected_ids: vec![vec![0, 1]; 4],
            final_status: GPU_NATIVE_STATUS_RETRYABLE_MASK,
            sampled_token: 9,
        };
        assert_eq!(report.first_failure_layer(), Some(0));
        assert_eq!(report.first_failure_layer_in(1..3).unwrap(), None);
        assert_eq!(report.first_failure_layer_in(2..4).unwrap(), Some(3));
        assert!(report.first_failure_layer_in(4..4).is_err());
    }

    #[test]
    fn only_one_completed_clean_attempt_authorizes_actual_route_recency() {
        let failed_attempt = GpuNativeBoundaryReport {
            layer_statuses: vec![GPU_NATIVE_STATUS_RETRYABLE_MASK, 0, 0, 0],
            selected_ids: vec![vec![0, 1]; 4],
            final_status: GPU_NATIVE_STATUS_RETRYABLE_MASK,
            sampled_token: 9,
        };
        let incomplete_attempt = GpuNativeBoundaryReport {
            layer_statuses: vec![0; 4],
            selected_ids: vec![vec![1, 0]; 4],
            final_status: 0,
            sampled_token: 10,
        };
        let completed_attempt = GpuNativeBoundaryReport {
            layer_statuses: vec![0; 4],
            selected_ids: vec![vec![2, 3]; 4],
            final_status: 0,
            sampled_token: 11,
        };
        let fresh = GpuNativeExecutionSegment::fresh(4).unwrap();
        let incomplete = GpuNativeRecoveryCursor::after_serviced_miss(4, recovery_miss(0))
            .unwrap()
            .plan(4)
            .unwrap();
        assert!(!incomplete.completes_token);

        let mut recency_invocations = 0;
        for (report, segment) in [
            (&failed_attempt, &fresh),
            (&incomplete_attempt, &incomplete),
            (&completed_attempt, &fresh),
        ] {
            if authoritative_actual_routes(report, segment)
                .unwrap()
                .is_some()
            {
                recency_invocations += 1;
            }
        }
        assert_eq!(recency_invocations, 1);
        assert_eq!(
            authoritative_actual_routes(&completed_attempt, &fresh).unwrap(),
            Some(completed_attempt.selected_ids.as_slice())
        );
    }

    #[test]
    fn recovery_kv_rows_equal_warm_reference_and_commit_once() {
        const NUM_LAYERS: usize = 8;
        const POSITION: u64 = 7;
        let warm = (0..NUM_LAYERS)
            .map(|layer| POSITION * 100 + layer as u64)
            .collect::<Vec<_>>();
        let invalid = u64::MAX;
        let first_miss = 2usize;

        // The monolithic miss attempt has authoritative KV through the failed
        // layer; later invalid tail rows will be overwritten at the same
        // absolute position by recovery.
        let mut recovered = warm.clone();
        recovered[first_miss + 1..].fill(invalid);
        let prior_rows = recovered[..=first_miss].to_vec();
        let mut committed_position = POSITION as usize;
        let mut cursor =
            GpuNativeRecoveryCursor::after_serviced_miss(NUM_LAYERS, recovery_miss(first_miss))
                .unwrap();
        let mut injected_second_miss = false;

        loop {
            let segment = cursor.plan(NUM_LAYERS).unwrap();
            for layer in segment.ordinary_layers.clone() {
                recovered[layer] = warm[layer];
            }
            if !injected_second_miss && segment.ordinary_layers.contains(&4) {
                let second_miss = 4;
                for value in &mut recovered[second_miss + 1..segment.ordinary_layers.end] {
                    *value = invalid;
                }
                cursor
                    .record_serviced_miss(&segment, recovery_miss(second_miss))
                    .unwrap();
                injected_second_miss = true;
                assert_eq!(committed_position, POSITION as usize);
                continue;
            }
            if cursor.record_clean_segment(&segment, NUM_LAYERS).unwrap() {
                committed_position += 1;
                break;
            }
            assert_eq!(committed_position, POSITION as usize);
        }

        assert_eq!(&recovered[..=first_miss], prior_rows.as_slice());
        assert_eq!(recovered, warm);
        assert_eq!(committed_position, POSITION as usize + 1);
    }

    #[test]
    fn tiny_cold_recovery_fixture_matches_warm_generated_token_sequence() {
        const NUM_LAYERS: usize = 6;
        const MISS_LAYER: usize = 2;

        fn layer_step(hidden: u32, layer: usize) -> u32 {
            hidden.wrapping_mul(33).wrapping_add(layer as u32 + 1) % 1_009
        }

        fn generate(cold_first_token: bool) -> (Vec<u32>, GpuNativeRecoverySnapshot) {
            let mut input = 7u32;
            let mut output = Vec::new();
            let counters = GpuNativeRecoveryCounters::default();
            for position in 0..3 {
                let mut hidden = input.wrapping_add(position);
                if cold_first_token && position == 0 {
                    for layer in 0..MISS_LAYER {
                        hidden = layer_step(hidden, layer);
                    }
                    let checkpoint = hidden;
                    let mut cursor = GpuNativeRecoveryCursor::after_serviced_miss(
                        NUM_LAYERS,
                        recovery_miss(MISS_LAYER),
                    )
                    .unwrap();
                    loop {
                        let segment = cursor.plan(NUM_LAYERS).unwrap();
                        if matches!(
                            segment.attempt_start,
                            GpuNativeAttemptStart::ResumeExpert { .. }
                        ) {
                            hidden = checkpoint;
                            hidden = layer_step(hidden, MISS_LAYER);
                            counters.resume_attempts.fetch_add(1, Ordering::Relaxed);
                            counters.checkpoint_restores.fetch_add(1, Ordering::Relaxed);
                        }
                        counters.recovery_segments.fetch_add(1, Ordering::Relaxed);
                        for layer in segment.ordinary_layers.clone() {
                            hidden = layer_step(hidden, layer);
                        }
                        if cursor.record_clean_segment(&segment, NUM_LAYERS).unwrap() {
                            break;
                        }
                    }
                } else {
                    for layer in 0..NUM_LAYERS {
                        hidden = layer_step(hidden, layer);
                    }
                }
                input = hidden;
                output.push(hidden);
            }
            (output, counters.snapshot())
        }

        let (warm_tokens, warm_recovery) = generate(false);
        let (cold_tokens, cold_recovery) = generate(true);
        assert_eq!(cold_tokens, warm_tokens);
        assert_eq!(warm_recovery.resume_attempts, 0);
        assert!(cold_recovery.resume_attempts > 0);
        assert_eq!(cold_recovery.full_token_replay_attempts, 0);
    }

    #[test]
    fn recovery_route_assembly_overwrites_only_attempted_layers() {
        let warm_routes = vec![vec![3, 1], vec![2, 0], vec![1, 3], vec![0, 2]];
        let mut assembled = warm_routes.clone();
        assembled[2] = vec![99, 99];
        assembled[3] = vec![88, 88];
        let attempted = 1..3;
        for layer in attempted.clone() {
            assembled[layer] = warm_routes[layer].clone();
        }
        assert_eq!(assembled[0], warm_routes[0]);
        assert_eq!(assembled[1], warm_routes[1]);
        assert_eq!(assembled[2], warm_routes[2]);
        assert_ne!(assembled[3], warm_routes[3]);
        assembled[3] = warm_routes[3].clone();
        assert_eq!(assembled, warm_routes);
    }

    #[test]
    fn recovery_counter_snapshot_keeps_full_replay_zero() {
        let counters = GpuNativeRecoveryCounters::default();
        counters.resume_attempts.fetch_add(2, Ordering::Relaxed);
        counters.recovery_segments.fetch_add(2, Ordering::Relaxed);
        counters.checkpoint_captures.fetch_add(3, Ordering::Relaxed);
        counters.checkpoint_restores.fetch_add(2, Ordering::Relaxed);
        counters.layers_encoded.fetch_add(4, Ordering::Relaxed);
        counters
            .attention_layers_reexecuted
            .fetch_add(1, Ordering::Relaxed);
        counters
            .expert_layers_reexecuted
            .fetch_add(3, Ordering::Relaxed);
        let snapshot = counters.snapshot();
        assert_eq!(snapshot.resume_attempts, 2);
        assert_eq!(snapshot.recovery_segments, 2);
        assert_eq!(snapshot.checkpoint_restores, 2);
        assert_eq!(snapshot.full_token_replay_attempts, 0);
        assert_eq!(snapshot.attention_layers_reexecuted, 1);
        assert_eq!(snapshot.expert_layers_reexecuted, 3);
    }

    #[test]
    fn greedy_argmax_reference_semantics() {
        let reference_argmax = |logits: &[f32]| -> Result<u32, &'static str> {
            if logits.is_empty() {
                return Err("empty logits");
            }
            let mut best_val = -f32::MAX;
            let mut best_idx = None;
            for (idx, &val) in logits.iter().enumerate() {
                if !val.is_finite() {
                    return Err("non-finite logit");
                }
                if best_idx.is_none() || val > best_val {
                    best_val = val;
                    best_idx = Some(idx as u32);
                }
            }
            best_idx.ok_or("no candidate")
        };

        // Unique max
        assert_eq!(reference_argmax(&[1.0, 5.0, 2.0, 3.0]).unwrap(), 1);

        // Tied max selects lower token id
        assert_eq!(reference_argmax(&[1.0, 5.0, 2.0, 5.0, 3.0]).unwrap(), 1);

        // All negative finite logits
        assert_eq!(reference_argmax(&[-10.0, -2.0, -5.0]).unwrap(), 1);

        // Non-finite logits fail
        assert!(reference_argmax(&[1.0, f32::NAN, 2.0]).is_err());
        assert!(reference_argmax(&[1.0, f32::INFINITY, 2.0]).is_err());
        assert!(reference_argmax(&[1.0, f32::NEG_INFINITY, 2.0]).is_err());
    }

    #[test]
    fn fatal_status_mask_and_retry_clear_invariants() {
        assert_eq!(
            GPU_NATIVE_STATUS_FATAL_MASK & GPU_NATIVE_STATUS_RETRYABLE_MASK,
            0
        );
        assert_eq!(
            crate::backend::gpu_native::status_after_retryable_clear(
                GPU_NATIVE_STATUS_RETRYABLE_MASK
            ),
            0
        );
        assert_eq!(
            crate::backend::gpu_native::status_after_retryable_clear(
                GPU_NATIVE_STATUS_FATAL_MASK | GPU_NATIVE_STATUS_RETRYABLE_MASK
            ),
            GPU_NATIVE_STATUS_FATAL_MASK
        );

        let unknown = 1u32 << 31;
        assert_eq!(
            crate::backend::gpu_native::status_after_retryable_clear(
                GPU_NATIVE_STATUS_FATAL_MASK | GPU_NATIVE_STATUS_RETRYABLE_MASK | unknown
            ),
            GPU_NATIVE_STATUS_FATAL_MASK | unknown
        );
        assert!(matches!(
            classify_gpu_native_status(
                GPU_NATIVE_STATUS_FATAL_MASK | GPU_NATIVE_STATUS_RETRYABLE_MASK,
                Some(4)
            ),
            Err(GpuNativeTokenLoopError::FatalNumericalFailure {
                layer_index: Some(4),
                ..
            })
        ));
        assert!(matches!(
            classify_gpu_native_status(unknown, Some(5)),
            Err(GpuNativeTokenLoopError::UnknownStatusBits {
                layer_index: Some(5),
                unknown_bits,
                ..
            }) if unknown_bits == unknown
        ));
        assert_eq!(
            classify_gpu_native_status(GPU_NATIVE_STATUS_RETRYABLE_MASK, Some(6)),
            Ok(GpuNativeStatusDisposition::RetryableResidencyMiss)
        );
    }

    #[test]
    fn gpu_native_token_state_hidden_and_residual_have_copy_src_and_no_map() {
        let usage = crate::backend::gpu_native::GpuNativeTokenStateLayout::tensor_usage();
        assert!(usage.contains(wgpu::BufferUsages::STORAGE));
        assert!(usage.contains(wgpu::BufferUsages::COPY_DST));
        assert!(usage.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!usage.contains(wgpu::BufferUsages::MAP_READ));
        assert!(!usage.contains(wgpu::BufferUsages::MAP_WRITE));
    }

    #[test]
    fn buffer_usage_boundary_isolation_contract() {
        use crate::backend::gpu_native::{
            GpuNativeRouterScratchLayout, GpuNativeScratchLayout, GpuNativeTokenStateLayout,
        };

        // 1. Generic scratch (hidden, residual, logits, attention/expert scratch):
        // STORAGE | COPY_DST | COPY_SRC, strictly no MAP_READ, no MAP_WRITE
        let generic_scratch = GpuNativeScratchLayout::usage();
        assert!(generic_scratch.contains(wgpu::BufferUsages::STORAGE));
        assert!(generic_scratch.contains(wgpu::BufferUsages::COPY_DST));
        assert!(generic_scratch.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!generic_scratch
            .intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));

        // 2. Specialized boundary result scratch (sampled_token_buf):
        // STORAGE | COPY_SRC, strictly no MAP_READ, no MAP_WRITE
        let boundary_scratch = GpuNativeScratchLayout::boundary_result_usage();
        assert!(boundary_scratch.contains(wgpu::BufferUsages::STORAGE));
        assert!(boundary_scratch.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!boundary_scratch.contains(wgpu::BufferUsages::COPY_DST));
        assert!(!boundary_scratch
            .intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));

        // 3. Status buffer:
        // STORAGE | COPY_DST | COPY_SRC, strictly no MAP_READ, no MAP_WRITE
        let status = GpuNativeTokenStateLayout::status_usage();
        assert!(status.contains(wgpu::BufferUsages::STORAGE));
        assert!(status.contains(wgpu::BufferUsages::COPY_DST));
        assert!(status.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!status.intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));

        // 4. Router selected ids/weights are checkpoint-restorable:
        // STORAGE | COPY_SRC | COPY_DST, strictly no MAP_READ, no MAP_WRITE
        let router_result = GpuNativeRouterScratchLayout::result_usage();
        assert!(router_result.contains(wgpu::BufferUsages::STORAGE));
        assert!(router_result.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(router_result.contains(wgpu::BufferUsages::COPY_DST));
        assert!(
            !router_result.intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE)
        );
    }

    #[test]
    fn gpu_native_config_defaults_and_validation() {
        let toml_str = r#"
            [model]
            data_dir = "."
            num_layers = 2
            num_experts = 4
            top_k = 2
            expert_size = 4096
            d_model = 32
            d_ff = 32
            dtype = "q4_0"

            [server]
            bind = "127.0.0.1:8080"
            max_concurrent_requests = 1
            session_ttl_secs = 0

            [storage]
            block_align = 4096
            cache_slots = 16

            [gpu_cache]
            enabled = true
            vram_capacity_mb = 128

            [real_transformer]
            enabled = true
            compute_offload = "gpu"
            gpu_native = true
            strict_weights = true
            max_batch_size = 1
        "#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(cfg.real_transformer.gpu_native);
        assert_eq!(cfg.real_transformer.gpu_native_max_seq_len, 4096);
        assert!(cfg.validate().is_ok());

        let mut invalid_cfg = cfg.clone();
        invalid_cfg.server.max_concurrent_requests = 2;
        assert!(invalid_cfg.validate().is_err());

        let mut invalid_cfg2 = cfg.clone();
        invalid_cfg2.real_transformer.max_batch_size = 2;
        assert!(invalid_cfg2.validate().is_err());
    }

    #[test]
    fn compatibility_validation_rejects_mla_and_oversized() {
        let mut cfg = make_test_qwen3_moe_config();
        cfg.num_experts = 16;
        cfg.top_k = 9;
        let model = make_test_qwen3_moe_model(cfg);
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model),
            Err(GpuNativeModelCompatibilityError::InvalidTopK { .. })
        ));

        let mut cfg2 = make_test_qwen3_moe_config();
        cfg2.num_experts = 129;
        let model2 = make_test_qwen3_moe_model(cfg2);
        assert!(matches!(
            GpuNativeTokenLoop::validate_model_compatibility(&model2),
            Err(GpuNativeModelCompatibilityError::TooManyExperts { .. })
        ));
    }

    #[test]
    fn boundary_report_parser_identifies_failure_modes() {
        let layout = GpuNativeBoundaryReportLayout::try_new(2, 2).unwrap();
        let mut bytes = vec![0u8; 32];

        // Fatal attention
        bytes[0..4].copy_from_slice(&GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE.to_le_bytes());
        let report = layout.parse(&bytes).unwrap();
        assert_eq!(report.first_failure_layer(), Some(0));
        assert_eq!(
            report.layer_statuses[0] & GPU_NATIVE_STATUS_FATAL_MASK,
            GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE
        );

        // Fatal router
        bytes[0..4].copy_from_slice(&0u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE.to_le_bytes());
        let report2 = layout.parse(&bytes).unwrap();
        assert_eq!(report2.first_failure_layer(), Some(1));
        assert_eq!(
            report2.layer_statuses[1] & GPU_NATIVE_STATUS_FATAL_MASK,
            GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE
        );

        // Fatal expert
        bytes[4..8].copy_from_slice(&GPU_NATIVE_STATUS_EXPERT_NUMERICAL_FAILURE.to_le_bytes());
        let report3 = layout.parse(&bytes).unwrap();
        assert_eq!(report3.first_failure_layer(), Some(1));
        assert_eq!(
            report3.layer_statuses[1] & GPU_NATIVE_STATUS_FATAL_MASK,
            GPU_NATIVE_STATUS_EXPERT_NUMERICAL_FAILURE
        );

        // Fatal LM head
        bytes[4..8].copy_from_slice(&0u32.to_le_bytes());
        bytes[24..28].copy_from_slice(&GPU_NATIVE_STATUS_LM_HEAD_NUMERICAL_FAILURE.to_le_bytes());
        let report4 = layout.parse(&bytes).unwrap();
        assert_eq!(report4.first_failure_layer(), None);
        assert_eq!(
            report4.final_status & GPU_NATIVE_STATUS_FATAL_MASK,
            GPU_NATIVE_STATUS_LM_HEAD_NUMERICAL_FAILURE
        );
    }

    #[test]
    fn local_to_global_conversion_and_validation() {
        let num_experts = 4u32;
        let layer_0_locals = vec![1u32, 2];
        let layer_0_globals: Vec<u32> = layer_0_locals
            .iter()
            .map(|&loc| 0 * num_experts + loc)
            .collect();
        assert_eq!(layer_0_globals, vec![1, 2]);

        let layer_1_locals = vec![0u32, 3];
        let layer_1_globals: Vec<u32> = layer_1_locals
            .iter()
            .map(|&loc| 1 * num_experts + loc)
            .collect();
        assert_eq!(layer_1_globals, vec![4, 7]);
    }

    #[test]
    fn context_capacity_accounting_boundary_and_overflow() {
        let max_seq_len = 8usize;

        // 1. P + C - 1 == max_seq_len -> accepted by preflight
        // Prompt of 4 tokens + 5 completion tokens -> 4 + 5 - 1 = 8 evaluations (positions 0..=7)
        let req_len = GpuNativeTokenLoop::calculate_required_context_len(0, 4, 5);
        assert_eq!(req_len, Some(8));
        assert!(req_len.unwrap() <= max_seq_len);

        // 2. P + C - 1 > max_seq_len -> rejected
        // Prompt of 4 tokens + 6 completion tokens -> 4 + 6 - 1 = 9 evaluations
        let req_len_overflow = GpuNativeTokenLoop::calculate_required_context_len(0, 4, 6);
        assert_eq!(req_len_overflow, Some(9));
        assert!(req_len_overflow.unwrap() > max_seq_len);

        // 3. Arithmetic overflow -> returns None (fail-closed)
        let req_overflow = GpuNativeTokenLoop::calculate_required_context_len(0, usize::MAX, 2);
        assert_eq!(req_overflow, None);
        let req_overflow2 =
            GpuNativeTokenLoop::calculate_required_context_len(usize::MAX - 1, 2, 2);
        assert_eq!(req_overflow2, None);

        // 4. Ordinary shorter request -> unchanged / accepted
        let req_len_short = GpuNativeTokenLoop::calculate_required_context_len(0, 2, 2);
        assert_eq!(req_len_short, Some(3));
        assert!(req_len_short.unwrap() <= max_seq_len);

        // 5. max_tokens == 0 -> zero completion tokens, consumes zero additional capacity
        let req_zero = GpuNativeTokenLoop::calculate_required_context_len(0, 5, 0);
        assert_eq!(req_zero, Some(0));

        // 6. Non-zero starting committed position
        let req_with_offset = GpuNativeTokenLoop::calculate_required_context_len(2, 3, 4);
        assert_eq!(req_with_offset, Some(8)); // 2 + 3 + 4 - 1 = 8
    }

    #[test]
    fn position_mismatch_error_behavior() {
        let err = GpuNativeTokenLoopError::PositionMismatch {
            requested_position: 3,
            committed_position: 2,
        };
        let msg = err.to_string();
        assert!(msg.contains("position mismatch"));
        assert!(msg.contains("requested position 3"));
        assert!(msg.contains("committed position 2"));
    }

    #[test]
    fn token_loop_counters_snapshot_and_accounting_semantics() {
        // Unit-tests the internal snapshotting, atomic counter representations,
        // and stage-by-stage accounting invariants of GpuNativeTokenLoopCounters.
        // (End-to-end WGPU hardware execution is validated under the ignored L4 test fixture).
        let counters = GpuNativeTokenLoopCounters::default();
        let snap0 = counters.snapshot();
        assert_eq!(snap0.token_attempts, 0);
        assert_eq!(snap0.tokens_completed, 0);
        assert_eq!(snap0.warm_tokens_completed, 0);
        assert_eq!(snap0.queue_submissions, 0);
        assert_eq!(snap0.boundary_maps, 0);
        assert_eq!(snap0.boundary_readbacks, 0);
        assert_eq!(snap0.replay_attempts, 0);

        // In GpuNativeTokenLoop::execute_token_attempt, token_attempts is incremented
        // when a valid attempt starts (after position validation).
        counters.token_attempts.fetch_add(1, Ordering::Relaxed);
        let snap1 = counters.snapshot();
        assert_eq!(snap1.token_attempts, 1);
        assert_eq!(snap1.queue_submissions, 0);
        assert_eq!(snap1.boundary_maps, 0);
        assert_eq!(snap1.boundary_readbacks, 0);

        // When queue.submit actually occurs:
        counters.queue_submissions.fetch_add(1, Ordering::Relaxed);
        let snap2 = counters.snapshot();
        assert_eq!(snap2.queue_submissions, 1);
        assert_eq!(snap2.boundary_maps, 0);
        assert_eq!(snap2.boundary_readbacks, 0);

        // When map_async actually initiates:
        counters.boundary_maps.fetch_add(1, Ordering::Relaxed);
        let snap3 = counters.snapshot();
        assert_eq!(snap3.boundary_maps, 1);
        assert_eq!(snap3.boundary_readbacks, 0);

        // When map succeeds and report is parsed:
        counters.boundary_readbacks.fetch_add(1, Ordering::Relaxed);
        let snap4 = counters.snapshot();
        assert_eq!(snap4.boundary_readbacks, 1);

        // Replay attempt increments replay_attempts and token_attempts
        counters.token_attempts.fetch_add(1, Ordering::Relaxed);
        counters.replay_attempts.fetch_add(1, Ordering::Relaxed);
        let snap5 = counters.snapshot();
        assert_eq!(snap5.token_attempts, 2);
        assert_eq!(snap5.replay_attempts, 1);
        assert_eq!(snap5.queue_submissions, 1);
    }

    #[test]
    fn disabled_mode_preserves_legacy_state() {
        let toml_str = r#"
            [model]
            data_dir = "."
            num_layers = 2
            num_experts = 4
            top_k = 2
            expert_size = 4096
            d_model = 32
            d_ff = 32
            dtype = "q4_0"

            [server]
            bind = "127.0.0.1:8080"
            max_concurrent_requests = 4
            session_ttl_secs = 0

            [storage]
            block_align = 4096
            cache_slots = 16

            [gpu_cache]
            enabled = false
            vram_capacity_mb = 0

            [real_transformer]
            enabled = false
            gpu_native = false
        "#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert!(!cfg.real_transformer.gpu_native);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn multi_layer_ram_cache_seeding_and_lookup() {
        use crate::buffer_pool::BufferPool;
        use crate::expert_cache::ExpertResident;
        use crate::multi_layer_cache::MultiLayerExpertCache;

        const NUM_LAYERS: usize = 2;
        const NUM_EXPERTS: u32 = 4;
        const EXPERT_SIZE: usize = 64;

        let ram_cache = Arc::new(MultiLayerExpertCache::with_uniform_capacity(
            NUM_LAYERS,
            16,
            NUM_EXPERTS,
        ));
        let pool = BufferPool::new(16, EXPERT_SIZE, 4);

        for layer_idx in 0..NUM_LAYERS {
            for local_id in 0..NUM_EXPERTS {
                let global_id = layer_idx as u32 * NUM_EXPERTS + local_id;
                let mut buffer = pool.try_acquire().expect("buffer slot");
                buffer.as_mut_slice().fill((global_id + 1) as u8);
                let resident = Arc::new(ExpertResident::new_with_block_align(global_id, buffer, 4));
                assert!(
                    ram_cache.insert(resident).is_ok(),
                    "RAM cache insert must succeed for global_id {global_id}"
                );
            }
        }

        for layer_idx in 0..NUM_LAYERS {
            for local_id in 0..NUM_EXPERTS {
                let global_id = layer_idx as u32 * NUM_EXPERTS + local_id;
                assert!(ram_cache.contains(global_id));
                let resident = ram_cache
                    .get(global_id)
                    .expect("must retrieve seeded resident");
                assert_eq!(resident.id, global_id);
                assert_eq!(resident.data()[0], (global_id + 1) as u8);
            }
        }
    }

    struct TempDir {
        path: std::path::PathBuf,
    }

    impl TempDir {
        fn new(tag: &str) -> Self {
            static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("mer-test-{tag}-{id}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    struct LiveL4Harness {
        _temp_dir: TempDir,
        engine: Arc<Engine>,
        token_loop: Arc<GpuNativeTokenLoop>,
        ram_cache: Arc<crate::multi_layer_cache::MultiLayerExpertCache>,
        gpu_cache: Arc<crate::expert_cache::GpuExpertCache>,
        residency_manager: Arc<GpuNativeTieredResidencyManager>,
    }

    const LIVE_L4_D_MODEL: usize = 32;
    const LIVE_L4_D_FF: usize = 32;
    const LIVE_L4_NUM_EXPERTS: usize = 4;
    const LIVE_L4_TOP_K: usize = 2;
    const LIVE_L4_NUM_LAYERS: usize = 2;
    const LIVE_L4_VOCAB_SIZE: usize = 32;
    const LIVE_L4_MAX_SEQ_LEN: usize = 8;

    fn setup_live_l4_harness(tag: &str) -> LiveL4Harness {
        use crate::backend::{resolve_execution_context_for_gpu_native, GpuBackendGeometry};
        use crate::buffer_pool::BufferPool;
        use crate::expert_cache::{ExpertResident, GpuExpertCache, GpuResident};

        let geometry = GpuNativeQ4ExpertGeometry::try_new(
            LIVE_L4_D_MODEL,
            LIVE_L4_D_FF,
            LIVE_L4_NUM_EXPERTS,
            LIVE_L4_TOP_K,
        )
        .unwrap();
        let payload_template =
            crate::backend::gpu_native::tests::q4_uniform_expert(geometry, 0.01, 0.02, 0.005);
        let payload_bytes = payload_template.len();

        let gpu_cache = Arc::new(GpuExpertCache::new(payload_bytes * 16, 0.0, u64::MAX));

        let execution = resolve_execution_context_for_gpu_native(
            GpuBackendGeometry {
                num_layers: LIVE_L4_NUM_LAYERS,
                max_seq_len: LIVE_L4_MAX_SEQ_LEN,
                num_heads: 2,
                num_kv_heads: 2,
                head_dim: 16,
                v_head_dim: 16,
                q4_truncation_tolerance: 0,
            },
            gpu_cache.clone(),
        )
        .expect("L4 must construct the authoritative production GPU backend");

        let executor = Arc::new(
            execution
                .create_gpu_native_executor_context(LIVE_L4_D_MODEL)
                .expect("GPU-native executor must retain the authoritative backend"),
        );
        assert_eq!(executor.device_identity().vendor_id, 0x10de);
        assert!(
            executor.device_identity().name.contains("L4"),
            "ignored full token loop test must run only on an NVIDIA L4, got {}",
            executor.device_identity().name
        );

        let total_budget =
            (payload_bytes as u64) * (LIVE_L4_TOP_K as u64) * (LIVE_L4_NUM_LAYERS as u64) * 2;
        let residency_manager = Arc::new(
            GpuNativeTieredResidencyManager::try_new(
                executor.clone(),
                gpu_cache.clone(),
                LIVE_L4_NUM_LAYERS,
                geometry,
                total_budget,
            )
            .unwrap(),
        );

        let temp_dir = TempDir::new(tag);
        let storage = Arc::new(
            crate::io_provider::NvmeStorage::new(crate::io_provider::StorageConfig {
                base_path: temp_dir.path.clone(),
                expert_size: payload_bytes,
                block_align: 4,
                use_direct_io: false,
                num_experts_per_layer: Some(LIVE_L4_NUM_EXPERTS as u32),
            })
            .unwrap(),
        );

        let router = crate::gating::Router::Markov(Arc::new(crate::router::TopKRouter::new(
            (LIVE_L4_NUM_EXPERTS * LIVE_L4_NUM_LAYERS) as u32,
            LIVE_L4_TOP_K,
            0xC0FFEE,
        )));
        let predictor = Arc::new(crate::router::PredictiveLoader::new(
            (LIVE_L4_NUM_EXPERTS * LIVE_L4_NUM_LAYERS) as u32,
            LIVE_L4_TOP_K,
            0.0,
            42,
        ));

        let ram_cache = Arc::new(
            crate::multi_layer_cache::MultiLayerExpertCache::with_uniform_capacity(
                LIVE_L4_NUM_LAYERS,
                16,
                LIVE_L4_NUM_EXPERTS as u32,
            ),
        );

        let mut engine_builder = Engine::with_options_and_execution_context(
            ram_cache.clone(),
            BufferPool::new(16, payload_bytes, 4),
            storage,
            router,
            predictor,
            crate::engine::ModelShape {
                d_model: LIVE_L4_D_MODEL,
                d_ff: LIVE_L4_D_FF,
                hidden_seed: 0xC0FFEE,
            },
            crate::engine::EngineOptions {
                io_only: false,
                dtype: crate::inference::WeightDtype::Q4_0,
                partial_load_fraction: 1.0,
                pin_after_observations: 0,
                use_qmm_for_q4: true,
                expert_execution_policy: crate::engine::ExpertExecutionPolicy::Auto,
                max_concurrent_prefetches: 64,
                max_fetch_yields: 128,
                prefetch_governor: false,
                prefetch_precision_floor: 0.0,
                prefetch_contention_weight: 0.0,
                cost_aware_eviction: false,
                pregate_enabled: false,
                collect_route_profile: false,
                policy: crate::inference::RealInferencePolicy::default(),
            },
            execution.clone(),
        );
        engine_builder
            .install_gpu_native_residency_manager(residency_manager.clone())
            .unwrap();
        let engine = Arc::new(engine_builder);

        let cfg = RealModelConfig {
            d_model: LIVE_L4_D_MODEL,
            d_ff: LIVE_L4_D_FF,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 16,
            vocab_size: LIVE_L4_VOCAB_SIZE,
            num_layers: LIVE_L4_NUM_LAYERS,
            num_experts: LIVE_L4_NUM_EXPERTS,
            top_k: LIVE_L4_TOP_K,
            rope_base: 10_000.0,
            rms_eps: 1e-5,
            window_size: None,
            architecture: Architecture::Qwen3Moe,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        };
        let model = make_test_qwen3_moe_model(cfg);

        let token_loop = GpuNativeTokenLoop::try_new(
            executor.clone(),
            residency_manager.clone(),
            &model,
            LIVE_L4_MAX_SEQ_LEN,
        )
        .unwrap();

        let pool = BufferPool::new(16, payload_bytes, 4);
        for layer_idx in 0..LIVE_L4_NUM_LAYERS {
            for local_id in 0..LIVE_L4_NUM_EXPERTS as u32 {
                let global_id = layer_idx as u32 * LIVE_L4_NUM_EXPERTS as u32 + local_id;
                let payload = crate::backend::gpu_native::tests::q4_uniform_expert(
                    geometry,
                    0.01 * (global_id + 1) as f32,
                    0.02,
                    0.005,
                );
                let mut buffer = pool.try_acquire().expect("synthetic RAM slot");
                buffer.as_mut_slice().copy_from_slice(&payload);
                let resident = Arc::new(ExpertResident::new_with_block_align(global_id, buffer, 4));
                assert!(
                    ram_cache.insert(resident).is_ok(),
                    "synthetic RAM expert must be admitted"
                );
                gpu_cache
                    .demand_admit_lru(Arc::new(GpuResident::new_with_dtype(
                        global_id,
                        payload.clone(),
                        crate::inference::WeightDtype::Q4_0,
                    )))
                    .unwrap();
                let admission = gpu_cache.current_admission(global_id).unwrap();
                let _ = admission;
            }
        }

        LiveL4Harness {
            _temp_dir: temp_dir,
            engine,
            token_loop,
            ram_cache,
            gpu_cache,
            residency_manager,
        }
    }

    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_full_token_loop_retry() {
        let harness = setup_live_l4_harness("full_token_loop");
        let mut request_state = harness.token_loop.create_request_state().unwrap();

        // Precondition assertions: RAM warm, logical GPU admissions present, physical VRAM cold.
        for layer_idx in 0..LIVE_L4_NUM_LAYERS {
            for local_id in 0..LIVE_L4_NUM_EXPERTS as u32 {
                let global_id = layer_idx as u32 * LIVE_L4_NUM_EXPERTS as u32 + local_id;
                assert!(
                    harness.ram_cache.contains(global_id),
                    "RAM cache must contain global_id {global_id}"
                );
                assert!(
                    harness.gpu_cache.current_admission(global_id).is_some(),
                    "gpu_cache must contain logical admission for global_id {global_id}"
                );
                assert_eq!(
                    harness
                        .residency_manager
                        .has_current_for_demand(global_id)
                        .unwrap(),
                    false,
                    "physical residency manager must initially be cold for global_id {global_id}"
                );
            }
        }

        let first_token = pollster::block_on(harness.token_loop.step_token(
            &harness.engine,
            &mut request_state,
            1,
            0,
            true,
        ))
        .unwrap()
        .expect("cold token 0 must succeed after retries");

        assert_eq!(request_state.committed_position, 1);
        let snap1 = harness.token_loop.snapshot();
        assert_eq!(snap1.tokens_completed, 1);
        assert_eq!(snap1.token_attempts, 3);
        assert_eq!(snap1.residency_miss_attempts, 2);
        assert_eq!(snap1.replay_attempts, 0);
        assert_eq!(snap1.residency_services, 2);
        assert_eq!(snap1.fatal_failures, 0);
        let recovery1 = harness.token_loop.recovery_snapshot();
        assert_eq!(recovery1.resume_attempts, 2);
        assert_eq!(recovery1.recovery_segments, 2);
        assert_eq!(recovery1.checkpoint_restores, 2);
        assert_eq!(recovery1.full_token_replay_attempts, 0);

        let _second_token = pollster::block_on(harness.token_loop.step_token(
            &harness.engine,
            &mut request_state,
            first_token,
            1,
            true,
        ))
        .unwrap()
        .expect("warm token 1 must succeed directly");

        assert_eq!(request_state.committed_position, 2);
        let snap2 = harness.token_loop.snapshot();
        assert_eq!(snap2.tokens_completed, 2);
        assert_eq!(snap2.warm_tokens_completed, 1);
        assert_eq!(snap2.token_attempts, 4);
        assert_eq!(snap2.queue_submissions, 4);
        assert_eq!(snap2.boundary_maps, 4);
        assert_eq!(snap2.boundary_readbacks, 4);
        assert_eq!(snap2.residency_services, 2);
        assert_eq!(snap2.fatal_failures, 0);
        let recovery2 = harness.token_loop.recovery_snapshot();
        assert_eq!(recovery2.full_token_replay_attempts, 0);

        assert_eq!(
            harness.engine.report().bytes_read,
            0,
            "pure RAM->VRAM qualification must not fall back to storage reads"
        );
    }

    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_diagnostic_smoke() {
        let harness = setup_live_l4_harness("diag_smoke");
        let mut request_state = harness.token_loop.create_request_state().unwrap();

        let trace_layout = crate::gpu_native_diagnostics::GpuNativeDiagnosticTraceLayout::try_new(
            LIVE_L4_NUM_LAYERS,
            LIVE_L4_D_MODEL,
            LIVE_L4_TOP_K,
            LIVE_L4_VOCAB_SIZE,
        )
        .unwrap();

        let staging_buffer = harness
            .token_loop
            .create_diagnostic_staging_buffer(&trace_layout)
            .unwrap();

        let (trace, attempts) = pollster::block_on(harness.token_loop.step_token_diagnostic(
            &harness.engine,
            &mut request_state,
            1,
            0,
            true,
            &trace_layout,
            &staging_buffer,
        ))
        .expect("diagnostic step on L4 must succeed");

        assert_eq!(trace.embedding.len(), LIVE_L4_D_MODEL);
        assert_eq!(trace.layer_post_attn.len(), LIVE_L4_NUM_LAYERS);
        assert_eq!(trace.layer_router_input.len(), LIVE_L4_NUM_LAYERS);
        assert_eq!(trace.layer_selected_ids.len(), LIVE_L4_NUM_LAYERS);
        assert_eq!(trace.layer_selected_weights.len(), LIVE_L4_NUM_LAYERS);
        assert_eq!(trace.layer_post_moe.len(), LIVE_L4_NUM_LAYERS);
        assert_eq!(trace.layer_statuses.len(), LIVE_L4_NUM_LAYERS);
        assert_eq!(trace.final_norm.len(), LIVE_L4_D_MODEL);
        assert_eq!(trace.logits.len(), LIVE_L4_VOCAB_SIZE);
        assert!(trace.sampled_token < LIVE_L4_VOCAB_SIZE as u32);
        assert_eq!(trace.final_status, 0);
        assert_eq!(request_state.committed_position, 1);
        assert!(attempts > 0);

        for l in 0..LIVE_L4_NUM_LAYERS {
            assert_eq!(trace.layer_selected_ids[l].len(), LIVE_L4_TOP_K);
            assert_eq!(trace.layer_selected_weights[l].len(), LIVE_L4_TOP_K);
            assert_eq!(trace.layer_statuses[l], 0);
            assert!(trace.layer_post_attn[l].iter().all(|v| v.is_finite()));
            assert!(trace.layer_router_input[l].iter().all(|v| v.is_finite()));
            assert!(trace.layer_selected_weights[l]
                .iter()
                .all(|v| v.is_finite()));
            assert!(trace.layer_post_moe[l].iter().all(|v| v.is_finite()));
        }

        assert!(trace.embedding.iter().all(|v| v.is_finite()));
        assert!(trace.final_norm.iter().all(|v| v.is_finite()));
        assert!(trace.logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_ingest_no_deadlock() {
        let harness = setup_live_l4_harness("ingest_deadlock");
        let mut request_state = harness.token_loop.create_request_state().unwrap();

        let prompt_ids = [1u32, 2u32];
        let completion = pollster::block_on(harness.token_loop.ingest_prompt_and_generate(
            &harness.engine,
            &mut request_state,
            &prompt_ids,
            1,
            &SamplingParams::greedy(),
        ))
        .expect("ingest_prompt_and_generate must succeed without deadlock");

        assert_eq!(completion.len(), 1);
        assert!(completion[0] < LIVE_L4_VOCAB_SIZE as u32);
        assert_eq!(request_state.committed_position, 2);
    }

    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_layer0_attention_diagnostic_smoke() {
        let harness = setup_live_l4_harness("layer0_smoke");
        let mut request_state = harness.token_loop.create_request_state().unwrap();

        let geom = harness.token_loop.model_geometry();
        let q_width = geom.num_heads * geom.head_dim;
        let kv_width = geom.num_kv_heads * geom.head_dim;

        let trace_layout =
            crate::gpu_native_layer0_diagnostics::Layer0AttentionDiagnosticTraceLayout::try_new(
                geom.d_model,
                q_width,
                kv_width,
            )
            .unwrap();

        let staging_buffer = harness
            .token_loop
            .create_layer0_diagnostic_staging_buffer(&trace_layout)
            .unwrap();

        // 1. Position 0
        let trace0 = pollster::block_on(harness.token_loop.step_layer0_attention_diagnostic(
            &harness.engine,
            &mut request_state,
            1,
            0,
            &trace_layout,
            &staging_buffer,
        ))
        .expect("layer-0 diagnostic step 0 must succeed");

        assert_eq!(trace0.embedding.len(), geom.d_model);
        assert_eq!(trace0.attention_pre_norm.len(), geom.d_model);
        assert_eq!(trace0.q_raw.len(), q_width);
        assert_eq!(trace0.k_raw.len(), kv_width);
        assert_eq!(trace0.v_raw.len(), kv_width);
        assert_eq!(trace0.q_after_norm.len(), q_width);
        assert_eq!(trace0.k_after_norm.len(), kv_width);
        assert_eq!(trace0.q_after_rope.len(), q_width);
        assert_eq!(trace0.k_after_rope.len(), kv_width);
        assert_eq!(trace0.attention_context.len(), q_width);
        assert_eq!(trace0.o_projection.len(), geom.d_model);
        assert_eq!(trace0.post_attention_residual.len(), geom.d_model);
        assert_eq!(trace0.status, 0);
        assert_eq!(request_state.committed_position, 1);

        assert!(trace0.embedding.iter().all(|v| v.is_finite()));
        assert!(trace0.attention_pre_norm.iter().all(|v| v.is_finite()));
        assert!(trace0.q_raw.iter().all(|v| v.is_finite()));
        assert!(trace0.k_raw.iter().all(|v| v.is_finite()));
        assert!(trace0.v_raw.iter().all(|v| v.is_finite()));
        assert!(trace0.q_after_norm.iter().all(|v| v.is_finite()));
        assert!(trace0.k_after_norm.iter().all(|v| v.is_finite()));
        assert!(trace0.q_after_rope.iter().all(|v| v.is_finite()));
        assert!(trace0.k_after_rope.iter().all(|v| v.is_finite()));
        assert!(trace0.attention_context.iter().all(|v| v.is_finite()));
        assert!(trace0.o_projection.iter().all(|v| v.is_finite()));
        assert!(trace0.post_attention_residual.iter().all(|v| v.is_finite()));

        let queries_per_kv_head = geom.num_heads / geom.num_kv_heads;
        for query_head in 0..geom.num_heads {
            let kv_head = query_head / queries_per_kv_head;
            for channel in 0..geom.head_dim {
                let context_index = query_head * geom.head_dim + channel;
                let v_index = kv_head * geom.head_dim + channel;
                let context = trace0.attention_context[context_index];
                let expected_v = trace0.v_raw[v_index];
                assert_eq!(
                    context.to_bits(),
                    expected_v.to_bits(),
                    "position-0 context/V identity failed: query_head={query_head}, kv_head={kv_head}, channel={channel}, expected_v={expected_v}, expected_v_bits=0x{:08x}, context={context}, context_bits=0x{:08x}",
                    expected_v.to_bits(),
                    context.to_bits(),
                );
            }
        }

        // 2. Position 1 (persisting KV across positions)
        let trace1 = pollster::block_on(harness.token_loop.step_layer0_attention_diagnostic(
            &harness.engine,
            &mut request_state,
            2,
            1,
            &trace_layout,
            &staging_buffer,
        ))
        .expect("layer-0 diagnostic step 1 must succeed");

        assert_eq!(trace1.status, 0);
        assert_eq!(request_state.committed_position, 2);
        assert!(trace1.post_attention_residual.iter().all(|v| v.is_finite()));
    }
}
