//! Isolated foundations for a future GPU-native token loop.
//!
//! This module is not reachable from current operator-facing execution modes.
//! It owns request-local device state, persistent dense/RMSNorm model weights,
//! and encoder-only embedding, GEMV, RMSNorm, residual, attention-preparation,
//! causal-attention completion, request-local KV, routing, and a mutable,
//! generation-safe banked Q4_0 expert arena on the authoritative WGPU device.
//! Later slices can compose those pieces without inheriting legacy host-shaped
//! upload/readback APIs.

// GPU-native slices remain intentionally unreachable from production token entrypoints.
#![allow(dead_code)]

use super::{create_startup_buffer, BackendBox, GpuDeviceIdentity, GpuStartupAllocationError};
use crate::dense_tensor::{DenseDType, DenseWeight};
use crate::inference::{Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};
use parking_lot::Mutex as ParkingMutex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const GPU_NATIVE_STATUS_BYTES: u64 = std::mem::size_of::<u32>() as u64;
// Request-local device status is latched until a future token-boundary
// scheduler handles it; production encoding never clears these bits. Fatal
// numerical failures retire the request. A residency miss is retryable after
// residency service and must not be classified as a numerical failure.
pub(crate) const GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE: u32 = 1 << 0;
pub(crate) const GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE: u32 = 1 << 1;
pub(crate) const GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS: u32 = 1 << 2;
pub(crate) const GPU_NATIVE_STATUS_EXPERT_NUMERICAL_FAILURE: u32 = 1 << 3;
pub(crate) const GPU_NATIVE_STATUS_LM_HEAD_NUMERICAL_FAILURE: u32 = 1 << 4;
pub(crate) const GPU_NATIVE_STATUS_FATAL_MASK: u32 = GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE
    | GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE
    | GPU_NATIVE_STATUS_EXPERT_NUMERICAL_FAILURE
    | GPU_NATIVE_STATUS_LM_HEAD_NUMERICAL_FAILURE;
pub(crate) const GPU_NATIVE_STATUS_RETRYABLE_MASK: u32 = GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS;

#[inline]
pub(crate) const fn status_after_retryable_clear(status: u32) -> u32 {
    status & !GPU_NATIVE_STATUS_RETRYABLE_MASK
}
const GPU_NATIVE_DENSE_GEMV_SHADER: &str = include_str!("wgpu_shaders/gpu_native_dense_gemv.wgsl");
const GPU_NATIVE_EMBEDDING_SHADER: &str = include_str!("wgpu_shaders/gpu_native_embedding.wgsl");
const GPU_NATIVE_RMSNORM_SHADER: &str = include_str!("wgpu_shaders/gpu_native_rmsnorm.wgsl");
const GPU_NATIVE_ROPE_SHADER: &str = include_str!("wgpu_shaders/gpu_native_rope.wgsl");
const GPU_NATIVE_KV_APPEND_SHADER: &str = include_str!("wgpu_shaders/gpu_native_kv_append.wgsl");
const GPU_NATIVE_ATTENTION_SHADER: &str = include_str!("wgpu_shaders/gpu_native_attention.wgsl");
const GPU_NATIVE_ROUTER_SHADER: &str = include_str!("wgpu_shaders/gpu_native_router.wgsl");
const GPU_NATIVE_Q4_EXPERT_SHADER: &str = include_str!("wgpu_shaders/gpu_native_q4_expert.wgsl");
const GPU_NATIVE_EXPERT_CONTROL_SHADER: &str =
    include_str!("wgpu_shaders/gpu_native_expert_control.wgsl");
const GPU_NATIVE_STATUS_CONTROL_SHADER: &str =
    include_str!("wgpu_shaders/gpu_native_status_control.wgsl");
const GPU_NATIVE_GREEDY_ARGMAX_SHADER: &str =
    include_str!("wgpu_shaders/gpu_native_greedy_argmax.wgsl");
const GPU_NATIVE_WORKGROUP_SIZE: u32 = 64;
const GPU_NATIVE_ATTENTION_WORKGROUP_SIZE: u32 = 32;
const GPU_NATIVE_ROUTER_WORKGROUP_SIZE: u32 = 64;
const GPU_NATIVE_EXPERT_WORKGROUP_SIZE: u32 = 64;
const GPU_NATIVE_ROUTER_WORKGROUP_STORAGE_BYTES: u32 = ((MAX_GPU_NATIVE_ROUTER_EXPERTS
    + GPU_NATIVE_ROUTER_WORKGROUP_SIZE as usize)
    * std::mem::size_of::<f32>()
    + std::mem::size_of::<u32>()) as u32;
pub(crate) const MAX_GPU_NATIVE_ROUTER_EXPERTS: usize = 128;
pub(crate) const MAX_GPU_NATIVE_ROUTER_TOP_K: usize = 8;
pub(crate) const MAX_GPU_NATIVE_EXPERT_BANKS: usize = 4;
const GPU_NATIVE_EXPERT_LOCATION_BANK_SHIFT: u32 = 30;
const GPU_NATIVE_EXPERT_LOCATION_SLOT_MASK: u32 = (1 << 30) - 1;
const GPU_NATIVE_EXPERT_UNMAPPED: u32 = u32::MAX;
const GPU_NATIVE_EXPERT_SLOT_EPOCH_BYTES: usize = std::mem::size_of::<u32>();
const GPU_NATIVE_EXPERT_MAPPING_ENTRY_BYTES: usize = std::mem::size_of::<u32>() * 2;
const GPU_NATIVE_EXPERT_REQUIRED_STORAGE_BUFFERS: u32 = 8;
const GPU_NATIVE_EXPERT_PUSH_CONSTANT_BYTES: u32 = 32;

/// Typed, fail-closed construction failure for the GPU-native bootstrap.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeBootstrapError {
    GpuBackendUnavailable,
    DeviceLost {
        detail: String,
    },
    InvalidDModel,
    StateSizeOverflow {
        d_model: usize,
    },
    InvalidDenseWeightKey,
    InvalidDenseWeightShape {
        rows: usize,
        cols: usize,
    },
    DenseWeightShapeOverflow {
        rows: usize,
        cols: usize,
    },
    DenseWeightDimensionTooLarge {
        rows: usize,
        cols: usize,
    },
    DenseWeightRowExceedsDeviceLimit {
        kind: GpuNativeDenseWeightKind,
        cols: usize,
        required: u64,
        maximum: u64,
    },
    DenseWeightByteLength {
        kind: GpuNativeDenseWeightKind,
        rows: usize,
        cols: usize,
        expected: usize,
        actual: usize,
    },
    DuplicateDenseWeight {
        key: String,
    },
    MissingDenseWeight {
        key: String,
    },
    ForeignDenseWeightHandle,
    StaleDenseWeightHandle {
        key: String,
    },
    DenseWeightKindMismatch {
        key: String,
        expected: GpuNativeDenseWeightKind,
        actual: GpuNativeDenseWeightKind,
    },
    DenseWeightShapeMismatch {
        key: String,
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    InvalidRmsNormWeightWidth {
        width: usize,
    },
    InvalidRmsNormEpsilon {
        epsilon_bits: u32,
    },
    ForeignRmsNormHandle,
    StaleRmsNormHandle {
        key: String,
    },
    RmsNormWeightWidth {
        expected: usize,
        actual: usize,
    },
    InvalidRmsNormGroups {
        groups: usize,
    },
    InvalidRmsNormGroupWidth {
        group_width: usize,
    },
    RmsNormGeometryOverflow {
        groups: usize,
        group_width: usize,
    },
    RmsNormScratchGeometry {
        expected: usize,
        actual: usize,
    },
    InvalidScratchElements,
    ScratchSizeOverflow {
        elements: usize,
    },
    ForeignTokenState,
    ForeignScratch,
    AliasedInputOutput,
    GemvInputLength {
        expected: usize,
        actual: usize,
    },
    GemvOutputLength {
        expected: usize,
        actual: usize,
    },
    InvalidEmbeddingToken {
        token_id: u32,
        vocab_size: usize,
    },
    EmbeddingWidth {
        expected: usize,
        actual: usize,
    },
    ResidualContributionWidth {
        expected: usize,
        actual: usize,
    },
    InvalidRouterExpertCount {
        num_experts: usize,
    },
    InvalidRouterTopK {
        top_k: usize,
        num_experts: usize,
    },
    RouterGeometryOverflow {
        d_model: usize,
        num_experts: usize,
        top_k: usize,
    },
    RouterDModelMismatch {
        expected: usize,
        actual: usize,
    },
    ForeignRouterPlan,
    ForeignRouterScratch,
    RouterScratchGeometry {
        expected: GpuNativeRouterGeometry,
        actual: GpuNativeRouterGeometry,
    },
    RouterLogitsLength {
        expected: usize,
        actual: usize,
    },
    RouterSelectedIdsLength {
        expected: usize,
        actual: usize,
    },
    RouterSelectedWeightsLength {
        expected: usize,
        actual: usize,
    },
    RouterGateShape {
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    RouterWorkgroupUnsupported {
        required: u32,
        max_size_x: u32,
        max_invocations: u32,
    },
    RouterWorkgroupStorageUnsupported {
        required: u32,
        maximum: u32,
    },
    InvalidExpertDff {
        d_ff: usize,
    },
    ExpertQ4GeometryIncompatible {
        d_model: usize,
        d_ff: usize,
        block_elements: usize,
    },
    ExpertGeometryOverflow {
        d_model: usize,
        d_ff: usize,
        num_experts: usize,
        top_k: usize,
    },
    ExpertDModelMismatch {
        expected: usize,
        actual: usize,
    },
    ExpertRouterGeometryMismatch {
        router: GpuNativeRouterGeometry,
        expert: GpuNativeQ4ExpertGeometry,
    },
    ForeignExpertArena,
    ForeignExpertScratch,
    ExpertLayerMismatch {
        router_layer: usize,
        expert_layer: usize,
    },
    InvalidExpertSlotCapacity {
        capacity: usize,
    },
    ExpertArenaBudgetTooSmall {
        requested_bytes: u64,
        minimum_bytes: u64,
    },
    ExpertArenaBudgetOverflow,
    ExpertArenaBankLimit {
        required_banks: usize,
        maximum_banks: usize,
    },
    ExpertArenaBankCapacity {
        bank: usize,
        required_bytes: u64,
        maximum_bytes: u64,
    },
    ExpertStorageBuffersUnsupported {
        required: u32,
        maximum: u32,
    },
    ExpertPushConstantsUnsupported {
        required: u32,
        maximum: u32,
    },
    ExpertWorkgroupUnsupported {
        required: u32,
        max_size_x: u32,
        max_invocations: u32,
    },
    DuplicateExpertLogicalId {
        logical_id: u32,
    },
    ExpertLogicalIdOutOfRange {
        logical_id: u32,
        num_experts: usize,
    },
    DuplicateExpertPhysicalLocation {
        packed: u32,
    },
    ExpertBankOutOfRange {
        bank: u32,
        active_banks: usize,
    },
    ExpertSlotOutOfRange {
        bank: u32,
        slot: u32,
        capacity: usize,
    },
    ExpertLocationUnrepresentable {
        bank: u32,
        slot: u32,
    },
    InvalidExpertLogicalGeneration {
        logical_generation: u64,
    },
    InvalidExpertSlotEpoch {
        slot_epoch: u32,
    },
    ExpertInstallTicketExhausted,
    ExpertSlotEpochExhausted {
        bank: u32,
        slot: u32,
    },
    ExpertInstallReservationLost,
    ExpertPayloadTooShort {
        logical_id: u32,
        expected: usize,
        actual: usize,
    },
    ExpertPayloadTrailingBytes {
        logical_id: u32,
        maximum: usize,
        actual: usize,
    },
    ExpertPayloadNonZeroPadding {
        logical_id: u32,
    },
    ExpertResidentCountExceedsCapacity {
        residents: usize,
        capacity: usize,
    },
    ExpertScratchGeometry {
        expected: GpuNativeQ4ExpertGeometry,
        actual: GpuNativeQ4ExpertGeometry,
    },
    InvalidAttentionHeadCount {
        tensor: GpuNativeAttentionTensor,
        heads: usize,
    },
    InvalidAttentionHeadGeometry {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    },
    AttentionGeometryOverflow {
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    },
    AttentionDModelMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidRopeDimension {
        rope_dim: usize,
        head_dim: usize,
    },
    OddRopeDimension {
        rope_dim: usize,
    },
    InvalidRopeBase {
        base_bits: u32,
    },
    InvalidRopeInverseFrequency {
        index: usize,
        value_bits: u32,
    },
    InvalidRopeAttentionFactor {
        factor_bits: u32,
    },
    ForeignRopeHandle,
    StaleRopeHandle {
        key: String,
    },
    RopeDimensionMismatch {
        expected: usize,
        actual: usize,
    },
    RopeParameterWidth {
        expected: usize,
        actual: usize,
    },
    ForeignAttentionPlan,
    AttentionPlanLayerOutOfRange {
        layer_index: usize,
        num_layers: usize,
    },
    ForeignAttentionScratch,
    AttentionScratchGeometry {
        expected: GpuNativeAttentionGeometry,
        actual: GpuNativeAttentionGeometry,
    },
    AttentionScratchWidth {
        tensor: GpuNativeAttentionTensor,
        expected: usize,
        actual: usize,
    },
    AttentionProjectionShape {
        tensor: GpuNativeAttentionTensor,
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    AttentionNormWidth {
        tensor: GpuNativeAttentionTensor,
        expected: usize,
        actual: usize,
    },
    InvalidKvLayerCount,
    InvalidKvCapacity,
    InvalidKvWidth,
    KvCapacityOverflow {
        num_layers: usize,
        max_seq_len: usize,
        kv_width: usize,
    },
    KvBufferLimit {
        required: u64,
        max_buffer_size: u64,
        max_storage_binding_size: u64,
    },
    ForeignKvState,
    InvalidKvLayer {
        layer: usize,
        num_layers: usize,
    },
    InvalidKvPosition {
        position: usize,
        max_seq_len: usize,
    },
    AttentionSequenceLengthOverflow {
        position: usize,
    },
    InvalidAttentionSequenceLength {
        seq_len: usize,
        max_seq_len: usize,
    },
    KvWidth {
        expected: usize,
        actual: usize,
    },
    DispatchGeometryUnsupported {
        workgroups: u64,
        maximum: u32,
    },
    AttentionWorkgroupUnsupported {
        required: u32,
        max_size_x: u32,
        max_invocations: u32,
    },
    Allocation(GpuStartupAllocationError),
}

impl fmt::Display for GpuNativeBootstrapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GpuBackendUnavailable => write!(
                f,
                "GPU-native execution requires the authoritative production GPU backend"
            ),
            Self::DeviceLost { detail } => {
                write!(
                    f,
                    "GPU device was lost before GPU-native bootstrap: {detail}"
                )
            }
            Self::InvalidDModel => write!(f, "GPU-native d_model must be non-zero"),
            Self::StateSizeOverflow { d_model } => write!(
                f,
                "GPU-native token-state size overflows for d_model={d_model}"
            ),
            Self::InvalidDenseWeightKey => {
                write!(f, "GPU-native dense weight key must be non-empty")
            }
            Self::InvalidDenseWeightShape { rows, cols } => write!(
                f,
                "GPU-native dense weight shape [{rows}, {cols}] must be non-empty"
            ),
            Self::DenseWeightShapeOverflow { rows, cols } => write!(
                f,
                "GPU-native dense weight shape [{rows}, {cols}] overflows"
            ),
            Self::DenseWeightDimensionTooLarge { rows, cols } => write!(
                f,
                "GPU-native dense weight shape [{rows}, {cols}] exceeds u32 shader geometry"
            ),
            Self::DenseWeightRowExceedsDeviceLimit {
                kind,
                cols,
                required,
                maximum,
            } => write!(
                f,
                "GPU-native {kind:?} dense weight row with {cols} columns requires {required} bytes, exceeding the physical chunk limit {maximum}"
            ),
            Self::DenseWeightByteLength {
                kind,
                rows,
                cols,
                expected,
                actual,
            } => write!(
                f,
                "GPU-native {kind:?} dense weight [{rows}, {cols}] has {actual} bytes, expected {expected}"
            ),
            Self::DuplicateDenseWeight { key } => {
                write!(f, "GPU-native dense weight {key:?} is already registered")
            }
            Self::MissingDenseWeight { key } => {
                write!(f, "GPU-native dense weight {key:?} is not registered")
            }
            Self::ForeignDenseWeightHandle => write!(
                f,
                "GPU-native dense weight handle belongs to a different executor context"
            ),
            Self::StaleDenseWeightHandle { key } => {
                write!(f, "GPU-native dense weight handle for {key:?} is stale")
            }
            Self::DenseWeightKindMismatch {
                key,
                expected,
                actual,
            } => write!(
                f,
                "GPU-native dense weight {key:?} has kind {actual:?}, expected {expected:?}"
            ),
            Self::DenseWeightShapeMismatch {
                key,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                f,
                "GPU-native dense weight {key:?} has shape [{actual_rows}, {actual_cols}], expected [{expected_rows}, {expected_cols}]"
            ),
            Self::InvalidRmsNormWeightWidth { width } => write!(
                f,
                "GPU-native RMSNorm weight width must be non-zero, got {width}"
            ),
            Self::InvalidRmsNormEpsilon { epsilon_bits } => write!(
                f,
                "GPU-native RMSNorm epsilon must be finite and non-negative, got {}",
                f32::from_bits(*epsilon_bits)
            ),
            Self::ForeignRmsNormHandle => write!(
                f,
                "GPU-native RMSNorm handle belongs to a different executor context"
            ),
            Self::StaleRmsNormHandle { key } => {
                write!(f, "GPU-native RMSNorm handle for {key:?} is stale")
            }
            Self::RmsNormWeightWidth { expected, actual } => write!(
                f,
                "GPU-native RMSNorm weight width is {actual}, expected {expected}"
            ),
            Self::InvalidRmsNormGroups { groups } => write!(
                f,
                "GPU-native RMSNorm group count must be non-zero, got {groups}"
            ),
            Self::InvalidRmsNormGroupWidth { group_width } => write!(
                f,
                "GPU-native RMSNorm group width must be non-zero, got {group_width}"
            ),
            Self::RmsNormGeometryOverflow {
                groups,
                group_width,
            } => write!(
                f,
                "GPU-native RMSNorm geometry {groups} x {group_width} overflows"
            ),
            Self::RmsNormScratchGeometry { expected, actual } => write!(
                f,
                "GPU-native RMSNorm scratch has {actual} elements, expected {expected}"
            ),
            Self::InvalidScratchElements => {
                write!(f, "GPU-native scratch length must be non-zero")
            }
            Self::ScratchSizeOverflow { elements } => write!(
                f,
                "GPU-native scratch size overflows for {elements} elements"
            ),
            Self::ForeignTokenState => write!(
                f,
                "GPU-native token state belongs to a different executor context"
            ),
            Self::ForeignScratch => write!(
                f,
                "GPU-native scratch belongs to a different executor context"
            ),
            Self::AliasedInputOutput => {
                write!(f, "GPU-native GEMV input and output buffers must be distinct")
            }
            Self::GemvInputLength { expected, actual } => write!(
                f,
                "GPU-native GEMV input has {actual} elements, expected {expected}"
            ),
            Self::GemvOutputLength { expected, actual } => write!(
                f,
                "GPU-native GEMV output has {actual} elements, expected {expected}"
            ),
            Self::InvalidEmbeddingToken {
                token_id,
                vocab_size,
            } => write!(
                f,
                "GPU-native embedding token {token_id} is outside vocabulary size {vocab_size}"
            ),
            Self::EmbeddingWidth { expected, actual } => write!(
                f,
                "GPU-native embedding width is {actual}, expected token-state width {expected}"
            ),
            Self::ResidualContributionWidth { expected, actual } => write!(
                f,
                "GPU-native residual contribution has {actual} elements, expected {expected}"
            ),
            Self::InvalidRouterExpertCount { num_experts } => write!(
                f,
                "GPU-native router expert count must be in 1..={MAX_GPU_NATIVE_ROUTER_EXPERTS}, got {num_experts}"
            ),
            Self::InvalidRouterTopK { top_k, num_experts } => write!(
                f,
                "GPU-native router top_k must be in 1..=min({MAX_GPU_NATIVE_ROUTER_TOP_K}, {num_experts}), got {top_k}"
            ),
            Self::RouterGeometryOverflow {
                d_model,
                num_experts,
                top_k,
            } => write!(
                f,
                "GPU-native router geometry exceeds u32 shader geometry for d_model={d_model}, num_experts={num_experts}, top_k={top_k}"
            ),
            Self::RouterDModelMismatch { expected, actual } => write!(
                f,
                "GPU-native router d_model is {actual}, expected executor width {expected}"
            ),
            Self::ForeignRouterPlan => write!(
                f,
                "GPU-native router plan belongs to a different executor context"
            ),
            Self::ForeignRouterScratch => write!(
                f,
                "GPU-native router scratch belongs to a different executor context"
            ),
            Self::RouterScratchGeometry { expected, actual } => write!(
                f,
                "GPU-native router scratch geometry {actual:?} does not match plan geometry {expected:?}"
            ),
            Self::RouterLogitsLength { expected, actual } => write!(
                f,
                "GPU-native router logits have {actual} elements, expected {expected}"
            ),
            Self::RouterSelectedIdsLength { expected, actual } => write!(
                f,
                "GPU-native router selected ids have {actual} elements, expected {expected}"
            ),
            Self::RouterSelectedWeightsLength { expected, actual } => write!(
                f,
                "GPU-native router selected weights have {actual} elements, expected {expected}"
            ),
            Self::RouterGateShape {
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                f,
                "GPU-native router gate has shape [{actual_rows}, {actual_cols}], expected [{expected_rows}, {expected_cols}]"
            ),
            Self::RouterWorkgroupUnsupported {
                required,
                max_size_x,
                max_invocations,
            } => write!(
                f,
                "GPU-native router requires a {required}-lane workgroup, exceeding max_compute_workgroup_size_x={max_size_x} or max_compute_invocations_per_workgroup={max_invocations}"
            ),
            Self::RouterWorkgroupStorageUnsupported { required, maximum } => write!(
                f,
                "GPU-native router requires {required} bytes of workgroup storage, exceeding device maximum {maximum}"
            ),
            Self::InvalidExpertDff { d_ff } => {
                write!(f, "GPU-native expert d_ff must be non-zero, got {d_ff}")
            }
            Self::ExpertQ4GeometryIncompatible {
                d_model,
                d_ff,
                block_elements,
            } => write!(
                f,
                "GPU-native Q4_0 expert geometry requires d_model and d_ff to be multiples of {block_elements}, got d_model={d_model} d_ff={d_ff}"
            ),
            Self::ExpertGeometryOverflow {
                d_model,
                d_ff,
                num_experts,
                top_k,
            } => write!(
                f,
                "GPU-native expert geometry overflows shader/address space for d_model={d_model}, d_ff={d_ff}, num_experts={num_experts}, top_k={top_k}"
            ),
            Self::ExpertDModelMismatch { expected, actual } => write!(
                f,
                "GPU-native expert d_model is {actual}, expected executor width {expected}"
            ),
            Self::ExpertRouterGeometryMismatch { router, expert } => write!(
                f,
                "GPU-native router geometry {router:?} does not agree with expert geometry {expert:?}"
            ),
            Self::ForeignExpertArena => write!(
                f,
                "GPU-native expert arena belongs to a different executor context"
            ),
            Self::ForeignExpertScratch => write!(
                f,
                "GPU-native expert scratch belongs to a different executor context"
            ),
            Self::ExpertLayerMismatch {
                router_layer,
                expert_layer,
            } => write!(
                f,
                "GPU-native router layer {router_layer} does not match expert arena layer {expert_layer}"
            ),
            Self::InvalidExpertSlotCapacity { capacity } => write!(
                f,
                "GPU-native expert arena physical slot capacity must be non-zero, got {capacity}"
            ),
            Self::ExpertArenaBudgetTooSmall {
                requested_bytes,
                minimum_bytes,
            } => write!(
                f,
                "GPU-native expert arena budget {requested_bytes} bytes cannot fit the minimum one-slot allocation of {minimum_bytes} bytes"
            ),
            Self::ExpertArenaBudgetOverflow => {
                write!(f, "GPU-native expert arena allocation accounting overflowed")
            }
            Self::ExpertArenaBankLimit {
                required_banks,
                maximum_banks,
            } => write!(
                f,
                "GPU-native expert arena requires {required_banks} banks, exceeding maximum {maximum_banks}"
            ),
            Self::ExpertArenaBankCapacity {
                bank,
                required_bytes,
                maximum_bytes,
            } => write!(
                f,
                "GPU-native expert arena bank {bank} requires {required_bytes} bytes, exceeding {maximum_bytes} bytes"
            ),
            Self::ExpertStorageBuffersUnsupported { required, maximum } => write!(
                f,
                "GPU-native expert execution requires {required} storage buffers per shader stage, device supports {maximum}"
            ),
            Self::ExpertPushConstantsUnsupported { required, maximum } => write!(
                f,
                "GPU-native expert execution requires {required} push-constant bytes, device supports {maximum}"
            ),
            Self::ExpertWorkgroupUnsupported {
                required,
                max_size_x,
                max_invocations,
            } => write!(
                f,
                "GPU-native expert execution requires a {required}-lane workgroup, exceeding max_compute_workgroup_size_x={max_size_x} or max_compute_invocations_per_workgroup={max_invocations}"
            ),
            Self::DuplicateExpertLogicalId { logical_id } => write!(
                f,
                "GPU-native expert logical id {logical_id} is installed more than once"
            ),
            Self::ExpertLogicalIdOutOfRange {
                logical_id,
                num_experts,
            } => write!(
                f,
                "GPU-native expert logical id {logical_id} is outside layer-local expert count {num_experts}"
            ),
            Self::DuplicateExpertPhysicalLocation { packed } => write!(
                f,
                "GPU-native expert physical location 0x{packed:08x} is installed more than once"
            ),
            Self::ExpertBankOutOfRange { bank, active_banks } => write!(
                f,
                "GPU-native expert bank {bank} is outside active bank count {active_banks}"
            ),
            Self::ExpertSlotOutOfRange {
                bank,
                slot,
                capacity,
            } => write!(
                f,
                "GPU-native expert bank {bank} slot {slot} is outside bank capacity {capacity}"
            ),
            Self::ExpertLocationUnrepresentable { bank, slot } => write!(
                f,
                "GPU-native expert location bank={bank} slot={slot} cannot be packed without aliasing UNMAPPED"
            ),
            Self::InvalidExpertLogicalGeneration { logical_generation } => write!(
                f,
                "GPU-native expert logical generation must be non-zero, got {logical_generation}"
            ),
            Self::InvalidExpertSlotEpoch { slot_epoch } => write!(
                f,
                "GPU-native expert physical slot epoch must be non-zero, got {slot_epoch}"
            ),
            Self::ExpertInstallTicketExhausted => {
                write!(f, "GPU-native expert install ticket space exhausted")
            }
            Self::ExpertSlotEpochExhausted { bank, slot } => write!(
                f,
                "GPU-native expert physical slot epoch exhausted for bank={bank} slot={slot}"
            ),
            Self::ExpertInstallReservationLost => {
                write!(f, "GPU-native expert install reservation is no longer current")
            }
            Self::ExpertPayloadTooShort {
                logical_id,
                expected,
                actual,
            } => write!(
                f,
                "GPU-native Q4_0 expert {logical_id} payload has {actual} bytes, expected at least {expected}"
            ),
            Self::ExpertPayloadTrailingBytes {
                logical_id,
                maximum,
                actual,
            } => write!(
                f,
                "GPU-native Q4_0 expert {logical_id} payload has {actual} bytes, exceeding canonical aligned maximum {maximum}"
            ),
            Self::ExpertPayloadNonZeroPadding { logical_id } => write!(
                f,
                "GPU-native Q4_0 expert {logical_id} has non-zero trailing alignment padding"
            ),
            Self::ExpertResidentCountExceedsCapacity {
                residents,
                capacity,
            } => write!(
                f,
                "GPU-native expert arena has {residents} residents for {capacity} physical slots"
            ),
            Self::ExpertScratchGeometry { expected, actual } => write!(
                f,
                "GPU-native expert scratch geometry {actual:?} does not match {expected:?}"
            ),
            Self::InvalidAttentionHeadCount { tensor, heads } => write!(
                f,
                "GPU-native {tensor} head count must be non-zero, got {heads}"
            ),
            Self::InvalidAttentionHeadGeometry {
                num_heads,
                num_kv_heads,
                head_dim,
            } => write!(
                f,
                "GPU-native attention geometry requires non-zero head_dim and query heads divisible by KV heads, got num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}"
            ),
            Self::AttentionGeometryOverflow {
                num_heads,
                num_kv_heads,
                head_dim,
            } => write!(
                f,
                "GPU-native attention geometry overflows for num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}"
            ),
            Self::AttentionDModelMismatch { expected, actual } => write!(
                f,
                "GPU-native attention d_model is {actual}, expected executor width {expected}"
            ),
            Self::InvalidRopeDimension { rope_dim, head_dim } => write!(
                f,
                "GPU-native RoPE dimension must be in 1..={head_dim}, got {rope_dim}"
            ),
            Self::OddRopeDimension { rope_dim } => write!(
                f,
                "GPU-native RoPE dimension must be even, got {rope_dim}"
            ),
            Self::InvalidRopeBase { base_bits } => write!(
                f,
                "GPU-native RoPE base must be finite and positive, got {}",
                f32::from_bits(*base_bits)
            ),
            Self::InvalidRopeInverseFrequency { index, value_bits } => write!(
                f,
                "GPU-native RoPE inverse frequency {index} must be finite and positive, got {}",
                f32::from_bits(*value_bits)
            ),
            Self::InvalidRopeAttentionFactor { factor_bits } => write!(
                f,
                "GPU-native RoPE attention factor must be finite and positive, got {}",
                f32::from_bits(*factor_bits)
            ),
            Self::ForeignRopeHandle => write!(
                f,
                "GPU-native RoPE handle belongs to a different executor context"
            ),
            Self::StaleRopeHandle { key } => {
                write!(f, "GPU-native RoPE handle for {key:?} is stale")
            }
            Self::RopeDimensionMismatch { expected, actual } => write!(
                f,
                "GPU-native RoPE dimension is {actual}, expected {expected}"
            ),
            Self::RopeParameterWidth { expected, actual } => write!(
                f,
                "GPU-native RoPE inverse-frequency table has {actual} values, expected {expected}"
            ),
            Self::ForeignAttentionPlan => write!(
                f,
                "GPU-native attention plan belongs to a different executor context"
            ),
            Self::AttentionPlanLayerOutOfRange {
                layer_index,
                num_layers,
            } => write!(
                f,
                "GPU-native attention plan layer {layer_index} is outside request-local KV layer count {num_layers}"
            ),
            Self::ForeignAttentionScratch => write!(
                f,
                "GPU-native attention scratch belongs to a different executor context"
            ),
            Self::AttentionScratchGeometry { expected, actual } => write!(
                f,
                "GPU-native attention scratch geometry {actual:?} does not match plan geometry {expected:?}"
            ),
            Self::AttentionScratchWidth {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "GPU-native {tensor} scratch has {actual} elements, expected {expected}"
            ),
            Self::AttentionProjectionShape {
                tensor,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                f,
                "GPU-native {tensor} projection has shape [{actual_rows}, {actual_cols}], expected [{expected_rows}, {expected_cols}]"
            ),
            Self::AttentionNormWidth {
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "GPU-native {tensor} norm gain has width {actual}, expected {expected}"
            ),
            Self::InvalidKvLayerCount => {
                write!(f, "GPU-native KV layer count must be non-zero")
            }
            Self::InvalidKvCapacity => {
                write!(f, "GPU-native KV maximum sequence length must be non-zero")
            }
            Self::InvalidKvWidth => write!(f, "GPU-native KV width must be non-zero"),
            Self::KvCapacityOverflow {
                num_layers,
                max_seq_len,
                kv_width,
            } => write!(
                f,
                "GPU-native KV capacity overflows for layers={num_layers}, max_seq_len={max_seq_len}, width={kv_width}"
            ),
            Self::KvBufferLimit {
                required,
                max_buffer_size,
                max_storage_binding_size,
            } => write!(
                f,
                "GPU-native per-layer KV buffer requires {required} bytes, exceeding max_buffer_size={max_buffer_size} or max_storage_buffer_binding_size={max_storage_binding_size}"
            ),
            Self::ForeignKvState => write!(
                f,
                "GPU-native KV state belongs to a different executor context"
            ),
            Self::InvalidKvLayer { layer, num_layers } => write!(
                f,
                "GPU-native KV layer {layer} is outside layer count {num_layers}"
            ),
            Self::InvalidKvPosition {
                position,
                max_seq_len,
            } => write!(
                f,
                "GPU-native KV position {position} is outside capacity {max_seq_len}"
            ),
            Self::AttentionSequenceLengthOverflow { position } => write!(
                f,
                "GPU-native causal attention sequence length overflows for position {position}"
            ),
            Self::InvalidAttentionSequenceLength {
                seq_len,
                max_seq_len,
            } => write!(
                f,
                "GPU-native causal attention sequence length {seq_len} is outside 1..={max_seq_len}"
            ),
            Self::KvWidth { expected, actual } => write!(
                f,
                "GPU-native KV width is {actual}, expected {expected}"
            ),
            Self::DispatchGeometryUnsupported {
                workgroups,
                maximum,
            } => write!(
                f,
                "GPU-native dispatch requires {workgroups} workgroups, exceeding device maximum {maximum}"
            ),
            Self::AttentionWorkgroupUnsupported {
                required,
                max_size_x,
                max_invocations,
            } => write!(
                f,
                "GPU-native causal attention requires a {required}-lane workgroup, exceeding max_compute_workgroup_size_x={max_size_x} or max_compute_invocations_per_workgroup={max_invocations}"
            ),
            Self::Allocation(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for GpuNativeBootstrapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GpuStartupAllocationError> for GpuNativeBootstrapError {
    fn from(error: GpuStartupAllocationError) -> Self {
        Self::Allocation(error)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeAttentionTensor {
    Query,
    Key,
    Value,
    Context,
    Output,
}

impl fmt::Display for GpuNativeAttentionTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Query => write!(f, "query"),
            Self::Key => write!(f, "key"),
            Self::Value => write!(f, "value"),
            Self::Context => write!(f, "attention context"),
            Self::Output => write!(f, "attention output"),
        }
    }
}

/// Dense storage identity understood by the GPU-native shaders.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeDenseWeightKind {
    F32,
    Q8_0,
}

impl From<DenseDType> for GpuNativeDenseWeightKind {
    fn from(dtype: DenseDType) -> Self {
        match dtype {
            DenseDType::F32 => Self::F32,
            DenseDType::Q8_0 => Self::Q8_0,
        }
    }
}

/// Checked immutable matrix layout. `payload_bytes` is the exact model
/// payload; `allocation_bytes` includes only the trailing zero padding WGPU
/// requires to make a storage-buffer write four-byte aligned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeDenseWeightLayout {
    kind: GpuNativeDenseWeightKind,
    rows: usize,
    cols: usize,
    payload_bytes: u64,
    allocation_bytes: u64,
}

impl GpuNativeDenseWeightLayout {
    fn try_new(
        kind: GpuNativeDenseWeightKind,
        rows: usize,
        cols: usize,
        actual_bytes: usize,
    ) -> Result<Self, GpuNativeBootstrapError> {
        if rows == 0 || cols == 0 {
            return Err(GpuNativeBootstrapError::InvalidDenseWeightShape { rows, cols });
        }
        let elements = rows
            .checked_mul(cols)
            .ok_or(GpuNativeBootstrapError::DenseWeightShapeOverflow { rows, cols })?;
        if u32::try_from(rows).is_err()
            || u32::try_from(cols).is_err()
            || u32::try_from(elements).is_err()
        {
            return Err(GpuNativeBootstrapError::DenseWeightDimensionTooLarge { rows, cols });
        }
        let expected_bytes = match kind {
            GpuNativeDenseWeightKind::F32 => elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(GpuNativeBootstrapError::DenseWeightShapeOverflow { rows, cols })?,
            GpuNativeDenseWeightKind::Q8_0 => elements
                .div_ceil(Q8_0_BLOCK_ELEMS)
                .checked_mul(Q8_0_BLOCK_BYTES)
                .ok_or(GpuNativeBootstrapError::DenseWeightShapeOverflow { rows, cols })?,
        };
        if actual_bytes != expected_bytes {
            return Err(GpuNativeBootstrapError::DenseWeightByteLength {
                kind,
                rows,
                cols,
                expected: expected_bytes,
                actual: actual_bytes,
            });
        }
        // Q8 byte extraction also uses u32 offsets in WGSL. F32 indexes an
        // array<u32> by element, so its already-checked element count is the
        // relevant bound rather than its four-times-larger byte count.
        if kind == GpuNativeDenseWeightKind::Q8_0 && u32::try_from(expected_bytes).is_err() {
            return Err(GpuNativeBootstrapError::DenseWeightDimensionTooLarge { rows, cols });
        }
        let allocation_bytes = expected_bytes
            .checked_add(3)
            .map(|bytes| bytes & !3)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(GpuNativeBootstrapError::DenseWeightShapeOverflow { rows, cols })?;
        let payload_bytes = u64::try_from(expected_bytes)
            .map_err(|_| GpuNativeBootstrapError::DenseWeightShapeOverflow { rows, cols })?;
        Ok(Self {
            kind,
            rows,
            cols,
            payload_bytes,
            allocation_bytes,
        })
    }

    fn from_weight(weight: &DenseWeight) -> Result<Self, GpuNativeBootstrapError> {
        Self::try_new(
            weight.dtype().into(),
            weight.rows(),
            weight.cols(),
            weight.resident_bytes(),
        )
    }

    pub(crate) const fn kind(self) -> GpuNativeDenseWeightKind {
        self.kind
    }

    pub(crate) const fn rows(self) -> usize {
        self.rows
    }

    pub(crate) const fn cols(self) -> usize {
        self.cols
    }

    pub(crate) const fn payload_bytes(self) -> u64 {
        self.payload_bytes
    }

    pub(crate) const fn allocation_bytes(self) -> u64 {
        self.allocation_bytes
    }

    fn usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
    }

    fn validate_for_limits(
        self,
        label: &str,
        limits: &wgpu::Limits,
    ) -> Result<(), GpuNativeBootstrapError> {
        super::validate_startup_buffer(label, self.allocation_bytes, Self::usage(), limits)?;
        Ok(())
    }

    fn validate_embedding_token(self, token_id: u32) -> Result<(), GpuNativeBootstrapError> {
        if token_id as usize >= self.rows {
            return Err(GpuNativeBootstrapError::InvalidEmbeddingToken {
                token_id,
                vocab_size: self.rows,
            });
        }
        Ok(())
    }
}

/// One independently bindable physical buffer covering complete logical rows.
/// Q8_0 chunks retain the source tensor's global flat-block convention and can
/// therefore duplicate the single boundary block shared by adjacent chunks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GpuNativeDenseWeightChunkPlan {
    row_start: usize,
    row_count: usize,
    first_block: usize,
    payload_offset_bytes: usize,
    payload_bytes: u64,
    allocation_bytes: u64,
}

impl GpuNativeDenseWeightChunkPlan {
    fn row_end(self) -> usize {
        self.row_start + self.row_count
    }

    fn contains_row(self, row: usize) -> bool {
        self.row_start <= row && row < self.row_end()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GpuNativeDenseWeightPlan {
    layout: GpuNativeDenseWeightLayout,
    chunks: Vec<GpuNativeDenseWeightChunkPlan>,
    physical_allocation_bytes: u64,
}

impl GpuNativeDenseWeightPlan {
    fn try_new(
        layout: GpuNativeDenseWeightLayout,
        limits: &wgpu::Limits,
    ) -> Result<Self, GpuNativeBootstrapError> {
        let maximum = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));
        let mut chunks = Vec::new();
        let mut row_start = 0usize;
        let mut physical_allocation_bytes = 0u64;

        while row_start < layout.rows {
            let remaining = layout.rows - row_start;
            let row_count = match layout.kind {
                GpuNativeDenseWeightKind::F32 => {
                    let row_bytes = layout.cols.checked_mul(std::mem::size_of::<f32>()).ok_or(
                        GpuNativeBootstrapError::DenseWeightShapeOverflow {
                            rows: layout.rows,
                            cols: layout.cols,
                        },
                    )?;
                    let row_bytes = u64::try_from(row_bytes).map_err(|_| {
                        GpuNativeBootstrapError::DenseWeightShapeOverflow {
                            rows: layout.rows,
                            cols: layout.cols,
                        }
                    })?;
                    if row_bytes > maximum {
                        return Err(GpuNativeBootstrapError::DenseWeightRowExceedsDeviceLimit {
                            kind: layout.kind,
                            cols: layout.cols,
                            required: row_bytes,
                            maximum,
                        });
                    }
                    remaining.min(usize::try_from(maximum / row_bytes).unwrap_or(usize::MAX))
                }
                GpuNativeDenseWeightKind::Q8_0 => {
                    let one_row = Self::q8_chunk(layout, row_start, 1)?;
                    if one_row.allocation_bytes > maximum {
                        return Err(GpuNativeBootstrapError::DenseWeightRowExceedsDeviceLimit {
                            kind: layout.kind,
                            cols: layout.cols,
                            required: one_row.allocation_bytes,
                            maximum,
                        });
                    }
                    let mut low = 1usize;
                    let mut high = remaining;
                    while low < high {
                        let middle = low + (high - low).div_ceil(2);
                        if Self::q8_chunk(layout, row_start, middle)?.allocation_bytes <= maximum {
                            low = middle;
                        } else {
                            high = middle - 1;
                        }
                    }
                    low
                }
            };

            let chunk = match layout.kind {
                GpuNativeDenseWeightKind::F32 => {
                    let row_bytes = layout.cols * std::mem::size_of::<f32>();
                    let payload_offset_bytes = row_start * row_bytes;
                    let payload_bytes = u64::try_from(row_count * row_bytes).map_err(|_| {
                        GpuNativeBootstrapError::DenseWeightShapeOverflow {
                            rows: layout.rows,
                            cols: layout.cols,
                        }
                    })?;
                    GpuNativeDenseWeightChunkPlan {
                        row_start,
                        row_count,
                        first_block: 0,
                        payload_offset_bytes,
                        payload_bytes,
                        allocation_bytes: payload_bytes,
                    }
                }
                GpuNativeDenseWeightKind::Q8_0 => Self::q8_chunk(layout, row_start, row_count)?,
            };
            physical_allocation_bytes = physical_allocation_bytes
                .checked_add(chunk.allocation_bytes)
                .ok_or(GpuNativeBootstrapError::DenseWeightShapeOverflow {
                    rows: layout.rows,
                    cols: layout.cols,
                })?;
            chunks.push(chunk);
            row_start += row_count;
        }

        Ok(Self {
            layout,
            chunks,
            physical_allocation_bytes,
        })
    }

    fn q8_chunk(
        layout: GpuNativeDenseWeightLayout,
        row_start: usize,
        row_count: usize,
    ) -> Result<GpuNativeDenseWeightChunkPlan, GpuNativeBootstrapError> {
        let element_start = row_start.checked_mul(layout.cols).ok_or(
            GpuNativeBootstrapError::DenseWeightShapeOverflow {
                rows: layout.rows,
                cols: layout.cols,
            },
        )?;
        let element_end = row_start
            .checked_add(row_count)
            .and_then(|row_end| row_end.checked_mul(layout.cols))
            .ok_or(GpuNativeBootstrapError::DenseWeightShapeOverflow {
                rows: layout.rows,
                cols: layout.cols,
            })?;
        let first_block = element_start / Q8_0_BLOCK_ELEMS;
        let block_end = element_end.div_ceil(Q8_0_BLOCK_ELEMS);
        let block_count = block_end - first_block;
        let payload_offset_bytes = first_block.checked_mul(Q8_0_BLOCK_BYTES).ok_or(
            GpuNativeBootstrapError::DenseWeightShapeOverflow {
                rows: layout.rows,
                cols: layout.cols,
            },
        )?;
        let payload_bytes_usize = block_count.checked_mul(Q8_0_BLOCK_BYTES).ok_or(
            GpuNativeBootstrapError::DenseWeightShapeOverflow {
                rows: layout.rows,
                cols: layout.cols,
            },
        )?;
        let allocation_bytes_usize = payload_bytes_usize
            .checked_add(3)
            .map(|bytes| bytes & !3)
            .ok_or(GpuNativeBootstrapError::DenseWeightShapeOverflow {
                rows: layout.rows,
                cols: layout.cols,
            })?;
        Ok(GpuNativeDenseWeightChunkPlan {
            row_start,
            row_count,
            first_block,
            payload_offset_bytes,
            payload_bytes: payload_bytes_usize as u64,
            allocation_bytes: allocation_bytes_usize as u64,
        })
    }
}

/// Checked logical grouping for one RMSNorm dispatch. The shader launches one
/// workgroup per group and reuses one `group_width`-element F32 gain vector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GpuNativeRmsNormGeometry {
    groups: usize,
    group_width: usize,
    elements: usize,
}

impl GpuNativeRmsNormGeometry {
    fn try_new(
        groups: usize,
        group_width: usize,
        actual_elements: usize,
        weight_width: usize,
    ) -> Result<Self, GpuNativeBootstrapError> {
        if groups == 0 {
            return Err(GpuNativeBootstrapError::InvalidRmsNormGroups { groups });
        }
        if group_width == 0 {
            return Err(GpuNativeBootstrapError::InvalidRmsNormGroupWidth { group_width });
        }
        let elements = groups.checked_mul(group_width).ok_or(
            GpuNativeBootstrapError::RmsNormGeometryOverflow {
                groups,
                group_width,
            },
        )?;
        if u32::try_from(groups).is_err()
            || u32::try_from(group_width).is_err()
            || u32::try_from(elements).is_err()
        {
            return Err(GpuNativeBootstrapError::RmsNormGeometryOverflow {
                groups,
                group_width,
            });
        }
        if actual_elements != elements {
            return Err(GpuNativeBootstrapError::RmsNormScratchGeometry {
                expected: elements,
                actual: actual_elements,
            });
        }
        if weight_width != group_width {
            return Err(GpuNativeBootstrapError::RmsNormWeightWidth {
                expected: group_width,
                actual: weight_width,
            });
        }
        Ok(Self {
            groups,
            group_width,
            elements,
        })
    }

    fn checked_workgroups(self, limits: &wgpu::Limits) -> Result<u32, GpuNativeBootstrapError> {
        let workgroups = self.groups as u64;
        if workgroups > limits.max_compute_workgroups_per_dimension as u64 {
            return Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
                workgroups,
                maximum: limits.max_compute_workgroups_per_dimension,
            });
        }
        Ok(self.groups as u32)
    }
}

/// Bounded Qwen/Mixtral router geometry for one token. This first GPU-native
/// contract intentionally supports only softmax scoring, deterministic top-k,
/// and selected-weight renormalisation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeRouterGeometry {
    d_model: usize,
    num_experts: usize,
    top_k: usize,
}

impl GpuNativeRouterGeometry {
    pub(crate) fn try_new(
        d_model: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Result<Self, GpuNativeBootstrapError> {
        if d_model == 0 {
            return Err(GpuNativeBootstrapError::InvalidDModel);
        }
        if num_experts == 0 || num_experts > MAX_GPU_NATIVE_ROUTER_EXPERTS {
            return Err(GpuNativeBootstrapError::InvalidRouterExpertCount { num_experts });
        }
        if top_k == 0 || top_k > MAX_GPU_NATIVE_ROUTER_TOP_K || top_k > num_experts {
            return Err(GpuNativeBootstrapError::InvalidRouterTopK { top_k, num_experts });
        }
        let gate_elements = d_model.checked_mul(num_experts).ok_or(
            GpuNativeBootstrapError::RouterGeometryOverflow {
                d_model,
                num_experts,
                top_k,
            },
        )?;
        if u32::try_from(d_model).is_err()
            || u32::try_from(num_experts).is_err()
            || u32::try_from(top_k).is_err()
            || u32::try_from(gate_elements).is_err()
        {
            return Err(GpuNativeBootstrapError::RouterGeometryOverflow {
                d_model,
                num_experts,
                top_k,
            });
        }
        Ok(Self {
            d_model,
            num_experts,
            top_k,
        })
    }

    pub(crate) const fn d_model(self) -> usize {
        self.d_model
    }

    pub(crate) const fn num_experts(self) -> usize {
        self.num_experts
    }

    pub(crate) const fn top_k(self) -> usize {
        self.top_k
    }
}

fn validate_router_dispatch(limits: &wgpu::Limits) -> Result<(), GpuNativeBootstrapError> {
    if limits.max_compute_workgroup_size_x < GPU_NATIVE_ROUTER_WORKGROUP_SIZE
        || limits.max_compute_invocations_per_workgroup < GPU_NATIVE_ROUTER_WORKGROUP_SIZE
    {
        return Err(GpuNativeBootstrapError::RouterWorkgroupUnsupported {
            required: GPU_NATIVE_ROUTER_WORKGROUP_SIZE,
            max_size_x: limits.max_compute_workgroup_size_x,
            max_invocations: limits.max_compute_invocations_per_workgroup,
        });
    }
    if limits.max_compute_workgroup_storage_size < GPU_NATIVE_ROUTER_WORKGROUP_STORAGE_BYTES {
        return Err(GpuNativeBootstrapError::RouterWorkgroupStorageUnsupported {
            required: GPU_NATIVE_ROUTER_WORKGROUP_STORAGE_BYTES,
            maximum: limits.max_compute_workgroup_storage_size,
        });
    }
    if limits.max_compute_workgroups_per_dimension == 0 {
        return Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
            workgroups: 1,
            maximum: 0,
        });
    }
    Ok(())
}

/// Canonical Q4_0 SwiGLU geometry for one layer's mutable expert arena.
/// Projection blocks are laid out as `[gate][up][down]`. Each physical slot
/// prefixes those unchanged logical bytes with a four-byte slot epoch; the
/// prefix and physical tail padding are excluded from logical block offsets.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeQ4ExpertGeometry {
    d_model: usize,
    d_ff: usize,
    num_experts: usize,
    top_k: usize,
    blocks_per_projection: usize,
    logical_expert_bytes: usize,
    slot_stride_bytes: usize,
}

impl GpuNativeQ4ExpertGeometry {
    pub(crate) fn try_new(
        d_model: usize,
        d_ff: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Result<Self, GpuNativeBootstrapError> {
        let router = GpuNativeRouterGeometry::try_new(d_model, num_experts, top_k)?;
        if d_ff == 0 {
            return Err(GpuNativeBootstrapError::InvalidExpertDff { d_ff });
        }
        if !d_model.is_multiple_of(Q4_0_BLOCK_ELEMS) || !d_ff.is_multiple_of(Q4_0_BLOCK_ELEMS) {
            return Err(GpuNativeBootstrapError::ExpertQ4GeometryIncompatible {
                d_model,
                d_ff,
                block_elements: Q4_0_BLOCK_ELEMS,
            });
        }
        let overflow = || GpuNativeBootstrapError::ExpertGeometryOverflow {
            d_model,
            d_ff,
            num_experts,
            top_k,
        };
        let projection_elements = d_model.checked_mul(d_ff).ok_or_else(overflow)?;
        let blocks_per_projection = projection_elements / Q4_0_BLOCK_ELEMS;
        let logical_expert_bytes = blocks_per_projection
            .checked_mul(3)
            .and_then(|blocks| blocks.checked_mul(Q4_0_BLOCK_BYTES))
            .ok_or_else(overflow)?;
        let slot_stride_bytes = GPU_NATIVE_EXPERT_SLOT_EPOCH_BYTES
            .checked_add(logical_expert_bytes)
            .and_then(|bytes| bytes.checked_add(3))
            .map(|bytes| bytes & !3)
            .ok_or_else(overflow)?;
        let route_output_elements = top_k.checked_mul(d_model).ok_or_else(overflow)?;
        if u32::try_from(d_ff).is_err()
            || u32::try_from(projection_elements).is_err()
            || u32::try_from(blocks_per_projection).is_err()
            || u32::try_from(logical_expert_bytes).is_err()
            || u32::try_from(slot_stride_bytes).is_err()
            || u32::try_from(route_output_elements).is_err()
        {
            return Err(overflow());
        }
        debug_assert_eq!(router.d_model, d_model);
        Ok(Self {
            d_model,
            d_ff,
            num_experts,
            top_k,
            blocks_per_projection,
            logical_expert_bytes,
            slot_stride_bytes,
        })
    }

    pub(crate) const fn d_model(self) -> usize {
        self.d_model
    }

    pub(crate) const fn d_ff(self) -> usize {
        self.d_ff
    }

    pub(crate) const fn num_experts(self) -> usize {
        self.num_experts
    }

    pub(crate) const fn top_k(self) -> usize {
        self.top_k
    }

    pub(crate) const fn blocks_per_projection(self) -> usize {
        self.blocks_per_projection
    }

    pub(crate) const fn gate_block_offset(self) -> usize {
        0
    }

    pub(crate) const fn up_block_offset(self) -> usize {
        self.blocks_per_projection
    }

    pub(crate) const fn down_block_offset(self) -> usize {
        self.blocks_per_projection * 2
    }

    pub(crate) const fn logical_expert_bytes(self) -> usize {
        self.logical_expert_bytes
    }

    pub(crate) const fn payload_offset_bytes(self) -> usize {
        GPU_NATIVE_EXPERT_SLOT_EPOCH_BYTES
    }

    pub(crate) const fn physical_payload_bytes(self) -> usize {
        self.slot_stride_bytes - GPU_NATIVE_EXPERT_SLOT_EPOCH_BYTES
    }

    pub(crate) const fn slot_stride_bytes(self) -> usize {
        self.slot_stride_bytes
    }

    fn router_geometry(self) -> GpuNativeRouterGeometry {
        GpuNativeRouterGeometry {
            d_model: self.d_model,
            num_experts: self.num_experts,
            top_k: self.top_k,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeQ4ExpertLocation {
    bank: u32,
    slot: u32,
}

impl GpuNativeQ4ExpertLocation {
    pub(crate) fn try_new(bank: u32, slot: u32) -> Result<Self, GpuNativeBootstrapError> {
        let location = Self { bank, slot };
        location.pack()?;
        Ok(location)
    }

    pub(crate) const fn bank(self) -> u32 {
        self.bank
    }

    pub(crate) const fn slot(self) -> u32 {
        self.slot
    }

    pub(crate) fn pack(self) -> Result<u32, GpuNativeBootstrapError> {
        if self.bank as usize >= MAX_GPU_NATIVE_EXPERT_BANKS
            || self.slot > GPU_NATIVE_EXPERT_LOCATION_SLOT_MASK
        {
            return Err(GpuNativeBootstrapError::ExpertLocationUnrepresentable {
                bank: self.bank,
                slot: self.slot,
            });
        }
        let packed = (self.bank << GPU_NATIVE_EXPERT_LOCATION_BANK_SHIFT) | self.slot;
        if packed == GPU_NATIVE_EXPERT_UNMAPPED {
            return Err(GpuNativeBootstrapError::ExpertLocationUnrepresentable {
                bank: self.bank,
                slot: self.slot,
            });
        }
        Ok(packed)
    }

    pub(crate) fn unpack(packed: u32) -> Option<Self> {
        (packed != GPU_NATIVE_EXPERT_UNMAPPED).then_some(Self {
            bank: packed >> GPU_NATIVE_EXPERT_LOCATION_BANK_SHIFT,
            slot: packed & GPU_NATIVE_EXPERT_LOCATION_SLOT_MASK,
        })
    }
}

/// GPU-visible logical mapping and resolved-route entry. The slot epoch is a
/// physical incarnation number, not a logical admission generation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuNativeQ4ExpertMappingEntry {
    location: u32,
    slot_epoch: u32,
}

impl GpuNativeQ4ExpertMappingEntry {
    const UNMAPPED: Self = Self {
        location: GPU_NATIVE_EXPERT_UNMAPPED,
        slot_epoch: 0,
    };

    fn try_new(
        location: GpuNativeQ4ExpertLocation,
        slot_epoch: u32,
    ) -> Result<Self, GpuNativeBootstrapError> {
        if slot_epoch == 0 {
            return Err(GpuNativeBootstrapError::InvalidExpertSlotEpoch { slot_epoch });
        }
        Ok(Self {
            location: location.pack()?,
            slot_epoch,
        })
    }

    const fn location(self) -> u32 {
        self.location
    }

    const fn slot_epoch(self) -> u32 {
        self.slot_epoch
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GpuNativeQ4ExpertBankLayout {
    slot_capacity: usize,
    allocation_bytes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GpuNativeQ4ExpertArenaLayout {
    slot_capacity: usize,
    active_banks: usize,
    banks: [GpuNativeQ4ExpertBankLayout; MAX_GPU_NATIVE_EXPERT_BANKS],
    mapping_bytes: u64,
}

impl GpuNativeQ4ExpertArenaLayout {
    fn try_new(
        geometry: GpuNativeQ4ExpertGeometry,
        slot_capacity: usize,
        limits: &wgpu::Limits,
    ) -> Result<Self, GpuNativeBootstrapError> {
        if slot_capacity == 0 {
            return Err(GpuNativeBootstrapError::InvalidExpertSlotCapacity {
                capacity: slot_capacity,
            });
        }
        let stride = u64::try_from(geometry.slot_stride_bytes).map_err(|_| {
            GpuNativeBootstrapError::ExpertGeometryOverflow {
                d_model: geometry.d_model,
                d_ff: geometry.d_ff,
                num_experts: geometry.num_experts,
                top_k: geometry.top_k,
            }
        })?;
        let shader_addressable_bytes = u64::from(u32::MAX & !3);
        let maximum_bank_bytes = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size))
            .min(shader_addressable_bytes);
        let maximum_slots = (maximum_bank_bytes / stride)
            .min(u64::from(GPU_NATIVE_EXPERT_LOCATION_SLOT_MASK))
            as usize;
        if maximum_slots == 0 {
            return Err(GpuNativeBootstrapError::ExpertArenaBankCapacity {
                bank: 0,
                required_bytes: stride,
                maximum_bytes: maximum_bank_bytes,
            });
        }
        let required_banks = slot_capacity.div_ceil(maximum_slots);
        if required_banks > MAX_GPU_NATIVE_EXPERT_BANKS {
            return Err(GpuNativeBootstrapError::ExpertArenaBankLimit {
                required_banks,
                maximum_banks: MAX_GPU_NATIVE_EXPERT_BANKS,
            });
        }
        let mut remaining = slot_capacity;
        let mut banks = [GpuNativeQ4ExpertBankLayout {
            slot_capacity: 0,
            allocation_bytes: 4,
        }; MAX_GPU_NATIVE_EXPERT_BANKS];
        for (bank, layout) in banks.iter_mut().take(required_banks).enumerate() {
            let slots = remaining.min(maximum_slots);
            let allocation_bytes = (slots as u64).checked_mul(stride).ok_or(
                GpuNativeBootstrapError::ExpertArenaBankCapacity {
                    bank,
                    required_bytes: u64::MAX,
                    maximum_bytes: maximum_bank_bytes,
                },
            )?;
            if allocation_bytes > maximum_bank_bytes {
                return Err(GpuNativeBootstrapError::ExpertArenaBankCapacity {
                    bank,
                    required_bytes: allocation_bytes,
                    maximum_bytes: maximum_bank_bytes,
                });
            }
            *layout = GpuNativeQ4ExpertBankLayout {
                slot_capacity: slots,
                allocation_bytes,
            };
            remaining -= slots;
        }
        let mapping_bytes = geometry
            .num_experts
            .checked_mul(GPU_NATIVE_EXPERT_MAPPING_ENTRY_BYTES)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(GpuNativeBootstrapError::ExpertGeometryOverflow {
                d_model: geometry.d_model,
                d_ff: geometry.d_ff,
                num_experts: geometry.num_experts,
                top_k: geometry.top_k,
            })?;
        Ok(Self {
            slot_capacity,
            active_banks: required_banks,
            banks,
            mapping_bytes,
        })
    }

    fn weight_usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
    }

    fn mapping_usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
    }

    fn physical_bank_allocation_bytes(self) -> Result<u64, GpuNativeBootstrapError> {
        self.banks.iter().try_fold(0u64, |total, bank| {
            total
                .checked_add(bank.allocation_bytes)
                .ok_or(GpuNativeBootstrapError::ExpertArenaBudgetOverflow)
        })
    }

    fn active_bank_allocation_bytes(self) -> Result<u64, GpuNativeBootstrapError> {
        self.banks
            .iter()
            .take(self.active_banks)
            .try_fold(0u64, |total, bank| {
                total
                    .checked_add(bank.allocation_bytes)
                    .ok_or(GpuNativeBootstrapError::ExpertArenaBudgetOverflow)
            })
    }

    fn total_allocation_bytes(self) -> Result<u64, GpuNativeBootstrapError> {
        self.physical_bank_allocation_bytes()?
            .checked_add(self.mapping_bytes)
            .ok_or(GpuNativeBootstrapError::ExpertArenaBudgetOverflow)
    }

    fn location_for_flat_slot(
        self,
        flat_slot: usize,
    ) -> Result<GpuNativeQ4ExpertLocation, GpuNativeBootstrapError> {
        if flat_slot >= self.slot_capacity {
            return Err(
                GpuNativeBootstrapError::ExpertResidentCountExceedsCapacity {
                    residents: flat_slot.saturating_add(1),
                    capacity: self.slot_capacity,
                },
            );
        }
        let mut remaining = flat_slot;
        for (bank, layout) in self.banks.iter().take(self.active_banks).enumerate() {
            if remaining < layout.slot_capacity {
                return GpuNativeQ4ExpertLocation::try_new(bank as u32, remaining as u32);
            }
            remaining -= layout.slot_capacity;
        }
        Err(
            GpuNativeBootstrapError::ExpertResidentCountExceedsCapacity {
                residents: flat_slot.saturating_add(1),
                capacity: self.slot_capacity,
            },
        )
    }

    fn flat_slot_for_location(
        self,
        location: GpuNativeQ4ExpertLocation,
    ) -> Result<usize, GpuNativeBootstrapError> {
        let bank = location.bank as usize;
        if bank >= self.active_banks {
            return Err(GpuNativeBootstrapError::ExpertBankOutOfRange {
                bank: location.bank,
                active_banks: self.active_banks,
            });
        }
        if location.slot as usize >= self.banks[bank].slot_capacity {
            return Err(GpuNativeBootstrapError::ExpertSlotOutOfRange {
                bank: location.bank,
                slot: location.slot,
                capacity: self.banks[bank].slot_capacity,
            });
        }
        self.banks[..bank]
            .iter()
            .try_fold(0usize, |total, layout| {
                total
                    .checked_add(layout.slot_capacity)
                    .ok_or(GpuNativeBootstrapError::ExpertArenaBudgetOverflow)
            })?
            .checked_add(location.slot as usize)
            .ok_or(GpuNativeBootstrapError::ExpertArenaBudgetOverflow)
    }
}

/// Deterministic plan for the fixed expert-bank allocation owned by one
/// layer. The caller supplies a post-headroom budget; WGPU exposes no reliable
/// free-VRAM value for this module to discover.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeQ4ExpertVramPlan {
    geometry: GpuNativeQ4ExpertGeometry,
    requested_expert_budget_bytes: u64,
    layout: GpuNativeQ4ExpertArenaLayout,
    active_bank_allocation_bytes: u64,
    physical_bank_allocation_bytes: u64,
    total_arena_allocation_bytes: u64,
}

impl GpuNativeQ4ExpertVramPlan {
    /// Construct the exact fixed allocation for a requested physical slot
    /// count. Model-wide planning uses this helper so Slice 8 remains the
    /// single owner of bank, dummy-binding, mapping, and device-limit
    /// arithmetic.
    pub(crate) fn try_for_slot_capacity(
        geometry: GpuNativeQ4ExpertGeometry,
        slot_capacity: usize,
        limits: &wgpu::Limits,
    ) -> Result<Self, GpuNativeBootstrapError> {
        validate_q4_expert_pipeline_limits(limits)?;
        if slot_capacity > geometry.num_experts {
            return Err(
                GpuNativeBootstrapError::ExpertResidentCountExceedsCapacity {
                    residents: slot_capacity,
                    capacity: geometry.num_experts,
                },
            );
        }
        let layout = GpuNativeQ4ExpertArenaLayout::try_new(geometry, slot_capacity, limits)?;
        let active_bank_allocation_bytes = layout.active_bank_allocation_bytes()?;
        let physical_bank_allocation_bytes = layout.physical_bank_allocation_bytes()?;
        let total_arena_allocation_bytes = layout.total_allocation_bytes()?;
        Ok(Self {
            geometry,
            requested_expert_budget_bytes: total_arena_allocation_bytes,
            layout,
            active_bank_allocation_bytes,
            physical_bank_allocation_bytes,
            total_arena_allocation_bytes,
        })
    }

    pub(crate) fn try_new(
        geometry: GpuNativeQ4ExpertGeometry,
        expert_budget_bytes: u64,
        limits: &wgpu::Limits,
    ) -> Result<Self, GpuNativeBootstrapError> {
        validate_q4_expert_pipeline_limits(limits)?;
        let one = GpuNativeQ4ExpertArenaLayout::try_new(geometry, 1, limits)?;
        let minimum_bytes = one.total_allocation_bytes()?;
        if expert_budget_bytes < minimum_bytes {
            return Err(GpuNativeBootstrapError::ExpertArenaBudgetTooSmall {
                requested_bytes: expert_budget_bytes,
                minimum_bytes,
            });
        }

        let stride = u64::try_from(geometry.slot_stride_bytes)
            .map_err(|_| GpuNativeBootstrapError::ExpertArenaBudgetOverflow)?;
        let maximum_bank_bytes = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size))
            .min(u64::from(u32::MAX & !3));
        let maximum_slots_per_bank =
            (maximum_bank_bytes / stride).min(u64::from(GPU_NATIVE_EXPERT_LOCATION_SLOT_MASK));
        let maximum_slots = maximum_slots_per_bank
            .checked_mul(MAX_GPU_NATIVE_EXPERT_BANKS as u64)
            .map(|slots| slots.min(geometry.num_experts as u64))
            .and_then(|slots| usize::try_from(slots).ok())
            .ok_or(GpuNativeBootstrapError::ExpertArenaBudgetOverflow)?;

        let mut low = 1usize;
        let mut high = maximum_slots;
        while low < high {
            let midpoint = low + (high - low).div_ceil(2);
            let fits = GpuNativeQ4ExpertArenaLayout::try_new(geometry, midpoint, limits)
                .and_then(GpuNativeQ4ExpertArenaLayout::total_allocation_bytes)
                .is_ok_and(|bytes| bytes <= expert_budget_bytes);
            if fits {
                low = midpoint;
            } else {
                high = midpoint - 1;
            }
        }
        let layout = GpuNativeQ4ExpertArenaLayout::try_new(geometry, low, limits)?;
        let active_bank_allocation_bytes = layout.active_bank_allocation_bytes()?;
        let physical_bank_allocation_bytes = layout.physical_bank_allocation_bytes()?;
        let total_arena_allocation_bytes = layout.total_allocation_bytes()?;
        debug_assert!(total_arena_allocation_bytes <= expert_budget_bytes);
        Ok(Self {
            geometry,
            requested_expert_budget_bytes: expert_budget_bytes,
            layout,
            active_bank_allocation_bytes,
            physical_bank_allocation_bytes,
            total_arena_allocation_bytes,
        })
    }

    pub(crate) const fn geometry(self) -> GpuNativeQ4ExpertGeometry {
        self.geometry
    }

    pub(crate) const fn requested_expert_budget_bytes(self) -> u64 {
        self.requested_expert_budget_bytes
    }

    pub(crate) const fn slot_capacity(self) -> usize {
        self.layout.slot_capacity
    }

    pub(crate) const fn active_banks(self) -> usize {
        self.layout.active_banks
    }

    pub(crate) const fn slot_stride_bytes(self) -> usize {
        self.geometry.slot_stride_bytes
    }

    pub(crate) const fn active_bank_allocation_bytes(self) -> u64 {
        self.active_bank_allocation_bytes
    }

    pub(crate) const fn physical_bank_allocation_bytes(self) -> u64 {
        self.physical_bank_allocation_bytes
    }

    pub(crate) const fn mapping_metadata_bytes(self) -> u64 {
        self.layout.mapping_bytes
    }

    pub(crate) const fn total_arena_allocation_bytes(self) -> u64 {
        self.total_arena_allocation_bytes
    }
}

/// Layer-qualified logical identity. `logical_generation` is allocated by
/// `GpuExpertCache`; this arena never allocates or advances it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GpuNativeQ4ExpertKey {
    layer_index: usize,
    expert_id: u32,
    logical_generation: u64,
}

impl GpuNativeQ4ExpertKey {
    pub(crate) const fn new(layer_index: usize, expert_id: u32, logical_generation: u64) -> Self {
        Self {
            layer_index,
            expert_id,
            logical_generation,
        }
    }

    pub(crate) const fn layer_index(self) -> usize {
        self.layer_index
    }

    pub(crate) const fn expert_id(self) -> u32 {
        self.expert_id
    }

    pub(crate) const fn logical_generation(self) -> u64 {
        self.logical_generation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeQ4ExpertResidency {
    key: GpuNativeQ4ExpertKey,
    location: GpuNativeQ4ExpertLocation,
    slot_epoch: u32,
}

impl GpuNativeQ4ExpertResidency {
    pub(crate) const fn key(self) -> GpuNativeQ4ExpertKey {
        self.key
    }

    pub(crate) const fn location(self) -> GpuNativeQ4ExpertLocation {
        self.location
    }

    pub(crate) const fn slot_epoch(self) -> u32 {
        self.slot_epoch
    }

    fn mapping_entry(self) -> Result<GpuNativeQ4ExpertMappingEntry, GpuNativeBootstrapError> {
        GpuNativeQ4ExpertMappingEntry::try_new(self.location, self.slot_epoch)
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct GpuNativeQ4ExpertResidencySnapshot {
    pub(crate) resident_slots: usize,
    pub(crate) installing_slots: usize,
    pub(crate) free_slots: usize,
    pub(crate) slot_capacity: usize,
    pub(crate) active_banks: usize,
    pub(crate) arena_allocation_bytes: u64,
    pub(crate) resident_logical_payload_bytes: u64,
    pub(crate) expert_slot_installs: u64,
    pub(crate) expert_slot_retires: u64,
    pub(crate) expert_slot_reuses: u64,
    pub(crate) expert_mapping_publications: u64,
    pub(crate) expert_mapping_unpublications: u64,
    pub(crate) expert_stale_install_rejections: u64,
    pub(crate) expert_install_cancellations: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct GpuNativeQ4ExpertResidencyCounters {
    expert_slot_installs: u64,
    expert_slot_retires: u64,
    expert_slot_reuses: u64,
    expert_mapping_publications: u64,
    expert_mapping_unpublications: u64,
    expert_stale_install_rejections: u64,
    expert_install_cancellations: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GpuNativeQ4ExpertSlotOwner {
    Free,
    Installing {
        key: GpuNativeQ4ExpertKey,
        slot_epoch: u32,
        install_ticket: u64,
    },
    Resident(GpuNativeQ4ExpertResidency),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GpuNativeQ4ExpertPhysicalSlot {
    location: GpuNativeQ4ExpertLocation,
    last_epoch: u32,
    ever_installed: bool,
    owner: GpuNativeQ4ExpertSlotOwner,
}

#[derive(Debug)]
struct GpuNativeQ4ExpertArenaState {
    slots: Vec<GpuNativeQ4ExpertPhysicalSlot>,
    logical_slots: Vec<Option<usize>>,
    latest_generations: Vec<Option<u64>>,
    next_install_ticket: u64,
    counters: GpuNativeQ4ExpertResidencyCounters,
}

impl GpuNativeQ4ExpertArenaState {
    fn new(
        geometry: GpuNativeQ4ExpertGeometry,
        layout: GpuNativeQ4ExpertArenaLayout,
    ) -> Result<Self, GpuNativeBootstrapError> {
        let slots = (0..layout.slot_capacity)
            .map(|flat| {
                Ok(GpuNativeQ4ExpertPhysicalSlot {
                    location: layout.location_for_flat_slot(flat)?,
                    last_epoch: 0,
                    ever_installed: false,
                    owner: GpuNativeQ4ExpertSlotOwner::Free,
                })
            })
            .collect::<Result<Vec<_>, GpuNativeBootstrapError>>()?;
        Ok(Self {
            slots,
            logical_slots: vec![None; geometry.num_experts],
            latest_generations: vec![None; geometry.num_experts],
            next_install_ticket: 1,
            counters: GpuNativeQ4ExpertResidencyCounters::default(),
        })
    }
}

/// One setup-time placement. The logical generation is supplied by the
/// caller; the arena assigns the first nonzero physical slot epoch.
pub(crate) struct GpuNativeQ4ExpertUpload<'a> {
    pub(crate) logical_id: u32,
    pub(crate) logical_generation: u64,
    pub(crate) location: GpuNativeQ4ExpertLocation,
    pub(crate) payload: &'a [u8],
}

struct GpuNativeQ4ExpertPreparedUploads {
    mapping: Vec<GpuNativeQ4ExpertMappingEntry>,
    physical_slots: Vec<(GpuNativeQ4ExpertLocation, Vec<u8>)>,
    state: GpuNativeQ4ExpertArenaState,
}

fn validate_q4_expert_payload(
    geometry: GpuNativeQ4ExpertGeometry,
    logical_id: u32,
    payload: &[u8],
) -> Result<(), GpuNativeBootstrapError> {
    if payload.len() < geometry.logical_expert_bytes {
        return Err(GpuNativeBootstrapError::ExpertPayloadTooShort {
            logical_id,
            expected: geometry.logical_expert_bytes,
            actual: payload.len(),
        });
    }
    if payload.len() > geometry.physical_payload_bytes() {
        return Err(GpuNativeBootstrapError::ExpertPayloadTrailingBytes {
            logical_id,
            maximum: geometry.physical_payload_bytes(),
            actual: payload.len(),
        });
    }
    if payload[geometry.logical_expert_bytes..]
        .iter()
        .any(|&byte| byte != 0)
    {
        return Err(GpuNativeBootstrapError::ExpertPayloadNonZeroPadding { logical_id });
    }
    Ok(())
}

fn physical_q4_expert_slot(
    geometry: GpuNativeQ4ExpertGeometry,
    logical_id: u32,
    slot_epoch: u32,
    payload: &[u8],
) -> Result<Vec<u8>, GpuNativeBootstrapError> {
    validate_q4_expert_payload(geometry, logical_id, payload)?;
    if slot_epoch == 0 {
        return Err(GpuNativeBootstrapError::InvalidExpertSlotEpoch { slot_epoch });
    }
    let mut physical = vec![0; geometry.slot_stride_bytes];
    physical[..GPU_NATIVE_EXPERT_SLOT_EPOCH_BYTES].copy_from_slice(&slot_epoch.to_le_bytes());
    let payload_start = geometry.payload_offset_bytes();
    let payload_end = payload_start
        .checked_add(geometry.logical_expert_bytes)
        .ok_or(GpuNativeBootstrapError::ExpertArenaBudgetOverflow)?;
    physical[payload_start..payload_end].copy_from_slice(&payload[..geometry.logical_expert_bytes]);
    Ok(physical)
}

fn validate_q4_expert_uploads(
    layer_index: usize,
    geometry: GpuNativeQ4ExpertGeometry,
    layout: GpuNativeQ4ExpertArenaLayout,
    uploads: &[GpuNativeQ4ExpertUpload<'_>],
) -> Result<GpuNativeQ4ExpertPreparedUploads, GpuNativeBootstrapError> {
    if uploads.len() > layout.slot_capacity {
        return Err(
            GpuNativeBootstrapError::ExpertResidentCountExceedsCapacity {
                residents: uploads.len(),
                capacity: layout.slot_capacity,
            },
        );
    }
    let mut mapping = vec![GpuNativeQ4ExpertMappingEntry::UNMAPPED; geometry.num_experts];
    let mut state = GpuNativeQ4ExpertArenaState::new(geometry, layout)?;
    let mut physical = HashSet::with_capacity(uploads.len());
    let mut physical_slots = Vec::with_capacity(uploads.len());
    for upload in uploads {
        let logical_id = upload.logical_id as usize;
        if logical_id >= geometry.num_experts {
            return Err(GpuNativeBootstrapError::ExpertLogicalIdOutOfRange {
                logical_id: upload.logical_id,
                num_experts: geometry.num_experts,
            });
        }
        if upload.logical_generation == 0 {
            return Err(GpuNativeBootstrapError::InvalidExpertLogicalGeneration {
                logical_generation: upload.logical_generation,
            });
        }
        if mapping[logical_id] != GpuNativeQ4ExpertMappingEntry::UNMAPPED {
            return Err(GpuNativeBootstrapError::DuplicateExpertLogicalId {
                logical_id: upload.logical_id,
            });
        }
        let bank = upload.location.bank as usize;
        if bank >= layout.active_banks {
            return Err(GpuNativeBootstrapError::ExpertBankOutOfRange {
                bank: upload.location.bank,
                active_banks: layout.active_banks,
            });
        }
        if upload.location.slot as usize >= layout.banks[bank].slot_capacity {
            return Err(GpuNativeBootstrapError::ExpertSlotOutOfRange {
                bank: upload.location.bank,
                slot: upload.location.slot,
                capacity: layout.banks[bank].slot_capacity,
            });
        }
        let packed = upload.location.pack()?;
        if !physical.insert(packed) {
            return Err(GpuNativeBootstrapError::DuplicateExpertPhysicalLocation { packed });
        }
        let slot_epoch = 1;
        let residency = GpuNativeQ4ExpertResidency {
            key: GpuNativeQ4ExpertKey::new(
                layer_index,
                upload.logical_id,
                upload.logical_generation,
            ),
            location: upload.location,
            slot_epoch,
        };
        let flat_slot = layout.flat_slot_for_location(upload.location)?;
        state.slots[flat_slot].last_epoch = slot_epoch;
        state.slots[flat_slot].ever_installed = true;
        state.slots[flat_slot].owner = GpuNativeQ4ExpertSlotOwner::Resident(residency);
        state.logical_slots[logical_id] = Some(flat_slot);
        state.latest_generations[logical_id] = Some(upload.logical_generation);
        mapping[logical_id] = residency.mapping_entry()?;
        physical_slots.push((
            upload.location,
            physical_q4_expert_slot(geometry, upload.logical_id, slot_epoch, upload.payload)?,
        ));
    }
    state.counters.expert_slot_installs = uploads.len() as u64;
    state.counters.expert_mapping_publications = uploads.len() as u64;
    Ok(GpuNativeQ4ExpertPreparedUploads {
        mapping,
        physical_slots,
        state,
    })
}

fn validate_q4_expert_pipeline_limits(
    limits: &wgpu::Limits,
) -> Result<(), GpuNativeBootstrapError> {
    if limits.max_storage_buffers_per_shader_stage < GPU_NATIVE_EXPERT_REQUIRED_STORAGE_BUFFERS {
        return Err(GpuNativeBootstrapError::ExpertStorageBuffersUnsupported {
            required: GPU_NATIVE_EXPERT_REQUIRED_STORAGE_BUFFERS,
            maximum: limits.max_storage_buffers_per_shader_stage,
        });
    }
    if limits.max_push_constant_size < GPU_NATIVE_EXPERT_PUSH_CONSTANT_BYTES {
        return Err(GpuNativeBootstrapError::ExpertPushConstantsUnsupported {
            required: GPU_NATIVE_EXPERT_PUSH_CONSTANT_BYTES,
            maximum: limits.max_push_constant_size,
        });
    }
    if limits.max_compute_workgroup_size_x < GPU_NATIVE_EXPERT_WORKGROUP_SIZE
        || limits.max_compute_invocations_per_workgroup < GPU_NATIVE_EXPERT_WORKGROUP_SIZE
    {
        return Err(GpuNativeBootstrapError::ExpertWorkgroupUnsupported {
            required: GPU_NATIVE_EXPERT_WORKGROUP_SIZE,
            max_size_x: limits.max_compute_workgroup_size_x,
            max_invocations: limits.max_compute_invocations_per_workgroup,
        });
    }
    Ok(())
}

/// Qwen-compatible GPU-native attention geometry. Query heads may use GQA,
/// but Q/K/V share one head width and V is deliberately not asymmetric.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeAttentionGeometry {
    d_model: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    q_width: usize,
    kv_width: usize,
}

impl GpuNativeAttentionGeometry {
    pub(crate) fn try_new(
        d_model: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_dim: usize,
    ) -> Result<Self, GpuNativeBootstrapError> {
        if num_heads == 0 {
            return Err(GpuNativeBootstrapError::InvalidAttentionHeadCount {
                tensor: GpuNativeAttentionTensor::Query,
                heads: num_heads,
            });
        }
        if num_kv_heads == 0 {
            return Err(GpuNativeBootstrapError::InvalidAttentionHeadCount {
                tensor: GpuNativeAttentionTensor::Key,
                heads: num_kv_heads,
            });
        }
        if head_dim == 0 || num_heads < num_kv_heads || num_heads % num_kv_heads != 0 {
            return Err(GpuNativeBootstrapError::InvalidAttentionHeadGeometry {
                num_heads,
                num_kv_heads,
                head_dim,
            });
        }
        validate_rope_dimension(rope_dim, head_dim)?;
        let q_width = num_heads.checked_mul(head_dim).ok_or(
            GpuNativeBootstrapError::AttentionGeometryOverflow {
                num_heads,
                num_kv_heads,
                head_dim,
            },
        )?;
        let kv_width = num_kv_heads.checked_mul(head_dim).ok_or(
            GpuNativeBootstrapError::AttentionGeometryOverflow {
                num_heads,
                num_kv_heads,
                head_dim,
            },
        )?;
        if d_model == 0
            || u32::try_from(d_model).is_err()
            || u32::try_from(q_width).is_err()
            || u32::try_from(kv_width).is_err()
            || u32::try_from(head_dim).is_err()
            || u32::try_from(rope_dim).is_err()
        {
            return Err(GpuNativeBootstrapError::AttentionGeometryOverflow {
                num_heads,
                num_kv_heads,
                head_dim,
            });
        }
        Ok(Self {
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dim,
            q_width,
            kv_width,
        })
    }

    pub(crate) const fn d_model(self) -> usize {
        self.d_model
    }

    pub(crate) const fn num_heads(self) -> usize {
        self.num_heads
    }

    pub(crate) const fn num_kv_heads(self) -> usize {
        self.num_kv_heads
    }

    pub(crate) const fn head_dim(self) -> usize {
        self.head_dim
    }

    pub(crate) const fn rope_dim(self) -> usize {
        self.rope_dim
    }

    pub(crate) const fn q_width(self) -> usize {
        self.q_width
    }

    pub(crate) const fn kv_width(self) -> usize {
        self.kv_width
    }
}

fn validate_causal_attention_dispatch(
    geometry: GpuNativeAttentionGeometry,
    limits: &wgpu::Limits,
) -> Result<u32, GpuNativeBootstrapError> {
    if limits.max_compute_workgroup_size_x < GPU_NATIVE_ATTENTION_WORKGROUP_SIZE
        || limits.max_compute_invocations_per_workgroup < GPU_NATIVE_ATTENTION_WORKGROUP_SIZE
    {
        return Err(GpuNativeBootstrapError::AttentionWorkgroupUnsupported {
            required: GPU_NATIVE_ATTENTION_WORKGROUP_SIZE,
            max_size_x: limits.max_compute_workgroup_size_x,
            max_invocations: limits.max_compute_invocations_per_workgroup,
        });
    }
    if geometry.num_heads as u64 > limits.max_compute_workgroups_per_dimension as u64 {
        return Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
            workgroups: geometry.num_heads as u64,
            maximum: limits.max_compute_workgroups_per_dimension,
        });
    }
    Ok(geometry.num_heads as u32)
}

fn validate_rope_dimension(
    rope_dim: usize,
    head_dim: usize,
) -> Result<(), GpuNativeBootstrapError> {
    if rope_dim == 0 || rope_dim > head_dim {
        return Err(GpuNativeBootstrapError::InvalidRopeDimension { rope_dim, head_dim });
    }
    if !rope_dim.is_multiple_of(2) {
        return Err(GpuNativeBootstrapError::OddRopeDimension { rope_dim });
    }
    Ok(())
}

fn validate_rope_parameters(
    layout: GpuNativeRopeLayout,
    inverse_frequencies: &[f32],
    attention_factor: f32,
) -> Result<(), GpuNativeBootstrapError> {
    if inverse_frequencies.len() != layout.pairs {
        return Err(GpuNativeBootstrapError::RopeParameterWidth {
            expected: layout.pairs,
            actual: inverse_frequencies.len(),
        });
    }
    for (index, value) in inverse_frequencies.iter().copied().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(GpuNativeBootstrapError::InvalidRopeInverseFrequency {
                index,
                value_bits: value.to_bits(),
            });
        }
    }
    if !attention_factor.is_finite() || attention_factor <= 0.0 {
        return Err(GpuNativeBootstrapError::InvalidRopeAttentionFactor {
            factor_bits: attention_factor.to_bits(),
        });
    }
    Ok(())
}

fn standard_rope_inverse_frequencies(
    rope_dim: usize,
    base: f32,
) -> Result<Vec<f32>, GpuNativeBootstrapError> {
    let layout = GpuNativeRopeLayout::try_new(rope_dim, rope_dim)?;
    if !base.is_finite() || base <= 0.0 {
        return Err(GpuNativeBootstrapError::InvalidRopeBase {
            base_bits: base.to_bits(),
        });
    }
    Ok((0..layout.pairs)
        .map(|index| 1.0 / base.powf(2.0 * index as f32 / rope_dim as f32))
        .collect())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GpuNativeRopeLayout {
    rope_dim: usize,
    pairs: usize,
}

impl GpuNativeRopeLayout {
    fn try_new(rope_dim: usize, head_dim: usize) -> Result<Self, GpuNativeBootstrapError> {
        validate_rope_dimension(rope_dim, head_dim)?;
        Ok(Self {
            rope_dim,
            pairs: rope_dim / 2,
        })
    }
}

/// Checked request-local per-layer F32 KV capacity. Each physical K or V
/// buffer stores `[max_seq_len, kv_width]` for exactly one layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeKvLayout {
    num_layers: usize,
    max_seq_len: usize,
    kv_width: usize,
    layer_elements: usize,
    layer_bytes: u64,
    total_bytes: u64,
}

impl GpuNativeKvLayout {
    fn try_new(
        num_layers: usize,
        max_seq_len: usize,
        kv_width: usize,
        limits: &wgpu::Limits,
    ) -> Result<Self, GpuNativeBootstrapError> {
        if num_layers == 0 {
            return Err(GpuNativeBootstrapError::InvalidKvLayerCount);
        }
        if max_seq_len == 0 {
            return Err(GpuNativeBootstrapError::InvalidKvCapacity);
        }
        if kv_width == 0 {
            return Err(GpuNativeBootstrapError::InvalidKvWidth);
        }
        let capacity_error = || GpuNativeBootstrapError::KvCapacityOverflow {
            num_layers,
            max_seq_len,
            kv_width,
        };
        let layer_elements = max_seq_len
            .checked_mul(kv_width)
            .ok_or_else(capacity_error)?;
        if u32::try_from(max_seq_len).is_err()
            || u32::try_from(kv_width).is_err()
            || u32::try_from(layer_elements).is_err()
        {
            return Err(capacity_error());
        }
        let layer_bytes = layer_elements
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(capacity_error)?;
        let maximum_binding = u64::from(limits.max_storage_buffer_binding_size);
        if layer_bytes > limits.max_buffer_size || layer_bytes > maximum_binding {
            return Err(GpuNativeBootstrapError::KvBufferLimit {
                required: layer_bytes,
                max_buffer_size: limits.max_buffer_size,
                max_storage_binding_size: maximum_binding,
            });
        }
        let num_layers_u64 = u64::try_from(num_layers).map_err(|_| capacity_error())?;
        let total_bytes = layer_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_mul(num_layers_u64))
            .ok_or_else(capacity_error)?;
        Ok(Self {
            num_layers,
            max_seq_len,
            kv_width,
            layer_elements,
            layer_bytes,
            total_bytes,
        })
    }

    fn usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE
    }

    fn validate_layer(self, layer: usize) -> Result<(), GpuNativeBootstrapError> {
        if layer >= self.num_layers {
            return Err(GpuNativeBootstrapError::InvalidKvLayer {
                layer,
                num_layers: self.num_layers,
            });
        }
        Ok(())
    }

    fn validate_position(self, position: usize) -> Result<(), GpuNativeBootstrapError> {
        if position >= self.max_seq_len {
            return Err(GpuNativeBootstrapError::InvalidKvPosition {
                position,
                max_seq_len: self.max_seq_len,
            });
        }
        Ok(())
    }

    fn element_offset(
        self,
        layer: usize,
        position: usize,
    ) -> Result<usize, GpuNativeBootstrapError> {
        self.validate_layer(layer)?;
        self.validate_position(position)?;
        position
            .checked_mul(self.kv_width)
            .ok_or(GpuNativeBootstrapError::KvCapacityOverflow {
                num_layers: self.num_layers,
                max_seq_len: self.max_seq_len,
                kv_width: self.kv_width,
            })
    }

    pub(crate) const fn num_layers(self) -> usize {
        self.num_layers
    }

    pub(crate) const fn max_seq_len(self) -> usize {
        self.max_seq_len
    }

    pub(crate) const fn kv_width(self) -> usize {
        self.kv_width
    }

    pub(crate) const fn layer_bytes(self) -> u64 {
        self.layer_bytes
    }

    pub(crate) const fn total_bytes(self) -> u64 {
        self.total_bytes
    }
}

/// Stable model-scoped key used to retrieve a registered dense tensor.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GpuNativeDenseWeightKey(Arc<str>);

impl GpuNativeDenseWeightKey {
    pub(crate) fn try_new(key: impl Into<String>) -> Result<Self, GpuNativeBootstrapError> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(GpuNativeBootstrapError::InvalidDenseWeightKey);
        }
        Ok(Self(Arc::<str>::from(key)))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

/// Opaque, context-bound reference to one persistent model weight.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeDenseWeightHandle {
    context_id: u64,
    weight_id: u64,
    key: GpuNativeDenseWeightKey,
    layout: GpuNativeDenseWeightLayout,
}

impl GpuNativeDenseWeightHandle {
    pub(crate) fn key(&self) -> &GpuNativeDenseWeightKey {
        &self.key
    }

    pub(crate) const fn layout(&self) -> GpuNativeDenseWeightLayout {
        self.layout
    }
}

/// Narrow semantic wrapper for a persistent F32 `[1, width]` gain vector.
/// The underlying registry identity remains model-scoped and context-bound,
/// while callers cannot accidentally use this handle as a matrix operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeRmsNormHandle {
    dense: GpuNativeDenseWeightHandle,
    width: usize,
}

impl GpuNativeRmsNormHandle {
    fn from_dense(dense: GpuNativeDenseWeightHandle) -> Self {
        Self {
            width: dense.layout.cols,
            dense,
        }
    }

    pub(crate) const fn width(&self) -> usize {
        self.width
    }
}

/// Persistent model-scoped RoPE parameters stored in the existing dense F32
/// registry. `inv_freq` is `[rope_dim / 2]`; `attention_factor` is folded into
/// both sine and cosine, matching the CPU helper when a derived scaling table
/// is registered.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeRopeHandle {
    dense: GpuNativeDenseWeightHandle,
    rope_dim: usize,
    attention_factor_bits: u32,
}

impl GpuNativeRopeHandle {
    pub(crate) const fn rope_dim(&self) -> usize {
        self.rope_dim
    }
}

/// Immutable context-bound gate handle and geometry for one Qwen-compatible
/// MoE router. `layer_index` is retained for the later expert-residency key;
/// it does not alter routing in this slice.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeRouterPlan {
    context_id: u64,
    layer_index: usize,
    geometry: GpuNativeRouterGeometry,
    gate: GpuNativeDenseWeightHandle,
}

impl GpuNativeRouterPlan {
    pub(crate) const fn geometry(&self) -> GpuNativeRouterGeometry {
        self.geometry
    }

    pub(crate) const fn layer_index(&self) -> usize {
        self.layer_index
    }
}

/// Fixed-allocation, mutable context- and layer-bound four-bank Q4_0 slot
/// arena. Unused shader banks hold a minimal zero buffer. Physical allocation
/// remains resident even when every slot is logically free.
pub(crate) struct GpuNativeQ4ExpertArena<B = wgpu::Buffer> {
    context_id: u64,
    layer_index: usize,
    geometry: GpuNativeQ4ExpertGeometry,
    plan: GpuNativeQ4ExpertVramPlan,
    banks: [B; MAX_GPU_NATIVE_EXPERT_BANKS],
    mapping: B,
    state: ParkingMutex<GpuNativeQ4ExpertArenaState>,
}

impl<B> GpuNativeQ4ExpertArena<B> {
    fn from_buffers(
        context_id: u64,
        layer_index: usize,
        plan: GpuNativeQ4ExpertVramPlan,
        banks: [B; MAX_GPU_NATIVE_EXPERT_BANKS],
        mapping: B,
        state: GpuNativeQ4ExpertArenaState,
    ) -> Self {
        Self {
            context_id,
            layer_index,
            geometry: plan.geometry,
            plan,
            banks,
            mapping,
            state: ParkingMutex::new(state),
        }
    }

    pub(crate) const fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub(crate) const fn geometry(&self) -> GpuNativeQ4ExpertGeometry {
        self.geometry
    }

    pub(crate) const fn slot_capacity(&self) -> usize {
        self.plan.layout.slot_capacity
    }

    pub(crate) const fn active_banks(&self) -> usize {
        self.plan.layout.active_banks
    }

    pub(crate) fn resident_experts(&self) -> usize {
        self.state
            .lock()
            .slots
            .iter()
            .filter(|slot| matches!(slot.owner, GpuNativeQ4ExpertSlotOwner::Resident(_)))
            .count()
    }

    pub(crate) const fn vram_plan(&self) -> GpuNativeQ4ExpertVramPlan {
        self.plan
    }

    pub(crate) fn residency_snapshot(&self) -> GpuNativeQ4ExpertResidencySnapshot {
        let state = self.state.lock();
        let resident_slots = state
            .slots
            .iter()
            .filter(|slot| matches!(slot.owner, GpuNativeQ4ExpertSlotOwner::Resident(_)))
            .count();
        let installing_slots = state
            .slots
            .iter()
            .filter(|slot| matches!(slot.owner, GpuNativeQ4ExpertSlotOwner::Installing { .. }))
            .count();
        let free_slots = state
            .slots
            .len()
            .saturating_sub(resident_slots)
            .saturating_sub(installing_slots);
        let resident_logical_payload_bytes = u64::try_from(resident_slots)
            .ok()
            .and_then(|slots| slots.checked_mul(self.geometry.logical_expert_bytes as u64))
            .unwrap_or(u64::MAX);
        GpuNativeQ4ExpertResidencySnapshot {
            resident_slots,
            installing_slots,
            free_slots,
            slot_capacity: self.plan.slot_capacity(),
            active_banks: self.plan.active_banks(),
            arena_allocation_bytes: self.plan.total_arena_allocation_bytes(),
            resident_logical_payload_bytes,
            expert_slot_installs: state.counters.expert_slot_installs,
            expert_slot_retires: state.counters.expert_slot_retires,
            expert_slot_reuses: state.counters.expert_slot_reuses,
            expert_mapping_publications: state.counters.expert_mapping_publications,
            expert_mapping_unpublications: state.counters.expert_mapping_unpublications,
            expert_stale_install_rejections: state.counters.expert_stale_install_rejections,
            expert_install_cancellations: state.counters.expert_install_cancellations,
        }
    }

    fn cancel_install(&self, flat_slot: usize, install_ticket: u64) {
        let mut state = self.state.lock();
        let key = match state.slots.get(flat_slot).map(|slot| slot.owner) {
            Some(GpuNativeQ4ExpertSlotOwner::Installing {
                key,
                install_ticket: current,
                ..
            }) if current == install_ticket => key,
            _ => return,
        };
        state.slots[flat_slot].owner = GpuNativeQ4ExpertSlotOwner::Free;
        let logical_id = key.expert_id as usize;
        if state.logical_slots.get(logical_id).copied().flatten() == Some(flat_slot) {
            state.logical_slots[logical_id] = None;
        }
        state.counters.expert_install_cancellations = state
            .counters
            .expert_install_cancellations
            .saturating_add(1);
    }

    fn acquire_with_unpublish<F>(
        &self,
        key: GpuNativeQ4ExpertKey,
        mut unpublish: F,
    ) -> Result<GpuNativeQ4ExpertAcquire<'_, B>, GpuNativeBootstrapError>
    where
        F: FnMut(u64, GpuNativeQ4ExpertMappingEntry),
    {
        if key.layer_index != self.layer_index {
            return Err(GpuNativeBootstrapError::ExpertLayerMismatch {
                router_layer: key.layer_index,
                expert_layer: self.layer_index,
            });
        }
        let logical_id = key.expert_id as usize;
        if logical_id >= self.geometry.num_experts {
            return Err(GpuNativeBootstrapError::ExpertLogicalIdOutOfRange {
                logical_id: key.expert_id,
                num_experts: self.geometry.num_experts,
            });
        }
        if key.logical_generation == 0 {
            return Err(GpuNativeBootstrapError::InvalidExpertLogicalGeneration {
                logical_generation: key.logical_generation,
            });
        }

        let mut state = self.state.lock();
        if let Some(latest) = state.latest_generations[logical_id] {
            if key.logical_generation < latest {
                state.counters.expert_stale_install_rejections = state
                    .counters
                    .expert_stale_install_rejections
                    .saturating_add(1);
                return Ok(GpuNativeQ4ExpertAcquire::StaleRequester);
            }
            if key.logical_generation == latest {
                if let Some(flat_slot) = state.logical_slots[logical_id] {
                    return match state.slots[flat_slot].owner {
                        GpuNativeQ4ExpertSlotOwner::Resident(residency) if residency.key == key => {
                            Ok(GpuNativeQ4ExpertAcquire::Hit(residency))
                        }
                        GpuNativeQ4ExpertSlotOwner::Installing {
                            key: installing, ..
                        } if installing == key => Ok(GpuNativeQ4ExpertAcquire::InstallInProgress),
                        _ => Err(GpuNativeBootstrapError::ExpertInstallReservationLost),
                    };
                }
            } else if let Some(flat_slot) = state.logical_slots[logical_id] {
                match state.slots[flat_slot].owner {
                    GpuNativeQ4ExpertSlotOwner::Resident(_) => {
                        let mapping_offset = u64::try_from(logical_id)
                            .ok()
                            .and_then(|id| {
                                id.checked_mul(GPU_NATIVE_EXPERT_MAPPING_ENTRY_BYTES as u64)
                            })
                            .ok_or(GpuNativeBootstrapError::ExpertArenaBudgetOverflow)?;
                        unpublish(mapping_offset, GpuNativeQ4ExpertMappingEntry::UNMAPPED);
                        state.counters.expert_slot_retires =
                            state.counters.expert_slot_retires.saturating_add(1);
                        state.counters.expert_mapping_unpublications = state
                            .counters
                            .expert_mapping_unpublications
                            .saturating_add(1);
                    }
                    GpuNativeQ4ExpertSlotOwner::Installing { .. } => {
                        state.counters.expert_install_cancellations = state
                            .counters
                            .expert_install_cancellations
                            .saturating_add(1);
                    }
                    GpuNativeQ4ExpertSlotOwner::Free => {
                        return Err(GpuNativeBootstrapError::ExpertInstallReservationLost);
                    }
                }
                state.slots[flat_slot].owner = GpuNativeQ4ExpertSlotOwner::Free;
                state.logical_slots[logical_id] = None;
            }
        }
        state.latest_generations[logical_id] = Some(key.logical_generation);

        let mut exhausted = None;
        let flat_slot = match state.slots.iter().enumerate().find_map(|(index, slot)| {
            if !matches!(slot.owner, GpuNativeQ4ExpertSlotOwner::Free) {
                return None;
            }
            if slot.last_epoch == u32::MAX {
                exhausted.get_or_insert(slot.location);
                None
            } else {
                Some(index)
            }
        }) {
            Some(slot) => slot,
            None => {
                if let Some(location) = exhausted {
                    return Err(GpuNativeBootstrapError::ExpertSlotEpochExhausted {
                        bank: location.bank,
                        slot: location.slot,
                    });
                }
                return Ok(GpuNativeQ4ExpertAcquire::NoPhysicalSlot);
            }
        };
        let install_ticket = state.next_install_ticket;
        state.next_install_ticket = install_ticket
            .checked_add(1)
            .ok_or(GpuNativeBootstrapError::ExpertInstallTicketExhausted)?;
        let slot = &mut state.slots[flat_slot];
        let slot_epoch = slot.last_epoch.checked_add(1).ok_or(
            GpuNativeBootstrapError::ExpertSlotEpochExhausted {
                bank: slot.location.bank,
                slot: slot.location.slot,
            },
        )?;
        let reused = slot.ever_installed;
        slot.last_epoch = slot_epoch;
        let residency = GpuNativeQ4ExpertResidency {
            key,
            location: slot.location,
            slot_epoch,
        };
        slot.owner = GpuNativeQ4ExpertSlotOwner::Installing {
            key,
            slot_epoch,
            install_ticket,
        };
        state.logical_slots[logical_id] = Some(flat_slot);
        drop(state);
        Ok(GpuNativeQ4ExpertAcquire::Install(
            GpuNativeQ4ExpertInstallPermit {
                arena: self,
                key,
                flat_slot,
                residency,
                install_ticket,
                reused,
                active: true,
            },
        ))
    }

    fn retire_with_unpublish<F>(
        &self,
        key: GpuNativeQ4ExpertKey,
        mut unpublish: F,
    ) -> Result<GpuNativeQ4ExpertRetire, GpuNativeBootstrapError>
    where
        F: FnMut(u64, GpuNativeQ4ExpertMappingEntry),
    {
        if key.layer_index != self.layer_index {
            return Err(GpuNativeBootstrapError::ExpertLayerMismatch {
                router_layer: key.layer_index,
                expert_layer: self.layer_index,
            });
        }
        let logical_id = key.expert_id as usize;
        if logical_id >= self.geometry.num_experts {
            return Err(GpuNativeBootstrapError::ExpertLogicalIdOutOfRange {
                logical_id: key.expert_id,
                num_experts: self.geometry.num_experts,
            });
        }
        let mut state = self.state.lock();
        let Some(latest) = state.latest_generations[logical_id] else {
            return Ok(GpuNativeQ4ExpertRetire::NotResident);
        };
        if key.logical_generation < latest {
            return Ok(GpuNativeQ4ExpertRetire::StaleRequester);
        }
        if key.logical_generation != latest {
            return Ok(GpuNativeQ4ExpertRetire::NotResident);
        }
        let Some(flat_slot) = state.logical_slots[logical_id] else {
            return Ok(GpuNativeQ4ExpertRetire::NotResident);
        };
        match state.slots[flat_slot].owner {
            GpuNativeQ4ExpertSlotOwner::Resident(residency) if residency.key == key => {
                let mapping_offset = u64::try_from(logical_id)
                    .ok()
                    .and_then(|id| id.checked_mul(GPU_NATIVE_EXPERT_MAPPING_ENTRY_BYTES as u64))
                    .ok_or(GpuNativeBootstrapError::ExpertArenaBudgetOverflow)?;
                unpublish(mapping_offset, GpuNativeQ4ExpertMappingEntry::UNMAPPED);
                state.slots[flat_slot].owner = GpuNativeQ4ExpertSlotOwner::Free;
                state.logical_slots[logical_id] = None;
                state.counters.expert_slot_retires =
                    state.counters.expert_slot_retires.saturating_add(1);
                state.counters.expert_mapping_unpublications = state
                    .counters
                    .expert_mapping_unpublications
                    .saturating_add(1);
                Ok(GpuNativeQ4ExpertRetire::Retired)
            }
            GpuNativeQ4ExpertSlotOwner::Installing {
                key: installing, ..
            } if installing == key => {
                state.slots[flat_slot].owner = GpuNativeQ4ExpertSlotOwner::Free;
                state.logical_slots[logical_id] = None;
                state.counters.expert_install_cancellations = state
                    .counters
                    .expert_install_cancellations
                    .saturating_add(1);
                Ok(GpuNativeQ4ExpertRetire::CancelledInstall)
            }
            _ => Err(GpuNativeBootstrapError::ExpertInstallReservationLost),
        }
    }
}

pub(crate) enum GpuNativeQ4ExpertAcquire<'a, B = wgpu::Buffer> {
    Hit(GpuNativeQ4ExpertResidency),
    Install(GpuNativeQ4ExpertInstallPermit<'a, B>),
    InstallInProgress,
    StaleRequester,
    NoPhysicalSlot,
}

pub(crate) struct GpuNativeQ4ExpertInstallPermit<'a, B = wgpu::Buffer> {
    arena: &'a GpuNativeQ4ExpertArena<B>,
    key: GpuNativeQ4ExpertKey,
    flat_slot: usize,
    residency: GpuNativeQ4ExpertResidency,
    install_ticket: u64,
    reused: bool,
    active: bool,
}

impl<B> GpuNativeQ4ExpertInstallPermit<'_, B> {
    pub(crate) const fn key(&self) -> GpuNativeQ4ExpertKey {
        self.key
    }

    pub(crate) const fn reserved_residency(&self) -> GpuNativeQ4ExpertResidency {
        self.residency
    }

    fn install_with_writes<PhysicalWrite, MappingWrite>(
        mut self,
        payload: &[u8],
        mut physical_write: PhysicalWrite,
        mut mapping_write: MappingWrite,
    ) -> Result<GpuNativeQ4ExpertResidency, GpuNativeBootstrapError>
    where
        PhysicalWrite: FnMut(u32, u64, &[u8]),
        MappingWrite: FnMut(u64, GpuNativeQ4ExpertMappingEntry),
    {
        let physical = physical_q4_expert_slot(
            self.arena.geometry,
            self.key.expert_id,
            self.residency.slot_epoch,
            payload,
        )?;
        let mapping = self.residency.mapping_entry()?;
        let logical_id = self.key.expert_id as usize;
        let physical_offset = u64::from(self.residency.location.slot)
            .checked_mul(self.arena.geometry.slot_stride_bytes as u64)
            .ok_or(GpuNativeBootstrapError::ExpertArenaBudgetOverflow)?;
        let mapping_offset = u64::try_from(logical_id)
            .ok()
            .and_then(|id| id.checked_mul(GPU_NATIVE_EXPERT_MAPPING_ENTRY_BYTES as u64))
            .ok_or(GpuNativeBootstrapError::ExpertArenaBudgetOverflow)?;

        let mut state = self.arena.state.lock();
        let reservation_current = state.latest_generations.get(logical_id).copied().flatten()
            == Some(self.key.logical_generation)
            && state.logical_slots.get(logical_id).copied().flatten() == Some(self.flat_slot)
            && matches!(
                state.slots.get(self.flat_slot).map(|slot| slot.owner),
                Some(GpuNativeQ4ExpertSlotOwner::Installing {
                    key,
                    slot_epoch,
                    install_ticket,
                }) if key == self.key
                    && slot_epoch == self.residency.slot_epoch
                    && install_ticket == self.install_ticket
            );
        if !reservation_current {
            state.counters.expert_stale_install_rejections = state
                .counters
                .expert_stale_install_rejections
                .saturating_add(1);
            self.active = false;
            return Err(GpuNativeBootstrapError::ExpertInstallReservationLost);
        }

        physical_write(self.residency.location.bank, physical_offset, &physical);
        mapping_write(mapping_offset, mapping);
        state.slots[self.flat_slot].owner = GpuNativeQ4ExpertSlotOwner::Resident(self.residency);
        state.slots[self.flat_slot].ever_installed = true;
        state.counters.expert_slot_installs = state.counters.expert_slot_installs.saturating_add(1);
        if self.reused {
            state.counters.expert_slot_reuses = state.counters.expert_slot_reuses.saturating_add(1);
        }
        state.counters.expert_mapping_publications =
            state.counters.expert_mapping_publications.saturating_add(1);
        self.active = false;
        Ok(self.residency)
    }
}

impl<B> Drop for GpuNativeQ4ExpertInstallPermit<'_, B> {
    fn drop(&mut self) {
        if self.active {
            self.arena
                .cancel_install(self.flat_slot, self.install_ticket);
            self.active = false;
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GpuNativeQ4ExpertRetire {
    Retired,
    CancelledInstall,
    NotResident,
    StaleRequester,
}

impl<B> fmt::Debug for GpuNativeQ4ExpertArena<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuNativeQ4ExpertArena")
            .field("layer_index", &self.layer_index)
            .field("geometry", &self.geometry)
            .field("plan", &self.plan)
            .field("residency", &self.residency_snapshot())
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeAttentionNorm {
    handle: GpuNativeRmsNormHandle,
    epsilon_bits: u32,
}

impl GpuNativeAttentionNorm {
    pub(crate) fn try_new(
        handle: GpuNativeRmsNormHandle,
        epsilon: f32,
    ) -> Result<Self, GpuNativeBootstrapError> {
        validate_rms_norm_epsilon(epsilon)?;
        Ok(Self {
            handle,
            epsilon_bits: epsilon.to_bits(),
        })
    }

    fn epsilon(&self) -> f32 {
        f32::from_bits(self.epsilon_bits)
    }
}

/// Immutable context-bound handles and geometry for one Qwen-compatible
/// attention layer. The layer index is part of the plan identity so request-
/// local KV buffers cannot be selected independently at encode time.
/// Projection bias, asymmetric V, attention sinks, sliding-window execution,
/// and post-attention scaling are outside this foundation contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeAttentionPlan {
    context_id: u64,
    layer_index: usize,
    geometry: GpuNativeAttentionGeometry,
    q_projection: GpuNativeDenseWeightHandle,
    k_projection: GpuNativeDenseWeightHandle,
    v_projection: GpuNativeDenseWeightHandle,
    o_projection: GpuNativeDenseWeightHandle,
    q_norm: Option<GpuNativeAttentionNorm>,
    k_norm: Option<GpuNativeAttentionNorm>,
    rope: GpuNativeRopeHandle,
}

impl GpuNativeAttentionPlan {
    pub(crate) const fn geometry(&self) -> GpuNativeAttentionGeometry {
        self.geometry
    }

    pub(crate) const fn layer_index(&self) -> usize {
        self.layer_index
    }
}

struct GpuNativeDenseWeightChunk<B = wgpu::Buffer> {
    plan: GpuNativeDenseWeightChunkPlan,
    buffer: B,
}

struct GpuNativeDenseWeight<B = wgpu::Buffer> {
    weight_id: u64,
    key: GpuNativeDenseWeightKey,
    layout: GpuNativeDenseWeightLayout,
    chunks: Vec<GpuNativeDenseWeightChunk<B>>,
}

impl<B> GpuNativeDenseWeight<B> {
    fn handle(&self, context_id: u64) -> GpuNativeDenseWeightHandle {
        GpuNativeDenseWeightHandle {
            context_id,
            weight_id: self.weight_id,
            key: self.key.clone(),
            layout: self.layout,
        }
    }
}

struct GpuNativeDenseWeightRegistry<B = wgpu::Buffer> {
    context_id: u64,
    weights: HashMap<GpuNativeDenseWeightKey, Arc<GpuNativeDenseWeight<B>>>,
}

impl<B> GpuNativeDenseWeightRegistry<B> {
    fn new(context_id: u64) -> Self {
        Self {
            context_id,
            weights: HashMap::new(),
        }
    }

    fn insert(
        &mut self,
        weight: GpuNativeDenseWeight<B>,
    ) -> Result<GpuNativeDenseWeightHandle, GpuNativeBootstrapError> {
        if self.weights.contains_key(&weight.key) {
            return Err(GpuNativeBootstrapError::DuplicateDenseWeight {
                key: weight.key.as_str().to_string(),
            });
        }
        let weight = Arc::new(weight);
        let handle = weight.handle(self.context_id);
        self.weights.insert(weight.key.clone(), weight);
        Ok(handle)
    }

    fn resolve(
        &self,
        handle: &GpuNativeDenseWeightHandle,
    ) -> Result<Arc<GpuNativeDenseWeight<B>>, GpuNativeBootstrapError> {
        if handle.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignDenseWeightHandle);
        }
        let weight = self.weights.get(&handle.key).ok_or_else(|| {
            GpuNativeBootstrapError::MissingDenseWeight {
                key: handle.key.as_str().to_string(),
            }
        })?;
        if weight.weight_id != handle.weight_id || weight.layout != handle.layout {
            return Err(GpuNativeBootstrapError::StaleDenseWeightHandle {
                key: handle.key.as_str().to_string(),
            });
        }
        Ok(weight.clone())
    }

    fn handle_for(
        &self,
        key: &GpuNativeDenseWeightKey,
        expected_kind: GpuNativeDenseWeightKind,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<GpuNativeDenseWeightHandle, GpuNativeBootstrapError> {
        let weight =
            self.weights
                .get(key)
                .ok_or_else(|| GpuNativeBootstrapError::MissingDenseWeight {
                    key: key.as_str().to_string(),
                })?;
        if weight.layout.kind != expected_kind {
            return Err(GpuNativeBootstrapError::DenseWeightKindMismatch {
                key: key.as_str().to_string(),
                expected: expected_kind,
                actual: weight.layout.kind,
            });
        }
        if weight.layout.rows != expected_rows || weight.layout.cols != expected_cols {
            return Err(GpuNativeBootstrapError::DenseWeightShapeMismatch {
                key: key.as_str().to_string(),
                expected_rows,
                expected_cols,
                actual_rows: weight.layout.rows,
                actual_cols: weight.layout.cols,
            });
        }
        Ok(weight.handle(self.context_id))
    }

    fn resolve_rms_norm(
        &self,
        handle: &GpuNativeRmsNormHandle,
    ) -> Result<Arc<GpuNativeDenseWeight<B>>, GpuNativeBootstrapError> {
        if handle.dense.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignRmsNormHandle);
        }
        let weight = self.weights.get(&handle.dense.key).ok_or_else(|| {
            GpuNativeBootstrapError::MissingDenseWeight {
                key: handle.dense.key.as_str().to_string(),
            }
        })?;
        if weight.weight_id != handle.dense.weight_id
            || weight.layout != handle.dense.layout
            || handle.width != handle.dense.layout.cols
        {
            return Err(GpuNativeBootstrapError::StaleRmsNormHandle {
                key: handle.dense.key.as_str().to_string(),
            });
        }
        if weight.layout.kind != GpuNativeDenseWeightKind::F32
            || weight.layout.rows != 1
            || weight.layout.cols != handle.width
        {
            return Err(GpuNativeBootstrapError::StaleRmsNormHandle {
                key: handle.dense.key.as_str().to_string(),
            });
        }
        Ok(weight.clone())
    }
}

fn validate_rope_handle_with_registry<B>(
    context_id: u64,
    registry: &GpuNativeDenseWeightRegistry<B>,
    handle: &GpuNativeRopeHandle,
    expected_rope_dim: usize,
) -> Result<(), GpuNativeBootstrapError> {
    if handle.dense.context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignRopeHandle);
    }
    if handle.rope_dim != expected_rope_dim {
        return Err(GpuNativeBootstrapError::RopeDimensionMismatch {
            expected: expected_rope_dim,
            actual: handle.rope_dim,
        });
    }
    let weight = registry
        .resolve(&handle.dense)
        .map_err(|error| match error {
            GpuNativeBootstrapError::ForeignDenseWeightHandle => {
                GpuNativeBootstrapError::ForeignRopeHandle
            }
            _ => GpuNativeBootstrapError::StaleRopeHandle {
                key: handle.dense.key.as_str().to_string(),
            },
        })?;
    if weight.layout.kind != GpuNativeDenseWeightKind::F32
        || weight.layout.rows != 1
        || weight.layout.cols != handle.rope_dim / 2
        || !f32::from_bits(handle.attention_factor_bits).is_finite()
        || f32::from_bits(handle.attention_factor_bits) <= 0.0
    {
        return Err(GpuNativeBootstrapError::StaleRopeHandle {
            key: handle.dense.key.as_str().to_string(),
        });
    }
    Ok(())
}

static NEXT_GPU_NATIVE_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_GPU_NATIVE_WEIGHT_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_GPU_NATIVE_SCRATCH_ID: AtomicU64 = AtomicU64::new(1);

fn next_nonzero_id(counter: &AtomicU64, label: &str) -> u64 {
    let id = counter.fetch_add(1, Ordering::Relaxed);
    assert_ne!(id, 0, "GPU-native {label} id space exhausted");
    id
}

/// Checked byte layout for one request's initial device-resident token state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeTokenStateLayout {
    d_model: usize,
    vector_bytes: u64,
    status_bytes: u64,
    total_buffer_bytes: u64,
}

impl GpuNativeTokenStateLayout {
    pub(crate) fn try_new(d_model: usize) -> Result<Self, GpuNativeBootstrapError> {
        if d_model == 0 {
            return Err(GpuNativeBootstrapError::InvalidDModel);
        }

        let vector_bytes = d_model
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(GpuNativeBootstrapError::StateSizeOverflow { d_model })?;
        let total_buffer_bytes = vector_bytes
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(GPU_NATIVE_STATUS_BYTES))
            .ok_or(GpuNativeBootstrapError::StateSizeOverflow { d_model })?;

        Ok(Self {
            d_model,
            vector_bytes,
            status_bytes: GPU_NATIVE_STATUS_BYTES,
            total_buffer_bytes,
        })
    }

    pub(crate) const fn d_model(self) -> usize {
        self.d_model
    }

    pub(crate) const fn vector_bytes(self) -> u64 {
        self.vector_bytes
    }

    pub(crate) const fn status_bytes(self) -> u64 {
        self.status_bytes
    }

    pub(crate) const fn total_buffer_bytes(self) -> u64 {
        self.total_buffer_bytes
    }

    pub(crate) fn tensor_usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
    }

    pub(crate) fn status_usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
    }

    fn validate_for_limits(self, limits: &wgpu::Limits) -> Result<(), GpuNativeBootstrapError> {
        super::validate_startup_buffer(
            "gpu_native_hidden",
            self.vector_bytes,
            Self::tensor_usage(),
            limits,
        )?;
        super::validate_startup_buffer(
            "gpu_native_residual",
            self.vector_bytes,
            Self::tensor_usage(),
            limits,
        )?;
        super::validate_startup_buffer(
            "gpu_native_status",
            self.status_bytes,
            Self::status_usage(),
            limits,
        )?;
        Ok(())
    }
}

/// Checked request-scoped F32 device scratch for variable-width GEMV output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeScratchLayout {
    elements: usize,
    bytes: u64,
}

impl GpuNativeScratchLayout {
    pub(crate) fn try_new(elements: usize) -> Result<Self, GpuNativeBootstrapError> {
        if elements == 0 {
            return Err(GpuNativeBootstrapError::InvalidScratchElements);
        }
        if u32::try_from(elements).is_err() {
            return Err(GpuNativeBootstrapError::ScratchSizeOverflow { elements });
        }
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(GpuNativeBootstrapError::ScratchSizeOverflow { elements })?;
        Ok(Self { elements, bytes })
    }

    pub(crate) const fn elements(self) -> usize {
        self.elements
    }

    pub(crate) const fn bytes(self) -> u64 {
        self.bytes
    }

    pub(crate) fn usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
    }

    pub(crate) fn boundary_result_usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
    }

    fn validate_for_limits(self, limits: &wgpu::Limits) -> Result<(), GpuNativeBootstrapError> {
        super::validate_startup_buffer("gpu_native_scratch", self.bytes, Self::usage(), limits)?;
        Ok(())
    }

    fn validate_for_boundary_result_limits(
        self,
        limits: &wgpu::Limits,
    ) -> Result<(), GpuNativeBootstrapError> {
        super::validate_startup_buffer(
            "gpu_native_boundary_result_scratch",
            self.bytes,
            Self::boundary_result_usage(),
            limits,
        )?;
        Ok(())
    }
}

pub(crate) struct GpuNativeScratch<B = wgpu::Buffer> {
    context_id: u64,
    scratch_id: u64,
    layout: GpuNativeScratchLayout,
    buffer: B,
}

impl<B> GpuNativeScratch<B> {
    fn from_buffer(
        context_id: u64,
        scratch_id: u64,
        layout: GpuNativeScratchLayout,
        buffer: B,
    ) -> Self {
        Self {
            context_id,
            scratch_id,
            layout,
            buffer,
        }
    }

    pub(crate) const fn layout(&self) -> GpuNativeScratchLayout {
        self.layout
    }

    pub(crate) fn buffer(&self) -> &B {
        &self.buffer
    }
}

impl<B> fmt::Debug for GpuNativeScratch<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuNativeScratch")
            .field("scratch_id", &self.scratch_id)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

/// Checked request-local router storage. Logits and selected weights are F32;
/// selected expert ids are a distinct u32 allocation. Result allocations are
/// copy sources only for the ignored hardware seam and are never mappable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeRouterScratchLayout {
    geometry: GpuNativeRouterGeometry,
    logits_elements: usize,
    selected_ids_elements: usize,
    selected_weights_elements: usize,
    logits_bytes: u64,
    selected_ids_bytes: u64,
    selected_weights_bytes: u64,
    total_bytes: u64,
}

impl GpuNativeRouterScratchLayout {
    fn try_new(geometry: GpuNativeRouterGeometry) -> Result<Self, GpuNativeBootstrapError> {
        GpuNativeRouterGeometry::try_new(geometry.d_model, geometry.num_experts, geometry.top_k)?;
        let bytes = |elements: usize| {
            elements
                .checked_mul(std::mem::size_of::<u32>())
                .and_then(|value| u64::try_from(value).ok())
                .ok_or(GpuNativeBootstrapError::RouterGeometryOverflow {
                    d_model: geometry.d_model,
                    num_experts: geometry.num_experts,
                    top_k: geometry.top_k,
                })
        };
        let logits_bytes = bytes(geometry.num_experts)?;
        let selected_ids_bytes = bytes(geometry.top_k)?;
        let selected_weights_bytes = bytes(geometry.top_k)?;
        let total_bytes = logits_bytes
            .checked_add(selected_ids_bytes)
            .and_then(|value| value.checked_add(selected_weights_bytes))
            .ok_or(GpuNativeBootstrapError::RouterGeometryOverflow {
                d_model: geometry.d_model,
                num_experts: geometry.num_experts,
                top_k: geometry.top_k,
            })?;
        Ok(Self {
            geometry,
            logits_elements: geometry.num_experts,
            selected_ids_elements: geometry.top_k,
            selected_weights_elements: geometry.top_k,
            logits_bytes,
            selected_ids_bytes,
            selected_weights_bytes,
            total_bytes,
        })
    }

    pub(crate) fn logits_usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE
    }

    pub(crate) fn diagnostic_logits_usage() -> wgpu::BufferUsages {
        Self::logits_usage() | wgpu::BufferUsages::COPY_SRC
    }

    pub(crate) fn result_usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
    }

    fn validate_for_limits(self, limits: &wgpu::Limits) -> Result<(), GpuNativeBootstrapError> {
        self.validate_for_limits_with_logits_usage(limits, Self::logits_usage())
    }

    fn validate_for_diagnostic_limits(
        self,
        limits: &wgpu::Limits,
    ) -> Result<(), GpuNativeBootstrapError> {
        self.validate_for_limits_with_logits_usage(limits, Self::diagnostic_logits_usage())
    }

    fn validate_for_limits_with_logits_usage(
        self,
        limits: &wgpu::Limits,
        logits_usage: wgpu::BufferUsages,
    ) -> Result<(), GpuNativeBootstrapError> {
        super::validate_startup_buffer(
            "gpu_native_router_logits",
            self.logits_bytes,
            logits_usage,
            limits,
        )?;
        super::validate_startup_buffer(
            "gpu_native_router_selected_ids",
            self.selected_ids_bytes,
            Self::result_usage(),
            limits,
        )?;
        super::validate_startup_buffer(
            "gpu_native_router_selected_weights",
            self.selected_weights_bytes,
            Self::result_usage(),
            limits,
        )?;
        Ok(())
    }

    pub(crate) const fn geometry(self) -> GpuNativeRouterGeometry {
        self.geometry
    }

    pub(crate) const fn logits_bytes(self) -> u64 {
        self.logits_bytes
    }

    pub(crate) const fn selected_ids_bytes(self) -> u64 {
        self.selected_ids_bytes
    }

    pub(crate) const fn selected_weights_bytes(self) -> u64 {
        self.selected_weights_bytes
    }

    pub(crate) const fn total_bytes(self) -> u64 {
        self.total_bytes
    }
}

/// Opaque request-local GPU router logits and result allocations.
pub(crate) struct GpuNativeRouterScratch<B = wgpu::Buffer> {
    context_id: u64,
    scratch_id: u64,
    layout: GpuNativeRouterScratchLayout,
    logits: B,
    selected_ids: B,
    selected_weights: B,
}

impl<B> GpuNativeRouterScratch<B> {
    fn from_buffers(
        context_id: u64,
        scratch_id: u64,
        layout: GpuNativeRouterScratchLayout,
        logits: B,
        selected_ids: B,
        selected_weights: B,
    ) -> Self {
        Self {
            context_id,
            scratch_id,
            layout,
            logits,
            selected_ids,
            selected_weights,
        }
    }

    pub(crate) const fn layout(&self) -> GpuNativeRouterScratchLayout {
        self.layout
    }

    pub(crate) fn selected_ids_buffer(&self) -> &B {
        &self.selected_ids
    }

    pub(crate) fn selected_weights_buffer(&self) -> &B {
        &self.selected_weights
    }

    pub(crate) fn logits_buffer(&self) -> &B {
        &self.logits
    }
}

impl<B> fmt::Debug for GpuNativeRouterScratch<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuNativeRouterScratch")
            .field("scratch_id", &self.scratch_id)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GpuNativeQ4ExpertScratchLayout {
    geometry: GpuNativeQ4ExpertGeometry,
    resolved_locations_bytes: u64,
    activation: GpuNativeScratchLayout,
    route_outputs: GpuNativeScratchLayout,
    combined: GpuNativeScratchLayout,
    total_bytes: u64,
}

impl GpuNativeQ4ExpertScratchLayout {
    fn try_new(geometry: GpuNativeQ4ExpertGeometry) -> Result<Self, GpuNativeBootstrapError> {
        GpuNativeQ4ExpertGeometry::try_new(
            geometry.d_model,
            geometry.d_ff,
            geometry.num_experts,
            geometry.top_k,
        )?;
        let resolved_locations_bytes = geometry
            .top_k
            .checked_mul(GPU_NATIVE_EXPERT_MAPPING_ENTRY_BYTES)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(GpuNativeBootstrapError::ExpertGeometryOverflow {
                d_model: geometry.d_model,
                d_ff: geometry.d_ff,
                num_experts: geometry.num_experts,
                top_k: geometry.top_k,
            })?;
        let activation = GpuNativeScratchLayout::try_new(geometry.d_ff)?;
        let route_outputs =
            GpuNativeScratchLayout::try_new(geometry.top_k.checked_mul(geometry.d_model).ok_or(
                GpuNativeBootstrapError::ExpertGeometryOverflow {
                    d_model: geometry.d_model,
                    d_ff: geometry.d_ff,
                    num_experts: geometry.num_experts,
                    top_k: geometry.top_k,
                },
            )?)?;
        let combined = GpuNativeScratchLayout::try_new(geometry.d_model)?;
        let total_bytes = resolved_locations_bytes
            .checked_add(activation.bytes)
            .and_then(|bytes| bytes.checked_add(route_outputs.bytes))
            .and_then(|bytes| bytes.checked_add(combined.bytes))
            .ok_or(GpuNativeBootstrapError::ExpertGeometryOverflow {
                d_model: geometry.d_model,
                d_ff: geometry.d_ff,
                num_experts: geometry.num_experts,
                top_k: geometry.top_k,
            })?;
        Ok(Self {
            geometry,
            resolved_locations_bytes,
            activation,
            route_outputs,
            combined,
            total_bytes,
        })
    }

    fn resolved_usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
    }

    fn tensor_usage() -> wgpu::BufferUsages {
        wgpu::BufferUsages::STORAGE
    }

    fn validate_for_limits(self, limits: &wgpu::Limits) -> Result<(), GpuNativeBootstrapError> {
        super::validate_startup_buffer(
            "gpu_native_expert_resolved_locations",
            self.resolved_locations_bytes,
            Self::resolved_usage(),
            limits,
        )?;
        for (label, layout) in [
            ("gpu_native_expert_activation", self.activation),
            ("gpu_native_expert_route_outputs", self.route_outputs),
            ("gpu_native_expert_combined", self.combined),
        ] {
            super::validate_startup_buffer(label, layout.bytes, Self::tensor_usage(), limits)?;
        }
        Ok(())
    }

    pub(crate) const fn geometry(self) -> GpuNativeQ4ExpertGeometry {
        self.geometry
    }

    pub(crate) const fn resolved_locations_bytes(self) -> u64 {
        self.resolved_locations_bytes
    }

    pub(crate) const fn activation_bytes(self) -> u64 {
        self.activation.bytes
    }

    pub(crate) const fn route_outputs_bytes(self) -> u64 {
        self.route_outputs.bytes
    }

    pub(crate) const fn combined_bytes(self) -> u64 {
        self.combined.bytes
    }

    pub(crate) const fn total_bytes(self) -> u64 {
        self.total_bytes
    }
}

/// Request-local expert resolution and compute scratch. None of these
/// allocations is CPU mappable; `resolved_locations` is a copy source solely
/// for the ignored physical-hardware seam.
pub(crate) struct GpuNativeQ4ExpertScratch<B = wgpu::Buffer> {
    context_id: u64,
    scratch_id: u64,
    layout: GpuNativeQ4ExpertScratchLayout,
    resolved_locations: B,
    activation: GpuNativeScratch<B>,
    route_outputs: GpuNativeScratch<B>,
    combined: GpuNativeScratch<B>,
}

impl<B> GpuNativeQ4ExpertScratch<B> {
    fn from_buffers(
        context_id: u64,
        scratch_id: u64,
        layout: GpuNativeQ4ExpertScratchLayout,
        resolved_locations: B,
        activation: GpuNativeScratch<B>,
        route_outputs: GpuNativeScratch<B>,
        combined: GpuNativeScratch<B>,
    ) -> Self {
        Self {
            context_id,
            scratch_id,
            layout,
            resolved_locations,
            activation,
            route_outputs,
            combined,
        }
    }

    pub(crate) const fn layout(&self) -> GpuNativeQ4ExpertScratchLayout {
        self.layout
    }
}

impl<B> fmt::Debug for GpuNativeQ4ExpertScratch<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuNativeQ4ExpertScratch")
            .field("scratch_id", &self.scratch_id)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

/// Request-scoped, non-mappable F32 attention intermediates with geometry
/// attached, preventing projection, context, and output buffers from being
/// interchanged at the composed API.
pub struct Layer0AttentionGpuDiagnosticSink<'a> {
    pub layout: &'a crate::gpu_native_layer0_diagnostics::Layer0AttentionDiagnosticTraceLayout,
    pub staging_buffer: &'a wgpu::Buffer,
}

pub(crate) struct GpuNativeAttentionScratch<B = wgpu::Buffer> {
    context_id: u64,
    geometry: GpuNativeAttentionGeometry,
    q: GpuNativeScratch<B>,
    k: GpuNativeScratch<B>,
    v: GpuNativeScratch<B>,
    context: GpuNativeScratch<B>,
    projected: GpuNativeScratch<B>,
}

impl<B> GpuNativeAttentionScratch<B> {
    fn from_scratch(
        context_id: u64,
        geometry: GpuNativeAttentionGeometry,
        q: GpuNativeScratch<B>,
        k: GpuNativeScratch<B>,
        v: GpuNativeScratch<B>,
        context: GpuNativeScratch<B>,
        projected: GpuNativeScratch<B>,
    ) -> Self {
        Self {
            context_id,
            geometry,
            q,
            k,
            v,
            context,
            projected,
        }
    }

    pub(crate) const fn geometry(&self) -> GpuNativeAttentionGeometry {
        self.geometry
    }
}

impl<B> fmt::Debug for GpuNativeAttentionScratch<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuNativeAttentionScratch")
            .field("geometry", &self.geometry)
            .field("q", &self.q)
            .field("k", &self.k)
            .field("v", &self.v)
            .field("context", &self.context)
            .field("projected", &self.projected)
            .finish()
    }
}

struct GpuNativeKvLayer<B = wgpu::Buffer> {
    key: B,
    value: B,
}

/// Explicitly request-local F32 KV storage. Physical buffers are per-layer so
/// no single WGPU storage binding grows with the model's layer count.
pub(crate) struct GpuNativeKvState<B = wgpu::Buffer> {
    context_id: u64,
    kv_id: u64,
    layout: GpuNativeKvLayout,
    layers: Vec<GpuNativeKvLayer<B>>,
}

impl<B> GpuNativeKvState<B> {
    fn from_layers(
        context_id: u64,
        kv_id: u64,
        layout: GpuNativeKvLayout,
        layers: Vec<GpuNativeKvLayer<B>>,
    ) -> Self {
        Self {
            context_id,
            kv_id,
            layout,
            layers,
        }
    }

    pub(crate) const fn layout(&self) -> GpuNativeKvLayout {
        self.layout
    }
}

impl<B> fmt::Debug for GpuNativeKvState<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuNativeKvState")
            .field("kv_id", &self.kv_id)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

static NEXT_GPU_NATIVE_KV_ID: AtomicU64 = AtomicU64::new(1);

static NEXT_GPU_NATIVE_TOKEN_STATE_ID: AtomicU64 = AtomicU64::new(1);

fn next_gpu_native_token_state_id() -> u64 {
    let id = NEXT_GPU_NATIVE_TOKEN_STATE_ID.fetch_add(1, Ordering::Relaxed);
    assert_ne!(id, 0, "GPU-native token state id space exhausted");
    id
}

/// Opaque request-local ownership of mutable GPU-native token buffers.
///
/// The generic defaults to the production WGPU buffer type. The parameter
/// lets hardware-independent tests exercise ownership and drop behavior
/// without introducing a mock WGPU device.
pub(crate) struct GpuNativeTokenState<B = wgpu::Buffer> {
    context_id: u64,
    state_id: u64,
    layout: GpuNativeTokenStateLayout,
    hidden: B,
    residual: B,
    status: B,
}

impl<B> GpuNativeTokenState<B> {
    fn from_buffers(
        context_id: u64,
        state_id: u64,
        layout: GpuNativeTokenStateLayout,
        hidden: B,
        residual: B,
        status: B,
    ) -> Self {
        Self {
            context_id,
            state_id,
            layout,
            hidden,
            residual,
            status,
        }
    }

    pub(crate) const fn state_id(&self) -> u64 {
        self.state_id
    }

    pub(crate) const fn layout(&self) -> GpuNativeTokenStateLayout {
        self.layout
    }

    pub(crate) fn status_buffer(&self) -> &B {
        &self.status
    }

    pub(crate) fn hidden_buffer(&self) -> &B {
        &self.hidden
    }

    pub(crate) fn residual_buffer(&self) -> &B {
        &self.residual
    }
}

impl<B> fmt::Debug for GpuNativeTokenState<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuNativeTokenState")
            .field("state_id", &self.state_id)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

fn validate_token_state_owner(
    context_id: u64,
    state_context_id: u64,
) -> Result<(), GpuNativeBootstrapError> {
    if state_context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignTokenState);
    }
    Ok(())
}

fn validate_scratch_owner(
    context_id: u64,
    scratch_context_id: u64,
) -> Result<(), GpuNativeBootstrapError> {
    if scratch_context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignScratch);
    }
    Ok(())
}

fn validate_router_scratch<B>(
    context_id: u64,
    geometry: GpuNativeRouterGeometry,
    scratch: &GpuNativeRouterScratch<B>,
) -> Result<(), GpuNativeBootstrapError> {
    if scratch.context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignRouterScratch);
    }
    if scratch.layout.geometry != geometry {
        return Err(GpuNativeBootstrapError::RouterScratchGeometry {
            expected: geometry,
            actual: scratch.layout.geometry,
        });
    }
    if scratch.layout.logits_elements != geometry.num_experts {
        return Err(GpuNativeBootstrapError::RouterLogitsLength {
            expected: geometry.num_experts,
            actual: scratch.layout.logits_elements,
        });
    }
    if scratch.layout.selected_ids_elements != geometry.top_k {
        return Err(GpuNativeBootstrapError::RouterSelectedIdsLength {
            expected: geometry.top_k,
            actual: scratch.layout.selected_ids_elements,
        });
    }
    if scratch.layout.selected_weights_elements != geometry.top_k {
        return Err(GpuNativeBootstrapError::RouterSelectedWeightsLength {
            expected: geometry.top_k,
            actual: scratch.layout.selected_weights_elements,
        });
    }
    Ok(())
}

fn validate_router_plan_with_registry<B>(
    context_id: u64,
    d_model: usize,
    registry: &GpuNativeDenseWeightRegistry<B>,
    plan: &GpuNativeRouterPlan,
) -> Result<Arc<GpuNativeDenseWeight<B>>, GpuNativeBootstrapError> {
    if plan.context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignRouterPlan);
    }
    GpuNativeRouterGeometry::try_new(
        plan.geometry.d_model,
        plan.geometry.num_experts,
        plan.geometry.top_k,
    )?;
    if plan.geometry.d_model != d_model {
        return Err(GpuNativeBootstrapError::RouterDModelMismatch {
            expected: d_model,
            actual: plan.geometry.d_model,
        });
    }
    let gate = registry.resolve(&plan.gate)?;
    if gate.layout.rows != plan.geometry.num_experts || gate.layout.cols != plan.geometry.d_model {
        return Err(GpuNativeBootstrapError::RouterGateShape {
            expected_rows: plan.geometry.num_experts,
            expected_cols: plan.geometry.d_model,
            actual_rows: gate.layout.rows,
            actual_cols: gate.layout.cols,
        });
    }
    Ok(gate)
}

fn validate_q4_expert_arena<B>(
    context_id: u64,
    arena: &GpuNativeQ4ExpertArena<B>,
    limits: &wgpu::Limits,
) -> Result<(), GpuNativeBootstrapError> {
    if arena.context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignExpertArena);
    }
    let geometry = GpuNativeQ4ExpertGeometry::try_new(
        arena.geometry.d_model,
        arena.geometry.d_ff,
        arena.geometry.num_experts,
        arena.geometry.top_k,
    )?;
    let expected = GpuNativeQ4ExpertVramPlan::try_new(
        geometry,
        arena.plan.requested_expert_budget_bytes,
        limits,
    )?;
    let residency = arena.residency_snapshot();
    if arena.plan != expected
        || residency
            .resident_slots
            .checked_add(residency.installing_slots)
            .and_then(|used| used.checked_add(residency.free_slots))
            != Some(arena.plan.slot_capacity())
    {
        return Err(
            GpuNativeBootstrapError::ExpertResidentCountExceedsCapacity {
                residents: residency
                    .resident_slots
                    .saturating_add(residency.installing_slots),
                capacity: arena.plan.slot_capacity(),
            },
        );
    }
    for (bank, layout) in arena.plan.layout.banks.iter().enumerate() {
        super::validate_startup_buffer(
            &format!("gpu_native_q4_expert_bank_{bank}"),
            layout.allocation_bytes,
            GpuNativeQ4ExpertArenaLayout::weight_usage(),
            limits,
        )?;
    }
    super::validate_startup_buffer(
        "gpu_native_q4_expert_mapping",
        arena.plan.layout.mapping_bytes,
        GpuNativeQ4ExpertArenaLayout::mapping_usage(),
        limits,
    )?;
    Ok(())
}

fn validate_q4_expert_scratch<B>(
    context_id: u64,
    geometry: GpuNativeQ4ExpertGeometry,
    scratch: &GpuNativeQ4ExpertScratch<B>,
    limits: &wgpu::Limits,
) -> Result<(), GpuNativeBootstrapError> {
    if scratch.context_id != context_id
        || scratch.activation.context_id != context_id
        || scratch.route_outputs.context_id != context_id
        || scratch.combined.context_id != context_id
    {
        return Err(GpuNativeBootstrapError::ForeignExpertScratch);
    }
    let scratch_ids = [
        scratch.activation.scratch_id,
        scratch.route_outputs.scratch_id,
        scratch.combined.scratch_id,
    ];
    if scratch_ids[0] == scratch_ids[1]
        || scratch_ids[0] == scratch_ids[2]
        || scratch_ids[1] == scratch_ids[2]
    {
        return Err(GpuNativeBootstrapError::AliasedInputOutput);
    }
    if scratch.layout.geometry != geometry {
        return Err(GpuNativeBootstrapError::ExpertScratchGeometry {
            expected: geometry,
            actual: scratch.layout.geometry,
        });
    }
    let expected = GpuNativeQ4ExpertScratchLayout::try_new(geometry)?;
    if scratch.layout != expected
        || scratch.activation.layout != expected.activation
        || scratch.route_outputs.layout != expected.route_outputs
        || scratch.combined.layout != expected.combined
    {
        return Err(GpuNativeBootstrapError::ExpertScratchGeometry {
            expected: geometry,
            actual: scratch.layout.geometry,
        });
    }
    scratch.layout.validate_for_limits(limits)?;
    Ok(())
}

fn validate_router_expert_geometry<B>(
    router: &GpuNativeRouterPlan,
    arena: &GpuNativeQ4ExpertArena<B>,
) -> Result<(), GpuNativeBootstrapError> {
    if router.layer_index != arena.layer_index {
        return Err(GpuNativeBootstrapError::ExpertLayerMismatch {
            router_layer: router.layer_index,
            expert_layer: arena.layer_index,
        });
    }
    if router.geometry != arena.geometry.router_geometry() {
        return Err(GpuNativeBootstrapError::ExpertRouterGeometryMismatch {
            router: router.geometry,
            expert: arena.geometry,
        });
    }
    Ok(())
}

fn validate_residual_contribution_width(
    expected: usize,
    actual: usize,
) -> Result<(), GpuNativeBootstrapError> {
    if actual != expected {
        return Err(GpuNativeBootstrapError::ResidualContributionWidth { expected, actual });
    }
    Ok(())
}

fn validate_rms_norm_weight_width(width: usize) -> Result<(), GpuNativeBootstrapError> {
    if width == 0 {
        return Err(GpuNativeBootstrapError::InvalidRmsNormWeightWidth { width });
    }
    Ok(())
}

fn validate_rms_norm_epsilon(epsilon: f32) -> Result<(), GpuNativeBootstrapError> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(GpuNativeBootstrapError::InvalidRmsNormEpsilon {
            epsilon_bits: epsilon.to_bits(),
        });
    }
    Ok(())
}

fn validate_attention_scratch<B>(
    context_id: u64,
    geometry: GpuNativeAttentionGeometry,
    scratch: &GpuNativeAttentionScratch<B>,
) -> Result<(), GpuNativeBootstrapError> {
    if scratch.context_id != context_id
        || scratch.q.context_id != context_id
        || scratch.k.context_id != context_id
        || scratch.v.context_id != context_id
        || scratch.context.context_id != context_id
        || scratch.projected.context_id != context_id
    {
        return Err(GpuNativeBootstrapError::ForeignAttentionScratch);
    }
    let scratch_ids = [
        scratch.q.scratch_id,
        scratch.k.scratch_id,
        scratch.v.scratch_id,
        scratch.context.scratch_id,
        scratch.projected.scratch_id,
    ];
    for (index, scratch_id) in scratch_ids.iter().copied().enumerate() {
        if scratch_ids[index + 1..].contains(&scratch_id) {
            return Err(GpuNativeBootstrapError::AliasedInputOutput);
        }
    }
    for (tensor, expected, actual) in [
        (
            GpuNativeAttentionTensor::Query,
            geometry.q_width,
            scratch.q.layout.elements,
        ),
        (
            GpuNativeAttentionTensor::Key,
            geometry.kv_width,
            scratch.k.layout.elements,
        ),
        (
            GpuNativeAttentionTensor::Value,
            geometry.kv_width,
            scratch.v.layout.elements,
        ),
        (
            GpuNativeAttentionTensor::Context,
            geometry.q_width,
            scratch.context.layout.elements,
        ),
        (
            GpuNativeAttentionTensor::Output,
            geometry.d_model,
            scratch.projected.layout.elements,
        ),
    ] {
        if actual != expected {
            return Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor,
                expected,
                actual,
            });
        }
    }
    if scratch.geometry != geometry {
        return Err(GpuNativeBootstrapError::AttentionScratchGeometry {
            expected: geometry,
            actual: scratch.geometry,
        });
    }
    Ok(())
}

fn validate_attention_plan_with_registry<B>(
    context_id: u64,
    d_model: usize,
    registry: &GpuNativeDenseWeightRegistry<B>,
    plan: &GpuNativeAttentionPlan,
) -> Result<(), GpuNativeBootstrapError> {
    if plan.context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignAttentionPlan);
    }
    if plan.geometry.d_model != d_model {
        return Err(GpuNativeBootstrapError::AttentionDModelMismatch {
            expected: d_model,
            actual: plan.geometry.d_model,
        });
    }
    for (tensor, handle, rows, cols) in [
        (
            GpuNativeAttentionTensor::Query,
            &plan.q_projection,
            plan.geometry.q_width,
            plan.geometry.d_model,
        ),
        (
            GpuNativeAttentionTensor::Key,
            &plan.k_projection,
            plan.geometry.kv_width,
            plan.geometry.d_model,
        ),
        (
            GpuNativeAttentionTensor::Value,
            &plan.v_projection,
            plan.geometry.kv_width,
            plan.geometry.d_model,
        ),
        (
            GpuNativeAttentionTensor::Output,
            &plan.o_projection,
            plan.geometry.d_model,
            plan.geometry.q_width,
        ),
    ] {
        let weight = registry.resolve(handle)?;
        if weight.layout.rows != rows || weight.layout.cols != cols {
            return Err(GpuNativeBootstrapError::AttentionProjectionShape {
                tensor,
                expected_rows: rows,
                expected_cols: cols,
                actual_rows: weight.layout.rows,
                actual_cols: weight.layout.cols,
            });
        }
    }
    for (tensor, norm) in [
        (GpuNativeAttentionTensor::Query, plan.q_norm.as_ref()),
        (GpuNativeAttentionTensor::Key, plan.k_norm.as_ref()),
    ] {
        let Some(norm) = norm else {
            continue;
        };
        validate_rms_norm_epsilon(norm.epsilon())?;
        let weight = registry.resolve_rms_norm(&norm.handle)?;
        if weight.layout.cols != plan.geometry.head_dim {
            return Err(GpuNativeBootstrapError::AttentionNormWidth {
                tensor,
                expected: plan.geometry.head_dim,
                actual: weight.layout.cols,
            });
        }
    }
    validate_rope_handle_with_registry(context_id, registry, &plan.rope, plan.geometry.rope_dim)?;
    Ok(())
}

fn validate_kv_state<B>(
    context_id: u64,
    expected_width: usize,
    kv: &GpuNativeKvState<B>,
    layer: usize,
    position: usize,
) -> Result<(), GpuNativeBootstrapError> {
    if kv.context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignKvState);
    }
    if kv.layout.kv_width != expected_width {
        return Err(GpuNativeBootstrapError::KvWidth {
            expected: expected_width,
            actual: kv.layout.kv_width,
        });
    }
    kv.layout.validate_layer(layer)?;
    kv.layout.validate_position(position)?;
    Ok(())
}

fn validate_attention_kv_state<B>(
    context_id: u64,
    geometry: GpuNativeAttentionGeometry,
    kv: &GpuNativeKvState<B>,
    layer_index: usize,
    position: usize,
) -> Result<usize, GpuNativeBootstrapError> {
    if kv.context_id != context_id {
        return Err(GpuNativeBootstrapError::ForeignKvState);
    }
    if kv.layout.kv_width != geometry.kv_width {
        return Err(GpuNativeBootstrapError::KvWidth {
            expected: geometry.kv_width,
            actual: kv.layout.kv_width,
        });
    }
    if layer_index >= kv.layout.num_layers {
        return Err(GpuNativeBootstrapError::AttentionPlanLayerOutOfRange {
            layer_index,
            num_layers: kv.layout.num_layers,
        });
    }
    let seq_len = position
        .checked_add(1)
        .ok_or(GpuNativeBootstrapError::AttentionSequenceLengthOverflow { position })?;
    if seq_len == 0 || seq_len > kv.layout.max_seq_len {
        return Err(GpuNativeBootstrapError::InvalidAttentionSequenceLength {
            seq_len,
            max_seq_len: kv.layout.max_seq_len,
        });
    }
    Ok(seq_len)
}

/// Immutable, serializable evidence for GPU-native execution boundaries.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct GpuNativeExecutionSnapshot {
    pub(crate) dense_weights_registered: u64,
    pub(crate) dense_weight_chunks: u64,
    pub(crate) dense_weight_uploads: u64,
    pub(crate) dense_weight_upload_bytes: u64,
    pub(crate) dense_weight_resident_bytes: u64,
    pub(crate) dense_gemv_dispatches: u64,
    pub(crate) dense_gemv_chunk_dispatches: u64,
    pub(crate) embedding_dispatches: u64,
    pub(crate) rms_norm_dispatches: u64,
    pub(crate) rms_norm_groups: u64,
    pub(crate) rms_norm_state_dispatches: u64,
    pub(crate) rms_norm_scratch_dispatches: u64,
    pub(crate) residual_add_dispatches: u64,
    pub(crate) rope_parameters_registered: u64,
    pub(crate) rope_parameter_uploads: u64,
    pub(crate) rope_parameter_upload_bytes: u64,
    pub(crate) attention_prepare_dispatches: u64,
    pub(crate) q_projection_dispatches: u64,
    pub(crate) k_projection_dispatches: u64,
    pub(crate) v_projection_dispatches: u64,
    pub(crate) rope_dispatches: u64,
    pub(crate) rope_groups: u64,
    pub(crate) kv_appends: u64,
    pub(crate) causal_attention_dispatches: u64,
    pub(crate) o_projection_dispatches: u64,
    pub(crate) attention_complete_dispatches: u64,
    pub(crate) router_logit_dispatches: u64,
    pub(crate) router_topk_dispatches: u64,
    pub(crate) expert_slots_registered: u64,
    pub(crate) expert_weight_upload_bytes: u64,
    pub(crate) expert_route_resolve_dispatches: u64,
    pub(crate) q4_expert_gate_up_dispatches: u64,
    pub(crate) q4_expert_down_dispatches: u64,
    pub(crate) expert_combine_dispatches: u64,
    pub(crate) tokens_submitted: u64,
    pub(crate) tokens_completed: u64,
    pub(crate) layers_encoded: u64,
    pub(crate) queue_submissions: u64,
    pub(crate) token_boundary_maps: u64,
    pub(crate) token_boundary_readbacks: u64,
    pub(crate) intermediate_maps: u64,
    pub(crate) intermediate_readbacks: u64,
    pub(crate) cpu_layer_reentries: u64,
    pub(crate) cpu_attention_calls: u64,
    pub(crate) cpu_kv_mutations: u64,
    pub(crate) cpu_router_calls: u64,
    pub(crate) cpu_expert_combines: u64,
    pub(crate) expert_slot_misses: u64,
    pub(crate) device_loss_failures: u64,
    pub(crate) numerical_failures: u64,
}

#[derive(Debug, Default)]
struct GpuNativeExecutionCounters {
    dense_weights_registered: AtomicU64,
    dense_weight_chunks: AtomicU64,
    dense_weight_uploads: AtomicU64,
    dense_weight_upload_bytes: AtomicU64,
    dense_weight_resident_bytes: AtomicU64,
    dense_gemv_dispatches: AtomicU64,
    dense_gemv_chunk_dispatches: AtomicU64,
    embedding_dispatches: AtomicU64,
    rms_norm_dispatches: AtomicU64,
    rms_norm_groups: AtomicU64,
    rms_norm_state_dispatches: AtomicU64,
    rms_norm_scratch_dispatches: AtomicU64,
    residual_add_dispatches: AtomicU64,
    rope_parameters_registered: AtomicU64,
    rope_parameter_uploads: AtomicU64,
    rope_parameter_upload_bytes: AtomicU64,
    attention_prepare_dispatches: AtomicU64,
    q_projection_dispatches: AtomicU64,
    k_projection_dispatches: AtomicU64,
    v_projection_dispatches: AtomicU64,
    rope_dispatches: AtomicU64,
    rope_groups: AtomicU64,
    kv_appends: AtomicU64,
    causal_attention_dispatches: AtomicU64,
    o_projection_dispatches: AtomicU64,
    attention_complete_dispatches: AtomicU64,
    router_logit_dispatches: AtomicU64,
    router_topk_dispatches: AtomicU64,
    expert_slots_registered: AtomicU64,
    expert_weight_upload_bytes: AtomicU64,
    expert_route_resolve_dispatches: AtomicU64,
    q4_expert_gate_up_dispatches: AtomicU64,
    q4_expert_down_dispatches: AtomicU64,
    expert_combine_dispatches: AtomicU64,
    tokens_submitted: AtomicU64,
    tokens_completed: AtomicU64,
    layers_encoded: AtomicU64,
    queue_submissions: AtomicU64,
    token_boundary_maps: AtomicU64,
    token_boundary_readbacks: AtomicU64,
    intermediate_maps: AtomicU64,
    intermediate_readbacks: AtomicU64,
    cpu_layer_reentries: AtomicU64,
    cpu_attention_calls: AtomicU64,
    cpu_kv_mutations: AtomicU64,
    cpu_router_calls: AtomicU64,
    cpu_expert_combines: AtomicU64,
    expert_slot_misses: AtomicU64,
    device_loss_failures: AtomicU64,
    numerical_failures: AtomicU64,
}

impl GpuNativeExecutionCounters {
    fn snapshot(&self) -> GpuNativeExecutionSnapshot {
        GpuNativeExecutionSnapshot {
            dense_weights_registered: self.dense_weights_registered.load(Ordering::Relaxed),
            dense_weight_chunks: self.dense_weight_chunks.load(Ordering::Relaxed),
            dense_weight_uploads: self.dense_weight_uploads.load(Ordering::Relaxed),
            dense_weight_upload_bytes: self.dense_weight_upload_bytes.load(Ordering::Relaxed),
            dense_weight_resident_bytes: self.dense_weight_resident_bytes.load(Ordering::Relaxed),
            dense_gemv_dispatches: self.dense_gemv_dispatches.load(Ordering::Relaxed),
            dense_gemv_chunk_dispatches: self.dense_gemv_chunk_dispatches.load(Ordering::Relaxed),
            embedding_dispatches: self.embedding_dispatches.load(Ordering::Relaxed),
            rms_norm_dispatches: self.rms_norm_dispatches.load(Ordering::Relaxed),
            rms_norm_groups: self.rms_norm_groups.load(Ordering::Relaxed),
            rms_norm_state_dispatches: self.rms_norm_state_dispatches.load(Ordering::Relaxed),
            rms_norm_scratch_dispatches: self.rms_norm_scratch_dispatches.load(Ordering::Relaxed),
            residual_add_dispatches: self.residual_add_dispatches.load(Ordering::Relaxed),
            rope_parameters_registered: self.rope_parameters_registered.load(Ordering::Relaxed),
            rope_parameter_uploads: self.rope_parameter_uploads.load(Ordering::Relaxed),
            rope_parameter_upload_bytes: self.rope_parameter_upload_bytes.load(Ordering::Relaxed),
            attention_prepare_dispatches: self.attention_prepare_dispatches.load(Ordering::Relaxed),
            q_projection_dispatches: self.q_projection_dispatches.load(Ordering::Relaxed),
            k_projection_dispatches: self.k_projection_dispatches.load(Ordering::Relaxed),
            v_projection_dispatches: self.v_projection_dispatches.load(Ordering::Relaxed),
            rope_dispatches: self.rope_dispatches.load(Ordering::Relaxed),
            rope_groups: self.rope_groups.load(Ordering::Relaxed),
            kv_appends: self.kv_appends.load(Ordering::Relaxed),
            causal_attention_dispatches: self.causal_attention_dispatches.load(Ordering::Relaxed),
            o_projection_dispatches: self.o_projection_dispatches.load(Ordering::Relaxed),
            attention_complete_dispatches: self
                .attention_complete_dispatches
                .load(Ordering::Relaxed),
            router_logit_dispatches: self.router_logit_dispatches.load(Ordering::Relaxed),
            router_topk_dispatches: self.router_topk_dispatches.load(Ordering::Relaxed),
            expert_slots_registered: self.expert_slots_registered.load(Ordering::Relaxed),
            expert_weight_upload_bytes: self.expert_weight_upload_bytes.load(Ordering::Relaxed),
            expert_route_resolve_dispatches: self
                .expert_route_resolve_dispatches
                .load(Ordering::Relaxed),
            q4_expert_gate_up_dispatches: self.q4_expert_gate_up_dispatches.load(Ordering::Relaxed),
            q4_expert_down_dispatches: self.q4_expert_down_dispatches.load(Ordering::Relaxed),
            expert_combine_dispatches: self.expert_combine_dispatches.load(Ordering::Relaxed),
            tokens_submitted: self.tokens_submitted.load(Ordering::Relaxed),
            tokens_completed: self.tokens_completed.load(Ordering::Relaxed),
            layers_encoded: self.layers_encoded.load(Ordering::Relaxed),
            queue_submissions: self.queue_submissions.load(Ordering::Relaxed),
            token_boundary_maps: self.token_boundary_maps.load(Ordering::Relaxed),
            token_boundary_readbacks: self.token_boundary_readbacks.load(Ordering::Relaxed),
            intermediate_maps: self.intermediate_maps.load(Ordering::Relaxed),
            intermediate_readbacks: self.intermediate_readbacks.load(Ordering::Relaxed),
            cpu_layer_reentries: self.cpu_layer_reentries.load(Ordering::Relaxed),
            cpu_attention_calls: self.cpu_attention_calls.load(Ordering::Relaxed),
            cpu_kv_mutations: self.cpu_kv_mutations.load(Ordering::Relaxed),
            cpu_router_calls: self.cpu_router_calls.load(Ordering::Relaxed),
            cpu_expert_combines: self.cpu_expert_combines.load(Ordering::Relaxed),
            expert_slot_misses: self.expert_slot_misses.load(Ordering::Relaxed),
            device_loss_failures: self.device_loss_failures.load(Ordering::Relaxed),
            numerical_failures: self.numerical_failures.load(Ordering::Relaxed),
        }
    }

    fn record_dense_weight_registration(&self, chunks: u64, allocation_bytes: u64) {
        self.dense_weights_registered
            .fetch_add(1, Ordering::Relaxed);
        self.dense_weight_chunks
            .fetch_add(chunks, Ordering::Relaxed);
        self.dense_weight_uploads
            .fetch_add(chunks, Ordering::Relaxed);
        self.dense_weight_upload_bytes
            .fetch_add(allocation_bytes, Ordering::Relaxed);
        self.dense_weight_resident_bytes
            .fetch_add(allocation_bytes, Ordering::Relaxed);
    }

    fn record_dense_gemv_dispatch(&self, chunks: u64) {
        self.dense_gemv_dispatches.fetch_add(1, Ordering::Relaxed);
        self.dense_gemv_chunk_dispatches
            .fetch_add(chunks, Ordering::Relaxed);
    }

    fn record_embedding_dispatch(&self) {
        self.embedding_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    fn record_rms_norm_state_dispatch(&self, groups: u64) {
        self.rms_norm_dispatches.fetch_add(1, Ordering::Relaxed);
        self.rms_norm_groups.fetch_add(groups, Ordering::Relaxed);
        self.rms_norm_state_dispatches
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_rms_norm_scratch_dispatch(&self, groups: u64) {
        self.rms_norm_dispatches.fetch_add(1, Ordering::Relaxed);
        self.rms_norm_groups.fetch_add(groups, Ordering::Relaxed);
        self.rms_norm_scratch_dispatches
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_residual_add_dispatch(&self) {
        self.residual_add_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    fn record_rope_registration(&self, upload_bytes: u64) {
        self.rope_parameters_registered
            .fetch_add(1, Ordering::Relaxed);
        self.rope_parameter_uploads.fetch_add(1, Ordering::Relaxed);
        self.rope_parameter_upload_bytes
            .fetch_add(upload_bytes, Ordering::Relaxed);
    }

    fn record_attention_prepare_dispatch(&self) {
        self.attention_prepare_dispatches
            .fetch_add(1, Ordering::Relaxed);
        self.q_projection_dispatches.fetch_add(1, Ordering::Relaxed);
        self.k_projection_dispatches.fetch_add(1, Ordering::Relaxed);
        self.v_projection_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    fn record_rope_dispatch(&self, groups: u64) {
        self.rope_dispatches.fetch_add(1, Ordering::Relaxed);
        self.rope_groups.fetch_add(groups, Ordering::Relaxed);
    }

    fn record_kv_append(&self) {
        self.kv_appends.fetch_add(1, Ordering::Relaxed);
    }

    fn record_causal_attention_dispatch(&self) {
        self.causal_attention_dispatches
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_attention_complete_dispatch(&self) {
        self.o_projection_dispatches.fetch_add(1, Ordering::Relaxed);
        self.attention_complete_dispatches
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_router_dispatch(&self) {
        self.router_logit_dispatches.fetch_add(1, Ordering::Relaxed);
        self.router_topk_dispatches.fetch_add(1, Ordering::Relaxed);
    }

    fn record_expert_arena_registration(&self, slots: u64, upload_bytes: u64) {
        self.expert_slots_registered
            .fetch_add(slots, Ordering::Relaxed);
        self.expert_weight_upload_bytes
            .fetch_add(upload_bytes, Ordering::Relaxed);
    }

    fn record_expert_residency_upload(&self, upload_bytes: u64) {
        self.expert_weight_upload_bytes
            .fetch_add(upload_bytes, Ordering::Relaxed);
    }

    fn record_expert_dispatches(&self, routes: u64) {
        self.expert_route_resolve_dispatches
            .fetch_add(1, Ordering::Relaxed);
        self.q4_expert_gate_up_dispatches
            .fetch_add(routes, Ordering::Relaxed);
        self.q4_expert_down_dispatches
            .fetch_add(routes, Ordering::Relaxed);
        self.expert_combine_dispatches
            .fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn record_token_boundary_readback(&self) {
        self.token_boundary_readbacks
            .fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn record_intermediate_readback(&self) {
        self.intermediate_readbacks.fetch_add(1, Ordering::Relaxed);
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeGemvPushConstants {
    rows: u32,
    cols: u32,
    global_row_base: u32,
    q8_first_block: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeEmbeddingPushConstants {
    local_row: u32,
    global_row: u32,
    cols: u32,
    q8_first_block: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeRmsNormPushConstants {
    groups: u32,
    group_width: u32,
    epsilon_bits: u32,
    _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeRopePushConstants {
    groups: u32,
    head_dim: u32,
    rope_dim: u32,
    position: u32,
    attention_factor_bits: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeKvAppendPushConstants {
    width: u32,
    position: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeAttentionPushConstants {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeRouterPushConstants {
    num_experts: u32,
    top_k: u32,
    _reserved_0: u32,
    _reserved_1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeExpertResolvePushConstants {
    num_experts: u32,
    top_k: u32,
    active_banks: u32,
    _reserved: u32,
    bank_slots: [u32; MAX_GPU_NATIVE_EXPERT_BANKS],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeQ4ExpertPushConstants {
    d_model: u32,
    d_ff: u32,
    blocks_per_projection: u32,
    slot_stride_bytes: u32,
    route_slot: u32,
    top_k: u32,
    swiglu_limit: f32,
    _reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeExpertCombinePushConstants {
    d_model: u32,
    top_k: u32,
    _reserved_0: u32,
    _reserved_1: u32,
}

struct GpuNativeDensePipelines {
    gemv_bind_group_layout: wgpu::BindGroupLayout,
    embedding_bind_group_layout: wgpu::BindGroupLayout,
    f32_gemv: wgpu::ComputePipeline,
    q8_0_gemv: wgpu::ComputePipeline,
    f32_embedding: wgpu::ComputePipeline,
    q8_0_embedding: wgpu::ComputePipeline,
}

impl GpuNativeDensePipelines {
    fn new(device: &wgpu::Device) -> Self {
        let read_only_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let read_write_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let gemv_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_dense_gemv_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                ],
            });
        let embedding_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_embedding_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                ],
            });
        let gemv_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_native_dense_gemv_pipeline_layout"),
            bind_group_layouts: &[&gemv_bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });
        let embedding_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_native_embedding_pipeline_layout"),
                bind_group_layouts: &[&embedding_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let gemv_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_dense_gemv_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_DENSE_GEMV_SHADER.into()),
        });
        let embedding_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_embedding_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_EMBEDDING_SHADER.into()),
        });
        let pipeline = |label: &'static str,
                        layout: &wgpu::PipelineLayout,
                        module: &wgpu::ShaderModule,
                        entry_point: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module,
                entry_point,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        };
        let f32_gemv = pipeline(
            "gpu_native_f32_gemv_pipeline",
            &gemv_pipeline_layout,
            &gemv_module,
            "f32_gemv_main",
        );
        let q8_0_gemv = pipeline(
            "gpu_native_q8_0_gemv_pipeline",
            &gemv_pipeline_layout,
            &gemv_module,
            "q8_0_gemv_main",
        );
        let f32_embedding = pipeline(
            "gpu_native_f32_embedding_pipeline",
            &embedding_pipeline_layout,
            &embedding_module,
            "f32_embedding_main",
        );
        let q8_0_embedding = pipeline(
            "gpu_native_q8_0_embedding_pipeline",
            &embedding_pipeline_layout,
            &embedding_module,
            "q8_0_embedding_main",
        );
        Self {
            gemv_bind_group_layout,
            embedding_bind_group_layout,
            f32_gemv,
            q8_0_gemv,
            f32_embedding,
            q8_0_embedding,
        }
    }
}

struct GpuNativeStatePipelines {
    rms_capture_bind_group_layout: wgpu::BindGroupLayout,
    rms_in_place_bind_group_layout: wgpu::BindGroupLayout,
    rms_capture: wgpu::ComputePipeline,
    rms_in_place: wgpu::ComputePipeline,
    residual_add: wgpu::ComputePipeline,
}

impl GpuNativeStatePipelines {
    fn new(device: &wgpu::Device) -> Self {
        let read_only_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let read_write_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let rms_capture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_rms_capture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                ],
            });
        let rms_in_place_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_rms_in_place_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                ],
            });
        let pipeline_layout = |label, bind_group_layout: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            })
        };
        let capture_pipeline_layout = pipeline_layout(
            "gpu_native_rms_capture_pipeline_layout",
            &rms_capture_bind_group_layout,
        );
        let in_place_pipeline_layout = pipeline_layout(
            "gpu_native_rms_in_place_pipeline_layout",
            &rms_in_place_bind_group_layout,
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_rmsnorm_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_RMSNORM_SHADER.into()),
        });
        let pipeline =
            |label: &'static str, layout: &wgpu::PipelineLayout, entry_point: &'static str| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    module: &module,
                    entry_point,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                })
            };
        let rms_capture = pipeline(
            "gpu_native_rms_capture_pipeline",
            &capture_pipeline_layout,
            "rms_norm_capture_main",
        );
        let rms_in_place = pipeline(
            "gpu_native_rms_in_place_pipeline",
            &in_place_pipeline_layout,
            "rms_norm_in_place_main",
        );
        let residual_add = pipeline(
            "gpu_native_residual_add_pipeline",
            &capture_pipeline_layout,
            "residual_add_main",
        );
        Self {
            rms_capture_bind_group_layout,
            rms_in_place_bind_group_layout,
            rms_capture,
            rms_in_place,
            residual_add,
        }
    }
}

struct GpuNativeAttentionPipelines {
    rope_bind_group_layout: wgpu::BindGroupLayout,
    kv_append_bind_group_layout: wgpu::BindGroupLayout,
    causal_attention_bind_group_layout: wgpu::BindGroupLayout,
    rope: wgpu::ComputePipeline,
    kv_append: wgpu::ComputePipeline,
    causal_attention: wgpu::ComputePipeline,
}

impl GpuNativeAttentionPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let read_only_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let read_write_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let rope_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_rope_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                ],
            });
        let kv_append_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_kv_append_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                ],
            });
        let causal_attention_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_causal_attention_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_only_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: read_write_storage,
                        count: None,
                    },
                ],
            });
        let rope_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_native_rope_pipeline_layout"),
            bind_group_layouts: &[&rope_bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..20,
            }],
        });
        let kv_append_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_native_kv_append_pipeline_layout"),
                bind_group_layouts: &[&kv_append_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..8,
                }],
            });
        let causal_attention_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_native_causal_attention_pipeline_layout"),
                bind_group_layouts: &[&causal_attention_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..16,
                }],
            });
        let rope_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_rope_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_ROPE_SHADER.into()),
        });
        let kv_append_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_kv_append_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_KV_APPEND_SHADER.into()),
        });
        let causal_attention_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_causal_attention_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_ATTENTION_SHADER.into()),
        });
        let rope = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_native_rope_pipeline"),
            layout: Some(&rope_pipeline_layout),
            module: &rope_module,
            entry_point: "rope_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let kv_append = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_native_kv_append_pipeline"),
            layout: Some(&kv_append_pipeline_layout),
            module: &kv_append_module,
            entry_point: "kv_append_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let causal_attention = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_native_causal_attention_pipeline"),
            layout: Some(&causal_attention_pipeline_layout),
            module: &causal_attention_module,
            entry_point: "causal_attention_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        Self {
            rope_bind_group_layout,
            kv_append_bind_group_layout,
            causal_attention_bind_group_layout,
            rope,
            kv_append,
            causal_attention,
        }
    }
}

struct GpuNativeRouterPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    topk: wgpu::ComputePipeline,
}

impl GpuNativeRouterPipeline {
    fn new(device: &wgpu::Device) -> Self {
        let read_only_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let read_write_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_native_router_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: read_only_storage,
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: read_write_storage,
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: read_write_storage,
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: read_write_storage,
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_native_router_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_router_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_ROUTER_SHADER.into()),
        });
        let topk = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_native_router_topk_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "router_topk_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        Self {
            bind_group_layout,
            topk,
        }
    }
}

struct GpuNativeQ4ExpertPipelines {
    route_bind_group_layout: wgpu::BindGroupLayout,
    expert_bind_group_layout: wgpu::BindGroupLayout,
    combine_bind_group_layout: wgpu::BindGroupLayout,
    control_empty_bind_group: wgpu::BindGroup,
    route_resolve: wgpu::ComputePipeline,
    gate_up: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
    validate: wgpu::ComputePipeline,
    combine: wgpu::ComputePipeline,
    contain: wgpu::ComputePipeline,
}

struct GpuNativeStatusControlPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    clear_retryable: wgpu::ComputePipeline,
}

impl GpuNativeStatusControlPipeline {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_native_status_control_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_native_status_control_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_status_control_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_STATUS_CONTROL_SHADER.into()),
        });
        let clear_retryable = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_native_clear_retryable_status_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "clear_retryable_expert_residency_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        Self {
            bind_group_layout,
            clear_retryable,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuNativeGreedyArgmaxPushConstants {
    vocab_size: u32,
    _reserved_0: u32,
    _reserved_1: u32,
    _reserved_2: u32,
}

struct GpuNativeGreedyArgmaxPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
}

impl GpuNativeGreedyArgmaxPipeline {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_native_greedy_argmax_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_native_greedy_argmax_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_greedy_argmax_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_GREEDY_ARGMAX_SHADER.into()),
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_native_greedy_argmax_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "greedy_argmax_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        Self {
            bind_group_layout,
            compute_pipeline,
        }
    }
}

impl GpuNativeQ4ExpertPipelines {
    fn try_new(device: &wgpu::Device) -> Result<Self, GpuNativeBootstrapError> {
        validate_q4_expert_pipeline_limits(&device.limits())?;
        let read_only_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let read_write_storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let route_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_expert_route_bind_group_layout"),
                entries: &[
                    entry(0, read_only_storage),
                    entry(1, read_only_storage),
                    entry(2, read_write_storage),
                    entry(3, read_write_storage),
                ],
            });
        let expert_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_q4_expert_bind_group_layout"),
                entries: &[
                    entry(0, read_only_storage),
                    entry(1, read_only_storage),
                    entry(2, read_only_storage),
                    entry(3, read_only_storage),
                    entry(4, read_only_storage),
                    entry(5, read_only_storage),
                    entry(6, read_write_storage),
                    entry(7, read_write_storage),
                ],
            });
        let combine_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_expert_combine_bind_group_layout"),
                entries: &[
                    entry(0, read_only_storage),
                    entry(1, read_only_storage),
                    entry(2, read_only_storage),
                    entry(3, read_write_storage),
                    entry(4, read_write_storage),
                ],
            });
        let empty_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_native_expert_control_empty_group_0_layout"),
                entries: &[],
            });
        let control_empty_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_expert_control_empty_group_0"),
            layout: &empty_bind_group_layout,
            entries: &[],
        });
        let push_range = [wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..GPU_NATIVE_EXPERT_PUSH_CONSTANT_BYTES,
        }];
        let route_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_native_expert_route_pipeline_layout"),
                bind_group_layouts: &[&route_bind_group_layout],
                push_constant_ranges: &push_range,
            });
        let expert_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_native_q4_expert_pipeline_layout"),
                bind_group_layouts: &[&expert_bind_group_layout],
                push_constant_ranges: &push_range,
            });
        let combine_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gpu_native_expert_combine_pipeline_layout"),
                bind_group_layouts: &[&empty_bind_group_layout, &combine_bind_group_layout],
                push_constant_ranges: &push_range,
            });
        let expert_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_q4_expert_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_Q4_EXPERT_SHADER.into()),
        });
        let control_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_expert_control_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_EXPERT_CONTROL_SHADER.into()),
        });
        let pipeline = |label, layout: &wgpu::PipelineLayout, module, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module,
                entry_point,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            })
        };
        Ok(Self {
            route_bind_group_layout,
            expert_bind_group_layout,
            combine_bind_group_layout,
            control_empty_bind_group,
            route_resolve: pipeline(
                "gpu_native_expert_route_resolve_pipeline",
                &route_pipeline_layout,
                &control_module,
                "expert_route_resolve_main",
            ),
            gate_up: pipeline(
                "gpu_native_q4_expert_gate_up_pipeline",
                &expert_pipeline_layout,
                &expert_module,
                "q4_expert_gate_up_main",
            ),
            down: pipeline(
                "gpu_native_q4_expert_down_pipeline",
                &expert_pipeline_layout,
                &expert_module,
                "q4_expert_down_main",
            ),
            validate: pipeline(
                "gpu_native_expert_validate_pipeline",
                &combine_pipeline_layout,
                &control_module,
                "expert_validate_main",
            ),
            combine: pipeline(
                "gpu_native_expert_combine_pipeline",
                &combine_pipeline_layout,
                &control_module,
                "expert_combine_main",
            ),
            contain: pipeline(
                "gpu_native_expert_contain_pipeline",
                &combine_pipeline_layout,
                &control_module,
                "expert_contain_main",
            ),
        })
    }
}

/// Internal bootstrap for future GPU-owned token execution.
///
/// Retaining the exact `Arc<BackendBox>` keeps the authoritative non-cloneable
/// WGPU `Device` and `Queue` alive. It does not request or select hardware and
/// it is intentionally absent from all current execution-plan resolution.
pub(crate) struct GpuNativeExecutorContext {
    context_id: u64,
    authoritative_backend: Arc<BackendBox>,
    device_identity: GpuDeviceIdentity,
    layout: GpuNativeTokenStateLayout,
    dense_weights: ParkingMutex<GpuNativeDenseWeightRegistry>,
    dense_pipelines: GpuNativeDensePipelines,
    state_pipelines: GpuNativeStatePipelines,
    attention_pipelines: GpuNativeAttentionPipelines,
    router_pipeline: GpuNativeRouterPipeline,
    q4_expert_pipelines: GpuNativeQ4ExpertPipelines,
    status_control_pipeline: GpuNativeStatusControlPipeline,
    greedy_argmax_pipeline: GpuNativeGreedyArgmaxPipeline,
    counters: GpuNativeExecutionCounters,
}

impl GpuNativeExecutorContext {
    pub(super) fn try_new(
        authoritative_backend: Arc<BackendBox>,
        d_model: usize,
    ) -> Result<Self, GpuNativeBootstrapError> {
        let gpu = match authoritative_backend.as_ref() {
            BackendBox::Gpu(gpu) => gpu,
            BackendBox::Cpu(_) => return Err(GpuNativeBootstrapError::GpuBackendUnavailable),
            #[cfg(test)]
            BackendBox::TestGpu(_) => {
                return Err(GpuNativeBootstrapError::GpuBackendUnavailable);
            }
        };
        if let Some(detail) = gpu.device_loss.detail() {
            return Err(GpuNativeBootstrapError::DeviceLost { detail });
        }

        let layout = GpuNativeTokenStateLayout::try_new(d_model)?;
        layout.validate_for_limits(&gpu.device.limits())?;
        let device_identity = gpu.gpu_device_identity();
        let context_id = next_nonzero_id(&NEXT_GPU_NATIVE_CONTEXT_ID, "context");
        let dense_pipelines = GpuNativeDensePipelines::new(&gpu.device);
        let state_pipelines = GpuNativeStatePipelines::new(&gpu.device);
        let attention_pipelines = GpuNativeAttentionPipelines::new(&gpu.device);
        let router_pipeline = GpuNativeRouterPipeline::new(&gpu.device);
        let q4_expert_pipelines = GpuNativeQ4ExpertPipelines::try_new(&gpu.device)?;
        let status_control_pipeline = GpuNativeStatusControlPipeline::new(&gpu.device);
        let greedy_argmax_pipeline = GpuNativeGreedyArgmaxPipeline::new(&gpu.device);

        Ok(Self {
            context_id,
            authoritative_backend,
            device_identity,
            layout,
            dense_weights: ParkingMutex::new(GpuNativeDenseWeightRegistry::new(context_id)),
            dense_pipelines,
            state_pipelines,
            attention_pipelines,
            router_pipeline,
            q4_expert_pipelines,
            status_control_pipeline,
            greedy_argmax_pipeline,
            counters: GpuNativeExecutionCounters::default(),
        })
    }

    pub(crate) fn create_token_state(
        &self,
    ) -> Result<GpuNativeTokenState, GpuNativeBootstrapError> {
        let gpu = match self.authoritative_backend.as_ref() {
            BackendBox::Gpu(gpu) => gpu,
            BackendBox::Cpu(_) => return Err(GpuNativeBootstrapError::GpuBackendUnavailable),
            #[cfg(test)]
            BackendBox::TestGpu(_) => {
                return Err(GpuNativeBootstrapError::GpuBackendUnavailable);
            }
        };
        if let Some(detail) = gpu.device_loss.detail() {
            return Err(GpuNativeBootstrapError::DeviceLost { detail });
        }

        let state_id = next_gpu_native_token_state_id();
        let hidden = create_startup_buffer(
            &gpu.device,
            &format!("gpu_native_state_{state_id}_hidden"),
            self.layout.vector_bytes,
            GpuNativeTokenStateLayout::tensor_usage(),
        )?;
        let residual = create_startup_buffer(
            &gpu.device,
            &format!("gpu_native_state_{state_id}_residual"),
            self.layout.vector_bytes,
            GpuNativeTokenStateLayout::tensor_usage(),
        )?;
        let status = create_startup_buffer(
            &gpu.device,
            &format!("gpu_native_state_{state_id}_status"),
            self.layout.status_bytes,
            GpuNativeTokenStateLayout::status_usage(),
        )?;

        Ok(GpuNativeTokenState::from_buffers(
            self.context_id,
            state_id,
            self.layout,
            hidden,
            residual,
            status,
        ))
    }

    /// Record the future replay boundary into the caller-owned encoder.
    /// Only the expert-residency retry bit is cleared; fatal and unknown bits
    /// remain latched. This method never submits, polls, maps, or reads back.
    pub(crate) fn encode_clear_retryable_status(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        state: &GpuNativeTokenState,
    ) -> Result<(), GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        validate_token_state_owner(self.context_id, state.context_id)?;
        state.layout.validate_for_limits(&gpu.device.limits())?;
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_clear_retryable_status_bind_group"),
            layout: &self.status_control_pipeline.bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: state.status.as_entire_binding(),
            }],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_clear_retryable_status_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.status_control_pipeline.clear_retryable);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        Ok(())
    }

    pub(crate) fn device_limits(&self) -> Result<wgpu::Limits, GpuNativeBootstrapError> {
        Ok(self.authoritative_gpu()?.device.limits())
    }

    /// Register one immutable model-scoped dense tensor and upload its payload
    /// exactly once per physical chunk. This is the only dense-weight upload path in the
    /// GPU-native plane; encoded GEMV and embedding calls only bind this
    /// persistent model-scoped storage.
    pub(crate) fn register_dense_weight(
        &self,
        key: GpuNativeDenseWeightKey,
        weight: &DenseWeight,
    ) -> Result<GpuNativeDenseWeightHandle, GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        let layout = GpuNativeDenseWeightLayout::from_weight(weight)?;
        let plan = GpuNativeDenseWeightPlan::try_new(layout, &gpu.device.limits())?;

        // Serialize the duplicate check through insertion so two startup
        // registrars cannot both upload the same stable key.
        let mut registry = self.dense_weights.lock();
        if registry.weights.contains_key(&key) {
            return Err(GpuNativeBootstrapError::DuplicateDenseWeight {
                key: key.as_str().to_string(),
            });
        }
        for (index, chunk) in plan.chunks.iter().enumerate() {
            super::validate_startup_buffer(
                &format!("gpu_native_dense_weight_{}_chunk_{index}", key.as_str()),
                chunk.allocation_bytes,
                GpuNativeDenseWeightLayout::usage(),
                &gpu.device.limits(),
            )?;
        }

        let mut chunks = Vec::with_capacity(plan.chunks.len());
        for (index, chunk_plan) in plan.chunks.iter().copied().enumerate() {
            let label = format!("gpu_native_dense_weight_{}_chunk_{index}", key.as_str());
            let buffer = create_startup_buffer(
                &gpu.device,
                &label,
                chunk_plan.allocation_bytes,
                GpuNativeDenseWeightLayout::usage(),
            )?;
            match weight {
                DenseWeight::F32 { values, .. } => {
                    let value_start = chunk_plan.payload_offset_bytes / std::mem::size_of::<f32>();
                    let value_count =
                        chunk_plan.payload_bytes as usize / std::mem::size_of::<f32>();
                    gpu.queue.write_buffer(
                        &buffer,
                        0,
                        bytemuck::cast_slice(&values[value_start..value_start + value_count]),
                    );
                }
                DenseWeight::Q8_0 { bytes, .. }
                    if chunk_plan.payload_bytes == chunk_plan.allocation_bytes =>
                {
                    let start = chunk_plan.payload_offset_bytes;
                    let end = start + chunk_plan.payload_bytes as usize;
                    gpu.queue.write_buffer(&buffer, 0, &bytes[start..end]);
                }
                DenseWeight::Q8_0 { bytes, .. } => {
                    let start = chunk_plan.payload_offset_bytes;
                    let end = start + chunk_plan.payload_bytes as usize;
                    let mut upload = Vec::with_capacity(chunk_plan.allocation_bytes as usize);
                    upload.extend_from_slice(&bytes[start..end]);
                    upload.resize(chunk_plan.allocation_bytes as usize, 0);
                    gpu.queue.write_buffer(&buffer, 0, &upload);
                }
            }
            chunks.push(GpuNativeDenseWeightChunk {
                plan: chunk_plan,
                buffer,
            });
        }
        let registered = GpuNativeDenseWeight {
            weight_id: next_nonzero_id(&NEXT_GPU_NATIVE_WEIGHT_ID, "dense weight"),
            key,
            layout,
            chunks,
        };
        let handle = registry.insert(registered)?;
        self.counters.record_dense_weight_registration(
            plan.chunks.len() as u64,
            plan.physical_allocation_bytes,
        );
        Ok(handle)
    }

    /// Register one immutable RMSNorm gain vector in the existing persistent
    /// F32 dense-weight registry. The typed handle deliberately exposes no
    /// matrix interface.
    pub(crate) fn register_rms_norm(
        &self,
        key: GpuNativeDenseWeightKey,
        weight: &[f32],
    ) -> Result<GpuNativeRmsNormHandle, GpuNativeBootstrapError> {
        validate_rms_norm_weight_width(weight.len())?;
        let dense = DenseWeight::from_f32(weight.to_vec(), 1, weight.len());
        self.register_dense_weight(key, &dense)
            .map(GpuNativeRmsNormHandle::from_dense)
    }

    /// Register already-derived model-scoped RoPE inverse frequencies. This
    /// reuses the persistent dense F32 registry and performs no per-token
    /// parameter upload.
    pub(crate) fn register_rope_parameters(
        &self,
        key: GpuNativeDenseWeightKey,
        rope_dim: usize,
        inverse_frequencies: &[f32],
        attention_factor: f32,
    ) -> Result<GpuNativeRopeHandle, GpuNativeBootstrapError> {
        let layout = GpuNativeRopeLayout::try_new(rope_dim, rope_dim)?;
        validate_rope_parameters(layout, inverse_frequencies, attention_factor)?;
        let dense =
            DenseWeight::from_f32(inverse_frequencies.to_vec(), 1, inverse_frequencies.len());
        let dense = self.register_dense_weight(key, &dense)?;
        self.counters
            .record_rope_registration(dense.layout.allocation_bytes);
        Ok(GpuNativeRopeHandle {
            dense,
            rope_dim,
            attention_factor_bits: attention_factor.to_bits(),
        })
    }

    /// Register the standard Qwen/Llama inverse-frequency schedule.
    pub(crate) fn register_standard_rope(
        &self,
        key: GpuNativeDenseWeightKey,
        rope_dim: usize,
        base: f32,
    ) -> Result<GpuNativeRopeHandle, GpuNativeBootstrapError> {
        let inverse_frequencies = standard_rope_inverse_frequencies(rope_dim, base)?;
        self.register_rope_parameters(key, rope_dim, &inverse_frequencies, 1.0)
    }

    pub(crate) fn dense_weight_handle(
        &self,
        key: &GpuNativeDenseWeightKey,
        expected_kind: GpuNativeDenseWeightKind,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<GpuNativeDenseWeightHandle, GpuNativeBootstrapError> {
        self.dense_weights
            .lock()
            .handle_for(key, expected_kind, expected_rows, expected_cols)
    }

    pub(crate) fn rms_norm_handle(
        &self,
        key: &GpuNativeDenseWeightKey,
        expected_width: usize,
    ) -> Result<GpuNativeRmsNormHandle, GpuNativeBootstrapError> {
        validate_rms_norm_weight_width(expected_width)?;
        self.dense_weights
            .lock()
            .handle_for(key, GpuNativeDenseWeightKind::F32, 1, expected_width)
            .map(GpuNativeRmsNormHandle::from_dense)
    }

    pub(crate) fn create_router_plan(
        &self,
        layer_index: usize,
        geometry: GpuNativeRouterGeometry,
        gate: GpuNativeDenseWeightHandle,
    ) -> Result<GpuNativeRouterPlan, GpuNativeBootstrapError> {
        let plan = GpuNativeRouterPlan {
            context_id: self.context_id,
            layer_index,
            geometry,
            gate,
        };
        validate_router_plan_with_registry(
            self.context_id,
            self.layout.d_model,
            &self.dense_weights.lock(),
            &plan,
        )?;
        Ok(plan)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_attention_plan(
        &self,
        layer_index: usize,
        geometry: GpuNativeAttentionGeometry,
        q_projection: GpuNativeDenseWeightHandle,
        k_projection: GpuNativeDenseWeightHandle,
        v_projection: GpuNativeDenseWeightHandle,
        o_projection: GpuNativeDenseWeightHandle,
        q_norm: Option<GpuNativeAttentionNorm>,
        k_norm: Option<GpuNativeAttentionNorm>,
        rope: GpuNativeRopeHandle,
    ) -> Result<GpuNativeAttentionPlan, GpuNativeBootstrapError> {
        let plan = GpuNativeAttentionPlan {
            context_id: self.context_id,
            layer_index,
            geometry,
            q_projection,
            k_projection,
            v_projection,
            o_projection,
            q_norm,
            k_norm,
            rope,
        };
        self.validate_attention_plan(&plan)?;
        Ok(plan)
    }

    pub(crate) fn create_scratch(
        &self,
        elements: usize,
    ) -> Result<GpuNativeScratch, GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        let layout = GpuNativeScratchLayout::try_new(elements)?;
        layout.validate_for_limits(&gpu.device.limits())?;
        let scratch_id = next_nonzero_id(&NEXT_GPU_NATIVE_SCRATCH_ID, "scratch");
        let buffer = create_startup_buffer(
            &gpu.device,
            &format!("gpu_native_scratch_{scratch_id}"),
            layout.bytes,
            GpuNativeScratchLayout::usage(),
        )?;
        Ok(GpuNativeScratch::from_buffer(
            self.context_id,
            scratch_id,
            layout,
            buffer,
        ))
    }

    pub(crate) fn create_boundary_result_scratch(
        &self,
        elements: usize,
    ) -> Result<GpuNativeScratch, GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        let layout = GpuNativeScratchLayout::try_new(elements)?;
        layout.validate_for_boundary_result_limits(&gpu.device.limits())?;
        let scratch_id = next_nonzero_id(&NEXT_GPU_NATIVE_SCRATCH_ID, "boundary result scratch");
        let buffer = create_startup_buffer(
            &gpu.device,
            &format!("gpu_native_boundary_result_scratch_{scratch_id}"),
            layout.bytes,
            GpuNativeScratchLayout::boundary_result_usage(),
        )?;
        Ok(GpuNativeScratch::from_buffer(
            self.context_id,
            scratch_id,
            layout,
            buffer,
        ))
    }

    pub(crate) fn create_router_scratch(
        &self,
        geometry: GpuNativeRouterGeometry,
    ) -> Result<GpuNativeRouterScratch, GpuNativeBootstrapError> {
        self.create_router_scratch_internal(geometry, false)
    }

    /// Allocate router scratch whose already-produced raw logits may be copied
    /// into diagnostic staging. Production router scratch keeps its narrower
    /// STORAGE-only usage.
    pub(crate) fn create_diagnostic_router_scratch(
        &self,
        geometry: GpuNativeRouterGeometry,
    ) -> Result<GpuNativeRouterScratch, GpuNativeBootstrapError> {
        self.create_router_scratch_internal(geometry, true)
    }

    fn create_router_scratch_internal(
        &self,
        geometry: GpuNativeRouterGeometry,
        diagnostic_logits_copy: bool,
    ) -> Result<GpuNativeRouterScratch, GpuNativeBootstrapError> {
        if geometry.d_model != self.layout.d_model {
            return Err(GpuNativeBootstrapError::RouterDModelMismatch {
                expected: self.layout.d_model,
                actual: geometry.d_model,
            });
        }
        let gpu = self.authoritative_gpu()?;
        let layout = GpuNativeRouterScratchLayout::try_new(geometry)?;
        if diagnostic_logits_copy {
            layout.validate_for_diagnostic_limits(&gpu.device.limits())?;
        } else {
            layout.validate_for_limits(&gpu.device.limits())?;
        }
        validate_router_dispatch(&gpu.device.limits())?;
        let scratch_id = next_nonzero_id(&NEXT_GPU_NATIVE_SCRATCH_ID, "router scratch");
        let logits = create_startup_buffer(
            &gpu.device,
            &format!("gpu_native_router_scratch_{scratch_id}_logits"),
            layout.logits_bytes,
            if diagnostic_logits_copy {
                GpuNativeRouterScratchLayout::diagnostic_logits_usage()
            } else {
                GpuNativeRouterScratchLayout::logits_usage()
            },
        )?;
        let selected_ids = create_startup_buffer(
            &gpu.device,
            &format!("gpu_native_router_scratch_{scratch_id}_selected_ids"),
            layout.selected_ids_bytes,
            GpuNativeRouterScratchLayout::result_usage(),
        )?;
        let selected_weights = create_startup_buffer(
            &gpu.device,
            &format!("gpu_native_router_scratch_{scratch_id}_selected_weights"),
            layout.selected_weights_bytes,
            GpuNativeRouterScratchLayout::result_usage(),
        )?;
        Ok(GpuNativeRouterScratch::from_buffers(
            self.context_id,
            scratch_id,
            layout,
            logits,
            selected_ids,
            selected_weights,
        ))
    }

    /// Construct one layer's fixed-allocation mutable Q4_0 arena from a
    /// caller-supplied post-headroom expert-VRAM plan. Setup may install an
    /// initial set; later residency mutations reuse these exact allocations.
    pub(crate) fn create_q4_expert_arena(
        &self,
        layer_index: usize,
        plan: GpuNativeQ4ExpertVramPlan,
        uploads: &[GpuNativeQ4ExpertUpload<'_>],
    ) -> Result<GpuNativeQ4ExpertArena, GpuNativeBootstrapError> {
        let geometry = plan.geometry;
        if geometry.d_model != self.layout.d_model {
            return Err(GpuNativeBootstrapError::ExpertDModelMismatch {
                expected: self.layout.d_model,
                actual: geometry.d_model,
            });
        }
        let gpu = self.authoritative_gpu()?;
        let limits = gpu.device.limits();
        validate_q4_expert_pipeline_limits(&limits)?;
        let expected_plan = GpuNativeQ4ExpertVramPlan::try_new(
            geometry,
            plan.requested_expert_budget_bytes,
            &limits,
        )?;
        if plan != expected_plan {
            return Err(GpuNativeBootstrapError::ExpertArenaBudgetOverflow);
        }
        let layout = plan.layout;
        let prepared = validate_q4_expert_uploads(layer_index, geometry, layout, uploads)?;
        for (bank, bank_layout) in layout.banks.iter().enumerate() {
            super::validate_startup_buffer(
                &format!("gpu_native_q4_expert_bank_{bank}"),
                bank_layout.allocation_bytes,
                GpuNativeQ4ExpertArenaLayout::weight_usage(),
                &limits,
            )?;
        }
        super::validate_startup_buffer(
            "gpu_native_q4_expert_mapping",
            layout.mapping_bytes,
            GpuNativeQ4ExpertArenaLayout::mapping_usage(),
            &limits,
        )?;

        // All payload, location, arithmetic, and device-limit validation is
        // complete before the first setup-time upload below.
        let mut banks = Vec::with_capacity(MAX_GPU_NATIVE_EXPERT_BANKS);
        for (bank, bank_layout) in layout.banks.iter().enumerate() {
            banks.push(create_startup_buffer(
                &gpu.device,
                &format!("gpu_native_q4_expert_layer_{layer_index}_bank_{bank}"),
                bank_layout.allocation_bytes,
                GpuNativeQ4ExpertArenaLayout::weight_usage(),
            )?);
        }
        for (location, physical) in &prepared.physical_slots {
            let offset = u64::from(location.slot)
                .checked_mul(geometry.slot_stride_bytes as u64)
                .expect("validated Q4 expert slot byte offset");
            gpu.queue
                .write_buffer(&banks[location.bank as usize], offset, physical);
        }
        let mapping_buffer = create_startup_buffer(
            &gpu.device,
            &format!("gpu_native_q4_expert_layer_{layer_index}_mapping"),
            layout.mapping_bytes,
            GpuNativeQ4ExpertArenaLayout::mapping_usage(),
        )?;
        gpu.queue
            .write_buffer(&mapping_buffer, 0, bytemuck::cast_slice(&prepared.mapping));
        let banks: [wgpu::Buffer; MAX_GPU_NATIVE_EXPERT_BANKS] = banks
            .try_into()
            .expect("exactly four GPU-native expert bank buffers");
        let upload_bytes = uploads
            .len()
            .checked_mul(geometry.slot_stride_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .expect("validated Q4 expert upload byte total");
        self.counters
            .record_expert_arena_registration(uploads.len() as u64, upload_bytes);
        Ok(GpuNativeQ4ExpertArena::from_buffers(
            self.context_id,
            layer_index,
            plan,
            banks,
            mapping_buffer,
            prepared.state,
        ))
    }

    /// Validate logical generation direction and reserve one free physical
    /// slot. Newer logical generations first unpublish any older physical
    /// mapping. No victim policy exists here: a full arena returns
    /// `NoPhysicalSlot`.
    pub(crate) fn acquire_q4_expert_residency<'a>(
        &self,
        arena: &'a GpuNativeQ4ExpertArena,
        key: GpuNativeQ4ExpertKey,
    ) -> Result<GpuNativeQ4ExpertAcquire<'a>, GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        validate_q4_expert_arena(self.context_id, arena, &gpu.device.limits())?;
        arena.acquire_with_unpublish(key, |offset, entry| {
            gpu.queue
                .write_buffer(&arena.mapping, offset, bytemuck::bytes_of(&entry));
        })
    }

    /// Queue a validated physical slot write followed by its logical mapping
    /// publication, then commit the host state to `Resident`.
    pub(crate) fn install_q4_expert_residency(
        &self,
        permit: GpuNativeQ4ExpertInstallPermit<'_>,
        payload: &[u8],
    ) -> Result<GpuNativeQ4ExpertResidency, GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        if permit.arena.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignExpertArena);
        }
        let arena = permit.arena;
        let upload_bytes = arena.geometry.slot_stride_bytes as u64;
        let residency = permit.install_with_writes(
            payload,
            |bank, offset, physical| {
                gpu.queue
                    .write_buffer(&arena.banks[bank as usize], offset, physical);
            },
            |offset, entry| {
                gpu.queue
                    .write_buffer(&arena.mapping, offset, bytemuck::bytes_of(&entry));
            },
        )?;
        self.counters.record_expert_residency_upload(upload_bytes);
        Ok(residency)
    }

    /// Retire only the exact logical generation. Mapping unpublication is
    /// queued before the physical slot becomes reusable; an older requester
    /// can never disturb newer state.
    pub(crate) fn retire_q4_expert_residency(
        &self,
        arena: &GpuNativeQ4ExpertArena,
        key: GpuNativeQ4ExpertKey,
    ) -> Result<GpuNativeQ4ExpertRetire, GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        validate_q4_expert_arena(self.context_id, arena, &gpu.device.limits())?;
        arena.retire_with_unpublish(key, |offset, entry| {
            gpu.queue
                .write_buffer(&arena.mapping, offset, bytemuck::bytes_of(&entry));
        })
    }

    pub(crate) fn create_q4_expert_scratch(
        &self,
        geometry: GpuNativeQ4ExpertGeometry,
    ) -> Result<GpuNativeQ4ExpertScratch, GpuNativeBootstrapError> {
        if geometry.d_model != self.layout.d_model {
            return Err(GpuNativeBootstrapError::ExpertDModelMismatch {
                expected: self.layout.d_model,
                actual: geometry.d_model,
            });
        }
        let gpu = self.authoritative_gpu()?;
        let layout = GpuNativeQ4ExpertScratchLayout::try_new(geometry)?;
        layout.validate_for_limits(&gpu.device.limits())?;
        validate_q4_expert_pipeline_limits(&gpu.device.limits())?;
        let scratch_id = next_nonzero_id(&NEXT_GPU_NATIVE_SCRATCH_ID, "expert scratch");
        let resolved_locations = create_startup_buffer(
            &gpu.device,
            &format!("gpu_native_expert_scratch_{scratch_id}_resolved"),
            layout.resolved_locations_bytes,
            GpuNativeQ4ExpertScratchLayout::resolved_usage(),
        )?;
        let create_tensor = |label: &str,
                             tensor_layout: GpuNativeScratchLayout|
         -> Result<GpuNativeScratch, GpuNativeBootstrapError> {
            Ok(GpuNativeScratch::from_buffer(
                self.context_id,
                next_nonzero_id(&NEXT_GPU_NATIVE_SCRATCH_ID, "expert tensor scratch"),
                tensor_layout,
                create_startup_buffer(
                    &gpu.device,
                    label,
                    tensor_layout.bytes,
                    GpuNativeQ4ExpertScratchLayout::tensor_usage(),
                )?,
            ))
        };
        let activation = create_tensor(
            &format!("gpu_native_expert_scratch_{scratch_id}_activation"),
            layout.activation,
        )?;
        let route_outputs = create_tensor(
            &format!("gpu_native_expert_scratch_{scratch_id}_route_outputs"),
            layout.route_outputs,
        )?;
        let combined = create_tensor(
            &format!("gpu_native_expert_scratch_{scratch_id}_combined"),
            layout.combined,
        )?;
        Ok(GpuNativeQ4ExpertScratch::from_buffers(
            self.context_id,
            scratch_id,
            layout,
            resolved_locations,
            activation,
            route_outputs,
            combined,
        ))
    }

    pub(crate) fn create_attention_scratch(
        &self,
        geometry: GpuNativeAttentionGeometry,
    ) -> Result<GpuNativeAttentionScratch, GpuNativeBootstrapError> {
        if geometry.d_model != self.layout.d_model {
            return Err(GpuNativeBootstrapError::AttentionDModelMismatch {
                expected: self.layout.d_model,
                actual: geometry.d_model,
            });
        }
        let gpu = self.authoritative_gpu()?;
        for elements in [
            geometry.q_width,
            geometry.kv_width,
            geometry.kv_width,
            geometry.q_width,
            geometry.d_model,
        ] {
            GpuNativeScratchLayout::try_new(elements)?.validate_for_limits(&gpu.device.limits())?;
        }
        let q = self.create_scratch(geometry.q_width)?;
        let k = self.create_scratch(geometry.kv_width)?;
        let v = self.create_scratch(geometry.kv_width)?;
        let context = self.create_scratch(geometry.q_width)?;
        let projected = self.create_scratch(geometry.d_model)?;
        Ok(GpuNativeAttentionScratch::from_scratch(
            self.context_id,
            geometry,
            q,
            k,
            v,
            context,
            projected,
        ))
    }

    pub(crate) fn create_kv_state(
        &self,
        num_layers: usize,
        max_seq_len: usize,
        kv_width: usize,
    ) -> Result<GpuNativeKvState, GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        let layout =
            GpuNativeKvLayout::try_new(num_layers, max_seq_len, kv_width, &gpu.device.limits())?;
        let kv_id = next_nonzero_id(&NEXT_GPU_NATIVE_KV_ID, "KV state");
        let mut layers = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let key = create_startup_buffer(
                &gpu.device,
                &format!("gpu_native_kv_{kv_id}_layer_{layer}_key"),
                layout.layer_bytes,
                GpuNativeKvLayout::usage(),
            )?;
            let value = create_startup_buffer(
                &gpu.device,
                &format!("gpu_native_kv_{kv_id}_layer_{layer}_value"),
                layout.layer_bytes,
                GpuNativeKvLayout::usage(),
            )?;
            layers.push(GpuNativeKvLayer { key, value });
        }
        Ok(GpuNativeKvState::from_layers(
            self.context_id,
            kv_id,
            layout,
            layers,
        ))
    }

    fn validate_attention_plan(
        &self,
        plan: &GpuNativeAttentionPlan,
    ) -> Result<(), GpuNativeBootstrapError> {
        let registry = self.dense_weights.lock();
        validate_attention_plan_with_registry(self.context_id, self.layout.d_model, &registry, plan)
    }

    /// Encode `weight[rows, cols] * state.hidden[cols] -> output[rows]`.
    pub(crate) fn encode_dense_gemv_hidden_to_scratch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeDenseWeightHandle,
        state: &GpuNativeTokenState,
        output: &GpuNativeScratch,
    ) -> Result<(), GpuNativeBootstrapError> {
        if state.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignTokenState);
        }
        if output.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignScratch);
        }
        self.encode_dense_gemv_buffers(
            encoder,
            handle,
            &state.hidden,
            state.layout.d_model,
            &output.buffer,
            output.layout.elements,
        )
    }

    /// Encode `weight * input -> output` between distinct request scratch
    /// buffers without exposing either raw WGPU buffer outside this module.
    pub(crate) fn encode_dense_gemv_scratch_to_scratch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeDenseWeightHandle,
        input: &GpuNativeScratch,
        output: &GpuNativeScratch,
    ) -> Result<(), GpuNativeBootstrapError> {
        if input.context_id != self.context_id || output.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignScratch);
        }
        if input.scratch_id == output.scratch_id {
            return Err(GpuNativeBootstrapError::AliasedInputOutput);
        }
        self.encode_dense_gemv_buffers(
            encoder,
            handle,
            &input.buffer,
            input.layout.elements,
            &output.buffer,
            output.layout.elements,
        )
    }

    /// Encode `weight * input -> state.hidden` for a matrix whose row count is
    /// exactly d_model.
    pub(crate) fn encode_dense_gemv_scratch_to_hidden(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeDenseWeightHandle,
        input: &GpuNativeScratch,
        state: &GpuNativeTokenState,
    ) -> Result<(), GpuNativeBootstrapError> {
        if input.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignScratch);
        }
        if state.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignTokenState);
        }
        self.encode_dense_gemv_buffers(
            encoder,
            handle,
            &input.buffer,
            input.layout.elements,
            &state.hidden,
            state.layout.d_model,
        )
    }

    fn encode_dense_gemv_buffers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeDenseWeightHandle,
        input: &wgpu::Buffer,
        input_elements: usize,
        output: &wgpu::Buffer,
        output_elements: usize,
    ) -> Result<(), GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        let weight = self.dense_weights.lock().resolve(handle)?;
        if input_elements != weight.layout.cols {
            return Err(GpuNativeBootstrapError::GemvInputLength {
                expected: weight.layout.cols,
                actual: input_elements,
            });
        }
        if output_elements != weight.layout.rows {
            return Err(GpuNativeBootstrapError::GemvOutputLength {
                expected: weight.layout.rows,
                actual: output_elements,
            });
        }
        let workgroups = weight
            .chunks
            .iter()
            .map(|chunk| self.checked_workgroups(chunk.plan.row_count, &gpu.device.limits()))
            .collect::<Result<Vec<_>, _>>()?;
        self.encode_dense_gemv_resolved(gpu, encoder, &weight, input, output, &workgroups);
        Ok(())
    }

    fn encode_dense_gemv_resolved(
        &self,
        gpu: &super::GpuBackend,
        encoder: &mut wgpu::CommandEncoder,
        weight: &GpuNativeDenseWeight,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        workgroups: &[u32],
    ) {
        let pipeline = match weight.layout.kind {
            GpuNativeDenseWeightKind::F32 => &self.dense_pipelines.f32_gemv,
            GpuNativeDenseWeightKind::Q8_0 => &self.dense_pipelines.q8_0_gemv,
        };
        for (chunk, workgroups) in weight.chunks.iter().zip(workgroups.iter().copied()) {
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu_native_dense_gemv_chunk_bind_group"),
                layout: &self.dense_pipelines.gemv_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: chunk.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_native_dense_gemv_chunk_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_push_constants(
                0,
                bytemuck::bytes_of(&GpuNativeGemvPushConstants {
                    rows: chunk.plan.row_count as u32,
                    cols: weight.layout.cols as u32,
                    global_row_base: chunk.plan.row_start as u32,
                    q8_first_block: chunk.plan.first_block as u32,
                }),
            );
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.counters
            .record_dense_gemv_dispatch(weight.chunks.len() as u64);
    }

    /// Encode Qwen/Mixtral routing into caller-owned command storage:
    /// persistent gate GEMV followed by stable softmax, deterministic top-k,
    /// and selected-weight renormalisation. Hidden and residual state are
    /// read-only, and selected ids/weights remain device-resident.
    pub(crate) fn encode_router(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &GpuNativeRouterPlan,
        state: &GpuNativeTokenState,
        scratch: &GpuNativeRouterScratch,
    ) -> Result<(), GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        validate_token_state_owner(self.context_id, state.context_id)?;
        let gate = validate_router_plan_with_registry(
            self.context_id,
            self.layout.d_model,
            &self.dense_weights.lock(),
            plan,
        )?;
        if state.layout.d_model != plan.geometry.d_model {
            return Err(GpuNativeBootstrapError::RouterDModelMismatch {
                expected: plan.geometry.d_model,
                actual: state.layout.d_model,
            });
        }
        validate_router_scratch(self.context_id, plan.geometry, scratch)?;
        scratch.layout.validate_for_limits(&gpu.device.limits())?;
        validate_router_dispatch(&gpu.device.limits())?;
        let gate_workgroups = gate
            .chunks
            .iter()
            .map(|chunk| self.checked_workgroups(chunk.plan.row_count, &gpu.device.limits()))
            .collect::<Result<Vec<_>, _>>()?;

        // Every fallible host validation above completes before this first
        // router command is recorded.
        self.encode_dense_gemv_resolved(
            gpu,
            encoder,
            &gate,
            &state.hidden,
            &scratch.logits,
            &gate_workgroups,
        );
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_router_bind_group"),
            layout: &self.router_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scratch.logits.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scratch.selected_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scratch.selected_weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.status.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_router_topk_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.router_pipeline.topk);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(
            0,
            bytemuck::bytes_of(&GpuNativeRouterPushConstants {
                num_experts: plan.geometry.num_experts as u32,
                top_k: plan.geometry.top_k as u32,
                _reserved_0: 0,
                _reserved_1: 0,
            }),
        );
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);
        self.counters.record_router_dispatch();
        Ok(())
    }

    /// Resolve Slice 6's device-resident routes, validate physical slot
    /// epochs, execute current Q4_0 residents, fail the whole mixture closed,
    /// combine with device-resident weights, and complete
    /// `hidden = residual + combined`. When the retryable residency-miss bit
    /// is set, the resulting `hidden == residual` is containment state, not a
    /// successfully completed MoE layer; a future scheduler must service and
    /// retry it. The caller owns command submission; this method performs no
    /// upload, map, poll, or readback.
    pub(crate) fn encode_q4_expert_arena_combine(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        router_plan: &GpuNativeRouterPlan,
        router_scratch: &GpuNativeRouterScratch,
        arena: &GpuNativeQ4ExpertArena,
        state: &GpuNativeTokenState,
        expert_scratch: &GpuNativeQ4ExpertScratch,
    ) -> Result<(), GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        let limits = gpu.device.limits();
        validate_token_state_owner(self.context_id, state.context_id)?;
        let _gate = validate_router_plan_with_registry(
            self.context_id,
            self.layout.d_model,
            &self.dense_weights.lock(),
            router_plan,
        )?;
        validate_router_scratch(self.context_id, router_plan.geometry, router_scratch)?;
        validate_q4_expert_arena(self.context_id, arena, &limits)?;
        validate_q4_expert_scratch(self.context_id, arena.geometry, expert_scratch, &limits)?;
        validate_router_expert_geometry(router_plan, arena)?;
        if state.layout.d_model != arena.geometry.d_model {
            return Err(GpuNativeBootstrapError::ExpertDModelMismatch {
                expected: arena.geometry.d_model,
                actual: state.layout.d_model,
            });
        }
        validate_q4_expert_pipeline_limits(&limits)?;
        router_scratch.layout.validate_for_limits(&limits)?;
        state.layout.validate_for_limits(&limits)?;
        let route_workgroups = self.checked_workgroups(arena.geometry.top_k, &limits)?;
        let gate_up_workgroups = self.checked_workgroups(arena.geometry.d_ff, &limits)?;
        let down_workgroups = self.checked_workgroups(arena.geometry.d_model, &limits)?;
        let combine_workgroups = down_workgroups;
        let residual_workgroups = down_workgroups;
        let bank_slots =
            arena.plan.layout.banks.map(|bank| {
                u32::try_from(bank.slot_capacity).expect("validated bank slot capacity")
            });
        let resolve_pc = GpuNativeExpertResolvePushConstants {
            num_experts: arena.geometry.num_experts as u32,
            top_k: arena.geometry.top_k as u32,
            active_banks: arena.plan.layout.active_banks as u32,
            _reserved: 0,
            bank_slots,
        };
        let combine_pc = GpuNativeExpertCombinePushConstants {
            d_model: arena.geometry.d_model as u32,
            top_k: arena.geometry.top_k as u32,
            _reserved_0: 0,
            _reserved_1: 0,
        };

        // All fallible host-side validation and dispatch planning above is
        // complete before this first expert command is recorded.
        let route_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_expert_route_resolve_bind_group"),
            layout: &self.q4_expert_pipelines.route_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: router_scratch.selected_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: arena.mapping.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: expert_scratch.resolved_locations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.status.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_native_expert_route_resolve_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.q4_expert_pipelines.route_resolve);
            pass.set_bind_group(0, &route_bind_group, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&resolve_pc));
            pass.dispatch_workgroups(route_workgroups, 1, 1);
        }

        let expert_bind_group =
            |label: &'static str, input: &wgpu::Buffer, output: &wgpu::Buffer| {
                gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &self.q4_expert_pipelines.expert_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: arena.banks[0].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: arena.banks[1].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: arena.banks[2].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: arena.banks[3].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: input.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: expert_scratch.resolved_locations.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: output.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: state.status.as_entire_binding(),
                        },
                    ],
                })
            };
        let gate_up_bind_group = expert_bind_group(
            "gpu_native_q4_expert_gate_up_bind_group",
            &state.hidden,
            &expert_scratch.activation.buffer,
        );
        let down_bind_group = expert_bind_group(
            "gpu_native_q4_expert_down_bind_group",
            &expert_scratch.activation.buffer,
            &expert_scratch.route_outputs.buffer,
        );
        for route_slot in 0..arena.geometry.top_k {
            let pc = GpuNativeQ4ExpertPushConstants {
                d_model: arena.geometry.d_model as u32,
                d_ff: arena.geometry.d_ff as u32,
                blocks_per_projection: arena.geometry.blocks_per_projection as u32,
                slot_stride_bytes: arena.geometry.slot_stride_bytes as u32,
                route_slot: route_slot as u32,
                top_k: arena.geometry.top_k as u32,
                swiglu_limit: crate::inference::swiglu_limit().unwrap_or(f32::INFINITY),
                _reserved: 0,
            };
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_native_q4_expert_gate_up_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.q4_expert_pipelines.gate_up);
                pass.set_bind_group(0, &gate_up_bind_group, &[]);
                pass.set_push_constants(0, bytemuck::bytes_of(&pc));
                pass.dispatch_workgroups(gate_up_workgroups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_native_q4_expert_down_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.q4_expert_pipelines.down);
                pass.set_bind_group(0, &down_bind_group, &[]);
                pass.set_push_constants(0, bytemuck::bytes_of(&pc));
                pass.dispatch_workgroups(down_workgroups, 1, 1);
            }
        }

        let combine_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_expert_combine_bind_group"),
            layout: &self.q4_expert_pipelines.combine_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: router_scratch.selected_weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: expert_scratch.resolved_locations.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: expert_scratch.route_outputs.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: expert_scratch.combined.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state.status.as_entire_binding(),
                },
            ],
        });
        let mut control_pass = |label, pipeline: &wgpu::ComputePipeline, workgroups| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            // Control resources intentionally occupy @group(1), but WGPU still
            // requires the pipeline's explicit empty group-0 layout to be bound.
            pass.set_bind_group(0, &self.q4_expert_pipelines.control_empty_bind_group, &[]);
            pass.set_bind_group(1, &combine_bind_group, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&combine_pc));
            pass.dispatch_workgroups(workgroups, 1, 1);
        };
        control_pass(
            "gpu_native_expert_validate_pass",
            &self.q4_expert_pipelines.validate,
            1,
        );
        control_pass(
            "gpu_native_expert_combine_pass",
            &self.q4_expert_pipelines.combine,
            combine_workgroups,
        );
        control_pass(
            "gpu_native_expert_contain_pass",
            &self.q4_expert_pipelines.contain,
            combine_workgroups,
        );
        drop(control_pass);
        self.encode_residual_add_pass(
            gpu,
            encoder,
            state,
            &expert_scratch.combined,
            residual_workgroups,
        );
        self.counters
            .record_expert_dispatches(arena.geometry.top_k as u64);
        Ok(())
    }

    /// Encode one embedding row directly into request-local hidden state.
    pub(crate) fn encode_embedding_lookup(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeDenseWeightHandle,
        token_id: u32,
        state: &GpuNativeTokenState,
    ) -> Result<(), GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        if state.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignTokenState);
        }
        let weight = self.dense_weights.lock().resolve(handle)?;
        weight.layout.validate_embedding_token(token_id)?;
        if weight.layout.cols != state.layout.d_model {
            return Err(GpuNativeBootstrapError::EmbeddingWidth {
                expected: state.layout.d_model,
                actual: weight.layout.cols,
            });
        }
        let workgroups = self.checked_workgroups(weight.layout.cols, &gpu.device.limits())?;
        let chunk = weight
            .chunks
            .iter()
            .find(|chunk| chunk.plan.contains_row(token_id as usize))
            .expect("validated embedding row must belong to exactly one chunk");
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_embedding_bind_group"),
            layout: &self.dense_pipelines.embedding_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: chunk.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state.hidden.as_entire_binding(),
                },
            ],
        });
        let pipeline = match weight.layout.kind {
            GpuNativeDenseWeightKind::F32 => &self.dense_pipelines.f32_embedding,
            GpuNativeDenseWeightKind::Q8_0 => &self.dense_pipelines.q8_0_embedding,
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_embedding_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(
            0,
            bytemuck::bytes_of(&GpuNativeEmbeddingPushConstants {
                local_row: token_id - chunk.plan.row_start as u32,
                global_row: token_id,
                cols: weight.layout.cols as u32,
                q8_first_block: chunk.plan.first_block as u32,
            }),
        );
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
        self.counters.record_embedding_dispatch();
        Ok(())
    }

    /// Preserve the current residual stream and RMS-normalise hidden in one
    /// dispatch: `residual = old hidden`, `hidden = rms_norm(old hidden)`.
    pub(crate) fn encode_rms_norm_state_in_place(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeRmsNormHandle,
        epsilon: f32,
        state: &GpuNativeTokenState,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_token_state_owner(self.context_id, state.context_id)?;
        self.encode_rms_norm_buffers(
            encoder,
            handle,
            epsilon,
            &state.hidden,
            state.layout.d_model,
            Some(&state.residual),
            1,
            state.layout.d_model,
            false,
        )
    }

    /// Apply final-model RMSNorm to hidden without changing the saved residual.
    pub(crate) fn encode_rms_norm_hidden_in_place(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeRmsNormHandle,
        epsilon: f32,
        state: &GpuNativeTokenState,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_token_state_owner(self.context_id, state.context_id)?;
        self.encode_rms_norm_buffers(
            encoder,
            handle,
            epsilon,
            &state.hidden,
            state.layout.d_model,
            None,
            1,
            state.layout.d_model,
            false,
        )
    }

    /// Perform GPU greedy argmax over logits and write the selected token to `sampled_token_buf`.
    /// On numerical failure or prior non-zero status, latches status and writes u32::MAX.
    pub(crate) fn encode_greedy_argmax(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        logits: &GpuNativeScratch,
        sampled_token_buf: &GpuNativeScratch,
        state: &GpuNativeTokenState,
        vocab_size: usize,
    ) -> Result<(), GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        if logits.context_id != self.context_id || sampled_token_buf.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignScratch);
        }
        if state.context_id != self.context_id {
            return Err(GpuNativeBootstrapError::ForeignTokenState);
        }
        if logits.layout.elements() < vocab_size {
            return Err(GpuNativeBootstrapError::GemvInputLength {
                expected: vocab_size,
                actual: logits.layout.elements(),
            });
        }
        if sampled_token_buf.layout.elements() < 1 {
            return Err(GpuNativeBootstrapError::GemvOutputLength {
                expected: 1,
                actual: sampled_token_buf.layout.elements(),
            });
        }
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_greedy_argmax_bind_group"),
            layout: &self.greedy_argmax_pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sampled_token_buf.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.status.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_greedy_argmax_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.greedy_argmax_pipeline.compute_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let pc = GpuNativeGreedyArgmaxPushConstants {
            vocab_size: vocab_size as u32,
            _reserved_0: 0,
            _reserved_1: 0,
            _reserved_2: 0,
        };
        pass.set_push_constants(0, bytemuck::bytes_of(&pc));
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);
        Ok(())
    }

    /// RMS-normalise each logical scratch group independently using a shared
    /// `group_width`-element gain vector.
    pub(crate) fn encode_rms_norm_scratch_in_place(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeRmsNormHandle,
        epsilon: f32,
        scratch: &GpuNativeScratch,
        groups: usize,
        group_width: usize,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_scratch_owner(self.context_id, scratch.context_id)?;
        self.encode_rms_norm_buffers(
            encoder,
            handle,
            epsilon,
            &scratch.buffer,
            scratch.layout.elements,
            None,
            groups,
            group_width,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_rms_norm_buffers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeRmsNormHandle,
        epsilon: f32,
        target: &wgpu::Buffer,
        target_elements: usize,
        residual: Option<&wgpu::Buffer>,
        groups: usize,
        group_width: usize,
        scratch_dispatch: bool,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_rms_norm_epsilon(epsilon)?;
        let gpu = self.authoritative_gpu()?;
        let weight = self.dense_weights.lock().resolve_rms_norm(handle)?;
        let geometry = GpuNativeRmsNormGeometry::try_new(
            groups,
            group_width,
            target_elements,
            weight.layout.cols,
        )?;
        let workgroups = geometry.checked_workgroups(&gpu.device.limits())?;
        let chunk = match weight.chunks.as_slice() {
            [chunk] => chunk,
            _ => {
                return Err(GpuNativeBootstrapError::StaleRmsNormHandle {
                    key: handle.dense.key.as_str().to_string(),
                });
            }
        };
        let push_constants = GpuNativeRmsNormPushConstants {
            groups: geometry.groups as u32,
            group_width: geometry.group_width as u32,
            epsilon_bits: epsilon.to_bits(),
            _reserved: 0,
        };

        match residual {
            Some(residual) => {
                let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gpu_native_rms_capture_bind_group"),
                    layout: &self.state_pipelines.rms_capture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: chunk.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: target.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: residual.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_native_rms_capture_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.state_pipelines.rms_capture);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_push_constants(0, bytemuck::bytes_of(&push_constants));
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            None => {
                let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gpu_native_rms_in_place_bind_group"),
                    layout: &self.state_pipelines.rms_in_place_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: chunk.buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: target.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("gpu_native_rms_in_place_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.state_pipelines.rms_in_place);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.set_push_constants(0, bytemuck::bytes_of(&push_constants));
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
        if scratch_dispatch {
            self.counters
                .record_rms_norm_scratch_dispatch(geometry.groups as u64);
        } else {
            self.counters
                .record_rms_norm_state_dispatch(geometry.groups as u64);
        }
        Ok(())
    }

    fn validate_attention_dispatch_limits(
        &self,
        plan: &GpuNativeAttentionPlan,
        limits: &wgpu::Limits,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_causal_attention_dispatch(plan.geometry, limits)?;
        let registry = self.dense_weights.lock();
        for handle in [
            &plan.q_projection,
            &plan.k_projection,
            &plan.v_projection,
            &plan.o_projection,
        ] {
            let weight = registry.resolve(handle)?;
            for chunk in &weight.chunks {
                self.checked_workgroups(chunk.plan.row_count, limits)?;
            }
        }
        for groups in [plan.geometry.num_heads, plan.geometry.num_kv_heads] {
            if groups as u64 > limits.max_compute_workgroups_per_dimension as u64 {
                return Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
                    workgroups: groups as u64,
                    maximum: limits.max_compute_workgroups_per_dimension,
                });
            }
            let pairs = groups.checked_mul(plan.geometry.rope_dim / 2).ok_or(
                GpuNativeBootstrapError::AttentionGeometryOverflow {
                    num_heads: plan.geometry.num_heads,
                    num_kv_heads: plan.geometry.num_kv_heads,
                    head_dim: plan.geometry.head_dim,
                },
            )?;
            self.checked_workgroups(pairs, limits)?;
        }
        self.checked_workgroups(plan.geometry.kv_width, limits)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_rope_scratch_in_place(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        handle: &GpuNativeRopeHandle,
        scratch: &GpuNativeScratch,
        tensor: GpuNativeAttentionTensor,
        groups: usize,
        head_dim: usize,
        position: usize,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_scratch_owner(self.context_id, scratch.context_id)?;
        validate_rope_dimension(handle.rope_dim, head_dim)?;
        let expected = groups.checked_mul(head_dim).ok_or(
            GpuNativeBootstrapError::AttentionGeometryOverflow {
                num_heads: groups,
                num_kv_heads: groups,
                head_dim,
            },
        )?;
        if scratch.layout.elements != expected {
            return Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor,
                expected,
                actual: scratch.layout.elements,
            });
        }
        let position =
            u32::try_from(position).map_err(|_| GpuNativeBootstrapError::InvalidKvPosition {
                position,
                max_seq_len: u32::MAX as usize,
            })?;
        let gpu = self.authoritative_gpu()?;
        let registry = self.dense_weights.lock();
        validate_rope_handle_with_registry(self.context_id, &registry, handle, handle.rope_dim)?;
        let weight = registry.resolve(&handle.dense)?;
        let chunk = match weight.chunks.as_slice() {
            [chunk] => chunk,
            _ => {
                return Err(GpuNativeBootstrapError::StaleRopeHandle {
                    key: handle.dense.key.as_str().to_string(),
                });
            }
        };
        let pairs = groups.checked_mul(handle.rope_dim / 2).ok_or(
            GpuNativeBootstrapError::AttentionGeometryOverflow {
                num_heads: groups,
                num_kv_heads: groups,
                head_dim,
            },
        )?;
        let workgroups = self.checked_workgroups(pairs, &gpu.device.limits())?;
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_rope_bind_group"),
            layout: &self.attention_pipelines.rope_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: chunk.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scratch.buffer.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_rope_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.attention_pipelines.rope);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(
            0,
            bytemuck::bytes_of(&GpuNativeRopePushConstants {
                groups: groups as u32,
                head_dim: head_dim as u32,
                rope_dim: handle.rope_dim as u32,
                position,
                attention_factor_bits: handle.attention_factor_bits,
            }),
        );
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
        self.counters.record_rope_dispatch(groups as u64);
        Ok(())
    }

    fn encode_kv_append(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        k: &GpuNativeScratch,
        v: &GpuNativeScratch,
        kv: &GpuNativeKvState,
        layer: usize,
        position: usize,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_scratch_owner(self.context_id, k.context_id)?;
        validate_scratch_owner(self.context_id, v.context_id)?;
        validate_kv_state(self.context_id, k.layout.elements, kv, layer, position)?;
        if v.layout.elements != kv.layout.kv_width {
            return Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor: GpuNativeAttentionTensor::Value,
                expected: kv.layout.kv_width,
                actual: v.layout.elements,
            });
        }
        let gpu = self.authoritative_gpu()?;
        let workgroups = self.checked_workgroups(kv.layout.kv_width, &gpu.device.limits())?;
        let layer_buffers = &kv.layers[layer];
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_kv_append_bind_group"),
            layout: &self.attention_pipelines.kv_append_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: k.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: v.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: layer_buffers.key.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: layer_buffers.value.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_kv_append_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.attention_pipelines.kv_append);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(
            0,
            bytemuck::bytes_of(&GpuNativeKvAppendPushConstants {
                width: kv.layout.kv_width as u32,
                position: position as u32,
            }),
        );
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
        self.counters.record_kv_append();
        Ok(())
    }

    /// Compose Q/K/V projection, optional per-head QK-Norm, per-head RoPE,
    /// and absolute-position request-local KV append into the caller's encoder.
    pub(crate) fn encode_attention_prepare(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &GpuNativeAttentionPlan,
        state: &GpuNativeTokenState,
        scratch: &GpuNativeAttentionScratch,
        kv: &GpuNativeKvState,
        position: usize,
    ) -> Result<(), GpuNativeBootstrapError> {
        self.encode_attention_prepare_impl(encoder, plan, state, scratch, kv, position, None)
    }

    /// Diagnostic variant of [`Self::encode_attention_prepare`] that captures intermediate
    /// Q/K/V activations before and after norm and RoPE into `diagnostic_sink`.
    pub(crate) fn encode_attention_prepare_layer0_diagnostic(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &GpuNativeAttentionPlan,
        state: &GpuNativeTokenState,
        scratch: &GpuNativeAttentionScratch,
        kv: &GpuNativeKvState,
        position: usize,
        diagnostic_sink: &Layer0AttentionGpuDiagnosticSink<'_>,
    ) -> Result<(), GpuNativeBootstrapError> {
        self.encode_attention_prepare_impl(
            encoder,
            plan,
            state,
            scratch,
            kv,
            position,
            Some(diagnostic_sink),
        )
    }

    fn encode_attention_prepare_impl(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &GpuNativeAttentionPlan,
        state: &GpuNativeTokenState,
        scratch: &GpuNativeAttentionScratch,
        kv: &GpuNativeKvState,
        position: usize,
        diagnostic_sink: Option<&Layer0AttentionGpuDiagnosticSink<'_>>,
    ) -> Result<(), GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        self.validate_attention_plan(plan)?;
        validate_token_state_owner(self.context_id, state.context_id)?;
        if state.layout.d_model != plan.geometry.d_model {
            return Err(GpuNativeBootstrapError::AttentionDModelMismatch {
                expected: plan.geometry.d_model,
                actual: state.layout.d_model,
            });
        }
        validate_attention_scratch(self.context_id, plan.geometry, scratch)?;
        validate_attention_kv_state(
            self.context_id,
            plan.geometry,
            kv,
            plan.layer_index,
            position,
        )?;
        self.validate_attention_dispatch_limits(plan, &gpu.device.limits())?;

        self.encode_dense_gemv_hidden_to_scratch(encoder, &plan.q_projection, state, &scratch.q)?;
        if let Some(sink) = diagnostic_sink {
            encoder.copy_buffer_to_buffer(
                &scratch.q.buffer,
                0,
                sink.staging_buffer,
                sink.layout.q_raw_offset as u64,
                sink.layout.q_raw_bytes as u64,
            );
        }
        self.encode_dense_gemv_hidden_to_scratch(encoder, &plan.k_projection, state, &scratch.k)?;
        if let Some(sink) = diagnostic_sink {
            encoder.copy_buffer_to_buffer(
                &scratch.k.buffer,
                0,
                sink.staging_buffer,
                sink.layout.k_raw_offset as u64,
                sink.layout.k_raw_bytes as u64,
            );
        }
        self.encode_dense_gemv_hidden_to_scratch(encoder, &plan.v_projection, state, &scratch.v)?;
        if let Some(sink) = diagnostic_sink {
            encoder.copy_buffer_to_buffer(
                &scratch.v.buffer,
                0,
                sink.staging_buffer,
                sink.layout.v_raw_offset as u64,
                sink.layout.v_raw_bytes as u64,
            );
        }
        if let Some(norm) = &plan.q_norm {
            self.encode_rms_norm_scratch_in_place(
                encoder,
                &norm.handle,
                norm.epsilon(),
                &scratch.q,
                plan.geometry.num_heads,
                plan.geometry.head_dim,
            )?;
        }
        if let Some(sink) = diagnostic_sink {
            encoder.copy_buffer_to_buffer(
                &scratch.q.buffer,
                0,
                sink.staging_buffer,
                sink.layout.q_after_norm_offset as u64,
                sink.layout.q_after_norm_bytes as u64,
            );
        }
        if let Some(norm) = &plan.k_norm {
            self.encode_rms_norm_scratch_in_place(
                encoder,
                &norm.handle,
                norm.epsilon(),
                &scratch.k,
                plan.geometry.num_kv_heads,
                plan.geometry.head_dim,
            )?;
        }
        if let Some(sink) = diagnostic_sink {
            encoder.copy_buffer_to_buffer(
                &scratch.k.buffer,
                0,
                sink.staging_buffer,
                sink.layout.k_after_norm_offset as u64,
                sink.layout.k_after_norm_bytes as u64,
            );
        }
        self.encode_rope_scratch_in_place(
            encoder,
            &plan.rope,
            &scratch.q,
            GpuNativeAttentionTensor::Query,
            plan.geometry.num_heads,
            plan.geometry.head_dim,
            position,
        )?;
        if let Some(sink) = diagnostic_sink {
            encoder.copy_buffer_to_buffer(
                &scratch.q.buffer,
                0,
                sink.staging_buffer,
                sink.layout.q_after_rope_offset as u64,
                sink.layout.q_after_rope_bytes as u64,
            );
        }
        self.encode_rope_scratch_in_place(
            encoder,
            &plan.rope,
            &scratch.k,
            GpuNativeAttentionTensor::Key,
            plan.geometry.num_kv_heads,
            plan.geometry.head_dim,
            position,
        )?;
        if let Some(sink) = diagnostic_sink {
            encoder.copy_buffer_to_buffer(
                &scratch.k.buffer,
                0,
                sink.staging_buffer,
                sink.layout.k_after_rope_offset as u64,
                sink.layout.k_after_rope_bytes as u64,
            );
        }
        self.encode_kv_append(
            encoder,
            &scratch.k,
            &scratch.v,
            kv,
            plan.layer_index,
            position,
        )?;
        self.counters.record_attention_prepare_dispatch();
        Ok(())
    }

    fn encode_causal_attention_pass(
        &self,
        gpu: &super::GpuBackend,
        encoder: &mut wgpu::CommandEncoder,
        plan: &GpuNativeAttentionPlan,
        state: &GpuNativeTokenState,
        scratch: &GpuNativeAttentionScratch,
        kv: &GpuNativeKvState,
        seq_len: u32,
    ) {
        let layer_buffers = &kv.layers[plan.layer_index];
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_causal_attention_bind_group"),
            layout: &self.attention_pipelines.causal_attention_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scratch.q.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: layer_buffers.key.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: layer_buffers.value.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch.context.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state.status.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_causal_attention_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.attention_pipelines.causal_attention);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(
            0,
            bytemuck::bytes_of(&GpuNativeAttentionPushConstants {
                num_heads: plan.geometry.num_heads as u32,
                num_kv_heads: plan.geometry.num_kv_heads as u32,
                head_dim: plan.geometry.head_dim as u32,
                seq_len,
            }),
        );
        pass.dispatch_workgroups(plan.geometry.num_heads as u32, 1, 1);
        drop(pass);
        self.counters.record_causal_attention_dispatch();
    }

    /// Complete one prepared incremental attention operation entirely in the
    /// caller's encoder: causal attention, persistent O projection, then the
    /// saved pre-attention residual add. `state.residual` is never a target.
    pub(crate) fn encode_attention_complete(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &GpuNativeAttentionPlan,
        state: &GpuNativeTokenState,
        scratch: &GpuNativeAttentionScratch,
        kv: &GpuNativeKvState,
        position: usize,
    ) -> Result<(), GpuNativeBootstrapError> {
        self.encode_attention_complete_impl(encoder, plan, state, scratch, kv, position, None)
    }

    /// Diagnostic variant of [`Self::encode_attention_complete`] that captures intermediate
    /// causal attention context and O-projection activations into `diagnostic_sink`.
    pub(crate) fn encode_attention_complete_layer0_diagnostic(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &GpuNativeAttentionPlan,
        state: &GpuNativeTokenState,
        scratch: &GpuNativeAttentionScratch,
        kv: &GpuNativeKvState,
        position: usize,
        diagnostic_sink: &Layer0AttentionGpuDiagnosticSink<'_>,
    ) -> Result<(), GpuNativeBootstrapError> {
        self.encode_attention_complete_impl(
            encoder,
            plan,
            state,
            scratch,
            kv,
            position,
            Some(diagnostic_sink),
        )
    }

    fn encode_attention_complete_impl(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        plan: &GpuNativeAttentionPlan,
        state: &GpuNativeTokenState,
        scratch: &GpuNativeAttentionScratch,
        kv: &GpuNativeKvState,
        position: usize,
        diagnostic_sink: Option<&Layer0AttentionGpuDiagnosticSink<'_>>,
    ) -> Result<(), GpuNativeBootstrapError> {
        let gpu = self.authoritative_gpu()?;
        self.validate_attention_plan(plan)?;
        validate_token_state_owner(self.context_id, state.context_id)?;
        if state.layout.d_model != plan.geometry.d_model {
            return Err(GpuNativeBootstrapError::AttentionDModelMismatch {
                expected: plan.geometry.d_model,
                actual: state.layout.d_model,
            });
        }
        validate_attention_scratch(self.context_id, plan.geometry, scratch)?;
        let seq_len = validate_attention_kv_state(
            self.context_id,
            plan.geometry,
            kv,
            plan.layer_index,
            position,
        )?;
        let seq_len = u32::try_from(seq_len).map_err(|_| {
            GpuNativeBootstrapError::InvalidAttentionSequenceLength {
                seq_len,
                max_seq_len: kv.layout.max_seq_len,
            }
        })?;
        self.validate_attention_dispatch_limits(plan, &gpu.device.limits())?;
        validate_residual_contribution_width(
            state.layout.d_model,
            scratch.projected.layout.elements,
        )?;

        // Resolve and validate every fallible O-projection property before the
        // first completion command is recorded. The registry never removes a
        // weight, so the Arc remains authoritative for the encoded passes.
        let o_projection = self.dense_weights.lock().resolve(&plan.o_projection)?;
        if scratch.context.layout.elements != o_projection.layout.cols {
            return Err(GpuNativeBootstrapError::GemvInputLength {
                expected: o_projection.layout.cols,
                actual: scratch.context.layout.elements,
            });
        }
        if scratch.projected.layout.elements != o_projection.layout.rows {
            return Err(GpuNativeBootstrapError::GemvOutputLength {
                expected: o_projection.layout.rows,
                actual: scratch.projected.layout.elements,
            });
        }
        let o_workgroups = o_projection
            .chunks
            .iter()
            .map(|chunk| self.checked_workgroups(chunk.plan.row_count, &gpu.device.limits()))
            .collect::<Result<Vec<_>, _>>()?;
        let residual_workgroups =
            self.checked_workgroups(state.layout.d_model, &gpu.device.limits())?;

        self.encode_causal_attention_pass(gpu, encoder, plan, state, scratch, kv, seq_len);
        if let Some(sink) = diagnostic_sink {
            encoder.copy_buffer_to_buffer(
                &scratch.context.buffer,
                0,
                sink.staging_buffer,
                sink.layout.attention_context_offset as u64,
                sink.layout.attention_context_bytes as u64,
            );
        }
        self.encode_dense_gemv_resolved(
            gpu,
            encoder,
            &o_projection,
            &scratch.context.buffer,
            &scratch.projected.buffer,
            &o_workgroups,
        );
        if let Some(sink) = diagnostic_sink {
            encoder.copy_buffer_to_buffer(
                &scratch.projected.buffer,
                0,
                sink.staging_buffer,
                sink.layout.o_projection_offset as u64,
                sink.layout.o_projection_bytes as u64,
            );
        }
        self.encode_residual_add_pass(gpu, encoder, state, &scratch.projected, residual_workgroups);
        self.counters.record_attention_complete_dispatch();
        Ok(())
    }

    /// Complete a prepared sub-block entirely on device:
    /// `hidden = residual + contribution`.
    pub(crate) fn encode_residual_add_scratch_to_hidden(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        state: &GpuNativeTokenState,
        contribution: &GpuNativeScratch,
    ) -> Result<(), GpuNativeBootstrapError> {
        validate_token_state_owner(self.context_id, state.context_id)?;
        validate_scratch_owner(self.context_id, contribution.context_id)?;
        validate_residual_contribution_width(state.layout.d_model, contribution.layout.elements)?;
        let gpu = self.authoritative_gpu()?;
        let workgroups = self.checked_workgroups(state.layout.d_model, &gpu.device.limits())?;
        self.encode_residual_add_pass(gpu, encoder, state, contribution, workgroups);
        Ok(())
    }

    fn encode_residual_add_pass(
        &self,
        gpu: &super::GpuBackend,
        encoder: &mut wgpu::CommandEncoder,
        state: &GpuNativeTokenState,
        contribution: &GpuNativeScratch,
        workgroups: u32,
    ) {
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_residual_add_bind_group"),
            layout: &self.state_pipelines.rms_capture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: contribution.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state.hidden.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: state.residual.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_residual_add_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.state_pipelines.residual_add);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(
            0,
            bytemuck::bytes_of(&GpuNativeRmsNormPushConstants {
                groups: 1,
                group_width: state.layout.d_model as u32,
                epsilon_bits: 0,
                _reserved: 0,
            }),
        );
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
        self.counters.record_residual_add_dispatch();
    }

    pub(crate) fn authoritative_gpu(&self) -> Result<&super::GpuBackend, GpuNativeBootstrapError> {
        let gpu = match self.authoritative_backend.as_ref() {
            BackendBox::Gpu(gpu) => gpu,
            BackendBox::Cpu(_) => return Err(GpuNativeBootstrapError::GpuBackendUnavailable),
            #[cfg(test)]
            BackendBox::TestGpu(_) => {
                return Err(GpuNativeBootstrapError::GpuBackendUnavailable);
            }
        };
        if let Some(detail) = gpu.device_loss.detail() {
            return Err(GpuNativeBootstrapError::DeviceLost { detail });
        }
        Ok(gpu)
    }

    fn checked_workgroups(
        &self,
        elements: usize,
        limits: &wgpu::Limits,
    ) -> Result<u32, GpuNativeBootstrapError> {
        let elements = u64::try_from(elements).map_err(|_| {
            GpuNativeBootstrapError::DispatchGeometryUnsupported {
                workgroups: u64::MAX,
                maximum: limits.max_compute_workgroups_per_dimension,
            }
        })?;
        let workgroups = elements.div_ceil(GPU_NATIVE_WORKGROUP_SIZE as u64);
        if workgroups > limits.max_compute_workgroups_per_dimension as u64 {
            return Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
                workgroups,
                maximum: limits.max_compute_workgroups_per_dimension,
            });
        }
        Ok(workgroups as u32)
    }

    pub(crate) fn device_identity(&self) -> &GpuDeviceIdentity {
        &self.device_identity
    }

    pub(crate) const fn token_state_layout(&self) -> GpuNativeTokenStateLayout {
        self.layout
    }

    pub(crate) fn execution_snapshot(&self) -> GpuNativeExecutionSnapshot {
        self.counters.snapshot()
    }
}

impl fmt::Debug for GpuNativeExecutorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuNativeExecutorContext")
            .field("device_identity", &self.device_identity)
            .field("layout", &self.layout)
            .field("snapshot", &self.execution_snapshot())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::inference::{dequantize_q8_0_block, quantize_q8_0_block};
    use std::sync::atomic::AtomicUsize;

    fn q8_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = vec![0; values.len().div_ceil(Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES];
        for (block, chunk) in values.chunks(Q8_0_BLOCK_ELEMS).enumerate() {
            quantize_q8_0_block(
                chunk,
                &mut bytes[block * Q8_0_BLOCK_BYTES..(block + 1) * Q8_0_BLOCK_BYTES],
            );
        }
        bytes
    }

    fn read_q8_mirror(bytes: &[u8], flat_index: usize) -> f32 {
        let block = flat_index / Q8_0_BLOCK_ELEMS;
        let in_block = flat_index % Q8_0_BLOCK_ELEMS;
        let offset = block * Q8_0_BLOCK_BYTES;
        let scale = half::f16::from_le_bytes([bytes[offset], bytes[offset + 1]]).to_f32();
        scale * (bytes[offset + 2 + in_block] as i8 as f32)
    }

    fn read_q8_chunk_mirror(
        bytes: &[u8],
        plan: GpuNativeDenseWeightChunkPlan,
        global_flat_index: usize,
    ) -> f32 {
        let global_block = global_flat_index / Q8_0_BLOCK_ELEMS;
        let local_block = global_block - plan.first_block;
        let in_block = global_flat_index % Q8_0_BLOCK_ELEMS;
        let offset = local_block * Q8_0_BLOCK_BYTES;
        let scale = half::f16::from_le_bytes([bytes[offset], bytes[offset + 1]]).to_f32();
        scale * (bytes[offset + 2 + in_block] as i8 as f32)
    }

    fn q8_gemv_mirror(bytes: &[u8], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
        (0..rows)
            .map(|row| {
                let mut sum = 0.0;
                for col in 0..cols {
                    sum += read_q8_mirror(bytes, row * cols + col) * input[col];
                }
                sum
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    fn router_topk_mirror(logits: &[f32], top_k: usize) -> (Vec<u32>, Vec<f32>, bool) {
        let sanitized = || (vec![0; top_k], vec![0.0; top_k], true);
        if logits.is_empty()
            || top_k == 0
            || top_k > logits.len()
            || logits.iter().any(|value| !value.is_finite())
        {
            return sanitized();
        }
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !maximum.is_finite() {
            return sanitized();
        }
        let mut scores = Vec::with_capacity(logits.len());
        let mut denominator = 0.0f32;
        for &logit in logits {
            let exponent = (logit - maximum).exp();
            if !exponent.is_finite() || exponent < 0.0 {
                return sanitized();
            }
            scores.push(exponent);
            denominator += exponent;
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return sanitized();
        }
        for score in &mut scores {
            *score /= denominator;
            if !score.is_finite() || *score < 0.0 {
                return sanitized();
            }
        }

        let mut ids = (0..logits.len()).collect::<Vec<_>>();
        ids.sort_by(|&left, &right| {
            scores[right]
                .total_cmp(&scores[left])
                .then_with(|| left.cmp(&right))
        });
        ids.truncate(top_k);
        let selected_sum = ids.iter().map(|&expert| scores[expert]).sum::<f32>();
        if !selected_sum.is_finite() || selected_sum <= 0.0 {
            return sanitized();
        }
        let mut selected_weights = Vec::with_capacity(top_k);
        for &expert in &ids {
            let weight = scores[expert] / selected_sum;
            if !weight.is_finite() || weight < 0.0 {
                return sanitized();
            }
            selected_weights.push(weight);
        }
        (
            ids.into_iter().map(|expert| expert as u32).collect(),
            selected_weights,
            false,
        )
    }

    fn q4_uniform_projection(rows: usize, cols: usize, weight: f32) -> Vec<u8> {
        assert!(cols.is_multiple_of(Q4_0_BLOCK_ELEMS));
        let (scale, nibble) = if weight > 0.0 {
            (weight, 9u8)
        } else if weight < 0.0 {
            (-weight, 7u8)
        } else {
            (0.0, 8u8)
        };
        let scale = half::f16::from_f32(scale).to_bits().to_le_bytes();
        let mut block = [0u8; Q4_0_BLOCK_BYTES];
        block[..2].copy_from_slice(&scale);
        block[2..].fill(nibble | (nibble << 4));
        block.repeat(rows * cols / Q4_0_BLOCK_ELEMS).to_vec()
    }

    pub(crate) fn q4_uniform_expert(
        geometry: GpuNativeQ4ExpertGeometry,
        gate_weight: f32,
        up_weight: f32,
        down_weight: f32,
    ) -> Vec<u8> {
        let mut payload = q4_uniform_projection(geometry.d_ff, geometry.d_model, gate_weight);
        payload.extend(q4_uniform_projection(
            geometry.d_ff,
            geometry.d_model,
            up_weight,
        ));
        payload.extend(q4_uniform_projection(
            geometry.d_model,
            geometry.d_ff,
            down_weight,
        ));
        assert_eq!(payload.len(), geometry.logical_expert_bytes);
        payload
    }

    fn q4_projection_mirror(
        payload: &[u8],
        first_block: usize,
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Vec<f32> {
        assert_eq!(input.len(), cols);
        let blocks_per_row = cols / Q4_0_BLOCK_ELEMS;
        (0..rows)
            .map(|row| {
                let mut sum = 0.0;
                for block in 0..blocks_per_row {
                    let block_index = first_block + row * blocks_per_row + block;
                    let offset = block_index * Q4_0_BLOCK_BYTES;
                    let mut decoded = [0.0; Q4_0_BLOCK_ELEMS];
                    crate::inference::dequantize_q4_0_block(
                        &payload[offset..offset + Q4_0_BLOCK_BYTES],
                        &mut decoded,
                    );
                    let input_offset = block * Q4_0_BLOCK_ELEMS;
                    sum += decoded
                        .iter()
                        .zip(&input[input_offset..input_offset + Q4_0_BLOCK_ELEMS])
                        .map(|(weight, value)| weight * value)
                        .sum::<f32>();
                }
                sum
            })
            .collect()
    }

    fn q4_expert_mirror(
        payload: &[u8],
        geometry: GpuNativeQ4ExpertGeometry,
        hidden: &[f32],
    ) -> Vec<f32> {
        let gate = q4_projection_mirror(
            payload,
            geometry.gate_block_offset(),
            geometry.d_ff,
            geometry.d_model,
            hidden,
        );
        let up = q4_projection_mirror(
            payload,
            geometry.up_block_offset(),
            geometry.d_ff,
            geometry.d_model,
            hidden,
        );
        let activation = gate
            .into_iter()
            .zip(up)
            .map(|(gate, up)| {
                let gate = crate::inference::swiglu_limit()
                    .map(|limit| gate.clamp(-limit, limit))
                    .unwrap_or(gate);
                gate / (1.0 + (-gate).exp()) * up
            })
            .collect::<Vec<_>>();
        q4_projection_mirror(
            payload,
            geometry.down_block_offset(),
            geometry.d_model,
            geometry.d_ff,
            &activation,
        )
    }

    fn weighted_expert_combine_mirror(outputs: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
        assert_eq!(outputs.len(), weights.len());
        let width = outputs.first().map(Vec::len).unwrap_or(0);
        assert!(outputs.iter().all(|output| output.len() == width));
        (0..width)
            .map(|element| {
                outputs
                    .iter()
                    .zip(weights)
                    .map(|(output, weight)| output[element] * weight)
                    .sum()
            })
            .collect()
    }

    fn fail_closed_expert_combine_mirror(
        outputs: &[Vec<f32>],
        weights: &[f32],
        status: u32,
    ) -> Vec<f32> {
        let width = outputs.first().map(Vec::len).unwrap_or(0);
        if status != 0
            || outputs.len() != weights.len()
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
            || outputs.iter().any(|output| {
                output.len() != width || output.iter().any(|value| !value.is_finite())
            })
        {
            return vec![0.0; width];
        }
        let combined = weighted_expert_combine_mirror(outputs, weights);
        if combined.iter().any(|value| !value.is_finite()) {
            vec![0.0; width]
        } else {
            combined
        }
    }

    fn resolve_routes_mirror(
        selected_ids: &[u32],
        mapping: &[GpuNativeQ4ExpertMappingEntry],
        layout: GpuNativeQ4ExpertArenaLayout,
        initial_status: u32,
    ) -> (Vec<GpuNativeQ4ExpertMappingEntry>, u32) {
        if initial_status != 0 {
            return (
                vec![GpuNativeQ4ExpertMappingEntry::UNMAPPED; selected_ids.len()],
                initial_status,
            );
        }
        let mut status = initial_status;
        let resolved = selected_ids
            .iter()
            .map(|&logical_id| {
                let Some(&entry) = mapping.get(logical_id as usize) else {
                    status |= GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS;
                    return GpuNativeQ4ExpertMappingEntry::UNMAPPED;
                };
                if entry.slot_epoch == 0 {
                    status |= GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS;
                    return GpuNativeQ4ExpertMappingEntry::UNMAPPED;
                }
                let Some(location) = GpuNativeQ4ExpertLocation::unpack(entry.location) else {
                    status |= GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS;
                    return GpuNativeQ4ExpertMappingEntry::UNMAPPED;
                };
                let bank = location.bank as usize;
                if bank >= layout.active_banks
                    || location.slot as usize >= layout.banks[bank].slot_capacity
                {
                    status |= GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS;
                    GpuNativeQ4ExpertMappingEntry::UNMAPPED
                } else {
                    entry
                }
            })
            .collect();
        (resolved, status)
    }

    fn resolved_route_epoch_matches(
        route: GpuNativeQ4ExpertMappingEntry,
        current_slot_epoch: u32,
    ) -> bool {
        route.location != GPU_NATIVE_EXPERT_UNMAPPED
            && route.slot_epoch != 0
            && route.slot_epoch == current_slot_epoch
    }

    fn rms_norm_mirror(
        values: &[f32],
        weight: &[f32],
        epsilon: f32,
        groups: usize,
        group_width: usize,
    ) -> Vec<f32> {
        assert_eq!(values.len(), groups * group_width);
        assert_eq!(weight.len(), group_width);
        let mut result = values.to_vec();
        for group in result.chunks_exact_mut(group_width) {
            let mut squared_sum = 0.0f32;
            for &value in group.iter() {
                squared_sum += value * value;
            }
            let mean_square = squared_sum / group_width as f32;
            let inverse_rms = 1.0 / (mean_square + epsilon).sqrt();
            for (value, gain) in group.iter_mut().zip(weight) {
                *value = *value * inverse_rms * *gain;
            }
        }
        result
    }

    fn residual_add_mirror(residual: &[f32], contribution: &[f32]) -> Vec<f32> {
        residual
            .iter()
            .zip(contribution)
            .map(|(&residual, &contribution)| residual + contribution)
            .collect()
    }

    fn rms_norm_capture_mirror(
        hidden: &mut Vec<f32>,
        residual: &mut Vec<f32>,
        weight: &[f32],
        epsilon: f32,
    ) {
        residual.clone_from(hidden);
        let width = hidden.len();
        *hidden = rms_norm_mirror(hidden, weight, epsilon, 1, width);
    }

    fn residual_complete_mirror(hidden: &mut Vec<f32>, residual: &[f32], contribution: &[f32]) {
        *hidden = residual_add_mirror(residual, contribution);
    }

    fn rope_mirror(
        values: &[f32],
        groups: usize,
        head_dim: usize,
        rope_dim: usize,
        position: usize,
        inverse_frequencies: &[f32],
        attention_factor: f32,
    ) -> Vec<f32> {
        assert_eq!(values.len(), groups * head_dim);
        assert_eq!(inverse_frequencies.len(), rope_dim / 2);
        let mut result = values.to_vec();
        let pairs = rope_dim / 2;
        for group in 0..groups {
            let head_start = group * head_dim;
            for pair in 0..pairs {
                let theta = position as f32 * inverse_frequencies[pair];
                let (sin_theta, cos_theta) = theta.sin_cos();
                let sin_theta = sin_theta * attention_factor;
                let cos_theta = cos_theta * attention_factor;
                let first = head_start + pair;
                let second = first + pairs;
                let a = result[first];
                let b = result[second];
                result[first] = a * cos_theta - b * sin_theta;
                result[second] = a * sin_theta + b * cos_theta;
            }
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_prepare_mirror(
        hidden: &[f32],
        geometry: GpuNativeAttentionGeometry,
        q_projection: &DenseWeight,
        k_projection: &DenseWeight,
        v_projection: &DenseWeight,
        q_norm: Option<(&[f32], f32)>,
        k_norm: Option<(&[f32], f32)>,
        inverse_frequencies: &[f32],
        attention_factor: f32,
        position: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut q = q_projection.matvec(hidden);
        let mut k = k_projection.matvec(hidden);
        let v = v_projection.matvec(hidden);
        if let Some((gain, epsilon)) = q_norm {
            q = rms_norm_mirror(&q, gain, epsilon, geometry.num_heads, geometry.head_dim);
        }
        if let Some((gain, epsilon)) = k_norm {
            k = rms_norm_mirror(&k, gain, epsilon, geometry.num_kv_heads, geometry.head_dim);
        }
        q = rope_mirror(
            &q,
            geometry.num_heads,
            geometry.head_dim,
            geometry.rope_dim,
            position,
            inverse_frequencies,
            attention_factor,
        );
        k = rope_mirror(
            &k,
            geometry.num_kv_heads,
            geometry.head_dim,
            geometry.rope_dim,
            position,
            inverse_frequencies,
            attention_factor,
        );
        (q, k, v)
    }

    fn causal_attention_mirror(
        q: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        geometry: GpuNativeAttentionGeometry,
        seq_len: usize,
    ) -> Vec<f32> {
        assert_eq!(q.len(), geometry.q_width);
        assert!(seq_len > 0);
        assert!(key_cache.len() >= seq_len * geometry.kv_width);
        assert!(value_cache.len() >= seq_len * geometry.kv_width);
        let mut context = vec![0.0; geometry.q_width];
        let scale = 1.0 / (geometry.head_dim as f32).sqrt();
        for query_head in 0..geometry.num_heads {
            let kv_head = query_head * geometry.num_kv_heads / geometry.num_heads;
            let q_start = query_head * geometry.head_dim;
            let q_head = &q[q_start..q_start + geometry.head_dim];
            let mut scores = Vec::with_capacity(seq_len);
            for position in 0..seq_len {
                let k_start = position * geometry.kv_width + kv_head * geometry.head_dim;
                let k_head = &key_cache[k_start..k_start + geometry.head_dim];
                scores.push(
                    q_head
                        .iter()
                        .zip(k_head)
                        .map(|(q_value, k_value)| q_value * k_value)
                        .sum::<f32>()
                        * scale,
                );
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0;
            for score in &mut scores {
                *score = (*score - maximum).exp();
                denominator += *score;
            }
            for score in &mut scores {
                *score /= denominator;
            }
            let context_head = &mut context[q_start..q_start + geometry.head_dim];
            for (position, weight) in scores.into_iter().enumerate() {
                let v_start = position * geometry.kv_width + kv_head * geometry.head_dim;
                let v_head = &value_cache[v_start..v_start + geometry.head_dim];
                for (output, value) in context_head.iter_mut().zip(v_head) {
                    *output += weight * value;
                }
            }
        }
        context
    }

    fn attention_complete_mirror(
        q: &[f32],
        key_cache: &[f32],
        value_cache: &[f32],
        geometry: GpuNativeAttentionGeometry,
        seq_len: usize,
        o_projection: &DenseWeight,
        residual: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        assert_eq!(o_projection.rows(), geometry.d_model);
        assert_eq!(o_projection.cols(), geometry.q_width);
        assert_eq!(residual.len(), geometry.d_model);
        let context = causal_attention_mirror(q, key_cache, value_cache, geometry, seq_len);
        let projected = o_projection.matvec(&context);
        let hidden = residual
            .iter()
            .zip(&projected)
            .map(|(saved, contribution)| saved + contribution)
            .collect();
        (context, projected, hidden)
    }

    fn test_scratch<B>(
        context_id: u64,
        scratch_id: u64,
        elements: usize,
        buffer: B,
    ) -> GpuNativeScratch<B> {
        GpuNativeScratch::from_buffer(
            context_id,
            scratch_id,
            GpuNativeScratchLayout::try_new(elements).unwrap(),
            buffer,
        )
    }

    fn test_weight<B>(
        weight_id: u64,
        key: &str,
        layout: GpuNativeDenseWeightLayout,
        buffer: B,
    ) -> GpuNativeDenseWeight<B> {
        let plan = GpuNativeDenseWeightPlan::try_new(layout, &wgpu::Limits::default()).unwrap();
        assert_eq!(plan.chunks.len(), 1);
        GpuNativeDenseWeight {
            weight_id,
            key: GpuNativeDenseWeightKey::try_new(key).unwrap(),
            layout,
            chunks: vec![GpuNativeDenseWeightChunk {
                plan: plan.chunks[0],
                buffer,
            }],
        }
    }

    fn insert_test_f32_weight(
        registry: &mut GpuNativeDenseWeightRegistry<()>,
        weight_id: u64,
        key: &str,
        rows: usize,
        cols: usize,
    ) -> GpuNativeDenseWeightHandle {
        let bytes = rows
            .checked_mul(cols)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .unwrap();
        let layout =
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, rows, cols, bytes)
                .unwrap();
        registry
            .insert(test_weight(weight_id, key, layout, ()))
            .unwrap()
    }

    #[test]
    fn f32_persistent_weight_layout_charges_exact_bytes() {
        let weight = DenseWeight::from_f32(vec![0.0; 15], 3, 5);
        let layout = GpuNativeDenseWeightLayout::from_weight(&weight).unwrap();
        assert_eq!(layout.kind(), GpuNativeDenseWeightKind::F32);
        assert_eq!((layout.rows(), layout.cols()), (3, 5));
        assert_eq!(layout.payload_bytes(), 60);
        assert_eq!(layout.allocation_bytes(), 60);
    }

    #[test]
    fn q8_persistent_weight_layout_preserves_flat_blocks_and_wgpu_padding() {
        // 3x35 deliberately makes rows cross block boundaries and leaves a
        // final partial block: ceil(105/32) * 34 = 136 bytes.
        let values = (0..105).map(|i| i as f32 - 52.0).collect::<Vec<_>>();
        let weight = DenseWeight::from_q8_0_bytes(q8_bytes(&values), 3, 35).unwrap();
        let layout = GpuNativeDenseWeightLayout::from_weight(&weight).unwrap();
        assert_eq!(layout.kind(), GpuNativeDenseWeightKind::Q8_0);
        assert_eq!((layout.rows(), layout.cols()), (3, 35));
        assert_eq!(layout.payload_bytes(), 136);
        assert_eq!(layout.allocation_bytes(), 136);

        let one_block = GpuNativeDenseWeightLayout::try_new(
            GpuNativeDenseWeightKind::Q8_0,
            1,
            32,
            Q8_0_BLOCK_BYTES,
        )
        .unwrap();
        assert_eq!(one_block.payload_bytes(), 34);
        assert_eq!(one_block.allocation_bytes(), 36);
    }

    #[test]
    fn qwen_dense_weight_plans_fit_physical_storage_limits_without_allocating_payloads() {
        const ROWS: usize = 151_936;
        const COLS: usize = 2_048;
        const STORAGE_LIMIT: u64 = 128 * 1024 * 1024;
        const BUFFER_LIMIT: u64 = 256 * 1024 * 1024;
        let limits = wgpu::Limits {
            max_buffer_size: BUFFER_LIMIT,
            max_storage_buffer_binding_size: STORAGE_LIMIT as u32,
            ..wgpu::Limits::default()
        };

        let elements = ROWS * COLS;
        let f32_layout = GpuNativeDenseWeightLayout::try_new(
            GpuNativeDenseWeightKind::F32,
            ROWS,
            COLS,
            elements * std::mem::size_of::<f32>(),
        )
        .unwrap();
        let f32_plan = GpuNativeDenseWeightPlan::try_new(f32_layout, &limits).unwrap();
        assert_eq!(f32_plan.chunks.len(), 10);
        assert_eq!(f32_plan.chunks[0].row_count, 16_384);
        assert_eq!(f32_plan.chunks.last().unwrap().row_count, 4_480);
        assert!(f32_plan
            .chunks
            .iter()
            .all(|chunk| chunk.allocation_bytes <= STORAGE_LIMIT));
        assert_eq!(
            f32_plan
                .chunks
                .iter()
                .map(|chunk| chunk.allocation_bytes)
                .max(),
            Some(STORAGE_LIMIT)
        );
        for (index, chunk) in f32_plan.chunks.iter().enumerate() {
            super::super::validate_startup_buffer(
                &format!("test_qwen_f32_chunk_{index}"),
                chunk.allocation_bytes,
                GpuNativeDenseWeightLayout::usage(),
                &limits,
            )
            .unwrap();
        }

        let q8_bytes = elements.div_ceil(Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES;
        let q8_layout = GpuNativeDenseWeightLayout::try_new(
            GpuNativeDenseWeightKind::Q8_0,
            ROWS,
            COLS,
            q8_bytes,
        )
        .unwrap();
        let q8_plan = GpuNativeDenseWeightPlan::try_new(q8_layout, &limits).unwrap();
        assert_eq!(q8_plan.chunks.len(), 3);
        assert_eq!(q8_plan.chunks[0].row_count, 61_680);
        assert_eq!(q8_plan.chunks.last().unwrap().row_count, 28_576);
        assert!(q8_plan
            .chunks
            .iter()
            .all(|chunk| chunk.allocation_bytes <= STORAGE_LIMIT));
        assert_eq!(
            q8_plan
                .chunks
                .iter()
                .map(|chunk| chunk.allocation_bytes)
                .max(),
            Some(134_215_680)
        );
        for (index, chunk) in q8_plan.chunks.iter().enumerate() {
            super::super::validate_startup_buffer(
                &format!("test_qwen_q8_chunk_{index}"),
                chunk.allocation_bytes,
                GpuNativeDenseWeightLayout::usage(),
                &limits,
            )
            .unwrap();
        }
    }

    #[test]
    fn q8_row_crossing_chunks_preserve_metadata_blocks_gemv_and_embedding() {
        let rows = 3;
        let cols = 35;
        let values = (0..rows * cols)
            .map(|i| ((i * 17 % 43) as f32 - 21.0) / 7.0)
            .collect::<Vec<_>>();
        let source = q8_bytes(&values);
        let weight = DenseWeight::from_q8_0_bytes(source.clone(), rows, cols).unwrap();
        let layout = GpuNativeDenseWeightLayout::from_weight(&weight).unwrap();
        let limits = wgpu::Limits {
            max_buffer_size: 68,
            max_storage_buffer_binding_size: 68,
            ..wgpu::Limits::default()
        };
        let plan = GpuNativeDenseWeightPlan::try_new(layout, &limits).unwrap();
        assert_eq!(plan.chunks.len(), 3);
        assert_eq!(plan.physical_allocation_bytes, 204);
        assert_eq!(
            plan.chunks,
            vec![
                GpuNativeDenseWeightChunkPlan {
                    row_start: 0,
                    row_count: 1,
                    first_block: 0,
                    payload_offset_bytes: 0,
                    payload_bytes: 68,
                    allocation_bytes: 68,
                },
                GpuNativeDenseWeightChunkPlan {
                    row_start: 1,
                    row_count: 1,
                    first_block: 1,
                    payload_offset_bytes: 34,
                    payload_bytes: 68,
                    allocation_bytes: 68,
                },
                GpuNativeDenseWeightChunkPlan {
                    row_start: 2,
                    row_count: 1,
                    first_block: 2,
                    payload_offset_bytes: 68,
                    payload_bytes: 68,
                    allocation_bytes: 68,
                },
            ]
        );

        let chunk_bytes = plan
            .chunks
            .iter()
            .map(|chunk| {
                &source[chunk.payload_offset_bytes
                    ..chunk.payload_offset_bytes + chunk.payload_bytes as usize]
            })
            .collect::<Vec<_>>();
        assert_eq!(&chunk_bytes[0][34..68], &chunk_bytes[1][0..34]);
        assert_eq!(&chunk_bytes[1][34..68], &chunk_bytes[2][0..34]);

        for (chunk, bytes) in plan.chunks.iter().copied().zip(&chunk_bytes) {
            for row in chunk.row_start..chunk.row_end() {
                for col in 0..cols {
                    let flat = row * cols + col;
                    assert_eq!(
                        read_q8_chunk_mirror(bytes, chunk, flat),
                        read_q8_mirror(&source, flat)
                    );
                }
            }
        }

        let input = (0..cols)
            .map(|i| ((i * 11 % 19) as f32 - 9.0) / 5.0)
            .collect::<Vec<_>>();
        let expected_gemv = weight.matvec(&input);
        let mut chunked_gemv = vec![0.0; rows];
        for (chunk, bytes) in plan.chunks.iter().copied().zip(&chunk_bytes) {
            for row in chunk.row_start..chunk.row_end() {
                for col in 0..cols {
                    chunked_gemv[row] +=
                        read_q8_chunk_mirror(bytes, chunk, row * cols + col) * input[col];
                }
            }
        }
        assert_close(&chunked_gemv, &expected_gemv, 1e-5);

        for token in 0..rows {
            let (chunk, bytes) = plan
                .chunks
                .iter()
                .copied()
                .zip(&chunk_bytes)
                .find(|(chunk, _)| chunk.contains_row(token))
                .unwrap();
            let actual = (0..cols)
                .map(|col| read_q8_chunk_mirror(bytes, chunk, token * cols + col))
                .collect::<Vec<_>>();
            let mut expected = Vec::new();
            weight.row_dequant_into(token, &mut expected);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn dense_weight_planner_fails_when_one_complete_row_cannot_fit() {
        let layout =
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, 2, 4, 32).unwrap();
        let limits = wgpu::Limits {
            max_buffer_size: 15,
            max_storage_buffer_binding_size: 15,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            GpuNativeDenseWeightPlan::try_new(layout, &limits),
            Err(GpuNativeBootstrapError::DenseWeightRowExceedsDeviceLimit {
                kind: GpuNativeDenseWeightKind::F32,
                cols: 4,
                required: 16,
                maximum: 15,
            })
        );
    }

    #[test]
    fn dense_weight_layout_rejects_empty_malformed_and_overflowing_shapes() {
        assert_eq!(
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, 0, 4, 0),
            Err(GpuNativeBootstrapError::InvalidDenseWeightShape { rows: 0, cols: 4 })
        );
        assert_eq!(
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::Q8_0, 1, 32, 33),
            Err(GpuNativeBootstrapError::DenseWeightByteLength {
                kind: GpuNativeDenseWeightKind::Q8_0,
                rows: 1,
                cols: 32,
                expected: 34,
                actual: 33,
            })
        );
        assert_eq!(
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, usize::MAX, 2, 0,),
            Err(GpuNativeBootstrapError::DenseWeightShapeOverflow {
                rows: usize::MAX,
                cols: 2,
            })
        );
        if usize::BITS > u32::BITS {
            let rows = u32::MAX as usize + 1;
            assert_eq!(
                GpuNativeDenseWeightLayout::try_new(
                    GpuNativeDenseWeightKind::F32,
                    rows,
                    1,
                    rows * std::mem::size_of::<f32>(),
                ),
                Err(GpuNativeBootstrapError::DenseWeightDimensionTooLarge { rows, cols: 1 })
            );
        }
    }

    #[test]
    fn rms_norm_mirror_is_exactly_the_cpu_reference_for_required_widths() {
        for width in [1usize, 7, 65, 2_048] {
            let values = (0..width)
                .map(|index| ((index * 17 % 41) as f32 - 20.0) / 9.0)
                .collect::<Vec<_>>();
            let weight = (0..width)
                .map(|index| 0.75 + (index * 11 % 13) as f32 / 20.0)
                .collect::<Vec<_>>();
            let epsilon = 1e-6;
            let expected =
                crate::transformer::RmsNorm::new(weight.clone(), epsilon).forward(&values);
            assert_eq!(
                rms_norm_mirror(&values, &weight, epsilon, 1, width),
                expected,
                "width={width} must preserve the scalar f32 accumulation contract"
            );
        }

        let zero = vec![0.0; 7];
        assert_eq!(rms_norm_mirror(&zero, &[1.0; 7], 1e-6, 1, 7), zero);
    }

    #[test]
    fn grouped_rms_norm_matches_per_head_cpu_reference_and_isolates_boundaries() {
        let groups = 4;
        let group_width = 7;
        let epsilon = 1e-5;
        let weight = (0..group_width)
            .map(|index| 0.5 + index as f32 / 7.0)
            .collect::<Vec<_>>();
        let values = (0..groups * group_width)
            .map(|index| ((index * 19 % 37) as f32 - 18.0) / 5.0)
            .collect::<Vec<_>>();
        let actual = rms_norm_mirror(&values, &weight, epsilon, groups, group_width);
        let norm = crate::transformer::RmsNorm::new(weight.clone(), epsilon);
        let expected = values
            .chunks_exact(group_width)
            .flat_map(|group| norm.forward(group))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);

        let mut changed_last_group = values.clone();
        changed_last_group[3 * group_width..].fill(10_000.0);
        let changed = rms_norm_mirror(&changed_last_group, &weight, epsilon, groups, group_width);
        assert_eq!(
            &actual[..3 * group_width],
            &changed[..3 * group_width],
            "one Q/K head must not affect another head's RMS reduction"
        );
    }

    #[test]
    fn rms_norm_geometry_fails_closed_for_every_invalid_shape() {
        assert_eq!(
            validate_rms_norm_weight_width(0),
            Err(GpuNativeBootstrapError::InvalidRmsNormWeightWidth { width: 0 })
        );
        assert_eq!(validate_rms_norm_weight_width(2_048), Ok(()));
        assert_eq!(
            GpuNativeRmsNormGeometry::try_new(0, 7, 0, 7),
            Err(GpuNativeBootstrapError::InvalidRmsNormGroups { groups: 0 })
        );
        assert_eq!(
            GpuNativeRmsNormGeometry::try_new(3, 0, 0, 0),
            Err(GpuNativeBootstrapError::InvalidRmsNormGroupWidth { group_width: 0 })
        );
        assert_eq!(
            GpuNativeRmsNormGeometry::try_new(usize::MAX, 2, 0, 2),
            Err(GpuNativeBootstrapError::RmsNormGeometryOverflow {
                groups: usize::MAX,
                group_width: 2,
            })
        );
        if usize::BITS > u32::BITS {
            let groups = u32::MAX as usize + 1;
            assert_eq!(
                GpuNativeRmsNormGeometry::try_new(groups, 1, groups, 1),
                Err(GpuNativeBootstrapError::RmsNormGeometryOverflow {
                    groups,
                    group_width: 1,
                })
            );
        }
        assert_eq!(
            GpuNativeRmsNormGeometry::try_new(3, 7, 20, 7),
            Err(GpuNativeBootstrapError::RmsNormScratchGeometry {
                expected: 21,
                actual: 20,
            })
        );
        assert_eq!(
            GpuNativeRmsNormGeometry::try_new(3, 7, 21, 8),
            Err(GpuNativeBootstrapError::RmsNormWeightWidth {
                expected: 7,
                actual: 8,
            })
        );
        let geometry = GpuNativeRmsNormGeometry::try_new(4, 7, 28, 7).unwrap();
        let limits = wgpu::Limits {
            max_compute_workgroups_per_dimension: 3,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            geometry.checked_workgroups(&limits),
            Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
                workgroups: 4,
                maximum: 3,
            })
        );
    }

    #[test]
    fn rms_norm_epsilon_rejects_non_finite_and_negative_values() {
        for epsilon in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1e-6] {
            assert_eq!(
                validate_rms_norm_epsilon(epsilon),
                Err(GpuNativeBootstrapError::InvalidRmsNormEpsilon {
                    epsilon_bits: epsilon.to_bits(),
                })
            );
        }

        assert_eq!(validate_rms_norm_epsilon(0.0), Ok(()));
        assert_eq!(validate_rms_norm_epsilon(1e-6), Ok(()));
    }

    #[test]
    fn router_geometry_accepts_qwen_bounds_and_rejects_invalid_geometry() {
        let geometry = GpuNativeRouterGeometry::try_new(16, 128, 8).unwrap();
        assert_eq!(geometry.d_model(), 16);
        assert_eq!(geometry.num_experts(), 128);
        assert_eq!(geometry.top_k(), 8);
        assert_eq!(MAX_GPU_NATIVE_ROUTER_EXPERTS, 128);
        assert_eq!(MAX_GPU_NATIVE_ROUTER_TOP_K, 8);

        assert_eq!(
            GpuNativeRouterGeometry::try_new(0, 128, 8),
            Err(GpuNativeBootstrapError::InvalidDModel)
        );
        for num_experts in [0, 129] {
            assert_eq!(
                GpuNativeRouterGeometry::try_new(16, num_experts, 1),
                Err(GpuNativeBootstrapError::InvalidRouterExpertCount { num_experts })
            );
        }
        for (num_experts, top_k) in [(128, 0), (128, 9), (4, 5)] {
            assert_eq!(
                GpuNativeRouterGeometry::try_new(16, num_experts, top_k),
                Err(GpuNativeBootstrapError::InvalidRouterTopK { top_k, num_experts })
            );
        }
        assert!(matches!(
            GpuNativeRouterGeometry::try_new(u32::MAX as usize + 1, 128, 8),
            Err(GpuNativeBootstrapError::RouterGeometryOverflow { .. })
        ));
        assert!(matches!(
            GpuNativeRouterGeometry::try_new(u32::MAX as usize / 128 + 1, 128, 8),
            Err(GpuNativeBootstrapError::RouterGeometryOverflow { .. })
        ));
    }

    #[test]
    fn router_dispatch_fails_closed_for_device_workgroup_limits() {
        assert_eq!(validate_router_dispatch(&wgpu::Limits::default()), Ok(()));
        let narrow_workgroup = wgpu::Limits {
            max_compute_workgroup_size_x: 32,
            max_compute_invocations_per_workgroup: 32,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            validate_router_dispatch(&narrow_workgroup),
            Err(GpuNativeBootstrapError::RouterWorkgroupUnsupported {
                required: 64,
                max_size_x: 32,
                max_invocations: 32,
            })
        );
        let narrow_storage = wgpu::Limits {
            max_compute_workgroup_storage_size: 512,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            validate_router_dispatch(&narrow_storage),
            Err(GpuNativeBootstrapError::RouterWorkgroupStorageUnsupported {
                required: GPU_NATIVE_ROUTER_WORKGROUP_STORAGE_BYTES,
                maximum: 512,
            })
        );
    }

    fn supported_expert_limits() -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffers_per_shader_stage: GPU_NATIVE_EXPERT_REQUIRED_STORAGE_BUFFERS,
            max_push_constant_size: GPU_NATIVE_EXPERT_PUSH_CONSTANT_BYTES,
            max_compute_workgroup_size_x: GPU_NATIVE_EXPERT_WORKGROUP_SIZE,
            max_compute_invocations_per_workgroup: GPU_NATIVE_EXPERT_WORKGROUP_SIZE,
            ..wgpu::Limits::default()
        }
    }

    fn test_mutable_expert_arena(slot_capacity: usize) -> GpuNativeQ4ExpertArena<()> {
        let geometry = GpuNativeQ4ExpertGeometry::try_new(32, 32, 128, 1).unwrap();
        let limits = supported_expert_limits();
        let layout =
            GpuNativeQ4ExpertArenaLayout::try_new(geometry, slot_capacity, &limits).unwrap();
        let plan = GpuNativeQ4ExpertVramPlan::try_new(
            geometry,
            layout.total_allocation_bytes().unwrap(),
            &limits,
        )
        .unwrap();
        assert_eq!(plan.slot_capacity(), slot_capacity);
        let state = GpuNativeQ4ExpertArenaState::new(geometry, plan.layout).unwrap();
        GpuNativeQ4ExpertArena::from_buffers(
            7,
            3,
            plan,
            [(); MAX_GPU_NATIVE_EXPERT_BANKS],
            (),
            state,
        )
    }

    fn expect_expert_install<'a, B>(
        acquired: GpuNativeQ4ExpertAcquire<'a, B>,
    ) -> GpuNativeQ4ExpertInstallPermit<'a, B> {
        match acquired {
            GpuNativeQ4ExpertAcquire::Install(permit) => permit,
            GpuNativeQ4ExpertAcquire::Hit(_) => panic!("unexpected residency hit"),
            GpuNativeQ4ExpertAcquire::InstallInProgress => {
                panic!("unexpected installation in progress")
            }
            GpuNativeQ4ExpertAcquire::StaleRequester => panic!("unexpected stale requester"),
            GpuNativeQ4ExpertAcquire::NoPhysicalSlot => panic!("unexpected full arena"),
        }
    }

    #[test]
    fn q4_expert_vram_planner_counts_banks_mapping_dummies_and_budget_boundaries() {
        let geometry = GpuNativeQ4ExpertGeometry::try_new(32, 32, 128, 1).unwrap();
        let limits = supported_expert_limits();
        let one_layout = GpuNativeQ4ExpertArenaLayout::try_new(geometry, 1, &limits).unwrap();
        let exact_budget = one_layout.total_allocation_bytes().unwrap();
        let one = GpuNativeQ4ExpertVramPlan::try_new(geometry, exact_budget, &limits).unwrap();
        assert_eq!(one.requested_expert_budget_bytes(), exact_budget);
        assert_eq!(one.slot_capacity(), 1);
        assert_eq!(one.active_banks(), 1);
        assert_eq!(one.slot_stride_bytes(), geometry.slot_stride_bytes());
        assert_eq!(
            one.mapping_metadata_bytes(),
            (geometry.num_experts() * GPU_NATIVE_EXPERT_MAPPING_ENTRY_BYTES) as u64
        );
        assert_eq!(
            one.active_bank_allocation_bytes(),
            geometry.slot_stride_bytes() as u64
        );
        assert_eq!(
            one.physical_bank_allocation_bytes(),
            one.active_bank_allocation_bytes() + 3 * 4
        );
        assert_eq!(
            one.total_arena_allocation_bytes(),
            one.physical_bank_allocation_bytes() + one.mapping_metadata_bytes()
        );
        assert!(matches!(
            GpuNativeQ4ExpertVramPlan::try_new(geometry, exact_budget - 1, &limits),
            Err(GpuNativeBootstrapError::ExpertArenaBudgetTooSmall { .. })
        ));

        let four_layout = GpuNativeQ4ExpertArenaLayout::try_new(geometry, 4, &limits).unwrap();
        let four = GpuNativeQ4ExpertVramPlan::try_new(
            geometry,
            four_layout.total_allocation_bytes().unwrap(),
            &limits,
        )
        .unwrap();
        assert!(four.slot_capacity() >= one.slot_capacity());
        assert_eq!(four.slot_capacity(), 4);

        let unsupported = wgpu::Limits {
            max_storage_buffers_per_shader_stage: GPU_NATIVE_EXPERT_REQUIRED_STORAGE_BUFFERS - 1,
            ..limits
        };
        assert!(matches!(
            GpuNativeQ4ExpertVramPlan::try_new(geometry, exact_budget, &unsupported),
            Err(GpuNativeBootstrapError::ExpertStorageBuffersUnsupported { .. })
        ));
    }

    #[test]
    fn q4_expert_geometry_preserves_projection_layout_and_checked_stride() {
        let geometry = GpuNativeQ4ExpertGeometry::try_new(32, 64, 128, 8).unwrap();
        let blocks = 32 * 64 / Q4_0_BLOCK_ELEMS;
        assert_eq!(geometry.blocks_per_projection(), blocks);
        assert_eq!(geometry.gate_block_offset(), 0);
        assert_eq!(geometry.up_block_offset(), blocks);
        assert_eq!(geometry.down_block_offset(), blocks * 2);
        assert_eq!(
            geometry.logical_expert_bytes(),
            blocks * 3 * Q4_0_BLOCK_BYTES
        );
        assert_eq!(
            geometry.slot_stride_bytes(),
            (GPU_NATIVE_EXPERT_SLOT_EPOCH_BYTES + geometry.logical_expert_bytes() + 3) & !3
        );
        assert_eq!(geometry.payload_offset_bytes(), 4);
        assert_eq!(
            geometry.physical_payload_bytes(),
            geometry.slot_stride_bytes() - 4
        );
        assert!(geometry.slot_stride_bytes().is_multiple_of(4));
        assert_eq!(
            geometry.router_geometry(),
            GpuNativeRouterGeometry::try_new(32, 128, 8).unwrap()
        );

        for (d_model, d_ff) in [(31, 32), (32, 31)] {
            assert!(matches!(
                GpuNativeQ4ExpertGeometry::try_new(d_model, d_ff, 128, 8),
                Err(GpuNativeBootstrapError::ExpertQ4GeometryIncompatible { .. })
            ));
        }
        assert_eq!(
            GpuNativeQ4ExpertGeometry::try_new(32, 0, 128, 8),
            Err(GpuNativeBootstrapError::InvalidExpertDff { d_ff: 0 })
        );
        let overflowing_d_ff = usize::MAX & !(Q4_0_BLOCK_ELEMS - 1);
        assert!(matches!(
            GpuNativeQ4ExpertGeometry::try_new(32, overflowing_d_ff, 128, 8),
            Err(GpuNativeBootstrapError::ExpertGeometryOverflow { .. })
        ));
    }

    #[test]
    fn q4_expert_bank_plan_virtualizes_logical_experts_and_honors_binding_limits() {
        let geometry = GpuNativeQ4ExpertGeometry::try_new(32, 32, 128, 8).unwrap();
        let stride = geometry.slot_stride_bytes() as u32;
        let limits = wgpu::Limits {
            max_buffer_size: u64::from(stride) * 2,
            max_storage_buffer_binding_size: stride * 2,
            ..wgpu::Limits::default()
        };
        let layout = GpuNativeQ4ExpertArenaLayout::try_new(geometry, 8, &limits).unwrap();
        assert_eq!(layout.slot_capacity, 8);
        assert_eq!(layout.active_banks, 4);
        assert_eq!(layout.banks.map(|bank| bank.slot_capacity), [2, 2, 2, 2]);
        assert!(layout
            .banks
            .iter()
            .all(|bank| bank.allocation_bytes <= u64::from(stride) * 2));
        assert!(8 < geometry.num_experts());
        assert!(matches!(
            GpuNativeQ4ExpertArenaLayout::try_new(geometry, 9, &limits),
            Err(GpuNativeBootstrapError::ExpertArenaBankLimit {
                required_banks: 5,
                maximum_banks: MAX_GPU_NATIVE_EXPERT_BANKS,
            })
        ));
    }

    #[test]
    fn q4_expert_pipeline_limits_require_four_banks_and_fixed_dispatch_geometry() {
        let supported = wgpu::Limits {
            max_storage_buffers_per_shader_stage: GPU_NATIVE_EXPERT_REQUIRED_STORAGE_BUFFERS,
            max_push_constant_size: GPU_NATIVE_EXPERT_PUSH_CONSTANT_BYTES,
            max_compute_workgroup_size_x: GPU_NATIVE_EXPERT_WORKGROUP_SIZE,
            max_compute_invocations_per_workgroup: GPU_NATIVE_EXPERT_WORKGROUP_SIZE,
            ..wgpu::Limits::default()
        };
        assert_eq!(validate_q4_expert_pipeline_limits(&supported), Ok(()));
        assert!(matches!(
            validate_q4_expert_pipeline_limits(&wgpu::Limits {
                max_storage_buffers_per_shader_stage: GPU_NATIVE_EXPERT_REQUIRED_STORAGE_BUFFERS
                    - 1,
                ..supported.clone()
            }),
            Err(GpuNativeBootstrapError::ExpertStorageBuffersUnsupported { .. })
        ));
        assert!(matches!(
            validate_q4_expert_pipeline_limits(&wgpu::Limits {
                max_push_constant_size: GPU_NATIVE_EXPERT_PUSH_CONSTANT_BYTES - 1,
                ..supported.clone()
            }),
            Err(GpuNativeBootstrapError::ExpertPushConstantsUnsupported { .. })
        ));
        assert!(matches!(
            validate_q4_expert_pipeline_limits(&wgpu::Limits {
                max_compute_workgroup_size_x: GPU_NATIVE_EXPERT_WORKGROUP_SIZE - 1,
                ..supported
            }),
            Err(GpuNativeBootstrapError::ExpertWorkgroupUnsupported { .. })
        ));
    }

    #[test]
    fn q4_expert_mapping_and_payload_validation_fail_closed() {
        let geometry = GpuNativeQ4ExpertGeometry::try_new(32, 32, 128, 8).unwrap();
        let layout =
            GpuNativeQ4ExpertArenaLayout::try_new(geometry, 2, &wgpu::Limits::default()).unwrap();
        let first_payload = q4_uniform_expert(geometry, 0.01, 0.02, 0.005);
        let second_payload = q4_uniform_expert(geometry, -0.01, 0.03, 0.004);
        let first_location = GpuNativeQ4ExpertLocation::try_new(0, 0).unwrap();
        let second_location = GpuNativeQ4ExpertLocation::try_new(0, 1).unwrap();
        let uploads = [
            GpuNativeQ4ExpertUpload {
                logical_id: 7,
                logical_generation: 10,
                location: first_location,
                payload: &first_payload,
            },
            GpuNativeQ4ExpertUpload {
                logical_id: 91,
                logical_generation: 11,
                location: second_location,
                payload: &second_payload,
            },
        ];
        let prepared = validate_q4_expert_uploads(3, geometry, layout, &uploads).unwrap();
        assert_eq!(
            prepared.mapping[7].location(),
            first_location.pack().unwrap()
        );
        assert_eq!(prepared.mapping[7].slot_epoch(), 1);
        assert_eq!(
            prepared.mapping[91].location(),
            second_location.pack().unwrap()
        );
        assert_eq!(prepared.mapping[91].slot_epoch(), 1);
        assert!(prepared
            .mapping
            .iter()
            .enumerate()
            .all(|(id, &entry)| id == 7
                || id == 91
                || entry == GpuNativeQ4ExpertMappingEntry::UNMAPPED));
        assert_eq!(prepared.physical_slots[0].1[..4], 1u32.to_le_bytes());
        assert_eq!(
            GpuNativeQ4ExpertMappingEntry::try_new(first_location, 0),
            Err(GpuNativeBootstrapError::InvalidExpertSlotEpoch { slot_epoch: 0 })
        );
        assert_eq!(
            &prepared.physical_slots[0].1[4..4 + geometry.logical_expert_bytes()],
            first_payload.as_slice()
        );
        assert_eq!(
            GpuNativeQ4ExpertLocation::unpack(first_location.pack().unwrap()),
            Some(first_location)
        );
        assert_eq!(
            GpuNativeQ4ExpertLocation::unpack(GPU_NATIVE_EXPERT_UNMAPPED),
            None
        );
        assert!(
            GpuNativeQ4ExpertLocation::try_new(3, GPU_NATIVE_EXPERT_LOCATION_SLOT_MASK).is_err()
        );

        let duplicate_logical = [
            GpuNativeQ4ExpertUpload {
                logical_id: 7,
                logical_generation: 10,
                location: first_location,
                payload: &first_payload,
            },
            GpuNativeQ4ExpertUpload {
                logical_id: 7,
                logical_generation: 10,
                location: second_location,
                payload: &second_payload,
            },
        ];
        assert!(matches!(
            validate_q4_expert_uploads(3, geometry, layout, &duplicate_logical),
            Err(GpuNativeBootstrapError::DuplicateExpertLogicalId { logical_id: 7 })
        ));
        let duplicate_physical = [
            GpuNativeQ4ExpertUpload {
                logical_id: 7,
                logical_generation: 10,
                location: first_location,
                payload: &first_payload,
            },
            GpuNativeQ4ExpertUpload {
                logical_id: 8,
                logical_generation: 11,
                location: first_location,
                payload: &second_payload,
            },
        ];
        assert!(matches!(
            validate_q4_expert_uploads(3, geometry, layout, &duplicate_physical),
            Err(GpuNativeBootstrapError::DuplicateExpertPhysicalLocation { .. })
        ));
        let out_of_range = [GpuNativeQ4ExpertUpload {
            logical_id: 128,
            logical_generation: 10,
            location: first_location,
            payload: &first_payload,
        }];
        assert!(matches!(
            validate_q4_expert_uploads(3, geometry, layout, &out_of_range),
            Err(GpuNativeBootstrapError::ExpertLogicalIdOutOfRange { .. })
        ));
        let short = [GpuNativeQ4ExpertUpload {
            logical_id: 7,
            logical_generation: 10,
            location: first_location,
            payload: &first_payload[..first_payload.len() - 1],
        }];
        assert!(matches!(
            validate_q4_expert_uploads(3, geometry, layout, &short),
            Err(GpuNativeBootstrapError::ExpertPayloadTooShort { .. })
        ));
        let mut unexpected_trailing = first_payload.clone();
        unexpected_trailing.push(1);
        let trailing = [GpuNativeQ4ExpertUpload {
            logical_id: 7,
            logical_generation: 10,
            location: first_location,
            payload: &unexpected_trailing,
        }];
        assert!(matches!(
            validate_q4_expert_uploads(3, geometry, layout, &trailing),
            Err(GpuNativeBootstrapError::ExpertPayloadTrailingBytes { .. })
                | Err(GpuNativeBootstrapError::ExpertPayloadNonZeroPadding { .. })
        ));
    }

    #[test]
    fn mutable_expert_install_reserve_retire_reinstall_and_full_capacity_are_safe() {
        use std::cell::RefCell;

        let arena = test_mutable_expert_arena(1);
        let geometry = arena.geometry();
        let payload = q4_uniform_expert(geometry, 0.01, 0.02, 0.005);
        let key_a = GpuNativeQ4ExpertKey::new(3, 7, 10);

        let first = expect_expert_install(arena.acquire_with_unpublish(key_a, |_, _| {}).unwrap());
        assert_eq!(first.key(), key_a);
        assert_eq!(first.reserved_residency().slot_epoch(), 1);
        assert_eq!(arena.residency_snapshot().installing_slots, 1);
        assert_eq!(arena.residency_snapshot().expert_mapping_publications, 0);
        assert!(matches!(
            arena.acquire_with_unpublish(key_a, |_, _| {}).unwrap(),
            GpuNativeQ4ExpertAcquire::InstallInProgress
        ));
        drop(first);
        let cancelled = arena.residency_snapshot();
        assert_eq!(cancelled.free_slots, 1);
        assert_eq!(cancelled.expert_install_cancellations, 1);

        let permit = expect_expert_install(arena.acquire_with_unpublish(key_a, |_, _| {}).unwrap());
        assert_eq!(permit.reserved_residency().slot_epoch(), 2);
        let order = RefCell::new(Vec::new());
        let physical = RefCell::new(Vec::new());
        let mapping = RefCell::new(Vec::new());
        let residency_a = permit
            .install_with_writes(
                &payload,
                |bank, offset, bytes| {
                    order.borrow_mut().push("physical");
                    physical.borrow_mut().push((bank, offset, bytes.to_vec()));
                },
                |offset, entry| {
                    order.borrow_mut().push("mapping");
                    mapping.borrow_mut().push((offset, entry));
                },
            )
            .unwrap();
        assert_eq!(&*order.borrow(), &["physical", "mapping"]);
        assert_eq!(physical.borrow()[0].2[..4], 2u32.to_le_bytes());
        assert_eq!(
            &physical.borrow()[0].2[4..4 + geometry.logical_expert_bytes()],
            payload.as_slice()
        );
        assert_eq!(
            mapping.borrow()[0].0,
            7 * GPU_NATIVE_EXPERT_MAPPING_ENTRY_BYTES as u64
        );
        assert_eq!(mapping.borrow()[0].1, residency_a.mapping_entry().unwrap());
        assert!(matches!(
            arena.acquire_with_unpublish(key_a, |_, _| {}).unwrap(),
            GpuNativeQ4ExpertAcquire::Hit(hit) if hit == residency_a
        ));
        assert!(matches!(
            arena
                .acquire_with_unpublish(GpuNativeQ4ExpertKey::new(3, 8, 1), |_, _| {})
                .unwrap(),
            GpuNativeQ4ExpertAcquire::NoPhysicalSlot
        ));
        assert_eq!(
            arena
                .retire_with_unpublish(GpuNativeQ4ExpertKey::new(3, 7, 9), |_, _| {})
                .unwrap(),
            GpuNativeQ4ExpertRetire::StaleRequester
        );
        let unpublishes = RefCell::new(Vec::new());
        assert_eq!(
            arena
                .retire_with_unpublish(key_a, |offset, entry| {
                    unpublishes.borrow_mut().push((offset, entry));
                })
                .unwrap(),
            GpuNativeQ4ExpertRetire::Retired
        );
        assert_eq!(
            unpublishes.borrow().as_slice(),
            &[(
                7 * GPU_NATIVE_EXPERT_MAPPING_ENTRY_BYTES as u64,
                GpuNativeQ4ExpertMappingEntry::UNMAPPED,
            )]
        );

        let reinstall =
            expect_expert_install(arena.acquire_with_unpublish(key_a, |_, _| {}).unwrap());
        assert_eq!(reinstall.reserved_residency().slot_epoch(), 3);
        let reinstalled = reinstall
            .install_with_writes(&payload, |_, _, _| {}, |_, _| {})
            .unwrap();
        assert_eq!(reinstalled.key().logical_generation(), 10);
        assert_eq!(arena.residency_snapshot().expert_slot_reuses, 1);
    }

    #[test]
    fn mutable_expert_newer_generation_cancels_older_and_stale_completion_cannot_publish() {
        use std::cell::Cell;

        let arena = test_mutable_expert_arena(1);
        let payload = q4_uniform_expert(arena.geometry(), 0.01, 0.02, 0.005);
        let old_key = GpuNativeQ4ExpertKey::new(3, 7, 10);
        let new_key = GpuNativeQ4ExpertKey::new(3, 7, 12);
        let old_permit =
            expect_expert_install(arena.acquire_with_unpublish(old_key, |_, _| {}).unwrap());
        let new_permit =
            expect_expert_install(arena.acquire_with_unpublish(new_key, |_, _| {}).unwrap());
        assert_eq!(new_permit.reserved_residency().slot_epoch(), 2);

        let stale_writes = Cell::new(0);
        assert_eq!(
            old_permit.install_with_writes(
                &payload,
                |_, _, _| stale_writes.set(stale_writes.get() + 1),
                |_, _| stale_writes.set(stale_writes.get() + 1),
            ),
            Err(GpuNativeBootstrapError::ExpertInstallReservationLost)
        );
        assert_eq!(stale_writes.get(), 0);
        let current = new_permit
            .install_with_writes(&payload, |_, _, _| {}, |_, _| {})
            .unwrap();
        assert_eq!(current.key(), new_key);
        assert!(matches!(
            arena.acquire_with_unpublish(old_key, |_, _| {}).unwrap(),
            GpuNativeQ4ExpertAcquire::StaleRequester
        ));
        assert_eq!(
            arena.retire_with_unpublish(old_key, |_, _| {}).unwrap(),
            GpuNativeQ4ExpertRetire::StaleRequester
        );
        assert!(matches!(
            arena.acquire_with_unpublish(new_key, |_, _| {}).unwrap(),
            GpuNativeQ4ExpertAcquire::Hit(hit) if hit == current
        ));
        let newest_key = GpuNativeQ4ExpertKey::new(3, 7, 13);
        let unpublishes = Cell::new(0);
        let newest_permit = expect_expert_install(
            arena
                .acquire_with_unpublish(newest_key, |_, entry| {
                    assert_eq!(entry, GpuNativeQ4ExpertMappingEntry::UNMAPPED);
                    unpublishes.set(unpublishes.get() + 1);
                })
                .unwrap(),
        );
        assert_eq!(unpublishes.get(), 1);
        assert_eq!(
            arena.retire_with_unpublish(new_key, |_, _| {}).unwrap(),
            GpuNativeQ4ExpertRetire::StaleRequester
        );
        let newest = newest_permit
            .install_with_writes(&payload, |_, _, _| {}, |_, _| {})
            .unwrap();
        assert!(matches!(
            arena.acquire_with_unpublish(newest_key, |_, _| {}).unwrap(),
            GpuNativeQ4ExpertAcquire::Hit(hit) if hit == newest
        ));
        let snapshot = arena.residency_snapshot();
        assert_eq!(snapshot.resident_slots, 1);
        assert_eq!(snapshot.expert_stale_install_rejections, 2);
        assert_eq!(snapshot.expert_install_cancellations, 1);
        assert_eq!(snapshot.expert_slot_retires, 1);
        assert_eq!(snapshot.expert_mapping_unpublications, 1);
    }

    #[test]
    fn mutable_expert_payload_failure_writes_nothing_and_epoch_exhaustion_fails_closed() {
        use std::cell::Cell;

        let arena = test_mutable_expert_arena(1);
        let key = GpuNativeQ4ExpertKey::new(3, 7, 1);
        let payload = q4_uniform_expert(arena.geometry(), 0.01, 0.02, 0.005);
        let permit = expect_expert_install(arena.acquire_with_unpublish(key, |_, _| {}).unwrap());
        let writes = Cell::new(0);
        assert!(matches!(
            permit.install_with_writes(
                &payload[..payload.len() - 1],
                |_, _, _| writes.set(writes.get() + 1),
                |_, _| writes.set(writes.get() + 1),
            ),
            Err(GpuNativeBootstrapError::ExpertPayloadTooShort { .. })
        ));
        assert_eq!(writes.get(), 0);
        assert_eq!(arena.residency_snapshot().free_slots, 1);

        arena.state.lock().slots[0].last_epoch = u32::MAX;
        assert_eq!(
            arena.acquire_with_unpublish(key, |_, _| {}).err(),
            Some(GpuNativeBootstrapError::ExpertSlotEpochExhausted { bank: 0, slot: 0 })
        );
        assert_eq!(arena.residency_snapshot().resident_slots, 0);
    }

    #[test]
    fn q4_expert_route_resolution_contains_prior_failure_and_latches_real_miss() {
        let geometry = GpuNativeQ4ExpertGeometry::try_new(32, 32, 128, 8).unwrap();
        let layout =
            GpuNativeQ4ExpertArenaLayout::try_new(geometry, 8, &wgpu::Limits::default()).unwrap();
        let mut mapping = vec![GpuNativeQ4ExpertMappingEntry::UNMAPPED; 128];
        mapping[0] = GpuNativeQ4ExpertMappingEntry::try_new(
            GpuNativeQ4ExpertLocation::try_new(0, 0).unwrap(),
            1,
        )
        .unwrap();
        mapping[1] = GpuNativeQ4ExpertMappingEntry {
            location: GpuNativeQ4ExpertLocation::try_new(0, 1)
                .unwrap()
                .pack()
                .unwrap(),
            slot_epoch: 0,
        };
        let (resolved, status) = resolve_routes_mirror(
            &[0; 8],
            &mapping,
            layout,
            GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE,
        );
        assert_eq!(resolved, vec![GpuNativeQ4ExpertMappingEntry::UNMAPPED; 8]);
        assert_eq!(status, GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE);
        assert_eq!(status & GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS, 0);

        let (resolved, status) = resolve_routes_mirror(&[0, 1, 17], &mapping, layout, 0);
        assert_ne!(resolved[0], GpuNativeQ4ExpertMappingEntry::UNMAPPED);
        assert_eq!(resolved[1], GpuNativeQ4ExpertMappingEntry::UNMAPPED);
        assert_eq!(resolved[2], GpuNativeQ4ExpertMappingEntry::UNMAPPED);
        assert_ne!(status & GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS, 0);
    }

    #[test]
    fn stale_resolved_route_epoch_cannot_execute_reused_slot_cpu_mirror() {
        let arena = test_mutable_expert_arena(1);
        let payload_a = q4_uniform_expert(arena.geometry(), 0.01, 0.02, 0.005);
        let payload_b = q4_uniform_expert(arena.geometry(), -0.02, 0.03, 0.004);
        let key_a = GpuNativeQ4ExpertKey::new(3, 7, 10);
        let key_b = GpuNativeQ4ExpertKey::new(3, 8, 20);
        let a = expect_expert_install(arena.acquire_with_unpublish(key_a, |_, _| {}).unwrap())
            .install_with_writes(&payload_a, |_, _, _| {}, |_, _| {})
            .unwrap();
        assert_eq!(
            arena.retire_with_unpublish(key_a, |_, _| {}).unwrap(),
            GpuNativeQ4ExpertRetire::Retired
        );
        let b = expect_expert_install(arena.acquire_with_unpublish(key_b, |_, _| {}).unwrap())
            .install_with_writes(&payload_b, |_, _, _| {}, |_, _| {})
            .unwrap();
        assert_eq!(a.location(), b.location());
        assert_ne!(a.slot_epoch(), b.slot_epoch());
        assert!(!resolved_route_epoch_matches(
            a.mapping_entry().unwrap(),
            b.slot_epoch()
        ));
        assert!(resolved_route_epoch_matches(
            b.mapping_entry().unwrap(),
            b.slot_epoch()
        ));
        assert_eq!(arena.residency_snapshot().expert_slot_reuses, 1);
    }

    #[test]
    fn q4_expert_layer_ownership_and_scratch_geometry_are_typed() {
        let geometry = GpuNativeQ4ExpertGeometry::try_new(32, 32, 128, 8).unwrap();
        let limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: GPU_NATIVE_EXPERT_REQUIRED_STORAGE_BUFFERS,
            max_push_constant_size: GPU_NATIVE_EXPERT_PUSH_CONSTANT_BYTES,
            max_compute_workgroup_size_x: GPU_NATIVE_EXPERT_WORKGROUP_SIZE,
            max_compute_invocations_per_workgroup: GPU_NATIVE_EXPERT_WORKGROUP_SIZE,
            ..wgpu::Limits::default()
        };
        let arena_layout = GpuNativeQ4ExpertArenaLayout::try_new(geometry, 8, &limits).unwrap();
        let budget = arena_layout.total_allocation_bytes().unwrap();
        let arena_plan = GpuNativeQ4ExpertVramPlan::try_new(geometry, budget, &limits).unwrap();
        assert_eq!(arena_plan.slot_capacity(), 8);
        let arena_state = GpuNativeQ4ExpertArenaState::new(geometry, arena_plan.layout).unwrap();
        let arena = GpuNativeQ4ExpertArena::from_buffers(
            7,
            3,
            arena_plan,
            [(); MAX_GPU_NATIVE_EXPERT_BANKS],
            (),
            arena_state,
        );
        let gate_layout = GpuNativeDenseWeightLayout::try_new(
            GpuNativeDenseWeightKind::F32,
            128,
            32,
            128 * 32 * std::mem::size_of::<f32>(),
        )
        .unwrap();
        let plan = GpuNativeRouterPlan {
            context_id: 7,
            layer_index: 2,
            geometry: geometry.router_geometry(),
            gate: GpuNativeDenseWeightHandle {
                context_id: 7,
                weight_id: 1,
                key: GpuNativeDenseWeightKey::try_new("test.expert.router").unwrap(),
                layout: gate_layout,
            },
        };
        assert_eq!(
            validate_router_expert_geometry(&plan, &arena),
            Err(GpuNativeBootstrapError::ExpertLayerMismatch {
                router_layer: 2,
                expert_layer: 3,
            })
        );
        assert_eq!(
            validate_q4_expert_arena(8, &arena, &limits),
            Err(GpuNativeBootstrapError::ForeignExpertArena)
        );

        let scratch_layout = GpuNativeQ4ExpertScratchLayout::try_new(geometry).unwrap();
        let scratch = GpuNativeQ4ExpertScratch::from_buffers(
            7,
            1,
            scratch_layout,
            (),
            test_scratch(7, 2, geometry.d_ff, ()),
            test_scratch(7, 3, geometry.top_k * geometry.d_model, ()),
            test_scratch(7, 4, geometry.d_model, ()),
        );
        validate_q4_expert_scratch(7, geometry, &scratch, &limits).unwrap();
        assert_eq!(
            validate_q4_expert_scratch(8, geometry, &scratch, &limits),
            Err(GpuNativeBootstrapError::ForeignExpertScratch)
        );
        assert!(
            GpuNativeQ4ExpertScratchLayout::resolved_usage().contains(wgpu::BufferUsages::COPY_SRC)
        );
        assert!(!GpuNativeQ4ExpertScratchLayout::resolved_usage()
            .intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));
        assert!(!GpuNativeQ4ExpertScratchLayout::tensor_usage()
            .intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));
        assert_eq!(
            scratch_layout.total_bytes(),
            scratch_layout.resolved_locations_bytes()
                + scratch_layout.activation_bytes()
                + scratch_layout.route_outputs_bytes()
                + scratch_layout.combined_bytes()
        );
    }

    #[test]
    fn q4_expert_cpu_mirror_covers_swiglu_weighted_combine_and_residual() {
        let geometry = GpuNativeQ4ExpertGeometry::try_new(32, 32, 128, 2).unwrap();
        let hidden = vec![1.0; geometry.d_model];
        let first_payload = q4_uniform_expert(geometry, 0.01, 0.02, 0.005);
        let second_payload = q4_uniform_expert(geometry, -0.005, 0.015, 0.002);
        let first = q4_expert_mirror(&first_payload, geometry, &hidden);
        let second = q4_expert_mirror(&second_payload, geometry, &hidden);
        assert!(first.iter().all(|value| value.is_finite()));
        assert!(second.iter().all(|value| value.is_finite()));
        assert!(first.windows(2).all(|values| values[0] == values[1]));
        assert!(second.windows(2).all(|values| values[0] == values[1]));
        let combined =
            weighted_expert_combine_mirror(&[first.clone(), second.clone()], &[0.25, 0.75]);
        let expected = vec![first[0] * 0.25 + second[0] * 0.75; geometry.d_model];
        assert_close(&combined, &expected, 1e-7);
        let residual = vec![0.5; geometry.d_model];
        assert_close(
            &residual_add_mirror(&residual, &combined),
            &combined.iter().map(|value| value + 0.5).collect::<Vec<_>>(),
            1e-7,
        );
        let failed = fail_closed_expert_combine_mirror(
            &[first.clone(), second.clone()],
            &[0.25, 0.75],
            GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS,
        );
        assert_eq!(failed, vec![0.0; geometry.d_model]);
        let mut non_finite = second;
        non_finite[3] = f32::NAN;
        assert_eq!(
            fail_closed_expert_combine_mirror(&[first, non_finite], &[0.25, 0.75], 0),
            vec![0.0; geometry.d_model]
        );
    }

    #[test]
    fn router_softmax_topk_matches_linear_gate_and_is_deterministic() {
        use crate::gating::LinearGate;
        use std::collections::HashSet;

        let (known_ids, known_weights, failed) =
            router_topk_mirror(&[0.25, 2.0, -1.0, 2.0, 1.0], 3);
        assert!(!failed);
        assert_eq!(known_ids, vec![1, 3, 4]);
        assert!((known_weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        let (tie_ids, tie_weights, failed) = router_topk_mirror(&[0.0; 128], 8);
        assert!(!failed);
        assert_eq!(tie_ids, (0..8).collect::<Vec<_>>());
        assert!(tie_weights.iter().all(|&weight| weight == 0.125));

        const D_MODEL: usize = 16;
        const NUM_EXPERTS: usize = 128;
        const TOP_K: usize = 8;
        let hidden = [
            1.0, -0.5, 0.25, 0.75, -1.0, 0.5, 0.0, 0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut gate_values = vec![0.0; NUM_EXPERTS * D_MODEL];
        for expert in 0..NUM_EXPERTS {
            gate_values[expert * D_MODEL] = expert as f32 / 10.0;
            gate_values[expert * D_MODEL + 1] = (expert % 3) as f32 / 100.0;
        }
        let gate = LinearGate::new(gate_values, NUM_EXPERTS, D_MODEL, TOP_K);
        let cpu_reference = gate.route(&hidden);
        let logits = gate.weights.matvec(&hidden);
        let first = router_topk_mirror(&logits, TOP_K);
        let second = router_topk_mirror(&logits, TOP_K);
        assert_eq!(first, second);
        assert!(!first.2);
        assert_eq!(first.0, cpu_reference.experts);
        assert_close(&first.1, &cpu_reference.weights, 1e-6);
        assert_eq!(first.0.len(), TOP_K);
        assert_eq!(first.0.iter().copied().collect::<HashSet<_>>().len(), TOP_K);
        assert!((first.1.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        let (ids, weights, failed) = router_topk_mirror(&[0.0, f32::NAN, 1.0], 2);
        assert!(failed);
        assert_eq!(ids, vec![0, 0]);
        assert_eq!(weights, vec![0.0, 0.0]);
    }

    #[test]
    fn router_plan_accepts_existing_dense_kinds_and_rejects_foreign_stale_or_wrong_gate() {
        let context_id = 41;
        let geometry = GpuNativeRouterGeometry::try_new(16, 128, 8).unwrap();
        let mut registry = GpuNativeDenseWeightRegistry::new(context_id);
        let gate = insert_test_f32_weight(
            &mut registry,
            1,
            "layer.2.router.gate",
            geometry.num_experts,
            geometry.d_model,
        );
        let plan = GpuNativeRouterPlan {
            context_id,
            layer_index: 2,
            geometry,
            gate,
        };
        assert_eq!(plan.layer_index(), 2);
        assert_eq!(plan.geometry(), geometry);
        assert!(
            validate_router_plan_with_registry(context_id, geometry.d_model, &registry, &plan)
                .is_ok()
        );

        let q8_elements = geometry.num_experts * geometry.d_model;
        let q8_layout = GpuNativeDenseWeightLayout::try_new(
            GpuNativeDenseWeightKind::Q8_0,
            geometry.num_experts,
            geometry.d_model,
            q8_elements.div_ceil(Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES,
        )
        .unwrap();
        let q8_gate = registry
            .insert(test_weight(2, "layer.3.router.q8_gate", q8_layout, ()))
            .unwrap();
        let q8_plan = GpuNativeRouterPlan {
            context_id,
            layer_index: 3,
            geometry,
            gate: q8_gate,
        };
        assert_eq!(
            validate_router_plan_with_registry(context_id, geometry.d_model, &registry, &q8_plan)
                .unwrap()
                .layout
                .kind,
            GpuNativeDenseWeightKind::Q8_0
        );

        let mut foreign_plan = plan.clone();
        foreign_plan.context_id += 1;
        assert!(matches!(
            validate_router_plan_with_registry(
                context_id,
                geometry.d_model,
                &registry,
                &foreign_plan
            ),
            Err(GpuNativeBootstrapError::ForeignRouterPlan)
        ));

        let mut foreign_gate = plan.clone();
        foreign_gate.gate.context_id += 1;
        assert!(matches!(
            validate_router_plan_with_registry(
                context_id,
                geometry.d_model,
                &registry,
                &foreign_gate
            ),
            Err(GpuNativeBootstrapError::ForeignDenseWeightHandle)
        ));

        let mut stale_gate = plan.clone();
        stale_gate.gate.weight_id += 1;
        assert!(matches!(
            validate_router_plan_with_registry(
                context_id,
                geometry.d_model,
                &registry,
                &stale_gate
            ),
            Err(GpuNativeBootstrapError::StaleDenseWeightHandle { .. })
        ));

        let transposed = insert_test_f32_weight(
            &mut registry,
            3,
            "layer.2.router.transposed_gate",
            geometry.d_model,
            geometry.num_experts,
        );
        let mut wrong_shape = plan;
        wrong_shape.gate = transposed;
        assert!(matches!(
            validate_router_plan_with_registry(
                context_id,
                geometry.d_model,
                &registry,
                &wrong_shape
            ),
            Err(GpuNativeBootstrapError::RouterGateShape { .. })
        ));
        assert!(matches!(
            validate_router_plan_with_registry(
                context_id,
                geometry.d_model + 1,
                &registry,
                &q8_plan
            ),
            Err(GpuNativeBootstrapError::RouterDModelMismatch { .. })
        ));
    }

    #[test]
    fn router_scratch_layout_and_ownership_fail_closed() {
        let geometry = GpuNativeRouterGeometry::try_new(16, 128, 8).unwrap();
        let layout = GpuNativeRouterScratchLayout::try_new(geometry).unwrap();
        assert_eq!(layout.geometry(), geometry);
        assert_eq!(layout.logits_bytes(), 512);
        assert_eq!(layout.selected_ids_bytes(), 32);
        assert_eq!(layout.selected_weights_bytes(), 32);
        assert_eq!(layout.total_bytes(), 576);
        assert!(!GpuNativeRouterScratchLayout::logits_usage()
            .intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));
        assert!(!GpuNativeRouterScratchLayout::result_usage()
            .intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));
        assert!(GpuNativeRouterScratchLayout::result_usage().contains(wgpu::BufferUsages::COPY_SRC));

        let scratch = GpuNativeRouterScratch::from_buffers(7, 1, layout, (), (), ());
        assert_eq!(validate_router_scratch(7, geometry, &scratch), Ok(()));
        assert_eq!(
            validate_router_scratch(8, geometry, &scratch),
            Err(GpuNativeBootstrapError::ForeignRouterScratch)
        );

        let smaller = GpuNativeRouterGeometry::try_new(16, 64, 4).unwrap();
        assert_eq!(
            validate_router_scratch(7, smaller, &scratch),
            Err(GpuNativeBootstrapError::RouterScratchGeometry {
                expected: smaller,
                actual: geometry,
            })
        );

        let mut wrong_logits_layout = layout;
        wrong_logits_layout.logits_elements -= 1;
        let wrong_logits =
            GpuNativeRouterScratch::from_buffers(7, 2, wrong_logits_layout, (), (), ());
        assert!(matches!(
            validate_router_scratch(7, geometry, &wrong_logits),
            Err(GpuNativeBootstrapError::RouterLogitsLength { .. })
        ));
        let mut wrong_ids_layout = layout;
        wrong_ids_layout.selected_ids_elements -= 1;
        let wrong_ids = GpuNativeRouterScratch::from_buffers(7, 3, wrong_ids_layout, (), (), ());
        assert!(matches!(
            validate_router_scratch(7, geometry, &wrong_ids),
            Err(GpuNativeBootstrapError::RouterSelectedIdsLength { .. })
        ));
        let mut wrong_weights_layout = layout;
        wrong_weights_layout.selected_weights_elements -= 1;
        let wrong_weights =
            GpuNativeRouterScratch::from_buffers(7, 4, wrong_weights_layout, (), (), ());
        assert!(matches!(
            validate_router_scratch(7, geometry, &wrong_weights),
            Err(GpuNativeBootstrapError::RouterSelectedWeightsLength { .. })
        ));
    }

    #[test]
    fn attention_geometry_accepts_qwen_gqa_and_non_power_of_two_head_counts() {
        let geometry = GpuNativeAttentionGeometry::try_new(12, 6, 2, 4, 4).unwrap();
        assert_eq!(geometry.d_model(), 12);
        assert_eq!(geometry.num_heads(), 6);
        assert_eq!(geometry.num_kv_heads(), 2);
        assert_eq!(geometry.head_dim(), 4);
        assert_eq!(geometry.rope_dim(), 4);
        assert_eq!(geometry.q_width(), 24);
        assert_eq!(geometry.kv_width(), 8);

        assert_eq!(
            GpuNativeAttentionGeometry::try_new(12, 0, 2, 4, 4),
            Err(GpuNativeBootstrapError::InvalidAttentionHeadCount {
                tensor: GpuNativeAttentionTensor::Query,
                heads: 0,
            })
        );
        assert_eq!(
            GpuNativeAttentionGeometry::try_new(12, 6, 0, 4, 4),
            Err(GpuNativeBootstrapError::InvalidAttentionHeadCount {
                tensor: GpuNativeAttentionTensor::Key,
                heads: 0,
            })
        );
        assert!(matches!(
            GpuNativeAttentionGeometry::try_new(12, 6, 4, 4, 4),
            Err(GpuNativeBootstrapError::InvalidAttentionHeadGeometry { .. })
        ));
        assert_eq!(
            GpuNativeAttentionGeometry::try_new(12, 6, 2, 4, 3),
            Err(GpuNativeBootstrapError::OddRopeDimension { rope_dim: 3 })
        );
        assert_eq!(
            GpuNativeAttentionGeometry::try_new(12, 6, 2, 4, 6),
            Err(GpuNativeBootstrapError::InvalidRopeDimension {
                rope_dim: 6,
                head_dim: 4,
            })
        );
        assert!(matches!(
            GpuNativeAttentionGeometry::try_new(12, u32::MAX as usize, 1, 2, 2),
            Err(GpuNativeBootstrapError::AttentionGeometryOverflow { .. })
        ));
    }

    #[test]
    fn attention_geometry_keeps_q_width_distinct_from_d_model() {
        let geometry = GpuNativeAttentionGeometry::try_new(6, 4, 2, 2, 2).unwrap();
        assert_eq!(geometry.d_model(), 6);
        assert_eq!(geometry.q_width(), 8);
        assert_eq!(geometry.kv_width(), 4);
        assert_ne!(geometry.q_width(), geometry.d_model());
    }

    #[test]
    fn causal_attention_dispatch_fails_closed_for_device_limits() {
        let geometry = GpuNativeAttentionGeometry::try_new(6, 4, 2, 2, 2).unwrap();
        assert_eq!(
            validate_causal_attention_dispatch(geometry, &wgpu::Limits::default()),
            Ok(4)
        );
        let narrow_dispatch = wgpu::Limits {
            max_compute_workgroups_per_dimension: 3,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            validate_causal_attention_dispatch(geometry, &narrow_dispatch),
            Err(GpuNativeBootstrapError::DispatchGeometryUnsupported {
                workgroups: 4,
                maximum: 3,
            })
        );
        let narrow_workgroup = wgpu::Limits {
            max_compute_workgroup_size_x: 16,
            max_compute_invocations_per_workgroup: 16,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            validate_causal_attention_dispatch(geometry, &narrow_workgroup),
            Err(GpuNativeBootstrapError::AttentionWorkgroupUnsupported {
                required: 32,
                max_size_x: 16,
                max_invocations: 16,
            })
        );
    }

    #[test]
    fn attention_plan_validates_layer_and_output_projection_shape() {
        let context_id = 41;
        let geometry = GpuNativeAttentionGeometry::try_new(6, 4, 2, 2, 2).unwrap();
        let mut registry = GpuNativeDenseWeightRegistry::new(context_id);
        let q_projection = insert_test_f32_weight(
            &mut registry,
            1,
            "layer.1.attention.q",
            geometry.q_width,
            geometry.d_model,
        );
        let k_projection = insert_test_f32_weight(
            &mut registry,
            2,
            "layer.1.attention.k",
            geometry.kv_width,
            geometry.d_model,
        );
        let v_projection = insert_test_f32_weight(
            &mut registry,
            3,
            "layer.1.attention.v",
            geometry.kv_width,
            geometry.d_model,
        );
        let o_projection = insert_test_f32_weight(
            &mut registry,
            4,
            "layer.1.attention.o",
            geometry.d_model,
            geometry.q_width,
        );
        let rope_dense = insert_test_f32_weight(
            &mut registry,
            5,
            "layer.1.attention.rope",
            1,
            geometry.rope_dim / 2,
        );
        let plan = GpuNativeAttentionPlan {
            context_id,
            layer_index: 1,
            geometry,
            q_projection,
            k_projection,
            v_projection,
            o_projection,
            q_norm: None,
            k_norm: None,
            rope: GpuNativeRopeHandle {
                dense: rope_dense,
                rope_dim: geometry.rope_dim,
                attention_factor_bits: 1.0f32.to_bits(),
            },
        };
        assert_eq!(plan.layer_index(), 1);
        assert_eq!(
            validate_attention_plan_with_registry(context_id, geometry.d_model, &registry, &plan),
            Ok(())
        );

        let transposed_o = insert_test_f32_weight(
            &mut registry,
            6,
            "layer.1.attention.o_transposed",
            geometry.q_width,
            geometry.d_model,
        );
        let mut wrong = plan.clone();
        wrong.o_projection = transposed_o;
        assert_eq!(
            validate_attention_plan_with_registry(context_id, geometry.d_model, &registry, &wrong),
            Err(GpuNativeBootstrapError::AttentionProjectionShape {
                tensor: GpuNativeAttentionTensor::Output,
                expected_rows: geometry.d_model,
                expected_cols: geometry.q_width,
                actual_rows: geometry.q_width,
                actual_cols: geometry.d_model,
            })
        );

        let square_o = insert_test_f32_weight(
            &mut registry,
            7,
            "layer.1.attention.o_square",
            geometry.d_model,
            geometry.d_model,
        );
        wrong.o_projection = square_o;
        assert_eq!(
            validate_attention_plan_with_registry(context_id, geometry.d_model, &registry, &wrong),
            Err(GpuNativeBootstrapError::AttentionProjectionShape {
                tensor: GpuNativeAttentionTensor::Output,
                expected_rows: geometry.d_model,
                expected_cols: geometry.q_width,
                actual_rows: geometry.d_model,
                actual_cols: geometry.d_model,
            })
        );

        wrong = plan.clone();
        wrong.o_projection.context_id += 1;
        assert_eq!(
            validate_attention_plan_with_registry(context_id, geometry.d_model, &registry, &wrong),
            Err(GpuNativeBootstrapError::ForeignDenseWeightHandle)
        );
        wrong = plan.clone();
        wrong.o_projection.weight_id += 1;
        assert!(matches!(
            validate_attention_plan_with_registry(context_id, geometry.d_model, &registry, &wrong),
            Err(GpuNativeBootstrapError::StaleDenseWeightHandle { .. })
        ));
    }

    #[test]
    fn layer_bound_kv_and_causal_sequence_lengths_fail_closed() {
        let geometry = GpuNativeAttentionGeometry::try_new(6, 4, 2, 2, 2).unwrap();
        let layout =
            GpuNativeKvLayout::try_new(2, 4, geometry.kv_width, &wgpu::Limits::default()).unwrap();
        let kv = GpuNativeKvState::from_layers(
            7,
            1,
            layout,
            vec![
                GpuNativeKvLayer { key: (), value: () },
                GpuNativeKvLayer { key: (), value: () },
            ],
        );
        assert_eq!(validate_attention_kv_state(7, geometry, &kv, 1, 0), Ok(1));
        assert_eq!(validate_attention_kv_state(7, geometry, &kv, 1, 2), Ok(3));
        assert_eq!(validate_attention_kv_state(7, geometry, &kv, 1, 3), Ok(4));
        assert_eq!(
            validate_attention_kv_state(7, geometry, &kv, 2, 0),
            Err(GpuNativeBootstrapError::AttentionPlanLayerOutOfRange {
                layer_index: 2,
                num_layers: 2,
            })
        );
        assert_eq!(
            validate_attention_kv_state(7, geometry, &kv, 1, 4),
            Err(GpuNativeBootstrapError::InvalidAttentionSequenceLength {
                seq_len: 5,
                max_seq_len: 4,
            })
        );
        assert_eq!(
            validate_attention_kv_state(7, geometry, &kv, 1, usize::MAX),
            Err(GpuNativeBootstrapError::AttentionSequenceLengthOverflow {
                position: usize::MAX,
            })
        );
    }

    #[test]
    fn qk_norm_groups_are_independent_for_gqa_geometry() {
        let geometry = GpuNativeAttentionGeometry::try_new(12, 6, 2, 4, 4).unwrap();
        let gain = [0.7, 1.1, 0.9, 1.3];
        let epsilon = 1e-6;
        let q = (0..geometry.q_width)
            .map(|index| (index as f32 - 11.0) / 3.0)
            .collect::<Vec<_>>();
        let k = (0..geometry.kv_width)
            .map(|index| (index as f32 - 3.0) / 2.0)
            .collect::<Vec<_>>();
        let q_norm = rms_norm_mirror(&q, &gain, epsilon, geometry.num_heads, geometry.head_dim);
        let k_norm = rms_norm_mirror(&k, &gain, epsilon, geometry.num_kv_heads, geometry.head_dim);
        let cpu_norm = crate::transformer::RmsNorm::new(gain.to_vec(), epsilon);
        let expected_q = q
            .chunks_exact(geometry.head_dim)
            .flat_map(|head| cpu_norm.forward(head))
            .collect::<Vec<_>>();
        let expected_k = k
            .chunks_exact(geometry.head_dim)
            .flat_map(|head| cpu_norm.forward(head))
            .collect::<Vec<_>>();
        assert_eq!(q_norm, expected_q);
        assert_eq!(k_norm, expected_k);

        let mut changed_q = q.clone();
        changed_q[5 * geometry.head_dim..].fill(10_000.0);
        let changed_q_norm = rms_norm_mirror(
            &changed_q,
            &gain,
            epsilon,
            geometry.num_heads,
            geometry.head_dim,
        );
        assert_eq!(
            &q_norm[..5 * geometry.head_dim],
            &changed_q_norm[..5 * geometry.head_dim]
        );
    }

    #[test]
    fn rope_mirror_matches_cpu_pairing_positions_and_partial_tail() {
        const GROUPS: usize = 3;
        const HEAD_DIM: usize = 6;
        const ROPE_DIM: usize = 4;
        let inverse_frequencies = [1.0, 0.01];
        let values = (0..GROUPS * HEAD_DIM)
            .map(|index| (index as f32 - 7.0) / 4.0)
            .collect::<Vec<_>>();
        assert_eq!(
            rope_mirror(
                &values,
                GROUPS,
                HEAD_DIM,
                ROPE_DIM,
                0,
                &inverse_frequencies,
                1.0,
            ),
            values
        );

        let actual = rope_mirror(
            &values,
            GROUPS,
            HEAD_DIM,
            ROPE_DIM,
            7,
            &inverse_frequencies,
            1.0,
        );
        let mut expected = values.clone();
        for head in expected.chunks_exact_mut(HEAD_DIM) {
            crate::transformer::apply_rope_inplace(&mut head[..ROPE_DIM], 7, 10_000.0);
        }
        assert_close(&actual, &expected, 1e-6);
        for head in 0..GROUPS {
            let start = head * HEAD_DIM;
            assert_eq!(
                &actual[start + ROPE_DIM..start + HEAD_DIM],
                &values[start + ROPE_DIM..start + HEAD_DIM]
            );
        }

        let one_head = rope_mirror(&[1.0, 2.0, 3.0, 4.0], 1, 4, 4, 1, &inverse_frequencies, 1.0);
        let mut cpu_pairing = vec![1.0, 2.0, 3.0, 4.0];
        crate::transformer::apply_rope_inplace(&mut cpu_pairing, 1, 10_000.0);
        assert_close(&one_head, &cpu_pairing, 1e-6);
    }

    #[test]
    fn rope_parameters_and_typed_handle_fail_closed() {
        let layout = GpuNativeRopeLayout::try_new(4, 4).unwrap();
        assert_close(
            &standard_rope_inverse_frequencies(4, 10_000.0).unwrap(),
            &[1.0, 0.01],
            1e-7,
        );
        for invalid_base in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                standard_rope_inverse_frequencies(4, invalid_base),
                Err(GpuNativeBootstrapError::InvalidRopeBase {
                    base_bits: invalid_base.to_bits(),
                })
            );
        }
        assert_eq!(validate_rope_parameters(layout, &[1.0, 0.01], 1.0), Ok(()));
        assert_eq!(
            validate_rope_parameters(layout, &[1.0], 1.0),
            Err(GpuNativeBootstrapError::RopeParameterWidth {
                expected: 2,
                actual: 1,
            })
        );
        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                validate_rope_parameters(layout, &[1.0, invalid], 1.0),
                Err(GpuNativeBootstrapError::InvalidRopeInverseFrequency {
                    index: 1,
                    value_bits: invalid.to_bits(),
                })
            );
        }
        assert_eq!(
            validate_rope_parameters(layout, &[1.0, 0.01], f32::NAN),
            Err(GpuNativeBootstrapError::InvalidRopeAttentionFactor {
                factor_bits: f32::NAN.to_bits(),
            })
        );

        let dense_layout =
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, 1, 2, 8).unwrap();
        let mut registry = GpuNativeDenseWeightRegistry::new(41);
        let dense = registry
            .insert(test_weight(7, "model.rope", dense_layout, ()))
            .unwrap();
        let handle = GpuNativeRopeHandle {
            dense,
            rope_dim: 4,
            attention_factor_bits: 1.0f32.to_bits(),
        };
        assert_eq!(
            validate_rope_handle_with_registry(41, &registry, &handle, 4),
            Ok(())
        );
        assert_eq!(
            validate_rope_handle_with_registry(42, &registry, &handle, 4),
            Err(GpuNativeBootstrapError::ForeignRopeHandle)
        );
        let mut stale = handle.clone();
        stale.dense.weight_id += 1;
        assert!(matches!(
            validate_rope_handle_with_registry(41, &registry, &stale, 4),
            Err(GpuNativeBootstrapError::StaleRopeHandle { .. })
        ));
    }

    #[test]
    fn kv_layout_checks_offsets_capacity_overflow_and_binding_limits() {
        let layout = GpuNativeKvLayout::try_new(3, 5, 8, &wgpu::Limits::default()).unwrap();
        assert_eq!(layout.num_layers(), 3);
        assert_eq!(layout.max_seq_len(), 5);
        assert_eq!(layout.kv_width(), 8);
        assert_eq!(layout.layer_bytes(), 160);
        assert_eq!(layout.total_bytes(), 960);
        assert_eq!(layout.element_offset(2, 4), Ok(32));
        assert_eq!(
            layout.element_offset(3, 0),
            Err(GpuNativeBootstrapError::InvalidKvLayer {
                layer: 3,
                num_layers: 3,
            })
        );
        assert_eq!(
            layout.element_offset(0, 5),
            Err(GpuNativeBootstrapError::InvalidKvPosition {
                position: 5,
                max_seq_len: 5,
            })
        );
        assert!(matches!(
            GpuNativeKvLayout::try_new(1, usize::MAX, 8, &wgpu::Limits::default()),
            Err(GpuNativeBootstrapError::KvCapacityOverflow { .. })
        ));
        let limits = wgpu::Limits {
            max_buffer_size: 128,
            max_storage_buffer_binding_size: 128,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            GpuNativeKvLayout::try_new(1, 5, 8, &limits),
            Err(GpuNativeBootstrapError::KvBufferLimit {
                required: 160,
                max_buffer_size: 128,
                max_storage_binding_size: 128,
            })
        );
        assert!(!GpuNativeKvLayout::usage()
            .intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));
        assert!(!GpuNativeKvLayout::usage().contains(wgpu::BufferUsages::COPY_SRC));
    }

    #[test]
    fn attention_scratch_and_kv_ownership_and_widths_fail_closed() {
        let geometry = GpuNativeAttentionGeometry::try_new(12, 6, 2, 4, 4).unwrap();
        let scratch = GpuNativeAttentionScratch::from_scratch(
            7,
            geometry,
            test_scratch(7, 1, geometry.q_width, ()),
            test_scratch(7, 2, geometry.kv_width, ()),
            test_scratch(7, 3, geometry.kv_width, ()),
            test_scratch(7, 4, geometry.q_width, ()),
            test_scratch(7, 5, geometry.d_model, ()),
        );
        assert_eq!(validate_attention_scratch(7, geometry, &scratch), Ok(()));
        assert_eq!(
            validate_attention_scratch(8, geometry, &scratch),
            Err(GpuNativeBootstrapError::ForeignAttentionScratch)
        );
        let wrong_q = GpuNativeAttentionScratch::from_scratch(
            7,
            geometry,
            test_scratch(7, 6, geometry.q_width - 1, ()),
            test_scratch(7, 7, geometry.kv_width, ()),
            test_scratch(7, 8, geometry.kv_width, ()),
            test_scratch(7, 9, geometry.q_width, ()),
            test_scratch(7, 10, geometry.d_model, ()),
        );
        assert_eq!(
            validate_attention_scratch(7, geometry, &wrong_q),
            Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor: GpuNativeAttentionTensor::Query,
                expected: geometry.q_width,
                actual: geometry.q_width - 1,
            })
        );
        let wrong_k = GpuNativeAttentionScratch::from_scratch(
            7,
            geometry,
            test_scratch(7, 11, geometry.q_width, ()),
            test_scratch(7, 12, geometry.kv_width - 1, ()),
            test_scratch(7, 13, geometry.kv_width, ()),
            test_scratch(7, 14, geometry.q_width, ()),
            test_scratch(7, 15, geometry.d_model, ()),
        );
        assert_eq!(
            validate_attention_scratch(7, geometry, &wrong_k),
            Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor: GpuNativeAttentionTensor::Key,
                expected: geometry.kv_width,
                actual: geometry.kv_width - 1,
            })
        );
        let wrong_v = GpuNativeAttentionScratch::from_scratch(
            7,
            geometry,
            test_scratch(7, 16, geometry.q_width, ()),
            test_scratch(7, 17, geometry.kv_width, ()),
            test_scratch(7, 18, geometry.kv_width - 1, ()),
            test_scratch(7, 19, geometry.q_width, ()),
            test_scratch(7, 20, geometry.d_model, ()),
        );
        assert_eq!(
            validate_attention_scratch(7, geometry, &wrong_v),
            Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor: GpuNativeAttentionTensor::Value,
                expected: geometry.kv_width,
                actual: geometry.kv_width - 1,
            })
        );
        let wrong_context = GpuNativeAttentionScratch::from_scratch(
            7,
            geometry,
            test_scratch(7, 21, geometry.q_width, ()),
            test_scratch(7, 22, geometry.kv_width, ()),
            test_scratch(7, 23, geometry.kv_width, ()),
            test_scratch(7, 24, geometry.q_width - 1, ()),
            test_scratch(7, 25, geometry.d_model, ()),
        );
        assert_eq!(
            validate_attention_scratch(7, geometry, &wrong_context),
            Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor: GpuNativeAttentionTensor::Context,
                expected: geometry.q_width,
                actual: geometry.q_width - 1,
            })
        );
        let wrong_projected = GpuNativeAttentionScratch::from_scratch(
            7,
            geometry,
            test_scratch(7, 26, geometry.q_width, ()),
            test_scratch(7, 27, geometry.kv_width, ()),
            test_scratch(7, 28, geometry.kv_width, ()),
            test_scratch(7, 29, geometry.q_width, ()),
            test_scratch(7, 30, geometry.d_model - 1, ()),
        );
        assert_eq!(
            validate_attention_scratch(7, geometry, &wrong_projected),
            Err(GpuNativeBootstrapError::AttentionScratchWidth {
                tensor: GpuNativeAttentionTensor::Output,
                expected: geometry.d_model,
                actual: geometry.d_model - 1,
            })
        );
        let aliased = GpuNativeAttentionScratch::from_scratch(
            7,
            geometry,
            test_scratch(7, 31, geometry.q_width, ()),
            test_scratch(7, 32, geometry.kv_width, ()),
            test_scratch(7, 33, geometry.kv_width, ()),
            test_scratch(7, 31, geometry.q_width, ()),
            test_scratch(7, 34, geometry.d_model, ()),
        );
        assert_eq!(
            validate_attention_scratch(7, geometry, &aliased),
            Err(GpuNativeBootstrapError::AliasedInputOutput)
        );

        let kv_layout = GpuNativeKvLayout::try_new(2, 4, 8, &wgpu::Limits::default()).unwrap();
        let kv = GpuNativeKvState::from_layers(
            7,
            1,
            kv_layout,
            vec![
                GpuNativeKvLayer { key: (), value: () },
                GpuNativeKvLayer { key: (), value: () },
            ],
        );
        assert_eq!(validate_kv_state(7, 8, &kv, 1, 3), Ok(()));
        assert_eq!(
            validate_kv_state(8, 8, &kv, 1, 3),
            Err(GpuNativeBootstrapError::ForeignKvState)
        );
        assert_eq!(
            validate_kv_state(7, 7, &kv, 1, 3),
            Err(GpuNativeBootstrapError::KvWidth {
                expected: 7,
                actual: 8,
            })
        );
        assert!(matches!(
            validate_kv_state(7, 8, &kv, 2, 3),
            Err(GpuNativeBootstrapError::InvalidKvLayer { .. })
        ));
        assert!(matches!(
            validate_kv_state(7, 8, &kv, 1, 4),
            Err(GpuNativeBootstrapError::InvalidKvPosition { .. })
        ));
    }

    #[test]
    fn hardware_independent_attention_prepare_chain_preserves_two_kv_positions() {
        let geometry = GpuNativeAttentionGeometry::try_new(6, 4, 2, 4, 4).unwrap();
        let q_projection = DenseWeight::from_f32(
            (0..geometry.q_width * geometry.d_model)
                .map(|index| ((index * 11 % 37) as f32 - 18.0) / 19.0)
                .collect(),
            geometry.q_width,
            geometry.d_model,
        );
        let k_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * geometry.d_model)
                .map(|index| ((index * 13 % 41) as f32 - 20.0) / 17.0)
                .collect(),
            geometry.kv_width,
            geometry.d_model,
        );
        let v_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * geometry.d_model)
                .map(|index| ((index * 17 % 43) as f32 - 21.0) / 23.0)
                .collect(),
            geometry.kv_width,
            geometry.d_model,
        );
        let q_gain = [0.8, 1.1, 0.9, 1.2];
        let k_gain = [1.3, 0.7, 1.0, 0.85];
        let epsilon = 1e-6;
        let inverse_frequencies = [1.0, 0.01];
        let kv_layout =
            GpuNativeKvLayout::try_new(1, 4, geometry.kv_width, &wgpu::Limits::default()).unwrap();
        let mut key_cache = vec![f32::NAN; kv_layout.layer_elements];
        let mut value_cache = vec![f32::NAN; kv_layout.layer_elements];
        let mut first_key = Vec::new();
        let mut first_value = Vec::new();

        for (position, hidden) in [
            (1usize, vec![0.5, -1.0, 1.5, -2.0, 2.5, -3.0]),
            (3usize, vec![-0.25, 0.75, -1.25, 1.75, -2.25, 2.75]),
        ] {
            let (q, k, v) = attention_prepare_mirror(
                &hidden,
                geometry,
                &q_projection,
                &k_projection,
                &v_projection,
                Some((&q_gain, epsilon)),
                Some((&k_gain, epsilon)),
                &inverse_frequencies,
                1.0,
                position,
            );
            assert_eq!(q.len(), geometry.q_width);
            assert_eq!(k.len(), geometry.kv_width);
            assert_eq!(v.len(), geometry.kv_width);
            assert!(q.iter().chain(&k).chain(&v).all(|value| value.is_finite()));
            let offset = kv_layout.element_offset(0, position).unwrap();
            key_cache[offset..offset + geometry.kv_width].copy_from_slice(&k);
            value_cache[offset..offset + geometry.kv_width].copy_from_slice(&v);
            if position == 1 {
                first_key = k;
                first_value = v;
            }
        }

        let first_offset = kv_layout.element_offset(0, 1).unwrap();
        assert_eq!(
            &key_cache[first_offset..first_offset + geometry.kv_width],
            first_key
        );
        assert_eq!(
            &value_cache[first_offset..first_offset + geometry.kv_width],
            first_value
        );
        assert!(key_cache[..geometry.kv_width]
            .iter()
            .all(|value| value.is_nan()));
        let unused_offset = kv_layout.element_offset(0, 2).unwrap();
        assert!(key_cache[unused_offset..unused_offset + geometry.kv_width]
            .iter()
            .all(|value| value.is_nan()));
    }

    #[test]
    fn causal_gqa_attention_o_projection_and_residual_match_cpu_mirror() {
        let geometry = GpuNativeAttentionGeometry::try_new(6, 4, 2, 2, 2).unwrap();
        let q = [0.8, -0.3, 1.1, 0.4, -0.7, 1.3, 0.2, -1.0];
        let key_cache = [
            0.2, -0.5, 1.1, 0.3, // position 0: KV heads 0, 1
            0.9, 0.1, -0.4, 0.8, // position 1
            -0.6, 1.2, 0.5, -0.9, // position 2
            10_000.0, -9_000.0, 8_000.0, -7_000.0, // future poison
        ];
        let value_cache = [
            0.5, -1.0, 1.5, 0.25, // position 0
            -0.75, 1.25, 0.4, -1.4, // position 1
            1.8, 0.6, -0.9, 1.1, // position 2
            50_000.0, -40_000.0, 30_000.0, -20_000.0, // future poison
        ];
        let o_projection = DenseWeight::from_f32(
            (0..geometry.d_model * geometry.q_width)
                .map(|index| ((index * 17 % 31) as f32 - 15.0) / 13.0)
                .collect(),
            geometry.d_model,
            geometry.q_width,
        );
        let residual = [0.25, -0.5, 0.75, -1.0, 1.25, -1.5];
        let residual_before = residual;
        let (context, projected, hidden) = attention_complete_mirror(
            &q,
            &key_cache,
            &value_cache,
            geometry,
            3,
            &o_projection,
            &residual,
        );
        assert_eq!(context.len(), geometry.q_width);
        assert_eq!(projected.len(), geometry.d_model);
        assert_eq!(hidden.len(), geometry.d_model);
        assert!(context
            .iter()
            .chain(&projected)
            .chain(&hidden)
            .all(|value| value.is_finite()));
        assert_eq!(residual, residual_before);
        for ((actual, saved), contribution) in hidden.iter().zip(residual).zip(&projected) {
            assert!((actual - (saved + contribution)).abs() < 1e-6);
        }

        let mut changed_future_keys = key_cache;
        let mut changed_future_values = value_cache;
        changed_future_keys[3 * geometry.kv_width..].fill(-123_456.0);
        changed_future_values[3 * geometry.kv_width..].fill(654_321.0);
        let future_poisoned = causal_attention_mirror(
            &q,
            &changed_future_keys,
            &changed_future_values,
            geometry,
            3,
        );
        assert_close(&context, &future_poisoned, 0.0);

        let uniform_head_zero = [
            (value_cache[0] + value_cache[4] + value_cache[8]) / 3.0,
            (value_cache[1] + value_cache[5] + value_cache[9]) / 3.0,
        ];
        assert!(context[..geometry.head_dim]
            .iter()
            .zip(uniform_head_zero)
            .any(|(actual, uniform)| (actual - uniform).abs() > 1e-3));
        // Heads 0 and 1 share KV head 0; heads 2 and 3 share KV head 1.
        assert_ne!(
            &context[..2 * geometry.head_dim],
            &context[2 * geometry.head_dim..]
        );
    }

    #[test]
    fn attention_prepare_allows_absent_qk_norm_as_a_clean_noop() {
        let geometry = GpuNativeAttentionGeometry::try_new(4, 2, 1, 4, 4).unwrap();
        let q_projection = DenseWeight::from_f32(
            (0..geometry.q_width * geometry.d_model)
                .map(|index| (index as f32 - 8.0) / 7.0)
                .collect(),
            geometry.q_width,
            geometry.d_model,
        );
        let k_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * geometry.d_model)
                .map(|index| (index as f32 - 5.0) / 9.0)
                .collect(),
            geometry.kv_width,
            geometry.d_model,
        );
        let v_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * geometry.d_model)
                .map(|index| (index as f32 - 3.0) / 11.0)
                .collect(),
            geometry.kv_width,
            geometry.d_model,
        );
        let hidden = [0.25, -0.5, 0.75, -1.0];
        let inverse_frequencies = [1.0, 0.01];
        let (q, k, v) = attention_prepare_mirror(
            &hidden,
            geometry,
            &q_projection,
            &k_projection,
            &v_projection,
            None,
            None,
            &inverse_frequencies,
            1.0,
            2,
        );
        assert_close(
            &q,
            &rope_mirror(
                &q_projection.matvec(&hidden),
                geometry.num_heads,
                geometry.head_dim,
                geometry.rope_dim,
                2,
                &inverse_frequencies,
                1.0,
            ),
            1e-6,
        );
        assert_close(
            &k,
            &rope_mirror(
                &k_projection.matvec(&hidden),
                geometry.num_kv_heads,
                geometry.head_dim,
                geometry.rope_dim,
                2,
                &inverse_frequencies,
                1.0,
            ),
            1e-6,
        );
        assert_eq!(v, v_projection.matvec(&hidden));
    }

    #[test]
    fn typed_rms_norm_handles_reuse_persistent_f32_registry_and_fail_closed() {
        let layout =
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, 1, 7, 28).unwrap();
        let mut registry = GpuNativeDenseWeightRegistry::new(41);
        let dense = registry
            .insert(test_weight(7, "layer.0.rms_attn", layout, ()))
            .unwrap();
        let handle = GpuNativeRmsNormHandle::from_dense(dense.clone());
        assert_eq!(handle.width(), 7);
        let first = registry.resolve_rms_norm(&handle).unwrap();
        let second = registry.resolve_rms_norm(&handle).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(registry.weights.len(), 1);

        let foreign_registry = GpuNativeDenseWeightRegistry::<()>::new(42);
        assert!(matches!(
            foreign_registry.resolve_rms_norm(&handle),
            Err(GpuNativeBootstrapError::ForeignRmsNormHandle)
        ));

        let mut stale = handle.clone();
        stale.dense.weight_id += 1;
        assert!(matches!(
            registry.resolve_rms_norm(&stale),
            Err(GpuNativeBootstrapError::StaleRmsNormHandle { .. })
        ));

        let mut wrong_width = handle.clone();
        wrong_width.width = 6;
        assert!(matches!(
            registry.resolve_rms_norm(&wrong_width),
            Err(GpuNativeBootstrapError::StaleRmsNormHandle { .. })
        ));
        assert_eq!(
            registry.handle_for(dense.key(), GpuNativeDenseWeightKind::F32, 1, 6,),
            Err(GpuNativeBootstrapError::DenseWeightShapeMismatch {
                key: "layer.0.rms_attn".to_string(),
                expected_rows: 1,
                expected_cols: 6,
                actual_rows: 1,
                actual_cols: 7,
            })
        );

        let q8_layout = GpuNativeDenseWeightLayout::try_new(
            GpuNativeDenseWeightKind::Q8_0,
            1,
            32,
            Q8_0_BLOCK_BYTES,
        )
        .unwrap();
        let q8_dense = registry
            .insert(test_weight(8, "invalid.q8_norm", q8_layout, ()))
            .unwrap();
        let q8_handle = GpuNativeRmsNormHandle::from_dense(q8_dense);
        assert!(matches!(
            registry.resolve_rms_norm(&q8_handle),
            Err(GpuNativeBootstrapError::StaleRmsNormHandle { .. })
        ));
    }

    #[test]
    fn new_state_operations_reject_foreign_request_ownership() {
        assert_eq!(validate_token_state_owner(7, 7), Ok(()));
        assert_eq!(validate_scratch_owner(7, 7), Ok(()));
        assert_eq!(
            validate_token_state_owner(7, 8),
            Err(GpuNativeBootstrapError::ForeignTokenState)
        );
        assert_eq!(
            validate_scratch_owner(7, 8),
            Err(GpuNativeBootstrapError::ForeignScratch)
        );
        assert_eq!(validate_residual_contribution_width(2_048, 2_048), Ok(()));
        assert_eq!(
            validate_residual_contribution_width(2_048, 1_024),
            Err(GpuNativeBootstrapError::ResidualContributionWidth {
                expected: 2_048,
                actual: 1_024,
            })
        );
    }

    #[test]
    fn hardware_independent_residual_chain_preserves_every_state_transition() {
        const WIDTH: usize = 7;
        let embedding = DenseWeight::from_f32(
            (0..3 * WIDTH)
                .map(|index| ((index * 13 % 29) as f32 - 14.0) / 6.0)
                .collect(),
            3,
            WIDTH,
        );
        let first_dense = DenseWeight::from_f32(
            (0..WIDTH * WIDTH)
                .map(|index| ((index * 17 % 31) as f32 - 15.0) / 23.0)
                .collect(),
            WIDTH,
            WIDTH,
        );
        let second_dense = DenseWeight::from_f32(
            (0..WIDTH * WIDTH)
                .map(|index| ((index * 19 % 37) as f32 - 18.0) / 29.0)
                .collect(),
            WIDTH,
            WIDTH,
        );
        let first_gain = (0..WIDTH)
            .map(|index| 0.7 + index as f32 / 20.0)
            .collect::<Vec<_>>();
        let second_gain = (0..WIDTH)
            .map(|index| 0.9 - index as f32 / 30.0)
            .collect::<Vec<_>>();
        let final_gain = (0..WIDTH)
            .map(|index| 1.1 + index as f32 / 40.0)
            .collect::<Vec<_>>();
        let epsilon = 1e-6;

        let mut hidden = Vec::new();
        embedding.row_dequant_into(1, &mut hidden);

        let original_hidden = hidden.clone();
        let mut residual = vec![f32::NAN; WIDTH];
        rms_norm_capture_mirror(&mut hidden, &mut residual, &first_gain, epsilon);
        assert_eq!(residual, original_hidden);
        assert_eq!(
            hidden,
            rms_norm_mirror(&original_hidden, &first_gain, epsilon, 1, WIDTH)
        );
        let first_contribution = first_dense.matvec(&hidden);
        residual_complete_mirror(&mut hidden, &residual, &first_contribution);
        assert_eq!(
            hidden,
            original_hidden
                .iter()
                .zip(&first_contribution)
                .map(|(&residual, &contribution)| residual + contribution)
                .collect::<Vec<_>>()
        );

        let before_second_norm = hidden.clone();
        rms_norm_capture_mirror(&mut hidden, &mut residual, &second_gain, epsilon);
        assert_eq!(residual, before_second_norm);
        let second_contribution = second_dense.matvec(&hidden);
        residual_complete_mirror(&mut hidden, &residual, &second_contribution);
        let before_final_norm = hidden.clone();
        let residual_before_final_norm = residual.clone();
        let expected_final_hidden =
            rms_norm_mirror(&before_final_norm, &final_gain, epsilon, 1, WIDTH);
        hidden = rms_norm_mirror(&hidden, &final_gain, epsilon, 1, WIDTH);

        assert_eq!(residual, residual_before_final_norm);
        assert_eq!(hidden, expected_final_hidden);
        assert_ne!(hidden, before_final_norm);
        assert_eq!(before_final_norm.len(), WIDTH);
        assert_eq!(hidden.len(), WIDTH);
        assert!(hidden.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn registry_missing_duplicate_kind_shape_and_context_checks_fail_closed() {
        let f32_layout =
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, 3, 5, 60).unwrap();
        let key = GpuNativeDenseWeightKey::try_new("embedding").unwrap();
        let mut registry = GpuNativeDenseWeightRegistry::new(41);
        assert_eq!(
            registry.handle_for(&key, GpuNativeDenseWeightKind::F32, 3, 5),
            Err(GpuNativeBootstrapError::MissingDenseWeight {
                key: "embedding".to_string(),
            })
        );

        let handle = registry
            .insert(test_weight(7, "embedding", f32_layout, ()))
            .unwrap();
        assert_eq!(handle.layout().kind(), GpuNativeDenseWeightKind::F32);
        assert_eq!(
            registry.insert(test_weight(8, "embedding", f32_layout, ())),
            Err(GpuNativeBootstrapError::DuplicateDenseWeight {
                key: "embedding".to_string(),
            })
        );
        assert_eq!(
            registry.handle_for(&key, GpuNativeDenseWeightKind::Q8_0, 3, 5),
            Err(GpuNativeBootstrapError::DenseWeightKindMismatch {
                key: "embedding".to_string(),
                expected: GpuNativeDenseWeightKind::Q8_0,
                actual: GpuNativeDenseWeightKind::F32,
            })
        );
        assert_eq!(
            registry.handle_for(&key, GpuNativeDenseWeightKind::F32, 4, 5),
            Err(GpuNativeBootstrapError::DenseWeightShapeMismatch {
                key: "embedding".to_string(),
                expected_rows: 4,
                expected_cols: 5,
                actual_rows: 3,
                actual_cols: 5,
            })
        );

        let other_registry = GpuNativeDenseWeightRegistry::<()>::new(42);
        assert!(matches!(
            other_registry.resolve(&handle),
            Err(GpuNativeBootstrapError::ForeignDenseWeightHandle)
        ));
        let mut stale = handle;
        stale.weight_id += 1;
        assert!(matches!(
            registry.resolve(&stale),
            Err(GpuNativeBootstrapError::StaleDenseWeightHandle { .. })
        ));
    }

    #[test]
    fn q8_shader_mirror_matches_repository_dequant_and_signed_scale_semantics() {
        let mut bytes = vec![0u8; 2 * Q8_0_BLOCK_BYTES];
        for (block, scale) in [-0.5f32, 0.25f32].into_iter().enumerate() {
            let offset = block * Q8_0_BLOCK_BYTES;
            bytes[offset..offset + 2].copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
            for i in 0..Q8_0_BLOCK_ELEMS {
                bytes[offset + 2 + i] = (i as i8 - 16) as u8;
            }
            let mut expected = [0.0; Q8_0_BLOCK_ELEMS];
            dequantize_q8_0_block(&bytes[offset..offset + Q8_0_BLOCK_BYTES], &mut expected);
            for (i, &expected) in expected.iter().enumerate() {
                assert_eq!(
                    read_q8_mirror(&bytes, block * Q8_0_BLOCK_ELEMS + i),
                    expected
                );
            }
        }
    }

    #[test]
    fn f32_and_q8_host_mirrors_match_dense_weight_cpu_matvec() {
        let rows = 3;
        let cols = 35;
        let values = (0..rows * cols)
            .map(|i| ((i * 17 % 43) as f32 - 21.0) / 7.0)
            .collect::<Vec<_>>();
        let input = (0..cols)
            .map(|i| ((i * 11 % 19) as f32 - 9.0) / 5.0)
            .collect::<Vec<_>>();

        let f32_weight = DenseWeight::from_f32(values.clone(), rows, cols);
        let mut f32_mirror = vec![0.0; rows];
        for row in 0..rows {
            for col in 0..cols {
                f32_mirror[row] += values[row * cols + col] * input[col];
            }
        }
        assert_close(&f32_mirror, &f32_weight.matvec(&input), 1e-5);

        let bytes = q8_bytes(&values);
        let q8_weight = DenseWeight::from_q8_0_bytes(bytes.clone(), rows, cols).unwrap();
        let q8_mirror = q8_gemv_mirror(&bytes, rows, cols, &input);
        assert_close(&q8_mirror, &q8_weight.matvec(&input), 1e-5);
    }

    #[test]
    fn embedding_bounds_and_row_mirror_cover_first_middle_last_and_invalid() {
        let rows = 5;
        let cols = 7;
        let values = (0..rows * cols)
            .map(|i| i as f32 - 10.0)
            .collect::<Vec<_>>();
        let bytes = q8_bytes(&values);
        let weight = DenseWeight::from_q8_0_bytes(bytes.clone(), rows, cols).unwrap();
        let layout = GpuNativeDenseWeightLayout::from_weight(&weight).unwrap();
        for token in [0u32, 2, 4] {
            layout.validate_embedding_token(token).unwrap();
            let mut expected = Vec::new();
            weight.row_dequant_into(token as usize, &mut expected);
            let actual = (0..cols)
                .map(|col| read_q8_mirror(&bytes, token as usize * cols + col))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
        assert_eq!(
            layout.validate_embedding_token(5),
            Err(GpuNativeBootstrapError::InvalidEmbeddingToken {
                token_id: 5,
                vocab_size: 5,
            })
        );
    }

    #[test]
    fn registration_and_dispatch_counters_keep_uploads_distinct() {
        let counters = GpuNativeExecutionCounters::default();
        counters.record_dense_weight_registration(3, 108);
        let after_registration = counters.snapshot();
        assert_eq!(after_registration.dense_weights_registered, 1);
        assert_eq!(after_registration.dense_weight_chunks, 3);
        assert_eq!(after_registration.dense_weight_uploads, 3);
        assert_eq!(after_registration.dense_weight_upload_bytes, 108);
        assert_eq!(after_registration.dense_weight_resident_bytes, 108);
        assert_eq!(after_registration.dense_gemv_dispatches, 0);
        assert_eq!(after_registration.dense_gemv_chunk_dispatches, 0);

        counters.record_dense_gemv_dispatch(3);
        counters.record_dense_gemv_dispatch(2);
        counters.record_embedding_dispatch();
        counters.record_rms_norm_state_dispatch(1);
        counters.record_rms_norm_state_dispatch(1);
        counters.record_rms_norm_scratch_dispatch(4);
        counters.record_residual_add_dispatch();
        counters.record_residual_add_dispatch();
        counters.record_rope_registration(8);
        counters.record_attention_prepare_dispatch();
        counters.record_rope_dispatch(6);
        counters.record_rope_dispatch(2);
        counters.record_kv_append();
        counters.record_causal_attention_dispatch();
        counters.record_attention_complete_dispatch();
        counters.record_router_dispatch();
        counters.record_expert_arena_registration(8, 13_824);
        counters.record_expert_dispatches(8);
        let after_dispatch = counters.snapshot();
        assert_eq!(after_dispatch.dense_weight_uploads, 3);
        assert_eq!(after_dispatch.dense_weight_upload_bytes, 108);
        assert_eq!(after_dispatch.dense_gemv_dispatches, 2);
        assert_eq!(after_dispatch.dense_gemv_chunk_dispatches, 5);
        assert_eq!(after_dispatch.embedding_dispatches, 1);
        assert_eq!(after_dispatch.rms_norm_dispatches, 3);
        assert_eq!(after_dispatch.rms_norm_groups, 6);
        assert_eq!(after_dispatch.rms_norm_state_dispatches, 2);
        assert_eq!(after_dispatch.rms_norm_scratch_dispatches, 1);
        assert_eq!(after_dispatch.residual_add_dispatches, 2);
        assert_eq!(after_dispatch.rope_parameters_registered, 1);
        assert_eq!(after_dispatch.rope_parameter_uploads, 1);
        assert_eq!(after_dispatch.rope_parameter_upload_bytes, 8);
        assert_eq!(after_dispatch.attention_prepare_dispatches, 1);
        assert_eq!(after_dispatch.q_projection_dispatches, 1);
        assert_eq!(after_dispatch.k_projection_dispatches, 1);
        assert_eq!(after_dispatch.v_projection_dispatches, 1);
        assert_eq!(after_dispatch.rope_dispatches, 2);
        assert_eq!(after_dispatch.rope_groups, 8);
        assert_eq!(after_dispatch.kv_appends, 1);
        assert_eq!(after_dispatch.causal_attention_dispatches, 1);
        assert_eq!(after_dispatch.o_projection_dispatches, 1);
        assert_eq!(after_dispatch.attention_complete_dispatches, 1);
        assert_eq!(after_dispatch.router_logit_dispatches, 1);
        assert_eq!(after_dispatch.router_topk_dispatches, 1);
        assert_eq!(after_dispatch.expert_slots_registered, 8);
        assert_eq!(after_dispatch.expert_weight_upload_bytes, 13_824);
        assert_eq!(after_dispatch.expert_route_resolve_dispatches, 1);
        assert_eq!(after_dispatch.q4_expert_gate_up_dispatches, 8);
        assert_eq!(after_dispatch.q4_expert_down_dispatches, 8);
        assert_eq!(after_dispatch.expert_combine_dispatches, 1);
        assert_eq!(after_dispatch.cpu_router_calls, 0);
        assert_eq!(after_dispatch.cpu_expert_combines, 0);
        assert_eq!(after_dispatch.expert_slot_misses, 0);
        assert_eq!(after_dispatch.numerical_failures, 0);
    }

    #[test]
    fn attention_numerical_failure_status_bit_is_stable_and_latched() {
        assert_eq!(GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE, 1);
        assert!(GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE.is_power_of_two());
        assert!(GPU_NATIVE_ATTENTION_SHADER
            .contains("atomicOr(&STATUS.bits, GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE)"));
        assert!(!GPU_NATIVE_ATTENTION_SHADER.contains("atomicStore(&STATUS"));
    }

    #[test]
    fn router_numerical_failure_status_bit_is_distinct_stable_and_latched() {
        assert_eq!(GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE, 1);
        assert_eq!(GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE, 2);
        assert!(GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE.is_power_of_two());
        assert!(GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE.is_power_of_two());
        assert_eq!(
            GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE
                & GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE,
            0
        );
        assert!(GPU_NATIVE_ROUTER_SHADER
            .contains("atomicOr(&STATUS.bits, GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE)"));
        assert!(!GPU_NATIVE_ROUTER_SHADER.contains("atomicStore(&STATUS"));
        assert!(GPU_NATIVE_ROUTER_SHADER.contains("SELECTED_IDS[slot] = 0u"));
        assert!(GPU_NATIVE_ROUTER_SHADER.contains("SELECTED_WEIGHTS[slot] = 0.0"));
    }

    #[test]
    fn expert_status_bits_are_stable_non_overlapping_and_only_latched() {
        assert_eq!(GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE, 1);
        assert_eq!(GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE, 2);
        assert_eq!(GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS, 4);
        assert_eq!(GPU_NATIVE_STATUS_EXPERT_NUMERICAL_FAILURE, 8);
        assert_eq!(GPU_NATIVE_STATUS_LM_HEAD_NUMERICAL_FAILURE, 16);
        assert_eq!(GPU_NATIVE_STATUS_FATAL_MASK, 27);
        assert_eq!(GPU_NATIVE_STATUS_RETRYABLE_MASK, 4);
        assert_eq!(
            GPU_NATIVE_STATUS_FATAL_MASK & GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS,
            0
        );
        assert_eq!(
            GPU_NATIVE_STATUS_RETRYABLE_MASK & GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS,
            GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS
        );
        let bits = [
            GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE,
            GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE,
            GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS,
            GPU_NATIVE_STATUS_EXPERT_NUMERICAL_FAILURE,
            GPU_NATIVE_STATUS_LM_HEAD_NUMERICAL_FAILURE,
        ];
        assert!(bits.iter().all(|bit| bit.is_power_of_two()));
        for (index, bit) in bits.iter().enumerate() {
            assert!(bits[index + 1..].iter().all(|other| bit & other == 0));
        }
        assert!(GPU_NATIVE_EXPERT_CONTROL_SHADER
            .contains("atomicOr(&RESOLVE_STATUS.bits, EXPERT_RESIDENCY_MISS)"));
        assert!(GPU_NATIVE_Q4_EXPERT_SHADER
            .contains("atomicOr(&STATUS.bits, EXPERT_NUMERICAL_FAILURE)"));
        assert!(
            GPU_NATIVE_Q4_EXPERT_SHADER.contains("atomicOr(&STATUS.bits, EXPERT_RESIDENCY_MISS)")
        );
        assert!(GPU_NATIVE_Q4_EXPERT_SHADER.contains("current_slot_epoch != route.slot_epoch"));
        assert!(GPU_NATIVE_EXPERT_CONTROL_SHADER.contains("FATAL_STATUS_MASK"));
        assert!(GPU_NATIVE_EXPERT_CONTROL_SHADER.contains("RETRYABLE_STATUS_MASK"));
        assert!(GPU_NATIVE_EXPERT_CONTROL_SHADER
            .contains("atomicOr(&COMBINE_STATUS.bits, EXPERT_NUMERICAL_FAILURE)"));
        assert!(!GPU_NATIVE_EXPERT_CONTROL_SHADER.contains("atomicStore(&RESOLVE_STATUS"));
        assert!(!GPU_NATIVE_EXPERT_CONTROL_SHADER.contains("atomicStore(&COMBINE_STATUS"));
        assert!(!GPU_NATIVE_Q4_EXPERT_SHADER.contains("atomicStore(&STATUS"));
    }

    #[test]
    fn retryable_status_clear_preserves_fatal_and_unknown_bits() {
        assert_eq!(
            status_after_retryable_clear(GPU_NATIVE_STATUS_RETRYABLE_MASK),
            0
        );
        assert_eq!(
            status_after_retryable_clear(GPU_NATIVE_STATUS_FATAL_MASK),
            GPU_NATIVE_STATUS_FATAL_MASK
        );
        assert_eq!(
            status_after_retryable_clear(
                GPU_NATIVE_STATUS_FATAL_MASK | GPU_NATIVE_STATUS_RETRYABLE_MASK
            ),
            GPU_NATIVE_STATUS_FATAL_MASK
        );
        let unknown = 1 << 17;
        assert_eq!(status_after_retryable_clear(unknown), unknown);
        assert!(GPU_NATIVE_STATUS_CONTROL_SHADER
            .contains("atomicAnd(&STATUS.bits, ~EXPERT_RESIDENCY_RETRYABLE)"));
        assert!(!GPU_NATIVE_STATUS_CONTROL_SHADER.contains("atomicStore"));
    }

    #[test]
    fn gpu_native_shaders_parse_and_validate_without_hardware() {
        for (source, entry_points) in [
            (
                GPU_NATIVE_DENSE_GEMV_SHADER,
                &["f32_gemv_main", "q8_0_gemv_main"][..],
            ),
            (
                GPU_NATIVE_EMBEDDING_SHADER,
                &["f32_embedding_main", "q8_0_embedding_main"][..],
            ),
            (
                GPU_NATIVE_RMSNORM_SHADER,
                &[
                    "rms_norm_capture_main",
                    "rms_norm_in_place_main",
                    "residual_add_main",
                ][..],
            ),
            (GPU_NATIVE_ROPE_SHADER, &["rope_main"][..]),
            (GPU_NATIVE_KV_APPEND_SHADER, &["kv_append_main"][..]),
            (GPU_NATIVE_ATTENTION_SHADER, &["causal_attention_main"][..]),
            (GPU_NATIVE_ROUTER_SHADER, &["router_topk_main"][..]),
            (
                GPU_NATIVE_Q4_EXPERT_SHADER,
                &["q4_expert_gate_up_main", "q4_expert_down_main"][..],
            ),
            (
                GPU_NATIVE_EXPERT_CONTROL_SHADER,
                &[
                    "expert_route_resolve_main",
                    "expert_validate_main",
                    "expert_combine_main",
                    "expert_contain_main",
                ][..],
            ),
            (
                GPU_NATIVE_STATUS_CONTROL_SHADER,
                &["clear_retryable_expert_residency_main"][..],
            ),
            (GPU_NATIVE_GREEDY_ARGMAX_SHADER, &["greedy_argmax_main"][..]),
            (GPU_NATIVE_TEST_COMPARE_SHADER, &["compare_main"][..]),
        ] {
            let module = naga::front::wgsl::parse_str(source).expect("GPU-native WGSL must parse");
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .expect("GPU-native WGSL must validate");
            for entry_point in entry_points {
                assert!(module
                    .entry_points
                    .iter()
                    .any(|entry| entry.name == *entry_point));
            }
        }
        assert!(!GPU_NATIVE_ATTENTION_SHADER.contains("layer_offset"));
        assert!(!GPU_NATIVE_ATTENTION_SHADER.contains("MAX_SEQ_LEN"));
        assert!(GPU_NATIVE_ATTENTION_SHADER.contains("KEY_CACHE"));
        assert!(GPU_NATIVE_ATTENTION_SHADER.contains("VALUE_CACHE"));
        assert!(GPU_NATIVE_ATTENTION_SHADER.contains("@group(0) @binding(4)"));
        assert!(GPU_NATIVE_ATTENTION_SHADER.contains("fn is_finite"));
        assert!(GPU_NATIVE_ATTENTION_SHADER.contains("numerically_valid"));
        assert!(GPU_NATIVE_ATTENTION_SHADER.contains("running_denominator"));
        assert!(GPU_NATIVE_ROUTER_SHADER.contains("@workgroup_size(64, 1, 1)"));
        assert!(GPU_NATIVE_ROUTER_SHADER.contains("MAX_EXPERTS: u32 = 128u"));
        assert!(GPU_NATIVE_ROUTER_SHADER.contains("MAX_TOP_K: u32 = 8u"));
        assert!(GPU_NATIVE_Q4_EXPERT_SHADER.contains("W0"));
        assert!(GPU_NATIVE_Q4_EXPERT_SHADER.contains("W3"));
        assert!(GPU_NATIVE_Q4_EXPERT_SHADER.contains("BLOCK_BYTES: u32 = 18u"));
        assert!(GPU_NATIVE_EXPERT_CONTROL_SHADER.contains("UNMAPPED: u32 = 0xffffffffu"));
    }

    const GPU_NATIVE_TEST_COMPARE_SHADER: &str = r#"
struct PushConstants {
    elements: u32,
    tolerance_bits: u32,
    actual_offset: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> ACTUAL: array<f32>;
@group(0) @binding(1) var<storage, read> EXPECTED: array<f32>;
@group(0) @binding(2) var<storage, read_write> STATUS: array<atomic<u32>>;

@compute @workgroup_size(64, 1, 1)
fn compare_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= pc.elements) {
        return;
    }
    let difference = abs(ACTUAL[pc.actual_offset + gid.x] - EXPECTED[gid.x]);
    if (!(difference <= bitcast<f32>(pc.tolerance_bits))) {
        atomicStore(&STATUS[0], 1u);
    }
}
"#;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GpuNativeTestComparePushConstants {
        elements: u32,
        tolerance_bits: u32,
        actual_offset: u32,
    }

    fn create_test_compare_pipeline(
        device: &wgpu::Device,
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_native_test_compare_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_native_test_compare_pipeline_layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..12,
            }],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_native_test_compare_shader"),
            source: wgpu::ShaderSource::Wgsl(GPU_NATIVE_TEST_COMPARE_SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu_native_test_compare_pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "compare_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        (layout, pipeline)
    }

    fn create_test_expected_buffer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        values: &[f32],
    ) -> wgpu::Buffer {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (values.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(values));
        buffer
    }

    fn encode_test_compare(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        layout: &wgpu::BindGroupLayout,
        pipeline: &wgpu::ComputePipeline,
        actual: &wgpu::Buffer,
        expected: &wgpu::Buffer,
        status: &wgpu::Buffer,
        elements: usize,
        tolerance: f32,
    ) {
        encode_test_compare_at(
            device, encoder, layout, pipeline, actual, expected, status, elements, tolerance, 0,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_test_compare_at(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        layout: &wgpu::BindGroupLayout,
        pipeline: &wgpu::ComputePipeline,
        actual: &wgpu::Buffer,
        expected: &wgpu::Buffer,
        status: &wgpu::Buffer,
        elements: usize,
        tolerance: f32,
        actual_offset: usize,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_native_test_compare_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: actual.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: expected.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: status.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_native_test_compare_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(
            0,
            bytemuck::bytes_of(&GpuNativeTestComparePushConstants {
                elements: elements as u32,
                tolerance_bits: tolerance.to_bits(),
                actual_offset: actual_offset as u32,
            }),
        );
        pass.dispatch_workgroups((elements as u32).div_ceil(GPU_NATIVE_WORKGROUP_SIZE), 1, 1);
    }

    /// Requires an actual hardware WGPU adapter. This intentionally uses the
    /// production execution-context resolver and maps only its four-byte
    /// aggregate validation status, never GPU-native hidden or scratch data.
    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_dense_gemv_embedding_persistence() {
        use super::super::{
            resolve_execution_context, ComputeOffload, GpuBackendGeometry, RoutedExpertGpuSpec,
        };
        use crate::inference::WeightDtype;

        const COLS: usize = 35;
        let expert_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(
            1024 * 1024,
            0.5,
            16,
        ));
        let execution = resolve_execution_context(
            ComputeOffload::Gpu,
            false,
            GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: 8,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 8,
                v_head_dim: 8,
                q4_truncation_tolerance: 0,
            },
            RoutedExpertGpuSpec {
                dtype: WeightDtype::F32,
                d_model: 32,
                d_ff: 64,
            },
            expert_cache,
        )
        .expect("L4 must construct the authoritative production GPU backend");
        let executor = execution
            .create_gpu_native_executor_context(COLS)
            .expect("GPU-native executor must retain the authoritative backend");
        let gpu = executor.authoritative_gpu().unwrap();

        let gemv_input_values = (0..COLS)
            .map(|i| ((i * 11 % 19) as f32 - 9.0) / 5.0)
            .collect::<Vec<_>>();
        let f32_gemv_values = (0..3 * COLS)
            .map(|i| ((i * 17 % 43) as f32 - 21.0) / 7.0)
            .collect::<Vec<_>>();
        let f32_gemv_weight = DenseWeight::from_f32(f32_gemv_values, 3, COLS);
        let q8_gemv_values = (0..3 * COLS)
            .map(|i| ((i * 23 % 47) as f32 - 23.0) / 9.0)
            .collect::<Vec<_>>();
        let q8_gemv_weight =
            DenseWeight::from_q8_0_bytes(q8_bytes(&q8_gemv_values), 3, COLS).unwrap();
        let f32_embedding_values = (0..5 * COLS)
            .map(|i| ((i * 13 % 41) as f32 - 20.0) / 6.0)
            .collect::<Vec<_>>();
        let f32_embedding_weight = DenseWeight::from_f32(f32_embedding_values, 5, COLS);
        let q8_embedding_values = (0..3 * COLS)
            .map(|i| ((i * 29 % 53) as f32 - 26.0) / 8.0)
            .collect::<Vec<_>>();
        let q8_embedding_weight =
            DenseWeight::from_q8_0_bytes(q8_bytes(&q8_embedding_values), 3, COLS).unwrap();

        let f32_gemv_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.f32_gemv").unwrap(),
                &f32_gemv_weight,
            )
            .unwrap();
        let q8_gemv_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.q8_gemv").unwrap(),
                &q8_gemv_weight,
            )
            .unwrap();
        let f32_embedding_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.f32_embedding").unwrap(),
                &f32_embedding_weight,
            )
            .unwrap();
        let q8_embedding_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.q8_embedding").unwrap(),
                &q8_embedding_weight,
            )
            .unwrap();
        let registered = executor.execution_snapshot();
        assert_eq!(registered.dense_weights_registered, 4);

        let input = executor.create_scratch(COLS).unwrap();
        let f32_output = executor.create_scratch(3).unwrap();
        let q8_output = executor.create_scratch(3).unwrap();
        let state = executor.create_token_state().unwrap();
        gpu.queue
            .write_buffer(&input.buffer, 0, bytemuck::cast_slice(&gemv_input_values));

        let f32_gemv_expected = create_test_expected_buffer(
            &gpu.device,
            &gpu.queue,
            "gpu_native_test_f32_gemv_expected",
            &f32_gemv_weight.matvec(&gemv_input_values),
        );
        let q8_gemv_expected = create_test_expected_buffer(
            &gpu.device,
            &gpu.queue,
            "gpu_native_test_q8_gemv_expected",
            &q8_gemv_weight.matvec(&gemv_input_values),
        );
        let f32_embedding_expected = [0usize, 2, 4]
            .into_iter()
            .map(|row| {
                let mut expected = Vec::new();
                f32_embedding_weight.row_dequant_into(row, &mut expected);
                create_test_expected_buffer(
                    &gpu.device,
                    &gpu.queue,
                    "gpu_native_test_f32_embedding_expected",
                    &expected,
                )
            })
            .collect::<Vec<_>>();
        let q8_embedding_expected = [0usize, 1, 2]
            .into_iter()
            .map(|row| {
                let mut expected = Vec::new();
                q8_embedding_weight.row_dequant_into(row, &mut expected);
                create_test_expected_buffer(
                    &gpu.device,
                    &gpu.queue,
                    "gpu_native_test_q8_embedding_expected",
                    &expected,
                )
            })
            .collect::<Vec<_>>();

        let status = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_test_validation_status"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&status, 0, &[0; 4]);
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_test_validation_staging"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (compare_layout, compare_pipeline) = create_test_compare_pipeline(&gpu.device);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_native_test_live_l4_encoder"),
            });

        for _ in 0..2 {
            executor
                .encode_dense_gemv_scratch_to_scratch(
                    &mut encoder,
                    &f32_gemv_handle,
                    &input,
                    &f32_output,
                )
                .unwrap();
            encode_test_compare(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &f32_output.buffer,
                &f32_gemv_expected,
                &status,
                3,
                1e-4,
            );
            executor
                .encode_dense_gemv_scratch_to_scratch(
                    &mut encoder,
                    &q8_gemv_handle,
                    &input,
                    &q8_output,
                )
                .unwrap();
            encode_test_compare(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &q8_output.buffer,
                &q8_gemv_expected,
                &status,
                3,
                1e-4,
            );
        }

        for (token, expected) in [0u32, 2, 4].into_iter().zip(&f32_embedding_expected) {
            executor
                .encode_embedding_lookup(&mut encoder, &f32_embedding_handle, token, &state)
                .unwrap();
            encode_test_compare(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &state.hidden,
                expected,
                &status,
                COLS,
                1e-6,
            );
        }
        for (token, expected) in [0u32, 1, 2].into_iter().zip(&q8_embedding_expected) {
            executor
                .encode_embedding_lookup(&mut encoder, &q8_embedding_handle, token, &state)
                .unwrap();
            encode_test_compare(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &state.hidden,
                expected,
                &status,
                COLS,
                1e-6,
            );
        }

        let encoded = executor.execution_snapshot();
        assert_eq!(
            encoded.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(
            encoded.dense_weight_upload_bytes,
            registered.dense_weight_upload_bytes
        );
        assert_eq!(encoded.intermediate_maps, 0);
        assert_eq!(encoded.intermediate_readbacks, 0);
        encoder.copy_buffer_to_buffer(&status, 0, &staging, 0, GPU_NATIVE_STATUS_BYTES);
        gpu.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("validation map callback must be drained")
            .expect("validation status must map");
        let mapped = slice.get_mapped_range();
        let status_value = u32::from_le_bytes(mapped[..4].try_into().unwrap());
        drop(mapped);
        staging.unmap();
        assert_eq!(status_value, 0, "on-device GPU-native comparison failed");

        let completed = executor.execution_snapshot();
        assert_eq!(
            completed.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(
            completed.dense_weight_upload_bytes,
            registered.dense_weight_upload_bytes
        );
        assert_eq!(completed.intermediate_maps, 0);
        assert_eq!(completed.intermediate_readbacks, 0);
    }

    /// Requires an actual NVIDIA L4 WGPU adapter. All production operations
    /// share one caller-owned encoder; validation maps only a four-byte status.
    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_rmsnorm_residual_chain() {
        use super::super::{
            resolve_execution_context, ComputeOffload, GpuBackendGeometry, RoutedExpertGpuSpec,
        };
        use crate::inference::WeightDtype;

        const WIDTH: usize = 65;
        const GROUPS: usize = 3;
        const GROUP_WIDTH: usize = 67;
        const EPSILON: f32 = 1e-6;

        let expert_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(
            1024 * 1024,
            0.5,
            16,
        ));
        let execution = resolve_execution_context(
            ComputeOffload::Gpu,
            false,
            GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: 8,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 8,
                v_head_dim: 8,
                q4_truncation_tolerance: 0,
            },
            RoutedExpertGpuSpec {
                dtype: WeightDtype::F32,
                d_model: 32,
                d_ff: 64,
            },
            expert_cache,
        )
        .expect("L4 must construct the authoritative production GPU backend");
        let executor = execution
            .create_gpu_native_executor_context(WIDTH)
            .expect("GPU-native executor must retain the authoritative backend");
        let gpu = executor.authoritative_gpu().unwrap();

        let first_gain = (0..WIDTH)
            .map(|index| 0.75 + (index * 7 % 17) as f32 / 40.0)
            .collect::<Vec<_>>();
        let second_gain = (0..WIDTH)
            .map(|index| 0.9 + (index * 11 % 19) as f32 / 50.0)
            .collect::<Vec<_>>();
        let final_gain = (0..WIDTH)
            .map(|index| 1.05 - (index * 5 % 13) as f32 / 60.0)
            .collect::<Vec<_>>();
        let grouped_gain = (0..GROUP_WIDTH)
            .map(|index| 0.8 + (index * 13 % 23) as f32 / 55.0)
            .collect::<Vec<_>>();

        let first_norm = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.norm.first").unwrap(),
                &first_gain,
            )
            .unwrap();
        let second_norm = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.norm.second").unwrap(),
                &second_gain,
            )
            .unwrap();
        let final_norm = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.norm.final").unwrap(),
                &final_gain,
            )
            .unwrap();
        let grouped_norm = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.norm.grouped").unwrap(),
                &grouped_gain,
            )
            .unwrap();
        let norm_registered = executor.execution_snapshot();
        assert_eq!(norm_registered.dense_weights_registered, 4);
        assert_eq!(norm_registered.dense_weight_uploads, 4);

        let embedding = DenseWeight::from_f32(
            (0..3 * WIDTH)
                .map(|index| ((index * 13 % 43) as f32 - 21.0) / 8.0)
                .collect(),
            3,
            WIDTH,
        );
        let first_dense = DenseWeight::from_f32(
            (0..WIDTH * WIDTH)
                .map(|index| ((index * 17 % 47) as f32 - 23.0) / 97.0)
                .collect(),
            WIDTH,
            WIDTH,
        );
        let second_dense_values = (0..WIDTH * WIDTH)
            .map(|index| ((index * 19 % 53) as f32 - 26.0) / 89.0)
            .collect::<Vec<_>>();
        let second_dense =
            DenseWeight::from_q8_0_bytes(q8_bytes(&second_dense_values), WIDTH, WIDTH).unwrap();
        let embedding_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.chain.embedding").unwrap(),
                &embedding,
            )
            .unwrap();
        let first_dense_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.chain.first_dense").unwrap(),
                &first_dense,
            )
            .unwrap();
        let second_dense_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.chain.second_dense").unwrap(),
                &second_dense,
            )
            .unwrap();
        let registered = executor.execution_snapshot();
        assert_eq!(registered.dense_weights_registered, 7);
        assert_eq!(registered.dense_weight_uploads, 7);

        let mut expected_hidden = Vec::new();
        embedding.row_dequant_into(1, &mut expected_hidden);
        let first_residual = expected_hidden.clone();
        expected_hidden =
            crate::transformer::RmsNorm::new(first_gain, EPSILON).forward(&expected_hidden);
        let first_contribution = first_dense.matvec(&expected_hidden);
        expected_hidden = residual_add_mirror(&first_residual, &first_contribution);
        let expected_residual = expected_hidden.clone();
        expected_hidden =
            crate::transformer::RmsNorm::new(second_gain, EPSILON).forward(&expected_hidden);
        let second_contribution = second_dense.matvec(&expected_hidden);
        expected_hidden = residual_add_mirror(&expected_residual, &second_contribution);
        expected_hidden =
            crate::transformer::RmsNorm::new(final_gain, EPSILON).forward(&expected_hidden);

        let grouped_input = (0..GROUPS * GROUP_WIDTH)
            .map(|index| ((index * 23 % 59) as f32 - 29.0) / 11.0)
            .collect::<Vec<_>>();
        let grouped_expected = grouped_input
            .chunks_exact(GROUP_WIDTH)
            .flat_map(|group| {
                crate::transformer::RmsNorm::new(grouped_gain.clone(), EPSILON).forward(group)
            })
            .collect::<Vec<_>>();

        let state = executor.create_token_state().unwrap();
        let contribution = executor.create_scratch(WIDTH).unwrap();
        let grouped = executor.create_scratch(GROUPS * GROUP_WIDTH).unwrap();
        gpu.queue
            .write_buffer(&grouped.buffer, 0, bytemuck::cast_slice(&grouped_input));
        let expected_hidden_buffer = create_test_expected_buffer(
            &gpu.device,
            &gpu.queue,
            "gpu_native_rms_chain_expected_hidden",
            &expected_hidden,
        );
        let expected_residual_buffer = create_test_expected_buffer(
            &gpu.device,
            &gpu.queue,
            "gpu_native_rms_chain_expected_residual",
            &expected_residual,
        );
        let grouped_expected_buffer = create_test_expected_buffer(
            &gpu.device,
            &gpu.queue,
            "gpu_native_grouped_rms_expected",
            &grouped_expected,
        );
        let status = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_rms_chain_validation_status"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&status, 0, &[0; 4]);
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_rms_chain_validation_staging"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (compare_layout, compare_pipeline) = create_test_compare_pipeline(&gpu.device);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_native_rms_chain_live_l4_encoder"),
            });

        executor
            .encode_embedding_lookup(&mut encoder, &embedding_handle, 1, &state)
            .unwrap();
        executor
            .encode_rms_norm_state_in_place(&mut encoder, &first_norm, EPSILON, &state)
            .unwrap();
        executor
            .encode_dense_gemv_hidden_to_scratch(
                &mut encoder,
                &first_dense_handle,
                &state,
                &contribution,
            )
            .unwrap();
        executor
            .encode_residual_add_scratch_to_hidden(&mut encoder, &state, &contribution)
            .unwrap();
        executor
            .encode_rms_norm_state_in_place(&mut encoder, &second_norm, EPSILON, &state)
            .unwrap();
        executor
            .encode_dense_gemv_hidden_to_scratch(
                &mut encoder,
                &second_dense_handle,
                &state,
                &contribution,
            )
            .unwrap();
        executor
            .encode_residual_add_scratch_to_hidden(&mut encoder, &state, &contribution)
            .unwrap();
        executor
            .encode_rms_norm_hidden_in_place(&mut encoder, &final_norm, EPSILON, &state)
            .unwrap();
        executor
            .encode_rms_norm_scratch_in_place(
                &mut encoder,
                &grouped_norm,
                EPSILON,
                &grouped,
                GROUPS,
                GROUP_WIDTH,
            )
            .unwrap();

        for (actual, expected, elements) in [
            (&state.hidden, &expected_hidden_buffer, WIDTH),
            (&state.residual, &expected_residual_buffer, WIDTH),
            (
                &grouped.buffer,
                &grouped_expected_buffer,
                GROUPS * GROUP_WIDTH,
            ),
        ] {
            encode_test_compare(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                actual,
                expected,
                &status,
                elements,
                2e-3,
            );
        }

        let encoded = executor.execution_snapshot();
        assert_eq!(
            encoded.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(
            encoded.dense_weight_upload_bytes,
            registered.dense_weight_upload_bytes
        );
        assert_eq!(encoded.embedding_dispatches, 1);
        assert_eq!(encoded.dense_gemv_dispatches, 2);
        assert_eq!(encoded.rms_norm_dispatches, 4);
        assert_eq!(encoded.rms_norm_groups, 6);
        assert_eq!(encoded.rms_norm_state_dispatches, 3);
        assert_eq!(encoded.rms_norm_scratch_dispatches, 1);
        assert_eq!(encoded.residual_add_dispatches, 2);
        assert_eq!(encoded.queue_submissions, 0);
        assert_eq!(encoded.intermediate_maps, 0);
        assert_eq!(encoded.intermediate_readbacks, 0);

        encoder.copy_buffer_to_buffer(&status, 0, &staging, 0, GPU_NATIVE_STATUS_BYTES);
        gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("validation map callback must be drained")
            .expect("validation status must map");
        let mapped = slice.get_mapped_range();
        let status_value = u32::from_le_bytes(mapped[..4].try_into().unwrap());
        drop(mapped);
        staging.unmap();
        assert_eq!(status_value, 0, "on-device RMS/residual comparison failed");

        let completed = executor.execution_snapshot();
        assert_eq!(
            completed.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(completed.intermediate_maps, 0);
        assert_eq!(completed.intermediate_readbacks, 0);
    }

    /// Requires an actual NVIDIA L4 WGPU adapter. The production Q/K/V/KV
    /// buffers remain non-readable; validation maps one aggregate status word.
    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_attention_prepare_kv() {
        use super::super::{
            resolve_execution_context, ComputeOffload, GpuBackendGeometry, RoutedExpertGpuSpec,
        };
        use crate::inference::WeightDtype;

        const D_MODEL: usize = 6;
        const NUM_HEADS: usize = 4;
        const NUM_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 4;
        const ROPE_DIM: usize = 4;
        const MAX_SEQ_LEN: usize = 4;
        const EPSILON: f32 = 1e-6;
        let geometry = GpuNativeAttentionGeometry::try_new(
            D_MODEL,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            ROPE_DIM,
        )
        .unwrap();
        let expert_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(
            1024 * 1024,
            0.5,
            16,
        ));
        let execution = resolve_execution_context(
            ComputeOffload::Gpu,
            false,
            GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: MAX_SEQ_LEN,
                num_heads: NUM_HEADS,
                num_kv_heads: NUM_KV_HEADS,
                head_dim: HEAD_DIM,
                v_head_dim: HEAD_DIM,
                q4_truncation_tolerance: 0,
            },
            RoutedExpertGpuSpec {
                dtype: WeightDtype::F32,
                d_model: D_MODEL,
                d_ff: 8,
            },
            expert_cache,
        )
        .expect("L4 must construct the authoritative production GPU backend");
        let executor = execution
            .create_gpu_native_executor_context(D_MODEL)
            .expect("GPU-native executor must retain the authoritative backend");
        let gpu = executor.authoritative_gpu().unwrap();

        let q_projection = DenseWeight::from_f32(
            (0..geometry.q_width * D_MODEL)
                .map(|index| ((index * 11 % 37) as f32 - 18.0) / 19.0)
                .collect(),
            geometry.q_width,
            D_MODEL,
        );
        let k_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * D_MODEL)
                .map(|index| ((index * 13 % 41) as f32 - 20.0) / 17.0)
                .collect(),
            geometry.kv_width,
            D_MODEL,
        );
        let v_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * D_MODEL)
                .map(|index| ((index * 17 % 43) as f32 - 21.0) / 23.0)
                .collect(),
            geometry.kv_width,
            D_MODEL,
        );
        let o_projection = DenseWeight::from_f32(
            (0..D_MODEL * geometry.q_width)
                .map(|index| ((index * 19 % 47) as f32 - 23.0) / 29.0)
                .collect(),
            D_MODEL,
            geometry.q_width,
        );
        let q_gain = vec![0.8, 1.1, 0.9, 1.2];
        let k_gain = vec![1.3, 0.7, 1.0, 0.85];
        let q_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.attention.q").unwrap(),
                &q_projection,
            )
            .unwrap();
        let k_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.attention.k").unwrap(),
                &k_projection,
            )
            .unwrap();
        let v_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.attention.v").unwrap(),
                &v_projection,
            )
            .unwrap();
        let o_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.attention.o").unwrap(),
                &o_projection,
            )
            .unwrap();
        let q_norm_handle = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.attention.q_norm").unwrap(),
                &q_gain,
            )
            .unwrap();
        let k_norm_handle = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.attention.k_norm").unwrap(),
                &k_gain,
            )
            .unwrap();
        let rope_handle = executor
            .register_standard_rope(
                GpuNativeDenseWeightKey::try_new("test.attention.rope").unwrap(),
                ROPE_DIM,
                10_000.0,
            )
            .unwrap();
        let plan = executor
            .create_attention_plan(
                0,
                geometry,
                q_handle,
                k_handle,
                v_handle,
                o_handle,
                Some(GpuNativeAttentionNorm::try_new(q_norm_handle, EPSILON).unwrap()),
                Some(GpuNativeAttentionNorm::try_new(k_norm_handle, EPSILON).unwrap()),
                rope_handle,
            )
            .unwrap();
        let scratch = executor.create_attention_scratch(geometry).unwrap();
        let kv = executor
            .create_kv_state(1, MAX_SEQ_LEN, geometry.kv_width)
            .unwrap();
        let states = [
            executor.create_token_state().unwrap(),
            executor.create_token_state().unwrap(),
        ];
        let inputs = [
            vec![0.5, -1.0, 1.5, -2.0, 2.5, -3.0],
            vec![-0.25, 0.75, -1.25, 1.75, -2.25, 2.75],
        ];
        let positions = [1usize, 3usize];
        for (state, input) in states.iter().zip(&inputs) {
            gpu.queue
                .write_buffer(&state.hidden, 0, bytemuck::cast_slice(input));
        }
        let inverse_frequencies = [1.0, 0.01];
        let expected = inputs
            .iter()
            .zip(positions)
            .map(|(input, position)| {
                attention_prepare_mirror(
                    input,
                    geometry,
                    &q_projection,
                    &k_projection,
                    &v_projection,
                    Some((&q_gain, EPSILON)),
                    Some((&k_gain, EPSILON)),
                    &inverse_frequencies,
                    1.0,
                    position,
                )
            })
            .collect::<Vec<_>>();
        let expected_buffers = expected
            .iter()
            .enumerate()
            .map(|(index, (q, k, v))| {
                (
                    create_test_expected_buffer(
                        &gpu.device,
                        &gpu.queue,
                        &format!("gpu_native_attention_q_expected_{index}"),
                        q,
                    ),
                    create_test_expected_buffer(
                        &gpu.device,
                        &gpu.queue,
                        &format!("gpu_native_attention_k_expected_{index}"),
                        k,
                    ),
                    create_test_expected_buffer(
                        &gpu.device,
                        &gpu.queue,
                        &format!("gpu_native_attention_v_expected_{index}"),
                        v,
                    ),
                )
            })
            .collect::<Vec<_>>();

        let registered = executor.execution_snapshot();
        assert_eq!(registered.dense_weights_registered, 7);
        assert_eq!(registered.rope_parameters_registered, 1);
        assert_eq!(registered.rope_parameter_uploads, 1);
        let status = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_attention_validation_status"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&status, 0, &[0; 4]);
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_attention_validation_staging"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (compare_layout, compare_pipeline) = create_test_compare_pipeline(&gpu.device);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_native_attention_prepare_live_l4_encoder"),
            });

        for index in 0..states.len() {
            executor
                .encode_attention_prepare(
                    &mut encoder,
                    &plan,
                    &states[index],
                    &scratch,
                    &kv,
                    positions[index],
                )
                .unwrap();
            let (expected_q, expected_k, expected_v) = &expected_buffers[index];
            for (actual, expected_buffer, elements) in [
                (&scratch.q.buffer, expected_q, geometry.q_width),
                (&scratch.k.buffer, expected_k, geometry.kv_width),
                (&scratch.v.buffer, expected_v, geometry.kv_width),
            ] {
                encode_test_compare(
                    &gpu.device,
                    &mut encoder,
                    &compare_layout,
                    &compare_pipeline,
                    actual,
                    expected_buffer,
                    &status,
                    elements,
                    2e-3,
                );
            }
        }
        for index in 0..positions.len() {
            let offset = kv.layout.element_offset(0, positions[index]).unwrap();
            let (_, expected_k, expected_v) = &expected_buffers[index];
            encode_test_compare_at(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &kv.layers[0].key,
                expected_k,
                &status,
                geometry.kv_width,
                2e-3,
                offset,
            );
            encode_test_compare_at(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &kv.layers[0].value,
                expected_v,
                &status,
                geometry.kv_width,
                2e-3,
                offset,
            );
        }

        let encoded = executor.execution_snapshot();
        assert_eq!(
            encoded.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(
            encoded.dense_weight_upload_bytes,
            registered.dense_weight_upload_bytes
        );
        assert_eq!(
            encoded.rope_parameter_uploads,
            registered.rope_parameter_uploads
        );
        assert_eq!(
            encoded.rope_parameter_upload_bytes,
            registered.rope_parameter_upload_bytes
        );
        assert_eq!(encoded.dense_gemv_dispatches, 6);
        assert_eq!(encoded.rms_norm_dispatches, 4);
        assert_eq!(encoded.rms_norm_groups, 12);
        assert_eq!(encoded.attention_prepare_dispatches, 2);
        assert_eq!(encoded.q_projection_dispatches, 2);
        assert_eq!(encoded.k_projection_dispatches, 2);
        assert_eq!(encoded.v_projection_dispatches, 2);
        assert_eq!(encoded.rope_dispatches, 4);
        assert_eq!(encoded.rope_groups, 12);
        assert_eq!(encoded.kv_appends, 2);
        assert_eq!(encoded.queue_submissions, 0);
        assert_eq!(encoded.intermediate_maps, 0);
        assert_eq!(encoded.intermediate_readbacks, 0);

        encoder.copy_buffer_to_buffer(&status, 0, &staging, 0, GPU_NATIVE_STATUS_BYTES);
        gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("validation map callback must be drained")
            .expect("validation status must map");
        let mapped = slice.get_mapped_range();
        let status_value = u32::from_le_bytes(mapped[..4].try_into().unwrap());
        drop(mapped);
        staging.unmap();
        assert_eq!(status_value, 0, "on-device attention preparation failed");

        let completed = executor.execution_snapshot();
        assert_eq!(
            completed.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(
            completed.rope_parameter_uploads,
            registered.rope_parameter_uploads
        );
        assert_eq!(completed.intermediate_maps, 0);
        assert_eq!(completed.intermediate_readbacks, 0);
    }

    /// Requires an actual NVIDIA L4 WGPU adapter. The production attention,
    /// KV, context, projected, and token-state buffers remain non-readable;
    /// validation maps one staging allocation containing the aggregate finite
    /// result and invalid-row status after one queue submission.
    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_causal_attention_residual() {
        use super::super::{
            resolve_execution_context, ComputeOffload, GpuBackendGeometry, RoutedExpertGpuSpec,
        };
        use crate::inference::WeightDtype;

        const D_MODEL: usize = 6;
        const NUM_HEADS: usize = 4;
        const NUM_KV_HEADS: usize = 2;
        const HEAD_DIM: usize = 2;
        const ROPE_DIM: usize = 2;
        const MAX_SEQ_LEN: usize = 4;
        const EPSILON: f32 = 1e-6;
        let geometry = GpuNativeAttentionGeometry::try_new(
            D_MODEL,
            NUM_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            ROPE_DIM,
        )
        .unwrap();
        assert_ne!(geometry.q_width, D_MODEL);

        let expert_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(
            1024 * 1024,
            0.5,
            16,
        ));
        let execution = resolve_execution_context(
            ComputeOffload::Gpu,
            false,
            GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: MAX_SEQ_LEN,
                num_heads: NUM_HEADS,
                num_kv_heads: NUM_KV_HEADS,
                head_dim: HEAD_DIM,
                v_head_dim: HEAD_DIM,
                q4_truncation_tolerance: 0,
            },
            RoutedExpertGpuSpec {
                dtype: WeightDtype::F32,
                d_model: D_MODEL,
                d_ff: 8,
            },
            expert_cache,
        )
        .expect("L4 must construct the authoritative production GPU backend");
        let executor = execution
            .create_gpu_native_executor_context(D_MODEL)
            .expect("GPU-native executor must retain the authoritative backend");
        let gpu = executor.authoritative_gpu().unwrap();

        let q_projection = DenseWeight::from_f32(
            (0..geometry.q_width * D_MODEL)
                .map(|index| ((index * 11 % 37) as f32 - 18.0) / 19.0)
                .collect(),
            geometry.q_width,
            D_MODEL,
        );
        let k_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * D_MODEL)
                .map(|index| ((index * 13 % 41) as f32 - 20.0) / 17.0)
                .collect(),
            geometry.kv_width,
            D_MODEL,
        );
        let v_projection = DenseWeight::from_f32(
            (0..geometry.kv_width * D_MODEL)
                .map(|index| ((index * 17 % 43) as f32 - 21.0) / 23.0)
                .collect(),
            geometry.kv_width,
            D_MODEL,
        );
        let o_projection = DenseWeight::from_f32(
            (0..D_MODEL * geometry.q_width)
                .map(|index| ((index * 19 % 47) as f32 - 23.0) / 29.0)
                .collect(),
            D_MODEL,
            geometry.q_width,
        );
        let q_gain = [0.8, 1.2];
        let k_gain = [1.1, 0.7];
        let q_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.causal_attention.q").unwrap(),
                &q_projection,
            )
            .unwrap();
        let k_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.causal_attention.k").unwrap(),
                &k_projection,
            )
            .unwrap();
        let v_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.causal_attention.v").unwrap(),
                &v_projection,
            )
            .unwrap();
        let o_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.causal_attention.o").unwrap(),
                &o_projection,
            )
            .unwrap();
        let q_norm_handle = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.causal_attention.q_norm").unwrap(),
                &q_gain,
            )
            .unwrap();
        let k_norm_handle = executor
            .register_rms_norm(
                GpuNativeDenseWeightKey::try_new("test.causal_attention.k_norm").unwrap(),
                &k_gain,
            )
            .unwrap();
        let rope_handle = executor
            .register_standard_rope(
                GpuNativeDenseWeightKey::try_new("test.causal_attention.rope").unwrap(),
                ROPE_DIM,
                10_000.0,
            )
            .unwrap();
        let plan = executor
            .create_attention_plan(
                0,
                geometry,
                q_handle,
                k_handle,
                v_handle,
                o_handle,
                Some(GpuNativeAttentionNorm::try_new(q_norm_handle, EPSILON).unwrap()),
                Some(GpuNativeAttentionNorm::try_new(k_norm_handle, EPSILON).unwrap()),
                rope_handle,
            )
            .unwrap();
        let scratch = executor.create_attention_scratch(geometry).unwrap();
        let poison_k = executor.create_scratch(geometry.kv_width).unwrap();
        let poison_v = executor.create_scratch(geometry.kv_width).unwrap();
        let kv = executor
            .create_kv_state(1, MAX_SEQ_LEN, geometry.kv_width)
            .unwrap();
        let states = [
            executor.create_token_state().unwrap(),
            executor.create_token_state().unwrap(),
            executor.create_token_state().unwrap(),
        ];
        let invalid_scratch = executor.create_attention_scratch(geometry).unwrap();
        let invalid_k = executor.create_scratch(geometry.kv_width).unwrap();
        let invalid_v = executor.create_scratch(geometry.kv_width).unwrap();
        let invalid_kv = executor
            .create_kv_state(1, MAX_SEQ_LEN, geometry.kv_width)
            .unwrap();
        let invalid_state = executor.create_token_state().unwrap();
        let prepared_inputs = [
            vec![0.5, -1.0, 1.5, -2.0, 2.5, -3.0],
            vec![-0.25, 0.75, -1.25, 1.75, -2.25, 2.75],
            vec![1.1, -0.4, 0.9, -1.6, 2.2, -2.8],
        ];
        let saved_residuals = [
            vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            vec![-0.6, 0.5, -0.4, 0.3, -0.2, 0.1],
            vec![0.7, -0.8, 0.9, -1.0, 1.1, -1.2],
        ];
        for ((state, prepared), residual) in
            states.iter().zip(&prepared_inputs).zip(&saved_residuals)
        {
            gpu.queue
                .write_buffer(&state.hidden, 0, bytemuck::cast_slice(prepared));
            gpu.queue
                .write_buffer(&state.residual, 0, bytemuck::cast_slice(residual));
        }
        let future_key_poison = [10_000.0, -9_000.0, 8_000.0, -7_000.0];
        let future_value_poison = [50_000.0, -40_000.0, 30_000.0, -20_000.0];
        gpu.queue.write_buffer(
            &poison_k.buffer,
            0,
            bytemuck::cast_slice(&future_key_poison),
        );
        gpu.queue.write_buffer(
            &poison_v.buffer,
            0,
            bytemuck::cast_slice(&future_value_poison),
        );
        let invalid_query = vec![f32::INFINITY; geometry.q_width];
        let invalid_key: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let invalid_value: [f32; 4] = [0.25, -0.5, 0.75, -1.0];
        assert_eq!(
            std::mem::size_of_val(&invalid_key),
            geometry.kv_width * std::mem::size_of::<f32>()
        );
        assert_eq!(
            std::mem::size_of_val(&invalid_value),
            geometry.kv_width * std::mem::size_of::<f32>()
        );
        gpu.queue.write_buffer(
            &invalid_scratch.q.buffer,
            0,
            bytemuck::cast_slice(&invalid_query),
        );
        gpu.queue
            .write_buffer(&invalid_k.buffer, 0, bytemuck::cast_slice(&invalid_key));
        gpu.queue
            .write_buffer(&invalid_v.buffer, 0, bytemuck::cast_slice(&invalid_value));

        let inverse_frequencies = [1.0];
        let mut expected_keys = vec![0.0; MAX_SEQ_LEN * geometry.kv_width];
        let mut expected_values = vec![0.0; MAX_SEQ_LEN * geometry.kv_width];
        expected_keys[3 * geometry.kv_width..].copy_from_slice(&future_key_poison);
        expected_values[3 * geometry.kv_width..].copy_from_slice(&future_value_poison);
        let mut expected = Vec::new();
        for position in 0..3 {
            let (q, k, v) = attention_prepare_mirror(
                &prepared_inputs[position],
                geometry,
                &q_projection,
                &k_projection,
                &v_projection,
                Some((&q_gain, EPSILON)),
                Some((&k_gain, EPSILON)),
                &inverse_frequencies,
                1.0,
                position,
            );
            let offset = position * geometry.kv_width;
            expected_keys[offset..offset + geometry.kv_width].copy_from_slice(&k);
            expected_values[offset..offset + geometry.kv_width].copy_from_slice(&v);
            let (context, projected, hidden) = attention_complete_mirror(
                &q,
                &expected_keys,
                &expected_values,
                geometry,
                position + 1,
                &o_projection,
                &saved_residuals[position],
            );
            expected.push((q, k, v, context, projected, hidden));
        }
        let expected_buffers = expected
            .iter()
            .enumerate()
            .map(|(index, (_, _, _, context, projected, hidden))| {
                (
                    create_test_expected_buffer(
                        &gpu.device,
                        &gpu.queue,
                        &format!("gpu_native_causal_context_expected_{index}"),
                        context,
                    ),
                    create_test_expected_buffer(
                        &gpu.device,
                        &gpu.queue,
                        &format!("gpu_native_causal_projected_expected_{index}"),
                        projected,
                    ),
                    create_test_expected_buffer(
                        &gpu.device,
                        &gpu.queue,
                        &format!("gpu_native_causal_hidden_expected_{index}"),
                        hidden,
                    ),
                    create_test_expected_buffer(
                        &gpu.device,
                        &gpu.queue,
                        &format!("gpu_native_causal_residual_expected_{index}"),
                        &saved_residuals[index],
                    ),
                )
            })
            .collect::<Vec<_>>();
        let expected_key_buffers = (0..MAX_SEQ_LEN)
            .map(|position| {
                let offset = position * geometry.kv_width;
                create_test_expected_buffer(
                    &gpu.device,
                    &gpu.queue,
                    &format!("gpu_native_causal_key_expected_{position}"),
                    &expected_keys[offset..offset + geometry.kv_width],
                )
            })
            .collect::<Vec<_>>();
        let expected_value_buffers = (0..MAX_SEQ_LEN)
            .map(|position| {
                let offset = position * geometry.kv_width;
                create_test_expected_buffer(
                    &gpu.device,
                    &gpu.queue,
                    &format!("gpu_native_causal_value_expected_{position}"),
                    &expected_values[offset..offset + geometry.kv_width],
                )
            })
            .collect::<Vec<_>>();
        let expected_sanitized_context = create_test_expected_buffer(
            &gpu.device,
            &gpu.queue,
            "gpu_native_causal_invalid_context_expected",
            &vec![0.0; geometry.q_width],
        );

        let registered = executor.execution_snapshot();
        assert_eq!(registered.dense_weights_registered, 7);
        assert_eq!(registered.rope_parameters_registered, 1);
        let status = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_causal_attention_validation_status"),
            size: GPU_NATIVE_STATUS_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        gpu.queue.write_buffer(&status, 0, &[0; 4]);
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_causal_attention_validation_staging"),
            size: GPU_NATIVE_STATUS_BYTES * 2,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (compare_layout, compare_pipeline) = create_test_compare_pipeline(&gpu.device);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_native_causal_attention_live_l4_encoder"),
            });

        executor
            .encode_kv_append(&mut encoder, &poison_k, &poison_v, &kv, 0, 3)
            .unwrap();
        for position in 0..3 {
            executor
                .encode_attention_prepare(
                    &mut encoder,
                    &plan,
                    &states[position],
                    &scratch,
                    &kv,
                    position,
                )
                .unwrap();
            executor
                .encode_attention_complete(
                    &mut encoder,
                    &plan,
                    &states[position],
                    &scratch,
                    &kv,
                    position,
                )
                .unwrap();
            let (context, projected, hidden, residual) = &expected_buffers[position];
            for (actual, expected_buffer, elements) in [
                (&scratch.context.buffer, context, geometry.q_width),
                (&scratch.projected.buffer, projected, geometry.d_model),
                (&states[position].hidden, hidden, geometry.d_model),
                (&states[position].residual, residual, geometry.d_model),
            ] {
                encode_test_compare(
                    &gpu.device,
                    &mut encoder,
                    &compare_layout,
                    &compare_pipeline,
                    actual,
                    expected_buffer,
                    &status,
                    elements,
                    3e-3,
                );
            }
        }

        // Keep the injected failure request-local: its own Q scratch, KV, and
        // token status share the caller encoder but cannot perturb the finite
        // 0/1/2-position correctness path above.
        executor
            .encode_kv_append(&mut encoder, &invalid_k, &invalid_v, &invalid_kv, 0, 0)
            .unwrap();
        executor
            .encode_attention_complete(
                &mut encoder,
                &plan,
                &invalid_state,
                &invalid_scratch,
                &invalid_kv,
                0,
            )
            .unwrap();
        encode_test_compare(
            &gpu.device,
            &mut encoder,
            &compare_layout,
            &compare_pipeline,
            &invalid_scratch.context.buffer,
            &expected_sanitized_context,
            &status,
            geometry.q_width,
            0.0,
        );
        for position in 0..MAX_SEQ_LEN {
            let offset = position * geometry.kv_width;
            encode_test_compare_at(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &kv.layers[0].key,
                &expected_key_buffers[position],
                &status,
                geometry.kv_width,
                3e-3,
                offset,
            );
            encode_test_compare_at(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &kv.layers[0].value,
                &expected_value_buffers[position],
                &status,
                geometry.kv_width,
                3e-3,
                offset,
            );
        }

        let encoded = executor.execution_snapshot();
        assert_eq!(
            encoded.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(
            encoded.dense_weight_upload_bytes,
            registered.dense_weight_upload_bytes
        );
        assert_eq!(
            encoded.rope_parameter_uploads,
            registered.rope_parameter_uploads
        );
        assert_eq!(encoded.dense_gemv_dispatches, 13);
        assert_eq!(encoded.rms_norm_dispatches, 6);
        assert_eq!(encoded.rms_norm_groups, 18);
        assert_eq!(encoded.attention_prepare_dispatches, 3);
        assert_eq!(encoded.q_projection_dispatches, 3);
        assert_eq!(encoded.k_projection_dispatches, 3);
        assert_eq!(encoded.v_projection_dispatches, 3);
        assert_eq!(encoded.rope_dispatches, 6);
        assert_eq!(encoded.rope_groups, 18);
        assert_eq!(encoded.kv_appends, 5);
        assert_eq!(encoded.causal_attention_dispatches, 4);
        assert_eq!(encoded.o_projection_dispatches, 4);
        assert_eq!(encoded.attention_complete_dispatches, 4);
        assert_eq!(encoded.residual_add_dispatches, 4);
        assert_eq!(encoded.queue_submissions, 0);
        assert_eq!(encoded.intermediate_maps, 0);
        assert_eq!(encoded.intermediate_readbacks, 0);
        assert_eq!(encoded.cpu_attention_calls, 0);
        assert_eq!(encoded.cpu_kv_mutations, 0);
        assert_eq!(encoded.cpu_layer_reentries, 0);
        assert_eq!(encoded.numerical_failures, 0);

        encoder.copy_buffer_to_buffer(&status, 0, &staging, 0, GPU_NATIVE_STATUS_BYTES);
        encoder.copy_buffer_to_buffer(
            &invalid_state.status,
            0,
            &staging,
            GPU_NATIVE_STATUS_BYTES,
            GPU_NATIVE_STATUS_BYTES,
        );
        gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("validation map callback must be drained")
            .expect("validation status must map");
        let mapped = slice.get_mapped_range();
        let status_value = u32::from_le_bytes(mapped[..4].try_into().unwrap());
        let invalid_status_value = u32::from_le_bytes(mapped[4..8].try_into().unwrap());
        drop(mapped);
        staging.unmap();
        assert_eq!(status_value, 0, "on-device causal attention failed");
        assert_ne!(
            invalid_status_value & GPU_NATIVE_STATUS_ATTENTION_NUMERICAL_FAILURE,
            0,
            "invalid causal-attention score did not latch device status"
        );

        let completed = executor.execution_snapshot();
        assert_eq!(
            completed.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(
            completed.rope_parameter_uploads,
            registered.rope_parameter_uploads
        );
        assert_eq!(completed.intermediate_maps, 0);
        assert_eq!(completed.intermediate_readbacks, 0);
        assert_eq!(completed.cpu_attention_calls, 0);
        assert_eq!(completed.cpu_kv_mutations, 0);
        assert_eq!(completed.cpu_layer_reentries, 0);
        assert_eq!(completed.numerical_failures, 0);
    }

    /// Requires an actual NVIDIA L4 WGPU adapter. Three isolated requests
    /// (finite, exact-tie, and non-finite) share one caller-owned encoder, one
    /// explicit queue submission, and one final test-only staging map.
    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_router_topk() {
        use super::super::{
            resolve_execution_context, ComputeOffload, GpuBackendGeometry, RoutedExpertGpuSpec,
        };
        use crate::gating::LinearGate;
        use crate::inference::WeightDtype;

        const D_MODEL: usize = 16;
        const NUM_EXPERTS: usize = 128;
        const TOP_K: usize = 8;
        const IDS_BYTES: u64 = (TOP_K * std::mem::size_of::<u32>()) as u64;
        const WEIGHTS_BYTES: u64 = (TOP_K * std::mem::size_of::<f32>()) as u64;
        const CASE_BYTES: u64 = IDS_BYTES + WEIGHTS_BYTES + GPU_NATIVE_STATUS_BYTES;
        const CASES: usize = 3;

        let expert_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(
            1024 * 1024,
            0.5,
            16,
        ));
        let execution = resolve_execution_context(
            ComputeOffload::Gpu,
            false,
            GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: 8,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: D_MODEL,
                v_head_dim: D_MODEL,
                q4_truncation_tolerance: 0,
            },
            RoutedExpertGpuSpec {
                dtype: WeightDtype::F32,
                d_model: D_MODEL,
                d_ff: 32,
            },
            expert_cache,
        )
        .expect("L4 must construct the authoritative production GPU backend");
        let executor = execution
            .create_gpu_native_executor_context(D_MODEL)
            .expect("GPU-native executor must retain the authoritative backend");
        assert_eq!(executor.device_identity().vendor_id, 0x10de);
        assert!(
            executor.device_identity().name.contains("L4"),
            "ignored router test must run only on an NVIDIA L4, got {}",
            executor.device_identity().name
        );
        let gpu = executor.authoritative_gpu().unwrap();
        let geometry = GpuNativeRouterGeometry::try_new(D_MODEL, NUM_EXPERTS, TOP_K).unwrap();

        let mut gate_values = vec![0.0; NUM_EXPERTS * D_MODEL];
        for expert in 0..NUM_EXPERTS {
            for column in 0..D_MODEL {
                gate_values[expert * D_MODEL + column] = if column == 0 {
                    expert as f32 / 4.0
                } else {
                    ((expert * 17 + column * 13) % 19) as f32 / 100_000.0
                };
            }
        }
        let gate_weight = DenseWeight::from_f32(gate_values.clone(), NUM_EXPERTS, D_MODEL);
        let gate_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.router.gate").unwrap(),
                &gate_weight,
            )
            .unwrap();
        let plan = executor
            .create_router_plan(0, geometry, gate_handle)
            .unwrap();

        let finite_hidden = [
            1.0, -0.5, 0.25, 0.75, -1.0, 0.5, 0.125, -0.25, 0.375, -0.625, 0.875, -0.75, 0.625,
            -0.375, 0.2, -0.1,
        ];
        let tie_hidden = [0.0; D_MODEL];
        let invalid_hidden = [f32::NAN; D_MODEL];
        let states = [
            executor.create_token_state().unwrap(),
            executor.create_token_state().unwrap(),
            executor.create_token_state().unwrap(),
        ];
        let scratches = [
            executor.create_router_scratch(geometry).unwrap(),
            executor.create_router_scratch(geometry).unwrap(),
            executor.create_router_scratch(geometry).unwrap(),
        ];
        for (state, hidden) in
            states
                .iter()
                .zip([&finite_hidden[..], &tie_hidden[..], &invalid_hidden[..]])
        {
            gpu.queue
                .write_buffer(&state.hidden, 0, bytemuck::cast_slice(hidden));
            gpu.queue.write_buffer(&state.status, 0, &[0; 4]);
        }

        let cpu_reference =
            LinearGate::new(gate_values, NUM_EXPERTS, D_MODEL, TOP_K).route(&finite_hidden);
        assert_eq!(cpu_reference.experts.len(), TOP_K);
        assert!((cpu_reference.weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);

        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_router_validation_staging"),
            size: CASE_BYTES * CASES as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let registered = executor.execution_snapshot();
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_native_router_live_l4_encoder"),
            });
        for (index, (state, scratch)) in states.iter().zip(&scratches).enumerate() {
            executor
                .encode_router(&mut encoder, &plan, state, scratch)
                .unwrap();
            let case_offset = index as u64 * CASE_BYTES;
            encoder.copy_buffer_to_buffer(
                &scratch.selected_ids,
                0,
                &staging,
                case_offset,
                IDS_BYTES,
            );
            encoder.copy_buffer_to_buffer(
                &scratch.selected_weights,
                0,
                &staging,
                case_offset + IDS_BYTES,
                WEIGHTS_BYTES,
            );
            encoder.copy_buffer_to_buffer(
                &state.status,
                0,
                &staging,
                case_offset + IDS_BYTES + WEIGHTS_BYTES,
                GPU_NATIVE_STATUS_BYTES,
            );
        }

        let encoded = executor.execution_snapshot();
        assert_eq!(
            encoded.dense_weight_uploads,
            registered.dense_weight_uploads
        );
        assert_eq!(encoded.dense_gemv_dispatches, 3);
        assert_eq!(encoded.router_logit_dispatches, 3);
        assert_eq!(encoded.router_topk_dispatches, 3);
        assert_eq!(encoded.queue_submissions, 0);
        assert_eq!(encoded.intermediate_maps, 0);
        assert_eq!(encoded.intermediate_readbacks, 0);
        assert_eq!(encoded.cpu_router_calls, 0);
        assert_eq!(encoded.numerical_failures, 0);

        gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("validation map callback must be drained")
            .expect("router validation staging must map");
        let mapped = slice.get_mapped_range();
        let parse_case = |index: usize| {
            let start = index * CASE_BYTES as usize;
            let ids = mapped[start..start + IDS_BYTES as usize]
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            let weights_start = start + IDS_BYTES as usize;
            let weights = mapped[weights_start..weights_start + WEIGHTS_BYTES as usize]
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            let status_start = weights_start + WEIGHTS_BYTES as usize;
            let status = u32::from_le_bytes(
                mapped[status_start..status_start + GPU_NATIVE_STATUS_BYTES as usize]
                    .try_into()
                    .unwrap(),
            );
            (ids, weights, status)
        };
        let finite = parse_case(0);
        let tied = parse_case(1);
        let invalid = parse_case(2);
        drop(mapped);
        staging.unmap();

        assert_eq!(finite.0, cpu_reference.experts);
        assert_close(&finite.1, &cpu_reference.weights, 2e-5);
        assert!((finite.1.iter().sum::<f32>() - 1.0).abs() < 2e-5);
        assert_eq!(finite.2, 0, "finite request status must remain clear");
        assert_eq!(tied.0, (0..TOP_K as u32).collect::<Vec<_>>());
        assert_close(&tied.1, &[0.125; TOP_K], 1e-6);
        assert_eq!(tied.2, 0, "tie request status must remain clear");
        assert_eq!(invalid.0, vec![0; TOP_K]);
        assert_eq!(invalid.1, vec![0.0; TOP_K]);
        assert_ne!(
            invalid.2 & GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE,
            0,
            "non-finite router row must latch request-local status"
        );

        let completed = executor.execution_snapshot();
        assert_eq!(completed.intermediate_maps, 0);
        assert_eq!(completed.intermediate_readbacks, 0);
        assert_eq!(completed.cpu_router_calls, 0);
        assert_eq!(completed.numerical_failures, 0);
    }

    /// Requires an actual NVIDIA L4. A one-slot arena proves install,
    /// retirement, physical reuse, ordinary unmapped miss, stale resolved
    /// epoch containment, and stale logical requester rejection. Residency
    /// queue writes and every submission/readback are explicitly test-owned.
    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_mutable_expert_residency() {
        use super::super::{
            resolve_execution_context, ComputeOffload, GpuBackendGeometry, RoutedExpertGpuSpec,
        };
        use crate::inference::WeightDtype;

        const D_MODEL: usize = 32;
        const D_FF: usize = 32;
        const NUM_EXPERTS: usize = 128;
        const TOP_K: usize = 1;
        const RESOLVED_BYTES: u64 = GPU_NATIVE_EXPERT_MAPPING_ENTRY_BYTES as u64;
        const READBACK_BYTES: u64 =
            GPU_NATIVE_STATUS_BYTES + RESOLVED_BYTES + GPU_NATIVE_STATUS_BYTES;

        let expert_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(
            1024 * 1024,
            0.5,
            16,
        ));
        let execution = resolve_execution_context(
            ComputeOffload::Gpu,
            false,
            GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: 8,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: D_MODEL,
                v_head_dim: D_MODEL,
                q4_truncation_tolerance: 0,
            },
            RoutedExpertGpuSpec {
                dtype: WeightDtype::Q4_0,
                d_model: D_MODEL,
                d_ff: D_FF,
            },
            expert_cache,
        )
        .expect("L4 must construct the authoritative production GPU backend");
        let executor = execution
            .create_gpu_native_executor_context(D_MODEL)
            .expect("GPU-native executor must retain the authoritative backend");
        assert_eq!(executor.device_identity().vendor_id, 0x10de);
        assert!(
            executor.device_identity().name.contains("L4"),
            "ignored mutable residency test must run only on an NVIDIA L4, got {}",
            executor.device_identity().name
        );
        let gpu = executor.authoritative_gpu().unwrap();
        let router_geometry =
            GpuNativeRouterGeometry::try_new(D_MODEL, NUM_EXPERTS, TOP_K).unwrap();
        let expert_geometry =
            GpuNativeQ4ExpertGeometry::try_new(D_MODEL, D_FF, NUM_EXPERTS, TOP_K).unwrap();

        const EXPERT_A: u32 = 7;
        const EXPERT_B: u32 = 8;
        let mut gate_values = vec![0.0; NUM_EXPERTS * D_MODEL];
        gate_values[EXPERT_A as usize * D_MODEL] = 4.0;
        gate_values[EXPERT_B as usize * D_MODEL + 1] = 4.0;
        let gate = DenseWeight::from_f32(gate_values, NUM_EXPERTS, D_MODEL);
        let gate_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.mutable_expert.router").unwrap(),
                &gate,
            )
            .unwrap();
        let router_plan = executor
            .create_router_plan(0, router_geometry, gate_handle)
            .unwrap();
        let one_slot_layout =
            GpuNativeQ4ExpertArenaLayout::try_new(expert_geometry, 1, &gpu.device.limits())
                .unwrap();
        let arena_plan = GpuNativeQ4ExpertVramPlan::try_new(
            expert_geometry,
            one_slot_layout.total_allocation_bytes().unwrap(),
            &gpu.device.limits(),
        )
        .unwrap();
        let arena = executor.create_q4_expert_arena(0, arena_plan, &[]).unwrap();
        assert_eq!(arena.slot_capacity(), 1);

        let payload_a = q4_uniform_expert(expert_geometry, 0.01, 0.02, 0.005);
        let payload_b = q4_uniform_expert(expert_geometry, -0.015, 0.025, 0.004);
        let key_a = GpuNativeQ4ExpertKey::new(0, EXPERT_A, 10);
        let key_b = GpuNativeQ4ExpertKey::new(0, EXPERT_B, 20);
        let permit_a = match executor.acquire_q4_expert_residency(&arena, key_a).unwrap() {
            GpuNativeQ4ExpertAcquire::Install(permit) => permit,
            _ => panic!("empty one-slot arena must reserve expert A"),
        };
        let residency_a = executor
            .install_q4_expert_residency(permit_a, &payload_a)
            .unwrap();
        assert_ne!(residency_a.slot_epoch(), 0);

        let (compare_layout, compare_pipeline) = create_test_compare_pipeline(&gpu.device);
        let run_case = |label: &str,
                        hidden: &[f32],
                        residual: &[f32],
                        expected_hidden: &[f32],
                        tolerance: f32| {
            let state = executor.create_token_state().unwrap();
            let router_scratch = executor.create_router_scratch(router_geometry).unwrap();
            let expert_scratch = executor.create_q4_expert_scratch(expert_geometry).unwrap();
            let expected_buffer = create_test_expected_buffer(
                &gpu.device,
                &gpu.queue,
                &format!("{label}_expected_hidden"),
                expected_hidden,
            );
            let comparison_status = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label}_comparison_status")),
                size: GPU_NATIVE_STATUS_BYTES,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            gpu.queue
                .write_buffer(&state.hidden, 0, bytemuck::cast_slice(hidden));
            gpu.queue
                .write_buffer(&state.residual, 0, bytemuck::cast_slice(residual));
            gpu.queue.write_buffer(&state.status, 0, &[0; 4]);
            gpu.queue.write_buffer(&comparison_status, 0, &[0; 4]);
            let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: READBACK_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            executor
                .encode_router(&mut encoder, &router_plan, &state, &router_scratch)
                .unwrap();
            executor
                .encode_q4_expert_arena_combine(
                    &mut encoder,
                    &router_plan,
                    &router_scratch,
                    &arena,
                    &state,
                    &expert_scratch,
                )
                .unwrap();
            encode_test_compare(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &state.hidden,
                &expected_buffer,
                &comparison_status,
                D_MODEL,
                tolerance,
            );
            encoder.copy_buffer_to_buffer(&state.status, 0, &staging, 0, GPU_NATIVE_STATUS_BYTES);
            encoder.copy_buffer_to_buffer(
                &expert_scratch.resolved_locations,
                0,
                &staging,
                GPU_NATIVE_STATUS_BYTES,
                RESOLVED_BYTES,
            );
            encoder.copy_buffer_to_buffer(
                &comparison_status,
                0,
                &staging,
                GPU_NATIVE_STATUS_BYTES + RESOLVED_BYTES,
                GPU_NATIVE_STATUS_BYTES,
            );
            gpu.queue.submit(Some(encoder.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            gpu.device.poll(wgpu::Maintain::Wait);
            rx.recv()
                .expect("mutable residency map callback must be drained")
                .expect("mutable residency staging must map");
            let mapped = slice.get_mapped_range();
            let status = u32::from_le_bytes(
                mapped[..GPU_NATIVE_STATUS_BYTES as usize]
                    .try_into()
                    .unwrap(),
            );
            let resolved_start = GPU_NATIVE_STATUS_BYTES as usize;
            let resolved = GpuNativeQ4ExpertMappingEntry {
                location: u32::from_le_bytes(
                    mapped[resolved_start..resolved_start + 4]
                        .try_into()
                        .unwrap(),
                ),
                slot_epoch: u32::from_le_bytes(
                    mapped[resolved_start + 4..resolved_start + 8]
                        .try_into()
                        .unwrap(),
                ),
            };
            let comparison_start = resolved_start + RESOLVED_BYTES as usize;
            let comparison = u32::from_le_bytes(
                mapped[comparison_start..comparison_start + GPU_NATIVE_STATUS_BYTES as usize]
                    .try_into()
                    .unwrap(),
            );
            drop(mapped);
            staging.unmap();
            (status, resolved, comparison)
        };

        let mut hidden_a = vec![0.0; D_MODEL];
        hidden_a[0] = 1.0;
        let residual_a = vec![0.25; D_MODEL];
        let expected_a = residual_add_mirror(
            &residual_a,
            &q4_expert_mirror(&payload_a, expert_geometry, &hidden_a),
        );
        let phase_a = run_case(
            "gpu_native_mutable_residency_phase_a",
            &hidden_a,
            &residual_a,
            &expected_a,
            5e-4,
        );
        assert_eq!(phase_a.2, 0);
        assert_eq!(phase_a.0, 0);
        assert_eq!(phase_a.1, residency_a.mapping_entry().unwrap());

        assert_eq!(
            executor.retire_q4_expert_residency(&arena, key_a).unwrap(),
            GpuNativeQ4ExpertRetire::Retired
        );
        let permit_b = match executor.acquire_q4_expert_residency(&arena, key_b).unwrap() {
            GpuNativeQ4ExpertAcquire::Install(permit) => permit,
            _ => panic!("retired one-slot arena must reserve expert B"),
        };
        let residency_b = executor
            .install_q4_expert_residency(permit_b, &payload_b)
            .unwrap();
        assert_eq!(residency_b.location(), residency_a.location());
        assert_ne!(residency_b.slot_epoch(), residency_a.slot_epoch());

        let mut hidden_b = vec![0.0; D_MODEL];
        hidden_b[1] = 1.0;
        let residual_b = vec![-0.125; D_MODEL];
        let expected_b = residual_add_mirror(
            &residual_b,
            &q4_expert_mirror(&payload_b, expert_geometry, &hidden_b),
        );
        let phase_b = run_case(
            "gpu_native_mutable_residency_phase_b",
            &hidden_b,
            &residual_b,
            &expected_b,
            5e-4,
        );
        assert_eq!(phase_b.2, 0);
        assert_eq!(phase_b.0, 0);
        assert_eq!(phase_b.1, residency_b.mapping_entry().unwrap());

        let miss_residual = vec![0.5; D_MODEL];
        let phase_c = run_case(
            "gpu_native_mutable_residency_phase_c",
            &hidden_a,
            &miss_residual,
            &miss_residual,
            0.0,
        );
        assert_eq!(phase_c.2, 0);
        assert_eq!(
            phase_c.0 & GPU_NATIVE_STATUS_RETRYABLE_MASK,
            GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS
        );
        assert_eq!(phase_c.0 & GPU_NATIVE_STATUS_FATAL_MASK, 0);
        assert_eq!(phase_c.1, GpuNativeQ4ExpertMappingEntry::UNMAPPED);

        assert!(
            expected_b
                .iter()
                .zip(&miss_residual)
                .all(|(&successful, &contained)| successful.is_finite() && contained.is_finite()),
            "successful B and containment expectations must be finite"
        );
        assert!(
            expected_b
                .iter()
                .zip(&miss_residual)
                .any(|(&successful, &contained)| successful != contained),
            "successful B output must differ from stale-route containment"
        );

        // Test-only fault injection constructs the exact stale resolved-route
        // pair while the physical slot contains B's newer epoch. The normal
        // route resolver copies it; Q4 execution must reject it on device.
        gpu.queue.write_buffer(
            &arena.mapping,
            EXPERT_A as u64 * GPU_NATIVE_EXPERT_MAPPING_ENTRY_BYTES as u64,
            bytemuck::bytes_of(&residency_a.mapping_entry().unwrap()),
        );
        let phase_d = run_case(
            "gpu_native_mutable_residency_phase_d",
            &hidden_a,
            &miss_residual,
            &miss_residual,
            0.0,
        );
        assert_eq!(phase_d.2, 0);
        assert_eq!(phase_d.1, residency_a.mapping_entry().unwrap());
        assert_eq!(
            phase_d.0 & GPU_NATIVE_STATUS_RETRYABLE_MASK,
            GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS
        );
        assert_eq!(phase_d.0 & GPU_NATIVE_STATUS_FATAL_MASK, 0);

        let before_stale = arena.residency_snapshot();
        assert!(matches!(
            executor
                .acquire_q4_expert_residency(&arena, GpuNativeQ4ExpertKey::new(0, EXPERT_B, 19),)
                .unwrap(),
            GpuNativeQ4ExpertAcquire::StaleRequester
        ));
        assert!(matches!(
            executor.acquire_q4_expert_residency(&arena, key_b).unwrap(),
            GpuNativeQ4ExpertAcquire::Hit(hit) if hit == residency_b
        ));
        let after_stale = arena.residency_snapshot();
        assert_eq!(after_stale.resident_slots, before_stale.resident_slots);
        assert_eq!(after_stale.expert_slot_installs, 2);
        assert_eq!(after_stale.expert_slot_retires, 1);
        assert_eq!(after_stale.expert_slot_reuses, 1);
        assert_eq!(after_stale.expert_stale_install_rejections, 1);
        let completed = executor.execution_snapshot();
        assert_eq!(completed.cpu_router_calls, 0);
        assert_eq!(completed.cpu_expert_combines, 0);
        assert_eq!(completed.expert_slot_misses, 0);
        assert_eq!(completed.numerical_failures, 0);
        assert_eq!(completed.queue_submissions, 0);
        assert_eq!(completed.intermediate_maps, 0);
        assert_eq!(completed.intermediate_readbacks, 0);
    }

    /// Requires an actual NVIDIA L4. Exercises one model-wide two-layer
    /// budget, real logical generations, per-layer replacement, retryable
    /// miss containment, encoder-only retry clearing, and replay against the
    /// new slot epochs. Test-owned submissions and final readbacks are the
    /// only submit/map operations in this fixture.
    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_tiered_residency_retry() {
        use super::super::{
            resolve_execution_context, ComputeOffload, GpuBackendGeometry, RoutedExpertGpuSpec,
        };
        use crate::buffer_pool::BufferPool;
        use crate::expert_cache::{ExpertResident, GpuResident};
        use crate::gpu_native_residency::{
            GpuNativeDemandExpert, GpuNativeModelExpertVramPlan, GpuNativeResidencyPriority,
            GpuNativeTieredResidencyError, GpuNativeTieredResidencyManager,
        };
        use crate::inference::WeightDtype;

        const D_MODEL: usize = 32;
        const D_FF: usize = 32;
        const NUM_EXPERTS: usize = 128;
        const TOP_K: usize = 2;
        const NUM_LAYERS: usize = 2;
        const READBACK_BYTES: u64 = GPU_NATIVE_STATUS_BYTES + GPU_NATIVE_STATUS_BYTES;

        let geometry =
            GpuNativeQ4ExpertGeometry::try_new(D_MODEL, D_FF, NUM_EXPERTS, TOP_K).unwrap();
        let payload_template = q4_uniform_expert(geometry, 0.01, 0.02, 0.005);
        let payload_bytes = payload_template.len();
        let gpu_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(
            payload_bytes * 8,
            0.0,
            u64::MAX,
        ));
        let execution = resolve_execution_context(
            ComputeOffload::Gpu,
            false,
            GpuBackendGeometry {
                num_layers: NUM_LAYERS,
                max_seq_len: 8,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: D_MODEL,
                v_head_dim: D_MODEL,
                q4_truncation_tolerance: 0,
            },
            RoutedExpertGpuSpec {
                dtype: WeightDtype::Q4_0,
                d_model: D_MODEL,
                d_ff: D_FF,
            },
            gpu_cache.clone(),
        )
        .expect("L4 must construct the authoritative production GPU backend");
        let executor = Arc::new(
            execution
                .create_gpu_native_executor_context(D_MODEL)
                .expect("GPU-native executor must retain the authoritative backend"),
        );
        assert_eq!(executor.device_identity().vendor_id, 0x10de);
        assert!(
            executor.device_identity().name.contains("L4"),
            "ignored tiered residency test must run only on an NVIDIA L4, got {}",
            executor.device_identity().name
        );
        let gpu = executor.authoritative_gpu().unwrap();
        let one_layer_min =
            GpuNativeQ4ExpertVramPlan::try_for_slot_capacity(geometry, TOP_K, &gpu.device.limits())
                .unwrap();
        let total_budget = one_layer_min.total_arena_allocation_bytes() * NUM_LAYERS as u64;
        let model_plan = GpuNativeModelExpertVramPlan::try_new(
            NUM_LAYERS,
            geometry,
            total_budget,
            &gpu.device.limits(),
        )
        .unwrap();
        assert_eq!(model_plan.total_arena_allocation_bytes(), total_budget);
        assert!(model_plan
            .layer_plans()
            .iter()
            .all(|plan| plan.slot_capacity() == TOP_K));
        assert!(model_plan
            .layer_plans()
            .iter()
            .all(|plan| plan.total_arena_allocation_bytes() < total_budget));

        let manager = GpuNativeTieredResidencyManager::try_new(
            executor.clone(),
            gpu_cache.clone(),
            NUM_LAYERS,
            geometry,
            total_budget,
        )
        .unwrap();
        let pool = BufferPool::new(16, payload_bytes, 4);
        let make_source = |global_id: u32, payload: &[u8]| {
            let mut buffer = pool.try_acquire().expect("synthetic RAM slot");
            buffer.as_mut_slice().copy_from_slice(payload);
            let resident = Arc::new(ExpertResident::new_with_block_align(global_id, buffer, 4));
            gpu_cache
                .demand_admit_lru(Arc::new(GpuResident::new_with_dtype(
                    global_id,
                    payload.to_vec(),
                    WeightDtype::Q4_0,
                )))
                .unwrap();
            let admission = gpu_cache.current_admission(global_id).unwrap();
            (resident, admission)
        };

        let layer1_payload_a = q4_uniform_expert(geometry, 0.004, 0.006, 0.002);
        let layer1_payload_b = q4_uniform_expert(geometry, -0.003, 0.005, 0.001);
        let (l1_a, l1_a_admission) = make_source(128, &layer1_payload_a);
        let (l1_b, l1_b_admission) = make_source(129, &layer1_payload_b);
        manager
            .ensure_demand_set(
                GpuNativeResidencyPriority::Demand,
                1,
                &[
                    GpuNativeDemandExpert::install(128, l1_a, l1_a_admission),
                    GpuNativeDemandExpert::install(129, l1_b, l1_b_admission),
                ],
            )
            .unwrap();

        let old_payload_a = q4_uniform_expert(geometry, 0.002, 0.003, 0.001);
        let old_payload_b = q4_uniform_expert(geometry, -0.002, 0.004, 0.001);
        let (old_a, old_a_admission) = make_source(0, &old_payload_a);
        let (old_b, old_b_admission) = make_source(1, &old_payload_b);
        let old_residencies = manager
            .ensure_demand_set(
                GpuNativeResidencyPriority::Demand,
                0,
                &[
                    GpuNativeDemandExpert::install(0, old_a, old_a_admission),
                    GpuNativeDemandExpert::install(1, old_b, old_b_admission),
                ],
            )
            .unwrap();
        let before_replace = manager.snapshot();
        assert_eq!(before_replace.layers[0].resident_slots, TOP_K);
        assert_eq!(before_replace.layers[1].resident_slots, TOP_K);

        let mut gate_values = vec![0.0; NUM_EXPERTS * D_MODEL];
        gate_values[2 * D_MODEL] = 4.0;
        gate_values[3 * D_MODEL + 1] = 3.0;
        let gate = DenseWeight::from_f32(gate_values, NUM_EXPERTS, D_MODEL);
        let gate_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.tiered_residency.router").unwrap(),
                &gate,
            )
            .unwrap();
        let router_geometry =
            GpuNativeRouterGeometry::try_new(D_MODEL, NUM_EXPERTS, TOP_K).unwrap();
        let router_plan = executor
            .create_router_plan(0, router_geometry, gate_handle)
            .unwrap();
        let (compare_layout, compare_pipeline) = create_test_compare_pipeline(&gpu.device);
        let state = executor.create_token_state().unwrap();
        let router_scratch = executor.create_router_scratch(router_geometry).unwrap();
        let expert_scratch = executor.create_q4_expert_scratch(geometry).unwrap();
        let hidden = vec![1.0; D_MODEL];
        let residual = vec![0.25; D_MODEL];
        let run = |label: &str, clear_retryable: bool, expected_hidden: &[f32], tolerance: f32| {
            gpu.queue
                .write_buffer(&state.hidden, 0, bytemuck::cast_slice(&hidden));
            gpu.queue
                .write_buffer(&state.residual, 0, bytemuck::cast_slice(&residual));
            let expected_buffer = create_test_expected_buffer(
                &gpu.device,
                &gpu.queue,
                &format!("{label}_expected_hidden"),
                expected_hidden,
            );
            let comparison_status = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label}_comparison_status")),
                size: GPU_NATIVE_STATUS_BYTES,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(&comparison_status, 0, &[0; 4]);
            let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: READBACK_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            if clear_retryable {
                executor
                    .encode_clear_retryable_status(&mut encoder, &state)
                    .unwrap();
            }
            executor
                .encode_router(&mut encoder, &router_plan, &state, &router_scratch)
                .unwrap();
            executor
                .encode_q4_expert_arena_combine(
                    &mut encoder,
                    &router_plan,
                    &router_scratch,
                    manager.arena(0).unwrap(),
                    &state,
                    &expert_scratch,
                )
                .unwrap();
            encode_test_compare(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &state.hidden,
                &expected_buffer,
                &comparison_status,
                D_MODEL,
                tolerance,
            );
            encoder.copy_buffer_to_buffer(&state.status, 0, &staging, 0, GPU_NATIVE_STATUS_BYTES);
            encoder.copy_buffer_to_buffer(
                &comparison_status,
                0,
                &staging,
                GPU_NATIVE_STATUS_BYTES,
                GPU_NATIVE_STATUS_BYTES,
            );
            gpu.queue.submit(Some(encoder.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::sync_channel(1);
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            gpu.device.poll(wgpu::Maintain::Wait);
            rx.recv()
                .expect("tiered residency map callback must be drained")
                .expect("tiered residency staging must map");
            let mapped = slice.get_mapped_range();
            let request_status = u32::from_le_bytes(
                mapped[..GPU_NATIVE_STATUS_BYTES as usize]
                    .try_into()
                    .unwrap(),
            );
            let comparison_status = u32::from_le_bytes(
                mapped[GPU_NATIVE_STATUS_BYTES as usize
                    ..GPU_NATIVE_STATUS_BYTES as usize + GPU_NATIVE_STATUS_BYTES as usize]
                    .try_into()
                    .unwrap(),
            );
            drop(mapped);
            staging.unmap();
            (request_status, comparison_status)
        };

        gpu.queue.write_buffer(&state.status, 0, &[0; 4]);
        let (miss_request_status, miss_comparison_status) =
            run("gpu_native_tiered_residency_miss", false, &residual, 0.0);
        assert_eq!(miss_comparison_status, 0);
        assert_eq!(
            miss_request_status & GPU_NATIVE_STATUS_RETRYABLE_MASK,
            GPU_NATIVE_STATUS_RETRYABLE_MASK
        );
        assert_eq!(miss_request_status & GPU_NATIVE_STATUS_FATAL_MASK, 0);

        let payload_a = q4_uniform_expert(geometry, 0.01, 0.02, 0.005);
        let payload_b = q4_uniform_expert(geometry, -0.012, 0.018, 0.004);
        let (resident_a, admission_a) = make_source(2, &payload_a);
        let stale_admission_a = admission_a.clone();
        let stale_resident_a = resident_a.clone();
        let (resident_b, admission_b) = make_source(3, &payload_b);
        let new_residencies = manager
            .ensure_demand_set(
                GpuNativeResidencyPriority::Demand,
                0,
                &[
                    GpuNativeDemandExpert::install(2, resident_a, admission_a),
                    GpuNativeDemandExpert::install(3, resident_b, admission_b),
                ],
            )
            .unwrap();
        assert!(new_residencies.iter().all(|new| old_residencies
            .iter()
            .filter(|old| old.location() == new.location())
            .all(|old| old.slot_epoch() != new.slot_epoch())));
        let after_replace = manager.snapshot();
        assert_eq!(after_replace.layers[0].physical_evictions, 2);
        assert_eq!(after_replace.layers[1].resident_slots, TOP_K);
        assert!(manager.has_current_for_demand(128).unwrap());
        assert!(manager.has_current_for_demand(129).unwrap());

        let logits = gate.matvec(&hidden);
        let (selected_ids, selected_weights, failed) = router_topk_mirror(&logits, TOP_K);
        assert!(!failed);
        assert_eq!(selected_ids, vec![2, 3]);
        let expert_outputs = [
            q4_expert_mirror(&payload_a, geometry, &hidden),
            q4_expert_mirror(&payload_b, geometry, &hidden),
        ];
        let expected = residual_add_mirror(
            &residual,
            &weighted_expert_combine_mirror(&expert_outputs, &selected_weights),
        );
        assert!(expected.iter().all(|value| value.is_finite()));
        assert!(expected
            .iter()
            .zip(&residual)
            .any(|(expected, residual)| expected != residual));

        let (replay_request_status, replay_comparison_status) =
            run("gpu_native_tiered_residency_replay", true, &expected, 5e-4);
        assert_eq!(replay_request_status, 0);
        assert_eq!(replay_comparison_status, 0);

        // Evict and re-admit logical id 2 to obtain a strictly newer real
        // generation, then prove the saved stale requester cannot disturb it.
        for id in 4..11u32 {
            let payload = q4_uniform_expert(geometry, 0.001 * id as f32, 0.002, 0.001);
            let _ = make_source(id, &payload);
        }
        let (newer_resident_a, newer_admission_a) = make_source(2, &payload_a);
        assert!(newer_admission_a.generation() > stale_admission_a.generation());
        let newest = manager
            .ensure_demand_set(
                GpuNativeResidencyPriority::Demand,
                0,
                &[GpuNativeDemandExpert::install(
                    2,
                    newer_resident_a,
                    newer_admission_a,
                )],
            )
            .unwrap()[0];
        let before_stale = manager.snapshot();
        assert!(matches!(
            manager.ensure_demand_set(
                GpuNativeResidencyPriority::Demand,
                0,
                &[GpuNativeDemandExpert::install(
                    2,
                    stale_resident_a,
                    stale_admission_a,
                )],
            ),
            Err(GpuNativeTieredResidencyError::LogicalAdmissionStale { .. })
        ));
        let after_stale = manager.snapshot();
        assert_eq!(after_stale.layers, before_stale.layers);
        assert_eq!(
            after_stale.stale_generation_rejections,
            before_stale.stale_generation_rejections + 1
        );
        assert!(manager.has_current_for_demand(2).unwrap());
        assert_eq!(
            manager
                .ensure_demand_set(
                    GpuNativeResidencyPriority::Demand,
                    0,
                    &[GpuNativeDemandExpert::current(2)],
                )
                .unwrap()[0],
            newest
        );
    }

    /// Requires an actual NVIDIA L4. Three isolated requests prove a finite
    /// eight-of-128 arena route, an unmapped demand miss with no partial
    /// mixture, and prior-router-failure containment. All commands share one
    /// caller-owned encoder, one submission, and one final test-only map.
    #[test]
    #[ignore = "requires authoritative NVIDIA L4 WGPU validation hardware"]
    fn live_l4_gpu_native_q4_expert_arena_combine() {
        use super::super::{
            resolve_execution_context, ComputeOffload, GpuBackendGeometry, RoutedExpertGpuSpec,
        };
        use crate::inference::WeightDtype;

        const D_MODEL: usize = 32;
        const D_FF: usize = 32;
        const NUM_EXPERTS: usize = 128;
        const TOP_K: usize = 8;
        const IDS_BYTES: u64 = (TOP_K * std::mem::size_of::<u32>()) as u64;
        const WEIGHTS_BYTES: u64 = (TOP_K * std::mem::size_of::<f32>()) as u64;
        const RESOLVED_BYTES: u64 = (TOP_K * GPU_NATIVE_EXPERT_MAPPING_ENTRY_BYTES) as u64;
        const CASE_BYTES: u64 =
            IDS_BYTES + WEIGHTS_BYTES + RESOLVED_BYTES + GPU_NATIVE_STATUS_BYTES * 2;
        const CASES: usize = 3;

        let expert_cache = Arc::new(crate::expert_cache::GpuExpertCache::new(
            1024 * 1024,
            0.5,
            16,
        ));
        let execution = resolve_execution_context(
            ComputeOffload::Gpu,
            false,
            GpuBackendGeometry {
                num_layers: 1,
                max_seq_len: 8,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: D_MODEL,
                v_head_dim: D_MODEL,
                q4_truncation_tolerance: 0,
            },
            RoutedExpertGpuSpec {
                dtype: WeightDtype::Q4_0,
                d_model: D_MODEL,
                d_ff: D_FF,
            },
            expert_cache,
        )
        .expect("L4 must construct the authoritative production GPU backend");
        let executor = execution
            .create_gpu_native_executor_context(D_MODEL)
            .expect("GPU-native executor must retain the authoritative backend");
        assert_eq!(executor.device_identity().vendor_id, 0x10de);
        assert!(
            executor.device_identity().name.contains("L4"),
            "ignored Q4 expert arena test must run only on an NVIDIA L4, got {}",
            executor.device_identity().name
        );
        let gpu = executor.authoritative_gpu().unwrap();
        let router_geometry =
            GpuNativeRouterGeometry::try_new(D_MODEL, NUM_EXPERTS, TOP_K).unwrap();
        let expert_geometry =
            GpuNativeQ4ExpertGeometry::try_new(D_MODEL, D_FF, NUM_EXPERTS, TOP_K).unwrap();

        let mut gate_values = vec![0.0; NUM_EXPERTS * D_MODEL];
        for expert in 0..NUM_EXPERTS {
            gate_values[expert * D_MODEL] = expert as f32 / 32.0;
        }
        let gate = DenseWeight::from_f32(gate_values.clone(), NUM_EXPERTS, D_MODEL);
        let gate_handle = executor
            .register_dense_weight(
                GpuNativeDenseWeightKey::try_new("test.q4_expert.router").unwrap(),
                &gate,
            )
            .unwrap();
        let router_plan = executor
            .create_router_plan(0, router_geometry, gate_handle)
            .unwrap();

        let finite_hidden = vec![1.0; D_MODEL];
        let finite_logits = gate.matvec(&finite_hidden);
        let (selected_ids, selected_weights, router_failed) =
            router_topk_mirror(&finite_logits, TOP_K);
        assert!(!router_failed);
        assert_eq!(selected_ids, (120..128).rev().collect::<Vec<_>>());

        let payloads = selected_ids
            .iter()
            .enumerate()
            .map(|(slot, &logical_id)| {
                (
                    logical_id,
                    q4_uniform_expert(
                        expert_geometry,
                        0.001 * (slot as f32 + 1.0),
                        0.0015 * (slot as f32 + 1.0),
                        0.0005 * (slot as f32 + 1.0),
                    ),
                )
            })
            .collect::<Vec<_>>();
        let full_uploads = payloads
            .iter()
            .enumerate()
            .map(|(slot, (logical_id, payload))| GpuNativeQ4ExpertUpload {
                logical_id: *logical_id,
                logical_generation: 1,
                location: GpuNativeQ4ExpertLocation::try_new(0, slot as u32).unwrap(),
                payload,
            })
            .collect::<Vec<_>>();
        let missing_uploads = payloads
            .iter()
            .enumerate()
            .skip(1)
            .map(|(slot, (logical_id, payload))| GpuNativeQ4ExpertUpload {
                logical_id: *logical_id,
                logical_generation: 1,
                location: GpuNativeQ4ExpertLocation::try_new(0, slot as u32).unwrap(),
                payload,
            })
            .collect::<Vec<_>>();
        let arena_budget =
            GpuNativeQ4ExpertArenaLayout::try_new(expert_geometry, TOP_K, &gpu.device.limits())
                .unwrap()
                .total_allocation_bytes()
                .unwrap();
        let arena_plan =
            GpuNativeQ4ExpertVramPlan::try_new(expert_geometry, arena_budget, &gpu.device.limits())
                .unwrap();
        let full_arena = executor
            .create_q4_expert_arena(0, arena_plan, &full_uploads)
            .unwrap();
        let missing_arena = executor
            .create_q4_expert_arena(0, arena_plan, &missing_uploads)
            .unwrap();
        assert_eq!(full_arena.slot_capacity(), TOP_K);
        assert_eq!(full_arena.resident_experts(), TOP_K);
        assert_eq!(missing_arena.resident_experts(), TOP_K - 1);
        assert_eq!(full_arena.geometry(), expert_geometry);
        assert_eq!(full_arena.layer_index(), 0);
        assert!(full_arena.active_banks() <= MAX_GPU_NATIVE_EXPERT_BANKS);

        let expert_outputs = payloads
            .iter()
            .map(|(_, payload)| q4_expert_mirror(payload, expert_geometry, &finite_hidden))
            .collect::<Vec<_>>();
        let combined = weighted_expert_combine_mirror(&expert_outputs, &selected_weights);
        let finite_residual = vec![0.25; D_MODEL];
        let missing_residual = vec![-0.125; D_MODEL];
        let prior_failure_residual = vec![0.5; D_MODEL];
        let finite_expected = residual_add_mirror(&finite_residual, &combined);
        let expected_hidden = [
            finite_expected,
            missing_residual.clone(),
            prior_failure_residual.clone(),
        ];

        let states = [
            executor.create_token_state().unwrap(),
            executor.create_token_state().unwrap(),
            executor.create_token_state().unwrap(),
        ];
        let router_scratches = [
            executor.create_router_scratch(router_geometry).unwrap(),
            executor.create_router_scratch(router_geometry).unwrap(),
            executor.create_router_scratch(router_geometry).unwrap(),
        ];
        let expert_scratches = [
            executor.create_q4_expert_scratch(expert_geometry).unwrap(),
            executor.create_q4_expert_scratch(expert_geometry).unwrap(),
            executor.create_q4_expert_scratch(expert_geometry).unwrap(),
        ];
        let invalid_hidden = vec![f32::NAN; D_MODEL];
        for (state, (hidden, residual)) in states.iter().zip([
            (&finite_hidden, &finite_residual),
            (&finite_hidden, &missing_residual),
            (&invalid_hidden, &prior_failure_residual),
        ]) {
            gpu.queue
                .write_buffer(&state.hidden, 0, bytemuck::cast_slice(hidden));
            gpu.queue
                .write_buffer(&state.residual, 0, bytemuck::cast_slice(residual));
            gpu.queue.write_buffer(&state.status, 0, &[0; 4]);
        }

        let comparison_statuses = std::array::from_fn::<_, CASES, _>(|index| {
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("gpu_native_q4_expert_compare_status_{index}")),
                size: GPU_NATIVE_STATUS_BYTES,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(&buffer, 0, &[0; 4]);
            buffer
        });
        let expected_buffers = std::array::from_fn::<_, CASES, _>(|index| {
            create_test_expected_buffer(
                &gpu.device,
                &gpu.queue,
                &format!("gpu_native_q4_expert_expected_{index}"),
                &expected_hidden[index],
            )
        });
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_native_q4_expert_validation_staging"),
            size: CASE_BYTES * CASES as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (compare_layout, compare_pipeline) = create_test_compare_pipeline(&gpu.device);
        let before = executor.execution_snapshot();
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_native_q4_expert_live_l4_encoder"),
            });
        for index in 0..CASES {
            executor
                .encode_router(
                    &mut encoder,
                    &router_plan,
                    &states[index],
                    &router_scratches[index],
                )
                .unwrap();
            executor
                .encode_q4_expert_arena_combine(
                    &mut encoder,
                    &router_plan,
                    &router_scratches[index],
                    if index == 1 {
                        &missing_arena
                    } else {
                        &full_arena
                    },
                    &states[index],
                    &expert_scratches[index],
                )
                .unwrap();
            encode_test_compare(
                &gpu.device,
                &mut encoder,
                &compare_layout,
                &compare_pipeline,
                &states[index].hidden,
                &expected_buffers[index],
                &comparison_statuses[index],
                D_MODEL,
                5e-4,
            );
            let offset = index as u64 * CASE_BYTES;
            encoder.copy_buffer_to_buffer(
                &router_scratches[index].selected_ids,
                0,
                &staging,
                offset,
                IDS_BYTES,
            );
            encoder.copy_buffer_to_buffer(
                &router_scratches[index].selected_weights,
                0,
                &staging,
                offset + IDS_BYTES,
                WEIGHTS_BYTES,
            );
            encoder.copy_buffer_to_buffer(
                &expert_scratches[index].resolved_locations,
                0,
                &staging,
                offset + IDS_BYTES + WEIGHTS_BYTES,
                RESOLVED_BYTES,
            );
            encoder.copy_buffer_to_buffer(
                &states[index].status,
                0,
                &staging,
                offset + IDS_BYTES + WEIGHTS_BYTES + RESOLVED_BYTES,
                GPU_NATIVE_STATUS_BYTES,
            );
            encoder.copy_buffer_to_buffer(
                &comparison_statuses[index],
                0,
                &staging,
                offset + IDS_BYTES + WEIGHTS_BYTES + RESOLVED_BYTES + GPU_NATIVE_STATUS_BYTES,
                GPU_NATIVE_STATUS_BYTES,
            );
        }
        let encoded = executor.execution_snapshot();
        assert_eq!(
            encoded.expert_slots_registered - before.expert_slots_registered,
            0
        );
        assert_eq!(
            encoded.expert_weight_upload_bytes - before.expert_weight_upload_bytes,
            0
        );
        assert_eq!(
            encoded.expert_route_resolve_dispatches - before.expert_route_resolve_dispatches,
            3
        );
        assert_eq!(
            encoded.q4_expert_gate_up_dispatches - before.q4_expert_gate_up_dispatches,
            24
        );
        assert_eq!(
            encoded.q4_expert_down_dispatches - before.q4_expert_down_dispatches,
            24
        );
        assert_eq!(
            encoded.expert_combine_dispatches - before.expert_combine_dispatches,
            3
        );
        assert_eq!(encoded.cpu_expert_combines, 0);
        assert_eq!(encoded.expert_slot_misses, 0);
        assert_eq!(encoded.queue_submissions, 0);
        assert_eq!(encoded.intermediate_maps, 0);
        assert_eq!(encoded.intermediate_readbacks, 0);

        gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("validation map callback must be drained")
            .expect("Q4 expert validation staging must map");
        let mapped = slice.get_mapped_range();
        let parse_u32s = |start: usize, bytes: usize| {
            mapped[start..start + bytes]
                .chunks_exact(4)
                .map(|value| u32::from_le_bytes(value.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let mut cases = Vec::with_capacity(CASES);
        for index in 0..CASES {
            let start = index * CASE_BYTES as usize;
            let ids = parse_u32s(start, IDS_BYTES as usize);
            let weights_start = start + IDS_BYTES as usize;
            let weights = mapped[weights_start..weights_start + WEIGHTS_BYTES as usize]
                .chunks_exact(4)
                .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
                .collect::<Vec<_>>();
            let resolved_start = weights_start + WEIGHTS_BYTES as usize;
            let resolved = parse_u32s(resolved_start, RESOLVED_BYTES as usize);
            let status_start = resolved_start + RESOLVED_BYTES as usize;
            let status = parse_u32s(status_start, GPU_NATIVE_STATUS_BYTES as usize)[0];
            let comparison = parse_u32s(
                status_start + GPU_NATIVE_STATUS_BYTES as usize,
                GPU_NATIVE_STATUS_BYTES as usize,
            )[0];
            cases.push((ids, weights, resolved, status, comparison));
        }
        drop(mapped);
        staging.unmap();

        assert_eq!(cases[0].0, selected_ids);
        assert_close(&cases[0].1, &selected_weights, 2e-5);
        assert!(cases[0]
            .2
            .chunks_exact(2)
            .all(|route| { route[0] != GPU_NATIVE_EXPERT_UNMAPPED && route[1] != 0 }));
        assert_eq!(cases[0].3, 0);
        assert_eq!(cases[0].4, 0);

        assert_eq!(cases[1].0, selected_ids);
        assert_eq!(&cases[1].2[..2], &[GPU_NATIVE_EXPERT_UNMAPPED, 0]);
        assert_eq!(
            cases[1].3 & GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS,
            GPU_NATIVE_STATUS_EXPERT_RESIDENCY_MISS
        );
        assert_eq!(cases[1].4, 0, "miss containment hidden must equal residual");

        assert_eq!(cases[2].0, vec![0; TOP_K]);
        assert_eq!(cases[2].1, vec![0.0; TOP_K]);
        assert_eq!(cases[2].2, [GPU_NATIVE_EXPERT_UNMAPPED, 0].repeat(TOP_K));
        assert_eq!(
            cases[2].3, GPU_NATIVE_STATUS_ROUTER_NUMERICAL_FAILURE,
            "prior router failure must not invent an expert miss"
        );
        assert_eq!(cases[2].4, 0);

        let completed = executor.execution_snapshot();
        assert_eq!(completed.cpu_router_calls, 0);
        assert_eq!(completed.cpu_expert_combines, 0);
        assert_eq!(completed.expert_slot_misses, 0);
        assert_eq!(completed.numerical_failures, 0);
        assert_eq!(completed.intermediate_maps, 0);
        assert_eq!(completed.intermediate_readbacks, 0);
    }

    #[test]
    fn checked_layout_derives_qwen_vector_bytes() {
        let layout = GpuNativeTokenStateLayout::try_new(2048).expect("Qwen layout");
        assert_eq!(layout.d_model(), 2048);
        assert_eq!(layout.vector_bytes(), 8192);
        assert_eq!(layout.status_bytes(), 4);
        assert_eq!(layout.total_buffer_bytes(), 16_388);
    }

    #[test]
    fn checked_layout_rejects_zero_and_overflow() {
        assert_eq!(
            GpuNativeTokenStateLayout::try_new(0),
            Err(GpuNativeBootstrapError::InvalidDModel)
        );
        let overflowing_d_model = usize::MAX / std::mem::size_of::<f32>() + 1;
        assert_eq!(
            GpuNativeTokenStateLayout::try_new(overflowing_d_model),
            Err(GpuNativeBootstrapError::StateSizeOverflow {
                d_model: overflowing_d_model,
            })
        );
        let total_overflowing_d_model = usize::MAX / std::mem::size_of::<f32>();
        assert_eq!(
            GpuNativeTokenStateLayout::try_new(total_overflowing_d_model),
            Err(GpuNativeBootstrapError::StateSizeOverflow {
                d_model: total_overflowing_d_model,
            })
        );
    }

    #[test]
    fn layout_rejects_device_limit_incompatibility_before_allocation() {
        let layout = GpuNativeTokenStateLayout::try_new(2048).unwrap();
        let limits = wgpu::Limits {
            max_buffer_size: 4096,
            max_storage_buffer_binding_size: 4096,
            ..wgpu::Limits::default()
        };
        assert_eq!(
            layout.validate_for_limits(&limits),
            Err(GpuNativeBootstrapError::Allocation(
                GpuStartupAllocationError::ExceedsMaxBufferSize {
                    label: "gpu_native_hidden".to_string(),
                    requested: 8192,
                    maximum: 4096,
                }
            ))
        );
    }

    #[test]
    fn intermediate_tensor_buffers_are_not_cpu_mappable() {
        let tensor = GpuNativeTokenStateLayout::tensor_usage();
        assert!(tensor.contains(wgpu::BufferUsages::STORAGE));
        assert!(tensor.contains(wgpu::BufferUsages::COPY_DST));
        assert!(tensor.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!tensor.intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));

        let status = GpuNativeTokenStateLayout::status_usage();
        assert!(status.contains(wgpu::BufferUsages::STORAGE));
        assert!(status.contains(wgpu::BufferUsages::COPY_DST));
        assert!(status.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!status.intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));

        let generic_scratch = GpuNativeScratchLayout::usage();
        assert!(generic_scratch.contains(wgpu::BufferUsages::STORAGE));
        assert!(generic_scratch.contains(wgpu::BufferUsages::COPY_DST));
        assert!(generic_scratch.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!generic_scratch
            .intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));

        let boundary_result = GpuNativeScratchLayout::boundary_result_usage();
        assert!(boundary_result.contains(wgpu::BufferUsages::STORAGE));
        assert!(boundary_result.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(!boundary_result.contains(wgpu::BufferUsages::COPY_DST));
        assert!(!boundary_result
            .intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));

        let router_result = GpuNativeRouterScratchLayout::result_usage();
        assert!(router_result.contains(wgpu::BufferUsages::STORAGE));
        assert!(router_result.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(
            !router_result.intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE)
        );

        for usage in [
            GpuNativeDenseWeightLayout::usage(),
            GpuNativeKvLayout::usage(),
        ] {
            assert!(usage.contains(wgpu::BufferUsages::STORAGE));
            assert!(!usage.contains(wgpu::BufferUsages::COPY_SRC));
            assert!(!usage.intersects(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE));
        }
    }

    #[test]
    fn scratch_layout_is_checked_and_variable_width() {
        let q = GpuNativeScratchLayout::try_new(4096).unwrap();
        let kv = GpuNativeScratchLayout::try_new(512).unwrap();
        let router = GpuNativeScratchLayout::try_new(128).unwrap();
        assert_eq!((q.elements(), q.bytes()), (4096, 16_384));
        assert_eq!((kv.elements(), kv.bytes()), (512, 2048));
        assert_eq!((router.elements(), router.bytes()), (128, 512));
        assert_eq!(
            GpuNativeScratchLayout::try_new(0),
            Err(GpuNativeBootstrapError::InvalidScratchElements)
        );
    }

    #[derive(Debug)]
    struct DropProbe {
        allocation_id: u64,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn test_state(
        layout: GpuNativeTokenStateLayout,
        allocation_base: u64,
        drops: Arc<AtomicUsize>,
    ) -> GpuNativeTokenState<DropProbe> {
        GpuNativeTokenState::from_buffers(
            1,
            next_gpu_native_token_state_id(),
            layout,
            DropProbe {
                allocation_id: allocation_base,
                drops: drops.clone(),
            },
            DropProbe {
                allocation_id: allocation_base + 1,
                drops: drops.clone(),
            },
            DropProbe {
                allocation_id: allocation_base + 2,
                drops,
            },
        )
    }

    #[test]
    fn token_states_own_distinct_mutable_buffers_and_state_ids() {
        let layout = GpuNativeTokenStateLayout::try_new(32).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let first = test_state(layout, 10, drops.clone());
        let second = test_state(layout, 20, drops.clone());

        assert_ne!(first.state_id(), second.state_id());
        assert_eq!(first.layout(), second.layout());
        assert_ne!(first.hidden.allocation_id, second.hidden.allocation_id);
        assert_ne!(first.residual.allocation_id, second.residual.allocation_id);
        assert_ne!(first.status.allocation_id, second.status.allocation_id);
        drop(first);
        drop(second);
        assert_eq!(drops.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn token_state_cleanup_drops_each_owned_buffer_exactly_once() {
        let layout = GpuNativeTokenStateLayout::try_new(32).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let state = test_state(layout, 1, drops.clone());

        assert_eq!(drops.load(Ordering::Relaxed), 0);
        drop(state);
        assert_eq!(drops.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn request_local_kv_drops_each_per_layer_buffer_exactly_once() {
        let layout = GpuNativeKvLayout::try_new(2, 4, 8, &wgpu::Limits::default()).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let probe = |allocation_id| DropProbe {
            allocation_id,
            drops: drops.clone(),
        };
        let kv = GpuNativeKvState::from_layers(
            7,
            1,
            layout,
            vec![
                GpuNativeKvLayer {
                    key: probe(10),
                    value: probe(11),
                },
                GpuNativeKvLayer {
                    key: probe(20),
                    value: probe(21),
                },
            ],
        );
        assert_eq!(kv.layout(), layout);
        assert_ne!(
            kv.layers[0].key.allocation_id,
            kv.layers[0].value.allocation_id
        );
        assert_ne!(
            kv.layers[0].key.allocation_id,
            kv.layers[1].key.allocation_id
        );
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        drop(kv);
        assert_eq!(drops.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn request_state_ownership_is_separate_from_model_weight_registry() {
        let layout = GpuNativeTokenStateLayout::try_new(32).unwrap();
        let state_drops = Arc::new(AtomicUsize::new(0));
        let weight_drops = Arc::new(AtomicUsize::new(0));
        let state = test_state(layout, 1, state_drops.clone());
        let weight_layout =
            GpuNativeDenseWeightLayout::try_new(GpuNativeDenseWeightKind::F32, 2, 2, 16).unwrap();
        let mut registry = GpuNativeDenseWeightRegistry::new(1);
        registry
            .insert(test_weight(
                1,
                "model.weight",
                weight_layout,
                DropProbe {
                    allocation_id: 100,
                    drops: weight_drops.clone(),
                },
            ))
            .unwrap();

        drop(state);
        assert_eq!(state_drops.load(Ordering::Relaxed), 3);
        assert_eq!(weight_drops.load(Ordering::Relaxed), 0);
        drop(registry);
        assert_eq!(weight_drops.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn initial_execution_snapshot_is_all_zero() {
        let counters = GpuNativeExecutionCounters::default();
        assert_eq!(counters.snapshot(), GpuNativeExecutionSnapshot::default());
    }

    #[test]
    fn cpu_execution_context_cannot_construct_gpu_native_executor() {
        let context = super::super::cpu_execution_context();
        assert!(matches!(
            context.create_gpu_native_executor_context(2048),
            Err(GpuNativeBootstrapError::GpuBackendUnavailable)
        ));
    }

    #[test]
    fn existing_operator_and_backend_modes_remain_distinct_from_bootstrap() {
        use super::super::{ComputeOffload, GpuBackendMode};

        let operator_mode_name = |mode| match mode {
            ComputeOffload::Cpu => "cpu",
            ComputeOffload::Gpu => "gpu",
            ComputeOffload::Auto => "auto",
            ComputeOffload::Hybrid => "hybrid",
        };
        assert_eq!(
            [
                ComputeOffload::Cpu,
                ComputeOffload::Gpu,
                ComputeOffload::Auto,
                ComputeOffload::Hybrid,
            ]
            .map(operator_mode_name),
            ["cpu", "gpu", "auto", "hybrid"]
        );

        let resource_mode_name = |mode| match mode {
            GpuBackendMode::RoutedExpertsOnly => "routed-experts-only",
            GpuBackendMode::Full => "legacy-full-resources",
        };
        assert_eq!(
            resource_mode_name(GpuBackendMode::Full),
            "legacy-full-resources"
        );
    }

    #[test]
    fn hardware_independent_test_backend_cannot_fake_gpu_native_executor() {
        let backend = Arc::new(BackendBox::TestGpu(super::super::TestGpuBackend::success(
            1.0,
        )));
        assert!(matches!(
            GpuNativeExecutorContext::try_new(backend, 2048),
            Err(GpuNativeBootstrapError::GpuBackendUnavailable)
        ));
    }

    #[test]
    fn snapshot_distinguishes_token_boundary_and_intermediate_readback() {
        let counters = GpuNativeExecutionCounters::default();
        counters.record_token_boundary_readback();
        let boundary = counters.snapshot();
        assert_eq!(boundary.token_boundary_readbacks, 1);
        assert_eq!(boundary.intermediate_readbacks, 0);

        counters.record_intermediate_readback();
        let intermediate = counters.snapshot();
        assert_eq!(intermediate.token_boundary_readbacks, 1);
        assert_eq!(intermediate.intermediate_readbacks, 1);
    }
}
