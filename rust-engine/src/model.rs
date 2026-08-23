//! Real Mixtral / Llama-style decoder-only transformer wired on top of
//! the SSD-streaming engine (gist Phase 5–6: "real transformer + MoE,
//! real generation loop, multi-layer support").
//!
//! What lives here:
//!
//! 1. [`RealModelConfig`] — the hyperparameters of the model
//!    (`d_model`, `d_ff`, `num_heads`, `vocab_size`, `num_layers`,
//!    `num_experts`, `top_k`, …).
//! 2. [`RealModel`] — owns the dense (resident) weights: token embedding,
//!    a stack of [`crate::transformer::TransformerLayer`]s, the final
//!    RMSNorm and the [`crate::transformer::LMHead`]. Expert FFN weights
//!    are **not** held here — they live on disk and are streamed by the
//!    engine on demand.
//! 3. [`RealModel::from_dir`] — loads weights from a directory of `.bin`
//!    files (one per tensor) and falls back to a deterministic seeded
//!    initialisation when files are missing, so the engine always has an
//!    end-to-end runnable path even without real model files.
//! 4. [`RealModel::forward_token_hidden`] / [`RealModel::sample_hidden`]
//!    — the split per-token API. Prompt ingestion runs embedding →
//!    stacked layers (each calling `attn_block`, `moe_pre`, awaiting
//!    the engine's SSD-streamed `moe_step`, then `moe_combine`) → final
//!    RMSNorm and returns the final hidden state. Decode then samples
//!    that hidden state through the LM head only when a next-token
//!    prediction is actually needed.
//!
//! Multi-layer expert addressing: when `num_layers > 1`, expert ids on
//! disk are encoded as `global_id = layer * num_experts_per_layer +
//! local_id`, so the existing single-namespace [`crate::expert_cache::ExpertCache`]
//! / [`crate::io_provider::NvmeStorage`] (which already use `u32`
//! ids and `expert_<id>.bin` paths) work unchanged. This keeps the run
//! summary statistics (hits, misses, I/O share) populated by the same
//! counters regardless of layer count. An alternative
//! [`crate::multi_layer_cache::MultiLayerExpertCache`] is also available
//! for users who want per-layer LRU isolation; the global-id flat scheme
//! is the default because it lets the existing engine instrumentation
//! and prefetcher keep working without per-layer sharding.

use crate::architecture::{
    Architecture, ComputeSupport, FfnKind, MlaDims, RopeScaling, TensorNaming,
};
use crate::dense_tensor::{dense_checksum, DenseDType, DenseTensorManifest, DenseWeight};
use crate::engine::Engine;
use crate::engine::MoeStepError;
use crate::gating::{LinearGate, ScoringFunc};
use crate::mla::MultiHeadLatentAttention;
use crate::transformer::{
    KvCache, LMHead, MultiHeadSelfAttention, RmsNorm, RopeCache, SharedExpert, TransformerLayer,
    YarnRope,
};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Advanced routing / attention parameters that only some architectures
/// use. Defaults reproduce Mixtral / Qwen3 behaviour exactly (softmax
/// routing, top-K renormalisation, no grouping, unit scaling, no MLA, no
/// RoPE scaling), so configs that don't set them are unaffected.
#[derive(Debug, Clone)]
pub struct AdvancedConfig {
    /// Router score function (DeepSeek-V3 uses `Sigmoid`).
    pub scoring_func: ScoringFunc,
    /// Re-normalise the selected top-K mixing weights (`norm_topk_prob`).
    pub norm_topk_prob: bool,
    /// Group-limited routing group count (`n_group`); `1` disables it.
    pub n_group: usize,
    /// Surviving groups in group-limited routing (`topk_group`).
    pub topk_group: usize,
    /// Final routed-weight multiplier (`routed_scaling_factor`).
    pub routed_scaling_factor: f32,
    /// Always-on shared experts per MoE layer (`n_shared_experts`).
    pub num_shared_experts: usize,
    /// Top-K selection method string (`topk_method`), e.g. `"noaux_tc"`.
    pub topk_method: Option<String>,
    /// MLA projection dims (DeepSeek-V3); `None` for every other family.
    pub mla: Option<MlaDims>,
    /// RoPE scaling (YaRN) parameters; `None` when absent.
    pub rope_scaling: Option<RopeScaling>,
    /// Hybrid-attention interleave ratio (number of consecutive
    /// Sliding-Window-Attention layers per Global layer). `None` ⇒ uniform
    /// attention (the legacy behaviour). Combined with the architecture's
    /// intrinsic ratio ([`Architecture::swa_global_ratio`]) to resolve each
    /// layer's [`crate::architecture::AttentionMode`] at construction time.
    pub swa_global_ratio: Option<usize>,
    /// Explicit per-layer hybrid-attention pattern (MiMo-V2-Flash
    /// `hybrid_layer_pattern`: `0 = global`, non-zero = SWA). When present
    /// it is used as a direct per-layer lookup and overrides
    /// `swa_global_ratio`. `None` ⇒ ratio-based resolution.
    pub hybrid_layer_pattern: Option<Vec<u8>>,
    /// Separate RoPE base for Sliding-Window-Attention layers
    /// (`swa_rope_theta`). MiMo-V2-Flash uses a smaller theta on SWA layers
    /// than on global layers (`rope_base`). `None` ⇒ every layer uses
    /// `rope_base`.
    pub swa_rope_theta: Option<f32>,
    /// FP8 block-quantisation tile size (`weight_block_size`). Drives the
    /// block edge used by the FP8 dequantiser; `None` ⇒ the default 128.
    pub fp8_block_size: Option<[usize; 2]>,
    /// GPT-OSS gate activation clamp (`swiglu_limit`). When `Some(limit)`,
    /// the SwiGLU gate is clamped to `[-limit, limit]` before the sigmoid.
    /// `None` (every other architecture) means no clamping.
    pub swiglu_limit: Option<f32>,
    /// Whether the attention Q/K/V/O projections carry additive biases
    /// (`attention_bias`, GPT-OSS). `false` for every other architecture.
    pub attention_bias: bool,
    /// V head dimension override (`v_head_dim`) for standard (non-MLA)
    /// attention. `Some(128)` for MiMo-V2-Flash (Q/K use `head_dim = 192`);
    /// `None` ⇒ V uses `head_dim` (every other architecture).
    pub v_head_dim: Option<usize>,
    /// Fraction of each head's dims that receive RoPE (`partial_rotary_factor`).
    /// `Some(0.334)` for MiMo-V2-Flash; `None` ⇒ full-head rotation.
    pub partial_rotary_factor: Option<f32>,
    /// Post-attention output scale (`attention_value_scale`). `Some(0.707)`
    /// for MiMo-V2-Flash; `None` ⇒ no scaling.
    pub attention_value_scale: Option<f32>,
    /// Tensor names excluded from FP8 quantisation
    /// (`quantization_config.ignored_layers`); loaded as BF16 instead of
    /// E4M3. Empty for every architecture without an FP8 ignore list.
    pub fp8_ignored_layers: Vec<String>,
    /// Separate KV-head count for Sliding-Window-Attention layers
    /// (`swa_num_key_value_heads`). MiMo-V2-Flash uses 8 KV heads on SWA
    /// layers and 4 (`num_kv_heads`) on global layers. `None` (every other
    /// architecture) ⇒ every layer uses `num_kv_heads`.
    pub swa_num_key_value_heads: Option<usize>,
    /// Whether SWA layers add a learnable per-head attention sink bias to the
    /// logit of the first (sink) token before softmax
    /// (`add_swa_attention_sink_bias`). `false` for every other architecture.
    pub add_swa_attention_sink_bias: bool,
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            scoring_func: ScoringFunc::Softmax,
            // Mixtral renormalises differently and has no `norm_topk_prob`
            // key; defaulting to `false` keeps TOML-only (no config.json)
            // Mixtral runs from silently applying Qwen3-style top-K
            // renormalisation. `true` is only set when a checkpoint's
            // config.json explicitly requests it (see `from_hf_config`).
            norm_topk_prob: false,
            n_group: 1,
            topk_group: 1,
            routed_scaling_factor: 1.0,
            num_shared_experts: 0,
            topk_method: None,
            mla: None,
            rope_scaling: None,
            swa_global_ratio: None,
            hybrid_layer_pattern: None,
            swa_rope_theta: None,
            fp8_block_size: None,
            swiglu_limit: None,
            attention_bias: false,
            v_head_dim: None,
            partial_rotary_factor: None,
            attention_value_scale: None,
            fp8_ignored_layers: Vec::new(),
            swa_num_key_value_heads: None,
            add_swa_attention_sink_bias: false,
        }
    }
}

/// Hyperparameters of the real transformer.
#[derive(Debug, Clone)]
pub struct RealModelConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub num_heads: usize,
    /// Grouped-Query-Attention KV head count. For Mixtral this equals
    /// `num_heads / 4`. Setting it equal to `num_heads` recovers
    /// vanilla multi-head attention.
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    /// Experts per layer (Mixtral-8x7B: 8).
    pub num_experts: usize,
    pub top_k: usize,
    pub rope_base: f32,
    pub rms_eps: f32,
    /// Sliding-window attention span. `None` (default) = full causal
    /// attention. Mixtral uses `Some(4096)`.
    pub window_size: Option<usize>,
    /// Model family this config describes. Drives tensor-name mapping in
    /// the `.safetensors` loader (gate name, `language_model.` prefix,
    /// fused Phi-4 projections) and whether the forward-compute path is
    /// supported. Defaults to [`Architecture::Mixtral`], preserving the
    /// historical behaviour for every existing call site.
    pub architecture: Architecture,
    /// Number of leading dense layers for DeepSeek-style MoE
    /// (`first_k_dense_replace`). `0` (default) means every MoE layer is
    /// sparse, matching Mixtral / Qwen3-MoE.
    pub first_k_dense_replace: usize,
    /// Advanced routing / MLA / RoPE-scaling parameters. Defaults to the
    /// Mixtral/Qwen3 behaviour; populated from `config.json` for DeepSeek.
    pub advanced: AdvancedConfig,
}

/// Options that control how on-disk resident transformer weights are
/// interpreted. The default preserves the historical "best effort"
/// loader: missing dense tensors keep their deterministic seeded values.
#[derive(Debug, Clone, Copy, Default)]
pub struct RealModelLoadOptions {
    /// When enabled, every required resident tensor for the selected
    /// architecture must be present, decodable and shape-compatible. The
    /// loader returns one aggregate error instead of retaining seeded
    /// fallback values for any required tensor.
    pub strict_weights: bool,
}

const GROUP_EMBEDDING: &str = "embedding";
const GROUP_ATTENTION: &str = "attention";
const GROUP_NORMS: &str = "norms";
const GROUP_ROUTING_GATES: &str = "routing_gates";
const GROUP_LM_HEAD: &str = "lm_head";
const GROUP_SHARED_FFN: &str = "shared_ffn";
const GROUP_DENSE_FFN: &str = "dense_ffn";
const GROUP_DENSE_MANIFEST: &str = "dense_manifest";

#[derive(Debug, Clone, Default)]
struct WeightLoadSummary {
    by_group: BTreeMap<&'static str, WeightLoadBucket>,
    by_dtype: BTreeMap<String, WeightLoadBucket>,
}

#[derive(Debug, Clone, Default)]
struct WeightLoadBucket {
    tensors: usize,
    resident_bytes: u64,
}

impl WeightLoadSummary {
    fn record(&mut self, group: &'static str, dtype: impl Into<String>, resident_bytes: usize) {
        let resident_bytes = resident_bytes as u64;
        let group_bucket = self.by_group.entry(group).or_default();
        group_bucket.tensors += 1;
        group_bucket.resident_bytes += resident_bytes;

        let dtype_bucket = self.by_dtype.entry(dtype.into()).or_default();
        dtype_bucket.tensors += 1;
        dtype_bucket.resident_bytes += resident_bytes;
    }
}

/// Structured summary of how a [`RealModel`]'s weights were resolved.
///
/// This is deliberately observable metadata (as opposed to an unstructured
/// log line) so that startup and benchmark code can report — and, on
/// production paths, *reject* — models that retained deterministic seeded
/// fallback weights instead of a complete real checkpoint.
#[derive(Debug, Clone)]
pub struct WeightLoadStatus {
    /// Whether the load ran under strict validation.
    pub strict: bool,
    /// Which loader produced the model: `"seeded"`, `"converted_dir"`, or
    /// `"safetensors"`.
    pub loader: &'static str,
    /// Number of required resident tensors successfully loaded from disk.
    pub loaded_tensors: usize,
    /// Number of required resident tensors the loader attempted.
    pub required_tensors: usize,
    /// Number of optional tensors (e.g. shared-expert sidecars) the loader
    /// probed for. Optional probes never contribute to
    /// [`Self::seeded_fallback_remained`]: an architecture without shared
    /// experts is complete even though these files are absent.
    pub optional_probed: usize,
    /// Number of optional tensors successfully loaded from disk.
    pub optional_loaded: usize,
    /// `true` when at least one *required* tensor kept its seeded fallback
    /// value (only possible outside strict mode, or for a purely seeded
    /// model). Optional tensors are deliberately excluded.
    pub seeded_fallback_remained: bool,
}

impl Default for WeightLoadStatus {
    fn default() -> Self {
        // A freshly seeded model: no real tensors loaded, everything seeded.
        Self {
            strict: false,
            loader: "seeded",
            loaded_tensors: 0,
            required_tensors: 0,
            optional_probed: 0,
            optional_loaded: 0,
            seeded_fallback_remained: true,
        }
    }
}

/// A strict checkpoint-load failure category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLoadFailureKind {
    Missing,
    Unreadable,
    Malformed,
    ShapeMismatch,
    Unsupported,
}

impl WeightLoadFailureKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Unreadable => "unreadable",
            Self::Malformed => "malformed",
            Self::ShapeMismatch => "shape_mismatch",
            Self::Unsupported => "unsupported",
        }
    }
}

/// One required tensor that failed strict checkpoint validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightLoadFailure {
    pub tensor: String,
    pub group: &'static str,
    pub kind: WeightLoadFailureKind,
    pub expected: String,
    pub actual: Option<String>,
    pub detail: Option<String>,
}

impl WeightLoadFailure {
    fn missing(
        tensor: impl Into<String>,
        group: &'static str,
        expected: impl Into<String>,
    ) -> Self {
        Self {
            tensor: tensor.into(),
            group,
            kind: WeightLoadFailureKind::Missing,
            expected: expected.into(),
            actual: None,
            detail: None,
        }
    }

    fn unsupported(
        tensor: impl Into<String>,
        group: &'static str,
        expected: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            tensor: tensor.into(),
            group,
            kind: WeightLoadFailureKind::Unsupported,
            expected: expected.into(),
            actual: None,
            detail: Some(detail.into()),
        }
    }

    fn malformed(
        tensor: impl Into<String>,
        group: &'static str,
        expected: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            tensor: tensor.into(),
            group,
            kind: WeightLoadFailureKind::Malformed,
            expected: expected.into(),
            actual: None,
            detail: Some(detail.into()),
        }
    }
}

/// Aggregate strict-load error. Keeping the individual failures attached
/// lets startup callers and tests inspect the structured inventory instead
/// of scraping a log line.
#[derive(Debug, Clone)]
pub struct StrictWeightLoadError {
    dir: PathBuf,
    failures: Vec<WeightLoadFailure>,
}

impl StrictWeightLoadError {
    fn new(dir: &Path, failures: Vec<WeightLoadFailure>) -> Self {
        Self {
            dir: dir.to_path_buf(),
            failures,
        }
    }

    #[allow(dead_code)]
    pub fn failures(&self) -> &[WeightLoadFailure] {
        &self.failures
    }
}

impl fmt::Display for StrictWeightLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "strict weight load failed for {} ({} failures):",
            self.dir.display(),
            self.failures.len()
        )?;
        for failure in &self.failures {
            write!(
                f,
                "- tensor={} group={} kind={} expected={}",
                failure.tensor,
                failure.group,
                failure.kind.as_str(),
                failure.expected
            )?;
            if let Some(actual) = &failure.actual {
                write!(f, " actual={actual}")?;
            }
            if let Some(detail) = &failure.detail {
                write!(f, " detail={detail}")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl std::error::Error for StrictWeightLoadError {}

impl RealModelConfig {
    /// Tiny default useful for tests / smoke runs (d_model=32, 1 layer).
    pub fn tiny() -> Self {
        Self {
            d_model: 32,
            d_ff: 64,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
            vocab_size: 256,
            num_layers: 1,
            num_experts: 4,
            top_k: 2,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: Architecture::Mixtral,
            first_k_dense_replace: 0,
            advanced: AdvancedConfig::default(),
        }
    }

    /// V head dimension for the standard attention path. Equals
    /// [`Self::head_dim`] for every architecture except MiMo-V2-Flash, which
    /// sets `v_head_dim = 128` while Q/K use `head_dim = 192`.
    pub fn v_head_dim(&self) -> usize {
        self.advanced.v_head_dim.unwrap_or(self.head_dim)
    }

    /// Number of head dims that receive RoPE rotation. Equals
    /// [`Self::head_dim`] (full rotation) unless `partial_rotary_factor` is
    /// set (MiMo-V2-Flash), in which case it is
    /// `floor(head_dim * factor)` rounded down to even.
    pub fn rope_dim(&self) -> usize {
        self.advanced
            .partial_rotary_factor
            .map(|f| ((self.head_dim as f32 * f).floor() as usize) & !1)
            .unwrap_or(self.head_dim)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.head_dim == 0 || self.num_heads == 0 {
            return Err("head_dim and num_heads must both be > 0".into());
        }
        // Mixtral ties the attention projection width to the residual
        // stream (`num_heads * head_dim == d_model`). Newer families
        // (e.g. Qwen3) specify an explicit `head_dim` that does NOT equal
        // `d_model / num_heads`; `MultiHeadSelfAttention` already supports
        // a query width independent of `d_model` (`wo` maps the
        // `num_heads * head_dim` query space back to `d_model`), so we only
        // enforce the tie for Mixtral.
        if self.architecture == Architecture::Mixtral
            && self.head_dim * self.num_heads != self.d_model
        {
            return Err(format!(
                "head_dim * num_heads ({} * {} = {}) must equal d_model ({})",
                self.head_dim,
                self.num_heads,
                self.head_dim * self.num_heads,
                self.d_model
            ));
        }
        if self.num_kv_heads == 0 || self.num_heads % self.num_kv_heads != 0 {
            return Err(format!(
                "num_heads ({}) must be a positive multiple of num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            ));
        }
        if self.top_k == 0 || self.top_k > self.num_experts {
            return Err(format!(
                "top_k ({}) must be in 1..=num_experts ({})",
                self.top_k, self.num_experts
            ));
        }
        if self.num_layers == 0 {
            return Err("num_layers must be > 0".into());
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".into());
        }
        if self.first_k_dense_replace > self.num_layers {
            return Err(format!(
                "first_k_dense_replace ({}) must not exceed num_layers ({})",
                self.first_k_dense_replace, self.num_layers
            ));
        }
        let adv = &self.advanced;
        if adv.n_group == 0 || adv.topk_group == 0 {
            return Err("n_group and topk_group must both be > 0".into());
        }
        if adv.topk_group > adv.n_group {
            return Err(format!(
                "topk_group ({}) must not exceed n_group ({})",
                adv.topk_group, adv.n_group
            ));
        }
        if adv.n_group > 1 && self.num_experts % adv.n_group != 0 {
            return Err(format!(
                "num_experts ({}) must be divisible by n_group ({})",
                self.num_experts, adv.n_group
            ));
        }
        if !adv.routed_scaling_factor.is_finite() || adv.routed_scaling_factor <= 0.0 {
            return Err(format!(
                "routed_scaling_factor ({}) must be a positive, finite number",
                adv.routed_scaling_factor
            ));
        }
        if let Some(f) = adv.partial_rotary_factor {
            if !f.is_finite() || f <= 0.0 || f > 1.0 {
                return Err(format!(
                    "partial_rotary_factor ({}) must be a finite number in (0, 1]",
                    f
                ));
            }
            if self.rope_dim() == 0 {
                return Err(format!(
                    "partial_rotary_factor ({}) with head_dim ({}) rounds the RoPE width down \
                     to zero, which would silently disable rotary embeddings; raise \
                     partial_rotary_factor or head_dim",
                    f, self.head_dim
                ));
            }
        }
        if let Some(v) = adv.v_head_dim {
            if v == 0 || v > self.head_dim {
                return Err(format!(
                    "v_head_dim ({}) must be in 1..=head_dim ({})",
                    v, self.head_dim
                ));
            }
        }
        // MiMo-V2-Flash SWA layers use a separate KV-head count; it must also
        // evenly divide `num_heads`. `None` leaves every layer on `num_kv_heads`.
        if let Some(swa_kv) = adv.swa_num_key_value_heads {
            if swa_kv == 0 || self.num_heads % swa_kv != 0 {
                return Err(format!(
                    "num_heads ({}) must be a positive multiple of swa_num_key_value_heads ({})",
                    self.num_heads, swa_kv
                ));
            }
        }
        Ok(())
    }

    /// Per-architecture tensor-name mapping for this config, carrying the
    /// `first_k_dense_replace` dense/MoE boundary.
    pub fn tensor_naming(&self) -> TensorNaming {
        TensorNaming::new(self.architecture, self.first_k_dense_replace)
    }

    /// Build a [`RealModelConfig`] from a parsed Hugging Face `config.json`
    /// ([`HfConfig`]). Maps the architecture and hyperparameters so a real
    /// checkpoint can be loaded without hand-editing the TOML `[model]` /
    /// `[real_transformer]` sections. Dense families (no routed experts)
    /// collapse to a single expert with `top_k = 1`.
    pub fn from_hf_config(hf: &crate::architecture::HfConfig) -> Self {
        let num_experts = hf.num_routed_experts.unwrap_or(1).max(1);
        let top_k = hf.num_experts_per_tok.unwrap_or(1).clamp(1, num_experts);
        let scoring_func = match hf.scoring_func.as_deref() {
            Some("sigmoid") => ScoringFunc::Sigmoid,
            _ => ScoringFunc::Softmax,
        };
        // Assemble MLA dims only when the config carries the full set.
        let mla = match (
            hf.q_lora_rank,
            hf.kv_lora_rank,
            hf.qk_rope_head_dim,
            hf.qk_nope_head_dim,
            hf.v_head_dim,
        ) {
            (Some(q), Some(kv), Some(rope), Some(nope), Some(v)) => Some(MlaDims {
                q_lora_rank: q,
                kv_lora_rank: kv,
                qk_rope_head_dim: rope,
                qk_nope_head_dim: nope,
                v_head_dim: v,
            }),
            _ => None,
        };
        let advanced = AdvancedConfig {
            scoring_func,
            norm_topk_prob: hf.norm_topk_prob.unwrap_or(false),
            n_group: hf.n_group.unwrap_or(1).max(1),
            topk_group: hf.topk_group.unwrap_or(1).max(1),
            routed_scaling_factor: hf.routed_scaling_factor.unwrap_or(1.0),
            num_shared_experts: hf.num_shared_experts.unwrap_or(0),
            topk_method: hf.topk_method.clone(),
            mla,
            rope_scaling: hf.rope_scaling.clone(),
            swa_global_ratio: hf.swa_global_ratio,
            hybrid_layer_pattern: hf.hybrid_layer_pattern.clone(),
            swa_rope_theta: hf.swa_rope_theta,
            fp8_block_size: hf.fp8_block_size,
            swiglu_limit: hf.swiglu_limit,
            attention_bias: hf.attention_bias,
            // V head dim override only applies to the standard attention
            // path. DeepSeek-style MLA carries its own `v_head_dim` inside
            // `MlaDims`, so when MLA is active the standard attention struct
            // stays symmetric (`v_head_dim == head_dim`) and unused.
            v_head_dim: if mla.is_some() { None } else { hf.v_head_dim },
            partial_rotary_factor: hf.partial_rotary_factor,
            attention_value_scale: hf.attention_value_scale,
            fp8_ignored_layers: hf.fp8_ignored_layers.clone(),
            // MiMo-V2-Flash SWA layers carry a different KV-head count and a
            // per-head attention sink bias. MLA (DeepSeek) ignores both.
            swa_num_key_value_heads: if mla.is_some() {
                None
            } else {
                hf.swa_num_key_value_heads
            },
            add_swa_attention_sink_bias: hf.add_swa_attention_sink_bias,
        };
        Self {
            d_model: hf.hidden_size,
            d_ff: hf.resolved_d_ff(),
            num_heads: hf.num_attention_heads,
            num_kv_heads: if hf.num_key_value_heads == 0 {
                hf.num_attention_heads
            } else {
                hf.num_key_value_heads
            },
            head_dim: hf.resolved_head_dim(),
            vocab_size: hf.vocab_size,
            num_layers: hf.num_hidden_layers,
            num_experts,
            top_k,
            rope_base: hf.rope_theta,
            rms_eps: hf.rms_norm_eps,
            // Hybrid families (MiMo-V2, GPT-OSS) use a 128-token SWA window;
            // fall back to the architecture's default when the config omits
            // `sliding_window` so the per-layer pattern still has a window.
            // `default_swa_window()` returns `None` for every non-hybrid
            // family, so this fallback only ever activates for MiMo-V2 and
            // GPT-OSS — legacy families keep their explicit window (or none).
            window_size: hf
                .sliding_window
                .or_else(|| hf.architecture.default_swa_window()),
            architecture: hf.architecture,
            first_k_dense_replace: hf.first_k_dense_replace.unwrap_or(0),
            advanced,
        }
    }
}

/// Request-level error for the real-model inference path (hardening
/// pass, Part A). Real-model methods fail closed: a corrupted or
/// mathematically invalid step returns one of these instead of
/// silently continuing with incomplete inference. Callers map it to an
/// HTTP 500 (server) or a CLI error (bench) — never a panic.
#[derive(Debug)]
pub enum RealInferenceError {
    /// A token id outside `[0, vocab_size)` reached real-model
    /// embedding lookup (tokenizer/model vocabulary mismatch or caller
    /// bug). Real inference never wraps ids with modulo.
    InvalidTokenId { token_id: u32, vocab_size: usize },
    /// A required routed or shared expert failed to load or execute.
    MoeStep(MoeStepError),
    /// An active attention softmax row produced unexpected NaN or
    /// infinity — numerical corruption, not a valid masked row (A3
    /// rework: propagated structurally from
    /// `softmax → attention → transformer layer → RealModel`, with
    /// full head/position/architecture context).
    NonFiniteAttention {
        layer: usize,
        head: usize,
        pos: usize,
        architecture: crate::architecture::Architecture,
        kind: crate::transformer::NumericalErrorKind,
    },
    /// A scheduler-registered KV state could not be recovered; the
    /// request must fail rather than continue on a fresh KV cache.
    KvStateLost(String),
    /// An architecturally required resident FFN (Qwen2-MoE / DeepSeek
    /// shared expert, or a dense layer's SwiGLU FFN) failed to execute
    /// (hardening pass, A1 completion). Strict mode fails the request;
    /// only the explicit development-only degraded mode may substitute
    /// a zero contribution, and it increments the engine's
    /// `degraded_expert_substitutions` counter.
    ResidentFfn {
        layer: usize,
        role: ResidentFfnRole,
        source: crate::transformer::SharedExpertError,
    },
}

/// Which resident FFN failed in [`RealInferenceError::ResidentFfn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentFfnRole {
    /// Always-on shared expert (Qwen2-MoE / DeepSeek-MoE).
    Shared,
    /// Dense layer FFN (Mistral Small 3, Phi-4, DeepSeek dense prefix).
    Dense,
}

impl std::fmt::Display for ResidentFfnRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResidentFfnRole::Shared => write!(f, "shared expert"),
            ResidentFfnRole::Dense => write!(f, "dense FFN"),
        }
    }
}

impl std::fmt::Display for RealInferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RealInferenceError::InvalidTokenId {
                token_id,
                vocab_size,
            } => write!(
                f,
                "invalid token id {token_id}: model vocab_size is {vocab_size}"
            ),
            RealInferenceError::MoeStep(e) => write!(f, "moe step failed: {e}"),
            RealInferenceError::NonFiniteAttention {
                layer,
                head,
                pos,
                architecture,
                kind,
            } => write!(
                f,
                "non-finite attention detected: {kind} at layer {layer}, head {head}, \
                 position {pos} (architecture {architecture:?})"
            ),
            RealInferenceError::KvStateLost(m) => write!(f, "kv state lost: {m}"),
            RealInferenceError::ResidentFfn {
                layer,
                role,
                source,
            } => write!(f, "required {role} failed at layer {layer}: {source}"),
        }
    }
}

impl std::error::Error for RealInferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RealInferenceError::MoeStep(e) => Some(e),
            RealInferenceError::ResidentFfn { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<MoeStepError> for RealInferenceError {
    fn from(e: MoeStepError) -> Self {
        RealInferenceError::MoeStep(e)
    }
}

/// Decoder-only transformer with MoE FFN blocks. Expert FFN weights are
/// streamed from SSD per token; everything in this struct is dense and
/// stays resident in RAM.
pub struct RealModel {
    pub config: RealModelConfig,
    pub embedding: DenseWeight, // [vocab_size, d_model]
    pub layers: Vec<TransformerLayer>,
    pub final_rms: RmsNorm,
    pub lm_head: LMHead,
    /// How this model's weights were resolved (seeded vs a real checkpoint).
    /// Startup and benchmark paths inspect this to fail closed on seeded
    /// fallback in production.
    pub load_status: WeightLoadStatus,
}

type RopeCacheKey = (usize, u32, u8, u64);

fn rope_cache_hash_yarn(yarn: Option<&YarnRope>) -> u64 {
    let Some(yarn) = yarn else {
        return 0;
    };
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |bits: u32| {
        h ^= bits as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    mix(yarn.attn_factor.to_bits());
    for &freq in &yarn.inv_freq {
        mix(freq.to_bits());
    }
    h
}

fn rope_cache_key(rope_dim: usize, base: f32, yarn: Option<&YarnRope>) -> RopeCacheKey {
    (
        rope_dim,
        base.to_bits(),
        u8::from(yarn.is_some()),
        rope_cache_hash_yarn(yarn),
    )
}

fn shared_rope_cache(
    caches: &mut BTreeMap<RopeCacheKey, Arc<RopeCache>>,
    rope_dim: usize,
    base: f32,
    yarn: Option<&YarnRope>,
) -> Option<Arc<RopeCache>> {
    let key = rope_cache_key(rope_dim, base, yarn);
    if let Some(cache) = caches.get(&key) {
        return Some(cache.clone());
    }
    let cache = Arc::new(RopeCache::new(rope_dim, base, yarn)?);
    caches.insert(key, cache.clone());
    Some(cache)
}

impl RealModel {
    /// Build a model with deterministic, well-conditioned random weights
    /// from a seed. Used as the fallback when on-disk weights aren't
    /// supplied — the engine still streams expert FFN weights from SSD,
    /// so the I/O behaviour the rest of the engine measures is unchanged.
    pub fn new_seeded(config: RealModelConfig, seed: u64) -> Self {
        config.validate().expect("invalid RealModelConfig");
        let mut rng = SplitMix64::new(seed);
        let embedding = DenseWeight::from_f32(
            sample_uniform_vec(&mut rng, config.vocab_size * config.d_model, 0.04),
            config.vocab_size,
            config.d_model,
        );

        let q_dim = config.num_heads * config.head_dim;
        // V projection / output-projection widths use `v_head_dim`, which
        // equals `head_dim` for every architecture except MiMo-V2-Flash.
        // Per-layer K/V projection widths are resolved inside the layer loop
        // (MiMo-V2-Flash SWA layers use a different KV-head count).
        let v_head_dim = config.v_head_dim();
        let attn_out_dim = config.num_heads * v_head_dim;
        let rope_dim = config.rope_dim();
        // Slightly smaller scale for the projections so the residual
        // stream doesn't blow up across many layers.
        let proj_scale = (1.0 / (config.d_model as f32).sqrt()).min(0.05);

        // QK-Norm (Qwen3 / Qwen3-MoE): seed unit-weight per-head RMSNorms so
        // the QK-Norm path is active and ready to receive the loaded
        // `self_attn.{q,k}_norm.weight` tensors. Architectures without
        // QK-Norm (Mixtral, Mistral, Phi-4, DeepSeek) leave these `None`.
        let seed_qk_norm = || -> (Option<RmsNorm>, Option<RmsNorm>) {
            if config.architecture.uses_qk_norm() {
                (
                    Some(RmsNorm::new(vec![1.0; config.head_dim], config.rms_eps)),
                    Some(RmsNorm::new(vec![1.0; config.head_dim], config.rms_eps)),
                )
            } else {
                (None, None)
            }
        };

        let mut rope_caches = BTreeMap::new();
        let layers: Vec<TransformerLayer> = (0..config.num_layers)
            .map(|layer_idx| {
                let (q_norm, k_norm) = seed_qk_norm();
                let mla = seed_mla(&config, &mut rng, proj_scale);
                // Per-layer attention mode (hybrid SWA:global for MiMo-V2 /
                // GPT-OSS, uniform for everything else). Resolve the mode for
                // this layer and store it as `window_size` so SWA and Global
                // layers can coexist in a single model. Legacy uniform-window
                // models (Mixtral) resolve to the same `Some(window)` on
                // every layer; full-causal models resolve to `None`.
                let layer_window = config
                    .architecture
                    .attention_mode(
                        layer_idx,
                        config.window_size,
                        config.advanced.swa_global_ratio,
                        config.advanced.hybrid_layer_pattern.as_deref(),
                    )
                    .window();
                // Per-layer RoPE base. MiMo-V2-Flash uses a separate (much
                // smaller) `swa_rope_theta` on SWA layers; global layers keep
                // the model-level `rope_base`. When `swa_rope_theta` is unset
                // every layer uses `rope_base` (legacy behaviour).
                let layer_rope_base = match (layer_window, config.advanced.swa_rope_theta) {
                    (Some(_), Some(swa_theta)) => swa_theta,
                    _ => config.rope_base,
                };
                // Per-layer KV-head count. MiMo-V2-Flash SWA layers use
                // `swa_num_key_value_heads` (8) while global layers use
                // `num_kv_heads` (4); `num_heads` is the same for both. All
                // other architectures leave `swa_num_key_value_heads = None`,
                // so every layer uses `num_kv_heads` (zero behaviour change).
                let layer_num_kv_heads =
                    match (layer_window, config.advanced.swa_num_key_value_heads) {
                        (Some(_), Some(swa_kv)) => swa_kv, // SWA layer
                        _ => config.num_kv_heads,          // global layer / no override
                    };
                let layer_kv_dim = layer_num_kv_heads * config.head_dim;
                let layer_v_proj_dim = layer_num_kv_heads * v_head_dim;
                // YaRN long-context RoPE scaling for the standard
                // attention path (the MLA path carries its own copy
                // built inside `seed_mla`).
                let rope_yarn = config
                    .advanced
                    .rope_scaling
                    .as_ref()
                    .and_then(|s| YarnRope::from_scaling(rope_dim, layer_rope_base, s));
                let rope_cache = shared_rope_cache(
                    &mut rope_caches,
                    rope_dim,
                    layer_rope_base,
                    rope_yarn.as_ref(),
                );
                TransformerLayer {
                    rms_attn: RmsNorm::new(vec![1.0; config.d_model], config.rms_eps),
                    attn: MultiHeadSelfAttention {
                        d_model: config.d_model,
                        num_heads: config.num_heads,
                        num_kv_heads: layer_num_kv_heads,
                        head_dim: config.head_dim,
                        rope_dim,
                        v_head_dim,
                        attention_value_scale: config.advanced.attention_value_scale,
                        rope_base: layer_rope_base,
                        wq: DenseWeight::from_f32(
                            sample_uniform_vec(&mut rng, q_dim * config.d_model, proj_scale),
                            q_dim,
                            config.d_model,
                        ),
                        wk: DenseWeight::from_f32(
                            sample_uniform_vec(
                                &mut rng,
                                layer_kv_dim * config.d_model,
                                proj_scale,
                            ),
                            layer_kv_dim,
                            config.d_model,
                        ),
                        wv: DenseWeight::from_f32(
                            sample_uniform_vec(
                                &mut rng,
                                layer_v_proj_dim * config.d_model,
                                proj_scale,
                            ),
                            layer_v_proj_dim,
                            config.d_model,
                        ),
                        wo: DenseWeight::from_f32(
                            sample_uniform_vec(
                                &mut rng,
                                config.d_model * attn_out_dim,
                                proj_scale,
                            ),
                            config.d_model,
                            attn_out_dim,
                        ),
                        window_size: layer_window,
                        q_norm,
                        k_norm,
                        rope_yarn,
                        rope_cache,
                        // Attention projection biases (GPT-OSS `attention_bias`)
                        // are seeded absent; the on-disk loader populates them
                        // when the checkpoint sets `attention_bias = true`.
                        bq: None,
                        bk: None,
                        bv: None,
                        bo: None,
                        // Attention sink bias (MiMo-V2-Flash) is seeded absent;
                        // the on-disk loader populates it on SWA layers when the
                        // checkpoint sets `add_swa_attention_sink_bias = true`.
                        sink_bias: None,
                    },
                    mla,
                    rms_moe: RmsNorm::new(vec![1.0; config.d_model], config.rms_eps),
                    gate: LinearGate::new(
                        sample_uniform_vec(
                            &mut rng,
                            config.num_experts * config.d_model,
                            proj_scale,
                        ),
                        config.num_experts,
                        config.d_model,
                        config.top_k,
                    ),
                    // Shared experts are an optional, architecture-specific
                    // tensor (Qwen2-MoE / DeepSeek-MoE). The seeded fallback
                    // leaves them absent so non-shared-expert models and the
                    // synthetic smoke path behave identically to before; real
                    // shared experts are populated by the on-disk loaders.
                    shared_expert: None,
                    // Dense FFN (Mistral Small 3 / Phi-4 / DeepSeek dense
                    // prefix). Populated by the on-disk loaders for dense
                    // layers; `None` means this layer routes to streamed
                    // experts (Mixtral / Qwen3-MoE / DeepSeek sparse layers).
                    dense_ffn: None,
                }
            })
            .collect();
        let final_rms = RmsNorm::new(vec![1.0; config.d_model], config.rms_eps);
        let lm_head = LMHead::new(
            sample_uniform_vec(&mut rng, config.vocab_size * config.d_model, proj_scale),
            config.vocab_size,
            config.d_model,
        );
        Self {
            config,
            embedding,
            layers,
            final_rms,
            lm_head,
            load_status: WeightLoadStatus::default(),
        }
    }

    /// Try to load weights from `dir`, populating any tensor present in
    /// `<dir>/<name>.bin` (raw little-endian `f32`) and falling back to a
    /// seeded random initialisation for anything missing. Logs a one-line
    /// summary of what was loaded vs synthesised.
    ///
    /// Expected file names (all optional — missing ones use the seed):
    /// - `embed.bin`               : `[vocab_size * d_model]` f32
    /// - `final_rms.bin`           : `[d_model]` f32 (gain)
    /// - `lm_head.bin`             : `[vocab_size * d_model]` f32
    /// - `rms_attn_<L>.bin`        : `[d_model]` f32 (gain)
    /// - `rms_moe_<L>.bin`         : `[d_model]` f32 (gain)
    /// - `attn_<L>_q.bin` / `_k` / `_v` / `_o` : projection weights
    /// - `gate_<L>.bin`            : `[num_experts * d_model]` f32
    pub fn from_dir(
        config: RealModelConfig,
        dir: &Path,
        seed: u64,
    ) -> Result<Self, std::io::Error> {
        Self::from_dir_with_options(config, dir, seed, RealModelLoadOptions::default())
    }

    /// Like [`Self::from_dir`], with opt-in strict validation for
    /// production checkpoint loading.
    pub fn from_dir_with_options(
        config: RealModelConfig,
        dir: &Path,
        seed: u64,
        options: RealModelLoadOptions,
    ) -> Result<Self, std::io::Error> {
        config
            .validate()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        let mut model = Self::new_seeded(config.clone(), seed);
        let naming = config.tensor_naming();
        let mut loaded = 0usize;
        let mut tried = 0usize;
        // The converted-directory (`.bin`) layout has no optional tensors: every
        // probed tensor is architecture-required (shared experts are required
        // when declared). These counters are reported for parity with the
        // safetensors loader.
        let optional_probed = 0usize;
        let optional_loaded = 0usize;
        let mut summary = WeightLoadSummary::default();
        let mut failures = Vec::new();
        let dense_manifest = resolve_dense_manifest(dir, options.strict_weights, &mut failures);

        macro_rules! maybe {
            ($group:expr, $name:expr, $expected:expr, $assign:expr) => {{
                tried += 1;
                if let Some(v) = try_load_resident_f32(
                    dir,
                    dense_manifest.as_ref(),
                    $name,
                    $expected,
                    $group,
                    options.strict_weights,
                    &mut failures,
                    &mut summary,
                ) {
                    $assign(v);
                    loaded += 1;
                }
            }};
        }
        macro_rules! maybe_dense {
            ($group:expr, $name:expr, $rows:expr, $cols:expr, $assign:expr) => {{
                tried += 1;
                if let Some(v) = try_load_dense_weight(
                    dir,
                    dense_manifest.as_ref(),
                    $name,
                    $rows,
                    $cols,
                    $group,
                    options.strict_weights,
                    &mut failures,
                    &mut summary,
                ) {
                    $assign(v);
                    loaded += 1;
                }
            }};
        }

        let d_model = config.d_model;
        let q_dim = config.num_heads * config.head_dim;

        maybe_dense!(
            GROUP_EMBEDDING,
            "embed.bin",
            config.vocab_size,
            d_model,
            |v| model.embedding = v
        );
        maybe!(GROUP_NORMS, "final_rms.bin", d_model, |v| {
            model.final_rms = RmsNorm::new(v, config.rms_eps);
        });
        maybe_dense!(
            GROUP_LM_HEAD,
            "lm_head.bin",
            config.vocab_size,
            d_model,
            |v| {
                model.lm_head = LMHead::from_dense(v, config.vocab_size, d_model);
            }
        );
        for l in 0..config.num_layers {
            // Per-layer K/V projection widths. MiMo-V2-Flash SWA layers use a
            // different KV-head count than global layers; `new_seeded` already
            // set `num_kv_heads`/`v_head_dim` per layer, so derive the loaded
            // tensor lengths from the seeded attention struct.
            let kv_dim = model.layers[l].attn.kv_dim();
            let v_proj_dim = model.layers[l].attn.v_proj_dim();
            maybe!(GROUP_NORMS, &format!("rms_attn_{l}.bin"), d_model, |v| {
                model.layers[l].rms_attn = RmsNorm::new(v, config.rms_eps);
            });
            maybe!(GROUP_NORMS, &format!("rms_moe_{l}.bin"), d_model, |v| {
                model.layers[l].rms_moe = RmsNorm::new(v, config.rms_eps);
            });
            maybe_dense!(
                GROUP_ATTENTION,
                &format!("attn_{l}_q.bin"),
                q_dim,
                d_model,
                |v| {
                    model.layers[l].attn.wq = v;
                }
            );
            maybe_dense!(
                GROUP_ATTENTION,
                &format!("attn_{l}_k.bin"),
                kv_dim,
                d_model,
                |v| {
                    model.layers[l].attn.wk = v;
                }
            );
            maybe_dense!(
                GROUP_ATTENTION,
                &format!("attn_{l}_v.bin"),
                v_proj_dim,
                d_model,
                |v| {
                    model.layers[l].attn.wv = v;
                }
            );
            maybe_dense!(
                GROUP_ATTENTION,
                &format!("attn_{l}_o.bin"),
                d_model,
                q_dim,
                |v| {
                    model.layers[l].attn.wo = v;
                }
            );
            // QK-Norm (Qwen3 / Qwen3-MoE): per-head RMSNorm weights of length
            // `head_dim`, applied to Q and K before RoPE. Seeded as unit-weight
            // in `new_seeded` for these architectures; the converted-directory
            // loader overwrites them from `q_norm_<L>.bin` / `k_norm_<L>.bin`
            // (or the `dense_manifest.json` aliases emitted by the GGUF
            // converter). Under strict loading a missing/malformed/wrong-shape
            // tensor is reported by `try_load_resident_f32`, so no seeded unit
            // norm can silently survive. Architectures without QK-Norm (Mixtral
            // and friends) never attempt these files.
            if config.architecture.uses_qk_norm() {
                maybe!(
                    GROUP_NORMS,
                    &format!("q_norm_{l}.bin"),
                    config.head_dim,
                    |v| {
                        model.layers[l].attn.q_norm = Some(RmsNorm::new(v, config.rms_eps));
                    }
                );
                maybe!(
                    GROUP_NORMS,
                    &format!("k_norm_{l}.bin"),
                    config.head_dim,
                    |v| {
                        model.layers[l].attn.k_norm = Some(RmsNorm::new(v, config.rms_eps));
                    }
                );
            }
            if model.layers[l].mla.is_some() && options.strict_weights {
                failures.push(WeightLoadFailure::unsupported(
                    format!("mla_layer_{l}"),
                    GROUP_ATTENTION,
                    "DeepSeek MLA projection stack",
                    "raw .bin loader does not define MLA tensor files; use .safetensors",
                ));
            }
            if naming.ffn_kind(l) == FfnKind::Moe {
                maybe_dense!(
                    GROUP_ROUTING_GATES,
                    &format!("gate_{l}.bin"),
                    config.num_experts,
                    d_model,
                    |v| {
                        model.layers[l].gate =
                            LinearGate::from_dense(v, config.num_experts, d_model, config.top_k);
                    }
                );
            } else if options.strict_weights {
                failures.push(WeightLoadFailure::unsupported(
                    format!("dense_ffn_{l}"),
                    GROUP_DENSE_FFN,
                    "resident dense FFN projections",
                    "raw .bin loader does not define dense FFN tensor files; use .safetensors",
                ));
            }
            // Shared expert (Qwen2-MoE / DeepSeek-MoE). Only architectures that
            // declare always-on shared experts (`num_shared_experts > 0`) own
            // these tensors; for those the shared-expert stack is *required* and
            // a missing/malformed set fails strict loading. Architectures without
            // shared experts (Mixtral, Qwen3-MoE) must not probe or count these
            // files, otherwise a complete checkpoint would report a lingering
            // seeded fallback. The shared expert's intermediate size is inferred
            // from the on-disk tensor length (`gate floats / d_model`).
            if config.advanced.num_shared_experts > 0 && naming.ffn_kind(l) == FfnKind::Moe {
                tried += 1;
                match Self::load_shared_expert_bin(dir, l, d_model) {
                    Ok(Some(se)) => {
                        let resident_bytes = (se.weights.len()
                            + se.gate_inp.as_ref().map(|g| g.len()).unwrap_or(0))
                            * 4;
                        model.layers[l].shared_expert = Some(se);
                        summary.record(GROUP_SHARED_FFN, "f32", resident_bytes);
                        loaded += 1;
                    }
                    Ok(None) => {
                        if options.strict_weights {
                            failures.push(WeightLoadFailure::missing(
                                format!("layer_{l}_shexp_*"),
                                GROUP_SHARED_FFN,
                                "resident shared-expert gate/up/down projections",
                            ));
                        }
                    }
                    Err(e) => {
                        // Present but malformed: never silently drop a
                        // required shared expert (A1) — strict loading
                        // records a malformed failure, non-strict keeps
                        // the seeded fallback with a loud warning.
                        warn!(layer = l, error = %e,
                            "shared expert tensors present but malformed");
                        if options.strict_weights {
                            failures.push(WeightLoadFailure::malformed(
                                format!("layer_{l}_shexp_*"),
                                GROUP_SHARED_FFN,
                                "resident shared-expert gate/up/down projections",
                                e.to_string(),
                            ));
                        }
                    }
                }
            }
        }
        if options.strict_weights && !failures.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                StrictWeightLoadError::new(dir, failures),
            ));
        }
        model.load_status = WeightLoadStatus {
            strict: options.strict_weights,
            loader: "converted_dir",
            loaded_tensors: loaded,
            required_tensors: tried,
            optional_probed,
            optional_loaded,
            seeded_fallback_remained: loaded < tried,
        };
        info!(
            dir = %dir.display(),
            loaded,
            tried,
            optional_probed,
            optional_loaded,
            strict_weights = options.strict_weights,
            fallback_seeded = !options.strict_weights,
            seeded_fallback_remained = model.load_status.seeded_fallback_remained,
            weight_groups = ?summary.by_group,
            weight_dtypes = ?summary.by_dtype,
            "real transformer weights loaded"
        );
        Ok(model)
    }

    /// Read a whole little-endian `f32` `.bin` file into a `Vec<f32>`.
    /// Returns `None` if the file is absent, unreadable, or not a whole
    /// number of `f32`s. Unlike the size-capped `try_load` helper this
    /// returns the *entire* tensor, which the shared-expert loader needs
    /// in order to infer the shared intermediate size from the length.
    fn read_full_f32(path: &Path) -> Option<Vec<f32>> {
        use std::io::Read;
        if !path.is_file() {
            return None;
        }
        let file = std::fs::File::open(path).ok()?;
        let len = file.metadata().ok()?.len();
        if len == 0 || len % 4 != 0 {
            return None;
        }
        // Stream-decode 4 bytes at a time straight into the `Vec<f32>`
        // rather than slurping the whole file into a `Vec<u8>` first; the
        // `BufReader` batches the underlying reads, so peak memory is just
        // the output buffer (no second full-size byte buffer).
        let mut reader = std::io::BufReader::with_capacity(64 * 1024, file);
        let want = (len / 4) as usize;
        let mut out = Vec::with_capacity(want);
        let mut chunk = [0u8; 4];
        loop {
            match reader.read_exact(&mut chunk) {
                Ok(()) => out.push(f32::from_le_bytes(chunk)),
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(_) => return None,
            }
        }
        if out.len() != want {
            return None;
        }
        Some(out)
    }

    /// Load a layer's optional shared expert from the GGUF-extractor
    /// `.bin` files (`layer_{l}_shexp_{gate,up,down,gate_inp}.bin`).
    /// The shared intermediate size is inferred from the gate tensor
    /// length. Absent tensor files return `Ok(None)` (the checkpoint
    /// genuinely has no shared expert on disk); files that are present
    /// but inconsistent return a typed error so an architecturally
    /// *required* shared expert is never silently dropped (A1).
    fn load_shared_expert_bin(
        dir: &Path,
        l: usize,
        d_model: usize,
    ) -> Result<Option<SharedExpert>, crate::transformer::SharedExpertError> {
        let gate = Self::read_full_f32(&dir.join(format!("layer_{l}_shexp_gate.bin")));
        let up = Self::read_full_f32(&dir.join(format!("layer_{l}_shexp_up.bin")));
        let down = Self::read_full_f32(&dir.join(format!("layer_{l}_shexp_down.bin")));
        let (Some(gate), Some(up), Some(down)) = (gate, up, down) else {
            return Ok(None);
        };
        let shape_err = |gate_inp_len: Option<usize>| {
            crate::transformer::SharedExpertError::ShapeMismatch {
                d_model,
                d_ff: if d_model == 0 { 0 } else { gate.len() / d_model },
                gate_len: gate.len(),
                up_len: up.len(),
                down_len: down.len(),
                gate_inp_len,
            }
        };
        if d_model == 0 || gate.len() % d_model != 0 {
            return Err(shape_err(None));
        }
        let shared_d_ff = gate.len() / d_model;
        if shared_d_ff == 0 {
            return Err(shape_err(None));
        }
        // The sigmoid gate (Qwen2-MoE) is optional; DeepSeek-MoE omits it —
        // but a *present* gate file with the wrong length is malformed.
        let gate_inp = Self::read_full_f32(&dir.join(format!("layer_{l}_shexp_gate_inp.bin")));
        if let Some(g) = gate_inp.as_ref() {
            if g.len() != d_model {
                return Err(shape_err(Some(g.len())));
            }
        }
        SharedExpert::from_projections(d_model, shared_d_ff, &gate, &up, &down, gate_inp).map(Some)
    }

    /// Like [`Self::from_dir`] but loads dense weights from
    /// HuggingFace-style `.safetensors` shards instead of per-tensor
    /// `.bin` files. The layout mirrors what
    /// `transformers.AutoModelForCausalLM.save_pretrained` writes:
    ///
    /// * `<dir>/model.safetensors` (single-shard) **or**
    /// * `<dir>/model-00001-of-00002.safetensors` etc. (multi-shard,
    ///   concatenated keys). Both are picked up automatically.
    ///
    /// Tensor names follow the standard Mixtral / Llama convention:
    ///
    /// ```text
    ///   model.embed_tokens.weight                                              -> embed
    ///   model.layers.{L}.input_layernorm.weight                                -> rms_attn[L]
    ///   model.layers.{L}.post_attention_layernorm.weight                       -> rms_moe[L]
    ///   model.layers.{L}.self_attn.{q,k,v,o}_proj.weight                       -> attn weights
    ///   model.layers.{L}.block_sparse_moe.gate.weight                          -> gate[L]
    ///   model.norm.weight                                                      -> final_rms
    ///   lm_head.weight                                                         -> lm_head
    /// ```
    ///
    /// Tensors are loaded as `f32` regardless of on-disk dtype: `bf16`
    /// and `f16` are dequantised at load time. Per-expert FFN weights
    /// (`block_sparse_moe.experts.*`) are **not** loaded here — they
    /// come through the SSD-streaming engine via `expert_<id>.bin`.
    /// Anything missing falls back to seeded init, exactly like
    /// [`Self::from_dir`].
    pub fn from_safetensors(
        config: RealModelConfig,
        dir: &Path,
        seed: u64,
    ) -> Result<Self, std::io::Error> {
        Self::from_safetensors_with_options(config, dir, seed, RealModelLoadOptions::default())
    }

    /// Like [`Self::from_safetensors`], with opt-in strict validation for
    /// production checkpoint loading.
    pub fn from_safetensors_with_options(
        config: RealModelConfig,
        dir: &Path,
        seed: u64,
        options: RealModelLoadOptions,
    ) -> Result<Self, std::io::Error> {
        config
            .validate()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        // Defensive guard: refuse — *before* touching the (potentially
        // multi-hundred-GB) shards — to build any architecture whose
        // forward-compute path is not implemented. Every recognised
        // architecture is currently executable (DeepSeek-V3 included, via
        // MLA latent-KV attention + FP8 dequant), so this never triggers
        // today; it is retained so a future, mapping-only architecture
        // fails loud here rather than routing on garbage activations.
        if let ComputeSupport::Unsupported { reason } = config.architecture.compute_support() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!(
                    "cannot build a runnable `{}` model from {}: {reason}",
                    config.architecture.model_type(),
                    dir.display()
                ),
            ));
        }

        let mut model = Self::new_seeded(config.clone(), seed);
        let naming = config.tensor_naming();

        // Discover .safetensors shards in `dir`. We hold each shard's
        // bytes for the duration of the load (SafeTensors borrows from
        // them) and the sharded checkpoints in HF rarely exceed a few
        // GiB on the dense (non-expert) tensors we care about.
        let mut shards: Vec<(std::path::PathBuf, Vec<u8>)> = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                let bytes = std::fs::read(&p)?;
                shards.push((p, bytes));
            }
        }
        if shards.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no .safetensors files found in {}", dir.display()),
            ));
        }
        // Stable order: sort by path so multi-shard checkpoints load
        // deterministically across runs.
        shards.sort_by(|a, b| a.0.cmp(&b.0));
        let parsed: Vec<safetensors::SafeTensors> = shards
            .iter()
            .map(|(p, bytes)| {
                safetensors::SafeTensors::deserialize(bytes).map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("failed to parse {}: {}", p.display(), e),
                    )
                })
            })
            .collect::<Result<_, _>>()?;

        // FP8 block-quantisation tile edge. MiMo-V2-Flash and DeepSeek-V3
        // both use square 128×128 tiles; drive the edge from the checkpoint's
        // `weight_block_size` (`config.advanced.fp8_block_size`) when present
        // so a checkpoint with a different tile size dequantises with the
        // matching scale grid, falling back to the historical `FP8_BLOCK`.
        // The dequantiser only models *square* tiles, so a non-square
        // `weight_block_size` (`block_rows != block_cols`) is rejected (warn +
        // fall back) rather than silently using one dimension for both.
        let fp8_block = match config.advanced.fp8_block_size {
            Some([r, c]) if r > 0 && r == c => r,
            Some([r, c]) => {
                warn!(
                    block_rows = r,
                    block_cols = c,
                    "non-square FP8 weight_block_size is unsupported; \
                     using default {FP8_BLOCK}x{FP8_BLOCK} tiles"
                );
                FP8_BLOCK
            }
            None => FP8_BLOCK,
        };

        // -- Multi-Token-Prediction (MTP) tensor skipping (MiMo-V2) --------
        //
        // MiMo-V2 ships `num_nextn_predict_layers` extra "next-token
        // prediction" heads whose weights appear in the checkpoint as
        // additional decoder layers (`model.layers.{idx}` with
        // `idx >= num_hidden_layers`) and/or tensors carrying an `mtp` /
        // `nextn` marker. MER does single-token decoding and never consults
        // these heads, so they are intentionally not looked up by the
        // name-driven loader below. The standard loader already ignores any
        // tensor it does not explicitly request, so this pass does not
        // change loading behaviour — it exists purely to make the skip
        // *intentional and observable*: we count the MTP tensors and emit a
        // single `debug!` line, rather than silently relying on the
        // name-lookup miss. (Not an error: extra MTP tensors are expected.)
        if config.architecture == Architecture::MiMoV2 {
            let num_layers = config.num_layers;
            let is_mtp_tensor = |name: &str| -> bool {
                // Match `mtp` / `nextn` only as whole dot-delimited path
                // segments (optionally with an `_…` suffix, e.g.
                // `nextn_predict`), so unrelated names that merely contain
                // the substring (e.g. `…attempts…` → `mtp`) are not flagged.
                let is_marker_segment = |seg: &str| {
                    let seg = seg.to_ascii_lowercase();
                    for marker in ["mtp", "nextn"] {
                        if seg == marker || seg.starts_with(&format!("{marker}_")) {
                            return true;
                        }
                    }
                    false
                };
                if name.split('.').any(is_marker_segment) {
                    return true;
                }
                // `model.layers.{idx}.…` with idx beyond the configured
                // decoder depth is an MTP predict layer.
                if let Some(rest) = name.split("model.layers.").nth(1) {
                    if let Some(idx_str) = rest.split('.').next() {
                        if let Ok(idx) = idx_str.parse::<usize>() {
                            return idx >= num_layers;
                        }
                    }
                }
                false
            };
            let mtp_skipped: usize = parsed
                .iter()
                .flat_map(|st| st.names().into_iter())
                .filter(|name| is_mtp_tensor(name))
                .count();
            if mtp_skipped > 0 {
                debug!(
                    arch = "mimo_v2_flash",
                    mtp_tensors_skipped = mtp_skipped,
                    "skipping MiMo-V2 Multi-Token-Prediction tensors (single-token decode)"
                );
            }
        }

        // Closure: search every shard for `name` and decode as f32.
        // Returns `None` when the tensor isn't found in any shard or
        // when the element count doesn't match `expected`.
        let find_f32 = |name: &str, expected: usize| -> Option<Vec<f32>> {
            for st in &parsed {
                if let Ok(view) = st.tensor(name) {
                    let n_elem: usize = view.shape().iter().product();
                    if n_elem != expected {
                        warn!(
                            tensor = name,
                            have = n_elem,
                            need = expected,
                            "safetensors shape mismatch; falling back to seeded init"
                        );
                        return None;
                    }
                    return Some(decode_safetensor_to_f32(&view, name));
                }
            }
            None
        };

        // Finding 3/4: like `find_f32`, but requires an exact rank-2
        // `[rows, cols]` tensor. A transpose, a flattened rank-1 tensor, or
        // any rank > 2 tensor is rejected (never accepted on an element-count
        // coincidence). Rank-1 vector compatibility lives only in `find_f32`.
        let find_f32_2d = |name: &str, rows: usize, cols: usize| -> Option<Vec<f32>> {
            for st in &parsed {
                if let Ok(view) = st.tensor(name) {
                    let shape = view.shape();
                    let n_elem: usize = shape.iter().product();
                    let expected = rows.saturating_mul(cols);
                    if n_elem != expected {
                        warn!(
                            tensor = name,
                            have = n_elem,
                            need = expected,
                            "safetensors shape mismatch; falling back to seeded init"
                        );
                        return None;
                    }
                    if !safetensor_orientation_ok(shape, rows, cols) {
                        warn!(
                            tensor = name,
                            shape = ?shape,
                            rows,
                            cols,
                            "safetensors tensor is transposed/misshaped; falling back to seeded init"
                        );
                        return None;
                    }
                    return Some(decode_safetensor_to_f32(&view, name));
                }
            }
            None
        };

        // FP8 `ignored_layers` (MiMo-V2-Flash): tensors listed in
        // `quantization_config.ignored_layers` (all `self_attn.o_proj`) are
        // stored as BF16, not FP8. The block-dequant closures below must
        // decode them via the standard path rather than misinterpreting the
        // bytes as E4M3. Each entry is a module path (e.g.
        // `model.layers.0.self_attn.o_proj`); the safetensors tensor names
        // append a suffix (`.weight`, `.bias`, `_scale_inv`) and may carry a
        // shard prefix. We therefore match the entry as a substring but
        // require a separator (`.`/`_`) or end-of-string boundary after it,
        // so a path like `...o_proj` never matches `...o_projection`.
        let fp8_ignored = config.advanced.fp8_ignored_layers.clone();
        let is_fp8_ignored = |name: &str| safetensor_name_is_fp8_ignored(&fp8_ignored, name);

        if options.strict_weights {
            let failures = strict_safetensors_failures(
                dir,
                &parsed,
                &model,
                &config,
                &naming,
                fp8_block,
                &fp8_ignored,
            );
            if !failures.is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    StrictWeightLoadError::new(dir, failures),
                ));
            }
        }

        // Closure: search every shard for the first matching `name` and
        // return its decoded f32 data regardless of length. Used by the
        // shared-expert loader, which infers the shared intermediate size
        // from the tensor length rather than asserting a configured one.
        // DeepSeek-V3 FP8 (`e4m3`) 2D weights are transparently
        // block-dequantised via their companion `<name>_scale_inv`.
        let find_f32_any_named = |names: &[String]| -> Option<(String, Vec<f32>)> {
            use safetensors::tensor::Dtype;
            for name in names {
                for st in &parsed {
                    if let Ok(view) = st.tensor(name) {
                        if view.dtype() != Dtype::F8_E4M3 {
                            return Some((name.clone(), decode_safetensor_to_f32(&view, name)));
                        }
                        if is_fp8_ignored(name) {
                            let raw = view.data();
                            // `ignored_layers` are stored as BF16. Some checkpoints still tag
                            // them as FP8 in metadata; detect BF16 payload by byte width.
                            if raw.len() % 2 == 0 {
                                return Some((
                                    name.clone(),
                                    raw.chunks_exact(2)
                                        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                                        .collect(),
                                ));
                            }
                            warn!(
                                tensor = name,
                                "FP8-ignored tensor did not look like BF16 payload"
                            );
                            return None;
                        }
                        let shape = view.shape();
                        if shape.len() != 2 {
                            return None;
                        }
                        let (rows, cols) = (shape[0], shape[1]);
                        let scale_name = format!("{name}_scale_inv");
                        let mut scale_inv = None;
                        for s in &parsed {
                            if let Ok(sv) = s.tensor(&scale_name) {
                                scale_inv = Some(decode_safetensor_to_f32(&sv, &scale_name));
                                break;
                            }
                        }
                        let scale_inv = scale_inv?;
                        let out = crate::mla::dequant_fp8_e4m3_blockwise(
                            view.data(),
                            &scale_inv,
                            rows,
                            cols,
                            fp8_block,
                        );
                        if out.is_empty() {
                            return None;
                        }
                        return Some((name.clone(), out));
                    }
                }
            }
            None
        };
        let find_f32_any =
            |names: &[String]| -> Option<Vec<f32>> { find_f32_any_named(names).map(|(_, v)| v) };

        // Closure: search every shard for `name`, decoding it as f32 and
        // transparently dequantising DeepSeek-V3 FP8 (`e4m3`) weights via
        // their companion `<name>_scale_inv` block-scale tensor. Non-FP8
        // dtypes route through the standard decoder. Returns `None` when
        // the tensor is missing or its element count != `expected`.
        let find_f32_dequant = |name: &str, expected: usize| -> Option<Vec<f32>> {
            use safetensors::tensor::Dtype;
            for st in &parsed {
                if let Ok(view) = st.tensor(name) {
                    let shape = view.shape();
                    let n_elem: usize = shape.iter().product();
                    match view.dtype() {
                        // U8 carries either OCP MXFP4 packed weights
                        // (GPT-OSS `*_blocks`: two E2M1 nibbles per byte,
                        // so `n_elem == expected / 2`) or some unrelated
                        // metadata tensor (e.g. attention-sink scales)
                        // that happens to share a name prefix.
                        Dtype::U8 => {
                            if shape.len() == 2
                                && shape[0] != 0
                                && expected % shape[0] == 0
                                && shape[1] == (expected / shape[0]).div_ceil(2)
                            {
                                let rows = shape[0];
                                let cols = expected / rows;
                                let Some(scales) = find_mxfp4_scales(&parsed, name, rows, cols)
                                else {
                                    warn!(
                                        tensor = name,
                                        "MXFP4 weight missing companion block scales; falling back to seeded init"
                                    );
                                    return None;
                                };
                                let out =
                                    crate::dequant::dequant_mxfp4(view.data(), &scales, rows, cols);
                                if out.is_empty() {
                                    warn!(
                                        tensor = name,
                                        "MXFP4 dequant failed (shape mismatch); falling back to seeded init"
                                    );
                                    return None;
                                }
                                return Some(out);
                            }
                            // Same element count as a real weight but raw
                            // U8: not a format we model. Skip quietly so it
                            // falls back to seeded init without log spam.
                            debug!(
                                tensor = name,
                                n_elem, "skipping non-weight U8 safetensors tensor"
                            );
                            return None;
                        }
                        // Per-tensor INT8: decode (cast) then apply the
                        // optional companion `<name>_scale` scalar.
                        Dtype::I8 => {
                            if n_elem != expected {
                                warn!(
                                    tensor = name,
                                    have = n_elem,
                                    need = expected,
                                    "safetensors shape mismatch; falling back to seeded init"
                                );
                                return None;
                            }
                            let mut out = decode_safetensor_to_f32(&view, name);
                            let scale_name = format!("{name}_scale");
                            let scale = (|| {
                                for s in &parsed {
                                    if let Ok(sv) = s.tensor(&scale_name) {
                                        let v = decode_safetensor_to_f32(&sv, &scale_name);
                                        if v.len() == 1 {
                                            return Some(v[0]);
                                        }
                                        warn!(
                                            tensor = name,
                                            scale = %scale_name,
                                            len = v.len(),
                                            "INT8 companion scale tensor is not a scalar; ignoring"
                                        );
                                        return None;
                                    }
                                }
                                None
                            })();
                            match scale {
                                Some(s) if s != 1.0 => {
                                    for x in out.iter_mut() {
                                        *x *= s;
                                    }
                                }
                                None => {
                                    warn!(
                                        tensor = name,
                                        scale = %scale_name,
                                        "INT8 weight missing companion scale; using unit scale"
                                    );
                                }
                                _ => {}
                            }
                            return Some(out);
                        }
                        Dtype::F8_E4M3 => {
                            // FP8 `ignored_layers` (MiMo-V2-Flash o_proj):
                            // stored as BF16 even though tagged here; decode
                            // via the standard path, never block-dequant.
                            if is_fp8_ignored(name) {
                                let raw = view.data();
                                if raw.len() % 2 == 0 {
                                    return Some(
                                        raw.chunks_exact(2)
                                            .map(|c| {
                                                half::bf16::from_le_bytes([c[0], c[1]]).to_f32()
                                            })
                                            .collect(),
                                    );
                                }
                                warn!(tensor = name, "FP8-ignored tensor did not look like BF16 payload; falling back to seeded init");
                                return None;
                            }
                            if n_elem != expected {
                                warn!(
                                    tensor = name,
                                    have = n_elem,
                                    need = expected,
                                    "safetensors shape mismatch; falling back to seeded init"
                                );
                                return None;
                            }
                            // FP8 block-wise quantised 2D weight. DeepSeek
                            // stores a companion `<name>_scale_inv` of
                            // reciprocal block scales.
                            if shape.len() != 2 {
                                warn!(
                                    tensor = name,
                                    "FP8 weight is not 2D; cannot block-dequantise"
                                );
                                return None;
                            }
                            let (rows, cols) = (shape[0], shape[1]);
                            let scale_name = format!("{name}_scale_inv");
                            let scale_inv = (|| {
                                for s in &parsed {
                                    if let Ok(sv) = s.tensor(&scale_name) {
                                        return Some(decode_safetensor_to_f32(&sv, &scale_name));
                                    }
                                }
                                None
                            })();
                            let Some(scale_inv) = scale_inv else {
                                warn!(
                                    tensor = name,
                                    scale = %scale_name,
                                    "FP8 weight missing companion scale_inv; falling back to seeded init"
                                );
                                return None;
                            };
                            let out = crate::mla::dequant_fp8_e4m3_blockwise(
                                view.data(),
                                &scale_inv,
                                rows,
                                cols,
                                fp8_block,
                            );
                            if out.is_empty() {
                                warn!(
                                    tensor = name,
                                    "FP8 block-dequant failed (shape mismatch); falling back to seeded init"
                                );
                                return None;
                            }
                            return Some(out);
                        }
                        _ => {
                            if n_elem != expected {
                                warn!(
                                    tensor = name,
                                    have = n_elem,
                                    need = expected,
                                    "safetensors shape mismatch; falling back to seeded init"
                                );
                                return None;
                            }
                            return Some(decode_safetensor_to_f32(&view, name));
                        }
                    }
                }
            }
            None
        };

        let mut tried = 0usize;
        let mut loaded = 0usize;
        let mut summary = WeightLoadSummary::default();
        macro_rules! maybe {
            ($group:expr, $name:expr, $expected:expr, $assign:expr) => {{
                tried += 1;
                let name = $name;
                if let Some(v) = find_f32(name, $expected) {
                    summary.record($group, safetensor_source_dtype(&parsed, name), v.len() * 4);
                    $assign(v);
                    loaded += 1;
                }
            }};
        }
        // Finding 3: 2-D matrix variant that enforces `[rows, cols]`
        // orientation in addition to the element-count check.
        macro_rules! maybe2d {
            ($group:expr, $name:expr, $rows:expr, $cols:expr, $assign:expr) => {{
                tried += 1;
                let name = $name;
                if let Some(v) = find_f32_2d(name, $rows, $cols) {
                    summary.record($group, safetensor_source_dtype(&parsed, name), v.len() * 4);
                    $assign(v);
                    loaded += 1;
                }
            }};
        }
        // FP8 dequant happens via `find_f32_dequant` / `find_f32_any`
        // directly (e.g. in `load_mla_layer`); no extra macro needed.

        let d_model = config.d_model;
        let q_dim = config.num_heads * config.head_dim;
        // V projection / output-projection widths use `v_head_dim`
        // (MiMo-V2-Flash asymmetric V); equal to `kv_dim` / `q_dim` for
        // every other architecture. Per-layer K/V widths are resolved inside
        // the layer loop (MiMo-V2-Flash SWA layers use a different KV-head
        // count); the output width is `num_heads * v_head_dim` for all layers.
        let attn_out_dim = config.num_heads * config.v_head_dim();

        maybe2d!(
            GROUP_EMBEDDING,
            &naming.embed(),
            config.vocab_size,
            d_model,
            |v| {
                model.embedding = DenseWeight::from_f32(v, config.vocab_size, d_model);
            }
        );
        maybe!(GROUP_NORMS, &naming.final_norm(), d_model, |v| {
            model.final_rms = RmsNorm::new(v, config.rms_eps);
        });
        maybe2d!(
            GROUP_LM_HEAD,
            &naming.lm_head(),
            config.vocab_size,
            d_model,
            |v| {
                model.lm_head = LMHead::new(v, config.vocab_size, d_model);
            }
        );
        for l in 0..config.num_layers {
            // Per-layer K/V projection widths. MiMo-V2-Flash SWA layers use a
            // different KV-head count than global layers; `new_seeded` already
            // resolved `num_kv_heads`/`v_head_dim` per layer, so derive the
            // expected tensor lengths from the seeded attention struct.
            let kv_dim = model.layers[l].attn.kv_dim();
            let v_proj_dim = model.layers[l].attn.v_proj_dim();
            maybe!(GROUP_NORMS, &naming.input_layernorm(l), d_model, |v| {
                model.layers[l].rms_attn = RmsNorm::new(v, config.rms_eps);
            });
            maybe!(
                GROUP_NORMS,
                &naming.post_attention_layernorm(l),
                d_model,
                |v| {
                    model.layers[l].rms_moe = RmsNorm::new(v, config.rms_eps);
                }
            );

            // Attention projections. DeepSeek-V3 uses MLA (latent-KV)
            // attention with its own low-rank projection stack; every
            // other family uses standard dense Q/K/V/O. The presence of an
            // `mla` block on the seeded layer (built from `config.advanced.mla`)
            // selects the path.
            if model.layers[l].mla.is_some() {
                load_mla_layer(
                    &mut model.layers[l],
                    l,
                    &naming,
                    &config,
                    &find_f32_dequant,
                    &mut |name, group, resident_bytes| {
                        summary.record(
                            group,
                            safetensor_source_dtype(&parsed, name),
                            resident_bytes,
                        );
                    },
                    &mut tried,
                    &mut loaded,
                );
            } else if naming.attn_qkv_fused() {
                // Phi-4 ships a single fused `qkv_proj`
                // ([(num_heads + 2*num_kv_heads) * head_dim, d_model],
                // row-major) that we split into separate Q/K/V weights.
                tried += 1;
                let qkv_name = naming.attn_qkv(l);
                if let Some(v) = find_f32_2d(&qkv_name, q_dim + 2 * kv_dim, d_model) {
                    summary.record(
                        GROUP_ATTENTION,
                        safetensor_source_dtype(&parsed, &qkv_name),
                        v.len() * 4,
                    );
                    let (q_part, rest) = v.split_at(q_dim * d_model);
                    let (k_part, v_part) = rest.split_at(kv_dim * d_model);
                    model.layers[l].attn.wq =
                        DenseWeight::from_f32(q_part.to_vec(), q_dim, d_model);
                    model.layers[l].attn.wk =
                        DenseWeight::from_f32(k_part.to_vec(), kv_dim, d_model);
                    model.layers[l].attn.wv =
                        DenseWeight::from_f32(v_part.to_vec(), kv_dim, d_model);
                    loaded += 1;
                }
                maybe2d!(GROUP_ATTENTION, &naming.attn_o(l), d_model, q_dim, |v| {
                    model.layers[l].attn.wo = DenseWeight::from_f32(v, d_model, q_dim);
                });
            } else {
                maybe2d!(GROUP_ATTENTION, &naming.attn_q(l), q_dim, d_model, |v| {
                    model.layers[l].attn.wq = DenseWeight::from_f32(v, q_dim, d_model);
                });
                maybe2d!(GROUP_ATTENTION, &naming.attn_k(l), kv_dim, d_model, |v| {
                    model.layers[l].attn.wk = DenseWeight::from_f32(v, kv_dim, d_model);
                });
                maybe2d!(
                    GROUP_ATTENTION,
                    &naming.attn_v(l),
                    v_proj_dim,
                    d_model,
                    |v| {
                        model.layers[l].attn.wv = DenseWeight::from_f32(v, v_proj_dim, d_model);
                    }
                );
                maybe2d!(
                    GROUP_ATTENTION,
                    &naming.attn_o(l),
                    d_model,
                    attn_out_dim,
                    |v| {
                        model.layers[l].attn.wo =
                            DenseWeight::from_f32(v, d_model, attn_out_dim);
                    }
                );
            }

            // QK-Norm (Qwen3 / Qwen3-MoE): per-head RMSNorm weights of
            // length `head_dim`, applied to Q and K before RoPE. Seeded as
            // unit-weight in `new_seeded` for these architectures; overwrite
            // with the loaded weights when present.
            if config.architecture.uses_qk_norm() {
                maybe!(GROUP_NORMS, &naming.attn_q_norm(l), config.head_dim, |v| {
                    model.layers[l].attn.q_norm = Some(RmsNorm::new(v, config.rms_eps));
                });
                maybe!(GROUP_NORMS, &naming.attn_k_norm(l), config.head_dim, |v| {
                    model.layers[l].attn.k_norm = Some(RmsNorm::new(v, config.rms_eps));
                });
            }

            // Attention projection biases (GPT-OSS `attention_bias = true`):
            // learnable additive terms on Q/K/V/O. Only loaded when the
            // checkpoint declares them; every other family leaves these
            // `None` (seeded in `new_seeded`). Missing tensors are tolerated
            // via `maybe!`, so a config that sets `attention_bias` but ships
            // no bias tensors degrades to the bias-free path rather than
            // failing the whole load.
            if config.advanced.attention_bias {
                maybe!(GROUP_ATTENTION, &naming.q_proj_bias(l), q_dim, |v| {
                    model.layers[l].attn.bq = Some(v);
                });
                maybe!(GROUP_ATTENTION, &naming.k_proj_bias(l), kv_dim, |v| {
                    model.layers[l].attn.bk = Some(v);
                });
                maybe!(GROUP_ATTENTION, &naming.v_proj_bias(l), v_proj_dim, |v| {
                    model.layers[l].attn.bv = Some(v);
                });
                maybe!(GROUP_ATTENTION, &naming.o_proj_bias(l), d_model, |v| {
                    model.layers[l].attn.bo = Some(v);
                });
            }

            // Attention sink bias (MiMo-V2-Flash `add_swa_attention_sink_bias`):
            // a learnable per-head scalar (length `num_heads`) added to the
            // logit of position 0 before softmax. Only loaded on SWA layers
            // (`window_size.is_some()`) and only when the config sets the flag;
            // every other family leaves `sink_bias = None` (seeded in
            // `new_seeded`). A missing tensor is tolerated via `maybe!`.
            if config.advanced.add_swa_attention_sink_bias
                && model.layers[l].attn.window_size.is_some()
            {
                maybe!(
                    GROUP_ATTENTION,
                    &naming.attn_sink_bias(l),
                    config.num_heads,
                    |v| {
                        model.layers[l].attn.sink_bias = Some(v);
                    }
                );
            }

            // Dense FFN layers (Mistral Small 3, Phi-4, and DeepSeek's
            // `first_k_dense_replace` prefix) carry a resident SwiGLU FFN
            // instead of routing to streamed experts. Load it here so
            // `RealModel::step` can run it directly. Phi-4 fuses gate+up
            // into a single `mlp.gate_up_proj` that we split.
            if naming.ffn_kind(l) == FfnKind::Dense {
                let dense: Result<Option<SharedExpert>, _> = if naming.mlp_gate_up_fused() {
                    // Phi-4: `mlp.gate_up_proj` is `[2*d_ff, d_model]`,
                    // row-major, gate rows first then up rows. `down_proj`
                    // is `[d_model, d_ff]`.
                    let gate_up = find_f32_any_named(&[naming.mlp_gate_up(l)]);
                    let down = find_f32_any_named(&[naming.mlp_down(l)]);
                    match (gate_up, down) {
                        (Some((gu_name, gu)), Some((down_name, down)))
                            if d_model != 0 && gu.len() % (2 * d_model) == 0 && !gu.is_empty() =>
                        {
                            let ffn_d = gu.len() / (2 * d_model);
                            let (gate, up) = gu.split_at(ffn_d * d_model);
                            let se = SharedExpert::from_projections(
                                d_model, ffn_d, gate, up, &down, None,
                            );
                            if se.is_ok() {
                                summary.record(
                                    GROUP_DENSE_FFN,
                                    safetensor_source_dtype(&parsed, &gu_name),
                                    gu.len() * 4,
                                );
                                summary.record(
                                    GROUP_DENSE_FFN,
                                    safetensor_source_dtype(&parsed, &down_name),
                                    down.len() * 4,
                                );
                            }
                            se.map(Some)
                        }
                        (Some((_, gu)), Some((_, down))) => {
                            // Present but inconsistently shaped fused
                            // projections: malformed, not absent (A1).
                            Err(crate::transformer::SharedExpertError::ShapeMismatch {
                                d_model,
                                d_ff: 0,
                                gate_len: gu.len(),
                                up_len: gu.len(),
                                down_len: down.len(),
                                gate_inp_len: None,
                            })
                        }
                        _ => Ok(None),
                    }
                } else {
                    let gate = find_f32_any_named(&[naming.mlp_gate(l)]);
                    let up = find_f32_any_named(&[naming.mlp_up(l)]);
                    let down = find_f32_any_named(&[naming.mlp_down(l)]);
                    match (gate, up, down) {
                        (Some((gate_name, gate)), Some((up_name, up)), Some((down_name, down)))
                            if d_model != 0 && gate.len() % d_model == 0 && !gate.is_empty() =>
                        {
                            let ffn_d = gate.len() / d_model;
                            let se = SharedExpert::from_projections(
                                d_model, ffn_d, &gate, &up, &down, None,
                            );
                            if se.is_ok() {
                                summary.record(
                                    GROUP_DENSE_FFN,
                                    safetensor_source_dtype(&parsed, &gate_name),
                                    gate.len() * 4,
                                );
                                summary.record(
                                    GROUP_DENSE_FFN,
                                    safetensor_source_dtype(&parsed, &up_name),
                                    up.len() * 4,
                                );
                                summary.record(
                                    GROUP_DENSE_FFN,
                                    safetensor_source_dtype(&parsed, &down_name),
                                    down.len() * 4,
                                );
                            }
                            se.map(Some)
                        }
                        (Some((_, gate)), Some((_, up)), Some((_, down))) => {
                            Err(crate::transformer::SharedExpertError::ShapeMismatch {
                                d_model,
                                d_ff: 0,
                                gate_len: gate.len(),
                                up_len: up.len(),
                                down_len: down.len(),
                                gate_inp_len: None,
                            })
                        }
                        _ => Ok(None),
                    }
                };
                tried += 1;
                match dense {
                    Ok(Some(se)) => {
                        model.layers[l].dense_ffn = Some(se);
                        loaded += 1;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        // A dense layer's FFN is architecturally required;
                        // present-but-malformed tensors must not silently
                        // degrade to the seeded fallback without a loud
                        // signal (A1). `loaded < tried` marks the seeded
                        // fallback, and strict startup validation rejects it.
                        warn!(layer = l, error = %e,
                            "dense FFN tensors present but malformed");
                    }
                }
            }

            // Routed MoE gate. Only present on sparse layers — DeepSeek's
            // first `first_k_dense_replace` layers and the dense families
            // (Mistral Small 3, Phi-4, dense Qwen3) have no router. The
            // gate name differs per family (`block_sparse_moe.gate` for
            // Mixtral vs `mlp.gate` for Qwen3-MoE / DeepSeek).
            if config.architecture.is_moe() && naming.ffn_kind(l) == FfnKind::Moe {
                let adv = &config.advanced;
                let expected = config.num_experts * d_model;
                // DeepSeek-V3 aux-loss-free balancing: a per-expert bias
                // added to selection scores only. Absent on Mixtral/Qwen3.
                let correction_bias = find_f32_any(&[naming.moe_gate_correction_bias(l)])
                    .filter(|b| b.len() == config.num_experts);
                // Prefer an extracted `gate_<L>.bin` sitting alongside the
                // shards over re-deriving the gate inline from the
                // checkpoint, so the two loader paths (`from_dir` /
                // `from_safetensors`, dispatched by `from_dir_auto`) agree
                // when both sources are present. The on-disk bin is only
                // honoured when its length matches the configured shape;
                // otherwise we fall back to the inline `moe_gate` tensor.
                tried += 1;
                let gate_bin = format!("gate_{l}.bin");
                let sidecar = Self::read_full_f32(&dir.join(&gate_bin));
                let (gate_vec, gate_dtype) = match sidecar {
                    // Accept the extracted sidecar only when its length is
                    // *exactly* `num_experts * d_model`. Never truncate an
                    // oversized file (Finding 8): a wrong-sized sidecar is a
                    // corruption/mismatch signal, so warn and fall back to the
                    // canonical inline `moe_gate` tensor instead.
                    Some(v) if v.len() == expected => (Some(v), Some("bin_f32".to_string())),
                    Some(v) => {
                        warn!(
                            file = %dir.join(&gate_bin).display(),
                            got = v.len(),
                            expected,
                            "gate sidecar has wrong length; not truncating — \
                             falling back to canonical gate tensor"
                        );
                        let gate_name = naming.moe_gate(l);
                        let cv = find_f32(&gate_name, expected);
                        let dtype = cv
                            .as_ref()
                            .map(|_| safetensor_source_dtype(&parsed, &gate_name));
                        (cv, dtype)
                    }
                    None => {
                        let gate_name = naming.moe_gate(l);
                        let v = find_f32(&gate_name, expected);
                        let dtype = v
                            .as_ref()
                            .map(|_| safetensor_source_dtype(&parsed, &gate_name));
                        (v, dtype)
                    }
                };
                if let Some(v) = gate_vec {
                    summary.record(
                        GROUP_ROUTING_GATES,
                        gate_dtype.unwrap_or_else(|| "unknown".to_string()),
                        v.len() * 4,
                    );
                    model.layers[l].gate = LinearGate::with_routing(
                        v,
                        config.num_experts,
                        d_model,
                        config.top_k,
                        adv.scoring_func,
                        adv.norm_topk_prob,
                        correction_bias.clone(),
                        adv.n_group,
                        adv.topk_group,
                        adv.routed_scaling_factor,
                    );
                    loaded += 1;
                }
            }
            // Optional shared expert (Qwen2-MoE / DeepSeek-MoE). Only
            // attempt the load when the checkpoint actually declares
            // always-on shared experts (`n_shared_experts > 0`): Qwen3-MoE
            // and Mixtral have none, so probing there is pure wasted I/O.
            // Checkpoints name the shared expert either `shared_expert`
            // (Qwen2-MoE) or `shared_experts` (DeepSeek-MoE), so we probe
            // both. The shared intermediate size is inferred from the gate
            // tensor length. Prefix-aware so the `language_model.`
            // checkpoints resolve correctly.
            if config.advanced.num_shared_experts > 0 {
                let p = naming.prefix();
                tried += 1;
                let shexp_gate = find_f32_any_named(&[
                    format!("{p}model.layers.{l}.mlp.shared_expert.gate_proj.weight"),
                    format!("{p}model.layers.{l}.mlp.shared_experts.gate_proj.weight"),
                ]);
                let shexp_up = find_f32_any_named(&[
                    format!("{p}model.layers.{l}.mlp.shared_expert.up_proj.weight"),
                    format!("{p}model.layers.{l}.mlp.shared_experts.up_proj.weight"),
                ]);
                let shexp_down = find_f32_any_named(&[
                    format!("{p}model.layers.{l}.mlp.shared_expert.down_proj.weight"),
                    format!("{p}model.layers.{l}.mlp.shared_experts.down_proj.weight"),
                ]);
                if let (Some(gate), Some(up), Some(down)) = (shexp_gate, shexp_up, shexp_down) {
                    let (gate_name, gate) = gate;
                    let (up_name, up) = up;
                    let (down_name, down) = down;
                    if d_model != 0 && gate.len() % d_model == 0 && gate.len() / d_model != 0 {
                        let shared_d_ff = gate.len() / d_model;
                        // Sigmoid gate is Qwen2-MoE-only (`shared_expert_gate`).
                        let gate_inp = find_f32_any_named(&[format!(
                            "{p}model.layers.{l}.mlp.shared_expert_gate.weight"
                        )])
                        .filter(|(_, g)| g.len() == d_model);
                        let gate_inp_vec = gate_inp.as_ref().map(|(_, g)| g.clone());
                        match SharedExpert::from_projections(
                            d_model,
                            shared_d_ff,
                            &gate,
                            &up,
                            &down,
                            gate_inp_vec,
                        ) {
                            Ok(se) => {
                                summary.record(
                                    GROUP_SHARED_FFN,
                                    safetensor_source_dtype(&parsed, &gate_name),
                                    gate.len() * 4,
                                );
                                summary.record(
                                    GROUP_SHARED_FFN,
                                    safetensor_source_dtype(&parsed, &up_name),
                                    up.len() * 4,
                                );
                                summary.record(
                                    GROUP_SHARED_FFN,
                                    safetensor_source_dtype(&parsed, &down_name),
                                    down.len() * 4,
                                );
                                if let Some((gate_inp_name, gate_inp)) = gate_inp {
                                    summary.record(
                                        GROUP_SHARED_FFN,
                                        safetensor_source_dtype(&parsed, &gate_inp_name),
                                        gate_inp.len() * 4,
                                    );
                                }
                                model.layers[l].shared_expert = Some(se);
                                loaded += 1;
                            }
                            // Present but malformed: leave the seeded
                            // fallback flagged (`loaded < tried`) instead of
                            // silently removing a required shared expert (A1).
                            Err(e) => warn!(
                                layer = l,
                                shared_d_ff,
                                d_model,
                                error = %e,
                                "shared expert tensors present but malformed"
                            ),
                        }
                    }
                }
            }
        }
        model.load_status = WeightLoadStatus {
            strict: options.strict_weights,
            loader: "safetensors",
            loaded_tensors: loaded,
            required_tensors: tried,
            // Every safetensors tensor counted above is architecture-required:
            // dense FFN is only probed on dense layers, and shared experts only
            // when the checkpoint declares them. There are no optional probes.
            optional_probed: 0,
            optional_loaded: 0,
            seeded_fallback_remained: loaded < tried,
        };
        info!(
            dir = %dir.display(),
            shards = shards.len(),
            loaded,
            tried,
            strict_weights = options.strict_weights,
            fallback_seeded = !options.strict_weights,
            seeded_fallback_remained = model.load_status.seeded_fallback_remained,
            weight_groups = ?summary.by_group,
            weight_dtypes = ?summary.by_dtype,
            "loaded dense weights from .safetensors"
        );
        Ok(model)
    }

    /// Auto-dispatching entry point used by the HTTP server: if `dir`
    /// contains any `.safetensors` files we use the safetensors path;
    /// otherwise we fall back to the legacy raw-`.bin` loader. Either
    /// way, missing tensors degrade to seeded init.
    pub fn from_dir_auto(
        config: RealModelConfig,
        dir: &Path,
        seed: u64,
    ) -> Result<Self, std::io::Error> {
        Self::from_dir_auto_with_options(config, dir, seed, RealModelLoadOptions::default())
    }

    /// Auto-dispatching entry point with explicit load options.
    pub fn from_dir_auto_with_options(
        config: RealModelConfig,
        dir: &Path,
        seed: u64,
        options: RealModelLoadOptions,
    ) -> Result<Self, std::io::Error> {
        let has_safetensors = std::fs::read_dir(dir)
            .map(|it| {
                it.flatten()
                    .any(|e| e.path().extension().and_then(|s| s.to_str()) == Some("safetensors"))
            })
            .unwrap_or(false);
        if has_safetensors {
            Self::from_safetensors_with_options(config, dir, seed, options)
        } else {
            Self::from_dir_with_options(config, dir, seed, options)
        }
    }

    /// Initial KV caches — one per layer, all empty.
    pub fn fresh_kv_caches(&self) -> Vec<KvCache> {
        self.layers
            .iter()
            .map(|l| KvCache::new_kv(l.kv_dim(), l.v_dim()))
            .collect()
    }

    /// Look up the embedding row for a token id, wrapping out-of-range
    /// ids with modulo. **Synthetic/speculative paths only** (routing
    /// peeks, draft-token KV previews, seeded benchmarks): real
    /// inference must use [`Self::try_embed`], which rejects invalid
    /// ids instead of silently aliasing them onto a valid row
    /// (hardening pass, Part A5).
    pub fn embed(&self, token_id: u32) -> Vec<f32> {
        let id = (token_id as usize) % self.config.vocab_size;
        let mut out = Vec::with_capacity(self.config.d_model);
        self.embedding.row_dequant_into(id, &mut out);
        out
    }

    /// Fail-closed embedding lookup for real inference (Part A5): a
    /// token id outside `[0, vocab_size)` is a tokenizer/model
    /// vocabulary mismatch or a caller bug and must fail the request,
    /// never wrap onto an unrelated embedding row.
    pub fn try_embed(&self, token_id: u32) -> Result<Vec<f32>, RealInferenceError> {
        let id = token_id as usize;
        if id >= self.config.vocab_size {
            return Err(RealInferenceError::InvalidTokenId {
                token_id,
                vocab_size: self.config.vocab_size,
            });
        }
        let mut out = Vec::with_capacity(self.config.d_model);
        self.embedding.row_dequant_into(id, &mut out);
        Ok(out)
    }

    /// Translate a `(layer, local_expert_id)` pair into the global flat
    /// expert id used by the engine's cache + storage. See module
    /// docstring; this is the addressing scheme that makes the existing
    /// single-namespace cache work for multi-layer models without any
    /// API changes.
    #[inline]
    pub fn global_expert_id(&self, layer: usize, local: u32) -> u32 {
        // Checked namespace arithmetic (hardening pass, Part A6): the
        // product/sum is validated at config load, so an overflow here
        // is a programming error — fail loudly instead of wrapping a
        // malformed id into another layer's cache namespace.
        let layer = u32::try_from(layer).expect("layer index exceeds u32");
        debug_assert!((local as usize) < self.config.num_experts);
        layer
            .checked_mul(self.config.num_experts as u32)
            .and_then(|base| base.checked_add(local))
            .expect("global expert id overflows u32 (num_layers * num_experts too large)")
    }

    /// **SSD-read pre-pass peek (gist Phase 1, reworked by hardening
    /// pass C2).** Return a best-effort, **lightweight** estimate of
    /// the global expert ids this step is likely to require. The
    /// pre-pass is cheap, side-effect-free and request-agnostic:
    ///
    /// * Every layer's gate is evaluated against the token's
    ///   **embedding** as a stand-in for that layer's residual
    ///   stream. This is not exact — real routing depends on
    ///   attention over the request's KV state and every preceding
    ///   MoE FFN output — but it captures the strong tokenwise /
    ///   layerwise routing correlation seen on Mixtral-class
    ///   checkpoints and provides useful hints for the warm pass.
    ///
    /// The previous implementation additionally *cloned the layer-0 KV
    /// cache and re-ran full layer-0 attention* per peeked token to
    /// make layer 0 exact. That paid an `O(pos · kv_dim)` copy plus a
    /// duplicate attention pass on every scheduled request per batch —
    /// far more than the one routing decision it sharpened was worth —
    /// and forced the scheduler to lock each request's KV state during
    /// the pre-pass. Both are gone: the peek reads nothing but the
    /// token id and the resident gate weights, so it never touches
    /// request state and cannot leak routing observations across
    /// requests.
    ///
    /// Misses are cheap by construction: anything the peek gets wrong
    /// is still caught by the in-flight singleflight on the critical
    /// path, and the scheduler's profitability gate disables the
    /// pre-pass entirely when its measured benefit is non-positive.
    ///
    /// The returned vector is **not** deduplicated; the caller is
    /// expected to fold it into a [`HashSet`] before calling
    /// [`Engine::warm_with`].
    pub fn peek_experts(&self, token_id: u32) -> Vec<u32> {
        // Each layer contributes exactly `routing.experts.len()` ids,
        // which by the gating contract equals `config.top_k`.
        let mut out = Vec::with_capacity(self.config.num_layers * self.config.top_k);
        let embed = self.embed(token_id);
        for layer_idx in 0..self.config.num_layers {
            let layer = &self.layers[layer_idx];
            let (_normed, routing) = layer.moe_pre(&embed);
            for &local in &routing.experts {
                out.push(self.global_expert_id(layer_idx, local));
            }
        }
        out
    }

    /// Ingest one token, update all per-layer KV caches, and return the
    /// final RMS-normalised hidden state without evaluating the LM head.
    ///
    /// This is the realisation of the gist's pseudocode:
    ///
    /// ```text
    ///   x = embedding[token_id]
    ///   for layer in layers:
    ///       x = layer.attn_block(x, pos, kv[layer])
    ///       (normed, routing) = layer.moe_pre(x)
    ///       experts_y = engine.moe_step(normed, routing.experts)  // SSD-streamed
    ///       x = layer.moe_combine(x, experts_y, routing.weights)
    ///   return final_rms(x)
    /// ```
    ///
    /// `engine.moe_step` is what reads expert weights from SSD via the
    /// LRU cache — that's the whole point of the substrate.
    /// This is the prompt-ingestion half of the real transformer path:
    /// it performs every model/layer side effect required for future
    /// attention, but deliberately does not pay for the LM head.
    pub async fn forward_token_hidden(
        &self,
        engine: &Arc<Engine>,
        token_id: u32,
        pos: usize,
        kv: &mut [KvCache],
    ) -> Result<Vec<f32>, RealInferenceError> {
        self.forward_token_hidden_with_timing(engine, token_id, pos, kv, None)
            .await
    }

    pub async fn forward_token_hidden_with_timing(
        &self,
        engine: &Arc<Engine>,
        token_id: u32,
        pos: usize,
        kv: &mut [KvCache],
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> Result<Vec<f32>, RealInferenceError> {
        assert_eq!(
            kv.len(),
            self.config.num_layers,
            "kv cache slice must have one entry per layer"
        );
        let mut x =
            crate::stage_timing::time_optional(timings, crate::stage_timing::EMBEDDING, || {
                self.try_embed(token_id)
            })?;
        let backend = engine.execution_context().attention_backend();
        let mut layer_scratch = crate::transformer::TransformerLayerScratch::new();
        let mut next_x = Vec::with_capacity(self.config.d_model);
        // Fail-closed numerical propagation (hardening pass, A3
        // rework + policy separation): strict production mode routes
        // attention through the `try_` variants, which return a typed
        // [`crate::transformer::AttentionNumericalError`] from the
        // softmax itself — no side-channel counters are consulted for
        // correctness (the global `nonfinite_softmax_fallbacks`
        // counter remains telemetry only). Only the *independent*
        // development-only `allow_nonfinite_attention_fallback` policy
        // keeps the legacy uniform-fallback behaviour — enabling
        // `allow_degraded_experts` does NOT fabricate attention.
        // GPU-offloaded attention performs its softmax on-device where
        // non-finite rows cannot be detected; strict mode therefore
        // always uses the checked CPU attention path (see
        // `forward_into_impl`).
        let strict_numerics = !engine.allow_nonfinite_attention_fallback();
        let allow_degraded = engine.allow_degraded_experts();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Attention sub-block.
            if strict_numerics {
                layer
                    .try_attn_block_into_with_timing(
                        &x,
                        pos,
                        layer_idx,
                        &mut kv[layer_idx],
                        &**backend,
                        &mut layer_scratch,
                        &mut next_x,
                        timings,
                    )
                    .map_err(|e| RealInferenceError::NonFiniteAttention {
                        layer: layer_idx,
                        head: e.head,
                        pos: e.pos,
                        architecture: self.config.architecture,
                        kind: e.kind,
                    })?;
            } else {
                layer.attn_block_into_with_timing(
                    &x,
                    pos,
                    layer_idx,
                    &mut kv[layer_idx],
                    &**backend,
                    &mut layer_scratch,
                    &mut next_x,
                    timings,
                );
            }
            std::mem::swap(&mut x, &mut next_x);
            next_x.clear();
            // Sliding-Window-Attention layers (MiMo-V2 SWA layers, GPT-OSS
            // banded layers, Mixtral's uniform window) only ever attend to
            // the last `window` positions, so KV entries older than that are
            // dead weight. Evict them to keep this layer's cache bounded at
            // O(window) instead of O(seq_len). Global layers (`window_size
            // == None`) keep their full history untouched.
            if let crate::architecture::AttentionMode::SlidingWindow { window } =
                layer.attn.attention_mode()
            {
                kv[layer_idx].evict_before(pos.saturating_sub(window));
            }
            // Dense FFN layers (Mistral Small 3, Phi-4, DeepSeek dense
            // prefix) bypass the SSD-streamed expert path entirely: run the
            // resident SwiGLU FFN and skip routing. A dense FFN failure
            // fails the request in strict mode (A1 completion); only the
            // explicit degraded mode substitutes a zero contribution
            // (residual add of zero == hidden unchanged) and counts it.
            match layer.dense_forward_with_timing(&x, timings) {
                Ok(Some(dense_out)) => {
                    x = dense_out;
                    continue;
                }
                Ok(None) => {}
                Err(source) => {
                    if !allow_degraded {
                        return Err(RealInferenceError::ResidentFfn {
                            layer: layer_idx,
                            role: ResidentFfnRole::Dense,
                            source,
                        });
                    }
                    engine.record_degraded_expert_substitution();
                    tracing::warn!(
                        layer = layer_idx,
                        error = %source,
                        "DEGRADED MODE: dense FFN failed; substituting zero \
                         contribution (allow_degraded_experts = true, output is \
                         non-authoritative)"
                    );
                    continue;
                }
            }
            // MoE sub-block: route, await SSD-streamed expert FFNs, combine.
            let routing = layer.moe_pre_into_with_timing(&x, &mut layer_scratch, timings);
            layer_scratch.global_expert_ids.clear();
            layer_scratch
                .global_expert_ids
                .extend(
                    routing
                        .experts
                        .iter()
                        .map(|&local| self.global_expert_id(layer_idx, local)),
                );
            let normed = &layer_scratch.moe_normed;
            // `token_idx` here is just a digest seed; positional info is
            // already baked into RoPE inside `attn_block`.
            let token_idx =
                (pos as u64).wrapping_mul(self.config.num_layers as u64) + layer_idx as u64;
            engine
                .moe_step_weighted_into_with_timing(
                    token_idx,
                    layer_idx as u32,
                    normed,
                    &layer_scratch.global_expert_ids,
                    &routing.weights,
                    &mut layer_scratch.moe_accum,
                    timings,
                )
                .await?;
            layer.moe_accumulated_into_with_timing(
                &x,
                &layer_scratch.moe_accum,
                &mut next_x,
                timings,
            );
            std::mem::swap(&mut x, &mut next_x);
            next_x.clear();
            layer_scratch.routing.recycle_decision(routing);
            // Qwen2-MoE / DeepSeek-MoE shared expert: a dense always-on
            // FFN over the same MoE-normalised hidden, added to the
            // residual alongside the routed experts. `None` for Mixtral
            // (no-op), keeping the engine MoE-architecture-agnostic. A
            // shared-expert failure fails the request in strict mode (A1
            // completion); only the explicit degraded mode drops the
            // contribution (and counts the substitution).
            match layer.shared_expert_forward_with_timing(normed, timings) {
                Ok(Some(shared)) => {
                    crate::transformer::add_residual_into(&x, &shared, &mut next_x);
                    std::mem::swap(&mut x, &mut next_x);
                    next_x.clear();
                }
                Ok(None) => {}
                Err(source) => {
                    if !allow_degraded {
                        return Err(RealInferenceError::ResidentFfn {
                            layer: layer_idx,
                            role: ResidentFfnRole::Shared,
                            source,
                        });
                    }
                    engine.record_degraded_expert_substitution();
                    tracing::warn!(
                        layer = layer_idx,
                        error = %source,
                        "DEGRADED MODE: shared expert failed; dropping its \
                         contribution (allow_degraded_experts = true, output is \
                         non-authoritative)"
                    );
                }
            }
        }
        Ok(crate::stage_timing::time_optional(
            timings,
            crate::stage_timing::FINAL_RMS_NORM,
            || self.final_rms.forward(&x),
        ))
    }

    /// Diagnostic variant of token forward pass that captures intermediate activation traces at each layer boundary.
    pub async fn forward_token_diagnostic_trace(
        &self,
        engine: &Arc<Engine>,
        token_id: u32,
        pos: usize,
        kv: &mut [KvCache],
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> Result<crate::gpu_native_diagnostics::ModelDiagnosticTrace, RealInferenceError> {
        assert_eq!(
            kv.len(),
            self.config.num_layers,
            "kv cache slice must have one entry per layer"
        );
        let mut x =
            crate::stage_timing::time_optional(timings, crate::stage_timing::EMBEDDING, || {
                self.try_embed(token_id)
            })?;
        let embedding_trace = x.clone();

        let backend = engine.execution_context().attention_backend();
        let mut layer_scratch = crate::transformer::TransformerLayerScratch::new();
        let mut next_x = Vec::with_capacity(self.config.d_model);
        let strict_numerics = !engine.allow_nonfinite_attention_fallback();
        let allow_degraded = engine.allow_degraded_experts();

        let mut layer_post_attn = Vec::with_capacity(self.layers.len());
        let mut layer_router_input = Vec::with_capacity(self.layers.len());
        let mut layer_router_logits = Vec::with_capacity(self.layers.len());
        let mut layer_selected_ids = Vec::with_capacity(self.layers.len());
        let mut layer_selected_weights = Vec::with_capacity(self.layers.len());
        let mut layer_post_moe = Vec::with_capacity(self.layers.len());

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Attention sub-block.
            if strict_numerics {
                layer
                    .try_attn_block_into_with_timing(
                        &x,
                        pos,
                        layer_idx,
                        &mut kv[layer_idx],
                        &**backend,
                        &mut layer_scratch,
                        &mut next_x,
                        timings,
                    )
                    .map_err(|e| RealInferenceError::NonFiniteAttention {
                        layer: layer_idx,
                        head: e.head,
                        pos: e.pos,
                        architecture: self.config.architecture,
                        kind: e.kind,
                    })?;
            } else {
                layer.attn_block_into_with_timing(
                    &x,
                    pos,
                    layer_idx,
                    &mut kv[layer_idx],
                    &**backend,
                    &mut layer_scratch,
                    &mut next_x,
                    timings,
                );
            }
            std::mem::swap(&mut x, &mut next_x);
            next_x.clear();
            if let crate::architecture::AttentionMode::SlidingWindow { window } =
                layer.attn.attention_mode()
            {
                kv[layer_idx].evict_before(pos.saturating_sub(window));
            }
            layer_post_attn.push(x.clone());
            // MoE sub-block: route, await SSD-streamed expert FFNs, combine.
            let routing = layer.moe_pre_into_with_timing(&x, &mut layer_scratch, timings);
            layer_scratch.global_expert_ids.clear();
            layer_scratch
                .global_expert_ids
                .extend(
                    routing
                        .experts
                        .iter()
                        .map(|&local| self.global_expert_id(layer_idx, local)),
                );
            let normed = &layer_scratch.moe_normed;
            layer_router_input.push(normed.clone());
            // Diagnostic-only replay of the same authoritative CPU gate GEMV.
            // The production routing decision above remains the source of the
            // selected ids and weights; this copy preserves the pre-softmax
            // logits needed to explain cutoff drift.
            layer_router_logits.push(layer.gate.weights.matvec(normed));
            layer_selected_ids.push(routing.experts.clone());
            layer_selected_weights.push(routing.weights.clone());

            let token_idx =
                (pos as u64).wrapping_mul(self.config.num_layers as u64) + layer_idx as u64;
            engine
                .moe_step_weighted_into_with_timing(
                    token_idx,
                    layer_idx as u32,
                    normed,
                    &layer_scratch.global_expert_ids,
                    &routing.weights,
                    &mut layer_scratch.moe_accum,
                    timings,
                )
                .await?;
            layer.moe_accumulated_into_with_timing(
                &x,
                &layer_scratch.moe_accum,
                &mut next_x,
                timings,
            );
            std::mem::swap(&mut x, &mut next_x);
            next_x.clear();
            layer_scratch.routing.recycle_decision(routing);

            match layer.shared_expert_forward_with_timing(normed, timings) {
                Ok(Some(shared)) => {
                    crate::transformer::add_residual_into(&x, &shared, &mut next_x);
                    std::mem::swap(&mut x, &mut next_x);
                    next_x.clear();
                }
                Ok(None) => {}
                Err(source) => {
                    if !allow_degraded {
                        return Err(RealInferenceError::ResidentFfn {
                            layer: layer_idx,
                            role: ResidentFfnRole::Shared,
                            source,
                        });
                    }
                    engine.record_degraded_expert_substitution();
                    tracing::warn!(
                        layer = layer_idx,
                        error = %source,
                        "DEGRADED MODE: shared expert failed; dropping its \
                         contribution (allow_degraded_experts = true, output is \
                         non-authoritative)"
                    );
                }
            }
            layer_post_moe.push(x.clone());
        }

        let final_norm = crate::stage_timing::time_optional(
            timings,
            crate::stage_timing::FINAL_RMS_NORM,
            || self.final_rms.forward(&x),
        );
        let logits = self.lm_head.diagnostic_greedy_logits(&final_norm);
        let sampled_token =
            self.sample_hidden(&final_norm, &crate::sampling::SamplingParams::greedy(), pos);

        Ok(crate::gpu_native_diagnostics::ModelDiagnosticTrace {
            embedding: embedding_trace,
            layer_post_attn,
            layer_router_input,
            layer_router_logits,
            layer_selected_ids,
            layer_selected_weights,
            layer_post_moe,
            final_norm,
            logits,
            sampled_token,
        })
    }

    /// Sample a next-token id from an already-computed final hidden state.
    ///
    /// Sampling is delegated to [`crate::sampling::sample`], so
    /// `temperature == 0.0` reproduces the original deterministic
    /// `argmax` behaviour bit-for-bit.
    pub fn sample_hidden(
        &self,
        hidden: &[f32],
        params: &crate::sampling::SamplingParams,
        pos: usize,
    ) -> u32 {
        self.lm_head.sample(hidden, params, pos as u64)
    }

    /// Timed variant of [`Self::sample_hidden`] (hardening pass, D3):
    /// delegates to [`crate::transformer::LMHead::sample_with_timing`],
    /// which is the *same* code path as the untimed sampler with
    /// per-stage timing recorded inside each fast path. Timing on/off
    /// therefore selects the same algorithm and returns the same
    /// token, and timed greedy / top-K-only sampling never allocates
    /// full logits.
    pub fn sample_hidden_with_timing(
        &self,
        hidden: &[f32],
        params: &crate::sampling::SamplingParams,
        pos: usize,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> u32 {
        self.lm_head
            .sample_with_timing(hidden, params, pos as u64, timings)
    }

    /// Ingest one token and sample the following token. This is the
    /// decode-step half of the split API and is equivalent to the old
    /// `step` behavior.
    pub async fn decode_step(
        &self,
        engine: &Arc<Engine>,
        token_id: u32,
        pos: usize,
        kv: &mut [KvCache],
        params: &crate::sampling::SamplingParams,
    ) -> Result<u32, RealInferenceError> {
        let hidden = self.forward_token_hidden(engine, token_id, pos, kv).await?;
        Ok(self.sample_hidden(&hidden, params, pos))
    }

    pub async fn decode_step_with_timing(
        &self,
        engine: &Arc<Engine>,
        token_id: u32,
        pos: usize,
        kv: &mut [KvCache],
        params: &crate::sampling::SamplingParams,
        timings: Option<&crate::stage_timing::StageTimings>,
    ) -> Result<u32, RealInferenceError> {
        let hidden = self
            .forward_token_hidden_with_timing(engine, token_id, pos, kv, timings)
            .await?;
        Ok(self.sample_hidden_with_timing(&hidden, params, pos, timings))
    }

    /// Backwards-compatible alias for callers that still express a full
    /// decoder step as "ingest one token and return the next token".
    pub async fn step(
        &self,
        engine: &Arc<Engine>,
        token_id: u32,
        pos: usize,
        kv: &mut [KvCache],
        params: &crate::sampling::SamplingParams,
    ) -> Result<u32, RealInferenceError> {
        self.decode_step(engine, token_id, pos, kv, params).await
    }
}

/// Small `splitmix64` PRNG so we can produce deterministic, dependency-free
/// weight initialisations.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E3779B97F4A7C15),
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Draw an `f32` in `[-scale, scale]`.
    fn next_uniform(&mut self, scale: f32) -> f32 {
        let bits = self.next_u64();
        let u = ((bits >> 40) as u32 & ((1 << 23) - 1)) as f32 / ((1u32 << 23) as f32);
        (u * 2.0 - 1.0) * scale
    }
}

fn sample_uniform_vec(rng: &mut SplitMix64, n: usize, scale: f32) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        v.push(rng.next_uniform(scale));
    }
    v
}

/// Build a deterministically seeded [`MultiHeadLatentAttention`] for a
/// single DeepSeek-V3 layer, or `None` when the config carries no MLA
/// dims (every non-DeepSeek architecture). The seeded weights let the
/// engine run DeepSeek-V3 end-to-end through the latent-attention path
/// even without an on-disk checkpoint, exactly mirroring how the
/// standard attention path is seeded in [`RealModel::new_seeded`]; the
/// on-disk loaders overwrite these with real tensors when present.
fn seed_mla(
    config: &RealModelConfig,
    rng: &mut SplitMix64,
    proj_scale: f32,
) -> Option<MultiHeadLatentAttention> {
    let dims = config.advanced.mla?;
    assert!(dims.kv_lora_rank > 0, "MLA kv_lora_rank must be > 0");
    assert!(dims.v_head_dim > 0, "MLA v_head_dim must be > 0");
    assert!(
        dims.qk_rope_head_dim % 2 == 0,
        "MLA qk_rope_head_dim must be even for RoPE"
    );
    let n_h = config.num_heads;
    let qk_head = dims.qk_nope_head_dim + dims.qk_rope_head_dim;
    let q_total = n_h * qk_head;
    let kv_proj_dim = dims.kv_lora_rank + dims.qk_rope_head_dim;
    let kv_b_out = n_h * (dims.qk_nope_head_dim + dims.v_head_dim);

    let (q_a_proj, q_a_layernorm, q_b_proj) = if dims.q_lora_rank > 0 {
        (
            sample_uniform_vec(rng, dims.q_lora_rank * config.d_model, proj_scale),
            Some(RmsNorm::new(vec![1.0; dims.q_lora_rank], config.rms_eps)),
            sample_uniform_vec(rng, q_total * dims.q_lora_rank, proj_scale),
        )
    } else {
        (
            Vec::new(),
            None,
            sample_uniform_vec(rng, q_total * config.d_model, proj_scale),
        )
    };

    Some(MultiHeadLatentAttention {
        d_model: config.d_model,
        num_heads: n_h,
        q_lora_rank: dims.q_lora_rank,
        kv_lora_rank: dims.kv_lora_rank,
        qk_nope_head_dim: dims.qk_nope_head_dim,
        qk_rope_head_dim: dims.qk_rope_head_dim,
        v_head_dim: dims.v_head_dim,
        rope_base: config.rope_base,
        // YaRN long-context scaling (DeepSeek-V3 ships `rope_scaling`
        // of type "yarn"): blended inverse frequencies over the rotary
        // portion plus the `mscale` attention-magnitude corrections.
        rope_yarn: config
            .advanced
            .rope_scaling
            .as_ref()
            .and_then(|s| YarnRope::from_scaling(dims.qk_rope_head_dim, config.rope_base, s)),
        softmax_scale: MultiHeadLatentAttention::yarn_softmax_scale(
            dims.qk_nope_head_dim,
            dims.qk_rope_head_dim,
            config.advanced.rope_scaling.as_ref(),
        ),
        q_a_proj,
        q_a_layernorm,
        q_b_proj,
        kv_a_proj_with_mqa: sample_uniform_vec(rng, kv_proj_dim * config.d_model, proj_scale),
        kv_a_layernorm: RmsNorm::new(vec![1.0; dims.kv_lora_rank], config.rms_eps),
        kv_b_proj: sample_uniform_vec(rng, kv_b_out * dims.kv_lora_rank, proj_scale),
        o_proj: sample_uniform_vec(rng, config.d_model * n_h * dims.v_head_dim, proj_scale),
    })
}

/// DeepSeek-V3 FP8 block-quantisation edge (`weight_scale_inv` is laid
/// out over a 128x128 block grid).
const FP8_BLOCK: usize = 128;

/// True when a dense manifest entry's recorded `dims` describe exactly the
/// `[rows, cols]` orientation the loader is about to impose (Finding 3).
/// The converter records 2-D tensors as `[rows, cols]` and 1-D tensors as
/// `[n, 1]`; the loader always constructs the `DenseWeight` from the
/// caller's `(rows, cols)`. Requiring an exact match rejects a transposed
/// `[cols, rows]` entry whose element count coincidentally matches.
fn dense_dims_match_orientation(dims: &[usize], rows: usize, cols: usize) -> bool {
    match dims {
        [r, c] => *r == rows && *c == cols,
        // A bare `[n]` entry is orientation-free; accept it when the single
        // dimension is the row count and the caller expects a column vector.
        [n] => *n == rows && cols == 1,
        _ => false,
    }
}

/// True when a safetensors tensor's stored `shape` is compatible with the
/// expected `[rows, cols]` matrix orientation (Finding 3 / Finding 4).
///
/// HF stores Linear weights as `[out_features, in_features]`, matching the
/// engine's `[rows, cols]`. A matrix load must therefore be exactly rank-2
/// with the expected orientation: a `[cols, rows]` transpose, a flattened
/// rank-1 `[rows*cols]` tensor, or any rank > 2 tensor is rejected outright
/// rather than being accepted on an element-count coincidence. Rank-1
/// compatibility lives only in the vector-specific loaders (`find_f32`),
/// never in matrix loaders.
fn safetensor_orientation_ok(shape: &[usize], rows: usize, cols: usize) -> bool {
    matches!(shape, [a, b] if *a == rows && *b == cols)
}

/// Locate `name` across shards and, when found as a 2-D tensor, verify its
/// stored orientation matches `[rows, cols]`. Returns `Some(reason)` when the
/// tensor exists but is transposed/misshaped, `None` when it is orientation-
/// compatible or absent (absence is handled by the normal missing-tensor
/// path).
fn safetensor_orientation_problem(
    parsed: &[safetensors::SafeTensors<'_>],
    name: &str,
    rows: usize,
    cols: usize,
) -> Option<String> {
    for st in parsed {
        if let Ok(view) = st.tensor(name) {
            let shape = view.shape();
            if !safetensor_orientation_ok(shape, rows, cols) {
                return Some(format!(
                    "tensor {name} has shape {shape:?}, expected [rows, cols] = [{rows}, {cols}] \
                     (transposed or misshaped)"
                ));
            }
            return None;
        }
    }
    None
}

/// Outcome of attempting to load the optional `dense_manifest.json`.
enum DenseManifestLoad {
    /// No manifest file present — legacy per-file layout. Not an error.
    Absent,
    /// The manifest exists but could not be read or parsed. Under strict
    /// loading this is a hard failure (Finding 6): a corrupt manifest must
    /// not silently degrade to seeded weights.
    Malformed(String),
    /// Successfully parsed manifest (structural integrity not yet checked).
    Loaded(DenseTensorManifest),
}

fn load_dense_manifest(dir: &Path) -> DenseManifestLoad {
    let path = dir.join("dense_manifest.json");
    if !path.is_file() {
        return DenseManifestLoad::Absent;
    }
    match std::fs::read_to_string(&path) {
        Ok(body) => match serde_json::from_str::<DenseTensorManifest>(&body) {
            Ok(manifest) => DenseManifestLoad::Loaded(manifest),
            Err(err) => {
                warn!(file = %path.display(), error = %err, "dense manifest parse failed");
                DenseManifestLoad::Malformed(format!("parse error: {err}"))
            }
        },
        Err(err) => {
            warn!(file = %path.display(), error = %err, "dense manifest read failed");
            DenseManifestLoad::Malformed(format!("read error: {err}"))
        }
    }
}

/// Resolve the dense manifest for a checkpoint directory, applying the
/// Finding 6 integrity policy. Returns the manifest only when it is present
/// and passes validation; any structural problem is recorded in `failures`
/// (fatal under strict loading) and the manifest is dropped so the loader
/// falls back to the per-file / seeded paths.
fn resolve_dense_manifest(
    dir: &Path,
    strict_weights: bool,
    failures: &mut Vec<WeightLoadFailure>,
) -> Option<DenseTensorManifest> {
    let push_problem = |failures: &mut Vec<WeightLoadFailure>, detail: String| {
        failures.push(WeightLoadFailure {
            tensor: "dense_manifest.json".to_string(),
            group: GROUP_DENSE_MANIFEST,
            kind: WeightLoadFailureKind::Malformed,
            expected: "valid dense manifest".to_string(),
            actual: None,
            detail: Some(detail),
        });
    };

    match load_dense_manifest(dir) {
        DenseManifestLoad::Absent => None,
        DenseManifestLoad::Malformed(err) => {
            if strict_weights {
                push_problem(failures, err);
            }
            None
        }
        DenseManifestLoad::Loaded(manifest) => {
            let problems = manifest.validate_integrity(strict_weights);
            if problems.is_empty() {
                return Some(manifest);
            }
            for problem in &problems {
                warn!(dir = %dir.display(), problem = %problem, "dense manifest integrity problem");
            }
            if strict_weights {
                for problem in problems {
                    push_problem(failures, problem);
                }
            }
            // Never consult a structurally invalid manifest, even in
            // non-strict mode: an ambiguous alias or a bad byte_len could
            // load the wrong tensor. Fall back to per-file / seeded init.
            None
        }
    }
}

fn try_load_resident_f32(
    dir: &Path,
    manifest: Option<&DenseTensorManifest>,
    name: &str,
    expected: usize,
    group: &'static str,
    strict_weights: bool,
    failures: &mut Vec<WeightLoadFailure>,
    summary: &mut WeightLoadSummary,
) -> Option<Vec<f32>> {
    try_load_dense_weight(
        dir,
        manifest,
        name,
        expected,
        1,
        group,
        strict_weights,
        failures,
        summary,
    )
    .map(|weight| weight.to_f32_vec())
}

#[allow(clippy::too_many_arguments)]
fn try_load_dense_weight(
    dir: &Path,
    manifest: Option<&DenseTensorManifest>,
    name: &str,
    rows: usize,
    cols: usize,
    group: &'static str,
    strict_weights: bool,
    failures: &mut Vec<WeightLoadFailure>,
    summary: &mut WeightLoadSummary,
) -> Option<DenseWeight> {
    let expected = rows.saturating_mul(cols);
    let Some(entry) = manifest.and_then(|m| m.find_alias(name)) else {
        return try_load_f32_bin(
            dir,
            name,
            expected,
            group,
            strict_weights,
            failures,
            summary,
        )
        .map(|v| DenseWeight::from_f32(v, rows, cols));
    };

    // Tied entry (Finding 2): an entry such as a tied LM head carries no file
    // of its own — it is reconstructed from a referenced source tensor (the
    // token embedding). Resolve `tied_to` to the target entry and load its
    // bytes, validating the reconstructed weight against the caller's expected
    // `(rows, cols)` exactly as for a normal entry.
    let entry = if let Some(target) = entry.tied_to.as_deref() {
        match manifest.and_then(|m| m.find_alias(target)) {
            Some(source) => source,
            None => {
                if strict_weights {
                    failures.push(WeightLoadFailure {
                        tensor: name.to_string(),
                        group,
                        kind: WeightLoadFailureKind::Missing,
                        expected: format!("tied source tensor {target}"),
                        actual: None,
                        detail: Some(format!(
                            "dense manifest entry {} is tied to {target}, which is not in the manifest",
                            entry.canonical_name
                        )),
                    });
                }
                warn!(
                    tensor = name,
                    tied_to = target,
                    "tied dense manifest target not found; falling back to seeded init"
                );
                return None;
            }
        }
    } else {
        entry
    };

    if entry.dims.iter().product::<usize>() != expected {
        if strict_weights {
            failures.push(WeightLoadFailure {
                tensor: name.to_string(),
                group,
                kind: WeightLoadFailureKind::ShapeMismatch,
                expected: format!("{rows}x{cols} ({expected} elements)"),
                actual: Some(format!("{:?}", entry.dims)),
                detail: Some(format!(
                    "dense manifest entry {} has incompatible dims",
                    entry.canonical_name
                )),
            });
        }
        warn!(
            tensor = name,
            canonical = %entry.canonical_name,
            dims = ?entry.dims,
            rows,
            cols,
            "dense manifest shape mismatch; falling back to seeded init"
        );
        return None;
    }

    // Orientation check (Finding 3). The element count matching is *not*
    // sufficient: a `[cols, rows]` entry has the same product as the
    // expected `[rows, cols]`, but its row-major bytes decode to the
    // transpose of the weight the engine expects. Because the loader
    // constructs the `DenseWeight` from the caller's `(rows, cols)` and
    // never consults the manifest dims for layout, a transposed entry
    // would load silently corrupted. Require the recorded dims to match
    // the expected orientation exactly.
    if !dense_dims_match_orientation(&entry.dims, rows, cols) {
        if strict_weights {
            failures.push(WeightLoadFailure {
                tensor: name.to_string(),
                group,
                kind: WeightLoadFailureKind::ShapeMismatch,
                expected: format!("dims [{rows}, {cols}] (row-major)"),
                actual: Some(format!("{:?}", entry.dims)),
                detail: Some(format!(
                    "dense manifest entry {} shape orientation mismatch; \
                     expected [rows, cols] = [{rows}, {cols}]",
                    entry.canonical_name
                )),
            });
        }
        warn!(
            tensor = name,
            canonical = %entry.canonical_name,
            dims = ?entry.dims,
            rows,
            cols,
            "dense manifest shape orientation mismatch; falling back to seeded init"
        );
        return None;
    }

    let path = dir.join(&entry.file);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(err) => {
            if strict_weights {
                failures.push(WeightLoadFailure {
                    tensor: name.to_string(),
                    group,
                    kind: WeightLoadFailureKind::Unreadable,
                    expected: format!("dense manifest file {}", entry.file),
                    actual: None,
                    detail: Some(err.to_string()),
                });
            }
            warn!(file = %path.display(), error = %err, "dense manifest file read failed");
            return None;
        }
    };
    if bytes.len() != entry.byte_len {
        if strict_weights {
            failures.push(WeightLoadFailure {
                tensor: name.to_string(),
                group,
                kind: WeightLoadFailureKind::ShapeMismatch,
                expected: format!("{} bytes", entry.byte_len),
                actual: Some(format!("{} bytes", bytes.len())),
                detail: Some(format!(
                    "dense manifest entry {} byte_len mismatch",
                    entry.canonical_name
                )),
            });
        }
        warn!(
            file = %path.display(),
            have = bytes.len(),
            need = entry.byte_len,
            "dense manifest byte length mismatch; falling back to seeded init"
        );
        return None;
    }
    if let Some(expected_checksum) = entry.checksum.as_ref() {
        let actual = dense_checksum(&bytes);
        if &actual != expected_checksum {
            if strict_weights {
                failures.push(WeightLoadFailure {
                    tensor: name.to_string(),
                    group,
                    kind: WeightLoadFailureKind::Malformed,
                    expected: expected_checksum.clone(),
                    actual: Some(actual.clone()),
                    detail: Some(format!(
                        "dense manifest entry {} checksum mismatch",
                        entry.canonical_name
                    )),
                });
            }
            warn!(
                file = %path.display(),
                expected = %expected_checksum,
                actual = %actual,
                "dense manifest checksum mismatch; falling back to seeded init"
            );
            return None;
        }
    }

    match entry.dtype {
        DenseDType::F32 => {
            if bytes.len() != expected.saturating_mul(4) || bytes.len() % 4 != 0 {
                if strict_weights {
                    failures.push(WeightLoadFailure {
                        tensor: name.to_string(),
                        group,
                        kind: WeightLoadFailureKind::ShapeMismatch,
                        expected: format!("{expected} f32 elements"),
                        actual: Some(format!("{} bytes", bytes.len())),
                        detail: Some(format!(
                            "dense manifest entry {} f32 payload size mismatch",
                            entry.canonical_name
                        )),
                    });
                }
                return None;
            }
            let mut values = Vec::with_capacity(expected);
            for chunk in bytes.chunks_exact(4) {
                values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            summary.record(group, entry.dtype.as_str(), values.len() * 4);
            Some(DenseWeight::from_f32(values, rows, cols))
        }
        DenseDType::Q8_0 => match DenseWeight::from_q8_0_bytes(bytes, rows, cols) {
            Ok(weight) => {
                summary.record(group, entry.dtype.as_str(), weight.resident_bytes());
                Some(weight)
            }
            Err(err) => {
                if strict_weights {
                    failures.push(WeightLoadFailure {
                        tensor: name.to_string(),
                        group,
                        kind: WeightLoadFailureKind::ShapeMismatch,
                        expected: format!("{rows}x{cols} q8_0 payload"),
                        actual: Some(err.to_string()),
                        detail: Some(format!(
                            "dense manifest entry {} q8_0 payload mismatch",
                            entry.canonical_name
                        )),
                    });
                }
                warn!(file = %path.display(), error = %err, "dense Q8_0 load failed");
                None
            }
        },
    }
}

fn try_load_f32_bin(
    dir: &Path,
    name: &str,
    expected: usize,
    group: &'static str,
    strict_weights: bool,
    failures: &mut Vec<WeightLoadFailure>,
    summary: &mut WeightLoadSummary,
) -> Option<Vec<f32>> {
    let path = dir.join(name);
    if !path.is_file() {
        if strict_weights {
            failures.push(WeightLoadFailure::missing(
                name,
                group,
                format!("{expected} f32 elements"),
            ));
        }
        return None;
    }
    match std::fs::read(&path) {
        Ok(bytes) => {
            let n = bytes.len() / 4;
            if strict_weights {
                let expected_bytes = expected.saturating_mul(4);
                if bytes.len() != expected_bytes || bytes.len() % 4 != 0 {
                    let kind = if bytes.len() % 4 != 0 {
                        WeightLoadFailureKind::Malformed
                    } else {
                        WeightLoadFailureKind::ShapeMismatch
                    };
                    failures.push(WeightLoadFailure {
                        tensor: name.to_string(),
                        group,
                        kind,
                        expected: format!("{expected} f32 elements ({expected_bytes} bytes)"),
                        actual: Some(format!("{} bytes", bytes.len())),
                        detail: None,
                    });
                    return None;
                }
            }
            if n < expected {
                warn!(
                    file = %path.display(),
                    have = n,
                    need = expected,
                    "weight file shorter than expected; falling back to seeded init"
                );
                return None;
            }
            let mut floats = Vec::with_capacity(expected);
            for chunk in bytes[..expected * 4].chunks_exact(4) {
                floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            summary.record(group, "f32", floats.len() * 4);
            Some(floats)
        }
        Err(e) => {
            if strict_weights {
                failures.push(WeightLoadFailure {
                    tensor: name.to_string(),
                    group,
                    kind: WeightLoadFailureKind::Unreadable,
                    expected: format!("{expected} f32 elements"),
                    actual: None,
                    detail: Some(e.to_string()),
                });
            }
            warn!(file = %path.display(), error = %e, "weight file read failed");
            None
        }
    }
}

fn safetensor_name_is_fp8_ignored(fp8_ignored: &[String], name: &str) -> bool {
    fp8_ignored.iter().any(|ignored| {
        let ignored = ignored.as_str();
        name.match_indices(ignored).any(|(idx, _)| {
            let after = &name[idx + ignored.len()..];
            after.is_empty() || after.starts_with('.') || after.starts_with('_')
        })
    })
}

fn safetensor_source_dtype(parsed: &[safetensors::SafeTensors<'_>], name: &str) -> String {
    for st in parsed {
        if let Ok(view) = st.tensor(name) {
            return format!("{:?}", view.dtype()).to_ascii_lowercase();
        }
    }
    "unknown".to_string()
}

fn safetensor_present(parsed: &[safetensors::SafeTensors<'_>], name: &str) -> bool {
    parsed.iter().any(|st| st.tensor(name).is_ok())
}

fn tensor_names_expected(names: &[String], expected: &str) -> String {
    if names.len() == 1 {
        expected.to_string()
    } else {
        format!("{expected}; one of [{}]", names.join(", "))
    }
}

fn strict_safetensors_failures(
    dir: &Path,
    parsed: &[safetensors::SafeTensors<'_>],
    model: &RealModel,
    config: &RealModelConfig,
    naming: &TensorNaming,
    fp8_block: usize,
    fp8_ignored: &[String],
) -> Vec<WeightLoadFailure> {
    let mut failures = Vec::new();

    let require = |name: String, expected: usize, group: &'static str, failures: &mut Vec<_>| {
        strict_validate_safetensor(
            parsed,
            &name,
            expected,
            group,
            fp8_block,
            fp8_ignored,
            failures,
        );
    };

    // 2-D matrix requirement: element-count check *plus* the Finding 3
    // orientation guard, so a transposed non-square projection is rejected
    // under strict loading instead of loading silently corrupted.
    let require_2d =
        |name: String, rows: usize, cols: usize, group: &'static str, failures: &mut Vec<_>| {
            let ok = strict_validate_safetensor(
                parsed,
                &name,
                rows.saturating_mul(cols),
                group,
                fp8_block,
                fp8_ignored,
                failures,
            );
            if ok {
                if let Some(reason) = safetensor_orientation_problem(parsed, &name, rows, cols) {
                    failures.push(WeightLoadFailure {
                        tensor: name,
                        group,
                        kind: WeightLoadFailureKind::ShapeMismatch,
                        expected: format!("[rows, cols] = [{rows}, {cols}]"),
                        actual: None,
                        detail: Some(reason),
                    });
                }
            }
        };

    let d_model = config.d_model;
    let q_dim = config.num_heads * config.head_dim;
    let attn_out_dim = config.num_heads * config.v_head_dim();

    require_2d(
        naming.embed(),
        config.vocab_size,
        d_model,
        GROUP_EMBEDDING,
        &mut failures,
    );
    require(naming.final_norm(), d_model, GROUP_NORMS, &mut failures);
    require_2d(
        naming.lm_head(),
        config.vocab_size,
        d_model,
        GROUP_LM_HEAD,
        &mut failures,
    );

    for l in 0..config.num_layers {
        let kv_dim = model.layers[l].attn.kv_dim();
        let v_proj_dim = model.layers[l].attn.v_proj_dim();
        require(
            naming.input_layernorm(l),
            d_model,
            GROUP_NORMS,
            &mut failures,
        );
        require(
            naming.post_attention_layernorm(l),
            d_model,
            GROUP_NORMS,
            &mut failures,
        );

        if let Some(mla) = model.layers[l].mla.as_ref() {
            let qk_head = mla.qk_nope_head_dim + mla.qk_rope_head_dim;
            let q_total = mla.num_heads * qk_head;
            let kv_proj_dim = mla.kv_lora_rank + mla.qk_rope_head_dim;
            let kv_b_out = mla.num_heads * (mla.qk_nope_head_dim + mla.v_head_dim);
            if mla.q_lora_rank > 0 {
                require(
                    naming.mla_q_a_proj(l),
                    mla.q_lora_rank * d_model,
                    GROUP_ATTENTION,
                    &mut failures,
                );
                require(
                    naming.mla_q_a_layernorm(l),
                    mla.q_lora_rank,
                    GROUP_NORMS,
                    &mut failures,
                );
                require(
                    naming.mla_q_b_proj(l),
                    q_total * mla.q_lora_rank,
                    GROUP_ATTENTION,
                    &mut failures,
                );
            } else {
                require(
                    naming.attn_q(l),
                    q_total * d_model,
                    GROUP_ATTENTION,
                    &mut failures,
                );
            }
            require(
                naming.mla_kv_a_proj(l),
                kv_proj_dim * d_model,
                GROUP_ATTENTION,
                &mut failures,
            );
            require(
                naming.mla_kv_a_layernorm(l),
                mla.kv_lora_rank,
                GROUP_NORMS,
                &mut failures,
            );
            require(
                naming.mla_kv_b_proj(l),
                kv_b_out * mla.kv_lora_rank,
                GROUP_ATTENTION,
                &mut failures,
            );
            require(
                naming.attn_o(l),
                d_model * mla.num_heads * mla.v_head_dim,
                GROUP_ATTENTION,
                &mut failures,
            );
        } else if naming.attn_qkv_fused() {
            require_2d(
                naming.attn_qkv(l),
                q_dim + 2 * kv_dim,
                d_model,
                GROUP_ATTENTION,
                &mut failures,
            );
            require_2d(
                naming.attn_o(l),
                d_model,
                q_dim,
                GROUP_ATTENTION,
                &mut failures,
            );
        } else {
            require_2d(
                naming.attn_q(l),
                q_dim,
                d_model,
                GROUP_ATTENTION,
                &mut failures,
            );
            require_2d(
                naming.attn_k(l),
                kv_dim,
                d_model,
                GROUP_ATTENTION,
                &mut failures,
            );
            require_2d(
                naming.attn_v(l),
                v_proj_dim,
                d_model,
                GROUP_ATTENTION,
                &mut failures,
            );
            require_2d(
                naming.attn_o(l),
                d_model,
                attn_out_dim,
                GROUP_ATTENTION,
                &mut failures,
            );
        }

        if config.architecture.uses_qk_norm() {
            require(
                naming.attn_q_norm(l),
                config.head_dim,
                GROUP_NORMS,
                &mut failures,
            );
            require(
                naming.attn_k_norm(l),
                config.head_dim,
                GROUP_NORMS,
                &mut failures,
            );
        }

        if config.advanced.attention_bias {
            validate_optional_safetensor(
                parsed,
                naming.q_proj_bias(l),
                q_dim,
                GROUP_ATTENTION,
                fp8_block,
                fp8_ignored,
                &mut failures,
            );
            validate_optional_safetensor(
                parsed,
                naming.k_proj_bias(l),
                kv_dim,
                GROUP_ATTENTION,
                fp8_block,
                fp8_ignored,
                &mut failures,
            );
            validate_optional_safetensor(
                parsed,
                naming.v_proj_bias(l),
                v_proj_dim,
                GROUP_ATTENTION,
                fp8_block,
                fp8_ignored,
                &mut failures,
            );
            validate_optional_safetensor(
                parsed,
                naming.o_proj_bias(l),
                d_model,
                GROUP_ATTENTION,
                fp8_block,
                fp8_ignored,
                &mut failures,
            );
        }
        if config.advanced.add_swa_attention_sink_bias && model.layers[l].attn.window_size.is_some()
        {
            validate_optional_safetensor(
                parsed,
                naming.attn_sink_bias(l),
                config.num_heads,
                GROUP_ATTENTION,
                fp8_block,
                fp8_ignored,
                &mut failures,
            );
        }

        match naming.ffn_kind(l) {
            FfnKind::Dense => validate_dense_ffn_strict(
                parsed,
                naming,
                l,
                d_model,
                config.d_ff,
                fp8_block,
                fp8_ignored,
                &mut failures,
            ),
            FfnKind::Moe if config.architecture.is_moe() => {
                validate_moe_gate_strict(
                    dir,
                    parsed,
                    &naming.moe_gate(l),
                    &format!("gate_{l}.bin"),
                    config.num_experts * d_model,
                    fp8_block,
                    fp8_ignored,
                    &mut failures,
                );
                validate_optional_safetensor(
                    parsed,
                    naming.moe_gate_correction_bias(l),
                    config.num_experts,
                    GROUP_ROUTING_GATES,
                    fp8_block,
                    fp8_ignored,
                    &mut failures,
                );
            }
            _ => {}
        }

        if config.advanced.num_shared_experts > 0 {
            validate_shared_expert_strict(
                parsed,
                naming,
                l,
                d_model,
                fp8_block,
                fp8_ignored,
                &mut failures,
            );
        }
    }

    failures
}

fn validate_dense_ffn_strict(
    parsed: &[safetensors::SafeTensors<'_>],
    naming: &TensorNaming,
    layer: usize,
    d_model: usize,
    d_ff: usize,
    fp8_block: usize,
    fp8_ignored: &[String],
    failures: &mut Vec<WeightLoadFailure>,
) {
    if naming.mlp_gate_up_fused() {
        strict_validate_safetensor(
            parsed,
            &naming.mlp_gate_up(layer),
            2 * d_ff * d_model,
            GROUP_DENSE_FFN,
            fp8_block,
            fp8_ignored,
            failures,
        );
        strict_validate_safetensor(
            parsed,
            &naming.mlp_down(layer),
            d_model * d_ff,
            GROUP_DENSE_FFN,
            fp8_block,
            fp8_ignored,
            failures,
        );
    } else {
        strict_validate_safetensor(
            parsed,
            &naming.mlp_gate(layer),
            d_ff * d_model,
            GROUP_DENSE_FFN,
            fp8_block,
            fp8_ignored,
            failures,
        );
        strict_validate_safetensor(
            parsed,
            &naming.mlp_up(layer),
            d_ff * d_model,
            GROUP_DENSE_FFN,
            fp8_block,
            fp8_ignored,
            failures,
        );
        strict_validate_safetensor(
            parsed,
            &naming.mlp_down(layer),
            d_model * d_ff,
            GROUP_DENSE_FFN,
            fp8_block,
            fp8_ignored,
            failures,
        );
    }
}

fn validate_shared_expert_strict(
    parsed: &[safetensors::SafeTensors<'_>],
    naming: &TensorNaming,
    layer: usize,
    d_model: usize,
    fp8_block: usize,
    fp8_ignored: &[String],
    failures: &mut Vec<WeightLoadFailure>,
) {
    let p = naming.prefix();
    let gate_names = [
        format!("{p}model.layers.{layer}.mlp.shared_expert.gate_proj.weight"),
        format!("{p}model.layers.{layer}.mlp.shared_experts.gate_proj.weight"),
    ];
    let up_names = [
        format!("{p}model.layers.{layer}.mlp.shared_expert.up_proj.weight"),
        format!("{p}model.layers.{layer}.mlp.shared_experts.up_proj.weight"),
    ];
    let down_names = [
        format!("{p}model.layers.{layer}.mlp.shared_expert.down_proj.weight"),
        format!("{p}model.layers.{layer}.mlp.shared_experts.down_proj.weight"),
    ];

    let Some((gate_name, gate_len)) = strict_find_any_safetensor_len(
        parsed,
        &gate_names,
        "non-empty multiple of d_model",
        GROUP_SHARED_FFN,
        fp8_block,
        fp8_ignored,
        failures,
    ) else {
        return;
    };
    let Some((up_name, up_len)) = strict_find_any_safetensor_len(
        parsed,
        &up_names,
        "same length as shared expert gate",
        GROUP_SHARED_FFN,
        fp8_block,
        fp8_ignored,
        failures,
    ) else {
        return;
    };
    let Some((down_name, down_len)) = strict_find_any_safetensor_len(
        parsed,
        &down_names,
        "d_model * shared_d_ff",
        GROUP_SHARED_FFN,
        fp8_block,
        fp8_ignored,
        failures,
    ) else {
        return;
    };

    if d_model == 0 || gate_len == 0 || gate_len % d_model != 0 {
        failures.push(WeightLoadFailure {
            tensor: gate_name,
            group: GROUP_SHARED_FFN,
            kind: WeightLoadFailureKind::ShapeMismatch,
            expected: "non-empty multiple of d_model".to_string(),
            actual: Some(format!("{gate_len} f32 elements")),
            detail: None,
        });
        return;
    }
    let shared_d_ff = gate_len / d_model;
    if up_len != gate_len {
        failures.push(WeightLoadFailure {
            tensor: up_name,
            group: GROUP_SHARED_FFN,
            kind: WeightLoadFailureKind::ShapeMismatch,
            expected: format!("{gate_len} f32 elements"),
            actual: Some(format!("{up_len} f32 elements")),
            detail: Some("shared expert up projection must match gate projection".to_string()),
        });
    }
    let expected_down = d_model * shared_d_ff;
    if down_len != expected_down {
        failures.push(WeightLoadFailure {
            tensor: down_name,
            group: GROUP_SHARED_FFN,
            kind: WeightLoadFailureKind::ShapeMismatch,
            expected: format!("{expected_down} f32 elements"),
            actual: Some(format!("{down_len} f32 elements")),
            detail: Some("shared expert down projection must be d_model * shared_d_ff".to_string()),
        });
    }

    validate_optional_safetensor_any(
        parsed,
        &[format!(
            "{p}model.layers.{layer}.mlp.shared_expert_gate.weight"
        )],
        d_model,
        GROUP_SHARED_FFN,
        fp8_block,
        fp8_ignored,
        failures,
    );
}

fn validate_moe_gate_strict(
    dir: &Path,
    parsed: &[safetensors::SafeTensors<'_>],
    safetensor_name: &str,
    bin_name: &str,
    expected: usize,
    fp8_block: usize,
    fp8_ignored: &[String],
    failures: &mut Vec<WeightLoadFailure>,
) {
    let bin_path = dir.join(bin_name);
    if bin_path.is_file() {
        match std::fs::metadata(&bin_path) {
            Ok(meta) => {
                let len = meta.len() as usize;
                let expected_bytes = expected.saturating_mul(4);
                if len == expected_bytes {
                    // Exact-sized extracted gate sidecar is a valid source.
                    return;
                }
                // A wrong-sized sidecar is never truncated. Warn and fall
                // through to validate the canonical safetensors gate; strict
                // loading still succeeds if that canonical source is valid.
                warn!(
                    "ignoring extracted gate override {bin_name}: {len} bytes is not the \
                     expected {expected} f32 elements ({expected_bytes} bytes); \
                     falling back to canonical {safetensor_name}"
                );
            }
            Err(e) => {
                warn!(
                    "ignoring unreadable extracted gate override {bin_name}: {e}; \
                     falling back to canonical {safetensor_name}"
                );
            }
        }
    }

    strict_validate_safetensor(
        parsed,
        safetensor_name,
        expected,
        GROUP_ROUTING_GATES,
        fp8_block,
        fp8_ignored,
        failures,
    );
}

fn validate_optional_safetensor(
    parsed: &[safetensors::SafeTensors<'_>],
    name: String,
    expected: usize,
    group: &'static str,
    fp8_block: usize,
    fp8_ignored: &[String],
    failures: &mut Vec<WeightLoadFailure>,
) {
    if safetensor_present(parsed, &name) {
        strict_validate_safetensor(
            parsed,
            &name,
            expected,
            group,
            fp8_block,
            fp8_ignored,
            failures,
        );
    }
}

fn validate_optional_safetensor_any(
    parsed: &[safetensors::SafeTensors<'_>],
    names: &[String],
    expected: usize,
    group: &'static str,
    fp8_block: usize,
    fp8_ignored: &[String],
    failures: &mut Vec<WeightLoadFailure>,
) {
    for name in names {
        if safetensor_present(parsed, name) {
            strict_validate_safetensor(
                parsed,
                name,
                expected,
                group,
                fp8_block,
                fp8_ignored,
                failures,
            );
            return;
        }
    }
}

fn strict_find_any_safetensor_len(
    parsed: &[safetensors::SafeTensors<'_>],
    names: &[String],
    expected: &str,
    group: &'static str,
    fp8_block: usize,
    fp8_ignored: &[String],
    failures: &mut Vec<WeightLoadFailure>,
) -> Option<(String, usize)> {
    for name in names {
        if safetensor_present(parsed, name) {
            return strict_safetensor_len(parsed, name, group, fp8_block, fp8_ignored, failures)
                .map(|len| (name.clone(), len));
        }
    }
    failures.push(WeightLoadFailure::missing(
        names.join(" | "),
        group,
        tensor_names_expected(names, expected),
    ));
    None
}

fn strict_validate_safetensor(
    parsed: &[safetensors::SafeTensors<'_>],
    name: &str,
    expected: usize,
    group: &'static str,
    fp8_block: usize,
    fp8_ignored: &[String],
    failures: &mut Vec<WeightLoadFailure>,
) -> bool {
    match strict_safetensor_len(parsed, name, group, fp8_block, fp8_ignored, failures) {
        Some(actual) if actual == expected => true,
        Some(actual) => {
            failures.push(WeightLoadFailure {
                tensor: name.to_string(),
                group,
                kind: WeightLoadFailureKind::ShapeMismatch,
                expected: format!("{expected} f32 elements"),
                actual: Some(format!("{actual} f32 elements")),
                detail: None,
            });
            false
        }
        None => false,
    }
}

fn strict_safetensor_len(
    parsed: &[safetensors::SafeTensors<'_>],
    name: &str,
    group: &'static str,
    fp8_block: usize,
    fp8_ignored: &[String],
    failures: &mut Vec<WeightLoadFailure>,
) -> Option<usize> {
    use safetensors::tensor::Dtype;
    for st in parsed {
        if let Ok(view) = st.tensor(name) {
            let shape = view.shape();
            let n_elem: usize = shape.iter().product();
            return match view.dtype() {
                Dtype::F32 | Dtype::F16 | Dtype::BF16 | Dtype::I8 | Dtype::F8_E5M2 => Some(n_elem),
                Dtype::F8_E4M3 if safetensor_name_is_fp8_ignored(fp8_ignored, name) => {
                    let raw = view.data();
                    if raw.len() % 2 != 0 {
                        failures.push(WeightLoadFailure {
                            tensor: name.to_string(),
                            group,
                            kind: WeightLoadFailureKind::Malformed,
                            expected: "BF16-compatible even byte length".to_string(),
                            actual: Some(format!("{} bytes", raw.len())),
                            detail: Some("FP8 ignored layer is decoded as BF16".to_string()),
                        });
                        None
                    } else {
                        Some(raw.len() / 2)
                    }
                }
                Dtype::F8_E4M3 => {
                    if shape.len() != 2 {
                        failures.push(WeightLoadFailure {
                            tensor: name.to_string(),
                            group,
                            kind: WeightLoadFailureKind::Unsupported,
                            expected: "2D FP8 E4M3 block-quantized weight".to_string(),
                            actual: Some(format!("{shape:?}")),
                            detail: None,
                        });
                        return None;
                    }
                    let rows = shape[0];
                    let cols = shape[1];
                    let scale_name = format!("{name}_scale_inv");
                    let mut scale_len = None;
                    for s in parsed {
                        if let Ok(sv) = s.tensor(&scale_name) {
                            scale_len = Some(sv.shape().iter().product::<usize>());
                            break;
                        }
                    }
                    let want = rows.div_ceil(fp8_block) * cols.div_ceil(fp8_block);
                    match scale_len {
                        Some(actual) if actual == want => Some(n_elem),
                        Some(actual) => {
                            failures.push(WeightLoadFailure {
                                tensor: scale_name,
                                group,
                                kind: WeightLoadFailureKind::ShapeMismatch,
                                expected: format!("{want} f32 scale elements"),
                                actual: Some(format!("{actual} elements")),
                                detail: Some("FP8 companion scale_inv shape mismatch".to_string()),
                            });
                            None
                        }
                        None => {
                            failures.push(WeightLoadFailure::missing(
                                scale_name,
                                group,
                                format!("{want} f32 scale elements"),
                            ));
                            None
                        }
                    }
                }
                Dtype::U8 => {
                    if shape.len() != 2 || shape[0] == 0 {
                        failures.push(WeightLoadFailure {
                            tensor: name.to_string(),
                            group,
                            kind: WeightLoadFailureKind::Unsupported,
                            expected: "2D MXFP4 packed weight".to_string(),
                            actual: Some(format!("{shape:?}")),
                            detail: None,
                        });
                        return None;
                    }
                    let rows = shape[0];
                    let packed_cols = shape[1];
                    let cols = packed_cols * 2;
                    if find_mxfp4_scales(parsed, name, rows, cols).is_none() {
                        failures.push(WeightLoadFailure {
                            tensor: name.to_string(),
                            group,
                            kind: WeightLoadFailureKind::Missing,
                            expected: "MXFP4 companion block scales".to_string(),
                            actual: None,
                            detail: None,
                        });
                        None
                    } else {
                        Some(rows * cols)
                    }
                }
                other => {
                    failures.push(WeightLoadFailure {
                        tensor: name.to_string(),
                        group,
                        kind: WeightLoadFailureKind::Unsupported,
                        expected: "supported resident weight dtype".to_string(),
                        actual: Some(format!("{other:?}")),
                        detail: None,
                    });
                    None
                }
            };
        }
    }
    failures.push(WeightLoadFailure::missing(name, group, "present tensor"));
    None
}

/// Load the on-disk MLA projection tensors for a single DeepSeek-V3
/// layer, overwriting the seeded weights in `layer.mla`. Missing tensors
/// keep their seeded values (matching the rest of the loader's
/// best-effort behaviour). `find` transparently dequantises FP8 weights.
/// Increments `tried`/`loaded` so the loader's summary stays accurate.
fn load_mla_layer<F>(
    layer: &mut TransformerLayer,
    l: usize,
    naming: &TensorNaming,
    config: &RealModelConfig,
    find: &F,
    record: &mut dyn FnMut(&str, &'static str, usize),
    tried: &mut usize,
    loaded: &mut usize,
) where
    F: Fn(&str, usize) -> Option<Vec<f32>>,
{
    let Some(mla) = layer.mla.as_mut() else {
        return;
    };
    let d_model = config.d_model;
    let n_h = mla.num_heads;
    let qk_head = mla.qk_nope_head_dim + mla.qk_rope_head_dim;
    let q_total = n_h * qk_head;
    let kv_proj_dim = mla.kv_lora_rank + mla.qk_rope_head_dim;
    let kv_b_out = n_h * (mla.qk_nope_head_dim + mla.v_head_dim);

    // (name, expected_len) -> Option<Vec<f32>>, counting every attempt.
    let mut try_load = |name: String, expected: usize, group: &'static str| -> Option<Vec<f32>> {
        *tried += 1;
        let v = find(&name, expected);
        if v.is_some() {
            *loaded += 1;
            record(&name, group, expected * 4);
        }
        v
    };

    if mla.q_lora_rank > 0 {
        if let Some(v) = try_load(
            naming.mla_q_a_proj(l),
            mla.q_lora_rank * d_model,
            GROUP_ATTENTION,
        ) {
            mla.q_a_proj = v;
        }
        if let Some(v) = try_load(naming.mla_q_a_layernorm(l), mla.q_lora_rank, GROUP_NORMS) {
            mla.q_a_layernorm = Some(RmsNorm::new(v, config.rms_eps));
        }
        if let Some(v) = try_load(
            naming.mla_q_b_proj(l),
            q_total * mla.q_lora_rank,
            GROUP_ATTENTION,
        ) {
            mla.q_b_proj = v;
        }
    } else {
        // q_lora_rank == 0: a single dense `q_proj` straight from d_model.
        if let Some(v) = try_load(naming.attn_q(l), q_total * d_model, GROUP_ATTENTION) {
            mla.q_b_proj = v;
        }
    }

    if let Some(v) = try_load(
        naming.mla_kv_a_proj(l),
        kv_proj_dim * d_model,
        GROUP_ATTENTION,
    ) {
        mla.kv_a_proj_with_mqa = v;
    }
    if let Some(v) = try_load(naming.mla_kv_a_layernorm(l), mla.kv_lora_rank, GROUP_NORMS) {
        mla.kv_a_layernorm = RmsNorm::new(v, config.rms_eps);
    }
    if let Some(v) = try_load(
        naming.mla_kv_b_proj(l),
        kv_b_out * mla.kv_lora_rank,
        GROUP_ATTENTION,
    ) {
        mla.kv_b_proj = v;
    }
    if let Some(v) = try_load(
        naming.attn_o(l),
        d_model * n_h * mla.v_head_dim,
        GROUP_ATTENTION,
    ) {
        mla.o_proj = v;
    }
}

/// Decode a `safetensors::TensorView` into an owned `Vec<f32>`. We
/// support `f32`, `f16`, and `bf16` source dtypes — the three formats
/// HuggingFace LLM checkpoints actually ship in. `int8` / `int4`
/// quantised checkpoints (AWQ, GPTQ) intentionally aren't supported
/// here: they need full per-tensor zero-points/scales which the
/// upstream extraction pipeline (`extract_mixtral_experts.py`) is the
/// right place to apply. Unknown dtypes fall back to seeded init via
/// the empty `Vec` — the caller's `expected` length check will reject
/// it cleanly.
fn decode_safetensor_to_f32(view: &safetensors::tensor::TensorView<'_>, name: &str) -> Vec<f32> {
    use safetensors::tensor::Dtype;
    let raw = view.data();
    match view.dtype() {
        Dtype::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => raw
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        // Per-tensor INT8 (AWQ-style / older Mixtral quants). The
        // safetensors tensor carries no scale, so we cast `i8 -> f32`
        // here; any companion `<name>_scale` is applied by the caller
        // (`find_f32_dequant`), mirroring the FP8 `_scale_inv` scan.
        Dtype::I8 => raw.iter().map(|&b| b as i8 as f32).collect(),
        // FP8 E5M2 (1-5-2, bias 15). Activation-oriented and carries no
        // companion block scale in current models, so decode each byte
        // standalone. (E4M3 weights still route through the block-wise
        // path in `find_f32_dequant` / `find_f32_any`.)
        Dtype::F8_E5M2 => crate::dequant::dequant_fp8_e5m2(raw),
        other => {
            warn!(
                tensor = name,
                dtype = ?other,
                "unsupported safetensors dtype; falling back to seeded init"
            );
            Vec::new()
        }
    }
}

/// Locate the OCP MXFP4 block-scale tensor that accompanies a packed
/// `*_blocks` weight and return its raw E8M0 bytes (one byte per 32-wide
/// block). GPT-OSS names the scales `<base>.scales` next to a
/// `<base>.blocks` weight (dot-separated, per `openai/gpt-oss`
/// weights.py); we probe that first, then the generic `<name>_scale`,
/// `<name>_scales`, `<name>.scale`, and the underscore `<base>_scales`
/// variant. The companion is only accepted when its element count equals
/// `rows * ceil(cols / 32)`, the E8M0 grid implied by the weight shape.
fn find_mxfp4_scales(
    parsed: &[safetensors::SafeTensors],
    name: &str,
    rows: usize,
    cols: usize,
) -> Option<Vec<u8>> {
    let blocks_per_row = cols.div_ceil(crate::inference::MXFP4_SCALE_BLOCK);
    let want = rows.saturating_mul(blocks_per_row);
    let mut candidates = vec![
        format!("{name}_scale"),
        format!("{name}_scales"),
        format!("{name}.scale"),
    ];
    // Primary GPT-OSS pattern: `<base>.blocks` packed weight is paired with
    // a `<base>.scales` E8M0 scale tensor (dot-separated suffix, per
    // `openai/gpt-oss` weights.py). Probe it first — it's the confirmed
    // correct spelling for GPT-OSS MoE weights.
    if let Some(base) = name.strip_suffix(".blocks") {
        candidates.insert(0, format!("{base}.scales"));
    }
    if let Some(base) = name.strip_suffix("_blocks") {
        candidates.push(format!("{base}_scales"));
        candidates.push(format!("{base}_scale"));
    }
    for cand in &candidates {
        for st in parsed {
            if let Ok(view) = st.tensor(cand) {
                let n_elem: usize = view.shape().iter().product();
                let raw = view.data();
                // E8M0 scales are one byte each. Accept a tensor whose
                // byte length matches the implied block grid regardless
                // of whether the loader reports it as U8.
                if n_elem == want && raw.len() == want {
                    return Some(raw.to_vec());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod mxfp4_scale_tests {
    use super::find_mxfp4_scales;
    use safetensors::tensor::{Dtype, TensorView};

    /// GPT-OSS names the packed weight `<base>.blocks` and its companion
    /// E8M0 scales `<base>.scales` (dot-separated, per `openai/gpt-oss`
    /// weights.py). `find_mxfp4_scales` must resolve the companion from the
    /// `.blocks` name even though none of the legacy underscore/dot-scale
    /// spellings match.
    #[test]
    fn find_mxfp4_scales_dot_suffix() {
        // rows = 2, cols = 64 ⇒ blocks_per_row = ceil(64 / 32) = 2,
        // so the E8M0 scale grid is 2 × 2 = 4 bytes.
        let rows = 2usize;
        let cols = 64usize;
        let name = "block.0.mlp.mlp1_weight.blocks";
        let scale_name = "block.0.mlp.mlp1_weight.scales";

        // Packed weight: two E2M1 nibbles per byte ⇒ rows * cols/2 bytes.
        let weight_bytes = vec![0u8; rows * cols / 2];
        // E8M0 scales: one byte per 32-wide block ⇒ 4 distinct values.
        let scale_bytes: Vec<u8> = vec![1, 2, 3, 4];

        let tensors = vec![
            (
                name.to_string(),
                TensorView::new(Dtype::U8, vec![rows, cols / 2], &weight_bytes).unwrap(),
            ),
            (
                scale_name.to_string(),
                TensorView::new(Dtype::U8, vec![rows, 2], &scale_bytes).unwrap(),
            ),
        ];
        let bytes = safetensors::serialize(tensors, &None).unwrap();
        let parsed = vec![safetensors::SafeTensors::deserialize(&bytes).unwrap()];

        let found = find_mxfp4_scales(&parsed, name, rows, cols)
            .expect("`.blocks` weight should resolve its `.scales` companion");
        assert_eq!(found, scale_bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;
    use crate::engine::{Engine, EngineOptions, ModelShape};
    use crate::io_provider::{generate_synthetic_experts, NvmeStorage, StorageConfig};
    use crate::multi_layer_cache::MultiLayerExpertCache;
    use crate::router::{PredictiveLoader, TopKRouter};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Self-cleaning unique temp directory for test fixtures.
    struct TempDir {
        path: PathBuf,
    }
    impl TempDir {
        fn new(label: &str) -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir()
                .join(format!("mer-model-{label}-{}-{n}-{ts}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn config_validation_catches_bad_shapes() {
        let mut c = RealModelConfig::tiny();
        c.head_dim = 7; // doesn't divide d_model
        assert!(c.validate().is_err());
        let mut c = RealModelConfig::tiny();
        c.top_k = 99;
        assert!(c.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_rope_dim() {
        // A `partial_rotary_factor` in (0, 1] that nonetheless rounds the
        // RoPE width down to zero must be rejected — otherwise rotary
        // embeddings are silently disabled for the whole model.
        let mut c = RealModelConfig::tiny(); // head_dim = 8
        c.advanced.partial_rotary_factor = Some(0.1); // floor(8*0.1) = 0
        assert_eq!(c.rope_dim(), 0);
        assert!(c.validate().is_err());
        // A factor that leaves a non-zero even width still validates.
        c.advanced.partial_rotary_factor = Some(0.5); // floor(8*0.5) = 4
        assert!(c.validate().is_ok());
    }

    #[test]
    fn seeded_model_has_correct_shapes() {
        let cfg = RealModelConfig::tiny();
        let m = RealModel::new_seeded(cfg.clone(), 1);
        assert_eq!(m.embedding.len(), cfg.vocab_size * cfg.d_model);
        assert_eq!(m.layers.len(), cfg.num_layers);
        assert_eq!(m.lm_head.weights.len(), cfg.vocab_size * cfg.d_model);
        assert_eq!(m.final_rms.weight.len(), cfg.d_model);
        assert!(m.embedding.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mimo_asymmetric_v_head_dim_shapes_and_forward() {
        use crate::transformer::KvCache;
        // MiMo-V2-Flash-style asymmetric attention: Q/K use head_dim=8,
        // V uses v_head_dim=4; partial RoPE over 4 of 8 dims; value scale.
        let mut adv = AdvancedConfig::default();
        adv.v_head_dim = Some(4);
        adv.partial_rotary_factor = Some(0.5); // floor(8*0.5)=4, even
        adv.attention_value_scale = Some(0.707);
        let cfg = RealModelConfig {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            num_layers: 2,
            advanced: adv,
            ..RealModelConfig::tiny()
        };
        assert_eq!(cfg.v_head_dim(), 4);
        assert_eq!(cfg.rope_dim(), 4);

        let m = RealModel::new_seeded(cfg.clone(), 7);
        let attn = &m.layers[0].attn;
        assert_eq!(attn.v_head_dim, 4);
        assert_eq!(attn.rope_dim, 4);
        assert_eq!(attn.attention_value_scale, Some(0.707));
        // wv rows = num_kv_heads * v_head_dim; wo cols = num_heads * v_head_dim.
        assert_eq!(attn.wv.len(), cfg.num_kv_heads * 4 * cfg.d_model);
        assert_eq!(attn.wo.len(), cfg.d_model * cfg.num_heads * 4);
        assert_eq!(attn.v_proj_dim(), cfg.num_kv_heads * 4);
        assert_eq!(attn.attn_out_dim(), cfg.num_heads * 4);

        // Fresh KV caches carry the asymmetric key/value widths.
        let caches = m.fresh_kv_caches();
        assert_eq!(caches[0].kv_dim, cfg.num_kv_heads * cfg.head_dim);
        assert_eq!(caches[0].v_dim, cfg.num_kv_heads * 4);

        // The attention forward runs end-to-end on the CPU path and yields
        // a finite, correctly-sized hidden state.
        let backend = crate::backend::current();
        let mut kv = KvCache::new_kv(attn.kv_dim(), attn.v_proj_dim());
        let x = vec![0.1f32; cfg.d_model];
        let out = attn.forward(&x, 0, 0, &mut kv, &*backend);
        assert_eq!(out.len(), cfg.d_model);
        assert!(out.iter().all(|v| v.is_finite()));
        assert_eq!(kv.seq_len, 1);
    }

    #[test]
    fn per_layer_swa_kv_heads_resolved() {
        use crate::architecture::Architecture;
        // MiMo-V2-Flash: SWA layers use `swa_num_key_value_heads`, global
        // layers use `num_kv_heads`; `num_heads` is the same for both. With
        // MiMoV2's 5:1 pattern over 12 layers, globals fall at layers 5 and 11.
        let mut adv = AdvancedConfig::default();
        adv.swa_num_key_value_heads = Some(2);
        let cfg = RealModelConfig {
            num_layers: 12,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
            architecture: Architecture::MiMoV2,
            window_size: Some(128),
            advanced: adv,
            ..RealModelConfig::tiny()
        };
        let m = RealModel::new_seeded(cfg.clone(), 5);
        for (l, layer) in m.layers.iter().enumerate() {
            let is_global = (l + 1) % 6 == 0;
            let expect_kv = if is_global { 4 } else { 2 };
            assert_eq!(layer.attn.num_kv_heads, expect_kv, "num_kv_heads layer {l}");
            assert_eq!(
                layer.attn.wk.len(),
                expect_kv * cfg.head_dim * cfg.d_model,
                "wk len layer {l}"
            );
            assert_eq!(
                layer.attn.wv.len(),
                expect_kv * cfg.head_dim * cfg.d_model,
                "wv len layer {l}"
            );
        }
        // KV caches follow the per-layer head count: layer 0 is SWA (2 heads),
        // layer 5 is global (4 heads).
        let caches = m.fresh_kv_caches();
        assert_eq!(caches[0].kv_dim, 2 * cfg.head_dim);
        assert_eq!(caches[5].kv_dim, 4 * cfg.head_dim);
    }

    #[test]
    fn swa_kv_heads_none_keeps_uniform_kv_heads() {
        use crate::architecture::Architecture;
        // No `swa_num_key_value_heads` ⇒ every layer (SWA or global) keeps
        // `num_kv_heads` — zero behaviour change for non-MiMo families.
        let cfg = RealModelConfig {
            num_layers: 12,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
            architecture: Architecture::MiMoV2,
            window_size: Some(128),
            ..RealModelConfig::tiny()
        };
        assert_eq!(cfg.advanced.swa_num_key_value_heads, None);
        let m = RealModel::new_seeded(cfg.clone(), 5);
        assert!(m.layers.iter().all(|l| l.attn.num_kv_heads == 4));
    }

    #[test]
    fn swa_kv_heads_not_dividing_num_heads_fails_validate() {
        let mut adv = AdvancedConfig::default();
        adv.swa_num_key_value_heads = Some(3); // 4 % 3 != 0
        let cfg = RealModelConfig {
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
            advanced: adv,
            ..RealModelConfig::tiny()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn mimo_v2_flash_advanced_threads_swa_kv_and_sink_bias() {
        use crate::architecture::HfConfig;
        let json = r#"{
            "architectures": ["MiMoV2FlashForCausalLM"],
            "model_type": "mimo_v2_flash",
            "hidden_size": 32, "num_hidden_layers": 6,
            "num_attention_heads": 4, "num_key_value_heads": 4,
            "head_dim": 8, "vocab_size": 256,
            "sliding_window": 128,
            "hybrid_layer_pattern": [0, 1, 1, 1, 1, 0],
            "swa_num_key_value_heads": 2,
            "add_swa_attention_sink_bias": true
        }"#;
        let hf = HfConfig::from_json_str(json).unwrap();
        let cfg = RealModelConfig::from_hf_config(&hf);
        assert_eq!(cfg.advanced.swa_num_key_value_heads, Some(2));
        assert!(cfg.advanced.add_swa_attention_sink_bias);
    }

    #[test]
    fn hybrid_attention_sets_per_layer_windows() {
        use crate::architecture::Architecture;
        // MiMo-V2: 12 layers, 5:1 SWA:global ⇒ global at layers 5 and 11,
        // SWA (window 128) everywhere else.
        let cfg = RealModelConfig {
            num_layers: 12,
            architecture: Architecture::MiMoV2,
            window_size: Some(128),
            ..RealModelConfig::tiny()
        };
        let m = RealModel::new_seeded(cfg, 3);
        for (l, layer) in m.layers.iter().enumerate() {
            if (l + 1) % 6 == 0 {
                assert_eq!(layer.attn.window_size, None, "layer {l} should be global");
            } else {
                assert_eq!(layer.attn.window_size, Some(128), "layer {l} should be SWA");
            }
        }

        // GPT-OSS: alternating 1:1 ⇒ even layers SWA, odd layers global.
        let cfg = RealModelConfig {
            num_layers: 6,
            architecture: Architecture::GptOss,
            window_size: Some(128),
            ..RealModelConfig::tiny()
        };
        let m = RealModel::new_seeded(cfg, 3);
        for (l, layer) in m.layers.iter().enumerate() {
            if l % 2 == 0 {
                assert_eq!(
                    layer.attn.window_size,
                    Some(128),
                    "even layer {l} should be SWA"
                );
            } else {
                assert_eq!(
                    layer.attn.window_size, None,
                    "odd layer {l} should be global"
                );
            }
        }

        // Legacy Mixtral: uniform window on every layer.
        let cfg = RealModelConfig {
            num_layers: 4,
            architecture: Architecture::Mixtral,
            window_size: Some(4096),
            ..RealModelConfig::tiny()
        };
        let m = RealModel::new_seeded(cfg, 3);
        assert!(m.layers.iter().all(|l| l.attn.window_size == Some(4096)));
    }

    #[test]
    fn global_expert_id_partitions_namespace() {
        let cfg = RealModelConfig {
            num_layers: 3,
            num_experts: 8,
            ..RealModelConfig::tiny()
        };
        let m = RealModel::new_seeded(cfg, 7);
        assert_eq!(m.global_expert_id(0, 0), 0);
        assert_eq!(m.global_expert_id(0, 7), 7);
        assert_eq!(m.global_expert_id(1, 0), 8);
        assert_eq!(m.global_expert_id(2, 5), 21);
    }

    fn build_engine_for_model(dir: &Path, cfg: &RealModelConfig) -> Arc<Engine> {
        // Total experts across all layers, addressed flat as
        // layer * num_experts + local.
        let total = cfg.num_layers as u32 * cfg.num_experts as u32;
        let weight_bytes = crate::inference::expert_weight_bytes(cfg.d_model, cfg.d_ff);
        let block = 4096usize;
        let expert_size = weight_bytes.div_ceil(block) * block;
        generate_synthetic_experts(dir, total, expert_size, cfg.d_model, cfg.d_ff)
            .expect("gen synthetic experts");
        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.to_path_buf(),
                expert_size,
                block_align: block,
                use_direct_io: false,
                num_experts_per_layer: None,
            })
            .expect("storage"),
        );
        storage.warmup_fds(0..total).expect("warmup");
        let pool = BufferPool::new(total as usize + 2, expert_size, block);
        let cache = Arc::new(MultiLayerExpertCache::single_layer((total as usize).max(2)));
        // The engine's TopKRouter is unused by `moe_step` (the gate
        // produces ids directly), but the engine constructor still
        // requires one.
        let router = crate::gating::Router::Markov(Arc::new(TopKRouter::new(total, cfg.top_k, 1)));
        let predictor = Arc::new(PredictiveLoader::new(total, 0, 0.05, 1));
        Arc::new(Engine::with_options(
            cache,
            pool,
            storage,
            router,
            predictor,
            ModelShape {
                d_model: cfg.d_model,
                d_ff: cfg.d_ff,
                hidden_seed: 1,
            },
            EngineOptions::default(),
        ))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_model_step_produces_valid_token_id() {
        let dir = TempDir::new("step");
        let cfg = RealModelConfig::tiny();
        let engine = build_engine_for_model(&dir.path, &cfg);
        let model = RealModel::new_seeded(cfg.clone(), 0xDEAD);
        let mut kv = model.fresh_kv_caches();
        let next = model
            .step(
                &engine,
                42,
                0,
                &mut kv,
                &crate::sampling::SamplingParams::greedy(),
            )
            .await
            .expect("model step");
        assert!((next as usize) < cfg.vocab_size);
        // KV caches grew by exactly one position.
        for c in &kv {
            assert_eq!(c.seq_len, 1);
        }
        // The engine's hit/miss counters were touched (cold start =>
        // misses).
        let r = engine.report();
        assert!(r.misses > 0, "first step should miss the cache");
        assert!(
            r.bytes_read > 0,
            "engine should have read expert bytes from disk"
        );
    }

    /// Regression test for MLA KV-cache sizing. `fresh_kv_caches` must
    /// allocate the *latent* width (`kv_lora_rank + qk_rope_head_dim`,
    /// 20 here) for MLA layers, not the unused standard `num_kv_heads *
    /// head_dim` (32 here). Before the fix the per-layer cache was sized
    /// with `attn.kv_dim()`, so the very first `step` panicked inside
    /// `KvCache::append` (`copy_from_slice` / debug-assert width
    /// mismatch) the moment `mla.forward` appended its latent vector.
    /// Stepping twice also exercises multi-position latent append.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mla_model_steps_through_fresh_kv_caches() {
        let dir = TempDir::new("mla-step");
        let mut advanced = AdvancedConfig::default();
        advanced.mla = Some(MlaDims {
            q_lora_rank: 0,
            kv_lora_rank: 16,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 8,
        });
        let cfg = RealModelConfig {
            architecture: Architecture::DeepSeekV3,
            advanced,
            num_layers: 2,
            ..RealModelConfig::tiny()
        };
        let engine = build_engine_for_model(&dir.path, &cfg);
        let model = RealModel::new_seeded(cfg.clone(), 0x5EED);
        assert!(
            model.layers.iter().all(|l| l.mla.is_some()),
            "all layers MLA"
        );

        let mut kv = model.fresh_kv_caches();
        let latent = 16 + 4; // kv_lora_rank + qk_rope_head_dim
        for c in &kv {
            assert_eq!(c.kv_dim, latent, "MLA K cache must be latent-width");
            assert_eq!(c.v_dim, latent, "MLA V cache must be latent-width");
        }

        let t1 = model
            .step(
                &engine,
                7,
                0,
                &mut kv,
                &crate::sampling::SamplingParams::greedy(),
            )
            .await
            .expect("model step");
        let t2 = model
            .step(
                &engine,
                t1,
                1,
                &mut kv,
                &crate::sampling::SamplingParams::greedy(),
            )
            .await
            .expect("model step");
        assert!((t1 as usize) < cfg.vocab_size);
        assert!((t2 as usize) < cfg.vocab_size);
        for c in &kv {
            assert_eq!(c.seq_len, 2, "each MLA layer cached two positions");
        }
    }

    /// Part A5 (fail-closed inference): real-model forward must reject
    /// token ids outside `[0, vocab_size)` instead of wrapping them
    /// with modulo onto an unrelated embedding row.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_model_rejects_out_of_range_token_ids() {
        let dir = TempDir::new("badtok");
        let cfg = RealModelConfig::tiny();
        let engine = build_engine_for_model(&dir.path, &cfg);
        let model = RealModel::new_seeded(cfg.clone(), 1);
        let vocab = cfg.vocab_size as u32;

        // Boundary-valid ids succeed: token 0 and the last valid token.
        for tid in [0u32, vocab - 1] {
            let mut kv = model.fresh_kv_caches();
            model
                .forward_token_hidden(&engine, tid, 0, &mut kv)
                .await
                .unwrap_or_else(|e| panic!("token {tid} must be valid: {e}"));
        }

        // `vocab_size` (one past the end) and `u32::MAX` fail closed.
        for tid in [vocab, u32::MAX] {
            let mut kv = model.fresh_kv_caches();
            let err = model
                .forward_token_hidden(&engine, tid, 0, &mut kv)
                .await
                .expect_err("out-of-range token id must fail");
            match err {
                RealInferenceError::InvalidTokenId {
                    token_id,
                    vocab_size,
                } => {
                    assert_eq!(token_id, tid);
                    assert_eq!(vocab_size, cfg.vocab_size);
                }
                other => panic!("expected InvalidTokenId, got: {other}"),
            }
        }

        // try_embed mirrors the same contract directly.
        assert!(model.try_embed(vocab - 1).is_ok());
        assert!(model.try_embed(vocab).is_err());
    }

    /// A1 completion: malformed required shared-expert projections are a
    /// typed [`SharedExpertError::ShapeMismatch`], never a silent `None`.
    #[test]
    fn shared_expert_malformed_projections_is_typed_error() {
        let d_model = 4;
        let d_ff = 3;
        let gate = vec![0.05f32; d_ff * d_model];
        let up = vec![0.07f32; d_ff * d_model];
        let down_short = vec![0.09f32; d_model * d_ff - 1];
        let err = SharedExpert::from_projections(d_model, d_ff, &gate, &up, &down_short, None)
            .expect_err("inconsistent shapes must be a typed error");
        assert!(matches!(
            err,
            crate::transformer::SharedExpertError::ShapeMismatch { .. }
        ));
        // A wrong-length sigmoid gate is equally malformed.
        let down = vec![0.09f32; d_model * d_ff];
        let err = SharedExpert::from_projections(
            d_model,
            d_ff,
            &gate,
            &up,
            &down,
            Some(vec![0.0f32; d_model + 1]),
        )
        .expect_err("wrong-length gate_inp must be a typed error");
        assert!(matches!(
            err,
            crate::transformer::SharedExpertError::ShapeMismatch { .. }
        ));
    }

    /// A1 completion: a shared expert whose cached Candle execution fails
    /// returns [`SharedExpertError::Compute`]; one whose per-call weight
    /// reconstruction fails returns [`SharedExpertError::Reconstruction`].
    /// Neither ever fabricates a zero vector.
    #[test]
    fn shared_expert_forward_failures_are_typed_errors() {
        let d_model = 4;
        let d_ff = 3;
        let gate = vec![0.05f32; d_ff * d_model];
        let up = vec![0.07f32; d_ff * d_model];
        let down = vec![0.09f32; d_model * d_ff];
        // Cached Candle tensor execution failure (shape-incompatible
        // cached tensors make the matmul fail at forward time).
        let mut se =
            SharedExpert::from_projections(d_model, d_ff, &gate, &up, &down, None).unwrap();
        se.poison_cached_tensors_for_test();
        let err = se
            .forward(&vec![0.1f32; d_model])
            .expect_err("candle execution failure must propagate");
        assert!(matches!(
            err,
            crate::transformer::SharedExpertError::Compute { .. }
        ));
        // Reconstruction fallback failure (tensor cache dropped and the
        // stored weight blob corrupted).
        let mut se =
            SharedExpert::from_projections(d_model, d_ff, &gate, &up, &down, None).unwrap();
        se.corrupt_weights_for_test();
        let err = se
            .forward(&vec![0.1f32; d_model])
            .expect_err("weight reconstruction failure must propagate");
        assert!(matches!(
            err,
            crate::transformer::SharedExpertError::Reconstruction { .. }
        ));
        // The reconstruction fallback itself stays numerically correct when
        // only the tensor cache is dropped.
        let mut se =
            SharedExpert::from_projections(d_model, d_ff, &gate, &up, &down, None).unwrap();
        se.drop_cached_tensors_for_test();
        se.forward(&vec![0.1f32; d_model])
            .expect("reconstruction fallback path must still succeed");
    }

    /// A1 completion (strict): a failing always-on shared expert fails the
    /// whole request with layer/role context instead of dropping its
    /// contribution from the residual stream.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_model_fails_closed_on_shared_expert_failure() {
        let dir = TempDir::new("shexp_fail_strict");
        let cfg = RealModelConfig::tiny();
        let engine = build_engine_for_model(&dir.path, &cfg);
        let mut model = RealModel::new_seeded(cfg.clone(), 1);
        let mut se = SharedExpert::from_projections(
            cfg.d_model,
            3,
            &vec![0.05f32; 3 * cfg.d_model],
            &vec![0.07f32; 3 * cfg.d_model],
            &vec![0.09f32; cfg.d_model * 3],
            None,
        )
        .unwrap();
        se.corrupt_weights_for_test();
        model.layers[0].shared_expert = Some(se);
        let mut kv = model.fresh_kv_caches();
        let err = model
            .forward_token_hidden(&engine, 1, 0, &mut kv)
            .await
            .expect_err("failing shared expert must fail the request in strict mode");
        match err {
            RealInferenceError::ResidentFfn {
                layer,
                role,
                source,
            } => {
                assert_eq!(layer, 0);
                assert_eq!(role, ResidentFfnRole::Shared);
                assert!(matches!(
                    source,
                    crate::transformer::SharedExpertError::Reconstruction { .. }
                ));
            }
            other => panic!("expected ResidentFfn, got: {other}"),
        }
    }

    /// A1 completion (strict): a failing dense-prefix FFN fails the whole
    /// request with layer/role context.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_model_fails_closed_on_dense_ffn_failure() {
        let dir = TempDir::new("dense_fail_strict");
        let cfg = RealModelConfig::tiny();
        let engine = build_engine_for_model(&dir.path, &cfg);
        let mut model = RealModel::new_seeded(cfg.clone(), 1);
        let mut ffn = SharedExpert::from_projections(
            cfg.d_model,
            3,
            &vec![0.05f32; 3 * cfg.d_model],
            &vec![0.07f32; 3 * cfg.d_model],
            &vec![0.09f32; cfg.d_model * 3],
            None,
        )
        .unwrap();
        ffn.corrupt_weights_for_test();
        model.layers[0].dense_ffn = Some(ffn);
        let mut kv = model.fresh_kv_caches();
        let err = model
            .forward_token_hidden(&engine, 1, 0, &mut kv)
            .await
            .expect_err("failing dense FFN must fail the request in strict mode");
        match err {
            RealInferenceError::ResidentFfn { layer, role, .. } => {
                assert_eq!(layer, 0);
                assert_eq!(role, ResidentFfnRole::Dense);
            }
            other => panic!("expected ResidentFfn, got: {other}"),
        }
    }

    /// A1 completion (explicit degraded mode): with
    /// `allow_degraded_experts = true` a failing shared expert or dense
    /// FFN substitutes a zero contribution, the request succeeds, and
    /// every substitution is counted so the output is marked
    /// non-authoritative.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn degraded_mode_substitutes_zero_for_failing_resident_ffn() {
        let dir = TempDir::new("shexp_fail_degraded");
        let cfg = RealModelConfig {
            num_layers: 2,
            ..RealModelConfig::tiny()
        };
        let engine = with_policy(
            build_engine_for_model(&dir.path, &cfg),
            crate::inference::RealInferencePolicy {
                allow_degraded_experts: true,
                ..crate::inference::RealInferencePolicy::STRICT
            },
        );
        let mut model = RealModel::new_seeded(cfg.clone(), 1);
        let broken = || {
            let mut se = SharedExpert::from_projections(
                cfg.d_model,
                3,
                &vec![0.05f32; 3 * cfg.d_model],
                &vec![0.07f32; 3 * cfg.d_model],
                &vec![0.09f32; cfg.d_model * 3],
                None,
            )
            .unwrap();
            se.corrupt_weights_for_test();
            se
        };
        model.layers[0].shared_expert = Some(broken());
        model.layers[1].dense_ffn = Some(broken());
        let mut kv = model.fresh_kv_caches();
        model
            .forward_token_hidden(&engine, 1, 0, &mut kv)
            .await
            .expect("degraded mode substitutes zero and continues");
        let report = engine.report();
        assert!(
            report.degraded_expert_substitutions >= 2,
            "each substituted resident FFN must be counted, got {}",
            report.degraded_expert_substitutions
        );
    }

    /// Part A3 (fail-closed inference): unexpected NaN in an active
    /// attention row must fail the request in strict mode instead of
    /// fabricating a uniform attention distribution. Poisoning the
    /// layer-1 Q projection makes every layer-1 attention row NaN.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_model_fails_closed_on_non_finite_attention() {
        let dir = TempDir::new("nanattn");
        let cfg = RealModelConfig::tiny();
        let engine = build_engine_for_model(&dir.path, &cfg);
        let mut model = RealModel::new_seeded(cfg.clone(), 1);
        {
            let attn = &mut model.layers[0].attn;
            let rows = cfg.num_heads * cfg.d_model / cfg.num_heads;
            attn.wq = crate::dense_tensor::DenseWeight::from_f32(
                vec![f32::NAN; rows * cfg.d_model],
                rows,
                cfg.d_model,
            );
        }
        let mut kv = model.fresh_kv_caches();
        let err = model
            .forward_token_hidden(&engine, 1, 0, &mut kv)
            .await
            .expect_err("NaN attention must fail the request in strict mode");
        match err {
            RealInferenceError::NonFiniteAttention {
                layer,
                head,
                pos,
                architecture,
                kind,
            } => {
                assert_eq!(layer, 0);
                assert_eq!(head, 0);
                assert_eq!(pos, 0);
                assert_eq!(architecture, cfg.architecture);
                assert_eq!(kind, crate::transformer::NumericalErrorKind::NaNScores);
            }
            other => panic!("expected NonFiniteAttention, got: {other}"),
        }
    }

    /// Rebuild an engine with a custom fail-open policy (the test owns
    /// the sole Arc right after construction).
    fn with_policy(
        engine: Arc<Engine>,
        policy: crate::inference::RealInferencePolicy,
    ) -> Arc<Engine> {
        let mut owned = match Arc::try_unwrap(engine) {
            Ok(e) => e,
            Err(_) => panic!("test owns the sole engine Arc"),
        };
        owned.core.options.policy = policy;
        Arc::new(owned)
    }

    /// Policy separation: `allow_degraded_experts` must NOT enable the
    /// uniform-attention fallback — NaN attention still fails the
    /// request even in expert-degraded mode.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn degraded_experts_policy_does_not_enable_attention_fallback() {
        let dir = TempDir::new("nanattn_degraded_experts");
        let cfg = RealModelConfig::tiny();
        let engine = with_policy(
            build_engine_for_model(&dir.path, &cfg),
            crate::inference::RealInferencePolicy {
                allow_degraded_experts: true,
                ..crate::inference::RealInferencePolicy::STRICT
            },
        );
        let mut model = RealModel::new_seeded(cfg.clone(), 1);
        {
            let attn = &mut model.layers[0].attn;
            let rows = cfg.num_heads * cfg.d_model / cfg.num_heads;
            attn.wq = crate::dense_tensor::DenseWeight::from_f32(
                vec![f32::NAN; rows * cfg.d_model],
                rows,
                cfg.d_model,
            );
        }
        let mut kv = model.fresh_kv_caches();
        let err = model
            .forward_token_hidden(&engine, 1, 0, &mut kv)
            .await
            .expect_err("expert degradation must not fabricate attention");
        assert!(matches!(err, RealInferenceError::NonFiniteAttention { .. }));
    }

    /// Policy separation (item 4): `allow_truncated_expert_payloads`
    /// must NOT enable the uniform-attention fallback either — NaN
    /// attention still fails the request under the payload-only policy.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn truncated_payload_policy_does_not_enable_attention_fallback() {
        let dir = TempDir::new("nanattn_truncated_payloads");
        let cfg = RealModelConfig::tiny();
        let engine = with_policy(
            build_engine_for_model(&dir.path, &cfg),
            crate::inference::RealInferencePolicy {
                allow_truncated_expert_payloads: true,
                ..crate::inference::RealInferencePolicy::STRICT
            },
        );
        let mut model = RealModel::new_seeded(cfg.clone(), 1);
        {
            let attn = &mut model.layers[0].attn;
            let rows = cfg.num_heads * cfg.d_model / cfg.num_heads;
            attn.wq = crate::dense_tensor::DenseWeight::from_f32(
                vec![f32::NAN; rows * cfg.d_model],
                rows,
                cfg.d_model,
            );
        }
        let mut kv = model.fresh_kv_caches();
        let err = model
            .forward_token_hidden(&engine, 1, 0, &mut kv)
            .await
            .expect_err("payload tolerance must not fabricate attention");
        assert!(matches!(err, RealInferenceError::NonFiniteAttention { .. }));
    }

    /// Policy separation: only the explicit
    /// `allow_nonfinite_attention_fallback` policy re-enables the
    /// legacy uniform-attention fallback for non-finite rows.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn nonfinite_attention_fallback_policy_enables_uniform_fallback() {
        let dir = TempDir::new("nanattn_fallback_policy");
        let cfg = RealModelConfig::tiny();
        let engine = with_policy(
            build_engine_for_model(&dir.path, &cfg),
            crate::inference::RealInferencePolicy {
                allow_nonfinite_attention_fallback: true,
                ..crate::inference::RealInferencePolicy::STRICT
            },
        );
        let mut model = RealModel::new_seeded(cfg.clone(), 1);
        {
            let attn = &mut model.layers[0].attn;
            let rows = cfg.num_heads * cfg.d_model / cfg.num_heads;
            attn.wq = crate::dense_tensor::DenseWeight::from_f32(
                vec![f32::NAN; rows * cfg.d_model],
                rows,
                cfg.d_model,
            );
        }
        let mut kv = model.fresh_kv_caches();
        model
            .forward_token_hidden(&engine, 1, 0, &mut kv)
            .await
            .expect("explicit fallback policy keeps the legacy uniform behaviour");
    }

    /// Part A3: a finite forward pass does not trip the non-finite
    /// check (guards against false positives from the thread-local
    /// counter snapshotting).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_model_finite_forward_passes_strict_numeric_check() {
        let dir = TempDir::new("finiteattn");
        let cfg = RealModelConfig::tiny();
        let engine = build_engine_for_model(&dir.path, &cfg);
        let model = RealModel::new_seeded(cfg.clone(), 1);
        let mut kv = model.fresh_kv_caches();
        for pos in 0..3 {
            model
                .forward_token_hidden(&engine, (pos as u32) + 1, pos, &mut kv)
                .await
                .expect("finite forward must pass");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_model_step_is_deterministic_across_two_runs() {
        let dir = TempDir::new("det");
        let cfg = RealModelConfig::tiny();
        let engine = build_engine_for_model(&dir.path, &cfg);
        let model = RealModel::new_seeded(cfg.clone(), 1);

        let mut kv1 = model.fresh_kv_caches();
        let t1 = model
            .step(
                &engine,
                7,
                0,
                &mut kv1,
                &crate::sampling::SamplingParams::greedy(),
            )
            .await
            .expect("model step");
        let t2 = model
            .step(
                &engine,
                t1,
                1,
                &mut kv1,
                &crate::sampling::SamplingParams::greedy(),
            )
            .await
            .expect("model step");

        let mut kv2 = model.fresh_kv_caches();
        let u1 = model
            .step(
                &engine,
                7,
                0,
                &mut kv2,
                &crate::sampling::SamplingParams::greedy(),
            )
            .await
            .expect("model step");
        let u2 = model
            .step(
                &engine,
                u1,
                1,
                &mut kv2,
                &crate::sampling::SamplingParams::greedy(),
            )
            .await
            .expect("model step");

        assert_eq!((t1, t2), (u1, u2));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_model_multi_layer_partitions_expert_namespace() {
        let dir = TempDir::new("multi");
        let cfg = RealModelConfig {
            num_layers: 2,
            num_experts: 4,
            top_k: 2,
            ..RealModelConfig::tiny()
        };
        let engine = build_engine_for_model(&dir.path, &cfg);
        let model = RealModel::new_seeded(cfg.clone(), 9);
        let mut kv = model.fresh_kv_caches();
        let _ = model
            .step(
                &engine,
                5,
                0,
                &mut kv,
                &crate::sampling::SamplingParams::greedy(),
            )
            .await
            .expect("model step");
        // Both layers contributed to KV cache growth.
        assert_eq!(kv.len(), 2);
        for c in &kv {
            assert_eq!(c.seq_len, 1);
        }
    }

    /// `from_safetensors` must load the dense projections, embedding,
    /// final RMSNorm and LM head from a HF-style `.safetensors` shard
    /// and leave anything missing as the seeded baseline. We build a
    /// minimal shard by hand and compare specific weights.
    #[test]
    fn from_safetensors_loads_dense_tensors() {
        use safetensors::serialize_to_file;
        use safetensors::tensor::{Dtype, TensorView};
        let cfg = RealModelConfig {
            vocab_size: 8,
            d_model: 4,
            d_ff: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            num_layers: 1,
            num_experts: 2,
            top_k: 1,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: Architecture::Mixtral,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        };
        let q_dim = cfg.num_heads * cfg.head_dim;
        let dir = TempDir::new("safetensors_load");

        // Distinct sentinel values per tensor so we can identify which
        // got loaded vs. which fell back to seeded.
        let embed = vec![0.25f32; cfg.vocab_size * cfg.d_model];
        let q = vec![0.5f32; q_dim * cfg.d_model];
        let final_rms = vec![1.5f32; cfg.d_model];

        // Pack each tensor as little-endian f32 bytes and feed to
        // `safetensors::serialize_to_file` — same on-disk format the
        // HF Python pipeline emits.
        let to_bytes = |v: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(v.len() * 4);
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        };
        let embed_bytes = to_bytes(&embed);
        let q_bytes = to_bytes(&q);
        let rms_bytes = to_bytes(&final_rms);
        let tensors: Vec<(String, TensorView)> = vec![
            (
                "model.embed_tokens.weight".to_string(),
                TensorView::new(Dtype::F32, vec![cfg.vocab_size, cfg.d_model], &embed_bytes)
                    .unwrap(),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                TensorView::new(Dtype::F32, vec![q_dim, cfg.d_model], &q_bytes).unwrap(),
            ),
            (
                "model.norm.weight".to_string(),
                TensorView::new(Dtype::F32, vec![cfg.d_model], &rms_bytes).unwrap(),
            ),
        ];
        let out_path = dir.path.join("model.safetensors");
        serialize_to_file(tensors, &None, &out_path).unwrap();

        let model = RealModel::from_safetensors(cfg.clone(), &dir.path, 1).unwrap();
        // Loaded tensors carry their sentinel values verbatim.
        assert!(model.embedding.iter().all(|x| x == 0.25));
        assert!(model.layers[0].attn.wq.iter().all(|x| x == 0.5));
        // Anything else is still the seeded init (gate, k/v/o, MoE
        // gate, lm_head, rms_attn / rms_moe, etc.). Sanity: lm_head
        // wasn't provided, so its weights stayed at whatever the seed
        // produced — they must not be all-equal (which would only
        // happen if our find-tensor logic spuriously matched).
        let lm = &model.lm_head.weights;
        let first = lm.iter().next().unwrap();
        assert!(
            lm.iter().any(|x| x != first),
            "lm_head should remain seeded, not constant"
        );
    }

    /// Regression: a checkpoint with `attention_bias = true` AND an
    /// asymmetric V head (`v_head_dim != head_dim`, MiMo-V2-Flash style)
    /// ships a `v_proj.bias` of length `num_kv_heads * v_head_dim`
    /// (`v_proj_dim`), NOT `num_kv_heads * head_dim` (`kv_dim`). The loader
    /// must size the V-bias expectation by `v_proj_dim`; otherwise the
    /// size-checked `find_f32` silently drops the tensor and the bias is
    /// lost.
    #[test]
    fn from_safetensors_loads_asymmetric_v_proj_bias() {
        use safetensors::serialize_to_file;
        use safetensors::tensor::{Dtype, TensorView};
        let mut adv = AdvancedConfig::default();
        adv.attention_bias = true;
        adv.v_head_dim = Some(2); // V uses 2 while Q/K use head_dim = 4
        let cfg = RealModelConfig {
            vocab_size: 8,
            d_model: 8, // = num_heads * head_dim (Mixtral tie)
            d_ff: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            num_layers: 1,
            num_experts: 2,
            top_k: 1,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: Architecture::Mixtral,
            first_k_dense_replace: 0,
            advanced: adv,
        };
        let kv_dim = cfg.num_kv_heads * cfg.head_dim; // 8
        let v_proj_dim = cfg.num_kv_heads * cfg.v_head_dim(); // 4
        assert_ne!(kv_dim, v_proj_dim, "test requires an asymmetric V head");

        let bv = vec![0.375f32; v_proj_dim];
        let to_bytes = |v: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(v.len() * 4);
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        };
        let bv_bytes = to_bytes(&bv);
        let dir = TempDir::new("safetensors_vbias");
        let tensors: Vec<(String, TensorView)> = vec![(
            "model.layers.0.self_attn.v_proj.bias".to_string(),
            TensorView::new(Dtype::F32, vec![v_proj_dim], &bv_bytes).unwrap(),
        )];
        let out_path = dir.path.join("model.safetensors");
        serialize_to_file(tensors, &None, &out_path).unwrap();

        let model = RealModel::from_safetensors(cfg.clone(), &dir.path, 1).unwrap();
        let bv_loaded = model.layers[0]
            .attn
            .bv
            .as_ref()
            .expect("v_proj.bias must load for an asymmetric V head (was sized by kv_dim)");
        assert_eq!(bv_loaded.len(), v_proj_dim);
        assert!(bv_loaded.iter().all(|&x| x == 0.375));
    }

    /// Phi-4 (`phi3`) ships a single fused `qkv_proj` tensor. The loader
    /// must split it into the engine's separate `wq` / `wk` / `wv` slabs at
    /// the `[q_dim | kv_dim | kv_dim]` row boundaries, in that order.
    #[test]
    fn from_safetensors_splits_phi4_fused_qkv() {
        use safetensors::serialize_to_file;
        use safetensors::tensor::{Dtype, TensorView};
        let cfg = RealModelConfig {
            vocab_size: 8,
            d_model: 4,
            d_ff: 8,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            num_layers: 1,
            num_experts: 1,
            top_k: 1,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: Architecture::Phi4,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        };
        let q_dim = cfg.num_heads * cfg.head_dim; // 4
        let kv_dim = cfg.num_kv_heads * cfg.head_dim; // 2
        let dir = TempDir::new("phi4_qkv");

        // Fused qkv: [(q_dim + 2*kv_dim) rows, d_model cols], row-major.
        // Use a distinct sentinel per region so the split is unambiguous.
        let mut fused = Vec::new();
        fused.extend(std::iter::repeat(0.1f32).take(q_dim * cfg.d_model));
        fused.extend(std::iter::repeat(0.2f32).take(kv_dim * cfg.d_model));
        fused.extend(std::iter::repeat(0.3f32).take(kv_dim * cfg.d_model));
        let to_bytes = |v: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(v.len() * 4);
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        };
        let fused_bytes = to_bytes(&fused);
        let tensors: Vec<(String, TensorView)> = vec![(
            "model.layers.0.self_attn.qkv_proj.weight".to_string(),
            TensorView::new(
                Dtype::F32,
                vec![q_dim + 2 * kv_dim, cfg.d_model],
                &fused_bytes,
            )
            .unwrap(),
        )];
        let out_path = dir.path.join("model.safetensors");
        serialize_to_file(tensors, &None, &out_path).unwrap();

        let model = RealModel::from_safetensors(cfg.clone(), &dir.path, 1).unwrap();
        let attn = &model.layers[0].attn;
        assert_eq!(attn.wq.len(), q_dim * cfg.d_model);
        assert_eq!(attn.wk.len(), kv_dim * cfg.d_model);
        assert_eq!(attn.wv.len(), kv_dim * cfg.d_model);
        assert!(attn.wq.iter().all(|x| x == 0.1), "wq region");
        assert!(attn.wk.iter().all(|x| x == 0.2), "wk region");
        assert!(attn.wv.iter().all(|x| x == 0.3), "wv region");
    }

    /// Mistral Small 3 (`mistral3`) is multimodal; its language-model
    /// tensors carry a `language_model.` prefix. The loader must prepend
    /// that prefix before looking tensors up.
    #[test]
    fn from_safetensors_handles_mistral_language_model_prefix() {
        use safetensors::serialize_to_file;
        use safetensors::tensor::{Dtype, TensorView};
        let cfg = RealModelConfig {
            vocab_size: 8,
            d_model: 4,
            d_ff: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            num_layers: 1,
            num_experts: 1,
            top_k: 1,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: Architecture::MistralSmall3,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        };
        let dir = TempDir::new("mistral3_prefix");
        let embed = vec![0.7f32; cfg.vocab_size * cfg.d_model];
        let to_bytes = |v: &[f32]| -> Vec<u8> {
            let mut out = Vec::with_capacity(v.len() * 4);
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        };
        let embed_bytes = to_bytes(&embed);
        // Prefixed name (what a real Mistral3 checkpoint emits) plus a
        // vision-tower tensor the loader must ignore.
        let vision = vec![9.0f32; 4];
        let vision_bytes = to_bytes(&vision);
        let tensors: Vec<(String, TensorView)> = vec![
            (
                "language_model.model.embed_tokens.weight".to_string(),
                TensorView::new(Dtype::F32, vec![cfg.vocab_size, cfg.d_model], &embed_bytes)
                    .unwrap(),
            ),
            (
                "vision_tower.patch_conv.weight".to_string(),
                TensorView::new(Dtype::F32, vec![4], &vision_bytes).unwrap(),
            ),
        ];
        let out_path = dir.path.join("model.safetensors");
        serialize_to_file(tensors, &None, &out_path).unwrap();

        let model = RealModel::from_safetensors(cfg.clone(), &dir.path, 1).unwrap();
        assert!(
            model.embedding.iter().all(|x| x == 0.7),
            "prefixed embed_tokens must load via language_model. prefix"
        );
    }

    /// DeepSeek-V3 (`deepseek_v3`) is now fully executable: `new_seeded`
    /// builds runnable MLA blocks for every layer instead of the loader
    /// failing loud. This guards against regressing the MLA + FP8
    /// integration back to the old "unsupported" behavior.
    #[test]
    fn deepseek_seeds_runnable_mla() {
        let mut advanced = AdvancedConfig::default();
        advanced.mla = Some(MlaDims {
            q_lora_rank: 0,
            kv_lora_rank: 16,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 8,
        });
        let cfg = RealModelConfig {
            architecture: Architecture::DeepSeekV3,
            first_k_dense_replace: 3,
            advanced,
            num_layers: 4,
            ..RealModelConfig::tiny()
        };
        let model = RealModel::new_seeded(cfg, 1);
        assert!(model.layers.iter().all(|l| l.mla.is_some()));
        // The latent KV-cache width must match the MLA latent dim so the
        // per-layer cache is sized correctly.
        let kv_dim = model.layers[0].kv_dim();
        assert_eq!(kv_dim, 16 + 4);
    }

    /// `rope_scaling` of type `yarn` must be wired into both attention
    /// paths at model build time: the standard path's `rope_yarn` and
    /// the MLA path's `rope_yarn` + mscale-corrected `softmax_scale`.
    #[test]
    fn seeded_model_wires_yarn_rope_scaling() {
        use crate::architecture::RopeScaling;
        let scaling = RopeScaling {
            rope_type: "yarn".to_string(),
            factor: 40.0,
            original_max_position_embeddings: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
            mscale_all_dim: 1.0,
        };
        let mut advanced = AdvancedConfig::default();
        advanced.mla = Some(MlaDims {
            q_lora_rank: 0,
            kv_lora_rank: 16,
            qk_nope_head_dim: 4,
            qk_rope_head_dim: 4,
            v_head_dim: 8,
        });
        advanced.rope_scaling = Some(scaling.clone());
        let cfg = RealModelConfig {
            architecture: Architecture::DeepSeekV3,
            advanced,
            num_layers: 2,
            ..RealModelConfig::tiny()
        };
        let model = RealModel::new_seeded(cfg, 1);
        let first_cache = model.layers[0]
            .attn
            .rope_cache
            .as_ref()
            .expect("standard path must carry a shared RoPE cache")
            .clone();
        for layer in &model.layers {
            assert!(
                layer.attn.rope_yarn.is_some(),
                "standard path must carry YaRN"
            );
            assert!(
                Arc::ptr_eq(
                    &first_cache,
                    layer
                        .attn
                        .rope_cache
                        .as_ref()
                        .expect("standard path must carry a shared RoPE cache")
                ),
                "identical layer RoPE configs should share one cache"
            );
            let mla = layer.mla.as_ref().expect("MLA seeded");
            assert!(mla.rope_yarn.is_some(), "MLA path must carry YaRN");
            let expected =
                crate::mla::MultiHeadLatentAttention::yarn_softmax_scale(4, 4, Some(&scaling));
            assert!((mla.softmax_scale - expected).abs() < 1e-7);
            assert!(
                mla.softmax_scale
                    > crate::mla::MultiHeadLatentAttention::default_softmax_scale(4, 4),
                "mscale^2 correction must raise the softmax scale"
            );
        }
        // Without a rope_scaling block, nothing is wired.
        let plain = RealModel::new_seeded(
            RealModelConfig {
                num_layers: 2,
                ..RealModelConfig::tiny()
            },
            1,
        );
        assert!(plain.layers.iter().all(|l| l.attn.rope_yarn.is_none()));
        let plain_cache = plain.layers[0].attn.rope_cache.as_ref().unwrap();
        assert!(Arc::ptr_eq(
            plain_cache,
            plain.layers[1].attn.rope_cache.as_ref().unwrap()
        ));
    }

    /// `from_hf_config` maps a parsed Qwen3-MoE `config.json` (explicit
    /// `head_dim` that differs from `hidden_size / num_heads`) into a
    /// loadable `RealModelConfig` that passes `validate()`.
    #[test]
    fn from_hf_config_maps_qwen3_moe() {
        let json = r#"{
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "moe_intermediate_size": 768,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "num_experts": 128,
            "num_experts_per_tok": 8
        }"#;
        let hf = crate::architecture::HfConfig::from_json_str(json).unwrap();
        let cfg = RealModelConfig::from_hf_config(&hf);
        assert_eq!(cfg.architecture, Architecture::Qwen3Moe);
        assert_eq!(cfg.d_model, 2048);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.num_heads, 32);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.num_experts, 128);
        assert_eq!(cfg.top_k, 8);
        // d_ff resolves to the MoE expert width, not the dense one.
        assert_eq!(cfg.d_ff, 768);
        // head_dim * num_heads (4096) != d_model (2048) — must still pass
        // because the Mixtral-only tie is relaxed for other architectures.
        cfg.validate().expect("Qwen3-MoE config must validate");
    }

    /// `from_dir_auto`-adjacent dispatch test.
    #[test]
    fn from_dir_auto_dispatches_on_safetensors_presence() {
        use safetensors::serialize_to_file;
        use safetensors::tensor::{Dtype, TensorView};
        let cfg = RealModelConfig {
            vocab_size: 4,
            d_model: 4,
            d_ff: 4,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            num_layers: 1,
            num_experts: 2,
            top_k: 1,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: Architecture::Mixtral,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        };

        // 1) Empty dir without safetensors falls back to from_dir
        // (which itself silently keeps the seeded init since no .bin
        // files exist) — so no panic / error.
        let empty = TempDir::new("auto_empty");
        let _ = RealModel::from_dir_auto(cfg.clone(), &empty.path, 7).unwrap();

        // 2) Dir with a .safetensors shard goes through the safetensors path.
        let st_dir = TempDir::new("auto_st");
        let embed = vec![0.75f32; cfg.vocab_size * cfg.d_model];
        let mut bytes = Vec::with_capacity(embed.len() * 4);
        for &x in &embed {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let view = TensorView::new(Dtype::F32, vec![cfg.vocab_size, cfg.d_model], &bytes).unwrap();
        serialize_to_file(
            [("model.embed_tokens.weight".to_string(), view)],
            &None,
            &st_dir.path.join("model.safetensors"),
        )
        .unwrap();
        let model = RealModel::from_dir_auto(cfg, &st_dir.path, 7).unwrap();
        assert!(model.embedding.iter().all(|x| x == 0.75));
    }

    /// Helper: write a slice of f32 as a little-endian `.bin` file.
    fn write_bin(path: &std::path::Path, v: &[f32]) {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for &x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn q8_bytes(values: &[f32]) -> Vec<u8> {
        use crate::inference::{quantize_q8_0_block, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMS};
        let blocks = values.len().div_ceil(Q8_0_BLOCK_ELEMS);
        let mut out = vec![0u8; blocks * Q8_0_BLOCK_BYTES];
        for block in 0..blocks {
            let start = block * Q8_0_BLOCK_ELEMS;
            let end = (start + Q8_0_BLOCK_ELEMS).min(values.len());
            quantize_q8_0_block(
                &values[start..end],
                &mut out[block * Q8_0_BLOCK_BYTES..(block + 1) * Q8_0_BLOCK_BYTES],
            );
        }
        out
    }

    fn tiny_qwen3_moe_loader_cfg() -> RealModelConfig {
        RealModelConfig {
            vocab_size: 8,
            d_model: 4,
            d_ff: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            num_layers: 1,
            num_experts: 2,
            top_k: 1,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: Architecture::Qwen3Moe,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        }
    }

    /// Write a complete converted-directory raw-`.bin` fixture (every
    /// strict-required base tensor except QK-Norm) for a single-layer MoE
    /// config. Each tensor is filled with `1.0` and sized from the config so
    /// the strict loader accepts it. QK-Norm tensors are supplied separately
    /// per test.
    fn write_converted_moe_base_raw(cfg: &RealModelConfig, dir: &Path) {
        let d_model = cfg.d_model;
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let write = |name: &str, len: usize| {
            std::fs::write(dir.join(name), f32_bytes(1.0, len)).unwrap();
        };
        write("embed.bin", cfg.vocab_size * d_model);
        write("final_rms.bin", d_model);
        write("lm_head.bin", cfg.vocab_size * d_model);
        for l in 0..cfg.num_layers {
            write(&format!("rms_attn_{l}.bin"), d_model);
            write(&format!("rms_moe_{l}.bin"), d_model);
            write(&format!("attn_{l}_q.bin"), q_dim * d_model);
            write(&format!("attn_{l}_k.bin"), kv_dim * d_model);
            write(&format!("attn_{l}_v.bin"), kv_dim * d_model);
            write(&format!("attn_{l}_o.bin"), d_model * q_dim);
            write(&format!("gate_{l}.bin"), cfg.num_experts * d_model);
        }
    }

    /// Write a `dense_manifest.json` mapping `canonical -> file` under the
    /// engine alias `alias`, with a payload of `values` as little-endian f32,
    /// matching the manifest metadata the GGUF converter emits for QK-Norm.
    fn write_qk_norm_manifest(dir: &Path, entries: &[(&str, &str, &str, Vec<usize>, Vec<f32>)]) {
        let mut tensors = Vec::new();
        for (canonical, file, alias, dims, values) in entries {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for v in values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            std::fs::write(dir.join(file), &bytes).unwrap();
            // Match the GGUF converter's `dense_manifest_dims`, which normalizes
            // 1D tensors to `[n, 1]`, so fixtures exercise the same shape
            // encoding the loader sees for real manifests.
            let dims: Vec<usize> = match dims.as_slice() {
                [n] => vec![*n, 1],
                other => other.to_vec(),
            };
            tensors.push(serde_json::json!({
                "canonical_name": canonical,
                "file": file,
                "aliases": [alias],
                "dtype": "f32",
                "dims": dims,
                "byte_len": bytes.len(),
                "checksum": crate::dense_tensor::dense_checksum(&bytes),
            }));
        }
        std::fs::write(
            dir.join("dense_manifest.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "format_version": 1,
                "tensors": tensors,
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn strict_converted_qwen3_moe_manifest_loads_qk_norm() {
        // A. Strict converted Qwen3-MoE manifest load succeeds and the QK-Norm
        // tensors carry the fixture's sentinel values (not seeded unit norm).
        let cfg = tiny_qwen3_moe_loader_cfg();
        assert!(cfg.architecture.uses_qk_norm());
        let dir = TempDir::new("qwen3_moe_qk_norm_ok");
        write_converted_moe_base_raw(&cfg, &dir.path);
        let q_sentinel = vec![0.25f32, 0.75];
        let k_sentinel = vec![0.5f32, 0.9];
        write_qk_norm_manifest(
            &dir.path,
            &[
                (
                    "blk.0.attn_q_norm.weight",
                    "dense_blk_0_attn_q_norm_weight.f32.bin",
                    "q_norm_0.bin",
                    vec![cfg.head_dim],
                    q_sentinel.clone(),
                ),
                (
                    "blk.0.attn_k_norm.weight",
                    "dense_blk_0_attn_k_norm_weight.f32.bin",
                    "k_norm_0.bin",
                    vec![cfg.head_dim],
                    k_sentinel.clone(),
                ),
            ],
        );

        let model = RealModel::from_dir_with_options(
            cfg.clone(),
            &dir.path,
            1,
            RealModelLoadOptions {
                strict_weights: true,
            },
        )
        .expect("strict converted Qwen3-MoE load with QK-Norm manifest should succeed");

        let q_norm = model.layers[0]
            .attn
            .q_norm
            .as_ref()
            .expect("q_norm must be present");
        let k_norm = model.layers[0]
            .attn
            .k_norm
            .as_ref()
            .expect("k_norm must be present");
        assert_eq!(q_norm.weight, q_sentinel);
        assert_eq!(k_norm.weight, k_sentinel);
        // Seeded QK-Norm is unit weight; the loaded sentinels must differ.
        assert_ne!(q_norm.weight, vec![1.0f32; cfg.head_dim]);
        assert_ne!(k_norm.weight, vec![1.0f32; cfg.head_dim]);
        assert_ne!(q_norm.weight, k_norm.weight);
    }

    #[test]
    fn strict_converted_qwen3_moe_reports_missing_qk_norm() {
        // B. Missing QK-Norm must be reported as Missing (never Unsupported).
        let cfg = tiny_qwen3_moe_loader_cfg();
        let dir = TempDir::new("qwen3_moe_qk_norm_missing");
        write_converted_moe_base_raw(&cfg, &dir.path);

        let err = match RealModel::from_dir_with_options(
            cfg,
            &dir.path,
            1,
            RealModelLoadOptions {
                strict_weights: true,
            },
        ) {
            Ok(_) => panic!("strict load must reject missing QK-Norm"),
            Err(err) => err,
        };
        let strict = strict_error(&err);
        let missing = |name: &str| {
            strict
                .failures()
                .iter()
                .any(|f| f.tensor == name && f.kind == WeightLoadFailureKind::Missing)
        };
        let unsupported = |name: &str| {
            strict
                .failures()
                .iter()
                .any(|f| f.tensor == name && f.kind == WeightLoadFailureKind::Unsupported)
        };
        assert!(missing("q_norm_0.bin"), "q_norm must be reported Missing");
        assert!(missing("k_norm_0.bin"), "k_norm must be reported Missing");
        assert!(
            !unsupported("q_norm_0.bin"),
            "q_norm must not be Unsupported"
        );
        assert!(
            !unsupported("k_norm_0.bin"),
            "k_norm must not be Unsupported"
        );
    }

    #[test]
    fn strict_converted_qwen3_moe_rejects_qk_norm_shape_mismatch() {
        // C. A QK-Norm tensor whose element count is not `head_dim` is a
        // ShapeMismatch.
        let cfg = tiny_qwen3_moe_loader_cfg();
        let dir = TempDir::new("qwen3_moe_qk_norm_shape");
        write_converted_moe_base_raw(&cfg, &dir.path);
        write_qk_norm_manifest(
            &dir.path,
            &[
                (
                    "blk.0.attn_q_norm.weight",
                    "dense_blk_0_attn_q_norm_weight.f32.bin",
                    "q_norm_0.bin",
                    // One element too many: not `head_dim`.
                    vec![cfg.head_dim + 1],
                    vec![0.1f32; cfg.head_dim + 1],
                ),
                (
                    "blk.0.attn_k_norm.weight",
                    "dense_blk_0_attn_k_norm_weight.f32.bin",
                    "k_norm_0.bin",
                    vec![cfg.head_dim],
                    vec![0.2f32; cfg.head_dim],
                ),
            ],
        );

        let err = match RealModel::from_dir_with_options(
            cfg,
            &dir.path,
            1,
            RealModelLoadOptions {
                strict_weights: true,
            },
        ) {
            Ok(_) => panic!("strict load must reject QK-Norm shape mismatch"),
            Err(err) => err,
        };
        let strict = strict_error(&err);
        assert!(
            strict.failures().iter().any(|f| {
                f.tensor == "q_norm_0.bin" && f.kind == WeightLoadFailureKind::ShapeMismatch
            }),
            "wrong-length q_norm must be reported ShapeMismatch"
        );
    }

    fn tiny_mixtral_loader_cfg() -> RealModelConfig {
        RealModelConfig {
            vocab_size: 8,
            d_model: 4,
            d_ff: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            num_layers: 1,
            num_experts: 2,
            top_k: 1,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: Architecture::Mixtral,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        }
    }

    fn f32_bytes(value: f32, len: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(len * 4);
        for _ in 0..len {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn write_safetensors_f32(
        dir: &TempDir,
        specs: Vec<(String, Vec<usize>, f32)>,
    ) -> std::path::PathBuf {
        use safetensors::serialize_to_file;
        use safetensors::tensor::{Dtype, TensorView};

        let bytes: Vec<Vec<u8>> = specs
            .iter()
            .map(|(_, shape, fill)| f32_bytes(*fill, shape.iter().product()))
            .collect();
        let tensors: Vec<(String, TensorView)> = specs
            .iter()
            .zip(bytes.iter())
            .map(|((name, shape, _), bytes)| {
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        let out_path = dir.path.join("model.safetensors");
        serialize_to_file(tensors, &None, &out_path).unwrap();
        out_path
    }

    fn complete_tiny_mixtral_specs(cfg: &RealModelConfig) -> Vec<(String, Vec<usize>, f32)> {
        let naming = cfg.tensor_naming();
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        vec![
            (naming.embed(), vec![cfg.vocab_size, cfg.d_model], 0.11),
            (naming.final_norm(), vec![cfg.d_model], 1.0),
            (naming.lm_head(), vec![cfg.vocab_size, cfg.d_model], 0.12),
            (naming.input_layernorm(0), vec![cfg.d_model], 1.0),
            (naming.post_attention_layernorm(0), vec![cfg.d_model], 1.0),
            (naming.attn_q(0), vec![q_dim, cfg.d_model], 0.21),
            (naming.attn_k(0), vec![kv_dim, cfg.d_model], 0.22),
            (naming.attn_v(0), vec![kv_dim, cfg.d_model], 0.23),
            (naming.attn_o(0), vec![cfg.d_model, q_dim], 0.24),
            (naming.moe_gate(0), vec![cfg.num_experts, cfg.d_model], 0.31),
        ]
    }

    fn strict_error(err: &std::io::Error) -> &StrictWeightLoadError {
        err.get_ref()
            .and_then(|e| e.downcast_ref::<StrictWeightLoadError>())
            .expect("loader error should carry StrictWeightLoadError")
    }

    #[test]
    fn from_dir_loads_native_q8_dense_manifest_without_alias_files() {
        // Q8_0 tensors must contain a whole number of 32-element blocks (F6).
        // Size the fixture so every Q8_0 dense tensor's element count is a
        // multiple of 32: d_model = 32 makes `k * d_model` a valid block
        // multiple for any integer `k` (vocab, q/kv widths, num_experts).
        let cfg = RealModelConfig {
            vocab_size: 32,
            d_model: 32,
            d_ff: 32,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            num_layers: 1,
            num_experts: 2,
            top_k: 1,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: Architecture::Mixtral,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        };
        let dir = TempDir::new("dense_manifest_q8");
        let q_dim = cfg.num_heads * cfg.head_dim;
        let kv_dim = cfg.num_kv_heads * cfg.head_dim;
        let mut tensors = Vec::new();
        let mut add = |canonical: &str,
                       file: &str,
                       aliases: Vec<String>,
                       dtype: &str,
                       dims: Vec<usize>,
                       bytes: Vec<u8>| {
            std::fs::write(dir.path.join(file), &bytes).unwrap();
            tensors.push(serde_json::json!({
                "canonical_name": canonical,
                "file": file,
                "aliases": aliases,
                "dtype": dtype,
                "dims": dims,
                "byte_len": bytes.len(),
                "checksum": crate::dense_tensor::dense_checksum(&bytes),
            }));
        };
        let q8_values = |len: usize, offset: f32| {
            (0..len)
                .map(|i| ((i % 11) as f32 - 5.0) / 7.0 + offset)
                .collect::<Vec<f32>>()
        };
        let f32_manifest_bytes = |value: f32, len: usize| {
            let mut bytes = Vec::with_capacity(len * 4);
            for _ in 0..len {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            bytes
        };
        let embed_values = q8_values(cfg.vocab_size * cfg.d_model, 0.0);
        let embed_bytes = q8_bytes(&embed_values);
        add(
            "token_embd.weight",
            "dense_token_embd_weight.q8_0.bin",
            vec!["embed.bin".to_string()],
            "q8_0",
            vec![cfg.vocab_size, cfg.d_model],
            embed_bytes.clone(),
        );
        add(
            "output_norm.weight",
            "dense_output_norm_weight.f32.bin",
            vec!["final_rms.bin".to_string()],
            "f32",
            vec![cfg.d_model, 1],
            f32_manifest_bytes(1.0, cfg.d_model),
        );
        add(
            "output.weight",
            "dense_output_weight.q8_0.bin",
            vec!["lm_head.bin".to_string()],
            "q8_0",
            vec![cfg.vocab_size, cfg.d_model],
            q8_bytes(&q8_values(cfg.vocab_size * cfg.d_model, 0.1)),
        );
        for (canonical, alias, rows, cols, offset) in [
            ("blk.0.attn_q.weight", "attn_0_q.bin", q_dim, cfg.d_model, 0.2),
            ("blk.0.attn_k.weight", "attn_0_k.bin", kv_dim, cfg.d_model, 0.3),
            ("blk.0.attn_v.weight", "attn_0_v.bin", kv_dim, cfg.d_model, 0.4),
            ("blk.0.attn_output.weight", "attn_0_o.bin", cfg.d_model, q_dim, 0.5),
        ] {
            add(
                canonical,
                &format!("{canonical}.q8_0.bin").replace('.', "_"),
                vec![alias.to_string()],
                "q8_0",
                vec![rows, cols],
                q8_bytes(&q8_values(rows * cols, offset)),
            );
        }
        add(
            "blk.0.attn_norm.weight",
            "dense_blk_0_attn_norm_weight.f32.bin",
            vec!["rms_attn_0.bin".to_string()],
            "f32",
            vec![cfg.d_model, 1],
            f32_manifest_bytes(1.0, cfg.d_model),
        );
        add(
            "blk.0.ffn_norm.weight",
            "dense_blk_0_ffn_norm_weight.f32.bin",
            vec!["rms_moe_0.bin".to_string()],
            "f32",
            vec![cfg.d_model, 1],
            f32_manifest_bytes(1.0, cfg.d_model),
        );
        add(
            "blk.0.ffn_gate_inp.weight",
            "dense_blk_0_ffn_gate_inp_weight.q8_0.bin",
            vec!["gate_0.bin".to_string()],
            "q8_0",
            vec![cfg.num_experts, cfg.d_model],
            q8_bytes(&q8_values(cfg.num_experts * cfg.d_model, 0.6)),
        );
        std::fs::write(
            dir.path.join("dense_manifest.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "format_version": 1,
                "tensors": tensors,
            }))
            .unwrap(),
        )
        .unwrap();

        let model = RealModel::from_dir_with_options(
            cfg.clone(),
            &dir.path,
            1,
            RealModelLoadOptions {
                strict_weights: true,
            },
        )
        .unwrap();
        assert!(!dir.path.join("embed.bin").exists());
        assert_eq!(model.embedding.dtype(), crate::dense_tensor::DenseDType::Q8_0);
        assert_eq!(
            model.layers[0].attn.wq.dtype(),
            crate::dense_tensor::DenseDType::Q8_0
        );
        assert_eq!(
            model.layers[0].gate.weights.dtype(),
            crate::dense_tensor::DenseDType::Q8_0
        );
        let expected_embedding =
            DenseWeight::from_q8_0_bytes(embed_bytes, cfg.vocab_size, cfg.d_model).unwrap();
        let mut expected_row = Vec::new();
        expected_embedding.row_dequant_into(1, &mut expected_row);
        assert_eq!(model.embed(1), expected_row);
    }

    #[test]
    fn from_dir_strict_reports_all_missing_and_malformed_required_tensors() {
        let cfg = tiny_mixtral_loader_cfg();
        let dir = TempDir::new("strict_bin_bad");
        std::fs::write(dir.path.join("embed.bin"), [0u8, 1, 2]).unwrap();

        let err = match RealModel::from_dir_with_options(
            cfg,
            &dir.path,
            1,
            RealModelLoadOptions {
                strict_weights: true,
            },
        ) {
            Ok(_) => panic!("strict raw .bin load should reject malformed and missing tensors"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let strict = strict_error(&err);
        assert!(
            strict
                .failures()
                .iter()
                .any(|f| { f.tensor == "embed.bin" && f.kind == WeightLoadFailureKind::Malformed }),
            "malformed present tensor must be reported"
        );
        assert!(
            strict
                .failures()
                .iter()
                .any(|f| { f.tensor == "lm_head.bin" && f.kind == WeightLoadFailureKind::Missing }),
            "strict mode must keep collecting after the first failure"
        );
        assert!(
            strict
                .failures()
                .iter()
                .any(|f| { f.tensor == "gate_0.bin" && f.group == GROUP_ROUTING_GATES }),
            "MoE router gate is required for sparse layers"
        );
    }

    #[test]
    fn from_safetensors_strict_reports_missing_and_shape_mismatched_tensors() {
        let cfg = tiny_mixtral_loader_cfg();
        let dir = TempDir::new("strict_st_bad");
        write_safetensors_f32(
            &dir,
            vec![(cfg.tensor_naming().embed(), vec![1, cfg.d_model], 0.5)],
        );

        let err = match RealModel::from_safetensors_with_options(
            cfg,
            &dir.path,
            1,
            RealModelLoadOptions {
                strict_weights: true,
            },
        ) {
            Ok(_) => panic!("strict safetensors load should reject an incomplete checkpoint"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let strict = strict_error(&err);
        assert!(
            strict.failures().iter().any(|f| {
                f.tensor == "model.embed_tokens.weight"
                    && f.kind == WeightLoadFailureKind::ShapeMismatch
            }),
            "bad embed shape must be reported"
        );
        assert!(
            strict.failures().iter().any(|f| {
                f.tensor == "lm_head.weight" && f.kind == WeightLoadFailureKind::Missing
            }),
            "missing LM head must be reported in the same aggregate error"
        );
        assert!(
            strict.failures().len() > 2,
            "strict error should aggregate every required tensor failure"
        );
    }

    #[test]
    fn from_safetensors_strict_accepts_complete_tiny_mixtral_checkpoint() {
        let cfg = tiny_mixtral_loader_cfg();
        let dir = TempDir::new("strict_st_good");
        write_safetensors_f32(&dir, complete_tiny_mixtral_specs(&cfg));

        let model = RealModel::from_safetensors_with_options(
            cfg.clone(),
            &dir.path,
            1,
            RealModelLoadOptions {
                strict_weights: true,
            },
        )
        .unwrap();
        assert!(model.embedding.iter().all(|x| x == 0.11));
        assert!(model.lm_head.weights.iter().all(|x| x == 0.12));
        assert!(model.layers[0].attn.wq.iter().all(|x| x == 0.21));
        assert_eq!(model.layers[0].gate.num_experts, cfg.num_experts);
    }

    // ---- Finding 8: gate sidecar must never be truncated ----

    /// Load a complete tiny-Mixtral safetensors checkpoint plus a
    /// `gate_0.bin` sidecar of the given length/value and return the
    /// resolved router-gate weights so tests can prove which source was used.
    /// The canonical safetensors gate has value 0.31; a sidecar written here
    /// uses a distinct value.
    fn load_with_gate_sidecar(sidecar_len: usize, sidecar_val: f32) -> Vec<f32> {
        let cfg = tiny_mixtral_loader_cfg();
        let dir = TempDir::new("gate_sidecar");
        write_safetensors_f32(&dir, complete_tiny_mixtral_specs(&cfg));
        write_bin(&dir.path.join("gate_0.bin"), &vec![sidecar_val; sidecar_len]);
        let model = RealModel::from_safetensors_with_options(
            cfg,
            &dir.path,
            1,
            RealModelLoadOptions {
                strict_weights: true,
            },
        )
        .unwrap();
        model.layers[0].gate.weights.iter().collect()
    }

    #[test]
    fn exact_sized_gate_sidecar_is_preferred() {
        let cfg = tiny_mixtral_loader_cfg();
        let expected = cfg.num_experts * cfg.d_model;
        let gate = load_with_gate_sidecar(expected, 0.77);
        assert!(
            gate.iter().all(|&x| (x - 0.77).abs() < 1e-6),
            "exact-sized sidecar (0.77) must be preferred over the canonical gate (0.31)"
        );
    }

    #[test]
    fn oversized_gate_sidecar_is_not_truncated_and_falls_back_to_canonical() {
        let cfg = tiny_mixtral_loader_cfg();
        let expected = cfg.num_experts * cfg.d_model;
        // One extra expert-row worth of data: must NOT be truncated to fit.
        let gate = load_with_gate_sidecar(expected + cfg.d_model, 0.77);
        assert!(
            gate.iter().all(|&x| (x - 0.31).abs() < 1e-6),
            "oversized sidecar must be rejected (not truncated); canonical gate (0.31) is used"
        );
    }

    #[test]
    fn undersized_gate_sidecar_falls_back_to_canonical() {
        let cfg = tiny_mixtral_loader_cfg();
        let expected = cfg.num_experts * cfg.d_model;
        let gate = load_with_gate_sidecar(expected - 1, 0.77);
        assert!(
            gate.iter().all(|&x| (x - 0.31).abs() < 1e-6),
            "undersized sidecar must fall back to the canonical gate (0.31)"
        );
    }

    #[test]
    fn wrong_gate_sidecar_without_canonical_fails_strict_loading() {
        let cfg = tiny_mixtral_loader_cfg();
        let expected = cfg.num_experts * cfg.d_model;
        let dir = TempDir::new("gate_sidecar_no_canonical");
        // Complete checkpoint minus the canonical MoE gate tensor.
        let specs: Vec<_> = complete_tiny_mixtral_specs(&cfg)
            .into_iter()
            .filter(|(name, _, _)| *name != cfg.tensor_naming().moe_gate(0))
            .collect();
        write_safetensors_f32(&dir, specs);
        // Oversized sidecar — must not be truncated to satisfy the gate.
        write_bin(
            &dir.path.join("gate_0.bin"),
            &vec![0.77f32; expected + cfg.d_model],
        );
        let err = match RealModel::from_safetensors_with_options(
            cfg,
            &dir.path,
            1,
            RealModelLoadOptions {
                strict_weights: true,
            },
        ) {
            Ok(_) => panic!("wrong-sized sidecar with no canonical gate must fail strict loading"),
            Err(err) => err,
        };
        let strict = strict_error(&err);
        assert!(
            strict
                .failures()
                .iter()
                .any(|f| f.group == GROUP_ROUTING_GATES),
            "missing valid router gate must be reported under strict loading"
        );
    }

    /// `from_dir` must pick up the GGUF-extractor shared-expert `.bin`
    /// files, infer the shared intermediate size from the gate tensor
    /// length, and populate `TransformerLayer::shared_expert`. Models
    /// without those files (Mixtral) keep `shared_expert == None`.
    #[test]
    fn from_dir_loads_shared_expert_and_infers_d_ff() {
        let mut cfg = RealModelConfig::tiny();
        // Shared experts are only probed when the checkpoint declares them
        // (Finding 1); tiny() is Mixtral (none), so opt in for this fixture.
        cfg.advanced.num_shared_experts = 1;
        let d_model = cfg.d_model;
        let shared_d_ff = 5; // deliberately != routed d_ff (64)
        let dir = TempDir::new("shexp_bin");

        write_bin(
            &dir.path.join("layer_0_shexp_gate.bin"),
            &vec![0.1f32; shared_d_ff * d_model],
        );
        write_bin(
            &dir.path.join("layer_0_shexp_up.bin"),
            &vec![0.2f32; shared_d_ff * d_model],
        );
        write_bin(
            &dir.path.join("layer_0_shexp_down.bin"),
            &vec![0.3f32; d_model * shared_d_ff],
        );
        write_bin(
            &dir.path.join("layer_0_shexp_gate_inp.bin"),
            &vec![0.0f32; d_model],
        );

        let model = RealModel::from_dir(cfg.clone(), &dir.path, 1).unwrap();
        let se = model.layers[0]
            .shared_expert
            .as_ref()
            .expect("shared expert should be loaded");
        assert_eq!(
            se.d_ff, shared_d_ff,
            "d_ff inferred from gate tensor length"
        );
        assert_eq!(se.d_model, d_model);
        assert!(se.gate_inp.is_some(), "sigmoid gate present");
        // forward produces a finite, d_model-length vector.
        let x: Vec<f32> = (0..d_model).map(|i| 0.01 * i as f32).collect();
        let y = se.forward(&x).expect("shared expert forward succeeds");
        assert_eq!(y.len(), d_model);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    /// A layer with no shared-expert files stays `None`, so Mixtral-style
    /// models are unaffected.
    #[test]
    fn from_dir_without_shared_expert_is_none() {
        let cfg = RealModelConfig::tiny();
        let dir = TempDir::new("shexp_absent");
        let model = RealModel::from_dir(cfg, &dir.path, 1).unwrap();
        assert!(model.layers[0].shared_expert.is_none());
    }

    /// `norm_topk_prob` must default to `false` so a TOML-only (no
    /// `config.json`) Mixtral run does not silently apply Qwen3-style
    /// top-K renormalisation. Explicit `true` is only honoured when a
    /// checkpoint's `config.json` sets it (see `from_hf_config`).
    #[test]
    fn advanced_config_norm_topk_prob_defaults_false() {
        assert!(!AdvancedConfig::default().norm_topk_prob);
    }

    /// Inconsistent shared-expert tensors (down proj wrong length) must
    /// degrade to `None` rather than abort the load.
    #[test]
    fn from_dir_inconsistent_shared_expert_is_ignored() {
        let mut cfg = RealModelConfig::tiny();
        cfg.advanced.num_shared_experts = 1;
        let d_model = cfg.d_model;
        let shared_d_ff = 5;
        let dir = TempDir::new("shexp_bad");
        write_bin(
            &dir.path.join("layer_0_shexp_gate.bin"),
            &vec![0.1f32; shared_d_ff * d_model],
        );
        write_bin(
            &dir.path.join("layer_0_shexp_up.bin"),
            &vec![0.2f32; shared_d_ff * d_model],
        );
        // Wrong length for down -> from_projections returns None.
        write_bin(
            &dir.path.join("layer_0_shexp_down.bin"),
            &vec![0.3f32; d_model * shared_d_ff + 1],
        );
        let model = RealModel::from_dir(cfg, &dir.path, 1).unwrap();
        assert!(model.layers[0].shared_expert.is_none());
    }

    /// The Qwen2-MoE sigmoid gate scales the shared expert output by
    /// `sigmoid(W_gate · x)`. A zero gate vector yields a 0.5 scale; an
    /// absent gate yields an unscaled (1.0) output (DeepSeek-MoE).
    #[test]
    fn shared_expert_sigmoid_gate_scales_output() {
        let d_model = 4;
        let d_ff = 3;
        let gate = vec![0.05f32; d_ff * d_model];
        let up = vec![0.07f32; d_ff * d_model];
        let down = vec![0.09f32; d_model * d_ff];
        let x: Vec<f32> = vec![0.5, -0.25, 0.1, 0.2];

        let ungated =
            SharedExpert::from_projections(d_model, d_ff, &gate, &up, &down, None).unwrap();
        let y_ungated = ungated.forward(&x).expect("ungated forward succeeds");

        // Zero gate -> sigmoid(0) = 0.5 -> output halved.
        let gated = SharedExpert::from_projections(
            d_model,
            d_ff,
            &gate,
            &up,
            &down,
            Some(vec![0.0f32; d_model]),
        )
        .unwrap();
        let y_gated = gated.forward(&x).expect("gated forward succeeds");
        for (a, b) in y_ungated.iter().zip(y_gated.iter()) {
            assert!(
                (a * 0.5 - b).abs() < 1e-6,
                "gate=0 must halve output: {a} {b}"
            );
        }
    }

    // ---- Finding 1: required-vs-optional load accounting ----

    /// A complete strict Mixtral converted directory (no shared experts) loads
    /// every required tensor, probes no optional tensors, and reports no
    /// lingering seeded fallback — so `bench-real` accepts it.
    #[test]
    fn complete_strict_mixtral_converted_dir_reports_no_seeded_fallback() {
        let cfg = RealModelConfig::tiny(); // Mixtral, num_shared_experts == 0
        assert_eq!(cfg.advanced.num_shared_experts, 0);
        let dir = TempDir::new("mixtral_complete_strict");
        write_converted_moe_base_raw(&cfg, &dir.path);
        let model = RealModel::from_dir_with_options(
            cfg,
            &dir.path,
            1,
            RealModelLoadOptions {
                strict_weights: true,
            },
        )
        .expect("complete strict Mixtral converted dir must load");
        let st = &model.load_status;
        assert_eq!(st.loaded_tensors, st.required_tensors);
        assert!(st.required_tensors > 0);
        assert_eq!(st.optional_probed, 0, "no shared experts to probe");
        assert!(
            !st.seeded_fallback_remained,
            "a complete strict load must not retain a seeded fallback"
        );
    }

    /// A complete strict Qwen3-MoE converted directory (QK-Norm required, no
    /// shared experts) reports no seeded fallback.
    #[test]
    fn complete_strict_qwen3_moe_converted_dir_reports_no_seeded_fallback() {
        let cfg = tiny_qwen3_moe_loader_cfg(); // Qwen3-MoE, no shared experts
        assert_eq!(cfg.advanced.num_shared_experts, 0);
        let dir = TempDir::new("qwen3_moe_complete_strict");
        write_converted_moe_base_raw(&cfg, &dir.path);
        write_qk_norm_manifest(
            &dir.path,
            &[
                (
                    "blk.0.attn_q_norm.weight",
                    "dense_blk_0_attn_q_norm_weight.f32.bin",
                    "q_norm_0.bin",
                    vec![cfg.head_dim],
                    vec![0.25f32, 0.75],
                ),
                (
                    "blk.0.attn_k_norm.weight",
                    "dense_blk_0_attn_k_norm_weight.f32.bin",
                    "k_norm_0.bin",
                    vec![cfg.head_dim],
                    vec![0.5f32, 0.9],
                ),
            ],
        );
        let model = RealModel::from_dir_with_options(
            cfg,
            &dir.path,
            1,
            RealModelLoadOptions {
                strict_weights: true,
            },
        )
        .expect("complete strict Qwen3-MoE converted dir must load");
        let st = &model.load_status;
        assert_eq!(st.loaded_tensors, st.required_tensors);
        assert_eq!(st.optional_probed, 0);
        assert!(!st.seeded_fallback_remained);
    }

    /// When shared experts are declared (`num_shared_experts > 0`) their tensor
    /// set is required: a converted directory missing the shared-expert files
    /// fails strict loading.
    #[test]
    fn declared_shared_expert_missing_required_weights_fails_strict() {
        let mut cfg = RealModelConfig::tiny();
        cfg.advanced.num_shared_experts = 1;
        let dir = TempDir::new("shared_required_missing");
        // Complete base tensors, but deliberately no `layer_*_shexp_*` files.
        write_converted_moe_base_raw(&cfg, &dir.path);
        let err = match RealModel::from_dir_with_options(
            cfg,
            &dir.path,
            1,
            RealModelLoadOptions {
                strict_weights: true,
            },
        ) {
            Ok(_) => panic!("declared-but-missing shared experts must fail strict loading"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(msg.contains("shexp") || msg.contains("shared"), "{msg}");
    }

    /// An architecture without shared experts is complete even though the
    /// optional shared-expert sidecar files are absent: the absent sidecar
    /// does not affect the required counts or the seeded-fallback signal.
    #[test]
    fn absent_optional_shared_sidecar_does_not_affect_required_counts() {
        let cfg = RealModelConfig::tiny(); // no shared experts
        let dir = TempDir::new("optional_absent");
        write_converted_moe_base_raw(&cfg, &dir.path);
        let with_probe = RealModel::from_dir_with_options(
            cfg.clone(),
            &dir.path,
            1,
            RealModelLoadOptions {
                strict_weights: true,
            },
        )
        .expect("complete strict load without shared experts");
        assert!(!with_probe.load_status.seeded_fallback_remained);
        assert_eq!(
            with_probe.load_status.required_tensors,
            with_probe.load_status.loaded_tensors
        );
    }

    // ---- Finding 3: safetensors / manifest orientation validation ----

    #[test]
    fn safetensor_orientation_accepts_exact_rank2_only() {
        // Exact [rows, cols].
        assert!(safetensor_orientation_ok(&[8, 4], 8, 4));
        // Square matrix accepted (exact rank-2 match).
        assert!(safetensor_orientation_ok(&[4, 4], 4, 4));
        // Finding 4: a flattened rank-1 tensor is NOT a valid matrix even
        // when its element count matches; matrix loaders require rank-2.
        assert!(!safetensor_orientation_ok(&[32], 8, 4));
        // A bare rank-1 vector is rejected by the matrix helper; rank-1
        // compatibility lives only in the vector-specific loaders.
        assert!(!safetensor_orientation_ok(&[8], 8, 1));
    }

    #[test]
    fn safetensor_orientation_rejects_transposed_and_higher_rank() {
        // Transposed [cols, rows]: same element count, wrong orientation.
        assert!(!safetensor_orientation_ok(&[4, 8], 8, 4));
        // Wrong shape entirely.
        assert!(!safetensor_orientation_ok(&[2, 16], 8, 4));
        // Finding 4: rank > 2 is rejected outright, never element-product.
        assert!(!safetensor_orientation_ok(&[8, 4, 1], 8, 4));
        assert!(!safetensor_orientation_ok(&[2, 4, 4], 8, 4));
    }

    #[test]
    fn manifest_orientation_rejects_transposed_nonsquare() {
        // Recorded [rows, cols] accepted; transpose rejected; square ok.
        assert!(dense_dims_match_orientation(&[8, 4], 8, 4));
        assert!(!dense_dims_match_orientation(&[4, 8], 8, 4));
        assert!(dense_dims_match_orientation(&[4, 4], 4, 4));
        // 1-D column vector compatibility only.
        assert!(dense_dims_match_orientation(&[8], 8, 1));
        assert!(!dense_dims_match_orientation(&[8], 8, 4));
    }
}
