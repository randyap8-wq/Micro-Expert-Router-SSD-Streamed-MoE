//! OpenAI-compatible HTTP server (gist Phase 6).
//!
//! Endpoints:
//! - `POST /v1/completions`      — text completion
//! - `POST /v1/chat/completions` — chat completion (single-turn, no streaming yet)
//! - `GET  /health`              — liveness
//! - `GET  /metrics`             — Prometheus metrics
//!
//! The server is stateless per request at the HTTP layer, but the
//! real-transformer path routes per-token decoder steps through a
//! shared [`crate::batch_scheduler::BatchScheduler`]: an `mpsc`-fed
//! background task fuses concurrent requests' steps into a single
//! batch (up to `max_batch_size` or `batch_timeout_ms`) and runs them
//! concurrently against the shared engine, so multiple HTTP clients
//! amortise each round of SSD-streamed expert FFN compute. Per-request
//! KV caches are moved into the scheduler and back, so attention state
//! stays strictly per-request.
//!
//! Generation strategy: the synthesised hidden state from
//! [`crate::inference::synth_hidden_state`] drives one
//! `Engine::generate` call per token. The "decoded" response text is
//! the round-tripped token ids through the configured tokenizer (real
//! HF tokenizer if `tokenizer.json` is present, byte tokenizer otherwise).
//! Once a real transformer decoder is wired (the [`crate::transformer`]
//! pieces), this loop swaps to producing real next-token logits — the
//! HTTP shape doesn't change.

use crate::backend::Backend;
use crate::batch_scheduler::{BatchError, BatchScheduler, RequestId};
use crate::config::LiveConfig;
use crate::engine::Engine;
use crate::metrics::Metrics;
use crate::middleware::{
    admission_layer, api_key_layer, rate_limit_layer, request_id_layer, MiddlewareState,
};
use crate::model::RealModel;
use crate::sampling::SamplingParams;
use crate::session::{SessionCheckout, SessionState, SessionStore};
use crate::tokenizer::Tokenizer;
use axum::{
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info};

/// Shared handler state. Cheap to clone — everything is `Arc`.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Engine>,
    pub tokenizer: Arc<Tokenizer>,
    pub metrics: Metrics,
    /// Optional real transformer. When set, [`generate`] runs the
    /// embedding → stacked layers → LM head pipeline (with experts
    /// streamed from SSD by the engine on every layer's MoE step) and
    /// samples next-token ids from real logits. When `None`, the
    /// legacy deterministic generator is used (the engine still drives
    /// SSD-streamed expert FFN compute either way, so cache / I/O
    /// metrics are populated identically).
    pub real_model: Option<Arc<RealModel>>,
    /// Optional continuous-batching scheduler. Always set together
    /// with `real_model` (see `cmd_serve` in `main.rs`); when `Some`,
    /// the real-transformer generation paths submit each per-token
    /// step to the scheduler instead of calling `RealModel::step`
    /// directly. The scheduler fuses concurrent requests' steps into a
    /// single batch so multiple HTTP clients share each round of
    /// SSD-streamed expert FFN compute.
    pub batch_scheduler: Option<Arc<BatchScheduler>>,
    /// Draft engine for speculative decoding. When `Some`, the generation
    /// loop calls `RealModel::step_speculative` instead of sequential
    /// `step_through_scheduler` calls, verifying `speculation_k` draft
    /// tokens per step with a single batched expert prefetch.
    /// Built from the main model's embedding via `DraftEngine::from_main`
    /// when `cfg.predictive.speculator_enabled = true`.
    pub draft_engine: Option<Arc<crate::draft::DraftEngine>>,

    /// Optional persistent GPU-native token loop for GPU-owned execution.
    pub gpu_native_token_loop: Option<Arc<crate::gpu_native_token_loop::GpuNativeTokenLoop>>,

    /// Number of draft tokens to attempt per speculative step.
    /// Populated from `cfg.real_transformer.speculation_base_depth`.
    /// `0` or `1` disables speculation (falls back to sequential).
    pub speculation_k: usize,
    /// Live, atomically-swappable runtime configuration. Reads on the
    /// hot path (sampling defaults, `max_tokens` cap) go through a
    /// single relaxed atomic load — no mutex, no contention across
    /// Tokio worker threads. SIGHUP refreshes this in place via
    /// [`LiveConfig::try_reload`]; in-flight requests keep observing
    /// their snapshot until they drop it. See
    /// [`crate::config::RuntimeConfig`].
    pub runtime: LiveConfig,
    /// Optional session store for KV-cache persistence between
    /// requests. `None` when `server.session_ttl_secs == 0`.
    pub sessions: Option<SessionStore>,
    /// Production-readiness middleware bundle: API-key gate, in-process
    /// rate limit, and admission controller. All three are no-ops
    /// when not configured, so this is safe to construct
    /// unconditionally.
    pub middleware: MiddlewareState,
}

/// Build the axum [`Router`] for the API.
///
/// Routes are split into two sub-routers:
///
/// * **Public**: `/health`, `/metrics`. These are required by
///   Kubernetes liveness/readiness probes and Prometheus scrapers,
///   which generally cannot present API keys and must not be subject
///   to per-tenant rate-limiting or admission shedding. They only
///   receive the request-id layer so logs remain correlated.
/// * **Protected**: every other route (`/v1/...`). These pass
///   through the full middleware stack: API key, in-process rate
///   limit, and admission control.
pub fn build_router(state: AppState) -> Router {
    let mw_state = state.middleware.clone();

    // Operational endpoints — always reachable, no auth/rate-limit/admission.
    let public = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics));

    // Tenant-facing endpoints — protected by the full middleware stack.
    let protected = Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/sessions/:id", delete(delete_session))
        .route("/v1/admin/health/experts", get(admin_health_experts))
        .route("/v1/admin/evict", post(admin_evict_overflow))
        .layer(axum::middleware::from_fn_with_state(
            mw_state.clone(),
            admission_layer,
        ))
        .layer(axum::middleware::from_fn_with_state(
            mw_state.clone(),
            rate_limit_layer,
        ))
        .layer(axum::middleware::from_fn_with_state(
            mw_state,
            api_key_layer,
        ));

    public
        .merge(protected)
        .layer(axum::middleware::from_fn(request_id_layer))
        .with_state(state)
}

// ----------------------------- /health ------------------------------

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
    })
}

// ----------------------------- /metrics ------------------------------

async fn metrics(State(state): State<AppState>) -> Response {
    match state.metrics.render() {
        Ok(mut body) => {
            if let Some(scheduler) = state.batch_scheduler.as_ref() {
                append_scheduler_metrics(&mut body, scheduler);
            }
            (
                StatusCode::OK,
                [("content-type", "text/plain; version=0.0.4")],
                body,
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("metrics render error: {e}"),
        )
            .into_response(),
    }
}

fn append_scheduler_metrics(body: &mut Vec<u8>, scheduler: &BatchScheduler) {
    let stats = scheduler.prepass_stats();
    let precision = stats.prefetch_precision();
    let metrics = format!(
        concat!(
            "\n# HELP mer_scheduler_prepass_runs_total Scheduler pre-pass batches with effective batch size greater than one.\n",
            "# TYPE mer_scheduler_prepass_runs_total counter\n",
            "mer_scheduler_prepass_runs_total {}\n",
            "# HELP mer_scheduler_prepass_singleton_skips_total Scheduler batches that skipped the pre-pass because effective batch size was one.\n",
            "# TYPE mer_scheduler_prepass_singleton_skips_total counter\n",
            "mer_scheduler_prepass_singleton_skips_total {}\n",
            "# HELP mer_scheduler_prepass_time_seconds_total Total scheduler pre-pass wall time in seconds.\n",
            "# TYPE mer_scheduler_prepass_time_seconds_total counter\n",
            "mer_scheduler_prepass_time_seconds_total {:.9}\n",
            "# HELP mer_scheduler_prepass_predicted_experts_total Raw expert prediction slots emitted by scheduler pre-pass peeks.\n",
            "# TYPE mer_scheduler_prepass_predicted_experts_total counter\n",
            "mer_scheduler_prepass_predicted_experts_total {}\n",
            "# HELP mer_scheduler_prepass_requested_experts_total Unique expert warm/fetch requests issued by the scheduler pre-pass after dedupe and placement.\n",
            "# TYPE mer_scheduler_prepass_requested_experts_total counter\n",
            "mer_scheduler_prepass_requested_experts_total {}\n",
            "# HELP mer_scheduler_prepass_prefetch_precision Unique warm/fetch requests divided by raw predicted expert slots.\n",
            "# TYPE mer_scheduler_prepass_prefetch_precision gauge\n",
            "mer_scheduler_prepass_prefetch_precision {:.9}\n",
            "# HELP mer_scheduler_prepass_local_warm_failures_total Local warm_with calls that failed during scheduler pre-pass.\n",
            "# TYPE mer_scheduler_prepass_local_warm_failures_total counter\n",
            "mer_scheduler_prepass_local_warm_failures_total {}\n",
            "# HELP mer_scheduler_prepass_gate_skips_total Batches the pre-pass profitability gate skipped because measured net benefit was non-positive.\n",
            "# TYPE mer_scheduler_prepass_gate_skips_total counter\n",
            "mer_scheduler_prepass_gate_skips_total {}\n",
            "# HELP mer_scheduler_prepass_useful_warm_experts_total Peeked local experts that were not resident and were actually fetched by the pre-pass warm.\n",
            "# TYPE mer_scheduler_prepass_useful_warm_experts_total counter\n",
            "mer_scheduler_prepass_useful_warm_experts_total {}\n",
            "# HELP mer_scheduler_prepass_cached_predictions_total Peeked local experts that were already resident (wasted prediction work).\n",
            "# TYPE mer_scheduler_prepass_cached_predictions_total counter\n",
            "mer_scheduler_prepass_cached_predictions_total {}\n",
        ),
        stats.runs_total,
        stats.singleton_skips_total,
        stats.time_ns_total as f64 / 1_000_000_000.0,
        stats.predicted_experts_total,
        stats.requested_experts_total,
        precision,
        stats.local_warm_failures_total,
        stats.gate_skips_total,
        stats.useful_warm_experts_total,
        stats.cached_predictions_total,
    );
    body.extend_from_slice(metrics.as_bytes());
}

// ------------------------ /v1/completions ----------------------------

#[derive(Deserialize, Debug)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Optional session id. When present, the per-layer KV caches for
    /// this session are restored before the prompt is ingested and
    /// stored back when the request completes — see `crate::session`
    /// for the persistence model. When absent, the request is
    /// stateless (matches the legacy behaviour bit-for-bit).
    #[serde(default)]
    pub session_id: Option<String>,
    /// When `true`, the response is streamed as Server-Sent Events
    /// (`text/event-stream`) — one OpenAI-style `text_completion` chunk
    /// per generated token, terminated by `data: [DONE]`. Defaults to
    /// `false` (single-shot JSON response).
    #[serde(default)]
    pub stream: Option<bool>,
}

fn default_max_tokens() -> usize {
    64
}
fn default_model() -> String {
    "micro-expert-router".to_string()
}

#[derive(Serialize, Debug)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: &'static str,
}

#[derive(Serialize, Debug)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize, Debug)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageStats,
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let started = Instant::now();
    let params = resolve_params(&state, req.temperature, req.top_p, req.top_k, req.seed);
    let session_id = req.session_id.clone();
    if req.stream.unwrap_or(false) {
        if state.gpu_native_token_loop.is_some() {
            state
                .metrics
                .record_request("/v1/completions", started.elapsed().as_secs_f64());
            return error_response(GenerateError::InvalidRequest(
                "streaming is unsupported in gpu_native mode in this slice".into(),
            ));
        }
        // Server-Sent Events streaming path: emit one OpenAI-style chunk
        // per generated token, terminated with `data: [DONE]`. The whole
        // SSD-streaming substrate is exercised inside the generator
        // exactly as in the non-streaming path.
        match build_completion_stream(
            &state,
            &req.prompt,
            req.max_tokens,
            req.model.clone(),
            params,
            session_id,
        )
        .await
        {
            Ok(s) => {
                state
                    .metrics
                    .record_request("/v1/completions", started.elapsed().as_secs_f64());
                return Sse::new(s).keep_alive(KeepAlive::default()).into_response();
            }
            Err(e) => {
                state
                    .metrics
                    .record_request("/v1/completions", started.elapsed().as_secs_f64());
                return error_response(e);
            }
        }
    }
    match generate(
        &state,
        &req.prompt,
        req.max_tokens,
        &req.model,
        params,
        session_id,
    )
    .await
    {
        Ok(resp) => {
            state
                .metrics
                .record_request("/v1/completions", started.elapsed().as_secs_f64());
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => {
            state
                .metrics
                .record_request("/v1/completions", started.elapsed().as_secs_f64());
            error_response(e)
        }
    }
}

// ---------------------- /v1/chat/completions -------------------------

#[derive(Deserialize, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Debug)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Serialize, Debug)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatResponseMessage,
    pub finish_reason: &'static str,
}

#[derive(Serialize, Debug)]
pub struct ChatResponseMessage {
    pub role: &'static str,
    pub content: String,
}

#[derive(Serialize, Debug)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: UsageStats,
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let started = Instant::now();
    let prompt = flatten_messages(&req.messages);
    let params = resolve_params(&state, req.temperature, req.top_p, req.top_k, req.seed);
    let session_id = req.session_id.clone();
    if req.stream.unwrap_or(false) {
        if state.gpu_native_token_loop.is_some() {
            state
                .metrics
                .record_request("/v1/chat/completions", started.elapsed().as_secs_f64());
            return error_response(GenerateError::InvalidRequest(
                "streaming is unsupported in gpu_native mode in this slice".into(),
            ));
        }
        match build_chat_stream(
            &state,
            &prompt,
            req.max_tokens,
            req.model.clone(),
            params,
            session_id,
        )
        .await
        {
            Ok(s) => {
                state
                    .metrics
                    .record_request("/v1/chat/completions", started.elapsed().as_secs_f64());
                return Sse::new(s).keep_alive(KeepAlive::default()).into_response();
            }
            Err(e) => {
                state
                    .metrics
                    .record_request("/v1/chat/completions", started.elapsed().as_secs_f64());
                return error_response(e);
            }
        }
    }
    // Flatten messages into a single prompt — exactly the same shape
    // simple OpenAI-compatible servers (vLLM, llama.cpp's HTTP) do when
    // no chat template is configured.
    match generate(
        &state,
        &prompt,
        req.max_tokens,
        &req.model,
        params,
        session_id,
    )
    .await
    {
        Ok(comp) => {
            let resp = ChatCompletionResponse {
                id: comp.id,
                object: "chat.completion",
                model: comp.model,
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatResponseMessage {
                        role: "assistant",
                        content: comp
                            .choices
                            .into_iter()
                            .next()
                            .map(|c| c.text)
                            .unwrap_or_default(),
                    },
                    finish_reason: "length",
                }],
                usage: comp.usage,
            };
            state
                .metrics
                .record_request("/v1/chat/completions", started.elapsed().as_secs_f64());
            (StatusCode::OK, Json(resp)).into_response()
        }
        Err(e) => {
            state
                .metrics
                .record_request("/v1/chat/completions", started.elapsed().as_secs_f64());
            error_response(e)
        }
    }
}

fn flatten_messages(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(&m.role);
        out.push_str(": ");
        out.push_str(&m.content);
        out.push('\n');
    }
    out
}

// ------------------------ generation core ----------------------------

#[derive(Debug)]
pub enum GenerateError {
    Tokenizer(String),
    InvalidRequest(String),
    /// A routed expert could not be read from storage even after
    /// retries (legacy benchmark path). Maps to a 500 — the request
    /// fails, the process survives.
    ExpertRead(String),
    /// Real-model inference failed closed (hardening pass, Part A):
    /// expert load/compute failure, non-finite attention, or invalid
    /// token id. Maps to a 500.
    Inference(String),
    /// The exact scheduler-registered KV state for an in-progress
    /// sequence could not be recovered (hardening pass, Part A7).
    /// Distinct from a model-compute failure: the model math was fine,
    /// but continuing (or persisting) the sequence would silently reset
    /// its context. Maps to a 500 with its own error type.
    KvStateLost(String),
}

pub(crate) fn map_gpu_native_token_loop_error(
    err: crate::gpu_native_token_loop::GpuNativeTokenLoopError,
) -> GenerateError {
    use crate::gpu_native_token_loop::GpuNativeTokenLoopError;
    match err {
        GpuNativeTokenLoopError::ContextLimitExceeded { .. }
        | GpuNativeTokenLoopError::UnsupportedSampling { .. } => {
            GenerateError::InvalidRequest(err.to_string())
        }
        GpuNativeTokenLoopError::PositionMismatch { .. }
        | GpuNativeTokenLoopError::IncompatibleModel(_)
        | GpuNativeTokenLoopError::AttemptBoundExceeded { .. }
        | GpuNativeTokenLoopError::NoProgress { .. }
        | GpuNativeTokenLoopError::FatalNumericalFailure { .. }
        | GpuNativeTokenLoopError::UnknownStatusBits { .. }
        | GpuNativeTokenLoopError::ResidencyServiceFailed(_)
        | GpuNativeTokenLoopError::Bootstrap(_)
        | GpuNativeTokenLoopError::InvalidBoundaryReport { .. }
        | GpuNativeTokenLoopError::InvalidSelectedExpertId { .. }
        | GpuNativeTokenLoopError::DuplicateSelectedExpertId { .. }
        | GpuNativeTokenLoopError::InvalidTopKCount { .. }
        | GpuNativeTokenLoopError::PrefetchShadowObserverFailed(_)
        | GpuNativeTokenLoopError::MapFailed(_) => GenerateError::Inference(err.to_string()),
    }
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerateError::Tokenizer(m) => write!(f, "tokenizer error: {m}"),
            GenerateError::InvalidRequest(m) => write!(f, "invalid request: {m}"),
            GenerateError::ExpertRead(m) => write!(f, "expert read error: {m}"),
            GenerateError::Inference(m) => write!(f, "inference error: {m}"),
            GenerateError::KvStateLost(m) => write!(f, "kv state lost: {m}"),
        }
    }
}

struct StageTimingPublisher {
    metrics: Metrics,
    timings: Arc<crate::stage_timing::StageTimings>,
    context: &'static str,
    published: bool,
}

impl StageTimingPublisher {
    fn new(
        metrics: Metrics,
        timings: Arc<crate::stage_timing::StageTimings>,
        context: &'static str,
    ) -> Self {
        Self {
            metrics,
            timings,
            context,
            published: false,
        }
    }

    fn publish(&mut self) {
        if self.published {
            return;
        }
        let snapshot = self.timings.snapshot();
        self.metrics.record_stage_timings(&snapshot);
        log_stage_timing_summary(self.context, &snapshot);
        self.published = true;
    }
}

impl Drop for StageTimingPublisher {
    fn drop(&mut self) {
        self.publish();
    }
}

struct RequestStageTiming {
    timings: Option<Arc<crate::stage_timing::StageTimings>>,
    publisher: Option<StageTimingPublisher>,
}

impl RequestStageTiming {
    fn new(metrics: &Metrics, enabled: bool, context: &'static str) -> Self {
        if enabled {
            let timings = Arc::new(crate::stage_timing::StageTimings::default());
            let publisher = StageTimingPublisher::new(metrics.clone(), timings.clone(), context);
            Self {
                timings: Some(timings),
                publisher: Some(publisher),
            }
        } else {
            Self {
                timings: None,
                publisher: None,
            }
        }
    }

    fn start(&self) -> Option<Instant> {
        Self::start_optional(self.timings.as_deref())
    }

    fn start_optional(timings: Option<&crate::stage_timing::StageTimings>) -> Option<Instant> {
        timings.map(|_| Instant::now())
    }

    fn record_since(&self, stage: &'static str, started: Option<Instant>) {
        Self::record_optional_since(self.timings.as_deref(), stage, started);
    }

    fn record_optional_since(
        timings: Option<&crate::stage_timing::StageTimings>,
        stage: &'static str,
        started: Option<Instant>,
    ) {
        if let (Some(timings), Some(started)) = (timings, started) {
            timings.record(stage, started.elapsed());
        }
    }

    fn publish(&mut self) {
        if let Some(publisher) = self.publisher.as_mut() {
            publisher.publish();
        }
    }
}

fn log_stage_timing_summary(
    context: &'static str,
    snapshot: &std::collections::BTreeMap<String, crate::stage_timing::StageTimingSnapshot>,
) {
    if snapshot.is_empty() {
        return;
    }
    let mut stages = Vec::with_capacity(snapshot.len());
    for (stage, timing) in snapshot {
        stages.push(format!(
            "{}={:.6}s/{}",
            stage, timing.total_seconds, timing.count
        ));
    }
    debug!(
        context,
        stage_timings = %stages.join(", "),
        "real request stage timing summary"
    );
}

/// Resolve per-request sampling parameters from the request fields,
/// falling back to the server-wide defaults pulled from the
/// atomically-swappable [`crate::config::RuntimeConfig`]. OpenAI's API
/// treats `temperature: 0` as "deterministic" — we mirror that by
/// passing it through verbatim (the sampler in [`crate::sampling`]
/// degrades to greedy `argmax`).
fn resolve_params(
    state: &AppState,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
) -> SamplingParams {
    // One relaxed atomic load — no lock, no contention with concurrent
    // SIGHUP reloads. The guard drops at the end of this function and
    // never crosses an `await` point.
    let mut p = state.runtime.snapshot().sampling;
    if let Some(t) = temperature {
        p.temperature = t;
    }
    if let Some(t) = top_p {
        p.top_p = t;
    }
    if let Some(t) = top_k {
        p.top_k = t;
    }
    if let Some(t) = seed {
        p.seed = t;
    }
    p
}

async fn generate(
    state: &AppState,
    prompt: &str,
    requested_max: usize,
    model_name: &str,
    params: SamplingParams,
    session_id: Option<String>,
) -> Result<CompletionResponse, GenerateError> {
    if prompt.is_empty() {
        return Err(GenerateError::InvalidRequest(
            "prompt must be non-empty".into(),
        ));
    }
    let max_tokens = requested_max
        .min(state.runtime.snapshot().max_tokens_cap)
        .max(1);

    // 1) Tokenize the prompt. The token ids drive the engine's deterministic
    //    routing seed so completions are reproducible for a given prompt.
    let prompt_ids = state
        .tokenizer
        .encode(prompt)
        .map_err(|e| GenerateError::Tokenizer(e.to_string()))?;
    let prompt_tokens = prompt_ids.len();
    if prompt_ids.is_empty() {
        return Err(GenerateError::InvalidRequest(
            "prompt must encode to at least one token".into(),
        ));
    }

    // 2) Drive next-token generation. Two paths:
    //
    //   * **Real-transformer path** (`real_model` is `Some`): for each
    //     output token, run the full decoder forward — embedding →
    //     stacked layers (each calling the engine's `moe_step` to load
    //     and run experts from SSD) → final RMSNorm → LM head — and
    //     sample the next token id from real logits via `params`.
    //     This is the gist's "actual generated text" path.
    //
    //   * **Legacy benchmark path** (`real_model` is `None`): drive
    //     `Engine::generate` for `max_tokens` cycles and synthesise a
    //     deterministic id stream. Same SSD-streaming behaviour, no
    //     real logits.
    //
    //   Both paths populate the same hits / misses / I/O counters, so
    //   the run summary stats look identical.
    let mut hits_total = 0u64;
    let mut misses_total = 0u64;
    let mut completion_ids: Vec<u32> = Vec::with_capacity(max_tokens);

    if let Some(ref gpu_native_loop) = state.gpu_native_token_loop {
        if session_id.is_some() {
            return Err(GenerateError::InvalidRequest(
                "session persistence is unsupported in gpu_native mode in this slice".into(),
            ));
        }
        if !params.is_greedy() {
            return Err(GenerateError::InvalidRequest(
                "only greedy sampling (temperature=0.0) is supported in gpu_native mode in this slice".into(),
            ));
        }
        let mut req_state = gpu_native_loop
            .create_request_state()
            .map_err(map_gpu_native_token_loop_error)?;
        let pre_hits = state.engine.report().hits;
        let pre_misses = state.engine.report().misses;
        completion_ids = gpu_native_loop
            .ingest_prompt_and_generate(
                &state.engine,
                &mut req_state,
                &prompt_ids,
                max_tokens,
                &params,
            )
            .await
            .map_err(map_gpu_native_token_loop_error)?;
        hits_total = state.engine.report().hits.saturating_sub(pre_hits);
        misses_total = state.engine.report().misses.saturating_sub(pre_misses);
    } else if let Some(model) = state.real_model.as_ref() {
        // Resolve session: take any existing KV state, then put it
        // back at the end. When no session is configured the request
        // is fully stateless (legacy behaviour).
        let (kv, mut start_pos, pending_token, checkout) =
            load_session_kv(state, model, session_id.as_deref());
        let timing_enabled = state.runtime.snapshot().stage_timing_enabled;
        let mut stage_timing =
            RequestStageTiming::new(&state.metrics, timing_enabled, "completion");
        let mut request = RealRequestState::new(state, model, kv, stage_timing.timings.clone());
        let pre_hits = state.engine.report().hits;
        let pre_misses = state.engine.report().misses;

        let prompt_started = stage_timing.start();
        // If the previous request ended after emitting a token that has not
        // yet been ingested into KV, ingest it first without evaluating the
        // LM head. This preserves P + C - 1 forward evaluations for the
        // request that produced it.
        if let Some(tid) = pending_token {
            request.forward_token_hidden(tid, start_pos).await?;
            start_pos += 1;
        }

        // Prime the KV cache with the prompt tokens so the first generated
        // token attends over the prompt, but do not run the LM head for
        // prompt tokens except the final hidden state that seeds completion
        // token 0.
        for &tid in &prompt_ids[..prompt_ids.len().saturating_sub(1)] {
            request.forward_token_hidden(tid, start_pos).await?;
            start_pos += 1;
        }
        let final_prompt_pos = start_pos;
        let final_prompt = *prompt_ids.last().expect("prompt_ids checked non-empty");
        let final_hidden = request
            .forward_token_hidden(final_prompt, final_prompt_pos)
            .await?;
        start_pos += 1;
        stage_timing.record_since(crate::stage_timing::TOTAL_PROMPT, prompt_started);
        let first = request.sample_hidden(&final_hidden, &params, final_prompt_pos);
        completion_ids.push(first);

        // Now generate the remaining real tokens.
        // When a draft engine is configured and speculation_k > 1, use
        // `step_speculative` which verifies k draft tokens per call with a
        // single batched SSD prefetch (gist Phase 2). Falls back to the
        // sequential scheduler/direct path otherwise.
        //
        // **Sampling-correctness gate.** `step_speculative` verifies and
        // emits tokens with `SamplingParams::greedy()` internally (see
        // `draft.rs::step_speculative`), so taking the speculative branch
        // for a request with `temperature > 0`, `top_p`, `top_k`, or a
        // non-default `seed` would silently downgrade it to greedy
        // decoding. Restrict the fast path to greedy requests; everything
        // else falls through to the sequential sampler that honours
        // `params` end-to-end.
        let mut last = first;
        let k = state.speculation_k;
        let decode_started = stage_timing.start();
        if k > 1 && state.draft_engine.is_some() && params.is_greedy() {
            let draft = state
                .draft_engine
                .as_ref()
                .expect("draft_engine is_some checked above")
                .clone();
            while completion_ids.len() < max_tokens {
                let remaining = max_tokens - completion_ids.len();
                let step_k = k.min(remaining);
                let kv = request.direct_kv()?;
                let result = model
                    .step_speculative(&state.engine, draft.as_ref(), last, start_pos, kv, step_k)
                    .await
                    .map_err(|e| GenerateError::Inference(e.to_string()))?;
                for &tok in &result.accepted {
                    if completion_ids.len() >= max_tokens {
                        break;
                    }
                    completion_ids.push(tok);
                    start_pos += 1;
                }
                last = *result.accepted.last().unwrap_or(&last);
            }
        } else {
            while completion_ids.len() < max_tokens {
                let next = request.decode_step(last, start_pos, &params).await?;
                completion_ids.push(next);
                last = next;
                start_pos += 1;
            }
        }
        stage_timing.record_since(crate::stage_timing::TOTAL_DECODE, decode_started);
        let pending_token = completion_ids.last().copied();
        // Part A7: KV loss at completion fails the request — never
        // return success for a sequence whose exact state was lost
        // (session persistence must not silently skip the error).
        let kv = request.finish()?;
        save_session_kv(
            state,
            session_id.as_deref(),
            kv,
            start_pos,
            pending_token,
            checkout,
        );
        let post = state.engine.report();
        hits_total = post.hits.saturating_sub(pre_hits);
        misses_total = post.misses.saturating_sub(pre_misses);
        stage_timing.publish();
    } else {
        // Legacy benchmark path — sampling params are unused (no real
        // logits) but we still respect `seed` for the deterministic
        // synthetic token stream so requests are reproducible.
        let base = prompt_ids
            .last()
            .copied()
            .unwrap_or(0)
            .wrapping_add(params.seed as u32) as u64;
        for i in 0..max_tokens {
            let stats = state
                .engine
                .generate(base.wrapping_add(i as u64))
                .await
                .map_err(|e| GenerateError::ExpertRead(e.to_string()))?;
            hits_total += stats.hits;
            misses_total += stats.misses;
            // Map engine cycle stats to a deterministic next-token id.
            let vocab = state.tokenizer.vocab_size().max(1) as u64;
            let next =
                ((base.wrapping_add(i as u64).wrapping_mul(0x9E3779B97F4A7C15)) % vocab) as u32;
            completion_ids.push(next);
        }
    }
    // Record the *actual* number of generated completion tokens, not the
    // requested `max_tokens` cap. The generation loops above stop at
    // `max_tokens`, but a future early-EOS stop, a cancellation, or an
    // error-after-partial path would leave `completion_ids` shorter than the
    // cap; the metric and usage accounting must follow the produced tokens
    // (gist Finding 10). Errors before this point propagate via `?` and are
    // never counted.
    let generated_tokens = completion_ids.len();
    state.metrics.record_tokens(generated_tokens as u64);
    state.metrics.record_cache(hits_total, misses_total);

    // 3) Decode and respond.
    let text = state
        .tokenizer
        .decode(&completion_ids)
        .map_err(|e| GenerateError::Tokenizer(e.to_string()))?;
    info!(
        prompt_tokens,
        completion_tokens = generated_tokens,
        cache_hits = hits_total,
        cache_misses = misses_total,
        "completed request"
    );
    Ok(CompletionResponse {
        id: format!("cmpl-{:x}", rand_request_id()),
        object: "text_completion",
        model: model_name.to_string(),
        choices: vec![CompletionChoice {
            text,
            index: 0,
            finish_reason: "length",
        }],
        usage: UsageStats {
            prompt_tokens,
            completion_tokens: generated_tokens,
            total_tokens: prompt_tokens + generated_tokens,
        },
    })
}

// ------------------------ scheduler helper --------------------------

/// Request-scoped real-model execution state. When a scheduler is present,
/// prompt prefill mutates local KV directly, then decode lazily registers the
/// KV cache once so scheduled decode operations carry only the small
/// [`RequestId`]. If the scheduler shuts down, the handle releases the registry
/// entry and falls back to direct model calls.
struct RealRequestState {
    engine: Arc<Engine>,
    model: Arc<RealModel>,
    scheduler: Option<Arc<BatchScheduler>>,
    request_id: Option<RequestId>,
    kv: Option<Vec<crate::transformer::KvCache>>,
    stage_timings: Option<Arc<crate::stage_timing::StageTimings>>,
}

impl RealRequestState {
    fn new(
        state: &AppState,
        model: &Arc<RealModel>,
        kv: Vec<crate::transformer::KvCache>,
        stage_timings: Option<Arc<crate::stage_timing::StageTimings>>,
    ) -> Self {
        Self {
            engine: state.engine.clone(),
            model: model.clone(),
            scheduler: state.batch_scheduler.clone(),
            request_id: None,
            kv: Some(kv),
            stage_timings,
        }
    }

    async fn forward_token_hidden(
        &mut self,
        token_id: u32,
        pos: usize,
    ) -> Result<Vec<f32>, GenerateError> {
        if let (Some(scheduler), Some(id)) = (self.scheduler.as_ref(), self.request_id) {
            let started = RequestStageTiming::start_optional(self.stage_timings.as_deref());
            let result = scheduler.forward_registered(id, token_id, pos).await;
            RequestStageTiming::record_optional_since(
                self.stage_timings.as_deref(),
                crate::stage_timing::SCHEDULER_OVERHEAD,
                started,
            );
            match result {
                Ok(hidden) => return Ok(hidden),
                // Fail-closed (Part A1): a deterministic model/storage
                // failure must surface — retrying it on the direct
                // path would recompute the same failure against a KV
                // cache the scheduler step may already have advanced.
                Err(e @ BatchError::Inference(_)) => {
                    return Err(GenerateError::Inference(e.to_string()));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "batch scheduler forward failed; falling back to direct model path");
                }
            }
        }
        let engine = self.engine.clone();
        let model = self.model.clone();
        let stage_timings = self.stage_timings.clone();
        let kv = self.direct_kv()?;
        model
            .forward_token_hidden_with_timing(&engine, token_id, pos, kv, stage_timings.as_deref())
            .await
            .map_err(|e| GenerateError::Inference(e.to_string()))
    }

    fn ensure_scheduler_registered(&mut self) -> Option<(Arc<BatchScheduler>, RequestId)> {
        let scheduler = self.scheduler.as_ref()?.clone();
        if self.request_id.is_none() {
            let kv = self.kv.take()?;
            let started = RequestStageTiming::start_optional(self.stage_timings.as_deref());
            let id = scheduler.register(kv);
            RequestStageTiming::record_optional_since(
                self.stage_timings.as_deref(),
                crate::stage_timing::SCHEDULER_OVERHEAD,
                started,
            );
            self.request_id = Some(id);
        }
        self.request_id.map(|id| (scheduler, id))
    }

    fn sample_hidden(
        &self,
        hidden: &[f32],
        params: &crate::sampling::SamplingParams,
        pos: usize,
    ) -> u32 {
        self.model
            .sample_hidden_with_timing(hidden, params, pos, self.stage_timings.as_deref())
    }

    async fn decode_step(
        &mut self,
        token_id: u32,
        pos: usize,
        params: &crate::sampling::SamplingParams,
    ) -> Result<u32, GenerateError> {
        if let Some((scheduler, id)) = self.ensure_scheduler_registered() {
            let started = RequestStageTiming::start_optional(self.stage_timings.as_deref());
            let result = scheduler.step_registered(id, token_id, pos, *params).await;
            RequestStageTiming::record_optional_since(
                self.stage_timings.as_deref(),
                crate::stage_timing::SCHEDULER_OVERHEAD,
                started,
            );
            match result {
                Ok(next) => return Ok(next),
                // Fail-closed (Part A1): deterministic inference
                // failures surface instead of retrying directly.
                Err(e @ BatchError::Inference(_)) => {
                    return Err(GenerateError::Inference(e.to_string()));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "batch scheduler decode failed; falling back to direct model path");
                }
            }
        }
        let engine = self.engine.clone();
        let model = self.model.clone();
        let stage_timings = self.stage_timings.clone();
        let kv = self.direct_kv()?;
        model
            .decode_step_with_timing(&engine, token_id, pos, kv, params, stage_timings.as_deref())
            .await
            .map_err(|e| GenerateError::Inference(e.to_string()))
    }

    /// Borrow the request's local KV caches, recovering them from the
    /// scheduler first if the request is registered. **Never fabricates
    /// fresh KV state** (hardening pass, Part A7): once a sequence has
    /// KV history, continuing it on empty caches would silently reset
    /// the context and produce mathematically wrong output — if the
    /// exact state cannot be recovered the request fails with a typed
    /// [`GenerateError::KvStateLost`].
    fn direct_kv(&mut self) -> Result<&mut Vec<crate::transformer::KvCache>, GenerateError> {
        if self.kv.is_none() {
            self.release_scheduler_to_local()?;
        }
        self.kv.as_mut().ok_or_else(|| {
            GenerateError::KvStateLost(
                "request holds no KV state and none is registered with the scheduler".into(),
            )
        })
    }

    /// Recover the registered KV state from the scheduler into
    /// `self.kv`. Returns [`GenerateError::KvStateLost`] when the
    /// request was registered but the scheduler could not return the
    /// exact state (Part A7: the caller must fail, never continue on
    /// fresh KV). Succeeds when nothing was registered or the state was
    /// recovered.
    fn release_scheduler_to_local(&mut self) -> Result<(), GenerateError> {
        if let (Some(scheduler), Some(id)) = (self.scheduler.take(), self.request_id.take()) {
            let started = RequestStageTiming::start_optional(self.stage_timings.as_deref());
            self.kv = scheduler.release(id);
            RequestStageTiming::record_optional_since(
                self.stage_timings.as_deref(),
                crate::stage_timing::SCHEDULER_OVERHEAD,
                started,
            );
            if self.kv.is_none() {
                tracing::error!(
                    request_id = id.0,
                    "scheduler lost this request's KV state; failing the request instead of \
                     resetting context"
                );
                return Err(GenerateError::KvStateLost(format!(
                    "scheduler could not return the exact KV state for request {}; refusing to \
                     continue the sequence on a fresh KV cache",
                    id.0
                )));
            }
        }
        Ok(())
    }

    /// Take ownership of the request's KV state for session
    /// persistence. Returns [`GenerateError::KvStateLost`] when the
    /// state was lost (Part A7): callers must fail the request rather
    /// than return success with an unpersistable (silently reset)
    /// session.
    fn take_kv(&mut self) -> Result<Vec<crate::transformer::KvCache>, GenerateError> {
        self.release_scheduler_to_local()?;
        self.kv.take().ok_or_else(|| {
            GenerateError::KvStateLost("request holds no KV state to persist".into())
        })
    }

    fn finish(mut self) -> Result<Vec<crate::transformer::KvCache>, GenerateError> {
        self.take_kv()
    }
}

impl Drop for RealRequestState {
    fn drop(&mut self) {
        if let (Some(scheduler), Some(id)) = (self.scheduler.take(), self.request_id.take()) {
            let started = RequestStageTiming::start_optional(self.stage_timings.as_deref());
            let _ = scheduler.release(id);
            RequestStageTiming::record_optional_since(
                self.stage_timings.as_deref(),
                crate::stage_timing::SCHEDULER_OVERHEAD,
                started,
            );
        }
    }
}

/// Fetch the persisted KV state for a session, or build a fresh one.
/// Returns `(kv_caches, next_position)`. The next position is `0` for
/// fresh sessions and the saved cursor otherwise — multi-turn chats
/// then continue token absolute positions across requests, which is
/// what RoPE expects.
fn load_session_kv(
    state: &AppState,
    model: &Arc<RealModel>,
    session_id: Option<&str>,
) -> (
    Vec<crate::transformer::KvCache>,
    usize,
    Option<u32>,
    Option<SessionCheckout>,
) {
    if let (Some(id), Some(store)) = (session_id, state.sessions.as_ref()) {
        if let Some((prev, checkout)) = store.take(id) {
            // Validate shape: layer count must match the live model.
            // Mismatches happen if the server is restarted with a
            // different config; treat as a fresh session.
            if prev.kv.len() == model.config.num_layers {
                return (prev.kv, prev.position, prev.pending_token, Some(checkout));
            }
        }
    }
    (model.fresh_kv_caches(), 0, None, None)
}

/// Persist KV state back to the store at request completion. No-op
/// when no session is configured.
fn save_session_kv(
    state: &AppState,
    session_id: Option<&str>,
    kv: Vec<crate::transformer::KvCache>,
    position: usize,
    pending_token: Option<u32>,
    checkout: Option<SessionCheckout>,
) {
    if let (Some(id), Some(store)) = (session_id, state.sessions.as_ref()) {
        store.put(
            id.to_string(),
            SessionState {
                kv,
                position,
                pending_token,
                last_used: Instant::now(),
            },
            checkout,
        );
    }
}

#[derive(Serialize, Debug)]
struct SessionDeleteResponse {
    id: String,
    deleted: bool,
}

async fn delete_session(State(state): State<AppState>, AxumPath(id): AxumPath<String>) -> Response {
    match state.sessions.as_ref() {
        Some(store) => {
            let deleted = store.delete(&id);
            (StatusCode::OK, Json(SessionDeleteResponse { id, deleted })).into_response()
        }
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": "session store disabled (server.session_ttl_secs == 0)"
            })),
        )
            .into_response(),
    }
}

// ---------------------------- /v1/admin/* ----------------------------

#[derive(Serialize, Debug)]
struct ExpertHealth {
    /// `"ok"` when no expert reads have failed since startup,
    /// `"degraded"` when one or more failed (i.e. the engine has
    /// served at least one request with a missing top-K expert).
    status: &'static str,
    expert_read_failures: u64,
    cache_hits: u64,
    cache_misses: u64,
    cache_pinned: usize,
    cache_capacity: usize,
    in_flight_requests: usize,
    block_pool_free: Option<usize>,
    block_pool_capacity: Option<usize>,
    block_pool_overflow_in_use: Option<usize>,
    tokens_generated: u64,
    /// Whether the logical GPU-admission cache is configured. Compatibility
    /// `vram_*` fields remain present for stable schema but describe logical
    /// host admission bytes; exact physical bytes use the PR4 Rust snapshot.
    gpu_cache_enabled: bool,
    gpu_cache_hits: u64,
    gpu_cache_misses: u64,
    gpu_promotions_total: u64,
    /// Logical admitted host payload bytes (compatibility field name).
    vram_used_bytes: u64,
    /// Logical admission budget, also used as the physical expert-weight cap.
    vram_capacity_bytes: u64,
    gpu_anchor_count: usize,
    gpu_lru_count: usize,
}

/// Lightweight production health probe that surfaces the engine's
/// per-process counters in a format alerting systems can consume
/// without scraping Prometheus.
async fn admin_health_experts(State(state): State<AppState>) -> Response {
    let report = state.engine.report();
    let status = if report.expert_read_failures > 0 {
        "degraded"
    } else {
        "ok"
    };
    let (free, cap, overflow) = match state.batch_scheduler.as_ref().and_then(|s| s.block_pool()) {
        Some(p) => (
            Some(p.free_blocks()),
            Some(p.capacity()),
            Some(p.overflow_in_use()),
        ),
        None => (None, None, None),
    };
    let body = ExpertHealth {
        status,
        expert_read_failures: report.expert_read_failures,
        cache_hits: report.hits,
        cache_misses: report.misses,
        cache_pinned: report.pinned_count,
        cache_capacity: report.cache_capacity,
        in_flight_requests: state.middleware.admission.in_flight(),
        block_pool_free: free,
        block_pool_capacity: cap,
        block_pool_overflow_in_use: overflow,
        tokens_generated: report.tokens_processed,
        gpu_cache_enabled: report.gpu_cache_enabled,
        gpu_cache_hits: report.gpu_cache_hits,
        gpu_cache_misses: report.gpu_cache_misses,
        gpu_promotions_total: report.gpu_promotions,
        vram_used_bytes: report.vram_used_bytes,
        vram_capacity_bytes: report.vram_capacity_bytes,
        gpu_anchor_count: report.gpu_anchor_count,
        gpu_lru_count: report.gpu_lru_count,
    };
    // Surface degraded health via HTTP status so naive uptime probes
    // (HAProxy health checks, K8s liveness) light up without parsing
    // the body.
    let code = if status == "ok" {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (code, Json(body)).into_response()
}

#[derive(Serialize, Debug)]
struct EvictResponse {
    reclaimed_overflow_blocks: usize,
}

/// Admin endpoint: trigger a one-shot reclaim pass on the paged-KV
/// pool's overflow slab. Useful for operators after a transient
/// burst inflated the heap-backed fallback — calling this returns
/// the memory to the allocator immediately rather than waiting for
/// the periodic background sweep.
async fn admin_evict_overflow(State(state): State<AppState>) -> Response {
    let reclaimed = match state.batch_scheduler.as_ref().and_then(|s| s.block_pool()) {
        Some(p) => p.shrink_overflow_to_fit(),
        None => 0,
    };
    (
        StatusCode::OK,
        Json(EvictResponse {
            reclaimed_overflow_blocks: reclaimed,
        }),
    )
        .into_response()
}

// ----------------------- generation helpers --------------------------

/// One streamed completion chunk. Mirrors OpenAI's `text_completion` SSE
/// event shape so any OpenAI-compatible client can consume it.
#[derive(Serialize, Debug)]
struct CompletionChunk {
    id: String,
    object: &'static str,
    model: String,
    choices: Vec<CompletionChunkChoice>,
}
#[derive(Serialize, Debug)]
struct CompletionChunkChoice {
    text: String,
    index: u32,
    finish_reason: Option<&'static str>,
}

/// One streamed chat-completion chunk. Same shape OpenAI uses for
/// `chat.completion.chunk`: each event carries a *delta* with the
/// incremental content rather than the full message.
#[derive(Serialize, Debug)]
struct ChatChunk {
    id: String,
    object: &'static str,
    model: String,
    choices: Vec<ChatChunkChoice>,
}
#[derive(Serialize, Debug)]
struct ChatChunkChoice {
    index: u32,
    delta: ChatDelta,
    finish_reason: Option<&'static str>,
}
#[derive(Serialize, Debug, Default)]
struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

/// Per-token output yielded by the streaming generators below: the new
/// piece of decoded text since the previous token, plus the cache
/// hits/misses recorded by the engine for that step.
struct StreamChunk {
    text: String,
    finished: bool,
    hits: u64,
    misses: u64,
}

/// Drive token-by-token generation, yielding incremental decoded text
/// after each step. Both the real-transformer and legacy paths are
/// supported (mirroring `generate`). Returns a boxed Stream so the
/// SSE handler can wrap it.
///
/// **Failures are visible** (hardening pass, item 5): the stream item
/// is `Result<StreamChunk, GenerateError>` — a real-model inference,
/// tokenizer/decoder, or KV-state failure is yielded as an `Err` and
/// terminates the stream. It is never converted into a successful
/// `finished` chunk. The legacy synthetic path keeps its historic
/// end-of-stream behaviour.
async fn stream_tokens(
    state: AppState,
    prompt: String,
    requested_max: usize,
    params: SamplingParams,
    session_id: Option<String>,
) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, GenerateError>> + Send>>, GenerateError> {
    if prompt.is_empty() {
        return Err(GenerateError::InvalidRequest(
            "prompt must be non-empty".into(),
        ));
    }
    let max_tokens = requested_max
        .min(state.runtime.snapshot().max_tokens_cap)
        .max(1);
    let prompt_ids = state
        .tokenizer
        .encode(&prompt)
        .map_err(|e| GenerateError::Tokenizer(e.to_string()))?;
    if prompt_ids.is_empty() {
        return Err(GenerateError::InvalidRequest(
            "prompt must encode to at least one token".into(),
        ));
    }

    // Generator state passed through `stream::unfold`. Holds everything a
    // single-stream generator needs across `await` points.
    enum GenMode {
        Real {
            request: RealRequestState,
            last_token: u32,
            position: usize,
            checkout: Option<SessionCheckout>,
            pending_emit: Option<u32>,
        },
        Legacy {
            base: u64,
            i: u64,
        },
    }
    let timing_enabled = state.runtime.snapshot().stage_timing_enabled;
    let stage_timing = RequestStageTiming::new(&state.metrics, timing_enabled, "stream");
    let mode = if let Some(model) = state.real_model.as_ref() {
        // Resume from session state if configured — otherwise start
        // fresh, matching the non-streaming `generate` path.
        let (kv, mut pos, pending_token, checkout) =
            load_session_kv(&state, model, session_id.as_deref());
        let mut request = RealRequestState::new(&state, model, kv, stage_timing.timings.clone());
        let prompt_started = stage_timing.start();
        if let Some(tid) = pending_token {
            request.forward_token_hidden(tid, pos).await?;
            pos += 1;
        }
        for &tid in &prompt_ids[..prompt_ids.len().saturating_sub(1)] {
            request.forward_token_hidden(tid, pos).await?;
            pos += 1;
        }
        let final_prompt_pos = pos;
        let final_prompt = *prompt_ids.last().expect("prompt_ids checked non-empty");
        let hidden = request
            .forward_token_hidden(final_prompt, final_prompt_pos)
            .await?;
        pos += 1;
        stage_timing.record_since(crate::stage_timing::TOTAL_PROMPT, prompt_started);
        let first = request.sample_hidden(&hidden, &params, final_prompt_pos);
        GenMode::Real {
            request,
            last_token: first,
            position: pos,
            checkout,
            pending_emit: Some(first),
        }
    } else {
        GenMode::Legacy {
            base: prompt_ids
                .last()
                .copied()
                .unwrap_or(0)
                .wrapping_add(params.seed as u32) as u64,
            i: 0,
        }
    };

    // Incremental decoder (hardening pass, F2): each step decodes only
    // a bounded look-behind window instead of re-decoding the entire
    // cumulative completion and cloning the cumulative string.
    struct St {
        state: AppState,
        mode: GenMode,
        completion_ids: Vec<u32>,
        decoder: crate::tokenizer::StreamDecoder,
        emitted: usize,
        max_tokens: usize,
        finished_emitted: bool,
        params: SamplingParams,
        session_id: Option<String>,
        stage_timing: RequestStageTiming,
    }

    let st = St {
        state,
        mode,
        completion_ids: Vec::with_capacity(max_tokens),
        decoder: crate::tokenizer::StreamDecoder::new(),
        emitted: 0,
        max_tokens,
        finished_emitted: false,
        params,
        session_id,
        stage_timing,
    };

    Ok(Box::pin(stream::unfold(st, move |mut st| async move {
        if st.finished_emitted {
            return None;
        }
        if st.emitted >= st.max_tokens {
            st.finished_emitted = true;
            // Persist KV-cache state back to the session store so a
            // follow-up request can pick up where we stopped. Part A7:
            // KV loss is a visible stream error — never a normal
            // `finished` chunk after a silently-skipped persistence.
            if let GenMode::Real {
                request,
                position,
                checkout,
                ..
            } = &mut st.mode
            {
                let kv_take = match request.take_kv() {
                    Ok(kv) => kv,
                    Err(e) => {
                        tracing::error!(error = %e, "real stream: KV state lost at completion");
                        st.stage_timing.publish();
                        return Some((Err(e), st));
                    }
                };
                let pending_token = st.completion_ids.last().copied();
                save_session_kv(
                    &st.state,
                    st.session_id.as_deref(),
                    kv_take,
                    *position,
                    pending_token,
                    checkout.take(),
                );
            }
            st.stage_timing.publish();
            // Flush any held-back (incomplete-sequence) text from the
            // incremental decoder so the final chunk never drops
            // trailing output (F2). Decoder failures propagate instead
            // of silently replacing the tail with an empty string.
            let tail = match st.decoder.finish(&st.state.tokenizer) {
                Ok(tail) => tail,
                Err(e) => {
                    tracing::error!(error = %e, "real stream: decoder failed on final flush");
                    return Some((Err(GenerateError::Tokenizer(e.to_string())), st));
                }
            };
            return Some((
                Ok(StreamChunk {
                    text: tail,
                    finished: true,
                    hits: 0,
                    misses: 0,
                }),
                st,
            ));
        }
        let pre_hits = st.state.engine.report().hits;
        let pre_misses = st.state.engine.report().misses;
        let next: u32 = match &mut st.mode {
            GenMode::Real {
                request,
                last_token,
                position,
                pending_emit,
                ..
            } => {
                if let Some(n) = pending_emit.take() {
                    n
                } else {
                    let decode_started = st.stage_timing.start();
                    let n = match request
                        .decode_step(*last_token, *position, &st.params)
                        .await
                    {
                        Ok(n) => n,
                        Err(e) => {
                            // Fail-closed real inference (Part A1 /
                            // item 5): a deterministic model failure is
                            // a visible stream error — never converted
                            // into a successful `finished` chunk.
                            tracing::error!(error = %e, "real stream: decode failed; failing stream");
                            st.finished_emitted = true;
                            st.stage_timing.publish();
                            return Some((Err(e), st));
                        }
                    };
                    st.stage_timing
                        .record_since(crate::stage_timing::TOTAL_DECODE, decode_started);
                    *position += 1;
                    n
                }
            }
            GenMode::Legacy { base, i } => {
                match st.state.engine.generate(base.wrapping_add(*i)).await {
                    Ok(stats) => {
                        let _ = stats;
                    }
                    Err(e) => {
                        // Legacy synthetic benchmark path (kept
                        // separate from real inference): a routed
                        // expert could not be read even after retries.
                        // Terminate the stream cleanly instead of
                        // crashing the process: emit a final
                        // `finished` chunk so the client sees EOS.
                        tracing::warn!(error = %e, "legacy stream: expert read failed; ending stream");
                        st.finished_emitted = true;
                        return Some((
                            Ok(StreamChunk {
                                text: String::new(),
                                finished: true,
                                hits: 0,
                                misses: 0,
                            }),
                            st,
                        ));
                    }
                }
                let vocab = st.state.tokenizer.vocab_size().max(1) as u64;
                let id = ((base.wrapping_add(*i).wrapping_mul(0x9E3779B97F4A7C15)) % vocab) as u32;
                *i = i.wrapping_add(1);
                id
            }
        };
        if let GenMode::Real { last_token, .. } = &mut st.mode {
            *last_token = next;
        }
        let post = st.state.engine.report();
        let hits = post.hits.saturating_sub(pre_hits);
        let misses = post.misses.saturating_sub(pre_misses);

        st.completion_ids.push(next);
        st.emitted += 1;

        // Incremental decode (F2): decode only the bounded look-behind
        // window and emit the newly stable text. Decoder failures are
        // visible stream errors (item 6): silently emitting an empty
        // delta would lose generated text.
        let delta = match st.decoder.push(&st.state.tokenizer, next) {
            Ok(delta) => delta,
            Err(e) => {
                tracing::error!(error = %e, "stream decoder failed; failing stream");
                st.finished_emitted = true;
                st.stage_timing.publish();
                return Some((Err(GenerateError::Tokenizer(e.to_string())), st));
            }
        };

        Some((
            Ok(StreamChunk {
                text: delta,
                finished: false,
                hits,
                misses,
            }),
            st,
        ))
    })))
}

async fn build_completion_stream(
    state: &AppState,
    prompt: &str,
    requested_max: usize,
    model_name: String,
    params: SamplingParams,
    session_id: Option<String>,
) -> Result<impl Stream<Item = Result<Event, Infallible>>, GenerateError> {
    let id = format!("cmpl-{:x}", rand_request_id());
    let metrics = state.metrics.clone();
    let inner = stream_tokens(
        state.clone(),
        prompt.to_string(),
        requested_max,
        params,
        session_id,
    )
    .await?;
    Ok(wrap_completion_events(inner, id, model_name, metrics))
}

/// Wrap the inner token stream in OpenAI-compatible completion SSE
/// events. Factored out of [`build_completion_stream`] so the
/// event-ordering contract (item 6: a non-empty final decoder tail is
/// emitted as a normal content chunk *before* `[DONE]`; item 5: errors
/// terminate the stream with an error event and never `[DONE]`) is
/// directly testable against a synthetic inner stream.
fn wrap_completion_events(
    inner: Pin<Box<dyn Stream<Item = Result<StreamChunk, GenerateError>> + Send>>,
    id: String,
    model_name: String,
    metrics: Metrics,
) -> impl Stream<Item = Result<Event, Infallible>> {
    struct WrapSt {
        inner: Pin<Box<dyn Stream<Item = Result<StreamChunk, GenerateError>> + Send>>,
        id: String,
        model_name: String,
        metrics: Metrics,
        hits: u64,
        misses: u64,
        tokens_done: u64,
        /// A queued terminal `[DONE]` event, emitted on the next poll
        /// after a final decoder-tail content chunk (item 6: the tail
        /// must reach the client *before* the terminator).
        pending_done: bool,
        terminated: bool,
    }
    let st = WrapSt {
        inner,
        id,
        model_name,
        metrics,
        hits: 0,
        misses: 0,
        tokens_done: 0,
        pending_done: false,
        terminated: false,
    };
    let s = stream::unfold(st, |mut st| async move {
        if st.terminated {
            return None;
        }
        if st.pending_done {
            st.pending_done = false;
            st.terminated = true;
            st.metrics.record_tokens(st.tokens_done);
            st.metrics.record_cache(st.hits, st.misses);
            info!(
                tokens = st.tokens_done,
                cache_hits = st.hits,
                cache_misses = st.misses,
                "streamed completion finished"
            );
            return Some((Ok(Event::default().data("[DONE]")), st));
        }
        use futures::stream::StreamExt;
        match st.inner.next().await {
            None => {
                // Inner exhausted unexpectedly — emit DONE.
                st.metrics.record_tokens(st.tokens_done);
                st.metrics.record_cache(st.hits, st.misses);
                st.terminated = true;
                Some((Ok(Event::default().data("[DONE]")), st))
            }
            Some(Err(e)) => {
                // Item 5: a real inference / decoder / KV-state failure
                // is surfaced as an OpenAI-compatible error event and
                // the stream terminates — never a successful
                // `finish_reason` or `[DONE]`, and no "finished" log.
                // Token/cache metrics reflect only the successfully
                // emitted work.
                st.metrics.record_tokens(st.tokens_done);
                st.metrics.record_cache(st.hits, st.misses);
                tracing::error!(error = %e, "streamed completion failed");
                st.terminated = true;
                let ev = Event::default().data(stream_error_payload(&e));
                Some((Ok(ev), st))
            }
            Some(Ok(chunk)) => {
                st.hits += chunk.hits;
                st.misses += chunk.misses;
                if chunk.finished {
                    if chunk.text.is_empty() {
                        // End of stream with no held-back tail: emit
                        // `[DONE]` and terminate.
                        st.metrics.record_tokens(st.tokens_done);
                        st.metrics.record_cache(st.hits, st.misses);
                        info!(
                            tokens = st.tokens_done,
                            cache_hits = st.hits,
                            cache_misses = st.misses,
                            "streamed completion finished"
                        );
                        st.terminated = true;
                        Some((Ok(Event::default().data("[DONE]")), st))
                    } else {
                        // Item 6: the final decoder tail is real
                        // generated text — emit it as a normal content
                        // chunk first, then `[DONE]` on the next poll.
                        let payload = CompletionChunk {
                            id: st.id.clone(),
                            object: "text_completion",
                            model: st.model_name.clone(),
                            choices: vec![CompletionChunkChoice {
                                text: chunk.text,
                                index: 0,
                                finish_reason: None,
                            }],
                        };
                        let ev = Event::default()
                            .data(serde_json::to_string(&payload).unwrap_or_default());
                        st.pending_done = true;
                        Some((Ok(ev), st))
                    }
                } else {
                    st.tokens_done += 1;
                    let payload = CompletionChunk {
                        id: st.id.clone(),
                        object: "text_completion",
                        model: st.model_name.clone(),
                        choices: vec![CompletionChunkChoice {
                            text: chunk.text,
                            index: 0,
                            finish_reason: None,
                        }],
                    };
                    let ev =
                        Event::default().data(serde_json::to_string(&payload).unwrap_or_default());
                    Some((Ok(ev), st))
                }
            }
        }
    });
    s
}

/// Serialize an OpenAI-compatible SSE error payload for a failed
/// stream (item 5): the same `{ "error": { message, type } }` shape as
/// the non-streaming error body, emitted as the final event before the
/// stream terminates (no `[DONE]`, no successful `finish_reason`).
fn stream_error_payload(e: &GenerateError) -> String {
    let kind = match e {
        GenerateError::InvalidRequest(_) => "invalid_request_error",
        GenerateError::KvStateLost(_) => "kv_state_lost",
        _ => "server_error",
    };
    serde_json::to_string(&ErrorBody {
        error: ErrorBodyInner {
            message: e.to_string(),
            kind,
        },
    })
    .unwrap_or_else(|_| {
        "{\"error\":{\"message\":\"stream failed\",\"type\":\"server_error\"}}".into()
    })
}

async fn build_chat_stream(
    state: &AppState,
    prompt: &str,
    requested_max: usize,
    model_name: String,
    params: SamplingParams,
    session_id: Option<String>,
) -> Result<impl Stream<Item = Result<Event, Infallible>>, GenerateError> {
    let id = format!("chatcmpl-{:x}", rand_request_id());
    let metrics = state.metrics.clone();
    let inner = stream_tokens(
        state.clone(),
        prompt.to_string(),
        requested_max,
        params,
        session_id,
    )
    .await?;
    Ok(wrap_chat_events(inner, id, model_name, metrics))
}

/// Wrap the inner token stream in OpenAI-compatible chat SSE events
/// (leading `role: assistant` delta, then content deltas). Factored
/// out of [`build_chat_stream`] so the event-ordering contract (item
/// 6: a non-empty final decoder tail is emitted as a content delta
/// *before* `[DONE]`; item 5: errors terminate the stream with an
/// error event and never `[DONE]`) is directly testable against a
/// synthetic inner stream.
fn wrap_chat_events(
    inner: Pin<Box<dyn Stream<Item = Result<StreamChunk, GenerateError>> + Send>>,
    id: String,
    model_name: String,
    metrics: Metrics,
) -> impl Stream<Item = Result<Event, Infallible>> {
    // OpenAI emits a first "delta: { role: assistant }" event before any
    // content tokens. We do the same so streaming chat clients see the
    // role before the first content delta.
    let role_chunk = ChatChunk {
        id: id.clone(),
        object: "chat.completion.chunk",
        model: model_name.clone(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: Some("assistant"),
                content: None,
            },
            finish_reason: None,
        }],
    };
    let role_event = Event::default().data(serde_json::to_string(&role_chunk).unwrap_or_default());

    struct ChatWrapSt {
        role_ev: Option<Event>,
        inner: Pin<Box<dyn Stream<Item = Result<StreamChunk, GenerateError>> + Send>>,
        id: String,
        model_name: String,
        metrics: Metrics,
        hits: u64,
        misses: u64,
        tokens_done: u64,
        /// A queued terminal `[DONE]` event, emitted on the next poll
        /// after a final decoder-tail content delta (item 6).
        pending_done: bool,
        terminated: bool,
    }
    let st = ChatWrapSt {
        role_ev: Some(role_event),
        inner,
        id,
        model_name,
        metrics,
        hits: 0,
        misses: 0,
        tokens_done: 0,
        pending_done: false,
        terminated: false,
    };
    let s = stream::unfold(st, |mut st| async move {
        if let Some(ev) = st.role_ev.take() {
            // Emit the role event first.
            return Some((Ok(ev), st));
        }
        if st.terminated {
            return None;
        }
        if st.pending_done {
            st.pending_done = false;
            st.terminated = true;
            st.metrics.record_tokens(st.tokens_done);
            st.metrics.record_cache(st.hits, st.misses);
            info!(
                tokens = st.tokens_done,
                cache_hits = st.hits,
                cache_misses = st.misses,
                "streamed chat completion finished"
            );
            return Some((Ok(Event::default().data("[DONE]")), st));
        }
        use futures::stream::StreamExt;
        match st.inner.next().await {
            None => {
                st.metrics.record_tokens(st.tokens_done);
                st.metrics.record_cache(st.hits, st.misses);
                st.terminated = true;
                Some((Ok(Event::default().data("[DONE]")), st))
            }
            Some(Err(e)) => {
                // Item 5: failures are surfaced as an OpenAI-compatible
                // error event and the stream terminates — never a
                // successful `finish_reason`, `[DONE]`, or "finished"
                // log. Metrics reflect only successfully emitted work.
                st.metrics.record_tokens(st.tokens_done);
                st.metrics.record_cache(st.hits, st.misses);
                tracing::error!(error = %e, "streamed chat completion failed");
                st.terminated = true;
                let ev = Event::default().data(stream_error_payload(&e));
                Some((Ok(ev), st))
            }
            Some(Ok(chunk)) => {
                st.hits += chunk.hits;
                st.misses += chunk.misses;
                if chunk.finished && chunk.text.is_empty() {
                    // End of stream with no held-back tail.
                    st.metrics.record_tokens(st.tokens_done);
                    st.metrics.record_cache(st.hits, st.misses);
                    info!(
                        tokens = st.tokens_done,
                        cache_hits = st.hits,
                        cache_misses = st.misses,
                        "streamed chat completion finished"
                    );
                    st.terminated = true;
                    Some((Ok(Event::default().data("[DONE]")), st))
                } else {
                    if chunk.finished {
                        // Item 6: emit the final decoder tail as a
                        // normal content delta, then `[DONE]` next poll.
                        st.pending_done = true;
                    } else {
                        st.tokens_done += 1;
                    }
                    let payload = ChatChunk {
                        id: st.id.clone(),
                        object: "chat.completion.chunk",
                        model: st.model_name.clone(),
                        choices: vec![ChatChunkChoice {
                            index: 0,
                            delta: ChatDelta {
                                role: None,
                                content: Some(chunk.text),
                            },
                            finish_reason: None,
                        }],
                    };
                    let ev =
                        Event::default().data(serde_json::to_string(&payload).unwrap_or_default());
                    Some((Ok(ev), st))
                }
            }
        }
    });
    s
}

/// 64-bit pseudo-random id derived from the wall clock and a per-call
/// counter. Good enough for a request id; not used for security.
fn rand_request_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    // 2^64 / phi — the standard "golden ratio" odd multiplier used by
    // Knuth's multiplicative hash (also used in Java's HashMap and
    // Linux's hash_64). Mixes the per-call counter into the high bits
    // before XORing with the wall clock so two requests issued in the
    // same nanosecond still get distinct ids.
    const GOLDEN_RATIO_U64: u64 = 0x9E3779B97F4A7C15;
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ n.wrapping_mul(GOLDEN_RATIO_U64)
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorBodyInner,
}
#[derive(Serialize)]
struct ErrorBodyInner {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

fn error_response(e: GenerateError) -> Response {
    let (status, kind) = match &e {
        GenerateError::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
        GenerateError::Tokenizer(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
        GenerateError::ExpertRead(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
        GenerateError::Inference(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
        // Distinguishable from a model-compute failure (Part A7): the
        // request's exact KV state could not be recovered, so the
        // sequence cannot be continued or persisted.
        GenerateError::KvStateLost(_) => (StatusCode::INTERNAL_SERVER_ERROR, "kv_state_lost"),
    };
    (
        status,
        Json(ErrorBody {
            error: ErrorBodyInner {
                message: e.to_string(),
                kind,
            },
        }),
    )
        .into_response()
}

// ------------------------------ run ---------------------------------

/// **Cold-start mitigation orchestrator (gist Task 1).**
///
/// Runs a synthetic pass through the full token pipeline so the
/// very first user request does not pay the one-time costs of:
///
/// * faulting the `AlignedBuffer` slab pool into the resident set
///   (a `BufferPool` slot is `posix_memalign`'d up front but the
///   pages are only zero-filled on first touch),
/// * registering `io_uring` fixed buffers / fixed files with the
///   kernel (`IORING_REGISTER_BUFFERS` is amortised across every
///   subsequent read, but the first read pays the syscall),
/// * JIT-warming the math backend kernels (AVX-512 dispatch
///   tables, candle Tensor allocators, future GPU command queues).
///
/// Implementation strategy:
///
/// 1. If a real-transformer pipeline + batch scheduler is wired in,
///    invoke [`BatchScheduler::submit_internal_warmup`] which drives
///    one synthetic decoder step through the existing scheduler
///    plumbing — bypassing every network / auth / rate-limit /
///    admission layer that gates real HTTP requests.
/// 2. Otherwise, fall back to a direct `engine.warm_with(0..8)` +
///    a tiny synthetic backend `swiglu_into` call. This still
///    primes the slab pool, the io_uring fixed-buffer table, and
///    the math kernel dispatch — just without the
///    `mpsc → scheduler_loop` exercise.
///
/// **Safety contract.** If any warm-up step fails the server must
/// still bind. We log the failure via [`tracing::warn!`] and
/// continue startup — a warm-up failure is *not* a fatal startup
/// error (gist Task 1, "Constraint: The server must still bind if
/// warm-up fails").
///
/// **Performance contract.** This routine runs **before** the
/// listener binds in [`serve`], so the listening socket only
/// accepts connections once warm-up has completed. The first
/// real user therefore sees a 0-ms-start system.
pub async fn run_engine_warmup(state: &AppState) {
    const SCHEDULER_WARMUP_TIMEOUT: Duration = Duration::from_secs(5);
    let start = Instant::now();
    tracing::info!("engine warm-up starting (cold-start mitigation)");

    let mut result: Result<&'static str, String> = Ok("noop");

    if let (Some(model), Some(scheduler)) =
        (state.real_model.as_ref(), state.batch_scheduler.as_ref())
    {
        // Full path: drive a synthetic decoder step through the
        // batch scheduler. Bypasses HTTP/auth/admission by design.
        match tokio::time::timeout(
            SCHEDULER_WARMUP_TIMEOUT,
            scheduler.submit_internal_warmup(model, &state.engine),
        )
        .await
        {
            Ok(Ok(())) => result = Ok("scheduler-synthetic-step"),
            Ok(Err(e)) => result = Err(format!("scheduler warm-up failed: {e}")),
            Err(_) => {
                result = Err(format!(
                    "scheduler warm-up timed out after {}s",
                    SCHEDULER_WARMUP_TIMEOUT.as_secs()
                ))
            }
        }
    } else {
        // Fallback path (no real-transformer / no scheduler wired in
        // — e.g. legacy benchmark generator). Still primes the slab
        // pool + io_uring + math backend so even the legacy path
        // sees a 0-ms start.
        let num_experts = state.engine.num_experts();
        if num_experts > 0 {
            let cap = num_experts.min(8);
            let ids: Vec<u32> = (0..cap).collect();
            if let Err(e) = state.engine.warm_with(&ids).await {
                result = Err(format!("engine.warm_with failed: {e}"));
            } else {
                result = Ok("engine-warm_with");
            }
        }
        // JIT-warm the math backend kernels regardless.
        let backend = state.engine.execution_context().routed_expert_backend();
        let gate_f16 = vec![half::f16::from_f32(0.1); 16];
        let up_f16 = vec![half::f16::from_f32(0.2); 16];
        let mut out_f16 = vec![half::f16::ZERO; 16];
        let gate_view = crate::backend::TensorView {
            data: &gate_f16,
            rows: 4,
            cols: 4,
        };
        let up_view = crate::backend::TensorView {
            data: &up_f16,
            rows: 4,
            cols: 4,
        };
        let mut out_view = crate::backend::TensorViewMut {
            data: &mut out_f16,
            rows: 4,
            cols: 4,
        };
        let _ = backend.swiglu_into(gate_view, up_view, &mut out_view);
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok(path) => tracing::info!(
            elapsed_ms = format!("{elapsed_ms:.2}"),
            path,
            "engine warm-up complete — first request sees a primed slab pool, io_uring \
             registrations, and JIT-warmed math kernels"
        ),
        Err(reason) => tracing::warn!(
            elapsed_ms = format!("{elapsed_ms:.2}"),
            reason,
            "engine warm-up failed; continuing to bind listener (cold-start cost will be \
             paid on the first user request)"
        ),
    }
}

/// Bind the server, listen, and run until the runtime is shut down.
pub async fn serve(state: AppState, bind: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Speculative engine warm-up — runs *before* the listener
    // binds so the first user request sees a 0-ms start. Failures
    // here are logged via `tracing::warn!` and the bind still
    // proceeds (gist Task 1 safety constraint).
    run_engine_warmup(&state).await;

    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    info!(bind, "HTTP server listening");
    // Graceful shutdown on SIGTERM / Ctrl-C.
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
        info!("shutdown signal received; draining...");
    };
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;
    use crate::engine::{Engine, EngineOptions, ModelShape};
    use crate::io_provider::{generate_synthetic_experts, NvmeStorage, StorageConfig};
    use crate::multi_layer_cache::MultiLayerExpertCache;
    use crate::router::{PredictiveLoader, TopKRouter};
    use axum::body::{to_bytes, Body};
    use axum::http::Request;
    use futures::StreamExt;
    use std::sync::Arc;
    use tempdir::TempDir;
    use tower::ServiceExt;

    /// Minimal valid [`crate::config::Config`] for the server tests. The
    /// per-test `make_state` overrides `server.max_tokens` to vary the
    /// runtime cap exercised by [`generate`].
    fn test_minimal_cfg() -> crate::config::Config {
        use std::path::PathBuf;
        crate::config::Config {
            server: crate::config::ServerConfig {
                bind: "127.0.0.1:0".into(),
                max_tokens: 32,
                session_ttl_secs: 0,
                max_concurrent_requests: 0,
                admission_min_free_blocks: 0,
            },
            performance: crate::config::PerformanceConfig::default(),
            model: crate::config::ModelConfig {
                data_dir: PathBuf::from("./data"),
                num_experts: 8,
                top_k: 2,
                d_model: 8,
                d_ff: 16,
                expert_size: 4096,
                num_layers: 1,
                dtype: crate::inference::WeightDtype::F32,
            },
            storage: crate::config::StorageConfigToml {
                cache_slots: 4,
                block_align: 4096,
                no_direct: true,
                predict_fanout: 2,
                pipeline_depth: crate::engine::DEFAULT_PIPELINE_DEPTH,
                predict_min_prob: 0.05,
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

    // We need a tempdir helper but don't want to add `tempfile` for one
    // test; reuse `std::env::temp_dir()` with a unique subpath.
    mod tempdir {
        use std::path::PathBuf;
        use std::sync::atomic::{AtomicU64, Ordering};

        static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

        pub struct TempDir {
            path: PathBuf,
        }
        impl TempDir {
            pub fn new(tag: &str) -> std::io::Result<Self> {
                let mut path = std::env::temp_dir();
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                let sequence = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
                path.push(format!(
                    "mer-server-test-{tag}-{}-{nanos}-{sequence}",
                    std::process::id()
                ));
                std::fs::create_dir_all(&path)?;
                Ok(Self { path })
            }
            pub fn path(&self) -> &std::path::Path {
                &self.path
            }
        }
        impl Drop for TempDir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.path);
            }
        }
    }

    async fn make_state() -> (AppState, TempDir) {
        let dir = TempDir::new("server").unwrap();
        // Tiny shape so the test stays cheap.
        let num_experts = 4u32;
        let d_model = 8;
        let d_ff = 16;
        let weight_bytes = crate::inference::expert_weight_bytes(d_model, d_ff);
        let block = 4096usize;
        let expert_size = ((weight_bytes + block - 1) / block) * block;
        generate_synthetic_experts(dir.path(), num_experts, expert_size, d_model, d_ff).unwrap();
        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.path().to_path_buf(),
                expert_size,
                block_align: block,
                use_direct_io: false, // tmpfs
                num_experts_per_layer: None,
            })
            .unwrap(),
        );
        let cache = Arc::new(MultiLayerExpertCache::single_layer(2));
        let pool = BufferPool::new(3, expert_size, block);
        let router = crate::gating::Router::Markov(Arc::new(TopKRouter::clustered(
            num_experts,
            2,
            2,
            0.9,
            1,
        )));
        let predictor = Arc::new(PredictiveLoader::new(num_experts, 1, 0.05, 1));
        let engine = Arc::new(Engine::with_options(
            cache,
            pool,
            storage,
            router,
            predictor,
            ModelShape {
                d_model,
                d_ff,
                hidden_seed: 1,
            },
            EngineOptions::default(),
        ));
        let state = AppState {
            engine,
            tokenizer: Arc::new(Tokenizer::bytes()),
            metrics: Metrics::new(),
            real_model: None,
            batch_scheduler: None,
            draft_engine: None,
            gpu_native_token_loop: None,
            speculation_k: 0,
            runtime: crate::config::LiveConfig::from_config(&{
                let mut c = test_minimal_cfg();
                c.server.max_tokens = 32;
                c
            }),
            sessions: None,
            middleware: MiddlewareState {
                api_keys: crate::middleware::ApiKeyGate::default(),
                rate_limit: crate::middleware::RateLimiter::new(0, 0),
                admission: crate::middleware::Admission::new(0, 0, None),
            },
        };
        (state, dir)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn health_endpoint_returns_ok() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 1024).await.unwrap();
        let s = String::from_utf8(body.to_vec()).unwrap();
        assert!(s.contains("\"status\":\"ok\""));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn completions_endpoint_generates_response() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state);
        let body = serde_json::json!({
            "prompt": "Once upon a time",
            "max_tokens": 4
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["object"], "text_completion");
        assert_eq!(v["choices"][0]["index"], 0);
        assert_eq!(v["usage"]["completion_tokens"], 4);
        assert!(v["choices"][0]["text"].is_string());
    }

    /// Finding 10: the server-wide `mer_tokens_generated_total` counter must
    /// track the *actual* produced completion tokens (summed across requests,
    /// no double counting), not the requested `max_tokens` cap.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tokens_generated_metric_tracks_produced_completion_tokens() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state);

        async fn complete(app: &axum::Router, max_tokens: usize) -> u64 {
            let body = serde_json::json!({ "prompt": "hi", "max_tokens": max_tokens }).to_string();
            let resp = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/completions")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            let bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
            let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            v["usage"]["completion_tokens"].as_u64().unwrap()
        }

        async fn scrape_generated(app: &axum::Router) -> u64 {
            let resp = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/metrics")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let bytes = to_bytes(resp.into_body(), 32 * 1024).await.unwrap();
            let s = String::from_utf8(bytes.to_vec()).unwrap();
            s.lines()
                .find_map(|l| {
                    l.strip_prefix("mer_tokens_generated_total ")
                        .and_then(|v| v.trim().parse::<f64>().ok())
                })
                .map(|f| f as u64)
                .unwrap_or(0)
        }

        let c1 = complete(&app, 4).await;
        let c2 = complete(&app, 2).await;
        let total = scrape_generated(&app).await;
        assert_eq!(
            total,
            c1 + c2,
            "generated-token counter ({total}) must equal produced completion tokens \
             ({c1} + {c2}) with no double counting"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metrics_endpoint_exposes_prometheus_format() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state.clone());
        // Generate one request first so the counters are non-zero.
        let _ = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from("{\"prompt\":\"hi\",\"max_tokens\":2}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 16 * 1024).await.unwrap();
        let s = String::from_utf8(body.to_vec()).unwrap();
        assert!(s.contains("mer_requests_total"));
        assert!(s.contains("mer_tokens_generated_total"));
    }

    /// When `MiddlewareState.api_keys` is configured (non-empty), the
    /// router must continue to expose `/health` and `/metrics` without
    /// any authentication so Kubernetes liveness/readiness probes and
    /// Prometheus scrapers keep working, while every `/v1/...` route
    /// is gated behind the API-key check and returns
    /// `401 Unauthorized` when no valid key is presented.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn public_endpoints_bypass_api_key_gate_while_protected_routes_require_it() {
        let (mut state, _tmp) = make_state().await;
        // Enable the API-key gate with a single key the test will
        // (deliberately) not present on the protected request.
        state.middleware.api_keys = crate::middleware::ApiKeyGate::new(&["sekret".to_string()]);
        let app = build_router(state);

        // /health: no auth header, must succeed.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/health must be reachable without an API key"
        );

        // /metrics: no auth header, must succeed.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/metrics must be reachable without an API key"
        );

        // Protected /v1/... route without a key → 401.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from("{\"prompt\":\"hi\",\"max_tokens\":1}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "/v1/completions must reject requests without a valid API key"
        );

        // And with the correct key the protected route is reachable
        // again (sanity-check that the gate isn't simply broken).
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer sekret")
                    .body(Body::from("{\"prompt\":\"hi\",\"max_tokens\":1}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "/v1/completions must succeed when a valid API key is presented"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn empty_prompt_returns_400() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from("{\"prompt\":\"\",\"max_tokens\":2}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chat_completions_round_trips() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state);
        let body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ],
            "max_tokens": 3
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn completions_with_stream_returns_sse_events() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state);
        let body = serde_json::json!({
            "prompt": "Once upon",
            "max_tokens": 3,
            "stream": true,
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .map(|v| v.to_str().unwrap_or("").to_string())
            .unwrap_or_default();
        assert!(
            ct.starts_with("text/event-stream"),
            "expected SSE content-type, got {ct:?}"
        );
        // Read enough body to capture all events for max_tokens=3.
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = String::from_utf8(body.to_vec()).unwrap();
        // Should contain at least one data: line with text_completion shape and a [DONE] terminator.
        assert!(s.contains("data: "), "expected SSE data: lines, got {s}");
        assert!(
            s.contains("text_completion"),
            "expected event payload to be a text_completion chunk; got {s}"
        );
        assert!(s.contains("[DONE]"), "expected [DONE] terminator; got {s}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chat_completions_with_stream_returns_sse_events() {
        let (state, _tmp) = make_state().await;
        let app = build_router(state);
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
            "stream": true,
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = String::from_utf8(body.to_vec()).unwrap();
        // Role event should appear before any content delta.
        assert!(
            s.contains("\"role\":\"assistant\""),
            "expected leading role event; got {s}"
        );
        assert!(
            s.contains("chat.completion.chunk"),
            "expected chunk objects; got {s}"
        );
        assert!(s.contains("[DONE]"));
    }

    /// Build an `AppState` whose engine + storage are sized for a real
    /// transformer with the given config. Returns the state plus the
    /// temp dir that holds the synthesised expert weight files.
    async fn make_state_with_real_model(cfg: crate::model::RealModelConfig) -> (AppState, TempDir) {
        let dir = TempDir::new("server-real").unwrap();
        let total = cfg.num_layers as u32 * cfg.num_experts as u32;
        let weight_bytes = crate::inference::expert_weight_bytes(cfg.d_model, cfg.d_ff);
        let block = 4096usize;
        let expert_size = ((weight_bytes + block - 1) / block) * block;
        crate::io_provider::generate_synthetic_experts(
            dir.path(),
            total,
            expert_size,
            cfg.d_model,
            cfg.d_ff,
        )
        .unwrap();
        let storage = Arc::new(
            NvmeStorage::new(StorageConfig {
                base_path: dir.path().to_path_buf(),
                expert_size,
                block_align: block,
                use_direct_io: false,
                num_experts_per_layer: None,
            })
            .unwrap(),
        );
        let cache = Arc::new(MultiLayerExpertCache::single_layer((total as usize).max(2)));
        let pool = BufferPool::new(total as usize + 2, expert_size, block);
        let router = crate::gating::Router::Markov(Arc::new(TopKRouter::new(total, cfg.top_k, 1)));
        let predictor = Arc::new(PredictiveLoader::new(total, 0, 0.05, 1));
        let engine = Arc::new(Engine::with_options(
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
        ));
        let model = Arc::new(crate::model::RealModel::new_seeded(cfg.clone(), 0xBEEF));
        let scheduler = Arc::new(crate::batch_scheduler::BatchScheduler::spawn(
            model.clone(),
            engine.clone(),
            crate::batch_scheduler::BatchConfig {
                max_batch_size: 4,
                batch_timeout: std::time::Duration::from_millis(2),
                ..Default::default()
            },
        ));
        let state = AppState {
            engine,
            tokenizer: Arc::new(Tokenizer::bytes()),
            metrics: Metrics::new(),
            real_model: Some(model),
            batch_scheduler: Some(scheduler),
            draft_engine: None,
            gpu_native_token_loop: None,
            speculation_k: 0,
            runtime: crate::config::LiveConfig::from_config(&{
                let mut c = test_minimal_cfg();
                c.server.max_tokens = 16;
                c
            }),
            sessions: Some(crate::session::SessionStore::new(
                std::time::Duration::from_secs(60),
            )),
            middleware: MiddlewareState {
                api_keys: crate::middleware::ApiKeyGate::default(),
                rate_limit: crate::middleware::RateLimiter::new(0, 0),
                admission: crate::middleware::Admission::new(0, 0, None),
            },
        };
        (state, dir)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metrics_endpoint_exposes_scheduler_prepass_metrics_when_enabled() {
        let (state, _tmp) = make_state_with_real_model(tiny_real_cfg()).await;
        let app = build_router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 32 * 1024).await.unwrap();
        let s = String::from_utf8(body.to_vec()).unwrap();
        assert!(s.contains("mer_scheduler_prepass_runs_total"));
        assert!(s.contains("mer_scheduler_prepass_prefetch_precision"));
    }

    #[test]
    fn request_stage_timing_disabled_does_not_publish_stage_data() {
        let metrics = Metrics::new();
        let mut timing = RequestStageTiming::new(&metrics, false, "completion");
        assert!(timing.timings.is_none());
        assert!(timing.publisher.is_none());

        let started = timing.start();
        assert!(started.is_none());
        timing.record_since(crate::stage_timing::TOTAL_PROMPT, started);
        timing.publish();
        drop(timing);

        let body = String::from_utf8(metrics.render().unwrap()).unwrap();
        assert!(!body.contains(r#"mer_stage_events_total{stage=""#));
        assert!(!body.contains(r#"mer_stage_seconds_total{stage=""#));
    }

    #[test]
    fn request_stage_timing_enabled_records_and_publishes_stage_data() {
        let metrics = Metrics::new();
        let mut timing = RequestStageTiming::new(&metrics, true, "completion");
        assert!(timing.timings.is_some());
        assert!(timing.publisher.is_some());

        let started = timing.start();
        assert!(started.is_some());
        timing.record_since(crate::stage_timing::TOTAL_PROMPT, started);
        timing.publish();

        let body = String::from_utf8(metrics.render().unwrap()).unwrap();
        assert!(body.contains(r#"mer_stage_events_total{stage="total_prompt"}"#));
        assert!(body.contains(r#"mer_stage_seconds_total{stage="total_prompt"}"#));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn completions_with_real_model_returns_logit_sampled_tokens() {
        let cfg = crate::model::RealModelConfig {
            // vocab=256 matches the byte tokenizer.
            vocab_size: 256,
            d_model: 16,
            d_ff: 32,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 4,
            num_layers: 2,
            num_experts: 4,
            top_k: 2,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: crate::architecture::Architecture::Mixtral,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        };
        let (state, _tmp) = make_state_with_real_model(cfg).await;
        let app = build_router(state.clone());
        let body = serde_json::json!({
            "prompt": "Hi",
            "max_tokens": 3
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["usage"]["completion_tokens"], 3);
        assert!(v["choices"][0]["text"].is_string());
        // The engine's hits/misses counters were populated by the
        // real-transformer path's `moe_step` calls.
        let r = state.engine.report();
        assert!(
            r.misses + r.hits > 0,
            "engine cache should be touched by real transformer path"
        );
        assert!(
            r.bytes_read > 0,
            "engine should have read expert bytes from SSD"
        );
        let metrics = String::from_utf8(state.metrics.render().unwrap()).unwrap();
        assert!(
            !metrics.contains(r#"mer_stage_events_total{stage=""#),
            "stage timing must stay unpublished by default:\n{metrics}"
        );
    }

    fn tiny_real_cfg() -> crate::model::RealModelConfig {
        crate::model::RealModelConfig {
            vocab_size: 256,
            d_model: 16,
            d_ff: 32,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 4,
            num_layers: 2,
            num_experts: 4,
            top_k: 2,
            rope_base: 10_000.0,
            rms_eps: 1e-6,
            window_size: None,
            architecture: crate::architecture::Architecture::Mixtral,
            first_k_dense_replace: 0,
            advanced: Default::default(),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_model_generation_uses_expected_forward_count_and_one_scheduler_registration() {
        let cfg = tiny_real_cfg();
        let layers = cfg.num_layers as u64;
        let (state, _tmp) = make_state_with_real_model(cfg).await;
        state
            .runtime
            .try_reload(&{
                let mut c = test_minimal_cfg();
                c.server.max_tokens = 16;
                c.real_transformer.stage_timing_enabled = true;
                c
            })
            .unwrap();
        let scheduler = state.batch_scheduler.as_ref().unwrap().clone();
        let before_forwards = state.engine.report().tokens_processed;

        let resp = generate(
            &state,
            "Hi",
            3,
            "test-model",
            crate::sampling::SamplingParams::greedy(),
            None,
        )
        .await
        .unwrap();

        assert_eq!(resp.usage.prompt_tokens, 2);
        assert_eq!(resp.usage.completion_tokens, 3);
        let routed_layer_calls = state
            .engine
            .report()
            .tokens_processed
            .saturating_sub(before_forwards);
        assert_eq!(
            routed_layer_calls,
            (resp.usage.prompt_tokens as u64 + resp.usage.completion_tokens as u64 - 1) * layers,
            "real request must perform P + C - 1 model forwards"
        );
        assert_eq!(scheduler.registrations_total(), 1);
        assert_eq!(scheduler.releases_total(), 1);
        assert_eq!(scheduler.active_requests(), 0);

        let metrics = String::from_utf8(state.metrics.render().unwrap()).unwrap();
        for stage in [
            crate::stage_timing::SCHEDULER_OVERHEAD,
            crate::stage_timing::TOTAL_PROMPT,
            crate::stage_timing::TOTAL_DECODE,
        ] {
            let needle = format!(r#"mer_stage_events_total{{stage="{stage}"}}"#);
            assert!(metrics.contains(&needle), "missing {needle} in:\n{metrics}");
        }
    }

    /// Part A7 (KV correctness): once a request has registered KV state
    /// with the scheduler, losing that state must fail the request —
    /// the direct fallback must never fabricate fresh KV caches for an
    /// in-progress sequence.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scheduler_kv_loss_fails_request_instead_of_resetting_context() {
        let cfg = tiny_real_cfg();
        let (state, _tmp) = make_state_with_real_model(cfg).await;
        let model = state.real_model.as_ref().unwrap().clone();
        let kv = model.fresh_kv_caches();
        let mut request = RealRequestState::new(&state, &model, kv, None);

        // First decode registers the request's KV with the scheduler.
        request
            .decode_step(1, 0, &crate::sampling::SamplingParams::greedy())
            .await
            .expect("first decode succeeds");
        let scheduler = state.batch_scheduler.as_ref().unwrap().clone();
        let id = request.request_id.expect("request registered");

        // Simulate the scheduler losing the request's KV ownership
        // (registry lookup failure / worker crash): release it out from
        // under the request state.
        let stolen = scheduler.release(id);
        assert!(stolen.is_some(), "state was registered");

        // The next decode hits NotRegistered, falls back to the direct
        // path, cannot recover the exact KV state — and must fail
        // rather than continue on a fresh cache.
        let err = request
            .decode_step(2, 1, &crate::sampling::SamplingParams::greedy())
            .await
            .expect_err("KV loss must fail the request");
        assert!(
            matches!(err, GenerateError::KvStateLost(_)),
            "expected typed KvStateLost error, got: {err}"
        );
        // And the request's terminal KV take must not fabricate state
        // either: session persistence must surface the typed loss.
        assert!(matches!(
            request.finish(),
            Err(GenerateError::KvStateLost(_))
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn one_token_prompt_and_one_completion_runs_one_model_forward() {
        let cfg = tiny_real_cfg();
        let layers = cfg.num_layers as u64;
        let (state, _tmp) = make_state_with_real_model(cfg).await;
        let before_forwards = state.engine.report().tokens_processed;

        let resp = generate(
            &state,
            "A",
            1,
            "test-model",
            crate::sampling::SamplingParams::greedy(),
            None,
        )
        .await
        .unwrap();

        assert_eq!(resp.usage.prompt_tokens, 1);
        assert_eq!(resp.usage.completion_tokens, 1);
        let routed_layer_calls = state
            .engine
            .report()
            .tokens_processed
            .saturating_sub(before_forwards);
        assert_eq!(routed_layer_calls, layers);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn session_persists_pending_generated_token_for_next_turn() {
        let cfg = tiny_real_cfg();
        let layers = cfg.num_layers as u64;
        let (state, _tmp) = make_state_with_real_model(cfg).await;
        let store = state.sessions.as_ref().unwrap().clone();

        let before_first = state.engine.report().tokens_processed;
        let _ = generate(
            &state,
            "Hi",
            1,
            "test-model",
            crate::sampling::SamplingParams::greedy(),
            Some("chat".to_string()),
        )
        .await
        .unwrap();
        let first_delta = state
            .engine
            .report()
            .tokens_processed
            .saturating_sub(before_first);
        assert_eq!(
            first_delta,
            2 * layers,
            "P=2,C=1 should forward only prompt tokens"
        );

        let (saved, checkout) = store.take("chat").expect("session must be saved");
        assert_eq!(saved.position, 2);
        assert!(
            saved.pending_token.is_some(),
            "last emitted token should be pending"
        );
        store.put("chat".to_string(), saved, Some(checkout));

        let before_second = state.engine.report().tokens_processed;
        let _ = generate(
            &state,
            "!",
            1,
            "test-model",
            crate::sampling::SamplingParams::greedy(),
            Some("chat".to_string()),
        )
        .await
        .unwrap();
        let second_delta = state
            .engine
            .report()
            .tokens_processed
            .saturating_sub(before_second);
        assert_eq!(
            second_delta,
            2 * layers,
            "continuation should ingest the pending token plus the one-token prompt"
        );

        let (saved, _checkout) = store.take("chat").expect("session must be saved again");
        assert_eq!(saved.position, 4);
        assert!(saved.pending_token.is_some());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streaming_and_non_streaming_real_model_outputs_match() {
        let cfg = tiny_real_cfg();
        let (state, _tmp) = make_state_with_real_model(cfg).await;
        let params = crate::sampling::SamplingParams::greedy();
        let non_stream = generate(&state, "Hi", 3, "test-model", params, None)
            .await
            .unwrap()
            .choices
            .into_iter()
            .next()
            .unwrap()
            .text;

        let mut stream = stream_tokens(state.clone(), "Hi".to_string(), 3, params, None)
            .await
            .unwrap();
        let mut streamed = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.expect("stream must not fail");
            // The final chunk may carry the decoder's held-back tail
            // (item 6): append it before stopping.
            streamed.push_str(&chunk.text);
            if chunk.finished {
                break;
            }
        }
        assert_eq!(streamed, non_stream);
    }

    /// Part A7 / item 5 (streaming): losing the scheduler-registered KV
    /// state mid-stream must surface as a typed stream error — never a
    /// successful `finished` chunk (which the SSE layer would turn into
    /// a normal `[DONE]`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streaming_kv_loss_yields_error_not_finished_chunk() {
        let cfg = tiny_real_cfg();
        let (state, _tmp) = make_state_with_real_model(cfg).await;
        let scheduler = state.batch_scheduler.as_ref().unwrap().clone();
        let params = crate::sampling::SamplingParams::greedy();
        let mut stream = stream_tokens(state.clone(), "Hi".to_string(), 4, params, None)
            .await
            .unwrap();

        // First chunk: the pending first token (no scheduler use yet).
        let c1 = stream
            .next()
            .await
            .unwrap()
            .expect("first chunk must succeed");
        assert!(!c1.finished);
        // Second chunk: the first decode step registers KV with the
        // scheduler.
        let c2 = stream
            .next()
            .await
            .unwrap()
            .expect("second chunk must succeed");
        assert!(!c2.finished);
        assert_eq!(scheduler.registrations_total(), 1);

        // Steal the registered KV state out from under the stream
        // (simulated registry loss). The fresh per-test scheduler
        // assigns id 0 to its first registration.
        let stolen = scheduler.release(crate::batch_scheduler::RequestId(0));
        assert!(stolen.is_some(), "the stream's KV state was registered");

        // The stream must now fail with a typed KV-state error and then
        // terminate — never emit a successful finished chunk.
        let mut saw_error = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(chunk) => assert!(
                    !chunk.finished,
                    "stream must not end successfully after KV loss"
                ),
                Err(e) => {
                    assert!(
                        matches!(e, GenerateError::KvStateLost(_)),
                        "expected KvStateLost, got: {e}"
                    );
                    saw_error = true;
                    break;
                }
            }
        }
        assert!(saw_error, "KV loss must surface as a stream error");
        assert!(
            stream.next().await.is_none(),
            "stream must terminate after the error"
        );
    }

    // =============== A7 closure tests (validation closure, item 1) ===============

    /// Spawn a [`BatchScheduler`] on a dedicated secondary runtime so
    /// the test can deterministically terminate the scheduler loop task
    /// (closing its mpsc channels) while the registry `Arc` — shared
    /// with the handle — stays alive.
    fn spawn_scheduler_on_runtime(
        model: &Arc<crate::model::RealModel>,
        engine: &Arc<Engine>,
    ) -> (
        tokio::runtime::Runtime,
        Arc<crate::batch_scheduler::BatchScheduler>,
    ) {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .expect("secondary runtime");
        let scheduler = {
            let _guard = rt.enter();
            Arc::new(crate::batch_scheduler::BatchScheduler::spawn(
                model.clone(),
                engine.clone(),
                crate::batch_scheduler::BatchConfig {
                    max_batch_size: 4,
                    batch_timeout: std::time::Duration::from_millis(2),
                    ..Default::default()
                },
            ))
        };
        (rt, scheduler)
    }

    /// Fully shut down the secondary runtime from inside an async test
    /// (dropping a runtime inline in async context is not allowed).
    /// After this returns the scheduler loop task has been dropped, so
    /// its submission channels are closed.
    async fn shutdown_runtime(rt: tokio::runtime::Runtime) {
        tokio::task::spawn_blocking(move || {
            rt.shutdown_timeout(std::time::Duration::from_secs(10));
        })
        .await
        .expect("runtime shutdown");
    }

    /// A7 item 1: the scheduler's channels close *before* any scheduled
    /// execution. The registered KV state is still in the shared
    /// registry, so the direct fallback must recover that exact state
    /// (never fabricate fresh KV) and complete the request.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scheduler_channel_closure_before_execution_recovers_registered_kv() {
        let cfg = tiny_real_cfg();
        let (mut state, _tmp) = make_state_with_real_model(cfg).await;
        let model = state.real_model.clone().unwrap();
        let (rt, scheduler) = spawn_scheduler_on_runtime(&model, &state.engine);
        state.batch_scheduler = Some(scheduler.clone());
        // Terminate the scheduler loop before it executes anything.
        shutdown_runtime(rt).await;

        let mut request = RealRequestState::new(&state, &model, model.fresh_kv_caches(), None);
        // decode_step registers the KV (registry op is synchronous and
        // survives the loop), then the channel send fails with
        // SchedulerClosed — the fallback must recover the registered
        // state and decode directly.
        request
            .decode_step(1, 0, &crate::sampling::SamplingParams::greedy())
            .await
            .expect("closure before execution must recover the registered KV and continue");
        assert_eq!(scheduler.registrations_total(), 1);
        assert_eq!(scheduler.releases_total(), 1);
        let kv = request.finish().expect("recovered KV must be persistable");
        assert_eq!(
            kv[0].seq_len, 1,
            "the direct decode advanced the recovered KV"
        );
    }

    /// A7 item 1: the scheduler's channels close *after* scheduled
    /// steps already mutated the registered KV state. The fallback
    /// must recover the exact mutated state — positions written by the
    /// scheduler are preserved, never reset.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scheduler_closure_after_kv_mutation_recovers_exact_mutated_state() {
        let cfg = tiny_real_cfg();
        let (mut state, _tmp) = make_state_with_real_model(cfg).await;
        let model = state.real_model.clone().unwrap();
        let (rt, scheduler) = spawn_scheduler_on_runtime(&model, &state.engine);
        state.batch_scheduler = Some(scheduler.clone());

        let greedy = crate::sampling::SamplingParams::greedy();
        let mut request = RealRequestState::new(&state, &model, model.fresh_kv_caches(), None);
        // Two scheduled decodes mutate the registered KV in place.
        let n1 = request
            .decode_step(1, 0, &greedy)
            .await
            .expect("scheduled step 1");
        let n2 = request
            .decode_step(n1, 1, &greedy)
            .await
            .expect("scheduled step 2");
        assert_eq!(scheduler.registrations_total(), 1);

        shutdown_runtime(rt).await;

        // Third decode: channel closed → direct fallback on the exact
        // recovered (mutated) KV state.
        request
            .decode_step(n2, 2, &greedy)
            .await
            .expect("closure after mutation must recover the mutated KV and continue");
        let kv = request.finish().expect("recovered KV must be persistable");
        assert_eq!(
            kv[0].seq_len, 3,
            "recovered KV must carry the scheduler-written positions plus the direct step"
        );
    }

    /// A7 item 1: end-to-end non-streaming request — losing the
    /// scheduler-registered KV state mid-request makes `generate`
    /// return a typed [`GenerateError::KvStateLost`], never a
    /// successful completion.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn non_streaming_generate_fails_with_kv_state_lost_after_scheduler_loss() {
        let cfg = tiny_real_cfg();
        let (state, _tmp) = make_state_with_real_model(cfg).await;
        state
            .runtime
            .try_reload(&{
                let mut c = test_minimal_cfg();
                c.server.max_tokens = 256;
                c
            })
            .unwrap();
        let scheduler = state.batch_scheduler.clone().unwrap();
        let st2 = state.clone();
        let task = tokio::spawn(async move {
            generate(
                &st2,
                "Hi",
                256,
                "test-model",
                crate::sampling::SamplingParams::greedy(),
                None,
            )
            .await
        });
        // Wait for the request to register its KV, then steal it.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while scheduler.registrations_total() < 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "request never registered"
            );
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        let _ = scheduler.release(crate::batch_scheduler::RequestId(0));
        let result = tokio::time::timeout(std::time::Duration::from_secs(60), task)
            .await
            .expect("request must terminate promptly after KV loss")
            .expect("task must not panic");
        let err = result.expect_err("KV loss must fail the non-streaming request");
        assert!(
            matches!(err, GenerateError::KvStateLost(_)),
            "expected KvStateLost, got: {err}"
        );
    }

    /// Read an SSE response body incrementally: accumulate frames into
    /// `acc` until `stop(acc)` returns true (or the body ends).
    async fn read_sse_until(
        body: &mut (impl Stream<Item = Result<axum::body::Bytes, axum::Error>> + Unpin),
        acc: &mut String,
        mut stop: impl FnMut(&str) -> bool,
    ) {
        use futures::stream::StreamExt;
        while !stop(acc) {
            match body.next().await {
                Some(frame) => {
                    let bytes = frame.expect("body frame");
                    acc.push_str(&String::from_utf8_lossy(&bytes));
                }
                None => break,
            }
        }
    }

    /// A7 item 1: completion SSE stream — KV loss mid-stream emits an
    /// OpenAI-compatible `kv_state_lost` error event and terminates
    /// with no `[DONE]` and no successful finish reason.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn completion_sse_stream_kv_loss_emits_error_event_and_no_done() {
        let cfg = tiny_real_cfg();
        let (state, _tmp) = make_state_with_real_model(cfg).await;
        state
            .runtime
            .try_reload(&{
                let mut c = test_minimal_cfg();
                c.server.max_tokens = 256;
                c
            })
            .unwrap();
        let scheduler = state.batch_scheduler.clone().unwrap();
        let app = build_router(state);
        let body = serde_json::json!({
            "prompt": "Hi",
            "max_tokens": 256,
            "stream": true,
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body = resp.into_body().into_data_stream();
        let mut acc = String::new();
        // Pull a couple of content events so the request has
        // registered its KV with the scheduler.
        read_sse_until(&mut body, &mut acc, |s| {
            s.matches("text_completion").count() >= 2
        })
        .await;
        assert!(scheduler.registrations_total() >= 1);
        let _ = scheduler.release(crate::batch_scheduler::RequestId(0));
        // Drain to the end of the stream.
        read_sse_until(&mut body, &mut acc, |_| false).await;
        assert!(
            acc.contains("kv_state_lost"),
            "expected a typed kv_state_lost error event; got:\n{acc}"
        );
        assert!(
            !acc.contains("[DONE]"),
            "a failed stream must not emit [DONE]; got:\n{acc}"
        );
        assert!(
            !acc.contains(r#""finish_reason":"length""#)
                && !acc.contains(r#""finish_reason":"stop""#),
            "a failed stream must not emit a normal finish reason; got:\n{acc}"
        );
    }

    /// A7 item 1: chat SSE stream — same contract as the completion
    /// stream (error event, no `[DONE]`, no normal finish reason).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chat_sse_stream_kv_loss_emits_error_event_and_no_done() {
        let cfg = tiny_real_cfg();
        let (state, _tmp) = make_state_with_real_model(cfg).await;
        state
            .runtime
            .try_reload(&{
                let mut c = test_minimal_cfg();
                c.server.max_tokens = 256;
                c
            })
            .unwrap();
        let scheduler = state.batch_scheduler.clone().unwrap();
        let app = build_router(state);
        let body = serde_json::json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 256,
            "stream": true,
        })
        .to_string();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body = resp.into_body().into_data_stream();
        let mut acc = String::new();
        // Role event + at least two content deltas → KV registered.
        read_sse_until(&mut body, &mut acc, |s| {
            s.matches("chat.completion.chunk").count() >= 3
        })
        .await;
        assert!(acc.contains(r#""role":"assistant""#));
        assert!(scheduler.registrations_total() >= 1);
        let _ = scheduler.release(crate::batch_scheduler::RequestId(0));
        read_sse_until(&mut body, &mut acc, |_| false).await;
        assert!(
            acc.contains("kv_state_lost"),
            "expected a typed kv_state_lost error event; got:\n{acc}"
        );
        assert!(
            !acc.contains("[DONE]"),
            "no [DONE] after failure; got:\n{acc}"
        );
        assert!(
            !acc.contains(r#""finish_reason":"length""#)
                && !acc.contains(r#""finish_reason":"stop""#),
            "no normal finish reason after failure; got:\n{acc}"
        );
    }

    /// A7 item 1: a request that fails on KV loss must not persist an
    /// empty or stale session cache, and a follow-up request with the
    /// same session id must start fresh and succeed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_kv_recovery_does_not_persist_session_and_next_request_starts_fresh() {
        let cfg = tiny_real_cfg();
        let (state, _tmp) = make_state_with_real_model(cfg).await;
        state
            .runtime
            .try_reload(&{
                let mut c = test_minimal_cfg();
                c.server.max_tokens = 256;
                c
            })
            .unwrap();
        let store = state.sessions.clone().unwrap();
        let scheduler = state.batch_scheduler.clone().unwrap();
        let greedy = crate::sampling::SamplingParams::greedy();

        // Turn 1: succeeds and persists the session.
        generate(&state, "Hi", 2, "test-model", greedy, Some("s".into()))
            .await
            .expect("turn 1 succeeds");
        assert_eq!(store.len(), 1, "turn 1 must persist the session");
        assert_eq!(scheduler.registrations_total(), 1);

        // Turn 2: KV loss mid-request → typed failure.
        let st2 = state.clone();
        let task = tokio::spawn(async move {
            generate(&st2, "!", 256, "test-model", greedy, Some("s".into())).await
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while scheduler.registrations_total() < 2 {
            assert!(
                std::time::Instant::now() < deadline,
                "turn 2 never registered"
            );
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
        let _ = scheduler.release(crate::batch_scheduler::RequestId(1));
        let err = tokio::time::timeout(std::time::Duration::from_secs(60), task)
            .await
            .expect("turn 2 must terminate promptly")
            .expect("task must not panic")
            .expect_err("turn 2 must fail on KV loss");
        assert!(matches!(err, GenerateError::KvStateLost(_)));

        // The failed request must not re-insert an empty or stale
        // session cache under the cursor.
        assert!(
            store.take("s").is_none(),
            "failed KV recovery must not persist a session entry"
        );

        // Turn 3: the same session id starts fresh and succeeds.
        let resp = generate(&state, "Hi", 2, "test-model", greedy, Some("s".into()))
            .await
            .expect("turn 3 must start a fresh session and succeed");
        assert_eq!(resp.usage.completion_tokens, 2);
        assert_eq!(store.len(), 1, "turn 3 must persist the fresh session");
    }

    /// A7 item 1: dropping an in-flight request (client cancellation)
    /// releases its scheduler registration exactly once.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drop_releases_scheduler_registration_exactly_once() {
        let cfg = tiny_real_cfg();
        let (state, _tmp) = make_state_with_real_model(cfg).await;
        let model = state.real_model.clone().unwrap();
        let scheduler = state.batch_scheduler.clone().unwrap();
        let mut request = RealRequestState::new(&state, &model, model.fresh_kv_caches(), None);
        request
            .decode_step(1, 0, &crate::sampling::SamplingParams::greedy())
            .await
            .expect("decode registers the request");
        assert_eq!(scheduler.registrations_total(), 1);
        assert_eq!(scheduler.active_requests(), 1);
        drop(request);
        assert_eq!(
            scheduler.releases_total(),
            1,
            "Drop must release exactly once"
        );
        assert_eq!(scheduler.active_requests(), 0);
    }

    /// A7 item 1: a normal `finish()` consumes the registration; the
    /// subsequent `Drop` must not double-release.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn finish_then_drop_does_not_double_release() {
        let cfg = tiny_real_cfg();
        let (state, _tmp) = make_state_with_real_model(cfg).await;
        let model = state.real_model.clone().unwrap();
        let scheduler = state.batch_scheduler.clone().unwrap();
        let mut request = RealRequestState::new(&state, &model, model.fresh_kv_caches(), None);
        request
            .decode_step(1, 0, &crate::sampling::SamplingParams::greedy())
            .await
            .expect("decode registers the request");
        let _kv = request.finish().expect("finish recovers the KV");
        assert_eq!(scheduler.releases_total(), 1);
        assert_eq!(
            scheduler.active_requests(),
            0,
            "finish + Drop must release the registration exactly once"
        );
    }

    // ============ SSE wrapper contracts (validation closure, item 2) ============

    /// Render a finite SSE event stream to its wire text.
    async fn sse_events_to_string(
        s: impl Stream<Item = Result<Event, Infallible>> + Send + 'static,
    ) -> String {
        let resp = Sse::new(s).into_response();
        let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    fn chunk(text: &str, finished: bool) -> Result<StreamChunk, GenerateError> {
        Ok(StreamChunk {
            text: text.to_string(),
            finished,
            hits: 0,
            misses: 0,
        })
    }

    fn scrape_tokens_generated(metrics: &Metrics) -> u64 {
        let s = String::from_utf8(metrics.render().unwrap()).unwrap();
        s.lines()
            .find_map(|l| {
                l.strip_prefix("mer_tokens_generated_total ")
                    .and_then(|v| v.trim().parse::<f64>().ok())
            })
            .map(|f| f as u64)
            .unwrap_or(0)
    }

    /// Item 2: the completion SSE wrapper emits a non-empty held-back
    /// decoder tail as a normal content chunk *before* `[DONE]`, and
    /// the token metric counts only real streamed tokens (not the
    /// tail chunk).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn completion_wrapper_emits_final_tail_before_done() {
        let metrics = Metrics::new();
        let inner: Pin<Box<dyn Stream<Item = Result<StreamChunk, GenerateError>> + Send>> =
            Box::pin(stream::iter(vec![
                chunk("he", false),
                chunk("y", false),
                chunk("TAIL€", true),
            ]));
        let s = wrap_completion_events(inner, "cmpl-t".into(), "m".into(), metrics.clone());
        let body = sse_events_to_string(s).await;
        let tail_idx = body.find("TAIL€").expect("tail chunk must be emitted");
        let done_idx = body
            .find("[DONE]")
            .expect("stream must terminate with [DONE]");
        assert!(
            tail_idx < done_idx,
            "the final decoder tail must be delivered before [DONE]:\n{body}"
        );
        assert!(body.contains("text_completion"));
        assert_eq!(
            scrape_tokens_generated(&metrics),
            2,
            "the tail chunk is held-back text from already-counted tokens"
        );
    }

    /// Item 2: the chat SSE wrapper behaves consistently with the
    /// completion wrapper — role event first, content deltas, tail
    /// before `[DONE]`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn chat_wrapper_emits_final_tail_before_done() {
        let metrics = Metrics::new();
        let inner: Pin<Box<dyn Stream<Item = Result<StreamChunk, GenerateError>> + Send>> =
            Box::pin(stream::iter(vec![
                chunk("he", false),
                chunk("y", false),
                chunk("TAIL€", true),
            ]));
        let s = wrap_chat_events(inner, "chatcmpl-t".into(), "m".into(), metrics.clone());
        let body = sse_events_to_string(s).await;
        let role_idx = body
            .find(r#""role":"assistant""#)
            .expect("role event first");
        let tail_idx = body.find("TAIL€").expect("tail chunk must be emitted");
        let done_idx = body
            .find("[DONE]")
            .expect("stream must terminate with [DONE]");
        assert!(role_idx < tail_idx && tail_idx < done_idx);
        assert!(body.contains("chat.completion.chunk"));
        assert_eq!(scrape_tokens_generated(&metrics), 2);
    }

    /// Item 2: a decoder/inference failure mid-stream emits an
    /// OpenAI-compatible error event, no `[DONE]`, no finish reason —
    /// and metrics count only the successfully emitted tokens. Both
    /// wrappers share the contract.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wrappers_emit_error_event_without_done_on_stream_failure() {
        for chat in [false, true] {
            let metrics = Metrics::new();
            let inner: Pin<Box<dyn Stream<Item = Result<StreamChunk, GenerateError>> + Send>> =
                Box::pin(stream::iter(vec![
                    chunk("ok", false),
                    Err(GenerateError::Tokenizer("decoder exploded".into())),
                ]));
            let body = if chat {
                let s = wrap_chat_events(inner, "chatcmpl-t".into(), "m".into(), metrics.clone());
                sse_events_to_string(s).await
            } else {
                let s = wrap_completion_events(inner, "cmpl-t".into(), "m".into(), metrics.clone());
                sse_events_to_string(s).await
            };
            assert!(
                body.contains(r#""type":"server_error""#),
                "chat={chat}: expected an error event; got:\n{body}"
            );
            assert!(body.contains("decoder exploded"), "chat={chat}");
            assert!(
                !body.contains("[DONE]"),
                "chat={chat}: no [DONE] after failure"
            );
            assert!(
                !body.contains(r#""finish_reason":"length""#)
                    && !body.contains(r#""finish_reason":"stop""#),
                "chat={chat}: no normal finish reason after failure"
            );
            assert_eq!(
                scrape_tokens_generated(&metrics),
                1,
                "chat={chat}: metrics must count only successfully emitted tokens"
            );
        }
    }

    #[test]
    fn gpu_native_error_mapping_to_http_status_codes() {
        use crate::gpu_native_token_loop::GpuNativeTokenLoopError;
        use axum::http::StatusCode;

        // 1. Client/Request errors map to GenerateError::InvalidRequest -> HTTP 400 (invalid_request_error)
        let context_limit_err = GpuNativeTokenLoopError::ContextLimitExceeded {
            requested_position: 9,
            max_seq_len: 8,
        };
        let gen_err1 = map_gpu_native_token_loop_error(context_limit_err);
        assert!(matches!(gen_err1, GenerateError::InvalidRequest(_)));
        let resp1 = error_response(gen_err1);
        assert_eq!(resp1.status(), StatusCode::BAD_REQUEST);

        let unsupported_sampling_err = GpuNativeTokenLoopError::UnsupportedSampling {
            reason: "temperature must be 0.0".into(),
        };
        let gen_err2 = map_gpu_native_token_loop_error(unsupported_sampling_err);
        assert!(matches!(gen_err2, GenerateError::InvalidRequest(_)));
        let resp2 = error_response(gen_err2);
        assert_eq!(resp2.status(), StatusCode::BAD_REQUEST);

        // 2. Runtime / GPU execution errors map to GenerateError::Inference -> HTTP 500 (server_error)
        let position_mismatch_err = GpuNativeTokenLoopError::PositionMismatch {
            requested_position: 3,
            committed_position: 2,
        };
        let gen_err_pos = map_gpu_native_token_loop_error(position_mismatch_err);
        assert!(matches!(gen_err_pos, GenerateError::Inference(_)));
        let resp_pos = error_response(gen_err_pos);
        assert_eq!(resp_pos.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let fatal_err = GpuNativeTokenLoopError::FatalNumericalFailure {
            layer_index: Some(0),
            status_bits: 0x1,
        };
        let gen_err3 = map_gpu_native_token_loop_error(fatal_err);
        assert!(matches!(gen_err3, GenerateError::Inference(_)));
        let resp3 = error_response(gen_err3);
        assert_eq!(resp3.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let unknown_status_err = GpuNativeTokenLoopError::UnknownStatusBits {
            layer_index: Some(0),
            status_bits: 1 << 31,
            unknown_bits: 1 << 31,
        };
        let gen_err_unknown = map_gpu_native_token_loop_error(unknown_status_err);
        assert!(matches!(gen_err_unknown, GenerateError::Inference(_)));
        let resp_unknown = error_response(gen_err_unknown);
        assert_eq!(resp_unknown.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let map_failed_err = GpuNativeTokenLoopError::MapFailed("device lost".into());
        let gen_err4 = map_gpu_native_token_loop_error(map_failed_err);
        assert!(matches!(gen_err4, GenerateError::Inference(_)));
        let resp4 = error_response(gen_err4);
        assert_eq!(resp4.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let no_progress_err = GpuNativeTokenLoopError::NoProgress {
            layer_index: 1,
            selected_ids: vec![0, 1],
        };
        let gen_err5 = map_gpu_native_token_loop_error(no_progress_err);
        assert!(matches!(gen_err5, GenerateError::Inference(_)));
        let resp5 = error_response(gen_err5);
        assert_eq!(resp5.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let attempt_bound_err = GpuNativeTokenLoopError::AttemptBoundExceeded {
            attempts: 4,
            max_attempts: 3,
        };
        let gen_err6 = map_gpu_native_token_loop_error(attempt_bound_err);
        assert!(matches!(gen_err6, GenerateError::Inference(_)));
        let resp6 = error_response(gen_err6);
        assert_eq!(resp6.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
