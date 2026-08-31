//! PR1C-D bounded, explicitly opt-in live GPU-native prefetch qualification.
//!
//! This module owns a private frozen predictor, lifecycle evidence, and the
//! OFF/ON qualification command. The normal token loop contains only an empty
//! optional slot; no controller is installed outside this command.

use crate::gpu_native_prefetch_shadow::{ShadowObserverError, ShadowPhase};
use crate::gpu_native_residency::{
    GpuNativeLivePostconditionEvidence, GpuNativeLiveReplacementPolicy,
    GpuNativeLiveSpeculativeInstall, GpuNativeTieredResidencyManager,
};
use crate::router::PredictiveLoader;
use parking_lot::Mutex;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(crate) const SCHEMA: &str = "mer.gpu-native-bounded-live-prefetch.v1";
pub(crate) const MODE: &str = "gpu-native-bounded-live-prefetch";
pub(crate) const COMMAND: &str = "qualify-gpu-native-bounded-live-prefetch";
pub(crate) const SOURCE_MAIN_COMMIT: &str = "e8b542110693e74aa8f1013bb16d1bed0bdd8ba7";
pub(crate) const TESTED_PR1BB_COMMIT: &str = "c7006c5c6fbcee74c91526f89a8c2b5b06d8a9c5";
pub(crate) const TESTED_PR1BB_REPORT_SHA256: &str =
    "82317ad85ba29da1401abf3041ebbb7c2ca72c039c1be0c5f2ea53df9e9fc529";
pub(crate) const PR1CA_COMMIT: &str = "5d1e16ad6f2d8c71809a8fb65b1fbb3e4c98972f";
pub(crate) const PR1CA_REPORT_SHA256: &str =
    "cd7d86d9c9ff6691c11320db9d5e44dc26d893a6e14a939a31ed6ab67b270830";
pub(crate) const PR1CB_COMMIT: &str = "7f313a5777b5cdb7394bcade1882abab54f0d797";
pub(crate) const PR1CB_REPORT_SHA256: &str =
    "bca66b00ecb54e815aa5a5c5fdd3f6ae1317ae267f96b39484cb41e11c300811";
pub(crate) const PR1CC_COMMIT: &str = "7309e53c3687f997a2143206e9e574858d4decaf";

pub(crate) const FROZEN_PREDICTOR: &str = "predictive-loader-second-order";
pub(crate) const FROZEN_FANOUT: usize = 8;
const MAX_RANKED_CANDIDATES: usize = 8;
const MARKOV_SEED: u64 = 0xC0FFEE;
const FROZEN_ADAPTER_NAME: &str = "NVIDIA L4";
const MAX_LIFECYCLE_SAMPLES: usize = 64;

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
    pub(crate) replacement_policy: GpuNativeLiveReplacementPolicy,
    pub(crate) report_out: Option<PathBuf>,
    pub(crate) progress_watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
}

pub(crate) fn require_explicit_replacement_policy(
    policy: Option<GpuNativeLiveReplacementPolicy>,
) -> Result<GpuNativeLiveReplacementPolicy, String> {
    policy.ok_or_else(|| {
        format!(
            "{COMMAND} requires explicit --replacement-policy <physical-lru|physical-lru-prediction-protected|route-recency|route-recency-prediction-protected>"
        )
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum LiveArm {
    Off,
    On,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct LiveResourceBounds {
    pub(crate) max_candidates_accepted_per_boundary: usize,
    pub(crate) max_installations_per_boundary: usize,
    pub(crate) max_replacements_per_boundary: usize,
    pub(crate) max_in_flight_source_acquisitions: usize,
    pub(crate) max_target_boundaries_survived: u64,
    pub(crate) max_lifecycle_samples: usize,
}

impl Default for LiveResourceBounds {
    fn default() -> Self {
        Self {
            max_candidates_accepted_per_boundary: 2,
            max_installations_per_boundary: 1,
            max_replacements_per_boundary: 1,
            max_in_flight_source_acquisitions: 1,
            max_target_boundaries_survived: 1,
            max_lifecycle_samples: MAX_LIFECYCLE_SAMPLES,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LiveControllerConfig {
    pub(crate) arm: LiveArm,
    pub(crate) policy: GpuNativeLiveReplacementPolicy,
    pub(crate) num_layers: usize,
    pub(crate) experts_per_layer: usize,
    pub(crate) top_k: usize,
    pub(crate) markov_min_prob: f64,
    pub(crate) bounds: LiveResourceBounds,
}

#[derive(Clone, Debug)]
struct RouteHistory {
    completed_token_position: usize,
    layer: usize,
    global_ids: Vec<u32>,
}

#[derive(Clone, Debug)]
struct FrozenCandidate {
    global_id: u32,
    score: f64,
}

#[derive(Clone, Debug)]
struct PendingTarget {
    completed_token_position: usize,
    target_layer: usize,
    boundary: u64,
    prediction_at_us: u64,
    target_probe_at_us: Option<u64>,
    candidates: Vec<FrozenCandidate>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum CandidateState {
    PredictionEmitted,
    SourceStarted,
    PhysicalInstallationStarted,
    PhysicalInstallationCompleted,
    Terminal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LiveCancellationReason {
    RejectedConcurrencyNoPermit,
    RejectedResidencyPressure,
    CancelledBoundaryExpired,
    CancelledBackgroundSpawn,
    CancelledTaskOrShutdown,
    CancelledStaleLogicalGeneration,
}

impl LiveCancellationReason {
    fn classification(self) -> &'static str {
        match self {
            Self::RejectedConcurrencyNoPermit => "rejected-concurrency-no-permit",
            Self::RejectedResidencyPressure => "rejected-residency-pressure",
            Self::CancelledBoundaryExpired => "cancelled-boundary-expired",
            Self::CancelledBackgroundSpawn => "cancelled-background-spawn",
            Self::CancelledTaskOrShutdown => "cancelled-task-or-shutdown",
            Self::CancelledStaleLogicalGeneration => "cancelled-stale-logical-generation",
        }
    }
}

#[derive(Clone, Debug)]
struct CandidateRecord {
    boundary: u64,
    target_layer: usize,
    global_id: u32,
    score: f64,
    prediction_at_us: u64,
    source_started_at_us: Option<u64>,
    physical_current_at_us: Option<u64>,
    first_demand_at_us: Option<u64>,
    state: CandidateState,
    source_bytes: u64,
    accepted: bool,
    terminal: bool,
    installed: bool,
    used: bool,
    demand_join_counted: bool,
    victim_demand_harm_counted: bool,
    victim_restored_after_speculation: bool,
    unused_eviction_counted: bool,
    replaced_global_id: Option<u32>,
    abort_handle: Option<tokio::task::AbortHandle>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LiveLifecycleSample {
    pub(crate) ticket_id: u64,
    pub(crate) run_index: usize,
    pub(crate) boundary: u64,
    pub(crate) target_layer: usize,
    pub(crate) global_id: u32,
    pub(crate) score: f64,
    pub(crate) terminal_classification: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct LiveLifecycleCounters {
    pub(crate) prediction_emitted: u64,
    pub(crate) rejected_live_disabled: u64,
    pub(crate) rejected_candidate_bound: u64,
    pub(crate) rejected_governor: u64,
    pub(crate) rejected_in_flight_bound: u64,
    pub(crate) rejected_install_bound: u64,
    pub(crate) rejected_replacement_bound: u64,
    pub(crate) already_physically_resident: u64,
    pub(crate) source_acquisition_started: u64,
    pub(crate) source_acquisition_deduplicated_joined: u64,
    pub(crate) speculative_joined_existing_acquisition: u64,
    pub(crate) demand_joined_speculative_acquisition: u64,
    pub(crate) source_acquisition_cancelled_stale: u64,
    pub(crate) rejected_concurrency_no_permit: u64,
    pub(crate) rejected_residency_pressure: u64,
    pub(crate) cancelled_boundary_expired: u64,
    pub(crate) cancelled_background_spawn: u64,
    pub(crate) cancelled_task_or_shutdown: u64,
    pub(crate) cancelled_stale_logical_generation: u64,
    pub(crate) physical_installation_started: u64,
    pub(crate) physical_installation_completed: u64,
    pub(crate) completed_before_first_demand: u64,
    pub(crate) demand_arrived_while_speculative_in_flight: u64,
    pub(crate) demand_reused_speculative_result: u64,
    pub(crate) prediction_useful: u64,
    pub(crate) prediction_useful_later: u64,
    pub(crate) prediction_unused: u64,
    pub(crate) speculative_replacements: u64,
    pub(crate) speculative_evictions: u64,
    pub(crate) evicted_expert_demanded_before_payoff: u64,
    pub(crate) misses_introduced_by_speculative_eviction: u64,
    pub(crate) miss_boundaries_introduced_by_speculative_eviction: u64,
    pub(crate) speculative_expert_evicted_unused: u64,
    pub(crate) prediction_protected_victim_skips: u64,
    pub(crate) forced_prediction_protected_evictions: u64,
    pub(crate) demand_independently_won_install_race: u64,
    pub(crate) candidate_already_ram_resident: u64,
    pub(crate) speculative_ram_hits: u64,
    pub(crate) speculative_ram_misses: u64,
    pub(crate) speculative_nvme_operations: u64,
    pub(crate) speculative_nvme_bytes: u64,
    pub(crate) speculative_h2d_installs: u64,
    pub(crate) speculative_h2d_bytes: u64,
    pub(crate) cancellations_wasted_source_bytes: u64,
    pub(crate) late_after_demand: u64,
    pub(crate) demand_physical_hits: u64,
    pub(crate) demand_physical_misses: u64,
    pub(crate) demand_miss_boundaries: u64,
    pub(crate) accepted_candidates: u64,
    pub(crate) completed_tasks: u64,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct LiveTimingSummary {
    pub(crate) samples: usize,
    pub(crate) p50_us: u64,
    pub(crate) p95_us: u64,
    pub(crate) p99_us: u64,
    pub(crate) max_us: u64,
}

impl LiveTimingSummary {
    fn from_values(values: &[u64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let at = |q: f64| {
            let index = ((sorted.len() - 1) as f64 * q).ceil() as usize;
            sorted[index.min(sorted.len() - 1)]
        };
        Self {
            samples: sorted.len(),
            p50_us: at(0.50),
            p95_us: at(0.95),
            p99_us: at(0.99),
            max_us: *sorted.last().expect("timing values checked nonempty"),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct LiveTimingEvidence {
    pub(crate) prediction_to_source_start: LiveTimingSummary,
    pub(crate) source_start_to_physical_current: LiveTimingSummary,
    pub(crate) prediction_to_physical_current: LiveTimingSummary,
    pub(crate) lead_time_before_first_demand: LiveTimingSummary,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LiveRunEvidence {
    pub(crate) arm: LiveArm,
    pub(crate) phase: ShadowPhase,
    pub(crate) run_index: usize,
    pub(crate) counters: LiveLifecycleCounters,
    pub(crate) timing: LiveTimingEvidence,
    pub(crate) lifecycle_samples: Vec<LiveLifecycleSample>,
    pub(crate) postconditions: GpuNativeLivePostconditionEvidence,
    pub(crate) counter_reconciliation_pass: bool,
    pub(crate) cancellation_reason_reconciliation_pass: bool,
    pub(crate) orphan_in_flight_source_entries: usize,
    pub(crate) no_orphan_in_flight_speculative_acquisitions: bool,
    pub(crate) no_leaked_install_reservations: bool,
}

struct TimingValues {
    prediction_to_source_start: Vec<u64>,
    source_start_to_physical_current: Vec<u64>,
    prediction_to_physical_current: Vec<u64>,
    lead_time_before_first_demand: Vec<u64>,
}

impl Default for TimingValues {
    fn default() -> Self {
        Self {
            prediction_to_source_start: Vec::new(),
            source_start_to_physical_current: Vec::new(),
            prediction_to_physical_current: Vec::new(),
            lead_time_before_first_demand: Vec::new(),
        }
    }
}

struct ActiveRun {
    phase: ShadowPhase,
    run_index: usize,
    next_boundary: u64,
    pending: Option<PendingTarget>,
    last_last_route: Option<RouteHistory>,
    last_route: Option<RouteHistory>,
    route_clock: u64,
    route_last_seen_clock: HashMap<u32, u64>,
    candidates: BTreeMap<u64, CandidateRecord>,
    installs_reserved_by_boundary: HashMap<u64, usize>,
    replacements_reserved_by_boundary: HashMap<u64, usize>,
    counters: LiveLifecycleCounters,
    timing: TimingValues,
    samples: Vec<LiveLifecycleSample>,
    failure: Option<String>,
}

struct ControllerInner {
    origin: Instant,
    next_ticket_id: u64,
    predictor: PredictiveLoader,
    run: Option<ActiveRun>,
}

pub(crate) struct LiveCandidateTicket {
    pub(crate) ticket_id: u64,
    pub(crate) global_id: u32,
    pub(crate) score: f64,
    pub(crate) protected_prediction_ids: Arc<HashSet<u32>>,
    pub(crate) route_last_seen_clock: Arc<HashMap<u32, u64>>,
}

pub(crate) struct GpuNativeBoundedLivePrefetchController {
    config: LiveControllerConfig,
    inner: Mutex<ControllerInner>,
    idle_notify: tokio::sync::Notify,
}

impl GpuNativeBoundedLivePrefetchController {
    pub(crate) fn new(config: LiveControllerConfig) -> Result<Arc<Self>, ShadowObserverError> {
        if config.num_layers < 2
            || config.experts_per_layer == 0
            || config.top_k == 0
            || config.top_k > config.experts_per_layer
            || config.bounds.max_candidates_accepted_per_boundary == 0
            || config.bounds.max_installations_per_boundary == 0
            || config.bounds.max_replacements_per_boundary == 0
            || config.bounds.max_in_flight_source_acquisitions == 0
        {
            return Err(ShadowObserverError::new(format!(
                "invalid bounded live-prefetch config: {config:?}"
            )));
        }
        let total_experts = config
            .num_layers
            .checked_mul(config.experts_per_layer)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| ShadowObserverError::new("live-prefetch expert namespace overflow"))?;
        Ok(Arc::new(Self {
            inner: Mutex::new(ControllerInner {
                origin: Instant::now(),
                next_ticket_id: 0,
                predictor: PredictiveLoader::new(
                    total_experts,
                    MAX_RANKED_CANDIDATES,
                    config.markov_min_prob,
                    MARKOV_SEED,
                ),
                run: None,
            }),
            config,
            idle_notify: tokio::sync::Notify::new(),
        }))
    }

    fn now_us(inner: &ControllerInner) -> u64 {
        inner.origin.elapsed().as_micros().min(u64::MAX as u128) as u64
    }

    pub(crate) fn policy(&self) -> GpuNativeLiveReplacementPolicy {
        self.config.policy
    }

    pub(crate) fn begin_run(
        &self,
        phase: ShadowPhase,
        run_index: usize,
    ) -> Result<(), ShadowObserverError> {
        let mut inner = self.inner.lock();
        if inner.run.is_some() {
            return Err(ShadowObserverError::new(
                "bounded live-prefetch begin_run called while a run is active",
            ));
        }
        inner.run = Some(ActiveRun {
            phase,
            run_index,
            next_boundary: 0,
            pending: None,
            last_last_route: None,
            last_route: None,
            route_clock: 0,
            route_last_seen_clock: HashMap::new(),
            candidates: BTreeMap::new(),
            installs_reserved_by_boundary: HashMap::new(),
            replacements_reserved_by_boundary: HashMap::new(),
            counters: LiveLifecycleCounters::default(),
            timing: TimingValues::default(),
            samples: Vec::new(),
            failure: None,
        });
        Ok(())
    }

    pub(crate) fn before_segment(
        &self,
        completed_token_position: usize,
        first_new_layer: usize,
    ) -> Result<(), ShadowObserverError> {
        let mut inner = self.inner.lock();
        let now = Self::now_us(&inner);
        let Some(run) = inner.run.as_mut() else {
            return Ok(());
        };
        let Some(pending) = run.pending.as_mut() else {
            return Ok(());
        };
        if pending.completed_token_position != completed_token_position
            || pending.target_layer != first_new_layer
        {
            return Err(ShadowObserverError::new(format!(
                "bounded live pending target position={} layer={} did not match segment position={completed_token_position} first_new_layer={first_new_layer}",
                pending.completed_token_position, pending.target_layer
            )));
        }
        if pending.target_probe_at_us.replace(now).is_some() {
            return Err(ShadowObserverError::new(
                "bounded live target probe was marked more than once",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn observe_boundary(
        self: &Arc<Self>,
        engine: &Arc<crate::engine::Engine>,
        residency: &Arc<GpuNativeTieredResidencyManager>,
        completed_token_position: usize,
        observed_layers: std::ops::RangeInclusive<usize>,
        selected_ids_by_layer: &[Vec<u32>],
    ) -> Result<(), ShadowObserverError> {
        let start = *observed_layers.start();
        let end = *observed_layers.end();
        if start > end || end >= self.config.num_layers || end >= selected_ids_by_layer.len() {
            return Err(ShadowObserverError::new(format!(
                "invalid bounded live observed layer range {start}..={end}"
            )));
        }

        let (expired, frozen) = {
            let mut inner = self.inner.lock();
            let now = Self::now_us(&inner);
            let ControllerInner { predictor, run, .. } = &mut *inner;
            let run = run.as_mut().ok_or_else(|| {
                ShadowObserverError::new("bounded live boundary observed without an active run")
            })?;
            for layer in start..=end {
                let local_ids = &selected_ids_by_layer[layer];
                validate_route(local_ids, self.config.top_k, self.config.experts_per_layer)?;
                let global_ids = local_ids
                    .iter()
                    .map(|&local_id| global_id(layer, local_id, self.config.experts_per_layer))
                    .collect::<Result<Vec<_>, _>>()?;

                if run.pending.as_ref().is_some_and(|pending| {
                    pending.completed_token_position == completed_token_position
                        && pending.target_layer == layer
                }) {
                    let pending = run.pending.take().expect("pending target checked");
                    let demand_at = pending.target_probe_at_us.ok_or_else(|| {
                        ShadowObserverError::new(
                            "bounded live target truth preceded its target-probe marker",
                        )
                    })?;
                    score_actual_demand(run, &pending, &global_ids, demand_at, residency.as_ref())?;
                }

                let previous = run.last_route.clone().filter(|route| {
                    route.completed_token_position == completed_token_position
                        && route.layer + 1 == layer
                });
                let previous_previous = run.last_last_route.clone().filter(|route| {
                    previous.as_ref().is_some_and(|previous| {
                        route.completed_token_position == completed_token_position
                            && route.layer + 1 == previous.layer
                    })
                });
                if let Some(previous) = previous.as_ref() {
                    predictor.observe_step2(
                        previous_previous
                            .as_ref()
                            .map(|route| route.global_ids.as_slice())
                            .unwrap_or(&[]),
                        &previous.global_ids,
                        &global_ids,
                    );
                }
                run.route_clock = run.route_clock.saturating_add(1);
                for &global_id in &global_ids {
                    run.route_last_seen_clock.insert(global_id, run.route_clock);
                }
                run.last_last_route = run.last_route.take();
                run.last_route = Some(RouteHistory {
                    completed_token_position,
                    layer,
                    global_ids,
                });
            }

            let frozen = if end + 1 < self.config.num_layers {
                let source = run.last_route.clone().ok_or_else(|| {
                    ShadowObserverError::new("bounded live observer lost the source route")
                })?;
                let source_previous = run.last_last_route.clone();
                let ranked =
                    freeze_ranked_predictions(predictor, &source, source_previous.as_ref())?;
                let boundary = run.next_boundary;
                run.next_boundary = run.next_boundary.saturating_add(1);
                let pending = PendingTarget {
                    completed_token_position,
                    target_layer: end + 1,
                    boundary,
                    prediction_at_us: now,
                    target_probe_at_us: None,
                    candidates: ranked,
                };
                run.counters.prediction_emitted = run
                    .counters
                    .prediction_emitted
                    .saturating_add(pending.candidates.len() as u64);
                run.pending = Some(pending.clone());
                Some(pending)
            } else {
                None
            };
            let current_boundary = run.next_boundary;
            let expired = run
                .candidates
                .iter()
                .filter(|(_, candidate)| {
                    !candidate.terminal
                        && current_boundary.saturating_sub(candidate.boundary)
                            > self.config.bounds.max_target_boundaries_survived
                })
                .map(|(&ticket_id, candidate)| (ticket_id, candidate.abort_handle.clone()))
                .collect::<Vec<_>>();
            (expired, frozen)
        };

        for (ticket_id, handle) in expired {
            self.record_cancelled(ticket_id, LiveCancellationReason::CancelledBoundaryExpired);
            if let Some(handle) = handle {
                handle.abort();
            }
        }
        if let Some(pending) = frozen {
            self.submit_frozen_candidates(engine, residency, pending)?;
        }
        Ok(())
    }

    fn submit_frozen_candidates(
        self: &Arc<Self>,
        engine: &Arc<crate::engine::Engine>,
        residency: &Arc<GpuNativeTieredResidencyManager>,
        pending: PendingTarget,
    ) -> Result<(), ShadowObserverError> {
        let protected = Arc::new(
            pending
                .candidates
                .iter()
                .map(|candidate| candidate.global_id)
                .collect::<HashSet<_>>(),
        );
        for (rank, candidate) in pending.candidates.iter().enumerate() {
            if rank >= self.config.bounds.max_candidates_accepted_per_boundary {
                let mut inner = self.inner.lock();
                if let Some(run) = inner.run.as_mut() {
                    run.counters.rejected_candidate_bound =
                        run.counters.rejected_candidate_bound.saturating_add(1);
                }
                continue;
            }
            let Some(ticket) = self.accept_candidate(&pending, candidate, protected.clone())?
            else {
                continue;
            };
            engine.spawn_bounded_live_prefetch(residency.clone(), self.clone(), ticket);
        }
        Ok(())
    }

    fn accept_candidate(
        &self,
        pending: &PendingTarget,
        candidate: &FrozenCandidate,
        protected: Arc<HashSet<u32>>,
    ) -> Result<Option<LiveCandidateTicket>, ShadowObserverError> {
        let mut inner = self.inner.lock();
        let ticket_id = inner.next_ticket_id;
        inner.next_ticket_id = inner.next_ticket_id.saturating_add(1);
        let run = inner.run.as_mut().ok_or_else(|| {
            ShadowObserverError::new("bounded live candidate accepted without active run")
        })?;
        if self.config.arm == LiveArm::Off {
            run.counters.rejected_live_disabled =
                run.counters.rejected_live_disabled.saturating_add(1);
            return Ok(None);
        }
        let active = run
            .candidates
            .values()
            .filter(|candidate| candidate.accepted && !candidate.terminal)
            .count();
        if active >= self.config.bounds.max_in_flight_source_acquisitions {
            run.counters.rejected_in_flight_bound =
                run.counters.rejected_in_flight_bound.saturating_add(1);
            return Ok(None);
        }
        run.counters.accepted_candidates = run.counters.accepted_candidates.saturating_add(1);
        run.candidates.insert(
            ticket_id,
            CandidateRecord {
                boundary: pending.boundary,
                target_layer: pending.target_layer,
                global_id: candidate.global_id,
                score: candidate.score,
                prediction_at_us: pending.prediction_at_us,
                source_started_at_us: None,
                physical_current_at_us: None,
                first_demand_at_us: None,
                state: CandidateState::PredictionEmitted,
                source_bytes: 0,
                accepted: true,
                terminal: false,
                installed: false,
                used: false,
                demand_join_counted: false,
                victim_demand_harm_counted: false,
                victim_restored_after_speculation: false,
                unused_eviction_counted: false,
                replaced_global_id: None,
                abort_handle: None,
            },
        );
        Ok(Some(LiveCandidateTicket {
            ticket_id,
            global_id: candidate.global_id,
            score: candidate.score,
            protected_prediction_ids: protected,
            route_last_seen_clock: Arc::new(run.route_last_seen_clock.clone()),
        }))
    }

    pub(crate) fn register_abort_handle(&self, ticket_id: u64, handle: tokio::task::AbortHandle) {
        let mut inner = self.inner.lock();
        if let Some(record) = inner
            .run
            .as_mut()
            .and_then(|run| run.candidates.get_mut(&ticket_id))
        {
            if !record.terminal {
                record.abort_handle = Some(handle);
            }
        }
    }

    pub(crate) fn record_already_physically_resident(&self, ticket_id: u64) {
        self.finish_ticket(
            ticket_id,
            "already-physically-resident",
            |run, record, now| {
                record.physical_current_at_us = Some(now);
                run.counters.already_physically_resident =
                    run.counters.already_physically_resident.saturating_add(1);
            },
        );
    }

    pub(crate) fn record_rejected_governor(&self, ticket_id: u64) {
        self.finish_ticket(ticket_id, "rejected-governor", |run, _, _| {
            run.counters.rejected_governor = run.counters.rejected_governor.saturating_add(1);
        });
    }

    pub(crate) fn record_source_started(&self, ticket_id: u64, ram_hit: bool, joined: bool) {
        let mut inner = self.inner.lock();
        let now = Self::now_us(&inner);
        let Some(run) = inner.run.as_mut() else {
            return;
        };
        let Some(record) = run.candidates.get_mut(&ticket_id) else {
            return;
        };
        if record.terminal || record.source_started_at_us.is_some() {
            return;
        }
        record.source_started_at_us = Some(now);
        record.state = CandidateState::SourceStarted;
        run.timing
            .prediction_to_source_start
            .push(now.saturating_sub(record.prediction_at_us));
        if joined {
            record.demand_join_counted = true;
            run.counters.source_acquisition_deduplicated_joined = run
                .counters
                .source_acquisition_deduplicated_joined
                .saturating_add(1);
            run.counters.speculative_joined_existing_acquisition = run
                .counters
                .speculative_joined_existing_acquisition
                .saturating_add(1);
        } else if ram_hit {
            run.counters.source_acquisition_started =
                run.counters.source_acquisition_started.saturating_add(1);
            run.counters.candidate_already_ram_resident = run
                .counters
                .candidate_already_ram_resident
                .saturating_add(1);
            run.counters.speculative_ram_hits = run.counters.speculative_ram_hits.saturating_add(1);
        } else {
            run.counters.source_acquisition_started =
                run.counters.source_acquisition_started.saturating_add(1);
            run.counters.speculative_ram_misses =
                run.counters.speculative_ram_misses.saturating_add(1);
        }
    }

    pub(crate) fn record_demand_joined_source(&self, global_id: u32) {
        let mut inner = self.inner.lock();
        let Some(run) = inner.run.as_mut() else {
            return;
        };
        let Some(record) = run.candidates.values_mut().find(|record| {
            !record.terminal
                && record.global_id == global_id
                && record.source_started_at_us.is_some()
                && !record.demand_join_counted
        }) else {
            return;
        };
        record.demand_join_counted = true;
        run.counters.source_acquisition_deduplicated_joined = run
            .counters
            .source_acquisition_deduplicated_joined
            .saturating_add(1);
        run.counters.demand_joined_speculative_acquisition = run
            .counters
            .demand_joined_speculative_acquisition
            .saturating_add(1);
    }

    /// Called only after the foreground demand transaction has returned the
    /// complete selected set. This closes the race where a speculative install
    /// finishes after the target probe but early enough for that same demand
    /// transaction to reuse it.
    pub(crate) fn record_demand_service_completed(&self, global_ids: &[u32]) {
        let selected = global_ids.iter().copied().collect::<HashSet<_>>();
        let mut inner = self.inner.lock();
        let Some(run) = inner.run.as_mut() else {
            return;
        };
        for record in run.candidates.values_mut() {
            if record.first_demand_at_us.is_some()
                && selected.contains(&record.global_id)
                && record.installed
                && !record.used
            {
                record.used = true;
                run.counters.demand_reused_speculative_result = run
                    .counters
                    .demand_reused_speculative_result
                    .saturating_add(1);
                run.counters.prediction_useful = run.counters.prediction_useful.saturating_add(1);
            }
        }
    }

    pub(crate) fn record_nvme_complete(&self, ticket_id: u64, bytes: u64) {
        let mut inner = self.inner.lock();
        let Some(run) = inner.run.as_mut() else {
            return;
        };
        let Some(record) = run.candidates.get_mut(&ticket_id) else {
            return;
        };
        record.source_bytes = record.source_bytes.saturating_add(bytes);
        run.counters.speculative_nvme_operations =
            run.counters.speculative_nvme_operations.saturating_add(1);
        run.counters.speculative_nvme_bytes =
            run.counters.speculative_nvme_bytes.saturating_add(bytes);
    }

    pub(crate) fn reserve_install(&self, ticket_id: u64, replacement_needed: bool) -> bool {
        let mut inner = self.inner.lock();
        let Some(run) = inner.run.as_mut() else {
            return false;
        };
        let Some(record) = run.candidates.get_mut(&ticket_id) else {
            return false;
        };
        if record.terminal {
            return false;
        }
        let install_count = run
            .installs_reserved_by_boundary
            .entry(record.boundary)
            .or_default();
        if *install_count >= self.config.bounds.max_installations_per_boundary {
            run.counters.rejected_install_bound =
                run.counters.rejected_install_bound.saturating_add(1);
            return false;
        }
        if replacement_needed {
            let replacement_count = run
                .replacements_reserved_by_boundary
                .entry(record.boundary)
                .or_default();
            if *replacement_count >= self.config.bounds.max_replacements_per_boundary {
                run.counters.rejected_replacement_bound =
                    run.counters.rejected_replacement_bound.saturating_add(1);
                return false;
            }
            *replacement_count += 1;
        }
        *install_count += 1;
        record.state = CandidateState::PhysicalInstallationStarted;
        run.counters.physical_installation_started =
            run.counters.physical_installation_started.saturating_add(1);
        true
    }

    pub(crate) fn record_install_outcome(
        &self,
        ticket_id: u64,
        outcome: GpuNativeLiveSpeculativeInstall,
        h2d_bytes: u64,
    ) {
        match outcome {
            GpuNativeLiveSpeculativeInstall::Hit(_) => {
                self.record_demand_or_peer_won_install_race(ticket_id);
            }
            GpuNativeLiveSpeculativeInstall::Installed {
                residency: _,
                replaced_global_id,
                prediction_protected_victim_skips,
                forced_prediction_protected_eviction,
            } => {
                self.record_installed(
                    ticket_id,
                    h2d_bytes,
                    replaced_global_id,
                    prediction_protected_victim_skips,
                    forced_prediction_protected_eviction,
                );
            }
            GpuNativeLiveSpeculativeInstall::DroppedCapacityOrPressure => {
                self.record_cancelled(
                    ticket_id,
                    LiveCancellationReason::RejectedResidencyPressure,
                );
            }
            GpuNativeLiveSpeculativeInstall::StaleLogicalGeneration => {
                self.record_cancelled(
                    ticket_id,
                    LiveCancellationReason::CancelledStaleLogicalGeneration,
                );
            }
        }
    }

    fn record_demand_or_peer_won_install_race(&self, ticket_id: u64) {
        self.finish_ticket(
            ticket_id,
            "demand-or-peer-won-install-race",
            |run, record, now| {
                record.physical_current_at_us = Some(now);
                run.counters.demand_independently_won_install_race = run
                    .counters
                    .demand_independently_won_install_race
                    .saturating_add(1);
                if record.first_demand_at_us.is_some() {
                    run.counters.late_after_demand =
                        run.counters.late_after_demand.saturating_add(1);
                }
                record_completion_timing(run, record, now);
            },
        );
    }

    fn record_installed(
        &self,
        ticket_id: u64,
        h2d_bytes: u64,
        replaced_global_id: Option<u32>,
        prediction_protected_victim_skips: u64,
        forced_prediction_protected_eviction: bool,
    ) {
        self.finish_ticket(
            ticket_id,
            "physical-installation-completed",
            |run, record, now| {
                record.state = CandidateState::PhysicalInstallationCompleted;
                record.physical_current_at_us = Some(now);
                record.installed = true;
                record.replaced_global_id = replaced_global_id;
                run.counters.physical_installation_completed = run
                    .counters
                    .physical_installation_completed
                    .saturating_add(1);
                run.counters.speculative_h2d_installs =
                    run.counters.speculative_h2d_installs.saturating_add(1);
                run.counters.speculative_h2d_bytes =
                    run.counters.speculative_h2d_bytes.saturating_add(h2d_bytes);
                if replaced_global_id.is_some() {
                    run.counters.speculative_replacements =
                        run.counters.speculative_replacements.saturating_add(1);
                    run.counters.speculative_evictions =
                        run.counters.speculative_evictions.saturating_add(1);
                }
                run.counters.prediction_protected_victim_skips = run
                    .counters
                    .prediction_protected_victim_skips
                    .saturating_add(prediction_protected_victim_skips);
                run.counters.forced_prediction_protected_evictions = run
                    .counters
                    .forced_prediction_protected_evictions
                    .saturating_add(u64::from(forced_prediction_protected_eviction));
                record_completion_timing(run, record, now);
            },
        );
    }

    pub(crate) fn record_cancelled(&self, ticket_id: u64, reason: LiveCancellationReason) {
        self.finish_ticket(ticket_id, reason.classification(), |run, record, _| {
            run.counters.source_acquisition_cancelled_stale = run
                .counters
                .source_acquisition_cancelled_stale
                .saturating_add(1);
            let attributed = match reason {
                LiveCancellationReason::RejectedConcurrencyNoPermit => {
                    &mut run.counters.rejected_concurrency_no_permit
                }
                LiveCancellationReason::RejectedResidencyPressure => {
                    &mut run.counters.rejected_residency_pressure
                }
                LiveCancellationReason::CancelledBoundaryExpired => {
                    &mut run.counters.cancelled_boundary_expired
                }
                LiveCancellationReason::CancelledBackgroundSpawn => {
                    &mut run.counters.cancelled_background_spawn
                }
                LiveCancellationReason::CancelledTaskOrShutdown => {
                    &mut run.counters.cancelled_task_or_shutdown
                }
                LiveCancellationReason::CancelledStaleLogicalGeneration => {
                    &mut run.counters.cancelled_stale_logical_generation
                }
            };
            *attributed = attributed.saturating_add(1);
            run.counters.cancellations_wasted_source_bytes = run
                .counters
                .cancellations_wasted_source_bytes
                .saturating_add(record.source_bytes);
        });
    }

    pub(crate) fn record_fatal(&self, ticket_id: u64, detail: String) {
        self.finish_ticket(ticket_id, "fatal-invariant-failure", |run, _, _| {
            if run.failure.is_none() {
                run.failure = Some(detail);
            }
        });
    }

    fn finish_ticket<F>(&self, ticket_id: u64, classification: &str, update: F)
    where
        F: FnOnce(&mut ActiveRun, &mut CandidateRecord, u64),
    {
        let mut inner = self.inner.lock();
        let now = Self::now_us(&inner);
        let Some(run) = inner.run.as_mut() else {
            return;
        };
        let Some(mut record) = run.candidates.remove(&ticket_id) else {
            return;
        };
        if record.terminal {
            run.candidates.insert(ticket_id, record);
            return;
        }
        update(run, &mut record, now);
        record.terminal = true;
        record.state = CandidateState::Terminal;
        record.abort_handle = None;
        run.counters.completed_tasks = run.counters.completed_tasks.saturating_add(1);
        if run.samples.len() < self.config.bounds.max_lifecycle_samples {
            run.samples.push(LiveLifecycleSample {
                ticket_id,
                run_index: run.run_index,
                boundary: record.boundary,
                target_layer: record.target_layer,
                global_id: record.global_id,
                score: record.score,
                terminal_classification: classification.to_string(),
            });
        }
        run.candidates.insert(ticket_id, record);
        self.idle_notify.notify_waiters();
    }

    pub(crate) async fn end_run(
        &self,
        engine: &crate::engine::Engine,
        residency: &GpuNativeTieredResidencyManager,
    ) -> Result<LiveRunEvidence, ShadowObserverError> {
        let cancellations = {
            let inner = self.inner.lock();
            let run = inner.run.as_ref().ok_or_else(|| {
                ShadowObserverError::new("bounded live end_run called without active run")
            })?;
            run.candidates
                .iter()
                .filter(|(_, candidate)| !candidate.terminal)
                .map(|(&ticket_id, candidate)| (ticket_id, candidate.abort_handle.clone()))
                .collect::<Vec<_>>()
        };
        for (ticket_id, handle) in cancellations {
            self.record_cancelled(
                ticket_id,
                LiveCancellationReason::CancelledTaskOrShutdown,
            );
            if let Some(handle) = handle {
                handle.abort();
            }
        }
        let wait =
            async {
                loop {
                    let notified = self.idle_notify.notified();
                    let idle =
                        self.inner.lock().run.as_ref().is_some_and(|run| {
                            run.candidates.values().all(|record| record.terminal)
                        });
                    if idle {
                        return;
                    }
                    notified.await;
                }
            };
        tokio::time::timeout(Duration::from_secs(5), wait)
            .await
            .map_err(|_| {
                ShadowObserverError::new("live-prefetch tasks did not retire within 5s")
            })?;
        let orphan_in_flight_source_entries = engine.bounded_live_source_in_flight_count();
        if orphan_in_flight_source_entries != 0 {
            return Err(ShadowObserverError::new(format!(
                "live-prefetch finalization found {orphan_in_flight_source_entries} orphan singleflight source entries"
            )));
        }
        let postconditions = residency.validate_live_postconditions()?;
        let mut inner = self.inner.lock();
        let mut run = inner.run.take().ok_or_else(|| {
            ShadowObserverError::new("bounded live active run disappeared during finalization")
        })?;
        if let Some(failure) = run.failure.as_ref() {
            return Err(ShadowObserverError::new(format!(
                "live-prefetch invariant failed closed: {failure}"
            )));
        }
        let accepted_terminal = run
            .candidates
            .values()
            .filter(|candidate| candidate.accepted && candidate.terminal)
            .count() as u64;
        let counter_reconciliation_pass = accepted_terminal == run.counters.accepted_candidates
            && run.counters.completed_tasks == run.counters.accepted_candidates;
        if !counter_reconciliation_pass {
            return Err(ShadowObserverError::new(format!(
                "live-prefetch counters did not reconcile: accepted={} terminal={} completed={}",
                run.counters.accepted_candidates, accepted_terminal, run.counters.completed_tasks
            )));
        }
        let attributed_cancellations = [
            run.counters.rejected_concurrency_no_permit,
            run.counters.rejected_residency_pressure,
            run.counters.cancelled_boundary_expired,
            run.counters.cancelled_background_spawn,
            run.counters.cancelled_task_or_shutdown,
            run.counters.cancelled_stale_logical_generation,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .ok_or_else(|| ShadowObserverError::new("live-prefetch cancellation counters overflowed"))?;
        let cancellation_reason_reconciliation_pass =
            attributed_cancellations == run.counters.source_acquisition_cancelled_stale;
        if !cancellation_reason_reconciliation_pass {
            return Err(ShadowObserverError::new(format!(
                "live-prefetch cancellation reasons did not reconcile: aggregate={} attributed={attributed_cancellations}",
                run.counters.source_acquisition_cancelled_stale
            )));
        }
        run.counters.prediction_unused = run
            .candidates
            .values()
            .filter(|candidate| candidate.installed && !candidate.used)
            .count() as u64;
        let timing = LiveTimingEvidence {
            prediction_to_source_start: LiveTimingSummary::from_values(
                &run.timing.prediction_to_source_start,
            ),
            source_start_to_physical_current: LiveTimingSummary::from_values(
                &run.timing.source_start_to_physical_current,
            ),
            prediction_to_physical_current: LiveTimingSummary::from_values(
                &run.timing.prediction_to_physical_current,
            ),
            lead_time_before_first_demand: LiveTimingSummary::from_values(
                &run.timing.lead_time_before_first_demand,
            ),
        };
        Ok(LiveRunEvidence {
            arm: self.config.arm,
            phase: run.phase,
            run_index: run.run_index,
            counters: run.counters,
            timing,
            lifecycle_samples: run.samples,
            postconditions,
            counter_reconciliation_pass,
            cancellation_reason_reconciliation_pass,
            orphan_in_flight_source_entries,
            no_orphan_in_flight_speculative_acquisitions: true,
            no_leaked_install_reservations: true,
        })
    }
}

fn record_completion_timing(run: &mut ActiveRun, record: &CandidateRecord, now: u64) {
    if let Some(source_started) = record.source_started_at_us {
        run.timing
            .source_start_to_physical_current
            .push(now.saturating_sub(source_started));
    }
    run.timing
        .prediction_to_physical_current
        .push(now.saturating_sub(record.prediction_at_us));
    if let Some(demand_at) = record.first_demand_at_us {
        if now <= demand_at {
            run.counters.completed_before_first_demand =
                run.counters.completed_before_first_demand.saturating_add(1);
            run.timing
                .lead_time_before_first_demand
                .push(demand_at.saturating_sub(now));
        }
    }
}

fn score_actual_demand(
    run: &mut ActiveRun,
    pending: &PendingTarget,
    selected: &[u32],
    demand_at: u64,
    residency: &GpuNativeTieredResidencyManager,
) -> Result<(), ShadowObserverError> {
    let selected_set = selected.iter().copied().collect::<HashSet<_>>();
    let mut misses = 0u64;
    for &global_id in selected {
        if residency.has_current_for_demand(global_id)? {
            run.counters.demand_physical_hits = run.counters.demand_physical_hits.saturating_add(1);
        } else {
            misses = misses.saturating_add(1);
            run.counters.demand_physical_misses =
                run.counters.demand_physical_misses.saturating_add(1);
        }
    }
    run.counters.demand_miss_boundaries = run
        .counters
        .demand_miss_boundaries
        .saturating_add(u64::from(misses > 0));

    classify_candidate_demand(run, pending, &selected_set, demand_at, |global_id| {
        residency
            .has_current_for_demand(global_id)
            .map_err(ShadowObserverError::from)
    })
}

fn classify_candidate_demand<F>(
    run: &mut ActiveRun,
    pending: &PendingTarget,
    selected_set: &HashSet<u32>,
    demand_at: u64,
    mut is_current: F,
) -> Result<(), ShadowObserverError>
where
    F: FnMut(u32) -> Result<bool, ShadowObserverError>,
{
    let mut attributed_miss_boundary = false;
    for record in run.candidates.values_mut() {
        if record.boundary == pending.boundary && selected_set.contains(&record.global_id) {
            record.first_demand_at_us.get_or_insert(demand_at);
            if record.terminal && record.installed && is_current(record.global_id)? {
                record.used = true;
                run.counters.demand_reused_speculative_result = run
                    .counters
                    .demand_reused_speculative_result
                    .saturating_add(1);
                run.counters.prediction_useful = run.counters.prediction_useful.saturating_add(1);
                if let Some(current_at) = record.physical_current_at_us {
                    if current_at <= demand_at {
                        run.counters.completed_before_first_demand =
                            run.counters.completed_before_first_demand.saturating_add(1);
                        run.timing
                            .lead_time_before_first_demand
                            .push(demand_at.saturating_sub(current_at));
                    }
                }
            } else if !record.terminal {
                run.counters.demand_arrived_while_speculative_in_flight = run
                    .counters
                    .demand_arrived_while_speculative_in_flight
                    .saturating_add(1);
            }
        }
        if record.boundary != pending.boundary
            && record.installed
            && !record.used
            && selected_set.contains(&record.global_id)
            && is_current(record.global_id)?
        {
            record.used = true;
            run.counters.prediction_useful = run.counters.prediction_useful.saturating_add(1);
            run.counters.prediction_useful_later =
                run.counters.prediction_useful_later.saturating_add(1);
        }
        if let Some(victim) = record.replaced_global_id {
            let victim_current = is_current(victim)?;
            record.victim_restored_after_speculation |= victim_current;
            if selected_set.contains(&victim) && !record.used && !record.victim_demand_harm_counted
            {
                record.victim_demand_harm_counted = true;
                run.counters.evicted_expert_demanded_before_payoff = run
                    .counters
                    .evicted_expert_demanded_before_payoff
                    .saturating_add(1);
                if !victim_current && !record.victim_restored_after_speculation {
                    run.counters.misses_introduced_by_speculative_eviction = run
                        .counters
                        .misses_introduced_by_speculative_eviction
                        .saturating_add(1);
                    attributed_miss_boundary = true;
                }
            }
        }
        if record.installed
            && !record.used
            && !record.unused_eviction_counted
            && !is_current(record.global_id)?
        {
            record.unused_eviction_counted = true;
            run.counters.speculative_expert_evicted_unused = run
                .counters
                .speculative_expert_evicted_unused
                .saturating_add(1);
        }
    }
    if attributed_miss_boundary {
        run.counters
            .miss_boundaries_introduced_by_speculative_eviction = run
            .counters
            .miss_boundaries_introduced_by_speculative_eviction
            .saturating_add(1);
    }
    Ok(())
}

fn freeze_ranked_predictions(
    predictor: &PredictiveLoader,
    source: &RouteHistory,
    source_previous: Option<&RouteHistory>,
) -> Result<Vec<FrozenCandidate>, ShadowObserverError> {
    let seed = *source
        .global_ids
        .last()
        .ok_or_else(|| ShadowObserverError::new("bounded live source route was empty"))?;
    let ranked = match source_previous
        .filter(|previous| {
            previous.completed_token_position == source.completed_token_position
                && previous.layer + 1 == source.layer
        })
        .and_then(|previous| previous.global_ids.last().copied())
    {
        Some(previous_seed) => {
            predictor.predict_next2_shadow_ranked(previous_seed, seed, MAX_RANKED_CANDIDATES)
        }
        None => predictor.predict_next_shadow_ranked(seed, MAX_RANKED_CANDIDATES),
    };
    let mut seen = HashSet::with_capacity(ranked.len());
    Ok(ranked
        .into_iter()
        .filter_map(|(global_id, score)| {
            seen.insert(global_id)
                .then_some(FrozenCandidate { global_id, score })
        })
        .take(FROZEN_FANOUT)
        .collect())
}

fn validate_route(
    local_ids: &[u32],
    top_k: usize,
    experts_per_layer: usize,
) -> Result<(), ShadowObserverError> {
    if local_ids.len() != top_k {
        return Err(ShadowObserverError::new(format!(
            "bounded live route contained {} experts, expected {top_k}",
            local_ids.len()
        )));
    }
    let mut seen = HashSet::with_capacity(local_ids.len());
    for &local_id in local_ids {
        if local_id as usize >= experts_per_layer || !seen.insert(local_id) {
            return Err(ShadowObserverError::new(format!(
                "invalid bounded live local route expert {local_id}"
            )));
        }
    }
    Ok(())
}

fn global_id(
    layer: usize,
    local_id: u32,
    experts_per_layer: usize,
) -> Result<u32, ShadowObserverError> {
    layer
        .checked_mul(experts_per_layer)
        .and_then(|base| base.checked_add(local_id as usize))
        .and_then(|id| u32::try_from(id).ok())
        .ok_or_else(|| ShadowObserverError::new("bounded live global expert id overflow"))
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LiveQualifiedRunEvidence {
    pub(crate) production: crate::gpu_native_real_benchmark::PerRunResult,
    pub(crate) live: LiveRunEvidence,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct DemandSourceEvidence {
    pub(crate) demand_source_acquisitions: u64,
    pub(crate) demand_ram_hits: Option<u64>,
    pub(crate) demand_ram_misses: Option<u64>,
    pub(crate) demand_ram_attribution_limitation: &'static str,
    pub(crate) engine_demand_nvme_operations: u64,
    pub(crate) demand_nvme_operations: u64,
    pub(crate) nvme_operation_attribution_semantics: &'static str,
    pub(crate) engine_nvme_bytes_including_speculation: u64,
    pub(crate) demand_nvme_bytes: u64,
    pub(crate) demand_h2d_installs: u64,
    pub(crate) demand_h2d_bytes: u64,
    pub(crate) speculative_ram_hits: u64,
    pub(crate) speculative_ram_misses: u64,
    pub(crate) speculative_nvme_operations: u64,
    pub(crate) speculative_nvme_bytes: u64,
    pub(crate) speculative_h2d_installs: u64,
    pub(crate) speculative_h2d_bytes: u64,
    pub(crate) cancellations_wasted_source_bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct LiveDerivedFractions {
    pub(crate) accepted_candidate_denominator: u64,
    pub(crate) completed_installation_denominator: u64,
    pub(crate) completed_before_first_demand_fraction: Option<f64>,
    pub(crate) demand_arrived_while_speculative_in_flight_fraction: Option<f64>,
    pub(crate) late_after_demand_fraction: Option<f64>,
    pub(crate) useful_installation_fraction: Option<f64>,
    pub(crate) unused_installation_fraction: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct QualificationResourceEvidence {
    pub(crate) production_predict_fanout: usize,
    pub(crate) qualification_shadow_slots: usize,
    pub(crate) effective_speculative_permits: usize,
    pub(crate) max_in_flight_source_acquisitions: usize,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LiveArmEvidence {
    pub(crate) arm: LiveArm,
    pub(crate) live_speculative_prefetch_enabled: bool,
    pub(crate) complete: bool,
    pub(crate) hardware: crate::backend::GpuDeviceIdentity,
    pub(crate) model_load: crate::greedy_parity::ModelLoadEvidence,
    pub(crate) runtime_contract: crate::gpu_native_real_benchmark::RuntimeContractEvidence,
    pub(crate) qualification_resources: QualificationResourceEvidence,
    pub(crate) warmup_run_evidence: Vec<LiveQualifiedRunEvidence>,
    pub(crate) measured_run_evidence: Vec<LiveQualifiedRunEvidence>,
    pub(crate) measured_aggregate: crate::gpu_native_real_benchmark::Aggregate,
    pub(crate) measured_live_totals: LiveLifecycleCounters,
    pub(crate) measured_live_fractions: LiveDerivedFractions,
    pub(crate) measured_source_attribution: DemandSourceEvidence,
    pub(crate) runtime_shutdown: crate::greedy_parity::BackgroundShutdownEvidence,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct BehavioralEquivalenceEvidence {
    pub(crate) pass: bool,
    pub(crate) compared_measured_requests: usize,
    pub(crate) identical_generated_token_ids: bool,
    pub(crate) identical_generated_token_counts: bool,
    pub(crate) identical_non_speculative_routed_counters: bool,
    pub(crate) identical_completed_position_counts: bool,
    pub(crate) deterministic_greedy_sampling: bool,
    pub(crate) no_full_token_replay: bool,
    pub(crate) no_cpu_expert_fallback: bool,
    pub(crate) no_degraded_expert_substitution: bool,
    pub(crate) no_fatal_or_no_progress_failure: bool,
    pub(crate) off_control_performed_zero_live_work: bool,
    pub(crate) mismatches: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct ResidencyEffectivenessEvidence {
    pub(crate) pass: bool,
    pub(crate) control_demand_physical_hits: u64,
    pub(crate) control_demand_physical_misses: u64,
    pub(crate) treatment_demand_physical_hits: u64,
    pub(crate) treatment_demand_physical_misses: u64,
    pub(crate) control_demand_miss_boundaries: u64,
    pub(crate) treatment_demand_miss_boundaries: u64,
    pub(crate) demand_physical_miss_reduction: i128,
    pub(crate) demand_physical_miss_reduction_fraction: Option<f64>,
    pub(crate) gross_demand_miss_boundary_reduction: i128,
    pub(crate) gross_demand_miss_boundary_reduction_fraction: Option<f64>,
    pub(crate) attributed_misses_introduced_by_speculative_eviction: u64,
    pub(crate) attributed_miss_boundaries_introduced_by_speculative_eviction: u64,
    pub(crate) net_demand_miss_boundary_reduction_after_attributed_harm: i128,
    pub(crate) control_speculative_state_churn_transitions: u128,
    pub(crate) treatment_speculative_state_churn_transitions: u128,
    pub(crate) speculative_state_churn_semantics: &'static str,
    pub(crate) formula: &'static str,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct CounterDeltaEvidence {
    pub(crate) control: u64,
    pub(crate) treatment: u64,
    pub(crate) treatment_minus_control: i128,
    pub(crate) relative_delta: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct DemandSourceComparisonEvidence {
    pub(crate) demand_source_acquisitions: CounterDeltaEvidence,
    pub(crate) demand_nvme_operations: CounterDeltaEvidence,
    pub(crate) demand_nvme_bytes: CounterDeltaEvidence,
    pub(crate) demand_h2d_installs: CounterDeltaEvidence,
    pub(crate) demand_h2d_bytes: CounterDeltaEvidence,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct PerformanceEvidence {
    pub(crate) pass: bool,
    pub(crate) eligible_for_interpretation: bool,
    pub(crate) behavioral_equivalence_pass: bool,
    pub(crate) live_path_exercised_pass: bool,
    pub(crate) control_decode_tps_mean: f64,
    pub(crate) treatment_decode_tps_mean: f64,
    pub(crate) decode_tps_relative_delta: Option<f64>,
    pub(crate) control_end_to_end_generated_tps_mean: f64,
    pub(crate) treatment_end_to_end_generated_tps_mean: f64,
    pub(crate) end_to_end_generated_tps_relative_delta: Option<f64>,
    pub(crate) control_request_wall_seconds_mean: f64,
    pub(crate) treatment_request_wall_seconds_mean: f64,
    pub(crate) request_wall_time_improvement_fraction: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct LivePathExercisedEvidence {
    pub(crate) pass: bool,
    pub(crate) nonresident_accepted_candidates: u64,
    pub(crate) source_acquisition_started: u64,
    pub(crate) speculative_source_joins: u64,
    pub(crate) physical_installation_started: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct LiveProductionSemantics {
    pub(crate) normal_runtime_default_enabled: bool,
    pub(crate) dedicated_qualification_command_only: bool,
    pub(crate) control_arm_enabled: bool,
    pub(crate) treatment_arm_enabled: bool,
    pub(crate) production_inference_math_changed: bool,
    pub(crate) production_q4_changed: bool,
    pub(crate) production_router_changed: bool,
    pub(crate) production_sampling_changed: bool,
    pub(crate) production_attention_or_kv_changed: bool,
    pub(crate) full_token_gpu_synchronization_added: bool,
    pub(crate) hidden_state_readback_added: bool,
    pub(crate) physical_source_of_truth_preserved: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct BoundedLivePrefetchReport {
    pub(crate) schema: &'static str,
    pub(crate) mode: &'static str,
    pub(crate) command: &'static str,
    pub(crate) source_main_commit: &'static str,
    pub(crate) tested_pr1bb_commit: &'static str,
    pub(crate) tested_pr1bb_report_sha256: &'static str,
    pub(crate) pr1ca_commit: &'static str,
    pub(crate) pr1ca_report_sha256: &'static str,
    pub(crate) pr1cb_commit: &'static str,
    pub(crate) pr1cb_report_sha256: &'static str,
    pub(crate) pr1cc_commit: &'static str,
    pub(crate) pr1cc_authoritative_report_sha256: Option<String>,
    pub(crate) external_pr1cc_policy_selection_pending: bool,
    pub(crate) policy_selection_source: &'static str,
    pub(crate) canonical_benchmark_schema_unchanged: &'static str,
    pub(crate) qualification_complete: bool,
    pub(crate) failure: Option<crate::gpu_native_real_benchmark::BenchmarkFailure>,
    pub(crate) qualification_pass: bool,
    pub(crate) behavioral_equivalence_pass: bool,
    pub(crate) live_path_exercised_pass: bool,
    pub(crate) residency_effectiveness_pass: bool,
    pub(crate) performance_pass: bool,
    pub(crate) performance_claim: bool,
    pub(crate) provenance: crate::gpu_native_real_benchmark::BenchmarkProvenance,
    pub(crate) model_identity: crate::greedy_parity::ModelIdentityEvidence,
    pub(crate) production_configuration: crate::gpu_native_real_benchmark::ProductionConfiguration,
    pub(crate) request: crate::gpu_native_real_benchmark::RequestEvidence,
    pub(crate) exact_cli_arguments: Vec<String>,
    pub(crate) config_path: String,
    pub(crate) resolved_config_sha256: String,
    pub(crate) predictor: &'static str,
    pub(crate) predictor_fanout: usize,
    pub(crate) predictor_retuned: bool,
    pub(crate) exact_causal_timing: &'static str,
    pub(crate) replacement_policy: GpuNativeLiveReplacementPolicy,
    pub(crate) live_resource_bounds: LiveResourceBounds,
    pub(crate) qualification_resources: QualificationResourceEvidence,
    pub(crate) off_on_resource_layout_identical: bool,
    pub(crate) replacement_policy_semantics: &'static str,
    pub(crate) normal_production_default_remains_disabled: bool,
    pub(crate) production_semantics: LiveProductionSemantics,
    pub(crate) cache_reset: crate::BenchRealCacheReset,
    pub(crate) warmup_runs: usize,
    pub(crate) measured_runs: usize,
    pub(crate) control: Option<LiveArmEvidence>,
    pub(crate) treatment: Option<LiveArmEvidence>,
    pub(crate) behavioral_equivalence: Option<BehavioralEquivalenceEvidence>,
    pub(crate) live_path_exercised: Option<LivePathExercisedEvidence>,
    pub(crate) residency_effectiveness: Option<ResidencyEffectivenessEvidence>,
    pub(crate) demand_source_comparison: Option<DemandSourceComparisonEvidence>,
    pub(crate) performance: Option<PerformanceEvidence>,
}

fn checked_add_counters(
    total: &mut LiveLifecycleCounters,
    value: &LiveLifecycleCounters,
) -> Result<(), crate::gpu_native_real_benchmark::BenchmarkFailure> {
    macro_rules! add {
        ($field:ident) => {
            total.$field = total.$field.checked_add(value.$field).ok_or_else(|| {
                crate::gpu_native_real_benchmark::BenchmarkFailure::new(
                    "postcondition",
                    "live-counter-overflow",
                    format!("live counter {} overflowed", stringify!($field)),
                )
            })?;
        };
    }
    add!(prediction_emitted);
    add!(rejected_live_disabled);
    add!(rejected_candidate_bound);
    add!(rejected_governor);
    add!(rejected_in_flight_bound);
    add!(rejected_install_bound);
    add!(rejected_replacement_bound);
    add!(already_physically_resident);
    add!(source_acquisition_started);
    add!(source_acquisition_deduplicated_joined);
    add!(speculative_joined_existing_acquisition);
    add!(demand_joined_speculative_acquisition);
    add!(source_acquisition_cancelled_stale);
    add!(rejected_concurrency_no_permit);
    add!(rejected_residency_pressure);
    add!(cancelled_boundary_expired);
    add!(cancelled_background_spawn);
    add!(cancelled_task_or_shutdown);
    add!(cancelled_stale_logical_generation);
    add!(physical_installation_started);
    add!(physical_installation_completed);
    add!(completed_before_first_demand);
    add!(demand_arrived_while_speculative_in_flight);
    add!(demand_reused_speculative_result);
    add!(prediction_useful);
    add!(prediction_useful_later);
    add!(prediction_unused);
    add!(speculative_replacements);
    add!(speculative_evictions);
    add!(evicted_expert_demanded_before_payoff);
    add!(misses_introduced_by_speculative_eviction);
    add!(miss_boundaries_introduced_by_speculative_eviction);
    add!(speculative_expert_evicted_unused);
    add!(prediction_protected_victim_skips);
    add!(forced_prediction_protected_evictions);
    add!(demand_independently_won_install_race);
    add!(candidate_already_ram_resident);
    add!(speculative_ram_hits);
    add!(speculative_ram_misses);
    add!(speculative_nvme_operations);
    add!(speculative_nvme_bytes);
    add!(speculative_h2d_installs);
    add!(speculative_h2d_bytes);
    add!(cancellations_wasted_source_bytes);
    add!(late_after_demand);
    add!(demand_physical_hits);
    add!(demand_physical_misses);
    add!(demand_miss_boundaries);
    add!(accepted_candidates);
    add!(completed_tasks);
    Ok(())
}

fn aggregate_live_runs(
    runs: &[LiveQualifiedRunEvidence],
) -> Result<LiveLifecycleCounters, crate::gpu_native_real_benchmark::BenchmarkFailure> {
    let mut total = LiveLifecycleCounters::default();
    for run in runs {
        checked_add_counters(&mut total, &run.live.counters)?;
    }
    Ok(total)
}

fn checked_sum<I>(
    values: I,
    field: &str,
) -> Result<u64, crate::gpu_native_real_benchmark::BenchmarkFailure>
where
    I: IntoIterator<Item = u64>,
{
    values.into_iter().try_fold(0u64, |total, value| {
        total.checked_add(value).ok_or_else(|| {
            crate::gpu_native_real_benchmark::BenchmarkFailure::new(
                "postcondition",
                "source-counter-overflow",
                format!("source counter {field} overflowed"),
            )
        })
    })
}

fn demand_source_evidence(
    runs: &[LiveQualifiedRunEvidence],
    live: &LiveLifecycleCounters,
) -> Result<DemandSourceEvidence, crate::gpu_native_real_benchmark::BenchmarkFailure> {
    use crate::gpu_native_real_benchmark::BenchmarkFailure;
    let demand_nvme_operations = checked_sum(
        runs.iter().map(|run| {
            run.production
                .counters
                .engine_storage_delta
                .nvme_read_operations
        }),
        "demand-nvme-operations",
    )?;
    let demand_source_acquisitions = checked_sum(
        runs.iter().map(|run| {
            run.production
                .counters
                .gpu_native_residency_delta
                .physical_source_acquisitions
        }),
        "demand-source-acquisitions",
    )?;
    let engine_nvme_bytes = checked_sum(
        runs.iter()
            .map(|run| run.production.counters.engine_storage_delta.nvme_bytes_read),
        "engine-nvme-bytes",
    )?;
    let total_h2d_installs = checked_sum(
        runs.iter().map(|run| {
            run.production
                .counters
                .gpu_expert_io_delta
                .expert_weight_uploads
        }),
        "total-h2d-installs",
    )?;
    let total_h2d_bytes = checked_sum(
        runs.iter().map(|run| {
            run.production
                .counters
                .gpu_expert_io_delta
                .expert_weight_upload_bytes
        }),
        "total-h2d-bytes",
    )?;
    let subtract = |total: u64, speculative: u64, field: &str| {
        total.checked_sub(speculative).ok_or_else(|| {
            BenchmarkFailure::new(
                "postcondition",
                "source-attribution-underflow",
                format!("{field}: total={total} speculative={speculative}"),
            )
        })
    };
    Ok(DemandSourceEvidence {
        demand_source_acquisitions,
        demand_ram_hits: None,
        demand_ram_misses: None,
        demand_ram_attribution_limitation: "the canonical GPU-native demand source helper does not increment the legacy Engine RAM hit/miss counters; v1 reports exact demand NVMe operations/bytes and exact speculative RAM hits/misses without fabricating demand RAM values",
        engine_demand_nvme_operations: demand_nvme_operations,
        demand_nvme_operations,
        nvme_operation_attribution_semantics: "the Engine I/O histogram counts fetch_once demand reads; the isolated live path counts each direct speculative storage read separately, so no operation-count subtraction is required",
        engine_nvme_bytes_including_speculation: engine_nvme_bytes,
        demand_nvme_bytes: subtract(
            engine_nvme_bytes,
            live.speculative_nvme_bytes,
            "NVMe bytes",
        )?,
        demand_h2d_installs: subtract(
            total_h2d_installs,
            live.speculative_h2d_installs,
            "H2D installs",
        )?,
        demand_h2d_bytes: subtract(
            total_h2d_bytes,
            live.speculative_h2d_bytes,
            "H2D bytes",
        )?,
        speculative_ram_hits: live.speculative_ram_hits,
        speculative_ram_misses: live.speculative_ram_misses,
        speculative_nvme_operations: live.speculative_nvme_operations,
        speculative_nvme_bytes: live.speculative_nvme_bytes,
        speculative_h2d_installs: live.speculative_h2d_installs,
        speculative_h2d_bytes: live.speculative_h2d_bytes,
        cancellations_wasted_source_bytes: live.cancellations_wasted_source_bytes,
    })
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn live_derived_fractions(counters: &LiveLifecycleCounters) -> LiveDerivedFractions {
    LiveDerivedFractions {
        accepted_candidate_denominator: counters.accepted_candidates,
        completed_installation_denominator: counters.physical_installation_completed,
        completed_before_first_demand_fraction: ratio(
            counters.completed_before_first_demand,
            counters.accepted_candidates,
        ),
        demand_arrived_while_speculative_in_flight_fraction: ratio(
            counters.demand_arrived_while_speculative_in_flight,
            counters.accepted_candidates,
        ),
        late_after_demand_fraction: ratio(counters.late_after_demand, counters.accepted_candidates),
        useful_installation_fraction: ratio(
            counters.prediction_useful,
            counters.physical_installation_completed,
        ),
        unused_installation_fraction: ratio(
            counters.prediction_unused,
            counters.physical_installation_completed,
        ),
    }
}

async fn execute_live_run(
    runtime: &crate::BenchRealRuntime,
    controller: &Arc<GpuNativeBoundedLivePrefetchController>,
    phase: ShadowPhase,
    run_index: usize,
    prompt_ids: &[u32],
    output_tokens: usize,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<LiveQualifiedRunEvidence, crate::gpu_native_real_benchmark::BenchmarkFailure> {
    use crate::gpu_native_real_benchmark::BenchmarkFailure;
    controller.begin_run(phase, run_index).map_err(|error| {
        BenchmarkFailure::new("live-controller", "begin-run-failed", error.to_string())
    })?;
    let phase_label = match phase {
        ShadowPhase::Warmup => "warmup",
        ShadowPhase::Measured => "measured",
    };
    let execution = crate::with_progress_timeout(
        format!("{COMMAND} {phase_label} run {run_index}"),
        watchdog,
        crate::gpu_native_real_benchmark::execute_request(
            runtime,
            prompt_ids,
            output_tokens,
            run_index,
        ),
    )
    .await
    .map_err(|error| {
        BenchmarkFailure::new(
            "inference",
            "bounded-live-request-failed",
            error.to_string(),
        )
    });
    let token_loop = runtime
        .gpu_native_token_loop
        .as_ref()
        .expect("validated bounded-live runtime has a token loop");
    let finalized = controller
        .end_run(
            runtime.engine.as_ref(),
            token_loop.residency_manager().as_ref(),
        )
        .await
        .map_err(|error| {
            BenchmarkFailure::new(
                "postcondition",
                "bounded-live-finalization-failed",
                error.to_string(),
            )
        });
    match (execution, finalized) {
        (Ok(production), Ok(live)) => Ok(LiveQualifiedRunEvidence { production, live }),
        (Err(error), Ok(_)) | (Ok(_), Err(error)) => Err(error),
        (Err(execution), Err(finalization)) => Err(BenchmarkFailure::new(
            "postcondition",
            "execution-and-live-finalization-failed",
            format!("{execution}; {finalization}"),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_arm(
    spec: &crate::ResolvedRealCliSpec,
    resolved_config_sha256: &str,
    expected_adapter_name: &str,
    arm: LiveArm,
    policy: GpuNativeLiveReplacementPolicy,
    bounds: LiveResourceBounds,
    markov_min_prob: f64,
    prompt_ids: &[u32],
    output_tokens: usize,
    warmup_runs: usize,
    measured_runs: usize,
    watchdog: crate::rayon_autotune::ProgressWatchdogConfig,
) -> Result<LiveArmEvidence, crate::gpu_native_real_benchmark::BenchmarkFailure> {
    use crate::gpu_native_real_benchmark::BenchmarkFailure;
    let tokenizer = crate::load_real_cli_tokenizer(
        &spec.cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBoundedLivePrefetch,
    )
    .map_err(|error| {
        BenchmarkFailure::new("startup", "tokenizer-load-failed", error.to_string())
    })?;
    let runtime = crate::build_isolated_greedy_runtime(
        spec,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBoundedLivePrefetch,
        tokenizer,
    )
    .await
    .map_err(|error| {
        BenchmarkFailure::new("startup", "runtime-construction-failed", error.to_string())
    })?;
    let (qualification_shadow_slots, effective_speculative_permits) = runtime
        .engine
        .bounded_live_speculative_resource_layout();
    let qualification_resources = QualificationResourceEvidence {
        production_predict_fanout: runtime.cfg.storage.predict_fanout,
        qualification_shadow_slots,
        effective_speculative_permits,
        max_in_flight_source_acquisitions: bounds.max_in_flight_source_acquisitions,
    };
    if runtime.cfg.storage.predict_fanout != spec.cfg.storage.predict_fanout
        || qualification_shadow_slots != bounds.max_in_flight_source_acquisitions
        || effective_speculative_permits < 1
        || effective_speculative_permits != bounds.max_in_flight_source_acquisitions
    {
        let failure = BenchmarkFailure::new(
            "startup",
            "qualification-resource-contract-mismatch",
            format!(
                "source_predict_fanout={} runtime_predict_fanout={} resources={qualification_resources:?}",
                spec.cfg.storage.predict_fanout, runtime.cfg.storage.predict_fanout
            ),
        );
        return match runtime.shutdown_isolated().await {
            Ok(_) => Err(failure),
            Err(shutdown) => Err(BenchmarkFailure::new(
                "postcondition",
                "resource-contract-and-shutdown-failed",
                format!("{failure}; {shutdown}"),
            )),
        };
    }
    let validation = crate::gpu_native_prefetch_shadow::validate_shadow_runtime(
        &runtime,
        resolved_config_sha256,
        expected_adapter_name,
    );
    let (runtime_contract, hardware, model_load) = match validation {
        Ok(evidence) => evidence,
        Err(error) => {
            let shutdown = runtime.shutdown_isolated().await;
            return match shutdown {
                Ok(_) => Err(error),
                Err(shutdown_error) => Err(BenchmarkFailure::new(
                    "postcondition",
                    "startup-and-shutdown-failed",
                    format!("{error}; {shutdown_error}"),
                )),
            };
        }
    };
    let token_loop = runtime
        .gpu_native_token_loop
        .as_ref()
        .expect("validated bounded-live runtime has a token loop");
    let geometry = token_loop.model_geometry();
    let controller = GpuNativeBoundedLivePrefetchController::new(LiveControllerConfig {
        arm,
        policy,
        num_layers: geometry.num_layers,
        experts_per_layer: geometry.num_experts,
        top_k: geometry.top_k,
        markov_min_prob,
        bounds,
    });
    let controller = match controller {
        Ok(controller) => controller,
        Err(error) => {
            let failure = BenchmarkFailure::new(
                "startup",
                "live-controller-construction-failed",
                error.to_string(),
            );
            return match runtime.shutdown_isolated().await {
                Ok(_) => Err(failure),
                Err(shutdown) => Err(BenchmarkFailure::new(
                    "postcondition",
                    "controller-construction-and-shutdown-failed",
                    format!("{failure}; {shutdown}"),
                )),
            };
        }
    };
    if let Err(error) = token_loop.install_bounded_live_prefetch_controller(controller.clone()) {
        let failure = BenchmarkFailure::new(
            "startup",
            "live-controller-install-failed",
            error.to_string(),
        );
        drop(controller);
        return match runtime.shutdown_isolated().await {
            Ok(_) => Err(failure),
            Err(shutdown) => Err(BenchmarkFailure::new(
                "postcondition",
                "controller-install-and-shutdown-failed",
                format!("{failure}; {shutdown}"),
            )),
        };
    }

    let execution = async {
        let mut warmup = Vec::with_capacity(warmup_runs);
        for run_index in 0..warmup_runs {
            warmup.push(
                execute_live_run(
                    &runtime,
                    &controller,
                    ShadowPhase::Warmup,
                    run_index,
                    prompt_ids,
                    output_tokens,
                    watchdog,
                )
                .await?,
            );
        }
        let mut measured = Vec::with_capacity(measured_runs);
        for run_index in 0..measured_runs {
            measured.push(
                execute_live_run(
                    &runtime,
                    &controller,
                    ShadowPhase::Measured,
                    run_index,
                    prompt_ids,
                    output_tokens,
                    watchdog,
                )
                .await?,
            );
        }
        Ok::<_, BenchmarkFailure>((warmup, measured))
    }
    .await;
    drop(controller);
    let shutdown = runtime.shutdown_isolated().await.map_err(|error| {
        BenchmarkFailure::new(
            "postcondition",
            "runtime-shutdown-failed",
            error.to_string(),
        )
    });
    let ((warmup_run_evidence, measured_run_evidence), runtime_shutdown) =
        match (execution, shutdown) {
            (Ok(runs), Ok(shutdown)) => (runs, shutdown),
            (Err(error), Ok(_)) | (Ok(_), Err(error)) => return Err(error),
            (Err(execution), Err(shutdown)) => {
                return Err(BenchmarkFailure::new(
                    "postcondition",
                    "execution-and-runtime-shutdown-failed",
                    format!("{execution}; {shutdown}"),
                ));
            }
        };
    if warmup_run_evidence.len() != warmup_runs || measured_run_evidence.len() != measured_runs {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "incomplete-arm-run-set",
            format!(
                "arm={arm:?} retained warmup={}/{} measured={}/{}",
                warmup_run_evidence.len(),
                warmup_runs,
                measured_run_evidence.len(),
                measured_runs
            ),
        ));
    }
    let measured_production = measured_run_evidence
        .iter()
        .map(|run| run.production.clone())
        .collect::<Vec<_>>();
    let measured_aggregate = crate::gpu_native_real_benchmark::aggregate(&measured_production)?;
    let measured_live_totals = aggregate_live_runs(&measured_run_evidence)?;
    let measured_live_fractions = live_derived_fractions(&measured_live_totals);
    let measured_source_attribution =
        demand_source_evidence(&measured_run_evidence, &measured_live_totals)?;
    if arm == LiveArm::Off
        && (measured_live_totals.accepted_candidates != 0
            || measured_live_totals.source_acquisition_started != 0
            || measured_live_totals.physical_installation_started != 0
            || measured_live_totals.speculative_nvme_bytes != 0
            || measured_live_totals.speculative_h2d_bytes != 0)
    {
        return Err(BenchmarkFailure::new(
            "postcondition",
            "off-control-performed-live-work",
            format!("OFF control live counters were nonzero: {measured_live_totals:?}"),
        ));
    }
    Ok(LiveArmEvidence {
        arm,
        live_speculative_prefetch_enabled: arm == LiveArm::On,
        complete: true,
        hardware,
        model_load,
        runtime_contract,
        qualification_resources,
        warmup_run_evidence,
        measured_run_evidence,
        measured_aggregate,
        measured_live_totals,
        measured_live_fractions,
        measured_source_attribution,
        runtime_shutdown,
    })
}

fn compare_behavior(
    control: &LiveArmEvidence,
    treatment: &LiveArmEvidence,
) -> BehavioralEquivalenceEvidence {
    let mut evidence = BehavioralEquivalenceEvidence {
        compared_measured_requests: control.measured_run_evidence.len(),
        identical_generated_token_ids: true,
        identical_generated_token_counts: true,
        identical_non_speculative_routed_counters: true,
        identical_completed_position_counts: true,
        deterministic_greedy_sampling: true,
        no_full_token_replay: true,
        no_cpu_expert_fallback: true,
        no_degraded_expert_substitution: true,
        no_fatal_or_no_progress_failure: true,
        off_control_performed_zero_live_work: control.measured_live_totals.accepted_candidates == 0
            && control.measured_live_totals.source_acquisition_started == 0
            && control.measured_live_totals.physical_installation_started == 0,
        ..BehavioralEquivalenceEvidence::default()
    };
    if control.measured_run_evidence.len() != treatment.measured_run_evidence.len() {
        evidence.mismatches.push(format!(
            "measured request counts differ: control={} treatment={}",
            control.measured_run_evidence.len(),
            treatment.measured_run_evidence.len()
        ));
    }
    for (index, (off, on)) in control
        .measured_run_evidence
        .iter()
        .zip(&treatment.measured_run_evidence)
        .enumerate()
    {
        let off_result = &off.production;
        let on_result = &on.production;
        if off_result.generated_token_ids != on_result.generated_token_ids {
            evidence.identical_generated_token_ids = false;
            evidence.mismatches.push(format!(
                "measured run {index} generated token IDs differ: off_sha={} on_sha={}",
                off_result.generated_token_ids_sha256, on_result.generated_token_ids_sha256
            ));
        }
        if off_result.generated_tokens != on_result.generated_tokens {
            evidence.identical_generated_token_counts = false;
            evidence.mismatches.push(format!(
                "measured run {index} generated token counts differ: off={} on={}",
                off_result.generated_tokens, on_result.generated_tokens
            ));
        }
        let off_routed = off_result.counters.routed_execution_delta;
        let on_routed = on_result.counters.routed_execution_delta;
        if off_routed != on_routed {
            evidence.identical_non_speculative_routed_counters = false;
            evidence.mismatches.push(format!(
                "measured run {index} routed execution counters differ: off={off_routed:?} on={on_routed:?}"
            ));
        }
        if off_result.counters.token_loop_delta.tokens_completed
            != on_result.counters.token_loop_delta.tokens_completed
        {
            evidence.identical_completed_position_counts = false;
            evidence.mismatches.push(format!(
                "measured run {index} completed-position counts differ"
            ));
        }
        for result in [off_result, on_result] {
            let recovery = result.counters.recovery_delta;
            let routed = result.counters.routed_execution_delta;
            let token = result.counters.token_loop_delta;
            evidence.no_full_token_replay &= recovery.full_token_replay_attempts == 0;
            evidence.no_cpu_expert_fallback &=
                routed.cpu_routed_expert_dispatches == 0 && routed.gpu_cpu_fallbacks == 0;
            evidence.no_degraded_expert_substitution &= routed.degraded_expert_substitutions == 0;
            evidence.no_fatal_or_no_progress_failure &=
                token.fatal_failures == 0 && token.no_progress_failures == 0;
        }
    }
    evidence.pass = control.measured_run_evidence.len() == treatment.measured_run_evidence.len()
        && evidence.identical_generated_token_ids
        && evidence.identical_generated_token_counts
        && evidence.identical_non_speculative_routed_counters
        && evidence.identical_completed_position_counts
        && evidence.deterministic_greedy_sampling
        && evidence.no_full_token_replay
        && evidence.no_cpu_expert_fallback
        && evidence.no_degraded_expert_substitution
        && evidence.no_fatal_or_no_progress_failure
        && evidence.off_control_performed_zero_live_work;
    evidence
}

fn compare_residency(
    control: &LiveArmEvidence,
    treatment: &LiveArmEvidence,
) -> ResidencyEffectivenessEvidence {
    let off = &control.measured_live_totals;
    let on = &treatment.measured_live_totals;
    let demand_physical_miss_reduction =
        off.demand_physical_misses as i128 - on.demand_physical_misses as i128;
    let gross = off.demand_miss_boundaries as i128 - on.demand_miss_boundaries as i128;
    let net = gross - on.miss_boundaries_introduced_by_speculative_eviction as i128;
    ResidencyEffectivenessEvidence {
        pass: net > 0,
        control_demand_physical_hits: off.demand_physical_hits,
        control_demand_physical_misses: off.demand_physical_misses,
        treatment_demand_physical_hits: on.demand_physical_hits,
        treatment_demand_physical_misses: on.demand_physical_misses,
        control_demand_miss_boundaries: off.demand_miss_boundaries,
        treatment_demand_miss_boundaries: on.demand_miss_boundaries,
        demand_physical_miss_reduction,
        demand_physical_miss_reduction_fraction: relative_reduction_u64(
            off.demand_physical_misses,
            on.demand_physical_misses,
        ),
        gross_demand_miss_boundary_reduction: gross,
        gross_demand_miss_boundary_reduction_fraction: relative_reduction_u64(
            off.demand_miss_boundaries,
            on.demand_miss_boundaries,
        ),
        attributed_misses_introduced_by_speculative_eviction: on
            .misses_introduced_by_speculative_eviction,
        attributed_miss_boundaries_introduced_by_speculative_eviction: on
            .miss_boundaries_introduced_by_speculative_eviction,
        net_demand_miss_boundary_reduction_after_attributed_harm: net,
        control_speculative_state_churn_transitions: off.physical_installation_completed as u128
            + off.speculative_evictions as u128,
        treatment_speculative_state_churn_transitions: on.physical_installation_completed as u128
            + on.speculative_evictions as u128,
        speculative_state_churn_semantics: "physical speculative installations plus physical speculative evictions; a replacement contributes two physical state transitions",
        formula: "control demand miss boundaries - treatment demand miss boundaries - treatment attributed miss boundaries introduced by speculative eviction",
    }
}

fn relative_delta(control: f64, treatment: f64) -> Option<f64> {
    (control.is_finite() && treatment.is_finite() && control > 0.0)
        .then_some((treatment - control) / control)
}

fn relative_reduction_u64(control: u64, treatment: u64) -> Option<f64> {
    (control > 0).then_some((control as f64 - treatment as f64) / control as f64)
}

fn counter_delta(control: u64, treatment: u64) -> CounterDeltaEvidence {
    CounterDeltaEvidence {
        control,
        treatment,
        treatment_minus_control: treatment as i128 - control as i128,
        relative_delta: relative_delta(control as f64, treatment as f64),
    }
}

fn compare_demand_sources(
    control: &LiveArmEvidence,
    treatment: &LiveArmEvidence,
) -> DemandSourceComparisonEvidence {
    let off = &control.measured_source_attribution;
    let on = &treatment.measured_source_attribution;
    DemandSourceComparisonEvidence {
        demand_source_acquisitions: counter_delta(
            off.demand_source_acquisitions,
            on.demand_source_acquisitions,
        ),
        demand_nvme_operations: counter_delta(
            off.demand_nvme_operations,
            on.demand_nvme_operations,
        ),
        demand_nvme_bytes: counter_delta(off.demand_nvme_bytes, on.demand_nvme_bytes),
        demand_h2d_installs: counter_delta(off.demand_h2d_installs, on.demand_h2d_installs),
        demand_h2d_bytes: counter_delta(off.demand_h2d_bytes, on.demand_h2d_bytes),
    }
}

fn mean_wall_seconds(arm: &LiveArmEvidence) -> f64 {
    arm.measured_run_evidence
        .iter()
        .map(|run| run.production.timing.end_to_end_seconds)
        .sum::<f64>()
        / arm.measured_run_evidence.len() as f64
}

fn live_path_exercised(counters: &LiveLifecycleCounters) -> LivePathExercisedEvidence {
    let nonresident_accepted_candidates = counters
        .accepted_candidates
        .saturating_sub(counters.already_physically_resident);
    let speculative_source_joins = counters.speculative_joined_existing_acquisition;
    let progressed = counters.source_acquisition_started > 0
        || speculative_source_joins > 0
        || counters.physical_installation_started > 0;
    LivePathExercisedEvidence {
        pass: nonresident_accepted_candidates > 0 && progressed,
        nonresident_accepted_candidates,
        source_acquisition_started: counters.source_acquisition_started,
        speculative_source_joins,
        physical_installation_started: counters.physical_installation_started,
    }
}

fn performance_eligible(behavioral_pass: bool, live_path_exercised_pass: bool) -> bool {
    behavioral_pass && live_path_exercised_pass
}

fn compare_performance(
    control: &LiveArmEvidence,
    treatment: &LiveArmEvidence,
    behavioral_pass: bool,
    live_path_exercised_pass: bool,
) -> PerformanceEvidence {
    let off_decode = control.measured_aggregate.decode_tps.mean;
    let on_decode = treatment.measured_aggregate.decode_tps.mean;
    let off_e2e = control.measured_aggregate.end_to_end_generated_tps.mean;
    let on_e2e = treatment.measured_aggregate.end_to_end_generated_tps.mean;
    let off_wall = mean_wall_seconds(control);
    let on_wall = mean_wall_seconds(treatment);
    let eligible_for_interpretation =
        performance_eligible(behavioral_pass, live_path_exercised_pass);
    PerformanceEvidence {
        pass: eligible_for_interpretation
            && on_decode > off_decode
            && on_e2e > off_e2e
            && on_wall < off_wall,
        eligible_for_interpretation,
        behavioral_equivalence_pass: behavioral_pass,
        live_path_exercised_pass,
        control_decode_tps_mean: off_decode,
        treatment_decode_tps_mean: on_decode,
        decode_tps_relative_delta: relative_delta(off_decode, on_decode),
        control_end_to_end_generated_tps_mean: off_e2e,
        treatment_end_to_end_generated_tps_mean: on_e2e,
        end_to_end_generated_tps_relative_delta: relative_delta(off_e2e, on_e2e),
        control_request_wall_seconds_mean: off_wall,
        treatment_request_wall_seconds_mean: on_wall,
        request_wall_time_improvement_fraction: (off_wall.is_finite()
            && on_wall.is_finite()
            && off_wall > 0.0)
            .then_some((off_wall - on_wall) / off_wall),
    }
}

fn replacement_semantics(policy: GpuNativeLiveReplacementPolicy) -> &'static str {
    match policy {
        GpuNativeLiveReplacementPolicy::PhysicalLru => {
            "select the current physical LRU tail at install time"
        }
        GpuNativeLiveReplacementPolicy::PhysicalLruPredictionProtected => {
            "select the current physical LRU tail excluding the frozen causal prediction set; if every victim is protected, deterministically force the physical LRU tail"
        }
        GpuNativeLiveReplacementPolicy::RouteRecency => {
            "select least recently seen in completed actual routed sets, with current physical LRU order then ascending global ID as deterministic ties"
        }
        GpuNativeLiveReplacementPolicy::RouteRecencyPredictionProtected => {
            "select least recently seen in completed actual routed sets excluding the frozen causal prediction set; if every victim is protected, deterministically force route-recency order with physical-LRU/global-ID ties"
        }
    }
}

fn resolve_predict_min_prob(configured: f64, num_experts: u32) -> f64 {
    if configured > 0.0 {
        configured
    } else {
        2.0 / num_experts.max(1) as f64
    }
}

fn validate_hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn emit_report(
    report: &BoundedLivePrefetchReport,
    path: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write as _;
    let mut bytes = serde_json::to_vec_pretty(report)?;
    bytes.push(b'\n');
    if let Some(path) = path {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bytes)?;
        eprintln!(
            "GPU-native bounded live-prefetch report written to {}",
            path.display()
        );
    } else {
        std::io::stdout().write_all(&bytes)?;
    }
    Ok(())
}

pub(crate) async fn run_command(args: CommandArgs) -> Result<(), Box<dyn std::error::Error>> {
    use crate::gpu_native_real_benchmark::{BenchmarkFailure, BenchmarkProvenance};

    if !args.greedy {
        return Err(BenchmarkFailure::new(
            "preflight",
            "greedy-required",
            format!("{COMMAND} requires the explicit --greedy flag"),
        )
        .into());
    }
    if args.measured_runs == 0 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "measured-runs-required",
            format!("{COMMAND} requires --measured-runs > 0"),
        )
        .into());
    }
    if args.cache_reset != crate::BenchRealCacheReset::Keep {
        return Err(BenchmarkFailure::new(
            "preflight",
            "cache-reset-contract",
            "bounded live-prefetch schema v1 supports only the frozen --cache-reset keep schedule",
        )
        .into());
    }
    if args.expected_adapter_name != FROZEN_ADAPTER_NAME {
        return Err(BenchmarkFailure::new(
            "preflight",
            "frozen-adapter-required",
            format!(
                "bounded live-prefetch schema v1 requires --expected-adapter-name {FROZEN_ADAPTER_NAME:?}; observed {:?}",
                args.expected_adapter_name
            ),
        )
        .into());
    }
    let request_input = crate::load_real_cli_request_input(
        COMMAND,
        args.prompt.as_ref(),
        args.request_json.as_deref(),
        args.output_tokens,
    )?;
    if request_input.output_tokens < 2 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "insufficient-output-tokens",
            format!("{COMMAND} requires --output-tokens >= 2"),
        )
        .into());
    }

    let build = crate::qualification::BuildProvenance::embedded();
    let cfg = crate::config::Config::from_file(&args.config)?;
    crate::gpu_native_real_benchmark::validate_source_config(&cfg)?;
    if cfg.storage.predict_fanout != 0 {
        return Err(BenchmarkFailure::new(
            "preflight",
            "production-prefetch-must-remain-disabled",
            format!(
                "{COMMAND} requires source storage.predict_fanout=0; observed {}",
                cfg.storage.predict_fanout
            ),
        )
        .into());
    }
    let (artifacts, artifact_errors) = crate::qualification_artifacts(&args.config, &cfg);
    let expert_metadata =
        crate::qualification::read_expert_metadata(&cfg.model.data_dir.join("metadata.json"))
            .map_err(|error| {
                BenchmarkFailure::new("preflight", "expert-metadata-unavailable", error)
            })?;
    crate::gpu_native_prefetch_shadow::validate_shadow_preflight(
        &build,
        &artifacts,
        &artifact_errors,
        &expert_metadata,
    )?;
    let total_experts = (cfg.model.num_layers as u32)
        .checked_mul(cfg.model.num_experts)
        .ok_or_else(|| {
            BenchmarkFailure::new(
                "preflight",
                "expert-namespace-overflow",
                "bounded live-prefetch expert namespace overflowed",
            )
        })?;
    let markov_min_prob = resolve_predict_min_prob(cfg.storage.predict_min_prob, total_experts);
    let spec = crate::resolve_real_cli_spec_from_config(
        cfg,
        crate::RealCliRuntimeMode::IsolatedGpuNativeBoundedLivePrefetch,
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
        crate::RealCliRuntimeMode::IsolatedGpuNativeBoundedLivePrefetch,
    )?;
    let prompt_ids = tokenizer.encode(&request_input.prompt)?;
    if prompt_ids.is_empty() {
        return Err(BenchmarkFailure::new(
            "preflight",
            "empty-prompt-tokenization",
            "prompt encoded to zero tokens",
        )
        .into());
    }
    drop(tokenizer);
    let (executable, executable_sha256) = crate::current_executable_identity()?;
    let executable_canonical_path = std::fs::canonicalize(&executable)?.display().to_string();
    if !validate_hex(&executable_sha256, 64) || !validate_hex(&resolved_config_sha256, 64) {
        return Err(BenchmarkFailure::new(
            "preflight",
            "provenance-unavailable",
            "executable or resolved-config SHA256 was unavailable",
        )
        .into());
    }
    let production_configuration =
        crate::gpu_native_real_benchmark::ProductionConfiguration::from_config(
            &spec.cfg,
            &expert_metadata,
        );
    let request = crate::gpu_native_real_benchmark::RequestEvidence {
        prompt_sha256: crate::greedy_parity::sha256_hex(request_input.prompt.as_bytes()),
        prompt_token_ids_sha256: crate::greedy_parity::token_ids_sha256(&prompt_ids),
        prompt_token_count: prompt_ids.len(),
        requested_output_tokens: request_input.output_tokens,
        greedy: true,
    };
    let bounds = LiveResourceBounds::default();
    let qualification_resources = QualificationResourceEvidence {
        production_predict_fanout: spec.cfg.storage.predict_fanout,
        qualification_shadow_slots: bounds.max_in_flight_source_acquisitions,
        effective_speculative_permits: bounds.max_in_flight_source_acquisitions,
        max_in_flight_source_acquisitions: bounds.max_in_flight_source_acquisitions,
    };
    let mut report = BoundedLivePrefetchReport {
        schema: SCHEMA,
        mode: MODE,
        command: COMMAND,
        source_main_commit: SOURCE_MAIN_COMMIT,
        tested_pr1bb_commit: TESTED_PR1BB_COMMIT,
        tested_pr1bb_report_sha256: TESTED_PR1BB_REPORT_SHA256,
        pr1ca_commit: PR1CA_COMMIT,
        pr1ca_report_sha256: PR1CA_REPORT_SHA256,
        pr1cb_commit: PR1CB_COMMIT,
        pr1cb_report_sha256: PR1CB_REPORT_SHA256,
        pr1cc_commit: PR1CC_COMMIT,
        pr1cc_authoritative_report_sha256: None,
        external_pr1cc_policy_selection_pending: true,
        policy_selection_source: "required-explicit-cli-selection-no-authoritative-pr1cc-winner-artifact-available",
        canonical_benchmark_schema_unchanged: crate::gpu_native_real_benchmark::SCHEMA,
        qualification_complete: false,
        failure: None,
        qualification_pass: false,
        behavioral_equivalence_pass: false,
        live_path_exercised_pass: false,
        residency_effectiveness_pass: false,
        performance_pass: false,
        performance_claim: false,
        provenance: BenchmarkProvenance {
            build,
            executable_canonical_path,
            executable_sha256,
            resolved_config_sha256: resolved_config_sha256.clone(),
            artifacts,
            expert_metadata,
        },
        model_identity,
        production_configuration,
        request,
        exact_cli_arguments: std::env::args_os()
            .skip(1)
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect(),
        config_path: args.config.display().to_string(),
        resolved_config_sha256: resolved_config_sha256.clone(),
        predictor: FROZEN_PREDICTOR,
        predictor_fanout: FROZEN_FANOUT,
        predictor_retuned: false,
        exact_causal_timing: "freeze after completed actual route L-1 is CPU-visible and all earlier target truth has been scored, before target layer L submission; update predictor and route-recency only after target scoring",
        replacement_policy: args.replacement_policy,
        live_resource_bounds: bounds,
        qualification_resources,
        off_on_resource_layout_identical: false,
        replacement_policy_semantics: replacement_semantics(args.replacement_policy),
        normal_production_default_remains_disabled: true,
        production_semantics: LiveProductionSemantics {
            normal_runtime_default_enabled: false,
            dedicated_qualification_command_only: true,
            control_arm_enabled: false,
            treatment_arm_enabled: true,
            production_inference_math_changed: false,
            production_q4_changed: false,
            production_router_changed: false,
            production_sampling_changed: false,
            production_attention_or_kv_changed: false,
            full_token_gpu_synchronization_added: false,
            hidden_state_readback_added: false,
            physical_source_of_truth_preserved: true,
        },
        cache_reset: args.cache_reset,
        warmup_runs: args.warmup_runs,
        measured_runs: args.measured_runs,
        control: None,
        treatment: None,
        behavioral_equivalence: None,
        live_path_exercised: None,
        residency_effectiveness: None,
        demand_source_comparison: None,
        performance: None,
    };

    let control = run_arm(
        &spec,
        &resolved_config_sha256,
        &args.expected_adapter_name,
        LiveArm::Off,
        args.replacement_policy,
        bounds,
        markov_min_prob,
        &prompt_ids,
        request_input.output_tokens,
        args.warmup_runs,
        args.measured_runs,
        args.progress_watchdog,
    )
    .await;
    let control = match control {
        Ok(control) => {
            report.control = Some(control.clone());
            control
        }
        Err(failure) => {
            report.failure = Some(failure.clone());
            emit_report(&report, args.report_out.as_deref())?;
            return Err(failure.into());
        }
    };
    let treatment = run_arm(
        &spec,
        &resolved_config_sha256,
        &args.expected_adapter_name,
        LiveArm::On,
        args.replacement_policy,
        bounds,
        markov_min_prob,
        &prompt_ids,
        request_input.output_tokens,
        args.warmup_runs,
        args.measured_runs,
        args.progress_watchdog,
    )
    .await;
    let treatment = match treatment {
        Ok(treatment) => {
            report.treatment = Some(treatment.clone());
            treatment
        }
        Err(failure) => {
            report.failure = Some(failure.clone());
            emit_report(&report, args.report_out.as_deref())?;
            return Err(failure.into());
        }
    };
    if control.hardware != treatment.hardware
        || control.model_load != treatment.model_load
        || control.runtime_contract.token_loop_geometry
            != treatment.runtime_contract.token_loop_geometry
        || control.qualification_resources != treatment.qualification_resources
    {
        let failure = BenchmarkFailure::new(
            "postcondition",
            "off-on-runtime-identity-drift",
            format!(
                "OFF/ON runtime identity differed: off_hardware={:?} on_hardware={:?} off_model={:?} on_model={:?} off_resources={:?} on_resources={:?}",
                control.hardware,
                treatment.hardware,
                control.model_load,
                treatment.model_load,
                control.qualification_resources,
                treatment.qualification_resources
            ),
        );
        report.failure = Some(failure.clone());
        emit_report(&report, args.report_out.as_deref())?;
        return Err(failure.into());
    }
    report.qualification_resources = control.qualification_resources;
    report.off_on_resource_layout_identical = true;
    let behavioral = compare_behavior(&control, &treatment);
    let live_path = live_path_exercised(&treatment.measured_live_totals);
    let residency = compare_residency(&control, &treatment);
    let demand_source_comparison = compare_demand_sources(&control, &treatment);
    let performance = compare_performance(&control, &treatment, behavioral.pass, live_path.pass);
    report.behavioral_equivalence_pass = behavioral.pass;
    report.live_path_exercised_pass = live_path.pass;
    report.residency_effectiveness_pass = residency.pass;
    report.performance_pass = performance.pass;
    report.performance_claim = performance.pass;
    report.qualification_complete = true;
    report.qualification_pass = control.complete
        && treatment.complete
        && behavioral.pass
        && live_path.pass
        && control.runtime_shutdown.all_runtime_resources_released
        && treatment.runtime_shutdown.all_runtime_resources_released;
    report.behavioral_equivalence = Some(behavioral);
    report.live_path_exercised = Some(live_path);
    report.residency_effectiveness = Some(residency);
    report.demand_source_comparison = Some(demand_source_comparison);
    report.performance = Some(performance);
    emit_report(&report, args.report_out.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;

    fn controller() -> Arc<GpuNativeBoundedLivePrefetchController> {
        controller_for_arm(LiveArm::On)
    }

    fn controller_for_arm(arm: LiveArm) -> Arc<GpuNativeBoundedLivePrefetchController> {
        GpuNativeBoundedLivePrefetchController::new(LiveControllerConfig {
            arm,
            policy: GpuNativeLiveReplacementPolicy::PhysicalLru,
            num_layers: 2,
            experts_per_layer: 4,
            top_k: 1,
            markov_min_prob: 0.0,
            bounds: LiveResourceBounds::default(),
        })
        .unwrap()
    }

    fn pending(global_id: u32) -> PendingTarget {
        PendingTarget {
            completed_token_position: 0,
            target_layer: 1,
            boundary: 0,
            prediction_at_us: 0,
            target_probe_at_us: Some(100),
            candidates: vec![FrozenCandidate {
                global_id,
                score: 1.0,
            }],
        }
    }

    fn accept(
        controller: &GpuNativeBoundedLivePrefetchController,
        pending: &PendingTarget,
    ) -> LiveCandidateTicket {
        controller
            .accept_candidate(
                pending,
                &pending.candidates[0],
                Arc::new(HashSet::from([pending.candidates[0].global_id])),
            )
            .unwrap()
            .unwrap()
    }

    fn classify(
        controller: &GpuNativeBoundedLivePrefetchController,
        pending: &PendingTarget,
        selected: HashSet<u32>,
        demand_at: u64,
        current: bool,
    ) {
        let mut inner = controller.inner.lock();
        let run = inner.run.as_mut().unwrap();
        classify_candidate_demand(run, pending, &selected, demand_at, |_| Ok(current)).unwrap();
    }

    fn counters(controller: &GpuNativeBoundedLivePrefetchController) -> LiveLifecycleCounters {
        controller
            .inner
            .lock()
            .run
            .as_ref()
            .unwrap()
            .counters
            .clone()
    }

    #[test]
    fn live_demand_arrives_before_speculative_source_acquisition_starts() {
        let controller = controller();
        controller.begin_run(ShadowPhase::Measured, 0).unwrap();
        let pending = pending(4);
        let _ticket = accept(&controller, &pending);
        classify(&controller, &pending, HashSet::from([4]), 100, false);
        let counters = counters(&controller);
        assert_eq!(counters.source_acquisition_started, 0);
        assert_eq!(counters.demand_arrived_while_speculative_in_flight, 1);
    }

    #[test]
    fn live_demand_joins_in_flight_speculative_acquisition() {
        let controller = controller();
        controller.begin_run(ShadowPhase::Measured, 0).unwrap();
        let pending = pending(4);
        let ticket = accept(&controller, &pending);
        controller.record_source_started(ticket.ticket_id, false, false);
        controller.record_demand_joined_source(4);
        classify(&controller, &pending, HashSet::from([4]), 100, false);
        assert!(controller.reserve_install(ticket.ticket_id, false));
        controller.record_installed(ticket.ticket_id, 4096, None, 0, false);
        controller.record_demand_service_completed(&[4]);
        let counters = counters(&controller);
        assert_eq!(counters.source_acquisition_deduplicated_joined, 1);
        assert_eq!(counters.demand_joined_speculative_acquisition, 1);
        assert_eq!(counters.speculative_joined_existing_acquisition, 0);
        assert_eq!(counters.demand_arrived_while_speculative_in_flight, 1);
        assert_eq!(counters.demand_reused_speculative_result, 1);
        assert_eq!(counters.prediction_useful, 1);
        assert_eq!(counters.late_after_demand, 0);
    }

    #[test]
    fn live_speculative_acquisition_finishes_just_before_demand() {
        let controller = controller();
        controller.begin_run(ShadowPhase::Measured, 0).unwrap();
        let pending = pending(4);
        let ticket = accept(&controller, &pending);
        controller.record_source_started(ticket.ticket_id, false, false);
        controller.record_installed(ticket.ticket_id, 4096, None, 0, false);
        classify(
            &controller,
            &pending,
            HashSet::from([4]),
            u64::MAX / 2,
            true,
        );
        let counters = counters(&controller);
        assert_eq!(counters.prediction_useful, 1);
        assert_eq!(counters.completed_before_first_demand, 1);
        assert_eq!(counters.demand_reused_speculative_result, 1);
    }

    #[test]
    fn live_demand_independently_wins_installation_race() {
        let controller = controller();
        controller.begin_run(ShadowPhase::Measured, 0).unwrap();
        let pending = pending(4);
        let ticket = accept(&controller, &pending);
        classify(&controller, &pending, HashSet::from([4]), 100, false);
        assert!(controller.reserve_install(ticket.ticket_id, false));
        controller.record_demand_or_peer_won_install_race(ticket.ticket_id);
        let counters = counters(&controller);
        assert_eq!(counters.demand_independently_won_install_race, 1);
        assert_eq!(counters.late_after_demand, 1);
        assert_eq!(counters.completed_tasks, 1);
    }

    #[test]
    fn live_speculative_requester_becomes_stale_before_install() {
        let controller = controller();
        controller.begin_run(ShadowPhase::Measured, 0).unwrap();
        let pending = pending(4);
        let ticket = accept(&controller, &pending);
        controller.record_source_started(ticket.ticket_id, false, false);
        controller.record_nvme_complete(ticket.ticket_id, 4096);
        controller.record_install_outcome(
            ticket.ticket_id,
            GpuNativeLiveSpeculativeInstall::StaleLogicalGeneration,
            4096,
        );
        let counters = counters(&controller);
        assert_eq!(counters.source_acquisition_cancelled_stale, 1);
        assert_eq!(counters.cancelled_stale_logical_generation, 1);
        assert_eq!(counters.cancellations_wasted_source_bytes, 4096);
    }

    #[test]
    fn live_slot_generation_change_before_speculative_completion_is_stale() {
        let controller = controller();
        controller.begin_run(ShadowPhase::Measured, 0).unwrap();
        let pending = pending(4);
        let ticket = accept(&controller, &pending);
        assert!(controller.reserve_install(ticket.ticket_id, true));
        controller.record_install_outcome(
            ticket.ticket_id,
            GpuNativeLiveSpeculativeInstall::StaleLogicalGeneration,
            4096,
        );
        let counters = counters(&controller);
        assert_eq!(counters.physical_installation_started, 1);
        assert_eq!(counters.physical_installation_completed, 0);
        assert_eq!(counters.source_acquisition_cancelled_stale, 1);
        assert_eq!(counters.cancelled_stale_logical_generation, 1);
    }

    #[test]
    fn live_same_expert_predicted_repeatedly_obeys_in_flight_bound() {
        let controller = controller();
        controller.begin_run(ShadowPhase::Measured, 0).unwrap();
        let pending = pending(4);
        let _first = accept(&controller, &pending);
        let second = controller
            .accept_candidate(
                &pending,
                &pending.candidates[0],
                Arc::new(HashSet::from([4])),
            )
            .unwrap();
        assert!(second.is_none());
        let counters = counters(&controller);
        assert_eq!(counters.accepted_candidates, 1);
        assert_eq!(counters.rejected_in_flight_bound, 1);
    }

    #[test]
    fn live_candidate_already_physically_current_terminates_without_source() {
        let controller = controller();
        controller.begin_run(ShadowPhase::Measured, 0).unwrap();
        let pending = pending(4);
        let ticket = accept(&controller, &pending);
        controller.record_already_physically_resident(ticket.ticket_id);
        let counters = counters(&controller);
        assert_eq!(counters.already_physically_resident, 1);
        assert_eq!(counters.source_acquisition_started, 0);
        assert_eq!(counters.completed_tasks, 1);
    }

    #[test]
    fn live_speculative_cancellation_preserves_demand_need_classification() {
        let controller = controller();
        controller.begin_run(ShadowPhase::Measured, 0).unwrap();
        let pending = pending(4);
        let ticket = accept(&controller, &pending);
        controller.record_source_started(ticket.ticket_id, false, false);
        classify(&controller, &pending, HashSet::from([4]), 100, false);
        controller.record_cancelled(
            ticket.ticket_id,
            LiveCancellationReason::CancelledTaskOrShutdown,
        );
        let counters = counters(&controller);
        assert_eq!(counters.demand_arrived_while_speculative_in_flight, 1);
        assert_eq!(counters.source_acquisition_cancelled_stale, 1);
        assert_eq!(counters.cancelled_task_or_shutdown, 1);
        assert_eq!(counters.completed_tasks, 1);
    }

    #[test]
    fn off_arm_accepts_no_candidates_and_performs_zero_live_work() {
        let controller = controller_for_arm(LiveArm::Off);
        controller.begin_run(ShadowPhase::Measured, 0).unwrap();
        let pending = pending(4);
        let accepted = controller
            .accept_candidate(
                &pending,
                &pending.candidates[0],
                Arc::new(HashSet::from([4])),
            )
            .unwrap();
        assert!(accepted.is_none());
        let counters = counters(&controller);
        assert_eq!(counters.rejected_live_disabled, 1);
        assert_eq!(counters.accepted_candidates, 0);
        assert_eq!(counters.source_acquisition_started, 0);
        assert_eq!(counters.speculative_joined_existing_acquisition, 0);
        assert_eq!(counters.physical_installation_started, 0);
    }

    #[test]
    fn cancellation_reason_accounting_reconciles_with_legacy_aggregate() {
        let controller = controller();
        controller.begin_run(ShadowPhase::Measured, 0).unwrap();
        let pending = pending(4);
        for reason in [
            LiveCancellationReason::RejectedConcurrencyNoPermit,
            LiveCancellationReason::RejectedResidencyPressure,
            LiveCancellationReason::CancelledBoundaryExpired,
            LiveCancellationReason::CancelledBackgroundSpawn,
            LiveCancellationReason::CancelledTaskOrShutdown,
            LiveCancellationReason::CancelledStaleLogicalGeneration,
        ] {
            let ticket = accept(&controller, &pending);
            controller.record_cancelled(ticket.ticket_id, reason);
        }
        let counters = counters(&controller);
        let attributed = counters.rejected_concurrency_no_permit
            + counters.rejected_residency_pressure
            + counters.cancelled_boundary_expired
            + counters.cancelled_background_spawn
            + counters.cancelled_task_or_shutdown
            + counters.cancelled_stale_logical_generation;
        assert_eq!(counters.source_acquisition_cancelled_stale, 6);
        assert_eq!(attributed, counters.source_acquisition_cancelled_stale);
        assert_eq!(counters.completed_tasks, 6);
    }

    #[test]
    fn nonresident_no_live_work_is_performance_ineligible() {
        let counters = LiveLifecycleCounters {
            accepted_candidates: 1,
            ..LiveLifecycleCounters::default()
        };
        let evidence = live_path_exercised(&counters);
        assert_eq!(evidence.nonresident_accepted_candidates, 1);
        assert!(!evidence.pass);
        assert!(!performance_eligible(true, evidence.pass));

        let progressed = LiveLifecycleCounters {
            accepted_candidates: 1,
            source_acquisition_started: 1,
            ..LiveLifecycleCounters::default()
        };
        assert!(live_path_exercised(&progressed).pass);
        assert!(performance_eligible(
            true,
            live_path_exercised(&progressed).pass
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn live_concurrent_request_teardown_retires_registered_task() {
        let controller = controller();
        controller.begin_run(ShadowPhase::Measured, 0).unwrap();
        let pending = pending(4);
        let ticket = accept(&controller, &pending);
        let (_sender, receiver) = tokio::sync::oneshot::channel::<()>();
        let handle = tokio::spawn(async move {
            let _ = receiver.await;
        });
        controller.register_abort_handle(ticket.ticket_id, handle.abort_handle());
        handle.abort();
        controller.record_cancelled(
            ticket.ticket_id,
            LiveCancellationReason::CancelledTaskOrShutdown,
        );
        let _ = handle.await;
        let inner = controller.inner.lock();
        let run = inner.run.as_ref().unwrap();
        assert!(run.candidates.values().all(|candidate| candidate.terminal));
        assert!(run
            .candidates
            .values()
            .all(|candidate| candidate.abort_handle.is_none()));
    }

    #[test]
    fn live_failed_speculative_install_releases_all_task_ownership() {
        let controller = controller();
        controller.begin_run(ShadowPhase::Measured, 0).unwrap();
        let pending = pending(4);
        let ticket = accept(&controller, &pending);
        assert!(controller.reserve_install(ticket.ticket_id, true));
        controller.record_fatal(
            ticket.ticket_id,
            "injected install identity failure".to_string(),
        );
        let inner = controller.inner.lock();
        let run = inner.run.as_ref().unwrap();
        assert_eq!(run.counters.accepted_candidates, 1);
        assert_eq!(run.counters.completed_tasks, 1);
        assert!(run.failure.is_some());
        assert!(run.candidates.values().all(|candidate| candidate.terminal));
        assert!(run
            .candidates
            .values()
            .all(|candidate| candidate.abort_handle.is_none()));
    }

    #[test]
    fn cli_rejects_missing_live_replacement_policy() {
        let raw = [
            "micro-expert-router",
            COMMAND,
            "--config",
            "config.toml",
            "--prompt",
            "hello",
            "--output-tokens",
            "2",
            "--greedy",
            "--expected-adapter-name",
            "NVIDIA L4",
        ]
        .into_iter()
        .map(std::ffi::OsString::from)
        .collect::<Vec<_>>();
        let (normalized, requested, replacement_policy) =
            crate::normalize_bounded_live_prefetch_command(&raw).unwrap();
        assert!(requested);
        let parsed = crate::Cli::try_parse_from(normalized).unwrap();
        assert!(matches!(
            parsed.cmd,
            crate::Cmd::QualifyGpuNativePrefetchMultipredictorShadow { .. }
        ));
        assert!(replacement_policy.is_none());
        assert!(require_explicit_replacement_policy(replacement_policy).is_err());
    }

    #[test]
    fn cli_parses_explicit_live_replacement_policy() {
        let raw = [
            "micro-expert-router",
            COMMAND,
            "--config",
            "config.toml",
            "--prompt",
            "hello",
            "--output-tokens",
            "2",
            "--greedy",
            "--expected-adapter-name",
            "NVIDIA L4",
            "--replacement-policy",
            "route-recency-prediction-protected",
        ]
        .into_iter()
        .map(std::ffi::OsString::from)
        .collect::<Vec<_>>();
        let (normalized, requested, replacement_policy) =
            crate::normalize_bounded_live_prefetch_command(&raw).unwrap();
        assert!(requested);
        let parsed = crate::Cli::try_parse_from(normalized).unwrap();
        assert!(matches!(
            parsed.cmd,
            crate::Cmd::QualifyGpuNativePrefetchMultipredictorShadow { .. }
        ));
        assert!(matches!(
            require_explicit_replacement_policy(replacement_policy),
            Ok(GpuNativeLiveReplacementPolicy::RouteRecencyPredictionProtected)
        ));
    }
}
