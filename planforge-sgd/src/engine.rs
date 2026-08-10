//! The optimizer loop.
//!
//! One round of the loop is: build the relaxed transcription from the current
//! logits, evaluate every constraint residual, form an augmented-Lagrangian
//! loss, backpropagate, take an Adam step. Periodically the argmax plan of each
//! particle is replayed under exact semantics; if it reaches the goal we are
//! done, and if it does not, the failure is used to *reweight constraints* —
//! never to choose an action.
//!
//! What is deliberately absent: any frontier, queue, successor enumeration or
//! branching, and any heuristic estimate. The only exact reasoning is replaying
//! one already-decoded sequence.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use candle_core::{DType, Device, Result as CandleResult, Tensor, Var};
use planforge_sas::numeric_task::{ExplicitFact, Operator, TaskRef};
use planforge_sas::plan_verification::{PlanRejection, Replay, ReplayOutcome, replay_plan};
use planforge_sas::state_registry::StateRegistry;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::adam::{Adam, AdamParams};
use crate::config::{CausalCopyMode, CausalStage, SgdConfig};
use crate::controller::{
    ControllerConfig, ExactFeedback, FailureWeightSchedule, GoalWeightSchedule, Phase,
    PhasePatience, RemeltWindow, VerifierController,
};
use crate::tensor::{
    DTYPE, TemporalTokenDistribution, TensorPlan, TensorPlanError, TwoLossForward,
    bottleneck_norm_per_particle,
};
use crate::transcription::{Transcription, TranscriptionError};

const CAUSAL_ACTION_RNG_DOMAIN: u64 = 0xCA05_A17C_10A1_C0DE;
const CAUSAL_LINK_RNG_DOMAIN: u64 = 0xC4A5_4C1A_1A7E_5EED;

/// Why a solve attempt could not run at all.
#[derive(Debug)]
pub enum SgdError {
    Config(crate::config::SgdConfigError),
    Transcription(TranscriptionError),
    TensorPlan(TensorPlanError),
    /// A failure inside the tensor backend or the state machinery. These are
    /// bugs or resource exhaustion, never "this task is hard".
    Backend(String),
}

impl std::fmt::Display for SgdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(error) => write!(f, "{error}"),
            Self::Transcription(error) => write!(f, "{error}"),
            Self::TensorPlan(error) => write!(f, "{error}"),
            Self::Backend(message) => write!(f, "sgd backend failure: {message}"),
        }
    }
}

impl std::error::Error for SgdError {}

impl From<candle_core::Error> for SgdError {
    fn from(error: candle_core::Error) -> Self {
        Self::Backend(error.to_string())
    }
}

/// How a solve attempt ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SgdStatus {
    /// A plan was found and verified.
    Solved,
    /// The budget ran out. This says nothing about solvability: the optimizer is
    /// incomplete by construction.
    BudgetExhausted,
    /// The task provably has no plan, seen structurally while transcribing.
    Unsolvable,
}

/// Result of a solve attempt.
#[derive(Debug, Clone)]
pub struct SgdOutcome {
    pub status: SgdStatus,
    /// Operator indices into the original task, no-ops removed.
    pub plan: Option<Vec<usize>>,
    pub cost: Option<f64>,
    pub updates: usize,
    pub verifier_calls: usize,
    pub horizon_rounds: usize,
    pub final_horizon: usize,
    /// Smallest total residual seen, as a progress diagnostic.
    pub best_total_residual: f64,
    /// Most goals any exact replay achieved, and how many there are.
    pub best_goals_reached: usize,
    pub num_goals: usize,
    /// Longest applicable prefix any exact replay produced.
    pub longest_applicable_prefix: usize,
    /// Whole-plan refreshes performed.
    pub refreshes: usize,
    /// Stalled particles remelted: noise burst plus Adam moment reset.
    pub remelts: usize,
    /// Full-support restarts of unlocked temporal repair variables. These are
    /// reported separately from direct row-window remelts.
    pub temporal_restarts: usize,
    /// Distinct verifier-exposed threat edges that contradict a particle's
    /// immutable repair-goal order.
    pub temporal_order_conflicts: usize,
    /// Distinct repeated-fact cycles exposed while recursively repairing
    /// obligated actions.
    pub temporal_causal_cycles: usize,
    /// One-check-interval symmetric rejections of the currently selected
    /// achiever after exact replay exposes a repeated-fact causal cycle.
    pub temporal_cycle_interventions: usize,
    /// Distinct producer-before-scaffold constraints added after an inserted
    /// repair action invalidated a previously applicable scaffold action.
    pub temporal_scaffold_repairs: usize,
    /// Updates on which at least one particle had an active backward bridge.
    pub backward_bridge_updates: usize,
    /// Largest population-mean backward-bridge loss seen during the run.
    pub max_backward_bridge_loss: f64,
    /// Diagnostics from the final update, for experiment reporting and for
    /// telling apart "the relaxation is still infeasible" from "the relaxation
    /// is feasible but will not round".
    pub final_diagnostics: Diagnostics,
    /// Best verifier observation ranked as one coherent checkpoint.  Unlike
    /// the historical maxima above, every field refers to the same particle,
    /// update, and decoded plan.
    pub best_exact_checkpoint: Option<ExactCheckpoint>,
    /// Read-only optimizer snapshots requested by `trace_period`.
    pub trace: Vec<SgdTracePoint>,
    /// Human-readable SAS fact names in the local flat-fact index space used
    /// by temporal obligations. Empty when tracing is disabled.
    pub trace_fact_names: Vec<String>,
}

/// One deterministic probability snapshot for diagnosing loss geometry.
#[derive(Debug, Clone)]
pub struct SgdTracePoint {
    pub round: usize,
    pub update: usize,
    pub particle: usize,
    pub phase: Phase,
    pub goal_weights: Vec<f64>,
    pub missing_goals: Vec<bool>,
    pub goal_repair_start: usize,
    pub action_temperatures: Vec<f64>,
    /// Full categorical probabilities, indexed `[row][action]`.
    pub action_probabilities: Vec<Vec<f64>>,
    /// Latent action identities, indexed `[token][action]`; empty without
    /// temporal tokens.
    pub token_action_probabilities: Vec<Vec<f64>>,
    /// Persistent verifier-derived fact role for each latent token.
    pub temporal_obligations: Vec<Option<usize>>,
    /// Every action that can establish each token's fact role. Empty for an
    /// unassigned token. This is trace-only structural evidence.
    pub temporal_obligation_achievers: Vec<Vec<usize>>,
    /// Persistent adaptive multiplier for each token's factual certificate.
    pub temporal_obligation_focus: Vec<f64>,
    /// Persistent first-failure multiplier for placing the token on an
    /// applicable achiever in a verifier-known scaffold gap.
    pub temporal_applicability_focus: Vec<f64>,
    /// Mean direct-transcription precondition residual at each row.
    pub precondition_by_row: Vec<f64>,
    /// Sum of the four mean transition residual families at each row.
    pub transition_by_row: Vec<f64>,
    /// Direct terminal residual for each task goal.
    pub goal_residuals: Vec<f64>,
    /// Delete-aware recurrent applicability loss at each row.
    pub recurrent_precondition_by_row: Vec<f64>,
    /// Delete-aware terminal residual for each task goal.
    pub recurrent_terminal_goals: Vec<f64>,
    /// Optimistic producer residual for each task goal.
    pub recurrent_producer_goals: Vec<f64>,
    /// Categorical impurity at each action row.
    pub action_integrality_by_row: Vec<f64>,
    /// Gradient of the complete scheduled loss with respect to raw action
    /// logits, indexed `[row][action]` at this same pre-step snapshot.
    pub action_logit_gradients: Vec<Vec<f64>>,
    /// Token-to-execution-row assignment `[token][row]`, empty in direct mode.
    pub temporal_assignment: Vec<Vec<f64>>,
    /// Continuous globally normalized path marginals `[token][row]`, empty in
    /// direct mode.
    pub temporal_soft_assignment: Vec<Vec<f64>>,
    /// Scheduled-loss gradient for the assignment logits.
    pub schedule_logit_gradients: Vec<Vec<f64>>,
}

/// One exact replay observation with matching soft diagnostics.
#[derive(Debug, Clone)]
pub struct ExactCheckpoint {
    pub update: usize,
    pub particle: usize,
    pub applicable_real_actions: usize,
    pub decoded_real_actions: usize,
    pub goals_reached: usize,
    pub num_goals: usize,
    pub failure_kind: &'static str,
    pub failure_slot: Option<usize>,
    pub max_residual: f64,
    pub worst_integrality: f64,
    /// Original-task operator indices in decoded real-action order.
    pub decoded_plan: Vec<usize>,
    /// Every tensor row in order; `None` is the explicit no-op.
    pub decoded_slots: Vec<Option<usize>>,
    /// Task goal indices false in the exact replay state.
    pub missing_goals: Vec<usize>,
}

impl ExactCheckpoint {
    fn fully_applicable(&self) -> bool {
        self.failure_slot.is_none()
    }

    fn applicable_fraction(&self) -> f64 {
        if self.fully_applicable() || self.decoded_real_actions == 0 {
            1.0
        } else {
            self.applicable_real_actions as f64 / self.decoded_real_actions as f64
        }
    }

    fn is_better_than(&self, incumbent: &Self) -> bool {
        self.fully_applicable()
            .cmp(&incumbent.fully_applicable())
            .then(self.goals_reached.cmp(&incumbent.goals_reached))
            .then_with(|| {
                self.applicable_fraction()
                    .total_cmp(&incumbent.applicable_fraction())
            })
            .then_with(|| incumbent.max_residual.total_cmp(&self.max_residual))
            .then_with(|| {
                incumbent
                    .worst_integrality
                    .total_cmp(&self.worst_integrality)
            })
            .is_gt()
    }
}

impl std::fmt::Display for ExactCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let failure_slot = self
            .failure_slot
            .map_or_else(|| "null".to_owned(), |slot| slot.to_string());
        write!(
            f,
            "update={} particle={} applicable_real_actions={} decoded_real_actions={} \
             goals_reached={} num_goals={} failure_kind={} failure_slot={} \
             max_residual={:.9} worst_integrality={:.9}",
            self.update,
            self.particle,
            self.applicable_real_actions,
            self.decoded_real_actions,
            self.goals_reached,
            self.num_goals,
            self.failure_kind,
            failure_slot,
            self.max_residual,
            self.worst_integrality,
        )
    }
}

/// A snapshot of the relaxation, taken at the last update.
#[derive(Debug, Clone, Default)]
pub struct Diagnostics {
    pub precondition_residual: f64,
    pub transition_residual: f64,
    pub goal_residual: f64,
    /// Delete-aware action-only applicability loss for execution plan P.
    pub recurrent_precondition: f64,
    /// Delete-aware terminal goal loss for execution plan P.
    pub recurrent_goal: f64,
    /// Analytic producer-to-terminal threat-survival loss for execution P.
    pub recurrent_survival: f64,
    /// Optimistic monotone producer/goal loss retained only as shaping.
    pub recurrent_producer: f64,
    /// Straight-through hardening applied to execution recurrence P.
    pub recurrent_hardening: f64,
    /// Local excess occupancy above the configured insertion-slack capacity.
    pub slot_slack: f64,
    /// Verifier-triggered optimistic producer insertion loss.
    pub insertion_raw: f64,
    /// Verifier-triggered applicability-supported survival insertion loss.
    pub insertion_supported: f64,
    /// Unsupported recursively induced facts at the goal-repair boundary.
    pub backward_bridge_boundary: f64,
    /// Goal-chain-relevant false preconditions inside the repair suffix.
    pub backward_bridge_precondition: f64,
    /// Soft cross-entropy to real actions outside the active repair gaps.
    pub anchor_trust: f64,
    pub action_integrality: f64,
    pub state_integrality: f64,
    pub causal_consensus: f64,
    pub causal_link_source: f64,
    pub causal_link_threat: f64,
    pub causal_link_integrality: f64,
}

impl std::fmt::Display for Diagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "pre={:.6} trans={:.6} goal={:.6} recurrent_pre={:.6} recurrent_goal={:.6} \
             recurrent_survival={:.6} recurrent_producer={:.6} recurrent_hardening={:.3} slot_slack={:.6} \
             insert_raw={:.6} insert_supported={:.6} bridge_boundary={:.6} bridge_pre={:.6} \
             anchor={:.6} int_action={:.6} int_state={:.6} \
             causal_consensus={:.6} link_source={:.6} link_threat={:.6} link_int={:.6}",
            self.precondition_residual,
            self.transition_residual,
            self.goal_residual,
            self.recurrent_precondition,
            self.recurrent_goal,
            self.recurrent_survival,
            self.recurrent_producer,
            self.recurrent_hardening,
            self.slot_slack,
            self.insertion_raw,
            self.insertion_supported,
            self.backward_bridge_boundary,
            self.backward_bridge_precondition,
            self.anchor_trust,
            self.action_integrality,
            self.state_integrality,
            self.causal_consensus,
            self.causal_link_source,
            self.causal_link_threat,
            self.causal_link_integrality,
        )
    }
}

/// Per-element duals, one tensor per residual family.
struct Duals {
    precondition: Tensor,
    transition: [Tensor; 4],
    goal: Tensor,
}

/// Optional dense causal-link parameter lane.
///
/// Keeping the variable, its optimizer state, and its independent RNG streams
/// behind one option prevents a partially initialized lane: either every piece
/// needed by link forward/refresh/remelting exists, or none of the
/// `O(M H^2 F)` allocation does.
struct CausalLinkLane {
    logits: Var,
    optimizer: Adam,
    streams: Vec<ChaCha8Rng>,
}

impl Duals {
    fn zeros(plan: &TensorPlan, device: &Device) -> CandleResult<Self> {
        let (m, h) = (plan.particles, plan.horizon);
        Ok(Self {
            precondition: Tensor::zeros((m, h, plan.num_preconditions), DTYPE, device)?,
            transition: [
                Tensor::zeros((m, h, plan.num_facts), DTYPE, device)?,
                Tensor::zeros((m, h, plan.num_facts), DTYPE, device)?,
                Tensor::zeros((m, h, plan.num_facts), DTYPE, device)?,
                Tensor::zeros((m, h, plan.num_facts), DTYPE, device)?,
            ],
            goal: Tensor::zeros((m, 1, plan.num_goals), DTYPE, device)?,
        })
    }

    /// Forget verifier/AL history for independently refreshed particles.
    fn reset_prefix(
        &mut self,
        refreshed: usize,
        particles: usize,
        device: &Device,
    ) -> CandleResult<()> {
        assert!(
            refreshed <= particles,
            "validated refresh count is within the particle population"
        );
        let mut keep = vec![1.0f64; particles];
        keep[..refreshed].fill(0.0);
        let keep = Tensor::from_vec(keep, (particles, 1, 1), device)?;
        self.precondition = self.precondition.broadcast_mul(&keep)?.detach();
        for transition in &mut self.transition {
            *transition = transition.broadcast_mul(&keep)?.detach();
        }
        self.goal = self.goal.broadcast_mul(&keep)?.detach();
        Ok(())
    }
}

/// `λ ← min(cap, growth·λ + ρ·r)` where the residual is still violated, and
/// `λ ← decay·λ` where it is satisfied.
///
/// The slow decay is deliberate: a repaired constraint keeps some memory, which
/// damps the oscillation where fixing one precondition breaks another and back.
/// Duals are *not* differentiated through: they are multipliers, and the loss
/// treats them as constants. Detaching is therefore not just an optimization —
/// a residual tensor carries the whole autograd graph of the update that
/// produced it, so building the next dual from an attached residual would chain
/// every iteration's graph onto the next one and leak without bound.
fn updated_dual(
    dual: &Tensor,
    residual: &Tensor,
    rho: f64,
    config: &SgdConfig,
) -> CandleResult<Tensor> {
    let residual = residual.detach();
    let violated = residual.ge(config.residual_tolerance)?.to_dtype(DTYPE)?;
    let satisfied = (violated.ones_like()? - &violated)?;
    let grown = ((dual * config.dual_growth)? + (residual * rho)?)?;
    let decayed = (dual * config.dual_decay)?;
    let combined = ((grown * &violated)? + (decayed * &satisfied)?)?;
    Ok(combined.clamp(0.0, config.dual_cap)?.detach())
}

/// Elementwise-weighted augmented-Lagrangian family loss.
///
/// The weight multiplies the complete AL contribution once. Scaling the
/// residual before squaring would accidentally turn a requested weight `w`
/// into `w²` on the quadratic term and make the primal and dual updates use
/// different residuals.
fn weighted_family_loss(
    residual: &Tensor,
    dual: &Tensor,
    rho: f64,
    weight: &Tensor,
) -> CandleResult<Tensor> {
    let linear = (residual * dual)?;
    let quadratic = (residual.sqr()? * (rho / 2.0))?;
    (linear + quadratic)?.broadcast_mul(weight)?.mean_all()
}

/// Apply independent particle schedules before taking the population mean.
///
/// Keeping the particle axis until after multiplication is essential: replacing
/// this with `mean(loss) * mean(weight)` makes one particle's controller phase
/// scale every other particle's gradient.
fn scheduled_particle_mean(per_particle: &Tensor, weight: &Tensor) -> CandleResult<Tensor> {
    assert_eq!(
        per_particle.dims(),
        weight.dims(),
        "per-particle loss and schedule shapes must match"
    );
    assert_eq!(
        per_particle.rank(),
        2,
        "scheduled per-particle losses have shape [M, 1]"
    );
    assert_eq!(
        per_particle.dim(1)?,
        1,
        "scheduled per-particle losses have one value per particle"
    );
    (per_particle * weight)?.mean_all()
}

/// Soft trust region around a detached exact-checkpoint sequence.
///
/// `target` is one-hot `[M,H,A]`; `active` is `[M,H,1]`. Every active token has
/// equal weight. Empty particles contribute exactly zero and do not dilute
/// another particle's anchor.
fn anchor_trust_loss(
    log_action: &Tensor,
    target: &Tensor,
    active: &Tensor,
) -> CandleResult<Tensor> {
    assert_eq!(log_action.dims(), target.dims());
    let (particles, horizon, _) = log_action.dims3()?;
    assert_eq!(active.dims(), &[particles, horizon, 1]);
    let cells = (log_action * target)?.sum(2)?.neg()?;
    let active_rows = active.reshape((particles, horizon))?;
    let numerator = (cells * &active_rows)?.sum_all()?;
    let denominator = active_rows.sum_all()?.clamp(1.0, f64::MAX)?;
    numerator / denominator
}

/// Reparameterize selected particles by moving their final overparameterized
/// row into `insert_at` and shifting the old suffix right by one row.
///
/// This is a pure differentiable view of the fixed `[M,H,A]` parameter tensor:
/// no action is chosen or edited on the host, and the optimizer still updates
/// the same coordinates. The last row is the insertion variable and the old
/// last suffix row is the sacrificed padding capacity.
fn insertion_warp_logits(logits: &Tensor, insert_at: &[Option<usize>]) -> CandleResult<Tensor> {
    let (particles, horizon, _actions) = logits.dims3()?;
    assert_eq!(
        insert_at.len(),
        particles,
        "one insertion coordinate is required per particle"
    );
    let mut warped_particles = Vec::with_capacity(particles);
    for (particle, &position) in insert_at.iter().enumerate() {
        let particle_logits = logits.narrow(0, particle, 1)?;
        let Some(position) = position else {
            warped_particles.push(particle_logits);
            continue;
        };
        assert!(
            position + 1 < horizon,
            "insertion requires one later padding row"
        );
        let insertion = particle_logits.narrow(1, horizon - 1, 1)?;
        let suffix = particle_logits.narrow(1, position, horizon - position - 1)?;
        let warped = if position == 0 {
            Tensor::cat(&[&insertion, &suffix], 1)?
        } else {
            let prefix = particle_logits.narrow(1, 0, position)?;
            Tensor::cat(&[&prefix, &insertion, &suffix], 1)?
        };
        warped_particles.push(warped);
    }
    Tensor::cat(&warped_particles.iter().collect::<Vec<_>>(), 0)
}

/// Symmetric exact-decoder certificate at verifier failure rows.
///
/// Let `z_good` and `z_bad` be the largest stable log probabilities among
/// exactly applicable and inapplicable actions. The hinge is
///
/// `max(0, z_bad - z_good - log(max_bad_good_ratio))`.
///
/// Its zero set has `z_good > z_bad` because the configured ratio is strictly
/// below one, so the decoded argmax is applicable for either the categorical
/// or factorized slot parameterization. Unlike a
/// bound on *total* bad probability, this does not force every irrelevant bad
/// action toward zero or make the always-applicable no-op absorb almost all
/// mass.  All good actions remain symmetric; the verifier does not select a
/// successor.
fn applicability_ranking_barrier(
    action_log_probability: &Tensor,
    applicable: &Tensor,
    active: &Tensor,
    focus: &Tensor,
    max_bad_good_ratio: f64,
) -> CandleResult<Tensor> {
    assert_eq!(
        action_log_probability.dims(),
        applicable.dims(),
        "action logits and exact applicability mask must have the same shape"
    );
    assert_eq!(
        action_log_probability.rank(),
        3,
        "action logits and applicability have shape [M, H, A]"
    );
    let (particles, horizon, _) = action_log_probability.dims3()?;
    assert_eq!(
        active.dims(),
        &[particles, horizon],
        "active failure mask has shape [M, H]"
    );
    assert_eq!(
        focus.dims(),
        &[particles, horizon, 1],
        "failure focus has shape [M, H, 1]"
    );
    assert!(
        max_bad_good_ratio.is_finite() && (0.0..1.0).contains(&max_bad_good_ratio),
        "validated bad/good ratio is finite and strictly between zero and one"
    );

    let active_row = active.unsqueeze(2)?;
    // Inactive rows are treated as all-good/no-bad.  This keeps their hinge
    // identically zero without relying on `0 * huge_mask_value` arithmetic.
    let inactive_row = (active_row.ones_like()? - &active_row)?;
    let good_mask = applicable.broadcast_add(&inactive_row)?.clamp(0.0, 1.0)?;
    let bad_mask = (applicable.ones_like()? - applicable)?.broadcast_mul(&active_row)?;
    let mask_floor = -1e300;
    let good_logits = ((action_log_probability * &good_mask)?
        + ((good_mask.ones_like()? - &good_mask)? * mask_floor)?)?
        .max(2)?;
    let bad_logits = ((action_log_probability * &bad_mask)?
        + ((bad_mask.ones_like()? - &bad_mask)? * mask_floor)?)?
        .max(2)?;
    let strict_logit_margin = -max_bad_good_ratio.ln();
    let hinge = ((bad_logits - good_logits)? + strict_logit_margin)?.relu()?;
    let active_weight = active.broadcast_mul(&focus.reshape((particles, horizon))?)?;
    (hinge * active_weight)?.sum_all()? / particles as f64
}

/// Dense symmetric pressure on the set of exactly applicable actions.
///
/// This is deliberately not the decoder certificate: it need not approach one
/// before the row can be repaired.  Its role is to give every good action a
/// useful gradient while the max-ranking hinge handles the exact argmax
/// ordering.  In particular, it does not select a successor among the good
/// actions returned by exact replay.
fn applicability_mass_loss(
    action: &Tensor,
    applicable: &Tensor,
    active: &Tensor,
    focus: &Tensor,
) -> CandleResult<Tensor> {
    assert_eq!(
        action.dims(),
        applicable.dims(),
        "action distribution and applicability mask have the same shape"
    );
    let (particles, horizon, _) = action.dims3()?;
    assert_eq!(active.dims(), &[particles, horizon]);
    assert_eq!(focus.dims(), &[particles, horizon, 1]);
    let good_mass = (action * applicable)?.sum(2)?.clamp(1e-300, 1.0)?;
    let active_weight = active.broadcast_mul(&focus.reshape((particles, horizon))?)?;
    (good_mass.log()?.neg()? * active_weight)?.sum_all()? / particles as f64
}

/// Conjoin a persistent producer role with exact applicability when possible.
///
/// If no achiever is currently applicable, returning all achievers keeps the
/// consumer role alive while an earlier obligation repairs its prerequisite.
/// The returned set is still symmetric and contains no host-selected action.
fn obligation_achiever_conjunction(
    achievers: &[usize],
    exact_applicable: Option<&[f64]>,
) -> Vec<usize> {
    assert!(
        !achievers.is_empty(),
        "an obligation has at least one achiever"
    );
    let Some(exact_applicable) = exact_applicable else {
        return achievers.to_vec();
    };
    let conjunction = achievers
        .iter()
        .copied()
        .filter(|&action| {
            exact_applicable
                .get(action)
                .is_some_and(|&applicable| applicable > 0.0)
        })
        .collect::<Vec<_>>();
    if conjunction.is_empty() {
        achievers.to_vec()
    } else {
        conjunction
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecurrentLossSource {
    Execution,
    CausalCopy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecurrentGoalMode {
    ProducerDiscovery,
    DeleteAwareTerminal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RecurrentLossRoute {
    source: RecurrentLossSource,
    goal: RecurrentGoalMode,
}

/// Select which coherent action-only rollout supplies causal gradients.
///
/// The cheap monotone producer objective is deliberately limited to early
/// scaffold discovery.  From Proof onward every non-staged execution plan is
/// trained against its delete-aware terminal state on every update.  In staged
/// mode Q receives that proof objective, then P becomes authoritative during
/// Transfer rather than waiting until Takeover.  This routing is independent
/// of whether the optional dense causal-link lane exists.
fn recurrent_loss_route(
    copy: CausalCopyMode,
    stage: CausalStage,
    update: usize,
) -> Option<RecurrentLossRoute> {
    let staged = matches!(copy, CausalCopyMode::Staged);
    match (staged, stage) {
        (true, CausalStage::Proof) => Some(RecurrentLossRoute {
            source: RecurrentLossSource::CausalCopy,
            goal: RecurrentGoalMode::DeleteAwareTerminal,
        }),
        (true, CausalStage::Transfer | CausalStage::Takeover | CausalStage::Polish)
        | (
            false,
            CausalStage::Proof
            | CausalStage::Transfer
            | CausalStage::Takeover
            | CausalStage::Polish,
        ) => Some(RecurrentLossRoute {
            source: RecurrentLossSource::Execution,
            goal: RecurrentGoalMode::DeleteAwareTerminal,
        }),
        (true, CausalStage::Shadow) if update % 4 == 0 => Some(RecurrentLossRoute {
            source: RecurrentLossSource::Execution,
            goal: RecurrentGoalMode::ProducerDiscovery,
        }),
        (true, CausalStage::Discovery) if update % 4 == 0 => Some(RecurrentLossRoute {
            source: RecurrentLossSource::CausalCopy,
            goal: RecurrentGoalMode::ProducerDiscovery,
        }),
        (false, CausalStage::Shadow | CausalStage::Discovery) if update % 4 == 0 => {
            Some(RecurrentLossRoute {
                source: RecurrentLossSource::Execution,
                goal: RecurrentGoalMode::ProducerDiscovery,
            })
        }
        _ => None,
    }
}

/// Goal, applicability, and no-op shaping from one coherent recurrent plan.
///
/// Producer discovery is intentionally optimistic so an unsupported achiever
/// can first acquire mass.  The late mode makes the delete-aware terminal loss
/// authoritative and retains only one tenth of the optimistic loss as a
/// bootstrap direction.  Exact verification remains the final authority.
fn recurrent_plan_loss(
    forward: &TwoLossForward,
    per_goal_weight: &Tensor,
    row_focus: &Tensor,
    protect_consumer: &Tensor,
    goal_mode: RecurrentGoalMode,
    rho_goal: f64,
    rho_precondition: f64,
    goal_survival_weight: f64,
) -> CandleResult<Tensor> {
    assert_eq!(
        forward.relaxed_goal_by_goal.dims(),
        per_goal_weight.dims(),
        "recurrent per-goal loss and schedule have the same shape"
    );
    assert_eq!(
        forward.failed_precondition_by_step.dims(),
        row_focus.dims(),
        "recurrent precondition loss and row focus have the same shape"
    );
    assert_eq!(
        protect_consumer.dims(),
        &[forward.failed_precondition_by_step.dim(0)?, 1],
        "consumer protection has one scalar per recurrent particle"
    );

    let mut loss = match goal_mode {
        RecurrentGoalMode::ProducerDiscovery => {
            ((&forward.relaxed_goal_by_goal * per_goal_weight)?.mean_all()? * rho_goal)?
        }
        RecurrentGoalMode::DeleteAwareTerminal => {
            assert!(
                goal_survival_weight.is_finite() && goal_survival_weight >= 0.0,
                "validated survival weight is finite and non-negative"
            );
            let terminal_geometry = ((&forward.terminal_goal_by_goal
                + (&forward.surviving_goal_by_goal * goal_survival_weight)?)?
                / (1.0 + goal_survival_weight))?;
            let terminal = ((terminal_geometry * per_goal_weight)?.mean_all()? * rho_goal)?;
            let producer = ((&forward.relaxed_goal_by_goal * per_goal_weight)?.mean_all()?
                * (0.1 * rho_goal))?;
            (terminal + producer)?
        }
    };
    let protected_precondition = (forward
        .failed_precondition_by_step
        .broadcast_mul(&(protect_consumer.ones_like()? - protect_consumer)?)?
        + forward
            .support_only_failed_precondition_by_step
            .broadcast_mul(protect_consumer)?)?;
    let precondition = (protected_precondition * row_focus)?.mean_all()?;
    loss = (loss + (precondition * rho_precondition)?)?;
    loss = (loss + (&forward.premature_noop.mean_all()? * rho_goal)?)?;
    Ok(loss)
}

/// Directional teacher loss `KL(stop_gradient(Q) || P)` per particle.
///
/// Q is deliberately detached.  During transfer it is a fixed causal proof,
/// not an equal partner that can be dragged back toward an arbitrary execution
/// plan.  The loss remains ordinary gradient descent on P and is zero exactly
/// when the two strictly-positive action distributions agree.
fn teacher_kl_per_particle(execution: &Tensor, teacher: &Tensor) -> CandleResult<Tensor> {
    assert_eq!(
        execution.dims(),
        teacher.dims(),
        "execution and causal action distributions have the same shape"
    );
    assert_eq!(
        execution.rank(),
        3,
        "action distributions have shape [M, H, A]"
    );
    let particles = execution.dim(0)?;
    let execution = execution.clamp(1e-300, 1.0)?;
    let teacher = teacher.detach().clamp(1e-300, 1.0)?;
    let log_ratio = (teacher.log()? - execution.log()?)?;
    (teacher * log_ratio)?
        .sum(2)?
        .mean(1)?
        .relu()?
        .reshape((particles, 1))
}

fn global_progress(update: usize, updates: usize) -> f64 {
    if updates <= 1 {
        1.0
    } else {
        update as f64 / (updates - 1) as f64
    }
}

fn ramp(progress: f64, start: f64, end: f64) -> f64 {
    assert!(start < end, "continuation interval must be nonempty");
    ((progress - start) / (end - start)).clamp(0.0, 1.0)
}

fn teacher_weight_at(config: &SgdConfig, progress: f64) -> f64 {
    if progress < config.causal_proof_end || progress >= config.causal_transfer_end {
        return 0.0;
    }
    let peak = (config.causal_proof_end + config.causal_transfer_end) / 2.0;
    if progress < peak {
        config.teacher_weight * ramp(progress, config.causal_proof_end, peak)
    } else {
        config.teacher_weight * (1.0 - ramp(progress, peak, config.causal_transfer_end))
    }
}

fn exact_integrality_scale(config: &SgdConfig, progress: f64) -> f64 {
    config.integrality_scale_at(progress)
}

/// Coarse loss-family schedule selected from exact replay diagnostics.
///
/// Every phase keeps every core family nonzero. In particular, applicability
/// repair never erases goal/causal pressure, because doing so would remove the
/// very producer whose prerequisites need to be established.
#[derive(Debug, Clone, Copy)]
struct PhaseLossWeights {
    precondition: f64,
    transition: f64,
    goal: f64,
    causal: f64,
}

fn phase_loss_weights(phase: Phase) -> PhaseLossWeights {
    match phase {
        Phase::BuildApplicability => PhaseLossWeights {
            precondition: 1.5,
            transition: 1.25,
            goal: 0.75,
            causal: 1.5,
        },
        Phase::Goal => PhaseLossWeights {
            precondition: 0.5,
            transition: 1.0,
            goal: 2.0,
            causal: 2.5,
        },
        Phase::GoalRepair => PhaseLossWeights {
            precondition: 0.75,
            transition: 1.5,
            goal: 1.75,
            causal: 2.0,
        },
    }
}

fn goal_bridge_is_active(phase: Phase, missing_goal_mask: &[bool]) -> bool {
    !matches!(phase, Phase::BuildApplicability) && missing_goal_mask.iter().any(|&missing| missing)
}

/// Mean of the largest `fraction` of residuals across all families.
///
/// A mean alone lets one decisive violation hide among thousands of nearly
/// satisfied constraints; this puts the pressure back on the bottleneck. The
/// selection is made on detached values and applied to the live tensors, so the
/// gradient flows only through the selected entries.
fn top_residual_loss(residuals: &[&Tensor], fraction: f64) -> CandleResult<Option<Tensor>> {
    if fraction <= 0.0 {
        return Ok(None);
    }
    let flattened: Vec<Tensor> = residuals
        .iter()
        .map(|tensor| tensor.flatten_all())
        .collect::<CandleResult<_>>()?;
    let all = Tensor::cat(&flattened, 0)?;
    let total = all.dim(0)?;
    if total == 0 {
        return Ok(None);
    }
    let count = ((total as f64 * fraction).ceil() as usize).clamp(1, total);

    let mut values = all.to_vec1::<f64>()?;
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).expect("residuals are never NaN"));
    let threshold = sorted[count - 1];

    // Mask the top `count` entries, breaking ties by first occurrence so the
    // mask size is exact rather than dependent on duplicates.
    let mut remaining = count;
    for value in values.iter_mut() {
        if remaining > 0 && *value >= threshold {
            *value = 1.0;
            remaining -= 1;
        } else {
            *value = 0.0;
        }
    }
    let mask = Tensor::from_vec(values, total, all.device())?;
    Ok(Some(((all * mask)?.sum_all()? / count as f64)?))
}

/// Standard normal samples from an explicit ChaCha stream.
///
/// Randomness never comes from candle: reproducibility depends on every draw
/// coming from a stream we seed and advance ourselves.
fn normal_vec(rng: &mut ChaCha8Rng, len: usize, scale: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let u1: f64 = rng.gen_range(f64::MIN_POSITIVE..1.0);
        let u2: f64 = rng.gen_range(0.0..1.0);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        out.push(radius * angle.cos() * scale);
        if out.len() < len {
            out.push(radius * angle.sin() * scale);
        }
    }
    out
}

/// Full-support initialization for factorized optional action slots.
///
/// Ordinary rows use the historical categorical logits. On every configured
/// slack row, the first `num_actions - 1` coordinates become conditional
/// real-action logits and the final coordinate becomes occupancy. Keeping
/// every coordinate finite retains full support in both parameterizations.
fn initial_action_vec(
    rng: &mut ChaCha8Rng,
    horizon: usize,
    num_actions: usize,
    noop_gap: f64,
    slack_window: usize,
    slack_gap: f64,
) -> Vec<f64> {
    assert!(num_actions >= 1, "the transcription always appends a no-op");
    let mut values = normal_vec(rng, horizon * num_actions, 1.0);
    for (timestep, row) in values.chunks_exact_mut(num_actions).enumerate() {
        let slack =
            slack_window > 0 && horizon >= slack_window && (timestep + 1) % slack_window == 0;
        if slack {
            // The last coordinate is occupancy on a reserved insertion row.
            row[num_actions - 1] -= noop_gap + slack_gap;
        } else {
            // Ordinary rows retain the historical categorical initialization.
            let half_gap = noop_gap / 2.0;
            for real in &mut row[..num_actions - 1] {
                *real -= half_gap;
            }
            row[num_actions - 1] += half_gap;
        }
    }
    values
}

/// Independently sample one valid source ticket for every causal witness.
///
/// Source zero is the initial state and source `s + 1` is action row `s`, so a
/// consumer row `t` may sample exactly `0..=t`. No operator is selected here:
/// the source loss determines which producer, if any, should occupy the ticket.
fn initial_link_vec(rng: &mut ChaCha8Rng, horizon: usize, num_facts: usize, bias: f64) -> Vec<f64> {
    let rows = horizon + 1;
    let mut values = normal_vec(rng, rows * num_facts * rows, 0.2);
    if bias == 0.0 {
        return values;
    }
    for consumer in 0..rows {
        for fact in 0..num_facts {
            let source = rng.gen_range(0..=consumer);
            values[(consumer * num_facts + fact) * rows + source] += bias;
        }
    }
    values
}

/// Gaussian noise whose scale follows each particle's cycle phase.
fn per_particle_noise(
    streams: &mut [ChaCha8Rng],
    phases: &[f64],
    config: &SgdConfig,
    shape: (usize, usize, usize),
    device: &Device,
) -> CandleResult<Tensor> {
    let (particles, horizon, inner) = shape;
    let per_particle = horizon * inner;
    let mut values = Vec::with_capacity(particles * per_particle);
    for (particle, &phase) in phases.iter().enumerate().take(particles) {
        let scale = config.noise_at(phase);
        values.extend(normal_vec(&mut streams[particle], per_particle, scale));
    }
    Tensor::from_vec(values, shape, device)
}

/// One particle's decoded action sequence, as transcription action indices.
fn decode_particle(action: &Tensor, particle: usize) -> CandleResult<Vec<usize>> {
    let row = action.get(particle)?.argmax(1)?;
    Ok(row
        .to_vec1::<u32>()?
        .into_iter()
        .map(|a| a as usize)
        .collect())
}

/// Project one square soft token schedule onto a maximum-weight bijection.
///
/// The returned vector is indexed by execution row and contains the unique
/// token assigned to that row. This is a decoder only: optimization still
/// differentiates through the complete Sinkhorn matrix. Projecting the
/// temporal order cannot choose an operator because token action identities
/// are decoded independently.
fn maximum_weight_bijection(weights: &[Vec<f64>]) -> Vec<usize> {
    let n = weights.len();
    assert!(n > 0, "a temporal schedule has at least one token");
    assert!(
        weights.iter().all(|row| row.len() == n),
        "a temporal schedule is square"
    );
    assert!(
        weights.iter().flatten().all(|value| value.is_finite()),
        "a temporal schedule contains only finite weights"
    );

    // Hungarian algorithm for minimum cost, applied to cost = -weight.
    // `p[column]` is the token currently assigned to that execution column.
    let mut token_potential = vec![0.0f64; n + 1];
    let mut row_potential = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];
    for token in 1..=n {
        p[0] = token;
        let mut column = 0usize;
        let mut min_value = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[column] = true;
            let active_token = p[column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0usize;
            for candidate in 1..=n {
                if used[candidate] {
                    continue;
                }
                let reduced_cost = -weights[active_token - 1][candidate - 1]
                    - token_potential[active_token]
                    - row_potential[candidate];
                if reduced_cost < min_value[candidate] {
                    min_value[candidate] = reduced_cost;
                    way[candidate] = column;
                }
                if min_value[candidate] < delta {
                    delta = min_value[candidate];
                    next_column = candidate;
                }
            }
            assert!(delta.is_finite(), "a finite perfect assignment exists");
            for candidate in 0..=n {
                if used[candidate] {
                    token_potential[p[candidate]] += delta;
                    row_potential[candidate] -= delta;
                } else {
                    min_value[candidate] -= delta;
                }
            }
            column = next_column;
            if p[column] == 0 {
                break;
            }
        }
        loop {
            let previous = way[column];
            p[column] = p[previous];
            column = previous;
            if column == 0 {
                break;
            }
        }
    }

    let execution_to_token = (1..=n).map(|column| p[column] - 1).collect::<Vec<_>>();
    let mut seen = vec![false; n];
    for &token in &execution_to_token {
        assert!(token < n && !seen[token], "projection is a token bijection");
        seen[token] = true;
    }
    execution_to_token
}

/// Decode temporal tokens without allowing row-wise argmax collisions.
#[cfg(test)]
fn decode_temporal_particle(
    token_action: &Tensor,
    assignment: &Tensor,
    particle: usize,
) -> CandleResult<Vec<usize>> {
    let token_actions = decode_particle(token_action, particle)?;
    let execution_to_token = temporal_execution_to_token(assignment, particle)?;
    assert_eq!(
        execution_to_token.len(),
        token_actions.len(),
        "schedule and token horizon agree"
    );
    Ok(execution_to_token
        .into_iter()
        .map(|token| token_actions[token])
        .collect())
}

fn temporal_execution_to_token(assignment: &Tensor, particle: usize) -> CandleResult<Vec<usize>> {
    Ok(maximum_weight_bijection(
        &assignment.get(particle)?.to_vec2::<f64>()?,
    ))
}

fn accumulate_log_score(target: &mut Option<Tensor>, candidate: Tensor) -> CandleResult<()> {
    *target = Some(match target.take() {
        Some(existing) => Tensor::stack(&[&existing, &candidate], 0)?.log_sum_exp(0)?,
        None => candidate,
    });
    Ok(())
}

/// Deterministically enumerate mixed-radix permutations across particles.
///
/// For `n` goals, the first `n!` particle indices cover every order exactly
/// once. This is representational overparameterization only: no particle is
/// selected and no action identity is chosen here.
fn permute_goal_order(goals: &[usize], particle: usize) -> Vec<usize> {
    let mut remaining = goals.to_vec();
    // Alternate the low and high ends of factoradic order. Adjacent particles
    // then represent maximally different orders (particle zero is the input
    // order, particle one its reverse), while the first n! particles still
    // cover every permutation exactly once. The complemented mixed-radix
    // digits avoid computing n!, which would overflow for large goal sets.
    let complement = particle % 2 == 1;
    let mut code = particle / 2;
    let mut order = Vec::with_capacity(goals.len());
    while !remaining.is_empty() {
        let radix = remaining.len();
        let digit = code % radix;
        code /= radix;
        let index = if complement { radix - 1 - digit } else { digit };
        order.push(remaining.remove(index));
    }
    order
}

/// Remove the latest unused repair token that is structurally before a
/// consumer in the immutable repair stream.
fn take_latest_preceding_token(
    unused: &mut Vec<usize>,
    repair_order: &[usize],
    consumer: usize,
) -> Option<usize> {
    let consumer_index = repair_order
        .iter()
        .position(|&token| token == consumer)
        .expect("an obligated consumer belongs to the repair stream");
    let candidate = unused
        .iter()
        .enumerate()
        .filter_map(|(unused_index, &token)| {
            let repair_index = repair_order
                .iter()
                .position(|&candidate| candidate == token)
                .expect("unused capacity belongs to the repair stream");
            (repair_index < consumer_index).then_some((unused_index, repair_index))
        })
        .max_by_key(|&(_, repair_index)| repair_index)
        .map(|(unused_index, _)| unused_index)?;
    Some(unused.remove(candidate))
}

/// Find the latest existing fact role that can structurally feed a consumer.
fn latest_preceding_obligation(
    obligations: &[Option<usize>],
    repair_order: &[usize],
    consumer: usize,
    fact: usize,
) -> Option<usize> {
    let consumer_index = repair_order
        .iter()
        .position(|&token| token == consumer)
        .expect("an obligated consumer belongs to the repair stream");
    repair_order[..consumer_index]
        .iter()
        .rev()
        .copied()
        .find(|&token| obligations[token] == Some(fact))
}

/// Select a fact-role token that can precede an invalidated scaffold action.
///
/// Existing roles are reused before unused capacity. Within either class take
/// the latest feasible token, preserving the largest possible prefix of repair
/// capacity for prerequisites of the selected achiever. Candidates that would
/// close a cross-stream precedence cycle are skipped.
fn scaffold_repair_candidate(
    obligations: &[Option<usize>],
    unused: &[usize],
    scaffold_order: &[usize],
    repair_order: &[usize],
    precedence: &[(usize, usize)],
    consumer: usize,
    fact: usize,
) -> Option<(usize, bool)> {
    repair_order
        .iter()
        .rev()
        .copied()
        .filter(|&candidate| obligations[candidate] == Some(fact))
        .map(|candidate| (candidate, false))
        .chain(
            repair_order
                .iter()
                .rev()
                .copied()
                .filter(|candidate| unused.contains(candidate))
                .map(|candidate| (candidate, true)),
        )
        .find(|&(candidate, _)| {
            let mut candidate_precedence = precedence.to_vec();
            try_add_temporal_precedence(
                &mut candidate_precedence,
                scaffold_order,
                repair_order,
                (candidate, consumer),
            )
        })
}

/// Whether `target` is downstream of `source` in the current repair DAG.
fn precedence_reaches(edges: &[(usize, usize)], source: usize, target: usize) -> bool {
    let mut frontier = vec![source];
    let mut visited = BTreeSet::new();
    while let Some(node) = frontier.pop() {
        if !visited.insert(node) {
            continue;
        }
        for &(_, successor) in edges.iter().filter(|&&(from, _)| from == node) {
            if successor == target {
                return true;
            }
            frontier.push(successor);
        }
    }
    false
}

/// A required fact repeats downstream on the same causal branch.
fn repeated_fact_cycle(
    obligations: &[Option<usize>],
    edges: &[(usize, usize)],
    consumer: usize,
    fact: usize,
) -> bool {
    obligations.iter().enumerate().any(|(token, &obligation)| {
        obligation == Some(fact) && precedence_reaches(edges, consumer, token)
    })
}

/// Whether a selected action's achievement of `target_fact` still requires a
/// previously assigned prerequisite fact.
///
/// Operator preconditions apply to every effect. Conditional-effect facts are
/// required only when every effect of this action that can establish the
/// target contains that condition. `None` means the selected action is not an
/// achiever, so its existing causal memory must be retained until the factual
/// role itself is repaired.
fn achievement_requires_fact(
    transcription: &Transcription,
    action_preconditions: &[Vec<usize>],
    action: usize,
    target_fact: usize,
    prerequisite_fact: usize,
) -> Option<bool> {
    if action_preconditions[action].contains(&prerequisite_fact) {
        return Some(true);
    }
    let matching_effects = transcription
        .group_action()
        .iter()
        .enumerate()
        .filter(|&(_, &group_action)| group_action as usize == action)
        .flat_map(|(group, _)| transcription.group_effects(group).iter().copied())
        .filter(|&effect| transcription.effect_fact()[effect as usize] as usize == target_fact)
        .collect::<Vec<_>>();
    if matching_effects.is_empty() {
        return None;
    }
    Some(matching_effects.iter().all(|&effect| {
        transcription
            .cond_effect()
            .iter()
            .zip(transcription.cond_fact())
            .any(|(&condition_effect, &condition_fact)| {
                condition_effect == effect && condition_fact as usize == prerequisite_fact
            })
    }))
}

/// Remove prerequisite memory made stale by a repair token changing achiever.
///
/// Causal edges have action-specific provenance. Once the selected action is a
/// valid achiever of its role, any predecessor fact it no longer requires is
/// obsolete. Removing such edges can orphan upstream working-memory tokens;
/// those are recursively returned to symmetric unused capacity. Goal roles
/// and producer-to-scaffold repairs remain live by construction.
#[allow(clippy::too_many_arguments)]
fn prune_stale_causal_memory(
    transcription: &Transcription,
    action_preconditions: &[Vec<usize>],
    selected_action_by_token: &[usize],
    goal_tokens: &[(usize, usize, usize)],
    repair_order: &[usize],
    obligations: &mut [Option<usize>],
    obligation_focus: &mut [f64],
    unused: &mut Vec<usize>,
    precedence: &mut Vec<(usize, usize)>,
    causal_precedence: &mut Vec<(usize, usize)>,
) -> Vec<usize> {
    assert_eq!(selected_action_by_token.len(), obligations.len());
    assert_eq!(obligation_focus.len(), obligations.len());
    let stale = causal_precedence
        .iter()
        .copied()
        .filter(|&(producer, consumer)| {
            let (Some(prerequisite), Some(target)) = (obligations[producer], obligations[consumer])
            else {
                return false;
            };
            achievement_requires_fact(
                transcription,
                action_preconditions,
                selected_action_by_token[consumer],
                target,
                prerequisite,
            ) == Some(false)
        })
        .collect::<BTreeSet<_>>();
    causal_precedence.retain(|edge| !stale.contains(edge));
    precedence.retain(|edge| !stale.contains(edge));

    let permanent_goals = goal_tokens
        .iter()
        .map(|&(token, _, _)| token)
        .collect::<BTreeSet<_>>();
    let mut freed = Vec::new();
    loop {
        let orphan = repair_order.iter().copied().find(|&token| {
            obligations[token].is_some()
                && !permanent_goals.contains(&token)
                && !causal_precedence
                    .iter()
                    .any(|&(producer, _)| producer == token)
        });
        let Some(orphan) = orphan else {
            break;
        };
        obligations[orphan] = None;
        obligation_focus[orphan] = 1.0;
        causal_precedence.retain(|&(producer, consumer)| producer != orphan && consumer != orphan);
        // Every edge involving a repair token is dynamic: immutable scaffold
        // order contains scaffold tokens only.
        precedence.retain(|&(producer, consumer)| producer != orphan && consumer != orphan);
        if !unused.contains(&orphan) {
            unused.push(orphan);
        }
        freed.push(orphan);
    }
    unused.sort_by_key(|token| {
        repair_order
            .iter()
            .position(|candidate| candidate == token)
            .expect("unused capacity belongs to the repair stream")
    });
    freed
}

/// Keep every factual achiever symmetric except one verifier-rejected choice.
///
/// The rejection lasts for only one verifier interval. It therefore supplies a
/// direct gradient away from the currently cyclic choice without permanently
/// deleting any operator from the bounded-plan representation or selecting its
/// replacement on the host.
fn achievers_except_rejected(achievers: &[usize], rejected: Option<usize>) -> Vec<usize> {
    let Some(rejected) = rejected else {
        return achievers.to_vec();
    };
    assert!(
        achievers.binary_search(&rejected).is_ok(),
        "a cycle-rejected action must achieve the token's obligated fact"
    );
    assert!(
        achievers.len() >= 2,
        "cycle rejection requires another factual achiever"
    );
    achievers
        .iter()
        .copied()
        .filter(|&action| action != rejected)
        .collect()
}

/// Whether two immutable token streams admit an interleaving satisfying every
/// precedence edge.
///
/// Same-stream order is fixed once a temporal scaffold is admitted. Cross-
/// stream edges remove lattice transitions until their producer has been
/// consumed. This host-side Boolean dynamic program is used only when exact
/// replay proposes a new structural edge; it prevents contradictory feedback
/// from entering the differentiable scheduler.
fn temporal_interleaving_exists(
    scaffold: &[usize],
    repair: &[usize],
    precedence: &[(usize, usize)],
) -> bool {
    let horizon = scaffold.len() + repair.len();
    let mut stream_position = vec![None::<(bool, usize)>; horizon];
    for (is_repair, stream) in [(false, scaffold), (true, repair)] {
        for (position, &token) in stream.iter().enumerate() {
            assert!(token < horizon, "temporal token is within the horizon");
            assert!(
                stream_position[token]
                    .replace((is_repair, position))
                    .is_none(),
                "temporal streams partition token identities"
            );
        }
    }
    assert!(
        stream_position.iter().all(Option::is_some),
        "temporal streams cover every token identity"
    );

    for &(producer, consumer) in precedence {
        let (producer_repair, producer_position) = stream_position
            .get(producer)
            .and_then(|position| *position)
            .expect("precedence producer is a temporal token");
        let (consumer_repair, consumer_position) = stream_position
            .get(consumer)
            .and_then(|position| *position)
            .expect("precedence consumer is a temporal token");
        if producer_repair == consumer_repair && producer_position >= consumer_position {
            return false;
        }
    }

    let transition_allowed = |consumer: usize, consumed_scaffold: usize, consumed_repair: usize| {
        precedence.iter().all(|&(producer, target)| {
            if target != consumer {
                return true;
            }
            let (producer_repair, producer_position) =
                stream_position[producer].expect("producer position exists");
            if producer_repair {
                producer_position < consumed_repair
            } else {
                producer_position < consumed_scaffold
            }
        })
    };
    let mut reachable = vec![vec![false; repair.len() + 1]; scaffold.len() + 1];
    reachable[0][0] = true;
    for consumed_scaffold in 0..=scaffold.len() {
        for consumed_repair in 0..=repair.len() {
            if !reachable[consumed_scaffold][consumed_repair] {
                continue;
            }
            if let Some(&consumer) = scaffold.get(consumed_scaffold)
                && transition_allowed(consumer, consumed_scaffold, consumed_repair)
            {
                reachable[consumed_scaffold + 1][consumed_repair] = true;
            }
            if let Some(&consumer) = repair.get(consumed_repair)
                && transition_allowed(consumer, consumed_scaffold, consumed_repair)
            {
                reachable[consumed_scaffold][consumed_repair + 1] = true;
            }
        }
    }
    reachable[scaffold.len()][repair.len()]
}

fn try_add_temporal_precedence(
    precedence: &mut Vec<(usize, usize)>,
    scaffold: &[usize],
    repair: &[usize],
    edge: (usize, usize),
) -> bool {
    if precedence.contains(&edge) {
        return true;
    }
    precedence.push(edge);
    if temporal_interleaving_exists(scaffold, repair, precedence) {
        true
    } else {
        assert_eq!(precedence.pop(), Some(edge));
        false
    }
}

/// Forward-backward and Viterbi for particles with equal stream dimensions.
///
/// Each lattice cell is a vector over the group, reducing the autograd graph
/// from one scalar dynamic program per particle to one batched dynamic
/// program. Token identities may still differ between particles; a batched
/// permutation maps stream coordinates back to latent-token coordinates.
fn batched_interleaving_assignments(
    schedule_logits: &Tensor,
    schedule_temperature: &Tensor,
    particles: &[usize],
    scaffold_order: &[Vec<usize>],
    repair_order: &[Vec<usize>],
    precedence: &[Vec<(usize, usize)>],
    horizon: usize,
) -> CandleResult<(Tensor, Tensor)> {
    assert!(!particles.is_empty(), "an interleaving group is nonempty");
    let batch = particles.len();
    let scaffold_len = scaffold_order[particles[0]].len();
    let repair_len = repair_order[particles[0]].len();
    assert!(repair_len > 0, "the batched lattice contains repair tokens");
    assert_eq!(scaffold_len + repair_len, horizon);
    for &particle in particles {
        assert_eq!(scaffold_order[particle].len(), scaffold_len);
        assert_eq!(repair_order[particle].len(), repair_len);
    }
    assert_eq!(precedence.len(), scaffold_order.len());

    let mask_size = batch * (scaffold_len + 1) * (repair_len + 1);
    let mut scaffold_allowed = vec![1.0f64; mask_size];
    let mut repair_allowed = vec![1.0f64; mask_size];
    let mask_index = |local: usize, consumed_scaffold: usize, consumed_repair: usize| {
        (local * (scaffold_len + 1) + consumed_scaffold) * (repair_len + 1) + consumed_repair
    };
    for (local, &particle) in particles.iter().enumerate() {
        let scaffold = &scaffold_order[particle];
        let repair = &repair_order[particle];
        for consumed_scaffold in 0..=scaffold_len {
            for consumed_repair in 0..=repair_len {
                let index = mask_index(local, consumed_scaffold, consumed_repair);
                if let Some(&consumer) = scaffold.get(consumed_scaffold) {
                    scaffold_allowed[index] =
                        f64::from(!precedence[particle].iter().any(|&(producer, target)| {
                            target == consumer
                                && repair
                                    .iter()
                                    .position(|&token| token == producer)
                                    .is_some_and(|producer_index| producer_index >= consumed_repair)
                        }));
                }
                if let Some(&consumer) = repair.get(consumed_repair) {
                    repair_allowed[index] =
                        f64::from(!precedence[particle].iter().any(|&(producer, target)| {
                            target == consumer
                                && scaffold
                                    .iter()
                                    .position(|&token| token == producer)
                                    .is_some_and(|producer_index| {
                                        producer_index >= consumed_scaffold
                                    })
                        }));
                }
            }
        }
    }
    let mask_shape = (batch, scaffold_len + 1, repair_len + 1);
    let scaffold_allowed_tensor = Tensor::from_vec(
        scaffold_allowed.clone(),
        mask_shape,
        schedule_logits.device(),
    )?;
    let repair_allowed_tensor =
        Tensor::from_vec(repair_allowed.clone(), mask_shape, schedule_logits.device())?;

    let particle_index = Tensor::from_vec(
        particles.iter().map(|&particle| particle as u32).collect(),
        batch,
        schedule_logits.device(),
    )?;
    let logits = schedule_logits.index_select(&particle_index, 0)?;
    let temperature = schedule_temperature
        .index_select(&particle_index, 0)?
        .reshape(batch)?;
    let zero = Tensor::zeros(batch, DTYPE, schedule_logits.device())?;
    let allowed_potential =
        |mask: &Tensor, consumed_scaffold: usize, consumed_repair: usize| -> CandleResult<Tensor> {
            let allowed = mask
                .narrow(1, consumed_scaffold, 1)?
                .narrow(2, consumed_repair, 1)?
                .reshape(batch)?;
            (allowed.ones_like()? - allowed)? * -1e300
        };
    let scaffold_edge =
        |consumed_scaffold: usize, consumed_repair: usize| -> CandleResult<Tensor> {
            allowed_potential(&scaffold_allowed_tensor, consumed_scaffold, consumed_repair)
        };
    let repair_edge = |consumed_scaffold: usize, consumed_repair: usize| -> CandleResult<Tensor> {
        let score = logits
            .narrow(1, consumed_scaffold, 1)?
            .narrow(2, consumed_repair, 1)?
            .reshape(batch)?
            .broadcast_div(&temperature)?;
        score + allowed_potential(&repair_allowed_tensor, consumed_scaffold, consumed_repair)?
    };

    let mut alpha = vec![vec![None::<Tensor>; repair_len + 1]; scaffold_len + 1];
    alpha[0][0] = Some(zero.clone());
    for row in 0..horizon {
        let minimum_repair = row.saturating_sub(scaffold_len);
        let maximum_repair = row.min(repair_len);
        for consumed_repair in minimum_repair..=maximum_repair {
            let consumed_scaffold = row - consumed_repair;
            let Some(path_score) = alpha[consumed_scaffold][consumed_repair].clone() else {
                continue;
            };
            if consumed_scaffold < scaffold_len {
                accumulate_log_score(
                    &mut alpha[consumed_scaffold + 1][consumed_repair],
                    (&path_score + scaffold_edge(consumed_scaffold, consumed_repair)?)?,
                )?;
            }
            if consumed_repair < repair_len {
                accumulate_log_score(
                    &mut alpha[consumed_scaffold][consumed_repair + 1],
                    (&path_score + repair_edge(consumed_scaffold, consumed_repair)?)?,
                )?;
            }
        }
    }
    let log_partition = alpha[scaffold_len][repair_len]
        .clone()
        .expect("the batched interleaving lattice reaches its terminal state");

    let mut beta = vec![vec![None::<Tensor>; repair_len + 1]; scaffold_len + 1];
    beta[scaffold_len][repair_len] = Some(zero.clone());
    for consumed_scaffold in (0..=scaffold_len).rev() {
        for consumed_repair in (0..=repair_len).rev() {
            if consumed_scaffold == scaffold_len && consumed_repair == repair_len {
                continue;
            }
            let mut value = None;
            if consumed_scaffold < scaffold_len {
                accumulate_log_score(
                    &mut value,
                    (scaffold_edge(consumed_scaffold, consumed_repair)?
                        + beta[consumed_scaffold + 1][consumed_repair]
                            .clone()
                            .expect("scaffold successor reaches the terminal"))?,
                )?;
            }
            if consumed_repair < repair_len {
                accumulate_log_score(
                    &mut value,
                    (repair_edge(consumed_scaffold, consumed_repair)?
                        + beta[consumed_scaffold][consumed_repair + 1]
                            .clone()
                            .expect("repair successor reaches the terminal"))?,
                )?;
            }
            beta[consumed_scaffold][consumed_repair] = value;
        }
    }

    let mut scaffold_cells = vec![vec![zero.clone(); horizon]; scaffold_len];
    let mut repair_cells = vec![vec![zero.clone(); horizon]; repair_len];
    for row in 0..horizon {
        let minimum_repair = row.saturating_sub(scaffold_len);
        let maximum_repair = row.min(repair_len);
        for consumed_repair in minimum_repair..=maximum_repair {
            let consumed_scaffold = row - consumed_repair;
            let prefix = alpha[consumed_scaffold][consumed_repair]
                .clone()
                .expect("every batched row state is reachable");
            if consumed_scaffold < scaffold_len {
                scaffold_cells[consumed_scaffold][row] = ((&prefix
                    + scaffold_edge(consumed_scaffold, consumed_repair)?)?
                    + beta[consumed_scaffold + 1][consumed_repair]
                        .clone()
                        .expect("scaffold suffix exists")
                    - &log_partition)?
                    .exp()?;
            }
            if consumed_repair < repair_len {
                repair_cells[consumed_repair][row] = (((&prefix
                    + repair_edge(consumed_scaffold, consumed_repair)?)?
                    + beta[consumed_scaffold][consumed_repair + 1]
                        .clone()
                        .expect("repair suffix exists"))?
                    - &log_partition)?
                    .exp()?;
            }
        }
    }
    let stack_stream = |cells: &[Vec<Tensor>]| -> CandleResult<Tensor> {
        let tokens = cells
            .iter()
            .map(|rows| Tensor::stack(&rows.iter().collect::<Vec<_>>(), 1))
            .collect::<CandleResult<Vec<_>>>()?;
        Tensor::stack(&tokens.iter().collect::<Vec<_>>(), 1)
    };
    let repair_soft = stack_stream(&repair_cells)?;
    let stream_soft = if scaffold_cells.is_empty() {
        repair_soft
    } else {
        Tensor::cat(&[&stack_stream(&scaffold_cells)?, &repair_soft], 1)?
    };
    assert_eq!(stream_soft.dims(), &[batch, horizon, horizon]);

    let mut stream_to_token = vec![0.0f64; batch * horizon * horizon];
    for (local, &particle) in particles.iter().enumerate() {
        for (stream, &token) in scaffold_order[particle]
            .iter()
            .chain(&repair_order[particle])
            .enumerate()
        {
            stream_to_token[(local * horizon + token) * horizon + stream] = 1.0;
        }
    }
    let stream_to_token = Tensor::from_vec(
        stream_to_token,
        (batch, horizon, horizon),
        schedule_logits.device(),
    )?;
    let soft = stream_to_token.matmul(&stream_soft)?;

    let edge_values = logits.to_vec3::<f64>()?;
    let mut hard_values = vec![0.0f64; batch * horizon * horizon];
    for (local, &particle) in particles.iter().enumerate() {
        let scaffold = &scaffold_order[particle];
        let repair = &repair_order[particle];
        let mut best = vec![vec![f64::NEG_INFINITY; repair_len + 1]; scaffold_len + 1];
        let mut take_repair = vec![vec![false; repair_len + 1]; scaffold_len + 1];
        best[scaffold_len][repair_len] = 0.0;
        for consumed_scaffold in (0..=scaffold_len).rev() {
            for consumed_repair in (0..=repair_len).rev() {
                if consumed_scaffold == scaffold_len && consumed_repair == repair_len {
                    continue;
                }
                let allowed_index = mask_index(local, consumed_scaffold, consumed_repair);
                let scaffold_score = (consumed_scaffold < scaffold_len
                    && scaffold_allowed[allowed_index] > 0.5)
                    .then(|| best[consumed_scaffold + 1][consumed_repair]);
                let repair_score = (consumed_repair < repair_len
                    && repair_allowed[allowed_index] > 0.5)
                    .then(|| {
                        edge_values[local][consumed_scaffold][consumed_repair]
                            + best[consumed_scaffold][consumed_repair + 1]
                    });
                match (scaffold_score, repair_score) {
                    (Some(scaffold_score), Some(repair_score)) => {
                        take_repair[consumed_scaffold][consumed_repair] =
                            repair_score > scaffold_score;
                        best[consumed_scaffold][consumed_repair] = scaffold_score.max(repair_score);
                    }
                    (Some(score), None) => best[consumed_scaffold][consumed_repair] = score,
                    (None, Some(score)) => {
                        take_repair[consumed_scaffold][consumed_repair] = true;
                        best[consumed_scaffold][consumed_repair] = score;
                    }
                    // A precedence mask can make an intermediate lattice state
                    // unreachable even though another complete interleaving
                    // remains. Keep its value at negative infinity; only the
                    // start state is required to reach the terminal.
                    (None, None) => {}
                }
            }
        }
        assert!(
            best[0][0].is_finite(),
            "temporal cross-stream precedence constraints retain a complete interleaving"
        );
        let (mut consumed_scaffold, mut consumed_repair) = (0usize, 0usize);
        for row in 0..horizon {
            let token = if take_repair[consumed_scaffold][consumed_repair] {
                let token = repair[consumed_repair];
                consumed_repair += 1;
                token
            } else {
                let token = scaffold[consumed_scaffold];
                consumed_scaffold += 1;
                token
            };
            hard_values[(local * horizon + token) * horizon + row] = 1.0;
        }
        assert_eq!(
            (consumed_scaffold, consumed_repair),
            (scaffold_len, repair_len)
        );
    }
    let hard = Tensor::from_vec(
        hard_values,
        (batch, horizon, horizon),
        schedule_logits.device(),
    )?;
    let assignment = (&soft + (&hard - &soft)?.detach())?;
    Ok((assignment, soft))
}

/// Merge an immutable scaffold with an ordered stream of repair tokens.
///
/// A state `(i, j)` means that the first `i` scaffold tokens and first `j`
/// repair tokens have executed. Its only successors consume the next token
/// from one of those streams. A repair edge at `(i, j)` has log-potential
/// `schedule_logits[i, j]`; scaffold edges have zero potential. Globally
/// normalizing complete path scores therefore represents all and only
/// order-preserving interleavings. Unlike locally normalized Bernoulli gates,
/// assigning a repair token to a late gap does not require surviving a product
/// of probabilities for every earlier scaffold edge.
///
/// Exact execution follows the maximum-score lattice path while autograd sees
/// forward-backward edge marginals. Unlike an unrestricted straight-through permutation, both
/// the hard and soft schedules preserve the verifier-proven scaffold order and
/// the prerequisite-to-consumer repair order.
fn monotone_interleaving_schedule(
    token_action: Tensor,
    schedule_logits: &Tensor,
    schedule_temperature: &Tensor,
    scaffold_order: &[Vec<usize>],
    repair_order: &[Vec<usize>],
    precedence: &[Vec<(usize, usize)>],
) -> CandleResult<TemporalTokenDistribution> {
    let [particles, horizon, actions]: [usize; 3] = token_action
        .dims()
        .try_into()
        .expect("token action distribution has rank three");
    assert_eq!(
        schedule_logits.dims(),
        &[particles, horizon, horizon],
        "interleaving gates use the temporal schedule tensor"
    );
    assert_eq!(
        schedule_temperature.dims(),
        &[particles, 1, 1],
        "each particle has one interleaving temperature"
    );
    assert_eq!(scaffold_order.len(), particles);
    assert_eq!(repair_order.len(), particles);
    assert_eq!(precedence.len(), particles);

    let mut particle_assignments = vec![None::<Tensor>; particles];
    let mut particle_soft_assignments = vec![None::<Tensor>; particles];
    let mut groups = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for particle in 0..particles {
        let scaffold = &scaffold_order[particle];
        let repair = &repair_order[particle];
        assert_eq!(
            scaffold.len() + repair.len(),
            horizon,
            "scaffold and repair streams partition the horizon"
        );
        let mut seen = vec![false; horizon];
        for &token in scaffold.iter().chain(repair) {
            assert!(
                token < horizon && !seen[token],
                "token streams form a partition"
            );
            seen[token] = true;
        }

        if repair.is_empty() {
            let mut exact = vec![0.0f64; horizon * horizon];
            for (row, &token) in scaffold.iter().enumerate() {
                exact[token * horizon + row] = 1.0;
            }
            let exact = Tensor::from_vec(exact, (horizon, horizon), token_action.device())?;
            particle_soft_assignments[particle] = Some(exact.clone());
            particle_assignments[particle] = Some(exact);
            continue;
        }
        groups
            .entry((scaffold.len(), repair.len()))
            .or_default()
            .push(particle);
    }
    for particles_in_group in groups.values() {
        let (assignment, soft) = batched_interleaving_assignments(
            schedule_logits,
            schedule_temperature,
            particles_in_group,
            scaffold_order,
            repair_order,
            precedence,
            horizon,
        )?;
        for (local, &particle) in particles_in_group.iter().enumerate() {
            particle_assignments[particle] = Some(assignment.get(local)?);
            particle_soft_assignments[particle] = Some(soft.get(local)?);
        }
    }

    let particle_assignments = particle_assignments
        .into_iter()
        .map(|assignment| assignment.expect("every particle receives an interleaving"))
        .collect::<Vec<_>>();
    let particle_soft_assignments = particle_soft_assignments
        .into_iter()
        .map(|assignment| assignment.expect("every particle receives soft path marginals"))
        .collect::<Vec<_>>();
    let assignment = Tensor::stack(&particle_assignments.iter().collect::<Vec<_>>(), 0)?;
    let soft_assignment = Tensor::stack(&particle_soft_assignments.iter().collect::<Vec<_>>(), 0)?;
    let action = assignment.transpose(1, 2)?.matmul(&token_action)?;
    let log_action = action.clamp(1e-300, 1.0)?.log()?;
    assert_eq!(action.dims(), &[particles, horizon, actions]);
    Ok(TemporalTokenDistribution {
        action,
        log_action,
        token_action,
        assignment,
        soft_assignment,
    })
}

/// Stop action-identity gradients outside each particle's repair stream.
///
/// The forward value is unchanged. Before temporal unlock the gate is all one,
/// so ordinary discovery remains unconstrained. Afterwards the verifier-proven
/// scaffold is a hard trust region: only reserved repair tokens can specialize.
fn repair_only_action_gradients(action: &Tensor, repair_gate: &Tensor) -> CandleResult<Tensor> {
    let [particles, horizon, _]: [usize; 3] = action
        .dims()
        .try_into()
        .expect("action distribution has rank three");
    assert_eq!(repair_gate.dims(), &[particles, horizon, 1]);
    let frozen_gate = (repair_gate.ones_like()? - repair_gate)?;
    action.broadcast_mul(repair_gate)? + action.detach().broadcast_mul(&frozen_gate)?
}

/// Execute inactive temporal memory as exact no-ops.
///
/// Unassigned capacity is not a plan variable yet. Giving it a merely soft
/// no-op penalty lets arbitrary stale logits become real actions and invalidate
/// the scaffold. Its action gradient is therefore exactly zero until verifier
/// feedback assigns a factual role and explicitly resets that token.
fn force_inactive_temporal_noops(
    action: &Tensor,
    inactive: &Tensor,
    noop_action: usize,
) -> CandleResult<Tensor> {
    let [particles, horizon, actions]: [usize; 3] = action
        .dims()
        .try_into()
        .expect("action distribution has rank three");
    assert_eq!(inactive.dims(), &[particles, horizon, 1]);
    assert!(noop_action < actions);
    let active = (inactive.ones_like()? - inactive)?;
    let mut noop = vec![0.0f64; actions];
    noop[noop_action] = 1.0;
    let noop = Tensor::from_vec(noop, (1, 1, actions), action.device())?;
    action.broadcast_mul(&active)? + noop.broadcast_mul(inactive)
}

/// Penalize violated strict precedence edges between latent action tokens.
///
/// At an integral permutation, zero loss certifies that every cross-stream
/// producer token executes at least one row before its consumer.
///
/// Comparing expected positions is insufficient: oppositely bimodal tokens
/// can have correctly ordered means while retaining substantial inverted
/// mass. Instead, sum the complete pairwise mass where `producer_row <
/// consumer_row` and minimize its negative logarithm. Edges within either
/// immutable stream are omitted because the interleaving lattice already
/// satisfies them on every hard and soft path.
fn temporal_precedence_loss(
    assignment: &Tensor,
    edges: &[Vec<(usize, usize)>],
    scaffold_order: &[Vec<usize>],
    repair_order: &[Vec<usize>],
) -> CandleResult<Tensor> {
    let [particles, horizon, rows]: [usize; 3] = assignment
        .dims()
        .try_into()
        .expect("temporal assignment has rank three");
    assert_eq!(horizon, rows, "temporal assignment is square");
    assert_eq!(
        edges.len(),
        particles,
        "every particle has precedence edges"
    );
    assert_eq!(scaffold_order.len(), particles);
    assert_eq!(repair_order.len(), particles);
    let mut producer_indices = Vec::<u32>::new();
    let mut consumer_indices = Vec::<u32>::new();
    for (particle, particle_edges) in edges.iter().enumerate() {
        for &(producer, consumer) in particle_edges {
            assert!(
                producer < horizon && consumer < horizon && producer != consumer,
                "a precedence edge connects distinct in-range tokens"
            );
            let both_scaffold = scaffold_order[particle].contains(&producer)
                && scaffold_order[particle].contains(&consumer);
            let both_repair = repair_order[particle].contains(&producer)
                && repair_order[particle].contains(&consumer);
            if both_scaffold || both_repair {
                continue;
            }
            producer_indices.push((particle * horizon + producer) as u32);
            consumer_indices.push((particle * horizon + consumer) as u32);
        }
    }
    if producer_indices.is_empty() {
        return Tensor::zeros((), DTYPE, assignment.device());
    }
    let edge_count = producer_indices.len();
    let producer_indices = Tensor::from_vec(producer_indices, edge_count, assignment.device())?;
    let consumer_indices = Tensor::from_vec(consumer_indices, edge_count, assignment.device())?;
    let flat = assignment.reshape((particles * horizon, horizon))?;
    let producer = flat.index_select(&producer_indices, 0)?;
    let consumer = flat.index_select(&consumer_indices, 0)?;
    let mut strict_upper = vec![0.0f64; horizon * horizon];
    for earlier in 0..horizon {
        for later in earlier + 1..horizon {
            strict_upper[earlier * horizon + later] = 1.0;
        }
    }
    let strict_upper = Tensor::from_vec(strict_upper, (horizon, horizon), assignment.device())?;
    let ordered_support = (producer.matmul(&strict_upper)? * consumer)?.sum(1)?;
    ordered_support.clamp(1e-300, 1.0)?.log()?.neg()?.mean_all()
}

/// Place obligated tokens where their action distribution has exact
/// applicability support in the verifier-known prefix states.
///
/// The mask contains every applicable action at every known row. Consequently
/// this loss neither selects an operator nor a position; it differentiates the
/// joint token-action and token-position mass.
fn temporal_obligation_applicability_loss(
    token_action: &Tensor,
    assignment: &Tensor,
    applicable_by_row: &Tensor,
    obligation_active: &Tensor,
    obligation_focus: &Tensor,
) -> CandleResult<Tensor> {
    let [particles, horizon, actions]: [usize; 3] = token_action
        .dims()
        .try_into()
        .expect("token action distribution has rank three");
    assert_eq!(assignment.dims(), &[particles, horizon, horizon]);
    assert_eq!(applicable_by_row.dims(), &[particles, horizon, actions]);
    assert_eq!(obligation_active.dims(), &[particles, horizon]);
    assert_eq!(obligation_focus.dims(), &[particles, horizon, 1]);
    // Action identity is governed by the factual obligation. Detach it here
    // so applicability can move the token in time but cannot erase the goal
    // achiever by replacing it with an unrelated currently applicable action.
    let action_support = token_action
        .detach()
        .matmul(&applicable_by_row.transpose(1, 2)?)?;
    let support = (action_support * assignment)?.sum(2)?;
    let active = obligation_active.sum_all()?.clamp(1.0, f64::MAX)?;
    let weight =
        obligation_active.broadcast_mul(&obligation_focus.reshape((particles, horizon))?)?;
    (support.clamp(1e-300, 1.0)?.log()?.neg()? * weight)?.sum_all()? / active
}

/// Applicability in a frozen scaffold gap after accounting for facts supplied
/// by direct repair predecessors.
///
/// This is symbolic condition evaluation, not state generation. Each action
/// receives either an exact indicator or, when `graded`, `exp(-missing)`,
/// where `missing` counts preconditions that are
/// neither true in the frozen gap nor supplied by a direct repair predecessor.
/// The score is exactly one iff the action is conditionally applicable, so a
/// zero auxiliary loss still certifies applicability, while incomplete gaps
/// retain a gradient ordered by how many prerequisites they already satisfy.
/// Exact replay remains the authority on whether the composed repair works.
#[allow(clippy::too_many_arguments)]
fn conditional_gap_applicability_masks(
    obligations: &[Vec<Option<usize>>],
    precedence: &[Vec<(usize, usize)>],
    scaffold_order: &[Vec<usize>],
    scaffold_fact_values: &[f64],
    action_preconditions: &[Vec<usize>],
    particles: usize,
    horizon: usize,
    num_facts: usize,
    num_actions: usize,
    graded: bool,
) -> Vec<Vec<Option<Vec<f64>>>> {
    assert_eq!(obligations.len(), particles);
    assert_eq!(precedence.len(), particles);
    assert_eq!(scaffold_order.len(), particles);
    assert_eq!(action_preconditions.len(), num_actions);
    assert_eq!(
        scaffold_fact_values.len(),
        particles * (horizon + 1) * num_facts
    );
    let mut masks = vec![vec![None; horizon]; particles];
    for particle in 0..particles {
        assert_eq!(obligations[particle].len(), horizon);
        for (token, obligation) in obligations[particle].iter().enumerate() {
            if obligation.is_none() {
                continue;
            }
            let supplied = precedence[particle]
                .iter()
                .filter_map(|&(producer, consumer)| {
                    (consumer == token)
                        .then_some(obligations[particle][producer])
                        .flatten()
                })
                .collect::<BTreeSet<_>>();
            let mut mask = vec![0.0f64; (horizon + 1) * num_actions];
            for gap in 0..=scaffold_order[particle].len() {
                let facts = &scaffold_fact_values[(particle * (horizon + 1) + gap) * num_facts
                    ..(particle * (horizon + 1) + gap + 1) * num_facts];
                for (action, preconditions) in action_preconditions.iter().enumerate() {
                    let missing = preconditions
                        .iter()
                        .filter(|fact| facts[**fact] <= 0.5 && !supplied.contains(fact))
                        .count();
                    mask[gap * num_actions + action] = if graded {
                        (-(missing as f64)).exp()
                    } else {
                        f64::from(missing == 0)
                    };
                }
            }
            masks[particle][token] = Some(mask);
        }
    }
    masks
}

/// Place repair tokens in verifier-known gaps of the frozen scaffold.
///
/// Repair token `j` executes at row `i + j` exactly when it is inserted after
/// `i` scaffold tokens. The scaffold replay supplies action applicability for
/// every such gap, including gaps unreachable under the repair tokens' current
/// hard placement. This avoids the zero-gradient trap where an early failing
/// repair token prevents exact replay from exposing any later supported row.
fn temporal_obligation_scaffold_gap_loss(
    token_action: &Tensor,
    assignment: &Tensor,
    achiever_by_token: &Tensor,
    conditional_applicable_by_token: &[Vec<Option<Vec<f64>>>],
    obligation_active: &Tensor,
    obligation_focus: &Tensor,
    scaffold_order: &[Vec<usize>],
    repair_order: &[Vec<usize>],
) -> CandleResult<Tensor> {
    let [particles, horizon, actions]: [usize; 3] = token_action
        .dims()
        .try_into()
        .expect("token action distribution has rank three");
    assert_eq!(assignment.dims(), &[particles, horizon, horizon]);
    assert_eq!(achiever_by_token.dims(), &[particles, horizon, actions]);
    assert_eq!(conditional_applicable_by_token.len(), particles);
    assert_eq!(obligation_active.dims(), &[particles, horizon]);
    assert_eq!(obligation_focus.dims(), &[particles, horizon, 1]);
    assert_eq!(scaffold_order.len(), particles);
    assert_eq!(repair_order.len(), particles);

    let gaps = horizon + 1;
    let mut applicable_values = vec![0.0f64; particles * horizon * gaps * actions];
    let mut row_selector_values = vec![0.0f64; particles * horizon * gaps * horizon];
    for particle in 0..particles {
        assert_eq!(
            scaffold_order[particle].len() + repair_order[particle].len(),
            horizon
        );
        assert_eq!(conditional_applicable_by_token[particle].len(), horizon);
        for (repair_index, &token) in repair_order[particle].iter().enumerate() {
            let Some(mask) = &conditional_applicable_by_token[particle][token] else {
                continue;
            };
            assert_eq!(mask.len(), gaps * actions);
            let applicable_begin = (particle * horizon + token) * gaps * actions;
            applicable_values[applicable_begin..applicable_begin + gaps * actions]
                .copy_from_slice(mask);
            for gap in 0..=scaffold_order[particle].len() {
                let execution_row = gap + repair_index;
                assert!(
                    execution_row < horizon,
                    "a nonfinal repair token fits a row"
                );
                let selector = ((particle * horizon + token) * gaps + gap) * horizon;
                row_selector_values[selector + execution_row] = 1.0;
            }
        }
    }
    let applicable = Tensor::from_vec(
        applicable_values,
        (particles, horizon, gaps, actions),
        token_action.device(),
    )?;
    let row_selector = Tensor::from_vec(
        row_selector_values,
        (particles, horizon, gaps, horizon),
        token_action.device(),
    )?;
    // One and the same action must both achieve the token's fact and be
    // applicable in the frozen gap. The separate selector maps gap `i` for
    // repair token `j` to execution row `i + j` without scalar graph nodes.
    let action_support = (token_action * achiever_by_token)?
        .unsqueeze(2)?
        .broadcast_mul(&applicable)?
        .sum(3)?;
    let placement = assignment
        .unsqueeze(2)?
        .broadcast_mul(&row_selector)?
        .sum(3)?;
    let support = (action_support * placement)?.sum(2)?;
    let weight =
        obligation_active.broadcast_mul(&obligation_focus.reshape((particles, horizon))?)?;
    let total = (support.clamp(1e-300, 1.0)?.log()?.neg()? * weight)?.sum_all()?;
    let active = obligation_active.sum_all()?.clamp(1.0, f64::MAX)?;
    total / active
}

/// Require each repair producer/consumer edge to have joint support in one gap.
///
/// Separate per-token support and equality of expected gaps admit fractional
/// cheating: two bimodal tokens can have the same expectation while never
/// sharing an executable gap. This conjunction is zero at integrality exactly
/// when both ordered tokens choose obligated achievers applicable in the same
/// frozen-scaffold gap. Their immutable repair order then makes the link
/// consecutive with respect to scaffold actions.
fn temporal_repair_edge_gap_support_loss(
    token_action: &Tensor,
    assignment: &Tensor,
    achiever_by_token: &Tensor,
    conditional_applicable_by_token: &[Vec<Option<Vec<f64>>>],
    precedence: &[Vec<(usize, usize)>],
    scaffold_order: &[Vec<usize>],
    repair_order: &[Vec<usize>],
) -> CandleResult<Tensor> {
    let [particles, horizon, actions]: [usize; 3] = token_action
        .dims()
        .try_into()
        .expect("token actions have rank three");
    assert_eq!(assignment.dims(), &[particles, horizon, horizon]);
    assert_eq!(achiever_by_token.dims(), &[particles, horizon, actions]);
    assert_eq!(conditional_applicable_by_token.len(), particles);
    assert_eq!(precedence.len(), particles);
    assert_eq!(scaffold_order.len(), particles);
    assert_eq!(repair_order.len(), particles);
    let gaps = horizon + 1;
    let mut producer_indices = Vec::<u32>::new();
    let mut consumer_indices = Vec::<u32>::new();
    let mut producer_masks = Vec::<f64>::new();
    let mut consumer_masks = Vec::<f64>::new();
    let mut producer_selectors = Vec::<f64>::new();
    let mut consumer_selectors = Vec::<f64>::new();
    for particle in 0..particles {
        for &(producer, consumer) in &precedence[particle] {
            let Some(producer_index) = repair_order[particle]
                .iter()
                .position(|&token| token == producer)
            else {
                continue;
            };
            let Some(consumer_index) = repair_order[particle]
                .iter()
                .position(|&token| token == consumer)
            else {
                continue;
            };
            assert!(
                producer_index < consumer_index,
                "repair precedence follows the immutable repair stream"
            );
            producer_indices.push((particle * horizon + producer) as u32);
            consumer_indices.push((particle * horizon + consumer) as u32);
            producer_masks.extend_from_slice(
                conditional_applicable_by_token[particle][producer]
                    .as_ref()
                    .expect("an obligated producer has a conditional gap mask"),
            );
            consumer_masks.extend_from_slice(
                conditional_applicable_by_token[particle][consumer]
                    .as_ref()
                    .expect("an obligated consumer has a conditional gap mask"),
            );
            let mut producer_selector = vec![0.0f64; gaps * horizon];
            let mut consumer_selector = vec![0.0f64; gaps * horizon];
            for gap in 0..=scaffold_order[particle].len() {
                producer_selector[gap * horizon + gap + producer_index] = 1.0;
                consumer_selector[gap * horizon + gap + consumer_index] = 1.0;
            }
            producer_selectors.extend(producer_selector);
            consumer_selectors.extend(consumer_selector);
        }
    }
    if producer_indices.is_empty() {
        return Tensor::zeros((), DTYPE, assignment.device());
    }
    let edge_count = producer_indices.len();
    let producer_indices = Tensor::from_vec(producer_indices, edge_count, assignment.device())?;
    let consumer_indices = Tensor::from_vec(consumer_indices, edge_count, assignment.device())?;
    let flat_action = token_action.reshape((particles * horizon, actions))?;
    let flat_achiever = achiever_by_token.reshape((particles * horizon, actions))?;
    let flat_assignment = assignment.reshape((particles * horizon, horizon))?;
    let producer_action = flat_action.index_select(&producer_indices, 0)?;
    let consumer_action = flat_action.index_select(&consumer_indices, 0)?;
    let producer_achiever = flat_achiever.index_select(&producer_indices, 0)?;
    let consumer_achiever = flat_achiever.index_select(&consumer_indices, 0)?;
    let producer_assignment = flat_assignment.index_select(&producer_indices, 0)?;
    let consumer_assignment = flat_assignment.index_select(&consumer_indices, 0)?;
    let producer_mask = Tensor::from_vec(
        producer_masks,
        (edge_count, gaps, actions),
        assignment.device(),
    )?;
    let consumer_mask = Tensor::from_vec(
        consumer_masks,
        (edge_count, gaps, actions),
        assignment.device(),
    )?;
    let producer_selector = Tensor::from_vec(
        producer_selectors,
        (edge_count, gaps, horizon),
        assignment.device(),
    )?;
    let consumer_selector = Tensor::from_vec(
        consumer_selectors,
        (edge_count, gaps, horizon),
        assignment.device(),
    )?;
    let producer_action_support = (producer_action * producer_achiever)?
        .unsqueeze(1)?
        .broadcast_mul(&producer_mask)?
        .sum(2)?;
    let consumer_action_support = (consumer_action * consumer_achiever)?
        .unsqueeze(1)?
        .broadcast_mul(&consumer_mask)?
        .sum(2)?;
    let producer_placement = producer_assignment
        .unsqueeze(1)?
        .broadcast_mul(&producer_selector)?
        .sum(2)?;
    let consumer_placement = consumer_assignment
        .unsqueeze(1)?
        .broadcast_mul(&consumer_selector)?
        .sum(2)?;
    let support = (producer_action_support * consumer_action_support)?
        .mul(&producer_placement)?
        .mul(&consumer_placement)?
        .sum(1)?;
    support.clamp(1e-300, 1.0)?.log()?.neg()?.mean_all()
}

/// Mean every non-row coordinate of one particle in a rank-three trace tensor.
fn particle_row_means(tensor: &Tensor, particle: usize) -> CandleResult<Vec<f64>> {
    let values = tensor.get(particle)?;
    assert_eq!(
        values.rank(),
        2,
        "removing the particle axis from a trace tensor leaves rows and features"
    );
    let rows = values.dim(0)?;
    let features = values.dim(1)?;
    if features == 0 {
        return Ok(vec![0.0; rows]);
    }
    values.mean(1)?.to_vec1::<f64>()
}

fn particle_flat_values(tensor: &Tensor, particle: usize) -> CandleResult<Vec<f64>> {
    tensor.get(particle)?.flatten_all()?.to_vec1::<f64>()
}

/// What an exact replay told us.
struct Feedback {
    /// Cost of the goal-reaching prefix, when the replay succeeded.
    solved: Option<f64>,
    /// Timestep of the earliest inapplicable action, if any.
    failure_step: Option<usize>,
    failure_fact: Option<ExplicitFact>,
    missing_goals: Vec<ExplicitFact>,
    goals_reached: usize,
    applicable_prefix: usize,
}

fn interpret(replay: &Replay, num_goals: usize) -> Feedback {
    match &replay.outcome {
        ReplayOutcome::Solved(plan) => Feedback {
            solved: Some(plan.cost),
            failure_step: None,
            failure_fact: None,
            missing_goals: Vec::new(),
            goals_reached: num_goals,
            applicable_prefix: plan.prefix_len,
        },
        ReplayOutcome::Rejected(PlanRejection::InapplicableOperator { step, fact, .. }) => {
            Feedback {
                solved: None,
                failure_step: Some(*step),
                failure_fact: Some(*fact),
                missing_goals: Vec::new(),
                goals_reached: 0,
                applicable_prefix: replay.applied,
            }
        }
        ReplayOutcome::Rejected(PlanRejection::GoalNotReached { unsatisfied }) => Feedback {
            solved: None,
            failure_step: None,
            failure_fact: None,
            missing_goals: unsatisfied.clone(),
            goals_reached: num_goals.saturating_sub(unsatisfied.len()),
            applicable_prefix: replay.applied,
        },
        ReplayOutcome::Rejected(PlanRejection::GlobalConstraintViolated { step }) => Feedback {
            solved: None,
            failure_step: Some(*step),
            failure_fact: None,
            missing_goals: Vec::new(),
            goals_reached: 0,
            applicable_prefix: replay.applied,
        },
    }
}

/// Synthesize a plan with the exact-at-integrality lifted transcription.
///
/// Every timestep has an action distribution and an independent finite-domain
/// state distribution. Local precondition and transition constraints couple
/// adjacent rows, while exact replay only validates decoded sequences and
/// schedules loss pressure; it never proposes a replacement action.
pub fn solve(
    task: TaskRef<'_>,
    global_constraint: ExplicitFact,
    config: &SgdConfig,
) -> Result<SgdOutcome, SgdError> {
    solve_direct_transcription(task, global_constraint, config)
}

/// Optimize independent action and state trajectories with local STRIPS
/// constraints. This is multiple shooting at every timestep: distant goals do
/// not have to backpropagate through one long recurrent state chain.
fn solve_direct_transcription(
    task: TaskRef<'_>,
    global_constraint: ExplicitFact,
    config: &SgdConfig,
) -> Result<SgdOutcome, SgdError> {
    config.validate().map_err(SgdError::Config)?;

    let transcription = match Transcription::build(&*task) {
        Ok(transcription) => transcription,
        Err(TranscriptionError::ProvablyUnsolvable(_)) => {
            return Ok(SgdOutcome {
                status: SgdStatus::Unsolvable,
                plan: None,
                cost: None,
                updates: 0,
                verifier_calls: 0,
                horizon_rounds: 0,
                final_horizon: 0,
                best_total_residual: f64::INFINITY,
                best_goals_reached: 0,
                num_goals: task.get_num_goals(),
                longest_applicable_prefix: 0,
                refreshes: 0,
                remelts: 0,
                temporal_restarts: 0,
                temporal_order_conflicts: 0,
                temporal_causal_cycles: 0,
                temporal_cycle_interventions: 0,
                temporal_scaffold_repairs: 0,
                backward_bridge_updates: 0,
                max_backward_bridge_loss: 0.0,
                final_diagnostics: Diagnostics::default(),
                best_exact_checkpoint: None,
                trace: Vec::new(),
                trace_fact_names: Vec::new(),
            });
        }
        Err(other) => return Err(SgdError::Transcription(other)),
    };

    let device = Device::Cpu;
    let num_goals = transcription.goal_facts().len();
    let operators: Vec<Operator> = task.get_operators().clone();
    let mut fact_achievers = vec![Vec::<usize>::new(); transcription.num_facts()];
    for (group, &action) in transcription.group_action().iter().enumerate() {
        for &effect in transcription.group_effects(group) {
            fact_achievers[transcription.effect_fact()[effect as usize] as usize]
                .push(action as usize);
        }
    }
    for achievers in &mut fact_achievers {
        achievers.sort_unstable();
        achievers.dedup();
    }
    let mut action_preconditions = vec![Vec::<usize>::new(); transcription.num_actions()];
    for (&action, &fact) in transcription
        .pre_action()
        .iter()
        .zip(transcription.pre_fact())
    {
        action_preconditions[action as usize].push(fact as usize);
    }
    let mut fact_threateners = vec![Vec::<usize>::new(); transcription.num_facts()];
    for fact in 0..transcription.num_facts() {
        let fact_var = transcription.var_of_fact()[fact];
        for (group, (&action, &group_var)) in transcription
            .group_action()
            .iter()
            .zip(transcription.group_var())
            .enumerate()
        {
            if group_var == fact_var
                && transcription
                    .group_effects(group)
                    .iter()
                    .any(|&effect| transcription.effect_fact()[effect as usize] as usize != fact)
            {
                fact_threateners[fact].push(action as usize);
            }
        }
        fact_threateners[fact].sort_unstable();
        fact_threateners[fact].dedup();
    }
    let mut registry = StateRegistry::for_task(Arc::clone(&task));

    let mut outcome = SgdOutcome {
        status: SgdStatus::BudgetExhausted,
        plan: None,
        cost: None,
        updates: 0,
        verifier_calls: 0,
        horizon_rounds: 0,
        final_horizon: 0,
        best_total_residual: f64::INFINITY,
        best_goals_reached: 0,
        num_goals,
        longest_applicable_prefix: 0,
        refreshes: 0,
        remelts: 0,
        temporal_restarts: 0,
        temporal_order_conflicts: 0,
        temporal_causal_cycles: 0,
        temporal_cycle_interventions: 0,
        temporal_scaffold_repairs: 0,
        backward_bridge_updates: 0,
        max_backward_bridge_loss: 0.0,
        final_diagnostics: Diagnostics::default(),
        best_exact_checkpoint: None,
        trace: Vec::new(),
        trace_fact_names: if config.trace_period == 0 {
            Vec::new()
        } else {
            (0..transcription.num_facts())
                .map(|fact| {
                    let local_var = transcription.var_of_fact()[fact] as usize;
                    let task_var = transcription.primary_vars()[local_var];
                    let value = fact - transcription.var_offset()[local_var] as usize;
                    let name = task.get_fact_name(&ExplicitFact::propositional(task_var, value));
                    if name.is_empty() {
                        format!("var{task_var}={value}")
                    } else {
                        name.to_string()
                    }
                })
                .collect()
        },
    };

    let mut round = 0usize;
    while let Some(horizon) = config.horizon.horizon_for_round(round) {
        round += 1;
        outcome.horizon_rounds = round;
        outcome.final_horizon = horizon;

        let plan = TensorPlan::new(&transcription, horizon, config.particles, device.clone())
            .map_err(SgdError::TensorPlan)?;
        let mut streams: Vec<ChaCha8Rng> = (0..config.particles)
            .map(|particle| {
                let mut rng = ChaCha8Rng::seed_from_u64(config.seed ^ ((round as u64) << 32));
                rng.set_stream(particle as u64);
                rng
            })
            .collect();
        let mut causal_action_streams: Vec<ChaCha8Rng> = (0..config.particles)
            .map(|particle| {
                let mut rng = ChaCha8Rng::seed_from_u64(
                    config.seed ^ ((round as u64) << 32) ^ CAUSAL_ACTION_RNG_DOMAIN,
                );
                rng.set_stream(particle as u64);
                rng
            })
            .collect();

        let mut initial_action = Vec::with_capacity(config.particles * horizon * plan.num_actions);
        let mut initial_state = Vec::with_capacity(config.particles * horizon * plan.num_facts);
        let mut initial_schedule = Vec::with_capacity(config.particles * horizon * horizon);
        for stream in streams.iter_mut() {
            initial_action.extend(initial_action_vec(
                stream,
                horizon,
                plan.num_actions,
                config.initial_noop_logit_gap,
                config.slot_slack_window,
                config.slot_slack_logit_gap,
            ));
            initial_state.extend(normal_vec(stream, horizon * plan.num_facts, 0.7));
            let mut schedule = normal_vec(stream, horizon * horizon, 0.1);
            for token in 0..horizon {
                schedule[token * horizon + token] += config.schedule_identity_bias;
            }
            initial_schedule.extend(schedule);
        }
        let initial_causal_action = initial_action.clone();
        let action_logits = Var::from_vec(
            initial_action,
            (config.particles, horizon, plan.num_actions),
            &device,
        )?;
        let state_logits = Var::from_vec(
            initial_state,
            (config.particles, horizon, plan.num_facts),
            &device,
        )?;
        let causal_action_logits = Var::from_vec(
            initial_causal_action,
            (config.particles, horizon, plan.num_actions),
            &device,
        )?;
        let schedule_logits = Var::from_vec(
            initial_schedule,
            (config.particles, horizon, horizon),
            &device,
        )?;
        let mut direct_variables = vec![action_logits.clone(), state_logits.clone()];
        if config.temporal_tokens {
            direct_variables.push(schedule_logits.clone());
        }
        let mut optimizer = Adam::new(
            direct_variables,
            AdamParams {
                learning_rate: config.learning_rate,
                grad_clip: config.grad_clip,
                particles: Some(config.particles),
                ..AdamParams::default()
            },
        )?;
        let mut link_lane = if config.causal_links_enabled() {
            let mut link_streams: Vec<ChaCha8Rng> = (0..config.particles)
                .map(|particle| {
                    let mut rng = ChaCha8Rng::seed_from_u64(
                        config.seed ^ ((round as u64) << 32) ^ CAUSAL_LINK_RNG_DOMAIN,
                    );
                    rng.set_stream(particle as u64);
                    rng
                })
                .collect();
            let link_shape = plan.causal_link_shape();
            let link_per_particle = link_shape[1]
                .checked_mul(link_shape[2])
                .and_then(|value| value.checked_mul(link_shape[3]))
                .ok_or_else(|| {
                    SgdError::Backend(format!(
                        "causal-link tensor size overflows usize for shape {link_shape:?}"
                    ))
                })?;
            let total_links = config
                .particles
                .checked_mul(link_per_particle)
                .ok_or_else(|| {
                    SgdError::Backend(format!(
                        "causal-link tensor size overflows usize for shape {link_shape:?}"
                    ))
                })?;
            let mut initial_link = Vec::with_capacity(total_links);
            for stream in &mut link_streams {
                let ticket = initial_link_vec(
                    stream,
                    horizon,
                    plan.num_facts,
                    config.causal_link_initial_bias,
                );
                assert_eq!(
                    ticket.len(),
                    link_per_particle,
                    "causal ticket size matches the checked link shape"
                );
                initial_link.extend(ticket);
            }
            let logits = Var::from_vec(initial_link, &link_shape, &device)?;
            let optimizer = Adam::new(
                vec![logits.clone()],
                AdamParams {
                    learning_rate: config.causal_link_learning_rate,
                    grad_clip: config.grad_clip,
                    particles: Some(config.particles),
                    ..AdamParams::default()
                },
            )?;
            Some(CausalLinkLane {
                logits,
                optimizer,
                streams: link_streams,
            })
        } else {
            None
        };
        assert_eq!(
            link_lane.is_some(),
            config.causal_links_enabled(),
            "the dense causal-link lane exactly follows its validated activation predicate"
        );
        let mut causal_action_optimizer = Adam::new(
            vec![causal_action_logits.clone()],
            AdamParams {
                learning_rate: config.learning_rate,
                grad_clip: config.grad_clip,
                particles: Some(config.particles),
                ..AdamParams::default()
            },
        )?;
        let mut duals = Duals::zeros(&plan, &device)?;
        let remelt_patience_checks = config.remelt_patience.div_ceil(config.verify_period).max(1);
        let controller_config = ControllerConfig {
            failure_row_weight: FailureWeightSchedule {
                initial: 1.0,
                growth: config.focus_growth,
                cap: config.focus_cap,
            },
            failure_fact_weight: FailureWeightSchedule {
                initial: 1.0,
                growth: config.focus_growth,
                cap: config.focus_cap,
            },
            missing_goal_weight: GoalWeightSchedule {
                baseline: 1.0,
                increment: 1.0,
                cap: config.dual_cap,
            },
            patience: PhasePatience {
                build_applicability: remelt_patience_checks,
                goal: remelt_patience_checks,
                goal_repair: remelt_patience_checks,
            },
            minimum_remelt_radius: 4,
        };
        let mut controller = VerifierController::new(
            config.particles,
            horizon,
            (0..num_goals).collect::<Vec<_>>(),
            controller_config,
        )
        .expect("engine dimensions and validated schedules form a valid controller");
        let mut exact_state_target =
            vec![0.0f64; config.particles * (horizon + 1) * plan.num_facts];
        let mut exact_state_active =
            vec![0.0f64; config.particles * (horizon + 1) * plan.num_facts];
        let mut applicable_mask = vec![0.0f64; config.particles * horizon * plan.num_actions];
        let mut applicable_active = vec![0.0f64; config.particles * horizon];
        let mut temporal_applicable_mask =
            vec![0.0f64; config.particles * horizon * plan.num_actions];
        let mut temporal_scaffold_gap_fact_values =
            vec![0.0f64; config.particles * (horizon + 1) * plan.num_facts];
        let mut active_failure_action = vec![None; config.particles];
        let mut failure_precondition_memory =
            vec![1.0f64; config.particles * horizon * plan.num_preconditions];
        let initial_repair_radius = 4usize.max(horizon / 16).min(horizon);
        let mut repair_start = vec![0usize; config.particles];
        let mut goal_repair_start = vec![horizon - initial_repair_radius; config.particles];
        let mut insert_target = vec![None::<(usize, usize)>; config.particles];
        let mut insert_mode = vec![false; config.particles];
        let mut insert_at = vec![None::<usize>; config.particles];
        let mut insert_required_fact = vec![None::<usize>; config.particles];
        let mut full_prefix_insert_stalls = vec![0usize; config.particles];
        // An anchor contains only a decoded checkpoint and the exclusive end
        // of its verifier-proven applicable prefix. The arbitrary suffix past
        // the first failed row is never evidence and must remain free.
        let mut anchor_actions = vec![None::<(Vec<usize>, usize)>; config.particles];
        let mut causal_goal_progress = vec![0.0f64; config.particles];
        let mut remelt_age = vec![config.remelt_cooldown_updates; config.particles * horizon];
        // Scheduling is an insertion mechanism. During discovery every token
        // remains exactly row-aligned, so this representation is behaviorally
        // identical to the direct plan and cannot scramble a nascent scaffold.
        let mut temporal_unlocked = vec![false; config.particles];
        let mut temporal_probed = vec![false; config.particles];
        // The schedule anneals from temporal unlock. Newly activated action
        // tokens have independent clocks below.
        let mut temporal_schedule_epoch = vec![None::<usize>; config.particles];
        let mut temporal_token_activation_update =
            vec![vec![None::<usize>; horizon]; config.particles];
        let mut temporal_last_progress_update = vec![None::<usize>; config.particles];
        let mut temporal_repair_capacity = vec![0usize; config.particles];
        // Persistent fact roles for reopened no-op tokens. A role names only the
        // fact that must be produced; the loss remains symmetric over every
        // grounded achiever and never selects an operator on the host.
        let mut temporal_obligations = vec![vec![None::<usize>; horizon]; config.particles];
        let mut temporal_obligation_focus = vec![vec![1.0f64; horizon]; config.particles];
        let mut temporal_applicability_focus = vec![vec![1.0f64; horizon]; config.particles];
        let mut temporal_unused_tokens = vec![Vec::<usize>::new(); config.particles];
        let mut temporal_frozen_noops = vec![Vec::<usize>::new(); config.particles];
        // Goal roles are simultaneous. Different particles deterministically
        // permute all goal orders, including goals currently true but possibly
        // needing restoration after a temporary deletion. Such a role may
        // remain a no-op until exact replay first observes that goal missing.
        let mut temporal_goal_tokens = vec![Vec::<(usize, usize, usize)>::new(); config.particles];
        let mut temporal_goal_required = vec![vec![false; num_goals]; config.particles];
        // Edges are over token identities, not execution rows. Scaffold edges
        // retain the verifier-proven real-action order; repair edges place a
        // prerequisite producer before its obligated consumer.
        let mut temporal_precedence = vec![Vec::<(usize, usize)>::new(); config.particles];
        // Subset of precedence edges justified by a failed precondition. Keep
        // these separate from immutable scaffold order and threat-before-goal
        // edges: only causal edges supply facts to a consumer and participate
        // in recursive prerequisite repair.
        let mut temporal_causal_precedence = vec![Vec::<(usize, usize)>::new(); config.particles];
        let mut temporal_order_conflicts =
            vec![BTreeSet::<(usize, usize)>::new(); config.particles];
        let mut temporal_causal_cycles = vec![BTreeSet::<(usize, usize)>::new(); config.particles];
        let mut temporal_interval_rejected_action =
            vec![vec![None::<usize>; horizon]; config.particles];
        let mut temporal_token_row = (0..config.particles)
            .map(|_| (0..horizon).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        // Every temporal schedule is an interleaving of these two streams.
        // Before unlock the repair stream is empty, giving the exact direct
        // plan. Unlock partitions off dedicated no-op tokens while retaining
        // the relative order of both partitions forever.
        let mut temporal_scaffold_order = (0..config.particles)
            .map(|_| (0..horizon).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut temporal_repair_order = vec![Vec::<usize>::new(); config.particles];
        let mut checks_since_refresh = 0usize;
        let mut q_perturbed = false;
        for update in 0..config.updates {
            outcome.updates += 1;
            for age in &mut remelt_age {
                *age = if *age == usize::MAX {
                    0
                } else {
                    age.saturating_add(1).min(config.remelt_cooldown_updates)
                };
            }
            let progress = global_progress(update, config.updates);
            let stage = config.causal_stage_at(progress);
            if matches!(config.causal_copy, CausalCopyMode::Shadow)
                || (matches!(config.causal_copy, CausalCopyMode::Staged)
                    && matches!(stage, CausalStage::Shadow))
            {
                causal_action_logits.set(&action_logits.as_tensor().detach())?;
            } else if matches!(config.causal_copy, CausalCopyMode::Staged) && !q_perturbed {
                let perturbation = Tensor::from_vec(
                    causal_action_streams
                        .iter_mut()
                        .flat_map(|stream| {
                            normal_vec(
                                stream,
                                horizon * plan.num_actions,
                                config.q_logit_perturbation,
                            )
                        })
                        .collect::<Vec<_>>(),
                    (config.particles, horizon, plan.num_actions),
                    &device,
                )?;
                causal_action_logits
                    .set(&(causal_action_logits.as_tensor() + perturbation)?.detach())?;
                q_perturbed = true;
            }
            let phases: Vec<f64> = (0..config.particles)
                .map(|particle| config.phase_at(update, particle))
                .collect();
            let execution_schedule_progress: Vec<f64> = phases
                .iter()
                .map(|&phase| {
                    if progress < config.causal_proof_end {
                        phase
                    } else if progress < config.causal_transfer_end {
                        0.5
                    } else {
                        0.5 + 0.5 * ramp(progress, config.causal_transfer_end, 1.0)
                    }
                })
                .collect();
            let q_schedule_progress =
                ramp(progress, config.causal_shadow_end, config.causal_proof_end);
            let q_drives_causality = matches!(config.causal_copy, CausalCopyMode::Staged)
                && !matches!(stage, CausalStage::Shadow);
            let temporal_progress = temporal_schedule_epoch
                .iter()
                .map(|unlock| match unlock {
                    None => 0.0,
                    Some(unlock) => {
                        let remaining = config.updates.saturating_sub(*unlock + 1).max(1);
                        update.saturating_sub(*unlock) as f64 / remaining as f64
                    }
                })
                .collect::<Vec<_>>();
            let temporal_token_progress = temporal_token_activation_update
                .iter()
                .flat_map(|particle_epochs| {
                    particle_epochs.iter().map(|epoch| match epoch {
                        None => 0.0,
                        Some(epoch) => {
                            let remaining = config.updates.saturating_sub(*epoch + 1).max(1);
                            update.saturating_sub(*epoch) as f64 / remaining as f64
                        }
                    })
                })
                .collect::<Vec<_>>();
            let mut action_temperatures = Vec::with_capacity(config.particles * horizon);
            let mut state_temperatures = Vec::with_capacity(config.particles * horizon);
            for (particle, &schedule) in execution_schedule_progress.iter().enumerate() {
                let base_action = config.action_temperature_at(schedule);
                let base_state = config.state_temperature_at(schedule);
                for row in 0..horizon {
                    let base_action = if temporal_unlocked[particle]
                        && temporal_repair_order[particle].contains(&row)
                    {
                        // A repair token is a newly initialized categorical
                        // variable. Anneal it from its own unlock epoch instead
                        // of inheriting a nearly crystallized global clock.
                        config.action_temperature_at(
                            temporal_token_progress[particle * horizon + row],
                        )
                    } else {
                        base_action
                    };
                    let cooldown = remelt_age[particle * horizon + row] as f64
                        / config.remelt_cooldown_updates as f64;
                    action_temperatures.push(
                        config.action_temperature.0
                            + (base_action - config.action_temperature.0) * cooldown,
                    );
                    state_temperatures.push(
                        config.state_temperature.0
                            + (base_state - config.state_temperature.0) * cooldown,
                    );
                }
            }
            let action_temperature =
                Tensor::from_vec(action_temperatures, (config.particles, horizon, 1), &device)?;
            let state_temperature =
                Tensor::from_vec(state_temperatures, (config.particles, horizon, 1), &device)?;
            let schedule_temperature = Tensor::from_vec(
                temporal_progress
                    .iter()
                    .map(|&schedule| config.schedule_temperature_at(schedule))
                    .collect::<Vec<_>>(),
                (config.particles, 1, 1),
                &device,
            )?;
            // Preserve the direct row order until a particle exposes an
            // applicable scaffold with repair capacity. Afterwards a monotone
            // lattice merges the immutable scaffold and repair streams.
            let schedule_gate_values = temporal_unlocked
                .iter()
                .map(|&unlocked| f64::from(unlocked))
                .collect::<Vec<_>>();
            let schedule_gate = Tensor::from_vec(
                schedule_gate_values.clone(),
                (config.particles, 1, 1),
                &device,
            )?;
            let mut temporal_action_gate_values = Vec::with_capacity(config.particles * horizon);
            for particle in 0..config.particles {
                for token in 0..horizon {
                    temporal_action_gate_values.push(f64::from(
                        !temporal_unlocked[particle]
                            || temporal_repair_order[particle].contains(&token),
                    ));
                }
            }
            let temporal_action_gate = Tensor::from_vec(
                temporal_action_gate_values,
                (config.particles, horizon, 1),
                &device,
            )?;
            let mut temporal_inactive_noop_values = Vec::with_capacity(config.particles * horizon);
            for particle in 0..config.particles {
                for token in 0..horizon {
                    let optional_goal = temporal_goal_tokens[particle]
                        .iter()
                        .find(|&&(goal_token, _, _)| goal_token == token)
                        .is_some_and(|&(_, goal, _)| !temporal_goal_required[particle][goal]);
                    temporal_inactive_noop_values.push(f64::from(
                        temporal_unlocked[particle]
                            && (temporal_unused_tokens[particle].contains(&token)
                                || temporal_frozen_noops[particle].contains(&token)
                                || optional_goal),
                    ));
                }
            }
            let temporal_inactive_noop = Tensor::from_vec(
                temporal_inactive_noop_values,
                (config.particles, horizon, 1),
                &device,
            )?;
            let q_action_temperature = Tensor::from_vec(
                vec![config.q_action_temperature_at(q_schedule_progress); config.particles],
                (config.particles, 1, 1),
                &device,
            )?;
            let link_temperature = if link_lane.is_some() {
                Some(Tensor::from_vec(
                    if q_drives_causality {
                        vec![
                            config.causal_link_temperature_at(q_schedule_progress);
                            config.particles
                        ]
                    } else {
                        phases
                            .iter()
                            .map(|&phase| config.causal_link_temperature_at(phase))
                            .collect()
                    },
                    (config.particles, 1, 1, 1),
                    &device,
                )?)
            } else {
                None
            };
            let exact_integrality = exact_integrality_scale(config, progress);
            let mut integrality_values = Vec::with_capacity(config.particles * horizon);
            for (particle, ages) in remelt_age.chunks_exact(horizon).enumerate() {
                for (token, &age) in ages.iter().enumerate() {
                    let token_integrality = if temporal_unlocked[particle]
                        && temporal_repair_order[particle].contains(&token)
                    {
                        exact_integrality.min(config.integrality_scale_at(
                            temporal_token_progress[particle * horizon + token],
                        ))
                    } else {
                        exact_integrality
                    };
                    integrality_values.push(
                        token_integrality * age as f64 / config.remelt_cooldown_updates as f64,
                    );
                }
            }
            let integrality_scale =
                Tensor::from_vec(integrality_values, (config.particles, horizon, 1), &device)?;

            let effective_action_logits =
                insertion_warp_logits(action_logits.as_tensor(), &insert_at)?;
            let effective_causal_action_logits =
                insertion_warp_logits(causal_action_logits.as_tensor(), &insert_at)?;

            let temporal = if config.temporal_tokens {
                let token_action = repair_only_action_gradients(
                    &plan.action_distribution(&effective_action_logits, &action_temperature)?,
                    &temporal_action_gate,
                )?;
                let token_action = force_inactive_temporal_noops(
                    &token_action,
                    &temporal_inactive_noop,
                    transcription.noop_action(),
                )?;
                Some(monotone_interleaving_schedule(
                    token_action,
                    schedule_logits.as_tensor(),
                    &schedule_temperature,
                    &temporal_scaffold_order,
                    &temporal_repair_order,
                    &temporal_precedence,
                )?)
            } else {
                None
            };
            let forward = if let Some(temporal) = &temporal {
                plan.forward_from_action_distribution(
                    temporal.action.clone(),
                    temporal.log_action.clone(),
                    state_logits.as_tensor(),
                    &state_temperature,
                )?
            } else if config.slot_slack_window == 0 {
                plan.forward(
                    &effective_action_logits,
                    state_logits.as_tensor(),
                    &action_temperature,
                    &state_temperature,
                )?
            } else {
                plan.forward_hybrid(
                    &effective_action_logits,
                    state_logits.as_tensor(),
                    &action_temperature,
                    &state_temperature,
                    config.slot_slack_window,
                )?
            };
            // Early recurrent planning uses stochastic action mass.  Once
            // external noise stops, continue toward the actual argmax decode
            // while retaining the softmax surrogate gradient.  At takeover
            // the recurrent forward pass is exactly the sequence verified by
            // the symbolic replay (modulo no-op compression).
            let execution_hardening = ramp(
                progress,
                config.causal_transfer_end,
                config.causal_takeover_end,
            );
            let q_hardening = ramp(
                progress,
                config.causal_discovery_end,
                config.causal_proof_end,
            );
            let execution_causal = if let Some(temporal) = &temporal {
                plan.two_loss_forward_from_action(temporal.action.clone(), execution_hardening)?
            } else if config.slot_slack_window == 0 {
                plan.two_loss_forward_hardened(
                    &effective_action_logits,
                    &action_temperature,
                    execution_hardening,
                )?
            } else {
                plan.two_loss_forward_hybrid_hardened(
                    &effective_action_logits,
                    &action_temperature,
                    config.slot_slack_window,
                    execution_hardening,
                )?
            };
            let q_causal = if matches!(config.causal_copy, CausalCopyMode::Staged) {
                Some(if config.slot_slack_window == 0 {
                    plan.two_loss_forward_hardened(
                        &effective_causal_action_logits,
                        &q_action_temperature,
                        q_hardening,
                    )?
                } else {
                    plan.two_loss_forward_hybrid_hardened(
                        &effective_causal_action_logits,
                        &q_action_temperature,
                        config.slot_slack_window,
                        q_hardening,
                    )?
                })
            } else {
                None
            };
            let mut focus_rows = vec![1.0f64; config.particles * horizon];
            let mut failed_fact_rows =
                vec![0.0f64; config.particles * (horizon + 1) * plan.num_facts];
            let mut goal_deadline_rows =
                vec![0.0f64; config.particles * (horizon + 1) * plan.num_facts];
            let mut deadline_source_mask = vec![0.0f64; config.particles * (horizon + 1) * horizon];
            let mut deadline_boundary_mask = vec![0.0f64; config.particles * horizon];
            let mut obligation_achiever_mask =
                vec![0.0f64; config.particles * horizon * plan.num_actions];
            let mut obligation_active = vec![0.0f64; config.particles * horizon];
            let mut obligation_focus = vec![0.0f64; config.particles * horizon];
            let mut obligation_applicability_focus = vec![0.0f64; config.particles * horizon];
            let mut bridge_terminal = vec![0.0f64; config.particles * plan.num_facts];
            let mut bridge_active = vec![0.0f64; config.particles * horizon];
            let mut bridge_boundary = vec![0.0f64; config.particles * horizon];
            let mut bridge_scale = vec![0.0f64; config.particles];
            let mut failed_deadlines = BTreeSet::new();
            let mut has_goal_deadline = false;
            let mut goal_rows = Vec::with_capacity(config.particles * num_goals);
            let mut phase_precondition = Vec::with_capacity(config.particles);
            let mut phase_transition = Vec::with_capacity(config.particles);
            let mut phase_goal = Vec::with_capacity(config.particles);
            let mut phase_causal = Vec::with_capacity(config.particles);
            let mut protect_consumer_values = Vec::with_capacity(config.particles);
            let mut precondition_focus = failure_precondition_memory.clone();
            for particle in 0..config.particles {
                let state = controller
                    .particle(particle)
                    .expect("particle index comes from controller dimensions");
                for (token, obligation) in temporal_obligations[particle].iter().enumerate() {
                    let Some(fact) = obligation else {
                        continue;
                    };
                    let achievers = &fact_achievers[*fact];
                    if achievers.is_empty() {
                        // This fact cannot be reconstructed by any action.
                        // Other losses must revise the earlier scaffold so it
                        // survives from the boundary; there is no truthful
                        // achiever set for a ranking certificate.
                        continue;
                    }
                    let execution_row = temporal_token_row[particle][token];
                    let applicability_begin =
                        (particle * horizon + execution_row) * plan.num_actions;
                    let exact_applicable =
                        if applicable_active[particle * horizon + execution_row] > 0.0 {
                            Some(
                                &applicable_mask
                                    [applicability_begin..applicability_begin + plan.num_actions],
                            )
                        } else {
                            None
                        };
                    // Separate max constraints for "is an achiever" and "is
                    // applicable" can be satisfied by different actions. If
                    // exact replay proves a nonempty intersection, rank that
                    // conjunction directly. If it is empty, retain all
                    // achievers while an earlier obligation constructs their
                    // missing prerequisite.
                    let mut ranked_actions = if config.temporal_tokens {
                        // Temporal applicability is coupled to every frozen
                        // scaffold gap below. Restricting the good set to the
                        // current hard row creates a self-confirming trap.
                        achievers.clone()
                    } else {
                        obligation_achiever_conjunction(achievers, exact_applicable)
                    };
                    if temporal_goal_tokens[particle]
                        .iter()
                        .find(|&&(goal_token, _, _)| goal_token == token)
                        .is_some_and(|&(_, goal, _)| !temporal_goal_required[particle][goal])
                    {
                        ranked_actions.push(transcription.noop_action());
                        ranked_actions.sort_unstable();
                        ranked_actions.dedup();
                    }
                    ranked_actions = achievers_except_rejected(
                        &ranked_actions,
                        temporal_interval_rejected_action[particle][token],
                    );
                    obligation_active[particle * horizon + token] = 1.0;
                    obligation_focus[particle * horizon + token] =
                        temporal_obligation_focus[particle][token];
                    obligation_applicability_focus[particle * horizon + token] =
                        temporal_applicability_focus[particle][token];
                    let begin = (particle * horizon + token) * plan.num_actions;
                    for action in ranked_actions {
                        obligation_achiever_mask[begin + action] = 1.0;
                    }
                }
                goal_rows.extend_from_slice(state.goal_weights());
                // The mask comes only from a completely applicable replay and
                // remains trustworthy during GoalRepair. Keep producer
                // pressure alive while the new achiever is temporarily
                // inapplicable; otherwise first-failure pressure simply
                // deletes the achiever it was meant to support.
                for (goal, (&missing, &weight)) in state
                    .missing_goal_mask()
                    .iter()
                    .zip(state.goal_weights())
                    .enumerate()
                {
                    if missing {
                        let fact = transcription.goal_facts()[goal] as usize;
                        goal_deadline_rows
                            [(particle * (horizon + 1) + horizon) * plan.num_facts + fact] = weight;
                    }
                }
                let phase = phase_loss_weights(state.phase());
                phase_precondition.push(phase.precondition);
                phase_transition.push(phase.transition);
                phase_goal.push(phase.goal);
                phase_causal.push(phase.causal);
                protect_consumer_values.push(f64::from(matches!(state.phase(), Phase::GoalRepair)));
                if let Some(active) = state.active_failure() {
                    focus_rows[particle * horizon + active.row] = active.row_weight;
                    if let (Some(fact), Some(weight)) = (active.fact, active.fact_weight) {
                        failed_fact_rows
                            [(particle * (horizon + 1) + active.row) * plan.num_facts + fact] =
                            weight;
                        let action = active_failure_action[particle]
                            .expect("a fact-specific active failure retains its rejected action");
                        let mut matched = 0usize;
                        for (incidence, (&pre_action, &pre_fact)) in transcription
                            .pre_action()
                            .iter()
                            .zip(transcription.pre_fact())
                            .enumerate()
                        {
                            if pre_action as usize == action && pre_fact as usize == fact {
                                let index = (particle * horizon + active.row)
                                    * plan.num_preconditions
                                    + incidence;
                                precondition_focus[index] = precondition_focus[index]
                                    .max(active.row_weight * weight)
                                    .min(config.focus_cap);
                                matched += 1;
                            }
                        }
                        assert_eq!(
                            matched, 1,
                            "canonical failed action/fact pair has exactly one incidence"
                        );
                        if insert_mode[particle] {
                            let insertion = insert_at[particle]
                                .expect("insertion mode retains its warp coordinate");
                            let deadline = insertion + 1;
                            let required_fact = insert_required_fact[particle]
                                .expect("insertion mode retains its original fact obligation");
                            let demand_index = (particle * (horizon + 1) + deadline)
                                * plan.num_facts
                                + required_fact;
                            failed_fact_rows[demand_index] =
                                failed_fact_rows[demand_index].max(active.row_weight * weight);
                            let source_start = repair_start[particle].min(insertion);
                            for source in source_start..deadline {
                                deadline_source_mask
                                    [(particle * (horizon + 1) + deadline) * horizon + source] =
                                    1.0;
                            }
                            if config.temporal_tokens {
                                deadline_boundary_mask[particle * horizon + source_start] = 1.0;
                            }
                            failed_deadlines.insert(deadline);
                        } else if config.temporal_tokens
                            && matches!(state.phase(), Phase::GoalRepair)
                            && active.row > goal_repair_start[particle]
                        {
                            // In token mode the missing-goal achiever is kept
                            // by the terminal deadline loss. Turn its exact
                            // first failed precondition into an earlier
                            // existential producer deadline instead of
                            // replacing the achiever with an applicable action.
                            let deadline = active.row;
                            let demand_index =
                                (particle * (horizon + 1) + deadline) * plan.num_facts + fact;
                            failed_fact_rows[demand_index] = f64::max(
                                failed_fact_rows[demand_index],
                                active.row_weight * weight,
                            );
                            for source in goal_repair_start[particle]..deadline {
                                deadline_source_mask
                                    [(particle * (horizon + 1) + deadline) * horizon + source] =
                                    1.0;
                            }
                            deadline_boundary_mask
                                [particle * horizon + goal_repair_start[particle]] = 1.0;
                            failed_deadlines.insert(deadline);
                        }
                    }
                }
                if state.missing_goal_mask().iter().any(|&missing| missing) {
                    has_goal_deadline = true;
                    let source_range = goal_repair_start[particle]..horizon;
                    for source in source_range {
                        deadline_source_mask
                            [(particle * (horizon + 1) + horizon) * horizon + source] = 1.0;
                    }
                    if config.temporal_tokens {
                        deadline_boundary_mask[particle * horizon + goal_repair_start[particle]] =
                            1.0;
                    }
                }
                // Once an applicable plan exposes a missing goal, retain its
                // backward chain while the newly introduced producer is being
                // made applicable. Gating this on `active_failure().is_none()`
                // would erase the bridge at exactly the first useful repair
                // step. BuildApplicability remains excluded: before the first
                // applicable checkpoint, ordinary precondition pressure owns
                // the trajectory and goal chains are too speculative.
                if goal_bridge_is_active(state.phase(), state.missing_goal_mask()) {
                    let start = goal_repair_start[particle];
                    assert!(start < horizon, "a missing-goal repair suffix is nonempty");
                    bridge_boundary[particle * horizon + start] = 1.0;
                    bridge_active[particle * horizon + start..(particle + 1) * horizon].fill(1.0);
                    for (goal, &missing) in state.missing_goal_mask().iter().enumerate() {
                        if missing {
                            bridge_scale[particle] =
                                bridge_scale[particle].max(state.goal_weights()[goal]);
                            bridge_terminal[particle * plan.num_facts
                                + transcription.goal_facts()[goal] as usize] = 1.0;
                        }
                    }
                }
            }
            let focus = Tensor::from_vec(focus_rows, (config.particles, horizon, 1), &device)?;
            let precondition_family =
                Tensor::from_vec(phase_precondition, (config.particles, 1, 1), &device)?;
            let transition_family =
                Tensor::from_vec(phase_transition, (config.particles, 1, 1), &device)?;
            let goal_family = Tensor::from_vec(phase_goal, (config.particles, 1, 1), &device)?;
            let causal_family = Tensor::from_vec(phase_causal, (config.particles, 1, 1), &device)?;
            let protect_consumer =
                Tensor::from_vec(protect_consumer_values, (config.particles, 1, 1), &device)?;
            if config.trace_period > 0 && update % config.trace_period == 0 {
                let particle = config.trace_particle;
                let controller_state = controller
                    .particle(particle)
                    .expect("validated trace particle remains in range");
                let action_probabilities = forward
                    .action
                    .narrow(0, particle, 1)?
                    .squeeze(0)?
                    .to_vec2::<f64>()?;
                let action_temperatures = action_temperature
                    .narrow(0, particle, 1)?
                    .squeeze(0)?
                    .to_vec2::<f64>()?
                    .into_iter()
                    .map(|row| {
                        assert_eq!(row.len(), 1, "one action temperature per row");
                        row[0]
                    })
                    .collect();
                let precondition_by_row = particle_row_means(&forward.precondition, particle)?;
                let mut transition_by_row = vec![0.0; horizon];
                for residual in &forward.transition {
                    for (total, value) in transition_by_row
                        .iter_mut()
                        .zip(particle_row_means(residual, particle)?)
                    {
                        *total += value;
                    }
                }
                let action_integrality = plan.action_integrality_per_particle(&forward.action)?;
                outcome.trace.push(SgdTracePoint {
                    round,
                    update,
                    particle,
                    phase: controller_state.phase(),
                    goal_weights: controller_state.goal_weights().to_vec(),
                    missing_goals: controller_state.missing_goal_mask().to_vec(),
                    goal_repair_start: goal_repair_start[particle],
                    action_temperatures,
                    action_probabilities,
                    token_action_probabilities: match &temporal {
                        Some(temporal) => temporal
                            .token_action
                            .narrow(0, particle, 1)?
                            .squeeze(0)?
                            .to_vec2::<f64>()?,
                        None => Vec::new(),
                    },
                    temporal_obligations: temporal_obligations[particle].clone(),
                    temporal_obligation_achievers: temporal_obligations[particle]
                        .iter()
                        .map(|obligation| {
                            obligation
                                .map(|fact| fact_achievers[fact].clone())
                                .unwrap_or_default()
                        })
                        .collect(),
                    temporal_obligation_focus: temporal_obligation_focus[particle].clone(),
                    temporal_applicability_focus: temporal_applicability_focus[particle].clone(),
                    precondition_by_row,
                    transition_by_row,
                    goal_residuals: particle_flat_values(&forward.goal, particle)?,
                    recurrent_precondition_by_row: particle_flat_values(
                        &execution_causal.failed_precondition_by_step,
                        particle,
                    )?,
                    recurrent_terminal_goals: particle_flat_values(
                        &execution_causal.terminal_goal_by_goal,
                        particle,
                    )?,
                    recurrent_producer_goals: particle_flat_values(
                        &execution_causal.relaxed_goal_by_goal,
                        particle,
                    )?,
                    action_integrality_by_row: particle_flat_values(&action_integrality, particle)?,
                    action_logit_gradients: Vec::new(),
                    temporal_assignment: match &temporal {
                        Some(temporal) => temporal
                            .assignment
                            .narrow(0, particle, 1)?
                            .squeeze(0)?
                            .to_vec2::<f64>()?,
                        None => Vec::new(),
                    },
                    temporal_soft_assignment: match &temporal {
                        Some(temporal) => temporal
                            .soft_assignment
                            .narrow(0, particle, 1)?
                            .squeeze(0)?
                            .to_vec2::<f64>()?,
                        None => Vec::new(),
                    },
                    schedule_logit_gradients: Vec::new(),
                });
            }
            let exact_causal_demand = if config.causal_link_objective_enabled() {
                let mut demand = failed_fact_rows.clone();
                for particle in 0..config.particles {
                    for (goal, &fact) in transcription.goal_facts().iter().enumerate() {
                        let weight = goal_rows[particle * num_goals + goal];
                        // Exact goal pressure is a continuation schedule: first
                        // build an applicable causal prefix, then let repeatedly
                        // missing goals dominate. Squaring gives long-prefix
                        // particles strong pressure without letting early
                        // failures turn every row into an unrelated goal producer.
                        let extra =
                            (weight - 1.0).max(0.0) * causal_goal_progress[particle].powi(2);
                        demand[(particle * (horizon + 1) + horizon) * plan.num_facts
                            + fact as usize] += extra;
                    }
                }
                Some(Tensor::from_vec(
                    demand,
                    (config.particles, horizon + 1, plan.num_facts),
                    &device,
                )?)
            } else {
                None
            };
            let failed_fact = Tensor::from_vec(
                failed_fact_rows,
                (config.particles, horizon + 1, plan.num_facts),
                &device,
            )?;
            let goal_deadline = Tensor::from_vec(
                goal_deadline_rows,
                (config.particles, horizon + 1, plan.num_facts),
                &device,
            )?;
            let deadline_sources = Tensor::from_vec(
                deadline_source_mask,
                (config.particles, horizon + 1, horizon),
                &device,
            )?;
            let deadline_boundary = Tensor::from_vec(
                deadline_boundary_mask,
                (config.particles, horizon, 1),
                &device,
            )?;
            let deadline_boundary_state = execution_causal
                .exact_state_by_step
                .broadcast_mul(&deadline_boundary)?
                .sum(1)?
                .reshape((config.particles, 1, plan.num_facts))?;
            let obligation_achievers = Tensor::from_vec(
                obligation_achiever_mask,
                (config.particles, horizon, plan.num_actions),
                &device,
            )?;
            let obligation_active =
                Tensor::from_vec(obligation_active, (config.particles, horizon), &device)?;
            let obligation_focus =
                Tensor::from_vec(obligation_focus, (config.particles, horizon, 1), &device)?;
            let obligation_applicability_focus = Tensor::from_vec(
                obligation_applicability_focus,
                (config.particles, horizon, 1),
                &device,
            )?;
            let bridge_active_this_update = bridge_terminal.iter().any(|&demand| demand > 0.0);
            let bridge_first_active_row = bridge_terminal
                .chunks_exact(plan.num_facts)
                .enumerate()
                .filter(|(_, demand)| demand.iter().any(|&value| value > 0.0))
                .map(|(particle, _)| goal_repair_start[particle])
                .min()
                .unwrap_or(horizon);
            let bridge_terminal =
                Tensor::from_vec(bridge_terminal, (config.particles, plan.num_facts), &device)?;
            let bridge_active =
                Tensor::from_vec(bridge_active, (config.particles, horizon, 1), &device)?;
            let bridge_boundary =
                Tensor::from_vec(bridge_boundary, (config.particles, horizon, 1), &device)?;
            let bridge_scale = Tensor::from_vec(bridge_scale, (config.particles, 1), &device)?;
            let precondition_weight = Tensor::from_vec(
                precondition_focus,
                (config.particles, horizon, plan.num_preconditions),
                &device,
            )?
            .broadcast_mul(&precondition_family)?;
            let optimized_precondition = plan.protected_precondition_residual(
                &forward.action,
                &forward.state,
                &protect_consumer,
            )?;
            let optimized_transition = plan.protected_transition_residual(
                &forward.action,
                &forward.state,
                &protect_consumer,
            )?;
            let mut loss = weighted_family_loss(
                &optimized_precondition,
                &duals.precondition,
                config.rho_precondition,
                &precondition_weight,
            )?;
            for family in 0..4 {
                loss = (loss
                    + weighted_family_loss(
                        &optimized_transition[family],
                        &duals.transition[family],
                        config.rho_transition,
                        &transition_family,
                    )?)?;
            }
            loss = (loss
                + weighted_family_loss(
                    &forward.goal,
                    &duals.goal,
                    config.rho_goal,
                    &goal_family,
                )?)?;

            let goal_weight =
                Tensor::from_vec(goal_rows, (config.particles, 1, num_goals), &device)?;
            let scheduled_goal_weight = goal_weight.broadcast_mul(&goal_family)?;
            loss = (loss + (&forward.goal * &scheduled_goal_weight)?.mean_all()?)?;

            if config.temporal_reserved_slots > 0 && config.temporal_reservation_weight > 0.0 {
                let reserved_start = horizon - config.temporal_reserved_slots;
                let mut reservation = vec![0.0f64; config.particles * horizon];
                for particle in 0..config.particles {
                    if !temporal_unlocked[particle] {
                        reservation[particle * horizon + reserved_start..(particle + 1) * horizon]
                            .fill(1.0);
                    } else {
                        // Releasing every repair token at commitment lets an
                        // unassigned token crystallize into an arbitrary real
                        // action and fail before replay reaches the obligated
                        // consumer. Working memory remains no-op until a fact
                        // role is explicitly assigned to that token.
                        for &token in &temporal_unused_tokens[particle] {
                            reservation[particle * horizon + token] = 1.0;
                        }
                        for &token in &temporal_frozen_noops[particle] {
                            reservation[particle * horizon + token] = 1.0;
                        }
                    }
                }
                let reservation =
                    Tensor::from_vec(reservation, (config.particles, horizon, 1), &device)?;
                let active = reservation.sum_all()?.clamp(1.0, f64::MAX)?;
                let noop_log = temporal
                    .as_ref()
                    .expect("temporal reservation addresses latent tokens")
                    .token_action
                    .clamp(1e-300, 1.0)?
                    .log()?
                    .narrow(2, transcription.noop_action(), 1)?;
                let reservation_loss = ((noop_log.neg()? * reservation)?.sum_all()? / active)?;
                loss = (loss + reservation_loss * config.temporal_reservation_weight)?;
            }

            let use_q_causal = q_drives_causality;
            let causal = if use_q_causal {
                q_causal
                    .as_ref()
                    .expect("staged Q causal phases allocate the causal rollout")
            } else {
                &execution_causal
            };
            // P keeps the historical causal route during shadowing. Q and its
            // witnesses are optimized only during proof discovery; transfer
            // freezes them so the teacher cannot be dragged toward P.
            let optimize_causal = !matches!(config.causal_copy, CausalCopyMode::Staged)
                || matches!(
                    stage,
                    CausalStage::Shadow | CausalStage::Discovery | CausalStage::Proof
                );
            if let Some(route) = recurrent_loss_route(config.causal_copy, stage, update) {
                let per_goal_weight =
                    scheduled_goal_weight.reshape((config.particles, num_goals))?;
                let recurrent = match route.source {
                    RecurrentLossSource::Execution => &execution_causal,
                    RecurrentLossSource::CausalCopy => q_causal
                        .as_ref()
                        .expect("only staged Q phases route through the causal copy"),
                };
                let row_focus = if matches!(route.source, RecurrentLossSource::CausalCopy) {
                    // Verifier rows name P's selected action. Applying that
                    // focus to a different Q action is semantically false.
                    causal_family
                        .broadcast_as((config.particles, horizon, 1))?
                        .reshape((config.particles, horizon))?
                } else {
                    focus
                        .broadcast_mul(&causal_family)?
                        .reshape((config.particles, horizon))?
                };
                loss = (loss
                    + recurrent_plan_loss(
                        recurrent,
                        &per_goal_weight,
                        &row_focus,
                        &protect_consumer.reshape((config.particles, 1))?,
                        route.goal,
                        config.rho_goal,
                        config.rho_precondition,
                        config.goal_survival_weight,
                    )?)?;
            }

            let (insertion_raw_diagnostic, insertion_supported_diagnostic) =
                if config.insertion_repair_weight > 0.0 {
                    // Persistent ordered obligations need an argmax-level
                    // certificate, not merely existential producer mass. The
                    // good set contains every grounded achiever of the fact,
                    // so neither the verifier nor the host selects an action.
                    if let Some(obligation_action) = &temporal {
                        let obligation_log =
                            obligation_action.token_action.clamp(1e-300, 1.0)?.log()?;
                        let obligation_barrier = applicability_ranking_barrier(
                            &obligation_log,
                            &obligation_achievers,
                            &obligation_active,
                            &obligation_focus,
                            config.applicability_barrier_margin,
                        )?;
                        let obligation_mass = applicability_mass_loss(
                            &obligation_action.token_action,
                            &obligation_achievers,
                            &obligation_active,
                            &obligation_focus,
                        )?;
                        loss = (loss
                            + (obligation_barrier + obligation_mass)?
                                * config.insertion_repair_weight)?;
                        loss = (loss
                            + temporal_precedence_loss(
                                &obligation_action.soft_assignment,
                                &temporal_precedence,
                                &temporal_scaffold_order,
                                &temporal_repair_order,
                            )? * config.insertion_repair_weight)?;
                        let conditional_gap_applicable = conditional_gap_applicability_masks(
                            &temporal_obligations,
                            &temporal_causal_precedence,
                            &temporal_scaffold_order,
                            &temporal_scaffold_gap_fact_values,
                            &action_preconditions,
                            config.particles,
                            horizon,
                            plan.num_facts,
                            plan.num_actions,
                            false,
                        );
                        let graded_gap_applicable = conditional_gap_applicability_masks(
                            &temporal_obligations,
                            &temporal_causal_precedence,
                            &temporal_scaffold_order,
                            &temporal_scaffold_gap_fact_values,
                            &action_preconditions,
                            config.particles,
                            horizon,
                            plan.num_facts,
                            plan.num_actions,
                            true,
                        );
                        loss = (loss
                            + temporal_repair_edge_gap_support_loss(
                                &obligation_action.token_action,
                                &obligation_action.soft_assignment,
                                &obligation_achievers,
                                &conditional_gap_applicable,
                                &temporal_causal_precedence,
                                &temporal_scaffold_order,
                                &temporal_repair_order,
                            )? * config.insertion_repair_weight)?;
                        let temporal_applicable = Tensor::from_vec(
                            temporal_applicable_mask.clone(),
                            (config.particles, horizon, plan.num_actions),
                            &device,
                        )?;
                        loss = (loss
                            + temporal_obligation_applicability_loss(
                                &obligation_action.token_action,
                                &obligation_action.soft_assignment,
                                &temporal_applicable,
                                &obligation_active,
                                &obligation_applicability_focus,
                            )? * config.insertion_repair_weight)?;
                        loss = (loss
                            + temporal_obligation_scaffold_gap_loss(
                                &obligation_action.token_action,
                                &obligation_action.soft_assignment,
                                &obligation_achievers,
                                &graded_gap_applicable,
                                &obligation_active,
                                &obligation_applicability_focus,
                                &temporal_scaffold_order,
                                &temporal_repair_order,
                            )? * config.insertion_repair_weight)?;
                    }
                    let failed_deadlines = failed_deadlines.into_iter().collect::<Vec<_>>();
                    let goal_deadlines = if has_goal_deadline {
                        vec![horizon]
                    } else {
                        Vec::new()
                    };
                    let failed_deadline = if config.temporal_tokens {
                        plan.deadline_support_forward_from_boundary(
                            &forward.action,
                            &execution_causal.causal,
                            &deadline_boundary_state,
                            &failed_fact,
                            &deadline_sources,
                            &failed_deadlines,
                        )?
                    } else {
                        plan.deadline_support_forward(
                            &forward.action,
                            &execution_causal.causal,
                            &failed_fact,
                            &deadline_sources,
                            &failed_deadlines,
                        )?
                    };
                    let goal_deadline_support = if config.temporal_tokens {
                        plan.deadline_support_forward_from_boundary(
                            &forward.action,
                            &execution_causal.causal,
                            &deadline_boundary_state,
                            &goal_deadline,
                            &deadline_sources,
                            &goal_deadlines,
                        )?
                    } else {
                        plan.deadline_support_forward(
                            &forward.action,
                            &execution_causal.causal,
                            &goal_deadline,
                            &deadline_sources,
                            &goal_deadlines,
                        )?
                    };
                    let raw_diagnostic = (failed_deadline.raw_loss.mean_all()?
                        + goal_deadline_support.raw_loss.mean_all()?)?
                    .to_scalar::<f64>()?;
                    let supported_diagnostic = (failed_deadline.supported_loss.mean_all()?
                        + goal_deadline_support.supported_loss.mean_all()?)?
                    .to_scalar::<f64>()?;
                    let insertion_raw =
                        (&failed_deadline.raw_loss.mean_all()? * config.rho_precondition)?;
                    let insertion_raw = (insertion_raw
                        + (&goal_deadline_support.raw_loss.mean_all()? * config.rho_goal)?)?;
                    let insertion_supported =
                        (&failed_deadline.supported_loss.mean_all()? * config.rho_precondition)?;
                    let insertion_supported = (insertion_supported
                        + (&goal_deadline_support.supported_loss.mean_all()? * config.rho_goal)?)?;
                    loss = (loss
                        + ((insertion_raw * 0.25)? + insertion_supported)?
                            * config.insertion_repair_weight)?;
                    (raw_diagnostic, supported_diagnostic)
                } else {
                    // A disabled experimental term is absent from the graph,
                    // rather than evaluated and multiplied by zero. This is
                    // both the exact legacy boundary and the cheap ablation.
                    (0.0, 0.0)
                };

            let (backward_bridge_boundary_diagnostic, backward_bridge_precondition_diagnostic) =
                if config.backward_bridge_weight > 0.0 {
                    let lane = link_lane
                        .as_ref()
                        .expect("an enabled backward bridge allocates temporal-link logits");
                    let bridge_link_temperature = link_temperature
                        .as_ref()
                        .expect("an enabled backward bridge allocates a link temperature");
                    let bridge_action_temperature =
                        Tensor::ones((config.particles, horizon, 1), DTYPE, &device)?;
                    let bridge_action = if config.temporal_tokens {
                        let token_action = repair_only_action_gradients(
                            &plan.action_distribution(
                                &effective_action_logits,
                                &bridge_action_temperature,
                            )?,
                            &temporal_action_gate,
                        )?;
                        let token_action = force_inactive_temporal_noops(
                            &token_action,
                            &temporal_inactive_noop,
                            transcription.noop_action(),
                        )?;
                        monotone_interleaving_schedule(
                            token_action,
                            schedule_logits.as_tensor(),
                            &schedule_temperature,
                            &temporal_scaffold_order,
                            &temporal_repair_order,
                            &temporal_precedence,
                        )?
                        .action
                    } else if config.slot_slack_window == 0 {
                        plan.action_distribution(
                            &effective_action_logits,
                            &bridge_action_temperature,
                        )?
                    } else {
                        plan.hybrid_action_distribution(
                            &effective_action_logits,
                            &bridge_action_temperature,
                            config.slot_slack_window,
                        )?
                        .action
                    };
                    let bridge = plan.backward_causal_flow(
                        &bridge_action,
                        &execution_causal.causal.delete,
                        &execution_causal.exact_state_by_step,
                        &bridge_terminal,
                        &bridge_active,
                        &bridge_boundary,
                        lane.logits.as_tensor(),
                        bridge_link_temperature,
                        bridge_first_active_row,
                    )?;
                    let boundary = bridge.boundary_loss.mean_all()?.to_scalar::<f64>()?;
                    let precondition = bridge
                        .relevant_precondition_loss
                        .mean_all()?
                        .to_scalar::<f64>()?;
                    if bridge_active_this_update {
                        outcome.backward_bridge_updates += 1;
                        outcome.max_backward_bridge_loss = outcome
                            .max_backward_bridge_loss
                            .max(boundary + precondition);
                    }
                    // The verifier's monotonically increasing missing-goal
                    // multiplier also schedules its backward chain. This
                    // changes the loss landscape only after repeated exact
                    // evidence that the same goal remains absent; a giant
                    // static bridge coefficient would distort initial repair.
                    let scheduled_bridge = (&bridge.loss * &bridge_scale)?.mean_all()?;
                    loss = (loss + scheduled_bridge * config.backward_bridge_weight)?;
                    (boundary, precondition)
                } else {
                    (0.0, 0.0)
                };

            let anchor_trust_diagnostic = if config.anchor_trust_weight > 0.0 {
                let mut anchor_target_values =
                    vec![0.0f64; config.particles * horizon * plan.num_actions];
                let mut anchor_active_values = vec![0.0f64; config.particles * horizon];
                for particle in 0..config.particles {
                    let Some((anchor, trusted_end)) = anchor_actions[particle].as_ref() else {
                        continue;
                    };
                    assert_eq!(anchor.len(), horizon, "checkpoint anchor spans the horizon");
                    assert!(
                        *trusted_end <= horizon,
                        "trusted anchor prefix lies inside the horizon"
                    );
                    let controller_state = controller
                        .particle(particle)
                        .expect("particle index comes from controller dimensions");
                    if controller_state.active_failure().is_some()
                        && !insert_mode[particle]
                        && !config.temporal_tokens
                    {
                        continue;
                    }
                    for (row, &action) in anchor.iter().enumerate() {
                        if row >= *trusted_end {
                            break;
                        }
                        anchor_target_values
                            [(particle * horizon + row) * plan.num_actions + action] = 1.0;
                        let preserve_real = if config.temporal_tokens {
                            // Anchor token identity, not its current execution
                            // row. The schedule must remain free to shift the
                            // scaffold around newly specialized no-op tokens.
                            true
                        } else if let Some(failure) = controller_state.active_failure() {
                            let in_window = if insert_mode[particle] {
                                repair_start[particle] <= row && row < failure.row
                            } else {
                                repair_start[particle] <= row && row <= failure.row
                            };
                            !in_window || insert_mode[particle]
                        } else {
                            true
                        };
                        // A no-op is insertion capacity, never an anchor. In
                        // particular, an exactly applicable empty plan must not
                        // become a trust-region attractor merely because all of
                        // its rows agree with the latest checkpoint.
                        let preserve = preserve_real && action != transcription.noop_action();
                        anchor_active_values[particle * horizon + row] = f64::from(preserve);
                    }
                }
                let anchor_target = Tensor::from_vec(
                    anchor_target_values,
                    (config.particles, horizon, plan.num_actions),
                    &device,
                )?;
                let anchor_active = Tensor::from_vec(
                    anchor_active_values,
                    (config.particles, horizon, 1),
                    &device,
                )?;
                let temporal_anchor_log = match &temporal {
                    Some(temporal) => Some(temporal.token_action.clamp(1e-300, 1.0)?.log()?),
                    None => None,
                };
                let anchor_log_probability = temporal_anchor_log
                    .as_ref()
                    .unwrap_or(&forward.action_log_probability);
                let anchor_trust =
                    anchor_trust_loss(anchor_log_probability, &anchor_target, &anchor_active)?;
                let diagnostic = anchor_trust.to_scalar::<f64>()?;
                let anchor_pressure = (0..config.particles)
                    .filter(|&particle| anchor_actions[particle].is_some())
                    .flat_map(|particle| {
                        controller
                            .particle(particle)
                            .expect("anchor particle remains in range")
                            .goal_weights()
                            .iter()
                            .copied()
                    })
                    .fold(1.0f64, f64::max);
                // As verifier pressure on an unresolved goal grows, preserve
                // the proven scaffold at least as strongly. This remains a
                // finite soft trust region; no-op-derived rows stay free.
                loss = (loss + (anchor_trust * (config.anchor_trust_weight * anchor_pressure))?)?;
                diagnostic
            } else {
                0.0
            };

            let links = if config.causal_link_objective_enabled() {
                let lane = link_lane
                    .as_ref()
                    .expect("global causal-link loss allocates temporal logits");
                let temperature = link_temperature
                    .as_ref()
                    .expect("global causal-link loss allocates a temperature");
                let demand = exact_causal_demand
                    .as_ref()
                    .expect("global causal-link loss allocates exact demand");
                Some(plan.causal_link_forward_from_input_with_demand(
                    &causal.causal,
                    lane.logits.as_tensor(),
                    temperature,
                    demand,
                )?)
            } else {
                None
            };
            let causal_link_diagnostics = if let Some(links) = links.as_ref() {
                let causal_scale = causal_family.reshape((config.particles, 1))?;
                let link_feasibility = (&links.source_loss + &links.threat_loss)?;
                if optimize_causal {
                    loss = (loss
                        + (scheduled_particle_mean(&link_feasibility, &causal_scale)?
                            * config.causal_link_weight)?)?;
                    let proof_integrality = if matches!(stage, CausalStage::Proof) {
                        0.25 * ramp(
                            progress,
                            config.causal_discovery_end,
                            config.causal_proof_end,
                        )
                    } else if matches!(config.causal_copy, CausalCopyMode::Staged) {
                        0.0
                    } else {
                        exact_integrality
                    };
                    let link_integrality_scale = Tensor::from_vec(
                        vec![proof_integrality; config.particles],
                        (config.particles, 1),
                        &device,
                    )?;
                    loss = (loss
                        + (scheduled_particle_mean(
                            &links.link_integrality,
                            &link_integrality_scale,
                        )? * config.causal_link_integrality_final)?)?;
                    if use_q_causal && matches!(stage, CausalStage::Proof) {
                        loss = (loss
                            + (&causal.action_integrality.mean_all()?
                                * (0.25 * config.action_integrality_final))?)?;
                    }
                }
                (
                    links.source_loss.mean_all()?.to_scalar::<f64>()?,
                    links.threat_loss.mean_all()?.to_scalar::<f64>()?,
                    links.link_integrality.mean_all()?.to_scalar::<f64>()?,
                )
            } else {
                assert!(
                    !config.causal_link_objective_enabled(),
                    "an enabled global causal-link objective must produce diagnostics"
                );
                (0.0, 0.0, 0.0)
            };
            let transfer_weight = teacher_weight_at(config, progress);
            let causal_consensus_diagnostic = if let Some(q_causal) = &q_causal {
                let teacher_kl = teacher_kl_per_particle(&forward.action, &q_causal.action)?;
                if transfer_weight > 0.0 {
                    let links = links
                        .as_ref()
                        .expect("staged causal copy always retains its causal-link lane");
                    let teacher_readiness = (links
                        .max_source_violation
                        .le(config.teacher_tolerance)?
                        .to_dtype(DTYPE)?
                        * links
                            .max_threat_violation
                            .le(config.teacher_tolerance)?
                            .to_dtype(DTYPE)?)?;
                    let q_failed_max = q_causal
                        .failed_precondition_by_step
                        .max(1)?
                        .reshape((config.particles, 1))?;
                    let q_goal_max = q_causal
                        .terminal_goal_by_goal
                        .max(1)?
                        .reshape((config.particles, 1))?;
                    let q_action_max = plan
                        .action_integrality_per_particle(&q_causal.action)?
                        .max(1)?
                        .reshape((config.particles, 1))?;
                    // A causal teacher must be a complete, nearly discrete
                    // proof: supported/threat-free links alone say nothing
                    // about simultaneous terminal goals, and a fractional Q
                    // can have a different argmax sequence.  Every gate is
                    // detached because readiness schedules transfer; it is
                    // not another differentiable objective.
                    let teacher_readiness = (teacher_readiness
                        * q_failed_max.le(config.teacher_tolerance)?.to_dtype(DTYPE)?)?;
                    let teacher_readiness = (teacher_readiness
                        * q_goal_max.le(config.teacher_tolerance)?.to_dtype(DTYPE)?)?;
                    let teacher_readiness = (teacher_readiness
                        * q_action_max.le(config.teacher_tolerance)?.to_dtype(DTYPE)?)?;
                    let teacher_readiness = (teacher_readiness
                        * links
                            .link_integrality
                            .le(config.teacher_tolerance)?
                            .to_dtype(DTYPE)?)?
                    .detach();
                    let teacher_weight = (teacher_readiness * transfer_weight)?;
                    loss = (loss + scheduled_particle_mean(&teacher_kl, &teacher_weight)?)?;
                }
                teacher_kl.mean_all()?.to_scalar::<f64>()?
            } else {
                0.0
            };

            let state_probability = forward.state.clamp(1e-300, 1.0)?;
            let state_target = Tensor::from_vec(
                exact_state_target.clone(),
                (config.particles, horizon + 1, plan.num_facts),
                &device,
            )?;
            let state_active = Tensor::from_vec(
                exact_state_active.clone(),
                (config.particles, horizon + 1, plan.num_facts),
                &device,
            )?;
            let state_match_cells =
                ((state_probability.log()?.neg()? * state_target)? * &state_active)?;
            let state_match_numerator = state_match_cells.sum(2)?.sum(1)?;
            let state_match_denominator = state_active.sum(2)?.sum(1)?.clamp(1.0, f64::MAX)?;
            let state_match = state_match_numerator
                .broadcast_div(&state_match_denominator)?
                .mean_all()?;
            loss = (loss + (state_match * config.rho_transition)?)?;

            let failed_fact_loss = ((state_probability.log()?.neg()? * failed_fact)?.sum_all()?
                / config.particles as f64)?;
            loss = (loss + failed_fact_loss)?;

            let applicable = Tensor::from_vec(
                applicable_mask.clone(),
                (config.particles, horizon, plan.num_actions),
                &device,
            )?;
            let mut ranking_active = applicable_active.clone();
            for particle in 0..config.particles {
                let state = controller
                    .particle(particle)
                    .expect("ranking particle is in range");
                if insert_mode[particle] || matches!(state.phase(), Phase::GoalRepair) {
                    ranking_active[particle * horizon..(particle + 1) * horizon].fill(0.0);
                }
            }
            let active = Tensor::from_vec(ranking_active, (config.particles, horizon), &device)?;
            let temporal_ranking_logits = if config.temporal_tokens {
                let locked = (schedule_gate.ones_like()? - &schedule_gate)?;
                Some(
                    (effective_action_logits.broadcast_mul(&locked)?
                        + forward
                            .action_log_probability
                            .broadcast_mul(&schedule_gate)?)?,
                )
            } else {
                None
            };
            let applicability_barrier = applicability_ranking_barrier(
                if let Some(logits) = &temporal_ranking_logits {
                    logits
                } else if config.slot_slack_window == 0 {
                    &effective_action_logits
                } else {
                    &forward.action_log_probability
                },
                &applicable,
                &active,
                &focus,
                config.applicability_barrier_margin,
            )?;
            loss = (loss + (applicability_barrier * config.rho_precondition)?)?;
            let applicability_mass =
                applicability_mass_loss(&forward.action, &applicable, &active, &focus)?;
            loss = (loss
                + (applicability_mass
                    * (config.applicability_mass_weight * config.rho_precondition))?)?;

            let mut residual_families: Vec<&Tensor> = vec![&optimized_precondition];
            residual_families.extend(optimized_transition.iter());
            residual_families.push(&forward.goal);
            if let Some(top) = top_residual_loss(&residual_families, config.top_residual_fraction)?
            {
                loss = (loss + top)?;
            }

            let action_integrality = plan.action_integrality_per_particle(&forward.action)?;
            let state_integrality = plan.state_integrality_per_particle(&forward.state)?;
            let schedule_integrality = match &temporal {
                Some(temporal) => plan.temporal_assignment_integrality(&temporal.assignment)?,
                None => Tensor::zeros((config.particles, horizon, 1), DTYPE, &device)?,
            };
            let schedule_integrality_scale = Tensor::from_vec(
                temporal_progress
                    .iter()
                    .zip(&schedule_gate_values)
                    .map(|(&phase, &gate)| gate * config.integrality_scale_at(phase))
                    .collect::<Vec<_>>(),
                (config.particles, 1, 1),
                &device,
            )?;
            loss = (loss
                + (action_integrality
                    .broadcast_mul(&integrality_scale)?
                    .mean_all()?
                    * config.action_integrality_final)?)?;
            loss = (loss
                + (state_integrality
                    .broadcast_mul(&integrality_scale)?
                    .mean_all()?
                    * config.state_integrality_final)?)?;
            loss = (loss
                + (schedule_integrality
                    .broadcast_mul(&schedule_integrality_scale)?
                    .mean_all()?
                    * config.schedule_integrality_final)?)?;
            let noop_suffix = plan.noop_suffix_penalty(&forward.action)?;
            let slot_slack = plan.slot_slack_penalty(&forward.action, config.slot_slack_window)?;
            let slack_scale = 1.0
                - ramp(
                    progress,
                    config.causal_takeover_end,
                    config.remelt_stop_progress,
                );
            loss = (loss + (&slot_slack.mean_all()? * (config.slot_slack_weight * slack_scale))?)?;
            let per_particle_integrality_scale = Tensor::from_vec(
                vec![
                    if config.slot_slack_window == 0 {
                        exact_integrality
                    } else {
                        exact_integrality * ramp(progress, config.remelt_stop_progress, 1.0)
                    };
                    config.particles
                ],
                (config.particles, 1),
                &device,
            )?;
            loss = (loss
                + (scheduled_particle_mean(&noop_suffix, &per_particle_integrality_scale)?
                    * config.noop_suffix_weight)?)?;

            let action_bottleneck =
                bottleneck_norm_per_particle(&action_integrality, config.polish_p_norm)?;
            let state_bottleneck =
                bottleneck_norm_per_particle(&state_integrality, config.polish_p_norm)?;
            let schedule_bottleneck =
                bottleneck_norm_per_particle(&schedule_integrality, config.polish_p_norm)?;
            let worst_integrality =
                ((&action_bottleneck + &state_bottleneck)? + schedule_bottleneck)?;
            if matches!(stage, CausalStage::Polish) {
                let polish_weight = Tensor::ones((config.particles, 1), DTYPE, &device)?;
                loss = (loss
                    + (scheduled_particle_mean(&worst_integrality, &polish_weight)?
                        * config.worst_integrality_final)?)?;
                loss = (loss
                    + (bottleneck_norm_per_particle(
                        &forward.precondition,
                        config.polish_p_norm,
                    )?
                    .mean_all()?
                        * config.rho_precondition)?)?;
                for residual in &forward.transition {
                    loss = (loss
                        + (bottleneck_norm_per_particle(residual, config.polish_p_norm)?
                            .mean_all()?
                            * config.rho_transition)?)?;
                }
                loss = (loss
                    + (bottleneck_norm_per_particle(&forward.goal, config.polish_p_norm)?
                        .mean_all()?
                        * config.rho_goal)?)?;
            }

            let precondition_residual = forward.precondition.mean_all()?.to_scalar::<f64>()?;
            let transition_residual =
                forward
                    .transition
                    .iter()
                    .try_fold(0.0, |sum, residual| -> CandleResult<f64> {
                        Ok(sum + residual.mean_all()?.to_scalar::<f64>()?)
                    })?;
            let goal_residual = forward.goal.mean_all()?.to_scalar::<f64>()?;
            let total_residual = precondition_residual + transition_residual + goal_residual;
            outcome.best_total_residual = outcome.best_total_residual.min(total_residual);
            outcome.final_diagnostics = Diagnostics {
                precondition_residual,
                transition_residual,
                goal_residual,
                recurrent_precondition: execution_causal
                    .failed_precondition
                    .mean_all()?
                    .to_scalar::<f64>()?,
                recurrent_goal: execution_causal
                    .terminal_goal_by_goal
                    .mean_all()?
                    .to_scalar::<f64>()?,
                recurrent_survival: execution_causal
                    .surviving_goal_by_goal
                    .mean_all()?
                    .to_scalar::<f64>()?,
                recurrent_producer: execution_causal
                    .relaxed_goal_by_goal
                    .mean_all()?
                    .to_scalar::<f64>()?,
                recurrent_hardening: execution_hardening,
                slot_slack: slot_slack.mean_all()?.to_scalar::<f64>()?,
                insertion_raw: insertion_raw_diagnostic,
                insertion_supported: insertion_supported_diagnostic,
                backward_bridge_boundary: backward_bridge_boundary_diagnostic,
                backward_bridge_precondition: backward_bridge_precondition_diagnostic,
                anchor_trust: anchor_trust_diagnostic,
                action_integrality: action_integrality.mean_all()?.to_scalar::<f64>()?,
                state_integrality: state_integrality.mean_all()?.to_scalar::<f64>()?,
                causal_consensus: causal_consensus_diagnostic,
                causal_link_source: causal_link_diagnostics.0,
                causal_link_threat: causal_link_diagnostics.1,
                causal_link_integrality: causal_link_diagnostics.2,
            };

            let gradients = loss.backward()?;
            if config.trace_period > 0 && update % config.trace_period == 0 {
                let trace = outcome
                    .trace
                    .last_mut()
                    .expect("a due trace snapshot was recorded before loss construction");
                assert_eq!(trace.update, update, "gradient completes the current trace");
                trace.action_logit_gradients = gradients
                    .get(&action_logits)
                    .expect("the scheduled loss always depends on executable action logits")
                    .narrow(0, config.trace_particle, 1)?
                    .squeeze(0)?
                    .to_vec2::<f64>()?;
                if config.temporal_tokens {
                    trace.schedule_logit_gradients = match gradients.get(&schedule_logits) {
                        Some(gradient) => gradient
                            .narrow(0, config.trace_particle, 1)?
                            .squeeze(0)?
                            .to_vec2::<f64>()?,
                        None => {
                            assert!(
                                !temporal_unlocked[config.trace_particle],
                                "an unlocked interleaving must depend on schedule gates"
                            );
                            vec![vec![0.0; horizon]; horizon]
                        }
                    };
                }
            }
            optimizer.step(&gradients)?;
            causal_action_optimizer.step(&gradients)?;
            if let Some(lane) = link_lane.as_mut() {
                lane.optimizer.step(&gradients)?;
            }

            if progress < config.causal_transfer_end {
                action_logits.set(
                    &(action_logits.as_tensor()
                        + per_particle_noise(
                            &mut streams,
                            &phases,
                            config,
                            (config.particles, horizon, plan.num_actions),
                            &device,
                        )?
                        .broadcast_mul(&temporal_action_gate)?)?
                    .detach(),
                )?;
                state_logits.set(
                    &(state_logits.as_tensor()
                        + per_particle_noise(
                            &mut streams,
                            &phases,
                            config,
                            (config.particles, horizon, plan.num_facts),
                            &device,
                        )?)?
                    .detach(),
                )?;
                if config.temporal_tokens {
                    let schedule_noise = per_particle_noise(
                        &mut streams,
                        &temporal_progress,
                        config,
                        (config.particles, horizon, horizon),
                        &device,
                    )?
                    .broadcast_mul(&schedule_gate)?;
                    schedule_logits
                        .set(&(schedule_logits.as_tensor() + schedule_noise)?.detach())?;
                }
            }
            if matches!(config.causal_copy, CausalCopyMode::Staged)
                && matches!(stage, CausalStage::Discovery | CausalStage::Proof)
            {
                causal_action_logits.set(
                    &(causal_action_logits.as_tensor()
                        + per_particle_noise(
                            &mut causal_action_streams,
                            &phases,
                            config,
                            (config.particles, horizon, plan.num_actions),
                            &device,
                        )?)?
                    .detach(),
                )?;
            }
            if let Some(lane) = link_lane.as_mut() {
                let clipped = lane
                    .logits
                    .as_tensor()
                    .clamp(-config.action_logit_clip, config.action_logit_clip)?;
                lane.logits.set(&clipped)?;
            }
            action_logits.set(
                &action_logits
                    .as_tensor()
                    .clamp(-config.action_logit_clip, config.action_logit_clip)?,
            )?;
            causal_action_logits.set(
                &causal_action_logits
                    .as_tensor()
                    .clamp(-config.action_logit_clip, config.action_logit_clip)?,
            )?;
            state_logits.set(
                &state_logits
                    .as_tensor()
                    .clamp(-config.state_logit_clip, config.state_logit_clip)?,
            )?;
            schedule_logits.set(
                &schedule_logits
                    .as_tensor()
                    .clamp(-config.action_logit_clip, config.action_logit_clip)?,
            )?;

            if update % config.dual_period == 0 {
                duals.precondition = updated_dual(
                    &duals.precondition,
                    &forward.precondition,
                    config.rho_precondition,
                    config,
                )?;
                for family in 0..4 {
                    duals.transition[family] = updated_dual(
                        &duals.transition[family],
                        &forward.transition[family],
                        config.rho_transition,
                        config,
                    )?;
                }
                duals.goal = updated_dual(&duals.goal, &forward.goal, config.rho_goal, config)?;
            }

            let last_update = update + 1 == config.updates;
            if update % config.verify_period == 0 || last_update {
                checks_since_refresh += 1;
                if config.refresh && checks_since_refresh >= config.refresh_period {
                    checks_since_refresh = 0;
                    let refreshed = refresh_particles(
                        &action_logits,
                        &causal_action_logits,
                        &state_logits,
                        link_lane.as_mut(),
                        &mut optimizer,
                        &mut causal_action_optimizer,
                        config,
                        &mut streams,
                        &mut causal_action_streams,
                        horizon,
                        &plan,
                        &device,
                    )?;
                    outcome.refreshes += refreshed;
                    duals.reset_prefix(refreshed, config.particles, &device)?;
                    for particle in 0..refreshed {
                        controller
                            .reset_particle(particle)
                            .expect("refreshed particle indices are validated");
                        active_failure_action[particle] = None;
                        causal_goal_progress[particle] = 0.0;
                        let state_begin = particle * (horizon + 1) * plan.num_facts;
                        exact_state_target
                            [state_begin..state_begin + (horizon + 1) * plan.num_facts]
                            .fill(0.0);
                        exact_state_active
                            [state_begin..state_begin + (horizon + 1) * plan.num_facts]
                            .fill(0.0);
                        let action_begin = particle * horizon * plan.num_actions;
                        applicable_mask[action_begin..action_begin + horizon * plan.num_actions]
                            .fill(0.0);
                        applicable_active[particle * horizon..(particle + 1) * horizon].fill(0.0);
                        let memory_begin = particle * horizon * plan.num_preconditions;
                        failure_precondition_memory
                            [memory_begin..memory_begin + horizon * plan.num_preconditions]
                            .fill(1.0);
                        repair_start[particle] = 0;
                        goal_repair_start[particle] = horizon - initial_repair_radius;
                        insert_target[particle] = None;
                        insert_mode[particle] = false;
                        insert_at[particle] = None;
                        insert_required_fact[particle] = None;
                        full_prefix_insert_stalls[particle] = 0;
                        anchor_actions[particle] = None;
                    }
                }
                // Recompute diagnostics after the optimizer/noise step so the
                // soft values and the exact decoded checkpoint describe the
                // same complete tensor assignment.
                let verification_action_logits =
                    insertion_warp_logits(action_logits.as_tensor(), &insert_at)?;
                let verification_temporal = if config.temporal_tokens {
                    let token_action = repair_only_action_gradients(
                        &plan.action_distribution(
                            &verification_action_logits,
                            &action_temperature,
                        )?,
                        &temporal_action_gate,
                    )?;
                    let token_action = force_inactive_temporal_noops(
                        &token_action,
                        &temporal_inactive_noop,
                        transcription.noop_action(),
                    )?;
                    Some(monotone_interleaving_schedule(
                        token_action,
                        schedule_logits.as_tensor(),
                        &schedule_temperature,
                        &temporal_scaffold_order,
                        &temporal_repair_order,
                        &temporal_precedence,
                    )?)
                } else {
                    None
                };
                let verification_forward = if let Some(temporal) = &verification_temporal {
                    plan.forward_from_action_distribution(
                        temporal.action.clone(),
                        temporal.log_action.clone(),
                        state_logits.as_tensor(),
                        &state_temperature,
                    )?
                } else if config.slot_slack_window == 0 {
                    plan.forward(
                        &verification_action_logits,
                        state_logits.as_tensor(),
                        &action_temperature,
                        &state_temperature,
                    )?
                } else {
                    plan.forward_hybrid(
                        &verification_action_logits,
                        state_logits.as_tensor(),
                        &action_temperature,
                        &state_temperature,
                        config.slot_slack_window,
                    )?
                };
                let decoded_action = verification_forward.action.clone();
                let mut flattened_residuals =
                    vec![verification_forward.precondition.flatten_from(1)?];
                for residual in &verification_forward.transition {
                    flattened_residuals.push(residual.flatten_from(1)?);
                }
                flattened_residuals.push(verification_forward.goal.flatten_from(1)?);
                let residual_refs = flattened_residuals.iter().collect::<Vec<_>>();
                let max_residual_by_particle =
                    Tensor::cat(&residual_refs, 1)?.max(1)?.to_vec1::<f64>()?;
                let verification_action_integrality =
                    plan.action_integrality_per_particle(&verification_forward.action)?;
                let verification_state_integrality =
                    plan.state_integrality_per_particle(&verification_forward.state)?;
                let worst_integrality_by_particle = (&bottleneck_norm_per_particle(
                    &verification_action_integrality,
                    config.polish_p_norm,
                )? + &bottleneck_norm_per_particle(
                    &verification_state_integrality,
                    config.polish_p_norm,
                )?)?
                    .flatten_all()?
                    .to_vec1::<f64>()?;
                let mut remelt = Vec::new();

                for particle in 0..config.particles {
                    // A cycle rejection influences exactly the interval that
                    // just ended. Exact replay below may renew it, replace it
                    // by the newly selected cyclic achiever, or leave every
                    // achiever available again when the branch is repaired.
                    temporal_interval_rejected_action[particle].fill(None);
                    let execution_to_token = match &verification_temporal {
                        Some(temporal) => {
                            temporal_execution_to_token(&temporal.assignment, particle)?
                        }
                        None => (0..horizon).collect(),
                    };
                    let decoded = match &verification_temporal {
                        Some(temporal) => {
                            let token_actions = decode_particle(&temporal.token_action, particle)?;
                            execution_to_token
                                .iter()
                                .map(|&token| token_actions[token])
                                .collect()
                        }
                        None => decode_particle(&decoded_action, particle)?,
                    };
                    for (row, &token) in execution_to_token.iter().enumerate() {
                        temporal_token_row[particle][token] = row;
                    }
                    let mut slot_of_step = Vec::new();
                    let mut sequence = Vec::new();
                    for (slot, &action) in decoded.iter().enumerate() {
                        if let Some(operator) = transcription.action_source(action) {
                            slot_of_step.push(slot);
                            sequence.push(&operators[operator]);
                        }
                    }
                    let replay = replay_plan(&*task, &mut registry, &global_constraint, &sequence)
                        .map_err(|error| SgdError::Backend(error.message))?;
                    outcome.verifier_calls += 1;
                    let mut feedback = interpret(&replay, task.get_num_goals());
                    if feedback.solved.is_none() {
                        let exact = replay
                            .states
                            .last()
                            .expect("exact replay always retains at least the initial state");
                        let missing: Vec<ExplicitFact> = (0..task.get_num_goals())
                            .map(|goal| task.get_goal_fact(goal))
                            .filter(|fact| !fact.is_hold(exact, &registry))
                            .copied()
                            .collect();
                        feedback.goals_reached = task.get_num_goals().saturating_sub(missing.len());
                        feedback.missing_goals = missing;
                    }
                    outcome.best_goals_reached =
                        outcome.best_goals_reached.max(feedback.goals_reached);
                    outcome.longest_applicable_prefix = outcome
                        .longest_applicable_prefix
                        .max(feedback.applicable_prefix);
                    causal_goal_progress[particle] = if feedback.failure_step.is_none() {
                        // No-ops are removed before replay.  A completely
                        // applicable decoded sequence has finished the
                        // applicability curriculum regardless of how many
                        // tensor rows are padding.
                        1.0
                    } else {
                        assert!(
                            !sequence.is_empty(),
                            "an empty real-action sequence cannot fail an operator precondition"
                        );
                        feedback.applicable_prefix as f64 / sequence.len() as f64
                    };

                    let failed_slot = match feedback.failure_step {
                        None => None,
                        Some(step) => Some(slot_of_step.get(step).copied().ok_or_else(|| {
                            SgdError::Backend(format!(
                                "exact replay rejected transition {step}, but the decoded plan has \
                                 only {} real operators; global constraints after the final \
                                 operator are not represented by an action row",
                                slot_of_step.len()
                            ))
                        })?),
                    };
                    let failure_kind = if feedback.solved.is_some() {
                        "solved"
                    } else if failed_slot.is_none() {
                        "missing_goals"
                    } else if feedback.failure_fact.is_some() {
                        "first_precondition"
                    } else {
                        "global_constraint"
                    };
                    let checkpoint = ExactCheckpoint {
                        update: update + 1,
                        particle,
                        applicable_real_actions: feedback.applicable_prefix,
                        decoded_real_actions: sequence.len(),
                        goals_reached: feedback.goals_reached,
                        num_goals,
                        failure_kind,
                        failure_slot: failed_slot,
                        max_residual: max_residual_by_particle[particle],
                        worst_integrality: worst_integrality_by_particle[particle],
                        decoded_plan: slot_of_step
                            .iter()
                            .map(|&slot| {
                                transcription
                                    .action_source(decoded[slot])
                                    .expect("real-action slot retains its task operator")
                            })
                            .collect(),
                        decoded_slots: decoded
                            .iter()
                            .map(|&action| transcription.action_source(action))
                            .collect(),
                        missing_goals: feedback
                            .missing_goals
                            .iter()
                            .map(|fact| {
                                (0..task.get_num_goals())
                                    .position(|goal| task.get_goal_fact(goal) == fact)
                                    .expect("verifier missing facts belong to task goals")
                            })
                            .collect(),
                    };
                    if outcome
                        .best_exact_checkpoint
                        .as_ref()
                        .is_none_or(|incumbent| checkpoint.is_better_than(incumbent))
                    {
                        outcome.best_exact_checkpoint = Some(checkpoint);
                    }

                    if let Some(cost) = feedback.solved {
                        let plan_indices = slot_of_step
                            .iter()
                            .take(feedback.applicable_prefix)
                            .map(|&slot| {
                                transcription
                                    .action_source(decoded[slot])
                                    .expect("compressed plan slots contain real operators")
                            })
                            .collect();
                        outcome.status = SgdStatus::Solved;
                        outcome.plan = Some(plan_indices);
                        outcome.cost = Some(cost);
                        return Ok(outcome);
                    }

                    let state_slice_start = particle * (horizon + 1) * plan.num_facts;
                    exact_state_target
                        [state_slice_start..state_slice_start + (horizon + 1) * plan.num_facts]
                        .fill(0.0);
                    exact_state_active
                        [state_slice_start..state_slice_start + (horizon + 1) * plan.num_facts]
                        .fill(0.0);
                    let action_slice_start = particle * horizon * plan.num_actions;
                    applicable_mask
                        [action_slice_start..action_slice_start + horizon * plan.num_actions]
                        .fill(0.0);
                    applicable_active[particle * horizon..(particle + 1) * horizon].fill(0.0);
                    temporal_applicable_mask
                        [action_slice_start..action_slice_start + horizon * plan.num_actions]
                        .fill(0.0);

                    active_failure_action[particle] = failed_slot.map(|row| decoded[row]);
                    let protected_until = failed_slot.unwrap_or(horizon);
                    for row in 0..protected_until {
                        let begin = (particle * horizon + row) * plan.num_preconditions;
                        for weight in
                            &mut failure_precondition_memory[begin..begin + plan.num_preconditions]
                        {
                            *weight = 1.0 + (*weight - 1.0) * config.dual_decay;
                        }
                    }
                    let mut replay_step = 0usize;
                    for slot in 0..protected_until {
                        let exact = replay
                            .states
                            .get(replay_step)
                            .expect("exact replay retains every state in its applicable prefix");
                        let values = exact.get_state(&registry);
                        for (local_var, &task_var) in
                            transcription.primary_vars().iter().enumerate()
                        {
                            let fact =
                                transcription.var_offset()[local_var] as usize + values[task_var];
                            let index = (particle * (horizon + 1) + slot) * plan.num_facts + fact;
                            exact_state_target[index] = 1.0;
                            exact_state_active[index] = 1.0;
                        }
                        let placement_start = (particle * horizon + slot) * plan.num_actions;
                        for (action, target) in temporal_applicable_mask
                            [placement_start..placement_start + plan.num_actions]
                            .iter_mut()
                            .enumerate()
                        {
                            *target = match transcription.action_source(action) {
                                None => 1.0,
                                Some(operator) => f64::from(
                                    operators[operator]
                                        .preconditions()
                                        .iter()
                                        .all(|fact| fact.is_hold(exact, &registry)),
                                ),
                            };
                        }
                        if transcription.action_source(decoded[slot]).is_some() {
                            replay_step += 1;
                        }
                    }
                    if let Some(row) = failed_slot {
                        let exact = replay
                            .states
                            .last()
                            .expect("a failed replay retains its failure state");
                        let placement_start = (particle * horizon + row) * plan.num_actions;
                        for (action, target) in temporal_applicable_mask
                            [placement_start..placement_start + plan.num_actions]
                            .iter_mut()
                            .enumerate()
                        {
                            *target = match transcription.action_source(action) {
                                None => 1.0,
                                Some(operator) => f64::from(
                                    operators[operator]
                                        .preconditions()
                                        .iter()
                                        .all(|fact| fact.is_hold(exact, &registry)),
                                ),
                            };
                        }
                    }
                    if feedback.failure_step.is_none() {
                        assert_eq!(
                            replay_step, replay.applied,
                            "a complete applicable decode and replay must have equal lengths"
                        );
                        let exact = replay
                            .states
                            .last()
                            .expect("exact replay always retains an initial state");
                        let values = exact.get_state(&registry);
                        for (local_var, &task_var) in
                            transcription.primary_vars().iter().enumerate()
                        {
                            let fact =
                                transcription.var_offset()[local_var] as usize + values[task_var];
                            let index =
                                (particle * (horizon + 1) + horizon) * plan.num_facts + fact;
                            exact_state_target[index] = 1.0;
                            exact_state_active[index] = 1.0;
                        }
                    }

                    let local_failure_fact = feedback.failure_fact.map(|fact| {
                        let local_var = transcription
                            .primary_vars()
                            .iter()
                            .position(|&task_var| task_var == fact.var())
                            .expect("a failed operator precondition cannot be folded away");
                        transcription.var_offset()[local_var] as usize + fact.value()
                    });
                    if let (Some(row), Some(fact)) = (failed_slot, local_failure_fact) {
                        let rejected_action = decoded[row];
                        let mut matched = 0usize;
                        for (incidence, (&pre_action, &pre_fact)) in transcription
                            .pre_action()
                            .iter()
                            .zip(transcription.pre_fact())
                            .enumerate()
                        {
                            if pre_action as usize == rejected_action && pre_fact as usize == fact {
                                let index =
                                    (particle * horizon + row) * plan.num_preconditions + incidence;
                                failure_precondition_memory[index] =
                                    (failure_precondition_memory[index] * config.focus_growth)
                                        .min(config.focus_cap);
                                matched += 1;
                            }
                        }
                        assert_eq!(
                            matched, 1,
                            "canonical failed action/fact pair has exactly one incidence"
                        );
                    }
                    let mut local_missing_goals = Vec::with_capacity(feedback.missing_goals.len());
                    for fact in &feedback.missing_goals {
                        let local_var = transcription
                            .primary_vars()
                            .iter()
                            .position(|&task_var| task_var == fact.var())
                            .expect("a task goal cannot be folded false");
                        let local_fact =
                            transcription.var_offset()[local_var] as usize + fact.value();
                        let goal = transcription
                            .goal_facts()
                            .iter()
                            .position(|&candidate| candidate as usize == local_fact)
                            .expect("verifier missing goals belong to the task goal set");
                        local_missing_goals.push(goal);
                    }

                    let num_missing_goals = local_missing_goals.len();
                    let has_missing_goals = num_missing_goals > 0;
                    if temporal_unlocked[particle] {
                        for &goal in &local_missing_goals {
                            temporal_goal_required[particle][goal] = true;
                        }
                        let mut selected_action_by_token =
                            vec![transcription.noop_action(); horizon];
                        for (&token, &action) in execution_to_token.iter().zip(&decoded) {
                            selected_action_by_token[token] = action;
                        }
                        let freed = prune_stale_causal_memory(
                            &transcription,
                            &action_preconditions,
                            &selected_action_by_token,
                            &temporal_goal_tokens[particle],
                            &temporal_repair_order[particle],
                            &mut temporal_obligations[particle],
                            &mut temporal_obligation_focus[particle],
                            &mut temporal_unused_tokens[particle],
                            &mut temporal_precedence[particle],
                            &mut temporal_causal_precedence[particle],
                        );
                        for token in freed {
                            temporal_token_activation_update[particle][token] = None;
                            temporal_interval_rejected_action[particle][token] = None;
                            temporal_applicability_focus[particle][token] = 1.0;
                        }
                        // Every factual role whose decoded argmax is not one
                        // of its achievers is an exact discrete failure, even
                        // if replay stopped at an earlier inapplicable action.
                        // These token-local certificates are independent, so
                        // dualize them simultaneously. State applicability
                        // still follows the verifier's single first failure.
                        // The verifier reweights selected bad choices but
                        // never proposes replacement operators.
                        for (&token, &action) in execution_to_token.iter().zip(&decoded) {
                            let Some(fact) = temporal_obligations[particle][token] else {
                                continue;
                            };
                            let optional_goal_noop = temporal_goal_tokens[particle]
                                .iter()
                                .find(|&&(goal_token, _, _)| goal_token == token)
                                .is_some_and(|&(_, goal, _)| {
                                    !temporal_goal_required[particle][goal]
                                        && action == transcription.noop_action()
                                });
                            if optional_goal_noop
                                || fact_achievers[fact].binary_search(&action).is_ok()
                            {
                                continue;
                            }
                            temporal_obligation_focus[particle][token] =
                                (temporal_obligation_focus[particle][token] * config.focus_growth)
                                    .min(config.dual_cap);
                        }
                    }
                    let controller_feedback = if let Some(row) = failed_slot {
                        let exact = replay
                            .states
                            .last()
                            .expect("a rejected replay retains its failure state");
                        let target_start = (particle * horizon + row) * plan.num_actions;
                        let target =
                            &mut applicable_mask[target_start..target_start + plan.num_actions];
                        for (action, target_value) in target.iter_mut().enumerate() {
                            *target_value = match transcription.action_source(action) {
                                // The explicit no-op is always applicable.  It
                                // must not be silently excluded merely because
                                // some real operator is also applicable.
                                None => 1.0,
                                Some(operator) => f64::from(
                                    operators[operator]
                                        .preconditions()
                                        .iter()
                                        .all(|fact| fact.is_hold(exact, &registry)),
                                ),
                            };
                        }
                        assert_eq!(
                            target[plan.num_actions - 1],
                            1.0,
                            "the transcription's final action is the applicable no-op"
                        );
                        applicable_active[particle * horizon + row] = 1.0;
                        ExactFeedback::FirstFailure {
                            row,
                            fact: local_failure_fact,
                            missing_goals: local_missing_goals.clone(),
                            applicable_prefix: feedback.applicable_prefix,
                        }
                    } else {
                        ExactFeedback::ApplicableMissingGoals {
                            missing_goals: local_missing_goals.clone(),
                            applicable_prefix: feedback.applicable_prefix,
                        }
                    };
                    let controller_update = controller
                        .observe(particle, controller_feedback)
                        .map_err(|error| SgdError::Backend(error.to_string()))?;
                    let mut unlocked_token_anchor = None;
                    if config.temporal_tokens
                        && failed_slot.is_none()
                        && has_missing_goals
                        && sequence.len() < horizon
                        && !temporal_unlocked[particle]
                    {
                        let first_noop = decoded
                            .iter()
                            .position(|&action| action == transcription.noop_action())
                            .expect("a below-horizon decoded plan contains an explicit no-op");
                        let temporal_repair_start = goal_repair_start[particle].min(first_noop);
                        let dedicated_repair_start = if config.temporal_reserved_slots > 0 {
                            horizon - config.temporal_reserved_slots
                        } else {
                            temporal_repair_start
                        };
                        let available_noops = decoded[dedicated_repair_start..]
                            .iter()
                            .filter(|&&action| action == transcription.noop_action())
                            .count();
                        let minimum_capacity = num_goals
                            .checked_add(num_missing_goals)
                            .expect("goal and initial prerequisite capacity fits usize");
                        let repair_is_admissible = available_noops >= minimum_capacity;
                        if !repair_is_admissible {
                            // The first applicable scaffold may be too poor to
                            // anchor. The delete-aware objective must first
                            // leave one optional restoration token for every
                            // goal and one potential prerequisite token per
                            // currently missing goal. Reopen no-ops once for
                            // exploration without committing the scaffold.
                            if !temporal_probed[particle] {
                                probe_temporal_noops(
                                    particle,
                                    &action_logits,
                                    &mut optimizer,
                                    config,
                                    &mut streams[particle],
                                    horizon,
                                    temporal_repair_start,
                                    &plan,
                                    &device,
                                )?;
                                temporal_probed[particle] = true;
                            }
                            continue;
                        }
                        let (reopened, token_anchor, mut reopened_tokens) =
                            unlock_temporal_particle(
                                particle,
                                &action_logits,
                                &schedule_logits,
                                &mut optimizer,
                                config,
                                &mut streams[particle],
                                horizon,
                                dedicated_repair_start,
                                &plan,
                                &device,
                            )?;
                        assert!(
                            reopened > 0,
                            "an applicable goal-incomplete plan below the horizon retains at least one no-op token"
                        );
                        temporal_unlocked[particle] = true;
                        temporal_schedule_epoch[particle] = Some(update + 1);
                        temporal_last_progress_update[particle] = Some(update + 1);
                        goal_repair_start[particle] = 0;
                        temporal_repair_capacity[particle] = reopened;
                        reopened_tokens.sort_by_key(|&token| temporal_token_row[particle][token]);
                        temporal_repair_order[particle] = reopened_tokens.clone();
                        temporal_scaffold_order[particle] = execution_to_token
                            .iter()
                            .copied()
                            .filter(|token| !reopened_tokens.contains(token))
                            .collect();
                        assert_eq!(
                            temporal_scaffold_order[particle].len()
                                + temporal_repair_order[particle].len(),
                            horizon,
                            "unlock partitions every token into one ordered stream"
                        );
                        let mut scaffold_replay_step = 0usize;
                        for gap in 0..=temporal_scaffold_order[particle].len() {
                            let exact = replay
                                .states
                                .get(scaffold_replay_step)
                                .expect("applicable scaffold replay retains every gap state");
                            let fact_begin = (particle * (horizon + 1) + gap) * plan.num_facts;
                            for (variable, &task_variable) in
                                transcription.primary_vars().iter().enumerate()
                            {
                                for value in 0..transcription.var_domain()[variable] as usize {
                                    let fact = transcription.fact(variable, value) as usize;
                                    temporal_scaffold_gap_fact_values[fact_begin + fact] =
                                        f64::from(
                                            ExplicitFact::propositional(task_variable, value)
                                                .is_hold(exact, &registry),
                                        );
                                }
                            }
                            if let Some(&token) = temporal_scaffold_order[particle].get(gap)
                                && transcription.action_source(token_anchor[token]).is_some()
                            {
                                scaffold_replay_step += 1;
                            }
                        }
                        assert_eq!(
                            scaffold_replay_step,
                            sequence.len(),
                            "repair no-ops remove no real action from the admitted scaffold"
                        );
                        unlocked_token_anchor = Some(token_anchor);
                        temporal_frozen_noops[particle] = unlocked_token_anchor
                            .as_ref()
                            .expect("unlock captures token identities")
                            .iter()
                            .enumerate()
                            .filter_map(|(token, &action)| {
                                (action == transcription.noop_action()
                                    && !reopened_tokens.contains(&token))
                                .then_some(token)
                            })
                            .collect();
                        let real_tokens = unlocked_token_anchor
                            .as_ref()
                            .expect("unlock captures token identities")
                            .iter()
                            .enumerate()
                            .filter_map(|(token, &action)| {
                                (action != transcription.noop_action()).then_some(token)
                            })
                            .collect::<Vec<_>>();
                        temporal_precedence[particle]
                            .extend(real_tokens.windows(2).map(|pair| (pair[0], pair[1])));
                        let all_goals = (0..num_goals).collect::<Vec<_>>();
                        let goal_order = permute_goal_order(&all_goals, particle);
                        let goal_token_start = reopened_tokens
                            .len()
                            .checked_sub(goal_order.len())
                            .expect("admissible repair has one restoration token per goal");
                        let goal_tokens = reopened_tokens.split_off(goal_token_start);
                        temporal_goal_tokens[particle].clear();
                        for (goal, goal_token) in goal_order.into_iter().zip(goal_tokens) {
                            let goal_fact = transcription.goal_facts()[goal] as usize;
                            temporal_obligations[particle][goal_token] = Some(goal_fact);
                            temporal_token_activation_update[particle][goal_token] =
                                Some(update + 1);
                            temporal_goal_required[particle][goal] =
                                local_missing_goals.contains(&goal);
                            temporal_goal_tokens[particle].push((goal_token, goal, goal_fact));
                        }
                        temporal_unused_tokens[particle] = reopened_tokens;
                        for row in 0..horizon {
                            remelt_age[particle * horizon + row] = usize::MAX;
                        }
                    }
                    if config.temporal_tokens && temporal_unlocked[particle] && has_missing_goals {
                        for &(goal_token, goal, goal_fact) in &temporal_goal_tokens[particle] {
                            if !local_missing_goals.contains(&goal) {
                                continue;
                            }
                            let goal_row = temporal_token_row[particle][goal_token];
                            for (row, &action) in decoded.iter().enumerate().skip(goal_row + 1) {
                                if fact_threateners[goal_fact].binary_search(&action).is_err() {
                                    continue;
                                }
                                let threat_token = execution_to_token[row];
                                let threat_repair_index = temporal_repair_order[particle]
                                    .iter()
                                    .position(|&token| token == threat_token);
                                let goal_repair_index = temporal_repair_order[particle]
                                    .iter()
                                    .position(|&token| token == goal_token)
                                    .expect("a temporal goal belongs to the repair stream");
                                if let Some(threat_repair_index) = threat_repair_index {
                                    if threat_repair_index < goal_repair_index {
                                        // The immutable repair stream already
                                        // places this threat before re-achievement.
                                        continue;
                                    }
                                    if temporal_order_conflicts[particle]
                                        .insert((threat_token, goal_token))
                                    {
                                        outcome.temporal_order_conflicts += 1;
                                    }
                                    // Reversing two repair tokens would violate
                                    // this particle's represented goal order.
                                    // Keep its terminal loss positive; another
                                    // particle represents the opposite order.
                                    continue;
                                }
                                if threat_token != goal_token
                                    && !temporal_precedence[particle]
                                        .contains(&(threat_token, goal_token))
                                {
                                    // Exact replay separated a later possible
                                    // overwrite of this still-missing goal.
                                    if !try_add_temporal_precedence(
                                        &mut temporal_precedence[particle],
                                        &temporal_scaffold_order[particle],
                                        &temporal_repair_order[particle],
                                        (threat_token, goal_token),
                                    ) && temporal_order_conflicts[particle]
                                        .insert((threat_token, goal_token))
                                    {
                                        outcome.temporal_order_conflicts += 1;
                                    }
                                }
                            }
                        }
                    }
                    if config.temporal_tokens
                        && temporal_unlocked[particle]
                        && let (Some(row), Some(fact)) = (failed_slot, local_failure_fact)
                    {
                        let consumer = execution_to_token[row];
                        if temporal_obligations[particle][consumer].is_some() {
                            temporal_applicability_focus[particle][consumer] =
                                (temporal_applicability_focus[particle][consumer]
                                    * config.focus_growth)
                                    .min(config.dual_cap);
                            // The exact verifier has exposed the first missing
                            // prerequisite of an obligated token. Reuse an
                            // existing preceding fact role when possible;
                            // otherwise allocate fresh symmetric capacity.
                            // This schedules facts and order only and never
                            // selects an operator.
                            let repeats_on_branch = repeated_fact_cycle(
                                &temporal_obligations[particle],
                                &temporal_causal_precedence[particle],
                                consumer,
                                fact,
                            );
                            if repeats_on_branch {
                                if temporal_causal_cycles[particle].insert((consumer, fact)) {
                                    outcome.temporal_causal_cycles += 1;
                                }
                                let selected_action = decoded[row];
                                let obligation_fact = temporal_obligations[particle][consumer]
                                    .expect("repair consumer retains its fact role");
                                let achievers = &fact_achievers[obligation_fact];
                                if achievers.binary_search(&selected_action).is_ok()
                                    && achievers.iter().any(|&action| action != selected_action)
                                {
                                    temporal_interval_rejected_action[particle][consumer] =
                                        Some(selected_action);
                                    outcome.temporal_cycle_interventions += 1;
                                    reset_temporal_token_action(
                                        particle,
                                        consumer,
                                        &action_logits,
                                        &mut optimizer,
                                        config,
                                        &mut streams[particle],
                                        horizon,
                                        &plan,
                                        &device,
                                    )?;
                                    temporal_token_activation_update[particle][consumer] =
                                        Some(update + 1);
                                }
                                // Do not unroll a repeated fact into another
                                // token. A different achiever must break the
                                // branch during the next verifier interval.
                            } else {
                                let already_assigned = temporal_precedence[particle].iter().any(
                                    |&(producer, target)| {
                                        target == consumer
                                            && temporal_obligations[particle][producer]
                                                == Some(fact)
                                    },
                                );
                                if !already_assigned {
                                    if let Some(producer) = latest_preceding_obligation(
                                        &temporal_obligations[particle],
                                        &temporal_repair_order[particle],
                                        consumer,
                                        fact,
                                    ) {
                                        assert!(try_add_temporal_precedence(
                                            &mut temporal_precedence[particle],
                                            &temporal_scaffold_order[particle],
                                            &temporal_repair_order[particle],
                                            (producer, consumer),
                                        ));
                                        if !temporal_causal_precedence[particle]
                                            .contains(&(producer, consumer))
                                        {
                                            temporal_causal_precedence[particle]
                                                .push((producer, consumer));
                                        }
                                    } else if let Some(producer) = take_latest_preceding_token(
                                        &mut temporal_unused_tokens[particle],
                                        &temporal_repair_order[particle],
                                        consumer,
                                    ) {
                                        reset_temporal_token_action(
                                            particle,
                                            producer,
                                            &action_logits,
                                            &mut optimizer,
                                            config,
                                            &mut streams[particle],
                                            horizon,
                                            &plan,
                                            &device,
                                        )?;
                                        temporal_obligations[particle][producer] = Some(fact);
                                        temporal_token_activation_update[particle][producer] =
                                            Some(update + 1);
                                        assert!(try_add_temporal_precedence(
                                            &mut temporal_precedence[particle],
                                            &temporal_scaffold_order[particle],
                                            &temporal_repair_order[particle],
                                            (producer, consumer),
                                        ));
                                        temporal_causal_precedence[particle]
                                            .push((producer, consumer));
                                    }
                                }
                            }
                        } else if temporal_scaffold_order[particle].contains(&consumer) {
                            // The scaffold was exactly applicable at unlock,
                            // so this failure was introduced by an interleaved
                            // repair action. Restore its first missing fact
                            // before the immutable scaffold consumer. The fact
                            // role remains symmetric over all achievers.
                            let already_assigned =
                                temporal_precedence[particle]
                                    .iter()
                                    .any(|&(producer, target)| {
                                        target == consumer
                                            && temporal_obligations[particle][producer]
                                                == Some(fact)
                                    });
                            if !already_assigned {
                                // Prefer a reusable fact role, then unused
                                // capacity, but only if its cross-stream edge
                                // leaves at least one complete interleaving.
                                let producer = scaffold_repair_candidate(
                                    &temporal_obligations[particle],
                                    &temporal_unused_tokens[particle],
                                    &temporal_scaffold_order[particle],
                                    &temporal_repair_order[particle],
                                    &temporal_precedence[particle],
                                    consumer,
                                    fact,
                                );
                                if let Some((producer, fresh)) = producer {
                                    assert!(try_add_temporal_precedence(
                                        &mut temporal_precedence[particle],
                                        &temporal_scaffold_order[particle],
                                        &temporal_repair_order[particle],
                                        (producer, consumer),
                                    ));
                                    temporal_causal_precedence[particle].push((producer, consumer));
                                    if fresh {
                                        let unused_index = temporal_unused_tokens[particle]
                                            .iter()
                                            .position(|&token| token == producer)
                                            .expect("a fresh scaffold repair token is unused");
                                        temporal_unused_tokens[particle].remove(unused_index);
                                        reset_temporal_token_action(
                                            particle,
                                            producer,
                                            &action_logits,
                                            &mut optimizer,
                                            config,
                                            &mut streams[particle],
                                            horizon,
                                            &plan,
                                            &device,
                                        )?;
                                        temporal_obligations[particle][producer] = Some(fact);
                                        temporal_token_activation_update[particle][producer] =
                                            Some(update + 1);
                                    }
                                    outcome.temporal_scaffold_repairs += 1;
                                } else if temporal_order_conflicts[particle]
                                    .insert((consumer, consumer))
                                {
                                    // Every matching or unused repair token
                                    // would contradict earlier exact ordering.
                                    outcome.temporal_order_conflicts += 1;
                                }
                            }
                        }
                    }
                    if let Some(token_anchor) = unlocked_token_anchor {
                        // `decoded` is indexed by execution row, while the
                        // temporal trust loss is indexed by latent token. The
                        // unlock permutation deliberately makes those orders
                        // differ, so preserve the token identities captured
                        // immediately before reopening the no-op capacity.
                        anchor_actions[particle] = Some((token_anchor, horizon));
                    } else if controller_update.exact_progress
                        && (!config.temporal_tokens || !temporal_unlocked[particle])
                    {
                        anchor_actions[particle] =
                            Some((decoded.clone(), failed_slot.unwrap_or(horizon)));
                    }

                    let mut started_insertion = false;
                    let new_insert_target = failed_slot.zip(local_failure_fact);
                    if new_insert_target != insert_target[particle] {
                        insert_target[particle] = new_insert_target;
                        full_prefix_insert_stalls[particle] = 0;
                        if let Some((row, _)) = new_insert_target {
                            repair_start[particle] = row.saturating_sub(initial_repair_radius);
                            if !insert_mode[particle] {
                                let mature_prefix = row as f64
                                    >= config.insertion_min_prefix_fraction * horizon as f64;
                                insert_mode[particle] = mature_prefix
                                    && !config.temporal_tokens
                                    && row + 1 < horizon
                                    && config.insertion_repair_weight > 0.0
                                    && config.anchor_trust_weight > 0.0;
                                insert_at[particle] = insert_mode[particle].then_some(row);
                                insert_required_fact[particle] = insert_mode[particle].then_some(
                                    new_insert_target
                                        .expect("mature insertion has a fact target")
                                        .1,
                                );
                                started_insertion = insert_mode[particle];
                            }
                        }
                    }
                    if failed_slot.is_none()
                        && controller_update.exact_progress
                        && !temporal_unlocked[particle]
                    {
                        goal_repair_start[particle] = horizon - initial_repair_radius;
                    }

                    let mut effective_remelt = controller_update.remelt;
                    if let Some(window) = controller_update.remelt {
                        let controller_state = controller
                            .particle(particle)
                            .expect("verified particle remains in range");
                        if goal_bridge_is_active(
                            controller_state.phase(),
                            controller_state.missing_goal_mask(),
                        ) {
                            goal_repair_start[particle] =
                                goal_repair_start[particle].min(window.start);
                        }
                        if let Some((row, _)) = insert_target[particle] {
                            repair_start[particle] = window.start.min(row);
                            if window.start == 0 {
                                full_prefix_insert_stalls[particle] += 1;
                                if full_prefix_insert_stalls[particle] >= 2 {
                                    insert_mode[particle] = false;
                                    insert_at[particle] = None;
                                    insert_required_fact[particle] = None;
                                }
                            }
                            if insert_mode[particle] {
                                effective_remelt = (window.start < row)
                                    .then_some(RemeltWindow { end: row, ..window });
                            }
                        } else if failed_slot.is_none() {
                            goal_repair_start[particle] = window.start;
                        }
                    }

                    // Temporal mode uses a dedicated repair-only scheduler.
                    // Applying direct row-window remelts either before or
                    // after unlock would confound scaffold discovery and could
                    // mutate the very checkpoint that repair is meant to keep.
                    if config.temporal_tokens {
                        effective_remelt = None;
                    }

                    if temporal_unlocked[particle] && controller_update.exact_progress {
                        temporal_last_progress_update[particle] = Some(update + 1);
                    }
                    let temporal_restart_due = config.temporal_restart_patience > 0
                        && temporal_unlocked[particle]
                        && update + 1
                            >= temporal_last_progress_update[particle]
                                .expect("an unlocked temporal particle has a progress epoch")
                                + config.temporal_restart_patience;
                    if temporal_restart_due && progress < config.remelt_stop_progress {
                        restart_temporal_repair_particle(
                            particle,
                            &action_logits,
                            &schedule_logits,
                            &mut optimizer,
                            config,
                            &mut streams[particle],
                            &temporal_repair_order[particle],
                            horizon,
                            &plan,
                            &device,
                        )?;
                        for &token in &temporal_repair_order[particle] {
                            remelt_age[particle * horizon + token] = usize::MAX;
                        }
                        let state_begin = particle * (horizon + 1) * plan.num_facts;
                        exact_state_target
                            [state_begin..state_begin + (horizon + 1) * plan.num_facts]
                            .fill(0.0);
                        exact_state_active
                            [state_begin..state_begin + (horizon + 1) * plan.num_facts]
                            .fill(0.0);
                        let action_begin = particle * horizon * plan.num_actions;
                        temporal_applicable_mask
                            [action_begin..action_begin + horizon * plan.num_actions]
                            .fill(0.0);
                        applicable_mask[action_begin..action_begin + horizon * plan.num_actions]
                            .fill(0.0);
                        applicable_active[particle * horizon..(particle + 1) * horizon].fill(0.0);
                        temporal_last_progress_update[particle] = Some(update + 1);
                        effective_remelt = None;
                        outcome.temporal_restarts += 1;
                    }

                    if started_insertion && progress < config.remelt_stop_progress {
                        // The tail was an ordinary plan row during discovery
                        // and may already be saturated. It becomes the sole
                        // insertion variable under the warp, so reopen exactly
                        // that raw coordinate and its optimizer history.
                        remelt_age[particle * horizon + horizon - 1] = usize::MAX;
                        remelt.push(RemeltWindow {
                            particle,
                            start: horizon - 1,
                            end: horizon,
                            phase: controller_update.phase,
                            radius: 1,
                            ordinal: 1,
                        });
                    }

                    if let Some(window) = effective_remelt
                        && progress < config.remelt_stop_progress
                    {
                        // Exact polish is precisely where a low-entropy wrong
                        // causal assignment can otherwise become permanent.
                        // A controller-requested remelt remains row-local,
                        // clears Adam memory, and temporarily removes
                        // integrality on only that window, so honoring it does
                        // not turn verification into branching or discard the
                        // rest of the plan.
                        for row in window.start..window.end {
                            remelt_age[window.particle * horizon + row] = usize::MAX;
                        }
                        remelt.push(window);
                    }
                }

                if !remelt.is_empty() {
                    for window in &remelt {
                        // Action rows [start,end) determine states
                        // S[start+1..=end]. Preserve the exact incoming
                        // boundary S[start], but release every state that the
                        // reopened actions are allowed to change.
                        for row in window.start + 1..=window.end {
                            let begin = (window.particle * (horizon + 1) + row) * plan.num_facts;
                            exact_state_target[begin..begin + plan.num_facts].fill(0.0);
                            exact_state_active[begin..begin + plan.num_facts].fill(0.0);
                        }
                    }
                    remelt_direct_windows(
                        &action_logits,
                        &causal_action_logits,
                        &state_logits,
                        link_lane.as_mut(),
                        &mut optimizer,
                        &mut causal_action_optimizer,
                        &remelt,
                        !matches!(config.causal_copy, CausalCopyMode::Staged)
                            || matches!(stage, CausalStage::Shadow),
                        config,
                        &mut streams,
                        &mut causal_action_streams,
                        horizon,
                        &plan,
                        &device,
                    )?;
                    outcome.remelts += remelt.len();
                }
            }
        }
    }
    Ok(outcome)
}

/// Reopen the no-op rows of an applicable but capacity-inadmissible scaffold
/// without moving or committing any part of that scaffold.
#[allow(clippy::too_many_arguments)]
fn probe_temporal_noops(
    particle: usize,
    action_logits: &Var,
    optimizer: &mut Adam,
    config: &SgdConfig,
    stream: &mut ChaCha8Rng,
    horizon: usize,
    repair_start: usize,
    plan: &TensorPlan,
    device: &Device,
) -> CandleResult<usize> {
    assert!(
        config.temporal_tokens,
        "only temporal plans have probe rows"
    );
    assert!(particle < config.particles, "probe particle is in range");
    assert!(repair_start < horizon, "probe suffix is nonempty");
    let noop = plan.num_actions - 1;
    let stride = horizon * plan.num_actions;
    let particle_begin = particle * stride;
    let mut action = action_logits.flatten_all()?.to_vec1::<f64>()?;
    let mut keep_action = vec![1.0f64; action.len()];
    let mut reopened = 0usize;
    for row in repair_start..horizon {
        let begin = particle_begin + row * plan.num_actions;
        let end = begin + plan.num_actions;
        let decoded = action[begin..end]
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .expect("a temporal row has the explicit no-op");
        if decoded != noop {
            continue;
        }
        for (logit, noise) in action[begin..end].iter_mut().zip(normal_vec(
            stream,
            plan.num_actions,
            config.remelt_noise,
        )) {
            *logit = config.remelt_shrink * *logit + noise;
        }
        let largest_real = action[begin..begin + noop]
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .expect("a planning task has at least one real action");
        action[begin + noop] = largest_real + 0.25;
        keep_action[begin..end].fill(0.0);
        reopened += 1;
    }
    assert!(
        reopened > 0,
        "a below-horizon plan exposes a no-op probe row"
    );
    action_logits.set(&Tensor::from_vec(
        action,
        (config.particles, horizon, plan.num_actions),
        device,
    )?)?;
    optimizer.reset_moments_where(&[
        Tensor::from_vec(
            keep_action,
            (config.particles, horizon, plan.num_actions),
            device,
        )?,
        Tensor::ones((config.particles, horizon, plan.num_facts), DTYPE, device)?,
        Tensor::ones((config.particles, horizon, horizon), DTYPE, device)?,
    ])?;
    Ok(reopened)
}

/// Preserve discovered real token identities while reopening explicit no-op
/// tokens as capacity for a missing causal chain.
///
/// No token is moved and no operator is chosen here. Subsequent gradients
/// jointly specialize the reopened no-op tokens and assign every token to a
/// unique execution row through the temporal schedule.
#[allow(clippy::too_many_arguments)]
fn unlock_temporal_particle(
    particle: usize,
    action_logits: &Var,
    schedule_logits: &Var,
    optimizer: &mut Adam,
    config: &SgdConfig,
    stream: &mut ChaCha8Rng,
    horizon: usize,
    repair_start: usize,
    plan: &TensorPlan,
    device: &Device,
) -> CandleResult<(usize, Vec<usize>, Vec<usize>)> {
    assert!(
        config.temporal_tokens,
        "only temporal plans may be unlocked"
    );
    assert!(particle < config.particles, "unlock particle is in range");
    assert!(repair_start < horizon, "temporal repair suffix is nonempty");
    let noop = plan.num_actions - 1;
    let action_stride = horizon * plan.num_actions;
    let schedule_stride = horizon * horizon;
    let mut action = action_logits.flatten_all()?.to_vec1::<f64>()?;
    let mut keep_action = vec![1.0f64; action.len()];
    let action_particle = particle * action_stride;
    let mut reopened = 0usize;
    let mut reopened_tokens = Vec::new();
    let mut decoded_tokens = Vec::with_capacity(horizon);
    for token in 0..horizon {
        let begin = action_particle + token * plan.num_actions;
        let end = begin + plan.num_actions;
        let decoded = action[begin..end]
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .expect("an action token has at least the explicit no-op");
        decoded_tokens.push(decoded);
        reopened += usize::from(decoded == noop && token >= repair_start);
        if decoded == noop && token >= repair_start {
            reopened_tokens.push(token);
        }
    }
    assert!(reopened > 0, "unlock requires explicit no-op capacity");
    for token in repair_start..horizon {
        if decoded_tokens[token] == noop {
            let begin = action_particle + token * plan.num_actions;
            for (logit, noise) in action[begin..begin + plan.num_actions]
                .iter_mut()
                .zip(normal_vec(stream, plan.num_actions, config.remelt_noise))
            {
                *logit = config.remelt_shrink * *logit + noise;
            }
            let largest_real = action[begin..begin + noop]
                .iter()
                .copied()
                .max_by(f64::total_cmp)
                .expect("a planning task has at least one real action");
            action[begin + noop] = largest_real + 0.25;
            keep_action[begin..begin + plan.num_actions].fill(0.0);
        }
    }
    action_logits.set(&Tensor::from_vec(
        action,
        (config.particles, horizon, plan.num_actions),
        device,
    )?)?;
    // Frozen scaffold tokens receive zero gradient after unlock. Reset the
    // complete particle here so stale Adam momentum cannot move them despite
    // that zero gradient; repair tokens also need fresh moments after remelt.
    keep_action[action_particle..action_particle + action_stride].fill(0.0);

    // The locked forward schedule ignored these logits. The old diagonal bias
    // encoded an unrestricted permutation and has no meaning for lattice
    // gates `(consumed_scaffold, consumed_repair)`. Start every gate near its
    // maximum-entropy decision boundary, with particle-local noise for broken
    // symmetry, and discard its stale Adam moments.
    let mut schedule = schedule_logits.flatten_all()?.to_vec1::<f64>()?;
    let mut keep_schedule = vec![1.0f64; config.particles * schedule_stride];
    let schedule_begin = particle * schedule_stride;
    schedule[schedule_begin..schedule_begin + schedule_stride].copy_from_slice(&normal_vec(
        stream,
        schedule_stride,
        0.1,
    ));
    schedule_logits.set(&Tensor::from_vec(
        schedule,
        (config.particles, horizon, horizon),
        device,
    )?)?;
    keep_schedule[schedule_begin..schedule_begin + schedule_stride].fill(0.0);
    optimizer.reset_moments_where(&[
        Tensor::from_vec(
            keep_action,
            (config.particles, horizon, plan.num_actions),
            device,
        )?,
        Tensor::ones((config.particles, horizon, plan.num_facts), DTYPE, device)?,
        Tensor::from_vec(keep_schedule, (config.particles, horizon, horizon), device)?,
    ])?;
    // Assert the optimizer and model own precisely the same schedule shape.
    assert_eq!(
        schedule_logits.dims(),
        &[config.particles, horizon, horizon],
        "temporal schedule shape matches its optimizer mask"
    );
    assert_eq!(reopened_tokens.len(), reopened);
    Ok((reopened, decoded_tokens, reopened_tokens))
}

/// Draw a fresh full-support repair subplan while preserving its scaffold.
///
/// This is a stochastic optimizer restart, not a plan transition or operator
/// choice: every repair action and every monotone interleaving gate receives a
/// continuous random logit, while scaffold logits and all verifier-derived
/// fact obligations remain untouched.
#[allow(clippy::too_many_arguments)]
fn restart_temporal_repair_particle(
    particle: usize,
    action_logits: &Var,
    schedule_logits: &Var,
    optimizer: &mut Adam,
    config: &SgdConfig,
    stream: &mut ChaCha8Rng,
    repair_tokens: &[usize],
    horizon: usize,
    plan: &TensorPlan,
    device: &Device,
) -> CandleResult<()> {
    assert!(config.temporal_tokens, "only temporal repair can restart");
    assert!(particle < config.particles);
    assert!(
        !repair_tokens.is_empty(),
        "an unlocked repair stream is nonempty"
    );
    let action_stride = horizon * plan.num_actions;
    let action_particle = particle * action_stride;
    let mut action = action_logits.flatten_all()?.to_vec1::<f64>()?;
    let mut keep_action = vec![1.0f64; action.len()];
    for &token in repair_tokens {
        assert!(token < horizon, "repair token lies inside the horizon");
        let begin = action_particle + token * plan.num_actions;
        action[begin..begin + plan.num_actions].copy_from_slice(&normal_vec(
            stream,
            plan.num_actions,
            0.7,
        ));
        keep_action[begin..begin + plan.num_actions].fill(0.0);
    }
    action_logits.set(&Tensor::from_vec(
        action,
        (config.particles, horizon, plan.num_actions),
        device,
    )?)?;

    let schedule_stride = horizon * horizon;
    let schedule_begin = particle * schedule_stride;
    let mut schedule = schedule_logits.flatten_all()?.to_vec1::<f64>()?;
    schedule[schedule_begin..schedule_begin + schedule_stride].copy_from_slice(&normal_vec(
        stream,
        schedule_stride,
        0.1,
    ));
    schedule_logits.set(&Tensor::from_vec(
        schedule,
        (config.particles, horizon, horizon),
        device,
    )?)?;
    let mut keep_schedule = vec![1.0f64; config.particles * schedule_stride];
    keep_schedule[schedule_begin..schedule_begin + schedule_stride].fill(0.0);
    optimizer.reset_moments_where(&[
        Tensor::from_vec(
            keep_action,
            (config.particles, horizon, plan.num_actions),
            device,
        )?,
        Tensor::ones((config.particles, horizon, plan.num_facts), DTYPE, device)?,
        Tensor::from_vec(keep_schedule, (config.particles, horizon, horizon), device)?,
    ])?;
    Ok(())
}

/// Reopen only the causal window selected by exact-feedback plateau diagnosis.
#[allow(clippy::too_many_arguments)]
fn remelt_direct_windows(
    action_logits: &Var,
    causal_action_logits: &Var,
    state_logits: &Var,
    mut link_lane: Option<&mut CausalLinkLane>,
    optimizer: &mut Adam,
    causal_action_optimizer: &mut Adam,
    windows: &[RemeltWindow],
    remelt_causal: bool,
    config: &SgdConfig,
    streams: &mut [ChaCha8Rng],
    causal_action_streams: &mut [ChaCha8Rng],
    horizon: usize,
    plan: &TensorPlan,
    device: &Device,
) -> CandleResult<()> {
    assert_eq!(
        link_lane.is_some(),
        config.causal_links_enabled(),
        "remelt receives exactly the causal-link lane enabled by the configuration"
    );
    let mut action = action_logits.flatten_all()?.to_vec1::<f64>()?;
    let mut causal_action = causal_action_logits.flatten_all()?.to_vec1::<f64>()?;
    let mut state = state_logits.flatten_all()?.to_vec1::<f64>()?;
    let mut link = match link_lane.as_ref() {
        Some(lane) => Some(lane.logits.flatten_all()?.to_vec1::<f64>()?),
        None => None,
    };
    let mut keep_action = vec![1.0f64; action.len()];
    let mut keep_causal_action = vec![1.0f64; causal_action.len()];
    let mut keep_state = vec![1.0f64; state.len()];
    let mut keep_link = link.as_ref().map(|values| vec![1.0f64; values.len()]);
    let action_stride = horizon * plan.num_actions;
    let state_stride = horizon * plan.num_facts;
    let link_rows = horizon + 1;
    let link_consumer_stride = plan.num_facts * link_rows;
    let link_stride = link_rows * link_consumer_stride;

    for window in windows {
        assert!(
            window.particle < config.particles,
            "remelt particle is in range"
        );
        assert!(
            window.start < window.end && window.end <= horizon,
            "controller emits a nonempty in-range remelt window"
        );
        let action_begin = window.particle * action_stride + window.start * plan.num_actions;
        let action_end = window.particle * action_stride + window.end * plan.num_actions;
        for (logit, noise) in action[action_begin..action_end].iter_mut().zip(normal_vec(
            &mut streams[window.particle],
            action_end - action_begin,
            config.remelt_noise,
        )) {
            *logit = config.remelt_shrink * *logit + noise;
        }
        keep_action[action_begin..action_end].fill(0.0);
        if remelt_causal {
            for (logit, noise) in
                causal_action[action_begin..action_end]
                    .iter_mut()
                    .zip(normal_vec(
                        &mut causal_action_streams[window.particle],
                        action_end - action_begin,
                        config.remelt_noise,
                    ))
            {
                *logit = config.remelt_shrink * *logit + noise;
            }
            keep_causal_action[action_begin..action_end].fill(0.0);
        }

        // State-logit row i parameterizes exact state row i+1. Preserve the
        // incoming boundary S[start] and reopen S[start+1..=end].
        let state_start = window.start;
        let state_end = window.end;
        let state_begin = window.particle * state_stride + state_start * plan.num_facts;
        let state_finish = window.particle * state_stride + state_end * plan.num_facts;
        for (logit, noise) in state[state_begin..state_finish].iter_mut().zip(normal_vec(
            &mut streams[window.particle],
            state_finish - state_begin,
            config.remelt_noise,
        )) {
            *logit = config.remelt_shrink * *logit + noise;
        }
        keep_state[state_begin..state_finish].fill(0.0);

        // A changed producer or delete can invalidate every later consumer,
        // including the terminal goals. Reopen the complete downstream causal
        // cone rather than retaining witnesses for a trajectory that no longer
        // exists.
        if remelt_causal && let Some(lane) = link_lane.as_mut() {
            let link_consumer_end = horizon + 1;
            let link_begin = window.particle * link_stride + window.start * link_consumer_stride;
            let link_end = window.particle * link_stride + link_consumer_end * link_consumer_stride;
            let link = link
                .as_mut()
                .expect("causal-link values exist exactly when their lane exists");
            let keep_link = keep_link
                .as_mut()
                .expect("causal-link moment mask exists exactly when its lane exists");
            for (logit, noise) in link[link_begin..link_end].iter_mut().zip(normal_vec(
                &mut lane.streams[window.particle],
                link_end - link_begin,
                config.remelt_noise,
            )) {
                *logit = config.remelt_shrink * *logit + noise;
            }
            keep_link[link_begin..link_end].fill(0.0);
        }
    }

    action_logits.set(&Tensor::from_vec(
        action,
        (config.particles, horizon, plan.num_actions),
        device,
    )?)?;
    causal_action_logits.set(&Tensor::from_vec(
        causal_action,
        (config.particles, horizon, plan.num_actions),
        device,
    )?)?;
    state_logits.set(&Tensor::from_vec(
        state,
        (config.particles, horizon, plan.num_facts),
        device,
    )?)?;
    reset_direct_moments_preserving_schedule(
        optimizer,
        Tensor::from_vec(
            keep_action,
            (config.particles, horizon, plan.num_actions),
            device,
        )?,
        Tensor::from_vec(
            keep_state,
            (config.particles, horizon, plan.num_facts),
            device,
        )?,
        config,
        horizon,
        device,
    )?;
    causal_action_optimizer.reset_moments_where(&[Tensor::from_vec(
        keep_causal_action,
        (config.particles, horizon, plan.num_actions),
        device,
    )?])?;
    match (link_lane, link, keep_link) {
        (Some(lane), Some(link), Some(keep_link)) => {
            lane.logits.set(&Tensor::from_vec(
                link,
                (config.particles, horizon + 1, plan.num_facts, horizon + 1),
                device,
            )?)?;
            lane.optimizer.reset_moments_where(&[Tensor::from_vec(
                keep_link,
                (config.particles, horizon + 1, plan.num_facts, horizon + 1),
                device,
            )?])
        }
        (None, None, None) => Ok(()),
        _ => unreachable!("causal-link remelt state is initialized atomically"),
    }
}

/// Replace the first `refresh_particles` particles with independently random
/// complete plans and clear their optimizer memory.
///
/// This is the note's probabilistic-completeness mechanism. It is off by
/// default, because with it on a solved instance cannot be attributed to
/// gradient descent rather than to random sampling. The dense causal-link
/// ticket is refreshed only when that optional optimization lane exists.
#[allow(clippy::too_many_arguments)]
fn refresh_particles(
    action_logits: &Var,
    causal_action_logits: &Var,
    state_logits: &Var,
    mut link_lane: Option<&mut CausalLinkLane>,
    optimizer: &mut Adam,
    causal_action_optimizer: &mut Adam,
    config: &SgdConfig,
    streams: &mut [ChaCha8Rng],
    causal_action_streams: &mut [ChaCha8Rng],
    horizon: usize,
    plan: &TensorPlan,
    device: &Device,
) -> CandleResult<usize> {
    // Validation guarantees this, so a mismatch is a bug rather than something
    // to quietly clamp.
    debug_assert!(config.refresh_particles <= config.particles);
    assert_eq!(
        link_lane.is_some(),
        config.causal_links_enabled(),
        "refresh receives exactly the causal-link lane enabled by the configuration"
    );
    let refreshed = config.refresh_particles;

    let mut action = action_logits.as_tensor().flatten_all()?.to_vec1::<f64>()?;
    let mut causal_action = causal_action_logits
        .as_tensor()
        .flatten_all()?
        .to_vec1::<f64>()?;
    let mut state = state_logits.as_tensor().flatten_all()?.to_vec1::<f64>()?;
    let mut link = match link_lane.as_ref() {
        Some(lane) => Some(lane.logits.as_tensor().flatten_all()?.to_vec1::<f64>()?),
        None => None,
    };
    let action_stride = horizon * plan.num_actions;
    let state_stride = horizon * plan.num_facts;
    let link_stride = (horizon + 1) * plan.num_facts * (horizon + 1);

    for particle in 0..refreshed {
        let fresh_action = initial_action_vec(
            &mut streams[particle],
            horizon,
            plan.num_actions,
            config.initial_noop_logit_gap,
            config.slot_slack_window,
            config.slot_slack_logit_gap,
        );
        let fresh_state = normal_vec(&mut streams[particle], state_stride, 0.7);
        let fresh_causal_action = initial_action_vec(
            &mut causal_action_streams[particle],
            horizon,
            plan.num_actions,
            config.initial_noop_logit_gap,
            config.slot_slack_window,
            config.slot_slack_logit_gap,
        );
        action[particle * action_stride..(particle + 1) * action_stride]
            .copy_from_slice(&fresh_action);
        causal_action[particle * action_stride..(particle + 1) * action_stride]
            .copy_from_slice(&fresh_causal_action);
        state[particle * state_stride..(particle + 1) * state_stride].copy_from_slice(&fresh_state);
        if let Some(lane) = link_lane.as_mut() {
            let fresh_link = initial_link_vec(
                &mut lane.streams[particle],
                horizon,
                plan.num_facts,
                config.causal_link_initial_bias,
            );
            assert_eq!(
                fresh_link.len(),
                link_stride,
                "fresh causal ticket size matches the checked link shape"
            );
            link.as_mut()
                .expect("causal-link values exist exactly when their lane exists")
                [particle * link_stride..(particle + 1) * link_stride]
                .copy_from_slice(&fresh_link);
        }
    }

    action_logits.set(&Tensor::from_vec(
        action,
        (config.particles, horizon, plan.num_actions),
        device,
    )?)?;
    causal_action_logits.set(&Tensor::from_vec(
        causal_action,
        (config.particles, horizon, plan.num_actions),
        device,
    )?)?;
    state_logits.set(&Tensor::from_vec(
        state,
        (config.particles, horizon, plan.num_facts),
        device,
    )?)?;
    if let Some(lane) = link_lane.as_mut() {
        let link = link
            .take()
            .expect("causal-link values exist exactly when their lane exists");
        lane.logits.set(&Tensor::from_vec(
            link,
            (config.particles, horizon + 1, plan.num_facts, horizon + 1),
            device,
        )?)?;
    } else {
        assert!(
            link.is_none(),
            "causal-link values cannot exist without their lane"
        );
    }

    // Clear Adam memory for the refreshed particles: a stale second moment
    // would damp exactly the exploration the refresh is meant to buy.
    let keep_action = particle_mask(
        refreshed,
        config.particles,
        (config.particles, horizon, plan.num_actions),
        device,
    )?;
    let keep_state = particle_mask(
        refreshed,
        config.particles,
        (config.particles, horizon, plan.num_facts),
        device,
    )?;
    reset_direct_moments_preserving_schedule(
        optimizer,
        keep_action,
        keep_state,
        config,
        horizon,
        device,
    )?;
    let keep_causal_action = particle_mask(
        refreshed,
        config.particles,
        (config.particles, horizon, plan.num_actions),
        device,
    )?;
    causal_action_optimizer.reset_moments_where(&[keep_causal_action])?;
    if let Some(lane) = link_lane {
        let keep_link = particle_mask4(
            refreshed,
            config.particles,
            (config.particles, horizon + 1, plan.num_facts, horizon + 1),
            device,
        )?;
        lane.optimizer.reset_moments_where(&[keep_link])?;
    }
    Ok(refreshed)
}

/// Reset action/state Adam entries while leaving the optional temporal
/// schedule untouched. The primary optimizer owns exactly two parameters in
/// direct mode and exactly three in temporal mode.
fn reset_direct_moments_preserving_schedule(
    optimizer: &mut Adam,
    keep_action: Tensor,
    keep_state: Tensor,
    config: &SgdConfig,
    horizon: usize,
    device: &Device,
) -> CandleResult<()> {
    let mut masks = vec![keep_action, keep_state];
    if config.temporal_tokens {
        masks.push(Tensor::ones(
            (config.particles, horizon, horizon),
            DTYPE,
            device,
        )?);
    }
    optimizer.reset_moments_where(&masks)
}

/// Reinitialize one previously reserved no-op token when it receives a fact
/// role, and reset only that token's Adam history.
///
/// Every action coordinate is sampled from the same distribution: the host
/// exposes plastic capacity but does not select or privilege an operator.
#[allow(clippy::too_many_arguments)]
fn reset_temporal_token_action(
    particle: usize,
    token: usize,
    action_logits: &Var,
    optimizer: &mut Adam,
    config: &SgdConfig,
    stream: &mut ChaCha8Rng,
    horizon: usize,
    plan: &TensorPlan,
    device: &Device,
) -> CandleResult<()> {
    assert!(
        particle < config.particles,
        "activation particle is in range"
    );
    assert!(token < horizon, "activation token is in range");
    let actions = plan.num_actions;
    let mut values = action_logits.flatten_all()?.to_vec1::<f64>()?;
    let begin = (particle * horizon + token) * actions;
    let end = begin + actions;
    values[begin..end].copy_from_slice(&normal_vec(stream, actions, config.remelt_noise));
    action_logits.set(&Tensor::from_vec(
        values,
        (config.particles, horizon, actions),
        device,
    )?)?;

    let mut keep_action = vec![1.0f64; config.particles * horizon * actions];
    keep_action[begin..end].fill(0.0);
    optimizer.reset_moments_where(&[
        Tensor::from_vec(keep_action, (config.particles, horizon, actions), device)?,
        Tensor::ones((config.particles, horizon, plan.num_facts), DTYPE, device)?,
        Tensor::ones((config.particles, horizon, horizon), DTYPE, device)?,
    ])
}

/// A `[M, H, K]` mask that is 0 for the first `refreshed` particles and 1 after.
fn particle_mask(
    refreshed: usize,
    particles: usize,
    shape: (usize, usize, usize),
    device: &Device,
) -> CandleResult<Tensor> {
    let (_, horizon, inner) = shape;
    let mut values = vec![1f64; particles * horizon * inner];
    for slot in values.iter_mut().take(refreshed * horizon * inner) {
        *slot = 0.0;
    }
    Tensor::from_vec(values, shape, device)?.to_dtype(DType::F64)
}

/// A `[M, C, F, S]` mask that is zero for refreshed particles.
fn particle_mask4(
    refreshed: usize,
    particles: usize,
    shape: (usize, usize, usize, usize),
    device: &Device,
) -> CandleResult<Tensor> {
    let (_, consumers, facts, sources) = shape;
    let particle_stride = consumers * facts * sources;
    let mut values = vec![1f64; particles * particle_stride];
    values[..refreshed * particle_stride].fill(0.0);
    Tensor::from_vec(values, shape, device)?.to_dtype(DType::F64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_moment_reset_accounts_for_temporal_schedule_parameter() {
        let device = Device::Cpu;
        let particles = 2;
        let horizon = 3;
        let actions = Var::zeros((particles, horizon, 4), DTYPE, &device).expect("actions");
        let states = Var::zeros((particles, horizon, 5), DTYPE, &device).expect("states");
        let schedule = Var::zeros((particles, horizon, horizon), DTYPE, &device).expect("schedule");
        let mut optimizer = Adam::new(
            vec![actions, states, schedule],
            AdamParams {
                particles: Some(particles),
                ..AdamParams::default()
            },
        )
        .expect("temporal optimizer");
        let mut config = SgdConfig::default();
        config.particles = particles;
        config.temporal_tokens = true;

        reset_direct_moments_preserving_schedule(
            &mut optimizer,
            Tensor::ones((particles, horizon, 4), DTYPE, &device).expect("action mask"),
            Tensor::ones((particles, horizon, 5), DTYPE, &device).expect("state mask"),
            &config,
            horizon,
            &device,
        )
        .expect("all three optimizer parameters receive a mask");
    }

    #[test]
    fn temporal_projection_is_bijective_and_globally_optimal() {
        // Row-wise argmax selects execution row 0 for both token 0 and token
        // 1. A global projection must instead use every token and row once.
        let weights = vec![
            vec![0.90, 0.80, 0.00],
            vec![0.85, 0.10, 0.05],
            vec![0.01, 0.20, 0.95],
        ];
        let execution_to_token = maximum_weight_bijection(&weights);
        assert_eq!(execution_to_token, vec![1, 0, 2]);
        let total = execution_to_token
            .iter()
            .enumerate()
            .map(|(row, &token)| weights[token][row])
            .sum::<f64>();
        assert!((total - 2.60).abs() < 1e-12);
    }

    #[test]
    fn temporal_projection_decodes_token_identity_after_reordering() {
        let device = Device::Cpu;
        let token_action = Var::from_vec(
            vec![
                0.1, 0.8, 0.1, // token 0 -> action 1
                0.7, 0.2, 0.1, // token 1 -> action 0
                0.1, 0.1, 0.8, // token 2 -> action 2
            ],
            (1, 3, 3),
            &device,
        )
        .expect("token action tensor");
        let assignment = Tensor::from_vec(
            vec![
                0.1, 0.8, 0.1, // token 0 -> row 1
                0.8, 0.1, 0.1, // token 1 -> row 0
                0.1, 0.1, 0.8, // token 2 -> row 2
            ],
            (1, 3, 3),
            &device,
        )
        .expect("assignment tensor");
        assert_eq!(
            decode_temporal_particle(&token_action, &assignment, 0).expect("decode"),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn monotone_interleaving_preserves_both_streams_and_is_bijective() {
        let device = Device::Cpu;
        let horizon = 5;
        let token_action = Tensor::eye(horizon, DTYPE, &device)
            .expect("one action identity per token")
            .unsqueeze(0)
            .expect("particle axis");
        // The signs along the realized lattice path choose
        // scaffold, repair, scaffold, repair, scaffold.
        let mut gates = vec![-1.0f64; horizon * horizon];
        gates[1 * horizon] = 1.0;
        gates[2 * horizon + 1] = 1.0;
        let schedule =
            Tensor::from_vec(gates, (1, horizon, horizon), &device).expect("lattice gates");
        let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
        let temporal = monotone_interleaving_schedule(
            token_action,
            &schedule,
            &temperature,
            &[vec![0, 2, 4]],
            &[vec![1, 3]],
            &[Vec::new()],
        )
        .expect("monotone schedule");
        let execution = temporal_execution_to_token(&temporal.assignment, 0).expect("decode");
        assert_eq!(execution, vec![0, 1, 2, 3, 4]);
        assert_eq!(
            execution
                .iter()
                .copied()
                .filter(|token| [0, 2, 4].contains(token))
                .collect::<Vec<_>>(),
            vec![0, 2, 4]
        );
        assert_eq!(
            execution
                .iter()
                .copied()
                .filter(|token| [1, 3].contains(token))
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
        let assignment = temporal.assignment.get(0).expect("particle assignment");
        assert_eq!(
            assignment
                .sum(0)
                .expect("column sums")
                .to_vec1::<f64>()
                .unwrap(),
            vec![1.0; horizon]
        );
        assert_eq!(
            assignment
                .sum(1)
                .expect("row sums")
                .to_vec1::<f64>()
                .unwrap(),
            vec![1.0; horizon]
        );
        let soft_assignment = temporal
            .soft_assignment
            .get(0)
            .expect("particle soft assignment");
        for sum in soft_assignment
            .sum(0)
            .expect("soft column sums")
            .to_vec1::<f64>()
            .expect("soft column vector")
            .into_iter()
            .chain(
                soft_assignment
                    .sum(1)
                    .expect("soft row sums")
                    .to_vec1::<f64>()
                    .expect("soft row vector"),
            )
        {
            assert!((sum - 1.0).abs() < 1e-12, "path marginals are bistochastic");
        }
    }

    #[test]
    fn particles_cover_all_small_goal_orders_without_duplicates() {
        let goals = [2usize, 5, 7];
        let orders = (0..6)
            .map(|particle| permute_goal_order(&goals, particle))
            .collect::<BTreeSet<_>>();
        assert_eq!(orders.len(), 6, "three goals have six represented orders");
        assert!(orders.iter().all(|order| {
            let mut sorted = order.clone();
            sorted.sort_unstable();
            sorted == goals
        }));
        assert_eq!(
            permute_goal_order(&goals, 1),
            vec![7, 5, 2],
            "the second particle maximizes order diversity"
        );
    }

    #[test]
    fn prerequisite_capacity_is_taken_only_before_its_consumer() {
        let repair_order = [8usize, 9, 10, 11, 12, 13];
        let mut unused = vec![8usize, 9, 10, 11];
        assert_eq!(
            take_latest_preceding_token(&mut unused, &repair_order, 12),
            Some(11)
        );
        assert_eq!(
            take_latest_preceding_token(&mut unused, &repair_order, 10),
            Some(9)
        );
        assert_eq!(unused, vec![8, 10]);
        assert_eq!(
            take_latest_preceding_token(&mut unused, &repair_order, 8),
            None,
            "the first repair token has no structurally valid predecessor"
        );
    }

    #[test]
    fn one_fact_role_can_feed_multiple_later_consumers() {
        let repair_order = [3usize, 7, 9, 12, 15];
        let mut obligations = vec![None; 16];
        obligations[3] = Some(4);
        obligations[7] = Some(4);
        obligations[9] = Some(6);
        assert_eq!(
            latest_preceding_obligation(&obligations, &repair_order, 15, 4),
            Some(7),
            "the latest compatible shared producer minimizes separation"
        );
        assert_eq!(
            latest_preceding_obligation(&obligations, &repair_order, 7, 4),
            Some(3)
        );
        assert_eq!(
            latest_preceding_obligation(&obligations, &repair_order, 3, 4),
            None
        );
    }

    #[test]
    fn scaffold_repair_reuses_then_allocates_the_latest_feasible_role() {
        let scaffold_order = [0usize, 1, 2, 3, 5, 6, 8, 10, 11];
        let repair_order = [4usize, 7, 9, 12];
        let mut obligations = vec![None; 13];
        obligations[7] = Some(3);
        obligations[12] = Some(3);
        assert_eq!(
            scaffold_repair_candidate(
                &obligations,
                &[4, 9],
                &scaffold_order,
                &repair_order,
                &[],
                8,
                3,
            ),
            Some((12, false))
        );
        obligations[7] = None;
        obligations[12] = None;
        assert_eq!(
            scaffold_repair_candidate(
                &obligations,
                &[12, 9, 4],
                &scaffold_order,
                &repair_order,
                &[],
                8,
                3,
            ),
            Some((12, true))
        );
    }

    #[test]
    fn repeated_fact_is_a_cycle_only_on_the_same_downstream_branch() {
        let mut obligations = vec![None; 8];
        obligations[2] = Some(11);
        obligations[5] = Some(7);
        obligations[6] = Some(11);
        let edges = [(2usize, 4usize), (4, 6)];
        assert!(repeated_fact_cycle(&obligations, &edges, 2, 11));
        assert!(!repeated_fact_cycle(&obligations, &edges, 2, 7));
        assert!(!repeated_fact_cycle(&obligations, &edges, 5, 11));
    }

    #[test]
    fn cycle_rejection_removes_exactly_one_achiever_for_one_interval() {
        let achievers = [1usize, 4, 9];
        assert_eq!(achievers_except_rejected(&achievers, Some(4)), vec![1, 9]);
        assert_eq!(
            achievers_except_rejected(&achievers, None),
            achievers,
            "clearing the interval-local rejection restores full support"
        );
    }

    #[test]
    fn temporal_precedence_rejects_a_cross_stream_cycle_without_mutation() {
        let scaffold = [0usize, 2];
        let repair = [1usize, 3];
        let mut precedence = vec![(3usize, 0usize)];
        assert!(temporal_interleaving_exists(
            &scaffold,
            &repair,
            &precedence
        ));
        assert!(!try_add_temporal_precedence(
            &mut precedence,
            &scaffold,
            &repair,
            (2, 1),
        ));
        assert_eq!(precedence, vec![(3, 0)]);
        assert!(try_add_temporal_precedence(
            &mut precedence,
            &scaffold,
            &repair,
            (0, 2),
        ));
        assert_eq!(precedence, vec![(3, 0), (0, 2)]);
    }

    #[test]
    fn monotone_interleaving_has_a_direct_gap_gradient() {
        let device = Device::Cpu;
        let horizon = 3;
        let token_action = Tensor::eye(horizon, DTYPE, &device)
            .expect("one action identity per token")
            .unsqueeze(0)
            .expect("particle axis");
        let schedule = Var::from_vec(
            vec![-1.0f64; horizon * horizon],
            (1, horizon, horizon),
            &device,
        )
        .expect("lattice gates");
        let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
        let temporal = monotone_interleaving_schedule(
            token_action,
            schedule.as_tensor(),
            &temperature,
            &[vec![0, 2]],
            &[vec![1]],
            &[Vec::new()],
        )
        .expect("monotone schedule");
        // Hard execution currently leaves repair token 1 until the end. The
        // straight-through derivative must still be able to move it into row 0.
        let loss = temporal
            .assignment
            .get(0)
            .and_then(|value| value.get(1))
            .and_then(|value| value.get(0))
            .and_then(|value| value.neg())
            .expect("early repair objective");
        let gradient = loss
            .backward()
            .expect("interleaving backward")
            .get(&schedule)
            .expect("schedule gradient")
            .to_vec3::<f64>()
            .expect("rank-three gradient");
        assert!(
            gradient[0][0][0] < 0.0,
            "gradient descent raises the initial repair gate"
        );
    }

    #[test]
    fn cross_stream_precedence_masks_every_soft_and_hard_path() {
        let device = Device::Cpu;
        let horizon = 3;
        let token_action = Tensor::eye(horizon, DTYPE, &device)
            .expect("one action identity per token")
            .unsqueeze(0)
            .expect("particle axis");
        // Without the precedence edge these scores put scaffold token 0 first.
        let mut scores = vec![0.0f64; horizon * horizon];
        scores[0] = -20.0;
        scores[horizon] = 20.0;
        let schedule =
            Tensor::from_vec(scores, (1, horizon, horizon), &device).expect("lattice edge scores");
        let temporal = monotone_interleaving_schedule(
            token_action,
            &schedule,
            &Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature"),
            &[vec![0, 2]],
            &[vec![1]],
            &[vec![(1, 0)]],
        )
        .expect("precedence-constrained schedule");

        assert_eq!(
            temporal_execution_to_token(&temporal.assignment, 0).expect("hard execution"),
            vec![1, 0, 2]
        );
        let soft = temporal
            .soft_assignment
            .get(0)
            .expect("particle")
            .to_vec2::<f64>()
            .expect("token by row marginals");
        assert_eq!(
            soft[0][0], 0.0,
            "no differentiable path may consume token 0 before producer token 1"
        );
        assert_eq!(soft[1][0], 1.0);
    }

    #[test]
    fn batched_interleavings_equal_independent_values_and_gradients() {
        let device = Device::Cpu;
        let horizon = 3;
        let one_particle_action =
            Tensor::eye(horizon, DTYPE, &device).expect("one action identity per token");
        let token_action =
            Tensor::stack(&[&one_particle_action, &one_particle_action], 0).expect("two particles");
        let values = vec![
            -0.8f64, 0.2, 0.0, 0.4, -0.3, 0.0, 0.0, 0.0, 0.0, 0.7, -0.1, 0.0, -0.5, 0.6, 0.0, 0.0,
            0.0, 0.0,
        ];
        let schedule = Var::from_vec(values.clone(), (2, horizon, horizon), &device)
            .expect("batched schedule logits");
        let temperature = Tensor::ones((2, 1, 1), DTYPE, &device).expect("temperature");
        let scaffolds = vec![vec![0usize, 2], vec![2usize, 0]];
        let repairs = vec![vec![1usize], vec![1usize]];
        let batched = monotone_interleaving_schedule(
            token_action,
            schedule.as_tensor(),
            &temperature,
            &scaffolds,
            &repairs,
            &[Vec::new(), Vec::new()],
        )
        .expect("batched interleavings");
        let batched_loss = (batched
            .soft_assignment
            .get(0)
            .unwrap()
            .get(1)
            .unwrap()
            .get(0)
            .unwrap()
            + (batched
                .soft_assignment
                .get(1)
                .unwrap()
                .get(1)
                .unwrap()
                .get(2)
                .unwrap()
                * 2.0)
                .unwrap())
        .unwrap();
        let batched_gradient = batched_loss
            .backward()
            .expect("batched backward")
            .get(&schedule)
            .expect("batched schedule gradient")
            .to_vec3::<f64>()
            .expect("rank-three gradient");

        for particle in 0..2 {
            let begin = particle * horizon * horizon;
            let single_schedule = Var::from_vec(
                values[begin..begin + horizon * horizon].to_vec(),
                (1, horizon, horizon),
                &device,
            )
            .expect("single schedule logits");
            let single = monotone_interleaving_schedule(
                one_particle_action.unsqueeze(0).unwrap(),
                single_schedule.as_tensor(),
                &Tensor::ones((1, 1, 1), DTYPE, &device).unwrap(),
                &[scaffolds[particle].clone()],
                &[repairs[particle].clone()],
                &[Vec::new()],
            )
            .expect("independent interleaving");
            let batched_values = batched
                .soft_assignment
                .get(particle)
                .unwrap()
                .to_vec2::<f64>()
                .unwrap();
            let single_values = single
                .soft_assignment
                .get(0)
                .unwrap()
                .to_vec2::<f64>()
                .unwrap();
            for (batched_row, single_row) in batched_values.iter().zip(single_values) {
                for (&batched_value, single_value) in batched_row.iter().zip(single_row) {
                    assert!((batched_value - single_value).abs() < 1e-12);
                }
            }
            let row = if particle == 0 { 0 } else { 2 };
            let scale = if particle == 0 { 1.0 } else { 2.0 };
            let single_loss = (single
                .soft_assignment
                .get(0)
                .unwrap()
                .get(1)
                .unwrap()
                .get(row)
                .unwrap()
                * scale)
                .unwrap();
            let single_gradient = single_loss
                .backward()
                .expect("single backward")
                .get(&single_schedule)
                .expect("single schedule gradient")
                .to_vec3::<f64>()
                .unwrap();
            for (batched_row, single_row) in
                batched_gradient[particle].iter().zip(&single_gradient[0])
            {
                for (&batched_value, &single_value) in batched_row.iter().zip(single_row) {
                    assert!((batched_value - single_value).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn temporal_unlock_freezes_scaffold_identity_but_not_repair_identity() {
        let device = Device::Cpu;
        let action =
            Var::from_vec(vec![0.8f64, 0.2, 0.4, 0.6], (1, 2, 2), &device).expect("token actions");
        let gate = Tensor::from_vec(vec![0.0f64, 1.0], (1, 2, 1), &device).expect("repair gate");
        let gated =
            repair_only_action_gradients(action.as_tensor(), &gate).expect("gradient trust region");
        assert_eq!(
            gated.to_vec3::<f64>().expect("forward values"),
            action.as_tensor().to_vec3::<f64>().expect("source values")
        );
        let gradient = gated
            .sum_all()
            .and_then(|loss| loss.backward())
            .expect("backward")
            .get(&action)
            .expect("repair action gradient")
            .to_vec3::<f64>()
            .expect("rank-three gradient");
        assert_eq!(gradient[0][0], vec![0.0, 0.0]);
        assert_eq!(gradient[0][1], vec![1.0, 1.0]);
    }

    #[test]
    fn inactive_temporal_memory_is_an_exact_gradient_frozen_noop() {
        let device = Device::Cpu;
        let action =
            Var::from_vec(vec![0.8f64, 0.2, 0.4, 0.6], (1, 2, 2), &device).expect("token actions");
        let inactive =
            Tensor::from_vec(vec![1.0f64, 0.0], (1, 2, 1), &device).expect("inactive mask");
        let forced = force_inactive_temporal_noops(action.as_tensor(), &inactive, 1)
            .expect("exact inactive no-op");
        assert_eq!(
            forced.to_vec3::<f64>().expect("forward values"),
            vec![vec![vec![0.0, 1.0], vec![0.4, 0.6]]]
        );
        let gradient = forced
            .sum_all()
            .and_then(|loss| loss.backward())
            .expect("backward")
            .get(&action)
            .expect("active action gradient")
            .to_vec3::<f64>()
            .expect("rank-three gradient");
        assert_eq!(gradient[0][0], vec![0.0, 0.0]);
        assert_eq!(gradient[0][1], vec![1.0, 1.0]);
    }

    #[test]
    fn temporal_precedence_has_exact_zero_and_moves_both_endpoint_tokens() {
        let device = Device::Cpu;
        let exact = Tensor::eye(3, DTYPE, &device)
            .expect("identity")
            .unsqueeze(0)
            .expect("particle axis");
        assert_eq!(
            temporal_precedence_loss(&exact, &[vec![(0, 2)]], &[vec![]], &[vec![]])
                .expect("precedence")
                .to_scalar::<f64>()
                .expect("scalar"),
            0.0
        );

        let assignment = Var::from_vec(
            vec![
                0.1, 0.2, 0.7, // producer is too late
                0.2, 0.6, 0.2, 0.7, 0.2, 0.1, // consumer is too early
            ],
            (1, 3, 3),
            &device,
        )
        .expect("soft assignment");
        let loss = temporal_precedence_loss(
            assignment.as_tensor(),
            &[vec![(0, 2)]],
            &[vec![]],
            &[vec![]],
        )
        .expect("precedence");
        assert!(loss.to_scalar::<f64>().expect("scalar") > 1.0);
        let gradient = loss
            .backward()
            .expect("precedence backward")
            .get(&assignment)
            .expect("assignment gradient")
            .to_vec3::<f64>()
            .expect("rank-three gradient");
        assert!(gradient[0][0][2] > gradient[0][0][0]);
        assert!(gradient[0][2][2] < gradient[0][2][0]);
    }

    #[test]
    fn repair_edge_support_certifies_and_learns_one_common_gap() {
        let device = Device::Cpu;
        let token_action = Tensor::ones((1, 4, 1), DTYPE, &device).expect("token actions");
        let achievers = Tensor::from_vec(vec![0.0f64, 1.0, 1.0, 0.0], (1, 4, 1), &device)
            .expect("repair achievers");
        let applicable = vec![vec![
            None,
            Some(vec![1.0f64; 5]),
            Some(vec![1.0f64; 5]),
            None,
        ]];
        // scaffold 0, then repair 1 -> repair 2 in gap 1, then scaffold 3.
        let compact = Tensor::from_vec(
            vec![
                1.0f64, 0.0, 0.0, 0.0, // scaffold token 0
                0.0, 1.0, 0.0, 0.0, // producer token 1
                0.0, 0.0, 1.0, 0.0, // consumer token 2
                0.0, 0.0, 0.0, 1.0, // scaffold token 3
            ],
            (1, 4, 4),
            &device,
        )
        .expect("compact assignment");
        let edges = [vec![(1usize, 2usize)]];
        let scaffold = [vec![0usize, 3usize]];
        let repair = [vec![1usize, 2usize]];
        assert_eq!(
            temporal_repair_edge_gap_support_loss(
                &token_action,
                &compact,
                &achievers,
                &applicable,
                &edges,
                &scaffold,
                &repair,
            )
            .expect("edge support")
            .to_scalar::<f64>()
            .expect("scalar"),
            0.0
        );

        let separated = Var::from_vec(
            vec![
                0.0f64, 1.0, 0.0, 0.0, // scaffold token 0
                0.8, 0.0, 0.2, 0.0, // producer mostly in gap 0
                0.0, 0.2, 0.0, 0.8, // consumer mostly in gap 2
                0.0, 0.0, 1.0, 0.0, // scaffold token 3
            ],
            (1, 4, 4),
            &device,
        )
        .expect("separated assignment");
        let loss = temporal_repair_edge_gap_support_loss(
            &token_action,
            separated.as_tensor(),
            &achievers,
            &applicable,
            &edges,
            &scaffold,
            &repair,
        )
        .expect("edge support");
        assert!(loss.to_scalar::<f64>().expect("scalar") > 1.0);
        let gradient = loss
            .backward()
            .expect("compactness backward")
            .get(&separated)
            .expect("assignment gradient")
            .to_vec3::<f64>()
            .expect("rank-three gradient");
        assert!(
            gradient[0][1][2] < gradient[0][1][0],
            "gradient descent moves the producer toward consumer gap 2"
        );
    }

    #[test]
    fn repair_edge_uses_soft_path_marginals_across_different_hard_gaps() {
        let device = Device::Cpu;
        let horizon = 4;
        let token_action =
            Tensor::ones((1, horizon, 1), DTYPE, &device).expect("one applicable action identity");
        let schedule = Var::from_vec(
            vec![
                4.0f64, -4.0, -4.0, -4.0, // producer before both scaffold tokens
                -4.0, -4.0, -4.0, -4.0, -4.0, 4.0, -4.0,
                -4.0, // consumer after both scaffold tokens
                -4.0, -4.0, -4.0, -4.0,
            ],
            (1, horizon, horizon),
            &device,
        )
        .expect("lattice edge scores");
        let temperature = Tensor::ones((1, 1, 1), DTYPE, &device).expect("temperature");
        let temporal = monotone_interleaving_schedule(
            token_action.clone(),
            schedule.as_tensor(),
            &temperature,
            &[vec![0, 3]],
            &[vec![1, 2]],
            &[Vec::new()],
        )
        .expect("monotone schedule");
        let hard_execution =
            temporal_execution_to_token(&temporal.assignment, 0).expect("hard execution");
        assert_eq!(hard_execution, vec![1, 0, 3, 2]);

        let achievers = Tensor::from_vec(vec![0.0f64, 1.0, 1.0, 0.0], (1, 4, 1), &device)
            .expect("repair achievers");
        let applicable = vec![vec![
            None,
            Some(vec![1.0f64; 5]),
            Some(vec![1.0f64; 5]),
            None,
        ]];
        let loss = temporal_repair_edge_gap_support_loss(
            &token_action,
            &temporal.soft_assignment,
            &achievers,
            &applicable,
            &[vec![(1usize, 2usize)]],
            &[vec![0usize, 3usize]],
            &[vec![1usize, 2usize]],
        )
        .expect("soft common-gap support");
        let scalar = loss.to_scalar::<f64>().expect("scalar loss");
        assert!(scalar.is_finite() && scalar > 0.0);
        let gradient = loss
            .backward()
            .expect("soft conjunction backward")
            .get(&schedule)
            .expect("schedule gradient")
            .to_vec3::<f64>()
            .expect("rank-three gradient");
        assert!(
            gradient[0]
                .iter()
                .flatten()
                .any(|value| value.abs() > 1e-12),
            "soft overlap must train placements before hard gaps coincide"
        );
    }

    #[test]
    fn conditional_gap_applicability_waives_a_supplied_precondition_only_downstream() {
        let obligations = [vec![Some(0usize), Some(1usize)]];
        let precedence = [vec![(0usize, 1usize)]];
        let scaffold = [vec![]];
        // Three padded gap rows for H=2, with neither fact initially true.
        let facts = vec![0.0f64; 1 * 3 * 2];
        let action_preconditions = [vec![], vec![0usize]];
        let masks = conditional_gap_applicability_masks(
            &obligations,
            &precedence,
            &scaffold,
            &facts,
            &action_preconditions,
            1,
            2,
            2,
            2,
            true,
        );
        let producer = masks[0][0].as_ref().expect("producer conditional mask");
        let consumer = masks[0][1].as_ref().expect("consumer conditional mask");
        assert_eq!(
            producer[1],
            (-1.0f64).exp(),
            "producer cannot assume its own output but retains graded support"
        );
        assert_eq!(
            consumer[1], 1.0,
            "consumer may rely on its direct producer's obligated fact"
        );
    }

    #[test]
    fn temporal_applicability_moves_an_obligated_token_to_a_supported_row() {
        let device = Device::Cpu;
        let token_action =
            Var::from_vec(vec![0.9f64, 0.1, 0.2, 0.8], (1, 2, 2), &device).expect("token actions");
        let assignment =
            Var::from_vec(vec![0.8f64, 0.2, 0.2, 0.8], (1, 2, 2), &device).expect("assignment");
        let applicable = Tensor::from_vec(vec![0.0f64, 1.0, 1.0, 0.0], (1, 2, 2), &device)
            .expect("row applicability");
        let active = Tensor::from_vec(vec![1.0f64, 0.0], (1, 2), &device).expect("active");
        let loss = temporal_obligation_applicability_loss(
            token_action.as_tensor(),
            assignment.as_tensor(),
            &applicable,
            &active,
            &Tensor::ones((1, 2, 1), DTYPE, &device).expect("focus"),
        )
        .expect("placement loss");
        let gradients = loss.backward().expect("placement backward");
        assert!(
            gradients.get(&token_action).is_none(),
            "placement must not erase obligated action identity"
        );
        let gradient = gradients
            .get(&assignment)
            .expect("assignment gradient")
            .to_vec3::<f64>()
            .expect("rank-three gradient");
        assert!(
            gradient[0][0][1] < gradient[0][0][0],
            "gradient descent moves obligated token 0 toward supported row 1"
        );
        assert_eq!(gradient[0][1], vec![0.0, 0.0]);
    }

    #[test]
    fn scaffold_gap_support_survives_an_early_repair_failure() {
        let device = Device::Cpu;
        let token_action = Var::from_vec(
            vec![
                0.5f64, 0.5, // scaffold token 0
                1.0, 0.0, // repair token 1 chooses action 0
                0.5, 0.5, // scaffold token 2
            ],
            (1, 3, 2),
            &device,
        )
        .expect("token actions");
        let assignment = Var::from_vec(
            vec![
                0.1f64, 0.8, 0.1, // scaffold token 0
                0.8, 0.1, 0.1, // repair is currently too early
                0.1, 0.1, 0.8, // scaffold token 2
            ],
            (1, 3, 3),
            &device,
        )
        .expect("assignment");
        // Action 0 is supported only after both scaffold tokens. A row mask
        // from the current early failure would never expose this state.
        let applicable_by_gap = vec![vec![
            None,
            Some(vec![
                0.0f64, 1.0, // gap 0
                0.0, 1.0, // gap 1
                1.0, 1.0, // gap 2
                0.0, 0.0, // unused padded gap
            ]),
            None,
        ]];
        let active =
            Tensor::from_vec(vec![0.0f64, 1.0, 0.0], (1, 3), &device).expect("obligation active");
        let achiever_by_token = Tensor::from_vec(
            vec![
                0.0f64, 0.0, // scaffold token 0 has no obligation
                1.0, 0.0, // only action 0 achieves repair token 1's fact
                0.0, 0.0, // scaffold token 2 has no obligation
            ],
            (1, 3, 2),
            &device,
        )
        .expect("token achievers");
        let loss = temporal_obligation_scaffold_gap_loss(
            token_action.as_tensor(),
            assignment.as_tensor(),
            &achiever_by_token,
            &applicable_by_gap,
            &active,
            &Tensor::ones((1, 3, 1), DTYPE, &device).expect("focus"),
            &[vec![0, 2]],
            &[vec![1]],
        )
        .expect("gap placement loss");
        let gradients = loss.backward().expect("gap placement backward");
        assert!(
            gradients
                .get(&token_action)
                .expect("joint action-gap gradient")
                .to_vec3::<f64>()
                .expect("rank-three action gradient")[0][1][0]
                < 0.0,
            "gradient descent increases the achiever supported in the target gap"
        );
        let gradient = gradients
            .get(&assignment)
            .expect("assignment gradient")
            .to_vec3::<f64>()
            .expect("rank-three gradient");
        assert!(
            gradient[0][1][2] < 0.0,
            "gradient descent moves repair token 1 to the supported final gap"
        );
        assert_eq!(gradient[0][1][0], 0.0);
    }

    #[test]
    fn scaffold_gap_support_breaks_achiever_symmetry_by_applicability() {
        let device = Device::Cpu;
        let token_action =
            Var::from_vec(vec![0.8f64, 0.2], (1, 1, 2), &device).expect("two achievers");
        let assignment = Tensor::ones((1, 1, 1), DTYPE, &device).expect("only execution row");
        let achievers =
            Tensor::ones((1, 1, 2), DTYPE, &device).expect("both actions achieve the fact");
        let applicable = vec![vec![Some(vec![
            0.0f64, 1.0, // only action 1 applies in the real gap
            0.0, 0.0, // padded gap
        ])]];
        let active = Tensor::ones((1, 1), DTYPE, &device).expect("active obligation");
        let loss = temporal_obligation_scaffold_gap_loss(
            token_action.as_tensor(),
            &assignment,
            &achievers,
            &applicable,
            &active,
            &Tensor::ones((1, 1, 1), DTYPE, &device).expect("focus"),
            &[vec![]],
            &[vec![0]],
        )
        .expect("joint support");
        let gradient = loss
            .backward()
            .expect("backward")
            .get(&token_action)
            .expect("action gradient")
            .to_vec3::<f64>()
            .expect("rank-three gradient");
        assert_eq!(gradient[0][0][0], 0.0);
        assert!(
            gradient[0][0][1] < 0.0,
            "gradient descent raises the achiever that is also applicable"
        );
    }

    #[test]
    fn every_verifier_phase_keeps_all_core_losses_alive() {
        for phase in [Phase::BuildApplicability, Phase::Goal, Phase::GoalRepair] {
            let weights = phase_loss_weights(phase);
            assert!(weights.precondition > 0.0);
            assert!(weights.transition > 0.0);
            assert!(weights.goal > 0.0);
            assert!(weights.causal > 0.0);
        }
    }

    #[test]
    fn phase_schedule_shifts_pressure_without_erasing_causality() {
        let build = phase_loss_weights(Phase::BuildApplicability);
        let goal = phase_loss_weights(Phase::Goal);
        let repair = phase_loss_weights(Phase::GoalRepair);

        assert!(build.precondition > goal.precondition);
        assert!(goal.goal > build.goal);
        assert!(goal.causal > build.causal);
        assert!(repair.transition > goal.transition);
        assert!(repair.goal > 1.0);
        assert!(repair.causal > 1.0);
    }

    #[test]
    fn goal_bridge_has_hysteresis_through_applicability_repair() {
        assert!(!goal_bridge_is_active(
            Phase::BuildApplicability,
            &[true, false]
        ));
        assert!(goal_bridge_is_active(Phase::Goal, &[true, false]));
        assert!(goal_bridge_is_active(Phase::GoalRepair, &[true, false]));
        assert!(!goal_bridge_is_active(Phase::GoalRepair, &[false, false]));
    }

    #[test]
    fn particle_schedule_is_applied_before_the_population_mean() {
        let device = Device::Cpu;
        let per_particle =
            Var::from_vec(vec![2.0f64, 100.0], (2, 1), &device).expect("particle losses");
        let schedule =
            Tensor::from_vec(vec![1.0f64, 0.0], (2, 1), &device).expect("particle schedule");
        let loss = scheduled_particle_mean(per_particle.as_tensor(), &schedule)
            .expect("scheduled population mean");
        assert_eq!(loss.to_scalar::<f64>().expect("scalar loss"), 1.0);

        let gradients = loss.backward().expect("scheduled loss backward");
        let gradient = gradients
            .get(&per_particle)
            .expect("particle-loss gradient")
            .to_vec2::<f64>()
            .expect("rank-2 gradient");
        assert_eq!(gradient, vec![vec![0.5], vec![0.0]]);
    }

    #[test]
    fn recurrent_route_never_returns_to_delete_relaxation_after_proof() {
        for copy in [CausalCopyMode::Off, CausalCopyMode::Shadow] {
            for stage in [
                CausalStage::Proof,
                CausalStage::Transfer,
                CausalStage::Takeover,
                CausalStage::Polish,
            ] {
                for update in [1, 2, 3, 4] {
                    assert_eq!(
                        recurrent_loss_route(copy, stage, update),
                        Some(RecurrentLossRoute {
                            source: RecurrentLossSource::Execution,
                            goal: RecurrentGoalMode::DeleteAwareTerminal,
                        })
                    );
                }
            }
        }
    }

    #[test]
    fn staged_route_proves_with_q_then_makes_p_authoritative_during_transfer() {
        assert_eq!(
            recurrent_loss_route(CausalCopyMode::Staged, CausalStage::Proof, 1),
            Some(RecurrentLossRoute {
                source: RecurrentLossSource::CausalCopy,
                goal: RecurrentGoalMode::DeleteAwareTerminal,
            })
        );
        for stage in [
            CausalStage::Transfer,
            CausalStage::Takeover,
            CausalStage::Polish,
        ] {
            assert_eq!(
                recurrent_loss_route(CausalCopyMode::Staged, stage, 1),
                Some(RecurrentLossRoute {
                    source: RecurrentLossSource::Execution,
                    goal: RecurrentGoalMode::DeleteAwareTerminal,
                })
            );
        }
    }

    #[test]
    fn zero_applicability_ranking_barrier_certifies_a_good_argmax() {
        let device = Device::Cpu;
        // The good action has only a small logit lead.  Most probability mass
        // is nevertheless on the four bad actions collectively, which is why
        // a total-bad-mass constraint is needlessly stronger than argmax
        // applicability.
        let logits = Tensor::from_vec(vec![0.1f64, 0.0, 0.0, 0.0, 0.0], (1, 1, 5), &device)
            .expect("action logits");
        let applicable = Tensor::from_vec(vec![1.0f64, 0.0, 0.0, 0.0, 0.0], (1, 1, 5), &device)
            .expect("applicability mask");
        let active = Tensor::ones((1, 1), DTYPE, &device).expect("active row");
        let focus = Tensor::ones((1, 1, 1), DTYPE, &device).expect("row focus");
        let barrier = applicability_ranking_barrier(&logits, &applicable, &active, &focus, 0.95)
            .expect("ranking barrier")
            .to_scalar::<f64>()
            .expect("scalar barrier");
        assert_eq!(barrier, 0.0);
        assert_eq!(
            logits.argmax(2).expect("argmax").to_vec2::<u32>().unwrap(),
            vec![vec![0]]
        );
    }

    #[test]
    fn applicability_ranking_barrier_pushes_a_bad_argmax_below_all_good_ones() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_vec(vec![0.0f64, 0.2, -0.3], (1, 1, 3), &device).expect("action logits");
        let applicable = Tensor::from_vec(vec![1.0f64, 0.0, 1.0], (1, 1, 3), &device)
            .expect("applicability mask");
        let active = Tensor::ones((1, 1), DTYPE, &device).expect("active row");
        let focus = Tensor::ones((1, 1, 1), DTYPE, &device).expect("row focus");
        let barrier = applicability_ranking_barrier(&logits, &applicable, &active, &focus, 0.95)
            .expect("ranking barrier")
            .to_scalar::<f64>()
            .expect("scalar barrier");
        assert!(barrier > 0.25, "bad argmax needs a strict correction");
    }

    #[test]
    fn applicability_mass_treats_all_good_actions_symmetrically() {
        let device = Device::Cpu;
        let logits = Var::zeros((1, 1, 4), DTYPE, &device).expect("action logits");
        let action = candle_nn::ops::softmax(logits.as_tensor(), 2).expect("softmax");
        let applicable = Tensor::from_vec(vec![1.0f64, 0.0, 1.0, 0.0], (1, 1, 4), &device)
            .expect("applicability mask");
        let active = Tensor::ones((1, 1), DTYPE, &device).expect("active row");
        let focus = Tensor::ones((1, 1, 1), DTYPE, &device).expect("row focus");
        let loss = applicability_mass_loss(&action, &applicable, &active, &focus)
            .expect("applicability mass");
        assert!((loss.to_scalar::<f64>().unwrap() - 2.0f64.ln()).abs() < 1e-12);
        let gradients = loss.backward().expect("mass backward");
        let gradient = gradients
            .get(&logits)
            .expect("logit gradient")
            .to_vec3::<f64>()
            .expect("rank-3 gradient");
        assert_eq!(gradient[0][0][0], gradient[0][0][2]);
        assert_eq!(gradient[0][0][1], gradient[0][0][3]);
        assert!(gradient[0][0][0] < gradient[0][0][1]);
    }

    #[test]
    fn obligation_ranks_the_achiever_applicability_intersection() {
        let applicable = [0.0, 1.0, 1.0, 0.0];
        assert_eq!(
            obligation_achiever_conjunction(&[0, 1, 3], Some(&applicable)),
            vec![1]
        );
    }

    #[test]
    fn obligation_retains_achievers_when_prerequisites_are_still_missing() {
        let applicable = [0.0, 0.0, 1.0, 0.0];
        assert_eq!(
            obligation_achiever_conjunction(&[0, 1, 3], Some(&applicable)),
            vec![0, 1, 3]
        );
        assert_eq!(
            obligation_achiever_conjunction(&[0, 1, 3], None),
            vec![0, 1, 3]
        );
    }

    #[test]
    fn anchor_trust_preserves_only_explicitly_active_rows() {
        let device = Device::Cpu;
        let log_action = Var::from_vec(
            vec![0.2f64.ln(), 0.8f64.ln(), 0.7f64.ln(), 0.3f64.ln()],
            (1, 2, 2),
            &device,
        )
        .expect("log probabilities");
        let target = Tensor::from_vec(vec![0.0f64, 1.0, 1.0, 0.0], (1, 2, 2), &device)
            .expect("anchor target");
        let active = Tensor::from_vec(vec![1.0f64, 0.0], (1, 2, 1), &device).expect("anchor mask");
        let loss =
            anchor_trust_loss(log_action.as_tensor(), &target, &active).expect("anchor loss");
        assert!((loss.to_scalar::<f64>().unwrap() + 0.8f64.ln()).abs() < 1e-12);
        let gradients = loss.backward().expect("anchor backward");
        let gradient = gradients
            .get(&log_action)
            .expect("log-probability gradient")
            .to_vec3::<f64>()
            .expect("rank-three gradient");
        assert_eq!(gradient[0][0], vec![0.0, -1.0]);
        assert_eq!(gradient[0][1], vec![0.0, 0.0]);
    }

    #[test]
    fn empty_particles_do_not_dilute_anchor_trust() {
        let device = Device::Cpu;
        let log_action = Var::from_vec(
            vec![
                0.2f64.ln(),
                0.8f64.ln(),
                0.7f64.ln(),
                0.3f64.ln(),
                0.4f64.ln(),
                0.6f64.ln(),
                0.9f64.ln(),
                0.1f64.ln(),
            ],
            (2, 2, 2),
            &device,
        )
        .expect("log probabilities");
        let target = Tensor::from_vec(
            vec![0.0f64, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            (2, 2, 2),
            &device,
        )
        .expect("anchor targets");
        let active = Tensor::from_vec(vec![1.0f64, 0.0, 0.0, 0.0], (2, 2, 1), &device)
            .expect("only the first particle has an anchor");
        let loss =
            anchor_trust_loss(log_action.as_tensor(), &target, &active).expect("anchor loss");
        assert!((loss.to_scalar::<f64>().unwrap() + 0.8f64.ln()).abs() < 1e-12);
        let gradients = loss.backward().expect("anchor backward");
        let gradient = gradients
            .get(&log_action)
            .expect("log-probability gradient")
            .to_vec3::<f64>()
            .expect("rank-three gradient");
        assert_eq!(gradient[0][0], vec![0.0, -1.0]);
        assert_eq!(gradient[1], vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    }

    #[test]
    fn insertion_warp_moves_the_free_tail_row_and_preserves_its_gradient() {
        let device = Device::Cpu;
        let logits = Var::from_tensor(
            &Tensor::from_vec(
                vec![0.0f64, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                (1, 4, 2),
                &device,
            )
            .expect("logits"),
        )
        .expect("variable logits");
        let warped = insertion_warp_logits(logits.as_tensor(), &[Some(1)])
            .expect("differentiable insertion warp");
        assert_eq!(
            warped.to_vec3::<f64>().expect("warped values"),
            vec![vec![
                vec![0.0, 0.5],
                vec![3.0, 3.5],
                vec![1.0, 1.5],
                vec![2.0, 2.5]
            ]]
        );
        let loss = warped
            .narrow(1, 1, 1)
            .expect("inserted row")
            .narrow(2, 0, 1)
            .expect("inserted coordinate")
            .sum_all()
            .expect("scalar loss");
        let gradient = loss
            .backward()
            .expect("warp backward")
            .get(&logits)
            .expect("logit gradient")
            .to_vec3::<f64>()
            .expect("gradient values");
        assert_eq!(gradient[0][3], vec![1.0, 0.0]);
        assert_eq!(gradient[0][1], vec![0.0, 0.0]);
    }
}
