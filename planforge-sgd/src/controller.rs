//! Verifier-driven loss scheduling.
//!
//! This module contains no planning semantics and makes no action choice.  It
//! consumes the result of replaying one already-decoded sequence and turns that
//! exact feedback into loss weights and, after a plateau, a remelt interval.
//! Keeping the controller independent of SAS facts makes both its boundary and
//! its state-machine invariants testable without the tensor backend.

use std::fmt;

/// Which constraint family currently needs the optimizer's attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// No complete decoded sequence has yet passed exact replay.
    BuildApplicability,
    /// Exact replay accepts the complete sequence, but task goals are missing.
    Goal,
    /// Goal pressure broke applicability; repair it without forgetting the
    /// accumulated per-goal pressure.
    GoalRepair,
}

impl Phase {
    const fn index(self) -> usize {
        match self {
            Self::BuildApplicability => 0,
            Self::Goal => 1,
            Self::GoalRepair => 2,
        }
    }
}

/// Multiplicative projected ascent for an active first-failure target.
///
/// A newly observed target starts at `initial`. Every repeated observation of
/// that same target applies `weight <- min(cap, growth * weight)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FailureWeightSchedule {
    pub initial: f64,
    pub growth: f64,
    pub cap: f64,
}

impl Default for FailureWeightSchedule {
    fn default() -> Self {
        Self {
            initial: 1.0,
            growth: 1.06,
            cap: 80.0,
        }
    }
}

impl FailureWeightSchedule {
    fn increased(self, weight: f64) -> f64 {
        (weight * self.growth).min(self.cap)
    }

    fn validate(self, field: &'static str) -> Result<(), ControllerError> {
        finite_positive(field, "initial", self.initial)?;
        finite_positive(field, "cap", self.cap)?;
        if !self.growth.is_finite() || self.growth <= 1.0 {
            return Err(ControllerError::InvalidConfig {
                field,
                problem: format!("growth must be finite and exceed 1, got {}", self.growth),
            });
        }
        if self.cap <= self.initial {
            return Err(ControllerError::InvalidConfig {
                field,
                problem: format!("cap ({}) must exceed initial ({})", self.cap, self.initial),
            });
        }
        Ok(())
    }
}

/// Additive projected ascent for goal-specific weights.
///
/// Missing goals ascend by `increment`. Reached goals retain their accumulated
/// pressure until the particle solves: an alternating replay can otherwise
/// achieve one goal, decay its weight, and promptly clobber it while pursuing
/// another.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GoalWeightSchedule {
    pub baseline: f64,
    pub increment: f64,
    pub cap: f64,
}

impl Default for GoalWeightSchedule {
    fn default() -> Self {
        Self {
            baseline: 1.0,
            increment: 1.0,
            cap: 80.0,
        }
    }
}

impl GoalWeightSchedule {
    fn increased(self, weight: f64) -> f64 {
        (weight + self.increment).min(self.cap)
    }

    fn validate(self) -> Result<(), ControllerError> {
        finite_non_negative("goal_weight", "baseline", self.baseline)?;
        finite_positive("goal_weight", "increment", self.increment)?;
        finite_positive("goal_weight", "cap", self.cap)?;
        if self.cap <= self.baseline {
            return Err(ControllerError::InvalidConfig {
                field: "goal_weight",
                problem: format!(
                    "cap ({}) must exceed baseline ({})",
                    self.cap, self.baseline
                ),
            });
        }
        Ok(())
    }
}

/// No-progress observations allowed in each controller phase.
///
/// Each phase has its own persistent streak, so interleaved Goal/GoalRepair
/// observations still diagnose an oscillating plateau.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhasePatience {
    pub build_applicability: usize,
    pub goal: usize,
    pub goal_repair: usize,
}

impl Default for PhasePatience {
    fn default() -> Self {
        Self {
            build_applicability: 12,
            goal: 12,
            goal_repair: 12,
        }
    }
}

impl PhasePatience {
    const fn for_phase(self, phase: Phase) -> usize {
        match phase {
            Phase::BuildApplicability => self.build_applicability,
            Phase::Goal => self.goal,
            Phase::GoalRepair => self.goal_repair,
        }
    }

    fn validate(self) -> Result<(), ControllerError> {
        for (field, value) in [
            ("build_applicability", self.build_applicability),
            ("goal", self.goal),
            ("goal_repair", self.goal_repair),
        ] {
            if value == 0 {
                return Err(ControllerError::InvalidConfig {
                    field: "patience",
                    problem: format!("{field} must be at least 1"),
                });
            }
        }
        Ok(())
    }
}

/// Parameters of the exact-feedback controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControllerConfig {
    pub failure_row_weight: FailureWeightSchedule,
    pub failure_fact_weight: FailureWeightSchedule,
    pub missing_goal_weight: GoalWeightSchedule,
    pub patience: PhasePatience,
    /// Minimum number of action rows in the first local remelt.
    ///
    /// The actual initial radius is
    /// `min(horizon, max(minimum_remelt_radius, horizon / 16))`.
    pub minimum_remelt_radius: usize,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            failure_row_weight: FailureWeightSchedule::default(),
            failure_fact_weight: FailureWeightSchedule::default(),
            missing_goal_weight: GoalWeightSchedule::default(),
            patience: PhasePatience::default(),
            minimum_remelt_radius: 4,
        }
    }
}

impl ControllerConfig {
    pub fn validate(self) -> Result<(), ControllerError> {
        self.failure_row_weight.validate("failure_row_weight")?;
        self.failure_fact_weight.validate("failure_fact_weight")?;
        self.missing_goal_weight.validate()?;
        self.patience.validate()?;
        if self.minimum_remelt_radius == 0 {
            return Err(ControllerError::InvalidConfig {
                field: "minimum_remelt_radius",
                problem: "must be at least 1".into(),
            });
        }
        Ok(())
    }
}

fn finite_positive(
    field: &'static str,
    component: &'static str,
    value: f64,
) -> Result<(), ControllerError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(ControllerError::InvalidConfig {
            field,
            problem: format!("{component} must be finite and positive, got {value}"),
        });
    }
    Ok(())
}

fn finite_non_negative(
    field: &'static str,
    component: &'static str,
    value: f64,
) -> Result<(), ControllerError> {
    if !value.is_finite() || value < 0.0 {
        return Err(ControllerError::InvalidConfig {
            field,
            problem: format!("{component} must be finite and non-negative, got {value}"),
        });
    }
    Ok(())
}

/// The only three semantically distinct results of exact replay.
///
/// `F` and `G` are caller-chosen identifiers. The controller never interprets
/// them, which keeps it independent of a particular planning representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExactFeedback<F, G> {
    Solved {
        applicable_prefix: usize,
    },
    FirstFailure {
        /// Tensor row corresponding to the first rejected real operator.
        row: usize,
        /// The first false precondition, if replay can identify one.
        fact: Option<F>,
        /// Goals false in the exact state at which replay stopped.
        missing_goals: Vec<G>,
        applicable_prefix: usize,
    },
    ApplicableMissingGoals {
        missing_goals: Vec<G>,
        applicable_prefix: usize,
    },
}

/// The one exact failure currently receiving additional loss.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveFailure<F> {
    pub row: usize,
    pub fact: Option<F>,
    pub row_weight: f64,
    /// `None` when replay rejected a global constraint rather than a fact.
    pub fact_weight: Option<f64>,
}

/// No-progress observations, accumulated separately within each phase.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StallCounters {
    pub build_applicability: usize,
    pub goal: usize,
    pub goal_repair: usize,
}

/// Half-open tensor-row interval to reopen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemeltWindow {
    pub particle: usize,
    pub start: usize,
    pub end: usize,
    pub phase: Phase,
    /// Requested radius before clipping at the beginning of the plan.
    pub radius: usize,
    /// One-based remelt count for this particle.
    pub ordinal: usize,
}

/// Observable result of consuming one verifier response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControllerUpdate {
    pub previous_phase: Phase,
    pub phase: Phase,
    pub phase_changed: bool,
    pub exact_progress: bool,
    pub solved: bool,
    pub remelt: Option<RemeltWindow>,
}

#[derive(Debug, Clone)]
struct PhaseTracker {
    consecutive_stalls: usize,
    /// Best number of real operators accepted by exact replay. Tensor rows are
    /// deliberately excluded: inserting a removable no-op is not progress.
    best_applicable_prefix: Option<usize>,
    best_goals_reached: Option<usize>,
    next_radius: usize,
}

impl PhaseTracker {
    fn new(initial_radius: usize) -> Self {
        Self {
            consecutive_stalls: 0,
            best_applicable_prefix: None,
            best_goals_reached: None,
            next_radius: initial_radius,
        }
    }
}

/// All scheduler state belonging to one independently optimized particle.
#[derive(Debug, Clone)]
pub struct ParticleController<F> {
    phase: Phase,
    solved: bool,
    active_failure: Option<ActiveFailure<F>>,
    goal_weights: Vec<f64>,
    missing_goal_mask: Vec<bool>,
    trackers: [PhaseTracker; 3],
    remelts: usize,
}

impl<F> ParticleController<F> {
    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn is_solved(&self) -> bool {
        self.solved
    }

    pub fn active_failure(&self) -> Option<&ActiveFailure<F>> {
        self.active_failure.as_ref()
    }

    /// Weights in the same order as [`VerifierController::goal_ids`].
    pub fn goal_weights(&self) -> &[f64] {
        &self.goal_weights
    }

    /// Last exact missing-goal set. It is retained during `GoalRepair`.
    pub fn missing_goal_mask(&self) -> &[bool] {
        &self.missing_goal_mask
    }

    pub fn stall_counters(&self) -> StallCounters {
        StallCounters {
            build_applicability: self.trackers[Phase::BuildApplicability.index()]
                .consecutive_stalls,
            goal: self.trackers[Phase::Goal.index()].consecutive_stalls,
            goal_repair: self.trackers[Phase::GoalRepair.index()].consecutive_stalls,
        }
    }

    pub fn remelts(&self) -> usize {
        self.remelts
    }
}

/// State-machine and projected-weight controller for independent particles.
#[derive(Debug, Clone)]
pub struct VerifierController<F, G> {
    config: ControllerConfig,
    horizon: usize,
    goal_ids: Vec<G>,
    particles: Vec<ParticleController<F>>,
    initial_radius: usize,
}

fn fresh_particle<F>(
    goal_count: usize,
    goal_weight_baseline: f64,
    initial_radius: usize,
) -> ParticleController<F> {
    ParticleController {
        phase: Phase::BuildApplicability,
        solved: false,
        active_failure: None,
        goal_weights: vec![goal_weight_baseline; goal_count],
        missing_goal_mask: vec![false; goal_count],
        trackers: std::array::from_fn(|_| PhaseTracker::new(initial_radius)),
        remelts: 0,
    }
}

impl<F, G> VerifierController<F, G>
where
    F: Eq,
    G: Eq,
{
    pub fn new(
        particles: usize,
        horizon: usize,
        goal_ids: Vec<G>,
        config: ControllerConfig,
    ) -> Result<Self, ControllerError> {
        config.validate()?;
        if particles == 0 {
            return Err(ControllerError::InvalidConfig {
                field: "particles",
                problem: "must be at least 1".into(),
            });
        }
        if horizon == 0 {
            return Err(ControllerError::InvalidConfig {
                field: "horizon",
                problem: "must be at least 1".into(),
            });
        }
        for right in 1..goal_ids.len() {
            if goal_ids[..right].contains(&goal_ids[right]) {
                return Err(ControllerError::DuplicateGoalDefinition { index: right });
            }
        }

        let initial_radius = config.minimum_remelt_radius.max(horizon / 16).min(horizon);
        let particle_states = (0..particles)
            .map(|_| {
                fresh_particle(
                    goal_ids.len(),
                    config.missing_goal_weight.baseline,
                    initial_radius,
                )
            })
            .collect();

        Ok(Self {
            config,
            horizon,
            goal_ids,
            particles: particle_states,
            initial_radius,
        })
    }

    pub fn horizon(&self) -> usize {
        self.horizon
    }

    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    pub fn goal_ids(&self) -> &[G] {
        &self.goal_ids
    }

    pub fn particle(&self, particle: usize) -> Result<&ParticleController<F>, ControllerError> {
        self.particles
            .get(particle)
            .ok_or(ControllerError::ParticleOutOfRange {
                particle,
                particles: self.particles.len(),
            })
    }

    /// Discard every piece of verifier-derived state for one particle.
    ///
    /// A whole-plan refresh must call this alongside resetting that particle's
    /// logits, optimizer moments, and duals. The phase, active failure, goal
    /// weights/mask, progress records, stall counters, and remelt radius/count
    /// all return to their constructor values. An invalid index is rejected
    /// without changing any particle.
    pub fn reset_particle(&mut self, particle: usize) -> Result<(), ControllerError> {
        if particle >= self.particles.len() {
            return Err(ControllerError::ParticleOutOfRange {
                particle,
                particles: self.particles.len(),
            });
        }
        self.particles[particle] = fresh_particle(
            self.goal_ids.len(),
            self.config.missing_goal_weight.baseline,
            self.initial_radius,
        );
        Ok(())
    }

    /// Consume one exact replay response.
    ///
    /// All externally supplied indices and goal identifiers are validated
    /// before the particle is mutated, so an error is transactional.
    pub fn observe(
        &mut self,
        particle: usize,
        feedback: ExactFeedback<F, G>,
    ) -> Result<ControllerUpdate, ControllerError> {
        if particle >= self.particles.len() {
            return Err(ControllerError::ParticleOutOfRange {
                particle,
                particles: self.particles.len(),
            });
        }

        let validated = self.validate_feedback(feedback)?;
        if self.particles[particle].solved {
            return Err(ControllerError::ParticleAlreadySolved { particle });
        }

        let config = self.config;
        let horizon = self.horizon;
        let initial_radius = self.initial_radius;
        let state = &mut self.particles[particle];
        let previous_phase = state.phase;

        let (exact_progress, solved, remelt) = match validated {
            ValidatedFeedback::Solved => {
                state.active_failure = None;
                update_goal_pressure(
                    state,
                    vec![false; self.goal_ids.len()],
                    config.missing_goal_weight,
                );
                transition(state, Phase::Goal);
                state.solved = true;
                (true, true, None)
            }
            ValidatedFeedback::FirstFailure {
                row,
                fact,
                applicable_prefix,
            } => {
                let phase = match state.phase {
                    Phase::BuildApplicability => Phase::BuildApplicability,
                    Phase::Goal | Phase::GoalRepair => Phase::GoalRepair,
                };
                transition(state, phase);
                // A replay that stops at its first inapplicable operator says
                // nothing about which goals the intended complete sequence
                // would reach. During initial applicability construction,
                // charging every goal false in that premature state pollutes
                // the curriculum before a terminal state exists. During goal
                // repair it is worse: it overwrites the last trustworthy goal
                // mask with facts that later rows were meant to establish.
                // GoalRepair retains that evidence and continues its dual
                // ascent: temporary inapplicability is not evidence that the
                // previously missing goal ceased to be the repair target.
                if matches!(phase, Phase::GoalRepair) {
                    update_goal_pressure(
                        state,
                        state.missing_goal_mask.clone(),
                        config.missing_goal_weight,
                    );
                }
                update_failure_focus(
                    state,
                    row,
                    fact,
                    config.failure_row_weight,
                    config.failure_fact_weight,
                );
                let (progress, remelt) = observe_applicability_progress(
                    state,
                    particle,
                    applicable_prefix,
                    row,
                    config.patience.for_phase(phase),
                    horizon,
                    initial_radius,
                );
                (progress, false, remelt)
            }
            ValidatedFeedback::ApplicableMissingGoals { missing_goal_mask } => {
                transition(state, Phase::Goal);
                state.active_failure = None;
                update_goal_pressure(state, missing_goal_mask, config.missing_goal_weight);
                let goals_reached = state
                    .missing_goal_mask
                    .iter()
                    .filter(|&&missing| !missing)
                    .count();
                let (progress, remelt) = observe_goal_progress(
                    state,
                    particle,
                    goals_reached,
                    config.patience.goal,
                    horizon,
                    initial_radius,
                );
                (progress, false, remelt)
            }
        };

        Ok(ControllerUpdate {
            previous_phase,
            phase: state.phase,
            phase_changed: previous_phase != state.phase,
            exact_progress,
            solved,
            remelt,
        })
    }

    fn validate_feedback(
        &self,
        feedback: ExactFeedback<F, G>,
    ) -> Result<ValidatedFeedback<F>, ControllerError> {
        let applicable_prefix = match &feedback {
            ExactFeedback::Solved { applicable_prefix }
            | ExactFeedback::FirstFailure {
                applicable_prefix, ..
            }
            | ExactFeedback::ApplicableMissingGoals {
                applicable_prefix, ..
            } => *applicable_prefix,
        };
        if applicable_prefix > self.horizon {
            return Err(ControllerError::ApplicablePrefixOutOfRange {
                prefix: applicable_prefix,
                horizon: self.horizon,
            });
        }

        match feedback {
            ExactFeedback::Solved { .. } => Ok(ValidatedFeedback::Solved),
            ExactFeedback::FirstFailure {
                row,
                fact,
                missing_goals,
                applicable_prefix,
            } => {
                if row >= self.horizon {
                    return Err(ControllerError::FailureRowOutOfRange {
                        row,
                        horizon: self.horizon,
                    });
                }
                // Validate external verifier data transactionally even though
                // a premature state is not terminal goal evidence.
                self.validate_missing_goals(missing_goals)?;
                Ok(ValidatedFeedback::FirstFailure {
                    row,
                    fact,
                    applicable_prefix,
                })
            }
            ExactFeedback::ApplicableMissingGoals {
                missing_goals,
                applicable_prefix: _,
            } => {
                if missing_goals.is_empty() {
                    return Err(ControllerError::EmptyMissingGoals);
                }
                let mask = self.validate_missing_goals(missing_goals)?;
                Ok(ValidatedFeedback::ApplicableMissingGoals {
                    missing_goal_mask: mask,
                })
            }
        }
    }

    fn validate_missing_goals(&self, missing_goals: Vec<G>) -> Result<Vec<bool>, ControllerError> {
        let mut mask = vec![false; self.goal_ids.len()];
        for goal in missing_goals {
            let Some(index) = self
                .goal_ids
                .iter()
                .position(|candidate| *candidate == goal)
            else {
                return Err(ControllerError::UnknownMissingGoal);
            };
            if mask[index] {
                return Err(ControllerError::DuplicateMissingGoal { index });
            }
            mask[index] = true;
        }
        Ok(mask)
    }
}

#[derive(Debug)]
enum ValidatedFeedback<F> {
    Solved,
    FirstFailure {
        row: usize,
        fact: Option<F>,
        applicable_prefix: usize,
    },
    ApplicableMissingGoals {
        missing_goal_mask: Vec<bool>,
    },
}

fn transition<F>(state: &mut ParticleController<F>, phase: Phase) {
    state.phase = phase;
}

fn update_goal_pressure<F>(
    state: &mut ParticleController<F>,
    missing_goal_mask: Vec<bool>,
    schedule: GoalWeightSchedule,
) {
    assert_eq!(
        missing_goal_mask.len(),
        state.goal_weights.len(),
        "validated goal masks match the controller dimensions"
    );
    for (goal, &missing) in missing_goal_mask.iter().enumerate() {
        if missing {
            state.goal_weights[goal] = schedule.increased(state.goal_weights[goal]);
        };
    }
    state.missing_goal_mask = missing_goal_mask;
}

fn update_failure_focus<F: Eq>(
    state: &mut ParticleController<F>,
    row: usize,
    fact: Option<F>,
    row_schedule: FailureWeightSchedule,
    fact_schedule: FailureWeightSchedule,
) {
    let same_target = state
        .active_failure
        .as_ref()
        .is_some_and(|active| active.row == row && active.fact == fact);
    if same_target {
        let active = state
            .active_failure
            .as_mut()
            .expect("same_target implies an active failure");
        active.row_weight = row_schedule.increased(active.row_weight);
        if let Some(weight) = active.fact_weight.as_mut() {
            *weight = fact_schedule.increased(*weight);
        }
    } else {
        let fact_weight = fact.as_ref().map(|_| fact_schedule.initial);
        state.active_failure = Some(ActiveFailure {
            row,
            fact,
            row_weight: row_schedule.initial,
            fact_weight,
        });
    }
}

fn observe_applicability_progress<F>(
    state: &mut ParticleController<F>,
    particle: usize,
    applicable_prefix: usize,
    failure_row: usize,
    patience: usize,
    horizon: usize,
    initial_radius: usize,
) -> (bool, Option<RemeltWindow>) {
    let phase = state.phase;
    let tracker = &mut state.trackers[phase.index()];
    let progress = tracker
        .best_applicable_prefix
        .is_none_or(|best| applicable_prefix > best);
    if progress {
        tracker.best_applicable_prefix = Some(applicable_prefix);
        tracker.consecutive_stalls = 0;
        tracker.next_radius = initial_radius;
        return (true, None);
    }

    tracker.consecutive_stalls += 1;
    if tracker.consecutive_stalls < patience {
        return (false, None);
    }

    tracker.consecutive_stalls = 0;
    let radius = tracker.next_radius.min(horizon);
    let failure_end = failure_row + 1;
    let start = failure_end.saturating_sub(radius);
    // Initial applicability repair is local: rows after the first failure
    // have not been executed and are not evidence. Goal repair is different.
    // A newly inserted goal achiever can require every later causal role to
    // move right (or otherwise change), so keeping the crystallized tail would
    // make the required coordinated repair unrepresentable to the optimizer.
    // Preserve only the prefix before `start` and reopen the whole suffix.
    let end = if matches!(phase, Phase::GoalRepair) {
        horizon
    } else {
        failure_end
    };
    tracker.next_radius = radius.saturating_mul(2).min(horizon);
    state.remelts += 1;
    (
        false,
        Some(RemeltWindow {
            particle,
            start,
            end,
            phase,
            radius,
            ordinal: state.remelts,
        }),
    )
}

fn observe_goal_progress<F>(
    state: &mut ParticleController<F>,
    particle: usize,
    goals_reached: usize,
    patience: usize,
    horizon: usize,
    initial_radius: usize,
) -> (bool, Option<RemeltWindow>) {
    let tracker = &mut state.trackers[Phase::Goal.index()];
    let progress = tracker
        .best_goals_reached
        .is_none_or(|best| goals_reached > best);
    if progress {
        tracker.best_goals_reached = Some(goals_reached);
        tracker.consecutive_stalls = 0;
        tracker.next_radius = initial_radius;
        return (true, None);
    }

    tracker.consecutive_stalls += 1;
    if tracker.consecutive_stalls < patience {
        return (false, None);
    }

    tracker.consecutive_stalls = 0;
    let radius = tracker.next_radius.min(horizon);
    tracker.next_radius = radius.saturating_mul(2).min(horizon);
    state.remelts += 1;
    (
        false,
        Some(RemeltWindow {
            particle,
            start: horizon - radius,
            end: horizon,
            phase: Phase::Goal,
            radius,
            ordinal: state.remelts,
        }),
    )
}

/// Invalid controller configuration or malformed verifier feedback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControllerError {
    InvalidConfig {
        field: &'static str,
        problem: String,
    },
    ParticleOutOfRange {
        particle: usize,
        particles: usize,
    },
    ParticleAlreadySolved {
        particle: usize,
    },
    FailureRowOutOfRange {
        row: usize,
        horizon: usize,
    },
    ApplicablePrefixOutOfRange {
        prefix: usize,
        horizon: usize,
    },
    DuplicateGoalDefinition {
        index: usize,
    },
    EmptyMissingGoals,
    UnknownMissingGoal,
    DuplicateMissingGoal {
        index: usize,
    },
}

impl fmt::Display for ControllerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { field, problem } => {
                write!(f, "controller config field `{field}`: {problem}")
            }
            Self::ParticleOutOfRange {
                particle,
                particles,
            } => write!(
                f,
                "particle {particle} is out of range for {particles} particles"
            ),
            Self::ParticleAlreadySolved { particle } => {
                write!(f, "particle {particle} has already solved the task")
            }
            Self::FailureRowOutOfRange { row, horizon } => {
                write!(f, "failure row {row} is outside horizon {horizon}")
            }
            Self::ApplicablePrefixOutOfRange { prefix, horizon } => {
                write!(f, "applicable prefix {prefix} exceeds horizon {horizon}")
            }
            Self::DuplicateGoalDefinition { index } => {
                write!(f, "goal definition at index {index} is a duplicate")
            }
            Self::EmptyMissingGoals => write!(
                f,
                "an applicable non-solution must identify at least one missing goal"
            ),
            Self::UnknownMissingGoal => {
                write!(
                    f,
                    "verifier reported a goal not registered with the controller"
                )
            }
            Self::DuplicateMissingGoal { index } => {
                write!(f, "verifier reported goal index {index} more than once")
            }
        }
    }
}

impl std::error::Error for ControllerError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_patience(patience: usize) -> ControllerConfig {
        ControllerConfig {
            failure_row_weight: FailureWeightSchedule {
                initial: 2.0,
                growth: 2.0,
                cap: 8.0,
            },
            failure_fact_weight: FailureWeightSchedule {
                initial: 4.0,
                growth: 1.5,
                cap: 7.0,
            },
            missing_goal_weight: GoalWeightSchedule {
                baseline: 1.0,
                increment: 2.0,
                cap: 6.0,
            },
            patience: PhasePatience {
                build_applicability: patience,
                goal: patience,
                goal_repair: patience,
            },
            minimum_remelt_radius: 4,
        }
    }

    #[test]
    fn phases_preserve_goal_pressure_during_repair() {
        let mut controller =
            VerifierController::new(1, 32, vec!["g0", "g1"], config_with_patience(12)).unwrap();

        let update = controller
            .observe(
                0,
                ExactFeedback::FirstFailure {
                    row: 3,
                    fact: Some("p"),
                    missing_goals: vec!["g1"],
                    applicable_prefix: 3,
                },
            )
            .unwrap();
        assert_eq!(update.phase, Phase::BuildApplicability);
        assert_eq!(controller.particle(0).unwrap().goal_weights(), &[1.0, 1.0]);
        assert_eq!(
            controller.particle(0).unwrap().active_failure(),
            Some(&ActiveFailure {
                row: 3,
                fact: Some("p"),
                row_weight: 2.0,
                fact_weight: Some(4.0),
            })
        );

        let update = controller
            .observe(
                0,
                ExactFeedback::ApplicableMissingGoals {
                    missing_goals: vec!["g1"],
                    applicable_prefix: 8,
                },
            )
            .unwrap();
        assert_eq!(update.phase, Phase::Goal);
        assert!(update.phase_changed);
        assert_eq!(controller.particle(0).unwrap().goal_weights(), &[1.0, 3.0]);

        let update = controller
            .observe(
                0,
                ExactFeedback::FirstFailure {
                    row: 6,
                    fact: Some("q"),
                    missing_goals: vec!["g1"],
                    applicable_prefix: 6,
                },
            )
            .unwrap();
        assert_eq!(update.phase, Phase::GoalRepair);
        assert_eq!(controller.particle(0).unwrap().goal_weights(), &[1.0, 5.0]);
        assert_eq!(
            controller.particle(0).unwrap().missing_goal_mask(),
            &[false, true]
        );

        let update = controller
            .observe(
                0,
                ExactFeedback::ApplicableMissingGoals {
                    missing_goals: vec!["g1"],
                    applicable_prefix: 8,
                },
            )
            .unwrap();
        assert_eq!(update.phase, Phase::Goal);
        assert_eq!(controller.particle(0).unwrap().goal_weights(), &[1.0, 6.0]);

        let update = controller
            .observe(
                0,
                ExactFeedback::Solved {
                    applicable_prefix: 8,
                },
            )
            .unwrap();
        assert!(update.solved);
        assert_eq!(controller.particle(0).unwrap().goal_weights(), &[1.0, 6.0]);
    }

    #[test]
    fn repeated_failure_uses_projected_multiplicative_weights() {
        let mut controller =
            VerifierController::new(1, 16, vec![0usize], config_with_patience(12)).unwrap();
        let failure = || ExactFeedback::FirstFailure {
            row: 5,
            fact: Some(9usize),
            missing_goals: vec![0],
            applicable_prefix: 5,
        };

        controller.observe(0, failure()).unwrap();
        controller.observe(0, failure()).unwrap();
        controller.observe(0, failure()).unwrap();
        controller.observe(0, failure()).unwrap();
        let active = controller.particle(0).unwrap().active_failure().unwrap();
        assert_eq!(active.row_weight, 8.0);
        assert_eq!(active.fact_weight, Some(7.0));

        controller
            .observe(
                0,
                ExactFeedback::FirstFailure {
                    row: 6,
                    fact: Some(10),
                    missing_goals: vec![0],
                    applicable_prefix: 6,
                },
            )
            .unwrap();
        let active = controller.particle(0).unwrap().active_failure().unwrap();
        assert_eq!(active.row, 6);
        assert_eq!(active.row_weight, 2.0);
        assert_eq!(active.fact_weight, Some(4.0));
    }

    #[test]
    fn goal_weights_remain_monotone_for_alternating_missing_sets() {
        let mut controller =
            VerifierController::<usize, _>::new(1, 16, vec![0, 1, 2], config_with_patience(12))
                .unwrap();
        controller
            .observe(
                0,
                ExactFeedback::ApplicableMissingGoals {
                    missing_goals: vec![0, 1],
                    applicable_prefix: 8,
                },
            )
            .unwrap();
        assert_eq!(
            controller.particle(0).unwrap().goal_weights(),
            &[3.0, 3.0, 1.0]
        );

        controller
            .observe(
                0,
                ExactFeedback::ApplicableMissingGoals {
                    missing_goals: vec![1],
                    applicable_prefix: 8,
                },
            )
            .unwrap();
        assert_eq!(
            controller.particle(0).unwrap().goal_weights(),
            &[3.0, 5.0, 1.0]
        );

        controller
            .observe(
                0,
                ExactFeedback::ApplicableMissingGoals {
                    missing_goals: vec![0, 2],
                    applicable_prefix: 8,
                },
            )
            .unwrap();
        assert_eq!(
            controller.particle(0).unwrap().goal_weights(),
            &[5.0, 5.0, 3.0]
        );
    }

    #[test]
    fn applicability_remelts_expand_backward_and_reset_after_progress() {
        let mut controller =
            VerifierController::new(1, 64, vec![0usize], config_with_patience(2)).unwrap();
        let failure = |row, prefix| ExactFeedback::FirstFailure {
            row,
            fact: Some(1usize),
            missing_goals: vec![0],
            applicable_prefix: prefix,
        };

        assert!(
            controller
                .observe(0, failure(20, 5))
                .unwrap()
                .exact_progress
        );
        assert!(
            controller
                .observe(0, failure(20, 5))
                .unwrap()
                .remelt
                .is_none()
        );
        assert_eq!(
            controller.observe(0, failure(20, 5)).unwrap().remelt,
            Some(RemeltWindow {
                particle: 0,
                start: 17,
                end: 21,
                phase: Phase::BuildApplicability,
                radius: 4,
                ordinal: 1,
            })
        );

        controller.observe(0, failure(20, 5)).unwrap();
        assert_eq!(
            controller.observe(0, failure(20, 5)).unwrap().remelt,
            Some(RemeltWindow {
                particle: 0,
                start: 13,
                end: 21,
                phase: Phase::BuildApplicability,
                radius: 8,
                ordinal: 2,
            })
        );

        assert!(
            controller
                .observe(0, failure(22, 6))
                .unwrap()
                .exact_progress
        );
        controller.observe(0, failure(22, 6)).unwrap();
        assert_eq!(
            controller.observe(0, failure(22, 6)).unwrap().remelt,
            Some(RemeltWindow {
                particle: 0,
                start: 19,
                end: 23,
                phase: Phase::BuildApplicability,
                radius: 4,
                ordinal: 3,
            })
        );
    }

    #[test]
    fn goal_plateau_expands_a_suffix_before_reopening_the_whole_plan() {
        let mut controller =
            VerifierController::<usize, _>::new(1, 64, vec![0, 1], config_with_patience(2))
                .unwrap();
        let feedback = || ExactFeedback::ApplicableMissingGoals {
            missing_goals: vec![1],
            applicable_prefix: 30,
        };

        controller.observe(0, feedback()).unwrap();
        controller.observe(0, feedback()).unwrap();
        let update = controller.observe(0, feedback()).unwrap();
        assert_eq!(
            update.remelt,
            Some(RemeltWindow {
                particle: 0,
                start: 60,
                end: 64,
                phase: Phase::Goal,
                radius: 4,
                ordinal: 1,
            })
        );
        assert_eq!(
            controller.particle(0).unwrap().stall_counters(),
            StallCounters::default()
        );
    }

    #[test]
    fn phase_interleaving_preserves_each_plateau_streak() {
        let mut controller =
            VerifierController::new(1, 16, vec![0usize], config_with_patience(2)).unwrap();
        let failure = || ExactFeedback::FirstFailure {
            row: 8,
            fact: Some(1usize),
            missing_goals: vec![0],
            applicable_prefix: 4,
        };
        let goal_feedback = || ExactFeedback::ApplicableMissingGoals {
            missing_goals: vec![0],
            applicable_prefix: 8,
        };

        // Establish one best observation in each phase.
        controller.observe(0, goal_feedback()).unwrap();
        controller.observe(0, failure()).unwrap();

        // Interleave the phases. Neither transition may erase the other
        // phase's no-progress observation.
        assert!(
            controller
                .observe(0, goal_feedback())
                .unwrap()
                .remelt
                .is_none()
        );
        assert!(controller.observe(0, failure()).unwrap().remelt.is_none());
        assert_eq!(
            controller.particle(0).unwrap().stall_counters(),
            StallCounters {
                build_applicability: 0,
                goal: 1,
                goal_repair: 1,
            }
        );

        let update = controller.observe(0, goal_feedback()).unwrap();
        assert_eq!(update.phase, Phase::Goal);
        assert_eq!(
            update.remelt,
            Some(RemeltWindow {
                particle: 0,
                start: 12,
                end: 16,
                phase: Phase::Goal,
                radius: 4,
                ordinal: 1,
            })
        );
        assert_eq!(
            controller.particle(0).unwrap().stall_counters(),
            StallCounters {
                build_applicability: 0,
                goal: 0,
                goal_repair: 1,
            }
        );
    }

    #[test]
    fn moving_a_failure_past_noops_is_not_exact_progress() {
        let mut controller =
            VerifierController::new(1, 32, vec![0usize], config_with_patience(2)).unwrap();
        let failure = |row| ExactFeedback::FirstFailure {
            row,
            fact: Some(1usize),
            missing_goals: vec![0],
            applicable_prefix: 3,
        };

        assert!(controller.observe(0, failure(5)).unwrap().exact_progress);
        let moved = controller.observe(0, failure(11)).unwrap();
        assert!(!moved.exact_progress);
        assert!(moved.remelt.is_none());
        let update = controller.observe(0, failure(14)).unwrap();
        assert!(!update.exact_progress);
        assert_eq!(
            update.remelt,
            Some(RemeltWindow {
                particle: 0,
                start: 11,
                end: 15,
                phase: Phase::BuildApplicability,
                radius: 4,
                ordinal: 1,
            })
        );
    }

    #[test]
    fn goal_repair_remelts_the_complete_downstream_suffix() {
        let mut controller =
            VerifierController::new(1, 16, vec![0usize], config_with_patience(1)).unwrap();

        controller
            .observe(
                0,
                ExactFeedback::ApplicableMissingGoals {
                    missing_goals: vec![0],
                    applicable_prefix: 8,
                },
            )
            .unwrap();
        let first = controller
            .observe(
                0,
                ExactFeedback::FirstFailure {
                    row: 7,
                    fact: Some(1usize),
                    missing_goals: vec![0],
                    applicable_prefix: 6,
                },
            )
            .unwrap();
        assert!(first.exact_progress);
        assert!(first.remelt.is_none());

        let stalled = controller
            .observe(
                0,
                ExactFeedback::FirstFailure {
                    row: 7,
                    fact: Some(1usize),
                    missing_goals: vec![0],
                    applicable_prefix: 6,
                },
            )
            .unwrap();
        assert_eq!(
            stalled.remelt,
            Some(RemeltWindow {
                particle: 0,
                start: 4,
                end: 16,
                phase: Phase::GoalRepair,
                radius: 4,
                ordinal: 1,
            })
        );
    }

    #[test]
    fn malformed_feedback_is_rejected_without_mutation() {
        let mut controller =
            VerifierController::<usize, _>::new(1, 8, vec![0, 1], config_with_patience(2)).unwrap();
        let before = controller.particle(0).unwrap().goal_weights().to_vec();

        assert_eq!(
            controller.observe(
                0,
                ExactFeedback::ApplicableMissingGoals {
                    missing_goals: vec![2],
                    applicable_prefix: 8,
                }
            ),
            Err(ControllerError::UnknownMissingGoal)
        );
        assert_eq!(controller.particle(0).unwrap().goal_weights(), before);
        assert_eq!(
            controller.observe(
                0,
                ExactFeedback::FirstFailure {
                    row: 2,
                    fact: Some(0),
                    missing_goals: vec![2],
                    applicable_prefix: 2,
                }
            ),
            Err(ControllerError::UnknownMissingGoal)
        );
        assert!(controller.particle(0).unwrap().active_failure().is_none());
        assert_eq!(controller.particle(0).unwrap().goal_weights(), before);
        assert_eq!(
            controller.observe(
                0,
                ExactFeedback::FirstFailure {
                    row: 2,
                    fact: Some(0),
                    missing_goals: vec![0, 0],
                    applicable_prefix: 2,
                }
            ),
            Err(ControllerError::DuplicateMissingGoal { index: 0 })
        );
        assert!(controller.particle(0).unwrap().active_failure().is_none());
        assert_eq!(controller.particle(0).unwrap().goal_weights(), before);
        assert_eq!(
            controller.observe(
                0,
                ExactFeedback::FirstFailure {
                    row: 8,
                    fact: Some(0),
                    missing_goals: vec![0],
                    applicable_prefix: 8,
                }
            ),
            Err(ControllerError::FailureRowOutOfRange { row: 8, horizon: 8 })
        );
        assert_eq!(
            controller.particle(0).unwrap().phase(),
            Phase::BuildApplicability
        );
    }

    #[test]
    fn reset_particle_restores_all_fresh_controller_state() {
        let mut controller =
            VerifierController::new(2, 16, vec![0usize, 1], config_with_patience(1)).unwrap();
        let failure = || ExactFeedback::FirstFailure {
            row: 6,
            fact: Some(4usize),
            missing_goals: vec![1],
            applicable_prefix: 3,
        };
        controller.observe(0, failure()).unwrap();
        assert!(controller.observe(0, failure()).unwrap().remelt.is_some());
        assert_eq!(controller.particle(0).unwrap().goal_weights(), &[1.0, 1.0]);
        assert_eq!(controller.particle(0).unwrap().remelts(), 1);

        assert_eq!(
            controller.reset_particle(2),
            Err(ControllerError::ParticleOutOfRange {
                particle: 2,
                particles: 2,
            })
        );
        assert_eq!(controller.particle(0).unwrap().remelts(), 1);

        controller.reset_particle(0).unwrap();
        let reset = controller.particle(0).unwrap();
        assert_eq!(reset.phase(), Phase::BuildApplicability);
        assert!(!reset.is_solved());
        assert!(reset.active_failure().is_none());
        assert_eq!(reset.goal_weights(), &[1.0, 1.0]);
        assert_eq!(reset.missing_goal_mask(), &[false, false]);
        assert_eq!(reset.stall_counters(), StallCounters::default());
        assert_eq!(reset.remelts(), 0);

        // Progress records were reset as well: the old prefix is new progress.
        assert!(controller.observe(0, failure()).unwrap().exact_progress);
        let untouched = controller.particle(1).unwrap();
        assert_eq!(untouched.phase(), Phase::BuildApplicability);
        assert_eq!(untouched.goal_weights(), &[1.0, 1.0]);
        assert_eq!(untouched.remelts(), 0);
    }

    #[test]
    fn solved_particles_are_terminal() {
        let mut controller =
            VerifierController::<usize, usize>::new(1, 8, Vec::new(), config_with_patience(2))
                .unwrap();
        let update = controller
            .observe(
                0,
                ExactFeedback::Solved {
                    applicable_prefix: 0,
                },
            )
            .unwrap();
        assert!(update.solved);
        assert!(controller.particle(0).unwrap().is_solved());
        assert_eq!(
            controller.observe(
                0,
                ExactFeedback::Solved {
                    applicable_prefix: 0
                }
            ),
            Err(ControllerError::ParticleAlreadySolved { particle: 0 })
        );
    }

    #[test]
    fn invalid_config_and_duplicate_goals_fail_fast() {
        let mut config = config_with_patience(2);
        config.failure_row_weight.cap = 1.0;
        assert_eq!(
            VerifierController::<usize, usize>::new(1, 8, vec![0], config).unwrap_err(),
            ControllerError::InvalidConfig {
                field: "failure_row_weight",
                problem: "cap (1) must exceed initial (2)".into(),
            }
        );
        assert_eq!(
            VerifierController::<usize, usize>::new(1, 8, vec![0, 0], config_with_patience(2))
                .unwrap_err(),
            ControllerError::DuplicateGoalDefinition { index: 1 }
        );

        let mut config = config_with_patience(2);
        config.failure_fact_weight.growth = 1.0;
        assert_eq!(
            VerifierController::<usize, usize>::new(1, 8, vec![0], config).unwrap_err(),
            ControllerError::InvalidConfig {
                field: "failure_fact_weight",
                problem: "growth must be finite and exceed 1, got 1".into(),
            }
        );
    }
}
