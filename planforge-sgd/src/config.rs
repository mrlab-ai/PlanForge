//! Engine configuration.
//!
//! Defaults reproduce the tuning snapshot of the research note the method comes
//! from, so a bare `sgd()` is the documented configuration rather than an
//! arbitrary one. Everything is validated up front and nothing is clamped: an
//! out-of-range value is a caller error, not something to quietly repair.

/// How the horizon is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HorizonPolicy {
    /// One fixed horizon. Use this for matched-budget comparisons, where the
    /// horizon has to be held still.
    Fixed(usize),
    /// Start at `start` and grow by `growth` after each exhausted budget, up to
    /// `max`. Needed because the engine may not compute a horizon bound — that
    /// would take a heuristic, which is exactly what it is not allowed to use.
    Dovetail {
        start: usize,
        growth: f64,
        max: usize,
    },
}

/// Relationship between the exact execution plan and the optional causal copy.
///
/// This is deliberately an explicit mode instead of a boolean: `shadow` is a
/// correctness check with an identical copy, while `staged` is the actual
/// overparameterized continuation method. Conflating those two made it too easy
/// to mistake a compiling experimental path for a validated default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalCopyMode {
    /// Use one action distribution: causal witnesses attach directly to the
    /// exact execution plan and no separate action copy participates.
    Off,
    /// Keep the causal copy identical to execution to check equivalence.
    Shadow,
    /// Discover a causal proof and transfer it to execution in global stages.
    Staged,
}

/// Global continuation stage selected from total update progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalStage {
    Shadow,
    Discovery,
    Proof,
    Transfer,
    Takeover,
    Polish,
}

impl std::fmt::Display for CausalCopyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::Shadow => "shadow",
            Self::Staged => "staged",
        })
    }
}

impl std::str::FromStr for CausalCopyMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "off" => Ok(Self::Off),
            "shadow" => Ok(Self::Shadow),
            "staged" => Ok(Self::Staged),
            other => Err(format!(
                "expected one of `off`, `shadow`, or `staged`, got `{other}`"
            )),
        }
    }
}

impl HorizonPolicy {
    /// The horizon of round `round`, or `None` once the schedule is exhausted.
    pub fn horizon_for_round(&self, round: usize) -> Option<usize> {
        match *self {
            Self::Fixed(horizon) => {
                if round == 0 {
                    Some(horizon)
                } else {
                    None
                }
            }
            Self::Dovetail { start, growth, max } => {
                let mut horizon = start as f64;
                for _ in 0..round {
                    horizon *= growth;
                }
                let horizon = horizon.round() as usize;
                if horizon <= max { Some(horizon) } else { None }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SgdConfig {
    pub horizon: HorizonPolicy,
    /// Independently optimized complete-plan particles. They share nothing:
    /// no ranking, no selection, no imitation.
    pub particles: usize,
    /// Update budget per horizon round.
    pub updates: usize,

    pub learning_rate: f64,
    /// Joint gradient-norm cap for each particle. Action, state, and causal
    /// variables retain their relative scale within a particle, while one
    /// explosive particle cannot shrink every other particle's update.
    pub grad_clip: f64,
    pub action_logit_clip: f64,
    pub state_logit_clip: f64,
    /// Initial logit gap between the explicit no-op and every real action.
    ///
    /// Starting from a sparse plan is the causal analogue of a lottery-ticket
    /// initialization: terminal goals activate producer rows, whose
    /// preconditions recursively activate earlier rows. A dense random action
    /// soup instead creates unrelated precondition demand at every timestep.
    pub initial_noop_logit_gap: f64,
    /// Carry action identities in latent tokens and assign them to execution
    /// rows with a doubly stochastic temporal schedule.
    pub temporal_tokens: bool,
    /// Tail rows softly reserved as no-op working memory until a temporal
    /// particle commits to a verifier-derived repair scaffold.
    pub temporal_reserved_slots: usize,
    /// Weight of the pre-commit no-op reservation loss.
    pub temporal_reservation_weight: f64,
    /// Updates without exact progress after temporal unlock before only the
    /// repair-token actions and interleaving gates receive a full-support
    /// stochastic restart. Zero disables repair restarts.
    pub temporal_restart_patience: usize,
    /// Temporal-assignment temperature from melted to crystallized.
    pub schedule_temperature: (f64, f64),
    /// Alternating row/column normalizations in the Sinkhorn layer.
    pub schedule_sinkhorn_iterations: usize,
    /// Initial preference for token `k` to occupy execution row `k`.
    pub schedule_identity_bias: f64,
    /// Final pressure toward a permutation schedule.
    pub schedule_integrality_final: f64,
    /// Every `slot_slack_window`th row starts with a lower occupancy logit.
    /// Zero disables the periodic insertion-slack prior; otherwise the window
    /// must contain at least two rows.
    pub slot_slack_window: usize,
    /// Initial occupancy-logit difference between ordinary and slack rows.
    pub slot_slack_logit_gap: f64,
    /// Weight on retaining at least one unit of no-op mass per local window.
    pub slot_slack_weight: f64,
    /// Multiplier on verifier-triggered raw and supported insertion evidence.
    pub insertion_repair_weight: f64,
    /// Minimum verifier-applicable prefix, as a fraction of the horizon,
    /// before a checkpoint may be treated as an insertion scaffold.
    pub insertion_min_prefix_fraction: f64,
    /// Multiplier on the soft trust region around the latest exact checkpoint.
    pub anchor_trust_weight: f64,
    /// Relative weight of analytic producer-to-terminal survival geometry in
    /// the late delete-aware goal objective. Zero is the recurrent legacy
    /// boundary; one averages terminal recurrence and analytic survival.
    pub goal_survival_weight: f64,
    /// Weight of verifier-triggered continuous backward chaining from missing
    /// terminal goals through producer preconditions. Zero removes the
    /// experimental objective from the graph.
    pub backward_bridge_weight: f64,

    /// Action temperature: start, midpoint, end of the continuation schedule.
    pub action_temperature: (f64, f64, f64),
    /// State temperature: start and end.
    pub state_temperature: (f64, f64),
    /// Fraction of the budget after which integrality starts being enforced.
    /// Enforcing it early freezes random action choices before any causal
    /// structure exists.
    pub crystallization_start: f64,

    pub rho_precondition: f64,
    pub rho_transition: f64,
    pub rho_goal: f64,
    pub dual_growth: f64,
    pub dual_decay: f64,
    pub dual_cap: f64,
    pub dual_period: usize,
    /// Fraction of largest residuals that get extra pressure, so one decisive
    /// violation cannot hide behind thousands of satisfied constraints.
    pub top_residual_fraction: f64,

    pub action_integrality_final: f64,
    pub state_integrality_final: f64,
    /// Final weight on the single least-integral action row and state row of
    /// each particle. Mean impurity can hide one decisive ambiguous row behind
    /// a long horizon; this bottleneck term cannot.
    pub worst_integrality_final: f64,
    /// Whether and how an overparameterized causal action copy is used.
    pub causal_copy: CausalCopyMode,
    /// End of the exact-copy baseline stage, as global update progress.
    pub causal_shadow_end: f64,
    /// End of causal-proof discovery.
    pub causal_discovery_end: f64,
    /// End of executable-proof construction.
    pub causal_proof_end: f64,
    /// End of frozen-teacher transfer into the exact execution plan.
    pub causal_transfer_end: f64,
    /// End of exact takeover; the remaining budget is deterministic polish.
    pub causal_takeover_end: f64,
    /// Causal-copy action temperature over global update progress.
    pub q_action_temperature: (f64, f64),
    /// Standard deviation of the one-time Q-only perturbation after shadowing.
    pub q_logit_perturbation: f64,
    /// Maximum causal residual at which Q is eligible to become a teacher.
    pub teacher_tolerance: f64,
    /// Peak weight of directional `KL(stop_gradient(Q) || P)` transfer.
    pub teacher_weight: f64,
    /// Maximum allowed softmax-probability ratio between the strongest
    /// inapplicable and applicable actions at an exact failure row.
    ///
    /// Because softmax preserves logit order, driving the corresponding
    /// ranking hinge to zero certifies that argmax decoding selects an
    /// applicable action.  A ratio below one makes that ordering strict without
    /// requiring all probability mass on other inapplicable actions to vanish.
    pub applicability_barrier_margin: f64,
    /// Weight on `-log` total exactly-applicable mass at the first failure row.
    ///
    /// This supplies a dense symmetric gradient to every applicable action;
    /// the separate logit-ranking hinge remains the hard argmax certificate.
    pub applicability_mass_weight: f64,
    /// Updates during which a remelted row stays hot and free of integrality.
    pub remelt_cooldown_updates: usize,
    /// Per-particle bottleneck p-norm used in deterministic polish.
    pub polish_p_norm: f64,
    /// Weight encouraging all real actions to precede the no-op suffix.
    pub noop_suffix_weight: f64,
    /// Weight of fact-grouped latent causal-link source/threat constraints.
    /// These links are distinct proof-witness parameters, not action aliases.
    pub causal_link_weight: f64,
    /// Adam learning rate for causal-link witness logits. They are optimized
    /// in a separate parameter group so their much larger tensor does not set
    /// the trajectory variables' global gradient clip.
    pub causal_link_learning_rate: f64,
    /// Final demand-weighted integrality pressure on causal-link choices.
    pub causal_link_integrality_final: f64,
    /// Causal-link temperature from the melted to crystallized point of each
    /// cycle. Links stay softer than actions long enough to move to a real
    /// achiever before committing to one witness.
    pub causal_link_temperature: (f64, f64),
    /// Extra initial logit on one independently sampled valid source for every
    /// `(consumer, fact)` witness. This gives each particle a distinct sparse
    /// temporal causal skeleton without selecting any operator.
    pub causal_link_initial_bias: f64,
    /// Residual below which a constraint counts as satisfied for dual updates.
    pub residual_tolerance: f64,

    /// Updates between exact verifications of the argmax plan.
    pub verify_period: usize,
    /// Emit a full action-probability snapshot every N updates. Zero disables
    /// tracing; this is diagnostic observation and never changes the graph.
    pub trace_period: usize,
    /// Particle whose trajectory is recorded when tracing is enabled.
    pub trace_particle: usize,
    /// Growth factor applied to duals implicated by an exact failure.
    pub focus_growth: f64,
    pub focus_cap: f64,

    /// Melt/crystallize cycles over the budget.
    ///
    /// A single monotone anneal is a trap: once a particle crystallizes onto an
    /// invalid integral assignment its softmax rows saturate, the gradients
    /// vanish, and nothing can move it again. Cycling the temperature reopens
    /// those rows. Particles are phase-shifted against each other, so at any
    /// moment some are exploring while others are committing — that is where the
    /// population's diversity comes from, with no ranking or selection.
    pub cycles: usize,
    /// Langevin noise on the logits, from start to end of each cycle.
    pub noise: (f64, f64),
    /// Optimizer updates without exact progress before a particle is remelted:
    /// its Adam moments are cleared and a noise burst is injected. Measuring
    /// this in updates makes the behavior independent of `verify_period`.
    pub remelt_patience: usize,
    /// Scale of that noise burst.
    pub remelt_noise: f64,
    /// Multiplicative logit shrink inside a remelt window before noise is
    /// injected. Values below one reduce saturated logit gaps and therefore
    /// genuinely raise entropy instead of merely jittering a frozen choice.
    pub remelt_shrink: f64,
    /// Global progress after which no new remelt may begin.
    ///
    /// Plateau escape remains available during most of polish, but the final
    /// tail is deterministic so recently reopened rows can recrystallize and
    /// exact verification can observe the repaired sequence.
    pub remelt_stop_progress: f64,

    /// Replace a stalled particle with an independently random complete plan.
    /// Off by default: with it on, a solved instance cannot be attributed to
    /// gradient descent rather than to random sampling.
    pub refresh: bool,
    /// Verifier checks between refresh events.
    pub refresh_period: usize,
    pub refresh_particles: usize,

    pub seed: u64,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            horizon: HorizonPolicy::Dovetail {
                start: 8,
                growth: 2.0,
                max: 512,
            },
            particles: 8,
            updates: 20_000,
            learning_rate: 0.04,
            grad_clip: 30.0,
            action_logit_clip: 12.0,
            state_logit_clip: 10.0,
            initial_noop_logit_gap: 0.0,
            temporal_tokens: false,
            temporal_reserved_slots: 0,
            temporal_reservation_weight: 10.0,
            temporal_restart_patience: 0,
            // Monotone temporal execution is already a hard lattice path.
            // Keep its soft backward distribution warm instead of destroying
            // gap gradients with redundant annealing and integrality pressure.
            schedule_temperature: (2.0, 2.0),
            schedule_sinkhorn_iterations: 12,
            schedule_identity_bias: 2.0,
            schedule_integrality_final: 0.0,
            slot_slack_window: 0,
            slot_slack_logit_gap: 2.0,
            slot_slack_weight: 1.0,
            insertion_repair_weight: 0.1,
            insertion_min_prefix_fraction: 0.9,
            anchor_trust_weight: 0.1,
            goal_survival_weight: 0.0,
            backward_bridge_weight: 0.0,
            action_temperature: (2.0, 0.75, 0.16),
            state_temperature: (1.5, 0.28),
            crystallization_start: 0.55,
            rho_precondition: 2.0,
            rho_transition: 3.0,
            rho_goal: 8.0,
            dual_growth: 1.03,
            dual_decay: 0.995,
            dual_cap: 80.0,
            dual_period: 5,
            // Top-k selection currently requires a host synchronization and
            // sort. Keep the mechanism available for diagnosed bottlenecks,
            // but do not pay that cost on every ordinary update.
            top_residual_fraction: 0.0,
            action_integrality_final: 12.0,
            state_integrality_final: 8.0,
            worst_integrality_final: 4.0,
            // The experimental causal copy remains opt-in until shadow
            // equivalence and the staged regression gates have passed.
            causal_copy: CausalCopyMode::Off,
            causal_shadow_end: 0.10,
            causal_discovery_end: 0.30,
            causal_proof_end: 0.50,
            causal_transfer_end: 0.70,
            causal_takeover_end: 0.85,
            q_action_temperature: (2.5, 0.5),
            q_logit_perturbation: 0.05,
            teacher_tolerance: 0.05,
            teacher_weight: 12.0,
            applicability_barrier_margin: 0.95,
            applicability_mass_weight: 0.25,
            remelt_cooldown_updates: 80,
            polish_p_norm: 8.0,
            noop_suffix_weight: 1.0,
            causal_link_weight: 10.0,
            causal_link_learning_rate: 0.12,
            causal_link_integrality_final: 1.0,
            causal_link_temperature: (1.5, 0.5),
            causal_link_initial_bias: 0.0,
            residual_tolerance: 2e-3,
            verify_period: 10,
            trace_period: 0,
            trace_particle: 0,
            focus_growth: 1.06,
            focus_cap: 18.0,
            cycles: 6,
            noise: (0.025, 0.001),
            // Preserve the former 12 verifier-check plateau at the default
            // verify period while expressing the invariant in optimizer updates.
            remelt_patience: 120,
            remelt_noise: 0.6,
            remelt_shrink: 0.35,
            remelt_stop_progress: 0.95,
            refresh: false,
            refresh_period: 30,
            refresh_particles: 1,
            seed: 1,
        }
    }
}

/// A configuration value that cannot be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SgdConfigError {
    pub field: &'static str,
    pub problem: String,
}

impl std::fmt::Display for SgdConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sgd config field `{}`: {}", self.field, self.problem)
    }
}

impl std::error::Error for SgdConfigError {}

impl SgdConfig {
    /// Whether the optimizer needs the dense latent causal-link lane.
    ///
    /// In the ordinary one-copy mode, zero feasibility and integrality weights
    /// make the link witnesses semantically inert.  Avoiding that lane removes
    /// its `O(M H^2 F)` logits, Adam state, and forward graph while retaining the
    /// action-only recurrent causal losses.  Non-`Off` modes keep the lane even
    /// at zero explicit weight because their copy/proof protocol owns the causal
    /// witness diagnostics as part of its invariant.
    pub fn causal_links_enabled(&self) -> bool {
        self.causal_link_objective_enabled() || self.backward_bridge_weight > 0.0
    }

    /// Whether global causal-link feasibility participates in the loss.
    /// Backward causal flow reuses the temporal logits but must not pay to
    /// construct an unrelated all-consumer proof on every update.
    pub fn causal_link_objective_enabled(&self) -> bool {
        self.causal_copy != CausalCopyMode::Off
            || self.causal_link_weight > 0.0
            || self.causal_link_integrality_final > 0.0
    }

    /// Validate before anything is allocated. Explicit bounds, never clamping.
    pub fn validate(&self) -> Result<(), SgdConfigError> {
        let bad = |field: &'static str, problem: String| SgdConfigError { field, problem };

        match self.horizon {
            HorizonPolicy::Fixed(horizon) if horizon == 0 => {
                return Err(bad("horizon", "must be at least 1".into()));
            }
            HorizonPolicy::Dovetail { start, growth, max } => {
                if start == 0 {
                    return Err(bad("horizon", "dovetail start must be at least 1".into()));
                }
                if !growth.is_finite() || growth <= 1.0 {
                    return Err(bad(
                        "horizon",
                        format!("dovetail growth must be finite and exceed 1, got {growth}"),
                    ));
                }
                if max < start {
                    return Err(bad(
                        "horizon",
                        format!("dovetail max {max} is below start {start}"),
                    ));
                }
            }
            HorizonPolicy::Fixed(_) => {}
        }

        if self.particles == 0 {
            return Err(bad("particles", "must be at least 1".into()));
        }
        if self.updates == 0 {
            return Err(bad("updates", "must be at least 1".into()));
        }
        if self.verify_period == 0 {
            return Err(bad("verify_period", "must be at least 1".into()));
        }
        if self.trace_period > 0 && self.trace_particle >= self.particles {
            return Err(bad(
                "trace_particle",
                format!(
                    "must be below particles ({}) when tracing is enabled, got {}",
                    self.particles, self.trace_particle
                ),
            ));
        }
        if self.dual_period == 0 {
            return Err(bad("dual_period", "must be at least 1".into()));
        }
        if self.cycles == 0 {
            return Err(bad("cycles", "must be at least 1".into()));
        }
        if self.remelt_patience == 0 {
            return Err(bad("remelt_patience", "must be at least 1".into()));
        }
        if !self.temporal_tokens && self.temporal_restart_patience != 0 {
            return Err(bad(
                "temporal_restart_patience",
                "must be zero when temporal_tokens is false".into(),
            ));
        }
        if self.remelt_cooldown_updates == 0 {
            return Err(bad("remelt_cooldown_updates", "must be at least 1".into()));
        }
        if self.slot_slack_window == 1 {
            return Err(bad(
                "slot_slack_window",
                "must be zero (disabled) or at least 2".into(),
            ));
        }

        // Every float is checked for finiteness and sign. A NaN here would
        // otherwise propagate silently through the gradients and only surface
        // much later as a panic in residual sorting, and a negative `grad_clip`
        // would quietly disable clipping rather than being reported.
        let non_negative: [(&'static str, f64); 33] = [
            ("learning_rate", self.learning_rate),
            ("grad_clip", self.grad_clip),
            ("rho_precondition", self.rho_precondition),
            ("rho_transition", self.rho_transition),
            ("rho_goal", self.rho_goal),
            ("dual_cap", self.dual_cap),
            ("focus_cap", self.focus_cap),
            ("action_integrality_final", self.action_integrality_final),
            ("state_integrality_final", self.state_integrality_final),
            ("worst_integrality_final", self.worst_integrality_final),
            ("q_logit_perturbation", self.q_logit_perturbation),
            ("teacher_tolerance", self.teacher_tolerance),
            ("teacher_weight", self.teacher_weight),
            ("applicability_mass_weight", self.applicability_mass_weight),
            ("noop_suffix_weight", self.noop_suffix_weight),
            ("causal_link_weight", self.causal_link_weight),
            ("causal_link_learning_rate", self.causal_link_learning_rate),
            (
                "causal_link_integrality_final",
                self.causal_link_integrality_final,
            ),
            ("noise_start", self.noise.0),
            ("noise_end", self.noise.1),
            ("remelt_noise", self.remelt_noise),
            ("remelt_shrink", self.remelt_shrink),
            ("initial_noop_logit_gap", self.initial_noop_logit_gap),
            (
                "temporal_reservation_weight",
                self.temporal_reservation_weight,
            ),
            ("causal_link_initial_bias", self.causal_link_initial_bias),
            ("slot_slack_logit_gap", self.slot_slack_logit_gap),
            ("slot_slack_weight", self.slot_slack_weight),
            ("insertion_repair_weight", self.insertion_repair_weight),
            ("anchor_trust_weight", self.anchor_trust_weight),
            ("goal_survival_weight", self.goal_survival_weight),
            ("backward_bridge_weight", self.backward_bridge_weight),
            ("schedule_identity_bias", self.schedule_identity_bias),
            (
                "schedule_integrality_final",
                self.schedule_integrality_final,
            ),
        ];
        for (name, value) in non_negative {
            if !value.is_finite() || value < 0.0 {
                return Err(bad(
                    name,
                    format!("must be finite and non-negative, got {value}"),
                ));
            }
        }
        if self.remelt_shrink >= 1.0 {
            return Err(bad(
                "remelt_shrink",
                format!("must be below 1, got {}", self.remelt_shrink),
            ));
        }
        if self.initial_noop_logit_gap > 2.0 * self.action_logit_clip {
            return Err(bad(
                "initial_noop_logit_gap",
                format!(
                    "must not exceed twice action_logit_clip ({}), got {}",
                    2.0 * self.action_logit_clip,
                    self.initial_noop_logit_gap
                ),
            ));
        }

        let strictly_positive: [(&'static str, f64); 15] = [
            ("action_logit_clip", self.action_logit_clip),
            ("state_logit_clip", self.state_logit_clip),
            ("residual_tolerance", self.residual_tolerance),
            ("action_temperature_start", self.action_temperature.0),
            ("action_temperature_mid", self.action_temperature.1),
            ("action_temperature_end", self.action_temperature.2),
            ("state_temperature_start", self.state_temperature.0),
            ("state_temperature_end", self.state_temperature.1),
            ("schedule_temperature_start", self.schedule_temperature.0),
            ("schedule_temperature_end", self.schedule_temperature.1),
            (
                "causal_link_temperature_start",
                self.causal_link_temperature.0,
            ),
            (
                "causal_link_temperature_end",
                self.causal_link_temperature.1,
            ),
            ("q_action_temperature_start", self.q_action_temperature.0),
            ("q_action_temperature_end", self.q_action_temperature.1),
            ("polish_p_norm", self.polish_p_norm),
        ];
        for (name, value) in strictly_positive {
            if !value.is_finite() || value <= 0.0 {
                return Err(bad(
                    name,
                    format!("must be finite and positive, got {value}"),
                ));
            }
        }

        if self.q_action_temperature.1 > self.q_action_temperature.0 {
            return Err(bad(
                "q_action_temperature_end",
                format!(
                    "must not exceed q_action_temperature_start ({}), got {}",
                    self.q_action_temperature.0, self.q_action_temperature.1
                ),
            ));
        }
        if self.schedule_temperature.1 > self.schedule_temperature.0 {
            return Err(bad(
                "schedule_temperature_end",
                format!(
                    "must not exceed schedule_temperature_start ({}), got {}",
                    self.schedule_temperature.0, self.schedule_temperature.1
                ),
            ));
        }
        if !(1.0..=64.0).contains(&self.polish_p_norm) {
            return Err(bad(
                "polish_p_norm",
                format!("must be in [1, 64], got {}", self.polish_p_norm),
            ));
        }

        // Multipliers that only make sense on one side of one.
        if !self.dual_growth.is_finite() || self.dual_growth < 1.0 {
            return Err(bad(
                "dual_growth",
                format!("must be finite and at least 1, got {}", self.dual_growth),
            ));
        }
        if !self.focus_growth.is_finite() || self.focus_growth <= 1.0 {
            return Err(bad(
                "focus_growth",
                format!("must be finite and exceed 1, got {}", self.focus_growth),
            ));
        }
        for (name, value) in [("dual_cap", self.dual_cap), ("focus_cap", self.focus_cap)] {
            if value <= 1.0 {
                return Err(bad(
                    name,
                    format!("must exceed the baseline 1, got {value}"),
                ));
            }
        }
        if !self.dual_decay.is_finite() || !(0.0..=1.0).contains(&self.dual_decay) {
            return Err(bad(
                "dual_decay",
                format!("must be finite and in [0, 1], got {}", self.dual_decay),
            ));
        }

        if !self.top_residual_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.top_residual_fraction)
        {
            return Err(bad(
                "top_residual_fraction",
                format!("must be in [0, 1], got {}", self.top_residual_fraction),
            ));
        }
        if !self.insertion_min_prefix_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.insertion_min_prefix_fraction)
        {
            return Err(bad(
                "insertion_min_prefix_fraction",
                format!(
                    "must be in [0, 1], got {}",
                    self.insertion_min_prefix_fraction
                ),
            ));
        }
        if !self.crystallization_start.is_finite()
            || !(0.0..=1.0).contains(&self.crystallization_start)
        {
            return Err(bad(
                "crystallization_start",
                format!("must be in [0, 1], got {}", self.crystallization_start),
            ));
        }

        let stages = [
            ("causal_shadow_end", self.causal_shadow_end),
            ("causal_discovery_end", self.causal_discovery_end),
            ("causal_proof_end", self.causal_proof_end),
            ("causal_transfer_end", self.causal_transfer_end),
            ("causal_takeover_end", self.causal_takeover_end),
        ];
        let mut previous = 0.0;
        for (name, value) in stages {
            if !value.is_finite() || value <= previous || value >= 1.0 {
                return Err(bad(
                    name,
                    format!(
                        "must be finite, greater than the preceding boundary {previous}, and below 1, got {value}"
                    ),
                ));
            }
            previous = value;
        }
        if !self.remelt_stop_progress.is_finite()
            || self.remelt_stop_progress <= self.causal_takeover_end
            || self.remelt_stop_progress > 1.0
        {
            return Err(bad(
                "remelt_stop_progress",
                format!(
                    "must exceed causal_takeover_end ({}) and be at most 1, got {}",
                    self.causal_takeover_end, self.remelt_stop_progress
                ),
            ));
        }
        if !self.applicability_barrier_margin.is_finite()
            || !(0.0..1.0).contains(&self.applicability_barrier_margin)
        {
            return Err(bad(
                "applicability_barrier_margin",
                format!(
                    "must be finite and strictly between 0 and 1, got {}",
                    self.applicability_barrier_margin
                ),
            ));
        }

        if self.refresh {
            if self.refresh_particles == 0 {
                return Err(bad(
                    "refresh_particles",
                    "must be at least 1 when refresh is enabled".into(),
                ));
            }
            // Rejected rather than clamped: silently refreshing fewer particles
            // than asked for would make the reported refresh count a lie.
            if self.refresh_particles > self.particles {
                return Err(bad(
                    "refresh_particles",
                    format!(
                        "{} exceeds particles ({})",
                        self.refresh_particles, self.particles
                    ),
                ));
            }
            if self.refresh_period == 0 {
                return Err(bad(
                    "refresh_period",
                    "must be at least 1 when refresh is enabled".into(),
                ));
            }
        }
        if self.temporal_tokens {
            let first_horizon = self
                .horizon
                .horizon_for_round(0)
                .expect("a validated horizon policy has a first round");
            if self.temporal_reserved_slots >= first_horizon {
                return Err(bad(
                    "temporal_reserved_slots",
                    format!(
                        "must be below the first horizon ({first_horizon}), got {}",
                        self.temporal_reserved_slots
                    ),
                ));
            }
            if self.schedule_sinkhorn_iterations == 0 {
                return Err(bad(
                    "schedule_sinkhorn_iterations",
                    "must be at least 1 when temporal tokens are enabled".into(),
                ));
            }
            if self.refresh {
                return Err(bad(
                    "refresh",
                    "temporal-token refresh semantics are not implemented".into(),
                ));
            }
            if self.slot_slack_window != 0
                || self.causal_link_objective_enabled()
                || !matches!(self.causal_copy, CausalCopyMode::Off)
            {
                return Err(bad(
                    "temporal_tokens",
                    "requires slot slack, global causal-link objectives, and causal copy to be disabled".into(),
                ));
            }
        } else if self.temporal_reserved_slots != 0 {
            return Err(bad(
                "temporal_reserved_slots",
                "requires temporal_tokens=true".into(),
            ));
        }
        Ok(())
    }

    /// Per-particle cycle phase at update `update` of `updates`.
    ///
    /// Particle `particle` is offset by `particle / particles` of a cycle, so the
    /// population spreads across the anneal instead of crystallizing in lockstep.
    pub fn phase_at(&self, update: usize, particle: usize) -> f64 {
        let through = update as f64 / self.updates.max(1) as f64;
        let offset = particle as f64 / self.particles.max(1) as f64;
        (through * self.cycles as f64 + offset).fract()
    }

    /// Noise scale at cycle phase `phase`.
    pub fn noise_at(&self, phase: f64) -> f64 {
        let (start, end) = self.noise;
        start + (end - start) * phase
    }

    /// Action temperature at `progress` in `[0, 1]`, interpolating
    /// start → mid → end.
    pub fn action_temperature_at(&self, progress: f64) -> f64 {
        let (start, mid, end) = self.action_temperature;
        if progress < 0.5 {
            start + (mid - start) * (progress / 0.5)
        } else {
            mid + (end - mid) * ((progress - 0.5) / 0.5)
        }
    }

    /// State temperature at `progress` in `[0, 1]`.
    pub fn state_temperature_at(&self, progress: f64) -> f64 {
        let (start, end) = self.state_temperature;
        start + (end - start) * progress
    }

    pub fn schedule_temperature_at(&self, progress: f64) -> f64 {
        let (start, end) = self.schedule_temperature;
        start + (end - start) * progress
    }

    /// Causal-link temperature at one cycle phase.
    pub fn causal_link_temperature_at(&self, progress: f64) -> f64 {
        let (start, end) = self.causal_link_temperature;
        start + (end - start) * progress
    }

    /// Causal-copy action temperature at global update progress.
    pub fn q_action_temperature_at(&self, progress: f64) -> f64 {
        let (start, end) = self.q_action_temperature;
        start + (end - start) * progress
    }

    /// Global causal-copy continuation stage at update progress in `[0, 1]`.
    pub fn causal_stage_at(&self, progress: f64) -> CausalStage {
        assert!(
            progress.is_finite() && (0.0..=1.0).contains(&progress),
            "global update progress must be finite and in [0, 1], got {progress}"
        );
        if progress < self.causal_shadow_end {
            CausalStage::Shadow
        } else if progress < self.causal_discovery_end {
            CausalStage::Discovery
        } else if progress < self.causal_proof_end {
            CausalStage::Proof
        } else if progress < self.causal_transfer_end {
            CausalStage::Transfer
        } else if progress < self.causal_takeover_end {
            CausalStage::Takeover
        } else {
            CausalStage::Polish
        }
    }

    /// Integrality weight scale at `progress`: zero until crystallization
    /// starts, then ramping to one.
    pub fn integrality_scale_at(&self, progress: f64) -> f64 {
        if progress <= self.crystallization_start {
            return 0.0;
        }
        let span = 1.0 - self.crystallization_start;
        if span <= 0.0 {
            return 1.0;
        }
        ((progress - self.crystallization_start) / span).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = SgdConfig::default();
        config.validate().expect("defaults are valid");
        assert_eq!(config.remelt_patience, 120);
        assert_eq!(config.temporal_restart_patience, 0);
        assert_eq!(config.causal_copy, CausalCopyMode::Off);
        assert!(config.causal_links_enabled());
    }

    #[test]
    fn inert_off_mode_causal_links_can_be_omitted_but_copy_modes_keep_them() {
        let mut config = SgdConfig::default();
        config.causal_link_weight = 0.0;
        config.causal_link_integrality_final = 0.0;
        assert!(!config.causal_links_enabled());

        config.causal_copy = CausalCopyMode::Shadow;
        assert!(config.causal_links_enabled());
        config.causal_copy = CausalCopyMode::Staged;
        assert!(config.causal_links_enabled());
    }

    #[test]
    fn invalid_values_are_rejected_with_the_field_named() {
        let mut config = SgdConfig::default();
        config.particles = 0;
        let error = config.validate().expect_err("zero particles is invalid");
        assert_eq!(error.field, "particles");

        let mut config = SgdConfig::default();
        config.remelt_stop_progress = config.causal_takeover_end;
        assert_eq!(
            config
                .validate()
                .expect_err("remelt stop must leave a deterministic tail")
                .field,
            "remelt_stop_progress"
        );

        let mut config = SgdConfig::default();
        config.horizon = HorizonPolicy::Fixed(0);
        assert_eq!(
            config.validate().expect_err("zero horizon").field,
            "horizon"
        );

        let mut config = SgdConfig::default();
        config.top_residual_fraction = 1.5;
        assert_eq!(
            config.validate().expect_err("fraction above one").field,
            "top_residual_fraction"
        );

        let mut config = SgdConfig::default();
        config.action_temperature.2 = 0.0;
        assert_eq!(
            config.validate().expect_err("zero temperature").field,
            "action_temperature_end"
        );

        let mut config = SgdConfig::default();
        config.slot_slack_window = 1;
        assert_eq!(
            config.validate().expect_err("unit slack window").field,
            "slot_slack_window"
        );

        let mut config = SgdConfig::default();
        config.insertion_min_prefix_fraction = 1.01;
        assert_eq!(
            config
                .validate()
                .expect_err("insertion maturity above one")
                .field,
            "insertion_min_prefix_fraction"
        );

        // NaN must be rejected everywhere it could otherwise propagate into the
        // gradients and surface much later as a panic in residual sorting.
        for (name, mutate) in [
            (
                "rho_goal",
                (|c: &mut SgdConfig| c.rho_goal = f64::NAN) as fn(&mut SgdConfig),
            ),
            ("grad_clip", |c: &mut SgdConfig| c.grad_clip = -1.0),
            ("dual_growth", |c: &mut SgdConfig| c.dual_growth = 0.5),
            ("focus_growth", |c: &mut SgdConfig| c.focus_growth = 1.0),
            ("dual_cap", |c: &mut SgdConfig| c.dual_cap = 1.0),
            ("focus_cap", |c: &mut SgdConfig| c.focus_cap = 0.5),
            ("dual_decay", |c: &mut SgdConfig| c.dual_decay = 2.0),
            ("residual_tolerance", |c: &mut SgdConfig| {
                c.residual_tolerance = f64::INFINITY
            }),
            ("noise_start", |c: &mut SgdConfig| c.noise.0 = f64::NAN),
            ("causal_link_weight", |c: &mut SgdConfig| {
                c.causal_link_weight = f64::NAN
            }),
            ("causal_link_learning_rate", |c: &mut SgdConfig| {
                c.causal_link_learning_rate = f64::NAN
            }),
            ("causal_link_temperature_end", |c: &mut SgdConfig| {
                c.causal_link_temperature.1 = 0.0
            }),
            ("teacher_weight", |c: &mut SgdConfig| {
                c.teacher_weight = f64::NAN
            }),
            ("remelt_shrink", |c: &mut SgdConfig| c.remelt_shrink = 1.0),
        ] {
            let mut config = SgdConfig::default();
            mutate(&mut config);
            let error = match config.validate() {
                Err(error) => error,
                Ok(()) => panic!("{name} should be rejected"),
            };
            assert_eq!(error.field, name, "wrong field reported for {name}");
        }

        // A NaN dovetail growth slips past a naive `growth <= 1.0` guard,
        // because every comparison with NaN is false.
        let mut config = SgdConfig::default();
        config.horizon = HorizonPolicy::Dovetail {
            start: 4,
            growth: f64::NAN,
            max: 64,
        };
        assert_eq!(config.validate().expect_err("NaN growth").field, "horizon");

        // Asking to refresh more particles than exist is rejected, not clamped:
        // clamping would make the reported refresh count a lie.
        let mut config = SgdConfig::default();
        config.particles = 2;
        config.refresh = true;
        config.refresh_particles = 100;
        assert_eq!(
            config.validate().expect_err("too many refreshes").field,
            "refresh_particles"
        );
    }

    #[test]
    fn fixed_horizon_yields_exactly_one_round() {
        let policy = HorizonPolicy::Fixed(12);
        assert_eq!(policy.horizon_for_round(0), Some(12));
        assert_eq!(policy.horizon_for_round(1), None);
    }

    #[test]
    fn dovetail_grows_then_stops() {
        let policy = HorizonPolicy::Dovetail {
            start: 4,
            growth: 2.0,
            max: 16,
        };
        assert_eq!(policy.horizon_for_round(0), Some(4));
        assert_eq!(policy.horizon_for_round(1), Some(8));
        assert_eq!(policy.horizon_for_round(2), Some(16));
        assert_eq!(policy.horizon_for_round(3), None);
    }

    #[test]
    fn schedules_move_monotonically_between_their_endpoints() {
        let config = SgdConfig::default();
        assert!((config.action_temperature_at(0.0) - 2.0).abs() < 1e-12);
        assert!((config.action_temperature_at(1.0) - 0.16).abs() < 1e-12);
        assert!((config.state_temperature_at(0.0) - 1.5).abs() < 1e-12);
        assert!((config.state_temperature_at(1.0) - 0.28).abs() < 1e-12);
        assert!((config.causal_link_temperature_at(0.0) - 1.5).abs() < 1e-12);
        assert!((config.causal_link_temperature_at(1.0) - 0.5).abs() < 1e-12);
        assert!((config.q_action_temperature_at(0.0) - 2.5).abs() < 1e-12);
        assert!((config.q_action_temperature_at(1.0) - 0.5).abs() < 1e-12);

        // Integrality stays off through the discovery phase, then ramps.
        assert_eq!(config.integrality_scale_at(0.0), 0.0);
        assert_eq!(config.integrality_scale_at(0.5), 0.0);
        assert!(config.integrality_scale_at(0.8) > 0.0);
        assert!((config.integrality_scale_at(1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn causal_stages_must_be_strictly_ordered() {
        let mut config = SgdConfig::default();
        config.causal_proof_end = config.causal_discovery_end;
        assert_eq!(
            config
                .validate()
                .expect_err("duplicate stage boundary")
                .field,
            "causal_proof_end"
        );
    }

    #[test]
    fn causal_copy_mode_has_a_strict_text_interface() {
        for (text, expected) in [
            ("off", CausalCopyMode::Off),
            ("shadow", CausalCopyMode::Shadow),
            ("staged", CausalCopyMode::Staged),
        ] {
            assert_eq!(text.parse::<CausalCopyMode>().unwrap(), expected);
            assert_eq!(expected.to_string(), text);
        }
        assert!("on".parse::<CausalCopyMode>().is_err());
    }

    #[test]
    fn staged_scalar_domains_are_enforced() {
        let mut config = SgdConfig::default();
        config.applicability_barrier_margin = 1.0;
        assert_eq!(
            config.validate().expect_err("unit barrier margin").field,
            "applicability_barrier_margin"
        );

        let mut config = SgdConfig::default();
        config.polish_p_norm = 0.5;
        assert_eq!(
            config.validate().expect_err("sublinear polish norm").field,
            "polish_p_norm"
        );

        let mut config = SgdConfig::default();
        config.q_action_temperature = (0.5, 1.0);
        assert_eq!(
            config.validate().expect_err("heating Q schedule").field,
            "q_action_temperature_end"
        );
    }

    #[test]
    fn causal_stage_uses_global_ordered_boundaries() {
        let config = SgdConfig::default();
        assert_eq!(config.causal_stage_at(0.0), CausalStage::Shadow);
        assert_eq!(config.causal_stage_at(0.10), CausalStage::Discovery);
        assert_eq!(config.causal_stage_at(0.30), CausalStage::Proof);
        assert_eq!(config.causal_stage_at(0.50), CausalStage::Transfer);
        assert_eq!(config.causal_stage_at(0.70), CausalStage::Takeover);
        assert_eq!(config.causal_stage_at(0.85), CausalStage::Polish);
        assert_eq!(config.causal_stage_at(1.0), CausalStage::Polish);
    }
}
