//! Constraint residuals over plain `f64` buffers.
//!
//! This is the reference semantics of the transcription. It carries no
//! autograd and no tensors, which means the whole formulation — including its
//! exactness at integrality — is testable before any backend exists, and any
//! tensor implementation can be differentially tested against it.
//!
//! # The encoding
//!
//! For horizon `H`, an [`Assignment`] holds
//!
//! * `P[t, a]` for `t` in `0..H`: a distribution over actions per timestep;
//! * `S[t, f]` for `t` in `0..=H`: per finite-domain variable, a distribution
//!   over that variable's values, laid out flat over facts.
//!
//! Because `S[t, ·]` is a distribution *per variable*, "this variable has
//! exactly one value" holds structurally rather than by penalty. That is what
//! makes a relaxed state unable to place one object in two mutually exclusive
//! situations at once, which is the classic failure mode of a per-fact
//! relaxation.
//!
//! Per timestep, action `a`, variable `v`, value `d`, with `φ` the product of an
//! effect's condition truth values at time `t`:
//!
//! ```text
//! w_i         = φ_i · Π_{j>i} (1 − φ_j)     (mass where effect i is the last write)
//! add[a,v,d]  = Σ_{i: effect_value(i)=d} w_i
//! chg[a,v]    = 1 − Π_i (1 − φ_i)
//! Add[t,v,d]  = Σ_a P[t,a] · add[a,v,d]
//! Chg[t,v]    = Σ_a P[t,a] · chg[a,v]
//! D[t,v,d]    = Chg[t,v] − Add[t,v,d]
//! ```
//!
//! Aggregating per operator with those suffix products is load-bearing, not a
//! refinement. Summing effect masses directly double-counts an operator with two
//! effects writing the same variable to the same value, which then reports a
//! nonzero residual on that operator's own true successor. The suffix form also
//! yields the identity `Σ_d add[a,v,d] = chg[a,v]` *exactly*, for fractional `φ`
//! and not merely at integrality, which is what guarantees `D ≥ 0` and hence
//! that the four transition inequalities keep their intended orientation.
//!
//! The four transition residuals per `(t, v, d)` are then the hinges of
//!
//! ```text
//! S[t+1] ≥ Add            S[t+1] ≤ 1 − Chg + Add
//! S[t+1] ≥ S[t] − D       S[t+1] ≤ S[t] + Add
//! ```
//!
//! At one-hot `P` and `S` rows these are satisfiable exactly when `S[t+1]` is
//! the true successor of `S[t]` under the selected action.

use crate::transcription::Transcription;

/// A relaxed action/state assignment over a fixed horizon.
#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    horizon: usize,
    num_actions: usize,
    num_facts: usize,
    /// Row-major `[H, num_actions]`.
    action: Vec<f64>,
    /// Row-major `[H + 1, num_facts]`.
    state: Vec<f64>,
}

impl Assignment {
    /// An all-zero assignment. Callers fill it; `zeros` is not a valid
    /// distribution and is only a starting buffer.
    pub fn zeros(transcription: &Transcription, horizon: usize) -> Self {
        let num_actions = transcription.num_actions();
        let num_facts = transcription.num_facts();
        Self {
            horizon,
            num_actions,
            num_facts,
            action: vec![0.0; horizon * num_actions],
            state: vec![0.0; (horizon + 1) * num_facts],
        }
    }

    pub fn horizon(&self) -> usize {
        self.horizon
    }

    pub fn action_row(&self, t: usize) -> &[f64] {
        &self.action[t * self.num_actions..(t + 1) * self.num_actions]
    }

    pub fn action_row_mut(&mut self, t: usize) -> &mut [f64] {
        &mut self.action[t * self.num_actions..(t + 1) * self.num_actions]
    }

    pub fn state_row(&self, t: usize) -> &[f64] {
        &self.state[t * self.num_facts..(t + 1) * self.num_facts]
    }

    pub fn state_row_mut(&mut self, t: usize) -> &mut [f64] {
        &mut self.state[t * self.num_facts..(t + 1) * self.num_facts]
    }

    pub fn action(&self, t: usize, a: usize) -> f64 {
        self.action[t * self.num_actions + a]
    }

    pub fn state(&self, t: usize, fact: usize) -> f64 {
        self.state[t * self.num_facts + fact]
    }

    /// Set action row `t` to the one-hot vector selecting `action`.
    pub fn set_action_one_hot(&mut self, t: usize, action: usize) {
        let row = self.action_row_mut(t);
        row.fill(0.0);
        row[action] = 1.0;
    }

    /// Set state row `t` so that each variable takes the value given by
    /// `values`, indexed by transcription variable.
    pub fn set_state_one_hot(&mut self, transcription: &Transcription, t: usize, values: &[usize]) {
        assert_eq!(
            values.len(),
            transcription.num_variables(),
            "one value per transcription variable"
        );
        let row = self.state_row_mut(t);
        row.fill(0.0);
        for (var, &value) in values.iter().enumerate() {
            row[transcription.fact(var, value) as usize] = 1.0;
        }
    }
}

/// Effect firing masses and their per-variable aggregates at one timestep.
///
/// Reuse one instance across timesteps: it owns the scratch space too, so the
/// per-update work allocates nothing.
#[derive(Debug, Clone)]
pub struct Masses {
    /// `Add[t, fact]`, indexed by flat fact.
    pub add: Vec<f64>,
    /// `Chg[t, var]`, indexed by transcription variable.
    pub chg: Vec<f64>,
    /// Per-effect condition product `φ`. An effect with no conditions has 1.
    phi: Vec<f64>,
}

impl Masses {
    pub fn zeros(transcription: &Transcription) -> Self {
        Self {
            add: vec![0.0; transcription.num_facts()],
            chg: vec![0.0; transcription.num_variables()],
            phi: vec![1.0; transcription.num_effects()],
        }
    }

    /// `D[t, fact] = Chg[t, var(fact)] − Add[t, fact]`.
    pub fn delete(&self, transcription: &Transcription, fact: usize) -> f64 {
        self.chg[transcription.var_of_fact()[fact] as usize] - self.add[fact]
    }
}

/// Compute [`Masses`] for timestep `t` into `out`.
pub fn masses_at(
    transcription: &Transcription,
    assignment: &Assignment,
    t: usize,
    out: &mut Masses,
) {
    out.add.fill(0.0);
    out.chg.fill(0.0);
    out.phi.fill(1.0);

    // Condition products per effect.
    let state = assignment.state_row(t);
    for (index, &effect) in transcription.cond_effect().iter().enumerate() {
        let fact = transcription.cond_fact()[index] as usize;
        out.phi[effect as usize] *= state[fact];
    }
    let phi = &out.phi;

    for group in 0..transcription.num_groups() {
        let action = transcription.group_action()[group] as usize;
        let var = transcription.group_var()[group] as usize;
        let action_mass = assignment.action(t, action);

        // Walk the group's effects from last to first, accumulating the
        // probability that no later effect fires.
        let mut no_later_write = 1.0f64;
        let effects = transcription.group_effects(group);
        for &effect in effects.iter().rev() {
            let effect = effect as usize;
            let last_write_mass = phi[effect] * no_later_write;
            let fact = transcription.effect_fact()[effect] as usize;
            out.add[fact] += action_mass * last_write_mass;
            no_later_write *= 1.0 - phi[effect];
        }
        out.chg[var] += action_mass * (1.0 - no_later_write);
    }
}

/// Residuals of one assignment. Every entry is non-negative, and every entry
/// being zero at an integral assignment means the assignment is a valid plan.
#[derive(Debug, Clone, PartialEq)]
pub struct Residuals {
    /// One per `(timestep, precondition incidence)`, row-major over timesteps.
    pub precondition: Vec<f64>,
    /// The four transition families, each one per `(timestep, fact)`.
    pub transition: [Vec<f64>; 4],
    /// One per goal fact, at the terminal row.
    pub goal: Vec<f64>,
}

impl Residuals {
    /// Largest residual over all families, i.e. the infinity norm.
    pub fn max(&self) -> f64 {
        let families = std::iter::once(&self.precondition)
            .chain(self.transition.iter())
            .chain(std::iter::once(&self.goal));
        families
            .flat_map(|family| family.iter().copied())
            .fold(0.0f64, f64::max)
    }

    /// Sum of all residuals.
    pub fn total(&self) -> f64 {
        let families = std::iter::once(&self.precondition)
            .chain(self.transition.iter())
            .chain(std::iter::once(&self.goal));
        families.flat_map(|family| family.iter().copied()).sum()
    }

    pub fn is_zero(&self, tolerance: f64) -> bool {
        self.max() <= tolerance
    }
}

#[inline]
fn hinge(value: f64) -> f64 {
    value.max(0.0)
}

/// Evaluate every residual family for `assignment`.
pub fn evaluate(transcription: &Transcription, assignment: &Assignment) -> Residuals {
    let horizon = assignment.horizon();
    let num_facts = transcription.num_facts();
    let num_pre = transcription.pre_action().len();

    let mut precondition = vec![0.0; horizon * num_pre];
    let mut transition = [
        vec![0.0; horizon * num_facts],
        vec![0.0; horizon * num_facts],
        vec![0.0; horizon * num_facts],
        vec![0.0; horizon * num_facts],
    ];
    let mut masses = Masses::zeros(transcription);

    for t in 0..horizon {
        let current = assignment.state_row(t);
        let next = assignment.state_row(t + 1);

        for index in 0..num_pre {
            let action = transcription.pre_action()[index] as usize;
            let fact = transcription.pre_fact()[index] as usize;
            precondition[t * num_pre + index] = hinge(assignment.action(t, action) - current[fact]);
        }

        masses_at(transcription, assignment, t, &mut masses);
        for fact in 0..num_facts {
            let add = masses.add[fact];
            let delete = masses.delete(transcription, fact);
            let slot = t * num_facts + fact;
            transition[0][slot] = hinge(add - next[fact]);
            transition[1][slot] = hinge(next[fact] - (1.0 - delete));
            transition[2][slot] = hinge(current[fact] - delete - next[fact]);
            transition[3][slot] = hinge(next[fact] - current[fact] - add);
        }
    }

    let terminal = assignment.state_row(horizon);
    let goal = transcription
        .goal_facts()
        .iter()
        .map(|&fact| 1.0 - terminal[fact as usize])
        .collect();

    Residuals {
        precondition,
        transition,
        goal,
    }
}

/// Mean action-integrality penalty, `1 − Σ_a P[t,a]²`, zero exactly on one-hot
/// action rows.
pub fn action_integrality(assignment: &Assignment) -> f64 {
    let horizon = assignment.horizon();
    if horizon == 0 {
        return 0.0;
    }
    let total: f64 = (0..horizon)
        .map(|t| {
            let sum_squares: f64 = assignment.action_row(t).iter().map(|p| p * p).sum();
            1.0 - sum_squares
        })
        .sum();
    total / horizon as f64
}

/// Mean state-integrality penalty, `1 − Σ_d S[t,v,d]²` per variable, zero
/// exactly on one-hot state rows. Rows `1..=H` only: row 0 is the fixed initial
/// state and is integral by construction.
pub fn state_integrality(transcription: &Transcription, assignment: &Assignment) -> f64 {
    let horizon = assignment.horizon();
    let num_variables = transcription.num_variables();
    if horizon == 0 || num_variables == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for t in 1..=horizon {
        let row = assignment.state_row(t);
        for var in 0..num_variables {
            let offset = transcription.var_offset()[var] as usize;
            let size = transcription.var_domain()[var] as usize;
            let sum_squares: f64 = row[offset..offset + size].iter().map(|s| s * s).sum();
            total += 1.0 - sum_squares;
        }
    }
    total / (horizon * num_variables) as f64
}
