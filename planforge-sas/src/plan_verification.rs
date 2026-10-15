//! Exact replay of a fixed operator sequence.
//!
//! This is deliberately *not* search: it applies one given sequence and reports
//! what happened. There is no frontier, no successor enumeration, and no choice
//! of which operator to try. That makes it usable both as a general plan
//! validator (the task language is covered in full, because
//! [`StateRegistry`] runs the axiom evaluator and the numeric effects itself)
//! and as a separation oracle for optimization-based planners that need to know
//! *which* literal broke a candidate sequence.
//!
//! Applicability is decided by [`Operator::preconditions`] alone. That is
//! exactly the test the search uses: `SuccessorTree` consults only
//! `preconditions()`, and the SAS parser has already hoisted every effect
//! `precondition_value` into the operator's preconditions. Compiled numeric
//! conditions are covered too, since they arrive as ordinary preconditions on
//! comparison-axiom-derived variables.

#[cfg(test)]
mod tests;

use crate::numeric_task::{AbstractNumericTask, ExplicitFact, NumericValue, Operator};
use crate::state_registry::{ConcreteState, StateRegistry};
use crate::utils::errors::StateInsertError;

/// Why a replayed operator sequence is not a valid plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanRejection {
    /// The operator at `step` (0-based) could not be applied because `fact`
    /// did not hold in the state reached after `step` operators.
    InapplicableOperator {
        step: usize,
        operator: String,
        fact: ExplicitFact,
    },
    /// Every operator applied, but no visited state satisfied the goal.
    GoalNotReached { unsatisfied: Vec<ExplicitFact> },
    /// The task's global constraint failed in the state reached after `step`
    /// operators. Kept separate from a precondition failure on purpose: the
    /// search never checks the global constraint at all, so if this ever fires
    /// on a plan another engine produced, it is a real soundness bug.
    GlobalConstraintViolated { step: usize },
}

/// A goal-reaching prefix of the replayed sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedPlan {
    /// Number of leading operators that form the plan. The goal holds in the
    /// state reached after exactly this many operators, and did not hold in any
    /// earlier state.
    pub prefix_len: usize,
    /// Accumulated transition cost of that prefix, i.e. the `g`-value the
    /// search would assign to the goal state.
    pub cost: f64,
}

/// Outcome of replaying a sequence.
#[derive(Debug, Clone, PartialEq)]
pub enum ReplayOutcome {
    /// The first goal-reaching prefix.
    Solved(VerifiedPlan),
    Rejected(PlanRejection),
}

/// Result of a replay, including the states visited along the way.
#[derive(Debug, Clone)]
pub struct Replay {
    pub outcome: ReplayOutcome,
    /// States actually reached: `states[i]` is the state after `i` operators,
    /// so `states[0]` is always the initial state. On rejection this stops at
    /// the last state reached, which is what a caller needs in order to
    /// attribute the failure.
    pub states: Vec<ConcreteState>,
    /// Number of operators successfully applied (`states.len() - 1`).
    pub applied: usize,
}

impl Replay {
    /// The verified plan, or `None` if the sequence was rejected.
    pub fn verified(&self) -> Option<&VerifiedPlan> {
        match &self.outcome {
            ReplayOutcome::Solved(plan) => Some(plan),
            ReplayOutcome::Rejected(_) => None,
        }
    }

    pub fn is_solved(&self) -> bool {
        matches!(self.outcome, ReplayOutcome::Solved(_))
    }
}

/// Goal facts that do not hold in `state`.
fn unsatisfied_goals<T: AbstractNumericTask + ?Sized>(
    task: &T,
    state: crate::state_registry::ConcreteStateView<'_>,
) -> Vec<ExplicitFact> {
    (0..task.get_num_goals())
        .map(|i| task.get_goal_fact(i))
        .filter(|fact| !fact.is_held(state))
        .copied()
        .collect()
}

/// Replay `operators` from the task's initial state under exact semantics.
///
/// Stops at the *first* state satisfying the goal and reports that prefix, so a
/// sequence padded beyond the goal still verifies. `global_constraint` is
/// checked in every visited state, including the initial one.
///
/// Errors are reserved for genuine failures of the state machinery (numeric
/// evaluation, axiom evaluation); an invalid *plan* is a `Rejected` outcome,
/// not an `Err`.
pub fn replay_plan<T: AbstractNumericTask + ?Sized>(
    task: &T,
    registry: &mut StateRegistry<'_>,
    global_constraint: &ExplicitFact,
    operators: &[&Operator],
) -> Result<Replay, StateInsertError> {
    let initial = registry.get_initial_state();
    let mut states = vec![initial];
    let mut cost = 0.0;

    // Scratch buffers reused across the whole replay.
    let mut numeric_values: Vec<NumericValue> = Vec::new();
    let mut cost_values: Vec<NumericValue> = Vec::new();

    for step in 0..=operators.len() {
        let current = &states[step];

        if !global_constraint.is_held(registry.view(current)) {
            return Ok(Replay {
                outcome: ReplayOutcome::Rejected(PlanRejection::GlobalConstraintViolated { step }),
                applied: states.len() - 1,
                states,
            });
        }

        if unsatisfied_goals(task, registry.view(current)).is_empty() {
            return Ok(Replay {
                outcome: ReplayOutcome::Solved(VerifiedPlan {
                    prefix_len: step,
                    cost,
                }),
                applied: states.len() - 1,
                states,
            });
        }

        // Not a goal state; apply the next operator if there is one.
        let Some(operator) = operators.get(step) else {
            break;
        };

        if let Some(fact) = operator
            .preconditions()
            .iter()
            .find(|fact| !fact.is_held(registry.view(current)))
        {
            return Ok(Replay {
                outcome: ReplayOutcome::Rejected(PlanRejection::InapplicableOperator {
                    step,
                    operator: operator.name().to_string(),
                    fact: *fact,
                }),
                applied: states.len() - 1,
                states,
            });
        }

        let (successor, op_cost) = registry.get_successor_state_with_buffers_and_cost(
            current,
            operator,
            &mut numeric_values,
            &mut cost_values,
        )?;
        cost += op_cost.value();
        states.push(successor);
    }

    let unsatisfied = unsatisfied_goals(task, registry.view(&states[operators.len()]));
    assert!(
        !unsatisfied.is_empty(),
        "goal-reaching prefix should have been reported inside the loop"
    );
    Ok(Replay {
        outcome: ReplayOutcome::Rejected(PlanRejection::GoalNotReached { unsatisfied }),
        applied: states.len() - 1,
        states,
    })
}
