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

use crate::numeric_task::{AbstractNumericTask, ExplicitFact, Operator};
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
    state: &ConcreteState,
    registry: &StateRegistry<'_>,
) -> Vec<ExplicitFact> {
    (0..task.get_num_goals())
        .map(|i| task.get_goal_fact(i))
        .filter(|fact| !fact.is_hold(state, registry))
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
    let mut numeric_values: Vec<f64> = Vec::new();
    let mut cost_values: Vec<f64> = Vec::new();

    for step in 0..=operators.len() {
        let current = &states[step];

        if !global_constraint.is_hold(current, registry) {
            return Ok(Replay {
                outcome: ReplayOutcome::Rejected(PlanRejection::GlobalConstraintViolated { step }),
                applied: states.len() - 1,
                states,
            });
        }

        if unsatisfied_goals(task, current, registry).is_empty() {
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
            .find(|fact| !fact.is_hold(current, registry))
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
        cost += op_cost;
        states.push(successor);
    }

    let unsatisfied = unsatisfied_goals(task, &states[operators.len()], registry);
    debug_assert!(
        !unsatisfied.is_empty(),
        "goal-reaching prefix should have been reported inside the loop"
    );
    Ok(Replay {
        outcome: ReplayOutcome::Rejected(PlanRejection::GoalNotReached { unsatisfied }),
        applied: states.len() - 1,
        states,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axioms::PropositionalAxiom;
    use crate::numeric_task::{Effect, ExplicitVariable, Metric, NumericRootTask};
    use std::sync::Arc;

    /// A three-position chain task.
    ///
    /// * `var0` — derived global-constraint atom, true (value 0) via an
    ///   unconditional axiom, mirroring the translator's `new-axiom@0()`.
    /// * `var1` — position, domain 3, initially 0, goal 2.
    ///
    /// Operators: `move_ab` (0→1), `move_bc` (1→2), `reset` (2→0). `reset`
    /// exists so a sequence can be padded past the goal.
    fn chain_task() -> NumericRootTask {
        chain_task_with(ExplicitFact::new(0, 0), 0)
    }

    /// `chain_task`, parameterized by the global constraint and the initial
    /// position, so tests can vary exactly one thing.
    fn chain_task_with(
        global_constraint: ExplicitFact,
        initial_position: usize,
    ) -> NumericRootTask {
        let variables = vec![
            ExplicitVariable::new(
                2,
                String::from("var0"),
                vec![String::from("gc"), String::from("not-gc")],
                Some(0),
                1,
            ),
            ExplicitVariable::new(
                3,
                String::from("var1"),
                vec![
                    String::from("at(a)"),
                    String::from("at(b)"),
                    String::from("at(c)"),
                ],
                None,
                0,
            ),
        ];
        let operators = vec![
            Operator::new(
                String::from("move_ab"),
                vec![ExplicitFact::new(1, 0)],
                vec![Effect::new(Vec::new(), 1, Some(0), 1)],
                Vec::new(),
                1,
            ),
            Operator::new(
                String::from("move_bc"),
                vec![ExplicitFact::new(1, 1)],
                vec![Effect::new(Vec::new(), 1, Some(1), 2)],
                Vec::new(),
                1,
            ),
            Operator::new(
                String::from("reset"),
                vec![ExplicitFact::new(1, 2)],
                vec![Effect::new(Vec::new(), 1, Some(2), 0)],
                Vec::new(),
                1,
            ),
        ];
        NumericRootTask::new(
            4,
            // No metric variable, so every transition costs 1.0.
            Metric::new(true, None),
            variables,
            Vec::new(),
            vec![ExplicitFact::new(1, 2)],
            Vec::new(),
            vec![1, initial_position],
            Vec::new(),
            operators,
            // Unconditionally derive var0 = 0 ("gc" holds).
            vec![PropositionalAxiom::new(Vec::new(), 0, 1, 0)],
            Vec::new(),
            Vec::new(),
            global_constraint,
        )
    }

    /// Replay `names` against a fresh registry over `task`.
    fn replay(task: NumericRootTask, names: &[&str]) -> Replay {
        let arc: Arc<NumericRootTask> = Arc::new(task);
        let mut registry = StateRegistry::for_task(arc.clone());
        let operators: Vec<&Operator> = names
            .iter()
            .map(|name| {
                arc.get_operators()
                    .iter()
                    .find(|op| op.name() == *name)
                    .unwrap_or_else(|| panic!("no operator named {name}"))
            })
            .collect();
        replay_plan(&*arc, &mut registry, arc.global_constraint(), &operators)
            .expect("replay machinery failed")
    }

    #[test]
    fn valid_plan_is_verified_with_its_cost() {
        let result = replay(chain_task(), &["move_ab", "move_bc"]);
        assert_eq!(
            result.outcome,
            ReplayOutcome::Solved(VerifiedPlan {
                prefix_len: 2,
                cost: 2.0,
            })
        );
        // states[0] initial, states[1] after move_ab, states[2] after move_bc.
        assert_eq!(result.states.len(), 3);
        assert_eq!(result.applied, 2);
    }

    #[test]
    fn padding_past_the_goal_returns_the_first_goal_reaching_prefix() {
        let result = replay(chain_task(), &["move_ab", "move_bc", "reset", "move_ab"]);
        // `reset` would leave the goal, but verification stops before it.
        assert_eq!(
            result.outcome,
            ReplayOutcome::Solved(VerifiedPlan {
                prefix_len: 2,
                cost: 2.0,
            })
        );
        assert_eq!(result.applied, 2, "must not apply operators past the goal");
    }

    #[test]
    fn inapplicable_operator_reports_step_and_offending_fact() {
        let result = replay(chain_task(), &["move_bc"]);
        assert_eq!(
            result.outcome,
            ReplayOutcome::Rejected(PlanRejection::InapplicableOperator {
                step: 0,
                operator: String::from("move_bc"),
                fact: ExplicitFact::new(1, 1),
            })
        );
        assert!(!result.is_solved());
        assert_eq!(result.applied, 0);
    }

    #[test]
    fn inapplicable_operator_mid_sequence_reports_the_earliest_failure() {
        // `move_ab` applies, then `move_ab` again does not.
        let result = replay(chain_task(), &["move_ab", "move_ab", "move_bc"]);
        assert_eq!(
            result.outcome,
            ReplayOutcome::Rejected(PlanRejection::InapplicableOperator {
                step: 1,
                operator: String::from("move_ab"),
                fact: ExplicitFact::new(1, 0),
            })
        );
        assert_eq!(result.applied, 1);
    }

    #[test]
    fn applicable_sequence_missing_the_goal_is_rejected() {
        let result = replay(chain_task(), &["move_ab"]);
        assert_eq!(
            result.outcome,
            ReplayOutcome::Rejected(PlanRejection::GoalNotReached {
                unsatisfied: vec![ExplicitFact::new(1, 2)],
            })
        );
        assert_eq!(result.applied, 1);
    }

    #[test]
    fn empty_sequence_is_rejected_when_the_initial_state_is_not_a_goal() {
        let result = replay(chain_task(), &[]);
        assert_eq!(
            result.outcome,
            ReplayOutcome::Rejected(PlanRejection::GoalNotReached {
                unsatisfied: vec![ExplicitFact::new(1, 2)],
            })
        );
        assert_eq!(result.states.len(), 1, "only the initial state is visited");
    }

    #[test]
    fn empty_sequence_verifies_when_the_initial_state_is_already_a_goal() {
        // Start at the goal position.
        let result = replay(chain_task_with(ExplicitFact::new(0, 0), 2), &[]);
        assert_eq!(
            result.outcome,
            ReplayOutcome::Solved(VerifiedPlan {
                prefix_len: 0,
                cost: 0.0,
            })
        );
    }

    #[test]
    fn violated_global_constraint_is_reported_separately() {
        // Demand the *negation* of the derived atom, which the axiom never
        // produces, so the constraint fails immediately in the initial state.
        let task = chain_task_with(ExplicitFact::new(0, 1), 0);
        let result = replay(task, &["move_ab", "move_bc"]);
        assert_eq!(
            result.outcome,
            ReplayOutcome::Rejected(PlanRejection::GlobalConstraintViolated { step: 0 })
        );
    }
}
