//! What every heuristic builds on: the [`Heuristic`] trait, the state it
//! evaluates, and the shared helpers -- random walks, seeds and a per-state
//! value cache. The heuristics themselves live in `planforge_heuristics`.

use planforge_sas::numeric_task::Task;
pub mod evaluator;
pub mod heuristic;
pub mod sampling;
pub mod seed;
pub mod state_value_cache;
pub use evaluator::{EvaluationError, EvaluationState};
pub use heuristic::Heuristic;

/// Rejects a task whose goal an abstraction cannot express, naming the variable
/// that makes it one.
///
/// Every abstraction family here supports a *conjunctive* goal: a set of facts an
/// abstract operator can bring about. A propositional variable an axiom writes is
/// not one. No abstract operator ever writes it — abstract operators come from the
/// task's operators, and a derived variable has no operator writing it either — so
/// an abstraction over such a goal has nothing to reach, and reports the goal
/// unreachable or drops it. Both answers have been wrong in this tree before, and
/// both are silent, so the family refuses the task by name instead.
///
/// A numeric comparison is *not* affected, which is the point of the distinction:
/// its variable is derived in the SAS sense too, but it is
/// [`FactNamespace::Condition`](planforge_sas::numeric_task::FactNamespace::Condition),
/// and the abstraction machinery reasons about the comparison itself — refining
/// its operands is what the numeric CEGAR loop does all day.
///
/// What is left, then, is a `:derived` predicate the problem puts in its goal, and
/// the `@goal-reachable` predicate the translator invents for a disjunctive,
/// quantified or nested goal. Every non-abstraction configuration — blind, `ff`,
/// `lmcutnumeric` — still solves both, since it tests the goal fact in a state the
/// axiom evaluator has closed.
pub fn validate_abstractable_goal(task: &Task) -> std::result::Result<(), String> {
    for fact in task.goals() {
        if fact.is_condition() {
            continue;
        }
        let variable = &task.variables()[fact.var()];
        if let Some(layer) = variable.axiom_layer() {
            return Err(format!(
                "abstractions support conjunctive goals only: goal fact {fact:?} names the \
                 derived variable {:?} ({:?}) in axiom layer {layer}, which no operator writes",
                variable.name(),
                task.fact_name(fact)
            ));
        }
    }
    Ok(())
}
