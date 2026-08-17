//! State evaluation and heuristic implementations.

pub mod abstraction_collections;
pub(crate) mod abstraction_task;
pub mod cartesian_abstractions;
pub mod cegar;
pub mod check_admissible;
pub mod domain_abstractions;
pub mod evaluator;
pub mod ff_heuristic;
pub mod heuristic;
pub(crate) mod maximal_cliques;
pub mod numeric_landmarks;
#[cfg(feature = "cplex")]
pub mod numeric_potentials;
pub mod pattern_databases;
pub(crate) mod state_value_cache;
pub use evaluator::{EvaluationError, EvaluationState};
pub use heuristic::Heuristic;

use planforge_sas::numeric_task::AbstractNumericTask;

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
pub fn validate_abstractable_goal(
    task: &dyn AbstractNumericTask,
) -> std::result::Result<(), String> {
    for index in 0..task.get_num_goals() {
        let fact = task.get_goal_fact(index);
        if fact.is_condition() {
            continue;
        }
        let variable = &task.variables()[fact.var()];
        if let Some(layer) = variable.axiom_layer() {
            return Err(format!(
                "abstractions support conjunctive goals only: goal fact {fact:?} names the \
                 derived variable {:?} ({:?}) in axiom layer {layer}, which no operator writes",
                variable.name(),
                task.get_fact_name(fact)
            ));
        }
    }
    Ok(())
}
