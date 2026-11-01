//! State evaluation and heuristic implementations.

use planforge_sas::numeric_task::Task;
pub mod abstractions;
pub mod check_admissible;
pub mod evaluator;
pub mod ff_heuristic;
pub mod heuristic;
pub(crate) mod maximal_cliques;
pub mod maximum;
pub mod numeric_landmarks;
pub mod numeric_potentials;
pub(crate) mod sampling;
pub mod seed;
pub(crate) mod state_value_cache;
pub use evaluator::{EvaluationError, EvaluationState};
pub use heuristic::Heuristic;

use crate::heuristic_factory::{HeuristicEntry, entry};
use abstractions::cartesian::CartesianAbstractionCollectionConfig;
use abstractions::domain::config::DomainAbstractionCollectionConfig;
use abstractions::features::{CartesianFeature, DomainFeature, SourceFeature};
use abstractions::icaps26::Icaps26CartesianConfig;
use abstractions::sources::PdbSourceConfig;

/// Every heuristic besides blind search, as a binary registers them.
pub fn heuristics() -> Vec<HeuristicEntry> {
    vec![
        entry::<check_admissible::CheckAdmissibleFeature>(),
        entry::<ff_heuristic::FfFeature>(),
        entry::<numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericFeature>(),
        entry::<numeric_potentials::NumericPotentialFeature>(),
        entry::<DomainFeature>(),
        entry::<CartesianFeature>(),
        entry::<SourceFeature<DomainAbstractionCollectionConfig>>(),
        entry::<SourceFeature<CartesianAbstractionCollectionConfig>>(),
        entry::<SourceFeature<PdbSourceConfig>>(),
        entry::<SourceFeature<Icaps26CartesianConfig>>(),
        entry::<abstractions::cost_partitioning::max::MaxFeature>(),
        entry::<abstractions::cost_partitioning::canonical::CanonicalFeature>(),
        entry::<abstractions::cost_partitioning::saturated::ScpFeature>(),
        entry::<abstractions::cost_partitioning::posthoc_optimization::PostHocFeature>(),
    ]
}

/// Registers [`heuristics`] for a test process.
#[cfg(test)]
pub(crate) fn register_heuristics_for_tests() {
    crate::heuristic_factory::ensure_heuristics_registered(heuristics);
}

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
