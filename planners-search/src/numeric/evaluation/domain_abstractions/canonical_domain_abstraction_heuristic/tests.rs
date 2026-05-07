use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::{
    ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
};
use planners_sas::numeric::state_registry::StateRegistry;
use planners_sas::numeric::utils::int_packer::IntDoublePacker;

use crate::numeric::evaluation::domain_abstractions::transition_system::TransitionSystem;
use crate::numeric::evaluation::evaluator::EvaluationState;

use super::super::domain_abstraction::NumericPartitions;
use super::super::domain_abstraction_factory::{AbstractDistanceTable, DomainAbstractionFactory};
use super::*;

fn simple_var(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        None,
        1,
    )
}

fn simple_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("p"), simple_var("q")],
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0, 0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn make_abstraction(task: &NumericRootTask, distances: Vec<f64>) -> DomainAbstraction {
    let factory = DomainAbstractionFactory::new(
        task,
        vec![vec![0, 1], vec![0, 1]],
        vec![2, 2],
        NumericPartitions::with_partitions(vec![]),
        vec![],
    )
    .unwrap();
    let hash_multipliers = TransitionSystem::compute_hash_multipliers(
        factory.domain_sizes(),
        factory.numeric_domain_sizes(),
    )
    .unwrap();

    DomainAbstraction {
        factory,
        distance_table: AbstractDistanceTable {
            distances,
            generating_op_ids: vec![None; 4],
            initial_state_hash: 0,
            goal_facts: vec![],
            hash_multipliers: hash_multipliers.clone(),
            numeric_domain_sizes: vec![],
        },
        combine_labels: false,
    }
}

#[test]
fn computes_max_additive_subsets_from_relevant_operators() {
    let subsets = compute_max_additive_subsets_from_relevant_operators(&[
        [0usize].into_iter().collect(),
        [1usize].into_iter().collect(),
        [0usize].into_iter().collect(),
    ]);

    assert_eq!(subsets, vec![vec![0, 1], vec![1, 2]]);
}

#[test]
fn canonical_domain_abstraction_uses_explicit_subsets() {
    let task = simple_task();
    let packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &packer);
    let mut registry = StateRegistry::new(&task, &packer, &axiom_evaluator);
    let initial_state = registry.get_initial_state();

    let heuristic = CanonicalDomainAbstractionHeuristic::with_explicit_subsets(
        None,
        vec![
            DomainAbstractionHeuristic::new(
                Some("da0".to_string()),
                make_abstraction(&task, vec![2.0, 0.0, 0.0, 0.0]),
            ),
            DomainAbstractionHeuristic::new(
                Some("da1".to_string()),
                make_abstraction(&task, vec![3.0, 0.0, 0.0, 0.0]),
            ),
            DomainAbstractionHeuristic::new(
                Some("da2".to_string()),
                make_abstraction(&task, vec![4.0, 0.0, 0.0, 0.0]),
            ),
        ],
        vec![vec![0, 1], vec![2]],
    );

    let eval_state =
        EvaluationState::new_with_registry(&initial_state, 0.0, false, &task, &registry);
    let value = heuristic.compute_heuristic(&eval_state).unwrap();

    assert_eq!(value, 5.0);
}
