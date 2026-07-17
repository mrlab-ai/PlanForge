use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, Operator,
};
use planforge_sas::state_registry::StateRegistry;

use crate::evaluation::abstraction_collections::max_heuristic::MaxAbstractionHeuristic;
use crate::evaluation::abstraction_task::AbstractionUse;
use crate::evaluation::cartesian_abstractions::{
    CartesianAbstractionConfig, CartesianAbstractionGenerator,
};
use crate::evaluation::evaluator::EvaluationState;
use crate::evaluation::pattern_databases::pattern_database::PatternDatabase;
use crate::evaluation::pattern_databases::projected_task::{Pattern, ProjectedTask};

use super::*;
use crate::evaluation::domain_abstractions::domain_abstraction::NumericPartitions;
use crate::evaluation::domain_abstractions::domain_abstraction_factory::{
    AbstractDistanceTable, DomainAbstractionFactory,
};
use crate::evaluation::domain_abstractions::domain_abstraction_generator::{
    DomainAbstraction, compute_hash_multipliers,
};

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
        vec![ExplicitFact::new(0, 1), ExplicitFact::new(1, 1)],
        vec![],
        vec![0, 0],
        vec![],
        vec![
            Operator::new(
                "set-p".to_string(),
                vec![],
                vec![Effect::new(vec![], 0, Some(0), 1)],
                vec![],
                2,
            ),
            Operator::new(
                "set-q".to_string(),
                vec![],
                vec![Effect::new(vec![], 1, Some(0), 1)],
                vec![],
                3,
            ),
        ],
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
    let hash_multipliers =
        compute_hash_multipliers(factory.domain_sizes(), factory.numeric_domain_sizes()).unwrap();

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
        hash_multipliers,
        combine_labels: false,
        relevant_operator_ids: Vec::new(),
        abstract_operators: Vec::new(),
        abstract_operator_footprints: Vec::new(),
        metadata: Default::default(),
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
    let mut registry = StateRegistry::for_task(std::sync::Arc::new(&task));
    let initial_state = registry.get_initial_state();
    let mut da0 = make_abstraction(&task, vec![2.0, 0.0, 0.0, 0.0]);
    da0.relevant_operator_ids = vec![0];
    let mut da1 = make_abstraction(&task, vec![3.0, 0.0, 0.0, 0.0]);
    da1.relevant_operator_ids = vec![1];
    let mut da2 = make_abstraction(&task, vec![4.0, 0.0, 0.0, 0.0]);
    da2.relevant_operator_ids = vec![0, 1];

    let heuristic = CanonicalAbstractionHeuristic::with_explicit_subsets(
        None,
        &task,
        vec![
            AbstractionComponent::domain(Some("da0".to_string()), da0),
            AbstractionComponent::domain(Some("da1".to_string()), da1),
            AbstractionComponent::domain(Some("da2".to_string()), da2),
        ],
        vec![vec![0, 1], vec![2]],
    )
    .unwrap();

    let eval_state =
        EvaluationState::new_with_registry(&initial_state, 0.0, false, &task, &registry);
    let value = heuristic.compute_heuristic(&eval_state).unwrap();

    assert_eq!(value, 5.0);
}

fn mixed_components<'task>(task: &'task NumericRootTask) -> Vec<AbstractionComponent<'task>> {
    let mut domain = make_abstraction(task, vec![2.0, 0.0, 2.0, 0.0]);
    domain.relevant_operator_ids = vec![0];

    let cartesian = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        max_states: 16,
        max_time: None,
        combine_labels: false,
        compute_operator_footprints: true,
        random_seed: None,
        debug: false,
    })
    .unwrap()
    .generate(task)
    .unwrap();

    let pattern = Pattern::new(vec![1], vec![]);
    let projected = ProjectedTask::new(task, &pattern).unwrap();
    let pdb = PatternDatabase::new(projected, 32).unwrap();

    vec![
        AbstractionComponent::domain(Some("domain".to_string()), domain),
        AbstractionComponent::cartesian(Some("Cartesian".to_string()), cartesian),
        AbstractionComponent::pattern_database(pdb),
    ]
}

#[test]
fn mixed_domain_cartesian_and_pdb_components_work_in_max_and_canonical() {
    let task = simple_task();
    let mut registry = StateRegistry::for_task(std::sync::Arc::new(&task));
    let initial_state = registry.get_initial_state();
    let eval_state =
        EvaluationState::new_with_registry(&initial_state, 0.0, false, &task, &registry);

    let max = MaxAbstractionHeuristic::new(None, mixed_components(&task)).unwrap();
    assert_eq!(max.compute_heuristic(&eval_state).unwrap(), 5.0);

    let canonical =
        CanonicalAbstractionHeuristic::new(None, &task, mixed_components(&task)).unwrap();
    assert!(
        canonical
            .max_additive_subsets()
            .iter()
            .any(|subset| subset == &[0, 2])
    );
    assert_eq!(canonical.compute_heuristic(&eval_state).unwrap(), 5.0);
}

#[test]
fn collection_combinators_never_claim_the_standalone_initial_optimality_proof() {
    let task = simple_task();
    let mut domain = make_abstraction(&task, vec![5.0, 0.0, 0.0, 0.0]);
    domain.relevant_operator_ids = vec![0];
    domain.metadata.solved_by_self = true;
    domain.metadata.abstraction_use = AbstractionUse::Standalone;
    let component = AbstractionComponent::domain(None, domain);
    assert!(component.proves_initial_state_optimal());
    let max = MaxAbstractionHeuristic::new(None, vec![component]).unwrap();
    assert!(!max.proves_initial_state_optimal());

    let mut domain = make_abstraction(&task, vec![5.0, 0.0, 0.0, 0.0]);
    domain.relevant_operator_ids = vec![0];
    domain.metadata.solved_by_self = true;
    domain.metadata.abstraction_use = AbstractionUse::Standalone;
    let canonical = CanonicalAbstractionHeuristic::new(
        None,
        &task,
        vec![AbstractionComponent::domain(None, domain)],
    )
    .unwrap();
    assert!(!canonical.proves_initial_state_optimal());
}
