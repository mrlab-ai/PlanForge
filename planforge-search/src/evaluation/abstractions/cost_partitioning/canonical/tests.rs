use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::common::abstraction::AbstractionMetadata;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;
use crate::evaluation::abstractions::common::cegar::refine::CegarOutcome;
use crate::evaluation::abstractions::common::cegar::{CegarStopReason, SplitKind};
use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, Operator, OperatorCost, OperatorIndex, VariableIndex,
};
use planforge_sas::state_registry::StateRegistry;

use crate::evaluation::abstractions::cartesian::{
    CartesianAbstractionConfig, CartesianAbstractionGenerator,
};
use crate::evaluation::abstractions::common::abstraction::AbstractionUse;
use crate::evaluation::abstractions::cost_partitioning::max::MaxAbstractionHeuristic;
use crate::evaluation::abstractions::pdb::pattern_database::PatternDatabase;
use crate::evaluation::abstractions::pdb::projected_task::{Pattern, ProjectedTask};
use crate::evaluation::evaluator::EvaluationState;

use super::*;
use crate::evaluation::abstractions::common::abstraction::{AbstractionCore, CegarAbstraction};
use crate::evaluation::abstractions::common::cegar::semantics::CegarSemantics;
use crate::evaluation::abstractions::domain::abstraction::DomainAbstraction;
use crate::evaluation::abstractions::domain::factory::DomainAbstractionFactory;
use crate::evaluation::abstractions::domain::partitions::NumericPartitions;
use crate::resource_limits::Deadline;

fn simple_var(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        None,
        ExplicitValueIndex::new(1),
    )
}

fn simple_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![simple_var("p"), simple_var("q")],
        numeric_variables: vec![],
        goals: vec![
            ExplicitFact::propositional(0, 1),
            ExplicitFact::propositional(1, 1),
        ],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![
            Operator::new(
                "set-p".to_string(),
                vec![],
                vec![Effect::new(
                    vec![],
                    VariableIndex::new(0),
                    Some(ExplicitValueIndex::new(0)),
                    ExplicitValueIndex::new(1),
                )],
                vec![],
                OperatorCost::new(2),
            ),
            Operator::new(
                "set-q".to_string(),
                vec![],
                vec![Effect::new(
                    vec![],
                    VariableIndex::new(1),
                    Some(ExplicitValueIndex::new(0)),
                    ExplicitValueIndex::new(1),
                )],
                vec![],
                OperatorCost::new(3),
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

fn make_abstraction(task: &NumericRootTask, distances: Vec<f64>) -> DomainAbstraction {
    let factory = DomainAbstractionFactory::new(
        &CegarSemantics::for_options(task, &CegarOptions::default()).unwrap(),
        vec![
            vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)],
            vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)],
        ],
        vec![2, 2],
        NumericPartitions::with_partitions(vec![]),
    )
    .unwrap();

    let transitions = factory.transitions(task, false, Deadline::NONE).unwrap();
    let core = AbstractionCore {
        distances,
        initial_state: 0,
        relevant_operator_ids: Vec::new(),
        num_abstract_operators: 0,
        metadata: AbstractionMetadata::standalone(
            CegarOutcome {
                stop_reason: CegarStopReason::SizeLimit,
                concrete_plan: None,
                pending_flaw: None,
                refinements: 0,
            },
            SplitKind::Concrete,
            None,
        ),
    };
    let mut abstraction = DomainAbstraction::new(factory, core, Some(transitions));
    abstraction.discard_transition_data();
    abstraction
}

#[test]
fn computes_max_additive_subsets_from_relevant_operators() {
    let subsets = compute_max_additive_subsets_from_relevant_operators(&[
        [OperatorIndex::new(0)].into_iter().collect(),
        [OperatorIndex::new(1)].into_iter().collect(),
        [OperatorIndex::new(0)].into_iter().collect(),
    ]);

    assert_eq!(subsets, vec![vec![0, 1], vec![1, 2]]);
}

#[test]
fn canonical_domain_abstraction_uses_explicit_subsets() {
    let task = simple_task();
    let mut registry = StateRegistry::for_task(std::sync::Arc::new(&task));
    let initial_state = registry.get_initial_state();
    let mut da0 = make_abstraction(&task, vec![2.0, 0.0, 0.0, 0.0]);
    da0.core_mut().relevant_operator_ids = vec![OperatorIndex::new(0)];
    let mut da1 = make_abstraction(&task, vec![3.0, 0.0, 0.0, 0.0]);
    da1.core_mut().relevant_operator_ids = vec![OperatorIndex::new(1)];
    let mut da2 = make_abstraction(&task, vec![4.0, 0.0, 0.0, 0.0]);
    da2.core_mut().relevant_operator_ids = vec![OperatorIndex::new(0), OperatorIndex::new(1)];

    let heuristic = CanonicalAbstractionHeuristic::with_explicit_subsets(
        None,
        &task,
        vec![
            AbstractionComponent::from(da0),
            AbstractionComponent::from(da1),
            AbstractionComponent::from(da2),
        ],
        vec![vec![0, 1], vec![2]],
    )
    .unwrap();

    let eval_state = EvaluationState::new(&initial_state, &task, &registry);
    let value = heuristic.compute_heuristic(&eval_state).unwrap();

    assert_eq!(value, 5.0);
}

fn mixed_components<'task>(task: &'task NumericRootTask) -> Vec<AbstractionComponent<'task>> {
    let mut domain = make_abstraction(task, vec![2.0, 0.0, 2.0, 0.0]);
    domain.core_mut().relevant_operator_ids = vec![OperatorIndex::new(0)];

    let cartesian = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        cegar: CegarOptions {
            max_states: 16,
            max_time: None,
            combine_labels: false,
            random_seed: None,
            debug: false,
            ..CartesianAbstractionConfig::default().cegar
        },
        ..Default::default()
    })
    .unwrap()
    .generate(task)
    .unwrap();

    let pattern = Pattern::new(vec![VariableIndex::from_usize(1)], vec![]);
    let projected = ProjectedTask::new(task, &pattern).unwrap();
    let pdb = PatternDatabase::new(projected, 32).unwrap();

    vec![
        AbstractionComponent::from(domain),
        AbstractionComponent::from(cartesian),
        AbstractionComponent::from(pdb),
    ]
}

#[test]
fn mixed_domain_cartesian_and_pdb_components_work_in_max_and_canonical() {
    let task = simple_task();
    let mut registry = StateRegistry::for_task(std::sync::Arc::new(&task));
    let initial_state = registry.get_initial_state();
    let eval_state = EvaluationState::new(&initial_state, &task, &registry);

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
    domain.core_mut().relevant_operator_ids = vec![OperatorIndex::new(0)];
    domain.metadata_mut().stop_reason = CegarStopReason::ConcretePlan;
    domain.metadata_mut().abstraction_use = AbstractionUse::Standalone;
    let component = AbstractionComponent::from(domain);
    assert!(component.proves_initial_state_optimal());
    let max = MaxAbstractionHeuristic::new(None, vec![component]).unwrap();
    assert!(!max.proves_initial_state_optimal());

    let mut domain = make_abstraction(&task, vec![5.0, 0.0, 0.0, 0.0]);
    domain.core_mut().relevant_operator_ids = vec![OperatorIndex::new(0)];
    domain.metadata_mut().stop_reason = CegarStopReason::ConcretePlan;
    domain.metadata_mut().abstraction_use = AbstractionUse::Standalone;
    let canonical =
        CanonicalAbstractionHeuristic::new(None, &task, vec![AbstractionComponent::from(domain)])
            .unwrap();
    assert!(!canonical.proves_initial_state_optimal());
}
