use crate::evaluation::abstractions::domain::factory::DomainAbstractionFactory;
use crate::evaluation::abstractions::domain::utils::identity_domain_mapping_and_sizes;

use super::*;

use planforge_sas::numeric_task::{
    ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, NumericVariable, Operator, OperatorCost,
};

#[test]
fn get_flaws_returns_empty_for_valid_wildcard_plan() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        None,
        ExplicitValueIndex::new(0),
    )];
    let numeric_variables: Vec<NumericVariable> = vec![];
    let goals = vec![ExplicitFact::propositional(0, 1)];
    let op = Operator::new(
        "set".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![planforge_sas::numeric_task::Effect::new(
            vec![],
            VariableIndex::from_usize(0),
            Some(ExplicitValueIndex::new(0)),
            ExplicitValueIndex::new(1),
        )],
        vec![],
        OperatorCost::new(1),
    );
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals,
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let partitions = NumericPartitions::trivial(&task);
    let numeric_domain_sizes: Vec<usize> = vec![];
    let factory = DomainAbstractionFactory::new(
        &task,
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
    )
    .unwrap();
    let plan = factory
        .compute_wildcard_plan(&task, true, false)
        .unwrap()
        .expect("plan exists");

    let mut flaws = get_flaws(
        &task,
        factory.partitions(),
        factory.domain_mapping(),
        &plan,
        FlawKind::Progression,
    )
    .unwrap()
    .flaws;
    assert!(flaws.is_empty());

    flaws = get_flaws(
        &task,
        factory.partitions(),
        factory.domain_mapping(),
        &plan,
        FlawKind::Regression,
    )
    .unwrap()
    .flaws;
    assert!(flaws.is_empty());
}
