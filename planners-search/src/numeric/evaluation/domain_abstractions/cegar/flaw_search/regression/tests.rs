use crate::numeric::evaluation::domain_abstractions::utils::identity_domain_mapping_and_sizes;
use crate::numeric::evaluation::domain_abstractions::{
    comparison_expression::Interval, domain_abstraction::NumericPartitions,
    domain_abstraction_factory::DomainAbstractionFactory,
};
use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator};
use planners_sas::numeric::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
    NumericRootTask, NumericType, NumericVariable, Operator,
};

use super::*;

#[test]
fn regression_flaws_find_precondition_violation() {
    let variables = vec![ExplicitVariable::new(
        3,
        "v".into(),
        vec!["v0".into(), "v1".into(), "v2".into()],
        None,
        0,
    )];
    let numeric_variables: Vec<NumericVariable> = vec![];
    let goals = vec![ExplicitFact::new(0, 2)];
    let op = Operator::new(
        "set".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 0, Some(0), 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        goals,
        vec![],
        vec![0],
        vec![],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let (mut domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    // Put 1 and 2 in the same mapping group.
    domain_mapping[0] = vec![0, 1, 1];
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

    task.set_initial_propositional_state_values(vec![0]);

    let flaws =
        get_regression_flaws(&task, &factory.partitions, &factory.domain_mapping, &plan).unwrap();
    assert_eq!(flaws.len(), 1);
    match &flaws[0] {
        Flaw::Propositional(pf) => assert_eq!(pf.fact, ExplicitFact::new(0, 1)),
        _ => panic!("expected propositional flaw"),
    }
}

#[test]
fn regression_flaws_find_initial_state_violation() {
    let variables = vec![ExplicitVariable::new(
        3,
        "v".into(),
        vec!["v0".into(), "v1".into(), "v2".into()],
        None,
        0,
    )];
    let numeric_variables: Vec<NumericVariable> = vec![];
    let goals = vec![ExplicitFact::new(0, 1)];
    let op = Operator::new(
        "set".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 0, Some(0), 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        goals,
        vec![],
        vec![0],
        vec![],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

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

    // Make initial state violation.
    task.set_initial_propositional_state_values(vec![1]);

    let flaws =
        get_regression_flaws(&task, &factory.partitions, &factory.domain_mapping, &plan).unwrap();
    assert_eq!(flaws.len(), 1);
    match &flaws[0] {
        Flaw::Propositional(pf) => assert_eq!(pf.fact, ExplicitFact::new(0, 1)),
        _ => panic!("expected propositional flaw"),
    }
}

#[test]
fn regression_flaws_regress_goal_comparison_through_additive_constant_effect() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("delta".into(), NumericType::Constant, None),
        NumericVariable::new("threshold".into(), NumericType::Constant, None),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(
        0,
        0,
        2,
        ComparisonOperator::GreaterThanOrEqual,
    )];
    let op = Operator::new(
        "inc".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Plus,
            1,
            false,
            vec![],
        )],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![1],
        vec![0.0, 3.0, 10.0],
        vec![op],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let partitions = NumericPartitions::with_partitions(vec![
        vec![Interval::unbounded()],
        vec![Interval::singleton(3.0)],
        vec![Interval::singleton(10.0)],
    ]);
    let numeric_domain_sizes: Vec<usize> = vec![1, 1, 1];
    let factory = DomainAbstractionFactory::new(
        &task,
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
    )
    .unwrap();
    let plan = WildcardPlanResult {
        wildcard_plan: vec![vec![0]],
        abstract_state_hashes: vec![],
        abstract_prop_states: vec![],
        abstract_numeric_states: vec![],
    };

    let flaws =
        get_regression_flaws(&task, &factory.partitions, &factory.domain_mapping, &plan).unwrap();
    assert!(
        flaws.iter().any(|flaw| matches!(
            flaw,
            Flaw::Numeric(NumericFlaw {
                numeric_var_id: 0,
                value,
                include_in_lower: false,
                step: 0,
            }) if *value == 7.0
        )),
        "expected split x >= 7 after regressing x >= 10 through x += 3, got {flaws:?}"
    );
}
