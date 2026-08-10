use crate::evaluation::domain_abstractions::utils::identity_domain_mapping_and_sizes;
use crate::evaluation::domain_abstractions::{
    domain_abstraction::NumericPartitions, domain_abstraction_factory::DomainAbstractionFactory,
};
use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
    NumericType, NumericVariable, Operator,
};
use planforge_sas::utils::interval::Interval;

use super::*;
use crate::evaluation::domain_abstractions::cegar::flaw_search::single_switch_task;

#[test]
fn regression_flaws_find_precondition_violation() {
    let task = single_switch_task(3, 2, vec![0]);

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

    // The abstraction cannot tell `v=1` from `v=2`, so regressing the goal
    // `v=2` through `set` leaves the unachievable precondition `v=1`.
    let flaws =
        get_regression_flaws(&task, &factory.partitions, &factory.domain_mapping, &plan).unwrap();
    assert_eq!(flaws.len(), 1);
    match &flaws[0] {
        Flaw::Propositional(pf) => assert_eq!(pf.fact, ExplicitFact::propositional(0, 1)),
        _ => panic!("expected propositional flaw"),
    }
}

#[test]
fn regression_flaws_find_initial_state_violation() {
    let task = single_switch_task(3, 1, vec![0]);

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

    // The same task started at `v=1` violates the plan's initial-state
    // requirement `v=0`.
    let flawed_task = single_switch_task(3, 1, vec![1]);
    let flaws = get_regression_flaws(
        &flawed_task,
        &factory.partitions,
        &factory.domain_mapping,
        &plan,
    )
    .unwrap();
    assert_eq!(flaws.len(), 1);
    match &flaws[0] {
        Flaw::Propositional(pf) => assert_eq!(pf.fact, ExplicitFact::propositional(0, 1)),
        _ => panic!("expected propositional flaw"),
    }
}

#[test]
fn regression_flaws_regress_goal_comparison_through_additive_constant_effect() {
    let variables = vec![ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        "cmp".into(),
        vec!["true".into(), "false".into()],
        Some(0),
        ConditionValue::False.as_usize(),
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
        vec![ExplicitFact::propositional(0, 0)],
        vec![],
        vec![1],
        vec![0.0, 3.0, 10.0],
        vec![op],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::propositional(0, 0),
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
