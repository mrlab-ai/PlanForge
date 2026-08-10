use crate::evaluation::domain_abstractions::domain_abstraction_factory::DomainAbstractionFactory;
use crate::evaluation::domain_abstractions::utils::identity_domain_mapping_and_sizes;
use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
    NumericRootTask, NumericType, NumericVariable, Operator,
};
use planforge_sas::utils::interval::Interval;

use super::*;
use crate::evaluation::domain_abstractions::cegar::flaw_search::single_switch_task;

// TODO: Test also sequence flaws beyond the first flaw.

#[test]
fn progression_sequence_flaws_find_precondition_violation() {
    let task = single_switch_task(2, 1, vec![0]);

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

    // The same task started at `v=1` makes the stored wildcard plan invalid:
    // the operator's precondition `v=0` no longer holds.
    let flawed_task = single_switch_task(2, 1, vec![1]);
    let mut flaws = Vec::new();
    get_sequence_progression_flaws(
        &flawed_task,
        factory.partitions(),
        &factory.domain_mapping,
        &plan,
        &mut flaws,
    )
    .unwrap();
    assert_eq!(flaws.len(), 1);
    match &flaws[0] {
        Flaw::Propositional(pf) => assert_eq!(pf.fact, ExplicitFact::propositional(0, 0)),
        _ => panic!("expected propositional flaw"),
    }
}

#[test]
fn progression_sequence_flaws_find_goal_violation() {
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

    // The abstraction cannot tell `v=1` from `v=2`, so the wildcard plan stops
    // at `v=1` and the concrete goal `v=2` stays open.
    let mut flaws = Vec::new();
    get_sequence_progression_flaws(
        &task,
        factory.partitions(),
        &factory.domain_mapping,
        &plan,
        &mut flaws,
    )
    .unwrap();
    assert_eq!(flaws.len(), 1);
    match &flaws[0] {
        Flaw::Propositional(pf) => assert_eq!(pf.fact, ExplicitFact::propositional(0, 2)),
        _ => panic!("expected propositional flaw"),
    }
}

#[test]
fn progression_sequence_flaws_find_numeric_deviation_flaw() {
    // Propositional vars: gt (comparison result), g (goal flag).
    let variables = vec![
        ExplicitVariable::new(
            ConditionValue::DOMAIN_SIZE,
            "gt".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            ConditionValue::False.as_usize(),
        ),
        ExplicitVariable::new(2, "g".into(), vec!["g0".into(), "g1".into()], None, 0),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c".into(), NumericType::Constant, None),
        NumericVariable::new("thresh".into(), NumericType::Constant, None),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(
        0,
        0,
        2,
        ComparisonOperator::GreaterThan,
    )];
    let op0 = Operator::new(
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
    let op1 = Operator::new(
        "set_g".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![Effect::new(vec![], 1, Some(0), 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::propositional(1, 1)],
        vec![],
        vec![2, 0],
        vec![-10.0, 3.0, -5.0],
        vec![op0, op1],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::propositional(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(f64::NEG_INFINITY, -5.0, false, true),
            Interval::new(-5.0, f64::INFINITY, false, false),
        ],
        vec![Interval::singleton(3.0)],
        vec![Interval::singleton(-5.0)],
    ]);

    // Hand-constructed wildcard plan:
    // - step 0 applies `op0` (inc)
    // - the abstract plan (optimistically) expects x to end up in the UPPER partition (index 1).
    let plan = WildcardPlanResult {
        wildcard_plan: vec![vec![0]],
        abstract_state_hashes: vec![],
        abstract_prop_states: vec![],
        abstract_numeric_states: vec![
            vec![0, 0, 0], // initial: x in LOWER
            vec![1, 0, 0], // expected after inc: x in UPPER
        ],
    };

    let domain_mapping = vec![vec![0, 0, 0], vec![0, 0]];
    let mut flaws = Vec::new();
    get_sequence_progression_flaws(&task, &partitions, &domain_mapping, &plan, &mut flaws).unwrap();
    assert!(
        flaws.iter().any(|f| matches!(f, Flaw::Numeric(_))),
        "expected a numeric deviation flaw"
    );
}

#[test]
fn regression_sequence_flaws_find_precondition_violation() {
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
    let mut flaws = Vec::new();
    get_sequence_regression_flaws(
        &task,
        &factory.partitions,
        &factory.domain_mapping,
        &plan,
        &mut flaws,
    )
    .unwrap();
    assert_eq!(flaws.len(), 1);
    match &flaws[0] {
        Flaw::Propositional(pf) => assert_eq!(pf.fact, ExplicitFact::propositional(0, 1)),
        _ => panic!("expected propositional flaw"),
    }
}

#[test]
fn regression_sequence_flaws_find_initial_state_violation() {
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
    let mut flaws = Vec::new();
    get_sequence_regression_flaws(
        &flawed_task,
        &factory.partitions,
        &factory.domain_mapping,
        &plan,
        &mut flaws,
    )
    .unwrap();
    assert_eq!(flaws.len(), 1);
    match &flaws[0] {
        Flaw::Propositional(pf) => assert_eq!(pf.fact, ExplicitFact::propositional(0, 1)),
        _ => panic!("expected propositional flaw"),
    }
}
