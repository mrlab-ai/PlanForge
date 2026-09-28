use crate::evaluation::domain_abstractions::domain_abstraction_factory::DomainAbstractionFactory;
use crate::evaluation::domain_abstractions::utils::identity_domain_mapping_and_sizes;
use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex,
    ExplicitVariable, INF_VALUE, Metric, NEG_INF_VALUE, NumericRootTask, NumericRootTaskParts,
    NumericType, NumericVariable, Operator, OperatorCost, OperatorIndex,
};
use planforge_sas::utils::interval::Interval;

use super::*;
use crate::evaluation::domain_abstractions::cegar::flaw_search::{
    SplitDirection, single_switch_task,
};

#[test]
fn progression_flaws_find_precondition_violation() {
    let task = single_switch_task(
        2,
        ExplicitValueIndex::new(1),
        vec![ExplicitValueIndex::new(0)],
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

    // The same task started at `v=1` makes the stored wildcard plan invalid:
    // the operator's precondition `v=0` no longer holds.
    let flawed_task = single_switch_task(
        2,
        ExplicitValueIndex::new(1),
        vec![ExplicitValueIndex::new(1)],
    );
    let flaws = get_progression_flaws(
        &flawed_task,
        factory.partitions(),
        &plan,
        SplitDirection::Forward,
    )
    .unwrap();
    assert_eq!(flaws.len(), 1);
    match &flaws[0] {
        Flaw::Propositional(pf) => assert_eq!(pf.fact, ExplicitFact::propositional(0, 0)),
        _ => panic!("expected propositional flaw"),
    }
}

#[test]
fn progression_flaws_find_goal_violation() {
    let task = single_switch_task(
        3,
        ExplicitValueIndex::new(2),
        vec![ExplicitValueIndex::new(0)],
    );

    let (mut domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    // Put 1 and 2 in the same mapping group.
    domain_mapping[0] = vec![
        ExplicitValueIndex::new(0),
        ExplicitValueIndex::new(1),
        ExplicitValueIndex::new(1),
    ];
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
    let flaws =
        get_progression_flaws(&task, factory.partitions(), &plan, SplitDirection::Forward).unwrap();
    assert_eq!(flaws.len(), 1);
    match &flaws[0] {
        Flaw::Propositional(pf) => assert_eq!(pf.fact, ExplicitFact::propositional(0, 2)),
        _ => panic!("expected propositional flaw"),
    }
}

#[test]
fn progression_flaws_find_numeric_deviation_flaw() {
    // Propositional vars: gt (comparison result), g (goal flag).
    let variables = vec![
        ExplicitVariable::new(
            ConditionValue::DOMAIN_SIZE,
            "gt".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
        ),
        ExplicitVariable::new(
            2,
            "g".into(),
            vec!["g0".into(), "g1".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c".into(), NumericType::Constant, None),
        NumericVariable::new("thresh".into(), NumericType::Constant, None),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::from_usize(0),
        VariableIndex::from_usize(0),
        VariableIndex::from_usize(2),
        ComparisonOperator::GreaterThan,
    )];
    let op0 = Operator::new(
        "inc".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::from_usize(0),
            AssignmentOperation::Plus,
            VariableIndex::from_usize(1),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );
    let op1 = Operator::new(
        "set_g".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![Effect::new(
            vec![],
            VariableIndex::from_usize(1),
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
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2), ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(-10.0),
            NumericValue::new(3.0),
            NumericValue::new(-5.0),
        ],
        operators: vec![op0, op1],
        axioms: vec![],
        comparison_axioms,
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(NEG_INF_VALUE, NumericValue::new(-5.0), false, true),
            Interval::new(NumericValue::new(-5.0), INF_VALUE, false, false),
        ],
        vec![Interval::singleton(NumericValue::new(3.0))],
        vec![Interval::singleton(NumericValue::new(-5.0))],
    ]);

    // Hand-constructed wildcard plan:
    // - step 0 applies `op0` (inc)
    // - the abstract plan (optimistically) expects x to end up in the UPPER partition (index 1).
    let plan = WildcardPlanResult {
        wildcard_plan: vec![vec![OperatorIndex::new(0)]],
        abstract_state_hashes: vec![],
        abstract_prop_states: vec![],
        abstract_numeric_states: vec![
            vec![0, 0, 0], // initial: x in LOWER
            vec![1, 0, 0], // expected after inc: x in UPPER
        ],
    };

    let forward_flaws =
        get_progression_flaws(&task, &partitions, &plan, SplitDirection::Forward).unwrap();
    assert!(
        forward_flaws.iter().any(|f| matches!(f, Flaw::Numeric(_))),
        "expected a numeric deviation flaw"
    );

    // Forward direction splits at the *concrete current* value (-10.0).
    let forward_numeric = forward_flaws
        .iter()
        .find_map(|f| match f {
            Flaw::Numeric(nf) => Some(nf),
            _ => None,
        })
        .unwrap();
    assert_eq!(forward_numeric.value, NumericValue::new(-10.0));

    // Backward direction splits at the *boundary* of the expected target
    // interval regressed by the operator's effect (+3): boundary 0.0 (the
    // lower bound of the UPPER partition (5, +inf) is unbounded on the lower
    // side, so the regressed split aligns with -5.0 - 3.0 = -8.0, the lower
    // boundary of the upper partition `(-5, +inf)`).
    let backward_flaws =
        get_progression_flaws(&task, &partitions, &plan, SplitDirection::Backward).unwrap();
    let backward_numeric = backward_flaws
        .iter()
        .find_map(|f| match f {
            Flaw::Numeric(nf) => Some(nf),
            _ => None,
        })
        .expect("backward direction should also produce a numeric flaw");
    assert_ne!(
        backward_numeric.value, forward_numeric.value,
        "backward split should differ from forward concrete-value split"
    );
    // The regressed boundary for the UPPER partition `(-5, +inf)` mapped back
    // through `+3` lands at `-5 - 3 = -8.0`; the boundary is open on the lower
    // side so include_in_lower flips to true on the regressed side.
    assert_eq!(backward_numeric.value, NumericValue::new(-8.0));
}
