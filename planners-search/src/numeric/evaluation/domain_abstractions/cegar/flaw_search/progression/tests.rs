use crate::numeric::evaluation::domain_abstractions::comparison_expression::Interval;
use crate::numeric::evaluation::domain_abstractions::domain_abstraction_factory::DomainAbstractionFactory;
use crate::numeric::evaluation::domain_abstractions::utils::identity_domain_mapping_and_sizes;
use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator};
use planners_sas::numeric::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
    NumericRootTask, NumericType, NumericVariable, Operator,
};

use super::*;
use crate::numeric::evaluation::domain_abstractions::cegar::flaw_search::SplitDirection;

#[test]
fn progression_flaws_find_precondition_violation() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
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

    // Make the stored wildcard plan invalid in the concrete initial state.
    task.set_initial_propositional_state_values(vec![1]);

    let flaws =
        get_progression_flaws(&task, factory.partitions(), &plan, SplitDirection::Forward).unwrap();
    assert_eq!(flaws.len(), 1);
    match &flaws[0] {
        Flaw::Propositional(pf) => assert_eq!(pf.fact, ExplicitFact::new(0, 0)),
        _ => panic!("expected propositional flaw"),
    }
}

#[test]
fn progression_flaws_find_goal_violation() {
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
        get_progression_flaws(&task, factory.partitions(), &plan, SplitDirection::Forward).unwrap();
    assert_eq!(flaws.len(), 1);
    match &flaws[0] {
        Flaw::Propositional(pf) => assert_eq!(pf.fact, ExplicitFact::new(0, 2)),
        _ => panic!("expected propositional flaw"),
    }
}

#[test]
fn progression_flaws_find_numeric_deviation_flaw() {
    // Propositional vars: gt (comparison result), g (goal flag).
    let variables = vec![
        ExplicitVariable::new(
            3,
            "gt".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
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
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 1, Some(0), 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![2, 0],
        vec![-10.0, 3.0, -5.0],
        vec![op0, op1],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
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
        abstract_operator_count: 1,
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
    assert_eq!(forward_numeric.value, -10.0);

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
    assert_eq!(backward_numeric.value, -8.0);
}
