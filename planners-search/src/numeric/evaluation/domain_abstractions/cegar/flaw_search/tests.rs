use crate::numeric::evaluation::domain_abstractions::cegar::{
    apply_initial_goal_splits, compute_initial_split_mapping, identity_domain_mapping_and_sizes
};
use crate::numeric::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::InitSplitMethod;
use crate::numeric::evaluation::domain_abstractions::domain_abstraction_factory::DomainAbstractionFactory;

use super::*;
use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator};
use rand::{SeedableRng, rngs::SmallRng};

use planners_sas::numeric::numeric_task::{
    Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericType, NumericVariable,
    Operator,
};

#[test]
fn get_flaws_returns_empty_for_valid_wildcard_plan() {
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
        vec![planners_sas::numeric::numeric_task::Effect::new(
            vec![],
            0,
            Some(0),
            1,
        )],
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

    let flaws = get_flaws(&task, factory.partitions(), &plan, false).unwrap();
    assert!(flaws.is_empty());
}

#[test]
fn get_flaws_reports_precondition_violation() {
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
        vec![planners_sas::numeric::numeric_task::Effect::new(
            vec![],
            0,
            Some(0),
            1,
        )],
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

    let flaws = get_flaws(&task, factory.partitions(), &plan, false).unwrap();
    assert_eq!(flaws.len(), 1);
    match &flaws[0] {
        Flaw::Propositional(pf) => assert_eq!(pf.fact, ExplicitFact::new(0, 0)),
        _ => panic!("expected propositional flaw"),
    }
}

#[test]
fn numeric_init_split_is_applied_for_encoded_init_split_var() {
    let variables = vec![ExplicitVariable::new(
        2,
        "g".into(),
        vec!["g0".into(), "g1".into()],
        None,
        0,
    )];
    let numeric_variables = vec![NumericVariable::new("x".into(), NumericType::Regular, None)];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![3.5],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let mut config = CegarConfig {
        init_split_method: InitSplitMethod::Identity,
        ..Default::default()
    };
    config.init_split_method = InitSplitMethod::Identity;
    config.init_split_var_ids = Some(HashSet::from([1usize]));

    let mut rng = SmallRng::seed_from_u64(7);
    let mut domain_mapping = vec![vec![0, 0]];
    let mut domain_sizes = vec![1];
    let mut partitions = NumericPartitions::trivial(&task);
    let mut numeric_domain_sizes = vec![1];

    apply_initial_goal_splits(
        &task,
        &config,
        &mut rng,
        &HashSet::new(),
        &HashSet::new(),
        &mut domain_mapping,
        &mut domain_sizes,
        &mut partitions,
        &mut numeric_domain_sizes,
    );

    assert_eq!(numeric_domain_sizes, vec![2]);
    let parts = partitions.partitions(0).unwrap();
    assert_eq!(parts.len(), 2);
    assert!(parts[0].contains(3.5) || parts[1].contains(3.5));
}

#[test]
fn get_flaws_reports_numeric_deviation_flaw() {
    use crate::numeric::evaluation::domain_abstractions::comparison_expression::Interval;
    use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator};
    use planners_sas::numeric::numeric_task::{AssignmentEffect, AssignmentOperation, NumericType};

    // Propositional vars: gt (comparison result), g (goal flag)
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
    // - step 0 applies op0 (inc)
    // - the abstract plan (optimistically) expects x to end up in the UPPER partition (index 1)
    let plan = WildcardPlanResult {
        wildcard_plan: vec![vec![0]],
        abstract_state_hashes: vec![],
        abstract_prop_states: vec![],
        abstract_numeric_states: vec![
            vec![0, 0, 0], // initial: x in LOWER
            vec![1, 0, 0], // expected after inc: x in UPPER
        ],
    };

    let flaws = get_flaws(&task, &partitions, &plan, false).unwrap();
    assert!(
        flaws.iter().any(|f| matches!(f, Flaw::Numeric(_))),
        "expected a numeric deviation flaw"
    );
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn fix_flaws_respects_max_abstraction_size_limit() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        None,
        0,
    )];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let mut config = CegarConfig::default();
    config.max_abstraction_size = 1;

    let mut domain_mapping = vec![vec![0, 0]];
    let mut domain_sizes = vec![1];
    let mut partitions = NumericPartitions::trivial(&task);
    let mut numeric_domain_sizes = vec![];
    let mut rng = SmallRng::seed_from_u64(7);
    let mut blacklisted_prop_var_ids = HashSet::new();
    let mut blacklisted_numeric_var_ids = HashSet::new();
    let flaws = vec![Flaw::Propositional(PropFlaw {
        fact: ExplicitFact::new(0, 1),
        dependent_numeric_flaws: vec![],
    })];

    let refined = fix_flaws(
        &config,
        &task,
        &flaws,
        &mut domain_mapping,
        &mut domain_sizes,
        &mut partitions,
        &mut numeric_domain_sizes,
        &mut rng,
        &mut blacklisted_prop_var_ids,
        &mut blacklisted_numeric_var_ids,
    )
    .unwrap();

    assert!(!refined);
    assert_eq!(domain_sizes, vec![1]);
    assert_eq!(domain_mapping, vec![vec![0, 0]]);
    assert!(blacklisted_prop_var_ids.contains(&0));
}

#[test]
fn init_value_split_uses_true_branch_for_comparison_variables() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(
        0,
        0,
        1,
        ComparisonOperator::GreaterThan,
    )];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![2],
        vec![1.0, 0.0],
        vec![],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    let config = CegarConfig {
        init_split_method: InitSplitMethod::InitValue,
        ..Default::default()
    };
    let mut rng = SmallRng::seed_from_u64(7);

    let (new_domain_size, mapping) =
        compute_initial_split_mapping(&task, &config, 0, Some(0), &mut rng).unwrap();

    assert_eq!(new_domain_size, 2);
    assert_eq!(mapping, vec![1, 0, 0]);
}
