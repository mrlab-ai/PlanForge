use std::collections::HashSet;

use crate::numeric::evaluation::domain_abstractions::cegar::{
    CegarConfig, apply_initial_goal_splits, compute_initial_split_mapping
};
use crate::numeric::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::InitSplitMethod;
use crate::numeric::evaluation::domain_abstractions::domain_abstraction_factory::DomainAbstractionFactory;
use crate::numeric::evaluation::domain_abstractions::utils::identity_domain_mapping_and_sizes;

use super::*;
use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator};
use rand::{SeedableRng, rngs::SmallRng};

use planners_sas::numeric::numeric_task::{
    ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericType, NumericVariable, Operator,
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

    let mut flaws = get_flaws(
        &task,
        factory.partitions(),
        factory.transition_system.domain_mapping(),
        &plan,
        FlawKind::Progression,
    )
    .unwrap();
    assert!(flaws.is_empty());

    flaws = get_flaws(
        &task,
        factory.partitions(),
        factory.transition_system.domain_mapping(),
        &plan,
        FlawKind::Regression,
    )
    .unwrap();
    assert!(flaws.is_empty());
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
