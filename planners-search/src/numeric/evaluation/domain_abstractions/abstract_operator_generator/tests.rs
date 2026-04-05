use super::*;

use planners_sas::numeric::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator,
};
use planners_sas::numeric::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
    NumericType, NumericVariable, Operator,
};

use crate::numeric::evaluation::domain_abstractions::comparison_expression::Interval;
use crate::numeric::evaluation::domain_abstractions::domain_abstraction::NumericPartitions;

#[test]
fn numeric_partition_transitions_and_comparison_filtering() {
    // Propositional var 0 is the derived comparison var for (x0 < c10).
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];

    // Numeric vars: x0 (regular), c10 (constant), c7 (constant)
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c10".into(), NumericType::Constant, None),
        NumericVariable::new("c7".into(), NumericType::Constant, None),
    ];

    // Comparison axiom (derived propositional var 0): x0 < c10
    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];

    // Operator requires comparison to hold (var0 == 0) and applies x0 += c7.
    let op = Operator::new(
        "op".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Plus,
            2,
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
        vec![],
        vec![],
        vec![0],
        vec![0.0, 10.0, 7.0],
        vec![op],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    // Partitions for x0: (-inf,10) and [10,inf)
    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(f64::NEG_INFINITY, 10.0, false, false),
            Interval::new(10.0, f64::INFINITY, true, false),
        ],
        vec![Interval::singleton(10.0)],
        vec![Interval::singleton(7.0)],
    ]);

    let numeric_domain_sizes = vec![2, 1, 1];

    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        numeric_domain_sizes,
        false,
    )
    .unwrap();
    let abs_ops = generator.build_abstract_operators(&task).unwrap();

    // From x0 in (-inf,10), x0 += 7 can stay in partition 0 or move to 1.
    // From [10,inf), the precondition x0 < 10 is definitely contradicted and filtered.
    assert_eq!(abs_ops.len(), 2);

    let mut hash_effects: Vec<i32> = abs_ops.iter().map(|o| o.hash_effect).collect();
    hash_effects.sort();
    assert_eq!(hash_effects, vec![-5, -2]);
}

#[test]
fn repeated_numeric_operator_generation_is_deterministic() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];

    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c10".into(), NumericType::Constant, None),
        NumericVariable::new("c7".into(), NumericType::Constant, None),
    ];

    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];

    let op = Operator::new(
        "op".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Plus,
            2,
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
        vec![],
        vec![],
        vec![0],
        vec![0.0, 10.0, 7.0],
        vec![op],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(f64::NEG_INFINITY, 10.0, false, false),
            Interval::new(10.0, f64::INFINITY, true, false),
        ],
        vec![Interval::singleton(10.0)],
        vec![Interval::singleton(7.0)],
    ]);

    #[allow(clippy::type_complexity)]
    let mut signatures: Vec<Vec<(i32, Vec<ExplicitFact>, Vec<ExplicitFact>)>> = Vec::new();
    for _ in 0..12 {
        let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
            &task,
            partitions.clone(),
            vec![2, 1, 1],
            false,
        )
        .unwrap();
        let ops = generator.build_abstract_operators(&task).unwrap();
        signatures.push(
            ops.iter()
                .map(|op| {
                    (
                        op.hash_effect,
                        op.preconditions.clone(),
                        op.regression_preconditions.clone(),
                    )
                })
                .collect(),
        );
    }

    for sig in signatures.iter().skip(1) {
        assert_eq!(sig, &signatures[0]);
    }
}

#[test]
fn numeric_transition_adds_implicit_comparison_preconditions() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];

    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c10".into(), NumericType::Constant, None),
        NumericVariable::new("c7".into(), NumericType::Constant, None),
    ];

    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];

    let op = Operator::new(
        "op".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Plus,
            2,
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
        vec![],
        vec![],
        vec![0],
        vec![0.0, 10.0, 7.0],
        vec![op],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(f64::NEG_INFINITY, 10.0, false, false),
            Interval::new(10.0, f64::INFINITY, true, false),
        ],
        vec![Interval::singleton(10.0)],
        vec![Interval::singleton(7.0)],
    ]);

    let numeric_domain_sizes = vec![2, 1, 1];

    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        numeric_domain_sizes,
        false,
    )
    .unwrap();
    let mut abs_ops = generator.build_abstract_operators(&task).unwrap();
    abs_ops.sort_by_key(|op| op.hash_effect);

    assert_eq!(abs_ops.len(), 3);

    assert_eq!(
        abs_ops[0].preconditions,
        vec![ExplicitFact::new(0, 0), ExplicitFact::new(1, 0)]
    );
    assert_eq!(
        abs_ops[0].regression_preconditions,
        vec![ExplicitFact::new(0, 1), ExplicitFact::new(1, 1)]
    );
    assert_eq!(abs_ops[0].hash_effect, -4);

    let trailing_pairs: Vec<(Vec<ExplicitFact>, Vec<ExplicitFact>)> = abs_ops[1..]
        .iter()
        .map(|op| {
            (
                op.preconditions.clone(),
                op.regression_preconditions.clone(),
            )
        })
        .collect();
    assert!(
        trailing_pairs.contains(&(vec![ExplicitFact::new(1, 0)], vec![ExplicitFact::new(1, 0)],))
    );
    assert!(
        trailing_pairs.contains(&(vec![ExplicitFact::new(1, 1)], vec![ExplicitFact::new(1, 1)],))
    );
}

#[test]
fn implicit_comparison_transition_requires_definite_change_on_both_sides() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];

    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c10".into(), NumericType::Constant, None),
        NumericVariable::new("c7".into(), NumericType::Constant, None),
    ];

    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];

    let op = Operator::new(
        "op".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Plus,
            2,
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
        vec![],
        vec![],
        vec![0],
        vec![0.0, 10.0, 7.0],
        vec![op],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(f64::NEG_INFINITY, 10.0, false, false),
            Interval::new(10.0, f64::INFINITY, true, false),
        ],
        vec![Interval::singleton(10.0)],
        vec![Interval::singleton(7.0)],
    ]);

    let numeric_domain_sizes = vec![2, 1, 1];

    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        numeric_domain_sizes,
        false,
    )
    .unwrap();
    let abs_ops = generator.build_abstract_operators(&task).unwrap();

    let changed_cmp_ops: Vec<&AbstractOperator> = abs_ops
        .iter()
        .filter(|op| op.regression_preconditions.iter().any(|fact| fact.var == 0))
        .collect();
    assert_eq!(changed_cmp_ops.len(), 1);
    assert_eq!(
        changed_cmp_ops[0].preconditions,
        vec![ExplicitFact::new(0, 0), ExplicitFact::new(1, 0)]
    );
    assert_eq!(
        changed_cmp_ops[0].regression_preconditions,
        vec![ExplicitFact::new(0, 1), ExplicitFact::new(1, 1)]
    );
}

#[test]
fn affected_numeric_var_stays_marked_changed_with_identity_partition_transition() {
    let variables: Vec<ExplicitVariable> = vec![];

    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
    ];

    let op = Operator::new(
        "op".into(),
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
        vec![],
        vec![],
        vec![],
        vec![0.0, 1.0],
        vec![op.clone()],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(f64::NEG_INFINITY, 10.0, false, false),
            Interval::new(10.0, f64::INFINITY, true, false),
        ],
        vec![Interval::singleton(1.0)],
    ]);

    let numeric_domain_sizes = vec![2, 1];
    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        numeric_domain_sizes,
        false,
    )
    .unwrap();

    let transitions = compute_hash_effects_with_preconditions(
        &task,
        &mut generator,
        &[],
        op.assignment_effects(),
    )
    .unwrap();

    let identity_transition = transitions.iter().find(|trans| {
        trans.source_partition_facts == vec![ExplicitFact::new(0, 0)]
            && trans.target_partition_facts == vec![ExplicitFact::new(0, 0)]
    });
    assert!(identity_transition.is_some());
    assert_eq!(identity_transition.unwrap().changed_numeric_vars, vec![0]);
}

#[test]
fn derived_numeric_partitions_are_not_materialized_in_transitions() {
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
        NumericVariable::new("c5".into(), NumericType::Constant, None),
        NumericVariable::new("d1".into(), NumericType::Derived, None),
        NumericVariable::new("d2".into(), NumericType::Derived, None),
    ];

    let assignment_axioms = vec![
        AssignmentAxiom::new(3, CalOperator::Sum, 0, 1),
        AssignmentAxiom::new(4, CalOperator::Difference, 3, 2),
    ];

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
        vec![],
        numeric_variables,
        vec![],
        vec![],
        vec![],
        vec![3.0, 1.0, 5.0, 4.0, -1.0],
        vec![op.clone()],
        vec![],
        vec![],
        assignment_axioms,
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![Interval::singleton(3.0), Interval::singleton(4.0)],
        vec![Interval::singleton(1.0)],
        vec![Interval::singleton(5.0)],
        vec![Interval::singleton(4.0), Interval::singleton(5.0)],
        vec![Interval::singleton(-1.0), Interval::singleton(0.0)],
    ]);

    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        vec![2, 1, 1, 2, 2],
        false,
    )
    .unwrap();

    let transitions = compute_hash_effects_with_preconditions(
        &task,
        &mut generator,
        &[],
        op.assignment_effects(),
    )
    .unwrap();

    assert!(transitions.iter().any(|trans| {
        trans
            .source_partition_facts
            .contains(&ExplicitFact::new(0, 0))
            && trans
                .target_partition_facts
                .contains(&ExplicitFact::new(0, 1))
    }));
    assert!(transitions.iter().all(|trans| {
        trans.source_partition_facts.iter().all(|fact| fact.var < 3)
            && trans.target_partition_facts.iter().all(|fact| fact.var < 3)
            && !trans.changed_numeric_vars.contains(&3)
            && !trans.changed_numeric_vars.contains(&4)
    }));
}

#[test]
fn multiply_out_unconditional_propositional_effects() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["0".into(), "1".into()],
        None,
        0,
    )];

    let op = Operator::new(
        "set".into(),
        vec![],
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
        vec![],
        vec![],
        vec![],
        vec![0],
        vec![],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![]);
    let numeric_domain_sizes: Vec<usize> = vec![];

    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        numeric_domain_sizes,
        false,
    )
    .unwrap();
    let abs_ops = generator.build_abstract_operators(&task).unwrap();

    // multiply_out creates one operator for predecessor value 0 -> 1.
    assert_eq!(abs_ops.len(), 1);
    assert_eq!(abs_ops[0].hash_effect, -1);
    assert_eq!(abs_ops[0].preconditions, vec![ExplicitFact::new(0, 0)]);
    assert_eq!(
        abs_ops[0].regression_preconditions,
        vec![ExplicitFact::new(0, 1)]
    );
}

#[test]
fn derived_comparison_precondition_forces_unknown_old_value() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c10".into(), NumericType::Constant, None),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];

    let op = Operator::new(
        "op".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![],
        1,
    );

    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![0],
        vec![0.0, 10.0],
        vec![op],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![Interval::unbounded()],
        vec![Interval::singleton(10.0)],
    ]);

    let numeric_domain_sizes = vec![1, 1];

    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        numeric_domain_sizes,
        false,
    )
    .unwrap();
    let abs_ops = generator.build_abstract_operators(&task).unwrap();
    assert_eq!(abs_ops.len(), 1);
    assert_eq!(abs_ops[0].hash_effect, -2);
    assert_eq!(abs_ops[0].preconditions, vec![ExplicitFact::new(0, 0)]);
    assert_eq!(
        abs_ops[0].regression_preconditions,
        vec![ExplicitFact::new(0, 2)]
    );
}

#[test]
fn metric_tasks_use_metric_delta_for_abstract_operator_cost() {
    let variables: Vec<ExplicitVariable> = vec![];
    let numeric_variables = vec![
        NumericVariable::new("fuel-used".into(), NumericType::Cost, None),
        NumericVariable::new("c5".into(), NumericType::Constant, None),
    ];

    let op = Operator::new(
        "fly".into(),
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
        Metric::new(true, Some(0)),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![],
        vec![0.0, 5.0],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![Interval::unbounded()],
        vec![Interval::singleton(5.0)],
    ]);
    let numeric_domain_sizes = vec![1, 1];

    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        numeric_domain_sizes,
        false,
    )
    .unwrap();
    let abs_ops = generator.build_abstract_operators(&task).unwrap();

    assert_eq!(abs_ops.len(), 1);
    assert_eq!(abs_ops[0].cost, 5.0);
}

#[test]
fn assignment_axiom_chain_can_propagate_through_changed_var_and_constant() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];

    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("c2".into(), NumericType::Constant, None),
    ];

    let assignment_axioms = vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)];
    let comparison_axioms = vec![ComparisonAxiom::new(0, 2, 3, ComparisonOperator::LessThan)];

    let op = Operator::new(
        "inc-x".into(),
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
        vec![],
        vec![],
        vec![0],
        vec![0.0, 1.0, 1.0, 2.0],
        vec![op],
        vec![],
        comparison_axioms,
        assignment_axioms,
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![Interval::singleton(0.0), Interval::singleton(1.0)],
        vec![Interval::singleton(1.0)],
        vec![Interval::singleton(1.0), Interval::singleton(2.0)],
        vec![Interval::singleton(2.0)],
    ]);
    let numeric_domain_sizes = vec![2, 1, 2, 1];

    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        numeric_domain_sizes,
        false,
    )
    .unwrap();
    let abs_ops = generator.build_abstract_operators(&task).unwrap();

    assert!(
        abs_ops
            .iter()
            .any(|op| op.regression_preconditions.iter().any(|fact| fact.var == 0)),
        "abs_ops={abs_ops:#?}"
    );
}

#[test]
fn derived_comparison_transition_is_recomputed_from_tree_snapshots() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];

    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
        NumericVariable::new("y".into(), NumericType::Derived, None),
        NumericVariable::new("c2".into(), NumericType::Constant, None),
    ];

    let assignment_axioms = vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)];
    let comparison_axioms = vec![ComparisonAxiom::new(0, 2, 3, ComparisonOperator::LessThan)];

    let op = Operator::new(
        "inc-x".into(),
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
        vec![],
        vec![],
        vec![0],
        vec![0.0, 1.0, 1.0, 2.0],
        vec![op.clone()],
        vec![],
        comparison_axioms,
        assignment_axioms,
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![Interval::singleton(0.0), Interval::singleton(1.0)],
        vec![Interval::singleton(1.0)],
        vec![Interval::singleton(1.0), Interval::singleton(2.0)],
        vec![Interval::singleton(2.0)],
    ]);

    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        vec![2, 1, 2, 1],
        false,
    )
    .unwrap();

    let transitions = compute_hash_effects_with_preconditions(
        &task,
        &mut generator,
        &[],
        op.assignment_effects(),
    )
    .unwrap();

    assert!(
        transitions.iter().any(|trans| {
            trans
                .source_partition_facts
                .contains(&ExplicitFact::new(0, COMPARISON_TRUE_VAL))
                && trans
                    .target_partition_facts
                    .contains(&ExplicitFact::new(0, COMPARISON_FALSE_VAL))
        }),
        "transitions={transitions:#?}"
    );
}

#[test]
fn derived_comparison_transition_is_skipped_when_target_becomes_unknown() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];

    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c0_5".into(), NumericType::Constant, None),
        NumericVariable::new("y".into(), NumericType::Derived, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
    ];

    let assignment_axioms = vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)];
    let comparison_axioms = vec![ComparisonAxiom::new(0, 2, 3, ComparisonOperator::LessThan)];

    let op = Operator::new(
        "inc-x".into(),
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
        vec![],
        vec![],
        vec![0],
        vec![0.0, 0.5, 0.5, 1.0],
        vec![op.clone()],
        vec![],
        comparison_axioms,
        assignment_axioms,
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::singleton(0.0),
            Interval::new(0.0, 1.0, false, true),
        ],
        vec![Interval::singleton(0.5)],
        vec![
            Interval::singleton(0.5),
            Interval::new(0.5, 1.5, false, true),
        ],
        vec![Interval::singleton(1.0)],
    ]);

    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        vec![2, 1, 2, 1],
        false,
    )
    .unwrap();

    let transitions = compute_hash_effects_with_preconditions(
        &task,
        &mut generator,
        &[],
        op.assignment_effects(),
    )
    .unwrap();

    assert!(
        transitions.iter().all(|trans| {
            trans
                .source_partition_facts
                .iter()
                .chain(trans.target_partition_facts.iter())
                .all(|fact| fact.var != 0)
        }),
        "transitions={transitions:#?}"
    );
}

#[test]
fn combo_interval_build_keeps_missing_derived_operand_unknown_during_propagation() {
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        vec![],
        vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("c0".into(), NumericType::Constant, None),
            NumericVariable::new("ghost".into(), NumericType::Derived, None),
            NumericVariable::new("d".into(), NumericType::Derived, None),
        ],
        vec![],
        vec![],
        vec![],
        vec![5.0, 0.0, 0.0, 0.0],
        vec![],
        vec![],
        vec![],
        vec![AssignmentAxiom::new(3, CalOperator::Product, 2, 1)],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![Interval::singleton(5.0), Interval::singleton(6.0)],
        vec![Interval::singleton(0.0)],
        vec![Interval::unbounded()],
        vec![Interval::singleton(0.0), Interval::unbounded()],
    ]);

    let generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        vec![2, 1, 1, 2],
        false,
    )
    .unwrap();

    let intervals =
        prepare_comparison_tree_inputs_for_combo(&task, &generator, &[(0, 0, 1)], false).unwrap();

    assert_eq!(intervals[0], Interval::singleton(5.0));
    assert_eq!(intervals[1], Interval::singleton(0.0));
    assert_eq!(intervals[2], Interval::unbounded());
    assert_eq!(intervals[3], Interval::unbounded());
}

#[test]
fn duplicate_assignment_effects_use_first_matching_effect() {
    let variables: Vec<ExplicitVariable> = vec![];

    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
        NumericVariable::new("c2".into(), NumericType::Constant, None),
    ];

    let op = Operator::new(
        "dup-ass".into(),
        vec![],
        vec![],
        vec![
            AssignmentEffect::new(0, AssignmentOperation::Plus, 1, false, vec![]),
            AssignmentEffect::new(0, AssignmentOperation::Plus, 2, false, vec![]),
        ],
        1,
    );

    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![],
        vec![0.0, 1.0, 2.0],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::singleton(0.0),
            Interval::singleton(1.0),
            Interval::singleton(2.0),
            Interval::singleton(3.0),
        ],
        vec![Interval::singleton(1.0)],
        vec![Interval::singleton(2.0)],
    ]);
    let numeric_domain_sizes = vec![4, 1, 1];

    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        numeric_domain_sizes,
        false,
    )
    .unwrap();
    let abs_ops = generator.build_abstract_operators(&task).unwrap();

    assert!(!abs_ops.is_empty());
    assert!(abs_ops.iter().any(|op| op.hash_effect == -1));
    assert!(!abs_ops.iter().any(|op| op.hash_effect == -2));
}

#[test]
fn conditional_propositional_effect_branches() {
    let variables = vec![
        ExplicitVariable::new(2, "c".into(), vec!["0".into(), "1".into()], None, 0),
        ExplicitVariable::new(2, "u".into(), vec!["0".into(), "1".into()], None, 0),
        ExplicitVariable::new(2, "v".into(), vec!["0".into(), "1".into()], None, 0),
    ];

    let op = Operator::new(
        "op".into(),
        vec![ExplicitFact::new(1, 0), ExplicitFact::new(2, 0)],
        vec![
            Effect::new(vec![], 1, Some(0), 1),
            Effect::new(vec![ExplicitFact::new(0, 1)], 2, Some(0), 1),
        ],
        vec![],
        1,
    );

    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        vec![],
        vec![],
        vec![],
        vec![0, 0, 0],
        vec![],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![]);
    let numeric_domain_sizes: Vec<usize> = vec![];

    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        numeric_domain_sizes,
        false,
    )
    .unwrap();
    let err = generator.build_abstract_operators(&task).unwrap_err();
    assert!(
        err.to_string()
            .contains("conditional propositional or numeric effects are unsupported")
    );
}

#[test]
fn conditional_assignment_effect_branches() {
    let variables = vec![
        ExplicitVariable::new(2, "c".into(), vec!["0".into(), "1".into()], None, 0),
        ExplicitVariable::new(2, "p".into(), vec!["0".into(), "1".into()], None, 0),
    ];

    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
    ];

    let op = Operator::new(
        "op".into(),
        vec![ExplicitFact::new(1, 0)],
        vec![planners_sas::numeric::numeric_task::Effect::new(
            vec![],
            1,
            Some(0),
            1,
        )],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Plus,
            1,
            true,
            vec![ExplicitFact::new(0, 1)],
        )],
        1,
    );

    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![0, 0],
        vec![0.0, 1.0],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(f64::NEG_INFINITY, 0.0, false, false),
            Interval::new(0.0, f64::INFINITY, true, false),
        ],
        vec![Interval::singleton(1.0)],
    ]);
    let numeric_domain_sizes = vec![2, 1];

    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        numeric_domain_sizes,
        false,
    )
    .unwrap();
    let err = generator.build_abstract_operators(&task).unwrap_err();
    assert!(
        err.to_string()
            .contains("conditional propositional or numeric effects are unsupported")
    );
}

#[test]
fn variable_rhs_assignment_effect_is_rejected_for_parity() {
    let variables: Vec<ExplicitVariable> = vec![];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
    ];

    let op = Operator::new(
        "assign-var".into(),
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
        vec![],
        vec![],
        vec![],
        vec![0.0, 0.0],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![Interval::unbounded()],
        vec![Interval::unbounded()],
    ]);
    let numeric_domain_sizes = vec![1, 1];

    let mut generator = AbstractOperatorGenerator::new_with_identity_mapping(
        &task,
        partitions,
        numeric_domain_sizes,
        false,
    )
    .unwrap();
    let err = generator.build_abstract_operators(&task).unwrap_err();
    assert!(
        err.to_string()
            .contains("assignment effects require constant RHS")
    );
}
