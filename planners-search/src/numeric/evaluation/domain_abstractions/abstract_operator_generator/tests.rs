use super::*;

use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator};
use planners_sas::numeric::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitVariable, Fact, Metric, NumericRootTask,
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
        0,
        2,
    )];

    // Numeric vars: x0 (regular), c10 (constant), c7 (constant)
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, -1),
        NumericVariable::new("c10".into(), NumericType::Constant, -1),
        NumericVariable::new("c7".into(), NumericType::Constant, -1),
    ];

    // Comparison axiom (derived propositional var 0): x0 < c10
    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];

    // Operator requires comparison to hold (var0 == 0) and applies x0 += c7.
    let op = Operator::new(
        "op".into(),
        vec![Fact::new(0, 0)],
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
        Metric::new(true, -1),
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
        (0, 0),
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
fn numeric_transition_adds_implicit_comparison_preconditions() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        0,
        2,
    )];

    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, -1),
        NumericVariable::new("c10".into(), NumericType::Constant, -1),
        NumericVariable::new("c7".into(), NumericType::Constant, -1),
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
        Metric::new(true, -1),
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
        (0, 0),
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
        vec![Fact::new(0, 0), Fact::new(1, 0)]
    );
    assert_eq!(
        abs_ops[0].regression_preconditions,
        vec![Fact::new(0, 2), Fact::new(1, 1)]
    );

    assert_eq!(
        abs_ops[1].preconditions,
        vec![Fact::new(0, 0), Fact::new(1, 0)]
    );
    assert_eq!(
        abs_ops[1].regression_preconditions,
        vec![Fact::new(0, 2), Fact::new(1, 0)]
    );

    assert_eq!(
        abs_ops[2].preconditions,
        vec![Fact::new(0, 1), Fact::new(1, 1)]
    );
    assert_eq!(
        abs_ops[2].regression_preconditions,
        vec![Fact::new(0, 2), Fact::new(1, 1)]
    );
}

#[test]
fn multiply_out_unconditional_propositional_effects() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["0".into(), "1".into()],
        -1,
        0,
    )];

    let op = Operator::new(
        "set".into(),
        vec![],
        vec![planners_sas::numeric::numeric_task::Effect::new(
            vec![],
            0,
            0,
            1,
        )],
        vec![],
        1,
    );

    let task = NumericRootTask::new(
        4,
        Metric::new(true, -1),
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
        (0, 0),
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
    assert_eq!(abs_ops[0].preconditions, vec![Fact::new(0, 0)]);
    assert_eq!(abs_ops[0].regression_preconditions, vec![Fact::new(0, 1)]);
}

#[test]
fn derived_comparison_precondition_forces_unknown_old_value() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        0,
        2,
    )];
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, -1),
        NumericVariable::new("c10".into(), NumericType::Constant, -1),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];

    let op = Operator::new("op".into(), vec![Fact::new(0, 0)], vec![], vec![], 1);

    let task = NumericRootTask::new(
        4,
        Metric::new(true, -1),
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
        (0, 0),
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
    assert_eq!(abs_ops[0].preconditions, vec![Fact::new(0, 0)]);
    assert_eq!(abs_ops[0].regression_preconditions, vec![Fact::new(0, 2)]);
}

#[test]
fn conditional_propositional_effect_branches() {
    let variables = vec![
        ExplicitVariable::new(2, "c".into(), vec!["0".into(), "1".into()], -1, 0),
        ExplicitVariable::new(2, "u".into(), vec!["0".into(), "1".into()], -1, 0),
        ExplicitVariable::new(2, "v".into(), vec!["0".into(), "1".into()], -1, 0),
    ];

    let op = Operator::new(
        "op".into(),
        vec![Fact::new(1, 0), Fact::new(2, 0)],
        vec![
            planners_sas::numeric::numeric_task::Effect::new(vec![], 1, 0, 1),
            planners_sas::numeric::numeric_task::Effect::new(vec![Fact::new(0, 1)], 2, 0, 1),
        ],
        vec![],
        1,
    );

    let task = NumericRootTask::new(
        4,
        Metric::new(true, -1),
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
        (0, 0),
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
    let mut abs_ops = generator.build_abstract_operators(&task).unwrap();
    abs_ops.sort_by_key(|o| o.hash_effect);

    assert_eq!(abs_ops.len(), 2);
    assert_eq!(abs_ops[0].hash_effect, -6);
    assert_eq!(
        abs_ops[0].preconditions,
        vec![Fact::new(0, 1), Fact::new(1, 0), Fact::new(2, 0)]
    );
    assert_eq!(abs_ops[1].hash_effect, -2);
    assert_eq!(
        abs_ops[1].preconditions,
        vec![Fact::new(1, 0), Fact::new(2, 0)]
    );
}

#[test]
fn conditional_assignment_effect_branches() {
    let variables = vec![
        ExplicitVariable::new(2, "c".into(), vec!["0".into(), "1".into()], -1, 0),
        ExplicitVariable::new(2, "p".into(), vec!["0".into(), "1".into()], -1, 0),
    ];

    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, -1),
        NumericVariable::new("c1".into(), NumericType::Constant, -1),
    ];

    let op = Operator::new(
        "op".into(),
        vec![Fact::new(1, 0)],
        vec![planners_sas::numeric::numeric_task::Effect::new(
            vec![],
            1,
            0,
            1,
        )],
        vec![AssignmentEffect::new(
            0,
            AssignmentOperation::Plus,
            1,
            true,
            vec![Fact::new(0, 1)],
        )],
        1,
    );

    let task = NumericRootTask::new(
        4,
        Metric::new(true, -1),
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
        (0, 0),
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
    let abs_ops = generator.build_abstract_operators(&task).unwrap();

    // Non-apply branch yields identity transitions for refined numeric vars.
    // Here n0 has 2 partitions, so the non-apply branch yields 2 abstract operators.
    // Apply branch yields three numeric transitions (0->0, 0->1, 1->1), each with condition c==1.
    assert_eq!(abs_ops.len(), 5);
    let with_cond: Vec<&AbstractOperator> = abs_ops
        .iter()
        .filter(|o| o.preconditions.contains(&Fact::new(0, 1)))
        .collect();
    assert_eq!(with_cond.len(), 3);
    assert_eq!(
        with_cond
            .iter()
            .filter(|o| o.changed_numeric_vars == vec![0])
            .count(),
        1
    );
}
