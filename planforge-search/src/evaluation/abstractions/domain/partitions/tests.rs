use super::*;
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::value_lattice::ValueLattice;

use planforge_sas::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    ExplicitFact, ExplicitValueIndex, ExplicitVariable, INF_VALUE, Metric, NEG_INF_VALUE,
    NumericRootTask, NumericRootTaskParts, NumericType, NumericVariable, ZERO_VALUE,
};

/// The propositional variable a comparison axiom writes: true or false,
/// defaulting to false until the closure derives it.
fn condition_variable(name: &str, layer: usize) -> ExplicitVariable {
    ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        name.into(),
        vec!["true".into(), "false".into()],
        Some(layer),
        ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
    )
}

#[test]
fn comparison_tree_interval_evaluates_definitely_and_undecided() {
    // numeric vars: x0 (regular), c1 (constant)
    // cmp: x0 < c1 (affected var id 0)
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
    ];

    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(0),
        VariableIndex::new(1),
        ComparisonOperator::LessThan,
    )];

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![condition_variable("x0 < c1", 0)],
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::from_usize(
            ConditionValue::False.as_usize(),
        )],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(10.0)],
        operators: vec![],
        axioms: vec![],
        comparison_axioms,
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let conditions = task.numeric_conditions();

    // x0 in [0, 5], c1 is exactly 10
    let intervals = [
        Interval::closed(NumericValue::new(0.0), NumericValue::new(5.0)),
        Interval::singleton(NumericValue::new(10.0)),
    ];

    // Every value of x0 is below c1, so requiring the comparison to hold is
    // satisfiable and requiring it to fail is not.
    let requires_true = ExplicitFact::propositional(0, ConditionValue::True.as_usize());
    let requires_false = ExplicitFact::propositional(0, ConditionValue::False.as_usize());
    assert!(!conditions.precondition_is_contradicted(&requires_true, &intervals));
    assert!(conditions.precondition_is_contradicted(&requires_false, &intervals));

    // Undecided case: x0 in [0, 20] straddles c1, so both outcomes remain
    // possible and neither requirement is contradicted.
    let intervals = [
        Interval::closed(NumericValue::new(0.0), NumericValue::new(20.0)),
        Interval::singleton(NumericValue::new(10.0)),
    ];
    assert!(!conditions.precondition_is_contradicted(&requires_true, &intervals));
    assert!(!conditions.precondition_is_contradicted(&requires_false, &intervals));
}

#[test]
fn reachable_partitions_overlaps_result_interval() {
    // Two partitions: (-inf, 9) and [9, inf)
    let parts = vec![vec![
        Interval::new(NEG_INF_VALUE, NumericValue::new(9.0), false, false),
        Interval::new(NumericValue::new(9.0), INF_VALUE, true, false),
    ]];

    let dummy_task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            1,
            "global-constraint".into(),
            vec!["true".into()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables: vec![NumericVariable::new(
            "x0".into(),
            NumericType::Regular,
            None,
        )],
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0)],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::with_partitions(parts);

    // From partition 0: (-inf,9) + 7 -> (-inf,16) overlaps both partitions.
    let targets = partitions.intersecting(
        VariableIndex::from_usize(0),
        partitions.partition_interval(VariableIndex::from_usize(0), 0)
            + Interval::singleton(NumericValue::new(7.0)),
    );
    assert_eq!(targets.collect::<Vec<_>>(), vec![0, 1]);

    // From partition 1: [9,inf) + 7 -> [16,inf) overlaps only partition 1.
    let targets = partitions.intersecting(
        VariableIndex::from_usize(0),
        partitions.partition_interval(VariableIndex::from_usize(0), 1)
            + Interval::singleton(NumericValue::new(7.0)),
    );
    assert_eq!(targets.collect::<Vec<_>>(), vec![1]);

    let _ = dummy_task.metric();
}

#[test]
fn partition_images_use_the_numeric_state_lattice() {
    let mut partitions = NumericPartitions::with_partitions(vec![vec![Interval::unbounded()]]);
    assert!(partitions.split_at(
        VariableIndex::from_usize(0),
        NumericValue::new(-5.9999999999999964),
        true,
        ValueLattice::GRID
    ));
    assert!(partitions.split_at(
        VariableIndex::from_usize(0),
        NumericValue::new(-5.799999999999997),
        true,
        ValueLattice::GRID
    ));
    assert!(partitions.split_at(
        VariableIndex::from_usize(0),
        NumericValue::new(-5.6999999999999975),
        true,
        ValueLattice::GRID
    ));

    assert_eq!(
        partitions.partitions(VariableIndex::from_usize(0)),
        &[
            Interval::new(NEG_INF_VALUE, NumericValue::new(-6.0), false, true),
            Interval::new(
                NumericValue::new(-6.0 + 1e-12),
                NumericValue::new(-5.8),
                true,
                true
            ),
            Interval::new(
                NumericValue::new(-5.8 + 1e-12),
                NumericValue::new(-5.7),
                true,
                true
            ),
            Interval::new(NumericValue::new(-5.7 + 1e-12), INF_VALUE, true, false),
        ]
    );

    // Concrete execution canonicalizes (-5.8, -5.7] - 0.2 to
    // (-6.0, -5.9]. It therefore cannot enter the partition ending at -6.0.
    assert_eq!(
        partitions
            .intersecting(
                VariableIndex::from_usize(0),
                partitions.partition_interval(VariableIndex::from_usize(0), 2)
                    + Interval::singleton(NumericValue::new(-0.2)),
            )
            .collect::<Vec<_>>(),
        vec![1]
    );
}

#[test]
fn trivial_partitions_use_singletons_for_constants() {
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c7".into(), NumericType::Constant, None),
    ];

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            1,
            "global-constraint".into(),
            vec!["true".into()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(7.0)],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::trivial(&task);

    assert_eq!(
        partitions.partitions(VariableIndex::from_usize(0)),
        &[Interval::unbounded()]
    );
    assert_eq!(
        partitions.partitions(VariableIndex::from_usize(1)),
        &[Interval::singleton(NumericValue::new(7.0))]
    );
}

#[test]
fn trivial_constant_partitions_use_canonical_initial_values() {
    let numeric_variables = vec![NumericVariable::new(
        "c9_45".into(),
        NumericType::Constant,
        None,
    )];

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            1,
            "global-constraint".into(),
            vec!["true".into()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(9.450000000000001)],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::trivial(&task);

    assert_eq!(
        partitions.partitions(VariableIndex::from_usize(0)),
        &[Interval::singleton(NumericValue::new(9.45))]
    );
}

#[test]
fn comparison_tree_index_can_build_for_assignment_axioms() {
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("x1".into(), NumericType::Regular, None),
        NumericVariable::new("d2".into(), NumericType::Derived, Some(0)),
    ];

    // d2 = x0 + x1
    let assignment_axioms = vec![AssignmentAxiom::new(
        VariableIndex::from_usize(2),
        CalOperator::Sum,
        VariableIndex::from_usize(0),
        VariableIndex::from_usize(1),
    )];

    // d2 == x0
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(2),
        VariableIndex::new(0),
        ComparisonOperator::Equal,
    )];

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![condition_variable("d2 == x0", 1)],
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::from_usize(
            ConditionValue::False.as_usize(),
        )],
        numeric_state: vec![ZERO_VALUE; 3],
        operators: vec![],
        axioms: vec![],
        comparison_axioms,
        assignment_axioms,
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    // x0 + x1 == x0 becomes the direct threshold x1 == 0.
    let condition = task
        .numeric_conditions()
        .for_var(VariableIndex::from_usize(0))
        .expect("var 0 carries the comparison's truth value");
    assert_eq!(
        condition.regular_numeric_var_dependencies(),
        &[VariableIndex::from_usize(1)]
    );
    assert_eq!(condition.required_numeric_len(), 3);
}

fn nv(value: f64) -> NumericValue {
    NumericValue::new(value)
}

/// `count` adjacent partitions with integer boundaries `0, 1, ..., count - 2`,
/// unbounded at both ends, alternating which side owns each boundary.
fn layout(count: usize) -> Vec<Interval> {
    assert!(count >= 2);
    let boundary = |index: usize| nv(index as f64);
    // Boundary `b` belongs to the lower partition when `b` is even.
    let lower_owns = |index: usize| index.is_multiple_of(2);
    let mut partitions = vec![Interval::new(
        NEG_INF_VALUE,
        boundary(0),
        false,
        lower_owns(0),
    )];
    for index in 1..count - 1 {
        partitions.push(Interval::new(
            boundary(index - 1),
            boundary(index),
            !lower_owns(index - 1),
            lower_owns(index),
        ));
    }
    partitions.push(Interval::new(
        boundary(count - 2),
        INF_VALUE,
        !lower_owns(count - 2),
        false,
    ));
    partitions
}

#[test]
fn boundary_values_map_by_their_closed_side_for_every_partition_count() {
    for count in [2, 3, 5, 8, 9, 17, 64, 257] {
        let partitions = layout(count);
        for index in 0..count - 1 {
            let boundary = index as f64;
            let expected = if index.is_multiple_of(2) {
                index
            } else {
                index + 1
            };
            assert_eq!(
                partition_for_value(&partitions, nv(boundary)),
                Some(expected),
                "{count} partitions, boundary {boundary}"
            );
            // Just inside either neighbour.
            assert_eq!(
                partition_for_value(&partitions, nv(boundary - 0.5)),
                Some(index)
            );
            assert_eq!(
                partition_for_value(&partitions, nv(boundary + 0.5)),
                Some(index + 1)
            );
        }
    }
}

#[test]
fn a_boundary_value_maps_to_the_same_partition_in_small_and_large_layouts() {
    let small = layout(4);
    let large = layout(300);
    for boundary in [0.0, 1.0, 2.0] {
        assert_eq!(
            partition_for_value(&small, nv(boundary)),
            partition_for_value(&large, nv(boundary)),
            "boundary {boundary}"
        );
    }
}

#[test]
fn values_are_compared_on_the_canonical_lattice_without_tolerance() {
    let partitions = vec![
        Interval::new(NEG_INF_VALUE, nv(0.3), false, false),
        Interval::new(nv(0.3), INF_VALUE, true, false),
    ];
    // 0.1 + 0.2 is 0.30000000000000004: the same lattice point as 0.3.
    assert_eq!(partition_for_value(&partitions, nv(0.1 + 0.2)), Some(1));
    assert_eq!(partition_for_value(&partitions, nv(0.3)), Some(1));
    // One lattice step below the boundary is below it.
    let below = nv(0.3 - float_tolerance::ABS_EPSILON);
    assert_eq!(partition_for_value(&partitions, below), Some(0));
}

#[test]
fn values_outside_a_bounded_layout_have_no_partition() {
    let partitions = vec![Interval::singleton(nv(2.0))];
    assert_eq!(partition_for_value(&partitions, nv(2.0)), Some(0));
    assert_eq!(partition_for_value(&partitions, nv(3.0)), None);
}
