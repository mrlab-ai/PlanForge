use super::*;

use planners_sas::numeric::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator,
};
use planners_sas::numeric::numeric_task::{
    ExplicitFact, Metric, NumericRootTask, NumericType, NumericVariable,
};

#[test]
fn comparison_tree_interval_evaluates_definitely_and_unknown() {
    // numeric vars: x0 (regular), c1 (constant)
    // cmp: x0 < c1 (affected var id 0)
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
    ];

    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];

    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        vec![],
        numeric_variables,
        vec![],
        vec![],
        vec![],
        vec![0.0, 10.0],
        vec![],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    let index = ComparisonAxiomIndex::from_task(&task).unwrap();

    // x0 in [0, 5], c1 is exactly 10
    let intervals = [Interval::closed(0.0, 5.0), Interval::singleton(10.0)];

    // precondition var0==0 means comparison is true (we store !result)
    assert_eq!(
        index.precondition_is_contradicted(&ExplicitFact::new(0, 0), &intervals),
        false
    );
    assert_eq!(
        index.precondition_is_contradicted(&ExplicitFact::new(0, 1), &intervals),
        true
    );

    // Unknown case: x0 in [0, 20]
    let intervals = [Interval::closed(0.0, 20.0), Interval::singleton(10.0)];
    assert_eq!(
        index.precondition_is_contradicted(&ExplicitFact::new(0, 0), &intervals),
        false
    );
    assert_eq!(
        index.precondition_is_contradicted(&ExplicitFact::new(0, 1), &intervals),
        false
    );
    assert_eq!(
        index.precondition_is_contradicted(&ExplicitFact::new(0, 2), &intervals),
        false
    );
}

#[test]
fn reachable_partitions_overlaps_result_interval() {
    // Two partitions: (-inf, 9) and [9, inf)
    let parts = vec![vec![
        Interval::new(f64::NEG_INFINITY, 9.0, false, false),
        Interval::new(9.0, f64::INFINITY, true, false),
    ]];

    let dummy_task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        vec![],
        vec![NumericVariable::new(
            "x0".into(),
            NumericType::Regular,
            None,
        )],
        vec![],
        vec![],
        vec![],
        vec![0.0],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(parts);

    // From partition 0: (-inf,9) + 7 -> (-inf,16) overlaps both partitions.
    let targets = partitions.reachable_partitions(
        0,
        0,
        &planners_sas::numeric::numeric_task::AssignmentOperation::Plus,
        Interval::singleton(7.0),
    );
    assert_eq!(targets, vec![0, 1]);

    // From partition 1: [9,inf) + 7 -> [16,inf) overlaps only partition 1.
    let targets = partitions.reachable_partitions(
        0,
        1,
        &planners_sas::numeric::numeric_task::AssignmentOperation::Plus,
        Interval::singleton(7.0),
    );
    assert_eq!(targets, vec![1]);

    // Silence unused dummy_task while keeping construction pattern consistent.
    let _ = dummy_task.metric();
}

#[test]
fn trivial_partitions_use_singletons_for_constants() {
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c7".into(), NumericType::Constant, None),
    ];

    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        vec![],
        numeric_variables,
        vec![],
        vec![],
        vec![],
        vec![0.0, 7.0],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::trivial(&task);

    assert_eq!(partitions.partitions(0).unwrap(), &[Interval::unbounded()]);
    assert_eq!(
        partitions.partitions(1).unwrap(),
        &[Interval::singleton(7.0)]
    );
}

#[test]
fn comparison_tree_index_can_build_for_assignment_axioms() {
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("x1".into(), NumericType::Regular, None),
        NumericVariable::new("d2".into(), NumericType::Derived, None),
    ];

    // d2 = x0 + x1
    let assignment_axioms = vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)];

    // d2 == x0
    let comparison_axioms = vec![ComparisonAxiom::new(0, 2, 0, ComparisonOperator::Equal)];

    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        vec![],
        numeric_variables,
        vec![],
        vec![],
        vec![],
        vec![0.0; 3],
        vec![],
        vec![],
        comparison_axioms,
        assignment_axioms,
        ExplicitFact::new(0, 0),
    );

    let _ = ComparisonAxiomIndex::from_task(&task).unwrap();
}
