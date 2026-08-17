use super::*;

use planforge_sas::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts, NumericType,
    NumericVariable,
};

/// The propositional variable a comparison axiom writes: true or false,
/// defaulting to false until the closure derives it.
fn condition_variable(name: &str, layer: usize) -> ExplicitVariable {
    ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        name.into(),
        vec!["true".into(), "false".into()],
        Some(layer),
        ConditionValue::False.as_usize(),
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

    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![condition_variable("x0 < c1", 0)],
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ConditionValue::False.as_usize()],
        numeric_state: vec![0.0, 10.0],
        operators: vec![],
        axioms: vec![],
        comparison_axioms,
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let conditions = task.numeric_conditions();

    // x0 in [0, 5], c1 is exactly 10
    let intervals = [Interval::closed(0.0, 5.0), Interval::singleton(10.0)];

    // Every value of x0 is below c1, so requiring the comparison to hold is
    // satisfiable and requiring it to fail is not.
    let requires_true = ExplicitFact::propositional(0, ConditionValue::True.as_usize());
    let requires_false = ExplicitFact::propositional(0, ConditionValue::False.as_usize());
    assert!(!conditions.precondition_is_contradicted(&requires_true, &intervals));
    assert!(conditions.precondition_is_contradicted(&requires_false, &intervals));

    // Undecided case: x0 in [0, 20] straddles c1, so both outcomes remain
    // possible and neither requirement is contradicted.
    let intervals = [Interval::closed(0.0, 20.0), Interval::singleton(10.0)];
    assert!(!conditions.precondition_is_contradicted(&requires_true, &intervals));
    assert!(!conditions.precondition_is_contradicted(&requires_false, &intervals));
}

#[test]
fn reachable_partitions_overlaps_result_interval() {
    // Two partitions: (-inf, 9) and [9, inf)
    let parts = vec![vec![
        Interval::new(f64::NEG_INFINITY, 9.0, false, false),
        Interval::new(9.0, f64::INFINITY, true, false),
    ]];

    let dummy_task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            1,
            "global-constraint".into(),
            vec!["true".into()],
            None,
            0,
        )],
        numeric_variables: vec![NumericVariable::new(
            "x0".into(),
            NumericType::Regular,
            None,
        )],
        goals: vec![],
        mutexes: vec![],
        state: vec![0],
        numeric_state: vec![0.0],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::with_partitions(parts);

    // From partition 0: (-inf,9) + 7 -> (-inf,16) overlaps both partitions.
    let targets = partitions.reachable_partitions(
        0,
        0,
        &planforge_sas::numeric_task::AssignmentOperation::Plus,
        Interval::singleton(7.0),
    );
    assert_eq!(targets, vec![0, 1]);

    // From partition 1: [9,inf) + 7 -> [16,inf) overlaps only partition 1.
    let targets = partitions.reachable_partitions(
        0,
        1,
        &planforge_sas::numeric_task::AssignmentOperation::Plus,
        Interval::singleton(7.0),
    );
    assert_eq!(targets, vec![1]);

    // Silence unused dummy_task while keeping construction pattern consistent.
    let _ = dummy_task.metric();
}

#[test]
fn reachable_partitions_use_the_numeric_state_lattice() {
    let mut partitions = NumericPartitions::with_partitions(vec![vec![Interval::unbounded()]]);
    assert!(partitions.split_at(0, -5.9999999999999964, true));
    assert!(partitions.split_at(0, -5.799999999999997, true));
    assert!(partitions.split_at(0, -5.6999999999999975, true));

    assert_eq!(
        partitions.partitions(0).unwrap(),
        &[
            Interval::new(f64::NEG_INFINITY, -6.0, false, true),
            Interval::new(-6.0, -5.8, false, true),
            Interval::new(-5.8, -5.7, false, true),
            Interval::new(-5.7, f64::INFINITY, false, false),
        ]
    );

    // Concrete execution canonicalizes (-5.8, -5.7] - 0.2 to
    // (-6.0, -5.9]. It therefore cannot enter the partition ending at -6.0.
    assert_eq!(
        partitions.reachable_partitions(
            0,
            2,
            &planforge_sas::numeric_task::AssignmentOperation::Plus,
            Interval::singleton(-0.2),
        ),
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
            0,
        )],
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![0],
        numeric_state: vec![0.0, 7.0],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::trivial(&task);

    assert_eq!(partitions.partitions(0).unwrap(), &[Interval::unbounded()]);
    assert_eq!(
        partitions.partitions(1).unwrap(),
        &[Interval::singleton(7.0)]
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
            0,
        )],
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![0],
        numeric_state: vec![9.450000000000001],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::trivial(&task);

    assert_eq!(
        partitions.partitions(0).unwrap(),
        &[Interval::singleton(9.45)]
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
    let assignment_axioms = vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)];

    // d2 == x0
    let comparison_axioms = vec![ComparisonAxiom::new(0, 2, 0, ComparisonOperator::Equal)];

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![condition_variable("d2 == x0", 1)],
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ConditionValue::False.as_usize()],
        numeric_state: vec![0.0; 3],
        operators: vec![],
        axioms: vec![],
        comparison_axioms,
        assignment_axioms,
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    // The derived variable is expanded into the condition's DAG instead of
    // being read from the state, so the condition depends on x0 and x1 only.
    let condition = task
        .numeric_conditions()
        .for_var(0)
        .expect("var 0 carries the comparison's truth value");
    assert_eq!(condition.regular_numeric_var_dependencies(), &[0, 1]);
    assert_eq!(condition.required_numeric_len(), 3);
}
