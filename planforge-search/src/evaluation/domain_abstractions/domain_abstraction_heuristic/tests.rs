use super::*;

use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, NumericType, NumericVariable, VariableIndex,
};

/// Variable 0 carries the comparison `x > one`, variable 1 is an ordinary
/// propositional variable. Numerically, `x = 2.0` and the constant `one = 1.0`,
/// so the comparison holds in the initial state.
fn comparison_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![
            ExplicitVariable::new(
                ConditionValue::DOMAIN_SIZE,
                "cmp".into(),
                vec!["true".into(), "false".into()],
                Some(0),
                ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
            ),
            ExplicitVariable::new(
                2,
                "p".into(),
                vec!["p".into(), "not-p".into()],
                None,
                ExplicitValueIndex::new(0),
            ),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        goals: vec![],
        mutexes: vec![],
        state: vec![
            ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
            ExplicitValueIndex::new(0),
        ],
        numeric_state: vec![NumericValue::new(2.0), NumericValue::new(1.0)],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(0),
            VariableIndex::new(1),
            ComparisonOperator::GreaterThan,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

#[test]
fn comparison_projection_uses_concrete_value_mapping() {
    let mapping = vec![vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)]];

    let abs_val = abstract_propositional_value(
        VariableIndex::from_usize(0),
        ExplicitValueIndex::new(1),
        &mapping,
    )
    .unwrap();

    assert_eq!(abs_val, ExplicitValueIndex::new(1));
}

#[test]
fn resolved_propositional_value_recomputes_comparison_axioms_from_numeric_state() {
    let task = comparison_task();

    // The stored value is ignored for a condition variable: the comparison is
    // recomputed from the numeric state, where x = 2.0 > one = 1.0.
    let concrete_val = resolved_propositional_value(
        VariableIndex::from_usize(0),
        ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
        &[NumericValue::new(2.0), NumericValue::new(1.0)],
        task.numeric_conditions(),
        None,
    )
    .unwrap();

    assert_eq!(
        concrete_val,
        ExplicitValueIndex::from_usize(ConditionValue::True.as_usize())
    );
}

#[test]
fn resolved_propositional_value_prefers_supplied_comparison_values() {
    let task = comparison_task();

    // A supplied comparison value wins over the numeric state, which on its
    // own would yield ConditionValue::True.as_usize().
    let concrete_val = resolved_propositional_value(
        VariableIndex::from_usize(0),
        ExplicitValueIndex::from_usize(ConditionValue::True.as_usize()),
        &[NumericValue::new(2.0), NumericValue::new(1.0)],
        task.numeric_conditions(),
        Some(&[Some(ExplicitValueIndex::from_usize(
            ConditionValue::False.as_usize(),
        ))]),
    )
    .unwrap();

    assert_eq!(
        concrete_val,
        ExplicitValueIndex::from_usize(ConditionValue::False.as_usize())
    );
}

#[test]
fn resolved_propositional_value_passes_through_ordinary_variables() {
    let task = comparison_task();

    // Variable 1 carries no comparison, so its stored value is returned as is.
    let concrete_val = resolved_propositional_value(
        VariableIndex::from_usize(1),
        ExplicitValueIndex::new(1),
        &[NumericValue::new(2.0), NumericValue::new(1.0)],
        task.numeric_conditions(),
        None,
    )
    .unwrap();

    assert_eq!(concrete_val, ExplicitValueIndex::new(1));
}
