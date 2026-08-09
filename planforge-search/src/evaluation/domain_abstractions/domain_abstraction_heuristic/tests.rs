use super::*;

use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericType,
    NumericVariable,
};

/// Variable 0 carries the comparison `x > one`, variable 1 is an ordinary
/// propositional variable. Numerically, `x = 2.0` and the constant `one = 1.0`,
/// so the comparison holds in the initial state.
fn comparison_task() -> NumericRootTask {
    NumericRootTask::new(
        4,
        Metric::new(true, None),
        vec![
            ExplicitVariable::new(
                3,
                "cmp".into(),
                vec!["true".into(), "false".into(), "unknown".into()],
                None,
                ConditionValue::Unknown.as_usize(),
            ),
            ExplicitVariable::new(2, "p".into(), vec!["p".into(), "not-p".into()], None, 0),
        ],
        vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        vec![],
        vec![],
        vec![ConditionValue::Unknown.as_usize(), 0],
        vec![2.0, 1.0],
        vec![],
        vec![],
        vec![ComparisonAxiom::new(
            0,
            0,
            1,
            ComparisonOperator::GreaterThan,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

#[test]
fn comparison_projection_uses_concrete_value_mapping() {
    let mapping = vec![vec![0, 1, 2]];

    let abs_val = abstract_propositional_value(0, 1, &mapping).unwrap();

    assert_eq!(abs_val, 1);
}

#[test]
fn resolved_propositional_value_recomputes_comparison_axioms_from_numeric_state() {
    let task = comparison_task();

    // The stored value is "not yet derived", so the comparison has to be
    // recomputed from the numeric state: x = 2.0 > one = 1.0.
    let concrete_val = resolved_propositional_value(
        0,
        ConditionValue::Unknown.as_usize(),
        &[2.0, 1.0],
        task.numeric_conditions(),
        None,
    )
    .unwrap();

    assert_eq!(concrete_val, ConditionValue::True.as_usize());
}

#[test]
fn resolved_propositional_value_prefers_supplied_comparison_values() {
    let task = comparison_task();

    // A supplied comparison value wins over the numeric state, which on its
    // own would yield ConditionValue::True.as_usize().
    let concrete_val = resolved_propositional_value(
        0,
        ConditionValue::Unknown.as_usize(),
        &[2.0, 1.0],
        task.numeric_conditions(),
        Some(&[Some(ConditionValue::False.as_usize())]),
    )
    .unwrap();

    assert_eq!(concrete_val, ConditionValue::False.as_usize());
}

#[test]
fn resolved_propositional_value_passes_through_ordinary_variables() {
    let task = comparison_task();

    // Variable 1 carries no comparison, so its stored value is returned as is.
    let concrete_val =
        resolved_propositional_value(1, 1, &[2.0, 1.0], task.numeric_conditions(), None).unwrap();

    assert_eq!(concrete_val, 1);
}
