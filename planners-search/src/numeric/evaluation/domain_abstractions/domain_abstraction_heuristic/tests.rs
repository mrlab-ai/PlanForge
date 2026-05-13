use super::*;

use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator};
use planners_sas::numeric::numeric_task::{
    ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericVariable,
};

#[test]
fn comparison_projection_uses_concrete_value_mapping() {
    let mapping = vec![vec![0, 1, 2]];

    let abs_val = abstract_propositional_value(0, 1, &mapping).unwrap();

    assert_eq!(abs_val, 1);
}

#[test]
fn resolved_propositional_value_recomputes_comparison_axioms_from_numeric_state() {
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            3,
            "cmp".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            None,
            2,
        )],
        vec![
            NumericVariable::new(
                "x".into(),
                planners_sas::numeric::numeric_task::NumericType::Regular,
                None,
            ),
            NumericVariable::new(
                "one".into(),
                planners_sas::numeric::numeric_task::NumericType::Constant,
                None,
            ),
        ],
        vec![],
        vec![],
        vec![2],
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
    );

    let tree = ComparisonTree::from_task(&task, 0).unwrap();

    let tree_by_var = vec![Some(0)];
    let tree_lens = vec![comparison_tree_numeric_len(&tree)];
    let concrete_val =
        resolved_propositional_value(0, 2, &[2.0, 1.0], &[tree], &tree_by_var, &tree_lens, None)
            .unwrap();

    assert_eq!(concrete_val, COMPARISON_TRUE_VAL);
}
