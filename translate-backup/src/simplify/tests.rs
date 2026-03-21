use super::{get_applicability_conditions, rebuild_canonical};
use crate::sas_tasks::{SASOperator, SASTask, Variable};

#[test]
fn applicability_conditions_skip_no_precondition_sentinel() {
    let op = SASOperator {
        name: "op".to_string(),
        prevails: vec![(2, 1)],
        effects: vec![(0, usize::MAX, 1, vec![]), (1, 0, 1, vec![])],
        numeric_effects: vec![],
        cost: 0.0,
    };

    let conditions = get_applicability_conditions(&op);

    assert_eq!(conditions, vec![(2, 1), (1, 0)]);
}

#[test]
fn rebuild_canonical_uses_none_for_no_precondition() {
    let mut task = SASTask {
        variables: vec![Variable {
            value_names: vec!["a".to_string(), "b".to_string()],
        }],
        operators: vec![SASOperator {
            name: "op".to_string(),
            prevails: vec![],
            effects: vec![(0, usize::MAX, 1, vec![])],
            numeric_effects: vec![],
            cost: 0.0,
        }],
        numeric_variables: vec![],
        numeric_axioms: vec![],
        comparison_axioms: vec![],
        axioms: vec![],
        numeric_init: vec![],
        mutex_groups: vec![],
        ranges: vec![2],
        axiom_layers: vec![-1],
        init: vec![0],
        goal: vec![(0, 1)],
        translation_key: vec![vec!["a".to_string(), "b".to_string()]],
        canonical_variables: vec![],
        canonical_operators: vec![],
        canonical_metric: None,
        metric: ("<".to_string(), -1),
        global_constraint: None,
        comp_axiom_layer: -1,
    };

    rebuild_canonical(&mut task);

    assert_eq!(task.canonical_operators[0].pre_post[0].pre, None);
}
