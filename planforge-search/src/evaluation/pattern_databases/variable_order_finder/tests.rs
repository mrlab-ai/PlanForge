use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_task::{
    ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericType, NumericVariable,
};

use super::*;

fn simple_var(name: &str, axiom_layer: Option<usize>) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        axiom_layer,
        1,
    )
}

#[test]
fn default_matches_fd_default() {
    assert_eq!(
        GreedyVariableOrderType::default(),
        GreedyVariableOrderType::GoalCgLevel
    );
}

#[test]
fn goal_cg_level_prefers_goal_numeric_variables() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("cmp", Some(0))],
        vec![
            NumericVariable::new("threshold".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![1.0, 0.0],
        vec![],
        vec![],
        vec![ComparisonAxiom::new(
            0,
            1,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let mut order = VariableOrderFinder::new(&task, GreedyVariableOrderType::GoalCgLevel, true, 0);

    let next = order.next();
    assert_eq!(next, Some((1, true)));
}
