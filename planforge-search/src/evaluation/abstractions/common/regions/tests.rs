use planforge_sas::numeric_task::{
    ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, Operator, OperatorCost, OperatorIndex,
};

use super::*;

/// A position that is `left` or `right`, and an operator that requires
/// `left` and changes nothing.
fn prevail_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            2,
            "position".into(),
            vec!["left".into(), "right".into()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![Operator::new(
            "prevail-left".into(),
            vec![ExplicitFact::propositional(0, 0)],
            vec![],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

#[test]
fn a_prevail_condition_fixes_the_successor_value() {
    let task = prevail_task();
    let regions = RegionSemantics::new(&task).unwrap();
    let both = CartesianSet::with_all_props_constrained(
        vec![vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)]],
        vec![],
    );
    let left =
        CartesianSet::with_all_props_constrained(vec![vec![ExplicitValueIndex::new(0)]], vec![]);
    let right =
        CartesianSet::with_all_props_constrained(vec![vec![ExplicitValueIndex::new(1)]], vec![]);
    let op = OperatorIndex::new(0);

    assert!(regions.may_transition(&both, op, &left).unwrap());
    assert!(
        !regions.may_transition(&both, op, &right).unwrap(),
        "the operator requires and keeps `left`"
    );
    assert!(!regions.may_transition(&right, op, &left).unwrap());
}
