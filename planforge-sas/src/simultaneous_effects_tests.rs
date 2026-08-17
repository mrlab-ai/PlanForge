//! All numeric effects of one operator apply simultaneously.
//!
//! Every right-hand side, left-hand side and effect condition reads the state
//! as it was *before* the operator was applied, so the stored order of an
//! operator's effects can never change the successor. Each test below fixes an
//! effect order under which sequential application would give a different
//! answer, so a regression fails rather than silently reordering.

use crate::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitVariable,
    Metric, NumericRootTask, NumericRootTaskParts, NumericType, NumericVariable, Operator, TaskRef,
};
use crate::state_registry::StateRegistry;
use std::sync::Arc;

const X: usize = 0;
const Y: usize = 1;
const ONE: usize = 2;
const COST: usize = 3;

/// Numeric variables: `x = 10` and `y = 3` (both regular), the constant `1`,
/// and a cost variable starting at `0`. One propositional variable `p`, false
/// in the initial state.
fn task_with_effects(effects: Vec<AssignmentEffect>, metric_var: Option<usize>) -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, metric_var),
        variables: vec![ExplicitVariable::new(
            2,
            "p".into(),
            vec!["p-false".into(), "p-true".into()],
            None,
            0,
        )],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("total_cost".into(), NumericType::Cost, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![0],
        numeric_state: vec![10.0, 3.0, 1.0, 0.0],
        operators: vec![Operator::new("op".into(), vec![], vec![], effects, 1)],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

fn successor_values(effects: Vec<AssignmentEffect>) -> Vec<f64> {
    let task: TaskRef = Arc::new(task_with_effects(effects, None));
    let mut registry = StateRegistry::for_task(task.clone());
    let initial = registry.get_initial_state();
    let successor = registry
        .get_successor_state(&initial, &task.get_operators()[0])
        .expect("operator applies to the initial state");
    registry
        .get_numeric_vars(&successor)
        .expect("successor has numeric values")
}

fn plus(affected: usize, rhs: usize) -> AssignmentEffect {
    AssignmentEffect::new(affected, AssignmentOperation::Plus, rhs, false, vec![])
}

#[test]
fn regular_right_hand_side_reads_the_parent_value() {
    // `y += 1` is stored first, so sequential application would feed the new
    // `y` into `x`: x = 10 + 4 = 14 instead of 10 + 3 = 13.
    let values = successor_values(vec![plus(Y, ONE), plus(X, Y)]);

    assert_eq!(values[Y], 4.0);
    assert_eq!(values[X], 13.0);
}

#[test]
fn cost_right_hand_side_reads_the_parent_value() {
    // Same shape, but the shared variable is a cost variable, which takes a
    // different read path than a regular one.
    let values = successor_values(vec![plus(COST, ONE), plus(X, COST)]);

    assert_eq!(values[COST], 1.0);
    assert_eq!(values[X], 10.0);
}

#[test]
fn effect_order_does_not_change_the_successor() {
    let forwards = successor_values(vec![plus(Y, ONE), plus(X, Y)]);
    let backwards = successor_values(vec![plus(X, Y), plus(Y, ONE)]);

    assert_eq!(forwards, backwards);
}

#[test]
fn unsatisfied_effect_condition_suppresses_the_effect() {
    // `p` is false in the initial state, so this effect must not fire.
    let effect = AssignmentEffect::new(
        X,
        AssignmentOperation::Plus,
        ONE,
        true,
        vec![ExplicitFact::propositional(0, 1)],
    );

    let values = successor_values(vec![effect]);

    assert_eq!(values[X], 10.0);
}

#[test]
fn satisfied_effect_condition_fires_the_effect() {
    // `p` is false, so condition `p = 0` holds and the effect must fire.
    let effect = AssignmentEffect::new(
        X,
        AssignmentOperation::Plus,
        ONE,
        true,
        vec![ExplicitFact::propositional(0, 0)],
    );

    let values = successor_values(vec![effect]);

    assert_eq!(values[X], 11.0);
}

#[test]
fn effect_conditions_are_read_from_the_parent_state() {
    // The operator sets `p` to true and, conditionally on `p` being true, adds
    // one to `x`. The condition must be evaluated before the operator's own
    // propositional effect, so `x` stays 10.
    let task: TaskRef = Arc::new(NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            2,
            "p".into(),
            vec!["p-false".into(), "p-true".into()],
            None,
            0,
        )],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("total_cost".into(), NumericType::Cost, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![0],
        numeric_state: vec![10.0, 3.0, 1.0, 0.0],
        operators: vec![Operator::new(
            "set-p-and-add".into(),
            vec![],
            vec![crate::numeric_task::Effect::new(vec![], 0, None, 1)],
            vec![AssignmentEffect::new(
                X,
                AssignmentOperation::Plus,
                ONE,
                true,
                vec![ExplicitFact::propositional(0, 1)],
            )],
            1,
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    }));
    let mut registry = StateRegistry::for_task(task.clone());
    let initial = registry.get_initial_state();
    let successor = registry
        .get_successor_state(&initial, &task.get_operators()[0])
        .expect("operator applies to the initial state");

    let values = registry
        .get_numeric_vars(&successor)
        .expect("successor has numeric values");

    assert_eq!(successor.get_state(&registry), [1], "p must become true");
    assert_eq!(values[X], 10.0, "condition must read the parent value of p");
}

#[test]
fn metric_cost_reads_parent_values_across_effects() {
    // `metric_operator_cost_from_initial_values` replays the same effects on a
    // plain value vector; it must agree with the search path.
    let task = task_with_effects(vec![plus(Y, ONE), plus(X, Y)], Some(X));

    let delta = crate::numeric_task::metric_operator_cost_from_initial_values(
        &task,
        &task.get_operators()[0],
    );

    assert_eq!(delta, 3.0);
}

/// PDDL grounding produces operators like mprime's `drink ?n ?n`, whose two
/// additive effects target the same variable. Both deltas apply to the parent
/// value, so they cancel — and the result must not depend on which one is
/// stored first.
#[test]
fn additive_effects_on_one_variable_accumulate() {
    let minus_one = AssignmentEffect::new(X, AssignmentOperation::Minus, ONE, false, vec![]);

    let forwards = successor_values(vec![plus(X, ONE), minus_one.clone()]);
    let backwards = successor_values(vec![minus_one, plus(X, ONE)]);

    assert_eq!(forwards[X], 10.0);
    assert_eq!(backwards[X], 10.0);
}

/// Two additive effects that do not cancel still sum onto the parent value.
#[test]
fn additive_effects_on_one_variable_sum() {
    let values = successor_values(vec![plus(X, ONE), plus(X, ONE)]);

    assert_eq!(values[X], 12.0);
}

/// A repeat that mixes an assignment with an addition has no order-independent
/// reading, so it must be rejected rather than resolved by effect order.
#[test]
#[should_panic(
    expected = "operator op writes numeric variable 0 more than once with a non-additive assignment"
)]
fn conflicting_repeated_assignment_target_is_rejected_at_construction() {
    let effects = vec![
        AssignmentEffect::new(X, AssignmentOperation::Assign, ONE, false, vec![]),
        plus(X, ONE),
    ];

    let _task = task_with_effects(effects, None);
}
