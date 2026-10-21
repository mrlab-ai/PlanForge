use planforge_sas::numeric_task::{ExplicitValueIndex, NumericValue, OperatorIndex, VariableIndex};
use planforge_sas::utils::interval::Interval;

use super::test_tasks::{blocked_task, diagonal_task, jump_task};
use super::*;

const X: VariableIndex = VariableIndex::new(0);

/// A box over the task's numeric variables in which `x` lies in `x` and
/// the constants keep their values; propositional variables are free.
fn state(task: &dyn AbstractNumericTask, numeric: Vec<Interval>) -> CartesianSet {
    let propositions = (0..task.get_num_variables())
        .map(|var| {
            (0..task.variables()[var].domain_size())
                .map(ExplicitValueIndex::from_usize)
                .collect()
        })
        .collect();
    CartesianSet::with_all_props_constrained(propositions, numeric)
}

fn jump_state(task: &dyn AbstractNumericTask, x: Interval) -> CartesianSet {
    state(
        task,
        vec![
            x,
            Interval::singleton(NumericValue::new(1.0)),
            Interval::singleton(NumericValue::new(3.0)),
        ],
    )
}

fn at_most(value: f64) -> Interval {
    Interval::new(
        NumericValue::new(f64::NEG_INFINITY),
        NumericValue::new(value),
        false,
        true,
    )
}

fn at_least(value: f64) -> Interval {
    Interval::new(
        NumericValue::new(value),
        NumericValue::new(f64::INFINITY),
        true,
        false,
    )
}

fn search_in_scope(
    task: &dyn AbstractNumericTask,
    states: &[CartesianSet],
    operators: &[OperatorIndex],
    numeric: &[NumericValue],
    initial_split: SplitKind,
    scope: FlawScope,
) -> Vec<RegressionFlaw> {
    let coordinates = Coordinates::new(task);
    ensure_regressable(task, &coordinates).unwrap();
    let propositions = task.get_initial_propositional_state_values();
    find_regression_flaws(
        task,
        &coordinates,
        states,
        operators,
        InitialState {
            propositions,
            numeric,
        },
        initial_split,
        scope,
    )
    .unwrap()
}

fn search(
    task: &dyn AbstractNumericTask,
    states: &[CartesianSet],
    operators: &[OperatorIndex],
    numeric: &[NumericValue],
    initial_split: SplitKind,
) -> Option<RegressionFlaw> {
    let flaws = search_in_scope(
        task,
        states,
        operators,
        numeric,
        initial_split,
        FlawScope::First,
    );
    assert!(flaws.len() <= 1, "a first-flaw search reports {flaws:?}");
    flaws.into_iter().next()
}

const JUMP: OperatorIndex = OperatorIndex::new(1);

#[test]
fn case_3_splits_the_initial_state_at_the_regressed_goal_or_next_to_its_value() {
    let task = jump_task();
    let trace = [
        jump_state(&task, at_most(2.0)),
        jump_state(&task, at_least(3.0)),
    ];
    let init = [0.0, 1.0, 3.0].map(NumericValue::new);
    // x >= 3 regresses through jump to x >= 2, which x = 0 violates.
    assert_eq!(
        search(&task, &trace, &[JUMP], &init, SplitKind::Wanted),
        Some(RegressionFlaw {
            state: 0,
            split: FlawSplit::Numeric {
                var: X,
                boundary: NumericValue::new(2.0),
                lower_includes_boundary: false,
                wanted_is_lower: false,
            },
        })
    );
    assert_eq!(
        search(&task, &trace, &[JUMP], &init, SplitKind::Concrete),
        Some(RegressionFlaw {
            state: 0,
            split: FlawSplit::Numeric {
                var: X,
                boundary: NumericValue::new(0.0),
                lower_includes_boundary: true,
                wanted_is_lower: false,
            },
        })
    );
}

#[test]
fn case_2_splits_the_successor_where_the_predecessor_cannot_reach_the_goal() {
    let task = jump_task();
    // jump takes x <= 1 to x <= 2, which misses the goal x >= 3 in a_1.
    let trace = [
        jump_state(&task, at_most(1.0)),
        jump_state(&task, at_least(1.0)),
    ];
    let init = [0.0, 1.0, 3.0].map(NumericValue::new);
    assert_eq!(
        search(&task, &trace, &[JUMP], &init, SplitKind::Wanted),
        Some(RegressionFlaw {
            state: 1,
            split: FlawSplit::Numeric {
                var: X,
                boundary: NumericValue::new(3.0),
                lower_includes_boundary: false,
                wanted_is_lower: false,
            },
        })
    );
}

#[test]
fn a_plan_has_no_regression_flaw() {
    let task = jump_task();
    let trace = [
        jump_state(&task, Interval::singleton(NumericValue::new(2.0))),
        jump_state(&task, at_least(3.0)),
    ];
    let init = [2.0, 1.0, 3.0].map(NumericValue::new);
    assert_eq!(
        search(&task, &trace, &[JUMP], &init, SplitKind::Wanted),
        None
    );
}

#[test]
fn the_coordinate_x_minus_y_keeps_the_flaw_single_variables_lose() {
    let task = diagonal_task();
    let unbounded = Interval::unbounded();
    let constants = |numeric: Vec<Interval>| {
        let mut numeric = numeric;
        numeric.extend([0.0, 1.0].map(|value| Interval::singleton(NumericValue::new(value))));
        numeric
    };
    let trace = [
        state(&task, constants(vec![at_most(0.0), unbounded, unbounded])),
        state(&task, constants(vec![at_least(1.0), unbounded, unbounded])),
    ];
    let init = [0.0, 1.0, -1.0, 0.0, 1.0].map(NumericValue::new);
    // p_0 = {x >= 0, d = 0}; the initial state has d = -1.
    let flaw = search(
        &task,
        &trace,
        &[OperatorIndex::new(0)],
        &init,
        SplitKind::Wanted,
    );
    assert!(
        matches!(
            flaw,
            Some(RegressionFlaw {
                state: 0,
                split: FlawSplit::Numeric { var, .. },
            }) if var == VariableIndex::new(2)
        ),
        "got {flaw:?}"
    );
}

#[test]
fn whole_plan_regression_ignores_a_contradicted_precondition_and_goes_on() {
    let task = blocked_task();
    let unbounded = Interval::unbounded();
    let constants = [0.0, 1.0, 2.0].map(|value| Interval::singleton(NumericValue::new(value)));
    let with_constants = |x: Interval| {
        let mut numeric = vec![x, unbounded];
        numeric.extend(constants);
        numeric
    };
    let trace = [
        state(&task, with_constants(unbounded)),
        state(&task, with_constants(at_least(2.0))),
    ];
    let init = [0.0, 0.0, 0.0, 1.0, 2.0].map(NumericValue::new);
    let y = VariableIndex::new(1);
    // y <= 0 regresses through o to y <= 0, which its precondition y >= 1
    // contradicts in a_0.
    let precondition_flaw = RegressionFlaw {
        state: 0,
        split: FlawSplit::Numeric {
            var: y,
            boundary: ZERO_VALUE,
            lower_includes_boundary: true,
            wanted_is_lower: true,
        },
    };
    // Ignoring it, p_0 = {x >= 1, y <= 0}, and x = 0 violates x >= 1.
    let initial_flaw = RegressionFlaw {
        state: 0,
        split: FlawSplit::Numeric {
            var: X,
            boundary: ONE_VALUE,
            lower_includes_boundary: false,
            wanted_is_lower: false,
        },
    };
    let o = [OperatorIndex::new(0)];
    assert_eq!(
        search_in_scope(
            &task,
            &trace,
            &o,
            &init,
            SplitKind::Wanted,
            FlawScope::First
        ),
        vec![precondition_flaw.clone()]
    );
    assert_eq!(
        search_in_scope(
            &task,
            &trace,
            &o,
            &init,
            SplitKind::Wanted,
            FlawScope::WholePlan
        ),
        vec![precondition_flaw, initial_flaw]
    );
}
