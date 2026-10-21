//! Tasks on which regression and progression flaws differ, shared by the
//! tests of every abstraction.

use planforge_sas::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex,
    ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts, NumericType, NumericValue,
    NumericVariable, ONE_VALUE, Operator, OperatorCost, VariableIndex, ZERO_VALUE,
};

const TRUE: usize = ConditionValue::True.as_usize();

fn comparison_variable(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        name.into(),
        vec!["true".into(), "false".into()],
        Some(0),
        ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
    )
}

fn flag_variable() -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        "flag".into(),
        vec!["unset".into(), "set".into()],
        None,
        ExplicitValueIndex::new(0),
    )
}

fn add(var: usize, constant: usize) -> AssignmentEffect {
    AssignmentEffect::new(
        VariableIndex::from_usize(var),
        AssignmentOperation::Plus,
        VariableIndex::from_usize(constant),
        false,
        vec![],
    )
}

fn compare(var: usize, left: usize, right: usize, op: ComparisonOperator) -> ComparisonAxiom {
    ComparisonAxiom::new(
        VariableIndex::from_usize(var),
        VariableIndex::from_usize(left),
        VariableIndex::from_usize(right),
        op,
    )
}

/// `x` starts at 0 and the goal is `x >= 3`. `jump` (cost 1) requires
/// `x >= 1` and increments `x`; `inc` (cost 2) increments it
/// unconditionally. Numeric variables: `x`, `one`, `three`.
pub(crate) fn jump_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![
            comparison_variable("x-at-least-three"),
            comparison_variable("x-at-least-one"),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("three".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(0, TRUE)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(1), ExplicitValueIndex::new(1)],
        numeric_state: vec![ZERO_VALUE, ONE_VALUE, NumericValue::new(3.0)],
        operators: vec![
            Operator::new(
                "inc".into(),
                vec![],
                vec![],
                vec![add(0, 1)],
                OperatorCost::new(2),
            ),
            Operator::new(
                "jump".into(),
                vec![ExplicitFact::propositional(1, TRUE)],
                vec![],
                vec![add(0, 1)],
                OperatorCost::new(1),
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![
            compare(0, 0, 2, ComparisonOperator::GreaterThanOrEqual),
            compare(1, 0, 1, ComparisonOperator::GreaterThanOrEqual),
        ],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

/// `o` requires `x + y <= 0` and sets the flag, `inc` increments `x`, and
/// the goal is the flag and `x + y >= 1`, so `h* = 2`. No finite
/// partition of the original variables separates `x + y <= 0` from
/// `x + y >= 1`; the coordinate `sum = x + y` does.
pub(crate) fn linear_goal_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![
            comparison_variable("sum-at-most-zero"),
            comparison_variable("sum-at-least-one"),
            flag_variable(),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
            NumericVariable::new("sum".into(), NumericType::Derived, None),
            NumericVariable::new("zero".into(), NumericType::Constant, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        goals: vec![
            ExplicitFact::propositional(1, TRUE),
            ExplicitFact::propositional(2, 1),
        ],
        mutexes: vec![],
        state: vec![
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(1),
            ExplicitValueIndex::new(0),
        ],
        numeric_state: vec![ZERO_VALUE, ZERO_VALUE, ZERO_VALUE, ZERO_VALUE, ONE_VALUE],
        operators: vec![
            Operator::new(
                "o".into(),
                vec![ExplicitFact::propositional(0, TRUE)],
                vec![Effect::new(
                    vec![],
                    VariableIndex::from_usize(2),
                    None,
                    ExplicitValueIndex::new(1),
                )],
                vec![],
                OperatorCost::new(1),
            ),
            Operator::new(
                "inc".into(),
                vec![],
                vec![],
                vec![add(0, 4)],
                OperatorCost::new(1),
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![
            compare(0, 2, 3, ComparisonOperator::LessThanOrEqual),
            compare(1, 2, 4, ComparisonOperator::GreaterThanOrEqual),
        ],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(2),
            CalOperator::Sum,
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(1),
        )],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

/// `o` requires `y >= 1` and increments `x`; the goal is `x >= 2` and
/// `y <= 0`, from `(0, 0)`. The goal regressed through `o` contradicts
/// its precondition on `y` and leaves `x >= 1`, which the initial state
/// violates. Numeric variables: `x`, `y`, `zero`, `one`, `two`.
pub(crate) fn blocked_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![
            comparison_variable("y-at-least-one"),
            comparison_variable("x-at-least-two"),
            comparison_variable("y-at-most-zero"),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
            NumericVariable::new("zero".into(), NumericType::Constant, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
        ],
        goals: vec![
            ExplicitFact::propositional(1, TRUE),
            ExplicitFact::propositional(2, TRUE),
        ],
        mutexes: vec![],
        state: vec![
            ExplicitValueIndex::new(1),
            ExplicitValueIndex::new(1),
            ExplicitValueIndex::from_usize(TRUE),
        ],
        numeric_state: vec![
            ZERO_VALUE,
            ZERO_VALUE,
            ZERO_VALUE,
            ONE_VALUE,
            NumericValue::new(2.0),
        ],
        operators: vec![Operator::new(
            "o".into(),
            vec![ExplicitFact::propositional(0, TRUE)],
            vec![],
            vec![add(0, 3)],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![
            compare(0, 1, 3, ComparisonOperator::GreaterThanOrEqual),
            compare(1, 0, 4, ComparisonOperator::GreaterThanOrEqual),
            compare(2, 1, 2, ComparisonOperator::LessThanOrEqual),
        ],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

/// The corrected example for Theorem 4. `move_right` requires `x = y`
/// and increments `x`; `move_down` decrements `y`. From `(0, 1)` the goal
/// `x >= 1` takes `<move_down, move_right>`. Regression over single
/// variables drops `x = y` and misses the flaw of `<move_right>`; the
/// coordinate `d = x - y` keeps it. Numeric variables: `x`, `y`, `d`,
/// `zero`, `one`.
pub(crate) fn diagonal_task() -> NumericRootTask {
    let subtract = |var: usize| {
        AssignmentEffect::new(
            VariableIndex::from_usize(var),
            AssignmentOperation::Minus,
            VariableIndex::from_usize(4),
            false,
            vec![],
        )
    };
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![
            comparison_variable("x-equals-y"),
            comparison_variable("x-at-least-one"),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
            NumericVariable::new("d".into(), NumericType::Derived, None),
            NumericVariable::new("zero".into(), NumericType::Constant, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(1, TRUE)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(1), ExplicitValueIndex::new(1)],
        numeric_state: vec![ZERO_VALUE, ONE_VALUE, ZERO_VALUE, ZERO_VALUE, ONE_VALUE],
        operators: vec![
            Operator::new(
                "move_right".into(),
                vec![ExplicitFact::propositional(0, TRUE)],
                vec![],
                vec![add(0, 4)],
                OperatorCost::new(1),
            ),
            Operator::new(
                "move_down".into(),
                vec![],
                vec![],
                vec![subtract(1)],
                OperatorCost::new(1),
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![
            compare(0, 2, 3, ComparisonOperator::Equal),
            compare(1, 0, 4, ComparisonOperator::GreaterThanOrEqual),
        ],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(2),
            CalOperator::Difference,
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(1),
        )],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}
