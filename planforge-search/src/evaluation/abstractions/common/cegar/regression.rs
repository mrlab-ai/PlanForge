//! Regression flaws (Pozo et al. 2024) over coordinates.
//!
//! An abstract plan trace `a_0 -o_1-> a_1 ... -o_n-> a_n` is regressed from
//! the goal: `p_n` is the goal and `p_{i-1}` is `p_i` regressed through
//! `o_i` together with the preconditions of `o_i`. A partial state gives each
//! coordinate an interval and each propositional variable at most one value,
//! so regressing through additive and constant effects is exact. The search
//! reports the first of three cases, walking back from the goal:
//!
//! 1. `p_i` contradicts `o_i`: an effect of `o_i` falsifies `p_i`, or a
//!    precondition of `o_i` contradicts the regression of `p_i`;
//! 2. `p_{i-1}` and `a_{i-1}` are disjoint;
//! 3. the initial state violates `p_0`.
//!
//! If none applies, the operators of the trace are a plan.
//!
//! Whole-plan regression ([`FlawScope::WholePlan`]) reports every flaw
//! instead, as whole-plan progression does: past a flaw it ignores
//! the contradicted condition and regresses the effects anyway. An effect that
//! falsifies `p_i` frees its variable, a contradicted precondition is dropped,
//! and a `p_{i-1}` disjoint from `a_{i-1}` is kept. Every flaw after the first
//! is thus witnessed by a counterfactual partial state, which need not overlap
//! the abstract states of the trace; a split that does not split its abstract
//! state strictly is no flaw of that state.
//!
//! Every first flaw yields
//! a strict split of one abstract state on one coordinate at a bound of a
//! regressed partial state (or, in case 3, next to the initial value). Such
//! bounds are condition constants shifted by the effects of a plan suffix, so
//! an optimal-plan CEGAR loop with these flaws is semi-complete.
//!
//! Abstract transitions are tested per dimension, so a transition `a_{i-1} ->
//! a_i` may exist although no state that satisfies `pre(o_i)` reaches `a_i`.
//! The split therefore goes to `a_i` when the image of `a_{i-1}` misses `p_i`
//! and to `a_{i-1}` when only the precondition and `p_i` disagree there.

use anyhow::{Context, Result, ensure};
use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitValueIndex, NumericValue, ONE_VALUE, OperatorIndex,
    VariableIndex, ZERO_VALUE,
};
use planforge_sas::utils::interval::Interval;

use std::collections::HashSet;

use super::progression::{ConcretePlan, PlanCheck};
use super::selection::{push_unique_split, select_refinement};
use super::semantics::CegarSemantics;
use super::split::{Split, numeric_split_intervals};
use super::view::{AbstractionView, TransitionKey};
use super::{FlawScope, SplitKind};
use crate::ensure_supported;
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use crate::evaluation::abstractions::common::coordinates::Coordinates;

/// The flaws found so far.
struct Flaws {
    scope: FlawScope,
    found: Vec<RegressionFlaw>,
}

impl Flaws {
    /// Records the flaw of abstract state `state` that asks for `split`,
    /// `None` if that split would not split the state strictly. Returns
    /// whether the search stops.
    fn record(&mut self, state: usize, split: Option<FlawSplit>) -> Result<bool> {
        match split {
            Some(split) => self.found.push(RegressionFlaw { state, split }),
            None => ensure!(
                !self.found.is_empty(),
                "the first regression flaw does not split abstract state {state} strictly"
            ),
        }
        Ok(self.scope == FlawScope::First)
    }
}

/// A split of one abstract state of the trace.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RegressionFlaw {
    /// Index of the abstract state in the trace, `0..=n`.
    pub(crate) state: usize,
    pub(crate) split: FlawSplit,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum FlawSplit {
    /// Split the coordinate at `boundary`; the regressed partial state lies on
    /// the lower side iff `wanted_is_lower`.
    Numeric {
        var: VariableIndex,
        boundary: NumericValue,
        lower_includes_boundary: bool,
        wanted_is_lower: bool,
    },
    /// Separate the value `wanted` of the variable from its other values.
    Propositional {
        var: VariableIndex,
        wanted: ExplicitValueIndex,
    },
}

/// The concrete initial state, as values of all variables.
#[derive(Clone, Copy)]
pub(crate) struct InitialState<'a> {
    pub(crate) propositions: &'a [ExplicitValueIndex],
    pub(crate) numeric: &'a [NumericValue],
}

/// Rejects a task whose preconditions or goal mention a propositional
/// variable that axioms derive, or a comparison that is not over exactly one
/// coordinate: regression over coordinates cannot express either.
pub(crate) fn ensure_regressable(
    task: &dyn AbstractNumericTask,
    coordinates: &Coordinates<'_>,
) -> Result<()> {
    let mut derived = vec![false; task.get_num_variables()];
    for axiom in task.axioms() {
        derived[axiom.var_id().index()] = true;
    }
    let check = |fact: &ExplicitFact, context: &dyn Fn() -> String| -> Result<()> {
        if task.numeric_conditions().is_condition_var(fact.var_index()) {
            ensure_supported!(
                coordinates.requirement(fact).is_some(),
                "regression flaws need conditions over one coordinate, but {} has {fact:?}; --restrict-task compiles every linear condition into one",
                context()
            );
        } else {
            ensure_supported!(
                !derived[fact.var()],
                "regression flaws cannot regress {} fact {fact:?}: its variable is derived by axioms",
                context()
            );
        }
        Ok(())
    };
    for (op_id, op) in task.get_operators().iter().enumerate() {
        for fact in op.preconditions() {
            check(fact, &|| format!("operator {op_id} ({})", op.name()))?;
        }
    }
    for goal in task.goals() {
        check(goal, &|| "the goal".to_string())?;
    }
    Ok(())
}

/// A conjunction of requirements: an interval per numeric variable
/// (unbounded when unconstrained) and at most one value per propositional
/// variable.
#[derive(Debug, Clone)]
struct PartialState {
    propositions: Vec<Option<ExplicitValueIndex>>,
    numeric: Vec<Interval>,
}

impl PartialState {
    fn unconstrained(task: &dyn AbstractNumericTask) -> Self {
        Self {
            propositions: vec![None; task.get_num_variables()],
            numeric: vec![Interval::unbounded(); task.numeric_variables().len()],
        }
    }

    /// Adds `facts`; `None` if they contradict each other or `self`.
    fn require(
        &mut self,
        coordinates: &Coordinates<'_>,
        task: &dyn AbstractNumericTask,
        facts: &[ExplicitFact],
    ) -> Result<Option<()>> {
        for fact in facts {
            if task.numeric_conditions().is_condition_var(fact.var_index()) {
                let requirement = coordinates
                    .requirement(fact)
                    .with_context(|| format!("{fact:?} is not a condition over one coordinate"))?;
                let interval = &mut self.numeric[requirement.coordinate.index()];
                *interval = interval.intersection(&requirement.values);
                if interval.is_empty() {
                    return Ok(None);
                }
            } else {
                let value = &mut self.propositions[fact.var()];
                match *value {
                    Some(required) if required != fact.value_index() => return Ok(None),
                    _ => *value = Some(fact.value_index()),
                }
            }
        }
        Ok(Some(()))
    }
}

/// The regression flaws of the trace `states[0] -operators[0]-> ...
/// states[n]` in `scope`, walking back from the goal; none iff `operators` is
/// a plan from `init`. Every abstract state is the Cartesian set of states it
/// represents.
pub(crate) fn find_regression_flaws(
    task: &dyn AbstractNumericTask,
    coordinates: &Coordinates<'_>,
    states: &[CartesianSet],
    operators: &[OperatorIndex],
    init: InitialState<'_>,
    initial_split: SplitKind,
    scope: FlawScope,
) -> Result<Vec<RegressionFlaw>> {
    ensure!(
        states.len() == operators.len() + 1,
        "a trace of {} operators has {} abstract states",
        operators.len(),
        states.len()
    );
    let mut partial = PartialState::unconstrained(task);
    ensure!(
        partial.require(coordinates, task, task.goals())?.is_some(),
        "the goal contradicts itself"
    );
    ensure!(
        overlaps(&partial, &states[operators.len()]),
        "the last abstract state of the trace admits no goal state"
    );
    let mut flaws = Flaws {
        scope,
        found: Vec::new(),
    };
    for step in (1..=operators.len()).rev() {
        let op_id = operators[step - 1];
        match regress(
            task,
            coordinates,
            &partial,
            op_id,
            &states[step - 1],
            &states[step],
            step,
            &mut flaws,
        )? {
            Some(predecessor) => partial = predecessor,
            None => return Ok(flaws.found),
        }
    }
    initial_flaws(&partial, &states[0], init, initial_split, &mut flaws)?;
    Ok(flaws.found)
}

/// Regresses `partial` (= `p_step`) through `op_id` into the source state
/// `source` (= `a_{step-1}`), recording cases 1 and 2 in `flaws`. `target` is
/// `a_step`; before the first flaw, `partial` overlaps it on every dimension.
/// Returns `p_{step-1}`, or `None` once the search stops.
#[allow(clippy::too_many_arguments)]
fn regress(
    task: &dyn AbstractNumericTask,
    coordinates: &Coordinates<'_>,
    partial: &PartialState,
    op_id: OperatorIndex,
    source: &CartesianSet,
    target: &CartesianSet,
    step: usize,
    flaws: &mut Flaws,
) -> Result<Option<PartialState>> {
    let op = &task.get_operators()[op_id.index()];
    let mut predecessor = PartialState::unconstrained(task);

    // Propositional effects falsify p_step (case 1) or free their variable;
    // past the flaw, the effect is regressed anyway and frees it too.
    let mut written = vec![false; task.get_num_variables()];
    for effect in op.effects() {
        let var = effect.var_id();
        written[var.index()] = true;
        if let Some(wanted) = partial.propositions[var.index()]
            && wanted != effect.value()
            && flaws.record(step, propositional_split(target, var, wanted))?
        {
            return Ok(None);
        }
    }
    let mut preconditions = PartialState::unconstrained(task);
    ensure!(
        preconditions
            .require(coordinates, task, op.preconditions())?
            .is_some(),
        "operator {op_id} ({}) has contradicting preconditions",
        op.name()
    );

    for (index, wanted) in partial.propositions.iter().enumerate() {
        let var = VariableIndex::from_usize(index);
        let precondition = preconditions.propositions[index];
        let Some(wanted) = wanted.filter(|_| !written[index]) else {
            if let Some(required) = precondition {
                ensure!(
                    admits_value(source, var, required),
                    "abstract state {} admits operator {op_id} ({}) although it excludes its precondition {var}={required}",
                    step - 1,
                    op.name()
                );
            }
            predecessor.propositions[index] = precondition;
            continue;
        };
        // Case 2; past it, p_{step-1} keeps the value.
        if !admits_value(source, var, wanted)
            && flaws.record(step, propositional_split(target, var, wanted))?
        {
            return Ok(None);
        }
        // Case 1; past it, the contradicted precondition is ignored.
        if precondition.is_some_and(|required| required != wanted)
            && flaws.record(step - 1, propositional_split(source, var, wanted))?
        {
            return Ok(None);
        }
        predecessor.propositions[index] = Some(wanted);
    }

    for (index, &wanted) in partial.numeric.iter().enumerate() {
        let var = VariableIndex::from_usize(index);
        let source_interval = source.numeric[index];
        let required = preconditions.numeric[index];
        let Some(regressed) = coordinates.preimage(wanted, op_id, var) else {
            // Case 1: op_id assigns var a value outside p_step, which a_step
            // contains. Past it, the assignment is regressed anyway and
            // leaves only the precondition on var.
            let assigned = coordinates.image(source_interval, op_id, var);
            if flaws.record(
                step,
                separate(var, wanted, assigned, target.numeric[index])?,
            )? {
                return Ok(None);
            }
            predecessor.numeric[index] = required;
            continue;
        };
        let reaches_source = regressed.intersects(&source_interval);
        // Case 2; past it, p_{step-1} keeps the regressed values.
        if !reaches_source {
            let image = coordinates.image(source_interval, op_id, var);
            if flaws.record(step, separate(var, wanted, image, target.numeric[index])?)? {
                return Ok(None);
            }
        }
        let combined = regressed.intersection(&required);
        // Case 1: the precondition contradicts the regression in a_{step-1};
        // past it, a precondition that contradicts the regression outright is
        // ignored.
        if reaches_source
            && !combined.intersects(&source_interval)
            && flaws.record(
                step - 1,
                separate(
                    var,
                    regressed,
                    required.intersection(&source_interval),
                    source_interval,
                )?,
            )?
        {
            return Ok(None);
        }
        predecessor.numeric[index] = if combined.is_empty() {
            regressed
        } else {
            combined
        };
    }
    Ok(Some(predecessor))
}

/// Case 3: the initial state against `p_0` in its abstract state `state`.
fn initial_flaws(
    partial: &PartialState,
    state: &CartesianSet,
    init: InitialState<'_>,
    initial_split: SplitKind,
    flaws: &mut Flaws,
) -> Result<()> {
    for (index, wanted) in partial.propositions.iter().enumerate() {
        let var = VariableIndex::from_usize(index);
        if let Some(wanted) = *wanted
            && init.propositions[index] != wanted
            && flaws.record(0, propositional_split(state, var, wanted))?
        {
            return Ok(());
        }
    }
    for (index, wanted) in partial.numeric.iter().enumerate() {
        let value = init.numeric[index];
        if wanted.contains(value) {
            continue;
        }
        let var = VariableIndex::from_usize(index);
        let split = match initial_split {
            SplitKind::Wanted => separate(
                var,
                *wanted,
                Interval::singleton(value),
                state.numeric[index],
            )?,
            SplitKind::Concrete => split_next_to(var, value, *wanted, state.numeric[index]),
        };
        if flaws.record(0, split)? {
            return Ok(());
        }
    }
    Ok(())
}

/// The split that separates the value `wanted` of `var` from its other values
/// in `state`, `None` unless `state` admits `wanted` and another value.
fn propositional_split(
    state: &CartesianSet,
    var: VariableIndex,
    wanted: ExplicitValueIndex,
) -> Option<FlawSplit> {
    (admits_value(state, var, wanted) && state.propositions()[var.index()].len() > 1)
        .then_some(FlawSplit::Propositional { var, wanted })
}

/// The split at the bound of `wanted` that faces the disjoint interval
/// `reached`, `None` unless it splits `within` strictly.
fn separate(
    var: VariableIndex,
    wanted: Interval,
    reached: Interval,
    within: Interval,
) -> Result<Option<FlawSplit>> {
    ensure!(
        !wanted.intersects(&reached),
        "numeric variable {var}: the separated intervals {wanted:?} and {reached:?} overlap"
    );
    let reached_is_lower = reached.upper < wanted.lower
        || (reached.upper == wanted.lower && !(reached.upper_closed && wanted.lower_closed));
    let (boundary, lower_includes_boundary) = if reached_is_lower {
        (wanted.lower, !wanted.lower_closed)
    } else {
        (wanted.upper, wanted.upper_closed)
    };
    Ok(
        (boundary.is_finite() && within.can_split_at(boundary, lower_includes_boundary)).then_some(
            FlawSplit::Numeric {
                var,
                boundary,
                lower_includes_boundary,
                wanted_is_lower: !reached_is_lower,
            },
        ),
    )
}

/// The split right next to `value`, keeping `value` on the side away from
/// `wanted`; `None` unless it splits `within` strictly.
fn split_next_to(
    var: VariableIndex,
    value: NumericValue,
    wanted: Interval,
    within: Interval,
) -> Option<FlawSplit> {
    let value_is_lower = value < wanted.lower || (value == wanted.lower && !wanted.lower_closed);
    let lower_includes_boundary = value_is_lower;
    within
        .can_split_at(value, lower_includes_boundary)
        .then_some(FlawSplit::Numeric {
            var,
            boundary: value,
            lower_includes_boundary,
            wanted_is_lower: !value_is_lower,
        })
}

fn admits_value(state: &CartesianSet, var: VariableIndex, value: ExplicitValueIndex) -> bool {
    state.propositions()[var.index()]
        .binary_search(&value)
        .is_ok()
}

fn overlaps(partial: &PartialState, state: &CartesianSet) -> bool {
    partial
        .propositions
        .iter()
        .enumerate()
        .all(|(index, wanted)| {
            wanted
                .is_none_or(|wanted| admits_value(state, VariableIndex::from_usize(index), wanted))
        })
        && partial
            .numeric
            .iter()
            .zip(state.numeric.iter())
            .all(|(wanted, interval)| wanted.intersects(interval))
}

#[cfg(test)]
pub(crate) mod test_tasks {
    //! Tasks on which regression and progression flaws differ, shared by the
    //! tests of every abstraction.

    use planforge_sas::axioms::{
        AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator,
    };
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
}

/// Regression flaw search in `scope` on the optimal abstract plan from the
/// abstract state of `refinement_root`. Without a flaw, the plan's operators
/// are a concrete plan; with several, the split is selected among them.
pub(crate) fn regression_plan_check(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    refinement_root: &ExplicitState,
    scope: FlawScope,
) -> Result<PlanCheck> {
    let initial_state =
        view.abstract_state(&refinement_root.propositional, &refinement_root.numeric)?;
    if !view.goal_distance(initial_state).is_finite() {
        return Ok(PlanCheck::AbstractDeadEnd);
    }
    let plan = abstract_plan(view, initial_state)?;
    let state_ids = std::iter::once(initial_state)
        .chain(plan.iter().map(|transition| transition.target))
        .collect::<Vec<_>>();
    let states = state_ids
        .iter()
        .map(|&state_id| view.region(state_id).into_owned())
        .collect::<Vec<_>>();
    let operator_ids = plan
        .iter()
        .map(|transition| transition.concrete_op_id)
        .collect::<Vec<_>>();
    let flaws = find_regression_flaws(
        semantics.task(),
        semantics.coordinates(),
        &states,
        &operator_ids,
        InitialState {
            propositions: &refinement_root.propositional,
            numeric: &refinement_root.numeric,
        },
        semantics.split_kind(),
        scope,
    )?;
    if flaws.is_empty() {
        let cost = operator_ids
            .iter()
            .map(|op_id| semantics.operator_costs()[op_id.index()].value())
            .sum();
        return Ok(PlanCheck::ConcretePlan(ConcretePlan::new(
            operator_ids,
            cost,
        )));
    }
    match scope {
        FlawScope::First => {
            let [flaw] = &flaws[..] else {
                unreachable!("a first-flaw search reports {} flaws", flaws.len());
            };
            Ok(PlanCheck::Refine(flaw_split(
                view, semantics, &state_ids, flaw,
            )?))
        }
        FlawScope::WholePlan => {
            let mut candidates = Vec::with_capacity(flaws.len());
            let mut identities = HashSet::with_capacity(flaws.len());
            for flaw in &flaws {
                push_unique_split(
                    &mut candidates,
                    &mut identities,
                    flaw_split(view, semantics, &state_ids, flaw)?,
                );
            }
            select_refinement(view, semantics, candidates)
        }
    }
}

/// The split of the abstract state `state_ids[flaw.state]` that `flaw` asks for.
fn flaw_split(
    view: &impl AbstractionView,
    semantics: &CegarSemantics<'_>,
    state_ids: &[usize],
    flaw: &RegressionFlaw,
) -> Result<Split> {
    let state_id = state_ids[flaw.state];
    Ok(match flaw.split {
        FlawSplit::Numeric {
            var,
            boundary,
            lower_includes_boundary,
            wanted_is_lower,
        } => {
            let integer_lattice = semantics.numeric_integer_lattice()[var.index()];
            let (lower, upper) = numeric_split_intervals(
                view.region(state_id).numeric[var.index()],
                boundary,
                lower_includes_boundary,
                integer_lattice,
            )
            .with_context(|| {
                format!("regression flaw boundary {boundary} does not split numeric var {var}")
            })?;
            Split::Numeric {
                state_id,
                var_id: var,
                boundary,
                lower_includes_boundary,
                witness_value: interior_value(if wanted_is_lower { upper } else { lower }),
                integer_lattice,
            }
        }
        FlawSplit::Propositional { var, wanted } => {
            let witness_value = *view.region(state_id).propositions()[var.index()]
                .iter()
                .find(|&&value| value != wanted)
                .with_context(|| {
                    format!("propositional split of {var} at {wanted} in abstract state {state_id} is not strict")
                })?;
            Split::Propositional {
                state_id,
                var_id: var,
                wanted: vec![wanted],
                witness_value,
            }
        }
    })
}

/// The transitions of the optimal abstract plan from `initial_state`: the
/// generating transitions of the shortest paths.
fn abstract_plan(view: &impl AbstractionView, initial_state: usize) -> Result<Vec<TransitionKey>> {
    let mut plan = Vec::new();
    let mut state = initial_state;
    while !view.is_goal(state)? {
        ensure!(
            plan.len() <= view.num_states(),
            "optimal abstract transitions contain a cycle"
        );
        let transition = view.optimal_transition(state)?;
        plan.push(transition);
        state = transition.target;
    }
    Ok(plan)
}

/// A value of the non-empty interval `interval`.
fn interior_value(interval: Interval) -> NumericValue {
    match (interval.lower.is_finite(), interval.upper.is_finite()) {
        (true, _) if interval.lower_closed => interval.lower,
        (_, true) if interval.upper_closed => interval.upper,
        (true, true) => NumericValue::new((interval.lower.value() + interval.upper.value()) / 2.0),
        (true, false) => interval.lower + ONE_VALUE,
        (false, true) => interval.upper - ONE_VALUE,
        (false, false) => ZERO_VALUE,
    }
}

#[cfg(test)]
mod tests {
    use planforge_sas::numeric_task::{
        ExplicitValueIndex, NumericValue, OperatorIndex, VariableIndex,
    };
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
}
