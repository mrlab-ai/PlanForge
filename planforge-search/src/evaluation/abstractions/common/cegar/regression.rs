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

#[cfg(test)]
pub(crate) mod test_tasks;
#[cfg(test)]
mod tests;

use anyhow::{Context, Result, ensure};
use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitValueIndex, NumericValue, ONE_VALUE, OperatorIndex,
    VariableIndex, ZERO_VALUE,
};
use planforge_sas::utils::interval::Interval;

use std::collections::HashSet;

use super::progression::{ConcretePlan, PlanCheck};
use super::selection::{WHOLE_PLAN_TAG, push_unique_split, select_split};
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

    /// Drops every requirement.
    fn clear(&mut self) {
        self.propositions.fill(None);
        self.numeric.fill(Interval::unbounded());
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
    let mut regression = Regression {
        task,
        coordinates,
        flaws: Flaws {
            scope,
            found: Vec::new(),
        },
        preconditions: PartialState::unconstrained(task),
        written: vec![false; task.get_num_variables()],
    };
    let mut predecessor = PartialState::unconstrained(task);
    for step in (1..=operators.len()).rev() {
        if !regression.regress(
            &partial,
            &mut predecessor,
            operators[step - 1],
            &states[step - 1],
            &states[step],
            step,
        )? {
            return Ok(regression.flaws.found);
        }
        std::mem::swap(&mut partial, &mut predecessor);
    }
    initial_flaws(
        &partial,
        &states[0],
        init,
        initial_split,
        &mut regression.flaws,
    )?;
    Ok(regression.flaws.found)
}

/// The regression of a trace: the flaws found so far, and the buffers every
/// step reuses.
struct Regression<'a> {
    task: &'a dyn AbstractNumericTask,
    coordinates: &'a Coordinates<'a>,
    flaws: Flaws,
    /// The preconditions of the operator regressed through.
    preconditions: PartialState,
    /// Per propositional variable, whether that operator writes it.
    written: Vec<bool>,
}

impl Regression<'_> {
    /// Regresses `partial` (= `p_step`) through `op_id` into the source
    /// state `source` (= `a_{step-1}`), recording cases 1 and 2. `target` is
    /// `a_step`; before the first flaw, `partial` overlaps it on every
    /// dimension. Writes `p_{step-1}` to `predecessor` and returns whether
    /// the search goes on.
    fn regress(
        &mut self,
        partial: &PartialState,
        predecessor: &mut PartialState,
        op_id: OperatorIndex,
        source: &CartesianSet,
        target: &CartesianSet,
        step: usize,
    ) -> Result<bool> {
        let Self {
            task,
            coordinates,
            flaws,
            preconditions,
            written,
        } = self;
        let op = &task.get_operators()[op_id.index()];

        // Propositional effects falsify p_step (case 1) or free their
        // variable; past the flaw, the effect is regressed anyway and frees
        // it too.
        written.fill(false);
        for effect in op.effects() {
            let var = effect.var_id();
            written[var.index()] = true;
            if let Some(wanted) = partial.propositions[var.index()]
                && wanted != effect.value()
                && flaws.record(step, propositional_split(target, var, wanted))?
            {
                return Ok(false);
            }
        }
        preconditions.clear();
        ensure!(
            preconditions
                .require(coordinates, *task, op.preconditions())?
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
                return Ok(false);
            }
            // Case 1; past it, the contradicted precondition is ignored.
            if precondition.is_some_and(|required| required != wanted)
                && flaws.record(step - 1, propositional_split(source, var, wanted))?
            {
                return Ok(false);
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
                    return Ok(false);
                }
                predecessor.numeric[index] = required;
                continue;
            };
            let reaches_source = regressed.intersects(&source_interval);
            // Case 2; past it, p_{step-1} keeps the regressed values.
            if !reaches_source {
                let image = coordinates.image(source_interval, op_id, var);
                if flaws.record(step, separate(var, wanted, image, target.numeric[index])?)? {
                    return Ok(false);
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
                return Ok(false);
            }
            predecessor.numeric[index] = if combined.is_empty() {
                regressed
            } else {
                combined
            };
        }
        Ok(true)
    }
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
            Ok(PlanCheck::Refine(select_split(
                view,
                semantics,
                candidates,
                WHOLE_PLAN_TAG,
            )?))
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
