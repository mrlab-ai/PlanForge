//! Regression flaws whose splits are anchored in the concrete trace.
//!
//! Progression replays an abstract plan `π = ⟨o_1, ..., o_n⟩` from the
//! initial state `s_0` and stops at the first step `k` whose state `s_k`
//! exposes a flaw. Regression continues past `k`: like sequence flaws
//! (Pozo et al. 2024), it applies `o_{k+1}, ..., o_n` to `s_k` although they
//! may be inapplicable, and collects the conditions that fail on the way --
//! the preconditions of `o_{j+1}` at step `j` and the goal at step `n`. A
//! failed condition `φ` of step `j` is then regressed through
//! `o_{i+1}, ..., o_j` to the abstract state `α(s_i)` of a prefix state,
//! `i ≤ k`, and the flaw is a split of `α(s_i)` that keeps `s_i` and
//! separates it from the states whose progression satisfies `φ`.
//!
//! Splitting only abstract states of concrete prefix states is what lets
//! CEGAR terminate on solvable tasks: optimal abstract plans cost at most
//! `h*`, so with positive operator costs they range over finitely many
//! operator sequences, each with finitely many prefix states, and every split
//! strictly shrinks the abstract state of one of them.
//!
//! The regression is exact without symbolic condition rewriting: evaluating
//! the regressed condition on a box equals evaluating `φ` on the box's
//! interval image under `o_{i+1}, ..., o_j`, because abstraction operators
//! have unconditional effects with constant right-hand sides.

use anyhow::{Result, bail, ensure};
use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, ExplicitFact, NumericType, NumericValue,
    OperatorIndex, VariableIndex, ZERO_VALUE,
};
use planforge_sas::state_registry::ConcreteStateView;
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::interval::Interval;
use planforge_sas::utils::state_packer::StatePacker;

use super::progress_concrete_state;
use crate::evaluation::domain_abstractions::additive_numeric_views::AdditiveNumericView;

/// Rejects a task whose operator preconditions or goal mention a
/// propositional variable that axioms derive: such a condition has no
/// operator effects to regress through.
pub(crate) fn ensure_conditions_are_regressable(task: &dyn AbstractNumericTask) -> Result<()> {
    let mut derived = vec![false; task.get_num_variables()];
    for axiom in task.axioms() {
        derived[axiom.var_id().index()] = true;
    }
    let is_derived = |fact: &ExplicitFact| {
        derived[fact.var()] && !task.numeric_conditions().is_condition_var(fact.var_index())
    };
    for (op_id, op) in task.get_operators().iter().enumerate() {
        if let Some(fact) = op.preconditions().iter().find(|fact| is_derived(fact)) {
            bail!(
                "regression flaws cannot regress precondition {fact:?} of operator {op_id} ({}): its variable is derived by axioms",
                op.name()
            );
        }
    }
    if let Some(fact) = (0..task.get_num_goals())
        .map(|goal_id| task.get_goal_fact(goal_id))
        .find(|fact| is_derived(fact))
    {
        bail!("regression flaws cannot regress goal {fact:?}: its variable is derived by axioms");
    }
    Ok(())
}

/// The conditions of one abstract plan step that its relaxed suffix state
/// violates: the preconditions of `o_{step+1}`, or the goal at `step = n`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FailedConditions {
    pub(crate) step: usize,
    pub(crate) facts: Vec<ExplicitFact>,
}

/// Applies `suffix = ⟨o_{k+1}, ..., o_n⟩` to the state `s_k` at step
/// `k = first_step`, ignoring preconditions, and returns the failed conditions
/// of steps `k, ..., n` in ascending step order, omitting steps without any.
pub(crate) fn failed_conditions_on_relaxed_suffix(
    task: &dyn AbstractNumericTask,
    axiom_evaluator: &AxiomEvaluator<'_>,
    packer: &StatePacker,
    mut propositions: Vec<u64>,
    mut numeric: Vec<NumericValue>,
    first_step: usize,
    suffix: &[OperatorIndex],
) -> Result<Vec<FailedConditions>> {
    let operators = task.get_operators();
    let mut failed = Vec::new();
    for (offset, &op_id) in suffix.iter().enumerate() {
        let op = &operators[op_id.index()];
        let state = ConcreteStateView::from_decoded(packer, &propositions, &[]);
        let facts = op
            .preconditions()
            .iter()
            .filter(|fact| !fact.is_held(state))
            .copied()
            .collect::<Vec<_>>();
        if !facts.is_empty() {
            failed.push(FailedConditions {
                step: first_step + offset,
                facts,
            });
        }
        progress_concrete_state(op, axiom_evaluator, packer, &mut propositions, &mut numeric)?;
    }
    let state = ConcreteStateView::from_decoded(packer, &propositions, &[]);
    let facts = (0..task.get_num_goals())
        .map(|goal_id| *task.get_goal_fact(goal_id))
        .filter(|fact| !fact.is_held(state))
        .collect::<Vec<_>>();
    if !facts.is_empty() {
        failed.push(FailedConditions {
            step: first_step + suffix.len(),
            facts,
        });
    }
    Ok(failed)
}

/// The `(anchor step, failed conditions)` pairs a regression flaw search tries,
/// in order: the conditions of the latest step first, each regressed to the
/// latest anchor first. `last_anchor` is the step `k` of the last prefix
/// state; every step in `failed` is at least `k`.
pub(crate) fn regression_candidates(
    failed: &[FailedConditions],
    last_anchor: usize,
) -> impl Iterator<Item = (usize, &FailedConditions)> {
    debug_assert!(
        failed.is_sorted_by_key(|conditions| conditions.step),
        "failed conditions are listed in step order"
    );
    failed.iter().rev().flat_map(move |conditions| {
        assert!(
            conditions.step >= last_anchor,
            "failed conditions of step {} precede the last anchor {last_anchor}",
            conditions.step
        );
        (0..=last_anchor)
            .rev()
            .map(move |anchor| (anchor, conditions))
    })
}

/// Whether an operator of `ops` writes the propositional variable `var_id`.
/// A propositional condition on `var_id` regresses through `ops` to itself
/// exactly when none does.
pub(crate) fn writes_propositional_variable(
    task: &dyn AbstractNumericTask,
    ops: &[OperatorIndex],
    var_id: VariableIndex,
) -> bool {
    let operators = task.get_operators();
    ops.iter().any(|op_id| {
        operators[op_id.index()]
            .effects()
            .iter()
            .any(|effect| effect.var_id() == var_id)
    })
}

/// Replaces the numeric box `intervals` by its interval image under `op_id`,
/// ignoring the operator's preconditions.
///
/// Regular variables take the image of their effect; derived additive
/// coordinates shift by the operator's delta. Every other derived variable is
/// never split by an abstraction and stays unbounded.
pub(crate) fn progress_numeric_box(
    task: &dyn AbstractNumericTask,
    additive_views: &[Option<AdditiveNumericView>],
    op_id: OperatorIndex,
    intervals: &mut [Interval],
) -> Result<()> {
    let variables = task.numeric_variables();
    let initial = task.get_initial_numeric_state_values();
    for effect in task.get_operators()[op_id.index()].assignment_effects() {
        let source = effect.var_id();
        debug_assert_eq!(
            variables[source.index()].get_type(),
            &NumericType::Constant,
            "abstraction operators have constant effect right-hand sides"
        );
        let rhs = Interval::singleton(float_tolerance::canonicalize(initial[source.index()]));
        let interval = &mut intervals[effect.affected_var_id().index()];
        interval.apply_op(effect.operation(), &rhs);
        *interval = interval.canonicalized();
    }
    for (numeric_var_id, variable) in variables.iter().enumerate() {
        if variable.get_type() != &NumericType::Derived {
            continue;
        }
        match &additive_views[numeric_var_id] {
            Some(view) => {
                let delta =
                    Interval::singleton(float_tolerance::canonicalize(view.operator_delta(op_id)?));
                intervals[numeric_var_id] = (intervals[numeric_var_id] + delta).canonicalized();
            }
            None => ensure!(
                intervals[numeric_var_id] == Interval::unbounded(),
                "derived numeric variable {numeric_var_id} has no additive view but is constrained to {:?}",
                intervals[numeric_var_id]
            ),
        }
    }
    Ok(())
}

/// The values of `numeric_var_id` before `op_id` whose image lies in
/// `target`, or `None` when no value's does.
///
/// Exact for the coordinates an abstraction splits: a derived additive
/// coordinate shifts back by the operator's delta, and a regular variable
/// inverts its effect, whose right-hand side is constant.
pub(crate) fn numeric_effect_preimage(
    task: &dyn AbstractNumericTask,
    additive_views: &[Option<AdditiveNumericView>],
    target: Interval,
    op_id: OperatorIndex,
    numeric_var_id: VariableIndex,
) -> Result<Option<Interval>> {
    let mut preimage = target;
    if let Some(view) = &additive_views[numeric_var_id.index()] {
        let delta = float_tolerance::canonicalize(view.operator_delta(op_id)?);
        preimage.apply_reverse_op(&AssignmentOperation::Plus, &Interval::singleton(delta));
        return Ok(Some(preimage));
    }
    let initial = task.get_initial_numeric_state_values();
    for effect in task.get_operators()[op_id.index()]
        .assignment_effects()
        .iter()
        .filter(|effect| effect.affected_var_id() == numeric_var_id)
        .rev()
    {
        let rhs = float_tolerance::canonicalize(initial[effect.var_id().index()]);
        match effect.operation() {
            AssignmentOperation::Assign => {
                if !preimage.contains(rhs) {
                    return Ok(None);
                }
                preimage = Interval::unbounded();
            }
            AssignmentOperation::Times if rhs == ZERO_VALUE => {
                if !preimage.contains(ZERO_VALUE) {
                    return Ok(None);
                }
                preimage = Interval::unbounded();
            }
            operation => preimage.apply_reverse_op(operation, &Interval::singleton(rhs)),
        }
        preimage = preimage.canonicalized();
    }
    Ok(Some(preimage))
}

/// The values of `numeric_var_id` whose progression through `ops` lies in
/// `target`, or `None` when no value's does.
pub(crate) fn regress_numeric_interval(
    task: &dyn AbstractNumericTask,
    additive_views: &[Option<AdditiveNumericView>],
    target: Interval,
    ops: &[OperatorIndex],
    numeric_var_id: VariableIndex,
) -> Result<Option<Interval>> {
    let mut interval = target;
    for &op_id in ops.iter().rev() {
        match numeric_effect_preimage(task, additive_views, interval, op_id, numeric_var_id)? {
            Some(preimage) if !preimage.is_empty() => interval = preimage,
            _ => return Ok(None),
        }
    }
    Ok(Some(interval))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conditions(step: usize) -> FailedConditions {
        FailedConditions {
            step,
            facts: vec![ExplicitFact::propositional(0, step)],
        }
    }

    #[test]
    fn candidates_regress_the_latest_conditions_to_the_latest_anchor_first() {
        let failed = [conditions(2), conditions(4)];
        let order = regression_candidates(&failed, 2)
            .map(|(anchor, conditions)| (anchor, conditions.step))
            .collect::<Vec<_>>();
        assert_eq!(
            order,
            [(2, 4), (1, 4), (0, 4), (2, 2), (1, 2), (0, 2)],
            "goal-side conditions come first, each from the latest anchor back"
        );
    }

    #[test]
    #[should_panic(expected = "precede the last anchor")]
    fn candidates_reject_conditions_before_the_last_anchor() {
        regression_candidates(&[conditions(1)], 2).for_each(drop);
    }
}
