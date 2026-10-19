//! Concrete states of a task, closed under its axioms, and the packed
//! propositional layout they use.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitValueIndex, NumericValue, OperatorIndex,
};
use planforge_sas::state_registry::ConcreteStateView;
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::state_packer::StatePacker;

use super::cegar::progress_concrete_state;

/// A concrete state of a task, closed under its axioms. Propositional values are
/// packed with the task's [`make_prop_state_packer`] layout.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConcreteState {
    pub(crate) propositions: Vec<u64>,
    pub(crate) numeric: Vec<NumericValue>,
}

impl ConcreteState {
    /// The task's initial state.
    pub(crate) fn initial(task: &dyn AbstractNumericTask) -> Result<Self> {
        let state_packer = Arc::new(make_prop_state_packer(task));
        let axiom_evaluator = AxiomEvaluator::new(Arc::new(task), state_packer.clone());
        let (propositions, numeric) = get_initial_state(task, &state_packer, &axiom_evaluator)?;
        Ok(Self {
            propositions,
            numeric,
        })
    }

    /// Whether `goal_id` of `task` holds in this state.
    pub(crate) fn satisfies_goal(
        &self,
        task: &dyn AbstractNumericTask,
        goal_id: usize,
    ) -> Result<bool> {
        let state_packer = self.checked_packer(task)?;
        ensure!(
            goal_id < task.get_num_goals(),
            "goal id {goal_id} exceeds {} goals",
            task.get_num_goals()
        );
        Ok(self.fact_holds(task, &state_packer, goal_id))
    }

    /// How many goals of `task` hold in this state.
    pub(crate) fn satisfied_goal_count(&self, task: &dyn AbstractNumericTask) -> Result<usize> {
        let state_packer = self.checked_packer(task)?;
        Ok((0..task.get_num_goals())
            .filter(|&goal_id| self.fact_holds(task, &state_packer, goal_id))
            .count())
    }

    /// Apply `operator_ids` in sequence, failing on an inapplicable operator.
    pub(crate) fn replay(
        &self,
        task: &dyn AbstractNumericTask,
        operator_ids: &[OperatorIndex],
    ) -> Result<Self> {
        let state_packer = Arc::new(self.checked_packer(task)?);
        let axiom_evaluator = AxiomEvaluator::new(Arc::new(task), state_packer.clone());
        let mut next = self.clone();
        for (step, &operator_id) in operator_ids.iter().enumerate() {
            let operator = task
                .get_operators()
                .get(operator_id.index())
                .with_context(|| {
                    format!("progressive plan step {step} has invalid operator {operator_id}")
                })?;
            ensure!(
                operator.preconditions().iter().all(|fact| fact.is_held(
                    ConcreteStateView::from_decoded(&state_packer, &next.propositions, &[])
                )),
                "progressive plan operator {operator_id} ({}) is inapplicable at step {step}",
                operator.name()
            );
            progress_concrete_state(
                operator,
                &axiom_evaluator,
                &state_packer,
                &mut next.propositions,
                &mut next.numeric,
            )?;
        }
        Ok(next)
    }

    /// Unpacked propositional values, one per task variable.
    pub(crate) fn propositional_values(
        &self,
        task: &dyn AbstractNumericTask,
    ) -> Result<Vec<ExplicitValueIndex>> {
        let state_packer = self.checked_packer(task)?;
        Ok((0..task.get_num_variables())
            .map(|var_id| {
                ExplicitValueIndex::from_usize(
                    usize::try_from(state_packer.get(&self.propositions, var_id))
                        .expect("packed propositional value fits usize"),
                )
            })
            .collect())
    }

    fn checked_packer(&self, task: &dyn AbstractNumericTask) -> Result<StatePacker> {
        let state_packer = make_prop_state_packer(task);
        ensure!(
            self.propositions.len() == state_packer.num_bins(),
            "concrete state has {} proposition bins, expected {}",
            self.propositions.len(),
            state_packer.num_bins()
        );
        ensure!(
            self.numeric.len() == task.numeric_variables().len(),
            "concrete state has {} numeric values, expected {}",
            self.numeric.len(),
            task.numeric_variables().len()
        );
        Ok(state_packer)
    }

    fn fact_holds(
        &self,
        task: &dyn AbstractNumericTask,
        state_packer: &StatePacker,
        goal_id: usize,
    ) -> bool {
        task.get_goal_fact(goal_id)
            .is_held(ConcreteStateView::from_decoded(
                state_packer,
                &self.propositions,
                &[],
            ))
    }
}

pub(crate) fn make_prop_state_packer(task: &dyn AbstractNumericTask) -> StatePacker {
    let mut domain_sizes: Vec<u64> = Vec::with_capacity(task.variables().len());
    for var in task.variables().iter() {
        domain_sizes.push(var.domain_size() as u64);
    }
    StatePacker::new(&domain_sizes)
}

pub(crate) fn set_initial_prop_values(
    task: &dyn AbstractNumericTask,
    packer: &StatePacker,
    buffer: &mut [u64],
) {
    let init = task.get_initial_propositional_state_values();
    for (var_id, &val) in init.iter().enumerate() {
        packer.set(buffer, var_id, val.index() as u64);
    }
}

pub(crate) fn get_initial_state(
    task: &dyn AbstractNumericTask,
    state_packer: &StatePacker,
    axiom_evaluator: &AxiomEvaluator,
) -> Result<(Vec<u64>, Vec<NumericValue>)> {
    let mut buffer = vec![0u64; state_packer.num_bins()];
    set_initial_prop_values(task, state_packer, &mut buffer);
    let mut numeric_state: Vec<NumericValue> = task.get_initial_numeric_state_values().to_vec();
    for value in &mut numeric_state {
        *value = float_tolerance::canonicalize(*value);
    }

    axiom_evaluator
        .evaluate(&mut buffer, &mut numeric_state)
        .map_err(|e| anyhow::anyhow!("failed to evaluate axioms for initial state: {e:?}"))?;

    Ok((buffer, numeric_state))
}
