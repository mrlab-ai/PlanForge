use std::collections::BTreeSet;

use anyhow::Result;

use planners_sas::numeric::numeric_task::{AbstractNumericTask, ExplicitFact};
use planners_sas::numeric::{
    axioms::AxiomEvaluator, numeric_task::Operator, utils::int_packer::IntDoublePacker,
};

use crate::numeric::evaluation::domain_abstractions::abstract_operator_generator::DomainMapping;
use crate::numeric::evaluation::domain_abstractions::comparison_expression::{
    Interval, UNBOUNDED_INTERVAL,
};
use crate::numeric::evaluation::domain_abstractions::utils::{
    fact_is_hold, set_initial_prop_values,
};

/// States used during the search of flaws.
/// Some variables may have a concrete value (`concrete_prop`), while
/// others only have an abstract value (`abstract_prop`), which is `None` for
/// partial states when the value of the variable is undefined.
/// `abstract_prop` contains the corresponding abstract value of all variables.
/// All numeric variables are directly handled as `Interval`s (often smaller
/// than the `Interval`s of the abstract plan states).
#[derive(Clone, Debug)]
pub struct FlawSearchState<'a> {
    pub concrete_prop: Vec<Option<usize>>,
    pub abstract_prop: Vec<Option<usize>>,
    pub numeric: Vec<Interval>,
    pub domain_mapping: &'a DomainMapping,
    pub unbounded: bool,
}

impl<'a> FlawSearchState<'a> {
    /// Transform a concrete state into a `FlawSearchState`.
    pub fn from_state(
        prop: Vec<u64>,
        numeric: Vec<f64>,
        domain_mapping: &'a DomainMapping,
    ) -> FlawSearchState<'a> {
        let abstract_prop = prop
            .iter()
            .enumerate()
            .map(|(i, v)| Some(domain_mapping[i][*v as usize]))
            .collect();
        FlawSearchState {
            concrete_prop: prop.into_iter().map(|v| Some(v as usize)).collect(),
            abstract_prop,
            numeric: numeric
                .into_iter()
                .map(|v| Interval::new(v, v, true, true))
                .collect(),
            domain_mapping,
            unbounded: false,
        }
    }

    pub fn goals_partial_state(
        task: &dyn AbstractNumericTask,
        domain_mapping: &'a DomainMapping,
    ) -> FlawSearchState<'a> {
        let mut seen: BTreeSet<ExplicitFact> = BTreeSet::new();
        let mut derived_goal_vars: BTreeSet<usize> = BTreeSet::new();
        let mut concrete_prop = vec![None; task.get_num_variables()];
        let mut abstract_prop = vec![None; task.get_num_variables()];
        for goal_id in 0..task.get_num_goals() {
            let goal_fact = task.get_goal_fact(goal_id);
            let goal_var = goal_fact.var;
            let goal_is_derived = task.axioms().iter().any(|ax| ax.var_id() == goal_var);
            if goal_is_derived {
                derived_goal_vars.insert(goal_var);
                continue;
            }
            concrete_prop[goal_var] = Some(goal_fact.value);
            abstract_prop[goal_var] =
                Some(domain_mapping[goal_var][concrete_prop[goal_var].unwrap()]);
        }

        // Reconstruct (potentially hidden) goal conditions from propositional goal axioms.
        for ax in task.axioms().iter() {
            if ax.conditions().is_empty() {
                continue;
            }
            if !derived_goal_vars.is_empty() && !derived_goal_vars.contains(&ax.var_id()) {
                continue;
            }
            for pre in ax.conditions().iter() {
                if seen.insert(pre.clone()) {
                    concrete_prop[pre.var] = Some(pre.value);
                    abstract_prop[pre.var] =
                        Some(domain_mapping[pre.var][concrete_prop[pre.var].unwrap()]);
                }
            }
        }

        FlawSearchState {
            concrete_prop,
            abstract_prop,
            numeric: vec![UNBOUNDED_INTERVAL; task.numeric_variables().len()],
            domain_mapping,
            unbounded: true,
        }
    }

    pub fn num_concrete_variables(&self) -> usize {
        self.concrete_prop.len()
    }

    pub fn num_numeric_variables(&self) -> usize {
        self.numeric.len()
    }

    pub fn fact_is_hold(&self, fact: &ExplicitFact) -> bool {
        self.value_is_hold_for_var(fact.var, fact.value)
    }

    pub fn value_is_hold_for_var(&self, var: usize, value: usize) -> bool {
        match self.concrete_prop[var] {
            Some(v) => v == value,
            None => {
                self.abstract_prop[var].is_none()
                    || self.domain_mapping[var][value] == self.abstract_prop[var].unwrap()
            }
        }
    }

    pub fn revert_axioms(&mut self, axiom_evaluator: &AxiomEvaluator) -> Result<()> {
        let mut affected_prop_vars_by_axioms = Vec::with_capacity(self.concrete_prop.len());
        axiom_evaluator.affected_propositional_vars(&mut affected_prop_vars_by_axioms);
        for var in &affected_prop_vars_by_axioms {
            self.concrete_prop[*var] = None;
            self.abstract_prop[*var] = None;
        }

        if !self.unbounded {
            let mut affected_numeric_vars_by_axioms = Vec::with_capacity(self.numeric.len());
            axiom_evaluator.affected_numeric_vars(&mut affected_numeric_vars_by_axioms);
            for var in &affected_numeric_vars_by_axioms {
                self.numeric[*var] = UNBOUNDED_INTERVAL;
            }
        }
        Ok(())
    }

    pub fn regress(&mut self, op: &Operator, axiom_evaluator: &AxiomEvaluator) -> Result<()> {
        // Variables affected by axioms set to `undefined`.
        self.revert_axioms(axiom_evaluator)?;

        // Propositional effects (conditional effects not supported).
        for eff in op.effects().iter() {
            self.concrete_prop[eff.var_id()] = None;
            self.abstract_prop[eff.var_id()] = None;
        }
        // Propositional preconditions.
        for cond in op.preconditions() {
            self.concrete_prop[cond.var] = Some(cond.value);
            self.abstract_prop[cond.var] = Some(self.domain_mapping[cond.var][cond.value]);
        }

        // Numeric assignment effects (conditional effects not supported).
        if !self.unbounded {
            for eff in op.assignment_effects().iter() {
                let assignment_var_id = eff.var_id();
                let affected_var_id = eff.affected_var_id();
                let operand = self.numeric[assignment_var_id];
                self.numeric[affected_var_id].apply_reverse_op(eff.operation(), &operand);
            }
        }

        Ok(())
    }
}

pub fn get_initial_flaw_search_state<'a>(
    task: &dyn AbstractNumericTask,
    state_packer: &IntDoublePacker,
    axiom_evaluator: &AxiomEvaluator,
    domain_mapping: &'a DomainMapping,
) -> Result<FlawSearchState<'a>> {
    let mut buffer = vec![0u64; state_packer.num_bins()];
    set_initial_prop_values(task, state_packer, &mut buffer);
    let mut numeric_state: Vec<f64> = task.get_initial_numeric_state_values().to_vec();

    axiom_evaluator
        .evaluate_arithmetic_axioms(&mut numeric_state)
        .map_err(|e| {
            anyhow::anyhow!("failed to evaluate arithmetic axioms for initial state: {e:?}")
        })?;
    axiom_evaluator
        .evaluate(&mut buffer, &mut numeric_state)
        .map_err(|e| anyhow::anyhow!("failed to evaluate axioms for initial state: {e:?}"))?;

    Ok(FlawSearchState::from_state(
        buffer,
        numeric_state,
        domain_mapping,
    ))
}

pub fn progress(
    op: &Operator,
    axiom_evaluator: &AxiomEvaluator,
    packer: &IntDoublePacker,
    prop_state: &mut [u64],
    numeric_state: &mut [f64],
) -> Result<()> {
    // Propositional effects (respect conditions).
    for eff in op.effects().iter() {
        let mut ok = true;
        for cond in eff.conditions().iter() {
            if !fact_is_hold(cond, packer, prop_state) {
                ok = false;
                break;
            }
        }
        if ok {
            packer.set(prop_state, eff.var_id(), eff.value() as u64);
        }
    }

    // Numeric assignment effects.
    for eff in op.assignment_effects().iter() {
        if eff.is_conditional() {
            let mut ok = true;
            for cond in eff.conditions().iter() {
                if !fact_is_hold(cond, packer, prop_state) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
        }

        let assignment_var_id = eff.var_id();
        let affected_var_id = eff.affected_var_id();
        if assignment_var_id >= numeric_state.len() || affected_var_id >= numeric_state.len() {
            continue;
        }
        let operand = numeric_state[assignment_var_id];
        numeric_state[affected_var_id] =
            planners_sas::numeric::numeric_task::AssignmentOperation::apply(
                numeric_state[affected_var_id],
                eff.operation(),
                operand,
            );
    }

    axiom_evaluator
        .evaluate_arithmetic_axioms(numeric_state)
        .map_err(|e| {
            anyhow::anyhow!("failed to evaluate arithmetic axioms after operator: {e:?}")
        })?;
    axiom_evaluator
        .evaluate(prop_state, numeric_state)
        .map_err(|e| anyhow::anyhow!("failed to evaluate axioms after operator: {e:?}"))?;

    Ok(())
}
