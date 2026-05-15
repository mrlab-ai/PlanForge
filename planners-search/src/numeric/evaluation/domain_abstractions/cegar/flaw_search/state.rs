use std::collections::BTreeSet;

use anyhow::{Result, ensure};

use planners_sas::numeric::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator,
};
use planners_sas::numeric::numeric_task::{AbstractNumericTask, ExplicitFact, NumericType};
use planners_sas::numeric::utils::errors::{AxiomEvalError, InvalidIndex};
use planners_sas::numeric::{
    axioms::AxiomEvaluator, numeric_task::Operator, utils::int_packer::IntDoublePacker,
};

use crate::numeric::evaluation::domain_abstractions::abstract_operator_generator::DomainMapping;
use crate::numeric::evaluation::domain_abstractions::comparison_expression::{
    Interval, UNBOUNDED_INTERVAL,
};
use crate::numeric::evaluation::domain_abstractions::utils::{fact_is_hold, get_initial_state};

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
    /// Transform a decoded concrete state into a `FlawSearchState`.
    pub fn from_decoded_state(
        prop: Vec<usize>,
        numeric: Vec<f64>,
        domain_mapping: &'a DomainMapping,
    ) -> FlawSearchState<'a> {
        let abstract_prop = prop
            .iter()
            .enumerate()
            .map(|(i, v)| Some(domain_mapping[i][*v]))
            .collect();
        FlawSearchState {
            concrete_prop: prop.into_iter().map(Some).collect(),
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

        let mut state = FlawSearchState {
            concrete_prop: vec![None; task.get_num_variables()],
            abstract_prop: vec![None; task.get_num_variables()],
            numeric: vec![UNBOUNDED_INTERVAL; task.numeric_variables().len()],
            domain_mapping,
            unbounded: true,
        };
        let initial_numeric = task.get_initial_numeric_state_values();
        for (numeric_var_id, numeric_var) in task.numeric_variables().iter().enumerate() {
            if matches!(
                numeric_var.get_type(),
                NumericType::Constant | NumericType::Cost
            ) {
                state.numeric[numeric_var_id] =
                    Interval::singleton(initial_numeric[numeric_var_id]);
            }
        }

        for goal_id in 0..task.get_num_goals() {
            let goal_fact = task.get_goal_fact(goal_id);
            let goal_var = goal_fact.var();
            let goal_is_derived = task.axioms().iter().any(|ax| ax.var_id() == goal_var);
            if goal_is_derived {
                derived_goal_vars.insert(goal_var);
                continue;
            }
            state.set_prop_value(goal_var, goal_fact.value());
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
                    state.set_prop_value(pre.var(), pre.value());
                }
            }
        }

        state
    }

    pub fn set_prop_value(&mut self, var: usize, value: usize) {
        self.concrete_prop[var] = Some(value);
        self.abstract_prop[var] = Some(self.domain_mapping[var][self.concrete_prop[var].unwrap()]);
    }

    pub fn set_numeric_value(&mut self, var: usize, value: f64) {
        self.numeric[var] = Interval::singleton(value);
    }

    pub fn num_concrete_variables(&self) -> usize {
        self.concrete_prop.len()
    }

    pub fn num_numeric_variables(&self) -> usize {
        self.numeric.len()
    }

    pub fn fact_is_hold(&self, fact: &ExplicitFact) -> bool {
        self.value_is_hold_for_var(fact.var(), fact.value())
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

    pub fn evaluate_axioms(
        &mut self,
        axiom_evaluator: &AxiomEvaluator,
    ) -> Result<(), AxiomEvalError> {
        if !axiom_evaluator.has_axioms() {
            return Ok(());
        }
        if axiom_evaluator.has_numeric_axioms() {
            self.evaluate_comparison_axioms(axiom_evaluator)?;
        }
        // Propositional axioms not supported.
        // if axiom_evaluator.has_propositional_axioms() {
        //     self.evaluate_propositional_axioms(axiom_evaluator)?;
        // }
        Ok(())
    }

    pub fn evaluate_comparison_axioms(
        &mut self,
        axiom_evaluator: &AxiomEvaluator,
    ) -> Result<bool, AxiomEvalError> {
        for axiom in axiom_evaluator.numeric_task.comparison_axioms() {
            let is_hold = self.is_hold(axiom).map_err(|e| {
                AxiomEvalError::InvalidIndex(InvalidIndex {
                    length: self.numeric.len(),
                    index: e.index,
                })
            })?;
            self.set_prop_value(axiom.get_affected_var_id(), !is_hold as usize);
        }

        Ok(true)
    }

    pub fn is_hold(&self, axiom: &ComparisonAxiom) -> Result<bool, InvalidIndex> {
        let left = axiom.left_hand_side;
        let right = axiom.right_hand_side;
        if left >= self.numeric.len() || right >= self.numeric.len() {
            return Err(InvalidIndex {
                length: self.numeric.len(),
                index: left,
            });
        }
        let comp_op = &axiom.operator;
        let result = self.compare(comp_op, axiom.left_hand_side, axiom.right_hand_side);
        Ok(result)
    }

    pub fn compare(&self, op: &ComparisonOperator, left: usize, right: usize) -> bool {
        let (left, right) = (self.numeric[left], self.numeric[right]);
        match op {
            ComparisonOperator::LessThan => left.lower_is_lower(&right),
            ComparisonOperator::LessThanOrEqual => left.lower_is_lower_or_equal(&right),
            ComparisonOperator::Equal => left.intersects(&right),
            ComparisonOperator::GreaterThanOrEqual => left.upper_is_higher_or_equal(&right),
            ComparisonOperator::GreaterThan => left.upper_is_higher(&right),
            ComparisonOperator::UnEqual => !left.intersects(&right),
        }
    }

    pub fn evaluate_arithmetic_axioms(
        &mut self,
        axiom_evaluator: &AxiomEvaluator,
    ) -> Result<(), InvalidIndex> {
        for axiom in axiom_evaluator.numeric_task.assignment_axioms() {
            self.update_assignment_axiom_values(axiom)?;
        }

        Ok(())
    }

    pub fn update_assignment_axiom_values(
        &mut self,
        axiom: &AssignmentAxiom,
    ) -> Result<(), InvalidIndex> {
        let left = axiom.left_hand_side;
        let right = axiom.right_hand_side;
        if left >= self.numeric.len() || right >= self.numeric.len() {
            return Err(InvalidIndex {
                length: self.numeric.len(),
                index: left,
            });
        }
        let affected = axiom.affected_var_id;
        if affected >= self.numeric.len() {
            return Err(InvalidIndex {
                length: self.numeric.len(),
                index: affected,
            });
        }
        self.numeric[affected] = match axiom.operator {
            CalOperator::Sum => self.numeric[left] + self.numeric[right],
            CalOperator::Difference => self.numeric[left] - self.numeric[right],
            CalOperator::Product => self.numeric[left] * self.numeric[right],
            CalOperator::Division => {
                if self.numeric[right].any_bound_is_zero() {
                    return Err(InvalidIndex {
                        length: self.numeric.len(),
                        index: right,
                    });
                }
                self.numeric[left] / self.numeric[right]
            }
        };

        Ok(())
    }

    pub fn progress(&mut self, op: &Operator, axiom_evaluator: &AxiomEvaluator) -> Result<()> {
        // Propositional effects (respect conditions).
        for eff in op.effects().iter() {
            let mut ok = true;
            for cond in eff.conditions().iter() {
                if !self.fact_is_hold(cond) {
                    ok = false;
                    break;
                }
            }
            if ok {
                self.set_prop_value(eff.var_id(), eff.value());
            }
        }

        // Numeric assignment effects.
        for eff in op.assignment_effects().iter() {
            if eff.is_conditional() {
                let mut ok = true;
                for cond in eff.conditions().iter() {
                    if !self.fact_is_hold(cond) {
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
            ensure!(
                assignment_var_id < self.numeric.len(),
                "assignment effect source numeric var {assignment_var_id} out of bounds for {} numeric vars",
                self.numeric.len()
            );
            ensure!(
                affected_var_id < self.numeric.len(),
                "assignment effect target numeric var {affected_var_id} out of bounds for {} numeric vars",
                self.numeric.len()
            );
            let operand = self.numeric[assignment_var_id];
            self.numeric[affected_var_id].apply_op(eff.operation(), &operand);
        }

        self.evaluate_arithmetic_axioms(axiom_evaluator)
            .map_err(|e| {
                anyhow::anyhow!("failed to evaluate arithmetic axioms after operator: {e:?}")
            })?;
        self.evaluate_axioms(axiom_evaluator)
            .map_err(|e| anyhow::anyhow!("failed to evaluate axioms after operator: {e:?}"))?;

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
            self.concrete_prop[cond.var()] = Some(cond.value());
            self.abstract_prop[cond.var()] = Some(self.domain_mapping[cond.var()][cond.value()]);
        }

        // Numeric assignment effects (conditional effects not supported).
        for eff in op.assignment_effects().iter() {
            let assignment_var_id = eff.var_id();
            let affected_var_id = eff.affected_var_id();
            if assignment_var_id >= self.numeric.len() || affected_var_id >= self.numeric.len() {
                continue;
            }
            if self.numeric[affected_var_id] == UNBOUNDED_INTERVAL {
                continue;
            }
            let operand = self.numeric[assignment_var_id];
            self.numeric[affected_var_id].apply_reverse_op(eff.operation(), &operand);
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
    let (buffer, numeric_state) = get_initial_state(task, state_packer, axiom_evaluator)?;
    let prop_state = domain_mapping
        .iter()
        .enumerate()
        .map(|(var, _)| state_packer.get(&buffer, var) as usize)
        .collect();

    Ok(FlawSearchState::from_decoded_state(
        prop_state,
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
