use anyhow::{Result, ensure};

use planforge_sas::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitValueIndex, NumericType, NumericValue, VariableIndex,
};
use planforge_sas::utils::errors::{AxiomEvalError, InvalidIndex};
use planforge_sas::{
    axioms::AxiomEvaluator, numeric_task::Operator, utils::state_packer::StatePacker,
};

use crate::evaluation::domain_abstractions::abstract_operator_generator::DomainMapping;
use crate::evaluation::domain_abstractions::cegar::flaw_search::goal_facts;
use crate::evaluation::domain_abstractions::utils::get_initial_state;
use planforge_sas::utils::interval::{Interval, UNBOUNDED_INTERVAL};

/// States used during the search of flaws.
/// Some variables may have a concrete value (`concrete_prop`), while
/// others only have an abstract value (`abstract_prop`), which is `None` for
/// partial states when the value of the variable is undefined.
/// `abstract_prop` contains the corresponding abstract value of all variables.
/// All numeric variables are directly handled as `Interval`s (often smaller
/// than the `Interval`s of the abstract plan states).
#[derive(Clone, Debug)]
pub struct FlawSearchState<'a> {
    pub concrete_prop: Vec<Option<ExplicitValueIndex>>,
    pub abstract_prop: Vec<Option<ExplicitValueIndex>>,
    pub numeric: Vec<Interval>,
    pub domain_mapping: &'a DomainMapping,
    pub unbounded: bool,
}

impl<'a> FlawSearchState<'a> {
    /// Transform a decoded concrete state into a `FlawSearchState`.
    pub fn from_decoded_state(
        prop: Vec<ExplicitValueIndex>,
        numeric: Vec<NumericValue>,
        domain_mapping: &'a DomainMapping,
    ) -> FlawSearchState<'a> {
        let abstract_prop = prop
            .iter()
            .enumerate()
            .map(|(i, v)| Some(domain_mapping[i][v.index()]))
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

        for requirement in goal_facts(task) {
            state.set_prop_value(requirement.var_index(), requirement.value_index());
        }

        state
    }

    pub fn set_prop_value(&mut self, var: VariableIndex, value: ExplicitValueIndex) {
        self.concrete_prop[var.index()] = Some(value);
        self.abstract_prop[var.index()] = Some(
            self.domain_mapping[var.index()][self.concrete_prop[var.index()].unwrap().index()],
        );
    }

    pub fn set_numeric_value(&mut self, var: VariableIndex, value: NumericValue) {
        self.numeric[var.index()] = Interval::singleton(value);
    }

    pub fn num_concrete_variables(&self) -> usize {
        self.concrete_prop.len()
    }

    pub fn num_numeric_variables(&self) -> usize {
        self.numeric.len()
    }

    pub fn fact_is_held(&self, fact: &ExplicitFact) -> bool {
        self.value_is_held_for_var(fact.var_index(), fact.value_index())
    }

    pub fn value_is_held_for_var(&self, var: VariableIndex, value: ExplicitValueIndex) -> bool {
        match self.concrete_prop[var.index()] {
            Some(v) => v == value,
            None => {
                self.abstract_prop[var.index()].is_none()
                    || self.domain_mapping[var.index()][value.index()]
                        == self.abstract_prop[var.index()].unwrap()
            }
        }
    }

    pub fn revert_axioms(&mut self, axiom_evaluator: &AxiomEvaluator) -> Result<()> {
        let mut affected_prop_vars_by_axioms = Vec::with_capacity(self.concrete_prop.len());
        axiom_evaluator.affected_propositional_vars(&mut affected_prop_vars_by_axioms);
        for var in &affected_prop_vars_by_axioms {
            self.concrete_prop[var.index()] = None;
            self.abstract_prop[var.index()] = None;
        }

        if !self.unbounded {
            let mut affected_numeric_vars_by_axioms = Vec::with_capacity(self.numeric.len());
            axiom_evaluator.affected_numeric_vars(&mut affected_numeric_vars_by_axioms);
            for var in &affected_numeric_vars_by_axioms {
                self.numeric[var.index()] = UNBOUNDED_INTERVAL;
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
            let is_held = self.is_held(axiom).map_err(|e| {
                AxiomEvalError::InvalidIndex(InvalidIndex {
                    length: self.numeric.len(),
                    index: e.index,
                })
            })?;
            self.set_prop_value(
                axiom.get_affected_var_id(),
                ExplicitValueIndex::from_usize(!is_held as usize),
            );
        }

        Ok(true)
    }

    pub fn is_held(&self, axiom: &ComparisonAxiom) -> Result<bool, InvalidIndex> {
        let left = axiom.left_hand_side;
        let right = axiom.right_hand_side;
        if left.index() >= self.numeric.len() || right.index() >= self.numeric.len() {
            return Err(InvalidIndex {
                length: self.numeric.len(),
                index: left.index(),
            });
        }
        let comp_op = &axiom.operator;
        let result = self.compare(comp_op, axiom.left_hand_side, axiom.right_hand_side);
        Ok(result)
    }

    pub fn compare(
        &self,
        op: &ComparisonOperator,
        left: VariableIndex,
        right: VariableIndex,
    ) -> bool {
        let (left, right) = (self.numeric[left.index()], self.numeric[right.index()]);
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
        if left.index() >= self.numeric.len() || right.index() >= self.numeric.len() {
            return Err(InvalidIndex {
                length: self.numeric.len(),
                index: left.index(),
            });
        }
        let affected = axiom.affected_var_id;
        if affected.index() >= self.numeric.len() {
            return Err(InvalidIndex {
                length: self.numeric.len(),
                index: affected.index(),
            });
        }
        self.numeric[affected.index()] = match axiom.operator {
            CalOperator::Sum => self.numeric[left.index()] + self.numeric[right.index()],
            CalOperator::Difference => self.numeric[left.index()] - self.numeric[right.index()],
            CalOperator::Product => self.numeric[left.index()] * self.numeric[right.index()],
            CalOperator::Division => {
                if self.numeric[right.index()].any_bound_is_zero() {
                    return Err(InvalidIndex {
                        length: self.numeric.len(),
                        index: right.index(),
                    });
                }
                self.numeric[left.index()] / self.numeric[right.index()]
            }
        };

        Ok(())
    }

    pub fn progress(&mut self, op: &Operator, axiom_evaluator: &AxiomEvaluator) -> Result<()> {
        // Propositional effects (respect conditions).
        for eff in op.effects().iter() {
            let mut ok = true;
            for cond in eff.conditions().iter() {
                if !self.fact_is_held(cond) {
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
                    if !self.fact_is_held(cond) {
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
                assignment_var_id.index() < self.numeric.len(),
                "assignment effect source numeric var {assignment_var_id} out of bounds for {} numeric vars",
                self.numeric.len()
            );
            ensure!(
                affected_var_id.index() < self.numeric.len(),
                "assignment effect target numeric var {affected_var_id} out of bounds for {} numeric vars",
                self.numeric.len()
            );
            let operand = self.numeric[assignment_var_id.index()];
            self.numeric[affected_var_id.index()].apply_op(eff.operation(), &operand);
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
            self.concrete_prop[eff.var_id().index()] = None;
            self.abstract_prop[eff.var_id().index()] = None;
        }
        // Propositional preconditions.
        for cond in op.preconditions() {
            self.concrete_prop[cond.var()] = Some(cond.value_index());
            self.abstract_prop[cond.var()] = Some(self.domain_mapping[cond.var()][cond.value()]);
        }

        // Numeric assignment effects (conditional effects not supported).
        for eff in op.assignment_effects().iter() {
            let assignment_var_id = eff.var_id();
            let affected_var_id = eff.affected_var_id();
            if assignment_var_id.index() >= self.numeric.len()
                || affected_var_id.index() >= self.numeric.len()
            {
                continue;
            }
            if self.numeric[affected_var_id.index()] == UNBOUNDED_INTERVAL {
                continue;
            }
            let operand = self.numeric[assignment_var_id.index()];
            self.numeric[affected_var_id.index()].apply_reverse_op(eff.operation(), &operand);
        }

        Ok(())
    }
}

pub fn get_initial_flaw_search_state<'a>(
    task: &dyn AbstractNumericTask,
    state_packer: &StatePacker,
    axiom_evaluator: &AxiomEvaluator,
    domain_mapping: &'a DomainMapping,
) -> Result<FlawSearchState<'a>> {
    let (buffer, numeric_state) = get_initial_state(task, state_packer, axiom_evaluator)?;
    let prop_state = domain_mapping
        .iter()
        .enumerate()
        .map(|(var, _)| ExplicitValueIndex::from_usize(state_packer.get(&buffer, var) as usize))
        .collect();

    Ok(FlawSearchState::from_decoded_state(
        prop_state,
        numeric_state,
        domain_mapping,
    ))
}
