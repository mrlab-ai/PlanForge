#[cfg(test)]
mod tests;

use std::cell::{Ref, RefCell, RefMut};
use std::collections::{BTreeSet, HashSet};
use std::fmt;
use std::rc::Rc;

use planforge_sas::axioms::{AssignmentAxiom, ComparisonAxiom, PropositionalAxiom};
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentEffect, Effect, ExplicitFact, ExplicitVariable, Metric,
    NumericRootTask, NumericType, NumericVariable, Operator,
    metric_operator_cost_from_initial_values,
};
use planforge_sas::state_registry::{ConcreteState, StateRegistry};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::int_packer::IntDoublePacker;

use crate::evaluation::pattern_databases::compiled_axiom_evaluator::{
    CompiledAxiomEvaluator, CompiledAxiomEvaluatorData, CompiledAxiomEvaluatorScratch,
};
use crate::task_restriction::validate_restricted_task;

pub type EvaluatedState = (Vec<usize>, Vec<f64>, Vec<u64>);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pattern {
    pub regular: Vec<usize>,
    pub numeric: Vec<usize>,
}

impl Pattern {
    pub fn new(regular: Vec<usize>, numeric: Vec<usize>) -> Self {
        let mut pattern = Self { regular, numeric };
        pattern.normalize_in_place();
        pattern
    }

    pub fn normalized(&self) -> Self {
        let mut pattern = self.clone();
        pattern.normalize_in_place();
        pattern
    }

    pub fn normalize_in_place(&mut self) {
        self.regular.sort_unstable();
        self.regular.dedup();
        self.numeric.sort_unstable();
        self.numeric.dedup();
    }

    pub fn is_empty(&self) -> bool {
        self.regular.is_empty() && self.numeric.is_empty()
    }

    pub fn total_len(&self) -> usize {
        self.regular.len() + self.numeric.len()
    }

    pub fn add_regular_var(&mut self, var_id: usize) -> bool {
        match self.regular.binary_search(&var_id) {
            Ok(_) => false,
            Err(index) => {
                self.regular.insert(index, var_id);
                true
            }
        }
    }

    pub fn add_numeric_var(&mut self, var_id: usize) -> bool {
        match self.numeric.binary_search(&var_id) {
            Ok(_) => false,
            Err(index) => {
                self.numeric.insert(index, var_id);
                true
            }
        }
    }

    pub fn is_subset_of(&self, other: &Self) -> bool {
        is_sorted_subset(&self.regular, &other.regular)
            && is_sorted_subset(&self.numeric, &other.numeric)
    }
}

fn is_sorted_subset(lhs: &[usize], rhs: &[usize]) -> bool {
    let mut lhs_index = 0;
    let mut rhs_index = 0;

    while lhs_index < lhs.len() && rhs_index < rhs.len() {
        match lhs[lhs_index].cmp(&rhs[rhs_index]) {
            std::cmp::Ordering::Less => return false,
            std::cmp::Ordering::Equal => {
                lhs_index += 1;
                rhs_index += 1;
            }
            std::cmp::Ordering::Greater => rhs_index += 1,
        }
    }

    lhs_index == lhs.len()
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProjectedTaskBuildError {
    UnrestrictedTask {
        reason: String,
    },
    InvalidRegularVarId {
        provided: usize,
        len: usize,
    },
    InvalidNumericVarId {
        provided: usize,
        len: usize,
    },
    UnsupportedPatternNumericVarType {
        numeric_var_id: usize,
        numeric_type: NumericType,
    },
    InitialStateEvaluationFailed {
        reason: String,
    },
}

impl fmt::Display for ProjectedTaskBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnrestrictedTask { reason } => write!(formatter, "{reason}"),
            Self::InvalidRegularVarId { provided, len } => write!(
                formatter,
                "invalid projected propositional variable {provided}; task has {len} variables"
            ),
            Self::InvalidNumericVarId { provided, len } => write!(
                formatter,
                "invalid projected numeric variable {provided}; task has {len} numeric variables"
            ),
            Self::UnsupportedPatternNumericVarType {
                numeric_var_id,
                numeric_type,
            } => write!(
                formatter,
                "pattern numeric variable {numeric_var_id} has unsupported type {:?}; PDB patterns over restricted tasks accept only regular numeric variables",
                numeric_type
            ),
            Self::InitialStateEvaluationFailed { reason } => write!(
                formatter,
                "failed to evaluate projected initial state: {reason}"
            ),
        }
    }
}

impl std::error::Error for ProjectedTaskBuildError {}

#[allow(unused)]
#[derive(Clone)]
pub struct ProjectedTask<'task> {
    base: &'task dyn AbstractNumericTask,
    variables: Vec<ExplicitVariable>,
    numeric_variables: Vec<NumericVariable>,
    assignment_axioms: Vec<AssignmentAxiom>,
    comparison_axioms: Vec<ComparisonAxiom>,
    axioms: Vec<PropositionalAxiom>,
    metric: Metric,
    operators: Vec<Operator>,
    operator_costs: Vec<f64>,
    base_operator_ids: Vec<usize>,
    propositional_packer: IntDoublePacker,
    initial_packed_propositional: Vec<u64>,
    compiled_axiom_evaluator_data: CompiledAxiomEvaluatorData,
    compiled_axiom_evaluator_scratch: RefCell<CompiledAxiomEvaluatorScratch>,
    operator_effect_facts: Vec<Vec<ExplicitFact>>,
    goals: Vec<ExplicitFact>,
    axiom_effect_facts: Vec<ExplicitFact>,
    state: Rc<RefCell<Vec<usize>>>,
    numeric_state: Rc<RefCell<Vec<f64>>>,
    projected_var_to_original: Vec<usize>,
    projected_num_var_to_original: Vec<usize>,
    original_var_to_projected: Vec<Option<usize>>,
    original_num_var_to_projected: Vec<Option<usize>>,
    pattern_regular_projected_ids: Vec<usize>,
    pattern_numeric_projected_ids: Vec<usize>,
    variable_names: Vec<String>,
    fact_names: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
pub(super) struct PatternLookupProjection {
    base_propositional_len: usize,
    source_numeric_len: usize,
    pattern_regular_original_ids: Vec<usize>,
    pattern_numeric_lookup: Vec<usize>,
    projected_regular_original_ids: Vec<usize>,
    projected_numeric_lookup: Vec<usize>,
}

impl PatternLookupProjection {
    pub(super) fn from_projected_task(task: &ProjectedTask<'_>) -> Self {
        let pattern_regular_original_ids = task
            .pattern_regular_projected_ids
            .iter()
            .map(|&projected_var_id| task.projected_var_to_original[projected_var_id])
            .collect();

        let pattern_numeric_lookup = task
            .pattern_numeric_projected_ids
            .iter()
            .map(|&projected_numeric_id| task.projected_num_var_to_original[projected_numeric_id])
            .collect();

        let projected_regular_original_ids = task.projected_var_to_original.clone();
        let projected_numeric_lookup = task.projected_num_var_to_original.clone();

        Self {
            base_propositional_len: task.base.variables().len(),
            source_numeric_len: task.base.numeric_variables().len(),
            pattern_regular_original_ids,
            pattern_numeric_lookup,
            projected_regular_original_ids,
            projected_numeric_lookup,
        }
    }

    pub(super) fn compact_prop_hash_from_state_values(
        &self,
        propositional_values: &[usize],
        multipliers: &[usize],
    ) -> Result<usize, String> {
        if self.pattern_regular_original_ids.len() != multipliers.len() {
            return Err("pattern regular ids and multipliers length mismatch".to_string());
        }
        if propositional_values.len() < self.base_propositional_len {
            return Err(format!(
                "propositional state too short: {} < {}",
                propositional_values.len(),
                self.base_propositional_len
            ));
        }

        let mut hash = 0usize;
        for (&original_var_id, &multiplier) in self
            .pattern_regular_original_ids
            .iter()
            .zip(multipliers.iter())
        {
            let value = propositional_values[original_var_id];
            hash = hash.saturating_add(value.saturating_mul(multiplier));
        }
        Ok(hash)
    }

    #[inline]
    pub(super) fn compact_prop_hash_from_state_values_unchecked(
        &self,
        propositional_values: &[usize],
        multipliers: &[usize],
    ) -> usize {
        debug_assert_eq!(self.pattern_regular_original_ids.len(), multipliers.len());
        debug_assert!(propositional_values.len() >= self.base_propositional_len);

        let mut hash = 0usize;
        for (&original_var_id, &multiplier) in self
            .pattern_regular_original_ids
            .iter()
            .zip(multipliers.iter())
        {
            let value = propositional_values[original_var_id];
            hash = hash.saturating_add(value.saturating_mul(multiplier));
        }
        hash
    }

    fn numeric_value_from_source(
        original_numeric_var: usize,
        source_numeric_values: &[f64],
    ) -> Result<f64, String> {
        source_numeric_values
            .get(original_numeric_var)
            .copied()
            .ok_or_else(|| {
                format!("source numeric state too short for index {original_numeric_var}")
            })
    }

    pub(super) fn fill_pattern_numeric_bins_from_source_numeric_into(
        &self,
        source_numeric_values: &[f64],
        bins: &mut Vec<u64>,
    ) -> Result<(), String> {
        if source_numeric_values.len() < self.source_numeric_len {
            return Err(format!(
                "source numeric state too short: {} < {}",
                source_numeric_values.len(),
                self.source_numeric_len
            ));
        }

        bins.clear();
        bins.resize(1 + self.pattern_numeric_lookup.len(), 0);
        for (numeric_index, &original_numeric_var) in self.pattern_numeric_lookup.iter().enumerate()
        {
            bins[numeric_index + 1] = float_tolerance::canonical_bits(
                Self::numeric_value_from_source(original_numeric_var, source_numeric_values)?,
            );
        }
        Ok(())
    }

    #[inline]
    pub(super) fn fill_pattern_numeric_bins_from_source_numeric_into_unchecked(
        &self,
        source_numeric_values: &[f64],
        bins: &mut Vec<u64>,
    ) {
        debug_assert!(source_numeric_values.len() >= self.source_numeric_len);

        bins.clear();
        bins.resize(1 + self.pattern_numeric_lookup.len(), 0);
        for (numeric_index, &original_numeric_var) in self.pattern_numeric_lookup.iter().enumerate()
        {
            let value = source_numeric_values[original_numeric_var];
            bins[numeric_index + 1] = float_tolerance::canonical_bits(value);
        }
    }

    pub(super) fn pack_pattern_state_values_from_source_numeric_into(
        &self,
        propositional_values: &[usize],
        source_numeric_values: &[f64],
        packer: &IntDoublePacker,
        packed_values: &mut Vec<u64>,
    ) -> Result<(), String> {
        if propositional_values.len() < self.base_propositional_len {
            return Err(format!(
                "propositional state too short: {} < {}",
                propositional_values.len(),
                self.base_propositional_len
            ));
        }
        if source_numeric_values.len() < self.source_numeric_len {
            return Err(format!(
                "source numeric state too short: {} < {}",
                source_numeric_values.len(),
                self.source_numeric_len
            ));
        }

        packed_values.clear();
        packed_values.resize(packer.num_bins(), 0);

        for (compact_index, &original_var_id) in
            self.pattern_regular_original_ids.iter().enumerate()
        {
            packer.set(
                packed_values,
                compact_index,
                propositional_values[original_var_id] as u64,
            );
        }

        let prop_len = self.pattern_regular_original_ids.len();
        for (numeric_index, &original_numeric_var) in self.pattern_numeric_lookup.iter().enumerate()
        {
            packer.set(
                packed_values,
                prop_len + numeric_index,
                packer.pack_double(Self::numeric_value_from_source(
                    original_numeric_var,
                    source_numeric_values,
                )?),
            );
        }

        Ok(())
    }

    pub(super) fn project_state_values_from_source_numeric_into(
        &self,
        propositional_values: &[usize],
        source_numeric_values: &[f64],
        projected_prop_values: &mut Vec<usize>,
        projected_numeric_values: &mut Vec<f64>,
    ) -> Result<(), String> {
        if propositional_values.len() < self.base_propositional_len {
            return Err(format!(
                "propositional state too short: {} < {}",
                propositional_values.len(),
                self.base_propositional_len
            ));
        }
        if source_numeric_values.len() < self.source_numeric_len {
            return Err(format!(
                "source numeric state too short: {} < {}",
                source_numeric_values.len(),
                self.source_numeric_len
            ));
        }

        projected_prop_values.clear();
        projected_prop_values.extend(
            self.projected_regular_original_ids
                .iter()
                .map(|&original_var_id| propositional_values[original_var_id]),
        );

        projected_numeric_values.clear();
        projected_numeric_values.reserve(self.projected_numeric_lookup.len());
        for &original_numeric_var in &self.projected_numeric_lookup {
            projected_numeric_values.push(Self::numeric_value_from_source(
                original_numeric_var,
                source_numeric_values,
            )?);
        }
        Ok(())
    }

    #[inline]
    pub(super) fn project_state_values_from_source_numeric_into_unchecked(
        &self,
        propositional_values: &[usize],
        source_numeric_values: &[f64],
        projected_prop_values: &mut Vec<usize>,
        projected_numeric_values: &mut Vec<f64>,
    ) {
        debug_assert!(propositional_values.len() >= self.base_propositional_len);
        debug_assert!(source_numeric_values.len() >= self.source_numeric_len);

        projected_prop_values.clear();
        projected_prop_values.extend(
            self.projected_regular_original_ids
                .iter()
                .map(|&original_var_id| propositional_values[original_var_id]),
        );

        projected_numeric_values.clear();
        projected_numeric_values.reserve(self.projected_numeric_lookup.len());
        for &original_numeric_var in &self.projected_numeric_lookup {
            projected_numeric_values.push(source_numeric_values[original_numeric_var]);
        }
    }
}

impl<'task> ProjectedTask<'task> {
    pub fn new(
        base: &'task dyn AbstractNumericTask,
        pattern: &Pattern,
    ) -> Result<Self, ProjectedTaskBuildError> {
        validate_restricted_task(base)
            .map_err(|reason| ProjectedTaskBuildError::UnrestrictedTask { reason })?;
        let num_vars = base.variables().len();
        let num_numeric_vars = base.numeric_variables().len();

        let base_initial_numeric_values = {
            let values = base.get_initial_numeric_state_values();
            values.to_vec()
        };

        let mut projected_var_to_original: Vec<usize> = Vec::new();
        let mut projected_num_var_to_original: Vec<usize> = Vec::new();
        let mut original_var_to_projected = vec![None; num_vars];
        let mut original_num_var_to_projected = vec![None; num_numeric_vars];
        let mut pattern_regular_projected_ids: Vec<usize> = Vec::new();
        let mut pattern_numeric_projected_ids: Vec<usize> = Vec::new();

        for &var_id in &pattern.regular {
            if var_id >= num_vars {
                return Err(ProjectedTaskBuildError::InvalidRegularVarId {
                    provided: var_id,
                    len: num_vars,
                });
            }
            push_unique_mapping(
                var_id,
                &mut projected_var_to_original,
                &mut original_var_to_projected,
            );
            if let Some(projected_id) = original_var_to_projected[var_id] {
                push_unique_projected_id(projected_id, &mut pattern_regular_projected_ids);
            }
        }

        for &numeric_var_id in &pattern.numeric {
            if numeric_var_id < num_numeric_vars {
                let numeric_type = base.numeric_variables()[numeric_var_id].get_type().clone();
                if numeric_type != NumericType::Regular {
                    return Err(ProjectedTaskBuildError::UnsupportedPatternNumericVarType {
                        numeric_var_id,
                        numeric_type,
                    });
                }
                push_projected_base_numeric_var(
                    numeric_var_id,
                    &mut projected_num_var_to_original,
                    &mut original_num_var_to_projected,
                );
                if let Some(projected_id) = original_num_var_to_projected[numeric_var_id] {
                    push_unique_projected_id(projected_id, &mut pattern_numeric_projected_ids);
                }
            } else {
                return Err(ProjectedTaskBuildError::InvalidNumericVarId {
                    provided: numeric_var_id,
                    len: num_numeric_vars,
                });
            }
        }

        let comparison_axiom_by_affected_var =
            build_comparison_axiom_lookup(base, base.variables().len());
        let propositional_axiom_by_affected_var =
            build_propositional_axiom_lookup(base, base.variables().len());
        for original_var_id in projected_var_to_original.clone() {
            let Some(comparison_axiom_id) = comparison_axiom_by_affected_var[original_var_id]
            else {
                continue;
            };
            include_restricted_comparison_operands(
                base,
                comparison_axiom_id,
                &mut projected_num_var_to_original,
                &mut original_num_var_to_projected,
            );
        }
        let goal_facts = collect_restricted_projected_goals(
            base,
            pattern,
            &comparison_axiom_by_affected_var,
            &propositional_axiom_by_affected_var,
            &mut projected_var_to_original,
            &mut original_var_to_projected,
            &mut projected_num_var_to_original,
            &mut original_num_var_to_projected,
        )?;
        let mut numeric_effect_sources_by_target = vec![Vec::new(); num_numeric_vars];
        for operator in base.get_operators() {
            for effect in operator.assignment_effects() {
                numeric_effect_sources_by_target[effect.affected_var_id()].push(effect.var_id());
            }
        }
        let mut closure_index = 0;
        while closure_index < projected_num_var_to_original.len() {
            let target_var_id = projected_num_var_to_original[closure_index];
            for &source_var_id in &numeric_effect_sources_by_target[target_var_id] {
                push_projected_base_numeric_var(
                    source_var_id,
                    &mut projected_num_var_to_original,
                    &mut original_num_var_to_projected,
                );
            }
            closure_index += 1;
        }

        let mut variable_names: Vec<String> = Vec::with_capacity(projected_var_to_original.len());
        let mut fact_names: Vec<Vec<String>> = Vec::with_capacity(projected_var_to_original.len());
        let mut variable_domain_sizes: Vec<usize> =
            Vec::with_capacity(projected_var_to_original.len());
        let mut variable_default_values: Vec<usize> =
            Vec::with_capacity(projected_var_to_original.len());
        for &original_var_id in &projected_var_to_original {
            let variable_name = base
                .get_variable_name(original_var_id)
                .expect("projected propositional variable ID must be valid")
                .to_string();
            let domain_size = base
                .get_variable_domain_size(original_var_id)
                .expect("projected propositional variable must have a domain");

            let var_fact_names = (0..domain_size)
                .map(|value| {
                    let original_fact = ExplicitFact::new(original_var_id, value);
                    let fact_name = base.get_fact_name(&original_fact);
                    if fact_name.is_empty() {
                        format!("{variable_name}={value}")
                    } else {
                        fact_name.to_string()
                    }
                })
                .collect();

            variable_domain_sizes.push(domain_size);
            variable_default_values.push(
                base.get_variable_default_axiom_value(original_var_id)
                    .expect("projected propositional variable must have an axiom default"),
            );
            variable_names.push(variable_name);
            fact_names.push(var_fact_names);
        }

        let initial_prop_values = base.get_initial_propositional_state_values();
        let projected_prop_values: Vec<usize> = projected_var_to_original
            .iter()
            .map(|&original| initial_prop_values[original])
            .collect();
        drop(initial_prop_values);

        let mut numeric_variables: Vec<NumericVariable> =
            Vec::with_capacity(projected_num_var_to_original.len());
        let mut projected_numeric_values: Vec<f64> =
            Vec::with_capacity(projected_num_var_to_original.len());
        for projected_index in 0..projected_num_var_to_original.len() {
            let source_original = projected_num_var_to_original[projected_index];
            numeric_variables.push(base.numeric_variables()[source_original].clone());
            projected_numeric_values.push(base_initial_numeric_values[source_original]);
        }

        let goals: Vec<ExplicitFact> = goal_facts
            .iter()
            .filter_map(|goal| project_fact(goal, &original_var_to_projected))
            .collect();

        let mut operators: Vec<Operator> = Vec::new();
        let mut operator_costs: Vec<f64> = Vec::new();
        let mut base_operator_ids: Vec<usize> = Vec::new();
        for (base_operator_id, operator) in base.get_operators().iter().enumerate() {
            let operator_cost = metric_operator_cost_from_initial_values(base, operator);
            if let Some(projected_operator) = project_restricted_operator(
                operator,
                &original_var_to_projected,
                &original_num_var_to_projected,
            ) {
                operators.push(projected_operator);
                operator_costs.push(operator_cost);
                base_operator_ids.push(base_operator_id);
            }
        }

        let axioms: Vec<PropositionalAxiom> = base
            .axioms()
            .iter()
            .filter_map(|axiom| project_propositional_axiom(axiom, &original_var_to_projected))
            .collect();

        let mut comparison_axioms: Vec<ComparisonAxiom> = Vec::new();
        for comparison_axiom_id in 0..base.comparison_axioms().len() {
            if let Some(projected_axiom) = project_restricted_comparison_axiom(
                base,
                comparison_axiom_id,
                &original_var_to_projected,
                &original_num_var_to_projected,
            ) {
                comparison_axioms.push(projected_axiom);
            }
        }

        let assignment_axioms: Vec<AssignmentAxiom> = base
            .assignment_axioms()
            .iter()
            .filter_map(|axiom| project_assignment_axiom(axiom, &original_num_var_to_projected))
            .collect();

        let variable_layers = normalize_projected_variable_layers(
            projected_var_to_original.len(),
            &numeric_variables,
            &comparison_axioms,
            &axioms,
        );
        let variables: Vec<ExplicitVariable> = variable_names
            .iter()
            .cloned()
            .zip(fact_names.iter().cloned())
            .zip(variable_domain_sizes.iter().copied())
            .zip(variable_default_values.iter().copied())
            .zip(variable_layers.iter().copied())
            .map(
                |((((name, fact_names), domain_size), axiom_default_value), axiom_layer)| {
                    ExplicitVariable::new(
                        domain_size,
                        name,
                        fact_names,
                        axiom_layer,
                        axiom_default_value,
                    )
                },
            )
            .collect();

        let operator_effect_facts: Vec<Vec<ExplicitFact>> = operators
            .iter()
            .map(|operator| {
                operator
                    .effects()
                    .iter()
                    .map(|effect| ExplicitFact::new(effect.var_id(), effect.value()))
                    .collect()
            })
            .collect();
        let axiom_effect_facts: Vec<ExplicitFact> = axioms
            .iter()
            .map(|axiom| ExplicitFact::new(axiom.var_id(), axiom.effect_value()))
            .collect();

        let metric_var_id = if base.metric().var_id().is_none() {
            None
        } else {
            original_num_var_to_projected
                .get(base.metric().var_id().unwrap())
                .and_then(|mapped| *mapped)
        };

        let compilation_task = NumericRootTask::new(
            1,
            Metric::new(base.metric().is_min(), metric_var_id),
            variables.clone(),
            numeric_variables.clone(),
            goals.clone(),
            vec![],
            projected_prop_values.clone(),
            projected_numeric_values.clone(),
            operators.clone(),
            axioms.clone(),
            comparison_axioms.clone(),
            assignment_axioms.clone(),
            ExplicitFact::new(0, 0),
        );
        let compiled_axiom_evaluator_data = CompiledAxiomEvaluatorData::new(&compilation_task);
        let compiled_axiom_evaluator_scratch = RefCell::new(CompiledAxiomEvaluatorScratch::new(
            &compiled_axiom_evaluator_data,
        ));

        let propositional_packer = projected_propositional_packer_from_variables(&variables);
        let mut initial_packed_propositional = vec![0u64; propositional_packer.num_bins()];
        for (var_id, value) in projected_prop_values.iter().enumerate() {
            propositional_packer.set(&mut initial_packed_propositional, var_id, *value as u64);
        }
        Ok(Self {
            base,
            variables,
            numeric_variables,
            assignment_axioms,
            comparison_axioms,
            axioms,
            metric: Metric::new(base.metric().is_min(), metric_var_id),
            operators,
            operator_costs,
            base_operator_ids,
            propositional_packer,
            initial_packed_propositional,
            compiled_axiom_evaluator_data,
            compiled_axiom_evaluator_scratch,
            operator_effect_facts,
            goals,
            axiom_effect_facts,
            state: Rc::new(RefCell::new(projected_prop_values)),
            numeric_state: Rc::new(RefCell::new(projected_numeric_values)),
            projected_var_to_original,
            projected_num_var_to_original,
            original_var_to_projected,
            original_num_var_to_projected,
            pattern_regular_projected_ids,
            pattern_numeric_projected_ids,
            variable_names,
            fact_names,
        })
    }

    pub fn to_numeric_root_task(&self) -> NumericRootTask {
        NumericRootTask::new(
            1,
            self.metric.clone(),
            self.variables.clone(),
            self.numeric_variables.clone(),
            self.goals.clone(),
            vec![],
            self.state.borrow().clone(),
            self.numeric_state.borrow().clone(),
            self.operators.clone(),
            self.axioms.clone(),
            self.comparison_axioms.clone(),
            self.assignment_axioms.clone(),
            ExplicitFact::new(0, 0),
        )
    }

    pub fn project_state_values(
        &self,
        propositional_values: &[usize],
        numeric_values: &[f64],
    ) -> Result<(Vec<usize>, Vec<f64>), String> {
        let mut projected_prop_values = Vec::with_capacity(self.projected_var_to_original.len());
        let mut projected_numeric_values =
            Vec::with_capacity(self.projected_num_var_to_original.len());
        self.project_state_values_into(
            propositional_values,
            numeric_values,
            &mut projected_prop_values,
            &mut projected_numeric_values,
        )?;
        Ok((projected_prop_values, projected_numeric_values))
    }

    pub fn project_state_values_from_source_numeric_into(
        &self,
        propositional_values: &[usize],
        source_numeric_values: &[f64],
        projected_prop_values: &mut Vec<usize>,
        projected_numeric_values: &mut Vec<f64>,
    ) -> Result<(), String> {
        if propositional_values.len() < self.base.variables().len() {
            return Err(format!(
                "propositional state too short: {} < {}",
                propositional_values.len(),
                self.base.variables().len()
            ));
        }
        if source_numeric_values.len() < self.base.numeric_variables().len() {
            return Err(format!(
                "source numeric state too short: {} < {}",
                source_numeric_values.len(),
                self.base.numeric_variables().len()
            ));
        }

        projected_prop_values.clear();
        projected_prop_values.extend(
            self.projected_var_to_original
                .iter()
                .map(|&original_var_id| propositional_values[original_var_id]),
        );

        projected_numeric_values.clear();
        projected_numeric_values.reserve(self.projected_num_var_to_original.len());
        for &original_numeric_var in &self.projected_num_var_to_original {
            projected_numeric_values.push(source_numeric_values[original_numeric_var]);
        }

        Ok(())
    }

    pub fn project_pattern_state_values_from_source_numeric_into(
        &self,
        propositional_values: &[usize],
        source_numeric_values: &[f64],
        pattern_prop_values: &mut Vec<usize>,
        pattern_numeric_values: &mut Vec<f64>,
    ) -> Result<(), String> {
        if propositional_values.len() < self.base.variables().len() {
            return Err(format!(
                "propositional state too short: {} < {}",
                propositional_values.len(),
                self.base.variables().len()
            ));
        }
        if source_numeric_values.len() < self.base.numeric_variables().len() {
            return Err(format!(
                "source numeric state too short: {} < {}",
                source_numeric_values.len(),
                self.base.numeric_variables().len()
            ));
        }

        pattern_prop_values.clear();
        pattern_prop_values.reserve(self.pattern_regular_projected_ids.len());
        for &projected_var_id in &self.pattern_regular_projected_ids {
            let original_var_id = self.projected_var_to_original[projected_var_id];
            pattern_prop_values.push(propositional_values[original_var_id]);
        }

        pattern_numeric_values.clear();
        pattern_numeric_values.reserve(self.pattern_numeric_projected_ids.len());
        for &projected_numeric_id in &self.pattern_numeric_projected_ids {
            pattern_numeric_values.push(self.projected_numeric_value_from_source_numeric(
                projected_numeric_id,
                source_numeric_values,
            )?);
        }

        Ok(())
    }

    pub fn pack_pattern_state_values_from_source_numeric_into(
        &self,
        propositional_values: &[usize],
        source_numeric_values: &[f64],
        packer: &IntDoublePacker,
        packed_values: &mut Vec<u64>,
    ) -> Result<(), String> {
        if propositional_values.len() < self.base.variables().len() {
            return Err(format!(
                "propositional state too short: {} < {}",
                propositional_values.len(),
                self.base.variables().len()
            ));
        }
        if source_numeric_values.len() < self.base.numeric_variables().len() {
            return Err(format!(
                "source numeric state too short: {} < {}",
                source_numeric_values.len(),
                self.base.numeric_variables().len()
            ));
        }

        packed_values.clear();
        packed_values.resize(packer.num_bins(), 0);

        for (compact_index, &projected_var_id) in
            self.pattern_regular_projected_ids.iter().enumerate()
        {
            let original_var_id = self.projected_var_to_original[projected_var_id];
            packer.set(
                packed_values,
                compact_index,
                propositional_values[original_var_id] as u64,
            );
        }

        let prop_len = self.pattern_regular_projected_ids.len();
        for (numeric_index, &projected_numeric_id) in
            self.pattern_numeric_projected_ids.iter().enumerate()
        {
            packer.set(
                packed_values,
                prop_len + numeric_index,
                packer.pack_double(self.projected_numeric_value_from_source_numeric(
                    projected_numeric_id,
                    source_numeric_values,
                )?),
            );
        }

        Ok(())
    }

    pub fn pack_pattern_numeric_state_values_from_source_numeric_into(
        &self,
        source_numeric_values: &[f64],
        packer: &IntDoublePacker,
        packed_values: &mut Vec<u64>,
    ) -> Result<(), String> {
        if source_numeric_values.len() < self.base.numeric_variables().len() {
            return Err(format!(
                "source numeric state too short: {} < {}",
                source_numeric_values.len(),
                self.base.numeric_variables().len()
            ));
        }

        packed_values.clear();
        packed_values.resize(packer.num_bins(), 0);

        for (numeric_index, &projected_numeric_id) in
            self.pattern_numeric_projected_ids.iter().enumerate()
        {
            packer.set(
                packed_values,
                numeric_index + 1,
                packer.pack_double(self.projected_numeric_value_from_source_numeric(
                    projected_numeric_id,
                    source_numeric_values,
                )?),
            );
        }

        Ok(())
    }

    pub fn fill_pattern_numeric_bins_from_source_numeric_into(
        &self,
        source_numeric_values: &[f64],
        bins: &mut Vec<u64>,
    ) -> Result<(), String> {
        if source_numeric_values.len() < self.base.numeric_variables().len() {
            return Err(format!(
                "source numeric state too short: {} < {}",
                source_numeric_values.len(),
                self.base.numeric_variables().len()
            ));
        }

        bins.clear();
        bins.resize(1 + self.pattern_numeric_projected_ids.len(), 0);
        for (numeric_index, &projected_numeric_id) in
            self.pattern_numeric_projected_ids.iter().enumerate()
        {
            bins[numeric_index + 1] =
                float_tolerance::canonical_bits(self.projected_numeric_value_from_source_numeric(
                    projected_numeric_id,
                    source_numeric_values,
                )?);
        }

        Ok(())
    }

    pub fn compact_pattern_prop_hash_from_state_values(
        &self,
        propositional_values: &[usize],
        multipliers: &[usize],
    ) -> Result<usize, String> {
        if self.pattern_regular_projected_ids.len() != multipliers.len() {
            return Err("pattern regular ids and multipliers length mismatch".to_string());
        }
        if propositional_values.len() < self.base.variables().len() {
            return Err(format!(
                "propositional state too short: {} < {}",
                propositional_values.len(),
                self.base.variables().len()
            ));
        }

        let mut hash = 0usize;
        for (&projected_var_id, &multiplier) in self
            .pattern_regular_projected_ids
            .iter()
            .zip(multipliers.iter())
        {
            let original_var_id = self.projected_var_to_original[projected_var_id];
            let value = propositional_values[original_var_id];
            hash = hash.saturating_add(value.saturating_mul(multiplier));
        }
        Ok(hash)
    }

    pub fn project_state_values_into(
        &self,
        propositional_values: &[usize],
        numeric_values: &[f64],
        projected_prop_values: &mut Vec<usize>,
        projected_numeric_values: &mut Vec<f64>,
    ) -> Result<(), String> {
        if propositional_values.len() < self.base.variables().len() {
            return Err(format!(
                "propositional state too short: {} < {}",
                propositional_values.len(),
                self.base.variables().len()
            ));
        }
        if numeric_values.len() < self.base.numeric_variables().len() {
            return Err(format!(
                "numeric state too short: {} < {}",
                numeric_values.len(),
                self.base.numeric_variables().len()
            ));
        }

        self.project_state_values_from_source_numeric_into(
            propositional_values,
            numeric_values,
            projected_prop_values,
            projected_numeric_values,
        )
    }

    pub fn evaluated_initial_state_values(
        &self,
    ) -> Result<(Vec<usize>, Vec<f64>), ProjectedTaskBuildError> {
        let mut propositional = self.state.borrow().clone();
        let mut numeric = self.numeric_state.borrow().clone();
        self.evaluate_axiom_closure(&mut propositional, &mut numeric)?;
        Ok((propositional, numeric))
    }

    pub fn pack_propositional_values(
        &self,
        propositional_values: &[usize],
    ) -> Result<Vec<u64>, String> {
        if propositional_values.len() != self.variables.len() {
            return Err(format!(
                "expected {} propositional values, got {}",
                self.variables.len(),
                propositional_values.len()
            ));
        }
        let mut packed = vec![0u64; self.propositional_packer.num_bins()];
        for (var_id, value) in propositional_values.iter().enumerate() {
            self.propositional_packer
                .set(&mut packed, var_id, *value as u64);
        }
        Ok(packed)
    }

    pub fn propositional_packer(&self) -> &IntDoublePacker {
        &self.propositional_packer
    }

    pub fn evaluated_initial_state(&self) -> Result<EvaluatedState, ProjectedTaskBuildError> {
        let mut propositional = self.state.borrow().clone();
        let mut numeric = self.numeric_state.borrow().clone();
        let mut packed = self.initial_packed_propositional.clone();
        self.evaluate_axiom_closure_with_buffer(&mut propositional, &mut numeric, &mut packed)?;
        Ok((propositional, numeric, packed))
    }

    pub fn is_goal_state_values(&self, propositional_values: &[usize]) -> bool {
        self.goals
            .iter()
            .all(|goal| propositional_values.get(goal.var()).copied() == Some(goal.value()))
    }

    pub fn min_operator_cost(&self) -> f64 {
        let min_operator_cost = self
            .operator_costs
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if min_operator_cost.is_finite() {
            min_operator_cost.max(0.0)
        } else {
            0.0
        }
    }

    pub fn base_operator_id(&self, projected_operator_id: usize) -> Option<usize> {
        self.base_operator_ids.get(projected_operator_id).copied()
    }

    pub fn base_operator_ids(&self) -> &[usize] {
        &self.base_operator_ids
    }

    pub fn pattern_numeric_projected_ids(&self) -> &[usize] {
        &self.pattern_numeric_projected_ids
    }

    pub fn pattern_regular_projected_ids(&self) -> &[usize] {
        &self.pattern_regular_projected_ids
    }

    pub fn state_dependent_numeric_projected_ids(&self) -> Vec<usize> {
        (0..self.projected_num_var_to_original.len())
            .filter(|&projected_index| {
                self.base
                    .numeric_variables()
                    .get(self.projected_num_var_to_original[projected_index])
                    .is_some_and(|numeric_var| numeric_var.get_type() != &NumericType::Constant)
            })
            .collect()
    }

    pub fn project_pattern_concrete_state_values_into(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
        pattern_prop_values: &mut Vec<usize>,
        pattern_numeric_values: &mut Vec<f64>,
        numeric_value_cache: &mut Vec<Option<f64>>,
    ) -> Result<(), String> {
        numeric_value_cache.clear();
        numeric_value_cache.resize(self.base.numeric_variables().len(), None);

        pattern_prop_values.clear();
        pattern_prop_values.reserve(self.pattern_regular_projected_ids.len());
        for &projected_var_id in &self.pattern_regular_projected_ids {
            let original_var_id = self.projected_var_to_original[projected_var_id];
            pattern_prop_values.push(
                registry
                    .get_propositional_var_value(state, original_var_id)
                    .map_err(|err| format!("{err:?}"))?,
            );
        }

        pattern_numeric_values.clear();
        pattern_numeric_values.reserve(self.pattern_numeric_projected_ids.len());
        for &projected_numeric_id in &self.pattern_numeric_projected_ids {
            pattern_numeric_values.push(self.projected_numeric_value_from_concrete_state(
                projected_numeric_id,
                state,
                registry,
                numeric_value_cache,
            )?);
        }

        Ok(())
    }

    pub fn pack_pattern_concrete_state_values_into(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
        packer: &IntDoublePacker,
        packed_values: &mut Vec<u64>,
        numeric_value_cache: &mut Vec<Option<f64>>,
    ) -> Result<(), String> {
        numeric_value_cache.clear();
        numeric_value_cache.resize(self.base.numeric_variables().len(), None);

        packed_values.clear();
        packed_values.resize(packer.num_bins(), 0);

        for (compact_index, &projected_var_id) in
            self.pattern_regular_projected_ids.iter().enumerate()
        {
            let original_var_id = self.projected_var_to_original[projected_var_id];
            packer.set(
                packed_values,
                compact_index,
                registry
                    .get_propositional_var_value(state, original_var_id)
                    .map_err(|err| format!("{err:?}"))? as u64,
            );
        }

        let prop_len = self.pattern_regular_projected_ids.len();
        for (numeric_index, &projected_numeric_id) in
            self.pattern_numeric_projected_ids.iter().enumerate()
        {
            packer.set(
                packed_values,
                prop_len + numeric_index,
                packer.pack_double(self.projected_numeric_value_from_concrete_state(
                    projected_numeric_id,
                    state,
                    registry,
                    numeric_value_cache,
                )?),
            );
        }

        Ok(())
    }

    pub fn pack_pattern_numeric_concrete_state_values_into(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
        packer: &IntDoublePacker,
        packed_values: &mut Vec<u64>,
        numeric_value_cache: &mut Vec<Option<f64>>,
    ) -> Result<(), String> {
        numeric_value_cache.clear();
        numeric_value_cache.resize(self.base.numeric_variables().len(), None);

        packed_values.clear();
        packed_values.resize(packer.num_bins(), 0);

        for (numeric_index, &projected_numeric_id) in
            self.pattern_numeric_projected_ids.iter().enumerate()
        {
            packer.set(
                packed_values,
                numeric_index + 1,
                packer.pack_double(self.projected_numeric_value_from_concrete_state(
                    projected_numeric_id,
                    state,
                    registry,
                    numeric_value_cache,
                )?),
            );
        }

        Ok(())
    }

    pub fn fill_pattern_numeric_concrete_state_bins_into(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
        bins: &mut Vec<u64>,
        numeric_value_cache: &mut Vec<Option<f64>>,
    ) -> Result<(), String> {
        numeric_value_cache.clear();
        numeric_value_cache.resize(self.base.numeric_variables().len(), None);

        bins.clear();
        bins.resize(1 + self.pattern_numeric_projected_ids.len(), 0);
        for (numeric_index, &projected_numeric_id) in
            self.pattern_numeric_projected_ids.iter().enumerate()
        {
            bins[numeric_index + 1] =
                float_tolerance::canonical_bits(self.projected_numeric_value_from_concrete_state(
                    projected_numeric_id,
                    state,
                    registry,
                    numeric_value_cache,
                )?);
        }

        Ok(())
    }

    pub fn compact_pattern_prop_hash_from_concrete_state(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
        multipliers: &[usize],
    ) -> Result<usize, String> {
        if self.pattern_regular_projected_ids.len() != multipliers.len() {
            return Err("pattern regular ids and multipliers length mismatch".to_string());
        }

        let mut hash = 0usize;
        for (&projected_var_id, &multiplier) in self
            .pattern_regular_projected_ids
            .iter()
            .zip(multipliers.iter())
        {
            let original_var_id = self.projected_var_to_original[projected_var_id];
            let value = registry
                .get_propositional_var_value(state, original_var_id)
                .map_err(|err| format!("{err:?}"))?;
            hash = hash.saturating_add(value.saturating_mul(multiplier));
        }
        Ok(hash)
    }

    pub fn project_concrete_state_values_into(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
        projected_prop_values: &mut Vec<usize>,
        projected_numeric_values: &mut Vec<f64>,
        numeric_value_cache: &mut Vec<Option<f64>>,
    ) -> Result<(), String> {
        numeric_value_cache.clear();
        numeric_value_cache.resize(self.base.numeric_variables().len(), None);

        projected_prop_values.clear();
        projected_prop_values.reserve(self.projected_var_to_original.len());
        for &original_var_id in &self.projected_var_to_original {
            projected_prop_values.push(
                registry
                    .get_propositional_var_value(state, original_var_id)
                    .map_err(|err| format!("{err:?}"))?,
            );
        }

        projected_numeric_values.clear();
        projected_numeric_values.reserve(self.projected_num_var_to_original.len());
        for projected_index in 0..self.projected_num_var_to_original.len() {
            projected_numeric_values.push(self.projected_numeric_value_from_concrete_state(
                projected_index,
                state,
                registry,
                numeric_value_cache,
            )?);
        }

        Ok(())
    }
    fn projected_numeric_value_from_source_numeric(
        &self,
        projected_index: usize,
        source_numeric_values: &[f64],
    ) -> Result<f64, String> {
        let original_numeric_var = self
            .projected_num_var_to_original
            .get(projected_index)
            .copied()
            .ok_or_else(|| {
                format!("projected numeric variable index out of bounds: {projected_index}")
            })?;
        source_numeric_values
            .get(original_numeric_var)
            .copied()
            .ok_or_else(|| {
                format!(
                    "source numeric state too short for numeric variable {original_numeric_var}"
                )
            })
    }

    fn projected_numeric_value_from_concrete_state(
        &self,
        projected_index: usize,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
        numeric_value_cache: &mut Vec<Option<f64>>,
    ) -> Result<f64, String> {
        let original_numeric_var = self
            .projected_num_var_to_original
            .get(projected_index)
            .copied()
            .ok_or_else(|| {
                format!("projected numeric variable index out of bounds: {projected_index}")
            })?;
        if let Some(value) = numeric_value_cache
            .get(original_numeric_var)
            .and_then(|value| *value)
        {
            return Ok(value);
        }

        if self.base.numeric_variables()[original_numeric_var].get_type() == &NumericType::Derived {
            return Err(format!(
                "restricted task invariant violated: projected numeric variable {original_numeric_var} is derived"
            ));
        }
        let value = registry
            .get_numeric_var_value_unevaluated(state, original_numeric_var)
            .map_err(|err| format!("{err:?}"))?;
        numeric_value_cache[original_numeric_var] = Some(value);
        Ok(value)
    }

    fn evaluate_axiom_closure(
        &self,
        propositional: &mut [usize],
        numeric: &mut [f64],
    ) -> Result<(), ProjectedTaskBuildError> {
        let mut buffer = self.initial_packed_propositional.clone();

        for (var_id, value) in propositional.iter().enumerate() {
            self.propositional_packer
                .set(&mut buffer, var_id, *value as u64);
        }

        self.evaluate_axiom_closure_with_buffer(propositional, numeric, &mut buffer)
    }

    fn evaluate_axiom_closure_with_buffer(
        &self,
        propositional: &mut [usize],
        numeric: &mut [f64],
        buffer: &mut [u64],
    ) -> Result<(), ProjectedTaskBuildError> {
        let axiom_evaluator = CompiledAxiomEvaluator::new(
            self,
            &self.propositional_packer,
            &self.compiled_axiom_evaluator_data,
        );
        let mut scratch = self.compiled_axiom_evaluator_scratch.borrow_mut();

        axiom_evaluator
            .evaluate_arithmetic_axioms(numeric)
            .map_err(
                |err| ProjectedTaskBuildError::InitialStateEvaluationFailed {
                    reason: format!("arithmetic axioms: {err:?}"),
                },
            )?;
        axiom_evaluator
            .evaluate(buffer, numeric, &mut scratch)
            .map_err(
                |err| ProjectedTaskBuildError::InitialStateEvaluationFailed {
                    reason: format!("propositional axioms: {err:?}"),
                },
            )?;

        for (var_id, slot) in propositional.iter_mut().enumerate() {
            *slot = self.propositional_packer.get(buffer, var_id) as usize;
        }

        Ok(())
    }
}

#[allow(unused)]
fn projected_propositional_packer(task: &dyn AbstractNumericTask) -> IntDoublePacker {
    projected_propositional_packer_from_variables(task.variables())
}

fn projected_propositional_packer_from_variables(
    variables: &[ExplicitVariable],
) -> IntDoublePacker {
    let ranges: Vec<u64> = variables
        .iter()
        .map(|variable| variable.domain_size() as u64)
        .collect();
    IntDoublePacker::new(&ranges)
}

impl AbstractNumericTask for ProjectedTask<'_> {
    fn variables(&self) -> &Vec<ExplicitVariable> {
        &self.variables
    }

    fn numeric_variables(&self) -> &Vec<NumericVariable> {
        &self.numeric_variables
    }

    fn assignment_axioms(&self) -> &Vec<AssignmentAxiom> {
        &self.assignment_axioms
    }

    fn comparison_axioms(&self) -> &Vec<ComparisonAxiom> {
        &self.comparison_axioms
    }

    fn axioms(&self) -> &Vec<PropositionalAxiom> {
        &self.axioms
    }

    fn metric(&self) -> &Metric {
        &self.metric
    }

    fn get_num_variables(&self) -> usize {
        self.variables.len()
    }

    fn get_variable_name(&self, index: usize) -> Result<&str, &str> {
        self.variable_names
            .get(index)
            .map(|name| name.as_str())
            .ok_or("Index out of bounds")
    }

    fn get_variable_domain_size(&self, index: usize) -> Result<usize, &str> {
        self.variables
            .get(index)
            .map(|var| var.domain_size())
            .ok_or("Index out of bounds")
    }

    fn get_variable_axiom_layer(&self, index: usize) -> Result<Option<usize>, &str> {
        self.variables
            .get(index)
            .map(ExplicitVariable::axiom_layer)
            .ok_or("Index out of bounds")
    }

    fn get_variable_default_axiom_value(&self, index: usize) -> Result<usize, &str> {
        let original_index = self
            .projected_var_to_original
            .get(index)
            .copied()
            .ok_or("Index out of bounds")?;
        self.base.get_variable_default_axiom_value(original_index)
    }

    fn get_fact_name(&self, fact: &ExplicitFact) -> &str {
        let Some(var_fact_names) = self.fact_names.get(fact.var()) else {
            return "";
        };
        var_fact_names.get(fact.value()).map_or("", String::as_str)
    }

    fn are_facts_mutex(&self, fact1: &ExplicitFact, fact2: &ExplicitFact) -> bool {
        let original_fact1 = restore_fact(fact1, &self.projected_var_to_original)
            .expect("projected fact must use a valid variable ID");
        let original_fact2 = restore_fact(fact2, &self.projected_var_to_original)
            .expect("projected fact must use a valid variable ID");
        self.base.are_facts_mutex(&original_fact1, &original_fact2)
    }

    fn get_operators(&self) -> &Vec<Operator> {
        &self.operators
    }

    fn get_operator_cost(&self, index: usize, is_axiom: bool) -> u64 {
        if is_axiom {
            0
        } else {
            self.operators[index].cost()
        }
    }

    fn get_operator_name(&self, index: usize, is_axiom: bool) -> &str {
        if is_axiom {
            "<axiom>"
        } else {
            self.operators[index].name()
        }
    }

    fn get_num_operators(&self) -> usize {
        self.operators.len()
    }

    fn get_num_operator_preconditions(&self, index: usize, is_axiom: bool) -> usize {
        if is_axiom {
            self.axioms[index].conditions().len()
        } else {
            self.operators[index].preconditions().len()
        }
    }

    fn get_operator_precondition(
        &self,
        index: usize,
        precond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact {
        if is_axiom {
            &self.axioms[index].conditions()[precond_index]
        } else {
            &self.operators[index].preconditions()[precond_index]
        }
    }

    fn get_num_operator_effects(&self, index: usize, is_axiom: bool) -> usize {
        if is_axiom {
            let _ = &self.axioms[index];
            1
        } else {
            self.operators[index].effects().len()
        }
    }

    fn get_num_operator_effect_conditions(
        &self,
        index: usize,
        eff_index: usize,
        is_axiom: bool,
    ) -> usize {
        if is_axiom {
            0
        } else {
            self.operators[index].effects()[eff_index]
                .conditions()
                .len()
        }
    }

    fn get_operator_effect_condition(
        &self,
        index: usize,
        eff_index: usize,
        cond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact {
        assert!(
            !is_axiom,
            "axioms do not expose conditional effects separately"
        );
        &self.operators[index].effects()[eff_index].conditions()[cond_index]
    }

    fn get_operator_effect(&self, index: usize, eff_index: usize, is_axiom: bool) -> &ExplicitFact {
        if is_axiom {
            assert_eq!(eff_index, 0, "axioms expose exactly one effect");
            &self.axiom_effect_facts[index]
        } else {
            &self.operator_effect_facts[index][eff_index]
        }
    }

    fn convert_operator_index(&self, _index: usize, _ancestor_task: &dyn AbstractNumericTask) {}

    fn get_num_axioms(&self) -> usize {
        self.axioms.len()
    }

    fn get_num_goals(&self) -> usize {
        self.goals.len()
    }

    fn get_goal_fact(&self, index: usize) -> &ExplicitFact {
        &self.goals[index]
    }

    fn get_initial_propositional_state_values(&self) -> Ref<'_, Vec<usize>> {
        self.state.borrow()
    }

    fn get_initial_numeric_state_values(&self) -> Ref<'_, Vec<f64>> {
        self.numeric_state.borrow()
    }

    fn get_initial_propositional_state_values_mut(&self) -> RefMut<'_, Vec<usize>> {
        self.state.borrow_mut()
    }

    fn get_initial_numeric_state_values_mut(&self) -> RefMut<'_, Vec<f64>> {
        self.numeric_state.borrow_mut()
    }

    fn set_initial_numeric_state_values(&self, values: Vec<f64>) {
        *self.numeric_state.borrow_mut() = values;
    }

    fn set_initial_propositional_state_values(&self, values: Vec<usize>) {
        *self.state.borrow_mut() = values;
    }

    fn convert_ancestor_state_values(
        &self,
        ancestor_state_values: &[usize],
        _ancestor_task: &dyn AbstractNumericTask,
    ) -> Vec<usize> {
        if ancestor_state_values.len() == self.variables.len() {
            return ancestor_state_values.to_vec();
        }
        self.projected_var_to_original
            .iter()
            .map(|&original| ancestor_state_values[original])
            .collect()
    }

    fn get_num_cmp_axioms(&self) -> usize {
        self.comparison_axioms.len()
    }

    fn abstract_state_values(
        &self,
        propositional_values: &[usize],
        numeric_values: &[f64],
    ) -> Result<(Vec<usize>, Vec<f64>), String> {
        ProjectedTask::project_state_values(self, propositional_values, numeric_values)
    }

    fn evaluated_initial_abstract_state_values(&self) -> Result<(Vec<usize>, Vec<f64>), String> {
        ProjectedTask::evaluated_initial_state_values(self).map_err(|err| err.to_string())
    }

    fn abstract_operator_cost(&self, operator_id: usize) -> f64 {
        self.operator_costs[operator_id]
    }
}

fn push_unique_mapping(
    original_id: usize,
    projected_to_original: &mut Vec<usize>,
    original_to_projected: &mut [Option<usize>],
) {
    if original_to_projected[original_id].is_none() {
        original_to_projected[original_id] = Some(projected_to_original.len());
        projected_to_original.push(original_id);
    }
}

fn push_projected_base_numeric_var(
    original_id: usize,
    projected_to_original: &mut Vec<usize>,
    original_to_projected: &mut [Option<usize>],
) {
    if original_to_projected[original_id].is_none() {
        original_to_projected[original_id] = Some(projected_to_original.len());
        projected_to_original.push(original_id);
    }
}

fn push_unique_projected_id(projected_id: usize, ids: &mut Vec<usize>) {
    if !ids.contains(&projected_id) {
        ids.push(projected_id);
    }
}

fn build_comparison_axiom_lookup(
    task: &dyn AbstractNumericTask,
    num_vars: usize,
) -> Vec<Option<usize>> {
    let mut lookup = vec![None; num_vars];
    for (comparison_axiom_id, comparison_axiom) in task.comparison_axioms().iter().enumerate() {
        let affected = comparison_axiom.get_affected_var_id();
        if affected < lookup.len() {
            lookup[affected] = Some(comparison_axiom_id);
        }
    }
    lookup
}

fn build_propositional_axiom_lookup(
    task: &dyn AbstractNumericTask,
    num_vars: usize,
) -> Vec<Option<usize>> {
    let mut lookup = vec![None; num_vars];
    for (axiom_id, axiom) in task.axioms().iter().enumerate() {
        let affected = axiom.var_id();
        if affected < lookup.len() {
            lookup[affected] = Some(axiom_id);
        }
    }
    lookup
}

fn include_restricted_comparison_operands(
    task: &dyn AbstractNumericTask,
    comparison_axiom_id: usize,
    projected_num_var_to_original: &mut Vec<usize>,
    original_num_var_to_projected: &mut [Option<usize>],
) {
    let comparison_axiom = &task.comparison_axioms()[comparison_axiom_id];
    for numeric_var_id in [
        comparison_axiom.get_left_var_id(),
        comparison_axiom.get_right_var_id(),
    ] {
        assert_ne!(
            task.numeric_variables()[numeric_var_id].get_type(),
            &NumericType::Derived,
            "restricted task validation excludes derived comparison operands"
        );
        push_projected_base_numeric_var(
            numeric_var_id,
            projected_num_var_to_original,
            original_num_var_to_projected,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_restricted_projected_goals(
    task: &dyn AbstractNumericTask,
    pattern: &Pattern,
    comparison_axiom_by_affected_var: &[Option<usize>],
    propositional_axiom_by_affected_var: &[Option<usize>],
    projected_var_to_original: &mut Vec<usize>,
    original_var_to_projected: &mut [Option<usize>],
    projected_num_var_to_original: &mut Vec<usize>,
    original_num_var_to_projected: &mut [Option<usize>],
) -> Result<Vec<ExplicitFact>, ProjectedTaskBuildError> {
    let pattern_regular: BTreeSet<usize> = pattern.regular.iter().copied().collect();
    let pattern_numeric: BTreeSet<usize> = pattern.numeric.iter().copied().collect();
    let mut goals = Vec::new();
    let mut visited_vars = HashSet::new();

    for goal_index in 0..task.get_num_goals() {
        collect_restricted_projected_goal_fact(
            task,
            task.get_goal_fact(goal_index),
            &pattern_regular,
            &pattern_numeric,
            comparison_axiom_by_affected_var,
            propositional_axiom_by_affected_var,
            projected_var_to_original,
            original_var_to_projected,
            projected_num_var_to_original,
            original_num_var_to_projected,
            &mut goals,
            &mut visited_vars,
        )?;
    }

    goals.sort();
    goals.dedup();
    Ok(goals)
}

#[allow(clippy::too_many_arguments)]
fn collect_restricted_projected_goal_fact(
    task: &dyn AbstractNumericTask,
    fact: &ExplicitFact,
    pattern_regular: &BTreeSet<usize>,
    pattern_numeric: &BTreeSet<usize>,
    comparison_axiom_by_affected_var: &[Option<usize>],
    propositional_axiom_by_affected_var: &[Option<usize>],
    projected_var_to_original: &mut Vec<usize>,
    original_var_to_projected: &mut [Option<usize>],
    projected_num_var_to_original: &mut Vec<usize>,
    original_num_var_to_projected: &mut [Option<usize>],
    goals: &mut Vec<ExplicitFact>,
    visited_vars: &mut HashSet<usize>,
) -> Result<(), ProjectedTaskBuildError> {
    if !visited_vars.insert(fact.var()) {
        return Ok(());
    }

    if let Some(comparison_axiom_id) = comparison_axiom_by_affected_var
        .get(fact.var())
        .copied()
        .flatten()
    {
        let comparison_axiom = &task.comparison_axioms()[comparison_axiom_id];
        let operands = [
            comparison_axiom.get_left_var_id(),
            comparison_axiom.get_right_var_id(),
        ];
        let selected = pattern_regular.contains(&fact.var())
            || operands.iter().any(|id| pattern_numeric.contains(id));
        if selected {
            push_unique_mapping(
                fact.var(),
                projected_var_to_original,
                original_var_to_projected,
            );
            include_restricted_comparison_operands(
                task,
                comparison_axiom_id,
                projected_num_var_to_original,
                original_num_var_to_projected,
            );
            goals.push(fact.clone());
        }
        return Ok(());
    }

    if let Some(axiom_id) = propositional_axiom_by_affected_var
        .get(fact.var())
        .copied()
        .flatten()
    {
        if pattern_regular.contains(&fact.var()) {
            push_unique_mapping(
                fact.var(),
                projected_var_to_original,
                original_var_to_projected,
            );
            goals.push(fact.clone());
        }
        for condition in task.axioms()[axiom_id].conditions() {
            collect_restricted_projected_goal_fact(
                task,
                condition,
                pattern_regular,
                pattern_numeric,
                comparison_axiom_by_affected_var,
                propositional_axiom_by_affected_var,
                projected_var_to_original,
                original_var_to_projected,
                projected_num_var_to_original,
                original_num_var_to_projected,
                goals,
                visited_vars,
            )?;
        }
        return Ok(());
    }

    if pattern_regular.contains(&fact.var()) {
        push_unique_mapping(
            fact.var(),
            projected_var_to_original,
            original_var_to_projected,
        );
        goals.push(fact.clone());
    }
    Ok(())
}

fn project_fact(fact: &ExplicitFact, var_map: &[Option<usize>]) -> Option<ExplicitFact> {
    var_map
        .get(fact.var())
        .and_then(|mapped| *mapped)
        .map(|mapped| ExplicitFact::new(mapped, fact.value()))
}

fn restore_fact(fact: &ExplicitFact, projected_to_original: &[usize]) -> Option<ExplicitFact> {
    projected_to_original
        .get(fact.var())
        .map(|&original| ExplicitFact::new(original, fact.value()))
}

fn project_effect(effect: &Effect, var_map: &[Option<usize>]) -> Option<Effect> {
    let mapped_var = var_map.get(effect.var_id()).and_then(|mapped| *mapped)?;
    let conditions: Vec<ExplicitFact> = effect
        .conditions()
        .iter()
        .filter_map(|fact| project_fact(fact, var_map))
        .collect();
    Some(Effect::new(
        conditions,
        mapped_var,
        effect.precondition_value(),
        effect.value(),
    ))
}

fn project_assignment_effect(
    effect: &AssignmentEffect,
    var_map: &[Option<usize>],
    num_var_map: &[Option<usize>],
) -> Option<AssignmentEffect> {
    let affected = num_var_map
        .get(effect.affected_var_id())
        .and_then(|mapped| *mapped)?;
    let source = num_var_map
        .get(effect.var_id())
        .and_then(|mapped| *mapped)?;
    let conditions: Vec<ExplicitFact> = effect
        .conditions()
        .iter()
        .filter_map(|fact| project_fact(fact, var_map))
        .collect();
    Some(AssignmentEffect::new(
        affected,
        effect.operation().clone(),
        source,
        effect.is_conditional(),
        conditions,
    ))
}

fn project_restricted_operator(
    operator: &Operator,
    var_map: &[Option<usize>],
    num_var_map: &[Option<usize>],
) -> Option<Operator> {
    let effects: Vec<Effect> = operator
        .effects()
        .iter()
        .filter_map(|effect| project_effect(effect, var_map))
        .collect();
    let assignment_effects: Vec<AssignmentEffect> = operator
        .assignment_effects()
        .iter()
        .filter_map(|effect| project_assignment_effect(effect, var_map, num_var_map))
        .collect();
    if effects.is_empty() && assignment_effects.is_empty() {
        return None;
    }

    let preconditions = operator
        .preconditions()
        .iter()
        .filter_map(|fact| project_fact(fact, var_map))
        .collect();
    Some(Operator::new(
        operator.name().to_string(),
        preconditions,
        effects,
        assignment_effects,
        operator.cost(),
    ))
}

fn project_propositional_axiom(
    axiom: &PropositionalAxiom,
    var_map: &[Option<usize>],
) -> Option<PropositionalAxiom> {
    let mapped_var = var_map.get(axiom.var_id()).and_then(|mapped| *mapped)?;
    let conditions = axiom
        .conditions()
        .iter()
        .filter_map(|fact| project_fact(fact, var_map))
        .collect();
    Some(PropositionalAxiom::new(
        conditions,
        mapped_var,
        axiom.precondition_value(),
        axiom.effect_value(),
    ))
}

fn project_restricted_comparison_axiom(
    task: &dyn AbstractNumericTask,
    comparison_axiom_id: usize,
    var_map: &[Option<usize>],
    num_var_map: &[Option<usize>],
) -> Option<ComparisonAxiom> {
    let axiom = &task.comparison_axioms()[comparison_axiom_id];
    let affected = var_map
        .get(axiom.get_affected_var_id())
        .copied()
        .flatten()?;
    let left = num_var_map
        .get(axiom.get_left_var_id())
        .copied()
        .flatten()?;
    let right = num_var_map
        .get(axiom.get_right_var_id())
        .copied()
        .flatten()?;
    Some(ComparisonAxiom::new(
        affected,
        left,
        right,
        axiom.get_operator().clone(),
    ))
}

fn project_assignment_axiom(
    axiom: &AssignmentAxiom,
    num_var_map: &[Option<usize>],
) -> Option<AssignmentAxiom> {
    let affected = num_var_map
        .get(axiom.get_affected_var_id())
        .and_then(|mapped| *mapped)?;
    let left = num_var_map
        .get(axiom.get_left_var_id())
        .and_then(|mapped| *mapped)?;
    let right = num_var_map
        .get(axiom.get_right_var_id())
        .and_then(|mapped| *mapped)?;
    Some(AssignmentAxiom::new(
        affected,
        axiom.get_operator().clone(),
        left,
        right,
    ))
}

fn normalize_projected_variable_layers(
    num_variables: usize,
    numeric_variables: &[NumericVariable],
    comparison_axioms: &[ComparisonAxiom],
    axioms: &[PropositionalAxiom],
) -> Vec<Option<usize>> {
    let last_arithmetic_layer = numeric_variables
        .iter()
        .map(NumericVariable::axiom_layer)
        .max()
        .map_or(-1, |x| x.map_or(-1, |y| y as i32));
    let comparison_layer = if comparison_axioms.is_empty() {
        None
    } else {
        Some((last_arithmetic_layer + 1) as usize)
    };
    let base_propositional_layer = if comparison_axioms.is_empty() {
        Some((last_arithmetic_layer + 1) as usize)
    } else {
        Some(comparison_layer.map_or(0, |x| x + 1))
    };

    let mut affects_comparison = vec![false; num_variables];
    for axiom in comparison_axioms {
        let var_id = axiom.get_affected_var_id();
        if var_id < num_variables {
            affects_comparison[var_id] = true;
        }
    }

    let mut axioms_by_var: Vec<Vec<&PropositionalAxiom>> = vec![Vec::new(); num_variables];
    for axiom in axioms {
        let affected_var = axiom.var_id();
        if affected_var < num_variables {
            axioms_by_var[affected_var].push(axiom);
        }
    }

    let mut layers = vec![None; num_variables];
    let mut visiting = vec![false; num_variables];

    fn compute_layer(
        var_id: usize,
        layers: &mut [Option<usize>],
        visiting: &mut [bool],
        affects_comparison: &[bool],
        axioms_by_var: &[Vec<&PropositionalAxiom>],
        comparison_layer: Option<usize>,
        base_propositional_layer: Option<usize>,
    ) -> Option<usize> {
        if layers[var_id].is_some() {
            return layers[var_id];
        }
        if affects_comparison[var_id] {
            layers[var_id] = comparison_layer;
            return comparison_layer;
        }
        if axioms_by_var[var_id].is_empty() {
            return None;
        }
        if visiting[var_id] {
            return base_propositional_layer;
        }

        visiting[var_id] = true;
        let mut layer = base_propositional_layer;
        for axiom in &axioms_by_var[var_id] {
            for condition in axiom.conditions() {
                let condition_var = condition.var();
                if condition_var >= layers.len() {
                    continue;
                }
                let dependency_layer = compute_layer(
                    condition_var,
                    layers,
                    visiting,
                    affects_comparison,
                    axioms_by_var,
                    comparison_layer,
                    base_propositional_layer,
                );
                if dependency_layer.is_some() {
                    layer = layer.max(dependency_layer.map(|x| x + 1));
                }
            }
        }
        visiting[var_id] = false;
        layers[var_id] = layer;
        layer
    }

    for var_id in 0..num_variables {
        if affects_comparison[var_id] || !axioms_by_var[var_id].is_empty() {
            compute_layer(
                var_id,
                &mut layers,
                &mut visiting,
                &affects_comparison,
                &axioms_by_var,
                comparison_layer,
                base_propositional_layer,
            );
        }
    }

    layers
}
