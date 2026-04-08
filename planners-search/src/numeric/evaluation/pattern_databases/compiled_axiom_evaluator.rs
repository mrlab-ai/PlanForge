use std::cmp::max;

use planners_sas::numeric::axioms::{AssignmentAxiom, ComparisonAxiom, PropositionalAxiom};
use planners_sas::numeric::numeric_task::AbstractNumericTask;
use planners_sas::numeric::utils::errors::{AxiomEvalError, InvalidIndex, WrongAxiomLayer};
use planners_sas::numeric::utils::int_packer::IntDoublePacker;

#[derive(Debug, Clone)]
struct AxiomRule {
    condition_count: i32,
    effect_var: i32,
    effect_value: u64,
}

impl AxiomRule {
    fn new(cond_count: usize, eff_var: usize, eff_val: usize) -> Self {
        Self {
            condition_count: cond_count as i32,
            effect_var: eff_var as i32,
            effect_value: eff_val as u64,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct AxiomLiteral {
    condition_of: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
struct NegationByFailureInfo {
    var_id: u32,
    literal_value: usize,
}

impl NegationByFailureInfo {
    fn new(var_id: u32, literal_value: usize) -> Self {
        Self {
            var_id,
            literal_value,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LiteralRef {
    var_id: usize,
    value: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct CompiledAxiomEvaluatorData {
    axiom_literals: Vec<Vec<AxiomLiteral>>,
    rules: Vec<AxiomRule>,
    comparison_axiom_layer: i32,
    first_propositional_axiom_layer: i32,
    last_propositional_axiom_layer: i32,
    last_arithmetic_axiom_layer: i32,
    nbf_info_by_layer: Vec<Vec<NegationByFailureInfo>>,
    initial_propositional_values: Vec<i32>,
    has_numeric_axioms: bool,
    has_propositional_axioms: bool,
}

impl CompiledAxiomEvaluatorData {
    pub(crate) fn new(numeric_task: &dyn AbstractNumericTask) -> Self {
        build_compiled_axiom_evaluator_data(numeric_task)
    }
}

#[derive(Debug, Default)]
pub(crate) struct CompiledAxiomEvaluatorScratch {
    queue: Vec<LiteralRef>,
    unsatisfied_conditions: Vec<i32>,
}

impl CompiledAxiomEvaluatorScratch {
    pub(crate) fn new(data: &CompiledAxiomEvaluatorData) -> Self {
        Self {
            queue: Vec::new(),
            unsatisfied_conditions: vec![0; data.rules.len()],
        }
    }
}

pub(crate) struct CompiledAxiomEvaluator<'a> {
    numeric_task: &'a dyn AbstractNumericTask,
    state_packer: &'a IntDoublePacker,
    data: &'a CompiledAxiomEvaluatorData,
}

impl<'a> CompiledAxiomEvaluator<'a> {
    pub(crate) fn new(
        numeric_task: &'a dyn AbstractNumericTask,
        state_packer: &'a IntDoublePacker,
        data: &'a CompiledAxiomEvaluatorData,
    ) -> Self {
        Self {
            numeric_task,
            state_packer,
            data,
        }
    }

    pub(crate) fn evaluate_arithmetic_axioms(
        &self,
        numeric_state: &mut Vec<f64>,
    ) -> Result<(), InvalidIndex> {
        for axiom in self.numeric_task.assignment_axioms() {
            axiom.update_values(numeric_state)?;
        }
        Ok(())
    }

    pub(crate) fn evaluate_comparison_axioms(
        &self,
        buffer: &mut [u64],
        numeric_state: &mut Vec<f64>,
    ) -> Result<bool, AxiomEvalError> {
        for axiom in self.numeric_task.comparison_axioms() {
            let result = axiom.update_values(numeric_state).map_err(|e| {
                AxiomEvalError::InvalidIndex(InvalidIndex {
                    length: numeric_state.len() as u32,
                    index: e.index,
                })
            })?;
            self.state_packer
                .set(buffer, axiom.get_affected_var_id(), !result as u64);
        }

        Ok(true)
    }

    pub(crate) fn evaluate_propositional_axioms(
        &self,
        buffer: &mut [u64],
        scratch: &mut CompiledAxiomEvaluatorScratch,
    ) -> Result<(), AxiomEvalError> {
        if !self.data.has_propositional_axioms {
            return Ok(());
        }

        scratch.queue.clear();

        if scratch.unsatisfied_conditions.len() != self.data.rules.len() {
            scratch
                .unsatisfied_conditions
                .resize(self.data.rules.len(), 0);
        }

        for i in 0..self.numeric_task.get_num_variables() {
            let axiom_layer = self.numeric_task.get_variable_axiom_layer(i).unwrap();
            if axiom_layer == -1 {
                scratch.queue.push(LiteralRef {
                    var_id: i as usize,
                    value: self.state_packer.get(buffer, i) as usize,
                });
            } else if axiom_layer <= self.data.last_arithmetic_axiom_layer {
                return Err(AxiomEvalError::WrongAxiomLayer(WrongAxiomLayer {
                    axiom_layer,
                    last_arithmetic_axiom_layer: self.data.last_arithmetic_axiom_layer,
                }));
            } else if axiom_layer == self.data.comparison_axiom_layer {
                scratch.queue.push(LiteralRef {
                    var_id: i as usize,
                    value: self.state_packer.get(buffer, i) as usize,
                });
            } else if axiom_layer <= self.data.last_propositional_axiom_layer {
                let default_value = self.data.initial_propositional_values[i as usize];
                self.state_packer.set(buffer, i, default_value as u64);
            } else {
                return Err(AxiomEvalError::WrongAxiomLayer(WrongAxiomLayer {
                    axiom_layer,
                    last_arithmetic_axiom_layer: self.data.last_arithmetic_axiom_layer,
                }));
            }
        }

        for (rule_index, rule) in self.data.rules.iter().enumerate() {
            scratch.unsatisfied_conditions[rule_index] = rule.condition_count;
            if rule.condition_count == 0 {
                let var_no = rule.effect_var;
                let val = rule.effect_value;
                if self.state_packer.get(buffer, var_no) != val {
                    self.state_packer.set(buffer, var_no, val);
                    scratch.queue.push(LiteralRef {
                        var_id: var_no as usize,
                        value: val as usize,
                    });
                }
            }
        }

        for layer_no in 0..self.data.nbf_info_by_layer.len() {
            while let Some(curr_literal) = scratch.queue.pop() {
                let dependent_rules =
                    &self.data.axiom_literals[curr_literal.var_id][curr_literal.value].condition_of;

                for &rule_index in dependent_rules {
                    let remaining = &mut scratch.unsatisfied_conditions[rule_index];
                    *remaining -= 1;

                    if *remaining == 0 {
                        let rule = &self.data.rules[rule_index];
                        let var_no = rule.effect_var;
                        let val = rule.effect_value;
                        if self.state_packer.get(buffer, var_no) != val {
                            self.state_packer.set(buffer, var_no, val);
                            scratch.queue.push(LiteralRef {
                                var_id: var_no as usize,
                                value: val as usize,
                            });
                        }
                    }
                }
            }

            if layer_no != self.data.nbf_info_by_layer.len() - 1 {
                let nbf_info = &self.data.nbf_info_by_layer[layer_no];
                for info in nbf_info {
                    let var_no = info.var_id as i32;
                    let default_value = self.data.initial_propositional_values[var_no as usize];
                    if self.state_packer.get(buffer, var_no) == default_value as u64 {
                        scratch.queue.push(LiteralRef {
                            var_id: var_no as usize,
                            value: info.literal_value,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn evaluate(
        &self,
        buffer: &mut [u64],
        numeric_state: &mut Vec<f64>,
        scratch: &mut CompiledAxiomEvaluatorScratch,
    ) -> Result<(), AxiomEvalError> {
        if !self.has_axioms() {
            return Ok(());
        }
        if self.has_numeric_axioms() {
            self.evaluate_comparison_axioms(buffer, numeric_state)?;
        }
        if self.has_propositional_axioms() {
            self.evaluate_propositional_axioms(buffer, scratch)?;
        }
        Ok(())
    }

    fn has_axioms(&self) -> bool {
        self.has_numeric_axioms() || self.has_propositional_axioms()
    }

    pub(crate) fn has_numeric_axioms(&self) -> bool {
        self.data.has_numeric_axioms
    }

    fn has_propositional_axioms(&self) -> bool {
        self.data.has_propositional_axioms
    }
}

fn build_compiled_axiom_evaluator_data(
    numeric_task: &dyn AbstractNumericTask,
) -> CompiledAxiomEvaluatorData {
    let mut axiom_literals = vec![];
    let mut nbf_info_by_layer = vec![];
    let mut rules = vec![];
    let mut comparison_axiom_layer = -1;
    let mut first_propositional_axiom_layer = -1;
    let mut last_propositional_axiom_layer = -1;
    let mut last_arithmetic_axiom_layer = -1;

    for numeric_var in numeric_task.numeric_variables().iter() {
        last_arithmetic_axiom_layer = max(last_arithmetic_axiom_layer, numeric_var.axiom_layer());
    }

    for i in 0..numeric_task.get_num_variables() {
        let axiom_layer = numeric_task.get_variable_axiom_layer(i).unwrap();
        if axiom_layer == -1 {
            continue;
        }
        last_propositional_axiom_layer = max(last_propositional_axiom_layer, axiom_layer);
        if axiom_layer < first_propositional_axiom_layer || first_propositional_axiom_layer == -1 {
            first_propositional_axiom_layer = axiom_layer;
        }
    }

    if first_propositional_axiom_layer >= 0 && numeric_task.get_num_cmp_axioms() > 0 {
        comparison_axiom_layer = first_propositional_axiom_layer;
        first_propositional_axiom_layer += 1;
        debug_assert_eq!(comparison_axiom_layer, last_arithmetic_axiom_layer + 1);
    }

    for var in numeric_task.variables().iter() {
        axiom_literals.push(vec![AxiomLiteral::default(); var.domain_size() as usize]);
    }

    for axiom in numeric_task.axioms().iter() {
        let cond_count = axiom.conditions().len();
        let eff_var = axiom.var_id() as usize;
        let eff_val = axiom.effect_value() as usize;
        rules.push(AxiomRule::new(cond_count, eff_var, eff_val));
    }

    for (index, axiom) in numeric_task.axioms().iter().enumerate() {
        for condition in axiom.conditions().iter() {
            axiom_literals[condition.var() as usize][condition.value() as usize]
                .condition_of
                .push(index);
        }
    }

    let mut last_layer = -1;
    for i in 0..numeric_task.get_num_variables() {
        last_layer = max(
            last_layer,
            numeric_task.get_variable_axiom_layer(i).unwrap(),
        );
    }
    nbf_info_by_layer.resize((last_layer + 1).max(0) as usize, vec![]);

    let initial_propositional_values = numeric_task
        .get_initial_propositional_state_values()
        .to_vec();
    for var_id in 0..numeric_task.get_num_variables() {
        let axiom_layer = numeric_task.get_variable_axiom_layer(var_id).unwrap();
        if axiom_layer != -1 && axiom_layer != last_layer {
            let nbf_value = initial_propositional_values[var_id as usize];
            let nbf_info = NegationByFailureInfo::new(var_id as u32, nbf_value as usize);
            nbf_info_by_layer[axiom_layer as usize].push(nbf_info);
        }
    }

    CompiledAxiomEvaluatorData {
        axiom_literals,
        rules,
        comparison_axiom_layer,
        first_propositional_axiom_layer,
        last_propositional_axiom_layer,
        last_arithmetic_axiom_layer,
        nbf_info_by_layer,
        initial_propositional_values,
        has_numeric_axioms: !numeric_task.assignment_axioms().is_empty()
            || !numeric_task.comparison_axioms().is_empty(),
        has_propositional_axioms: !numeric_task.axioms().is_empty(),
    }
}
