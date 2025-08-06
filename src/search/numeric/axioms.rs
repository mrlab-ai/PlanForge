use std::{cmp::max, collections::btree_map::Values};

use crate::{
    parser::numeric_parser,
    search::numeric::{
        self,
        numeric_task::{self, AbstractNumericTask, Fact, GlobalCondition},
        utils::{errors::InvalidIndex, int_packer::IntDoublePacker},
    },
};

#[derive(Debug)]
pub struct PropositionalAxiom {
    conditions: Vec<Fact>,
    var_id: u32,
    precondition_value: u32,
    effect_value: u32,
}

impl PropositionalAxiom {
    pub fn new(
        conditions: Vec<Fact>,
        var_id: u32,
        precondition_value: u32,
        effect_value: u32,
    ) -> Self {
        PropositionalAxiom {
            conditions,
            var_id,
            precondition_value,
            effect_value,
        }
    }

    pub fn var_id(&self) -> u32 {
        self.var_id
    }

    pub fn conditions(&self) -> &Vec<Fact> {
        &self.conditions
    }
}

#[derive(Debug)]
pub enum CalOperator {
    Sum,
    Difference,
    Product,
    Division,
}
#[derive(Debug)]
pub struct AssignmentAxiom {
    affected_var_id: u32,
    operator: CalOperator,
    left_hand_side: u32,
    right_hand_side: u32,
}

impl AssignmentAxiom {
    pub fn new(
        affected_var_id: u32,

        operator: CalOperator,
        left_hand_side: u32,
        right_hand_side: u32,
    ) -> Self {
        AssignmentAxiom {
            affected_var_id,
            operator,
            left_hand_side,
            right_hand_side,
        }
    }

    pub fn update_values(&self, numeric_state: &mut Vec<f64>) -> Result<f64, InvalidIndex> {
        let left = self.left_hand_side as usize;
        let right = self.right_hand_side as usize;
        if left >= numeric_state.len() || right >= numeric_state.len() {
            return Err(InvalidIndex {
                length: numeric_state.len() as u32,
                index: left as u32,
            });
        }
        let affected = self.affected_var_id as usize;
        if affected >= numeric_state.len() {
            return Err(InvalidIndex {
                length: numeric_state.len() as u32,
                index: affected as u32,
            });
        }
        let result = match self.operator {
            CalOperator::Sum => numeric_state[left] + numeric_state[right],
            CalOperator::Difference => numeric_state[left] - numeric_state[right],
            CalOperator::Product => numeric_state[left] * numeric_state[right],
            CalOperator::Division => {
                if numeric_state[right] == 0.0 {
                    return Err(InvalidIndex {
                        length: numeric_state.len() as u32,
                        index: right as u32,
                    });
                }
                numeric_state[left] / numeric_state[right]
            }
        };
        numeric_state[affected] = result;
        Ok(result)
    }

    pub fn get_left_var_id(&self) -> u32 {
        self.left_hand_side
    }

    pub fn get_right_var_id(&self) -> u32 {
        self.right_hand_side
    }

    pub fn get_affected_var_id(&self) -> u32 {
        self.affected_var_id
    }

    pub fn get_operator(&self) -> &CalOperator {
        &self.operator
    }
}

#[derive(Debug)]
pub enum ComparisonOperator {
    LessThan,
    LessThanOrEqual,
    Equal,
    GreaterThanOrEqual,
    GreaterThan,
    UnEqual,
}

impl ComparisonOperator {
    pub fn update_values(
        &self,
        numeric_values: &mut Vec<f64>,
        affected: i32,
        left: i32,
        right: i32,
    ) -> f64 {
        let (left, right) = (
            numeric_values[left as usize],
            numeric_values[right as usize],
        );
        let result = match self {
            ComparisonOperator::LessThan => left < right,
            ComparisonOperator::LessThanOrEqual => left <= right,
            ComparisonOperator::Equal => left == right,
            ComparisonOperator::GreaterThanOrEqual => left >= right,
            ComparisonOperator::GreaterThan => left > right,
            ComparisonOperator::UnEqual => left != right,
        };
        numeric_values[affected as usize] = if result { 1.0 } else { 0.0 };
        numeric_values[affected as usize]
    }
}

#[derive(Debug)]
pub struct ComparisonAxiom {
    affected_var_id: i32,
    left_hand_side: i32,
    right_hand_side: i32,
    operator: ComparisonOperator,
}

impl ComparisonAxiom {
    pub fn new(
        affected_var_id: i32,
        left_hand_side: i32,
        right_hand_side: i32,
        operator: ComparisonOperator,
    ) -> Self {
        ComparisonAxiom {
            affected_var_id,
            left_hand_side,
            right_hand_side,
            operator,
        }
    }

    pub fn update_values(&self, numeric_state: &mut Vec<f64>) -> Result<f64, InvalidIndex> {
        let left = self.left_hand_side as usize;
        let right = self.right_hand_side as usize;
        if left >= numeric_state.len() || right >= numeric_state.len() {
            return Err(InvalidIndex {
                length: numeric_state.len() as u32,
                index: left as u32,
            });
        }
        let affected = self.affected_var_id as usize;
        if affected >= numeric_state.len() {
            return Err(InvalidIndex {
                length: numeric_state.len() as u32,
                index: affected as u32,
            });
        }
        let result = ComparisonOperator::update_values(
            &ComparisonOperator::Equal, // Assuming Equal for simplicity, this should be parameterized
            numeric_state,
            self.affected_var_id,
            self.left_hand_side,
            self.right_hand_side,
        );
        Ok(result)
    }

    pub fn get_affected_var_id(&self) -> i32 {
        self.affected_var_id
    }
    pub fn get_left_var_id(&self) -> i32 {
        self.left_hand_side
    }
    pub fn get_right_var_id(&self) -> i32 {
        self.right_hand_side
    }
}
#[derive(Debug, Clone)]
struct AxiomRule {
    condition_count: i32,
    unsatisfied_conditions: i32,
    effect_var: i32,
    effect_value: u64,
    effect_literal: AxiomLiteral,
}

impl AxiomRule {
    pub fn new(
        cond_count: usize,
        eff_var: usize,
        eff_val: usize,
        eff_literal: &AxiomLiteral,
    ) -> Self {
        AxiomRule {
            condition_count: cond_count as i32,
            unsatisfied_conditions: cond_count as i32,
            effect_var: eff_var as i32,
            effect_value: eff_val as u64,
            effect_literal: eff_literal.clone(), //TODO: Get rid of clone. Either use lifetime or Rc
        }
    }
}

#[derive(Clone, Debug, Default)]
struct AxiomLiteral {
    condition_of: Vec<AxiomRule>,
}

#[derive(Debug, Clone, Default)]
struct NegationByFailureInfo {
    var_id: u32,
    literal: AxiomLiteral, // TODO: make this a reference to avoid cloning.
}

impl NegationByFailureInfo {
    pub fn new(var_id: u32, literal: AxiomLiteral) -> Self {
        NegationByFailureInfo { var_id, literal }
    }
}

struct AxiomEvaluator {
    numeric_task: Box<dyn AbstractNumericTask>,
    state_packer: IntDoublePacker,
    axiom_literals: Vec<Vec<AxiomLiteral>>,
    rules: Vec<AxiomRule>,
    comparison_axiom_layer: i32,
    first_propositional_axiom_layer: i32,
    last_propositional_axiom_layer: i32,
    last_arithmetic_axiom_layer: i32,
    nbf_info_by_layer: Vec<Vec<NegationByFailureInfo>>,
    queue: Vec<AxiomLiteral>, // Queue for processing axioms
}

impl AxiomEvaluator {
    pub fn new(numeric_task: Box<dyn AbstractNumericTask>, state_packer: IntDoublePacker) -> Self {
        let mut axiom_literals = vec![];
        let mut nbf_info_by_layer = vec![];

        let mut rules = vec![];
        let mut comparison_axiom_layer = -1;
        let mut first_propositional_axiom_layer = -1;
        let mut last_propositional_axiom_layer = -1;
        let mut last_arithmetic_axiom_layer = -1;

        for numeric_var in numeric_task.numeric_variables().iter() {
            last_arithmetic_axiom_layer =
                max(last_arithmetic_axiom_layer, numeric_var.axiom_layer());
        }

        for i in 0..numeric_task.get_num_variables() {
            let axiom_layer = numeric_task.get_variable_axiom_layer(i).unwrap();
            if axiom_layer == -1 {
                continue; // Skip regular variables
            }
            last_propositional_axiom_layer = max(last_propositional_axiom_layer, axiom_layer);
            if axiom_layer < first_propositional_axiom_layer
                || first_propositional_axiom_layer == -1
            {
                first_propositional_axiom_layer = axiom_layer;
            }
        }

        if first_propositional_axiom_layer >= 0 && numeric_task.get_num_cmp_axioms() > 0 {
            comparison_axiom_layer = last_propositional_axiom_layer;
            last_propositional_axiom_layer += 1;
            assert!(comparison_axiom_layer == last_arithmetic_axiom_layer + 1);
        }

        for var in numeric_task.variables().iter() {
            let literal = vec![AxiomLiteral::default(); var.domain_size() as usize];
            axiom_literals.push(literal);
        }

        for axiom in numeric_task.axioms().iter() {
            let cond_count = axiom.conditions.len();
            let eff_var = axiom.var_id as usize;
            let eff_val = axiom.effect_value as usize;
            let eff_literal = &axiom_literals[eff_var][eff_var];
            rules.push(AxiomRule::new(cond_count, eff_var, eff_val, eff_literal));
        }

        for i in 0..numeric_task.axioms().len() {
            let axiom = &numeric_task.axioms()[i];
            let conditions = axiom.conditions();
            for condition in conditions.iter() {
                axiom_literals[condition.var() as usize][condition.value() as usize]
                    .condition_of
                    .push(rules[i].clone()); //TODO: Get rid of clone
            }
        }

        let mut last_layer = -1;
        for i in 0..numeric_task.get_num_variables() {
            last_layer = max(
                last_layer,
                numeric_task.get_variable_axiom_layer(i).unwrap(),
            );
        }
        nbf_info_by_layer.resize(
            (last_layer + 1) as usize,
            vec![],
        );

        for var_id in 0..numeric_task.get_num_variables() {
            let axiom_layer = numeric_task.get_variable_axiom_layer(var_id).unwrap();
            if axiom_layer != -1 && axiom_layer != last_layer {
                let nbf_value = numeric_task.get_initial_state_values()[var_id as usize];
                let literal = axiom_literals[var_id as usize][nbf_value as usize].clone();
                let nbf_info = NegationByFailureInfo::new(
                    var_id as u32,
                    literal,
                );
                nbf_info_by_layer[axiom_layer as usize].push(nbf_info);
            }
        }

        AxiomEvaluator {
            numeric_task,
            state_packer,
            axiom_literals,
            rules,
            comparison_axiom_layer,
            first_propositional_axiom_layer,
            last_propositional_axiom_layer,
            last_arithmetic_axiom_layer,
            nbf_info_by_layer,
            queue: vec![],
        }
    }

    pub fn evaluate_arithmetic_axioms(
        &self,
        numeric_state: &mut Vec<f64>,
    ) -> Result<(), InvalidIndex> {
        for axiom in self.numeric_task.assignment_axioms() {
            let result = axiom.update_values(numeric_state)?;
        }

        Ok(())
    }

    pub fn evaluate_comparison_axioms(
        &mut self,
        buffer: &mut [u64],
        numeric_state: &mut Vec<f64>,
    ) -> Result<bool, InvalidIndex> {
        for axiom in self.numeric_task.comparison_axioms() {
            let result = axiom.update_values(numeric_state)?;
            self.state_packer
                .set(buffer, axiom.get_affected_var_id(), result as u64);
        }

        Ok(true)
    }

    pub fn evaluate_propositional_axioms(&mut self, buffer: &mut [u64]) -> Result<(), InvalidIndex> {
        if self.numeric_task.axioms().is_empty() {
            return Ok(());
        }

        for i in 0..self.numeric_task.get_num_variables() {
            let axiom_layer = self.numeric_task.get_variable_axiom_layer(i).unwrap();
            match axiom_layer  {
                -1 => {
                    self.queue.push(self.axiom_literals[i as usize][self.state_packer.get(buffer, i) as usize].clone());
                },
                _ => {}
            }
        }


        Ok(())
    }
}
