use std::{cmp::max, collections::btree_map::Values};

use crate::{
    parser::numeric_parser,
    search::numeric::{
        self,
        numeric_task::{self, AbstractNumericTask, Fact},
        utils::{errors::InvalidIndex, int_packer::IntDoublePacker},
    },
};

#[derive(Debug)]
pub struct Axiom {
    conditions: Vec<Fact>,
    var_id: u32,
    precondition_value: u32,
    effect_value: u32,
}

impl Axiom {
    pub fn new(
        conditions: Vec<Fact>,
        var_id: u32,
        precondition_value: u32,
        effect_value: u32,
    ) -> Self {
        Axiom {
            conditions,
            var_id,
            precondition_value,
            effect_value,
        }
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
struct AxiomRule;
#[derive(Clone, Debug, Default)]
struct AxiomLiteral {
    condition_of: Vec<AxiomRule>,
}

struct AxiomEvaluator {
    numeric_task: Box<dyn AbstractNumericTask>,
    state_packer: IntDoublePacker,
    axiom_literals: Vec<Vec<AxiomLiteral>>,
    comparison_axiom_layer: i32,
    first_propositional_axiom_layer: i32,
    last_propositional_axiom_layer: i32,
}

impl AxiomEvaluator {
    pub fn new(numeric_task: Box<dyn AbstractNumericTask>, state_packer: IntDoublePacker) -> Self {
        let mut axiom_literals = vec![vec![]; numeric_task.get_num_axioms() as usize];
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

        AxiomEvaluator {
            numeric_task,
            state_packer,
            axiom_literals,
            comparison_axiom_layer,
            first_propositional_axiom_layer,
            last_propositional_axiom_layer,
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

    pub fn evaluate_propositional_axioms(&self, buffer: &mut [u64]) -> Result<(), InvalidIndex> {
        if self.numeric_task.axioms().is_empty() {
            return Ok(());
        }

        Ok(())
    }
}
