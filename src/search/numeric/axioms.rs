use std::collections::btree_map::Values;

use crate::{
    parser::numeric_parser,
    search::numeric::{
        numeric_task::{self, AbstractNumericTask},
        utils::{errors::InvalidIndex, int_packer::IntDoublePacker},
    },
};

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

struct AxiomEvaluator {
    pub numeric_task: Box<dyn AbstractNumericTask>,
    pub state_packer: IntDoublePacker,
}

impl AxiomEvaluator {
    pub fn new(numeric_task: Box<dyn AbstractNumericTask>, state_packer: IntDoublePacker) -> Self {
        AxiomEvaluator {
            numeric_task,
            state_packer,
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
}
