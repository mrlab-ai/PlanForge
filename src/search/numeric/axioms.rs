use crate::{parser::numeric_parser, search::numeric::{numeric_task::{ self, AbstractNumericTask, CalOperator }, utils::int_packer::IntDoublePacker}};

struct AxiomEvaluator {
    pub numeric_task: Box<dyn AbstractNumericTask>,
    pub state_packer: IntDoublePacker,
}

impl AxiomEvaluator {
    pub fn new(numeric_task: Box<dyn AbstractNumericTask>, state_packer: IntDoublePacker) -> Self {
        AxiomEvaluator { numeric_task, state_packer }
    }

    pub fn evaluate_arithmetic_axioms(&self, numeric_state: &mut Vec<f64>) -> Result<bool, String> {
        for axiom in self.numeric_task.assignment_axioms() {
            let left_var = axiom.get_left_var_id();
            let right_var = axiom.get_right_var_id();
            let left_value = numeric_state
                .get(left_var as usize)
                .ok_or_else(|| format!("Left variable {} not found in numeric state", left_var))?;
            let right_value = numeric_state
                .get(right_var as usize)
                .ok_or_else(|| format!("Right variable {} not found in numeric state", right_var))?;
            let affected_var_id = axiom.get_affected_var_id();

            let mut result = 0.0;

            match axiom.get_operator() {
                CalOperator::Sum => {
                    numeric_state[affected_var_id as usize] = left_value + right_value;
                }
                CalOperator::Difference => {
                    numeric_state[affected_var_id as usize] = left_value - right_value;
                }
                CalOperator::Product => {
                    numeric_state[affected_var_id as usize] = left_value * right_value;
                }
                CalOperator::Division => {
                    if *right_value == 0.0 {
                        return Err("Division by zero in arithmetic axiom".into());
                    }
                    numeric_state[affected_var_id as usize] = left_value / right_value;
                }
                _ => {
                    return Err("Unsupported operator".into());
                }
            }
        }

        Ok(true)
    }

    pub fn evaluate_comparison_axioms(&mut self, buffer: &mut [u64], numeric_state: &Vec<f64>) -> Result<bool, String> {
        for axiom in self.numeric_task.comparison_axioms() {
            let result = axiom.evaluate(numeric_state);
            let result = if result { 1 } else { 0 };
            self.state_packer.set(buffer, axiom.get_affected_var_id(), result);
           
        }

        Ok(true)
    }




}
