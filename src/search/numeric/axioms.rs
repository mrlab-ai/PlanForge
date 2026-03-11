use std::cmp::max;
use std::cell::RefCell;

use crate::search::numeric::{
    self,
    numeric_task::{AbstractNumericTask, Fact},
    utils::{
        errors::{AxiomEvalError, InvalidIndex},
        int_packer::IntDoublePacker,
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
    pub fn compare(&self, numeric_values: &mut Vec<f64>, left: i32, right: i32) -> bool {
        let (left, right) = (
            numeric_values[left as usize],
            numeric_values[right as usize],
        );
        match self {
            ComparisonOperator::LessThan => left < right,
            ComparisonOperator::LessThanOrEqual => left <= right,
            ComparisonOperator::Equal => left == right,
            ComparisonOperator::GreaterThanOrEqual => left >= right,
            ComparisonOperator::GreaterThan => left > right,
            ComparisonOperator::UnEqual => left != right,
        }
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

    pub fn update_values(&self, numeric_state: &mut Vec<f64>) -> Result<bool, InvalidIndex> {
        let left = self.left_hand_side as usize;
        let right = self.right_hand_side as usize;
        if left >= numeric_state.len() || right >= numeric_state.len() {
            return Err(InvalidIndex {
                length: numeric_state.len() as u32,
                index: left as u32,
            });
        }
        let comp_op = &self.operator;
        let result = comp_op.compare(numeric_state, self.left_hand_side, self.right_hand_side);
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
#[derive(Debug)]
struct AxiomRule {
    condition_count: i32,
    effect_var: i32,
    effect_value: u64,
}

impl AxiomRule {
    pub fn new(cond_count: usize, eff_var: usize, eff_val: usize) -> Self {
        AxiomRule {
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
    pub fn new(var_id: u32, literal_value: usize) -> Self {
        NegationByFailureInfo {
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

pub struct AxiomEvaluator<'a> {
    numeric_task: &'a dyn AbstractNumericTask,
    state_packer: &'a IntDoublePacker,
    axiom_literals: Vec<Vec<AxiomLiteral>>,
    rules: Vec<AxiomRule>,
    comparison_axiom_layer: i32,
    first_propositional_axiom_layer: i32,
    last_propositional_axiom_layer: i32,
    last_arithmetic_axiom_layer: i32,
    nbf_info_by_layer: Vec<Vec<NegationByFailureInfo>>,
    queue: RefCell<Vec<LiteralRef>>,
    unsatisfied_conditions: RefCell<Vec<i32>>,
}

impl<'a> AxiomEvaluator<'a> {
    pub fn new(
        numeric_task: &'a dyn AbstractNumericTask,
        state_packer: &'a IntDoublePacker,
    ) -> Self {
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
            comparison_axiom_layer = first_propositional_axiom_layer;
            first_propositional_axiom_layer += 1;
            debug_assert_eq!(comparison_axiom_layer, last_arithmetic_axiom_layer + 1);
        }

        for var in numeric_task.variables().iter() {
            let literal = vec![AxiomLiteral::default(); var.domain_size() as usize];
            axiom_literals.push(literal);
        }

        for axiom in numeric_task.axioms().iter() {
            let cond_count = axiom.conditions.len();
            let eff_var = axiom.var_id as usize;
            let eff_val = axiom.effect_value as usize;
            rules.push(AxiomRule::new(cond_count, eff_var, eff_val));
        }

        for i in 0..numeric_task.axioms().len() {
            let axiom = &numeric_task.axioms()[i];
            let conditions = axiom.conditions();
            for condition in conditions.iter() {
                axiom_literals[condition.var() as usize][condition.value() as usize]
                    .condition_of
                    .push(i);
            }
        }

        let mut last_layer = -1;
        for i in 0..numeric_task.get_num_variables() {
            last_layer = max(
                last_layer,
                numeric_task.get_variable_axiom_layer(i).unwrap(),
            );
        }
        nbf_info_by_layer.resize((last_layer + 1) as usize, vec![]);

        for var_id in 0..numeric_task.get_num_variables() {
            let axiom_layer = numeric_task.get_variable_axiom_layer(var_id).unwrap();
            if axiom_layer != -1 && axiom_layer != last_layer {
                let nbf_value = numeric_task.get_variable_default_axiom_value(var_id).unwrap();
                let nbf_info = NegationByFailureInfo::new(var_id as u32, nbf_value as usize);
                nbf_info_by_layer[axiom_layer as usize].push(nbf_info);
            }
        }

        //TODO: evaluate arithmetic axioms here instead of state_registry

        let rule_count = rules.len();

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
            queue: RefCell::new(Vec::new()),
            unsatisfied_conditions: RefCell::new(vec![0; rule_count]),
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
        &self,
        buffer: &mut [u64],
        numeric_state: &mut Vec<f64>,
    ) -> Result<bool, AxiomEvalError> {
        for axiom in self.numeric_task.comparison_axioms() {
            let result = axiom.update_values(numeric_state).map_err(|e| {
                AxiomEvalError::InvalidIndex(numeric::utils::errors::InvalidIndex {
                    length: numeric_state.len() as u32,
                    index: e.index,
                })
            })?;
            self.state_packer
                .set(buffer, axiom.get_affected_var_id(), !result as u64);
        }

        Ok(true)
    }

    pub fn evaluate_propositional_axioms(&self, buffer: &mut [u64]) -> Result<(), AxiomEvalError> {
        if self.numeric_task.axioms().is_empty() {
            return Ok(());
        }

        let mut queue = self.queue.borrow_mut();
        queue.clear();

        let mut unsatisfied_conditions = self.unsatisfied_conditions.borrow_mut();
        if unsatisfied_conditions.len() != self.rules.len() {
            unsatisfied_conditions.resize(self.rules.len(), 0);
        }

        // Initialize queue with current variable values (following C++ logic)
        for i in 0..self.numeric_task.get_num_variables() {
            let axiom_layer = self.numeric_task.get_variable_axiom_layer(i).unwrap();
            if axiom_layer == -1 {
                // Non-derived variable -> push immediately
                queue.push(LiteralRef {
                    var_id: i as usize,
                    value: self.state_packer.get(buffer, i) as usize,
                });
            } else if axiom_layer <= self.last_arithmetic_axiom_layer {
                return Err(AxiomEvalError::WrongAxiomLayer(
                    numeric::utils::errors::WrongAxiomLayer {
                        axiom_layer,
                        last_arithmetic_axiom_layer: self.last_arithmetic_axiom_layer,
                    },
                ));
            } else if axiom_layer == self.comparison_axiom_layer {
                // Variable is the result of a comparison axiom
                queue.push(LiteralRef {
                    var_id: i as usize,
                    value: self.state_packer.get(buffer, i) as usize,
                });
            } else if axiom_layer <= self.last_propositional_axiom_layer {
                // Set derived variables to their default values initially
                let default_value = self.numeric_task.get_variable_default_axiom_value(i).unwrap();
                self.state_packer.set(buffer, i, default_value as u64);
            } else {
                return Err(AxiomEvalError::WrongAxiomLayer(
                    numeric::utils::errors::WrongAxiomLayer {
                        axiom_layer,
                        last_arithmetic_axiom_layer: self.last_arithmetic_axiom_layer,
                    },
                ));
            }
        }

        for (rule_index, rule) in self.rules.iter().enumerate() {
            unsatisfied_conditions[rule_index] = rule.condition_count;

            // Handle trivial axioms (no conditions)
            if rule.condition_count == 0 {
                let var_no = rule.effect_var;
                let val = rule.effect_value;
                if self.state_packer.get(buffer, var_no) != val {
                    self.state_packer.set(buffer, var_no, val);
                    queue.push(LiteralRef {
                        var_id: var_no as usize,
                        value: val as usize,
                    });
                }
            }
        }

        // Process each axiom layer
        for layer_no in 0..self.nbf_info_by_layer.len() {
            // Apply Horn rules - continue until queue is empty
            while let Some(curr_literal) = queue.pop() {
                let dependent_rules = &self.axiom_literals[curr_literal.var_id][curr_literal.value].condition_of;

                // For each rule that depends on this literal
                for &rule_index in dependent_rules {
                    let remaining = &mut unsatisfied_conditions[rule_index];
                    *remaining -= 1;

                    if *remaining == 0 {
                        let rule = &self.rules[rule_index];
                        let var_no = rule.effect_var;
                        let val = rule.effect_value;
                        if self.state_packer.get(buffer, var_no) != val {
                            self.state_packer.set(buffer, var_no, val);
                            queue.push(LiteralRef {
                                var_id: var_no as usize,
                                value: val as usize,
                            });
                        }
                    }
                }
            }

            // Apply negation by failure rules (skip in last iteration for optimization)
            if layer_no != self.nbf_info_by_layer.len() - 1 {
                let nbf_info = &self.nbf_info_by_layer[layer_no];
                for info in nbf_info {
                    let var_no = info.var_id as i32;
                    let default_value = self
                        .numeric_task
                        .get_variable_default_axiom_value(var_no)
                        .unwrap();
                    if self.state_packer.get(buffer, var_no) == default_value as u64 {
                        queue.push(LiteralRef {
                            var_id: var_no as usize,
                            value: info.literal_value,
                        });
                    }
                }
            }
        }

        Ok(())
    }

    pub fn evaluate(
        &self,
        buffer: &mut [u64],
        numeric_state: &mut Vec<f64>,
    ) -> Result<(), AxiomEvalError> {
        if !self.has_axioms() {
            return Ok(());
        }
        if self.has_numeric_axioms() {
            self.evaluate_comparison_axioms(buffer, numeric_state)?;
        }
        if self.has_propositional_axioms() {
            self.evaluate_propositional_axioms(buffer)?;
        }
        Ok(())
    }

    fn has_axioms(&self) -> bool {
        self.has_numeric_axioms() || self.has_propositional_axioms()
    }

    pub fn has_numeric_axioms(&self) -> bool {
        self.numeric_task.assignment_axioms().len() > 0
            || self.numeric_task.comparison_axioms().len() > 0
    }

    fn has_propositional_axioms(&self) -> bool {
        self.numeric_task.axioms().len() > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{preprocess::numeric_parser, search::numeric::numeric_task::NumericRootTask};

    fn setup_problems() -> Vec<NumericRootTask> {
        let mut problems = vec![];
        for file in std::fs::read_dir("misc/numeric_sas").unwrap() {
            let file = file.unwrap();
            if !file.file_name().to_string_lossy().contains("example1") {
                continue;
            }
            if file.path().extension().unwrap() == "sas" {
                let input = std::fs::read_to_string(file.path()).unwrap();
                let (unconsumed_input, problem) =
                    numeric_parser::parse_numeric_sas_output(&input).unwrap();
                assert!(
                    unconsumed_input.is_empty(),
                    "Unconsumed input: {}",
                    unconsumed_input
                );
                problems.push(problem);
                println!("Parsed problem from {:?}", file.path());
                break;
            }
        }
        problems
    }

    #[test]
    fn test_axiom_evaluator_creation() {
        let problems = setup_problems();
        assert!(!problems.is_empty());

        for problem in problems {
            let mut domain_sizes = vec![];
            for var in problem.variables().iter() {
                domain_sizes.push(var.domain_size() as u64);
            }
            for numeric_var in problem.numeric_variables().iter() {
                domain_sizes.push(u64::MAX);
            }

            let state_packer = IntDoublePacker::new(&domain_sizes);
            let axiom_evaluator = AxiomEvaluator::new(&problem, &state_packer);

            let init_state = problem.get_initial_propositional_state_values();
            let mut buffer = vec![0; axiom_evaluator.state_packer.num_bins() as usize];
            for (i, value) in init_state.iter().enumerate() {
                dbg!(i, value);
                axiom_evaluator
                    .state_packer
                    .set(&mut buffer, i as i32, *value as u64);
            }

            dbg!(axiom_evaluator.state_packer.get(&buffer, 0));

            dbg!(&buffer);
            dbg!(problem.numeric_variables().len());
        }
    }

    #[test]
    fn test_example1_axiom_evaluation() {
        // Load specifically example1.sas
        let input = std::fs::read_to_string("misc/numeric_sas/example1.sas").unwrap();
        let (unconsumed_input, problem) = numeric_parser::parse_numeric_sas_output(&input).unwrap();
        assert!(unconsumed_input.is_empty());

        // Set up state packer and axiom evaluator
        let mut domain_sizes = vec![];
        for var in problem.variables().iter() {
            domain_sizes.push(var.domain_size() as u64);
        }
        for numeric_var in problem.numeric_variables().iter() {
            domain_sizes.push(u64::MAX);
        }

        let state_packer = IntDoublePacker::new(&domain_sizes);
        let axiom_evaluator = AxiomEvaluator::new(&problem, &state_packer);

        // Verify axiom structure is set up correctly
        assert!(
            axiom_evaluator.has_numeric_axioms(),
            "Should have numeric axioms"
        );
        assert!(
            axiom_evaluator.has_propositional_axioms(),
            "Should have propositional axioms"
        );
        assert_eq!(
            problem.comparison_axioms().len(),
            5,
            "Should have 5 comparison axioms"
        );
        assert_eq!(
            problem.axioms().len(),
            2,
            "Should have 2 propositional axioms"
        );

        // Set up initial state buffer
        let init_state = problem.get_initial_propositional_state_values();
        let mut buffer = vec![0; axiom_evaluator.state_packer.num_bins() as usize];

        // Pack initial propositional state into buffer
        for (i, value) in init_state.iter().enumerate() {
            axiom_evaluator
                .state_packer
                .set(&mut buffer, i as i32, *value as u64);
        }

        // Test initial state before axiom evaluation
        println!("=== Testing Example1 Axiom Evaluation ===");
        println!("Initial buffer state:");
        for i in 0..problem.variables().len() {
            let val = axiom_evaluator.state_packer.get(&buffer, i as i32);
            println!("  var {} = {}", i, val);
        }

        // Set up initial numeric state
        let mut numeric_state = problem.get_initial_numeric_state_values().clone();
        println!("Initial numeric state:");
        for (i, val) in numeric_state.iter().enumerate() {
            println!("  numeric_var_{} = {}", i, val);
        }

        // Test arithmetic axiom evaluation
        let result = axiom_evaluator.evaluate_arithmetic_axioms(&mut numeric_state);
        assert!(result.is_ok(), "Arithmetic axiom evaluation should succeed");

        println!("After arithmetic axioms:");
        for (i, val) in numeric_state.iter().enumerate() {
            println!("  numeric_var_{} = {}", i, val);
        }

        // Test comparison axiom evaluation
        let result = axiom_evaluator.evaluate_comparison_axioms(&mut buffer, &mut numeric_state);
        assert!(result.is_ok(), "Comparison axiom evaluation should succeed");

        println!("After comparison axioms:");
        for i in 0..problem.variables().len() {
            let val = axiom_evaluator.state_packer.get(&buffer, i as i32);
            println!("  var {} = {}", i, val);
        }

        // Test propositional axiom evaluation
        let result = axiom_evaluator.evaluate_propositional_axioms(&mut buffer);
        assert!(
            result.is_ok(),
            "Propositional axiom evaluation should succeed"
        );

        println!("After propositional axioms:");
        for i in 0..problem.variables().len() {
            let val = axiom_evaluator.state_packer.get(&buffer, i as i32);
            println!("  var {} = {}", i, val);
        }

        // Test complete axiom evaluation
        let mut numeric_state_copy = problem.get_initial_numeric_state_values().clone();
        let mut buffer_copy = vec![0; axiom_evaluator.state_packer.num_bins() as usize];
        for (i, value) in init_state.iter().enumerate() {
            axiom_evaluator
                .state_packer
                .set(&mut buffer_copy, i as i32, *value as u64);
        }

        let result = axiom_evaluator.evaluate(&mut buffer_copy, &mut numeric_state_copy);
        assert!(result.is_ok(), "Complete axiom evaluation should succeed");

        println!("After complete evaluation:");
        for i in 0..problem.variables().len() {
            let val = axiom_evaluator.state_packer.get(&buffer_copy, i as i32);
            println!("  var {} = {}", i, val);
        }

        // Test specific axiom behavior based on example1.sas analysis
        // The complete evaluation should actually reach the goal state!
        let var5_value = axiom_evaluator.state_packer.get(&buffer_copy, 5);
        println!("Variable 5 final value: {}", var5_value);

        let var4_value = axiom_evaluator.state_packer.get(&buffer_copy, 4);
        println!("Variable 4 final value: {}", var4_value);
        println!(
            "  numeric_var_16 = {}, numeric_var_2 = {}",
            numeric_state_copy[16], numeric_state_copy[2]
        );
        println!(
            "  Comparison result: {} >= {} = {}",
            numeric_state_copy[16],
            numeric_state_copy[2],
            numeric_state_copy[16] >= numeric_state_copy[2]
        );

        // Variables 0,1,2,3 should all be 0 (comparison results should be true)
        for i in 0..4 {
            let val = axiom_evaluator.state_packer.get(&buffer_copy, i);
            println!("Variable {} = {} (comparison axiom result)", i, val);
        }

        // The complete evaluation actually reaches the goal state where:
        // - Variable 4 becomes 0 (because numeric_var_16 becomes >= numeric_var_2)
        // - Variable 5 becomes 0 (because all conditions var1=0, var2=0, var4=0 are met)
        assert_eq!(
            var4_value, 0,
            "Variable 4 should be 0 after complete evaluation"
        );
        assert_eq!(
            var5_value, 0,
            "Variable 5 should be 0 after complete evaluation (goal reached!)"
        );

        // Verify that the goal condition is actually satisfied
        println!(
            "🎉 Goal state reached! Variable 5 = {} (required: 0)",
            var5_value
        );
    }
}
