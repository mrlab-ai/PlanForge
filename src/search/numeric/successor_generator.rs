use crate::search::numeric::{
    numeric_task::{AbstractNumericTask, Fact, Operator},
    utils::errors::ConstructError,
};
use std::collections::VecDeque;
use std::fmt::Debug;

type Condition<'a> = Vec<&'a Fact>;
pub type ApplicableOperator<'a> = (&'a Operator, usize);

pub struct GroundedSuccessorGenerator<'a> {
    task: &'a dyn AbstractNumericTask,
    conditions: Vec<Condition<'a>>,
    next_condition_by_operator: Vec<usize>, // index into conditions
}

impl<'a> GroundedSuccessorGenerator<'a> {
    pub fn new(task: &'a dyn AbstractNumericTask) -> GroundedSuccessorGenerator<'a> {
        let operators = task.get_operators();
        let mut conditions = vec![];
        let mut next_condition_by_operator = vec![];

        for operator in operators.iter() {
            let mut condition = vec![];
            for precondition in operator.preconditions().iter() {
                condition.push(precondition);
            }
            // Sort by (var, value) to group by variable and keep order stable
            condition.sort_by_key(|f| (f.var(), f.value()));
            conditions.push(condition);
            next_condition_by_operator.push(0);
        }

        GroundedSuccessorGenerator {
            task,
            conditions,
            next_condition_by_operator,
        }
    }

    pub fn construct(
        &mut self,
        branch_var_id: &mut u32,
        queue: &mut VecDeque<ApplicableOperator<'a>>,
    ) -> Result<Box<dyn Node<'a>>, ConstructError> {
        if queue.is_empty() {
            return Ok(Box::new(LeafNode::new(None)));
        }
        loop {
            // Test if no further switch is necessary (or possible).
            if *branch_var_id as usize >= self.task.variables().len() {
                let ops: Vec<ApplicableOperator<'a>> = queue.iter().copied().collect();
                return Ok(Box::new(LeafNode::new(Some(ops))));
            }

            let branch_var = &self.task.variables()[*branch_var_id as usize];
            let num_children = branch_var.domain_size();

            let mut operators_for_value = vec![VecDeque::new(); num_children as usize];
            let mut default_operators = VecDeque::new();
            let mut applicable_operators = Vec::new();

            let mut all_ops_immediate = true;
            let mut var_interesting = false;

            while !queue.is_empty() {
                let (op, op_id) = queue.pop_front().ok_or(ConstructError {
                    message: "Queue is empty".to_string(),
                })?;
                let condition_index = self.next_condition_by_operator[op_id as usize];

                if condition_index >= self.conditions[op_id as usize].len() {
                    var_interesting = true;
                    applicable_operators.push((op, op_id));
                } else {
                    all_ops_immediate = false;
                    let fact = &self.conditions[op_id as usize][condition_index];
                    if fact.var() == *branch_var_id {
                        var_interesting = true;
                        let mut new_index = condition_index;
                        while new_index < self.conditions[op_id as usize].len()
                            && self.conditions[op_id as usize][new_index].var() == *branch_var_id
                        {
                            new_index += 1;
                        }
                        self.next_condition_by_operator[op_id as usize] = new_index;
                        operators_for_value[fact.value() as usize].push_back((op, op_id));
                    } else {
                        default_operators.push_back((op, op_id));
                    }
                }
            }

            if all_ops_immediate {
                return Ok(Box::new(LeafNode::new(Some(applicable_operators))));
            } else if var_interesting {
                let mut children = Vec::with_capacity(operators_for_value.len());
                for ops in operators_for_value.iter_mut() {
                    children.push(self.construct(&mut (*branch_var_id + 1), ops)?);
                }
                let default_branch =
                    self.construct(&mut (*branch_var_id + 1), &mut default_operators)?;
                return Ok(Box::new(BranchNode::new(
                    *branch_var_id,
                    applicable_operators,
                    children,
                    Some(default_branch),
                )));
            } else {
                *branch_var_id += 1;
                std::mem::swap(&mut default_operators, queue);
            }
        }
    }
}

pub trait Node<'a>: 'a + Debug {
    fn get_applicable_operators(
        &self,
        state: &[i32],
        applicable_operators: &mut Vec<ApplicableOperator<'a>>,
    );
}

#[derive(Debug)]
struct BranchNode<'a> {
    var_id: u32,
    immediate_operators: Vec<ApplicableOperator<'a>>,
    value_children: Vec<Box<dyn Node<'a>>>,
    default_child: Option<Box<dyn Node<'a>>>,
}

impl<'a> BranchNode<'a> {
    pub fn new(
        var_id: u32,
        immediate_operators: Vec<ApplicableOperator<'a>>,
        value_children: Vec<Box<dyn Node<'a>>>,
        default_child: Option<Box<dyn Node<'a>>>,
    ) -> BranchNode<'a> {
        BranchNode {
            var_id,
            immediate_operators,
            value_children,
            default_child,
        }
    }
}

impl<'a> Node<'a> for BranchNode<'a> {
    fn get_applicable_operators(
        &self,
        state: &[i32],
        applicable_operators: &mut Vec<ApplicableOperator<'a>>,
    ) {
        applicable_operators.extend(self.immediate_operators.iter().copied());
        let value = state[self.var_id as usize];
        self.value_children[value as usize].get_applicable_operators(state, applicable_operators);

        // Also process the default child, which contains operators that don't depend on this variable
        if let Some(ref default_child) = self.default_child {
            default_child.get_applicable_operators(state, applicable_operators);
        }
    }
}

#[derive(Debug)]
struct LeafNode<'a> {
    applicable_operators: Option<Vec<ApplicableOperator<'a>>>,
}

impl<'a> LeafNode<'a> {
    pub fn new(applicable_operators: Option<Vec<ApplicableOperator<'a>>>) -> LeafNode<'a> {
        LeafNode {
            applicable_operators,
        }
    }
}

impl<'a> Node<'a> for LeafNode<'a> {
    fn get_applicable_operators(
        &self,
        _state: &[i32],
        applicable_operators: &mut Vec<ApplicableOperator<'a>>,
    ) {
        if let Some(operators) = &self.applicable_operators {
            applicable_operators.extend(operators.iter().copied());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        preprocess::numeric_parser::parse_numeric_sas_output,
        search::numeric::{
            numeric_task::NumericRootTask, state_registry::StateRegistry,
            utils::int_packer::IntDoublePacker,
        },
        setup_axiom_evaluator, setup_state_packer, setup_state_registry,
    };

    fn setup_problems() -> Vec<NumericRootTask> {
        let mut problems = vec![];
        for file in std::fs::read_dir("misc/numeric_sas").unwrap() {
            let file = file.unwrap();
            let path = file.path();
            if !path.is_file() || path.extension().and_then(|ext| ext.to_str()) != Some("sas") {
                continue;
            }

            let input = std::fs::read_to_string(&path).unwrap();
            let (unconsumed_input, problem) = parse_numeric_sas_output(&input).unwrap();
            assert!(
                unconsumed_input.is_empty(),
                "Unconsumed input: {}",
                unconsumed_input
            );
            if path.file_name().and_then(|name| name.to_str()) == Some("example3.sas") {
                problems.push(problem);
            }
        }
        problems
    }

    #[test]
    fn test_grounded_successor_generator() {
        let problems = setup_problems();

        for problem in problems {
            let mut generator = GroundedSuccessorGenerator::new(&problem);

            let mut queue = VecDeque::new();
            for (op_id, operator) in problem.get_operators().iter().enumerate() {
                queue.push_back((operator, op_id));
            }

            let state_packer = setup_state_packer(&problem);
            let axiom_evaluator = setup_axiom_evaluator(&problem, &state_packer);
            let mut state_registry =
                setup_state_registry(&problem, &state_packer, &axiom_evaluator);

            let state = state_registry.get_initial_state();
            let state = state.get_state(&state_registry);
            println!("State values: {:?}", state);

            let node = generator.construct(&mut 0, &mut queue).unwrap();

            let mut applicable_operators: Vec<ApplicableOperator<'_>> = Vec::new();
            node.get_applicable_operators(&state[..], &mut applicable_operators);

            //dbg!("Facts: {:?}", facts_refs);
            dbg!("Applicable operators: {:?}", applicable_operators);
            //dbg!("Node: {:?}", node);
        }
    }
}
