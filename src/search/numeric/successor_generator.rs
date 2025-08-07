use std::collections::{LinkedList, VecDeque};

use crate::search::{
    classical::classical_task::Operator,
    numeric::{
        numeric_task::{AbstractNumericTask, Fact},
        utils::errors::ConstructError,
    },
};

trait OperatorGenerator {
    fn generate_applicable_operators(
        &self,
        state: &Vec<i32>,
        numeric_state: &Vec<f64>,
    ) -> Vec<&Operator>;
}

type Condition<'a> = Vec<&'a Fact>;

struct GroundedSuccessorGenerator<'a> {
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
            condition.sort(); // only works if &Condition<'a>: Ord
            conditions.push(condition);
            next_condition_by_operator.push(conditions.len() - 1);
        }

        GroundedSuccessorGenerator {
            task,
            conditions,
            next_condition_by_operator,
        }
    }

    fn construct(
        &self,
        branch_var_id: &mut u32,
        queue: &mut VecDeque<(&'a Operator, u32)>,
    ) -> Result<Box<dyn Node<'a>>, ConstructError> {
        if queue.is_empty() {
            return Ok(Box::new(LeafNode::new(None)));
        }
        let branch_var = &self.task.variables()[*branch_var_id as usize];
        let num_children = branch_var.domain_size();

        let mut operators_for_value = vec![VecDeque::new(); num_children as usize];
        let mut default_operators = VecDeque::new();
        let mut applicable_operators = VecDeque::new();

        let mut all_ops_immediate = true;
        let mut var_interesting = false;


        while !queue.is_empty() {
            let (op, op_id) = queue.pop_front().ok_or(ConstructError {
                message: "Queue is empty".to_string(),
            })?;
            let condition_index = self.next_condition_by_operator[op_id as usize];
            
            let mut condition_iter = self.conditions[condition_index].iter();

            if condition_iter.len() == 0 {
                var_interesting = true;
                applicable_operators.push_back(op);
            } else {
                all_ops_immediate = false;
                let mut fact = condition_iter.next().ok_or(ConstructError {
                    message: "Condition iterator is empty".to_string(),
                })?;
                if fact.var() == *branch_var_id {
                    while condition_iter.len() > 0 {
                        fact = condition_iter.next().ok_or(ConstructError {
                            message: "Condition iterator is empty".to_string(),
                        })?;
                    }
                    operators_for_value[fact.value() as usize].push_back((op, op_id));
                } else {
                    default_operators.push_back((op, op_id));
                }
            }
        }

        if all_ops_immediate {
            return Ok(Box::new(LeafNode::new(Some(applicable_operators))));
        } else if var_interesting {
            let mut children = vec![];
            for ops in operators_for_value.iter_mut() {
                children.push(self.construct(&mut (*branch_var_id + 1), ops)?);
            }
            let default_branch =
                self.construct(&mut (*branch_var_id + 1), &mut default_operators)?;
            return Ok(Box::new(BranchNode::new(
                *branch_var_id,
                applicable_operators,
                children,
                default_branch,
            )));
        } else {
            *branch_var_id += 1;
            std::mem::swap(&mut default_operators, queue);
        }

        todo!()
    }
}
trait Node<'a>: 'a {
    fn get_applicable_operators(
        &self,
        state: &Vec<&'a Fact>,
        applicable_operators: &mut VecDeque<&'a Operator>,
    );
}

struct BranchNode<'a> {
    var_id: u32,
    immediate_operators: VecDeque<&'a Operator>,
    value_children: Vec<Box<dyn Node<'a>>>,
    default_child: Box<dyn Node<'a>>,
}

impl<'a> BranchNode<'a> {
    pub fn new(
        var_id: u32,
        immediate_operators: VecDeque<&'a Operator>,
        value_children: Vec<Box<dyn Node<'a>>>,
        default_child: Box<dyn Node<'a>>,
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
        state: &Vec<&'a Fact>,
        applicable_operators: &mut VecDeque<&'a Operator>,
    ) {
        for operator in &self.immediate_operators {
            applicable_operators.push_back(operator);
        }
        let value = state[self.var_id as usize].value();
        self.value_children[value as usize].get_applicable_operators(state, applicable_operators);
    }
}

struct LeafNode<'a> {
    applicable_operators: Option<VecDeque<&'a Operator>>,
}

impl<'a> LeafNode<'a> {
    pub fn new(applicable_operators: Option<VecDeque<&'a Operator>>) -> LeafNode<'a> {
        LeafNode {
            applicable_operators,
        }
    }
}

impl<'a> Node<'a> for LeafNode<'a> {
    fn get_applicable_operators(
        &self,
        _state: &Vec<&'a Fact>,
        applicable_operators: &mut VecDeque<&'a Operator>,
    ) {
        if let Some(operators) = &self.applicable_operators {
            applicable_operators.extend(operators.iter());
        }
    }
}
