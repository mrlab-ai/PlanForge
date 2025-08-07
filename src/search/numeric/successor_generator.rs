use std::{
    collections::{LinkedList, VecDeque},
};

use crate::search::{
    classical::classical_task::Operator,
    numeric::{
        numeric_task::{AbstractNumericTask, Fact},
        utils::errors::ConstructError,
    },
};

type Condition<'a> = Vec<&'a Fact>;

struct GroundedSuccessorGenerator<'a> {
    task: &'a dyn AbstractNumericTask,
    conditions: Vec<Condition<'a>>,
    next_condition_by_operator: Vec<Condition<'a>>,
}

trait OperatorGenerator {
    fn generate_applicable_operators(
        &self,
        state: &Vec<i32>,
        numeric_state: &Vec<f64>,
    ) -> Vec<&Operator>;
}

impl GroundedSuccessorGenerator<'_> {
    pub fn new(task: &dyn AbstractNumericTask) -> GroundedSuccessorGenerator {
        let operators = task.get_operators();
        let mut conditions = vec![];

        for operator in operators.iter() {
            let mut condition = vec![];
            for precondition in operator.preconditions().iter() {
                condition.push(precondition);
            }
            condition.sort();
            conditions.push(condition);
            let num_vars = task.get_num_variables() as i32;
        }

        todo!();

        GroundedSuccessorGenerator {
            task,
            conditions: vec![],
            next_condition_by_operator: vec![],
        }
    }

    fn construct(
        &self,
        branch_var_id: u32,
        queue: &mut VecDeque<(Operator, u32)>,
    ) -> Result<Box<dyn Node>, ConstructError> {
        if queue.is_empty() {
            return Ok(Box::new(LeafNode::new(None)));
        }
        let branch_var = &self.task.variables()[branch_var_id as usize];
        let num_children = branch_var.domain_size();

        let mut operators_for_value = vec![];
        let mut default_operators = LinkedList::new();
        let mut applicable_operators = LinkedList::new();

        let mut all_ops_immediate = true;
        let mut var_interesting = false;

        let num_operators = self.task.get_num_operators();

        while !queue.is_empty() {
            let (op, op_id) = queue.pop_front().ok_or(ConstructError {
                message: "Queue is empty".to_string(),
            })?;
            assert!(op_id >= 0 && op_id < self.next_condition_by_operator.len() as u32);

            let mut condition_iter = self.next_condition_by_operator[op_id as usize].iter();

            if condition_iter.len() == 0 {
                applicable_operators.push_back(op);
            } else {
                all_ops_immediate = false;
                let mut fact = condition_iter.next().ok_or(ConstructError {
                    message: "Condition iterator is empty".to_string(),
                })?;
                if fact.var() == branch_var_id {
                    while condition_iter.len() > 0 {
                        fact = condition_iter.next().ok_or(ConstructError {
                            message: "Condition iterator is empty".to_string(),
                        })?;
                    }
                    operators_for_value.push(op);
                } else {
                    default_operators.push_back(op);
                }
            }
        }

        todo!()
    }
}
trait Node {
    fn get_applicable_operators(&self) -> Option<&Vec<Operator>>;
}

struct BranchNode {
    children: Vec<Box<dyn Node>>,
}

impl BranchNode {
    pub fn new(children: Vec<Box<dyn Node>>) -> BranchNode {
        BranchNode { children }
    }

    pub fn add_child(&mut self, child: Box<dyn Node>) {
        self.children.push(child);
    }
}

impl Node for BranchNode {
    fn get_applicable_operators(&self) -> Option<&Vec<Operator>> {
        for child in &self.children {
            if let Some(operators) = child.get_applicable_operators() {
                return Some(operators);
            }
        }
        None
    }
}

struct LeafNode<'a> {
    applicable_operators: Option<&'a Vec<Operator>>,
}

impl<'a> LeafNode<'a> {
    pub fn new(applicable_operators: Option<&'a Vec<Operator>>) -> LeafNode<'a> {
        LeafNode {
            applicable_operators,
        }
    }
}

impl Node for LeafNode<'_> {
    fn get_applicable_operators(&self) -> Option<&Vec<Operator>> {
        self.applicable_operators
    }
}
