use crate::search::numeric::numeric_task::{AbstractNumericTask, Fact};

struct SuccessorGenerator<'a> {
    task: &'a dyn AbstractNumericTask,
    conditions: Vec<Vec<Fact>>,
}

impl SuccessorGenerator<'_> {
    pub fn new(task: &dyn AbstractNumericTask) -> SuccessorGenerator {

        let operators = task.get_operators();

        for operator in operators.iter() {
            for precondition in operator.preconditions.iter() {
                if !task.get_fact_name(precondition).is_empty() {
                    // Initialize conditions with the preconditions of the operator
                    let condition = vec![precondition.clone()];
                    // Add the condition to the conditions vector
                    self.conditions.push(condition);
                }
            }
        }

        SuccessorGenerator { task, conditions: vec![] }
    }
}