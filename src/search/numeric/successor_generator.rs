use crate::search::numeric::numeric_task::{AbstractNumericTask, Fact};

struct GroundedSuccessorGenerator<'a> {
    task: &'a dyn AbstractNumericTask,
    conditions: Vec<Vec<&'a Fact>>,
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
        }

        GroundedSuccessorGenerator { task, conditions: vec![] }
    }
}