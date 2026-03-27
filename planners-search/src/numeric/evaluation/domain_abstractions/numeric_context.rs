use planners_sas::numeric::axioms::CalOperator;
use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericType};

use super::comparison_expression::{ArithOp, Interval};

fn arith_op_from_axiom(op: &CalOperator) -> ArithOp {
    match op {
        CalOperator::Sum => ArithOp::Add,
        CalOperator::Difference => ArithOp::Sub,
        CalOperator::Product => ArithOp::Mul,
        CalOperator::Division => ArithOp::Div,
    }
}

pub fn propagate_assignment_axiom_intervals(
    task: &dyn AbstractNumericTask,
    numeric_intervals: &mut [Interval],
) {
    let max_iterations = task.assignment_axioms().len().saturating_add(1).max(1);
    for _ in 0..max_iterations {
        let mut changed = false;
        for axiom in task.assignment_axioms() {
            let Ok(affected_var_id) = usize::try_from(axiom.get_affected_var_id()) else {
                continue;
            };
            let Ok(left_var_id) = usize::try_from(axiom.get_left_var_id()) else {
                continue;
            };
            let Ok(right_var_id) = usize::try_from(axiom.get_right_var_id()) else {
                continue;
            };
            if affected_var_id >= numeric_intervals.len()
                || left_var_id >= numeric_intervals.len()
                || right_var_id >= numeric_intervals.len()
            {
                continue;
            }

            let next = arith_op_from_axiom(axiom.get_operator()).apply_interval(
                numeric_intervals[left_var_id],
                numeric_intervals[right_var_id],
            );
            if numeric_intervals[affected_var_id] != next {
                numeric_intervals[affected_var_id] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

pub fn seed_numeric_intervals_from_initial_state(task: &dyn AbstractNumericTask) -> Vec<Interval> {
    let initial_numeric_values = task.get_initial_numeric_state_values();
    let mut numeric_intervals: Vec<Interval> =
        vec![Interval::unbounded(); task.numeric_variables().len()];
    for (i, v) in task.numeric_variables().iter().enumerate() {
        if v.get_type() == &NumericType::Constant {
            numeric_intervals[i] = Interval::singleton(initial_numeric_values[i]);
        }
    }
    numeric_intervals
}
