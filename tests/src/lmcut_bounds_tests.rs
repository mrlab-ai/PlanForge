//! LM-cut with numeric bounds prices a linear-cost operator at its cost
//! expression under the bounds of its variables before the operator.

use std::sync::Arc;

use planforge_sas::numeric_task::{
    AbstractNumericTask, NumericRootTask, NumericTaskExt, OperatorIndex,
};
use planforge_sas::state_registry::StateRegistry;
use planforge_search::evaluation::evaluator::EvaluationState;
use planforge_search::evaluation::heuristic::Heuristic;
use planforge_search::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::{
    LandmarkCutNumericHeuristic, LmCutNumericConfig,
};

use crate::corpus::{self, translate_in_memory};

fn load_dependent_cost_task() -> NumericRootTask {
    let dir = corpus::assets().join("lmcut-bounds/load-dependent-cost");
    translate_in_memory(&dir.join("domain.pddl"), &dir.join("problem.pddl"))
}

fn initial_h(task: &NumericRootTask, bound_iterations: usize) -> f64 {
    let heuristic = LandmarkCutNumericHeuristic::from_config(
        task,
        LmCutNumericConfig {
            bound_iterations,
            ..Default::default()
        },
    )
    .expect("the LM-cut configuration is supported");
    let mut registry = StateRegistry::for_task(Arc::new(task));
    let initial = registry.get_initial_state();
    let mut eval = EvaluationState::new(&initial, task as &dyn AbstractNumericTask, &registry);
    eval.set_is_goal(false);
    heuristic
        .compute_heuristic(&eval)
        .expect("the initial state evaluates")
}

/// `deliver` costs `1 + load`, and `load` only grows from 3, so its lower
/// bound before `deliver` is 3 and LM-cut prices `deliver` at 4 = h*.
#[test]
fn bounds_price_a_linear_cost_operator_at_the_lower_bound_of_its_variables() {
    let task = load_dependent_cost_task();
    let deliver = (0..task.get_operators().len())
        .map(OperatorIndex::from_usize)
        .find(|&op_id| {
            task.get_operators()[op_id.index()]
                .name()
                .starts_with("deliver")
        })
        .expect("the fixture has a deliver action");
    assert!(
        task.is_linear_cost_operator(deliver),
        "deliver must have a linear cost for this test to reach the bounds"
    );
    assert_eq!(initial_h(&task, 0), 4.0);
    assert_eq!(initial_h(&task, 1), 4.0);
}
