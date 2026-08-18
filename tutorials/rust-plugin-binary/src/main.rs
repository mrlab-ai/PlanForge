use planforge_sas::numeric_task::{AbstractNumericTask, TaskRef};
use planforge_search::config::HeuristicSpec;
use planforge_search::evaluation::{EvaluationError, EvaluationState, Heuristic};
use planforge_search::heuristic_factory::{
    ExternalHeuristic, HeuristicBuildError, RequiredBackend, TaskRequirements,
};

struct GoalCountHeuristic;

impl Heuristic for GoalCountHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let task = eval_state.task();
        let registry = eval_state.state_registry();
        let state = eval_state.state();
        let mut unsatisfied = 0usize;
        for goal_id in 0..task.get_num_goals() {
            if !task.get_goal_fact(goal_id).is_hold(registry.view(state)) {
                unsatisfied += 1;
            }
        }
        Ok(unsatisfied as f64)
    }

    fn heuristic_name(&self) -> &str {
        "goalcount"
    }
}

fn any_task(_: &HeuristicSpec) -> Result<TaskRequirements, String> {
    Ok(TaskRequirements::ANY)
}

fn no_nested_heuristics(_: &HeuristicSpec) -> Result<Vec<HeuristicSpec>, String> {
    Ok(Vec::new())
}

fn build_goal_count<'a>(
    spec: &HeuristicSpec,
    _: &'a dyn AbstractNumericTask,
    _: TaskRef<'a>,
) -> Result<Option<Box<dyn Heuristic + 'a>>, HeuristicBuildError> {
    if !spec.args.is_empty() {
        return Err("`goalcount` does not accept arguments".to_string().into());
    }
    Ok(Some(Box::new(GoalCountHeuristic)))
}

fn main() -> std::io::Result<()> {
    planforge::run_with_heuristics(vec![ExternalHeuristic {
        name: "goalcount",
        backend: RequiredBackend::None,
        requirements: any_task,
        nested_heuristics: no_nested_heuristics,
        build: build_goal_count,
    }])
}
