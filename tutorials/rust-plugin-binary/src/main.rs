use planforge_search::evaluation::{EvaluationError, EvaluationState, Heuristic};
use planforge_search::heuristic_factory::{
    BuildContext, EmptyConfig, HeuristicBuildError, HeuristicFeature, entry,
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
            if !task.get_goal_fact(goal_id).is_held(registry.view(state)) {
                unsatisfied += 1;
            }
        }
        Ok(unsatisfied as f64)
    }

    fn heuristic_name(&self) -> &str {
        "goalcount"
    }
}

struct GoalCountFeature;
impl HeuristicFeature for GoalCountFeature {
    const NAME: &'static str = "goalcount";
    type Config = EmptyConfig;
    fn build<'a>(
        _: &Self::Config,
        _: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        Ok(Box::new(GoalCountHeuristic))
    }
}
fn main() -> std::io::Result<()> {
    planforge::run_with_heuristics(vec![entry::<GoalCountFeature>()])
}
