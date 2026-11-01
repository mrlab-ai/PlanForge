use planforge_search::evaluation::{EvaluationError, EvaluationState, Heuristic};
use planforge_search::heuristic_factory::{
    BuildContext, EmptyConfig, HeuristicBuildError, HeuristicFeature, entry,
};

struct GoalCountHeuristic;

impl Heuristic for GoalCountHeuristic {
    fn evaluate(&mut self, eval_state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        let state = eval_state.state_registry().view(eval_state.state());
        let unsatisfied = eval_state
            .task()
            .goals()
            .iter()
            .filter(|goal| !goal.is_held(state))
            .count();
        Ok(unsatisfied as f64)
    }

    fn name(&self) -> &str {
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
    planforge::run_with(planforge::Extensions {
        heuristics: vec![entry::<GoalCountFeature>()],
        ..Default::default()
    })
}
