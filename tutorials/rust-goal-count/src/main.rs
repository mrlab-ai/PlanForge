use std::sync::Arc;
use std::time::Duration;

use planforge_sas::numeric_task::Task;
use planforge_sas::state_registry::StateRegistry;
use planforge_search::evaluation::{Estimate, EvaluationError, EvaluationState, Heuristic};
use planforge_search::search::best_first::{AStarSearch, SearchParams};
use planforge_search::search::{SearchLimits, SearchOutcome};

/// Goal-count heuristic: number of goal facts not satisfied in the state.
struct GoalCountHeuristic;

impl Heuristic for GoalCountHeuristic {
    fn evaluate(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<Estimate, EvaluationError> {
        let state = eval_state.state_registry().view(eval_state.state());
        let unsatisfied = eval_state
            .task()
            .goals()
            .iter()
            .filter(|goal| !goal.is_held(state))
            .count();
        Ok(Estimate::new(unsatisfied as f64))
    }

    fn name(&self) -> &str {
        "goal_count"
    }
}

fn run(path: &str) {
    let task: Arc<Task> = Arc::new(Task::from_file(path));
    let mut heuristic = GoalCountHeuristic;
    let mut registry = StateRegistry::for_task(task.clone());
    let initial_state = registry.register_initial_state();
    let initial_h = heuristic
        .evaluate(&EvaluationState::new(&initial_state, &task, &registry))
        .expect("goal-count should evaluate the initial state");

    let params = SearchParams {
        limits: SearchLimits {
            time: Some(Duration::from_secs(5)),
            memory_bytes: None,
        },
        ..SearchParams::new(Box::new(heuristic))
    };
    let result = AStarSearch::new(task, params).run().expect("search failed");
    println!("instance: {path}");
    println!("  h(initial):  {initial_h}");
    match &result.outcome {
        SearchOutcome::Solved { plan, cost } => {
            println!("  plan length: {}", plan.len());
            println!("  plan cost:   {cost}");
        }
        outcome => println!("  outcome:     {outcome:?}"),
    }
    println!("  expanded:    {}", result.statistics.nodes.expanded);
    println!("  evaluated:   {}", result.statistics.nodes.evaluated);
    println!();
}

fn main() {
    // A* with the goal-count heuristic on a couple of committed SAS instances.
    for path in [
        "tests/assets/numeric_sas/example2.sas",
        "tests/assets/numeric_sas/example5.sas",
    ] {
        run(path);
    }
}
