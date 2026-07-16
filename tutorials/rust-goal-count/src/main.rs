use std::sync::Arc;
use std::time::Duration;

use planforge_sas::numeric_task::{NumericRootTask, TaskRef};
use planforge_sas::state_registry::StateRegistry;
use planforge_search::evaluation::{EvaluationError, EvaluationState, Heuristic};
use planforge_search::search::{AStarSearch, SearchEngine, SearchStatus};

/// Goal-count heuristic: number of goal facts not satisfied in the state.
struct GoalCountHeuristic {
    name: String,
}

impl Heuristic for GoalCountHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let task = eval_state.task().expect("goal-count needs the task");
        let registry = eval_state
            .state_registry()
            .expect("goal-count needs the registry");
        let state = eval_state.state();
        let mut unsatisfied = 0usize;
        for i in 0..task.get_num_goals() {
            if !task.get_goal_fact(i).is_hold(state, registry) {
                unsatisfied += 1;
            }
        }
        Ok(unsatisfied as f64)
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
}

fn run(path: &str) {
    let task: TaskRef<'static> = Arc::new(NumericRootTask::from_file(path));
    let mut registry = StateRegistry::for_task(task.clone());
    let initial_state = registry.get_initial_state();
    let heuristic = GoalCountHeuristic {
        name: "goal_count".to_string(),
    };
    let initial_eval =
        EvaluationState::new_with_registry(&initial_state, 0.0, false, &*task, &registry);
    let initial_h = heuristic
        .compute_heuristic(&initial_eval)
        .expect("goal-count should evaluate the initial state");
    let heuristic = Box::new(heuristic);
    let mut search = AStarSearch::new(
        task.clone(),
        registry,
        Some(heuristic),
        Some(Duration::from_secs(5)),
        None,
    );
    let result = search.search();
    let status = match &result.status {
        SearchStatus::Solved(_) => "solved",
        SearchStatus::Failed => "unsolvable",
        SearchStatus::Timeout => "timeout",
        SearchStatus::MemoryLimitReached => "memory_limit",
        SearchStatus::InProgress => "in_progress",
    };
    println!("instance: {path}");
    println!("  h(initial):  {initial_h}");
    println!("  status:      {status}");
    println!("  plan length: {:?}", result.plan.as_ref().map(|p| p.len()));
    println!("  plan cost:   {:?}", result.solution_cost);
    println!("  expanded:    {}", result.nodes_expanded);
    println!("  evaluated:   {}", result.nodes_evaluated);
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
