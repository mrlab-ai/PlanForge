# Goal-count heuristic in Rust

This tutorial is for a heuristic evaluated on every search node. It implements
PlanForge's native `Heuristic` interface, so the expansion path remains in Rust
and does not cross the Python FFI boundary.

The essential implementation is:

```rust
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
            if !task
                .get_goal_fact(goal_id)
                .is_hold(registry.view(state))
            {
                unsatisfied += 1;
            }
        }
        Ok(unsatisfied as f64)
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}
```

`EvaluationState` guarantees that the task and decoding registry are present;
there are no optional lookups. The executable evaluates the initial state and
runs the regular generic A* driver on two committed SAS tasks.

From the repository root:

```console
cargo run -p tutorial-goal-count
```
