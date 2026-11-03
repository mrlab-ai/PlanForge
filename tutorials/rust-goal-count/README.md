# Goal-count heuristic in Rust

This tutorial is for a heuristic evaluated on every search node. It implements
PlanForge's native `Heuristic` interface, so the expansion path remains in Rust
and does not cross the Python FFI boundary.

The essential implementation is:

```rust
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
```

`EvaluationState` holds the task, the registry that decodes the state, and
whether the state is a goal; `Estimate` is a non-negative cost or
`Estimate::DEAD_END`. The executable evaluates the initial state and
runs the regular generic A* driver on two committed SAS tasks.

From the repository root:

```console
cargo run -p tutorial-goal-count
```
