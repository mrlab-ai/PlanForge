# Ship a Rust heuristic inside the PlanForge binary

This tutorial adds a native goal-count heuristic to PlanForge's features and
then hands control to the standard PlanForge CLI. The resulting executable keeps the normal search
spec parsing, option validation, portfolio, resource limits, re-exec wrapper,
logging, plan output, and exit codes. It adds two names: the `goalcount()`
heuristic and the `f_heap()` open list.

From the repository root, run it exactly like the stock binary:

```console
cargo run -p tutorial-plugin-binary -- --search 'astar(goalcount())' domain.pddl problem.pddl
```

Built-in heuristics remain available through the same executable, for example
`--search 'astar(blind())'`. The features are constructed once before CLI
parsing; heuristic evaluation uses the same native `dyn Heuristic` call that
the built-in search loop already uses.

`f_heap()` is a user-written open list. A registered open list is used through
`Box<dyn OpenList>` with any best-first search; it sees each entry's `g`, `h`,
state and insertion order. It orders like A*'s built-in `tiebreaking()`, so

```console
cargo run --release -p tutorial-plugin-binary -- --search 'astar(blind(), open_list=f_heap())' domain.pddl problem.pddl
```

expands the same states as `astar(blind())` and shows the cost of the dynamic
dispatch.
