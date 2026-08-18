# Ship a Rust heuristic inside the PlanForge binary

This tutorial registers a native goal-count heuristic and then hands control to
the standard PlanForge CLI. The resulting executable keeps the normal search
spec parsing, option validation, portfolio, resource limits, re-exec wrapper,
logging, plan output, and exit codes. Its only extension is the additional
`goalcount()` heuristic name.

From the repository root, run it exactly like the stock binary:

```console
cargo run -p tutorial-plugin-binary -- --search 'astar(goalcount())' domain.pddl problem.pddl
```

Built-in heuristics remain available through the same executable, for example
`--search 'astar(blind())'`. The registry is constructed once before CLI
parsing; heuristic evaluation uses the same native `dyn Heuristic` call that
the built-in search loop already uses.
