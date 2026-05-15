# planforge

A grounded numeric planner written in Rust. Accepts PDDL or pre-translated SAS+ input and produces a sequential plan when one exists within the configured resource budget.

## Status

Production-quality on the admissible search and heuristic paths (A\* with blind, lmcutnumeric, pattern databases, canonical and SCP-based domain abstractions). Greedy best-first search and an FF-style relaxed-plan heuristic with Metric-FF monotonic numeric relaxation are also available. Preferred-operator integration is still planned.

## Input formats

- **PDDL** — domain and problem files. The translator and preprocessor are invoked internally; intermediate SAS+ output is not exposed unless requested.
- **SAS+** — pre-translated tasks (one positional argument). Useful for benchmarking, where the translator is run once and the search is run repeatedly.

## Heuristics

- **Pattern databases** — projection-based goal-distance tables over selected variable subsets.
- **Domain abstractions** — CEGAR-built abstractions with comparison-axiom-aware refinement. Multiple combination strategies are available:
  - *Canonical* (max over compatible additive subsets).
  - *Saturated cost partitioning* (SCP), including a fill-SCP variant that combines per-label SCP with LM-cut over residual costs.
- **LM-cut** — numeric landmark-cut heuristic, usable standalone or as a residual-cost component inside SCP.
- **FF** — Hoffmann/Nebel relaxed-plan heuristic with Metric-FF style monotonic numeric relaxation. Each numeric variable tracks a `(max_reachable, min_reachable)` envelope through the relaxed planning graph; comparison-axiom facts become available when the envelope makes them satisfiable. Non-admissible in general; useful as a fast guide for greedy search and competitive with blind on small numeric instances.

## Search

- **A\*** — admissible best-first search (`f = g + h`). The production path for guaranteed-optimal planning under an admissible heuristic.
- **Greedy best-first search (GBFS)** — non-admissible best-first search (`f = h`). Often finds plans far faster than A\* with the same heuristic, at the cost of optimality.
- **FF-style preferred operators** — planned.

## Building

Stable Rust, no nightly features:

    cargo build --release

The primary binary is `target/release/planforge`. Smaller-scope binaries (`planforge-translator`, `planforge-preprocessor`, `planforge-searcher`) are built alongside it and are useful for staging.

## Running

Single-call PDDL pipeline:

    planforge --search 'astar(canonical_domain_abstractions(...))' \
              --max-time 30m --max-memory 8G \
              domain.pddl problem.pddl

Pre-translated SAS+:

    planforge --search 'astar(lmcutnumeric())' \
              --max-time 30m --max-memory 8G \
              task.sas

Common options:

- `--search SPEC` — search algorithm with a heuristic configuration. Examples:
  - `astar(blind())`
  - `astar(lmcutnumeric())`
  - `astar(canonical_domain_abstractions(...))`
  - `astar(fillSCP(...))`
  - `astar(ff())`
  - `gbfs(ff())` — fast non-admissible search
  - `gbfs(lmcutnumeric())`
- `--max-time DURATION` — wall-clock budget (`30m`, `1h`, `45s`).
- `--max-memory SIZE` — address-space cap (`8G`, `4096M`).

## Layout

Workspace crates:

- `planforge` — top-level entry point and CLI.
- `planforge-translator`, `planforge-preprocessor`, `planforge-searcher` — staged binaries for translator-only, preprocessor-only, and search-only invocations.
- `planforge-translate`, `planforge-preprocess`, `planforge-search`, `planforge-sas` — the corresponding libraries.
- `planforge-cli-utils` — shared CLI plumbing (exit codes, resource limits, allocator).
- `tests` — integration tests.

## Testing

    cargo test

Integration tests cover translator output, preprocessor invariants, state-registry deduplication, heuristic admissibility, and end-to-end planning on representative tasks.

## Resource limits

`--max-memory` sets the process's `RLIMIT_AS`. Heuristic construction additionally consults a polled RSS limit derived from `--max-memory` so the planner can stop adding abstractions cleanly before any external (slurm, cgroup) limit fires. Search-loop polling is not yet wired through; long A\* runs near the memory ceiling can still be killed by external supervisors.

## References

- Helmert, M. *The Fast Downward planning system*. JAIR 2006.
- Seipp, J. & Helmert, M. *Counterexample-guided cartesian abstraction refinement for classical planning*. JAIR 2018.
- Helmert, M., Haslum, P., Hoffmann, J., & Nissim, R. *Merge-and-shrink abstraction*. JACM 2014.
- Hoffmann, J. & Nebel, B. *The FF planning system*. JAIR 2001.

## License

Binary crates are licensed under GPLv3; library crates under LGPLv3; integration tests under GPLv3; lab files under MIT. See individual `Cargo.toml` files and `LICENSE` for details.
