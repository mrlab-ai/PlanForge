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
- **Cartesian abstractions** — numeric Cartesian CEGAR abstractions usable through
  the same canonical, label-SCP, and regional-SCP combinators. The separate
  `icaps26_cartesian(...)` source implements the full-task, first-flaw,
  desired-region split policies from Schindler, Speck, and Helmert (ICAPS 2026).
- **LM-cut** — numeric landmark-cut heuristic, usable standalone or as a residual-cost component inside SCP.
- **Posthoc optimization** — Pommerening/Röger/Helmert AAAI 2013 LP heuristic over a CEGAR-built domain-abstraction collection. The dual LP `max Σ h_i(s)·X_i s.t. Σ_{i : o relevant for i} X_i ≤ 1 for each positive-cost operator o` is solved per state by HiGHS. Dominates canonical (max-over-additive) but pays per-state LP cost; useful when the abstractions overlap heavily and a strict max underuses them.
- **FF** — Hoffmann/Nebel relaxed-plan heuristic with Metric-FF style monotonic numeric relaxation. Each numeric variable tracks a `(max_reachable, min_reachable)` envelope through the relaxed planning graph; comparison-axiom facts become available when the envelope makes them satisfiable. Non-admissible in general; useful as a fast guide for greedy search and competitive with blind on small numeric instances.

## Search

- **A\*** — admissible best-first search (`f = g + h`). The production path for guaranteed-optimal planning under an admissible heuristic.
- **Greedy best-first search (GBFS)** — non-admissible best-first search (`f = h`). Often finds plans far faster than A\* with the same heuristic, at the cost of optimality.
- **FF-style preferred operators** — planned.

## Building

Stable Rust, no nightly features:

    cargo build --release

The primary binary is `target/release/planforge`. Smaller-scope binaries (`planforge-translator`, `planforge-preprocessor`, `planforge-searcher`) are built alongside it and are useful for staging.

### HiGHS prerequisites

`planforge-search` depends on the [HiGHS](https://highs.dev) LP solver via the `highs` crate, which builds HiGHS from C++ source and runs `bindgen` over its C headers. The build therefore needs:

- a C++17 compiler (`g++` 11+)
- `cmake` 3.20+
- a working `libclang` (set `LIBCLANG_PATH` to its directory if it is not on the default loader path)

On the cluster nodes used during development, `LIBCLANG_PATH` pointed at the Clang module's `lib/` directory and an `LD_LIBRARY_PATH` entry shadowed the missing `libtinfo.so.5` with the system's `libtinfo.so.6`.

## Running

Single-call PDDL pipeline:

    planforge --search 'astar(canonical_domain_abstractions(...))' \
              --max-time 30m --max-memory 8G \
              domain.pddl problem.pddl

Hierarchical abstraction sources make collection generation and combination
independent. For example:

    planforge --restrict-task --search \
      'astar(canonical(cartesian_collection(max_states=1000,max_collection_size=100000),construction_max_time=900))' \
      domain.pddl problem.pddl

    planforge --restrict-task --search \
      'astar(scp(domain(max_abstraction_size=1000,max_collection_size=100000),online=false,partitioning=region,construction_max_time=900))' \
      domain.pddl problem.pddl

    planforge --restrict-task --search \
      'astar(canonical(icaps26_cartesian(pick=min_unwanted,max_time=900)))' \
      domain.pddl problem.pddl

`icaps26_cartesian` accepts `pick=random|min_unwanted|max_unwanted` and requires
an integer restricted SNP task. `construction_max_time` is one shared budget
for source generation and offline SCP table construction.

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
  - `astar(posthoc_optimization(...))` — LP-based dominator of canonical; `pho(...)` is accepted as an alias
  - `astar(ff())`
  - `gbfs(ff())` — fast non-admissible search
  - `gbfs(lmcutnumeric())`
- `--max-time DURATION` — wall-clock budget (`30m`, `1h`, `45s`).
- `--max-memory SIZE` — address-space cap (`8G`, `4096M`).
- `--restrict-task` — convert an SNP task to its restricted representation;
  already restricted tasks are retained unchanged.
- `--compact-numeric-states` — intern exact canonical `f64` values behind
  compact integer IDs in the search state registry.

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

On Linux, `--max-memory` is enforced against resident memory by the parent
process. A larger `RLIMIT_AS` remains as an emergency ceiling because mimalloc
reserves address space ahead of committed pages. Heuristic construction and
search also release their fixed memory padding as the resident limit is
approached, leaving room for a controlled planner exit before an external
Slurm or cgroup limit fires.

## License

Binary crates are licensed under GPLv3; library crates under LGPLv3; integration tests under GPLv3; lab files under MIT. See individual `Cargo.toml` files and `LICENSE` for details.
