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
- **Numeric potentials** — admissible CPLEX-backed numeric and propositional
  potential functions with initial-state, all-states, sampled, and diverse
  portfolio objectives. The implementation also supports reachable bounds,
  goal-conditioned functions and cost partitions, exact dead-end rays,
  online enrichment with `mpd=true`, duality validation, and joint
  potential/domain-abstraction OCP.
- **Posthoc optimization** — Pommerening/Röger/Helmert AAAI 2013 LP heuristic over a CEGAR-built domain-abstraction collection. The dual LP `max Σ h_i(s)·X_i s.t. Σ_{i : o relevant for i} X_i ≤ 1 for each positive-cost operator o` is solved per state by CPLEX. The model remains resident and only its objective changes between states. It dominates canonical (max-over-additive) but pays per-state LP cost; useful when the abstractions overlap heavily and a strict max underuses them.
- **FF** — Hoffmann/Nebel relaxed-plan heuristic with Metric-FF style monotonic numeric relaxation. Each numeric variable tracks a `(max_reachable, min_reachable)` envelope through the relaxed planning graph; comparison-axiom facts become available when the envelope makes them satisfiable. Non-admissible in general; useful as a fast guide for greedy search and competitive with blind on small numeric instances.

## Search

- **A\*** — admissible best-first search (`f = g + h`). The production path for guaranteed-optimal planning under an admissible heuristic.
- **Greedy best-first search (GBFS)** — non-admissible best-first search (`f = h`). Often finds plans far faster than A\* with the same heuristic, at the cost of optimality.
- **FF-style preferred operators** — planned.

## Building

Stable Rust, no nightly features:

    cargo build --release

The primary binary is `target/release/planforge`. Smaller-scope binaries (`planforge-translator`, `planforge-searcher`) are built alongside it and are useful for staging.

### CPLEX prerequisites

LP-backed heuristics use the native IBM ILOG CPLEX 22.2 C API. Build them with:

    CPLEX_ROOT=/path/to/CPLEX_Studio/cplex cargo build --release --features cplex

`CPLEX_ROOT` must contain `include/ilcplex/cplex.h` and
`lib/x86-64_linux/static_pic/libcplex.a`. PlanForge links the provided
position-independent static library, uses one solver thread, and checks at
heuristic startup that the active license accepts and solves a 1001-column
model. Community Edition and any other size-restricted license are rejected.
A build without `cplex` can run non-LP heuristics, but requesting an LP-backed
heuristic is an explicit configuration error.

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
`--restrict-task`. It reproduces the artifact semantics on integer restricted
tasks; the dedicated affine restriction and continuous-split extensions cover
the full supported SNP benchmark set without changing other heuristics.
`construction_max_time` is one shared budget
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
  - `astar(numeric_potential(opt=initial_state,max_potential=1e8))`
  - `astar(numeric_potential(opt=diverse_samples,num_heuristics=4,num_samples=100,max_potential=1e8,cache_estimates=true,invalidate_online_cache_on_growth=true),mpd=true)`
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
- `planforge-translator`, `planforge-searcher` — staged binaries for translator-only and search-only invocations.
- `planforge-translate`, `planforge-search`, `planforge-sas` — the corresponding libraries.
- `planforge-cli-utils` — shared CLI plumbing (exit codes, resource limits, allocator).
- `planforge-cplex` — small checked native CPLEX ownership and sparse-LP layer.
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
