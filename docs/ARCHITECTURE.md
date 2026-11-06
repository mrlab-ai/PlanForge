# Architecture

PlanForge turns a PDDL task into a plan in three stages: translation, the
task, and search. Each stage is a crate, and dependencies point one way only,
so a crate cannot reach into a stage after it.

```
planforge-config-derive
          \
planforge-sas  <-  planforge-translate
      ^
      |
planforge-search   (search core: engine, Heuristic trait, registry)
      ^
      |
planforge-heuristics   (abstractions, landmarks, relaxation, potentials, combinations)
      ^          \
      |           planforge-cplex (optional, LP backend)
      |
planforge   (binaries, CLI, limits; parses with planforge_heuristics::features())
```

## The task (`planforge-sas`)

A [`Task`](../planforge-sas/src/numeric_task/task_api.rs) is plain, immutable
data: propositional and numeric variables, operators and their costs,
propositional, comparison and arithmetic axioms, the numeric conditions, the
translator's mutex groups, the initial state closed under the axioms, and the
goal. `Task::new` is the one constructor and checks every invariant;
consumers can rely on them without checking again.

Everything but the goal sits behind one `Arc`, so a task is cheap to clone
and `task.with_goals(..)` gives a task with another goal that shares every
operator. A transformation that changes a task's structure is a function
returning a new `Task`, plus a mapping back where ids change:

- the restriction (`--transform restrict`) folds derived arithmetic into the
  comparisons and the metric;
- the projection onto a pattern builds a pattern database's task.

Both live in `planforge-search/src/task_transformations`.

States are stored packed in the [`StateRegistry`](../planforge-sas/src/state_registry.rs);
a `ConcreteStateView` reads them, and `StateValues` lets lookups read single
propositional values without decoding a whole state.

## The search core (`planforge-search`)

- `search`: the building blocks of a search algorithm -- `SearchSpace` (the
  state registry, successor generation, the search nodes and plan
  extraction) and `SearchProgress` (limits, statistics, progress reports) --,
  best-first search on them (A*, greedy, lazy and multi-path variants) over a
  `NodeOrdering`, and the `--search` configuration language. Best-first
  search expands a node in three phases: it generates the successors and
  drops the duplicates, evaluates the rest in one loop, and queues them in
  generation order. A `NodeOrdering` turns g and h into a key (A*:
  `(f, h)`, greedy: `h`); the built-in open lists (`tiebreaking`,
  `preferred_first`) are generic over it and statically dispatched, a
  registered one is a `Box<dyn OpenList>`. Open-list entries are 24 bytes.
- `evaluation`: the `Heuristic` trait, `EvaluationState`, blind search, and
  shared helpers (random walks, seeds, a per-state value cache).
- `registry`: `Features`, the value that holds one registry per kind of
  feature the configuration language names, and parses configuration
  strings. Each feature is a typed configuration (derived with
  `#[derive(ApplyOptions)]`) plus what it does: a `HeuristicFeature`
  (`heuristic_features`) builds a heuristic, says what it needs of the build
  (`preflight`, such as an LP solver, checked before translation) and of the
  task (`TaskRequirements`: transformations by name and `TaskCheck`s the
  transformed task must pass, such as
  `planforge_heuristics::requirements::RESTRICTED`), a `SearchFeature` (A*
  and greedy best-first search built in) runs a search, a
  `TransformationFeature` (`restrict` built in) transforms the task before
  it. `Features::builtin()` holds the built-in features; a program adds the
  others to its value, and nested configurations, such as the heuristic
  inside `astar(...)`, are parsed with the same value.
- `successor_generator`: the `SuccessorGenerator` trait (which operators
  are applicable in a state; applying them stays with the `StateRegistry`),
  and the standard grounded tree that is the default.
- `state_space` (the exact state space), `graph` (the
  backward goal-distance search shared with the abstractions),
  `task_transformations`, `resource_limits`.

## Evaluation, caching and batching

Best-first search evaluates states in one place per phase:
`BestFirstSearch::expand` generates the successors of a node and detects
duplicates, then evaluates the remaining successors in one plain loop over
one call, `BestFirstSearch::evaluate`, then queues them in generation order.
Two rules keep that loop the only place that decides when the primary
heuristic of an expansion runs:

- **Open lists never evaluate.** An `OpenList` orders `OpenEntry`s (g, h,
  state, insertion order) whose estimates the search computed before pushing
  them.
- **The engine keeps no heuristic cache.** A heuristic that wants to reuse
  estimates owns that cache (`&mut self`, as `numeric_potential` does). The
  engine records the last estimate of a state only for `mpd`, to decide
  whether to evaluate it again, never instead of evaluating it.

`Task`, `StateRegistry` and `EvaluationState` are `Sync`, which
`planforge-search/src/evaluation/evaluator.rs` asserts at compile time, so
the states of one expansion can be evaluated in parallel without copying.

### Adding batched evaluation

Batched evaluation, for example a neural network over all successors of an
expansion, touches three places and nothing else:

1. the `Heuristic` trait: a default method
   `evaluate_batch(&mut self, states: &[EvaluationState], estimates: &mut Vec<Estimate>)`
   that calls `evaluate` per state, so every heuristic keeps working
   unchanged;
2. the evaluation phase of `BestFirstSearch::expand`: build the
   `EvaluationState`s of the successors (`SearchSpace::evaluation_state`),
   call `evaluate_batch` once, and hand each estimate to
   `BestFirstSearch::record_evaluation`, which checks, bounds, counts and
   records it exactly as for a single evaluation;
3. the batched heuristic, which overrides `evaluate_batch`.

Preferred operators stay per state: an open list that uses them
(`preferred_first`) keeps the single-state call.

## The heuristics (`planforge-heuristics`)

One module per technique, each depending on the core only:

- `abstractions`: Cartesian and domain abstractions refined by CEGAR (shared
  machinery in `common/cegar`), pattern databases, goal-focused and diverse
  collections, and their cost partitionings (maximum, canonical, saturated,
  post-hoc optimization);
- `landmarks`: numeric LM-cut;
- `relaxation`: the numeric FF heuristic;
- `potentials`: numeric potential heuristics, with the `cplex` feature;
- `combinations`: the maximum over arbitrary heuristics and the admissibility
  check around one.

`planforge_heuristics::features()` is the built-in features with the
crate's heuristics and transformations added; `planforge` parses its
command line with it.

## Extending PlanForge

| To add | Implement | Register | Example |
|---|---|---|---|
| a heuristic | `Heuristic`, and a `HeuristicFeature` for its configuration | one entry in `planforge_heuristics::features()`, or `features.heuristics.add(..)` before `planforge::run_with(features)` in your own binary | `tutorials/rust-goal-count`, `tutorials/rust-plugin-binary` |
| a search algorithm | a `SearchFeature` whose search composes `SearchSpace` and `SearchProgress` (for best-first search, a `NodeOrdering`) | `features.search_algorithms.add(..)` | `tutorials/rust-custom-search` |
| an open list | `OpenList` over `OpenEntry` (g, h, state, insertion order), and an `OpenListFeature` | `features.open_lists.add(..)`; a best-first search names it with `open_list=...` | `tutorials/rust-plugin-binary` (`f_heap`) |
| a node ordering | `NodeOrdering` (a key from g and h) | a `SearchFeature` that runs `BestFirstSearch<YourOrdering, _>` | `search::best_first::AStar` |
| a successor generator | `SuccessorGenerator` (which operators are applicable), and a `SuccessorGeneratorFeature` | `features.successor_generators.add(..)`; a search names it with `successor_generator=...` | the `successor_generator` module documentation, `successor_generator::StandardGroundedTree` |
| a task transformation | a `TransformationFeature` | one entry in `planforge_heuristics::features()`, or `features.transformations.add(..)` | `task_transformations::restriction::RestrictFeature` |
| what a heuristic needs | a `TaskCheck` (of the task) or `HeuristicFeature::preflight` (of the build) | returned from the heuristic's `HeuristicFeature` | `planforge_heuristics::requirements` |
| what a search needs | `TaskRequirements` from `SearchFeature::requirements` (besides its heuristics') | the search's `SearchFeature` | `keep_unimportant_variables` for a search over whole states |

A feature that also reads the lifted task, such as a learned policy over
lifted state features, finds the PDDL files of a translated task in
`Task::pddl_files` (none for a task read from a SAS file or derived from
another) and asks for `TaskRequirements::keep_unimportant_variables` if it
observes facts the goal does not depend on.

A heuristic owns its state (`&mut self`) and everything it reads, such as a
shared `Arc<Task>`, so it is `Send` and outlives the context that built it. It
returns an `Estimate`: a non-negative cost or `Estimate::DEAD_END`. It reports
invalid internal states by panicking, never by returning a default.

## Errors

Each area of the libraries has one error type to match on; the binaries map
them to exit codes.

| Area | Error |
|---|---|
| loading a SAS+ task | `planforge_sas::numeric_task::TaskLoadError` |
| translating PDDL | `planforge_translate::TranslateError` (with `ParseError`, `GroundingLimitError`) |
| configuration strings | `planforge_search::config::ConfigError` |
| task transformations | `planforge_search::task_transformations::TransformError` |
| building a heuristic | `planforge_search::heuristic_factory::HeuristicBuildError` |
| evaluating a state | `planforge_search::evaluation::EvaluationError` |
| running a search | `planforge_search::search::SearchError` |

A heuristic that reaches a state it cannot handle because of a bug panics
instead of returning one of these.

## Verifying a change

Before a commit, with the crates listed explicitly (`--workspace` would pull
in `planforge-cplex`, which needs the CPLEX headers):

```sh
CRATES="-p planforge-sas -p planforge-translate -p planforge-search -p planforge-heuristics -p planforge -p planforge-config-derive -p tests"
cargo fmt --all --check
cargo clippy $CRATES --all-targets -- -D warnings
cargo test $CRATES
cargo test -p tests --release -- --ignored search_snapshot
```

A refactoring must not change search behaviour. The `search_snapshot` test
runs the main heuristic families on the test instances and compares plan
costs and expanded and generated states with `tests/snapshots/search.txt`;
identical counts are the requirement. Behaviour changes are separate commits
that rewrite the snapshot (`PLANFORGE_BLESS=1`), so its diff shows what they
moved. Run times are compared only when a change claims to be faster or must
not be slower: a release build of the change against one of its base
revision, alternating runs, medians, on tasks whose search reaches the
changed code.
