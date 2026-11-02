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
planforge   (binaries, CLI, limits; registers the heuristics)
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
- `registry`: the registries of the features the configuration language
  names. Each feature is a typed configuration (derived with
  `#[derive(ApplyOptions)]`) plus what it does: a `HeuristicFeature`
  (`heuristic_factory::HEURISTICS`) builds a heuristic, says what it needs
  of the build (`preflight`, such as an LP solver, checked before
  translation) and of the task (`TaskRequirements`: transformations by name
  and `TaskCheck`s the transformed task must pass, such as
  `planforge_heuristics::requirements::RESTRICTED`), a `SearchFeature`
  (`search::SEARCH_ALGORITHMS`, with A* and greedy best-first search built in)
  runs a search, a `TransformationFeature`
  (`task_transformations::TRANSFORMATIONS`, with `restrict` built in)
  transforms the task before it. A binary registers the other features once,
  before it parses a configuration.
- `successor_generator`: the `SuccessorGenerator` trait (which operators
  are applicable in a state; applying them stays with the `StateRegistry`),
  the standard grounded tree that is the default, and their registry
  `SUCCESSOR_GENERATORS`.
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
   `evaluate_batch(&mut self, states: &[EvaluationState], estimates: &mut Vec<f64>)`
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

`heuristics()` and `transformations()` list the features of the crate;
`planforge` registers them.

## Extending PlanForge

| To add | Implement | Register | Example |
|---|---|---|---|
| a heuristic | `Heuristic`, and a `HeuristicFeature` for its configuration | one entry in `planforge_heuristics::heuristics()`, or `planforge::Extensions::heuristics` with `planforge::run_with` from your own binary | `tutorials/rust-goal-count`, `tutorials/rust-plugin-binary` |
| a search algorithm | a `SearchFeature` whose search composes `SearchSpace` and `SearchProgress` (for best-first search, a `NodeOrdering`) | `planforge::Extensions::search_algorithms` | `tutorials/rust-custom-search` |
| an open list | `OpenList` over `OpenEntry` (g, h, state, insertion order), and an `OpenListFeature` | `planforge::Extensions::open_lists`; a best-first search names it with `open_list=...` | `tutorials/rust-plugin-binary` (`f_heap`) |
| a node ordering | `NodeOrdering` (a key from g and h) | a `SearchFeature` that runs `BestFirstSearch<YourOrdering, _>` | `search::best_first::AStar` |
| a successor generator | `SuccessorGenerator` (which operators are applicable), and a `SuccessorGeneratorFeature` | `planforge::Extensions::successor_generators`; a search names it with `successor_generator=...` | the `successor_generator` module documentation, `successor_generator::StandardGroundedTree` |
| a task transformation | a `TransformationFeature` | one entry in `planforge_heuristics::transformations()`, or `planforge::Extensions::transformations` | `task_transformations::restriction::RestrictFeature` |
| what a heuristic needs | a `TaskCheck` (of the task) or `HeuristicFeature::preflight` (of the build) | returned from the heuristic's `HeuristicFeature` | `planforge_heuristics::requirements` |

A heuristic owns its state (`&mut self`), reads the task through `&Task` or a
shared `Arc<Task>`, and reports invalid internal states by panicking, never by
returning a default.

## Verifying a change

A refactoring must not change search behaviour. The `search_snapshot` test
(`tools/gate.sh semantic`) runs the main heuristic families on the test
instances and compares plan costs and expanded and generated states with
`tests/snapshots/search.txt`; identical counts are the requirement, run times
are only checked for regressions (`tools/gate.sh timing`). Behaviour changes
are separate commits that rewrite the snapshot (`PLANFORGE_BLESS=1`), so its
diff shows what they moved.
