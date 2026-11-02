# Architecture

PlanForge turns a PDDL task into a plan in three stages: translation, the
task, and search. Each stage is a crate, and dependencies point one way only,
so a crate cannot reach into a stage after it.

```
planforge-config-derive        planforge-cplex (optional, LP backend)
          \                          |
planforge-sas  <-  planforge-translate
      ^
      |
planforge-search   (search core: engine, Heuristic trait, registry)
      ^
      |
planforge-heuristics   (abstractions, landmarks, relaxation, potentials, combinations)
      ^
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
  generation order.
- `evaluation`: the `Heuristic` trait, `EvaluationState`, blind search, and
  shared helpers (random walks, seeds, a per-state value cache).
- `registry`: the registries of the features the configuration language
  names. Each feature is a typed configuration (derived with
  `#[derive(ApplyOptions)]`) plus what it does: a `HeuristicFeature`
  (`heuristic_factory::HEURISTICS`) builds a heuristic and names the task
  properties and backend it needs, a `SearchFeature`
  (`search::SEARCH_ALGORITHMS`, with A* and greedy best-first search built in)
  runs a search, a `TransformationFeature`
  (`task_transformations::TRANSFORMATIONS`, with `restrict` built in)
  transforms the task before it. A binary registers the other features once,
  before it parses a configuration.
- `successor_generator`: the `SuccessorGenerator` trait (which operators
  are applicable in a state; applying them stays with the `StateRegistry`),
  the decision tree that is the default, and their registry
  `SUCCESSOR_GENERATORS`.
- `state_space` (the exact state space), `graph` (the
  backward goal-distance search shared with the abstractions),
  `task_transformations`, `resource_limits`.

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
| a successor generator | `SuccessorGenerator` (which operators are applicable), and a `SuccessorGeneratorFeature` | `planforge::Extensions::successor_generators`; a search names it with `successor_generator=...` | `successor_generator::DecisionTree` |
| a task transformation | a `TransformationFeature` | one entry in `planforge_heuristics::transformations()`, or `planforge::Extensions::transformations` | `task_transformations::restriction::RestrictFeature` |

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
