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

A [`Task`](planforge-sas/src/numeric_task/task_api.rs) is plain, immutable
data: propositional and numeric variables, operators and their costs,
propositional, comparison and arithmetic axioms, the numeric conditions, the
translator's mutex groups, the initial state closed under the axioms, and the
goal. `Task::new` is the one constructor and checks every invariant;
consumers can rely on them without checking again.

Everything but the goal sits behind one `Arc`, so a task is cheap to clone
and `task.with_goals(..)` gives a task with another goal that shares every
operator. A transformation that changes a task's structure is a function
returning a new `Task`, plus a mapping back where ids change:

- the restriction (`--restrict-task`) folds derived arithmetic into the
  comparisons and the metric;
- the projection onto a pattern builds a pattern database's task.

Both live in `planforge-search/src/task_transformations`.

States are stored packed in the [`StateRegistry`](planforge-sas/src/state_registry.rs);
a `ConcreteStateView` reads them, and `StateValues` lets lookups read single
propositional values without decoding a whole state.

## The search core (`planforge-search`)

- `search`: best-first search (A*, greedy, lazy and multi-path variants) over
  a `NodeOrdering`, and the `--search` configuration language.
- `evaluation`: the `Heuristic` trait, `EvaluationState`, blind search, and
  shared helpers (random walks, seeds, a per-state value cache).
- `heuristic_factory`: the registry. A heuristic the configuration language
  names is a `HeuristicFeature`: a typed configuration (derived with
  `#[derive(ApplyOptions)]`), the task properties and backend it needs, and
  how to build it. The core knows only blind search; a binary registers every
  other heuristic once with `register_heuristics`.
- `successor_generator`, `state_space` (the exact state space), `graph` (the
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

`heuristics()` lists every feature of the crate; `planforge` registers it.

## Extending PlanForge

| To add | Implement | Register | Example |
|---|---|---|---|
| a heuristic | `Heuristic`, and a `HeuristicFeature` for its configuration | one entry in `planforge_heuristics::heuristics()`, or `planforge::run_with_heuristics` from your own binary | `tutorials/rust-goal-count`, `tutorials/rust-plugin-binary` |
| a node ordering | `search::best_first::NodeOrdering` | none: instantiate best-first search with it | `tutorials/rust-custom-search` |
| a task transformation | a function `&Task -> Task` (plus an id mapping if ids change) | none | `task_transformations::projection` |

A heuristic owns its state (`&mut self`), reads the task through `&Task` or a
shared `Arc<Task>`, and reports invalid internal states by panicking, never by
returning a default.

## Verifying a change

A refactoring must not change search behaviour. The gate compares plan costs
and expansion counts on the test instances for the main heuristic families,
before and after; identical counts are the requirement, run times are only
checked for regressions. Behaviour changes are separate commits with their
own measurements.
