# Coupling review

This is a source-level coupling measurement at `review-fixes` commit `5c1282f`, before the Stage 4 decoupling changes. Test-only modules are excluded from the import graph because they do not constrain algorithm extensions. Counts include production `.rs` files under each crate's `src/`; an edge is one distinct `use crate::`, `use super::`, or `use self::` dependency between source modules. Module-declaration containment is not an edge. Cycles are reported as strongly connected components (SCCs), rather than attempting to enumerate every simple path through an SCC.

## Summary

| Crate | Modules | Import edges | Cyclic SCCs |
|---|---:|---:|---:|
| `planforge-sas` | 21 | 48 | 1 |
| `planforge-translate` | 38 | 113 | 1 |
| `planforge-search` | 102 | 288 | 2 |
| `planforge-config-derive` | 1 | 0 | 0 |
| `planforge-cplex` | 2 | 0 | 0 |
| `planforge-searcher` | 3 | 1 | 0 |
| `planforge-sgd` | 11 | 13 | 1 |
| `planforge` | 7 | 1 | 0 |
| `planforge-py` | 1 | 0 | 0 |
| `tutorial-goal-count` | 1 | 0 | 0 |
| `tutorial-custom-search` | 1 | 0 | 0 |
| `tests` | 12 | 13 | 0 |

Modules imported by more than ten others:

- `planforge-sas::numeric_task`: 13.
- `planforge-translate::pddl::conditions`: 13; `pddl::pddl_types`: 11.
- `planforge-search::evaluation::domain_abstractions::domain_abstraction_factory`: 17; `evaluation::heuristic`: 13; `evaluation::evaluator`: 13; `evaluation::domain_abstractions::utils`: 13; `evaluation`: 11.
- No module in the other crates exceeds ten incoming imports.

The SCCs are:

- `planforge-sas`: `axioms`, `numeric_conditions`, `numeric_parser`, `numeric_task`, `sas_format`, `state_registry`, `utils::interval`, `utils::linear_effects`, and `utils::state_packer`. The known direct edge is still present: `numeric_task` imports `numeric_conditions`, while `numeric_conditions` imports `numeric_task`. Extracting `value_types` reduced what must be imported but did not break the cycle because the unused `NumericConditions::from_task` still requires `AbstractNumericTask`.
- `planforge-translate`: `preprocess` and `preprocess::causal_graph`. This is a structural parent/child cycle (`preprocess` re-exports the child, the child imports parent data), not a broad translator mesh.
- `planforge-search`: one 12-module SCC spanning the domain-abstraction factory, abstraction/operator types, CEGAR flaw search, numeric context, and domain utilities; and one two-module SCC between `numeric_landmarks::lm_cut_numeric_heuristic` and `numeric_landmarks::numeric_lm_cut_landmarks`.
- `planforge-sgd`: `residuals`, `tensor`, and `transcription` form one SCC.
- All remaining crates are acyclic under this measure.

The full per-module degrees are in the appendix.

## God-types and threaded context

`StateRegistry` now occurs as a function or closure parameter 22 times in production code (excluding struct fields, return types, comments, and tests), confirming the earlier 67 to 22 reduction. Most remaining occurrences are concentrated in numeric-potential/PDB construction and the `ConcreteState` decoding API; ordinary heuristic calls receive it through `EvaluationState`.

`NumericRootTaskParts` has 13 required fields and is constructed at 177 sites in 46 files. Only 16 of those sites in nine files are production code; 161 are fixtures. It is a construction DTO, not a runtime context threaded through algorithms. Replacing it with setters or a builder would hide required task data and add fixture boilerplate, so the raw count overstates runtime coupling.

`AbstractNumericTask` has 46 methods, four implementations (`&T`, `NumericRootTask`, `SingleGoalTask`, and `ProjectedTask`), and is referenced by 64 production files in `planforge-search`. It is broad, but it is read-only and is the deliberate task boundary. Splitting it would add several bounds/type parameters to the same algorithms and is not justified without an extension that needs a genuinely smaller interface.

`EvaluationState` is the argument to all 15 production `Heuristic` implementations (20 including tests, Python, and the tutorial), and explicit `&EvaluationState` parameters occur 35 times in 20 files. The type has four fields: state, task, registry, and goal status. This common evaluation interface is useful, but its constructor still takes two dead parameters (`g_value` and `is_preferred`) and stores `task` as an `Option` even though all 12 constructors provide it. Those remnants create avoidable call-site coupling.

## Public mutable configuration and representation

| Type | Public fields | Mutation/read findings |
|---|---:|---|
| `ScpOnlineConfig` | 25 of 25 | Post-option normalization is now in two production modules: deadline caps are assigned in `heuristic_factory.rs`, while collection normalization is implemented in `config.rs` and invoked from both the factory and `fill_scp.rs`. Tests still mutate fields directly. |
| `DomainAbstractionCollectionGeneratorMultipleCegarConfig` | 23 of 23 | Post-option changes occur in `heuristic_factory/abstraction_config.rs` and SCP's `config.rs` (time caps, footprint mode, and full-goal normalization). |
| `DomainAbstractionFactory` | 4 of 7 | The four representation fields are public despite same-named read accessors. Across production plus tests there are 45 direct field reads versus 40 accessor calls; production alone has 29 direct versus 27 accessor calls. `cegar.rs` is the production mutator and takes all four as separate mutable references during a split. |

`DomainAbstractionFactory` is the clearest fix: callers currently mix two access styles, and refinement can violate the relationship among mapping, domain sizes, partitions, and numeric sizes between individual writes. A factory-owned split operation can make the four fields private and keep mutation atomic without adding a context object or a dynamic call.

For the option structs, public fields are partly their intended declarative surface and are consumed widely. The valuable change is to centralize the few post-parse transformations in named methods; generating 48 trivial getters/setters would add code without reducing algorithmic coupling.

## Concrete change impact

### Change how abstract states are hashed

Three production files define the representation and must agree:

1. `planforge-search/src/evaluation/domain_abstractions/domain_abstraction_generator.rs` computes/stores mixed-radix multipliers.
2. `planforge-search/src/evaluation/domain_abstractions/domain_abstraction_factory/state_encoding.rs` encodes, decodes, and enumerates hash IDs.
3. `planforge-search/src/evaluation/domain_abstractions/domain_abstraction_heuristic.rs` hashes concrete/projected states on the lookup path.

The focused tests are in `domain_abstraction_factory/tests.rs`, `domain_abstraction_heuristic/tests.rs`, and `abstraction_collections/canonical_heuristic/tests.rs`: six files total for a representation change. Consumers such as SCP use abstract IDs through component methods and do not need edits. This is adequately localized.

### Add a new saturator

A saturator expressible as a cap sequence needs `saturated_cost_partitioning_online_heuristic/config.rs` (enum, spelling, and sequence), `saturated_cost_partitioning_online_heuristic/mod.rs` (construction behavior if the existing sequence protocol is insufficient), `domain_abstraction_factory/saturation.rs` (new saturation calculation), and `saturated_cost_partitioning_online_heuristic/tests.rs`: three production files and one test file. `fill_scp.rs` and the heuristic factory already pass the enum generically. This is adequately localized and uses static/concrete calls on the construction path.

### Change the open-list discipline

The core discipline spans `search/open_list.rs` (entry ordering and queues), `search/policy.rs` (algorithm policy), and `search/engine.rs` (insertion/pop use): three production files, with tests embedded in `open_list.rs` and `search/tests.rs`. Exposing a new user-visible search name additionally touches `search/registry.rs` and `planforge-searcher/src/recursive_config.rs`. The hot driver is generic over `SearchAlgorithm`, so this extension does not add a per-node `dyn` call. This is a good separation.

### Change numeric-effect semantics

This is the least localized scenario. Twenty-three production files match on `AssignmentOperation` or parse its syntax:

- Syntax/data/concrete semantics: `planforge-translate/src/preprocess/sas_parts.rs`; `planforge-sas/src/numeric_parser.rs`; `numeric_task/value_types.rs`; `numeric_task/task_api.rs`; `sas_format.rs`; `state_registry.rs`; `utils/interval.rs`; `utils/linear_effects.rs`.
- Search abstractions and analyses: `causal_graph.rs`; `task_restriction.rs`; `evaluation/abstraction_task.rs`; `evaluation/cegar.rs`; `evaluation/ff_heuristic.rs`; `evaluation/cartesian_abstractions/mod.rs`; `evaluation/domain_abstractions/domain_abstraction.rs`; `evaluation/domain_abstractions/additive_numeric_views.rs`; `evaluation/domain_abstractions/abstract_operator_generator.rs`; `evaluation/domain_abstractions/domain_abstraction_factory/footprints.rs`; `evaluation/numeric_landmarks/numeric_bound.rs`; `evaluation/numeric_landmarks/numeric_helper.rs`; `evaluation/numeric_potentials/task.rs`; `evaluation/pattern_databases/numeric_size_estimator.rs`; and `evaluation/pattern_databases/max_additive_subsets.rs`.

Not every semantic change requires all 23 edits, but a new operation does because Rust's exhaustive matches force each concrete, interval, relaxation, projection, relevance, and admissibility interpretation to decide what it means. That is real semantic coupling rather than merely misplaced code. Centralizing the concrete arithmetic in `AssignmentOperation::apply` and interval arithmetic in `Interval` is already useful; forcing every abstraction through one generic or dynamic interpreter would either lose precision or add a per-node call. Also, 16 `PARITY(numeric-fd)` markers currently exist, 15 under `numeric_landmarks/` (one more there than the contextual estimate), so scattering or hiding that algorithm is not warranted.

## Crate layering and evaluation structure

`planforge-sas` has no workspace dependency at all; it depends only on `hashbrown` and `tracing`. `planforge-translate` and `planforge-search` both depend downward on SAS, and searcher/CLI/Python depend upward on those libraries. SAS therefore does not depend on search, translation, bindings, binaries, or solvers. The crate layering is sound.

`planforge-search/src/evaluation/` contains 83 production modules: 12 abstraction-collection, eight Cartesian, 27 domain-abstraction, five numeric-landmark, nine numeric-potential, 14 PDB, and eight root modules. Most families form a DAG around `Heuristic`/`EvaluationState`. It is not a crate-wide mesh, but the 12-module domain-abstraction SCC is a genuine mesh: factory representation is imported by 17 modules and domain utilities by 13. The numeric-landmark two-module cycle is small and parity-sensitive. Highest-value work should narrow the domain-factory mutation boundary and break the cheap SAS cycle; moving more files alone would not help.

## Fix ranking

1. Break the direct `numeric_task` to `numeric_conditions` back-edge by deleting the unused trait-based constructor and importing the extracted value types directly.
2. Make `DomainAbstractionFactory`'s four representation fields private and move the coordinated split mutation behind the factory API.
3. Remove `EvaluationState`'s dead constructor parameters and impossible optional task.
4. Add named post-parse normalization methods for SCP/collection deadline and footprint changes if doing so removes the remaining external assignments without proliferating accessors.

Do not replace `NumericRootTaskParts` or `EvaluationState` with a larger context, split `AbstractNumericTask` speculatively, introduce dynamic dispatch in the open-list/node path, or rearrange parity-critical numeric-landmark code merely to reduce graph counts.

## Appendix: module in-degree and out-degree

### `planforge-sas`

| Module | In | Out |
|---|---:|---:|
| `axioms` | 9 | 3 |
| `default_value_axioms` | 0 | 3 |
| `lib` | 0 | 0 |
| `numeric_conditions` | 3 | 3 |
| `numeric_parser` | 2 | 4 |
| `numeric_task` | 13 | 7 |
| `numeric_task::task_api` | 0 | 2 |
| `numeric_task::value_types` | 0 | 1 |
| `plan_verification` | 0 | 4 |
| `sas_format` | 2 | 2 |
| `sas_writer` | 0 | 3 |
| `state_registry` | 2 | 6 |
| `utils` | 3 | 0 |
| `utils::errors` | 4 | 0 |
| `utils::float_tolerance` | 3 | 1 |
| `utils::hashing` | 0 | 0 |
| `utils::interval` | 1 | 2 |
| `utils::linear_effects` | 1 | 3 |
| `utils::scc` | 1 | 1 |
| `utils::segmented_vector` | 1 | 1 |
| `utils::state_packer` | 3 | 2 |

### `planforge-translate`

| Module | In | Out |
|---|---:|---:|
| `api` | 0 | 4 |
| `axiom_rules` | 1 | 4 |
| `build_model` | 1 | 3 |
| `constraints` | 1 | 1 |
| `fact_groups` | 1 | 5 |
| `greedy_join` | 0 | 1 |
| `instantiate` | 0 | 10 |
| `invariant_finder` | 1 | 7 |
| `invariants` | 1 | 6 |
| `lib` | 0 | 0 |
| `normalize` | 3 | 6 |
| `numeric_axiom_rules` | 1 | 2 |
| `options` | 5 | 0 |
| `pddl` | 2 | 0 |
| `pddl::actions` | 7 | 5 |
| `pddl::axioms` | 7 | 4 |
| `pddl::conditions` | 13 | 1 |
| `pddl::effects` | 4 | 4 |
| `pddl::f_expression` | 10 | 1 |
| `pddl::functions` | 2 | 1 |
| `pddl::pddl_types` | 11 | 0 |
| `pddl::predicates` | 2 | 1 |
| `pddl::tasks` | 7 | 8 |
| `pddl_parser` | 2 | 0 |
| `pddl_parser::lisp_parser` | 2 | 0 |
| `pddl_parser::parsing_functions` | 1 | 11 |
| `pddl_parser::pddl_file` | 0 | 3 |
| `pddl_to_prolog` | 4 | 4 |
| `preprocess` | 2 | 2 |
| `preprocess::causal_graph` | 1 | 3 |
| `preprocess::max_dag` | 1 | 0 |
| `preprocess::sas_parts` | 0 | 2 |
| `sas_tasks` | 6 | 0 |
| `simplify` | 1 | 1 |
| `split_rules` | 0 | 1 |
| `symbols` | 4 | 0 |
| `tools` | 9 | 0 |
| `translate` | 0 | 12 |

### `planforge-search`

| Module | In | Out |
|---|---:|---:|
| `causal_graph` | 2 | 1 |
| `config` | 5 | 0 |
| `config::parser` | 0 | 1 |
| `evaluation` | 11 | 0 |
| `evaluation::abstraction_collections` | 3 | 0 |
| `evaluation::abstraction_collections::canonical_heuristic` | 1 | 5 |
| `evaluation::abstraction_collections::component` | 5 | 7 |
| `evaluation::abstraction_collections::cost_partitioning` | 7 | 1 |
| `evaluation::abstraction_collections::explicit_scp` | 0 | 2 |
| `evaluation::abstraction_collections::max_heuristic` | 1 | 3 |
| `evaluation::abstraction_collections::portfolio` | 4 | 1 |
| `evaluation::abstraction_collections::region` | 1 | 0 |
| `evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic` | 3 | 12 |
| `evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::config` | 0 | 4 |
| `evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::diagnostics` | 0 | 1 |
| `evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::fill_scp` | 0 | 1 |
| `evaluation::abstraction_task` | 4 | 0 |
| `evaluation::cartesian_abstractions` | 10 | 10 |
| `evaluation::cartesian_abstractions::finalize` | 0 | 1 |
| `evaluation::cartesian_abstractions::flaw_splits` | 0 | 1 |
| `evaluation::cartesian_abstractions::icaps26` | 1 | 0 |
| `evaluation::cartesian_abstractions::plan_replay` | 0 | 1 |
| `evaluation::cartesian_abstractions::shortest_paths` | 0 | 1 |
| `evaluation::cartesian_abstractions::split_generation` | 0 | 1 |
| `evaluation::cartesian_abstractions::split_selector` | 0 | 1 |
| `evaluation::cegar` | 5 | 1 |
| `evaluation::check_admissible` | 1 | 4 |
| `evaluation::domain_abstractions` | 5 | 0 |
| `evaluation::domain_abstractions::abstract_operator_generator` | 9 | 6 |
| `evaluation::domain_abstractions::additive_numeric_views` | 9 | 1 |
| `evaluation::domain_abstractions::cegar` | 5 | 9 |
| `evaluation::domain_abstractions::cegar::flaw_search` | 8 | 11 |
| `evaluation::domain_abstractions::cegar::flaw_search::flaw_selection` | 1 | 2 |
| `evaluation::domain_abstractions::cegar::flaw_search::progression` | 2 | 6 |
| `evaluation::domain_abstractions::cegar::flaw_search::regression` | 2 | 6 |
| `evaluation::domain_abstractions::cegar::flaw_search::sequence` | 1 | 8 |
| `evaluation::domain_abstractions::cegar::flaw_search::state` | 4 | 3 |
| `evaluation::domain_abstractions::cegar::flaw_search::target_centered` | 1 | 3 |
| `evaluation::domain_abstractions::domain_abstraction` | 10 | 1 |
| `evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar` | 3 | 9 |
| `evaluation::domain_abstractions::domain_abstraction_factory` | 17 | 7 |
| `evaluation::domain_abstractions::domain_abstraction_factory::distances` | 0 | 1 |
| `evaluation::domain_abstractions::domain_abstraction_factory::footprints` | 0 | 1 |
| `evaluation::domain_abstractions::domain_abstraction_factory::plan_extraction` | 0 | 1 |
| `evaluation::domain_abstractions::domain_abstraction_factory::saturation` | 0 | 1 |
| `evaluation::domain_abstractions::domain_abstraction_factory::state_encoding` | 0 | 1 |
| `evaluation::domain_abstractions::domain_abstraction_factory::transition_system` | 0 | 1 |
| `evaluation::domain_abstractions::domain_abstraction_generator` | 7 | 5 |
| `evaluation::domain_abstractions::domain_abstraction_heuristic` | 5 | 5 |
| `evaluation::domain_abstractions::numeric_context` | 2 | 2 |
| `evaluation::domain_abstractions::posthoc_optimization_heuristic` | 1 | 5 |
| `evaluation::domain_abstractions::utils` | 13 | 6 |
| `evaluation::domain_abstractions::utils::debug_dump` | 0 | 1 |
| `evaluation::domain_abstractions::utils::partitioning` | 0 | 0 |
| `evaluation::evaluator` | 13 | 0 |
| `evaluation::ff_heuristic` | 0 | 2 |
| `evaluation::heuristic` | 13 | 1 |
| `evaluation::maximal_cliques` | 2 | 0 |
| `evaluation::numeric_landmarks` | 0 | 0 |
| `evaluation::numeric_landmarks::lm_cut_numeric_heuristic` | 5 | 5 |
| `evaluation::numeric_landmarks::numeric_bound` | 1 | 1 |
| `evaluation::numeric_landmarks::numeric_helper` | 3 | 0 |
| `evaluation::numeric_landmarks::numeric_lm_cut_landmarks` | 2 | 4 |
| `evaluation::numeric_potentials` | 8 | 0 |
| `evaluation::numeric_potentials::config` | 0 | 1 |
| `evaluation::numeric_potentials::function` | 0 | 1 |
| `evaluation::numeric_potentials::heuristic` | 0 | 4 |
| `evaluation::numeric_potentials::ocp` | 0 | 5 |
| `evaluation::numeric_potentials::optimizer` | 1 | 1 |
| `evaluation::numeric_potentials::rays` | 1 | 2 |
| `evaluation::numeric_potentials::sampling` | 1 | 2 |
| `evaluation::numeric_potentials::task` | 0 | 3 |
| `evaluation::pattern_databases` | 3 | 1 |
| `evaluation::pattern_databases::canonical_pdb_heuristic` | 1 | 8 |
| `evaluation::pattern_databases::max_additive_subsets` | 2 | 3 |
| `evaluation::pattern_databases::numeric_size_estimator` | 1 | 1 |
| `evaluation::pattern_databases::pattern_collection` | 5 | 1 |
| `evaluation::pattern_databases::pattern_collection_information` | 0 | 3 |
| `evaluation::pattern_databases::pattern_database` | 8 | 4 |
| `evaluation::pattern_databases::pattern_generator_greedy` | 1 | 4 |
| `evaluation::pattern_databases::pattern_generator_systematic` | 3 | 3 |
| `evaluation::pattern_databases::pdb_collection` | 4 | 5 |
| `evaluation::pattern_databases::pdb_heuristic` | 1 | 8 |
| `evaluation::pattern_databases::projected_task` | 8 | 2 |
| `evaluation::pattern_databases::utils` | 2 | 2 |
| `evaluation::pattern_databases::variable_order_finder` | 1 | 1 |
| `evaluation::state_value_cache` | 5 | 0 |
| `heuristic_factory` | 1 | 19 |
| `heuristic_factory::abstraction_config` | 0 | 12 |
| `lib` | 0 | 0 |
| `resource_limits` | 1 | 0 |
| `search` | 6 | 0 |
| `search::config` | 1 | 0 |
| `search::engine` | 0 | 9 |
| `search::open_list` | 1 | 1 |
| `search::policy` | 1 | 2 |
| `search::registry` | 0 | 2 |
| `search::space` | 1 | 1 |
| `search::stats` | 1 | 0 |
| `state_space` | 0 | 2 |
| `successor_generator` | 6 | 0 |
| `task_restriction` | 5 | 0 |

### `planforge-config-derive`

| Module | In | Out |
|---|---:|---:|
| `lib` | 0 | 0 |

### `planforge-cplex`

| Module | In | Out |
|---|---:|---:|
| `ffi` | 0 | 0 |
| `lib` | 0 | 0 |

### `planforge-searcher`

| Module | In | Out |
|---|---:|---:|
| `lib` | 0 | 0 |
| `recursive_config` | 1 | 0 |
| `sgd` | 0 | 1 |

### `planforge-sgd`

| Module | In | Out |
|---|---:|---:|
| `adam` | 1 | 0 |
| `classical` | 1 | 0 |
| `config` | 1 | 0 |
| `controller` | 1 | 0 |
| `engine` | 0 | 5 |
| `exactness` | 0 | 3 |
| `lib` | 0 | 0 |
| `residuals` | 2 | 1 |
| `tensor` | 2 | 1 |
| `testing` | 1 | 0 |
| `transcription` | 4 | 3 |

### `planforge`

| Module | In | Out |
|---|---:|---:|
| `allocator` | 0 | 0 |
| `bin::planforge-translator` | 0 | 0 |
| `lib` | 0 | 0 |
| `limits` | 0 | 1 |
| `main` | 0 | 0 |
| `output` | 1 | 0 |
| `portfolio` | 0 | 0 |

### `planforge-py`

| Module | In | Out |
|---|---:|---:|
| `lib` | 0 | 0 |

### `tutorial-goal-count`

| Module | In | Out |
|---|---:|---:|
| `main` | 0 | 0 |

### `tutorial-custom-search`

| Module | In | Out |
|---|---:|---:|
| `main` | 0 | 0 |

### `tests`

| Module | In | Out |
|---|---:|---:|
| `corpus` | 10 | 0 |
| `derived_predicate_tests` | 0 | 2 |
| `determinism_tests` | 0 | 1 |
| `goal_census` | 0 | 2 |
| `lib` | 0 | 0 |
| `numeric_condition_tests` | 0 | 2 |
| `numeric_corpus_tests` | 3 | 1 |
| `sailing_simple_tests` | 0 | 1 |
| `sgd_engine_tests` | 0 | 1 |
| `sgd_transcription_tests` | 0 | 1 |
| `strips_corpus_tests` | 0 | 1 |
| `task_equivalence_tests` | 0 | 1 |
