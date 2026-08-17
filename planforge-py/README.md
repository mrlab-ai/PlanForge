# PlanForge Python API

PlanForge exposes two deliberately different Python workflows.

In the prototyping workflow, Python owns the search loop and calls
`Task.applicable_operators`, `Task.apply` or `Task.apply_with_cost`, and
`Task.is_goal`. `State` is an owned snapshot: `values` contains one
finite-domain value per propositional variable and `numeric_values` contains
one value per numeric variable. Crossing the FFI boundary once or twice per
successor is acceptable here because this mode is for small instances and fast
iteration on an algorithm.

In the validation workflow, Rust owns A* or greedy best-first search and calls a
Python heuristic through `Task.search_with_heuristic`. This acquires the GIL and
copies a state snapshot for every evaluated state. It is useful for checking a
prototype against the production search implementation, but it is not a
benchmarking interface.

Pure Rust search has no Python callback or Python-specific branch in its
expansion loop. `Task.solve` and the module-level `solve` use that path.

For bulk analysis, `Task.enumerate_state_space` crosses the FFI boundary once.
Rust exhaustively enumerates and interns states, records CSR transitions, and
runs backward Dijkstra for exact `h_star`; Python receives the completed graph
as NumPy arrays. `max_states`, `max_transitions`, and `max_time` are mandatory.
If any bound is reached, `EnumerationError` names the bound and reached counts,
and no graph or h* array is returned. This is distinct from the Python-driven
prototyping loop above and is the appropriate interface for large state spaces.

`Task.operators()` exposes each operator's finite-domain preconditions,
conditional finite-domain effects, numeric assignment effects, and cost.
Finite-domain facts are `(variable, value)` pairs. An `Effect` assigns `value`
to `variable`; its `conditions` and optional `precondition_value` are additional
requirements for that effect. A `NumericEffect` names the affected numeric
variable, its source numeric variable, and one of `assign`, `increase`,
`decrease`, `scale_up`, or `scale_down`.

Run `examples/prototyping_search.py` for a complete Python-driven A* using an
`h_max` implementation written from this operator data.
