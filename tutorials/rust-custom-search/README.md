# Custom search algorithm

This crate adds enforced hill-climbing to the `--search` grammar as
`ehc(heuristic)`: a `SearchFeature` registered in `SEARCH_ALGORITHMS` whose
search is composed of the public building blocks of `planforge_search::search`:

- `SearchSpace` registers states, generates successors, detects duplicates on
  the search nodes and traces the plan back;
- `SearchProgress` checks the time and memory limits, keeps the statistics and
  reports progress.

Best-first search (`astar`, `gbfs`) is built on the same blocks. A binary that
ships the standard CLI registers its algorithms with `planforge::run_with`.

From the repository root:

```sh
cargo run -p tutorial-custom-search
```
