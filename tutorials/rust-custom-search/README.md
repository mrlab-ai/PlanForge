# Custom search algorithm

This crate adds uniform-cost search to the `--search` grammar as `ucs()`: a
`SearchFeature` that instantiates best-first search with the `UniformCost`
ordering and a `TieBreakingOpenList`, registered in `SEARCH_ALGORITHMS`.
Implement `NodeOrdering` to supply another priority function; the shared
best-first loop keeps it statically dispatched. A binary that ships the
standard CLI registers its algorithms with `planforge::run_with`.

From the repository root:

```sh
cargo run -p tutorial-custom-search
```
