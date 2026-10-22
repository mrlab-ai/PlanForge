# Custom search priority

This crate instantiates uniform-cost search with the `G` priority and a
`TieBreakingOpenList`. Implement `Priority` to supply another priority function;
the shared best-first loop keeps priority calculation statically dispatched.

From the repository root:

```sh
cargo run -p tutorial-custom-search
```
