# Custom search algorithm

This crate defines and registers uniform-cost search outside PlanForge's search
crate. It implements only `SearchAlgorithm::priority`, inheriting the generic
best-first loop. The registry erases the concrete driver once per run; priority
evaluation remains monomorphized inside the expansion loop.

From the repository root:

```sh
cargo run -p tutorial-custom-search
```
