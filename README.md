# Work in Progress Numeric Fast Downward Replacement in Rust

Right now, only a parser, applicable for regular SAS+ files exist. 

## Usage
```
cargo run <sas file>
```

The final goal is to create a library that can be extended with heuristics, task transformations, and search algorithms using either python or rust. 

## Planned Features

* All public functions and methods will be exposed to python
* supports classical and numeric planning
* Supports SAS and PDDL
* Adds a lifted and grounded planner
* Adds various functions required for learning, among others: 
  * Sample state spaces
  * Candle compatible 
  * Serialization of states for saving and loading from disk
  * A PDDL2SAS translator written in rust