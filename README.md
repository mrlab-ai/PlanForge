# Numeric PlanneRS

<pre>
                ....... ...:.
          .......@*#@...+#=#@..
        ...@@%::-======-@##*@.
    ....+@---=====:---====%#@-----........
    ..@=*+========%@@*=====:@--.....----..
    @*******=======+*=======:@..:----------.....
  ..@**%@#**================-:@---------------..
  ..@*******===============-==%@*------------:..
  . @*#*****===================:@---------.........
  ...*@******==============-====:@+---.........:++:..
  ...*@@@@@*=================:====%@%......=+++++++:..
  ....%@@@@@@@++++*+++=======-======:@@++++++++++++..
    ...-----#*++++++++=============-===::=@++++++++...
    ..:----:@+++++++=======================:@#+=......
    ..:.....@....@+=======+....%-........:%%=:@....*##
      .....#.  ..@==+===@.....%-............@=%%#####..
    . ..-++*.  ...:*+%=%.   ..@@@@@@@@%+.....#=*%####.
    ..++++++@...    .#@...   ..%======@#==@....-=@+... .
    ..++++++@..*.@  ......-. ..%====@==-==%....===@
    ..:++++++. .%@#..  ..@+....##****##*......@===@.*.
    ..++++++   .@==....#=+..  ..............#===+@#*.
      .+++++   .@#*+@:#*=+..     .@@@%....@====*+@#.
      ..++++.  .@#@**+****..      @**=@.. ..=+++*%..
      ...:...  .@%:*@@#**#..      %@++**.. ..@++@.
          ..@....@@#@@#@**#........%*@%+++@....-@..
      ..:@@@#%##@@@@@+%*#**@@@#*%%#%***##++++@@@@@..
        ......+*%@@@@@@@@@@@@@@@@@@@@@@@@@#*-.. ..
                ..   .....    ..  .        ........

</pre>

[![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)](https://www.rust-lang.org/)

A high-performance automated planning library written in Rust, designed as a modern replacement for Fast Downward with enhanced support for numeric planning.

## 🚀 Features

### Current Implementation

- ✅ **SAS+ Parser**: Complete parser for SAS+ files (classical and numeric)
- ✅ **State Registry**: Efficient state management with deduplication
- ✅ **Axiom Evaluation**: Support for propositional and arithmetic axioms
- ✅ **Successor Generation**: Grounded successor generation with operator applicability
- ✅ **Per-State Information**: Generic storage for associating data with states
- ✅ **Numeric Planning**: Full support for numeric variables and operations

### 🔧 Architecture

- **Modular Design**: Clean separation between parsing, search, and utilities
- **Memory Efficient**: Uses segmented vectors and smart caching
- **Type Safe**: Leverages Rust's type system for compile-time guarantees
- **Zero-Copy**: Minimal data copying with efficient reference management

## 🚧 Usage

### Command Line Interface

```bash
# Parse and analyze a SAS+ file
cargo run --bin planners path/to/problem.sas

# Run with debug information
cargo run --bin planners --log-level=debug path/to/problem.sas

# Run tests
cargo test

# Run specific test with output
cargo test test_name -- --nocapture
```

### Profiling

Enable debug function names in the `release` profile of Cargo.toml by adding the following line

```toml
debug = "line-tables-only"
```

**Run all command-lines to profile with the `--internal-run` argument**.

```bash
# Profile with flamegraph (requires cargo-flamegraph: cargo install flamegraph or the devenv profile)
cargo flamegraph --bin planners -- <sas file here>

# This will generate a flamegraph.svg file showing performance hotspots
# Open flamegraph.svg in a web browser to view the interactive flame graph

# Alternatively, you use samply to get interactive analysis (already in the devenv profile).
# The following command will run the profiling and open it in Firefox Profiler in the
# default browser with a local server opened
samply record command

# Or you can save it in a file with the following command
samply record --save-only -o file.json -- command

# And load it with the following one (unfortunately loading the json file
# directly in Firefox Profiler does not get the symbol names).
samply load file.json


# Alternatively, you can use callgrind (but much slower):
valgrind --tool=callgrind --callgrind-out-file=file.out command

# And then open the output file with kcachegrind
kcachegrind file.out
```

### Library Usage (Planned)

```rust
use numeric_planners::{parse_sas, StateRegistry, SearchAlgorithm};

// Parse a planning problem
let task = parse_sas("problem.sas")?;

// Set up state management
let mut registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);
let initial_state = registry.get_initial_state();

// Run planning algorithm
let solution = SearchAlgorithm::new().solve(&task, initial_state)?;
```

## 🎯 Roadmap

### Phase 1: Core Infrastructure ✅

- [x] SAS+ parsing with numeric extensions
- [x] State representation and management
- [x] Basic successor generation
- [x] Axiom evaluation system

### Phase 2: Search Algorithms 🔄

- [x] A* search implementation
- [ ] Greedy best-first search
- [ ] Lazy search with deferred evaluation
- [ ] Multi-threaded search algorithms

### Phase 3: Heuristics 📋

- [ ] Landmark-based heuristics
- [x] Pattern database heuristics
- [ ] Numeric planning heuristics (h^max, h^add)
- [ ] Learning-based heuristics

### Phase 4: Advanced Features 📋

- [ ] PDDL parsing support
- [ ] Python bindings via PyO3
- [ ] Task transformations and preprocessing
- [ ] Lifted planning support
- [ ] Goal estimation and sampling

### Phase 5: ML Integration 📋

- [ ] Candle tensor library integration
- [ ] State space sampling for learning
- [ ] Serialization for training data
- [ ] Neural network heuristic integration

## 🔬 Technical Details

### Performance Optimizations

- **Segmented Vectors**: Memory-efficient storage for large datasets
- **State Packing**: Compact representation using bit manipulation
- **Caching**: Smart caching of frequently accessed data
- **Zero-Allocation**: Minimal heap allocations in hot paths

### Numeric Planning Support

- **Variable Types**: Regular, constant, derived, and cost variables
- **Operations**: Addition, subtraction, multiplication, division
- **Axioms**: Arithmetic and comparison axioms
- **Assignment Effects**: Complex numeric state updates

## 🧪 Testing

The project includes comprehensive tests covering:

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture
```

### Test Coverage

- Parser validation with real SAS+ files
- State registry operations and deduplication
- Successor generation with numeric effects
- Axiom evaluation correctness
- Error handling and edge cases

## 🤝 Contributing

We welcome contributions!
<!-- Please see our [contributing guidelines](CONTRIBUTING.md) for details. -->

### Development Setup

1. Install Rust (stable): <https://rustup.rs/>
2. Clone the repository: `git clone https://github.com/mrlab-ai/numeric_planneRS.git`
3. Run tests: `cargo test`
4. Build: `cargo build --release`

### Code Style

- Follow standard Rust formatting: `cargo fmt`
- Run clippy for lints: `cargo clippy`
- Ensure tests pass: `cargo test`

## 📚 References

- [Fast Downward](https://www.fast-downward.org/) - Original planning system
- [PDDL](https://planning.wiki/ref/pddl) - Planning Domain Definition Language (TODO: Is there a reference for numeric planning?)

## 📄 License

Binary crates in this project are licensed under GPLv3:

- [planners](./planners)
- [planners-translator](./planners-translator)
- [planners-preprocessor](./planners-preprocessor)
- [planners-searcher](./planners-searcher)

Library crates in this project are licensed under LGPLv3:

- [planners-cli-utils](./planners-cli-utils)
- [planners-sas](./planners-sas)
- [planners-translate](./planners-translate)
- [planners-preprocess](./planners-preprocess)
- [planners-search](./planners-search)

Integration tests ([tests](./tests)) are licensed under GPLv3.

Lab files ([lab-files](lab-files)) are licensed under MIT license.

## 🏆 Acknowledgments

- Fast Downward team for the original architecture and inspiration
- Rust community for excellent tooling and documentation
- Planning research community for algorithmic foundations
