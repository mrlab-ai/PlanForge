# Python to Rust Translation Status — Fast Downward Numeric PDDL Translator

## Goal
Port the Fast Downward `python/translate/` pipeline to Rust for numeric planning with perfect semantic parity. The final product should produce identical SAS+ files and require no Python runtime.

## Python Translation Pipeline Overview

### Main Flow (from `python/translate/translate.py`)
1. **Parse PDDL** (`pddl_parser.open()`) → Parse domain and problem files
2. **Add Global Constraints** (`task.add_global_constraints()`) → Handle global constraints
3. **Normalize** (`normalize.normalize()`) → Normalize task representation
4. **PDDL to SAS Conversion** (`pddl_to_sas()`) → Main translation pipeline:
   - **Instantiate** (`instantiate.explore()`) → Ground actions and collect reachable facts/fluents
   - **Build invariants** (`invariant_finder` + `fact_groups`) → Find invariants and group facts
   - **Build dictionaries** (`strips_to_sas_dictionary()`) → Map STRIPS to SAS variables
   - **Translate task** (`translate_task()`) → Convert to SAS format
   - **Simplify** (`simplify.filter_unreachable_propositions()`) → Remove unreachable facts
5. **Write output** → Generate `output.sas` file

## Implementation Status by Module

| Python Module | Python Functionality | Rust Module | Implementation Status | Differences/Issues |
|---|---|---|---|---|
| **Core Pipeline** |
| `translate.py` | Main pipeline orchestration, PDDL→SAS conversion | `bin/translator.rs` + `to_sas.rs` | 🟡 **Partial** | Missing normalization, simplified task building |
| `pddl_parser/` | PDDL parsing (S-expressions) | `pddl_parser.rs` | ✅ **Complete** | Basic S-expr parsing works |
| `pddl/` | PDDL AST classes | `pddl_ast.rs` | ✅ **Complete** | Domain/Problem/Action/Effect/Condition structures |
| `sas_tasks.py` | SAS+ data structures | `sas.rs` + `sas_writer.rs` | ✅ **Complete** | SAS task representation and file writing |
| **Instantiation & Grounding** |
| `instantiate.py` | Action grounding, reachability analysis | `instantiate.rs` | 🟡 **Partial** | Basic grounding works, missing fluent analysis |
| `build_model.py` | Prolog model building | *Not ported* | ❌ **Missing** | Uses external Prolog solver for reachability |
| `normalize.py` | Task normalization | *Not ported* | ❌ **Missing** | Missing axiom normalization, goal simplification |
| **Fact Grouping & Invariants** |
| `invariant_finder.py` | Balance checking, invariant discovery | `invariant_finder.rs` | 🔴 **Stub** | Missing BalanceChecker, heavy actions, reachable_action_params |
| `fact_groups.py` | Fact grouping, GroupCoverQueue | `fact_groups.rs` | 🔴 **Simplified** | Basic predicate grouping only, missing GroupCoverQueue |
| `invariants.py` | Invariant data structures | `invariants.rs` | ✅ **Complete** | Basic invariant representation |
| **Numeric Planning** |
| `numeric_axiom_rules.py` | Numeric axiom analysis, layering | `numeric_axiom_rules.rs` | 🟡 **Partial** | Missing equivalence detection, constant analysis |
| `derived_function_admin.py` | Derived function canonicalization | `derived_function_admin.rs` | 🔴 **Stub** | Basic placeholder naming only |
| **Constraint & Mutex Handling** |
| `constraints.py` | Constraint enumeration | `constraints.rs` | ✅ **Complete** | Constraint generation works |
| `axiom_rules.py` | Axiom handling | *Not ported* | ❌ **Missing** | Axiom layer computation, axiom rules |
| **Simplification** |
| `simplify.py` | Unreachable fact filtering | *Not ported* | ❌ **Missing** | DTG-based simplification |
| **Utilities** |
| `tools.py`, `timers.py`, `options.py` | Utilities | *Various* | 🟡 **Partial** | Basic timing, missing memory tracking |

## Critical Missing Components (High Priority)

### 1. **Invariant Finding & Fact Grouping**
**Python:** `invariant_finder.py` + `fact_groups.py`
- `BalanceChecker` with `add_inequality_preconds()` 
- Heavy action duplication for universal effects
- `GroupCoverQueue` for optimal fact group selection
- Translation key generation with proper mutex handling

**Rust Status:** Basic predicate-based grouping only
**Impact:** Different variable encoding → completely different SAS+ structure

### 2. **Numeric Axiom Analysis**
**Python:** `numeric_axiom_rules.py`
- `handle_axioms()`: constant detection, layer computation, equivalence mapping
- `identify_constants()`, `compute_axiom_layers()`, `identify_equivalent_axioms()`

**Rust Status:** Basic axiom creation, missing analysis
**Impact:** Incorrect numeric variable types, missing constant folding

### 3. **Task Normalization**
**Python:** `normalize.py`
- Axiom normalization, goal simplification
- Function symbol management

**Rust Status:** Not implemented
**Impact:** May miss optimizations, incorrect axiom handling

### 4. **Derived Function Canonicalization**
**Python:** `derived_function_admin.py` (implied from usage)
- Canonical naming for derived expressions
- Parameter management for arithmetic expressions

**Rust Status:** Basic stub
**Impact:** Different derived variable names → numeric variable mismatch

### 5. **Simplification**
**Python:** `simplify.py`
- `filter_unreachable_propositions()` using DTG analysis
- Removes unreachable facts after encoding

**Rust Status:** Not implemented
**Impact:** May include spurious variables, larger SAS+ files

## Implementation Priorities

### Phase 1: Core Translation Pipeline
1. **Fix fact grouping** - Implement proper `GroupCoverQueue` and invariant-based grouping
2. **Add task normalization** - Port essential normalization steps
3. **Improve numeric axiom analysis** - Complete constant detection and layering

### Phase 2: Advanced Features  
4. **Implement simplification** - Port DTG-based unreachable fact removal
5. **Add derived function canonicalization** - Proper derived variable naming
6. **Complete axiom handling** - Port axiom layer computation

### Phase 3: Polish & Optimization
7. **Add missing utilities** - Memory tracking, advanced timing
8. **Performance optimization** - Optimize hot paths
9. **Testing & validation** - Comprehensive test suite

## Key Technical Differences

### Translation Key Generation
**Python:** Uses invariant-based grouping with `BalanceChecker` to find optimal fact groups
**Rust:** Simple predicate-based grouping
**Effect:** Different SAS+ variable structure

### Numeric Variable Handling  
**Python:** Sophisticated axiom analysis with constant folding and equivalence detection
**Rust:** Basic axiom creation without analysis
**Effect:** Missing optimizations, different variable counts

### Comparison Axioms
**Python:** Integrated with main translation pipeline, proper variable indexing
**Rust:** Basic implementation, may have indexing issues
**Effect:** Incorrect comparison variable references

## Testing Strategy

1. **Unit tests** for each ported module
2. **Integration tests** comparing SAS+ output
3. **Regression tests** against known working examples
4. **Performance benchmarks** vs Python implementation

## Success Criteria

- [ ] Produces identical SAS+ files for test domains
- [ ] No Python runtime dependency  
- [ ] Passes all regression tests
- [ ] Performance comparable to or better than Python
