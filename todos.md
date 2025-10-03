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

## Dependency-Ordered Task List

Based on the analysis of the Python translation pipeline, here is the dependency-ordered implementation plan:

### Phase 1: Foundation & Core Infrastructure (Prerequisites)
**Timeline: 1-2 weeks**

#### Task 1.1: Task Normalization (`normalize.py` → Rust)
**Dependencies:** None  
**Effort:** Medium  
**Impact:** High

- **What it does:** Normalizes PDDL tasks by simplifying goal conditions, handling derived predicates, and standardizing function symbols
- **Why first:** Required by subsequent steps; affects how facts and axioms are processed
- **Implementation:** Create `src/translate/normalize.rs`
  - Port essential normalization functions
  - Handle goal simplification and constraint normalization
  - Function symbol management

#### Task 1.2: Build Model Integration (`build_model.py` integration)
**Dependencies:** Task 1.1  
**Effort:** Medium  
**Impact:** High

- **What it does:** Provides reachability analysis using external Prolog solver
- **Why needed:** Required for accurate fact collection and action parameter analysis
- **Implementation:** 
  - Create Rust interface to external Prolog (or port core logic)
  - Integrate with `instantiate.rs` for better reachability analysis
  - Generate `reachable_action_params` for invariant finding

### Phase 2: Invariant Finding & Fact Grouping (Core Logic)
**Timeline: 2-3 weeks**

#### Task 2.1: Complete Invariant Finder (`invariant_finder.py` → `invariant_finder.rs`)
**Dependencies:** Task 1.2  
**Effort:** Hard  
**Impact:** Critical

- **What it does:** 
  - `BalanceChecker` with `add_inequality_preconds()` for parameter constraints
  - Heavy action duplication for universal effects  
  - Candidate queue management with balance checking
  - Invariant refinement and useful group detection
- **Implementation:**
  - Complete `BalanceChecker` with proper inequality preconditions
  - Implement heavy action creation and parameter management
  - Port balance checking algorithms and candidate refinement
  - Ensure deterministic invariant ordering

#### Task 2.2: Advanced Fact Grouping (`fact_groups.py` → `fact_groups.rs`)
**Dependencies:** Task 2.1  
**Effort:** Hard  
**Impact:** Critical

- **What it does:**
  - `GroupCoverQueue` for optimal fact group selection
  - Translation key generation with proper value ordering
  - Mutex group computation and verification
- **Implementation:**
  - Complete `GroupCoverQueue` implementation
  - Port `choose_groups()` algorithm with proper cost evaluation
  - Implement translation key building with exact Python formatting
  - Add mutex group collection and validation

### Phase 3: Numeric Planning Enhancement
**Timeline: 1-2 weeks**

#### Task 3.1: Complete Numeric Axiom Analysis (`numeric_axiom_rules.py` → `numeric_axiom_rules.rs`)
**Dependencies:** Task 2.2  
**Effort:** Medium  
**Impact:** High

- **What it does:**
  - `handle_axioms()`: constant detection, equivalence mapping, layer computation
  - `identify_constants()`, `compute_axiom_layers()`, `identify_equivalent_axioms()`
- **Implementation:**
  - Complete constant detection and folding
  - Implement axiom layer computation with proper ordering
  - Add equivalence detection and axiom mapping
  - Ensure numeric variable type classification matches Python

#### Task 3.2: Derived Function Canonicalization (`derived_function_admin.py` → `derived_function_admin.rs`)
**Dependencies:** Task 3.1  
**Effort:** Medium  
**Impact:** High

- **What it does:** Canonical naming for derived expressions, parameter management for arithmetic
- **Implementation:**
  - Complete canonical derived expression naming
  - Implement parameter propagation for nested expressions
  - Ensure derived variable names match Python exactly
  - Add placeholder argument generation

### Phase 4: Task Translation & Processing
**Timeline: 1-2 weeks**

#### Task 4.1: Complete `strips_to_sas_dictionary()` Logic
**Dependencies:** Task 3.2  
**Effort:** Medium  
**Impact:** High

- **What it does:** Maps STRIPS facts to SAS variables with proper indexing
- **Implementation:**
  - Complete variable range computation
  - Implement proper fact-to-variable mapping
  - Add numeric variable dictionary construction
  - Ensure index consistency with Python

#### Task 4.2: Enhance Operator Translation
**Dependencies:** Task 4.1  
**Effort:** Medium  
**Impact:** Medium

- **What it does:** Complete operator precondition/effect translation with comparison axioms
- **Implementation:**
  - Improve comparison axiom creation and indexing
  - Add proper conditional effect handling
  - Implement numeric effect translation
  - Add mutex checking integration

### Phase 5: Advanced Features & Optimization
**Timeline: 1-2 weeks**

#### Task 5.1: Axiom Handling (`axiom_rules.py` → Rust)
**Dependencies:** Task 4.2  
**Effort:** Medium  
**Impact:** Medium

- **What it does:** Axiom layer computation, axiom rule processing
- **Implementation:**
  - Create `src/translate/axiom_rules.rs`
  - Port axiom layer computation
  - Implement axiom rule processing and validation

#### Task 5.2: Task Simplification (`simplify.py` → Rust)
**Dependencies:** Task 5.1  
**Effort:** Hard  
**Impact:** Medium

- **What it does:** DTG-based unreachable fact removal, task optimization
- **Implementation:**
  - Create `src/translate/simplify.rs`
  - Implement DTG construction and analysis
  - Port unreachable proposition filtering
  - Add task size optimization

### Phase 6: Integration & Testing
**Timeline: 1 week**

#### Task 6.1: Pipeline Integration
**Dependencies:** All previous tasks  
**Effort:** Medium  
**Impact:** Critical

- **Implementation:**
  - Update `bin/translator.rs` to use complete pipeline
  - Ensure proper module integration and data flow
  - Add comprehensive error handling

#### Task 6.2: Testing & Validation
**Dependencies:** Task 6.1  
**Effort:** Medium  
**Impact:** Critical

- **Implementation:**
  - Create comprehensive test suite
  - Add regression testing against Python output
  - Performance benchmarking and optimization

## Critical Path Analysis

**Critical Path:** 1.1 → 1.2 → 2.1 → 2.2 → 6.1 → 6.2  
**Estimated Total Time:** 8-12 weeks

**Parallel Opportunities:**
- Tasks 3.1 and 3.2 can run in parallel after Task 2.2
- Tasks 4.1 and 4.2 can run in parallel after Task 3.2  
- Tasks 5.1 and 5.2 can run in parallel after Task 4.2

**Highest Risk Tasks:**
1. **Task 2.1 (Invariant Finder)** - Most complex algorithm with intricate balance checking
2. **Task 2.2 (Fact Grouping)** - Critical for SAS structure, complex optimization algorithms  
3. **Task 5.2 (Simplification)** - Complex DTG analysis, potential performance bottlenecks

**Early Wins (Low-hanging fruit):**
1. **Task 1.1 (Normalization)** - Straightforward porting with clear interfaces
2. **Task 3.1 (Numeric Axioms)** - Well-defined mathematical operations
3. **Task 5.1 (Axiom Rules)** - Clear algorithmic steps

## Success Criteria

- [ ] **Semantic Parity:** Identical SAS+ files for all test domains
- [ ] **Performance:** Comparable or better than Python (target: 1-2x speed improvement)
- [ ] **No Dependencies:** Zero Python runtime requirements
- [ ] **Maintainability:** Clear, well-documented Rust code with comprehensive tests
