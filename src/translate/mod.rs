// Level 0 modules (no dependencies on other translate modules)
pub mod pddl;          // PDDL AST types
pub mod pddl_parser;   // PDDL parsing
pub mod sas;           // SAS format data structures  
pub mod constraints;   // Constraint system
pub mod tools;         // Utility functions ✅ WORKING
pub mod options;       // Command line parsing
pub mod timers;        // Timing utilities
pub mod sas_tasks;     // SAS task representation

// Level 1+ modules (EXCLUDED from compilation for now - don't delete!)
// Re-enable these gradually once Level 0 is working perfectly

pub mod numeric_axiom_rules;     // Level 1 - RE-ENABLED for testing ✅ WORKING
pub mod axiom_rules;             // Level 2 - RE-ENABLED for testing
pub mod invariants;              // Level 2 - RE-ENABLED for testing
// pub mod fact_groups;             // Level 2 - NEEDS invariant_finder dependency
pub mod normalize;               // Level 3 - RE-ENABLED for testing ✅ WORKING
pub mod simplify;                // Level 3 - RE-ENABLED for testing ✅ WORKING
pub mod instantiate;             // Level 3 - RE-ENABLED for testing ✅ API FIXED
// pub mod invariant_finder;        // Level 3 - NEEDS API fixes
// pub mod build_model;             // Level 4
// pub mod translate;               // Level 4 - main orchestrator

// Support modules (mixed levels - may need case-by-case analysis)
// pub mod to_sas;
// pub mod sas_writer;
pub mod derived_function_admin;  // Support - RE-ENABLED for instantiate
// pub mod graph;
// pub mod greedy_join;
pub mod pddl_to_prolog;      // Support - RE-ENABLED and COMPLETED ✅
// pub mod simple_to_restricted_task;
// pub mod split_rules;
