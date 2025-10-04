//! Task normalization for PDDL tasks
//! Port of python/translate/normalize.py

use crate::translate::pddl::PddlTask;

/// Main normalization function that transforms a task into normalized form
/// This is a simplified version of the Python normalize() function
pub fn normalize(task: &mut PddlTask) {
    // TODO: Implement all normalization steps:
    // 1. remove_universal_quantifiers(task)
    // 2. substitute_complicated_goal(task)
    // 3. build_DNF(task)
    // 4. split_disjunctions(task)
    // 5. move_existential_quantifiers(task)
    // 6. eliminate_existential_quantifiers_from_axioms(task)
    // 7. eliminate_existential_quantifiers_from_preconditions(task)
    // 8. eliminate_existential_quantifiers_from_conditional_effects(task)
    // 9. verify_and_fix_arithmetic_expressions(task)
    // 10. remove_arithmetic_expressions(task)
    // 11. verify_axiom_predicates(task)
    
    // For now, just log that normalization is called
    println!("Normalize called for task: {}", task.summary());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_normalize_basic() {
        // Try to load a test task if available
        if let Ok(mut task) = PddlTask::from_files(
            Path::new("pddl/domain.pddl"), 
            Path::new("pddl/pfile1.pddl")
        ) {
            normalize(&mut task);
            // Test should not crash
        }
    }
}
