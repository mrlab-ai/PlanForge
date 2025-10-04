//! Unit tests for enhanced PDDL modules
//! Tests to verify semantic equivalence with Python implementation

#[cfg(test)]
mod tests {
    use crate::translate::pddl::{Condition, Literal, TypedObject, Task};
    use crate::translate::pddl::tasks::Requirements;
    use std::collections::HashMap;

    #[test]
    fn test_literal_methods() {
        let lit = Literal::new("at".to_string(), vec!["?x".to_string(), "room1".to_string()]);
        
        // Test basic properties
        assert_eq!(lit.predicate, "at");
        assert_eq!(lit.args.len(), 2);
        assert!(!lit.negated);
        
        // Test free variables
        let free_vars = lit.free_variables();
        assert!(free_vars.contains("?x"));
        assert!(!free_vars.contains("room1"));
        
        // Test key method
        let (pred, args) = lit.key();
        assert_eq!(pred, &"at".to_string());
        assert_eq!(args.len(), 2);
        
        // Test negation
        let negated = lit.negate();
        assert!(negated.negated);
        assert_eq!(negated.predicate, "at");
        
        // Test positive
        let positive = negated.positive();
        assert!(!positive.negated);
    }
    
    #[test]
    fn test_literal_rename_variables() {
        let lit = Literal::new("at".to_string(), vec!["?x".to_string(), "?y".to_string()]);
        
        let mut renamings = HashMap::new();
        renamings.insert("?x".to_string(), "obj1".to_string());
        
        let renamed = lit.rename_variables(&renamings);
        assert_eq!(renamed.args[0], "obj1");
        assert_eq!(renamed.args[1], "?y"); // Unchanged
    }
    
    #[test]
    fn test_literal_replace_argument() {
        let lit = Literal::new("at".to_string(), vec!["?x".to_string(), "room1".to_string()]);
        
        let replaced = lit.replace_argument(0, "obj1".to_string());
        assert_eq!(replaced.args[0], "obj1");
        assert_eq!(replaced.args[1], "room1");
    }
    
    #[test]
    fn test_condition_simplification() {
        // Test conjunction simplification
        let truth1 = Condition::Truth;
        let truth2 = Condition::Truth;
        let conjunction = Condition::And(vec![truth1, truth2]);
        
        let simplified = conjunction.simplified();
        match simplified {
            Condition::Truth => {}, // Expected
            _ => panic!("Conjunction of Truth should simplify to Truth"),
        }
        
        // Test nested conjunction
        let nested = Condition::And(vec![
            Condition::And(vec![Condition::Truth, Condition::Truth]),
            Condition::Truth
        ]);
        
        let simplified_nested = nested.simplified();
        match simplified_nested {
            Condition::Truth => {}, // Expected
            _ => panic!("Nested conjunction should simplify"),
        }
    }
    
    #[test]
    fn test_condition_free_variables() {
        let lit = Literal::new("at".to_string(), vec!["?x".to_string(), "room1".to_string()]);
        let condition = Condition::Literal(lit.clone());
        
        let free_vars = condition.free_variables();
        assert!(free_vars.contains("?x"));
        assert_eq!(free_vars.len(), 1);
        
        // Test conjunction
        let lit2 = Literal::new("holding".to_string(), vec!["?y".to_string()]);
        let conjunction = Condition::And(vec![
            Condition::Literal(lit),
            Condition::Literal(lit2)
        ]);
        
        let conj_vars = conjunction.free_variables();
        assert!(conj_vars.contains("?x"));
        assert!(conj_vars.contains("?y"));
        assert_eq!(conj_vars.len(), 2);
    }
    
    #[test]
    fn test_typed_object_methods() {
        let obj = TypedObject::new("?x".to_string(), Some("object".to_string()));
        
        // Test basic properties
        assert_eq!(obj.name, "?x");
        assert_eq!(obj.get_type_name(), "object");
        
        // Test with no type
        let untyped = TypedObject::new("?y".to_string(), None);
        assert_eq!(untyped.get_type_name(), "");
        
        // Test with_type constructor
        let typed = TypedObject::with_type("?z".to_string(), "location".to_string());
        assert_eq!(typed.get_type_name(), "location");
    }
    
    #[test]
    fn test_typed_object_uniquify() {
        let obj = TypedObject::with_type("?x".to_string(), "object".to_string());
        
        let mut type_map = HashMap::new();
        let mut renamings = HashMap::new();
        
        // First uniquification should keep the same name
        let unique1 = obj.uniquify_name(&mut type_map, &mut renamings);
        assert_eq!(unique1.name, "?x");
        assert!(type_map.contains_key("?x"));
        
        // Second uniquification should create a new name
        let unique2 = obj.uniquify_name(&mut type_map, &mut renamings);
        assert_eq!(unique2.name, "?x1");
        assert!(renamings.contains_key("?x"));
    }
    
    #[test]
    fn test_typed_object_get_atom() {
        let obj = TypedObject::with_type("obj1".to_string(), "object".to_string());
        let atom = obj.get_atom();
        
        assert_eq!(atom.predicate, "type@object");
        assert_eq!(atom.args.len(), 1);
        assert_eq!(atom.args[0], "obj1");
        assert!(!atom.negated);
    }
    
    #[test]
    fn test_requirements() {
        // Test valid requirements
        let req = Requirements::new(vec![":strips".to_string(), ":typing".to_string()]);
        assert!(req.is_ok());
        
        let requirements = req.unwrap();
        assert_eq!(requirements.requirements.len(), 2);
        assert!(requirements.requirements.contains(&":strips".to_string()));
        
        // Test invalid requirement
        let invalid = Requirements::new(vec![":invalid".to_string()]);
        assert!(invalid.is_err());
    }
    
    #[test]
    fn test_task_creation() {
        let task = Task::new(
            "test-domain".to_string(),
            "test-problem".to_string(),
            Requirements::new(vec![]).unwrap(),
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            Condition::Truth,
            vec![],
            vec![],
            None,
        );
        
        assert_eq!(task.domain_name, "test-domain");
        assert_eq!(task.task_name, "test-problem");
        assert_eq!(task.axiom_counter, 0);
    }
    
    #[test]
    fn test_task_add_axiom() {
        let mut task = Task::new(
            "test".to_string(),
            "test".to_string(),
            Requirements::new(vec![]).unwrap(),
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            Condition::Truth,
            vec![],
            vec![],
            None,
        );
        
        let initial_axiom_count = task.axioms.len();
        let initial_predicate_count = task.predicates.len();
        
        task.add_axiom(vec![], Condition::Truth);
        
        assert_eq!(task.axioms.len(), initial_axiom_count + 1);
        assert_eq!(task.predicates.len(), initial_predicate_count + 1);
        assert_eq!(task.axiom_counter, 1);
        
        // Check axiom name format
        assert!(task.axioms[0].name.starts_with("new-axiom@"));
    }
}
