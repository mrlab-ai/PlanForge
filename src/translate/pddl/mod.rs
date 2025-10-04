// Core PDDL types - moved from pddl_ast.rs

use std::collections::HashMap;

// Re-export all modules
pub mod actions;
pub mod axioms;
pub mod conditions;
pub mod effects;
pub mod f_expression;
pub mod functions;
pub mod pddl_types;
pub mod predicates;
pub mod tasks;

#[cfg(test)]
mod test_enhanced_pddl;

// Re-export core types
pub use tasks::{Task, PddlTask};
pub use pddl_types::*;
pub use predicates::*;
pub use functions::*;
pub use actions::*;
pub use axioms::*;
pub use conditions::{Condition, Literal};
pub use effects::Effect;

#[derive(Debug, Clone)]
pub struct Domain {
    pub name: String,
    pub requirements: Vec<String>,
    pub types: Vec<Type>,
    pub predicates: Vec<Predicate>,
    pub functions: Vec<Function>,
    pub actions: Vec<Action>,
    pub axioms: Vec<Axiom>,
}

#[derive(Debug, Clone)]
pub struct Problem {
    pub name: String,
    pub domain: String,
    pub objects: HashMap<String, String>,
    pub init: Vec<Literal>,
    pub goal: Condition,
    pub metric: Option<String>,
}

// Helper functions for S-expression to AST conversion
pub fn sexpr_to_condition(_sexpr: &crate::translate::pddl_parser::SExpr) -> Option<Condition> {
    // TODO: Implement actual parsing
    Some(Condition::Truth)
}

pub fn sexpr_to_effect(_sexpr: &crate::translate::pddl_parser::SExpr) -> Option<Effect> {
    // TODO: Implement actual parsing  
    use conditions::Literal;
    use effects::PrimitiveEffect;
    Some(Effect::new(
        vec![], // parameters
        Condition::Truth, // condition
        PrimitiveEffect::Literal(Literal {
            predicate: "dummy".to_string(),
            args: vec![],
            negated: false,
        })
    ))
}

pub fn substitute_condition(condition: &Condition, _mapping: &HashMap<String, String>) -> Condition {
    // TODO: Implement actual substitution
    condition.clone()
}

pub fn substitute_effect(effect: &Effect, _mapping: &HashMap<String, String>) -> Effect {
    // TODO: Implement actual substitution
    effect.clone()
}