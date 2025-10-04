//! PDDL types
//! Port of python/translate/pddl/pddl_types.py

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};

/// Helper function to get type predicate name
/// PDDL allows mixing types and predicates, but some PDDL files
/// have name collisions. We internally give types predicate names
/// that cannot be confused with non-type predicates.
pub fn get_type_predicate_name(type_name: &str) -> String {
    format!("type@{}", type_name)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Type {
    pub name: String,
    pub basetype_name: Option<String>,
}

impl Type {
    pub fn new(name: String, basetype_name: Option<String>) -> Self {
        Self { name, basetype_name }
    }
    
    /// Get the predicate name for this type
    pub fn get_predicate_name(&self) -> String {
        get_type_predicate_name(&self.name)
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedObject {
    pub name: String,
    pub type_name: Option<String>,
}

impl TypedObject {
    pub fn new(name: String, type_name: Option<String>) -> Self {
        Self { name, type_name }
    }
    
    /// Create with explicit type name (more direct API)
    pub fn with_type(name: String, type_name: String) -> Self {
        Self { 
            name, 
            type_name: Some(type_name) 
        }
    }
    
    /// Get the type name or default to empty string
    pub fn get_type_name(&self) -> &str {
        self.type_name.as_deref().unwrap_or("")
    }
    
    /// Uniquify name to avoid conflicts
    pub fn uniquify_name(&self, type_map: &mut HashMap<String, String>, renamings: &mut HashMap<String, String>) -> Self {
        if !type_map.contains_key(&self.name) {
            type_map.insert(self.name.clone(), self.get_type_name().to_string());
            return self.clone();
        }
        
        for counter in 1.. {
            let new_name = format!("{}{}", self.name, counter);
            if !type_map.contains_key(&new_name) {
                renamings.insert(self.name.clone(), new_name.clone());
                type_map.insert(new_name.clone(), self.get_type_name().to_string());
                return TypedObject::with_type(new_name, self.get_type_name().to_string());
            }
        }
        
        unreachable!("Failed to find unique name")
    }
    
    /// Get atom representation for this typed object
    pub fn get_atom(&self) -> crate::translate::pddl::conditions::Literal {
        let predicate_name = get_type_predicate_name(self.get_type_name());
        crate::translate::pddl::conditions::Literal {
            predicate: predicate_name,
            args: vec![self.name.clone()],
            negated: false,
        }
    }
}

impl Hash for TypedObject {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.get_type_name().hash(state);
    }
}

impl fmt::Display for TypedObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.get_type_name())
    }
}

/// Type hierarchy utilities
pub struct TypeHierarchy {
    pub types: Vec<Type>,
}

impl TypeHierarchy {
    pub fn new() -> Self {
        Self { types: vec![] }
    }

    pub fn add_type(&mut self, type_def: Type) {
        self.types.push(type_def);
    }

    pub fn is_subtype(&self, _subtype: &str, _supertype: &str) -> bool {
        // TODO: Implement subtype checking
        false
    }
}
