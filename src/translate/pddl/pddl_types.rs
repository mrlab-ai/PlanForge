/// Port of pddl/pddl_types.py
use std::fmt;

/// Represents a PDDL type with a name and optional base type.
/// Python: class Type(object): def __init__(self, name, basetype_name=None)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Type {
    pub name: String,
    pub basetype_name: Option<String>,
}

impl Type {
    pub fn new(name: &str, basetype_name: Option<&str>) -> Self {
        Type {
            name: name.to_string(),
            basetype_name: basetype_name.map(|s| s.to_string()),
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Type({}, {})", self.name,
               self.basetype_name.as_deref().unwrap_or("None"))
    }
}

/// Represents a typed object (variable or constant) in PDDL.
/// Python: class TypedObject(object): def __init__(self, name, type_name)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypedObject {
    pub name: String,
    pub type_name: String,
}

impl TypedObject {
    pub fn new(name: &str, type_name: &str) -> Self {
        TypedObject {
            name: name.to_string(),
            type_name: type_name.to_string(),
        }
    }

    /// Python: def uniquify_name(self, type_map, renamings)
    /// Renames the variable to avoid clashes with existing names.
    pub fn uniquify_name(&mut self, type_map: &mut std::collections::HashMap<String, usize>, renamings: &mut std::collections::HashMap<String, String>) {
        if self.name.starts_with('?') {
            let type_name = &self.type_name;
            let counter = type_map.entry(type_name.clone()).or_insert(0);
            let new_name = format!("?{}_{}", type_name, counter);
            *counter += 1;
            renamings.insert(self.name.clone(), new_name.clone());
            self.name = new_name;
        }
    }

    /// Python: def get_atom(self)
    /// Returns an Atom representing the type predicate for this object.
    pub fn get_atom(&self) -> super::Atom {
        let predicate = get_type_predicate_name(&self.type_name);
        super::Atom::new(predicate, vec![self.name.clone()])
    }
}

impl fmt::Display for TypedObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.type_name)
    }
}

/// Python: def _get_type_predicate_name(type_name)
pub fn get_type_predicate_name(type_name: &str) -> String {
    format!("=={}", type_name)
}
