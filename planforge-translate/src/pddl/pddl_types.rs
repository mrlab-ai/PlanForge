//! PDDL types, typed objects and the type hierarchy.

use std::fmt;

/// Represents a PDDL type with a name and optional base type.
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
        write!(
            f,
            "Type({}, {})",
            self.name,
            self.basetype_name.as_deref().unwrap_or("None")
        )
    }
}

/// Represents a typed object (variable or constant) in PDDL.
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

    /// Renames the variable to avoid clashes with existing names.
    pub fn uniquify_name(
        &mut self,
        type_map: &mut std::collections::HashMap<String, usize>,
        renamings: &mut std::collections::HashMap<String, String>,
    ) {
        if self.name.starts_with('?') {
            let type_name = &self.type_name;
            let counter = type_map.entry(type_name.clone()).or_insert(0);
            let new_name = format!("?{}_{}", type_name, counter);
            *counter += 1;
            renamings.insert(self.name.clone(), new_name.clone());
            self.name = new_name;
        }
    }

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

pub fn get_type_predicate_name(type_name: &str) -> String {
    format!("=={}", type_name)
}

/// The declared types and, for each, its chain of supertypes up to `object`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeHierarchy {
    ancestors: std::collections::HashMap<String, Vec<String>>,
}

impl TypeHierarchy {
    /// The hierarchy `types` declare. Fails on a type declared below an
    /// undeclared one and on a cycle, which no object could inhabit.
    pub fn new(types: &[Type]) -> Result<Self, String> {
        let parents: std::collections::HashMap<&str, &str> = types
            .iter()
            .map(|t| {
                let parent = t.basetype_name.as_deref().unwrap_or("object");
                (t.name.as_str(), parent)
            })
            .filter(|(name, _)| *name != "object")
            .collect();
        let mut ancestors = std::collections::HashMap::new();
        ancestors.insert("object".to_string(), vec!["object".to_string()]);
        for &name in parents.keys() {
            let mut chain = vec![name.to_string()];
            let mut current = name;
            while current != "object" {
                current = parents.get(current).copied().ok_or_else(|| {
                    format!(
                        "type {} is declared below the undeclared type {current}",
                        chain[chain.len() - 1]
                    )
                })?;
                if chain.iter().any(|seen| seen == current) {
                    return Err(format!("type {name} is its own supertype"));
                }
                chain.push(current.to_string());
            }
            ancestors.insert(name.to_string(), chain);
        }
        Ok(Self { ancestors })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.ancestors.contains_key(name)
    }

    /// `name` and every supertype of it, nearest first, ending in `object`.
    pub fn ancestors(&self, name: &str) -> &[String] {
        self.ancestors
            .get(name)
            .unwrap_or_else(|| panic!("type {name} is declared"))
    }

    /// Every declared type, `object` included.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.ancestors.keys().map(String::as_str)
    }
}
