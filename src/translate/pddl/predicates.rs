/// Port of pddl/predicates.py
use std::fmt;
use super::pddl_types::TypedObject;

/// Python: class Predicate(object): def __init__(self, name, arguments)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Predicate {
    pub name: String,
    pub arguments: Vec<TypedObject>,
}

impl Predicate {
    pub fn new(name: String, arguments: Vec<TypedObject>) -> Self {
        Predicate { name, arguments }
    }

    /// Python: def get_arity(self)
    pub fn get_arity(&self) -> usize {
        self.arguments.len()
    }
}

impl fmt::Display for Predicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Predicate({}, {:?})", self.name, self.arguments)
    }
}
