/// Port of pddl/functions.py
use std::fmt;
use super::pddl_types::TypedObject;

/// Python: class Function(object): def __init__(self, name, arguments, type_name)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<TypedObject>,
    pub type_name: String,
}

impl Function {
    pub fn new(name: String, arguments: Vec<TypedObject>, type_name: String) -> Self {
        Function { name, arguments, type_name }
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Function({}, {:?}, {})", self.name, self.arguments, self.type_name)
    }
}
