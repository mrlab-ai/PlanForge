//! PDDL predicates
//! Port of python/translate/pddl/predicates.py

use super::TypedObject;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Predicate {
    pub name: String,
    pub arguments: Vec<TypedObject>,
}

impl Predicate {
    pub fn new(name: String, arguments: Vec<TypedObject>) -> Self {
        Self { name, arguments }
    }

    pub fn parse(lisp_list: &[String]) -> Result<Self, String> {
        // TODO: Implement predicate parsing from lisp
        if lisp_list.is_empty() {
            return Err("Empty predicate definition".to_string());
        }
        
        Ok(Predicate::new(
            lisp_list[0].clone(),
            vec![], // TODO: Parse arguments
        ))
    }
    
    pub fn get_arity(&self) -> usize {
        self.arguments.len()
    }
}

impl std::fmt::Display for Predicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let args: Vec<String> = self.arguments.iter().map(|arg| arg.to_string()).collect();
        write!(f, "{}({})", self.name, args.join(", "))
    }
}
