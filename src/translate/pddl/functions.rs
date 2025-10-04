//! PDDL functions
//! Port of python/translate/pddl/functions.py

use super::TypedObject;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<TypedObject>,
    pub type_name: String,
}

impl Function {
    pub fn new(name: String, arguments: Vec<TypedObject>, type_name: String) -> Self {
        Self {
            name,
            arguments,
            type_name,
        }
    }

    pub fn parse(lisp_list: &[String]) -> Result<Self, String> {
        // TODO: Implement function parsing from lisp
        if lisp_list.is_empty() {
            return Err("Empty function definition".to_string());
        }
        
        Ok(Function::new(
            lisp_list[0].clone(),
            vec![], // TODO: Parse arguments
            "number".to_string(), // Default type
        ))
    }
}

impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let args: Vec<String> = self.arguments.iter().map(|arg| arg.to_string()).collect();
        let result = format!("{}({})", self.name, args.join(", "));
        if !self.type_name.is_empty() {
            write!(f, "{}: {}", result, self.type_name)
        } else {
            write!(f, "{}", result)
        }
    }
}
