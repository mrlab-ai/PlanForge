//! PDDL parsing functions
//! Port of python/translate/pddl_parser/parsing_functions.py

use super::SExpr;
use crate::translate::pddl::{TypedObject, Predicate, Function};

pub fn parse_typed_list(sexp: &SExpr, constructor: fn(String, Option<String>) -> TypedObject) -> Result<Vec<TypedObject>, String> {
    match sexp {
        SExpr::List(items) => {
            let mut result: Vec<TypedObject> = Vec::new();
            let mut i = 0;
            
            while i < items.len() {
                if let SExpr::Atom(name) = &items[i] {
                    if name == "-" {
                        // Type annotation follows
                        i += 1;
                        if i < items.len() {
                            if let SExpr::Atom(type_name) = &items[i] {
                                // Apply type to previous items
                                for item in &mut result {
                                    if item.type_name.is_none() {
                                        item.type_name = Some(type_name.clone());
                                    }
                                }
                                i += 1;
                            } else {
                                return Err("Expected type name after -".to_string());
                            }
                        } else {
                            return Err("Expected type name after -".to_string());
                        }
                    } else {
                        result.push(constructor(name.clone(), None));
                        i += 1;
                    }
                } else {
                    return Err("Expected atom in typed list".to_string());
                }
            }
            
            Ok(result)
        }
        _ => Err("Expected list for typed list".to_string()),
    }
}

pub fn parse_predicate_definition(sexp: &SExpr) -> Result<Predicate, String> {
    match sexp {
        SExpr::List(items) => {
            if items.is_empty() {
                return Err("Empty predicate definition".to_string());
            }
            
            if let SExpr::Atom(name) = &items[0] {
                let mut arguments = Vec::new();
                
                if items.len() > 1 {
                    let args_sexp = SExpr::List(items[1..].to_vec());
                    arguments = parse_typed_list(&args_sexp, |name, type_name| {
                        TypedObject::new(name, type_name)
                    })?;
                }
                
                Ok(Predicate::new(name.clone(), arguments))
            } else {
                Err("Expected predicate name".to_string())
            }
        }
        SExpr::Atom(name) => Ok(Predicate::new(name.clone(), vec![])),
    }
}

pub fn parse_function_definition(sexp: &SExpr) -> Result<Function, String> {
    match sexp {
        SExpr::List(items) => {
            if items.is_empty() {
                return Err("Empty function definition".to_string());
            }
            
            if let SExpr::Atom(name) = &items[0] {
                let mut arguments = Vec::new();
                
                if items.len() > 1 {
                    let args_sexp = SExpr::List(items[1..].to_vec());
                    arguments = parse_typed_list(&args_sexp, |name, type_name| {
                        TypedObject::new(name, type_name)
                    })?;
                }
                
                Ok(Function::new(name.clone(), arguments, "number".to_string()))
            } else {
                Err("Expected function name".to_string())
            }
        }
        SExpr::Atom(name) => Ok(Function::new(name.clone(), vec![], "number".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translate::pddl_parser::lisp_parser::parse_lisp;

    #[test]
    fn test_parse_predicate() {
        let sexpr = parse_lisp("(at ?x ?y)").unwrap();
        let predicate = parse_predicate_definition(&sexpr[0]).unwrap();
        assert_eq!(predicate.name, "at");
    }
}
