//! PDDL file handling
//! Port of python/translate/pddl_parser/pddl_file.py

use std::fs;
use std::path::Path;
use super::{SExpr, parse_lisp};

pub struct PddlFile {
    pub content: String,
    pub sexpr: Vec<SExpr>,
}

impl PddlFile {
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        
        // Convert to static string for parser (this is a limitation we need to address)
        let static_content = Box::leak(content.clone().into_boxed_str());
        let sexpr = parse_lisp(static_content)?;
        
        Ok(Self { content, sexpr })
    }

    pub fn from_string(content: String) -> Result<Self, String> {
        let static_content = Box::leak(content.clone().into_boxed_str());
        let sexpr = parse_lisp(static_content)?;
        Ok(Self { content, sexpr })
    }

    pub fn get_domain_name(&self) -> Option<String> {
        for expr in &self.sexpr {
            if let SExpr::List(items) = expr {
                if items.len() >= 3 {
                    if let (SExpr::Atom(define), SExpr::Atom(domain), SExpr::Atom(name)) = 
                        (&items[0], &items[1], &items[2]) {
                        if define == "define" && domain == "domain" {
                            return Some(name.clone());
                        }
                    }
                }
            }
        }
        None
    }

    pub fn get_problem_name(&self) -> Option<String> {
        for expr in &self.sexpr {
            if let SExpr::List(items) = expr {
                if items.len() >= 3 {
                    if let (SExpr::Atom(define), SExpr::Atom(problem), SExpr::Atom(name)) = 
                        (&items[0], &items[1], &items[2]) {
                        if define == "define" && problem == "problem" {
                            return Some(name.clone());
                        }
                    }
                }
            }
        }
        None
    }
}
