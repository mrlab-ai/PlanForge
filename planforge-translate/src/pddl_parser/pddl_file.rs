/// Entry point for parsing PDDL files.
use std::path::Path;

use super::lisp_parser::{self, SExpr};
use super::parsing_functions;
use crate::pddl::tasks::Task;

/// Holds the parsed S-expressions for domain and problem files.
#[derive(Debug)]
pub struct PddlTask {
    pub domain_forms: SExpr,
    pub problem_forms: SExpr,
}

impl PddlTask {
    /// Parse domain and problem PDDL files.
    pub fn from_files(domain_path: &Path, problem_path: &Path) -> Result<PddlTask, String> {
        let domain_forms = lisp_parser::parse_nested_list(domain_path)?;
        let problem_forms = lisp_parser::parse_nested_list(problem_path)?;
        Ok(PddlTask {
            domain_forms,
            problem_forms,
        })
    }

    /// Parse the task from the already-parsed S-expressions.
    pub fn to_task(&self) -> Task {
        parsing_functions::parse_task(&self.domain_forms, &self.problem_forms)
    }
}
