//! Entry point for parsing PDDL files.
use std::path::{Path, PathBuf};

use super::lisp_parser::{self, ParseResult, SExpr};
use super::parsing_functions;
use crate::pddl::tasks::Task;

/// The parsed S-expressions of a domain and a problem file.
#[derive(Debug)]
pub struct PddlTask {
    domain_path: PathBuf,
    problem_path: PathBuf,
    domain_forms: SExpr,
    problem_forms: SExpr,
}

impl PddlTask {
    /// Reads and tokenizes the domain and problem files.
    pub fn from_files(domain_path: &Path, problem_path: &Path) -> anyhow::Result<PddlTask> {
        Ok(PddlTask {
            domain_forms: lisp_parser::parse_nested_list(domain_path)?,
            problem_forms: lisp_parser::parse_nested_list(problem_path)?,
            domain_path: domain_path.to_path_buf(),
            problem_path: problem_path.to_path_buf(),
        })
    }

    /// The task the two files define; an error names the file it is in.
    pub fn to_task(&self) -> ParseResult<Task> {
        let domain = parsing_functions::parse_domain(&self.domain_forms)
            .map_err(|error| error.in_file(&self.domain_path))?;
        let problem = parsing_functions::parse_problem(&self.problem_forms, &domain.types)
            .map_err(|error| error.in_file(&self.problem_path))?;
        Ok(Task::new(domain, problem))
    }
}
