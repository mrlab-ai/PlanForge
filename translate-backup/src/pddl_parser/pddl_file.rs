#[cfg(test)]
mod tests;

use std::fs;
use std::path::Path;

use crate::translate::options;
use crate::translate::pddl_parser::{parse_sexprs, SExpr};

fn read_pddl_text(filename: &Path) -> anyhow::Result<String> {
    let bytes = fs::read(filename)?;
    Ok(bytes.into_iter().map(char::from).collect())
}

pub fn parse_pddl_file(file_type: &str, filename: &Path) -> anyhow::Result<Vec<SExpr>> {
    let text = read_pddl_text(filename).map_err(|error| {
        anyhow::anyhow!(
            "Error: Could not read file: {}\nReason: {}.",
            filename.display(),
            error
        )
    })?;
    parse_sexprs(&text).map_err(|error| {
        anyhow::anyhow!(
            "Error: Could not parse {} file: {}\nReason: {}.",
            file_type,
            filename.display(),
            error
        )
    })
}

pub fn open(
    domain_filename: Option<&Path>,
    task_filename: Option<&Path>,
) -> anyhow::Result<PddlTask> {
    let global_options = options::get();
    let domain = match domain_filename {
        Some(path) => path.to_path_buf(),
        None => std::path::PathBuf::from(
            global_options
                .ok_or_else(|| {
                    anyhow::anyhow!("domain path not provided and options are not initialized")
                })?
                .domain
                .clone(),
        ),
    };
    let task = match task_filename {
        Some(path) => path.to_path_buf(),
        None => std::path::PathBuf::from(
            global_options
                .ok_or_else(|| {
                    anyhow::anyhow!("task path not provided and options are not initialized")
                })?
                .task
                .clone(),
        ),
    };
    PddlTask::from_files(&domain, &task)
}

/// Minimal placeholder AST for PDDL task used while porting the translator.
#[derive(Debug, Clone)]
pub struct PddlTask {
    pub domain_text: String,
    pub problem_text: String,
    pub domain_forms: Vec<SExpr>,
    pub problem_forms: Vec<SExpr>,
}

impl PddlTask {
    pub fn from_files(domain: &Path, problem: &Path) -> anyhow::Result<Self> {
        let d = read_pddl_text(domain)?;
        let p = read_pddl_text(problem)?;
        let domain_forms = parse_pddl_file("domain", domain)?;
        let problem_forms = parse_pddl_file("task", problem)?;
        Ok(PddlTask {
            domain_text: d,
            problem_text: p,
            domain_forms,
            problem_forms,
        })
    }

    /// Helper: return a small summary of the task for smoke-tests.
    pub fn summary(&self) -> String {
        format!(
            "domain={} bytes ({} forms), problem={} bytes ({} forms)",
            self.domain_text.len(),
            self.domain_forms.len(),
            self.problem_text.len(),
            self.problem_forms.len()
        )
    }
}
