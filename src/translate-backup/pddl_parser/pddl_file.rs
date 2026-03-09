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
        anyhow::anyhow!("Error: Could not read file: {}\nReason: {}.", filename.display(), error)
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

pub fn open(domain_filename: Option<&Path>, task_filename: Option<&Path>) -> anyhow::Result<PddlTask> {
    let global_options = options::get();
    let domain = match domain_filename {
        Some(path) => path.to_path_buf(),
        None => std::path::PathBuf::from(
            global_options
                .ok_or_else(|| anyhow::anyhow!("domain path not provided and options are not initialized"))?
                .domain
                .clone(),
        ),
    };
    let task = match task_filename {
        Some(path) => path.to_path_buf(),
        None => std::path::PathBuf::from(
            global_options
                .ok_or_else(|| anyhow::anyhow!("task path not provided and options are not initialized"))?
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translate::pddl::{Domain, Problem};

    #[test]
    fn parse_satellite_files_smoke() {
        let task = PddlTask::from_files(
            std::path::Path::new("others/satellite/domain.pddl"),
            std::path::Path::new("others/satellite/pfile1.pddl"),
        )
        .expect("pddl task should load");
        assert!(!task.domain_forms.is_empty());
        assert!(!task.problem_forms.is_empty());
        assert!(task.summary().contains("forms"));
    }

    #[test]
    fn parse_pddl_file_returns_toplevel_define_form() {
        let forms = parse_pddl_file("domain", std::path::Path::new("others/satellite/domain.pddl"))
            .expect("domain parse should succeed");
        assert!(!forms.is_empty());
        match &forms[0] {
            SExpr::List(items) => match &items[0] {
                SExpr::Atom(atom) => assert_eq!(atom, "define"),
                SExpr::List(_) => panic!("expected define atom"),
            },
            SExpr::Atom(_) => panic!("expected top-level define list"),
        }
    }

    #[test]
    fn parse_satellite_ast_smoke() {
        let task = PddlTask::from_files(
            std::path::Path::new("others/satellite/domain.pddl"),
            std::path::Path::new("others/satellite/pfile1.pddl"),
        )
        .expect("pddl task should load");
        let domain = Domain::from_sexprs(&task.domain_forms).expect("domain should parse");
        let problem = Problem::from_sexprs(&task.problem_forms).expect("problem should parse");
        assert_eq!(domain.name, "satellite");
        assert!(!problem.name.is_empty());
    }
}
