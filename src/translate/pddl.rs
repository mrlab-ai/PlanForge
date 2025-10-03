use std::path::Path;
use std::fs::File;
use std::io::Read;

use crate::translate::pddl_parser::{parse_sexprs, SExpr};

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
        let mut d = String::new();
        let mut p = String::new();
        File::open(domain)?.read_to_string(&mut d)?;
        File::open(problem)?.read_to_string(&mut p)?;
    let domain_forms = parse_sexprs(&d).map_err(|e| anyhow::anyhow!(e))?;
    let problem_forms = parse_sexprs(&p).map_err(|e| anyhow::anyhow!(e))?;
    Ok(PddlTask { domain_text: d, problem_text: p, domain_forms, problem_forms })
    }

    /// Helper: return a small summary of the task for smoke-tests.
    pub fn summary(&self) -> String {
    format!("domain={} bytes ({} forms), problem={} bytes ({} forms)",
        self.domain_text.len(), self.domain_forms.len(),
        self.problem_text.len(), self.problem_forms.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translate::pddl_ast::{Domain, Problem};

    #[test]
    fn build_ast_smoke() {
        let task = PddlTask::from_files(std::path::Path::new("pddl/domain.pddl"), std::path::Path::new("pddl/pfile1.pddl")).unwrap();
        // attempt to build Domain and Problem ASTs
        let dom = Domain::from_sexprs(&task.domain_forms).expect("domain parsed");
        let prob = Problem::from_sexprs(&task.problem_forms).expect("problem parsed");
        assert!(!dom.name.is_empty());
        assert!(!prob.name.is_empty());
    }
}
