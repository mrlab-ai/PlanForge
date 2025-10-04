#[cfg(test)]
mod integration_tests {
    use crate::translate::pddl::{PddlTask};
    use crate::translate::pddl::{Domain, Problem};
    use crate::translate::instantiate::ground;
    use crate::translate::to_sas::build_sas;

    #[test]
    fn translator_integration_numeric_smoke() {
        // load sample pddl files included in the repo
        let task = PddlTask::from_files(std::path::Path::new("pddl/domain.pddl"), std::path::Path::new("pddl/pfile1.pddl")).expect("read pddl files");
        let dom = Domain::from_sexprs(&task.domain_forms).expect("domain parsed");
        let prob = Problem::from_sexprs(&task.problem_forms).expect("problem parsed");
        // ground and build SAS
        let ops = ground(&dom, &prob);
        assert!(!ops.is_empty(), "grounding produced operators");
        let sas = build_sas(&ops, &prob);
        // numeric variables should be present for this domain
        assert!(sas.numeric_variables.len() > 0, "expected numeric variables");
        // numeric init length should match numeric_variables length
        assert_eq!(sas.numeric_init.len(), sas.numeric_variables.len());
        // at least one operator should contain a numeric effect or precondition
        let has_numeric = sas.operators.iter().any(|o| !o.numeric_effects.is_empty() || !o.numeric_preconds.is_empty());
        assert!(has_numeric, "expected at least one operator with numeric effect or precondition");
    }
}
