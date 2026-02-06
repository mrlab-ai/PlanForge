use std::path::Path;

use crate::translate::instantiate;
use crate::translate::normalize;
use crate::translate::pddl;
use crate::translate::pddl_parser::PddlTask;
use crate::translate::simplify;
use crate::translate::to_sas;
use crate::translate::sas::SASTask;

#[derive(Debug, Clone)]
pub struct TranslateConfig {
    pub simplify: bool,
}

impl Default for TranslateConfig {
    fn default() -> Self {
        Self { simplify: true }
    }
}

pub fn translate_from_files(domain: &Path, problem: &Path) -> Result<SASTask, String> {
    translate_from_files_with_config(domain, problem, &TranslateConfig::default())
}

pub fn translate_from_files_with_config(
    domain: &Path,
    problem: &Path,
    config: &TranslateConfig,
) -> Result<SASTask, String> {
    let task = PddlTask::from_files(domain, problem).map_err(|err| err.to_string())?;
    let dom = pddl::Domain::from_sexprs(&task.domain_forms)
        .ok_or_else(|| "failed to parse domain PDDL".to_string())?;
    let prob = pddl::Problem::from_sexprs(&task.problem_forms)
        .ok_or_else(|| "failed to parse problem PDDL".to_string())?;
    translate_from_ast(&dom, &prob, config)
}

pub fn translate_from_ast(
    dom: &pddl::Domain,
    prob: &pddl::Problem,
    config: &TranslateConfig,
) -> Result<SASTask, String> {
    let mut norm_task = normalize::NormalizableTask::from_ast(dom, prob);
    norm_task.add_global_constraints();
    normalize::normalize(&mut norm_task)?;

    let exploration = instantiate::explore_normalized(&norm_task)
        .map_err(|err| err.to_string())?;

    let py_groups: Option<Vec<Vec<String>>> = None;
    let mut sas_task = to_sas::build_sas(
        &exploration.grounded_ops,
        dom,
        prob,
        &exploration.numeric_axioms,
        py_groups,
        &exploration.grounded_axioms,
        &norm_task.goal,
        &norm_task,
    )
    .map_err(|err| err.to_string())?;

    if config.simplify {
        match simplify::filter_unreachable_propositions(&mut sas_task) {
            Ok(_) => {}
            Err(simplify::SimplifyError::Impossible) => {
                sas_task = simplify::trivial_task(false);
            }
            Err(simplify::SimplifyError::TriviallySolvable) => {
                sas_task = simplify::trivial_task(true);
            }
        }
    }

    Ok(sas_task)
}
