use super::*;
use crate::pddl::{Domain, Problem};

#[test]
fn translate_smoke() {
    let task = crate::translate::pddl_parser::PddlTask::from_files(
        std::path::Path::new("misc/plant-watering/domain.pddl"),
        std::path::Path::new("misc/plant-watering/prob_4_1_1.pddl"),
    )
    .unwrap();
    let dom = Domain::from_sexprs(&task.domain_forms).expect("domain parsed");
    let prob = Problem::from_sexprs(&task.problem_forms).expect("problem parsed");
    let mut norm_task = normalize::NormalizableTask::from_ast(&dom, &prob);
    normalize::normalize(&mut norm_task).expect("normalization failed");
    let prog = translate(&norm_task);
    assert!(!prog.facts.is_empty());
}
