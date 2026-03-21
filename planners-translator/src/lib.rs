use planners_translate::{normalize, pddl_parser::PddlTask};

pub fn translate_to_sas(domain: &str, problem: &str) -> anyhow::Result<()> {
    translate_to_sas_to_path(domain, problem, std::path::Path::new("output.sas"))
}

pub fn translate_to_sas_to_path(
    domain: &str,
    problem: &str,
    output_path: &std::path::Path,
) -> anyhow::Result<()> {
    translate_to_sas_to_path_internal(domain, problem, output_path, false)
}

pub fn translate_to_sas_to_path_fast(
    domain: &str,
    problem: &str,
    output_path: &std::path::Path,
) -> anyhow::Result<()> {
    translate_to_sas_to_path_internal(domain, problem, output_path, true)
}

fn translate_to_sas_to_path_internal(
    domain: &str,
    problem: &str,
    output_path: &std::path::Path,
    fast_groups: bool,
) -> anyhow::Result<()> {
    let task = PddlTask::from_files(std::path::Path::new(domain), std::path::Path::new(problem))
        .map_err(|e| anyhow::anyhow!(e))?;
    let parsed_task = task.to_task();

    let mut norm_task = normalize::NormalizableTask::from_task(parsed_task);
    norm_task.add_global_constraints();
    normalize::normalize(&mut norm_task).expect("normalization failed");

    let result = planners_translate::instantiate::explore_normalized(&norm_task)
        .map_err(|e| anyhow::anyhow!(e))?;

    let instantiated_num_axioms = result.numeric_axioms;
    let py_groups: Option<Vec<Vec<String>>> = if fast_groups { Some(vec![]) } else { None };
    let mut sastask = planners_translate::translate::translate_task_from_grounded_internal(
        &result.atoms,
        &result.grounded_ops,
        &task.domain_forms,
        &task.problem_forms,
        &result.num_fluents,
        &instantiated_num_axioms,
        py_groups,
        &result.grounded_axioms,
        &result.reachable_action_params,
        &norm_task.goal,
        &norm_task,
    )
    .map_err(|err| anyhow::anyhow!(err))?;

    match planners_translate::simplify::filter_unreachable_propositions(&mut sastask) {
        Ok(()) => {}
        Err(planners_translate::simplify::SimplifyError::Impossible) => {
            sastask = planners_translate::simplify::trivial_task(false);
        }
        Err(planners_translate::simplify::SimplifyError::TriviallySolvable) => {
            sastask = planners_translate::simplify::trivial_task(true);
        }
        Err(planners_translate::simplify::SimplifyError::DoesNothing) => {
            // Task unchanged
        }
    }

    let py_task = planners_translate::sas_tasks::from_internal(&sastask);
    let mut out_file = std::fs::File::create(output_path)?;
    py_task.output(&mut out_file)?;
    Ok(())
}
