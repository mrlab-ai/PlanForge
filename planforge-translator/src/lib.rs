use planforge_translate::{normalize, pddl_parser::PddlTask};
use std::num::NonZero;
use time::format_description::well_known::iso8601::{Config, TimePrecision};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt::time::UtcTime;
use tracing_subscriber::prelude::*;

pub fn init_logger(level: LevelFilter) {
    let timer = UtcTime::new(
        time::format_description::well_known::Iso8601::<
            {
                Config::DEFAULT
                    .set_time_precision(TimePrecision::Second {
                        decimal_digits: NonZero::new(3),
                    })
                    .encode()
            },
        >,
    );
    // Layer for stdout (info + debug + trace)
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_target(false)
        .with_timer(timer)
        .with_filter(level);

    // Layer for stderr (error + warn only)
    let stderr_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_target(false)
        .with_filter(LevelFilter::WARN);

    tracing_subscriber::registry()
        .with(stdout_layer)
        .with(stderr_layer)
        .init();
}

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

    let result = planforge_translate::instantiate::explore_normalized(&norm_task)
        .map_err(|e| anyhow::anyhow!(e))?;

    let instantiated_num_axioms = result.numeric_axioms;
    let py_groups: Option<Vec<Vec<String>>> = if fast_groups { Some(vec![]) } else { None };
    let mut sastask = planforge_translate::translate::translate_task_from_grounded_internal(
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

    match planforge_translate::simplify::filter_unreachable_propositions(&mut sastask) {
        Ok(()) => {}
        Err(planforge_translate::simplify::SimplifyError::Impossible) => {
            sastask = planforge_translate::simplify::trivial_task(false);
        }
        Err(planforge_translate::simplify::SimplifyError::TriviallySolvable) => {
            sastask = planforge_translate::simplify::trivial_task(true);
        }
        Err(planforge_translate::simplify::SimplifyError::DoesNothing) => {
            // Task unchanged
        }
    }

    let py_task = planforge_translate::sas_tasks::from_internal(&sastask);
    let mut out_file = std::fs::File::create(output_path)?;
    py_task.output(&mut out_file)?;
    Ok(())
}
