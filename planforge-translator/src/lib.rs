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
    let mut out_file = std::fs::File::create(output_path)?;
    translate_to_sas_writer(domain, problem, false, &mut out_file)
}

/// As [`translate_to_sas_to_path`], but with one SAS variable per fact instead
/// of the invariant-based encoding.
pub fn translate_to_sas_to_path_fast(
    domain: &str,
    problem: &str,
    output_path: &std::path::Path,
) -> anyhow::Result<()> {
    let mut out_file = std::fs::File::create(output_path)?;
    translate_to_sas_writer(domain, problem, true, &mut out_file)
}

/// In-memory entry point: emit the translator's SAS+ text as a `String`.
/// Used by the in-process planforge pipeline so the `output.sas` file
/// never has to materialize on disk.
pub fn translate_to_sas_string(domain: &str, problem: &str) -> anyhow::Result<String> {
    let mut buf: Vec<u8> = Vec::new();
    translate_to_sas_writer(domain, problem, false, &mut buf)?;
    Ok(String::from_utf8(buf).expect("translator output is valid UTF-8"))
}

/// Core: translate the (domain, problem) PDDL pair and write the SAS+ text
/// to an arbitrary `Write` sink.
pub fn translate_to_sas_writer<W: std::io::Write>(
    domain: &str,
    problem: &str,
    fast_groups: bool,
    out: &mut W,
) -> anyhow::Result<()> {
    let task = PddlTask::from_files(std::path::Path::new(domain), std::path::Path::new(problem))
        .map_err(|e| anyhow::anyhow!(e))?;
    let parsed_task = task.to_task();

    let mut norm_task = normalize::NormalizableTask::from_task(parsed_task);
    norm_task.add_global_constraints();
    normalize::normalize(&mut norm_task).expect("normalization failed");

    let result = planforge_translate::instantiate::explore_normalized(&norm_task)
        .map_err(|e| anyhow::anyhow!(e))?;

    // `translate_task_from_grounded_internal` already filters unreachable
    // propositions and answers with a trivial task when that proves the task
    // impossible or trivially solvable, so nothing is left to simplify here.
    let sastask = planforge_translate::translate::translate_task_from_grounded_internal(
        &result,
        &norm_task,
        fast_groups,
    )
    .map_err(|err| anyhow::anyhow!(err))?;

    planforge_translate::preprocess::write_reordered_sas(sastask, out);
    Ok(())
}
