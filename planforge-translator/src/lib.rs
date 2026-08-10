use planforge_sas::numeric_task::NumericRootTask;
use planforge_translate::sas_tasks::SASTask;
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

pub fn translate_to_sas_to_path(
    domain: &str,
    problem: &str,
    output_path: &std::path::Path,
) -> anyhow::Result<()> {
    write_sas_file(domain, problem, false, output_path)
}

/// As [`translate_to_sas_to_path`], but with one SAS variable per fact instead
/// of the invariant-based encoding.
pub fn translate_to_sas_to_path_fast(
    domain: &str,
    problem: &str,
    output_path: &std::path::Path,
) -> anyhow::Result<()> {
    write_sas_file(domain, problem, true, output_path)
}

/// The SAS+ file is written a line at a time, so it goes through a buffer: on a
/// task with fifty thousand operators, writing straight to the file spends more
/// time in `write` syscalls than the rest of the translation takes.
fn write_sas_file(
    domain: &str,
    problem: &str,
    fast_groups: bool,
    output_path: &std::path::Path,
) -> anyhow::Result<()> {
    use std::io::Write;

    let mut out = std::io::BufWriter::new(std::fs::File::create(output_path)?);
    translate_to_sas_writer(domain, problem, fast_groups, &mut out)?;
    out.flush()?;
    Ok(())
}

/// In-memory entry point: emit the translator's SAS+ text as a `String`.
///
/// The text format is how the task reaches *other* planners and the reader of a
/// bug report; a search in this process gets its task from
/// [`translate_to_task`] instead, which does not go through text at all.
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
    let sas_task = translate_to_sas_task(domain, problem, fast_groups)?;
    planforge_translate::preprocess::write_reordered_sas(sas_task, out)?;
    Ok(())
}

/// Translate the (domain, problem) PDDL pair into the task the search reads.
///
/// The default way in: no SAS+ text is produced, and nothing is parsed.
pub fn translate_to_task(domain: &str, problem: &str) -> anyhow::Result<NumericRootTask> {
    translate_to_task_with_groups(domain, problem, false)
}

/// As [`translate_to_task`], but with one SAS variable per fact instead of the
/// invariant-based encoding.
pub fn translate_to_task_fast(domain: &str, problem: &str) -> anyhow::Result<NumericRootTask> {
    translate_to_task_with_groups(domain, problem, true)
}

fn translate_to_task_with_groups(
    domain: &str,
    problem: &str,
    fast_groups: bool,
) -> anyhow::Result<NumericRootTask> {
    let sas_task = translate_to_sas_task(domain, problem, fast_groups)?;
    Ok(planforge_translate::preprocess::reordered_numeric_task(
        sas_task,
    ))
}

/// PDDL to the translation's own task, which is what both the file and the
/// search task are built from.
fn translate_to_sas_task(
    domain: &str,
    problem: &str,
    fast_groups: bool,
) -> anyhow::Result<SASTask> {
    let task = PddlTask::from_files(std::path::Path::new(domain), std::path::Path::new(problem))
        .map_err(|e| anyhow::anyhow!(e))?;
    let parsed_task = task.to_task();

    let mut norm_task = normalize::NormalizableTask::from_task(parsed_task);
    norm_task.add_global_constraints();
    normalize::normalize(&mut norm_task);

    let result = planforge_translate::instantiate::explore(&norm_task.task);

    // `translate_task_from_grounded_internal` already filters unreachable
    // propositions and answers with a trivial task when that proves the task
    // impossible or trivially solvable, so nothing is left to simplify here.
    planforge_translate::translate::translate_task_from_grounded_internal(
        &result,
        &norm_task,
        fast_groups,
    )
    .map_err(|err| anyhow::anyhow!(err))
}
