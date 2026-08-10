//! The crate's entry points: PDDL in, either a task or SAS+ text out.
//!
//! Everything else in this crate is an implementation detail of the pipeline
//! these four functions run.

use planforge_sas::numeric_task::NumericRootTask;

use crate::options::LayerStrategy;
use crate::sas_tasks::SASTask;
use crate::{normalize, pddl_parser::PddlTask};

/// Translate the (domain, problem) PDDL pair into the task the search reads.
///
/// The default way in: no SAS+ text is produced, and nothing is parsed.
pub fn translate_to_task(domain: &str, problem: &str) -> anyhow::Result<NumericRootTask> {
    let sas_task = translate_to_sas_task(domain, problem, false, LayerStrategy::default())?;
    Ok(crate::preprocess::reordered_numeric_task(sas_task))
}

/// Translate the pair and write the SAS+ text to `output_path`.
///
/// `layer_strategy` is here rather than on [`translate_to_task`] because the
/// layering is something the *file* carries: the search reads the layers it is
/// given, so choosing between them is a question for whoever writes the task
/// out, mainline's `--layer-strategy` included.
pub fn translate_to_sas_to_path(
    domain: &str,
    problem: &str,
    output_path: &std::path::Path,
    layer_strategy: LayerStrategy,
) -> anyhow::Result<()> {
    write_sas_file(domain, problem, false, layer_strategy, output_path)
}

/// As [`translate_to_sas_to_path`], but with one SAS variable per fact instead
/// of the invariant-based encoding.
pub fn translate_to_sas_to_path_fast(
    domain: &str,
    problem: &str,
    output_path: &std::path::Path,
) -> anyhow::Result<()> {
    write_sas_file(domain, problem, true, LayerStrategy::default(), output_path)
}

/// In-memory entry point: emit the translator's SAS+ text as a `String`.
///
/// The text format is how the task reaches *other* planners and the reader of a
/// bug report; a search in this process gets its task from
/// [`translate_to_task`] instead, which does not go through text at all.
pub fn translate_to_sas_string(domain: &str, problem: &str) -> anyhow::Result<String> {
    let mut buf: Vec<u8> = Vec::new();
    translate_to_sas_writer(domain, problem, false, LayerStrategy::default(), &mut buf)?;
    Ok(String::from_utf8(buf).expect("translator output is valid UTF-8"))
}

/// The SAS+ file is written a line at a time, so it goes through a buffer: on a
/// task with fifty thousand operators, writing straight to the file spends more
/// time in `write` syscalls than the rest of the translation takes.
fn write_sas_file(
    domain: &str,
    problem: &str,
    fast_groups: bool,
    layer_strategy: LayerStrategy,
    output_path: &std::path::Path,
) -> anyhow::Result<()> {
    use std::io::Write;

    let mut out = std::io::BufWriter::new(std::fs::File::create(output_path)?);
    translate_to_sas_writer(domain, problem, fast_groups, layer_strategy, &mut out)?;
    out.flush()?;
    Ok(())
}

/// Translate the (domain, problem) PDDL pair and write the SAS+ text to an
/// arbitrary `Write` sink.
fn translate_to_sas_writer<W: std::io::Write>(
    domain: &str,
    problem: &str,
    fast_groups: bool,
    layer_strategy: LayerStrategy,
    out: &mut W,
) -> anyhow::Result<()> {
    let sas_task = translate_to_sas_task(domain, problem, fast_groups, layer_strategy)?;
    crate::preprocess::write_reordered_sas(sas_task, out)?;
    Ok(())
}

/// PDDL to the translation's own task, which is what both the file and the
/// search task are built from.
pub(crate) fn translate_to_sas_task(
    domain: &str,
    problem: &str,
    fast_groups: bool,
    layer_strategy: LayerStrategy,
) -> anyhow::Result<SASTask> {
    let task = PddlTask::from_files(std::path::Path::new(domain), std::path::Path::new(problem))
        .map_err(|e| anyhow::anyhow!(e))?;
    let parsed_task = task.to_task();

    let mut norm_task = normalize::NormalizableTask::from_task(parsed_task);
    norm_task.add_global_constraints();
    normalize::normalize(&mut norm_task);

    let result = crate::instantiate::explore(&norm_task.task);

    // `translate_task_from_grounded_internal` already filters unreachable
    // propositions and answers with a trivial task when that proves the task
    // impossible or trivially solvable, so nothing is left to simplify here.
    crate::translate::translate_task_from_grounded_internal(
        &result,
        &norm_task,
        fast_groups,
        layer_strategy,
    )
    .map_err(|err| anyhow::anyhow!(err))
}
