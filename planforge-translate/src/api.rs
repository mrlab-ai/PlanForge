//! The crate's entry points: PDDL in, either a task or SAS+ text out.
//!
//! Everything else in this crate is an implementation detail of the pipeline
//! these four functions run.

use planforge_sas::numeric_task::NumericRootTask;

use crate::grounding::GroundingLimits;
use crate::options::LayerStrategy;
use crate::sas_tasks::SASTask;
use crate::{normalize, pddl_parser::PddlTask};

/// Translate the (domain, problem) PDDL pair into the task the search reads.
///
/// The default way in: no SAS+ text is produced, and nothing is parsed.
pub fn translate_to_task(domain: &str, problem: &str) -> anyhow::Result<NumericRootTask> {
    translate_to_task_with_limits(domain, problem, GroundingLimits::default())
}

/// Translate a PDDL pair with explicit grounding resource limits.
pub fn translate_to_task_with_limits(
    domain: &str,
    problem: &str,
    limits: GroundingLimits,
) -> anyhow::Result<NumericRootTask> {
    let sas_task = translate_to_sas_task(domain, problem, false, LayerStrategy::default(), limits)?;
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
    translate_to_sas_to_path_with_limits(
        domain,
        problem,
        output_path,
        layer_strategy,
        GroundingLimits::default(),
    )
}

/// Translate to a SAS+ file with explicit grounding resource limits.
///
/// The output is created only after grounding succeeds, so a limit error never
/// leaves behind a partial task.
pub fn translate_to_sas_to_path_with_limits(
    domain: &str,
    problem: &str,
    output_path: &std::path::Path,
    layer_strategy: LayerStrategy,
    limits: GroundingLimits,
) -> anyhow::Result<()> {
    write_sas_file(domain, problem, false, layer_strategy, output_path, limits)
}

/// As [`translate_to_sas_to_path`], but with one SAS variable per fact instead
/// of the invariant-based encoding.
pub fn translate_to_sas_to_path_fast(
    domain: &str,
    problem: &str,
    output_path: &std::path::Path,
) -> anyhow::Result<()> {
    write_sas_file(
        domain,
        problem,
        true,
        LayerStrategy::default(),
        output_path,
        GroundingLimits::default(),
    )
}

/// In-memory entry point: emit the translator's SAS+ text as a `String`.
///
/// The text format is how the task reaches *other* planners and the reader of a
/// bug report; a search in this process gets its task from
/// [`translate_to_task`] instead, which does not go through text at all.
pub fn translate_to_sas_string(domain: &str, problem: &str) -> anyhow::Result<String> {
    let mut buf: Vec<u8> = Vec::new();
    translate_to_sas_writer(
        domain,
        problem,
        false,
        LayerStrategy::default(),
        GroundingLimits::default(),
        &mut buf,
    )?;
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
    limits: GroundingLimits,
) -> anyhow::Result<()> {
    use std::io::Write;

    let sas_task = translate_to_sas_task(domain, problem, fast_groups, layer_strategy, limits)?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(output_path)?);
    crate::preprocess::write_reordered_sas(sas_task, &mut out)?;
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
    limits: GroundingLimits,
    out: &mut W,
) -> anyhow::Result<()> {
    let sas_task = translate_to_sas_task(domain, problem, fast_groups, layer_strategy, limits)?;
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
    limits: GroundingLimits,
) -> anyhow::Result<SASTask> {
    let task = PddlTask::from_files(std::path::Path::new(domain), std::path::Path::new(problem))
        .map_err(|e| anyhow::anyhow!(e))?;
    let parsed_task = task.to_task();

    let mut norm_task = normalize::NormalizableTask::from_task(parsed_task);
    norm_task.add_global_constraints();
    normalize::normalize(&mut norm_task);

    let result = crate::instantiate::explore(&norm_task.task, limits)?;

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
