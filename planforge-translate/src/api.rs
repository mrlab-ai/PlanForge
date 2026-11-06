//! The crate's entry points: PDDL in, either a task or SAS+ text out.
//!
//! Everything else in this crate is an implementation detail of the pipeline
//! these functions run, except the parsed PDDL task ([`crate::pddl`]), which
//! [`parse_pddl_task`] and [`parse_pddl_domain`] return for consumers that
//! work on the lifted task.

use std::fmt;
use std::path::PathBuf;

use planforge_sas::numeric_task::Task;

use crate::grounding::{GroundingLimitError, GroundingLimits};
use crate::options::LayerStrategy;
use crate::pddl_parser::lisp_parser::ParseError;
use crate::sas_tasks::SASTask;
use crate::{normalize, pddl_parser::PddlTask};

/// Why a translation failed.
#[derive(Debug)]
pub enum TranslateError {
    /// A PDDL file could not be read.
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    /// The PDDL input is malformed or uses what the translator does not
    /// implement ([`ParseError::kind`] tells which).
    Parse(ParseError),
    /// Grounding exceeded a configured limit.
    GroundingLimit(GroundingLimitError),
    /// The grounded task could not be encoded as SAS+.
    Encoding(String),
    /// Writing the SAS+ output failed.
    Write(std::io::Error),
}

impl fmt::Display for TranslateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => write!(f, "could not read {}: {source}", path.display()),
            Self::Parse(error) => fmt::Display::fmt(error, f),
            Self::GroundingLimit(error) => fmt::Display::fmt(error, f),
            Self::Encoding(message) => f.write_str(message),
            Self::Write(error) => fmt::Display::fmt(error, f),
        }
    }
}

impl std::error::Error for TranslateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } | Self::Write(source) => Some(source),
            Self::Parse(error) => Some(error),
            Self::GroundingLimit(error) => Some(error),
            Self::Encoding(_) => None,
        }
    }
}

impl From<ParseError> for TranslateError {
    fn from(error: ParseError) -> Self {
        Self::Parse(error)
    }
}

impl From<GroundingLimitError> for TranslateError {
    fn from(error: GroundingLimitError) -> Self {
        Self::GroundingLimit(error)
    }
}

impl From<std::io::Error> for TranslateError {
    fn from(error: std::io::Error) -> Self {
        Self::Write(error)
    }
}

/// Translate the (domain, problem) PDDL pair into the task the search reads.
///
/// The default way in: no SAS+ text is produced, and nothing is parsed.
pub fn translate_to_task(domain: &str, problem: &str) -> Result<Task, TranslateError> {
    translate_to_task_with_limits(domain, problem, GroundingLimits::default())
}

/// Translate a PDDL pair with explicit grounding resource limits.
pub fn translate_to_task_with_limits(
    domain: &str,
    problem: &str,
    limits: GroundingLimits,
) -> Result<Task, TranslateError> {
    translate_to_task_with_options(
        domain,
        problem,
        &TranslateOptions {
            limits,
            ..TranslateOptions::default()
        },
    )
}

/// What a translation to a task may ground and which variables it keeps.
#[derive(Clone, Copy, Debug, Default)]
pub struct TranslateOptions {
    pub limits: GroundingLimits,
    /// Keep every propositional variable, and every operator that changes
    /// one, instead of only those a goal, the global constraint or the metric
    /// depends on (Fast Downward's `--keep-unimportant-variables`). For
    /// consumers that observe whole states, such as learned policies over
    /// lifted state features: pruning drops facts that still change along
    /// reachable paths, and the operators that change them.
    pub keep_unimportant_variables: bool,
}

/// Translate a PDDL pair with explicit [`TranslateOptions`]. The task
/// records the files ([`Task::pddl_files`]).
pub fn translate_to_task_with_options(
    domain: &str,
    problem: &str,
    options: &TranslateOptions,
) -> Result<Task, TranslateError> {
    let sas_task = translate_to_sas_task(
        domain,
        problem,
        false,
        LayerStrategy::default(),
        options.limits,
    )?;
    let task =
        crate::preprocess::reordered_numeric_task(sas_task, options.keep_unimportant_variables);
    Ok(
        task.with_pddl_files(planforge_sas::numeric_task::PddlFiles {
            domain: domain.into(),
            problem: problem.into(),
        }),
    )
}

/// Parse a PDDL pair into the lifted task, as the files state it: before
/// normalization and grounding.
pub fn parse_pddl_task(
    domain: &str,
    problem: &str,
) -> Result<crate::pddl::tasks::Task, TranslateError> {
    let task = PddlTask::from_files(std::path::Path::new(domain), std::path::Path::new(problem))?;
    Ok(task.to_task()?)
}

/// Parse a PDDL domain file on its own.
pub fn parse_pddl_domain(
    domain: &str,
) -> Result<crate::pddl::tasks::DomainDefinition, TranslateError> {
    let path = std::path::Path::new(domain);
    let forms = crate::pddl_parser::lisp_parser::parse_nested_list(path)?;
    Ok(crate::pddl_parser::parsing_functions::parse_domain(&forms)
        .map_err(|error| error.in_file(path))?)
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
) -> Result<(), TranslateError> {
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
) -> Result<(), TranslateError> {
    write_sas_file(domain, problem, false, layer_strategy, output_path, limits)
}

/// As [`translate_to_sas_to_path`], but with one SAS variable per fact instead
/// of the invariant-based encoding.
pub fn translate_to_sas_to_path_fast(
    domain: &str,
    problem: &str,
    output_path: &std::path::Path,
) -> Result<(), TranslateError> {
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
pub fn translate_to_sas_string(domain: &str, problem: &str) -> Result<String, TranslateError> {
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
) -> Result<(), TranslateError> {
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
) -> Result<(), TranslateError> {
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
) -> Result<SASTask, TranslateError> {
    let task = PddlTask::from_files(std::path::Path::new(domain), std::path::Path::new(problem))?;
    let parsed_task = task.to_task()?;

    let mut task = parsed_task;
    task.add_global_constraints();
    normalize::normalize(&mut task);

    let result = crate::instantiate::explore(&task, limits)?;

    // `translate_task_from_grounded_internal` already filters unreachable
    // propositions and answers with a trivial task when that proves the task
    // impossible or trivially solvable, so nothing is left to simplify here.
    crate::translate::translate_grounded_task(&result, &task, fast_groups, layer_strategy)
        .map_err(TranslateError::Encoding)
}
