//! The planner's single entry point: one CLI, one binary, every `--search`
//! spec.
//!
//! The crate owns the process-level concerns -- argument parsing, resource
//! limits and the wrapped child process, logging, the plan file and the exit
//! code -- and delegates the planning itself to `planforge-translate` and
//! `planforge-searcher`/`planforge-search`.

mod allocator;
mod limits;
pub mod output;
pub mod portfolio;
#[cfg(test)]
mod tests;

pub use output::{
    EXIT_TIMEOUT, exit_code_for_search_status, print_plan_result, print_search_result,
    write_plan_file,
};
pub use portfolio::PortfolioOptions;

use allocator::register_event_handlers;
use clap::{Args, Parser, Subcommand};
use limits::{
    apply_process_limits, format_time_limit, normalize_wrapped_exit, parse_memory_limit,
    parse_time_limit,
};
use planforge_sas::numeric_task::{AbstractNumericTask, NumericRootTask, TaskRef};
use planforge_sas::state_registry::StateRegistry;
use planforge_search::heuristic_factory::{
    ExternalHeuristic, HeuristicBuildError, register_external_heuristics,
};
use planforge_search::search::{SearchBuildContext, SearchResult, search_algorithm};
use planforge_search::state_space::{EnumerationLimits, OwnedStateSpace, enumerate_state_space};
use planforge_search::task_restriction::{build_icaps26_restricted_task, build_restricted_task};
use planforge_searcher::{HeuristicSpec, SearchSpec};
use std::ffi::OsString;
use std::io::{BufWriter, Write};
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use time::format_description::well_known::iso8601::{Config, TimePrecision};
use tracing::info;
use tracing_subscriber::fmt::time::UtcTime;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use tracing_subscriber::filter::LevelFilter;
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

/// Run the standard PlanForge CLI, with extra heuristics registered first.
/// This is the supported way to ship your own heuristic inside the real binary.
pub fn run_with_heuristics(extra: Vec<ExternalHeuristic>) -> std::io::Result<()> {
    register_external_heuristics(extra).map_err(std::io::Error::other)?;

    let cli = PlannersCli::parse();
    init_logger(
        cli.log_level
            .unwrap_or(tracing_subscriber::filter::LevelFilter::INFO),
    );
    if let Some(command) = &cli.command {
        return run_command(command);
    }
    if cli.inputs.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "search requires one SAS input or a domain/problem PDDL pair",
        ));
    }
    // The portfolio drives whole `planforge` runs as its stages, so it has to
    // branch off before this process turns itself into one of them.
    if cli.portfolio.portfolio {
        return portfolio::run_portfolio(&cli);
    }
    #[cfg(unix)]
    if !cli.internal_run {
        return run_wrapped_process(&cli);
    }

    install_process_hooks(cli.max_memory)?;
    match run_internal(&cli) {
        Ok(result) => std::process::exit(exit_code_for_search_status(&result.status)),
        Err(error) if error.kind() == std::io::ErrorKind::TimedOut => {
            tracing::info!("Time limit reached during heuristic construction.");
            std::process::exit(EXIT_TIMEOUT);
        }
        Err(error) => Err(error),
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Numeric planner")]
pub struct PlannersCli {
    #[arg(long = "max-memory", value_name = "SIZE", value_parser = parse_memory_limit)]
    pub max_memory: Option<u64>,

    #[arg(long = "max-time", value_name = "DURATION", value_parser = parse_time_limit)]
    pub max_time: Option<Duration>,

    #[arg(long = "log-level")]
    pub log_level: Option<tracing_subscriber::filter::LevelFilter>,

    #[arg(long, hide = true)]
    pub internal_run: bool,

    #[arg(long = "restrict-task")]
    pub restrict_task: bool,

    /// Recursive search configuration.
    /// Examples: `astar(blind())`, `astar(domain_abstraction())`,
    /// `astar(check_admissible(domain_abstraction()))`.
    #[arg(
        long,
        value_name = "SPEC",
        default_value = "astar(blind())",
        value_parser = planforge_searcher::parse_search_spec,
        long_help = planforge_search::search::search_algorithm_help()
    )]
    pub search: SearchSpec,

    #[command(flatten)]
    pub portfolio: PortfolioOptions,

    #[command(subcommand)]
    pub command: Option<PlanforgeCommand>,

    #[arg(value_name = "INPUT", num_args = 0..=2)]
    pub inputs: Vec<String>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum PlanforgeCommand {
    /// Enumerate the complete reachable state space and write standard CSV files.
    Enumerate(EnumerateCli),
}

#[derive(Args, Debug, Clone)]
pub struct EnumerateCli {
    #[arg(long)]
    pub max_states: usize,

    #[arg(long)]
    pub max_transitions: usize,

    #[arg(long, value_name = "DURATION", value_parser = parse_time_limit)]
    pub max_time: Duration,

    /// New directory that will receive states.csv, transitions.csv,
    /// summary.csv, and h_star_histogram.csv.
    #[arg(long, value_name = "DIRECTORY")]
    pub output: PathBuf,

    #[arg(long = "restrict-task")]
    pub restrict_task: bool,

    #[arg(value_name = "INPUT", required = true, num_args = 1..=2)]
    pub inputs: Vec<String>,
}

pub fn run_command(command: &PlanforgeCommand) -> std::io::Result<()> {
    match command {
        PlanforgeCommand::Enumerate(arguments) => run_enumeration(arguments),
    }
}

#[cfg(unix)]
pub fn run_wrapped_process(cli: &PlannersCli) -> std::io::Result<()> {
    let current_executable = std::env::current_exe()?;
    let mut child_args = vec![OsString::from("--internal-run")];
    let memory_limit = cli
        .max_memory
        .map(limits::effective_rss_limit)
        .transpose()?;
    if let Some(max_memory) = memory_limit {
        child_args.push(OsString::from("--max-memory"));
        child_args.push(OsString::from(max_memory.to_string()));
    }
    if let Some(max_time) = cli.max_time {
        child_args.push(OsString::from("--max-time"));
        child_args.push(OsString::from(format_time_limit(max_time)));
    }
    if let Some(level) = cli.log_level {
        child_args.push(OsString::from("--log-level"));
        child_args.push(OsString::from(level.to_string()));
    }
    if cli.restrict_task {
        child_args.push(OsString::from("--restrict-task"));
    }
    child_args.push(OsString::from("--search"));
    child_args.push(OsString::from(cli.search.to_string()));
    child_args.extend(cli.inputs.iter().cloned().map(OsString::from));

    let time_limit = cli.max_time;
    let mut command = Command::new(current_executable);
    command.args(child_args);
    command.stdin(std::process::Stdio::inherit());
    command.stdout(std::process::Stdio::inherit());
    command.stderr(std::process::Stdio::inherit());

    unsafe {
        command.pre_exec(move || apply_process_limits(time_limit, memory_limit));
    }

    let mut child = command.spawn()?;
    #[cfg(target_os = "linux")]
    let status = match memory_limit {
        Some(memory_limit) => limits::wait_with_memory_limit(&mut child, memory_limit)?,
        None => child.wait()?,
    };
    #[cfg(not(target_os = "linux"))]
    let status = child.wait()?;
    let exit_code = normalize_wrapped_exit(status, time_limit, memory_limit);

    std::process::exit(exit_code)
}

/// Install the process-level hooks the in-process run depends on: the memory
/// padding that lets an out-of-memory condition be *reported* rather than
/// aborted, and the recovery hook the allocator calls to release it.
pub fn install_process_hooks(memory_limit: Option<u64>) -> std::io::Result<()> {
    planforge_search::resource_limits::reserve_memory_padding(memory_limit)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;
    #[cfg(unix)]
    allocator::install_oom_recovery(planforge_search::resource_limits::release_padding_for_oom);
    Ok(())
}

/// Run search for an already-parsed task and return the result. Contains no
/// CLI, signal-handler, logging-setup, or file-writing side effects -- those
/// stay in `run_internal`. Handles the astar / gbfs / astar_fs specs; `sgd`
/// is dispatched directly in `run_internal`.
pub fn solve_task(
    task: TaskRef<'_>,
    spec: &SearchSpec,
    time_limit: Option<Duration>,
    memory_limit: Option<u64>,
) -> std::io::Result<SearchResult> {
    let state_registry = StateRegistry::for_task(task.clone());
    let (algorithm_name, primary_heuristic, secondary_heuristic, mpd) = match spec {
        SearchSpec::Astar(heuristic, mpd) => {
            let heuristic_override = build_heuristic_from_spec(heuristic, &*task, task.clone())?;
            info!("Starting A* search with {heuristic:?}, mpd={mpd}...");
            ("astar", heuristic_override, None, *mpd)
        }
        SearchSpec::Gbfs(heuristic) => {
            let heuristic_override = build_heuristic_from_spec(heuristic, &*task, task.clone())?;
            info!("Starting GBFS search with {heuristic:?}...");
            ("gbfs", heuristic_override, None, false)
        }
        SearchSpec::AstarFs(fast_spec, slow_spec) => {
            // A* with two admissible heuristics: a fast one for ordering
            // and a slow one evaluated lazily on second-pop. Treats the
            // user's `blind` choice as a placeholder by materializing a
            // real `BlindHeuristic` with the task's min-action-cost.
            let task_ref: &dyn AbstractNumericTask = &*task;
            let original_costs: Vec<f64> = task
                .get_operators()
                .iter()
                .map(|op| {
                    planforge_sas::numeric_task::metric_operator_cost_from_initial_values(
                        task_ref, op,
                    )
                })
                .collect();
            let min_cost = original_costs
                .iter()
                .copied()
                .fold(f64::INFINITY, |a, b| a.min(b));
            let min_action_cost = if min_cost.is_finite() {
                min_cost.max(0.0)
            } else {
                1.0
            };
            let make_blind = || {
                Box::new(
                    planforge_search::evaluation::heuristic::BlindHeuristic::with_min_action_cost(
                        min_action_cost,
                        None,
                    ),
                ) as Box<dyn planforge_search::evaluation::Heuristic + '_>
            };
            let fast_h = build_heuristic_from_spec(fast_spec, task_ref, task.clone())?
                .unwrap_or_else(make_blind);
            let slow_h = build_heuristic_from_spec(slow_spec, task_ref, task.clone())?
                .unwrap_or_else(make_blind);
            info!("Starting A* fast/slow search with fast={fast_spec:?} slow={slow_spec:?}...");
            ("astar_fs", Some(fast_h), Some(slow_h), false)
        }
        // `sgd(...)` is not a search engine and does not go through a state
        // registry or an open list; `run_internal` dispatches it directly.
        SearchSpec::Sgd(_) => {
            return Err(std::io::Error::other(
                "solve_task does not handle `sgd(...)`; it is dispatched in run_internal",
            ));
        }
    };
    let plugin = search_algorithm(algorithm_name)
        .expect("every parsed state-space search has a registered builder");
    let mut driver = (plugin.build)(SearchBuildContext {
        task: &*task,
        state_registry,
        primary_heuristic,
        secondary_heuristic,
        time_limit,
        max_memory_bytes: memory_limit,
        mpd,
    })
    .map_err(|error| std::io::Error::other(format!("could not build search: {error:#}")))?;
    driver
        .run()
        .map_err(|error| std::io::Error::other(format!("search failed: {error:#}")))
}

fn load_enumeration_task(arguments: &EnumerateCli) -> std::io::Result<NumericRootTask> {
    let mut task = if arguments.inputs.len() == 2 {
        planforge_translate::translate_to_task(&arguments.inputs[0], &arguments.inputs[1])
            .map_err(|error| std::io::Error::other(error.to_string()))?
    } else {
        NumericRootTask::try_from_file(&arguments.inputs[0]).map_err(std::io::Error::other)?
    };
    if arguments.restrict_task
        && let Some(restricted) = build_restricted_task(&task)
            .map_err(|error| std::io::Error::other(format!("failed to restrict task: {error:#}")))?
    {
        task = restricted.into_task();
    }
    Ok(task)
}

fn create_csv(path: &Path) -> std::io::Result<BufWriter<std::fs::File>> {
    std::fs::File::create(path).map(BufWriter::new)
}

fn write_state_space_csv(output: &Path, graph: &OwnedStateSpace) -> std::io::Result<()> {
    if output.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!(
                "enumeration output directory {} already exists; refusing to overwrite it",
                output.display()
            ),
        ));
    }
    std::fs::create_dir(output)?;

    let mut states = create_csv(&output.join("states.csv"))?;
    write!(states, "state_id,is_goal,h_star")?;
    for variable in 0..graph.num_propositional_variables {
        write!(states, ",proposition_{variable}")?;
    }
    for variable in 0..graph.num_numeric_variables {
        write!(states, ",numeric_{variable}")?;
    }
    writeln!(states)?;
    for state_id in 0..graph.num_states() {
        write!(
            states,
            "{state_id},{},{}",
            graph.goal_states[state_id], graph.h_star[state_id]
        )?;
        let propositional_start = state_id
            .checked_mul(graph.num_propositional_variables)
            .expect("propositional CSV row offset overflow");
        let propositional_end = propositional_start + graph.num_propositional_variables;
        for value in &graph.propositional_values[propositional_start..propositional_end] {
            write!(states, ",{value}")?;
        }
        let numeric_start = state_id
            .checked_mul(graph.num_numeric_variables)
            .expect("numeric CSV row offset overflow");
        let numeric_end = numeric_start + graph.num_numeric_variables;
        for value in &graph.numeric_values[numeric_start..numeric_end] {
            write!(states, ",{value}")?;
        }
        writeln!(states)?;
    }
    states.flush()?;

    let mut transitions = create_csv(&output.join("transitions.csv"))?;
    writeln!(
        transitions,
        "source_state_id,operator_id,successor_state_id,cost"
    )?;
    for source_state_id in 0..graph.num_states() {
        let start = usize::try_from(graph.transition_offsets[source_state_id])
            .expect("transition offset exceeds usize");
        let end = usize::try_from(graph.transition_offsets[source_state_id + 1])
            .expect("transition offset exceeds usize");
        for transition_id in start..end {
            writeln!(
                transitions,
                "{source_state_id},{},{},{}",
                graph.transition_operator_ids[transition_id],
                graph.transition_successor_ids[transition_id],
                graph.transition_costs[transition_id]
            )?;
        }
    }
    transitions.flush()?;

    let summary = graph.summary();
    let mut summary_csv = create_csv(&output.join("summary.csv"))?;
    writeln!(summary_csv, "metric,value")?;
    writeln!(summary_csv, "states,{}", summary.state_count)?;
    writeln!(summary_csv, "transitions,{}", summary.transition_count)?;
    writeln!(summary_csv, "goal_states,{}", summary.goal_state_count)?;
    writeln!(summary_csv, "dead_ends,{}", summary.dead_end_count)?;
    match summary.diameter {
        Some(diameter) => writeln!(summary_csv, "diameter,{diameter}")?,
        None => writeln!(summary_csv, "diameter,inf")?,
    }
    summary_csv.flush()?;

    let mut histogram = create_csv(&output.join("h_star_histogram.csv"))?;
    writeln!(histogram, "h_star,state_count")?;
    for (distance, count) in summary.h_star_histogram {
        writeln!(histogram, "{distance},{count}")?;
    }
    if summary.dead_end_count > 0 {
        writeln!(histogram, "inf,{}", summary.dead_end_count)?;
    }
    histogram.flush()
}

fn run_enumeration(arguments: &EnumerateCli) -> std::io::Result<()> {
    let task: TaskRef<'static> = Arc::new(load_enumeration_task(arguments)?);
    let start = std::time::Instant::now();
    let graph = enumerate_state_space(
        task,
        EnumerationLimits {
            max_states: arguments.max_states,
            max_transitions: arguments.max_transitions,
            max_time: arguments.max_time,
        },
    )
    .map_err(|error| std::io::Error::other(error.to_string()))?;
    let elapsed = start.elapsed();
    let summary = graph.summary();
    write_state_space_csv(&arguments.output, &graph)?;
    println!("states={}", summary.state_count);
    println!("transitions={}", summary.transition_count);
    println!("goal_states={}", summary.goal_state_count);
    println!("dead_ends={}", summary.dead_end_count);
    match summary.diameter {
        Some(diameter) => println!("diameter={diameter}"),
        None => println!("diameter=inf"),
    }
    println!("h_star_histogram={:?}", summary.h_star_histogram);
    println!("elapsed_seconds={:.6}", elapsed.as_secs_f64());
    println!("output={}", arguments.output.display());
    Ok(())
}

pub fn run_internal(cli: &PlannersCli) -> std::io::Result<SearchResult> {
    assert!(
        cli.command.is_none(),
        "subcommands are dispatched before the search pipeline"
    );
    register_event_handlers();
    planforge_searcher::preflight_required_backends(&cli.search)?;

    let start_time = std::time::Instant::now();
    let (mut task, sas_label) = if cli.inputs.len() == 2 {
        let domain = &cli.inputs[0];
        let problem = &cli.inputs[1];
        // The default path: the translation hands over the task it built, with
        // no SAS+ text in between.
        (
            planforge_translate::translate_to_task(domain, problem)
                .map_err(|err| std::io::Error::other(err.to_string()))?,
            format!("{domain} + {problem} (in-memory)"),
        )
    } else {
        let path = cli.inputs[0].clone();
        let task = NumericRootTask::try_from_file(&path).map_err(std::io::Error::other)?;
        (task, path)
    };
    if cli.restrict_task {
        let original_numeric_count = task.numeric_variables().len();
        let restricted_task = if cli.search.contains_call("icaps26_cartesian") {
            build_icaps26_restricted_task(&task)
        } else {
            build_restricted_task(&task)
        };
        if let Some(restricted_task) = restricted_task.map_err(|err| {
            std::io::Error::other(format!("failed to build restricted task: {err:#}"))
        })? {
            task = restricted_task.into_task();
            info!(
                "restricted task: numeric variables {} -> {}",
                original_numeric_count,
                task.numeric_variables().len()
            );
        }
    }
    // Captured before the task is type-erased into a `TaskRef`: the global
    // constraint is only reachable on the concrete root task, and the gradient
    // engine's verifier requires it.
    #[cfg_attr(not(feature = "sgd"), allow(unused_variables))]
    let global_constraint = *task.global_constraint();
    let task: TaskRef<'static> = Arc::new(task);
    let parse_time = start_time.elapsed();
    info!("Parsed numeric SAS output in: {:?}", parse_time);

    if matches!(cli.search, SearchSpec::Sgd(_)) {
        info!("=== Gradient Plan Synthesis ===");
    } else {
        info!("=== Search Engine ===");
    }
    info!("File: {}", sas_label);
    info!(
        "Variables: {} regular, {} numeric",
        task.variables().len(),
        task.numeric_variables().len()
    );

    let time_limit = cli.max_time;
    let memory_limit = cli.max_memory;
    let result = match &cli.search {
        SearchSpec::Astar(_, _) | SearchSpec::Gbfs(_) | SearchSpec::AstarFs(_, _) => {
            solve_task(task.clone(), &cli.search, time_limit, memory_limit)?
        }
        #[cfg(feature = "sgd")]
        SearchSpec::Sgd(args) => {
            planforge_searcher::sgd::run_sgd(task.clone(), global_constraint, args)?
        }
        #[cfg(not(feature = "sgd"))]
        SearchSpec::Sgd(_) => {
            return Err(std::io::Error::other(
                "`sgd(...)` requires the `sgd` cargo feature; rebuild with \
                 `cargo build --release -p planforge --features sgd`",
            ));
        }
    };

    write_plan_file(&result)?;
    // The gradient engine reports no search statistics, so it gets the plan
    // block only.
    if matches!(cli.search, SearchSpec::Sgd(_)) {
        print_plan_result(&result);
    } else {
        print_search_result(&result);
    }

    Ok(result)
}

fn build_heuristic_from_spec<'a>(
    spec: &HeuristicSpec,
    task_ref: &'a dyn AbstractNumericTask,
    sampling_task: TaskRef<'a>,
) -> std::io::Result<Option<Box<dyn planforge_search::evaluation::Heuristic + 'a>>> {
    planforge_search::heuristic_factory::build_heuristic_from_spec(spec, task_ref, sampling_task)
        .map_err(HeuristicBuildError::into_io_error)
}
