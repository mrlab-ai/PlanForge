#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use clap::Parser;
use planners_preprocess::planner::run_preprocess;
use planners_sas::numeric::numeric_parser::parse_numeric_sas_output;
use planners_translate::normalize;
use planners_translate::pddl_parser::PddlTask;
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::VecDeque;
use std::ffi::OsString;
use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;
use std::time::Duration;

#[cfg(unix)]
use std::os::unix::process::{CommandExt, ExitStatusExt};

use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::AbstractNumericTask;
use planners_sas::numeric::numeric_task::NumericRootTask;
use planners_sas::numeric::numeric_task::NumericType;
use planners_sas::numeric::state_registry::StateRegistry;
use planners_sas::numeric::utils::int_packer::IntDoublePacker;
use planners_search::numeric::search_engine::{AStarSearch, SearchEngine};
use planners_search::numeric::search_engine::{SearchResult, SearchStatus};
use planners_search::numeric::successor_generator;
use planners_search::numeric::successor_generator::GroundedSuccessorGenerator;
use planners_search::numeric::successor_generator::Node;

const EXIT_SUCCESS: i32 = 0;
const EXIT_OUT_OF_MEMORY: i32 = 6;
const EXIT_TIMEOUT: i32 = 7;

#[cfg(unix)]
static OOM_REPORTED: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
struct ReportingAllocator;

#[cfg(unix)]
#[global_allocator]
static GLOBAL_ALLOCATOR: ReportingAllocator = ReportingAllocator;

#[cfg(unix)]
unsafe impl GlobalAlloc for ReportingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if ptr.is_null() {
            report_out_of_memory_and_exit();
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if ptr.is_null() {
            report_out_of_memory_and_exit();
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if new_ptr.is_null() {
            report_out_of_memory_and_exit();
        }
        new_ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Numeric planner")]
struct Cli {
    #[arg(long = "max-memory", value_name = "SIZE", value_parser = parse_memory_limit)]
    max_memory: Option<u64>,

    #[arg(long = "max-time", value_name = "DURATION", value_parser = parse_time_limit)]
    max_time: Option<Duration>,

    #[arg(long, hide = true)]
    internal_run: bool,

    #[arg(value_name = "INPUT", required = true, num_args = 1..=2)]
    inputs: Vec<String>,
}

fn parse_suffixed_value(
    input: &str,
    default_multiplier: u64,
    units: &[(&str, u64)],
    kind: &str,
) -> Result<u64, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(format!("{} cannot be empty", kind));
    }

    let suffix_start = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    if suffix_start == 0 {
        return Err(format!("{} must start with a number: {}", kind, input));
    }

    let value = trimmed[..suffix_start]
        .parse::<u64>()
        .map_err(|_| format!("invalid {} value: {}", kind, input))?;
    let suffix = trimmed[suffix_start..].trim().to_ascii_lowercase();

    let multiplier = if suffix.is_empty() {
        default_multiplier
    } else {
        units
            .iter()
            .find_map(|(unit, factor)| (*unit == suffix).then_some(*factor))
            .ok_or_else(|| format!("invalid {} suffix '{}': {}", kind, suffix, input))?
    };

    value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("{} is too large: {}", kind, input))
}

fn parse_memory_limit(input: &str) -> Result<u64, String> {
    parse_suffixed_value(
        input,
        1,
        &[
            ("b", 1),
            ("k", 1024),
            ("kb", 1024),
            ("m", 1024 * 1024),
            ("mb", 1024 * 1024),
            ("g", 1024 * 1024 * 1024),
            ("gb", 1024 * 1024 * 1024),
            ("t", 1024_u64.pow(4)),
            ("tb", 1024_u64.pow(4)),
        ][..],
        "memory limit",
    )
}

fn parse_time_limit(input: &str) -> Result<Duration, String> {
    let seconds = parse_suffixed_value(
        input,
        1,
        &[("ms", 0), ("s", 1), ("m", 60), ("h", 60 * 60)][..],
        "time limit",
    )?;

    if input.trim().to_ascii_lowercase().ends_with("ms") {
        let millis = input.trim()[..input.trim().len() - 2]
            .trim()
            .parse::<u64>()
            .map_err(|_| format!("invalid time limit value: {}", input))?;
        Ok(Duration::from_millis(millis))
    } else {
        Ok(Duration::from_secs(seconds))
    }
}

fn setup_state_registry<'a>(
    problem: &'a NumericRootTask,
    state_packer: &'a IntDoublePacker,
    axiom_evaluator: &'a AxiomEvaluator<'a>,
) -> StateRegistry<'a> {
    StateRegistry::new(problem, state_packer, axiom_evaluator)
}

fn setup_axiom_evaluator<'a>(
    problem: &'a NumericRootTask,
    state_packer: &'a IntDoublePacker,
) -> AxiomEvaluator<'a> {
    let task: &'a dyn AbstractNumericTask = problem;
    let axiom_evaluator = AxiomEvaluator::new(task, &state_packer);
    axiom_evaluator
}

fn setup_state_packer<'a>(problem: &'a NumericRootTask) -> IntDoublePacker {
    let mut domain_sizes = vec![];
    for var in problem.variables().iter() {
        domain_sizes.push(var.domain_size() as u64);
    }
    for numeric_var in problem.numeric_variables().iter() {
        if numeric_var.get_type() == &NumericType::Regular {
            domain_sizes.push(u64::MAX);
        }
    }
    IntDoublePacker::new(&domain_sizes)
}

fn setup_numeric_task(file_name: &str) -> NumericRootTask {
    // This function should create a NumericRootTask with the necessary setup for testing
    // For now, we return an empty task as a placeholder
    let file_content = std::fs::read_to_string(file_name).unwrap();
    parse_numeric_sas_output(&file_content)
        .unwrap() //TODO: Handle errors properly
        .1
}

fn setup_successor_generator<'a>(task: &'a dyn AbstractNumericTask) -> Box<dyn Node<'a> + 'a> {
    let mut queue = VecDeque::new();
    for (op_id, operator) in task.get_operators().iter().enumerate() {
        queue.push_back((operator, op_id));
    }

    let mut generator = GroundedSuccessorGenerator::new(task);

    let node = generator.construct(&mut 0, &mut queue).unwrap();

    node
}

fn translate_to_sas(domain: &str, problem: &str) -> anyhow::Result<()> {
    let task = PddlTask::from_files(std::path::Path::new(domain), std::path::Path::new(problem))
        .map_err(|e| anyhow::anyhow!(e))?;
    let parsed_task = task.to_task();

    let mut norm_task = normalize::NormalizableTask::from_task(parsed_task);
    norm_task.add_global_constraints();
    normalize::normalize(&mut norm_task).expect("normalization failed");

    let result = planners_translate::instantiate::explore_normalized(&norm_task)
        .map_err(|e| anyhow::anyhow!(e))?;

    let instantiated_num_axioms = result.numeric_axioms;
    let py_groups: Option<Vec<Vec<String>>> = None;
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
    let mut out_file = std::fs::File::create("output.sas")?;
    py_task.output(&mut out_file)?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn peak_memory_kb() -> u64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(value) = line.strip_prefix("VmPeak:") {
                if let Some(kb) = value
                    .split_whitespace()
                    .next()
                    .and_then(|part| part.parse::<u64>().ok())
                {
                    return kb;
                }
            }
        }
    }
    0
}

#[cfg(not(target_os = "linux"))]
fn peak_memory_kb() -> u64 {
    0
}

#[cfg(unix)]
fn register_event_handlers() {
    static INIT: Once = Once::new();

    INIT.call_once(|| unsafe {
        libc::signal(libc::SIGABRT, signal_handler as libc::sighandler_t);
        libc::signal(libc::SIGTERM, signal_handler as libc::sighandler_t);
        libc::signal(libc::SIGSEGV, signal_handler as libc::sighandler_t);
        libc::signal(libc::SIGINT, signal_handler as libc::sighandler_t);
        libc::signal(libc::SIGXCPU, signal_handler as libc::sighandler_t);
    });
}

#[cfg(not(unix))]
fn register_event_handlers() {}

#[cfg(unix)]
extern "C" fn signal_handler(signal_number: libc::c_int) {
    unsafe {
        print_peak_memory_reentrant(libc::STDOUT_FILENO);
        write_fd(libc::STDOUT_FILENO, b"caught signal ");
        write_number_fd(libc::STDOUT_FILENO, signal_number as u64);
        write_fd(libc::STDOUT_FILENO, b" -- exiting\n");
        libc::_exit(128 + signal_number);
    }
}

#[cfg(unix)]
unsafe fn report_out_of_memory_and_exit() -> ! {
    if OOM_REPORTED.swap(true, Ordering::SeqCst) {
        libc::_exit(6);
    }

    write_fd(libc::STDOUT_FILENO, b"Failed to allocate memory.\n");
    write_fd(libc::STDOUT_FILENO, b"Memory limit has been reached.\n");
    print_peak_memory_reentrant(libc::STDOUT_FILENO);
    libc::_exit(6)
}

#[cfg(target_os = "linux")]
unsafe fn print_peak_memory_reentrant(fd: libc::c_int) {
    let proc_fd = libc::open(c"/proc/self/status".as_ptr(), libc::O_RDONLY);
    if proc_fd < 0 {
        return;
    }

    let magic = b"VmPeak:";
    let mut matched = 0usize;
    let mut found = false;
    let mut wrote_prefix = false;
    let mut buffer = [0u8; 4096];

    loop {
        let bytes_read = libc::read(proc_fd, buffer.as_mut_ptr().cast(), buffer.len());
        if bytes_read <= 0 {
            break;
        }

        for &byte in &buffer[..bytes_read as usize] {
            if !found {
                if byte == magic[matched] {
                    matched += 1;
                    if matched == magic.len() {
                        found = true;
                    }
                } else {
                    matched = if byte == magic[0] { 1 } else { 0 };
                }
                continue;
            }

            if byte.is_ascii_digit() {
                if !wrote_prefix {
                    write_fd(fd, b"Peak memory: ");
                    wrote_prefix = true;
                }
                write_fd(fd, std::slice::from_ref(&byte));
            } else if wrote_prefix {
                write_fd(fd, b" KB\n");
                let _ = libc::close(proc_fd);
                return;
            }
        }
    }

    let _ = libc::close(proc_fd);
}

#[cfg(all(unix, not(target_os = "linux")))]
unsafe fn print_peak_memory_reentrant(_fd: libc::c_int) {}

#[cfg(unix)]
unsafe fn write_fd(fd: libc::c_int, mut bytes: &[u8]) {
    while !bytes.is_empty() {
        let written = libc::write(fd, bytes.as_ptr().cast(), bytes.len());
        if written <= 0 {
            break;
        }
        bytes = &bytes[written as usize..];
    }
}

#[cfg(unix)]
unsafe fn write_number_fd(fd: libc::c_int, value: u64) {
    let mut buffer = [0u8; 32];
    let mut index = buffer.len();
    let mut current = value;

    if current == 0 {
        write_fd(fd, b"0");
        return;
    }

    while current > 0 {
        index -= 1;
        buffer[index] = b'0' + (current % 10) as u8;
        current /= 10;
    }

    write_fd(fd, &buffer[index..]);
}

fn exit_code_for_search_status(status: &SearchStatus) -> i32 {
    match status {
        SearchStatus::Timeout => EXIT_TIMEOUT,
        SearchStatus::MemoryLimitReached => EXIT_OUT_OF_MEMORY,
        SearchStatus::InProgress | SearchStatus::Solved(_) | SearchStatus::Failed => EXIT_SUCCESS,
    }
}

fn print_search_result(result: &SearchResult) {
    match result.status {
        SearchStatus::Solved(_) => {
            println!("SOLVED!");
            if let Some(plan) = result.plan.as_ref() {
                println!("Solution plan ({} steps):", plan.len());

                let mut plan_content = String::new();
                for op in plan.iter() {
                    plan_content.push_str(&format!("({})\n", op.name()));
                }

                match fs::write("sas_plan", plan_content) {
                    Ok(()) => println!("Plan written to sas_plan file"),
                    Err(e) => eprintln!("Error writing plan file: {}", e),
                }

                for (i, op) in plan.iter().enumerate() {
                    println!("  {}: {}", i + 1, op.name());
                }
            }
        }
        SearchStatus::Failed => {
            println!("No solution found");
        }
        SearchStatus::Timeout => {
            println!("Search timed out");
        }
        SearchStatus::MemoryLimitReached => {
            println!("Search stopped after reaching the memory limit");
        }
        SearchStatus::InProgress => {
            println!("Search ended in progress");
        }
    }

    println!(
        "Statistics: {} expanded, {} generated, {:?}",
        result.nodes_expanded, result.nodes_generated, result.search_time
    );
}

#[cfg(unix)]
fn wrapper_exit_code(status: std::process::ExitStatus) -> i32 {
    status
        .code()
        .unwrap_or_else(|| status.signal().map(|signal| 128 + signal).unwrap_or(1))
}

#[cfg(unix)]
fn normalize_wrapped_exit(
    status: std::process::ExitStatus,
    time_limit: Option<Duration>,
    memory_limit: Option<u64>,
) -> i32 {
    if let Some(signal) = status.signal() {
        if signal == libc::SIGXCPU && time_limit.is_some() {
            println!("Time limit reached. Abort search.");
            return EXIT_TIMEOUT;
        }

        if memory_limit.is_some()
            && (signal == libc::SIGABRT || signal == libc::SIGSEGV || signal == libc::SIGKILL)
        {
            println!("Failed to allocate memory.");
            println!("Memory limit has been reached.");
            return EXIT_OUT_OF_MEMORY;
        }
    }

    let exit_code = wrapper_exit_code(status);

    if time_limit.is_some() && exit_code == 128 + libc::SIGXCPU {
        println!("Time limit reached. Abort search.");
        return EXIT_TIMEOUT;
    }

    if memory_limit.is_some()
        && (exit_code == 128 + libc::SIGABRT
            || exit_code == 128 + libc::SIGSEGV
            || exit_code == 128 + libc::SIGKILL)
    {
        println!("Failed to allocate memory.");
        println!("Memory limit has been reached.");
        return EXIT_OUT_OF_MEMORY;
    }

    exit_code
}

#[cfg(unix)]
fn run_wrapped_process(cli: &Cli) -> std::io::Result<()> {
    let current_executable = std::env::current_exe()?;
    let mut child_args = vec![OsString::from("--internal-run")];
    child_args.extend(cli.inputs.iter().cloned().map(OsString::from));

    let time_limit = cli.max_time;
    let memory_limit = cli.max_memory;

    let mut command = Command::new(current_executable);
    command.args(child_args);
    command.stdin(std::process::Stdio::inherit());
    command.stdout(std::process::Stdio::inherit());
    command.stderr(std::process::Stdio::inherit());

    unsafe {
        command.pre_exec(move || apply_process_limits(time_limit, memory_limit));
    }

    let status = command.status()?;
    let exit_code = normalize_wrapped_exit(status, time_limit, memory_limit);

    std::process::exit(exit_code)
}

#[cfg(unix)]
fn apply_process_limits(
    time_limit: Option<Duration>,
    memory_limit: Option<u64>,
) -> std::io::Result<()> {
    if let Some(time_limit) = time_limit {
        let mut soft_limit = time_limit.as_secs();
        if time_limit.subsec_nanos() > 0 {
            soft_limit = soft_limit.saturating_add(1);
        }
        let hard_limit = soft_limit.saturating_add(1);
        let cpu_limit = libc::rlimit {
            rlim_cur: soft_limit as libc::rlim_t,
            rlim_max: hard_limit as libc::rlim_t,
        };

        let result = unsafe { libc::setrlimit(libc::RLIMIT_CPU, &cpu_limit) };
        if result != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }

    if let Some(memory_limit) = memory_limit {
        let address_space_limit = libc::rlimit {
            rlim_cur: memory_limit as libc::rlim_t,
            rlim_max: memory_limit as libc::rlim_t,
        };

        let result = unsafe { libc::setrlimit(libc::RLIMIT_AS, &address_space_limit) };
        if result != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }

    Ok(())
}

fn run_internal(cli: &Cli) -> std::io::Result<SearchResult> {
    register_event_handlers();

    let sas_file = if cli.inputs.len() == 2 {
        let domain = &cli.inputs[0];
        let problem = &cli.inputs[1];
        translate_to_sas(domain, problem)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))?;

        run_preprocess(&vec!["preprocess".to_string(), "output.sas".to_string()]);
        "output"
    } else {
        &cli.inputs[0]
    };

    let start_time = std::time::Instant::now();
    let task = setup_numeric_task(sas_file);
    let parse_time = start_time.elapsed();
    println!("Parsed numeric SAS output in: {:?}", parse_time);

    println!("=== A* Search Engine ===");
    println!("File: {}", sas_file);
    println!(
        "Variables: {} regular, {} numeric",
        task.variables().len(),
        task.numeric_variables().len()
    );

    let state_packer = setup_state_packer(&task);
    let axiom_evaluator = setup_axiom_evaluator(&task, &state_packer);
    let state_registry = setup_state_registry(&task, &state_packer, &axiom_evaluator);

    let result = {
        let task_ref: &dyn AbstractNumericTask = &task;
        let mut search = AStarSearch::new(
            task_ref,
            state_registry,
            None,
            if cli.internal_run { None } else { cli.max_time },
            if cli.internal_run {
                None
            } else {
                cli.max_memory
            },
        );

        println!("Starting search...");
        search.search()
    };

    print_search_result(&result);

    Ok(result)
}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();
    #[cfg(unix)]
    if !cli.internal_run {
        return run_wrapped_process(&cli);
    }

    let result = run_internal(&cli)?;
    std::process::exit(exit_code_for_search_status(&result.status));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    use std::os::unix::process::ExitStatusExt;

    #[test]
    fn parses_memory_limit_suffixes() {
        assert_eq!(parse_memory_limit("500M").unwrap(), 500 * 1024 * 1024);
        assert_eq!(parse_memory_limit("8g").unwrap(), 8 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_limit("1024").unwrap(), 1024);
    }

    #[test]
    fn parses_time_limit_suffixes() {
        assert_eq!(parse_time_limit("60s").unwrap(), Duration::from_secs(60));
        assert_eq!(parse_time_limit("5m").unwrap(), Duration::from_secs(300));
        assert_eq!(
            parse_time_limit("250ms").unwrap(),
            Duration::from_millis(250)
        );
    }

    #[test]
    fn maps_search_statuses_to_exit_codes() {
        assert_eq!(
            exit_code_for_search_status(&SearchStatus::Solved(0)),
            EXIT_SUCCESS
        );
        assert_eq!(
            exit_code_for_search_status(&SearchStatus::Failed),
            EXIT_SUCCESS
        );
        assert_eq!(
            exit_code_for_search_status(&SearchStatus::Timeout),
            EXIT_TIMEOUT
        );
        assert_eq!(
            exit_code_for_search_status(&SearchStatus::MemoryLimitReached),
            EXIT_OUT_OF_MEMORY
        );
    }

    #[cfg(unix)]
    #[test]
    fn normalizes_wrapped_timeout_signal() {
        let status = std::process::ExitStatus::from_raw(libc::SIGXCPU);
        assert_eq!(
            normalize_wrapped_exit(status, Some(Duration::from_secs(1)), None),
            EXIT_TIMEOUT
        );
    }

    #[cfg(unix)]
    #[test]
    fn normalizes_wrapped_memory_signal() {
        let status = std::process::ExitStatus::from_raw(libc::SIGSEGV);
        assert_eq!(
            normalize_wrapped_exit(status, None, Some(1024)),
            EXIT_OUT_OF_MEMORY
        );
    }

    #[cfg(unix)]
    #[test]
    fn normalizes_wrapped_timeout_exit_code() {
        let status = std::process::ExitStatus::from_raw((128 + libc::SIGXCPU) << 8);
        assert_eq!(
            normalize_wrapped_exit(status, Some(Duration::from_secs(1)), None),
            EXIT_TIMEOUT
        );
    }
}
