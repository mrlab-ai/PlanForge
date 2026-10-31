//! What the planner reports when a run finishes: the exit code, the `sas_plan`
//! file, and the statistics block.
//!
//! These are properties of the single entry point rather than of the search, so
//! they live next to the CLI and not in `planforge-search`.

use std::fs;

use planforge_search::search::{SearchOutcome, SearchResult};
use tracing::info;

pub const EXIT_SUCCESS: i32 = 0;
pub const EXIT_OUT_OF_MEMORY: i32 = 6;
pub const EXIT_TIMEOUT: i32 = 7;
/// The task uses a feature the configured heuristic does not support; Fast
/// Downward's `SEARCH_UNSUPPORTED`.
pub const EXIT_UNSUPPORTED: i32 = 34;

pub fn exit_code_for_search_outcome(outcome: &SearchOutcome) -> i32 {
    match outcome {
        SearchOutcome::Timeout => EXIT_TIMEOUT,
        SearchOutcome::OutOfMemory => EXIT_OUT_OF_MEMORY,
        SearchOutcome::Solved { .. } | SearchOutcome::Unsolvable => EXIT_SUCCESS,
    }
}

/// Write the plan to `sas_plan`, if the search found one.
///
/// Kept separate from [`print_search_result`] because this can fail and the
/// failure has to reach the caller: exiting successfully with no plan on disk
/// is worse than exiting with an error.
pub fn write_plan_file(result: &SearchResult) -> std::io::Result<()> {
    let SearchOutcome::Solved { plan, .. } = &result.outcome else {
        return Ok(());
    };
    let plan_content: String = plan.iter().map(|op| format!("({})\n", op.name())).collect();
    fs::write("sas_plan", plan_content)
}

/// Print the outcome and Fast Downward's statistics block.
pub fn print_search_result(result: &SearchResult) {
    match &result.outcome {
        SearchOutcome::Solved { plan, cost } => {
            info!("Solution found!");
            for (i, op) in plan.iter().enumerate() {
                info!("  {}: {}", i + 1, op.name());
            }
            info!("Plan length: {} step(s).", plan.len());
            info!("Plan cost: {cost:.6}");
        }
        SearchOutcome::Unsolvable => info!("No solution found"),
        SearchOutcome::Timeout => info!("Search timed out"),
        SearchOutcome::OutOfMemory => info!("Search stopped after reaching the memory limit"),
    }

    let statistics = &result.statistics;
    let nodes = &statistics.nodes;
    let until_last_jump = &statistics.nodes_until_last_jump;
    info!("Expanded {} state(s).", nodes.expanded);
    info!("Reopened {} state(s).", nodes.reopened);
    info!("Evaluated {} state(s).", nodes.evaluated);
    info!("Evaluations: {}", statistics.evaluations);
    info!("Generated {} state(s).", nodes.generated);
    info!("Dead ends: {} state(s).", statistics.dead_ends);
    info!(
        "Expanded until last jump: {} state(s).",
        until_last_jump.expanded
    );
    info!(
        "Reopened until last jump: {} state(s).",
        until_last_jump.reopened
    );
    info!(
        "Evaluated until last jump: {} state(s).",
        until_last_jump.evaluated
    );
    info!(
        "Generated until last jump: {} state(s).",
        until_last_jump.generated
    );
    info!(
        "Number of registered states: {}",
        statistics.registered_states
    );
    info!("Search time: {:.6}s", result.search_time.as_secs_f64());
    print_peak_memory();
}

/// Peak address space (`VmPeak`, what `--max-memory` limits; the out-of-memory
/// and signal handlers print the same line) and peak resident memory
/// (`VmHWM`) of the process that searched.
#[cfg(target_os = "linux")]
fn print_peak_memory() {
    let status = fs::read_to_string("/proc/self/status").expect("/proc/self/status is readable");
    let kb = |field: &str| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(field))
            .and_then(|value| value.split_whitespace().next())
            .unwrap_or_else(|| panic!("/proc/self/status reports {field} in KB"))
            .to_owned()
    };
    info!("Peak memory: {} KB", kb("VmPeak:"));
    info!("Peak resident memory: {} KB", kb("VmHWM:"));
}

#[cfg(not(target_os = "linux"))]
fn print_peak_memory() {}
