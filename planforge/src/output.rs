//! What the planner reports when a run finishes: the exit code, the `sas_plan`
//! file, and the statistics block.
//!
//! These are properties of the single entry point rather than of the search, so
//! they live next to the CLI and not in `planforge-searcher`.

use std::fs;

use planforge_cli_utils::{EXIT_OUT_OF_MEMORY, EXIT_SUCCESS, EXIT_TIMEOUT};
use planforge_search::search::{SearchResult, SearchStatus};
use tracing::info;

pub fn exit_code_for_search_status(status: &SearchStatus) -> i32 {
    match status {
        SearchStatus::Timeout => EXIT_TIMEOUT,
        SearchStatus::MemoryLimitReached => EXIT_OUT_OF_MEMORY,
        SearchStatus::InProgress | SearchStatus::Solved(_) | SearchStatus::Failed => EXIT_SUCCESS,
    }
}

/// Write the plan to `sas_plan`, if the search found one.
///
/// Kept separate from [`print_search_result`] because this can fail and the
/// failure has to reach the caller: exiting successfully with no plan on disk
/// is worse than exiting with an error.
pub fn write_plan_file(result: &SearchResult) -> std::io::Result<()> {
    let Some(plan) = result.plan.as_ref() else {
        return Ok(());
    };
    if !matches!(result.status, SearchStatus::Solved(_)) {
        return Ok(());
    }

    let mut plan_content = String::new();
    for op in plan.iter() {
        plan_content.push_str(&format!("({})\n", op.name()));
    }
    fs::write("sas_plan", plan_content)
}

pub fn print_search_result(result: &SearchResult) {
    print_plan_result(result);

    // Fast Downward-style statistics block.
    info!("Expanded {} state(s).", result.nodes_expanded);
    info!("Reopened {} state(s).", result.nodes_reopened);
    info!("Evaluated {} state(s).", result.nodes_evaluated);
    info!("Evaluations: {}", result.evaluations);
    info!("Generated {} state(s).", result.nodes_generated);
    info!("Dead ends: {} state(s).", result.dead_ends);
    info!(
        "Expanded until last jump: {} state(s).",
        result.nodes_expanded_until_last_jump
    );
    info!(
        "Reopened until last jump: {} state(s).",
        result.nodes_reopened_until_last_jump
    );
    info!(
        "Evaluated until last jump: {} state(s).",
        result.nodes_evaluated_until_last_jump
    );
    info!(
        "Generated until last jump: {} state(s).",
        result.nodes_generated_until_last_jump
    );
    info!("Number of registered states: {}", result.registered_states);
    info!("Search time: {:.6}s", result.search_time.as_secs_f64());
}

/// Print the outcome and write a found plan without claiming search statistics.
///
/// Gradient plan synthesis uses this directly because it has no search nodes,
/// frontier, generated states, or expansions.
pub fn print_plan_result(result: &SearchResult) {
    match result.status {
        SearchStatus::Solved(_) => {
            info!("Solution found!");
            if let Some(plan) = result.plan.as_ref() {
                let plan_cost = result
                    .solution_cost
                    .unwrap_or_else(|| plan.iter().map(|op| op.cost() as f64).sum());

                for (i, op) in plan.iter().enumerate() {
                    info!("  {}: {}", i + 1, op.name());
                }

                info!("Plan length: {} step(s).", plan.len());
                info!("Plan cost: {:.6}", plan_cost);
            }
        }
        SearchStatus::Failed => {
            info!("No solution found");
        }
        SearchStatus::Timeout => {
            info!("Search timed out");
        }
        SearchStatus::MemoryLimitReached => {
            info!("Search stopped after reaching the memory limit");
        }
        SearchStatus::InProgress => {
            info!("Search ended in progress");
        }
    }
}
