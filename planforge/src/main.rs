use clap::Parser;
use planforge::*;
use planforge_searcher::exit_code_for_search_status;

fn main() -> std::io::Result<()> {
    let cli = PlannersCli::parse();
    init_logger(
        cli.log_level
            .unwrap_or(tracing_subscriber::filter::LevelFilter::INFO),
    );
    // Reserve a memory padding mirroring numeric-FD's
    // `reserve_extra_memory_padding`. CEGAR's collection generator polls
    // `memory_padding::poll_and_release_if_exceeded` once per abstraction
    // and stops cleanly if the RSS limit is exceeded.
    //
    // Pass `--max-memory` so the padding's release threshold sits ~10%
    // below the CLI ceiling. Without this, the default release threshold
    // can sit at or slightly above an external slurm/cgroup limit (e.g.
    // `memory_per_cpu=8300M` with `--max-memory 8G`) and slurm's OOM
    // killer fires between two CEGAR polls instead of the planner
    // exiting cleanly.
    planforge_search::numeric::evaluation::domain_abstractions::memory_padding::reserve_memory_padding(cli.max_memory);
    #[cfg(unix)]
    if !cli.internal_run {
        return run_wrapped_process(&cli);
    }

    let result = run_internal(&cli)?;
    std::process::exit(exit_code_for_search_status(&result.status));
}
