use clap::Parser;
use planners::*;
use planners_searcher::exit_code_for_search_status;

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
    planners_search::numeric::evaluation::domain_abstractions::memory_padding::reserve_memory_padding();
    #[cfg(unix)]
    if !cli.internal_run {
        return run_wrapped_process(&cli);
    }

    let result = run_internal(&cli)?;
    std::process::exit(exit_code_for_search_status(&result.status));
}
