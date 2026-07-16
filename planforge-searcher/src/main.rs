use clap::Parser;
use planforge_searcher::exit_code_for_search_status;
use planforge_searcher::*;

fn main() -> std::io::Result<()> {
    let cli = PlannersSearcherCli::parse();
    init_logger(
        cli.log_level
            .unwrap_or(tracing_subscriber::filter::LevelFilter::INFO),
    );
    // Reserve a memory padding mirroring numeric-FD's
    // `reserve_extra_memory_padding`. CEGAR's collection generator polls
    // `memory_padding::poll_and_release_if_exceeded` once per abstraction
    // and stops cleanly if the RSS limit is exceeded. Configurable via
    // `DA_MEMORY_PADDING_MB` (default 512 MB) and `DA_MEMORY_LIMIT_MB`
    // (default derives from `--max-memory`, leaving ~10% headroom).
    planforge_search::evaluation::domain_abstractions::memory_padding::reserve_memory_padding(
        cli.max_memory,
    );
    #[cfg(unix)]
    if !cli.internal_run {
        return run_wrapped_process(&cli);
    }

    let result = run_internal(&cli)?;
    std::process::exit(exit_code_for_search_status(&result.status));
}
