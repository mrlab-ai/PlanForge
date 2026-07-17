use clap::Parser;
use planforge::*;
use planforge_searcher::exit_code_for_search_status;

fn main() -> std::io::Result<()> {
    let cli = PlannersCli::parse();
    init_logger(
        cli.log_level
            .unwrap_or(tracing_subscriber::filter::LevelFilter::INFO),
    );
    #[cfg(unix)]
    if !cli.internal_run {
        return run_wrapped_process(&cli);
    }

    planforge_search::resource_limits::reserve_memory_padding(cli.max_memory)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;

    let result = run_internal(&cli)?;
    std::process::exit(exit_code_for_search_status(&result.status));
}
