use clap::Parser;
use planforge::*;

fn main() -> std::io::Result<()> {
    let cli = PlannersCli::parse();
    init_logger(
        cli.log_level
            .unwrap_or(tracing_subscriber::filter::LevelFilter::INFO),
    );
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
