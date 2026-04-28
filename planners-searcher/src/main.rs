use clap::Parser;
use planners_searcher::exit_code_for_search_status;
use planners_searcher::*;

fn main() -> std::io::Result<()> {
    let cli = PlannersSearcherCli::parse();
    init_logger(cli.log_level.unwrap_or(tracing_subscriber::filter::LevelFilter::INFO));
    #[cfg(unix)]
    if !cli.internal_run {
        return run_wrapped_process(&cli);
    }

    let result = run_internal(&cli)?;
    std::process::exit(exit_code_for_search_status(&result.status));
}
