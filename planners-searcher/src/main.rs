use clap::Parser;
use planners_searcher::exit_code_for_search_status;
use planners_searcher::*;

fn main() -> std::io::Result<()> {
    let cli = PlannersSearcherCli::parse();
    #[cfg(unix)]
    if !cli.internal_run {
        return run_wrapped_process(&cli);
    }

    let result = run_internal(&cli)?;
    std::process::exit(exit_code_for_search_status(&result.status));
}
