use clap::Parser;
use planners::*;
use planners_searcher::exit_code_for_search_status;

fn main() -> std::io::Result<()> {
    let cli = PlannersCli::parse();
    #[cfg(unix)]
    if !cli.internal_run {
        return run_wrapped_process(&cli);
    }

    let result = run_internal(&cli)?;
    std::process::exit(exit_code_for_search_status(&result.status));
}
