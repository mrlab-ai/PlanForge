use std::process::ExitCode;

fn main() -> ExitCode {
    match planforge::run_with(planforge_heuristics::features()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
