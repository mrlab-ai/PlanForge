use std::process::ExitCode;

fn main() -> ExitCode {
    match planforge::run_with_heuristics(Vec::new()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
