use std::process::ExitCode;

fn main() -> ExitCode {
    match planforge::run_with(planforge::Extensions::default()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
