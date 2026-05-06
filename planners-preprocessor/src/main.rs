use clap::Parser;
use planners_preprocess::run_preprocess;
use planners_preprocessor::{PlannersPreprocessorCli, init_logger};
use planners_translator::translate_to_sas;
use tracing::info;

fn main() -> std::io::Result<()> {
    let cli = PlannersPreprocessorCli::parse();

    init_logger(
        cli.log_level
            .unwrap_or(tracing_subscriber::filter::LevelFilter::INFO),
    );
    if cli.inputs.len() == 1 {
        run_preprocess(&[cli.inputs[0].to_string(), "output.sas".to_string()]);
    } else if cli.inputs.len() == 2 {
        let domain = &cli.inputs[0];
        let problem = &cli.inputs[1];
        translate_to_sas(domain, problem).map_err(|err| std::io::Error::other(err.to_string()))?;

        run_preprocess(&["preprocess".to_string(), "output.sas".to_string()]);
    }

    let start_time = std::time::Instant::now();
    let parse_time = start_time.elapsed();
    info!("Parsed numeric SAS output in: {:?}", parse_time);

    Ok(())
}
