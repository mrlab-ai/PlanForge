use std::num::NonZero;
use std::path::PathBuf;

use clap::Parser;
use time::format_description::well_known::iso8601::{Config, TimePrecision};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt::time::UtcTime;
use tracing_subscriber::prelude::*;

#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about = "Preprocess one SAS file into a search input. Usage: planforge-preprocess INPUT [-o OUTPUT]"
)]
struct PreprocessCli {
    /// SAS input file to preprocess.
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Preprocessed output file (default: output).
    #[arg(short, long, value_name = "OUTPUT", default_value = "output")]
    output: PathBuf,

    #[arg(long = "log-level")]
    log_level: Option<LevelFilter>,
}

fn init_logger(level: LevelFilter) {
    let timer = UtcTime::new(
        time::format_description::well_known::Iso8601::<
            {
                Config::DEFAULT
                    .set_time_precision(TimePrecision::Second {
                        decimal_digits: NonZero::new(3),
                    })
                    .encode()
            },
        >,
    );
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_target(false)
        .with_timer(timer)
        .with_filter(level);
    let stderr_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_target(false)
        .with_filter(LevelFilter::WARN);

    tracing_subscriber::registry()
        .with(stdout_layer)
        .with(stderr_layer)
        .init();
}

fn main() -> std::io::Result<()> {
    let cli = PreprocessCli::parse();
    init_logger(cli.log_level.unwrap_or(LevelFilter::INFO));

    let input = cli
        .input
        .to_str()
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "input path must be valid Unicode",
            )
        })?
        .to_owned();
    let args = ["planforge-preprocess".to_string(), input];
    planforge_translate::preprocess::run_preprocess_to_output(&args, &cli.output);
    Ok(())
}
