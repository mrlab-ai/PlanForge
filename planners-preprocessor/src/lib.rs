use clap::Parser;

use tracing_subscriber::prelude::*;
use tracing_subscriber::filter::LevelFilter;

pub fn init_logger(level: LevelFilter) {
    // Layer for stdout (info + debug + trace)
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_target(false)
        .with_filter(level);

    // Layer for stderr (error + warn only)
    let stderr_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_target(false)
        .with_filter(LevelFilter::WARN);

    tracing_subscriber::registry()
        .with(stdout_layer)
        .with(stderr_layer)
        .init();
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Numeric planner preprocessor")]
pub struct PlannersPreprocessorCli {
    #[arg(value_name = "INPUT", required = true, num_args = 1..=2)]
    pub inputs: Vec<String>,

    #[arg(long = "log-level")]
    pub log_level: Option<LevelFilter>,
}
