use clap::Parser;

use std::io::Write;

pub fn init_logger(level: log::LevelFilter) -> Result<(), log::SetLoggerError> {
    let mut builder = env_logger::Builder::new();
    builder.filter_level(level);
    builder.format(|formatter, record| {
        writeln!(
            formatter,
            "[{}] {}: {}",
            formatter.timestamp_seconds(),
            record.level(),
            record.args()
        )
    });

    builder.try_init()
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Numeric planner preprocessor")]
pub struct PlannersPreprocessorCli {
    #[arg(value_name = "INPUT", required = true, num_args = 1..=2)]
    pub inputs: Vec<String>,

    #[arg(long = "log-level")]
    pub log_level: Option<log::LevelFilter>,
}
