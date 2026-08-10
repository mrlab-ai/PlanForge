use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Parser, Subcommand};

use planforge_translator::{init_logger, translate_to_sas_to_path};
use tracing::info;

/// The translator's entry points name files, and a path that is not valid
/// Unicode is bad user input rather than a broken invariant.
fn as_str(path: &Path) -> anyhow::Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow::anyhow!("path must be valid Unicode: {}", path.display()))
}

/// CLI for the numeric PDDL to SAS+ pipeline.
#[derive(Parser)]
#[clap(
    name = "translator",
    version = "0.1",
    about = "Rust translator for numeric PDDL (minimal stub)"
)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Translate DOMAIN PDDL and PROBLEM PDDL into SAS+ (writes output.sas)
    Translate {
        /// Domain PDDL file
        domain: PathBuf,
        /// Problem PDDL file
        problem: PathBuf,
        /// Optional output file (default: output.sas)
        #[clap(short, long)]
        output: Option<PathBuf>,
        #[arg(long = "log-level")]
        log_level: Option<tracing_subscriber::filter::LevelFilter>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Translate {
            domain,
            problem,
            output,
            log_level,
        } => {
            init_logger(log_level.unwrap_or(tracing_subscriber::filter::LevelFilter::INFO));

            let start = Instant::now();
            let out_path = output.unwrap_or_else(|| PathBuf::from("output.sas"));
            translate_to_sas_to_path(as_str(&domain)?, as_str(&problem)?, &out_path)?;
            info!(
                "translator: wrote {} in {:.2?}",
                out_path.display(),
                start.elapsed()
            );
        }
    }

    Ok(())
}
