use std::fs::File;
use std::io::{self, Read, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use planners::translate::pddl::PddlTask;
use planners::translate::normalize;

/// Minimal translator CLI for numeric PDDL -> SAS+ pipeline (placeholder)
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
    },
    /// Preprocess: read SAS+ from stdin and write a preprocessed search input (writes to stdout or file)
    Preprocess {
        /// Optional output file (default: output)
        #[clap(short, long)]
        output: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Translate {
            domain,
            problem,
            output,
        } => {
            eprintln!(
                "translator: reading domain {:?} and problem {:?}",
                domain, problem
            );
            let task = PddlTask::from_files(&domain, &problem)?;
            eprintln!(
                "translator: parsed forms: {} domain / {} problem",
                task.domain_forms.len(),
                task.problem_forms.len()
            );
            let dom = planners::translate::pddl_ast::Domain::from_sexprs(&task.domain_forms)
                .expect("domain parse");
            let prob = planners::translate::pddl_ast::Problem::from_sexprs(&task.problem_forms)
                .expect("problem parse");
            let (ops, instantiated_num_axioms) =
                planners::translate::instantiate::ground_with_numeric_axioms(&dom, &prob);
            eprintln!("translator: grounded {} operators", ops.len());
            let py_groups: Option<Vec<Vec<String>>> = None;
            let sastask = planners::translate::to_sas::build_sas(
                &ops,
                &dom,
                &prob,
                &instantiated_num_axioms,
                py_groups,
            );
            let out_path = output.unwrap_or_else(|| PathBuf::from("output.sas"));
            planners::translate::sas_writer::write_sas(&sastask, &out_path)?;
            eprintln!("translator: wrote {}", out_path.display());
        }
        Commands::Preprocess { output } => {
            //not implemented yet, raise error
            todo!("Preprocess command not implemented yet");
        }
    }

    Ok(())
}
