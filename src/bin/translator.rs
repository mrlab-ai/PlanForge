use std::fs::File;
use std::io::{self, Read, Write};
use std::path::PathBuf;

use clap::{Parser, Subcommand};

use planners::translate::pddl::PddlTask;
use serde_json;

/// Minimal translator CLI for numeric PDDL -> SAS+ pipeline (placeholder)
#[derive(Parser)]
#[clap(name = "translator", version = "0.1", about = "Rust translator for numeric PDDL (minimal stub)")]
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
    /// Transform a simple numeric SAS+ task into a restricted numeric task
    RestrictedTransform {
        /// SAS input file
        input: PathBuf,
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
        Commands::Translate { domain, problem, output } => {
            eprintln!("translator: reading domain {:?} and problem {:?}", domain, problem);
            let task = PddlTask::from_files(&domain, &problem)?;
            eprintln!("translator: parsed forms: {} domain / {} problem", task.domain_forms.len(), task.problem_forms.len());
            let dom = planners::translate::pddl_ast::Domain::from_sexprs(&task.domain_forms).expect("domain parse");
            let prob = planners::translate::pddl_ast::Problem::from_sexprs(&task.problem_forms).expect("problem parse");
            let (ops, instantiated_num_axioms) = planners::translate::instantiate::ground_with_numeric_axioms(&dom, &prob);
            eprintln!("translator: grounded {} operators", ops.len());
            // attempt to call the python grouping helper to compute exact mutex groups
            let mut py_groups: Option<Vec<Vec<String>>> = None;
            if let Ok(output) = std::process::Command::new("python3")
                .arg("scripts/compute_groups.py")
                .arg(domain.as_os_str())
                .arg(problem.as_os_str())
                .output() {
                if output.status.success() {
                    if let Ok(s) = String::from_utf8(output.stdout) {
                        if let Ok(j) = serde_json::from_str::<Vec<Vec<String>>>(&s) {
                            py_groups = Some(j);
                        }
                    }
                }
            }
            let sastask = planners::translate::to_sas::build_sas(&ops, &prob, &instantiated_num_axioms, py_groups);
            let out_path = output.unwrap_or_else(|| PathBuf::from("output.sas"));
            planners::translate::sas_writer::write_sas(&sastask, &out_path)?;
            eprintln!("translator: wrote {}", out_path.display());
        }
        Commands::RestrictedTransform { input, output } => {
            eprintln!("restricted transform: reading {:?}", input);
            let mut s = String::new();
            File::open(&input)?.read_to_string(&mut s)?;
            let out_path = output.unwrap_or_else(|| PathBuf::from("output.sas"));
            let mut f = File::create(&out_path)?;
            writeln!(f, "# restricted-transform placeholder")?;
            writeln!(f, "# original-size: {}", s.len())?;
            eprintln!("restricted transform: wrote {}", out_path.display());
        }
        Commands::Preprocess { output } => {
            eprintln!("preprocess: reading SAS+ from stdin");
            let mut stdin = String::new();
            io::stdin().read_to_string(&mut stdin)?;
            let out_path = output.unwrap_or_else(|| PathBuf::from("output"));
            let mut f = File::create(&out_path)?;
            // In a real implementation, preprocessing would convert the SAS+ into
            // the binary 'output' directory format. Here we pass-through.
            f.write_all(stdin.as_bytes())?;
            eprintln!("preprocess: wrote {}", out_path.display());
        }
    }

    Ok(())
}
