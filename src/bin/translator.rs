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

            // Create normalizable task and run normalization
            eprintln!("translator: normalizing task...");
            let mut norm_task = normalize::NormalizableTask::from_ast(&dom, &prob);
            normalize::normalize(&mut norm_task).expect("normalization failed");
            eprintln!(
                "translator: normalized - {} actions, {} axioms, {} numeric axioms",
                norm_task.actions.len(),
                norm_task.axioms.len(),
                norm_task.numeric_axioms.len()
            );

            // Run instantiation (Phase 1: model-guided grounding)
            // Use the normalized task for proper exploration rule generation
            eprintln!("\ntranslator: running instantiation...");
            let result = planners::translate::instantiate::explore_normalized(&norm_task);
            eprintln!(
                "translator: instantiated {} grounded operators (model-guided)",
                result.grounded_ops.len()
            );
            eprintln!("translator: relaxed reachable: {}", result.relaxed_reachable);
            eprintln!("translator: model size: {} atoms", result.model.len());
            
            // Debug: print action breakdown
            eprintln!("\nAction breakdown:");
            let mut action_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for op in &result.grounded_ops {
                let action_type = op.name.split('(').next().unwrap_or("unknown");
                *action_counts.entry(action_type.to_string()).or_insert(0) += 1;
            }
            for (action_type, count) in action_counts.iter() {
                eprintln!("  {}: {}", action_type, count);
            }
            
            eprintln!("\nFirst 20 grounded actions:");
            for (i, op) in result.grounded_ops.iter().take(20).enumerate() {
                eprintln!("  {}: {}", i + 1, op.name);
            }

            //let py_groups: Option<Vec<Vec<String>>> = None;
            //let sastask = planners::translate::to_sas::build_sas(
            //    &ops,
            //    &dom,
            //    &prob,
            //    &instantiated_num_axioms,
            //    py_groups,
            //);
            //let out_path = output.unwrap_or_else(|| PathBuf::from("output.sas"));
            //planners::translate::sas_writer::write_sas(&sastask, &out_path)?;
            //eprintln!("translator: wrote {}", out_path.display());
        }
        Commands::Preprocess { output } => {
            //not implemented yet, raise error
            todo!("Preprocess command not implemented yet");
        }
    }

    Ok(())
}
