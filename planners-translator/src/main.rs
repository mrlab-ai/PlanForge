use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};

use log::info;
use planners_translate::normalize;
use planners_translate::pddl_parser::PddlTask;
use planners_translator::init_logger;
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
        #[arg(long = "log-level")]
        log_level: Option<log::LevelFilter>,
    },
    // /// Preprocess: read SAS+ from stdin and write a preprocessed search input (writes to stdout or file)
    // Preprocess {
    //     /// Optional input file (default: stdin)
    //     input: Option<PathBuf>,
    //     /// Optional output file (default: output)
    //     #[clap(short, long)]
    //     output: Option<PathBuf>,
    // },
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
            init_logger(log_level.unwrap_or(log::LevelFilter::Info))
                .expect("Error initialising logging");

            let start = Instant::now();
            info!(
                "translator: reading domain {:?} and problem {:?}",
                domain, problem
            );
            let task = PddlTask::from_files(&domain, &problem).map_err(|e| anyhow::anyhow!(e))?;
            info!(
                "translator: parsed forms: {} domain / {} problem",
                task.domain_forms.len(),
                task.problem_forms.len()
            );
            let parsed_task = task.to_task();

            // Create normalizable task and run normalization
            info!("translator: normalizing task...");
            let mut norm_task = normalize::NormalizableTask::from_task(parsed_task);
            norm_task.add_global_constraints();
            normalize::normalize(&mut norm_task).expect("normalization failed");
            info!(
                "translator: normalized - {} actions, {} axioms, {} numeric axioms",
                norm_task.task.actions.len(),
                norm_task.task.axioms.len(),
                norm_task.task.function_administrator.axioms.len()
            );
            // Debug: print axioms
            for (i, ax) in norm_task.task.axioms.iter().enumerate() {
                info!(
                    "  axiom[{}]: name={}, condition={:?}",
                    i, ax.name, ax.condition
                );
            }
            info!("  goal={:?}", norm_task.goal);

            // Run instantiation (Phase 1: model-guided grounding)
            // Use the normalized task for proper exploration rule generation
            info!("\ntranslator: running instantiation...");
            let result = planners_translate::instantiate::explore_normalized(&norm_task)
                .map_err(|e| anyhow::anyhow!(e))?;
            info!(
                "translator: instantiated {} grounded operators (model-guided)",
                result.grounded_ops.len()
            );
            info!(
                "translator: relaxed reachable: {}",
                result.relaxed_reachable
            );
            info!("translator: atoms: {}", result.atoms.len());

            // Debug: print action breakdown
            info!("\nAction breakdown:");
            let mut action_counts: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for op in &result.grounded_ops {
                let action_type = op.name.split('(').next().unwrap_or("unknown");
                *action_counts.entry(action_type.to_string()).or_insert(0) += 1;
            }
            for (action_type, count) in action_counts.iter() {
                info!("  {}: {}", action_type, count);
            }

            info!("\nFirst 20 grounded actions:");
            for (i, op) in result.grounded_ops.iter().take(20).enumerate() {
                info!("  {}: {}", i + 1, op.name);
            }

            // Build SAS task
            info!("\ntranslator: building SAS task...");

            // Use the instantiated numeric axioms from the model-guided grounding
            // These are the 60+ axioms that were instantiated from the 8 templates
            info!(
                "translator: processing {} instantiated numeric axioms from model",
                result.numeric_axioms.len()
            );
            let instantiated_num_axioms = result.numeric_axioms;

            let py_groups: Option<Vec<Vec<String>>> = None;
            let mut sastask = planners_translate::translate::translate_task_from_grounded_internal(
                &result.atoms,
                &result.grounded_ops,
                &task.domain_forms,
                &task.problem_forms,
                &result.num_fluents,
                &instantiated_num_axioms,
                py_groups,
                &result.grounded_axioms,
                &result.reachable_action_params,
                &norm_task.goal,
                &norm_task,
            )
            .map_err(|err| anyhow::anyhow!(err))?;
            match planners_translate::simplify::filter_unreachable_propositions(&mut sastask) {
                Ok(()) => {
                    info!("translator: simplified task");
                }
                Err(planners_translate::simplify::SimplifyError::Impossible) => {
                    info!("translator: task simplified to unsolvable");
                    sastask = planners_translate::simplify::trivial_task(false);
                }
                Err(planners_translate::simplify::SimplifyError::TriviallySolvable) => {
                    info!("translator: task simplified to trivially solvable");
                    sastask = planners_translate::simplify::trivial_task(true);
                }
                Err(planners_translate::simplify::SimplifyError::DoesNothing) => {
                    info!("translator: simplification made no changes");
                }
            }
            let out_path = output.unwrap_or_else(|| PathBuf::from("output.sas"));
            let py_task = planners_translate::sas_tasks::from_internal(&sastask);
            let mut out_file = std::fs::File::create(&out_path)?;
            py_task.output(&mut out_file)?;
            info!("translator: wrote {}", out_path.display());

            let duration = start.elapsed();
            info!("translator: completed in {:.2?} seconds", duration);
        }
    }

    Ok(())
}
