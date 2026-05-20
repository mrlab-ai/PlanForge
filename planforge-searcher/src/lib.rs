use clap::Parser;
use tracing::{error, info};
use planforge_cli_utils::*;
use planforge_sas::numeric::axioms::AxiomEvaluator;
use planforge_sas::numeric::numeric_task::{AbstractNumericTask, NumericRootTask};
use planforge_sas::numeric::state_registry::StateRegistry;
use planforge_sas::numeric::utils::int_packer::IntDoublePacker;
use planforge_search::numeric::evaluation::domain_abstractions::cegar::CegarConfig;
use planforge_search::numeric::evaluation::domain_abstractions::canonical_domain_abstraction_heuristic::CanonicalDomainAbstractionHeuristic;
use planforge_search::numeric::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegar,
};
use planforge_search::numeric::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstractionGenerator;
use planforge_search::numeric::evaluation::domain_abstractions::domain_abstraction_heuristic::DomainAbstractionHeuristic;
use planforge_search::numeric::evaluation::domain_abstractions::max_domain_abstraction_heuristic::MaxDomainAbstractionHeuristic;
use planforge_search::numeric::evaluation::domain_abstractions::posthoc_optimization_heuristic::PostHocOptimizationHeuristic;
use planforge_search::numeric::evaluation::domain_abstractions::saturated_cost_partitioning_online_heuristic::{
    FillScpHeuristic, SaturatedCostPartitioningOnlineHeuristic,
};
use planforge_search::numeric::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LandmarkCutNumericHeuristic;
use planforge_search::numeric::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbHeuristic;
use planforge_search::numeric::evaluation::pattern_databases::pattern_generator_systematic::{
    SystematicPatternGeneratorConfig, generate_systematic_patterns,
};
use planforge_search::numeric::evaluation::pattern_databases::pdb_collection::PdbCollection;
use planforge_search::numeric::evaluation::pattern_databases::pdb_heuristic::GreedyNumericPdbHeuristic;
use planforge_search::numeric::search_engine::{
    AStarSearch, SearchEngine, SearchResult, SearchStatus,
};
use std::ffi::OsString;
use std::fs;
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::Command;
use std::time::Duration;
use std::num::NonZero;
use time::format_description::well_known::iso8601::{Config, TimePrecision};
use tracing_subscriber::fmt::time::UtcTime;
use tracing_subscriber::prelude::*;

pub mod recursive_config;

pub use recursive_config::{HeuristicSpec, SearchSpec, parse_search_spec};

use planforge_search::numeric::evaluation::Heuristic;

/// Build a heuristic from a parsed `HeuristicSpec`. Used by both this crate's
/// `run()` and by the top-level `planforge` binary.
pub fn build_heuristic_from_spec<'a>(
    spec: &HeuristicSpec,
    task: &'a dyn AbstractNumericTask,
) -> Result<Option<Box<dyn Heuristic + 'a>>, String> {
    match spec.name.as_str() {
        "blind" => {
            if !spec.args.is_empty() {
                return Err("`blind` does not accept arguments".to_string());
            }
            Ok(None)
        }
        "ff" => {
            if !spec.args.is_empty() {
                return Err("`ff` does not accept arguments".to_string());
            }
            let h =
                planforge_search::numeric::evaluation::ff_heuristic::FfHeuristic::new(task)
                    .map_err(|e| format!("failed to construct ff heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "domain_abstraction" => {
            info!("Building domain abstraction (CEGAR)...");
            let mut cfg = CegarConfig::default();
            recursive_config::apply_da_options(&mut cfg, &spec.args)?;
            // Single DA reads only the distance table; footprints are
            // SCP-specific. Skip the per-concrete-op StateRegion cost.
            cfg.compute_operator_footprints = false;
            let generator = DomainAbstractionGenerator::new(cfg)
                .map_err(|e| format!("failed to construct DomainAbstractionGenerator: {e:#}"))?;
            let abstraction = generator
                .generate(task)
                .map_err(|e| format!("failed to build domain abstraction: {e:#}"))?;
            Ok(Some(Box::new(DomainAbstractionHeuristic::new(None, abstraction))
                as Box<dyn Heuristic + 'a>))
        }
        "canonical_domain_abstractions" => {
            let mut cfg = Default::default();
            recursive_config::apply_da_collection_options(&mut cfg, &spec.args)?;
            // Canonical never consumes operator footprints — skip ~12 GB of
            // per-concrete-op StateRegion storage on big tasks.
            cfg.compute_operator_footprints = false;
            let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(cfg);
            info!("Building canonical domain abstractions (CEGAR)...");
            let abstractions = generator
                .generate_collection(task)
                .map_err(|e| format!("failed to build canonical domain abstractions: {e:#}"))?;
            let h = CanonicalDomainAbstractionHeuristic::new(None, task, abstractions)
                .map_err(|e| {
                    format!("failed to construct canonical domain abstraction heuristic: {e}")
                })?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "multi_domain_abstractions" => {
            let mut cfg = Default::default();
            recursive_config::apply_da_collection_options(&mut cfg, &spec.args)?;
            cfg.compute_operator_footprints = false;
            let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(cfg);
            info!("Building multiple domain abstractions (CEGAR)...");
            let abstractions = generator
                .generate_collection(task)
                .map_err(|e| format!("failed to build multi domain abstractions: {e:#}"))?;
            Ok(Some(Box::new(MaxDomainAbstractionHeuristic::new(None, abstractions))
                as Box<dyn Heuristic + 'a>))
        }
        "posthoc_optimization" | "pho" => {
            let mut cfg = Default::default();
            recursive_config::apply_da_collection_options(&mut cfg, &spec.args)?;
            cfg.compute_operator_footprints = false;
            let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(cfg);
            info!("Building posthoc_optimization domain abstractions (CEGAR)...");
            let abstractions = generator.generate_collection(task).map_err(|e| {
                format!("failed to build posthoc_optimization domain abstractions: {e:#}")
            })?;
            let h = PostHocOptimizationHeuristic::new(None, task, abstractions)
                .map_err(|e| format!("failed to construct posthoc_optimization heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "scp_online" => {
            let mut cfg = planforge_search::numeric::evaluation::domain_abstractions::saturated_cost_partitioning_online_heuristic::ScpOnlineConfig::default();
            recursive_config::apply_scp_online_options(&mut cfg, &spec.args)?;
            let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(
                cfg.collection_config.clone(),
            );
            info!("Building scp_online domain abstractions (CEGAR)...");
            let abstractions = generator
                .generate_collection(task)
                .map_err(|e| format!("failed to build scp_online domain abstractions: {e:#}"))?;
            let pdbs = if cfg.use_numeric_pdbs {
                info!("Building scp_online systematic numeric PDBs...");
                let patterns = generate_systematic_patterns(
                    task,
                    SystematicPatternGeneratorConfig {
                        max_pdb_states: cfg.max_pdb_states,
                        max_pattern_size: cfg.max_pattern_size,
                        only_interesting_patterns: cfg.only_interesting_patterns,
                    },
                );
                PdbCollection::with_heuristic_config(
                    task,
                    patterns,
                    cfg.max_pdb_states,
                    cfg.pdb_heuristic_config(),
                )
                .map_err(|e| format!("failed to build scp_online numeric PDBs: {e}"))?
                .into_pdbs()
            } else {
                Vec::new()
            };
            let h = SaturatedCostPartitioningOnlineHeuristic::new(None, abstractions, pdbs, cfg, task)
                .map_err(|e| format!("failed to construct scp_online heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "fillscp" | "fill_scp" => {
            let mut cfg = planforge_search::numeric::evaluation::domain_abstractions::saturated_cost_partitioning_online_heuristic::FillScpConfig::default();
            recursive_config::apply_fill_scp_options(&mut cfg, &spec.args)?;
            cfg.force_full_goal_tasks();
            let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(
                cfg.collection_config.clone(),
            );
            info!("Building fillSCP domain abstractions (CEGAR)...");
            let abstractions = generator
                .generate_collection(task)
                .map_err(|e| format!("failed to build fillSCP domain abstractions: {e:#}"))?;
            let h = FillScpHeuristic::new(None, abstractions, cfg, task)
                .map_err(|e| format!("failed to construct fillSCP heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "greedy_numeric_pdb" => {
            let mut cfg = planforge_search::numeric::evaluation::pattern_databases::pattern_generator_greedy::GreedyPatternGeneratorConfig::default();
            recursive_config::apply_greedy_pdb_options(&mut cfg, &spec.args)?;
            let h = GreedyNumericPdbHeuristic::new(task, cfg)
                .map_err(|e| format!("failed to build greedy numeric pdb heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "canonical_numeric_pdb" => {
            let mut cfg = planforge_search::numeric::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbConfig::default();
            recursive_config::apply_canonical_pdb_options(&mut cfg, &spec.args)?;
            let h = CanonicalNumericPdbHeuristic::from_config(task, cfg)
                .map_err(|e| format!("failed to build canonical numeric pdb heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        "lmcutnumeric" => {
            let mut cfg = planforge_search::numeric::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig::default();
            recursive_config::apply_lmcut_options(&mut cfg, &spec.args)?;
            let h = LandmarkCutNumericHeuristic::from_config(task, cfg)
                .map_err(|e| format!("failed to build lmcutnumeric heuristic: {e}"))?;
            Ok(Some(Box::new(h) as Box<dyn Heuristic + 'a>))
        }
        other => Err(format!("unknown heuristic `{other}`")),
    }
}

use tracing_subscriber::filter::LevelFilter;

pub fn init_logger(level: LevelFilter) {
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
    // Layer for stdout (info + deubg + trace)
    let stdout_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_target(false)
        .with_timer(timer)
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
#[command(author, version, about = "Numeric planner")]
pub struct PlannersSearcherCli {
    #[arg(long = "max-memory", value_name = "SIZE", value_parser = parse_memory_limit)]
    pub max_memory: Option<u64>,

    #[arg(long = "max-time", value_name = "DURATION", value_parser = parse_time_limit)]
    pub max_time: Option<Duration>,

    #[arg(long = "log-level")]
    pub log_level: Option<LevelFilter>,

    #[arg(long, hide = true)]
    pub internal_run: bool,

    /// Recursive search configuration.
    /// Examples: `astar(blind())`, `astar(domain_abstraction())`, `da_debug()`.
    #[arg(
        long,
        value_name = "SPEC",
        default_value = "astar(blind())",
        value_parser = crate::recursive_config::parse_search_spec
    )]
    pub search: crate::recursive_config::SearchSpec,

    #[arg(value_name = "SAS_FILE", required = true)]
    pub sas_file: String,
}

#[cfg(unix)]
pub fn run_wrapped_process(cli: &PlannersSearcherCli) -> std::io::Result<()> {
    let current_executable = std::env::current_exe()?;
    let mut child_args = vec![OsString::from("--internal-run")];
    // Preserve the selected search configuration when re-executing ourselves.
    if let Some(level) = cli.log_level {
        child_args.push(OsString::from("--log-level"));
        child_args.push(OsString::from(level.to_string()));
    }
    child_args.push(OsString::from("--search"));
    child_args.push(OsString::from(cli.search.to_string()));
    child_args.extend([cli.sas_file.clone()].iter().map(OsString::from));

    let time_limit = cli.max_time;
    let memory_limit = cli.max_memory;

    let mut command = Command::new(current_executable);
    command.args(child_args);
    command.stdin(std::process::Stdio::inherit());
    command.stdout(std::process::Stdio::inherit());
    command.stderr(std::process::Stdio::inherit());

    unsafe {
        command.pre_exec(move || apply_process_limits(time_limit, memory_limit));
    }

    let status = command.status()?;
    let exit_code = normalize_wrapped_exit(status, time_limit, memory_limit);

    std::process::exit(exit_code)
}

#[allow(clippy::field_reassign_with_default)]
pub fn run_internal(cli: &PlannersSearcherCli) -> std::io::Result<SearchResult> {
    register_event_handlers();

    let sas_file = &cli.sas_file;

    let start_time = std::time::Instant::now();
    let task = NumericRootTask::from_file(sas_file);
    let parse_time = start_time.elapsed();
    info!("Parsed numeric SAS output in: {:?}", parse_time);

    info!("=== Search Engine ===");
    info!("File: {}", sas_file);
    info!(
        "Variables: {} regular, {} numeric",
        task.variables().len(),
        task.numeric_variables().len()
    );

    let state_packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
    let state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);

    // Both A* and GBFS go through identical heuristic construction; only the
    // open-list priority differs. Project the search spec onto (heuristic,
    // priority kind) and let the shared block below build the heuristic.
    let (heuristic_spec, gbfs_priority) = match &cli.search {
        crate::recursive_config::SearchSpec::Astar(h) => (h, false),
        crate::recursive_config::SearchSpec::Gbfs(h) => (h, true),
        crate::recursive_config::SearchSpec::DaDebug => {
            return Err(std::io::Error::other(
                "`da_debug()` is implemented in the `planforge` binary path, not `planforge-searcher`",
            ));
        }
        crate::recursive_config::SearchSpec::AstarDaDebug => {
            return Err(std::io::Error::other(
                "`astar_da_debug()` is implemented in the `planforge` binary path, not `planforge-searcher`",
            ));
        }
        crate::recursive_config::SearchSpec::AstarFs(_, _) => {
            return Err(std::io::Error::other(
                "`astar_fs(...)` is implemented in the `planforge` binary path, not `planforge-searcher`",
            ));
        }
    };
    let result = {
        {
            let task_ref: &dyn AbstractNumericTask = &task;
            let heuristic_override =
                build_heuristic_from_spec(heuristic_spec, task_ref).map_err(std::io::Error::other)?;

            let time_limit = if cli.internal_run { None } else { cli.max_time };
            let memory_limit = if cli.internal_run { None } else { cli.max_memory };
            let mut search = if gbfs_priority {
                AStarSearch::new_gbfs(
                    task_ref,
                    state_registry,
                    heuristic_override,
                    time_limit,
                    memory_limit,
                )
            } else {
                AStarSearch::new(
                    task_ref,
                    state_registry,
                    heuristic_override,
                    time_limit,
                    memory_limit,
                )
            };

            info!(
                "Starting {} search with {:?}...",
                if gbfs_priority { "GBFS" } else { "A*" },
                heuristic_spec,
            );
            search.search()
        }
    };

    print_search_result(&result);

    Ok(result)
}

pub fn exit_code_for_search_status(status: &SearchStatus) -> i32 {
    match status {
        SearchStatus::Timeout => EXIT_TIMEOUT,
        SearchStatus::MemoryLimitReached => EXIT_OUT_OF_MEMORY,
        SearchStatus::InProgress | SearchStatus::Solved(_) | SearchStatus::Failed => EXIT_SUCCESS,
    }
}

pub fn print_search_result(result: &SearchResult) {
    match result.status {
        SearchStatus::Solved(_) => {
            info!("Solution found!");
            if let Some(plan) = result.plan.as_ref() {
                let plan_cost = result
                    .solution_cost
                    .unwrap_or_else(|| plan.iter().map(|op| op.cost() as f64).sum());

                let mut plan_content = String::new();
                for op in plan.iter() {
                    plan_content.push_str(&format!("({})\n", op.name()));
                }

                match fs::write("sas_plan", plan_content) {
                    Ok(()) => {}
                    Err(e) => error!("Error writing plan file: {}", e),
                }

                for (i, op) in plan.iter().enumerate() {
                    info!("  {}: {}", i + 1, op.name());
                }

                info!("Plan length: {} step(s).", plan.len());
                info!("Plan cost: {:.6}", plan_cost);
            }
        }
        SearchStatus::Failed => {
            info!("No solution found");
        }
        SearchStatus::Timeout => {
            info!("Search timed out");
        }
        SearchStatus::MemoryLimitReached => {
            info!("Search stopped after reaching the memory limit");
        }
        SearchStatus::InProgress => {
            info!("Search ended in progress");
        }
    }

    // Fast Downward-style statistics block.
    info!("Expanded {} state(s).", result.nodes_expanded);
    info!("Reopened {} state(s).", result.nodes_reopened);
    info!("Evaluated {} state(s).", result.nodes_evaluated);
    info!("Evaluations: {}", result.evaluations);
    info!("Generated {} state(s).", result.nodes_generated);
    info!("Dead ends: {} state(s).", result.dead_ends);
    info!(
        "Expanded until last jump: {} state(s).",
        result.nodes_expanded_until_last_jump
    );
    info!(
        "Reopened until last jump: {} state(s).",
        result.nodes_reopened_until_last_jump
    );
    info!(
        "Evaluated until last jump: {} state(s).",
        result.nodes_evaluated_until_last_jump
    );
    info!(
        "Generated until last jump: {} state(s).",
        result.nodes_generated_until_last_jump
    );
    info!("Number of registered states: {}", result.registered_states);
    info!("Search time: {:.6}s", result.search_time.as_secs_f64());
}
