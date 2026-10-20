//! The CEGAR loop that refines a domain abstraction: find flaws in the
//! abstract plan, refine, and repeat until a limit is reached or no flaw
//! remains.

#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::{AbstractNumericTask, Operator, OperatorIndex, VariableIndex};
use planforge_sas::state_registry::ConcreteStateView;
use planforge_sas::utils::state_packer::StatePacker;
use rand::{SeedableRng, rngs::SmallRng};
use tracing::{debug, info};

use super::config::CegarConfig;
use super::factory::{DistanceTableOptions, DomainAbstractionFactory, WildcardPlanResult};
use super::initial_splits::RefinementState;
use super::initial_splits::{
    apply_initial_goal_splits, sorted_goal_facts, trivial_domain_mapping_and_sizes,
};
use super::partitions::NumericPartitions;
use super::view::{DomainView, RefinementLimits};
use crate::evaluation::abstractions::common::cegar::progression::{
    ConcretePlan, PlanCheck, replay_entire_optimal_abstract_trace, replay_optimal_abstract_trace,
};
use crate::evaluation::abstractions::common::cegar::regression::{
    ensure_regressable, regression_plan_check,
};
use crate::evaluation::abstractions::common::cegar::semantics::{CegarSemantics, SplitOptions};
use crate::evaluation::abstractions::common::cegar::view::AbstractionView;
use crate::evaluation::abstractions::common::cegar::{
    CegarDriver, CegarIterationResult, CegarStopReason, FlawKind, progress_concrete_state,
};
use crate::evaluation::abstractions::common::concrete_state::{
    ConcreteState, get_initial_state, make_prop_state_packer,
};

#[derive(Debug, Clone)]
pub struct CegarState {
    pub factory: DomainAbstractionFactory,
    pub iteration: usize,
}

impl CegarState {
    pub fn new(factory: DomainAbstractionFactory, iteration: usize) -> CegarState {
        CegarState { factory, iteration }
    }
}

#[derive(Debug, Clone)]
pub struct CegarOutcome {
    pub final_state: CegarState,
    pub stop_reason: CegarStopReason,
    /// True iff CEGAR's loop exited because no flaws remained. For a
    /// standalone full-task abstraction, the abstract plan is then a real
    /// concrete plan and `h(init)` is optimal.
    pub solved_by_self: bool,
    /// The concrete plan the flaw search executed when no flaws remained;
    /// `Some` exactly when `solved_by_self`.
    pub concrete_plan_operator_ids: Option<Vec<OperatorIndex>>,
}

#[derive(Debug, Clone)]
pub struct Cegar {
    config: CegarConfig,
}

impl Cegar {
    pub fn new(config: CegarConfig) -> Result<Self> {
        ensure!(
            config.max_abstraction_size > 0,
            "max_abstraction_size must be > 0"
        );
        ensure!(config.max_iterations > 0, "max_iterations must be > 0");
        Ok(Self { config })
    }

    pub fn build_abstraction(&self, task: &dyn AbstractNumericTask) -> Result<CegarOutcome> {
        self.run_cegar(task)
    }

    fn run_cegar(&self, task: &dyn AbstractNumericTask) -> Result<CegarOutcome> {
        let config = &self.config;
        let semantics = CegarSemantics::new(
            task,
            SplitOptions {
                kind: config.split_kind(),
                selection: config.flaw_treatment.split_selection(),
                rank: config.flaw_selection_rank,
                random_seed: config.random_seed,
            },
        )?;
        if config.flaw_kind == FlawKind::Regression {
            ensure_regressable(task, semantics.coordinates())?;
        }

        let start = Instant::now();
        // The run is a function of its configuration alone: collection
        // generators hand every member its own seed.
        let mut rng = SmallRng::seed_from_u64(
            config
                .random_seed
                .unwrap_or(crate::evaluation::DEFAULT_RANDOM_SEED),
        );

        let (mut domain_mapping, mut domain_sizes) = trivial_domain_mapping_and_sizes(task)
            .context("failed to build trivial domain mapping")?;
        let mut partitions = NumericPartitions::trivial(task);
        let mut numeric_domain_sizes: Vec<usize> = vec![1; task.numeric_variables().len()];
        let mut blacklisted_prop_var_ids = config.blacklisted_prop_var_ids.clone();
        let mut blacklisted_numeric_var_ids = config.blacklisted_numeric_var_ids.clone();
        apply_initial_goal_splits(
            task,
            config,
            semantics.numeric_integer_lattice(),
            &mut rng,
            RefinementState {
                domain_mapping: &mut domain_mapping,
                domain_sizes: &mut domain_sizes,
                partitions: &mut partitions,
                numeric_domain_sizes: &mut numeric_domain_sizes,
                blacklisted_prop_var_ids: &mut blacklisted_prop_var_ids,
                blacklisted_numeric_var_ids: &mut blacklisted_numeric_var_ids,
            },
        )?;
        let mut factory = DomainAbstractionFactory::new(
            task,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
        )
        .context("failed to construct the initial domain abstraction")?;

        let state_packer = Arc::new(make_prop_state_packer(task));
        let axiom_evaluator = AxiomEvaluator::new(Arc::new(task), state_packer.clone());
        let root = ConcreteState::initial(task)?;
        let limits = RefinementLimits {
            max_states: config.max_abstraction_size,
            blacklisted_prop_var_ids: &blacklisted_prop_var_ids,
            blacklisted_numeric_var_ids: &blacklisted_numeric_var_ids,
        };
        let mut solved_plan = None;
        let run = CegarDriver::new(config.max_iterations, config.max_time).run_from(
            start,
            |iteration, deadline| {
                if config.debug {
                    super::debug_dump::debug_print_abstraction_stats(
                        iteration,
                        factory.domain_sizes(),
                        factory.numeric_domain_sizes(),
                    );
                }
                let mut generator = factory.make_operator_generator(task, config.combine_labels)?;
                let operators =
                    match generator.build_abstract_operators_with_deadline(task, deadline) {
                        Ok(operators) => operators,
                        Err(error) if is_deadline_error(&error) => {
                            return Ok(CegarIterationResult::Stop(CegarStopReason::TimeLimit));
                        }
                        Err(error) => return Err(error),
                    };
                let table = match factory.build_distance_table(
                    task,
                    &generator,
                    &operators,
                    DistanceTableOptions::default()
                        .with_dump_distances(config.debug)
                        .with_deadline(deadline),
                ) {
                    Ok(table) => table,
                    Err(error) if is_deadline_error(&error) => {
                        return Ok(CegarIterationResult::Stop(CegarStopReason::TimeLimit));
                    }
                    Err(error) => return Err(error),
                };
                let view = DomainView::new(
                    task,
                    &factory,
                    generator.numeric_domain_sizes(),
                    generator.hash_multipliers(),
                    &operators,
                    &table,
                    limits,
                    config.use_wildcard_plans,
                );
                let check = match config.flaw_kind {
                    FlawKind::Progression => replay_optimal_abstract_trace(
                        &view,
                        &semantics,
                        &state_packer,
                        &axiom_evaluator,
                        &root,
                    )?,
                    FlawKind::ExecuteEntirePlan => replay_entire_optimal_abstract_trace(
                        &view,
                        &semantics,
                        &state_packer,
                        &axiom_evaluator,
                        &root,
                    )?,
                    FlawKind::Regression => {
                        regression_plan_check(&view, &semantics, &state_packer, &root)?
                    }
                };
                let split = match check {
                    PlanCheck::ConcretePlan(plan) => {
                        solved_plan = Some(plan);
                        return Ok(CegarIterationResult::Stop(CegarStopReason::ConcretePlan));
                    }
                    PlanCheck::AbstractDeadEnd => {
                        return Ok(CegarIterationResult::Stop(CegarStopReason::DeadEnd));
                    }
                    PlanCheck::Refine(split) => split,
                };
                if !view.refinable(&split) {
                    return Ok(CegarIterationResult::Stop(CegarStopReason::NoRefinableFlaw));
                }
                if config.debug {
                    debug!("domain CEGAR iteration {iteration}: {split:?}");
                }
                drop(view);
                factory.apply_split(task, &split)?;
                Ok(CegarIterationResult::Continue)
            },
        )?;

        if config.debug {
            log_final_abstraction(task, &factory);
        }
        Ok(CegarOutcome {
            final_state: CegarState::new(factory, run.next_iteration),
            stop_reason: run.stop_reason,
            solved_by_self: solved_plan.is_some(),
            concrete_plan_operator_ids: solved_plan.map(ConcretePlan::into_operator_ids),
        })
    }
}

fn is_deadline_error(error: &anyhow::Error) -> bool {
    crate::resource_limits::is_deadline_exceeded(error)
}

fn log_final_abstraction(task: &dyn AbstractNumericTask, factory: &DomainAbstractionFactory) {
    info!("domain abstraction final domains:");
    for (numeric_var_id, &size) in factory
        .numeric_domain_sizes()
        .iter()
        .enumerate()
        .filter(|(_, size)| **size > 1)
    {
        let name = task
            .numeric_variables()
            .get(numeric_var_id)
            .map(|variable| variable.name())
            .unwrap_or("<unknown>");
        let Some(parts) = factory
            .partitions()
            .partitions(VariableIndex::from_usize(numeric_var_id))
        else {
            continue;
        };
        let intervals = parts
            .iter()
            .enumerate()
            .map(|(part_id, interval)| format!("p{part_id}:{interval:?}"))
            .collect::<Vec<_>>()
            .join(" ");
        info!("  n{numeric_var_id}={name}, size={size}, {intervals}");
    }
    for (var_id, &size) in factory
        .domain_sizes()
        .iter()
        .enumerate()
        .filter(|(_, size)| **size > 1)
    {
        let name = task
            .get_variable_name(VariableIndex::from_usize(var_id))
            .unwrap_or("<unknown>");
        info!("  p{var_id}={name}, size={size}");
    }
}

#[allow(dead_code)]
fn wildcard_plan_is_real(
    task: &dyn AbstractNumericTask,
    wildcard_plan: &WildcardPlanResult,
) -> Result<bool> {
    // Note for Martin:
    // This verifies whether *any* concrete operator choice through a wildcard
    // abstract plan is a real plan. For long wildcard plans the search is
    // exponential in the number of equivalent concrete operators per step. On
    // sailing it cost several seconds per CEGAR iteration once plans became
    // long, so the CEGAR loop intentionally does not call this hot-path check.
    let state_packer = std::sync::Arc::new(make_prop_state_packer(task));
    let axiom_evaluator = AxiomEvaluator::new(std::sync::Arc::new(task), state_packer.clone());

    let plan_length = wildcard_plan.wildcard_plan.len();
    let (mut prop_state, mut numeric_state) =
        get_initial_state(task, &state_packer, &axiom_evaluator)?;
    if plan_length == 0 {
        return Ok(is_goal(task, &prop_state, &state_packer));
    }
    let mut last_state_per_layer = vec![None; plan_length];
    last_state_per_layer[0] = Some((prop_state.clone(), numeric_state.clone()));

    let mut real_plan_exists = false;
    let mut current_step: usize = 0;
    let mut equiv_op_iterators = Vec::with_capacity(plan_length);
    equiv_op_iterators.push(wildcard_plan.wildcard_plan[0].iter());

    loop {
        if let Some(op_id) = equiv_op_iterators[current_step].next() {
            let Some(op) = task.get_operators().get(op_id.index()) else {
                continue;
            };
            if is_applicable(&prop_state, &state_packer, op) {
                progress_concrete_state(
                    op,
                    &axiom_evaluator,
                    &state_packer,
                    &mut prop_state,
                    &mut numeric_state,
                )?;
                current_step += 1;
                if current_step == plan_length {
                    if is_goal(task, &prop_state, &state_packer) {
                        real_plan_exists = true;
                        break;
                    }
                    current_step -= 1;
                    (prop_state, numeric_state) =
                        last_state_per_layer[current_step].clone().unwrap();
                    continue;
                }
                last_state_per_layer[current_step] =
                    Some((prop_state.clone(), numeric_state.clone()));
                equiv_op_iterators.push(wildcard_plan.wildcard_plan[current_step].iter());
            }
        } else if current_step == 0 {
            break;
        } else {
            current_step -= 1;
            equiv_op_iterators.pop();
            (prop_state, numeric_state) = last_state_per_layer[current_step].clone().unwrap();
        }
    }

    Ok(real_plan_exists)
}

#[allow(dead_code)]
fn is_applicable(buffer: &[u64], packer: &StatePacker, op: &Operator) -> bool {
    op.preconditions()
        .iter()
        .all(|pre| pre.is_held(ConcreteStateView::from_decoded(packer, buffer, &[])))
}

fn is_goal(task: &dyn AbstractNumericTask, buffer: &[u64], packer: &StatePacker) -> bool {
    sorted_goal_facts(task)
        .iter()
        .all(|goal_fact| goal_fact.is_held(ConcreteStateView::from_decoded(packer, buffer, &[])))
}
