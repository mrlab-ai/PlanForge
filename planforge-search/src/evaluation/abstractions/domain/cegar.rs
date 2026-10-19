//! The CEGAR loop that refines a domain abstraction: find flaws in the
//! abstract plan, refine, and repeat until a limit is reached or no flaw
//! remains.

#[cfg(test)]
mod tests;

use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::{AbstractNumericTask, Operator, OperatorIndex, VariableIndex};
use planforge_sas::state_registry::ConcreteStateView;
use planforge_sas::utils::state_packer::StatePacker;
use rand::{SeedableRng, rngs::SmallRng};
use tracing::{debug, info};

use super::abstraction::compute_abstraction_size_u128;
use super::config::CegarConfig;
use super::debug_dump::debug_print_refinement_summary;
use super::factory::{DistanceTableOptions, DomainAbstractionFactory, WildcardPlanResult};
use super::flaws::{Flaw, FlawSearchOutcome};
use super::initial_splits::{
    apply_initial_goal_splits, sorted_goal_facts, trivial_domain_mapping_and_sizes,
};
use super::partitions::NumericPartitions;
use super::refinement::{RefinementState, filter_eligible_flaws, fix_flaws};
use crate::evaluation::abstractions::collections::goal_focused::RefinementRootDeadEnd;
use crate::evaluation::abstractions::common::cegar::{
    CegarDriver, CegarIterationResult, progress_concrete_state,
};
use crate::evaluation::abstractions::common::cegar::{CegarStopReason, FlawKind};
use crate::evaluation::abstractions::common::concrete_state::{
    get_initial_state, make_prop_state_packer,
};
use crate::evaluation::validate_abstractable_goal;

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
pub struct CegarStep {
    pub wildcard_plan: Option<WildcardPlanResult>,
}

#[derive(Debug, Clone)]
pub struct CegarOutcome {
    pub final_state: CegarState,
    pub last_step: CegarStep,
    pub stop_reason: CegarStopReason,
    /// True iff CEGAR's loop exited because no flaws remained. For a
    /// standalone full-task abstraction, the abstract plan is then a real
    /// concrete plan and `h(init)` is optimal.
    pub solved_by_self: bool,
    /// The concrete operators the flaw search executed when no flaws remained.
    /// `Some` exactly when `solved_by_self` and the flaw kind executes a
    /// concrete plan (see [`FlawSearchOutcome::concrete_plan`]).
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
        validate_abstractable_goal(task).map_err(anyhow::Error::msg)?;
        if config.flaw_kind == FlawKind::Regression {
            crate::evaluation::abstractions::common::cegar::regression::ensure_regressable(
                task,
                &crate::evaluation::abstractions::common::coordinates::Coordinates::new(task),
            )?;
        }
        ensure!(
            config.max_abstraction_size > 0,
            "max_abstraction_size must be > 0"
        );
        ensure!(config.max_iterations > 0, "max_iterations must be > 0");

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

        let iteration = 1usize;
        let mut factory = DomainAbstractionFactory::new(
            task,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
        )
        .with_context(|| {
            format!("failed to construct DomainAbstractionFactory (iteration {iteration})")
        })?;
        let mut wildcard_plan = None;
        let mut solved_by_self = false;
        let mut concrete_plan_operator_ids = None;
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

            let iteration_start = Instant::now();
            let plan_start = Instant::now();
            match factory.compute_plan_with_rng(
                task,
                config.combine_labels,
                config.use_wildcard_plans,
                Some(&mut rng),
                DistanceTableOptions::default()
                    .with_dump_distances(config.debug)
                    .with_deadline(deadline),
            ) {
                Ok(plan) => wildcard_plan = plan,
                Err(error) if is_deadline_error(&error) => {
                    info!(
                        "CEGAR: deadline expired while computing abstract plan at iteration {}; stopping refinement",
                        iteration
                    );
                    return Ok(CegarIterationResult::Stop(CegarStopReason::TimeLimit));
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("failed to compute abstract plan (iteration {iteration})")
                    });
                }
            }
            let plan_time = plan_start.elapsed();
            if config.debug {
                match wildcard_plan.as_ref() {
                    Some(plan) => super::debug_dump::debug_print_wildcard_plan(
                        task,
                        plan,
                        factory.domain_sizes(),
                        factory.numeric_domain_sizes(),
                        factory.partitions(),
                    )?,
                    None => debug!("[Abstract Plan] <none>"),
                }
            }

            let Some(plan) = wildcard_plan.as_ref() else {
                return Err(RefinementRootDeadEnd::new(format!(
                    "domain abstraction at CEGAR iteration {iteration}; abstraction_size={:?}, prop_domains={:?}, numeric_domains={:?}",
                    compute_abstraction_size_u128(
                        factory.domain_sizes(),
                        factory.numeric_domain_sizes(),
                    ),
                    factory.domain_sizes(),
                    factory.numeric_domain_sizes()
                ))
                .into());
            };
            let real_check_time = Duration::ZERO;

            let flaw_start = Instant::now();
            let FlawSearchOutcome {
                flaws,
                concrete_plan,
            } = self
                .config
                .flaw_kind
                .get_flaws_with_direction(
                    task,
                    factory.partitions(),
                    factory.domain_mapping(),
                    plan,
                    self.config.effective_split_direction(),
                )
                .with_context(|| format!("failed to collect flaws (iteration {iteration})"))?;
            let flaw_time = flaw_start.elapsed();
            if config.debug {
                super::debug_dump::debug_print_flaws(&flaws);
            }
            if flaws.is_empty() {
                solved_by_self = true;
                concrete_plan_operator_ids = concrete_plan;
                return Ok(CegarIterationResult::Stop(CegarStopReason::ConcretePlan));
            }
            let eligible_flaws =
                filter_eligible_flaws(task, factory.partitions(), factory.domain_sizes(), &flaws);
            if eligible_flaws.filtered_stale > 0 {
                debug!(
                    "filtered {} stale flaws (already-split values / fully refined vars)",
                    eligible_flaws.filtered_stale
                );
            }
            if eligible_flaws.flaws.is_empty() {
                info!(
                    "CEGAR: {} flaws remain but none are refinable (stale boundaries / fully refined vars); stopping refinement at iteration {}",
                    flaws.len(),
                    iteration
                );
                return Ok(CegarIterationResult::Stop(
                    CegarStopReason::NoRefinableFlaw,
                ));
            }

            let before_size = if config.debug {
                compute_abstraction_size_u128(
                    factory.domain_sizes(),
                    factory.numeric_domain_sizes(),
                )
            } else {
                None
            };
            // Overflow-free progress signal: every landed split increments
            // exactly one entry of `domain_sizes` or `numeric_domain_sizes`,
            // so the sum strictly increases iff any refinement landed. The
            // u128 size product cannot serve here: it saturates to `None` on
            // overflow, which would make progress undetectable.
            let before_partition_count = factory.domain_sizes().iter().sum::<usize>()
                + factory.numeric_domain_sizes().iter().sum::<usize>();
            let refine_start = Instant::now();
            let refined = factory
                .refine(
                    |domain_mapping, domain_sizes, partitions, numeric_domain_sizes| {
                        fix_flaws(
                            &self.config,
                            task,
                            &eligible_flaws.flaws,
                            RefinementState {
                                domain_mapping,
                                domain_sizes,
                                partitions,
                                numeric_domain_sizes,
                                blacklisted_prop_var_ids: &mut blacklisted_prop_var_ids,
                                blacklisted_numeric_var_ids: &mut blacklisted_numeric_var_ids,
                            },
                            &mut rng,
                        )
                    },
                )
                .with_context(|| format!("failed to fix flaws (iteration {iteration})"))?;
            let refine_time = refine_start.elapsed();
            let after_partition_count = factory.domain_sizes().iter().sum::<usize>()
                + factory.numeric_domain_sizes().iter().sum::<usize>();
            if config.debug {
                let after_size = compute_abstraction_size_u128(
                    factory.domain_sizes(),
                    factory.numeric_domain_sizes(),
                );
                debug_print_refinement_summary(
                    before_size,
                    after_size,
                    factory.domain_sizes(),
                    factory.numeric_domain_sizes(),
                    !refined.is_empty(),
                );
            }
            if refined.is_empty() {
                return Ok(CegarIterationResult::Stop(
                    CegarStopReason::NoRefinableFlaw,
                ));
            }
            let split_values: Vec<_> = eligible_flaws
                .flaws
                .iter()
                .filter_map(|flaw| match flaw {
                    Flaw::Numeric(numeric) => Some((
                        numeric.numeric_var_id,
                        numeric.value,
                        numeric.include_in_lower,
                    )),
                    Flaw::Propositional(_) => None,
                })
                .collect();
            ensure!(
                after_partition_count > before_partition_count,
                "CEGAR refinement made no progress at iteration {iteration}: flaws={:?}, refined={:?}, split_values={:?}",
                eligible_flaws.flaws,
                refined,
                split_values
            );
            if config.debug {
                let abstraction_size = compute_abstraction_size_u128(
                    factory.domain_sizes(),
                    factory.numeric_domain_sizes(),
                )
                .unwrap_or(u128::MAX);
                info!(
                    "CEGAR iteration {iteration}: plan_len={}, flaws={}, refined={:?}, size={}, elapsed={:.3}s, plan={:.3}s, real_check={:.3}s, flaws_time={:.3}s, refine={:.3}s",
                    plan.wildcard_plan.len(),
                    flaws.len(),
                    refined,
                    abstraction_size,
                    iteration_start.elapsed().as_secs_f64(),
                    plan_time.as_secs_f64(),
                    real_check_time.as_secs_f64(),
                    flaw_time.as_secs_f64(),
                    refine_time.as_secs_f64()
                );
            }

            Ok(CegarIterationResult::Continue)
        })?;
        let iteration = run.next_iteration;
        let stop_reason = run.stop_reason;

        if config.debug {
            log_final_target_centered_abstraction(task, &factory);
        }

        let last_step = CegarStep { wildcard_plan };
        Ok(CegarOutcome {
            final_state: CegarState::new(factory, iteration),
            last_step,
            stop_reason,
            solved_by_self,
            concrete_plan_operator_ids,
        })
    }
}

fn is_deadline_error(error: &anyhow::Error) -> bool {
    crate::resource_limits::is_deadline_exceeded(error)
}

fn log_final_target_centered_abstraction(
    task: &dyn AbstractNumericTask,
    factory: &DomainAbstractionFactory,
) {
    info!("target-centered domain abstraction final domains:");
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
