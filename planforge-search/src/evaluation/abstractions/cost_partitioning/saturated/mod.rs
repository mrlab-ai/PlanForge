use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use planforge_sas::numeric_task::{
    AbstractNumericTask, NumericValue, TaskRef, metric_operator_cost_from_initial_values,
};
use planforge_sas::state_registry::{ConcreteState, ExpansionContext, StateRegistry};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use tracing::{debug, info};

use crate::evaluation::abstractions::cartesian::CartesianAbstraction;
use crate::evaluation::abstractions::pdb::pattern_database::PatternDatabase;
use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;
use crate::successor_generator::SuccessorTree;

use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use crate::evaluation::abstractions::common::cegar::SplitKind;
use crate::evaluation::abstractions::common::transition_system::AbstractDistanceTable;
use crate::evaluation::abstractions::common::transition_system::AbstractOperatorSourceSets;
use crate::evaluation::abstractions::component::AbstractionComponent;
use crate::evaluation::abstractions::cost_partitioning::costs::{
    RemainingCosts, SourceSetCostAllocation, build_explicit_label_cost_partitioning_table,
    build_explicit_regional_cost_partitioning_table,
};
use crate::evaluation::abstractions::domain::abstract_operators::AbstractOperator;
use crate::evaluation::abstractions::domain::abstraction::DomainAbstraction;
use crate::evaluation::abstractions::domain::factory::DistanceTableOptions;
use crate::evaluation::abstractions::domain::heuristic::DomainAbstractionLookupScratch;

mod config;
mod deadline;
mod diagnostics;
mod diverse_orders;
mod goal_cover_orders;
mod lookup;
mod orders;
mod sampling;
mod saturation;
mod scoring;
pub use config::{CostPartitioningMethod, OrderGenerator, Saturator, ScpConfig};
use deadline::*;
use diagnostics::*;
use goal_cover_orders::*;
use lookup::*;
use saturation::*;
use scoring::*;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Main heuristic
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ScpState {
    /// Whether the cost partitionings have been built, which happens at the
    /// first evaluated state.
    built: bool,
    cp_heuristics: Vec<CostPartitioningHeuristic>,
    goal_distances_by_abstraction: Vec<Vec<f64>>,
    abstraction_lookup_tables: Vec<LookupTable>,
    stolen_costs_by_abstraction: Vec<f64>,
    rng: SmallRng,
    lookup_mask: Vec<bool>,
    sample_abstract_state_ids: Vec<Vec<Option<usize>>>,
}

impl ScpState {
    fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or(crate::evaluation::DEFAULT_RANDOM_SEED);
        Self {
            built: false,
            cp_heuristics: Vec::new(),
            goal_distances_by_abstraction: Vec::new(),
            abstraction_lookup_tables: Vec::new(),
            stolen_costs_by_abstraction: Vec::new(),
            rng: SmallRng::seed_from_u64(seed),
            lookup_mask: Vec::new(),
            sample_abstract_state_ids: Vec::new(),
        }
    }
}

pub struct SaturatedCostPartitioningHeuristic<'task> {
    name: String,
    task: &'task dyn AbstractNumericTask,
    components: RefCell<Vec<AbstractionComponent<'task>>>,
    config: ScpConfig,
    debug_diagnostics: bool,
    original_operator_costs: Vec<f64>,
    state: RefCell<ScpState>,
    lookup_scratch: RefCell<DomainAbstractionLookupScratch>,
    component_ids_scratch: RefCell<Vec<Option<usize>>>,
    sampling_task: RefCell<Option<TaskRef<'task>>>,
}

/// What a [`SaturatedCostPartitioningHeuristic`] needs besides its abstractions
/// and configuration.
#[derive(Default)]
pub struct ScpBuildOptions<'task> {
    /// An owned task to sample states from by random walks; required for
    /// diverse orders.
    pub sampling_task: Option<TaskRef<'task>>,
    /// Log per-abstraction diagnostics while building cost partitionings.
    pub debug_diagnostics: bool,
}

/// The components of a collection of domain abstractions, Cartesian
/// abstractions and pattern databases, in that order, named `{prefix}_{i}` and
/// `{prefix}_cartesian_{i}`.
pub fn collection_components<'task>(
    prefix: &str,
    domain_abstractions: Vec<DomainAbstraction>,
    cartesian_abstractions: Vec<CartesianAbstraction>,
    pdbs: Vec<PatternDatabase<'task>>,
) -> Vec<AbstractionComponent<'task>> {
    let mut components =
        Vec::with_capacity(domain_abstractions.len() + cartesian_abstractions.len() + pdbs.len());
    components.extend(
        domain_abstractions
            .into_iter()
            .enumerate()
            .map(|(index, abstraction)| {
                AbstractionComponent::domain(Some(format!("{prefix}_{index}")), abstraction)
            }),
    );
    components.extend(cartesian_abstractions.into_iter().enumerate().map(
        |(index, abstraction)| {
            AbstractionComponent::cartesian(
                Some(format!("{prefix}_cartesian_{index}")),
                abstraction,
            )
        },
    ));
    components.extend(pdbs.into_iter().map(AbstractionComponent::pattern_database));
    components
}

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    pub fn new(
        name: Option<String>,
        components: Vec<AbstractionComponent<'task>>,
        config: ScpConfig,
        task: &'task dyn AbstractNumericTask,
        options: ScpBuildOptions<'task>,
    ) -> Result<Self, EvaluationError> {
        let ScpBuildOptions {
            sampling_task,
            debug_diagnostics,
        } = options;
        if components.is_empty() {
            return Err(EvaluationError::ComputationFailed(
                "SCP requires at least one abstraction component".to_string(),
            ));
        }
        config
            .validate(sampling_task.is_some())
            .map_err(EvaluationError::ComputationFailed)?;
        if config.partitioning.uses_regions() {
            for (component_id, component) in components.iter().enumerate() {
                if matches!(component, AbstractionComponent::PatternDatabase(_)) {
                    info!(
                        "PDB component {component_id}: partitioning=region requested, using label cost partitioning (PDBs have no source sets)"
                    );
                }
            }
        }
        let original_costs: Vec<f64> = task
            .get_operators()
            .iter()
            .map(|op| metric_operator_cost_from_initial_values(task, op).value())
            .collect();

        let mut h_values: Vec<Vec<f64>> = Vec::with_capacity(components.len());
        let mut saturated_costs_by_abstraction: Vec<Vec<f64>> =
            Vec::with_capacity(components.len());
        let mut debug_initial_h_values = Vec::new();

        for (component_id, component) in components.iter().enumerate() {
            let (distances, saturated, initial_state_id) = saturate_original_costs(
                component,
                component_id,
                task,
                config.combine_labels,
                &original_costs,
                debug_diagnostics,
            )?;
            if debug_diagnostics {
                let initial_h = distances.get(initial_state_id).copied().ok_or_else(|| {
                    EvaluationError::InvalidState(format!(
                        "{} component {component_id} initial state {initial_state_id} out of bounds for {} states",
                        component.kind(),
                        distances.len()
                    ))
                })?;
                debug_initial_h_values.push(initial_h);
                info!(
                    "scp debug: {} component {component_id}: original_initial_h={initial_h}, states={}",
                    component.kind(),
                    component.num_states()
                );
            }
            h_values.push(distances);
            saturated_costs_by_abstraction.push(saturated);
        }

        if debug_diagnostics {
            let max_initial_h = debug_initial_h_values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            info!(
                "scp debug: collection max original-cost initial h before cost partitioning = {max_initial_h}"
            );
        }

        let surplus_costs =
            compute_all_surplus_costs(&original_costs, &saturated_costs_by_abstraction);
        let mut stolen_costs: Vec<f64> = saturated_costs_by_abstraction
            .iter()
            .map(|saturated| compute_costs_stolen_by_heuristic(saturated, &surplus_costs))
            .collect();
        if config.partitioning.uses_regions() {
            let regional_scores = interval_stolen_costs(
                &components,
                &saturated_costs_by_abstraction,
                &original_costs,
            )?;
            for (component_id, score) in regional_scores.into_iter().enumerate() {
                if let Some(score) = score {
                    stolen_costs[component_id] = score;
                }
            }
        }

        let mut st = ScpState::new(config.random_seed);
        st.goal_distances_by_abstraction = h_values;
        st.stolen_costs_by_abstraction = stolen_costs;
        st.lookup_mask = vec![false; components.len()];

        Ok(Self {
            name: name.unwrap_or_else(|| "scp".to_string()),
            task,
            components: RefCell::new(components),
            config,
            debug_diagnostics,
            original_operator_costs: original_costs,
            state: RefCell::new(st),
            lookup_scratch: RefCell::new(DomainAbstractionLookupScratch::new()),
            component_ids_scratch: RefCell::new(Vec::new()),
            sampling_task: RefCell::new(sampling_task),
        })
    }

    /// The cost partitionings of the collection, built at the first
    /// evaluated state: the greedy order's, or the diverse orders' over the
    /// random-walk samples.
    fn build_cost_partitionings(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpState,
        abstract_state_ids: &[Option<usize>],
    ) -> Result<Vec<CostPartitioningHeuristic>, EvaluationError> {
        let components = self.components.borrow();
        let deadline = self
            .config
            .table_construction_max_time
            .is_finite()
            .then(|| {
                Instant::now() + Duration::from_secs_f64(self.config.table_construction_max_time)
            });
        let abstraction_h = abstraction_heuristic_values(state, abstract_state_ids);
        let collection = PartitionedCollection {
            task,
            components: &components,
            abstract_state_ids,
            abstraction_h: &abstraction_h,
            original_costs: &self.original_operator_costs,
        };
        let mut order = self.compute_order_for_state(state, abstract_state_ids, &components);
        let mode = if self.config.partitioning.uses_regions() {
            "interval"
        } else {
            "label"
        };
        info!(
            "scp: building {mode} cost partitionings, h={}, order_len={}, saturator={}",
            Self::compute_max_h(state, abstract_state_ids),
            order.len(),
            self.config.saturator,
        );
        if self.debug_diagnostics {
            log_abstraction_candidate_report(mode, state, &components, &order, abstract_state_ids);
        }

        if !self.config.diversify {
            let cp = self.build_cp(collection, &order, deadline)?;
            if cp.is_empty() {
                info!("scp: the greedy order's {mode} cost partitioning has no lookup tables");
                return Ok(Vec::new());
            }
            return Ok(vec![cp]);
        }
        let mut candidates = self.build_best_cp_from_candidate_orders(
            collection,
            &mut order,
            deadline,
            self.config.initial_order_generation_max_time,
        )?;
        let best = candidates.partitions.swap_remove(candidates.best_index);
        candidates.partitions.retain(|cp| !cp.is_empty());
        if best.is_empty() && candidates.partitions.is_empty() {
            info!("scp: no candidate order gives a {mode} cost partitioning with lookup tables");
            return Ok(Vec::new());
        }
        let mut initial_candidates = Vec::with_capacity(candidates.partitions.len() + 1);
        if !best.is_empty() {
            initial_candidates.push(best);
        }
        initial_candidates.extend(candidates.partitions);
        self.select_diverse_orders(collection, state, initial_candidates, deadline)
    }

    /// Stores `cp` unless an identical cost partitioning is stored already.
    fn retain_cp(state: &mut ScpState, cp: CostPartitioningHeuristic) {
        if state.cp_heuristics.contains(&cp) {
            info!(
                "scp: discarded duplicate cost partitioning with {} lookup tables",
                cp.lookup_tables.len()
            );
            return;
        }
        state.cp_heuristics.push(cp);
    }
}

impl Heuristic for SaturatedCostPartitioningHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let built = self.state.borrow().built;
        let mut component_ids = self.component_ids_scratch.borrow_mut();
        {
            let state = self.state.borrow();
            let lookup_mask = built.then_some(state.lookup_mask.as_slice());
            self.compute_abstract_state_ids_into(eval_state, lookup_mask, &mut component_ids)?;
        }
        let abstract_state_ids = component_ids.as_slice();
        let mut state = self.state.borrow_mut();
        if built {
            return Ok(Self::compute_max_h(&state, abstract_state_ids));
        }
        let max_h = Self::compute_max_h(&state, abstract_state_ids);
        if max_h.is_infinite() {
            // The first state is an abstract dead end; there is nothing to
            // partition for.
            return Ok(max_h);
        }

        let cost_partitionings =
            self.build_cost_partitionings(eval_state.task(), &mut state, abstract_state_ids)?;
        Self::retain_abstraction_lookup_tables(&mut state);
        for cp in cost_partitionings {
            Self::retain_cp(&mut state, cp);
        }
        state.lookup_mask = Self::compute_lookup_mask(&state, abstract_state_ids.len());
        state.stolen_costs_by_abstraction = Vec::new();
        for component in self.components.borrow_mut().iter_mut() {
            component.discard_transition_data();
        }
        state.built = true;
        info!(
            "scp: retained {} cost partitionings",
            state.cp_heuristics.len()
        );
        Ok(Self::compute_max_h(&state, abstract_state_ids))
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests;
