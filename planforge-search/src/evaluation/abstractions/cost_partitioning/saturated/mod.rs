use std::cell::RefCell;

use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::numeric_task::TaskRef;
use rand::{SeedableRng, rngs::SmallRng};
use tracing::info;

use crate::evaluation::abstractions::common::regions::RegionSemantics;
use crate::evaluation::abstractions::component::AbstractionComponent;
use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;
use crate::resource_limits::Deadline;

mod config;
mod deadline;
mod diagnostics;
mod diverse_orders;
mod goal_cover_orders;
mod lookup;
mod orders;
mod partition;
mod sampling;
mod scoring;
pub use config::{CostPartitioningMethod, OrderGenerator, Saturator, ScpConfig};
use deadline::deadline_from_now;
use lookup::{
    CostPartitioningHeuristic, abstraction_heuristic_values, compute_lookup_mask, compute_max_h,
    max_abstraction_heuristic,
};
use partition::{PartitionedCollection, saturate_component_labels};
use scoring::{compute_all_surplus_costs, compute_costs_stolen_by_heuristic, region_stolen_costs};

/// The abstraction components and the phase of the heuristic, which builds
/// its cost partitionings at the first evaluated state.
struct ScpState<'task> {
    components: Vec<AbstractionComponent<'task>>,
    phase: ScpPhase<'task>,
}

enum ScpPhase<'task> {
    /// Before the first evaluated state that is not an abstract dead end.
    Unbuilt(OrderSelection<'task>),
    /// The retained cost partitionings, whose maximum is the heuristic.
    Built(RetainedCostPartitionings),
}

/// What choosing orders needs: every component's goal distances under the
/// original costs and the costs it steals from the others there (Seipp
/// 2017), and the random sources of orders and samples.
struct OrderSelection<'task> {
    goal_distances_by_abstraction: Vec<Vec<f64>>,
    stolen_costs_by_abstraction: Vec<f64>,
    rng: SmallRng,
    /// The owned task diversification samples states from.
    sampling_task: Option<TaskRef<'task>>,
}

impl OrderSelection<'_> {
    fn new(seed: u64) -> Self {
        Self {
            goal_distances_by_abstraction: Vec::new(),
            stolen_costs_by_abstraction: Vec::new(),
            rng: SmallRng::seed_from_u64(seed),
            sampling_task: None,
        }
    }
}

struct RetainedCostPartitionings {
    cp_heuristics: Vec<CostPartitioningHeuristic>,
    /// The components some lookup table reads, the only ones a state is
    /// mapped into.
    lookup_mask: Vec<bool>,
}

pub struct SaturatedCostPartitioningHeuristic<'task> {
    name: String,
    task: TaskRef<'task>,
    config: ScpConfig,
    original_operator_costs: Vec<f64>,
    state: RefCell<ScpState<'task>>,
    state_values: RefCell<ExplicitState>,
    component_ids_scratch: RefCell<Vec<Option<usize>>>,
    construction_deadline: Deadline,
}

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    pub fn new(
        name: Option<String>,
        components: Vec<AbstractionComponent<'task>>,
        config: ScpConfig,
        task: TaskRef<'task>,
        construction_deadline: Deadline,
    ) -> Result<Self, EvaluationError> {
        if components.is_empty() {
            return Err(EvaluationError::ComputationFailed(
                "SCP requires at least one abstraction component".to_string(),
            ));
        }
        config
            .validate()
            .map_err(EvaluationError::ComputationFailed)?;
        let mut components = components;
        if config.partitioning.uses_regions() {
            let regions = RegionSemantics::new(&*task)
                .map_err(|error| EvaluationError::ComputationFailed(format!("{error:#}")))?;
            for (component_id, component) in components.iter_mut().enumerate() {
                match component {
                    AbstractionComponent::Cegar(abstraction) => abstraction
                        .prepare_regional_saturation(&regions, Deadline::NONE)
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to build the source sets of component {component_id}: {error:#}"
                            ))
                        })?,
                    AbstractionComponent::PatternDatabase(_) => info!(
                        "PDB component {component_id}: partitioning=region requested, using label cost partitioning (PDBs have no source sets)"
                    ),
                }
            }
        }
        let original_costs: Vec<f64> = task
            .effective_operator_costs()
            .iter()
            .map(|cost| cost.value())
            .collect();

        let mut h_values: Vec<Vec<f64>> = Vec::with_capacity(components.len());
        let mut saturated_costs_by_abstraction: Vec<Vec<f64>> =
            Vec::with_capacity(components.len());
        for (component_id, component) in components.iter().enumerate() {
            let (distances, saturated) =
                saturate_component_labels(component, &original_costs, None, Deadline::NONE)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to saturate component {component_id} under the original costs: {error:#}"
                        ))
                    })?;
            h_values.push(distances);
            saturated_costs_by_abstraction.push(saturated);
        }

        let surplus_costs =
            compute_all_surplus_costs(&original_costs, &saturated_costs_by_abstraction);
        let mut stolen_costs: Vec<f64> = saturated_costs_by_abstraction
            .iter()
            .map(|saturated| compute_costs_stolen_by_heuristic(saturated, &surplus_costs))
            .collect();
        if config.partitioning.uses_regions() {
            let regional_scores = region_stolen_costs(
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

        let selection = OrderSelection {
            goal_distances_by_abstraction: h_values,
            stolen_costs_by_abstraction: stolen_costs,
            sampling_task: Some(task.clone()),
            ..OrderSelection::new(config.random_seed)
        };

        Ok(Self {
            name: name.unwrap_or_else(|| "scp".to_string()),
            task,
            config,
            original_operator_costs: original_costs,
            state: RefCell::new(ScpState {
                components,
                phase: ScpPhase::Unbuilt(selection),
            }),
            state_values: RefCell::default(),
            component_ids_scratch: RefCell::new(Vec::new()),
            construction_deadline,
        })
    }

    /// The cost partitionings of the collection, built at the first
    /// evaluated state: the greedy order's, or the diverse orders' over the
    /// random-walk samples, which are pairwise distinct.
    fn build_cost_partitionings(
        &self,
        components: &[AbstractionComponent<'task>],
        state: &mut OrderSelection<'task>,
        abstract_state_ids: &[Option<usize>],
    ) -> Result<Vec<CostPartitioningHeuristic>, EvaluationError> {
        let budget = self.construction_deadline.earliest(deadline_from_now(
            "table_construction_max_time",
            self.config
                .table_construction_max_time
                .duration()
                .map_err(EvaluationError::ComputationFailed)?,
        )?);
        let abstraction_h = abstraction_heuristic_values(state, abstract_state_ids);
        let collection = PartitionedCollection {
            components,
            abstract_state_ids,
            abstraction_h: &abstraction_h,
            original_costs: &self.original_operator_costs,
        };
        let mut order = self.compute_order_for_state(state, abstract_state_ids, components);
        let mode = if self.config.partitioning.uses_regions() {
            "region"
        } else {
            "label"
        };
        info!(
            "scp: building {mode} cost partitionings, max abstraction h={}, order_len={}, saturator={}",
            max_abstraction_heuristic(state, abstract_state_ids),
            order.len(),
            self.config.saturator,
        );

        if !self.config.diversify {
            let (cp, _) = self.build_cp(collection, &order, budget)?;
            if cp.is_empty() {
                info!("scp: the greedy order's {mode} cost partitioning has no lookup tables");
                return Ok(Vec::new());
            }
            return Ok(vec![cp]);
        }
        let mut candidates = self.build_best_cp_from_candidate_orders(
            collection,
            &mut order,
            budget,
            self.config
                .initial_order_generation_max_time
                .duration()
                .map_err(EvaluationError::ComputationFailed)?,
        )?;
        let best = candidates.partitions.swap_remove(candidates.best_index);
        if best.is_empty() {
            // At an equal estimate the best order is one whose cost
            // partitioning has lookup tables, and an empty one estimates 0.
            assert!(
                candidates
                    .partitions
                    .iter()
                    .all(CostPartitioningHeuristic::is_empty),
                "the best candidate order has lookup tables if any has"
            );
            info!("scp: no candidate order gives a {mode} cost partitioning with lookup tables");
            return Ok(Vec::new());
        }
        candidates.partitions.retain(|cp| !cp.is_empty());
        let mut initial_candidates = Vec::with_capacity(candidates.partitions.len() + 1);
        initial_candidates.push(best);
        initial_candidates.extend(candidates.partitions);
        self.select_diverse_orders(collection, state, initial_candidates, budget)
    }
}

impl Heuristic for SaturatedCostPartitioningHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let mut state = self.state.borrow_mut();
        let ScpState { components, phase } = &mut *state;
        let mut component_ids = self.component_ids_scratch.borrow_mut();
        let selection = match phase {
            ScpPhase::Built(retained) => {
                self.compute_abstract_state_ids_into(
                    components,
                    eval_state,
                    Some(&retained.lookup_mask),
                    &mut component_ids,
                )?;
                return Ok(compute_max_h(&retained.cp_heuristics, &component_ids));
            }
            ScpPhase::Unbuilt(selection) => selection,
        };
        self.compute_abstract_state_ids_into(components, eval_state, None, &mut component_ids)?;
        let abstract_state_ids = component_ids.as_slice();
        if max_abstraction_heuristic(selection, abstract_state_ids).is_infinite() {
            // The first state is an abstract dead end, which every cost
            // partitioning reports too; there is nothing to partition for.
            return Ok(f64::INFINITY);
        }

        let cp_heuristics =
            self.build_cost_partitionings(components, selection, abstract_state_ids)?;
        let lookup_mask = compute_lookup_mask(&cp_heuristics, components.len());
        for component in components.iter_mut() {
            component.discard_transitions();
        }
        info!("scp: retained {} cost partitionings", cp_heuristics.len());
        let h = compute_max_h(&cp_heuristics, abstract_state_ids);
        *phase = ScpPhase::Built(RetainedCostPartitionings {
            cp_heuristics,
            lookup_mask,
        });
        Ok(h)
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests;

pub use registry_feature::ScpFeature;
mod registry_feature {
    use crate::config::ConfigError;
    use crate::evaluation::Heuristic;
    use crate::heuristic_factory::features::{source_requirements, validate_sources};
    use crate::heuristic_factory::*;

    pub struct ScpFeature;
    impl HeuristicFeature for ScpFeature {
        const NAME: &'static str = "scp";
        type Config = ScpHeuristicConfig;
        fn requirements(config: &Self::Config) -> TaskRequirements {
            source_requirements(&config.sources)
        }
        fn validate(config: &Self::Config) -> Result<(), ConfigError> {
            validate_sources(&config.sources)?;
            config.scp.validate().map_err(Into::into)
        }
        fn build<'a>(
            config: &Self::Config,
            context: &BuildContext<'a>,
        ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
            build_scp_from_sources(context, config)
        }
    }
}
