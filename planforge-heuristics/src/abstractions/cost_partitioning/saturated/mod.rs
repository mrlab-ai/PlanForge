//! Saturated cost partitioning (Seipp and Helmert 2018) over a collection of
//! abstractions, offline over orders chosen once or online per state.

use planforge_sas::numeric_task::NumericValue;
use planforge_sas::numeric_task::Task;
use planforge_sas::state_registry::StateRegistry;
use rand::{SeedableRng, rngs::SmallRng};
use std::sync::Arc;
use tracing::info;

use crate::abstractions::common::regions::RegionSemantics;
use crate::abstractions::component::AbstractionComponent;
use crate::abstractions::sources::{
    AbstractionPurpose, CombinationConfig, build_components, remaining_construction_time,
};
use planforge_search::config::ConfigError;
use planforge_search::evaluation::Heuristic;
use planforge_search::evaluation::{Estimate, EvaluationError, EvaluationState};
use planforge_search::heuristic_features::{
    BuildContext, HeuristicBuildError, HeuristicFeature, TaskRequirements,
};
use planforge_search::resource_limits::Deadline;

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
    CostPartitioningHeuristic, abstract_state_ids_into, abstraction_heuristic_values,
    compute_lookup_mask, compute_max_h, dead_end_abstraction, max_abstraction_heuristic,
};
use partition::{PartitionedCollection, saturate_component_labels};
use scoring::{compute_all_surplus_costs, compute_costs_stolen_by_heuristic, region_stolen_costs};

/// What choosing orders needs: every component's goal distances under the
/// original costs and the costs it steals from the others there (Seipp
/// 2017), and the random source of orders and samples.
struct OrderSelection {
    goal_distances_by_abstraction: Vec<Vec<f64>>,
    stolen_costs_by_abstraction: Vec<f64>,
    rng: SmallRng,
}

/// The construction of the cost partitionings: the task, the configuration
/// and the costs it partitions.
struct ScpBuilder {
    task: Arc<Task>,
    config: ScpConfig,
    original_operator_costs: Vec<f64>,
    construction_deadline: Deadline,
}

/// The maximum over saturated cost partitionings of abstraction heuristics
/// (Seipp, Keller and Helmert 2020), built for the initial state at
/// construction.
#[derive(Debug)]
pub struct SaturatedCostPartitioningHeuristic {
    name: String,
    components: Vec<AbstractionComponent>,
    cost_partitionings: Vec<CostPartitioningHeuristic>,
    /// The components some lookup table reads, the only ones a state is
    /// mapped into.
    lookup_mask: Vec<bool>,
    numeric_values: Vec<NumericValue>,
    abstract_state_ids: Vec<Option<usize>>,
}

impl SaturatedCostPartitioningHeuristic {
    pub fn new(
        name: Option<String>,
        components: Vec<AbstractionComponent>,
        config: ScpConfig,
        task: Arc<Task>,
        construction_deadline: Deadline,
    ) -> Result<Self, EvaluationError> {
        if components.is_empty() {
            return Err(EvaluationError::ComputationFailed(
                "SCP requires at least one abstraction component".to_string(),
            ));
        }
        config
            .validate()
            .map_err(|error| EvaluationError::ComputationFailed(error.to_string()))?;
        let mut components = components;
        if config.partitioning.uses_regions() {
            let regions = RegionSemantics::new(&task)
                .map_err(|error| EvaluationError::ComputationFailed(format!("{error:#}")))?;
            for (component_id, component) in components.iter_mut().enumerate() {
                match component {
                    AbstractionComponent::Cegar { abstraction, .. } => abstraction
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
        let original_costs: Vec<f64> = task.effective_operator_costs().to_vec();

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

        let mut selection = OrderSelection {
            goal_distances_by_abstraction: h_values,
            stolen_costs_by_abstraction: stolen_costs,
            rng: SmallRng::seed_from_u64(config.random_seed),
        };

        let mut numeric_values = Vec::new();
        let mut abstract_state_ids = Vec::new();
        {
            let mut registry = StateRegistry::for_task(task.clone());
            let initial_state = registry.register_initial_state();
            abstract_state_ids_into(
                &components,
                &EvaluationState::new(&initial_state, &task, &registry),
                None,
                &mut numeric_values,
                &mut abstract_state_ids,
            )?;
        }
        let builder = ScpBuilder {
            task,
            config,
            original_operator_costs: original_costs,
            construction_deadline,
        };
        let cost_partitionings = match dead_end_abstraction(&selection, &abstract_state_ids) {
            // Every cost partitioning that keeps this abstraction's table
            // proves the dead end, so its table alone is the heuristic, as in
            // Scorpion.
            Some(abstraction_id) => {
                info!("scp: the initial state is unsolvable");
                let mut cp = CostPartitioningHeuristic::default();
                cp.add_h_values(
                    abstraction_id,
                    selection.goal_distances_by_abstraction[abstraction_id].clone(),
                );
                vec![cp]
            }
            None => builder.build_cost_partitionings(
                &components,
                &mut selection,
                &abstract_state_ids,
            )?,
        };
        let lookup_mask = compute_lookup_mask(&cost_partitionings, components.len());
        for component in &mut components {
            component.discard_transitions();
        }
        info!(
            "scp: retained {} cost partitionings",
            cost_partitionings.len()
        );
        Ok(Self {
            name: name.unwrap_or_else(|| "scp".to_string()),
            components,
            cost_partitionings,
            lookup_mask,
            numeric_values,
            abstract_state_ids,
        })
    }
}

impl ScpBuilder {
    /// The cost partitionings of the collection for the state with the
    /// abstract states `abstract_state_ids`: the greedy order's, or the
    /// diverse orders' over the random-walk samples, which are pairwise
    /// distinct.
    fn build_cost_partitionings(
        &self,
        components: &[AbstractionComponent],
        state: &mut OrderSelection,
        abstract_state_ids: &[Option<usize>],
    ) -> Result<Vec<CostPartitioningHeuristic>, EvaluationError> {
        let budget = self.construction_deadline.earliest(deadline_from_now(
            "table_construction_max_time",
            self.config
                .table_construction_max_time
                .duration()
                .expect("ScpConfig::validate checked the time limits"),
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
                .expect("ScpConfig::validate checked the time limits"),
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

impl Heuristic for SaturatedCostPartitioningHeuristic {
    fn evaluate(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<Estimate, EvaluationError> {
        abstract_state_ids_into(
            &self.components,
            eval_state,
            Some(&self.lookup_mask),
            &mut self.numeric_values,
            &mut self.abstract_state_ids,
        )?;
        Ok(Estimate::new(compute_max_h(
            &self.cost_partitionings,
            &self.abstract_state_ids,
        )))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests;

/// `scp(sources, ...)`: saturated cost partitioning over the abstractions of
/// the sources.
#[derive(Debug)]
pub struct ScpFeature;
impl HeuristicFeature for ScpFeature {
    const NAME: &'static str = "scp";
    type Config = CombinationConfig<ScpConfig>;
    fn requirements(config: &Self::Config) -> TaskRequirements {
        config.requirements()
    }
    fn validate(config: &Self::Config) -> Result<(), ConfigError> {
        config.validate()?;
        config.options.validate()
    }
    fn build(
        config: &Self::Config,
        context: &BuildContext,
    ) -> Result<Box<dyn Heuristic>, HeuristicBuildError> {
        let deadline = config.deadline()?;
        let purpose = if config.options.partitioning.uses_regions() {
            AbstractionPurpose::RegionalCostPartitioning
        } else {
            AbstractionPurpose::LabelCostPartitioning
        };
        let components = build_components(&context.task, &config.sources, purpose, deadline)?;
        remaining_construction_time(deadline)?;
        let heuristic = SaturatedCostPartitioningHeuristic::new(
            Some(context.name.clone()),
            components,
            config.options.clone(),
            context.task.clone(),
            deadline,
        )
        .map_err(|error| format!("failed to construct `{}`: {error}", context.name))?;
        Ok(Box::new(heuristic))
    }
}
