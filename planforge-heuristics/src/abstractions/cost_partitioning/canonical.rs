#[cfg(test)]
mod tests;

use planforge_sas::numeric_task::Task;
use std::collections::BTreeSet;

use planforge_sas::numeric_task::OperatorIndex;
use tracing::{debug, info};

use crate::abstractions::sources::{
    AbstractionPurpose, CombinationConfig, build_components, remaining_construction_time,
};
use crate::maximal_cliques::maximal_cliques;
use planforge_search::config::ConfigError;
use planforge_search::evaluation::Heuristic;
use planforge_search::evaluation::state_value_cache::StateValueCache;
use planforge_search::evaluation::{Estimate, EvaluationError, EvaluationState};
use planforge_search::heuristic_features::{
    BuildContext, HeuristicBuildError, HeuristicFeature, TaskRequirements,
};

use planforge_sas::numeric_task::NumericValue;

use super::standalone_estimate;
use crate::abstractions::component::AbstractionComponent;

pub struct CanonicalAbstractionHeuristic {
    name: String,
    components: Vec<AbstractionComponent>,
    max_additive_subsets: Vec<Vec<usize>>,
    /// The estimates of the components on the state under evaluation.
    component_values: Vec<Option<f64>>,
    /// Estimates by the state ids of the search's registry.
    state_value_cache: StateValueCache,
    numeric_values: Vec<NumericValue>,
}

impl CanonicalAbstractionHeuristic {
    /// The canonical heuristic over the maximal additive subsets of
    /// `components`, which it reads only the goal distances and relevant
    /// operators of: components built for standalone use keep no
    /// transition data.
    pub fn new(
        name: Option<String>,
        task: &Task,
        components: Vec<AbstractionComponent>,
    ) -> Result<Self, HeuristicBuildError> {
        if components.is_empty() {
            return Err("canonical abstraction heuristic requires at least one component".into());
        }
        let relevant_operator_ids: Vec<BTreeSet<OperatorIndex>> = components
            .iter()
            .map(|component| component.relevant_operator_ids(task))
            .collect::<Result<Vec<_>, _>>()?;
        let subsets = compute_max_additive_subsets_from_relevant_operators(&relevant_operator_ids);
        Ok(Self::from_validated_parts(
            name,
            components,
            subsets,
            &relevant_operator_ids,
        ))
    }

    fn from_validated_parts(
        name: Option<String>,
        components: Vec<AbstractionComponent>,
        max_additive_subsets: Vec<Vec<usize>>,
        relevant_operator_ids: &[BTreeSet<OperatorIndex>],
    ) -> Self {
        assert!(
            !max_additive_subsets.is_empty(),
            "a nonempty collection has an additive subset"
        );
        info!(
            "canonical abstraction heuristic: components={}, max_additive_subsets={}",
            components.len(),
            max_additive_subsets.len()
        );
        for (component_id, component) in components.iter().enumerate() {
            debug!(
                "canonical component {component_id}: kind={}, states={}, relevant_ops={}",
                component.kind(),
                component.num_states(),
                relevant_operator_ids[component_id].len(),
            );
        }
        Self {
            name: name.unwrap_or_else(|| "canonical_abstractions".to_string()),
            component_values: vec![None; components.len()],
            components,
            max_additive_subsets,
            state_value_cache: StateValueCache::default(),
            numeric_values: Vec::new(),
        }
    }

    /// The best sum over the additive subsets, estimating each component at
    /// most once.
    fn evaluate_subsets(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let state = eval_state.values(&mut self.numeric_values)?;
        self.component_values.fill(None);
        let mut best = 0.0_f64;
        for subset in &self.max_additive_subsets {
            let mut sum = 0.0_f64;
            for &component_id in subset {
                let value = match self.component_values[component_id] {
                    Some(value) => value,
                    None => {
                        let value = standalone_estimate(
                            &self.components[component_id],
                            component_id,
                            state,
                        );
                        self.component_values[component_id] = Some(value);
                        value
                    }
                };
                if value == f64::INFINITY {
                    return Ok(f64::INFINITY);
                }
                sum += value;
            }
            best = best.max(sum);
        }
        Ok(best)
    }
}

impl Heuristic for CanonicalAbstractionHeuristic {
    fn evaluate(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<Estimate, EvaluationError> {
        let state_id = eval_state.state().id();
        if let Some(value) = self.state_value_cache.get(state_id) {
            return Ok(Estimate::new(value));
        }
        let value = self.evaluate_subsets(eval_state)?;
        self.state_value_cache.insert(state_id, value);
        Ok(Estimate::new(value))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

pub(crate) fn are_operator_sets_additive(
    left: &BTreeSet<OperatorIndex>,
    right: &BTreeSet<OperatorIndex>,
) -> bool {
    !left.iter().any(|operator_id| right.contains(operator_id))
}

/// The maximal sets of components no two of which share a relevant operator.
pub(crate) fn compute_max_additive_subsets_from_relevant_operators(
    relevant_operators: &[BTreeSet<OperatorIndex>],
) -> Vec<Vec<usize>> {
    let mut maximal_cliques = maximal_cliques(relevant_operators.len(), |left, right| {
        are_operator_sets_additive(&relevant_operators[left], &relevant_operators[right])
    });
    maximal_cliques.sort();
    maximal_cliques.dedup();
    maximal_cliques
}

/// `canonical(sources)`: the canonical heuristic over the abstractions of
/// the sources.
pub struct CanonicalFeature;
impl HeuristicFeature for CanonicalFeature {
    const NAME: &'static str = "canonical";
    type Config = CombinationConfig;
    fn requirements(config: &Self::Config) -> TaskRequirements {
        config.requirements()
    }
    fn validate(config: &Self::Config) -> Result<(), ConfigError> {
        config.validate()
    }
    fn build(
        config: &Self::Config,
        context: &BuildContext,
    ) -> Result<Box<dyn Heuristic>, HeuristicBuildError> {
        let deadline = config.deadline()?;
        let task = &*context.task;
        let components = build_components(
            task,
            &config.sources,
            AbstractionPurpose::Standalone,
            deadline,
        )?;
        remaining_construction_time(deadline)?;
        let heuristic =
            CanonicalAbstractionHeuristic::new(Some(context.name.clone()), task, components)
                .map_err(|error| format!("failed to construct `{}`: {error}", context.name))?;
        remaining_construction_time(deadline)?;
        Ok(Box::new(heuristic))
    }
}
