use crate::config::ConfigError;
use crate::evaluation::abstractions::features::max_over_sources;
use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;
use crate::evaluation::maximum::MaximumHeuristic;
use crate::heuristic_factory::{
    BuildContext, HeuristicBuildError, HeuristicConfig, HeuristicFeature,
};

use planforge_sas::numeric_task::NumericValue;

use super::standalone_estimate;
use crate::evaluation::abstractions::component::AbstractionComponent;

pub struct MaxAbstractionHeuristic<'task> {
    name: String,
    components: Vec<AbstractionComponent<'task>>,
    numeric_values: Vec<NumericValue>,
}

impl<'task> MaxAbstractionHeuristic<'task> {
    /// The maximum over `components`, which it reads only the goal distances
    /// of: components built for standalone use keep no transition data.
    pub fn new(
        name: Option<String>,
        components: Vec<AbstractionComponent<'task>>,
    ) -> Result<Self, String> {
        if components.is_empty() {
            return Err("max abstraction heuristic requires at least one component".to_string());
        }
        Ok(Self {
            name: name.unwrap_or_else(|| "max_abstractions".to_string()),
            components,
            numeric_values: Vec::new(),
        })
    }
}

impl Heuristic for MaxAbstractionHeuristic<'_> {
    fn evaluate(&mut self, eval_state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        let state = eval_state.values(&mut self.numeric_values)?;
        let mut best = 0.0_f64;
        for (component_id, component) in self.components.iter().enumerate() {
            best = best.max(standalone_estimate(component, component_id, state));
        }
        Ok(best)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// The heuristics a maximum ranges over: one list, or a single heuristic.
#[derive(Debug, Clone, Default, crate::config::ApplyOptions)]
pub struct MaxConfig {
    #[option(positional)]
    pub heuristics: Vec<HeuristicConfig>,
}

/// `max(h_1, ..., h_n)`. The abstraction heuristics among the children are
/// one maximum over their abstractions; the others are maximized with it.
pub struct MaxFeature;
impl HeuristicFeature for MaxFeature {
    const NAME: &'static str = "max";
    type Config = MaxConfig;
    fn nested(config: &Self::Config) -> Vec<&HeuristicConfig> {
        config.heuristics.iter().collect()
    }
    fn validate(config: &Self::Config) -> Result<(), ConfigError> {
        if config.heuristics.is_empty() {
            return Err("max requires at least one heuristic".into());
        }
        Ok(())
    }
    fn build<'a>(
        config: &Self::Config,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        let mut sources = Vec::new();
        let mut others = Vec::new();
        for child in &config.heuristics {
            match child.as_abstraction_source() {
                Some(source) => sources.push(source),
                None => others.push(child),
            }
        }
        if others.is_empty() {
            return max_over_sources(&**context.task, &sources, &context.name);
        }
        let mut children = Vec::new();
        if !sources.is_empty() {
            children.push(max_over_sources(&**context.task, &sources, &context.name)?);
        }
        for child in others {
            children.push(child.build(context)?);
        }
        Ok(Box::new(MaximumHeuristic::new(
            context.name.clone(),
            children,
        )))
    }
}
