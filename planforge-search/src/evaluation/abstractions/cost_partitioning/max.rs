use std::cell::RefCell;

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;

use planforge_sas::explicit_state::ExplicitState;

use super::standalone_estimate;
use crate::evaluation::abstractions::component::AbstractionComponent;

pub struct MaxAbstractionHeuristic<'task> {
    name: String,
    components: Vec<AbstractionComponent<'task>>,
    state_values: RefCell<ExplicitState>,
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
            state_values: RefCell::default(),
        })
    }
}

impl Heuristic for MaxAbstractionHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let mut state_values = self.state_values.borrow_mut();
        eval_state.fill_values(&mut state_values)?;
        let mut best = 0.0_f64;
        for (component_id, component) in self.components.iter().enumerate() {
            best = best.max(standalone_estimate(component, component_id, &state_values)?);
        }
        Ok(best)
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}

pub use registry_feature::MaxFeature;
mod registry_feature {
    use crate::config::ConfigError;
    use crate::evaluation::{Heuristic, maximum::MaximumHeuristic};
    use crate::heuristic_factory::*;

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
                return build_max_from_sources(&**context.task, &sources, &context.name);
            }
            let mut children = Vec::new();
            if !sources.is_empty() {
                children.push(build_max_from_sources(
                    &**context.task,
                    &sources,
                    &context.name,
                )?);
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
}
