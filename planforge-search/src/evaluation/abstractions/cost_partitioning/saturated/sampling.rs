use planforge_sas::state_registry::{ConcreteState, StateRegistry};
use tracing::info;

use super::lookup::CostPartitioningHeuristic;
use super::{OrderSelection, SaturatedCostPartitioningHeuristic};
use crate::evaluation::abstractions::component::AbstractionComponent;
use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::sampling::RandomWalkSampler;
use crate::resource_limits::Deadline;

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    /// The abstract states in every component of the initial state and of
    /// the ends of random walks from it (Seipp et al. 2020), until there are
    /// `samples` or `deadline` passes. A walk restarts at a dead end of
    /// `dead_end_cp`, as Scorpion's does. Takes the sampling task, which only
    /// diversification needs.
    pub(super) fn sample_states(
        &self,
        components: &[AbstractionComponent<'_>],
        state: &mut OrderSelection<'task>,
        dead_end_cp: &CostPartitioningHeuristic,
        initial_h: f64,
        deadline: Deadline,
    ) -> Result<Vec<Vec<Option<usize>>>, EvaluationError> {
        let sampling_task = state
            .sampling_task
            .take()
            .expect("offline diversification was validated to have an owned sampling task");
        let mut sampler = RandomWalkSampler::new(sampling_task);
        let mut ids = Vec::new();
        let mut samples = Vec::new();

        self.abstract_state_ids_of_sample(
            components,
            sampler.initial_state(),
            sampler.registry(),
            &mut ids,
        )?;
        samples.push(ids.clone());

        while samples.len() < self.config.samples && !deadline.passed() {
            let walk_end = sampler.sample(
                initial_h,
                &mut state.rng,
                deadline,
                |registry, walk_state| {
                    self.abstract_state_ids_of_sample(components, walk_state, registry, &mut ids)?;
                    Ok(dead_end_cp.compute_heuristic(&ids).is_infinite())
                },
            )?;
            let Some(sample) = walk_end else {
                break;
            };
            self.abstract_state_ids_of_sample(components, &sample, sampler.registry(), &mut ids)?;
            samples.push(ids.clone());
        }
        if samples.len() < self.config.samples {
            info!(
                "scp: offline sampling deadline reached after {} samples",
                samples.len()
            );
        }
        info!(
            "scp: generated {} offline random-walk samples",
            samples.len()
        );
        Ok(samples)
    }

    fn abstract_state_ids_of_sample(
        &self,
        components: &[AbstractionComponent<'_>],
        concrete_state: &ConcreteState,
        registry: &StateRegistry<'task>,
        ids: &mut Vec<Option<usize>>,
    ) -> Result<(), EvaluationError> {
        let eval_state = EvaluationState::new(concrete_state, &*self.task, registry);
        self.compute_abstract_state_ids_into(components, &eval_state, None, ids)
    }
}
