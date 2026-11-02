//! The random walks of Haslum et al. (2007) that iPDB and the order
//! diversification of saturated cost partitioning (Seipp et al. 2020) sample
//! states with.

use planforge_sas::numeric_task::{OperatorIndex, Task};
use planforge_sas::state_registry::{ConcreteState, StateRegistry};
use planforge_sas::utils::float_tolerance;
use rand::Rng;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use std::sync::Arc;

use crate::evaluation::EvaluationError;
use crate::resource_limits::Deadline;
use crate::successor_generator::{RegistryExpander, StandardGroundedTree};

/// Samples states as the ends of random walks from the initial state.
pub struct RandomWalkSampler {
    task: Arc<Task>,
    registry: StateRegistry,
    expander: RegistryExpander<StandardGroundedTree>,
    initial_state: ConcreteState,
    average_operator_cost: f64,
}

impl RandomWalkSampler {
    pub fn new(task: Arc<Task>) -> Self {
        let mut registry = StateRegistry::for_task(task.clone());
        let initial_state = registry.get_initial_state();
        let num_operators = task.operators().len();
        let average_operator_cost = if num_operators == 0 {
            0.0
        } else {
            (0..num_operators)
                .map(|id| task.operator_cost(OperatorIndex::from_usize(id)))
                .sum::<f64>()
                / num_operators as f64
        };
        Self {
            expander: RegistryExpander::new(StandardGroundedTree::new(&task)),
            task,
            registry,
            initial_state,
            average_operator_cost,
        }
    }

    /// The registry every sampled state lives in.
    pub fn registry(&self) -> &StateRegistry {
        &self.registry
    }

    pub fn initial_state(&self) -> &ConcreteState {
        &self.initial_state
    }

    /// The end of a random walk from the initial state, its length binomial
    /// around twice the goal distance `initial_h` estimates. A walk that
    /// reaches a state `is_dead_end` holds restarts from the initial state; a
    /// walk without an applicable operator ends early. `None` when `deadline`
    /// passes.
    pub fn sample(
        &mut self,
        initial_h: f64,
        rng: &mut SmallRng,
        deadline: Deadline,
        mut is_dead_end: impl FnMut(&StateRegistry, &ConcreteState) -> Result<bool, EvaluationError>,
    ) -> Result<Option<ConcreteState>, EvaluationError> {
        let length = random_walk_length(initial_h, self.average_operator_cost, rng)?;
        let mut current = self.initial_state.clone();
        for _ in 0..length {
            if deadline.passed() {
                return Ok(None);
            }
            let mut expansion =
                self.expander
                    .expand(&self.registry, &current)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to expand a random-walk state: {error:?}"
                        ))
                    })?;
            let Some(&operator_id) = expansion.applicable_operators().choose(rng) else {
                break;
            };
            let operator = &self.task.operators()[operator_id.index()];
            let (successor, _) =
                expansion
                    .successor(&mut self.registry, operator)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to apply random-walk operator {}: {error:?}",
                            operator.name()
                        ))
                    })?;
            current = if is_dead_end(&self.registry, &successor)? {
                self.initial_state.clone()
            } else {
                successor
            };
        }
        Ok(Some(current))
    }
}

/// A random walk length from the estimate `initial_h` of the initial state.
fn random_walk_length(
    initial_h: f64,
    average_operator_cost: f64,
    rng: &mut SmallRng,
) -> Result<usize, EvaluationError> {
    if !initial_h.is_finite() || initial_h < 0.0 {
        return Err(EvaluationError::ComputationFailed(format!(
            "random-walk sampling requires a finite non-negative initial h, got {initial_h}"
        )));
    }
    if !average_operator_cost.is_finite() || average_operator_cost < 0.0 {
        return Err(EvaluationError::ComputationFailed(format!(
            "random-walk sampling requires a finite non-negative average operator cost, got {average_operator_cost}"
        )));
    }
    let trials_f64 = if initial_h <= float_tolerance::SEARCH_EPSILON
        || average_operator_cost <= float_tolerance::SEARCH_EPSILON
    {
        UNSCALED_WALK_TRIALS
    } else {
        TRIALS_PER_ESTIMATED_STEP * (initial_h / average_operator_cost).round()
    };
    if trials_f64 > usize::MAX as f64 {
        return Err(EvaluationError::ComputationFailed(format!(
            "random-walk trial count does not fit usize: {trials_f64}"
        )));
    }
    let trials = trials_f64 as usize;
    Ok((0..trials)
        .filter(|_| rng.gen_bool(STEP_PROBABILITY))
        .count())
}

/// The walk length is binomial, `trials` trials with success probability
/// [`STEP_PROBABILITY`], as in the random-walk sampling of Haslum et al.
/// (2007) that Seipp et al. (2020) diversify over.
const STEP_PROBABILITY: f64 = 0.5;

/// Trials per estimated step to the goal, `h / average operator cost`: the
/// expected walk length is twice the estimated goal distance.
const TRIALS_PER_ESTIMATED_STEP: f64 = 4.0;

/// Trials when the initial state's estimate or the average operator cost is
/// zero and gives no estimated goal distance.
const UNSCALED_WALK_TRIALS: f64 = 10.0;

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::random_walk_length;

    #[test]
    fn random_walk_lengths_are_deterministic() {
        let mut left = SmallRng::seed_from_u64(7);
        let mut right = SmallRng::seed_from_u64(7);
        let left_lengths = (0..20)
            .map(|_| random_walk_length(110.0, 1.0, &mut left).unwrap())
            .collect::<Vec<_>>();
        let right_lengths = (0..20)
            .map(|_| random_walk_length(110.0, 1.0, &mut right).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(left_lengths, right_lengths);
        assert!(left_lengths.iter().all(|&length| length <= 440));
    }
}
