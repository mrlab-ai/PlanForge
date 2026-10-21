use super::*;

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    /// The abstract states in every component of the initial state and of
    /// the ends of random walks from it (Seipp et al. 2020), until there are
    /// `samples` or `deadline` passes. Takes the sampling task, which only
    /// diversification needs.
    pub(super) fn sample_states(
        &self,
        components: &[AbstractionComponent<'_>],
        state: &mut OrderSelection<'task>,
        initial_h: f64,
        deadline: Deadline,
    ) -> Result<Vec<Vec<Option<usize>>>, EvaluationError> {
        let sampling_task = state
            .sampling_task
            .take()
            .expect("offline diversification was validated to have an owned sampling task");
        let mut registry = StateRegistry::for_task(sampling_task.clone());
        let mut expander = RegistryExpander::new(&*sampling_task);
        let initial_state = registry.get_initial_state();
        let average_cost = if self.original_operator_costs.is_empty() {
            0.0
        } else {
            self.original_operator_costs.iter().sum::<f64>()
                / self.original_operator_costs.len() as f64
        };
        let mut ids = Vec::new();
        let mut samples = Vec::new();

        self.abstract_state_ids_of_sample(components, &initial_state, &registry, &mut ids)?;
        samples.push(ids.clone());

        while samples.len() < self.config.samples {
            if deadline.passed() {
                info!(
                    "scp: offline sampling deadline reached after {} samples",
                    samples.len()
                );
                break;
            }
            let walk_length = random_walk_length(initial_h, average_cost, &mut state.rng)?;
            let mut current = initial_state.clone();
            for _ in 0..walk_length {
                if deadline.passed() {
                    break;
                }
                let mut expansion = expander.expand(&registry, &current).map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to expand a random-walk state: {error:?}"
                    ))
                })?;
                let Some(&operator_id) = expansion.applicable_operators().choose(&mut state.rng)
                else {
                    break;
                };
                let operator = sampling_task
                    .get_operators()
                    .get(operator_id.index())
                    .expect("successor generator returned an invalid operator id");
                let (successor, _) =
                    expansion
                        .successor(&mut registry, operator)
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to apply random-walk operator {}: {error:?}",
                                operator.name()
                            ))
                        })?;
                current = successor;
            }
            self.abstract_state_ids_of_sample(components, &current, &registry, &mut ids)?;
            samples.push(ids.clone());
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
        let eval_state = EvaluationState::new(concrete_state, self.task, registry);
        self.compute_abstract_state_ids_into(components, &eval_state, None, ids)
    }
}

pub(super) fn random_walk_length(
    initial_h: f64,
    average_operator_cost: f64,
    rng: &mut SmallRng,
) -> Result<usize, EvaluationError> {
    if !initial_h.is_finite() || initial_h < 0.0 {
        return Err(EvaluationError::ComputationFailed(format!(
            "offline SCP sampling requires a finite non-negative initial h, got {initial_h}"
        )));
    }
    if !average_operator_cost.is_finite() || average_operator_cost < 0.0 {
        return Err(EvaluationError::ComputationFailed(format!(
            "offline SCP sampling requires a finite non-negative average operator cost, got {average_operator_cost}"
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
            "offline SCP random-walk trial count does not fit usize: {trials_f64}"
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
