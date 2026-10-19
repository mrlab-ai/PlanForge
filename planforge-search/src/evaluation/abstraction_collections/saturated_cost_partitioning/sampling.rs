use super::*;

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    pub(super) fn sample_states(
        &self,
        state: &mut ScpState,
        initial_h: f64,
        deadline: Option<Instant>,
    ) -> Result<(), EvaluationError> {
        if !state.sample_abstract_state_ids.is_empty() {
            return Ok(());
        }
        let sampling_task = self
            .sampling_task
            .borrow_mut()
            .take()
            .expect("offline diversification was validated to have an owned sampling task");
        let mut registry = StateRegistry::for_task(sampling_task.clone());
        let successor_generator = SuccessorTree::new(&*sampling_task);
        let initial_state = registry.get_initial_state();
        let average_cost = if self.original_operator_costs.is_empty() {
            0.0
        } else {
            self.original_operator_costs.iter().sum::<f64>()
                / self.original_operator_costs.len() as f64
        };
        let mut applicable = Vec::new();
        let mut propositional = Vec::new();
        let mut successor_numeric = Vec::new();
        let mut successor_cost = Vec::new();
        let mut expansion_context = ExpansionContext::default();
        let mut ids = Vec::new();

        self.abstract_state_ids_of_sample(&initial_state, &registry, self.task, &mut ids)?;
        state.sample_abstract_state_ids.push(ids.clone());

        while state.sample_abstract_state_ids.len() < self.config.samples {
            if deadline.is_some_and(|end| Instant::now() >= end) {
                info!(
                    "scp: offline sampling deadline reached after {} samples",
                    state.sample_abstract_state_ids.len()
                );
                break;
            }
            let walk_length = random_walk_length(initial_h, average_cost, &mut state.rng)?;
            let mut current = initial_state.clone();
            for _ in 0..walk_length {
                if deadline.is_some_and(|end| Instant::now() >= end) {
                    break;
                }
                current.fill_state(&registry, &mut propositional);
                applicable.clear();
                successor_generator.get_applicable_operators(&propositional, &mut applicable);
                let Some(&operator_id) = applicable.choose(&mut state.rng) else {
                    break;
                };
                registry
                    .build_expansion_context(&current, &mut expansion_context)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to build random-walk expansion context: {error:?}"
                        ))
                    })?;
                let operator = sampling_task
                    .get_operators()
                    .get(operator_id.index())
                    .expect("successor generator returned an invalid operator id");
                let (successor, _) = registry
                    .apply_operator_in_context(
                        &current,
                        operator,
                        &expansion_context,
                        &mut successor_numeric,
                        &mut successor_cost,
                    )
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to apply random-walk operator {}: {error:?}",
                            operator.name()
                        ))
                    })?;
                current = successor;
            }
            self.abstract_state_ids_of_sample(&current, &registry, self.task, &mut ids)?;
            state.sample_abstract_state_ids.push(ids.clone());
        }
        info!(
            "scp: generated {} offline random-walk samples",
            state.sample_abstract_state_ids.len()
        );
        Ok(())
    }

    pub(super) fn abstract_state_ids_of_sample(
        &self,
        concrete_state: &ConcreteState,
        registry: &StateRegistry<'task>,
        task: &'task dyn AbstractNumericTask,
        ids: &mut Vec<Option<usize>>,
    ) -> Result<(), EvaluationError> {
        let eval_state = EvaluationState::new(concrete_state, task, registry);
        self.compute_abstract_state_ids_into(&eval_state, None, ids)?;
        Ok(())
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
    let trials_f64 = if initial_h <= f64::EPSILON || average_operator_cost <= f64::EPSILON {
        10.0
    } else {
        4.0 * (initial_h / average_operator_cost).round()
    };
    if trials_f64 > usize::MAX as f64 {
        return Err(EvaluationError::ComputationFailed(format!(
            "offline SCP random-walk trial count does not fit usize: {trials_f64}"
        )));
    }
    let trials = trials_f64 as usize;
    Ok((0..trials).filter(|_| rng.gen_bool(0.5)).count())
}
