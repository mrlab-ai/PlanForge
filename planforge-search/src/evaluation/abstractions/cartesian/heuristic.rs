use super::*;

pub struct CartesianAbstractionHeuristic {
    pub(super) name: String,
    pub(super) abstraction: CartesianAbstraction,
    pub(super) prop_scratch: std::cell::RefCell<Vec<ExplicitValueIndex>>,
    pub(super) numeric_scratch: std::cell::RefCell<Vec<NumericValue>>,
}

impl CartesianAbstractionHeuristic {
    pub fn new(name: Option<String>, abstraction: CartesianAbstraction) -> Self {
        Self {
            name: name.unwrap_or_else(|| "cartesian_abstraction".to_string()),
            abstraction,
            prop_scratch: std::cell::RefCell::new(Vec::new()),
            numeric_scratch: std::cell::RefCell::new(Vec::new()),
        }
    }

    pub fn abstraction(&self) -> &CartesianAbstraction {
        &self.abstraction
    }

    pub fn discard_transition_data(&mut self) {
        self.abstraction.discard_transition_data();
    }

    pub fn into_abstraction(self) -> CartesianAbstraction {
        self.abstraction
    }

    pub fn abstract_state_id(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<usize, EvaluationError> {
        let registry = eval_state.state_registry();
        let mut prop = self.prop_scratch.borrow_mut();
        eval_state.state().fill_state(registry, &mut prop);
        let mut numeric = self.numeric_scratch.borrow_mut();
        registry
            .fill_numeric_vars(eval_state.state(), &mut numeric)
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to read numeric state for Cartesian abstraction: {error:?}"
                ))
            })?;
        self.abstraction
            .hierarchy
            .map_state(&prop, &numeric)
            .map_err(|error| EvaluationError::ComputationFailed(error.to_string()))
    }
}

impl Heuristic for CartesianAbstractionHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let state_id = self.abstract_state_id(eval_state)?;
        self.abstraction
            .distance_table
            .distances
            .get(state_id)
            .copied()
            .ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "Cartesian abstract state id {state_id} out of bounds"
                ))
            })
    }

    fn proves_initial_state_optimal(&self) -> bool {
        self.abstraction.metadata.solved_by_self
            && self
                .abstraction
                .metadata
                .abstraction_use
                .permits_initial_optimality_proof()
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}
