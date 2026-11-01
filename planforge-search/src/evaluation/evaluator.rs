use planforge_sas::explicit_state::{ExplicitState, StateValues};
use planforge_sas::numeric_task::NumericValue;
use planforge_sas::numeric_task::Task;
use planforge_sas::state_registry::ConcreteState;
use planforge_sas::state_registry::StateRegistry;
use std::fmt;

/// Why a heuristic could not evaluate a state. A dead end is not an error:
/// the heuristic estimates it as infinite.
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationError {
    /// Heuristic computation failed.
    ComputationFailed(String),
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvaluationError::ComputationFailed(msg) => {
                write!(f, "Evaluation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for EvaluationError {}

/// State and task context for a single heuristic evaluation.
pub struct EvaluationState<'state, 'task> {
    backing_state: &'state ConcreteState,
    task: &'task Task,
    state_registry: &'state StateRegistry,
    is_goal: bool,
}

impl<'state, 'task> EvaluationState<'state, 'task> {
    /// Create an evaluation state with its mandatory decoding context.
    pub fn new(
        state: &'state ConcreteState,
        task: &'task Task,
        state_registry: &'state StateRegistry,
    ) -> Self {
        Self {
            backing_state: state,
            task,
            state_registry,
            is_goal: false,
        }
    }

    /// Borrowed concrete state being evaluated.
    pub fn state(&self) -> &'state ConcreteState {
        self.backing_state
    }

    /// Task whose state is being evaluated.
    pub fn task(&self) -> &'task Task {
        self.task
    }

    /// State registry used to decode the concrete state.
    pub fn state_registry(&self) -> &'state StateRegistry {
        self.state_registry
    }

    /// Writes the evaluated state's values, derived numeric values included,
    /// into `values`.
    pub fn fill_values(&self, values: &mut ExplicitState) -> Result<(), EvaluationError> {
        self.state_registry
            .view(self.backing_state)
            .fill_explicit(values)
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to evaluate the state's derived numeric values: {error:?}"
                ))
            })
    }

    /// The evaluated state's values, with its numeric values, derived ones
    /// included, decoded into `numeric` and its propositional values read
    /// from the registry on demand.
    pub fn values<'v>(
        &'v self,
        numeric: &'v mut Vec<NumericValue>,
    ) -> Result<StateValues<'v>, EvaluationError> {
        let view = self.state_registry.view(self.backing_state);
        view.fill_numeric(numeric).map_err(|error| {
            EvaluationError::ComputationFailed(format!(
                "failed to evaluate the state's derived numeric values: {error:?}"
            ))
        })?;
        Ok(StateValues::packed(
            view.packer(),
            view.propositional(),
            numeric,
        ))
    }

    /// Mark whether this state is a goal.
    pub fn set_is_goal(&mut self, is_goal: bool) {
        self.is_goal = is_goal;
    }

    /// Query whether this state is a goal.
    pub fn is_goal(&self) -> bool {
        self.is_goal
    }
}
