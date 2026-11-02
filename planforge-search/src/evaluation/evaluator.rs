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

// What an evaluation reads is shared, so the states of one expansion can be
// evaluated on several threads without copying them.
const _: () = {
    const fn assert_sync<T: Sync + ?Sized>() {}
    assert_sync::<Task>();
    assert_sync::<StateRegistry>();
    assert_sync::<EvaluationState<'static, 'static>>();
};

/// State and task context for a single heuristic evaluation.
pub struct EvaluationState<'state, 'task> {
    backing_state: &'state ConcreteState,
    task: &'task Task,
    state_registry: &'state StateRegistry,
    is_goal: bool,
}

impl<'state, 'task> EvaluationState<'state, 'task> {
    /// The registered `state` of `task`, decoded through `state_registry`;
    /// whether it is a goal is checked here.
    pub fn new(
        state: &'state ConcreteState,
        task: &'task Task,
        state_registry: &'state StateRegistry,
    ) -> Self {
        let is_goal = state_registry.view(state).holds_all(task.goals());
        Self::with_goal_flag(state, task, state_registry, is_goal)
    }

    /// [`Self::new`] for a search that already knows whether `state` is a
    /// goal.
    pub(crate) fn with_goal_flag(
        state: &'state ConcreteState,
        task: &'task Task,
        state_registry: &'state StateRegistry,
        is_goal: bool,
    ) -> Self {
        Self {
            backing_state: state,
            task,
            state_registry,
            is_goal,
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

    /// Whether the state satisfies the task's goal.
    pub fn is_goal(&self) -> bool {
        self.is_goal
    }
}
