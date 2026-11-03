use planforge_sas::explicit_state::{ExplicitState, StateValues};
use planforge_sas::numeric_task::NumericValue;
use planforge_sas::numeric_task::Task;
use planforge_sas::state_registry::ConcreteState;
use planforge_sas::state_registry::{StateID, StateRegistry};
use planforge_sas::utils::errors::AssignmentAxiomError;
use std::fmt;

/// Why a heuristic could not evaluate a state. A dead end is not an error:
/// the heuristic estimates it as [`super::Estimate::DEAD_END`].
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationError {
    /// The state's derived numeric values are undefined: an arithmetic axiom
    /// divides by zero.
    DerivedValues(AssignmentAxiomError),
    /// The LP solver failed; its message.
    Solver(String),
    /// A checked heuristic estimated more than the state's true goal
    /// distance (`check_admissible`).
    Inadmissible {
        heuristic: String,
        state: StateID,
        h: f64,
        goal_distance: f64,
    },
    /// Any other failure, such as a Python heuristic raising an exception.
    ComputationFailed(String),
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DerivedValues(error) => write!(
                f,
                "Evaluation failed: failed to evaluate the state's derived numeric values: {error}"
            ),
            Self::Solver(message) => write!(f, "Evaluation failed: LP solver: {message}"),
            Self::Inadmissible {
                heuristic,
                state,
                h,
                goal_distance,
            } => write!(
                f,
                "Evaluation failed: `{heuristic}` is inadmissible on state {state}: it returned \
                 h = {h} for a state whose true goal distance is h* = {goal_distance} (h - h* = {})",
                h - goal_distance
            ),
            Self::ComputationFailed(message) => write!(f, "Evaluation failed: {message}"),
        }
    }
}

impl From<AssignmentAxiomError> for EvaluationError {
    fn from(error: AssignmentAxiomError) -> Self {
        Self::DerivedValues(error)
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
        Ok(self
            .state_registry
            .view(self.backing_state)
            .fill_explicit(values)?)
    }

    /// The evaluated state's values, with its numeric values, derived ones
    /// included, decoded into `numeric` and its propositional values read
    /// from the registry on demand.
    pub fn values<'v>(
        &'v self,
        numeric: &'v mut Vec<NumericValue>,
    ) -> Result<StateValues<'v>, EvaluationError> {
        let view = self.state_registry.view(self.backing_state);
        view.fill_numeric(numeric)?;
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
