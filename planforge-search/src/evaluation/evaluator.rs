use planforge_sas::numeric_task::AbstractNumericTask;
use planforge_sas::state_registry::ConcreteState;
use planforge_sas::state_registry::StateRegistry;
use std::fmt;

/// Errors that can occur during evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationError {
    /// State is a dead end (no solution possible).
    DeadEnd { reliable: bool },
    /// Heuristic computation failed.
    ComputationFailed(String),
    /// Invalid state for evaluation.
    InvalidState(String),
    /// A bounded heuristic-construction attempt exhausted its own deadline.
    ConstructionDeadlineExceeded,
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvaluationError::DeadEnd { reliable } => {
                write!(f, "Dead end detected (reliable: {})", reliable)
            }
            EvaluationError::ComputationFailed(msg) => {
                write!(f, "Evaluation failed: {}", msg)
            }
            EvaluationError::InvalidState(msg) => {
                write!(f, "Invalid state: {}", msg)
            }
            EvaluationError::ConstructionDeadlineExceeded => {
                write!(f, "heuristic construction deadline exceeded")
            }
        }
    }
}

impl std::error::Error for EvaluationError {}

/// State and task context for a single heuristic evaluation.
pub struct EvaluationState<'state, 'task> {
    backing_state: &'state ConcreteState,
    task: &'task dyn AbstractNumericTask,
    state_registry: &'state StateRegistry<'task>,
    is_goal: bool,
}

impl<'state, 'task> EvaluationState<'state, 'task> {
    /// Create an evaluation state with its mandatory decoding context.
    pub fn new(
        state: &'state ConcreteState,
        task: &'task dyn AbstractNumericTask,
        state_registry: &'state StateRegistry<'task>,
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
    pub fn task(&self) -> &'task dyn AbstractNumericTask {
        self.task
    }

    /// State registry used to decode the concrete state.
    pub fn state_registry(&self) -> &'state StateRegistry<'task> {
        self.state_registry
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
