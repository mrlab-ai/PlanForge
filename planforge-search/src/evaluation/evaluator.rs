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
    task: Option<&'task dyn AbstractNumericTask>,
    state_registry: Option<&'state StateRegistry<'task>>,
    is_goal: bool,
}

impl<'state, 'task> EvaluationState<'state, 'task> {
    pub fn new(state: &'state ConcreteState, _g_value: f64, _is_preferred: bool) -> Self {
        Self {
            backing_state: state,
            task: None,
            state_registry: None,
            is_goal: false,
        }
    }

    /// Create a new evaluation state with access to task and state registry.
    ///
    /// Heuristics that need to inspect the concrete state's variable values
    /// should require these to be present.
    pub fn new_with_registry(
        state: &'state ConcreteState,
        g_value: f64,
        is_preferred: bool,
        task: &'task dyn AbstractNumericTask,
        state_registry: &'state StateRegistry<'task>,
    ) -> Self {
        let mut s = Self::new(state, g_value, is_preferred);
        s.task = Some(task);
        s.state_registry = Some(state_registry);
        s
    }

    /// Borrowed concrete state being evaluated.
    pub fn state(&self) -> &'state ConcreteState {
        self.backing_state
    }

    /// Task reference, if provided.
    pub fn task(&self) -> Option<&'task dyn AbstractNumericTask> {
        self.task
    }

    /// State registry reference, if provided.
    pub fn state_registry(&self) -> Option<&'state StateRegistry<'task>> {
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
