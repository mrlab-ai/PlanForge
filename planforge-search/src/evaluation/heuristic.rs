//! Base trait for heuristic evaluators.

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use planforge_sas::numeric_task::{AbstractNumericTask, OperatorIndex};

/// A heuristic: an estimate of the cost of reaching a goal from a state.
///
/// Evaluating takes `&mut self` because heuristics keep state between
/// evaluations -- caches, scratch buffers, online-learned components -- and
/// the search owns the heuristic it calls.
pub trait Heuristic {
    /// The estimated cost of reaching a goal from `state`; infinite for a
    /// state the heuristic proves to be a dead end.
    fn evaluate(&mut self, state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError>;

    /// [`Self::evaluate`], additionally appending to `preferred` the
    /// operators applicable in `state` that the heuristic recommends (helpful
    /// actions). A heuristic without preferred operators appends none.
    fn evaluate_with_preferred(
        &mut self,
        state: &EvaluationState<'_, '_>,
        _preferred: &mut Vec<OperatorIndex>,
    ) -> Result<f64, EvaluationError> {
        self.evaluate(state)
    }

    /// The name the heuristic reports itself under.
    fn name(&self) -> &str;

    /// Whether construction proved that the initial state's estimate is its
    /// optimal plan cost.
    ///
    /// A* then asserts that it never enters a higher f-layer, so a heuristic
    /// must not claim this from an estimate or a resource-limited
    /// construction.
    fn proves_initial_state_optimal(&self) -> bool {
        false
    }

    /// A counter that grows whenever the heuristic's estimates may have grown,
    /// such as when an online heuristic adds a component. A static heuristic
    /// keeps the default zero. A* with multi-path dependence (`mpd`)
    /// re-evaluates a popped state whose estimate is older than the current
    /// revision.
    fn revision(&self) -> u64 {
        0
    }

    /// Whether A* with multi-path dependence must re-evaluate every popped
    /// state, because the heuristic keeps no revision its estimates follow.
    fn reevaluate_on_every_pop(&self) -> bool {
        false
    }
}

/// A heuristic that returns `0` for goal states and `min_action_cost` for
/// non-goal states.
/// This implements the classical blind search heuristic behavior.
pub struct BlindHeuristic {
    name: String,
    // Cost to return for non-goal states (minimum action cost).
    min_action_cost: f64,
}

impl BlindHeuristic {
    /// Build the blind estimate from the task's cheapest effective action cost.
    pub fn for_task(task: &dyn AbstractNumericTask) -> Self {
        Self {
            name: "blind_heuristic".to_string(),
            min_action_cost: minimum_action_cost(&task.effective_operator_costs()),
        }
    }
}

impl Heuristic for BlindHeuristic {
    fn evaluate(&mut self, eval_state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        Ok(if eval_state.is_goal() {
            0.0
        } else {
            self.min_action_cost
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// The cheapest operator cost: a lower bound on the remaining cost of every
/// non-goal state, and zero for a task without operators. Tasks reject
/// negative costs when they are constructed.
pub(crate) fn minimum_action_cost(operator_costs: &[f64]) -> f64 {
    operator_costs
        .iter()
        .copied()
        .inspect(|&cost| {
            assert!(
                cost.is_finite() && cost >= 0.0,
                "operator costs are finite and non-negative, got {cost}"
            );
        })
        .reduce(f64::min)
        .unwrap_or(0.0)
}

/// Registry feature for the blind heuristic.
pub struct BlindFeature;
impl crate::heuristic_factory::HeuristicFeature for BlindFeature {
    const NAME: &'static str = "blind";
    type Config = crate::heuristic_factory::EmptyConfig;
    fn build<'a>(
        _: &Self::Config,
        context: &crate::heuristic_factory::BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, crate::heuristic_factory::HeuristicBuildError> {
        Ok(Box::new(BlindHeuristic::for_task(&**context.task)))
    }
}
