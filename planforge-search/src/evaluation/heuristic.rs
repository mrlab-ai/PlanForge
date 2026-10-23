//! Base trait for heuristic evaluators.

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use planforge_sas::numeric_task::{AbstractNumericTask, OperatorIndex};

/// Base trait for heuristic functions.
///
/// This replaces the C++ Heuristic class with a clean trait-based design.
/// Heuristics are specialized evaluators that estimate the cost to reach the goal.
pub trait Heuristic {
    /// Compute the heuristic value for the given state.
    ///
    /// This is the core method that sub-classes must implement.
    /// Return the estimated cost to reach the goal, or infinity for dead ends.
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError>;

    /// Return true when construction proved that the initial heuristic value
    /// equals the optimal solution cost.
    ///
    /// A* uses this stronger contract to assert that it never enters a higher
    /// f-layer. Heuristics must not return true based on an estimate or a
    /// resource-limited construction.
    fn proves_initial_state_optimal(&self) -> bool {
        false
    }

    /// Monotonic version of the heuristic function used for open-list keys.
    ///
    /// Static heuristics keep the default revision zero. A dynamic admissible
    /// heuristic increments this value only when its pointwise estimate can
    /// increase (for example, after adding a certified potential to a maximum
    /// ensemble). A* can then opt into pop-time re-evaluation without adding
    /// version fields to every open-list entry.
    fn revision(&self) -> u64 {
        0
    }

    /// Whether MPD must recompute this heuristic on every pop.
    ///
    /// This is the uncached C++ heuristic contract: without cached estimates,
    /// MPD cannot use a clean-cache marker to skip the pop-time computation.
    fn reevaluate_on_every_pop(&self) -> bool {
        false
    }

    /// Get the name of this heuristic (it allows custom names).
    fn heuristic_name(&self) -> &str;

    /// Get the task-operator indices of preferred operators for the most
    /// recently evaluated state.
    ///
    /// The search engine snapshots this immediately after `compute_heuristic`
    /// returns and stores the IDs on the successor's search-node record.
    /// Returning indices instead of cloned `Operator`s lets the engine
    /// check "was operator `O` preferred by my parent?" with a constant-time
    /// integer comparison and avoids cloning per-state precondition vectors.
    ///
    /// The default leaves `out` empty for heuristics that do not implement
    /// preferred operators.
    fn copy_preferred_operator_ids(&self, out: &mut Vec<OperatorIndex>) {
        out.clear();
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
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        // `Blind heuristic`: `0` for goal states, `min_action_cost` otherwise.
        Ok(if eval_state.is_goal() {
            0.0
        } else {
            self.min_action_cost
        })
    }

    fn heuristic_name(&self) -> &str {
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
