//! The heuristic trait and the estimates it returns.

use crate::evaluation::{EvaluationError, EvaluationState};
use planforge_sas::numeric_task::OperatorIndex;
use planforge_sas::numeric_task::Task;

/// A heuristic estimate: a non-negative cost of reaching a goal, or a dead
/// end, a state from which the heuristic proved no goal is reachable.
///
/// The estimate is an `f64` underneath, with a dead end as `f64::INFINITY`,
/// so it costs nothing over a plain number, and estimates compare as their
/// values do: a dead end is larger than every finite estimate.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Estimate(f64);

impl Estimate {
    /// The estimate of a dead end.
    pub const DEAD_END: Self = Self(f64::INFINITY);

    /// The estimate `h`; `f64::INFINITY` is a dead end, as in a goal-distance
    /// table.
    ///
    /// # Panics
    ///
    /// If `h` is NaN or negative: a heuristic that computes such a value has
    /// a bug.
    #[inline]
    #[track_caller]
    pub fn new(h: f64) -> Self {
        assert!(h >= 0.0, "a heuristic estimate is non-negative, got {h}");
        Self(h)
    }

    /// Whether the heuristic proved the state a dead end.
    #[inline]
    pub fn is_dead_end(self) -> bool {
        self.0 == f64::INFINITY
    }

    /// The estimated cost; `f64::INFINITY` for a dead end.
    #[inline]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl std::fmt::Display for Estimate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// A heuristic: an estimate of the cost of reaching a goal from a state.
///
/// Evaluating takes `&mut self` because heuristics keep state between
/// evaluations -- caches, scratch buffers, online-learned components -- and
/// the search owns the heuristic it calls. The search keeps no cache of
/// estimates: a heuristic that wants one keeps it itself.
///
/// A heuristic owns what it needs, such as an `Arc<Task>` and its own tables,
/// and is `Send`, so it can be built on one thread and evaluate on another.
pub trait Heuristic: Send {
    /// The estimated cost of reaching a goal from `state`.
    fn evaluate(&mut self, state: &EvaluationState<'_, '_>) -> Result<Estimate, EvaluationError>;

    /// [`Self::evaluate`], additionally appending to `preferred` the
    /// operators applicable in `state` that the heuristic recommends (helpful
    /// actions). A heuristic without preferred operators appends none.
    fn evaluate_with_preferred(
        &mut self,
        state: &EvaluationState<'_, '_>,
        _preferred: &mut Vec<OperatorIndex>,
    ) -> Result<Estimate, EvaluationError> {
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
    pub fn for_task(task: &Task) -> Self {
        Self {
            name: "blind_heuristic".to_string(),
            min_action_cost: minimum_action_cost(task.effective_operator_costs()),
        }
    }
}

impl Heuristic for BlindHeuristic {
    fn evaluate(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<Estimate, EvaluationError> {
        Ok(Estimate::new(if eval_state.is_goal() {
            0.0
        } else {
            self.min_action_cost
        }))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// The cheapest operator cost: a lower bound on the remaining cost of every
/// non-goal state, and zero for a task without operators. Tasks reject
/// negative costs when they are constructed.
pub fn minimum_action_cost(operator_costs: &[f64]) -> f64 {
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
    fn build(
        _: &Self::Config,
        context: &crate::heuristic_factory::BuildContext,
    ) -> Result<Box<dyn Heuristic>, crate::heuristic_factory::HeuristicBuildError> {
        Ok(Box::new(BlindHeuristic::for_task(&context.task)))
    }
}
