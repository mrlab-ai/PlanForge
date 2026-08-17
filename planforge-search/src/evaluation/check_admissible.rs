//! Admissibility guard around another heuristic.
//!
//! `check_admissible(<h>)` forwards every query to `<h>` and cross-checks the
//! returned estimate against the true goal distance of the same state, which it
//! computes by blind A* over the concrete task. An estimate above the true
//! distance costs A* its optimality guarantee, so the wrapper reports it as an
//! evaluation error instead of letting the search run on a broken bound.
//!
//! This is a debugging tool: it solves the remaining task from scratch for
//! every state it sees and is therefore far slower than the heuristic it
//! guards.

#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use ordered_float::NotNan;
use planforge_sas::numeric_task::{Operator, TaskRef};
use planforge_sas::state_registry::{ConcreteState, StateID, StateRegistry};

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::{BlindHeuristic, CostType, Heuristic};
use crate::search::compute_effective_operator_costs;
use crate::successor_generator::SuccessorTree;

/// Slack allowed on `h <= h*` before an estimate counts as inadmissible.
const ADMISSIBILITY_TOLERANCE: f64 = 1e-9;

/// A `g`-value improvement below this is float noise, not a cheaper path.
const G_IMPROVEMENT_TOLERANCE: f64 = 1e-12;

/// Wraps a heuristic and rejects any estimate that exceeds the true goal
/// distance of the evaluated state.
pub struct CheckAdmissibleHeuristic<'task> {
    inner: Box<dyn Heuristic + 'task>,
    name: String,
    oracle: RefCell<GoalDistanceOracle<'task>>,
    /// True goal distances keyed by the id the *search's* registry gave the
    /// state. The oracle numbers its own states independently, so its ids
    /// cannot be used here.
    goal_distances: RefCell<HashMap<StateID, f64>>,
}

impl<'task> CheckAdmissibleHeuristic<'task> {
    /// Wrap `inner`, verifying it against the goal distances of `task`.
    ///
    /// `inner` is `None` for `check_admissible(blind())`: the heuristic factory
    /// reports `blind` as "no heuristic object" and leaves the caller to
    /// materialize one from the task's minimum action cost, which the oracle
    /// computes anyway.
    pub fn new(
        inner: Option<Box<dyn Heuristic + 'task>>,
        task: TaskRef<'task>,
    ) -> Result<Self, String> {
        let oracle = GoalDistanceOracle::new(task)?;
        let inner = inner.unwrap_or_else(|| {
            Box::new(BlindHeuristic::with_min_action_cost(
                oracle.min_action_cost,
                None,
            ))
        });
        Ok(Self {
            // Keep the wrapper distinct from the heuristic it checks in diagnostics.
            name: format!("check_admissible_{}", inner.heuristic_name()),
            inner,
            oracle: RefCell::new(oracle),
            goal_distances: RefCell::new(HashMap::new()),
        })
    }

    /// True goal distance of the state under evaluation, memoized per state.
    fn goal_distance(&self, eval_state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        let state = eval_state.state();
        if let Some(&distance) = self.goal_distances.borrow().get(&state.get_id()) {
            return Ok(distance);
        }

        let registry = eval_state.state_registry();
        let distance = self
            .oracle
            .borrow_mut()
            .goal_distance(state, registry)
            .map_err(EvaluationError::ComputationFailed)?;
        self.goal_distances
            .borrow_mut()
            .insert(state.get_id(), distance);
        Ok(distance)
    }

    fn inadmissible(&self, state_id: StateID, h_value: f64, goal_distance: f64) -> EvaluationError {
        EvaluationError::ComputationFailed(format!(
            "`{}` is inadmissible on state {state_id}: it returned h = {h_value} for a state \
             whose true goal distance is h* = {goal_distance} (h - h* = {})",
            self.inner.heuristic_name(),
            h_value - goal_distance,
        ))
    }
}

impl Heuristic for CheckAdmissibleHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let h_value = self.inner.compute_heuristic(eval_state)?;
        let goal_distance = self.goal_distance(eval_state)?;
        if h_value > goal_distance + ADMISSIBILITY_TOLERANCE {
            return Err(self.inadmissible(eval_state.state().get_id(), h_value, goal_distance));
        }
        Ok(h_value)
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }

    fn proves_initial_state_optimal(&self) -> bool {
        self.inner.proves_initial_state_optimal()
    }

    fn revision(&self) -> u64 {
        self.inner.revision()
    }

    fn reevaluate_on_every_pop(&self) -> bool {
        self.inner.reevaluate_on_every_pop()
    }

    fn dead_ends_are_reliable(&self) -> bool {
        Heuristic::dead_ends_are_reliable(&*self.inner)
    }

    fn reach_state(
        &mut self,
        parent_state: &ConcreteState,
        operator: &Operator,
        state: &ConcreteState,
    ) -> bool {
        self.inner.reach_state(parent_state, operator, state)
    }

    fn get_preferred_operators(&self, state: &ConcreteState) -> Vec<Operator> {
        self.inner.get_preferred_operators(state)
    }

    fn copy_preferred_operator_ids(&self, out: &mut Vec<u32>) {
        self.inner.copy_preferred_operator_ids(out)
    }

    fn get_cost_type(&self) -> CostType {
        self.inner.get_cost_type()
    }

    fn print_statistics(&self) {
        self.inner.print_statistics();
    }
}

/// Exact goal distance of an arbitrary state, computed by blind A* over the
/// concrete task.
///
/// The oracle keeps its own [`StateRegistry`]: `EvaluationState` only lends an
/// immutable one, while generating successors needs a mutable one.
struct GoalDistanceOracle<'task> {
    task: TaskRef<'task>,
    registry: StateRegistry<'task>,
    successor_generator: SuccessorTree,
    operator_costs: Vec<f64>,
    min_action_cost: f64,
}

impl<'task> GoalDistanceOracle<'task> {
    fn new(task: TaskRef<'task>) -> Result<Self, String> {
        let mut registry = StateRegistry::for_task(task.clone());
        // Creating the initial state is what fixes the registry's numeric
        // layout (constant values and packed slots); `copy_state_in` needs it.
        registry.get_initial_state();
        let operator_costs = compute_effective_operator_costs(&*task);
        let min_action_cost = minimum_action_cost(&operator_costs)?;
        let successor_generator = SuccessorTree::new(&*task);
        Ok(Self {
            task,
            registry,
            successor_generator,
            operator_costs,
            min_action_cost,
        })
    }

    /// Cost of a cheapest plan from `state`, or `f64::INFINITY` when no goal is
    /// reachable. `state` belongs to `registry`, not to the oracle.
    fn goal_distance(
        &mut self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
    ) -> Result<f64, String> {
        let start = self.copy_state_in(state, registry)?;
        self.blind_a_star(start)
    }

    /// Re-register `state` in the oracle's own registry. The two registries
    /// number and pack states independently, so the copy goes through variable
    /// values rather than through the state id or the packed buffer.
    fn copy_state_in(
        &mut self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
    ) -> Result<ConcreteState, String> {
        let propositional_values: Vec<u64> = state
            .get_state(registry)
            .into_iter()
            .map(|value| value as u64)
            .collect();
        let numeric_values = registry.get_numeric_vars(state).map_err(|error| {
            format!(
                "could not read the numeric values of state {}: {error:?}",
                state.get_id()
            )
        })?;
        self.registry
            .register_state(propositional_values, numeric_values)
            .map_err(|error| {
                format!(
                    "could not copy state {} into the admissibility oracle: {}",
                    state.get_id(),
                    error.message
                )
            })
    }

    fn is_goal(&self, state: &ConcreteState) -> bool {
        (0..self.task.get_num_goals()).all(|index| {
            self.task
                .get_goal_fact(index)
                .is_hold(state, &self.registry)
        })
    }

    /// Blind estimate of the remaining cost: zero in a goal, the cheapest
    /// action otherwise.
    fn blind_estimate(&self, state: &ConcreteState) -> f64 {
        if self.is_goal(state) {
            0.0
        } else {
            self.min_action_cost
        }
    }

    /// A* with the blind estimate, i.e. Dijkstra with a goal test. Runs to
    /// completion; the caller is responsible for the overall time budget.
    fn blind_a_star(&mut self, start: ConcreteState) -> Result<f64, String> {
        if self.is_goal(&start) {
            return Ok(0.0);
        }

        // A borrowed handle keeps operator lookups off `self` so the registry
        // can be mutated inside the expansion loop.
        let task = self.task.clone();
        let mut best_g: HashMap<StateID, f64> = HashMap::new();
        let mut open: BinaryHeap<OpenEntry> = BinaryHeap::new();
        best_g.insert(start.get_id(), 0.0);
        open.push(open_entry(self.min_action_cost, 0.0, start.get_id())?);

        let mut propositional_values: Vec<usize> = Vec::new();
        let mut applicable_operators: Vec<u32> = Vec::new();
        let mut successor_numeric_values: Vec<f64> = Vec::new();
        let mut successor_cost_values: Vec<f64> = Vec::new();

        while let Some((Reverse(_f_value), Reverse(g_value), state_id)) = open.pop() {
            let g_value = g_value.into_inner();
            let best = *best_g.get(&state_id).ok_or_else(|| {
                format!("the admissibility oracle queued state {state_id} without a g-value")
            })?;
            if g_value > best + G_IMPROVEMENT_TOLERANCE {
                // A cheaper path was found after this entry was queued.
                continue;
            }

            let state = self.registry.lookup_state(state_id).map_err(|error| {
                format!("the admissibility oracle lost state {state_id}: {error:?}")
            })?;
            if self.is_goal(&state) {
                return Ok(g_value);
            }

            state.fill_state(&self.registry, &mut propositional_values);
            applicable_operators.clear();
            self.successor_generator
                .get_applicable_operators(&propositional_values, &mut applicable_operators);

            for &applicable_id in &applicable_operators {
                let operator_id = applicable_id as usize;
                let operator = &task.get_operators()[operator_id];
                let successor = self
                    .registry
                    .get_successor_state_with_buffers(
                        &state,
                        operator,
                        &mut successor_numeric_values,
                        &mut successor_cost_values,
                    )
                    .map_err(|error| {
                        format!(
                            "the admissibility oracle could not apply `{}`: {}",
                            operator.name(),
                            error.message
                        )
                    })?;
                let successor_g = g_value + self.operator_costs[operator_id];
                let successor_id = successor.get_id();
                let successor_best = best_g.get(&successor_id).copied().unwrap_or(f64::INFINITY);
                if successor_g + G_IMPROVEMENT_TOLERANCE >= successor_best {
                    continue;
                }

                best_g.insert(successor_id, successor_g);
                let successor_f = successor_g + self.blind_estimate(&successor);
                open.push(open_entry(successor_f, successor_g, successor_id)?);
            }
        }

        Ok(f64::INFINITY)
    }
}

/// `(f, g, state)` ordered so that the max-heap pops the smallest `f`, then the
/// smallest `g`.
type OpenEntry = (Reverse<NotNan<f64>>, Reverse<NotNan<f64>>, StateID);

fn open_entry(f_value: f64, g_value: f64, state_id: StateID) -> Result<OpenEntry, String> {
    let f_value = NotNan::new(f_value).map_err(|_| {
        format!("the admissibility oracle computed a NaN f-value for state {state_id}")
    })?;
    let g_value = NotNan::new(g_value).map_err(|_| {
        format!("the admissibility oracle computed a NaN g-value for state {state_id}")
    })?;
    Ok((Reverse(f_value), Reverse(g_value), state_id))
}

/// Cheapest operator in the task, used as the oracle's blind estimate.
///
/// Dijkstra is only correct for non-negative edge costs, so a task the oracle
/// cannot measure exactly is rejected here rather than silently mis-measured.
fn minimum_action_cost(operator_costs: &[f64]) -> Result<f64, String> {
    let mut minimum = f64::INFINITY;
    for (operator_id, &cost) in operator_costs.iter().enumerate() {
        if !cost.is_finite() || cost < 0.0 {
            return Err(format!(
                "`check_admissible` cannot verify a task whose operator {operator_id} costs \
                 {cost}; it needs finite non-negative costs"
            ));
        }
        minimum = minimum.min(cost);
    }
    // A task without operators has no transitions at all, so zero is the
    // tightest lower bound on the remaining cost.
    Ok(if operator_costs.is_empty() {
        0.0
    } else {
        minimum
    })
}
