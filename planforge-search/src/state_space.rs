//! Exhaustive reachable-state enumeration and exact goal distances.

use std::cell::Cell;
use std::fmt;
use std::time::{Duration, Instant};

use anyhow::{Result as AnyResult, bail};
use ordered_float::NotNan;
use planforge_sas::numeric_task::TaskRef;
use planforge_sas::state_registry::{ExpansionContext, StateRegistry};

use crate::evaluation::abstraction_collections::cost_partitioning::{
    BackwardDijkstraGraph, backward_goal_distances,
};
use crate::successor_generator::SuccessorTree;

#[derive(Debug, Clone, Copy)]
pub struct EnumerationLimits {
    pub max_states: usize,
    pub max_transitions: usize,
    pub max_time: Duration,
}

impl EnumerationLimits {
    fn validate(self) -> Result<Self, StateSpaceEnumerationError> {
        if self.max_states == 0 {
            return Err(StateSpaceEnumerationError::InvalidLimit(
                "max_states must be greater than zero".to_string(),
            ));
        }
        if self.max_transitions == 0 {
            return Err(StateSpaceEnumerationError::InvalidLimit(
                "max_transitions must be greater than zero".to_string(),
            ));
        }
        Ok(self)
    }
}

#[derive(Debug)]
pub enum StateSpaceEnumerationError {
    InvalidLimit(String),
    StateLimit {
        limit: usize,
        states: usize,
        transitions: usize,
    },
    TransitionLimit {
        limit: usize,
        states: usize,
        transitions: usize,
    },
    TimeLimit {
        limit: Duration,
        states: usize,
        transitions: usize,
    },
    Enumeration(String),
    GoalDistances(String),
}

impl fmt::Display for StateSpaceEnumerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimit(message) => {
                write!(formatter, "invalid enumeration limit: {message}")
            }
            Self::StateLimit {
                limit,
                states,
                transitions,
            } => write!(
                formatter,
                "max_states bound {limit} reached: states={states}, transitions={transitions}; no complete state space or h* values were returned"
            ),
            Self::TransitionLimit {
                limit,
                states,
                transitions,
            } => write!(
                formatter,
                "max_transitions bound {limit} reached: states={states}, transitions={transitions}; no complete state space or h* values were returned"
            ),
            Self::TimeLimit {
                limit,
                states,
                transitions,
            } => write!(
                formatter,
                "time limit {:.6}s reached: states={states}, transitions={transitions}; no complete state space or h* values were returned",
                limit.as_secs_f64()
            ),
            Self::Enumeration(message) => {
                write!(formatter, "state-space enumeration failed: {message}")
            }
            Self::GoalDistances(message) => {
                write!(formatter, "exact h* computation failed: {message}")
            }
        }
    }
}

impl std::error::Error for StateSpaceEnumerationError {}

#[derive(Debug)]
pub struct OwnedStateSpace {
    pub num_propositional_variables: usize,
    pub num_numeric_variables: usize,
    /// Row-major `num_states x num_propositional_variables` values.
    pub propositional_values: Vec<u32>,
    /// Row-major `num_states x num_numeric_variables` values.
    pub numeric_values: Vec<f64>,
    /// CSR offsets into the four transition arrays, one row per source state.
    pub transition_offsets: Vec<u64>,
    pub transition_operator_ids: Vec<u32>,
    pub transition_successor_ids: Vec<u32>,
    pub transition_costs: Vec<f64>,
    pub goal_states: Vec<bool>,
    pub h_star: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StateSpaceSummary {
    pub state_count: usize,
    pub transition_count: usize,
    pub goal_state_count: usize,
    pub dead_end_count: usize,
    /// Finite h* values only; dead ends are reported separately.
    pub h_star_histogram: Vec<(f64, usize)>,
    /// Maximum finite distance to a goal. `None` when no reachable goal exists.
    pub diameter: Option<f64>,
}

impl OwnedStateSpace {
    pub fn num_states(&self) -> usize {
        self.goal_states.len()
    }

    pub fn num_transitions(&self) -> usize {
        self.transition_costs.len()
    }

    pub fn summary(&self) -> StateSpaceSummary {
        let mut histogram = std::collections::BTreeMap::<NotNan<f64>, usize>::new();
        let mut dead_end_count = 0usize;
        let mut diameter: Option<f64> = None;
        for &distance in &self.h_star {
            if distance.is_infinite() && distance.is_sign_positive() {
                dead_end_count += 1;
                continue;
            }
            let distance = NotNan::new(distance)
                .expect("validated transition costs cannot produce NaN goal distances");
            *histogram.entry(distance).or_default() += 1;
            diameter =
                Some(diameter.map_or(distance.into_inner(), |old| old.max(distance.into_inner())));
        }
        StateSpaceSummary {
            state_count: self.num_states(),
            transition_count: self.num_transitions(),
            goal_state_count: self.goal_states.iter().filter(|&&is_goal| is_goal).count(),
            dead_end_count,
            h_star_histogram: histogram
                .into_iter()
                .map(|(distance, count)| (distance.into_inner(), count))
                .collect(),
            diameter,
        }
    }
}

struct ConcreteBackwardGraph<'a> {
    num_states: usize,
    goal_state_ids: &'a [usize],
    backward_offsets: &'a [usize],
    backward_transition_ids: &'a [usize],
    transition_sources: &'a [u32],
    transition_targets: &'a [u32],
    transition_operator_ids: &'a [u32],
}

impl BackwardDijkstraGraph for ConcreteBackwardGraph<'_> {
    fn num_states(&self) -> usize {
        self.num_states
    }

    fn goal_state_ids(&self) -> &[usize] {
        self.goal_state_ids
    }

    fn incoming_transition_ids(&self, target_id: usize) -> Option<&[usize]> {
        let start = *self.backward_offsets.get(target_id)?;
        let end = *self.backward_offsets.get(target_id + 1)?;
        self.backward_transition_ids.get(start..end)
    }

    fn transition_endpoints(&self, transition_id: usize) -> Option<(usize, usize)> {
        Some((
            *self.transition_sources.get(transition_id)? as usize,
            *self.transition_targets.get(transition_id)? as usize,
        ))
    }

    fn transition_operator_id(&self, transition_id: usize) -> Option<usize> {
        self.transition_operator_ids
            .get(transition_id)
            .map(|&operator_id| operator_id as usize)
    }

    fn num_transitions(&self) -> usize {
        self.transition_targets.len()
    }
}

fn limit_deadline(start: Instant, limit: Duration) -> Result<Instant, StateSpaceEnumerationError> {
    start.checked_add(limit).ok_or_else(|| {
        StateSpaceEnumerationError::InvalidLimit(
            "max_time is too large for the platform clock".to_string(),
        )
    })
}

fn check_time_limit(
    deadline: Instant,
    limit: Duration,
    states: usize,
    transitions: usize,
) -> Result<(), StateSpaceEnumerationError> {
    if Instant::now() >= deadline {
        return Err(StateSpaceEnumerationError::TimeLimit {
            limit,
            states,
            transitions,
        });
    }
    Ok(())
}

pub fn enumerate_state_space(
    task: TaskRef<'_>,
    limits: EnumerationLimits,
) -> Result<OwnedStateSpace, StateSpaceEnumerationError> {
    let limits = limits.validate()?;
    let start = Instant::now();
    let deadline = limit_deadline(start, limits.max_time)?;
    check_time_limit(deadline, limits.max_time, 0, 0)?;

    let successor_generator = SuccessorTree::new(&*task);
    let operators = task.get_operators();
    let num_propositional_variables = task.variables().len();
    let num_numeric_variables = task.numeric_variables().len();
    let mut registry = StateRegistry::for_task(task.clone());
    let _initial = registry.get_initial_state();
    if registry.num_registered_states() > limits.max_states {
        return Err(StateSpaceEnumerationError::StateLimit {
            limit: limits.max_states,
            states: registry.num_registered_states(),
            transitions: 0,
        });
    }

    let mut propositional_values = Vec::new();
    let mut numeric_values = Vec::new();
    let mut transition_offsets = vec![0u64];
    let mut transition_sources = Vec::new();
    let mut transition_operator_ids = Vec::new();
    let mut transition_successor_ids = Vec::new();
    let mut transition_costs = Vec::new();
    let mut goal_states = Vec::new();
    let mut goal_state_ids = Vec::new();

    let mut propositional_scratch = Vec::with_capacity(num_propositional_variables);
    let mut numeric_scratch = Vec::with_capacity(num_numeric_variables);
    let mut applicable = Vec::new();
    let mut expansion_context = ExpansionContext::default();
    let mut successor_numeric = Vec::with_capacity(num_numeric_variables);
    let mut successor_cost = Vec::new();
    let mut state_id = 0usize;

    while state_id < registry.num_registered_states() {
        check_time_limit(
            deadline,
            limits.max_time,
            registry.num_registered_states(),
            transition_costs.len(),
        )?;
        let state = registry.lookup_state(state_id).map_err(|error| {
            StateSpaceEnumerationError::Enumeration(format!(
                "registered state {state_id} disappeared: {error:?}"
            ))
        })?;
        registry
            .fill_state_and_numeric_vars(&state, &mut propositional_scratch, &mut numeric_scratch)
            .map_err(|error| {
                StateSpaceEnumerationError::Enumeration(format!(
                    "could not decode state {state_id}: {error:?}"
                ))
            })?;
        for &value in &propositional_scratch {
            propositional_values.push(u32::try_from(value).map_err(|_| {
                StateSpaceEnumerationError::Enumeration(format!(
                    "state {state_id} has propositional value {value} above u32::MAX"
                ))
            })?);
        }
        numeric_values.extend_from_slice(&numeric_scratch);

        let is_goal = (0..task.get_num_goals())
            .all(|goal_id| task.get_goal_fact(goal_id).is_hold(registry.view(&state)));
        goal_states.push(is_goal);
        if is_goal {
            goal_state_ids.push(state_id);
        }

        applicable.clear();
        successor_generator.get_applicable_operators(&propositional_scratch, &mut applicable);
        registry
            .build_expansion_context(&state, &mut expansion_context)
            .map_err(|error| {
                StateSpaceEnumerationError::Enumeration(format!(
                    "could not prepare expansion of state {state_id}: {error:?}"
                ))
            })?;
        for &operator_id in &applicable {
            if transition_costs.len() == limits.max_transitions {
                return Err(StateSpaceEnumerationError::TransitionLimit {
                    limit: limits.max_transitions,
                    states: registry.num_registered_states(),
                    transitions: transition_costs.len(),
                });
            }
            if transition_costs.len().is_multiple_of(1024) {
                check_time_limit(
                    deadline,
                    limits.max_time,
                    registry.num_registered_states(),
                    transition_costs.len(),
                )?;
            }
            let operator = operators.get(operator_id as usize).unwrap_or_else(|| {
                panic!("successor generator returned invalid operator id {operator_id}")
            });
            let (successor, cost) = registry
                .apply_operator_in_context(
                    &state,
                    operator,
                    &expansion_context,
                    &mut successor_numeric,
                    &mut successor_cost,
                )
                .map_err(|error| {
                    StateSpaceEnumerationError::Enumeration(format!(
                        "operator {operator_id} ({}) failed in state {state_id}: {error:?}",
                        operator.name()
                    ))
                })?;
            if !cost.is_finite() || cost < 0.0 {
                return Err(StateSpaceEnumerationError::Enumeration(format!(
                    "transition from state {state_id} via operator {operator_id} has invalid cost {cost}"
                )));
            }
            transition_sources.push(u32::try_from(state_id).map_err(|_| {
                StateSpaceEnumerationError::Enumeration(format!(
                    "source state id {state_id} exceeds u32::MAX"
                ))
            })?);
            transition_operator_ids.push(operator_id);
            transition_successor_ids.push(u32::try_from(successor.get_id()).map_err(|_| {
                StateSpaceEnumerationError::Enumeration(format!(
                    "successor state id {} exceeds u32::MAX",
                    successor.get_id()
                ))
            })?);
            transition_costs.push(cost);
            if registry.num_registered_states() > limits.max_states {
                return Err(StateSpaceEnumerationError::StateLimit {
                    limit: limits.max_states,
                    states: registry.num_registered_states(),
                    transitions: transition_costs.len(),
                });
            }
        }
        transition_offsets.push(u64::try_from(transition_costs.len()).map_err(|_| {
            StateSpaceEnumerationError::Enumeration("transition count exceeds u64::MAX".to_string())
        })?);
        state_id += 1;
    }

    let num_states = registry.num_registered_states();
    assert_eq!(
        state_id, num_states,
        "every registered state must be expanded"
    );
    assert_eq!(
        transition_offsets.len(),
        num_states + 1,
        "CSR offsets must delimit every source state"
    );
    check_time_limit(
        deadline,
        limits.max_time,
        num_states,
        transition_costs.len(),
    )?;

    let mut incoming_counts = vec![0usize; num_states];
    for &target in &transition_successor_ids {
        let target = target as usize;
        let count = incoming_counts.get_mut(target).ok_or_else(|| {
            StateSpaceEnumerationError::Enumeration(format!(
                "transition target {target} is out of bounds for {num_states} states"
            ))
        })?;
        *count = count.checked_add(1).ok_or_else(|| {
            StateSpaceEnumerationError::Enumeration("incoming edge count overflow".to_string())
        })?;
    }
    let mut backward_offsets = Vec::with_capacity(num_states + 1);
    backward_offsets.push(0usize);
    for count in incoming_counts {
        let next = backward_offsets
            .last()
            .copied()
            .expect("the zero offset was inserted")
            .checked_add(count)
            .ok_or_else(|| {
                StateSpaceEnumerationError::Enumeration("backward CSR offset overflow".to_string())
            })?;
        backward_offsets.push(next);
    }
    let mut next_incoming = backward_offsets[..num_states].to_vec();
    let mut backward_transition_ids = vec![0usize; transition_costs.len()];
    for (transition_id, &target) in transition_successor_ids.iter().enumerate() {
        let target = target as usize;
        let slot = next_incoming[target];
        backward_transition_ids[slot] = transition_id;
        next_incoming[target] = next_incoming[target].checked_add(1).ok_or_else(|| {
            StateSpaceEnumerationError::Enumeration("backward CSR cursor overflow".to_string())
        })?;
    }

    let backward_graph = ConcreteBackwardGraph {
        num_states,
        goal_state_ids: &goal_state_ids,
        backward_offsets: &backward_offsets,
        backward_transition_ids: &backward_transition_ids,
        transition_sources: &transition_sources,
        transition_targets: &transition_successor_ids,
        transition_operator_ids: &transition_operator_ids,
    };
    let timed_out = Cell::new(false);
    let h_star = backward_goal_distances(
        &backward_graph,
        &transition_costs,
        || -> AnyResult<()> {
            if Instant::now() >= deadline {
                timed_out.set(true);
                bail!("state-space enumeration time limit reached");
            }
            Ok(())
        },
        None,
    )
    .map_err(|error| {
        if timed_out.get() {
            StateSpaceEnumerationError::TimeLimit {
                limit: limits.max_time,
                states: num_states,
                transitions: transition_costs.len(),
            }
        } else {
            StateSpaceEnumerationError::GoalDistances(error.to_string())
        }
    })?;

    Ok(OwnedStateSpace {
        num_propositional_variables,
        num_numeric_variables,
        propositional_values,
        numeric_values,
        transition_offsets,
        transition_operator_ids,
        transition_successor_ids,
        transition_costs,
        goal_states,
        h_star,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use planforge_sas::numeric_task::NumericRootTask;
    use std::sync::Arc;

    fn pddl_task(directory: &str, problem: &str) -> TaskRef<'static> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/assets")
            .join(directory);
        let task: NumericRootTask = planforge_translate::translate_to_task(
            root.join("domain.pddl").to_str().unwrap(),
            root.join(problem).to_str().unwrap(),
        )
        .unwrap();
        Arc::new(task)
    }

    fn generous_limits() -> EnumerationLimits {
        EnumerationLimits {
            max_states: 100_000,
            max_transitions: 1_000_000,
            max_time: Duration::from_secs(30),
        }
    }

    #[test]
    fn blocks_two_has_exact_goal_distances() {
        let graph = enumerate_state_space(
            pddl_task(
                "strips-pddl-files/blocks-minimal",
                "probBLOCKS-2-reverse.pddl",
            ),
            generous_limits(),
        )
        .unwrap();
        assert_eq!(graph.num_states(), 5);
        assert_eq!(graph.h_star[0], 4.0);
        for (&is_goal, &distance) in graph.goal_states.iter().zip(&graph.h_star) {
            if is_goal {
                assert_eq!(distance, 0.0);
            }
        }
    }

    #[test]
    fn unreachable_goal_states_are_exact_dead_ends() {
        let graph = enumerate_state_space(
            pddl_task("adl/unreachable-goal", "problem.pddl"),
            generous_limits(),
        )
        .unwrap();
        assert!(graph.h_star.iter().all(|distance| distance.is_infinite()));
        assert_eq!(graph.summary().dead_end_count, graph.num_states());
    }

    #[test]
    fn a_state_bound_returns_no_success_shaped_partial_graph() {
        let error = enumerate_state_space(
            pddl_task(
                "strips-pddl-files/blocks-minimal",
                "probBLOCKS-2-reverse.pddl",
            ),
            EnumerationLimits {
                max_states: 1,
                ..generous_limits()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            StateSpaceEnumerationError::StateLimit { limit: 1, .. }
        ));
    }

    #[test]
    fn a_transition_bound_returns_no_success_shaped_partial_graph() {
        let error = enumerate_state_space(
            pddl_task(
                "strips-pddl-files/blocks-minimal",
                "probBLOCKS-2-reverse.pddl",
            ),
            EnumerationLimits {
                max_transitions: 1,
                ..generous_limits()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            StateSpaceEnumerationError::TransitionLimit { limit: 1, .. }
        ));
    }

    #[test]
    fn a_time_bound_returns_no_success_shaped_partial_graph() {
        let error = enumerate_state_space(
            pddl_task(
                "strips-pddl-files/blocks-minimal",
                "probBLOCKS-2-reverse.pddl",
            ),
            EnumerationLimits {
                max_time: Duration::ZERO,
                ..generous_limits()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            StateSpaceEnumerationError::TimeLimit { limit, .. } if limit == Duration::ZERO
        ));
    }
}
