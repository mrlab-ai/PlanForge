//! The search space: the registered states of a task, their successors, and
//! the search node of every reached state, from which a plan is traced.

#[cfg(test)]
mod tests;

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use planforge_sas::numeric_task::{OperatorIndex, Task};
use planforge_sas::state_registry::{ConcreteState, StateID, StateRegistry};
use tracing::info;

use super::{Plan, SearchOutcome};
use crate::evaluation::EvaluationState;
use crate::successor_generator::RegistryExpander;

/// One transition out of an expanded state: the registered successor, the
/// operator leading to it and its cost.
#[derive(Debug, Clone)]
pub struct Successor {
    pub state: ConcreteState,
    pub operator: OperatorIndex,
    /// The operator's cost, or the metric's increase when the task has a
    /// metric.
    pub cost: f64,
}

/// The search node of a reached state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchNode {
    /// The predecessor on the cheapest path found and the operator leading
    /// from it; `None` for the initial state.
    pub parent: Option<(StateID, OperatorIndex)>,
    /// The cost of the cheapest path found.
    pub g: f64,
    pub is_dead_end: bool,
    pub is_closed: bool,
    pub is_goal: bool,
}

/// What [`SearchSpace::reach`] did with a path to a state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reached {
    /// The state was reached for the first time.
    New,
    /// The path is cheaper than every earlier one and now its node's path;
    /// `reopened` if the node had been closed.
    Cheaper { reopened: bool },
    /// The state was reached at most as expensively before, or is a dead
    /// end; its node is unchanged.
    Duplicate,
}

/// The states a search has registered, how to generate successors, and the
/// search node of each reached state, indexed by [`StateID`].
pub struct SearchSpace {
    task: Arc<Task>,
    registry: StateRegistry,
    initial_state: ConcreteState,
    expander: RegistryExpander,
    /// Per-operator costs, for a task whose costs do not depend on the state.
    operator_costs: Vec<f64>,
    /// Whether a metric prices each transition instead.
    use_metric: bool,
    nodes: SearchNodes,
}

impl SearchSpace {
    /// The search space of `task`, holding its registered initial state.
    pub fn new(task: Arc<Task>) -> Self {
        let mut registry = StateRegistry::for_task(task.clone());
        // Registering the initial state fixes the numeric constants the
        // metric's operator costs read.
        let initial_state = registry.get_initial_state();
        info!(
            "State representation: bins={}",
            registry.global_state_packer().num_bins()
        );
        Self {
            expander: RegistryExpander::new(&task),
            operator_costs: task.effective_operator_costs().to_vec(),
            use_metric: task.metric().use_metric(),
            task,
            registry,
            initial_state,
            nodes: SearchNodes::default(),
        }
    }

    pub fn task(&self) -> &Arc<Task> {
        &self.task
    }

    pub fn registry(&self) -> &StateRegistry {
        &self.registry
    }

    pub fn initial_state(&self) -> ConcreteState {
        self.initial_state.clone()
    }

    /// The registered state `state_id`.
    #[inline]
    pub fn state(&self, state_id: StateID) -> ConcreteState {
        self.registry
            .lookup_state(state_id)
            .unwrap_or_else(|error| panic!("state {state_id} is not registered: {error:?}"))
    }

    /// Whether `state` satisfies the goal.
    pub fn is_goal_state(&self, state: &ConcreteState) -> bool {
        self.registry.view(state).holds_all(self.task.goals())
    }

    /// The reached state `state` as a heuristic evaluates it.
    #[inline]
    pub fn evaluation_state<'s>(&'s self, state: &'s ConcreteState) -> EvaluationState<'s, 's> {
        let mut eval_state = EvaluationState::new(state, &self.task, &self.registry);
        eval_state.set_is_goal(self.is_goal(state.get_id()));
        eval_state
    }

    pub fn num_registered_states(&self) -> usize {
        self.registry.num_registered_states()
    }

    /// Replaces `successors` with the successors of `state` in the successor
    /// generator's order, registering the new ones. A state reached by
    /// several operators appears once per operator.
    pub fn generate_successors(
        &mut self,
        state: &ConcreteState,
        successors: &mut Vec<Successor>,
    ) -> Result<()> {
        successors.clear();
        self.for_each_successor(state, |_, _, successor| successors.push(successor))
    }

    /// Generates the successors of the reached state `state` and detects
    /// duplicates in one pass: [`Self::reach`]es each successor on the path
    /// through `state` and replaces `successors` with those whose path it
    /// recorded, in generation order; `reached` learns the outcome for every
    /// successor. A state generated twice is kept twice if the second path
    /// is cheaper.
    pub fn reach_successors(
        &mut self,
        state: &ConcreteState,
        successors: &mut Vec<Successor>,
        mut reached: impl FnMut(Reached),
    ) -> Result<()> {
        successors.clear();
        let parent = state.get_id();
        let parent_g = self.g(parent);
        let task = self.task.clone();
        self.for_each_successor(state, |nodes, registry, successor| {
            let path = Some((parent, successor.operator));
            let outcome = nodes.reach(&successor.state, path, parent_g + successor.cost, || {
                registry.view(&successor.state).holds_all(task.goals())
            });
            reached(outcome);
            if outcome != Reached::Duplicate {
                successors.push(successor);
            }
        })
    }

    /// Calls `visit` with the search nodes, the registry and each successor
    /// of `state`, in the successor generator's order.
    #[inline]
    fn for_each_successor(
        &mut self,
        state: &ConcreteState,
        mut visit: impl FnMut(&mut SearchNodes, &StateRegistry, Successor),
    ) -> Result<()> {
        let state_id = state.get_id();
        let mut expansion = self
            .expander
            .expand(&self.registry, state)
            .map_err(|error| anyhow!("failed to expand state {state_id}: {error:?}"))?;
        for &operator_id in expansion.applicable_operators() {
            let operator = &self.task.operators()[operator_id.index()];
            let (successor, metric_cost) = expansion
                .successor(&mut self.registry, operator)
                .map_err(|error| anyhow!("{error:?}"))
                .with_context(|| {
                    format!(
                        "failed to apply operator {operator_id} ({}) to state {state_id}",
                        operator.name()
                    )
                })?;
            let cost = if self.use_metric {
                metric_cost
            } else {
                self.operator_costs[operator_id.index()]
            };
            visit(
                &mut self.nodes,
                &self.registry,
                Successor {
                    state: successor,
                    operator: operator_id,
                    cost,
                },
            );
        }
        Ok(())
    }

    /// The node of `state_id`, if the state was reached.
    #[inline]
    pub fn node(&self, state_id: StateID) -> Option<SearchNode> {
        self.nodes.node(state_id)
    }

    /// Duplicate detection: records the path of cost `g` to `state` through
    /// `parent` unless the state was reached at most as expensively before or
    /// is a dead end. A recorded path opens the node.
    pub fn reach(
        &mut self,
        state: &ConcreteState,
        parent: Option<(StateID, OperatorIndex)>,
        g: f64,
    ) -> Reached {
        let (registry, goals) = (&self.registry, self.task.goals());
        self.nodes
            .reach(state, parent, g, || registry.view(state).holds_all(goals))
    }

    /// The cost of the cheapest path found to `state_id`.
    #[inline]
    pub fn g(&self, state_id: StateID) -> f64 {
        self.nodes.status(state_id);
        self.nodes.g_values[state_id]
    }

    #[inline]
    pub fn is_goal(&self, state_id: StateID) -> bool {
        self.nodes.status(state_id) & NODE_GOAL != 0
    }

    #[inline]
    pub fn close(&mut self, state_id: StateID) {
        *self.nodes.status_mut(state_id) |= NODE_CLOSED;
    }

    /// Marks `state_id` as a dead end, which [`Self::reach`] never reopens.
    #[inline]
    pub fn mark_dead_end(&mut self, state_id: StateID) {
        *self.nodes.status_mut(state_id) |= NODE_DEAD_END;
    }

    /// The operators on the cheapest path found from the initial state to
    /// `state_id`.
    pub fn trace_path(&self, state_id: StateID) -> Plan {
        let mut plan = Vec::new();
        let mut current = state_id;
        while let Some((parent, operator)) = self.node(current).and_then(|node| node.parent) {
            plan.push(self.task.operators()[operator.index()].clone());
            current = parent;
        }
        assert_eq!(
            current,
            self.initial_state.get_id(),
            "the path to state {state_id} does not start in the initial state"
        );
        plan.reverse();
        plan
    }

    /// The plan to the goal state `goal` and its cost.
    pub fn solution(&self, goal: StateID) -> SearchOutcome {
        assert!(self.is_goal(goal), "state {goal} is not a goal state");
        SearchOutcome::Solved {
            plan: self.trace_path(goal),
            cost: self.g(goal),
        }
    }
}

const NO_PARENT: u32 = u32::MAX;
const NODE_PRESENT: u8 = 1 << 0;
const NODE_DEAD_END: u8 = 1 << 1;
const NODE_CLOSED: u8 = 1 << 2;
const NODE_GOAL: u8 = 1 << 3;

/// The search nodes as a structure of arrays: 17 bytes per state instead of
/// 32 for a padded [`SearchNode`].
#[derive(Default)]
struct SearchNodes {
    parent_states: Vec<u32>,
    parent_operators: Vec<OperatorIndex>,
    g_values: Vec<f64>,
    status: Vec<u8>,
}

impl SearchNodes {
    #[inline]
    fn node(&self, state_id: StateID) -> Option<SearchNode> {
        let &status = self.status.get(state_id)?;
        if status & NODE_PRESENT == 0 {
            return None;
        }
        let parent_state = self.parent_states[state_id];
        Some(SearchNode {
            parent: (parent_state != NO_PARENT)
                .then(|| (parent_state as StateID, self.parent_operators[state_id])),
            g: self.g_values[state_id],
            is_dead_end: status & NODE_DEAD_END != 0,
            is_closed: status & NODE_CLOSED != 0,
            is_goal: status & NODE_GOAL != 0,
        })
    }

    /// [`SearchSpace::reach`]; `is_goal` tests a new state.
    #[inline]
    fn reach(
        &mut self,
        state: &ConcreteState,
        parent: Option<(StateID, OperatorIndex)>,
        g: f64,
        is_goal: impl FnOnce() -> bool,
    ) -> Reached {
        let state_id = state.get_id();
        let (reached, is_goal) = match self.node(state_id) {
            Some(node) if node.is_dead_end || node.g <= g => return Reached::Duplicate,
            Some(node) => (
                Reached::Cheaper {
                    reopened: node.is_closed,
                },
                node.is_goal,
            ),
            None => (Reached::New, is_goal()),
        };
        if state_id >= self.status.len() {
            self.grow(state_id);
        }
        let (parent_state, parent_operator) = match parent {
            Some((state, operator)) => {
                let state = u32::try_from(state).expect("parent state id exceeds u32");
                assert_ne!(
                    state, NO_PARENT,
                    "parent state id is the no-parent sentinel"
                );
                (state, operator)
            }
            None => (NO_PARENT, OperatorIndex::new(u32::MAX)),
        };
        self.parent_states[state_id] = parent_state;
        self.parent_operators[state_id] = parent_operator;
        self.g_values[state_id] = g;
        self.status[state_id] = NODE_PRESENT | if is_goal { NODE_GOAL } else { 0 };
        reached
    }

    /// Makes room for the node of `state_id` and as many again, so that
    /// growing is rare although states are registered one by one.
    #[cold]
    fn grow(&mut self, state_id: StateID) {
        let len = (state_id + 1).max(2 * self.status.len());
        self.parent_states.resize(len, NO_PARENT);
        self.parent_operators
            .resize(len, OperatorIndex::new(u32::MAX));
        self.g_values.resize(len, 0.0);
        self.status.resize(len, 0);
    }

    #[inline]
    fn status(&self, state_id: StateID) -> u8 {
        let status = self.status.get(state_id).copied().unwrap_or(0);
        assert!(
            status & NODE_PRESENT != 0,
            "state {state_id} has no search node"
        );
        status
    }

    #[inline]
    fn status_mut(&mut self, state_id: StateID) -> &mut u8 {
        self.status(state_id);
        &mut self.status[state_id]
    }
}
