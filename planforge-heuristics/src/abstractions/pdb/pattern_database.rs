//! A pattern database: the projection of a task to a pattern, explored from
//! the initial state, and the goal distances of its abstract states.

#[cfg(test)]
mod tests;

use planforge_sas::explicit_state::StateValues;
use planforge_search::graph::{BackwardGraph, backward_goal_distances, no_limit};
use planforge_search::heuristic_features::HeuristicBuildError;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;
use std::sync::Arc;

use ordered_float::NotNan;
use planforge_sas::numeric_task::{
    ExplicitFact, ExplicitValueIndex, NumericValue, OperatorIndex, Task, VariableIndex,
    VariableKind,
};
use planforge_sas::state_registry::{ConcreteState, StateRegistry};

use crate::landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig;
use crate::landmarks::numeric_lm_cut_landmarks::LandmarkCutLandmarks;
use planforge_search::config::option_enum;
use planforge_search::successor_generator::{StandardGroundedTree, StateExpander};

use planforge_search::task_transformations::projection::ProjectedTask;

/// The heuristic a pattern database estimates abstract goal distances with
/// where its exploration gives none.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PdbInternalHeuristic {
    Zero,
    #[default]
    Blind,
    Lmcut,
}

option_enum!("PDB internal heuristic", PdbInternalHeuristic {
    Zero => "zero",
    Blind => "blind",
    Lmcut => "lmcut",
});

/// The inner heuristics of a pattern database (Fritzsche et al. 2026): the
/// exploration orders its open list by the first, a frontier state takes
/// the second as its distance, and a state the exploration did not reach
/// the third.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PdbHeuristicConfig {
    pub exploration_heuristic: PdbInternalHeuristic,
    pub frontier_heuristic: PdbInternalHeuristic,
    pub failed_lookup_heuristic: PdbInternalHeuristic,
}

impl PdbHeuristicConfig {
    fn uses_lmcut(self) -> bool {
        [
            self.exploration_heuristic,
            self.frontier_heuristic,
            self.failed_lookup_heuristic,
        ]
        .contains(&PdbInternalHeuristic::Lmcut)
    }
}

impl fmt::Display for PdbHeuristicConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "exploration_heuristic={}, frontier_heuristic={}, failed_lookup_heuristic={}",
            self.exploration_heuristic, self.frontier_heuristic, self.failed_lookup_heuristic,
        )
    }
}

/// How a pattern database explores its projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PdbBuildConfig {
    /// The newly reached abstract states after which the exploration of a
    /// projection with numeric variables stops. A purely propositional
    /// projection is finite and explored in full.
    pub max_states: usize,
    pub heuristics: PdbHeuristicConfig,
    /// Keep the transitions of the abstract state space, which only
    /// saturated cost partitioning reads.
    pub retain_transitions: bool,
}

/// A pattern database over a projection of a numeric task (Fritzsche et al.
/// 2026).
///
/// The abstract state space is explored forward from the abstract initial
/// state, best-first under the exploration heuristic. A purely propositional
/// projection is explored in full; one with numeric variables until
/// `max_states` new states were reached. A state the exploration generated
/// but did not expand lies on the frontier, and takes the frontier heuristic,
/// at least the cheapest operator cost, as its distance. The goal distances
/// then follow by backward Dijkstra from the goal and frontier states.
#[derive(Debug)]
pub struct PatternDatabase {
    task: Arc<ProjectedTask>,
    states: AbstractStates,
    /// The goal distance of every abstract state, by id; infinite for a
    /// state from which neither a goal nor the frontier is reachable.
    distances: Vec<f64>,
    /// Every abstract state reachable from the abstract initial state was
    /// expanded, so a state the exploration did not reach is unreachable.
    exhausted: bool,
    min_operator_cost: f64,
    relevant_operator_ids: Vec<OperatorIndex>,
    failed_lookup: FailedLookupHeuristic,
    transitions: Option<AbstractTransitions>,
    scratch: RefCell<LookupScratch>,
}

impl PatternDatabase {
    /// The pattern database over `task` with blind inner heuristics, keeping
    /// its transitions.
    pub fn new(task: ProjectedTask, max_states: usize) -> Result<Self, HeuristicBuildError> {
        Self::build(
            task,
            PdbBuildConfig {
                max_states,
                heuristics: PdbHeuristicConfig::default(),
                retain_transitions: true,
            },
        )
    }

    pub fn build(task: ProjectedTask, config: PdbBuildConfig) -> Result<Self, HeuristicBuildError> {
        let task = Arc::new(task);
        let task_ref: Arc<Task> = task.task().clone();
        let mut registry = StateRegistry::for_task(task_ref.clone());
        let mut inner = InnerHeuristics {
            lmcut: config
                .heuristics
                .uses_lmcut()
                .then(|| ProjectedLmcut::new(task_ref)),
            cache: Vec::new(),
        };
        let max_new_states = if task.has_regular_numeric_variables() {
            config.max_states
        } else {
            usize::MAX
        };
        let exploration = explore(
            task.task(),
            &mut registry,
            max_new_states,
            config.heuristics.exploration_heuristic,
            &mut inner,
        )?;
        let num_states = registry.num_registered_states();
        let transitions = AbstractTransitions::new(num_states, exploration);
        let exhausted = transitions.frontier.is_empty();

        let min_operator_cost = task.min_operator_cost();
        let mut seeds: Vec<(usize, f64)> = transitions
            .goal_states
            .iter()
            .map(|&state_id| (state_id, 0.0))
            .collect();
        for &state_id in &transitions.frontier {
            let state = ConcreteState::new(state_id);
            if registry.view(&state).holds_all(task.task().goals()) {
                seeds.push((state_id, 0.0));
            } else if let Some(h) =
                inner.estimate(config.heuristics.frontier_heuristic, state_id, &registry)
            {
                seeds.push((state_id, h.max(min_operator_cost)));
            }
        }
        let distances = transitions
            .backward_dijkstra(&seeds, |operator_id| task.task().operator_cost(operator_id));

        let mut relevant_operator_ids: Vec<OperatorIndex> = if exhausted {
            transitions
                .predecessors
                .iter()
                .map(|&(_, operator_id)| task.base_operator_id(operator_id))
                .collect()
        } else {
            task.base_operator_ids().to_vec()
        };
        relevant_operator_ids.sort_unstable();
        relevant_operator_ids.dedup();

        let dense = exhausted
            .then(|| DenseStates::new(&task, &registry))
            .flatten();
        let states = match dense {
            Some(dense) => AbstractStates::Dense(dense),
            None => AbstractStates::Registry(Box::new(registry)),
        };
        let failed_lookup = match config.heuristics.failed_lookup_heuristic {
            PdbInternalHeuristic::Zero => FailedLookupHeuristic::Zero,
            PdbInternalHeuristic::Blind => FailedLookupHeuristic::Blind,
            PdbInternalHeuristic::Lmcut => FailedLookupHeuristic::Lmcut(Box::new(RefCell::new(
                inner
                    .lmcut
                    .expect("LM-cut was built for the failed-lookup heuristic"),
            ))),
        };
        Ok(Self {
            task,
            states,
            distances,
            exhausted,
            min_operator_cost,
            relevant_operator_ids,
            failed_lookup,
            transitions: config.retain_transitions.then_some(transitions),
            scratch: RefCell::default(),
        })
    }

    pub fn num_states(&self) -> usize {
        self.distances.len()
    }

    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    pub fn min_operator_cost(&self) -> f64 {
        self.min_operator_cost
    }

    /// The base operators that can label a transition of the abstract state
    /// space: those on a transition when it was explored in full, otherwise
    /// every operator the projection keeps.
    pub fn relevant_operator_ids(&self) -> &[OperatorIndex] {
        &self.relevant_operator_ids
    }

    /// The goal distance this pattern database estimates for the state of the
    /// base task with values `propositional` and `numeric`.
    ///
    /// A state the exploration did not reach is a dead end when the
    /// exploration was exhaustive, has distance zero when it is an abstract
    /// goal, and otherwise takes the failed-lookup heuristic, at least the
    /// cheapest operator cost.
    pub fn value(&self, state: StateValues<'_>) -> f64 {
        let registry = match &self.states {
            AbstractStates::Dense(dense) => {
                return dense
                    .state_id(state)
                    .map_or(f64::INFINITY, |state_id| self.distances[state_id]);
            }
            AbstractStates::Registry(registry) => registry,
        };
        let mut scratch = self.scratch.borrow_mut();
        let scratch = &mut *scratch;
        if let Some(state_id) = self.find_state_id(registry, state, scratch) {
            return self.distances[state_id];
        }
        if self.exhausted {
            return f64::INFINITY;
        }
        if is_goal(self.task.task().goals(), &scratch.propositional) {
            return 0.0;
        }
        match &self.failed_lookup {
            FailedLookupHeuristic::Zero => 0.0,
            FailedLookupHeuristic::Blind => self.min_operator_cost,
            FailedLookupHeuristic::Lmcut(lmcut) => lmcut
                .borrow_mut()
                .estimate(&scratch.propositional, &scratch.numeric)
                .map_or(f64::INFINITY, |h| h.max(self.min_operator_cost)),
        }
    }

    /// The id of the abstract state of the state of the base task with values
    /// `propositional` and `numeric`, if the exploration reached it.
    pub fn abstract_state_id(&self, state: StateValues<'_>) -> Option<usize> {
        match &self.states {
            AbstractStates::Dense(dense) => dense.state_id(state),
            AbstractStates::Registry(registry) => {
                self.find_state_id(registry, state, &mut self.scratch.borrow_mut())
            }
        }
    }

    /// Projects the state into `scratch` and looks it up in `registry`.
    fn find_state_id(
        &self,
        registry: &StateRegistry,
        state: StateValues<'_>,
        scratch: &mut LookupScratch,
    ) -> Option<usize> {
        self.task.project_state_values_into(
            state,
            &mut scratch.propositional,
            &mut scratch.numeric,
        );
        registry.find_state_id(
            &scratch.propositional,
            &scratch.numeric,
            &mut scratch.packed,
        )
    }

    /// The goal distances of the abstract states under `operator_costs`, one
    /// cost per base operator, and the saturated costs of those distances.
    pub fn build_cost_partitioned_distance_table(
        &self,
        operator_costs: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let distances = self.goal_distances_under(operator_costs);
        let saturated_costs = self.saturated_costs(&distances, operator_costs.len());
        (distances, saturated_costs)
    }

    /// The perimeter variant of [`Self::build_cost_partitioned_distance_table`]:
    /// the saturated costs only preserve the distances up to the goal
    /// distance of `cap_state_id`.
    ///
    /// The returned distances are recomputed from the saturated costs, so
    /// they stay admissible for the states beyond the perimeter.
    pub fn build_cost_partitioned_distance_table_capped(
        &self,
        operator_costs: &[f64],
        cap_state_id: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut distances = self.goal_distances_under(operator_costs);
        let cap = distances[cap_state_id];
        if cap.is_finite() {
            for distance in &mut distances {
                if *distance > cap {
                    *distance = f64::NEG_INFINITY;
                }
            }
        }
        let saturated_costs = self.saturated_costs(&distances, operator_costs.len());
        (self.goal_distances_under(&saturated_costs), saturated_costs)
    }

    fn transitions(&self) -> &AbstractTransitions {
        self.transitions
            .as_ref()
            .expect("cost partitioning reads a pattern database that kept its transitions")
    }

    /// The goal distances under `operator_costs`, with the frontier states
    /// at distance zero.
    fn goal_distances_under(&self, operator_costs: &[f64]) -> Vec<f64> {
        let transitions = self.transitions();
        let seeds: Vec<(usize, f64)> = transitions
            .goal_states
            .iter()
            .chain(&transitions.frontier)
            .map(|&state_id| (state_id, 0.0))
            .collect();
        transitions.backward_dijkstra(&seeds, |operator_id| {
            operator_costs[self.task.base_operator_id(operator_id).index()]
        })
    }

    /// For every base operator, the largest goal distance difference along a
    /// transition it labels between two states of finite distance.
    fn saturated_costs(&self, distances: &[f64], num_operators: usize) -> Vec<f64> {
        let transitions = self.transitions();
        let mut saturated_costs = vec![0.0_f64; num_operators];
        for target_id in 0..transitions.num_states() {
            let target_h = distances[target_id];
            if !target_h.is_finite() {
                continue;
            }
            for &(parent_id, operator_id) in transitions.predecessors_of(target_id) {
                let parent_h = distances[parent_id as usize];
                if !parent_h.is_finite() {
                    continue;
                }
                let slot = &mut saturated_costs[self.task.base_operator_id(operator_id).index()];
                *slot = slot.max(parent_h - target_h);
            }
        }
        saturated_costs
    }
}

/// The abstract states a pattern database explored, and how a projected
/// state finds its id among them.
#[derive(Debug)]
enum AbstractStates {
    Dense(DenseStates),
    /// The registry the exploration registered the states in.
    Registry(Box<StateRegistry>),
}

/// The states of an exhausted projection without numeric input variables,
/// addressed by the perfect hash of their propositional input values.
#[derive(Debug)]
struct DenseStates {
    variables: Vec<DenseVariable>,
    /// The abstract state at each hash, [`DenseStates::UNREACHED`] where the
    /// exploration reached none.
    ids: Vec<u32>,
}

#[derive(Debug)]
struct DenseVariable {
    projected: usize,
    original: usize,
    multiplier: usize,
}

impl DenseStates {
    const UNREACHED: u32 = u32::MAX;

    /// A table with this many slots per reached state costs about as much
    /// memory as a registry lookup does; beyond it the registry is used.
    const MAX_SLOTS_PER_STATE: usize = 8;

    /// Slots a table may always have: a few kilobytes, less than the
    /// registry itself.
    const MIN_SLOTS: usize = 1024;

    /// The dense table of the states in `registry`, or `None` when the
    /// projection has numeric input variables or too many value combinations
    /// for its reached states.
    fn new(task: &ProjectedTask, registry: &StateRegistry) -> Option<Self> {
        if task.has_regular_numeric_variables() {
            return None;
        }
        let num_states = registry.num_registered_states();
        let max_slots = Self::MAX_SLOTS_PER_STATE
            .checked_mul(num_states)?
            .max(Self::MIN_SLOTS);
        let mut variables = Vec::new();
        let mut num_slots = 1usize;
        for (projected, variable) in task.task().variables().iter().enumerate() {
            if task
                .task()
                .variable_kind(VariableIndex::from_usize(projected))
                != VariableKind::Regular
            {
                continue;
            }
            variables.push(DenseVariable {
                projected,
                original: task.original_variable(projected).index(),
                multiplier: num_slots,
            });
            num_slots = num_slots
                .checked_mul(variable.domain_size())
                .filter(|&slots| slots <= max_slots)?;
        }
        let mut ids = vec![Self::UNREACHED; num_slots];
        let mut values = Vec::new();
        for state_id in 0..num_states {
            registry
                .view(&ConcreteState::new(state_id))
                .fill_propositional(&mut values);
            let index: usize = variables
                .iter()
                .map(|variable| values[variable.projected].index() * variable.multiplier)
                .sum();
            assert_eq!(
                ids[index],
                Self::UNREACHED,
                "two abstract states share their input values"
            );
            ids[index] = u32::try_from(state_id).expect("abstract state ids fit in u32");
        }
        Some(Self { variables, ids })
    }

    /// The abstract state of the base task state with `propositional` values.
    fn state_id(&self, state: StateValues<'_>) -> Option<usize> {
        let index: usize = self
            .variables
            .iter()
            .map(|variable| state.propositional(variable.original).index() * variable.multiplier)
            .sum();
        let id = self.ids[index];
        (id != Self::UNREACHED).then_some(id as usize)
    }
}

/// The transitions of an explored abstract state space, grouped by target
/// state, and the states backward searches over them start from.
#[derive(Debug)]
struct AbstractTransitions {
    /// `predecessors[offsets[s]..offsets[s + 1]]` are the
    /// `(parent, projected operator)` transitions entering state `s`.
    offsets: Vec<usize>,
    predecessors: Vec<(u32, OperatorIndex)>,
    goal_states: Vec<usize>,
    /// The states the exploration generated but did not expand, ascending.
    frontier: Vec<usize>,
}

impl AbstractTransitions {
    fn new(num_states: usize, exploration: Exploration) -> Self {
        let mut offsets = vec![0usize; num_states + 1];
        for transition in &exploration.transitions {
            offsets[transition.target as usize + 1] += 1;
        }
        for state_id in 0..num_states {
            offsets[state_id + 1] += offsets[state_id];
        }
        let mut next = offsets.clone();
        let mut predecessors =
            vec![(0, OperatorIndex::from_usize(0)); exploration.transitions.len()];
        for transition in exploration.transitions {
            let slot = &mut next[transition.target as usize];
            predecessors[*slot] = (transition.parent, transition.operator);
            *slot += 1;
        }
        Self {
            offsets,
            predecessors,
            goal_states: exploration.goal_states,
            frontier: exploration.frontier,
        }
    }

    fn num_states(&self) -> usize {
        self.offsets.len() - 1
    }

    fn predecessors_of(&self, state_id: usize) -> &[(u32, OperatorIndex)] {
        &self.predecessors[self.offsets[state_id]..self.offsets[state_id + 1]]
    }

    /// The goal distances of all states by backward Dijkstra from `seeds`,
    /// `(state, distance)` pairs, under `operator_cost`.
    fn backward_dijkstra(
        &self,
        seeds: &[(usize, f64)],
        operator_cost: impl Fn(OperatorIndex) -> f64,
    ) -> Vec<f64> {
        let Ok(distances) = backward_goal_distances(
            &mut ProjectionGraph {
                transitions: self,
                operator_cost,
            },
            seeds.iter().copied(),
            no_limit,
            None,
        );
        distances
    }
}

/// The explored abstract state space under per-operator costs, as backward
/// Dijkstra reads it.
struct ProjectionGraph<'a, C> {
    transitions: &'a AbstractTransitions,
    operator_cost: C,
}

impl<C: Fn(OperatorIndex) -> f64> BackwardGraph for ProjectionGraph<'_, C> {
    type Label = OperatorIndex;

    fn num_states(&self) -> usize {
        self.transitions.num_states()
    }

    fn incoming(&mut self, target: usize, incoming: &mut Vec<(usize, OperatorIndex, f64)>) {
        incoming.clear();
        incoming.extend(self.transitions.predecessors_of(target).iter().map(
            |&(parent, operator)| (parent as usize, operator, (self.operator_cost)(operator)),
        ));
    }
}

/// What exploring a projection found.
struct Exploration {
    transitions: Vec<Transition>,
    goal_states: Vec<usize>,
    frontier: Vec<usize>,
}

struct Transition {
    target: u32,
    parent: u32,
    operator: OperatorIndex,
}

/// An open-list entry of the exploration. Entries order by `g + h`, then by
/// `h`, as in numeric Fast Downward, and then by state id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct OpenEntry {
    f: NotNan<f64>,
    h: NotNan<f64>,
    state_id: usize,
    g: NotNan<f64>,
}

/// Explores the abstract state space of `task` best-first from its initial
/// state until `max_new_states` states were newly generated or the open list
/// is empty.
///
/// As in numeric Fast Downward, the open list has no duplicate detection: a
/// state is pushed again whenever it is generated while not yet expanded, and
/// a blind exploration orders by `g` alone.
fn explore(
    task: &Task,
    registry: &mut StateRegistry,
    max_new_states: usize,
    heuristic: PdbInternalHeuristic,
    inner: &mut InnerHeuristics,
) -> Result<Exploration, String> {
    let mut expander = StateExpander::new(StandardGroundedTree::new(task));
    let operators = task.operators();
    let initial_state = registry.register_initial_state();
    assert_eq!(
        initial_state.id(),
        0,
        "a fresh registry registers the initial state first"
    );
    let mut transitions = Vec::new();
    let mut goal_states = Vec::new();
    let mut closed = vec![false];
    let mut generated = vec![true];
    let mut new_states = 0usize;
    let mut open = BinaryHeap::from([Reverse(OpenEntry {
        f: not_nan(0.0),
        h: not_nan(0.0),
        state_id: 0,
        g: not_nan(0.0),
    })]);

    while new_states < max_new_states
        && let Some(Reverse(entry)) = open.pop()
    {
        let state_id = entry.state_id;
        if closed[state_id] {
            continue;
        }
        closed[state_id] = true;
        let state = ConcreteState::new(state_id);
        if registry.view(&state).holds_all(task.goals()) {
            goal_states.push(state_id);
        }
        let mut expansion = expander
            .expand(registry, &state)
            .map_err(|error| error.message)?;
        for &operator_id in expansion.applicable_operators() {
            let (successor, _) = expansion
                .successor(registry, &operators[operator_id.index()])
                .map_err(|error| error.message)?;
            let successor_id = successor.id();
            if successor_id == state_id {
                continue;
            }
            if successor_id == closed.len() {
                closed.push(false);
                generated.push(false);
            }
            transitions.push(Transition {
                target: u32::try_from(successor_id).expect("abstract state ids fit in u32"),
                parent: u32::try_from(state_id).expect("abstract state ids fit in u32"),
                operator: operator_id,
            });
            if closed[successor_id] {
                continue;
            }
            if !generated[successor_id] {
                generated[successor_id] = true;
                new_states += 1;
            }
            let Some(h) = inner.estimate(heuristic, successor_id, registry) else {
                continue;
            };
            let g = entry.g.into_inner() + task.operator_cost(operator_id);
            let h = match heuristic {
                PdbInternalHeuristic::Zero | PdbInternalHeuristic::Blind => 0.0,
                PdbInternalHeuristic::Lmcut => h,
            };
            open.push(Reverse(OpenEntry {
                f: not_nan(g + h),
                h: not_nan(h),
                state_id: successor_id,
                g: not_nan(g),
            }));
        }
    }

    let mut frontier: Vec<usize> = open
        .into_iter()
        .map(|Reverse(entry)| entry.state_id)
        .filter(|&state_id| !closed[state_id])
        .collect();
    frontier.sort_unstable();
    frontier.dedup();
    Ok(Exploration {
        transitions,
        goal_states,
        frontier,
    })
}

/// The inner heuristics a pattern database is built with, memoised per
/// abstract state.
struct InnerHeuristics {
    lmcut: Option<ProjectedLmcut>,
    /// The LM-cut estimate of each abstract state, `Some(None)` for a dead
    /// end.
    cache: Vec<Option<Option<f64>>>,
}

impl InnerHeuristics {
    /// The estimate of `heuristic` for the abstract state `state_id`; `None`
    /// for a dead end.
    fn estimate(
        &mut self,
        heuristic: PdbInternalHeuristic,
        state_id: usize,
        registry: &StateRegistry,
    ) -> Option<f64> {
        match heuristic {
            PdbInternalHeuristic::Zero | PdbInternalHeuristic::Blind => Some(0.0),
            PdbInternalHeuristic::Lmcut => {
                if self.cache.len() <= state_id {
                    self.cache.resize(state_id + 1, None);
                }
                if let Some(estimate) = self.cache[state_id] {
                    return estimate;
                }
                let lmcut = self
                    .lmcut
                    .as_mut()
                    .expect("LM-cut is built when an inner heuristic is LM-cut");
                let estimate = lmcut.estimate_registered(registry, state_id);
                self.cache[state_id] = Some(estimate);
                estimate
            }
        }
    }
}

/// The failed-lookup heuristic of a built pattern database.
#[derive(Debug)]
enum FailedLookupHeuristic {
    Zero,
    Blind,
    Lmcut(Box<RefCell<ProjectedLmcut>>),
}

/// LM-cut on the projection of a pattern database.
#[derive(Debug)]
struct ProjectedLmcut {
    landmarks: LandmarkCutLandmarks,
    propositional: Vec<ExplicitValueIndex>,
    numeric: Vec<NumericValue>,
}

impl ProjectedLmcut {
    fn new(task: Arc<Task>) -> Self {
        Self {
            landmarks: LandmarkCutLandmarks::new(task, LmCutNumericConfig::default()),
            propositional: Vec::new(),
            numeric: Vec::new(),
        }
    }

    /// The estimate for projected state values; `None` for a dead end.
    fn estimate(
        &mut self,
        propositional: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Option<f64> {
        lmcut_estimate(&mut self.landmarks, propositional, numeric)
    }

    fn estimate_registered(&mut self, registry: &StateRegistry, state_id: usize) -> Option<f64> {
        let state = ConcreteState::new(state_id);
        let view = registry.view(&state);
        view.fill_propositional(&mut self.propositional);
        view.fill_numeric(&mut self.numeric)
            .unwrap_or_else(|error| {
                panic!("projected state {state_id} has no numeric values: {error:?}")
            });
        lmcut_estimate(&mut self.landmarks, &self.propositional, &self.numeric)
    }
}

fn lmcut_estimate(
    landmarks: &mut LandmarkCutLandmarks,
    propositional: &[ExplicitValueIndex],
    numeric: &[NumericValue],
) -> Option<f64> {
    // The projection is a task the pattern database built itself, so a
    // failure is an internal inconsistency.
    let (dead_end, value) = landmarks
        .compute_landmark_cost(propositional, numeric)
        .unwrap_or_else(|error| panic!("LM-cut failed on a projected state: {error}"));
    (!dead_end).then_some(value)
}

/// Reused buffers for projecting a base task state and looking it up.
#[derive(Debug, Default)]
struct LookupScratch {
    propositional: Vec<ExplicitValueIndex>,
    numeric: Vec<NumericValue>,
    packed: Vec<u64>,
}

fn is_goal(goals: &[ExplicitFact], propositional: &[ExplicitValueIndex]) -> bool {
    goals
        .iter()
        .all(|goal| propositional[goal.var()] == goal.value_index())
}

fn not_nan(value: f64) -> NotNan<f64> {
    NotNan::new(value).expect("pattern database distances are never NaN")
}
