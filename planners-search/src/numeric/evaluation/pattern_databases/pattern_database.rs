#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fmt;

use ordered_float::NotNan;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::AbstractNumericTask;
use planners_sas::numeric::state_registry::{ConcreteState, StateRegistry};
use planners_sas::numeric::utils::int_packer::IntDoublePacker;
use rustc_hash::FxBuildHasher;
use serde::{Deserialize, Serialize};

type HashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;

use crate::numeric::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig;
use crate::numeric::evaluation::numeric_landmarks::numeric_lm_cut_landmarks::LandmarkCutLandmarks;
use crate::numeric::successor_generator::{ApplicableOperator, GroundedSuccessorGenerator, Node};

use super::projected_task::ProjectedTask;
use super::utils;

#[inline]
fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    const FNV_PRIME: u64 = 0x00000100000001B3;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[inline]
fn hash_state_components(propositional: &[i32], numeric: &[f64]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    let mut hash = FNV_OFFSET;
    for value in propositional {
        hash = hash_bytes(hash, &value.to_le_bytes());
    }
    hash = hash_bytes(hash, &(propositional.len() as u64).to_le_bytes());
    for value in numeric {
        hash = hash_bytes(hash, &value.to_bits().to_le_bytes());
    }
    hash_bytes(hash, &(numeric.len() as u64).to_le_bytes())
}

#[inline]
fn hash_pattern_components(
    propositional: &[i32],
    numeric: &[f64],
    pattern_regular_ids: &[usize],
    pattern_numeric_ids: &[usize],
) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    let mut hash = FNV_OFFSET;
    for &var_id in pattern_regular_ids {
        hash = hash_bytes(hash, &propositional[var_id].to_le_bytes());
    }
    hash = hash_bytes(hash, &(pattern_regular_ids.len() as u64).to_le_bytes());
    for &var_id in pattern_numeric_ids {
        hash = hash_bytes(hash, &numeric[var_id].to_bits().to_le_bytes());
    }
    hash_bytes(hash, &(pattern_numeric_ids.len() as u64).to_le_bytes())
}

#[inline]
fn build_prop_hash_multipliers(task: &ProjectedTask<'_>) -> Vec<usize> {
    let mut multipliers = Vec::with_capacity(task.variables().len());
    let mut product = 1usize;
    for variable in task.variables() {
        multipliers.push(product);
        product = product.saturating_mul(variable.domain_size() as usize);
    }
    multipliers
}

#[inline]
fn compute_prop_hash(propositional: &[i32], multipliers: &[usize]) -> Option<usize> {
    if propositional.len() != multipliers.len() {
        return None;
    }

    let mut hash = 0usize;
    for (value, multiplier) in propositional.iter().zip(multipliers.iter()) {
        let value = usize::try_from(*value).ok()?;
        hash = hash.saturating_add(value.saturating_mul(*multiplier));
    }
    Some(hash)
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PdbState {
    propositional: Vec<i32>,
    numeric: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PdbInternalHeuristic {
    Blind,
    Lmcut,
}

impl Default for PdbInternalHeuristic {
    fn default() -> Self {
        Self::Blind
    }
}

impl fmt::Display for PdbInternalHeuristic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Blind => write!(f, "blind"),
            Self::Lmcut => write!(f, "lmcut"),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct PdbHeuristicConfig {
    pub exploration_heuristic: PdbInternalHeuristic,
    pub frontier_heuristic: PdbInternalHeuristic,
    pub failed_lookup_heuristic: PdbInternalHeuristic,
}

impl Default for PdbHeuristicConfig {
    fn default() -> Self {
        Self {
            exploration_heuristic: PdbInternalHeuristic::Blind,
            frontier_heuristic: PdbInternalHeuristic::Blind,
            failed_lookup_heuristic: PdbInternalHeuristic::Blind,
        }
    }
}

impl fmt::Display for PdbHeuristicConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "exploration_heuristic={}, frontier_heuristic={}, failed_lookup_heuristic={}",
            self.exploration_heuristic,
            self.frontier_heuristic,
            self.failed_lookup_heuristic,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct InnerHeuristicResult {
    dead_end: bool,
    value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PdbOpenEntry {
    state_id: usize,
    f: NotNan<f64>,
    g: NotNan<f64>,
}

impl Eq for PdbOpenEntry {}

impl Ord for PdbOpenEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f
            .cmp(&self.f)
            .then_with(|| other.g.cmp(&self.g))
            .then_with(|| other.state_id.cmp(&self.state_id))
    }
}

impl PartialOrd for PdbOpenEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct LmcutInnerHeuristic<'task> {
    landmark_generator: LandmarkCutLandmarks<'task>,
    propositional_scratch: Vec<i32>,
    numeric_scratch: Vec<f64>,
    default_state_buffer_len: usize,
}

impl<'task> LmcutInnerHeuristic<'task> {
    fn new(task: &'task dyn AbstractNumericTask) -> Self {
        Self {
            landmark_generator: LandmarkCutLandmarks::new(task, LmCutNumericConfig::default()),
            propositional_scratch: Vec::new(),
            numeric_scratch: Vec::new(),
            default_state_buffer_len: IntDoublePacker::from_abstract_task(task).num_bins() as usize,
        }
    }

    fn evaluate_from_concrete_state(
        &mut self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
    ) -> Result<InnerHeuristicResult, String> {
        state.fill_state(registry, &mut self.propositional_scratch);
        registry
            .fill_numeric_vars(state, &mut self.numeric_scratch)
            .map_err(|err| format!("failed to read projected numeric state: {err:?}"))?;
        let propositional = self.propositional_scratch.clone();
        let numeric = self.numeric_scratch.clone();
        self.evaluate_from_values(
            &propositional,
            &numeric,
            state.buffer(registry).len(),
        )
    }

    fn evaluate_from_values(
        &mut self,
        propositional: &[i32],
        numeric: &[f64],
        state_buffer_len: usize,
    ) -> Result<InnerHeuristicResult, String> {
        let (dead_end, value) = self.landmark_generator.compute_landmark_cost(
            propositional,
            state_buffer_len,
            numeric,
            false,
        )?;
        Ok(InnerHeuristicResult { dead_end, value })
    }

    fn evaluate_projected_values(
        &mut self,
        propositional: &[i32],
        numeric: &[f64],
    ) -> Result<InnerHeuristicResult, String> {
        self.evaluate_from_values(propositional, numeric, self.default_state_buffer_len)
    }
}

pub struct PatternDatabase<'task> {
    pub(super) task: ProjectedTask<'task>,
    heuristic_config: PdbHeuristicConfig,
    pub(super) states: Vec<PdbState>,
    state_index: HashMap<u64, Vec<usize>>,
    pattern_index: HashMap<u64, Vec<usize>>,
    full_prop_index: HashMap<usize, Vec<usize>>,
    pub(super) distances: Vec<f64>,
    pub(super) min_operator_cost: f64,
    pub(super) reached_goal_states: usize,
    pub(super) truncated: bool,
    exhausted_abstract_state_space: bool,
    pub(super) frontier_states: Vec<usize>,
    full_prop_hash_multipliers: Vec<usize>,
    state_dependent_numeric_projected_ids: Vec<usize>,
    projection_prop_scratch: RefCell<Vec<i32>>,
    projection_numeric_scratch: RefCell<Vec<f64>>,
    projection_helper_scratch: RefCell<Vec<f64>>,
    direct_numeric_cache_scratch: RefCell<Vec<Option<f64>>>,
}

impl<'task> PatternDatabase<'task> {
    pub fn new(task: ProjectedTask<'task>, max_states: usize) -> Result<Self, String> {
        Self::with_heuristic_config(task, max_states, PdbHeuristicConfig::default())
    }

    pub fn with_heuristic_config(
        task: ProjectedTask<'task>,
        max_states: usize,
        heuristic_config: PdbHeuristicConfig,
    ) -> Result<Self, String> {
        let min_operator_cost = task.min_operator_cost();
        let projection_prop_capacity = task.variables().len();
        let projection_numeric_capacity = task.numeric_variables().len();

        let mut pdb = Self {
            task,
            heuristic_config,
            states: Vec::with_capacity(max_states),
            state_index: HashMap::with_capacity_and_hasher(max_states, FxBuildHasher),
            pattern_index: HashMap::with_capacity_and_hasher(max_states, FxBuildHasher),
            full_prop_index: HashMap::with_capacity_and_hasher(max_states, FxBuildHasher),
            distances: Vec::new(),
            min_operator_cost,
            reached_goal_states: 0,
            truncated: false,
            exhausted_abstract_state_space: false,
            frontier_states: Vec::new(),
            full_prop_hash_multipliers: Vec::new(),
            state_dependent_numeric_projected_ids: Vec::new(),
            projection_prop_scratch: RefCell::new(Vec::with_capacity(projection_prop_capacity)),
            projection_numeric_scratch: RefCell::new(Vec::with_capacity(
                projection_numeric_capacity,
            )),
            projection_helper_scratch: RefCell::new(Vec::new()),
            direct_numeric_cache_scratch: RefCell::new(Vec::new()),
        };
        pdb.full_prop_hash_multipliers = build_prop_hash_multipliers(&pdb.task);
        pdb.state_dependent_numeric_projected_ids = pdb.task.state_dependent_numeric_projected_ids();
        pdb.build(max_states)?;
        // NOTE: un-comment to print summary of the built PDB
        utils::dump_distance_table(&pdb);
        Ok(pdb)
    }

    pub fn lookup(&self, propositional: &[i32], numeric: &[f64]) -> Option<f64> {
        let state_id = self.lookup_state_id(propositional, numeric)?;
        self.distances.get(state_id).copied()
    }

    pub fn lookup_or_fallback(&self, propositional: &[i32], numeric: &[f64]) -> f64 {
        match self.lookup(propositional, numeric) {
            Some(distance) if distance.is_finite() => distance,
            Some(_) if self.is_goal_state(propositional) => 0.0,
            Some(_) if self.exhausted_abstract_state_space => f64::INFINITY,
            Some(_) if self.truncated => self.evaluate_failed_lookup(propositional, numeric),
            Some(distance) => distance,
            None => self.evaluate_failed_lookup(propositional, numeric),
        }
    }

    fn evaluate_failed_lookup(&self, propositional: &[i32], numeric: &[f64]) -> f64 {
        if self.exhausted_abstract_state_space {
            return f64::INFINITY;
        }
        if self.is_goal_state(propositional) {
            return 0.0;
        }

        match self.heuristic_config.failed_lookup_heuristic {
            PdbInternalHeuristic::Blind => self.min_operator_cost(),
            PdbInternalHeuristic::Lmcut => {
                let mut evaluator = LmcutInnerHeuristic::new(&self.task);
                match evaluator
                    .evaluate_projected_values(propositional, numeric)
                {
                    Ok(result) if result.dead_end => f64::INFINITY,
                    Ok(result) => result.value.max(self.min_operator_cost()),
                    Err(_) => self.min_operator_cost(),
                }
            }
        }
    }

    pub fn is_goal_state(&self, propositional: &[i32]) -> bool {
        (0..usize::try_from(self.task.get_num_goals().max(0)).unwrap_or(0)).all(|goal_index| {
            let goal = self.task.get_goal_fact(goal_index as i32);
            propositional.get(goal.var() as usize).copied() == Some(goal.value())
        })
    }

    pub fn min_operator_cost(&self) -> f64 {
        self.min_operator_cost
    }

    pub fn requires_derived_numeric_values(&self) -> bool {
        self.task.requires_derived_numeric_values()
    }

    pub fn abstract_state_values(
        &self,
        propositional: &[i32],
        numeric: &[f64],
    ) -> Result<(Vec<i32>, Vec<f64>), String> {
        self.task.project_state_values(propositional, numeric)
    }

    pub fn lookup_projected_or_fallback_from_state_values(
        &self,
        propositional: &[i32],
        numeric: &[f64],
    ) -> Result<f64, String> {
        let mut projected_prop = self.projection_prop_scratch.borrow_mut();
        let mut projected_num = self.projection_numeric_scratch.borrow_mut();
        let mut helper_values = self.projection_helper_scratch.borrow_mut();

        self.task.project_state_values_into(
            propositional,
            numeric,
            &mut projected_prop,
            &mut projected_num,
            &mut helper_values,
        )?;

        Ok(self.lookup_or_fallback(&projected_prop, &projected_num))
    }

    pub fn expand_numeric_state_values_into(
        &self,
        numeric: &[f64],
        expanded_numeric: &mut Vec<f64>,
    ) -> Result<(), String> {
        self.task
            .expand_numeric_state_values_into(numeric, expanded_numeric)
    }

    pub fn lookup_projected_or_fallback_from_expanded_state_values(
        &self,
        propositional: &[i32],
        expanded_numeric: &[f64],
    ) -> Result<f64, String> {
        let mut projected_prop = self.projection_prop_scratch.borrow_mut();
        let mut projected_num = self.projection_numeric_scratch.borrow_mut();

        self.task.project_state_values_from_expanded_numeric_into(
            propositional,
            expanded_numeric,
            &mut projected_prop,
            &mut projected_num,
        )?;

        Ok(self.lookup_or_fallback(&projected_prop, &projected_num))
    }

    pub fn lookup_or_fallback_from_concrete_state(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
    ) -> Result<f64, String> {
        if self.task.supports_direct_concrete_state_projection() {
            let mut projected_prop = self.projection_prop_scratch.borrow_mut();
            let mut projected_num = self.projection_numeric_scratch.borrow_mut();
            let mut numeric_cache = self.direct_numeric_cache_scratch.borrow_mut();
            self.task.project_concrete_state_values_into(
                state,
                registry,
                &mut projected_prop,
                &mut projected_num,
                &mut numeric_cache,
            )?;
            return Ok(self.lookup_or_fallback(&projected_prop, &projected_num));
        }

        let mut propositional = Vec::new();
        let mut numeric = Vec::new();
        registry
            .fill_state_and_numeric_vars_with_options(
                state,
                &mut propositional,
                &mut numeric,
                self.requires_derived_numeric_values(),
            )
            .map_err(|err| format!("{err:?}"))?;
        self.lookup_projected_or_fallback_from_state_values(&propositional, &numeric)
    }

    fn lookup_state_id(&self, propositional: &[i32], numeric: &[f64]) -> Option<usize> {
        let full_state_lookup = propositional.len() == self.task.variables().len()
            && numeric.len() == self.task.numeric_variables().len();
        let pattern_regular_ids = self.task.pattern_regular_projected_ids();
        let pattern_numeric_ids = self.task.pattern_numeric_projected_ids();

        if full_state_lookup {
            let prop_hash = compute_prop_hash(propositional, &self.full_prop_hash_multipliers)?;
            let candidates = self.full_prop_index.get(&prop_hash)?;
            return candidates.iter().copied().find(|&state_id| {
                let state = &self.states[state_id];
                state.numeric.len() == numeric.len()
                    && state
                        .numeric
                        .iter()
                        .zip(numeric.iter())
                        .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
            });
        }

        let lookup_key = hash_state_components(propositional, numeric);
        let candidates = self.pattern_index.get(&lookup_key)?;

        candidates.iter().copied().find(|&state_id| {
            let state = &self.states[state_id];
            let same_propositional = if full_state_lookup {
                state.propositional == propositional
            } else {
                pattern_regular_ids
                    .iter()
                    .enumerate()
                    .all(|(pattern_index, &var_id)| {
                        state.propositional.get(var_id).copied()
                            == propositional.get(pattern_index).copied()
                    })
            };
            let same_numeric = same_propositional
                && if full_state_lookup {
                    state.numeric.len() == numeric.len()
                        && state
                            .numeric
                            .iter()
                            .zip(numeric.iter())
                            .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
                } else {
                    pattern_numeric_ids
                        .iter()
                        .enumerate()
                        .all(|(pattern_index, &var_id)| {
                            state.numeric.get(var_id).map(|value| value.to_bits())
                                == numeric.get(pattern_index).map(|value| value.to_bits())
                        })
                };
            same_numeric
        })
    }

    pub(super) fn state_propositional_values<'state>(
        &self,
        state: &'state PdbState,
    ) -> &'state [i32] {
        &state.propositional
    }

    pub(super) fn state_numeric_values<'state>(&self, state: &'state PdbState) -> &'state [f64] {
        &state.numeric
    }

    fn rebuild_lookup_indexes(&mut self) {
        self.state_index.clear();
        self.pattern_index.clear();
        self.full_prop_index.clear();

        let pattern_regular_ids = self.task.pattern_regular_projected_ids();
        let pattern_numeric_ids = self.task.pattern_numeric_projected_ids();

        for (state_id, state) in self.states.iter().enumerate() {
            let full_key = hash_state_components(&state.propositional, &state.numeric);
            self.state_index
                .entry(full_key)
                .or_insert_with(|| Vec::with_capacity(1))
                .push(state_id);

            if let Some(prop_hash) = compute_prop_hash(
                &state.propositional,
                &self.full_prop_hash_multipliers,
            ) {
                self.full_prop_index
                    .entry(prop_hash)
                    .or_insert_with(|| Vec::with_capacity(1))
                    .push(state_id);
            }

            let pattern_key = hash_pattern_components(
                &state.propositional,
                &state.numeric,
                pattern_regular_ids,
                pattern_numeric_ids,
            );
            self.pattern_index
                .entry(pattern_key)
                .or_insert_with(|| Vec::with_capacity(1))
                .push(state_id);
        }
    }

    fn build(&mut self, max_states: usize) -> Result<(), String> {
        let (
            built_states,
            distances,
            reached_goal_states,
            frontier_states,
            truncated,
            exhausted_abstract_state_space,
        ) = {
            let mut predecessors: Vec<Vec<(usize, f64)>> = Vec::with_capacity(max_states);
            let successor_generator = GroundedSuccessorGenerator::construct_node_from_task(&self.task);
            let state_packer = IntDoublePacker::from_abstract_task(&self.task);
            let axiom_evaluator = AxiomEvaluator::new(&self.task, &state_packer);
            let mut state_registry = StateRegistry::new(&self.task, &state_packer, &axiom_evaluator);
            let mut applicable_operators: Vec<ApplicableOperator<'_>> = Vec::new();
            let initial_registry_state = state_registry.get_initial_state();
            let mut current_propositional: Vec<i32> = Vec::new();
            let mut successor_numeric: Vec<f64> = Vec::new();
            let mut successor_cost_values: Vec<f64> = Vec::new();
            let mut representative_states: Vec<ConcreteState> = vec![initial_registry_state];
            let mut closed = vec![false];
            let mut seen_or_closed = vec![true];
            let mut open: BinaryHeap<PdbOpenEntry> = BinaryHeap::new();
            open.push(PdbOpenEntry {
                state_id: 0,
                f: NotNan::new(0.0).unwrap(),
                g: NotNan::new(0.0).unwrap(),
            });
            let mut seen_count = 0usize;
            let mut goal_states: Vec<usize> = Vec::new();
            let mut truncated = false;
            let uses_lmcut = matches!(
                self.heuristic_config.exploration_heuristic,
                PdbInternalHeuristic::Lmcut
            ) || matches!(
                self.heuristic_config.frontier_heuristic,
                PdbInternalHeuristic::Lmcut
            ) || matches!(
                self.heuristic_config.failed_lookup_heuristic,
                PdbInternalHeuristic::Lmcut
            );
            let mut construction_lmcut = if uses_lmcut {
                Some(LmcutInnerHeuristic::new(&self.task))
            } else {
                None
            };
            let mut heuristic_cache: Vec<Option<InnerHeuristicResult>> = Vec::new();
            let mut compute_inner_h = |
                heuristic: PdbInternalHeuristic,
                state_id: usize,
                state: &ConcreteState,
                registry: &StateRegistry<'_>,
            | -> Result<InnerHeuristicResult, String> {
                match heuristic {
                    PdbInternalHeuristic::Blind => Ok(InnerHeuristicResult {
                        dead_end: false,
                        value: 0.0,
                    }),
                    PdbInternalHeuristic::Lmcut => {
                        if heuristic_cache.len() <= state_id {
                            heuristic_cache.resize(state_id + 1, None);
                        }
                        if let Some(result) = heuristic_cache[state_id] {
                            return Ok(result);
                        }
                        let result = construction_lmcut
                            .as_mut()
                            .expect("LM-cut inner heuristic must be initialized when configured")
                            .evaluate_from_concrete_state(state, registry)?;
                        heuristic_cache[state_id] = Some(result);
                        println!(
                            "Computed LM-cut heuristic for state {}: value={}, dead_end={}",
                            state_id, result.value, result.dead_end
                        );
                        Ok(result)
                    }
                }
            };
            predecessors.push(Vec::new());

            loop {
                if seen_count >= max_states {
                    truncated = true;
                    break;
                }
                let Some(entry) = open.pop() else {
                    break;
                };
                let state_id = entry.state_id;
                if representative_states.len() % 500 == 0 {
                    println!(
                        "Expanding state {}/{} ({} reached goal states, {} truncated frontier states)",
                        state_id + 1,
                        representative_states.len(),
                        goal_states.len(),
                        0
                    );
                }
                if state_id < closed.len() && closed[state_id] {
                    continue;
                }
                if state_id >= closed.len() {
                    closed.resize(state_id + 1, false);
                }
                closed[state_id] = true;

                applicable_operators.clear();
                let current_registry_state = representative_states[state_id].clone();
                current_registry_state.fill_state(&state_registry, &mut current_propositional);
                if self.is_goal_state(&current_propositional) {
                    goal_states.push(state_id);
                }
                successor_generator
                    .get_applicable_operators(&current_propositional, &mut applicable_operators);

                for (operator, operator_id) in applicable_operators.iter().copied() {
                    let operator_cost = self.task.abstract_operator_cost(operator_id);
                    let successor_state = state_registry
                        .get_successor_state_with_buffers(
                            &current_registry_state,
                            operator,
                            &mut successor_numeric,
                            &mut successor_cost_values,
                        )
                        .map_err(|err| err.message)?;
                    if successor_state.get_id() == current_registry_state.get_id() {
                        continue;
                    }

                    let next_id = successor_state.get_id();
                    if next_id >= representative_states.len() {
                        if next_id != representative_states.len() {
                            return Err(format!(
                                "state registry produced non-contiguous abstract state id {next_id} while {} states are represented",
                                representative_states.len()
                            ));
                        }

                        representative_states.push(successor_state);
                        predecessors.push(Vec::new());
                        if next_id >= closed.len() {
                            closed.resize(next_id + 1, false);
                        }
                        if next_id >= seen_or_closed.len() {
                            seen_or_closed.resize(next_id + 1, false);
                        }
                    }

                    predecessors[next_id].push((state_id, operator_cost));

                    if !seen_or_closed[next_id] {
                        seen_or_closed[next_id] = true;
                        seen_count += 1;
                        let successor_ref = &representative_states[next_id];
                        let inner_h = compute_inner_h(
                            self.heuristic_config.exploration_heuristic,
                            next_id,
                            successor_ref,
                            &state_registry,
                        )?;
                        if !inner_h.dead_end {
                            let g = entry.g.into_inner() + operator_cost;
                            let h = if matches!(
                                self.heuristic_config.exploration_heuristic,
                                PdbInternalHeuristic::Blind
                            ) {
                                0.0
                            } else {
                                inner_h.value
                            };
                            open.push(PdbOpenEntry {
                                state_id: next_id,
                                f: NotNan::new(g + h).map_err(|err| err.to_string())?,
                                g: NotNan::new(g).map_err(|err| err.to_string())?,
                            });
                        }
                    }
                }
            }

            let exhausted_abstract_state_space = open.is_empty();

            let built_states = representative_states
                .iter()
                .map(|state| {
                    Ok(PdbState {
                        propositional: state.get_state(&state_registry),
                        numeric: state_registry
                            .get_numeric_vars(state)
                            .map_err(|err| format!("{err:?}"))?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;

            let mut distances = vec![f64::INFINITY; built_states.len()];
            let mut heap: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();

            let mut reached_goal_states = 0usize;
            for goal_state_id in goal_states {
                reached_goal_states += 1;
                distances[goal_state_id] = 0.0;
                heap.push((Reverse(NotNan::new(0.0).unwrap()), goal_state_id));
            }

            let mut frontier_states: Vec<usize> = Vec::new();
            if truncated {
                let mut seen_frontier = vec![false; built_states.len()];
                while let Some(entry) = open.pop() {
                    let state_id = entry.state_id;
                    if state_id < closed.len() && closed[state_id] {
                        continue;
                    }
                    if state_id >= seen_frontier.len() || seen_frontier[state_id] {
                        continue;
                    }
                    seen_frontier[state_id] = true;
                    frontier_states.push(state_id);
                    let state = &representative_states[state_id];
                    let seed_cost = if self.is_goal_state(&built_states[state_id].propositional) {
                        0.0
                    } else {
                        let inner_h = compute_inner_h(
                            self.heuristic_config.frontier_heuristic,
                            state_id,
                            state,
                            &state_registry,
                        )?;
                        if inner_h.dead_end {
                            continue;
                        }
                        inner_h.value.max(self.min_operator_cost())
                    };
                    if seed_cost + 1e-12 < distances[state_id] {
                        distances[state_id] = seed_cost;
                        heap.push((Reverse(NotNan::new(seed_cost).unwrap()), state_id));
                    }
                }
                frontier_states.sort_unstable();
                frontier_states.dedup();
            }

            while let Some((Reverse(distance), state_id)) = heap.pop() {
                let distance = distance.into_inner();
                if distance > distances[state_id] + 1e-12 {
                    continue;
                }

                for &(parent_id, operator_cost) in &predecessors[state_id] {
                    let alternative = distance + operator_cost;
                    if alternative + 1e-12 < distances[parent_id] {
                        distances[parent_id] = alternative;
                        heap.push((Reverse(NotNan::new(alternative).unwrap()), parent_id));
                    }
                }
            }

            (
                built_states,
                distances,
                reached_goal_states,
                frontier_states,
                truncated,
                exhausted_abstract_state_space,
            )
        };

        self.truncated = truncated;
        self.exhausted_abstract_state_space = exhausted_abstract_state_space;
        self.states = built_states;
        self.distances = distances;
        self.reached_goal_states = reached_goal_states;
        self.frontier_states = frontier_states;
        self.rebuild_lookup_indexes();

        Ok(())
    }
}
