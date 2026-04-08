#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};

use ordered_float::NotNan;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::AbstractNumericTask;
use planners_sas::numeric::state_registry::{ConcreteState, StateRegistry};
use rustc_hash::FxBuildHasher;

type HashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;

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

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PdbState {
    propositional: Vec<i32>,
    numeric: Vec<f64>,
}

pub struct PatternDatabase<'task> {
    pub(super) task: ProjectedTask<'task>,
    pub(super) states: Vec<PdbState>,
    state_index: HashMap<u64, Vec<usize>>,
    pattern_index: HashMap<u64, Vec<usize>>,
    pub(super) distances: Vec<f64>,
    pub(super) min_operator_cost: f64,
    pub(super) reached_goal_states: usize,
    pub(super) truncated: bool,
    pub(super) frontier_states: Vec<usize>,
    projection_prop_scratch: RefCell<Vec<i32>>,
    projection_numeric_scratch: RefCell<Vec<f64>>,
    projection_helper_scratch: RefCell<Vec<f64>>,
}

impl<'task> PatternDatabase<'task> {
    pub fn new(task: ProjectedTask<'task>, max_states: usize) -> Result<Self, String> {
        let min_operator_cost = task.min_operator_cost();
        let projection_prop_capacity = task.variables().len();
        let projection_numeric_capacity = task.numeric_variables().len();

        let mut pdb = Self {
            task,
            states: Vec::with_capacity(max_states),
            state_index: HashMap::with_capacity_and_hasher(max_states, FxBuildHasher),
            pattern_index: HashMap::with_capacity_and_hasher(max_states, FxBuildHasher),
            distances: Vec::new(),
            min_operator_cost,
            reached_goal_states: 0,
            truncated: false,
            frontier_states: Vec::new(),
            projection_prop_scratch: RefCell::new(Vec::with_capacity(projection_prop_capacity)),
            projection_numeric_scratch: RefCell::new(Vec::with_capacity(
                projection_numeric_capacity,
            )),
            projection_helper_scratch: RefCell::new(Vec::new()),
        };
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
            Some(_) if self.truncated => self.min_operator_cost(),
            Some(distance) => distance,
            None => {
                if self.is_goal_state(propositional) {
                    0.0
                } else {
                    self.min_operator_cost()
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

    fn lookup_state_id(&self, propositional: &[i32], numeric: &[f64]) -> Option<usize> {
        let full_state_lookup = propositional.len() == self.task.variables().len()
            && numeric.len() == self.task.numeric_variables().len();
        let pattern_regular_ids = self.task.pattern_regular_projected_ids();
        let pattern_numeric_ids = self.task.pattern_numeric_projected_ids();

        let lookup_key = hash_state_components(propositional, numeric);
        let candidates = if full_state_lookup {
            self.state_index.get(&lookup_key)
        } else {
            self.pattern_index.get(&lookup_key)
        }?;

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

        let pattern_regular_ids = self.task.pattern_regular_projected_ids();
        let pattern_numeric_ids = self.task.pattern_numeric_projected_ids();

        for (state_id, state) in self.states.iter().enumerate() {
            let full_key = hash_state_components(&state.propositional, &state.numeric);
            self.state_index
                .entry(full_key)
                .or_insert_with(|| Vec::with_capacity(1))
                .push(state_id);

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
        let mut predecessors: Vec<Vec<(usize, f64)>> = Vec::with_capacity(max_states);
        let mut frontier_seed_costs: HashMap<usize, f64> =
            HashMap::with_capacity_and_hasher(max_states, FxBuildHasher);
        let successor_generator = GroundedSuccessorGenerator::construct_node_from_task(&self.task);
        let state_packer =
            planners_sas::numeric::utils::int_packer::IntDoublePacker::from_abstract_task(
                &self.task,
            );
        let axiom_evaluator = AxiomEvaluator::new(&self.task, &state_packer);
        let mut state_registry = StateRegistry::new(&self.task, &state_packer, &axiom_evaluator);
        let mut applicable_operators: Vec<ApplicableOperator<'_>> = Vec::new();
        let initial_registry_state = state_registry.get_initial_state();
        let mut current_propositional: Vec<i32> = Vec::new();
        let mut successor_numeric: Vec<f64> = Vec::new();
        let mut successor_cost_values: Vec<f64> = Vec::new();
        let mut representative_states: Vec<ConcreteState> = vec![initial_registry_state];
        predecessors.push(Vec::new());

        let mut queue = VecDeque::from([0usize]);
        while let Some(state_id) = queue.pop_front() {
            if representative_states.len() % 500 == 0 {
                println!(
                    "Expanding state {}/{} ({} reached goal states, {} truncated frontier states)",
                    state_id + 1,
                    representative_states.len(),
                    self.reached_goal_states,
                    self.frontier_states.len()
                );
            }
            if representative_states.len() >= max_states {
                self.truncated = true;
                frontier_seed_costs
                    .entry(state_id)
                    .and_modify(|cost| *cost = cost.min(self.min_operator_cost))
                    .or_insert(self.min_operator_cost);
                for queued_state_id in queue.iter().copied() {
                    frontier_seed_costs
                        .entry(queued_state_id)
                        .and_modify(|cost| *cost = cost.min(self.min_operator_cost))
                        .or_insert(self.min_operator_cost);
                }
                break;
            }
            applicable_operators.clear();
            let current_registry_state = representative_states[state_id].clone();
            current_registry_state.fill_state(&state_registry, &mut current_propositional);
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
                    if representative_states.len() >= max_states {
                        self.truncated = true;
                        frontier_seed_costs
                            .entry(state_id)
                            .and_modify(|cost| *cost = cost.min(operator_cost))
                            .or_insert(operator_cost);
                        continue;
                    }

                    if next_id != representative_states.len() {
                        return Err(format!(
                            "state registry produced non-contiguous abstract state id {next_id} while {} states are represented",
                            representative_states.len()
                        ));
                    }

                    representative_states.push(successor_state);
                    predecessors.push(Vec::new());
                    queue.push_back(next_id);
                }

                predecessors[next_id].push((state_id, operator_cost));
            }
        }

        self.states = representative_states
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
        drop(state_registry);
        drop(axiom_evaluator);
        drop(state_packer);
        drop(successor_generator);
        self.rebuild_lookup_indexes();

        self.distances = vec![f64::INFINITY; self.states.len()];
        let mut heap: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();

        self.reached_goal_states = 0;
        for (state_id, state) in self.states.iter().enumerate() {
            if self.is_goal_state(&state.propositional) {
                self.reached_goal_states += 1;
                self.distances[state_id] = 0.0;
                heap.push((Reverse(NotNan::new(0.0).unwrap()), state_id));
            }
        }

        if self.truncated {
            let mut frontier_states: Vec<usize> = frontier_seed_costs.keys().copied().collect();
            frontier_states.sort_unstable();
            frontier_states.dedup();
            self.frontier_states = frontier_states;

            for (&state_id, &seed_cost) in &frontier_seed_costs {
                if seed_cost + 1e-12 < self.distances[state_id] {
                    self.distances[state_id] = seed_cost;
                    heap.push((Reverse(NotNan::new(seed_cost).unwrap()), state_id));
                }
            }
        } else {
            self.frontier_states.clear();
        }

        while let Some((Reverse(distance), state_id)) = heap.pop() {
            let distance = distance.into_inner();
            if distance > self.distances[state_id] + 1e-12 {
                continue;
            }

            for &(parent_id, operator_cost) in &predecessors[state_id] {
                let alternative = distance + operator_cost;
                if alternative + 1e-12 < self.distances[parent_id] {
                    self.distances[parent_id] = alternative;
                    heap.push((Reverse(NotNan::new(alternative).unwrap()), parent_id));
                }
            }
        }

        Ok(())
    }
}
