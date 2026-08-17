#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fmt;

use ordered_float::NotNan;
use planforge_sas::numeric_task::AbstractNumericTask;
use planforge_sas::state_registry::{ConcreteState, ExpansionContext, StateRegistry};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::int_packer::IntDoublePacker;
use rustc_hash::FxBuildHasher;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

type HashMap<K, V> = std::collections::HashMap<K, V, FxBuildHasher>;

use crate::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig;
use crate::evaluation::numeric_landmarks::numeric_lm_cut_landmarks::LandmarkCutLandmarks;
use crate::successor_generator::GroundedSuccessorGenerator;

use super::projected_task::{PatternLookupProjection, ProjectedTask};
//use super::utils;

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
fn hash_state_components(propositional: &[usize], numeric: &[f64]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    let mut hash = FNV_OFFSET;
    for value in propositional {
        hash = hash_bytes(hash, &value.to_le_bytes());
    }
    hash = hash_bytes(hash, &(propositional.len() as u64).to_le_bytes());
    for value in numeric {
        hash = hash_bytes(hash, &float_tolerance::canonical_bits(*value).to_le_bytes());
    }
    hash_bytes(hash, &(numeric.len() as u64).to_le_bytes())
}

#[inline]
fn hash_pattern_components(
    propositional: &[usize],
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
        hash = hash_bytes(
            hash,
            &float_tolerance::canonical_bits(numeric[var_id]).to_le_bytes(),
        );
    }
    hash_bytes(hash, &(pattern_numeric_ids.len() as u64).to_le_bytes())
}

#[inline]
fn build_prop_hash_multipliers(task: &ProjectedTask<'_>) -> Vec<usize> {
    let mut multipliers = Vec::with_capacity(task.variables().len());
    let mut product = 1usize;
    for variable in task.variables() {
        multipliers.push(product);
        product = product.saturating_mul(variable.domain_size());
    }
    multipliers
}

#[inline]
fn compute_prop_hash(propositional: &[usize], multipliers: &[usize]) -> Option<usize> {
    if propositional.len() != multipliers.len() {
        return None;
    }

    let mut hash = 0usize;
    for (value, multiplier) in propositional.iter().zip(multipliers.iter()) {
        hash = hash.saturating_add(value.saturating_mul(*multiplier));
    }
    Some(hash)
}

#[inline]
fn build_compact_prop_hash_multipliers(
    task: &ProjectedTask<'_>,
    pattern_regular_ids: &[usize],
) -> Vec<usize> {
    let mut multipliers = Vec::with_capacity(pattern_regular_ids.len());
    let mut product = 1usize;
    for &projected_var_id in pattern_regular_ids {
        multipliers.push(product);
        product = product.saturating_mul(task.variables()[projected_var_id].domain_size());
    }
    multipliers
}

#[inline]
fn compute_projected_compact_prop_hash(
    propositional: &[usize],
    pattern_regular_ids: &[usize],
    multipliers: &[usize],
) -> Option<usize> {
    if pattern_regular_ids.len() != multipliers.len() {
        return None;
    }

    let mut hash = 0usize;
    for (&projected_var_id, &multiplier) in pattern_regular_ids.iter().zip(multipliers.iter()) {
        let value = propositional.get(projected_var_id).copied()?;
        hash = hash.saturating_add(value.saturating_mul(multiplier));
    }
    Some(hash)
}

#[inline]
fn fast_hash_bins(bins: &[u64]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;
    let mut hash = FNV_OFFSET;
    for &value in bins {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

struct FrozenPackedDistanceRegistry {
    bin_count: usize,
    state_index: HashMap<u64, Vec<usize>>,
    state_bins: Vec<u64>,
    distances: Vec<f64>,
}

impl FrozenPackedDistanceRegistry {
    fn new(bin_count: usize, capacity: usize) -> Self {
        Self {
            bin_count,
            state_index: HashMap::with_capacity_and_hasher(capacity, FxBuildHasher),
            state_bins: Vec::with_capacity(capacity.saturating_mul(bin_count)),
            distances: Vec::with_capacity(capacity),
        }
    }

    fn clear(&mut self) {
        self.state_index.clear();
        self.state_bins.clear();
        self.distances.clear();
    }

    fn is_empty(&self) -> bool {
        self.distances.is_empty()
    }

    fn lookup_distance(&self, bins: &[u64]) -> Option<f64> {
        let hash = fast_hash_bins(bins);
        let candidates = self.state_index.get(&hash)?;
        candidates.iter().copied().find_map(|entry_id| {
            let start = entry_id.checked_mul(self.bin_count)?;
            let end = start.checked_add(self.bin_count)?;
            let entry_bins = self.state_bins.get(start..end)?;
            (entry_bins == bins)
                .then(|| self.distances.get(entry_id).copied())
                .flatten()
        })
    }

    fn insert_min_distance(&mut self, bins: &[u64], distance: f64) {
        let hash = fast_hash_bins(bins);
        let bucket = self
            .state_index
            .entry(hash)
            .or_insert_with(|| Vec::with_capacity(1));
        if let Some(existing_entry_id) = bucket.iter().copied().find(|&entry_id| {
            let start = entry_id * self.bin_count;
            let end = start + self.bin_count;
            self.state_bins[start..end] == bins[..]
        }) {
            if distance < self.distances[existing_entry_id] {
                self.distances[existing_entry_id] = distance;
            }
        } else {
            let entry_id = self.distances.len();
            self.state_bins.extend_from_slice(bins);
            self.distances.push(distance);
            bucket.push(entry_id);
        }
    }
}

enum CompactNumericDistanceIndex {
    Empty,
    One(HashMap<(u64, u64), f64>),
    Two(HashMap<(u64, u64, u64), f64>),
    Many(FrozenPackedDistanceRegistry),
}

impl CompactNumericDistanceIndex {
    fn new(numeric_count: usize, capacity: usize) -> Self {
        match numeric_count {
            0 => Self::Empty,
            1 => Self::One(HashMap::with_capacity_and_hasher(capacity, FxBuildHasher)),
            2 => Self::Two(HashMap::with_capacity_and_hasher(capacity, FxBuildHasher)),
            _ => Self::Many(FrozenPackedDistanceRegistry::new(
                1 + numeric_count,
                capacity,
            )),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::Empty => {}
            Self::One(index) => index.clear(),
            Self::Two(index) => index.clear(),
            Self::Many(index) => index.clear(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Empty => true,
            Self::One(index) => index.is_empty(),
            Self::Two(index) => index.is_empty(),
            Self::Many(index) => index.is_empty(),
        }
    }

    fn lookup_distance(&self, bins: &[u64]) -> Option<f64> {
        match self {
            Self::Empty => None,
            Self::One(index) => {
                let [prop_hash, value] = *bins else {
                    return None;
                };
                index.get(&(prop_hash, value)).copied()
            }
            Self::Two(index) => {
                let [prop_hash, first, second] = *bins else {
                    return None;
                };
                index.get(&(prop_hash, first, second)).copied()
            }
            Self::Many(index) => index.lookup_distance(bins),
        }
    }

    fn insert_min_distance(&mut self, bins: &[u64], distance: f64) {
        match self {
            Self::Empty => {}
            Self::One(index) => {
                let [prop_hash, value] = *bins else {
                    return;
                };
                let slot = index.entry((prop_hash, value)).or_insert(distance);
                *slot = slot.min(distance);
            }
            Self::Two(index) => {
                let [prop_hash, first, second] = *bins else {
                    return;
                };
                let slot = index.entry((prop_hash, first, second)).or_insert(distance);
                *slot = slot.min(distance);
            }
            Self::Many(index) => index.insert_min_distance(bins, distance),
        }
    }
}

fn build_pattern_lookup_packer(task: &ProjectedTask<'_>) -> IntDoublePacker {
    let mut ranges = Vec::with_capacity(
        task.pattern_regular_projected_ids().len() + task.pattern_numeric_projected_ids().len(),
    );
    for &projected_var_id in task.pattern_regular_projected_ids() {
        ranges.push(task.variables()[projected_var_id].domain_size() as u64);
    }
    ranges.extend(std::iter::repeat_n(
        u64::MAX,
        task.pattern_numeric_projected_ids().len(),
    ));
    IntDoublePacker::new(&ranges)
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PdbState {
    propositional: Vec<usize>,
    numeric: Vec<f64>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PdbInternalHeuristic {
    Zero,
    #[default]
    Blind,
    Lmcut,
}

impl fmt::Display for PdbInternalHeuristic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "zero"),
            Self::Blind => write!(f, "blind"),
            Self::Lmcut => write!(f, "lmcut"),
        }
    }
}

impl crate::config::FromOptionValue for PdbInternalHeuristic {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "zero" => Ok(Self::Zero),
            "blind" => Ok(Self::Blind),
            "lmcut" => Ok(Self::Lmcut),
            other => Err(format!("invalid PdbInternalHeuristic `{other}`")),
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
            self.exploration_heuristic, self.frontier_heuristic, self.failed_lookup_heuristic,
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
    propositional_scratch: Vec<usize>,
    numeric_scratch: Vec<f64>,
    default_state_buffer_len: usize,
}

impl<'task> LmcutInnerHeuristic<'task> {
    fn new(task: &'task dyn AbstractNumericTask) -> Self {
        Self {
            landmark_generator: LandmarkCutLandmarks::new(task, LmCutNumericConfig::default()),
            propositional_scratch: Vec::new(),
            numeric_scratch: Vec::new(),
            default_state_buffer_len: IntDoublePacker::from_abstract_task(task).num_bins(),
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
        self.evaluate_from_values(&propositional, &numeric, state.buffer(registry).len())
    }

    fn evaluate_from_values(
        &mut self,
        propositional: &[usize],
        numeric: &[f64],
        state_buffer_len: usize,
    ) -> Result<InnerHeuristicResult, String> {
        let (dead_end, value) = self.landmark_generator.compute_landmark_cost(
            propositional,
            state_buffer_len,
            numeric,
        )?;
        Ok(InnerHeuristicResult { dead_end, value })
    }

    fn evaluate_projected_values(
        &mut self,
        propositional: &[usize],
        numeric: &[f64],
    ) -> Result<InnerHeuristicResult, String> {
        self.evaluate_from_values(propositional, numeric, self.default_state_buffer_len)
    }
}

pub struct PatternDatabase<'task> {
    pub(super) task: ProjectedTask<'task>,
    lookup_projection: PatternLookupProjection,
    heuristic_config: PdbHeuristicConfig,
    pub(super) states: Vec<PdbState>,
    state_index: HashMap<u64, Vec<usize>>,
    pattern_index: HashMap<u64, Vec<usize>>,
    packed_pattern_registry: FrozenPackedDistanceRegistry,
    full_prop_index: HashMap<usize, Vec<usize>>,
    pub(super) distances: Vec<f64>,
    goal_state_ids: Vec<usize>,
    transition_predecessors: Vec<Vec<(usize, usize)>>,
    pub(super) min_operator_cost: f64,
    pub(super) reached_goal_states: usize,
    pub(super) truncated: bool,
    exhausted_abstract_state_space: bool,
    pub(super) frontier_states: Vec<usize>,
    full_prop_hash_multipliers: Vec<usize>,
    compact_prop_hash_multipliers: Vec<usize>,
    compact_prop_distances: Vec<f64>,
    pattern_lookup_packer: IntDoublePacker,
    compact_numeric_registry: CompactNumericDistanceIndex,
    state_dependent_numeric_projected_ids: Vec<usize>,
    failed_lookup_cache: RefCell<HashMap<u64, f64>>,
    projection_prop_scratch: RefCell<Vec<usize>>,
    projection_numeric_scratch: RefCell<Vec<f64>>,
    direct_numeric_cache_scratch: RefCell<Vec<Option<f64>>>,
    pattern_lookup_bins_scratch: RefCell<Vec<u64>>,
    compact_numeric_bins_scratch: RefCell<Vec<u64>>,
    failed_lookup_lmcut: RefCell<Option<LmcutInnerHeuristic<'task>>>,
    failed_lookup_lmcut_task: RefCell<Option<Box<ProjectedTask<'task>>>>,
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
        let lookup_projection = PatternLookupProjection::from_projected_task(&task);
        let projection_prop_capacity = task.variables().len();
        let projection_numeric_capacity = task.numeric_variables().len();
        let compact_prop_hash_multipliers =
            build_compact_prop_hash_multipliers(&task, task.pattern_regular_projected_ids());
        let pattern_lookup_packer = build_pattern_lookup_packer(&task);
        let pattern_lookup_bin_count = pattern_lookup_packer.num_bins();
        let compact_numeric_bin_count = 1 + task.pattern_numeric_projected_ids().len();

        let mut pdb = Self {
            task,
            lookup_projection,
            heuristic_config,
            states: Vec::with_capacity(max_states),
            state_index: HashMap::with_capacity_and_hasher(max_states, FxBuildHasher),
            pattern_index: HashMap::with_capacity_and_hasher(max_states, FxBuildHasher),
            packed_pattern_registry: FrozenPackedDistanceRegistry::new(
                pattern_lookup_bin_count,
                max_states,
            ),
            full_prop_index: HashMap::with_capacity_and_hasher(max_states, FxBuildHasher),
            distances: Vec::new(),
            goal_state_ids: Vec::new(),
            transition_predecessors: Vec::new(),
            min_operator_cost,
            reached_goal_states: 0,
            truncated: false,
            exhausted_abstract_state_space: false,
            frontier_states: Vec::new(),
            full_prop_hash_multipliers: Vec::new(),
            compact_prop_hash_multipliers,
            compact_prop_distances: Vec::new(),
            pattern_lookup_packer,
            compact_numeric_registry: CompactNumericDistanceIndex::new(
                compact_numeric_bin_count.saturating_sub(1),
                max_states,
            ),
            state_dependent_numeric_projected_ids: Vec::new(),
            failed_lookup_cache: RefCell::new(HashMap::default()),
            projection_prop_scratch: RefCell::new(Vec::with_capacity(projection_prop_capacity)),
            projection_numeric_scratch: RefCell::new(Vec::with_capacity(
                projection_numeric_capacity,
            )),
            direct_numeric_cache_scratch: RefCell::new(Vec::new()),
            pattern_lookup_bins_scratch: RefCell::new(vec![0; pattern_lookup_bin_count]),
            compact_numeric_bins_scratch: RefCell::new(vec![0; compact_numeric_bin_count]),
            failed_lookup_lmcut: RefCell::new(None),
            failed_lookup_lmcut_task: RefCell::new(None),
        };
        pdb.full_prop_hash_multipliers = build_prop_hash_multipliers(&pdb.task);
        pdb.state_dependent_numeric_projected_ids =
            pdb.task.state_dependent_numeric_projected_ids();
        pdb.build(max_states)?;
        // NOTE: Uncomment to print summary of the built PDB.
        // utils::dump_distance_table(&pdb);
        Ok(pdb)
    }

    pub fn lookup(&self, propositional: &[usize], numeric: &[f64]) -> Option<f64> {
        let full_state_lookup = propositional.len() == self.task.variables().len()
            && numeric.len() == self.task.numeric_variables().len();
        let pattern_regular_ids = self.task.pattern_regular_projected_ids();
        let pattern_numeric_ids = self.task.pattern_numeric_projected_ids();

        if numeric.is_empty()
            && pattern_numeric_ids.is_empty()
            && propositional.len() == pattern_regular_ids.len()
            && let Some(distance) = self.lookup_compact_prop_distance(propositional)
        {
            return Some(distance);
        }

        if !numeric.is_empty()
            && !pattern_numeric_ids.is_empty()
            && propositional.len() == pattern_regular_ids.len()
            && numeric.len() == pattern_numeric_ids.len()
            && let Some(prop_hash) =
                compute_prop_hash(propositional, &self.compact_prop_hash_multipliers)
            && let Some(distance) =
                self.lookup_compact_numeric_distance_from_compact_values(prop_hash, numeric)
        {
            return Some(distance);
        }

        if full_state_lookup {
            let prop_hash = compute_prop_hash(propositional, &self.full_prop_hash_multipliers)?;
            let candidates = self.full_prop_index.get(&prop_hash)?;
            return candidates
                .iter()
                .copied()
                .filter(|&state_id| {
                    let state = &self.states[state_id];
                    state.numeric.len() == numeric.len()
                        && state
                            .numeric
                            .iter()
                            .zip(numeric.iter())
                            .all(|(lhs, rhs)| float_tolerance::equal(*lhs, *rhs))
                })
                .filter_map(|state_id| self.distances.get(state_id).copied())
                .min_by(|lhs, rhs| lhs.total_cmp(rhs));
        }

        if propositional.len() == self.task.pattern_regular_projected_ids().len()
            && numeric.len() == self.task.pattern_numeric_projected_ids().len()
        {
            return self.lookup_packed_pattern_distance_from_compact_values(propositional, numeric);
        }

        let lookup_key = hash_state_components(propositional, numeric);
        let candidates = self.pattern_index.get(&lookup_key)?;

        candidates
            .iter()
            .copied()
            .filter(|&state_id| {
                let state = &self.states[state_id];
                pattern_regular_ids
                    .iter()
                    .enumerate()
                    .all(|(pattern_index, &var_id)| {
                        state.propositional.get(var_id).copied()
                            == propositional.get(pattern_index).copied()
                    })
                    && pattern_numeric_ids
                        .iter()
                        .enumerate()
                        .all(|(pattern_index, &var_id)| {
                            state
                                .numeric
                                .get(var_id)
                                .map(|value| float_tolerance::canonical_bits(*value))
                                == numeric
                                    .get(pattern_index)
                                    .map(|value| float_tolerance::canonical_bits(*value))
                        })
            })
            .filter_map(|state_id| self.distances.get(state_id).copied())
            .min_by(|lhs, rhs| lhs.total_cmp(rhs))
    }

    pub fn lookup_or_fallback(&self, propositional: &[usize], numeric: &[f64]) -> f64 {
        match self.lookup(propositional, numeric) {
            Some(distance) if distance.is_finite() => distance,
            Some(_) if self.is_goal_state(propositional) => 0.0,
            Some(_) if self.exhausted_abstract_state_space => f64::INFINITY,
            Some(_) if self.truncated => self.evaluate_failed_lookup(propositional, numeric),
            Some(distance) => distance,
            None => self.evaluate_failed_lookup(propositional, numeric),
        }
    }

    fn lookup_pattern_distance_in_projected_values(
        &self,
        propositional: &[usize],
        numeric: &[f64],
    ) -> Option<f64> {
        if propositional.len() != self.task.variables().len()
            || numeric.len() != self.task.numeric_variables().len()
        {
            return self.lookup(propositional, numeric);
        }

        if numeric.is_empty()
            && self.task.pattern_numeric_projected_ids().is_empty()
            && let Some(distance) = self.lookup_projected_compact_prop_distance(propositional)
        {
            return Some(distance);
        }

        if !numeric.is_empty()
            && !self.task.pattern_numeric_projected_ids().is_empty()
            && let Some(prop_hash) = self.lookup_projected_compact_prop_hash(propositional)
            && let Some(distance) =
                self.lookup_compact_numeric_distance_from_projected_values(prop_hash, numeric)
        {
            return Some(distance);
        }

        self.lookup_packed_pattern_distance_from_projected_values(propositional, numeric)
    }

    fn lookup_projected_compact_prop_hash(&self, propositional: &[usize]) -> Option<usize> {
        compute_projected_compact_prop_hash(
            propositional,
            self.task.pattern_regular_projected_ids(),
            &self.compact_prop_hash_multipliers,
        )
    }

    fn lookup_compact_prop_distance(&self, propositional: &[usize]) -> Option<f64> {
        if self.compact_prop_distances.is_empty() {
            return None;
        }
        let index = compute_prop_hash(propositional, &self.compact_prop_hash_multipliers)?;
        self.lookup_compact_prop_distance_by_hash(index)
    }

    fn lookup_compact_prop_distance_by_hash(&self, index: usize) -> Option<f64> {
        self.compact_prop_distances
            .get(index)
            .copied()
            .filter(|distance| distance.is_finite())
    }

    fn lookup_projected_compact_prop_distance(&self, propositional: &[usize]) -> Option<f64> {
        if self.compact_prop_distances.is_empty() {
            return None;
        }
        let index = self.lookup_projected_compact_prop_hash(propositional)?;
        self.lookup_compact_prop_distance_by_hash(index)
    }

    fn lookup_compact_numeric_distance(&self, bins: &[u64]) -> Option<f64> {
        if self.compact_numeric_registry.is_empty() {
            return None;
        }
        self.compact_numeric_registry.lookup_distance(bins)
    }

    fn lookup_compact_numeric_distance_from_compact_values(
        &self,
        prop_hash: usize,
        numeric: &[f64],
    ) -> Option<f64> {
        let mut bins = self.compact_numeric_bins_scratch.borrow_mut();
        bins.clear();
        bins.resize(1 + numeric.len(), 0);
        bins[0] = prop_hash as u64;
        for (numeric_index, value) in numeric.iter().enumerate() {
            bins[numeric_index + 1] = float_tolerance::canonical_bits(*value);
        }
        self.lookup_compact_numeric_distance(&bins)
    }

    fn lookup_compact_numeric_distance_from_projected_values(
        &self,
        prop_hash: usize,
        numeric: &[f64],
    ) -> Option<f64> {
        let mut bins = self.compact_numeric_bins_scratch.borrow_mut();
        bins.clear();
        bins.resize(1 + self.task.pattern_numeric_projected_ids().len(), 0);
        bins[0] = prop_hash as u64;
        for (numeric_index, &projected_numeric_id) in
            self.task.pattern_numeric_projected_ids().iter().enumerate()
        {
            bins[numeric_index + 1] =
                float_tolerance::canonical_bits(numeric[projected_numeric_id]);
        }
        self.lookup_compact_numeric_distance(&bins)
    }

    fn pack_pattern_values_into_bins(
        &self,
        propositional: &[usize],
        numeric: &[f64],
        bins: &mut Vec<u64>,
    ) -> Option<()> {
        if propositional.len() != self.task.pattern_regular_projected_ids().len()
            || numeric.len() != self.task.pattern_numeric_projected_ids().len()
        {
            return None;
        }

        bins.clear();
        bins.resize(self.pattern_lookup_packer.num_bins(), 0);

        for (var_id, value) in propositional.iter().enumerate() {
            self.pattern_lookup_packer.set(bins, var_id, *value as u64);
        }
        let prop_len = propositional.len();
        for (numeric_index, value) in numeric.iter().enumerate() {
            self.pattern_lookup_packer.set(
                bins,
                prop_len + numeric_index,
                self.pattern_lookup_packer.pack_double(*value),
            );
        }

        Some(())
    }

    fn pack_pattern_projected_values_into_bins(
        &self,
        propositional: &[usize],
        numeric: &[f64],
        bins: &mut Vec<u64>,
    ) -> Option<()> {
        if propositional.len() != self.task.variables().len()
            || numeric.len() != self.task.numeric_variables().len()
        {
            return None;
        }

        bins.clear();
        bins.resize(self.pattern_lookup_packer.num_bins(), 0);

        for (compact_index, &projected_var_id) in
            self.task.pattern_regular_projected_ids().iter().enumerate()
        {
            self.pattern_lookup_packer.set(
                bins,
                compact_index,
                propositional[projected_var_id] as u64,
            );
        }
        let prop_len = self.task.pattern_regular_projected_ids().len();
        for (numeric_index, &projected_numeric_id) in
            self.task.pattern_numeric_projected_ids().iter().enumerate()
        {
            self.pattern_lookup_packer.set(
                bins,
                prop_len + numeric_index,
                self.pattern_lookup_packer
                    .pack_double(numeric[projected_numeric_id]),
            );
        }

        Some(())
    }

    fn lookup_packed_pattern_distance(&self, bins: &[u64]) -> Option<f64> {
        self.packed_pattern_registry.lookup_distance(bins)
    }

    fn lookup_packed_pattern_distance_from_compact_values(
        &self,
        propositional: &[usize],
        numeric: &[f64],
    ) -> Option<f64> {
        let mut bins = self.pattern_lookup_bins_scratch.borrow_mut();
        self.pack_pattern_values_into_bins(propositional, numeric, &mut bins)?;
        self.lookup_packed_pattern_distance(&bins)
    }

    fn lookup_packed_pattern_distance_from_projected_values(
        &self,
        propositional: &[usize],
        numeric: &[f64],
    ) -> Option<f64> {
        let mut bins = self.pattern_lookup_bins_scratch.borrow_mut();
        self.pack_pattern_projected_values_into_bins(propositional, numeric, &mut bins)?;
        self.lookup_packed_pattern_distance(&bins)
    }

    fn lookup_pattern_distance_from_concrete_state(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
    ) -> Result<Option<f64>, String> {
        let prop_hash = self.task.compact_pattern_prop_hash_from_concrete_state(
            state,
            registry,
            &self.compact_prop_hash_multipliers,
        )?;

        if !self.task.pattern_numeric_projected_ids().is_empty() {
            let mut packed_bins = self.compact_numeric_bins_scratch.borrow_mut();
            let mut numeric_cache = self.direct_numeric_cache_scratch.borrow_mut();
            self.task.fill_pattern_numeric_concrete_state_bins_into(
                state,
                registry,
                &mut packed_bins,
                &mut numeric_cache,
            )?;
            packed_bins[0] = prop_hash as u64;
            return Ok(self.lookup_compact_numeric_distance(&packed_bins));
        }

        if let Some(distance) = self.lookup_compact_prop_distance_by_hash(prop_hash) {
            return Ok(Some(distance));
        }

        let mut packed_bins = self.pattern_lookup_bins_scratch.borrow_mut();
        let mut numeric_cache = self.direct_numeric_cache_scratch.borrow_mut();
        self.task.pack_pattern_concrete_state_values_into(
            state,
            registry,
            &self.pattern_lookup_packer,
            &mut packed_bins,
            &mut numeric_cache,
        )?;
        Ok(self.lookup_packed_pattern_distance(&packed_bins))
    }

    fn lookup_pattern_distance_from_source_state_values(
        &self,
        propositional: &[usize],
        source_numeric: &[f64],
    ) -> Result<Option<f64>, String> {
        if self.task.pattern_numeric_projected_ids().is_empty() {
            let prop_hash = self.lookup_projection.compact_prop_hash_from_state_values(
                propositional,
                &self.compact_prop_hash_multipliers,
            )?;
            if let Some(distance) = self.lookup_compact_prop_distance_by_hash(prop_hash) {
                return Ok(Some(distance));
            }
        }

        if !self.task.pattern_numeric_projected_ids().is_empty() {
            let prop_hash = self.lookup_projection.compact_prop_hash_from_state_values(
                propositional,
                &self.compact_prop_hash_multipliers,
            )?;
            let mut packed_bins = self.compact_numeric_bins_scratch.borrow_mut();
            self.lookup_projection
                .fill_pattern_numeric_bins_from_source_numeric_into(
                    source_numeric,
                    &mut packed_bins,
                )?;
            packed_bins[0] = prop_hash as u64;
            return Ok(self.lookup_compact_numeric_distance(&packed_bins));
        }

        let mut packed_bins = self.pattern_lookup_bins_scratch.borrow_mut();
        self.lookup_projection
            .pack_pattern_state_values_from_source_numeric_into(
                propositional,
                source_numeric,
                &self.pattern_lookup_packer,
                &mut packed_bins,
            )?;
        Ok(self.lookup_packed_pattern_distance(&packed_bins))
    }

    #[inline]
    fn lookup_pattern_distance_from_source_state_values_fast(
        &self,
        propositional: &[usize],
        source_numeric: &[f64],
    ) -> Option<f64> {
        if self.task.pattern_numeric_projected_ids().is_empty() {
            let prop_hash = self
                .lookup_projection
                .compact_prop_hash_from_state_values_unchecked(
                    propositional,
                    &self.compact_prop_hash_multipliers,
                );
            return self.lookup_compact_prop_distance_by_hash(prop_hash);
        }

        let prop_hash = self
            .lookup_projection
            .compact_prop_hash_from_state_values_unchecked(
                propositional,
                &self.compact_prop_hash_multipliers,
            );
        let mut packed_bins = self.compact_numeric_bins_scratch.borrow_mut();
        self.lookup_projection
            .fill_pattern_numeric_bins_from_source_numeric_into_unchecked(
                source_numeric,
                &mut packed_bins,
            );
        packed_bins[0] = prop_hash as u64;
        self.lookup_compact_numeric_distance(&packed_bins)
    }

    fn lookup_pattern_or_fallback_in_projected_values(
        &self,
        propositional: &[usize],
        numeric: &[f64],
    ) -> f64 {
        match self.lookup_pattern_distance_in_projected_values(propositional, numeric) {
            Some(distance) if distance.is_finite() => distance,
            Some(_) if self.is_goal_state(propositional) => 0.0,
            Some(_) if self.exhausted_abstract_state_space => f64::INFINITY,
            Some(_) if self.truncated => self.evaluate_failed_lookup(propositional, numeric),
            Some(distance) => distance,
            None => self.evaluate_failed_lookup(propositional, numeric),
        }
    }

    fn evaluate_failed_lookup(&self, propositional: &[usize], numeric: &[f64]) -> f64 {
        if self.exhausted_abstract_state_space {
            // Infinity is sound *here*, and the reason is worth stating, because
            // the same expression is unsound one line of reasoning away.
            //
            // The abstract state space was enumerated in full. A state with no
            // entry therefore has no abstract counterpart that reaches the goal,
            // so infinity is a valid lower bound: it was proved, not assumed.
            //
            // Where the space was *not* exhausted, a missing entry only means
            // "not computed", an unknown lower bound is zero, and returning
            // infinity would record a solvable state as a dead end. That is
            // exactly the bug the online saturated cost partitioning carried, and
            // it made the planner report a solvable task unsolvable. The branches
            // below are what keep this side of that line.
            return f64::INFINITY;
        }
        if self.is_goal_state(propositional) {
            return 0.0;
        }

        match self.heuristic_config.failed_lookup_heuristic {
            PdbInternalHeuristic::Blind => self.min_operator_cost(),
            PdbInternalHeuristic::Zero => 0.0,
            PdbInternalHeuristic::Lmcut => {
                let lookup_key = hash_state_components(propositional, numeric);
                if let Some(distance) = self.failed_lookup_cache.borrow().get(&lookup_key).copied()
                {
                    return distance;
                }

                let distance = match self.evaluate_failed_lookup_lmcut(propositional, numeric) {
                    Ok(result) if result.dead_end => f64::INFINITY,
                    Ok(result) => result.value.max(self.min_operator_cost()),
                    // The LM-cut instance runs on the projected task this PDB
                    // built itself, so a failure is an internal inconsistency.
                    // Substituting `min_operator_cost` here would fabricate a
                    // heuristic value and, worse, cache it for every later
                    // lookup of the same state.
                    Err(error) => {
                        panic!("LM-cut fallback failed on a projected PDB state: {error}")
                    }
                };
                self.failed_lookup_cache
                    .borrow_mut()
                    .insert(lookup_key, distance);
                distance
            }
        }
    }

    fn evaluate_failed_lookup_lmcut(
        &self,
        propositional: &[usize],
        numeric: &[f64],
    ) -> Result<InnerHeuristicResult, String> {
        if self.failed_lookup_lmcut.borrow().is_none() {
            let mut task_slot = self.failed_lookup_lmcut_task.borrow_mut();
            if task_slot.is_none() {
                *task_slot = Some(Box::new(self.task.clone()));
            }
            let task_ref = task_slot
                .as_deref()
                .expect("failed lookup LM-cut task must be initialized")
                as &dyn AbstractNumericTask;
            // The boxed task is stored in `failed_lookup_lmcut_task` and is
            // declared after `failed_lookup_lmcut`, so it outlives the LM-cut
            // object during `PatternDatabase` drop. The box allocation is
            // stable even if `PatternDatabase` moves.
            let task_ref = unsafe {
                std::mem::transmute::<&dyn AbstractNumericTask, &'task dyn AbstractNumericTask>(
                    task_ref,
                )
            };
            drop(task_slot);

            *self.failed_lookup_lmcut.borrow_mut() = Some(LmcutInnerHeuristic::new(task_ref));
        }

        self.failed_lookup_lmcut
            .borrow_mut()
            .as_mut()
            .expect("failed lookup LM-cut must be initialized")
            .evaluate_projected_values(propositional, numeric)
    }

    pub fn is_goal_state(&self, propositional: &[usize]) -> bool {
        (0..self.task.get_num_goals()).all(|goal_index| {
            let goal = self.task.get_goal_fact(goal_index);
            propositional.get(goal.var()).copied() == Some(goal.value())
        })
    }

    pub fn min_operator_cost(&self) -> f64 {
        self.min_operator_cost
    }

    fn lookup_exact_state_id(&self, propositional: &[usize], numeric: &[f64]) -> Option<usize> {
        let lookup_key = hash_state_components(propositional, numeric);
        self.state_index
            .get(&lookup_key)?
            .iter()
            .copied()
            .find(|&state_id| {
                let state = &self.states[state_id];
                state.propositional == propositional
                    && state.numeric.len() == numeric.len()
                    && state
                        .numeric
                        .iter()
                        .zip(numeric.iter())
                        .all(|(lhs, rhs)| float_tolerance::equal(*lhs, *rhs))
            })
    }

    pub fn abstract_state_id_from_source_state_values(
        &self,
        propositional: &[usize],
        source_numeric: &[f64],
    ) -> Result<Option<usize>, String> {
        let mut projected_prop = self.projection_prop_scratch.borrow_mut();
        let mut projected_num = self.projection_numeric_scratch.borrow_mut();
        self.lookup_projection
            .project_state_values_from_source_numeric_into(
                propositional,
                source_numeric,
                &mut projected_prop,
                &mut projected_num,
            )?;
        Ok(self.lookup_exact_state_id(&projected_prop, &projected_num))
    }

    pub fn abstract_state_id_from_concrete_state(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
    ) -> Result<Option<usize>, String> {
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
        Ok(self.lookup_exact_state_id(&projected_prop, &projected_num))
    }

    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    pub fn distance_for_state_id(&self, state_id: usize) -> Result<f64, String> {
        self.distances.get(state_id).copied().ok_or_else(|| {
            format!(
                "PDB abstract state id {state_id} out of bounds for {} states",
                self.distances.len()
            )
        })
    }

    pub fn relevant_operator_ids(&self) -> Vec<usize> {
        let mut relevant = if self.exhausted_abstract_state_space {
            self.transition_predecessors
                .iter()
                .flatten()
                .map(|&(_, projected_operator_id)| {
                    self.task
                        .base_operator_id(projected_operator_id)
                        .expect("projected operator must retain its base operator ID")
                })
                .collect()
        } else {
            self.task.base_operator_ids().to_vec()
        };
        relevant.sort_unstable();
        relevant.dedup();
        relevant
    }

    pub fn build_cost_partitioned_distance_table(
        &self,
        operator_costs: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let distances = self.build_goal_distances(operator_costs)?;
        let saturated_costs =
            self.saturated_costs_from_distances(&distances, operator_costs.len())?;
        Ok((distances, saturated_costs))
    }

    /// Builds goal distances using the supplied operator costs without computing
    /// saturated costs.  Used by the SCP online order generator.
    pub fn build_goal_distances(&self, operator_costs: &[f64]) -> Result<Vec<f64>, String> {
        let mut distances = vec![f64::INFINITY; self.states.len()];
        let mut heap: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();

        for &goal_state_id in &self.goal_state_ids {
            if goal_state_id < distances.len() {
                distances[goal_state_id] = 0.0;
                heap.push((
                    Reverse(NotNan::new(0.0).map_err(|err| err.to_string())?),
                    goal_state_id,
                ));
            }
        }
        if self.truncated {
            for &frontier_state_id in &self.frontier_states {
                if frontier_state_id < distances.len() && distances[frontier_state_id] > 0.0 {
                    distances[frontier_state_id] = 0.0;
                    heap.push((
                        Reverse(NotNan::new(0.0).map_err(|err| err.to_string())?),
                        frontier_state_id,
                    ));
                }
            }
        }

        while let Some((Reverse(distance), state_id)) = heap.pop() {
            let distance = distance.into_inner();
            if distance > distances[state_id] + float_tolerance::DIJKSTRA_EPSILON {
                continue;
            }
            for &(parent_id, projected_operator_id) in &self.transition_predecessors[state_id] {
                let base_operator_id = self
                    .task
                    .base_operator_id(projected_operator_id)
                    .ok_or_else(|| format!("missing base operator id for projected operator {projected_operator_id}"))?;
                let operator_cost = *operator_costs.get(base_operator_id).ok_or_else(|| {
                    format!("missing residual cost for operator {base_operator_id}")
                })?;
                let alternative = distance + operator_cost;
                if alternative + float_tolerance::DIJKSTRA_EPSILON < distances[parent_id] {
                    distances[parent_id] = alternative;
                    heap.push((
                        Reverse(NotNan::new(alternative).map_err(|err| err.to_string())?),
                        parent_id,
                    ));
                }
            }
        }

        Ok(distances)
    }

    /// Builds a PERIM cost-partitioned distance table.
    ///
    /// The perimeter table is used only to derive saturated costs. The returned
    /// lookup table is recomputed globally from the nonnegative saturated costs,
    /// so it remains an admissible table when reused for later states outside
    /// the original perimeter.
    pub fn build_cost_partitioned_distance_table_capped(
        &self,
        operator_costs: &[f64],
        h_cap: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), String> {
        let mut distances = self.build_goal_distances(operator_costs)?;

        if h_cap.is_finite() {
            for h in &mut distances {
                if !h.is_finite() || *h > h_cap {
                    *h = f64::NEG_INFINITY;
                }
            }
        }

        let saturated_costs =
            self.saturated_costs_from_distances(&distances, operator_costs.len())?;

        let mut lookup_costs = saturated_costs.clone();
        for cost in &mut lookup_costs {
            if !cost.is_finite() {
                *cost = 0.0;
            }
        }
        let global_distances = self.build_goal_distances(&lookup_costs)?;

        Ok((global_distances, saturated_costs))
    }

    fn saturated_costs_from_distances(
        &self,
        distances: &[f64],
        num_operators: usize,
    ) -> Result<Vec<f64>, String> {
        let mut saturated_costs = vec![f64::NEG_INFINITY; num_operators];
        for (target_id, predecessors) in self.transition_predecessors.iter().enumerate() {
            let target_h = distances[target_id];
            if !target_h.is_finite() {
                continue;
            }
            for &(parent_id, projected_operator_id) in predecessors {
                let parent_h = distances[parent_id];
                if !parent_h.is_finite() {
                    continue;
                }
                let base_operator_id = self
                    .task
                    .base_operator_id(projected_operator_id)
                    .ok_or_else(|| {
                        format!(
                            "missing base operator id for projected operator {projected_operator_id}"
                        )
                    })?;
                let slot = saturated_costs.get_mut(base_operator_id).ok_or_else(|| {
                    format!(
                        "base operator id {base_operator_id} out of bounds for {num_operators} operators"
                    )
                })?;
                *slot = slot.max((parent_h - target_h).max(0.0));
            }
        }

        for cost in &mut saturated_costs {
            if *cost == f64::NEG_INFINITY {
                *cost = 0.0;
            }
        }

        Ok(saturated_costs)
    }

    pub fn abstract_state_values(
        &self,
        propositional: &[usize],
        numeric: &[f64],
    ) -> Result<(Vec<usize>, Vec<f64>), String> {
        self.task.project_state_values(propositional, numeric)
    }

    pub fn lookup_projected_or_fallback_from_state_values(
        &self,
        propositional: &[usize],
        numeric: &[f64],
    ) -> Result<f64, String> {
        let mut projected_prop = self.projection_prop_scratch.borrow_mut();
        let mut projected_num = self.projection_numeric_scratch.borrow_mut();
        self.task.project_state_values_into(
            propositional,
            numeric,
            &mut projected_prop,
            &mut projected_num,
        )?;

        Ok(self.lookup_pattern_or_fallback_in_projected_values(&projected_prop, &projected_num))
    }

    pub fn lookup_projected_or_fallback_from_source_state_values(
        &self,
        propositional: &[usize],
        source_numeric: &[f64],
    ) -> Result<f64, String> {
        if let Some(distance) =
            self.lookup_pattern_distance_from_source_state_values(propositional, source_numeric)?
            && distance.is_finite()
        {
            return Ok(distance);
        }

        let mut projected_prop = self.projection_prop_scratch.borrow_mut();
        let mut projected_num = self.projection_numeric_scratch.borrow_mut();

        self.lookup_projection
            .project_state_values_from_source_numeric_into(
                propositional,
                source_numeric,
                &mut projected_prop,
                &mut projected_num,
            )?;

        Ok(self.lookup_pattern_or_fallback_in_projected_values(&projected_prop, &projected_num))
    }

    pub(crate) fn lookup_projected_or_fallback_from_source_state_values_fast(
        &self,
        propositional: &[usize],
        source_numeric: &[f64],
    ) -> f64 {
        if let Some(distance) = self
            .lookup_pattern_distance_from_source_state_values_fast(propositional, source_numeric)
            && distance.is_finite()
        {
            return distance;
        }

        let mut projected_prop = self.projection_prop_scratch.borrow_mut();
        let mut projected_num = self.projection_numeric_scratch.borrow_mut();

        self.lookup_projection
            .project_state_values_from_source_numeric_into_unchecked(
                propositional,
                source_numeric,
                &mut projected_prop,
                &mut projected_num,
            );

        self.lookup_pattern_or_fallback_in_projected_values(&projected_prop, &projected_num)
    }

    pub fn lookup_or_fallback_from_concrete_state(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
    ) -> Result<f64, String> {
        if let Some(distance) = self.lookup_pattern_distance_from_concrete_state(state, registry)?
            && distance.is_finite()
        {
            return Ok(distance);
        }

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
        Ok(self.lookup_pattern_or_fallback_in_projected_values(&projected_prop, &projected_num))
    }

    pub(super) fn state_propositional_values<'state>(
        &self,
        state: &'state PdbState,
    ) -> &'state [usize] {
        &state.propositional
    }

    pub(super) fn state_numeric_values<'state>(&self, state: &'state PdbState) -> &'state [f64] {
        &state.numeric
    }

    fn rebuild_lookup_indexes(&mut self) {
        self.state_index.clear();
        self.pattern_index.clear();
        self.packed_pattern_registry.clear();
        self.compact_numeric_registry.clear();
        self.full_prop_index.clear();
        self.compact_prop_distances.clear();
        self.failed_lookup_cache.borrow_mut().clear();

        let pattern_regular_ids = self.task.pattern_regular_projected_ids();
        let pattern_numeric_ids = self.task.pattern_numeric_projected_ids();
        let mut packed_bins = vec![0; self.pattern_lookup_packer.num_bins()];
        let mut compact_numeric_bins = vec![0; 1 + pattern_numeric_ids.len()];

        let compact_prop_table_len = if pattern_numeric_ids.is_empty()
            && !self.compact_prop_hash_multipliers.is_empty()
        {
            let last_var_id = *pattern_regular_ids.last().unwrap_or(&0);
            self.compact_prop_hash_multipliers
                .last()
                .copied()
                .and_then(|last_multiplier| {
                    self.task
                        .variables()
                        .get(last_var_id)
                        .map(|var| last_multiplier.saturating_mul(var.domain_size()))
                })
                .filter(|&len| len > 0 && len <= self.states.len().saturating_mul(8).max(1024))
        } else if pattern_numeric_ids.is_empty() && self.compact_prop_hash_multipliers.is_empty() {
            Some(1)
        } else {
            None
        };

        if let Some(table_len) = compact_prop_table_len {
            self.compact_prop_distances.resize(table_len, f64::INFINITY);
        }

        for (state_id, state) in self.states.iter().enumerate() {
            let full_key = hash_state_components(&state.propositional, &state.numeric);
            self.state_index
                .entry(full_key)
                .or_insert_with(|| Vec::with_capacity(1))
                .push(state_id);

            if let Some(prop_hash) =
                compute_prop_hash(&state.propositional, &self.full_prop_hash_multipliers)
            {
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

            let compact_prop_hash = compute_projected_compact_prop_hash(
                &state.propositional,
                pattern_regular_ids,
                &self.compact_prop_hash_multipliers,
            );

            if !self.compact_prop_distances.is_empty()
                && let Some(compact_prop_hash) = compact_prop_hash
                && let Some(slot) = self.compact_prop_distances.get_mut(compact_prop_hash)
            {
                *slot = slot.min(self.distances[state_id]);
            }

            if !pattern_numeric_ids.is_empty()
                && let Some(compact_prop_hash) = compact_prop_hash
            {
                compact_numeric_bins.fill(0);
                compact_numeric_bins[0] = compact_prop_hash as u64;
                for (numeric_index, &projected_numeric_id) in pattern_numeric_ids.iter().enumerate()
                {
                    compact_numeric_bins[numeric_index + 1] =
                        float_tolerance::canonical_bits(state.numeric[projected_numeric_id]);
                }
                let distance = self.distances[state_id];
                self.compact_numeric_registry
                    .insert_min_distance(&compact_numeric_bins, distance);
            }

            packed_bins.fill(0);
            for (compact_index, &projected_var_id) in pattern_regular_ids.iter().enumerate() {
                self.pattern_lookup_packer.set(
                    &mut packed_bins,
                    compact_index,
                    state.propositional[projected_var_id] as u64,
                );
            }
            let prop_len = pattern_regular_ids.len();
            for (numeric_index, &projected_numeric_id) in pattern_numeric_ids.iter().enumerate() {
                self.pattern_lookup_packer.set(
                    &mut packed_bins,
                    prop_len + numeric_index,
                    self.pattern_lookup_packer
                        .pack_double(state.numeric[projected_numeric_id]),
                );
            }
            let distance = self.distances[state_id];
            self.packed_pattern_registry
                .insert_min_distance(&packed_bins, distance);
        }
    }

    fn build(&mut self, max_states: usize) -> Result<(), String> {
        let table = self.build_distance_table(max_states)?;
        self.states = table.states;
        self.distances = table.distances;
        self.goal_state_ids = table.goal_state_ids;
        self.transition_predecessors = table.transition_predecessors;
        self.reached_goal_states = table.reached_goal_states;
        self.frontier_states = table.frontier_states;
        self.truncated = table.truncated;
        self.exhausted_abstract_state_space = table.exhausted_abstract_state_space;
        self.rebuild_lookup_indexes();
        Ok(())
    }

    /// Explore the projected task's abstract state space and compute the goal
    /// distance of every state reached.
    fn build_distance_table(&self, max_states: usize) -> Result<PdbDistanceTable, String> {
        let mut registry = StateRegistry::for_task(Arc::new(&self.task));
        let mut inner_heuristic = PdbInnerHeuristic::new(&self.task, self.uses_lmcut());
        let mut exploration =
            self.explore_abstract_state_space(max_states, &mut registry, &mut inner_heuristic)?;
        let exhausted_abstract_state_space = exploration.open.is_empty();
        let states = collect_pdb_states(&exploration.representative_states, &registry)?;

        // Backward Dijkstra from the goal states over the recorded
        // predecessors, seeded with the goal states at distance zero.
        let mut distances = vec![f64::INFINITY; states.len()];
        let mut heap: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();
        for &goal_state_id in &exploration.goal_states {
            distances[goal_state_id] = 0.0;
            heap.push((Reverse(NotNan::new(0.0).unwrap()), goal_state_id));
        }
        let frontier_states = if exploration.truncated {
            self.seed_frontier_distances(
                &mut exploration,
                &states,
                &registry,
                &mut inner_heuristic,
                &mut distances,
                &mut heap,
            )?
        } else {
            Vec::new()
        };
        self.propagate_distances_to_predecessors(
            &exploration.predecessors,
            &mut distances,
            &mut heap,
        );

        let reached_goal_states = exploration.goal_states.len();
        Ok(PdbDistanceTable {
            states,
            distances,
            reached_goal_states,
            goal_state_ids: exploration.goal_states,
            frontier_states,
            transition_predecessors: exploration.predecessors,
            truncated: exploration.truncated,
            exhausted_abstract_state_space,
        })
    }

    fn uses_lmcut(&self) -> bool {
        matches!(
            self.heuristic_config.exploration_heuristic,
            PdbInternalHeuristic::Lmcut
        ) || matches!(
            self.heuristic_config.frontier_heuristic,
            PdbInternalHeuristic::Lmcut
        )
    }

    /// Best-first exploration of the abstract state space, recording every
    /// transition as a predecessor edge so distances can be propagated
    /// backwards afterwards.
    ///
    /// Stops at `max_states` newly seen states; the states still queued then
    /// are the frontier, and `truncated` says the table only bounds distances
    /// rather than giving them exactly.
    fn explore_abstract_state_space(
        &self,
        max_states: usize,
        registry: &mut StateRegistry<'_>,
        inner_heuristic: &mut PdbInnerHeuristic<'_>,
    ) -> Result<PdbExploration, String> {
        let successor_generator = GroundedSuccessorGenerator::construct_node_from_task(&self.task);
        let mut applicable_operators: Vec<u32> = Vec::new();
        let mut current_propositional: Vec<usize> = Vec::new();
        let mut successor_numeric: Vec<f64> = Vec::new();
        let mut successor_cost_values: Vec<f64> = Vec::new();
        let mut expansion_context = ExpansionContext::default();

        let mut exploration = PdbExploration {
            representative_states: vec![registry.get_initial_state()],
            predecessors: Vec::with_capacity(max_states),
            goal_states: Vec::new(),
            closed: vec![false],
            seen_or_closed: vec![true],
            open: BinaryHeap::new(),
            truncated: false,
        };
        exploration.predecessors.push(Vec::new());
        exploration.open.push(PdbOpenEntry {
            state_id: 0,
            f: NotNan::new(0.0).unwrap(),
            g: NotNan::new(0.0).unwrap(),
        });
        let mut seen_count = 0usize;

        loop {
            if seen_count >= max_states {
                exploration.truncated = true;
                break;
            }
            let Some(entry) = exploration.open.pop() else {
                break;
            };
            let state_id = entry.state_id;
            if exploration.representative_states.len().is_multiple_of(500) {
                info!(
                    "Expanding state {}/{} ({} reached goal states, {} truncated frontier states)",
                    state_id + 1,
                    exploration.representative_states.len(),
                    exploration.goal_states.len(),
                    0
                );
            }
            if exploration.is_closed(state_id) {
                continue;
            }
            exploration.mark_closed(state_id);

            applicable_operators.clear();
            let current_registry_state = exploration.representative_states[state_id].clone();
            current_registry_state.fill_state(registry, &mut current_propositional);
            if self.is_goal_state(&current_propositional) {
                exploration.goal_states.push(state_id);
            }
            successor_generator
                .get_applicable_operators(&current_propositional, &mut applicable_operators);
            registry
                .build_expansion_context(&current_registry_state, &mut expansion_context)
                .map_err(|err| err.message)?;

            let operators = self.task.get_operators();
            for &op_id in applicable_operators.iter() {
                let operator_id = op_id as usize;
                let (successor_state, _) = registry
                    .apply_operator_in_context(
                        &current_registry_state,
                        &operators[operator_id],
                        &expansion_context,
                        &mut successor_numeric,
                        &mut successor_cost_values,
                    )
                    .map_err(|err| err.message)?;
                if successor_state.get_id() == current_registry_state.get_id() {
                    continue;
                }
                let next_id = successor_state.get_id();
                exploration.register_state(next_id, successor_state)?;
                exploration.predecessors[next_id].push((state_id, operator_id));

                if exploration.seen_or_closed[next_id] {
                    continue;
                }
                exploration.seen_or_closed[next_id] = true;
                seen_count += 1;
                let inner_h = inner_heuristic.evaluate(
                    self.heuristic_config.exploration_heuristic,
                    next_id,
                    &exploration.representative_states[next_id],
                    registry,
                )?;
                if inner_h.dead_end {
                    continue;
                }
                let g = entry.g.into_inner() + self.task.abstract_operator_cost(operator_id);
                // Blind and Zero explore breadth-first: their value is a
                // constant, so folding it into `f` would only shift every key.
                let h = if matches!(
                    self.heuristic_config.exploration_heuristic,
                    PdbInternalHeuristic::Blind | PdbInternalHeuristic::Zero
                ) {
                    0.0
                } else {
                    inner_h.value
                };
                exploration.open.push(PdbOpenEntry {
                    state_id: next_id,
                    f: NotNan::new(g + h).map_err(|err| err.to_string())?,
                    g: NotNan::new(g).map_err(|err| err.to_string())?,
                });
            }
        }
        Ok(exploration)
    }

    /// Seed the backward search with the states the exploration left queued.
    ///
    /// A truncated table only knows a lower bound for a frontier state, so it
    /// takes the frontier heuristic — floored at the cheapest operator cost,
    /// since a non-goal state always needs at least one more operator.
    /// Returns the frontier state ids, ascending and deduplicated.
    fn seed_frontier_distances(
        &self,
        exploration: &mut PdbExploration,
        states: &[PdbState],
        registry: &StateRegistry<'_>,
        inner_heuristic: &mut PdbInnerHeuristic<'_>,
        distances: &mut [f64],
        heap: &mut BinaryHeap<(Reverse<NotNan<f64>>, usize)>,
    ) -> Result<Vec<usize>, String> {
        let mut frontier_states: Vec<usize> = Vec::new();
        let mut seen_frontier = vec![false; states.len()];
        while let Some(entry) = exploration.open.pop() {
            let state_id = entry.state_id;
            if exploration.is_closed(state_id) {
                continue;
            }
            if state_id >= seen_frontier.len() || seen_frontier[state_id] {
                continue;
            }
            seen_frontier[state_id] = true;
            frontier_states.push(state_id);

            let seed_cost = if self.is_goal_state(&states[state_id].propositional) {
                0.0
            } else {
                let inner_h = inner_heuristic.evaluate(
                    self.heuristic_config.frontier_heuristic,
                    state_id,
                    &exploration.representative_states[state_id],
                    registry,
                )?;
                if inner_h.dead_end {
                    continue;
                }
                inner_h.value.max(self.min_operator_cost())
            };
            if seed_cost + float_tolerance::DIJKSTRA_EPSILON < distances[state_id] {
                distances[state_id] = seed_cost;
                heap.push((Reverse(NotNan::new(seed_cost).unwrap()), state_id));
            }
        }
        frontier_states.sort_unstable();
        frontier_states.dedup();
        Ok(frontier_states)
    }

    /// Backward Dijkstra over the recorded predecessor edges.
    fn propagate_distances_to_predecessors(
        &self,
        predecessors: &[Vec<(usize, usize)>],
        distances: &mut [f64],
        heap: &mut BinaryHeap<(Reverse<NotNan<f64>>, usize)>,
    ) {
        while let Some((Reverse(distance), state_id)) = heap.pop() {
            let distance = distance.into_inner();
            if distance > distances[state_id] + float_tolerance::DIJKSTRA_EPSILON {
                continue;
            }
            for &(parent_id, operator_id) in &predecessors[state_id] {
                let alternative = distance + self.task.abstract_operator_cost(operator_id);
                if alternative + float_tolerance::DIJKSTRA_EPSILON < distances[parent_id] {
                    distances[parent_id] = alternative;
                    heap.push((Reverse(NotNan::new(alternative).unwrap()), parent_id));
                }
            }
        }
    }
}

/// Everything `PatternDatabase::build` writes back onto itself.
struct PdbDistanceTable {
    states: Vec<PdbState>,
    distances: Vec<f64>,
    reached_goal_states: usize,
    goal_state_ids: Vec<usize>,
    frontier_states: Vec<usize>,
    transition_predecessors: Vec<Vec<(usize, usize)>>,
    truncated: bool,
    exhausted_abstract_state_space: bool,
}

/// The abstract state space the exploration reached.
struct PdbExploration {
    /// One concrete state per abstract state, indexed by abstract state id.
    representative_states: Vec<ConcreteState>,
    /// `predecessors[s]` are the `(parent, operator)` edges entering `s`.
    predecessors: Vec<Vec<(usize, usize)>>,
    goal_states: Vec<usize>,
    closed: Vec<bool>,
    seen_or_closed: Vec<bool>,
    /// States generated but not expanded — the frontier, when truncated.
    open: BinaryHeap<PdbOpenEntry>,
    /// The exploration hit its state limit, so distances are bounds.
    truncated: bool,
}

impl PdbExploration {
    fn is_closed(&self, state_id: usize) -> bool {
        self.closed.get(state_id).copied().unwrap_or(false)
    }

    fn mark_closed(&mut self, state_id: usize) {
        if state_id >= self.closed.len() {
            self.closed.resize(state_id + 1, false);
        }
        self.closed[state_id] = true;
    }

    /// Give a newly reached abstract state its slot. The state registry hands
    /// out ids densely, so a gap means the registry and this table disagree
    /// about which states exist.
    fn register_state(&mut self, state_id: usize, state: ConcreteState) -> Result<(), String> {
        if state_id < self.representative_states.len() {
            return Ok(());
        }
        if state_id != self.representative_states.len() {
            return Err(format!(
                "state registry produced non-contiguous abstract state id {state_id} while {} states are represented",
                self.representative_states.len()
            ));
        }
        self.representative_states.push(state);
        self.predecessors.push(Vec::new());
        if state_id >= self.closed.len() {
            self.closed.resize(state_id + 1, false);
        }
        if state_id >= self.seen_or_closed.len() {
            self.seen_or_closed.resize(state_id + 1, false);
        }
        Ok(())
    }
}

/// The inner heuristic PDB construction consults, memoised per abstract state.
///
/// `Blind` and `Zero` are constants and need no memory; only LM-cut is worth
/// caching, and it is the only one that has to be built at all.
struct PdbInnerHeuristic<'task> {
    lmcut: Option<LmcutInnerHeuristic<'task>>,
    cache: Vec<Option<InnerHeuristicResult>>,
}

impl<'task> PdbInnerHeuristic<'task> {
    fn new(task: &'task dyn AbstractNumericTask, uses_lmcut: bool) -> Self {
        Self {
            lmcut: uses_lmcut.then(|| LmcutInnerHeuristic::new(task)),
            cache: Vec::new(),
        }
    }

    fn evaluate(
        &mut self,
        heuristic: PdbInternalHeuristic,
        state_id: usize,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
    ) -> Result<InnerHeuristicResult, String> {
        match heuristic {
            PdbInternalHeuristic::Zero | PdbInternalHeuristic::Blind => Ok(InnerHeuristicResult {
                dead_end: false,
                value: 0.0,
            }),
            PdbInternalHeuristic::Lmcut => {
                if self.cache.len() <= state_id {
                    self.cache.resize(state_id + 1, None);
                }
                if let Some(result) = self.cache[state_id] {
                    return Ok(result);
                }
                let result = self
                    .lmcut
                    .as_mut()
                    .expect("LM-cut inner heuristic must be initialized when configured")
                    .evaluate_from_concrete_state(state, registry)?;
                self.cache[state_id] = Some(result);
                Ok(result)
            }
        }
    }
}

/// Read every abstract state's propositional and numeric values out of the
/// registry.
fn collect_pdb_states(
    representative_states: &[ConcreteState],
    registry: &StateRegistry<'_>,
) -> Result<Vec<PdbState>, String> {
    representative_states
        .iter()
        .map(|state| {
            Ok(PdbState {
                propositional: state.get_state(registry),
                numeric: registry
                    .get_numeric_vars(state)
                    .map_err(|err| format!("{err:?}"))?,
            })
        })
        .collect()
}
