//! State Registry for Numeric Planning.
//!
//! This module provides the `StateRegistry` which is responsible for managing
//! planning states in a numeric planning context. It handles:
//!
//! - State creation and deduplication.
//! - Efficient state representation using bit packing.
//! - Numeric variable management (regular, constant, cost, derived).
//! - Axiom evaluation for derived predicates and variables.
//! - Successor state generation.
//!
//! # Key Components:
//!
//! - `ConcreteState`: Represents a concrete planning state.
//! - `StateRegistry`: Central registry for state management.
//! - Efficient storage using segmented vectors and bit packing.
//! - Integration with axiom evaluation system.

#[cfg(test)]
mod tests;

use crate::axioms::AxiomEvaluator;
use crate::numeric_task::{AssignmentOperation, Operator, TaskRef};
use crate::utils::errors::{InvalidIndex, StateInsertError, StateNotFoundError};
use crate::utils::float_tolerance;
use crate::utils::per_state_info::PerStateInformation;
use crate::utils::segmented_vector2::SegmentedArrayVector;
use crate::{numeric_task::NumericType, utils::int_packer::IntDoublePacker};
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Pass-through hasher for `u64` keys that are *already* hashes
/// (e.g. `fast_hash_bins` outputs). Avoids re-hashing the key with `SipHash`.
#[derive(Default)]
pub struct IdentityU64Hasher(u64);

impl Hasher for IdentityU64Hasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // The map only ever hashes a single `u64`, so we expect exactly 8 bytes.
        // We still handle the general case defensively in case the API is reused.
        if bytes.len() == 8 {
            self.0 = u64::from_ne_bytes(bytes.try_into().unwrap());
        } else {
            for &b in bytes {
                self.0 = self.0.rotate_left(5) ^ u64::from(b);
            }
        }
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.0 = value as u64;
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

type IdentityHasherBuilder = BuildHasherDefault<IdentityU64Hasher>;
type RegisteredStatesMap = HashMap<u64, Vec<StateID>, IdentityHasherBuilder>;

/// Type alias for the state packer used throughout the system.
type StatePacker = IntDoublePacker;

/// Type alias for state identifiers.
pub type StateID = usize;

/// Type alias for the underlying data storage.
type DataStorage = SegmentedArrayVector<u64>;

/// Represent a concrete state in the planning problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcreteState {
    pool_offset: usize,
}

impl ConcreteState {
    /// Create a new concrete state with the given pool offset.
    pub const fn new(pool_offset: usize) -> Self {
        Self { pool_offset }
    }

    /// Get the state ID (equivalent to C++ `GlobalState::get_id()`).
    /// This is the index into the state registry's data pool.
    pub fn get_id(&self) -> usize {
        self.pool_offset
    }

    /// Get the propositional state values as a vector.
    pub fn get_state(&self, state_registry: &StateRegistry) -> Vec<usize> {
        let mut values = Vec::with_capacity(state_registry.task.variables().len());
        self.fill_state(state_registry, &mut values);
        values
    }

    /// Fill `output` with the propositional state values without allocating a new vector.
    pub fn fill_state(&self, state_registry: &StateRegistry, output: &mut Vec<usize>) {
        let buffer = state_registry.get_buffer(self.pool_offset);
        let task = &state_registry.task;
        let state_packer = &state_registry.global_state_packer;

        output.resize(task.variables().len(), 0);
        output
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = state_packer.get(buffer, i) as usize);
    }

    pub fn get_propositional_value(
        &self,
        state_registry: &StateRegistry,
        var_id: usize,
    ) -> Result<usize, InvalidIndex> {
        if var_id >= state_registry.task.variables().len() {
            return Err(InvalidIndex {
                index: var_id,
                length: state_registry.task.variables().len(),
            });
        }

        let buffer = state_registry.get_buffer(self.pool_offset);
        Ok(state_registry.global_state_packer.get(buffer, var_id) as usize)
    }

    /// Get the numeric state values for regular variables.
    pub fn get_numeric_state(&self, state_registry: &StateRegistry) -> Vec<f64> {
        let buffer = state_registry.get_buffer(self.pool_offset);
        let task = &state_registry.task;
        let state_packer = &state_registry.global_state_packer;

        task.numeric_variables()
            .iter()
            .enumerate()
            .filter_map(|(i, var)| {
                if var.get_type() == &NumericType::Regular {
                    Some(state_packer.get_double(buffer, i))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get a reference to the underlying buffer for this state.
    pub fn buffer<'a>(&self, state_registry: &'a StateRegistry) -> &'a [u64] {
        state_registry.get_buffer(self.pool_offset)
    }

    /// Return the number of propositional variables in this state.
    pub fn len(&self, state_registry: &StateRegistry) -> usize {
        state_registry.task.variables().len()
    }

    /// Return `true` if the state has no variables (should never happen in practice).
    pub fn is_empty(&self, state_registry: &StateRegistry) -> bool {
        self.len(state_registry) == 0
    }

    /// Create a debug representation of this state with variable values.
    pub fn debug_with_registry(&self, registry: &StateRegistry) -> String {
        let task = &registry.task;
        let num_variables = task.variables().len();
        let num_regular_numeric_vars = task
            .numeric_variables()
            .iter()
            .filter(|v| v.get_type() == &NumericType::Regular)
            .count();

        let buffer = self.buffer(registry);
        let state_packer = &registry.global_state_packer;

        let mut result = format!("ConcreteState with {} bins\n", buffer.len());

        // Add propositional variables.
        for i in 0..num_variables {
            let value = state_packer.get(buffer, i);
            result.push_str(&format!("Var {}: {}\n", i, value));
        }

        // Add numeric variables.
        for i in 0..num_regular_numeric_vars {
            let numeric_var_id = i + num_variables;
            let packed_value = state_packer.get(buffer, numeric_var_id);
            let numeric_value = state_packer.unpack_double(packed_value);
            result.push_str(&format!(
                "Numeric Var {}: {}\n",
                numeric_var_id, numeric_value
            ));
        }

        result
    }
}

// No external-key comparator like C++ `unordered_set` functors in Rust.
// We use a bucketed map from content hash -> list of `StateID`s, then compare
// only within the bucket using the packed bins for semantic equality.

/// SplitMix64 finalizer. Spreads bits well across both halves of the output
/// so the result is suitable as a key for hashbrown (which uses both the
/// low bits for the bucket index and the top 7 bits for the SIMD tag).
#[inline]
fn finalize_mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

#[inline]
fn fast_hash_bins(bins: &[u64]) -> u64 {
    // 64-bit `FNV-1a`. Hashes byte-by-byte rather than u64-by-u64. Earlier
    // experiments showed that the u64-chunk variant produces noticeably more
    // hashbrown bucket collisions on packed planning state buffers (where
    // many bins share large stretches of zero/sparse bits), and `memcmp`
    // dominates the dedup path when buckets grow. The 8x extra
    // multiplications per bin are cheap by comparison.
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;
    let mut hash = FNV_OFFSET;
    for &x in bins {
        let bytes = x.to_le_bytes();
        for b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    finalize_mix(hash)
}

/// Hash only the bits that the per-bin `mask` selects, byte-by-byte to match
/// `fast_hash_bins`'s distribution properties. Used to dedup buffers where
/// some bins also contain derived (axiom-computed) bits we want to ignore.
#[inline]
fn fast_hash_bins_masked(bins: &[u64], mask: &[u64]) -> u64 {
    debug_assert_eq!(bins.len(), mask.len());
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;
    let mut hash = FNV_OFFSET;
    for (&x, &m) in bins.iter().zip(mask.iter()) {
        let masked = x & m;
        let bytes = masked.to_le_bytes();
        for b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    finalize_mix(hash)
}

/// Compare two buffers, considering only the bits selected by `mask`.
#[inline]
fn bins_eq_masked(left: &[u64], right: &[u64], mask: &[u64]) -> bool {
    debug_assert_eq!(left.len(), right.len());
    debug_assert_eq!(left.len(), mask.len());
    for ((&l, &r), &m) in left.iter().zip(right.iter()).zip(mask.iter()) {
        if (l & m) != (r & m) {
            return false;
        }
    }
    true
}

/// Reusable scratch holding the parent state's per-expansion data. Filled
/// once per expansion via `StateRegistry::build_expansion_context`, then
/// shared across every successor produced by that expansion. This avoids
/// re-reading the same parent on every operator application.
#[derive(Debug, Default, Clone)]
pub struct ExpansionContext {
    pub parent_numeric: Vec<f64>,
    pub parent_cost: Vec<f64>,
    pub parent_metric: f64,
}

/// Static counter for generating unique registry IDs.
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// Central registry for managing planning states with deduplication and axiom evaluation.
///
/// The `StateRegistry` is responsible for:
/// - Creating and managing planning states.
/// - Deduplicating identical states to save memory.
/// - Evaluating axioms when states change.
/// - Managing numeric variables and cost information.
pub struct StateRegistry<'a> {
    /// Unique identifier for this registry instance.
    id: usize,
    /// Shared handle to the planning task.
    task: TaskRef<'a>,
    /// Axiom evaluator for handling derived predicates and numeric axioms.
    axiom_evaluator: Arc<AxiomEvaluator<'a>>,
    /// State packer for efficient bit-level state representation.
    global_state_packer: Arc<StatePacker>,
    /// Pool of state data, each entry is a packed state representation.
    state_data_pool: DataStorage,
    /// Constants for numeric variables.
    numeric_constants: Vec<f64>,
    /// Mapping from numeric variable index to packed state index.
    numeric_indices: Vec<Option<usize>>,
    /// Buckets of registered states for duplicate detection (hash -> `Vec<StateID>`).
    /// Uses an identity hasher because keys are already 64-bit content hashes.
    registered_states: RegisteredStatesMap,
    /// Per-state cost information storage.
    cost_info: RefCell<PerStateInformation<Vec<f64>>>,
    /// Snapshot of the numeric variable layout, populated at construction.
    /// Iterating this avoids per-call vtable dispatch through `task.numeric_variables()`
    /// in hot paths like `fill_numeric_vars`.
    numeric_var_types: Vec<NumericType>,
    /// Cached count returned by `count_cost_variables`. Constant for the task.
    cost_variable_count: usize,
    /// Cached `task.metric().var_id()` and the metric variable's type when set.
    /// Used to short-circuit `metric_value_for_state` for the common Regular case.
    metric_var: Option<(usize, NumericType)>,
    /// Whether the task uses a metric (cached `task.metric().use_metric()`).
    metric_use_metric: bool,
    /// Whether the task is a minimization (cached `task.metric().is_min()`).
    metric_is_min: bool,
    /// Per-state metric value cache, indexed by state id. Populated as states
    /// are registered. Used to bypass the per-state cost-info `HashMap` in the
    /// hot duplicate-handling path of `apply_operator_in_context`.
    metric_value_by_state: RefCell<Vec<f64>>,
    /// Per-bin mask covering only the bits owned by non-derived (input)
    /// variables: regular propositional vars (axiom_layer == None) and regular
    /// numeric vars. Used by `insert_id_or_pop_state_masked` to dedup
    /// successors before running the comparison/propositional axiom passes.
    /// Two states with identical non-derived bits are guaranteed to produce
    /// identical full buffers because axioms are deterministic functions of
    /// the inputs, so masked equality matches full equality.
    non_derived_bits_mask: Vec<u64>,
    /// True iff the task has at least one comparison or propositional axiom
    /// AND `non_derived_bits_mask` actually masks anything off. When true, the
    /// successor flow defers comparison/propositional axiom evaluation until
    /// after dedup so we can skip it entirely on duplicate states.
    has_axiom_derived_bits: bool,
}

impl<'a> StateRegistry<'a> {
    /// Build the state packer, axiom evaluator, and registry for `task` in
    /// one step. This is the common construction path; use [`Self::new`]
    /// when a custom packer or axiom evaluator is needed.
    pub fn for_task(task: TaskRef<'a>) -> Self {
        let packer = Arc::new(StatePacker::from_abstract_task(&*task));
        let axiom_evaluator = Arc::new(AxiomEvaluator::new(task.clone(), packer.clone()));
        Self::new(task, packer, axiom_evaluator)
    }

    /// Create a new state registry for the given planning task.
    pub fn new(
        task: TaskRef<'a>,
        global_state_packer: Arc<StatePacker>,
        axiom_evaluator: Arc<AxiomEvaluator<'a>>,
    ) -> Self {
        let numeric_vars = task.numeric_variables();
        let number_numeric_vars = numeric_vars.len();
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);

        // Create cost info and subscribe it to this registry
        let mut cost_info = PerStateInformation::new();
        cost_info.subscribe(id);

        let numeric_var_types: Vec<NumericType> =
            numeric_vars.iter().map(|var| *var.get_type()).collect();
        let cost_variable_count = numeric_var_types
            .iter()
            .filter(|&&ty| ty == NumericType::Cost)
            .count();
        let metric_use_metric = task.metric().use_metric();
        let metric_var = task.metric().var_id().and_then(|var_id| {
            numeric_var_types
                .get(var_id)
                .copied()
                .map(|ty| (var_id, ty))
        });

        // Collect packer var ids for non-derived (input) variables.
        // Propositional vars are 0..num_prop_vars; numeric Regular vars start
        // at num_prop_vars in the packer's variable index space.
        let prop_vars = task.variables();
        let mut non_derived_var_ids: Vec<usize> = Vec::new();
        let mut has_propositional_derived = false;
        for (i, _) in prop_vars.iter().enumerate() {
            match task.get_variable_axiom_layer(i) {
                Ok(None) => non_derived_var_ids.push(i),
                Ok(Some(_)) => {
                    has_propositional_derived = true;
                }
                Err(_) => non_derived_var_ids.push(i),
            }
        }
        let num_prop_vars = prop_vars.len();
        let mut numeric_packer_index = num_prop_vars;
        for &ty in &numeric_var_types {
            if ty == NumericType::Regular {
                non_derived_var_ids.push(numeric_packer_index);
                numeric_packer_index += 1;
            }
        }
        let non_derived_bits_mask = global_state_packer.build_var_subset_mask(&non_derived_var_ids);
        let mask_covers_any_bits = non_derived_bits_mask.iter().any(|&m| m != 0);
        let has_axiom_derived_bits = has_propositional_derived && mask_covers_any_bits;
        if has_propositional_derived && !mask_covers_any_bits {
            tracing::warn!(
                "state_registry: skipping masked dedup because non-derived bit mask is empty (every variable is axiom-derived)"
            );
        }

        let metric_is_min = task.metric().is_min();
        let state_data_pool = DataStorage::new(global_state_packer.num_bins());
        Self {
            id,
            task,
            global_state_packer,
            state_data_pool,
            numeric_constants: Vec::new(),
            numeric_indices: vec![None; number_numeric_vars],
            registered_states: RegisteredStatesMap::with_capacity_and_hasher(
                1024,
                IdentityHasherBuilder::default(),
            ),
            axiom_evaluator,
            cost_info: RefCell::new(cost_info),
            numeric_var_types,
            cost_variable_count,
            metric_var,
            metric_use_metric,
            metric_is_min,
            metric_value_by_state: RefCell::new(Vec::new()),
            non_derived_bits_mask,
            has_axiom_derived_bits,
        }
    }

    /// Return the unique ID of this registry.
    pub const fn id(&self) -> usize {
        self.id
    }

    pub fn get_state_data_pool(&self) -> &DataStorage {
        &self.state_data_pool
    }

    pub fn get_numeric_indices(&self) -> &[Option<usize>] {
        &self.numeric_indices
    }

    /// Return the value of a numeric constant variable, if available.
    ///
    /// Constant values are initialized when the initial state is created.
    pub fn get_numeric_constant_value(&self, numeric_var_id: usize) -> Option<f64> {
        if numeric_var_id >= self.task.numeric_variables().len() {
            return None;
        }
        if self.task.numeric_variables()[numeric_var_id].get_type() != &NumericType::Constant {
            return None;
        }

        let constant_index = *self.numeric_indices.get(numeric_var_id)?;
        if let Some(index) = constant_index {
            self.numeric_constants.get(index).copied()
        } else {
            None
        }
    }

    pub fn get_registered_states(&self) -> &RegisteredStatesMap {
        &self.registered_states
    }

    /// Return the total number of distinct states registered in this registry.
    pub fn num_registered_states(&self) -> usize {
        self.registered_states
            .values()
            .map(|bucket| bucket.len())
            .sum()
    }

    pub fn get_cost_info(&self) -> &RefCell<PerStateInformation<Vec<f64>>> {
        &self.cost_info
    }

    /// Subscribe to another registry (placeholder for future functionality).
    pub fn subscribe(&mut self, _registry: &StateRegistry) {
        todo!("Registry subscription not yet implemented")
    }

    /// Subscribe a `PerStateInformation` instance to this registry.
    ///
    /// When this registry is dropped, it will notify the subscribed `PerStateInformation`
    /// instances to clean up their data. This follows the C++ Fast Downward pattern
    /// where `PerStateInformation` instances register themselves with
    /// `StateRegistry` instances.
    ///
    /// Note: In Rust, we can't hold mutable references to the `PerStateInformation`,
    /// so this method just ensures the `PerStateInformation` knows about this registry.
    pub fn subscribe_per_state_info<T>(&self, per_state_info: &mut PerStateInformation<T>)
    where
        T: Clone + Default,
    {
        per_state_info.subscribe(self.id);
    }

    /// Unsubscribes a PerStateInformation instance from this registry.
    pub fn unsubscribe_per_state_info<T>(&self, per_state_info: &mut PerStateInformation<T>)
    where
        T: Clone + Default,
    {
        per_state_info.unsubscribe(self.id);
    }

    /// Get the buffer at the specified index.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    pub fn get_buffer(&self, index: usize) -> &[u64] {
        self.state_data_pool
            .get(index)
            .expect("State index out of bounds")
    }

    fn get_buffer_mut(&mut self, index: usize) -> &mut [u64] {
        self.state_data_pool
            .get_mut(index)
            .expect("State index out of bounds")
    }

    /// Return a reference to the global state packer.
    pub fn global_state_packer(&self) -> &StatePacker {
        &self.global_state_packer
    }

    fn num_state_bins(&self) -> usize {
        self.global_state_packer.num_bins()
    }

    fn find_registered_state_id(&self, key: u64, bins: &[u64]) -> Option<StateID> {
        let num_bins = self.num_state_bins();
        self.registered_states.get(&key).and_then(|bucket| {
            bucket
                .iter()
                .copied()
                .find(|&existing_id| self.get_buffer(existing_id)[..num_bins] == bins[..num_bins])
        })
    }

    fn insert_id_or_pop_state(&mut self) -> (StateID, bool) {
        // When the task has axiom-derived bits, route through the masked path
        // so that all registrations live in a single map and stay consistent
        // with `insert_id_or_pop_state_masked` (used by the successor flow
        // when axioms are deferred).
        if self.has_axiom_derived_bits {
            return self.insert_id_or_pop_state_masked();
        }

        let state_id = self.state_data_pool.len() - 1;
        let key = {
            let state_data = self.get_buffer(state_id);
            fast_hash_bins(&state_data[..self.num_state_bins()])
        };

        let existing_id = {
            let state_data = self.get_buffer(state_id);
            self.find_registered_state_id(key, state_data)
        };

        if let Some(existing_id) = existing_id {
            self.state_data_pool.pop_back();
            return (existing_id, false);
        }

        self.registered_states
            .entry(key)
            .or_insert_with(|| Vec::with_capacity(4))
            .push(state_id);
        (state_id, true)
    }

    /// Variant of `insert_id_or_pop_state` that hashes and compares only the
    /// non-derived bits of the buffer, as configured by `non_derived_bits_mask`.
    /// Use this when the buffer's derived (axiom-computed) bits have not yet
    /// been refreshed for the current input. Two states with identical
    /// non-derived bits are guaranteed to be equal because axioms are
    /// deterministic functions of their inputs, so masked equality matches
    /// full equality and the resulting dedup is sound.
    fn insert_id_or_pop_state_masked(&mut self) -> (StateID, bool) {
        let state_id = self.state_data_pool.len() - 1;
        let num_bins = self.num_state_bins();
        let key = {
            let state_data = self.get_buffer(state_id);
            fast_hash_bins_masked(
                &state_data[..num_bins],
                &self.non_derived_bits_mask[..num_bins],
            )
        };

        let mut existing_id: Option<StateID> = None;
        if let Some(bucket) = self.registered_states.get(&key) {
            for &candidate in bucket {
                let existing = self.get_buffer(candidate);
                let probe = self.get_buffer(state_id);
                if bins_eq_masked(
                    &existing[..num_bins],
                    &probe[..num_bins],
                    &self.non_derived_bits_mask[..num_bins],
                ) {
                    existing_id = Some(candidate);
                    break;
                }
            }
        }

        if let Some(existing_id) = existing_id {
            self.state_data_pool.pop_back();
            return (existing_id, false);
        }

        self.registered_states
            .entry(key)
            .or_insert_with(|| Vec::with_capacity(4))
            .push(state_id);
        (state_id, true)
    }

    /// Create and registers the initial state of the planning problem.
    ///
    /// This method:
    /// 1. Packs propositional variables into the state buffer.
    /// 2. Processes numeric variables by type (regular, constant, cost, derived).
    /// 3. Evaluates axioms to compute derived values.
    /// 4. Registers the resulting state.
    pub fn get_initial_state(&mut self) -> ConcreteState {
        let mut init_buffer = vec![0u64; self.global_state_packer.num_bins()];

        // Get copies of initial state values to avoid borrowing conflicts.
        let initial_propositional_values =
            self.task.get_initial_propositional_state_values().clone();
        let initial_numeric_values = self.task.get_initial_numeric_state_values().clone();

        // Pack propositional variables.
        self.pack_propositional_variables(&mut init_buffer, &initial_propositional_values);

        // Process numeric variables and get cost variables.
        let _cost_variables =
            self.process_numeric_variables(&mut init_buffer, &initial_numeric_values);

        // Evaluate axioms.
        let mut numeric_state_copy = initial_numeric_values;
        self.evaluate_axioms(&mut init_buffer, &mut numeric_state_copy)
            .expect("Failed to evaluate axioms during initial state creation");

        // Register the state.
        self.state_data_pool.push_back(&init_buffer);
        let (state_id, _) = self.insert_id_or_pop_state();

        let init_state = ConcreteState::new(state_id);

        // Update the task's initial state values to reflect axiom evaluation.
        *self.task.get_initial_propositional_state_values_mut() = init_state.get_state(self);
        *self.task.get_initial_numeric_state_values_mut() = self
            .get_numeric_vars(&init_state)
            .expect("Failed to get numeric variables for initial state");

        // Store cost information for the initial state (after other operations).
        if !_cost_variables.is_empty() {
            let cost_variables_copy = _cost_variables.clone();
            self.set_cost_information(&init_state, cost_variables_copy);
        }

        // Seed the metric-value cache for the initial state.
        if self.metric_use_metric
            && let Ok(initial_metric) = self.metric_value_for_state(&init_state)
        {
            self.cache_metric_value(state_id, initial_metric);
        }

        #[cfg(debug_assertions)]
        self.log_initial_state_info(&_cost_variables);

        init_state
    }

    /// Pack propositional variables into the state buffer.
    fn pack_propositional_variables(&self, buffer: &mut [u64], initial_values: &[usize]) {
        for (i, &value) in initial_values.iter().enumerate() {
            self.global_state_packer.set(buffer, i, value as u64);
        }
    }

    /// Process numeric variables by type and returns cost variables.
    fn process_numeric_variables(
        &mut self,
        buffer: &mut [u64],
        initial_numeric_values: &[f64],
    ) -> Vec<f64> {
        let mut numeric_var_index = self.task.get_initial_propositional_state_values().len();
        let mut constant_index = 0;
        let mut cost_variables = Vec::new();

        for (i, &value) in initial_numeric_values.iter().enumerate() {
            let numeric_var = &self.task.numeric_variables()[i];

            match numeric_var.get_type() {
                NumericType::Cost => {
                    self.numeric_indices[i] = Some(cost_variables.len());
                    cost_variables.push(float_tolerance::canonicalize(value));
                }
                NumericType::Constant => {
                    self.numeric_indices[i] = Some(constant_index);
                    self.numeric_constants
                        .push(float_tolerance::canonicalize(value));
                    constant_index += 1;
                }
                NumericType::Derived => {
                    // Derived variables don't get indices as they're computed by axioms.
                }
                NumericType::Regular => {
                    self.numeric_indices[i] = Some(numeric_var_index);
                    let packed_value = self.global_state_packer.pack_double(value);
                    self.global_state_packer
                        .set(buffer, numeric_var_index, packed_value);
                    numeric_var_index += 1;
                }
            }
        }

        cost_variables
    }

    /// Evaluate axioms on the given state.
    fn evaluate_axioms(
        &self,
        buffer: &mut [u64],
        numeric_state: &mut [f64],
    ) -> Result<(), StateInsertError> {
        canonicalize_numeric_values(numeric_state);
        self.axiom_evaluator
            .evaluate_arithmetic_axioms(numeric_state)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate arithmetic axioms: {:?}", e),
            })?;

        self.axiom_evaluator
            .evaluate(buffer, numeric_state)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate axioms: {:?}", e),
            })?;

        Ok(())
    }

    /// Log initial state information in debug builds.
    #[cfg(debug_assertions)]
    fn log_initial_state_info(&self, cost_variables: &[f64]) {
        use tracing::info;

        let initial_propositional_len = self.task.get_initial_propositional_state_values().len();
        let regular_count = self
            .numeric_indices
            .iter()
            .filter(|&&idx| idx.is_some() && idx.unwrap() >= initial_propositional_len)
            .count();
        let constant_count = self.numeric_constants.len();
        let derived_count = self
            .task
            .numeric_variables()
            .iter()
            .filter(|var| var.get_type() == &NumericType::Derived)
            .count();

        info!(
            "Initial state: {} regular, {} constants, {} cost variables, {} derived variables",
            regular_count,
            constant_count,
            cost_variables.len(),
            derived_count
        );
    }

    /// Register a new state with the given propositional and numeric values.
    ///
    /// This method creates a new state from the provided values, evaluates axioms,
    /// and registers it in the state pool.
    pub fn register_state(
        &mut self,
        values: Vec<u64>,
        numeric_values: Vec<f64>,
    ) -> Result<ConcreteState, StateInsertError> {
        self.register_state_with_status(values, numeric_values)
            .map(|(state, _is_new)| state)
    }

    pub fn register_state_with_status(
        &mut self,
        values: Vec<u64>,
        numeric_values: Vec<f64>,
    ) -> Result<(ConcreteState, bool), StateInsertError> {
        let mut buffer = vec![0; self.global_state_packer.num_bins()];

        // Pack propositional variables.
        for (i, &value) in values.iter().enumerate() {
            self.global_state_packer.set(&mut buffer, i, value);
        }

        // Process numeric variables.
        let _cost_variables =
            self.process_register_numeric_variables(&mut buffer, &numeric_values)?;

        // Evaluate axioms
        let mut numeric_values_copy = numeric_values;
        self.evaluate_axioms(&mut buffer, &mut numeric_values_copy)?;

        self.state_data_pool.push_back(&buffer);
        let (id, is_new_state) = self.insert_id_or_pop_state();

        let new_state = ConcreteState::new(id);

        // Handle cost information based on whether this is a new or existing state.
        if is_new_state {
            // New state: store cost information.
            if !_cost_variables.is_empty() {
                self.set_cost_information(&new_state, _cost_variables);
            }
        } else {
            // Existing state: use metric optimization to determine which cost info to keep.
            let cost_info_borrow = self.cost_info.borrow();
            let keep_old_cost_information =
                self.should_keep_old_cost_information(&new_state, &numeric_values_copy);
            drop(cost_info_borrow); // Drop the borrow before calling set.

            match keep_old_cost_information {
                Ok(false) => {
                    self.set_cost_information(&new_state, _cost_variables);
                }
                Ok(true) => {}
                Err(e) => {
                    return Err(StateInsertError {
                        message: format!("Failed to select cost information: {:?}", e),
                    });
                }
            }
        }

        Ok((new_state, is_new_state))
    }

    /// Process numeric variables during state registration.
    fn process_register_numeric_variables(
        &mut self,
        buffer: &mut [u64],
        numeric_values: &[f64],
    ) -> Result<Vec<f64>, StateInsertError> {
        let mut regular_index = self.task.get_initial_propositional_state_values().len();
        let mut cost_variables = Vec::new();

        for (i, &value) in numeric_values.iter().enumerate() {
            let numeric_variable =
                self.task
                    .numeric_variables()
                    .get(i)
                    .ok_or_else(|| StateInsertError {
                        message: format!("Numeric variable at index {} not found", i),
                    })?;

            match numeric_variable.get_type() {
                NumericType::Cost => {
                    // Initialize the index if not set.
                    if self.numeric_indices[i].is_none() {
                        self.numeric_indices[i] = Some(cost_variables.len());
                    }
                    cost_variables.push(float_tolerance::canonicalize(value));
                }
                NumericType::Regular => {
                    // Initialize the index if not set.
                    if self.numeric_indices[i].is_none() {
                        self.numeric_indices[i] = Some(regular_index);
                        regular_index += 1;
                    }
                    let packed_value = self.global_state_packer.pack_double(value);
                    self.global_state_packer.set(
                        buffer,
                        self.numeric_indices[i].unwrap(),
                        packed_value,
                    );
                }
                _ => {
                    return Err(StateInsertError {
                        message: format!(
                            "Only regular and cost variables are allowed during registration: {:?}",
                            numeric_variable.get_type()
                        ),
                    });
                }
            }
        }

        Ok(cost_variables)
    }

    /// Look up a state by its index in the state pool.
    ///
    /// Returns an error if the index is out of bounds.
    pub fn lookup_state(&self, index: usize) -> Result<ConcreteState, StateNotFoundError> {
        if index >= self.state_data_pool.len() {
            Err(StateNotFoundError { index })
        } else {
            Ok(ConcreteState::new(index))
        }
    }

    /// Generates a successor state by applying an operator to the current state
    pub fn get_successor_state(
        &mut self,
        current_state: &ConcreteState,
        operator: &Operator,
    ) -> Result<ConcreteState, StateInsertError> {
        let mut successor_values = Vec::new();
        let mut cost_values = Vec::new();
        self.get_successor_state_with_buffers(
            current_state,
            operator,
            &mut successor_values,
            &mut cost_values,
        )
    }

    pub fn get_successor_state_with_buffers(
        &mut self,
        current_state: &ConcreteState,
        operator: &Operator,
        successor_values: &mut Vec<f64>,
        cost_values: &mut Vec<f64>,
    ) -> Result<ConcreteState, StateInsertError> {
        self.get_successor_state_with_buffers_and_cost(
            current_state,
            operator,
            successor_values,
            cost_values,
        )
        .map(|(successor, _)| successor)
    }

    pub fn get_successor_state_with_buffers_and_cost(
        &mut self,
        current_state: &ConcreteState,
        operator: &Operator,
        successor_values: &mut Vec<f64>,
        cost_values: &mut Vec<f64>,
    ) -> Result<(ConcreteState, f64), StateInsertError> {
        let mut ctx = ExpansionContext::default();
        self.build_expansion_context(current_state, &mut ctx)?;
        self.apply_operator_in_context(current_state, operator, &ctx, successor_values, cost_values)
    }

    /// Fill `ctx` with the parent's numeric values, cost vars, and metric
    /// value. Doing this once per expansion (rather than per successor) avoids
    /// repeatedly walking the numeric variables and re-reading the metric for
    /// the same parent state.
    pub fn build_expansion_context(
        &self,
        parent: &ConcreteState,
        ctx: &mut ExpansionContext,
    ) -> Result<(), StateInsertError> {
        self.fill_numeric_vars(parent, &mut ctx.parent_numeric)
            .map_err(|e| StateInsertError {
                message: format!("Failed to get numeric variables: {:?}", e),
            })?;
        self.fill_cost_information(parent, &mut ctx.parent_cost);
        let expected_cost_vars = self.count_cost_variables();
        if ctx.parent_cost.len() < expected_cost_vars {
            ctx.parent_cost.resize(expected_cost_vars, 0.0);
        }
        ctx.parent_metric = if self.metric_use_metric {
            // Prefer the dense per-state cache; fall back to the slower
            // cost-info-backed read if the parent isn't cached yet.
            match self.cached_metric_value(parent.get_id()) {
                Some(value) => value,
                None => self
                    .metric_value_for_state(parent)
                    .map_err(|e| StateInsertError {
                        message: format!("Failed to read metric for parent state: {e:?}"),
                    })?,
            }
        } else {
            0.0
        };
        Ok(())
    }

    /// Apply `operator` to `parent`, reusing the cached parent values from
    /// `ctx`. Compared to `get_successor_state_with_buffers_and_cost`, this
    /// avoids re-running `fill_numeric_vars`, `fill_cost_information`, and
    /// `metric_value_for_state` per successor.
    pub fn apply_operator_in_context(
        &mut self,
        parent: &ConcreteState,
        operator: &Operator,
        ctx: &ExpansionContext,
        successor_values: &mut Vec<f64>,
        cost_values: &mut Vec<f64>,
    ) -> Result<(ConcreteState, f64), StateInsertError> {
        // Seed successor scratch from the cached parent values; the numeric
        // and cost effects below will mutate them in place.
        successor_values.clear();
        successor_values.extend_from_slice(&ctx.parent_numeric);
        cost_values.clear();
        cost_values.extend_from_slice(&ctx.parent_cost);

        self.state_data_pool.push_copy(parent.get_id());
        let successor_state_id = self.state_data_pool.len() - 1;
        let previous_buffer_ptr = self.get_buffer(parent.get_id()).as_ptr();
        let next_buffer_ptr = self.get_buffer_mut(successor_state_id).as_mut_ptr();
        let num_bins = self.num_state_bins();

        let (previous_buffer, next_buffer) = unsafe {
            (
                std::slice::from_raw_parts(previous_buffer_ptr, num_bins),
                std::slice::from_raw_parts_mut(next_buffer_ptr, num_bins),
            )
        };

        self.apply_propositional_effects(next_buffer, parent, operator);

        // Skip the comparison/propositional axiom passes during effect
        // application when the task has axiom-derived bits we can mask off.
        // We dedup using only non-derived bits below; if the successor is a
        // duplicate, the existing state already has correct derived bits and
        // we save the (typically expensive) axiom evaluation. If the
        // successor is new, we run the full axiom pass after dedup.
        let defer_full_axioms = self.has_axiom_derived_bits;
        self.apply_numeric_effects_inner(
            successor_values,
            cost_values,
            operator,
            next_buffer,
            previous_buffer,
            !defer_full_axioms,
        )?;

        // Compute `op_cost` from the cached parent metric instead of going
        // back through `metric_value_for_state(parent)`. Compute new_metric
        // even if we're not using the metric (in that case we just return
        // 1.0 for `op_cost` but still want to pre-fill the metric cache).
        let new_metric = if self.metric_use_metric {
            self.evaluate_metric(successor_values)
                .map_err(|e| StateInsertError {
                    message: format!("Failed to evaluate metric: {e:?}"),
                })?
        } else {
            0.0
        };
        let op_cost = if self.metric_use_metric {
            new_metric - ctx.parent_metric
        } else {
            1.0
        };

        let (id, is_new_state) = self.insert_id_or_pop_state();
        let successor = ConcreteState::new(id);

        if is_new_state && defer_full_axioms {
            let new_buffer_ptr = self.get_buffer_mut(id).as_mut_ptr();
            let new_buffer = unsafe { std::slice::from_raw_parts_mut(new_buffer_ptr, num_bins) };
            self.axiom_evaluator
                .evaluate(new_buffer, successor_values)
                .map_err(|e| StateInsertError {
                    message: format!("Failed to evaluate axioms: {:?}", e),
                })?;
        }

        // Cost-info bookkeeping. For tasks without `Cost`-typed numeric
        // variables this is entirely a no-op, and we skip the whole branch.
        // For tasks where the metric variable is itself a `Regular` numeric
        // var, duplicates produced by the masked dedup are guaranteed to
        // have identical metric values (the mask covers the metric's bits),
        // so `should_keep_old_cost_information` is always `false` and we
        // can avoid that read on every duplicate.
        //
        // For Cost-typed metrics, we maintain `metric_value_by_state` as a
        // dense `Vec<f64>` cache so the duplicate path can decide using two
        // direct loads instead of going through the per-state cost-info
        // `HashMap`.
        if self.cost_variable_count > 0 {
            if is_new_state {
                if !cost_values.is_empty() {
                    // Clone (rather than move) so `cost_values` keeps its
                    // capacity across successors — the next call's
                    // `extend_from_slice(&ctx.parent_cost)` reuses it instead
                    // of allocating from zero.
                    self.set_cost_information(&successor, cost_values.clone());
                }
                if self.metric_use_metric {
                    self.cache_metric_value(id, new_metric);
                }
            } else {
                let metric_is_regular =
                    matches!(self.metric_var, Some((_, NumericType::Regular)) | None);
                let keep_old = if !self.metric_use_metric {
                    false
                } else if metric_is_regular {
                    // Masked dedup guarantees metric bits agree.
                    false
                } else {
                    // Compare via the dense metric cache (no HashMap probe).
                    match self.cached_metric_value(id) {
                        Some(old_metric) => {
                            if self.metric_is_min {
                                old_metric < new_metric
                            } else {
                                old_metric > new_metric
                            }
                        }
                        None => {
                            // Cache miss — fall back to cost_info-backed read.
                            let cost_info_borrow = self.cost_info.borrow();
                            let result =
                                self.should_keep_old_cost_information(&successor, successor_values);
                            drop(cost_info_borrow);
                            match result {
                                Ok(v) => v,
                                Err(e) => {
                                    return Err(StateInsertError {
                                        message: format!(
                                            "Failed to select cost information: {:?}",
                                            e
                                        ),
                                    });
                                }
                            }
                        }
                    }
                };

                if !keep_old {
                    self.set_cost_information(&successor, cost_values.clone());
                    if self.metric_use_metric {
                        self.cache_metric_value(id, new_metric);
                    }
                }
            }
        }

        Ok((successor, op_cost))
    }

    #[inline]
    fn cache_metric_value(&self, state_id: StateID, value: f64) {
        let mut cache = self.metric_value_by_state.borrow_mut();
        if state_id >= cache.len() {
            cache.resize(state_id + 1, f64::NAN);
        }
        cache[state_id] = value;
    }

    #[inline]
    fn cached_metric_value(&self, state_id: StateID) -> Option<f64> {
        let cache = self.metric_value_by_state.borrow();
        cache.get(state_id).copied().filter(|v| !v.is_nan())
    }

    /// Apply propositional effects of an operator to the state buffer.
    fn apply_propositional_effects(
        &self,
        buffer: &mut [u64],
        current_state: &ConcreteState,
        operator: &Operator,
    ) {
        for effect in operator.effects() {
            if effect.conditions_met(current_state, self) {
                let var_id = effect.var_id();
                let value = effect.value() as u64;
                self.global_state_packer.set(buffer, var_id, value);
            }
        }
    }

    /// Count the number of cost variables in the planning task.
    /// Returns the cached count populated at construction.
    #[inline]
    fn count_cost_variables(&self) -> usize {
        self.cost_variable_count
    }

    fn fill_cost_information(&self, state: &ConcreteState, output: &mut Vec<f64>) {
        let cost_info_borrow = self.cost_info.borrow();
        let cost_info_data = cost_info_borrow.get(state, self);
        output.resize(cost_info_data.len(), 0.0);
        output.copy_from_slice(cost_info_data);
    }

    /// Retrieve all numeric variable values for a given state.
    ///
    /// This method reconstructs the complete numeric state by:
    /// - Reading regular variables from the packed state buffer.
    /// - Using stored constants for constant variables.
    /// - Retrieving cost variables from per-state storage.
    /// - Evaluating arithmetic axioms to compute derived values.
    pub fn get_numeric_vars(&self, state: &ConcreteState) -> Result<Vec<f64>, InvalidIndex> {
        let mut result = vec![0.0; self.task.numeric_variables().len()];
        self.fill_numeric_vars(state, &mut result)?;
        Ok(result)
    }

    pub fn fill_state_and_numeric_vars(
        &self,
        state: &ConcreteState,
        propositional_output: &mut Vec<usize>,
        numeric_output: &mut Vec<f64>,
    ) -> Result<(), InvalidIndex> {
        self.fill_state_and_numeric_vars_with_options(
            state,
            propositional_output,
            numeric_output,
            true,
        )
    }

    pub fn fill_state_and_numeric_vars_with_options(
        &self,
        state: &ConcreteState,
        propositional_output: &mut Vec<usize>,
        numeric_output: &mut Vec<f64>,
        evaluate_arithmetic_axioms: bool,
    ) -> Result<(), InvalidIndex> {
        propositional_output.clear();
        propositional_output.reserve(self.task.variables().len());

        numeric_output.clear();
        numeric_output.resize(self.task.numeric_variables().len(), 0.0);

        let buffer = state.buffer(self);
        let state_packer = &self.global_state_packer;
        propositional_output
            .extend((0..self.task.variables().len()).map(|i| state_packer.get(buffer, i) as usize));

        let cost_info_borrow = self.cost_info.borrow();
        let cost_variables = cost_info_borrow.get(state, self);

        for (i, numeric_var) in self.task.numeric_variables().iter().enumerate() {
            numeric_output[i] = match numeric_var.get_type() {
                NumericType::Cost => {
                    let cost_index = self.numeric_indices[i];
                    if let Some(cost) = cost_index
                        && cost < cost_variables.len()
                    {
                        cost_variables[cost]
                    } else {
                        0.0
                    }
                }
                NumericType::Constant => self.numeric_constants[self.numeric_indices[i].unwrap()],
                NumericType::Regular => {
                    state_packer.get_double(buffer, self.numeric_indices[i].unwrap())
                }
                NumericType::Derived => 0.0,
            };
        }

        if evaluate_arithmetic_axioms && self.axiom_evaluator.has_numeric_axioms() {
            self.axiom_evaluator
                .evaluate_arithmetic_axioms(numeric_output)?;
        }

        Ok(())
    }

    pub fn get_propositional_var_value(
        &self,
        state: &ConcreteState,
        var_id: usize,
    ) -> Result<usize, InvalidIndex> {
        state.get_propositional_value(self, var_id)
    }

    pub fn get_numeric_var_value_unevaluated(
        &self,
        state: &ConcreteState,
        numeric_var_id: usize,
    ) -> Result<f64, InvalidIndex> {
        let Some(numeric_var) = self.task.numeric_variables().get(numeric_var_id) else {
            return Err(InvalidIndex {
                index: numeric_var_id,
                length: self.task.numeric_variables().len(),
            });
        };

        let buffer = state.buffer(self);
        let cost_info_borrow = self.cost_info.borrow();
        let cost_variables = cost_info_borrow.get(state, self);

        let value = match numeric_var.get_type() {
            NumericType::Cost => {
                let cost_index = self.numeric_indices[numeric_var_id];
                if let Some(cost) = cost_index
                    && cost < cost_variables.len()
                {
                    cost_variables[cost]
                } else {
                    0.0
                }
            }
            NumericType::Constant => {
                self.numeric_constants[self.numeric_indices[numeric_var_id].unwrap()]
            }
            NumericType::Regular => self
                .global_state_packer
                .get_double(buffer, self.numeric_indices[numeric_var_id].unwrap()),
            NumericType::Derived => 0.0,
        };

        Ok(value)
    }

    pub fn fill_numeric_vars(
        &self,
        state: &ConcreteState,
        output: &mut Vec<f64>,
    ) -> Result<(), InvalidIndex> {
        output.resize(self.numeric_var_types.len(), 0.0);

        let buffer = state.buffer(self);

        // Get cost information for this state.
        let cost_info_borrow = self.cost_info.borrow();
        let cost_variables = cost_info_borrow.get(state, self);

        // Fill in values by variable type. Iterate the cached layout to avoid
        // a vtable dispatch through `task.numeric_variables()` per element.
        for (i, ty) in self.numeric_var_types.iter().enumerate() {
            output[i] = match ty {
                NumericType::Cost => {
                    // Retrieve cost variable from per-state storage.
                    if let Some(cost_index) = self.numeric_indices[i]
                        && cost_index < cost_variables.len()
                    {
                        cost_variables[cost_index]
                    } else {
                        // Default if not found.
                        0.0
                    }
                }
                NumericType::Constant => self.numeric_constants[self.numeric_indices[i].unwrap()],
                NumericType::Regular => self
                    .global_state_packer
                    .get_double(buffer, self.numeric_indices[i].unwrap()),
                NumericType::Derived => {
                    // Derived variables are computed by axioms.
                    0.0
                }
            };
        }

        if self.axiom_evaluator.has_numeric_axioms() {
            self.axiom_evaluator.evaluate_arithmetic_axioms(output)?;
        }

        Ok(())
    }

    /// Apply numeric assignment effects to create a successor state.
    ///
    /// This is the improved version that works directly with buffers for efficiency.
    fn apply_numeric_effects(
        &self,
        current_values: &mut [f64],
        cost_part: &mut [f64],
        operator: &Operator,
        next_buffer: &mut [u64],
        previous_buffer: &[u64],
    ) -> Result<(), StateInsertError> {
        self.apply_numeric_effects_inner(
            current_values,
            cost_part,
            operator,
            next_buffer,
            previous_buffer,
            true,
        )
    }

    /// Like `apply_numeric_effects`, but if `run_full_axioms` is false the
    /// comparison and propositional axiom passes are skipped (arithmetic
    /// axioms still run because the metric and any subsequent computations
    /// may need derived numeric values). The caller is then responsible for
    /// running `axiom_evaluator.evaluate` once it is known the successor is a
    /// new state worth registering — see `get_successor_state_with_buffers_and_cost`.
    fn apply_numeric_effects_inner(
        &self,
        current_values: &mut [f64],
        cost_part: &mut [f64],
        operator: &Operator,
        next_buffer: &mut [u64],
        previous_buffer: &[u64],
        run_full_axioms: bool,
    ) -> Result<(), StateInsertError> {
        for effect in operator.assignment_effects() {
            let assignment_var_id = effect.var_id();
            let affected_var_id = effect.affected_var_id();

            if assignment_var_id >= current_values.len() {
                return Err(StateInsertError {
                    message: format!("Assignment variable ID {} out of bounds", assignment_var_id),
                });
            }

            let assignment_ty = self.numeric_var_types[assignment_var_id];
            let affected_ty = self.numeric_var_types[affected_var_id];

            let assignment_value = if assignment_ty == NumericType::Regular {
                self.global_state_packer.get_double(
                    previous_buffer,
                    self.numeric_indices[assignment_var_id].unwrap(),
                )
            } else {
                current_values[assignment_var_id]
            };

            let result = AssignmentOperation::apply(
                current_values[affected_var_id],
                effect.operation(),
                assignment_value,
            );
            let result = float_tolerance::canonicalize(result);

            match affected_ty {
                NumericType::Cost => {
                    let cost_index = self.numeric_indices[affected_var_id].unwrap();
                    if cost_index >= cost_part.len() {
                        return Err(StateInsertError {
                            message: format!("Cost variable index {} out of bounds", cost_index),
                        });
                    }
                    cost_part[cost_index] = result;
                    current_values[affected_var_id] = result;
                }
                NumericType::Regular => {
                    let packed_result = self.global_state_packer.pack_double(result);
                    self.global_state_packer.set(
                        next_buffer,
                        self.numeric_indices[affected_var_id].unwrap(),
                        packed_result,
                    );
                    current_values[affected_var_id] = result;
                }
                _ => {
                    return Err(StateInsertError {
                        message: format!(
                            "Only regular and cost variables can be affected by assignment operations: {:?}",
                            affected_ty
                        ),
                    });
                }
            }
        }

        self.axiom_evaluator
            .evaluate_arithmetic_axioms(current_values)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate arithmetic axioms: {:?}", e),
            })?;

        if run_full_axioms {
            self.axiom_evaluator
                .evaluate(next_buffer, current_values)
                .map_err(|e| StateInsertError {
                    message: format!("Failed to evaluate axioms: {:?}", e),
                })?;
        }

        Ok(())
    }

    /// Evaluate the metric value for a given numeric state.
    ///
    /// This corresponds to the C++ evaluate_metric function that retrieves
    /// the value of the metric fluent from the numeric state.
    pub fn evaluate_metric(&self, numeric_state: &[f64]) -> Result<f64, InvalidIndex> {
        match self.metric_var {
            Some((metric, _)) => {
                if metric < numeric_state.len() {
                    Ok(numeric_state[metric])
                } else {
                    Err(InvalidIndex {
                        length: numeric_state.len(),
                        index: metric,
                    })
                }
            }
            None => Ok(0.0),
        }
    }

    /// Compute the *raw* metric delta obtained by applying `operator` in `state`.
    /// - Evaluate the metric in the given state.
    /// - Apply the operator's propositional + numeric effects (without checking applicability).
    /// - Evaluate the metric in the resulting values.
    /// - Return `metric_after - metric_before`.
    pub fn metric_delta_applying_operator(
        &self,
        state: &ConcreteState,
        operator: &Operator,
    ) -> Result<f64, StateInsertError> {
        if !self.task.metric().use_metric() {
            // Numeric-FD treats non-metric tasks as unit-cost.
            return Ok(1.0);
        }

        let old_metric = self
            .metric_value_for_state(state)
            .map_err(|e| StateInsertError {
                message: format!("Failed to read metric value for state: {e:?}"),
            })?;

        let previous_buffer = state.buffer(self);
        let mut next_buffer = previous_buffer.to_vec();
        let mut successor_numeric_values = Vec::with_capacity(self.task.numeric_variables().len());
        self.fill_numeric_vars(state, &mut successor_numeric_values)
            .map_err(|e| StateInsertError {
                message: format!("Failed to read numeric variables for state: {e:?}"),
            })?;

        let mut cost_values = Vec::new();
        self.fill_cost_information(state, &mut cost_values);
        let expected_cost_vars = self.count_cost_variables();
        if cost_values.len() < expected_cost_vars {
            cost_values.resize(expected_cost_vars, 0.0);
        }

        self.apply_propositional_effects(&mut next_buffer, state, operator);
        self.apply_numeric_effects(
            &mut successor_numeric_values,
            cost_values.as_mut_slice(),
            operator,
            &mut next_buffer,
            previous_buffer,
        )?;

        let new_metric = self
            .evaluate_metric(&successor_numeric_values)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate metric after applying operator: {e:?}"),
            })?;

        Ok(new_metric - old_metric)
    }

    fn metric_value_for_state(&self, state: &ConcreteState) -> Result<f64, InvalidIndex> {
        let Some((metric_fluent_id, metric_type)) = self.metric_var else {
            return Ok(0.0);
        };
        if metric_fluent_id >= self.numeric_var_types.len() {
            return Err(InvalidIndex {
                length: self.numeric_var_types.len(),
                index: metric_fluent_id,
            });
        }

        match metric_type {
            NumericType::Regular => {
                let buffer = state.buffer(self);
                Ok(self
                    .global_state_packer
                    .get_double(buffer, self.numeric_indices[metric_fluent_id].unwrap()))
            }
            NumericType::Cost => {
                let cost_index = self.numeric_indices[metric_fluent_id];
                let cost_info_borrow = self.cost_info.borrow();
                let cost_values = cost_info_borrow.get(state, self);
                if let Some(cost) = cost_index
                    && cost < cost_values.len()
                {
                    Ok(cost_values[cost])
                } else {
                    Ok(0.0)
                }
            }
            NumericType::Constant => {
                Ok(self.numeric_constants[self.numeric_indices[metric_fluent_id].unwrap()])
            }
            NumericType::Derived => {
                let numeric_vals = self.get_numeric_vars(state)?;
                self.evaluate_metric(&numeric_vals)
            }
        }
    }

    /// Compute the transition cost between two states based on the metric fluent.
    /// If a metric is defined, the cost is the absolute change according to min/max:
    /// - For minimizing metrics, cost = max(0, new - old).
    /// - For maximizing metrics, cost = max(0, old - new).
    ///
    /// If no metric is defined, returns 1.0 as a default unit cost.
    pub fn transition_cost(
        &self,
        predecessor: &ConcreteState,
        successor: &ConcreteState,
    ) -> Result<f64, InvalidIndex> {
        if !self.task.metric().use_metric() {
            return Ok(1.0);
        }

        let old_metric = self.metric_value_for_state(predecessor)?;
        let new_metric = self.metric_value_for_state(successor)?;
        let is_min = self.task.metric().is_min();
        let delta = if is_min {
            new_metric - old_metric
        } else {
            old_metric - new_metric
        };
        Ok(delta.max(0.0))
    }

    /// Determine which cost information to keep when states are deduplicated.
    ///
    /// This implements the C++ logic for metric optimization when duplicate states are found.
    /// Return the cost information that should be kept based on metric optimization.
    #[allow(clippy::if_same_then_else)]
    fn should_keep_old_cost_information(
        &self,
        existing_state: &ConcreteState,
        successor_numeric_vals: &[f64],
    ) -> Result<bool, InvalidIndex> {
        if !self.task.metric().use_metric() {
            return Ok(false);
        }

        let old_metric_val = self.metric_value_for_state(existing_state)?;
        let new_metric_val = self.evaluate_metric(successor_numeric_vals)?;

        let metric_minimizes = self.task.metric().is_min();

        if metric_minimizes && old_metric_val < new_metric_val {
            Ok(true)
        } else if !metric_minimizes && old_metric_val > new_metric_val {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get cost information for a given state.
    ///
    /// This corresponds to the C++ g_cost_information[state] access pattern.
    /// Return an empty vector if no cost information is stored for the state.
    pub fn get_cost_information(&self, state: &ConcreteState) -> Vec<f64> {
        self.cost_info.borrow().get(state, self).clone()
    }

    /// Set cost information for a given state.
    ///
    /// This corresponds to the C++ g_cost_information[state] = values assignment.
    /// It uses `RefCell` for interior mutability to resolve borrowing conflicts.
    fn set_cost_information(&self, state: &ConcreteState, mut values: Vec<f64>) {
        canonicalize_numeric_values(&mut values);
        self.cost_info.borrow_mut().set(state, self, values);
    }
}

fn canonicalize_numeric_values(values: &mut [f64]) {
    for value in values {
        *value = float_tolerance::canonicalize(*value);
    }
}

impl<'a> Drop for StateRegistry<'a> {
    /// Implements the C++ `StateRegistry` destructor pattern.
    ///
    /// When a `StateRegistry` is destroyed, it notifies all subscribed
    /// `PerStateInformation` instances to clean up their data for this registry.
    /// This prevents memory leaks and dangling references.
    fn drop(&mut self) {
        // Notify the cost_info that this registry is being destroyed.
        // This follows the C++ pattern where StateRegistry destructor
        // calls remove_state_registry on all subscribed `PerStateInformation`
        // instances.
        self.cost_info.borrow_mut().cleanup_registry(self.id);
    }
}
