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

use crate::numeric::axioms::AxiomEvaluator;
use crate::numeric::numeric_task::{AbstractNumericTask, AssignmentOperation, Operator};
use crate::numeric::utils::errors::{InvalidIndex, StateInsertError, StateNotFoundError};
use crate::numeric::utils::float_tolerance;
use crate::numeric::utils::per_state_info::PerStateInformation;
use crate::numeric::utils::segmented_vector2::SegmentedArrayVector;
use crate::numeric::{numeric_task::NumericType, utils::int_packer::IntDoublePacker};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

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
        let task = state_registry.task;
        let state_packer = state_registry.global_state_packer;

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
        let task = state_registry.task;
        let state_packer = state_registry.global_state_packer;

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
        let task = registry.task;
        let num_variables = task.variables().len();
        let num_regular_numeric_vars = task
            .numeric_variables()
            .iter()
            .filter(|v| v.get_type() == &NumericType::Regular)
            .count();

        let buffer = self.buffer(registry);
        let state_packer = registry.global_state_packer;

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

#[inline]
fn fast_hash_bins(bins: &[u64]) -> u64 {
    // 64-bit `FNV-1a`.
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
    hash
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
    /// Reference to the root planning task.
    task: &'a dyn AbstractNumericTask,
    /// Axiom evaluator for handling derived predicates and numeric axioms.
    axiom_evaluator: &'a AxiomEvaluator<'a>,
    /// State packer for efficient bit-level state representation.
    global_state_packer: &'a StatePacker,
    /// Pool of state data, each entry is a packed state representation.
    state_data_pool: DataStorage,
    /// Constants for numeric variables.
    numeric_constants: Vec<f64>,
    /// Mapping from numeric variable index to packed state index.
    numeric_indices: Vec<Option<usize>>,
    /// Buckets of registered states for duplicate detection (hash -> `Vec<StateID>`).
    registered_states: HashMap<u64, Vec<StateID>>,
    /// Per-state cost information storage.
    cost_info: RefCell<PerStateInformation<Vec<f64>>>,
}

impl<'a> StateRegistry<'a> {
    /// Create a new state registry for the given planning task.
    pub fn new(
        task: &'a dyn AbstractNumericTask,
        global_state_packer: &'a StatePacker,
        axiom_evaluator: &'a AxiomEvaluator<'a>,
    ) -> Self {
        let number_numeric_vars = task.numeric_variables().len();
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);

        // Create cost info and subscribe it to this registry
        let mut cost_info = PerStateInformation::new();
        cost_info.subscribe(id);

        Self {
            id,
            task,
            global_state_packer,
            state_data_pool: DataStorage::new(global_state_packer.num_bins()),
            numeric_constants: Vec::new(),
            numeric_indices: vec![None; number_numeric_vars],
            registered_states: HashMap::with_capacity(1024),
            axiom_evaluator,
            cost_info: RefCell::new(cost_info),
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

    pub fn get_registered_states(&self) -> &HashMap<u64, Vec<usize>> {
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
    pub const fn global_state_packer(&self) -> &StatePacker {
        self.global_state_packer
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
        self.fill_numeric_vars(current_state, successor_values)
            .map_err(|e| StateInsertError {
                message: format!("Failed to get numeric variables: {:?}", e),
            })?;

        self.fill_cost_information(current_state, cost_values);
        let expected_cost_vars = self.count_cost_variables();
        if cost_values.len() < expected_cost_vars {
            cost_values.resize(expected_cost_vars, 0.0);
        }

        self.state_data_pool.push_copy(current_state.get_id());
        let successor_state_id = self.state_data_pool.len() - 1;
        let previous_buffer_ptr = self.get_buffer(current_state.get_id()).as_ptr();
        let next_buffer_ptr = self.get_buffer_mut(successor_state_id).as_mut_ptr();
        let num_bins = self.num_state_bins();

        let (previous_buffer, next_buffer) = unsafe {
            (
                std::slice::from_raw_parts(previous_buffer_ptr, num_bins),
                std::slice::from_raw_parts_mut(next_buffer_ptr, num_bins),
            )
        };

        self.apply_propositional_effects(next_buffer, current_state, operator);

        self.apply_numeric_effects(
            successor_values,
            cost_values,
            operator,
            next_buffer,
            previous_buffer,
        )?;

        let op_cost =
            self.metric_delta_from_successor_numeric_values(current_state, successor_values)?;

        let (id, is_new_state) = self.insert_id_or_pop_state();
        let successor = ConcreteState::new(id);

        if is_new_state {
            if !cost_values.is_empty() {
                self.set_cost_information(&successor, cost_values.clone());
            }
        } else {
            let cost_info_borrow = self.cost_info.borrow();
            let keep_old_cost_information =
                self.should_keep_old_cost_information(&successor, successor_values);
            drop(cost_info_borrow);

            match keep_old_cost_information {
                Ok(false) => {
                    self.set_cost_information(&successor, cost_values.clone());
                }
                Ok(true) => {}
                Err(e) => {
                    return Err(StateInsertError {
                        message: format!("Failed to select cost information: {:?}", e),
                    });
                }
            }
        }

        Ok((successor, op_cost))
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
    fn count_cost_variables(&self) -> usize {
        self.task
            .numeric_variables()
            .iter()
            .filter(|var| var.get_type() == &NumericType::Cost)
            .count()
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
        let state_packer = self.global_state_packer;
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
        output.resize(self.task.numeric_variables().len(), 0.0);

        let buffer = state.buffer(self);

        // Get cost information for this state.
        let cost_info_borrow = self.cost_info.borrow();
        let cost_variables = cost_info_borrow.get(state, self);

        // Fill in values by variable type.
        for (i, numeric_var) in self.task.numeric_variables().iter().enumerate() {
            output[i] = match numeric_var.get_type() {
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
        for effect in operator.assignment_effects() {
            let assignment_var_id = effect.var_id();
            let affected_var_id = effect.affected_var_id();

            if assignment_var_id >= current_values.len() {
                return Err(StateInsertError {
                    message: format!("Assignment variable ID {} out of bounds", assignment_var_id),
                });
            }

            let assignment_var = &self.task.numeric_variables()[assignment_var_id];
            let affected_var = &self.task.numeric_variables()[affected_var_id];

            let assignment_value = if assignment_var.get_type() == &NumericType::Regular {
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

            match affected_var.get_type() {
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
                            affected_var.get_type()
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

        self.axiom_evaluator
            .evaluate(next_buffer, current_values)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate axioms: {:?}", e),
            })?;

        Ok(())
    }

    /// Evaluate the metric value for a given numeric state.
    ///
    /// This corresponds to the C++ evaluate_metric function that retrieves
    /// the value of the metric fluent from the numeric state.
    pub fn evaluate_metric(&self, numeric_state: &[f64]) -> Result<f64, InvalidIndex> {
        let metric_fluent_id = self.task.metric().var_id();

        match metric_fluent_id {
            Some(metric) => {
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

    fn metric_delta_from_successor_numeric_values(
        &self,
        state: &ConcreteState,
        successor_numeric_values: &[f64],
    ) -> Result<f64, StateInsertError> {
        if !self.task.metric().use_metric() {
            return Ok(1.0);
        }

        let old_metric = self
            .metric_value_for_state(state)
            .map_err(|e| StateInsertError {
                message: format!("Failed to read metric value for state: {e:?}"),
            })?;

        let new_metric = self
            .evaluate_metric(successor_numeric_values)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate metric after applying operator: {e:?}"),
            })?;

        Ok(new_metric - old_metric)
    }

    fn metric_value_for_state(&self, state: &ConcreteState) -> Result<f64, InvalidIndex> {
        let metric_fluent_id = self.task.metric().var_id();
        if metric_fluent_id.is_none() {
            return Ok(0.0);
        }

        let metric_fluent_id = metric_fluent_id.unwrap();
        if metric_fluent_id >= self.task.numeric_variables().len() {
            return Err(InvalidIndex {
                length: self.task.numeric_variables().len(),
                index: metric_fluent_id,
            });
        }

        let metric_var = &self.task.numeric_variables()[metric_fluent_id];
        match metric_var.get_type() {
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
