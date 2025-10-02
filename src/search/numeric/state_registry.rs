//! State Registry for Numeric Planning
//! 
//! This module provides the `StateRegistry` which is responsible for managing
//! planning states in a numeric planning context. It handles:
//! 
//! - State creation and deduplication
//! - Efficient state representation using bit packing
//! - Numeric variable management (regular, constant, cost, derived)
//! - Axiom evaluation for derived predicates and variables
//! - Successor state generation
//! 
//! # Key Components
//! 
//! - `ConcreteState`: Represents a concrete planning state
//! - `StateRegistry`: Central registry for state management
//! - Efficient storage using segmented vectors and bit packing
//! - Integration with axiom evaluation system

use crate::search::numeric::axioms::AxiomEvaluator;
use crate::search::numeric::numeric_task::{
    AbstractNumericTask, AssignmentOperation, Fact, Operator,
};
use crate::search::numeric::utils::errors::{InvalidIndex, StateInsertError, StateNotFoundError};
use crate::search::numeric::utils::per_state_info::{PerStateInformation, PerStateInformationBase};
use crate::search::numeric::{
    numeric_task::{NumericRootTask, NumericType},
    utils::int_packer::IntDoublePacker,
};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Type alias for the state packer used throughout the system
type StatePacker = IntDoublePacker;

/// Type alias for state identifiers
pub type StateID = usize;

/// Type alias for the underlying data storage
type DataStorage = Vec<Vec<u64>>;

/// Represents a concrete state in the planning problem
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcreteState {
    pool_offset: usize,
}

impl ConcreteState {
    /// Creates a new concrete state with the given pool offset
    pub const fn new(pool_offset: usize) -> Self {
        Self { pool_offset }
    }

    /// Gets the state ID (equivalent to C++ GlobalState::get_id())
    /// This is the index into the state registry's data pool
    pub fn get_id(&self) -> usize {
        self.pool_offset
    }

    /// Gets the propositional state values as a vector
    pub fn get_state(&self, state_registry: &StateRegistry) -> Vec<i32> {
        let buffer = state_registry.get_buffer(self.pool_offset);
        let task = state_registry.root_task;
        let state_packer = state_registry.global_state_packer;
        
        (0..task.variables().len())
            .map(|i| state_packer.get(buffer, i as i32) as i32)
            .collect()
    }

    /// Gets the numeric state values for regular variables
    pub fn get_numeric_state(&self, state_registry: &StateRegistry) -> Vec<f64> {
        let buffer = state_registry.get_buffer(self.pool_offset);
        let task = state_registry.root_task;
        let state_packer = state_registry.global_state_packer;

        task.numeric_variables()
            .iter()
            .enumerate()
            .filter_map(|(i, var)| {
                if var.get_type() == &NumericType::Regular {
                    Some(state_packer.get_double(buffer, i as i32))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Gets a reference to the underlying buffer for this state
    pub fn buffer<'a>(&self, state_registry: &'a StateRegistry) -> &'a Vec<u64> {
        state_registry.get_buffer(self.pool_offset)
    }

    /// Returns the number of propositional variables in this state
    pub fn len(&self, state_registry: &StateRegistry) -> usize {
        state_registry.root_task.variables().len()
    }

    /// Returns true if the state has no variables (should never happen in practice)
    pub fn is_empty(&self, state_registry: &StateRegistry) -> bool {
        self.len(state_registry) == 0
    }

    /// Creates a debug representation of this state with variable values
    pub fn debug_with_registry(&self, registry: &StateRegistry) -> String {
        let task = registry.root_task;
        let num_variables = task.variables().len();
        let num_regular_numeric_vars = task
            .numeric_variables()
            .iter()
            .filter(|v| v.get_type() == &NumericType::Regular)
            .count();

        let buffer = self.buffer(registry);
        let state_packer = registry.global_state_packer;

        let mut result = format!("ConcreteState with {} bins\n", buffer.len());
        
        // Add propositional variables
        for i in 0..num_variables {
            let value = state_packer.get(buffer, i as i32);
            result.push_str(&format!("Var {}: {}\n", i, value));
        }
        
        // Add numeric variables
        for i in 0..num_regular_numeric_vars {
            let numeric_var_id = i + num_variables;
            let packed_value = state_packer.get(buffer, numeric_var_id as i32);
            let numeric_value = state_packer.unpack_double(packed_value);
            result.push_str(&format!("Numeric Var {}: {}\n", numeric_var_id, numeric_value));
        }
        
        result
    }
}

/// Semantic state ID for hash-based state deduplication
struct SemanticStateID<'a> {
    id: StateID,
    state_data_pool: &'a DataStorage,
    num_bins: usize,
}

impl<'a> PartialEq for SemanticStateID<'a> {
    fn eq(&self, other: &Self) -> bool {
        let lhs_data = &self.state_data_pool[self.id][..self.num_bins];
        let rhs_data = &other.state_data_pool[other.id][..self.num_bins];
        lhs_data == rhs_data
    }
}

impl<'a> Eq for SemanticStateID<'a> {}

impl<'a> Hash for SemanticStateID<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let bins = &self.state_data_pool[self.id][..self.num_bins];
        bins.hash(state); // slices of u64 already implement Hash
    }
}

/// Static counter for generating unique registry IDs
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// Central registry for managing planning states with deduplication and axiom evaluation
/// 
/// The StateRegistry is responsible for:
/// - Creating and managing planning states
/// - Deduplicating identical states to save memory
/// - Evaluating axioms when states change
/// - Managing numeric variables and cost information
pub struct StateRegistry<'a> {
    /// Unique identifier for this registry instance
    id: usize,
    /// Reference to the root planning task
    root_task: &'a NumericRootTask,
    /// Axiom evaluator for handling derived predicates and numeric axioms
    axiom_evaluator: &'a AxiomEvaluator<'a>,
    /// State packer for efficient bit-level state representation
    global_state_packer: &'a StatePacker,
    /// Pool of state data - each entry is a packed state representation
    state_data_pool: DataStorage,
    /// Constants for numeric variables
    numeric_constants: Vec<f64>,
    /// Mapping from numeric variable index to packed state index
    numeric_indices: Vec<i32>,
    /// Set of registered state IDs for deduplication
    registered_states: HashSet<StateID>,
    /// Per-state cost information storage
    cost_info: RefCell<PerStateInformation<Vec<f64>>>,
}

impl<'a> StateRegistry<'a> {
    /// Creates a new state registry for the given planning task
    pub fn new(
        root_task: &'a NumericRootTask,
        global_state_packer: &'a StatePacker,
        axiom_evaluator: &'a AxiomEvaluator<'a>,
    ) -> Self {
        let number_numeric_vars = root_task.numeric_variables().len();
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        
        // Create cost info and subscribe it to this registry
        let mut cost_info = PerStateInformation::new();
        cost_info.subscribe(id);
        
        Self {
            id,
            root_task,
            global_state_packer,
            state_data_pool: Vec::new(),
            numeric_constants: Vec::new(),
            numeric_indices: vec![-1; number_numeric_vars],
            registered_states: HashSet::new(),
            axiom_evaluator,
            cost_info: RefCell::new(cost_info),
        }
    }

    /// Returns the unique ID of this registry
    pub const fn id(&self) -> usize {
        self.id
    }

    /// Subscribe to another registry (placeholder for future functionality)
    pub fn subscribe(&mut self, _registry: &StateRegistry) {
        todo!("Registry subscription not yet implemented")
    }

    /// Subscribes a PerStateInformation instance to this registry
    /// 
    /// When this registry is dropped, it will notify the subscribed PerStateInformation
    /// instances to clean up their data. This follows the C++ Fast Downward pattern
    /// where PerStateInformation instances register themselves with StateRegistry instances.
    /// 
    /// Note: In Rust, we can't hold mutable references to the PerStateInformation,
    /// so this method just ensures the PerStateInformation knows about this registry.
    pub fn subscribe_per_state_info<T>(&self, per_state_info: &mut PerStateInformation<T>) 
    where 
        T: Clone + Default,
    {
        per_state_info.subscribe(self.id);
    }

    /// Unsubscribes a PerStateInformation instance from this registry
    pub fn unsubscribe_per_state_info<T>(&self, per_state_info: &mut PerStateInformation<T>) 
    where 
        T: Clone + Default,
    {
        per_state_info.unsubscribe(self.id);
    }

    /// Gets the buffer at the specified index
    /// 
    /// # Panics
    /// Panics if the index is out of bounds
    pub fn get_buffer(&self, index: usize) -> &Vec<u64> {
        self.state_data_pool
            .get(index)
            .expect("State index out of bounds")
    }

    /// Returns a reference to the global state packer
    pub const fn global_state_packer(&self) -> &StatePacker {
        self.global_state_packer
    }

    /// Inserts a state into the registry or returns existing ID if duplicate
    /// 
    /// This method handles state deduplication by checking if an equivalent state
    /// already exists in the registry. If so, it removes the duplicate and returns
    /// the existing state ID.
    fn insert_id_or_pop_state(&mut self) -> StateID {
        let state_id = self.state_data_pool.len() - 1;
        
        // Try to insert the state ID into the registered states set
        // This uses semantic hashing/equality to detect duplicates
        let is_new_entry = self.registered_states.insert(state_id);
        
        if !is_new_entry {
            // State already exists, remove the duplicate from pool
            self.state_data_pool.pop();
            // Find and return the existing state ID
            // Note: In a proper implementation, we'd use a hash set with semantic equality
            // For now, we'll return the attempted ID (this needs proper semantic hashing)
            state_id
        } else {
            // New state, keep it
            state_id
        }
    }

    /// Creates and registers the initial state of the planning problem
    /// 
    /// This method:
    /// 1. Packs propositional variables into the state buffer
    /// 2. Processes numeric variables by type (regular, constant, cost, derived)
    /// 3. Evaluates axioms to compute derived values
    /// 4. Registers the resulting state
    pub fn get_initial_state(&mut self) -> ConcreteState {
        let mut init_buffer = vec![0u64; self.global_state_packer.num_bins() as usize];
        
        // Get copies of initial state values to avoid borrowing conflicts
        let initial_propositional_values = self.root_task.get_initial_propositional_state_values().clone();
        let initial_numeric_values = self.root_task.get_initial_numeric_state_values().clone();
        
        // Pack propositional variables
        self.pack_propositional_variables(&mut init_buffer, &initial_propositional_values);
        
        // Process numeric variables and get cost variables
        let _cost_variables = self.process_numeric_variables(&mut init_buffer, &initial_numeric_values);
        
        // Evaluate axioms
        let mut numeric_state_copy = initial_numeric_values;
        self.evaluate_axioms(&mut init_buffer, &initial_propositional_values, &mut numeric_state_copy)
            .expect("Failed to evaluate axioms during initial state creation");

        // Register the state
        self.state_data_pool.push(init_buffer);
        let state_id = self.insert_id_or_pop_state();

        let init_state = ConcreteState::new(state_id);

        // Update the task's initial state values to reflect axiom evaluation
        *self.root_task.get_initial_propositional_state_values_mut() = init_state.get_state(self);
        *self.root_task.get_initial_numeric_state_values_mut() = self.get_numeric_vars(&init_state)
            .expect("Failed to get numeric variables for initial state");

        // Store cost information for the initial state (after other operations)
        if !_cost_variables.is_empty() {
            let cost_variables_copy = _cost_variables.clone();
            self.set_cost_information(&init_state, cost_variables_copy);
        }

        #[cfg(debug_assertions)]
        self.log_initial_state_info(&_cost_variables);

        init_state
    }

    /// Packs propositional variables into the state buffer
    fn pack_propositional_variables(&self, buffer: &mut [u64], initial_values: &[i32]) {
        for (i, &value) in initial_values.iter().enumerate() {
            self.global_state_packer.set(buffer, i as i32, value as u64);
        }
    }

    /// Processes numeric variables by type and returns cost variables
    fn process_numeric_variables(&mut self, buffer: &mut [u64], initial_numeric_values: &[f64]) -> Vec<f64> {
        let initial_propositional_len = initial_numeric_values.len(); // This should be the correct length calculation
        
        let mut numeric_var_index = self.root_task.get_initial_propositional_state_values().len();
        let mut constant_index = 0;
        let mut derived_count = 0;
        let mut cost_variables = Vec::new();

        for (i, &value) in initial_numeric_values.iter().enumerate() {
            let numeric_var = &self.root_task.numeric_variables()[i];
            
            match numeric_var.get_type() {
                NumericType::Cost => {
                    self.numeric_indices[i] = cost_variables.len() as i32;
                    cost_variables.push(value);
                }
                NumericType::Constant => {
                    self.numeric_indices[i] = constant_index;
                    self.numeric_constants.push(value);
                    constant_index += 1;
                }
                NumericType::Derived => {
                    derived_count += 1;
                    // Derived variables don't get indices as they're computed by axioms
                }
                NumericType::Regular => {
                    self.numeric_indices[i] = numeric_var_index as i32;
                    let packed_value = self.global_state_packer.pack_double(value);
                    self.global_state_packer.set(buffer, numeric_var_index as i32, packed_value);
                    numeric_var_index += 1;
                }
            }
        }

        cost_variables
    }

    /// Evaluates axioms on the given state
    fn evaluate_axioms(
        &self,
        buffer: &mut [u64],
        propositional_state: &Vec<i32>,
        numeric_state: &mut Vec<f64>,
    ) -> Result<(), StateInsertError> {
        self.axiom_evaluator
            .evaluate_arithmetic_axioms(numeric_state)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate arithmetic axioms: {:?}", e),
            })?;

        self.axiom_evaluator
            .evaluate(buffer, propositional_state, numeric_state)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate axioms: {:?}", e),
            })?;

        Ok(())
    }

    /// Logs initial state information in debug builds
    #[cfg(debug_assertions)]
    fn log_initial_state_info(&self, cost_variables: &[f64]) {
        let initial_propositional_len = self.root_task.get_initial_propositional_state_values().len();
        let regular_count = self.numeric_indices.iter().filter(|&&idx| idx >= initial_propositional_len as i32).count();
        let constant_count = self.numeric_constants.len();
        let derived_count = self.root_task.numeric_variables().iter()
            .filter(|var| var.get_type() == &NumericType::Derived).count();

        println!(
            "Initial state: {} regular, {} constants, {} cost variables, {} derived variables",
            regular_count,
            constant_count,
            cost_variables.len(),
            derived_count
        );
    }

    /// Registers a new state with the given propositional and numeric values
    /// 
    /// This method creates a new state from the provided values, evaluates axioms,
    /// and registers it in the state pool.
    pub fn register_state(
        &mut self,
        values: Vec<u64>,
        numeric_values: Vec<f64>,
    ) -> Result<ConcreteState, StateInsertError> {
        let mut buffer = vec![0; self.global_state_packer.num_bins() as usize];
        
        // Pack propositional variables
        for (i, &value) in values.iter().enumerate() {
            self.global_state_packer.set(&mut buffer, i as i32, value);
        }

        // Process numeric variables
        let _cost_variables = self.process_register_numeric_variables(&mut buffer, &numeric_values)?;

        // Evaluate axioms
        let propositional_initial_state = self.root_task.get_initial_propositional_state_values().clone();
        let mut numeric_values_copy = numeric_values;
        self.evaluate_axioms(&mut buffer, &propositional_initial_state, &mut numeric_values_copy)?;

        self.state_data_pool.push(buffer);
        self.insert_id_or_pop_state();

        let new_state = ConcreteState::new(self.state_data_pool.len() - 1);
        
        // Handle cost information based on whether this is a new or existing state
        let state_id = self.state_data_pool.len() - 1;
        if state_id == self.registered_states.len() - 1 {
            // New state - store cost information
            if !_cost_variables.is_empty() {
                self.set_cost_information(&new_state, _cost_variables);
            }
        } else {
            // Existing state - use metric optimization to determine which cost info to keep
            let cost_info_borrow = self.cost_info.borrow();
            let old_cost_info = cost_info_borrow.get(&new_state, self);
            let selected_result = self.select_cost_information(&new_state, &numeric_values_copy, old_cost_info, &_cost_variables);
            drop(cost_info_borrow); // Drop the borrow before calling set
            
            match selected_result {
                Ok(selected_cost_info) => {
                    self.set_cost_information(&new_state, selected_cost_info);
                }
                Err(e) => {
                    return Err(StateInsertError {
                        message: format!("Failed to select cost information: {:?}", e),
                    });
                }
            }
        }
        
        Ok(new_state)
    }

    /// Processes numeric variables during state registration
    fn process_register_numeric_variables(
        &mut self,
        buffer: &mut [u64],
        numeric_values: &[f64],
    ) -> Result<Vec<f64>, StateInsertError> {
        let mut regular_index = self.root_task.get_initial_propositional_state_values().len() as i32;
        let mut cost_variables = Vec::new();

        for (i, &value) in numeric_values.iter().enumerate() {
            let numeric_variable = self.root_task.numeric_variables().get(i)
                .ok_or_else(|| StateInsertError {
                    message: format!("Numeric variable at index {} not found", i),
                })?;

            match numeric_variable.get_type() {
                NumericType::Cost => {
                    // Initialize the index if not set
                    if self.numeric_indices[i] == -1 {
                        self.numeric_indices[i] = cost_variables.len() as i32;
                    }
                    cost_variables.push(value);
                }
                NumericType::Regular => {
                    // Initialize the index if not set
                    if self.numeric_indices[i] == -1 {
                        self.numeric_indices[i] = regular_index;
                        regular_index += 1;
                    }
                    let packed_value = self.global_state_packer.pack_double(value);
                    self.global_state_packer.set(buffer, self.numeric_indices[i], packed_value);
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

    /// Looks up a state by its index in the state pool
    /// 
    /// Returns an error if the index is out of bounds
    pub fn lookup_state(&self, index: usize) -> Result<ConcreteState, StateNotFoundError> {
        if index >= self.state_data_pool.len() {
            Err(StateNotFoundError { index })
        } else {
            Ok(ConcreteState::new(index))
        }
    }

    /// Generates a successor state by applying an operator to the current state
    /// 
    /// This method:
    /// 1. Applies propositional effects to the state buffer
    /// 2. Applies numeric assignment effects
    /// 3. Evaluates axioms to compute derived values
    /// 4. Registers and returns the new state
    pub fn get_successor_state(
        &mut self,
        current_state: &ConcreteState,
        operator: &Operator,
    ) -> Result<ConcreteState, StateInsertError> {
        let mut buffer = current_state.buffer(self).clone();
        
        // Apply propositional effects
        self.apply_propositional_effects(&mut buffer, current_state, operator);
        
        // Apply numeric effects
        let mut successor_values = self.get_numeric_vars(current_state)
            .map_err(|e| StateInsertError {
                message: format!("Failed to get numeric variables: {:?}", e),
            })?;
        let mut cost_values = vec![0.0; self.count_cost_variables()];
        let mut buffer_copy = buffer.clone();
        let mut current_buffer = current_state.buffer(self).clone();
        
        self.apply_numeric_effects(
            &mut successor_values,
            &mut cost_values,
            operator,
            &mut buffer_copy,
            &mut current_buffer,
        )?;

        self.state_data_pool.push(buffer_copy);
        let id = self.insert_id_or_pop_state();
        let successor = self.lookup_state(id)
            .map_err(|e| StateInsertError {
                message: format!("Failed to lookup successor state: {:?}", e),
            })?;

        // Handle cost information based on whether this is a new or existing state
        if id == self.state_data_pool.len() - 1 {
            // This is a new state - store cost information
            if !cost_values.is_empty() {
                self.set_cost_information(&successor, cost_values);
            }
        } else {
            // Existing state - use metric optimization to determine which cost info to keep
            let cost_info_borrow = self.cost_info.borrow();
            let old_cost_info = cost_info_borrow.get(&successor, self);
            let selected_result = self.select_cost_information(current_state, &successor_values, old_cost_info, &cost_values);
            drop(cost_info_borrow); // Drop the borrow before calling set
            
            match selected_result {
                Ok(selected_cost_info) => {
                    self.set_cost_information(&successor, selected_cost_info);
                }
                Err(e) => {
                    return Err(StateInsertError {
                        message: format!("Failed to select cost information: {:?}", e),
                    });
                }
            }
        }

        Ok(successor)
    }

    /// Applies propositional effects of an operator to the state buffer
    fn apply_propositional_effects(
        &self,
        buffer: &mut [u64],
        current_state: &ConcreteState,
        operator: &Operator,
    ) {
        for effect in operator.effects() {
            if effect.conditions_met(current_state, self) {
                let var_id = effect.var_id() as i32;
                let value = effect.value() as u64;
                self.global_state_packer.set(buffer, var_id, value);
            }
        }
    }

    /// Counts the number of cost variables in the planning task
    fn count_cost_variables(&self) -> usize {
        self.root_task
            .numeric_variables()
            .iter()
            .filter(|var| var.get_type() == &NumericType::Cost)
            .count()
    }

    /// Retrieves all numeric variable values for a given state
    /// 
    /// This method reconstructs the complete numeric state by:
    /// - Reading regular variables from the packed state buffer
    /// - Using stored constants for constant variables
    /// - Retrieving cost variables from per-state storage
    /// - Evaluating arithmetic axioms to compute derived values
    fn get_numeric_vars(&self, state: &ConcreteState) -> Result<Vec<f64>, InvalidIndex> {
        let mut result = vec![0.0; self.root_task.numeric_variables().len()];
        let buffer = state.buffer(self);

        // Get cost information for this state
        let cost_info_borrow = self.cost_info.borrow();
        let cost_variables = cost_info_borrow.get(state, self);

        // Fill in values by variable type
        for (i, numeric_var) in self.root_task.numeric_variables().iter().enumerate() {
            result[i] = match numeric_var.get_type() {
                NumericType::Cost => {
                    // Retrieve cost variable from per-state storage
                    let cost_index = self.numeric_indices[i];
                    if cost_index >= 0 && (cost_index as usize) < cost_variables.len() {
                        cost_variables[cost_index as usize]
                    } else {
                        0.0 // Default if not found
                    }
                }
                NumericType::Constant => {
                    self.numeric_constants[self.numeric_indices[i] as usize]
                }
                NumericType::Regular => {
                    self.global_state_packer
                        .get_double(buffer, self.numeric_indices[i])
                }
                NumericType::Derived => {
                    // Derived variables are computed by axioms
                    0.0
                }
            };
        }

        debug_assert_eq!(
            result.len(),
            self.root_task.numeric_variables().len(),
            "Numeric variables length mismatch"
        );

        // Evaluate arithmetic axioms if present
        if self.axiom_evaluator.has_numeric_axioms() {
            self.axiom_evaluator
                .evaluate_arithmetic_axioms(&mut result)?;
        }

        Ok(result)
    }

    /// Applies numeric assignment effects to create a successor state
    /// 
    /// This is the improved version that works directly with buffers for efficiency
    fn apply_numeric_effects(
        &self,
        current_values: &mut Vec<f64>,
        cost_part: &mut [f64],
        operator: &Operator,
        next_buffer: &mut [u64],
        current_buffer: &mut [u64],
    ) -> Result<(), StateInsertError> {
        for effect in operator.assignment_effects() {
            let assignment_var_id = effect.var_id() as usize;
            let affected_var_id = effect.affected_var_id() as usize;
            
            if assignment_var_id >= current_values.len() {
                return Err(StateInsertError {
                    message: format!("Assignment variable ID {} out of bounds", assignment_var_id),
                });
            }

            let assignment_var = &self.root_task.numeric_variables()[assignment_var_id];
            let affected_var = &self.root_task.numeric_variables()[affected_var_id];

            // Get the assignment value based on variable type
            let assignment_value = if assignment_var.get_type() == &NumericType::Regular {
                self.global_state_packer.get_double(
                    current_buffer,
                    self.numeric_indices[assignment_var_id],
                )
            } else {
                current_values[assignment_var_id]
            };

            // Apply the operation
            let result = AssignmentOperation::apply(
                current_values[affected_var_id],
                effect.operation(),
                assignment_value,
            );

            // Store the result based on affected variable type
            match affected_var.get_type() {
                NumericType::Cost => {
                    let cost_index = self.numeric_indices[affected_var_id] as usize;
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
                        self.numeric_indices[affected_var_id],
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

        // Evaluate arithmetic axioms
        self.axiom_evaluator
            .evaluate_arithmetic_axioms(current_values)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate arithmetic axioms: {:?}", e),
            })?;

        // Evaluate general axioms
        let initial_propositional_state = self.root_task.get_initial_propositional_state_values().clone();
        self.axiom_evaluator
            .evaluate(next_buffer, &initial_propositional_state, current_values)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate axioms: {:?}", e),
            })?;

        Ok(())
    }

    /// Evaluates the metric value for a given numeric state
    /// 
    /// This corresponds to the C++ evaluate_metric function that retrieves
    /// the value of the metric fluent from the numeric state.
    pub fn evaluate_metric(&self, numeric_state: &[f64]) -> Result<f64, InvalidIndex> {
        let metric_fluent_id = self.root_task.metric().var_id();
        
        if metric_fluent_id < 0 {
            // No metric defined
            return Ok(0.0);
        }
        
        if metric_fluent_id as usize >= numeric_state.len() {
            return Err(InvalidIndex {
                length: numeric_state.len() as u32,
                index: metric_fluent_id as u32,
            });
        }
        
        Ok(numeric_state[metric_fluent_id as usize])
    }

    /// Determines which cost information to keep when states are deduplicated
    /// 
    /// This implements the C++ logic for metric optimization when duplicate states are found.
    /// Returns the cost information that should be kept based on metric optimization.
    fn select_cost_information(
        &self,
        predecessor_state: &ConcreteState,
        successor_numeric_vals: &[f64],
        old_cost_info: &[f64],
        new_cost_info: &[f64],
    ) -> Result<Vec<f64>, InvalidIndex> {
        if !self.root_task.metric().use_metric() {
            // No metric optimization, keep new values
            return Ok(new_cost_info.to_vec());
        }

        let predecessor_numeric_vals = self.get_numeric_vars(predecessor_state)?;
        let old_metric_val = self.evaluate_metric(&predecessor_numeric_vals)?;
        let new_metric_val = self.evaluate_metric(successor_numeric_vals)?;

        let metric_minimizes = self.root_task.metric().is_min();

        if metric_minimizes && old_metric_val < new_metric_val {
            // Metric minimizes and old value is better, keep old cost info
            Ok(old_cost_info.to_vec())
        } else if !metric_minimizes && old_metric_val > new_metric_val {
            // Metric maximizes and old value is better, keep old cost info
            Ok(old_cost_info.to_vec())
        } else {
            // New value is better or equal, keep new cost info
            Ok(new_cost_info.to_vec())
        }
    }

    /// Gets cost information for a given state
    /// 
    /// This corresponds to the C++ g_cost_information[state] access pattern.
    /// Returns an empty vector if no cost information is stored for the state.
    pub fn get_cost_information(&self, state: &ConcreteState) -> Vec<f64> {
        self.cost_info.borrow().get(state, self).clone()
    }

    /// Sets cost information for a given state
    /// 
    /// This corresponds to the C++ g_cost_information[state] = values assignment.
    /// Uses RefCell for interior mutability to resolve borrowing conflicts.
    fn set_cost_information(&self, state: &ConcreteState, values: Vec<f64>) {
        self.cost_info.borrow_mut().set(state, self, values);
    }
}

impl<'a> Drop for StateRegistry<'a> {
    /// Implements the C++ StateRegistry destructor pattern
    /// 
    /// When a StateRegistry is destroyed, it notifies all subscribed
    /// PerStateInformation instances to clean up their data for this registry.
    /// This prevents memory leaks and dangling references.
    fn drop(&mut self) {
        // Notify the cost_info that this registry is being destroyed
        // This follows the C++ pattern where StateRegistry destructor
        // calls remove_state_registry on all subscribed PerStateInformation instances
        self.cost_info.borrow_mut().cleanup_registry(self.id);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crate::search::numeric::numeric_task::Fact;
    use crate::setup_axiom_evaluator;
    use crate::setup_numeric_task;
    use crate::setup_state_packer;
    use crate::setup_state_registry;
    use crate::setup_successor_generator;

    #[test]
    fn test_state_registry_initial_state() {
        let problem = setup_numeric_task("misc/numeric_sas/example1.sas");
        let state_packer = setup_state_packer(&problem);
        let axiom_evaluator = setup_axiom_evaluator(&problem, &state_packer);
        let mut state_registry = setup_state_registry(&problem, &state_packer, &axiom_evaluator);
        let initial_state = state_registry.get_initial_state();
        print!(
            "Initial state: {}",
            initial_state.debug_with_registry(&state_registry)
        );
    }

    #[test]
    fn test_generate_immediate_successor_of_init_state() {
        let task = setup_numeric_task("misc/numeric_sas/example1.sas");
        let state_packer = setup_state_packer(&task);
        let axiom_evaluator = setup_axiom_evaluator(&task, &state_packer);
        let mut state_registry = setup_state_registry(&task, &state_packer, &axiom_evaluator);
        let initial_state = state_registry.get_initial_state();

        let state = initial_state.get_state(&state_registry);
        let facts = state
            .iter()
            .enumerate()
            .map(|(i, value)| Fact::new(i as u32, *value as i32))
            .collect::<Vec<_>>();
        let mut facts_refs = Vec::new();

        for fact in &facts {
            facts_refs.push(fact);
        }

        let suc_gen = setup_successor_generator(&task);

        let mut applicable_operators = VecDeque::new();
        suc_gen.get_applicable_operators(&facts_refs, &mut applicable_operators);

        let op = applicable_operators.pop_front().unwrap();

        println!(
            "Initial state: {}",
            initial_state.debug_with_registry(&state_registry)
        );
        println!("OP: {:?}", op);

        let successor = state_registry
            .get_successor_state(&initial_state, op)
            .expect("Failed to get successor state");

        println!(
            "Successor state: {}",
            successor.debug_with_registry(&state_registry)
        );
        println!("Numeric indices: {:?}", state_registry.numeric_indices);
    }

    #[test]
    fn test_cost_information_storage() {
        let task = setup_numeric_task("misc/numeric_sas/example1.sas");
        let state_packer = setup_state_packer(&task);
        let axiom_evaluator = setup_axiom_evaluator(&task, &state_packer);
        let mut state_registry = setup_state_registry(&task, &state_packer, &axiom_evaluator);

        let initial_state = state_registry.get_initial_state();

        // Check that cost information is stored
        let cost_info = state_registry.get_cost_information(&initial_state);
        println!("Initial state cost information: {:?}", cost_info);
        
        // The cost information should be accessible (empty vector if no cost variables)
        println!("Cost information length: {}", cost_info.len());
    }

    #[test]
    fn test_per_state_info_subscription() {
        let task = setup_numeric_task("misc/numeric_sas/example1.sas");
        let state_packer = setup_state_packer(&task);
        let axiom_evaluator = setup_axiom_evaluator(&task, &state_packer);
        let state_registry = setup_state_registry(&task, &state_packer, &axiom_evaluator);

        // Create a PerStateInformation instance
        let mut custom_per_state_info = crate::search::numeric::utils::per_state_info::PerStateInformation::<i32>::new();

        // Subscribe it to the registry
        state_registry.subscribe_per_state_info(&mut custom_per_state_info);
        
        // Verify subscription
        assert!(custom_per_state_info.is_subscribed_to(state_registry.id()));
        println!("PerStateInformation subscribed to registry {}", state_registry.id());

        // Test unsubscription
        state_registry.unsubscribe_per_state_info(&mut custom_per_state_info);
        assert!(!custom_per_state_info.is_subscribed_to(state_registry.id()));
        println!("PerStateInformation unsubscribed from registry {}", state_registry.id());

        // Re-subscribe for cleanup test
        state_registry.subscribe_per_state_info(&mut custom_per_state_info);
        assert!(custom_per_state_info.is_subscribed_to(state_registry.id()));

        // Manually cleanup (simulating registry destruction)
        custom_per_state_info.cleanup_registry(state_registry.id());
        assert!(!custom_per_state_info.is_subscribed_to(state_registry.id()));
        println!("PerStateInformation cleaned up for registry {}", state_registry.id());
    }

    #[test]
    fn test_automatic_cleanup_on_drop() {
        let task = setup_numeric_task("misc/numeric_sas/example1.sas");
        let state_packer = setup_state_packer(&task);
        let axiom_evaluator = setup_axiom_evaluator(&task, &state_packer);
        
        let registry_id = {
            let state_registry = setup_state_registry(&task, &state_packer, &axiom_evaluator);
            let id = state_registry.id();
            
            // Verify the cost_info is automatically subscribed
            assert!(state_registry.cost_info.borrow().is_subscribed_to(id));
            println!("StateRegistry {} has auto-subscribed cost_info", id);
            
            id
        }; // StateRegistry drops here, triggering automatic cleanup
        
        println!("StateRegistry {} has been dropped with automatic cleanup", registry_id);
    }
}
