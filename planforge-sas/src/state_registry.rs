//! The [`StateRegistry`]: every state a search reaches, packed and stored
//! once, with its numeric values interned and its axioms evaluated, and the
//! successors of a state under an operator.
//!
//! - `view`: [`ConcreteState`], a registered state's id, and
//!   [`ConcreteStateView`], a read-only view of its values;
//! - `layout`: what a registry fixes at construction -- the packer, where
//!   each numeric variable lives, the metric and the axioms;
//! - `hashing`: hashing and comparing packed states.

#[cfg(test)]
mod tests;

mod hashing;
mod layout;
mod view;

use self::hashing::{MixedHasherBuilder, bins_eq_masked, fast_hash_bins, fast_hash_bins_masked};
use self::layout::StateLayout;
use self::view::ConcreteStateViewBacking;
pub use self::view::{ConcreteState, ConcreteStateView};

use crate::axioms::AxiomEvaluator;
use crate::numeric_task::{ExplicitValueIndex, NumericValue, Operator, Task, VariableIndex};
use crate::utils::errors::{AssignmentAxiomError, StateInsertError, StateNotFoundError};
use crate::utils::segmented_vector::SegmentedArrayVector;
use crate::{numeric_task::NumericType, utils::state_packer::StatePacker};
use hashbrown::HashTable;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The registered states, as `(hash, compact state id)` pairs.
///
/// The hash is stored rather than recomputed. hashbrown asks for an entry's hash
/// again every time the table grows, and deriving it here means fetching that
/// state's packed bins out of the data pool and running `fast_hash_bins` over
/// them. Growing to a few hundred thousand states doubles the table around nine
/// times, and because the sizes form a geometric series the last doublings
/// dominate: raising the initial capacity is almost worthless, while making the
/// rehash itself cheap removes the whole cost. It was 6% of search runtime.
///
/// The stored hash also pre-filters lookups. A bucket collision used to go
/// straight to comparing packed bins, which is another data-pool fetch; now it
/// compares two `u64`s first.
type RegisteredStates = HashTable<(u64, u32)>;

#[derive(Debug)]
struct DenseCostInformation {
    values: Vec<NumericValue>,
    numcost_variables: usize,
}

impl DenseCostInformation {
    fn new(numcost_variables: usize) -> Self {
        Self {
            values: Vec::new(),
            numcost_variables,
        }
    }

    fn get(&self, state_id: StateID) -> &[NumericValue] {
        if self.numcost_variables == 0 {
            return &[];
        }
        let start = state_id
            .checked_mul(self.numcost_variables)
            .expect("cost-information index overflow");
        let end = start + self.numcost_variables;
        self.values.get(start..end).unwrap_or_else(|| {
            panic!("missing f64 cost information for registered state {state_id}")
        })
    }

    fn set(&mut self, state_id: StateID, values: &[NumericValue]) {
        assert_eq!(
            values.len(),
            self.numcost_variables,
            "cost-information row has the wrong number of f64 values"
        );
        let start = state_id
            .checked_mul(self.numcost_variables)
            .expect("cost-information index overflow");
        let end = start
            .checked_add(self.numcost_variables)
            .expect("cost-information index overflow");
        if self.values.len() < end {
            self.values.resize(end, NumericValue::new(0.0));
        }
        for (target, &value) in self.values[start..end].iter_mut().zip(values) {
            *target = value;
        }
    }
}

/// Type alias for state identifiers.
pub type StateID = usize;

/// Type alias for the underlying data storage.
type DataStorage = SegmentedArrayVector<u64>;

/// Reusable scratch holding the parent state's per-expansion data. Filled
/// once per expansion via `StateRegistry::build_expansion_context`, then
/// shared across every successor produced by that expansion. This avoids
/// re-reading the same parent on every operator application.
#[derive(Debug, Default, Clone)]
pub struct ExpansionContext {
    pub parent_numeric: Vec<NumericValue>,
    pub parent_cost: Vec<NumericValue>,
    pub parent_metric: NumericValue,
}

/// The parent and successor views one operator application reads and writes.
///
/// Bundled so applying an operator takes the whole transition rather than five
/// individually orderable slices; nothing is copied, the fields are the same
/// borrows the caller already holds.
struct NumericTransition<'a> {
    /// The parent state's numeric values. Every effect operand is read from
    /// here, so the reads stay independent of effect order.
    parent_values: &'a [NumericValue],
    /// The successor's numeric values, updated in place.
    current_values: &'a mut [NumericValue],
    /// This operator's contribution to each cost variable.
    cost_part: &'a mut [NumericValue],
    /// The successor's packed buffer, updated in place.
    next_buffer: &'a mut [u64],
    /// The parent's packed buffer, read for effect conditions and operands.
    previous_buffer: &'a [u64],
}

/// Static counter for generating unique registry IDs.
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// Where a numeric variable's value lives in a registered state.
#[derive(Debug, Clone, Copy, PartialEq)]
enum NumericSlot {
    /// In the packed state, at this packer variable.
    Packed(usize),
    /// In the state's dense cost row, at this index.
    Cost(usize),
    /// Nowhere: the value is the same in every state.
    Constant(NumericValue),
    /// Nowhere: the arithmetic axioms recompute it from the others.
    Derived,
}

/// Central registry for managing planning states with deduplication and axiom evaluation.
///
/// The `StateRegistry` is responsible for:
/// - Creating and managing planning states.
/// - Deduplicating identical states to save memory.
/// - Evaluating axioms when states change.
/// - Managing numeric variables and cost information.
#[derive(Debug)]
pub struct StateRegistry {
    /// Unique identifier for this registry instance.
    id: usize,
    /// Shared handle to the planning task.
    task: Arc<Task>,
    /// Pool of state data, each entry is a packed state representation.
    state_data_pool: DataStorage,
    /// Scratch space successors are built in before they are registered.
    successor_bins: Vec<u64>,
    /// Registered state IDs indexed by the hash of their packed state. The
    /// table stores no duplicate hash key: lookup always verifies the exact
    /// packed state, so distinct states remain sound under hash collisions.
    registered_states: RegisteredStates,
    /// Dense row-major `NumericValue` cost values indexed by state ID. Cost variables
    /// are not part of state identity, but storing one allocation per state is
    /// unnecessary because state IDs are dense.
    cost_info: DenseCostInformation,
    /// Exact canonical-NumericValue interning for compact state storage. Regular
    /// numeric values are represented by checked 32-bit IDs in the packer.
    compact_numeric_values: CompactNumericValues,
    /// What the registry fixed at construction.
    layout: StateLayout,
}

#[derive(Debug, Default)]
struct CompactNumericValues {
    ids_by_value: HashMap<NumericValue, u32, MixedHasherBuilder>,
    values: Vec<NumericValue>,
}

impl CompactNumericValues {
    /// The id of `value`, interning it if it is new.
    fn intern(&mut self, value: NumericValue) -> u64 {
        if let Some(&id) = self.ids_by_value.get(&value) {
            return id as u64;
        }
        let id = u32::try_from(self.values.len())
            .expect("compact numeric state value table exceeds 32-bit ID capacity");
        self.values.push(value);
        self.ids_by_value.insert(value, id);
        id as u64
    }

    /// The value a packed regular numeric slot holds the ID of.
    fn value(&self, packed: u64) -> NumericValue {
        let id = usize::try_from(packed).expect("compact numeric value ID exceeds usize");
        *self
            .values
            .get(id)
            .unwrap_or_else(|| panic!("missing compact numeric value ID {id}"))
    }
}

impl StateRegistry {
    pub fn view<'s>(&'s self, state: &'s ConcreteState) -> ConcreteStateView<'s> {
        ConcreteStateView {
            packer: &self.layout.global_state_packer,
            prop: self.buffer(state.id()),
            backing: ConcreteStateViewBacking::Registered {
                state_id: state.id(),
                numeric_template: &self.layout.numeric_template,
                regular_numeric_slots: &self.layout.regular_numeric_slots,
                cost_numeric_indices: &self.layout.cost_numeric_indices,
                compact_numeric_values: &self.compact_numeric_values,
                cost_info: &self.cost_info,
                axiom_evaluator: &self.layout.axiom_evaluator,
            },
        }
    }

    /// Build the state packer, axiom evaluator, and registry for `task` in
    /// one step. This is the common construction path; use `Self::new`
    /// when a custom packer or axiom evaluator is needed.
    pub fn for_task(task: Arc<Task>) -> Self {
        let packer = Arc::new(StatePacker::from_abstract_task_with_numeric_range(
            &task,
            u32::MAX as u64,
        ));
        let axiom_evaluator = Arc::new(AxiomEvaluator::new(&task, packer.clone()));
        Self::new(task, packer, axiom_evaluator)
    }

    /// Create a new state registry for the given planning task.
    fn new(
        task: Arc<Task>,
        global_state_packer: Arc<StatePacker>,
        axiom_evaluator: Arc<AxiomEvaluator>,
    ) -> Self {
        let numeric_vars = task.numeric_variables();
        let number_numeric_vars = numeric_vars.len();
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);

        let numeric_var_types: Vec<NumericType> =
            numeric_vars.iter().map(|var| var.numeric_type()).collect();
        let initial_numeric_values = task.initial_numeric_values();
        assert_eq!(
            initial_numeric_values.len(),
            numeric_var_types.len(),
            "initial numeric state must contain one value per numeric variable"
        );
        let mut numeric_template = vec![NumericValue::new(0.0); number_numeric_vars];
        let mut regular_numeric_slots = Vec::new();
        let mut cost_numeric_indices = Vec::new();
        let mut next_regular_slot = global_state_packer.numeric_slot_offset();
        let mut next_cost_index = 0;
        let mut numeric_slots = Vec::with_capacity(number_numeric_vars);
        for (numeric_var_id, &ty) in numeric_var_types.iter().enumerate() {
            numeric_slots.push(match ty {
                NumericType::Constant => {
                    let value = initial_numeric_values[numeric_var_id];
                    numeric_template[numeric_var_id] = value;
                    NumericSlot::Constant(value)
                }
                NumericType::Regular => {
                    regular_numeric_slots.push((numeric_var_id, next_regular_slot));
                    next_regular_slot += 1;
                    NumericSlot::Packed(next_regular_slot - 1)
                }
                NumericType::Cost => {
                    cost_numeric_indices.push((numeric_var_id, next_cost_index));
                    next_cost_index += 1;
                    NumericSlot::Cost(next_cost_index - 1)
                }
                NumericType::Derived => NumericSlot::Derived,
            });
        }
        let cost_variable_count = numeric_var_types
            .iter()
            .filter(|&&ty| ty == NumericType::Cost)
            .count();
        let metric_use_metric = task.metric().use_metric();
        let metric_var = task.metric().var_id().and_then(|var_id| {
            numeric_var_types
                .get(var_id.index())
                .copied()
                .map(|ty| (var_id.index(), ty))
        });

        // Collect packer slots for the non-derived (input) variables. The
        // propositional slots are the ones below the packer's numeric offset;
        // the regular numeric variables follow in order from there.
        let numeric_slot_offset = global_state_packer.numeric_slot_offset();
        assert_eq!(
            numeric_slot_offset,
            task.variables().len(),
            "the packer's propositional section must cover exactly the task's propositional variables"
        );
        let mut non_derived_var_ids: Vec<usize> = Vec::new();
        let mut has_propositional_derived = false;
        for var_id in 0..numeric_slot_offset {
            let axiom_layer = task
                .variable(VariableIndex::from_usize(var_id))
                .axiom_layer();
            match axiom_layer {
                None => non_derived_var_ids.push(var_id),
                Some(_) => has_propositional_derived = true,
            }
        }
        let mut numeric_packer_index = numeric_slot_offset;
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
            state_data_pool,
            successor_bins: Vec::new(),
            registered_states: RegisteredStates::with_capacity(1024),
            cost_info: DenseCostInformation::new(cost_variable_count),
            compact_numeric_values: CompactNumericValues::default(),
            layout: StateLayout {
                global_state_packer,
                numeric_slots,
                axiom_evaluator,
                numeric_var_types,
                numeric_template,
                regular_numeric_slots,
                cost_numeric_indices,
                cost_variable_count,
                metric_var,
                metric_use_metric,
                metric_is_min,
                non_derived_bits_mask,
                has_axiom_derived_bits,
            },
        }
    }

    /// Return the unique ID of this registry.
    pub const fn id(&self) -> usize {
        self.id
    }

    /// Return the total number of distinct states registered in this registry.
    pub fn num_registered_states(&self) -> usize {
        self.state_data_pool.len()
    }

    /// Get the buffer at the specified index.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    pub fn buffer(&self, index: usize) -> &[u64] {
        self.state_data_pool
            .get(index)
            .expect("State index out of bounds")
    }

    /// Return a reference to the global state packer.
    pub fn global_state_packer(&self) -> &StatePacker {
        &self.layout.global_state_packer
    }

    fn num_state_bins(&self) -> usize {
        self.layout.global_state_packer.num_bins()
    }

    fn find_registered_state_id(&self, key: u64, bins: &[u64]) -> Option<StateID> {
        let num_bins = self.num_state_bins();
        self.registered_states
            .find(key, |&(existing_hash, compact_existing_id)| {
                existing_hash == key && {
                    let existing_id = compact_existing_id as StateID;
                    self.buffer(existing_id)[..num_bins] == bins[..num_bins]
                }
            })
            .map(|&(_, id)| id as StateID)
    }

    fn insert_registered_state_id(&mut self, key: u64, state_id: StateID) {
        let compact_state_id = u32::try_from(state_id)
            .unwrap_or_else(|_| panic!("registered state ID {state_id} exceeds u32"));
        self.registered_states.insert_unique(
            key,
            (key, compact_state_id),
            |&(existing_hash, _)| existing_hash,
        );
    }

    /// The hash key of `bins` and the id of the registered state it equals
    /// in the bits that identify a state: every bit, or only the non-derived
    /// ones when the task has derived bits. Axioms are deterministic
    /// functions of their inputs, so states equal in the non-derived bits
    /// are equal.
    fn lookup(&self, bins: &[u64]) -> (u64, Option<StateID>) {
        let num_bins = self.num_state_bins();
        let bins = &bins[..num_bins];
        if !self.layout.has_axiom_derived_bits {
            let key = fast_hash_bins(bins);
            return (key, self.find_registered_state_id(key, bins));
        }
        let mask = &self.layout.non_derived_bits_mask[..num_bins];
        let key = fast_hash_bins_masked(bins, mask);
        let existing = self
            .registered_states
            .find(key, |&(candidate_hash, compact_candidate)| {
                candidate_hash == key
                    && bins_eq_masked(
                        &self.buffer(compact_candidate as StateID)[..num_bins],
                        bins,
                        mask,
                    )
            })
            .map(|&(_, id)| id as StateID);
        (key, existing)
    }

    /// Registers `bins`, which [`Self::lookup`] found no state for under
    /// `key`, as a new state.
    fn register_new(&mut self, key: u64, bins: &[u64]) -> StateID {
        self.state_data_pool.push_back(bins);
        let state_id = self.state_data_pool.len() - 1;
        self.insert_registered_state_id(key, state_id);
        state_id
    }

    /// The id of the state `bins`, registering it if it is new, and whether
    /// it was.
    fn insert_or_find(&mut self, bins: &[u64]) -> (StateID, bool) {
        match self.lookup(bins) {
            (_, Some(existing_id)) => (existing_id, false),
            (key, None) => (self.register_new(key, bins), true),
        }
    }

    /// The id of the registered state with the input values of
    /// `propositional` and `numeric`, without registering anything.
    ///
    /// Both hold one value per variable of the task. Only the input
    /// variables identify a state: derived values are functions of them and
    /// cost values are not part of a state, so those entries are ignored.
    /// `buffer` is scratch space for the packed state.
    pub fn find_state_id(
        &self,
        propositional: &[ExplicitValueIndex],
        numeric: &[NumericValue],
        buffer: &mut Vec<u64>,
    ) -> Option<StateID> {
        assert_eq!(
            propositional.len(),
            self.layout.global_state_packer.numeric_slot_offset(),
            "a state lookup needs one value per propositional variable"
        );
        assert_eq!(
            numeric.len(),
            self.layout.numeric_var_types.len(),
            "a state lookup needs one value per numeric variable"
        );
        let num_bins = self.num_state_bins();
        buffer.clear();
        buffer.resize(num_bins, 0);
        self.pack_propositional_variables(buffer, propositional);
        {
            // A value no registered state holds has no compact id, and then
            // no registered state matches.
            let interner = &self.compact_numeric_values;
            for &(numeric_var_id, slot) in &self.layout.regular_numeric_slots {
                let id = *interner.ids_by_value.get(&numeric[numeric_var_id])?;
                self.layout
                    .global_state_packer
                    .set(buffer, slot, u64::from(id));
            }
        }
        if !self.layout.has_axiom_derived_bits {
            return self.find_registered_state_id(fast_hash_bins(buffer), buffer);
        }
        let mask = &self.layout.non_derived_bits_mask[..num_bins];
        let key = fast_hash_bins_masked(buffer, mask);
        self.registered_states
            .find(key, |&(candidate_hash, compact_candidate)| {
                candidate_hash == key
                    && bins_eq_masked(
                        &self.buffer(compact_candidate as StateID)[..num_bins],
                        buffer,
                        mask,
                    )
            })
            .map(|&(_, id)| id as StateID)
    }

    /// Create and registers the initial state of the planning problem.
    ///
    /// This method:
    /// 1. Packs propositional variables into the state buffer.
    /// 2. Processes numeric variables by type (regular, constant, cost, derived).
    /// 3. Evaluates axioms to compute derived values.
    /// 4. Registers the resulting state.
    pub fn register_initial_state(&mut self) -> ConcreteState {
        let mut init_buffer = vec![0u64; self.layout.global_state_packer.num_bins()];

        // Get copies of initial state values to avoid borrowing conflicts.
        let initial_propositional_values = self.task.initial_propositional_values().to_vec();
        let initial_numeric_values = self.task.initial_numeric_values().to_vec();

        // Pack propositional variables.
        self.pack_propositional_variables(&mut init_buffer, &initial_propositional_values);

        // Process numeric variables and get cost variables.
        let cost_variables = self.layout.process_numeric_variables(
            &mut self.compact_numeric_values,
            &mut init_buffer,
            &initial_numeric_values,
        );

        // Evaluate axioms.
        let mut numeric_state_copy = initial_numeric_values;
        self.evaluate_axioms(&mut init_buffer, &mut numeric_state_copy)
            .expect("Failed to evaluate axioms during initial state creation");

        let (state_id, _) = self.insert_or_find(&init_buffer);

        let init_state = ConcreteState::new(state_id);

        // Cost values are excluded from state identity and therefore live in
        // dense side storage. Install the row before any numeric state read.
        if self.layout.cost_variable_count > 0 {
            self.set_cost_information(&init_state, &cost_variables);
        }

        #[cfg(debug_assertions)]
        self.log_initial_state_info(&cost_variables);

        init_state
    }

    /// Pack propositional variables into the state buffer.
    fn pack_propositional_variables(
        &self,
        buffer: &mut [u64],
        initial_values: &[ExplicitValueIndex],
    ) {
        for (i, &value) in initial_values.iter().enumerate() {
            self.layout
                .global_state_packer
                .set(buffer, i, value.index() as u64);
        }
    }

    /// Evaluate axioms on the given state.
    fn evaluate_axioms(
        &self,
        buffer: &mut [u64],
        numeric_state: &mut [NumericValue],
    ) -> Result<(), StateInsertError> {
        self.layout
            .axiom_evaluator
            .evaluate(buffer, numeric_state)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate axioms: {:?}", e),
            })?;

        Ok(())
    }

    /// Log initial state information in debug builds.
    #[cfg(debug_assertions)]
    fn log_initial_state_info(&self, cost_variables: &[NumericValue]) {
        use tracing::info;

        let regular_count = self
            .layout
            .numeric_slots
            .iter()
            .filter(|slot| matches!(slot, NumericSlot::Packed(_)))
            .count();
        let constant_count = self
            .layout
            .numeric_slots
            .iter()
            .filter(|slot| matches!(slot, NumericSlot::Constant(_)))
            .count();
        let derived_count = self
            .task
            .numeric_variables()
            .iter()
            .filter(|var| var.numeric_type() == NumericType::Derived)
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
    ///
    /// `numeric_values` is indexed by numeric-variable id, so it accepts a full
    /// vector as returned by [`Self::numeric_values`]. Constant entries are
    /// checked against the values this registry already holds and derived
    /// entries are recomputed by the axiom pass; see
    /// `Self::process_register_numeric_variables`.
    pub fn register_state(
        &mut self,
        values: Vec<u64>,
        numeric_values: Vec<NumericValue>,
    ) -> Result<ConcreteState, StateInsertError> {
        self.register_state_with_status(values, numeric_values)
            .map(|(state, _is_new)| state)
    }

    fn register_state_with_status(
        &mut self,
        values: Vec<u64>,
        numeric_values: Vec<NumericValue>,
    ) -> Result<(ConcreteState, bool), StateInsertError> {
        let mut buffer = vec![0; self.layout.global_state_packer.num_bins()];

        // Pack propositional variables.
        for (i, &value) in values.iter().enumerate() {
            self.layout.global_state_packer.set(&mut buffer, i, value);
        }

        // Process numeric variables.
        let cost_variables = self.layout.process_register_numeric_variables(
            &mut self.compact_numeric_values,
            &mut buffer,
            &numeric_values,
        )?;

        // Evaluate axioms
        let mut numeric_values_copy = numeric_values;
        self.evaluate_axioms(&mut buffer, &mut numeric_values_copy)?;

        let (id, is_new_state) = self.insert_or_find(&buffer);

        let new_state = ConcreteState::new(id);

        // Handle cost information based on whether this is a new or existing state.
        if is_new_state {
            // New state: store cost information.
            if !cost_variables.is_empty() {
                self.set_cost_information(&new_state, &cost_variables);
            }
        } else {
            // Existing state: use metric optimization to determine which cost info to keep.
            let keep_old_cost_information =
                self.should_keep_old_cost_information(&new_state, &numeric_values_copy);

            match keep_old_cost_information {
                Ok(false) => {
                    self.set_cost_information(&new_state, &cost_variables);
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

    /// The successor of `state` under `operator`, registered, and the
    /// operator's metric cost in `state`. Expanding all successors of one
    /// state, [`Self::build_expansion_context`] and
    /// [`Self::apply_operator_in_context`] decode the state once.
    pub fn successor(
        &mut self,
        state: &ConcreteState,
        operator: &Operator,
    ) -> Result<(ConcreteState, f64), StateInsertError> {
        let mut context = ExpansionContext::default();
        self.build_expansion_context(state, &mut context)?;
        self.apply_operator_in_context(state, operator, &context, &mut Vec::new(), &mut Vec::new())
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
        self.view(parent)
            .fill_numeric(&mut ctx.parent_numeric)
            .map_err(|e| StateInsertError {
                message: format!("Failed to get numeric variables: {:?}", e),
            })?;
        self.fill_cost_information(parent, &mut ctx.parent_cost);
        let expected_cost_vars = self.countcost_variables();
        if ctx.parent_cost.len() < expected_cost_vars {
            ctx.parent_cost
                .resize(expected_cost_vars, NumericValue::new(0.0));
        }
        ctx.parent_metric = if self.layout.metric_use_metric {
            self.evaluate_metric(&ctx.parent_numeric)
        } else {
            NumericValue::new(0.0)
        };
        Ok(())
    }

    /// Apply `operator` to `parent`, whose decoded values, cost row and
    /// metric value `ctx` holds, so that the successors of one expansion
    /// share one decoding of the parent.
    pub fn apply_operator_in_context(
        &mut self,
        parent: &ConcreteState,
        operator: &Operator,
        ctx: &ExpansionContext,
        successor_values: &mut Vec<NumericValue>,
        cost_values: &mut Vec<NumericValue>,
    ) -> Result<(ConcreteState, f64), StateInsertError> {
        // Seed successor scratch from the cached parent values; the numeric
        // and cost effects below will mutate them in place.
        successor_values.clear();
        successor_values.extend_from_slice(&ctx.parent_numeric);
        cost_values.clear();
        cost_values.extend_from_slice(&ctx.parent_cost);

        // The successor is built in scratch space and enters the pool only
        // if it is new.
        let mut next_buffer = std::mem::take(&mut self.successor_bins);
        next_buffer.clear();
        // Borrowing the pool field alone leaves the interner free for the
        // effects to write.
        let previous_buffer = self
            .state_data_pool
            .get(parent.id())
            .expect("the parent is a registered state");
        next_buffer.extend_from_slice(previous_buffer);

        self.layout
            .apply_propositional_effects(&mut next_buffer, previous_buffer, operator);

        // Skip the comparison/propositional axiom passes during effect
        // application when the task has axiom-derived bits we can mask off.
        // We dedup using only non-derived bits below; if the successor is a
        // duplicate, the existing state already has correct derived bits and
        // we save the (typically expensive) axiom evaluation. If the
        // successor is new, we run the full axiom pass after dedup.
        let defer_full_axioms = self.layout.has_axiom_derived_bits;
        self.layout.apply_numeric_effects_inner(
            &mut self.compact_numeric_values,
            NumericTransition {
                parent_values: &ctx.parent_numeric,
                current_values: successor_values,
                cost_part: cost_values,
                next_buffer: &mut next_buffer,
                previous_buffer,
            },
            operator,
            !defer_full_axioms,
        )?;

        // Compute `op_cost` from the cached parent metric instead of going
        // back through `metric_value_for_state(parent)`. Compute new_metric
        // even if we're not using the metric (in that case we just return
        // 1.0 for `op_cost` but still want to pre-fill the metric cache).
        let new_metric = if self.layout.metric_use_metric {
            self.evaluate_metric(successor_values)
        } else {
            NumericValue::new(0.0)
        };
        let op_cost = if self.layout.metric_use_metric {
            self.task
                .metric()
                .step_cost(ctx.parent_metric.value(), new_metric.value())
        } else {
            1.0
        };

        let (id, is_new_state) = match self.lookup(&next_buffer) {
            (_, Some(existing_id)) => (existing_id, false),
            (key, None) => {
                // The arithmetic axioms already ran with the effects.
                if defer_full_axioms {
                    self.layout
                        .axiom_evaluator
                        .evaluate_derived_facts(&mut next_buffer, successor_values);
                }
                (self.register_new(key, &next_buffer), true)
            }
        };
        self.successor_bins = next_buffer;
        let successor = ConcreteState::new(id);

        // Cost-info bookkeeping. For tasks without `Cost`-typed numeric
        // variables this is entirely a no-op, and we skip the whole branch.
        // For tasks where the metric variable is itself a `Regular` numeric
        // var, duplicates produced by the masked dedup are guaranteed to
        // have identical metric values (the mask covers the metric's bits),
        // so `should_keep_old_cost_information` is always `false` and we
        // can avoid that read on every duplicate.
        //
        // Cost values use dense row-major storage, so duplicate comparison is
        // a direct indexed load without per-state allocation or hashing.
        if self.layout.cost_variable_count > 0 {
            if is_new_state {
                assert_eq!(
                    cost_values.len(),
                    self.layout.cost_variable_count,
                    "successor must define every f64 cost variable"
                );
                self.set_cost_information(&successor, cost_values);
            } else {
                let metric_is_regular = matches!(
                    self.layout.metric_var,
                    Some((_, NumericType::Regular)) | None
                );
                let keep_old = if !self.layout.metric_use_metric {
                    false
                } else if metric_is_regular {
                    // Masked dedup guarantees metric bits agree.
                    false
                } else {
                    let old_metric = self.metric_value_for_state(&successor).map_err(|error| {
                        StateInsertError {
                            message: format!("Failed to select cost information: {error:?}"),
                        }
                    })?;
                    if self.layout.metric_is_min {
                        old_metric < new_metric
                    } else {
                        old_metric > new_metric
                    }
                };

                if !keep_old {
                    self.set_cost_information(&successor, cost_values);
                }
            }
        }

        Ok((successor, op_cost))
    }

    /// Count the number of cost variables in the planning task.
    /// Returns the cached count populated at construction.
    #[inline]
    fn countcost_variables(&self) -> usize {
        self.layout.cost_variable_count
    }

    fn fill_cost_information(&self, state: &ConcreteState, output: &mut Vec<NumericValue>) {
        let cost_info_borrow = &self.cost_info;
        let cost_info_data = cost_info_borrow.get(state.id());
        output.resize(cost_info_data.len(), NumericValue::new(0.0));
        output.copy_from_slice(cost_info_data);
    }

    /// Retrieve all numeric variable values for a given state.
    ///
    /// This method reconstructs the complete numeric state by:
    /// - Reading regular variables from the packed state buffer.
    /// - Using stored constants for constant variables.
    /// - Retrieving cost variables from per-state storage.
    /// - Evaluating arithmetic axioms to compute derived values.
    pub fn numeric_values(
        &self,
        state: &ConcreteState,
    ) -> Result<Vec<NumericValue>, AssignmentAxiomError> {
        let mut result = Vec::with_capacity(self.layout.numeric_template.len());
        self.view(state).fill_numeric(&mut result)?;
        Ok(result)
    }

    /// Evaluate the metric value for a given numeric state.
    ///
    /// This corresponds to the C++ evaluate_metric function that retrieves
    /// the value of the metric fluent from the numeric state.
    pub fn evaluate_metric(&self, numeric_state: &[NumericValue]) -> NumericValue {
        match self.layout.metric_var {
            Some((metric, _)) => numeric_state[metric],
            None => NumericValue::new(0.0),
        }
    }

    fn metric_value_for_state(
        &self,
        state: &ConcreteState,
    ) -> Result<NumericValue, AssignmentAxiomError> {
        let Some((metric_fluent_id, metric_type)) = self.layout.metric_var else {
            return Ok(NumericValue::new(0.0));
        };

        match metric_type {
            NumericType::Regular => {
                let buffer = state.buffer(self);
                Ok(self.compact_numeric_values.value(
                    self.layout
                        .global_state_packer
                        .get(buffer, self.layout.packed_slot(metric_fluent_id)),
                ))
            }
            NumericType::Cost => {
                let cost_index = self.layout.cost_index(metric_fluent_id);
                let cost_info_borrow = &self.cost_info;
                let cost_values = cost_info_borrow.get(state.id());
                Ok(cost_values[cost_index])
            }
            NumericType::Constant => {
                let NumericSlot::Constant(value) = self.layout.numeric_slots[metric_fluent_id]
                else {
                    unreachable!("a constant numeric variable has a constant slot")
                };
                Ok(value)
            }
            NumericType::Derived => {
                let numeric_vals = self.numeric_values(state)?;
                Ok(self.evaluate_metric(&numeric_vals))
            }
        }
    }

    /// Determine which cost information to keep when states are deduplicated.
    ///
    /// This implements the C++ logic for metric optimization when duplicate states are found.
    /// The stored information survives exactly when its metric value is the
    /// better of the two under the task's optimization direction; ties keep the
    /// new one, matching the C++ strict comparison.
    fn should_keep_old_cost_information(
        &self,
        existing_state: &ConcreteState,
        successor_numeric_vals: &[NumericValue],
    ) -> Result<bool, AssignmentAxiomError> {
        if !self.task.metric().use_metric() {
            return Ok(false);
        }

        let old_metric_val = self.metric_value_for_state(existing_state)?;
        let new_metric_val = self.evaluate_metric(successor_numeric_vals);

        Ok(if self.task.metric().is_min() {
            old_metric_val < new_metric_val
        } else {
            old_metric_val > new_metric_val
        })
    }

    /// Get cost information for a given state.
    ///
    /// This corresponds to the C++ `g_cost_information[state]` access pattern.
    /// Return an empty vector if no cost information is stored for the state.
    #[cfg(test)]
    pub(crate) fn get_cost_information(&self, state: &ConcreteState) -> Vec<NumericValue> {
        self.cost_info.get(state.id()).to_vec()
    }

    /// Set cost information for a given state.
    ///
    fn set_cost_information(&mut self, state: &ConcreteState, values: &[NumericValue]) {
        self.cost_info.set(state.id(), values);
    }
}
