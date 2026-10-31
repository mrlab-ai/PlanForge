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
use crate::explicit_state::ExplicitState;
use crate::numeric_task::{
    Effect, ExplicitFact, ExplicitValueIndex, NumericValue, Operator, TaskRef, VariableIndex,
};
use crate::utils::errors::{
    AssignmentAxiomError, AxiomEvalError, InvalidIndex, StateInsertError, StateNotFoundError,
};
use crate::utils::segmented_vector::SegmentedArrayVector;
use crate::{numeric_task::NumericType, utils::state_packer::StatePacker};
use hashbrown::HashTable;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Fast mixer for exact `f64` bit patterns. Raw IEEE-754 bits are not
/// well-distributed in their low bits (consecutive integers share long zero
/// suffixes), so a pass-through hasher makes the compact-value table
/// degenerate into long probe chains.
#[derive(Default)]
struct MixedU64Hasher(u64);

impl Hasher for MixedU64Hasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        if bytes.len() == 8 {
            self.0 = u64::from_ne_bytes(bytes.try_into().unwrap());
        } else {
            for &byte in bytes {
                self.0 = self.0.rotate_left(5) ^ u64::from(byte);
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
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

type MixedHasherBuilder = BuildHasherDefault<MixedU64Hasher>;

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

/// Represent a concrete state in the planning problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConcreteState {
    pool_offset: usize,
}

/// A short-lived, read-only view of concrete state values.
///
/// Registered states borrow their registry only for the lifetime of the view;
/// decoded CEGAR states borrow their numeric slice directly. The type cannot
/// outlive either backing store and therefore does not create the long-lived
/// aliasing problem of a `(StateID, &StateRegistry)` state handle.
#[derive(Clone, Copy)]
pub struct ConcreteStateView<'a> {
    packer: &'a StatePacker,
    prop: &'a [u64],
    backing: ConcreteStateViewBacking<'a>,
}

#[derive(Clone, Copy)]
enum ConcreteStateViewBacking<'a> {
    Decoded(&'a [NumericValue]),
    Registered {
        state_id: StateID,
        numeric_template: &'a [NumericValue],
        regular_numeric_slots: &'a [(usize, usize)],
        cost_numeric_indices: &'a [(usize, usize)],
        compact_numeric_values: &'a CompactNumericValues,
        cost_info: &'a DenseCostInformation,
        axiom_evaluator: &'a AxiomEvaluator<'a>,
    },
}

impl<'a> ConcreteStateView<'a> {
    pub fn from_decoded(
        packer: &'a StatePacker,
        prop: &'a [u64],
        numeric: &'a [NumericValue],
    ) -> Self {
        Self {
            packer,
            prop,
            backing: ConcreteStateViewBacking::Decoded(numeric),
        }
    }
}

impl ExplicitFact {
    pub fn is_held(&self, state: ConcreteStateView<'_>) -> bool {
        let value = state.packer().get(state.propositional(), self.var());
        value == self.value() as u64
    }
}

impl Effect {
    pub fn conditions_met(&self, state: ConcreteStateView<'_>) -> bool {
        state.holds_all(self.conditions())
    }
}

impl<'a> ConcreteStateView<'a> {
    pub fn packer(self) -> &'a StatePacker {
        self.packer
    }

    pub fn propositional(self) -> &'a [u64] {
        self.prop
    }

    /// Whether every one of `facts` holds.
    pub fn holds_all(self, facts: &[ExplicitFact]) -> bool {
        facts.iter().all(|fact| fact.is_held(self))
    }

    /// The `facts` that do not hold, in order.
    pub fn unsatisfied<'f>(
        self,
        facts: &'f [ExplicitFact],
    ) -> impl Iterator<Item = &'f ExplicitFact> + use<'a, 'f> {
        facts.iter().filter(move |fact| !fact.is_held(self))
    }

    pub fn fill_propositional(self, output: &mut Vec<ExplicitValueIndex>) {
        output.clear();
        output.extend(
            (0..self.packer.numeric_slot_offset()).map(|slot| {
                ExplicitValueIndex::from_usize(self.packer.get(self.prop, slot) as usize)
            }),
        );
    }

    pub fn decoded_numeric(self) -> Option<&'a [NumericValue]> {
        match self.backing {
            ConcreteStateViewBacking::Decoded(values) => Some(values),
            ConcreteStateViewBacking::Registered { .. } => None,
        }
    }

    /// Writes the numeric values into `output`, the derived ones evaluated by
    /// the arithmetic axioms.
    pub fn fill_numeric(self, output: &mut Vec<NumericValue>) -> Result<(), AssignmentAxiomError> {
        match self.backing {
            ConcreteStateViewBacking::Decoded(values) => {
                output.clear();
                output.extend_from_slice(values);
            }
            ConcreteStateViewBacking::Registered {
                axiom_evaluator, ..
            } => {
                self.fill_numeric_unevaluated(output);
                if axiom_evaluator.has_numeric_axioms() {
                    axiom_evaluator.evaluate_arithmetic_axioms(output)?;
                }
            }
        }
        Ok(())
    }

    /// Writes the numeric values into `output` before the arithmetic axioms:
    /// the regular, constant and cost values, with every derived value zero.
    ///
    /// # Panics
    /// Panics on a decoded view, whose values are already closed under the
    /// axioms.
    pub fn fill_numeric_unevaluated(self, output: &mut Vec<NumericValue>) {
        let ConcreteStateViewBacking::Registered {
            state_id,
            numeric_template,
            regular_numeric_slots,
            cost_numeric_indices,
            compact_numeric_values,
            cost_info,
            ..
        } = self.backing
        else {
            panic!("a decoded state view has no numeric values before the axioms");
        };
        output.clear();
        output.extend_from_slice(numeric_template);
        let interned = compact_numeric_values;
        for &(out_idx, packed_slot) in regular_numeric_slots {
            output[out_idx] = interned.value(self.packer.get(self.prop, packed_slot));
        }
        let cost_values = cost_info.get(state_id);
        for &(out_idx, cost_idx) in cost_numeric_indices {
            output[out_idx] = cost_values[cost_idx];
        }
    }

    /// Writes the propositional and numeric values, derived ones included,
    /// into `state`.
    pub fn fill_explicit(self, state: &mut ExplicitState) -> Result<(), AssignmentAxiomError> {
        self.fill_propositional(&mut state.propositional);
        self.fill_numeric(&mut state.numeric)
    }
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
    pub fn get_state(&self, state_registry: &StateRegistry) -> Vec<ExplicitValueIndex> {
        let mut values = Vec::new();
        state_registry.view(self).fill_propositional(&mut values);
        values
    }

    /// Fill `output` with the propositional state values without allocating a new vector.
    pub fn fill_state(&self, state_registry: &StateRegistry, output: &mut Vec<ExplicitValueIndex>) {
        state_registry.view(self).fill_propositional(output);
    }

    /// Get the numeric state values for regular variables.
    pub fn get_numeric_state(&self, state_registry: &StateRegistry) -> Vec<NumericValue> {
        let buffer = self.buffer(state_registry);
        let packer = &state_registry.layout.global_state_packer;
        let interned = &state_registry.compact_numeric_values;
        state_registry
            .layout
            .regular_numeric_slots
            .iter()
            .map(|&(_, packed_slot)| interned.value(packer.get(buffer, packed_slot)))
            .collect()
    }

    /// Get a reference to the underlying buffer for this state.
    pub fn buffer<'a>(&self, state_registry: &'a StateRegistry) -> &'a [u64] {
        state_registry.get_buffer(self.pool_offset)
    }
}

/// MurmurHash3's 64-bit finalizer (`fmix64`). Spreads bits well across both halves of the output
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

/// 64-bit FNV-1a over the bytes of `words`, finished with [`finalize_mix`].
///
/// Hashes byte by byte rather than word by word: the word variant produces
/// noticeably more hashbrown bucket collisions on packed state buffers, whose
/// bins share long stretches of zero bits, and comparing the buffers of a
/// grown bucket dominates deduplication. The extra multiplications per bin
/// are cheap by comparison.
#[inline]
fn fnv1a_words(words: impl Iterator<Item = u64>) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;
    let mut hash = FNV_OFFSET;
    for word in words {
        for byte in word.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    finalize_mix(hash)
}

#[inline]
fn fast_hash_bins(bins: &[u64]) -> u64 {
    fnv1a_words(bins.iter().copied())
}

/// The hash of only the bits the per-bin `mask` selects, to deduplicate
/// buffers whose bins also hold derived (axiom-computed) bits.
#[inline]
fn fast_hash_bins_masked(bins: &[u64], mask: &[u64]) -> u64 {
    debug_assert_eq!(bins.len(), mask.len());
    fnv1a_words(bins.iter().zip(mask).map(|(&bin, &mask)| bin & mask))
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
pub struct StateRegistry<'a> {
    /// Unique identifier for this registry instance.
    id: usize,
    /// Shared handle to the planning task.
    task: TaskRef<'a>,
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
    layout: StateLayout<'a>,
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

impl<'a> StateRegistry<'a> {
    pub fn view<'s>(&'s self, state: &'s ConcreteState) -> ConcreteStateView<'s>
    where
        'a: 's,
    {
        ConcreteStateView {
            packer: &self.layout.global_state_packer,
            prop: self.get_buffer(state.get_id()),
            backing: ConcreteStateViewBacking::Registered {
                state_id: state.get_id(),
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
    pub fn for_task(task: TaskRef<'a>) -> Self {
        let packer = Arc::new(StatePacker::from_abstract_task_with_numeric_range(
            &*task,
            u32::MAX as u64,
        ));
        let axiom_evaluator = Arc::new(AxiomEvaluator::new(task.clone(), packer.clone()));
        Self::new(task, packer, axiom_evaluator)
    }

    /// Create a new state registry for the given planning task.
    fn new(
        task: TaskRef<'a>,
        global_state_packer: Arc<StatePacker>,
        axiom_evaluator: Arc<AxiomEvaluator<'a>>,
    ) -> Self {
        let numeric_vars = task.numeric_variables();
        let number_numeric_vars = numeric_vars.len();
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);

        let numeric_var_types: Vec<NumericType> =
            numeric_vars.iter().map(|var| *var.get_type()).collect();
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
    pub fn get_buffer(&self, index: usize) -> &[u64] {
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
                    self.get_buffer(existing_id)[..num_bins] == bins[..num_bins]
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
                        &self.get_buffer(compact_candidate as StateID)[..num_bins],
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
                        &self.get_buffer(compact_candidate as StateID)[..num_bins],
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
    pub fn get_initial_state(&mut self) -> ConcreteState {
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
    ///
    /// `numeric_values` is indexed by numeric-variable id, so it accepts a full
    /// vector as returned by [`Self::get_numeric_vars`]. Constant entries are
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

    /// Generates a successor state by applying an operator to the current state
    pub fn get_successor_state(
        &mut self,
        current_state: &ConcreteState,
        operator: &Operator,
    ) -> Result<ConcreteState, StateInsertError> {
        let mut successor_values = Vec::new();
        let mut cost_values = Vec::new();
        self.get_successor_state_with_buffers_and_cost(
            current_state,
            operator,
            &mut successor_values,
            &mut cost_values,
        )
        .map(|(successor, _)| successor)
    }

    pub fn get_successor_state_with_buffers_and_cost(
        &mut self,
        current_state: &ConcreteState,
        operator: &Operator,
        successor_values: &mut Vec<NumericValue>,
        cost_values: &mut Vec<NumericValue>,
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
                .map_err(|e| StateInsertError {
                    message: format!("Failed to evaluate metric for parent state: {e:?}"),
                })?
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
            .get(parent.get_id())
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
                .map_err(|e| StateInsertError {
                    message: format!("Failed to evaluate metric: {e:?}"),
                })?
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
                if defer_full_axioms {
                    self.layout
                        .axiom_evaluator
                        .evaluate(&mut next_buffer, successor_values)
                        .map_err(|e| StateInsertError {
                            message: format!("Failed to evaluate axioms: {:?}", e),
                        })?;
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
        let cost_info_data = cost_info_borrow.get(state.get_id());
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
    pub fn get_numeric_vars(
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
    pub fn evaluate_metric(
        &self,
        numeric_state: &[NumericValue],
    ) -> Result<NumericValue, InvalidIndex> {
        match self.layout.metric_var {
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
            None => Ok(NumericValue::new(0.0)),
        }
    }

    fn metric_value_for_state(
        &self,
        state: &ConcreteState,
    ) -> Result<NumericValue, AxiomEvalError> {
        let Some((metric_fluent_id, metric_type)) = self.layout.metric_var else {
            return Ok(NumericValue::new(0.0));
        };
        if metric_fluent_id >= self.layout.numeric_var_types.len() {
            return Err(AxiomEvalError::InvalidIndex(InvalidIndex {
                length: self.layout.numeric_var_types.len(),
                index: metric_fluent_id,
            }));
        }

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
                let cost_values = cost_info_borrow.get(state.get_id());
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
                let numeric_vals = self
                    .get_numeric_vars(state)
                    .map_err(AxiomEvalError::Assignment)?;
                self.evaluate_metric(&numeric_vals)
                    .map_err(AxiomEvalError::InvalidIndex)
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
    ) -> Result<bool, AxiomEvalError> {
        if !self.task.metric().use_metric() {
            return Ok(false);
        }

        let old_metric_val = self.metric_value_for_state(existing_state)?;
        let new_metric_val = self
            .evaluate_metric(successor_numeric_vals)
            .map_err(AxiomEvalError::InvalidIndex)?;

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
        self.cost_info.get(state.get_id()).to_vec()
    }

    /// Set cost information for a given state.
    ///
    fn set_cost_information(&mut self, state: &ConcreteState, values: &[NumericValue]) {
        self.cost_info.set(state.get_id(), values);
    }
}

/// What a registry fixes at construction: the packer, where each numeric
/// variable lives, the metric, and the axioms closing a state.
struct StateLayout<'a> {
    /// Axiom evaluator for handling derived predicates and numeric axioms.
    axiom_evaluator: Arc<AxiomEvaluator<'a>>,
    /// State packer for efficient bit-level state representation.
    global_state_packer: Arc<StatePacker>,
    /// Where each numeric variable's value lives, fixed at construction.
    numeric_slots: Vec<NumericSlot>,
    /// Snapshot of the numeric variable layout, populated at construction.
    /// Iterating this avoids per-call vtable dispatch through `task.numeric_variables()`
    /// in hot paths like numeric decoding.
    numeric_var_types: Vec<NumericType>,
    /// Initial numeric row with immutable constants in place and every
    /// state-dependent or derived entry zeroed. Numeric-state materialization
    /// copies this row before filling the two state-dependent layouts below.
    numeric_template: Vec<NumericValue>,
    /// `(numeric variable id, packed-state slot)` for regular variables.
    regular_numeric_slots: Vec<(usize, usize)>,
    /// `(numeric variable id, dense cost-row slot)` for cost variables.
    cost_numeric_indices: Vec<(usize, usize)>,
    /// Cached count returned by `countcost_variables`. Constant for the task.
    cost_variable_count: usize,
    /// Cached `task.metric().var_id()` and the metric variable's type when set.
    /// Used to short-circuit `metric_value_for_state` for the common Regular case.
    metric_var: Option<(usize, NumericType)>,
    /// Whether the task uses a metric (cached `task.metric().use_metric()`).
    metric_use_metric: bool,
    /// Whether the task is a minimization (cached `task.metric().is_min()`).
    metric_is_min: bool,
    /// Per-bin mask covering only the bits owned by non-derived (input)
    /// variables: regular propositional vars (axiom_layer == None) and regular
    /// numeric vars. `StateRegistry::lookup` uses it to deduplicate
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

impl<'a> StateLayout<'a> {
    /// Packs the regular numeric values into `buffer` and returns the cost
    /// values.
    fn process_numeric_variables(
        &self,
        interner: &mut CompactNumericValues,
        buffer: &mut [u64],
        initial_numeric_values: &[NumericValue],
    ) -> Vec<NumericValue> {
        let mut cost_variables = Vec::with_capacity(self.cost_variable_count);
        for (&value, &slot) in initial_numeric_values.iter().zip(&self.numeric_slots) {
            match slot {
                NumericSlot::Packed(packed_slot) => {
                    self.global_state_packer
                        .set(buffer, packed_slot, interner.intern(value));
                }
                NumericSlot::Cost(cost_index) => {
                    debug_assert_eq!(cost_index, cost_variables.len());
                    cost_variables.push(value);
                }
                NumericSlot::Constant(_) | NumericSlot::Derived => {}
            }
        }
        cost_variables
    }

    /// Process numeric variables during state registration.
    ///
    /// Only regular values enter the packed buffer and only cost values are
    /// returned; constant values are the same in every state and must equal
    /// the task's, and derived values are a function of the inputs that
    /// [`Self::evaluate_axioms`] recomputes right after this call.
    fn process_register_numeric_variables(
        &self,
        interner: &mut CompactNumericValues,
        buffer: &mut [u64],
        numeric_values: &[NumericValue],
    ) -> Result<Vec<NumericValue>, StateInsertError> {
        assert_eq!(
            numeric_values.len(),
            self.numeric_slots.len(),
            "a state has one value per numeric variable"
        );
        let mut cost_variables = Vec::with_capacity(self.cost_variable_count);
        for (numeric_var_id, (&value, &slot)) in
            numeric_values.iter().zip(&self.numeric_slots).enumerate()
        {
            match slot {
                NumericSlot::Packed(packed_slot) => {
                    self.global_state_packer
                        .set(buffer, packed_slot, interner.intern(value));
                }
                NumericSlot::Cost(cost_index) => {
                    debug_assert_eq!(cost_index, cost_variables.len());
                    cost_variables.push(value);
                }
                NumericSlot::Constant(constant) => {
                    if constant != value {
                        return Err(StateInsertError {
                            message: format!(
                                "constant numeric variable {numeric_var_id} is {constant}, but the state to register has {value}"
                            ),
                        });
                    }
                }
                NumericSlot::Derived => {}
            }
        }
        Ok(cost_variables)
    }

    /// Whether every condition of a conditional assignment effect holds in the
    /// packed state `buffer`.
    ///
    /// Reads `buffer` directly instead of going through `ExplicitFact::is_hold`
    /// so that callers holding only the parent's raw buffer (and no
    /// `ConcreteState` for it) can check conditions on the hot successor path.
    #[inline]
    fn assignment_conditions_met(&self, conditions: &[ExplicitFact], buffer: &[u64]) -> bool {
        conditions.iter().all(|condition| {
            self.global_state_packer.get(buffer, condition.var()) == condition.value() as u64
        })
    }

    /// Apply propositional effects of an operator to the state buffer.
    fn apply_propositional_effects(
        &self,
        buffer: &mut [u64],
        previous_buffer: &[u64],
        operator: &Operator,
    ) {
        for effect in operator.effects() {
            if self.assignment_conditions_met(effect.conditions(), previous_buffer) {
                let var_id = effect.var_id();
                let value = effect.value().index() as u64;
                self.global_state_packer.set(buffer, var_id.index(), value);
            }
        }
    }

    /// The packer variable of the regular numeric variable `numeric_var_id`.
    fn packed_slot(&self, numeric_var_id: usize) -> usize {
        match self.numeric_slots[numeric_var_id] {
            NumericSlot::Packed(packed_slot) => packed_slot,
            slot => panic!("regular numeric variable {numeric_var_id} lives in {slot:?}"),
        }
    }

    fn cost_index(&self, numeric_var_id: usize) -> usize {
        match self.numeric_slots[numeric_var_id] {
            NumericSlot::Cost(cost_index) => cost_index,
            slot => panic!("cost numeric variable {numeric_var_id} lives in {slot:?}"),
        }
    }

    /// Applies `operator`'s numeric effects, interning the new regular values
    /// in `interner`, and closes the successor under the axioms. Without
    /// `run_full_axioms` only the arithmetic axioms run, which the metric
    /// needs; the caller then runs the comparison and propositional passes
    /// once the successor turns out to be new.
    fn apply_numeric_effects_inner(
        &self,
        interner: &mut CompactNumericValues,
        transition: NumericTransition<'_>,
        operator: &Operator,
        run_full_axioms: bool,
    ) -> Result<(), StateInsertError> {
        let NumericTransition {
            parent_values,
            current_values,
            cost_part,
            next_buffer,
            previous_buffer,
        } = transition;
        // The effects' semantics -- simultaneous, operands read in the
        // parent, additive repeats accumulated -- are the operator's; this
        // only stores each result where its variable lives. Effect
        // conditions read the parent's buffer.
        let packer = &*self.global_state_packer;
        operator.apply_numeric_effects(
            |conditions| self.assignment_conditions_met(conditions, previous_buffer),
            parent_values,
            current_values,
            |variable, result| match self.numeric_slots[variable.index()] {
                NumericSlot::Packed(packed_slot) => {
                    packer.set(next_buffer, packed_slot, interner.intern(result));
                }
                NumericSlot::Cost(cost_index) => cost_part[cost_index] = result,
                slot @ (NumericSlot::Constant(_) | NumericSlot::Derived) => panic!(
                    "an effect of {} writes numeric variable {variable}, which lives in {slot:?}",
                    operator.name()
                ),
            },
        );

        if run_full_axioms {
            self.axiom_evaluator
                .evaluate(next_buffer, current_values)
                .map_err(|e| StateInsertError {
                    message: format!("Failed to evaluate axioms: {:?}", e),
                })?;
        } else {
            self.axiom_evaluator
                .evaluate_arithmetic_axioms(current_values)
                .map_err(|e| StateInsertError {
                    message: format!("Failed to evaluate arithmetic axioms: {:?}", e),
                })?;
        }

        Ok(())
    }
}
