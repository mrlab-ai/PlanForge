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
use crate::numeric_task::{AssignmentOperation, ExplicitFact, Operator, RepeatedTarget, TaskRef};
use crate::utils::errors::{
    AssignmentAxiomError, AxiomEvalError, InvalidIndex, StateInsertError, StateNotFoundError,
};
use crate::utils::float_tolerance;
use crate::utils::segmented_vector::SegmentedArrayVector;
use crate::{numeric_task::NumericType, utils::state_packer::StatePacker};
use hashbrown::HashTable;
use std::cell::RefCell;
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
    values: Vec<f64>,
    num_cost_variables: usize,
}

impl DenseCostInformation {
    fn new(num_cost_variables: usize) -> Self {
        Self {
            values: Vec::new(),
            num_cost_variables,
        }
    }

    fn get(&self, state_id: StateID) -> &[f64] {
        if self.num_cost_variables == 0 {
            return &[];
        }
        let start = state_id
            .checked_mul(self.num_cost_variables)
            .expect("cost-information index overflow");
        let end = start + self.num_cost_variables;
        self.values.get(start..end).unwrap_or_else(|| {
            panic!("missing f64 cost information for registered state {state_id}")
        })
    }

    fn set(&mut self, state_id: StateID, values: &[f64]) {
        assert_eq!(
            values.len(),
            self.num_cost_variables,
            "cost-information row has the wrong number of f64 values"
        );
        let start = state_id
            .checked_mul(self.num_cost_variables)
            .expect("cost-information index overflow");
        let end = start
            .checked_add(self.num_cost_variables)
            .expect("cost-information index overflow");
        if self.values.len() < end {
            self.values.resize(end, 0.0);
        }
        for (target, &value) in self.values[start..end].iter_mut().zip(values) {
            *target = float_tolerance::canonicalize(value);
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
    Decoded(&'a [f64]),
    Registered {
        state_id: StateID,
        numeric_template: &'a [f64],
        regular_numeric_slots: &'a [(usize, usize)],
        cost_numeric_indices: &'a [(usize, usize)],
        numeric_var_types: &'a [NumericType],
        numeric_indices: &'a [Option<usize>],
        numeric_constants: &'a [f64],
        compact_numeric_values: &'a RefCell<CompactNumericValues>,
        cost_info: &'a RefCell<DenseCostInformation>,
        axiom_evaluator: &'a AxiomEvaluator<'a>,
    },
}

impl<'a> ConcreteStateView<'a> {
    pub fn from_decoded(packer: &'a StatePacker, prop: &'a [u64], numeric: &'a [f64]) -> Self {
        Self {
            packer,
            prop,
            backing: ConcreteStateViewBacking::Decoded(numeric),
        }
    }
}

impl<'a> ConcreteStateView<'a> {
    pub fn packer(self) -> &'a StatePacker {
        self.packer
    }

    pub fn propositional(self) -> &'a [u64] {
        self.prop
    }

    pub fn fill_propositional(self, output: &mut Vec<usize>) {
        output.clear();
        output.extend(
            (0..self.packer.numeric_slot_offset())
                .map(|slot| self.packer.get(self.prop, slot) as usize),
        );
    }

    pub fn decoded_numeric(self) -> Option<&'a [f64]> {
        match self.backing {
            ConcreteStateViewBacking::Decoded(values) => Some(values),
            ConcreteStateViewBacking::Registered { .. } => None,
        }
    }

    pub fn fill_numeric(self, output: &mut Vec<f64>) -> Result<(), AssignmentAxiomError> {
        match self.backing {
            ConcreteStateViewBacking::Decoded(values) => {
                output.clear();
                output.extend_from_slice(values);
                Ok(())
            }
            ConcreteStateViewBacking::Registered {
                state_id,
                numeric_template,
                regular_numeric_slots,
                cost_numeric_indices,
                compact_numeric_values,
                cost_info,
                axiom_evaluator,
                ..
            } => {
                output.resize(numeric_template.len(), 0.0);
                output.copy_from_slice(numeric_template);
                let interned = compact_numeric_values.borrow();
                for &(out_idx, packed_slot) in regular_numeric_slots {
                    let id = self.packer.get(self.prop, packed_slot) as usize;
                    output[out_idx] = *interned
                        .values
                        .get(id)
                        .unwrap_or_else(|| panic!("missing compact numeric value ID {id}"));
                }
                let costs = cost_info.borrow();
                let cost_values = costs.get(state_id);
                for &(out_idx, cost_idx) in cost_numeric_indices {
                    output[out_idx] = cost_values[cost_idx];
                }
                if axiom_evaluator.has_numeric_axioms() {
                    axiom_evaluator.evaluate_arithmetic_axioms(output)?;
                }
                Ok(())
            }
        }
    }

    pub fn numeric_value_unevaluated(self, var_id: usize) -> Result<f64, InvalidIndex> {
        match self.backing {
            ConcreteStateViewBacking::Decoded(values) => {
                values.get(var_id).copied().ok_or(InvalidIndex {
                    index: var_id,
                    length: values.len(),
                })
            }
            ConcreteStateViewBacking::Registered {
                state_id,
                numeric_var_types,
                numeric_indices,
                numeric_constants,
                compact_numeric_values,
                cost_info,
                ..
            } => {
                let Some(&numeric_type) = numeric_var_types.get(var_id) else {
                    return Err(InvalidIndex {
                        index: var_id,
                        length: numeric_var_types.len(),
                    });
                };
                let value = match numeric_type {
                    NumericType::Cost => {
                        let cost_idx = numeric_indices[var_id].unwrap();
                        cost_info.borrow().get(state_id)[cost_idx]
                    }
                    NumericType::Constant => numeric_constants[numeric_indices[var_id].unwrap()],
                    NumericType::Regular => {
                        let id =
                            self.packer.get(self.prop, numeric_indices[var_id].unwrap()) as usize;
                        *compact_numeric_values
                            .borrow()
                            .values
                            .get(id)
                            .unwrap_or_else(|| panic!("missing compact numeric value ID {id}"))
                    }
                    NumericType::Derived => 0.0,
                };
                Ok(value)
            }
        }
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
    pub fn get_state(&self, state_registry: &StateRegistry) -> Vec<usize> {
        let mut values =
            Vec::with_capacity(state_registry.global_state_packer.numeric_slot_offset());
        self.fill_state(state_registry, &mut values);
        values
    }

    /// Fill `output` with the propositional state values without allocating a new vector.
    pub fn fill_state(&self, state_registry: &StateRegistry, output: &mut Vec<usize>) {
        let buffer = state_registry.get_buffer(self.pool_offset);
        let state_packer = &state_registry.global_state_packer;

        output.resize(state_packer.numeric_slot_offset(), 0);
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
        let num_propositional_slots = state_registry.global_state_packer.numeric_slot_offset();
        if var_id >= num_propositional_slots {
            return Err(InvalidIndex {
                index: var_id,
                length: num_propositional_slots,
            });
        }

        let buffer = state_registry.get_buffer(self.pool_offset);
        Ok(state_registry.global_state_packer.get(buffer, var_id) as usize)
    }

    /// Get the numeric state values for regular variables.
    pub fn get_numeric_state(&self, state_registry: &StateRegistry) -> Vec<f64> {
        state_registry
            .task
            .numeric_variables()
            .iter()
            .enumerate()
            .filter_map(|(i, var)| {
                if var.get_type() == &NumericType::Regular {
                    Some(
                        state_registry
                            .get_numeric_var_value_unevaluated(self, i)
                            .expect("regular numeric variable must have a packed state value"),
                    )
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
}

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

/// The parent and successor views one operator application reads and writes.
///
/// Bundled so applying an operator takes the whole transition rather than five
/// individually orderable slices; nothing is copied, the fields are the same
/// borrows the caller already holds.
struct NumericTransition<'a> {
    /// The parent state's numeric values. Every effect operand is read from
    /// here, so the reads stay independent of effect order.
    parent_values: &'a [f64],
    /// The successor's numeric values, updated in place.
    current_values: &'a mut [f64],
    /// This operator's contribution to each cost variable.
    cost_part: &'a mut [f64],
    /// The successor's packed buffer, updated in place.
    next_buffer: &'a mut [u64],
    /// The parent's packed buffer, read for effect conditions and operands.
    previous_buffer: &'a [u64],
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
    /// Registered state IDs indexed by the hash of their packed state. The
    /// table stores no duplicate hash key: lookup always verifies the exact
    /// packed state, so distinct states remain sound under hash collisions.
    registered_states: RegisteredStates,
    /// Dense row-major `f64` cost values indexed by state ID. Cost variables
    /// are not part of state identity, but storing one allocation per state is
    /// unnecessary because state IDs are dense.
    cost_info: RefCell<DenseCostInformation>,
    /// Snapshot of the numeric variable layout, populated at construction.
    /// Iterating this avoids per-call vtable dispatch through `task.numeric_variables()`
    /// in hot paths like `fill_numeric_vars`.
    numeric_var_types: Vec<NumericType>,
    /// Initial numeric row with immutable constants in place and every
    /// state-dependent or derived entry zeroed. Numeric-state materialization
    /// copies this row before filling the two state-dependent layouts below.
    numeric_template: Vec<f64>,
    /// `(numeric variable id, packed-state slot)` for regular variables.
    regular_numeric_slots: Vec<(usize, usize)>,
    /// `(numeric variable id, dense cost-row slot)` for cost variables.
    cost_numeric_indices: Vec<(usize, usize)>,
    /// Cached count returned by `count_cost_variables`. Constant for the task.
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
    /// Exact canonical-f64 interning for compact state storage. Regular
    /// numeric values are represented by checked 32-bit IDs in the packer.
    compact_numeric_values: RefCell<CompactNumericValues>,
}

#[derive(Debug, Default)]
struct CompactNumericValues {
    ids_by_bits: HashMap<u64, u32, MixedHasherBuilder>,
    values: Vec<f64>,
}

impl<'a> StateRegistry<'a> {
    pub fn view<'s>(&'s self, state: &'s ConcreteState) -> ConcreteStateView<'s>
    where
        'a: 's,
    {
        ConcreteStateView {
            packer: &self.global_state_packer,
            prop: self.get_buffer(state.get_id()),
            backing: ConcreteStateViewBacking::Registered {
                state_id: state.get_id(),
                numeric_template: &self.numeric_template,
                regular_numeric_slots: &self.regular_numeric_slots,
                cost_numeric_indices: &self.cost_numeric_indices,
                numeric_var_types: &self.numeric_var_types,
                numeric_indices: &self.numeric_indices,
                numeric_constants: &self.numeric_constants,
                compact_numeric_values: &self.compact_numeric_values,
                cost_info: &self.cost_info,
                axiom_evaluator: &self.axiom_evaluator,
            },
        }
    }

    /// Build the state packer, axiom evaluator, and registry for `task` in
    /// one step. This is the common construction path; use [`Self::new`]
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
    pub fn new(
        task: TaskRef<'a>,
        global_state_packer: Arc<StatePacker>,
        axiom_evaluator: Arc<AxiomEvaluator<'a>>,
    ) -> Self {
        let numeric_vars = task.numeric_variables();
        let number_numeric_vars = numeric_vars.len();
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);

        let numeric_var_types: Vec<NumericType> =
            numeric_vars.iter().map(|var| *var.get_type()).collect();
        let initial_numeric_values = task.get_initial_numeric_state_values();
        assert_eq!(
            initial_numeric_values.len(),
            numeric_var_types.len(),
            "initial numeric state must contain one value per numeric variable"
        );
        let mut numeric_template = vec![0.0; number_numeric_vars];
        let mut regular_numeric_slots = Vec::new();
        let mut cost_numeric_indices = Vec::new();
        let mut next_regular_slot = global_state_packer.numeric_slot_offset();
        let mut next_cost_index = 0;
        for (numeric_var_id, &ty) in numeric_var_types.iter().enumerate() {
            match ty {
                NumericType::Constant => {
                    numeric_template[numeric_var_id] =
                        float_tolerance::canonicalize(initial_numeric_values[numeric_var_id]);
                }
                NumericType::Regular => {
                    regular_numeric_slots.push((numeric_var_id, next_regular_slot));
                    next_regular_slot += 1;
                }
                NumericType::Cost => {
                    cost_numeric_indices.push((numeric_var_id, next_cost_index));
                    next_cost_index += 1;
                }
                NumericType::Derived => {}
            }
        }
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
                .get_variable_axiom_layer(var_id)
                .expect("variable id below the packer's numeric offset must exist in the task");
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
            global_state_packer,
            state_data_pool,
            numeric_constants: Vec::new(),
            numeric_indices: vec![None; number_numeric_vars],
            registered_states: RegisteredStates::with_capacity(1024),
            axiom_evaluator,
            cost_info: RefCell::new(DenseCostInformation::new(cost_variable_count)),
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
            compact_numeric_values: RefCell::new(CompactNumericValues::default()),
        }
    }

    fn pack_regular_numeric(&self, value: f64) -> u64 {
        let canonical = float_tolerance::canonicalize(value);
        let bits = canonical.to_bits();
        let mut interner = self.compact_numeric_values.borrow_mut();
        if let Some(&id) = interner.ids_by_bits.get(&bits) {
            return id as u64;
        }
        let id = u32::try_from(interner.values.len())
            .expect("compact numeric state value table exceeds 32-bit ID capacity");
        interner.values.push(canonical);
        let previous = interner.ids_by_bits.insert(bits, id);
        assert!(
            previous.is_none(),
            "duplicate compact numeric value insertion"
        );
        id as u64
    }

    fn unpack_regular_numeric(&self, packed: u64) -> f64 {
        let id = usize::try_from(packed).expect("compact numeric value ID exceeds usize");
        *self
            .compact_numeric_values
            .borrow()
            .values
            .get(id)
            .unwrap_or_else(|| panic!("missing compact numeric value ID {id}"))
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

        self.insert_registered_state_id(key, state_id);
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

        let existing_id = self
            .registered_states
            .find(key, |&(candidate_hash, compact_candidate)| {
                candidate_hash == key && {
                    let candidate = compact_candidate as StateID;
                    let existing = self.get_buffer(candidate);
                    let probe = self.get_buffer(state_id);
                    bins_eq_masked(
                        &existing[..num_bins],
                        &probe[..num_bins],
                        &self.non_derived_bits_mask[..num_bins],
                    )
                }
            })
            .map(|&(_, id)| id as StateID);

        if let Some(existing_id) = existing_id {
            self.state_data_pool.pop_back();
            return (existing_id, false);
        }

        self.insert_registered_state_id(key, state_id);
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
            self.task.get_initial_propositional_state_values().to_vec();
        let initial_numeric_values = self.task.get_initial_numeric_state_values().to_vec();

        // Pack propositional variables.
        self.pack_propositional_variables(&mut init_buffer, &initial_propositional_values);

        // Process numeric variables and get cost variables.
        let _cost_variables =
            self.process_numeric_variables(&mut init_buffer, &initial_numeric_values);
        self.assert_cost_layout(&_cost_variables);

        // Evaluate axioms.
        let mut numeric_state_copy = initial_numeric_values;
        self.evaluate_axioms(&mut init_buffer, &mut numeric_state_copy)
            .expect("Failed to evaluate axioms during initial state creation");

        // Register the state.
        self.state_data_pool.push_back(&init_buffer);
        let (state_id, _) = self.insert_id_or_pop_state();

        let init_state = ConcreteState::new(state_id);

        // Cost values are excluded from state identity and therefore live in
        // dense side storage. Install the row before any numeric state read.
        if self.cost_variable_count > 0 {
            self.set_cost_information(&init_state, &_cost_variables);
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
        let mut numeric_var_index = self.global_state_packer.numeric_slot_offset();
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
                    let packed_value = self.pack_regular_numeric(value);
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

        // Regular numeric variables are the ones whose recorded index is a
        // packer slot in the numeric section; cost and constant variables index
        // dense side storage and always land below the offset.
        let numeric_slot_offset = self.global_state_packer.numeric_slot_offset();
        let regular_count = self
            .numeric_indices
            .iter()
            .filter(|index| index.is_some_and(|index| index >= numeric_slot_offset))
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
    ///
    /// `numeric_values` is indexed by numeric-variable id, so it accepts a full
    /// vector as returned by [`Self::get_numeric_vars`]. Constant entries are
    /// checked against the values this registry already holds and derived
    /// entries are recomputed by the axiom pass; see
    /// [`Self::process_register_numeric_variables`]. The initial state must
    /// have been created first, because that is where the numeric layout of the
    /// registry is fixed.
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
                self.set_cost_information(&new_state, &_cost_variables);
            }
        } else {
            // Existing state: use metric optimization to determine which cost info to keep.
            let keep_old_cost_information =
                self.should_keep_old_cost_information(&new_state, &numeric_values_copy);

            match keep_old_cost_information {
                Ok(false) => {
                    self.set_cost_information(&new_state, &_cost_variables);
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
    ///
    /// Only `Regular` values enter the packed buffer and only `Cost` values are
    /// returned; `Constant` values are the same in every state and are verified
    /// against the ones this registry recorded when the initial state was
    /// created, and `Derived` values are a function of the inputs that
    /// [`Self::evaluate_axioms`] recomputes right after this call.
    fn process_register_numeric_variables(
        &mut self,
        buffer: &mut [u64],
        numeric_values: &[f64],
    ) -> Result<Vec<f64>, StateInsertError> {
        let mut regular_index = self.global_state_packer.numeric_slot_offset();
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
                    let packed_value = self.pack_regular_numeric(value);
                    self.global_state_packer.set(
                        buffer,
                        self.numeric_indices[i].unwrap(),
                        packed_value,
                    );
                }
                NumericType::Constant => {
                    let constant_index =
                        self.numeric_indices[i].ok_or_else(|| StateInsertError {
                            message: format!(
                                "Constant numeric variable {i} has no value yet; create the initial state before registering other states"
                            ),
                        })?;
                    let registered = *self.numeric_constants.get(constant_index).ok_or_else(|| {
                        StateInsertError {
                            message: format!(
                                "Constant numeric variable {i} points at constant slot {constant_index}, but only {} constants are registered",
                                self.numeric_constants.len()
                            ),
                        }
                    })?;
                    if !float_tolerance::equal(registered, value) {
                        return Err(StateInsertError {
                            message: format!(
                                "Constant numeric variable {i} is {registered} in this registry, but the state to register has {value}"
                            ),
                        });
                    }
                }
                NumericType::Derived => {
                    // Recomputed from the inputs by `evaluate_axioms`, so the
                    // supplied value is deliberately not stored.
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
            self.evaluate_metric(&ctx.parent_numeric)
                .map_err(|e| StateInsertError {
                    message: format!("Failed to evaluate metric for parent state: {e:?}"),
                })?
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

        self.apply_propositional_effects(next_buffer, previous_buffer, operator);

        // Skip the comparison/propositional axiom passes during effect
        // application when the task has axiom-derived bits we can mask off.
        // We dedup using only non-derived bits below; if the successor is a
        // duplicate, the existing state already has correct derived bits and
        // we save the (typically expensive) axiom evaluation. If the
        // successor is new, we run the full axiom pass after dedup.
        let defer_full_axioms = self.has_axiom_derived_bits;
        self.apply_numeric_effects_inner(
            NumericTransition {
                parent_values: &ctx.parent_numeric,
                current_values: successor_values,
                cost_part: cost_values,
                next_buffer,
                previous_buffer,
            },
            operator,
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
        // Cost values use dense row-major storage, so duplicate comparison is
        // a direct indexed load without per-state allocation or hashing.
        if self.cost_variable_count > 0 {
            if is_new_state {
                assert_eq!(
                    cost_values.len(),
                    self.cost_variable_count,
                    "successor must define every f64 cost variable"
                );
                self.set_cost_information(&successor, cost_values);
            } else {
                let metric_is_regular =
                    matches!(self.metric_var, Some((_, NumericType::Regular)) | None);
                let keep_old = if !self.metric_use_metric {
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
                    if self.metric_is_min {
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

    fn cost_index(&self, numeric_var_id: usize) -> usize {
        self.numeric_indices[numeric_var_id]
            .unwrap_or_else(|| panic!("cost numeric variable {numeric_var_id} has no cost slot"))
    }

    /// Validate the cost-variable-to-row mapping when the initial state fixes
    /// the registry's numeric layout. Every later row has the same length,
    /// enforced by [`DenseCostInformation::set`].
    fn assert_cost_layout(&self, cost_variables: &[f64]) {
        assert_eq!(
            cost_variables.len(),
            self.cost_variable_count,
            "initial state must define every cost variable"
        );
        for (numeric_var_id, ty) in self.numeric_var_types.iter().enumerate() {
            if *ty == NumericType::Cost {
                let cost_index = self.cost_index(numeric_var_id);
                assert!(
                    cost_index < cost_variables.len(),
                    "cost numeric variable {numeric_var_id} points at slot {cost_index}, but the cost row has {} entries",
                    cost_variables.len()
                );
            }
        }
    }

    fn fill_cost_information(&self, state: &ConcreteState, output: &mut Vec<f64>) {
        let cost_info_borrow = self.cost_info.borrow();
        let cost_info_data = cost_info_borrow.get(state.get_id());
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
    pub fn get_numeric_vars(
        &self,
        state: &ConcreteState,
    ) -> Result<Vec<f64>, AssignmentAxiomError> {
        let mut result = vec![0.0; self.task.numeric_variables().len()];
        self.fill_numeric_vars(state, &mut result)?;
        Ok(result)
    }

    pub fn fill_state_and_numeric_vars(
        &self,
        state: &ConcreteState,
        propositional_output: &mut Vec<usize>,
        numeric_output: &mut Vec<f64>,
    ) -> Result<(), AssignmentAxiomError> {
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
    ) -> Result<(), AssignmentAxiomError> {
        let buffer = state.buffer(self);
        let state_packer = &self.global_state_packer;

        propositional_output.clear();
        propositional_output.extend(
            (0..state_packer.numeric_slot_offset())
                .map(|slot| state_packer.get(buffer, slot) as usize),
        );

        numeric_output.clear();
        numeric_output.resize(self.task.numeric_variables().len(), 0.0);

        let cost_info_borrow = self.cost_info.borrow();
        let cost_variables = cost_info_borrow.get(state.get_id());

        for (i, numeric_var) in self.task.numeric_variables().iter().enumerate() {
            numeric_output[i] = match numeric_var.get_type() {
                NumericType::Cost => cost_variables[self.cost_index(i)],
                NumericType::Constant => self.numeric_constants[self.numeric_indices[i].unwrap()],
                NumericType::Regular => self.unpack_regular_numeric(
                    state_packer.get(buffer, self.numeric_indices[i].unwrap()),
                ),
                // Axioms fill derived values after this input-state pass.
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
        let cost_variables = cost_info_borrow.get(state.get_id());

        let value = match numeric_var.get_type() {
            NumericType::Cost => cost_variables[self.cost_index(numeric_var_id)],
            NumericType::Constant => {
                self.numeric_constants[self.numeric_indices[numeric_var_id].unwrap()]
            }
            NumericType::Regular => self.unpack_regular_numeric(
                self.global_state_packer
                    .get(buffer, self.numeric_indices[numeric_var_id].unwrap()),
            ),
            // Axioms fill derived values after this input-state pass.
            NumericType::Derived => 0.0,
        };

        Ok(value)
    }

    pub fn fill_numeric_vars(
        &self,
        state: &ConcreteState,
        output: &mut Vec<f64>,
    ) -> Result<(), AssignmentAxiomError> {
        output.resize(self.numeric_template.len(), 0.0);
        output.copy_from_slice(&self.numeric_template);

        let buffer = state.buffer(self);

        // Get cost information for this state.
        let cost_info_borrow = self.cost_info.borrow();
        let cost_variables = cost_info_borrow.get(state.get_id());

        for &(out_idx, packed_slot) in &self.regular_numeric_slots {
            output[out_idx] =
                self.unpack_regular_numeric(self.global_state_packer.get(buffer, packed_slot));
        }
        for &(out_idx, cost_idx) in &self.cost_numeric_indices {
            output[out_idx] = cost_variables[cost_idx];
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
        transition: NumericTransition<'_>,
        operator: &Operator,
    ) -> Result<(), StateInsertError> {
        self.apply_numeric_effects_inner(transition, operator, true)
    }

    /// Like `apply_numeric_effects`, but if `run_full_axioms` is false the
    /// comparison and propositional axiom passes are skipped (arithmetic
    /// axioms still run because the metric and any subsequent computations
    /// may need derived numeric values). The caller is then responsible for
    /// running `axiom_evaluator.evaluate` once it is known the successor is a
    /// new state worth registering — see `get_successor_state_with_buffers_and_cost`.
    fn apply_numeric_effects_inner(
        &self,
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
        // All assignment effects of one operator take effect simultaneously:
        // every right-hand side, left-hand side and effect condition reads the
        // state *before* the operator was applied. An operator with
        // `x += y` and `y += 1` must therefore add the parent's `y` to `x`,
        // whatever order the effects happen to be stored in.
        //
        // `parent_values` is never written here, so one pass suffices: reads
        // come from it (and from `previous_buffer`), writes go to
        // `current_values`, `cost_part` and `next_buffer`. Reading operands
        // out of `current_values` instead would make each effect visible to
        // its successors and reintroduce order dependence.
        //
        // One operator may legitimately write the same variable twice, as long
        // as every such effect is additive: PDDL grounding produces operators
        // like `drink ?n ?n` with both `v += 1` and `v -= 1`, and the
        // simultaneous reading of additive effects is to apply every delta to
        // the parent value, which is order-independent. Accumulating through
        // `current_values` does exactly that. Any other repeat — an assignment,
        // a scaling, or a mix — has no order-independent reading, so it is
        // rejected rather than resolved by whichever effect happens to be
        // stored last.
        let effects = operator.assignment_effects();
        let repeated_targets = operator.repeated_assignment_targets();
        assert_eq!(
            effects.len(),
            repeated_targets.len(),
            "operator assignment-effect classification must be complete"
        );
        for (effect, repeated_target) in effects.iter().zip(repeated_targets) {
            let assignment_var_id = effect.var_id();
            let affected_var_id = effect.affected_var_id();

            if assignment_var_id >= parent_values.len() {
                return Err(StateInsertError {
                    message: format!("Assignment variable ID {} out of bounds", assignment_var_id),
                });
            }
            if affected_var_id >= parent_values.len() {
                return Err(StateInsertError {
                    message: format!("Affected variable ID {} out of bounds", affected_var_id),
                });
            }

            // Effect conditions are evaluated against the parent state, in the
            // same way `apply_propositional_effects` evaluates them against
            // `current_state` rather than the partially updated buffer.
            if !self.assignment_conditions_met(effect.conditions(), previous_buffer) {
                continue;
            }

            let assignment_value =
                if self.numeric_var_types[assignment_var_id] == NumericType::Regular {
                    self.unpack_regular_numeric(self.global_state_packer.get(
                        previous_buffer,
                        self.numeric_indices[assignment_var_id].unwrap(),
                    ))
                } else {
                    parent_values[assignment_var_id]
                };

            // Accumulate additive deltas; every other operand reads the parent.
            // The target classification is immutable operator data computed
            // once at construction.
            let left_value = match repeated_target {
                RepeatedTarget::First => parent_values[affected_var_id],
                RepeatedTarget::Additive => current_values[affected_var_id],
            };

            let result = float_tolerance::canonicalize(AssignmentOperation::apply(
                left_value,
                effect.operation(),
                assignment_value,
            ));

            match self.numeric_var_types[affected_var_id] {
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
                    let packed_result = self.pack_regular_numeric(result);
                    self.global_state_packer.set(
                        next_buffer,
                        self.numeric_indices[affected_var_id].unwrap(),
                        packed_result,
                    );
                    current_values[affected_var_id] = result;
                }
                affected_ty => {
                    return Err(StateInsertError {
                        message: format!(
                            "Only regular and cost variables can be affected by assignment operations: {:?}",
                            affected_ty
                        ),
                    });
                }
            }
        }

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

        // `successor_numeric_values` still holds `state`'s values here, so this
        // snapshot is the parent every effect must read from.
        let parent_numeric_values = successor_numeric_values.clone();

        self.apply_propositional_effects(&mut next_buffer, previous_buffer, operator);
        self.apply_numeric_effects(
            NumericTransition {
                parent_values: &parent_numeric_values,
                current_values: &mut successor_numeric_values,
                cost_part: cost_values.as_mut_slice(),
                next_buffer: &mut next_buffer,
                previous_buffer,
            },
            operator,
        )?;

        let new_metric = self
            .evaluate_metric(&successor_numeric_values)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate metric after applying operator: {e:?}"),
            })?;

        Ok(new_metric - old_metric)
    }

    fn metric_value_for_state(&self, state: &ConcreteState) -> Result<f64, AxiomEvalError> {
        let Some((metric_fluent_id, metric_type)) = self.metric_var else {
            return Ok(0.0);
        };
        if metric_fluent_id >= self.numeric_var_types.len() {
            return Err(AxiomEvalError::InvalidIndex(InvalidIndex {
                length: self.numeric_var_types.len(),
                index: metric_fluent_id,
            }));
        }

        match metric_type {
            NumericType::Regular => {
                let buffer = state.buffer(self);
                Ok(self.unpack_regular_numeric(
                    self.global_state_packer
                        .get(buffer, self.numeric_indices[metric_fluent_id].unwrap()),
                ))
            }
            NumericType::Cost => {
                let cost_index = self.cost_index(metric_fluent_id);
                let cost_info_borrow = self.cost_info.borrow();
                let cost_values = cost_info_borrow.get(state.get_id());
                Ok(cost_values[cost_index])
            }
            NumericType::Constant => {
                Ok(self.numeric_constants[self.numeric_indices[metric_fluent_id].unwrap()])
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
        successor_numeric_vals: &[f64],
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
    /// This corresponds to the C++ g_cost_information[state] access pattern.
    /// Return an empty vector if no cost information is stored for the state.
    pub fn get_cost_information(&self, state: &ConcreteState) -> Vec<f64> {
        self.cost_info.borrow().get(state.get_id()).to_vec()
    }

    /// Set cost information for a given state.
    ///
    /// This corresponds to the C++ g_cost_information[state] = values assignment.
    /// It uses `RefCell` for interior mutability to resolve borrowing conflicts.
    fn set_cost_information(&self, state: &ConcreteState, values: &[f64]) {
        self.cost_info.borrow_mut().set(state.get_id(), values);
    }
}

fn canonicalize_numeric_values(values: &mut [f64]) {
    for value in values {
        *value = float_tolerance::canonicalize(*value);
    }
}
