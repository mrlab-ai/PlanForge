//! Numeric Cartesian abstractions refined by concrete counterexamples.
//!
//! Unlike the factorized domain abstraction, splitting one Cartesian state
//! adds exactly one state. Every abstract transition is a may-transition of a
//! grounded concrete operator. CEGAR replays a deterministic optimal abstract
//! trace. The default refines its first witnessed flaw; the explicit
//! whole-plan mode ranks all sound witnesses on an acyclic adaptive trace.
//! Only a successfully replayed concrete plan may set `solved_by_self`.

mod finalize;
mod flaw_splits;
pub mod icaps26;
mod plan_replay;
mod shortest_paths;
mod split_generation;
mod split_selector;
#[cfg(test)]
mod tests;

use finalize::*;
use flaw_splits::*;
use plan_replay::*;
use shortest_paths::*;
use split_generation::*;
use split_selector::*;

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use ordered_float::NotNan;
use planforge_sas::axioms::{AxiomEvaluator, ComparisonOperator};
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, ExplicitFact, NumericType,
    metric_operator_cost_from_initial_values,
};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::state_packer::StatePacker;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tracing::{debug, info};

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;
use crate::evaluation::validate_abstractable_goal;

use super::abstraction_collections::cost_partitioning::{
    AbstractOperatorFootprint, AbstractTransition, AbstractTransitionSystem,
    ConcreteOperatorFootprint, PropValueId, StateRegion, build_explicit_goal_distances,
    sorted_value_sets_overlap,
};
use super::abstraction_collections::portfolio::{
    CollectionStrategy, derive_variant_seed, mix_seed, stable_text_seed,
};
use super::abstraction_task::{AbstractionUse, SingleGoalTask, validate_abstraction_operator};
use super::cegar::{
    CegarDriver, CegarIterationResult, CegarStopReason, FlawKind, progress_concrete_state,
};
use super::domain_abstractions::additive_numeric_views::{
    AdditiveNumericView, analyze_additive_numeric_view, comparison_refinement_dimensions,
    numeric_dimension_delta_for_operator,
};
use super::domain_abstractions::domain_abstraction_factory::AbstractDistanceTable;
use super::domain_abstractions::utils::{get_initial_state, make_prop_state_packer};
use icaps26::{ArtifactMt19937, Icaps26SplitSelection};
use planforge_sas::utils::interval::Interval;

#[inline]
fn fact_is_hold(fact: &ExplicitFact, packer: &StatePacker, buffer: &[u64]) -> bool {
    fact.is_hold(
        planforge_sas::state_registry::ConcreteStateView::from_decoded(packer, buffer, &[]),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartesianStopReason {
    ConcretePlan,
    StateLimit,
    TimeLimit,
    MemoryLimit,
}

#[derive(Debug, Clone)]
pub struct CartesianAbstractionMetadata {
    pub solved_by_self: bool,
    pub abstraction_use: AbstractionUse,
    pub stop_reason: CartesianStopReason,
    pub pending_flaw: Option<String>,
    pub refinements: usize,
    pub collection_goal_id: Option<usize>,
    pub collection_variant_id: Option<usize>,
    pub refinement_direction: CartesianRefinementDirection,
    pub split_selection_rank: Option<usize>,
    pub concrete_plan_operator_ids: Option<Vec<usize>>,
    pub progressive_refinement_root: bool,
    /// Number of non-loop transitions built before optional standalone compaction.
    pub transition_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartesianRefinementDirection {
    Progression,
    Regression,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartesianSplitSelection {
    MinTransitionGrowth,
    MaxAdditiveSteps,
    Random,
    LeastRefined,
    Icaps26(Icaps26SplitSelection),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartesianAbstractPlanSelection {
    BackwardShortestPath,
    StableAStar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartesianFlawCandidateGeneration {
    General,
    DesiredRegion,
}

#[derive(Debug, Clone)]
pub struct CartesianAbstractionConfig {
    pub max_states: usize,
    pub max_time: Option<Duration>,
    pub combine_labels: bool,
    pub compute_operator_footprints: bool,
    /// Retain the explicit graph required by cost partitioning.
    pub retain_transition_system: bool,
    pub random_seed: Option<u64>,
    pub flaw_kind: FlawKind,
    pub refinement_direction: CartesianRefinementDirection,
    pub abstract_plan_selection: CartesianAbstractPlanSelection,
    pub flaw_candidate_generation: CartesianFlawCandidateGeneration,
    pub split_selection_rank: Option<usize>,
    pub split_selection: CartesianSplitSelection,
    pub debug: bool,
}

impl Default for CartesianAbstractionConfig {
    fn default() -> Self {
        Self {
            max_states: 10_000,
            max_time: None,
            combine_labels: false,
            compute_operator_footprints: true,
            retain_transition_system: true,
            random_seed: None,
            flaw_kind: FlawKind::Progression,
            refinement_direction: CartesianRefinementDirection::Progression,
            abstract_plan_selection: CartesianAbstractPlanSelection::BackwardShortestPath,
            flaw_candidate_generation: CartesianFlawCandidateGeneration::General,
            split_selection_rank: None,
            split_selection: CartesianSplitSelection::MinTransitionGrowth,
            debug: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CartesianAbstractionCollectionConfig {
    pub abstraction: CartesianAbstractionConfig,
    pub collection_strategy: CollectionStrategy,
    pub variants_per_goal: usize,
    pub max_collection_states: usize,
    pub total_max_time: Option<Duration>,
    pub progressive_goal_roots: bool,
}

impl Default for CartesianAbstractionCollectionConfig {
    fn default() -> Self {
        Self {
            abstraction: CartesianAbstractionConfig::default(),
            collection_strategy: CollectionStrategy::Standard,
            variants_per_goal: 1,
            max_collection_states: 10_000_000,
            total_max_time: None,
            progressive_goal_roots: false,
        }
    }
}

#[derive(Debug, Clone)]
struct CartesianConcreteState {
    propositions: Vec<u64>,
    numeric: Vec<f64>,
}

#[derive(Debug, Clone)]
enum RefinementNode {
    Leaf {
        state_id: usize,
    },
    Propositional {
        var_id: usize,
        wanted: Vec<PropValueId>,
        wanted_child: usize,
        other_child: usize,
    },
    Numeric {
        var_id: usize,
        boundary: f64,
        lower_includes_boundary: bool,
        lower_child: usize,
        upper_child: usize,
    },
}

#[derive(Debug, Clone)]
pub struct CartesianRefinementHierarchy {
    nodes: Vec<RefinementNode>,
}

impl CartesianRefinementHierarchy {
    fn trivial() -> Self {
        Self {
            nodes: vec![RefinementNode::Leaf { state_id: 0 }],
        }
    }

    pub fn map_state(&self, propositional: &[usize], numeric: &[f64]) -> Result<usize> {
        let mut node_id = 0;
        loop {
            match self
                .nodes
                .get(node_id)
                .with_context(|| format!("missing Cartesian hierarchy node {node_id}"))?
            {
                RefinementNode::Leaf { state_id } => return Ok(*state_id),
                RefinementNode::Propositional {
                    var_id,
                    wanted,
                    wanted_child,
                    other_child,
                } => {
                    let value = *propositional.get(*var_id).with_context(|| {
                        format!("propositional state has no value for var {var_id}")
                    })? as PropValueId;
                    node_id = if wanted.binary_search(&value).is_ok() {
                        *wanted_child
                    } else {
                        *other_child
                    };
                }
                RefinementNode::Numeric {
                    var_id,
                    boundary,
                    lower_includes_boundary,
                    lower_child,
                    upper_child,
                } => {
                    let value = *numeric
                        .get(*var_id)
                        .with_context(|| format!("numeric state has no value for var {var_id}"))?;
                    ensure!(
                        value.is_finite(),
                        "numeric state var {var_id} is not finite: {value}"
                    );
                    let in_lower =
                        value < *boundary || (*lower_includes_boundary && value == *boundary);
                    node_id = if in_lower { *lower_child } else { *upper_child };
                }
            }
        }
    }

    fn split_propositional(
        &mut self,
        leaf_node_id: usize,
        old_state_id: usize,
        new_state_id: usize,
        var_id: usize,
        mut wanted: Vec<PropValueId>,
        old_state_is_wanted: bool,
    ) -> Result<()> {
        wanted.sort_unstable();
        wanted.dedup();
        let wanted_node_id = self.nodes.len();
        let other_node_id = wanted_node_id + 1;
        self.nodes.push(RefinementNode::Leaf {
            state_id: if old_state_is_wanted {
                old_state_id
            } else {
                new_state_id
            },
        });
        self.nodes.push(RefinementNode::Leaf {
            state_id: if old_state_is_wanted {
                new_state_id
            } else {
                old_state_id
            },
        });
        let node = self
            .nodes
            .get_mut(leaf_node_id)
            .with_context(|| format!("missing hierarchy leaf node {leaf_node_id}"))?;
        ensure!(
            matches!(node, RefinementNode::Leaf { state_id } if *state_id == old_state_id),
            "hierarchy node {leaf_node_id} is not leaf state {old_state_id}"
        );
        *node = RefinementNode::Propositional {
            var_id,
            wanted,
            wanted_child: wanted_node_id,
            other_child: other_node_id,
        };
        Ok(())
    }

    fn split_numeric(
        &mut self,
        leaf_node_id: usize,
        old_state_id: usize,
        new_state_id: usize,
        split: NumericSplit,
    ) -> Result<()> {
        let NumericSplit {
            var_id,
            boundary,
            lower_includes_boundary,
            old_state_is_lower,
        } = split;
        ensure!(
            boundary.is_finite(),
            "Cartesian split boundary must be finite"
        );
        let lower_node_id = self.nodes.len();
        let upper_node_id = lower_node_id + 1;
        self.nodes.push(RefinementNode::Leaf {
            state_id: if old_state_is_lower {
                old_state_id
            } else {
                new_state_id
            },
        });
        self.nodes.push(RefinementNode::Leaf {
            state_id: if old_state_is_lower {
                new_state_id
            } else {
                old_state_id
            },
        });
        let node = self
            .nodes
            .get_mut(leaf_node_id)
            .with_context(|| format!("missing hierarchy leaf node {leaf_node_id}"))?;
        ensure!(
            matches!(node, RefinementNode::Leaf { state_id } if *state_id == old_state_id),
            "hierarchy node {leaf_node_id} is not leaf state {old_state_id}"
        );
        *node = RefinementNode::Numeric {
            var_id,
            boundary,
            lower_includes_boundary,
            lower_child: lower_node_id,
            upper_child: upper_node_id,
        };
        Ok(())
    }
}

/// What is left of the collection's state and time budget when one member is
/// about to be built.
#[derive(Debug, Clone, Copy)]
struct MemberBudget {
    remaining_states: usize,
    remaining_time: Option<Duration>,
}

/// Where a numeric refinement cuts one variable's range, and on which side of
/// the cut the state being split ends up.
#[derive(Debug, Clone, Copy)]
struct NumericSplit {
    var_id: usize,
    boundary: f64,
    /// Whether `boundary` itself belongs to the lower child.
    lower_includes_boundary: bool,
    /// Whether the state that existed before the split becomes the lower child.
    old_state_is_lower: bool,
}

#[derive(Debug, Clone)]
pub struct CartesianAbstraction {
    pub hierarchy: CartesianRefinementHierarchy,
    pub distance_table: AbstractDistanceTable,
    pub transition_system: AbstractTransitionSystem,
    pub relevant_operator_ids: Vec<usize>,
    pub abstract_operator_footprints: Vec<AbstractOperatorFootprint>,
    pub metadata: CartesianAbstractionMetadata,
}

impl CartesianAbstraction {
    pub fn num_states(&self) -> usize {
        self.distance_table.distances.len()
    }

    pub fn abstract_state_id(&self, propositional: &[usize], numeric: &[f64]) -> Result<usize> {
        self.hierarchy.map_state(propositional, numeric)
    }

    pub fn discard_transition_data(&mut self) {
        self.transition_system.transitions = Vec::new();
        self.transition_system.backward = Vec::new();
        self.transition_system.forward = Vec::new();
        self.transition_system.state_regions = Vec::new();
        self.abstract_operator_footprints = Vec::new();
    }
}

#[derive(Debug, Clone)]
struct WorkingTransition {
    source: usize,
    target: usize,
    concrete_op_id: usize,
}

#[derive(Debug, Clone)]
struct OperatorBitSet {
    words: Box<[u64]>,
    operator_count: usize,
}

impl OperatorBitSet {
    fn empty(operator_count: usize) -> Self {
        Self {
            words: vec![0; operator_count.div_ceil(u64::BITS as usize)].into_boxed_slice(),
            operator_count,
        }
    }

    fn insert(&mut self, operator_id: usize) -> bool {
        debug_assert!(
            operator_id < self.operator_count,
            "operator {operator_id} exceeds Cartesian operator-set size {}",
            self.operator_count
        );
        let word = &mut self.words[operator_id / u64::BITS as usize];
        let mask = 1_u64 << (operator_id % u64::BITS as usize);
        if *word & mask != 0 {
            return false;
        }
        *word |= mask;
        true
    }

    fn contains(&self, operator_id: usize) -> bool {
        debug_assert!(
            operator_id < self.operator_count,
            "operator {operator_id} exceeds Cartesian operator-set size {}",
            self.operator_count
        );
        self.words[operator_id / u64::BITS as usize] & (1_u64 << (operator_id % u64::BITS as usize))
            != 0
    }

    fn intersection_iter<'a>(&'a self, other: &'a Self) -> OperatorBitSetIntersectionIter<'a> {
        debug_assert_eq!(
            self.operator_count, other.operator_count,
            "cannot intersect Cartesian operator sets of different sizes"
        );
        OperatorBitSetIntersectionIter {
            left: &self.words,
            right: &other.words,
            operator_count: self.operator_count,
            word_id: 0,
            remaining: self.words.first().copied().unwrap_or(0)
                & other.words.first().copied().unwrap_or(0),
        }
    }

    fn clone_without(&self, excluded: &Self) -> Self {
        debug_assert_eq!(
            self.operator_count, excluded.operator_count,
            "cannot subtract Cartesian operator sets of different sizes"
        );
        Self {
            words: self
                .words
                .iter()
                .zip(excluded.words.iter())
                .map(|(&word, &excluded_word)| word & !excluded_word)
                .collect(),
            operator_count: self.operator_count,
        }
    }
}

struct OperatorBitSetIntersectionIter<'a> {
    left: &'a [u64],
    right: &'a [u64],
    operator_count: usize,
    word_id: usize,
    remaining: u64,
}

impl Iterator for OperatorBitSetIntersectionIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.remaining != 0 {
                let bit = self.remaining.trailing_zeros() as usize;
                self.remaining &= self.remaining - 1;
                let operator_id = self.word_id * u64::BITS as usize + bit;
                debug_assert!(
                    operator_id < self.operator_count,
                    "Cartesian operator intersection has a set padding bit"
                );
                return Some(operator_id);
            }
            self.word_id += 1;
            self.remaining = *self.left.get(self.word_id)? & self.right[self.word_id];
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TransitionKey {
    source: usize,
    concrete_op_id: usize,
    target: usize,
}

#[derive(Debug, Clone)]
struct WorkingAbstraction {
    states: Vec<StateRegion>,
    leaf_node_ids: Vec<usize>,
    hierarchy: CartesianRefinementHierarchy,
    transitions: Vec<Option<WorkingTransition>>,
    free_transition_ids: Vec<usize>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    self_loop_operator_ids: Vec<OperatorBitSet>,
    icaps_self_loop_order: Option<Vec<Vec<usize>>>,
    transition_ids_by_key: Option<HashMap<TransitionKey, usize>>,
    propositional_refinement_counts: Vec<usize>,
    numeric_refinement_counts: Vec<usize>,
}

impl WorkingAbstraction {
    fn states(&self) -> &[StateRegion] {
        &self.states
    }

    fn hierarchy(&self) -> &CartesianRefinementHierarchy {
        &self.hierarchy
    }

    fn outgoing(&self) -> &[Vec<usize>] {
        &self.outgoing
    }

    fn incoming(&self) -> &[Vec<usize>] {
        &self.incoming
    }

    fn self_loop_operator_ids(&self) -> &[OperatorBitSet] {
        &self.self_loop_operator_ids
    }

    fn propositional_refinement_counts(&self) -> &[usize] {
        &self.propositional_refinement_counts
    }

    fn numeric_refinement_counts(&self) -> &[usize] {
        &self.numeric_refinement_counts
    }

    fn new(initial_region: StateRegion, operator_count: usize) -> Self {
        Self::new_with_transition_index(initial_region, operator_count, true)
    }

    fn new_icaps26(initial_region: StateRegion, operator_count: usize) -> Self {
        Self::new_with_transition_index(initial_region, operator_count, false)
    }

    fn new_with_transition_index(
        initial_region: StateRegion,
        operator_count: usize,
        index_transitions: bool,
    ) -> Self {
        let propositional_refinement_counts = vec![0; initial_region.propositions.len()];
        let numeric_refinement_counts = vec![0; initial_region.numeric.len()];
        Self {
            states: vec![initial_region],
            leaf_node_ids: vec![0],
            hierarchy: CartesianRefinementHierarchy::trivial(),
            transitions: Vec::new(),
            free_transition_ids: Vec::new(),
            outgoing: vec![Vec::new()],
            incoming: vec![Vec::new()],
            self_loop_operator_ids: vec![OperatorBitSet::empty(if index_transitions {
                operator_count
            } else {
                0
            })],
            icaps_self_loop_order: (!index_transitions).then(|| vec![Vec::new()]),
            transition_ids_by_key: index_transitions.then(HashMap::new),
            propositional_refinement_counts,
            numeric_refinement_counts,
        }
    }

    fn add_transition(&mut self, source: usize, op_id: usize, target: usize) {
        if source == target {
            if let Some(loop_order) = &mut self.icaps_self_loop_order {
                debug_assert!(
                    !loop_order[source].contains(&op_id),
                    "ICAPS Cartesian refinement generated a duplicate self-loop"
                );
                loop_order[source].push(op_id);
            } else {
                self.self_loop_operator_ids[source].insert(op_id);
            }
            return;
        }
        let key = TransitionKey {
            source,
            concrete_op_id: op_id,
            target,
        };
        if let Some(transition_ids_by_key) = &self.transition_ids_by_key {
            if transition_ids_by_key.contains_key(&key) {
                return;
            }
        } else {
            debug_assert!(
                !self.outgoing[source].iter().any(|&transition_id| {
                    self.transitions[transition_id]
                        .as_ref()
                        .is_some_and(|transition| {
                            transition.concrete_op_id == op_id && transition.target == target
                        })
                }),
                "ICAPS Cartesian refinement generated a duplicate transition"
            );
        }
        let transition = WorkingTransition {
            source,
            target,
            concrete_op_id: op_id,
        };
        let transition_id = if let Some(transition_id) = self.free_transition_ids.pop() {
            assert!(
                self.transitions[transition_id].is_none(),
                "reused Cartesian transition slot is occupied"
            );
            self.transitions[transition_id] = Some(transition);
            transition_id
        } else {
            let transition_id = self.transitions.len();
            self.transitions.push(Some(transition));
            transition_id
        };
        if let Some(transition_ids_by_key) = &mut self.transition_ids_by_key {
            let old = transition_ids_by_key.insert(key, transition_id);
            assert!(old.is_none(), "duplicate Cartesian transition key");
        }
        self.outgoing[source].push(transition_id);
        self.incoming[target].push(transition_id);
    }

    fn remove_transition(&mut self, transition_id: usize) -> WorkingTransition {
        let transition = self.transitions[transition_id]
            .take()
            .expect("Cartesian adjacency references a removed transition");
        if let Some(transition_ids_by_key) = &mut self.transition_ids_by_key {
            let removed_id = transition_ids_by_key.remove(&TransitionKey {
                source: transition.source,
                concrete_op_id: transition.concrete_op_id,
                target: transition.target,
            });
            assert_eq!(
                removed_id,
                Some(transition_id),
                "active Cartesian transition key is missing or inconsistent"
            );
        }
        self.free_transition_ids.push(transition_id);
        transition
    }

    fn remove_incident_transitions(&mut self, state_id: usize) -> Vec<WorkingTransition> {
        let mut incident = self.outgoing[state_id].clone();
        incident.extend(self.incoming[state_id].iter().copied());
        incident.sort_unstable();
        incident.dedup();

        let mut old_transitions = Vec::with_capacity(incident.len());
        let mut changed_outgoing = Vec::with_capacity(incident.len());
        let mut changed_incoming = Vec::with_capacity(incident.len());
        for transition_id in incident {
            let transition = self.remove_transition(transition_id);
            changed_outgoing.push(transition.source);
            changed_incoming.push(transition.target);
            old_transitions.push(transition);
        }
        changed_outgoing.sort_unstable();
        changed_outgoing.dedup();
        changed_incoming.sort_unstable();
        changed_incoming.dedup();

        let transitions = &self.transitions;
        for source in changed_outgoing {
            self.outgoing[source].retain(|&id| transitions[id].is_some());
        }
        for target in changed_incoming {
            self.incoming[target].retain(|&id| transitions[id].is_some());
        }
        old_transitions
    }

    fn active_transition_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.transitions
            .iter()
            .enumerate()
            .filter_map(|(id, transition)| transition.as_ref().map(|_| id))
    }

    fn transition(&self, transition_id: usize) -> &WorkingTransition {
        self.transitions[transition_id]
            .as_ref()
            .expect("Cartesian adjacency references a removed transition")
    }

    fn contains_transition(&self, key: TransitionKey) -> bool {
        if key.source == key.target {
            if let Some(loop_order) = &self.icaps_self_loop_order {
                loop_order[key.source].contains(&key.concrete_op_id)
            } else {
                self.self_loop_operator_ids[key.source].contains(key.concrete_op_id)
            }
        } else if let Some(transition_ids_by_key) = &self.transition_ids_by_key {
            transition_ids_by_key.contains_key(&key)
        } else {
            self.outgoing[key.source].iter().any(|&transition_id| {
                let transition = self.transition(transition_id);
                transition.concrete_op_id == key.concrete_op_id && transition.target == key.target
            })
        }
    }
}

#[derive(Debug, Clone)]
enum Split {
    Propositional {
        state_id: usize,
        var_id: usize,
        wanted: Vec<PropValueId>,
        witness_value: PropValueId,
        description: String,
    },
    Numeric {
        state_id: usize,
        var_id: usize,
        boundary: f64,
        lower_includes_boundary: bool,
        witness_value: f64,
        desired_contains_witness: bool,
        integer_lattice: bool,
        description: String,
    },
}

impl Split {
    fn state_id(&self) -> usize {
        match self {
            Self::Propositional { state_id, .. } | Self::Numeric { state_id, .. } => *state_id,
        }
    }

    fn description(&self) -> &str {
        match self {
            Self::Propositional { description, .. } | Self::Numeric { description, .. } => {
                description
            }
        }
    }

    fn dimension(&self) -> SplitDimension {
        match self {
            Self::Propositional { var_id, .. } => SplitDimension::Propositional(*var_id),
            Self::Numeric { var_id, .. } => SplitDimension::Numeric(*var_id),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum SplitIdentity {
    Propositional {
        state_id: usize,
        var_id: usize,
        wanted: Vec<PropValueId>,
        witness_value: PropValueId,
    },
    Numeric {
        state_id: usize,
        var_id: usize,
        boundary_bits: u64,
        lower_includes_boundary: bool,
        witness_bits: u64,
        integer_lattice: bool,
    },
}

impl From<&Split> for SplitIdentity {
    fn from(split: &Split) -> Self {
        match split {
            Split::Propositional {
                state_id,
                var_id,
                wanted,
                witness_value,
                ..
            } => Self::Propositional {
                state_id: *state_id,
                var_id: *var_id,
                wanted: wanted.clone(),
                witness_value: *witness_value,
            },
            Split::Numeric {
                state_id,
                var_id,
                boundary,
                lower_includes_boundary,
                witness_value,
                integer_lattice,
                ..
            } => Self::Numeric {
                state_id: *state_id,
                var_id: *var_id,
                boundary_bits: boundary.to_bits(),
                lower_includes_boundary: *lower_includes_boundary,
                witness_bits: witness_value.to_bits(),
                integer_lattice: *integer_lattice,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SplitDimension {
    Propositional(usize),
    Numeric(usize),
}

struct CartesianSemantics<'task> {
    task: &'task dyn AbstractNumericTask,
    propositional_axioms_by_prop_var: Vec<Vec<usize>>,
    operator_costs: Vec<f64>,
    prop_split_dependent_operators: Vec<OperatorBitSet>,
    numeric_split_dependent_operators: Vec<OperatorBitSet>,
    additive_numeric_views: Vec<Option<AdditiveNumericView>>,
    additive_effect_deltas: Vec<Vec<f64>>,
    numeric_integer_lattice: Vec<bool>,
    random_seed: Option<u64>,
    random_split_rng: Option<RefCell<SmallRng>>,
    icaps_random: Option<RefCell<ArtifactMt19937>>,
    refinement_direction: CartesianRefinementDirection,
    flaw_candidate_generation: CartesianFlawCandidateGeneration,
    split_selection_rank: Option<usize>,
    split_selection: CartesianSplitSelection,
    target_split_boundaries: Vec<f64>,
}

fn mark_fact_split_dependencies(
    task: &dyn AbstractNumericTask,
    fact: &ExplicitFact,
    propositional_axioms_by_prop_var: &[Vec<usize>],
    visiting: &mut [bool],
    prop_dependencies: &mut [bool],
    numeric_dependencies: &mut [bool],
) -> Result<()> {
    let var_id = fact.var();
    if let Some(condition) = task.numeric_conditions().for_var(var_id) {
        let mut dimensions = condition.regular_numeric_var_dependencies().to_vec();
        dimensions.extend(comparison_refinement_dimensions(task, condition));
        dimensions.sort_unstable();
        dimensions.dedup();
        for numeric_var_id in dimensions {
            numeric_dependencies[numeric_var_id] = true;
        }
        return Ok(());
    }
    if propositional_axioms_by_prop_var[var_id].is_empty() {
        prop_dependencies[var_id] = true;
        return Ok(());
    }
    ensure!(
        !visiting[var_id],
        "cyclic propositional axiom dependency through variable {var_id}"
    );
    visiting[var_id] = true;
    for &axiom_id in &propositional_axioms_by_prop_var[var_id] {
        let axiom = task
            .axioms()
            .get(axiom_id)
            .with_context(|| format!("missing propositional axiom {axiom_id}"))?;
        for condition in axiom.conditions() {
            mark_fact_split_dependencies(
                task,
                condition,
                propositional_axioms_by_prop_var,
                visiting,
                prop_dependencies,
                numeric_dependencies,
            )?;
        }
    }
    visiting[var_id] = false;
    Ok(())
}

impl<'task> CartesianSemantics<'task> {
    fn task(&self) -> &dyn AbstractNumericTask {
        self.task
    }

    fn propositional_axioms_by_prop_var(&self) -> &[Vec<usize>] {
        &self.propositional_axioms_by_prop_var
    }

    fn operator_costs(&self) -> &[f64] {
        &self.operator_costs
    }

    fn additive_numeric_views(&self) -> &[Option<AdditiveNumericView>] {
        &self.additive_numeric_views
    }

    fn additive_effect_deltas(&self) -> &[Vec<f64>] {
        &self.additive_effect_deltas
    }

    fn numeric_integer_lattice(&self) -> &[bool] {
        &self.numeric_integer_lattice
    }

    fn refinement_direction(&self) -> CartesianRefinementDirection {
        self.refinement_direction
    }

    fn flaw_candidate_generation(&self) -> CartesianFlawCandidateGeneration {
        self.flaw_candidate_generation
    }

    fn split_selection(&self) -> CartesianSplitSelection {
        self.split_selection
    }

    fn target_split_boundaries(&self) -> &[f64] {
        &self.target_split_boundaries
    }

    fn new(
        task: &'task dyn AbstractNumericTask,
        config: &CartesianAbstractionConfig,
    ) -> Result<Self> {
        validate_abstractable_goal(task).map_err(anyhow::Error::msg)?;
        for (op_id, op) in task.get_operators().iter().enumerate() {
            validate_abstraction_operator(task, op, op_id)?;
        }

        let mut propositional_axioms_by_prop_var = vec![Vec::new(); task.get_num_variables()];
        for (axiom_id, axiom) in task.axioms().iter().enumerate() {
            let var_id = axiom.var_id();
            ensure!(
                var_id < propositional_axioms_by_prop_var.len(),
                "propositional axiom {axiom_id} affects missing prop var {var_id}"
            );
            propositional_axioms_by_prop_var[var_id].push(axiom_id);
        }
        let operator_costs = task
            .get_operators()
            .iter()
            .map(|op| metric_operator_cost_from_initial_values(task, op))
            .collect();
        let operator_count = task.get_operators().len();
        let additive_numeric_views = (0..task.numeric_variables().len())
            .map(|numeric_var_id| analyze_additive_numeric_view(task, numeric_var_id))
            .collect::<Vec<_>>();
        let mut additive_effect_deltas = Vec::with_capacity(task.numeric_variables().len());
        for (numeric_var_id, variable) in task.numeric_variables().iter().enumerate() {
            let mut deltas = match variable.get_type() {
                NumericType::Regular => task
                    .get_operators()
                    .iter()
                    .filter_map(|operator| {
                        numeric_dimension_delta_for_operator(task, numeric_var_id, operator)
                    })
                    .collect::<Vec<_>>(),
                NumericType::Derived => additive_numeric_views[numeric_var_id]
                    .as_ref()
                    .map(|view| view.operator_deltas().to_vec())
                    .unwrap_or_default(),
                NumericType::Constant | NumericType::Cost => Vec::new(),
            };
            deltas.retain(|delta| delta.abs() > float_tolerance::SEARCH_EPSILON);
            deltas.sort_by(f64::total_cmp);
            deltas.dedup_by(|left, right| approximately_equal(*left, *right));
            additive_effect_deltas.push(deltas);
        }
        let mut target_split_boundaries = task
            .numeric_variables()
            .iter()
            .enumerate()
            .filter(|(_, variable)| variable.get_type() == &NumericType::Constant)
            .filter_map(|(numeric_var_id, _)| {
                task.get_initial_numeric_state_values()
                    .get(numeric_var_id)
                    .copied()
                    .filter(|value| value.is_finite())
            })
            .map(float_tolerance::canonicalize)
            .collect::<Vec<_>>();
        target_split_boundaries.sort_by(f64::total_cmp);
        target_split_boundaries.dedup_by(|left, right| left.to_bits() == right.to_bits());
        let mut prop_split_dependent_operators = (0..task.get_num_variables())
            .map(|_| OperatorBitSet::empty(operator_count))
            .collect::<Vec<_>>();
        let mut numeric_split_dependent_operators = (0..task.numeric_variables().len())
            .map(|_| OperatorBitSet::empty(operator_count))
            .collect::<Vec<_>>();
        for (op_id, op) in task.get_operators().iter().enumerate() {
            let mut prop_dependencies = vec![false; task.get_num_variables()];
            let mut numeric_dependencies = vec![false; task.numeric_variables().len()];
            let mut visiting = vec![false; task.get_num_variables()];
            for precondition in op.preconditions() {
                mark_fact_split_dependencies(
                    task,
                    precondition,
                    &propositional_axioms_by_prop_var,
                    &mut visiting,
                    &mut prop_dependencies,
                    &mut numeric_dependencies,
                )?;
            }
            for effect in op.effects() {
                let var_id = effect.var_id();
                if !task.numeric_conditions().is_condition_var(var_id)
                    && propositional_axioms_by_prop_var[var_id].is_empty()
                {
                    prop_dependencies[var_id] = true;
                }
            }
            for effect in op.assignment_effects() {
                let var_id = effect.affected_var_id();
                if task.numeric_variables()[var_id].get_type() == &NumericType::Regular {
                    numeric_dependencies[var_id] = true;
                }
            }
            for (numeric_var_id, view) in additive_numeric_views.iter().enumerate() {
                if view.as_ref().is_some_and(|view| {
                    view.operator_delta(op_id)
                        .is_ok_and(|delta| delta.abs() > float_tolerance::SEARCH_EPSILON)
                }) {
                    numeric_dependencies[numeric_var_id] = true;
                }
            }
            debug_assert_eq!(
                prop_dependencies.len(),
                task.get_num_variables(),
                "operator {op_id} propositional dependency width changed"
            );
            for (var_id, depends) in prop_dependencies.into_iter().enumerate() {
                if depends {
                    prop_split_dependent_operators[var_id].insert(op_id);
                }
            }
            for (var_id, depends) in numeric_dependencies.into_iter().enumerate() {
                if depends {
                    numeric_split_dependent_operators[var_id].insert(op_id);
                }
            }
        }
        let icaps_random = if matches!(
            config.split_selection,
            CartesianSplitSelection::Icaps26(Icaps26SplitSelection::Random)
        ) {
            let seed = config.random_seed.unwrap_or(2011);
            ensure!(
                u32::try_from(seed).is_ok(),
                "ICAPS artifact random seed exceeds 32 bits: {seed}"
            );
            Some(RefCell::new(ArtifactMt19937::new(seed as u32)))
        } else {
            None
        };
        let random_split_rng = matches!(config.split_selection, CartesianSplitSelection::Random)
            .then(|| RefCell::new(SmallRng::seed_from_u64(config.random_seed.unwrap_or(2011))));
        let initial_numeric = task.get_initial_numeric_state_values();
        let mut numeric_integer_lattice = initial_numeric
            .iter()
            .map(|&value| approximately_equal(value, value.round()))
            .collect::<Vec<_>>();
        for op in task.get_operators() {
            for effect in op.assignment_effects() {
                let rhs = initial_numeric[effect.var_id()];
                let preserves_integers = match effect.operation() {
                    AssignmentOperation::Plus
                    | AssignmentOperation::Minus
                    | AssignmentOperation::Assign
                    | AssignmentOperation::Times => approximately_equal(rhs, rhs.round()),
                    AssignmentOperation::Divide => approximately_equal(rhs.abs(), 1.0),
                };
                numeric_integer_lattice[effect.affected_var_id()] &= preserves_integers;
            }
        }
        for (numeric_var_id, view) in additive_numeric_views.iter().enumerate() {
            let Some(view) = view else {
                continue;
            };
            let initial_value = view.evaluate(initial_numeric);
            numeric_integer_lattice[numeric_var_id] =
                approximately_equal(initial_value, initial_value.round())
                    && (0..operator_count).all(|op_id| {
                        view.operator_delta(op_id)
                            .is_ok_and(|delta| approximately_equal(delta, delta.round()))
                    });
        }
        Ok(Self {
            task,
            propositional_axioms_by_prop_var,
            operator_costs,
            prop_split_dependent_operators,
            numeric_split_dependent_operators,
            additive_numeric_views,
            additive_effect_deltas,
            numeric_integer_lattice,
            random_seed: config.random_seed,
            random_split_rng,
            icaps_random,
            refinement_direction: config.refinement_direction,
            flaw_candidate_generation: config.flaw_candidate_generation,
            split_selection_rank: config.split_selection_rank,
            split_selection: config.split_selection,
            target_split_boundaries,
        })
    }

    fn choose_keyed_index(&self, keys: &[u64], tag: u64) -> usize {
        debug_assert!(
            !keys.is_empty(),
            "cannot choose from an empty Cartesian candidate set"
        );
        let Some(seed) = self.random_seed else {
            return 0;
        };
        keys.iter()
            .enumerate()
            .min_by_key(|(_, key)| mix_seed(seed ^ tag ^ **key))
            .map(|(index, _)| index)
            .expect("nonempty Cartesian key set has no minimum")
    }

    fn choose_split_index(&self, candidates: &[Split], tag: u64) -> usize {
        debug_assert!(
            !candidates.is_empty(),
            "cannot choose from an empty split set"
        );
        if let Some(rank) = self.split_selection_rank {
            let mut indices = (0..candidates.len()).collect::<Vec<_>>();
            indices.sort_by_key(|&index| {
                let dimension = match candidates[index].dimension() {
                    SplitDimension::Propositional(var_id) => (0usize, var_id),
                    SplitDimension::Numeric(var_id) => (1usize, var_id),
                };
                (dimension, split_choice_key(self, &candidates[index]))
            });
            return indices[rank % indices.len()];
        }
        let keys = candidates
            .iter()
            .map(|split| split_choice_key(self, split))
            .collect::<Vec<_>>();
        self.choose_keyed_index(&keys, tag)
    }

    fn choose_icaps_random_index(&self, candidate_count: usize) -> usize {
        debug_assert!(candidate_count > 0, "cannot choose from an empty split set");
        self.icaps_random
            .as_ref()
            .expect("ICAPS random selector must initialize its RNG")
            .borrow_mut()
            .uniform_index(candidate_count)
    }

    fn choose_random_split_index(&self, candidate_count: usize) -> usize {
        debug_assert!(candidate_count > 0, "cannot choose from an empty split set");
        self.random_split_rng
            .as_ref()
            .expect("native random selector must initialize its RNG")
            .borrow_mut()
            .gen_range(0..candidate_count)
    }

    fn operator_depends_on_split(&self, op_id: usize, dimension: SplitDimension) -> bool {
        self.split_dependent_operators(dimension).contains(op_id)
    }

    fn split_dependent_operators(&self, dimension: SplitDimension) -> &OperatorBitSet {
        match dimension {
            SplitDimension::Propositional(var_id) => &self.prop_split_dependent_operators[var_id],
            SplitDimension::Numeric(var_id) => &self.numeric_split_dependent_operators[var_id],
        }
    }

    fn invariant_split_dimension_overlaps(
        &self,
        source: &StateRegion,
        target: &StateRegion,
        dimension: SplitDimension,
    ) -> bool {
        match dimension {
            SplitDimension::Propositional(var_id) => sorted_value_sets_overlap(
                &source.propositions[var_id],
                &target.propositions[var_id],
            ),
            SplitDimension::Numeric(var_id) => {
                source.numeric[var_id].intersects(&target.numeric[var_id])
            }
        }
    }

    fn may_transition_after_independent_split(
        &self,
        source: &StateRegion,
        op_id: usize,
        target: &StateRegion,
        dimension: SplitDimension,
    ) -> Result<bool> {
        debug_assert!(!self.operator_depends_on_split(op_id, dimension));
        let result = self.invariant_split_dimension_overlaps(source, target, dimension);
        #[cfg(debug_assertions)]
        assert_eq!(
            result,
            self.may_transition(source, op_id, target)?,
            "Cartesian split-dependency routing disagrees with full transition semantics for operator {op_id} and dimension {dimension:?}"
        );
        Ok(result)
    }

    fn trivial_region(&self) -> Result<StateRegion> {
        let propositions = (0..self.task.get_num_variables())
            .map(|var_id| {
                let size = self
                    .task
                    .get_variable_domain_size(var_id)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                ensure!(size > 0, "propositional var {var_id} has an empty domain");
                ensure!(
                    u32::try_from(size).is_ok(),
                    "propositional var {var_id} domain is too large: {size}"
                );
                Ok((0..size as PropValueId).collect())
            })
            .collect::<Result<Vec<_>>>()?;
        let initial_numeric = self.task.get_initial_numeric_state_values();
        let numeric: Vec<_> = self
            .task
            .numeric_variables()
            .iter()
            .enumerate()
            .map(|(var_id, var)| {
                if matches!(var.get_type(), NumericType::Constant) {
                    Interval::singleton(float_tolerance::canonicalize(initial_numeric[var_id]))
                } else {
                    Interval::unbounded()
                }
            })
            .collect();
        Ok(StateRegion {
            propositions: propositions.into(),
            numeric: numeric.into(),
        })
    }

    fn region_admits_fact(&self, region: &StateRegion, fact: &ExplicitFact) -> Result<bool> {
        let mut visiting = vec![false; self.task.get_num_variables()];
        self.region_admits_fact_inner(region, fact, &mut visiting)
    }

    fn region_admits_fact_inner(
        &self,
        region: &StateRegion,
        fact: &ExplicitFact,
        visiting: &mut [bool],
    ) -> Result<bool> {
        let var_id = fact.var();
        if let Some(axiom_id) = self.task.numeric_conditions().id_for_var(var_id) {
            let (may_true, may_false) = self.comparison_truths(region, axiom_id)?;
            return Ok(match fact.value() {
                0 => may_true,
                1 => may_false,
                2 => may_true || may_false,
                value => bail!("invalid comparison proposition value {value} for var {var_id}"),
            });
        }
        if !self.propositional_axioms_by_prop_var[var_id].is_empty() {
            ensure!(
                !visiting[var_id],
                "cyclic propositional axiom support for variable {var_id}"
            );
            visiting[var_id] = true;
            let result = (|| {
                let default_value = self.propositional_axiom_default(var_id)?;
                if fact.value() == default_value {
                    for &axiom_id in &self.propositional_axioms_by_prop_var[var_id] {
                        let axiom = &self.task.axioms()[axiom_id];
                        if self.all_conditions_guaranteed(region, axiom.conditions(), visiting)? {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }

                for &axiom_id in &self.propositional_axioms_by_prop_var[var_id] {
                    let axiom = &self.task.axioms()[axiom_id];
                    if axiom.effect_value() == fact.value()
                        && self.all_conditions_admitted(region, axiom.conditions(), visiting)?
                    {
                        return Ok(true);
                    }
                }
                Ok(false)
            })();
            visiting[var_id] = false;
            return result;
        }
        Ok(region
            .propositions
            .get(var_id)
            .is_some_and(|values| values.binary_search(&(fact.value() as u32)).is_ok()))
    }

    fn region_guarantees_fact(&self, region: &StateRegion, fact: &ExplicitFact) -> Result<bool> {
        let mut visiting = vec![false; self.task.get_num_variables()];
        self.region_guarantees_fact_inner(region, fact, &mut visiting)
    }

    fn region_guarantees_fact_inner(
        &self,
        region: &StateRegion,
        fact: &ExplicitFact,
        visiting: &mut [bool],
    ) -> Result<bool> {
        let var_id = fact.var();
        if let Some(axiom_id) = self.task.numeric_conditions().id_for_var(var_id) {
            let (may_true, may_false) = self.comparison_truths(region, axiom_id)?;
            return Ok(match fact.value() {
                0 => may_true && !may_false,
                1 => may_false && !may_true,
                2 => false,
                value => bail!("invalid comparison proposition value {value} for var {var_id}"),
            });
        }
        if !self.propositional_axioms_by_prop_var[var_id].is_empty() {
            ensure!(
                !visiting[var_id],
                "cyclic propositional axiom support for variable {var_id}"
            );
            visiting[var_id] = true;
            let result = (|| {
                let default_value = self.propositional_axiom_default(var_id)?;
                if fact.value() == default_value {
                    for &axiom_id in &self.propositional_axioms_by_prop_var[var_id] {
                        let axiom = &self.task.axioms()[axiom_id];
                        if self.all_conditions_admitted(region, axiom.conditions(), visiting)? {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }

                for &axiom_id in &self.propositional_axioms_by_prop_var[var_id] {
                    let axiom = &self.task.axioms()[axiom_id];
                    if axiom.effect_value() == fact.value()
                        && self.all_conditions_guaranteed(region, axiom.conditions(), visiting)?
                    {
                        return Ok(true);
                    }
                }
                Ok(false)
            })();
            visiting[var_id] = false;
            return result;
        }
        let Some(values) = region.propositions.get(var_id) else {
            return Ok(false);
        };
        Ok(values.len() == 1 && values[0] == fact.value() as u32)
    }

    fn all_conditions_admitted(
        &self,
        region: &StateRegion,
        conditions: &[ExplicitFact],
        visiting: &mut [bool],
    ) -> Result<bool> {
        for condition in conditions {
            if !self.region_admits_fact_inner(region, condition, visiting)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn all_conditions_guaranteed(
        &self,
        region: &StateRegion,
        conditions: &[ExplicitFact],
        visiting: &mut [bool],
    ) -> Result<bool> {
        for condition in conditions {
            if !self.region_guarantees_fact_inner(region, condition, visiting)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn propositional_axiom_default(&self, var_id: usize) -> Result<usize> {
        let axiom_ids = self
            .propositional_axioms_by_prop_var
            .get(var_id)
            .with_context(|| format!("missing propositional variable {var_id}"))?;
        let (&first_axiom_id, remaining) = axiom_ids
            .split_first()
            .with_context(|| format!("variable {var_id} has no propositional axiom"))?;
        let default_value = self.task.axioms()[first_axiom_id].precondition_value();
        for &axiom_id in remaining {
            ensure!(
                self.task.axioms()[axiom_id].precondition_value() == default_value,
                "propositional axioms for variable {var_id} disagree on default value"
            );
        }
        Ok(default_value)
    }

    fn comparison_truths(&self, region: &StateRegion, tree_id: usize) -> Result<(bool, bool)> {
        let tree = self
            .task
            .numeric_conditions()
            .get(tree_id)
            .with_context(|| format!("missing comparison tree {tree_id}"))?;
        ensure!(
            region.numeric.iter().all(|interval| !interval.is_empty()),
            "comparison tree {tree_id} evaluated on an empty Cartesian region"
        );
        Ok(match tree.evaluate_interval(&region.numeric) {
            Some(true) => (true, false),
            Some(false) => (false, true),
            None => (true, true),
        })
    }

    fn operator_may_apply(&self, source: &StateRegion, op_id: usize) -> Result<bool> {
        let op = self
            .task
            .get_operators()
            .get(op_id)
            .with_context(|| format!("missing operator {op_id}"))?;
        for fact in op.preconditions() {
            if !self.region_admits_fact(source, fact)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn propositional_dimension_may_transition(
        &self,
        source: &StateRegion,
        op_id: usize,
        target: &StateRegion,
        var_id: usize,
    ) -> bool {
        debug_assert!(
            !self.task.numeric_conditions().is_condition_var(var_id)
                && self.propositional_axioms_by_prop_var[var_id].is_empty(),
            "derived proposition {var_id} has no explicit transition relation"
        );
        let op = &self.task.get_operators()[op_id];
        if let Some(effect) = op.effects().iter().find(|effect| effect.var_id() == var_id) {
            debug_assert!(
                effect.conditions().is_empty(),
                "validated Cartesian operator {op_id} has a conditional effect"
            );
            return target.propositions[var_id]
                .binary_search(&(effect.value() as PropValueId))
                .is_ok();
        }
        if matches!(self.split_selection, CartesianSplitSelection::Icaps26(_))
            && let Some(precondition) = op
                .preconditions()
                .iter()
                .find(|precondition| precondition.var() == var_id)
        {
            return target.propositions[var_id]
                .binary_search(&(precondition.value() as PropValueId))
                .is_ok();
        }
        sorted_value_sets_overlap(&source.propositions[var_id], &target.propositions[var_id])
    }

    fn split_dimension_may_transition(
        &self,
        source: &StateRegion,
        op_id: usize,
        target: &StateRegion,
        dimension: SplitDimension,
    ) -> Result<bool> {
        Ok(match dimension {
            SplitDimension::Propositional(var_id) => {
                self.propositional_dimension_may_transition(source, op_id, target, var_id)
            }
            SplitDimension::Numeric(var_id) => self.numeric_dimension_may_transition(
                source.numeric[var_id],
                op_id,
                target.numeric[var_id],
                var_id,
            )?,
        })
    }

    fn numeric_dimension_may_transition(
        &self,
        source: Interval,
        op_id: usize,
        target: Interval,
        var_id: usize,
    ) -> Result<bool> {
        let Some(preimage) = self.numeric_effect_preimage(target, op_id, var_id)? else {
            return Ok(false);
        };
        let source = if matches!(self.split_selection, CartesianSplitSelection::Icaps26(_)) {
            source.intersection(&self.icaps_numeric_precondition(op_id, var_id)?)
        } else {
            source
        };
        Ok(!source.is_empty() && preimage.intersects(&source))
    }

    fn icaps_numeric_precondition(&self, op_id: usize, var_id: usize) -> Result<Interval> {
        let mut interval = Interval::unbounded();
        for fact in self.task.get_operators()[op_id].preconditions() {
            let Some(tree_id) = self.task.numeric_conditions().id_for_var(fact.var()) else {
                continue;
            };
            let (condition_var_id, condition) =
                desired_comparison_interval(self, tree_id, fact.value())?;
            if condition_var_id == var_id {
                interval = interval.intersection(&condition);
            }
        }
        Ok(interval)
    }

    fn parent_loop_source_to_split_children(
        &self,
        source: &StateRegion,
        op_id: usize,
        targets: [&StateRegion; 2],
        dimension: SplitDimension,
    ) -> Result<[bool; 2]> {
        let may_apply = self.operator_may_apply(source, op_id)?;
        let mut result = [false; 2];
        if may_apply {
            for (index, target) in targets.into_iter().enumerate() {
                result[index] =
                    self.split_dimension_may_transition(source, op_id, target, dimension)?;
            }
        }
        #[cfg(debug_assertions)]
        for (index, target) in targets.into_iter().enumerate() {
            assert_eq!(
                result[index],
                self.may_transition(source, op_id, target)?,
                "split-dimension routing disagrees with full transition semantics for parent-loop operator {op_id} and dimension {dimension:?}"
            );
        }
        Ok(result)
    }

    fn may_transition(
        &self,
        source: &StateRegion,
        op_id: usize,
        target: &StateRegion,
    ) -> Result<bool> {
        if !self.operator_may_apply(source, op_id)? {
            return Ok(false);
        }
        for var_id in 0..self.task.get_num_variables() {
            if self.task.numeric_conditions().is_condition_var(var_id)
                || !self.propositional_axioms_by_prop_var[var_id].is_empty()
            {
                continue;
            }
            if !self.propositional_dimension_may_transition(source, op_id, target, var_id) {
                return Ok(false);
            }
        }

        for (numeric_var_id, variable) in self.task.numeric_variables().iter().enumerate() {
            match variable.get_type() {
                NumericType::Constant => {
                    if !source.numeric[numeric_var_id].intersects(&target.numeric[numeric_var_id]) {
                        return Ok(false);
                    }
                }
                NumericType::Regular => {
                    if !self.numeric_dimension_may_transition(
                        source.numeric[numeric_var_id],
                        op_id,
                        target.numeric[numeric_var_id],
                        numeric_var_id,
                    )? {
                        return Ok(false);
                    }
                }
                NumericType::Derived => {
                    if self.additive_numeric_views[numeric_var_id].is_some()
                        && !self.numeric_dimension_may_transition(
                            source.numeric[numeric_var_id],
                            op_id,
                            target.numeric[numeric_var_id],
                            numeric_var_id,
                        )?
                    {
                        return Ok(false);
                    }
                }
                NumericType::Cost => {}
            }
        }
        Ok(true)
    }

    fn numeric_effect_preimage(
        &self,
        target: Interval,
        op_id: usize,
        numeric_var_id: usize,
    ) -> Result<Option<Interval>> {
        let mut preimage = target;
        if let Some(view) = self.additive_numeric_views[numeric_var_id].as_ref() {
            let delta = float_tolerance::canonicalize(view.operator_delta(op_id)?);
            preimage.apply_reverse_op(&AssignmentOperation::Plus, &Interval::singleton(delta));
            return Ok(Some(preimage));
        }
        let op = &self.task.get_operators()[op_id];
        for effect in op
            .assignment_effects()
            .iter()
            .filter(|effect| effect.affected_var_id() == numeric_var_id)
            .rev()
        {
            let rhs = float_tolerance::canonicalize(
                self.task.get_initial_numeric_state_values()[effect.var_id()],
            );
            match effect.operation() {
                AssignmentOperation::Assign => {
                    if !preimage.contains(rhs) {
                        return Ok(None);
                    }
                    preimage = Interval::unbounded();
                }
                AssignmentOperation::Plus => {
                    preimage
                        .apply_reverse_op(&AssignmentOperation::Plus, &Interval::singleton(rhs));
                }
                AssignmentOperation::Minus => {
                    preimage
                        .apply_reverse_op(&AssignmentOperation::Minus, &Interval::singleton(rhs));
                }
                AssignmentOperation::Times => {
                    if rhs == 0.0 {
                        if !preimage.contains(0.0) {
                            return Ok(None);
                        }
                        preimage = Interval::unbounded();
                    } else {
                        preimage.apply_reverse_op(
                            &AssignmentOperation::Times,
                            &Interval::singleton(rhs),
                        );
                    }
                }
                AssignmentOperation::Divide => {
                    preimage
                        .apply_reverse_op(&AssignmentOperation::Divide, &Interval::singleton(rhs));
                }
            }
            preimage = preimage.canonicalized();
        }
        Ok(Some(preimage))
    }

    fn transition_source_footprint(
        &self,
        source: &StateRegion,
        op_id: usize,
        target: &StateRegion,
    ) -> Result<Option<StateRegion>> {
        debug_assert_eq!(
            source.numeric.len(),
            target.numeric.len(),
            "Cartesian transition source/target numeric dimension mismatch"
        );
        let mut footprint = source.clone();
        for (numeric_var_id, variable) in self.task.numeric_variables().iter().enumerate() {
            match variable.get_type() {
                NumericType::Constant => {
                    if !source.numeric[numeric_var_id].intersects(&target.numeric[numeric_var_id]) {
                        return Ok(None);
                    }
                }
                NumericType::Regular => {
                    let Some(preimage) = self.numeric_effect_preimage(
                        target.numeric[numeric_var_id],
                        op_id,
                        numeric_var_id,
                    )?
                    else {
                        return Ok(None);
                    };
                    let regressed = source.numeric[numeric_var_id].intersection(&preimage);
                    if regressed.is_empty() {
                        return Ok(None);
                    }
                    if regressed != source.numeric[numeric_var_id] {
                        Arc::make_mut(&mut footprint.numeric)[numeric_var_id] = regressed;
                    }
                }
                NumericType::Derived => {
                    if self.additive_numeric_views[numeric_var_id].is_none() {
                        continue;
                    }
                    let preimage = self
                        .numeric_effect_preimage(
                            target.numeric[numeric_var_id],
                            op_id,
                            numeric_var_id,
                        )?
                        .expect("additive-view preimage is always defined");
                    let regressed = source.numeric[numeric_var_id].intersection(&preimage);
                    if regressed.is_empty() {
                        return Ok(None);
                    }
                    if regressed != source.numeric[numeric_var_id] {
                        Arc::make_mut(&mut footprint.numeric)[numeric_var_id] = regressed;
                    }
                }
                NumericType::Cost => {}
            }
        }
        Ok(Some(footprint))
    }

    fn region_is_goal(&self, region: &StateRegion) -> Result<bool> {
        for goal_id in 0..self.task.get_num_goals() {
            if !self.region_admits_fact(region, self.task.get_goal_fact(goal_id))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn concrete_prop_values(&self, packer: &StatePacker, packed: &[u64], out: &mut Vec<usize>) {
        out.clear();
        out.extend(
            (0..self.task.get_num_variables()).map(|var_id| packer.get(packed, var_id) as usize),
        );
    }
}

pub struct CartesianAbstractionGenerator {
    config: CartesianAbstractionConfig,
}

impl CartesianAbstractionGenerator {
    pub fn new(config: CartesianAbstractionConfig) -> Result<Self> {
        ensure!(config.max_states > 0, "Cartesian max_states must be > 0");
        ensure!(
            matches!(
                config.flaw_kind,
                FlawKind::Progression | FlawKind::ExecuteEntirePlan
            ),
            "Cartesian abstractions support flaw_kind=progression and flaw_kind=execute_entire_plan, not flaw_kind={}",
            config.flaw_kind
        );
        Ok(Self { config })
    }

    pub fn generate(&self, task: &dyn AbstractNumericTask) -> Result<CartesianAbstraction> {
        self.generate_from_root(task, None)
    }

    fn generate_from_root(
        &self,
        task: &dyn AbstractNumericTask,
        refinement_root: Option<&CartesianConcreteState>,
    ) -> Result<CartesianAbstraction> {
        let start = Instant::now();
        let semantics = CartesianSemantics::new(task, &self.config)?;
        let initial_region = semantics.trivial_region()?;
        let operator_count = semantics.task.get_operators().len();
        let mut working = if matches!(
            self.config.split_selection,
            CartesianSplitSelection::Icaps26(_)
        ) {
            WorkingAbstraction::new_icaps26(initial_region, operator_count)
        } else {
            WorkingAbstraction::new(initial_region, operator_count)
        };
        for op_id in 0..task.get_operators().len() {
            if semantics.may_transition(&working.states[0], op_id, &working.states[0])? {
                working.add_transition(0, op_id, 0);
            }
        }
        let state_packer = Arc::new(make_prop_state_packer(task));
        let axiom_evaluator = AxiomEvaluator::new(Arc::new(task), state_packer.clone());
        let (initial_propositions, initial_numeric) =
            get_initial_state(task, &state_packer, &axiom_evaluator)?;
        let refinement_root = match refinement_root {
            Some(root) => {
                ensure!(
                    root.propositions.len() == state_packer.num_bins(),
                    "Cartesian refinement root has {} proposition bins, expected {}",
                    root.propositions.len(),
                    state_packer.num_bins()
                );
                ensure!(
                    root.numeric.len() == task.numeric_variables().len(),
                    "Cartesian refinement root has {} numeric values, expected {}",
                    root.numeric.len(),
                    task.numeric_variables().len()
                );
                root.clone()
            }
            None => CartesianConcreteState {
                propositions: initial_propositions.clone(),
                numeric: initial_numeric.clone(),
            },
        };
        let mut refinements: usize = 0;
        let poll_memory_every_iteration = matches!(
            self.config.split_selection,
            CartesianSplitSelection::Icaps26(_)
        );

        let mut shortest_paths = if matches!(
            self.config.split_selection,
            CartesianSplitSelection::Icaps26(_)
        ) {
            compute_shortest_paths_with_goals(&working, &semantics, vec![true])?
        } else {
            compute_shortest_paths(&working, &semantics)?
        };
        let mut stable_abstract_search = matches!(
            self.config.abstract_plan_selection,
            CartesianAbstractPlanSelection::StableAStar
        )
        .then(StableAbstractSearch::trivial);
        let mut pending_flaw = None;
        let mut solved_plan = None;
        let run = CegarDriver::new(usize::MAX, None).run_from_zero_based(
            start,
            |_iteration, _deadline| {
                if poll_memory_every_iteration
                    && !crate::resource_limits::poll_and_release_if_exceeded()
                {
                    return Ok(CegarIterationResult::Stop(CegarStopReason::MemoryLimit));
                }
                let selected_plan = if let Some(abstract_search) = &mut stable_abstract_search {
                    let mut initial_prop_values = Vec::new();
                    semantics.concrete_prop_values(
                        &state_packer,
                        &refinement_root.propositions,
                        &mut initial_prop_values,
                    );
                    let initial_state = working
                        .hierarchy
                        .map_state(&initial_prop_values, &refinement_root.numeric)?;
                    match abstract_search.find_plan(
                        &working,
                        &semantics,
                        initial_state,
                        shortest_paths.goal_flags(),
                    )? {
                        Some(plan) => Some(plan),
                        None => {
                            return Err(RefinementRootDeadEnd::new(initial_state).into());
                        }
                    }
                } else {
                    None
                };
                let check = match self.config.flaw_kind {
                    FlawKind::Progression => replay_optimal_abstract_trace(
                        &working,
                        &semantics,
                        &shortest_paths,
                        &state_packer,
                        &axiom_evaluator,
                        &refinement_root,
                        selected_plan.as_deref(),
                    )?,
                    FlawKind::ExecuteEntirePlan => replay_entire_optimal_abstract_trace(
                        &working,
                        &semantics,
                        &shortest_paths,
                        &state_packer,
                        &axiom_evaluator,
                        &refinement_root,
                    )?,
                    _ => unreachable!(
                        "unsupported Cartesian flaw kind passed constructor validation"
                    ),
                };
                match check {
                    PlanCheck::ConcretePlan(plan) => {
                        solved_plan = Some(plan);
                        Ok(CegarIterationResult::Stop(CegarStopReason::ConcretePlan))
                    }
                    PlanCheck::AbstractDeadEnd(abstract_state_id) => {
                        Err(RefinementRootDeadEnd::new(abstract_state_id).into())
                    }
                    PlanCheck::Refine(split) => {
                        if working.states.len() >= self.config.max_states {
                            pending_flaw = Some(split.description().to_string());
                            return Ok(CegarIterationResult::Stop(CegarStopReason::SizeLimit));
                        }
                        if self
                            .config
                            .max_time
                            .is_some_and(|max_time| start.elapsed() >= max_time)
                        {
                            pending_flaw = Some(split.description().to_string());
                            return Ok(CegarIterationResult::Stop(CegarStopReason::TimeLimit));
                        }
                        if self.config.debug {
                            info!(
                                "Cartesian refinement {} at {} states: {split:?}",
                                refinements,
                                working.states.len()
                            );
                        }
                        let old_state_id = split.state_id();
                        let new_state_id = working.apply_split(&semantics, split)?;
                        update_shortest_paths_after_split(
                            &working,
                            &semantics,
                            &mut shortest_paths,
                            old_state_id,
                            new_state_id,
                        )?;
                        if let Some(abstract_search) = &mut stable_abstract_search {
                            abstract_search.inherit_split_state(old_state_id, new_state_id);
                        }
                        refinements += 1;
                        Ok(CegarIterationResult::Continue)
                    }
                }
            },
        )?;
        let stop_reason = match run.stop_reason {
            CegarStopReason::ConcretePlan => CartesianStopReason::ConcretePlan,
            CegarStopReason::SizeLimit => CartesianStopReason::StateLimit,
            CegarStopReason::TimeLimit => CartesianStopReason::TimeLimit,
            CegarStopReason::MemoryLimit => CartesianStopReason::MemoryLimit,
            CegarStopReason::IterationLimit | CegarStopReason::NoRefinableFlaw => {
                unreachable!("Cartesian CEGAR driver returned unsupported stop reason")
            }
        };

        let working_transition_count = working.active_transition_ids().count();
        let mut initial_prop_values = Vec::new();
        semantics.concrete_prop_values(
            &state_packer,
            &initial_propositions,
            &mut initial_prop_values,
        );
        let initial_state_hash = working
            .hierarchy
            .map_state(&initial_prop_values, &initial_numeric)?;
        let (transition_system, distance_table, relevant_operator_ids, footprints) =
            if self.config.retain_transition_system {
                finalize_abstraction(
                    &working,
                    &semantics,
                    initial_state_hash,
                    self.config.combine_labels,
                    self.config.compute_operator_footprints,
                )?
            } else {
                finalize_standalone_abstraction(
                    &working,
                    &semantics,
                    &shortest_paths,
                    initial_state_hash,
                )?
            };
        let transition_count = if self.config.retain_transition_system {
            transition_system.transitions.len()
        } else {
            working_transition_count
        };
        if let Some(plan) = &solved_plan {
            validate_concrete_plan(
                &semantics,
                &state_packer,
                &axiom_evaluator,
                &refinement_root,
                plan,
            )?;
            let mut root_prop_values = Vec::new();
            semantics.concrete_prop_values(
                &state_packer,
                &refinement_root.propositions,
                &mut root_prop_values,
            );
            let root_state_id = working
                .hierarchy
                .map_state(&root_prop_values, &refinement_root.numeric)?;
            let h = distance_table.distances[root_state_id];
            ensure!(
                (plan.cost() - h).abs() <= 1e-7,
                "concrete Cartesian plan cost {} differs from abstract h(refinement root) {h}",
                plan.cost()
            );
        }
        info!(
            "Cartesian abstraction: states={}, transitions={}, refinements={}, h(init)={}, stop={stop_reason:?}, elapsed={:.3}s",
            distance_table.distances.len(),
            transition_count,
            refinements,
            distance_table.distances[distance_table.initial_state_hash],
            start.elapsed().as_secs_f64()
        );
        Ok(CartesianAbstraction {
            hierarchy: working.hierarchy,
            distance_table,
            transition_system,
            relevant_operator_ids,
            abstract_operator_footprints: footprints,
            metadata: CartesianAbstractionMetadata {
                solved_by_self: solved_plan.is_some(),
                abstraction_use: AbstractionUse::Standalone,
                stop_reason,
                pending_flaw,
                refinements,
                collection_goal_id: None,
                collection_variant_id: None,
                refinement_direction: self.config.refinement_direction,
                split_selection_rank: self.config.split_selection_rank,
                concrete_plan_operator_ids: solved_plan.map(ConcretePlan::into_operator_ids),
                progressive_refinement_root: false,
                transition_count,
            },
        })
    }
}

pub struct CartesianAbstractionCollectionGenerator {
    config: CartesianAbstractionCollectionConfig,
}

impl CartesianAbstractionCollectionGenerator {
    pub fn new(config: CartesianAbstractionCollectionConfig) -> Result<Self> {
        ensure!(
            config.max_collection_states > 0,
            "Cartesian max_collection_size must be > 0"
        );
        ensure!(
            config.variants_per_goal > 0,
            "Cartesian variants_per_goal must be > 0"
        );
        CartesianAbstractionGenerator::new(config.abstraction.clone())?;
        Ok(Self { config })
    }

    /// Builds variants for task goals until the configured collection limit
    /// is reached, or one full-task abstraction when the goal is empty. With
    /// progressive roots enabled, each variant replays its validated concrete
    /// plans and uses a reached non-goal state as the next CEGAR refinement
    /// root. Reaching the complete task goal makes that lane terminal; later
    /// members use the task initial state independently. After the requested
    /// variants, missing initial-root goal specialists are added within the
    /// same resource limits.
    ///
    /// Each member changes only the goal view. Operators, state mappings, and
    /// concrete operator IDs stay identical to the base task. Changing the
    /// refinement root only chooses counterexamples; every hierarchy still
    /// partitions the full task state space, so the resulting transition
    /// systems remain admissible components for canonical and cost-partitioned
    /// collection heuristics.
    pub fn generate(&self, task: &dyn AbstractNumericTask) -> Result<Vec<CartesianAbstraction>> {
        let goal_count = task.get_num_goals();
        let variants_per_goal = if goal_count == 0 {
            1
        } else {
            self.config.variants_per_goal
        };
        let abstraction_count = goal_count
            .max(1)
            .checked_mul(variants_per_goal)
            .context("Cartesian collection abstraction count overflow")?;
        let progressive_goal_roots = self.config.progressive_goal_roots;

        let start = Instant::now();
        let mut remaining_states = self.config.max_collection_states;
        let mut abstractions = Vec::with_capacity(abstraction_count);
        let mut abstraction_id = 0usize;
        let mut state = CartesianCollectionState::new(
            task,
            goal_count,
            variants_per_goal,
            progressive_goal_roots,
        )?;

        let mut stop_reason = "requested abstraction count reached";
        while state.has_work_left(abstraction_count, progressive_goal_roots) {
            if remaining_states < 2 && !abstractions.is_empty() {
                stop_reason = "collection size limit";
                break;
            }
            let member = match state.select_next_member(
                task,
                goal_count,
                abstraction_count,
                variants_per_goal,
                progressive_goal_roots,
            )? {
                NextMember::Build(member) => member,
                NextMember::Stop(reason) => {
                    stop_reason = reason;
                    break;
                }
            };
            let remaining_time = match remaining_collection_time(
                self.config.total_max_time,
                start,
                abstractions.is_empty(),
            ) {
                CollectionBudget::Remaining(remaining) => remaining,
                CollectionBudget::Exhausted => {
                    stop_reason = "collection time limit";
                    break;
                }
            };

            let BuiltMember {
                abstraction,
                built_from_initial_root,
                reset_progressive_root,
            } = self.build_member(
                task,
                &state,
                &member,
                goal_count,
                abstraction_id,
                MemberBudget {
                    remaining_states,
                    remaining_time,
                },
            )?;

            let state_count = abstraction.num_states();
            ensure!(
                state_count <= remaining_states,
                "Cartesian goal abstraction used {state_count} states with only {remaining_states} remaining"
            );
            remaining_states -= state_count;

            state.record_built_member(
                &member,
                goal_count,
                built_from_initial_root,
                reset_progressive_root,
                &abstraction,
            );
            state.advance_lane(
                task,
                goal_count,
                &member,
                &abstraction,
                reset_progressive_root,
            )?;

            abstractions.push(abstraction);
            abstraction_id += 1;
            if !crate::resource_limits::poll_and_release_if_exceeded() {
                stop_reason = "memory limit";
                break;
            }
        }

        if stop_reason == "requested abstraction count reached"
            && progressive_goal_roots
            && state
                .initial_root_goal_covered
                .iter()
                .all(|covered| *covered)
        {
            stop_reason = "requested abstractions and initial-root goal coverage reached";
        }
        state.log_summary(
            &abstractions,
            goal_count,
            self.config.max_collection_states - remaining_states,
            start.elapsed().as_secs_f64(),
            stop_reason,
        );
        Ok(abstractions)
    }

    /// Configuration for one member: the collection's per-abstraction
    /// configuration, capped by what is left of the collection's state and
    /// time budgets, then specialised for this variant.
    ///
    /// A complementary collection alternates progression and regression by
    /// variant parity and ranks split selection by variant; otherwise only the
    /// random seed varies, and only when more than one variant per goal is
    /// requested.
    fn member_abstraction_config(
        &self,
        member: &CollectionMember,
        goal_count: usize,
        budget: MemberBudget,
    ) -> CartesianAbstractionConfig {
        let MemberBudget {
            remaining_states,
            remaining_time,
        } = budget;
        let mut config = self.config.abstraction.clone();
        config.max_states = config.max_states.min(remaining_states);
        // Whichever of the two budgets exist bind; the tighter one wins.
        config.max_time = config.max_time.into_iter().chain(remaining_time).min();

        let variant = member.construction_variant_id();
        let base_seed = config.random_seed.unwrap_or(0);
        if goal_count > 0 && self.config.collection_strategy.is_complementary() {
            config.refinement_direction = if variant.is_multiple_of(2) {
                CartesianRefinementDirection::Progression
            } else {
                CartesianRefinementDirection::Regression
            };
            config.split_selection_rank = Some(variant / 2);
            config.random_seed =
                (variant > 0).then(|| derive_variant_seed(base_seed, member.goal_id, variant - 1));
        } else if goal_count > 0 && self.config.variants_per_goal > 1 && variant > 0 {
            config.random_seed = Some(derive_variant_seed(base_seed, member.goal_id, variant - 1));
        }
        config
    }

    /// Build one member's abstraction: derive its configuration, restrict the
    /// task to its goal, refine from its lane's root, and tag the result so
    /// collection heuristics can tell members apart.
    fn build_member(
        &self,
        task: &dyn AbstractNumericTask,
        state: &CartesianCollectionState,
        member: &CollectionMember,
        goal_count: usize,
        abstraction_id: usize,
        budget: MemberBudget,
    ) -> Result<BuiltMember> {
        let abstraction_config = self.member_abstraction_config(member, goal_count, budget);
        let goal_task = (goal_count > 0)
            .then(|| SingleGoalTask::new(task, *task.get_goal_fact(member.goal_id)));
        let abstraction_task = goal_task
            .as_ref()
            .map_or(task, |goal_task| goal_task as &dyn AbstractNumericTask);
        debug!(
            "Cartesian collection: building abstraction {}, goal={}, variant={}, continuation={}, initial_root_specialist={}, direction={:?}, split_rank={:?}, max_states={}, seed={:?}",
            abstraction_id + 1,
            member.goal_id,
            member.variant_id,
            member.is_continuation(),
            member.is_initial_root_specialist(),
            abstraction_config.refinement_direction,
            abstraction_config.split_selection_rank,
            abstraction_config.max_states,
            abstraction_config.random_seed
        );
        let generator = CartesianAbstractionGenerator::new(abstraction_config)?;

        // A specialist and a terminal lane both refine from the task initial
        // state, which `generate_from_root` takes as `None`.
        let lane_is_complete = state.lane_is_complete(member.variant_id);
        let refinement_root = (!member.is_initial_root_specialist() && !lane_is_complete)
            .then(|| state.refinement_roots.get(member.variant_id))
            .flatten();
        let built_from_initial_root = member.is_initial_root_specialist()
            || refinement_root.is_none()
            || !state.lane_root_advanced(member.variant_id);

        let (mut abstraction, reset_progressive_root) = refine_collection_member(
            &generator,
            abstraction_task,
            refinement_root,
            abstraction_id,
            member,
        )?;

        abstraction.metadata.collection_goal_id = (goal_count > 0).then_some(member.goal_id);
        abstraction.metadata.collection_variant_id = (goal_count > 0).then_some(member.variant_id);
        abstraction.metadata.abstraction_use = AbstractionUse::CollectionMember;
        abstraction.metadata.progressive_refinement_root = !member.is_initial_root_specialist()
            && !lane_is_complete
            && state.lane_root_advanced(member.variant_id)
            && !reset_progressive_root;

        Ok(BuiltMember {
            abstraction,
            built_from_initial_root,
            reset_progressive_root,
        })
    }
}

/// One finished member, with the two facts the collection bookkeeping needs
/// about how it was built.
struct BuiltMember {
    abstraction: CartesianAbstraction,
    /// The member refined from the task initial state, so it covers its goal
    /// for every state.
    built_from_initial_root: bool,
    /// The lane's progressive root was an abstract dead end and the member had
    /// to be rebuilt from the task initial state.
    reset_progressive_root: bool,
}

/// Why the collection is building a member.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MemberKind {
    /// One of the `abstraction_count` members the configuration asks for.
    Scheduled,
    /// A goal that no member built from the task initial state covers yet.
    InitialRootSpecialist,
    /// A goal some lane already attempted, retried from that lane's advanced
    /// refinement root.
    Continuation,
}

/// One member the collection has decided to build.
struct CollectionMember {
    goal_id: usize,
    variant_id: usize,
    kind: MemberKind,
}

impl CollectionMember {
    /// Variant the abstraction *configuration* derives from. An initial-root
    /// specialist reuses variant 0's configuration; its own `variant_id` only
    /// records that it sits past the scheduled variants.
    fn construction_variant_id(&self) -> usize {
        match self.kind {
            MemberKind::InitialRootSpecialist => 0,
            MemberKind::Scheduled | MemberKind::Continuation => self.variant_id,
        }
    }

    fn is_initial_root_specialist(&self) -> bool {
        self.kind == MemberKind::InitialRootSpecialist
    }

    fn is_continuation(&self) -> bool {
        self.kind == MemberKind::Continuation
    }
}

enum NextMember {
    Build(CollectionMember),
    Stop(&'static str),
}

/// What is left of the collection's total time budget.
enum CollectionBudget {
    /// Time limit for the next member, `None` when the collection is
    /// unbounded.
    Remaining(Option<Duration>),
    /// The budget is spent and the collection already has a member.
    Exhausted,
}

fn remaining_collection_time(
    total_max_time: Option<Duration>,
    start: Instant,
    collection_is_empty: bool,
) -> CollectionBudget {
    let Some(total_max_time) = total_max_time else {
        return CollectionBudget::Remaining(None);
    };
    let elapsed = start.elapsed();
    if elapsed < total_max_time {
        return CollectionBudget::Remaining(Some(total_max_time - elapsed));
    }
    if collection_is_empty {
        // An empty collection is useless, so the first member is built even
        // with no time left — with a zero budget, which stops it immediately.
        CollectionBudget::Remaining(Some(Duration::ZERO))
    } else {
        CollectionBudget::Exhausted
    }
}

/// Build one member, retrying from the task initial state when the lane's
/// progressive refinement root turns out to be an abstract dead end. The
/// returned flag reports whether that retry happened.
fn refine_collection_member(
    generator: &CartesianAbstractionGenerator,
    abstraction_task: &dyn AbstractNumericTask,
    refinement_root: Option<&CartesianConcreteState>,
    abstraction_id: usize,
    member: &CollectionMember,
) -> Result<(CartesianAbstraction, bool)> {
    match generator.generate_from_root(abstraction_task, refinement_root) {
        Ok(abstraction) => Ok((abstraction, false)),
        Err(error)
            if refinement_root.is_some()
                && error.downcast_ref::<RefinementRootDeadEnd>().is_some() =>
        {
            info!(
                "Cartesian collection: progressive root is an abstract dead end for goal {}, variant {}; rebuilding this member from the task initial state",
                member.goal_id, member.variant_id
            );
            let abstraction = generator
                .generate_from_root(abstraction_task, None)
                .with_context(|| {
                    format!(
                        "failed to rebuild Cartesian collection abstraction {abstraction_id} from the task initial state"
                    )
                })?;
            Ok((abstraction, true))
        }
        Err(error) => Err(error).with_context(|| {
            format!("failed to build Cartesian collection abstraction {abstraction_id}")
        }),
    }
}

/// Mutable state of one collection run.
///
/// With progressive goal roots enabled the collection keeps `variants_per_goal`
/// independent *lanes*. A lane owns a refinement root that walks forward
/// through the concrete plans its members produce; it becomes terminal once it
/// reaches the complete task goal or its root turns out to be a dead end, and
/// from then on its members start from the task initial state again.
struct CartesianCollectionState {
    /// `None` when progressive goal roots are off, in which case there are no
    /// lanes at all.
    initial_refinement_root: Option<CartesianConcreteState>,
    refinement_roots: Vec<CartesianConcreteState>,
    satisfied_goals_by_root: Vec<usize>,
    progressive_root_advanced: Vec<bool>,
    progressive_lane_complete: Vec<bool>,
    initial_root_goal_covered: Vec<bool>,
    variants_built_by_goal: Vec<usize>,
    best_initial_h_by_goal: Vec<f64>,
    /// `(goal_id, variant_id)` pairs to retry from a lane's advanced root.
    continuation_queue: VecDeque<(usize, usize)>,
    initial_abstractions_built: usize,
}

impl CartesianCollectionState {
    fn new(
        task: &dyn AbstractNumericTask,
        goal_count: usize,
        variants_per_goal: usize,
        progressive_goal_roots: bool,
    ) -> Result<Self> {
        let initial_refinement_root = (progressive_goal_roots && goal_count > 0)
            .then(|| initial_cartesian_concrete_state(task))
            .transpose()?;
        let refinement_roots = initial_refinement_root
            .as_ref()
            .map_or_else(Vec::new, |root| vec![root.clone(); variants_per_goal]);
        let satisfied_goals_by_root = refinement_roots
            .iter()
            .map(|root| count_satisfied_cartesian_goals(task, root))
            .collect::<Result<Vec<_>>>()?;
        let lane_count = refinement_roots.len();
        Ok(Self {
            initial_refinement_root,
            refinement_roots,
            satisfied_goals_by_root,
            progressive_root_advanced: vec![false; lane_count],
            progressive_lane_complete: vec![false; lane_count],
            initial_root_goal_covered: vec![false; goal_count],
            variants_built_by_goal: vec![0usize; goal_count],
            best_initial_h_by_goal: vec![0.0f64; goal_count],
            continuation_queue: VecDeque::new(),
            initial_abstractions_built: 0,
        })
    }

    /// `false` when progressive goal roots are off — there is then no lane
    /// that could have advanced.
    fn lane_root_advanced(&self, variant_id: usize) -> bool {
        self.progressive_root_advanced
            .get(variant_id)
            .copied()
            .unwrap_or(false)
    }

    /// `false` when progressive goal roots are off — there is then no lane
    /// that could have completed.
    fn lane_is_complete(&self, variant_id: usize) -> bool {
        self.progressive_lane_complete
            .get(variant_id)
            .copied()
            .unwrap_or(false)
    }

    /// More scheduled members are due, a lane queued a retry, or some goal
    /// still lacks a member built from the task initial state.
    fn has_work_left(&self, abstraction_count: usize, progressive_goal_roots: bool) -> bool {
        self.initial_abstractions_built < abstraction_count
            || !self.continuation_queue.is_empty()
            || (progressive_goal_roots
                && self
                    .initial_root_goal_covered
                    .iter()
                    .any(|covered| !covered))
    }

    fn select_next_member(
        &mut self,
        task: &dyn AbstractNumericTask,
        goal_count: usize,
        abstraction_count: usize,
        variants_per_goal: usize,
        progressive_goal_roots: bool,
    ) -> Result<NextMember> {
        let scheduled_member_pending = self.initial_abstractions_built < abstraction_count;
        let initial_root_specialist_goal = (!scheduled_member_pending && progressive_goal_roots)
            .then(|| {
                self.initial_root_goal_covered
                    .iter()
                    .position(|covered| !covered)
            })
            .flatten();
        let continuation = if progressive_goal_roots
            && !scheduled_member_pending
            && initial_root_specialist_goal.is_none()
        {
            self.pop_unsatisfied_continuation(task)?
        } else {
            None
        };

        if goal_count == 0 {
            // A goal-free task gets exactly one abstraction of the full task.
            return Ok(NextMember::Build(CollectionMember {
                goal_id: 0,
                variant_id: 0,
                kind: MemberKind::Scheduled,
            }));
        }
        if scheduled_member_pending {
            let Some(goal_id) = select_next_cartesian_collection_goal(
                &self.variants_built_by_goal,
                &self.best_initial_h_by_goal,
                variants_per_goal,
            ) else {
                return Ok(NextMember::Stop("requested abstraction count reached"));
            };
            return Ok(NextMember::Build(CollectionMember {
                goal_id,
                variant_id: self.variants_built_by_goal[goal_id],
                kind: MemberKind::Scheduled,
            }));
        }
        if let Some(goal_id) = initial_root_specialist_goal {
            return Ok(NextMember::Build(CollectionMember {
                goal_id,
                variant_id: variants_per_goal,
                kind: MemberKind::InitialRootSpecialist,
            }));
        }
        if let Some((goal_id, variant_id)) = continuation {
            return Ok(NextMember::Build(CollectionMember {
                goal_id,
                variant_id,
                kind: MemberKind::Continuation,
            }));
        }
        Ok(NextMember::Stop(
            "requested abstractions and initial-root goal coverage reached",
        ))
    }

    /// Drop queued retries whose goal the lane's root now satisfies anyway,
    /// and return the first one that still has work to do.
    fn pop_unsatisfied_continuation(
        &mut self,
        task: &dyn AbstractNumericTask,
    ) -> Result<Option<(usize, usize)>> {
        while let Some((goal_id, variant_id)) = self.continuation_queue.pop_front() {
            let root = self
                .refinement_roots
                .get(variant_id)
                .expect("progressive continuation references missing root");
            if !cartesian_goal_is_satisfied(task, root, goal_id)? {
                return Ok(Some((goal_id, variant_id)));
            }
        }
        Ok(None)
    }

    fn record_built_member(
        &mut self,
        member: &CollectionMember,
        goal_count: usize,
        built_from_initial_root: bool,
        reset_progressive_root: bool,
        abstraction: &CartesianAbstraction,
    ) {
        if goal_count == 0 {
            self.initial_abstractions_built += 1;
            return;
        }
        if built_from_initial_root || reset_progressive_root {
            self.initial_root_goal_covered[member.goal_id] = true;
        }
        // Specialists and retries fill gaps; only scheduled members count
        // against the requested abstraction count.
        if member.kind == MemberKind::Scheduled {
            self.variants_built_by_goal[member.goal_id] += 1;
            self.initial_abstractions_built += 1;
            let initial_h =
                abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash];
            self.best_initial_h_by_goal[member.goal_id] =
                self.best_initial_h_by_goal[member.goal_id].max(initial_h);
        }
    }

    /// Replay the member's concrete plan onto its lane's refinement root, so
    /// the next member of that lane refines from further along.
    fn advance_lane(
        &mut self,
        task: &dyn AbstractNumericTask,
        goal_count: usize,
        member: &CollectionMember,
        abstraction: &CartesianAbstraction,
        reset_progressive_root: bool,
    ) -> Result<()> {
        let variant_id = member.variant_id;
        if member.is_initial_root_specialist()
            || self.lane_is_complete(variant_id)
            || variant_id >= self.refinement_roots.len()
        {
            return Ok(());
        }
        if reset_progressive_root {
            self.make_lane_terminal(task, variant_id)?;
            debug!(
                "Cartesian collection: dead root made progressive variant {variant_id} terminal after rebuilding goal {} from the initial state",
                member.goal_id
            );
            return Ok(());
        }
        let Some(operator_ids) = abstraction.metadata.concrete_plan_operator_ids.as_deref() else {
            debug!(
                "Cartesian collection: goal {} variant {variant_id} produced no concrete plan; progressive root remains unchanged",
                member.goal_id
            );
            return Ok(());
        };

        let previous_satisfied_goals = self.satisfied_goals_by_root[variant_id];
        let root = &mut self.refinement_roots[variant_id];
        *root = replay_cartesian_operator_sequence(task, root, operator_ids)?;
        let satisfied_goals = count_satisfied_cartesian_goals(task, root)?;
        self.satisfied_goals_by_root[variant_id] = satisfied_goals;
        debug!(
            "Cartesian collection: advanced progressive root for variant {variant_id} through {} concrete operators; satisfied_goals={satisfied_goals}/{goal_count}",
            operator_ids.len(),
        );

        if satisfied_goals == goal_count {
            self.make_lane_terminal(task, variant_id)?;
            debug!(
                "Cartesian collection: full goal reached for variant {variant_id}; made the progressive lane terminal"
            );
            return Ok(());
        }
        self.progressive_root_advanced[variant_id] = true;
        if satisfied_goals > previous_satisfied_goals {
            self.queue_reopened_goals(task, variant_id)?;
        }
        Ok(())
    }

    /// Return a lane to the task initial state and stop advancing it, because
    /// its root is a dead end or has reached the complete task goal. Retries
    /// queued for the lane go with it.
    fn make_lane_terminal(
        &mut self,
        task: &dyn AbstractNumericTask,
        variant_id: usize,
    ) -> Result<()> {
        let initial_root = self
            .initial_refinement_root
            .as_ref()
            .expect("progressive refinement root requires an initial root")
            .clone();
        self.refinement_roots[variant_id] = initial_root;
        self.progressive_root_advanced[variant_id] = false;
        self.progressive_lane_complete[variant_id] = true;
        self.satisfied_goals_by_root[variant_id] =
            count_satisfied_cartesian_goals(task, &self.refinement_roots[variant_id])?;
        self.continuation_queue
            .retain(|(_, queued_variant_id)| *queued_variant_id != variant_id);
        Ok(())
    }

    /// The lane's root now satisfies goals it did not before, so goals this
    /// lane already attempted may be worth another abstraction from here.
    fn queue_reopened_goals(
        &mut self,
        task: &dyn AbstractNumericTask,
        variant_id: usize,
    ) -> Result<()> {
        let root = &self.refinement_roots[variant_id];
        for (retry_goal_id, &variants_built) in self.variants_built_by_goal.iter().enumerate() {
            let was_already_attempted = variants_built > variant_id;
            if was_already_attempted
                && !cartesian_goal_is_satisfied(task, root, retry_goal_id)?
                && !self
                    .continuation_queue
                    .contains(&(retry_goal_id, variant_id))
            {
                self.continuation_queue
                    .push_back((retry_goal_id, variant_id));
            }
        }
        Ok(())
    }

    fn log_summary(
        &self,
        abstractions: &[CartesianAbstraction],
        goal_count: usize,
        states_used: usize,
        elapsed_secs: f64,
        stop_reason: &str,
    ) {
        info!(
            "Cartesian collection: abstractions={}, states={states_used}, elapsed={elapsed_secs:.3}s, stop_reason={stop_reason}",
            abstractions.len(),
        );
        if !self.satisfied_goals_by_root.is_empty() {
            info!(
                "Cartesian collection: progressive root goal coverage={:?}/{goal_count}",
                self.satisfied_goals_by_root
            );
        }
        if !self.initial_root_goal_covered.is_empty() {
            info!(
                "Cartesian collection: initial-root goal coverage={}/{goal_count}",
                self.initial_root_goal_covered
                    .iter()
                    .filter(|covered| **covered)
                    .count(),
            );
        }
    }
}

fn initial_cartesian_concrete_state(
    task: &dyn AbstractNumericTask,
) -> Result<CartesianConcreteState> {
    let state_packer = Arc::new(make_prop_state_packer(task));
    let axiom_evaluator = AxiomEvaluator::new(Arc::new(task), state_packer.clone());
    let (propositions, numeric) = get_initial_state(task, &state_packer, &axiom_evaluator)?;
    Ok(CartesianConcreteState {
        propositions,
        numeric,
    })
}

fn replay_cartesian_operator_sequence(
    task: &dyn AbstractNumericTask,
    root: &CartesianConcreteState,
    operator_ids: &[usize],
) -> Result<CartesianConcreteState> {
    let state_packer = Arc::new(make_prop_state_packer(task));
    let axiom_evaluator = AxiomEvaluator::new(Arc::new(task), state_packer.clone());
    let mut next = root.clone();
    for (step, &operator_id) in operator_ids.iter().enumerate() {
        let operator = task.get_operators().get(operator_id).with_context(|| {
            format!("progressive Cartesian plan step {step} has invalid operator {operator_id}")
        })?;
        ensure!(
            operator.preconditions().iter().all(|fact| fact_is_hold(
                fact,
                &state_packer,
                &next.propositions
            )),
            "progressive Cartesian plan operator {operator_id} ({}) is inapplicable at step {step}",
            operator.name()
        );
        progress_concrete_state(
            operator,
            &axiom_evaluator,
            &state_packer,
            &mut next.propositions,
            &mut next.numeric,
        )?;
    }
    Ok(next)
}

fn count_satisfied_cartesian_goals(
    task: &dyn AbstractNumericTask,
    state: &CartesianConcreteState,
) -> Result<usize> {
    let state_packer = make_prop_state_packer(task);
    ensure!(
        state.propositions.len() == state_packer.num_bins(),
        "Cartesian concrete state has {} proposition bins, expected {}",
        state.propositions.len(),
        state_packer.num_bins()
    );
    Ok((0..task.get_num_goals())
        .filter(|&goal_id| {
            fact_is_hold(
                task.get_goal_fact(goal_id),
                &state_packer,
                &state.propositions,
            )
        })
        .count())
}

fn cartesian_goal_is_satisfied(
    task: &dyn AbstractNumericTask,
    state: &CartesianConcreteState,
    goal_id: usize,
) -> Result<bool> {
    let state_packer = make_prop_state_packer(task);
    ensure!(
        state.propositions.len() == state_packer.num_bins(),
        "Cartesian concrete state has {} proposition bins, expected {}",
        state.propositions.len(),
        state_packer.num_bins()
    );
    ensure!(
        goal_id < task.get_num_goals(),
        "Cartesian goal id {goal_id} exceeds {} goals",
        task.get_num_goals()
    );
    Ok(fact_is_hold(
        task.get_goal_fact(goal_id),
        &state_packer,
        &state.propositions,
    ))
}

fn select_next_cartesian_collection_goal(
    variants_built_by_goal: &[usize],
    best_initial_h_by_goal: &[f64],
    variants_per_goal: usize,
) -> Option<usize> {
    assert_eq!(
        variants_built_by_goal.len(),
        best_initial_h_by_goal.len(),
        "Cartesian collection goal statistics must have equal lengths"
    );
    let guaranteed_variants = variants_per_goal.min(2);
    if let Some(minimum_built) = variants_built_by_goal
        .iter()
        .copied()
        .filter(|&count| count < guaranteed_variants)
        .min()
    {
        return variants_built_by_goal
            .iter()
            .position(|&count| count == minimum_built && count < guaranteed_variants);
    }

    variants_built_by_goal
        .iter()
        .enumerate()
        .filter(|(_, count)| **count < variants_per_goal)
        .max_by(|(left_id, _), (right_id, _)| {
            best_initial_h_by_goal[*left_id]
                .total_cmp(&best_initial_h_by_goal[*right_id])
                .then_with(|| right_id.cmp(left_id))
        })
        .map(|(goal_id, _)| goal_id)
}

fn icaps26_retire_arc(working: &mut WorkingAbstraction, transition_id: usize) -> WorkingTransition {
    assert!(
        working.transition_ids_by_key.is_none(),
        "ICAPS arc refinement requires unindexed transitions"
    );
    working.transitions[transition_id]
        .take()
        .expect("ICAPS adjacency references a removed transition")
}

fn icaps26_release_arc_slot(working: &mut WorkingAbstraction, transition_id: usize) {
    assert!(
        working.transitions[transition_id].is_none(),
        "ICAPS retired transition slot is occupied"
    );
    working.free_transition_ids.push(transition_id);
}

fn icaps26_swap_remove_arc(adjacency: &mut Vec<usize>, transition_id: usize) {
    let position = adjacency
        .iter()
        .position(|&candidate| candidate == transition_id)
        .expect("ICAPS counterpart adjacency is missing an arc");
    adjacency.swap_remove(position);
}

fn add_icaps26_propositional_loop_replacements(
    working: &mut WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    op_id: usize,
    var_id: usize,
    old_state_id: usize,
    new_state_id: usize,
) {
    let op = &semantics.task.get_operators()[op_id];
    let pre = op
        .preconditions()
        .iter()
        .find(|fact| fact.var() == var_id)
        .map(ExplicitFact::value);
    let effect = op
        .effects()
        .iter()
        .find(|effect| effect.var_id() == var_id)
        .map(|effect| effect.value());
    let post = effect.or(pre);
    let old_contains = |value: usize| {
        working.states[old_state_id].propositions[var_id]
            .binary_search(&(value as PropValueId))
            .is_ok()
    };

    match (pre, post) {
        (None, None) => {
            working.add_transition(old_state_id, op_id, old_state_id);
            working.add_transition(new_state_id, op_id, new_state_id);
        }
        (None, Some(post)) if !old_contains(post) => {
            working.add_transition(old_state_id, op_id, new_state_id);
            working.add_transition(new_state_id, op_id, new_state_id);
        }
        (None, Some(post)) => {
            assert!(old_contains(post));
            working.add_transition(old_state_id, op_id, old_state_id);
            working.add_transition(new_state_id, op_id, old_state_id);
        }
        (Some(pre), Some(post)) if old_contains(pre) => {
            if old_contains(post) {
                working.add_transition(old_state_id, op_id, old_state_id);
            } else {
                working.add_transition(old_state_id, op_id, new_state_id);
            }
        }
        (Some(_), Some(post)) => {
            if old_contains(post) {
                working.add_transition(new_state_id, op_id, old_state_id);
            } else {
                working.add_transition(new_state_id, op_id, new_state_id);
            }
        }
        (Some(_), None) => unreachable!("a prevail condition defines its own post value"),
    }
}

fn apply_icaps26_transition_split(
    working: &mut WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    old_state_id: usize,
    new_state_id: usize,
    split_dimension: SplitDimension,
    old_loop_order: Vec<usize>,
) -> Result<()> {
    let old_incoming = std::mem::take(&mut working.incoming[old_state_id]);
    for transition_id in old_incoming {
        let transition = icaps26_retire_arc(working, transition_id);
        debug_assert_eq!(transition.target, old_state_id);
        let source = transition.source;
        for target in [old_state_id, new_state_id] {
            if semantics.split_dimension_may_transition(
                &working.states[source],
                transition.concrete_op_id,
                &working.states[target],
                split_dimension,
            )? {
                working.add_transition(source, transition.concrete_op_id, target);
            }
        }
        icaps26_swap_remove_arc(&mut working.outgoing[source], transition_id);
        icaps26_release_arc_slot(working, transition_id);
    }

    let old_outgoing = std::mem::take(&mut working.outgoing[old_state_id]);
    for transition_id in old_outgoing {
        let transition = icaps26_retire_arc(working, transition_id);
        debug_assert_eq!(transition.source, old_state_id);
        let target = transition.target;
        for source in [old_state_id, new_state_id] {
            if semantics.operator_may_apply(&working.states[source], transition.concrete_op_id)?
                && semantics.split_dimension_may_transition(
                    &working.states[source],
                    transition.concrete_op_id,
                    &working.states[target],
                    split_dimension,
                )?
            {
                working.add_transition(source, transition.concrete_op_id, target);
            }
        }
        icaps26_swap_remove_arc(&mut working.incoming[target], transition_id);
        icaps26_release_arc_slot(working, transition_id);
    }

    for op_id in old_loop_order {
        match split_dimension {
            SplitDimension::Propositional(var_id) => {
                add_icaps26_propositional_loop_replacements(
                    working,
                    semantics,
                    op_id,
                    var_id,
                    old_state_id,
                    new_state_id,
                );
            }
            SplitDimension::Numeric(_) => {
                let old_targets = semantics.parent_loop_source_to_split_children(
                    &working.states[old_state_id],
                    op_id,
                    [&working.states[old_state_id], &working.states[new_state_id]],
                    split_dimension,
                )?;
                let new_targets = semantics.parent_loop_source_to_split_children(
                    &working.states[new_state_id],
                    op_id,
                    [&working.states[old_state_id], &working.states[new_state_id]],
                    split_dimension,
                )?;
                if old_targets[0] {
                    working.add_transition(old_state_id, op_id, old_state_id);
                }
                if new_targets[1] {
                    working.add_transition(new_state_id, op_id, new_state_id);
                }
                if old_targets[1] {
                    working.add_transition(old_state_id, op_id, new_state_id);
                }
                if new_targets[0] {
                    working.add_transition(new_state_id, op_id, old_state_id);
                }
            }
        }
    }
    Ok(())
}

impl WorkingAbstraction {
    fn apply_split(&mut self, semantics: &CartesianSemantics<'_>, split: Split) -> Result<usize> {
        let working = self;
        let old_state_id = split.state_id();
        let split_dimension = split.dimension();
        let old_region = working
            .states
            .get(old_state_id)
            .with_context(|| format!("missing split state {old_state_id}"))?
            .clone();
        let leaf_node_id = working.leaf_node_ids[old_state_id];
        let new_state_id = working.states.len();
        let (old_child, new_child) = match split {
            Split::Propositional {
                var_id,
                wanted,
                witness_value,
                ..
            } => {
                let current = old_region
                    .propositions
                    .get(var_id)
                    .with_context(|| format!("split references missing prop var {var_id}"))?;
                let wanted_child_values: Vec<_> = current
                    .iter()
                    .copied()
                    .filter(|value| wanted.binary_search(value).is_ok())
                    .collect();
                let other_child_values: Vec<_> = current
                    .iter()
                    .copied()
                    .filter(|value| wanted.binary_search(value).is_err())
                    .collect();
                ensure!(
                    !wanted_child_values.is_empty() && !other_child_values.is_empty(),
                    "non-strict propositional Cartesian split on var {var_id}: current={current:?}, wanted={wanted:?}"
                );
                let witness_is_wanted = wanted_child_values.binary_search(&witness_value).is_ok();
                let mut wanted_region = old_region.clone();
                Arc::make_mut(&mut wanted_region.propositions)[var_id] = wanted_child_values;
                let mut other_region = old_region.clone();
                Arc::make_mut(&mut other_region.propositions)[var_id] = other_child_values;
                working.propositional_refinement_counts[var_id] += 1;
                working.hierarchy.split_propositional(
                    leaf_node_id,
                    old_state_id,
                    new_state_id,
                    var_id,
                    wanted,
                    witness_is_wanted,
                )?;
                if witness_is_wanted {
                    (wanted_region, other_region)
                } else {
                    (other_region, wanted_region)
                }
            }
            Split::Numeric {
                var_id,
                boundary,
                lower_includes_boundary,
                witness_value,
                integer_lattice,
                ..
            } => {
                let parent = old_region.numeric[var_id];
                let (lower, upper) = numeric_split_intervals(
                    parent,
                    boundary,
                    lower_includes_boundary,
                    integer_lattice,
                )?;
                let witness_is_lower = lower.contains(witness_value);
                ensure!(
                    witness_is_lower ^ upper.contains(witness_value),
                    "numeric split does not place witness {witness_value} in exactly one child"
                );
                let mut lower_region = old_region.clone();
                Arc::make_mut(&mut lower_region.numeric)[var_id] = lower;
                let mut upper_region = old_region.clone();
                Arc::make_mut(&mut upper_region.numeric)[var_id] = upper;
                working.numeric_refinement_counts[var_id] += 1;
                working.hierarchy.split_numeric(
                    leaf_node_id,
                    old_state_id,
                    new_state_id,
                    NumericSplit {
                        var_id,
                        boundary,
                        lower_includes_boundary,
                        old_state_is_lower: witness_is_lower,
                    },
                )?;
                if witness_is_lower {
                    (lower_region, upper_region)
                } else {
                    (upper_region, lower_region)
                }
            }
        };

        working.states[old_state_id] = old_child;
        working.states.push(new_child);
        working.outgoing.push(Vec::new());
        working.incoming.push(Vec::new());
        let icaps_old_loop_order = working.icaps_self_loop_order.as_mut().map(|loop_order| {
            let old = std::mem::take(&mut loop_order[old_state_id]);
            loop_order.push(Vec::new());
            old
        });
        let old_self_loops = if icaps_old_loop_order.is_some() {
            working
                .self_loop_operator_ids
                .push(OperatorBitSet::empty(0));
            None
        } else {
            let operator_count = semantics.task.get_operators().len();
            let old_self_loops = std::mem::replace(
                &mut working.self_loop_operator_ids[old_state_id],
                OperatorBitSet::empty(operator_count),
            );
            let split_dependent_operators = semantics.split_dependent_operators(split_dimension);
            working.self_loop_operator_ids[old_state_id] =
                old_self_loops.clone_without(split_dependent_operators);
            working
                .self_loop_operator_ids
                .push(old_self_loops.clone_without(split_dependent_operators));
            Some(old_self_loops)
        };
        let new_leaf_nodes = match &working.hierarchy.nodes[leaf_node_id] {
            RefinementNode::Propositional {
                wanted_child,
                other_child,
                ..
            } => (*wanted_child, *other_child),
            RefinementNode::Numeric {
                lower_child,
                upper_child,
                ..
            } => (*lower_child, *upper_child),
            RefinementNode::Leaf { .. } => unreachable!(),
        };
        let old_leaf_node = if matches!(working.hierarchy.nodes[new_leaf_nodes.0], RefinementNode::Leaf { state_id } if state_id == old_state_id)
        {
            new_leaf_nodes.0
        } else {
            new_leaf_nodes.1
        };
        let new_leaf_node = if old_leaf_node == new_leaf_nodes.0 {
            new_leaf_nodes.1
        } else {
            new_leaf_nodes.0
        };
        working.leaf_node_ids[old_state_id] = old_leaf_node;
        working.leaf_node_ids.push(new_leaf_node);

        if let Some(old_loop_order) = icaps_old_loop_order {
            apply_icaps26_transition_split(
                working,
                semantics,
                old_state_id,
                new_state_id,
                split_dimension,
                old_loop_order,
            )?;
            return Ok(new_state_id);
        }

        let old_self_loops = old_self_loops.expect("native Cartesian split lost its self loops");
        let split_dependent_operators = semantics.split_dependent_operators(split_dimension);
        let old_transitions = working.remove_incident_transitions(old_state_id);
        for transition in old_transitions {
            debug_assert!(
                transition.source != transition.target,
                "Cartesian non-loop storage contains a self loop"
            );
            let sources: &[usize] = if transition.source == old_state_id {
                &[old_state_id, new_state_id]
            } else {
                std::slice::from_ref(&transition.source)
            };
            let targets: &[usize] = if transition.target == old_state_id {
                &[old_state_id, new_state_id]
            } else {
                std::slice::from_ref(&transition.target)
            };
            for &source in sources {
                for &target in targets {
                    let may_transition = if semantics
                        .operator_depends_on_split(transition.concrete_op_id, split_dimension)
                    {
                        semantics.may_transition(
                            &working.states[source],
                            transition.concrete_op_id,
                            &working.states[target],
                        )?
                    } else {
                        semantics.may_transition_after_independent_split(
                            &working.states[source],
                            transition.concrete_op_id,
                            &working.states[target],
                            split_dimension,
                        )?
                    };
                    if may_transition {
                        working.add_transition(source, transition.concrete_op_id, target);
                    }
                }
            }
        }
        for concrete_op_id in old_self_loops.intersection_iter(split_dependent_operators) {
            for source in [old_state_id, new_state_id] {
                let targets = [old_state_id, new_state_id];
                let may_targets = semantics.parent_loop_source_to_split_children(
                    &working.states[source],
                    concrete_op_id,
                    [&working.states[old_state_id], &working.states[new_state_id]],
                    split_dimension,
                )?;
                for (target, may_transition) in targets.into_iter().zip(may_targets) {
                    if may_transition {
                        working.add_transition(source, concrete_op_id, target);
                    }
                }
            }
        }
        Ok(new_state_id)
    }
}

pub struct CartesianAbstractionHeuristic {
    name: String,
    abstraction: CartesianAbstraction,
    prop_scratch: std::cell::RefCell<Vec<usize>>,
    numeric_scratch: std::cell::RefCell<Vec<f64>>,
}

impl CartesianAbstractionHeuristic {
    pub fn new(name: Option<String>, abstraction: CartesianAbstraction) -> Self {
        Self {
            name: name.unwrap_or_else(|| "cartesian_abstraction".to_string()),
            abstraction,
            prop_scratch: std::cell::RefCell::new(Vec::new()),
            numeric_scratch: std::cell::RefCell::new(Vec::new()),
        }
    }

    pub fn abstraction(&self) -> &CartesianAbstraction {
        &self.abstraction
    }

    pub fn discard_transition_data(&mut self) {
        self.abstraction.discard_transition_data();
    }

    pub fn into_abstraction(self) -> CartesianAbstraction {
        self.abstraction
    }

    pub fn abstract_state_id(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<usize, EvaluationError> {
        let registry = eval_state.state_registry();
        let mut prop = self.prop_scratch.borrow_mut();
        eval_state.state().fill_state(registry, &mut prop);
        let mut numeric = self.numeric_scratch.borrow_mut();
        registry
            .fill_numeric_vars(eval_state.state(), &mut numeric)
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to read numeric state for Cartesian abstraction: {error:?}"
                ))
            })?;
        self.abstraction
            .hierarchy
            .map_state(&prop, &numeric)
            .map_err(|error| EvaluationError::ComputationFailed(error.to_string()))
    }
}

impl Heuristic for CartesianAbstractionHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let state_id = self.abstract_state_id(eval_state)?;
        self.abstraction
            .distance_table
            .distances
            .get(state_id)
            .copied()
            .ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "Cartesian abstract state id {state_id} out of bounds"
                ))
            })
    }

    fn proves_initial_state_optimal(&self) -> bool {
        self.abstraction.metadata.solved_by_self
            && self
                .abstraction
                .metadata
                .abstraction_use
                .permits_initial_optimality_proof()
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}
