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
mod regression;
mod shortest_paths;
mod split_generation;
mod split_selector;
#[cfg(test)]
mod tests;

use finalize::*;
use flaw_splits::*;
use plan_replay::*;
use regression::*;
use shortest_paths::*;
use split_generation::*;
use split_selector::*;

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use ordered_float::NotNan;
use planforge_sas::axioms::{AxiomEvaluator, ComparisonOperator};
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, ExplicitFact, ExplicitValueIndex, NumericType,
    NumericValue, OperatorIndex, VariableIndex, metric_operator_cost_from_initial_values,
};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::state_packer::StatePacker;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tracing::info;

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;
use crate::evaluation::validate_abstractable_goal;

use super::abstraction_collections::cost_partitioning::{
    AbstractOperatorRegions, AbstractTransition, AbstractTransitionSystem, OperatorRegion,
    StateRegion, build_explicit_goal_distances, sorted_value_sets_overlap,
};
use super::abstraction_collections::goal_focused::{
    BuiltReport, CollectionMembership, ConcreteState, FULL_TASK_PROGRESSIVE_ROOTS_ERROR,
    GoalFocusedBackend, GoalFocusedSchedule, GoalVariant, MemberBudget, MemberDirection,
    RefinementRootDeadEnd, build_goal_focused_collection,
};
use super::abstraction_collections::portfolio::{CollectionStrategy, mix_seed, stable_text_seed};
use super::abstraction_task::{AbstractionUse, validate_abstraction_operator};
use super::cegar::regression::{ensure_conditions_are_regressable, numeric_effect_preimage};
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
fn fact_is_held(fact: &ExplicitFact, packer: &StatePacker, buffer: &[u64]) -> bool {
    fact.is_held(
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
    /// Set when the abstraction is a member of a goal-focused collection.
    pub collection_member: Option<CollectionMembership>,
    pub refinement_direction: CartesianRefinementDirection,
    pub split_selection_rank: Option<usize>,
    pub concrete_plan_operator_ids: Option<Vec<OperatorIndex>>,
    /// Number of non-loop transitions built before optional standalone compaction.
    pub transition_count: usize,
}

/// Rejects the flaw kinds Cartesian CEGAR does not implement.
pub fn validate_cartesian_flaw_kind(flaw_kind: FlawKind) -> Result<(), String> {
    match flaw_kind {
        FlawKind::Progression | FlawKind::Regression | FlawKind::ExecuteEntirePlan => Ok(()),
        FlawKind::SequenceProgression
        | FlawKind::SequenceRegression
        | FlawKind::SequenceBidirectional
        | FlawKind::TargetCentered => Err(format!(
            "Cartesian abstractions do not support flaw_kind={flaw_kind}; expected progression, regression or execute_entire_plan"
        )),
    }
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
    pub compute_operator_regions: bool,
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
            compute_operator_regions: true,
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
    /// Complementary collections only: alternate progression and regression
    /// by variant parity. When false, every variant refines in the configured
    /// `refinement_direction`.
    pub alternate_split_kinds: bool,
    /// Complementary collections only: build every member on its goal's
    /// single-goal task. When false, every member abstracts the full task and
    /// goals only label the `G * V` members, which are scheduled round-robin
    /// and carry no collection membership. Excludes `progressive_goal_roots`.
    pub single_goal_tasks: bool,
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
            alternate_split_kinds: true,
            single_goal_tasks: true,
        }
    }
}

#[derive(Debug, Clone)]
enum RefinementNode {
    Leaf {
        state_id: usize,
    },
    Propositional {
        var_id: VariableIndex,
        wanted: Vec<ExplicitValueIndex>,
        wanted_child: usize,
        other_child: usize,
    },
    Numeric {
        var_id: VariableIndex,
        boundary: NumericValue,
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

    pub fn map_state(
        &self,
        propositional: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize> {
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
                    let value = *propositional.get(var_id.index()).with_context(|| {
                        format!("propositional state has no value for var {var_id}")
                    })? as ExplicitValueIndex;
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
                        .get(var_id.index())
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
        var_id: VariableIndex,
        mut wanted: Vec<ExplicitValueIndex>,
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

/// Where a numeric refinement cuts one variable's range, and on which side of
/// the cut the state being split ends up.
#[derive(Debug, Clone, Copy)]
struct NumericSplit {
    var_id: VariableIndex,
    boundary: NumericValue,
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
    pub relevant_operator_ids: Vec<OperatorIndex>,
    pub abstract_operator_regions: Vec<AbstractOperatorRegions>,
    pub metadata: CartesianAbstractionMetadata,
}

impl CartesianAbstraction {
    pub fn num_states(&self) -> usize {
        self.distance_table.distances.len()
    }

    pub fn abstract_state_id(
        &self,
        propositional: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize> {
        self.hierarchy.map_state(propositional, numeric)
    }

    pub fn discard_transition_data(&mut self) {
        self.transition_system.transitions = Vec::new();
        self.transition_system.backward = Vec::new();
        self.transition_system.forward = Vec::new();
        self.transition_system.state_regions = Vec::new();
        self.abstract_operator_regions = Vec::new();
    }
}

#[derive(Debug, Clone)]
struct WorkingTransition {
    source: usize,
    target: usize,
    concrete_op_id: OperatorIndex,
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

    fn insert(&mut self, operator_id: OperatorIndex) -> bool {
        debug_assert!(
            operator_id.index() < self.operator_count,
            "operator {operator_id} exceeds Cartesian operator-set size {}",
            self.operator_count
        );
        let word = &mut self.words[operator_id.index() / u64::BITS as usize];
        let mask = 1_u64 << (operator_id.index() % u64::BITS as usize);
        if *word & mask != 0 {
            return false;
        }
        *word |= mask;
        true
    }

    fn contains(&self, operator_id: OperatorIndex) -> bool {
        debug_assert!(
            operator_id.index() < self.operator_count,
            "operator {operator_id} exceeds Cartesian operator-set size {}",
            self.operator_count
        );
        self.words[operator_id.index() / u64::BITS as usize]
            & (1_u64 << (operator_id.index() % u64::BITS as usize))
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
    type Item = OperatorIndex;

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
                return Some(OperatorIndex::from_usize(operator_id));
            }
            self.word_id += 1;
            self.remaining = *self.left.get(self.word_id)? & self.right[self.word_id];
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TransitionKey {
    source: usize,
    concrete_op_id: OperatorIndex,
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
    icaps_self_loop_order: Option<Vec<Vec<OperatorIndex>>>,
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
        let propositional_refinement_counts = vec![0; initial_region.propositions().len()];
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

    fn add_transition(&mut self, source: usize, op_id: OperatorIndex, target: usize) {
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
        var_id: VariableIndex,
        wanted: Vec<ExplicitValueIndex>,
        witness_value: ExplicitValueIndex,
        description: String,
    },
    Numeric {
        state_id: usize,
        var_id: VariableIndex,
        boundary: NumericValue,
        lower_includes_boundary: bool,
        witness_value: NumericValue,
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
        var_id: VariableIndex,
        wanted: Vec<ExplicitValueIndex>,
        witness_value: ExplicitValueIndex,
    },
    Numeric {
        state_id: usize,
        var_id: VariableIndex,
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
    Propositional(VariableIndex),
    Numeric(VariableIndex),
}

struct CartesianSemantics<'task> {
    task: &'task dyn AbstractNumericTask,
    propositional_axioms_by_prop_var: Vec<Vec<usize>>,
    operator_costs: Vec<NumericValue>,
    prop_split_dependent_operators: Vec<OperatorBitSet>,
    numeric_split_dependent_operators: Vec<OperatorBitSet>,
    additive_numeric_views: Vec<Option<AdditiveNumericView>>,
    additive_effect_deltas: Vec<Vec<NumericValue>>,
    numeric_integer_lattice: Vec<bool>,
    random_seed: Option<u64>,
    random_split_rng: Option<RefCell<SmallRng>>,
    icaps_random: Option<RefCell<ArtifactMt19937>>,
    refinement_direction: CartesianRefinementDirection,
    flaw_candidate_generation: CartesianFlawCandidateGeneration,
    split_selection_rank: Option<usize>,
    split_selection: CartesianSplitSelection,
    target_split_boundaries: Vec<NumericValue>,
}

fn mark_fact_split_dependencies(
    task: &dyn AbstractNumericTask,
    fact: &ExplicitFact,
    propositional_axioms_by_prop_var: &[Vec<usize>],
    visiting: &mut [bool],
    prop_dependencies: &mut [bool],
    numeric_dependencies: &mut [bool],
) -> Result<()> {
    let var_id = fact.var_index();
    if let Some(condition) = task.numeric_conditions().for_var(var_id) {
        let mut dimensions = condition.regular_numeric_var_dependencies().to_vec();
        dimensions.extend(comparison_refinement_dimensions(task, condition));
        dimensions.sort_unstable();
        dimensions.dedup();
        for numeric_var_id in dimensions {
            numeric_dependencies[numeric_var_id.index()] = true;
        }
        return Ok(());
    }
    if propositional_axioms_by_prop_var[var_id.index()].is_empty() {
        prop_dependencies[var_id.index()] = true;
        return Ok(());
    }
    ensure!(
        !visiting[var_id.index()],
        "cyclic propositional axiom dependency through variable {var_id}"
    );
    visiting[var_id.index()] = true;
    for &axiom_id in &propositional_axioms_by_prop_var[var_id.index()] {
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
    visiting[var_id.index()] = false;
    Ok(())
}

impl<'task> CartesianSemantics<'task> {
    fn task(&self) -> &dyn AbstractNumericTask {
        self.task
    }

    fn propositional_axioms_by_prop_var(&self) -> &[Vec<usize>] {
        &self.propositional_axioms_by_prop_var
    }

    fn operator_costs(&self) -> &[NumericValue] {
        &self.operator_costs
    }

    fn additive_numeric_views(&self) -> &[Option<AdditiveNumericView>] {
        &self.additive_numeric_views
    }

    fn additive_effect_deltas(&self) -> &[Vec<NumericValue>] {
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

    fn target_split_boundaries(&self) -> &[NumericValue] {
        &self.target_split_boundaries
    }

    fn new(
        task: &'task dyn AbstractNumericTask,
        config: &CartesianAbstractionConfig,
    ) -> Result<Self> {
        validate_abstractable_goal(task).map_err(anyhow::Error::msg)?;
        if config.flaw_kind == FlawKind::Regression {
            ensure_conditions_are_regressable(task)?;
        }
        for (op_id, op) in task.get_operators().iter().enumerate() {
            validate_abstraction_operator(task, op, OperatorIndex::from_usize(op_id))?;
        }

        let mut propositional_axioms_by_prop_var = vec![Vec::new(); task.get_num_variables()];
        for (axiom_id, axiom) in task.axioms().iter().enumerate() {
            let var_id = axiom.var_id();
            ensure!(
                var_id.index() < propositional_axioms_by_prop_var.len(),
                "propositional axiom {axiom_id} affects missing prop var {var_id}"
            );
            propositional_axioms_by_prop_var[var_id.index()].push(axiom_id);
        }
        let operator_costs = task
            .get_operators()
            .iter()
            .map(|op| metric_operator_cost_from_initial_values(task, op))
            .collect();
        let operator_count = task.get_operators().len();
        let additive_numeric_views = (0..task.numeric_variables().len())
            .map(|numeric_var_id| {
                analyze_additive_numeric_view(task, VariableIndex::from_usize(numeric_var_id))
            })
            .collect::<Vec<_>>();
        let mut additive_effect_deltas = Vec::with_capacity(task.numeric_variables().len());
        for (numeric_var_id, variable) in task.numeric_variables().iter().enumerate() {
            let mut deltas = match variable.get_type() {
                NumericType::Regular => task
                    .get_operators()
                    .iter()
                    .filter_map(|operator| {
                        numeric_dimension_delta_for_operator(
                            task,
                            VariableIndex::from_usize(numeric_var_id),
                            operator,
                        )
                    })
                    .collect::<Vec<_>>(),
                NumericType::Derived => additive_numeric_views[numeric_var_id]
                    .as_ref()
                    .map(|view| view.operator_deltas().to_vec())
                    .unwrap_or_default(),
                NumericType::Constant | NumericType::Cost => Vec::new(),
            };
            deltas.retain(|delta| delta.value().abs() > float_tolerance::SEARCH_EPSILON);
            deltas.sort_by(NumericValue::total_cmp);
            deltas.dedup_by(|left, right| approximately_equal(left.value(), right.value()));
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
        target_split_boundaries.sort_by(NumericValue::total_cmp);
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
                    && propositional_axioms_by_prop_var[var_id.index()].is_empty()
                {
                    prop_dependencies[var_id.index()] = true;
                }
            }
            for effect in op.assignment_effects() {
                let var_id = effect.affected_var_id();
                if task.numeric_variables()[var_id.index()].get_type() == &NumericType::Regular {
                    numeric_dependencies[var_id.index()] = true;
                }
            }
            for (numeric_var_id, view) in additive_numeric_views.iter().enumerate() {
                if view.as_ref().is_some_and(|view| {
                    view.operator_delta(OperatorIndex::from_usize(op_id))
                        .is_ok_and(|delta| delta.value().abs() > float_tolerance::SEARCH_EPSILON)
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
                    prop_split_dependent_operators[var_id].insert(OperatorIndex::from_usize(op_id));
                }
            }
            for (var_id, depends) in numeric_dependencies.into_iter().enumerate() {
                if depends {
                    numeric_split_dependent_operators[var_id]
                        .insert(OperatorIndex::from_usize(op_id));
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
            .map(|&value| approximately_equal(value.value(), value.value().round()))
            .collect::<Vec<_>>();
        for op in task.get_operators() {
            for effect in op.assignment_effects() {
                let rhs = initial_numeric[effect.var_id().index()];
                let preserves_integers = match effect.operation() {
                    AssignmentOperation::Plus
                    | AssignmentOperation::Minus
                    | AssignmentOperation::Assign
                    | AssignmentOperation::Times => {
                        approximately_equal(rhs.value(), rhs.value().round())
                    }
                    AssignmentOperation::Divide => approximately_equal(rhs.value().abs(), 1.0),
                };
                numeric_integer_lattice[effect.affected_var_id().index()] &= preserves_integers;
            }
        }
        for (numeric_var_id, view) in additive_numeric_views.iter().enumerate() {
            let Some(view) = view else {
                continue;
            };
            let initial_value = view.evaluate(initial_numeric);
            numeric_integer_lattice[numeric_var_id] =
                approximately_equal(initial_value.value(), initial_value.value().round())
                    && (0..operator_count).all(|op_id| {
                        view.operator_delta(OperatorIndex::from_usize(op_id))
                            .is_ok_and(|delta| {
                                approximately_equal(delta.value(), delta.value().round())
                            })
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

    fn operator_depends_on_split(&self, op_id: OperatorIndex, dimension: SplitDimension) -> bool {
        self.split_dependent_operators(dimension).contains(op_id)
    }

    fn split_dependent_operators(&self, dimension: SplitDimension) -> &OperatorBitSet {
        match dimension {
            SplitDimension::Propositional(var_id) => {
                &self.prop_split_dependent_operators[var_id.index()]
            }
            SplitDimension::Numeric(var_id) => {
                &self.numeric_split_dependent_operators[var_id.index()]
            }
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
                &source.propositions()[var_id.index()],
                &target.propositions()[var_id.index()],
            ),
            SplitDimension::Numeric(var_id) => {
                source.numeric[var_id.index()].intersects(&target.numeric[var_id.index()])
            }
        }
    }

    fn may_transition_after_independent_split(
        &self,
        source: &StateRegion,
        op_id: OperatorIndex,
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
                    .get_variable_domain_size(VariableIndex::from_usize(var_id))
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                ensure!(size > 0, "propositional var {var_id} has an empty domain");
                ensure!(
                    u32::try_from(size).is_ok(),
                    "propositional var {var_id} domain is too large: {size}"
                );
                Ok((0..size as u32).map(ExplicitValueIndex::new).collect())
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
        // Every dimension carries its whole domain, so nothing is constrained
        // yet; refinement splits add constraints one dimension at a time.
        Ok(StateRegion::with_constrained_props(
            propositions,
            numeric,
            Arc::from([]),
        ))
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
        let var_id = fact.var_index();
        if let Some(axiom_id) = self.task.numeric_conditions().id_for_var(var_id) {
            let (may_true, may_false) = self.comparison_truths(region, axiom_id)?;
            return Ok(match fact.value() {
                0 => may_true,
                1 => may_false,
                2 => may_true || may_false,
                value => bail!("invalid comparison proposition value {value} for var {var_id}"),
            });
        }
        if !self.propositional_axioms_by_prop_var[var_id.index()].is_empty() {
            ensure!(
                !visiting[var_id.index()],
                "cyclic propositional axiom support for variable {var_id}"
            );
            visiting[var_id.index()] = true;
            let result = (|| {
                let default_value = self.propositional_axiom_default(var_id.index())?;
                if fact.value_index() == default_value {
                    for &axiom_id in &self.propositional_axioms_by_prop_var[var_id.index()] {
                        let axiom = &self.task.axioms()[axiom_id];
                        if self.all_conditions_guaranteed(region, axiom.conditions(), visiting)? {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }

                for &axiom_id in &self.propositional_axioms_by_prop_var[var_id.index()] {
                    let axiom = &self.task.axioms()[axiom_id];
                    if axiom.effect_value() == fact.value_index()
                        && self.all_conditions_admitted(region, axiom.conditions(), visiting)?
                    {
                        return Ok(true);
                    }
                }
                Ok(false)
            })();
            visiting[var_id.index()] = false;
            return result;
        }
        Ok(region
            .propositions()
            .get(var_id.index())
            .is_some_and(|values| values.binary_search(&(fact.value_index())).is_ok()))
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
        let var_id = fact.var_index();
        if let Some(axiom_id) = self.task.numeric_conditions().id_for_var(var_id) {
            let (may_true, may_false) = self.comparison_truths(region, axiom_id)?;
            return Ok(match fact.value() {
                0 => may_true && !may_false,
                1 => may_false && !may_true,
                2 => false,
                value => bail!("invalid comparison proposition value {value} for var {var_id}"),
            });
        }
        if !self.propositional_axioms_by_prop_var[var_id.index()].is_empty() {
            ensure!(
                !visiting[var_id.index()],
                "cyclic propositional axiom support for variable {var_id}"
            );
            visiting[var_id.index()] = true;
            let result = (|| {
                let default_value = self.propositional_axiom_default(var_id.index())?;
                if fact.value_index() == default_value {
                    for &axiom_id in &self.propositional_axioms_by_prop_var[var_id.index()] {
                        let axiom = &self.task.axioms()[axiom_id];
                        if self.all_conditions_admitted(region, axiom.conditions(), visiting)? {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }

                for &axiom_id in &self.propositional_axioms_by_prop_var[var_id.index()] {
                    let axiom = &self.task.axioms()[axiom_id];
                    if axiom.effect_value() == fact.value_index()
                        && self.all_conditions_guaranteed(region, axiom.conditions(), visiting)?
                    {
                        return Ok(true);
                    }
                }
                Ok(false)
            })();
            visiting[var_id.index()] = false;
            return result;
        }
        let Some(values) = region.propositions().get(var_id.index()) else {
            return Ok(false);
        };
        Ok(values.len() == 1 && values[0] == fact.value_index())
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

    fn propositional_axiom_default(&self, var_id: usize) -> Result<ExplicitValueIndex> {
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

    fn operator_may_apply(&self, source: &StateRegion, op_id: OperatorIndex) -> Result<bool> {
        let op = self
            .task
            .get_operators()
            .get(op_id.index())
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
        op_id: OperatorIndex,
        target: &StateRegion,
        var_id: VariableIndex,
    ) -> bool {
        debug_assert!(
            !self.task.numeric_conditions().is_condition_var(var_id)
                && self.propositional_axioms_by_prop_var[var_id.index()].is_empty(),
            "derived proposition {var_id} has no explicit transition relation"
        );
        let op = &self.task.get_operators()[op_id.index()];
        if let Some(effect) = op.effects().iter().find(|effect| effect.var_id() == var_id) {
            debug_assert!(
                effect.conditions().is_empty(),
                "validated Cartesian operator {op_id} has a conditional effect"
            );
            return target.propositions()[var_id.index()]
                .binary_search(&(effect.value() as ExplicitValueIndex))
                .is_ok();
        }
        if matches!(self.split_selection, CartesianSplitSelection::Icaps26(_))
            && let Some(precondition) = op
                .preconditions()
                .iter()
                .find(|precondition| precondition.var_index() == var_id)
        {
            return target.propositions()[var_id.index()]
                .binary_search(&(precondition.value_index()))
                .is_ok();
        }
        sorted_value_sets_overlap(
            &source.propositions()[var_id.index()],
            &target.propositions()[var_id.index()],
        )
    }

    fn split_dimension_may_transition(
        &self,
        source: &StateRegion,
        op_id: OperatorIndex,
        target: &StateRegion,
        dimension: SplitDimension,
    ) -> Result<bool> {
        Ok(match dimension {
            SplitDimension::Propositional(var_id) => {
                self.propositional_dimension_may_transition(source, op_id, target, var_id)
            }
            SplitDimension::Numeric(var_id) => self.numeric_dimension_may_transition(
                source.numeric[var_id.index()],
                op_id,
                target.numeric[var_id.index()],
                var_id,
            )?,
        })
    }

    fn numeric_dimension_may_transition(
        &self,
        source: Interval,
        op_id: OperatorIndex,
        target: Interval,
        var_id: VariableIndex,
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

    fn icaps_numeric_precondition(
        &self,
        op_id: OperatorIndex,
        var_id: VariableIndex,
    ) -> Result<Interval> {
        let mut interval = Interval::unbounded();
        for fact in self.task.get_operators()[op_id.index()].preconditions() {
            let Some(tree_id) = self.task.numeric_conditions().id_for_var(fact.var_index()) else {
                continue;
            };
            let (condition_var_id, condition) =
                desired_comparison_interval(self, tree_id, fact.value_index())?;
            if condition_var_id == var_id {
                interval = interval.intersection(&condition);
            }
        }
        Ok(interval)
    }

    fn parent_loop_source_to_split_children(
        &self,
        source: &StateRegion,
        op_id: OperatorIndex,
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
        op_id: OperatorIndex,
        target: &StateRegion,
    ) -> Result<bool> {
        if !self.operator_may_apply(source, op_id)? {
            return Ok(false);
        }
        for var_id in 0..self.task.get_num_variables() {
            if self
                .task
                .numeric_conditions()
                .is_condition_var(VariableIndex::from_usize(var_id))
                || !self.propositional_axioms_by_prop_var[var_id].is_empty()
            {
                continue;
            }
            if !self.propositional_dimension_may_transition(
                source,
                op_id,
                target,
                VariableIndex::from_usize(var_id),
            ) {
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
                        VariableIndex::from_usize(numeric_var_id),
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
                            VariableIndex::from_usize(numeric_var_id),
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
        op_id: OperatorIndex,
        numeric_var_id: VariableIndex,
    ) -> Result<Option<Interval>> {
        numeric_effect_preimage(
            self.task,
            &self.additive_numeric_views,
            target,
            op_id,
            numeric_var_id,
        )
    }

    fn operator_region_source_for_transition(
        &self,
        source: &StateRegion,
        op_id: OperatorIndex,
        target: &StateRegion,
    ) -> Result<Option<StateRegion>> {
        debug_assert_eq!(
            source.numeric.len(),
            target.numeric.len(),
            "Cartesian transition source/target numeric dimension mismatch"
        );
        let mut operator_region = source.clone();
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
                        VariableIndex::from_usize(numeric_var_id),
                    )?
                    else {
                        return Ok(None);
                    };
                    let regressed = source.numeric[numeric_var_id].intersection(&preimage);
                    if regressed.is_empty() {
                        return Ok(None);
                    }
                    if regressed != source.numeric[numeric_var_id] {
                        Arc::make_mut(&mut operator_region.numeric)[numeric_var_id] = regressed;
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
                            VariableIndex::from_usize(numeric_var_id),
                        )?
                        .expect("additive-view preimage is always defined");
                    let regressed = source.numeric[numeric_var_id].intersection(&preimage);
                    if regressed.is_empty() {
                        return Ok(None);
                    }
                    if regressed != source.numeric[numeric_var_id] {
                        Arc::make_mut(&mut operator_region.numeric)[numeric_var_id] = regressed;
                    }
                }
                NumericType::Cost => {}
            }
        }
        Ok(Some(operator_region))
    }

    fn region_is_goal(&self, region: &StateRegion) -> Result<bool> {
        for goal_id in 0..self.task.get_num_goals() {
            if !self.region_admits_fact(region, self.task.get_goal_fact(goal_id))? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn concrete_prop_values(
        &self,
        packer: &StatePacker,
        packed: &[u64],
        out: &mut Vec<ExplicitValueIndex>,
    ) {
        out.clear();
        out.extend(
            (0..self.task.get_num_variables())
                .map(|var_id| ExplicitValueIndex::from_usize(packer.get(packed, var_id) as usize)),
        );
    }
}

pub struct CartesianAbstractionGenerator {
    config: CartesianAbstractionConfig,
}

impl CartesianAbstractionGenerator {
    pub fn new(config: CartesianAbstractionConfig) -> Result<Self> {
        ensure!(config.max_states > 0, "Cartesian max_states must be > 0");
        validate_cartesian_flaw_kind(config.flaw_kind).map_err(anyhow::Error::msg)?;
        ensure!(
            config.flaw_kind != FlawKind::Regression
                || !matches!(config.split_selection, CartesianSplitSelection::Icaps26(_)),
            "Cartesian regression flaws are not defined for the ICAPS 2026 artifact transition semantics"
        );
        Ok(Self { config })
    }

    pub fn generate(&self, task: &dyn AbstractNumericTask) -> Result<CartesianAbstraction> {
        self.generate_from_root(task, None)
    }

    fn generate_from_root(
        &self,
        task: &dyn AbstractNumericTask,
        refinement_root: Option<&ConcreteState>,
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
            if semantics.may_transition(
                &working.states[0],
                OperatorIndex::from_usize(op_id),
                &working.states[0],
            )? {
                working.add_transition(0, OperatorIndex::from_usize(op_id), 0);
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
            None => ConcreteState {
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
                            return Err(RefinementRootDeadEnd::new(format!(
                                "Cartesian abstract state {initial_state}"
                            ))
                            .into());
                        }
                    }
                } else {
                    None
                };
                let check = match self.config.flaw_kind {
                    FlawKind::Progression => find_progression_flaw(
                        &working,
                        &semantics,
                        &shortest_paths,
                        &state_packer,
                        &axiom_evaluator,
                        &refinement_root,
                        selected_plan.as_deref(),
                    )?,
                    FlawKind::Regression => find_regression_flaw(
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
                        Err(RefinementRootDeadEnd::new(format!(
                            "Cartesian abstract state {abstract_state_id}"
                        ))
                        .into())
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
        let (transition_system, distance_table, relevant_operator_ids, operator_regions) =
            if self.config.retain_transition_system {
                finalize_abstraction(
                    &working,
                    &semantics,
                    initial_state_hash,
                    self.config.combine_labels,
                    self.config.compute_operator_regions,
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
            abstract_operator_regions: operator_regions,
            metadata: CartesianAbstractionMetadata {
                solved_by_self: solved_plan.is_some(),
                abstraction_use: AbstractionUse::Standalone,
                stop_reason,
                pending_flaw,
                refinements,
                collection_member: None,
                refinement_direction: self.config.refinement_direction,
                split_selection_rank: self.config.split_selection_rank,
                concrete_plan_operator_ids: solved_plan.map(ConcretePlan::into_operator_ids),
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
        ensure!(
            config.alternate_split_kinds || config.collection_strategy.is_complementary(),
            "`alternate_split_kinds=false` requires `collection_strategy=complementary`"
        );
        ensure!(
            config.single_goal_tasks || config.collection_strategy.is_complementary(),
            "`single_goal_tasks=false` requires `collection_strategy=complementary`"
        );
        ensure!(
            config.single_goal_tasks || !config.progressive_goal_roots,
            "{FULL_TASK_PROGRESSIVE_ROOTS_ERROR}"
        );
        CartesianAbstractionGenerator::new(config.abstraction.clone())?;
        Ok(Self { config })
    }

    /// Builds the goal-focused collection of the task; see
    /// [`goal_focused`](super::abstraction_collections::goal_focused) for the
    /// schedule, the progressive goal roots, and the stop rules.
    ///
    /// Each member changes only the goal view. Operators, state mappings, and
    /// concrete operator IDs stay identical to the base task. Changing the
    /// refinement root only chooses counterexamples; every hierarchy still
    /// partitions the full task state space, so the resulting transition
    /// systems remain admissible components for canonical and cost-partitioned
    /// collection heuristics.
    pub fn generate(&self, task: &dyn AbstractNumericTask) -> Result<Vec<CartesianAbstraction>> {
        build_goal_focused_collection(
            task,
            &GoalFocusedSchedule {
                strategy: self.config.collection_strategy,
                variants_per_goal: self.config.variants_per_goal,
                random_seed: self.config.abstraction.random_seed,
                max_collection_states: self.config.max_collection_states,
                total_max_time: self.config.total_max_time,
                progressive_goal_roots: self.config.progressive_goal_roots,
                alternate_split_kinds: self.config.alternate_split_kinds,
                single_goal_tasks: self.config.single_goal_tasks,
            },
            self,
        )
    }

    /// Configuration for one member: the collection's per-abstraction
    /// configuration, capped by what is left of the collection's state and
    /// time budgets, then specialised for this variant.
    fn member_abstraction_config(
        &self,
        variant: &GoalVariant,
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
        if let Some(direction) = variant.direction {
            config.refinement_direction = match direction {
                MemberDirection::Progression => CartesianRefinementDirection::Progression,
                MemberDirection::Regression => CartesianRefinementDirection::Regression,
            };
        }
        if let Some(rank) = variant.tie_rank {
            config.split_selection_rank = Some(rank);
        }
        config.random_seed = variant.seed;
        config
    }
}

impl GoalFocusedBackend for CartesianAbstractionCollectionGenerator {
    type Member = CartesianAbstraction;

    const NAME: &'static str = "Cartesian";

    fn build(
        &self,
        task: &dyn AbstractNumericTask,
        root: Option<&ConcreteState>,
        variant: &GoalVariant,
        budget: MemberBudget,
    ) -> Result<(CartesianAbstraction, BuiltReport)> {
        let abstraction =
            CartesianAbstractionGenerator::new(self.member_abstraction_config(variant, budget))?
                .generate_from_root(task, root)?;
        let report = BuiltReport {
            initial_h: abstraction.distance_table.distances
                [abstraction.distance_table.initial_state_hash],
            num_states: abstraction.num_states(),
            concrete_plan: abstraction.metadata.concrete_plan_operator_ids.clone(),
        };
        Ok((abstraction, report))
    }

    fn record_membership(
        &self,
        abstraction: &mut CartesianAbstraction,
        membership: Option<CollectionMembership>,
    ) {
        abstraction.metadata.collection_member = membership;
        abstraction.metadata.abstraction_use = AbstractionUse::CollectionMember;
    }
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
    op_id: OperatorIndex,
    var_id: VariableIndex,
    old_state_id: usize,
    new_state_id: usize,
) {
    let op = &semantics.task.get_operators()[op_id.index()];
    let pre: Option<ExplicitValueIndex> = op
        .preconditions()
        .iter()
        .find(|fact| fact.var_index() == var_id)
        .map(ExplicitFact::value_index);
    let effect = op
        .effects()
        .iter()
        .find(|effect| effect.var_id() == var_id)
        .map(|effect| effect.value());
    let post = effect.or(pre);
    let old_contains = |value: usize| {
        working.states[old_state_id].propositions()[var_id.index()]
            .binary_search(&ExplicitValueIndex::from_usize(value))
            .is_ok()
    };

    match (pre, post) {
        (None, None) => {
            working.add_transition(old_state_id, op_id, old_state_id);
            working.add_transition(new_state_id, op_id, new_state_id);
        }
        (None, Some(post)) if !old_contains(post.index()) => {
            working.add_transition(old_state_id, op_id, new_state_id);
            working.add_transition(new_state_id, op_id, new_state_id);
        }
        (None, Some(post)) => {
            assert!(old_contains(post.index()));
            working.add_transition(old_state_id, op_id, old_state_id);
            working.add_transition(new_state_id, op_id, old_state_id);
        }
        (Some(pre), Some(post)) if old_contains(pre.index()) => {
            if old_contains(post.index()) {
                working.add_transition(old_state_id, op_id, old_state_id);
            } else {
                working.add_transition(old_state_id, op_id, new_state_id);
            }
        }
        (Some(_), Some(post)) => {
            if old_contains(post.index()) {
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
    old_loop_order: Vec<OperatorIndex>,
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
                    .propositions()
                    .get(var_id.index())
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
                wanted_region.narrow_prop(var_id, wanted_child_values);
                let mut other_region = old_region.clone();
                other_region.narrow_prop(var_id, other_child_values);
                working.propositional_refinement_counts[var_id.index()] += 1;
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
                let parent = old_region.numeric[var_id.index()];
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
                Arc::make_mut(&mut lower_region.numeric)[var_id.index()] = lower;
                let mut upper_region = old_region.clone();
                Arc::make_mut(&mut upper_region.numeric)[var_id.index()] = upper;
                working.numeric_refinement_counts[var_id.index()] += 1;
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
    prop_scratch: std::cell::RefCell<Vec<ExplicitValueIndex>>,
    numeric_scratch: std::cell::RefCell<Vec<NumericValue>>,
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
