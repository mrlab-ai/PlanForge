//! Numeric Cartesian abstractions refined by concrete counterexamples.
//!
//! Unlike the factorized domain abstraction, splitting one Cartesian state
//! adds exactly one state. Every abstract transition is a may-transition of a
//! grounded concrete operator. CEGAR replays a deterministic optimal abstract
//! trace and refines its first witnessed flaw; only a successfully replayed
//! concrete plan may set `solved_by_self`.

#[cfg(test)]
mod tests;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use ordered_float::NotNan;
use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, ExplicitFact, NumericType,
    metric_operator_cost_from_initial_values,
};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::int_packer::IntDoublePacker;
use tracing::{debug, info};

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;

use super::abstraction_collections::cost_partitioning::{
    AbstractOperatorFootprint, AbstractTransition, AbstractTransitionSystem,
    ConcreteOperatorFootprint, PropValueId, StateRegion,
};
use super::abstraction_collections::portfolio::{
    CollectionStrategy, derive_variant_seed, mix_seed, stable_text_seed,
};
use super::abstraction_task::{AbstractionUse, SingleGoalTask, validate_abstraction_operator};
use super::domain_abstractions::cegar::flaw_search::state::progress;
use super::domain_abstractions::comparison_expression::{ComparisonTree, Interval};
use super::domain_abstractions::domain_abstraction_factory::AbstractDistanceTable;
use super::domain_abstractions::utils::{fact_is_hold, get_initial_state, make_prop_state_packer};

const EPSILON: f64 = 1e-9;

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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartesianRefinementDirection {
    Progression,
    Regression,
}

#[derive(Debug, Clone)]
pub struct CartesianAbstractionConfig {
    pub max_states: usize,
    pub max_time: Option<Duration>,
    pub combine_labels: bool,
    pub compute_operator_footprints: bool,
    pub random_seed: Option<u64>,
    pub refinement_direction: CartesianRefinementDirection,
    pub split_selection_rank: Option<usize>,
    pub debug: bool,
}

impl Default for CartesianAbstractionConfig {
    fn default() -> Self {
        Self {
            max_states: 10_000,
            max_time: None,
            combine_labels: false,
            compute_operator_footprints: true,
            random_seed: None,
            refinement_direction: CartesianRefinementDirection::Progression,
            split_selection_rank: None,
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
        var_id: usize,
        boundary: f64,
        lower_includes_boundary: bool,
        old_state_is_lower: bool,
    ) -> Result<()> {
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
    transition_ids_by_key: HashMap<TransitionKey, usize>,
    propositional_refinement_counts: Vec<usize>,
    numeric_refinement_counts: Vec<usize>,
}

impl WorkingAbstraction {
    fn new(initial_region: StateRegion, operator_count: usize) -> Self {
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
            self_loop_operator_ids: vec![OperatorBitSet::empty(operator_count)],
            transition_ids_by_key: HashMap::new(),
            propositional_refinement_counts,
            numeric_refinement_counts,
        }
    }

    fn add_transition(&mut self, source: usize, op_id: usize, target: usize) {
        if source == target {
            self.self_loop_operator_ids[source].insert(op_id);
            return;
        }
        let key = TransitionKey {
            source,
            concrete_op_id: op_id,
            target,
        };
        if self.transition_ids_by_key.contains_key(&key) {
            return;
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
        let old = self.transition_ids_by_key.insert(key, transition_id);
        assert!(old.is_none(), "duplicate Cartesian transition key");
        self.outgoing[source].push(transition_id);
        self.incoming[target].push(transition_id);
    }

    fn remove_transition(&mut self, transition_id: usize) -> WorkingTransition {
        let transition = self.transitions[transition_id]
            .take()
            .expect("Cartesian adjacency references a removed transition");
        let removed_id = self.transition_ids_by_key.remove(&TransitionKey {
            source: transition.source,
            concrete_op_id: transition.concrete_op_id,
            target: transition.target,
        });
        assert_eq!(
            removed_id,
            Some(transition_id),
            "active Cartesian transition key is missing or inconsistent"
        );
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
            self.self_loop_operator_ids[key.source].contains(key.concrete_op_id)
        } else {
            self.transition_ids_by_key.contains_key(&key)
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

#[derive(Debug, Clone, Copy)]
enum SplitDimension {
    Propositional(usize),
    Numeric(usize),
}

struct CartesianSemantics<'task> {
    task: &'task dyn AbstractNumericTask,
    comparison_tree_by_prop_var: Vec<Option<usize>>,
    comparison_trees: Vec<ComparisonTree>,
    propositional_axioms_by_prop_var: Vec<Vec<usize>>,
    operator_costs: Vec<f64>,
    prop_split_dependent_operators: Vec<OperatorBitSet>,
    numeric_split_dependent_operators: Vec<OperatorBitSet>,
    random_seed: Option<u64>,
    refinement_direction: CartesianRefinementDirection,
    split_selection_rank: Option<usize>,
    target_split_boundaries: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn mark_fact_split_dependencies(
    task: &dyn AbstractNumericTask,
    fact: &ExplicitFact,
    comparison_tree_by_prop_var: &[Option<usize>],
    comparison_trees: &[ComparisonTree],
    propositional_axioms_by_prop_var: &[Vec<usize>],
    visiting: &mut [bool],
    prop_dependencies: &mut [bool],
    numeric_dependencies: &mut [bool],
) -> Result<()> {
    let var_id = fact.var();
    if let Some(tree_id) = comparison_tree_by_prop_var[var_id] {
        let tree = comparison_trees
            .get(tree_id)
            .with_context(|| format!("missing comparison tree {tree_id}"))?;
        for numeric_var_id in tree.regular_numeric_var_dependencies(task) {
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
                comparison_tree_by_prop_var,
                comparison_trees,
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
    fn new(
        task: &'task dyn AbstractNumericTask,
        config: &CartesianAbstractionConfig,
    ) -> Result<Self> {
        for (op_id, op) in task.get_operators().iter().enumerate() {
            validate_abstraction_operator(task, op, op_id)?;
        }

        let mut comparison_tree_by_prop_var = vec![None; task.get_num_variables()];
        let mut comparison_trees = Vec::with_capacity(task.comparison_axioms().len());
        for (axiom_id, axiom) in task.comparison_axioms().iter().enumerate() {
            let var_id = axiom.get_affected_var_id();
            ensure!(
                var_id < comparison_tree_by_prop_var.len(),
                "comparison axiom {axiom_id} affects missing prop var {var_id}"
            );
            let tree = ComparisonTree::from_task(task, axiom_id).map_err(|error| {
                anyhow::anyhow!("invalid comparison axiom {axiom_id}: {error:?}")
            })?;
            let tree_id = comparison_trees.len();
            comparison_trees.push(tree);
            ensure!(
                comparison_tree_by_prop_var[var_id]
                    .replace(tree_id)
                    .is_none(),
                "multiple comparison axioms affect prop var {var_id}"
            );
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
                    &comparison_tree_by_prop_var,
                    &comparison_trees,
                    &propositional_axioms_by_prop_var,
                    &mut visiting,
                    &mut prop_dependencies,
                    &mut numeric_dependencies,
                )?;
            }
            for effect in op.effects() {
                let var_id = effect.var_id();
                if comparison_tree_by_prop_var[var_id].is_none()
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
        Ok(Self {
            task,
            comparison_tree_by_prop_var,
            comparison_trees,
            propositional_axioms_by_prop_var,
            operator_costs,
            prop_split_dependent_operators,
            numeric_split_dependent_operators,
            random_seed: config.random_seed,
            refinement_direction: config.refinement_direction,
            split_selection_rank: config.split_selection_rank,
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
            SplitDimension::Propositional(var_id) => {
                sorted_values_overlap(&source.propositions[var_id], &target.propositions[var_id])
            }
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
        if let Some(axiom_id) = self
            .comparison_tree_by_prop_var
            .get(var_id)
            .copied()
            .flatten()
        {
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
        if let Some(axiom_id) = self
            .comparison_tree_by_prop_var
            .get(var_id)
            .copied()
            .flatten()
        {
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
            .comparison_trees
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
            self.comparison_tree_by_prop_var[var_id].is_none()
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
        sorted_values_overlap(&source.propositions[var_id], &target.propositions[var_id])
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
            SplitDimension::Numeric(var_id) => self
                .numeric_effect_preimage(target.numeric[var_id], op_id, var_id)?
                .is_some_and(|preimage| preimage.intersects(&source.numeric[var_id])),
        })
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
            if self.comparison_tree_by_prop_var[var_id].is_some()
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
                    let Some(preimage) = self.numeric_effect_preimage(
                        target.numeric[numeric_var_id],
                        op_id,
                        numeric_var_id,
                    )?
                    else {
                        return Ok(false);
                    };
                    if !preimage.intersects(&source.numeric[numeric_var_id]) {
                        return Ok(false);
                    }
                }
                NumericType::Derived | NumericType::Cost => {}
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
                    let regressed = interval_intersection(source.numeric[numeric_var_id], preimage);
                    if regressed.is_empty() {
                        return Ok(None);
                    }
                    if regressed != source.numeric[numeric_var_id] {
                        Arc::make_mut(&mut footprint.numeric)[numeric_var_id] = regressed;
                    }
                }
                // Derived values are functions of regular roots and cost
                // variables are not Cartesian split dimensions. Their source
                // restrictions are already represented through the regular
                // dimensions and operator preconditions.
                NumericType::Derived | NumericType::Cost => {}
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

    fn concrete_prop_values(&self, packer: &IntDoublePacker, packed: &[u64], out: &mut Vec<usize>) {
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
        let mut working = WorkingAbstraction::new(
            semantics.trivial_region()?,
            semantics.task.get_operators().len(),
        );
        for op_id in 0..task.get_operators().len() {
            if semantics.may_transition(&working.states[0], op_id, &working.states[0])? {
                working.add_transition(0, op_id, 0);
            }
        }
        let state_packer = Arc::new(make_prop_state_packer(task));
        let axiom_evaluator = AxiomEvaluator::new(Arc::new(task), state_packer.clone());
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
            None => {
                let (propositions, numeric) =
                    get_initial_state(task, &state_packer, &axiom_evaluator)?;
                CartesianConcreteState {
                    propositions,
                    numeric,
                }
            }
        };
        let mut refinements: usize = 0;

        let mut shortest_paths = compute_shortest_paths(&working, &semantics)?;
        let (stop_reason, pending_flaw, solved_plan) = loop {
            if refinements.is_multiple_of(64)
                && !crate::resource_limits::poll_and_release_if_exceeded()
            {
                break (CartesianStopReason::MemoryLimit, None, None);
            }
            let check = replay_optimal_abstract_trace(
                &working,
                &semantics,
                &shortest_paths,
                &state_packer,
                &axiom_evaluator,
                &refinement_root,
            )?;
            match check {
                PlanCheck::ConcretePlan(plan) => {
                    break (CartesianStopReason::ConcretePlan, None, Some(plan));
                }
                PlanCheck::AbstractDeadEnd(abstract_state_id) => {
                    return Err(RefinementRootDeadEnd { abstract_state_id }.into());
                }
                PlanCheck::Refine(split) => {
                    if working.states.len() >= self.config.max_states {
                        break (
                            CartesianStopReason::StateLimit,
                            Some(split.description().to_string()),
                            None,
                        );
                    }
                    if self
                        .config
                        .max_time
                        .is_some_and(|max_time| start.elapsed() >= max_time)
                    {
                        break (
                            CartesianStopReason::TimeLimit,
                            Some(split.description().to_string()),
                            None,
                        );
                    }
                    if self.config.debug {
                        debug!(
                            "Cartesian refinement {} at {} states: {}",
                            refinements,
                            working.states.len(),
                            split.description()
                        );
                    }
                    let old_state_id = split.state_id();
                    let new_state_id = apply_split(&mut working, &semantics, split)?;
                    shortest_paths = update_shortest_paths_after_split(
                        &working,
                        &semantics,
                        shortest_paths,
                        old_state_id,
                        new_state_id,
                    )?;
                    refinements += 1;
                }
            }
        };

        let (transition_system, distance_table, relevant_operator_ids, footprints) =
            finalize_abstraction(
                &working,
                &semantics,
                self.config.combine_labels,
                self.config.compute_operator_footprints,
            )?;
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
                (plan.cost - h).abs() <= 1e-7,
                "concrete Cartesian plan cost {} differs from abstract h(refinement root) {h}",
                plan.cost
            );
        }
        info!(
            "Cartesian abstraction: states={}, transitions={}, refinements={}, h(init)={}, stop={stop_reason:?}, elapsed={:.3}s",
            distance_table.distances.len(),
            transition_system.transitions.len(),
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
                concrete_plan_operator_ids: solved_plan.map(|plan| plan.operator_ids),
                progressive_refinement_root: false,
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
        let start = Instant::now();
        let mut remaining_states = self.config.max_collection_states;
        let mut abstractions = Vec::with_capacity(abstraction_count);
        let initial_refinement_root = (self.config.progressive_goal_roots && goal_count > 0)
            .then(|| initial_cartesian_concrete_state(task))
            .transpose()?;
        let mut refinement_roots = initial_refinement_root
            .as_ref()
            .map_or_else(Vec::new, |root| vec![root.clone(); variants_per_goal]);
        let mut satisfied_goals_by_root = refinement_roots
            .iter()
            .map(|root| count_satisfied_cartesian_goals(task, root))
            .collect::<Result<Vec<_>>>()?;
        let mut progressive_root_advanced = vec![false; refinement_roots.len()];
        let mut progressive_lane_complete = vec![false; refinement_roots.len()];
        let mut initial_root_goal_covered = vec![false; goal_count];
        let mut variants_built_by_goal = vec![0usize; goal_count];
        let mut best_initial_h_by_goal = vec![0.0f64; goal_count];
        let mut continuation_queue = VecDeque::<(usize, usize)>::new();
        let mut abstraction_id = 0usize;
        let mut initial_abstractions_built = 0usize;
        let mut stop_reason = "requested abstraction count reached";
        while initial_abstractions_built < abstraction_count
            || !continuation_queue.is_empty()
            || (self.config.progressive_goal_roots
                && initial_root_goal_covered.iter().any(|covered| !covered))
        {
            if remaining_states < 2 && !abstractions.is_empty() {
                stop_reason = "collection size limit";
                break;
            }
            let scheduled_member_pending = initial_abstractions_built < abstraction_count;
            let initial_root_specialist_goal = (!scheduled_member_pending
                && self.config.progressive_goal_roots)
                .then(|| {
                    initial_root_goal_covered
                        .iter()
                        .position(|covered| !covered)
                })
                .flatten();
            let continuation = if self.config.progressive_goal_roots
                && !scheduled_member_pending
                && initial_root_specialist_goal.is_none()
            {
                loop {
                    let Some((goal_id, variant_id)) = continuation_queue.pop_front() else {
                        break None;
                    };
                    if !cartesian_goal_is_satisfied(
                        task,
                        refinement_roots
                            .get(variant_id)
                            .expect("progressive continuation references missing root"),
                        goal_id,
                    )? {
                        break Some((goal_id, variant_id));
                    }
                }
            } else {
                None
            };
            let is_continuation = continuation.is_some();
            let mut is_initial_root_specialist = false;
            let (goal_id, variant_id) = if goal_count == 0 {
                (0, 0)
            } else if scheduled_member_pending {
                let Some(goal_id) = select_next_cartesian_collection_goal(
                    &variants_built_by_goal,
                    &best_initial_h_by_goal,
                    variants_per_goal,
                ) else {
                    stop_reason = "requested abstraction count reached";
                    break;
                };
                (goal_id, variants_built_by_goal[goal_id])
            } else if let Some(goal_id) = initial_root_specialist_goal {
                is_initial_root_specialist = true;
                (goal_id, variants_per_goal)
            } else if let Some(continuation) = continuation {
                continuation
            } else {
                stop_reason = "requested abstractions and initial-root goal coverage reached";
                break;
            };
            let remaining_time = match self.config.total_max_time {
                Some(total_max_time) => {
                    let elapsed = start.elapsed();
                    if elapsed >= total_max_time {
                        if abstractions.is_empty() {
                            Some(Duration::ZERO)
                        } else {
                            stop_reason = "collection time limit";
                            break;
                        }
                    } else {
                        Some(total_max_time - elapsed)
                    }
                }
                None => None,
            };
            let mut abstraction_config = self.config.abstraction.clone();
            abstraction_config.max_states = abstraction_config.max_states.min(remaining_states);
            abstraction_config.max_time = match (abstraction_config.max_time, remaining_time) {
                (Some(per_abstraction), Some(remaining)) => Some(per_abstraction.min(remaining)),
                (Some(per_abstraction), None) => Some(per_abstraction),
                (None, Some(remaining)) => Some(remaining),
                (None, None) => None,
            };
            let construction_variant_id = if is_initial_root_specialist {
                0
            } else {
                variant_id
            };
            if goal_count > 0 && self.config.collection_strategy.is_complementary() {
                abstraction_config.refinement_direction =
                    if construction_variant_id.is_multiple_of(2) {
                        CartesianRefinementDirection::Progression
                    } else {
                        CartesianRefinementDirection::Regression
                    };
                abstraction_config.split_selection_rank = Some(construction_variant_id / 2);
                abstraction_config.random_seed = if construction_variant_id == 0 {
                    None
                } else {
                    Some(derive_variant_seed(
                        abstraction_config.random_seed.unwrap_or(0),
                        goal_id,
                        construction_variant_id - 1,
                    ))
                };
            } else if goal_count > 0
                && self.config.variants_per_goal > 1
                && construction_variant_id > 0
            {
                abstraction_config.random_seed = Some(derive_variant_seed(
                    abstraction_config.random_seed.unwrap_or(0),
                    goal_id,
                    construction_variant_id - 1,
                ));
            }

            let goal_task =
                (goal_count > 0).then(|| SingleGoalTask::new(task, *task.get_goal_fact(goal_id)));
            let abstraction_task = goal_task
                .as_ref()
                .map_or(task, |goal_task| goal_task as &dyn AbstractNumericTask);
            debug!(
                "Cartesian collection: building abstraction {}, goal={}, variant={}, continuation={}, initial_root_specialist={}, direction={:?}, split_rank={:?}, max_states={}, seed={:?}",
                abstraction_id + 1,
                goal_id,
                variant_id,
                is_continuation,
                is_initial_root_specialist,
                abstraction_config.refinement_direction,
                abstraction_config.split_selection_rank,
                abstraction_config.max_states,
                abstraction_config.random_seed
            );
            let generator = CartesianAbstractionGenerator::new(abstraction_config)?;
            let lane_is_complete = progressive_lane_complete
                .get(variant_id)
                .copied()
                .unwrap_or(false);
            let refinement_root = (!is_initial_root_specialist && !lane_is_complete)
                .then(|| refinement_roots.get(variant_id))
                .flatten();
            let built_from_initial_root = is_initial_root_specialist
                || refinement_root.is_none()
                || !progressive_root_advanced
                    .get(variant_id)
                    .copied()
                    .unwrap_or(false);
            let mut reset_progressive_root = false;
            let mut abstraction = match generator
                .generate_from_root(abstraction_task, refinement_root)
            {
                Ok(abstraction) => abstraction,
                Err(error)
                    if refinement_root.is_some()
                        && error.downcast_ref::<RefinementRootDeadEnd>().is_some() =>
                {
                    reset_progressive_root = true;
                    info!(
                        "Cartesian collection: progressive root is an abstract dead end for goal {goal_id}, variant {variant_id}; rebuilding this member from the task initial state"
                    );
                    generator.generate_from_root(abstraction_task, None)
                        .with_context(|| {
                            format!("failed to rebuild Cartesian collection abstraction {abstraction_id} from the task initial state")
                        })?
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("failed to build Cartesian collection abstraction {abstraction_id}")
                    });
                }
            };
            let state_count = abstraction.num_states();
            ensure!(
                state_count <= remaining_states,
                "Cartesian goal abstraction used {state_count} states with only {remaining_states} remaining"
            );
            remaining_states -= state_count;
            abstraction.metadata.collection_goal_id = (goal_count > 0).then_some(goal_id);
            abstraction.metadata.collection_variant_id = (goal_count > 0).then_some(variant_id);
            abstraction.metadata.abstraction_use = AbstractionUse::CollectionMember;
            abstraction.metadata.progressive_refinement_root = !is_initial_root_specialist
                && !lane_is_complete
                && progressive_root_advanced
                    .get(variant_id)
                    .copied()
                    .unwrap_or(false)
                && !reset_progressive_root;
            if goal_count > 0 && (built_from_initial_root || reset_progressive_root) {
                initial_root_goal_covered[goal_id] = true;
            }
            if goal_count > 0 && !is_continuation && !is_initial_root_specialist {
                variants_built_by_goal[goal_id] += 1;
                initial_abstractions_built += 1;
                let initial_h = abstraction.distance_table.distances
                    [abstraction.distance_table.initial_state_hash];
                best_initial_h_by_goal[goal_id] = best_initial_h_by_goal[goal_id].max(initial_h);
            } else if goal_count == 0 {
                initial_abstractions_built += 1;
            }
            if !is_initial_root_specialist
                && !lane_is_complete
                && let Some(root) = refinement_roots.get_mut(variant_id)
            {
                if reset_progressive_root {
                    *root = initial_refinement_root
                        .as_ref()
                        .expect("progressive refinement root requires an initial root")
                        .clone();
                    progressive_root_advanced[variant_id] = false;
                    progressive_lane_complete[variant_id] = true;
                    satisfied_goals_by_root[variant_id] =
                        count_satisfied_cartesian_goals(task, root)?;
                    continuation_queue
                        .retain(|(_, queued_variant_id)| *queued_variant_id != variant_id);
                    debug!(
                        "Cartesian collection: dead root made progressive variant {variant_id} terminal after rebuilding goal {goal_id} from the initial state"
                    );
                } else {
                    match abstraction.metadata.concrete_plan_operator_ids.as_deref() {
                        Some(operator_ids) => {
                            let previous_satisfied_goals = satisfied_goals_by_root[variant_id];
                            *root = replay_cartesian_operator_sequence(task, root, operator_ids)?;
                            let satisfied_goals = count_satisfied_cartesian_goals(task, root)?;
                            satisfied_goals_by_root[variant_id] = satisfied_goals;
                            debug!(
                                "Cartesian collection: advanced progressive root for variant {variant_id} through {} concrete operators; satisfied_goals={}/{}",
                                operator_ids.len(),
                                satisfied_goals,
                                goal_count,
                            );
                            if satisfied_goals == goal_count {
                                *root = initial_refinement_root
                                    .as_ref()
                                    .expect("progressive refinement root requires an initial root")
                                    .clone();
                                progressive_root_advanced[variant_id] = false;
                                progressive_lane_complete[variant_id] = true;
                                satisfied_goals_by_root[variant_id] =
                                    count_satisfied_cartesian_goals(task, root)?;
                                continuation_queue.retain(|(_, queued_variant_id)| {
                                    *queued_variant_id != variant_id
                                });
                                debug!(
                                    "Cartesian collection: full goal reached for variant {variant_id}; made the progressive lane terminal"
                                );
                            } else {
                                progressive_root_advanced[variant_id] = true;
                                if satisfied_goals > previous_satisfied_goals {
                                    for (retry_goal_id, &variants_built) in
                                        variants_built_by_goal.iter().enumerate()
                                    {
                                        let was_already_attempted = variants_built > variant_id;
                                        if was_already_attempted
                                            && !cartesian_goal_is_satisfied(
                                                task,
                                                root,
                                                retry_goal_id,
                                            )?
                                            && !continuation_queue
                                                .contains(&(retry_goal_id, variant_id))
                                        {
                                            continuation_queue
                                                .push_back((retry_goal_id, variant_id));
                                        }
                                    }
                                }
                            }
                        }
                        None => {
                            debug!(
                                "Cartesian collection: goal {goal_id} variant {variant_id} produced no concrete plan; progressive root remains unchanged"
                            );
                        }
                    }
                }
            }
            abstractions.push(abstraction);
            abstraction_id += 1;
            if !crate::resource_limits::poll_and_release_if_exceeded() {
                stop_reason = "memory limit";
                break;
            }
        }

        if stop_reason == "requested abstraction count reached"
            && self.config.progressive_goal_roots
            && initial_root_goal_covered.iter().all(|covered| *covered)
        {
            stop_reason = "requested abstractions and initial-root goal coverage reached";
        }
        info!(
            "Cartesian collection: abstractions={}, states={}, elapsed={:.3}s, stop_reason={}",
            abstractions.len(),
            self.config.max_collection_states - remaining_states,
            start.elapsed().as_secs_f64(),
            stop_reason
        );
        if !satisfied_goals_by_root.is_empty() {
            info!(
                "Cartesian collection: progressive root goal coverage={satisfied_goals_by_root:?}/{goal_count}"
            );
        }
        if !initial_root_goal_covered.is_empty() {
            info!(
                "Cartesian collection: initial-root goal coverage={}/{}",
                initial_root_goal_covered
                    .iter()
                    .filter(|covered| **covered)
                    .count(),
                goal_count
            );
        }
        Ok(abstractions)
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
        progress(
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

#[derive(Debug)]
struct ShortestPaths {
    distances: Vec<f64>,
    generating_transition: Vec<Option<TransitionKey>>,
    dependents: Vec<Vec<usize>>,
    dependent_positions: Vec<Option<usize>>,
    is_goal: Vec<bool>,
    invalid: Vec<bool>,
}

impl ShortestPaths {
    fn remove_generating_transition(&mut self, source: usize) {
        let Some(old) = self.generating_transition[source].take() else {
            assert!(
                self.dependent_positions[source].is_none(),
                "Cartesian state without a generating transition has a dependency position"
            );
            return;
        };
        let position = self.dependent_positions[source]
            .take()
            .expect("Cartesian generating transition has no dependency position");
        let removed = self.dependents[old.target].swap_remove(position);
        assert_eq!(
            removed, source,
            "Cartesian dependency position references another state"
        );
        if position < self.dependents[old.target].len() {
            let moved = self.dependents[old.target][position];
            self.dependent_positions[moved] = Some(position);
        }
    }

    fn set_generating_transition(&mut self, source: usize, transition: TransitionKey) {
        assert_eq!(transition.source, source);
        assert_ne!(
            transition.target, source,
            "self-loop cannot generate a shortest path with nonnegative costs"
        );
        self.remove_generating_transition(source);
        let position = self.dependents[transition.target].len();
        self.dependents[transition.target].push(source);
        self.dependent_positions[source] = Some(position);
        self.generating_transition[source] = Some(transition);
    }
}

fn compute_shortest_paths(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
) -> Result<ShortestPaths> {
    let mut is_goal = vec![false; working.states.len()];
    for (state_id, region) in working.states.iter().enumerate() {
        if semantics.region_is_goal(region)? {
            is_goal[state_id] = true;
        }
    }
    ensure!(
        is_goal.iter().any(|is_goal| *is_goal),
        "Cartesian abstraction has no abstract goal state"
    );
    let mut distances = vec![f64::INFINITY; working.states.len()];
    let mut generating_transition = vec![None; working.states.len()];
    let mut heap = BinaryHeap::new();
    for (state_id, &state_is_goal) in is_goal.iter().enumerate() {
        if state_is_goal {
            distances[state_id] = 0.0;
            heap.push((Reverse(NotNan::new(0.0).unwrap()), state_id));
        }
    }
    while let Some((Reverse(distance), target)) = heap.pop() {
        let distance = distance.into_inner();
        if distance > distances[target] + EPSILON {
            continue;
        }
        for &transition_id in &working.incoming[target] {
            let transition = working.transition(transition_id);
            if transition.source == target {
                continue;
            }
            let cost = semantics.operator_costs[transition.concrete_op_id];
            ensure!(
                cost >= -EPSILON && cost.is_finite(),
                "invalid operator cost {cost}"
            );
            let alternative = distance + cost.max(0.0);
            let source = transition.source;
            if alternative + EPSILON < distances[source] {
                distances[source] = alternative;
                generating_transition[source] = Some(TransitionKey {
                    source,
                    concrete_op_id: transition.concrete_op_id,
                    target,
                });
                heap.push((Reverse(NotNan::new(alternative).unwrap()), source));
            }
        }
    }
    let mut dependents = vec![Vec::new(); working.states.len()];
    let mut dependent_positions = vec![None; working.states.len()];
    for (source, transition) in generating_transition.iter().enumerate() {
        if let Some(transition) = transition {
            let position = dependents[transition.target].len();
            dependents[transition.target].push(source);
            dependent_positions[source] = Some(position);
        }
    }
    Ok(ShortestPaths {
        distances,
        generating_transition,
        dependents,
        dependent_positions,
        is_goal,
        invalid: vec![false; working.states.len()],
    })
}

fn update_shortest_paths_after_split(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    mut shortest_paths: ShortestPaths,
    split_state_id: usize,
    new_state_id: usize,
) -> Result<ShortestPaths> {
    let old_num_states = shortest_paths.distances.len();
    ensure!(
        new_state_id == old_num_states && working.states.len() == old_num_states + 1,
        "Cartesian incremental shortest-path update requires one appended split state"
    );

    let mut queue = std::collections::VecDeque::new();
    let mut invalid_states = Vec::new();
    let invalidate = |state_id: usize,
                      shortest_paths: &mut ShortestPaths,
                      invalid_states: &mut Vec<usize>,
                      queue: &mut std::collections::VecDeque<usize>| {
        if !shortest_paths.invalid[state_id] {
            shortest_paths.invalid[state_id] = true;
            invalid_states.push(state_id);
            queue.push_back(state_id);
        }
    };
    let parent_distance = shortest_paths.distances[split_state_id];
    shortest_paths.distances.push(parent_distance);
    shortest_paths.generating_transition.push(None);
    shortest_paths.dependents.push(Vec::new());
    shortest_paths.dependent_positions.push(None);
    shortest_paths.is_goal[split_state_id] =
        semantics.region_is_goal(&working.states[split_state_id])?;
    shortest_paths
        .is_goal
        .push(semantics.region_is_goal(&working.states[new_state_id])?);
    shortest_paths.invalid.push(false);

    invalidate(
        split_state_id,
        &mut shortest_paths,
        &mut invalid_states,
        &mut queue,
    );
    invalidate(
        new_state_id,
        &mut shortest_paths,
        &mut invalid_states,
        &mut queue,
    );
    while let Some(target) = queue.pop_front() {
        shortest_paths.remove_generating_transition(target);
        let dependents = std::mem::take(&mut shortest_paths.dependents[target]);
        for source in dependents {
            let transition = shortest_paths.generating_transition[source]
                .take()
                .expect("Cartesian shortest-path dependent has no generating transition");
            assert_eq!(transition.target, target);
            shortest_paths.dependent_positions[source] = None;
            invalidate(source, &mut shortest_paths, &mut invalid_states, &mut queue);
        }
    }

    for &state_id in &invalid_states {
        shortest_paths.distances[state_id] = f64::INFINITY;
    }

    let mut heap = BinaryHeap::new();
    for &state_id in &invalid_states {
        if shortest_paths.is_goal[state_id] {
            shortest_paths.distances[state_id] = 0.0;
            heap.push((Reverse(NotNan::new(0.0).unwrap()), state_id));
        }
    }

    for &source in &invalid_states {
        for &transition_id in &working.outgoing[source] {
            let transition = working.transition(transition_id);
            if transition.source == transition.target || shortest_paths.invalid[transition.target] {
                continue;
            }
            let target_distance = shortest_paths.distances[transition.target];
            if !target_distance.is_finite() {
                continue;
            }
            let candidate =
                target_distance + semantics.operator_costs[transition.concrete_op_id].max(0.0);
            if candidate + EPSILON < shortest_paths.distances[source] {
                shortest_paths.distances[source] = candidate;
                shortest_paths.set_generating_transition(
                    source,
                    TransitionKey {
                        source,
                        concrete_op_id: transition.concrete_op_id,
                        target: transition.target,
                    },
                );
                heap.push((Reverse(NotNan::new(candidate).unwrap()), source));
            }
        }
    }

    while let Some((Reverse(distance), target)) = heap.pop() {
        let distance = distance.into_inner();
        if distance > shortest_paths.distances[target] + EPSILON {
            continue;
        }
        for &transition_id in &working.incoming[target] {
            let transition = working.transition(transition_id);
            if transition.source == target || !shortest_paths.invalid[transition.source] {
                continue;
            }
            let alternative =
                distance + semantics.operator_costs[transition.concrete_op_id].max(0.0);
            if alternative + EPSILON < shortest_paths.distances[transition.source] {
                shortest_paths.distances[transition.source] = alternative;
                shortest_paths.set_generating_transition(
                    transition.source,
                    TransitionKey {
                        source: transition.source,
                        concrete_op_id: transition.concrete_op_id,
                        target,
                    },
                );
                heap.push((
                    Reverse(NotNan::new(alternative).unwrap()),
                    transition.source,
                ));
            }
        }
    }

    #[cfg(debug_assertions)]
    if working.states.len() <= 512 {
        let reference = compute_shortest_paths(working, semantics)?;
        for state_id in 0..working.states.len() {
            let actual = shortest_paths.distances[state_id];
            let expected = reference.distances[state_id];
            assert!(
                (actual == expected) || (actual - expected).abs() <= 1e-7,
                "incremental Cartesian distance mismatch at state {state_id}: {actual} vs {expected}"
            );
        }
    }

    for state_id in invalid_states {
        shortest_paths.invalid[state_id] = false;
    }
    Ok(shortest_paths)
}

#[derive(Debug)]
struct ConcretePlan {
    operator_ids: Vec<usize>,
    cost: f64,
}

enum PlanCheck {
    ConcretePlan(ConcretePlan),
    AbstractDeadEnd(usize),
    Refine(Split),
}

#[derive(Debug)]
struct RefinementRootDeadEnd {
    abstract_state_id: usize,
}

impl std::fmt::Display for RefinementRootDeadEnd {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "concrete refinement root maps to abstract dead end {}",
            self.abstract_state_id
        )
    }
}

impl std::error::Error for RefinementRootDeadEnd {}

fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-7 * left.abs().max(right.abs()).max(1.0)
}

fn concrete_is_goal(
    semantics: &CartesianSemantics<'_>,
    state_packer: &IntDoublePacker,
    propositions: &[u64],
) -> bool {
    (0..semantics.task.get_num_goals()).all(|goal_id| {
        fact_is_hold(
            semantics.task.get_goal_fact(goal_id),
            state_packer,
            propositions,
        )
    })
}

fn numeric_split_choice_key(variable_name: &str, boundary: f64, lower_closed: bool) -> u64 {
    mix_seed(stable_text_seed(variable_name) ^ boundary.to_bits()) ^ (u64::from(lower_closed) << 63)
}

fn split_choice_key(semantics: &CartesianSemantics<'_>, split: &Split) -> u64 {
    match split {
        Split::Propositional { var_id, wanted, .. } => {
            let var_id = u64::try_from(*var_id).expect("split variable id does not fit u64");
            wanted
                .iter()
                .fold(var_id, |key, value| mix_seed(key ^ u64::from(*value)))
        }
        Split::Numeric {
            var_id,
            boundary,
            lower_includes_boundary,
            ..
        } => {
            let variable_name = semantics.task.numeric_variables()[*var_id].name();
            numeric_split_choice_key(variable_name, *boundary, *lower_includes_boundary)
        }
    }
}

fn split_child_regions(
    working: &WorkingAbstraction,
    split: &Split,
) -> Result<(StateRegion, StateRegion)> {
    let parent = working
        .states
        .get(split.state_id())
        .with_context(|| format!("missing split state {}", split.state_id()))?;
    match split {
        Split::Propositional {
            var_id,
            wanted,
            witness_value,
            ..
        } => {
            let current = parent
                .propositions
                .get(*var_id)
                .with_context(|| format!("split references missing prop var {var_id}"))?;
            ensure!(
                wanted.windows(2).all(|values| values[0] < values[1]),
                "propositional Cartesian split values must be sorted and unique: {wanted:?}"
            );
            let wanted_values = current
                .iter()
                .copied()
                .filter(|value| wanted.binary_search(value).is_ok())
                .collect::<Vec<_>>();
            let other_values = current
                .iter()
                .copied()
                .filter(|value| wanted.binary_search(value).is_err())
                .collect::<Vec<_>>();
            ensure!(
                !wanted_values.is_empty() && !other_values.is_empty(),
                "non-strict propositional Cartesian split on var {var_id}: current={current:?}, wanted={wanted:?}"
            );
            let witness_is_wanted = wanted_values.binary_search(witness_value).is_ok();
            let mut wanted_region = parent.clone();
            Arc::make_mut(&mut wanted_region.propositions)[*var_id] = wanted_values;
            let mut other_region = parent.clone();
            Arc::make_mut(&mut other_region.propositions)[*var_id] = other_values;
            Ok(if witness_is_wanted {
                (wanted_region, other_region)
            } else {
                (other_region, wanted_region)
            })
        }
        Split::Numeric {
            var_id,
            boundary,
            lower_includes_boundary,
            witness_value,
            ..
        } => {
            let current = *parent
                .numeric
                .get(*var_id)
                .with_context(|| format!("split references missing numeric var {var_id}"))?;
            ensure!(
                current.can_split_at(*boundary, *lower_includes_boundary),
                "non-strict numeric Cartesian split on var {var_id} at {boundary}: parent={current:?}, include_lower={lower_includes_boundary}"
            );
            let lower = interval_intersection(
                current,
                Interval::new(
                    f64::NEG_INFINITY,
                    *boundary,
                    false,
                    *lower_includes_boundary,
                ),
            );
            let upper = interval_intersection(
                current,
                Interval::new(*boundary, f64::INFINITY, !*lower_includes_boundary, false),
            );
            let witness_is_lower = lower.contains(*witness_value);
            ensure!(
                witness_is_lower ^ upper.contains(*witness_value),
                "numeric split does not place witness {witness_value} in exactly one child"
            );
            let mut lower_region = parent.clone();
            Arc::make_mut(&mut lower_region.numeric)[*var_id] = lower;
            let mut upper_region = parent.clone();
            Arc::make_mut(&mut upper_region.numeric)[*var_id] = upper;
            Ok(if witness_is_lower {
                (lower_region, upper_region)
            } else {
                (upper_region, lower_region)
            })
        }
    }
}

fn projected_transition_count(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    split: &Split,
) -> Result<usize> {
    let split_state_id = split.state_id();
    let split_dimension = split.dimension();
    let new_state_id = working.states.len();
    let (old_child, new_child) = split_child_regions(working, split)?;
    let mut incident = working.outgoing[split_state_id].clone();
    incident.extend(working.incoming[split_state_id].iter().copied());
    incident.sort_unstable();
    incident.dedup();

    let unaffected = working
        .transition_ids_by_key
        .len()
        .checked_sub(incident.len())
        .expect("incident Cartesian transition count exceeds active transition count");
    let mut replacements = HashSet::new();
    for transition_id in incident {
        let transition = working.transition(transition_id);
        debug_assert!(
            transition.source != transition.target,
            "Cartesian non-loop storage contains a self loop"
        );
        let sources: &[usize] = if transition.source == split_state_id {
            &[split_state_id, new_state_id]
        } else {
            std::slice::from_ref(&transition.source)
        };
        let targets: &[usize] = if transition.target == split_state_id {
            &[split_state_id, new_state_id]
        } else {
            std::slice::from_ref(&transition.target)
        };
        for &source in sources {
            let source_region = if source == split_state_id {
                &old_child
            } else if source == new_state_id {
                &new_child
            } else {
                &working.states[source]
            };
            for &target in targets {
                let target_region = if target == split_state_id {
                    &old_child
                } else if target == new_state_id {
                    &new_child
                } else {
                    &working.states[target]
                };
                let may_transition = if semantics
                    .operator_depends_on_split(transition.concrete_op_id, split_dimension)
                {
                    semantics.may_transition(
                        source_region,
                        transition.concrete_op_id,
                        target_region,
                    )?
                } else {
                    semantics.may_transition_after_independent_split(
                        source_region,
                        transition.concrete_op_id,
                        target_region,
                        split_dimension,
                    )?
                };
                if may_transition && source != target {
                    replacements.insert(TransitionKey {
                        source,
                        concrete_op_id: transition.concrete_op_id,
                        target,
                    });
                }
            }
        }
    }
    let split_dependent_operators = semantics.split_dependent_operators(split_dimension);
    for concrete_op_id in
        working.self_loop_operator_ids[split_state_id].intersection_iter(split_dependent_operators)
    {
        for (source, source_region) in [(split_state_id, &old_child), (new_state_id, &new_child)] {
            let targets = [(split_state_id, &old_child), (new_state_id, &new_child)];
            let may_targets = semantics.parent_loop_source_to_split_children(
                source_region,
                concrete_op_id,
                [targets[0].1, targets[1].1],
                split_dimension,
            )?;
            for ((target, _), may_transition) in targets.into_iter().zip(may_targets) {
                if source != target && may_transition {
                    replacements.insert(TransitionKey {
                        source,
                        concrete_op_id,
                        target,
                    });
                }
            }
        }
    }
    unaffected
        .checked_add(replacements.len())
        .context("projected Cartesian transition count overflow")
}

fn retain_min_growth_splits<T>(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    candidates: &mut Vec<T>,
    split: impl Fn(&T) -> &Split,
) -> Result<()> {
    let projected_transition_counts = candidates
        .iter()
        .map(|candidate| projected_transition_count(working, semantics, split(candidate)))
        .collect::<Result<Vec<_>>>()?;
    let minimum = projected_transition_counts
        .iter()
        .copied()
        .min()
        .context("cannot rank an empty split candidate set by growth")?;
    let mut index = 0;
    candidates.retain(|_| {
        let retain = projected_transition_counts[index] == minimum;
        index += 1;
        retain
    });
    Ok(())
}

fn replay_optimal_abstract_trace(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    shortest_paths: &ShortestPaths,
    state_packer: &Arc<IntDoublePacker>,
    axiom_evaluator: &AxiomEvaluator<'_>,
    refinement_root: &CartesianConcreteState,
) -> Result<PlanCheck> {
    let mut propositions = refinement_root.propositions.clone();
    let mut numeric = refinement_root.numeric.clone();
    let mut prop_values = Vec::new();
    let mut successor_prop_values = Vec::new();
    semantics.concrete_prop_values(state_packer, &propositions, &mut prop_values);
    let initial_abstract_state = working.hierarchy.map_state(&prop_values, &numeric)?;
    if !shortest_paths.distances[initial_abstract_state].is_finite() {
        return Ok(PlanCheck::AbstractDeadEnd(initial_abstract_state));
    }
    let abstract_plan_cost = shortest_paths.distances[initial_abstract_state];
    let mut operator_ids = Vec::new();
    let mut concrete_cost = 0.0;

    loop {
        semantics.concrete_prop_values(state_packer, &propositions, &mut prop_values);
        let abstract_state = working.hierarchy.map_state(&prop_values, &numeric)?;
        let abstract_distance = shortest_paths.distances[abstract_state];
        ensure!(
            approximately_equal(concrete_cost + abstract_distance, abstract_plan_cost),
            "concrete trace left optimal abstract path: g={concrete_cost} h={abstract_distance} initial_h={abstract_plan_cost}"
        );

        if shortest_paths.is_goal[abstract_state] {
            if concrete_is_goal(semantics, state_packer, &propositions) {
                return Ok(PlanCheck::ConcretePlan(ConcretePlan {
                    operator_ids,
                    cost: concrete_cost,
                }));
            }
            let failed_goals = (0..semantics.task.get_num_goals())
                .map(|goal_id| semantics.task.get_goal_fact(goal_id))
                .filter(|goal| !fact_is_hold(goal, state_packer, &propositions))
                .collect::<Vec<_>>();
            ensure!(
                !failed_goals.is_empty(),
                "abstract goal contains a concrete non-goal without a failed goal fact"
            );
            let mut candidates = failed_goals
                .iter()
                .map(|goal| {
                    split_failed_fact(
                        working,
                        semantics,
                        abstract_state,
                        goal,
                        &prop_values,
                        &numeric,
                        format!("goal {goal:?}"),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            retain_min_growth_splits(working, semantics, &mut candidates, |split| split)?;
            let index = semantics.choose_split_index(&candidates, 0x474F_414C);
            return Ok(PlanCheck::Refine(candidates.swap_remove(index)));
        }

        ensure!(
            operator_ids.len() <= working.states.len(),
            "Cartesian generating transitions contain a cycle"
        );
        let transition = shortest_paths.generating_transition[abstract_state].context(
            "non-goal Cartesian state with finite distance has no generating transition",
        )?;
        ensure!(
            working.contains_transition(transition),
            "Cartesian shortest path references missing transition {transition:?}"
        );
        let op_id = transition.concrete_op_id;
        let op = &semantics.task.get_operators()[op_id];
        let failed_preconditions = op
            .preconditions()
            .iter()
            .filter(|fact| !fact_is_hold(fact, state_packer, &propositions))
            .collect::<Vec<_>>();
        if !failed_preconditions.is_empty() {
            let mut candidates = failed_preconditions
                .iter()
                .map(|failed| {
                    split_failed_fact(
                        working,
                        semantics,
                        abstract_state,
                        failed,
                        &prop_values,
                        &numeric,
                        format!("operator {op_id} ({}) precondition {failed:?}", op.name()),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            retain_min_growth_splits(working, semantics, &mut candidates, |split| split)?;
            let index = semantics.choose_split_index(&candidates, 0x5052_4543);
            return Ok(PlanCheck::Refine(candidates.swap_remove(index)));
        }

        let source_numeric = numeric.clone();
        progress(
            op,
            axiom_evaluator,
            state_packer,
            &mut propositions,
            &mut numeric,
        )?;
        semantics.concrete_prop_values(state_packer, &propositions, &mut successor_prop_values);
        let concrete_target = working
            .hierarchy
            .map_state(&successor_prop_values, &numeric)?;
        if concrete_target != transition.target {
            return Ok(PlanCheck::Refine(split_deviation(
                working,
                semantics,
                abstract_state,
                transition.target,
                op_id,
                &successor_prop_values,
                &source_numeric,
                &numeric,
            )?));
        }

        let op_cost = semantics.operator_costs[op_id];
        ensure!(
            approximately_equal(
                op_cost + shortest_paths.distances[transition.target],
                abstract_distance
            ),
            "Cartesian generating transition is not distance preserving"
        );
        concrete_cost += op_cost;
        operator_ids.push(op_id);
    }
}

fn validate_concrete_plan(
    semantics: &CartesianSemantics<'_>,
    state_packer: &Arc<IntDoublePacker>,
    axiom_evaluator: &AxiomEvaluator<'_>,
    refinement_root: &CartesianConcreteState,
    plan: &ConcretePlan,
) -> Result<()> {
    let mut propositions = refinement_root.propositions.clone();
    let mut numeric = refinement_root.numeric.clone();
    let mut cost = 0.0;
    for (step, &op_id) in plan.operator_ids.iter().enumerate() {
        let op =
            semantics.task.get_operators().get(op_id).with_context(|| {
                format!("concrete plan step {step} has invalid operator {op_id}")
            })?;
        for precondition in op.preconditions() {
            ensure!(
                fact_is_hold(precondition, state_packer, &propositions),
                "concrete plan operator {op_id} ({}) has false precondition {precondition:?} at step {step}",
                op.name()
            );
        }
        progress(
            op,
            axiom_evaluator,
            state_packer,
            &mut propositions,
            &mut numeric,
        )?;
        cost += semantics.operator_costs[op_id];
    }
    ensure!(
        concrete_is_goal(semantics, state_packer, &propositions),
        "replayed Cartesian concrete plan does not satisfy the full goal"
    );
    ensure!(
        approximately_equal(cost, plan.cost),
        "replayed Cartesian concrete plan cost {cost} differs from recorded cost {}",
        plan.cost
    );
    Ok(())
}

fn split_failed_fact(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    state_id: usize,
    fact: &ExplicitFact,
    prop_values: &[usize],
    numeric_values: &[f64],
    description: String,
) -> Result<Split> {
    if let Some(tree_id) = semantics
        .comparison_tree_by_prop_var
        .get(fact.var())
        .copied()
        .flatten()
    {
        return comparison_refinement(
            working,
            semantics,
            state_id,
            tree_id,
            numeric_values,
            ComparisonRefinementGoal::exclude(fact.value())?,
            description,
        );
    }
    if !semantics.propositional_axioms_by_prop_var[fact.var()].is_empty() {
        let default_value = semantics.propositional_axiom_default(fact.var())?;
        if fact.value() == default_value {
            let concrete_value = *prop_values
                .get(fact.var())
                .with_context(|| format!("missing concrete prop var {}", fact.var()))?;
            ensure!(
                concrete_value != default_value,
                "failed default-valued derived fact unexpectedly holds for variable {}",
                fact.var()
            );
            for &axiom_id in &semantics.propositional_axioms_by_prop_var[fact.var()] {
                let axiom = &semantics.task.axioms()[axiom_id];
                if axiom.effect_value() != concrete_value
                    || !conditions_hold_concretely(axiom.conditions(), prop_values)?
                {
                    continue;
                }
                for condition in axiom.conditions() {
                    if !semantics.region_guarantees_fact(&working.states[state_id], condition)? {
                        return split_to_guarantee_fact(
                            working,
                            semantics,
                            state_id,
                            condition,
                            prop_values,
                            numeric_values,
                            format!(
                                "{description} via concrete axiom {axiom_id} condition {condition:?}"
                            ),
                        );
                    }
                }
                bail!(
                    "derived default fact {fact:?} is abstractly admitted although concrete axiom {axiom_id} is guaranteed"
                );
            }
            bail!(
                "concrete derived value {concrete_value} for variable {} has no supporting axiom",
                fact.var()
            );
        }
        for &axiom_id in &semantics.propositional_axioms_by_prop_var[fact.var()] {
            let axiom = &semantics.task.axioms()[axiom_id];
            if axiom.effect_value() != fact.value()
                || !all_conditions_admitted(
                    semantics,
                    &working.states[state_id],
                    axiom.conditions(),
                )?
            {
                continue;
            }
            for condition in axiom.conditions() {
                let value = *prop_values
                    .get(condition.var())
                    .with_context(|| format!("missing concrete prop var {}", condition.var()))?;
                if value != condition.value() {
                    return split_failed_fact(
                        working,
                        semantics,
                        state_id,
                        condition,
                        prop_values,
                        numeric_values,
                        format!("{description} via axiom {axiom_id} condition {condition:?}"),
                    );
                }
            }
        }
        bail!(
            "derived fact {fact:?} is false in the concrete state, but every supporting axiom condition holds"
        );
    }
    let witness_value = *prop_values
        .get(fact.var())
        .with_context(|| format!("missing concrete prop var {}", fact.var()))?
        as PropValueId;
    ensure!(
        witness_value != fact.value() as PropValueId,
        "failed fact split witness unexpectedly satisfies {fact:?}"
    );
    Ok(Split::Propositional {
        state_id,
        var_id: fact.var(),
        wanted: vec![fact.value() as PropValueId],
        witness_value,
        description,
    })
}

fn split_to_guarantee_fact(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    state_id: usize,
    fact: &ExplicitFact,
    prop_values: &[usize],
    numeric_values: &[f64],
    description: String,
) -> Result<Split> {
    let concrete_value = *prop_values
        .get(fact.var())
        .with_context(|| format!("missing concrete prop var {}", fact.var()))?;
    ensure!(
        concrete_value == fact.value(),
        "cannot guarantee fact {fact:?}: concrete value is {concrete_value}"
    );
    if let Some(tree_id) = semantics
        .comparison_tree_by_prop_var
        .get(fact.var())
        .copied()
        .flatten()
    {
        return comparison_refinement(
            working,
            semantics,
            state_id,
            tree_id,
            numeric_values,
            ComparisonRefinementGoal::guarantee(fact.value())?,
            description,
        );
    }
    if !semantics.propositional_axioms_by_prop_var[fact.var()].is_empty() {
        let default_value = semantics.propositional_axiom_default(fact.var())?;
        if fact.value() == default_value {
            for &axiom_id in &semantics.propositional_axioms_by_prop_var[fact.var()] {
                let axiom = &semantics.task.axioms()[axiom_id];
                if !all_conditions_admitted(
                    semantics,
                    &working.states[state_id],
                    axiom.conditions(),
                )? {
                    continue;
                }
                let condition = axiom
                    .conditions()
                    .iter()
                    .find(|condition| {
                        prop_values
                            .get(condition.var())
                            .is_some_and(|&value| value != condition.value())
                    })
                    .with_context(|| {
                        format!(
                            "concrete default value for derived variable {} conflicts with firing axiom {axiom_id}",
                            fact.var()
                        )
                    })?;
                let witness_value = prop_values[condition.var()];
                let witness_fact = ExplicitFact::new(condition.var(), witness_value);
                return split_to_guarantee_fact(
                    working,
                    semantics,
                    state_id,
                    &witness_fact,
                    prop_values,
                    numeric_values,
                    format!("{description} by disabling axiom {axiom_id} condition {condition:?}"),
                );
            }
            bail!(
                "derived default fact {fact:?} is not guaranteed although no competing axiom is admitted"
            );
        }

        for &axiom_id in &semantics.propositional_axioms_by_prop_var[fact.var()] {
            let axiom = &semantics.task.axioms()[axiom_id];
            if axiom.effect_value() != fact.value()
                || !conditions_hold_concretely(axiom.conditions(), prop_values)?
            {
                continue;
            }
            for condition in axiom.conditions() {
                if !semantics.region_guarantees_fact(&working.states[state_id], condition)? {
                    return split_to_guarantee_fact(
                        working,
                        semantics,
                        state_id,
                        condition,
                        prop_values,
                        numeric_values,
                        format!("{description} via axiom {axiom_id} condition {condition:?}"),
                    );
                }
            }
            bail!(
                "derived fact {fact:?} is not guaranteed although supporting axiom {axiom_id} is guaranteed"
            );
        }
        bail!("concrete derived fact {fact:?} has no supporting axiom");
    }

    let witness_value = concrete_value as PropValueId;
    let allowed = working
        .states
        .get(state_id)
        .and_then(|state| state.propositions.get(fact.var()))
        .with_context(|| format!("missing Cartesian state {state_id} prop var {}", fact.var()))?;
    ensure!(
        allowed.binary_search(&witness_value).is_ok() && allowed.len() > 1,
        "fact {fact:?} is already guaranteed in Cartesian state {state_id}"
    );
    Ok(Split::Propositional {
        state_id,
        var_id: fact.var(),
        wanted: vec![witness_value],
        witness_value,
        description,
    })
}

#[derive(Debug, Clone, Copy)]
enum ComparisonRefinementGoal {
    ExcludeDesired(bool),
    GuaranteeDesired(bool),
}

impl ComparisonRefinementGoal {
    fn exclude(prop_value: usize) -> Result<Self> {
        Ok(Self::ExcludeDesired(comparison_truth(prop_value)?))
    }

    fn guarantee(prop_value: usize) -> Result<Self> {
        Ok(Self::GuaranteeDesired(comparison_truth(prop_value)?))
    }

    fn desired_truth(self) -> bool {
        match self {
            Self::ExcludeDesired(truth) | Self::GuaranteeDesired(truth) => truth,
        }
    }
}

fn comparison_refinement(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    state_id: usize,
    tree_id: usize,
    numeric_values: &[f64],
    goal: ComparisonRefinementGoal,
    description: String,
) -> Result<Split> {
    let desired_truth = goal.desired_truth();
    let tree = semantics
        .comparison_trees
        .get(tree_id)
        .with_context(|| format!("missing comparison tree {tree_id}"))?;
    let concrete_truth = tree.evaluate_point(numeric_values);
    ensure!(
        matches!(goal, ComparisonRefinementGoal::ExcludeDesired(_))
            == (concrete_truth != desired_truth),
        "comparison refinement goal disagrees with concrete truth for tree {tree_id}"
    );
    let state = working
        .states
        .get(state_id)
        .with_context(|| format!("missing Cartesian state {state_id}"))?;
    let mut candidates = Vec::new();
    for var_id in tree.regular_numeric_var_dependencies(semantics.task) {
        let witness_value = float_tolerance::canonicalize(
            *numeric_values
                .get(var_id)
                .with_context(|| format!("missing concrete numeric var {var_id}"))?,
        );
        ensure!(
            witness_value.is_finite(),
            "comparison split witness for numeric var {var_id} is non-finite: {witness_value}"
        );
        let parent = *state
            .numeric
            .get(var_id)
            .with_context(|| format!("missing Cartesian numeric var {var_id}"))?;
        let mut boundaries = Vec::new();
        if semantics.refinement_direction == CartesianRefinementDirection::Regression {
            boundaries.extend(semantics.target_split_boundaries.iter().copied());
        }
        boundaries.push(witness_value);
        boundaries.sort_by(f64::total_cmp);
        boundaries.dedup_by(|left, right| left.to_bits() == right.to_bits());

        for boundary in boundaries {
            for lower_includes_boundary in [true, false] {
                if !parent.can_split_at(boundary, lower_includes_boundary) {
                    continue;
                }
                let lower = interval_intersection(
                    parent,
                    Interval::new(f64::NEG_INFINITY, boundary, false, lower_includes_boundary),
                );
                let upper = interval_intersection(
                    parent,
                    Interval::new(boundary, f64::INFINITY, !lower_includes_boundary, false),
                );
                let (witness_child, other_child) = if lower.contains(witness_value) {
                    (lower, upper)
                } else {
                    ensure!(
                        upper.contains(witness_value),
                        "comparison split at {boundary} loses witness {witness_value} for numeric var {var_id}"
                    );
                    (upper, lower)
                };
                let mut child_numeric = state.numeric.clone();
                Arc::make_mut(&mut child_numeric)[var_id] = witness_child;
                let witness_result = tree.evaluate_interval(&child_numeric);
                ensure!(
                    witness_result != Some(!concrete_truth),
                    "comparison interval for tree {tree_id} excludes its concrete witness after splitting numeric var {var_id}"
                );
                Arc::make_mut(&mut child_numeric)[var_id] = other_child;
                let other_result = tree.evaluate_interval(&child_numeric);
                let achieved = match goal {
                    ComparisonRefinementGoal::ExcludeDesired(_) => {
                        witness_result == Some(!desired_truth)
                    }
                    ComparisonRefinementGoal::GuaranteeDesired(_) => {
                        witness_result == Some(desired_truth)
                    }
                };
                let separates_truth = achieved && other_result == Some(!concrete_truth);
                let candidate = Split::Numeric {
                    state_id,
                    var_id,
                    boundary,
                    lower_includes_boundary,
                    witness_value,
                    description: description.clone(),
                };
                candidates.push((separates_truth, achieved, candidate));
            }
        }
    }
    ensure!(
        !candidates.is_empty(),
        "comparison tree {tree_id} has no strict regular-variable split in Cartesian state {state_id}"
    );
    retain_min_growth_splits(working, semantics, &mut candidates, |(_, _, split)| split)?;
    let has_target_centered_candidate = semantics.refinement_direction
        == CartesianRefinementDirection::Regression
        && candidates
            .iter()
            .any(|(separates_truth, _, _)| *separates_truth);
    if has_target_centered_candidate {
        candidates.retain(|(separates_truth, _, _)| *separates_truth);
    }
    let has_achieving_candidate = candidates.iter().any(|(_, achieved, _)| *achieved);
    if has_achieving_candidate {
        candidates.retain(|(_, achieved, _)| *achieved);
    }
    let keys = candidates
        .iter()
        .map(|(_, _, split)| split_choice_key(semantics, split))
        .collect::<Vec<_>>();
    let index = semantics.choose_keyed_index(&keys, 0x434F_4D50);
    Ok(candidates.swap_remove(index).2)
}

fn comparison_truth(prop_value: usize) -> Result<bool> {
    match prop_value {
        0 => Ok(true),
        1 => Ok(false),
        _ => bail!("invalid comparison fact value {prop_value}"),
    }
}

fn conditions_hold_concretely(conditions: &[ExplicitFact], prop_values: &[usize]) -> Result<bool> {
    for condition in conditions {
        let value = *prop_values
            .get(condition.var())
            .with_context(|| format!("missing concrete prop var {}", condition.var()))?;
        if value != condition.value() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn all_conditions_admitted(
    semantics: &CartesianSemantics<'_>,
    region: &StateRegion,
    conditions: &[ExplicitFact],
) -> Result<bool> {
    for condition in conditions {
        if !semantics.region_admits_fact(region, condition)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn split_deviation(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    source_state_id: usize,
    target_state_id: usize,
    op_id: usize,
    successor_prop: &[usize],
    source_numeric: &[f64],
    successor_numeric: &[f64],
) -> Result<Split> {
    let target = &working.states[target_state_id];
    let mut candidates = Vec::new();
    let mut rejected_numeric_splits = Vec::new();
    for (var_id, allowed) in target.propositions.iter().enumerate() {
        if semantics.comparison_tree_by_prop_var[var_id].is_some()
            || !semantics.propositional_axioms_by_prop_var[var_id].is_empty()
        {
            continue;
        }
        let value = successor_prop[var_id] as PropValueId;
        if allowed.binary_search(&value).is_err() {
            let op = &semantics.task.get_operators()[op_id];
            let unaffected = !op.effects().iter().any(|effect| effect.var_id() == var_id);
            ensure!(
                unaffected,
                "operator {op_id} effect image admitted wrong target prop region for var {var_id}"
            );
            candidates.push(Split::Propositional {
                state_id: source_state_id,
                var_id,
                wanted: allowed.clone(),
                witness_value: value,
                description: format!(
                    "operator {op_id} successor prop var {var_id}={value} outside target {allowed:?}"
                ),
            });
        }
    }

    for (var_id, target_interval) in target.numeric.iter().copied().enumerate() {
        let successor = successor_numeric[var_id];
        if target_interval.contains(successor) {
            continue;
        }
        let preimage = semantics
            .numeric_effect_preimage(target_interval, op_id, var_id)?
            .with_context(|| {
                format!(
                    "Cartesian transition for operator {op_id} has no numeric preimage for var {var_id} and target {target_interval:?}"
                )
            })?;
        let source = source_numeric[var_id];
        if preimage.contains(source) {
            rejected_numeric_splits.push(format!(
                "var {var_id}: source={source}, successor={successor}, target={target_interval:?}, preimage={preimage:?} contains source"
            ));
            continue;
        }
        let (boundary, lower_includes_boundary) =
            if source < preimage.lower || (source == preimage.lower && !preimage.lower_closed) {
                (preimage.lower, !preimage.lower_closed)
            } else {
                ensure!(
                    source > preimage.upper || (source == preimage.upper && !preimage.upper_closed),
                    "numeric successor mismatch has no separating preimage boundary"
                );
                (preimage.upper, preimage.upper_closed)
            };
        let parent = working.states[source_state_id].numeric[var_id];
        ensure!(
            parent.contains(source),
            "Cartesian source state {source_state_id} interval {parent:?} does not contain concrete numeric var {var_id}={source}"
        );
        if !boundary.is_finite() {
            rejected_numeric_splits.push(format!(
                "var {var_id}: source={source}, successor={successor}, target={target_interval:?}, preimage={preimage:?}, parent={parent:?} has only infinite separating boundary"
            ));
            continue;
        }
        if !parent.can_split_at(boundary, lower_includes_boundary) {
            rejected_numeric_splits.push(format!(
                "var {var_id}: source={source}, successor={successor}, target={target_interval:?}, preimage={preimage:?}, parent={parent:?}, boundary={boundary}, lower_includes_boundary={lower_includes_boundary} is not strict"
            ));
            continue;
        }
        candidates.push(Split::Numeric {
            state_id: source_state_id,
            var_id,
            boundary,
            lower_includes_boundary,
            witness_value: source,
            description: format!(
                "operator {op_id} successor numeric var {var_id}={successor} outside target {target_interval:?}"
            ),
        });
    }
    ensure!(
        !candidates.is_empty(),
        "concrete successor maps from Cartesian state {source_state_id} to a state other than abstract target {target_state_id}, but no sound strict split exists for operator {op_id} ({}); numeric split rejections: [{}]",
        semantics.task.get_operators()[op_id].name(),
        rejected_numeric_splits.join("; ")
    );
    retain_min_growth_splits(working, semantics, &mut candidates, |split| split)?;
    let index = semantics.choose_split_index(&candidates, 0x4445_5649);
    Ok(candidates.swap_remove(index))
}

fn apply_split(
    working: &mut WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    split: Split,
) -> Result<usize> {
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
            ..
        } => {
            let parent = old_region.numeric[var_id];
            let lower = interval_intersection(
                parent,
                Interval::new(f64::NEG_INFINITY, boundary, false, lower_includes_boundary),
            );
            let upper = interval_intersection(
                parent,
                Interval::new(boundary, f64::INFINITY, !lower_includes_boundary, false),
            );
            ensure!(
                !lower.is_empty() && !upper.is_empty(),
                "non-strict numeric Cartesian split on var {var_id} at {boundary}: parent={parent:?}, include_lower={lower_includes_boundary}"
            );
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
                var_id,
                boundary,
                lower_includes_boundary,
                witness_is_lower,
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

fn finalize_abstraction(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    combine_labels: bool,
    compute_operator_footprints: bool,
) -> Result<(
    AbstractTransitionSystem,
    AbstractDistanceTable,
    Vec<usize>,
    Vec<AbstractOperatorFootprint>,
)> {
    let mut grouped: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut raw = Vec::new();
    for transition_id in working.active_transition_ids() {
        let transition = working.transition(transition_id);
        if combine_labels {
            grouped
                .entry((transition.source, transition.target))
                .or_default()
                .push(transition.concrete_op_id);
        } else {
            raw.push((
                transition.source,
                transition.target,
                vec![transition.concrete_op_id],
            ));
        }
    }
    // Self loops have zero shortest-path and saturated-cost requirements. Keep
    // them only while refining, where a later split can turn one into an exact
    // cross-child transition; materializing them here wastes memory without
    // changing standalone, canonical, label-SCP, or regional-SCP values.
    if combine_labels {
        raw.extend(grouped.into_iter().map(|((source, target), mut labels)| {
            labels.sort_unstable();
            labels.dedup();
            (source, target, labels)
        }));
    }
    raw.sort();
    let mut transitions = Vec::with_capacity(raw.len());
    let mut forward = vec![Vec::new(); working.states.len()];
    let mut backward = vec![Vec::new(); working.states.len()];
    let mut footprints = if compute_operator_footprints {
        Vec::with_capacity(raw.len())
    } else {
        Vec::new()
    };
    let shared_state_regions = working
        .states
        .iter()
        .cloned()
        .map(Arc::new)
        .collect::<Vec<_>>();
    let mut relevant = HashSet::new();
    for (transition_id, (source, target, labels)) in raw.into_iter().enumerate() {
        if source != target {
            for &label in &labels {
                relevant.insert(label);
            }
        }
        if compute_operator_footprints {
            footprints.push(AbstractOperatorFootprint {
                labels: labels
                    .iter()
                    .copied()
                    .map(|concrete_op_id| {
                        let footprint = semantics.transition_source_footprint(
                            &shared_state_regions[source],
                            concrete_op_id,
                            &shared_state_regions[target],
                        )?
                        .with_context(|| {
                            format!(
                                "emitted Cartesian transition {source} --{concrete_op_id}--> {target} has an empty source footprint"
                            )
                        })?;
                        let source_region = if footprint == *shared_state_regions[source] {
                            Arc::clone(&shared_state_regions[source])
                        } else {
                            Arc::new(footprint)
                        };
                        Ok(ConcreteOperatorFootprint {
                            concrete_op_id,
                            source_region,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            });
        }
        transitions.push(AbstractTransition {
            transition_id,
            abstract_op_id: transition_id,
            concrete_op_ids: labels,
            source_hash: source,
            target_hash: target,
        });
        forward[source].push(transition_id);
        backward[target].push(transition_id);
    }
    let mut goal_state_hashes = Vec::new();
    for (state_id, region) in working.states.iter().enumerate() {
        if semantics.region_is_goal(region)? {
            goal_state_hashes.push(state_id);
        }
    }
    let initial_prop = semantics.task.get_initial_propositional_state_values();
    let initial_numeric = semantics
        .task
        .get_initial_numeric_state_values()
        .iter()
        .copied()
        .map(float_tolerance::canonicalize)
        .collect::<Vec<_>>();
    let initial_state_hash = working
        .hierarchy
        .map_state(&initial_prop, &initial_numeric)?;
    let transition_system = AbstractTransitionSystem {
        transitions,
        duplicate_transition_attempts: 0,
        backward,
        forward,
        goal_facts: (0..semantics.task.get_num_goals())
            .map(|goal_id| *semantics.task.get_goal_fact(goal_id))
            .collect(),
        goal_state_hashes,
        initial_state_hash,
        hash_multipliers: Vec::new(),
        numeric_domain_sizes: Vec::new(),
        state_regions: shared_state_regions,
    };
    let transition_costs = transition_system
        .transitions
        .iter()
        .map(|transition| {
            transition
                .concrete_op_ids
                .iter()
                .map(|&op_id| semantics.operator_costs[op_id])
                .fold(f64::INFINITY, f64::min)
        })
        .collect::<Vec<_>>();
    let (distances, generating_op_ids) = explicit_distances(&transition_system, &transition_costs)?;
    let distance_table = AbstractDistanceTable {
        distances,
        generating_op_ids,
        initial_state_hash,
        goal_facts: transition_system.goal_facts.clone(),
        hash_multipliers: Vec::new(),
        numeric_domain_sizes: Vec::new(),
    };
    let mut relevant_operator_ids: Vec<_> = relevant.into_iter().collect();
    relevant_operator_ids.sort_unstable();
    Ok((
        transition_system,
        distance_table,
        relevant_operator_ids,
        footprints,
    ))
}

pub fn explicit_distances(
    transition_system: &AbstractTransitionSystem,
    transition_costs: &[f64],
) -> Result<(Vec<f64>, Vec<Option<usize>>)> {
    ensure!(
        transition_system.transitions.len() == transition_costs.len(),
        "transition/cost length mismatch"
    );
    let mut distances = vec![f64::INFINITY; transition_system.backward.len()];
    let mut generating = vec![None; distances.len()];
    let mut heap = BinaryHeap::new();
    for &goal in &transition_system.goal_state_hashes {
        distances[goal] = 0.0;
        heap.push((Reverse(NotNan::new(0.0).unwrap()), goal));
    }
    while let Some((Reverse(distance), target)) = heap.pop() {
        let distance = distance.into_inner();
        if distance > distances[target] + EPSILON {
            continue;
        }
        for &transition_id in &transition_system.backward[target] {
            let transition = &transition_system.transitions[transition_id];
            let cost = transition_costs[transition_id];
            ensure!(
                cost >= -EPSILON && cost.is_finite(),
                "invalid transition cost {cost}"
            );
            let alternative = distance + cost.max(0.0);
            if alternative + EPSILON < distances[transition.source_hash] {
                distances[transition.source_hash] = alternative;
                generating[transition.source_hash] = Some(transition.abstract_op_id);
                heap.push((
                    Reverse(NotNan::new(alternative).unwrap()),
                    transition.source_hash,
                ));
            }
        }
    }
    Ok((distances, generating))
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
        let registry = eval_state.state_registry().ok_or_else(|| {
            EvaluationError::InvalidState(
                "Cartesian abstraction lookup requires state registry".to_string(),
            )
        })?;
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

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
}

fn sorted_values_overlap(left: &[PropValueId], right: &[PropValueId]) -> bool {
    let mut left_id = 0;
    let mut right_id = 0;
    while left_id < left.len() && right_id < right.len() {
        match left[left_id].cmp(&right[right_id]) {
            std::cmp::Ordering::Less => left_id += 1,
            std::cmp::Ordering::Greater => right_id += 1,
            std::cmp::Ordering::Equal => return true,
        }
    }
    false
}

fn interval_intersection(left: Interval, right: Interval) -> Interval {
    let (lower, lower_closed) = if left.lower > right.lower {
        (left.lower, left.lower_closed)
    } else if right.lower > left.lower {
        (right.lower, right.lower_closed)
    } else {
        (left.lower, left.lower_closed && right.lower_closed)
    };
    let (upper, upper_closed) = if left.upper < right.upper {
        (left.upper, left.upper_closed)
    } else if right.upper < left.upper {
        (right.upper, right.upper_closed)
    } else {
        (left.upper, left.upper_closed && right.upper_closed)
    };
    Interval::new(lower, upper, lower_closed, upper_closed)
}
