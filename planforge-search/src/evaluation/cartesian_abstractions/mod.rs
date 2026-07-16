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
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use ordered_float::NotNan;
use planforge_sas::axioms::AxiomEvaluator;
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, ExplicitFact, NumericType,
    metric_operator_cost_from_initial_values,
};
use planforge_sas::utils::int_packer::IntDoublePacker;
use tracing::{debug, info};

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;

use super::abstraction_collections::portfolio::{derive_variant_seed, mix_seed};
use super::abstraction_collections::transition_cost_partitioning::{
    AbstractOperatorFootprint, AbstractTransition, AbstractTransitionSystem,
    ConcreteOperatorFootprint, PropValueId, StateRegion,
};
use super::abstraction_task::{SingleGoalTask, validate_abstraction_operator};
use super::domain_abstractions::cegar::flaw_search::state::progress;
use super::domain_abstractions::comparison_expression::{ComparisonTree, Interval};
use super::domain_abstractions::domain_abstraction_factory::AbstractDistanceTable;
use super::domain_abstractions::utils::{fact_is_hold, get_initial_state, make_prop_state_packer};

const EPSILON: f64 = 1e-9;

fn fact_choice_key(fact: &ExplicitFact) -> u64 {
    let var_id = u64::try_from(fact.var()).expect("fact variable id does not fit u64");
    let value = u64::try_from(fact.value()).expect("fact value does not fit u64");
    var_id.rotate_left(32) ^ value
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CartesianStopReason {
    ConcretePlan,
    StateLimit,
    TimeLimit,
}

#[derive(Debug, Clone)]
pub struct CartesianAbstractionMetadata {
    pub solved_by_self: bool,
    pub stop_reason: CartesianStopReason,
    pub pending_flaw: Option<String>,
    pub refinements: usize,
    pub collection_goal_id: Option<usize>,
    pub collection_variant_id: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct CartesianAbstractionConfig {
    pub max_states: usize,
    pub max_time: Option<Duration>,
    pub combine_labels: bool,
    pub compute_operator_footprints: bool,
    pub random_seed: Option<u64>,
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
            debug: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CartesianAbstractionCollectionConfig {
    pub abstraction: CartesianAbstractionConfig,
    pub variants_per_goal: usize,
    pub max_collection_states: usize,
    pub total_max_time: Option<Duration>,
}

impl Default for CartesianAbstractionCollectionConfig {
    fn default() -> Self {
        Self {
            abstraction: CartesianAbstractionConfig::default(),
            variants_per_goal: 1,
            max_collection_states: 10_000_000,
            total_max_time: None,
        }
    }
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
}

#[derive(Debug, Clone)]
struct WorkingTransition {
    source: usize,
    target: usize,
    concrete_op_id: usize,
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
    transition_ids_by_key: HashMap<TransitionKey, usize>,
    propositional_refinement_counts: Vec<usize>,
    numeric_refinement_counts: Vec<usize>,
}

impl WorkingAbstraction {
    fn new(initial_region: StateRegion) -> Self {
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
            transition_ids_by_key: HashMap::new(),
            propositional_refinement_counts,
            numeric_refinement_counts,
        }
    }

    fn add_transition(&mut self, source: usize, op_id: usize, target: usize) {
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
        self.transition_ids_by_key.contains_key(&key)
    }

    fn refinement_count(&self, split: &Split) -> usize {
        match split {
            Split::Propositional { var_id, .. } => self.propositional_refinement_counts[*var_id],
            Split::Numeric { var_id, .. } => self.numeric_refinement_counts[*var_id],
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
}

struct CartesianSemantics<'task> {
    task: &'task dyn AbstractNumericTask,
    comparison_tree_by_prop_var: Vec<Option<usize>>,
    comparison_trees: Vec<ComparisonTree>,
    propositional_axioms_by_prop_var: Vec<Vec<usize>>,
    operator_costs: Vec<f64>,
    random_seed: Option<u64>,
}

impl<'task> CartesianSemantics<'task> {
    fn new(task: &'task dyn AbstractNumericTask, random_seed: Option<u64>) -> Result<Self> {
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
        Ok(Self {
            task,
            comparison_tree_by_prop_var,
            comparison_trees,
            propositional_axioms_by_prop_var,
            operator_costs,
            random_seed,
        })
    }

    fn choose_keyed_index(&self, keys: &[u64], tag: u64) -> usize {
        assert!(
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
        let numeric = self
            .task
            .numeric_variables()
            .iter()
            .enumerate()
            .map(|(var_id, var)| {
                if matches!(var.get_type(), NumericType::Constant) {
                    Interval::singleton(initial_numeric[var_id])
                } else {
                    Interval::unbounded()
                }
            })
            .collect();
        Ok(StateRegion {
            propositions,
            numeric,
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

    fn may_transition(
        &self,
        source: &StateRegion,
        op_id: usize,
        target: &StateRegion,
    ) -> Result<bool> {
        if !self.operator_may_apply(source, op_id)? {
            return Ok(false);
        }
        let op = &self.task.get_operators()[op_id];

        for var_id in 0..self.task.get_num_variables() {
            if self.comparison_tree_by_prop_var[var_id].is_some()
                || !self.propositional_axioms_by_prop_var[var_id].is_empty()
            {
                continue;
            }
            let source_values = &source.propositions[var_id];
            let target_values = &target.propositions[var_id];
            let mut possible = source_values.clone();
            for effect in op
                .effects()
                .iter()
                .filter(|effect| effect.var_id() == var_id)
            {
                let mut conditions_may_hold = true;
                for condition in effect.conditions() {
                    if !self.region_admits_fact(source, condition)? {
                        conditions_may_hold = false;
                        break;
                    }
                }
                if !conditions_may_hold {
                    continue;
                }
                let mut conditions_guaranteed = true;
                for condition in effect.conditions() {
                    if !self.region_guarantees_fact(source, condition)? {
                        conditions_guaranteed = false;
                        break;
                    }
                }
                if effect.conditions().is_empty() || conditions_guaranteed {
                    possible.clear();
                }
                possible.push(effect.value() as u32);
            }
            possible.sort_unstable();
            possible.dedup();
            if !sorted_values_overlap(&possible, target_values) {
                return Ok(false);
            }
        }

        for numeric_var_id in 0..self.task.numeric_variables().len() {
            if matches!(
                self.task.numeric_variables()[numeric_var_id].get_type(),
                NumericType::Constant
            ) {
                if !source.numeric[numeric_var_id].intersects(&target.numeric[numeric_var_id]) {
                    return Ok(false);
                }
                continue;
            }
            let image =
                self.numeric_effect_image(source.numeric[numeric_var_id], op_id, numeric_var_id)?;
            if !image.intersects(&target.numeric[numeric_var_id]) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn numeric_effect_image(
        &self,
        source: Interval,
        op_id: usize,
        numeric_var_id: usize,
    ) -> Result<Interval> {
        let mut image = source;
        let op = &self.task.get_operators()[op_id];
        for effect in op
            .assignment_effects()
            .iter()
            .filter(|effect| effect.affected_var_id() == numeric_var_id)
        {
            let rhs = self.task.get_initial_numeric_state_values()[effect.var_id()];
            image.apply_op(effect.operation(), &Interval::singleton(rhs));
        }
        Ok(image)
    }

    fn numeric_effect_preimage(
        &self,
        target: Interval,
        op_id: usize,
        numeric_var_id: usize,
    ) -> Result<Interval> {
        let mut preimage = target;
        let op = &self.task.get_operators()[op_id];
        for effect in op
            .assignment_effects()
            .iter()
            .filter(|effect| effect.affected_var_id() == numeric_var_id)
            .rev()
        {
            let rhs = self.task.get_initial_numeric_state_values()[effect.var_id()];
            match effect.operation() {
                AssignmentOperation::Assign => {
                    preimage = if preimage.contains(rhs) {
                        Interval::unbounded()
                    } else {
                        bail!(
                            "operator {op_id} assignment image for numeric var {numeric_var_id} cannot intersect target {target:?}"
                        )
                    };
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
                        preimage = if preimage.contains(0.0) {
                            Interval::unbounded()
                        } else {
                            bail!(
                                "operator {op_id} zero multiplication image for numeric var {numeric_var_id} cannot intersect target {target:?}"
                            )
                        };
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
        }
        Ok(preimage)
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
        let start = Instant::now();
        let semantics = CartesianSemantics::new(task, self.config.random_seed)?;
        let mut working = WorkingAbstraction::new(semantics.trivial_region()?);
        for op_id in 0..task.get_operators().len() {
            if semantics.may_transition(&working.states[0], op_id, &working.states[0])? {
                working.add_transition(0, op_id, 0);
            }
        }

        let state_packer = Arc::new(make_prop_state_packer(task));
        let axiom_evaluator = AxiomEvaluator::new(Arc::new(task), state_packer.clone());
        let mut refinements = 0;

        let mut shortest_paths = compute_shortest_paths(&working, &semantics)?;
        let (stop_reason, pending_flaw, solved_plan) = loop {
            let check = replay_optimal_abstract_trace(
                &working,
                &semantics,
                &shortest_paths,
                &state_packer,
                &axiom_evaluator,
            )?;
            match check {
                PlanCheck::ConcretePlan(plan) => {
                    break (CartesianStopReason::ConcretePlan, None, Some(plan));
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
            validate_concrete_plan(&semantics, &state_packer, &axiom_evaluator, plan)?;
            let h = distance_table.distances[distance_table.initial_state_hash];
            ensure!(
                (plan.cost - h).abs() <= 1e-7,
                "concrete Cartesian plan cost {} differs from abstract h(init) {h}",
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
                stop_reason,
                pending_flaw,
                refinements,
                collection_goal_id: None,
                collection_variant_id: None,
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

    /// Builds the configured number of variants for every task goal, or one
    /// full-task abstraction when the goal is empty.
    ///
    /// Each member changes only the goal view. Operators, state mappings, and
    /// concrete operator IDs stay identical to the base task, which makes the
    /// resulting transition systems valid components for canonical and
    /// transition-cost-partitioned collection heuristics.
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
        ensure!(
            self.config.max_collection_states >= abstraction_count,
            "Cartesian max_collection_size {} cannot give at least one state to each of the {abstraction_count} abstractions",
            self.config.max_collection_states
        );

        let start = Instant::now();
        let mut remaining_states = self.config.max_collection_states;
        let mut abstractions = Vec::with_capacity(abstraction_count);
        for abstraction_id in 0..abstraction_count {
            let goal_id = abstraction_id / variants_per_goal;
            let variant_id = abstraction_id % variants_per_goal;
            let remaining_abstractions = abstraction_count - abstraction_id;
            let remaining_time = match self.config.total_max_time {
                Some(total_max_time) => {
                    let elapsed = start.elapsed();
                    ensure!(
                        elapsed < total_max_time,
                        "Cartesian collection total_max_time expired after {abstraction_id} of {abstraction_count} abstractions"
                    );
                    Some(total_max_time - elapsed)
                }
                None => None,
            };
            let mut abstraction_config = self.config.abstraction.clone();
            abstraction_config.max_states = abstraction_config
                .max_states
                .min(remaining_states - (remaining_abstractions - 1));
            abstraction_config.max_time = match (abstraction_config.max_time, remaining_time) {
                (Some(per_abstraction), Some(remaining)) => Some(per_abstraction.min(remaining)),
                (Some(per_abstraction), None) => Some(per_abstraction),
                (None, Some(remaining)) => Some(remaining),
                (None, None) => None,
            };
            if goal_count > 0 && self.config.variants_per_goal > 1 {
                abstraction_config.random_seed = if variant_id == 0 {
                    None
                } else {
                    Some(derive_variant_seed(
                        abstraction_config.random_seed.unwrap_or(0),
                        goal_id,
                        variant_id - 1,
                    ))
                };
            }

            let goal_task = (goal_count > 0)
                .then(|| SingleGoalTask::new(task, task.get_goal_fact(goal_id).clone()));
            let abstraction_task = goal_task
                .as_ref()
                .map_or(task, |goal_task| goal_task as &dyn AbstractNumericTask);
            info!(
                "Cartesian collection: building abstraction {}/{abstraction_count}, goal={}, variant={}, max_states={}, seed={:?}",
                abstraction_id + 1,
                goal_id,
                variant_id,
                abstraction_config.max_states,
                abstraction_config.random_seed
            );
            let mut abstraction = CartesianAbstractionGenerator::new(abstraction_config)?
                .generate(abstraction_task)
                .with_context(|| {
                    format!("failed to build Cartesian collection abstraction {abstraction_id}")
                })?;
            let state_count = abstraction.num_states();
            ensure!(
                state_count <= remaining_states,
                "Cartesian goal abstraction used {state_count} states with only {remaining_states} remaining"
            );
            remaining_states -= state_count;
            abstraction.metadata.collection_goal_id = (goal_count > 0).then_some(goal_id);
            abstraction.metadata.collection_variant_id = (goal_count > 0).then_some(variant_id);
            abstractions.push(abstraction);
        }

        info!(
            "Cartesian collection: abstractions={}, states={}, elapsed={:.3}s",
            abstractions.len(),
            self.config.max_collection_states - remaining_states,
            start.elapsed().as_secs_f64()
        );
        Ok(abstractions)
    }
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
    Refine(Split),
}

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

fn fact_refinement_count(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    fact: &ExplicitFact,
) -> Result<usize> {
    if let Some(tree_id) = semantics
        .comparison_tree_by_prop_var
        .get(fact.var())
        .copied()
        .flatten()
    {
        let tree = semantics
            .comparison_trees
            .get(tree_id)
            .with_context(|| format!("missing comparison tree {tree_id}"))?;
        return Ok(tree
            .regular_numeric_var_dependencies(semantics.task)
            .into_iter()
            .map(|var_id| working.numeric_refinement_counts[var_id])
            .max()
            .unwrap_or(0));
    }
    Ok(working
        .propositional_refinement_counts
        .get(fact.var())
        .copied()
        .with_context(|| {
            format!(
                "missing propositional refinement count for var {}",
                fact.var()
            )
        })?)
}

fn retain_min_growth_facts(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    facts: &mut Vec<&ExplicitFact>,
) -> Result<()> {
    let refinement_counts = facts
        .iter()
        .map(|fact| fact_refinement_count(working, semantics, fact))
        .collect::<Result<Vec<_>>>()?;
    let most_refined = refinement_counts
        .iter()
        .copied()
        .max()
        .context("cannot rank an empty flaw set by growth")?;
    let mut index = 0;
    facts.retain(|_| {
        let retain = refinement_counts[index] == most_refined;
        index += 1;
        retain
    });
    Ok(())
}

fn split_choice_key(split: &Split) -> u64 {
    match split {
        Split::Propositional { var_id, wanted, .. } => {
            let var_id = u64::try_from(*var_id).expect("split variable id does not fit u64");
            wanted
                .iter()
                .fold(var_id, |key, value| mix_seed(key ^ u64::from(*value)))
        }
        Split::Numeric {
            var_id,
            lower_includes_boundary,
            ..
        } => {
            let var_id = u64::try_from(*var_id).expect("split variable id does not fit u64");
            var_id ^ (u64::from(*lower_includes_boundary) << 63)
        }
    }
}

fn retain_min_growth_splits<T>(
    working: &WorkingAbstraction,
    candidates: &mut Vec<T>,
    split: impl Fn(&T) -> &Split,
) {
    let most_refined = candidates
        .iter()
        .map(|candidate| working.refinement_count(split(candidate)))
        .max()
        .expect("cannot rank an empty split candidate set by growth");
    candidates.retain(|candidate| working.refinement_count(split(candidate)) == most_refined);
}

fn replay_optimal_abstract_trace(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    shortest_paths: &ShortestPaths,
    state_packer: &Arc<IntDoublePacker>,
    axiom_evaluator: &AxiomEvaluator<'_>,
) -> Result<PlanCheck> {
    let (mut propositions, mut numeric) =
        get_initial_state(semantics.task, state_packer, axiom_evaluator)?;
    let mut prop_values = Vec::new();
    let mut successor_prop_values = Vec::new();
    semantics.concrete_prop_values(state_packer, &propositions, &mut prop_values);
    let initial_abstract_state = working.hierarchy.map_state(&prop_values, &numeric)?;
    ensure!(
        shortest_paths.distances[initial_abstract_state].is_finite(),
        "concrete initial state maps to abstract dead end {initial_abstract_state}"
    );
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
            let mut failed_goals = (0..semantics.task.get_num_goals())
                .map(|goal_id| semantics.task.get_goal_fact(goal_id))
                .filter(|goal| !fact_is_hold(goal, state_packer, &propositions))
                .collect::<Vec<_>>();
            ensure!(
                !failed_goals.is_empty(),
                "abstract goal contains a concrete non-goal without a failed goal fact"
            );
            retain_min_growth_facts(working, semantics, &mut failed_goals)?;
            let goal_keys = failed_goals
                .iter()
                .map(|goal| fact_choice_key(goal))
                .collect::<Vec<_>>();
            let failed_goal = failed_goals[semantics.choose_keyed_index(&goal_keys, 0x474F_414C)];
            return Ok(PlanCheck::Refine(split_failed_fact(
                working,
                semantics,
                abstract_state,
                failed_goal,
                &prop_values,
                &numeric,
                format!("goal {failed_goal:?}"),
            )?));
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
        let mut failed_preconditions = op
            .preconditions()
            .iter()
            .filter(|fact| !fact_is_hold(fact, state_packer, &propositions))
            .collect::<Vec<_>>();
        if !failed_preconditions.is_empty() {
            retain_min_growth_facts(working, semantics, &mut failed_preconditions)?;
            let precondition_keys = failed_preconditions
                .iter()
                .map(|precondition| fact_choice_key(precondition))
                .collect::<Vec<_>>();
            let failed =
                failed_preconditions[semantics.choose_keyed_index(&precondition_keys, 0x5052_4543)];
            return Ok(PlanCheck::Refine(split_failed_fact(
                working,
                semantics,
                abstract_state,
                failed,
                &prop_values,
                &numeric,
                format!("operator {op_id} ({}) precondition {failed:?}", op.name()),
            )?));
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
    plan: &ConcretePlan,
) -> Result<()> {
    let (mut propositions, mut numeric) =
        get_initial_state(semantics.task, state_packer, axiom_evaluator)?;
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
        let witness_value = *numeric_values
            .get(var_id)
            .with_context(|| format!("missing concrete numeric var {var_id}"))?;
        ensure!(
            witness_value.is_finite(),
            "comparison split witness for numeric var {var_id} is non-finite: {witness_value}"
        );
        let parent = *state
            .numeric
            .get(var_id)
            .with_context(|| format!("missing Cartesian numeric var {var_id}"))?;
        for lower_includes_boundary in [true, false] {
            if !parent.can_split_at(witness_value, lower_includes_boundary) {
                continue;
            }
            let lower = interval_intersection(
                parent,
                Interval::new(
                    f64::NEG_INFINITY,
                    witness_value,
                    false,
                    lower_includes_boundary,
                ),
            );
            let upper = interval_intersection(
                parent,
                Interval::new(
                    witness_value,
                    f64::INFINITY,
                    !lower_includes_boundary,
                    false,
                ),
            );
            let witness_child = if lower.contains(witness_value) {
                lower
            } else {
                ensure!(
                    upper.contains(witness_value),
                    "comparison split loses witness {witness_value} for numeric var {var_id}"
                );
                upper
            };
            let mut child_numeric = state.numeric.clone();
            child_numeric[var_id] = witness_child;
            let result = tree.evaluate_interval(&child_numeric);
            ensure!(
                result != Some(!concrete_truth),
                "comparison interval for tree {tree_id} excludes its concrete witness after splitting numeric var {var_id}"
            );
            let achieved = match goal {
                ComparisonRefinementGoal::ExcludeDesired(_) => result == Some(!desired_truth),
                ComparisonRefinementGoal::GuaranteeDesired(_) => result == Some(desired_truth),
            };
            let candidate = Split::Numeric {
                state_id,
                var_id,
                boundary: witness_value,
                lower_includes_boundary,
                witness_value,
                description: description.clone(),
            };
            candidates.push((achieved, candidate));
        }
    }
    ensure!(
        !candidates.is_empty(),
        "comparison tree {tree_id} has no strict regular-variable split in Cartesian state {state_id}"
    );
    retain_min_growth_splits(working, &mut candidates, |(_, split)| split);
    let has_achieving_candidate = candidates.iter().any(|(achieved, _)| *achieved);
    if has_achieving_candidate {
        candidates.retain(|(achieved, _)| *achieved);
    }
    let keys = candidates
        .iter()
        .map(|(_, split)| split_choice_key(split))
        .collect::<Vec<_>>();
    let index = semantics.choose_keyed_index(&keys, 0x434F_4D50);
    Ok(candidates.swap_remove(index).1)
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
        let preimage = semantics.numeric_effect_preimage(target_interval, op_id, var_id)?;
        let source = source_numeric[var_id];
        ensure!(
            !preimage.contains(source),
            "operator {op_id} concrete source {source} for numeric var {var_id} lies in the preimage {preimage:?} of target {target_interval:?}, but successor {successor} is outside"
        );
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
        ensure!(
            boundary.is_finite(),
            "cannot refine an infinite numeric-effect preimage boundary"
        );
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
        "concrete successor maps to a different Cartesian state but no differing component was found"
    );
    retain_min_growth_splits(working, &mut candidates, |split| split);
    let keys = candidates.iter().map(split_choice_key).collect::<Vec<_>>();
    let index = semantics.choose_keyed_index(&keys, 0x4445_5649);
    Ok(candidates.swap_remove(index))
}

fn apply_split(
    working: &mut WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    split: Split,
) -> Result<usize> {
    let old_state_id = split.state_id();
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
            wanted_region.propositions[var_id] = wanted_child_values;
            let mut other_region = old_region.clone();
            other_region.propositions[var_id] = other_child_values;
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
            lower_region.numeric[var_id] = lower;
            let mut upper_region = old_region.clone();
            upper_region.numeric[var_id] = upper;
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
                if semantics.may_transition(
                    &working.states[source],
                    transition.concrete_op_id,
                    &working.states[target],
                )? {
                    working.add_transition(source, transition.concrete_op_id, target);
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
    if combine_labels {
        raw.extend(grouped.into_iter().map(|((source, target), mut labels)| {
            labels.sort_unstable();
            labels.dedup();
            (source, target, labels)
        }));
    }
    raw.sort_by(|left, right| left.cmp(right));
    let mut transitions = Vec::with_capacity(raw.len());
    let mut forward = vec![Vec::new(); working.states.len()];
    let mut backward = vec![Vec::new(); working.states.len()];
    let mut footprints = if compute_operator_footprints {
        Vec::with_capacity(raw.len())
    } else {
        Vec::new()
    };
    let shared_state_regions = compute_operator_footprints.then(|| {
        working
            .states
            .iter()
            .cloned()
            .map(Arc::new)
            .collect::<Vec<_>>()
    });
    let mut relevant = HashSet::new();
    for (transition_id, (source, target, labels)) in raw.into_iter().enumerate() {
        for &label in &labels {
            relevant.insert(label);
        }
        if compute_operator_footprints {
            footprints.push(AbstractOperatorFootprint {
                labels: labels
                    .iter()
                    .copied()
                    .map(|concrete_op_id| ConcreteOperatorFootprint {
                        concrete_op_id,
                        source_region: Arc::clone(
                            &shared_state_regions
                                .as_ref()
                                .expect("Cartesian footprints require shared state regions")
                                [source],
                        ),
                        allocable: true,
                        max_allocation_fraction: 1.0,
                        non_allocable_reason: None,
                    })
                    .collect(),
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
    let initial_numeric = semantics.task.get_initial_numeric_state_values();
    let initial_state_hash = working
        .hierarchy
        .map_state(&initial_prop, &initial_numeric)?;
    let transition_system = AbstractTransitionSystem {
        transitions,
        duplicate_transition_attempts: 0,
        backward,
        forward,
        goal_facts: (0..semantics.task.get_num_goals())
            .map(|goal_id| semantics.task.get_goal_fact(goal_id).clone())
            .collect(),
        goal_state_hashes,
        initial_state_hash,
        hash_multipliers: Vec::new(),
        numeric_domain_sizes: Vec::new(),
        state_regions: working.states.clone(),
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
