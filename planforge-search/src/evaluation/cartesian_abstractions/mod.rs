//! Numeric Cartesian abstractions refined by concrete counterexamples.
//!
//! Unlike the factorized domain abstraction, splitting one Cartesian state
//! adds exactly one state. Every abstract transition is a may-transition of a
//! grounded concrete operator. CEGAR searches the complete optimal abstract
//! transition graph for concrete executions and refines witnessed flaws; only
//! a successfully replayed concrete plan may set `solved_by_self`.

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
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::int_packer::IntDoublePacker;
use tracing::{debug, info};

use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;

use super::abstraction_collections::transition_cost_partitioning::{
    AbstractOperatorFootprint, AbstractTransition, AbstractTransitionSystem,
    ConcreteOperatorFootprint, PropValueId, StateRegion,
};
use super::abstraction_task::validate_abstraction_operator;
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
}

#[derive(Debug, Clone)]
pub struct CartesianAbstractionMetadata {
    pub solved_by_self: bool,
    pub stop_reason: CartesianStopReason,
    pub pending_flaw: Option<String>,
    pub refinements: usize,
}

#[derive(Debug, Clone)]
pub struct CartesianAbstractionConfig {
    pub max_states: usize,
    pub max_time: Option<Duration>,
    pub combine_labels: bool,
    pub compute_operator_footprints: bool,
    pub debug: bool,
}

impl Default for CartesianAbstractionConfig {
    fn default() -> Self {
        Self {
            max_states: 10_000,
            max_time: None,
            combine_labels: false,
            compute_operator_footprints: true,
            debug: false,
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
    active: bool,
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
    transitions: Vec<WorkingTransition>,
    outgoing: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
    active_transition_keys: HashSet<TransitionKey>,
}

impl WorkingAbstraction {
    fn new(initial_region: StateRegion) -> Self {
        Self {
            states: vec![initial_region],
            leaf_node_ids: vec![0],
            hierarchy: CartesianRefinementHierarchy::trivial(),
            transitions: Vec::new(),
            outgoing: vec![Vec::new()],
            incoming: vec![Vec::new()],
            active_transition_keys: HashSet::new(),
        }
    }

    fn add_transition(&mut self, source: usize, op_id: usize, target: usize) {
        let key = TransitionKey {
            source,
            concrete_op_id: op_id,
            target,
        };
        if !self.active_transition_keys.insert(key) {
            return;
        }
        let transition_id = self.transitions.len();
        self.transitions.push(WorkingTransition {
            source,
            target,
            concrete_op_id: op_id,
            active: true,
        });
        self.outgoing[source].push(transition_id);
        self.incoming[target].push(transition_id);
    }

    fn deactivate_transition(&mut self, transition_id: usize) {
        let transition = &mut self.transitions[transition_id];
        if !transition.active {
            return;
        }
        transition.active = false;
        let removed = self.active_transition_keys.remove(&TransitionKey {
            source: transition.source,
            concrete_op_id: transition.concrete_op_id,
            target: transition.target,
        });
        assert!(removed, "active Cartesian transition key is missing");
    }

    fn active_transition_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.transitions
            .iter()
            .enumerate()
            .filter_map(|(id, transition)| transition.active.then_some(id))
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum SplitKey {
    Propositional {
        state_id: usize,
        var_id: usize,
        wanted: Vec<PropValueId>,
    },
    Numeric {
        state_id: usize,
        var_id: usize,
        boundary_bits: u64,
        lower_includes_boundary: bool,
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

    fn key(&self) -> SplitKey {
        match self {
            Self::Propositional {
                state_id,
                var_id,
                wanted,
                ..
            } => SplitKey::Propositional {
                state_id: *state_id,
                var_id: *var_id,
                wanted: wanted.clone(),
            },
            Self::Numeric {
                state_id,
                var_id,
                boundary,
                lower_includes_boundary,
                ..
            } => SplitKey::Numeric {
                state_id: *state_id,
                var_id: *var_id,
                boundary_bits: float_tolerance::canonical_bits(*boundary),
                lower_includes_boundary: *lower_includes_boundary,
            },
        }
    }
}

struct CartesianSemantics<'task> {
    task: &'task dyn AbstractNumericTask,
    comparison_tree_by_prop_var: Vec<Option<usize>>,
    comparison_trees: Vec<ComparisonTree>,
    propositional_axioms_by_prop_var: Vec<Vec<usize>>,
    operator_costs: Vec<f64>,
}

impl<'task> CartesianSemantics<'task> {
    fn new(task: &'task dyn AbstractNumericTask) -> Result<Self> {
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
        })
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
        let semantics = CartesianSemantics::new(task)?;
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
            let check = search_optimal_abstract_graph(
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
            finalize_abstraction(&working, &semantics, self.config.combine_labels)?;
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
            abstract_operator_footprints: if self.config.compute_operator_footprints {
                footprints
            } else {
                Vec::new()
            },
            metadata: CartesianAbstractionMetadata {
                solved_by_self: solved_plan.is_some(),
                stop_reason,
                pending_flaw,
                refinements,
            },
        })
    }
}

#[derive(Debug)]
struct ShortestPaths {
    distances: Vec<f64>,
    generating_transition: Vec<Option<usize>>,
    goal_states: Vec<usize>,
}

fn compute_shortest_paths(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
) -> Result<ShortestPaths> {
    let mut goal_states = Vec::new();
    for (state_id, region) in working.states.iter().enumerate() {
        if semantics.region_is_goal(region)? {
            goal_states.push(state_id);
        }
    }
    ensure!(
        !goal_states.is_empty(),
        "Cartesian abstraction has no abstract goal state"
    );
    let mut distances = vec![f64::INFINITY; working.states.len()];
    let mut generating_transition: Vec<Option<usize>> = vec![None; working.states.len()];
    let mut heap = BinaryHeap::new();
    for &goal_state in &goal_states {
        distances[goal_state] = 0.0;
        heap.push((Reverse(NotNan::new(0.0).unwrap()), goal_state));
    }
    while let Some((Reverse(distance), target)) = heap.pop() {
        let distance = distance.into_inner();
        if distance > distances[target] + EPSILON {
            continue;
        }
        for &transition_id in &working.incoming[target] {
            let transition = &working.transitions[transition_id];
            if !transition.active {
                continue;
            }
            let cost = semantics.operator_costs[transition.concrete_op_id];
            ensure!(
                cost >= -EPSILON && cost.is_finite(),
                "invalid operator cost {cost}"
            );
            let alternative = distance + cost.max(0.0);
            let source = transition.source;
            let improves = alternative + EPSILON < distances[source];
            let ties_better = (alternative - distances[source]).abs() <= EPSILON
                && generating_transition[source].is_none_or(|old_id| {
                    let old = &working.transitions[old_id];
                    (transition.concrete_op_id, transition.target, transition_id)
                        < (old.concrete_op_id, old.target, old_id)
                });
            if improves || ties_better {
                distances[source] = alternative;
                generating_transition[source] = Some(transition_id);
                heap.push((Reverse(NotNan::new(alternative).unwrap()), source));
            }
        }
    }
    Ok(ShortestPaths {
        distances,
        generating_transition,
        goal_states,
    })
}

fn update_shortest_paths_after_split(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    old: ShortestPaths,
    split_state_id: usize,
    new_state_id: usize,
) -> Result<ShortestPaths> {
    let old_num_states = old.distances.len();
    ensure!(
        new_state_id == old_num_states && working.states.len() == old_num_states + 1,
        "Cartesian incremental shortest-path update requires one appended split state"
    );

    let mut invalid = vec![false; working.states.len()];
    let mut queue = std::collections::VecDeque::new();
    let invalidate = |state_id: usize,
                      invalid: &mut Vec<bool>,
                      queue: &mut std::collections::VecDeque<usize>| {
        if !invalid[state_id] {
            invalid[state_id] = true;
            queue.push_back(state_id);
        }
    };
    invalidate(split_state_id, &mut invalid, &mut queue);
    invalidate(new_state_id, &mut invalid, &mut queue);

    let mut shortest_path_dependents = vec![Vec::new(); old_num_states];
    for source in 0..old_num_states {
        if let Some(transition_id) = old.generating_transition[source] {
            let transition = &working.transitions[transition_id];
            if !transition.active {
                invalidate(source, &mut invalid, &mut queue);
            }
            if transition.target < old_num_states {
                shortest_path_dependents[transition.target].push(source);
            }
        }
    }
    while let Some(target) = queue.pop_front() {
        if target >= old_num_states {
            continue;
        }
        for &source in &shortest_path_dependents[target] {
            invalidate(source, &mut invalid, &mut queue);
        }
    }

    let mut distances = old.distances;
    distances.push(distances[split_state_id]);
    let mut generating_transition = old.generating_transition;
    generating_transition.push(None);
    for state_id in 0..working.states.len() {
        if invalid[state_id] {
            distances[state_id] = f64::INFINITY;
            generating_transition[state_id] = None;
        }
    }

    let mut goal_states = Vec::new();
    let mut heap = BinaryHeap::new();
    for (state_id, region) in working.states.iter().enumerate() {
        if semantics.region_is_goal(region)? {
            goal_states.push(state_id);
            if invalid[state_id] {
                distances[state_id] = 0.0;
                heap.push((Reverse(NotNan::new(0.0).unwrap()), state_id));
            }
        }
    }

    for source in 0..working.states.len() {
        if !invalid[source] {
            continue;
        }
        for &transition_id in &working.outgoing[source] {
            let transition = &working.transitions[transition_id];
            if !transition.active || invalid[transition.target] {
                continue;
            }
            let target_distance = distances[transition.target];
            if !target_distance.is_finite() {
                continue;
            }
            let candidate = target_distance + semantics.operator_costs[transition.concrete_op_id];
            if candidate + EPSILON < distances[source] {
                distances[source] = candidate;
                generating_transition[source] = Some(transition_id);
                heap.push((Reverse(NotNan::new(candidate).unwrap()), source));
            }
        }
    }

    while let Some((Reverse(distance), target)) = heap.pop() {
        let distance = distance.into_inner();
        if distance > distances[target] + EPSILON {
            continue;
        }
        for &transition_id in &working.incoming[target] {
            let transition = &working.transitions[transition_id];
            if !transition.active || !invalid[transition.source] {
                continue;
            }
            let alternative = distance + semantics.operator_costs[transition.concrete_op_id];
            if alternative + EPSILON < distances[transition.source] {
                distances[transition.source] = alternative;
                generating_transition[transition.source] = Some(transition_id);
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
            let actual = distances[state_id];
            let expected = reference.distances[state_id];
            assert!(
                (actual == expected) || (actual - expected).abs() <= 1e-7,
                "incremental Cartesian distance mismatch at state {state_id}: {actual} vs {expected}"
            );
        }
    }

    Ok(ShortestPaths {
        distances,
        generating_transition,
        goal_states,
    })
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ConcreteStateKey {
    propositions: Vec<u64>,
    numeric_bits: Vec<u64>,
}

impl ConcreteStateKey {
    fn new(propositions: &[u64], numeric: &[f64]) -> Self {
        Self {
            propositions: propositions.to_vec(),
            numeric_bits: numeric
                .iter()
                .map(|value| float_tolerance::canonical_bits(*value))
                .collect(),
        }
    }
}

#[derive(Debug)]
struct ConcreteSearchNode {
    propositions: Vec<u64>,
    numeric: Vec<f64>,
    cost: f64,
    parent: Option<(usize, usize)>,
}

#[derive(Debug)]
struct SplitCandidate {
    split: Split,
    witness_nodes: HashSet<usize>,
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

fn add_split_candidate(
    candidates: &mut HashMap<SplitKey, SplitCandidate>,
    split: Split,
    witness_node: usize,
) {
    let key = split.key();
    let candidate = candidates.entry(key).or_insert_with(|| SplitCandidate {
        split,
        witness_nodes: HashSet::new(),
    });
    candidate.witness_nodes.insert(witness_node);
}

fn select_split_candidate(candidates: HashMap<SplitKey, SplitCandidate>) -> Result<Split> {
    let mut best: Option<(SplitKey, SplitCandidate)> = None;
    for (key, candidate) in candidates {
        let replace = best.as_ref().is_none_or(|(best_key, best_candidate)| {
            candidate.witness_nodes.len() > best_candidate.witness_nodes.len()
                || (candidate.witness_nodes.len() == best_candidate.witness_nodes.len()
                    && key < *best_key)
        });
        if replace {
            best = Some((key, candidate));
        }
    }
    best.map(|(_, candidate)| candidate.split)
        .context("optimal abstract graph has no concrete plan and no refinement flaw")
}

fn reconstruct_concrete_plan(nodes: &[ConcreteSearchNode], mut node_id: usize) -> ConcretePlan {
    let cost = nodes[node_id].cost;
    let mut operator_ids = Vec::new();
    while let Some((parent_id, op_id)) = nodes[node_id].parent {
        operator_ids.push(op_id);
        node_id = parent_id;
    }
    operator_ids.reverse();
    ConcretePlan { operator_ids, cost }
}

fn search_optimal_abstract_graph(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    shortest_paths: &ShortestPaths,
    state_packer: &Arc<IntDoublePacker>,
    axiom_evaluator: &AxiomEvaluator<'_>,
) -> Result<PlanCheck> {
    let (initial_propositions, initial_numeric) =
        get_initial_state(semantics.task, state_packer, axiom_evaluator)?;
    let mut prop_values = Vec::new();
    let mut successor_prop_values = Vec::new();
    semantics.concrete_prop_values(state_packer, &initial_propositions, &mut prop_values);
    let initial_abstract_state = working
        .hierarchy
        .map_state(&prop_values, &initial_numeric)?;
    ensure!(
        shortest_paths.distances[initial_abstract_state].is_finite(),
        "concrete initial state maps to abstract dead end {initial_abstract_state}"
    );
    let abstract_plan_cost = shortest_paths.distances[initial_abstract_state];
    let initial_key = ConcreteStateKey::new(&initial_propositions, &initial_numeric);
    let mut registry = HashMap::from([(initial_key, 0usize)]);
    let mut nodes = vec![ConcreteSearchNode {
        propositions: initial_propositions,
        numeric: initial_numeric,
        cost: 0.0,
        parent: None,
    }];
    let mut open = BinaryHeap::from([(Reverse(NotNan::new(0.0).unwrap()), 0usize)]);
    let mut candidates = HashMap::new();

    while let Some((Reverse(queue_cost), node_id)) = open.pop() {
        let node_cost = nodes[node_id].cost;
        if !approximately_equal(queue_cost.into_inner(), node_cost) {
            continue;
        }
        let node_propositions = nodes[node_id].propositions.clone();
        let node_numeric = nodes[node_id].numeric.clone();
        if concrete_is_goal(semantics, state_packer, &node_propositions) {
            ensure!(
                approximately_equal(node_cost, abstract_plan_cost),
                "concrete goal cost {} differs from optimal abstract cost {abstract_plan_cost}",
                node_cost
            );
            return Ok(PlanCheck::ConcretePlan(reconstruct_concrete_plan(
                &nodes, node_id,
            )));
        }

        semantics.concrete_prop_values(state_packer, &node_propositions, &mut prop_values);
        let abstract_state = working.hierarchy.map_state(&prop_values, &node_numeric)?;
        let abstract_distance = shortest_paths.distances[abstract_state];
        ensure!(
            approximately_equal(node_cost + abstract_distance, abstract_plan_cost),
            "concrete state reached outside optimal abstract graph: g={} h={abstract_distance} initial_h={abstract_plan_cost}",
            node_cost
        );

        if shortest_paths.goal_states.contains(&abstract_state) {
            for goal_id in 0..semantics.task.get_num_goals() {
                let goal = semantics.task.get_goal_fact(goal_id);
                if !fact_is_hold(goal, state_packer, &node_propositions) {
                    let split = split_failed_fact(
                        working,
                        semantics,
                        abstract_state,
                        goal,
                        &prop_values,
                        &node_numeric,
                        format!("goal {goal:?}"),
                    )?;
                    add_split_candidate(&mut candidates, split, node_id);
                }
            }
        }

        let mut optimal_targets_by_operator: HashMap<usize, Vec<usize>> = HashMap::new();
        for &transition_id in &working.outgoing[abstract_state] {
            let transition = &working.transitions[transition_id];
            if !transition.active {
                continue;
            }
            let target_distance = shortest_paths.distances[transition.target];
            let op_cost = semantics.operator_costs[transition.concrete_op_id];
            if approximately_equal(op_cost + target_distance, abstract_distance) {
                optimal_targets_by_operator
                    .entry(transition.concrete_op_id)
                    .or_default()
                    .push(transition.target);
            }
        }
        for (op_id, mut expected_targets) in optimal_targets_by_operator {
            expected_targets.sort_unstable();
            expected_targets.dedup();
            let op = &semantics.task.get_operators()[op_id];
            let failed_preconditions: Vec<_> = op
                .preconditions()
                .iter()
                .filter(|fact| !fact_is_hold(fact, state_packer, &node_propositions))
                .collect();
            if !failed_preconditions.is_empty() {
                for failed in failed_preconditions {
                    let split = split_failed_fact(
                        working,
                        semantics,
                        abstract_state,
                        failed,
                        &prop_values,
                        &node_numeric,
                        format!("operator {op_id} ({}) precondition {failed:?}", op.name()),
                    )?;
                    add_split_candidate(&mut candidates, split, node_id);
                }
                continue;
            }

            let mut successor_propositions = node_propositions.clone();
            let mut successor_numeric = node_numeric.clone();
            progress(
                op,
                axiom_evaluator,
                state_packer,
                &mut successor_propositions,
                &mut successor_numeric,
            )?;
            semantics.concrete_prop_values(
                state_packer,
                &successor_propositions,
                &mut successor_prop_values,
            );
            let concrete_target = working
                .hierarchy
                .map_state(&successor_prop_values, &successor_numeric)?;
            for &expected_target in &expected_targets {
                if expected_target != concrete_target {
                    let split = split_deviation(
                        working,
                        semantics,
                        abstract_state,
                        expected_target,
                        op_id,
                        &successor_prop_values,
                        &node_numeric,
                        &successor_numeric,
                    )?;
                    add_split_candidate(&mut candidates, split, node_id);
                }
            }
            if expected_targets.binary_search(&concrete_target).is_err() {
                continue;
            }

            let successor_cost = node_cost + semantics.operator_costs[op_id];
            let successor_key = ConcreteStateKey::new(&successor_propositions, &successor_numeric);
            if registry.contains_key(&successor_key) {
                continue;
            }
            let successor_id = nodes.len();
            registry.insert(successor_key, successor_id);
            nodes.push(ConcreteSearchNode {
                propositions: successor_propositions,
                numeric: successor_numeric,
                cost: successor_cost,
                parent: Some((node_id, op_id)),
            });
            open.push((
                Reverse(NotNan::new(successor_cost).context("non-finite concrete path cost")?),
                successor_id,
            ));
        }
    }

    Ok(PlanCheck::Refine(select_split_candidate(candidates)?))
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
    let mut best: Option<(bool, Split)> = None;
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
            if best
                .as_ref()
                .is_none_or(|(best_achieved, _)| achieved && !best_achieved)
            {
                best = Some((achieved, candidate));
            }
        }
    }
    best.map(|(_, split)| split).with_context(|| {
        format!(
            "comparison tree {tree_id} has no strict regular-variable split in Cartesian state {state_id}"
        )
    })
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
            return Ok(Split::Propositional {
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
        return Ok(Split::Numeric {
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
    bail!(
        "concrete successor maps to a different Cartesian state but no differing component was found"
    )
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

    let mut incident = working.outgoing[old_state_id].clone();
    incident.extend(working.incoming[old_state_id].iter().copied());
    incident.sort_unstable();
    incident.dedup();
    let old_transitions: Vec<_> = incident
        .iter()
        .filter_map(|&id| {
            working.transitions[id]
                .active
                .then(|| working.transitions[id].clone())
        })
        .collect();
    for transition_id in incident {
        working.deactivate_transition(transition_id);
    }
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
) -> Result<(
    AbstractTransitionSystem,
    AbstractDistanceTable,
    Vec<usize>,
    Vec<AbstractOperatorFootprint>,
)> {
    let mut grouped: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut raw = Vec::new();
    for transition_id in working.active_transition_ids() {
        let transition = &working.transitions[transition_id];
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
    let mut footprints = Vec::with_capacity(raw.len());
    let mut relevant = HashSet::new();
    for (transition_id, (source, target, labels)) in raw.into_iter().enumerate() {
        for &label in &labels {
            relevant.insert(label);
        }
        transitions.push(AbstractTransition {
            transition_id,
            abstract_op_id: transition_id,
            concrete_op_ids: labels.clone(),
            source_hash: source,
            target_hash: target,
        });
        forward[source].push(transition_id);
        backward[target].push(transition_id);
        footprints.push(AbstractOperatorFootprint {
            labels: labels
                .into_iter()
                .map(|concrete_op_id| ConcreteOperatorFootprint {
                    concrete_op_id,
                    source_region: working.states[source].clone(),
                    allocable: true,
                    max_allocation_fraction: 1.0,
                    non_allocable_reason: None,
                })
                .collect(),
        });
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
