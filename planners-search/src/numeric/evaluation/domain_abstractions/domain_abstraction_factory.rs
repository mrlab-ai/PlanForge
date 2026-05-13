#[cfg(test)]
mod tests;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};
use ordered_float::NotNan;
use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::SmallRng};

use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentOperation, ExplicitFact, NumericType, Operator,
    metric_operator_cost_from_initial_values,
};
use planners_sas::numeric::utils::float_tolerance;

use super::abstract_operator_generator::{
    AbstractOperator, AbstractOperatorGenerator, DomainMapping,
};
use super::comparison_expression::{ComparisonTree, Interval};
use super::domain_abstraction::{ComparisonAxiomIndex, NumericPartitions};
use super::numeric_context::{
    prepare_comparison_tree_inputs_from_abstract_state,
    prepare_comparison_tree_inputs_from_abstract_state_into,
};
use super::transition_cost_partitioning::{
    AbstractOperatorCostBudget, AbstractOperatorCostFunction, AbstractOperatorFootprint,
    AbstractTransition, AbstractTransitionCostFunction, AbstractTransitionSystem,
    ConcreteOperatorFootprint, FiniteSupportConfig, NonAllocableFootprintReason, StateRegion,
    TransitionResidualCosts,
};
use super::utils;

const COMPARISON_TRUE_VAL: usize = 0;
const COMPARISON_FALSE_VAL: usize = 1;
const COMPARISON_UNKNOWN_VAL: usize = 2;
// The comparison-enumeration cache short-circuits the per-state walk over
// the comparison-tree forest in the regression Dijkstra. Each cached entry
// is a `Vec<usize>` of resolved successor hashes, typically only a handful
// of entries. For a 1M-state abstraction the cache may peak around tens of
// MB which is acceptable for the build phase.
const COMPARISON_ENUMERATION_CACHE_MAX_ENTRIES: usize = 2_000_000;
const COMPARISON_ENUMERATION_CACHE_MAX_STATES: usize = 10_000_000;

/// Cache used by `enumerate_states_with_evaluated_comparisons_cached`. Keyed
/// by a precomputed 64-bit signature of `(base_state_hash, fixed_comparisons)`
/// — `comparison_var_ids` is intentionally omitted because every call site
/// in this factory passes the same slice (`self.comparison_var_ids()`), so it
/// doesn't disambiguate.
///
/// The previous design hashed a `(usize, Vec<usize>, Vec<(usize, usize)>)`
/// struct via SipHash on every cache lookup; on a 200k-state minecraft build
/// `sip::Hasher::write` reached 11% of total CPU and the per-lookup
/// `Vec::to_vec`/`collect` allocations dominated `_int_malloc`. The new
/// table is a `HashMap<u64, Vec<usize>>` with an identity hasher: lookup is
/// a single load + probe, no allocation, no hash function.
/// Cache value for `enumerate_states_with_evaluated_comparisons_cached`.
/// `states` is the enumerated predecessor set; `representative` is the
/// largest entry, precomputed at insertion so the Dijkstra inner loop
/// doesn't recompute it on every heap relaxation.
#[derive(Clone)]
struct CachedEnumeration {
    states: Arc<[usize]>,
    representative: Option<usize>,
}

type ComparisonEnumerationCache = HashMap<
    u64,
    CachedEnumeration,
    std::hash::BuildHasherDefault<planners_sas::numeric::state_registry::IdentityU64Hasher>,
>;

#[inline]
fn comparison_enumeration_signature(
    base_state_hash: usize,
    fixed_comparisons: &[ExplicitFact],
) -> u64 {
    // FNV-1a u64 mix + a SplitMix64 finalizer for even bit distribution
    // (same construction as `compute_signature_hash` for abstract operators).
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;

    let mut h = FNV_OFFSET;
    h ^= base_state_hash as u64;
    h = h.wrapping_mul(FNV_PRIME);
    h ^= fixed_comparisons.len() as u64;
    h = h.wrapping_mul(FNV_PRIME);
    for fact in fixed_comparisons {
        h ^= fact.var as u64;
        h = h.wrapping_mul(FNV_PRIME);
        h ^= fact.value as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }

    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    h
}

fn current_time_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0x9e37_79b9_7f4a_7c15)
}

fn ensure_online_scp_deadline(deadline: Option<Instant>) -> Result<()> {
    ensure!(
        deadline.is_none_or(|deadline| Instant::now() < deadline),
        "online SCP deadline exceeded"
    );
    Ok(())
}

#[derive(Debug, Clone, Default)]
struct MatchTreeNode {
    value_children: Vec<Option<Box<MatchTreeNode>>>,
    wildcard_child: Option<Box<MatchTreeNode>>,
    ops: Vec<usize>,
}

#[derive(Debug, Clone)]
struct MatchTree {
    var_order: Vec<usize>,
    domain_sizes: Vec<usize>,
    numeric_domain_sizes: Vec<usize>,
    hash_multipliers: Vec<usize>,
    root: MatchTreeNode,
}

#[derive(Debug, Clone, Default)]
struct OperatorComparisonSemantics {
    affected_comparison_var_ids: Vec<usize>,
}

fn domain_size_for_var(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    var: usize,
) -> usize {
    if var < domain_sizes.len() {
        domain_sizes.get(var).copied().unwrap_or(0)
    } else {
        numeric_domain_sizes
            .get(var - domain_sizes.len())
            .copied()
            .unwrap_or(0)
    }
}

fn fact_value_for_var(facts: &[ExplicitFact], var: usize) -> Option<usize> {
    facts
        .binary_search_by_key(&var, |fact| fact.var)
        .ok()
        .map(|index| facts[index].value)
}

impl MatchTree {
    fn build(
        domain_sizes: &[usize],
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        operators: &[AbstractOperator],
        _comparison_var_ids: &[usize],
    ) -> Self {
        let total_vars = domain_sizes.len() + numeric_domain_sizes.len();
        let mut var_counts = vec![0usize; total_vars];
        for op in operators {
            for f in op.regression_preconditions.iter() {
                if f.var >= total_vars {
                    continue;
                }
                let domain_size = domain_size_for_var(domain_sizes, numeric_domain_sizes, f.var);
                if domain_size > 1 {
                    var_counts[f.var] += 1;
                }
            }
        }
        let mut var_order: Vec<usize> = var_counts
            .iter()
            .enumerate()
            .filter_map(|(var, &count)| (count > 0).then_some(var))
            .collect();
        var_order.sort_by(|&left, &right| {
            var_counts[right]
                .cmp(&var_counts[left])
                .then_with(|| left.cmp(&right))
        });

        let mut tree = Self {
            var_order,
            domain_sizes: domain_sizes.to_vec(),
            numeric_domain_sizes: numeric_domain_sizes.to_vec(),
            hash_multipliers: hash_multipliers.to_vec(),
            root: MatchTreeNode::default(),
        };

        for (op_id, op) in operators.iter().enumerate() {
            tree.insert(op_id, &op.regression_preconditions);
        }

        tree
    }

    fn insert(&mut self, op_id: usize, conds: &[ExplicitFact]) {
        fn insert_rec(
            node: &mut MatchTreeNode,
            depth: usize,
            var_order: &[usize],
            conds: &[ExplicitFact],
            domain_sizes: &[usize],
            numeric_domain_sizes: &[usize],
            op_id: usize,
        ) {
            if depth == var_order.len() {
                node.ops.push(op_id);
                return;
            }
            let var = var_order[depth];
            if let Some(val) = fact_value_for_var(conds, var) {
                let domain_size = domain_size_for_var(domain_sizes, numeric_domain_sizes, var);
                if node.value_children.len() < domain_size {
                    node.value_children.resize_with(domain_size, || None);
                }
                let child = node.value_children[val]
                    .get_or_insert_with(|| Box::new(MatchTreeNode::default()));
                insert_rec(
                    child.as_mut(),
                    depth + 1,
                    var_order,
                    conds,
                    domain_sizes,
                    numeric_domain_sizes,
                    op_id,
                );
            } else {
                let child = node
                    .wildcard_child
                    .get_or_insert_with(|| Box::new(MatchTreeNode::default()));
                insert_rec(
                    child.as_mut(),
                    depth + 1,
                    var_order,
                    conds,
                    domain_sizes,
                    numeric_domain_sizes,
                    op_id,
                );
            }
        }

        insert_rec(
            &mut self.root,
            0,
            &self.var_order,
            conds,
            &self.domain_sizes,
            &self.numeric_domain_sizes,
            op_id,
        );
    }

    fn get_applicable_operator_ids(&self, state_hash: usize, out: &mut Vec<usize>) {
        out.clear();
        self.collect_applicable(&self.root, 0, state_hash, out);
    }

    fn collect_applicable(
        &self,
        node: &MatchTreeNode,
        depth: usize,
        state_hash: usize,
        out: &mut Vec<usize>,
    ) {
        if depth == self.var_order.len() {
            out.extend_from_slice(&node.ops);
            return;
        }
        let var = self.var_order[depth];
        let actual = self.get_var_value(state_hash, var);
        if let Some(child) = node.value_children.get(actual).and_then(Option::as_deref) {
            self.collect_applicable(child, depth + 1, state_hash, out);
        }
        if let Some(child) = node.wildcard_child.as_deref() {
            self.collect_applicable(child, depth + 1, state_hash, out);
        }
    }
    fn get_var_value(&self, state_hash: usize, var: usize) -> usize {
        let num_props = self.domain_sizes.len();
        debug_assert!(
            var < self.hash_multipliers.len(),
            "match tree var out of bounds for hash multipliers: {} >= {}",
            var,
            self.hash_multipliers.len()
        );
        let Some(mult) = self.hash_multipliers.get(var).copied() else {
            return 0;
        };
        let state = state_hash;
        let dom_size = if var < num_props {
            debug_assert!(
                var < self.domain_sizes.len(),
                "match tree propositional var out of bounds: {} >= {}",
                var,
                self.domain_sizes.len()
            );
            self.domain_sizes.get(var).copied().unwrap_or(0)
        } else {
            let n = var - num_props;
            debug_assert!(
                n < self.numeric_domain_sizes.len(),
                "match tree numeric var out of bounds: {} >= {}",
                n,
                self.numeric_domain_sizes.len()
            );
            self.numeric_domain_sizes.get(n).copied().unwrap_or(0)
        };
        debug_assert!(
            dom_size > 0,
            "match tree domain size must be positive for var {var}"
        );
        if dom_size == 0 {
            return 0;
        }

        (state / mult) % dom_size
    }
}

#[derive(Debug, Clone)]
pub struct AbstractDistanceTable {
    pub distances: Vec<f64>,
    // Per-state operator leading to a goal along a shortest path.
    pub generating_op_ids: Vec<Option<usize>>,
    pub initial_state_hash: usize,
    pub goal_facts: Vec<ExplicitFact>,
    pub hash_multipliers: Vec<usize>,
    pub numeric_domain_sizes: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct WildcardPlanResult {
    // Per-step set of concrete operator IDs.
    pub wildcard_plan: Vec<Vec<usize>>,
    // Path of abstract state hashes (`len = steps+1`).
    pub abstract_state_hashes: Vec<usize>,
    // Decoded propositional values along path.
    pub abstract_prop_states: Vec<Vec<usize>>,
    // Decoded numeric partitions along path.
    pub abstract_numeric_states: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct DomainAbstractionFactory {
    pub domain_mapping: DomainMapping,
    pub domain_sizes: Vec<usize>,
    pub partitions: NumericPartitions,
    pub numeric_domain_sizes: Vec<usize>,
    comparison_index: Option<ComparisonAxiomIndex>,
    comparison_trees: Vec<ComparisonTree>,
    /// Per-concrete-operator metric cost, evaluated once over the initial
    /// numeric state. The cost is task-deterministic, so caching here (and
    /// sharing the `Arc` into every per-iteration `AbstractOperatorGenerator`)
    /// avoids the `task.get_operators() × assignment_effects` scan that
    /// `metric_operator_cost_from_initial_values` does on every call.
    cached_operator_costs: Arc<[f64]>,
}

impl DomainAbstractionFactory {
    pub fn new(
        task: &dyn AbstractNumericTask,
        domain_mapping: DomainMapping,
        domain_sizes: Vec<usize>,
        partitions: NumericPartitions,
        numeric_domain_sizes: Vec<usize>,
    ) -> Result<Self> {
        ensure!(
            domain_mapping.len() == domain_sizes.len(),
            "domain_mapping/domain_sizes length mismatch"
        );
        for (var, &abs_size) in domain_sizes.iter().enumerate() {
            ensure!(
                abs_size > 0,
                "non-positive abstract domain size for var {var}: {abs_size}"
            );

            let concrete_size = task
                .get_variable_domain_size(var)
                .map_err(|e| anyhow!(e.to_string()))
                .with_context(|| format!("get_variable_domain_size({var}) failed"))?;
            ensure!(
                concrete_size > 0,
                "non-positive concrete domain size for var {var}: {concrete_size}"
            );
            ensure!(
                abs_size <= concrete_size,
                "abstract domain size for var {var} exceeds concrete size ({abs_size} > {concrete_size})"
            );

            ensure!(
                domain_mapping[var].len() == concrete_size,
                "domain_mapping[{var}] has len {}, expected concrete size {concrete_size}",
                domain_mapping[var].len()
            );

            for (val, &mapped) in domain_mapping[var].iter().enumerate() {
                ensure!(
                    mapped < abs_size,
                    "domain_mapping[{var}][{val}]={mapped} out of range for abstract size {abs_size}"
                );
            }
        }
        for (n, &parts) in numeric_domain_sizes.iter().enumerate() {
            ensure!(parts > 0, "numeric_domain_sizes[{n}] must be > 0");
            let actual = partitions.partitions(n).map(|p| p.len()).unwrap_or(0);
            ensure!(
                actual == parts,
                "numeric_domain_sizes[{n}]={parts} does not match partitions len {actual}"
            );
        }

        let comparison_index = if task.comparison_axioms().is_empty() {
            None
        } else {
            Some(
                ComparisonAxiomIndex::from_task(task)
                    .map_err(|e| anyhow!(e))
                    .context("failed to build ComparisonAxiomIndex")?,
            )
        };

        let mut comparison_trees: Vec<ComparisonTree> =
            Vec::with_capacity(task.comparison_axioms().len());
        for comparison_axiom_id in 0..task.comparison_axioms().len() {
            let tree = ComparisonTree::from_task(task, comparison_axiom_id).map_err(|e| {
                anyhow!(
                    "failed to build ComparisonTree for comparison axiom {comparison_axiom_id}: {e:?}"
                )
            })?;
            comparison_trees.push(tree);
        }

        let cached_operator_costs: Arc<[f64]> = task
            .get_operators()
            .iter()
            .map(|op| metric_operator_cost_from_initial_values(task, op))
            .collect();
        Ok(Self {
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            comparison_index,
            comparison_trees,
            cached_operator_costs,
        })
    }

    pub fn cached_operator_costs(&self) -> &Arc<[f64]> {
        &self.cached_operator_costs
    }

    pub fn partitions(&self) -> &NumericPartitions {
        &self.partitions
    }

    pub fn domain_mapping(&self) -> &DomainMapping {
        &self.domain_mapping
    }

    pub fn domain_sizes(&self) -> &[usize] {
        &self.domain_sizes
    }

    pub fn numeric_domain_sizes(&self) -> &[usize] {
        &self.numeric_domain_sizes
    }

    pub fn comparison_index(&self) -> Option<&ComparisonAxiomIndex> {
        self.comparison_index.as_ref()
    }

    pub fn comparison_trees(&self) -> &[ComparisonTree] {
        &self.comparison_trees
    }

    pub fn make_operator_generator(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
    ) -> Result<AbstractOperatorGenerator> {
        AbstractOperatorGenerator::new_with_cached_costs(
            task,
            self.domain_mapping.clone(),
            self.domain_sizes.clone(),
            self.partitions.clone(),
            self.numeric_domain_sizes.clone(),
            combine_labels,
            Arc::clone(&self.cached_operator_costs),
        )
    }

    /// Runs numeric-fd style implicit regression Dijkstra and returns distances-to-goal for
    /// all abstract states plus the generating operator per state.
    pub fn build_abstract_distance_table(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        dump_distances: bool,
    ) -> Result<AbstractDistanceTable> {
        let mut generator = self.make_operator_generator(task, combine_labels)?;
        let operators = generator.build_abstract_operators(task)?;
        self.build_distance_table_with_operators(task, &generator, &operators, dump_distances)
    }

    /// Builds an abstract distance table using the supplied per-concrete-operator costs and
    /// returns the saturated costs induced by the resulting distances.
    pub fn build_cost_partitioned_distance_table(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operator_costs: &[f64],
        dump_distances: bool,
    ) -> Result<(AbstractDistanceTable, Vec<f64>)> {
        let goal_facts = self.compute_abstract_goals(task);
        self.build_cost_partitioned_distance_table_for_goals(
            task,
            combine_labels,
            operator_costs,
            dump_distances,
            &goal_facts,
        )
    }

    pub fn build_cost_partitioned_distance_table_for_goals(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operator_costs: &[f64],
        dump_distances: bool,
        goal_facts: &[ExplicitFact],
    ) -> Result<(AbstractDistanceTable, Vec<f64>)> {
        let mut generator = self.make_operator_generator(task, combine_labels)?;
        let mut operators = generator.build_abstract_operators(task)?;
        apply_operator_costs(&mut operators, operator_costs)?;
        let table = self.build_distance_table_with_operators_for_goals(
            task,
            &generator,
            &operators,
            dump_distances,
            goal_facts,
        )?;
        let saturated_costs = self.compute_saturated_costs(task, &generator, &operators, &table)?;
        Ok((table, saturated_costs))
    }

    /// Builds goal distances using the supplied operator costs, without computing
    /// saturated costs. Used by the order generator during diversification.
    pub fn build_goal_distances(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operator_costs: &[f64],
    ) -> Result<AbstractDistanceTable> {
        let goal_facts = self.compute_abstract_goals(task);
        self.build_goal_distances_for_goals(task, combine_labels, operator_costs, &goal_facts)
    }

    pub fn build_goal_distances_for_goals(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operator_costs: &[f64],
        goal_facts: &[ExplicitFact],
    ) -> Result<AbstractDistanceTable> {
        let mut generator = self.make_operator_generator(task, combine_labels)?;
        let mut operators = generator.build_abstract_operators(task)?;
        apply_operator_costs(&mut operators, operator_costs)?;
        self.build_distance_table_with_operators_for_goals_inner(
            task, &generator, &operators, false, goal_facts, None,
        )
    }

    /// Computes saturated costs for the *already-built* distance table and
    /// abstract operators.  This is public so the online SCP heuristic can
    /// cap h-values for PERIM saturation before computing saturated costs.
    pub fn saturated_costs_for_table(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operators: &[AbstractOperator],
        table: &AbstractDistanceTable,
    ) -> Result<Vec<f64>> {
        let generator = self.make_operator_generator(task, combine_labels)?;
        self.compute_saturated_costs(task, &generator, operators, table)
    }

    pub fn build_abstract_transition_system(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
    ) -> Result<AbstractTransitionSystem> {
        self.build_abstract_transition_system_with_deadline(task, combine_labels, None)
    }

    pub fn build_abstract_transition_system_with_deadline(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        deadline: Option<Instant>,
    ) -> Result<AbstractTransitionSystem> {
        let mut generator = self.make_operator_generator(task, combine_labels)?;
        let operators = generator.build_abstract_operators(task)?;
        self.build_transition_system_with_operators(task, &generator, &operators, deadline, true)
    }

    pub fn build_abstract_transition_system_from_operators_with_deadline(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operators: &[AbstractOperator],
        deadline: Option<Instant>,
    ) -> Result<AbstractTransitionSystem> {
        let generator = self.make_operator_generator(task, combine_labels)?;
        self.build_transition_system_with_operators(task, &generator, operators, deadline, true)
    }

    pub fn build_abstract_transition_system_from_operators_without_regions_with_deadline(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operators: &[AbstractOperator],
        deadline: Option<Instant>,
    ) -> Result<AbstractTransitionSystem> {
        let generator = self.make_operator_generator(task, combine_labels)?;
        self.build_transition_system_with_operators(task, &generator, operators, deadline, false)
    }

    pub fn relevant_operator_ids_from_operators_with_deadline(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operators: &[AbstractOperator],
        deadline: Option<Instant>,
    ) -> Result<Vec<usize>> {
        let generator = self.make_operator_generator(task, combine_labels)?;
        self.relevant_operator_ids_with_operators(task, &generator, operators, deadline)
    }

    pub fn build_transition_cost_partitioned_distance_table(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        residual_costs: &TransitionResidualCosts,
        abstraction_id: usize,
        cap_state_id: Option<usize>,
    ) -> Result<(
        AbstractDistanceTable,
        AbstractTransitionCostFunction,
        AbstractTransitionSystem,
    )> {
        let transition_system = self.build_abstract_transition_system(task, combine_labels)?;
        let (table, tcf) = self.build_transition_cost_partitioned_distance_table_from_system(
            &transition_system,
            residual_costs,
            abstraction_id,
            cap_state_id,
        )?;
        Ok((table, tcf, transition_system))
    }

    pub fn build_transition_cost_partitioned_distance_table_from_system(
        &self,
        transition_system: &AbstractTransitionSystem,
        residual_costs: &TransitionResidualCosts,
        abstraction_id: usize,
        cap_state_id: Option<usize>,
    ) -> Result<(AbstractDistanceTable, AbstractTransitionCostFunction)> {
        self.build_transition_cost_partitioned_distance_table_from_system_with_deadline(
            transition_system,
            residual_costs,
            abstraction_id,
            cap_state_id,
            None,
        )
    }

    pub fn build_transition_cost_partitioned_distance_table_from_system_with_deadline(
        &self,
        transition_system: &AbstractTransitionSystem,
        residual_costs: &TransitionResidualCosts,
        abstraction_id: usize,
        cap_state_id: Option<usize>,
        deadline: Option<Instant>,
    ) -> Result<(AbstractDistanceTable, AbstractTransitionCostFunction)> {
        ensure_online_scp_deadline(deadline)?;
        let residuals_have_reductions = residual_costs.has_reductions();
        let transition_costs = transition_system
            .transitions
            .iter()
            .enumerate()
            .map(|(index, transition)| {
                if index % 1024 == 0 {
                    ensure_online_scp_deadline(deadline)?;
                }
                Ok(transition
                    .concrete_op_ids
                    .iter()
                    .map(|&concrete_op_id| {
                        if residuals_have_reductions {
                            residual_costs.cost_for_indexed_transition(
                                concrete_op_id,
                                abstraction_id,
                                transition.source_hash,
                                transition.abstract_op_id,
                                transition.target_hash,
                                &transition_system.state_regions[transition.source_hash],
                                &transition_system.state_regions[transition.target_hash],
                            )
                        } else {
                            residual_costs.base_cost(concrete_op_id)
                        }
                    })
                    .fold(f64::INFINITY, f64::min))
            })
            .collect::<Result<Vec<_>>>()?;
        let table = self.build_distance_table_with_transition_costs(
            transition_system,
            &transition_costs,
            &transition_system.hash_multipliers,
            &transition_system.numeric_domain_sizes,
        )?;

        if let Some(state_id) = cap_state_id
            && let Some(&h_cap) = table.distances.get(state_id)
            && h_cap.is_finite()
        {
            let mut perim_table = table.clone();
            for h in &mut perim_table.distances {
                if !h.is_finite() || *h > h_cap {
                    *h = f64::NEG_INFINITY;
                }
            }
            let tcf = self.compute_saturated_transition_costs(
                transition_system,
                &transition_costs,
                &perim_table,
            )?;
            let global_table = self.build_distance_table_with_transition_costs(
                transition_system,
                &tcf.transition_costs,
                &transition_system.hash_multipliers,
                &transition_system.numeric_domain_sizes,
            )?;
            return Ok((global_table, tcf));
        }

        let tcf =
            self.compute_saturated_transition_costs(transition_system, &transition_costs, &table)?;
        Ok((table, tcf))
    }

    pub fn build_abstract_operator_cost_partitioned_distance_table_from_system_with_deadline(
        &self,
        transition_system: &AbstractTransitionSystem,
        residual_costs: &TransitionResidualCosts,
        abstraction_id: usize,
        cap_state_id: Option<usize>,
        deadline: Option<Instant>,
    ) -> Result<(AbstractDistanceTable, AbstractOperatorCostFunction)> {
        self.build_abstract_operator_cost_partitioned_distance_table_from_system_and_footprints_with_deadline(
            transition_system,
            None,
            residual_costs,
            abstraction_id,
            cap_state_id,
            deadline,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_abstract_operator_cost_partitioned_distance_table_with_operators_and_footprints_with_deadline(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operators: &[AbstractOperator],
        footprints: &[AbstractOperatorFootprint],
        budgets: Option<&[AbstractOperatorCostBudget]>,
        label_rescue_operator_ids: Option<&HashSet<usize>>,
        residual_costs: &TransitionResidualCosts,
        abstraction_id: usize,
        current_state_id: Option<usize>,
        cap_state_id: Option<usize>,
        deadline: Option<Instant>,
    ) -> Result<(AbstractDistanceTable, AbstractOperatorCostFunction)> {
        ensure_online_scp_deadline(deadline)?;
        ensure!(
            footprints.len() >= operators.len(),
            "abstract-operator footprint/operator size mismatch: {} vs {}",
            footprints.len(),
            operators.len()
        );

        let operator_costs = abstract_operator_costs_from_footprints(
            operators.len(),
            footprints,
            budgets,
            label_rescue_operator_ids,
            residual_costs,
            abstraction_id,
            deadline,
        )?;
        let mut operators = operators.to_vec();
        apply_abstract_operator_costs(&mut operators, &operator_costs)?;
        let generator = self.make_operator_generator(task, combine_labels)?;
        if operator_costs.iter().all(|&cost| cost <= 1e-12) {
            let table = self.zero_distance_table_for_generator(task, &generator)?;
            let tcf = AbstractOperatorCostFunction {
                operator_costs: vec![0.0; operator_costs.len()],
            };
            return Ok((table, tcf));
        }
        if residual_costs.has_reductions()
            && let Some(current_state_id) = current_state_id
        {
            let current_distance = self.compute_distance_to_goal_state_with_operators(
                task,
                &generator,
                &operators,
                current_state_id,
                deadline,
            )?;
            if current_distance <= 1e-12 {
                let table = self.zero_distance_table_for_generator(task, &generator)?;
                let tcf = AbstractOperatorCostFunction {
                    operator_costs: vec![0.0; operator_costs.len()],
                };
                return Ok((table, tcf));
            }
        }
        // Build the match tree once. It depends only on
        // (domain_sizes, numeric_domain_sizes, hash_multipliers, regression
        // preconditions of `operators`); none of those change when we re-apply
        // costs below. Reusing it avoids 2x (or 4x for perimstar) rebuilds.
        let comparison_var_ids_for_tree = self.comparison_var_ids();
        let match_tree = MatchTree::build(
            generator.domain_sizes(),
            generator.numeric_domain_sizes(),
            generator.hash_multipliers(),
            &operators,
            &comparison_var_ids_for_tree,
        );

        let goal_facts = self.compute_abstract_goals(task);
        let table = self.build_distance_table_with_operators_for_goals_inner(
            task,
            &generator,
            &operators,
            false,
            &goal_facts,
            Some(&match_tree),
        )?;

        if let Some(state_id) = cap_state_id
            && let Some(&h_cap) = table.distances.get(state_id)
            && h_cap.is_finite()
        {
            let mut perim_table = table.clone();
            for h in &mut perim_table.distances {
                if !h.is_finite() || *h > h_cap {
                    *h = f64::NEG_INFINITY;
                }
            }
            let tcf = self.compute_saturated_abstract_operator_costs_from_operators_inner(
                task,
                &generator,
                &operators,
                &operator_costs,
                &perim_table,
                deadline,
                Some(&match_tree),
            )?;
            let mut saturated_operators = operators;
            apply_abstract_operator_costs(&mut saturated_operators, &tcf.operator_costs)?;
            let global_table = self.build_distance_table_with_operators_for_goals_inner(
                task,
                &generator,
                &saturated_operators,
                false,
                &goal_facts,
                Some(&match_tree),
            )?;
            return Ok((global_table, tcf));
        }

        let tcf = self.compute_saturated_abstract_operator_costs_from_operators_inner(
            task,
            &generator,
            &operators,
            &operator_costs,
            &table,
            deadline,
            Some(&match_tree),
        )?;
        // For Saturator::All, the saturated abstract-operator costs are tight
        // wrt `table.distances`: by construction every transition (u,v) using
        // operator op has saturated[op] >= h(u) - h(v), so any path from s to
        // the goal has length >= h(s) under saturated costs (telescoping), and
        // the original shortest path remains feasible. Therefore distances
        // under saturated costs equal `table.distances`, and the historic
        // second Dijkstra over saturated_operators was redundant.
        Ok((table, tcf))
    }

    pub fn build_abstract_operator_cost_partitioned_distance_table_from_system_and_footprints_with_deadline(
        &self,
        transition_system: &AbstractTransitionSystem,
        footprints: Option<&[AbstractOperatorFootprint]>,
        residual_costs: &TransitionResidualCosts,
        abstraction_id: usize,
        cap_state_id: Option<usize>,
        deadline: Option<Instant>,
    ) -> Result<(AbstractDistanceTable, AbstractOperatorCostFunction)> {
        ensure_online_scp_deadline(deadline)?;
        let concrete_op_ids = transition_system.concrete_operator_ids_by_abstract_operator();
        if let Some(footprints) = footprints {
            ensure!(
                footprints.len() >= concrete_op_ids.len(),
                "abstract-operator footprint/operator size mismatch: {} vs {}",
                footprints.len(),
                concrete_op_ids.len()
            );
        }
        let has_reductions = residual_costs.has_reductions();
        let operator_regions = (has_reductions && footprints.is_none())
            .then(|| transition_system.abstract_operator_regions());
        let mut operator_costs = vec![f64::INFINITY; concrete_op_ids.len()];
        for abstract_op_id in 0..concrete_op_ids.len() {
            if abstract_op_id % 64 == 0 {
                ensure_online_scp_deadline(deadline)?;
            }
            operator_costs[abstract_op_id] = if has_reductions && let Some(footprints) = footprints
            {
                footprints[abstract_op_id]
                    .labels
                    .iter()
                    .enumerate()
                    .map(|(label_id, footprint)| {
                        abstract_operator_label_cost(
                            residual_costs.cost_for_operator_footprint(
                                abstraction_id,
                                abstract_op_id,
                                footprint,
                            ),
                            residual_costs,
                            footprint,
                            None,
                            label_id,
                        )
                    })
                    .fold(f64::INFINITY, f64::min)
            } else if let Some(footprints) = footprints {
                footprints[abstract_op_id]
                    .labels
                    .iter()
                    .enumerate()
                    .map(|(label_id, footprint)| {
                        abstract_operator_label_cost(
                            residual_costs.base_cost(footprint.concrete_op_id),
                            residual_costs,
                            footprint,
                            None,
                            label_id,
                        )
                    })
                    .fold(f64::INFINITY, f64::min)
            } else if let Some(operator_regions) = &operator_regions {
                let Some(region) = operator_regions[abstract_op_id].as_ref() else {
                    continue;
                };
                concrete_op_ids[abstract_op_id]
                    .iter()
                    .map(|&concrete_op_id| {
                        residual_costs.cost_for_abstract_operator(
                            concrete_op_id,
                            abstraction_id,
                            abstract_op_id,
                            region,
                        )
                    })
                    .fold(f64::INFINITY, f64::min)
            } else {
                concrete_op_ids[abstract_op_id]
                    .iter()
                    .map(|&concrete_op_id| residual_costs.base_cost(concrete_op_id))
                    .fold(f64::INFINITY, f64::min)
            };
        }
        let transition_costs =
            transition_costs_from_abstract_operator_costs(transition_system, &operator_costs);
        let table = self.build_distance_table_with_transition_costs(
            transition_system,
            &transition_costs,
            &transition_system.hash_multipliers,
            &transition_system.numeric_domain_sizes,
        )?;

        if let Some(state_id) = cap_state_id
            && let Some(&h_cap) = table.distances.get(state_id)
            && h_cap.is_finite()
        {
            let mut perim_table = table.clone();
            for h in &mut perim_table.distances {
                if !h.is_finite() || *h > h_cap {
                    *h = f64::NEG_INFINITY;
                }
            }
            let tcf = self.compute_saturated_abstract_operator_costs(
                transition_system,
                &operator_costs,
                &perim_table,
            )?;
            let saturated_transition_costs = transition_costs_from_abstract_operator_costs(
                transition_system,
                &tcf.operator_costs,
            );
            let global_table = self.build_distance_table_with_transition_costs(
                transition_system,
                &saturated_transition_costs,
                &transition_system.hash_multipliers,
                &transition_system.numeric_domain_sizes,
            )?;
            return Ok((global_table, tcf));
        }

        let tcf = self.compute_saturated_abstract_operator_costs(
            transition_system,
            &operator_costs,
            &table,
        )?;
        Ok((table, tcf))
    }

    pub fn build_abstract_operator_footprints(
        &self,
        task: &dyn AbstractNumericTask,
        operators: &[AbstractOperator],
        finite_support: &FiniteSupportConfig,
    ) -> Result<Vec<AbstractOperatorFootprint>> {
        operators
            .iter()
            .map(|operator| {
                let labels = operator
                    .concrete_op_ids
                    .iter()
                    .copied()
                    .map(|concrete_op_id| {
                        self.build_concrete_operator_footprint(
                            task,
                            operator,
                            concrete_op_id,
                            finite_support,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(AbstractOperatorFootprint { labels })
            })
            .collect()
    }

    fn build_concrete_operator_footprint(
        &self,
        task: &dyn AbstractNumericTask,
        abstract_operator: &AbstractOperator,
        concrete_op_id: usize,
        finite_support: &FiniteSupportConfig,
    ) -> Result<ConcreteOperatorFootprint> {
        let concrete_operator = task.get_operators().get(concrete_op_id).with_context(|| {
            format!("abstract operator references missing concrete operator {concrete_op_id}")
        })?;
        let abstract_source_region =
            self.state_region_from_facts(task, &abstract_operator.preconditions)?;
        // `state_region_from_facts` already initializes numeric intervals to
        // `Interval::unbounded()` for variables that have no partition fact in
        // the operator's preconditions, and to the partition's interval for
        // variables that do (which includes affected vars and variables pulled
        // in via comparison-axiom preconditions).
        //
        // The loop below tightens affected-variable intervals further by
        // intersecting with the inverse target image. For non-affected
        // variables that are pinned by a partition fact, the partition
        // interval is the tightest superset of the concrete preimage we can
        // recover at this layer, so we keep it. Wiping it back to unbounded
        // would still be admissible, but it would force the cost-partitioning
        // overlap check to treat distinct partitions as universally
        // overlapping on those axes, which is the over-conservativeness that
        // hides per-region cost claims.
        let mut source_region = abstract_source_region.clone();
        let target_region =
            self.state_region_from_facts(task, &abstract_operator.regression_preconditions)?;
        let mut non_allocable_reason = None;
        let mut has_finite_changed_source = false;
        let mut has_infinite_changed_source = false;
        let mut precision_sum = 0.0;
        let mut precision_count = 0usize;

        for numeric_var_id in deterministic_affected_regular_numeric_vars(task, concrete_operator) {
            ensure!(
                numeric_var_id < abstract_source_region.numeric.len(),
                "abstract operator references affected numeric var {numeric_var_id}, but footprint has {} numeric vars",
                abstract_source_region.numeric.len()
            );
            let source_interval = abstract_source_region.numeric[numeric_var_id];
            let Some(effect_image) = deterministic_numeric_effect_image(
                task,
                concrete_operator,
                numeric_var_id,
                source_interval,
            ) else {
                non_allocable_reason
                    .get_or_insert(NonAllocableFootprintReason::UnsupportedEffectImage);
                continue;
            };
            if effect_image.is_noop_for_source(source_interval) {
                continue;
            }
            if effect_image.image.is_empty() {
                non_allocable_reason
                    .get_or_insert(NonAllocableFootprintReason::UnsupportedEffectImage);
                continue;
            }
            let target_interval = target_region.numeric[numeric_var_id];
            let Some(inverse_source) = effect_image.inverse_source_for_target(target_interval)
            else {
                non_allocable_reason
                    .get_or_insert(NonAllocableFootprintReason::UnsupportedEffectImage);
                continue;
            };
            source_region.numeric[numeric_var_id] =
                interval_intersection(source_interval, inverse_source);
            if source_region.numeric[numeric_var_id].is_empty() {
                non_allocable_reason
                    .get_or_insert(NonAllocableFootprintReason::UnsupportedEffectImage);
                continue;
            }
            precision_sum += changed_source_precision(
                source_region.numeric[numeric_var_id],
                effect_image.inverse,
            );
            precision_count += 1;
            let preimage = source_region.numeric[numeric_var_id];
            if interval_is_finite(preimage) && finite_support_stealable(preimage, finite_support) {
                has_finite_changed_source = true;
            } else {
                // Either an infinite preimage or a finite-but-too-wide one. Both
                // have the same admissibility consequence: this label cannot
                // safely steal cost under the finite-support gate.
                has_infinite_changed_source = true;
            }
        }
        let has_changed_numeric_source = has_finite_changed_source || has_infinite_changed_source;
        let allocable = has_finite_changed_source
            || (!has_changed_numeric_source && non_allocable_reason.is_none());
        if !allocable {
            if non_allocable_reason.is_none() && has_infinite_changed_source {
                non_allocable_reason
                    .get_or_insert(NonAllocableFootprintReason::InfiniteActiveSource);
            }
            if !footprint_has_informative_source(self, &source_region)? {
                non_allocable_reason
                    .get_or_insert(NonAllocableFootprintReason::UninformativeSource);
            }
        } else {
            non_allocable_reason = None;
        }

        Ok(ConcreteOperatorFootprint {
            concrete_op_id,
            source_region,
            allocable,
            max_allocation_fraction: if allocable {
                let fraction = if precision_count == 0 {
                    1.0
                } else {
                    precision_sum / precision_count as f64
                };
                ensure!(
                    fraction.is_finite() && (-1e-9..=1.0 + 1e-9).contains(&fraction),
                    "invalid abstract-operator footprint precision {fraction} for operator {concrete_op_id}"
                );
                fraction.clamp(0.0, 1.0)
            } else {
                0.0
            },
            non_allocable_reason,
        })
    }

    /// Computes an abstract wildcard plan (sequence of per-step concrete-op-ID sets) by:
    /// 1) Computing abstract goal distances with implicit regression Dijkstra.
    /// 2) Extracting a shortest-path abstract plan from the initial abstract state.
    /// 3) Collecting all cheapest realizations per step.
    pub fn compute_wildcard_plan(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        dump_distances: bool,
    ) -> Result<Option<WildcardPlanResult>> {
        self.compute_plan(task, combine_labels, dump_distances, true)
    }

    pub fn compute_plan(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        dump_distances: bool,
        use_wildcard_plans: bool,
    ) -> Result<Option<WildcardPlanResult>> {
        let mut local_rng = Some(SmallRng::seed_from_u64(current_time_seed()));
        self.compute_plan_with_rng(
            task,
            combine_labels,
            dump_distances,
            use_wildcard_plans,
            local_rng.as_mut(),
        )
    }

    pub(crate) fn compute_plan_with_rng(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        dump_distances: bool,
        use_wildcard_plans: bool,
        plan_step_rng: Option<&mut SmallRng>,
    ) -> Result<Option<WildcardPlanResult>> {
        let mut generator = self.make_operator_generator(task, combine_labels)?;
        let operators = generator.build_abstract_operators(task)?;
        let table =
            self.build_distance_table_with_operators(task, &generator, &operators, dump_distances)?;

        let comparison_var_ids = self.comparison_var_ids();
        let match_tree = MatchTree::build(
            generator.domain_sizes(),
            generator.numeric_domain_sizes(),
            generator.hash_multipliers(),
            &operators,
            &comparison_var_ids,
        );

        self.compute_wildcard_plan_from_table(
            task,
            &generator,
            &operators,
            &table,
            &comparison_var_ids,
            &match_tree,
            use_wildcard_plans,
            plan_step_rng,
        )
    }

    pub(crate) fn build_distance_table_with_operators(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        dump_distances: bool,
    ) -> Result<AbstractDistanceTable> {
        let goal_facts = self.compute_abstract_goals(task);
        self.build_distance_table_with_operators_for_goals(
            task,
            generator,
            operators,
            dump_distances,
            &goal_facts,
        )
    }

    fn zero_distance_table_for_generator(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
    ) -> Result<AbstractDistanceTable> {
        let hash_multipliers = generator.hash_multipliers();
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let comparison_var_ids = self.comparison_var_ids();
        let init_hash = self.compute_initial_state_hash_determined(
            task,
            numeric_domain_sizes,
            hash_multipliers,
            &comparison_var_ids,
        )?;
        let num_states = compute_num_states(&self.domain_sizes, numeric_domain_sizes)?;
        Ok(AbstractDistanceTable {
            distances: vec![0.0; num_states],
            generating_op_ids: vec![None; num_states],
            initial_state_hash: init_hash,
            goal_facts: self.compute_abstract_goals(task),
            hash_multipliers: hash_multipliers.to_vec(),
            numeric_domain_sizes: numeric_domain_sizes.to_vec(),
        })
    }

    fn compute_distance_to_goal_state_with_operators(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        target_state_hash: usize,
        deadline: Option<Instant>,
    ) -> Result<f64> {
        let goal_facts = self.compute_abstract_goals(task);
        let hash_multipliers = generator.hash_multipliers();
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let num_states = compute_num_states(&self.domain_sizes, numeric_domain_sizes)?;
        ensure!(
            target_state_hash < num_states,
            "target abstract state {target_state_hash} is out of bounds for {num_states} states"
        );
        let comparison_var_ids = self.comparison_var_ids();
        let match_tree = MatchTree::build(
            &self.domain_sizes,
            numeric_domain_sizes,
            hash_multipliers,
            operators,
            &comparison_var_ids,
        );
        self.compute_distance_to_goal_state(
            task,
            operators,
            &match_tree,
            &goal_facts,
            target_state_hash,
            numeric_domain_sizes,
            hash_multipliers,
            &comparison_var_ids,
            num_states,
            deadline,
        )
    }

    fn build_distance_table_with_operators_for_goals(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        dump_distances: bool,
        goal_facts: &[ExplicitFact],
    ) -> Result<AbstractDistanceTable> {
        self.build_distance_table_with_operators_for_goals_inner(
            task,
            generator,
            operators,
            dump_distances,
            goal_facts,
            None,
        )
    }

    fn build_distance_table_with_operators_for_goals_inner(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        dump_distances: bool,
        goal_facts: &[ExplicitFact],
        prebuilt_match_tree: Option<&MatchTree>,
    ) -> Result<AbstractDistanceTable> {
        let hash_multipliers = generator.hash_multipliers();
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let comparison_var_ids = self.comparison_var_ids();

        // Numeric-fd computes a *single* initial abstract state hash directly from the
        // concrete initial state (comparisons are evaluated, not enumerated).
        let init_hash = self.compute_initial_state_hash_determined(
            task,
            numeric_domain_sizes,
            hash_multipliers,
            &comparison_var_ids,
        )?;

        let num_states = compute_num_states(&self.domain_sizes, numeric_domain_sizes)?;

        let owned_match_tree = if prebuilt_match_tree.is_none() {
            Some(MatchTree::build(
                &self.domain_sizes,
                numeric_domain_sizes,
                hash_multipliers,
                operators,
                &comparison_var_ids,
            ))
        } else {
            None
        };
        let match_tree = prebuilt_match_tree.unwrap_or_else(|| owned_match_tree.as_ref().unwrap());
        let (distances, generating_op_ids) = self.compute_distances_and_generating_ops(
            task,
            operators,
            match_tree,
            &goal_facts,
            init_hash,
            numeric_domain_sizes,
            hash_multipliers,
            &comparison_var_ids,
            num_states,
        )?;

        let goal_facts = goal_facts.to_vec();
        let table = AbstractDistanceTable {
            distances,
            generating_op_ids,
            initial_state_hash: init_hash,
            goal_facts,
            hash_multipliers: hash_multipliers.to_vec(),
            numeric_domain_sizes: numeric_domain_sizes.to_vec(),
        };

        if dump_distances {
            self.dump_distances(task, &table);
        }

        Ok(table)
    }

    #[allow(clippy::needless_range_loop)]
    fn build_transition_system_with_operators(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        deadline: Option<Instant>,
        materialize_state_regions: bool,
    ) -> Result<AbstractTransitionSystem> {
        ensure_online_scp_deadline(deadline)?;
        let hash_multipliers = generator.hash_multipliers();
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let comparison_var_ids = self.comparison_var_ids();
        let goal_facts = self.compute_abstract_goals(task);
        let init_hash = self.compute_initial_state_hash_determined(
            task,
            numeric_domain_sizes,
            hash_multipliers,
            &comparison_var_ids,
        )?;
        let num_states = compute_num_states(&self.domain_sizes, numeric_domain_sizes)?;
        let match_tree = MatchTree::build(
            &self.domain_sizes,
            numeric_domain_sizes,
            hash_multipliers,
            operators,
            &comparison_var_ids,
        );

        let mut transitions: Vec<AbstractTransition> = Vec::new();
        transitions.reserve(num_states);
        let mut backward: Vec<Vec<usize>> = vec![Vec::new(); num_states];
        let mut forward: Vec<Vec<usize>> = vec![Vec::new(); num_states];
        let mut state_regions = Vec::new();
        if materialize_state_regions {
            state_regions.reserve(num_states);
            for state_hash in 0..num_states {
                state_regions.push(self.state_region_from_hash(
                    state_hash,
                    numeric_domain_sizes,
                    hash_multipliers,
                )?);
            }
        }
        let duplicate_transition_attempts = 0usize;
        let mut applicable_operator_ids: Vec<usize> = Vec::new();
        // Debug-only triple-uniqueness witness: every pushed AbstractTransition
        // must have a unique `(abstract_op_id, source_hash, target_hash)`. Now
        // that comparison-axiom bits are baked into operators via pre/eff/prev
        // facts, the source hash is fully determined by
        // `target_hash + op.hash_effect`, so each (target, op) yields a single
        // transition by construction.
        #[cfg(debug_assertions)]
        let mut seen_transition_triples: HashSet<(usize, usize, usize)> = HashSet::new();

        for target_hash in 0..num_states {
            if target_hash % 64 == 0 {
                ensure_online_scp_deadline(deadline)?;
            }
            match_tree.get_applicable_operator_ids(target_hash, &mut applicable_operator_ids);
            for &abstract_op_id in &applicable_operator_ids {
                let op = &operators[abstract_op_id];
                // Comparison bits live inside the operator's pre/eff facts, so
                // `op.hash_effect` already carries the comparison-var deltas.
                // The match-tree filter guarantees `target_hash` is consistent
                // with `op`'s effect values (including comparison effects);
                // applying `op.hash_effect` recovers the unique source hash.
                let predecessor_i64 = target_hash as i64 + op.hash_effect as i64;
                if predecessor_i64 < 0 || predecessor_i64 >= num_states as i64 {
                    continue;
                }
                let source_hash = predecessor_i64 as usize;
                if source_hash == target_hash {
                    continue;
                }
                #[cfg(debug_assertions)]
                {
                    let triple = (abstract_op_id, source_hash, target_hash);
                    debug_assert!(
                        seen_transition_triples.insert(triple),
                        "duplicate AbstractTransition triple {:?}",
                        triple
                    );
                }
                let transition_id = transitions.len();
                transitions.push(AbstractTransition {
                    transition_id,
                    abstract_op_id,
                    concrete_op_ids: op.concrete_op_ids.clone(),
                    source_hash,
                    target_hash,
                });
                backward[target_hash].push(transition_id);
                forward[source_hash].push(transition_id);
            }
        }

        // Tight invariant: within one abstraction, every transition sharing an
        // `abstract_op_id` must have identical numeric source and target regions.
        // The partition-fact enumeration in `abstract_operator_generator.rs`
        // (build_abstract_operators → enumerate_partition_combos) bakes the
        // numeric (source_partition, target_partition) pair into the abstract
        // operator's identity, so two transitions sharing the abstract op can
        // only differ in propositional wildcard dimensions of `source_hash` /
        // `target_hash`. This homogeneity is the property that lets the
        // finite-support cost-partitioning gate decide stealability per abstract
        // op rather than per individual transition.
        #[cfg(debug_assertions)]
        if materialize_state_regions {
            let mut representative_per_op: HashMap<usize, (usize, usize)> = HashMap::new();
            for transition in &transitions {
                match representative_per_op.get(&transition.abstract_op_id) {
                    Some(&(rep_src_hash, rep_tgt_hash)) => {
                        debug_assert_eq!(
                            state_regions[rep_src_hash].numeric,
                            state_regions[transition.source_hash].numeric,
                            "abstract_op_id {} has transitions with differing numeric source regions",
                            transition.abstract_op_id
                        );
                        debug_assert_eq!(
                            state_regions[rep_tgt_hash].numeric,
                            state_regions[transition.target_hash].numeric,
                            "abstract_op_id {} has transitions with differing numeric target regions",
                            transition.abstract_op_id
                        );
                    }
                    None => {
                        representative_per_op.insert(
                            transition.abstract_op_id,
                            (transition.source_hash, transition.target_hash),
                        );
                    }
                }
            }
        }

        // Goal states are simply those whose hash matches the goal facts.
        // The old self-consistency check via
        // `enumerate_states_with_evaluated_comparisons` filtered to states
        // whose comparison bits agreed with the (potentially ambiguous)
        // interval evaluation — that filtering is no longer needed because
        // operators only land transitions in states with the *optimistic*
        // comparison bit, and the initial state hash is computed with the
        // same optimistic semantics, so every reachable state has
        // self-consistent bits by construction.
        let mut goal_state_hashes = Vec::new();
        for state_hash in 0..num_states {
            if self.is_goal_state(
                state_hash,
                &goal_facts,
                numeric_domain_sizes,
                hash_multipliers,
            ) {
                goal_state_hashes.push(state_hash);
            }
        }

        Ok(AbstractTransitionSystem {
            transitions,
            duplicate_transition_attempts,
            backward,
            forward,
            goal_facts,
            goal_state_hashes,
            initial_state_hash: init_hash,
            hash_multipliers: hash_multipliers.to_vec(),
            numeric_domain_sizes: numeric_domain_sizes.to_vec(),
            state_regions,
        })
    }

    fn relevant_operator_ids_with_operators(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        deadline: Option<Instant>,
    ) -> Result<Vec<usize>> {
        ensure_online_scp_deadline(deadline)?;
        let hash_multipliers = generator.hash_multipliers();
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let comparison_var_ids = self.comparison_var_ids();
        let num_states = compute_num_states(&self.domain_sizes, numeric_domain_sizes)?;
        let match_tree = MatchTree::build(
            &self.domain_sizes,
            numeric_domain_sizes,
            hash_multipliers,
            operators,
            &comparison_var_ids,
        );
        let mut seen_operator_ids = vec![false; task.get_operators().len()];
        let mut num_seen = 0usize;
        let mut applicable_operator_ids: Vec<usize> = Vec::new();

        for target_hash in 0..num_states {
            if target_hash % 64 == 0 {
                ensure_online_scp_deadline(deadline)?;
            }
            if num_seen == seen_operator_ids.len() {
                break;
            }
            match_tree.get_applicable_operator_ids(target_hash, &mut applicable_operator_ids);
            for &abstract_op_id in &applicable_operator_ids {
                let op = &operators[abstract_op_id];
                if op
                    .concrete_op_ids
                    .iter()
                    .all(|&op_id| seen_operator_ids.get(op_id).copied().unwrap_or(false))
                {
                    continue;
                }
                let predecessor_i64 = target_hash as i64 + op.hash_effect as i64;
                if predecessor_i64 < 0 || predecessor_i64 >= num_states as i64 {
                    continue;
                }
                let source_hash = predecessor_i64 as usize;
                if source_hash == target_hash {
                    continue;
                }
                for &op_id in &op.concrete_op_ids {
                    ensure!(
                        op_id < seen_operator_ids.len(),
                        "concrete operator id out of range: {op_id} >= {}",
                        seen_operator_ids.len()
                    );
                    if !seen_operator_ids[op_id] {
                        seen_operator_ids[op_id] = true;
                        num_seen += 1;
                    }
                }
            }
        }

        Ok(seen_operator_ids
            .into_iter()
            .enumerate()
            .filter_map(|(op_id, seen)| seen.then_some(op_id))
            .collect())
    }

    fn state_region_from_hash(
        &self,
        state_hash: usize,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
    ) -> Result<StateRegion> {
        Ok(StateRegion {
            propositions: self.propositional_region_from_hash(state_hash, hash_multipliers)?,
            numeric: self.numeric_region_from_hash(
                state_hash,
                numeric_domain_sizes,
                hash_multipliers,
            )?,
        })
    }

    fn state_region_from_facts(
        &self,
        task: &dyn AbstractNumericTask,
        facts: &[ExplicitFact],
    ) -> Result<StateRegion> {
        let num_props = self.domain_sizes.len();
        let mut propositions = self.full_propositional_region()?;
        let mut numeric = vec![Interval::unbounded(); task.numeric_variables().len()];

        for fact in facts {
            if fact.var < num_props {
                propositions[fact.var] =
                    self.concrete_values_for_abstract_value(fact.var, fact.value)?;
            } else {
                let numeric_var_id = fact.var - num_props;
                ensure!(
                    numeric_var_id < numeric.len(),
                    "abstract-operator footprint fact references numeric var {numeric_var_id}, but task has {} numeric vars",
                    numeric.len()
                );
                numeric[numeric_var_id] = self
                    .partitions
                    .partition_interval(numeric_var_id, fact.value)
                    .with_context(|| {
                        format!(
                            "missing interval for numeric var {numeric_var_id} partition {}",
                            fact.value
                        )
                    })?;
            }
        }

        Ok(StateRegion {
            propositions,
            numeric,
        })
    }

    fn full_propositional_region(&self) -> Result<Vec<Vec<usize>>> {
        let mut region = Vec::with_capacity(self.domain_sizes.len());
        for var_id in 0..self.domain_sizes.len() {
            let mapping = self
                .domain_mapping
                .get(var_id)
                .with_context(|| format!("missing domain mapping for var {var_id}"))?;
            ensure!(
                !mapping.is_empty(),
                "empty concrete value set for propositional var {var_id}"
            );
            region.push((0..mapping.len()).collect());
        }
        Ok(region)
    }

    fn concrete_values_for_abstract_value(
        &self,
        var_id: usize,
        abstract_value: usize,
    ) -> Result<Vec<usize>> {
        let values = self
            .domain_mapping
            .get(var_id)
            .with_context(|| format!("missing domain mapping for var {var_id}"))?
            .iter()
            .enumerate()
            .filter_map(|(concrete_value, &mapped_value)| {
                (mapped_value == abstract_value).then_some(concrete_value)
            })
            .collect::<Vec<_>>();
        ensure!(
            !values.is_empty(),
            "empty concrete value set for var {var_id} abstract value {abstract_value}"
        );
        Ok(values)
    }

    fn propositional_region_from_hash(
        &self,
        state_hash: usize,
        hash_multipliers: &[usize],
    ) -> Result<Vec<Vec<usize>>> {
        let mut region = Vec::with_capacity(self.domain_sizes.len());
        for (var_id, &domain_size) in self.domain_sizes.iter().enumerate() {
            ensure!(domain_size > 0, "domain size must be > 0 for var {var_id}");
            let multiplier = *hash_multipliers
                .get(var_id)
                .with_context(|| format!("missing hash multiplier for var {var_id}"))?;
            let abstract_value = (state_hash / multiplier) % domain_size;
            region.push(self.concrete_values_for_abstract_value(var_id, abstract_value)?);
        }
        Ok(region)
    }

    fn numeric_region_from_hash(
        &self,
        state_hash: usize,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
    ) -> Result<Vec<Interval>> {
        let num_props = self.domain_sizes.len();
        let mut region = Vec::with_capacity(numeric_domain_sizes.len());
        for (numeric_var_id, &domain_size) in numeric_domain_sizes.iter().enumerate() {
            ensure!(
                domain_size > 0,
                "numeric domain size must be > 0 for var {numeric_var_id}"
            );
            let abs_var_id = num_props + numeric_var_id;
            let multiplier = *hash_multipliers.get(abs_var_id).with_context(|| {
                format!("missing hash multiplier for numeric var {numeric_var_id}")
            })?;
            let partition_id = (state_hash / multiplier) % domain_size;
            let interval = self
                .partitions
                .partition_interval(numeric_var_id, partition_id)
                .with_context(|| {
                    format!(
                        "missing interval for numeric var {numeric_var_id} partition {partition_id}"
                    )
                })?;
            region.push(interval);
        }
        Ok(region)
    }

    fn build_distance_table_with_transition_costs(
        &self,
        transition_system: &AbstractTransitionSystem,
        transition_costs: &[f64],
        hash_multipliers: &[usize],
        numeric_domain_sizes: &[usize],
    ) -> Result<AbstractDistanceTable> {
        ensure!(
            transition_system.transitions.len() == transition_costs.len(),
            "transition system/cost vector size mismatch: {} vs {}",
            transition_system.transitions.len(),
            transition_costs.len()
        );

        let num_states = transition_system.backward.len();
        let mut distances = vec![f64::INFINITY; num_states];
        let mut generating_op_ids = vec![None; num_states];
        let mut heap: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();

        for &state_hash in &transition_system.goal_state_hashes {
            ensure!(
                state_hash < num_states,
                "goal state hash out of range: {state_hash} >= {num_states}"
            );
            distances[state_hash] = 0.0;
            heap.push((Reverse(NotNan::new(0.0).unwrap()), state_hash));
        }

        while let Some((Reverse(d), target_hash)) = heap.pop() {
            let d = d.into_inner();
            if d > distances[target_hash] + 1e-12 {
                continue;
            }
            for &transition_id in &transition_system.backward[target_hash] {
                let transition = &transition_system.transitions[transition_id];
                let transition_cost = transition_costs[transition_id];
                if !transition_cost.is_finite() {
                    continue;
                }
                ensure!(
                    transition_cost >= -1e-9,
                    "transition costs must be nonnegative, got {transition_cost}"
                );
                let transition_cost = transition_cost.max(0.0);
                let alternative_cost = d + transition_cost;
                if alternative_cost + 1e-12 < distances[transition.source_hash] {
                    distances[transition.source_hash] = alternative_cost;
                    generating_op_ids[transition.source_hash] = Some(transition.abstract_op_id);
                    heap.push((
                        Reverse(NotNan::new(alternative_cost).context("alternative cost is NaN")?),
                        transition.source_hash,
                    ));
                }
            }
        }

        Ok(AbstractDistanceTable {
            distances,
            generating_op_ids,
            initial_state_hash: transition_system.initial_state_hash,
            goal_facts: transition_system.goal_facts.clone(),
            hash_multipliers: hash_multipliers.to_vec(),
            numeric_domain_sizes: numeric_domain_sizes.to_vec(),
        })
    }

    fn compute_saturated_transition_costs(
        &self,
        transition_system: &AbstractTransitionSystem,
        transition_costs: &[f64],
        table: &AbstractDistanceTable,
    ) -> Result<AbstractTransitionCostFunction> {
        ensure!(
            transition_system.transitions.len() == transition_costs.len(),
            "transition system/cost vector size mismatch: {} vs {}",
            transition_system.transitions.len(),
            transition_costs.len()
        );
        let mut saturated = vec![0.0; transition_system.transitions.len()];
        for transition in &transition_system.transitions {
            let source_h = table.distances[transition.source_hash];
            let target_h = table.distances[transition.target_hash];
            if !source_h.is_finite() || !target_h.is_finite() {
                continue;
            }
            let mut needed = source_h - target_h;
            if needed < 0.0 && needed > -1e-9 {
                needed = 0.0;
            }
            if needed < 0.0 {
                needed = 0.0;
            }
            ensure!(
                needed <= transition_costs[transition.transition_id] + 1e-7,
                "saturated transition cost exceeds residual transition cost: {} > {}",
                needed,
                transition_costs[transition.transition_id]
            );
            saturated[transition.transition_id] = needed;
        }
        Ok(AbstractTransitionCostFunction {
            transition_costs: saturated,
        })
    }

    fn compute_saturated_abstract_operator_costs(
        &self,
        transition_system: &AbstractTransitionSystem,
        operator_costs: &[f64],
        table: &AbstractDistanceTable,
    ) -> Result<AbstractOperatorCostFunction> {
        let mut saturated = vec![0.0_f64; operator_costs.len()];
        for transition in &transition_system.transitions {
            let source_h = table.distances[transition.source_hash];
            let target_h = table.distances[transition.target_hash];
            if !source_h.is_finite() || !target_h.is_finite() {
                continue;
            }
            let mut needed = source_h - target_h;
            if needed < 0.0 && needed > -1e-9 {
                needed = 0.0;
            }
            if needed < 0.0 {
                needed = 0.0;
            }
            let abstract_op_id = transition.abstract_op_id;
            ensure!(
                abstract_op_id < operator_costs.len(),
                "abstract operator id out of range: {abstract_op_id} >= {}",
                operator_costs.len()
            );
            ensure!(
                needed <= operator_costs[abstract_op_id] + 1e-7,
                "saturated abstract-operator cost exceeds residual abstract-operator cost: {} > {}",
                needed,
                operator_costs[abstract_op_id]
            );
            saturated[abstract_op_id] = saturated[abstract_op_id].max(needed);
        }
        Ok(AbstractOperatorCostFunction {
            operator_costs: saturated,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_saturated_abstract_operator_costs_from_operators_inner(
        &self,
        _task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        operator_costs: &[f64],
        table: &AbstractDistanceTable,
        deadline: Option<Instant>,
        prebuilt_match_tree: Option<&MatchTree>,
    ) -> Result<AbstractOperatorCostFunction> {
        ensure!(
            operators.len() == operator_costs.len(),
            "abstract operator/cost vector size mismatch: {} vs {}",
            operators.len(),
            operator_costs.len()
        );

        let num_states = table.distances.len();
        let comparison_var_ids = self.comparison_var_ids();
        let owned_match_tree = if prebuilt_match_tree.is_none() {
            Some(MatchTree::build(
                generator.domain_sizes(),
                generator.numeric_domain_sizes(),
                generator.hash_multipliers(),
                operators,
                &comparison_var_ids,
            ))
        } else {
            None
        };
        let match_tree = prebuilt_match_tree.unwrap_or_else(|| owned_match_tree.as_ref().unwrap());
        let mut saturated = vec![0.0_f64; operators.len()];
        let mut applicable_operator_ids = Vec::new();

        for target_hash in 0..num_states {
            if target_hash % 64 == 0 {
                ensure_online_scp_deadline(deadline)?;
            }
            let target_h = table.distances[target_hash];
            if !target_h.is_finite() {
                continue;
            }

            match_tree.get_applicable_operator_ids(target_hash, &mut applicable_operator_ids);
            for &abstract_op_id in &applicable_operator_ids {
                let op = &operators[abstract_op_id];
                // Comparison-axiom bits live in op.pre/eff/prev, so the source
                // hash is fully determined by `target_hash + op.hash_effect`.
                let predecessor_i64 = target_hash as i64 + op.hash_effect as i64;
                if predecessor_i64 < 0 || predecessor_i64 >= num_states as i64 {
                    continue;
                }
                let source_hash = predecessor_i64 as usize;
                if table.generating_op_ids.get(source_hash).copied().flatten()
                    == Some(abstract_op_id)
                {
                    let source_h = table.distances[source_hash];
                    if source_h.is_finite() {
                        let mut needed = source_h - target_h;
                        if needed < 0.0 {
                            needed = 0.0;
                        }
                        ensure!(
                            needed <= operator_costs[abstract_op_id] + 1e-7,
                            "saturated abstract-operator cost exceeds residual abstract-operator cost: {} > {}",
                            needed,
                            operator_costs[abstract_op_id]
                        );
                        saturated[abstract_op_id] = saturated[abstract_op_id].max(needed);
                    }
                }
            }
        }

        Ok(AbstractOperatorCostFunction {
            operator_costs: saturated,
        })
    }

    fn compute_saturated_costs(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        table: &AbstractDistanceTable,
    ) -> Result<Vec<f64>> {
        let num_operators = task.get_operators().len();
        let num_states = table.distances.len();
        let mut saturated_costs = vec![f64::NEG_INFINITY; num_operators];

        let comparison_var_ids = self.comparison_var_ids();
        let match_tree = MatchTree::build(
            generator.domain_sizes(),
            generator.numeric_domain_sizes(),
            generator.hash_multipliers(),
            operators,
            &comparison_var_ids,
        );

        let mut applicable_operator_ids = Vec::new();
        for target_hash in 0..num_states {
            let target_h = table.distances[target_hash];
            if !target_h.is_finite() {
                continue;
            }

            match_tree.get_applicable_operator_ids(target_hash, &mut applicable_operator_ids);
            for &abstract_op_id in &applicable_operator_ids {
                let op = &operators[abstract_op_id];
                // Comparison-axiom bits are baked into the operator's
                // pre/eff/prev facts, so `op.hash_effect` already includes
                // their delta. Source hash is fully determined.
                let predecessor_i64 = target_hash as i64 + op.hash_effect as i64;
                if predecessor_i64 < 0 || predecessor_i64 >= num_states as i64 {
                    continue;
                }
                let source_hash = predecessor_i64 as usize;
                if table.generating_op_ids.get(source_hash).copied().flatten()
                    == Some(abstract_op_id)
                    && let Some(&src_h) = table.distances.get(source_hash)
                    && src_h.is_finite()
                {
                    let needed = (src_h - target_h).max(0.0);
                    for &op_id in &op.concrete_op_ids {
                        if let Some(slot) = saturated_costs.get_mut(op_id) {
                            *slot = slot.max(needed);
                        }
                    }
                }
            }
        }

        for cost in &mut saturated_costs {
            if *cost == f64::NEG_INFINITY {
                *cost = 0.0;
            }
        }

        Ok(saturated_costs)
    }

    /// Prints a numeric-fd style table of core variables for all reachable abstract states.
    ///
    /// Core variables are:
    /// - all numeric variables with more than one partition,
    /// - all non-axiom propositional variables with abstract domain size > 1.
    pub fn dump_distances(&self, task: &dyn AbstractNumericTask, table: &AbstractDistanceTable) {
        utils::dump_distances(self, task, table);
    }
    fn comparison_var_ids(&self) -> Vec<usize> {
        self.comparison_trees
            .iter()
            .map(|t| t.affected_var_id)
            .collect()
    }

    fn compute_abstract_goals(&self, task: &dyn AbstractNumericTask) -> Vec<ExplicitFact> {
        let mut goal_axiom_map: HashMap<usize, usize> = HashMap::new();
        for (idx, ax) in task.axioms().iter().enumerate() {
            if !ax.conditions().is_empty() {
                goal_axiom_map.insert(ax.var_id(), idx);
            }
        }

        let mut out: Vec<ExplicitFact> = Vec::new();
        for i in 0..task.get_num_goals() {
            let g = task.get_goal_fact(i);
            let var = g.var;
            if let Some(&ax_idx) = goal_axiom_map.get(&var) {
                let ax = &task.axioms()[ax_idx];
                for cond in ax.conditions() {
                    let v = cond.var;
                    if self.domain_sizes.get(v).copied().unwrap_or(1) <= 1 {
                        continue;
                    }
                    let val = cond.value;
                    let mapped = self
                        .domain_mapping
                        .get(v)
                        .and_then(|m| m.get(val))
                        .copied()
                        .unwrap_or(cond.value);
                    out.push(ExplicitFact::new(cond.var, mapped));
                }
            } else {
                let v = g.var;
                if self.domain_sizes.get(v).copied().unwrap_or(1) <= 1 {
                    continue;
                }
                let val = g.value;
                let mapped = self
                    .domain_mapping
                    .get(v)
                    .and_then(|m| m.get(val))
                    .copied()
                    .unwrap_or(g.value);
                out.push(ExplicitFact::new(g.var, mapped));
            }
        }

        out
    }

    pub fn is_goal_state(
        &self,
        state_hash: usize,
        goals: &[ExplicitFact],
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
    ) -> bool {
        let num_props = self.domain_sizes.len();
        for g in goals {
            let var = g.var;
            let expected = g.value;
            let mult = hash_multipliers[var];
            let state = state_hash;
            let dom_size = if var < num_props {
                self.domain_sizes[var]
            } else {
                let n = var - num_props;
                numeric_domain_sizes.get(n).copied().unwrap_or(0)
            };
            let actual = (state / mult) % dom_size;
            if actual != expected {
                return false;
            }
        }
        true
    }

    fn compute_initial_state_hash_determined(
        &self,
        task: &dyn AbstractNumericTask,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        comparison_var_ids: &[usize],
    ) -> Result<usize> {
        let prop_init = task.get_initial_propositional_state_values();
        let num_init = task.get_initial_numeric_state_values();
        let num_props = self.domain_sizes.len();
        ensure!(
            prop_init.len() >= num_props,
            "initial propositional state too short: {} < {num_props}",
            prop_init.len()
        );
        ensure!(
            num_init.len() >= numeric_domain_sizes.len(),
            "initial numeric state too short: {} < {}",
            num_init.len(),
            numeric_domain_sizes.len()
        );

        // Smart α: comparison axiom vars in the abstraction's hash always
        // map to UNKNOWN at lookup time. By construction the
        // backward-reachable abstract state space has c ∈ {T, UNKNOWN}
        // per c (never F), and d(P, c=UNKNOWN) ≤ d(P, c=T) by the
        // unidirectional axiom edge UNKNOWN → definite. Probing at
        // UNKNOWN is an admissible lower bound regardless of α(init)'s
        // concrete c bits.
        //
        // Use `domain_mapping[var][COMPARISON_UNKNOWN_VAL]` rather than the
        // raw concrete UNKNOWN value: CEGAR may have collapsed the
        // 3-valued comparison domain to size 2 (TRUE separate, FALSE+UNKNOWN
        // merged) — see `apply_initial_goal_splits`. The mapping carries the
        // active abstract value for UNKNOWN regardless of size.
        let comparison_var_ids: HashSet<usize> = comparison_var_ids.iter().copied().collect();
        let mut index: usize = 0;
        for var in 0..num_props {
            let mult = hash_multipliers[var];
            let abs_val = if comparison_var_ids.contains(&var) {
                let mapping = &self.domain_mapping[var];
                ensure!(
                    mapping.len() > COMPARISON_UNKNOWN_VAL,
                    "comparison var {var} domain_mapping missing UNKNOWN slot: {mapping:?}"
                );
                mapping[COMPARISON_UNKNOWN_VAL]
            } else {
                *self.domain_mapping[var]
                    .get(prop_init[var])
                    .with_context(|| {
                        format!(
                            "missing mapping for propositional var {var} value index {}",
                            prop_init[var]
                        )
                    })?
            };
            index += mult * abs_val;
        }

        for num_var_id in 0..numeric_domain_sizes.len() {
            let abs_var = num_props + num_var_id;
            let mult = hash_multipliers[abs_var];
            let val = float_tolerance::canonicalize(num_init[num_var_id]);
            ensure!(
                val.is_finite() && !val.is_nan(),
                "initial numeric value for var {num_var_id} must be finite, got {val}"
            );
            let parts = self
                .partitions
                .partitions(num_var_id)
                .with_context(|| format!("missing partitions for numeric var {num_var_id}"))?;
            let part = utils::partition_for_value(parts, val).with_context(|| {
                format!(
                    "initial numeric value {val} not contained in any partition for numeric var {num_var_id}"
                )
            })?;
            index += mult * part;
        }

        Ok(index)
    }

    fn reset_comparison_vars_to_unknown_except(
        &self,
        state_hash: usize,
        hash_multipliers: &[usize],
        comparison_var_ids: &[usize],
        fixed_comparisons: &[ExplicitFact],
    ) -> Result<usize> {
        let mut out = state_hash;
        for &var_id in comparison_var_ids {
            ensure!(
                var_id < self.domain_sizes.len(),
                "comparison var id out of range: {var_id}"
            );
            if self.domain_sizes[var_id] <= 1 {
                continue;
            }
            let mult = hash_multipliers[var_id];
            let dom = self.domain_sizes[var_id];
            ensure!(dom > 0, "domain size must be > 0 for var {var_id}");
            let cur = (out / mult) % dom;
            let target_abs = if let Some(fixed_value) = fixed_comparisons
                .iter()
                .find(|fact| fact.var == var_id)
                .map(|fact| fact.value)
            {
                ensure!(
                    fixed_value < dom,
                    "fixed comparison value {fixed_value} out of abstract domain for var {var_id} with size {dom}"
                );
                fixed_value
            } else {
                *self.domain_mapping[var_id]
                    .get(COMPARISON_UNKNOWN_VAL)
                    .with_context(|| {
                        format!("missing UNKNOWN mapping for comparison var {var_id}")
                    })?
            };
            let cur_offset = cur
                .checked_mul(mult)
                .context("comparison current digit offset overflow")?;
            let target_offset = target_abs
                .checked_mul(mult)
                .context("comparison target digit offset overflow")?;
            out = out
                .checked_sub(cur_offset)
                .context("comparison reset encountered an invalid state hash")?;
            out = out
                .checked_add(target_offset)
                .context("comparison reset hash overflow")?;
        }
        Ok(out)
    }

    fn abstract_value_of_prop_var(
        &self,
        state_hash: usize,
        hash_multipliers: &[usize],
        var_id: usize,
    ) -> Result<usize> {
        ensure!(
            var_id < self.domain_sizes.len(),
            "propositional var id out of range: {var_id}"
        );
        ensure!(
            var_id < hash_multipliers.len(),
            "missing hash multiplier for propositional var {var_id}"
        );
        let domain_size = self.domain_sizes[var_id];
        ensure!(
            domain_size > 0,
            "domain size must be > 0 for propositional var {var_id}"
        );
        Ok((state_hash / hash_multipliers[var_id]) % domain_size)
    }

    fn state_satisfies_comparison_facts(
        &self,
        state_hash: usize,
        hash_multipliers: &[usize],
        facts: &[ExplicitFact],
    ) -> Result<bool> {
        for fact in facts {
            if self.domain_sizes[fact.var] <= 1 {
                continue;
            }
            if self.abstract_value_of_prop_var(state_hash, hash_multipliers, fact.var)?
                != fact.value
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn comparison_semantics_by_operator(
        &self,
        task: &dyn AbstractNumericTask,
        operators: &[AbstractOperator],
        comparison_var_ids: &[usize],
    ) -> Vec<OperatorComparisonSemantics> {
        // Invert `comparison var → [numeric var]` into `numeric var → [comparison var]`
        // so that the per-operator inner loop scales with the operator's
        // (small) changed-numeric-var set instead of the entire set of
        // comparison vars.
        debug_assert!(
            self.comparison_trees
                .iter()
                .all(|t| comparison_var_ids.contains(&t.affected_var_id)),
            "comparison_var_ids must contain every comparison tree's affected_var_id"
        );
        let num_numeric_vars = task.numeric_variables().len();
        let mut comp_vars_by_numeric: Vec<Vec<usize>> = vec![Vec::new(); num_numeric_vars];
        for tree in &self.comparison_trees {
            let deps = tree.regular_numeric_var_dependencies(task);
            for &dep in &deps {
                if dep < num_numeric_vars {
                    comp_vars_by_numeric[dep].push(tree.affected_var_id);
                }
            }
        }
        for list in &mut comp_vars_by_numeric {
            if list.len() > 1 {
                list.sort_unstable();
                list.dedup();
            }
        }

        operators
            .iter()
            .map(|op| {
                let mut affected_comparison_var_ids: Vec<usize> = Vec::new();
                for &numeric_var in &op.changed_numeric_vars {
                    if let Some(comp_vars) = comp_vars_by_numeric.get(numeric_var) {
                        affected_comparison_var_ids.extend_from_slice(comp_vars);
                    }
                }
                if affected_comparison_var_ids.len() > 1 {
                    affected_comparison_var_ids.sort_unstable();
                    affected_comparison_var_ids.dedup();
                }

                OperatorComparisonSemantics {
                    affected_comparison_var_ids,
                }
            })
            .collect()
    }

    #[allow(unused)]
    fn build_numeric_intervals(
        &self,
        state_hash: usize,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        task: &dyn AbstractNumericTask,
    ) -> Result<Vec<Interval>> {
        prepare_comparison_tree_inputs_from_abstract_state(
            task,
            &self.comparison_trees,
            &self.partitions,
            state_hash,
            self.domain_sizes.len(),
            numeric_domain_sizes,
            hash_multipliers,
        )
    }

    fn enumerate_states_with_evaluated_comparisons(
        &self,
        base_state_hash: usize,
        task: &dyn AbstractNumericTask,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        comparison_var_ids: &[usize],
        fixed_comparisons: &[ExplicitFact],
    ) -> Result<Vec<usize>> {
        if comparison_var_ids.is_empty() {
            return Ok(vec![base_state_hash]);
        }
        let num_props = self.domain_sizes.len();
        let state_unknown = self.reset_comparison_vars_to_unknown_except(
            base_state_hash,
            hash_multipliers,
            comparison_var_ids,
            fixed_comparisons,
        )?;

        // `fixed_comparisons` is typically empty or has 1-3 entries — replace the
        // per-call `HashMap<usize, usize>` (with default SipHash + heap alloc) with
        // a stack-friendly slice scan.
        let is_fixed_var =
            |var_id: usize| -> bool { fixed_comparisons.iter().any(|f| f.var == var_id) };
        let is_evaluated_var = |var_id: usize| -> bool { comparison_var_ids.contains(&var_id) };

        // Build the numeric intervals for this abstract state ONCE, then
        // evaluate each comparison tree against the shared buffer. The old
        // path called `evaluate_comparison_tree_from_abstract_state` per tree,
        // each call allocating a fresh `Vec<Interval>` and re-running
        // `fill_derived_numeric_intervals_from_comparison_trees`.
        let mut numeric_intervals: Vec<Interval> = Vec::new();
        let mut intervals_built = false;

        let mut states: Vec<usize> = vec![state_unknown];
        for tree in &self.comparison_trees {
            let var_id = tree.affected_var_id;
            ensure!(
                var_id < num_props,
                "comparison tree affected_var_id out of range: {var_id} >= {num_props}"
            );
            if !is_evaluated_var(var_id) {
                continue;
            }
            if self.domain_sizes[var_id] <= 1 {
                continue;
            }
            if is_fixed_var(var_id) {
                continue;
            }

            let mult = hash_multipliers[var_id];
            let unknown_abs = *self.domain_mapping[var_id]
                .get(COMPARISON_UNKNOWN_VAL)
                .with_context(|| format!("missing UNKNOWN mapping for comparison var {var_id}"))?
                as i32;
            let delta_true = (self.domain_mapping[var_id]
                .get(COMPARISON_TRUE_VAL)
                .copied()
                .with_context(|| format!("missing TRUE mapping for comparison var {var_id}"))?
                as i32
                - unknown_abs)
                * mult as i32;
            let delta_false = (self.domain_mapping[var_id]
                .get(COMPARISON_FALSE_VAL)
                .copied()
                .with_context(|| format!("missing FALSE mapping for comparison var {var_id}"))?
                as i32
                - unknown_abs)
                * mult as i32;

            if !intervals_built {
                prepare_comparison_tree_inputs_from_abstract_state_into(
                    task,
                    &self.comparison_trees,
                    &self.partitions,
                    base_state_hash,
                    num_props,
                    numeric_domain_sizes,
                    hash_multipliers,
                    &mut numeric_intervals,
                )?;
                intervals_built = true;
            }

            match tree.evaluate_interval(&numeric_intervals) {
                Some(true) => {
                    for s in &mut states {
                        *s = (*s as i32 + delta_true) as usize;
                    }
                }
                Some(false) => {
                    for s in &mut states {
                        *s = (*s as i32 + delta_false) as usize;
                    }
                }
                None => {
                    let mut next: Vec<usize> = Vec::with_capacity(states.len() * 2);
                    for &s in &states {
                        next.push((s as i32 + delta_true) as usize);
                        next.push((s as i32 + delta_false) as usize);
                    }
                    states = next;
                }
            }
        }
        Ok(states)
    }

    fn enumerate_states_with_evaluated_comparisons_cached(
        &self,
        base_state_hash: usize,
        task: &dyn AbstractNumericTask,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        comparison_var_ids: &[usize],
        fixed_comparisons: &[ExplicitFact],
        cache: &mut ComparisonEnumerationCache,
        cached_state_count: &mut usize,
    ) -> Result<CachedEnumeration> {
        let key = comparison_enumeration_signature(base_state_hash, fixed_comparisons);
        if let Some(entry) = cache.get(&key) {
            return Ok(entry.clone());
        }

        let states = self.enumerate_states_with_evaluated_comparisons(
            base_state_hash,
            task,
            numeric_domain_sizes,
            hash_multipliers,
            comparison_var_ids,
            fixed_comparisons,
        )?;
        let representative = states.iter().copied().max();
        let entry = CachedEnumeration {
            states: Arc::from(states),
            representative,
        };
        if cache.len() < COMPARISON_ENUMERATION_CACHE_MAX_ENTRIES
            && *cached_state_count + entry.states.len() <= COMPARISON_ENUMERATION_CACHE_MAX_STATES
        {
            *cached_state_count += entry.states.len();
            cache.insert(key, entry.clone());
        }
        Ok(entry)
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_wildcard_plan_from_table(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        table: &AbstractDistanceTable,
        comparison_var_ids: &[usize],
        match_tree: &MatchTree,
        use_wildcard_plans: bool,
        mut plan_step_rng: Option<&mut SmallRng>,
    ) -> Result<Option<WildcardPlanResult>> {
        let domain_sizes = generator.domain_sizes();
        let hash_multipliers = generator.hash_multipliers();
        let num_props = domain_sizes.len();
        let numeric_domain_sizes = generator.numeric_domain_sizes();

        let dist = &table.distances;
        let generating_op = &table.generating_op_ids;
        let comparison_preconditions =
            comparison_preconditions_by_operator(operators, comparison_var_ids);
        let comparison_semantics =
            self.comparison_semantics_by_operator(task, operators, comparison_var_ids);

        let mut current_hash = table.initial_state_hash;
        if current_hash >= dist.len() || !dist[current_hash].is_finite() {
            return Ok(None);
        }

        let mut wildcard_plan: Vec<Vec<usize>> = Vec::new();
        let mut abstract_state_hashes: Vec<usize> = vec![current_hash];
        let mut seen_states: Vec<usize> = Vec::new();

        // For debugging / parity with numeric-fd deviation code.
        let mut abstract_prop_states: Vec<Vec<usize>> = Vec::new();
        let mut abstract_numeric_states: Vec<Vec<usize>> = Vec::new();
        decode_state_to_vectors(
            current_hash,
            num_props,
            domain_sizes,
            numeric_domain_sizes,
            hash_multipliers,
            &mut abstract_prop_states,
            &mut abstract_numeric_states,
        );

        let mut safety_steps = 0usize;
        while !self.is_goal_state(
            current_hash,
            &table.goal_facts,
            numeric_domain_sizes,
            hash_multipliers,
        ) {
            safety_steps += 1;
            if safety_steps > dist.len() + 1 {
                bail!("abstract plan extraction exceeded safety limit")
            }
            let Some(op_id) = generating_op.get(current_hash).copied().flatten() else {
                bail!("missing generating operator for state {current_hash} with finite distance");
            };
            let op = operators
                .get(op_id)
                .with_context(|| format!("generating op id out of bounds: {op_id}"))?;
            let semantics = &comparison_semantics[op_id];
            ensure!(
                self.state_satisfies_comparison_facts(
                    current_hash,
                    hash_multipliers,
                    &comparison_preconditions[op_id],
                )?,
                "generating operator {op_id} is not applicable in abstract state {current_hash}"
            );

            // Smart plan extraction.
            //
            //   backward step: predecessor = state + hash_effect, then mask
            //                  op.affected_comparison_ids → UNKNOWN.
            //
            // The backward mask conflates source states differing only in
            // their affected-c bits (each ∈ {T, UNKNOWN} per the
            // backward-reachable invariant) into one predecessor entry
            // d[mask(state + hash_effect)] = d[state] + cost. To extract a
            // forward plan we must enumerate the 2^|affected_c| candidates
            // (each bit either T or UNKNOWN) and pick the one whose stored
            // distance satisfies the cost relation.
            let _ = semantics;
            let candidate_hash_effect = op.hash_effect;
            let base_successor =
                (current_hash as i32).wrapping_sub(candidate_hash_effect) as usize;
            let cur_d = dist[current_hash];
            ensure!(cur_d.is_finite(), "current distance must be finite");

            // Precompute per-affected-c (multiplier, T-abs, UNKNOWN-abs)
            // and the current bit value in `base_successor` (which inherited
            // current_hash's c bits since hash_effect has 0 contribution on
            // affected dims).
            let mut affected_meta: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(
                op.affected_comparison_ids.len(),
            );
            for &c_var in &op.affected_comparison_ids {
                let mult = hash_multipliers[c_var];
                let size = domain_sizes[c_var];
                let mapping = generator.domain_mapping().get(c_var).with_context(|| {
                    format!("missing domain mapping for affected comparison var {c_var}")
                })?;
                ensure!(
                    mapping.len() > super::domain_abstraction_heuristic::COMPARISON_UNKNOWN_VAL,
                    "comparison var {c_var} has no UNKNOWN slot in domain_mapping"
                );
                let unknown_abs =
                    mapping[super::domain_abstraction_heuristic::COMPARISON_UNKNOWN_VAL];
                let true_abs = mapping[super::domain_abstraction_heuristic::COMPARISON_TRUE_VAL];
                let current_bit = (base_successor / mult) % size;
                affected_meta.push((mult, current_bit, true_abs, unknown_abs));
            }
            ensure!(
                affected_meta.len() < 24,
                "plan-extraction enumeration would exceed 2^24 candidates (|affected|={})",
                affected_meta.len()
            );

            // Enumerate 2^|affected| bit assignments. Skip seen / out-of-range
            // candidates. Pick the one whose distance satisfies the cost
            // relation; tie-break by largest hash (matches numeric-fd parity).
            let mut chosen_successor: Option<usize> = None;
            let mut lowest_so_far = cur_d;
            let n_assignments = 1usize << affected_meta.len();
            for assignment in 0..n_assignments {
                let mut cand = base_successor;
                for (i, &(mult, current_bit, true_abs, unknown_abs)) in
                    affected_meta.iter().enumerate()
                {
                    let target_abs = if (assignment >> i) & 1 == 0 {
                        true_abs
                    } else {
                        unknown_abs
                    };
                    if current_bit != target_abs {
                        let delta = (target_abs as isize - current_bit as isize) * mult as isize;
                        let masked = cand as isize + delta;
                        ensure!(
                            masked >= 0 && (masked as usize) < dist.len(),
                            "plan-extraction enumeration produced out-of-range candidate"
                        );
                        cand = masked as usize;
                    }
                }
                if cand == current_hash {
                    continue;
                }
                if seen_states.contains(&cand) {
                    continue;
                }
                let cd = dist[cand];
                if !cd.is_finite() {
                    continue;
                }
                let valid_progress =
                    (cd < cur_d && op.cost > 0.0) || ((cd - cur_d).abs() <= 1e-9 && op.cost == 0.0);
                if valid_progress && chosen_successor.is_none_or(|x| cand > x) {
                    chosen_successor = Some(cand);
                    lowest_so_far = cd;
                }
            }
            let successor_hash = chosen_successor.with_context(|| {
                format!(
                    "plan-extraction: no successor satisfies dist equation for state {current_hash} with op {op_id} (cur_d={cur_d}, op.cost={}, |affected|={})",
                    op.cost,
                    op.affected_comparison_ids.len()
                )
            })?;
            ensure!(
                successor_hash < dist.len(),
                "successor hash out of range: {successor_hash}"
            );
            ensure!(
                (lowest_so_far - cur_d + op.cost).abs() <= 1e-6,
                "chosen successor violates plan-extraction distance relation"
            );
            let base_successor = successor_hash;
            let required_cost = op.cost;

            // Collect cheapest concrete ops. For smart-design ops, the
            // candidate connects (current_hash → base_successor) iff
            //   mask_unknown(base_successor + cand.hash_effect,
            //                cand.affected_comparison_ids) == current_hash
            // (mirror of the backward Dijkstra step).
            let mut step: Vec<usize> = Vec::new();
            let mut applicable_operator_ids: Vec<usize> = Vec::new();
            match_tree.get_applicable_operator_ids(base_successor, &mut applicable_operator_ids);
            for &cand_op_id in &applicable_operator_ids {
                let cand_op = operators
                    .get(cand_op_id)
                    .with_context(|| format!("candidate op id out of bounds: {cand_op_id}"))?;
                let _ = comparison_semantics;
                if (cand_op.cost - required_cost).abs() > 1e-9 {
                    continue;
                }
                if !self.state_satisfies_comparison_facts(
                    current_hash,
                    hash_multipliers,
                    &comparison_preconditions[cand_op_id],
                )? {
                    continue;
                }
                let mut cand_pred =
                    (base_successor as i32).wrapping_add(cand_op.hash_effect) as usize;
                for &c_var in &cand_op.affected_comparison_ids {
                    let mult = hash_multipliers[c_var];
                    let size = domain_sizes[c_var];
                    let mapping = generator.domain_mapping().get(c_var).with_context(|| {
                        format!("missing domain mapping for affected comparison var {c_var}")
                    })?;
                    ensure!(
                        mapping.len()
                            > super::domain_abstraction_heuristic::COMPARISON_UNKNOWN_VAL,
                        "comparison var {c_var} has no UNKNOWN slot in domain_mapping"
                    );
                    let unknown_abs =
                        mapping[super::domain_abstraction_heuristic::COMPARISON_UNKNOWN_VAL];
                    let current_bit = (cand_pred / mult) % size;
                    if current_bit != unknown_abs {
                        let delta =
                            (unknown_abs as isize - current_bit as isize) * mult as isize;
                        let masked = cand_pred as isize + delta;
                        ensure!(
                            masked >= 0 && (masked as usize) < dist.len(),
                            "concrete-step extraction: predecessor hash out of range after UNKNOWN mask"
                        );
                        cand_pred = masked as usize;
                    }
                }
                if cand_pred == current_hash {
                    step = cand_op.concrete_op_ids.clone();
                    step.sort_unstable();
                    step.dedup();
                    if use_wildcard_plans {
                        if let Some(rng) = plan_step_rng.as_deref_mut() {
                            step.shuffle(rng);
                        }
                    } else {
                        let selected_op = match plan_step_rng.as_deref_mut() {
                            Some(rng) => step.choose(rng).copied(),
                            None => step.first().copied(),
                        }
                        .with_context(|| {
                            format!(
                                "failed to choose a representative concrete operator for abstract state {current_hash}"
                            )
                        })?;
                        step.clear();
                        step.push(selected_op);
                    }
                    break;
                }
            }
            ensure!(
                !step.is_empty(),
                "failed to extract a concrete plan step for abstract state {current_hash}"
            );
            wildcard_plan.push(step);

            seen_states.push(current_hash);
            current_hash = successor_hash;
            abstract_state_hashes.push(current_hash);
            decode_state_to_vectors(
                current_hash,
                num_props,
                domain_sizes,
                numeric_domain_sizes,
                hash_multipliers,
                &mut abstract_prop_states,
                &mut abstract_numeric_states,
            );
        }

        Ok(Some(WildcardPlanResult {
            wildcard_plan,
            abstract_state_hashes,
            abstract_prop_states,
            abstract_numeric_states,
        }))
    }

    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn compute_distances_and_generating_ops(
        &self,
        _task: &dyn AbstractNumericTask,
        operators: &[AbstractOperator],
        match_tree: &MatchTree,
        goal_facts: &[ExplicitFact],
        _initial_state_hash: usize,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        _comparison_var_ids: &[usize],
        num_states: usize,
    ) -> Result<(Vec<f64>, Vec<Option<usize>>)> {
        let num_props = self.domain_sizes.len();
        let mut distances: Vec<f64> = vec![f64::INFINITY; num_states];
        let mut generating_op_ids: Vec<Option<usize>> = vec![None; num_states];

        let mut core_vars: Vec<usize> = Vec::new();
        for (v, &dom) in self.domain_sizes.iter().enumerate() {
            if dom > 1 {
                core_vars.push(v);
            }
        }
        for (n, &dom) in numeric_domain_sizes.iter().enumerate() {
            if dom > 1 {
                core_vars.push(num_props + n);
            }
        }
        core_vars.sort_unstable();
        core_vars.dedup();

        let mut heap: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();

        // Initialize with goal states. With comparison-axiom bits baked into
        // every operator's pre/eff/prev, the old self-consistency filter via
        // `enumerate_states_with_evaluated_comparisons_cached` is moot: every
        // reachable state already has the optimistic bits its partition
        // implies (the operators only land in such states, and the initial
        // state hash is computed with the same optimistic semantics).
        for state_hash in 0..num_states {
            if self.is_goal_state(
                state_hash,
                goal_facts,
                numeric_domain_sizes,
                hash_multipliers,
            ) {
                distances[state_hash] = 0.0;
                heap.push((Reverse(NotNan::new(0.0).unwrap()), state_hash));
            }
        }

        // Precompute per-comparison-var hash delta for the axiom edge
        // (P, c=definite) → (P, c=UNKNOWN). The axiom edge in backward
        // direction relaxes d(P, c=UNKNOWN) ≤ d(P, c=definite) + 0, so we
        // process it inline alongside operator expansion.
        let comparison_var_ids_set: HashSet<usize> = self
            .comparison_trees
            .iter()
            .map(|tree| tree.affected_var_id)
            .collect();
        let mut axiom_var_data: Vec<(usize, usize, isize, isize)> = Vec::new();
        for &var in &comparison_var_ids_set {
            let size = self.domain_sizes.get(var).copied().unwrap_or(0);
            if size < 3 {
                continue;
            }
            let mult = hash_multipliers.get(var).copied().unwrap_or(0);
            if mult == 0 {
                continue;
            }
            let to_unknown_from_true = (super::domain_abstraction_heuristic::COMPARISON_UNKNOWN_VAL
                as isize
                - super::domain_abstraction_heuristic::COMPARISON_TRUE_VAL as isize)
                * mult as isize;
            let to_unknown_from_false = (super::domain_abstraction_heuristic::COMPARISON_UNKNOWN_VAL
                as isize
                - super::domain_abstraction_heuristic::COMPARISON_FALSE_VAL as isize)
                * mult as isize;
            axiom_var_data.push((var, mult, to_unknown_from_true, to_unknown_from_false));
        }

        let mut applicable_operator_ids: Vec<usize> = Vec::new();
        while let Some((Reverse(d), state_hash)) = heap.pop() {
            let d = d.into_inner();
            if d > distances[state_hash] + 1e-12 {
                continue;
            }

            // numeric-fd parity: from this state, also relax the
            // (P, c=UNKNOWN) sibling for every comparison var c whose
            // current bit is definite. This implements the 0-cost axiom
            // edge (P, c=UNKNOWN) → (P, c=definite) in backward direction,
            // so that ops with `eff = c = UNKNOWN` can later route their
            // backward predecessors through (P, c=UNKNOWN).
            //
            // Unidirectional: we do NOT propagate (P, c=UNKNOWN) → (P, c=T)
            // / (P, c=F) because the forward axiom only goes UNKNOWN →
            // definite. Adding the reverse breaks tightness on instances
            // where (P, c=T) and (P, c=F) have genuinely different
            // distances (plant-watering, sailing prob_1_2), even though
            // it would be admissible.
            for &(var, mult, delta_t, delta_f) in &axiom_var_data {
                let bit_value = (state_hash / mult) % 3;
                let delta = if bit_value == super::domain_abstraction_heuristic::COMPARISON_TRUE_VAL
                {
                    delta_t
                } else if bit_value
                    == super::domain_abstraction_heuristic::COMPARISON_FALSE_VAL
                {
                    delta_f
                } else {
                    continue;
                };
                let unknown_i = state_hash as isize + delta;
                if unknown_i < 0 || unknown_i as usize >= num_states {
                    continue;
                }
                let unknown_hash = unknown_i as usize;
                if d + 1e-12 < distances[unknown_hash] {
                    distances[unknown_hash] = d;
                    generating_op_ids[unknown_hash] = generating_op_ids[state_hash];
                    heap.push((
                        Reverse(NotNan::new(d).context("axiom-propagated distance is NaN")?),
                        unknown_hash,
                    ));
                }
                let _ = var;
            }

            match_tree.get_applicable_operator_ids(state_hash, &mut applicable_operator_ids);
            for &op_id in &applicable_operator_ids {
                let op = &operators[op_id];
                ensure!(op.cost.is_finite(), "abstract operator cost must be finite");
                let alternative_cost = d + op.cost;
                // Backward Dijkstra step:
                //
                // 1. Compute the predecessor base by applying `op.hash_effect`
                //    — this accounts for every paired pre/eff fact and every
                //    prevail. For comparison vars that this op can change but
                //    does NOT pin (`affected_comparison_ids`), the
                //    `hash_effect` is 0 on c's dim, so `pred_base` inherits
                //    the current state's c bit.
                //
                // 2. Mask each c in `affected_comparison_ids` to UNKNOWN on
                //    the predecessor hash — the op's forward semantics reset
                //    the bit to UNKNOWN, so backward the predecessor's c bit
                //    must be UNKNOWN regardless of the current state's c.
                //    This is the max-hash representative for c (UNKNOWN = 2,
                //    the largest digit), consistent with the design that
                //    backward-reachable states have c ∈ {T, UNKNOWN}.
                let predecessor_i64 = state_hash as i64 + op.hash_effect as i64;
                if predecessor_i64 < 0 || predecessor_i64 >= num_states as i64 {
                    continue;
                }
                let mut pred = predecessor_i64 as usize;
                // Mask each affected c's bit on the predecessor hash to the
                // abstract UNKNOWN value. Use `domain_mapping[c][CONCRETE_UNKNOWN]`
                // because CEGAR may collapse the 3-valued comparison domain
                // to size 2 — see `apply_initial_goal_splits`.
                for &c_var in &op.affected_comparison_ids {
                    let mult = hash_multipliers[c_var];
                    let size = self.domain_sizes[c_var];
                    let mapping = &self.domain_mapping[c_var];
                    ensure!(
                        mapping.len() > super::domain_abstraction_heuristic::COMPARISON_UNKNOWN_VAL,
                        "affected comparison var {c_var} has no UNKNOWN mapping: {mapping:?}"
                    );
                    let unknown_abs = mapping[super::domain_abstraction_heuristic::COMPARISON_UNKNOWN_VAL];
                    let current_bit = (pred / mult) % size;
                    if current_bit != unknown_abs {
                        let delta = (unknown_abs as isize - current_bit as isize) * mult as isize;
                        let masked = pred as isize + delta;
                        debug_assert!(
                            masked >= 0 && (masked as usize) < num_states,
                            "predecessor hash out of range after UNKNOWN mask: c_var={c_var}, current_bit={current_bit}, unknown_abs={unknown_abs}"
                        );
                        pred = masked as usize;
                    }
                }
                debug_assert!(pred < num_states, "predecessor hash does not fit usize");

                if alternative_cost + 1e-12 < distances[pred] {
                    distances[pred] = alternative_cost;
                    generating_op_ids[pred] = Some(op_id);
                    heap.push((
                        Reverse(
                            NotNan::new(alternative_cost)
                                .context("alternative cost is NaN")?,
                        ),
                        pred,
                    ));
                }
            }
        }

        Ok((distances, generating_op_ids))
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_distance_to_goal_state(
        &self,
        task: &dyn AbstractNumericTask,
        operators: &[AbstractOperator],
        match_tree: &MatchTree,
        goal_facts: &[ExplicitFact],
        target_state_hash: usize,
        numeric_domain_sizes: &[usize],
        hash_multipliers: &[usize],
        comparison_var_ids: &[usize],
        num_states: usize,
        deadline: Option<Instant>,
    ) -> Result<f64> {
        let mut distances: Vec<f64> = vec![f64::INFINITY; num_states];
        let mut heap: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();
        let mut comparison_enumeration_cache: ComparisonEnumerationCache =
            ComparisonEnumerationCache::default();
        let mut cached_comparison_state_count = 0usize;

        for state_hash in 0..num_states {
            if state_hash % 4096 == 0 {
                ensure_online_scp_deadline(deadline)?;
            }
            if !self.is_goal_state(
                state_hash,
                goal_facts,
                numeric_domain_sizes,
                hash_multipliers,
            ) {
                continue;
            }
            let alts = self.enumerate_states_with_evaluated_comparisons_cached(
                state_hash,
                task,
                numeric_domain_sizes,
                hash_multipliers,
                comparison_var_ids,
                &[],
                &mut comparison_enumeration_cache,
                &mut cached_comparison_state_count,
            )?;
            if !alts.states.contains(&state_hash) {
                continue;
            }
            distances[state_hash] = 0.0;
            heap.push((Reverse(NotNan::new(0.0).unwrap()), state_hash));
        }

        let comparison_semantics =
            self.comparison_semantics_by_operator(task, operators, comparison_var_ids);
        let comparison_preconditions =
            comparison_preconditions_by_operator(operators, comparison_var_ids);
        let mut applicable_operator_ids: Vec<usize> = Vec::new();
        while let Some((Reverse(d), state_hash)) = heap.pop() {
            let d = d.into_inner();
            if d > distances[state_hash] + 1e-12 {
                continue;
            }
            if state_hash == target_state_hash {
                return Ok(d);
            }

            match_tree.get_applicable_operator_ids(state_hash, &mut applicable_operator_ids);
            for &op_id in &applicable_operator_ids {
                let op = &operators[op_id];
                let semantics = &comparison_semantics[op_id];
                ensure!(op.cost.is_finite(), "abstract operator cost must be finite");
                let alternative_cost = d + op.cost;
                let comparison_reset_vars = comparison_vars_to_reset_for_operator(
                    semantics,
                    &comparison_preconditions[op_id],
                );
                let target_base = self.reset_comparison_vars_to_unknown_except(
                    state_hash,
                    hash_multipliers,
                    &comparison_reset_vars,
                    &[],
                )?;
                let predecessor_base_i64 = target_base as i64 + op.hash_effect as i64;
                if predecessor_base_i64 < 0 || predecessor_base_i64 >= num_states as i64 {
                    continue;
                }
                let predecessor_base = predecessor_base_i64 as usize;
                let possible_predecessors = self
                    .enumerate_states_with_evaluated_comparisons_cached(
                        predecessor_base,
                        task,
                        numeric_domain_sizes,
                        hash_multipliers,
                        &comparison_reset_vars,
                        &comparison_preconditions[op_id],
                        &mut comparison_enumeration_cache,
                        &mut cached_comparison_state_count,
                    )?;
                let representative_predecessor = possible_predecessors.representative;

                for pred in possible_predecessors.states.iter().copied() {
                    debug_assert!(pred < num_states, "predecessor hash does not fit usize");
                    if alternative_cost + 1e-12 < distances[pred] {
                        distances[pred] = alternative_cost;
                        if pred == target_state_hash || Some(pred) == representative_predecessor {
                            heap.push((
                                Reverse(
                                    NotNan::new(alternative_cost)
                                        .context("alternative cost is NaN")?,
                                ),
                                pred,
                            ));
                        }
                    }
                }
            }
        }

        Ok(f64::INFINITY)
    }
}

fn compute_num_states(domain_sizes: &[usize], numeric_domain_sizes: &[usize]) -> Result<usize> {
    let mut num: usize = 1;
    for (i, &s) in domain_sizes.iter().enumerate() {
        ensure!(s > 0, "domain size for var {i} must be > 0, got {s}");
        num = num
            .checked_mul(s)
            .context("abstract state space too large (overflow)")?;
    }
    for &s in numeric_domain_sizes.iter() {
        num = num
            .checked_mul(s)
            .context("abstract state space too large (overflow)")?;
    }
    Ok(num)
}

fn apply_operator_costs(operators: &mut [AbstractOperator], operator_costs: &[f64]) -> Result<()> {
    for op in operators {
        ensure!(
            !op.concrete_op_ids.is_empty(),
            "abstract operator without concrete labels"
        );
        let mut cost = f64::INFINITY;
        for &concrete_op_id in &op.concrete_op_ids {
            let concrete_cost = *operator_costs.get(concrete_op_id).with_context(|| {
                format!("missing residual cost for concrete operator {concrete_op_id}")
            })?;
            ensure!(
                concrete_cost.is_finite(),
                "residual cost for concrete operator {concrete_op_id} must be finite"
            );
            cost = cost.min(concrete_cost);
        }
        op.cost = cost;
    }
    Ok(())
}

fn apply_abstract_operator_costs(
    operators: &mut [AbstractOperator],
    operator_costs: &[f64],
) -> Result<()> {
    ensure!(
        operators.len() == operator_costs.len(),
        "abstract operator/cost vector size mismatch: {} vs {}",
        operators.len(),
        operator_costs.len()
    );
    for (abstract_op_id, op) in operators.iter_mut().enumerate() {
        let cost = operator_costs[abstract_op_id];
        ensure!(
            cost.is_finite(),
            "residual cost for abstract operator {abstract_op_id} must be finite"
        );
        op.cost = cost;
    }
    Ok(())
}

fn abstract_operator_costs_from_footprints(
    num_operators: usize,
    footprints: &[AbstractOperatorFootprint],
    budgets: Option<&[AbstractOperatorCostBudget]>,
    label_rescue_operator_ids: Option<&HashSet<usize>>,
    residual_costs: &TransitionResidualCosts,
    abstraction_id: usize,
    deadline: Option<Instant>,
) -> Result<Vec<f64>> {
    if let Some(budgets) = budgets {
        ensure!(
            budgets.len() >= num_operators,
            "abstract-operator budget/operator size mismatch: budgets={} operators={num_operators}",
            budgets.len()
        );
    }
    let has_reductions = residual_costs.has_reductions();
    let uniform_label_residuals =
        label_rescue_operator_ids.map(|_| residual_costs.operator_costs_for_label_cp());
    let mut operator_costs = vec![f64::INFINITY; num_operators];
    for abstract_op_id in 0..num_operators {
        if abstract_op_id % 64 == 0 {
            ensure_online_scp_deadline(deadline)?;
        }
        let footprint = footprints
            .get(abstract_op_id)
            .with_context(|| format!("missing footprint for abstract operator {abstract_op_id}"))?;
        ensure!(
            !footprint.labels.is_empty(),
            "abstract operator {abstract_op_id} has no concrete footprint labels"
        );
        let budget = budgets.map(|budgets| &budgets[abstract_op_id]);
        if let Some(budget) = budget {
            ensure!(
                budget.label_fractions.len() == footprint.labels.len(),
                "abstract-operator budget label count mismatch for abstract op {abstract_op_id}: budgets={} labels={}",
                budget.label_fractions.len(),
                footprint.labels.len()
            );
        }
        operator_costs[abstract_op_id] = if has_reductions {
            footprint
                .labels
                .iter()
                .enumerate()
                .map(|(label_id, label)| {
                    if label.allocable {
                        let residual = residual_costs.cost_for_operator_footprint(
                            abstraction_id,
                            abstract_op_id,
                            label,
                        );
                        abstract_operator_label_cost(
                            residual,
                            residual_costs,
                            label,
                            budget,
                            label_id,
                        )
                    } else if matches!(
                        label.non_allocable_reason,
                        Some(
                            NonAllocableFootprintReason::InfiniteActiveSource
                                | NonAllocableFootprintReason::UninformativeSource
                        )
                    ) && label_rescue_operator_ids
                        .is_some_and(|ids| ids.contains(&label.concrete_op_id))
                    {
                        let uniform_residual = uniform_label_residuals
                            .as_ref()
                            .and_then(|costs| costs.get(label.concrete_op_id))
                            .copied()
                            .unwrap_or(f64::INFINITY);
                        abstract_operator_non_region_label_cost(
                            uniform_residual,
                            residual_costs,
                            label,
                            budget,
                            label_id,
                        )
                    } else {
                        0.0
                    }
                })
                .fold(f64::INFINITY, f64::min)
        } else {
            footprint
                .labels
                .iter()
                .enumerate()
                .map(|(label_id, label)| {
                    let residual = residual_costs.base_cost(label.concrete_op_id);
                    if label.allocable {
                        abstract_operator_label_cost(
                            residual,
                            residual_costs,
                            label,
                            budget,
                            label_id,
                        )
                    } else if matches!(
                        label.non_allocable_reason,
                        Some(
                            NonAllocableFootprintReason::InfiniteActiveSource
                                | NonAllocableFootprintReason::UninformativeSource
                        )
                    ) && label_rescue_operator_ids
                        .is_some_and(|ids| ids.contains(&label.concrete_op_id))
                    {
                        abstract_operator_non_region_label_cost(
                            residual,
                            residual_costs,
                            label,
                            budget,
                            label_id,
                        )
                    } else {
                        0.0
                    }
                })
                .fold(f64::INFINITY, f64::min)
        };
        ensure!(
            operator_costs[abstract_op_id].is_finite(),
            "residual cost for abstract operator {abstract_op_id} is not finite"
        );
    }
    Ok(operator_costs)
}

fn abstract_operator_label_cost(
    residual: f64,
    residual_costs: &TransitionResidualCosts,
    label: &ConcreteOperatorFootprint,
    budget: Option<&AbstractOperatorCostBudget>,
    label_id: usize,
) -> f64 {
    if !label.allocable {
        return 0.0;
    }
    let fraction = if let Some(budget) = budget {
        budget.label_fractions[label_id]
    } else {
        1.0
    };
    assert!(
        fraction.is_finite() && (-1e-9..=1.0 + 1e-9).contains(&fraction),
        "invalid abstract-operator allocation fraction {fraction}"
    );
    residual.min(residual_costs.base_cost(label.concrete_op_id) * fraction.clamp(0.0, 1.0))
}

fn abstract_operator_non_region_label_cost(
    residual: f64,
    residual_costs: &TransitionResidualCosts,
    label: &ConcreteOperatorFootprint,
    budget: Option<&AbstractOperatorCostBudget>,
    label_id: usize,
) -> f64 {
    let fraction = if let Some(budget) = budget {
        budget.label_fractions[label_id]
    } else {
        1.0
    };
    assert!(
        fraction.is_finite() && (-1e-9..=1.0 + 1e-9).contains(&fraction),
        "invalid abstract-operator allocation fraction {fraction}"
    );
    residual.min(residual_costs.base_cost(label.concrete_op_id) * fraction.clamp(0.0, 1.0))
}

fn get_comparison_preconditions(
    op: &AbstractOperator,
    comparison_var_ids: &[usize],
) -> Vec<ExplicitFact> {
    let mut out: Vec<ExplicitFact> = Vec::new();
    for f in &op.preconditions {
        if comparison_var_ids.contains(&f.var) && f.value != COMPARISON_UNKNOWN_VAL {
            out.push(f.clone());
        }
    }
    out
}

fn comparison_preconditions_by_operator(
    operators: &[AbstractOperator],
    comparison_var_ids: &[usize],
) -> Vec<Vec<ExplicitFact>> {
    operators
        .iter()
        .map(|op| get_comparison_preconditions(op, comparison_var_ids))
        .collect()
}

fn comparison_vars_to_reset_for_operator(
    semantics: &OperatorComparisonSemantics,
    comparison_preconditions: &[ExplicitFact],
) -> Vec<usize> {
    let mut vars = semantics.affected_comparison_var_ids.clone();
    vars.retain(|var_id| {
        !comparison_preconditions
            .iter()
            .any(|fact| fact.var == *var_id)
    });
    vars.sort_unstable();
    vars.dedup();
    vars
}

fn transition_costs_from_abstract_operator_costs(
    transition_system: &AbstractTransitionSystem,
    operator_costs: &[f64],
) -> Vec<f64> {
    transition_system
        .transitions
        .iter()
        .map(|transition| {
            operator_costs
                .get(transition.abstract_op_id)
                .copied()
                .unwrap_or(f64::INFINITY)
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct DeterministicNumericEffectImage {
    image: Interval,
    inverse: DeterministicNumericEffectInverse,
}

#[derive(Debug, Clone, Copy)]
enum DeterministicNumericEffectInverse {
    Additive { delta: f64 },
    AssignmentConstant { value: f64 },
}

impl DeterministicNumericEffectImage {
    fn is_noop_for_source(&self, source_interval: Interval) -> bool {
        match self.inverse {
            DeterministicNumericEffectInverse::Additive { delta } => delta.abs() <= 1e-12,
            DeterministicNumericEffectInverse::AssignmentConstant { value } => {
                interval_is_singleton(source_interval) && source_interval.contains(value)
            }
        }
    }

    fn inverse_source_for_target(&self, target_interval: Interval) -> Option<Interval> {
        match self.inverse {
            DeterministicNumericEffectInverse::Additive { delta } => {
                Some(shift_interval(target_interval, -delta))
            }
            DeterministicNumericEffectInverse::AssignmentConstant { value } => target_interval
                .contains(value)
                .then_some(Interval::unbounded()),
        }
    }
}

fn deterministic_numeric_effect_image(
    task: &dyn AbstractNumericTask,
    operator: &Operator,
    numeric_var_id: usize,
    source_interval: Interval,
) -> Option<DeterministicNumericEffectImage> {
    let initial_numeric = task.get_initial_numeric_state_values();
    let mut delta = 0.0;
    let mut assignment = None;
    let mut touched = false;
    for effect in operator
        .assignment_effects()
        .iter()
        .filter(|effect| effect.affected_var_id() == numeric_var_id)
    {
        if effect.is_conditional() || !effect.conditions().is_empty() {
            return None;
        }
        let rhs_value = match task.numeric_variables()[effect.var_id()].get_type() {
            NumericType::Constant | NumericType::Cost => *initial_numeric.get(effect.var_id())?,
            _ => return None,
        };
        if !rhs_value.is_finite() {
            return None;
        }
        match effect.operation() {
            AssignmentOperation::Plus => {
                if assignment.is_some() {
                    return None;
                }
                delta += rhs_value;
                touched = true;
            }
            AssignmentOperation::Minus => {
                if assignment.is_some() {
                    return None;
                }
                delta -= rhs_value;
                touched = true;
            }
            AssignmentOperation::Assign => {
                if touched || assignment.is_some() {
                    return None;
                }
                assignment = Some(rhs_value);
                touched = true;
            }
            AssignmentOperation::Times | AssignmentOperation::Divide => return None,
        }
    }
    if let Some(value) = assignment {
        Some(DeterministicNumericEffectImage {
            image: Interval::singleton(value),
            inverse: DeterministicNumericEffectInverse::AssignmentConstant { value },
        })
    } else if touched && delta.abs() > 1e-12 {
        Some(DeterministicNumericEffectImage {
            image: shift_interval(source_interval, delta),
            inverse: DeterministicNumericEffectInverse::Additive { delta },
        })
    } else {
        None
    }
}

fn deterministic_affected_regular_numeric_vars(
    task: &dyn AbstractNumericTask,
    operator: &Operator,
) -> Vec<usize> {
    let mut deltas = vec![0.0; task.numeric_variables().len()];
    let mut assignments = Vec::new();
    for effect in operator.assignment_effects() {
        let affected_var_id = effect.affected_var_id();
        if task
            .numeric_variables()
            .get(affected_var_id)
            .is_none_or(|variable| variable.get_type() != &NumericType::Regular)
        {
            continue;
        }
        if effect.is_conditional() || !effect.conditions().is_empty() {
            continue;
        }
        if !matches!(
            effect.operation(),
            AssignmentOperation::Plus | AssignmentOperation::Minus | AssignmentOperation::Assign
        ) {
            continue;
        }
        if !matches!(
            task.numeric_variables()[effect.var_id()].get_type(),
            NumericType::Constant | NumericType::Cost
        ) {
            continue;
        }
        let Some(&rhs_value) = task.get_initial_numeric_state_values().get(effect.var_id()) else {
            continue;
        };
        if !rhs_value.is_finite() {
            continue;
        }
        match effect.operation() {
            AssignmentOperation::Plus => deltas[affected_var_id] += rhs_value,
            AssignmentOperation::Minus => deltas[affected_var_id] -= rhs_value,
            AssignmentOperation::Assign => assignments.push(affected_var_id),
            AssignmentOperation::Times | AssignmentOperation::Divide => unreachable!(),
        }
    }
    let mut vars: Vec<usize> = deltas
        .iter()
        .enumerate()
        .filter_map(|(var_id, &delta)| (delta.abs() > 1e-12).then_some(var_id))
        .collect();
    vars.extend(assignments);
    vars.sort_unstable();
    vars.dedup();
    vars
}

fn shift_interval(interval: Interval, delta: f64) -> Interval {
    Interval::new(
        interval.lower + delta,
        interval.upper + delta,
        interval.lower_closed,
        interval.upper_closed,
    )
}

fn interval_is_finite(interval: Interval) -> bool {
    interval.lower.is_finite() && interval.upper.is_finite()
}

fn interval_is_singleton(interval: Interval) -> bool {
    interval.lower == interval.upper && interval.lower_closed && interval.upper_closed
}

/// Width-threshold gate for the finite-support cost-partitioning extension.
///
/// Returns `true` iff the interval is a singleton, or iff its width fits inside
/// `cfg.max_stealable_width`. Caller must have already established that the
/// interval is finite — infinite intervals are rejected upstream.
fn finite_support_stealable(interval: Interval, cfg: &FiniteSupportConfig) -> bool {
    if interval_is_singleton(interval) {
        return true;
    }
    (interval.upper - interval.lower) <= cfg.max_stealable_width
}

fn changed_source_precision(
    source_interval: Interval,
    inverse: DeterministicNumericEffectInverse,
) -> f64 {
    if !interval_is_finite(source_interval) {
        return 0.0;
    }
    match inverse {
        DeterministicNumericEffectInverse::Additive { delta } => {
            let delta = delta.abs();
            if delta <= 1e-9 {
                return 0.0;
            }
            let width = source_interval.upper - source_interval.lower;
            if width <= 1e-9 {
                1.0
            } else {
                (delta / width).min(1.0)
            }
        }
        DeterministicNumericEffectInverse::AssignmentConstant { .. } => 1.0,
    }
}

fn interval_intersection(lhs: Interval, rhs: Interval) -> Interval {
    let (lower, lower_closed) = if lhs.lower > rhs.lower {
        (lhs.lower, lhs.lower_closed)
    } else if lhs.lower < rhs.lower {
        (rhs.lower, rhs.lower_closed)
    } else {
        (lhs.lower, lhs.lower_closed && rhs.lower_closed)
    };
    let (upper, upper_closed) = if lhs.upper < rhs.upper {
        (lhs.upper, lhs.upper_closed)
    } else if lhs.upper > rhs.upper {
        (rhs.upper, rhs.upper_closed)
    } else {
        (lhs.upper, lhs.upper_closed && rhs.upper_closed)
    };
    Interval::new(lower, upper, lower_closed, upper_closed)
}

fn footprint_has_informative_source(
    factory: &DomainAbstractionFactory,
    source_region: &StateRegion,
) -> Result<bool> {
    for (var_id, values) in source_region.propositions.iter().enumerate() {
        let full_size = factory
            .domain_mapping
            .get(var_id)
            .with_context(|| format!("missing domain mapping for propositional var {var_id}"))?
            .len();
        if values.len() < full_size {
            return Ok(true);
        }
    }
    Ok(source_region
        .numeric
        .iter()
        .any(|interval| interval_is_finite(*interval) && !interval.is_empty()))
}

fn decode_state_to_vectors(
    state_hash: usize,
    num_props: usize,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    hash_multipliers: &[usize],
    prop_out: &mut Vec<Vec<usize>>,
    num_out: &mut Vec<Vec<usize>>,
) {
    let mut props: Vec<usize> = Vec::with_capacity(num_props);
    for var_id in 0..num_props {
        let mult = hash_multipliers[var_id];
        let dom = domain_sizes[var_id];
        let val = (state_hash / mult) % dom;
        props.push(val);
    }
    let mut nums: Vec<usize> = Vec::with_capacity(numeric_domain_sizes.len());
    for (num_id, &dom_u) in numeric_domain_sizes.iter().enumerate() {
        let abs_var = num_props + num_id;
        let mult = hash_multipliers[abs_var];
        let dom = dom_u;
        let part = (state_hash / mult) % dom;
        nums.push(part);
    }
    prop_out.push(props);
    num_out.push(nums);
}
