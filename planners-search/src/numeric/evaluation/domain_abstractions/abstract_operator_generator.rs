#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, anyhow, ensure};

use planners_sas::numeric::axioms::CalOperator;

use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentEffect, Effect, ExplicitFact, NumericType, Operator,
    metric_operator_cost_from_initial_values,
};
use planners_sas::numeric::utils::float_tolerance;

use super::comparison_expression::{ArithOp, ComparisonTree, Interval};
use super::domain_abstraction::{ComparisonAxiomIndex, NumericPartitions};
use super::numeric_context::fill_derived_numeric_intervals_from_comparison_trees;
use super::utils;

const COMPARISON_TRUE_VAL: usize = 0;
const COMPARISON_FALSE_VAL: usize = 1;
const COMPARISON_UNKNOWN_VAL: usize = 2;

pub type DomainMapping = Vec<Vec<usize>>;

#[derive(Debug, Clone, Default)]
pub struct IncrementalAbstractOperatorCache {
    per_operator: Vec<CachedConcreteOperatorBuild>,
    dirty_propositional_vars: HashSet<usize>,
    dirty_numeric: bool,
}

impl IncrementalAbstractOperatorCache {
    pub fn mark_refined(
        &mut self,
        summary: &crate::numeric::evaluation::domain_abstractions::cegar::RefinementSummary,
    ) {
        if !summary.refined_numeric_vars.is_empty() {
            self.dirty_numeric = true;
        }
        self.dirty_propositional_vars
            .extend(summary.refined_propositional_vars.iter().copied());
    }

    fn take_dirty_state(&mut self) -> (HashSet<usize>, bool) {
        (
            std::mem::take(&mut self.dirty_propositional_vars),
            std::mem::take(&mut self.dirty_numeric),
        )
    }
}

#[derive(Debug, Clone, Default)]
struct CachedConcreteOperatorBuild {
    dependencies: OperatorDependencies,
    skeletons: Vec<AbstractOperatorSkeleton>,
}

#[derive(Debug, Clone, Default)]
struct OperatorDependencies {
    propositional_vars: HashSet<usize>,
}

impl OperatorDependencies {
    fn intersects_propositional(&self, dirty_vars: &HashSet<usize>) -> bool {
        !dirty_vars.is_disjoint(&self.propositional_vars)
    }
}

#[derive(Debug, Clone)]
struct AbstractOperatorCandidate {
    concrete_op_id: usize,
    cost: f64,
    prev_pairs: Vec<ExplicitFact>,
    pre_pairs: Vec<ExplicitFact>,
    eff_pairs: Vec<ExplicitFact>,
    changed_numeric_vars: Vec<usize>,
    /// `(cost_bits, FNV+SplitMix64 hash of prev+pre+eff+cost)`.
    /// Precomputed once at candidate creation so the (often-cached) candidate
    /// can be matched against the grouping map in `push_candidate` without
    /// re-walking its fact slices on every CEGAR iteration.
    cost_bits: u64,
    signature_hash: u64,
}

#[derive(Debug, Clone)]
struct AbstractOperatorSkeleton {
    concrete_op_id: usize,
    cost: f64,
    prev_pairs: Vec<ExplicitFact>,
    pre_pairs: Vec<ExplicitFact>,
    eff_pairs: Vec<ExplicitFact>,
    ass_effects: Vec<AssignmentEffect>,
    op_preconditions: Vec<ExplicitFact>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AbstractOperator {
    pub concrete_op_ids: Vec<usize>,
    pub cost: f64,
    pub hash_effect: i32,
    pub regression_preconditions: Vec<ExplicitFact>,
    pub preconditions: Vec<ExplicitFact>,
    pub changed_numeric_vars: Vec<usize>,
}

impl AbstractOperator {
    pub fn new(
        prev_pairs: &[ExplicitFact],
        pre_pairs: &[ExplicitFact],
        eff_pairs: &[ExplicitFact],
        cost: f64,
        hash_multipliers: &[usize],
        concrete_op_ids: Vec<usize>,
        changed_numeric_vars: Vec<usize>,
    ) -> Self {
        let mut preconditions: Vec<ExplicitFact> = pre_pairs.to_vec();
        preconditions.extend_from_slice(prev_pairs);
        preconditions.sort();
        debug_assert!(preconditions.windows(2).all(|w| w[0].var != w[1].var));

        let mut regression_preconditions: Vec<ExplicitFact> = prev_pairs.to_vec();
        regression_preconditions.extend_from_slice(eff_pairs);
        regression_preconditions.sort();
        debug_assert!(
            regression_preconditions
                .windows(2)
                .all(|w| w[0].var != w[1].var)
        );

        debug_assert_eq!(
            pre_pairs.len(),
            eff_pairs.len(),
            "abstract operator pre/eff pair mismatch: pre_pairs={pre_pairs:?} eff_pairs={eff_pairs:?}"
        );

        let mut hash_effect: i32 = 0;
        for (pre, eff) in pre_pairs.iter().zip(eff_pairs.iter()) {
            debug_assert_eq!(
                pre.var, eff.var,
                "abstract operator transition var mismatch: pre={pre:?} eff={eff:?}"
            );

            let var = pre.var;
            let multiplier = hash_multipliers[var];
            let new_val = pre.value as i32;
            let old_val = eff.value as i32;
            hash_effect += (new_val - old_val) * multiplier as i32;
        }

        Self {
            concrete_op_ids,
            cost,
            hash_effect,
            regression_preconditions,
            preconditions,
            changed_numeric_vars,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TransitionInfo {
    pub source_partition_facts: Vec<ExplicitFact>,
    pub target_partition_facts: Vec<ExplicitFact>,
    pub prevail_facts: Vec<ExplicitFact>,
    pub changed_numeric_vars: Vec<usize>,
}

/// Stored signature per *operator* (post-grouping). Holding the original
/// pair lists once per operator (instead of per candidate) lets us match
/// later candidates without re-allocating their signature each call.
#[derive(Debug, Clone)]
struct StoredOperatorSignature {
    prev_pairs: Vec<ExplicitFact>,
    pre_pairs: Vec<ExplicitFact>,
    eff_pairs: Vec<ExplicitFact>,
    cost_bits: u64,
}

impl StoredOperatorSignature {
    fn matches_candidate(&self, candidate: &AbstractOperatorCandidate, cost_bits: u64) -> bool {
        self.cost_bits == cost_bits
            && self.prev_pairs.as_slice() == candidate.prev_pairs.as_slice()
            && self.pre_pairs.as_slice() == candidate.pre_pairs.as_slice()
            && self.eff_pairs.as_slice() == candidate.eff_pairs.as_slice()
    }

    fn from_candidate(candidate: &AbstractOperatorCandidate, cost_bits: u64) -> Self {
        Self {
            prev_pairs: candidate.prev_pairs.clone(),
            pre_pairs: candidate.pre_pairs.clone(),
            eff_pairs: candidate.eff_pairs.clone(),
            cost_bits,
        }
    }
}

/// Compute a 64-bit signature hash from the operator's prev/pre/eff fact
/// slices and `cost_bits`. Precomputed once per candidate at creation time
/// and stored on the candidate, so `push_candidate` does not have to walk
/// the slices again on every CEGAR iteration.
///
/// FNV-1a-style mix at the u64 chunk level + a SplitMix64 finalizer for
/// even bit distribution. Same construction as the state-registry hash.
#[inline]
fn compute_signature_hash(
    prev_pairs: &[ExplicitFact],
    pre_pairs: &[ExplicitFact],
    eff_pairs: &[ExplicitFact],
    cost_bits: u64,
) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;

    #[inline(always)]
    fn mix_facts(hash: &mut u64, facts: &[ExplicitFact]) {
        *hash ^= facts.len() as u64;
        *hash = hash.wrapping_mul(FNV_PRIME);
        for fact in facts {
            *hash ^= fact.var as u64;
            *hash = hash.wrapping_mul(FNV_PRIME);
            *hash ^= fact.value as u64;
            *hash = hash.wrapping_mul(FNV_PRIME);
        }
        *hash ^= 0xdeadbeef_cafef00d;
        *hash = hash.wrapping_mul(FNV_PRIME);
    }

    let mut hash = FNV_OFFSET;
    mix_facts(&mut hash, prev_pairs);
    mix_facts(&mut hash, pre_pairs);
    mix_facts(&mut hash, eff_pairs);
    hash ^= cost_bits;
    hash = hash.wrapping_mul(FNV_PRIME);

    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51afd7ed558ccd);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xc4ceb9fe1a85ec53);
    hash ^= hash >> 33;
    hash
}

/// Identity hasher reused from the state registry (pass-through for u64
/// keys). Keeps grouping-map probes off the SipHash slow path.
type SignatureMap = HashMap<
    u64,
    Vec<u32>,
    std::hash::BuildHasherDefault<planners_sas::numeric::state_registry::IdentityU64Hasher>,
>;

struct AbstractOperatorFinalizer {
    combine_labels: bool,
    hash_multipliers: Vec<usize>,
    operators: Vec<AbstractOperator>,
    /// Per-operator stored signatures, indexed parallel to `operators`.
    stored_signatures: Vec<StoredOperatorSignature>,
    /// Map from signature hash to the operator indices that share that hash.
    /// Most buckets hold a single index; collisions degrade gracefully via
    /// the small-vec scan and `StoredOperatorSignature::matches_candidate`.
    grouping: SignatureMap,
}

impl AbstractOperatorFinalizer {
    fn new(combine_labels: bool, hash_multipliers: &[usize]) -> Self {
        Self {
            combine_labels,
            hash_multipliers: hash_multipliers.to_vec(),
            operators: Vec::new(),
            stored_signatures: Vec::new(),
            grouping: SignatureMap::default(),
        }
    }

    fn push_candidate(&mut self, candidate: &AbstractOperatorCandidate) {
        if self.combine_labels {
            // Hash and cost bits are precomputed at candidate-creation time.
            let cost_bits = candidate.cost_bits;
            let hash = candidate.signature_hash;
            if let Some(bucket) = self.grouping.get(&hash) {
                for &idx in bucket {
                    let stored = &self.stored_signatures[idx as usize];
                    if stored.matches_candidate(candidate, cost_bits) {
                        self.operators[idx as usize]
                            .concrete_op_ids
                            .push(candidate.concrete_op_id);
                        return;
                    }
                }
            }

            let idx = self.operators.len();
            self.operators.push(AbstractOperator::new(
                &candidate.prev_pairs,
                &candidate.pre_pairs,
                &candidate.eff_pairs,
                candidate.cost,
                &self.hash_multipliers,
                vec![candidate.concrete_op_id],
                candidate.changed_numeric_vars.clone(),
            ));
            self.stored_signatures
                .push(StoredOperatorSignature::from_candidate(
                    candidate, cost_bits,
                ));
            self.grouping.entry(hash).or_default().push(idx as u32);
            return;
        }

        // No label combining — every candidate becomes its own operator.
        self.operators.push(AbstractOperator::new(
            &candidate.prev_pairs,
            &candidate.pre_pairs,
            &candidate.eff_pairs,
            candidate.cost,
            &self.hash_multipliers,
            vec![candidate.concrete_op_id],
            candidate.changed_numeric_vars.clone(),
        ));
    }

    fn into_operators(self) -> Vec<AbstractOperator> {
        self.operators
    }
}

fn has_in_list_conflict(facts: &[ExplicitFact]) -> bool {
    facts
        .windows(2)
        .any(|w| w[0].var == w[1].var && w[0].value != w[1].value)
}

fn has_cross_list_conflict(a: &[ExplicitFact], b: &[ExplicitFact]) -> bool {
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        let va = a[i].var;
        let vb = b[j].var;
        if va < vb {
            i += 1;
        } else if vb < va {
            j += 1;
        } else {
            if a[i].value != b[j].value {
                return true;
            }
            i += 1;
            j += 1;
        }
    }
    false
}

fn transition_vars_match(pre_pairs: &[ExplicitFact], eff_pairs: &[ExplicitFact]) -> bool {
    pre_pairs.len() == eff_pairs.len()
        && pre_pairs
            .iter()
            .zip(eff_pairs.iter())
            .all(|(pre, eff)| pre.var == eff.var)
}

fn fact_value_for_var(facts: &[ExplicitFact], var_id: usize) -> Option<usize> {
    facts
        .binary_search_by_key(&var_id, |f| f.var)
        .ok()
        .map(|idx| facts[idx].value)
}

fn build_candidate_from_transition(
    skeleton: &AbstractOperatorSkeleton,
    trans: &TransitionInfo,
) -> Option<AbstractOperatorCandidate> {
    let mut extended_pre_pairs =
        Vec::with_capacity(skeleton.pre_pairs.len() + trans.source_partition_facts.len());
    extended_pre_pairs.extend_from_slice(&skeleton.pre_pairs);
    let mut extended_eff_pairs =
        Vec::with_capacity(skeleton.eff_pairs.len() + trans.target_partition_facts.len());
    extended_eff_pairs.extend_from_slice(&skeleton.eff_pairs);
    let mut extended_prev_pairs =
        Vec::with_capacity(skeleton.prev_pairs.len() + trans.prevail_facts.len());
    extended_prev_pairs.extend_from_slice(&skeleton.prev_pairs);

    let mut source_pos = 0;
    let mut target_pos = 0;
    while source_pos < trans.source_partition_facts.len()
        || target_pos < trans.target_partition_facts.len()
    {
        let source = trans.source_partition_facts.get(source_pos);
        let target = trans.target_partition_facts.get(target_pos);
        let var_id = match (source, target) {
            (Some(src), Some(tgt)) => src.var.min(tgt.var),
            (Some(src), None) => src.var,
            (None, Some(tgt)) => tgt.var,
            (None, None) => unreachable!(),
        };

        if let Some(src) = source
            && src.var == var_id
        {
            if let Some(existing) = fact_value_for_var(&skeleton.pre_pairs, var_id) {
                if existing != src.value {
                    return None;
                }
            } else {
                extended_pre_pairs.push(src.clone());
            }
        }
        if let Some(tgt) = target
            && tgt.var == var_id
            && fact_value_for_var(&skeleton.eff_pairs, var_id).is_none()
        {
            extended_eff_pairs.push(tgt.clone());
        }

        while trans
            .source_partition_facts
            .get(source_pos)
            .is_some_and(|f| f.var == var_id)
        {
            source_pos += 1;
        }
        while trans
            .target_partition_facts
            .get(target_pos)
            .is_some_and(|f| f.var == var_id)
        {
            target_pos += 1;
        }
    }
    extended_prev_pairs.extend_from_slice(&trans.prevail_facts);

    extended_pre_pairs.sort();
    extended_eff_pairs.sort();
    extended_prev_pairs.sort();
    extended_pre_pairs.dedup();
    extended_eff_pairs.dedup();
    extended_prev_pairs.dedup();

    if has_in_list_conflict(&extended_pre_pairs)
        || has_in_list_conflict(&extended_eff_pairs)
        || has_in_list_conflict(&extended_prev_pairs)
        || has_cross_list_conflict(&extended_pre_pairs, &extended_prev_pairs)
        || has_cross_list_conflict(&extended_prev_pairs, &extended_eff_pairs)
        || !transition_vars_match(&extended_pre_pairs, &extended_eff_pairs)
    {
        return None;
    }

    let cost_bits = float_tolerance::canonical_bits(skeleton.cost);
    let signature_hash = compute_signature_hash(
        &extended_prev_pairs,
        &extended_pre_pairs,
        &extended_eff_pairs,
        cost_bits,
    );
    Some(AbstractOperatorCandidate {
        concrete_op_id: skeleton.concrete_op_id,
        cost: skeleton.cost,
        prev_pairs: extended_prev_pairs,
        pre_pairs: extended_pre_pairs,
        eff_pairs: extended_eff_pairs,
        changed_numeric_vars: trans.changed_numeric_vars.clone(),
        cost_bits,
        signature_hash,
    })
}

fn materialize_skeletons_into(
    task: &dyn AbstractNumericTask,
    generator: &mut AbstractOperatorGenerator,
    skeletons: &[AbstractOperatorSkeleton],
    finalizer: &mut AbstractOperatorFinalizer,
) -> Result<()> {
    let Some(first) = skeletons.first() else {
        return Ok(());
    };
    let transitions = compute_hash_effects_with_preconditions(
        task,
        generator,
        &first.op_preconditions,
        &first.ass_effects,
    )?;

    for skeleton in skeletons {
        debug_assert_eq!(skeleton.ass_effects, first.ass_effects);
        debug_assert_eq!(skeleton.op_preconditions, first.op_preconditions);
        for trans in &transitions {
            if let Some(candidate) = build_candidate_from_transition(skeleton, trans) {
                finalizer.push_candidate(&candidate);
            }
        }
    }
    Ok(())
}

#[allow(unused)]
fn arith_op_from_axiom(operator: &CalOperator) -> ArithOp {
    match operator {
        CalOperator::Sum => ArithOp::Add,
        CalOperator::Difference => ArithOp::Sub,
        CalOperator::Product => ArithOp::Mul,
        CalOperator::Division => ArithOp::Div,
    }
}

#[derive(Clone)]
pub struct AbstractOperatorGenerator {
    domain_mapping: DomainMapping,
    domain_sizes: Vec<usize>,
    numeric_domain_sizes: Vec<usize>,
    hash_multipliers: Vec<usize>,
    partitions: NumericPartitions,
    comparison_index: Option<ComparisonAxiomIndex>,
    comparison_trees: Vec<ComparisonTree>,
    comparisons_by_numeric_dep: Vec<Vec<usize>>,
    derived_prop_vars: HashSet<usize>,
    combine_labels: bool,
}

impl AbstractOperatorGenerator {
    pub fn new(
        task: &dyn AbstractNumericTask,
        domain_mapping: DomainMapping,
        domain_sizes: Vec<usize>,
        partitions: NumericPartitions,
        numeric_domain_sizes: Vec<usize>,
        combine_labels: bool,
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
        }

        let hash_multipliers = compute_hash_multipliers(&domain_sizes, &numeric_domain_sizes)?;

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

        let mut comparisons_by_numeric_dep: Vec<Vec<usize>> =
            vec![Vec::new(); task.numeric_variables().len()];
        for (tree_idx, tree) in comparison_trees.iter().enumerate() {
            for dep in tree.regular_numeric_var_dependencies(task) {
                ensure!(
                    dep < comparisons_by_numeric_dep.len(),
                    "comparison tree depends on numeric var {dep}, but only {} numeric vars exist",
                    comparisons_by_numeric_dep.len()
                );
                comparisons_by_numeric_dep[dep].push(tree_idx);
            }
        }

        let derived_prop_vars: HashSet<usize> = task
            .comparison_axioms()
            .iter()
            .map(|ax| ax.get_affected_var_id())
            .collect();

        Ok(Self {
            domain_mapping,
            domain_sizes,
            numeric_domain_sizes,
            hash_multipliers,
            partitions,
            comparison_index,
            comparison_trees,
            comparisons_by_numeric_dep,
            derived_prop_vars,
            combine_labels,
        })
    }

    /// Convenience constructor that mirrors numeric-fd's default setup when no CEGAR mapping
    /// exists yet: identity mapping for non-derived variables, and a 3-valued mapping
    /// (false/true/unknown) for comparison-axiom variables.
    pub fn new_with_identity_mapping(
        task: &dyn AbstractNumericTask,
        partitions: NumericPartitions,
        numeric_domain_sizes: Vec<usize>,
        combine_labels: bool,
    ) -> Result<Self> {
        let num_vars = task.get_num_variables();
        let derived_prop: HashSet<usize> = task
            .comparison_axioms()
            .iter()
            .map(|ax| ax.get_affected_var_id())
            .collect();

        let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);
        let mut domain_sizes: Vec<usize> = Vec::with_capacity(num_vars);
        for var_id in 0..num_vars {
            if derived_prop.contains(&var_id) {
                domain_mapping.push(vec![
                    COMPARISON_TRUE_VAL,
                    COMPARISON_FALSE_VAL,
                    COMPARISON_UNKNOWN_VAL,
                ]);
                domain_sizes.push(3);
            } else {
                let size = task
                    .get_variable_domain_size(var_id)
                    .map_err(|e| anyhow!(e.to_string()))
                    .with_context(|| format!("failed to get domain size for variable {var_id}"))?;
                let mapping: Vec<usize> = (0..size).collect();
                domain_mapping.push(mapping);
                domain_sizes.push(size);
            }
        }

        Self::new(
            task,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            combine_labels,
        )
    }

    pub fn hash_multipliers(&self) -> &[usize] {
        &self.hash_multipliers
    }

    pub fn domain_sizes(&self) -> &[usize] {
        &self.domain_sizes
    }

    pub fn domain_mapping(&self) -> &DomainMapping {
        &self.domain_mapping
    }

    pub fn numeric_domain_sizes(&self) -> &[usize] {
        &self.numeric_domain_sizes
    }

    pub fn build_abstract_operators(
        &mut self,
        task: &dyn AbstractNumericTask,
    ) -> Result<Vec<AbstractOperator>> {
        let mut finalizer =
            AbstractOperatorFinalizer::new(self.combine_labels, &self.hash_multipliers);
        for (concrete_op_id, op) in task.get_operators().iter().enumerate() {
            let skeletons = self.build_for_concrete_operator(task, op, concrete_op_id)?;
            materialize_skeletons_into(task, self, &skeletons, &mut finalizer)?;
        }

        Ok(finalizer.into_operators())
    }

    pub fn build_abstract_operators_with_cache(
        &mut self,
        task: &dyn AbstractNumericTask,
        cache: &mut IncrementalAbstractOperatorCache,
    ) -> Result<Vec<AbstractOperator>> {
        // Strategy: cache only the *skeletons* (and per-op dependency
        // metadata), not the candidates. Candidates are heavy
        // (`AbstractOperatorCandidate` carries 4 `Vec`s) and are essentially
        // always invalidated on numeric refinements anyway because partition
        // indices shift on `NumericPartitions::split_at`. Re-materializing
        // skeletons → operators directly via `materialize_skeletons_into`
        // (which pushes into `AbstractOperatorFinalizer` without an
        // intermediate candidate vec) saves 4 vec allocations per emitted
        // abstract operator on every CEGAR iteration.
        let mut finalizer =
            AbstractOperatorFinalizer::new(self.combine_labels, &self.hash_multipliers);

        if cache.per_operator.len() != task.get_operators().len() {
            cache.per_operator.clear();
            cache.per_operator.reserve(task.get_operators().len());
            for (concrete_op_id, op) in task.get_operators().iter().enumerate() {
                let skeletons = self.build_for_concrete_operator(task, op, concrete_op_id)?;
                cache.per_operator.push(CachedConcreteOperatorBuild {
                    dependencies: self.compute_operator_dependencies(op),
                    skeletons,
                });
            }
            // Drain the dirty state — the fresh build subsumes it.
            let _ = cache.take_dirty_state();
        } else {
            let (dirty_propositional_vars, dirty_numeric) = cache.take_dirty_state();
            if dirty_numeric {
                for (concrete_op_id, op) in task.get_operators().iter().enumerate() {
                    let entry = &mut cache.per_operator[concrete_op_id];
                    entry.dependencies = self.compute_operator_dependencies(op);
                    entry.skeletons = self.build_for_concrete_operator(task, op, concrete_op_id)?;
                }
            } else if !dirty_propositional_vars.is_empty() {
                for (concrete_op_id, op) in task.get_operators().iter().enumerate() {
                    let entry = &mut cache.per_operator[concrete_op_id];
                    let prop_dirty = entry
                        .dependencies
                        .intersects_propositional(&dirty_propositional_vars);
                    if prop_dirty {
                        entry.dependencies = self.compute_operator_dependencies(op);
                        entry.skeletons =
                            self.build_for_concrete_operator(task, op, concrete_op_id)?;
                    }
                }
            }
        }

        // Materialize skeletons → operators directly into the finalizer for
        // every concrete op every iteration. Numeric refinements invalidate
        // numeric partition transitions but not skeletons, so the skeleton
        // cache still serves its purpose for the propositional part.
        for entry in &cache.per_operator {
            materialize_skeletons_into(task, self, &entry.skeletons, &mut finalizer)?;
        }

        Ok(finalizer.into_operators())
    }

    fn build_for_concrete_operator(
        &mut self,
        task: &dyn AbstractNumericTask,
        op: &Operator,
        concrete_op_id: usize,
    ) -> Result<Vec<AbstractOperatorSkeleton>> {
        let (unconditional_effects, conditional_effects): (Vec<&Effect>, Vec<&Effect>) =
            op.effects().iter().partition(|e| e.conditions().is_empty());

        let (_unconditional_ass, conditional_ass): (
            Vec<&AssignmentEffect>,
            Vec<&AssignmentEffect>,
        ) = op
            .assignment_effects()
            .iter()
            .partition(|e| !e.is_conditional());

        ensure!(
            conditional_effects.is_empty() && conditional_ass.is_empty(),
            "numeric-fd parity: conditional propositional or numeric effects are unsupported in abstract operator generation"
        );

        for eff in op.assignment_effects() {
            let rhs_var_id = eff.var_id();
            ensure!(
                rhs_var_id < task.numeric_variables().len(),
                "assignment effect rhs var id out of bounds: {} >= {}",
                rhs_var_id,
                task.numeric_variables().len()
            );
            ensure!(
                task.numeric_variables()[rhs_var_id].get_type() == &NumericType::Constant,
                "numeric-fd parity: assignment effects require constant RHS, got {:?} for numeric var {}",
                task.numeric_variables()[rhs_var_id].get_type(),
                rhs_var_id
            );
        }

        let ass_effects = op.assignment_effects().clone();
        build_branch_for_operator(
            task,
            op,
            &unconditional_effects,
            &ass_effects,
            op.preconditions(),
            concrete_op_id,
            self,
        )
    }

    fn compute_operator_dependencies(&self, op: &Operator) -> OperatorDependencies {
        let mut dependencies = OperatorDependencies::default();
        for pre in op.preconditions() {
            dependencies.propositional_vars.insert(pre.var);
        }
        for eff in op.effects() {
            dependencies.propositional_vars.insert(eff.var_id());
            for cond in eff.conditions() {
                dependencies.propositional_vars.insert(cond.var);
            }
        }
        for eff in op.assignment_effects() {
            for cond in eff.conditions() {
                dependencies.propositional_vars.insert(cond.var);
            }
        }
        dependencies
    }

    #[inline]
    fn variable_is_trivial(&self, var_id: usize) -> bool {
        self.domain_sizes
            .get(var_id)
            .copied()
            .unwrap_or_else(|| panic!("variable_is_trivial: var_id {var_id} out of bounds"))
            <= 1
    }

    #[inline]
    fn abstract_value(&self, var_id: usize, concrete_value: usize) -> usize {
        let mapping = self
            .domain_mapping
            .get(var_id)
            .unwrap_or_else(|| panic!("abstract_value: var_id {var_id} out of bounds"));
        *mapping.get(concrete_value).unwrap_or_else(|| {
            panic!(
                "abstract_value: concrete value {concrete_value} out of bounds for variable {var_id}"
            )
        })
    }
}

#[allow(unused)]
fn format_abstract_fact(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator,
    fact: &ExplicitFact,
) -> String {
    let num_props = generator.domain_sizes.len();
    let var_id = fact.var;
    if var_id < num_props {
        let var_name = task.get_variable_name(var_id).unwrap_or("<unknown>");
        let concrete_size = task.get_variable_domain_size(var_id).unwrap_or(0);
        let mapping = generator.domain_mapping.get(var_id);
        let mut mapped_concretes: Vec<String> = Vec::new();
        for concrete_val in 0..concrete_size {
            let Some(abs_val) = mapping.and_then(|m| m.get(concrete_val)).copied() else {
                continue;
            };
            if abs_val == fact.value {
                mapped_concretes.push(
                    task.get_fact_name(&ExplicitFact::new(fact.var, concrete_val))
                        .to_string(),
                );
            }
        }
        if mapped_concretes.is_empty() {
            format!("var{var_id}({var_name})=abs{}", fact.value)
        } else {
            format!(
                "var{var_id}({var_name})=abs{} => [{}]",
                fact.value,
                mapped_concretes.join(" | ")
            )
        }
    } else {
        let numeric_var_id = var_id - num_props;
        let var_name = task
            .numeric_variables()
            .get(numeric_var_id)
            .map(|v| v.name())
            .unwrap_or("<unknown>");
        let interval = generator
            .partitions
            .partition_interval(numeric_var_id, fact.value);
        match interval {
            Some(iv) => format!(
                "num{numeric_var_id}({var_name})=p{}:{}",
                fact.value,
                utils::fmt_interval(iv)
            ),
            None => format!("num{numeric_var_id}({var_name})=p{}", fact.value),
        }
    }
}

#[allow(unused)]
fn normalize_preconditions(mut preconditions: Vec<ExplicitFact>) -> Option<Vec<ExplicitFact>> {
    preconditions.sort();
    let mut out: Vec<ExplicitFact> = Vec::with_capacity(preconditions.len());
    for pre in preconditions {
        if let Some(last) = out.last()
            && last.var == pre.var
        {
            if last.value != pre.value {
                return None;
            }
            continue;
        }
        out.push(pre);
    }
    Some(out)
}

#[allow(clippy::too_many_arguments)]
fn build_branch_for_operator(
    task: &dyn AbstractNumericTask,
    op: &Operator,
    effects: &[&Effect],
    ass_effects: &[AssignmentEffect],
    merged_preconditions: &[ExplicitFact],
    concrete_op_id: usize,
    generator: &mut AbstractOperatorGenerator,
) -> Result<Vec<AbstractOperatorSkeleton>> {
    let abstract_cost = abstract_operator_cost(task, op);
    let num_variables = task.get_num_variables();
    let mut precondition_on_var: Vec<Option<usize>> = vec![None; num_variables];
    let mut effect_on_var: Vec<Option<usize>> = vec![None; num_variables];

    let mut prev_pairs: Vec<ExplicitFact> = Vec::new();
    let mut pre_pairs: Vec<ExplicitFact> = Vec::new();
    let mut eff_pairs: Vec<ExplicitFact> = Vec::new();
    let mut effects_without_pre: Vec<ExplicitFact> = Vec::new();

    for pre in merged_preconditions {
        let var_id = pre.var;
        if generator.variable_is_trivial(var_id) {
            precondition_on_var[var_id] = Some(0);
            continue;
        }
        let abs_val = generator.abstract_value(var_id, pre.value);
        precondition_on_var[var_id] = Some(abs_val);
    }

    for eff in effects {
        let var_id = eff.var_id();
        if generator.variable_is_trivial(var_id) {
            continue;
        }

        debug_assert!(!generator.derived_prop_vars.contains(&eff.var_id()));

        let abs_val = generator.abstract_value(var_id, eff.value());
        let pre = precondition_on_var[var_id];
        if let Some(pre_val) = pre {
            if pre_val != abs_val {
                effect_on_var[var_id] = Some(abs_val);
                eff_pairs.push(ExplicitFact::new(var_id, abs_val));
            }
        } else {
            effects_without_pre.push(ExplicitFact::new(var_id, abs_val));
        }
    }

    for pre in merged_preconditions {
        let var_id = pre.var;
        if generator.variable_is_trivial(var_id) {
            continue;
        }
        let abs_val = generator.abstract_value(var_id, pre.value);
        if effect_on_var[var_id].is_some() {
            pre_pairs.push(ExplicitFact::new(var_id, abs_val));
        } else if !generator.derived_prop_vars.contains(&(var_id)) {
            prev_pairs.push(ExplicitFact::new(var_id, abs_val));
        }
    }

    for pre in merged_preconditions {
        let var_id = pre.var;
        if generator.variable_is_trivial(var_id) {
            continue;
        }
        if generator.derived_prop_vars.contains(&(var_id)) {
            continue;
        }
    }

    multiply_out_propositional(
        0,
        abstract_cost,
        &mut prev_pairs,
        &mut pre_pairs,
        &mut eff_pairs,
        &effects_without_pre,
        ass_effects,
        merged_preconditions,
        concrete_op_id,
        task,
        generator,
    )
}

fn abstract_operator_cost(task: &dyn AbstractNumericTask, op: &Operator) -> f64 {
    metric_operator_cost_from_initial_values(task, op)
}

#[allow(clippy::too_many_arguments, clippy::only_used_in_recursion)]
fn multiply_out_propositional(
    pos: usize,
    cost: f64,
    prev_pairs: &mut Vec<ExplicitFact>,
    pre_pairs: &mut Vec<ExplicitFact>,
    eff_pairs: &mut Vec<ExplicitFact>,
    effects_without_pre: &[ExplicitFact],
    ass_effects: &[AssignmentEffect],
    op_preconditions: &[ExplicitFact],
    concrete_op_id: usize,
    task: &dyn AbstractNumericTask,
    generator: &mut AbstractOperatorGenerator,
) -> Result<Vec<AbstractOperatorSkeleton>> {
    fn has_in_list_conflict(facts: &[ExplicitFact]) -> bool {
        facts
            .windows(2)
            .any(|w| w[0].var == w[1].var && w[0].value != w[1].value)
    }

    fn has_cross_list_conflict(a: &[ExplicitFact], b: &[ExplicitFact]) -> bool {
        let mut i = 0;
        let mut j = 0;
        while i < a.len() && j < b.len() {
            let va = a[i].var;
            let vb = b[j].var;
            if va < vb {
                i += 1;
            } else if vb < va {
                j += 1;
            } else {
                if a[i].value != b[j].value {
                    return true;
                }
                i += 1;
                j += 1;
            }
        }
        false
    }

    if pos == effects_without_pre.len() {
        if eff_pairs.is_empty() && ass_effects.is_empty() {
            return Ok(Vec::new());
        }

        let mut normalized_pre_pairs = pre_pairs.clone();
        let mut normalized_eff_pairs = eff_pairs.clone();
        let mut normalized_prev_pairs = prev_pairs.clone();
        normalized_pre_pairs.sort();
        normalized_eff_pairs.sort();
        normalized_prev_pairs.sort();
        normalized_pre_pairs.dedup();
        normalized_eff_pairs.dedup();
        normalized_prev_pairs.dedup();

        if has_in_list_conflict(&normalized_pre_pairs)
            || has_in_list_conflict(&normalized_eff_pairs)
            || has_in_list_conflict(&normalized_prev_pairs)
            || has_cross_list_conflict(&normalized_pre_pairs, &normalized_prev_pairs)
            || has_cross_list_conflict(&normalized_prev_pairs, &normalized_eff_pairs)
            || !transition_vars_match(&normalized_pre_pairs, &normalized_eff_pairs)
        {
            return Ok(Vec::new());
        }

        return Ok(vec![AbstractOperatorSkeleton {
            concrete_op_id,
            cost,
            prev_pairs: normalized_prev_pairs,
            pre_pairs: normalized_pre_pairs,
            eff_pairs: normalized_eff_pairs,
            ass_effects: ass_effects.to_vec(),
            op_preconditions: op_preconditions.to_vec(),
        }]);
    }

    let var_id = effects_without_pre[pos].var;
    let eff = effects_without_pre[pos].value;
    let domain_size = generator.domain_sizes[var_id];
    let mut out: Vec<AbstractOperatorSkeleton> = Vec::new();
    for i in 0..domain_size {
        if i != eff {
            pre_pairs.push(ExplicitFact::new(var_id, i));
            eff_pairs.push(ExplicitFact::new(var_id, eff));
        } else {
            prev_pairs.push(ExplicitFact::new(var_id, i));
        }

        out.extend(multiply_out_propositional(
            pos + 1,
            cost,
            prev_pairs,
            pre_pairs,
            eff_pairs,
            effects_without_pre,
            ass_effects,
            op_preconditions,
            concrete_op_id,
            task,
            generator,
        )?);

        if i != eff {
            pre_pairs.pop();
            eff_pairs.pop();
        } else {
            prev_pairs.pop();
        }
    }

    Ok(out)
}

#[allow(clippy::needless_range_loop)]
fn compute_hash_effects_with_preconditions(
    task: &dyn AbstractNumericTask,
    generator: &mut AbstractOperatorGenerator,
    op_preconditions: &[ExplicitFact],
    ass_effects: &[planners_sas::numeric::numeric_task::AssignmentEffect],
) -> Result<Vec<TransitionInfo>> {
    if generator.numeric_domain_sizes.is_empty() {
        return Ok(vec![TransitionInfo {
            source_partition_facts: Vec::new(),
            target_partition_facts: Vec::new(),
            prevail_facts: Vec::new(),
            changed_numeric_vars: Vec::new(),
        }]);
    }

    let num_props = generator.domain_sizes.len();

    // Enumerate transitions only for numeric variables whose partition value is
    // needed by this operator. Other refined numeric variables are implicit
    // frame variables: they are unchanged because the hash effect has no delta
    // on their dimensions.
    let num_numeric_vars = generator.numeric_domain_sizes.len();
    let mut effects_by_var: Vec<Vec<&planners_sas::numeric::numeric_task::AssignmentEffect>> =
        vec![Vec::new(); num_numeric_vars];
    let mut affected_numeric_vars: HashSet<usize> = HashSet::new();
    for eff in ass_effects {
        let v = eff.affected_var_id();
        debug_assert!(
            v < effects_by_var.len(),
            "assignment effect affected_var_id out of bounds: {v} >= {}",
            effects_by_var.len()
        );
        if v >= effects_by_var.len() {
            continue;
        }
        effects_by_var[v].push(eff);
        affected_numeric_vars.insert(v);
    }

    // Compute the set of numeric vars that this operator's preconditions
    // *transitively* depend on through comparison-axiom-derived propositional
    // variables. A refined numeric var that is neither affected by the
    // operator nor referenced by any of its comparison preconditions has no
    // observable role: the operator's effect on its abstract-state hash is
    // 0 along that dimension, and the match tree never queries the var. We
    // can therefore omit it from the cartesian product entirely (treating it
    // as a wildcard at the abstract-operator level), which can shrink the
    // transition count by orders of magnitude in domains like minecraft
    // where most concrete operators do not query most refined numerics.
    let mut needed_numeric_vars: HashSet<usize> = HashSet::new();
    if let Some(index) = &generator.comparison_index {
        for pre in op_preconditions {
            if !generator.derived_prop_vars.contains(&pre.var) {
                continue;
            }
            if let Some(tree) = index.comparison_tree(pre.var) {
                for dep in tree.regular_numeric_var_dependencies(task) {
                    needed_numeric_vars.insert(dep);
                }
            }
        }
    }

    let mut per_var: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
    for v in 0..num_numeric_vars {
        ensure!(
            v < task.numeric_variables().len(),
            "abstract operator numeric domain size/task variable mismatch: numeric_domain_sizes has {}, task has {} numeric variables",
            generator.numeric_domain_sizes.len(),
            task.numeric_variables().len()
        );
        if task.numeric_variables()[v].get_type() == &NumericType::Derived {
            continue;
        }
        let num_parts = generator.numeric_domain_sizes[v];
        if num_parts <= 1 {
            continue;
        }
        let effs = &effects_by_var[v];
        if let Some(eff) = effs.first() {
            let rhs = eff.var_id();
            let rhs_parts = generator
                .partitions
                .partitions(rhs)
                .map(|partitions| partitions.len())
                .ok_or_else(|| anyhow!("missing partitions for rhs numeric var {rhs}"))?;

            let mut pairs: HashSet<(usize, usize)> = HashSet::new();
            for src in 0..num_parts {
                for rhs_part in 0..rhs_parts {
                    let rhs_iv = generator
                        .partitions
                        .partition_interval(rhs, rhs_part)
                        .with_context(|| {
                            format!("missing partition interval for rhs var {rhs} part {rhs_part}")
                        })?;
                    let targets =
                        generator
                            .partitions
                            .reachable_partitions(v, src, eff.operation(), rhs_iv);
                    for tgt in targets {
                        pairs.insert((src, tgt));
                    }
                }
            }
            let mut transitions: Vec<(usize, usize)> = pairs.into_iter().collect();
            transitions.sort_unstable();
            per_var.push((v, transitions));
        } else if needed_numeric_vars.contains(&v) {
            // Unaffected but referenced via a comparison precondition:
            // enumerate identity transitions `p` -> `p` so the comparison
            // can be evaluated on the partition interval.
            let transitions: Vec<(usize, usize)> = (0..num_parts).map(|p| (p, p)).collect();
            per_var.push((v, transitions));
        }
        // Otherwise: skip — the operator is wildcarded on `v`. Its
        // hash effect on `v`'s dimension is 0, the match tree never queries
        // `v`, and no comparison precondition depends on `v`.
    }

    if per_var.is_empty() {
        return Ok(vec![TransitionInfo {
            source_partition_facts: Vec::new(),
            target_partition_facts: Vec::new(),
            prevail_facts: Vec::new(),
            changed_numeric_vars: Vec::new(),
        }]);
    }

    // Determine whether any of the operator's preconditions reference a
    // comparison-axiom-derived propositional variable. Combined with whether
    // the operator changes any numeric variable that feeds a comparison tree,
    // this lets us short-circuit the interval/cascade work on every combo.
    let op_has_comparison_preconditions = op_preconditions
        .iter()
        .any(|pre| generator.derived_prop_vars.contains(&pre.var));

    // Pre-decide the "this combo can possibly change a comparison's truth
    // value" flag at the level of the operator: any affected (changed)
    // numeric var that participates in a comparison tree is a trigger. We
    // re-check per combo (a combo may have src==tgt and thus no actual
    // change), but use this as the upper bound.
    let any_changed_var_affects_comparison = ass_effects.iter().any(|eff| {
        let v = eff.affected_var_id();
        generator
            .comparisons_by_numeric_dep
            .get(v)
            .is_some_and(|trees| !trees.is_empty())
    });

    // Backtracking enumeration. Following the C++ reference, we share the
    // partition-fact scratch buffers across recursive frames (push/pop) and
    // only clone into a `TransitionInfo` at the leaf. This avoids the
    // per-combo `Vec<(usize,usize,usize)>` materialization that the older
    // cartesian-product expansion did.
    let mut out: Vec<TransitionInfo> = Vec::new();
    let mut source_partition_facts: Vec<ExplicitFact> = Vec::new();
    let mut target_partition_facts: Vec<ExplicitFact> = Vec::new();
    let mut changed_numeric_vars: Vec<usize> = Vec::new();
    let mut combo_scratch: Vec<(usize, usize, usize)> = Vec::with_capacity(per_var.len());
    let mut source_intervals_buf: Vec<Interval> = Vec::new();
    let mut target_intervals_buf: Vec<Interval> = Vec::new();

    enumerate_partition_combos(
        task,
        generator,
        op_preconditions,
        &per_var,
        &affected_numeric_vars,
        num_props,
        op_has_comparison_preconditions,
        any_changed_var_affects_comparison,
        0,
        &mut source_partition_facts,
        &mut target_partition_facts,
        &mut changed_numeric_vars,
        &mut combo_scratch,
        &mut source_intervals_buf,
        &mut target_intervals_buf,
        &mut out,
    )?;

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn enumerate_partition_combos(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator,
    op_preconditions: &[ExplicitFact],
    per_var: &[(usize, Vec<(usize, usize)>)],
    affected_numeric_vars: &HashSet<usize>,
    num_props: usize,
    op_has_comparison_preconditions: bool,
    any_changed_var_affects_comparison: bool,
    pos: usize,
    source_partition_facts: &mut Vec<ExplicitFact>,
    target_partition_facts: &mut Vec<ExplicitFact>,
    changed_numeric_vars: &mut Vec<usize>,
    combo_scratch: &mut Vec<(usize, usize, usize)>,
    source_intervals_buf: &mut Vec<Interval>,
    target_intervals_buf: &mut Vec<Interval>,
    out: &mut Vec<TransitionInfo>,
) -> Result<()> {
    if pos == per_var.len() {
        // Leaf: emit a TransitionInfo for the current (source, target,
        // changed_vars) state of the scratch.

        // Fast path: when this operator has no comparison preconditions and
        // its changes can't affect any comparison axiom, no interval or
        // cascade work is needed — emit directly. Mirrors the C++ early-exit
        // before the optimistic filtering pass.
        if !op_has_comparison_preconditions && !any_changed_var_affects_comparison {
            let mut source_facts = source_partition_facts.clone();
            let mut target_facts = target_partition_facts.clone();
            source_facts.sort();
            source_facts.dedup();
            target_facts.sort();
            target_facts.dedup();
            let mut changed_vars = changed_numeric_vars.clone();
            changed_vars.sort_unstable();
            changed_vars.dedup();
            out.push(TransitionInfo {
                source_partition_facts: source_facts,
                target_partition_facts: target_facts,
                prevail_facts: Vec::new(),
                changed_numeric_vars: changed_vars,
            });
            return Ok(());
        }

        // Otherwise we need the source intervals to (a) prune via the
        // comparison-precondition index and (b) compute cascade facts.
        prepare_comparison_tree_inputs_for_combo_into(
            task,
            generator,
            combo_scratch,
            false,
            source_intervals_buf,
        )?;
        if op_has_comparison_preconditions
            && let Some(index) = &generator.comparison_index
            && op_preconditions
                .iter()
                .any(|pre| index.precondition_is_contradicted(pre, source_intervals_buf))
        {
            return Ok(());
        }

        let comparison_variants = if op_has_comparison_preconditions {
            prepare_comparison_tree_inputs_for_combo_into(
                task,
                generator,
                combo_scratch,
                true,
                target_intervals_buf,
            )?;
            compute_comparison_precondition_transition_variants(
                task,
                generator,
                op_preconditions,
                combo_scratch,
                source_intervals_buf,
                target_intervals_buf,
            )?
        } else {
            vec![ComparisonTransitionFacts::default()]
        };

        let mut changed_vars = changed_numeric_vars.clone();
        changed_vars.sort_unstable();
        changed_vars.dedup();
        for comparison_facts in comparison_variants {
            let mut source_facts = source_partition_facts.clone();
            let mut target_facts = target_partition_facts.clone();
            let mut prevail_facts = comparison_facts.prevail_facts;
            source_facts.extend(comparison_facts.source_facts);
            target_facts.extend(comparison_facts.target_facts);
            source_facts.sort();
            source_facts.dedup();
            target_facts.sort();
            target_facts.dedup();
            prevail_facts.sort();
            prevail_facts.dedup();
            out.push(TransitionInfo {
                source_partition_facts: source_facts,
                target_partition_facts: target_facts,
                prevail_facts,
                changed_numeric_vars: changed_vars.clone(),
            });
        }
        return Ok(());
    }

    let (var_id, transitions) = &per_var[pos];
    let var_id = *var_id;
    let abs_var_id = num_props + var_id;
    let var_is_affected = affected_numeric_vars.contains(&var_id);
    for &(src, tgt) in transitions {
        source_partition_facts.push(ExplicitFact::new(abs_var_id, src));
        target_partition_facts.push(ExplicitFact::new(abs_var_id, tgt));
        combo_scratch.push((var_id, src, tgt));
        let pushed_changed = if var_is_affected {
            changed_numeric_vars.push(var_id);
            true
        } else {
            false
        };

        enumerate_partition_combos(
            task,
            generator,
            op_preconditions,
            per_var,
            affected_numeric_vars,
            num_props,
            op_has_comparison_preconditions,
            any_changed_var_affects_comparison,
            pos + 1,
            source_partition_facts,
            target_partition_facts,
            changed_numeric_vars,
            combo_scratch,
            source_intervals_buf,
            target_intervals_buf,
            out,
        )?;

        source_partition_facts.pop();
        target_partition_facts.pop();
        combo_scratch.pop();
        if pushed_changed {
            changed_numeric_vars.pop();
        }
    }
    Ok(())
}

#[cfg(test)]
fn prepare_comparison_tree_inputs_for_combo(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator,
    combo: &[(usize, usize, usize)],
    use_target_partitions: bool,
) -> Result<Vec<Interval>> {
    let mut buf: Vec<Interval> = Vec::new();
    prepare_comparison_tree_inputs_for_combo_into(
        task,
        generator,
        combo,
        use_target_partitions,
        &mut buf,
    )?;
    Ok(buf)
}

/// Resize-and-overwrite variant: lets callers reuse a scratch buffer across
/// combos. Avoids per-combo `Vec<Interval>` allocations during the cartesian
/// product in `compute_hash_effects_with_preconditions`.
fn prepare_comparison_tree_inputs_for_combo_into(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator,
    combo: &[(usize, usize, usize)],
    use_target_partitions: bool,
    out: &mut Vec<Interval>,
) -> Result<()> {
    let initial_numeric_values = task.get_initial_numeric_state_values();
    let num_numeric = task.numeric_variables().len();
    out.clear();
    out.resize(num_numeric, Interval::new(0.0, 0.0, false, false));

    for (var_id, numeric_var) in task.numeric_variables().iter().enumerate() {
        if numeric_var.get_type() == &NumericType::Constant {
            out[var_id] = Interval::singleton(float_tolerance::canonicalize(
                initial_numeric_values[var_id],
            ));
        } else if numeric_var.get_type() != &NumericType::Derived {
            out[var_id] = Interval::unbounded();
        }
    }

    for (var_id, src, tgt) in combo {
        let partition_id = if use_target_partitions { *tgt } else { *src };
        let iv = generator
            .partitions
            .partition_interval(*var_id, partition_id)
            .with_context(|| {
                format!("missing partition interval for var {var_id} part {partition_id}")
            })?;
        out[*var_id] = iv;
    }

    fill_derived_numeric_intervals_from_comparison_trees(&generator.comparison_trees, out);

    for interval in out.iter_mut() {
        if interval.is_empty() {
            *interval = Interval::unbounded();
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Default)]
struct ComparisonTransitionFacts {
    source_facts: Vec<ExplicitFact>,
    target_facts: Vec<ExplicitFact>,
    prevail_facts: Vec<ExplicitFact>,
}

fn comparison_target_values(
    tree: &ComparisonTree,
    inputs: &[Interval],
    affected_var_id: usize,
    domain_mapping: &DomainMapping,
) -> Vec<usize> {
    let mut values = match tree.evaluate_interval(inputs) {
        Some(true) => vec![domain_mapping[affected_var_id][COMPARISON_TRUE_VAL]],
        Some(false) => vec![domain_mapping[affected_var_id][COMPARISON_FALSE_VAL]],
        None => vec![
            domain_mapping[affected_var_id][COMPARISON_TRUE_VAL],
            domain_mapping[affected_var_id][COMPARISON_FALSE_VAL],
        ],
    };
    values.sort_unstable();
    values.dedup();
    values
}

fn compute_comparison_precondition_transition_variants(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator,
    op_preconditions: &[ExplicitFact],
    combo: &[(usize, usize, usize)],
    source_inputs: &[Interval],
    target_inputs: &[Interval],
) -> Result<Vec<ComparisonTransitionFacts>> {
    let Some(index) = &generator.comparison_index else {
        return Ok(vec![ComparisonTransitionFacts::default()]);
    };

    let mut variants = vec![ComparisonTransitionFacts::default()];
    for precondition in op_preconditions {
        if !generator.derived_prop_vars.contains(&precondition.var) {
            continue;
        }
        let Some(tree) = index.comparison_tree(precondition.var) else {
            continue;
        };
        let source_value = match tree.evaluate_interval(source_inputs) {
            Some(false) => return Ok(Vec::new()),
            Some(true) | None => generator.domain_mapping[precondition.var][COMPARISON_TRUE_VAL],
        };
        let precondition_value = generator.abstract_value(precondition.var, precondition.value);
        if source_value != precondition_value {
            return Ok(Vec::new());
        }

        let dependency_changed = comparison_dependency_partition_changed(task, tree, combo);
        let target_values = if dependency_changed {
            comparison_target_values(
                tree,
                target_inputs,
                precondition.var,
                &generator.domain_mapping,
            )
        } else {
            vec![precondition_value]
        };

        let mut next = Vec::with_capacity(variants.len() * target_values.len().max(1));
        for variant in variants {
            if dependency_changed {
                for &target_value in &target_values {
                    let mut branch = variant.clone();
                    branch
                        .source_facts
                        .push(ExplicitFact::new(precondition.var, precondition_value));
                    branch
                        .target_facts
                        .push(ExplicitFact::new(precondition.var, target_value));
                    next.push(branch);
                }
            } else {
                let mut branch = variant;
                branch
                    .prevail_facts
                    .push(ExplicitFact::new(precondition.var, precondition_value));
                next.push(branch);
            }
        }
        variants = next;
    }

    Ok(variants)
}

#[allow(unused)]
fn comparison_dependency_partition_changed(
    task: &dyn AbstractNumericTask,
    tree: &ComparisonTree,
    combo: &[(usize, usize, usize)],
) -> bool {
    tree.regular_numeric_var_dependencies(task)
        .into_iter()
        .any(|var_id| {
            combo
                .iter()
                .any(|(changed_var_id, source_partition, target_partition)| {
                    *changed_var_id == var_id && source_partition != target_partition
                })
        })
}

fn compute_hash_multipliers(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
) -> Result<Vec<usize>> {
    let mut multipliers: Vec<usize> =
        Vec::with_capacity(domain_sizes.len() + numeric_domain_sizes.len());
    let mut num_states: usize = 1;

    for &size in domain_sizes {
        multipliers.push(num_states);
        num_states = num_states
            .checked_mul(size)
            .ok_or_else(|| anyhow!("hash multiplier overflow (too many abstract states)"))?;
    }
    for &parts in numeric_domain_sizes {
        multipliers.push(num_states);
        num_states = num_states
            .checked_mul(parts)
            .ok_or_else(|| anyhow!("hash multiplier overflow (too many abstract states)"))?;
    }
    Ok(multipliers)
}
