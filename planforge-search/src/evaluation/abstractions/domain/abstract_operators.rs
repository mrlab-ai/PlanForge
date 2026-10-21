#[cfg(test)]
mod tests;

use crate::resource_limits::Deadline;
use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};

use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentEffect, Effect, ExplicitFact, ExplicitValueIndex, NumericType,
    NumericValue, Operator, OperatorIndex, VariableIndex,
};
use planforge_sas::utils::float_tolerance;

use super::abstraction_numeric_var;
use super::comparison_inputs::fill_condition_intervals;
use super::factory::DomainAbstractionFactory;
use super::layout::StateLayout;
use crate::evaluation::abstractions::common::coordinates::active_comparison_dimensions;
use crate::evaluation::abstractions::common::coordinates::coordinate_image;
use planforge_sas::utils::interval::Interval;

/// The operation an expired generation deadline names.
const OPERATOR_GENERATION: &str = "abstract operator generation";

pub type DomainMapping = Vec<Vec<ExplicitValueIndex>>;

/// Per-(skeleton × transition) candidate. The four `Vec` fields are reused
/// across `build_candidate_from_transition` calls via a scratch instance held
/// in `build_abstract_operators` — minecraft CEGAR runs ~10⁷
/// candidate-builds and the per-call allocations dominated the build phase.
#[derive(Debug, Clone, Default)]
struct AbstractOperatorCandidate {
    concrete_op_id: OperatorIndex,
    cost: NumericValue,
    prev_pairs: Vec<ExplicitFact>,
    pre_pairs: Vec<ExplicitFact>,
    eff_pairs: Vec<ExplicitFact>,
    /// `(cost_bits, FNV+SplitMix64 hash of prev+pre+eff+cost)`.
    /// Precomputed once at candidate creation so the candidate can be matched
    /// against the grouping map in `push_candidate` without re-walking its fact
    /// slices.
    cost_bits: u64,
    signature_hash: u64,
}

#[derive(Debug, Clone)]
struct AbstractOperatorSkeleton {
    concrete_op_id: OperatorIndex,
    cost: NumericValue,
    prev_pairs: Vec<ExplicitFact>,
    pre_pairs: Vec<ExplicitFact>,
    eff_pairs: Vec<ExplicitFact>,
    ass_effects: Vec<AssignmentEffect>,
    op_preconditions: Vec<ExplicitFact>,
}

/// An operator of a domain abstraction: it leads from every abstract state
/// that satisfies `preconditions` to the state that differs from it by
/// `regression_offset`, which satisfies `regression_preconditions`.
#[derive(Debug, Clone, PartialEq)]
pub struct AbstractOperator {
    pub concrete_op_ids: Vec<OperatorIndex>,
    pub cost: NumericValue,
    /// The source state id minus the target state id of every transition
    /// the operator induces.
    pub regression_offset: isize,
    pub regression_preconditions: Vec<ExplicitFact>,
    pub preconditions: Vec<ExplicitFact>,
}

impl AbstractOperator {
    pub fn new(
        prev_pairs: &[ExplicitFact],
        pre_pairs: &[ExplicitFact],
        eff_pairs: &[ExplicitFact],
        cost: NumericValue,
        layout: &StateLayout,
        concrete_op_ids: Vec<OperatorIndex>,
    ) -> Self {
        let mut preconditions: Vec<ExplicitFact> = pre_pairs.to_vec();
        preconditions.extend_from_slice(prev_pairs);
        preconditions.sort();
        debug_assert!(preconditions.windows(2).all(|w| w[0].var() != w[1].var()));

        let mut regression_preconditions: Vec<ExplicitFact> = prev_pairs.to_vec();
        regression_preconditions.extend_from_slice(eff_pairs);
        regression_preconditions.sort();
        debug_assert!(
            regression_preconditions
                .windows(2)
                .all(|w| w[0].var() != w[1].var())
        );

        debug_assert_eq!(
            pre_pairs.len(),
            eff_pairs.len(),
            "abstract operator pre/eff pair mismatch: pre_pairs={pre_pairs:?} eff_pairs={eff_pairs:?}"
        );

        let mut regression_offset: isize = 0;
        for (pre, eff) in pre_pairs.iter().zip(eff_pairs.iter()) {
            debug_assert_eq!(
                pre.var(),
                eff.var(),
                "abstract operator transition var mismatch: pre={pre:?} eff={eff:?}"
            );
            let multiplier = layout.multipliers()[pre.var()] as isize;
            regression_offset += (pre.value() as isize - eff.value() as isize) * multiplier;
        }

        Self {
            concrete_op_ids,
            cost,
            regression_offset,
            regression_preconditions,
            preconditions,
        }
    }
}

/// One numeric combination a concrete operator may induce: the source and
/// target partitions of the refined numeric variables it reads or changes.
#[derive(Clone, Debug)]
struct TransitionInfo {
    source_partition_facts: Vec<ExplicitFact>,
    target_partition_facts: Vec<ExplicitFact>,
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
            *hash ^= fact.var() as u64;
            *hash = hash.wrapping_mul(FNV_PRIME);
            *hash ^= fact.value() as u64;
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
    std::hash::BuildHasherDefault<planforge_sas::utils::hashing::IdentityU64Hasher>,
>;

struct AbstractOperatorFinalizer<'a> {
    combine_labels: bool,
    layout: &'a StateLayout,
    operators: Vec<AbstractOperator>,
    /// Per-operator stored signatures, indexed parallel to `operators`.
    stored_signatures: Vec<StoredOperatorSignature>,
    /// Map from signature hash to the operator indices that share that hash.
    /// Most buckets hold a single index; collisions degrade gracefully via
    /// the small-vec scan and `StoredOperatorSignature::matches_candidate`.
    grouping: SignatureMap,
}

impl<'a> AbstractOperatorFinalizer<'a> {
    fn new(combine_labels: bool, layout: &'a StateLayout) -> Self {
        Self {
            combine_labels,
            layout,
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
                        let operator = &mut self.operators[idx as usize];
                        operator.concrete_op_ids.push(candidate.concrete_op_id);
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
                self.layout,
                vec![candidate.concrete_op_id],
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
            self.layout,
            vec![candidate.concrete_op_id],
        ));
    }

    fn into_operators(self) -> Vec<AbstractOperator> {
        self.operators
    }
}

fn has_cross_list_conflict(a: &[ExplicitFact], b: &[ExplicitFact]) -> bool {
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        let va = a[i].var();
        let vb = b[j].var();
        if va < vb {
            i += 1;
        } else if vb < va {
            j += 1;
        } else {
            if a[i].value() != b[j].value() {
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
            .all(|(pre, eff)| pre.var() == eff.var())
}

/// Fill `candidate` with the merged result of `skeleton` and `trans`. Returns
/// `true` on success; `false` when the skeleton and transition are
/// incompatible (in which case the candidate should not be pushed).
///
/// The candidate's `Vec` fields are *reused* across calls — the caller owns a
/// single scratch candidate and reuses its allocations on every invocation,
/// which (combined with the merge-instead-of-sort logic below) removes the
/// per-call `Vec::with_capacity(...)` overhead that dominated minecraft's
/// CEGAR build phase.
fn build_candidate_from_transition(
    skeleton: &AbstractOperatorSkeleton,
    trans: &TransitionInfo,
    candidate: &mut AbstractOperatorCandidate,
) -> bool {
    // INVARIANTS used below to avoid per-call sort + dedup
    // (they fire several million times during minecraft CEGAR):
    //
    // * `skeleton.{prev,pre,eff}_pairs` are sorted-by-var and unique-by-var
    //   (see `multiply_out_propositional` and `fact_value_for_var`'s
    //   binary_search precondition).
    // * `trans.source_partition_facts` and `trans.target_partition_facts`
    //   are sorted and deduplicated (see `enumerate_partition_combos`).
    //
    // We can therefore merge instead of sort+dedup: the merged outputs are
    // built in-order with O(n+m) work and no `Vec::sort()` cost.
    debug_assert!(
        skeleton
            .pre_pairs
            .windows(2)
            .all(|w| w[0].var() < w[1].var()),
        "skeleton.pre_pairs must be strictly ascending by var"
    );
    debug_assert!(
        skeleton
            .eff_pairs
            .windows(2)
            .all(|w| w[0].var() < w[1].var()),
        "skeleton.eff_pairs must be strictly ascending by var"
    );
    debug_assert!(
        skeleton
            .prev_pairs
            .windows(2)
            .all(|w| w[0].var() < w[1].var()),
        "skeleton.prev_pairs must be strictly ascending by var"
    );
    debug_assert!(
        trans
            .source_partition_facts
            .windows(2)
            .all(|w| w[0].var() <= w[1].var()),
        "TransitionInfo.source_partition_facts must be sorted by var"
    );
    debug_assert!(
        trans
            .target_partition_facts
            .windows(2)
            .all(|w| w[0].var() <= w[1].var()),
        "TransitionInfo.target_partition_facts must be sorted by var"
    );

    // Reuse the candidate's vecs as the merge output buffers.
    candidate.prev_pairs.clear();
    candidate.pre_pairs.clear();
    candidate.eff_pairs.clear();

    let extended_pre_pairs = &mut candidate.pre_pairs;
    let extended_eff_pairs = &mut candidate.eff_pairs;
    let extended_prev_pairs = &mut candidate.prev_pairs;

    let mut sk_pre = skeleton.pre_pairs.as_slice();
    let mut sk_eff = skeleton.eff_pairs.as_slice();
    let mut src = trans.source_partition_facts.as_slice();
    let mut tgt = trans.target_partition_facts.as_slice();

    while !sk_pre.is_empty() || !src.is_empty() {
        let push_sk = match (sk_pre.first(), src.first()) {
            (Some(a), Some(b)) => match a.var().cmp(&b.var()) {
                std::cmp::Ordering::Less => true,
                std::cmp::Ordering::Greater => false,
                std::cmp::Ordering::Equal => {
                    if a.value() != b.value() {
                        return false;
                    }
                    true
                }
            },
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        let var_id = if push_sk {
            let a = sk_pre.first().unwrap();
            extended_pre_pairs.push(*a);
            let v = a.var();
            sk_pre = &sk_pre[1..];
            v
        } else {
            let b = src.first().unwrap();
            extended_pre_pairs.push(*b);
            let v = b.var();
            src = &src[1..];
            v
        };
        while sk_pre.first().is_some_and(|f| f.var() == var_id) {
            sk_pre = &sk_pre[1..];
        }
        while src.first().is_some_and(|f| f.var() == var_id) {
            src = &src[1..];
        }
    }

    while !sk_eff.is_empty() || !tgt.is_empty() {
        let push_sk = match (sk_eff.first(), tgt.first()) {
            (Some(a), Some(b)) => match a.var().cmp(&b.var()) {
                std::cmp::Ordering::Less => true,
                std::cmp::Ordering::Greater => false,
                // The skeleton already pins this variable's effect: take its
                // entry and skip both sides for this variable.
                std::cmp::Ordering::Equal => true,
            },
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        let var_id = if push_sk {
            let a = sk_eff.first().unwrap();
            extended_eff_pairs.push(*a);
            let v = a.var();
            sk_eff = &sk_eff[1..];
            v
        } else {
            let b = tgt.first().unwrap();
            extended_eff_pairs.push(*b);
            let v = b.var();
            tgt = &tgt[1..];
            v
        };
        while sk_eff.first().is_some_and(|f| f.var() == var_id) {
            sk_eff = &sk_eff[1..];
        }
        while tgt.first().is_some_and(|f| f.var() == var_id) {
            tgt = &tgt[1..];
        }
    }

    extended_prev_pairs.extend_from_slice(&skeleton.prev_pairs);

    if has_cross_list_conflict(extended_pre_pairs, extended_prev_pairs)
        || has_cross_list_conflict(extended_prev_pairs, extended_eff_pairs)
        || !transition_vars_match(extended_pre_pairs, extended_eff_pairs)
    {
        return false;
    }

    let cost_bits = float_tolerance::canonical_bits(skeleton.cost);
    let signature_hash = compute_signature_hash(
        extended_prev_pairs,
        extended_pre_pairs,
        extended_eff_pairs,
        cost_bits,
    );
    candidate.concrete_op_id = skeleton.concrete_op_id;
    candidate.cost = skeleton.cost;
    candidate.cost_bits = cost_bits;
    candidate.signature_hash = signature_hash;
    true
}

fn materialize_skeletons_into(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator<'_>,
    skeletons: &[AbstractOperatorSkeleton],
    finalizer: &mut AbstractOperatorFinalizer<'_>,
    candidate_scratch: &mut AbstractOperatorCandidate,
    deadline: Deadline,
) -> Result<()> {
    deadline.ensure(OPERATOR_GENERATION)?;
    let Some(first) = skeletons.first() else {
        return Ok(());
    };
    let transitions = compute_hash_effects_with_preconditions(
        task,
        generator,
        first.concrete_op_id,
        &first.op_preconditions,
        &first.ass_effects,
        deadline,
    )?;

    for skeleton in skeletons {
        deadline.ensure(OPERATOR_GENERATION)?;
        debug_assert_eq!(skeleton.ass_effects, first.ass_effects);
        debug_assert_eq!(skeleton.op_preconditions, first.op_preconditions);
        for trans in &transitions {
            if build_candidate_from_transition(skeleton, trans, candidate_scratch) {
                finalizer.push_candidate(candidate_scratch);
            }
        }
    }
    Ok(())
}

/// The abstract operators of the abstraction `factory` describes, with
/// labels combined if `combine_labels`.
pub(super) fn build_abstract_operators(
    factory: &DomainAbstractionFactory,
    task: &dyn AbstractNumericTask,
    combine_labels: bool,
    deadline: Deadline,
) -> Result<Vec<AbstractOperator>> {
    let mut generator = AbstractOperatorGenerator::new(factory, task);
    let mut finalizer = AbstractOperatorFinalizer::new(combine_labels, factory.layout());
    let mut candidate_scratch = AbstractOperatorCandidate::default();
    for (concrete_op_id, op) in task.get_operators().iter().enumerate() {
        deadline.ensure(OPERATOR_GENERATION)?;
        let skeletons = generator.build_for_concrete_operator(
            task,
            op,
            OperatorIndex::from_usize(concrete_op_id),
        )?;
        materialize_skeletons_into(
            task,
            &generator,
            &skeletons,
            &mut finalizer,
            &mut candidate_scratch,
            deadline,
        )?;
    }
    Ok(finalizer.into_operators())
}

/// What building the abstract operators of one abstraction reads and reuses.
struct AbstractOperatorGenerator<'a> {
    factory: &'a DomainAbstractionFactory,
    /// Per-operator scratch buffers reused across `build_branch_for_operator`
    /// calls. Each entry is sized to `num_variables` and the buffer is
    /// cleared by overwriting at use sites — `vec![None; n]` on every
    /// operator was a noticeable allocator hot spot on minecraft (`_int_malloc`
    /// at ~9% of total CPU was dominated by `from_elem<Option<usize>>`).
    precondition_on_var_scratch: Vec<Option<ExplicitValueIndex>>,
    effect_on_var_scratch: Vec<Option<ExplicitValueIndex>>,
}

impl<'a> AbstractOperatorGenerator<'a> {
    fn new(factory: &'a DomainAbstractionFactory, task: &dyn AbstractNumericTask) -> Self {
        let num_variables = task.get_num_variables();
        Self {
            factory,
            precondition_on_var_scratch: vec![None; num_variables],
            effect_on_var_scratch: vec![None; num_variables],
        }
    }

    fn layout(&self) -> &StateLayout {
        self.factory.layout()
    }

    fn build_for_concrete_operator(
        &mut self,
        task: &dyn AbstractNumericTask,
        op: &Operator,
        concrete_op_id: OperatorIndex,
    ) -> Result<Vec<AbstractOperatorSkeleton>> {
        let effects: Vec<&Effect> = op.effects().iter().collect();
        build_branch_for_operator(
            task,
            &effects,
            op.assignment_effects(),
            op.preconditions(),
            concrete_op_id,
            self,
        )
    }

    fn variable_is_trivial(&self, var_id: VariableIndex) -> bool {
        self.layout().domain_sizes()[var_id.index()] <= 1
    }

    fn abstract_value(
        &self,
        var_id: VariableIndex,
        concrete_value: ExplicitValueIndex,
    ) -> ExplicitValueIndex {
        self.factory.domain_mapping()[var_id.index()][concrete_value.index()]
    }
}

fn build_branch_for_operator(
    task: &dyn AbstractNumericTask,
    effects: &[&Effect],
    ass_effects: &[AssignmentEffect],
    merged_preconditions: &[ExplicitFact],
    concrete_op_id: OperatorIndex,
    generator: &mut AbstractOperatorGenerator<'_>,
) -> Result<Vec<AbstractOperatorSkeleton>> {
    let abstract_cost = generator.factory.operator_cost(concrete_op_id);
    let domain_mapping = generator.factory.domain_mapping();
    // Reuse per-operator scratch buffers stored on the generator instead of
    // `vec![None; num_variables]` per call. We have to reset the slots
    // touched on the previous call; the bookkeeping below tracks them.
    let mut touched_pre: Vec<usize> = Vec::with_capacity(merged_preconditions.len());
    for pre in merged_preconditions {
        let var_id = pre.var();
        generator.precondition_on_var_scratch[var_id] =
            Some(if generator.variable_is_trivial(pre.var_index()) {
                ExplicitValueIndex::new(0)
            } else {
                domain_mapping[var_id][pre.value()]
            });
        touched_pre.push(var_id);
    }

    let mut touched_eff: Vec<VariableIndex> = Vec::with_capacity(effects.len());
    let mut facts = PropositionalFactScratch::default();
    let mut effects_without_pre: Vec<ExplicitFact> = Vec::new();

    for eff in effects {
        let var_id = eff.var_id();
        if generator.variable_is_trivial(var_id) {
            continue;
        }
        debug_assert!(!task.numeric_conditions().is_condition_var(eff.var_id()));

        let abs_val = domain_mapping[var_id.index()][eff.value().index()];
        if let Some(pre_val) = generator.precondition_on_var_scratch[var_id.index()] {
            if pre_val != abs_val {
                generator.effect_on_var_scratch[var_id.index()] = Some(abs_val);
                touched_eff.push(var_id);
                facts
                    .eff_pairs
                    .push(ExplicitFact::propositional(var_id.index(), abs_val.index()));
            }
        } else {
            effects_without_pre.push(ExplicitFact::propositional(var_id.index(), abs_val.index()));
        }
    }

    // Condition variables are never refined, so they are trivial here: a
    // comparison precondition only filters the numeric partition combinations
    // the operator applies in (`comparison_preconditions_admit_combo`).
    for pre in merged_preconditions {
        let var_id = pre.var_index();
        if generator.variable_is_trivial(var_id) {
            continue;
        }
        let abs_val = generator.abstract_value(var_id, pre.value_index());
        if generator.effect_on_var_scratch[var_id.index()].is_some() {
            facts.pre_pairs.push(ExplicitFact::in_namespace(
                pre.namespace(),
                var_id.index(),
                abs_val.index(),
            ));
        } else {
            facts
                .prev_pairs
                .push(ExplicitFact::propositional(var_id.index(), abs_val.index()));
        }
    }

    // Clear only the slots we touched, so subsequent calls start clean
    // without re-zeroing the whole vector.
    for &v in &touched_pre {
        generator.precondition_on_var_scratch[v] = None;
    }
    for &v in &touched_eff {
        generator.effect_on_var_scratch[v.index()] = None;
    }

    multiply_out_propositional(
        &PropositionalCrossProduct {
            effects_without_pre: &effects_without_pre,
            ass_effects,
            op_preconditions: merged_preconditions,
            concrete_op_id,
            cost: abstract_cost,
        },
        0,
        &mut facts,
        generator.layout(),
    )
}

/// One concrete operator's propositional effects that have no matching
/// precondition, together with everything a finished skeleton copies out. Fixed
/// for the whole cross-product, so the recursion carries a reference.
struct PropositionalCrossProduct<'a> {
    effects_without_pre: &'a [ExplicitFact],
    ass_effects: &'a [AssignmentEffect],
    op_preconditions: &'a [ExplicitFact],
    concrete_op_id: OperatorIndex,
    cost: NumericValue,
}

/// The three fact lists a propositional cross-product shares across recursive
/// frames: pushed on the way down, popped on the way back up, so one allocation
/// each serves the whole product.
#[derive(Default)]
struct PropositionalFactScratch {
    prev_pairs: Vec<ExplicitFact>,
    pre_pairs: Vec<ExplicitFact>,
    eff_pairs: Vec<ExplicitFact>,
}

fn multiply_out_propositional(
    operator: &PropositionalCrossProduct<'_>,
    pos: usize,
    facts: &mut PropositionalFactScratch,
    layout: &StateLayout,
) -> Result<Vec<AbstractOperatorSkeleton>> {
    let PropositionalCrossProduct {
        effects_without_pre,
        ass_effects,
        op_preconditions,
        concrete_op_id,
        cost,
    } = *operator;
    fn has_in_list_conflict(facts: &[ExplicitFact]) -> bool {
        facts
            .windows(2)
            .any(|w| w[0].var() == w[1].var() && w[0].value() != w[1].value())
    }

    if pos == effects_without_pre.len() {
        if facts.eff_pairs.is_empty() && ass_effects.is_empty() {
            return Ok(Vec::new());
        }

        let mut normalized_pre_pairs = facts.pre_pairs.clone();
        let mut normalized_eff_pairs = facts.eff_pairs.clone();
        let mut normalized_prev_pairs = facts.prev_pairs.clone();
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

    let namespace = effects_without_pre[pos].namespace();
    let var_id = effects_without_pre[pos].var();
    let eff = effects_without_pre[pos].value();
    let domain_size = layout.radix(var_id);
    let mut out: Vec<AbstractOperatorSkeleton> = Vec::new();
    for i in 0..domain_size {
        if i != eff {
            facts
                .pre_pairs
                .push(ExplicitFact::in_namespace(namespace, var_id, i));
            facts
                .eff_pairs
                .push(ExplicitFact::in_namespace(namespace, var_id, eff));
        } else {
            facts
                .prev_pairs
                .push(ExplicitFact::in_namespace(namespace, var_id, i));
        }

        out.extend(multiply_out_propositional(
            operator,
            pos + 1,
            facts,
            layout,
        )?);

        if i != eff {
            facts.pre_pairs.pop();
            facts.eff_pairs.pop();
        } else {
            facts.prev_pairs.pop();
        }
    }

    Ok(out)
}

fn compute_hash_effects_with_preconditions(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator<'_>,
    concrete_op_id: OperatorIndex,
    op_preconditions: &[ExplicitFact],
    ass_effects: &[AssignmentEffect],
    deadline: Deadline,
) -> Result<Vec<TransitionInfo>> {
    deadline.ensure(OPERATOR_GENERATION)?;
    let layout = generator.layout();
    let numeric_domain_sizes = layout.numeric_domain_sizes();
    let additive_numeric_views = generator.factory.additive_numeric_views();
    let num_props = layout.domain_sizes().len();

    // Enumerate every refined numeric coordinate needed by this operator.
    // Affected regular variables and exact additive views get source->target
    // transitions; needed unaffected coordinates get identity transitions.
    let mut assigned = vec![false; numeric_domain_sizes.len()];
    for eff in ass_effects {
        assigned[eff.affected_var_id().index()] = true;
    }

    // A refined numeric variable that the operator neither affects nor
    // reads through a comparison precondition has no observable role: the
    // operator's regression offset is 0 along it, and the match tree never
    // queries it. It is therefore left out of the cross product (a wildcard
    // at the abstract-operator level), which can shrink the transition count
    // by orders of magnitude in domains like minecraft where most concrete
    // operators do not read most refined numeric variables. The coordinates
    // of comparison preconditions are needed to filter combinations whose
    // source intervals contradict the precondition.
    let mut needed_numeric_vars: HashSet<VariableIndex> = HashSet::new();
    let mut op_has_comparison_preconditions = false;
    for pre in op_preconditions {
        if let Some(tree) = task.numeric_conditions().for_var(pre.var_index()) {
            op_has_comparison_preconditions = true;
            needed_numeric_vars.extend(active_comparison_dimensions(
                tree,
                numeric_domain_sizes,
                additive_numeric_views,
            ));
        }
    }

    let mut per_var: Vec<(VariableIndex, Vec<(usize, usize)>)> = Vec::new();
    for (v, &assigned) in assigned.iter().enumerate() {
        let var_id = VariableIndex::from_usize(v);
        let num_parts = numeric_domain_sizes[v];
        if num_parts <= 1 {
            continue;
        }
        if task.numeric_variables()[v].get_type() == &NumericType::Derived {
            let view = additive_numeric_views[v].as_ref().with_context(|| {
                format!("refined derived numeric variable {v} has no additive-view semantics")
            })?;
            if view.operator_delta(concrete_op_id).value().abs() >= float_tolerance::ABS_EPSILON {
                per_var.push((
                    var_id,
                    partition_transitions(task, generator, concrete_op_id, var_id),
                ));
            } else if needed_numeric_vars.contains(&var_id) {
                per_var.push((var_id, (0..num_parts).map(|part| (part, part)).collect()));
            }
            continue;
        }
        if assigned {
            per_var.push((
                var_id,
                partition_transitions(task, generator, concrete_op_id, var_id),
            ));
        } else if needed_numeric_vars.contains(&var_id) {
            // An unaffected refined regular numeric variable that a comparison
            // precondition reads: frame it with identity partition transitions
            // so the comparison evaluation sees the precise source partition.
            per_var.push((var_id, (0..num_parts).map(|p| (p, p)).collect()));
        }
    }

    if per_var.is_empty() {
        return Ok(vec![TransitionInfo {
            source_partition_facts: Vec::new(),
            target_partition_facts: Vec::new(),
        }]);
    }

    // Backtracking enumeration. Following the C++ reference, we share the
    // partition-fact scratch buffers across recursive frames (push/pop) and
    // only clone into a `TransitionInfo` at the leaf.
    let mut scratch = PartitionComboScratch {
        source_partition_facts: Vec::new(),
        target_partition_facts: Vec::new(),
        combo: Vec::with_capacity(per_var.len()),
        source_intervals: Vec::new(),
        transitions: Vec::new(),
    };

    enumerate_partition_combos(
        &PartitionComboEnumeration {
            task,
            generator,
            op_preconditions,
            per_var: &per_var,
            num_props,
            op_has_comparison_preconditions,
            deadline,
        },
        0,
        &mut scratch,
    )?;

    Ok(scratch.transitions)
}

/// The partition pairs `(source, target)` of coordinate `var_id` that
/// `concrete_op_id` may lead between, ascending: every partition the image of
/// a source partition intersects.
fn partition_transitions(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator<'_>,
    concrete_op_id: OperatorIndex,
    var_id: VariableIndex,
) -> Vec<(usize, usize)> {
    let partitions = generator.factory.partitions();
    let views = generator.factory.additive_numeric_views();
    partitions
        .partitions(var_id)
        .iter()
        .enumerate()
        .flat_map(|(source, &interval)| {
            let image = coordinate_image(task, views, interval, concrete_op_id, var_id);
            partitions
                .intersecting(var_id, image)
                .map(move |target| (source, target))
        })
        .collect()
}

/// One concrete operator's numeric cross-product, fixed for the whole
/// enumeration: the task and generator that turn partitions into intervals, the
/// operator's preconditions, the per-variable source-to-target transitions the
/// recursion walks, where numeric dimensions start in the state layout, and
/// whether a leaf needs comparison filtering at all.
struct PartitionComboEnumeration<'a> {
    task: &'a dyn AbstractNumericTask,
    generator: &'a AbstractOperatorGenerator<'a>,
    op_preconditions: &'a [ExplicitFact],
    per_var: &'a [(VariableIndex, Vec<(usize, usize)>)],
    num_props: usize,
    op_has_comparison_preconditions: bool,
    deadline: Deadline,
}

/// What one enumeration shares across recursive frames: the partition facts and
/// combo triples pushed on the way down and popped on the way back up, the
/// interval buffer each leaf reads, and the
/// transitions collected so far. Following the C++ reference, sharing these is
/// what keeps a combo from allocating.
struct PartitionComboScratch {
    source_partition_facts: Vec<ExplicitFact>,
    target_partition_facts: Vec<ExplicitFact>,
    combo: Vec<(VariableIndex, usize, usize)>,
    source_intervals: Vec<Interval>,
    transitions: Vec<TransitionInfo>,
}

fn enumerate_partition_combos(
    enumeration: &PartitionComboEnumeration<'_>,
    pos: usize,
    scratch: &mut PartitionComboScratch,
) -> Result<()> {
    let PartitionComboEnumeration {
        task,
        generator,
        op_preconditions,
        per_var,
        num_props,
        op_has_comparison_preconditions,
        deadline,
    } = *enumeration;
    if pos.is_multiple_of(4) {
        deadline.ensure(OPERATOR_GENERATION)?;
    }
    if pos == per_var.len() {
        // Leaf: emit a TransitionInfo for the source and target partition
        // facts the scratch holds. An operator with a comparison
        // precondition only applies in the combinations whose source
        // intervals admit it.
        if op_has_comparison_preconditions {
            prepare_comparison_tree_inputs_for_combo_into(
                task,
                generator,
                &scratch.combo,
                &mut scratch.source_intervals,
            );
            if !comparison_preconditions_admit_combo(
                task,
                op_preconditions,
                &scratch.source_intervals,
            ) {
                return Ok(());
            }
        }
        scratch.push_transition();
        return Ok(());
    }
    let (var_id, transitions) = &per_var[pos];
    let var_id = *var_id;
    let abs_var_id = abstraction_numeric_var(num_props, var_id);
    for &(src, tgt) in transitions {
        scratch
            .source_partition_facts
            .push(ExplicitFact::numeric_variable(abs_var_id.index(), src));
        scratch
            .target_partition_facts
            .push(ExplicitFact::numeric_variable(abs_var_id.index(), tgt));
        scratch.combo.push((var_id, src, tgt));

        enumerate_partition_combos(enumeration, pos + 1, scratch)?;

        scratch.source_partition_facts.pop();
        scratch.target_partition_facts.pop();
        scratch.combo.pop();
    }
    Ok(())
}

impl PartitionComboScratch {
    /// Record the combo the recursion currently holds. The three lists are
    /// pushed in ascending `var_id` order and each numeric var appears at most
    /// once, so no sort or dedup is needed here -- only the assertions that say
    /// so.
    fn push_transition(&mut self) {
        debug_assert!(
            self.source_partition_facts.windows(2).all(|w| w[0] <= w[1]),
            "source_partition_facts must be sorted by construction"
        );
        debug_assert!(
            self.target_partition_facts.windows(2).all(|w| w[0] <= w[1]),
            "target_partition_facts must be sorted by construction"
        );
        self.transitions.push(TransitionInfo {
            source_partition_facts: self.source_partition_facts.clone(),
            target_partition_facts: self.target_partition_facts.clone(),
        });
    }
}

/// Fills `out` with the interval of every numeric variable over the source
/// partitions of `combo`, which are its refined coordinates; see
/// [`fill_condition_intervals`]. `out` is reused across combos, which keeps
/// the cross product in `compute_hash_effects_with_preconditions` from
/// allocating.
fn prepare_comparison_tree_inputs_for_combo_into(
    task: &dyn AbstractNumericTask,
    generator: &AbstractOperatorGenerator<'_>,
    combo: &[(VariableIndex, usize, usize)],
    out: &mut Vec<Interval>,
) {
    let partitions = generator.factory.partitions();
    fill_condition_intervals(
        task,
        |var_id| {
            combo
                .iter()
                .find(|&&(combo_var, _, _)| combo_var == var_id)
                .map(|&(_, source_partition, _)| {
                    partitions.partition_interval(var_id, source_partition)
                })
        },
        out,
    );
}

/// Whether a partition combo survives this operator's comparison-axiom
/// preconditions, evaluated optimistically over the combo's source intervals.
///
/// The abstract operator carries no condition facts of its own, since
/// condition variables are never refined; this function is purely a filter.
fn comparison_preconditions_admit_combo(
    task: &dyn AbstractNumericTask,
    op_preconditions: &[ExplicitFact],
    source_inputs: &[Interval],
) -> bool {
    // Filter on the concrete TRUE/FALSE precondition value: a condition
    // variable has one abstract value, which says nothing.
    let conditions = task.numeric_conditions();
    !op_preconditions
        .iter()
        .filter(|precondition| conditions.is_condition_var(precondition.var_index()))
        .any(|precondition| conditions.precondition_is_contradicted(precondition, source_inputs))
}
