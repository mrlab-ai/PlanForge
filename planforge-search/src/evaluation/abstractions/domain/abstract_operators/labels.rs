//! Label combining: abstract operators with the same preconditions, effects
//! and cost become one operator whose labels are their concrete operators.

use std::collections::HashMap;

use planforge_sas::numeric_task::ExplicitFact;

use super::{AbstractOperator, AbstractOperatorCandidate};
use crate::evaluation::abstractions::domain::layout::StateLayout;

/// The fact lists and cost an abstract operator was built from, which is
/// what two candidates must share to be combined.
#[derive(Debug, Clone)]
struct StoredOperatorSignature {
    prev_pairs: Vec<ExplicitFact>,
    pre_pairs: Vec<ExplicitFact>,
    eff_pairs: Vec<ExplicitFact>,
    cost_bits: u64,
}

impl StoredOperatorSignature {
    fn matches(&self, candidate: &AbstractOperatorCandidate) -> bool {
        self.cost_bits == candidate.cost_bits
            && self.prev_pairs == candidate.prev_pairs
            && self.pre_pairs == candidate.pre_pairs
            && self.eff_pairs == candidate.eff_pairs
    }

    fn of(candidate: &AbstractOperatorCandidate) -> Self {
        Self {
            prev_pairs: candidate.prev_pairs.clone(),
            pre_pairs: candidate.pre_pairs.clone(),
            eff_pairs: candidate.eff_pairs.clone(),
            cost_bits: candidate.cost_bits,
        }
    }
}

/// The 64-bit hash label combining groups candidates by: the prev, pre and
/// eff fact lists and `cost_bits`, mixed FNV-1a style per fact and finished
/// with the SplitMix64 finalizer, as the state registry hashes states.
#[inline]
pub(super) fn signature_hash(
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

/// The operators that share a signature hash, by index. The keys are already
/// hashes, so the map passes them through.
type SignatureMap = HashMap<
    u64,
    Vec<u32>,
    std::hash::BuildHasherDefault<planforge_sas::utils::hashing::IdentityU64Hasher>,
>;

/// Turns candidates into abstract operators, combining the labels of
/// candidates with equal signatures if `combine_labels`.
pub(super) struct LabelCombiner<'a> {
    combine_labels: bool,
    layout: &'a StateLayout,
    operators: Vec<AbstractOperator>,
    /// The signature of every operator, parallel to `operators`; empty unless
    /// `combine_labels`.
    signatures: Vec<StoredOperatorSignature>,
    grouping: SignatureMap,
}

impl<'a> LabelCombiner<'a> {
    pub(super) fn new(combine_labels: bool, layout: &'a StateLayout) -> Self {
        Self {
            combine_labels,
            layout,
            operators: Vec::new(),
            signatures: Vec::new(),
            grouping: SignatureMap::default(),
        }
    }

    /// Adds `candidate` as a label of the operator with its signature, or as
    /// a new operator. Candidates arrive in ascending concrete operator
    /// order, so the labels of every operator are ascending and unique.
    pub(super) fn push(&mut self, candidate: &AbstractOperatorCandidate) {
        if self.combine_labels {
            let bucket = self.grouping.entry(candidate.signature_hash).or_default();
            if let Some(&op_id) = bucket
                .iter()
                .find(|&&op_id| self.signatures[op_id as usize].matches(candidate))
            {
                let labels = &mut self.operators[op_id as usize].concrete_op_ids;
                assert!(
                    labels.last() < Some(&candidate.concrete_op_id),
                    "label {} combined into an operator with labels {labels:?}",
                    candidate.concrete_op_id
                );
                labels.push(candidate.concrete_op_id);
                return;
            }
            bucket.push(
                u32::try_from(self.operators.len()).expect("abstract operator ids fit in u32"),
            );
            self.signatures.push(StoredOperatorSignature::of(candidate));
        }
        self.operators
            .push(AbstractOperator::new(candidate, self.layout));
    }

    pub(super) fn into_operators(self) -> Vec<AbstractOperator> {
        self.operators
    }
}
