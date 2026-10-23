//! The abstract operators of a domain abstraction.
//!
//! A concrete operator induces the abstract operators of a cross product:
//! every propositional skeleton of it ([`propositional`]) joined with every
//! numeric partition combination it may lead between ([`numeric`]).
//! Abstract operators with equal preconditions, effects and cost may then
//! share one operator with several labels ([`labels`]).

#[cfg(test)]
mod tests;

mod labels;
mod numeric;
mod propositional;

use anyhow::Result;
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitValueIndex, NumericValue, OperatorIndex,
};

use super::function::DomainAbstractionFunction;
use super::layout::StateLayout;
use crate::resource_limits::Deadline;
use labels::{LabelCombiner, signature_hash};
use numeric::{NumericTransition, numeric_partition_transitions};
use propositional::{PropositionalSkeleton, PropositionalSkeletons};

/// The operation an expired generation deadline names.
const OPERATOR_GENERATION: &str = "abstract operator generation";

pub type DomainMapping = Vec<Vec<ExplicitValueIndex>>;

/// An operator of a domain abstraction: it leads from every abstract state
/// that satisfies `preconditions` to the state that differs from it by
/// `regression_offset`, which satisfies `regression_preconditions`.
#[derive(Debug, Clone, PartialEq)]
pub struct AbstractOperator {
    /// The labels: the concrete operators the operator stands for, ascending
    /// and unique.
    pub concrete_op_ids: Vec<OperatorIndex>,
    pub cost: NumericValue,
    /// The source state id minus the target state id of every transition
    /// the operator induces.
    pub regression_offset: isize,
    pub regression_preconditions: Vec<ExplicitFact>,
    pub preconditions: Vec<ExplicitFact>,
}

impl AbstractOperator {
    /// The operator `candidate` describes, labelled with its concrete
    /// operator.
    fn new(candidate: &AbstractOperatorCandidate, layout: &StateLayout) -> Self {
        let AbstractOperatorCandidate {
            prev_pairs,
            pre_pairs,
            eff_pairs,
            ..
        } = candidate;
        let mut preconditions: Vec<ExplicitFact> = pre_pairs.clone();
        preconditions.extend_from_slice(prev_pairs);
        preconditions.sort();
        debug_assert!(preconditions.windows(2).all(|w| w[0].var() != w[1].var()));

        let mut regression_preconditions: Vec<ExplicitFact> = prev_pairs.clone();
        regression_preconditions.extend_from_slice(eff_pairs);
        regression_preconditions.sort();
        debug_assert!(
            regression_preconditions
                .windows(2)
                .all(|w| w[0].var() != w[1].var())
        );

        debug_assert!(transition_vars_match(pre_pairs, eff_pairs));
        let regression_offset = pre_pairs
            .iter()
            .zip(eff_pairs)
            .map(|(pre, eff)| {
                (pre.value() as isize - eff.value() as isize)
                    * layout.multipliers()[pre.var()] as isize
            })
            .sum();

        Self {
            concrete_op_ids: vec![candidate.concrete_op_id],
            cost: candidate.cost,
            regression_offset,
            regression_preconditions,
            preconditions,
        }
    }
}

/// One abstract operator of one concrete operator, before label combining:
/// a propositional skeleton joined with a numeric transition. Its buffers are
/// reused from candidate to candidate.
#[derive(Debug, Clone, Default)]
struct AbstractOperatorCandidate {
    concrete_op_id: OperatorIndex,
    cost: NumericValue,
    prev_pairs: Vec<ExplicitFact>,
    pre_pairs: Vec<ExplicitFact>,
    eff_pairs: Vec<ExplicitFact>,
    /// The canonical bits of `cost`.
    cost_bits: u64,
    /// [`signature_hash`] of the fact lists and `cost_bits`.
    signature_hash: u64,
}

impl AbstractOperatorCandidate {
    /// Becomes the join of `skeleton` and `transition` for `concrete_op_id`.
    ///
    /// Propositional variables precede numeric ones in the abstraction's
    /// variable order, so the numeric facts go after the propositional ones
    /// and the lists stay ascending. The two parts share no variable, so the
    /// join is consistent because each part is.
    fn join(
        &mut self,
        concrete_op_id: OperatorIndex,
        cost: NumericValue,
        skeleton: &PropositionalSkeleton,
        transition: &NumericTransition,
    ) {
        let precedes = |propositional: &[ExplicitFact], numeric: &[ExplicitFact]| {
            propositional
                .last()
                .zip(numeric.first())
                .is_none_or(|(last, first)| last.var() < first.var())
        };
        assert!(
            precedes(&skeleton.pre_pairs, &transition.source_partition_facts)
                && precedes(&skeleton.eff_pairs, &transition.target_partition_facts),
            "propositional {skeleton:?} does not precede numeric {transition:?}"
        );
        self.prev_pairs.clone_from(&skeleton.prev_pairs);
        self.pre_pairs.clear();
        self.pre_pairs.extend_from_slice(&skeleton.pre_pairs);
        self.pre_pairs
            .extend_from_slice(&transition.source_partition_facts);
        self.eff_pairs.clear();
        self.eff_pairs.extend_from_slice(&skeleton.eff_pairs);
        self.eff_pairs
            .extend_from_slice(&transition.target_partition_facts);
        self.concrete_op_id = concrete_op_id;
        self.cost = cost;
        self.cost_bits = (cost).to_bits();
        self.signature_hash = signature_hash(
            &self.prev_pairs,
            &self.pre_pairs,
            &self.eff_pairs,
            self.cost_bits,
        );
    }
}

/// Whether `pre_pairs` and `eff_pairs` name the same variables in the same
/// order.
fn transition_vars_match(pre_pairs: &[ExplicitFact], eff_pairs: &[ExplicitFact]) -> bool {
    pre_pairs.len() == eff_pairs.len()
        && pre_pairs
            .iter()
            .zip(eff_pairs)
            .all(|(pre, eff)| pre.var() == eff.var())
}

/// The abstract operators of the abstraction `function` describes, with
/// labels combined if `combine_labels`.
pub(super) fn build_abstract_operators(
    function: &DomainAbstractionFunction,
    task: &dyn AbstractNumericTask,
    combine_labels: bool,
    deadline: Deadline,
) -> Result<Vec<AbstractOperator>> {
    let mut skeletons = PropositionalSkeletons::new(function, task);
    let mut combiner = LabelCombiner::new(combine_labels, function.layout());
    let mut candidate = AbstractOperatorCandidate::default();
    for (concrete_op_id, op) in task.get_operators().iter().enumerate() {
        deadline.ensure(OPERATOR_GENERATION)?;
        let concrete_op_id = OperatorIndex::from_usize(concrete_op_id);
        let op_skeletons = skeletons.of_operator(task, op);
        if op_skeletons.is_empty() {
            continue;
        }
        let transitions = numeric_partition_transitions(
            task,
            function,
            concrete_op_id,
            op.preconditions(),
            op.assignment_effects(),
            deadline,
        )?;
        let cost = function.operator_cost(concrete_op_id);
        for skeleton in &op_skeletons {
            deadline.ensure(OPERATOR_GENERATION)?;
            for transition in &transitions {
                candidate.join(concrete_op_id, cost, skeleton, transition);
                combiner.push(&candidate);
            }
        }
    }
    Ok(combiner.into_operators())
}
