use super::*;
use crate::evaluation::abstractions::common::regions::{RegionSemantics, SOURCE_SETS};
use crate::evaluation::abstractions::common::transition_system::AbstractOperatorSourceSets;

/// `source_set` narrowed to `region`, the abstract state a transition of
/// its abstract operator leaves from.
pub(crate) fn source_set_within(
    region: &CartesianSet,
    source_set: &LabelSourceSet,
) -> Result<LabelSourceSet> {
    let source = cartesian_set_intersection(region, &source_set.source).with_context(|| {
        format!(
            "the source set of operator {} misses the abstract state a transition of it leaves",
            source_set.concrete_op_id
        )
    })?;
    Ok(LabelSourceSet {
        concrete_op_id: source_set.concrete_op_id,
        source: Arc::new(source),
    })
}

impl DomainAbstractionFactory {
    /// The source sets of every label of every abstract operator: the states
    /// of the operator's source region from which the label may lead into its
    /// target region.
    pub(crate) fn build_abstract_source_sets(
        &self,
        regions: &RegionSemantics<'_>,
        operators: &[AbstractOperator],
        deadline: Deadline,
    ) -> Result<Vec<AbstractOperatorSourceSets>> {
        let task = regions.task();
        operators
            .iter()
            .enumerate()
            .map(|(abstract_op_id, operator)| {
                if abstract_op_id.is_multiple_of(1024) {
                    deadline.ensure(SOURCE_SETS)?;
                }
                let source = self.cartesian_set_from_facts(task, &operator.preconditions);
                let target =
                    self.cartesian_set_from_facts(task, &operator.regression_preconditions);
                regions
                    .abstract_operator_source_sets(
                        &Arc::new(source),
                        &operator.concrete_op_ids,
                        &target,
                    )
                    .with_context(|| format!("abstract operator {abstract_op_id}"))
            })
            .collect()
    }
}
