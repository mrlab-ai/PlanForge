use super::*;
use crate::evaluation::abstractions::common::regions::RegionSemantics;

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
    /// The source set of every label of every abstract operator: the states
    /// of the operator's source region from which the label may lead into its
    /// target region.
    pub fn build_abstract_source_sets(
        &self,
        task: &dyn AbstractNumericTask,
        operators: &[AbstractOperator],
    ) -> Result<Vec<AbstractOperatorSourceSets>> {
        let regions = RegionSemantics::new(task)?;
        operators
            .iter()
            .enumerate()
            .map(|(abstract_op_id, operator)| {
                let source = self.cartesian_set_from_facts(task, &operator.preconditions);
                let target =
                    self.cartesian_set_from_facts(task, &operator.regression_preconditions);
                let labels = operator
                    .concrete_op_ids
                    .iter()
                    .map(|&concrete_op_id| {
                        let source_set = regions
                            .transition_source_set(&source, concrete_op_id, &target)?
                            .with_context(|| {
                                format!(
                                    "abstract operator {abstract_op_id} has an empty source set for concrete operator {concrete_op_id}"
                                )
                            })?;
                        Ok(LabelSourceSet {
                            concrete_op_id,
                            source: Arc::new(source_set),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(AbstractOperatorSourceSets { labels })
            })
            .collect()
    }
}
