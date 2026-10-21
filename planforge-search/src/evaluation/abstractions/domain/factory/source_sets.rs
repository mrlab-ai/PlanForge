use planforge_sas::numeric_task::OperatorIndex;

use super::*;
use crate::evaluation::abstractions::common::regions::RegionSemantics;

pub(super) fn precise_source_set_for_transition(
    transition: &AbstractTransition,
    concrete_op_id: OperatorIndex,
    source_cartesian_set: &CartesianSet,
    abstract_source_sets: &[AbstractOperatorSourceSets],
) -> Result<LabelSourceSet> {
    let abstract_regions = abstract_source_sets
        .get(transition.abstract_op_id)
        .with_context(|| {
            format!(
                "transition {} references missing abstract-operator region {}",
                transition.transition_id, transition.abstract_op_id
            )
        })?;
    let label_source_set = abstract_regions
        .labels
        .iter()
        .find(|source_set| source_set.concrete_op_id == concrete_op_id)
        .with_context(|| {
            format!(
                "transition {} abstract operator {} has no operator region for concrete operator {concrete_op_id}",
                transition.transition_id, transition.abstract_op_id
            )
        })?;
    let source_region = cartesian_set_intersection(
        source_cartesian_set,
        &label_source_set.source,
    )
    .with_context(|| {
        format!(
            "transition {} has an empty precise source operator region for concrete operator {concrete_op_id}",
            transition.transition_id
        )
    })?;
    Ok(LabelSourceSet {
        concrete_op_id,
        source: Arc::new(source_region),
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
