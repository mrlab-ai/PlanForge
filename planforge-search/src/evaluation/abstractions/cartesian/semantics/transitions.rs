//! Whether a concrete operator may induce an abstract transition, and the
//! source sets of the transitions it induces.

use anyhow::Result;
use planforge_sas::numeric_task::OperatorIndex;

use super::super::working::OperatorBitSet;
use super::CartesianSemantics;
use crate::evaluation::abstractions::common::cartesian_set::{
    CartesianSet, sorted_value_sets_overlap,
};
use crate::evaluation::abstractions::common::cegar::split::SplitDimension;

impl CartesianSemantics<'_> {
    pub(in super::super) fn operator_depends_on_split(
        &self,
        op_id: OperatorIndex,
        dimension: SplitDimension,
    ) -> bool {
        self.split_dependent_operators(dimension).contains(op_id)
    }

    pub(in super::super) fn split_dependent_operators(
        &self,
        dimension: SplitDimension,
    ) -> &OperatorBitSet {
        match dimension {
            SplitDimension::Propositional(var_id) => {
                &self.prop_split_dependent_operators[var_id.index()]
            }
            SplitDimension::Numeric(var_id) => {
                &self.numeric_split_dependent_operators[var_id.index()]
            }
        }
    }

    pub(in super::super) fn invariant_split_dimension_overlaps(
        &self,
        source: &CartesianSet,
        target: &CartesianSet,
        dimension: SplitDimension,
    ) -> bool {
        match dimension {
            SplitDimension::Propositional(var_id) => sorted_value_sets_overlap(
                &source.propositions()[var_id.index()],
                &target.propositions()[var_id.index()],
            ),
            SplitDimension::Numeric(var_id) => {
                source.numeric[var_id.index()].intersects(&target.numeric[var_id.index()])
            }
        }
    }

    pub(in super::super) fn may_transition_after_independent_split(
        &self,
        source: &CartesianSet,
        op_id: OperatorIndex,
        target: &CartesianSet,
        dimension: SplitDimension,
    ) -> Result<bool> {
        debug_assert!(!self.operator_depends_on_split(op_id, dimension));
        let result = self.invariant_split_dimension_overlaps(source, target, dimension);
        #[cfg(debug_assertions)]
        assert_eq!(
            result,
            self.cegar.regions().may_transition(source, op_id, target)?,
            "Cartesian split-dependency routing disagrees with full transition semantics for operator {op_id} and dimension {dimension:?}"
        );
        Ok(result)
    }

    pub(in super::super) fn split_dimension_may_transition(
        &self,
        source: &CartesianSet,
        op_id: OperatorIndex,
        target: &CartesianSet,
        dimension: SplitDimension,
    ) -> Result<bool> {
        Ok(match dimension {
            SplitDimension::Propositional(var_id) => self
                .cegar
                .regions()
                .propositional_may_transition(source, op_id, target, var_id),
            SplitDimension::Numeric(var_id) => self.cegar.regions().numeric_may_transition(
                source.numeric[var_id.index()],
                op_id,
                target.numeric[var_id.index()],
                var_id,
            )?,
        })
    }

    pub(in super::super) fn parent_loop_source_to_split_children(
        &self,
        source: &CartesianSet,
        op_id: OperatorIndex,
        targets: [&CartesianSet; 2],
        dimension: SplitDimension,
    ) -> Result<[bool; 2]> {
        let may_apply = self.cegar.regions().operator_may_apply(source, op_id)?;
        let mut result = [false; 2];
        if may_apply {
            for (index, target) in targets.into_iter().enumerate() {
                result[index] =
                    self.split_dimension_may_transition(source, op_id, target, dimension)?;
            }
        }
        #[cfg(debug_assertions)]
        for (index, target) in targets.into_iter().enumerate() {
            assert_eq!(
                result[index],
                self.cegar.regions().may_transition(source, op_id, target)?,
                "split-dimension routing disagrees with full transition semantics for parent-loop operator {op_id} and dimension {dimension:?}"
            );
        }
        Ok(result)
    }
}
