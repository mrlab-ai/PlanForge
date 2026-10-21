use super::*;
use crate::evaluation::abstractions::common::abstraction::{AbstractionMetadata, CegarAbstraction};
use crate::evaluation::abstractions::cost_partitioning::costs::{
    ExplicitSaturationGraph, RemainingCosts, SourceSetCostAllocation, saturate_labels,
    saturate_regions,
};
use crate::resource_limits::Deadline;

#[derive(Debug, Clone)]
pub struct CartesianAbstraction {
    pub hierarchy: CartesianRefinementHierarchy,
    pub distance_table: AbstractDistanceTable,
    pub transition_system: AbstractTransitionSystem,
    pub relevant_operator_ids: Vec<OperatorIndex>,
    pub abstract_source_sets: Vec<AbstractOperatorSourceSets>,
    pub metadata: AbstractionMetadata,
    /// Number of non-loop transitions built before optional standalone compaction.
    pub transition_count: usize,
}

impl CegarAbstraction for CartesianAbstraction {
    fn abstract_state(
        &self,
        propositional: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize> {
        self.hierarchy.map_state(propositional, numeric)
    }

    fn distance_table(&self) -> &AbstractDistanceTable {
        &self.distance_table
    }

    fn metadata(&self) -> &AbstractionMetadata {
        &self.metadata
    }

    fn metadata_mut(&mut self) -> &mut AbstractionMetadata {
        &mut self.metadata
    }

    fn discard_transition_data(&mut self) {
        self.transition_system.transitions = Vec::new();
        self.transition_system.backward = Vec::new();
        self.transition_system.cartesian_sets = Vec::new();
        self.abstract_source_sets = Vec::new();
    }
}

impl CartesianAbstraction {
    /// Label saturation of the abstraction under `operator_costs`; see
    /// [`saturate_labels`].
    pub(crate) fn saturate_labels(
        &self,
        operator_costs: &[f64],
        perimeter_state: Option<usize>,
        deadline: Deadline,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        saturate_labels(
            &mut ExplicitSaturationGraph::new(&self.transition_system),
            operator_costs,
            perimeter_state,
            deadline,
        )
    }

    /// Regional saturation of the abstraction under `remaining`; see
    /// [`saturate_regions`]. Every abstract operator is one transition, so
    /// its source sets are already those of the transition.
    pub(crate) fn saturate_regions(
        &self,
        remaining: &RemainingCosts,
        perimeter_state: Option<usize>,
        deadline: Deadline,
    ) -> Result<(Vec<f64>, SourceSetCostAllocation)> {
        saturate_regions(
            &self.transition_system,
            &self.abstract_source_sets,
            |_, source_set| Ok(source_set.clone()),
            remaining,
            perimeter_state,
            deadline,
        )
    }
}
