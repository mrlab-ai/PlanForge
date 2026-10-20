use super::*;
use crate::evaluation::abstractions::common::abstraction::{AbstractionMetadata, CegarAbstraction};

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
        self.transition_system.forward = Vec::new();
        self.transition_system.cartesian_sets = Vec::new();
        self.abstract_source_sets = Vec::new();
    }
}
