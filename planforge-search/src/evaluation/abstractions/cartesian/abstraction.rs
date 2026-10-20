use super::*;

#[derive(Debug, Clone)]
pub struct CartesianAbstractionMetadata {
    pub solved_by_self: bool,
    pub abstraction_use: AbstractionUse,
    pub stop_reason: CegarStopReason,
    pub pending_flaw: Option<String>,
    pub refinements: usize,
    /// Set when the abstraction is a member of a goal-focused collection.
    pub collection_member: Option<CollectionMembership>,
    pub split_kind: SplitKind,
    pub split_selection_rank: Option<usize>,
    pub concrete_plan_operator_ids: Option<Vec<OperatorIndex>>,
    /// Number of non-loop transitions built before optional standalone compaction.
    pub transition_count: usize,
}

#[derive(Debug, Clone)]
pub struct CartesianAbstraction {
    pub hierarchy: CartesianRefinementHierarchy,
    pub distance_table: AbstractDistanceTable,
    pub transition_system: AbstractTransitionSystem,
    pub relevant_operator_ids: Vec<OperatorIndex>,
    pub abstract_source_sets: Vec<AbstractOperatorSourceSets>,
    pub metadata: CartesianAbstractionMetadata,
}

impl CartesianAbstraction {
    pub fn num_states(&self) -> usize {
        self.distance_table.distances.len()
    }

    pub fn abstract_state_id(
        &self,
        propositional: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize> {
        self.hierarchy.map_state(propositional, numeric)
    }

    pub fn discard_transition_data(&mut self) {
        self.transition_system.transitions = Vec::new();
        self.transition_system.backward = Vec::new();
        self.transition_system.forward = Vec::new();
        self.transition_system.cartesian_sets = Vec::new();
        self.abstract_source_sets = Vec::new();
    }
}
