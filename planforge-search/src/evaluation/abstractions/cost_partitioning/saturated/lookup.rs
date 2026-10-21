use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct LookupTable {
    pub(super) abstraction_id: usize,
    pub(super) distances: Vec<f64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct CostPartitioningHeuristic {
    pub(super) lookup_tables: Vec<LookupTable>,
    pub(super) goal_cover_goal_id: Option<usize>,
}

impl CostPartitioningHeuristic {
    pub(super) fn is_empty(&self) -> bool {
        self.lookup_tables.is_empty()
    }

    pub(super) fn add_h_values(&mut self, abstraction_id: usize, distances: Vec<f64>) {
        if distances.iter().any(|value| *value > 0.0) {
            self.lookup_tables.push(LookupTable {
                abstraction_id,
                distances,
            });
        }
    }

    pub(super) fn compute_heuristic(&self, abstract_state_ids: &[Option<usize>]) -> f64 {
        let mut sum = 0.0;
        for table in &self.lookup_tables {
            let value =
                current_h_for_distances(table.abstraction_id, &table.distances, abstract_state_ids);
            if value == f64::INFINITY {
                return f64::INFINITY;
            }
            sum += value;
        }
        sum
    }

    pub(super) fn estimate_size_in_kb(&self) -> usize {
        let values = self
            .lookup_tables
            .iter()
            .map(|table| table.distances.len())
            .sum::<usize>();
        values * std::mem::size_of::<f64>() / 1024
    }
}

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    /// The abstract state of `eval_state` in every component, or in those
    /// `lookup_mask` selects.
    pub(super) fn compute_abstract_state_ids_into(
        &self,
        components: &[AbstractionComponent<'_>],
        eval_state: &EvaluationState<'_, '_>,
        lookup_mask: Option<&[bool]>,
        ids: &mut Vec<Option<usize>>,
    ) -> Result<(), EvaluationError> {
        if let Some(mask) = lookup_mask {
            assert_eq!(
                mask.len(),
                components.len(),
                "required component mask must match the abstraction collection"
            );
        }
        ids.clear();
        ids.resize(components.len(), None);

        let mut values = self.state_values.borrow_mut();
        eval_state.fill_values(&mut values)?;
        for (component_id, component) in components.iter().enumerate() {
            if lookup_mask.is_none_or(|mask| mask[component_id]) {
                ids[component_id] = component.abstract_state_id(&values)?;
            }
        }

        Ok(())
    }
}

/// The heuristic: the maximum over the retained cost partitionings (Seipp et
/// al. 2020).
pub(super) fn compute_max_h(
    cp_heuristics: &[CostPartitioningHeuristic],
    ids: &[Option<usize>],
) -> f64 {
    cp_heuristics
        .iter()
        .map(|cp| cp.compute_heuristic(ids))
        .fold(0.0, f64::max)
}

/// Which of the `component_count` components a lookup table of
/// `cp_heuristics` reads.
pub(super) fn compute_lookup_mask(
    cp_heuristics: &[CostPartitioningHeuristic],
    component_count: usize,
) -> Vec<bool> {
    let mut mask = vec![false; component_count];
    for table in cp_heuristics.iter().flat_map(|cp| &cp.lookup_tables) {
        mask[table.abstraction_id] = true;
    }
    mask
}

pub(super) fn max_abstraction_heuristic(
    state: &OrderSelection<'_>,
    abstract_state_ids: &[Option<usize>],
) -> f64 {
    state
        .goal_distances_by_abstraction
        .iter()
        .enumerate()
        .map(|(abstraction_id, distances)| {
            // A miss contributes nothing, for every kind of abstraction.
            //
            // This used to be infinity for a prefix of component kinds, on the
            // reading that a domain abstraction covers every concrete state, so
            // a state it has no entry for cannot reach the goal. That reading is
            // wrong here: when the online path is not rebuilding a partitioning,
            // it computes abstract state ids only for abstractions selected by
            // `lookup_mask`, so every other one is legitimately absent.
            // Absent then meant infinite, the maximum swallowed every finite
            // estimate, and the search recorded a solvable state as a dead end.
            //
            // A distance that is present and infinite still means unreachable,
            // and still propagates, because that one was computed rather than
            // assumed.
            current_h_for_distances(abstraction_id, distances, abstract_state_ids)
        })
        .fold(0.0, f64::max)
}

/// What a lookup table contributes for a component without an abstract state
/// for the evaluated state: a pattern database state outside the database's
/// enumerated states, or a component this evaluation did not look up. Such a
/// component claims nothing, in cost partitionings and in order selection.
pub(super) const UNKNOWN_STATE_H: f64 = 0.0;

/// The goal distance of the evaluated state in component `abstraction_id`'s
/// table `distances`: infinite for an abstract dead end, and
/// [`UNKNOWN_STATE_H`] without an abstract state.
pub(super) fn current_h_for_distances(
    abstraction_id: usize,
    distances: &[f64],
    abstract_state_ids: &[Option<usize>],
) -> f64 {
    abstract_state_ids[abstraction_id].map_or(UNKNOWN_STATE_H, |state_id| distances[state_id])
}

pub(super) fn abstraction_heuristic_values(
    state: &OrderSelection<'_>,
    abstract_state_ids: &[Option<usize>],
) -> Vec<f64> {
    state
        .goal_distances_by_abstraction
        .iter()
        .enumerate()
        .map(|(abstraction_id, distances)| {
            current_h_for_distances(abstraction_id, distances, abstract_state_ids)
        })
        .collect()
}
