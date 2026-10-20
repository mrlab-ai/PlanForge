use super::*;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct LookupTable {
    pub(super) abstraction_id: usize,
    pub(super) distances: Vec<f64>,
    pub(super) unknown_value: f64,
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
                // As in max_abstraction_heuristic: an entry this table does not
                // have is an unknown lower bound, which is zero, not a proof
                // that the state is a dead end.
                unknown_value: 0.0,
            });
        }
    }

    pub(super) fn compute_heuristic(&self, abstract_state_ids: &[Option<usize>]) -> f64 {
        let mut sum = 0.0;
        for table in &self.lookup_tables {
            let Some(state_id) = abstract_state_ids
                .get(table.abstraction_id)
                .copied()
                .flatten()
            else {
                sum += table.unknown_value;
                continue;
            };
            let Some(&value) = table.distances.get(state_id) else {
                sum += table.unknown_value;
                continue;
            };
            if value.is_infinite() && value.is_sign_positive() {
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
        values.saturating_mul(std::mem::size_of::<f64>()) / 1024
    }
}

impl<'task> SaturatedCostPartitioningHeuristic<'task> {
    pub(super) fn compute_abstract_state_ids_into(
        &self,
        eval_state: &EvaluationState<'_, '_>,
        lookup_mask: Option<&[bool]>,
        ids: &mut Vec<Option<usize>>,
    ) -> Result<(), EvaluationError> {
        let components = self.components.borrow();
        if let Some(mask) = lookup_mask {
            assert_eq!(
                mask.len(),
                components.len(),
                "required component mask must match the abstraction collection"
            );
        }
        ids.clear();
        ids.resize(components.len(), None);

        let registry = eval_state.state_registry();
        let mut scratch = self.lookup_scratch.borrow_mut();
        eval_state.state().fill_state(registry, &mut scratch.prop);
        registry
            .fill_numeric_vars(eval_state.state(), &mut scratch.numeric)
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to read numeric state: {error:?}"
                ))
            })?;
        scratch.required_domain_ids.clear();
        scratch.required_domain_ids.extend(
            components
                .iter()
                .enumerate()
                .filter(|(id, component)| {
                    lookup_mask.is_none_or(|mask| mask[*id])
                        && matches!(component, AbstractionComponent::Domain(_))
                })
                .map(|(id, _)| id),
        );
        scratch.comparisons.clear();
        if scratch.required_domain_ids.len() > 1 {
            let first_id = scratch.required_domain_ids[0];
            let AbstractionComponent::Domain(heuristic) = &components[first_id] else {
                unreachable!("required domain component id must reference a domain abstraction")
            };
            let DomainAbstractionLookupScratch {
                numeric,
                comparisons,
                ..
            } = &mut *scratch;
            heuristic.fill_comparison_values_from_projected_state_values(numeric, comparisons)?;
        }

        for (component_id, component) in components.iter().enumerate() {
            if lookup_mask.is_some_and(|mask| !mask[component_id]) {
                continue;
            }
            ids[component_id] = match component {
                AbstractionComponent::Domain(heuristic) => Some(
                    heuristic.compute_abstract_hash_from_projected_state_values_inner(
                        &scratch.prop,
                        &scratch.numeric,
                        Some(&scratch.comparisons),
                    )?,
                ),
                AbstractionComponent::Cartesian(heuristic) => Some(
                    heuristic
                        .abstraction()
                        .hierarchy
                        .map_state(&scratch.prop, &scratch.numeric)
                        .map_err(|error| EvaluationError::ComputationFailed(error.to_string()))?,
                ),
                AbstractionComponent::PatternDatabase(pdb) => pdb
                    .abstract_state_id_from_source_state_values(&scratch.prop, &scratch.numeric)
                    .map_err(EvaluationError::ComputationFailed)?,
            };
        }

        Ok(())
    }

    /// The heuristic: the maximum over the retained cost partitionings
    /// (Seipp et al. 2020).
    pub(super) fn compute_max_h(state: &ScpState, ids: &[Option<usize>]) -> f64 {
        state
            .cp_heuristics
            .iter()
            .map(|cp| cp.compute_heuristic(ids))
            .fold(0.0, f64::max)
    }

    pub(super) fn compute_lookup_mask(state: &ScpState, component_count: usize) -> Vec<bool> {
        let mut ids: Vec<usize> = state
            .cp_heuristics
            .iter()
            .flat_map(|cp| cp.lookup_tables.iter().map(|table| table.abstraction_id))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        let mut mask = vec![false; component_count];
        for id in ids {
            *mask
                .get_mut(id)
                .expect("retained lookup table must reference an existing component") = true;
        }
        mask
    }
}

pub(super) fn lookup_distance(
    abstraction_id: usize,
    distances: &[f64],
    unknown_value: f64,
    abstract_state_ids: &[Option<usize>],
) -> f64 {
    abstract_state_ids
        .get(abstraction_id)
        .copied()
        .flatten()
        .and_then(|state_id| distances.get(state_id))
        .copied()
        .unwrap_or(unknown_value)
}

pub(super) fn max_abstraction_heuristic(
    state: &ScpState,
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
            lookup_distance(abstraction_id, distances, 0.0, abstract_state_ids)
        })
        .fold(0.0, f64::max)
}

/// What a lookup table contributes for a component without an abstract state
/// for the evaluated state: a pattern database state outside the database's
/// enumerated states, or a component this evaluation did not look up. Such a
/// component claims nothing, which is how cost-partitioning tables read it
/// (`unknown_value`), and order selection scores it the same way.
pub(super) const UNKNOWN_STATE_H: f64 = 0.0;

/// The goal distance of the evaluated state in one component's table;
/// infinite for an abstract dead end.
pub(super) fn current_h_for_distances(
    abstraction_id: usize,
    distances: &[f64],
    abstract_state_ids: &[Option<usize>],
) -> f64 {
    let Some(state_id) = abstract_state_ids[abstraction_id] else {
        return UNKNOWN_STATE_H;
    };
    *distances.get(state_id).unwrap_or_else(|| {
        panic!(
            "component {abstraction_id} state id {state_id} is outside its {} distances",
            distances.len()
        )
    })
}

pub(super) fn abstraction_heuristic_values(
    state: &ScpState,
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
