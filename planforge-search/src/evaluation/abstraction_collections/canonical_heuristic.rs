#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::collections::{BTreeSet, HashSet};

use planforge_sas::numeric_task::AbstractNumericTask;
use tracing::info;

use crate::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstraction;
use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;

use super::component::AbstractionComponent;

pub struct CanonicalAbstractionHeuristic<'task> {
    name: String,
    components: Vec<AbstractionComponent<'task>>,
    max_additive_subsets: Vec<Vec<usize>>,
    relevant_operator_ids: Vec<BTreeSet<usize>>,
    component_value_cache: RefCell<Vec<Option<f64>>>,
    state_value_cache: RefCell<Vec<Option<f64>>>,
    diagnostics_logged: RefCell<bool>,
}

impl<'task> CanonicalAbstractionHeuristic<'task> {
    pub fn new(
        name: Option<String>,
        task: &dyn AbstractNumericTask,
        components: Vec<AbstractionComponent<'task>>,
    ) -> Result<Self, String> {
        if components.is_empty() {
            return Err(
                "canonical abstraction heuristic requires at least one component".to_string(),
            );
        }
        let relevant_operator_ids = components
            .iter()
            .map(|component| component.relevant_operator_ids(task))
            .collect::<Result<Vec<_>, _>>()?;
        let subsets = compute_max_additive_subsets_from_relevant_operators(&relevant_operator_ids);
        Self::from_validated_parts(name, components, subsets, relevant_operator_ids)
    }

    pub fn with_explicit_subsets(
        name: Option<String>,
        task: &dyn AbstractNumericTask,
        components: Vec<AbstractionComponent<'task>>,
        max_additive_subsets: Vec<Vec<usize>>,
    ) -> Result<Self, String> {
        if components.is_empty() {
            return Err(
                "canonical abstraction heuristic requires at least one component".to_string(),
            );
        }
        let relevant_operator_ids = components
            .iter()
            .map(|component| component.relevant_operator_ids(task))
            .collect::<Result<Vec<_>, _>>()?;
        validate_additive_subsets(
            components.len(),
            &max_additive_subsets,
            &relevant_operator_ids,
        )?;
        Self::from_validated_parts(
            name,
            components,
            max_additive_subsets,
            relevant_operator_ids,
        )
    }

    fn from_validated_parts(
        name: Option<String>,
        components: Vec<AbstractionComponent<'task>>,
        max_additive_subsets: Vec<Vec<usize>>,
        relevant_operator_ids: Vec<BTreeSet<usize>>,
    ) -> Result<Self, String> {
        if max_additive_subsets.is_empty() {
            return Err("canonical abstraction heuristic has no additive subsets".to_string());
        }
        Ok(Self {
            name: name.unwrap_or_else(|| "canonical_abstractions".to_string()),
            components,
            max_additive_subsets,
            relevant_operator_ids,
            component_value_cache: RefCell::new(Vec::new()),
            state_value_cache: RefCell::new(Vec::new()),
            diagnostics_logged: RefCell::new(false),
        })
    }

    pub fn components(&self) -> &[AbstractionComponent<'task>] {
        &self.components
    }

    pub fn max_additive_subsets(&self) -> &[Vec<usize>] {
        &self.max_additive_subsets
    }

    fn cached_state_value(&self, state_id: usize) -> Option<f64> {
        self.state_value_cache
            .borrow()
            .get(state_id)
            .and_then(|value| *value)
    }

    fn cache_state_value(&self, state_id: usize, value: f64) {
        let mut cache = self.state_value_cache.borrow_mut();
        if cache.len() <= state_id {
            cache.resize(state_id + 1, None);
        }
        cache[state_id] = Some(value);
    }

    fn component_value(
        &self,
        component_id: usize,
        eval_state: &EvaluationState<'_, '_>,
        cache: &mut [Option<f64>],
    ) -> Result<f64, EvaluationError> {
        let slot = cache.get_mut(component_id).ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "invalid canonical abstraction component index {component_id}"
            ))
        })?;
        if let Some(value) = *slot {
            return Ok(value);
        }
        let component = self.components.get(component_id).ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "missing canonical abstraction component {component_id}"
            ))
        })?;
        let value = component.standalone_value(eval_state).map_err(|error| {
            EvaluationError::ComputationFailed(format!(
                "failed to evaluate {} component {component_id}: {error}",
                component.kind()
            ))
        })?;
        if value.is_nan() || value < 0.0 {
            return Err(EvaluationError::ComputationFailed(format!(
                "{} component {component_id} returned invalid heuristic value {value}",
                component.kind()
            )));
        }
        *slot = Some(value);
        Ok(value)
    }

    fn evaluate_subsets(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let mut cache = self.component_value_cache.borrow_mut();
        cache.clear();
        cache.resize(self.components.len(), None);
        let mut best = 0.0_f64;

        for subset in &self.max_additive_subsets {
            let mut sum = 0.0_f64;
            for &component_id in subset {
                let value = self.component_value(component_id, eval_state, &mut cache)?;
                if value.is_infinite() {
                    self.log_diagnostics_once(&cache);
                    return Ok(f64::INFINITY);
                }
                sum += value;
            }
            best = best.max(sum);
        }
        self.log_diagnostics_once(&cache);
        Ok(best)
    }

    fn log_diagnostics_once(&self, values: &[Option<f64>]) {
        let mut logged = self.diagnostics_logged.borrow_mut();
        if *logged {
            return;
        }
        *logged = true;
        info!(
            "canonical abstraction diagnostics: components={}, max_additive_subsets={}",
            self.components.len(),
            self.max_additive_subsets.len()
        );
        for (component_id, component) in self.components.iter().enumerate() {
            info!(
                "canonical component {component_id}: kind={}, states={}, h={}, relevant_ops={}",
                component.kind(),
                component.num_states(),
                values
                    .get(component_id)
                    .copied()
                    .flatten()
                    .unwrap_or(f64::NAN),
                self.relevant_operator_ids[component_id].len(),
            );
        }
    }
}

impl Heuristic for CanonicalAbstractionHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let state_id = eval_state.state().get_id();
        if let Some(value) = self.cached_state_value(state_id) {
            return Ok(value);
        }
        let value = self.evaluate_subsets(eval_state)?;
        self.cache_state_value(state_id, value);
        Ok(value)
    }

    fn proves_initial_state_optimal(&self) -> bool {
        self.components
            .iter()
            .any(AbstractionComponent::proves_initial_state_optimal)
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
}

/// Compatibility wrapper for domain-abstraction-only callers.
pub struct CanonicalDomainAbstractionHeuristic {
    inner: CanonicalAbstractionHeuristic<'static>,
}

impl CanonicalDomainAbstractionHeuristic {
    pub fn new(
        name: Option<String>,
        task: &dyn AbstractNumericTask,
        abstractions: Vec<DomainAbstraction>,
    ) -> Result<Self, String> {
        let components = abstractions
            .into_iter()
            .enumerate()
            .map(|(index, abstraction)| {
                AbstractionComponent::domain(
                    Some(format!("canonical_domain_abstraction_{index}")),
                    abstraction,
                )
            })
            .collect();
        Ok(Self {
            inner: CanonicalAbstractionHeuristic::new(
                name.or_else(|| Some("canonical_domain_abstractions".to_string())),
                task,
                components,
            )?,
        })
    }
}

impl Heuristic for CanonicalDomainAbstractionHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        self.inner.compute_heuristic(eval_state)
    }

    fn proves_initial_state_optimal(&self) -> bool {
        self.inner.proves_initial_state_optimal()
    }

    fn heuristic_name(&self) -> String {
        self.inner.heuristic_name()
    }
}

fn validate_additive_subsets(
    component_count: usize,
    subsets: &[Vec<usize>],
    relevant_operator_ids: &[BTreeSet<usize>],
) -> Result<(), String> {
    if subsets.is_empty() {
        return Err("canonical abstraction heuristic requires at least one subset".to_string());
    }
    for (subset_id, subset) in subsets.iter().enumerate() {
        if subset.is_empty() {
            return Err(format!("canonical subset {subset_id} is empty"));
        }
        let mut seen = HashSet::new();
        for &component_id in subset {
            if component_id >= component_count {
                return Err(format!(
                    "canonical subset {subset_id} references component {component_id}, but collection has {component_count} components"
                ));
            }
            if !seen.insert(component_id) {
                return Err(format!(
                    "canonical subset {subset_id} contains duplicate component {component_id}"
                ));
            }
        }
        for left_index in 0..subset.len() {
            for right_index in (left_index + 1)..subset.len() {
                let left = subset[left_index];
                let right = subset[right_index];
                if !are_operator_sets_additive(
                    &relevant_operator_ids[left],
                    &relevant_operator_ids[right],
                ) {
                    return Err(format!(
                        "canonical subset {subset_id} contains non-additive components {left} and {right}"
                    ));
                }
            }
        }
    }
    Ok(())
}

fn are_operator_sets_additive(left: &BTreeSet<usize>, right: &BTreeSet<usize>) -> bool {
    !left.iter().any(|operator_id| right.contains(operator_id))
}

fn compute_max_additive_subsets_from_relevant_operators(
    relevant_operators: &[BTreeSet<usize>],
) -> Vec<Vec<usize>> {
    let mut compatibility_graph = vec![Vec::new(); relevant_operators.len()];
    for left in 0..relevant_operators.len() {
        for right in (left + 1)..relevant_operators.len() {
            if are_operator_sets_additive(&relevant_operators[left], &relevant_operators[right]) {
                compatibility_graph[left].push(right);
                compatibility_graph[right].push(left);
            }
        }
    }

    let mut maximal_cliques = Vec::new();
    bron_kerbosch(
        &compatibility_graph,
        &mut Vec::new(),
        (0..relevant_operators.len()).collect(),
        Vec::new(),
        &mut maximal_cliques,
    );
    maximal_cliques.sort();
    maximal_cliques.dedup();
    maximal_cliques
}

fn bron_kerbosch(
    graph: &[Vec<usize>],
    current: &mut Vec<usize>,
    candidates: Vec<usize>,
    excluded: Vec<usize>,
    maximal_cliques: &mut Vec<Vec<usize>>,
) {
    if candidates.is_empty() && excluded.is_empty() {
        let mut clique = current.clone();
        clique.sort_unstable();
        maximal_cliques.push(clique);
        return;
    }

    let pivot = candidates
        .iter()
        .chain(excluded.iter())
        .copied()
        .max_by_key(|&vertex| graph[vertex].len());
    let pivot_neighbors: BTreeSet<_> = pivot
        .map(|vertex| graph[vertex].iter().copied().collect())
        .unwrap_or_default();
    let mut remaining_candidates = candidates.clone();
    let mut local_excluded = excluded;
    let vertices: Vec<_> = candidates
        .iter()
        .copied()
        .filter(|candidate| !pivot_neighbors.contains(candidate))
        .collect();

    for vertex in vertices {
        current.push(vertex);
        let neighbors: BTreeSet<_> = graph[vertex].iter().copied().collect();
        let next_candidates = remaining_candidates
            .iter()
            .copied()
            .filter(|candidate| neighbors.contains(candidate))
            .collect();
        let next_excluded = local_excluded
            .iter()
            .copied()
            .filter(|candidate| neighbors.contains(candidate))
            .collect();
        bron_kerbosch(
            graph,
            current,
            next_candidates,
            next_excluded,
            maximal_cliques,
        );
        current.pop();
        remaining_candidates.retain(|candidate| *candidate != vertex);
        local_excluded.push(vertex);
    }
}
