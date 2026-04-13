use std::cell::RefCell;
use std::collections::BTreeSet;

use planners_sas::numeric::numeric_task::AbstractNumericTask;

use crate::numeric::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::numeric::evaluation::heuristic::Heuristic;

use super::domain_abstraction_generator::DomainAbstraction;
use super::domain_abstraction_heuristic::DomainAbstractionHeuristic;

#[derive(Debug, Clone)]
pub struct CanonicalDomainAbstractionHeuristic {
    name: String,
    heuristics: Vec<DomainAbstractionHeuristic>,
    max_additive_subsets: Vec<Vec<usize>>,
    state_value_cache: RefCell<Vec<Option<f64>>>,
}

impl CanonicalDomainAbstractionHeuristic {
    pub fn new(
        name: Option<String>,
        task: &dyn AbstractNumericTask,
        abstractions: Vec<DomainAbstraction>,
    ) -> Result<Self, String> {
        let heuristics: Vec<_> = abstractions
            .into_iter()
            .enumerate()
            .map(|(index, abstraction)| {
                DomainAbstractionHeuristic::new(
                    Some(format!("canonical_domain_abstraction_{index}")),
                    abstraction,
                )
            })
            .collect();

        let mut relevant_operators = Vec::with_capacity(heuristics.len());
        for heuristic in &heuristics {
            relevant_operators.push(compute_relevant_operator_ids(task, heuristic.abstraction())?);
        }

        Ok(Self::with_explicit_subsets(
            name,
            heuristics,
            compute_max_additive_subsets_from_relevant_operators(&relevant_operators),
        ))
    }

    pub fn with_explicit_subsets(
        name: Option<String>,
        heuristics: Vec<DomainAbstractionHeuristic>,
        max_additive_subsets: Vec<Vec<usize>>,
    ) -> Self {
        Self {
            name: name.unwrap_or_else(|| "canonical_domain_abstractions".to_string()),
            heuristics,
            max_additive_subsets,
            state_value_cache: RefCell::new(Vec::new()),
        }
    }

    pub fn heuristics(&self) -> &[DomainAbstractionHeuristic] {
        &self.heuristics
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

    fn evaluate_subsets(&self, eval_state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        if self.max_additive_subsets.is_empty() {
            return Ok(0.0);
        }

        let mut abstraction_value_cache = vec![None; self.heuristics.len()];
        let mut best = 0.0_f64;

        for subset in &self.max_additive_subsets {
            let mut sum = 0.0_f64;
            for &abstraction_id in subset {
                let value = if let Some(value) = abstraction_value_cache
                    .get(abstraction_id)
                    .and_then(|cached| *cached)
                {
                    value
                } else {
                    let Some(heuristic) = self.heuristics.get(abstraction_id) else {
                        return Err(EvaluationError::InvalidState(format!(
                            "invalid canonical abstraction index {abstraction_id}"
                        )));
                    };
                    let value = heuristic.compute_heuristic(eval_state)?;
                    let Some(cache_slot) = abstraction_value_cache.get_mut(abstraction_id) else {
                        return Err(EvaluationError::InvalidState(format!(
                            "invalid canonical abstraction cache index {abstraction_id}"
                        )));
                    };
                    *cache_slot = Some(value);
                    value
                };

                if value.is_infinite() && value.is_sign_positive() {
                    return Ok(f64::INFINITY);
                }

                sum += value;
            }
            best = best.max(sum);
        }

        Ok(best)
    }
}

impl Heuristic for CanonicalDomainAbstractionHeuristic {
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

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
}

fn compute_relevant_operator_ids(
    task: &dyn AbstractNumericTask,
    abstraction: &DomainAbstraction,
) -> Result<BTreeSet<usize>, String> {
    let mut generator = abstraction
        .factory
        .make_operator_generator(task, abstraction.combine_labels)
        .map_err(|error| {
            format!(
                "failed to build operator generator for canonical domain abstraction: {error:#}"
            )
        })?;
    let operators = generator.build_abstract_operators(task).map_err(|error| {
        format!(
            "failed to build abstract operators for canonical domain abstraction: {error:#}"
        )
    })?;

    Ok(operators
        .into_iter()
        .flat_map(|operator| operator.concrete_op_ids.into_iter())
        .collect())
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

#[cfg(test)]
mod tests {
    use planners_sas::numeric::axioms::AxiomEvaluator;
    use planners_sas::numeric::numeric_task::{ExplicitVariable, Fact, Metric, NumericRootTask};
    use planners_sas::numeric::state_registry::StateRegistry;
    use planners_sas::numeric::utils::int_packer::IntDoublePacker;

    use crate::numeric::evaluation::evaluator::EvaluationState;

    use super::super::domain_abstraction::NumericPartitions;
    use super::super::domain_abstraction_factory::{AbstractDistanceTable, DomainAbstractionFactory};
    use super::super::domain_abstraction_generator::compute_hash_multipliers;
    use super::*;

    fn simple_var(name: &str) -> ExplicitVariable {
        ExplicitVariable::new(
            2,
            name.to_string(),
            vec![format!("{name}=0"), format!("{name}=1")],
            -1,
            1,
        )
    }

    fn simple_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![simple_var("p"), simple_var("q")],
            vec![],
            vec![Fact::new(0, 1)],
            vec![],
            vec![0, 0],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            (0, 0),
        )
    }

    fn make_abstraction(task: &NumericRootTask, distances: Vec<f64>) -> DomainAbstraction {
        let factory = DomainAbstractionFactory::new(
            task,
            vec![vec![0, 1], vec![0, 1]],
            vec![2, 2],
            NumericPartitions::with_partitions(vec![]),
            vec![],
        )
        .unwrap();
        let hash_multipliers =
            compute_hash_multipliers(factory.domain_sizes(), factory.numeric_domain_sizes())
                .unwrap();

        DomainAbstraction {
            factory,
            distance_table: AbstractDistanceTable {
                distances,
                generating_op_ids: vec![None; 4],
                initial_state_hash: 0,
                goal_facts: vec![],
                hash_multipliers: hash_multipliers.clone(),
                numeric_domain_sizes: vec![],
            },
            hash_multipliers,
            combine_labels: false,
        }
    }

    #[test]
    fn computes_max_additive_subsets_from_relevant_operators() {
        let subsets = compute_max_additive_subsets_from_relevant_operators(&[
            [0usize].into_iter().collect(),
            [1usize].into_iter().collect(),
            [0usize].into_iter().collect(),
        ]);

        assert_eq!(subsets, vec![vec![0, 1], vec![1, 2]]);
    }

    #[test]
    fn canonical_domain_abstraction_uses_explicit_subsets() {
        let task = simple_task();
        let packer = IntDoublePacker::from_task(&task);
        let axiom_evaluator = AxiomEvaluator::new(&task, &packer);
        let mut registry = StateRegistry::new(&task, &packer, &axiom_evaluator);
        let initial_state = registry.get_initial_state();

        let heuristic = CanonicalDomainAbstractionHeuristic::with_explicit_subsets(
            None,
            vec![
                DomainAbstractionHeuristic::new(
                    Some("da0".to_string()),
                    make_abstraction(&task, vec![2.0, 0.0, 0.0, 0.0]),
                ),
                DomainAbstractionHeuristic::new(
                    Some("da1".to_string()),
                    make_abstraction(&task, vec![3.0, 0.0, 0.0, 0.0]),
                ),
                DomainAbstractionHeuristic::new(
                    Some("da2".to_string()),
                    make_abstraction(&task, vec![4.0, 0.0, 0.0, 0.0]),
                ),
            ],
            vec![vec![0, 1], vec![2]],
        );

        let eval_state =
            EvaluationState::new_with_registry(&initial_state, 0.0, false, &task, &registry);
        let value = heuristic.compute_heuristic(&eval_state).unwrap();

        assert_eq!(value, 5.0);
    }
}