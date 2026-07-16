use std::collections::BTreeSet;

use planforge_sas::numeric_task::AbstractNumericTask;

use crate::evaluation::cartesian_abstractions::{
    CartesianAbstraction, CartesianAbstractionHeuristic,
};
use crate::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstraction;
use crate::evaluation::domain_abstractions::domain_abstraction_heuristic::DomainAbstractionHeuristic;
use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;
use crate::evaluation::pattern_databases::pattern_database::PatternDatabase;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbstractionKind {
    Domain,
    Cartesian,
    PatternDatabase,
}

impl std::fmt::Display for AbstractionKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Domain => formatter.write_str("domain abstraction"),
            Self::Cartesian => formatter.write_str("Cartesian abstraction"),
            Self::PatternDatabase => formatter.write_str("pattern database"),
        }
    }
}

pub enum AbstractionComponent<'task> {
    Domain(Box<DomainAbstractionHeuristic>),
    Cartesian(Box<CartesianAbstractionHeuristic>),
    PatternDatabase(Box<PatternDatabase<'task>>),
}

impl<'task> AbstractionComponent<'task> {
    pub fn domain(name: Option<String>, abstraction: DomainAbstraction) -> Self {
        Self::Domain(Box::new(DomainAbstractionHeuristic::new(name, abstraction)))
    }

    pub fn cartesian(name: Option<String>, abstraction: CartesianAbstraction) -> Self {
        Self::Cartesian(Box::new(CartesianAbstractionHeuristic::new(
            name,
            abstraction,
        )))
    }

    pub fn pattern_database(pdb: PatternDatabase<'task>) -> Self {
        Self::PatternDatabase(Box::new(pdb))
    }

    pub fn kind(&self) -> AbstractionKind {
        match self {
            Self::Domain(_) => AbstractionKind::Domain,
            Self::Cartesian(_) => AbstractionKind::Cartesian,
            Self::PatternDatabase(_) => AbstractionKind::PatternDatabase,
        }
    }

    pub fn num_states(&self) -> usize {
        match self {
            Self::Domain(heuristic) => heuristic.abstraction().distance_table.distances.len(),
            Self::Cartesian(heuristic) => heuristic.abstraction().num_states(),
            Self::PatternDatabase(pdb) => pdb.num_states(),
        }
    }

    pub fn standalone_value(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        match self {
            Self::Domain(heuristic) => heuristic.compute_heuristic(eval_state),
            Self::Cartesian(heuristic) => heuristic.compute_heuristic(eval_state),
            Self::PatternDatabase(pdb) => {
                let registry = eval_state.state_registry().ok_or_else(|| {
                    EvaluationError::InvalidState("PDB lookup requires state registry".to_string())
                })?;
                pdb.lookup_or_fallback_from_concrete_state(eval_state.state(), registry)
                    .map_err(EvaluationError::ComputationFailed)
            }
        }
    }

    pub fn exact_state_id(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<Option<usize>, EvaluationError> {
        match self {
            Self::Domain(heuristic) => heuristic.abstract_state_hash(eval_state).map(Some),
            Self::Cartesian(heuristic) => heuristic.abstract_state_id(eval_state).map(Some),
            Self::PatternDatabase(pdb) => {
                let registry = eval_state.state_registry().ok_or_else(|| {
                    EvaluationError::InvalidState("PDB lookup requires state registry".to_string())
                })?;
                pdb.abstract_state_id_from_concrete_state(eval_state.state(), registry)
                    .map_err(EvaluationError::ComputationFailed)
            }
        }
    }

    pub fn distance_for_state_id(&self, state_id: usize) -> Result<f64, EvaluationError> {
        let value = match self {
            Self::Domain(heuristic) => heuristic
                .abstraction()
                .distance_table
                .distances
                .get(state_id)
                .copied(),
            Self::Cartesian(heuristic) => heuristic
                .abstraction()
                .distance_table
                .distances
                .get(state_id)
                .copied(),
            Self::PatternDatabase(pdb) => {
                return pdb
                    .distance_for_state_id(state_id)
                    .map_err(EvaluationError::InvalidState);
            }
        };
        value.ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "{} state id {state_id} out of bounds for {} states",
                self.kind(),
                self.num_states()
            ))
        })
    }

    pub fn relevant_operator_ids(
        &self,
        task: &dyn AbstractNumericTask,
    ) -> Result<BTreeSet<usize>, String> {
        let ids = match self {
            Self::Domain(heuristic) => {
                let abstraction = heuristic.abstraction();
                if !abstraction.relevant_operator_ids.is_empty() {
                    abstraction.relevant_operator_ids.clone()
                } else {
                    let abstraction_task = abstraction.task_for_factory(task);
                    let mut generator = abstraction
                        .factory
                        .make_operator_generator(abstraction_task, abstraction.combine_labels)
                        .map_err(|error| {
                            format!(
                                "failed to build domain abstraction operator generator: {error:#}"
                            )
                        })?;
                    let operators = generator
                        .build_abstract_operators(abstraction_task)
                        .map_err(|error| {
                            format!("failed to build domain abstraction operators: {error:#}")
                        })?;
                    abstraction
                        .factory
                        .relevant_operator_ids_from_operators_with_deadline(
                            abstraction_task,
                            abstraction.combine_labels,
                            &operators,
                            None,
                        )
                        .map_err(|error| {
                            format!("failed to compute domain relevant operators: {error:#}")
                        })?
                }
            }
            Self::Cartesian(heuristic) => heuristic.abstraction().relevant_operator_ids.clone(),
            Self::PatternDatabase(pdb) => pdb.relevant_operator_ids(),
        };

        let operator_count = task.get_operators().len();
        if let Some(operator_id) = ids.iter().copied().find(|id| *id >= operator_count) {
            return Err(format!(
                "{} references operator {operator_id}, but task has {operator_count} operators",
                self.kind()
            ));
        }
        Ok(ids.into_iter().collect())
    }

    pub fn as_domain(&self) -> Option<&DomainAbstraction> {
        match self {
            Self::Domain(heuristic) => Some(heuristic.abstraction()),
            Self::Cartesian(_) | Self::PatternDatabase(_) => None,
        }
    }

    pub fn as_cartesian(&self) -> Option<&CartesianAbstraction> {
        match self {
            Self::Cartesian(heuristic) => Some(heuristic.abstraction()),
            Self::Domain(_) | Self::PatternDatabase(_) => None,
        }
    }

    pub fn as_pattern_database(&self) -> Option<&PatternDatabase<'task>> {
        match self {
            Self::PatternDatabase(pdb) => Some(pdb.as_ref()),
            Self::Domain(_) | Self::Cartesian(_) => None,
        }
    }
}

impl From<DomainAbstraction> for AbstractionComponent<'static> {
    fn from(abstraction: DomainAbstraction) -> Self {
        Self::domain(None, abstraction)
    }
}

impl From<CartesianAbstraction> for AbstractionComponent<'static> {
    fn from(abstraction: CartesianAbstraction) -> Self {
        Self::cartesian(None, abstraction)
    }
}

impl<'task> From<PatternDatabase<'task>> for AbstractionComponent<'task> {
    fn from(pdb: PatternDatabase<'task>) -> Self {
        Self::pattern_database(pdb)
    }
}
