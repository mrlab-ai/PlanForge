use std::collections::BTreeSet;

use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::numeric_task::{AbstractNumericTask, OperatorIndex};

use crate::evaluation::abstractions::cartesian::CartesianAbstraction;
use crate::evaluation::abstractions::common::abstraction::{
    CegarAbstraction, CollectionMembership,
};
use crate::evaluation::abstractions::common::cegar::SplitKind;
use crate::evaluation::abstractions::common::transition_system::AbstractOperatorSourceSets;
use crate::evaluation::abstractions::domain::abstraction::DomainAbstraction;
use crate::evaluation::abstractions::pdb::pattern_database::PatternDatabase;
use crate::evaluation::evaluator::EvaluationError;

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

/// The backend-independent facts about a member of a goal-focused collection
/// that collection heuristics order components by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoalFocusedMember {
    pub membership: CollectionMembership,
    pub split_kind: SplitKind,
    pub tie_rank: Option<usize>,
    /// The abstract operators, which operator regions are attached to.
    pub abstract_operators: usize,
}

/// An abstraction that collection heuristics combine.
pub enum AbstractionComponent<'task> {
    Domain(Box<DomainAbstraction>),
    Cartesian(Box<CartesianAbstraction>),
    PatternDatabase(Box<PatternDatabase<'task>>),
}

impl<'task> AbstractionComponent<'task> {
    /// The component as an abstraction CEGAR built; `None` for a pattern
    /// database.
    pub fn cegar(&self) -> Option<&dyn CegarAbstraction> {
        match self {
            Self::Domain(abstraction) => Some(abstraction.as_ref()),
            Self::Cartesian(abstraction) => Some(abstraction.as_ref()),
            Self::PatternDatabase(_) => None,
        }
    }

    /// Builds the source sets regional cost partitioning reads; a pattern
    /// database has none.
    pub fn ensure_abstract_source_sets(
        &mut self,
        task: &dyn AbstractNumericTask,
    ) -> anyhow::Result<()> {
        match self {
            Self::Domain(abstraction) => abstraction.ensure_abstract_source_sets(task),
            Self::Cartesian(abstraction) => abstraction.ensure_abstract_source_sets(task),
            Self::PatternDatabase(_) => Ok(()),
        }
    }

    pub fn discard_transition_data(&mut self) {
        match self {
            Self::Domain(abstraction) => abstraction.discard_transition_data(),
            Self::Cartesian(abstraction) => abstraction.discard_transition_data(),
            Self::PatternDatabase(_) => {}
        }
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
            Self::PatternDatabase(pdb) => pdb.num_states(),
            _ => self.cegar().expect("not a pattern database").num_states(),
        }
    }

    pub fn proves_initial_state_optimal(&self) -> bool {
        self.cegar().is_some_and(|abstraction| {
            let metadata = abstraction.metadata();
            metadata.solved_by_self() && metadata.abstraction_use.permits_initial_optimality_proof()
        })
    }

    /// The abstract state of the concrete state `values`; `None` for a state
    /// outside the states a pattern database enumerated.
    pub fn abstract_state_id(
        &self,
        values: &ExplicitState,
    ) -> Result<Option<usize>, EvaluationError> {
        match self {
            Self::PatternDatabase(pdb) => pdb
                .abstract_state_id_from_source_state_values(&values.propositional, &values.numeric)
                .map_err(EvaluationError::ComputationFailed),
            _ => self
                .cegar()
                .expect("not a pattern database")
                .abstract_state(&values.propositional, &values.numeric)
                .map(Some)
                .map_err(|error| EvaluationError::ComputationFailed(format!("{error:#}"))),
        }
    }

    /// The goal distance this component alone estimates for `values`.
    pub fn standalone_value(&self, values: &ExplicitState) -> Result<f64, EvaluationError> {
        match self {
            // A pattern database projects the state itself, and falls back
            // for a state it did not enumerate.
            Self::PatternDatabase(pdb) => pdb
                .lookup_projected_or_fallback_from_source_state_values(
                    &values.propositional,
                    &values.numeric,
                )
                .map_err(EvaluationError::ComputationFailed),
            _ => {
                let abstraction = self.cegar().expect("not a pattern database");
                let state_id = abstraction
                    .abstract_state(&values.propositional, &values.numeric)
                    .map_err(|error| EvaluationError::ComputationFailed(format!("{error:#}")))?;
                Ok(abstraction.distance_table().distances[state_id])
            }
        }
    }

    pub fn relevant_operator_ids(
        &self,
        task: &dyn AbstractNumericTask,
    ) -> Result<BTreeSet<OperatorIndex>, String> {
        let ids: Vec<OperatorIndex> = match self {
            Self::Domain(abstraction) => abstraction.relevant_operator_ids.clone(),
            Self::Cartesian(abstraction) => abstraction.relevant_operator_ids.clone(),
            Self::PatternDatabase(pdb) => pdb.relevant_operator_ids(),
        };
        let operator_count = task.get_operators().len();
        if let Some(operator_id) = ids.iter().copied().find(|id| id.index() >= operator_count) {
            return Err(format!(
                "{} references operator {operator_id}, but task has {operator_count} operators",
                self.kind()
            ));
        }
        Ok(ids.into_iter().collect())
    }

    /// `Some` exactly for members of a goal-focused collection.
    pub fn goal_focused_member(&self) -> Option<GoalFocusedMember> {
        let abstraction = self.cegar()?;
        let metadata = abstraction.metadata();
        metadata
            .collection_membership()
            .map(|membership| GoalFocusedMember {
                membership,
                split_kind: metadata.split_kind,
                tie_rank: metadata.split_selection_rank,
                abstract_operators: abstraction.num_abstract_operators(),
            })
    }

    #[cfg(test)]
    pub fn as_domain(&self) -> Option<&DomainAbstraction> {
        match self {
            Self::Domain(abstraction) => Some(abstraction),
            Self::Cartesian(_) | Self::PatternDatabase(_) => None,
        }
    }

    #[cfg(test)]
    pub fn as_cartesian(&self) -> Option<&CartesianAbstraction> {
        match self {
            Self::Cartesian(abstraction) => Some(abstraction),
            Self::Domain(_) | Self::PatternDatabase(_) => None,
        }
    }

    /// The source sets of the abstract operators, which regional cost
    /// partitioning attaches costs to; `None` for a pattern database.
    pub fn abstract_source_sets(&self) -> Option<&[AbstractOperatorSourceSets]> {
        self.cegar().map(CegarAbstraction::abstract_source_sets)
    }
}

impl<'task> From<DomainAbstraction> for AbstractionComponent<'task> {
    fn from(abstraction: DomainAbstraction) -> Self {
        Self::Domain(Box::new(abstraction))
    }
}

impl<'task> From<CartesianAbstraction> for AbstractionComponent<'task> {
    fn from(abstraction: CartesianAbstraction) -> Self {
        Self::Cartesian(Box::new(abstraction))
    }
}

impl<'task> From<PatternDatabase<'task>> for AbstractionComponent<'task> {
    fn from(pdb: PatternDatabase<'task>) -> Self {
        Self::PatternDatabase(Box::new(pdb))
    }
}
