use planforge_sas::explicit_state::StateValues;
use planforge_sas::numeric_task::Task;
use std::collections::BTreeSet;

use planforge_sas::numeric_task::OperatorIndex;

use crate::abstractions::cartesian::CartesianAbstraction;
use crate::abstractions::collections::member::{CollectionMembership, GoalMembership, Member};
use crate::abstractions::common::abstraction::CegarAbstraction;
use crate::abstractions::common::cegar::SplitKind;
use crate::abstractions::common::transition_system::AbstractOperatorSourceSets;
use crate::abstractions::domain::abstraction::DomainAbstraction;
use crate::abstractions::pdb::pattern_database::PatternDatabase;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbstractionKind {
    Domain,
    Cartesian,
    PatternDatabase,
}

impl AbstractionKind {
    /// The representation's name in log and error messages.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Domain => "domain abstraction",
            Self::Cartesian => "Cartesian abstraction",
            Self::PatternDatabase => "pattern database",
        }
    }
}

impl std::fmt::Display for AbstractionKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// The backend-independent facts about a member of a goal-focused collection
/// that collection heuristics order components by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GoalFocusedMember {
    pub membership: GoalMembership,
    pub split_kind: SplitKind,
    pub tie_rank: Option<usize>,
    /// The abstract operators, which operator regions are attached to.
    pub abstract_operators: usize,
}

/// An abstraction that collection heuristics combine.
pub enum AbstractionComponent {
    Cegar {
        abstraction: Box<dyn CegarAbstraction>,
        /// Set exactly for a member of a collection.
        membership: Option<CollectionMembership>,
    },
    PatternDatabase(Box<PatternDatabase>),
}

impl AbstractionComponent {
    /// The component as an abstraction CEGAR built; `None` for a pattern
    /// database.
    pub fn cegar(&self) -> Option<&dyn CegarAbstraction> {
        match self {
            Self::Cegar { abstraction, .. } => Some(abstraction.as_ref()),
            Self::PatternDatabase(_) => None,
        }
    }

    pub fn discard_transitions(&mut self) {
        match self {
            Self::Cegar { abstraction, .. } => abstraction.discard_transitions(),
            Self::PatternDatabase(_) => {}
        }
    }

    pub fn kind(&self) -> AbstractionKind {
        match self {
            Self::Cegar { abstraction, .. } => abstraction.kind(),
            Self::PatternDatabase(_) => AbstractionKind::PatternDatabase,
        }
    }

    pub fn num_states(&self) -> usize {
        match self {
            Self::Cegar { abstraction, .. } => abstraction.num_states(),
            Self::PatternDatabase(pdb) => pdb.num_states(),
        }
    }

    /// The component's goal distance of the initial state is the optimal
    /// plan cost: CEGAR found a concrete plan on an abstraction of the full
    /// task refined from its initial state, which a collection member need
    /// not be.
    pub fn proves_initial_state_optimal(&self) -> bool {
        match self {
            Self::Cegar {
                abstraction,
                membership: None,
            } => abstraction.metadata().solved_by_self(),
            Self::Cegar { .. } | Self::PatternDatabase(_) => false,
        }
    }

    /// Where the component sits in its collection; `None` outside one.
    pub fn membership(&self) -> Option<CollectionMembership> {
        match self {
            Self::Cegar { membership, .. } => *membership,
            Self::PatternDatabase(_) => None,
        }
    }

    /// The abstract state of the concrete state `values`; `None` for a state
    /// outside the states a pattern database enumerated.
    pub fn abstract_state_id(&self, values: StateValues<'_>) -> Option<usize> {
        match self {
            Self::Cegar { abstraction, .. } => Some(abstraction.abstract_state(values)),
            Self::PatternDatabase(pdb) => pdb.abstract_state_id(values),
        }
    }

    /// The goal distance this component alone estimates for `values`.
    pub fn standalone_value(&self, values: StateValues<'_>) -> f64 {
        match self {
            Self::Cegar { abstraction, .. } => abstraction.goal_distance(values),
            Self::PatternDatabase(pdb) => pdb.value(values),
        }
    }

    pub fn relevant_operator_ids(&self, task: &Task) -> Result<BTreeSet<OperatorIndex>, String> {
        let ids: Vec<OperatorIndex> = match self {
            Self::Cegar { abstraction, .. } => abstraction.relevant_operator_ids().to_vec(),
            Self::PatternDatabase(pdb) => pdb.relevant_operator_ids().to_vec(),
        };
        let operator_count = task.operators().len();
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
        let membership = self.membership()?.goal?;
        let abstraction = self
            .cegar()
            .expect("only CEGAR abstractions are collection members");
        let metadata = abstraction.metadata();
        Some(GoalFocusedMember {
            membership,
            split_kind: metadata.split_kind,
            tie_rank: metadata.split_selection_rank,
            abstract_operators: abstraction.num_abstract_operators(),
        })
    }

    /// The source sets of the abstract operators, which regional cost
    /// partitioning attaches costs to; `None` for a pattern database and
    /// before regional saturation was prepared.
    pub fn abstract_source_sets(&self) -> Option<&[AbstractOperatorSourceSets]> {
        self.cegar()?.abstract_source_sets()
    }
}

impl From<DomainAbstraction> for AbstractionComponent {
    fn from(abstraction: DomainAbstraction) -> Self {
        Self::Cegar {
            abstraction: Box::new(abstraction),
            membership: None,
        }
    }
}

impl From<CartesianAbstraction> for AbstractionComponent {
    fn from(abstraction: CartesianAbstraction) -> Self {
        Self::Cegar {
            abstraction: Box::new(abstraction),
            membership: None,
        }
    }
}

impl From<PatternDatabase> for AbstractionComponent {
    fn from(pdb: PatternDatabase) -> Self {
        Self::PatternDatabase(Box::new(pdb))
    }
}

impl<A: CegarAbstraction + 'static> From<Member<A>> for AbstractionComponent {
    fn from(member: Member<A>) -> Self {
        Self::Cegar {
            abstraction: Box::new(member.abstraction),
            membership: Some(member.membership),
        }
    }
}
