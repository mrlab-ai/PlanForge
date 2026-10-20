use super::*;
use crate::evaluation::abstractions::collections::config::MemberSpecialization;

pub struct CartesianAbstractionCollectionGenerator {
    pub(super) config: CartesianAbstractionCollectionConfig,
}

impl CartesianAbstractionCollectionGenerator {
    pub fn new(config: CartesianAbstractionCollectionConfig) -> Result<Self> {
        config.validate()?;
        CartesianAbstractionGenerator::new(config.member.clone())?;
        Ok(Self { config })
    }

    /// Builds the collection of the task: by random restarts for the
    /// standard strategy (see
    /// [`standard`](crate::evaluation::abstractions::collections::standard)),
    /// otherwise by the goal-focused schedule (see
    /// [`goal_focused`](crate::evaluation::abstractions::collections::goal_focused)
    /// for the schedule, the progress states, and the stop rules).
    ///
    /// Each member changes only the goal view. Operators, state mappings, and
    /// concrete operator IDs stay identical to the base task. Changing the
    /// refinement root only chooses counterexamples; every hierarchy still
    /// partitions the full task state space, so the resulting transition
    /// systems remain admissible components for canonical and cost-partitioned
    /// collection heuristics.
    pub fn generate(&self, task: &dyn AbstractNumericTask) -> Result<Vec<CartesianAbstraction>> {
        if self.config.collection_strategy == CollectionStrategy::Standard {
            return build_standard_collection(task, &self.config.standard_schedule(), self);
        }
        build_goal_focused_collection(task, &self.config.goal_focused_schedule(), self)
    }
}

impl GoalFocusedBackend for CartesianAbstractionCollectionGenerator {
    type Member = CartesianAbstraction;

    const NAME: &'static str = "Cartesian";

    fn build(
        &self,
        task: &dyn AbstractNumericTask,
        root: Option<&ExplicitState>,
        variant: &GoalVariant,
        budget: MemberBudget,
    ) -> Result<(CartesianAbstraction, BuiltReport)> {
        let config = self
            .config
            .member
            .for_variant(variant, self.config.split_kind, budget);
        let abstraction =
            CartesianAbstractionGenerator::new(config)?.generate_from_root(task, root)?;
        let report = BuiltReport {
            initial_h: abstraction.distance_table.distances
                [abstraction.distance_table.initial_state_hash],
            num_states: abstraction.num_states(),
            concrete_plan: abstraction.metadata.concrete_plan_operator_ids.clone(),
            root_is_dead_end: abstraction.metadata.stop_reason == CegarStopReason::DeadEnd,
        };
        Ok((abstraction, report))
    }

    fn record_membership(
        &self,
        abstraction: &mut CartesianAbstraction,
        membership: Option<CollectionMembership>,
    ) {
        abstraction.metadata.collection_member = membership;
        abstraction.metadata.abstraction_use = AbstractionUse::CollectionMember;
    }
}

/// A refinement hierarchy as a value: equal for Cartesian abstractions that
/// the same splits refined in the same order.
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) enum HierarchyKey {
    Leaf(usize),
    Propositional {
        var_id: VariableIndex,
        wanted: Vec<ExplicitValueIndex>,
        children: (usize, usize),
    },
    Numeric {
        var_id: VariableIndex,
        boundary_bits: u64,
        lower_includes_boundary: bool,
        children: (usize, usize),
    },
}

impl StandardBackend for CartesianAbstractionCollectionGenerator {
    type Member = CartesianAbstraction;
    type Key = Vec<HierarchyKey>;

    const NAME: &'static str = "Cartesian";

    fn build(
        &self,
        task: &dyn AbstractNumericTask,
        member: &StandardMember,
    ) -> Result<CartesianAbstraction> {
        let config = self.config.member.for_restart(member);
        let mut abstraction = CartesianAbstractionGenerator::new(config)?.generate(task)?;
        abstraction.metadata.abstraction_use = AbstractionUse::CollectionMember;
        Ok(abstraction)
    }

    fn key(&self, abstraction: &CartesianAbstraction) -> Vec<HierarchyKey> {
        abstraction
            .hierarchy
            .nodes
            .iter()
            .map(|node| match node {
                RefinementNode::Leaf { state_id } => HierarchyKey::Leaf(*state_id),
                RefinementNode::Propositional {
                    var_id,
                    wanted,
                    wanted_child,
                    other_child,
                } => HierarchyKey::Propositional {
                    var_id: *var_id,
                    wanted: wanted.clone(),
                    children: (*wanted_child, *other_child),
                },
                RefinementNode::Numeric {
                    var_id,
                    boundary,
                    lower_includes_boundary,
                    lower_child,
                    upper_child,
                } => HierarchyKey::Numeric {
                    var_id: *var_id,
                    boundary_bits: float_tolerance::canonical_bits(*boundary),
                    lower_includes_boundary: *lower_includes_boundary,
                    children: (*lower_child, *upper_child),
                },
            })
            .collect()
    }

    fn num_states(&self, abstraction: &CartesianAbstraction) -> usize {
        abstraction.num_states()
    }

    fn settles(&self, abstraction: &CartesianAbstraction) -> Option<&'static str> {
        if abstraction.metadata.solved_by_self {
            Some("concrete plan")
        } else if abstraction.metadata.stop_reason == CegarStopReason::DeadEnd {
            Some("initial state is a dead end")
        } else {
            None
        }
    }
}
