use super::*;

pub struct CartesianAbstractionCollectionGenerator {
    pub(super) config: CartesianAbstractionCollectionConfig,
}

impl CartesianAbstractionCollectionGenerator {
    pub fn new(config: CartesianAbstractionCollectionConfig) -> Result<Self> {
        validate_collection_options(CollectionOptions {
            strategy: config.collection_strategy,
            variants_per_goal: config.variants_per_goal,
            max_collection_size: config.max_collection_states,
            progress_states: config.progress_states,
            alternate_split_kinds: config.alternate_split_kinds,
            single_goal_tasks: config.single_goal_tasks,
            fixed_split_kind: config.fixed_split_kind,
            flaw_kind: config.abstraction.flaw_kind,
        })?;
        CartesianAbstractionGenerator::new(config.abstraction.clone())?;
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
            return build_standard_collection(task, &self.standard_schedule(), self);
        }
        build_goal_focused_collection(
            task,
            &GoalFocusedSchedule {
                strategy: self.config.collection_strategy,
                variants_per_goal: self.config.variants_per_goal,
                random_seed: self.config.abstraction.random_seed,
                max_collection_states: self.config.max_collection_states,
                total_max_time: self.config.total_max_time,
                progress_states: self.config.progress_states,
                alternate_split_kinds: self
                    .config
                    .collection_strategy
                    .alternates_split_kinds(self.config.alternate_split_kinds),
                single_goal_tasks: self.config.single_goal_tasks,
            },
            self,
        )
    }

    fn standard_schedule(&self) -> StandardSchedule {
        let seconds = |limit: Option<Duration>| limit.map_or(f64::INFINITY, |d| d.as_secs_f64());
        StandardSchedule {
            max_states: self.config.abstraction.max_states,
            max_collection_size: self.config.max_collection_states,
            max_time: seconds(self.config.abstraction.max_time),
            total_max_time: seconds(self.config.total_max_time),
            stagnation_limit: self.config.stagnation_limit,
            blacklist_trigger_percentage: self.config.blacklist_trigger_percentage,
            enable_blacklist_on_stagnation: self.config.enable_blacklist_on_stagnation,
            random_seed: self.config.abstraction.random_seed,
            alternate_split_kinds: self
                .config
                .collection_strategy
                .alternates_split_kinds(self.config.alternate_split_kinds),
            split_kind: self
                .config
                .fixed_split_kind
                .then_some(self.config.abstraction.split_kind),
        }
    }

    /// Configuration for one member: the collection's per-abstraction
    /// configuration, capped by what is left of the collection's state and
    /// time budgets, then specialised for this variant.
    pub(super) fn member_abstraction_config(
        &self,
        variant: &GoalVariant,
        budget: MemberBudget,
    ) -> CartesianAbstractionConfig {
        let MemberBudget {
            remaining_states,
            remaining_time,
        } = budget;
        let mut config = self.config.abstraction.clone();
        config.max_states = config.max_states.min(remaining_states);
        // Whichever of the two budgets exist bind; the tighter one wins.
        config.max_time = config.max_time.into_iter().chain(remaining_time).min();
        if let Some(split_kind) = variant.split_kind {
            config.split_kind = split_kind;
        }
        if let Some(rank) = variant.tie_rank {
            config.split_selection_rank = Some(rank);
        }
        config.random_seed = variant.seed;
        config
    }
}

impl GoalFocusedBackend for CartesianAbstractionCollectionGenerator {
    type Member = CartesianAbstraction;

    const NAME: &'static str = "Cartesian";

    fn build(
        &self,
        task: &dyn AbstractNumericTask,
        root: Option<&ConcreteState>,
        variant: &GoalVariant,
        budget: MemberBudget,
    ) -> Result<(CartesianAbstraction, BuiltReport)> {
        let abstraction =
            CartesianAbstractionGenerator::new(self.member_abstraction_config(variant, budget))?
                .generate_from_root(task, root)?;
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
        let mut config = self.config.abstraction.clone();
        config.max_states = member.max_states;
        config.max_time = member
            .max_time
            .is_finite()
            .then(|| Duration::from_secs_f64(member.max_time.max(0.0)));
        config.random_seed = Some(member.seed);
        if let Some(split_kind) = member.split_kind {
            config.split_kind = split_kind;
        }
        config.initial_split_var = member.initial_split_var;
        config.blacklisted_prop_var_ids = member.blacklisted_prop_var_ids.clone();
        config.blacklisted_numeric_var_ids = member.blacklisted_numeric_var_ids.clone();
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
