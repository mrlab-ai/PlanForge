use super::*;

pub struct CartesianAbstractionCollectionGenerator {
    pub(super) config: CartesianAbstractionCollectionConfig,
}

impl CartesianAbstractionCollectionGenerator {
    pub fn new(config: CartesianAbstractionCollectionConfig) -> Result<Self> {
        ensure!(
            config.max_collection_states > 0,
            "Cartesian max_collection_size must be > 0"
        );
        ensure!(
            config.variants_per_goal > 0,
            "Cartesian variants_per_goal must be > 0"
        );
        ensure!(
            config.alternate_split_kinds || config.collection_strategy.is_complementary(),
            "`alternate_split_kinds=false` requires `collection_strategy=complementary`"
        );
        ensure!(
            config.single_goal_tasks || config.collection_strategy.is_complementary(),
            "`single_goal_tasks=false` requires `collection_strategy=complementary`"
        );
        ensure!(
            config.single_goal_tasks || !config.progress_states,
            "{FULL_TASK_PROGRESS_STATES_ERROR}"
        );
        CartesianAbstractionGenerator::new(config.abstraction.clone())?;
        Ok(Self { config })
    }

    /// Builds the goal-focused collection of the task; see
    /// [`goal_focused`](crate::evaluation::abstractions::collections::goal_focused) for the
    /// schedule, the progress states, and the stop rules.
    ///
    /// Each member changes only the goal view. Operators, state mappings, and
    /// concrete operator IDs stay identical to the base task. Changing the
    /// refinement root only chooses counterexamples; every hierarchy still
    /// partitions the full task state space, so the resulting transition
    /// systems remain admissible components for canonical and cost-partitioned
    /// collection heuristics.
    pub fn generate(&self, task: &dyn AbstractNumericTask) -> Result<Vec<CartesianAbstraction>> {
        build_goal_focused_collection(
            task,
            &GoalFocusedSchedule {
                strategy: self.config.collection_strategy,
                variants_per_goal: self.config.variants_per_goal,
                random_seed: self.config.abstraction.random_seed,
                max_collection_states: self.config.max_collection_states,
                total_max_time: self.config.total_max_time,
                progress_states: self.config.progress_states,
                alternate_split_kinds: self.config.alternate_split_kinds,
                single_goal_tasks: self.config.single_goal_tasks,
            },
            self,
        )
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
        if let Some(direction) = variant.direction {
            config.split_kind = match direction {
                MemberDirection::Progression => SplitKind::Concrete,
                MemberDirection::Regression => SplitKind::Wanted,
            };
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
            root_is_dead_end: abstraction.metadata.stop_reason == CartesianStopReason::DeadEnd,
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
