//! The progress states of a goal-focused collection's variants; see the
//! [module documentation](super).

use std::collections::VecDeque;

use anyhow::{Result, ensure};
use planforge_sas::explicit_state::{ExplicitState, ExplicitStateSpace};
use planforge_sas::numeric_task::{AbstractNumericTask, OperatorIndex};
use tracing::debug;

use super::{GoalFocusedBackend, MemberRoot};

/// The progress state of one variant.
pub(super) struct VariantProgress {
    pub(super) state: ExplicitState,
    satisfied_goals: usize,
    /// `state` is past the task initial state.
    pub(super) advanced: bool,
    /// The variant refines from the task initial state again: its progress
    /// state reached the complete task goal or turned out to be a dead end.
    pub(super) complete: bool,
}

/// The progress states of all variants of a collection on a task with
/// goals, and the retries they queued.
pub(super) struct ProgressStates<'task> {
    /// The concrete states progress states move through.
    states: ExplicitStateSpace<'task>,
    /// The task initial state, which progress states start from and return
    /// to.
    initial_state: ExplicitState,
    goal_count: usize,
    /// One per variant.
    variants: Vec<VariantProgress>,
    /// `(goal_id, variant_id)` pairs to retry from a variant's advanced
    /// progress state.
    continuation_queue: VecDeque<(usize, usize)>,
}

impl<'task> ProgressStates<'task> {
    /// `variants_per_goal` progress states at the initial state of `task`.
    pub(super) fn new(
        task: &'task dyn AbstractNumericTask,
        variants_per_goal: usize,
    ) -> Result<Self> {
        let states = ExplicitStateSpace::new(task);
        let initial_state = states.initial_state()?;
        let satisfied_goals = states.satisfied_goal_count(&initial_state);
        let variants = (0..variants_per_goal)
            .map(|_| VariantProgress {
                state: initial_state.clone(),
                satisfied_goals,
                advanced: false,
                complete: false,
            })
            .collect();
        Ok(Self {
            states,
            initial_state,
            goal_count: task.get_num_goals(),
            variants,
            continuation_queue: VecDeque::new(),
        })
    }

    pub(super) fn variant(&self, variant_id: usize) -> &VariantProgress {
        &self.variants[variant_id]
    }

    pub(super) fn has_continuations(&self) -> bool {
        !self.continuation_queue.is_empty()
    }

    /// Drop queued retries whose goal the variant's progress state now
    /// satisfies anyway, and return the first one that still has work to do.
    pub(super) fn pop_unsatisfied_continuation(&mut self) -> Option<(usize, usize)> {
        while let Some((goal_id, variant_id)) = self.continuation_queue.pop_front() {
            if !self
                .states
                .satisfies_goal(&self.variants[variant_id].state, goal_id)
            {
                return Some((goal_id, variant_id));
            }
        }
        None
    }

    /// Replay the concrete plan of a member of goal `goal_id` that refined
    /// from `root` onto variant `variant_id`'s progress state, so the
    /// variant's next member refines from further along. Goals the variant
    /// attempted already, which `variants_built_by_goal` tells, may be
    /// retried from the advanced state.
    pub(super) fn advance<B: GoalFocusedBackend>(
        &mut self,
        goal_id: usize,
        variant_id: usize,
        root: MemberRoot,
        concrete_plan: Option<&[OperatorIndex]>,
        variants_built_by_goal: &[usize],
    ) -> Result<()> {
        if self.variants[variant_id].complete {
            return Ok(());
        }
        if root == MemberRoot::DeadProgress {
            self.complete(variant_id);
            debug!(
                "{} collection: dead progress state completed variant {variant_id} after rebuilding goal {goal_id} from the initial state",
                B::NAME
            );
            return Ok(());
        }
        let Some(operator_ids) = concrete_plan else {
            debug!(
                "{} collection: goal {goal_id} variant {variant_id} produced no concrete plan; progress state remains unchanged",
                B::NAME
            );
            return Ok(());
        };

        let progress = &mut self.variants[variant_id];
        let previous_satisfied_goals = progress.satisfied_goals;
        let state = self.states.replay(&progress.state, operator_ids)?;
        ensure!(
            self.states.satisfies_goal(&state, goal_id),
            "{} collection: the concrete plan of goal {goal_id} variant {variant_id} does not reach its goal",
            B::NAME
        );
        let satisfied_goals = self.states.satisfied_goal_count(&state);
        progress.state = state;
        progress.satisfied_goals = satisfied_goals;
        debug!(
            "{} collection: advanced the progress state of variant {variant_id} through {} concrete operators; satisfied_goals={satisfied_goals}/{}",
            B::NAME,
            operator_ids.len(),
            self.goal_count,
        );

        if satisfied_goals == self.goal_count {
            self.complete(variant_id);
            debug!(
                "{} collection: full goal reached for variant {variant_id}; completed its progress",
                B::NAME
            );
            return Ok(());
        }
        self.variants[variant_id].advanced = true;
        if satisfied_goals > previous_satisfied_goals {
            self.queue_reopened_goals(variant_id, variants_built_by_goal);
        }
        Ok(())
    }

    /// Return a variant's progress state to the task initial state and stop
    /// advancing it, because it is a dead end or has reached the complete task
    /// goal. Retries queued for the variant go with it.
    fn complete(&mut self, variant_id: usize) {
        self.variants[variant_id] = VariantProgress {
            state: self.initial_state.clone(),
            satisfied_goals: self.states.satisfied_goal_count(&self.initial_state),
            advanced: false,
            complete: true,
        };
        self.continuation_queue
            .retain(|(_, queued_variant_id)| *queued_variant_id != variant_id);
    }

    /// The variant's progress state now satisfies goals it did not before, so
    /// goals this variant already attempted may be worth another abstraction
    /// from here.
    fn queue_reopened_goals(&mut self, variant_id: usize, variants_built_by_goal: &[usize]) {
        let state = &self.variants[variant_id].state;
        for (retry_goal_id, &variants_built) in variants_built_by_goal.iter().enumerate() {
            let was_already_attempted = variants_built > variant_id;
            if was_already_attempted
                && !self.states.satisfies_goal(state, retry_goal_id)
                && !self
                    .continuation_queue
                    .contains(&(retry_goal_id, variant_id))
            {
                self.continuation_queue
                    .push_back((retry_goal_id, variant_id));
            }
        }
    }

    /// The goals every variant's progress state satisfies.
    pub(super) fn satisfied_goals(&self) -> Vec<usize> {
        self.variants
            .iter()
            .map(|progress| progress.satisfied_goals)
            .collect()
    }
}
