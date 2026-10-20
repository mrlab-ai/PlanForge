//! A domain abstraction with its goal distances, as CEGAR's flaw search sees
//! it.

use std::borrow::Cow;

use anyhow::{Context, Result, ensure};
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitValueIndex, NumericValue, OperatorIndex,
};

use std::time::Instant;

use super::abstract_operators::{AbstractOperator, AbstractOperatorGenerator};
use super::config::DomainAbstractionConfig;
use super::factory::{DistanceTableOptions, DomainAbstractionFactory};
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use crate::evaluation::abstractions::common::cegar::refine::Refinable;
use crate::evaluation::abstractions::common::cegar::split::Split;
use crate::evaluation::abstractions::common::cegar::view::{
    AbstractionView, RefinementLimits, TransitionKey,
};
use crate::evaluation::abstractions::common::transition_system::AbstractDistanceTable;

pub(super) struct DomainView<'a> {
    pub(super) task: &'a dyn AbstractNumericTask,
    pub(super) factory: &'a DomainAbstractionFactory,
    pub(super) numeric_domain_sizes: &'a [usize],
    pub(super) hash_multipliers: &'a [usize],
    pub(super) operators: &'a [AbstractOperator],
    pub(super) table: &'a AbstractDistanceTable,
    pub(super) limits: RefinementLimits<'a>,
    /// Plan steps execute any label of their abstract operator, not just
    /// the first.
    pub(super) wildcard_plans: bool,
}

impl AbstractionView for DomainView<'_> {
    /// The number of abstract states after the split.
    type Growth = u128;

    fn num_states(&self) -> usize {
        self.table.distances.len()
    }

    fn abstract_state(
        &self,
        propositions: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize> {
        self.factory.abstract_state_hash(
            propositions,
            numeric,
            self.numeric_domain_sizes,
            self.hash_multipliers,
        )
    }

    fn region(&self, state: usize) -> Cow<'_, CartesianSet> {
        Cow::Owned(
            self.factory
                .cartesian_set_from_hash(state, self.numeric_domain_sizes, self.hash_multipliers)
                .expect("every abstract state hash decodes to a region"),
        )
    }

    fn goal_distance(&self, state: usize) -> f64 {
        self.table.distances[state]
    }

    /// Evaluated on demand: the flaw search only asks about the states along
    /// the abstract plan.
    fn is_goal(&self, state: usize) -> Result<bool> {
        self.factory.is_goal_state(
            self.task,
            state,
            &self.table.goal_facts,
            self.numeric_domain_sizes,
            self.hash_multipliers,
        )
    }

    fn optimal_transition(&self, state: usize) -> Result<TransitionKey> {
        let op_id = self.table.generating_op_ids[state].with_context(|| {
            format!(
                "non-goal abstract state {state} with finite distance has no generating operator"
            )
        })?;
        let op = &self.operators[op_id];
        let target = (state as i64 - op.hash_effect as i64) as usize;
        ensure!(
            target < self.table.distances.len(),
            "abstract operator {op_id} leads from state {state} out of the abstraction"
        );
        let concrete_op_id = *op
            .concrete_op_ids
            .iter()
            .min()
            .with_context(|| format!("abstract operator {op_id} has no concrete operator"))?;
        Ok(TransitionKey {
            source: state,
            concrete_op_id,
            target,
        })
    }

    fn growth(&self, split: &Split) -> Result<u128> {
        Ok(self.states_after(split))
    }

    fn limits(&self) -> RefinementLimits<'_> {
        self.limits
    }

    /// Splitting a dimension of `values` values multiplies the abstract state
    /// count by `(values + 1) / values`.
    fn states_after(&self, split: &Split) -> u128 {
        let values = match split {
            Split::Propositional { var_id, .. } => self.factory.domain_sizes()[var_id.index()],
            Split::Numeric { var_id, .. } => self.numeric_domain_sizes[var_id.index()],
        } as u128;
        self.table.distances.len() as u128 / values * (values + 1)
    }

    fn step_operators(&self, transition: TransitionKey) -> Vec<OperatorIndex> {
        if !self.wildcard_plans {
            return vec![transition.concrete_op_id];
        }
        let op_id = self.table.generating_op_ids[transition.source]
            .expect("an optimal transition leaves through its generating operator");
        let mut labels = self.operators[op_id].concrete_op_ids.clone();
        labels.sort_unstable();
        labels.dedup();
        labels
    }
}

/// A domain abstraction under refinement. A split refines a variable in every
/// abstract state, so every iteration rebuilds the abstract operators and the
/// goal distances.
pub(super) struct DomainRefinement<'a> {
    pub(super) task: &'a dyn AbstractNumericTask,
    pub(super) factory: DomainAbstractionFactory,
    pub(super) limits: RefinementLimits<'a>,
    pub(super) combine_labels: bool,
    pub(super) wildcard_plans: bool,
    pub(super) debug: bool,
    /// The operators and goal distances of the current abstraction, built by
    /// [`Refinable::view`].
    built: Option<(
        AbstractOperatorGenerator,
        Vec<AbstractOperator>,
        AbstractDistanceTable,
    )>,
}

impl<'a> DomainRefinement<'a> {
    pub(super) fn new(
        task: &'a dyn AbstractNumericTask,
        factory: DomainAbstractionFactory,
        limits: RefinementLimits<'a>,
        config: &DomainAbstractionConfig,
    ) -> Self {
        Self {
            task,
            factory,
            limits,
            combine_labels: config.cegar.combine_labels,
            wildcard_plans: config.use_wildcard_plans,
            debug: config.cegar.debug,
            built: None,
        }
    }

    pub(super) fn into_factory(self) -> DomainAbstractionFactory {
        self.factory
    }
}

impl Refinable for DomainRefinement<'_> {
    type View<'b>
        = DomainView<'b>
    where
        Self: 'b;

    fn view(&mut self, deadline: Option<Instant>) -> Result<DomainView<'_>> {
        let mut generator = self
            .factory
            .make_operator_generator(self.task, self.combine_labels)?;
        let operators = generator.build_abstract_operators_with_deadline(self.task, deadline)?;
        let table = self.factory.build_distance_table(
            self.task,
            &generator,
            &operators,
            DistanceTableOptions::default()
                .with_dump_distances(self.debug)
                .with_deadline(deadline),
        )?;
        let (generator, operators, table) = self.built.insert((generator, operators, table));
        Ok(DomainView {
            task: self.task,
            factory: &self.factory,
            numeric_domain_sizes: generator.numeric_domain_sizes(),
            hash_multipliers: generator.hash_multipliers(),
            operators,
            table,
            limits: self.limits,
            wildcard_plans: self.wildcard_plans,
        })
    }

    fn apply_split(&mut self, split: Split) -> Result<()> {
        self.built = None;
        self.factory.apply_split(self.task, &split)
    }
}
