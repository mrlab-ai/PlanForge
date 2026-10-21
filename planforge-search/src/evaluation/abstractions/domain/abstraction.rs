//! A built domain abstraction: the factory that maps concrete states to
//! abstract ones, its goal distances, and what was recorded about its
//! construction.

use std::cell::{Ref, RefCell};

use anyhow::{Context, Result, ensure};

use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitValueIndex, NumericValue, OperatorIndex,
};

use super::abstract_operators::AbstractOperator;
use super::factory::{
    DomainAbstractionFactory, MatchTree, MatchTreeGraph, StateRegions, source_set_within,
};
use crate::evaluation::abstractions::common::abstraction::{AbstractionMetadata, CegarAbstraction};
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use crate::evaluation::abstractions::common::transition_system::AbstractDistanceTable;
use crate::evaluation::abstractions::common::transition_system::{
    AbstractOperatorSourceSets, AbstractTransitionSystem,
};
use crate::evaluation::abstractions::cost_partitioning::costs::{
    RemainingCosts, SourceSetCostAllocation, saturate_labels, saturate_regions,
};
use crate::resource_limits::Deadline;

/// Fully built abstraction artifact that can be used to evaluate concrete states.
#[derive(Debug, Clone)]
pub struct DomainAbstraction {
    pub factory: DomainAbstractionFactory,
    pub distance_table: AbstractDistanceTable,
    pub combine_labels: bool,
    pub relevant_operator_ids: Vec<OperatorIndex>,
    pub abstract_operators: Vec<AbstractOperator>,
    /// `abstract_operators` holds the abstraction's operators; false once
    /// they were discarded or for a lookup-only abstraction. An abstraction
    /// may have no operators at all, so emptiness cannot tell.
    pub transition_data_retained: bool,
    pub abstract_source_sets: Vec<AbstractOperatorSourceSets>,
    pub(crate) interval_transition_system: RefCell<Option<AbstractTransitionSystem>>,
    pub metadata: AbstractionMetadata,
}

impl CegarAbstraction for DomainAbstraction {
    fn abstract_state(
        &self,
        propositional: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize> {
        self.factory.abstract_state(propositional, numeric)
    }

    fn distance_table(&self) -> &AbstractDistanceTable {
        &self.distance_table
    }

    fn metadata(&self) -> &AbstractionMetadata {
        &self.metadata
    }

    fn metadata_mut(&mut self) -> &mut AbstractionMetadata {
        &mut self.metadata
    }

    fn num_abstract_operators(&self) -> usize {
        self.abstract_operators.len()
    }

    fn abstract_source_sets(&self) -> &[AbstractOperatorSourceSets] {
        &self.abstract_source_sets
    }

    fn discard_transition_data(&mut self) {
        self.abstract_operators.clear();
        self.transition_data_retained = false;
        self.abstract_source_sets.clear();
        self.interval_transition_system.get_mut().take();
    }
}

impl DomainAbstraction {
    pub fn ensure_abstract_source_sets(&mut self, task: &dyn AbstractNumericTask) -> Result<()> {
        if !self.abstract_source_sets.is_empty() {
            ensure!(
                self.abstract_source_sets.len() == self.abstract_operators.len(),
                "domain abstraction has {} operator regions for {} abstract operators",
                self.abstract_source_sets.len(),
                self.abstract_operators.len()
            );
            return Ok(());
        }
        ensure!(
            self.transition_data_retained,
            "cannot construct regional operator regions after abstract operators were discarded"
        );
        self.abstract_source_sets = self
            .factory
            .build_abstract_source_sets(task, &self.abstract_operators)
            .context("failed to build abstract-operator regions")?;
        ensure!(
            self.abstract_source_sets.len() == self.abstract_operators.len(),
            "domain operator-region construction produced {} entries for {} abstract operators",
            self.abstract_source_sets.len(),
            self.abstract_operators.len()
        );
        Ok(())
    }

    pub fn interval_transition_system<'a>(
        &'a self,
        task: &dyn AbstractNumericTask,
        deadline: Deadline,
    ) -> Result<Ref<'a, AbstractTransitionSystem>> {
        if self.interval_transition_system.borrow().is_none() {
            let transition_system = self.factory.build_transition_system(
                task,
                &self.abstract_operators,
                StateRegions::Omit,
                deadline,
            )?;
            *self.interval_transition_system.borrow_mut() = Some(transition_system);
        }
        Ok(Ref::map(
            self.interval_transition_system.borrow(),
            |transition_system| {
                transition_system
                    .as_ref()
                    .expect("regional transition system was initialized")
            },
        ))
    }

    /// Label saturation of the abstraction under `operator_costs`; see
    /// [`saturate_labels`].
    pub(crate) fn saturate_labels(
        &self,
        task: &dyn AbstractNumericTask,
        operator_costs: &[f64],
        perimeter_state: Option<usize>,
        deadline: Deadline,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        ensure!(
            self.transition_data_retained,
            "cannot saturate a domain abstraction whose abstract operators were discarded"
        );
        let goal_states =
            self.factory
                .goal_states(task, &self.distance_table.goal_facts, deadline)?;
        let match_tree = MatchTree::new(self.factory.shared_layout(), &self.abstract_operators);
        let mut graph = MatchTreeGraph::new(
            &self.abstract_operators,
            &match_tree,
            self.factory.layout().num_states(),
            &goal_states,
        );
        saturate_labels(&mut graph, operator_costs, perimeter_state, deadline)
    }

    /// Regional saturation of the abstraction under `remaining`; see
    /// [`saturate_regions`]. A transition's labels cost what remains on the
    /// part of their source sets inside its source state.
    pub(crate) fn saturate_regions(
        &self,
        task: &dyn AbstractNumericTask,
        remaining: &RemainingCosts,
        perimeter_state: Option<usize>,
        deadline: Deadline,
    ) -> Result<(Vec<f64>, SourceSetCostAllocation)> {
        let system = self.interval_transition_system(task, deadline)?;
        // Several transitions leave the same state, and both passes of the
        // saturation ask for its region, so each is materialized once.
        let mut regions: Vec<Option<CartesianSet>> = vec![None; system.backward.len()];
        saturate_regions(
            &system,
            &self.abstract_source_sets,
            |transition, source_set| {
                let region = regions[transition.source_state]
                    .get_or_insert_with(|| self.factory.state_region(transition.source_state));
                source_set_within(region, source_set)
            },
            remaining,
            perimeter_state,
            deadline,
        )
    }
}
