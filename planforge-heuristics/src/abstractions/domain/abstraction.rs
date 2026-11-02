//! A built domain abstraction: the function that maps concrete states to
//! abstract ones, its goal distances, and, until discarded, its transitions.

use crate::abstractions::build_error;
use anyhow::{Context, Result};
use planforge_sas::explicit_state::StateValues;
use planforge_search::heuristic_features::HeuristicBuildError;

use super::function::{DomainAbstractionFunction, DomainTransitions, source_set_within};
use crate::abstractions::common::abstraction::{AbstractionCore, CegarAbstraction};
use crate::abstractions::common::cartesian_set::CartesianSet;
use crate::abstractions::common::regions::RegionSemantics;
use crate::abstractions::common::transition_system::{
    AbstractOperatorSourceSets, AbstractTransitionSystem,
};
use crate::abstractions::component::AbstractionKind;
use crate::abstractions::cost_partitioning::costs::{
    RemainingCosts, SourceSetCostAllocation, saturate_labels, saturate_regions,
};
use planforge_search::resource_limits::Deadline;

#[derive(Debug, Clone)]
pub struct DomainAbstraction {
    function: DomainAbstractionFunction,
    core: AbstractionCore,
    transitions: Option<DomainTransitionData>,
}

/// The transitions of a domain abstraction and, once regional saturation was
/// prepared, the explicit transition system it saturates with the source sets
/// of every abstract operator's labels.
#[derive(Debug, Clone)]
struct DomainTransitionData {
    transitions: DomainTransitions,
    regional: Option<(AbstractTransitionSystem, Vec<AbstractOperatorSourceSets>)>,
}

impl DomainAbstraction {
    pub(crate) fn new(
        function: DomainAbstractionFunction,
        core: AbstractionCore,
        transitions: Option<DomainTransitions>,
    ) -> Self {
        Self {
            function,
            core,
            transitions: transitions.map(|transitions| DomainTransitionData {
                transitions,
                regional: None,
            }),
        }
    }

    /// The abstraction function: which abstract value every concrete value
    /// maps to.
    pub fn function(&self) -> &DomainAbstractionFunction {
        &self.function
    }

    /// The abstract operators and goal states; fails once discarded.
    pub(crate) fn transitions(&self) -> Result<&DomainTransitions> {
        self.transitions
            .as_ref()
            .map(|data| &data.transitions)
            .context("the domain abstraction's transitions were discarded")
    }
}

impl CegarAbstraction for DomainAbstraction {
    fn kind(&self) -> AbstractionKind {
        AbstractionKind::Domain
    }

    fn core(&self) -> &AbstractionCore {
        &self.core
    }

    fn core_mut(&mut self) -> &mut AbstractionCore {
        &mut self.core
    }

    fn abstract_state(&self, state: StateValues<'_>) -> usize {
        self.function.abstract_state(state)
    }

    fn discard_transitions(&mut self) {
        self.transitions = None;
    }

    fn retains_transitions(&self) -> bool {
        self.transitions.is_some()
    }

    fn prepare_regional_saturation(
        &mut self,
        regions: &RegionSemantics<'_>,
        deadline: Deadline,
    ) -> Result<(), HeuristicBuildError> {
        self.build_source_sets(regions, deadline)
            .map_err(build_error)
    }

    fn abstract_source_sets(&self) -> Option<&[AbstractOperatorSourceSets]> {
        let (_, source_sets) = self.transitions.as_ref()?.regional.as_ref()?;
        Some(source_sets)
    }

    fn saturate_labels(
        &self,
        operator_costs: &[f64],
        perimeter_state: Option<usize>,
        deadline: Deadline,
    ) -> Result<(Vec<f64>, Vec<f64>), HeuristicBuildError> {
        self.label_saturation(operator_costs, perimeter_state, deadline)
            .map_err(build_error)
    }

    /// A transition's labels cost what remains on the part of their source
    /// sets inside its source state.
    fn saturate_regions(
        &self,
        remaining: &RemainingCosts,
        perimeter_state: Option<usize>,
        deadline: Deadline,
    ) -> Result<(Vec<f64>, SourceSetCostAllocation), HeuristicBuildError> {
        self.regional_saturation(remaining, perimeter_state, deadline)
            .map_err(build_error)
    }
}

impl DomainAbstraction {
    fn regional_saturation(
        &self,
        remaining: &RemainingCosts,
        perimeter_state: Option<usize>,
        deadline: Deadline,
    ) -> Result<(Vec<f64>, SourceSetCostAllocation)> {
        let (system, source_sets) = self
            .transitions
            .as_ref()
            .and_then(|data| data.regional.as_ref())
            .context("regional saturation of a domain abstraction needs its source sets")?;
        // Several transitions leave the same state, so each state's region is
        // materialized once.
        let mut regions: Vec<Option<CartesianSet>> = vec![None; system.backward.len()];
        saturate_regions(
            system,
            source_sets,
            |transition, source_set| {
                let region = regions[transition.source_state]
                    .get_or_insert_with(|| self.function.state_region(transition.source_state));
                source_set_within(region, source_set)
            },
            remaining,
            perimeter_state,
            deadline,
        )
    }

    fn label_saturation(
        &self,
        operator_costs: &[f64],
        perimeter_state: Option<usize>,
        deadline: Deadline,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        saturate_labels(
            &mut self.transitions()?.graph(),
            operator_costs,
            perimeter_state,
            deadline,
        )
    }

    fn build_source_sets(
        &mut self,
        regions: &RegionSemantics<'_>,
        deadline: Deadline,
    ) -> Result<()> {
        let data = self
            .transitions
            .as_mut()
            .context("cannot build source sets after the transitions were discarded")?;
        if data.regional.is_some() {
            return Ok(());
        }
        let system = data
            .transitions
            .explicit_system(deadline)
            .context("failed to build the explicit transition system")?;
        let source_sets = self.function.build_abstract_source_sets(
            regions,
            &data.transitions.operators,
            deadline,
        )?;
        data.regional = Some((system, source_sets));
        Ok(())
    }
}
