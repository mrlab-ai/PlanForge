//! A built domain abstraction: the factory that maps concrete states to
//! abstract ones, its goal distances, and, until discarded, its transitions.

use anyhow::{Context, Result};

use planforge_sas::numeric_task::{ExplicitValueIndex, NumericValue};

use super::factory::{DomainAbstractionFactory, DomainTransitions, source_set_within};
use crate::evaluation::abstractions::common::abstraction::{AbstractionCore, CegarAbstraction};
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use crate::evaluation::abstractions::common::regions::RegionSemantics;
use crate::evaluation::abstractions::common::transition_system::{
    AbstractOperatorSourceSets, AbstractTransitionSystem,
};
use crate::evaluation::abstractions::component::AbstractionKind;
use crate::evaluation::abstractions::cost_partitioning::costs::{
    RemainingCosts, SourceSetCostAllocation, saturate_labels, saturate_regions,
};
use crate::resource_limits::Deadline;

#[derive(Debug, Clone)]
pub struct DomainAbstraction {
    factory: DomainAbstractionFactory,
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
        factory: DomainAbstractionFactory,
        core: AbstractionCore,
        transitions: Option<DomainTransitions>,
    ) -> Self {
        Self {
            factory,
            core,
            transitions: transitions.map(|transitions| DomainTransitionData {
                transitions,
                regional: None,
            }),
        }
    }

    /// The abstraction function: which abstract value every concrete value
    /// maps to.
    pub fn factory(&self) -> &DomainAbstractionFactory {
        &self.factory
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

    fn abstract_state(
        &self,
        propositional: &[ExplicitValueIndex],
        numeric: &[NumericValue],
    ) -> Result<usize> {
        self.factory.abstract_state(propositional, numeric)
    }

    fn discard_transition_data(&mut self) {
        self.transitions = None;
    }

    fn retains_transitions(&self) -> bool {
        self.transitions.is_some()
    }

    fn prepare_regional_saturation(
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
        let source_sets = self.factory.build_abstract_source_sets(
            regions,
            &data.transitions.operators,
            deadline,
        )?;
        data.regional = Some((system, source_sets));
        Ok(())
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
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        saturate_labels(
            &mut self.transitions()?.graph(),
            operator_costs,
            perimeter_state,
            deadline,
        )
    }

    /// A transition's labels cost what remains on the part of their source
    /// sets inside its source state.
    fn saturate_regions(
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
                    .get_or_insert_with(|| self.factory.state_region(transition.source_state));
                source_set_within(region, source_set)
            },
            remaining,
            perimeter_state,
            deadline,
        )
    }
}
