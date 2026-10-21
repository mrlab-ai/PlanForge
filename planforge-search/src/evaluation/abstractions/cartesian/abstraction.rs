use super::*;
use crate::evaluation::abstractions::common::abstraction::{AbstractionCore, CegarAbstraction};
use crate::evaluation::abstractions::common::coordinates::{AdditiveNumericView, coordinate_value};
use crate::evaluation::abstractions::common::regions::SOURCE_SETS;
use crate::evaluation::abstractions::component::AbstractionKind;
use crate::evaluation::abstractions::cost_partitioning::costs::{
    ExplicitSaturationGraph, RemainingCosts, SourceSetCostAllocation, saturate_labels,
    saturate_regions,
};
use crate::resource_limits::Deadline;

/// A Cartesian abstraction: the refinement hierarchy that maps concrete
/// states to abstract ones, and, until discarded, its transitions.
#[derive(Debug, Clone)]
pub struct CartesianAbstraction {
    pub(crate) hierarchy: CartesianRefinementHierarchy,
    /// The additive view of every derived coordinate, which states are
    /// mapped through.
    views: Arc<[Option<AdditiveNumericView>]>,
    core: AbstractionCore,
    pub(super) transitions: Option<CartesianTransitions>,
}

/// The transitions of a Cartesian abstraction, with the region of every
/// abstract state and, once regional saturation was prepared, the source
/// sets of every transition's labels.
#[derive(Debug, Clone)]
pub(crate) struct CartesianTransitions {
    pub(crate) system: AbstractTransitionSystem,
    pub(crate) regions: Vec<Arc<CartesianSet>>,
    pub(crate) source_sets: Option<Vec<AbstractOperatorSourceSets>>,
}

impl CartesianAbstraction {
    pub(crate) fn new(
        hierarchy: CartesianRefinementHierarchy,
        views: Arc<[Option<AdditiveNumericView>]>,
        core: AbstractionCore,
        transitions: Option<CartesianTransitions>,
    ) -> Self {
        Self {
            hierarchy,
            views,
            core,
            transitions,
        }
    }

    fn transitions(&self) -> Result<&CartesianTransitions> {
        self.transitions
            .as_ref()
            .context("the Cartesian abstraction's transitions were discarded")
    }
}

impl CegarAbstraction for CartesianAbstraction {
    fn kind(&self) -> AbstractionKind {
        AbstractionKind::Cartesian
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
        self.hierarchy.map_state_with(propositional, |var_id| {
            coordinate_value(&self.views, numeric, var_id)
        })
    }

    fn discard_transition_data(&mut self) {
        self.transitions = None;
    }

    fn retains_transitions(&self) -> bool {
        self.transitions.is_some()
    }

    /// Every abstract operator is one transition, so its source sets are
    /// those of the transition.
    fn prepare_regional_saturation(
        &mut self,
        regions: &RegionSemantics<'_>,
        deadline: Deadline,
    ) -> Result<()> {
        let transitions = self
            .transitions
            .as_mut()
            .context("cannot build source sets after the transitions were discarded")?;
        if transitions.source_sets.is_some() {
            return Ok(());
        }
        let source_sets = transitions
            .system
            .transitions
            .iter()
            .map(|transition| {
                if transition.transition_id.is_multiple_of(1024) {
                    deadline.ensure(SOURCE_SETS)?;
                }
                regions
                    .abstract_operator_source_sets(
                        &transitions.regions[transition.source_state],
                        &transition.concrete_op_ids,
                        &transitions.regions[transition.target_state],
                    )
                    .with_context(|| format!("transition {}", transition.transition_id))
            })
            .collect::<Result<_>>()?;
        transitions.source_sets = Some(source_sets);
        Ok(())
    }

    fn abstract_source_sets(&self) -> Option<&[AbstractOperatorSourceSets]> {
        self.transitions.as_ref()?.source_sets.as_deref()
    }

    fn saturate_labels(
        &self,
        operator_costs: &[f64],
        perimeter_state: Option<usize>,
        deadline: Deadline,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        saturate_labels(
            &mut ExplicitSaturationGraph::new(&self.transitions()?.system),
            operator_costs,
            perimeter_state,
            deadline,
        )
    }

    fn saturate_regions(
        &self,
        remaining: &RemainingCosts,
        perimeter_state: Option<usize>,
        deadline: Deadline,
    ) -> Result<(Vec<f64>, SourceSetCostAllocation)> {
        let transitions = self.transitions()?;
        let source_sets = transitions
            .source_sets
            .as_deref()
            .context("regional saturation of a Cartesian abstraction needs its source sets")?;
        saturate_regions(
            &transitions.system,
            source_sets,
            |_, source_set| Ok(source_set.clone()),
            remaining,
            perimeter_state,
            deadline,
        )
    }
}
