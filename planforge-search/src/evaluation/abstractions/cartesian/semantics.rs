//! The task as a Cartesian abstraction sees it: the shared CEGAR semantics,
//! and for every split dimension the operators whose transitions it can
//! change, which lets a split update only the transitions it affects.

use anyhow::{Result, ensure};
use planforge_sas::numeric_task::{AbstractNumericTask, NumericType, OperatorIndex, VariableIndex};
use planforge_sas::utils::float_tolerance;

use super::config::CartesianAbstractionConfig;
use super::working::OperatorBitSet;
use crate::evaluation::abstractions::common::cegar::semantics::CegarSemantics;
use crate::evaluation::abstractions::common::cegar::split::SplitDimension;
use crate::evaluation::abstractions::common::regions::RegionSemantics;

mod transitions;

pub(super) use transitions::SplitEnd;

pub(super) struct CartesianSemantics<'task> {
    pub(super) cegar: CegarSemantics<'task>,
    /// Per propositional variable, the dimensions a region's admission of
    /// its facts depends on: the variable itself, the coordinates of its
    /// comparison, or those of the facts its axioms read.
    pub(super) variable_dimensions: Vec<Box<[SplitDimension]>>,
    pub(super) prop_split_dependent_operators: Vec<OperatorBitSet>,
    pub(super) numeric_split_dependent_operators: Vec<OperatorBitSet>,
}

/// The dimensions of `var_id`; see the `variable_dimensions` of
/// [`CartesianSemantics`].
fn variable_dimensions(
    task: &dyn AbstractNumericTask,
    var_id: VariableIndex,
    regions: &RegionSemantics<'_>,
    memo: &mut [Option<Box<[SplitDimension]>>],
    visiting: &mut [bool],
) -> Result<Box<[SplitDimension]>> {
    if let Some(dimensions) = &memo[var_id.index()] {
        return Ok(dimensions.clone());
    }
    let mut dimensions = Vec::new();
    if let Some(condition) = task.numeric_conditions().for_var(var_id) {
        dimensions.extend(
            condition
                .regular_numeric_var_dependencies()
                .iter()
                .copied()
                .chain(regions.coordinates().refinement_dimensions(condition))
                .map(SplitDimension::Numeric),
        );
    } else if regions.axioms_for(var_id).is_empty() {
        dimensions.push(SplitDimension::Propositional(var_id));
    } else {
        ensure!(
            !visiting[var_id.index()],
            "cyclic propositional axiom dependency through variable {var_id}"
        );
        visiting[var_id.index()] = true;
        for &axiom_id in regions.axioms_for(var_id) {
            for condition in task.axioms()[axiom_id].conditions() {
                dimensions.extend(variable_dimensions(
                    task,
                    condition.var_index(),
                    regions,
                    memo,
                    visiting,
                )?);
            }
        }
        visiting[var_id.index()] = false;
    }
    dimensions.sort_unstable();
    dimensions.dedup();
    let dimensions = dimensions.into_boxed_slice();
    memo[var_id.index()] = Some(dimensions.clone());
    Ok(dimensions)
}

impl<'task> CartesianSemantics<'task> {
    pub(super) fn cegar(&self) -> &CegarSemantics<'task> {
        &self.cegar
    }

    pub(super) fn task(&self) -> &'task dyn AbstractNumericTask {
        self.cegar.task()
    }

    pub(super) fn regions(&self) -> &RegionSemantics<'task> {
        self.cegar.regions()
    }

    pub(super) fn new(
        task: &'task dyn AbstractNumericTask,
        config: &CartesianAbstractionConfig,
    ) -> Result<Self> {
        let cegar = CegarSemantics::for_options(task, &config.cegar)?;
        let regions = cegar.regions();
        let additive_numeric_views = cegar.coordinates().views();
        let mut memo = vec![None; task.get_num_variables()];
        let mut visiting = vec![false; task.get_num_variables()];
        let variable_dimensions = (0..task.get_num_variables())
            .map(|var_id| {
                variable_dimensions(
                    task,
                    VariableIndex::from_usize(var_id),
                    regions,
                    &mut memo,
                    &mut visiting,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let operator_count = task.get_operators().len();
        let mut prop_split_dependent_operators = (0..task.get_num_variables())
            .map(|_| OperatorBitSet::empty(operator_count))
            .collect::<Vec<_>>();
        let mut numeric_split_dependent_operators = (0..task.numeric_variables().len())
            .map(|_| OperatorBitSet::empty(operator_count))
            .collect::<Vec<_>>();
        let mut dimensions = Vec::new();
        for (op_id, op) in task.get_operators().iter().enumerate() {
            let op_id = OperatorIndex::from_usize(op_id);
            dimensions.clear();
            for precondition in op.preconditions() {
                dimensions
                    .extend_from_slice(&variable_dimensions[precondition.var_index().index()]);
            }
            for effect in op.effects() {
                let var_id = effect.var_id();
                if !task.numeric_conditions().is_condition_var(var_id)
                    && regions.axioms_for(var_id).is_empty()
                {
                    dimensions.push(SplitDimension::Propositional(var_id));
                }
            }
            for effect in op.assignment_effects() {
                let var_id = effect.affected_var_id();
                if task.numeric_variables()[var_id.index()].get_type() == &NumericType::Regular {
                    dimensions.push(SplitDimension::Numeric(var_id));
                }
            }
            for (numeric_var_id, view) in additive_numeric_views.iter().enumerate() {
                if view.as_ref().is_some_and(|view| {
                    view.operator_delta(op_id).value().abs() > float_tolerance::SEARCH_EPSILON
                }) {
                    dimensions.push(SplitDimension::Numeric(VariableIndex::from_usize(
                        numeric_var_id,
                    )));
                }
            }
            for &dimension in &dimensions {
                match dimension {
                    SplitDimension::Propositional(var_id) => {
                        prop_split_dependent_operators[var_id.index()].insert(op_id);
                    }
                    SplitDimension::Numeric(var_id) => {
                        numeric_split_dependent_operators[var_id.index()].insert(op_id);
                    }
                }
            }
        }
        Ok(Self {
            cegar,
            variable_dimensions,
            prop_split_dependent_operators,
            numeric_split_dependent_operators,
        })
    }
}
