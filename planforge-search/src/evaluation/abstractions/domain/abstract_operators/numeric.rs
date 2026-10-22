//! The numeric partition combinations of a concrete operator: the source and
//! target partitions of every refined numeric variable it changes or reads
//! through a comparison precondition.

use anyhow::{Context, Result};
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentEffect, ExplicitFact, NumericType, OperatorIndex, VariableIndex,
};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::interval::Interval;

use super::OPERATOR_GENERATION;
use crate::evaluation::abstractions::common::coordinates::{
    active_comparison_dimensions, coordinate_image,
};
use crate::evaluation::abstractions::domain::abstraction_numeric_var;
use crate::evaluation::abstractions::domain::comparison_inputs::fill_condition_intervals;
use crate::evaluation::abstractions::domain::function::DomainAbstractionFunction;
use crate::resource_limits::Deadline;

/// One numeric combination a concrete operator may induce: the source and
/// target partitions of the refined numeric variables it reads or changes,
/// both ascending by variable.
#[derive(Clone, Debug)]
pub(super) struct NumericTransition {
    pub(super) source_partition_facts: Vec<ExplicitFact>,
    pub(super) target_partition_facts: Vec<ExplicitFact>,
}

/// The numeric transitions of concrete operator `concrete_op_id`, which has
/// these preconditions and assignment effects.
///
/// A refined variable the operator changes contributes the partition pairs it
/// may lead between, and one it only reads through a comparison precondition
/// contributes the identity pairs, so the comparison is evaluated over the
/// precise source partition and combinations that contradict it are dropped.
/// A refined variable the operator neither changes nor reads is left out: its
/// regression offset is 0 and the match tree never queries it, so it is a
/// wildcard of the abstract operator.
pub(super) fn numeric_partition_transitions(
    task: &dyn AbstractNumericTask,
    function: &DomainAbstractionFunction,
    concrete_op_id: OperatorIndex,
    preconditions: &[ExplicitFact],
    assignment_effects: &[AssignmentEffect],
    deadline: Deadline,
) -> Result<Vec<NumericTransition>> {
    deadline.ensure(OPERATOR_GENERATION)?;
    let layout = function.layout();
    let numeric_domain_sizes = layout.numeric_domain_sizes();
    let additive_numeric_views = function.additive_numeric_views();

    let mut assigned = vec![false; numeric_domain_sizes.len()];
    for effect in assignment_effects {
        assigned[effect.affected_var_id().index()] = true;
    }
    let mut read_by_comparison = vec![false; numeric_domain_sizes.len()];
    let mut has_comparison_preconditions = false;
    for pre in preconditions {
        if let Some(tree) = task.numeric_conditions().for_var(pre.var_index()) {
            has_comparison_preconditions = true;
            for var_id in
                active_comparison_dimensions(tree, numeric_domain_sizes, additive_numeric_views)
            {
                read_by_comparison[var_id.index()] = true;
            }
        }
    }

    let mut per_var: Vec<(VariableIndex, Vec<(usize, usize)>)> = Vec::new();
    for (v, &num_parts) in numeric_domain_sizes.iter().enumerate() {
        if num_parts <= 1 {
            continue;
        }
        let var_id = VariableIndex::from_usize(v);
        let changed = if task.numeric_variables()[v].get_type() == &NumericType::Derived {
            let view = additive_numeric_views[v].as_ref().with_context(|| {
                format!("refined derived numeric variable {v} has no additive-view semantics")
            })?;
            view.operator_delta(concrete_op_id).value().abs() >= float_tolerance::ABS_EPSILON
        } else {
            assigned[v]
        };
        if changed {
            per_var.push((
                var_id,
                variable_partition_transitions(task, function, concrete_op_id, var_id),
            ));
        } else if read_by_comparison[v] {
            per_var.push((var_id, (0..num_parts).map(|part| (part, part)).collect()));
        }
    }

    if per_var.is_empty() {
        return Ok(vec![NumericTransition {
            source_partition_facts: Vec::new(),
            target_partition_facts: Vec::new(),
        }]);
    }

    let mut scratch = PartitionComboScratch {
        source_partition_facts: Vec::with_capacity(per_var.len()),
        target_partition_facts: Vec::with_capacity(per_var.len()),
        source_partitions: Vec::with_capacity(per_var.len()),
        source_intervals: Vec::new(),
        transitions: Vec::new(),
    };
    enumerate_partition_combos(
        &PartitionComboEnumeration {
            task,
            function,
            preconditions,
            per_var: &per_var,
            num_props: layout.domain_sizes().len(),
            has_comparison_preconditions,
            deadline,
        },
        0,
        &mut scratch,
    )?;
    Ok(scratch.transitions)
}

/// The partition pairs `(source, target)` of coordinate `var_id` that
/// `concrete_op_id` may lead between, ascending: every partition the image of
/// a source partition intersects.
fn variable_partition_transitions(
    task: &dyn AbstractNumericTask,
    function: &DomainAbstractionFunction,
    concrete_op_id: OperatorIndex,
    var_id: VariableIndex,
) -> Vec<(usize, usize)> {
    let partitions = function.partitions();
    let views = function.additive_numeric_views();
    partitions
        .partitions(var_id)
        .iter()
        .enumerate()
        .flat_map(|(source, &interval)| {
            let image = coordinate_image(task, views, interval, concrete_op_id, var_id);
            partitions
                .intersecting(var_id, image)
                .map(move |target| (source, target))
        })
        .collect()
}

/// One concrete operator's numeric cross product, fixed for the whole
/// enumeration: the per-variable partition pairs the recursion walks, where
/// numeric dimensions start in the state layout, and whether a combination
/// must be checked against comparison preconditions at all.
struct PartitionComboEnumeration<'a> {
    task: &'a dyn AbstractNumericTask,
    function: &'a DomainAbstractionFunction,
    preconditions: &'a [ExplicitFact],
    per_var: &'a [(VariableIndex, Vec<(usize, usize)>)],
    num_props: usize,
    has_comparison_preconditions: bool,
    deadline: Deadline,
}

/// What the recursive frames of one enumeration share: the partition facts and
/// source partitions pushed on the way down and popped on the way back up, the
/// interval buffer a combination is checked over, and the transitions found.
struct PartitionComboScratch {
    source_partition_facts: Vec<ExplicitFact>,
    target_partition_facts: Vec<ExplicitFact>,
    source_partitions: Vec<(VariableIndex, usize)>,
    source_intervals: Vec<Interval>,
    transitions: Vec<NumericTransition>,
}

fn enumerate_partition_combos(
    enumeration: &PartitionComboEnumeration<'_>,
    pos: usize,
    scratch: &mut PartitionComboScratch,
) -> Result<()> {
    let PartitionComboEnumeration {
        task,
        function,
        preconditions,
        per_var,
        num_props,
        has_comparison_preconditions,
        deadline,
    } = *enumeration;
    if pos.is_multiple_of(4) {
        deadline.ensure(OPERATOR_GENERATION)?;
    }
    if pos == per_var.len() {
        if has_comparison_preconditions {
            // A combination no state lies in has no transition.
            if !fill_source_intervals(
                task,
                function,
                &scratch.source_partitions,
                &mut scratch.source_intervals,
            ) || !comparison_preconditions_admit(task, preconditions, &scratch.source_intervals)
            {
                return Ok(());
            }
        }
        // The facts were pushed in ascending variable order, one per
        // variable, so they are sorted as they are.
        debug_assert!(
            scratch
                .source_partition_facts
                .windows(2)
                .all(|w| w[0].var() < w[1].var())
        );
        scratch.transitions.push(NumericTransition {
            source_partition_facts: scratch.source_partition_facts.clone(),
            target_partition_facts: scratch.target_partition_facts.clone(),
        });
        return Ok(());
    }
    let (var_id, ref transitions) = per_var[pos];
    let abs_var_id = abstraction_numeric_var(num_props, var_id).index();
    for &(source, target) in transitions {
        scratch
            .source_partition_facts
            .push(ExplicitFact::numeric_variable(abs_var_id, source));
        scratch
            .target_partition_facts
            .push(ExplicitFact::numeric_variable(abs_var_id, target));
        scratch.source_partitions.push((var_id, source));

        enumerate_partition_combos(enumeration, pos + 1, scratch)?;

        scratch.source_partition_facts.pop();
        scratch.target_partition_facts.pop();
        scratch.source_partitions.pop();
    }
    Ok(())
}

/// Fills `out` with the interval of every numeric variable over these source
/// partitions of a combination, the refined coordinates, and returns whether
/// a state may lie in them; see [`fill_condition_intervals`].
#[must_use]
pub(super) fn fill_source_intervals(
    task: &dyn AbstractNumericTask,
    function: &DomainAbstractionFunction,
    source_partitions: &[(VariableIndex, usize)],
    out: &mut Vec<Interval>,
) -> bool {
    let partitions = function.partitions();
    fill_condition_intervals(
        task,
        |var_id| {
            source_partitions
                .iter()
                .find(|&&(combo_var, _)| combo_var == var_id)
                .map(|&(_, source)| partitions.partition_interval(var_id, source))
        },
        out,
    )
}

/// Whether the comparison preconditions of an operator may hold over these
/// source intervals of a partition combination.
///
/// Condition variables are never refined, so the abstract operator carries
/// no condition facts; the preconditions only filter combinations, by their
/// concrete true or false value.
fn comparison_preconditions_admit(
    task: &dyn AbstractNumericTask,
    preconditions: &[ExplicitFact],
    source_intervals: &[Interval],
) -> bool {
    let conditions = task.numeric_conditions();
    !preconditions
        .iter()
        .filter(|precondition| conditions.is_condition_var(precondition.var_index()))
        .any(|precondition| conditions.precondition_is_contradicted(precondition, source_intervals))
}
