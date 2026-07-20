//! CEGAR flaw search and refinement-value selection.
//!
//! The flaw-emission walk is shared between progression and target-centered
//! enumeration: [`progression::get_progression_flaws`] iterates the wildcard
//! plan once and dispatches the per-flaw split-value choice through a
//! [`SplitDirection`] parameter. `Forward` reproduces the concrete-value
//! split used by classical progression CEGAR; `Backward` reuses the
//! goal-directed boundary primitives in [`target_centered`] to place each
//! split at the boundary derived from the regressed-target / required
//! interval.
//!
//! `SplitDirection` controls refinement only. Transition cost partitioning
//! separately tracks exact source regions, including unbounded regions.

pub mod flaw_selection;
pub mod progression;
pub mod regression;
pub mod sequence;
pub mod state;
pub mod target_centered;
#[cfg(test)]
mod tests;

use anyhow::Result;
use std::fmt;

use planforge_sas::numeric_task::{AbstractNumericTask, ExplicitFact};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::linear_effects::{LinearExpression, linearize_numeric_var};

use serde::{Deserialize, Serialize};

use super::determine_include_in_lower;
use crate::evaluation::cegar::FlawKind;
use crate::evaluation::domain_abstractions::abstract_operator_generator::DomainMapping;
use crate::evaluation::domain_abstractions::additive_numeric_views::{
    comparison_refinement_dimensions, is_refinable_numeric_dimension,
};
use crate::evaluation::domain_abstractions::cegar::determine_include_in_lower_for_flaw_search_state;
use crate::evaluation::domain_abstractions::cegar::flaw_search::progression::{
    get_execute_entire_plan_flaws, get_progression_flaws,
};
use crate::evaluation::domain_abstractions::cegar::flaw_search::regression::get_regression_flaws;
use crate::evaluation::domain_abstractions::cegar::flaw_search::sequence::{
    SequenceDirection, get_sequence_flaws,
};
use crate::evaluation::domain_abstractions::cegar::flaw_search::state::FlawSearchState;
use crate::evaluation::domain_abstractions::comparison_expression::{CompOp, Interval};
use crate::evaluation::domain_abstractions::domain_abstraction::{
    ComparisonAxiomIndex, NumericPartitions,
};
use crate::evaluation::domain_abstractions::domain_abstraction_factory::WildcardPlanResult;
use crate::evaluation::domain_abstractions::utils::partition_for_value;

/// Mirrors numeric-FD's `NumericFlaw = tuple<int, ap_float, bool>`.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericFlaw {
    pub numeric_var_id: usize,
    pub value: f64,
    pub include_in_lower: bool,
    pub step: usize,
}

/// Mirrors numeric-FD's `PropFlaw = pair<Fact, vector<NumericFlaw>>`.
#[derive(Debug, Clone, PartialEq)]
pub struct PropFlaw {
    pub fact: ExplicitFact,
    pub dependent_numeric_flaws: Vec<NumericFlaw>,
    pub step: usize,
}

/// Mirrors numeric-FD's `Flaw = variant<PropFlaw, NumericFlaw>`.
#[derive(Debug, Clone, PartialEq)]
pub enum Flaw {
    Propositional(PropFlaw),
    Numeric(NumericFlaw),
}
impl Flaw {
    pub fn step(&self) -> usize {
        match self {
            Self::Propositional(prop) => prop.step,
            Self::Numeric(num) => num.step,
        }
    }
}

/// Direction of the split value computed for a numeric flaw.
///
/// `Forward` matches today's progression behavior: when a precondition,
/// goal, or deviation flaw is detected, the split is placed at the *concrete*
/// value that produced the flaw. `Backward` matches the target-centered
/// behavior: the split is placed at the *boundary* derived from the
/// regressed target / required interval, separating the cell containing the
/// goal-directed region from the cell that does not.
///
/// The direction is orthogonal to [`FlawKind`]: it selects how a split value
/// is chosen, not which iteration of the flaw search is run.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SplitDirection {
    Forward,
    ForwardPartitionDeviation,
    Backward,
}

impl Default for SplitDirection {
    fn default() -> Self {
        SplitDirection::Forward
    }
}

impl fmt::Display for SplitDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forward => write!(f, "forward"),
            Self::ForwardPartitionDeviation => write!(f, "forward_partition_deviation"),
            Self::Backward => write!(f, "backward"),
        }
    }
}

impl crate::config::sealed::Sealed for Option<SplitDirection> {}

impl crate::config::FromOptionValue for Option<SplitDirection> {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "default" => Ok(None),
            "forward" => Ok(Some(SplitDirection::Forward)),
            "forward_partition_deviation" => Ok(Some(SplitDirection::ForwardPartitionDeviation)),
            "backward" => Ok(Some(SplitDirection::Backward)),
            other => Err(format!("invalid SplitDirection `{other}`")),
        }
    }
}

impl FlawKind {
    /// Default split-value direction associated with this flaw-search variant.
    ///
    /// `TargetCentered` defaults to `Backward` (boundary splits); all other
    /// variants default to `Forward` (concrete-value splits). Callers may
    /// override via [`get_flaws_with_direction`].
    pub fn default_split_direction(self) -> SplitDirection {
        match self {
            Self::TargetCentered => SplitDirection::Backward,
            _ => SplitDirection::Forward,
        }
    }

    pub fn get_flaws(
        &self,
        task: &dyn AbstractNumericTask,
        partitions: &NumericPartitions,
        domain_mapping: &DomainMapping,
        wildcard_plan: &WildcardPlanResult,
    ) -> Result<Vec<Flaw>> {
        self.get_flaws_with_direction(
            task,
            partitions,
            domain_mapping,
            wildcard_plan,
            self.default_split_direction(),
        )
    }

    pub fn get_flaws_with_direction(
        &self,
        task: &dyn AbstractNumericTask,
        partitions: &NumericPartitions,
        domain_mapping: &DomainMapping,
        wildcard_plan: &WildcardPlanResult,
        direction: SplitDirection,
    ) -> Result<Vec<Flaw>> {
        match self {
            Self::Progression | Self::TargetCentered => {
                get_progression_flaws(task, partitions, wildcard_plan, direction)
            }
            Self::Regression => {
                let mut flaws =
                    get_regression_flaws(task, partitions, domain_mapping, wildcard_plan);
                // Progression flaw fallback if no regression flaw is found
                // (numeric deviation flaws not detected).
                if let Ok(ref flaws_ok) = flaws
                    && flaws_ok.is_empty()
                {
                    flaws = get_progression_flaws(task, partitions, wildcard_plan, direction);
                }

                flaws
            }
            Self::ExecuteEntirePlan => {
                get_execute_entire_plan_flaws(task, partitions, wildcard_plan, direction)
            }
            Self::SequenceProgression => get_sequence_flaws(
                task,
                partitions,
                domain_mapping,
                wildcard_plan,
                SequenceDirection::Progression,
            ),
            // Sequence progression flaw fallback already searched here.
            Self::SequenceRegression => get_sequence_flaws(
                task,
                partitions,
                domain_mapping,
                wildcard_plan,
                SequenceDirection::Regression,
            ),
            Self::SequenceBidirectional => get_sequence_flaws(
                task,
                partitions,
                domain_mapping,
                wildcard_plan,
                SequenceDirection::Bidirectional,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependentNumericRefinement {
    None,
    One,
    All,
}

pub fn get_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    domain_mapping: &DomainMapping,
    wildcard_plan: &WildcardPlanResult,
    flaw_kind: FlawKind,
) -> Result<Vec<Flaw>> {
    flaw_kind.get_flaws(task, partitions, domain_mapping, wildcard_plan)
}

#[allow(unused)]
fn score_flaw(
    flaw: &Flaw,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    _abstraction_size: usize,
) -> usize {
    match flaw {
        Flaw::Numeric(nf) => numeric_domain_sizes
            .get(nf.numeric_var_id)
            .copied()
            .unwrap_or(0),
        Flaw::Propositional(pf) => {
            let var_id = pf.fact.var();
            let base = domain_sizes.get(var_id).copied().unwrap_or(0);
            let max_dep = pf
                .dependent_numeric_flaws
                .iter()
                .filter_map(|nf| numeric_domain_sizes.get(nf.numeric_var_id).copied())
                .max()
                .unwrap_or(0);
            base + max_dep
        }
    }
}

fn dependent_numeric_flaws_for_comparison_prop_var(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    prop_var_id: usize,
    numeric_state: &[f64],
    step: usize,
) -> Vec<NumericFlaw> {
    let Some(tree) = comparison_index.comparison_tree(prop_var_id) else {
        return vec![];
    };

    let mut out: Vec<NumericFlaw> = Vec::new();
    for dep_var_id in comparison_refinement_dimensions(task, tree) {
        let Some(&concrete_value) = numeric_state.get(dep_var_id) else {
            continue;
        };
        let include_in_lower =
            determine_include_in_lower(tree, dep_var_id, concrete_value, numeric_state);

        if can_split_numeric_var(partitions, dep_var_id, concrete_value, include_in_lower) {
            out.push(NumericFlaw {
                numeric_var_id: dep_var_id,
                value: concrete_value,
                include_in_lower,
                step,
            });
        } else if can_split_numeric_var(partitions, dep_var_id, concrete_value, !include_in_lower) {
            out.push(NumericFlaw {
                numeric_var_id: dep_var_id,
                value: concrete_value,
                include_in_lower: !include_in_lower,
                step,
            });
        }
    }
    out
}

fn dependent_numeric_flaws_in_interval_for_comparison_prop_var(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    comparison_index: &ComparisonAxiomIndex,
    prop_var_id: usize,
    state: &FlawSearchState,
    step: usize,
) -> Vec<NumericFlaw> {
    let Some(tree) = comparison_index.comparison_tree(prop_var_id) else {
        return vec![];
    };

    let mut out: Vec<NumericFlaw> = Vec::new();
    for dep_var_id in comparison_refinement_dimensions(task, tree) {
        let include_in_lower = determine_include_in_lower_for_flaw_search_state(tree, state);

        if can_split_numeric_var(
            partitions,
            dep_var_id,
            state.numeric[dep_var_id].upper,
            include_in_lower,
        ) {
            out.push(NumericFlaw {
                numeric_var_id: dep_var_id,
                value: state.numeric[dep_var_id].upper,
                include_in_lower,
                step,
            });
        } else if can_split_numeric_var(
            partitions,
            dep_var_id,
            state.numeric[dep_var_id].lower,
            !include_in_lower,
        ) {
            out.push(NumericFlaw {
                numeric_var_id: dep_var_id,
                value: state.numeric[dep_var_id].lower,
                include_in_lower: !include_in_lower,
                step,
            });
        }
    }
    out
}

pub(crate) fn numeric_requirement_for_comparison_fact(
    task: &dyn AbstractNumericTask,
    comparison_index: &ComparisonAxiomIndex,
    fact: &ExplicitFact,
) -> Option<(usize, Interval)> {
    let tree = comparison_index.comparison_tree(fact.var())?;
    let required_op = required_comparison_op(tree.op, fact.value())?;
    let left = linearize_numeric_var(task, tree.left_numeric_var_id).ok()?;
    let right = linearize_numeric_var(task, tree.right_numeric_var_id).ok()?;
    let num_numeric_vars = task.numeric_variables().len();
    if is_refinable_numeric_dimension(task, tree.left_numeric_var_id) && right.is_constant() {
        let expression =
            LinearExpression::variable(num_numeric_vars, tree.left_numeric_var_id).subtract(
                &LinearExpression::constant(num_numeric_vars, right.constant),
            );
        if let Some(requirement) =
            single_var_interval_for_linear_zero_comparison(&expression, required_op)
        {
            return Some(requirement);
        }
    }
    if is_refinable_numeric_dimension(task, tree.right_numeric_var_id) && left.is_constant() {
        let expression = LinearExpression::constant(num_numeric_vars, left.constant).subtract(
            &LinearExpression::variable(num_numeric_vars, tree.right_numeric_var_id),
        );
        if let Some(requirement) =
            single_var_interval_for_linear_zero_comparison(&expression, required_op)
        {
            return Some(requirement);
        }
    }
    let expression = left.subtract(&right);
    single_var_interval_for_linear_zero_comparison(&expression, required_op)
}

fn required_comparison_op(op: CompOp, prop_value: usize) -> Option<CompOp> {
    match prop_value {
        0 => Some(op),
        1 => Some(match op {
            CompOp::Lt => CompOp::Ge,
            CompOp::Le => CompOp::Gt,
            CompOp::Gt => CompOp::Le,
            CompOp::Ge => CompOp::Lt,
            CompOp::Eq => CompOp::Ne,
            CompOp::Ne => CompOp::Eq,
        }),
        _ => None,
    }
}

fn single_var_interval_for_linear_zero_comparison(
    expression: &LinearExpression,
    op: CompOp,
) -> Option<(usize, Interval)> {
    if op == CompOp::Ne {
        return None;
    }

    let mut non_zero_coefficients = expression
        .coefficients
        .iter()
        .enumerate()
        .filter(|(_, coefficient)| coefficient.abs() >= 1e-12);
    let (numeric_var_id, coefficient) = non_zero_coefficients.next()?;
    if non_zero_coefficients.next().is_some() {
        return None;
    }

    let threshold = -expression.constant / *coefficient;
    if !threshold.is_finite() {
        return None;
    }

    let interval = match (op, coefficient.is_sign_positive()) {
        (CompOp::Lt, true) | (CompOp::Gt, false) => {
            Interval::new(f64::NEG_INFINITY, threshold, false, false)
        }
        (CompOp::Le, true) | (CompOp::Ge, false) => {
            Interval::new(f64::NEG_INFINITY, threshold, false, true)
        }
        (CompOp::Gt, true) | (CompOp::Lt, false) => {
            Interval::new(threshold, f64::INFINITY, false, false)
        }
        (CompOp::Ge, true) | (CompOp::Le, false) => {
            Interval::new(threshold, f64::INFINITY, true, false)
        }
        (CompOp::Eq, _) => Interval::singleton(threshold),
        (CompOp::Ne, _) => return None,
    };
    Some((numeric_var_id, interval))
}

pub(crate) fn can_split_numeric_var(
    partitions: &NumericPartitions,
    numeric_var_id: usize,
    value: f64,
    include_in_lower: bool,
) -> bool {
    let value = f64::from_bits(float_tolerance::canonical_bits(value));
    let Some(parts) = partitions.partitions(numeric_var_id) else {
        return false;
    };
    let Some(part_id) = partition_for_value(parts, value) else {
        return false;
    };
    parts[part_id].can_split_at(value, include_in_lower)
}
