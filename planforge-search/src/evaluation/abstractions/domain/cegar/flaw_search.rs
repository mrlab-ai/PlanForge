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
use std::collections::BTreeSet;
use std::fmt;

use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, NumericValue, OperatorIndex, VariableIndex,
};
#[cfg(test)]
use planforge_sas::numeric_task::{ExplicitValueIndex, OperatorCost};
use planforge_sas::utils::float_tolerance;

use serde::{Deserialize, Serialize};

use super::determine_include_in_lower;
use crate::evaluation::abstractions::cegar::FlawKind;
use crate::evaluation::abstractions::coordinates::comparison_refinement_dimensions;
use crate::evaluation::abstractions::coordinates::numeric_requirement_for_comparison_fact;
use crate::evaluation::abstractions::domain::abstract_operator_generator::DomainMapping;
use crate::evaluation::abstractions::domain::cegar::determine_include_in_lower_for_flaw_search_state;
use crate::evaluation::abstractions::domain::cegar::flaw_search::progression::{
    get_execute_entire_plan_flaws, get_progression_flaws,
};
use crate::evaluation::abstractions::domain::cegar::flaw_search::regression::get_regression_flaws;
use crate::evaluation::abstractions::domain::cegar::flaw_search::sequence::{
    SequenceDirection, get_sequence_flaws,
};
use crate::evaluation::abstractions::domain::cegar::flaw_search::state::FlawSearchState;
use crate::evaluation::abstractions::domain::factory::WildcardPlanResult;
use crate::evaluation::abstractions::domain::numeric_partitions::NumericPartitions;
use crate::evaluation::abstractions::domain::utils::partition_for_value;

/// Mirrors numeric-FD's `NumericFlaw = tuple<int, ap_float, bool>`.
#[derive(Debug, Clone, PartialEq)]
pub struct NumericFlaw {
    pub numeric_var_id: VariableIndex,
    pub value: NumericValue,
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

/// The flaws of one abstract plan, and the concrete plan the flaw search
/// executed when it found none.
#[derive(Debug, Clone, PartialEq)]
pub struct FlawSearchOutcome {
    pub flaws: Vec<Flaw>,
    /// The concrete operator chosen at every abstract plan step. `Some`
    /// exactly when `flaws` is empty and the flaw search executes the plan
    /// concretely (progression, target-centered, execute-entire-plan, and
    /// regression's progression fallback); the plan then reaches the goal from
    /// the task initial state.
    pub concrete_plan: Option<Vec<OperatorIndex>>,
}

impl FlawSearchOutcome {
    /// Flaws found by a search that does not execute a concrete plan.
    fn without_plan(flaws: Vec<Flaw>) -> Self {
        Self {
            flaws,
            concrete_plan: None,
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
#[derive(Default)]
pub enum SplitDirection {
    #[default]
    Forward,
    ForwardPartitionDeviation,
    Backward,
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
    /// override via [`Self::get_flaws_with_direction`].
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
    ) -> Result<FlawSearchOutcome> {
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
    ) -> Result<FlawSearchOutcome> {
        match self {
            Self::Progression | Self::TargetCentered => {
                get_progression_flaws(task, partitions, wildcard_plan, direction)
            }
            Self::Regression => {
                let flaws = get_regression_flaws(task, partitions, domain_mapping, wildcard_plan)?;
                // Progression flaw fallback if no regression flaw is found
                // (numeric deviation flaws not detected).
                if flaws.is_empty() {
                    get_progression_flaws(task, partitions, wildcard_plan, direction)
                } else {
                    Ok(FlawSearchOutcome::without_plan(flaws))
                }
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
            )
            .map(FlawSearchOutcome::without_plan),
            // Sequence progression flaw fallback already searched here.
            Self::SequenceRegression => get_sequence_flaws(
                task,
                partitions,
                domain_mapping,
                wildcard_plan,
                SequenceDirection::Regression,
            )
            .map(FlawSearchOutcome::without_plan),
            Self::SequenceBidirectional => get_sequence_flaws(
                task,
                partitions,
                domain_mapping,
                wildcard_plan,
                SequenceDirection::Bidirectional,
            )
            .map(FlawSearchOutcome::without_plan),
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
) -> Result<FlawSearchOutcome> {
    flaw_kind.get_flaws(task, partitions, domain_mapping, wildcard_plan)
}

/// The goal facts a flaw search has to reach, deduplicated.
///
/// This used to substitute the bodies of the rules proving a derived goal, since a
/// derived variable is written by no operator and a goal on one is nothing a flaw
/// search can regress through or split on. `validate_abstractable_goal` refuses
/// such a task instead, so what is left is the task's own goal facts, in the order
/// the task lists them -- which is the order the flaws come out in, and therefore
/// the order flaw selection sees.
pub fn goal_facts(task: &dyn AbstractNumericTask) -> Vec<ExplicitFact> {
    let mut seen: BTreeSet<ExplicitFact> = BTreeSet::new();
    (0..task.get_num_goals())
        .map(|goal_id| *task.get_goal_fact(goal_id))
        .filter(|fact| seen.insert(*fact))
        .collect()
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
            .get(nf.numeric_var_id.index())
            .copied()
            .unwrap_or(0),
        Flaw::Propositional(pf) => {
            let var_id = pf.fact.var();
            let base = domain_sizes.get(var_id).copied().unwrap_or(0);
            let max_dep = pf
                .dependent_numeric_flaws
                .iter()
                .filter_map(|nf| numeric_domain_sizes.get(nf.numeric_var_id.index()).copied())
                .max()
                .unwrap_or(0);
            base + max_dep
        }
    }
}

fn dependent_numeric_flaws_for_comparison_prop_var(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    prop_var_id: VariableIndex,
    numeric_state: &[NumericValue],
    step: usize,
) -> Vec<NumericFlaw> {
    let Some(tree) = task.numeric_conditions().for_var(prop_var_id) else {
        return vec![];
    };

    let mut out: Vec<NumericFlaw> = Vec::new();
    for dep_var_id in comparison_refinement_dimensions(task, tree) {
        let Some(&concrete_value) = numeric_state.get(dep_var_id.index()) else {
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
    prop_var_id: VariableIndex,
    state: &FlawSearchState,
    step: usize,
) -> Vec<NumericFlaw> {
    let Some(tree) = task.numeric_conditions().for_var(prop_var_id) else {
        return vec![];
    };

    let mut out: Vec<NumericFlaw> = Vec::new();
    for dep_var_id in comparison_refinement_dimensions(task, tree) {
        let include_in_lower = determine_include_in_lower_for_flaw_search_state(tree, state);

        if can_split_numeric_var(
            partitions,
            dep_var_id,
            state.numeric[dep_var_id.index()].upper,
            include_in_lower,
        ) {
            out.push(NumericFlaw {
                numeric_var_id: dep_var_id,
                value: state.numeric[dep_var_id.index()].upper,
                include_in_lower,
                step,
            });
        } else if can_split_numeric_var(
            partitions,
            dep_var_id,
            state.numeric[dep_var_id.index()].lower,
            !include_in_lower,
        ) {
            out.push(NumericFlaw {
                numeric_var_id: dep_var_id,
                value: state.numeric[dep_var_id.index()].lower,
                include_in_lower: !include_in_lower,
                step,
            });
        }
    }
    out
}

pub(crate) fn can_split_numeric_var(
    partitions: &NumericPartitions,
    numeric_var_id: VariableIndex,
    value: NumericValue,
    include_in_lower: bool,
) -> bool {
    let value = NumericValue::new(f64::from_bits(float_tolerance::canonical_bits(value)));
    let Some(parts) = partitions.partitions(numeric_var_id) else {
        return false;
    };
    let Some(part_id) = partition_for_value(parts, value) else {
        return false;
    };
    parts[part_id].can_split_at(value, include_in_lower)
}

/// A single-variable fixture for the flaw-search tests.
///
/// The task has one propositional variable `v` with `domain_size` values, one
/// operator `set` with precondition `v=0` and effect `v:=1`, and the goal
/// `v=goal_value`. `initial` fixes the concrete initial state, which is how the
/// tests make a wildcard plan computed for one initial state inapplicable in
/// another.
#[cfg(test)]
pub(crate) fn single_switch_task(
    domain_size: usize,
    goal_value: ExplicitValueIndex,
    initial: Vec<ExplicitValueIndex>,
) -> planforge_sas::numeric_task::NumericRootTask {
    use planforge_sas::numeric_task::{
        Effect, ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts, Operator,
    };

    let variable = ExplicitVariable::new(
        domain_size,
        "v".into(),
        (0..domain_size).map(|value| format!("v{value}")).collect(),
        None,
        ExplicitValueIndex::new(0),
    );
    let set = Operator::new(
        "set".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![Effect::new(
            vec![],
            VariableIndex::from_usize(0),
            Some(ExplicitValueIndex::new(0)),
            ExplicitValueIndex::new(1),
        )],
        vec![],
        OperatorCost::new(1),
    );
    NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![variable],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, goal_value.index())],
        mutexes: vec![],
        state: initial,
        numeric_state: vec![],
        operators: vec![set],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}
