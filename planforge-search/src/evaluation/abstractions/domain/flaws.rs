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

use planforge_sas::numeric_conditions::NumericCondition;
use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, NumericValue, OperatorIndex, VariableIndex,
};
#[cfg(test)]
use planforge_sas::numeric_task::{ExplicitValueIndex, OperatorCost};
use planforge_sas::numeric_task::{INF_VALUE, NEG_INF_VALUE};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::interval::Interval;

use serde::{Deserialize, Serialize};

use crate::evaluation::abstractions::common::cegar::FlawKind;
use crate::evaluation::abstractions::common::coordinates::comparison_refinement_dimensions;
use crate::evaluation::abstractions::common::coordinates::numeric_requirement_for_comparison_fact;
use crate::evaluation::abstractions::domain::abstract_operators::DomainMapping;
use crate::evaluation::abstractions::domain::factory::WildcardPlanResult;
use crate::evaluation::abstractions::domain::flaws::progression::{
    get_execute_entire_plan_flaws, get_progression_flaws,
};
use crate::evaluation::abstractions::domain::flaws::regression::get_regression_flaws;
use crate::evaluation::abstractions::domain::flaws::sequence::{
    SequenceDirection, get_sequence_flaws,
};
use crate::evaluation::abstractions::domain::flaws::state::FlawSearchState;
use crate::evaluation::abstractions::domain::partitions::NumericPartitions;
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
    /// regression); the plan then reaches the goal from
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
                get_regression_flaws(task, partitions, domain_mapping, wildcard_plan, direction)
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

/// Which side of a split the boundary value belongs to, given the verdict each
/// half's interval evaluation reaches — `None` where the half straddles the
/// comparison and no verdict follows from it.
///
/// The split point goes to whichever half is further from establishing the
/// condition, so that the half that does establish it stays as small as the
/// split can make it: a decided `false` beats an undecided half, which beats a
/// decided `true`. Two halves at the same rank leave the boundary in the upper
/// one, as numeric-FD does.
///
/// Indeterminacy lives here and nowhere else. It is a property of evaluating a
/// comparison over a box of numeric values, answered per call, not a value a
/// state can hold — which is why the abstract domain of a condition variable is
/// [`ConditionValue::DOMAIN_SIZE`] wide and not one wider.
#[inline]
fn prefers_lower_half(eval_lower: Option<bool>, eval_upper: Option<bool>) -> bool {
    match (eval_lower, eval_upper) {
        (Some(false), Some(false)) => false,
        (Some(false), _) => true,
        (None, Some(true)) => true,
        _ => false,
    }
}

fn determine_include_in_lower(
    tree: &NumericCondition,
    split_var_id: VariableIndex,
    split_value: NumericValue,
    concrete_values: &[NumericValue],
) -> bool {
    let mut lower_inputs: Vec<Interval> = concrete_values
        .iter()
        .copied()
        .map(Interval::singleton)
        .collect();
    let mut upper_inputs = lower_inputs.clone();

    if split_var_id.index() < lower_inputs.len() {
        // If the split point is included in the lower interval, the current concrete
        // value belongs to (-inf, split_value].
        lower_inputs[split_var_id.index()] = Interval::new(NEG_INF_VALUE, split_value, false, true);
    }
    if split_var_id.index() < upper_inputs.len() {
        // If the split point is included in the upper interval, the current concrete
        // value belongs to [split_value, inf).
        upper_inputs[split_var_id.index()] = Interval::new(split_value, INF_VALUE, true, false);
    }

    prefers_lower_half(
        tree.evaluate_interval(&lower_inputs),
        tree.evaluate_interval(&upper_inputs),
    )
}

fn determine_include_in_lower_for_flaw_search_state(
    tree: &NumericCondition,
    state: &FlawSearchState,
) -> bool {
    let lower_inputs: Vec<Interval> = state
        .numeric
        .iter()
        .map(|v| Interval::singleton(v.lower))
        .collect();
    let upper_inputs: Vec<Interval> = state
        .numeric
        .iter()
        .map(|v| Interval::singleton(v.upper))
        .collect();

    prefers_lower_half(
        tree.evaluate_interval(&lower_inputs),
        tree.evaluate_interval(&upper_inputs),
    )
}
