//! Typed-config option machinery shared between this crate (the typed
//! configs themselves) and `planforge-searcher` (the search-spec parser).
//!
//! - [`ConfigArg`] / [`ConfigValue`] / [`ConfigCall`] are the AST nodes the
//!   parser produces (one per `key=value` pair, with optional nested calls).
//! - [`ApplyOptions`] is implemented by each typed config struct (typically
//!   via `#[derive(ApplyOptions)]`) — it walks a `&[ConfigArg]` and writes
//!   each option into the typed config.
//! - [`FromOptionValue`] is the per-type "parse a single option value"
//!   trait. The derive picks it up automatically per field; you just need
//!   one impl per option type. Primitive impls (bool, usize, u64, f64,
//!   `Option<u64>`, `String`) live here; per-enum impls live next to each
//!   enum definition.

use std::collections::HashSet;

pub use planforge_config_derive::ApplyOptions;

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigCall {
    pub name: String,
    pub args: Vec<ConfigArg>,
}

impl ConfigCall {
    pub fn new(name: impl Into<String>, args: Vec<ConfigArg>) -> Self {
        Self {
            name: name.into(),
            args,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn args(&self) -> &[ConfigArg] {
        &self.args
    }

    pub fn into_parts(self) -> (String, Vec<ConfigArg>) {
        (self.name, self.args)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigArg {
    pub key: Option<String>,
    pub value: ConfigValue,
}

impl ConfigArg {
    pub fn new(key: Option<String>, value: ConfigValue) -> Self {
        Self { key, value }
    }

    pub fn key(&self) -> Option<&str> {
        self.key.as_deref()
    }

    pub fn value(&self) -> &ConfigValue {
        &self.value
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValue {
    Atom(String),
    Call(ConfigCall),
}

impl ConfigValue {
    pub fn as_atom(&self) -> Result<&str, String> {
        match self {
            ConfigValue::Atom(value) => Ok(value),
            ConfigValue::Call(call) => Err(format!("expected scalar value, got `{}`", call.name)),
        }
    }

    pub fn as_call(&self) -> Result<&ConfigCall, String> {
        match self {
            ConfigValue::Call(call) => Ok(call),
            ConfigValue::Atom(name) => Err(format!("expected call, got atom `{name}`")),
        }
    }
}

pub fn atom(value: &ConfigValue) -> Result<&str, String> {
    value.as_atom()
}

/// Walk `args` and dispatch each one as either named (`arg.key`) or
/// positional (mapped through `positional_order`). Errors on duplicate
/// keys and positional overflow; unknown keys are the closure's
/// responsibility.
pub fn for_each_option(
    args: &[ConfigArg],
    positional_order: &[&str],
    mut apply: impl FnMut(&str, &ConfigValue) -> Result<(), String>,
) -> Result<(), String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut next_positional = 0usize;
    for arg in args {
        let key: &str = match arg.key() {
            Some(k) => k,
            None => {
                let k = positional_order
                    .get(next_positional)
                    .copied()
                    .ok_or_else(|| {
                        format!(
                            "too many positional arguments (maximum {})",
                            positional_order.len()
                        )
                    })?;
                next_positional += 1;
                k
            }
        };
        if !seen.insert(key.to_string()) {
            return Err(format!("duplicate option `{key}`"));
        }
        apply(key, arg.value())?;
    }
    Ok(())
}

// =============================================================================
// Traits
// =============================================================================

/// Implemented by typed configs that can be populated from a `&[ConfigArg]`.
/// Normally derived via `#[derive(ApplyOptions)]`; written by hand only for
/// configs whose CLI surface differs structurally from the struct layout
/// (e.g. coupled writes, curated subsets).
pub trait ApplyOptions {
    fn apply_options(&mut self, args: &[ConfigArg]) -> Result<(), String>;
}

/// Implemented by every type that can appear as the value of an option.
/// The derive picks `from_option_value` for each field automatically.
pub trait FromOptionValue: Sized {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String>;
}

// =============================================================================
// Primitive impls
// =============================================================================

impl FromOptionValue for bool {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match atom(value)? {
            "true" => Ok(true),
            "false" => Ok(false),
            other => Err(format!("expected boolean, got `{other}`")),
        }
    }
}

impl FromOptionValue for usize {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        atom(value)?
            .parse::<usize>()
            .map_err(|_| format!("expected non-negative integer, got `{}`", atom(value).unwrap()))
    }
}

impl FromOptionValue for u64 {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        atom(value)?
            .parse::<u64>()
            .map_err(|_| format!("expected non-negative integer, got `{}`", atom(value).unwrap()))
    }
}

impl FromOptionValue for f64 {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        let s = atom(value)?;
        if s.eq_ignore_ascii_case("infinity") {
            Ok(f64::INFINITY)
        } else {
            s.parse::<f64>()
                .map_err(|_| format!("expected float or infinity, got `{s}`"))
        }
    }
}

impl FromOptionValue for Option<u64> {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        let s = atom(value)?;
        if s.eq_ignore_ascii_case("none") {
            Ok(None)
        } else {
            s.parse::<u64>()
                .map(Some)
                .map_err(|_| format!("expected non-negative integer or `none`, got `{s}`"))
        }
    }
}

impl FromOptionValue for String {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        Ok(atom(value)?.to_string())
    }
}

// =============================================================================
// Enum impls — kept centralized for now; each impl is mechanical.
//
// Note: adding a new heuristic that introduces a new option-typed enum means
// one new `impl FromOptionValue for X` here. That's it — the derive picks it
// up automatically.
// =============================================================================

use crate::numeric::evaluation::domain_abstractions::cegar::{
    FlawKind, FlawTreatmentVariants, SplitDirection,
};
use crate::numeric::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    InitSplitMethod, InitSplitQuantity, NumericSplitStrategy, PortfolioStrategy, VariableSubset,
};
use crate::numeric::evaluation::domain_abstractions::saturated_cost_partitioning_online_heuristic::{
    OrderGenerator, Saturator, ScoringFunction,
};
use crate::numeric::evaluation::pattern_databases::pattern_database::PdbInternalHeuristic;
use crate::numeric::evaluation::pattern_databases::variable_order_finder::GreedyVariableOrderType;

impl FromOptionValue for GreedyVariableOrderType {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match atom(value)? {
            "cg_goal_level" => Ok(Self::CgGoalLevel),
            "cg_goal_random" => Ok(Self::CgGoalRandom),
            "goal_cg_level" => Ok(Self::GoalCgLevel),
            other => Err(format!("invalid GreedyVariableOrderType `{other}`")),
        }
    }
}

impl FromOptionValue for PdbInternalHeuristic {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match atom(value)? {
            "zero" => Ok(Self::Zero),
            "blind" => Ok(Self::Blind),
            "lmcut" => Ok(Self::Lmcut),
            other => Err(format!("invalid PdbInternalHeuristic `{other}`")),
        }
    }
}

impl FromOptionValue for Saturator {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match atom(value)? {
            "all" => Ok(Self::All),
            "perim" => Ok(Self::Perim),
            "perimstar" => Ok(Self::Perimstar),
            other => Err(format!("invalid Saturator `{other}`")),
        }
    }
}

impl FromOptionValue for ScoringFunction {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match atom(value)? {
            "max_heuristic" => Ok(Self::MaxHeuristic),
            "min_stolen_costs" => Ok(Self::MinStolenCosts),
            "max_heuristic_per_stolen_costs" => Ok(Self::MaxHeuristicPerStolenCosts),
            other => Err(format!("invalid ScoringFunction `{other}`")),
        }
    }
}

impl FromOptionValue for OrderGenerator {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match atom(value)? {
            "greedy_orders" | "greedy_orders()" => Ok(Self::Greedy),
            "dynamic_greedy_orders" | "dynamic_greedy_orders()" => Ok(Self::DynamicGreedy),
            "random_orders" | "random_orders()" => Ok(Self::Random),
            other => Err(format!("invalid OrderGenerator `{other}`")),
        }
    }
}

impl FromOptionValue for VariableSubset {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match atom(value)? {
            "goals" => Ok(Self::Goals),
            "non_goals" => Ok(Self::NonGoals),
            "all" => Ok(Self::All),
            other => Err(format!("invalid VariableSubset `{other}`")),
        }
    }
}

impl FromOptionValue for InitSplitQuantity {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match atom(value)? {
            "none" => Ok(Self::None),
            "single" => Ok(Self::Single),
            "all" => Ok(Self::All),
            other => Err(format!("invalid InitSplitQuantity `{other}`")),
        }
    }
}

impl FromOptionValue for FlawKind {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match atom(value)? {
            "progression" => Ok(Self::Progression),
            "regression" => Ok(Self::Regression),
            "execute_entire_plan" => Ok(Self::ExecuteEntirePlan),
            "sequence_progression" => Ok(Self::SequenceProgression),
            "sequence_regression" => Ok(Self::SequenceRegression),
            "sequence_bidirectional" => Ok(Self::SequenceBidirectional),
            "target_centered" => Ok(Self::TargetCentered),
            other => Err(format!("invalid FlawKind `{other}`")),
        }
    }
}

impl FromOptionValue for FlawTreatmentVariants {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match atom(value)? {
            "random_single_atom" => Ok(Self::RandomSingleAtom),
            "one_split_per_atom" => Ok(Self::OneSplitPerAtom),
            "one_split_per_variable" => Ok(Self::OneSplitPerVariable),
            "max_refined_single_atom" => Ok(Self::MaxRefinedSingleAtom),
            "min_growth_single_atom" => Ok(Self::MinGrowthSingleAtom),
            "max_refined_preferring_prop" => Ok(Self::MaxRefinedPreferringProp),
            "closest_to_goal" => Ok(Self::ClosestToGoal),
            "balance_max_refined_and_closest_to_goal" => {
                Ok(Self::BalanceMaxRefinedAndClosestToGoal)
            }
            "balance_max_refined_preferring_prop_and_closest_to_goal" => {
                Ok(Self::BalanceMaxRefinedPreferringPropAndClosestToGoal)
            }
            other => Err(format!("invalid FlawTreatment `{other}`")),
        }
    }
}

impl FromOptionValue for InitSplitMethod {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match atom(value)? {
            "goal_value" => Ok(Self::GoalValue),
            "goal_value_or_random_if_non_goal" => Ok(Self::GoalValueOrRandomIfNonGoal),
            "init_value" => Ok(Self::InitValue),
            "random_value" => Ok(Self::RandomValue),
            "random_partition" => Ok(Self::RandomPartition),
            "random_binary_partition_separating_init_goal" => {
                Ok(Self::RandomBinaryPartitionSeparatingInitGoal)
            }
            "identity" => Ok(Self::Identity),
            other => Err(format!("invalid InitSplitMethod `{other}`")),
        }
    }
}

impl FromOptionValue for NumericSplitStrategy {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match atom(value)? {
            "standard" => Ok(Self::Standard),
            "exclusion" => Ok(Self::Exclusion),
            other => Err(format!("invalid NumericSplitStrategy `{other}`")),
        }
    }
}

impl FromOptionValue for PortfolioStrategy {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match atom(value)? {
            "standard" => Ok(Self::Standard),
            "complementary" => Ok(Self::Complementary),
            other => Err(format!("invalid PortfolioStrategy `{other}`")),
        }
    }
}

impl FromOptionValue for Option<SplitDirection> {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match atom(value)? {
            "default" => Ok(None),
            "forward" => Ok(Some(SplitDirection::Forward)),
            "forward_partition_deviation" => Ok(Some(SplitDirection::ForwardPartitionDeviation)),
            "backward" => Ok(Some(SplitDirection::Backward)),
            other => Err(format!("invalid SplitDirection `{other}`")),
        }
    }
}
