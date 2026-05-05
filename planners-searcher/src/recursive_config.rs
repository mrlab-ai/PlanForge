use planners_search::numeric::evaluation::domain_abstractions::cegar::{
    FlawKind, FlawTreatmentVariants,
};
use serde::{Deserialize, Serialize};
use std::fmt;

use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegarConfig,
    InitSplitMethod, InitSplitQuantity, NumericSplitStrategy, PortfolioStrategy, VariableSubset,
};
use planners_search::numeric::evaluation::domain_abstractions::saturated_cost_partitioning_online_heuristic::{
    Saturator, ScpOnlineConfig,
};
use planners_search::numeric::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig;
use planners_search::numeric::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbConfig;
use planners_search::numeric::evaluation::pattern_databases::pattern_database::PdbInternalHeuristic;
use planners_search::numeric::evaluation::pattern_databases::pattern_generator_greedy::GreedyPatternGeneratorConfig;
use planners_search::numeric::evaluation::pattern_databases::variable_order_finder::GreedyVariableOrderType;
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct DomainAbstractionConfig {
    pub max_abstraction_size: usize,
    pub max_iterations: usize,
    pub use_wildcard_plans: bool,
    pub combine_labels: bool,
    pub random_seed: Option<u64>,
    pub flaw_kind: FlawKind,
    pub flaw_treatment: FlawTreatmentVariants,
    pub init_split_method: InitSplitMethod,
    pub transform_linear_task: bool,
}

impl Default for DomainAbstractionConfig {
    fn default() -> Self {
        Self {
            max_abstraction_size: usize::MAX,
            max_iterations: 10_000,
            use_wildcard_plans: true,
            combine_labels: true,
            random_seed: None,
            flaw_kind: FlawKind::Progression,
            flaw_treatment: FlawTreatmentVariants::RandomSingleAtom,
            init_split_method: InitSplitMethod::InitValue,
            transform_linear_task: false,
        }
    }
}

impl fmt::Display for DomainAbstractionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            concat!(
                "max_abstraction_size={}, ",
                "max_iterations={}, ",
                "use_wildcard_plans={}, ",
                "combine_labels={}, ",
                "random_seed={}, ",
                "flaw_kind={}, ",
                "flaw_treatment={}, ",
                "init_split_method={}, ",
                "transform_linear_task={}, ",
            ),
            self.max_abstraction_size,
            self.max_iterations,
            self.use_wildcard_plans,
            self.combine_labels,
            format_optional_seed(self.random_seed),
            self.flaw_kind,
            self.flaw_treatment,
            self.init_split_method,
            self.transform_linear_task,
        )
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum HeuristicSpec {
    Blind,
    #[serde(rename = "canonical_domain_abstractions")]
    CanonicalDomainAbstractions(DomainAbstractionCollectionGeneratorMultipleCegarConfig),
    #[serde(rename = "domain_abstraction")]
    DomainAbstraction(DomainAbstractionConfig),
    #[serde(rename = "canonical_numeric_pdb")]
    CanonicalNumericPdb(CanonicalNumericPdbConfig),
    #[serde(rename = "greedy_numeric_pdb")]
    GreedyNumericPdb(GreedyPatternGeneratorConfig),
    #[serde(rename = "lmcutnumeric")]
    Lmcutnumeric(LmCutNumericConfig),
    #[serde(rename = "multi_domain_abstractions")]
    MultiDomainAbstractions(DomainAbstractionCollectionGeneratorMultipleCegarConfig),
    #[serde(rename = "scp_online")]
    ScpOnline(ScpOnlineConfig),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SearchSpec {
    Astar(HeuristicSpec),
    #[serde(rename = "da_debug")]
    DaDebug,
    #[serde(rename = "astar_da_debug")]
    AstarDaDebug,
}

impl fmt::Display for HeuristicSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeuristicSpec::Blind => write!(f, "blind()"),
            HeuristicSpec::CanonicalDomainAbstractions(config) => {
                write!(
                    f,
                    "{}",
                    config.format_config_call("canonical_domain_abstractions")
                )
            }
            HeuristicSpec::DomainAbstraction(config) => {
                write!(f, "{}", config.format_config_call("domain_abstraction"))
            }
            HeuristicSpec::CanonicalNumericPdb(config) => {
                write!(f, "{}", config.format_config_call("canonical_numeric_pdb"))
            }
            HeuristicSpec::GreedyNumericPdb(config) => {
                write!(f, "{}", config.format_config_call("greedy_numeric_pdb"))
            }
            HeuristicSpec::Lmcutnumeric(config) => {
                write!(f, "{}", config.format_config_call("lmcutnumeric"))
            }
            HeuristicSpec::MultiDomainAbstractions(config) => {
                write!(
                    f,
                    "multi_domain_abstractions({})",
                    config.format_config_args()
                )
            }
            HeuristicSpec::ScpOnline(config) => {
                write!(f, "{}", config.format_config_call("scp_online"))
            }
        }
    }
}

impl fmt::Display for SearchSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchSpec::Astar(h) => write!(f, "astar({h})"),
            SearchSpec::DaDebug => write!(f, "da_debug()"),
            SearchSpec::AstarDaDebug => write!(f, "astar_da_debug()"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct AStarConfig {
    heuristic: HeuristicSpec,
}

impl Default for AStarConfig {
    fn default() -> Self {
        Self {
            heuristic: HeuristicSpec::Blind,
        }
    }
}

pub fn parse_search_spec(raw: &str) -> Result<SearchSpec, String> {
    let mut input = raw.trim();
    input = input
        .strip_suffix('.')
        .or_else(|| input.strip_suffix(';'))
        .unwrap_or(input)
        .trim();

    let call = ConfigParser::new(input).parse_all()?;
    build_search_spec(&call)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigCall {
    name: String,
    args: Vec<ConfigArg>,
}

impl ConfigCall {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn args(&self) -> &[ConfigArg] {
        &self.args
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigArg {
    key: Option<String>,
    value: ConfigValue,
}

impl ConfigArg {
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
}

struct ConfigParser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> ConfigParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn parse_all(mut self) -> Result<ConfigCall, String> {
        let call = self.parse_call_or_bare()?;
        self.skip_ws();
        if self.pos != self.input.len() {
            return Err(format!(
                "Invalid --search config: unexpected input at byte {} near `{}`",
                self.pos,
                &self.input[self.pos..]
            ));
        }
        Ok(call)
    }

    fn parse_call_or_bare(&mut self) -> Result<ConfigCall, String> {
        self.skip_ws();
        let name = self.parse_identifier()?;
        self.skip_ws();
        if !self.consume_char('(') {
            return Ok(ConfigCall {
                name,
                args: Vec::new(),
            });
        }

        let mut args = Vec::new();
        self.skip_ws();
        if self.consume_char(')') {
            return Ok(ConfigCall { name, args });
        }

        loop {
            args.push(self.parse_arg()?);
            self.skip_ws();
            if self.consume_char(',') {
                self.skip_ws();
                if self.consume_char(')') {
                    break;
                }
                continue;
            }
            self.expect_char(')')?;
            break;
        }

        Ok(ConfigCall { name, args })
    }

    fn parse_arg(&mut self) -> Result<ConfigArg, String> {
        self.skip_ws();
        let checkpoint = self.pos;
        if let Ok(key) = self.parse_identifier() {
            self.skip_ws();
            if self.consume_char('=') {
                let value = self.parse_value()?;
                return Ok(ConfigArg {
                    key: Some(key),
                    value,
                });
            }
        }

        self.pos = checkpoint;
        let value = self.parse_value()?;
        Ok(ConfigArg { key: None, value })
    }

    fn parse_value(&mut self) -> Result<ConfigValue, String> {
        self.skip_ws();
        let checkpoint = self.pos;
        if let Ok(call) = self.parse_call_or_bare() {
            if !call.args.is_empty() || self.peek_non_ws() == Some('(') {
                return Ok(ConfigValue::Call(call));
            }
            self.skip_ws();
            if matches!(self.peek_char(), Some(',') | Some(')') | None) {
                return Ok(ConfigValue::Atom(call.name));
            }
        }
        self.pos = checkpoint;
        Ok(ConfigValue::Atom(self.parse_scalar()?))
    }

    fn parse_identifier(&mut self) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                self.pos += ch.len_utf8();
            } else {
                break;
            }
        }
        if self.pos == start {
            Err(format!(
                "Invalid --search config: expected identifier at byte {}",
                start
            ))
        } else {
            Ok(self.input[start..self.pos].to_ascii_lowercase())
        }
    }

    fn parse_scalar(&mut self) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        while let Some(ch) = self.peek_char() {
            if ch == ',' || ch == ')' {
                break;
            }
            self.pos += ch.len_utf8();
        }
        let value = self.input[start..self.pos].trim();
        if value.is_empty() {
            Err(format!(
                "Invalid --search config: expected value at byte {start}"
            ))
        } else {
            Ok(value.to_ascii_lowercase())
        }
    }

    fn skip_ws(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() {
                self.pos += ch.len_utf8();
            } else {
                break;
            }
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn peek_non_ws(&self) -> Option<char> {
        self.input[self.pos..]
            .chars()
            .find(|ch| !ch.is_whitespace())
    }

    fn consume_char(&mut self, expected: char) -> bool {
        self.skip_ws();
        if self.peek_char() == Some(expected) {
            self.pos += expected.len_utf8();
            true
        } else {
            false
        }
    }

    fn expect_char(&mut self, expected: char) -> Result<(), String> {
        if self.consume_char(expected) {
            Ok(())
        } else {
            Err(format!(
                "Invalid --search config: expected `{expected}` at byte {}",
                self.pos
            ))
        }
    }
}

pub struct Field<T> {
    name: &'static str,
    apply: fn(&mut T, &ConfigValue) -> Result<(), String>,
    format: fn(&T) -> String,
}

impl<T> Field<T> {
    pub fn new(
        name: &'static str,
        apply: fn(&mut T, &ConfigValue) -> Result<(), String>,
        format: fn(&T) -> String,
    ) -> Self {
        Self {
            name,
            apply,
            format,
        }
    }
}

fn apply_config_fields<T: Default>(call: &ConfigCall, fields: &[Field<T>]) -> Result<T, String> {
    let mut config = T::default();
    let mut seen = std::collections::BTreeSet::new();
    let mut next_positional = 0;

    for arg in &call.args {
        let field = if let Some(key) = &arg.key {
            if !seen.insert(key.clone()) {
                return Err(format!("duplicate option `{key}`"));
            }
            fields
                .iter()
                .find(|field| field.name == key)
                .ok_or_else(|| format!("unknown option `{key}` for `{}`", call.name))?
        } else {
            let field = fields
                .get(next_positional)
                .ok_or_else(|| format!("too many positional arguments for `{}`", call.name))?;
            next_positional += 1;
            if !seen.insert(field.name.to_string()) {
                return Err(format!("duplicate option `{}`", field.name));
            }
            field
        };
        (field.apply)(&mut config, &arg.value)?;
    }

    Ok(config)
}

pub trait FromConfig: Default + PartialEq + Sized {
    fn config_fields() -> Vec<Field<Self>>;

    fn from_config(call: &ConfigCall) -> Result<Self, String> {
        apply_config_fields(call, &Self::config_fields())
    }

    fn format_config_args(&self) -> String {
        Self::config_fields()
            .iter()
            .map(|field| format!("{}={}", field.name, (field.format)(self)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn format_config_call(&self, name: &str) -> String {
        if *self == Self::default() {
            format!("{name}()")
        } else {
            format!("{name}({})", self.format_config_args())
        }
    }
}

fn atom(value: &ConfigValue) -> Result<&str, String> {
    value.as_atom()
}

fn set_usize<T>(
    slot: fn(&mut T) -> &mut usize,
    config: &mut T,
    value: &ConfigValue,
) -> Result<(), String> {
    *slot(config) = parse_usize(atom(value)?)?;
    Ok(())
}

fn set_u64<T>(
    slot: fn(&mut T) -> &mut u64,
    config: &mut T,
    value: &ConfigValue,
) -> Result<(), String> {
    *slot(config) = parse_u64(atom(value)?)?;
    Ok(())
}

fn set_bool<T>(
    slot: fn(&mut T) -> &mut bool,
    config: &mut T,
    value: &ConfigValue,
) -> Result<(), String> {
    *slot(config) = parse_bool(atom(value)?)?;
    Ok(())
}

fn set_f64<T>(
    slot: fn(&mut T) -> &mut f64,
    config: &mut T,
    value: &ConfigValue,
) -> Result<(), String> {
    *slot(config) = parse_f64_or_infinity(atom(value)?)?;
    Ok(())
}

fn parse_bool(value: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("expected boolean, got `{value}`")),
    }
}

fn parse_usize(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("expected non-negative integer, got `{value}`"))
}

fn parse_u64(value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("expected non-negative integer, got `{value}`"))
}

fn parse_optional_seed(value: &str) -> Result<Option<u64>, String> {
    if value.eq_ignore_ascii_case("none") {
        Ok(None)
    } else {
        parse_u64(value).map(Some)
    }
}

fn parse_f64_or_infinity(value: &str) -> Result<f64, String> {
    if value.eq_ignore_ascii_case("infinity") {
        Ok(f64::INFINITY)
    } else {
        value
            .parse::<f64>()
            .map_err(|_| format!("expected float or infinity, got `{value}`"))
    }
}

fn format_f64_or_infinity(value: f64) -> String {
    if value.is_infinite() {
        "infinity".to_string()
    } else {
        value.to_string()
    }
}

fn format_optional_seed(seed: Option<u64>) -> String {
    seed.map_or_else(|| "none".to_string(), |seed| seed.to_string())
}

fn parse_greedy_variable_order_type(value: &str) -> Result<GreedyVariableOrderType, String> {
    match value {
        "cg_goal_level" => Ok(GreedyVariableOrderType::CgGoalLevel),
        "cg_goal_random" => Ok(GreedyVariableOrderType::CgGoalRandom),
        "goal_cg_level" => Ok(GreedyVariableOrderType::GoalCgLevel),
        _ => Err(format!("invalid GreedyVariableOrderType `{value}`")),
    }
}

fn parse_pdb_internal_heuristic(value: &str) -> Result<PdbInternalHeuristic, String> {
    match value {
        "zero" => Ok(PdbInternalHeuristic::Zero),
        "blind" => Ok(PdbInternalHeuristic::Blind),
        "lmcut" => Ok(PdbInternalHeuristic::Lmcut),
        _ => Err(format!("invalid PdbInternalHeuristic `{value}`")),
    }
}

fn parse_saturator(value: &str) -> Result<Saturator, String> {
    match value {
        "all" => Ok(Saturator::All),
        "perim" => Ok(Saturator::Perim),
        "perimstar" => Ok(Saturator::Perimstar),
        _ => Err(format!("invalid Saturator `{value}`")),
    }
}

fn parse_variable_subset(value: &str) -> Result<VariableSubset, String> {
    match value {
        "goals" => Ok(VariableSubset::Goals),
        "non_goals" => Ok(VariableSubset::NonGoals),
        "all" => Ok(VariableSubset::All),
        _ => Err(format!("invalid VariableSubset `{value}`")),
    }
}

fn parse_init_split_quantity(value: &str) -> Result<InitSplitQuantity, String> {
    match value {
        "none" => Ok(InitSplitQuantity::None),
        "single" => Ok(InitSplitQuantity::Single),
        "all" => Ok(InitSplitQuantity::All),
        _ => Err(format!("invalid InitSplitQuantity `{value}`")),
    }
}

fn parse_flaw_kind(value: &str) -> Result<FlawKind, String> {
    match value {
        "progression" => Ok(FlawKind::Progression),
        "regression" => Ok(FlawKind::Regression),
        "sequence_progression" => Ok(FlawKind::SequenceProgression),
        "sequence_regression" => Ok(FlawKind::SequenceRegression),
        "sequence_bidirectional" => Ok(FlawKind::SequenceBidirectional),
        _ => Err(format!("invalid FlawKind `{value}`")),
    }
}

fn parse_flaw_treatment(value: &str) -> Result<FlawTreatmentVariants, String> {
    match value {
        "random_single_atom" => Ok(FlawTreatmentVariants::RandomSingleAtom),
        "one_split_per_atom" => Ok(FlawTreatmentVariants::OneSplitPerAtom),
        "one_split_per_variable" => Ok(FlawTreatmentVariants::OneSplitPerVariable),
        "max_refined_single_atom" => Ok(FlawTreatmentVariants::MaxRefinedSingleAtom),
        "max_refined_preferring_prop" => Ok(FlawTreatmentVariants::MaxRefinedPreferringProp),
        "closest_to_goal" => Ok(FlawTreatmentVariants::ClosestToGoal),
        "balance_max_refined_and_closest_to_goal" => {
            Ok(FlawTreatmentVariants::BalanceMaxRefinedAndClosestToGoal)
        }
        "balance_max_refined_preferring_prop_and_closest_to_goal" => {
            Ok(FlawTreatmentVariants::BalanceMaxRefinedPreferringPropAndClosestToGoal)
        }
        _ => Err(format!("invalid FlawTreatment `{value}`")),
    }
}

fn parse_init_split_method(value: &str) -> Result<InitSplitMethod, String> {
    match value {
        "goal_value" => Ok(InitSplitMethod::GoalValue),
        "goal_value_or_random_if_non_goal" => Ok(InitSplitMethod::GoalValueOrRandomIfNonGoal),
        "init_value" => Ok(InitSplitMethod::InitValue),
        "random_value" => Ok(InitSplitMethod::RandomValue),
        "random_partition" => Ok(InitSplitMethod::RandomPartition),
        "random_binary_partition_separating_init_goal" => {
            Ok(InitSplitMethod::RandomBinaryPartitionSeparatingInitGoal)
        }
        "identity" => Ok(InitSplitMethod::Identity),
        _ => Err(format!("invalid InitSplitMethod `{value}`")),
    }
}

fn parse_numeric_split_strategy(value: &str) -> Result<NumericSplitStrategy, String> {
    match value {
        "standard" => Ok(NumericSplitStrategy::Standard),
        "exclusion" => Ok(NumericSplitStrategy::Exclusion),
        _ => Err(format!("invalid NumericSplitStrategy `{value}`")),
    }
}

fn parse_portfolio_strategy(value: &str) -> Result<PortfolioStrategy, String> {
    match value {
        "standard" => Ok(PortfolioStrategy::Standard),
        "view_diverse" => Ok(PortfolioStrategy::ViewDiverse),
        "region_landmarks" => Ok(PortfolioStrategy::RegionLandmarks),
        _ => Err(format!("invalid PortfolioStrategy `{value}`")),
    }
}

macro_rules! field_usize {
    ($name:literal, $ty:ty, $field:ident) => {
        Field {
            name: $name,
            apply: |config: &mut $ty, value| set_usize(|c| &mut c.$field, config, value),
            format: |config: &$ty| config.$field.to_string(),
        }
    };
}
macro_rules! field_u64 {
    ($name:literal, $ty:ty, $field:ident) => {
        Field {
            name: $name,
            apply: |config: &mut $ty, value| set_u64(|c| &mut c.$field, config, value),
            format: |config: &$ty| config.$field.to_string(),
        }
    };
}
macro_rules! field_bool {
    ($name:literal, $ty:ty, $field:ident) => {
        Field {
            name: $name,
            apply: |config: &mut $ty, value| set_bool(|c| &mut c.$field, config, value),
            format: |config: &$ty| config.$field.to_string(),
        }
    };
}
macro_rules! field_f64 {
    ($name:literal, $ty:ty, $field:ident) => {
        Field {
            name: $name,
            apply: |config: &mut $ty, value| set_f64(|c| &mut c.$field, config, value),
            format: |config: &$ty| format_f64_or_infinity(config.$field),
        }
    };
}

fn domain_abstraction_fields() -> Vec<Field<DomainAbstractionConfig>> {
    vec![
        field_usize!(
            "max_abstraction_size",
            DomainAbstractionConfig,
            max_abstraction_size
        ),
        field_usize!("max_iterations", DomainAbstractionConfig, max_iterations),
        field_bool!(
            "use_wildcard_plans",
            DomainAbstractionConfig,
            use_wildcard_plans
        ),
        field_bool!("combine_labels", DomainAbstractionConfig, combine_labels),
        field_bool!(
            "transform_linear_task",
            DomainAbstractionConfig,
            transform_linear_task
        ),
        Field {
            name: "random_seed",
            apply: |config, value| {
                config.random_seed = parse_optional_seed(atom(value)?)?;
                Ok(())
            },
            format: |config| format_optional_seed(config.random_seed),
        },
        Field {
            name: "flaw_treatment",
            apply: |config, value| {
                config.flaw_treatment = parse_flaw_treatment(atom(value)?)?;
                Ok(())
            },
            format: |config| config.flaw_treatment.to_string(),
        },
        Field {
            name: "flaw_kind",
            apply: |config, value| {
                config.flaw_kind = parse_flaw_kind(atom(value)?)?;
                Ok(())
            },
            format: |config| config.flaw_kind.to_string(),
        },
        Field {
            name: "init_split_method",
            apply: |config, value| {
                config.init_split_method = parse_init_split_method(atom(value)?)?;
                Ok(())
            },
            format: |config| config.init_split_method.to_string(),
        },
    ]
}

fn multi_domain_abstractions_fields()
-> Vec<Field<DomainAbstractionCollectionGeneratorMultipleCegarConfig>> {
    vec![
        field_usize!(
            "max_abstraction_size",
            DomainAbstractionCollectionGeneratorMultipleCegarConfig,
            max_abstraction_size
        ),
        field_usize!(
            "max_collection_size",
            DomainAbstractionCollectionGeneratorMultipleCegarConfig,
            max_collection_size
        ),
        field_f64!(
            "abstraction_generation_max_time",
            DomainAbstractionCollectionGeneratorMultipleCegarConfig,
            abstraction_generation_max_time
        ),
        field_f64!(
            "total_max_time",
            DomainAbstractionCollectionGeneratorMultipleCegarConfig,
            total_max_time
        ),
        field_f64!(
            "stagnation_limit",
            DomainAbstractionCollectionGeneratorMultipleCegarConfig,
            stagnation_limit
        ),
        field_f64!(
            "blacklist_trigger_percentage",
            DomainAbstractionCollectionGeneratorMultipleCegarConfig,
            blacklist_trigger_percentage
        ),
        field_bool!(
            "enable_blacklist_on_stagnation",
            DomainAbstractionCollectionGeneratorMultipleCegarConfig,
            enable_blacklist_on_stagnation
        ),
        Field {
            name: "blacklist_option",
            apply: |config, value| {
                config.blacklist_option = parse_variable_subset(atom(value)?)?;
                Ok(())
            },
            format: |config| config.blacklist_option.to_string(),
        },
        Field {
            name: "init_split_candidates",
            apply: |config, value| {
                config.init_split_candidates = parse_variable_subset(atom(value)?)?;
                Ok(())
            },
            format: |config| config.init_split_candidates.to_string(),
        },
        Field {
            name: "init_split_quantity",
            apply: |config, value| {
                config.init_split_quantity = parse_init_split_quantity(atom(value)?)?;
                Ok(())
            },
            format: |config| config.init_split_quantity.to_string(),
        },
        Field {
            name: "random_seed",
            apply: |config, value| {
                config.random_seed = parse_optional_seed(atom(value)?)?;
                Ok(())
            },
            format: |config| format_optional_seed(config.random_seed),
        },
        field_bool!(
            "debug",
            DomainAbstractionCollectionGeneratorMultipleCegarConfig,
            debug
        ),
        field_bool!(
            "use_wildcard_plans",
            DomainAbstractionCollectionGeneratorMultipleCegarConfig,
            use_wildcard_plans
        ),
        field_bool!(
            "combine_labels",
            DomainAbstractionCollectionGeneratorMultipleCegarConfig,
            combine_labels
        ),
        field_bool!(
            "transform_linear_task",
            DomainAbstractionCollectionGeneratorMultipleCegarConfig,
            transform_linear_task
        ),
        Field {
            name: "flaw_treatment",
            apply: |config, value| {
                config.flaw_treatment = parse_flaw_treatment(atom(value)?)?;
                Ok(())
            },
            format: |config| config.flaw_treatment.to_string(),
        },
        Field {
            name: "init_split_method",
            apply: |config, value| {
                config.init_split_method = parse_init_split_method(atom(value)?)?;
                Ok(())
            },
            format: |config| config.init_split_method.to_string(),
        },
        Field {
            name: "numeric_split_strategy",
            apply: |config, value| {
                config.numeric_split_strategy = parse_numeric_split_strategy(atom(value)?)?;
                Ok(())
            },
            format: |config| config.numeric_split_strategy.to_string(),
        },
        Field {
            name: "portfolio_strategy",
            apply: |config, value| {
                config.portfolio_strategy = parse_portfolio_strategy(atom(value)?)?;
                Ok(())
            },
            format: |config| config.portfolio_strategy.to_string(),
        },
    ]
}

fn scp_online_fields() -> Vec<Field<ScpOnlineConfig>> {
    vec![
        field_f64!("max_time", ScpOnlineConfig, max_time),
        field_usize!("max_size", ScpOnlineConfig, max_size),
        field_usize!("interval", ScpOnlineConfig, interval),
        field_bool!("use_numeric_pdbs", ScpOnlineConfig, use_numeric_pdbs),
        field_bool!(
            "use_abstract_operator_cost_partitioning",
            ScpOnlineConfig,
            use_abstract_operator_cost_partitioning
        ),
        Field {
            name: "saturator",
            apply: |config, value| {
                config.saturator = parse_saturator(atom(value)?)?;
                Ok(())
            },
            format: |config| config.saturator.to_string(),
        },
        field_usize!("max_pdb_states", ScpOnlineConfig, max_pdb_states),
        field_usize!("max_pattern_size", ScpOnlineConfig, max_pattern_size),
        field_bool!(
            "only_interesting_patterns",
            ScpOnlineConfig,
            only_interesting_patterns
        ),
        Field {
            name: "pdb_exploration_heuristic",
            apply: |config, value| {
                config.pdb_exploration_heuristic = parse_pdb_internal_heuristic(atom(value)?)?;
                Ok(())
            },
            format: |config| config.pdb_exploration_heuristic.to_string(),
        },
        Field {
            name: "pdb_frontier_heuristic",
            apply: |config, value| {
                config.pdb_frontier_heuristic = parse_pdb_internal_heuristic(atom(value)?)?;
                Ok(())
            },
            format: |config| config.pdb_frontier_heuristic.to_string(),
        },
        Field {
            name: "pdb_failed_lookup_heuristic",
            apply: |config, value| {
                config.pdb_failed_lookup_heuristic = parse_pdb_internal_heuristic(atom(value)?)?;
                Ok(())
            },
            format: |config| config.pdb_failed_lookup_heuristic.to_string(),
        },
        Field {
            name: "max_abstraction_size",
            apply: |config, value| {
                config.collection_config.max_abstraction_size = parse_usize(atom(value)?)?;
                Ok(())
            },
            format: |config| config.collection_config.max_abstraction_size.to_string(),
        },
        Field {
            name: "max_collection_size",
            apply: |config, value| {
                config.collection_config.max_collection_size = parse_usize(atom(value)?)?;
                Ok(())
            },
            format: |config| config.collection_config.max_collection_size.to_string(),
        },
        Field {
            name: "abstraction_generation_max_time",
            apply: |config, value| {
                config.collection_config.abstraction_generation_max_time =
                    parse_f64_or_infinity(atom(value)?)?;
                Ok(())
            },
            format: |config| {
                format_f64_or_infinity(config.collection_config.abstraction_generation_max_time)
            },
        },
        Field {
            name: "total_max_time",
            apply: |config, value| {
                config.collection_config.total_max_time = parse_f64_or_infinity(atom(value)?)?;
                Ok(())
            },
            format: |config| format_f64_or_infinity(config.collection_config.total_max_time),
        },
        Field {
            name: "stagnation_limit",
            apply: |config, value| {
                config.collection_config.stagnation_limit = parse_f64_or_infinity(atom(value)?)?;
                Ok(())
            },
            format: |config| format_f64_or_infinity(config.collection_config.stagnation_limit),
        },
        Field {
            name: "blacklist_trigger_percentage",
            apply: |config, value| {
                config.collection_config.blacklist_trigger_percentage =
                    parse_f64_or_infinity(atom(value)?)?;
                Ok(())
            },
            format: |config| {
                format_f64_or_infinity(config.collection_config.blacklist_trigger_percentage)
            },
        },
        Field {
            name: "enable_blacklist_on_stagnation",
            apply: |config, value| {
                config.collection_config.enable_blacklist_on_stagnation = parse_bool(atom(value)?)?;
                Ok(())
            },
            format: |config| {
                config
                    .collection_config
                    .enable_blacklist_on_stagnation
                    .to_string()
            },
        },
        Field {
            name: "blacklist_option",
            apply: |config, value| {
                config.collection_config.blacklist_option = parse_variable_subset(atom(value)?)?;
                Ok(())
            },
            format: |config| config.collection_config.blacklist_option.to_string(),
        },
        Field {
            name: "init_split_candidates",
            apply: |config, value| {
                config.collection_config.init_split_candidates =
                    parse_variable_subset(atom(value)?)?;
                Ok(())
            },
            format: |config| config.collection_config.init_split_candidates.to_string(),
        },
        Field {
            name: "init_split_quantity",
            apply: |config, value| {
                config.collection_config.init_split_quantity =
                    parse_init_split_quantity(atom(value)?)?;
                Ok(())
            },
            format: |config| config.collection_config.init_split_quantity.to_string(),
        },
        Field {
            name: "random_seed",
            apply: |config, value| {
                let random_seed = parse_optional_seed(atom(value)?)?;
                config.collection_config.random_seed = random_seed;
                config.random_seed = random_seed;
                Ok(())
            },
            format: |config| format_optional_seed(config.collection_config.random_seed),
        },
        Field {
            name: "debug",
            apply: |config, value| {
                config.collection_config.debug = parse_bool(atom(value)?)?;
                Ok(())
            },
            format: |config| config.collection_config.debug.to_string(),
        },
        Field {
            name: "use_wildcard_plans",
            apply: |config, value| {
                config.collection_config.use_wildcard_plans = parse_bool(atom(value)?)?;
                Ok(())
            },
            format: |config| config.collection_config.use_wildcard_plans.to_string(),
        },
        Field {
            name: "combine_labels",
            apply: |config, value| {
                let combine_labels = parse_bool(atom(value)?)?;
                config.combine_labels = combine_labels;
                config.collection_config.combine_labels = combine_labels;
                Ok(())
            },
            format: |config| config.combine_labels.to_string(),
        },
        Field {
            name: "flaw_treatment",
            apply: |config, value| {
                config.collection_config.flaw_treatment = parse_flaw_treatment(atom(value)?)?;
                Ok(())
            },
            format: |config| config.collection_config.flaw_treatment.to_string(),
        },
        Field {
            name: "flaw_kind",
            apply: |config, value| {
                config.collection_config.flaw_kind = parse_flaw_kind(atom(value)?)?;
                Ok(())
            },
            format: |config| config.collection_config.flaw_kind.to_string(),
        },
        Field {
            name: "init_split_method",
            apply: |config, value| {
                config.collection_config.init_split_method = parse_init_split_method(atom(value)?)?;
                Ok(())
            },
            format: |config| config.collection_config.init_split_method.to_string(),
        },
        Field {
            name: "numeric_split_strategy",
            apply: |config, value| {
                config.collection_config.numeric_split_strategy =
                    parse_numeric_split_strategy(atom(value)?)?;
                Ok(())
            },
            format: |config| config.collection_config.numeric_split_strategy.to_string(),
        },
        Field {
            name: "transform_linear_task",
            apply: |config, value| {
                config.collection_config.transform_linear_task = parse_bool(atom(value)?)?;
                Ok(())
            },
            format: |config| config.collection_config.transform_linear_task.to_string(),
        },
        Field {
            name: "portfolio_strategy",
            apply: |config, value| {
                config.collection_config.portfolio_strategy =
                    parse_portfolio_strategy(atom(value)?)?;
                Ok(())
            },
            format: |config| config.collection_config.portfolio_strategy.to_string(),
        },
    ]
}

fn greedy_numeric_pdb_fields() -> Vec<Field<GreedyPatternGeneratorConfig>> {
    vec![
        field_usize!(
            "max_pdb_states",
            GreedyPatternGeneratorConfig,
            max_pdb_states
        ),
        field_bool!("numeric_first", GreedyPatternGeneratorConfig, numeric_first),
        field_u64!("random_seed", GreedyPatternGeneratorConfig, random_seed),
        Field {
            name: "variable_order_type",
            apply: |config, value| {
                config.variable_order_type = parse_greedy_variable_order_type(atom(value)?)?;
                Ok(())
            },
            format: |config| config.variable_order_type.to_string(),
        },
        Field {
            name: "exploration_heuristic",
            apply: |config, value| {
                config.exploration_heuristic = parse_pdb_internal_heuristic(atom(value)?)?;
                Ok(())
            },
            format: |config| config.exploration_heuristic.to_string(),
        },
        Field {
            name: "frontier_heuristic",
            apply: |config, value| {
                config.frontier_heuristic = parse_pdb_internal_heuristic(atom(value)?)?;
                Ok(())
            },
            format: |config| config.frontier_heuristic.to_string(),
        },
        Field {
            name: "failed_lookup_heuristic",
            apply: |config, value| {
                config.failed_lookup_heuristic = parse_pdb_internal_heuristic(atom(value)?)?;
                Ok(())
            },
            format: |config| config.failed_lookup_heuristic.to_string(),
        },
    ]
}

fn canonical_numeric_pdb_fields() -> Vec<Field<CanonicalNumericPdbConfig>> {
    vec![
        field_usize!("max_pdb_states", CanonicalNumericPdbConfig, max_pdb_states),
        field_usize!(
            "max_pattern_size",
            CanonicalNumericPdbConfig,
            max_pattern_size
        ),
        field_bool!(
            "only_interesting_patterns",
            CanonicalNumericPdbConfig,
            only_interesting_patterns
        ),
        Field {
            name: "exploration_heuristic",
            apply: |config, value| {
                config.exploration_heuristic = parse_pdb_internal_heuristic(atom(value)?)?;
                Ok(())
            },
            format: |config| config.exploration_heuristic.to_string(),
        },
        Field {
            name: "frontier_heuristic",
            apply: |config, value| {
                config.frontier_heuristic = parse_pdb_internal_heuristic(atom(value)?)?;
                Ok(())
            },
            format: |config| config.frontier_heuristic.to_string(),
        },
        Field {
            name: "failed_lookup_heuristic",
            apply: |config, value| {
                config.failed_lookup_heuristic = parse_pdb_internal_heuristic(atom(value)?)?;
                Ok(())
            },
            format: |config| config.failed_lookup_heuristic.to_string(),
        },
    ]
}

fn lmcutnumeric_fields() -> Vec<Field<LmCutNumericConfig>> {
    vec![
        field_bool!(
            "ceiling_less_than_one",
            LmCutNumericConfig,
            ceiling_less_than_one
        ),
        field_bool!("ignore_numeric", LmCutNumericConfig, ignore_numeric),
        field_bool!("random_pcf", LmCutNumericConfig, random_pcf),
        field_bool!("irmax", LmCutNumericConfig, irmax),
        field_bool!("disable_ma", LmCutNumericConfig, disable_ma),
        field_bool!(
            "use_second_order_simple",
            LmCutNumericConfig,
            use_second_order_simple
        ),
        field_bool!(
            "use_constant_assignment",
            LmCutNumericConfig,
            use_constant_assignment
        ),
        field_usize!("bound_iterations", LmCutNumericConfig, bound_iterations),
        field_f64!("precision", LmCutNumericConfig, precision),
        field_f64!("epsilon", LmCutNumericConfig, epsilon),
    ]
}

impl FromConfig for AStarConfig {
    fn config_fields() -> Vec<Field<Self>> {
        vec![Field {
            name: "heuristic",
            apply: |config, value| {
                config.heuristic = build_heuristic(&call_from_value(value)?)?;
                Ok(())
            },
            format: |config| config.heuristic.to_string(),
        }]
    }
}

impl FromConfig for DomainAbstractionConfig {
    fn config_fields() -> Vec<Field<Self>> {
        domain_abstraction_fields()
    }
}

impl FromConfig for DomainAbstractionCollectionGeneratorMultipleCegarConfig {
    fn config_fields() -> Vec<Field<Self>> {
        multi_domain_abstractions_fields()
    }
}

impl FromConfig for ScpOnlineConfig {
    fn config_fields() -> Vec<Field<Self>> {
        scp_online_fields()
    }
}

impl FromConfig for GreedyPatternGeneratorConfig {
    fn config_fields() -> Vec<Field<Self>> {
        greedy_numeric_pdb_fields()
    }
}

impl FromConfig for CanonicalNumericPdbConfig {
    fn config_fields() -> Vec<Field<Self>> {
        canonical_numeric_pdb_fields()
    }
}

impl FromConfig for LmCutNumericConfig {
    fn config_fields() -> Vec<Field<Self>> {
        lmcutnumeric_fields()
    }
}

struct HeuristicPlugin {
    name: &'static str,
    build: fn(&ConfigCall) -> Result<HeuristicSpec, String>,
}

struct SearchPlugin {
    name: &'static str,
    build: fn(&ConfigCall) -> Result<SearchSpec, String>,
}

fn heuristic_registry() -> Vec<HeuristicPlugin> {
    vec![
        HeuristicPlugin {
            name: "blind",
            build: |call| {
                ensure_no_args(call)?;
                Ok(HeuristicSpec::Blind)
            },
        },
        HeuristicPlugin {
            name: "domain_abstraction",
            build: |call| {
                Ok(HeuristicSpec::DomainAbstraction(
                    DomainAbstractionConfig::from_config(call)?,
                ))
            },
        },
        HeuristicPlugin {
            name: "canonical_domain_abstractions",
            build: |call| {
                Ok(HeuristicSpec::CanonicalDomainAbstractions(
                    DomainAbstractionCollectionGeneratorMultipleCegarConfig::from_config(call)?,
                ))
            },
        },
        HeuristicPlugin {
            name: "multi_domain_abstractions",
            build: |call| {
                Ok(HeuristicSpec::MultiDomainAbstractions(
                    DomainAbstractionCollectionGeneratorMultipleCegarConfig::from_config(call)?,
                ))
            },
        },
        HeuristicPlugin {
            name: "scp_online",
            build: |call| {
                Ok(HeuristicSpec::ScpOnline(ScpOnlineConfig::from_config(
                    call,
                )?))
            },
        },
        HeuristicPlugin {
            name: "greedy_numeric_pdb",
            build: |call| {
                Ok(HeuristicSpec::GreedyNumericPdb(
                    GreedyPatternGeneratorConfig::from_config(call)?,
                ))
            },
        },
        HeuristicPlugin {
            name: "canonical_numeric_pdb",
            build: |call| {
                Ok(HeuristicSpec::CanonicalNumericPdb(
                    CanonicalNumericPdbConfig::from_config(call)?,
                ))
            },
        },
        HeuristicPlugin {
            name: "lmcutnumeric",
            build: |call| {
                Ok(HeuristicSpec::Lmcutnumeric(
                    LmCutNumericConfig::from_config(call)?,
                ))
            },
        },
    ]
}

fn search_registry() -> Vec<SearchPlugin> {
    vec![
        SearchPlugin {
            name: "astar",
            build: |call| Ok(SearchSpec::Astar(AStarConfig::from_config(call)?.heuristic)),
        },
        SearchPlugin {
            name: "da_debug",
            build: |call| {
                ensure_no_args(call)?;
                Ok(SearchSpec::DaDebug)
            },
        },
        SearchPlugin {
            name: "astar_da_debug",
            build: |call| {
                ensure_no_args(call)?;
                Ok(SearchSpec::AstarDaDebug)
            },
        },
    ]
}

fn ensure_no_args(call: &ConfigCall) -> Result<(), String> {
    if call.args.is_empty() {
        Ok(())
    } else {
        Err(format!("`{}` does not accept arguments", call.name))
    }
}

fn build_heuristic(call: &ConfigCall) -> Result<HeuristicSpec, String> {
    heuristic_registry()
        .into_iter()
        .find(|plugin| plugin.name == call.name)
        .ok_or_else(|| format!("unknown heuristic `{}`", call.name))
        .and_then(|plugin| (plugin.build)(call))
}

fn build_search_spec(call: &ConfigCall) -> Result<SearchSpec, String> {
    if call.name == "search" {
        if call.args.len() != 1 {
            return Err("`search(...)` expects exactly one search engine".to_string());
        }
        let nested = call_from_value(&call.args[0].value)?;
        return build_search_spec(&nested);
    }

    search_registry()
        .into_iter()
        .find(|plugin| plugin.name == call.name)
        .ok_or_else(|| format!("unknown search engine `{}`", call.name))
        .and_then(|plugin| (plugin.build)(call))
}

fn call_from_value(value: &ConfigValue) -> Result<ConfigCall, String> {
    match value {
        ConfigValue::Call(call) => Ok(call.clone()),
        ConfigValue::Atom(name) => Ok(ConfigCall {
            name: name.clone(),
            args: Vec::new(),
        }),
    }
}

#[cfg(test)]
mod tests;
