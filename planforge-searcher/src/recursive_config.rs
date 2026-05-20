use std::collections::HashMap;
use std::fmt;

use planforge_search::numeric::evaluation::domain_abstractions::cegar::{
    CegarConfig, FlawKind, FlawTreatmentVariants, SplitDirection,
};
use planforge_search::numeric::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegarConfig,
    InitSplitMethod, InitSplitQuantity, NumericSplitStrategy, PortfolioStrategy, VariableSubset,
};
use planforge_search::numeric::evaluation::domain_abstractions::saturated_cost_partitioning_online_heuristic::{
    FillScpConfig, OrderGenerator, Saturator, ScoringFunction, ScpOnlineConfig,
};
use planforge_search::numeric::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig;
use planforge_search::numeric::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbConfig;
use planforge_search::numeric::evaluation::pattern_databases::pattern_database::PdbInternalHeuristic;
use planforge_search::numeric::evaluation::pattern_databases::pattern_generator_greedy::GreedyPatternGeneratorConfig;
use planforge_search::numeric::evaluation::pattern_databases::variable_order_finder::GreedyVariableOrderType;

#[cfg(test)]
mod tests;

// =============================================================================
// HeuristicSpec + SearchSpec
// =============================================================================

/// A parsed heuristic configuration. The heuristic is identified by `name`;
/// its options are a raw HashMap, applied to a typed config struct at
/// construction time via the `apply_*_options` helpers below.
#[derive(Debug, Clone, PartialEq)]
pub struct HeuristicSpec {
    pub name: String,
    pub args: HashMap<String, ConfigValue>,
}

impl HeuristicSpec {
    pub fn new(name: impl Into<String>, args: HashMap<String, ConfigValue>) -> Self {
        Self {
            name: name.into(),
            args,
        }
    }

    pub fn blind() -> Self {
        Self {
            name: "blind".to_string(),
            args: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SearchSpec {
    Astar(HeuristicSpec),
    Gbfs(HeuristicSpec),
    /// A* with two admissible heuristics: a *fast* one for initial open-
    /// list ordering and a *slow* but possibly tighter one evaluated
    /// lazily when a state is about to be expanded.
    AstarFs(HeuristicSpec, HeuristicSpec),
    DaDebug,
    AstarDaDebug,
}

// =============================================================================
// Display — used to round-trip the spec back into `--search SPEC` form
// =============================================================================

impl fmt::Display for HeuristicSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.is_empty() {
            return write!(f, "{}()", self.name);
        }
        let mut pairs: Vec<_> = self.args.iter().collect();
        pairs.sort_by(|(a, _), (b, _)| a.cmp(b));
        write!(f, "{}(", self.name)?;
        for (i, (k, v)) in pairs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{k}={}", fmt_value(v))?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for SearchSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Astar(h) => write!(f, "astar({h})"),
            Self::Gbfs(h) => write!(f, "gbfs({h})"),
            Self::AstarFs(fast, slow) => write!(f, "astar_fs(fast={fast}, slow={slow})"),
            Self::DaDebug => write!(f, "da_debug()"),
            Self::AstarDaDebug => write!(f, "astar_da_debug()"),
        }
    }
}

fn fmt_value(v: &ConfigValue) -> String {
    match v {
        ConfigValue::Atom(s) => s.clone(),
        ConfigValue::Call(c) => fmt_call(c),
    }
}

fn fmt_call(c: &ConfigCall) -> String {
    if c.args.is_empty() {
        return format!("{}()", c.name);
    }
    let parts: Vec<String> = c
        .args
        .iter()
        .map(|a| match &a.key {
            Some(k) => format!("{k}={}", fmt_value(&a.value)),
            None => fmt_value(&a.value),
        })
        .collect();
    format!("{}({})", c.name, parts.join(", "))
}

// =============================================================================
// Parser entry
// =============================================================================

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

pub(crate) struct ConfigParser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> ConfigParser<'a> {
    pub(crate) fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    pub(crate) fn parse_all(mut self) -> Result<ConfigCall, String> {
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

// =============================================================================
// Search-engine dispatch
// =============================================================================

fn build_search_spec(call: &ConfigCall) -> Result<SearchSpec, String> {
    if call.name == "search" {
        if call.args.len() != 1 {
            return Err("`search(...)` expects exactly one search engine".to_string());
        }
        let nested = call_from_value(&call.args[0].value)?;
        return build_search_spec(&nested);
    }

    match call.name.as_str() {
        "astar" => Ok(SearchSpec::Astar(extract_heuristic_for_search(call)?)),
        "gbfs" => Ok(SearchSpec::Gbfs(extract_heuristic_for_search(call)?)),
        "astar_fs" => {
            let mut fast = None;
            let mut slow = None;
            for arg in &call.args {
                let key = arg.key.as_deref().ok_or_else(|| {
                    "`astar_fs(...)` expects named `fast=...` and `slow=...` arguments".to_string()
                })?;
                match key {
                    "fast" => fast = Some(heuristic_spec_from_value(&arg.value)?),
                    "slow" => slow = Some(heuristic_spec_from_value(&arg.value)?),
                    other => return Err(format!("unknown option `{other}` for `astar_fs`")),
                }
            }
            let fast = fast.ok_or_else(|| "`astar_fs(...)` requires `fast=...`".to_string())?;
            let slow = slow.ok_or_else(|| "`astar_fs(...)` requires `slow=...`".to_string())?;
            Ok(SearchSpec::AstarFs(fast, slow))
        }
        "da_debug" => {
            ensure_no_args(call)?;
            Ok(SearchSpec::DaDebug)
        }
        "astar_da_debug" => {
            ensure_no_args(call)?;
            Ok(SearchSpec::AstarDaDebug)
        }
        other => Err(format!("unknown search engine `{other}`")),
    }
}

fn extract_heuristic_for_search(call: &ConfigCall) -> Result<HeuristicSpec, String> {
    if call.args.is_empty() {
        return Ok(HeuristicSpec::blind());
    }
    if call.args.len() != 1 {
        return Err(format!(
            "`{}(...)` expects a single heuristic argument",
            call.name
        ));
    }
    let arg = &call.args[0];
    if let Some(key) = &arg.key {
        if key != "heuristic" {
            return Err(format!(
                "`{}(...)` expects `heuristic=...`, got `{key}=...`",
                call.name
            ));
        }
    }
    heuristic_spec_from_value(&arg.value)
}

fn heuristic_spec_from_value(value: &ConfigValue) -> Result<HeuristicSpec, String> {
    let call = match value {
        ConfigValue::Atom(name) => ConfigCall {
            name: name.clone(),
            args: Vec::new(),
        },
        ConfigValue::Call(c) => c.clone(),
    };
    let mut args = HashMap::new();
    for arg in call.args {
        let key = arg.key.ok_or_else(|| {
            format!(
                "positional arguments not supported in `{}(...)` — use named options",
                call.name
            )
        })?;
        if args.contains_key(&key) {
            return Err(format!("duplicate option `{key}` in `{}(...)`", call.name));
        }
        args.insert(key, arg.value);
    }
    Ok(HeuristicSpec {
        name: call.name,
        args,
    })
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

fn ensure_no_args(call: &ConfigCall) -> Result<(), String> {
    if call.args.is_empty() {
        Ok(())
    } else {
        Err(format!("`{}` does not accept arguments", call.name))
    }
}

// =============================================================================
// Per-config option appliers — call these at heuristic construction sites
// =============================================================================

fn atom(value: &ConfigValue) -> Result<&str, String> {
    value.as_atom()
}

/// Apply `domain_abstraction(...)` options directly onto a `CegarConfig`.
pub fn apply_da_options(
    cfg: &mut CegarConfig,
    opts: &HashMap<String, ConfigValue>,
) -> Result<(), String> {
    for (key, value) in opts {
        match key.as_str() {
            "max_abstraction_size" => cfg.max_abstraction_size = parse_usize(atom(value)?)?,
            "max_iterations" => cfg.max_iterations = parse_usize(atom(value)?)?,
            "use_wildcard_plans" => cfg.use_wildcard_plans = parse_bool(atom(value)?)?,
            "combine_labels" => cfg.combine_labels = parse_bool(atom(value)?)?,
            "transform_linear_task" => cfg.transform_linear_task = parse_bool(atom(value)?)?,
            "random_seed" => cfg.random_seed = parse_optional_seed(atom(value)?)?,
            "flaw_treatment" => cfg.flaw_treatment = parse_flaw_treatment(atom(value)?)?,
            "flaw_kind" => cfg.flaw_kind = parse_flaw_kind(atom(value)?)?,
            "init_split_method" => cfg.init_split_method = parse_init_split_method(atom(value)?)?,
            other => return Err(format!("unknown option `{other}` for `domain_abstraction`")),
        }
    }
    Ok(())
}

/// Apply collection-generator options (used by canonical/multi/posthoc).
pub fn apply_da_collection_options(
    cfg: &mut DomainAbstractionCollectionGeneratorMultipleCegarConfig,
    opts: &HashMap<String, ConfigValue>,
) -> Result<(), String> {
    for (key, value) in opts {
        apply_da_collection_key(cfg, key, value)?;
    }
    Ok(())
}

fn apply_da_collection_key(
    cfg: &mut DomainAbstractionCollectionGeneratorMultipleCegarConfig,
    key: &str,
    value: &ConfigValue,
) -> Result<(), String> {
    match key {
        "max_abstraction_size" => cfg.max_abstraction_size = parse_usize(atom(value)?)?,
        "max_collection_size" => cfg.max_collection_size = parse_usize(atom(value)?)?,
        "abstraction_generation_max_time" => {
            cfg.abstraction_generation_max_time = parse_f64_or_infinity(atom(value)?)?
        }
        "total_max_time" => cfg.total_max_time = parse_f64_or_infinity(atom(value)?)?,
        "stagnation_limit" => cfg.stagnation_limit = parse_f64_or_infinity(atom(value)?)?,
        "blacklist_trigger_percentage" => {
            cfg.blacklist_trigger_percentage = parse_f64_or_infinity(atom(value)?)?
        }
        "enable_blacklist_on_stagnation" => {
            cfg.enable_blacklist_on_stagnation = parse_bool(atom(value)?)?
        }
        "blacklist_option" => cfg.blacklist_option = parse_variable_subset(atom(value)?)?,
        "init_split_candidates" => {
            cfg.init_split_candidates = parse_variable_subset(atom(value)?)?
        }
        "init_split_quantity" => {
            cfg.init_split_quantity = parse_init_split_quantity(atom(value)?)?
        }
        "random_seed" => cfg.random_seed = parse_optional_seed(atom(value)?)?,
        "debug" => cfg.debug = parse_bool(atom(value)?)?,
        "use_wildcard_plans" => cfg.use_wildcard_plans = parse_bool(atom(value)?)?,
        "combine_labels" => cfg.combine_labels = parse_bool(atom(value)?)?,
        "transform_linear_task" => cfg.transform_linear_task = parse_bool(atom(value)?)?,
        "flaw_treatment" => cfg.flaw_treatment = parse_flaw_treatment(atom(value)?)?,
        "flaw_kind" => cfg.flaw_kind = parse_flaw_kind(atom(value)?)?,
        "init_split_method" => cfg.init_split_method = parse_init_split_method(atom(value)?)?,
        "numeric_split_strategy" => {
            cfg.numeric_split_strategy = parse_numeric_split_strategy(atom(value)?)?
        }
        "portfolio_strategy" => {
            cfg.portfolio_strategy = parse_portfolio_strategy(atom(value)?)?
        }
        "split_direction" => cfg.split_direction = parse_split_direction(atom(value)?)?,
        "max_stealable_width" => {
            cfg.finite_support.max_stealable_width = parse_f64_or_infinity(atom(value)?)?
        }
        other => return Err(format!("unknown option `{other}` for domain abstraction collection")),
    }
    Ok(())
}

/// Apply `scp_online(...)` options.
pub fn apply_scp_online_options(
    cfg: &mut ScpOnlineConfig,
    opts: &HashMap<String, ConfigValue>,
) -> Result<(), String> {
    for (key, value) in opts {
        match key.as_str() {
            "max_time" => cfg.max_time = parse_f64_or_infinity(atom(value)?)?,
            "table_construction_max_time" => {
                cfg.table_construction_max_time = parse_f64_or_infinity(atom(value)?)?
            }
            "max_size" => cfg.max_size = parse_usize(atom(value)?)?,
            "interval" => cfg.interval = parse_usize(atom(value)?)?,
            "use_numeric_pdbs" => cfg.use_numeric_pdbs = parse_bool(atom(value)?)?,
            "use_abstract_operator_cost_partitioning" => {
                cfg.use_abstract_operator_cost_partitioning = parse_bool(atom(value)?)?
            }
            "saturator" => cfg.saturator = parse_saturator(atom(value)?)?,
            "scoring_function" => cfg.scoring_function = parse_scoring_function(atom(value)?)?,
            "orders" => cfg.order_generator = parse_order_generator(atom(value)?)?,
            "order_optimization_max_time" => {
                cfg.order_optimization_max_time = parse_f64_or_infinity(atom(value)?)?
            }
            "max_pdb_states" => cfg.max_pdb_states = parse_usize(atom(value)?)?,
            "max_pattern_size" => cfg.max_pattern_size = parse_usize(atom(value)?)?,
            "only_interesting_patterns" => {
                cfg.only_interesting_patterns = parse_bool(atom(value)?)?
            }
            "pdb_exploration_heuristic" => {
                cfg.pdb_exploration_heuristic = parse_pdb_internal_heuristic(atom(value)?)?
            }
            "pdb_frontier_heuristic" => {
                cfg.pdb_frontier_heuristic = parse_pdb_internal_heuristic(atom(value)?)?
            }
            "pdb_failed_lookup_heuristic" => {
                cfg.pdb_failed_lookup_heuristic = parse_pdb_internal_heuristic(atom(value)?)?
            }
            "combine_labels" => {
                let v = parse_bool(atom(value)?)?;
                cfg.combine_labels = v;
                cfg.collection_config.combine_labels = v;
            }
            "random_seed" => {
                let v = parse_optional_seed(atom(value)?)?;
                cfg.random_seed = v;
                cfg.collection_config.random_seed = v;
            }
            other => apply_da_collection_key(&mut cfg.collection_config, other, value)?,
        }
    }
    Ok(())
}

/// Apply `fillSCP(...)` options. Caller is responsible for invoking
/// `cfg.force_full_goal_tasks()` after applying.
pub fn apply_fill_scp_options(
    cfg: &mut FillScpConfig,
    opts: &HashMap<String, ConfigValue>,
) -> Result<(), String> {
    for (key, value) in opts {
        match key.as_str() {
            "table_construction_max_time" => {
                cfg.table_construction_max_time = parse_f64_or_infinity(atom(value)?)?
            }
            "use_abstract_operator_cost_partitioning" => {
                cfg.use_abstract_operator_cost_partitioning = parse_bool(atom(value)?)?
            }
            "saturator" => cfg.saturator = parse_saturator(atom(value)?)?,
            "scoring_function" => cfg.scoring_function = parse_scoring_function(atom(value)?)?,
            "orders" => cfg.order_generator = parse_order_generator(atom(value)?)?,
            "order_optimization_max_time" => {
                cfg.order_optimization_max_time = parse_f64_or_infinity(atom(value)?)?
            }
            "combine_labels" => {
                let v = parse_bool(atom(value)?)?;
                cfg.combine_labels = v;
                cfg.collection_config.combine_labels = v;
            }
            "random_seed" => {
                let v = parse_optional_seed(atom(value)?)?;
                cfg.random_seed = v;
                cfg.collection_config.random_seed = v;
            }
            // lmcut subfields
            "ceiling_less_than_one" => {
                cfg.lmcut_config.ceiling_less_than_one = parse_bool(atom(value)?)?
            }
            "ignore_numeric" => cfg.lmcut_config.ignore_numeric = parse_bool(atom(value)?)?,
            "random_pcf" => cfg.lmcut_config.random_pcf = parse_bool(atom(value)?)?,
            "irmax" => cfg.lmcut_config.irmax = parse_bool(atom(value)?)?,
            "disable_ma" => cfg.lmcut_config.disable_ma = parse_bool(atom(value)?)?,
            "use_second_order_simple" => {
                cfg.lmcut_config.use_second_order_simple = parse_bool(atom(value)?)?
            }
            "use_constant_assignment" => {
                cfg.lmcut_config.use_constant_assignment = parse_bool(atom(value)?)?
            }
            "bound_iterations" => {
                cfg.lmcut_config.bound_iterations = parse_usize(atom(value)?)?
            }
            "precision" => cfg.lmcut_config.precision = parse_f64_or_infinity(atom(value)?)?,
            "epsilon" => cfg.lmcut_config.epsilon = parse_f64_or_infinity(atom(value)?)?,
            other => apply_da_collection_key(&mut cfg.collection_config, other, value)?,
        }
    }
    Ok(())
}

/// Apply `greedy_numeric_pdb(...)` options.
pub fn apply_greedy_pdb_options(
    cfg: &mut GreedyPatternGeneratorConfig,
    opts: &HashMap<String, ConfigValue>,
) -> Result<(), String> {
    for (key, value) in opts {
        match key.as_str() {
            "max_pdb_states" => cfg.max_pdb_states = parse_usize(atom(value)?)?,
            "numeric_first" => cfg.numeric_first = parse_bool(atom(value)?)?,
            "random_seed" => cfg.random_seed = parse_u64(atom(value)?)?,
            "variable_order_type" => {
                cfg.variable_order_type = parse_greedy_variable_order_type(atom(value)?)?
            }
            "exploration_heuristic" => {
                cfg.exploration_heuristic = parse_pdb_internal_heuristic(atom(value)?)?
            }
            "frontier_heuristic" => {
                cfg.frontier_heuristic = parse_pdb_internal_heuristic(atom(value)?)?
            }
            "failed_lookup_heuristic" => {
                cfg.failed_lookup_heuristic = parse_pdb_internal_heuristic(atom(value)?)?
            }
            other => return Err(format!("unknown option `{other}` for `greedy_numeric_pdb`")),
        }
    }
    Ok(())
}

/// Apply `canonical_numeric_pdb(...)` options.
pub fn apply_canonical_pdb_options(
    cfg: &mut CanonicalNumericPdbConfig,
    opts: &HashMap<String, ConfigValue>,
) -> Result<(), String> {
    for (key, value) in opts {
        match key.as_str() {
            "max_pdb_states" => cfg.max_pdb_states = parse_usize(atom(value)?)?,
            "max_pattern_size" => cfg.max_pattern_size = parse_usize(atom(value)?)?,
            "only_interesting_patterns" => {
                cfg.only_interesting_patterns = parse_bool(atom(value)?)?
            }
            "exploration_heuristic" => {
                cfg.exploration_heuristic = parse_pdb_internal_heuristic(atom(value)?)?
            }
            "frontier_heuristic" => {
                cfg.frontier_heuristic = parse_pdb_internal_heuristic(atom(value)?)?
            }
            "failed_lookup_heuristic" => {
                cfg.failed_lookup_heuristic = parse_pdb_internal_heuristic(atom(value)?)?
            }
            other => return Err(format!("unknown option `{other}` for `canonical_numeric_pdb`")),
        }
    }
    Ok(())
}

/// Apply `lmcutnumeric(...)` options.
pub fn apply_lmcut_options(
    cfg: &mut LmCutNumericConfig,
    opts: &HashMap<String, ConfigValue>,
) -> Result<(), String> {
    for (key, value) in opts {
        match key.as_str() {
            "ceiling_less_than_one" => cfg.ceiling_less_than_one = parse_bool(atom(value)?)?,
            "ignore_numeric" => cfg.ignore_numeric = parse_bool(atom(value)?)?,
            "random_pcf" => cfg.random_pcf = parse_bool(atom(value)?)?,
            "irmax" => cfg.irmax = parse_bool(atom(value)?)?,
            "disable_ma" => cfg.disable_ma = parse_bool(atom(value)?)?,
            "use_second_order_simple" => cfg.use_second_order_simple = parse_bool(atom(value)?)?,
            "use_constant_assignment" => cfg.use_constant_assignment = parse_bool(atom(value)?)?,
            "bound_iterations" => cfg.bound_iterations = parse_usize(atom(value)?)?,
            "precision" => cfg.precision = parse_f64_or_infinity(atom(value)?)?,
            "epsilon" => cfg.epsilon = parse_f64_or_infinity(atom(value)?)?,
            other => return Err(format!("unknown option `{other}` for `lmcutnumeric`")),
        }
    }
    Ok(())
}

// =============================================================================
// Scalar value parsers
// =============================================================================

pub(crate) fn parse_bool(value: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("expected boolean, got `{value}`")),
    }
}

pub(crate) fn parse_usize(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("expected non-negative integer, got `{value}`"))
}

pub(crate) fn parse_u64(value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("expected non-negative integer, got `{value}`"))
}

pub(crate) fn parse_optional_seed(value: &str) -> Result<Option<u64>, String> {
    if value.eq_ignore_ascii_case("none") {
        Ok(None)
    } else {
        parse_u64(value).map(Some)
    }
}

pub(crate) fn parse_f64_or_infinity(value: &str) -> Result<f64, String> {
    if value.eq_ignore_ascii_case("infinity") {
        Ok(f64::INFINITY)
    } else {
        value
            .parse::<f64>()
            .map_err(|_| format!("expected float or infinity, got `{value}`"))
    }
}

pub(crate) fn parse_greedy_variable_order_type(
    value: &str,
) -> Result<GreedyVariableOrderType, String> {
    match value {
        "cg_goal_level" => Ok(GreedyVariableOrderType::CgGoalLevel),
        "cg_goal_random" => Ok(GreedyVariableOrderType::CgGoalRandom),
        "goal_cg_level" => Ok(GreedyVariableOrderType::GoalCgLevel),
        _ => Err(format!("invalid GreedyVariableOrderType `{value}`")),
    }
}

pub(crate) fn parse_pdb_internal_heuristic(value: &str) -> Result<PdbInternalHeuristic, String> {
    match value {
        "zero" => Ok(PdbInternalHeuristic::Zero),
        "blind" => Ok(PdbInternalHeuristic::Blind),
        "lmcut" => Ok(PdbInternalHeuristic::Lmcut),
        _ => Err(format!("invalid PdbInternalHeuristic `{value}`")),
    }
}

pub(crate) fn parse_saturator(value: &str) -> Result<Saturator, String> {
    match value {
        "all" => Ok(Saturator::All),
        "perim" => Ok(Saturator::Perim),
        "perimstar" => Ok(Saturator::Perimstar),
        _ => Err(format!("invalid Saturator `{value}`")),
    }
}

pub(crate) fn parse_scoring_function(value: &str) -> Result<ScoringFunction, String> {
    match value {
        "max_heuristic" => Ok(ScoringFunction::MaxHeuristic),
        "min_stolen_costs" => Ok(ScoringFunction::MinStolenCosts),
        "max_heuristic_per_stolen_costs" => Ok(ScoringFunction::MaxHeuristicPerStolenCosts),
        _ => Err(format!("invalid ScoringFunction `{value}`")),
    }
}

pub(crate) fn parse_order_generator(value: &str) -> Result<OrderGenerator, String> {
    match value {
        "greedy_orders" | "greedy_orders()" => Ok(OrderGenerator::Greedy),
        "dynamic_greedy_orders" | "dynamic_greedy_orders()" => Ok(OrderGenerator::DynamicGreedy),
        "random_orders" | "random_orders()" => Ok(OrderGenerator::Random),
        _ => Err(format!("invalid OrderGenerator `{value}`")),
    }
}

pub(crate) fn parse_variable_subset(value: &str) -> Result<VariableSubset, String> {
    match value {
        "goals" => Ok(VariableSubset::Goals),
        "non_goals" => Ok(VariableSubset::NonGoals),
        "all" => Ok(VariableSubset::All),
        _ => Err(format!("invalid VariableSubset `{value}`")),
    }
}

pub(crate) fn parse_init_split_quantity(value: &str) -> Result<InitSplitQuantity, String> {
    match value {
        "none" => Ok(InitSplitQuantity::None),
        "single" => Ok(InitSplitQuantity::Single),
        "all" => Ok(InitSplitQuantity::All),
        _ => Err(format!("invalid InitSplitQuantity `{value}`")),
    }
}

pub(crate) fn parse_flaw_kind(value: &str) -> Result<FlawKind, String> {
    match value {
        "progression" => Ok(FlawKind::Progression),
        "regression" => Ok(FlawKind::Regression),
        "execute_entire_plan" => Ok(FlawKind::ExecuteEntirePlan),
        "sequence_progression" => Ok(FlawKind::SequenceProgression),
        "sequence_regression" => Ok(FlawKind::SequenceRegression),
        "sequence_bidirectional" => Ok(FlawKind::SequenceBidirectional),
        "target_centered" => Ok(FlawKind::TargetCentered),
        _ => Err(format!("invalid FlawKind `{value}`")),
    }
}

pub(crate) fn parse_flaw_treatment(value: &str) -> Result<FlawTreatmentVariants, String> {
    match value {
        "random_single_atom" => Ok(FlawTreatmentVariants::RandomSingleAtom),
        "one_split_per_atom" => Ok(FlawTreatmentVariants::OneSplitPerAtom),
        "one_split_per_variable" => Ok(FlawTreatmentVariants::OneSplitPerVariable),
        "max_refined_single_atom" => Ok(FlawTreatmentVariants::MaxRefinedSingleAtom),
        "min_growth_single_atom" => Ok(FlawTreatmentVariants::MinGrowthSingleAtom),
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

pub(crate) fn parse_init_split_method(value: &str) -> Result<InitSplitMethod, String> {
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

pub(crate) fn parse_numeric_split_strategy(value: &str) -> Result<NumericSplitStrategy, String> {
    match value {
        "standard" => Ok(NumericSplitStrategy::Standard),
        "exclusion" => Ok(NumericSplitStrategy::Exclusion),
        _ => Err(format!("invalid NumericSplitStrategy `{value}`")),
    }
}

pub(crate) fn parse_portfolio_strategy(value: &str) -> Result<PortfolioStrategy, String> {
    match value {
        "standard" => Ok(PortfolioStrategy::Standard),
        "complementary" => Ok(PortfolioStrategy::Complementary),
        _ => Err(format!("invalid PortfolioStrategy `{value}`")),
    }
}

pub(crate) fn parse_split_direction(value: &str) -> Result<Option<SplitDirection>, String> {
    match value {
        "default" => Ok(None),
        "forward" => Ok(Some(SplitDirection::Forward)),
        "forward_partition_deviation" => Ok(Some(SplitDirection::ForwardPartitionDeviation)),
        "backward" => Ok(Some(SplitDirection::Backward)),
        _ => Err(format!("invalid SplitDirection `{value}`")),
    }
}
