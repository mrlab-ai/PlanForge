use std::fmt;
use std::time::Duration;

// Re-export the parser AST nodes + the trait from `planforge-search`, where
// the typed configs live. The derive emits absolute paths into those types,
// so they have to live in one canonical location.
pub use planforge_search::config::{
    ApplyOptions, ConfigArg, ConfigCall, ConfigValue, FromOptionValue, atom, for_each_option,
};

use planforge_search::evaluation::domain_abstractions::cegar::CegarConfig;

#[cfg(test)]
mod tests;

// =============================================================================
// HeuristicSpec + SearchSpec
// =============================================================================

/// A parsed heuristic configuration. The heuristic is identified by `name`;
/// its options are an ordered list of `ConfigArg`s (each optionally keyed),
/// applied to a typed config struct at construction time via the
/// `apply_*_options` helpers below.
///
/// Storing args as `Vec<ConfigArg>` (not `HashMap`) lets each applier
/// resolve positional args against its own `ORDER` list — so both
/// `greedy_numeric_pdb(max_pdb_states=321)` and `greedy_numeric_pdb(321)`
/// work, and they can be mixed: `greedy_numeric_pdb(321, numeric_first=false)`.
#[derive(Debug, Clone, PartialEq)]
pub struct HeuristicSpec {
    pub name: String,
    pub args: Vec<ConfigArg>,
}

impl HeuristicSpec {
    pub fn new(name: impl Into<String>, args: Vec<ConfigArg>) -> Self {
        Self {
            name: name.into(),
            args,
        }
    }

    pub fn blind() -> Self {
        Self {
            name: "blind".to_string(),
            args: Vec::new(),
        }
    }

    pub fn contains_call(&self, name: &str) -> bool {
        self.name == name
            || self
                .args
                .iter()
                .any(|arg| config_value_contains_call(arg.value(), name))
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

impl SearchSpec {
    pub fn contains_call(&self, name: &str) -> bool {
        match self {
            Self::Astar(heuristic) | Self::Gbfs(heuristic) => heuristic.contains_call(name),
            Self::AstarFs(fast, slow) => fast.contains_call(name) || slow.contains_call(name),
            Self::DaDebug | Self::AstarDaDebug => false,
        }
    }
}

fn config_value_contains_call(value: &ConfigValue, name: &str) -> bool {
    match value {
        ConfigValue::Atom(_) => false,
        ConfigValue::Call(call) => {
            call.name() == name
                || call
                    .args()
                    .iter()
                    .any(|arg| config_value_contains_call(arg.value(), name))
        }
    }
}

// =============================================================================
// Display — used to round-trip the spec back into `--search SPEC` form
// =============================================================================

impl fmt::Display for HeuristicSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.args.is_empty() {
            return write!(f, "{}()", self.name);
        }
        write!(f, "{}(", self.name)?;
        for (i, arg) in self.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            match arg.key() {
                Some(k) => write!(f, "{k}={}", fmt_value(arg.value()))?,
                None => write!(f, "{}", fmt_value(arg.value()))?,
            }
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

pub fn parse_heuristic_spec(raw: &str) -> Result<HeuristicSpec, String> {
    let mut input = raw.trim();
    input = input
        .strip_suffix('.')
        .or_else(|| input.strip_suffix(';'))
        .unwrap_or(input)
        .trim();
    let call = ConfigParser::new(input).parse_all()?;
    Ok(HeuristicSpec {
        name: call.name,
        args: call.args,
    })
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
    // Defer named/positional/duplicate validation to the applier — it owns the
    // canonical option order, so it's the natural place to enforce it.
    Ok(HeuristicSpec {
        name: call.name,
        args: call.args,
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
// Per-config option appliers
//
// Most typed configs derive `ApplyOptions` directly (in planforge-search,
// next to each struct). At a construction site, call `cfg.apply_options(&args)?`.
//
// `CegarConfig` is the one exception: it's in planforge-search but only a
// curated subset of its fields is CLI-exposed (the rest are internal-only
// runtime flags). Sprinkling `#[option(skip)]` on those would couple the
// search-engine struct to CLI concerns, so `apply_da_options` below stays
// hand-written. It uses `for_each_option` + the `parse` helper directly.
// =============================================================================

/// Type-inferring shortcut for `<T as FromOptionValue>::from_option_value(value)`.
/// `cfg.field = parse(value)?` — the field type drives `T`.
fn parse<T: FromOptionValue>(value: &ConfigValue) -> Result<T, String> {
    T::from_option_value(value)
}

/// Apply `domain_abstraction(...)` options directly onto a `CegarConfig`.
pub fn apply_da_options(cfg: &mut CegarConfig, args: &[ConfigArg]) -> Result<(), String> {
    const ORDER: &[&str] = &[
        "max_abstraction_size",
        "max_iterations",
        "max_time",
        "use_wildcard_plans",
        "combine_labels",
        "random_seed",
        "flaw_treatment",
        "flaw_kind",
        "init_split_method",
    ];
    for_each_option(args, ORDER, |key, value| {
        match key {
            "max_abstraction_size" => cfg.max_abstraction_size = parse(value)?,
            "max_iterations" => cfg.max_iterations = parse(value)?,
            "max_time" => {
                let seconds = f64::from_option_value(value)?;
                cfg.max_time = if seconds.is_infinite() {
                    None
                } else {
                    Some(Duration::try_from_secs_f64(seconds).map_err(|error| {
                        format!("invalid domain-abstraction max_time {seconds}: {error}")
                    })?)
                };
            }
            "use_wildcard_plans" => cfg.use_wildcard_plans = parse(value)?,
            "combine_labels" => cfg.combine_labels = parse(value)?,
            "random_seed" => cfg.random_seed = parse(value)?,
            "flaw_treatment" => cfg.flaw_treatment = parse(value)?,
            "flaw_kind" => cfg.flaw_kind = parse(value)?,
            "init_split_method" => cfg.init_split_method = parse(value)?,
            other => {
                return Err(format!("unknown option `{other}` for `domain_abstraction`"));
            }
        }
        Ok(())
    })
}
