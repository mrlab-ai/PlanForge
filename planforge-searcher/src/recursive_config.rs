use std::fmt;

// Re-export the parser AST nodes + the trait from `planforge-search`, where
// the typed configs live. The derive emits absolute paths into those types,
// so they have to live in one canonical location.
pub use planforge_search::config::{
    ApplyOptions, ConfigArg, ConfigCall, ConfigValue, FromOptionValue, atom, for_each_option,
};

use planforge_search::numeric::evaluation::domain_abstractions::cegar::CegarConfig;
use planforge_search::numeric::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
use planforge_search::numeric::evaluation::domain_abstractions::saturated_cost_partitioning_online_heuristic::{
    FillScpConfig, ScpOnlineConfig,
};

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
// Most typed configs derive `ApplyOptions` directly (see planforge-search:
// `GreedyPatternGeneratorConfig`, `CanonicalNumericPdbConfig`,
// `LmCutNumericConfig`, `FiniteSupportConfig`,
// `DomainAbstractionCollectionGeneratorMultipleCegarConfig`). For those,
// call `cfg.apply_options(&args)?` and you're done.
//
// The configs with structural quirks — `CegarConfig` (curated subset of a
// large struct), `ScpOnlineConfig` / `FillScpConfig` (coupled writes and
// nested call dispatch) — keep their hand-written `apply_*_options`
// functions below, written with the `apply_options!` macro.
// =============================================================================

/// Build an applier body from a list of `"key" => action,` arms.
///
/// Usage (closure-style heading names the bindings the actions can use —
/// macro hygiene means we have to pass them as identifiers):
///
/// ```ignore
/// apply_options!(args, "name", |key, value| {
///     "max_size" => cfg.max_size = parse_usize(atom(value)?)?,
///     "label"    => cfg.label = atom(value)?.to_string(),
///     // optional catch-all (otherwise unknown keys error):
///     _ => fallback_applier(cfg, key, value)?,
/// })
/// ```
macro_rules! apply_options {
    (
        $args:expr, $name:literal, |$key:ident, $value:ident|
        { $($k:literal => $action:expr,)* _ => $fallback:expr $(,)? }
    ) => {{
        const ORDER: &[&str] = &[ $($k,)* ];
        for_each_option($args, ORDER, |$key, $value| {
            match $key {
                $($k => { $action; })*
                _ => { $fallback; }
            }
            Ok(())
        })
    }};
    (
        $args:expr, $name:literal, |$key:ident, $value:ident|
        { $($k:literal => $action:expr,)* $(,)? }
    ) => {{
        const ORDER: &[&str] = &[ $($k,)* ];
        for_each_option($args, ORDER, |$key, $value| {
            match $key {
                $($k => { $action; })*
                other => return Err(format!("unknown option `{other}` for `{}`", $name)),
            }
            Ok(())
        })
    }};
}

/// Type-inferring shortcut for `<T as FromOptionValue>::from_option_value(value)`.
/// Inside an applier arm, write `cfg.field = parse(value)?;` — the field's
/// type drives `T`, which picks the right primitive or enum impl.
fn parse<T: FromOptionValue>(value: &ConfigValue) -> Result<T, String> {
    T::from_option_value(value)
}

/// Apply `domain_abstraction(...)` options directly onto a `CegarConfig`.
///
/// `CegarConfig` is in `planforge-search` and has many internal-only fields,
/// so we can't `#[derive(ApplyOptions)]` on it without polluting it with
/// CLI-specific attributes. Written manually using the `apply_options!`
/// macro + `FromOptionValue` (no per-type parser function call needed).
pub fn apply_da_options(cfg: &mut CegarConfig, args: &[ConfigArg]) -> Result<(), String> {
    apply_options!(args, "domain_abstraction", |key, value| {
        "max_abstraction_size" => cfg.max_abstraction_size = parse(value)?,
        "max_iterations"       => cfg.max_iterations       = parse(value)?,
        "use_wildcard_plans"   => cfg.use_wildcard_plans   = parse(value)?,
        "combine_labels"       => cfg.combine_labels       = parse(value)?,
        "transform_linear_task"=> cfg.transform_linear_task= parse(value)?,
        "random_seed"          => cfg.random_seed          = parse(value)?,
        "flaw_treatment"       => cfg.flaw_treatment       = parse(value)?,
        "flaw_kind"            => cfg.flaw_kind            = parse(value)?,
        "init_split_method"    => cfg.init_split_method    = parse(value)?,
    })
}

/// Backwards-compatible wrapper around the derived
/// `DomainAbstractionCollectionGeneratorMultipleCegarConfig::apply_options`.
/// New code should call the trait method directly.
pub fn apply_da_collection_options(
    cfg: &mut DomainAbstractionCollectionGeneratorMultipleCegarConfig,
    args: &[ConfigArg],
) -> Result<(), String> {
    cfg.apply_options(args)
}


// `apply_scp_online_options`, `apply_fill_scp_options`, `apply_greedy_pdb_options`,
// `apply_canonical_pdb_options`, and `apply_lmcut_options` are all gone —
// their configs `#[derive(ApplyOptions)]` (with `also_sets` for the coupled
// `combine_labels` / `random_seed`, `flatten` + `nested = "collection"` for
// the DA collection sub-config, and `nested = "lmcut"` for the FillScp
// LMcut sub-config). Just call `cfg.apply_options(args)` on them.

// All per-type parser functions are gone — `FromOptionValue` impls in
// `planforge_search::config` cover primitives and option-typed enums, and
// the `parse` helper above dispatches by inferred type at each call site.
