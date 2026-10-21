//! Typed-config option machinery: the AST a `--search` spec parses into, and
//! the two traits that turn it into typed config structs.
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
use std::time::Duration;

pub use parser::{parse_call, parse_heuristic_spec};
pub use planforge_config_derive::{ApplyOptions, FromOptionValue};

mod parser;
mod search;
pub use search::{SearchConfig, parse_search_spec};

/// Declares a name-indexed static plugin table and its lookup function.
///
/// The descriptor type decides what a plugin must declare. Registry users add
/// one entry containing every required field; omitted fields are therefore a
/// compile error. Heuristics and search algorithms share this mechanism.
#[macro_export]
macro_rules! plugin_registry {
    (
        $registry_vis:vis static $registry:ident: $descriptor:ty;
        $lookup_vis:vis fn $lookup:ident;
        entries {
            $(
                $(#[$entry_meta:meta])*
                $name:expr => $value:expr
            ),+ $(,)?
        }
    ) => {
        $registry_vis static $registry: &[(&str, $descriptor)] = &[
            $(
                $(#[$entry_meta])*
                ($name, $value),
            )+
        ];

        $lookup_vis fn $lookup(name: &str) -> Option<&'static $descriptor> {
            $registry
                .iter()
                .find_map(|(registered, descriptor)| (*registered == name).then_some(descriptor))
        }
    };
}

/// Implements `Display` and [`FromOptionValue`] for a fieldless option enum
/// from its `Variant => "name"` pairs; `option` names the option in errors. A
/// variant may accept further spellings, as in
/// `Greedy => "greedy_orders" | "greedy_orders()"`, and displays as the first.
macro_rules! option_enum {
    (
        $option:literal,
        $type:ident { $($variant:ident => $name:literal $(| $alias:literal)*),+ $(,)? }
    ) => {
        impl ::std::fmt::Display for $type {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(match self {
                    $(Self::$variant => $name),+
                })
            }
        }

        impl $crate::config::FromOptionValue for $type {
            fn from_option_value(
                value: &$crate::config::ConfigValue,
            ) -> ::std::result::Result<Self, String> {
                match value.as_atom()? {
                    $($name $(| $alias)* => Ok(Self::$variant),)+
                    other => Err(format!(
                        "invalid {} `{other}`; expected one of {}",
                        $option,
                        [$($name),+].join(", ")
                    )),
                }
            }
        }
    };
}
pub(crate) use option_enum;

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
    List(Vec<ConfigValue>),
}

impl ConfigValue {
    pub fn as_atom(&self) -> Result<&str, String> {
        match self {
            ConfigValue::Atom(value) => Ok(value),
            ConfigValue::Call(call) => Err(format!("expected scalar value, got `{}`", call.name)),
            ConfigValue::List(_) => Err("expected scalar value, got a list".to_string()),
        }
    }

    pub fn as_call(&self) -> Result<&ConfigCall, String> {
        match self {
            ConfigValue::Call(call) => Ok(call),
            ConfigValue::Atom(name) => Err(format!("expected call, got atom `{name}`")),
            ConfigValue::List(_) => Err("expected call, got a list".to_string()),
        }
    }

    /// Whether this value is, or nests, a call to `name`.
    pub fn contains_call(&self, name: &str) -> bool {
        match self {
            // A zero-argument call such as `blind()` is parsed as a bare atom,
            // so an atom spelled `name` *is* a call to `name`.
            ConfigValue::Atom(atom) => atom == name,
            ConfigValue::Call(call) => {
                call.name() == name
                    || call
                        .args()
                        .iter()
                        .any(|arg| arg.value().contains_call(name))
            }
            ConfigValue::List(values) => values.iter().any(|value| value.contains_call(name)),
        }
    }
}

/// A parsed heuristic configuration. The heuristic is identified by `name`; its
/// options are an ordered list of [`ConfigArg`]s (each optionally keyed),
/// applied to a typed config struct at construction time.
///
/// Storing args as `Vec<ConfigArg>` (not a map) lets each config resolve
/// positional args against its own option order -- so both
/// `greedy_numeric_pdb(max_pdb_states=321)` and `greedy_numeric_pdb(321)` work,
/// and they can be mixed: `greedy_numeric_pdb(321, numeric_first=false)`.
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
        Self::new("blind", Vec::new())
    }

    /// Read a nested heuristic out of an option value, as in
    /// `check_admissible(<inner>)` or `astar_fs(fast=<inner>, ...)`.
    ///
    /// Named/positional/duplicate validation is deferred to the heuristic's own
    /// config, which owns the canonical option order.
    pub fn from_value(value: &ConfigValue) -> Result<Self, String> {
        match value {
            ConfigValue::Atom(name) => Ok(Self::new(name.clone(), Vec::new())),
            ConfigValue::Call(call) => Ok(Self::new(call.name.clone(), call.args.clone())),
            ConfigValue::List(_) => Err("expected heuristic, got a list".to_string()),
        }
    }

    pub fn contains_call(&self, name: &str) -> bool {
        self.name == name || self.args.iter().any(|arg| arg.value().contains_call(name))
    }
}

impl std::fmt::Display for HeuristicSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.args.is_empty() {
            return write!(f, "{}()", self.name);
        }
        write!(f, "{}(", self.name)?;
        for (index, arg) in self.args.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            match arg.key() {
                Some(key) => write!(f, "{key}={}", arg.value())?,
                None => write!(f, "{}", arg.value())?,
            }
        }
        write!(f, ")")
    }
}

impl std::fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigValue::Atom(atom) => f.write_str(atom),
            ConfigValue::Call(call) => write!(f, "{call}"),
            ConfigValue::List(values) => {
                f.write_str("[")?;
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{value}")?;
                }
                f.write_str("]")
            }
        }
    }
}

impl std::fmt::Display for ConfigCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.args.is_empty() {
            return write!(f, "{}()", self.name);
        }
        write!(f, "{}(", self.name)?;
        for (index, arg) in self.args.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            match arg.key() {
                Some(key) => write!(f, "{key}={}", arg.value())?,
                None => write!(f, "{}", arg.value())?,
            }
        }
        write!(f, ")")
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
                        if positional_order.is_empty() {
                            "options must be named, as in `key=value`".to_string()
                        } else {
                            format!(
                                "too many positional arguments (maximum {})",
                                positional_order.len()
                            )
                        }
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
//
// Deliberately not sealed. Sealing them meant a new option type could only be
// added inside this crate, which is why heuristic construction used to live in
// a crate of its own and adding a heuristic touched two crates. Anything that
// can be spelled as a `--search` option value is welcome to implement these.
// =============================================================================

/// Implemented by typed configs that can be populated from a `&[ConfigArg]`.
/// Normally derived via `#[derive(ApplyOptions)]`; written by hand only for
/// configs whose CLI surface differs structurally from the struct layout
/// (e.g. coupled writes, curated subsets).
pub trait ApplyOptions {
    fn apply_options(&mut self, args: &[ConfigArg]) -> Result<(), String>;
}

/// `config` with the options `args` of `name(...)` applied; errors name it.
pub fn configure<C: ApplyOptions>(
    name: &str,
    mut config: C,
    args: &[ConfigArg],
) -> Result<C, String> {
    config
        .apply_options(args)
        .map_err(|error| format!("{name}(...): {error}"))?;
    Ok(config)
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
        atom(value)?.parse::<usize>().map_err(|_| {
            format!(
                "expected non-negative integer, got `{}`",
                atom(value).unwrap()
            )
        })
    }
}

impl FromOptionValue for u64 {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        atom(value)?.parse::<u64>().map_err(|_| {
            format!(
                "expected non-negative integer, got `{}`",
                atom(value).unwrap()
            )
        })
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

impl<T: FromOptionValue> FromOptionValue for Option<T> {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        if matches!(value, ConfigValue::Atom(atom) if atom.eq_ignore_ascii_case("none")) {
            Ok(None)
        } else {
            T::from_option_value(value).map(Some)
        }
    }
}

impl<T: FromOptionValue> FromOptionValue for Vec<T> {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        match value {
            ConfigValue::List(values) => values.iter().map(T::from_option_value).collect(),
            value => T::from_option_value(value).map(|value| vec![value]),
        }
    }
}

/// A time limit in seconds; `infinity` is none.
impl FromOptionValue for Option<Duration> {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        crate::resource_limits::time_limit_from_secs(f64::from_option_value(value)?)
            .map_err(|error| error.to_string())
    }
}

impl FromOptionValue for String {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String> {
        Ok(atom(value)?.to_string())
    }
}

// Enum `FromOptionValue` impls live next to each enum, generated by
// [`option_enum!`].

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default, PartialEq, ApplyOptions)]
    struct ExampleConfig {
        #[option(positional)]
        count: usize,
        named_only: bool,
    }

    #[derive(Debug, PartialEq, FromOptionValue)]
    enum Example {
        #[option(rename = "example", alias = "legacy_example")]
        Config(ExampleConfig),
        #[option(rename = "unit")]
        Unit,
    }

    fn decode(raw: &str) -> Result<Example, String> {
        Example::from_option_value(&ConfigValue::Call(parse_call(raw)?))
    }

    #[test]
    fn data_variants_decode_typed_calls_and_explicit_positions() {
        assert_eq!(
            decode("example(7,named_only=true)").unwrap(),
            Example::Config(ExampleConfig {
                count: 7,
                named_only: true
            })
        );
        assert_eq!(
            decode("legacy_example(count=7)").unwrap(),
            decode("example(7)").unwrap()
        );
        assert_eq!(decode("unit()").unwrap(), Example::Unit);
        assert!(decode("unit(1)").is_err());
        assert!(decode("unknown()").unwrap_err().contains("Example"));
        assert!(
            decode("example(7,true)")
                .unwrap_err()
                .contains("too many positional")
        );
    }

    #[test]
    fn vectors_accept_lists_or_singletons_and_options_share_none_handling() {
        let scalar = ConfigValue::Atom("7".into());
        assert_eq!(Vec::<usize>::from_option_value(&scalar).unwrap(), [7]);
        assert_eq!(
            Vec::<usize>::from_option_value(&ConfigValue::List(vec![scalar.clone(), scalar]))
                .unwrap(),
            [7, 7]
        );
        let none = ConfigValue::Atom("NoNe".into());
        assert_eq!(Option::<usize>::from_option_value(&none).unwrap(), None);
        assert_eq!(Option::<bool>::from_option_value(&none).unwrap(), None);
        let nested = ConfigValue::List(vec![ConfigValue::List(vec![none])]);
        assert_eq!(
            Vec::<Vec<Option<bool>>>::from_option_value(&nested).unwrap(),
            [vec![None]]
        );
    }
}
