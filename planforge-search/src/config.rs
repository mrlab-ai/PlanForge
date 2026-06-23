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
//
// Sealed via a `pub(crate)` private module so downstream crates can't add
// impls — they'd need to name `::planforge_search::config::sealed::Sealed`
// and that path is `pub(crate)`. Inside this workspace the derive macro
// emits the `Sealed` impl alongside the real impl; hand-written impls do
// the same.
// =============================================================================

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Implemented by typed configs that can be populated from a `&[ConfigArg]`.
/// Normally derived via `#[derive(ApplyOptions)]`; written by hand only for
/// configs whose CLI surface differs structurally from the struct layout
/// (e.g. coupled writes, curated subsets).
pub trait ApplyOptions: sealed::Sealed {
    fn apply_options(&mut self, args: &[ConfigArg]) -> Result<(), String>;
}

/// Implemented by every type that can appear as the value of an option.
/// The derive picks `from_option_value` for each field automatically.
pub trait FromOptionValue: sealed::Sealed + Sized {
    fn from_option_value(value: &ConfigValue) -> Result<Self, String>;
}

// Primitive `Sealed` impls — keep alongside the `FromOptionValue` impls below.
impl sealed::Sealed for bool {}
impl sealed::Sealed for usize {}
impl sealed::Sealed for u64 {}
impl sealed::Sealed for f64 {}
impl sealed::Sealed for Option<u64> {}
impl sealed::Sealed for String {}

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

// Enum `FromOptionValue` impls live next to each enum (search for
// `impl FromOptionValue` in the cegar / pattern_databases / SCP modules).
