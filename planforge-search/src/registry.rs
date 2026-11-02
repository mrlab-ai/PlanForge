//! Registries of the features the configuration grammar names: heuristics
//! ([`crate::heuristic_factory::HEURISTICS`]), search algorithms
//! ([`crate::search::SEARCH_ALGORITHMS`]), successor generators
//! ([`crate::successor_generator::SUCCESSOR_GENERATORS`]) and task
//! transformations ([`crate::task_transformations::TRANSFORMATIONS`]).
//!
//! Each registry has built-in entries; a binary registers every other entry
//! once, before it parses a configuration, and a crate adds a feature by
//! handing its entry to that registration.

use std::sync::OnceLock;

use crate::config::{ApplyOptions, ConfigCall, ConfigError, OptionDoc};

/// A named feature: how to parse its configuration `C` from a
/// specification, and the options that configuration accepts.
pub struct Entry<C: 'static> {
    name: &'static str,
    parse: fn(&ConfigCall) -> Result<C, ConfigError>,
    options: fn() -> Vec<OptionDoc>,
}

impl<C> Clone for Entry<C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<C> Copy for Entry<C> {}

impl<C> Entry<C> {
    pub const fn new(
        name: &'static str,
        parse: fn(&ConfigCall) -> Result<C, ConfigError>,
        options: fn() -> Vec<OptionDoc>,
    ) -> Self {
        Self {
            name,
            parse,
            options,
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The options the feature's configuration accepts, with its defaults.
    pub fn options(&self) -> Vec<OptionDoc> {
        (self.options)()
    }

    fn matches(&self, name: &str) -> bool {
        self.name.eq_ignore_ascii_case(name)
    }
}

/// The options a configuration type accepts, with its defaults.
pub fn options_of<C: ApplyOptions + Default>() -> Vec<OptionDoc> {
    let mut options = Vec::new();
    C::default().describe_options(&mut options);
    options
}

/// The entries of one kind of feature: the built-in ones and those a binary
/// registered.
pub struct Registry<C: 'static> {
    kind: &'static str,
    builtin: &'static [Entry<C>],
    registered: OnceLock<Vec<Entry<C>>>,
}

impl<C> Registry<C> {
    pub const fn new(kind: &'static str, builtin: &'static [Entry<C>]) -> Self {
        Self {
            kind,
            builtin,
            registered: OnceLock::new(),
        }
    }

    fn registered(&self) -> &[Entry<C>] {
        self.registered.get_or_init(Vec::new)
    }

    /// Registers the entries besides the built-in ones, once per process and
    /// before any configuration is parsed.
    pub fn register(&self, entries: Vec<Entry<C>>) -> Result<(), String> {
        let kind = self.kind;
        let mut names = std::collections::HashSet::new();
        for entry in &entries {
            if entry.name.is_empty() {
                return Err(format!("{kind} names must not be empty"));
            }
            if self
                .builtin
                .iter()
                .any(|builtin| builtin.matches(entry.name))
            {
                return Err(format!(
                    "{kind} `{}` collides with a built-in {kind}",
                    entry.name
                ));
            }
            if !names.insert(entry.name.to_ascii_lowercase()) {
                return Err(format!(
                    "{kind} `{}` is registered more than once",
                    entry.name
                ));
            }
        }
        self.registered
            .set(entries)
            .map_err(|_| format!("{kind}s have already been registered or used"))
    }

    /// Registers `entries` unless entries were registered already: for test
    /// processes, which all register the same entries.
    pub fn ensure_registered(&self, entries: impl FnOnce() -> Vec<Entry<C>>) {
        self.registered.get_or_init(entries);
    }

    /// Every entry, built-in first, in registration order.
    pub fn entries(&self) -> impl Iterator<Item = &Entry<C>> {
        self.builtin.iter().chain(self.registered())
    }

    pub fn names(&self) -> impl Iterator<Item = &'static str> {
        self.entries().map(Entry::name)
    }

    /// The configuration `spec` names.
    pub fn parse(&self, spec: &ConfigCall) -> Result<C, ConfigError> {
        let entry = self
            .entries()
            .find(|entry| entry.matches(&spec.name))
            .ok_or_else(|| {
                ConfigError::from(format!(
                    "unknown {} `{}`; expected one of {}",
                    self.kind,
                    spec.name,
                    self.names().collect::<Vec<_>>().join(", ")
                ))
            })?;
        (entry.parse)(spec)
    }

    /// Every entry with its options and their defaults, generated from the
    /// configuration types themselves.
    pub fn reference(&self) -> String {
        let mut text = String::new();
        for entry in self.entries() {
            let options = entry.options();
            let positional: Vec<&str> = options
                .iter()
                .filter(|option| option.positional)
                .map(|option| option.name.as_str())
                .collect();
            text.push_str(entry.name());
            text.push('(');
            text.push_str(&positional.join(", "));
            text.push_str(")\n");
            for option in &options {
                text.push_str(&format!("    {} = {}", option.name, option.default));
                if !option.doc.is_empty() {
                    text.push_str(&format!("  -- {}", option.doc));
                }
                text.push('\n');
            }
        }
        text
    }
}
