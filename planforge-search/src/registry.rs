//! The features the configuration language names, as a value: [`Features`]
//! holds one [`Registry`] per kind of feature -- heuristics, search
//! algorithms, open lists, successor generators and task transformations.
//!
//! A program starts from [`Features::builtin`], adds the entries of the
//! crates it uses and its own with [`Registry::add`], and parses every
//! configuration string with the result. Two programs, or two parts of one,
//! can use different feature sets side by side.
//!
//! A feature of any kind is its kind's trait, such as
//! [`HeuristicFeature`](crate::heuristic_features::HeuristicFeature),
//! implemented once; the kind's `entry::<F>()` makes it a registry entry, and
//! parsing yields a [`Configured`] feature behind the kind's interface.

use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

use crate::config::{
    ApplyOptions, ConfigCall, ConfigError, OptionDoc, canonical_call, configure, parse_call,
};
use crate::heuristic_features::HeuristicConfig;
use crate::search::SearchConfig;
use crate::search::best_first::OpenListConfig;
use crate::successor_generator::SuccessorGeneratorConfig;
use crate::task_transformations::TransformationConfig;

/// Every feature the configuration language names, one registry per kind.
pub struct Features {
    pub heuristics: Registry<HeuristicConfig>,
    pub search_algorithms: Registry<SearchConfig>,
    pub open_lists: Registry<OpenListConfig>,
    pub successor_generators: Registry<SuccessorGeneratorConfig>,
    pub transformations: Registry<TransformationConfig>,
}

impl Features {
    /// The features of this crate: blind search, A* and greedy best-first
    /// search, the tie-breaking and preferred-first open lists, the standard
    /// grounded tree and the restriction.
    pub fn builtin() -> Self {
        Self {
            heuristics: Registry::new(
                "heuristic",
                vec![crate::heuristic_features::entry::<
                    crate::evaluation::BlindFeature,
                >()],
            ),
            search_algorithms: Registry::new(
                "search algorithm",
                vec![
                    crate::search::entry::<crate::search::AStarFeature>(),
                    crate::search::entry::<crate::search::GreedyBestFirstFeature>(),
                ],
            ),
            open_lists: Registry::new("open list", crate::search::best_first::builtin_entries()),
            successor_generators: Registry::new(
                "successor generator",
                vec![crate::successor_generator::entry::<
                    crate::successor_generator::StandardGroundedTreeFeature,
                >()],
            ),
            transformations: Registry::new(
                "task transformation",
                vec![crate::task_transformations::entry::<
                    crate::task_transformations::restriction::RestrictFeature,
                >()],
            ),
        }
    }

    /// The search `text`, a `--search` string, configures.
    pub fn parse_search(&self, text: &str) -> Result<SearchConfig, ConfigError> {
        self.search_algorithms.parse(&parse_call(text)?, self)
    }

    /// The heuristic `text`, such as `lmcutnumeric()`, configures.
    pub fn parse_heuristic(&self, text: &str) -> Result<HeuristicConfig, ConfigError> {
        self.heuristics.parse(&parse_call(text)?, self)
    }

    /// The task transformation `text`, a `--transform` string, configures.
    pub fn parse_transformation(&self, text: &str) -> Result<TransformationConfig, ConfigError> {
        self.transformations.parse(&parse_call(text)?, self)
    }

    /// The `--search` help text: the search algorithms with their options,
    /// and the heuristics.
    pub fn search_help(&self) -> String {
        let algorithms: Vec<String> = self
            .search_algorithms
            .entries()
            .map(|entry| {
                let options: Vec<String> = entry
                    .options()
                    .iter()
                    .map(|option| {
                        if option.positional {
                            format!("[{}]", option.name)
                        } else {
                            format!("{}={}", option.name, option.default)
                        }
                    })
                    .collect();
                format!("{}({})", entry.name(), options.join(", "))
            })
            .collect();
        format!(
            "Search specification. Available algorithms: {}\n\nHeuristics: {}. `planforge heuristics` lists their options and defaults. Custom Rust heuristics can be passed directly as Box<dyn Heuristic>.",
            algorithms.join(", "),
            self.heuristics.names().collect::<Vec<_>>().join(", ")
        )
    }

    /// The `--transform` help text.
    pub fn transformation_help(&self) -> String {
        format!(
            "Task transformation, applied before the search in the order given. Available transformations: {}",
            self.transformations.names().collect::<Vec<_>>().join(", ")
        )
    }
}

/// A named feature: how to parse its configuration `C` from a call, and the
/// options that configuration accepts.
pub struct Entry<C: 'static> {
    name: &'static str,
    parse: fn(&ConfigCall, &Features) -> Result<C, ConfigError>,
    options: fn() -> Vec<OptionDoc>,
}

impl<C> Clone for Entry<C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<C> Copy for Entry<C> {}

impl<C> Debug for Entry<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Entry").field(&self.name).finish()
    }
}

impl<C> Entry<C> {
    pub const fn new(
        name: &'static str,
        parse: fn(&ConfigCall, &Features) -> Result<C, ConfigError>,
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
pub(crate) fn options_of<C: ApplyOptions + Default>() -> Vec<OptionDoc> {
    let mut options = Vec::new();
    C::default().describe_options(&mut options);
    options
}

/// The entries of one kind of feature, in the order they were added.
#[derive(Debug, Clone)]
pub struct Registry<C: 'static> {
    kind: &'static str,
    entries: Vec<Entry<C>>,
}

impl<C> Registry<C> {
    fn new(kind: &'static str, entries: Vec<Entry<C>>) -> Self {
        Self { kind, entries }
    }

    /// Adds `entry`, which configurations then name; fails if its name is
    /// empty or names another entry, ignoring case.
    pub fn add(&mut self, entry: Entry<C>) -> Result<(), ConfigError> {
        let kind = self.kind;
        if entry.name.is_empty() {
            return Err(format!("{kind} names must not be empty").into());
        }
        if self.entries.iter().any(|other| other.matches(entry.name)) {
            return Err(format!("there is a {kind} named `{}` already", entry.name).into());
        }
        self.entries.push(entry);
        Ok(())
    }

    /// Adds each of `entries` as [`Self::add`] does.
    pub fn extend(
        &mut self,
        entries: impl IntoIterator<Item = Entry<C>>,
    ) -> Result<(), ConfigError> {
        entries.into_iter().try_for_each(|entry| self.add(entry))
    }

    /// Every entry, in the order they were added.
    pub fn entries(&self) -> impl Iterator<Item = &Entry<C>> {
        self.entries.iter()
    }

    pub fn names(&self) -> impl Iterator<Item = &'static str> {
        self.entries().map(Entry::name)
    }

    /// The configuration `spec` names, its nested features parsed with
    /// `features`.
    pub fn parse(&self, spec: &ConfigCall, features: &Features) -> Result<C, ConfigError> {
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
        (entry.parse)(spec, features)
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

/// A configured feature behind its kind's interface `T`, such as
/// `dyn ConfiguredHeuristic`, with the call that configures it. Two
/// configurations are equal exactly when their calls are.
pub struct Configured<T: ?Sized> {
    call: String,
    feature: Arc<T>,
}

impl<T: ?Sized> Configured<T> {
    pub(crate) fn new(call: String, feature: Arc<T>) -> Self {
        Self { call, feature }
    }
}

impl<T: ?Sized> Clone for Configured<T> {
    fn clone(&self) -> Self {
        Self {
            call: self.call.clone(),
            feature: Arc::clone(&self.feature),
        }
    }
}

impl<T: ?Sized> Deref for Configured<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.feature
    }
}

/// The call, which parses back to an equal configuration.
impl<T: ?Sized> fmt::Display for Configured<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.call)
    }
}

impl<T: ?Sized> Debug for Configured<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.call)
    }
}

impl<T: ?Sized> PartialEq for Configured<T> {
    fn eq(&self, other: &Self) -> bool {
        self.call == other.call
    }
}

/// The typed configuration `config` of the feature `F`, which implements its
/// kind's interface.
pub struct Typed<F, C> {
    pub(crate) config: C,
    feature: PhantomData<fn() -> F>,
}

impl<F, C> Typed<F, C> {
    pub(crate) fn new(config: C) -> Self {
        Self {
            config,
            feature: PhantomData,
        }
    }
}

/// The feature `F` named `name` and configured by `spec`, behind its kind's
/// interface, into which `erase` turns the typed configuration.
pub(crate) fn parse_configured<F, C, T: ?Sized>(
    name: &'static str,
    spec: &ConfigCall,
    features: &Features,
    erase: fn(Typed<F, C>) -> Arc<T>,
) -> Result<Configured<T>, ConfigError>
where
    C: ApplyOptions + Default,
{
    let config = configure(name, C::default(), &spec.args, features)?;
    Ok(Configured::new(
        canonical_call(name, &config),
        erase(Typed::new(config)),
    ))
}
