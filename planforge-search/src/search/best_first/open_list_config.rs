//! The open lists the `open_list` option of a best-first search names.

use std::fmt::{self, Debug};
use std::sync::Arc;

use super::OpenList;
use crate::config::{
    ApplyOptions, ConfigCall, ConfigError, ConfigValue, FromOptionValue, canonical_call, configure,
};
use crate::heuristic_factory::EmptyConfig;
use crate::registry::{Entry, Registry, options_of};

/// An open list the configuration grammar names: its typed configuration
/// and how to build it. The search uses it as a `Box<dyn OpenList>`, so it
/// sees the entries' `g` and `h` but not the search's node ordering.
pub trait OpenListFeature: 'static {
    const NAME: &'static str;
    type Config: ApplyOptions + Default + Debug + Send + Sync + 'static;
    fn build(config: &Self::Config) -> Box<dyn OpenList>;
}

trait ErasedOpenListConfig: Debug + Send + Sync {
    fn call(&self) -> String;
    fn build(&self) -> Box<dyn OpenList>;
}

struct Configured<F: OpenListFeature>(F::Config);

impl<F: OpenListFeature> Debug for Configured<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(F::NAME).field(&self.0).finish()
    }
}

impl<F: OpenListFeature> ErasedOpenListConfig for Configured<F> {
    fn call(&self) -> String {
        canonical_call(F::NAME, &self.0)
    }
    fn build(&self) -> Box<dyn OpenList> {
        F::build(&self.0)
    }
}

/// A configured open list. The built-in lists are generic over the node
/// ordering and dispatched statically; a registered one goes through
/// `Box<dyn OpenList>`.
#[derive(Debug, Clone)]
pub enum OpenListConfig {
    /// `tiebreaking()`: [`super::TieBreakingOpenList`].
    TieBreaking,
    /// `preferred_first()`: [`super::PreferredFirstOpenList`].
    PreferredFirst,
    Registered(RegisteredOpenList),
}

/// A registered open list's configuration.
#[derive(Debug, Clone)]
pub struct RegisteredOpenList(Arc<dyn ErasedOpenListConfig>);

impl RegisteredOpenList {
    pub fn build(&self) -> Box<dyn OpenList> {
        self.0.build()
    }
}

/// The canonical call, which parses back to an equal configuration.
impl fmt::Display for OpenListConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TieBreaking => f.write_str("tiebreaking()"),
            Self::PreferredFirst => f.write_str("preferred_first()"),
            Self::Registered(list) => f.write_str(&list.0.call()),
        }
    }
}

impl PartialEq for OpenListConfig {
    fn eq(&self, other: &Self) -> bool {
        self.to_string() == other.to_string()
    }
}

impl FromOptionValue for OpenListConfig {
    fn from_option_value(value: &ConfigValue) -> Result<Self, ConfigError> {
        OPEN_LISTS.parse(&ConfigCall::from_value(value).map_err(ConfigError::from)?)
    }

    fn option_text(&self) -> String {
        self.to_string()
    }
}

/// A registered open list.
pub type OpenListEntry = Entry<OpenListConfig>;

pub const fn entry<F: OpenListFeature>() -> OpenListEntry {
    Entry::new(F::NAME, parse_feature::<F>, options_of::<F::Config>)
}

fn parse_feature<F: OpenListFeature>(spec: &ConfigCall) -> Result<OpenListConfig, ConfigError> {
    let config = configure(F::NAME, F::Config::default(), &spec.args)?;
    Ok(OpenListConfig::Registered(RegisteredOpenList(Arc::new(
        Configured::<F>(config),
    ))))
}

fn parse_tiebreaking(spec: &ConfigCall) -> Result<OpenListConfig, ConfigError> {
    configure("tiebreaking", EmptyConfig {}, &spec.args)?;
    Ok(OpenListConfig::TieBreaking)
}

fn parse_preferred_first(spec: &ConfigCall) -> Result<OpenListConfig, ConfigError> {
    configure("preferred_first", EmptyConfig {}, &spec.args)?;
    Ok(OpenListConfig::PreferredFirst)
}

/// Every open list the `open_list` option names: the built-in ones, and
/// those a binary registers.
pub static OPEN_LISTS: Registry<OpenListConfig> = Registry::new(
    "open list",
    &[
        Entry::new("tiebreaking", parse_tiebreaking, options_of::<EmptyConfig>),
        Entry::new(
            "preferred_first",
            parse_preferred_first,
            options_of::<EmptyConfig>,
        ),
    ],
);

#[cfg(test)]
mod tests {
    use crate::search::parse_search_spec;

    #[test]
    fn a_best_first_search_names_its_open_list() {
        for (spec, canonical) in [
            ("astar(blind(), open_list=tiebreaking())", "astar(blind())"),
            (
                "astar(blind(), open_list=preferred_first())",
                "astar(blind(), open_list=preferred_first())",
            ),
            (
                "gbfs(blind(), open_list=preferred_first())",
                "gbfs(blind())",
            ),
        ] {
            assert_eq!(parse_search_spec(spec).unwrap().to_string(), canonical);
        }
        let error = parse_search_spec("astar(blind(), open_list=tiebreaking(x=1))").unwrap_err();
        assert!(error.contains("tiebreaking"), "{error}");
    }
}
