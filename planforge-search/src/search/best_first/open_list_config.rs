//! The open lists the `open_list` option of a best-first search names.

use std::fmt::{self, Debug};
use std::sync::Arc;

use super::OpenList;
use crate::config::{
    ApplyOptions, ConfigCall, ConfigError, ConfigValue, EmptyConfig, FromOptionValue, configure,
};
use crate::registry::{Configured, Entry, Features, Typed, options_of, parse_configured};

/// An open list the configuration grammar names: its typed configuration
/// and how to build it. The search uses it as a `Box<dyn OpenList>`, so it
/// sees the entries' `g` and `h` but not the search's node ordering.
pub trait OpenListFeature: 'static {
    const NAME: &'static str;
    type Config: ApplyOptions + Default + Debug + Send + Sync + 'static;
    fn build(config: &Self::Config) -> Box<dyn OpenList>;
}

/// The interface of a registered open list's configuration, whatever its
/// type: what [`RegisteredOpenList`] dereferences to.
pub trait ConfiguredOpenList: Send + Sync {
    fn build(&self) -> Box<dyn OpenList>;
}

impl<F: OpenListFeature> ConfiguredOpenList for Typed<F, F::Config> {
    fn build(&self) -> Box<dyn OpenList> {
        F::build(&self.config)
    }
}

/// A configured open list. The built-in lists are generic over the node
/// ordering and dispatched statically; a registered one goes through
/// `Box<dyn OpenList>`.
#[derive(Debug, Clone, PartialEq)]
pub enum OpenListConfig {
    /// `tiebreaking()`: [`super::TieBreakingOpenList`].
    TieBreaking,
    /// `preferred_first()`: [`super::PreferredFirstOpenList`].
    PreferredFirst,
    Registered(RegisteredOpenList),
}

/// A registered open list's configuration. It displays as its canonical
/// call.
pub type RegisteredOpenList = Configured<dyn ConfiguredOpenList>;

/// The canonical call, which parses back to an equal configuration.
impl fmt::Display for OpenListConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TieBreaking => f.write_str("tiebreaking()"),
            Self::PreferredFirst => f.write_str("preferred_first()"),
            Self::Registered(list) => fmt::Display::fmt(list, f),
        }
    }
}

impl FromOptionValue for OpenListConfig {
    fn from_option_value(value: &ConfigValue, features: &Features) -> Result<Self, ConfigError> {
        features
            .open_lists
            .parse(&ConfigCall::from_value(value)?, features)
    }

    fn option_text(&self) -> String {
        self.to_string()
    }
}

/// An open list's registry entry.
pub type OpenListEntry = Entry<OpenListConfig>;

pub const fn entry<F: OpenListFeature>() -> OpenListEntry {
    Entry::new(F::NAME, parse_feature::<F>, options_of::<F::Config>)
}

fn parse_feature<F: OpenListFeature>(
    spec: &ConfigCall,
    features: &Features,
) -> Result<OpenListConfig, ConfigError> {
    let list = parse_configured(F::NAME, spec, features, |typed: Typed<F, F::Config>| {
        Arc::new(typed) as Arc<dyn ConfiguredOpenList>
    })?;
    Ok(OpenListConfig::Registered(list))
}

fn parse_tiebreaking(
    spec: &ConfigCall,
    features: &Features,
) -> Result<OpenListConfig, ConfigError> {
    configure("tiebreaking", EmptyConfig {}, &spec.args, features)?;
    Ok(OpenListConfig::TieBreaking)
}

fn parse_preferred_first(
    spec: &ConfigCall,
    features: &Features,
) -> Result<OpenListConfig, ConfigError> {
    configure("preferred_first", EmptyConfig {}, &spec.args, features)?;
    Ok(OpenListConfig::PreferredFirst)
}

/// The built-in open lists, `tiebreaking()` and `preferred_first()`.
pub(crate) fn builtin_entries() -> Vec<OpenListEntry> {
    vec![
        Entry::new("tiebreaking", parse_tiebreaking, options_of::<EmptyConfig>),
        Entry::new(
            "preferred_first",
            parse_preferred_first,
            options_of::<EmptyConfig>,
        ),
    ]
}
