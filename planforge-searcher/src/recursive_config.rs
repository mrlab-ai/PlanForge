//! Compatibility exports for the search configuration owned by planforge-search.
pub use planforge_search::config::{
    ApplyOptions, ConfigArg, ConfigCall, ConfigValue, FromOptionValue, SearchConfig,
    for_each_option, parse_call, parse_search_spec,
};

#[cfg(test)]
mod tests;
