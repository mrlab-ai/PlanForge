//! The `--search` grammar: the text the user types, parsed into a
//! [`SearchSpec`].
//!
//! Only the *shape* of a search configuration lives here -- which engine, with
//! which heuristics. What a heuristic name means, and how it is built, lives in
//! `planforge_search::heuristic_factory`, next to the heuristics themselves, so
//! that adding a heuristic never touches this crate.

pub mod recursive_config;
#[cfg(feature = "sgd")]
pub mod sgd;

pub use planforge_search::config::HeuristicSpec;
pub use recursive_config::{SearchSpec, parse_heuristic_spec, parse_search_spec};

/// Fail before translation if any heuristic in `spec` needs a solver backend
/// this build does not have. The per-heuristic knowledge belongs to
/// `planforge-search`; all this adds is walking the engine's heuristics.
pub fn preflight_required_backends(spec: &SearchSpec) -> std::io::Result<()> {
    for heuristic in spec.heuristics() {
        planforge_search::heuristic_factory::preflight_required_backends(heuristic)?;
    }
    Ok(())
}
