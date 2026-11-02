//! PlanForge's heuristics, one module per technique, built on the search core
//! of `planforge_search` and nothing else:
//!
//! - [`abstractions`]: Cartesian and domain abstractions refined by CEGAR,
//!   pattern databases, their collections and cost partitionings;
//! - [`landmarks`]: numeric LM-cut;
//! - [`relaxation`]: the numeric FF heuristic;
//! - [`potentials`]: numeric potential heuristics (with the `cplex` feature);
//! - [`combinations`]: the maximum over heuristics and the admissibility
//!   check around one;
//! - [`requirements`]: what these heuristics need of the task and of the
//!   build.
//!
//! A binary registers [`heuristics`] and [`transformations`] with the core's
//! registries before it parses a configuration; `planforge` does so for its
//! command line.

// The configuration derive names `::planforge_search::config` paths.
pub mod abstractions;
pub mod combinations;
pub mod landmarks;
pub(crate) mod maximal_cliques;
pub mod potentials;
pub mod relaxation;
pub mod requirements;

use abstractions::cartesian::CartesianAbstractionCollectionConfig;
use abstractions::domain::config::DomainAbstractionCollectionConfig;
use abstractions::features::{CartesianFeature, DomainFeature, SourceFeature};
use abstractions::icaps26::Icaps26CartesianConfig;
use abstractions::sources::PdbSourceConfig;
use planforge_search::heuristic_features::{HeuristicEntry, entry};
use planforge_search::task_transformations::TransformationEntry;

/// Every heuristic of this crate, as a binary registers them.
pub fn heuristics() -> Vec<HeuristicEntry> {
    vec![
        entry::<combinations::check_admissible::CheckAdmissibleFeature>(),
        entry::<relaxation::FfFeature>(),
        entry::<landmarks::lm_cut_numeric_heuristic::LmCutNumericFeature>(),
        entry::<potentials::NumericPotentialFeature>(),
        entry::<DomainFeature>(),
        entry::<CartesianFeature>(),
        entry::<SourceFeature<DomainAbstractionCollectionConfig>>(),
        entry::<SourceFeature<CartesianAbstractionCollectionConfig>>(),
        entry::<SourceFeature<PdbSourceConfig>>(),
        entry::<SourceFeature<Icaps26CartesianConfig>>(),
        entry::<abstractions::cost_partitioning::max::MaxFeature>(),
        entry::<abstractions::cost_partitioning::canonical::CanonicalFeature>(),
        entry::<abstractions::cost_partitioning::saturated::ScpFeature>(),
        entry::<abstractions::cost_partitioning::posthoc_optimization::PostHocFeature>(),
    ]
}

/// Every task transformation of this crate, as a binary registers them.
pub fn transformations() -> Vec<TransformationEntry> {
    vec![planforge_search::task_transformations::entry::<
        abstractions::icaps26::Icaps26RestrictFeature,
    >()]
}

/// Registers [`heuristics`] for a test process.
#[cfg(test)]
pub(crate) fn register_heuristics_for_tests() {
    planforge_search::heuristic_features::HEURISTICS.ensure_registered(heuristics);
}
