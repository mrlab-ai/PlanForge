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
//! [`features`] is the core's built-in features extended by this crate's;
//! a program parses its configurations with it, or with its own extension
//! of it.

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
use planforge_search::heuristic_features::entry;
use planforge_search::registry::Features;

/// The core's built-in features with this crate's heuristics and task
/// transformations added.
pub fn features() -> Features {
    let mut features = Features::builtin();
    features
        .heuristics
        .extend([
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
        ])
        .expect("the heuristics of this crate have distinct names");
    features
        .transformations
        .add(planforge_search::task_transformations::entry::<
            abstractions::icaps26::Icaps26RestrictFeature,
        >())
        .expect("the transformations of this crate have distinct names");
    features
}
