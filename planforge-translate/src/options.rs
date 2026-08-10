//! Configuration options for the translator.

/// Whether to use partial encoding (default: true)
pub const USE_PARTIAL_ENCODING: bool = true;

/// Maximum candidates for invariant generation (default: 100000)
pub const INVARIANT_GENERATION_MAX_CANDIDATES: usize = 100000;

/// Maximum time for invariant generation in seconds (default: 300)
pub const INVARIANT_GENERATION_MAX_TIME: u64 = 300;

/// Whether to add implied preconditions (default: false)
pub const ADD_IMPLIED_PRECONDITIONS: bool = false;

/// Whether to filter unreachable facts (default: true)
pub const FILTER_UNREACHABLE_FACTS: bool = true;

/// How the translator spreads derived variables over axiom layers.
///
/// Both strategies produce a valid layering; they trade the number of layers
/// against how much each layer's fixpoint has to do. A layer is a fixpoint
/// computation over the rules assigned to it, so `Min` runs fewer, larger ones
/// and `Max` more, smaller ones.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LayerStrategy {
    /// Put as many variables in one layer as the negative edges allow: every
    /// cluster takes the lowest layer its children leave it.
    #[default]
    Min,
    /// Put every cluster in a layer of its own, so a derived variable shares a
    /// layer only with the variables it is in a positive cycle with.
    Max,
}

impl std::str::FromStr for LayerStrategy {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "min" => Ok(LayerStrategy::Min),
            "max" => Ok(LayerStrategy::Max),
            other => Err(format!(
                "unknown layer strategy {other:?}; use `min` or `max`"
            )),
        }
    }
}
