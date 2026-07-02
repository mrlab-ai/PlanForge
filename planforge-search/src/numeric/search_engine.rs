//! Backward-compatible re-export shim. The implementation moved to
//! [`crate::numeric::search`]; import from there in new code.

pub use crate::numeric::search::{
    AStarSearch, Plan, PriorityMode, SearchEngine, SearchResult, SearchStatus,
    compute_effective_operator_costs,
};
