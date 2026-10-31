//! The order in which best-first search expands its open nodes.

/// The priority of an open node, from its path cost `g` and heuristic
/// estimate `h`; smaller priorities are expanded first. Statically dispatched,
/// so the search loop is compiled once per ordering.
pub trait NodeOrdering {
    /// The name of the priority in progress reports.
    const LABEL: &'static str;
    /// Whether the priority never decreases along the expansion order, which
    /// makes priority layers (and "until last jump" statistics) meaningful.
    const MONOTONE: bool;
    fn priority(g: f64, h: f64) -> f64;
}

/// A*: `g + h`.
pub struct AStar;
impl NodeOrdering for AStar {
    const LABEL: &'static str = "f";
    const MONOTONE: bool = true;
    #[inline]
    fn priority(g: f64, h: f64) -> f64 {
        g + h
    }
}

/// Greedy best-first search: `h`.
pub struct Greedy;
impl NodeOrdering for Greedy {
    const LABEL: &'static str = "h";
    const MONOTONE: bool = false;
    #[inline]
    fn priority(_g: f64, h: f64) -> f64 {
        h
    }
}

/// Uniform-cost search: `g`.
pub struct UniformCost;
impl NodeOrdering for UniformCost {
    const LABEL: &'static str = "g";
    const MONOTONE: bool = true;
    #[inline]
    fn priority(g: f64, _h: f64) -> f64 {
        g
    }
}
