//! The order in which best-first search expands its open nodes.

use ordered_float::OrderedFloat;

/// The order of open nodes: a key computed from a node's path cost `g` and
/// heuristic estimate `h`. Open lists expand smaller keys first and equal
/// keys first in, first out. Statically dispatched, so the search loop and
/// the built-in open lists are compiled once per ordering.
pub trait NodeOrdering: 'static {
    /// The name of the priority in progress reports.
    const LABEL: &'static str;
    /// Whether the priority never decreases along the expansion order, which
    /// makes priority layers (and "until last jump" statistics) meaningful.
    const MONOTONE: bool;
    type Key: Ord + Copy;
    fn key(g: f64, h: f64) -> Self::Key;
    /// The priority, the first component of the key.
    fn priority(g: f64, h: f64) -> f64;
}

/// A*: `f = g + h`, ties broken by smaller `h`.
pub struct AStar;
impl NodeOrdering for AStar {
    const LABEL: &'static str = "f";
    const MONOTONE: bool = true;
    type Key = (OrderedFloat<f64>, OrderedFloat<f64>);
    #[inline]
    fn key(g: f64, h: f64) -> Self::Key {
        (OrderedFloat(g + h), OrderedFloat(h))
    }
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
    type Key = OrderedFloat<f64>;
    #[inline]
    fn key(_g: f64, h: f64) -> Self::Key {
        OrderedFloat(h)
    }
    #[inline]
    fn priority(_g: f64, h: f64) -> f64 {
        h
    }
}
