use super::{BestFirstSearch, SearchBuildContext, SearchDriver};
use crate::evaluation::Heuristic;
use anyhow::{Result, bail};

/// The extension point for best-first search algorithms.
///
/// The common case only implements [`priority`](Self::priority). The generic
/// [`BestFirstSearch`](super::BestFirstSearch) driver supplies the open list,
/// duplicate detection, resource limits, statistics, and plan extraction.
/// Because the driver is generic over this trait, the priority calculation is
/// statically dispatched and can be inlined at every generated node.
pub trait SearchAlgorithm {
    /// Open-list priority for a node with path cost `g` and heuristic `h`.
    fn priority(&self, g_value: f64, h_value: f64) -> f64;

    /// Label used in progress logging.
    fn priority_label(&self) -> &'static str {
        "f"
    }

    /// Whether monotonically increasing priority layers are meaningful.
    fn reports_priority_layers(&self) -> bool {
        true
    }

    /// Whether preferred operators get their own first-choice queue.
    fn uses_preferred_first(&self) -> bool {
        false
    }

    /// Optional slow heuristic evaluated lazily on the first pop.
    fn slow_heuristic(&self) -> Option<&dyn Heuristic> {
        None
    }

    /// Build a complete driver for this algorithm.
    ///
    /// Shallow extensions inherit the monomorphized best-first loop. A
    /// fundamentally different algorithm can override this method and return
    /// its own driver; type erasure still occurs only once, at this boundary.
    fn build<'a>(self, ctx: SearchBuildContext<'a>) -> Result<Box<dyn SearchDriver + 'a>>
    where
        Self: Sized + 'a,
    {
        if ctx.secondary_heuristic.is_some() {
            bail!("the standard best-first driver accepts one heuristic")
        }
        Ok(Box::new(BestFirstSearch::with_algorithm(
            ctx.task,
            ctx.state_registry,
            ctx.primary_heuristic,
            ctx.time_limit,
            ctx.max_memory_bytes,
            self,
            ctx.mpd,
        )))
    }
}

/// Ordinary A*: `f = g + h`.
#[derive(Debug, Default, Clone, Copy)]
pub struct AStar;

impl SearchAlgorithm for AStar {
    #[inline]
    fn priority(&self, g_value: f64, h_value: f64) -> f64 {
        g_value + h_value
    }
}

/// Greedy best-first search: `f = h`.
#[derive(Debug, Default, Clone, Copy)]
pub struct GreedyBestFirst;

impl SearchAlgorithm for GreedyBestFirst {
    #[inline]
    fn priority(&self, _g_value: f64, h_value: f64) -> f64 {
        h_value
    }

    fn priority_label(&self) -> &'static str {
        "h"
    }

    fn reports_priority_layers(&self) -> bool {
        false
    }

    fn uses_preferred_first(&self) -> bool {
        true
    }
}

/// A* whose tighter second heuristic is evaluated lazily on first pop.
pub struct FastSlow<'a> {
    pub(crate) slow: Box<dyn Heuristic + 'a>,
}

impl SearchAlgorithm for FastSlow<'_> {
    #[inline]
    fn priority(&self, g_value: f64, h_value: f64) -> f64 {
        g_value + h_value
    }

    fn slow_heuristic(&self) -> Option<&dyn Heuristic> {
        Some(&*self.slow)
    }
}
