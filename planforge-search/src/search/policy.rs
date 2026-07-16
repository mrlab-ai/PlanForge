use crate::evaluation::Heuristic;

/// The best-first search strategy. Closed set of three variants; owns the
/// per-strategy differences (open-list priority key, whether f-layer progress
/// is reported, and the lazily-evaluated slow heuristic for fast/slow A*).
pub enum SearchPolicy<'a> {
    /// A*: open-list key f = g + h; reports f-layers.
    AStar,
    /// Greedy best-first: key is h only; g still tracked for plan cost but not
    /// used for ordering; does not report f-layers (h is non-monotonic).
    Gbfs,
    /// A* with a fast ordering heuristic plus a slower, tighter heuristic
    /// evaluated lazily on first pop. Same priority key and reporting as A*.
    FastSlow { slow: Box<dyn Heuristic + 'a> },
}

impl<'a> SearchPolicy<'a> {
    /// Open-list priority for a state with path cost `g` and heuristic `h`.
    #[inline]
    pub(crate) fn priority_value(&self, g_value: f64, h_value: f64) -> f64 {
        match self {
            SearchPolicy::Gbfs => h_value,
            _ => g_value + h_value,
        }
    }

    /// Label used in progress logging ("f" for A*/fast-slow, "h" for GBFS).
    #[inline]
    pub(crate) fn priority_label(&self) -> &'static str {
        match self {
            SearchPolicy::Gbfs => "h",
            _ => "f",
        }
    }

    /// GBFS's h-only key is non-monotonic, so the "next f-layer" abstraction
    /// does not apply; only A*/fast-slow report f-layers.
    #[inline]
    pub(crate) fn reports_f_layers(&self) -> bool {
        !matches!(self, SearchPolicy::Gbfs)
    }

    #[inline]
    pub(crate) fn is_gbfs(&self) -> bool {
        matches!(self, SearchPolicy::Gbfs)
    }
}
