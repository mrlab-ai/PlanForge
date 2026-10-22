/// Statically dispatched priority function for best-first search.
pub trait Priority {
    const LABEL: &'static str;
    const MONOTONE: bool;
    fn key(g: f64, h: f64) -> f64;
}

/// A*: path cost plus the heuristic estimate.
pub struct GPlusH;
impl Priority for GPlusH {
    const LABEL: &'static str = "f";
    const MONOTONE: bool = true;
    #[inline]
    fn key(g: f64, h: f64) -> f64 {
        g + h
    }
}

/// Greedy best-first search: the heuristic estimate alone.
pub struct H;
impl Priority for H {
    const LABEL: &'static str = "h";
    const MONOTONE: bool = false;
    #[inline]
    fn key(_g: f64, h: f64) -> f64 {
        h
    }
}

/// Uniform-cost search: path cost alone.
pub struct G;
impl Priority for G {
    const LABEL: &'static str = "g";
    const MONOTONE: bool = true;
    #[inline]
    fn key(g: f64, _h: f64) -> f64 {
        g
    }
}
