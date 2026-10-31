/// The node counts a search reports, both in total and up to the last jump in
/// the priority of the expanded nodes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NodeCounters {
    pub expanded: usize,
    /// Expansions of a state that had already been expanded on a costlier
    /// path.
    pub reopened: usize,
    /// States evaluated for the first time.
    pub evaluated: usize,
    pub generated: usize,
}

/// What a search counted, in Fast Downward's terms.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchStatistics {
    pub nodes: NodeCounters,
    /// [`Self::nodes`] when the search first expanded a node of its final
    /// priority layer (Fast Downward's "until last jump"). Only searches with
    /// a monotone priority have layers; the others leave these at zero.
    pub nodes_until_last_jump: NodeCounters,
    /// Heuristic calls: [`NodeCounters::evaluated`] plus re-evaluations of
    /// known states and lazy evaluations.
    pub evaluations: usize,
    pub dead_ends: usize,
    pub registered_states: usize,
}
