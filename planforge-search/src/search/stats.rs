#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SearchCounters {
    pub(crate) expanded: usize,
    pub(crate) reopened: usize,
    pub(crate) evaluated: usize,
    pub(crate) generated: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SearchStats {
    pub(crate) nodes_evaluated: usize,
    pub(crate) evaluations: usize,
    pub(crate) nodes_expanded: usize,
    pub(crate) nodes_reopened: usize,
    pub(crate) nodes_generated: usize,
    pub(crate) dead_ends: usize,
    pub(crate) counters_at_last_jump: SearchCounters,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ProgressSnapshot {
    pub(crate) improved: bool,
}
