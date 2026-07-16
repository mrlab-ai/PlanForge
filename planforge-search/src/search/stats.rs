use std::env;

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

#[derive(Debug, Clone, Copy)]
pub(crate) struct TraceFlags {
    pub(crate) expanded_states: bool,
    pub(crate) initial_successors: bool,
    pub(crate) improved_duplicates: bool,
    pub(crate) generated_states: bool,
    pub(crate) evaluated_successors: bool,
}

impl TraceFlags {
    pub(crate) fn from_environment() -> Self {
        Self {
            expanded_states: env::var_os("TRACE_EXPANDED_STATES").is_some(),
            initial_successors: env::var_os("TRACE_INITIAL_SUCCESSORS").is_some(),
            improved_duplicates: env::var_os("TRACE_IMPROVED_DUPLICATES").is_some(),
            generated_states: env::var_os("TRACE_GENERATED_STATES").is_some(),
            evaluated_successors: env::var_os("TRACE_EVALUATED_SUCCESSORS").is_some(),
        }
    }
}
