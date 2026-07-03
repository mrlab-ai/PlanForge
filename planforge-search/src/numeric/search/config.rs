use super::stats::TraceFlags;
use planforge_sas::numeric::state_registry::ExpansionContext;
use std::time::Duration;

pub(crate) struct SearchConfig {
    pub(crate) operator_costs: Vec<f64>,
    pub(crate) use_metric: bool,
    pub(crate) time_limit: Option<Duration>,
    pub(crate) max_memory_bytes: Option<u64>,
    pub(crate) trace: TraceFlags,
}

#[derive(Default)]
pub(crate) struct ExpansionScratch {
    pub(crate) state_values: Vec<usize>,
    pub(crate) applicable_operators: Vec<u32>,
    pub(crate) successor_numeric: Vec<f64>,
    pub(crate) successor_cost: Vec<f64>,
    pub(crate) expansion_context: ExpansionContext,
}

impl ExpansionScratch {
    pub(crate) fn with_capacity(num_variables: usize, num_numeric_variables: usize) -> Self {
        Self {
            state_values: Vec::with_capacity(num_variables),
            applicable_operators: Vec::new(),
            successor_numeric: Vec::with_capacity(num_numeric_variables),
            successor_cost: Vec::new(),
            expansion_context: ExpansionContext::default(),
        }
    }
}
