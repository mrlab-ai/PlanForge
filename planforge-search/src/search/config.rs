use planforge_sas::{
    numeric_task::{ExplicitValueIndex, NumericValue, OperatorIndex},
    state_registry::ExpansionContext,
};
use std::time::Duration;

pub(crate) struct SearchConfig {
    pub(crate) operator_costs: Vec<NumericValue>,
    pub(crate) use_metric: bool,
    pub(crate) time_limit: Option<Duration>,
    pub(crate) max_memory_bytes: Option<u64>,
}

#[derive(Default)]
pub(crate) struct ExpansionScratch {
    pub(crate) state_values: Vec<ExplicitValueIndex>,
    pub(crate) applicable_operators: Vec<OperatorIndex>,
    pub(crate) successor_numeric: Vec<NumericValue>,
    pub(crate) successor_cost: Vec<NumericValue>,
    pub(crate) preferred_ids: Vec<OperatorIndex>,
    pub(crate) expansion_context: ExpansionContext,
}

impl ExpansionScratch {
    pub(crate) fn with_capacity(num_variables: usize, num_numeric_variables: usize) -> Self {
        Self {
            state_values: Vec::with_capacity(num_variables),
            applicable_operators: Vec::new(),
            successor_numeric: Vec::with_capacity(num_numeric_variables),
            successor_cost: Vec::new(),
            preferred_ids: Vec::new(),
            expansion_context: ExpansionContext::default(),
        }
    }
}
