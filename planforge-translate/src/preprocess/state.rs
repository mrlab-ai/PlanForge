use std::collections::HashMap;

use tracing::debug;

use super::variable::{ExplicitVariable, NumericVariable};
use crate::sas_tasks::SASInit;

#[derive(Debug, Clone, Default)]
pub struct State {
    values: HashMap<usize, usize>,
    numeric_values: HashMap<usize, f64>,
}

impl State {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_sas(init: &SASInit) -> Self {
        let values = init
            .values
            .iter()
            .enumerate()
            .map(|(var, &value)| {
                let value = usize::try_from(value).unwrap_or_else(|_| {
                    panic!("variable {var} has no initial value (got {value})")
                });
                (var, value)
            })
            .collect();
        let numeric_values = init.num_values.iter().copied().enumerate().collect();
        Self {
            values,
            numeric_values,
        }
    }

    pub fn get_nv(&self, var: usize) -> f64 {
        *self.numeric_values.get(&var).unwrap()
    }

    pub fn get(&self, var: usize) -> usize {
        *self.values.get(&var).unwrap()
    }

    pub fn dump(&self, variables: &[ExplicitVariable], numeric_variables: &[NumericVariable]) {
        for (var, value) in &self.values {
            let name = variables[*var].get_name();
            debug!("  {}: {}", name, *value);
        }
        for (var, value) in &self.numeric_values {
            let name = numeric_variables[*var].get_name();
            debug!("  {}: {}", name, *value);
        }
    }
}
