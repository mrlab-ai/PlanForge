use std::collections::HashMap;

use log::debug;

use crate::helper_functions::InputStream;
use crate::helper_functions::check_magic;
use crate::variable::{ExplicitVariable, NumericVariable};

#[derive(Debug, Clone, Default)]
pub struct State {
    values: HashMap<usize, usize>,
    numeric_values: HashMap<usize, f64>,
}

impl State {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_stream(
        stream: &mut InputStream,
        variables: &[ExplicitVariable],
        numeric_variables: &[NumericVariable],
    ) -> Self {
        check_magic(stream, "begin_state");
        let mut values: HashMap<usize, usize> = HashMap::new();
        for var in variables {
            let value = stream.read_usize();
            values.insert(var.index, value);
        }
        check_magic(stream, "end_state");

        check_magic(stream, "begin_numeric_state");
        let mut numeric_values: HashMap<usize, f64> = HashMap::new();
        for numvar in numeric_variables {
            let value_str = stream.read_token();
            let num_value = value_str.parse::<f64>().unwrap_or(0.0);
            numeric_values.insert(numvar.index, num_value);
        }
        check_magic(stream, "end_numeric_state");

        Self {
            values,
            numeric_values,
        }
    }

    pub fn get_nv(&self, var: usize) -> f64 {
        *self.numeric_values.get(&var).unwrap()
    }

    pub fn numeric_size(&self) -> usize {
        self.numeric_values.len()
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
