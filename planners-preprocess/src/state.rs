use std::collections::HashMap;

use crate::helper_functions::check_magic;
use crate::helper_functions::InputStream;
use crate::variable::{NumericVariable, Variable};

#[derive(Debug, Clone)]
pub struct State {
    values: HashMap<*const Variable, i32>,
    numeric_values: HashMap<*const NumericVariable, f64>,
}

impl State {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
            numeric_values: HashMap::new(),
        }
    }

    pub fn from_stream(
        stream: &mut InputStream,
        variables: &Vec<*mut Variable>,
        numeric_variables: &Vec<*mut NumericVariable>,
    ) -> Self {
        check_magic(stream, "begin_state");
        let mut values: HashMap<*const Variable, i32> = HashMap::new();
        for var in variables {
            let value = stream.read_i32();
            values.insert(*var as *const Variable, value);
        }
        check_magic(stream, "end_state");

        check_magic(stream, "begin_numeric_state");
        let mut numeric_values: HashMap<*const NumericVariable, f64> = HashMap::new();
        for numvar in numeric_variables {
            let value_str = stream.read_token();
            let num_value = value_str.parse::<f64>().unwrap_or(0.0);
            numeric_values.insert(*numvar as *const NumericVariable, num_value);
        }
        check_magic(stream, "end_numeric_state");

        Self {
            values,
            numeric_values,
        }
    }

    pub fn get_nv(&self, var: *const NumericVariable) -> f64 {
        assert!(!var.is_null());
        *self.numeric_values.get(&var).unwrap()
    }

    pub fn numeric_size(&self) -> usize {
        self.numeric_values.len()
    }

    pub fn get(&self, var: *const Variable) -> i32 {
        *self.values.get(&var).unwrap()
    }

    pub fn dump(&self) {
        for (var, value) in &self.values {
            let name = unsafe { &**var }.get_name();
            println!("  {}: {}", name, value);
        }
    }
}
