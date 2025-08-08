use std::collections::HashSet;

use crate::search::numeric::numeric_task::AbstractNumericTask;
use crate::search::numeric::{
    numeric_task::{NumericRootTask, NumericType},
    utils::int_packer::IntDoublePacker,
};

type StatePacker = IntDoublePacker;

struct StateNotFoundError {
    index: usize,
}

struct StateInsertError {
    message: String,
}

struct GlobalState {}

impl GlobalState {
    pub fn new(state: &StatePacker) -> Self {
        todo!()
    }
}

struct StateRegistry {
    root_task: Box<NumericRootTask>,
    global_state_packer: StatePacker,
    state_data_pool: Vec<StatePacker>,
    numeric_constants: Vec<f64>,
    numeric_indices: Vec<i32>,
    registered_states: HashSet<usize>,
}

impl StateRegistry {
    pub fn new(root_task: Box<NumericRootTask>, global_state_packer: StatePacker) -> Self {
        let number_numeric_vars = root_task.numeric_variables().len();
        StateRegistry {
            root_task,
            global_state_packer,
            state_data_pool: Vec::new(),
            numeric_constants: Vec::new(),
            numeric_indices: vec![-1; number_numeric_vars],
            registered_states: HashSet::new(),
        }
    }

    pub fn register_state(
        &mut self,
        values: Vec<u64>,
        numeric_values: Vec<f64>,
    ) -> Result<GlobalState, StateInsertError> {
        let mut buffer = vec![0; self.global_state_packer.num_bins() as usize];
        for i in 0..values.len() {
            let var_id = i as i32;
            self.global_state_packer.set(&mut buffer, var_id, values[i]);
        }

        let mut regular_index = values.len() as i32;
        let mut derived_index = 0;

        let mut instrumentation_variables = vec![];

        for i in 0..numeric_values.len() {
            let numeric_variable =
                self.root_task
                    .numeric_variables()
                    .get(i)
                    .ok_or_else(|| StateInsertError {
                        message: format!("Numeric variable at index {} not found", i),
                    })?;
            match numeric_variable.get_type() {
                NumericType::Instrumentation => {
                    assert!(self.numeric_indices.get(i) == Some(&-1));
                    self.numeric_indices[i] = instrumentation_variables.len() as i32;
                    instrumentation_variables.push(numeric_values[i]);
                }

                NumericType::Constant => {
                    assert!(self.numeric_indices.get(i) == Some(&-1));
                }

                NumericType::Unknown => {
                    assert!(false);
                    return Err(StateInsertError {
                        message: "Unknown numeric type encountered".to_string(),
                    });
                }

                NumericType::Derived => {
                    derived_index += 1;
                }

                NumericType::Regular => {
                    assert!(self.numeric_indices.get(i) == Some(&-1));
                    self.numeric_indices[i] = regular_index as i32;
                    let packed_numeric_value =
                        self.global_state_packer.pack_double(numeric_values[i]);
                    self.global_state_packer
                        .set(&mut buffer, regular_index, packed_numeric_value);
                    regular_index += 1;
                }

                _ => {
                    return Err(StateInsertError {
                        message: format!(
                            "Unexpected numeric type at index {}: {:?}",
                            i,
                            numeric_variable.get_type()
                        ),
                    });
                }
            }
        }

        drop(buffer);

        todo!()
    }

    pub fn lookup_state(&self, index: usize) -> Result<GlobalState, StateNotFoundError> {
        match self.state_data_pool.get(index) {
            Some(state) => Ok(GlobalState::new(&state)),
            None => Err(StateNotFoundError { index }),
        }
    }
}
