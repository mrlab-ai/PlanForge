use crate::search::numeric;
use crate::search::numeric::axioms::AxiomEvaluator;
use crate::search::numeric::numeric_task::AbstractNumericTask;
use crate::search::numeric::utils::errors::{StateInsertError, StateNotFoundError};
use crate::search::numeric::{
    numeric_task::{NumericRootTask, NumericType},
    utils::int_packer::IntDoublePacker,
};
use std::collections::HashSet;
use std::ops::Index;

type StatePacker = IntDoublePacker;

struct ConcreteState<'a> {
    state_registry: &'a StateRegistry<'a>,
    buffer: Vec<u64>,
}

impl<'a> ConcreteState<'a> {
    pub fn new(state_registry: &'a StateRegistry, buffer: Vec<u64>) -> Self {
        ConcreteState {
            state_registry,
            buffer,
        }
    }
}

impl Index<usize> for ConcreteState<'_> {
    type Output = u64;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.buffer.len(), "Index for State out of bounds");
        &self.buffer[index]
    }
}

//TODO: There should be only a single axiom evaluator so it should be fine if the StateRegistry has it
struct StateRegistry<'a> {
    root_task: &'a NumericRootTask,
    axiom_evaluator: &'a AxiomEvaluator<'a>,
    global_state_packer: StatePacker,
    state_data_pool: Vec<StatePacker>,
    numeric_constants: Vec<f64>,
    numeric_indices: Vec<i32>,
    registered_states: HashSet<usize>,
}

impl<'a> StateRegistry<'a> {
    pub fn new(root_task: &'a NumericRootTask, global_state_packer: StatePacker, axiom_evaluator: &'a AxiomEvaluator<'a>) -> Self {
        let number_numeric_vars = root_task.numeric_variables().len();
        StateRegistry {
            root_task,
            global_state_packer,
            state_data_pool: Vec::new(),
            numeric_constants: Vec::new(),
            numeric_indices: vec![-1; number_numeric_vars],
            registered_states: HashSet::new(),
            axiom_evaluator,
        }
    }

    pub fn get_initial_state(&mut self) -> ConcreteState {
        let mut init_buffer =
            vec![0 as u64; self.global_state_packer.num_bins() as usize].into_boxed_slice();
        let initial_propositional_state = self.root_task.get_initial_propositional_state_values();

        for i in 0..initial_propositional_state.len() {
            self.global_state_packer.set(
                &mut init_buffer,
                i as i32,
                initial_propositional_state[i] as u64,
            );
        }

        let mut numeric_var_index = initial_propositional_state.len();
        let mut constant_index = 0;
        let mut derived_index = 0;
        let initial_numeric_state = self.root_task.get_initial_numeric_state_values();

        let mut instrumentation_variables = vec![];

        for i in 0..initial_numeric_state.len() {
            let numeric_var = self.root_task.numeric_variables().get(i).unwrap(); //TODO: Remove unwrap
            let numeric_var_type = numeric_var.get_type();
            match numeric_var_type {
                NumericType::Instrumentation => {
                    assert!(self.numeric_indices.get(i) == Some(&-1));
                    self.numeric_indices[i] = instrumentation_variables.len() as i32;
                    instrumentation_variables.push(initial_numeric_state[i]);
                }

                NumericType::Constant => {
                    assert!(self.numeric_indices.get(i) == Some(&-1));
                    self.numeric_indices[i] = constant_index;
                    self.numeric_constants.push(initial_numeric_state[i]);
                    constant_index += 1;
                }

                NumericType::Derived => {
                    assert!(self.numeric_indices.get(i) == Some(&-1));
                    //TODO: derived index never used besides printing...
                    derived_index += 1;
                }

                NumericType::Regular => {
                    assert!(self.numeric_indices.get(i) == Some(&-1));
                    self.numeric_indices[i] = numeric_var_index as i32;
                    let packed_numeric_value = self
                        .global_state_packer
                        .pack_double(initial_numeric_state[i]);
                    self.global_state_packer.set(
                        &mut init_buffer,
                        numeric_var_index as i32,
                        packed_numeric_value,
                    );
                    numeric_var_index += 1;
                }
            }
        }

        #[cfg(debug_assertions)]
        println!(
            "Initial state: {} regular, {} constants, {} instrumentation variables, {} derived variables",
            numeric_var_index - initial_propositional_state.len(),
            constant_index,
            instrumentation_variables.len(),
            derived_index
        );

        todo!()
    }

    pub fn register_state(
        &mut self,
        values: Vec<u64>,
        numeric_values: Vec<f64>,
    ) -> Result<ConcreteState, StateInsertError> {
        let mut buffer = vec![0; self.global_state_packer.num_bins() as usize];
        for i in 0..values.len() {
            let var_id = i as i32;
            self.global_state_packer.set(&mut buffer, var_id, values[i]);
        }

        let mut regular_index = values.len() as i32;
        let mut constant_index = 0;
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
                    self.numeric_indices[i] = constant_index;
                    self.numeric_constants.push(numeric_values[i]);
                    constant_index += 1;
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

    pub fn lookup_state(&self, index: usize) -> Result<ConcreteState, StateNotFoundError> {
        todo!()
        //match self.state_data_pool.get(index) {
        //    Some(state) => Ok(ConcreteState::new(&state)),
        //    None => Err(StateNotFoundError { index }),
        //}
    }
}
