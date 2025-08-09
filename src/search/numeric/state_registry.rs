use crate::search::numeric::axioms::AxiomEvaluator;
use crate::search::numeric::numeric_task::AbstractNumericTask;
use crate::search::numeric::utils::errors::{StateInsertError, StateNotFoundError};
use crate::search::numeric::{
    numeric_task::{NumericRootTask, NumericType},
    utils::int_packer::IntDoublePacker,
};
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Index;

type StatePacker = IntDoublePacker;

pub struct ConcreteState<'a> {
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

impl fmt::Debug for ConcreteState<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let task = &self.state_registry.root_task;

        let num_variables = task.variables().len();
        let num_regular_numeric_vars = task
            .numeric_variables()
            .iter()
            .filter(|v| v.get_type() == &NumericType::Regular)
            .count();

        write!(f, "ConcreteState with {} bins\n", self.buffer.len())?;
        let state_packer = &self.state_registry.global_state_packer;
        for i in 0..num_variables {
            let value = state_packer.get(&self.buffer, i as i32);
            write!(f, "Var {}: {}\n", i, value)?;
        }
        for i in 0..num_regular_numeric_vars {
            let numeric_var_id = i + num_variables;
            let packed_value = state_packer.get(&self.buffer, numeric_var_id as i32);
            let numeric_value = state_packer.unpack_double(packed_value);
            write!(f, "Numeric Var {}: {}\n", numeric_var_id, numeric_value)?;
        }

        Ok(())
    }
}

impl Index<usize> for ConcreteState<'_> {
    type Output = u64;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.buffer.len(), "Index for State out of bounds");
        &self.buffer[index]
    }
}

type StateID = usize;
type DataStorage = Vec<Vec<u64>>; //TODO: Make this a vector of boxed slices for better performance

struct SemanticStateID<'a> {
    id: StateID,
    state_data_pool: &'a DataStorage,
    num_bins: usize,
}

impl<'a> PartialEq for SemanticStateID<'a> {
    fn eq(&self, other: &Self) -> bool {
        let lhs_data = &self.state_data_pool[self.id][..self.num_bins];
        let rhs_data = &other.state_data_pool[other.id][..self.num_bins];
        lhs_data == rhs_data
    }
}

impl<'a> Eq for SemanticStateID<'a> {}

impl<'a> Hash for SemanticStateID<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let bins = &self.state_data_pool[self.id][..self.num_bins];
        bins.hash(state); // slices of u64 already implement Hash
    }
}

//TODO: There should be only a single axiom evaluator so it should be fine if the StateRegistry has it
pub struct StateRegistry<'a> {
    root_task: &'a NumericRootTask,
    axiom_evaluator: &'a AxiomEvaluator<'a>,
    global_state_packer: &'a StatePacker,
    state_data_pool: Vec<Vec<u64>>, // This is a pool of state data, each state is a vector of u64
    numeric_constants: Vec<f64>,
    numeric_indices: Vec<i32>,
    registered_states: HashSet<StateID>,
}

impl<'a> StateRegistry<'a> {
    pub fn new(
        root_task: &'a NumericRootTask,
        global_state_packer: &'a StatePacker,
        axiom_evaluator: &'a AxiomEvaluator<'a>,
    ) -> Self {
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

    fn insert_id_or_pop_state(&mut self) -> StateID {
        let state_id = self.state_data_pool.len() - 1;
        if self.registered_states.contains(&state_id) {
            self.state_data_pool.remove(state_id);
        }
        if let Some(existing_id) = self.registered_states.get(&state_id) {
            return *existing_id;
        } else {
            self.registered_states.insert(state_id);
            return state_id;
        }
    }

    pub fn get_initial_state(&mut self) -> ConcreteState {
        let mut init_buffer = vec![0 as u64; self.global_state_packer.num_bins() as usize];
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

        //TODO: Figure out if we can omit clone() without using Rc and RefCell...
        //Probably this struct needs to own the task, axiom evaluator and state packer
        //The current design could lead to issues
        let mut initial_numeric_state = initial_numeric_state.clone();
        self.axiom_evaluator
            .evaluate_arithmetic_axioms(&mut initial_numeric_state)
            .unwrap();
        self.axiom_evaluator
            .evaluate(&mut init_buffer, &mut initial_numeric_state)
            .unwrap();

        self.state_data_pool.push(init_buffer);
        println!("Init buffer: {:?}", self.state_data_pool.last());
        let state_id = self.insert_id_or_pop_state();

        // TODO get rid of this clone
        let concrete_state = ConcreteState::new(self, self.state_data_pool[state_id].clone());

        concrete_state
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

#[cfg(test)]
mod tests {
    use crate::setup_axiom_evaluator;
    use crate::setup_numeric_task;
    use crate::setup_state_packer;
    use crate::setup_state_registry;

    #[test]
    fn test_state_registry_initial_state() {
        let problem = setup_numeric_task("misc/numeric_sas/example1.sas");
        let state_packer = setup_state_packer(&problem);
        let axiom_evaluator = setup_axiom_evaluator(&problem, &state_packer);
        let mut state_registry = setup_state_registry(&problem, &state_packer, &axiom_evaluator);
        let initial_state = state_registry.get_initial_state();
        print!("Initial state: {:?}", initial_state);
    }
}
