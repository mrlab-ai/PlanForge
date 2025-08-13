use crate::search::numeric::axioms::AxiomEvaluator;
use crate::search::numeric::numeric_task::{
    AbstractNumericTask, AssignmentOperation, Fact, Operator,
};
use crate::search::numeric::utils::errors::{InvalidIndex, StateInsertError, StateNotFoundError};
use crate::search::numeric::{
    numeric_task::{NumericRootTask, NumericType},
    utils::int_packer::IntDoublePacker,
};
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Index;
use std::thread::current;

type StatePacker = IntDoublePacker;

pub struct ConcreteState {
    pool_offset: usize,
}

impl ConcreteState {
    pub fn new(pool_offset: usize) -> Self {
        ConcreteState { pool_offset }
    }

    pub fn get_state(&self, state_registry: &StateRegistry) -> Vec<i32> {
        let buffer = state_registry.get_buffer(self.pool_offset);
        let mut facts = vec![];
        let task = state_registry.root_task;
        let state_packer = state_registry.global_state_packer;
        for i in 0..task.variables().len() {
            let value = state_packer.get(buffer, i as i32);
            facts.push(value as i32);
        }
        facts
    }

    pub fn buffer<'a>(&self, state_registry: &'a StateRegistry) -> &'a Vec<u64> {
        state_registry.get_buffer(self.pool_offset)
    }

    pub fn len(&self, state_registry: &StateRegistry) -> usize {
        state_registry.root_task.variables().len()
    }

    pub fn debug_with_registry(&self, registry: &StateRegistry) -> String {
        let task = &registry.root_task;
        let num_variables = task.variables().len();
        let num_regular_numeric_vars = task
            .numeric_variables()
            .iter()
            .filter(|v| v.get_type() == &NumericType::Regular)
            .count();

        let buffer = self.buffer(registry);

        let mut s = format!("ConcreteState with {} bins\n", buffer.len());
        let state_packer = &registry.global_state_packer;
        for i in 0..num_variables {
            let value = state_packer.get(buffer, i as i32);
            s += &format!("Var {}: {}\n", i, value);
        }
        for i in 0..num_regular_numeric_vars {
            let numeric_var_id = i + num_variables;
            let packed_value = state_packer.get(buffer, numeric_var_id as i32);
            let numeric_value = state_packer.unpack_double(packed_value);
            s += &format!("Numeric Var {}: {}\n", numeric_var_id, numeric_value);
        }
        s
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

    pub fn get_buffer(&self, index: usize) -> &Vec<u64> {
        self.state_data_pool
            .get(index)
            .expect("State index out of bounds")
    }

    pub fn global_state_packer(&self) -> &StatePacker {
        self.global_state_packer
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
                NumericType::Cost => {
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

        ConcreteState {
            pool_offset: state_id,
        }
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

        let mut cost_variables = vec![];

        for i in 0..numeric_values.len() {
            let numeric_variable =
                self.root_task
                    .numeric_variables()
                    .get(i)
                    .ok_or_else(|| StateInsertError {
                        message: format!("Numeric variable at index {} not found", i),
                    })?;
            match numeric_variable.get_type() {
                NumericType::Cost => {
                    assert!(self.numeric_indices.get(i) == Some(&-1));
                    self.numeric_indices[i] = cost_variables.len() as i32;
                    cost_variables.push(numeric_values[i]);
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
                            "Only regular and cost variables are allowed here: {:?}",
                            numeric_variable.get_type()
                        ),
                    });
                }
            }
        }

        self.axiom_evaluator
            .evaluate_arithmetic_axioms(&mut numeric_values.clone())
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate arithmetic axioms: {:?}", e),
            })?;
        self.axiom_evaluator
            .evaluate(&mut buffer, &mut numeric_values.clone())
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate axioms: {:?}", e),
            })?;

        self.state_data_pool.push(buffer);
        self.insert_id_or_pop_state();

        // TODO get rid of this clone
        let new_state = ConcreteState::new(self.state_data_pool.len() - 1);

        //TODO: Add cost information
        Ok(new_state)
    }

    pub fn lookup_state(&self, index: usize) -> Result<ConcreteState, StateNotFoundError> {
        if index >= self.state_data_pool.len() {
            return Err(StateNotFoundError { index: index });
        }
        let state_data = &self.state_data_pool[index];
        Ok(ConcreteState::new(index))
    }

    pub fn get_successor_state(
        &mut self,
        current_state: &ConcreteState,
        operator: &Operator,
    ) -> Result<ConcreteState, StateInsertError> {
        let mut buffer = current_state.buffer(&self).clone();
        for eff in operator.effects().iter() {
            let var_id = eff.var_id() as i32;
            let value = eff.value() as u64;
            if eff.conditions_met(&current_state, &self) {
                self.global_state_packer.set(&mut buffer, var_id, value);
            }
        }

        //TODO: Add cost here
        let mut successor_values = self.get_numeric_vars(current_state).unwrap();
        self.get_numeric_successor2(
            &mut successor_values,
            operator,
            &mut buffer,
            &mut current_state.buffer(&self),
        )?;

        self.state_data_pool.push(buffer); //TODO: Figure out how to initialize that in the vector.

        let id = self.insert_id_or_pop_state();
        let successor = self.lookup_state(id).unwrap();

        if id == self.state_data_pool.len() - 1 {
            //TODO: Update cost here
        } else {
            //TODO: cost
        }

        Ok(successor)
    }

    fn get_numeric_vars(&self, state: &ConcreteState) -> Result<Vec<f64>, InvalidIndex> {
        let mut result = vec![0.0; self.root_task.numeric_variables().len()];
        let cost_variables: Option<i32> = None; //TODO: Add cost variables handling

        let buffer = state.buffer(&self);
        for i in 0..self.root_task.numeric_variables().len() {
            let numeric_var = self.root_task.numeric_variables().get(i).unwrap();
            match numeric_var.get_type() {
                NumericType::Cost => {
                    //result[i] = cost_variables
                }
                NumericType::Constant => {
                    result[i] = self.numeric_constants[self.numeric_indices[i] as usize];
                }
                NumericType::Regular => {
                    result[i] = self
                        .global_state_packer
                        .get_double(buffer, self.numeric_indices[i]);
                }
                _ => {}
            }
        }
        //TODO: Change initial state once constructed.
        //debug_assert!(
        //    result.len() == self.root_task.numeric_variables().len(),
        //    "Numeric variables length mismatch"
        //);
        if self.axiom_evaluator.has_numeric_axioms() {
            self.axiom_evaluator
                .evaluate_arithmetic_axioms(&mut result)?;
        }

        Ok(result)
    }

    fn get_numeric_successor(
        &self,
        current_values: &mut Vec<f64>,
        operator: &Operator,
    ) -> Result<(), StateInsertError> {
        let values_before_changing = current_values.clone(); //TODO: Can I get rid of this clone?
        for effect in operator.assignment_effects().iter() {
            let var_id = effect.var_id() as usize;
            debug_assert!(
                effect.var_id() < current_values.len() as u32,
                "Effect variable ID out of bounds"
            );
            let numeric_var = self.root_task.numeric_variables().get(var_id).unwrap();

            let mut assignment_value = current_values[var_id];
            if numeric_var.get_type() == &NumericType::Regular {
                assignment_value = values_before_changing[var_id];
            }

            let result = AssignmentOperation::apply(
                current_values[effect.affected_var_id() as usize],
                effect.operation(),
                assignment_value,
            );

            match numeric_var.get_type() {
                NumericType::Cost => {
                    //TODO: Add instrumentation handling
                    current_values[effect.affected_var_id() as usize] = result;
                }
                NumericType::Regular => {
                    current_values[effect.affected_var_id() as usize] = result;
                }
                _ => {
                    return Err(StateInsertError {
                        message: format!(
                            "Only regular and cost variables are allowed here: {:?}",
                            numeric_var.get_type()
                        ),
                    });
                }
            }
        }
        self.axiom_evaluator
            .evaluate_arithmetic_axioms(current_values)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate arithmetic axioms: {:?}", e),
            })?;
        Ok(())
    }

    fn get_numeric_successor2(
        &self,
        current_values: &mut Vec<f64>,
        operator: &Operator,
        next_buffer: &mut Vec<u64>,
        current_buffer: &Vec<u64>,
    ) -> Result<(), StateInsertError> {
        for effect in operator.assignment_effects().iter() {
            let assignment_var_id = effect.var_id() as usize;
            let affected_var_id = effect.affected_var_id() as usize;
            debug_assert!(
                effect.var_id() < current_values.len() as u32,
                "Effect variable ID out of bounds"
            );

            let assignment_var = self
                .root_task
                .numeric_variables()
                .get(assignment_var_id)
                .unwrap();
            let mut assignment_value = current_values[assignment_var_id];

            let affected_var = self
                .root_task
                .numeric_variables()
                .get(affected_var_id as usize)
                .unwrap();

            if assignment_var.get_type() == &NumericType::Regular {
                assignment_value = self
                    .global_state_packer
                    .get_double(current_buffer, assignment_var_id as i32);
            }

            let result = AssignmentOperation::apply(
                current_values[affected_var_id as usize],
                effect.operation(),
                assignment_value,
            );

            match affected_var.get_type() {
                NumericType::Cost => {
                    //TODO: Add instrumentation handling
                    current_values[effect.affected_var_id() as usize] = result;
                }
                NumericType::Regular => {
                    self.global_state_packer.set(
                        next_buffer,
                        effect.affected_var_id() as i32,
                        self.global_state_packer.pack_double(result),
                    );
                    current_values[effect.affected_var_id() as usize] = result;
                }
                _ => {
                    return Err(StateInsertError {
                        message: format!(
                            "!!!!!Only regular and cost variables are allowed here: {:?}",
                            affected_var.get_type()
                        ),
                    });
                }
            }
        }
        self.axiom_evaluator
            .evaluate_arithmetic_axioms(current_values)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate arithmetic axioms: {:?}", e),
            })?;

        self.axiom_evaluator
            .evaluate(next_buffer, current_values)
            .map_err(|e| StateInsertError {
                message: format!("Failed to evaluate axioms: {:?}", e),
            })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crate::search::numeric::numeric_task::Fact;
    use crate::search::numeric::numeric_task::Operator;
    use crate::setup_axiom_evaluator;
    use crate::setup_numeric_task;
    use crate::setup_state_packer;
    use crate::setup_state_registry;
    use crate::setup_successor_generator;

    #[test]
    fn test_state_registry_initial_state() {
        let problem = setup_numeric_task("misc/numeric_sas/example1.sas");
        let state_packer = setup_state_packer(&problem);
        let axiom_evaluator = setup_axiom_evaluator(&problem, &state_packer);
        let mut state_registry = setup_state_registry(&problem, &state_packer, &axiom_evaluator);
        let initial_state = state_registry.get_initial_state();
        print!(
            "Initial state: {:?}",
            initial_state.debug_with_registry(&state_registry)
        );
    }

    #[test]
    fn test_generate_immediate_successor_of_init_state() {
        let task = setup_numeric_task("misc/numeric_sas/example1.sas");
        let state_packer = setup_state_packer(&task);
        let axiom_evaluator = setup_axiom_evaluator(&task, &state_packer);
        let mut state_registry = setup_state_registry(&task, &state_packer, &axiom_evaluator);
        let initial_state = state_registry.get_initial_state();

        let state = initial_state.get_state(&state_registry);
        let facts = state
            .iter()
            .enumerate()
            .map(|(i, value)| Fact::new(i as u32, *value as i32))
            .collect::<Vec<_>>();
        let mut facts_refs = Vec::new();

        for fact in &facts {
            facts_refs.push(fact);
        }

        let suc_gen = setup_successor_generator(&task);

        let mut applicable_operators = VecDeque::new();
        suc_gen.get_applicable_operators(&facts_refs, &mut applicable_operators);

        let op = applicable_operators.pop_front().unwrap();

        println!(
            "Initial state: {}",
            initial_state.debug_with_registry(&state_registry)
        );
        println!("OP: {:?}", op);

        let successor = state_registry
            .get_successor_state(&initial_state, op)
            .expect("Failed to get successor state");

        println!(
            "Successor state: {}",
            successor.debug_with_registry(&state_registry)
        );
    }
}
