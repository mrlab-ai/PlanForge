use crate::search::numeric::{
    axioms::{AssignmentAxiom, ComparisonAxiom, PropositionalAxiom},
    state_registry::{ConcreteState, StateRegistry},
    utils::int_packer::IntDoublePacker,
};
use std::{cell::{Ref, RefCell, RefMut}, fmt, rc::Rc};

pub trait AbstractNumericTask {
    fn variables(&self) -> &Vec<ExplicitVariable>;
    fn numeric_variables(&self) -> &Vec<NumericVariable>;
    fn assignment_axioms(&self) -> &Vec<AssignmentAxiom>;
    fn comparison_axioms(&self) -> &Vec<ComparisonAxiom>;
    fn axioms(&self) -> &Vec<PropositionalAxiom>;

    fn get_num_variables(&self) -> i32;
    fn get_variable_name(&self, index: i32) -> Result<&str, &str>;
    fn get_variable_domain_size(&self, index: i32) -> Result<i32, &str>;
    fn get_variable_axiom_layer(&self, index: i32) -> Result<i32, &str>;
    fn get_variable_default_axiom_value(&self, index: i32) -> Result<i32, &str>;
    fn get_fact_name(&self, fact: &Fact) -> &str;

    fn are_facts_mutex(&self, fact1: &Fact, fact2: &Fact) -> bool;

    fn get_operators(&self) -> &Vec<Operator>;
    fn get_operator_cost(&self, index: i32, is_axiom: bool) -> i32;
    fn get_operator_name(&self, index: i32, is_axiom: bool) -> &str;
    fn get_num_operators(&self) -> i32;
    fn get_num_operator_preconditions(&self, index: i32, is_axiom: bool) -> i32;
    fn get_operator_precondition(&self, index: i32, precond_index: i32, is_axiom: bool) -> &Fact;
    fn get_num_operator_effects(&self, index: i32, is_axiom: bool) -> i32;
    fn get_num_operator_effect_conditions(&self, index: i32, eff_index: i32, is_axiom: bool)
        -> i32;
    fn get_operator_effect_condition(
        &self,
        index: i32,
        eff_index: i32,
        cond_index: i32,
        is_axiom: bool,
    ) -> &Fact;
    fn get_operator_effect(&self, index: i32, eff_index: i32, is_axiom: bool) -> &Fact;

    fn convert_operator_index(&self, index: i32, ancestor_task: &dyn AbstractNumericTask);

    fn get_num_axioms(&self) -> i32;
    fn get_num_goals(&self) -> i32;
    fn get_goal_fact(&self, index: i32) -> &Fact;

    fn get_initial_propositional_state_values(&self) -> Ref<Vec<i32>>;
    fn get_initial_numeric_state_values(&self) -> Ref<Vec<f64>>;

    fn get_initial_propositional_state_values_mut(&self) -> RefMut<Vec<i32>>;
    fn get_initial_numeric_state_values_mut(&self) -> RefMut<Vec<f64>>;

    fn set_initial_numeric_state_values(&self, values: Vec<f64>);
    fn set_initial_propositional_state_values(&self, values: Vec<i32>);

    fn convert_ancestor_state_values(
        &self,
        ancestor_state_values: &Vec<i32>,
        ancestor_task: &dyn AbstractNumericTask,
    ) -> Vec<i32>;

    fn get_num_cmp_axioms(&self) -> i32;
}


#[derive(Debug)]
pub struct Metric {
    is_min: bool,
    var_id: i32,
}

impl Metric {
    pub fn new(is_min: bool, var_id: i32) -> Self {
        Metric { is_min, var_id }
    }

    pub fn is_min(&self) -> bool {
        self.is_min
    }

    pub fn var_id(&self) -> i32 {
        self.var_id
    }

    pub fn use_metric(&self) -> bool {
        self.var_id >= 0
    }
}

#[derive(Debug)]
pub struct ExplicitVariable {
    domain_size: u32,
    name: String,
    fact_names: Vec<String>,
    axiom_layer: i32,
    axiom_default_value: u32, //Is this field even required?
}

impl ExplicitVariable {
    pub fn new(
        domain_size: u32,
        name: String,
        fact_names: Vec<String>,
        axiom_layer: i32,
        axiom_default_value: u32,
    ) -> Self {
        ExplicitVariable {
            domain_size,
            name,
            fact_names,
            axiom_layer,
            axiom_default_value,
        }
    }

    pub fn axiom_layer(&self) -> i32 {
        self.axiom_layer
    }

    pub fn domain_size(&self) -> u32 {
        self.domain_size
    }
}

#[derive(Debug)]
pub struct NumericVariable {
    name: String,
    numeric_type: NumericType,
    axiom_layer: i32,
}

impl NumericVariable {
    pub fn new(name: String, numeric_type: NumericType, axiom_layer: i32) -> Self {
        NumericVariable {
            name,
            numeric_type,
            axiom_layer,
        }
    }

    pub fn get_type(&self) -> &NumericType {
        &self.numeric_type
    }

    pub fn axiom_layer(&self) -> i32 {
        self.axiom_layer
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct Fact {
    var: u32,
    value: i32,
}

impl Fact {
    pub fn new(var: u32, value: i32) -> Self {
        Fact { var, value }
    }

    pub fn var(&self) -> u32 {
        self.var
    }

    pub fn value(&self) -> i32 {
        self.value
    }

    pub fn is_true(&self, state: &ConcreteState, state_registry: &StateRegistry) -> bool {
        let buffer = state.buffer(state_registry);
        let state_packer = state_registry.global_state_packer();
        let value = state_packer.get(buffer, self.var as i32);
        value == self.value as u64
    }
}

impl fmt::Debug for Fact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fact(var: {}, value: {})", self.var, self.value)
    }
}

#[derive(Debug)]
pub struct Effect {
    conditions: Vec<Fact>,
    var_id: u32,
    precondition_value: i32,
    effect_value: u32,
}

impl Effect {
    pub fn new(
        conditions: Vec<Fact>,
        var_id: u32,
        precondition_value: i32,
        effect_value: u32,
    ) -> Self {
        Effect {
            conditions,
            var_id,
            precondition_value,
            effect_value,
        }
    }

    pub fn var_id(&self) -> u32 {
        self.var_id
    }

    pub fn value(&self) -> u32 {
        self.effect_value
    }

    pub fn conditions_met(&self, state: &ConcreteState, state_registry: &StateRegistry) -> bool {
        for condition in &self.conditions {
            if !condition.is_true(&state, state_registry) {
                return false;
            }
        }
        true
    }
}

#[derive(Debug)]
pub enum AssignmentOperation {
    Assign,
    Plus,
    Minus,
    Times,
    Divide,
}

impl AssignmentOperation {
    pub fn apply(left: f64, operation: &AssignmentOperation, right: f64) -> f64 {
        match operation {
            AssignmentOperation::Assign => right,
            AssignmentOperation::Plus => left + right,
            AssignmentOperation::Minus => left - right,
            AssignmentOperation::Times => left * right,
            AssignmentOperation::Divide => {
                if right == 0.0 {
                    panic!("Division by zero is not allowed");
                }
                left / right
            }
        }
    }
}

#[derive(Debug)]
pub struct AssignmentEffect {
    affected_var_id: u32,
    operation: AssignmentOperation,
    var_id: u32,
    is_conditional: bool,
    conditions: Vec<Fact>,
}

impl AssignmentEffect {
    pub fn new(
        affected_var_id: u32,
        operation: AssignmentOperation,
        var_id: u32,
        is_conditional: bool,
        conditions: Vec<Fact>,
    ) -> Self {
        AssignmentEffect {
            affected_var_id,
            operation,
            var_id,
            is_conditional,
            conditions,
        }
    }

    pub fn affected_var_id(&self) -> u32 {
        self.affected_var_id
    }
    pub fn var_id(&self) -> u32 {
        self.var_id
    }

    pub fn operation(&self) -> &AssignmentOperation {
        &self.operation
    }
}

#[derive(Debug)]
pub struct Operator {
    name: String,
    preconditions: Vec<Fact>,
    effects: Vec<Effect>,
    assignment_effects: Vec<AssignmentEffect>,
    cost: u32,
}

impl Operator {
    pub fn new(
        name: String,
        preconditions: Vec<Fact>,
        effects: Vec<Effect>,
        assignment_effects: Vec<AssignmentEffect>,
        cost: u32,
    ) -> Self {
        Operator {
            name,
            preconditions,
            effects,
            assignment_effects,
            cost,
        }
    }

    pub fn conditions_met(&self, state: &Vec<&Fact>) -> bool {
        for precondition in &self.preconditions {
            if !state.iter().any(|fact| {
                fact.var() == precondition.var() && fact.value() == precondition.value()
            }) {
                return false;
            }
        }
        true
    }

    pub fn effects(&self) -> &Vec<Effect> {
        &self.effects
    }

    pub fn assignment_effects(&self) -> &Vec<AssignmentEffect> {
        &self.assignment_effects
    }

    pub fn preconditions(&self) -> &Vec<Fact> {
        &self.preconditions
    }
}

#[derive(Debug)]
pub struct NumericRootTask {
    version: u32,
    metric: Metric,
    variables: Vec<ExplicitVariable>,
    numeric_variables: Vec<NumericVariable>,
    goals: Vec<Fact>,
    mutexes: Vec<Vec<Fact>>,
    state: Rc<RefCell<Vec<i32>>>,
    numeric_state: Rc<RefCell<Vec<f64>>>,
    operators: Vec<Operator>,
    axioms: Vec<PropositionalAxiom>,
    comparison_axioms: Vec<ComparisonAxiom>,
    assignment_axioms: Vec<AssignmentAxiom>,
    global_constraint: (u32, u32),
}

impl NumericRootTask {
    pub fn new(
        version: u32,
        metric: Metric,
        variables: Vec<ExplicitVariable>,
        numeric_variables: Vec<NumericVariable>,
        goals: Vec<Fact>,
        mutexes: Vec<Vec<Fact>>,
        state: Vec<i32>,
        numeric_state: Vec<f64>,
        operators: Vec<Operator>,
        axioms: Vec<PropositionalAxiom>,
        comparison_axioms: Vec<ComparisonAxiom>,
        assignment_axioms: Vec<AssignmentAxiom>,
        global_constraint: (u32, u32),
    ) -> Self {
        NumericRootTask {
            version,
            metric,
            variables,
            numeric_variables,
            goals,
            mutexes,
            state: Rc::new(RefCell::new(state)),
            numeric_state: Rc::new(RefCell::new(numeric_state)),
            operators,
            axioms,
            comparison_axioms,
            assignment_axioms,
            global_constraint,
        }
    }

    /// Returns a reference to the metric configuration
    pub fn metric(&self) -> &Metric {
        &self.metric
    }
}

#[derive(Debug, PartialEq)]
pub enum NumericType {
    Constant,
    Derived,
    Cost,
    Regular, // not sure if Root is correct
}

impl AbstractNumericTask for NumericRootTask {
    fn variables(&self) -> &Vec<ExplicitVariable> {
        &self.variables
    }

    fn numeric_variables(&self) -> &Vec<NumericVariable> {
        &self.numeric_variables
    }

    fn assignment_axioms(&self) -> &Vec<AssignmentAxiom> {
        &self.assignment_axioms
    }

    fn comparison_axioms(&self) -> &Vec<ComparisonAxiom> {
        &self.comparison_axioms
    }

    fn get_operators(&self) -> &Vec<Operator> {
        &self.operators
    }

    fn axioms(&self) -> &Vec<PropositionalAxiom> {
        &self.axioms
    }

    fn get_num_variables(&self) -> i32 {
        self.variables.len() as i32
    }

    fn get_variable_name(&self, index: i32) -> Result<&str, &str> {
        if index < 0 || index >= (self.variables.len() as i32) {
            return Err("Index out of bounds");
        }
        Ok(&self.variables[index as usize].name)
    }

    fn get_variable_domain_size(&self, index: i32) -> Result<i32, &str> {
        if index < 0 || index >= (self.variables.len() as i32) {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index as usize].domain_size as i32)
    }

    fn get_variable_axiom_layer(&self, index: i32) -> Result<i32, &str> {
        if index < 0 || index >= (self.variables.len() as i32) {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index as usize].axiom_layer)
    }

    fn get_variable_default_axiom_value(&self, index: i32) -> Result<i32, &str> {
        if index < 0 || index >= (self.variables.len() as i32) {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index as usize].axiom_default_value as i32)
    }

    fn get_fact_name(&self, fact: &Fact) -> &str {
        ""
    }

    fn are_facts_mutex(&self, fact1: &Fact, fact2: &Fact) -> bool {
        false
    }

    fn get_operator_cost(&self, index: i32, is_axiom: bool) -> i32 {
        0
    }

    fn get_operator_name(&self, index: i32, is_axiom: bool) -> &str {
        ""
    }

    fn get_num_operators(&self) -> i32 {
        self.operators.len() as i32
    }

    fn get_num_operator_preconditions(&self, index: i32, is_axiom: bool) -> i32 {
        if is_axiom {
            // Axioms don't have preconditions in the same way
            return 0;
        }
        if index < 0 || index >= self.operators.len() as i32 {
            return 0;
        }
        self.operators[index as usize].preconditions().len() as i32
    }

    fn get_operator_precondition(&self, index: i32, precond_index: i32, is_axiom: bool) -> &Fact {
        unimplemented!("This function is not yet implemented");
    }

    fn get_num_operator_effects(&self, index: i32, is_axiom: bool) -> i32 {
        if is_axiom {
            // Handle axiom effects differently
            return 0;
        }
        if index < 0 || index >= self.operators.len() as i32 {
            return 0;
        }
        self.operators[index as usize].effects().len() as i32
    }

    fn get_num_operator_effect_conditions(
        &self,
        index: i32,
        eff_index: i32,
        is_axiom: bool,
    ) -> i32 {
        0
    }

    fn get_operator_effect_condition(
        &self,
        index: i32,
        eff_index: i32,
        cond_index: i32,
        is_axiom: bool,
    ) -> &Fact {
        unimplemented!("This function is not yet implemented");
    }

    fn get_operator_effect(&self, index: i32, eff_index: i32, is_axiom: bool) -> &Fact {
        unimplemented!("This function is not yet implemented");
    }

    fn convert_operator_index(&self, index: i32, ancestor_task: &dyn AbstractNumericTask) {}

    fn get_num_axioms(&self) -> i32 {
        self.axioms.len() as i32
    }

    fn get_num_goals(&self) -> i32 {
        self.goals.len() as i32
    }

    fn get_goal_fact(&self, index: i32) -> &Fact {
        if index < 0 || index >= self.goals.len() as i32 {
            panic!("Goal index {} out of bounds", index);
        }
        &self.goals[index as usize]
    }

    fn get_initial_propositional_state_values(&self) -> Ref<Vec<i32>> {
        self.state.borrow()
    }

    fn get_initial_numeric_state_values(&self) -> Ref<Vec<f64>> {
        self.numeric_state.borrow()
    }

    fn get_initial_propositional_state_values_mut(&self) -> RefMut<Vec<i32>> {
        self.state.borrow_mut()
    }

    fn get_initial_numeric_state_values_mut(&self) -> RefMut<Vec<f64>> {
        self.numeric_state.borrow_mut()
    }

    fn set_initial_numeric_state_values(&self, values: Vec<f64>) {
        *self.numeric_state.borrow_mut() = values;
    }

    fn set_initial_propositional_state_values(&self, values: Vec<i32>) {
        *self.state.borrow_mut() = values;
    }

    fn convert_ancestor_state_values(
        &self,
        ancestor_state_values: &Vec<i32>,
        ancestor_task: &dyn AbstractNumericTask,
    ) -> Vec<i32> {
        vec![]
    }

    fn get_num_cmp_axioms(&self) -> i32 {
        self.comparison_axioms.len() as i32
    }
}
