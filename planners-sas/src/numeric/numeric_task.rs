use crate::numeric::axioms::{AssignmentAxiom, ComparisonAxiom, PropositionalAxiom};
use crate::numeric::numeric_parser::parse_numeric_sas_output;
use crate::numeric::state_registry::{ConcreteState, StateRegistry};
use std::{
    cell::{Ref, RefCell, RefMut},
    fmt,
    rc::Rc,
};

pub trait AbstractNumericTask {
    fn variables(&self) -> &Vec<ExplicitVariable>;
    fn numeric_variables(&self) -> &Vec<NumericVariable>;
    fn assignment_axioms(&self) -> &Vec<AssignmentAxiom>;
    fn comparison_axioms(&self) -> &Vec<ComparisonAxiom>;
    fn axioms(&self) -> &Vec<PropositionalAxiom>;
    fn metric(&self) -> &Metric;

    fn get_num_variables(&self) -> usize;
    fn get_variable_name(&self, index: usize) -> Result<&str, &str>;
    fn get_variable_domain_size(&self, index: usize) -> Result<usize, &str>;
    fn get_variable_axiom_layer(&self, index: usize) -> Result<Option<usize>, &str>;
    fn get_variable_default_axiom_value(&self, index: usize) -> Result<usize, &str>;
    fn get_fact_name(&self, fact: &ExplicitFact) -> &str;

    fn are_facts_mutex(&self, fact1: &ExplicitFact, fact2: &ExplicitFact) -> bool;

    fn get_operators(&self) -> &Vec<Operator>;
    fn get_operator_cost(&self, index: usize, is_axiom: bool) -> u64;
    fn get_operator_name(&self, index: usize, is_axiom: bool) -> &str;
    fn get_num_operators(&self) -> usize;
    fn get_num_operator_preconditions(&self, index: usize, is_axiom: bool) -> usize;
    fn get_operator_precondition(
        &self,
        index: usize,
        precond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact;
    fn get_num_operator_effects(&self, index: usize, is_axiom: bool) -> usize;
    fn get_num_operator_effect_conditions(
        &self,
        index: usize,
        eff_index: usize,
        is_axiom: bool,
    ) -> usize;
    fn get_operator_effect_condition(
        &self,
        index: usize,
        eff_index: usize,
        cond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact;
    fn get_operator_effect(&self, index: usize, eff_index: usize, is_axiom: bool) -> &ExplicitFact;

    fn convert_operator_index(&self, index: usize, ancestor_task: &dyn AbstractNumericTask);

    fn get_num_axioms(&self) -> usize;
    fn get_num_goals(&self) -> usize;
    fn get_goal_fact(&self, index: usize) -> &ExplicitFact;

    fn get_initial_propositional_state_values(&self) -> Ref<'_, Vec<usize>>;
    fn get_initial_numeric_state_values(&self) -> Ref<'_, Vec<f64>>;

    fn get_initial_propositional_state_values_mut(&self) -> RefMut<'_, Vec<usize>>;
    fn get_initial_numeric_state_values_mut(&self) -> RefMut<'_, Vec<f64>>;

    fn set_initial_propositional_state_values(&self, values: Vec<usize>);
    fn set_initial_numeric_state_values(&self, values: Vec<f64>);

    fn convert_ancestor_state_values(
        &self,
        ancestor_state_values: &[usize],
        ancestor_task: &dyn AbstractNumericTask,
    ) -> Vec<usize>;

    fn get_num_cmp_axioms(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct Metric {
    is_min: bool,
    var_id: Option<usize>,
}

impl Metric {
    pub fn new(is_min: bool, var_id: Option<usize>) -> Self {
        Metric { is_min, var_id }
    }

    pub fn is_min(&self) -> bool {
        self.is_min
    }

    pub fn var_id(&self) -> Option<usize> {
        self.var_id
    }

    pub fn use_metric(&self) -> bool {
        self.var_id.is_some()
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct ExplicitVariable {
    domain_size: usize,
    name: String,
    fact_names: Vec<String>,
    axiom_layer: Option<usize>,
    axiom_default_value: usize, // Is this field even required?
}

impl ExplicitVariable {
    pub fn new(
        domain_size: usize,
        name: String,
        fact_names: Vec<String>,
        axiom_layer: Option<usize>,
        axiom_default_value: usize,
    ) -> Self {
        ExplicitVariable {
            domain_size,
            name,
            fact_names,
            axiom_layer,
            axiom_default_value,
        }
    }

    pub fn axiom_layer(&self) -> Option<usize> {
        self.axiom_layer
    }

    pub fn domain_size(&self) -> usize {
        self.domain_size
    }
}

#[derive(Debug, Clone)]
pub struct NumericVariable {
    name: String,
    numeric_type: NumericType,
    axiom_layer: Option<usize>,
}

impl NumericVariable {
    pub fn new(name: String, numeric_type: NumericType, axiom_layer: Option<usize>) -> Self {
        NumericVariable {
            name,
            numeric_type,
            axiom_layer,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn get_type(&self) -> &NumericType {
        &self.numeric_type
    }

    pub fn axiom_layer(&self) -> Option<usize> {
        self.axiom_layer
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
pub struct ExplicitFact {
    pub var: usize,
    pub value: usize,
}

impl ExplicitFact {
    pub fn new(var: usize, value: usize) -> Self {
        ExplicitFact { var, value }
    }
    pub fn is_hold(&self, state: &ConcreteState, state_registry: &StateRegistry) -> bool {
        let buffer = state.buffer(state_registry);
        let state_packer = state_registry.global_state_packer();
        let value = state_packer.get(buffer, self.var);
        value == self.value as u64
    }
}

impl fmt::Debug for ExplicitFact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fact(var: {}, value: {})", self.var, self.value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Effect {
    conditions: Vec<ExplicitFact>,
    var_id: usize,
    precondition_value: Option<usize>,
    effect_value: usize,
}

impl Effect {
    pub fn new(
        conditions: Vec<ExplicitFact>,
        var_id: usize,
        precondition_value: Option<usize>,
        effect_value: usize,
    ) -> Self {
        Effect {
            conditions,
            var_id,
            precondition_value,
            effect_value,
        }
    }

    pub fn var_id(&self) -> usize {
        self.var_id
    }

    pub fn precondition_value(&self) -> Option<usize> {
        self.precondition_value
    }

    pub fn conditions(&self) -> &Vec<ExplicitFact> {
        &self.conditions
    }

    pub fn value(&self) -> usize {
        self.effect_value
    }

    pub fn conditions_met(&self, state: &ConcreteState, state_registry: &StateRegistry) -> bool {
        for condition in &self.conditions {
            if !condition.is_hold(state, state_registry) {
                return false;
            }
        }
        true
    }
}

#[derive(Debug, Clone, PartialEq)]
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

pub fn evaluate_metric_from_values(task: &dyn AbstractNumericTask, numeric_values: &[f64]) -> f64 {
    let metric_var_id = task.metric().var_id();
    match metric_var_id {
        Some(var_id) => numeric_values.get(var_id).copied().unwrap_or(0.0),
        None => 0.0,
    }
}

pub fn propagate_assignment_axiom_values(
    task: &dyn AbstractNumericTask,
    numeric_values: &mut [f64],
) {
    let mut changed = true;
    while changed {
        changed = false;
        for axiom in task.assignment_axioms() {
            let affected_var_id = axiom.get_affected_var_id();
            if affected_var_id >= numeric_values.len() {
                continue;
            }

            let previous_value = numeric_values[affected_var_id];
            let Ok(updated_value) = axiom.update_values(numeric_values) else {
                continue;
            };
            if updated_value != previous_value {
                changed = true;
            }
        }
    }
}

pub fn metric_operator_cost_from_initial_values(
    task: &dyn AbstractNumericTask,
    operator: &Operator,
) -> f64 {
    if !task.metric().use_metric() {
        return operator.cost() as f64;
    }

    let initial_numeric_values = task.get_initial_numeric_state_values();
    let mut numeric_values = initial_numeric_values.to_vec();
    let old_metric = evaluate_metric_from_values(task, &numeric_values);

    for effect in operator.assignment_effects() {
        let assignment_var_id = effect.var_id();
        let affected_var_id = effect.affected_var_id();
        if assignment_var_id >= numeric_values.len() || affected_var_id >= numeric_values.len() {
            continue;
        }

        let assignment_value = numeric_values[assignment_var_id];
        let result = AssignmentOperation::apply(
            numeric_values[affected_var_id],
            effect.operation(),
            assignment_value,
        );
        numeric_values[affected_var_id] = result;
    }

    propagate_assignment_axiom_values(task, &mut numeric_values);
    let new_metric = evaluate_metric_from_values(task, &numeric_values);
    let delta = if task.metric().is_min() {
        new_metric - old_metric
    } else {
        old_metric - new_metric
    };
    delta.max(0.0)
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssignmentEffect {
    affected_var_id: usize,
    operation: AssignmentOperation,
    var_id: usize,
    is_conditional: bool,
    conditions: Vec<ExplicitFact>,
}

impl AssignmentEffect {
    pub fn new(
        affected_var_id: usize,
        operation: AssignmentOperation,
        var_id: usize,
        is_conditional: bool,
        conditions: Vec<ExplicitFact>,
    ) -> Self {
        AssignmentEffect {
            affected_var_id,
            operation,
            var_id,
            is_conditional,
            conditions,
        }
    }

    pub fn affected_var_id(&self) -> usize {
        self.affected_var_id
    }
    pub fn var_id(&self) -> usize {
        self.var_id
    }

    pub fn operation(&self) -> &AssignmentOperation {
        &self.operation
    }

    pub fn is_conditional(&self) -> bool {
        self.is_conditional
    }

    pub fn conditions(&self) -> &Vec<ExplicitFact> {
        &self.conditions
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Operator {
    name: String,
    preconditions: Vec<ExplicitFact>,
    effects: Vec<Effect>,
    assignment_effects: Vec<AssignmentEffect>,
    cost: u64,
}

impl Operator {
    pub fn new(
        name: String,
        preconditions: Vec<ExplicitFact>,
        effects: Vec<Effect>,
        assignment_effects: Vec<AssignmentEffect>,
        cost: u64,
    ) -> Self {
        Operator {
            name,
            preconditions,
            effects,
            assignment_effects,
            cost,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn conditions_met(&self, state: &Vec<&ExplicitFact>) -> bool {
        for precondition in &self.preconditions {
            if !state
                .iter()
                .any(|fact| fact.var == precondition.var && fact.value == precondition.value)
            {
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

    pub fn preconditions(&self) -> &Vec<ExplicitFact> {
        &self.preconditions
    }

    pub fn cost(&self) -> u64 {
        self.cost
    }
}

#[allow(unused)]
#[derive(Debug)]
pub struct NumericRootTask {
    version: u32,
    metric: Metric,
    variables: Vec<ExplicitVariable>,
    numeric_variables: Vec<NumericVariable>,
    goals: Vec<ExplicitFact>,
    mutexes: Vec<Vec<ExplicitFact>>,
    state: Rc<RefCell<Vec<usize>>>,
    numeric_state: Rc<RefCell<Vec<f64>>>,
    operators: Vec<Operator>,
    axioms: Vec<PropositionalAxiom>,
    comparison_axioms: Vec<ComparisonAxiom>,
    assignment_axioms: Vec<AssignmentAxiom>,
    global_constraint: ExplicitFact,
}

impl NumericRootTask {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        version: u32,
        metric: Metric,
        variables: Vec<ExplicitVariable>,
        numeric_variables: Vec<NumericVariable>,
        goals: Vec<ExplicitFact>,
        mutexes: Vec<Vec<ExplicitFact>>,
        state: Vec<usize>,
        numeric_state: Vec<f64>,
        operators: Vec<Operator>,
        axioms: Vec<PropositionalAxiom>,
        comparison_axioms: Vec<ComparisonAxiom>,
        assignment_axioms: Vec<AssignmentAxiom>,
        global_constraint: ExplicitFact,
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

    pub fn from_file(file_name: impl AsRef<std::path::Path>) -> Self {
        let file_content = std::fs::read_to_string(file_name).unwrap();
        parse_numeric_sas_output(&file_content)
            .unwrap() // TODO: Handle errors properly.
            .1
    }

    /// Returns a reference to the metric configuration
    pub fn metric(&self) -> &Metric {
        &self.metric
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum NumericType {
    Constant,
    Derived,
    Cost,
    Regular, // Not sure if Root is correct.
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

    fn metric(&self) -> &Metric {
        &self.metric
    }

    fn get_num_variables(&self) -> usize {
        self.variables.len()
    }

    fn get_variable_name(&self, index: usize) -> Result<&str, &str> {
        if index >= (self.variables.len()) {
            return Err("Index out of bounds");
        }
        Ok(&self.variables[index].name)
    }

    fn get_variable_domain_size(&self, index: usize) -> Result<usize, &str> {
        if index >= (self.variables.len()) {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index].domain_size)
    }

    fn get_variable_axiom_layer(&self, index: usize) -> Result<Option<usize>, &str> {
        if index >= (self.variables.len()) {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index].axiom_layer)
    }

    fn get_variable_default_axiom_value(&self, index: usize) -> Result<usize, &str> {
        if index >= (self.variables.len()) {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index].axiom_default_value)
    }

    fn get_fact_name(&self, _fact: &ExplicitFact) -> &str {
        ""
    }

    fn are_facts_mutex(&self, _fact1: &ExplicitFact, _fact2: &ExplicitFact) -> bool {
        false
    }

    fn get_operator_cost(&self, index: usize, is_axiom: bool) -> u64 {
        if is_axiom {
            return 0;
        }
        if index >= self.operators.len() {
            return 0;
        }
        self.operators[index].cost()
    }

    fn get_operator_name(&self, index: usize, is_axiom: bool) -> &str {
        if is_axiom {
            return "<axiom>";
        }
        if index >= self.operators.len() {
            return "";
        }
        self.operators[index].name()
    }

    fn get_num_operators(&self) -> usize {
        self.operators.len()
    }

    fn get_num_operator_preconditions(&self, index: usize, is_axiom: bool) -> usize {
        if is_axiom {
            // Axioms don't have preconditions in the same way
            return 0;
        }
        if index >= self.operators.len() {
            return 0;
        }
        self.operators[index].preconditions().len()
    }

    fn get_operator_precondition(
        &self,
        _index: usize,
        _precond_index: usize,
        _is_axiom: bool,
    ) -> &ExplicitFact {
        unimplemented!("This function is not yet implemented");
    }

    fn get_num_operator_effects(&self, index: usize, is_axiom: bool) -> usize {
        if is_axiom {
            // Handle axiom effects differently.
            return 0;
        }
        if index >= self.operators.len() {
            return 0;
        }
        self.operators[index].effects().len()
    }

    fn get_num_operator_effect_conditions(
        &self,
        _index: usize,
        _eff_index: usize,
        _is_axiom: bool,
    ) -> usize {
        0
    }

    fn get_operator_effect_condition(
        &self,
        _index: usize,
        _eff_index: usize,
        _cond_index: usize,
        _is_axiom: bool,
    ) -> &ExplicitFact {
        unimplemented!("This function is not yet implemented");
    }

    fn get_operator_effect(
        &self,
        _index: usize,
        _eff_index: usize,
        _is_axiom: bool,
    ) -> &ExplicitFact {
        unimplemented!("This function is not yet implemented");
    }

    fn convert_operator_index(&self, _index: usize, _ancestor_task: &dyn AbstractNumericTask) {}

    fn get_num_axioms(&self) -> usize {
        self.axioms.len()
    }

    fn get_num_goals(&self) -> usize {
        self.goals.len()
    }

    fn get_goal_fact(&self, index: usize) -> &ExplicitFact {
        if index >= self.goals.len() {
            panic!("Goal index {} out of bounds", index);
        }
        &self.goals[index]
    }

    fn get_initial_propositional_state_values(&self) -> Ref<'_, Vec<usize>> {
        self.state.borrow()
    }

    fn get_initial_numeric_state_values(&self) -> Ref<'_, Vec<f64>> {
        self.numeric_state.borrow()
    }

    fn get_initial_propositional_state_values_mut(&self) -> RefMut<'_, Vec<usize>> {
        self.state.borrow_mut()
    }

    fn get_initial_numeric_state_values_mut(&self) -> RefMut<'_, Vec<f64>> {
        self.numeric_state.borrow_mut()
    }

    fn set_initial_numeric_state_values(&self, values: Vec<f64>) {
        *self.numeric_state.borrow_mut() = values;
    }

    fn set_initial_propositional_state_values(&self, values: Vec<usize>) {
        *self.state.borrow_mut() = values;
    }

    fn convert_ancestor_state_values(
        &self,
        _ancestor_state_values: &[usize],
        _ancestor_task: &dyn AbstractNumericTask,
    ) -> Vec<usize> {
        vec![]
    }

    fn get_num_cmp_axioms(&self) -> usize {
        self.comparison_axioms.len()
    }
}
