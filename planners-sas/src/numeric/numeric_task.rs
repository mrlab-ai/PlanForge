use crate::numeric::axioms::{
    AssignmentAxiom, AxiomEvaluator, ComparisonAxiom, PropositionalAxiom,
};
use crate::numeric::numeric_parser::parse_numeric_sas_output;
use crate::numeric::state_registry::{ConcreteState, StateRegistry};
use crate::numeric::utils::int_packer::IntDoublePacker;
use crate::numeric::utils::linear_effects::{
    LinearNumericEffect, LinearizationError, build_assignment_axiom_lookup, linearize_numeric_var,
    linearize_operator_assignment_effects,
};
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

    fn get_initial_propositional_state_values(&self) -> Ref<'_, Vec<i32>>;
    fn get_initial_numeric_state_values(&self) -> Ref<'_, Vec<f64>>;

    fn get_initial_propositional_state_values_mut(&self) -> RefMut<'_, Vec<i32>>;
    fn get_initial_numeric_state_values_mut(&self) -> RefMut<'_, Vec<f64>>;

    fn set_initial_numeric_state_values(&self, values: Vec<f64>);
    fn set_initial_propositional_state_values(&self, values: Vec<i32>);

    fn convert_ancestor_state_values(
        &self,
        ancestor_state_values: &Vec<i32>,
        ancestor_task: &dyn AbstractNumericTask,
    ) -> Vec<i32>;

    fn get_num_cmp_axioms(&self) -> i32;

    //TODO: Helpers to get PDB development fast but we dont want the next 4 methods.
    fn abstract_state_values(
        &self,
        propositional_values: &[i32],
        numeric_values: &[f64],
    ) -> Result<(Vec<i32>, Vec<f64>), String> {
        if propositional_values.len() != self.variables().len() {
            return Err(format!(
                "expected {} propositional values, got {}",
                self.variables().len(),
                propositional_values.len()
            ));
        }
        if numeric_values.len() != self.numeric_variables().len() {
            return Err(format!(
                "expected {} numeric values, got {}",
                self.numeric_variables().len(),
                numeric_values.len()
            ));
        }
        Ok((propositional_values.to_vec(), numeric_values.to_vec()))
    }

    fn evaluated_initial_abstract_state_values(&self) -> Result<(Vec<i32>, Vec<f64>), String>;

    fn abstract_operator_cost(&self, operator_id: usize) -> f64 {
        self.get_operators()
            .get(operator_id)
            .map(|operator| metric_operator_cost_from_initial_values(self, operator))
            .unwrap_or(0.0)
    }

    fn min_abstract_operator_cost(&self) -> f64 {
        let min_operator_cost = (0..self.get_operators().len())
            .map(|operator_id| self.abstract_operator_cost(operator_id))
            .fold(f64::INFINITY, f64::min);
        if min_operator_cost.is_finite() {
            min_operator_cost.max(0.0)
        } else {
            0.0
        }
    }

    fn assignment_axiom_lookup(&self) -> Vec<Option<usize>> {
        build_assignment_axiom_lookup(self)
    }

    fn linearize_numeric_var(
        &self,
        numeric_var_id: usize,
    ) -> Result<crate::numeric::utils::linear_effects::LinearExpression, LinearizationError> {
        linearize_numeric_var(self, numeric_var_id)
    }

    fn linearized_assignment_effects(
        &self,
        operator_id: usize,
    ) -> Result<Vec<LinearNumericEffect>, LinearizationError> {
        linearize_operator_assignment_effects(self, operator_id)
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn get_type(&self) -> &NumericType {
        &self.numeric_type
    }

    pub fn axiom_layer(&self) -> i32 {
        self.axiom_layer
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
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

#[derive(Debug, Clone, PartialEq)]
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

    pub fn precondition_value(&self) -> i32 {
        self.precondition_value
    }

    pub fn conditions(&self) -> &Vec<Fact> {
        &self.conditions
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

pub fn evaluate_metric_from_values<T: AbstractNumericTask + ?Sized>(
    task: &T,
    numeric_values: &[f64],
) -> f64 {
    let metric_var_id = task.metric().var_id();
    if metric_var_id < 0 {
        return 0.0;
    }

    numeric_values
        .get(metric_var_id as usize)
        .copied()
        .unwrap_or(0.0)
}

pub fn propagate_assignment_axiom_values<T: AbstractNumericTask + ?Sized>(
    task: &T,
    numeric_values: &mut Vec<f64>,
) {
    let mut changed = true;
    while changed {
        changed = false;
        for axiom in task.assignment_axioms() {
            let affected_var_id = axiom.get_affected_var_id() as usize;
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

pub fn metric_operator_cost_from_initial_values<T: AbstractNumericTask + ?Sized>(
    task: &T,
    operator: &Operator,
) -> f64 {
    if !task.metric().use_metric() {
        return operator.cost() as f64;
    }

    let initial_numeric_values = task.get_initial_numeric_state_values();
    let mut numeric_values = initial_numeric_values.to_vec();
    let old_metric = evaluate_metric_from_values(task, &numeric_values);

    for effect in operator.assignment_effects() {
        let assignment_var_id = effect.var_id() as usize;
        let affected_var_id = effect.affected_var_id() as usize;
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

    pub fn is_conditional(&self) -> bool {
        self.is_conditional
    }

    pub fn conditions(&self) -> &Vec<Fact> {
        &self.conditions
    }
}

#[derive(Debug, Clone, PartialEq)]
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

    pub fn name(&self) -> &str {
        &self.name
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

    pub fn cost(&self) -> u32 {
        self.cost
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
    abstract_propositional_var_ids: Vec<usize>,
    abstract_numeric_var_ids: Vec<usize>,
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
        let abstract_propositional_var_ids = (0..state.len()).collect();
        let abstract_numeric_var_ids = (0..numeric_state.len()).collect();
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
            abstract_propositional_var_ids,
            abstract_numeric_var_ids,
            axioms,
            comparison_axioms,
            assignment_axioms,
            global_constraint,
        }
    }

    pub fn from_file(file_name: impl AsRef<std::path::Path>) -> Self {
        let file_content = std::fs::read_to_string(file_name).unwrap();
        parse_numeric_sas_output(&file_content)
            .unwrap() // TODO: Handle errors properly
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

    fn metric(&self) -> &Metric {
        &self.metric
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
        if is_axiom {
            return 0;
        }
        if index < 0 || index >= self.operators.len() as i32 {
            return 0;
        }
        self.operators[index as usize].cost() as i32
    }

    fn get_operator_name(&self, index: i32, is_axiom: bool) -> &str {
        if is_axiom {
            return "<axiom>";
        }
        if index < 0 || index >= self.operators.len() as i32 {
            return "";
        }
        self.operators[index as usize].name()
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

    fn get_initial_propositional_state_values(&self) -> Ref<'_, Vec<i32>> {
        self.state.borrow()
    }

    fn get_initial_numeric_state_values(&self) -> Ref<'_, Vec<f64>> {
        self.numeric_state.borrow()
    }

    fn get_initial_propositional_state_values_mut(&self) -> RefMut<'_, Vec<i32>> {
        self.state.borrow_mut()
    }

    fn get_initial_numeric_state_values_mut(&self) -> RefMut<'_, Vec<f64>> {
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

    fn evaluated_initial_abstract_state_values(&self) -> Result<(Vec<i32>, Vec<f64>), String> {
        let mut propositional = self.get_initial_propositional_state_values().to_vec();
        let mut numeric = self.get_initial_numeric_state_values().to_vec();
        evaluate_state_with_axiom_closure(self, &mut propositional, &mut numeric)?;
        Ok((propositional, numeric))
    }
}

fn evaluate_state_with_axiom_closure(
    task: &dyn AbstractNumericTask,
    propositional: &mut Vec<i32>,
    numeric: &mut Vec<f64>,
) -> Result<(), String> {
    let packer = abstract_propositional_packer(task);
    let mut packed = vec![0u64; packer.num_bins() as usize];
    for (var_id, value) in propositional.iter().enumerate() {
        packer.set(&mut packed, var_id as i32, *value as u64);
    }
    evaluate_state_with_axiom_closure_and_packed_dyn(
        task,
        &packer,
        propositional,
        numeric,
        &mut packed,
    )
}

fn abstract_propositional_packer<T: AbstractNumericTask + ?Sized>(task: &T) -> IntDoublePacker {
    let ranges: Vec<u64> = task
        .variables()
        .iter()
        .map(|variable| variable.domain_size() as u64)
        .collect();
    IntDoublePacker::new(&ranges)
}

fn evaluate_state_with_axiom_closure_and_packed_sized<T: AbstractNumericTask>(
    task: &T,
    packer: &IntDoublePacker,
    propositional: &mut Vec<i32>,
    numeric: &mut Vec<f64>,
    packed: &mut Vec<u64>,
) -> Result<(), String> {
    let axiom_evaluator = AxiomEvaluator::new(task, packer);
    finish_axiom_closure(packer, propositional, numeric, packed, &axiom_evaluator)
}

fn evaluate_state_with_axiom_closure_and_packed_dyn(
    task: &dyn AbstractNumericTask,
    packer: &IntDoublePacker,
    propositional: &mut Vec<i32>,
    numeric: &mut Vec<f64>,
    packed: &mut Vec<u64>,
) -> Result<(), String> {
    let axiom_evaluator = AxiomEvaluator::new(task, packer);
    finish_axiom_closure(packer, propositional, numeric, packed, &axiom_evaluator)
}

fn finish_axiom_closure(
    packer: &IntDoublePacker,
    propositional: &mut Vec<i32>,
    numeric: &mut Vec<f64>,
    packed: &mut Vec<u64>,
    axiom_evaluator: &AxiomEvaluator<'_>,
) -> Result<(), String> {
    axiom_evaluator
        .evaluate_arithmetic_axioms(numeric)
        .map_err(|err| format!("failed to evaluate arithmetic axioms: {err:?}"))?;
    axiom_evaluator
        .evaluate(packed, numeric)
        .map_err(|err| format!("failed to evaluate axioms: {err:?}"))?;

    for (var_id, slot) in propositional.iter_mut().enumerate() {
        *slot = packer.get(packed, var_id as i32) as i32;
    }

    Ok(())
}

fn facts_hold_values(propositional: &[i32], facts: &[Fact]) -> bool {
    facts
        .iter()
        .all(|fact| propositional.get(fact.var() as usize).copied() == Some(fact.value()))
}

fn assignment_effect_holds_values(propositional: &[i32], effect: &AssignmentEffect) -> bool {
    !effect.is_conditional() || facts_hold_values(propositional, effect.conditions())
}

fn overwrite_vec<T: Copy>(dst: &mut Vec<T>, src: &[T]) {
    dst.clear();
    dst.extend_from_slice(src);
}
