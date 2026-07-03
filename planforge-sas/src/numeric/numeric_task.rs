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
    sync::Arc,
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

    //TODO: Helpers to get PDB development fast but we don't want the next 4 methods.
    fn abstract_state_values(
        &self,
        propositional_values: &[usize],
        numeric_values: &[f64],
    ) -> Result<(Vec<usize>, Vec<f64>), String> {
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

    fn evaluated_initial_abstract_state_values(&self) -> Result<(Vec<usize>, Vec<f64>), String>;

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

    fn regular_numeric_variable_ids(&self) -> Vec<usize> {
        self.numeric_variables()
            .iter()
            .enumerate()
            .filter_map(|(numeric_var_id, numeric_var)| {
                (numeric_var.get_type() == &NumericType::Regular).then_some(numeric_var_id)
            })
            .collect()
    }

    fn is_linear_cost_operator(&self, operator_id: usize) -> bool {
        linear_metric_operator_cost_expression(self, operator_id).is_some()
    }

    fn operator_cost_coefficients(&self, operator_id: usize) -> Vec<f64> {
        let regular_numeric_variable_ids = self.regular_numeric_variable_ids();
        linear_metric_operator_cost_expression(self, operator_id)
            .map(|expression| {
                regular_numeric_variable_ids
                    .iter()
                    .map(|&numeric_var_id| expression.coefficients[numeric_var_id])
                    .collect()
            })
            .unwrap_or_else(|| {
                todo!(
                    "requested linear action-cost coefficients for non-linear-cost operator {operator_id}"
                )
            })
    }

    fn operator_cost_constant(&self, operator_id: usize) -> f64 {
        linear_metric_operator_cost_expression(self, operator_id)
            .map(|expression| expression.constant)
            .unwrap_or_else(|| {
                todo!(
                    "requested linear action-cost constant for non-linear-cost operator {operator_id}"
                )
            })
    }
}

/// Shared-ownership handle to a task.
///
/// `'a` bounds the borrows the task may hold internally: root tasks are
/// `'static` (`Arc<NumericRootTask>` coerces to `TaskRef<'static>`), while
/// projected/abstracted tasks borrow their parent and instantiate at the
/// parent's lifetime.
pub type TaskRef<'a> = Arc<dyn AbstractNumericTask + 'a>;

/// Delegation impl so a *borrowed* task can be wrapped into a [`TaskRef`]
/// at sites that don't own the task: `Arc::new(task)` with
/// `task: &'a dyn AbstractNumericTask` coerces to `TaskRef<'a>`.
///
/// Every method — including the ones with default bodies — forwards to the
/// referent, so trait-object overrides are preserved through the wrapper.
impl<T: AbstractNumericTask + ?Sized> AbstractNumericTask for &T {
    fn variables(&self) -> &Vec<ExplicitVariable> {
        (**self).variables()
    }
    fn numeric_variables(&self) -> &Vec<NumericVariable> {
        (**self).numeric_variables()
    }
    fn assignment_axioms(&self) -> &Vec<AssignmentAxiom> {
        (**self).assignment_axioms()
    }
    fn comparison_axioms(&self) -> &Vec<ComparisonAxiom> {
        (**self).comparison_axioms()
    }
    fn axioms(&self) -> &Vec<PropositionalAxiom> {
        (**self).axioms()
    }
    fn metric(&self) -> &Metric {
        (**self).metric()
    }
    fn get_num_variables(&self) -> usize {
        (**self).get_num_variables()
    }
    fn get_variable_name(&self, index: usize) -> Result<&str, &str> {
        (**self).get_variable_name(index)
    }
    fn get_variable_domain_size(&self, index: usize) -> Result<usize, &str> {
        (**self).get_variable_domain_size(index)
    }
    fn get_variable_axiom_layer(&self, index: usize) -> Result<Option<usize>, &str> {
        (**self).get_variable_axiom_layer(index)
    }
    fn get_variable_default_axiom_value(&self, index: usize) -> Result<usize, &str> {
        (**self).get_variable_default_axiom_value(index)
    }
    fn get_fact_name(&self, fact: &ExplicitFact) -> &str {
        (**self).get_fact_name(fact)
    }
    fn are_facts_mutex(&self, fact1: &ExplicitFact, fact2: &ExplicitFact) -> bool {
        (**self).are_facts_mutex(fact1, fact2)
    }
    fn get_operators(&self) -> &Vec<Operator> {
        (**self).get_operators()
    }
    fn get_operator_cost(&self, index: usize, is_axiom: bool) -> u64 {
        (**self).get_operator_cost(index, is_axiom)
    }
    fn get_operator_name(&self, index: usize, is_axiom: bool) -> &str {
        (**self).get_operator_name(index, is_axiom)
    }
    fn get_num_operators(&self) -> usize {
        (**self).get_num_operators()
    }
    fn get_num_operator_preconditions(&self, index: usize, is_axiom: bool) -> usize {
        (**self).get_num_operator_preconditions(index, is_axiom)
    }
    fn get_operator_precondition(
        &self,
        index: usize,
        precond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact {
        (**self).get_operator_precondition(index, precond_index, is_axiom)
    }
    fn get_num_operator_effects(&self, index: usize, is_axiom: bool) -> usize {
        (**self).get_num_operator_effects(index, is_axiom)
    }
    fn get_num_operator_effect_conditions(
        &self,
        index: usize,
        eff_index: usize,
        is_axiom: bool,
    ) -> usize {
        (**self).get_num_operator_effect_conditions(index, eff_index, is_axiom)
    }
    fn get_operator_effect_condition(
        &self,
        index: usize,
        eff_index: usize,
        cond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact {
        (**self).get_operator_effect_condition(index, eff_index, cond_index, is_axiom)
    }
    fn get_operator_effect(&self, index: usize, eff_index: usize, is_axiom: bool) -> &ExplicitFact {
        (**self).get_operator_effect(index, eff_index, is_axiom)
    }
    fn convert_operator_index(&self, index: usize, ancestor_task: &dyn AbstractNumericTask) {
        (**self).convert_operator_index(index, ancestor_task)
    }
    fn get_num_axioms(&self) -> usize {
        (**self).get_num_axioms()
    }
    fn get_num_goals(&self) -> usize {
        (**self).get_num_goals()
    }
    fn get_goal_fact(&self, index: usize) -> &ExplicitFact {
        (**self).get_goal_fact(index)
    }
    fn get_initial_propositional_state_values(&self) -> Ref<'_, Vec<usize>> {
        (**self).get_initial_propositional_state_values()
    }
    fn get_initial_numeric_state_values(&self) -> Ref<'_, Vec<f64>> {
        (**self).get_initial_numeric_state_values()
    }
    fn get_initial_propositional_state_values_mut(&self) -> RefMut<'_, Vec<usize>> {
        (**self).get_initial_propositional_state_values_mut()
    }
    fn get_initial_numeric_state_values_mut(&self) -> RefMut<'_, Vec<f64>> {
        (**self).get_initial_numeric_state_values_mut()
    }
    fn set_initial_propositional_state_values(&self, values: Vec<usize>) {
        (**self).set_initial_propositional_state_values(values)
    }
    fn set_initial_numeric_state_values(&self, values: Vec<f64>) {
        (**self).set_initial_numeric_state_values(values)
    }
    fn convert_ancestor_state_values(
        &self,
        ancestor_state_values: &[usize],
        ancestor_task: &dyn AbstractNumericTask,
    ) -> Vec<usize> {
        (**self).convert_ancestor_state_values(ancestor_state_values, ancestor_task)
    }
    fn get_num_cmp_axioms(&self) -> usize {
        (**self).get_num_cmp_axioms()
    }
    fn abstract_state_values(
        &self,
        propositional_values: &[usize],
        numeric_values: &[f64],
    ) -> Result<(Vec<usize>, Vec<f64>), String> {
        (**self).abstract_state_values(propositional_values, numeric_values)
    }
    fn evaluated_initial_abstract_state_values(&self) -> Result<(Vec<usize>, Vec<f64>), String> {
        (**self).evaluated_initial_abstract_state_values()
    }
    fn abstract_operator_cost(&self, operator_id: usize) -> f64 {
        (**self).abstract_operator_cost(operator_id)
    }
    fn min_abstract_operator_cost(&self) -> f64 {
        (**self).min_abstract_operator_cost()
    }
    fn assignment_axiom_lookup(&self) -> Vec<Option<usize>> {
        (**self).assignment_axiom_lookup()
    }
    fn linearize_numeric_var(
        &self,
        numeric_var_id: usize,
    ) -> Result<crate::numeric::utils::linear_effects::LinearExpression, LinearizationError> {
        (**self).linearize_numeric_var(numeric_var_id)
    }
    fn linearized_assignment_effects(
        &self,
        operator_id: usize,
    ) -> Result<Vec<LinearNumericEffect>, LinearizationError> {
        (**self).linearized_assignment_effects(operator_id)
    }
    fn regular_numeric_variable_ids(&self) -> Vec<usize> {
        (**self).regular_numeric_variable_ids()
    }
    fn is_linear_cost_operator(&self, operator_id: usize) -> bool {
        (**self).is_linear_cost_operator(operator_id)
    }
    fn operator_cost_coefficients(&self, operator_id: usize) -> Vec<f64> {
        (**self).operator_cost_coefficients(operator_id)
    }
    fn operator_cost_constant(&self, operator_id: usize) -> f64 {
        (**self).operator_cost_constant(operator_id)
    }
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

/// Variable/value pair. `u32` fields halve the per-fact footprint compared
/// to `usize` on 64-bit targets (16 B → 8 B), at the cost of a hard 4 G
/// ceiling on variable and value IDs — vastly above anything realistic
/// planning tasks reach. The cap is checked at SAS-load time.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct ExplicitFact {
    var_id: u32,
    value_id: u32,
}

impl ExplicitFact {
    /// Constructor accepts `usize` to minimize call-site churn; values are
    /// narrowed at construction. Out-of-range arguments are caught in
    /// debug builds.
    pub fn new(var: usize, value: usize) -> Self {
        debug_assert!(
            var <= u32::MAX as usize,
            "ExplicitFact var {var} > u32::MAX"
        );
        debug_assert!(
            value <= u32::MAX as usize,
            "ExplicitFact value {value} > u32::MAX"
        );
        ExplicitFact {
            var_id: var as u32,
            value_id: value as u32,
        }
    }
    #[inline(always)]
    pub fn var(&self) -> usize {
        self.var_id as usize
    }
    #[inline(always)]
    pub fn value(&self) -> usize {
        self.value_id as usize
    }
    pub fn is_hold(&self, state: &ConcreteState, state_registry: &StateRegistry) -> bool {
        let buffer = state.buffer(state_registry);
        let state_packer = state_registry.global_state_packer();
        let value = state_packer.get(buffer, self.var());
        value == self.value() as u64
    }
}

impl fmt::Debug for ExplicitFact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fact(var: {}, value: {})", self.var(), self.value())
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

pub fn evaluate_metric_from_values<T: AbstractNumericTask + ?Sized>(
    task: &T,
    numeric_values: &[f64],
) -> f64 {
    let metric_var_id = task.metric().var_id();
    match metric_var_id {
        Some(var_id) => numeric_values.get(var_id).copied().unwrap_or(0.0),
        None => 0.0,
    }
}

pub fn propagate_assignment_axiom_values<T: AbstractNumericTask + ?Sized>(
    task: &T,
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

fn linear_metric_operator_cost_expression<T: AbstractNumericTask + ?Sized>(
    task: &T,
    operator_id: usize,
) -> Option<crate::numeric::utils::linear_effects::LinearExpression> {
    if !task.metric().use_metric() {
        return None;
    }

    let metric_var_id = task.metric().var_id().unwrap();
    let metric_variable = task.numeric_variables().get(metric_var_id)?;
    if metric_variable.get_type() != &NumericType::Cost {
        return None;
    }

    let operator = task.get_operators().get(operator_id).unwrap_or_else(|| {
        panic!("operator id {operator_id} is out of bounds for linear metric-cost extraction")
    });
    let metric_direction = if task.metric().is_min() { 1.0 } else { -1.0 };
    let mut linear_cost_expression = None;

    for assignment_effect in operator.assignment_effects() {
        if assignment_effect.affected_var_id() != metric_var_id {
            continue;
        }
        if assignment_effect.is_conditional() || !assignment_effect.conditions().is_empty() {
            continue;
        }

        let source_expression = task
            .linearize_numeric_var(assignment_effect.var_id)
            .unwrap_or_else(|error| {
                panic!(
                    "failed to linearize metric-cost source variable {} for operator {operator_id}: {error}",
                    assignment_effect.var_id()
                )
            });
        let candidate = match assignment_effect.operation() {
            AssignmentOperation::Plus => source_expression.scale(metric_direction),
            AssignmentOperation::Minus => source_expression.scale(-metric_direction),
            AssignmentOperation::Assign
            | AssignmentOperation::Times
            | AssignmentOperation::Divide => continue,
        };

        if candidate
            .coefficients
            .iter()
            .all(|&coefficient| coefficient == 0.0)
        {
            continue;
        }

        if linear_cost_expression.is_some() {
            todo!(
                "multiple unconditional linear metric-cost effects for operator {operator_id} are not implemented yet"
            );
        }
        linear_cost_expression = Some(candidate);
    }

    linear_cost_expression
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
    name: Box<str>,
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
        // `Box<str>` is two words (ptr + len) vs `String`'s three words
        // (ptr + len + cap) and drops spare capacity from any growth steps
        // during parsing. For tasks with 10^6 operators this trims the
        // task-loading peak by 20-30 MB. Names are immutable so we never
        // need the `cap` field again.
        Operator {
            name: name.into_boxed_str(),
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
    abstract_propositional_var_ids: Vec<usize>,
    abstract_numeric_var_ids: Vec<usize>,
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
        Self::from_str(&file_content)
    }

    /// Parse a `NumericRootTask` from the preprocessor's text format held in
    /// memory. Equivalent to `from_file` minus the disk read; used by the
    /// in-memory translate→preprocess→search pipeline so the binary
    /// `output` file never has to materialize on disk.
    pub fn from_str(content: &str) -> Self {
        parse_numeric_sas_output(content)
            .unwrap() // TODO: Handle errors properly.
            .1
    }

    /// Returns a reference to the metric configuration
    pub fn metric(&self) -> &Metric {
        &self.metric
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
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

    fn evaluated_initial_abstract_state_values(&self) -> Result<(Vec<usize>, Vec<f64>), String> {
        let mut propositional = self.get_initial_propositional_state_values().to_vec();
        let mut numeric = self.get_initial_numeric_state_values().to_vec();
        evaluate_state_with_axiom_closure(self, &mut propositional, &mut numeric)?;
        Ok((propositional, numeric))
    }
}

fn evaluate_state_with_axiom_closure(
    task: &dyn AbstractNumericTask,
    propositional: &mut [usize],
    numeric: &mut [f64],
) -> Result<(), String> {
    let packer = Arc::new(abstract_propositional_packer(task));
    let mut packed = vec![0u64; packer.num_bins() as usize];
    for (var_id, value) in propositional.iter().enumerate() {
        packer.set(&mut packed, var_id, *value as u64);
    }
    let axiom_evaluator = AxiomEvaluator::new(Arc::new(task), packer.clone());
    finish_axiom_closure(
        &packer,
        propositional,
        numeric,
        &mut packed,
        &axiom_evaluator,
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

fn finish_axiom_closure(
    packer: &IntDoublePacker,
    propositional: &mut [usize],
    numeric: &mut [f64],
    packed: &mut [u64],
    axiom_evaluator: &AxiomEvaluator<'_>,
) -> Result<(), String> {
    axiom_evaluator
        .evaluate_arithmetic_axioms(numeric)
        .map_err(|err| format!("failed to evaluate arithmetic axioms: {err:?}"))?;
    axiom_evaluator
        .evaluate(packed, numeric)
        .map_err(|err| format!("failed to evaluate axioms: {err:?}"))?;

    for (var_id, slot) in propositional.iter_mut().enumerate() {
        *slot = packer.get(packed, var_id) as usize;
    }

    Ok(())
}

#[allow(unused)]
fn facts_hold_values(propositional: &[usize], facts: &[ExplicitFact]) -> bool {
    facts
        .iter()
        .all(|fact| propositional.get(fact.var()).copied() == Some(fact.value()))
}

#[allow(unused)]
fn assignment_effect_holds_values(propositional: &[usize], effect: &AssignmentEffect) -> bool {
    !effect.is_conditional() || facts_hold_values(propositional, effect.conditions())
}

#[allow(unused)]
fn overwrite_vec<T: Copy>(dst: &mut Vec<T>, src: &[T]) {
    dst.clear();
    dst.extend_from_slice(src);
}
