//! The immutable root planning task and its four variable-id spaces.
//!
//! Task propositional ids index [`NumericRootTask::variables`]. Numeric-condition
//! ids use that same index range but are tagged as [`FactNamespace::Condition`]
//! so callers cannot confuse a comparison verdict with an ordinary proposition.
//! Task numeric ids independently index [`NumericRootTask::numeric_variables`].
//! Domain abstractions have a fourth, private combined space, tagged
//! [`FactNamespace::NumericVariable`], whose values are partition ids rather
//! than task-domain values.
//!
//! Axiom layers are stratified across those spaces: derived numeric assignment
//! axioms run first, all numeric comparisons occupy the next layer, and
//! propositional axioms occupy higher layers. [`NumericRootTask::new`] checks
//! these representation and layer invariants once before publishing a task.

use crate::axioms::{AssignmentAxiom, AxiomEvaluator, ComparisonAxiom, PropositionalAxiom};
use crate::numeric_conditions::{
    ConditionValue, NumericConditionError, NumericConditions, assignment_axiom_lookup,
};
use crate::numeric_parser::parse_numeric_sas_output;
use crate::state_registry::ConcreteStateView;
use crate::utils::errors::AssignmentAxiomError;
use crate::utils::int_packer::IntDoublePacker;
use crate::utils::linear_effects::{
    LinearNumericEffect, LinearizationError, linearize_numeric_var,
    linearize_operator_assignment_effects,
};
use std::{collections::HashSet, fmt, sync::Arc};

/// A planning task the search can read.
///
/// The trait is read-only: a task's initial state is closed under its axioms
/// once, when the task is built, so nothing needs to mutate it afterwards.
/// That is what lets the trait require `Send + Sync`, which in turn makes
/// [`TaskRef`] shareable across threads.
pub trait AbstractNumericTask: Send + Sync {
    fn variables(&self) -> &Vec<ExplicitVariable>;
    fn numeric_variables(&self) -> &Vec<NumericVariable>;
    fn assignment_axioms(&self) -> &Vec<AssignmentAxiom>;
    fn comparison_axioms(&self) -> &Vec<ComparisonAxiom>;
    /// The task's numeric conditions, one per comparison axiom, built once
    /// when the task is constructed. This is the only place the
    /// "propositional variable -> comparison axiom" mapping lives.
    ///
    /// Returned behind an `Arc` so components that outlive the borrow —
    /// abstraction factories, heuristics — can share the conditions instead
    /// of rebuilding or deep-copying them. Method calls auto-deref, so
    /// `task.numeric_conditions().for_var(v)` reads as usual.
    fn numeric_conditions(&self) -> &Arc<NumericConditions>;
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
    fn goals(&self) -> &[ExplicitFact];
    fn get_num_goals(&self) -> usize;
    fn get_goal_fact(&self, index: usize) -> &ExplicitFact;

    /// The initial values of the propositional variables, already closed
    /// under the task's axioms.
    fn get_initial_propositional_state_values(&self) -> &[usize];
    /// The initial values of the numeric variables, already closed under the
    /// task's axioms.
    fn get_initial_numeric_state_values(&self) -> &[f64];

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
        let operator = self.get_operators().get(operator_id).unwrap_or_else(|| {
            panic!("operator id {operator_id} is out of bounds for cost lookup")
        });
        metric_operator_cost_from_initial_values(self, operator)
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

    fn assignment_axiom_lookup(&self) -> Result<Vec<Option<usize>>, NumericConditionError> {
        assignment_axiom_lookup(self.numeric_variables().len(), self.assignment_axioms())
    }

    fn linearize_numeric_var(
        &self,
        numeric_var_id: usize,
    ) -> Result<crate::utils::linear_effects::LinearExpression, LinearizationError> {
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
    fn numeric_conditions(&self) -> &Arc<NumericConditions> {
        (**self).numeric_conditions()
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
    fn goals(&self) -> &[ExplicitFact] {
        (**self).goals()
    }
    fn get_num_goals(&self) -> usize {
        (**self).get_num_goals()
    }
    fn get_goal_fact(&self, index: usize) -> &ExplicitFact {
        (**self).get_goal_fact(index)
    }
    fn get_initial_propositional_state_values(&self) -> &[usize] {
        (**self).get_initial_propositional_state_values()
    }
    fn get_initial_numeric_state_values(&self) -> &[f64] {
        (**self).get_initial_numeric_state_values()
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
    fn assignment_axiom_lookup(&self) -> Result<Vec<Option<usize>>, NumericConditionError> {
        (**self).assignment_axiom_lookup()
    }
    fn linearize_numeric_var(
        &self,
        numeric_var_id: usize,
    ) -> Result<crate::utils::linear_effects::LinearExpression, LinearizationError> {
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

#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct ExplicitVariable {
    domain_size: usize,
    name: String,
    fact_names: Vec<String>,
    axiom_layer: Option<usize>,
    /// The value a *derived* variable holds until an axiom proves otherwise.
    ///
    /// The axiom closure resets every derived variable to this value before it
    /// runs, and reads "still at the default" as "not proven" when it admits
    /// negation-by-failure literals. The SAS format carries it in the
    /// initial-state block, which is why the parser can only fill it in once
    /// that block has been read.
    ///
    /// A non-derived variable is never reset, so this field is not read for
    /// one; it then just repeats the variable's initial value.
    axiom_default_value: usize,
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

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn axiom_layer(&self) -> Option<usize> {
        self.axiom_layer
    }

    pub fn with_axiom_layer(&self, axiom_layer: Option<usize>) -> Self {
        Self {
            axiom_layer,
            ..self.clone()
        }
    }

    pub fn domain_size(&self) -> usize {
        self.domain_size
    }
}

#[derive(Debug, Clone, PartialEq)]
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

/// Panic unless every fact `task` exposes names the namespace the task's own
/// numeric conditions put its variable in.
///
/// A mistagged fact is invisible later: it still denotes a well-formed
/// variable, just the wrong kind of one, so the search produces a wrong plan
/// and no crash. That is why this is an assertion and not a diagnostic.
pub fn assert_fact_namespaces(task: &dyn AbstractNumericTask) {
    let conditions = task.numeric_conditions();
    let check = |fact: &ExplicitFact, origin: &dyn fmt::Display| {
        assert_ne!(
            fact.namespace(),
            FactNamespace::NumericVariable,
            "{origin}: {fact:?} names a domain abstraction's private numeric id space, \
             which is not a variable of this task at all"
        );
        assert!(
            fact.var() < conditions.num_propositional_vars(),
            "{origin}: {fact:?} names variable {}, past the task's {} propositional variables",
            fact.var(),
            conditions.num_propositional_vars()
        );
        let expected = conditions.namespace_of(fact.var());
        assert_eq!(
            fact.namespace(),
            expected,
            "{origin}: {fact:?} is tagged {:?}, but variable {} belongs to {expected:?}",
            fact.namespace(),
            fact.var()
        );
    };

    for (operator_id, operator) in task.get_operators().iter().enumerate() {
        let origin = format_args!("operator {operator_id}").to_string();
        for precondition in operator.preconditions() {
            check(precondition, &origin);
        }
        for effect in operator.effects() {
            for condition in effect.conditions() {
                check(condition, &origin);
            }
        }
        for effect in operator.assignment_effects() {
            for condition in effect.conditions() {
                check(condition, &origin);
            }
        }
    }
    for (axiom_id, axiom) in task.axioms().iter().enumerate() {
        let origin = format_args!("axiom {axiom_id}").to_string();
        for condition in axiom.conditions() {
            check(condition, &origin);
        }
    }
    for goal_index in 0..task.get_num_goals() {
        check(task.get_goal_fact(goal_index), &"goal");
    }
}

/// [`assert_fact_namespaces`] in debug builds only. The check walks every fact
/// of the task, which is linear in the task size and therefore too costly to
/// pay for on every release run.
pub fn debug_assert_fact_namespaces(task: &dyn AbstractNumericTask) {
    if cfg!(debug_assertions) {
        assert_fact_namespaces(task);
    }
}

/// Which id space a fact's variable belongs to.
///
/// A propositional variable, a numeric condition and a domain abstraction's
/// numeric variable are different kinds of thing that happen to share the
/// `(variable, value)` shape, so a fact carries the answer instead of leaving
/// callers to rediscover it from [`NumericConditions::is_condition_var`] or
/// from an unlabelled `num_propositional_vars + numeric_var_id` offset.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
#[repr(u32)]
pub enum FactNamespace {
    /// The fact's variable is a genuine propositional variable.
    Propositional = 0,
    /// The fact's variable carries the truth value of a numeric condition.
    Condition = 1,
    /// The fact's variable is a *numeric* variable, addressed in a domain
    /// abstraction's own id space: propositional variables occupy
    /// `[0, num_propositional_vars)` and numeric variables follow them, so the
    /// id is `num_propositional_vars + numeric_var_id` and the value is a
    /// partition index rather than a propositional value.
    ///
    /// That id space is private to the abstraction machinery. No fact a task
    /// exposes may carry this tag, which is what
    /// [`assert_fact_namespaces`] checks.
    NumericVariable = 2,
}

impl FactNamespace {
    /// Number of tag bits [`ExplicitFact`] reserves in its variable id.
    const BITS: u32 = 4;

    #[inline(always)]
    const fn from_tag(tag: u32) -> Self {
        match tag {
            0 => FactNamespace::Propositional,
            1 => FactNamespace::Condition,
            2 => FactNamespace::NumericVariable,
            // Only `ExplicitFact`'s constructors write the tag, and they only
            // ever write a discriminant of this enum.
            _ => unreachable!(),
        }
    }
}

/// Variable/value pair, tagged with the variable's [`FactNamespace`].
///
/// `u32` fields halve the per-fact footprint compared to `usize` on 64-bit
/// targets (16 B → 8 B). The namespace occupies the top
/// [`FactNamespace::BITS`] bits of the variable id, leaving a hard
/// [`Self::MAX_VAR_ID`] ceiling — vastly above anything realistic planning
/// tasks reach, and checked at construction.
///
/// The tag lives in the *high* bits on purpose: [`Self::var`] is a plain mask
/// of the low bits, so every existing var-major sort and dense per-variable
/// sweep keeps its meaning.
#[derive(Clone, Copy)]
pub struct ExplicitFact {
    /// `namespace << (32 - FactNamespace::BITS) | var`.
    var_id: u32,
    value_id: u32,
}

impl ExplicitFact {
    const VAR_BITS: u32 = u32::BITS - FactNamespace::BITS;
    const VAR_MASK: u32 = (1 << Self::VAR_BITS) - 1;

    /// Largest variable id a fact can name.
    pub const MAX_VAR_ID: usize = Self::VAR_MASK as usize;

    /// Fact on a genuine propositional variable.
    #[inline]
    pub fn propositional(var: usize, value: usize) -> Self {
        Self::in_namespace(FactNamespace::Propositional, var, value)
    }

    /// Fact on the variable carrying a numeric condition's truth value.
    #[inline]
    pub fn condition(var: usize, value: usize) -> Self {
        Self::in_namespace(FactNamespace::Condition, var, value)
    }

    /// Fact on a numeric variable in a domain abstraction's id space.
    ///
    /// `abstraction_var` is the abstraction id, not the numeric variable id;
    /// see [`FactNamespace::NumericVariable`] for the offset encoding. `value`
    /// is a partition index of that numeric variable.
    #[inline]
    pub fn numeric_variable(abstraction_var: usize, value: usize) -> Self {
        Self::in_namespace(FactNamespace::NumericVariable, abstraction_var, value)
    }

    /// Constructors accept `usize` to minimize call-site churn; values are
    /// narrowed here, and out-of-range arguments fail rather than silently
    /// aliasing another variable or bleeding into the namespace tag.
    pub fn in_namespace(namespace: FactNamespace, var: usize, value: usize) -> Self {
        assert!(
            var <= Self::MAX_VAR_ID,
            "ExplicitFact var {var} exceeds the {} packed variable-id bits",
            Self::VAR_BITS
        );
        debug_assert!(
            value <= u32::MAX as usize,
            "ExplicitFact value {value} > u32::MAX"
        );
        ExplicitFact {
            var_id: (namespace as u32) << Self::VAR_BITS | var as u32,
            value_id: value as u32,
        }
    }

    /// The same fact re-tagged. Used by the one pass that owns namespace
    /// assignment, [`NumericRootTask::assign_fact_namespaces`].
    #[inline]
    #[must_use]
    pub fn with_namespace(self, namespace: FactNamespace) -> Self {
        ExplicitFact {
            var_id: (namespace as u32) << Self::VAR_BITS | (self.var_id & Self::VAR_MASK),
            value_id: self.value_id,
        }
    }

    #[inline(always)]
    pub fn namespace(&self) -> FactNamespace {
        FactNamespace::from_tag(self.var_id >> Self::VAR_BITS)
    }

    #[inline(always)]
    pub fn is_condition(&self) -> bool {
        self.namespace() == FactNamespace::Condition
    }

    #[inline(always)]
    pub fn var(&self) -> usize {
        (self.var_id & Self::VAR_MASK) as usize
    }
    #[inline(always)]
    pub fn value(&self) -> usize {
        self.value_id as usize
    }
    pub fn is_hold(&self, state: ConcreteStateView<'_>) -> bool {
        let value = state.packer().get(state.propositional(), self.var());
        value == self.value() as u64
    }
}

/// Identity, ordering and hashing are all over `(variable, value)` and ignore
/// the namespace tag.
///
/// The tag is a function of the variable, so dropping it loses nothing — and
/// it means the tag can never split one fact into two that compare unequal but
/// order equal, which would quietly break `sort` + `dedup` pairs and mutex
/// lookups. Ordering stays variable-major, which is what the successor
/// generator's decision tree is built from.
impl PartialEq for ExplicitFact {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value_id == other.value_id && self.var() == other.var()
    }
}

impl Eq for ExplicitFact {}

impl std::hash::Hash for ExplicitFact {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.var().hash(state);
        self.value_id.hash(state);
    }
}

impl Ord for ExplicitFact {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.var(), self.value_id).cmp(&(other.var(), other.value_id))
    }
}

impl PartialOrd for ExplicitFact {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for ExplicitFact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.namespace() {
            FactNamespace::Propositional => {
                write!(f, "Fact(var: {}, value: {})", self.var(), self.value())
            }
            FactNamespace::Condition => {
                write!(f, "Fact(cond: {}, value: {})", self.var(), self.value())
            }
            FactNamespace::NumericVariable => {
                write!(f, "Fact(num: {}, partition: {})", self.var(), self.value())
            }
        }
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

    pub fn conditions_met(&self, state: ConcreteStateView<'_>) -> bool {
        for condition in &self.conditions {
            if !condition.is_hold(state) {
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
        Some(var_id) => *numeric_values.get(var_id).unwrap_or_else(|| {
            panic!(
                "metric variable {var_id} is out of bounds for {} numeric values",
                numeric_values.len()
            )
        }),
        None => 0.0,
    }
}

pub fn propagate_assignment_axiom_values<T: AbstractNumericTask + ?Sized>(
    task: &T,
    numeric_values: &mut [f64],
) -> Result<(), AssignmentAxiomError> {
    // Assignment axioms are stored in dependency-layer order, so each RHS is
    // complete when it is visited and one forward pass closes the values.
    for axiom in task.assignment_axioms() {
        let affected_var_id = axiom.get_affected_var_id();
        assert!(
            affected_var_id < numeric_values.len(),
            "assignment axiom target {affected_var_id} is out of bounds for {} numeric values",
            numeric_values.len()
        );
        axiom.update_values(numeric_values)?;
    }
    Ok(())
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

    // Effects of one operator apply simultaneously, so every read must see the
    // pre-application values. Collect first, publish second; see
    // `StateRegistry::apply_numeric_effects_inner` for the same reasoning on
    // the search path.
    let mut results = Vec::with_capacity(operator.assignment_effects().len());
    for effect in operator.assignment_effects() {
        let assignment_var_id = effect.var_id();
        let affected_var_id = effect.affected_var_id();
        assert!(
            assignment_var_id < numeric_values.len(),
            "assignment variable {assignment_var_id} of operator {} is out of bounds for {} numeric variables",
            operator.name(),
            numeric_values.len(),
        );
        assert!(
            affected_var_id < numeric_values.len(),
            "affected variable {affected_var_id} of operator {} is out of bounds for {} numeric variables",
            operator.name(),
            numeric_values.len(),
        );

        let result = AssignmentOperation::apply(
            numeric_values[affected_var_id],
            effect.operation(),
            numeric_values[assignment_var_id],
        );
        results.push((affected_var_id, result));
    }
    for (affected_var_id, result) in results {
        numeric_values[affected_var_id] = result;
    }

    propagate_assignment_axiom_values(task, &mut numeric_values).unwrap_or_else(|error| {
        panic!(
            "operator {} cannot evaluate assignment axioms while computing its metric cost: \
             {error:?}",
            operator.name()
        )
    });
    let new_metric = evaluate_metric_from_values(task, &numeric_values);
    let delta = if task.metric().is_min() {
        new_metric - old_metric
    } else {
        old_metric - new_metric
    };
    assert!(
        delta >= 0.0,
        "operator {} has negative metric cost {delta}, which search does not support",
        operator.name()
    );
    delta
}

fn linear_metric_operator_cost_expression<T: AbstractNumericTask + ?Sized>(
    task: &T,
    operator_id: usize,
) -> Option<crate::utils::linear_effects::LinearExpression> {
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
    repeated_assignment_targets: Box<[RepeatedTarget]>,
    cost: u64,
}

/// Whether an assignment effect is the first write to its target within its
/// operator or a further additive write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RepeatedTarget {
    First,
    Additive,
}

impl Operator {
    pub fn new(
        name: String,
        preconditions: Vec<ExplicitFact>,
        effects: Vec<Effect>,
        assignment_effects: Vec<AssignmentEffect>,
        cost: u64,
    ) -> Self {
        let mut repeated_assignment_targets = Vec::with_capacity(assignment_effects.len());
        let mut target_is_additive = std::collections::HashMap::new();
        for effect in &assignment_effects {
            let affected_var_id = effect.affected_var_id();
            let is_additive = matches!(
                effect.operation(),
                AssignmentOperation::Plus | AssignmentOperation::Minus
            );
            match target_is_additive.get_mut(&affected_var_id) {
                None => {
                    target_is_additive.insert(affected_var_id, is_additive);
                    repeated_assignment_targets.push(RepeatedTarget::First);
                }
                Some(previous_are_additive) if *previous_are_additive && is_additive => {
                    repeated_assignment_targets.push(RepeatedTarget::Additive);
                }
                Some(_) => {
                    panic!(
                        "operator {name} writes numeric variable {affected_var_id} more than once \
                         with a non-additive assignment, which has no order-independent result"
                    );
                }
            }
        }

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
            repeated_assignment_targets: repeated_assignment_targets.into_boxed_slice(),
            cost,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn effects(&self) -> &Vec<Effect> {
        &self.effects
    }

    pub fn assignment_effects(&self) -> &Vec<AssignmentEffect> {
        &self.assignment_effects
    }

    pub(crate) fn repeated_assignment_targets(&self) -> &[RepeatedTarget] {
        &self.repeated_assignment_targets
    }

    pub fn preconditions(&self) -> &Vec<ExplicitFact> {
        &self.preconditions
    }

    pub fn cost(&self) -> u64 {
        self.cost
    }
}

#[allow(unused)]
#[derive(Debug, PartialEq)]
pub struct NumericRootTask {
    version: u32,
    metric: Metric,
    variables: Vec<ExplicitVariable>,
    numeric_variables: Vec<NumericVariable>,
    goals: Vec<ExplicitFact>,
    mutexes: Vec<Vec<ExplicitFact>>,
    mutex_pairs: HashSet<(ExplicitFact, ExplicitFact)>,
    state: Vec<usize>,
    numeric_state: Vec<f64>,
    operators: Vec<Operator>,
    operator_costs: Vec<f64>,
    axioms: Vec<PropositionalAxiom>,
    comparison_axioms: Vec<ComparisonAxiom>,
    assignment_axioms: Vec<AssignmentAxiom>,
    numeric_conditions: Arc<NumericConditions>,
    global_constraint: ExplicitFact,
}

/// Everything a root task is built from, before the invariants [`
/// NumericRootTask::new`] establishes over it.
///
/// This is [`crate::sas_format::SasTaskParts`] after the two conversions the
/// format needs -- variables carrying their axiom defaults, operators with
/// their condition lists merged -- and it is what every other producer of a
/// task, from the abstractions to the tests, fills in directly. The fields are
/// the task's own, so nothing here is derived: `new` computes the abstract
/// variable ids, the numeric condition DAG, the fact namespaces and the axiom
/// closure of the initial state, and none of those can be supplied.
pub struct NumericRootTaskParts {
    pub version: u32,
    pub metric: Metric,
    pub variables: Vec<ExplicitVariable>,
    pub numeric_variables: Vec<NumericVariable>,
    pub goals: Vec<ExplicitFact>,
    pub mutexes: Vec<Vec<ExplicitFact>>,
    /// One entry per variable, in variable order. For a derived variable this
    /// is its axiom default rather than its initial value.
    pub state: Vec<usize>,
    pub numeric_state: Vec<f64>,
    pub operators: Vec<Operator>,
    pub axioms: Vec<PropositionalAxiom>,
    pub comparison_axioms: Vec<ComparisonAxiom>,
    pub assignment_axioms: Vec<AssignmentAxiom>,
    pub global_constraint: ExplicitFact,
}

impl NumericRootTask {
    pub fn new(parts: NumericRootTaskParts) -> Self {
        let NumericRootTaskParts {
            version,
            metric,
            mut variables,
            numeric_variables,
            goals,
            mutexes,
            mut state,
            numeric_state,
            operators,
            axioms,
            comparison_axioms,
            assignment_axioms,
            global_constraint,
        } = parts;
        let numeric_conditions = Arc::new(
            NumericConditions::build(
                variables.len(),
                &numeric_variables,
                &comparison_axioms,
                &assignment_axioms,
            )
            .unwrap_or_else(|error| panic!("malformed numeric axioms in SAS task: {error}")),
        );
        narrow_condition_variables(&numeric_conditions, &mut variables, &mut state);
        let mut task = NumericRootTask {
            version,
            metric,
            variables,
            numeric_variables,
            goals,
            mutexes,
            mutex_pairs: HashSet::new(),
            state,
            numeric_state,
            operators,
            operator_costs: Vec::new(),
            axioms,
            comparison_axioms,
            assignment_axioms,
            numeric_conditions,
            global_constraint,
        };
        task.assign_fact_namespaces();
        task.close_initial_state_under_axioms();
        for group in &task.mutexes {
            for (index, &left) in group.iter().enumerate() {
                for &right in &group[index + 1..] {
                    let pair = if left <= right {
                        (left, right)
                    } else {
                        (right, left)
                    };
                    task.mutex_pairs.insert(pair);
                }
            }
        }
        task.operator_costs = task
            .operators
            .iter()
            .map(|operator| metric_operator_cost_from_initial_values(&task, operator))
            .collect();
        task.assert_invariants();
        debug_assert_fact_namespaces(&task);
        task
    }

    /// Assert the cross-field contracts every task consumer relies on.
    fn assert_invariants(&self) {
        assert_eq!(
            self.state.len(),
            self.variables.len(),
            "initial propositional state has {} values for {} variables",
            self.state.len(),
            self.variables.len()
        );
        assert_eq!(
            self.numeric_state.len(),
            self.numeric_variables.len(),
            "initial numeric state has {} values for {} variables",
            self.numeric_state.len(),
            self.numeric_variables.len()
        );

        if let Some(metric_var_id) = self.metric.var_id() {
            let metric_var = self
                .numeric_variables
                .get(metric_var_id)
                .unwrap_or_else(|| {
                    panic!(
                        "metric variable {metric_var_id} is out of bounds for {} numeric variables",
                        self.numeric_variables.len()
                    )
                });
            assert!(
                matches!(
                    metric_var.get_type(),
                    NumericType::Cost | NumericType::Derived
                ),
                "metric variable {metric_var_id} has type {:?}, expected Cost or Derived",
                metric_var.get_type(),
            );
        }

        let assert_task_fact = |fact: &ExplicitFact, origin: &str| {
            let variable = self.variables.get(fact.var()).unwrap_or_else(|| {
                panic!(
                    "{origin} fact names variable {}, past the task's {} propositional variables",
                    fact.var(),
                    self.variables.len()
                )
            });
            assert!(
                fact.value() < variable.domain_size(),
                "{origin} fact value {} is outside variable {}'s domain of size {}",
                fact.value(),
                fact.var(),
                variable.domain_size()
            );
        };
        for goal in &self.goals {
            assert_task_fact(goal, "goal");
        }
        for fact in self.mutexes.iter().flatten() {
            assert_task_fact(fact, "mutex");
        }
        assert_task_fact(&self.global_constraint, "global constraint");

        if !self.comparison_axioms.is_empty() {
            let last_arithmetic_layer = self
                .numeric_variables
                .iter()
                .filter_map(NumericVariable::axiom_layer)
                .max();
            let expected_comparison_layer = last_arithmetic_layer.map_or(0, |layer| layer + 1);
            let mut comparison_layer = None;
            for (axiom_id, axiom) in self.comparison_axioms.iter().enumerate() {
                let head = axiom.get_affected_var_id();
                let layer = self.variables[head].axiom_layer().unwrap_or_else(|| {
                    panic!("comparison axiom {axiom_id} writes non-derived variable {head}")
                });
                if let Some(previous) = comparison_layer {
                    assert_eq!(
                        layer, previous,
                        "comparison axioms occupy both layer {previous} and layer {layer}"
                    );
                } else {
                    comparison_layer = Some(layer);
                }
            }
            let comparison_layer = comparison_layer.unwrap();
            assert_eq!(
                comparison_layer,
                expected_comparison_layer,
                "comparison axiom layer {comparison_layer} must directly follow arithmetic layer {}",
                last_arithmetic_layer.map_or_else(|| "none".to_string(), |layer| layer.to_string())
            );
            let first_derived_propositional_layer = self
                .variables
                .iter()
                .filter_map(ExplicitVariable::axiom_layer)
                .min()
                .expect("a comparison axiom has a derived propositional head");
            assert_eq!(
                first_derived_propositional_layer, comparison_layer,
                "comparison axiom layer {comparison_layer} must be the first derived propositional layer, got {first_derived_propositional_layer}"
            );
        }
    }

    /// Tag every fact the task stores with the namespace of its variable.
    ///
    /// The parser cannot do this: which propositional variables carry numeric
    /// conditions only becomes known when `numeric_conditions` is built, which
    /// happens in [`Self::new`]. So namespace assignment happens exactly once,
    /// over the whole task at once, and no later consumer has to rediscover
    /// the "propositional variable -> comparison axiom" mapping to know what
    /// kind of variable a fact names.
    fn assign_fact_namespaces(&mut self) {
        let conditions = Arc::clone(&self.numeric_conditions);
        let retag = |fact: &mut ExplicitFact| {
            *fact = fact.with_namespace(conditions.namespace_of(fact.var()));
        };

        self.goals.iter_mut().for_each(retag);
        self.mutexes.iter_mut().flatten().for_each(retag);
        for operator in &mut self.operators {
            operator.preconditions.iter_mut().for_each(retag);
            for effect in &mut operator.effects {
                effect.conditions.iter_mut().for_each(retag);
            }
            for effect in &mut operator.assignment_effects {
                effect.conditions.iter_mut().for_each(retag);
            }
        }
        for axiom in &mut self.axioms {
            axiom.conditions_mut().iter_mut().for_each(retag);
        }
        retag(&mut self.global_constraint);
    }

    /// Replace the initial state by its axiom closure.
    ///
    /// The values a SAS file gives a derived variable are not its initial
    /// values but its axiom *defaults*; the real ones follow from the axioms.
    /// Running the closure once here means `get_initial_propositional_state_values`
    /// and `get_initial_numeric_state_values` describe a state the search can
    /// use as it stands, instead of one every consumer has to finish for
    /// itself.
    ///
    /// The closure is a function of the non-derived variables alone, so it is
    /// idempotent: applying it to an already-closed state is a no-op, which is
    /// what makes it safe for a task built out of another task's initial state.
    fn close_initial_state_under_axioms(&mut self) {
        let (propositional, numeric) = self
            .evaluated_initial_abstract_state_values()
            .unwrap_or_else(|error| {
                panic!("initial state does not satisfy the task's own axioms: {error}")
            });
        self.state = propositional;
        self.numeric_state = numeric;
    }

    /// The task's mutex groups.
    ///
    /// The search only ever asks whether two given facts are mutex, which is
    /// what [`AbstractNumericTask::are_facts_mutex`] answers; this is for the
    /// one caller that has to see the groups themselves rather than query them.
    pub fn mutexes(&self) -> &[Vec<ExplicitFact>] {
        &self.mutexes
    }

    /// The fact that must hold in every reachable state.
    ///
    /// The translator injects a global constraint into every task (see
    /// `add_global_constraints`), so one is always present; for tasks without
    /// real global constraints it is a derived atom that is unconditionally
    /// true. The search engines never consult it, so a verifier that does is
    /// strictly stronger than they are.
    pub fn global_constraint(&self) -> &ExplicitFact {
        &self.global_constraint
    }

    pub fn from_file(file_name: impl AsRef<std::path::Path>) -> Self {
        Self::try_from_file(file_name).expect("failed to read numeric SAS task")
    }

    pub fn try_from_file(file_name: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let path = file_name.as_ref();
        let file_content = std::fs::read_to_string(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        Self::try_from_str(&file_content)
    }

    /// Parse a `NumericRootTask` from the preprocessor's text format held in
    /// memory. Equivalent to `try_from_file` minus the disk read; used by the
    /// in-memory translate→preprocess→search pipeline so the binary
    /// `output` file never has to materialize on disk.
    pub fn try_from_str(content: &str) -> Result<Self, String> {
        match parse_numeric_sas_output(content) {
            Ok((_, task)) => Ok(task),
            Err(err) => Err(format!("failed to parse numeric SAS output: {err}")),
        }
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

    fn numeric_conditions(&self) -> &Arc<NumericConditions> {
        &self.numeric_conditions
    }

    fn get_operators(&self) -> &Vec<Operator> {
        &self.operators
    }

    fn goals(&self) -> &[ExplicitFact] {
        &self.goals
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

    fn are_facts_mutex(&self, fact1: &ExplicitFact, fact2: &ExplicitFact) -> bool {
        if fact1.var() == fact2.var() {
            return fact1.value() != fact2.value();
        }
        let pair = if fact1 <= fact2 {
            (*fact1, *fact2)
        } else {
            (*fact2, *fact1)
        };
        self.mutex_pairs.contains(&pair)
    }

    fn abstract_operator_cost(&self, operator_id: usize) -> f64 {
        self.operator_costs[operator_id]
    }

    fn get_operator_cost(&self, index: usize, is_axiom: bool) -> u64 {
        if is_axiom {
            return 0;
        }
        self.operators
            .get(index)
            .unwrap_or_else(|| panic!("operator id {index} is out of bounds for cost lookup"))
            .cost()
    }

    fn get_operator_name(&self, index: usize, is_axiom: bool) -> &str {
        if is_axiom {
            return "<axiom>";
        }
        self.operators
            .get(index)
            .unwrap_or_else(|| panic!("operator id {index} is out of bounds for name lookup"))
            .name()
    }

    fn get_num_operators(&self) -> usize {
        self.operators.len()
    }

    fn get_num_operator_preconditions(&self, index: usize, is_axiom: bool) -> usize {
        if is_axiom {
            // Axioms don't have preconditions in the same way
            return 0;
        }
        self.operators
            .get(index)
            .unwrap_or_else(|| {
                panic!("operator id {index} is out of bounds for precondition lookup")
            })
            .preconditions()
            .len()
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
        self.operators
            .get(index)
            .unwrap_or_else(|| panic!("operator id {index} is out of bounds for effect lookup"))
            .effects()
            .len()
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

    fn get_initial_propositional_state_values(&self) -> &[usize] {
        &self.state
    }

    fn get_initial_numeric_state_values(&self) -> &[f64] {
        &self.numeric_state
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

/// Pin every condition variable to the two-valued [`ConditionValue`] domain.
///
/// A condition variable's domain is fixed by what a comparison can answer, so a
/// task does not get to choose it. SAS files written before the domain shrank
/// declare a third value, `<none of those>`, and name it in the initial-state
/// block as the variable's axiom default. It was never a value a state could
/// hold: the comparison axioms write a verdict for every condition variable
/// before anything reads one, so the placeholder is overwritten by the closure
/// [`NumericRootTask::new`] runs a few lines later. Dropping it here is what
/// makes the domain two everywhere — packed states, abstract states and the
/// per-variable domain mappings the abstractions build on top of them.
fn narrow_condition_variables(
    conditions: &NumericConditions,
    variables: &mut [ExplicitVariable],
    state: &mut [usize],
) {
    /// The value a legacy file's `<none of those>` occupies, one past the domain.
    const LEGACY_PLACEHOLDER: usize = ConditionValue::DOMAIN_SIZE;

    for var_id in conditions.condition_var_ids() {
        let variable = &mut variables[var_id];
        assert!(
            variable.domain_size == ConditionValue::DOMAIN_SIZE
                || variable.domain_size == LEGACY_PLACEHOLDER + 1,
            "variable {var_id} ({}) carries a numeric condition but has domain size {}, \
             which is neither {} nor the legacy {} that adds the placeholder",
            variable.name,
            variable.domain_size,
            ConditionValue::DOMAIN_SIZE,
            LEGACY_PLACEHOLDER + 1
        );
        variable.domain_size = ConditionValue::DOMAIN_SIZE;
        variable.fact_names.truncate(ConditionValue::DOMAIN_SIZE);

        // "Not derived yet" and "does not hold" are the same statement about a
        // state, so the placeholder collapses onto `False`. Nothing else can
        // stand where it stood.
        for value in [&mut variable.axiom_default_value, &mut state[var_id]] {
            assert!(
                *value <= LEGACY_PLACEHOLDER,
                "condition variable {var_id} holds value {value}, which is outside \
                 even the legacy domain of {} values",
                LEGACY_PLACEHOLDER + 1
            );
            if *value == LEGACY_PLACEHOLDER {
                *value = ConditionValue::False.as_usize();
            }
        }
    }
}

fn evaluate_state_with_axiom_closure(
    task: &dyn AbstractNumericTask,
    propositional: &mut [usize],
    numeric: &mut [f64],
) -> Result<(), String> {
    let packer = Arc::new(abstract_propositional_packer(task));
    let mut packed = vec![0u64; packer.num_bins()];
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
        .evaluate(packed, numeric)
        .map_err(|err| format!("failed to evaluate axioms: {err:?}"))?;

    for (var_id, slot) in propositional.iter_mut().enumerate() {
        *slot = packer.get(packed, var_id) as usize;
    }

    Ok(())
}

/// Lives here rather than in `crate::tests` because it has to plant a
/// mistagged fact in a *built* task, and `goals` is private to this module.
#[cfg(test)]
mod namespace_assertion {
    use super::{ExplicitFact, assert_fact_namespaces};

    #[test]
    #[should_panic(expected = "names a domain abstraction's private numeric id space")]
    fn a_task_fact_may_not_name_the_abstraction_id_space() {
        let mut task = crate::tests::get_root_task();
        task.goals[0] = ExplicitFact::numeric_variable(1, 5);
        assert_fact_namespaces(&task);
    }

    #[test]
    #[should_panic(expected = "past the task's 3 propositional variables")]
    fn a_task_fact_may_not_name_a_variable_the_task_does_not_have() {
        let mut task = crate::tests::get_root_task();
        task.goals[0] = ExplicitFact::propositional(3, 0);
        assert_fact_namespaces(&task);
    }
}

#[cfg(test)]
mod root_task_invariants {
    use super::*;
    use crate::axioms::CalOperator;

    fn valid_parts() -> NumericRootTaskParts {
        NumericRootTaskParts {
            version: 4,
            metric: Metric::new(true, Some(0)),
            variables: vec![ExplicitVariable::new(
                2,
                "location".to_string(),
                vec!["here".to_string(), "there".to_string()],
                None,
                0,
            )],
            numeric_variables: vec![NumericVariable::new(
                "total-cost".to_string(),
                NumericType::Cost,
                None,
            )],
            goals: vec![ExplicitFact::propositional(0, 1)],
            mutexes: vec![vec![
                ExplicitFact::propositional(0, 0),
                ExplicitFact::propositional(0, 1),
            ]],
            state: vec![0],
            numeric_state: vec![0.0],
            operators: Vec::new(),
            axioms: Vec::new(),
            comparison_axioms: Vec::new(),
            assignment_axioms: Vec::new(),
            global_constraint: ExplicitFact::propositional(0, 0),
        }
    }

    #[test]
    #[should_panic(expected = "initial propositional state has 0 values for 1 variables")]
    fn rejects_short_propositional_initial_state() {
        let mut parts = valid_parts();
        parts.state.clear();
        NumericRootTask::new(parts);
    }

    #[test]
    #[should_panic(expected = "initial numeric state has 0 values for 1 variables")]
    fn rejects_short_numeric_initial_state() {
        let mut parts = valid_parts();
        parts.numeric_state.clear();
        NumericRootTask::new(parts);
    }

    #[test]
    #[should_panic(expected = "metric variable 1 is out of bounds for 1 numeric variables")]
    fn rejects_out_of_range_metric_variable() {
        let mut parts = valid_parts();
        parts.metric = Metric::new(true, Some(1));
        NumericRootTask::new(parts);
    }

    #[test]
    #[should_panic(expected = "metric variable 0 has type Regular, expected Cost or Derived")]
    fn rejects_regular_metric_variable() {
        let mut parts = valid_parts();
        parts.numeric_variables[0] =
            NumericVariable::new("fuel".to_string(), NumericType::Regular, None);
        NumericRootTask::new(parts);
    }

    #[test]
    #[should_panic(expected = "goal fact value 2 is outside variable 0's domain of size 2")]
    fn rejects_out_of_range_goal_value() {
        let mut parts = valid_parts();
        parts.goals[0] = ExplicitFact::propositional(0, 2);
        NumericRootTask::new(parts);
    }

    #[test]
    #[should_panic(expected = "mutex fact value 2 is outside variable 0's domain of size 2")]
    fn rejects_out_of_range_mutex_value() {
        let mut parts = valid_parts();
        parts.mutexes[0][0] = ExplicitFact::propositional(0, 2);
        NumericRootTask::new(parts);
    }

    #[test]
    #[should_panic(
        expected = "global constraint fact value 2 is outside variable 0's domain of size 2"
    )]
    fn rejects_out_of_range_global_constraint_value() {
        let mut parts = valid_parts();
        parts.global_constraint = ExplicitFact::propositional(0, 2);
        NumericRootTask::new(parts);
    }

    #[test]
    #[should_panic(expected = "comparison axiom layer 2 must directly follow arithmetic layer 0")]
    fn rejects_a_gap_between_arithmetic_and_comparison_layers() {
        let mut parts = valid_parts();
        parts.variables[0] = ExplicitVariable::new(
            ConditionValue::DOMAIN_SIZE,
            "sum-exceeds-left".to_string(),
            vec![
                "sum-exceeds-left".to_string(),
                "not-sum-exceeds-left".to_string(),
            ],
            Some(2),
            ConditionValue::False.as_usize(),
        );
        parts.numeric_variables = vec![
            NumericVariable::new("left".to_string(), NumericType::Constant, None),
            NumericVariable::new("right".to_string(), NumericType::Constant, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
            NumericVariable::new("total-cost".to_string(), NumericType::Cost, None),
        ];
        parts.metric = Metric::new(true, Some(3));
        parts.numeric_state = vec![2.0, 3.0, 0.0, 0.0];
        parts.assignment_axioms = vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)];
        parts.comparison_axioms = vec![ComparisonAxiom::new(
            0,
            2,
            0,
            crate::axioms::ComparisonOperator::GreaterThan,
        )];
        NumericRootTask::new(parts);
    }
}
