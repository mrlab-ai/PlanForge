use super::*;

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
    fn get_variable_name(&self, index: VariableIndex) -> Result<&str, &str>;
    fn get_variable_domain_size(&self, index: VariableIndex) -> Result<usize, &str>;
    fn get_variable_axiom_layer(&self, index: VariableIndex) -> Result<Option<usize>, &str>;
    fn get_variable_default_axiom_value(
        &self,
        index: VariableIndex,
    ) -> Result<ExplicitValueIndex, &str>;
    fn get_fact_name(&self, fact: &ExplicitFact) -> &str;

    fn are_facts_mutex(&self, fact1: &ExplicitFact, fact2: &ExplicitFact) -> bool;

    fn get_operators(&self) -> &Vec<Operator>;
    fn get_operator_cost(&self, index: OperatorIndex, is_axiom: bool) -> OperatorCost;
    fn get_operator_name(&self, index: OperatorIndex, is_axiom: bool) -> &str;
    fn get_num_operators(&self) -> usize;
    fn get_num_operator_preconditions(&self, index: OperatorIndex, is_axiom: bool) -> usize;
    fn get_operator_precondition(
        &self,
        index: OperatorIndex,
        precond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact;
    fn get_num_operator_effects(&self, index: OperatorIndex, is_axiom: bool) -> usize;
    fn get_num_operator_effect_conditions(
        &self,
        index: OperatorIndex,
        eff_index: usize,
        is_axiom: bool,
    ) -> usize;
    fn get_operator_effect_condition(
        &self,
        index: OperatorIndex,
        eff_index: usize,
        cond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact;
    fn get_operator_effect(
        &self,
        index: OperatorIndex,
        eff_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact;

    fn convert_operator_index(&self, index: OperatorIndex, ancestor_task: &dyn AbstractNumericTask);

    fn get_num_axioms(&self) -> usize;
    fn goals(&self) -> &[ExplicitFact];
    fn get_num_goals(&self) -> usize;
    fn get_goal_fact(&self, index: usize) -> &ExplicitFact;

    /// The initial values of the propositional variables, already closed
    /// under the task's axioms.
    fn get_initial_propositional_state_values(&self) -> &[ExplicitValueIndex];
    /// The initial values of the numeric variables, already closed under the
    /// task's axioms.
    fn get_initial_numeric_state_values(&self) -> &[NumericValue];

    fn convert_ancestor_state_values(
        &self,
        ancestor_state_values: &[usize],
        ancestor_task: &dyn AbstractNumericTask,
    ) -> Vec<usize>;

    fn get_num_cmp_axioms(&self) -> usize;

    /// Customization hook used by [`NumericTaskExt::abstract_state_values`].
    fn project_state_values(
        &self,
        propositional_values: &[ExplicitValueIndex],
        numeric_values: &[NumericValue],
    ) -> Result<(Vec<ExplicitValueIndex>, Vec<NumericValue>), String>;

    /// Customization hook used by
    /// [`NumericTaskExt::evaluated_initial_abstract_state_values`].
    fn evaluate_initial_state_values(
        &self,
    ) -> Result<(Vec<ExplicitValueIndex>, Vec<NumericValue>), String>;

    /// Customization hook used by [`NumericTaskExt::abstract_operator_cost`].
    fn operator_cost_for_abstraction(&self, operator_id: OperatorIndex) -> NumericValue;
}

/// Derived task operations shared by every [`AbstractNumericTask`].
///
/// Keeping these methods in one blanket implementation prevents reference
/// wrappers from accidentally inheriting a default instead of forwarding an
/// implementor's projection or cost semantics.
pub trait NumericTaskExt: AbstractNumericTask {
    fn abstract_state_values(
        &self,
        propositional_values: &[ExplicitValueIndex],
        numeric_values: &[NumericValue],
    ) -> Result<(Vec<ExplicitValueIndex>, Vec<NumericValue>), String> {
        self.project_state_values(propositional_values, numeric_values)
    }

    fn evaluated_initial_abstract_state_values(
        &self,
    ) -> Result<(Vec<ExplicitValueIndex>, Vec<NumericValue>), String> {
        self.evaluate_initial_state_values()
    }

    fn abstract_operator_cost(&self, operator_id: OperatorIndex) -> NumericValue {
        self.operator_cost_for_abstraction(operator_id)
    }

    fn min_abstract_operator_cost(&self) -> NumericValue {
        let min_operator_cost = (0..self.get_operators().len())
            .map(|operator_id| self.abstract_operator_cost(OperatorIndex::from_usize(operator_id)))
            .fold(NumericValue::new(f64::INFINITY), |acc, x| {
                NumericValue::new(f64::min(acc.value(), x.value()))
            });
        if min_operator_cost.value().is_finite() {
            NumericValue::new(min_operator_cost.value().max(0.0))
        } else {
            NumericValue::new(0.0)
        }
    }

    fn assignment_axiom_lookup(&self) -> Result<Vec<Option<AxiomIndex>>, NumericConditionError> {
        assignment_axiom_lookup(self.numeric_variables().len(), self.assignment_axioms())
    }

    fn linearize_numeric_var(
        &self,
        numeric_var_id: VariableIndex,
    ) -> Result<crate::utils::linear_effects::LinearExpression, LinearizationError> {
        linearize_numeric_var(self, numeric_var_id)
    }

    fn linearized_assignment_effects(
        &self,
        operator_id: OperatorIndex,
    ) -> Result<Vec<LinearNumericEffect>, LinearizationError> {
        linearize_operator_assignment_effects(self, operator_id)
    }

    fn regular_numeric_variable_ids(&self) -> Vec<VariableIndex> {
        self.numeric_variables()
            .iter()
            .enumerate()
            .filter_map(|(numeric_var_id, numeric_var)| {
                (numeric_var.get_type() == &NumericType::Regular)
                    .then_some(VariableIndex::from_usize(numeric_var_id))
            })
            .collect()
    }

    fn is_linear_cost_operator(&self, operator_id: OperatorIndex) -> bool {
        linear_metric_operator_cost_expression(self, operator_id).is_some()
    }

    fn operator_cost_coefficients(&self, operator_id: OperatorIndex) -> Vec<f64> {
        let regular_numeric_variable_ids = self.regular_numeric_variable_ids();
        linear_metric_operator_cost_expression(self, operator_id)
            .map(|expression| {
                regular_numeric_variable_ids
                    .iter()
                    .map(|&numeric_var_id| expression.coefficients[numeric_var_id.index()])
                    .collect()
            })
            .unwrap_or_else(|| {
                todo!(
                    "requested linear action-cost coefficients for non-linear-cost operator {operator_id}"
                )
            })
    }

    fn operator_cost_constant(&self, operator_id: OperatorIndex) -> f64 {
        linear_metric_operator_cost_expression(self, operator_id)
            .map(|expression| expression.constant)
            .unwrap_or_else(|| {
                todo!(
                    "requested linear action-cost constant for non-linear-cost operator {operator_id}"
                )
            })
    }
}

impl<T: AbstractNumericTask + ?Sized> NumericTaskExt for T {}

fn identity_state_values(
    task: &(impl AbstractNumericTask + ?Sized),
    propositional_values: &[ExplicitValueIndex],
    numeric_values: &[NumericValue],
) -> Result<(Vec<ExplicitValueIndex>, Vec<NumericValue>), String> {
    if propositional_values.len() != task.variables().len() {
        return Err(format!(
            "expected {} propositional values, got {}",
            task.variables().len(),
            propositional_values.len()
        ));
    }
    if numeric_values.len() != task.numeric_variables().len() {
        return Err(format!(
            "expected {} numeric values, got {}",
            task.numeric_variables().len(),
            numeric_values.len()
        ));
    }
    Ok((propositional_values.to_vec(), numeric_values.to_vec()))
}

pub fn evaluate_metric_from_values<T: AbstractNumericTask + ?Sized>(
    task: &T,
    numeric_values: &[NumericValue],
) -> NumericValue {
    let metric_var_id = task.metric().var_id();
    match metric_var_id {
        Some(var_id) => *numeric_values.get(var_id.index()).unwrap_or_else(|| {
            panic!(
                "metric variable {} is out of bounds for {} numeric values",
                var_id.index(),
                numeric_values.len()
            )
        }),
        None => NumericValue::new(0.0),
    }
}

pub fn propagate_assignment_axiom_values<T: AbstractNumericTask + ?Sized>(
    task: &T,
    numeric_values: &mut [NumericValue],
) -> Result<(), AssignmentAxiomError> {
    // Assignment axioms are stored in dependency-layer order, so each RHS is
    // complete when it is visited and one forward pass closes the values.
    for axiom in task.assignment_axioms() {
        let affected_var_id = axiom.get_affected_var_id().index();
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
) -> NumericValue {
    if !task.metric().use_metric() {
        return NumericValue::new(operator.cost().value() as f64);
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
            (assignment_var_id.index()) < numeric_values.len(),
            "assignment variable {} of operator {} is out of bounds for {} numeric variables",
            assignment_var_id.index(),
            operator.name(),
            numeric_values.len(),
        );
        assert!(
            (affected_var_id.index()) < numeric_values.len(),
            "affected variable {} of operator {} is out of bounds for {} numeric variables",
            affected_var_id.index(),
            operator.name(),
            numeric_values.len(),
        );

        let result = AssignmentOperation::apply(
            numeric_values[affected_var_id.index()],
            effect.operation(),
            numeric_values[assignment_var_id.index()],
        );
        results.push((affected_var_id, result));
    }
    for (affected_var_id, result) in results {
        numeric_values[affected_var_id.index()] = result;
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
        new_metric.value() - old_metric.value()
    } else {
        old_metric.value() - new_metric.value()
    };
    assert!(
        delta >= 0.0,
        "operator {} has negative metric cost {delta}, which search does not support",
        operator.name()
    );
    NumericValue::new(delta)
}

fn linear_metric_operator_cost_expression<T: AbstractNumericTask + ?Sized>(
    task: &T,
    operator_id: OperatorIndex,
) -> Option<crate::utils::linear_effects::LinearExpression> {
    if !task.metric().use_metric() {
        return None;
    }

    let metric_var_id = task.metric().var_id().unwrap();
    let metric_variable = task.numeric_variables().get(metric_var_id.index())?;
    if metric_variable.get_type() != &NumericType::Cost {
        return None;
    }

    let operator = task
        .get_operators()
        .get(operator_id.index())
        .unwrap_or_else(|| {
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
            .linearize_numeric_var(assignment_effect.var_id())
            .unwrap_or_else(|error| {
                panic!(
                    "failed to linearize metric-cost source variable {} for operator {operator_id}: {error}",
                    assignment_effect.var_id().index()
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
/// Every required task hook forwards to the referent. Derived operations live
/// in the single [`NumericTaskExt`] blanket implementation, so there are no
/// default methods here that a reference wrapper can accidentally inherit.
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
    fn get_variable_name(&self, index: VariableIndex) -> Result<&str, &str> {
        (**self).get_variable_name(index)
    }
    fn get_variable_domain_size(&self, index: VariableIndex) -> Result<usize, &str> {
        (**self).get_variable_domain_size(index)
    }
    fn get_variable_axiom_layer(&self, index: VariableIndex) -> Result<Option<usize>, &str> {
        (**self).get_variable_axiom_layer(index)
    }
    fn get_variable_default_axiom_value(
        &self,
        index: VariableIndex,
    ) -> Result<ExplicitValueIndex, &str> {
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
    fn get_operator_cost(&self, index: OperatorIndex, is_axiom: bool) -> OperatorCost {
        (**self).get_operator_cost(index, is_axiom)
    }
    fn get_operator_name(&self, index: OperatorIndex, is_axiom: bool) -> &str {
        (**self).get_operator_name(index, is_axiom)
    }
    fn get_num_operators(&self) -> usize {
        (**self).get_num_operators()
    }
    fn get_num_operator_preconditions(&self, index: OperatorIndex, is_axiom: bool) -> usize {
        (**self).get_num_operator_preconditions(index, is_axiom)
    }
    fn get_operator_precondition(
        &self,
        index: OperatorIndex,
        precond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact {
        (**self).get_operator_precondition(index, precond_index, is_axiom)
    }
    fn get_num_operator_effects(&self, index: OperatorIndex, is_axiom: bool) -> usize {
        (**self).get_num_operator_effects(index, is_axiom)
    }
    fn get_num_operator_effect_conditions(
        &self,
        index: OperatorIndex,
        eff_index: usize,
        is_axiom: bool,
    ) -> usize {
        (**self).get_num_operator_effect_conditions(index, eff_index, is_axiom)
    }
    fn get_operator_effect_condition(
        &self,
        index: OperatorIndex,
        eff_index: usize,
        cond_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact {
        (**self).get_operator_effect_condition(index, eff_index, cond_index, is_axiom)
    }
    fn get_operator_effect(
        &self,
        index: OperatorIndex,
        eff_index: usize,
        is_axiom: bool,
    ) -> &ExplicitFact {
        (**self).get_operator_effect(index, eff_index, is_axiom)
    }
    fn convert_operator_index(
        &self,
        index: OperatorIndex,
        ancestor_task: &dyn AbstractNumericTask,
    ) {
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
    fn get_initial_propositional_state_values(&self) -> &[ExplicitValueIndex] {
        (**self).get_initial_propositional_state_values()
    }
    fn get_initial_numeric_state_values(&self) -> &[NumericValue] {
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
    fn project_state_values(
        &self,
        propositional_values: &[ExplicitValueIndex],
        numeric_values: &[NumericValue],
    ) -> Result<(Vec<ExplicitValueIndex>, Vec<NumericValue>), String> {
        (**self).project_state_values(propositional_values, numeric_values)
    }
    fn evaluate_initial_state_values(
        &self,
    ) -> Result<(Vec<ExplicitValueIndex>, Vec<NumericValue>), String> {
        (**self).evaluate_initial_state_values()
    }
    fn operator_cost_for_abstraction(&self, operator_id: OperatorIndex) -> NumericValue {
        (**self).operator_cost_for_abstraction(operator_id)
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
        let expected = conditions.namespace_of(fact.var_index());
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
    state: Vec<ExplicitValueIndex>,
    numeric_state: Vec<NumericValue>,
    operators: Vec<Operator>,
    operator_costs: Vec<NumericValue>,
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
    pub state: Vec<ExplicitValueIndex>,
    pub numeric_state: Vec<NumericValue>,
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
                .get(metric_var_id.index())
                .unwrap_or_else(|| {
                    panic!(
                        "metric variable {} is out of bounds for {} numeric variables",
                        metric_var_id.index(),
                        self.numeric_variables.len()
                    )
                });
            assert!(
                matches!(
                    metric_var.get_type(),
                    NumericType::Cost | NumericType::Derived
                ),
                "metric variable {} has type {:?}, expected Cost or Derived",
                metric_var_id.index(),
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
                let head = axiom.get_affected_var_id().index();
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
            *fact = fact.with_namespace(conditions.namespace_of(fact.var_index()));
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

impl AbstractNumericTask for NumericRootTask {
    fn project_state_values(
        &self,
        propositional_values: &[ExplicitValueIndex],
        numeric_values: &[NumericValue],
    ) -> Result<(Vec<ExplicitValueIndex>, Vec<NumericValue>), String> {
        identity_state_values(self, propositional_values, numeric_values)
    }

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

    fn get_variable_name(&self, index: VariableIndex) -> Result<&str, &str> {
        if index.index() >= (self.variables.len()) {
            return Err("Index out of bounds");
        }
        Ok(&self.variables[index.index()].name)
    }

    fn get_variable_domain_size(&self, index: VariableIndex) -> Result<usize, &str> {
        if index.index() >= (self.variables.len()) {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index.index()].domain_size)
    }

    fn get_variable_axiom_layer(&self, index: VariableIndex) -> Result<Option<usize>, &str> {
        if index.index() >= (self.variables.len()) {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index.index()].axiom_layer)
    }

    fn get_variable_default_axiom_value(
        &self,
        index: VariableIndex,
    ) -> Result<ExplicitValueIndex, &str> {
        if index.index() >= (self.variables.len()) {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index.index()].axiom_default_value)
    }

    fn get_fact_name(&self, fact: &ExplicitFact) -> &str {
        // Only facts on genuine propositional variables name a ground atom. A
        // condition fact carries a numeric condition's truth value and a
        // numeric-variable fact a partition index; both reuse the variable-id
        // bits for their own id space, so indexing `variables` with them would
        // return another variable's atom name.
        if fact.namespace() != FactNamespace::Propositional {
            return "";
        }
        let variable = self.variables.get(fact.var()).unwrap_or_else(|| {
            panic!(
                "propositional fact names variable {} but the task has {}",
                fact.var(),
                self.variables.len()
            )
        });
        // A task assembled in memory rather than parsed may carry no names.
        variable
            .fact_names
            .get(fact.value())
            .map_or("", String::as_str)
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

    fn operator_cost_for_abstraction(&self, operator_id: OperatorIndex) -> NumericValue {
        self.operator_costs[operator_id.index()]
    }

    fn get_operator_cost(&self, index: OperatorIndex, is_axiom: bool) -> OperatorCost {
        if is_axiom {
            return ZERO_OP_COST;
        }
        self.operators
            .get(index.index())
            .unwrap_or_else(|| {
                panic!(
                    "operator id {} is out of bounds for cost lookup",
                    index.index()
                )
            })
            .cost()
    }

    fn get_operator_name(&self, index: OperatorIndex, is_axiom: bool) -> &str {
        if is_axiom {
            return "<axiom>";
        }
        self.operators
            .get(index.index())
            .unwrap_or_else(|| {
                panic!(
                    "operator id {} is out of bounds for name lookup",
                    index.index()
                )
            })
            .name()
    }

    fn get_num_operators(&self) -> usize {
        self.operators.len()
    }

    fn get_num_operator_preconditions(&self, index: OperatorIndex, is_axiom: bool) -> usize {
        if is_axiom {
            // Axioms don't have preconditions in the same way
            return 0;
        }
        self.operators
            .get(index.index())
            .unwrap_or_else(|| {
                panic!(
                    "operator id {} is out of bounds for precondition lookup",
                    index.index()
                )
            })
            .preconditions()
            .len()
    }

    fn get_operator_precondition(
        &self,
        _index: OperatorIndex,
        _precond_index: usize,
        _is_axiom: bool,
    ) -> &ExplicitFact {
        unimplemented!("This function is not yet implemented");
    }

    fn get_num_operator_effects(&self, index: OperatorIndex, is_axiom: bool) -> usize {
        if is_axiom {
            // Handle axiom effects differently.
            return 0;
        }
        self.operators
            .get(index.index())
            .unwrap_or_else(|| {
                panic!(
                    "operator id {} is out of bounds for effect lookup",
                    index.index()
                )
            })
            .effects()
            .len()
    }

    fn get_num_operator_effect_conditions(
        &self,
        _index: OperatorIndex,
        _eff_index: usize,
        _is_axiom: bool,
    ) -> usize {
        0
    }

    fn get_operator_effect_condition(
        &self,
        _index: OperatorIndex,
        _eff_index: usize,
        _cond_index: usize,
        _is_axiom: bool,
    ) -> &ExplicitFact {
        unimplemented!("This function is not yet implemented");
    }

    fn get_operator_effect(
        &self,
        _index: OperatorIndex,
        _eff_index: usize,
        _is_axiom: bool,
    ) -> &ExplicitFact {
        unimplemented!("This function is not yet implemented");
    }

    fn convert_operator_index(
        &self,
        _index: OperatorIndex,
        _ancestor_task: &dyn AbstractNumericTask,
    ) {
    }

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

    fn get_initial_propositional_state_values(&self) -> &[ExplicitValueIndex] {
        &self.state
    }

    fn get_initial_numeric_state_values(&self) -> &[NumericValue] {
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

    fn evaluate_initial_state_values(
        &self,
    ) -> Result<(Vec<ExplicitValueIndex>, Vec<NumericValue>), String> {
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
    state: &mut [ExplicitValueIndex],
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
                value.index() <= LEGACY_PLACEHOLDER,
                "condition variable {var_id} holds value {}, which is outside \
                 even the legacy domain of {} values",
                value.index(),
                LEGACY_PLACEHOLDER + 1
            );
            if value.index() == LEGACY_PLACEHOLDER {
                *value = ExplicitValueIndex::new(ConditionValue::False.as_u32());
            }
        }
    }
}

fn evaluate_state_with_axiom_closure(
    task: &dyn AbstractNumericTask,
    propositional: &mut [ExplicitValueIndex],
    numeric: &mut [NumericValue],
) -> Result<(), String> {
    let packer = Arc::new(abstract_propositional_packer(task));
    let mut packed = vec![0u64; packer.num_bins()];
    for (var_id, value) in propositional.iter().enumerate() {
        packer.set(&mut packed, var_id, value.index() as u64);
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

fn abstract_propositional_packer<T: AbstractNumericTask + ?Sized>(task: &T) -> StatePacker {
    let ranges: Vec<u64> = task
        .variables()
        .iter()
        .map(|variable| variable.domain_size() as u64)
        .collect();
    StatePacker::new(&ranges)
}

fn finish_axiom_closure(
    packer: &StatePacker,
    propositional: &mut [ExplicitValueIndex],
    numeric: &mut [NumericValue],
    packed: &mut [u64],
    axiom_evaluator: &AxiomEvaluator<'_>,
) -> Result<(), String> {
    axiom_evaluator
        .evaluate(packed, numeric)
        .map_err(|err| format!("failed to evaluate axioms: {err:?}"))?;

    for (var_id, slot) in propositional.iter_mut().enumerate() {
        *slot = ExplicitValueIndex::new(packer.get(packed, var_id) as u32);
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
            metric: Metric::new(true, Some(VariableIndex::new(0))),
            variables: vec![ExplicitVariable::new(
                2,
                "location".to_string(),
                vec!["here".to_string(), "there".to_string()],
                None,
                ExplicitValueIndex::new(0),
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
            state: vec![ExplicitValueIndex::new(0)],
            numeric_state: vec![NumericValue::new(0.0)],
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
        parts.metric = Metric::new(true, Some(VariableIndex::new(1)));
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
            ExplicitValueIndex::new(ConditionValue::False.as_u32()),
        );
        parts.numeric_variables = vec![
            NumericVariable::new("left".to_string(), NumericType::Constant, None),
            NumericVariable::new("right".to_string(), NumericType::Constant, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
            NumericVariable::new("total-cost".to_string(), NumericType::Cost, None),
        ];
        parts.metric = Metric::new(true, Some(VariableIndex::new(3)));
        parts.numeric_state = vec![
            NumericValue::new(2.0),
            NumericValue::new(3.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
        ];
        parts.assignment_axioms = vec![AssignmentAxiom::new(
            VariableIndex::new(2),
            CalOperator::Sum,
            VariableIndex::new(0),
            VariableIndex::new(1),
        )];
        parts.comparison_axioms = vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(2),
            VariableIndex::new(0),
            crate::axioms::ComparisonOperator::GreaterThan,
        )];
        NumericRootTask::new(parts);
    }
}
