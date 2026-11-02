mod linear_normal_form;
mod metric;
mod namespaces;

use super::*;
use metric::linear_metric_operator_cost_expression;
pub use metric::{
    evaluate_metric_from_values, metric_operator_costs, propagate_assignment_axiom_values,
};
pub use namespaces::{assert_fact_namespaces, debug_assert_fact_namespaces};
use std::sync::Arc;

/// Why a task could not be loaded from the preprocessor's text format.
#[derive(Debug)]
pub enum TaskLoadError {
    /// The file could not be read.
    Read {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    /// The text is not a numeric SAS task; the parser's message.
    Parse(String),
}

impl std::fmt::Display for TaskLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read { path, source } => write!(f, "failed to read {}: {source}", path.display()),
            Self::Parse(message) => write!(f, "failed to parse numeric SAS output: {message}"),
        }
    }
}

impl std::error::Error for TaskLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::Parse(_) => None,
        }
    }
}

/// A planning task: its goal, and the goal-independent rest it shares with
/// the tasks [`Task::with_goals`] derives from it.
///
/// A task is immutable once built. Everything but the goal sits behind one
/// `Arc`, so cloning a task or giving it another goal copies no operator.
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    data: Arc<TaskData>,
    goals: Arc<[ExplicitFact]>,
}

/// Everything of a [`Task`] but its goal.
#[derive(Debug, PartialEq)]
pub(crate) struct TaskData {
    metric: Metric,
    variables: Vec<ExplicitVariable>,
    numeric_variables: Vec<NumericVariable>,
    mutexes: Vec<Vec<ExplicitFact>>,
    mutex_pairs: HashSet<(ExplicitFact, ExplicitFact)>,
    state: Vec<ExplicitValueIndex>,
    numeric_state: Vec<NumericValue>,
    operators: Vec<Operator>,
    operator_costs: Vec<f64>,
    axioms: Vec<PropositionalAxiom>,
    comparison_axioms: Vec<ComparisonAxiom>,
    assignment_axioms: Vec<AssignmentAxiom>,
    numeric_conditions: Arc<NumericConditions>,
    global_constraint: Option<ExplicitFact>,
}

/// Everything a task is built from, before the invariants [`Task::new`]
/// establishes over it.
///
/// This is [`crate::sas_format::SasTaskParts`] after the two conversions the
/// format needs -- variables carrying their axiom defaults, operators with
/// their condition lists merged -- and it is what every other producer of a
/// task, from the abstractions to the tests, fills in directly. The fields are
/// the task's own, so nothing here is derived: `new` computes the abstract
/// variable ids, the numeric condition DAG, the fact namespaces and the axiom
/// closure of the initial state, and none of those can be supplied.
pub struct TaskParts {
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
    /// The fact the translator requires to hold in every reachable state;
    /// a task derived from another, such as a projection, has none.
    pub global_constraint: Option<ExplicitFact>,
}

impl Task {
    pub fn new(parts: TaskParts) -> Self {
        let TaskParts {
            metric,
            variables,
            numeric_variables,
            mut goals,
            mutexes,
            state,
            numeric_state,
            operators,
            axioms,
            comparison_axioms,
            assignment_axioms,
            global_constraint,
        } = parts;
        let mut data = TaskData {
            metric,
            variables,
            numeric_variables,
            mutexes,
            mutex_pairs: HashSet::new(),
            state,
            numeric_state,
            operators,
            operator_costs: Vec::new(),
            axioms,
            comparison_axioms,
            assignment_axioms,
            numeric_conditions: Arc::new(NumericConditions::default()),
            global_constraint,
        };
        data.assert_invariants(&goals);
        data.normalize_linear_comparisons();
        data.numeric_conditions = Arc::new(
            NumericConditions::build(
                data.variables.len(),
                &data.numeric_variables,
                &data.comparison_axioms,
                &data.assignment_axioms,
            )
            .unwrap_or_else(|error| panic!("malformed numeric axioms in SAS task: {error}")),
        );
        narrow_condition_variables(
            &data.numeric_conditions,
            &mut data.variables,
            &mut data.state,
        );
        data.assign_fact_namespaces(&mut goals);
        data.assert_invariants(&goals);
        for group in &data.mutexes {
            for (index, &left) in group.iter().enumerate() {
                for &right in &group[index + 1..] {
                    let pair = if left <= right {
                        (left, right)
                    } else {
                        (right, left)
                    };
                    data.mutex_pairs.insert(pair);
                }
            }
        }
        let mut task = Task {
            data: Arc::new(data),
            goals: goals.into(),
        };
        // The values a SAS file gives a derived variable are not its initial
        // values but its axiom *defaults*; the real ones follow from the
        // axioms. Closing the initial state once here means the initial values
        // describe a state the search can use as it stands. The closure is a
        // function of the non-derived variables alone, so it is idempotent,
        // which makes it safe for a task built out of another task's initial
        // state. The metric's operator costs are charged in the closed state.
        let (propositional, numeric) =
            task.evaluate_initial_state_values()
                .unwrap_or_else(|error| {
                    panic!("initial state does not satisfy the task's own axioms: {error}")
                });
        task.data_mut().state = propositional;
        task.data_mut().numeric_state = numeric;
        let operator_costs = metric_operator_costs(&task);
        task.data_mut().operator_costs = operator_costs;
        debug_assert_fact_namespaces(&task);
        task
    }

    /// Everything of the task but its goal, for the crate's helpers that do
    /// not read the goal.
    #[cfg(test)]
    pub(crate) fn data(&self) -> &TaskData {
        &self.data
    }

    /// The task data while the task is built and not yet shared.
    fn data_mut(&mut self) -> &mut TaskData {
        Arc::get_mut(&mut self.data).expect("a task is not shared before it is built")
    }

    /// This task with `goals` as its goal; everything else is shared.
    ///
    /// # Panics
    /// Panics unless every goal is a fact of the task tagged with its
    /// variable's namespace.
    pub fn with_goals(&self, goals: Vec<ExplicitFact>) -> Self {
        for goal in &goals {
            self.data.assert_task_fact(goal, "goal");
            assert_eq!(
                goal.namespace(),
                self.data.numeric_conditions.namespace_of(goal.var_index()),
                "goal {goal:?} is tagged with another namespace than its variable's"
            );
        }
        Self {
            data: Arc::clone(&self.data),
            goals: goals.into(),
        }
    }

    /// The task with `operator_costs` charged instead of the costs its metric
    /// implies, for a task derived from another that keeps the other's costs
    /// while dropping the variables its metric reads.
    ///
    /// # Panics
    /// Panics unless there is one finite, non-negative cost per operator, or
    /// if the task is already shared.
    pub fn with_operator_costs(mut self, operator_costs: Vec<f64>) -> Self {
        assert_eq!(
            operator_costs.len(),
            self.data.operators.len(),
            "one cost per operator"
        );
        for (operator_id, &cost) in operator_costs.iter().enumerate() {
            assert!(
                cost.is_finite() && cost >= 0.0,
                "operator {operator_id} costs {cost}, not a finite non-negative cost"
            );
        }
        self.data_mut().operator_costs = operator_costs;
        self
    }

    pub fn from_file(file_name: impl AsRef<std::path::Path>) -> Self {
        Self::try_from_file(file_name).expect("failed to read numeric SAS task")
    }

    pub fn try_from_file(file_name: impl AsRef<std::path::Path>) -> Result<Self, TaskLoadError> {
        let path = file_name.as_ref();
        let file_content = std::fs::read_to_string(path).map_err(|source| TaskLoadError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Self::try_from_str(&file_content)
    }

    /// Parse a task from the preprocessor's text format held in memory.
    /// Equivalent to `try_from_file` minus the disk read; used by the
    /// in-memory translate→preprocess→search pipeline so the binary
    /// `output` file never has to materialize on disk.
    pub fn try_from_str(content: &str) -> Result<Self, TaskLoadError> {
        match parse_numeric_sas_output(content) {
            Ok((_, task)) => Ok(task),
            Err(err) => Err(TaskLoadError::Parse(err.to_string())),
        }
    }

    pub fn variables(&self) -> &[ExplicitVariable] {
        &self.data.variables
    }

    pub fn numeric_variables(&self) -> &[NumericVariable] {
        &self.data.numeric_variables
    }

    /// The propositional variable `var`.
    pub fn variable(&self, var: VariableIndex) -> &ExplicitVariable {
        &self.data.variables[var.index()]
    }

    /// The type of numeric variable `var`.
    pub fn numeric_type(&self, var: VariableIndex) -> NumericType {
        self.data.numeric_variables[var.index()].numeric_type()
    }

    /// What determines the value of propositional variable `var`.
    pub fn variable_kind(&self, var: VariableIndex) -> VariableKind {
        if self.data.numeric_conditions.is_condition_var(var) {
            VariableKind::Condition
        } else if self.variable(var).axiom_layer().is_some() {
            VariableKind::Derived
        } else {
            VariableKind::Regular
        }
    }

    pub fn assignment_axioms(&self) -> &[AssignmentAxiom] {
        &self.data.assignment_axioms
    }

    pub fn comparison_axioms(&self) -> &[ComparisonAxiom] {
        &self.data.comparison_axioms
    }

    /// The task's numeric conditions, one per comparison axiom: the only
    /// place the "propositional variable -> comparison axiom" mapping lives.
    /// Behind an `Arc` so components that outlive a borrow of the task can
    /// share them.
    pub fn numeric_conditions(&self) -> &Arc<NumericConditions> {
        &self.data.numeric_conditions
    }

    pub fn axioms(&self) -> &[PropositionalAxiom] {
        &self.data.axioms
    }

    pub fn metric(&self) -> &Metric {
        &self.data.metric
    }

    pub fn operators(&self) -> &[Operator] {
        &self.data.operators
    }

    pub fn goals(&self) -> &[ExplicitFact] {
        &self.goals
    }

    pub fn fact_name(&self, fact: &ExplicitFact) -> &str {
        // Only facts on genuine propositional variables name a ground atom. A
        // condition fact carries a numeric condition's truth value and a
        // numeric-variable fact a partition index; both reuse the variable-id
        // bits for their own id space, so indexing `variables` with them would
        // return another variable's atom name.
        if fact.namespace() != FactNamespace::Propositional {
            return "";
        }
        let variable = self.data.variables.get(fact.var()).unwrap_or_else(|| {
            panic!(
                "propositional fact names variable {} but the task has {}",
                fact.var(),
                self.data.variables.len()
            )
        });
        // A task assembled in memory rather than parsed may carry no names.
        variable
            .fact_names
            .get(fact.value())
            .map_or("", String::as_str)
    }

    /// The mutex groups the translator found: within a group, at most one
    /// fact holds in any reachable state. [`Self::are_facts_mutex`] answers
    /// the pairwise question; a task derived from this one restricts the
    /// groups to its own variables.
    pub fn mutexes(&self) -> &[Vec<ExplicitFact>] {
        &self.data.mutexes
    }

    pub fn are_facts_mutex(&self, fact1: &ExplicitFact, fact2: &ExplicitFact) -> bool {
        if fact1.var() == fact2.var() {
            return fact1.value() != fact2.value();
        }
        let pair = if fact1 <= fact2 {
            (*fact1, *fact2)
        } else {
            (*fact2, *fact1)
        };
        self.data.mutex_pairs.contains(&pair)
    }

    /// The cost of every operator, in task order.
    pub fn effective_operator_costs(&self) -> &[f64] {
        &self.data.operator_costs
    }

    /// What applying `operator_id` costs: its declared cost, or the metric
    /// increase it causes in the initial state.
    pub fn operator_cost(&self, operator_id: OperatorIndex) -> f64 {
        self.data.operator_costs[operator_id.index()]
    }

    /// The initial values of the propositional variables, closed under the
    /// task's axioms.
    pub fn initial_propositional_values(&self) -> &[ExplicitValueIndex] {
        &self.data.state
    }

    /// The initial values of the numeric variables, closed under the task's
    /// axioms.
    pub fn initial_numeric_values(&self) -> &[NumericValue] {
        &self.data.numeric_state
    }

    /// The fact that must hold in every reachable state.
    ///
    /// The translator injects a global constraint into every task it writes
    /// (see `add_global_constraints`); for tasks without real global
    /// constraints it is a derived atom that is unconditionally true. A task
    /// derived from another, such as a projection, has none. The search
    /// engines never consult it, so a verifier that does is strictly stronger
    /// than they are.
    pub fn global_constraint(&self) -> Option<&ExplicitFact> {
        self.data.global_constraint.as_ref()
    }

    pub fn assignment_axiom_lookup(
        &self,
    ) -> Result<Vec<Option<AxiomIndex>>, NumericConditionError> {
        self.data.assignment_axiom_lookup()
    }

    pub fn linearize_numeric_var(
        &self,
        numeric_var_id: VariableIndex,
    ) -> Result<crate::utils::linear_effects::LinearExpression, LinearizationError> {
        self.data.linearize_numeric_var(numeric_var_id)
    }

    pub fn linearized_assignment_effects(
        &self,
        operator_id: OperatorIndex,
    ) -> Result<Vec<LinearNumericEffect>, LinearizationError> {
        linearize_operator_assignment_effects(&self.data, operator_id)
    }

    pub fn regular_numeric_variable_ids(&self) -> Vec<VariableIndex> {
        self.data
            .numeric_variables
            .iter()
            .enumerate()
            .filter_map(|(numeric_var_id, numeric_var)| {
                (numeric_var.numeric_type() == NumericType::Regular)
                    .then_some(VariableIndex::from_usize(numeric_var_id))
            })
            .collect()
    }

    pub fn is_linear_cost_operator(&self, operator_id: OperatorIndex) -> bool {
        linear_metric_operator_cost_expression(&self.data, operator_id).is_some()
    }

    pub fn operator_cost_coefficients(&self, operator_id: OperatorIndex) -> Vec<f64> {
        let regular_numeric_variable_ids = self.regular_numeric_variable_ids();
        linear_metric_operator_cost_expression(&self.data, operator_id)
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

    pub fn operator_cost_constant(&self, operator_id: OperatorIndex) -> f64 {
        linear_metric_operator_cost_expression(&self.data, operator_id)
            .map(|expression| expression.constant)
            .unwrap_or_else(|| {
                todo!(
                    "requested linear action-cost constant for non-linear-cost operator {operator_id}"
                )
            })
    }

    pub fn evaluate_initial_state_values(
        &self,
    ) -> Result<(Vec<ExplicitValueIndex>, Vec<NumericValue>), AssignmentAxiomError> {
        // Initial values are canonical by construction, as are
        // `ExplicitStateSpace::initial_state`.
        let mut state = ExplicitState {
            propositional: self.initial_propositional_values().to_vec(),
            numeric: self.initial_numeric_values().to_vec(),
        };
        ExplicitStateSpace::new(self).close_under_axioms(&mut state)?;
        Ok((state.propositional, state.numeric))
    }
}

impl TaskData {
    pub(crate) fn numeric_variables(&self) -> &[NumericVariable] {
        &self.numeric_variables
    }

    pub(crate) fn assignment_axioms(&self) -> &[AssignmentAxiom] {
        &self.assignment_axioms
    }

    pub(crate) fn operators(&self) -> &[Operator] {
        &self.operators
    }

    pub(crate) fn metric(&self) -> &Metric {
        &self.metric
    }

    pub(crate) fn initial_numeric_values(&self) -> &[NumericValue] {
        &self.numeric_state
    }

    pub(crate) fn assignment_axiom_lookup(
        &self,
    ) -> Result<Vec<Option<AxiomIndex>>, NumericConditionError> {
        assignment_axiom_lookup(self.numeric_variables.len(), &self.assignment_axioms)
    }

    pub(crate) fn linearize_numeric_var(
        &self,
        numeric_var_id: VariableIndex,
    ) -> Result<crate::utils::linear_effects::LinearExpression, LinearizationError> {
        linearize_numeric_var(self, numeric_var_id)
    }

    /// Assert the cross-field contracts every task consumer relies on.
    fn assert_invariants(&self, goals: &[ExplicitFact]) {
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

        let numeric_count = self.numeric_variables.len();
        let assert_numeric_operand = |variable: VariableIndex, origin: &str| {
            assert!(
                variable.index() < numeric_count,
                "{origin} names numeric variable {variable}, past the task's {numeric_count} \
                 numeric variables"
            );
        };
        for axiom in &self.assignment_axioms {
            let head = axiom.affected_var_id();
            assert_numeric_operand(head, "an assignment axiom head");
            for operand in axiom.expression.dependencies() {
                assert_numeric_operand(operand, &format!("assignment axiom {head}"));
            }
        }
        for (axiom_id, axiom) in self.comparison_axioms.iter().enumerate() {
            for operand in [axiom.left_var_id(), axiom.right_var_id()] {
                assert_numeric_operand(operand, &format!("comparison axiom {axiom_id}"));
            }
        }

        for axiom in &self.assignment_axioms {
            if let crate::axioms::NumericExpression::Linear(terms) = &axiom.expression {
                for &(variable, _) in terms {
                    assert!(
                        self.numeric_variables
                            .get(variable.index())
                            .is_some_and(
                                |definition| definition.numeric_type() == NumericType::Regular
                            ),
                        "linear sum {} reads non-regular numeric variable {variable}",
                        axiom.affected_var_id()
                    );
                }
            }
        }

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
                    metric_var.numeric_type(),
                    NumericType::Cost | NumericType::Derived
                ),
                "metric variable {} has type {:?}, expected Cost or Derived",
                metric_var_id.index(),
                metric_var.numeric_type(),
            );
        }

        for goal in goals {
            self.assert_task_fact(goal, "goal");
        }
        for fact in self.mutexes.iter().flatten() {
            self.assert_task_fact(fact, "mutex");
        }
        if let Some(global_constraint) = &self.global_constraint {
            self.assert_task_fact(global_constraint, "global constraint");
        }

        if !self.comparison_axioms.is_empty() {
            let last_arithmetic_layer = self
                .numeric_variables
                .iter()
                .filter_map(NumericVariable::axiom_layer)
                .max();
            let expected_comparison_layer = last_arithmetic_layer.map_or(0, |layer| layer + 1);
            let mut comparison_layer = None;
            for (axiom_id, axiom) in self.comparison_axioms.iter().enumerate() {
                let head = axiom.affected_var_id().index();
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
    /// happens in [`Task::new`]. So namespace assignment happens exactly once,
    /// over the whole task at once, and no later consumer has to rediscover
    /// the "propositional variable -> comparison axiom" mapping to know what
    /// kind of variable a fact names.
    fn assign_fact_namespaces(&mut self, goals: &mut [ExplicitFact]) {
        let conditions = Arc::clone(&self.numeric_conditions);
        let retag = |fact: &mut ExplicitFact| {
            *fact = fact.with_namespace(conditions.namespace_of(fact.var_index()));
        };

        goals.iter_mut().for_each(retag);
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
        self.global_constraint.iter_mut().for_each(retag);
    }

    /// Panic unless `fact` names a variable of the task and a value of its
    /// domain.
    fn assert_task_fact(&self, fact: &ExplicitFact, origin: &str) {
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
    }
}

/// Pin every condition variable to the two-valued [`ConditionValue`] domain.
///
/// A condition variable's domain is fixed by what a comparison can answer, so a
/// task does not get to choose it. numeric Fast Downward's SAS files declare a
/// third value, `<none of those>`, and name it in the initial-state block as
/// the variable's axiom default. It was never a value a state could
/// hold: the comparison axioms write a verdict for every condition variable
/// before anything reads one, so the placeholder is overwritten by the closure
/// [`Task::new`] runs a few lines later. Dropping it here is what
/// makes the domain two everywhere — packed states, abstract states and the
/// per-variable domain mappings the abstractions build on top of them.
fn narrow_condition_variables(
    conditions: &NumericConditions,
    variables: &mut [ExplicitVariable],
    state: &mut [ExplicitValueIndex],
) {
    /// The value numeric Fast Downward's `<none of those>` occupies, one past
    /// the domain.
    const PLACEHOLDER: usize = ConditionValue::DOMAIN_SIZE;

    for var_id in conditions.condition_var_ids() {
        let variable = &mut variables[var_id];
        assert!(
            variable.domain_size == ConditionValue::DOMAIN_SIZE
                || variable.domain_size == PLACEHOLDER + 1,
            "variable {var_id} ({}) carries a numeric condition but has domain size {}, \
             which is neither {} nor numeric Fast Downward's {} that adds the placeholder",
            variable.name,
            variable.domain_size,
            ConditionValue::DOMAIN_SIZE,
            PLACEHOLDER + 1
        );
        variable.domain_size = ConditionValue::DOMAIN_SIZE;
        variable.fact_names.truncate(ConditionValue::DOMAIN_SIZE);

        // "Not derived yet" and "does not hold" are the same statement about a
        // state, so the placeholder collapses onto `False`. Nothing else can
        // stand where it stood.
        for value in [&mut variable.axiom_default_value, &mut state[var_id]] {
            assert!(
                value.index() <= PLACEHOLDER,
                "condition variable {var_id} holds value {}, which is outside \
                 even numeric Fast Downward's domain of {} values",
                value.index(),
                PLACEHOLDER + 1
            );
            if value.index() == PLACEHOLDER {
                *value = ExplicitValueIndex::new(ConditionValue::False.as_u32());
            }
        }
    }
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
        task.goals = vec![ExplicitFact::numeric_variable(1, 5)].into();
        assert_fact_namespaces(&task);
    }

    #[test]
    #[should_panic(expected = "past the task's 3 propositional variables")]
    fn a_task_fact_may_not_name_a_variable_the_task_does_not_have() {
        let mut task = crate::tests::get_root_task();
        task.goals = vec![ExplicitFact::propositional(3, 0)].into();
        assert_fact_namespaces(&task);
    }
}

#[cfg(test)]
mod root_task_invariants {
    use super::*;
    use crate::axioms::CalOperator;

    fn valid_parts() -> TaskParts {
        TaskParts {
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
            global_constraint: Some(ExplicitFact::propositional(0, 0)),
        }
    }

    #[test]
    fn closing_the_initial_state_preserves_canonical_values() {
        let off_lattice = NumericValue::new(0.1 + 0.2);
        assert_eq!(off_lattice, NumericValue::new(0.3));
        let mut parts = valid_parts();
        parts.numeric_state = vec![off_lattice];
        let task = Task::new(parts);
        assert_eq!(
            task.initial_numeric_values()[0].to_bits(),
            off_lattice.to_bits()
        );
    }

    #[test]
    #[should_panic(expected = "initial propositional state has 0 values for 1 variables")]
    fn rejects_short_propositional_initial_state() {
        let mut parts = valid_parts();
        parts.state.clear();
        Task::new(parts);
    }

    #[test]
    #[should_panic(expected = "initial numeric state has 0 values for 1 variables")]
    fn rejects_short_numeric_initial_state() {
        let mut parts = valid_parts();
        parts.numeric_state.clear();
        Task::new(parts);
    }

    #[test]
    #[should_panic(expected = "metric variable 1 is out of bounds for 1 numeric variables")]
    fn rejects_out_of_range_metric_variable() {
        let mut parts = valid_parts();
        parts.metric = Metric::new(true, Some(VariableIndex::new(1)));
        Task::new(parts);
    }

    #[test]
    #[should_panic(expected = "metric variable 0 has type Regular, expected Cost or Derived")]
    fn rejects_regular_metric_variable() {
        let mut parts = valid_parts();
        parts.numeric_variables[0] =
            NumericVariable::new("fuel".to_string(), NumericType::Regular, None);
        Task::new(parts);
    }

    #[test]
    #[should_panic(expected = "goal fact value 2 is outside variable 0's domain of size 2")]
    fn rejects_out_of_range_goal_value() {
        let mut parts = valid_parts();
        parts.goals[0] = ExplicitFact::propositional(0, 2);
        Task::new(parts);
    }

    #[test]
    #[should_panic(expected = "mutex fact value 2 is outside variable 0's domain of size 2")]
    fn rejects_out_of_range_mutex_value() {
        let mut parts = valid_parts();
        parts.mutexes[0][0] = ExplicitFact::propositional(0, 2);
        Task::new(parts);
    }

    #[test]
    #[should_panic(
        expected = "global constraint fact value 2 is outside variable 0's domain of size 2"
    )]
    fn rejects_out_of_range_global_constraint_value() {
        let mut parts = valid_parts();
        parts.global_constraint = Some(ExplicitFact::propositional(0, 2));
        Task::new(parts);
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
        Task::new(parts);
    }
}
