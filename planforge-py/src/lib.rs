use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyFileNotFoundError, PyIndexError, PyValueError};
use pyo3::prelude::*;

use planforge_sas::numeric_task::{
    AssignmentOperation, Effect, NumericRootTask, NumericType, Operator, TaskRef,
};
use planforge_sas::state_registry::{ConcreteState, StateRegistry};
use planforge_search::evaluation::{EvaluationError, EvaluationState, Heuristic};
use planforge_search::search::{AStarSearch, SearchEngine, SearchResult, SearchStatus};
use planforge_search::state_space::{
    EnumerationLimits, OwnedStateSpace,
    StateSpaceEnumerationError as RustStateSpaceEnumerationError, enumerate_state_space,
};
use planforge_search::successor_generator::SuccessorTree;
use planforge_search::task_restriction::build_restricted_task;

create_exception!(planforge, PlanforgeError, PyException);
create_exception!(planforge, TranslateError, PlanforgeError);
create_exception!(planforge, ParseError, PlanforgeError);
create_exception!(planforge, SpecError, PyValueError);
create_exception!(planforge, EnumerationError, PlanforgeError);

/// Internal error carried out of the GIL-released closure. PyErr values are
/// constructed only after the GIL is reacquired.
enum SolveError {
    Translate(String),
    Parse(String),
    Restrict(String),
    Search(String),
    FileNotFound(String),
}

type PyFact = (usize, usize);

#[derive(Clone)]
struct EffectData {
    conditions: Vec<PyFact>,
    variable: usize,
    precondition_value: Option<usize>,
    value: usize,
}

#[pyclass(name = "Effect", frozen, get_all)]
struct PyEffect {
    conditions: Vec<PyFact>,
    variable: usize,
    precondition_value: Option<usize>,
    value: usize,
}

#[pymethods]
impl PyEffect {
    fn __repr__(&self) -> String {
        format!(
            "Effect(variable={}, value={}, precondition_value={:?}, conditions={:?})",
            self.variable, self.value, self.precondition_value, self.conditions
        )
    }
}

#[derive(Clone)]
struct NumericEffectData {
    conditions: Vec<PyFact>,
    affected_variable: usize,
    operation: String,
    source_variable: usize,
    conditional: bool,
}

#[pyclass(name = "NumericEffect", frozen, get_all)]
struct PyNumericEffect {
    conditions: Vec<PyFact>,
    affected_variable: usize,
    operation: String,
    source_variable: usize,
    conditional: bool,
}

#[pymethods]
impl PyNumericEffect {
    fn __repr__(&self) -> String {
        format!(
            "NumericEffect(affected_variable={}, operation={:?}, source_variable={}, conditions={:?})",
            self.affected_variable, self.operation, self.source_variable, self.conditions
        )
    }
}

#[pyclass(name = "Operator", frozen)]
struct PyOperator {
    id: Option<usize>,
    name: String,
    cost: f64,
    preconditions: Vec<PyFact>,
    effects: Vec<EffectData>,
    numeric_effects: Vec<NumericEffectData>,
    task_id: Option<usize>,
}

#[pymethods]
impl PyOperator {
    #[getter]
    fn id(&self) -> Option<usize> {
        self.id
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn cost(&self) -> f64 {
        self.cost
    }

    #[getter]
    fn preconditions(&self) -> Vec<PyFact> {
        self.preconditions.clone()
    }

    #[getter]
    fn effects(&self, py: Python<'_>) -> PyResult<Vec<Py<PyEffect>>> {
        self.effects
            .iter()
            .map(|effect| {
                Py::new(
                    py,
                    PyEffect {
                        conditions: effect.conditions.clone(),
                        variable: effect.variable,
                        precondition_value: effect.precondition_value,
                        value: effect.value,
                    },
                )
            })
            .collect()
    }

    #[getter]
    fn numeric_effects(&self, py: Python<'_>) -> PyResult<Vec<Py<PyNumericEffect>>> {
        self.numeric_effects
            .iter()
            .map(|effect| {
                Py::new(
                    py,
                    PyNumericEffect {
                        conditions: effect.conditions.clone(),
                        affected_variable: effect.affected_variable,
                        operation: effect.operation.clone(),
                        source_variable: effect.source_variable,
                        conditional: effect.conditional,
                    },
                )
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        match self.id {
            Some(id) => format!("Operator(id={id}, {:?}, cost={})", self.name, self.cost),
            None => format!("Operator({:?}, cost={})", self.name, self.cost),
        }
    }
}

#[pyclass(name = "SearchResult", frozen, get_all)]
struct PySearchResult {
    /// "solved" | "unsolvable" | "timeout" | "memory_limit"
    status: String,
    plan: Option<Vec<Py<PyOperator>>>,
    cost: Option<f64>,
    nodes_expanded: usize,
    nodes_reopened: usize,
    nodes_evaluated: usize,
    evaluations: usize,
    nodes_generated: usize,
    dead_ends: usize,
    registered_states: usize,
    search_time: f64,
}

#[pyclass(name = "StateSpace", frozen)]
struct PyStateSpace {
    state_count: usize,
    transition_count: usize,
    goal_state_count: usize,
    dead_end_count: usize,
    diameter: Option<f64>,
    h_star_histogram: Vec<(f64, usize)>,
    propositional_values: Py<PyAny>,
    numeric_values: Py<PyAny>,
    transition_offsets: Py<PyAny>,
    transition_operator_ids: Py<PyAny>,
    transition_successor_ids: Py<PyAny>,
    transition_costs: Py<PyAny>,
    goal_states: Py<PyAny>,
    h_star: Py<PyAny>,
}

#[pymethods]
impl PyStateSpace {
    #[getter]
    fn state_count(&self) -> usize {
        self.state_count
    }

    #[getter]
    fn transition_count(&self) -> usize {
        self.transition_count
    }

    #[getter]
    fn goal_state_count(&self) -> usize {
        self.goal_state_count
    }

    #[getter]
    fn dead_end_count(&self) -> usize {
        self.dead_end_count
    }

    #[getter]
    fn diameter(&self) -> Option<f64> {
        self.diameter
    }

    #[getter]
    fn h_star_histogram(&self) -> Vec<(f64, usize)> {
        self.h_star_histogram.clone()
    }

    #[getter]
    fn propositional_values(&self, py: Python<'_>) -> Py<PyAny> {
        self.propositional_values.clone_ref(py)
    }

    #[getter]
    fn numeric_values(&self, py: Python<'_>) -> Py<PyAny> {
        self.numeric_values.clone_ref(py)
    }

    #[getter]
    fn transition_offsets(&self, py: Python<'_>) -> Py<PyAny> {
        self.transition_offsets.clone_ref(py)
    }

    #[getter]
    fn transition_operator_ids(&self, py: Python<'_>) -> Py<PyAny> {
        self.transition_operator_ids.clone_ref(py)
    }

    #[getter]
    fn transition_successor_ids(&self, py: Python<'_>) -> Py<PyAny> {
        self.transition_successor_ids.clone_ref(py)
    }

    #[getter]
    fn transition_costs(&self, py: Python<'_>) -> Py<PyAny> {
        self.transition_costs.clone_ref(py)
    }

    #[getter]
    fn goal_states(&self, py: Python<'_>) -> Py<PyAny> {
        self.goal_states.clone_ref(py)
    }

    #[getter]
    fn h_star(&self, py: Python<'_>) -> Py<PyAny> {
        self.h_star.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "StateSpace(states={}, transitions={}, goals={}, dead_ends={})",
            self.state_count, self.transition_count, self.goal_state_count, self.dead_end_count
        )
    }
}

#[pymethods]
impl PySearchResult {
    /// Reproduce the `sas_plan` file body: one `(operator name)` per line.
    fn plan_to_sas(&self, py: Python<'_>) -> String {
        match &self.plan {
            Some(ops) => ops
                .iter()
                .map(|op| format!("({})\n", op.borrow(py).name))
                .collect(),
            None => String::new(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SearchResult(status={:?}, cost={:?}, nodes_expanded={})",
            self.status, self.cost, self.nodes_expanded
        )
    }
}

#[pyclass(frozen)]
#[derive(Clone)]
struct State {
    #[pyo3(get)]
    values: Vec<usize>,
    #[pyo3(get)]
    numeric_values: Vec<f64>,
    registry_id: usize,
    state_id: usize,
}

#[pymethods]
impl State {
    /// Read one finite-domain variable without copying the complete snapshot.
    fn value(&self, variable: usize) -> PyResult<usize> {
        self.values.get(variable).copied().ok_or_else(|| {
            PyIndexError::new_err(format!(
                "propositional variable {variable} is out of bounds for {} values",
                self.values.len()
            ))
        })
    }

    /// Read one numeric variable without copying the complete snapshot.
    fn numeric_value(&self, variable: usize) -> PyResult<f64> {
        self.numeric_values.get(variable).copied().ok_or_else(|| {
            PyIndexError::new_err(format!(
                "numeric variable {variable} is out of bounds for {} values",
                self.numeric_values.len()
            ))
        })
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};

        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.values.hash(&mut h);
        for v in &self.numeric_values {
            v.to_bits().hash(&mut h);
        }
        h.finish()
    }

    fn __eq__(&self, other: &State) -> bool {
        self.values == other.values
            && self.numeric_values.len() == other.numeric_values.len()
            && self
                .numeric_values
                .iter()
                .zip(&other.numeric_values)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    fn __repr__(&self) -> String {
        format!(
            "State(values={:?}, numeric_values={:?})",
            self.values, self.numeric_values
        )
    }
}

impl State {
    fn snapshot(cstate: &ConcreteState, reg: &StateRegistry) -> State {
        State {
            values: cstate.get_state(reg),
            numeric_values: cstate.get_numeric_state(reg),
            registry_id: reg.id(),
            state_id: cstate.get_id(),
        }
    }
}

struct PyHeuristic {
    callable: Py<PyAny>,
    /// First error raised by the Python callable, re-raised after search.
    error: Rc<RefCell<Option<PyErr>>>,
    name: String,
}

impl Heuristic for PyHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        // If a previous call already failed, stop doing work.
        if self.error.borrow().is_some() {
            return Err(EvaluationError::ComputationFailed(
                "Python heuristic callback previously failed".to_string(),
            ));
        }
        let registry = eval_state.state_registry();
        let snapshot = State::snapshot(eval_state.state(), registry);
        let value = Python::with_gil(|py| -> PyResult<f64> {
            let state_obj = Py::new(py, snapshot)?;
            let result = self.callable.call1(py, (state_obj,))?;
            result.extract::<f64>(py)
        });
        match value {
            Ok(h) => Ok(h),
            Err(err) => {
                // Capture the first error and abort the Rust search immediately;
                // `search_with_heuristic` re-raises the original Python error.
                *self.error.borrow_mut() = Some(err);
                Err(EvaluationError::ComputationFailed(
                    "Python heuristic callback failed".to_string(),
                ))
            }
        }
    }

    fn heuristic_name(&self) -> &str {
        &self.name
    }
}

#[pyclass(unsendable)]
struct Task {
    task: TaskRef<'static>,
    registry: RefCell<StateRegistry<'static>>,
    succ: SuccessorTree,
}

struct GilReleasedTask(TaskRef<'static>);

// PyO3's stable `allow_threads` bound requires captured values to be `Send`,
// even though the closure is executed synchronously on the current thread with
// only the GIL released. Keep this private and use it only for that handoff.
unsafe impl Send for GilReleasedTask {}

impl GilReleasedTask {
    fn solve(
        self,
        spec: &planforge_searcher::SearchSpec,
        time_limit: Option<Duration>,
        max_memory: Option<u64>,
    ) -> std::io::Result<SearchResult> {
        planforge_core::solve_task(self.0, spec, time_limit, max_memory)
    }

    fn enumerate(
        self,
        limits: EnumerationLimits,
    ) -> Result<OwnedStateSpace, RustStateSpaceEnumerationError> {
        enumerate_state_space(self.0, limits)
    }
}

/// The default way from PDDL to a task: no SAS+ text in between. `from_sas` and
/// `from_sas_text` remain for a file produced earlier or by another tool.
fn translate_pddl_to_task(domain: &Path, problem: &Path) -> Result<NumericRootTask, String> {
    planforge_translate::translate_to_task(&domain.to_string_lossy(), &problem.to_string_lossy())
        .map_err(|err| err.to_string())
}

fn restrict_numeric_task(
    task: NumericRootTask,
    restrict_task: bool,
) -> Result<NumericRootTask, String> {
    if !restrict_task {
        return Ok(task);
    }
    match build_restricted_task(&task).map_err(|err| format!("{err:#}"))? {
        Some(restricted_task) => Ok(restricted_task.into_task()),
        None => Ok(task),
    }
}

fn fact_to_py(fact: &planforge_sas::numeric_task::ExplicitFact) -> PyFact {
    (fact.var(), fact.value())
}

fn operation_name(operation: &AssignmentOperation) -> &'static str {
    match operation {
        AssignmentOperation::Assign => "assign",
        AssignmentOperation::Plus => "increase",
        AssignmentOperation::Minus => "decrease",
        AssignmentOperation::Times => "scale_up",
        AssignmentOperation::Divide => "scale_down",
    }
}

fn numeric_type_name(numeric_type: &NumericType) -> &'static str {
    match numeric_type {
        NumericType::Constant => "constant",
        NumericType::Derived => "derived",
        NumericType::Cost => "cost",
        NumericType::Regular => "regular",
    }
}

fn effect_data(effect: &Effect) -> EffectData {
    EffectData {
        conditions: effect.conditions().iter().map(fact_to_py).collect(),
        variable: effect.var_id(),
        precondition_value: effect.precondition_value(),
        value: effect.value(),
    }
}

fn operator_to_py(
    py: Python<'_>,
    operator: &Operator,
    id: Option<usize>,
    task_id: Option<usize>,
    cost: f64,
) -> PyResult<Py<PyOperator>> {
    let numeric_effects = operator
        .assignment_effects()
        .iter()
        .map(|effect| NumericEffectData {
            conditions: effect.conditions().iter().map(fact_to_py).collect(),
            affected_variable: effect.affected_var_id(),
            operation: operation_name(effect.operation()).to_string(),
            source_variable: effect.var_id(),
            conditional: effect.is_conditional(),
        })
        .collect();
    Py::new(
        py,
        PyOperator {
            id,
            name: operator.name().to_string(),
            cost,
            preconditions: operator.preconditions().iter().map(fact_to_py).collect(),
            effects: operator.effects().iter().map(effect_data).collect(),
            numeric_effects,
            task_id,
        },
    )
}

fn checked_duration(seconds: f64, name: &str) -> PyResult<Duration> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be a finite non-negative number of seconds"
        )));
    }
    Ok(Duration::from_secs_f64(seconds))
}

fn state_space_to_py(py: Python<'_>, graph: OwnedStateSpace) -> PyResult<PyStateSpace> {
    let summary = graph.summary();
    let state_count = graph.num_states();
    let numpy = py.import("numpy").map_err(|error| {
        PyErr::new::<pyo3::exceptions::PyImportError, _>(format!(
            "state-space arrays require numpy (declared by planforge-py's package metadata): {error}"
        ))
    })?;
    let propositional_values = numpy
        .call_method1("asarray", (graph.propositional_values,))?
        .call_method1(
            "reshape",
            ((state_count, graph.num_propositional_variables),),
        )?
        .unbind();
    let numeric_values = numpy
        .call_method1("asarray", (graph.numeric_values,))?
        .call_method1("reshape", ((state_count, graph.num_numeric_variables),))?
        .unbind();
    let transition_offsets = numpy
        .call_method1("asarray", (graph.transition_offsets,))?
        .unbind();
    let transition_operator_ids = numpy
        .call_method1("asarray", (graph.transition_operator_ids,))?
        .unbind();
    let transition_successor_ids = numpy
        .call_method1("asarray", (graph.transition_successor_ids,))?
        .unbind();
    let transition_costs = numpy
        .call_method1("asarray", (graph.transition_costs,))?
        .unbind();
    let goal_states = numpy
        .call_method1("asarray", (graph.goal_states,))?
        .unbind();
    let h_star = numpy.call_method1("asarray", (graph.h_star,))?.unbind();

    Ok(PyStateSpace {
        state_count: summary.state_count,
        transition_count: summary.transition_count,
        goal_state_count: summary.goal_state_count,
        dead_end_count: summary.dead_end_count,
        diameter: summary.diameter,
        h_star_histogram: summary.h_star_histogram,
        propositional_values,
        numeric_values,
        transition_offsets,
        transition_operator_ids,
        transition_successor_ids,
        transition_costs,
        goal_states,
        h_star,
    })
}

#[pymethods]
impl Task {
    #[staticmethod]
    #[pyo3(signature = (text, restrict_task=false))]
    fn from_sas_text(text: &str, restrict_task: bool) -> PyResult<Self> {
        let task = restrict_numeric_task(
            NumericRootTask::try_from_str(text).map_err(ParseError::new_err)?,
            restrict_task,
        )
        .map_err(PlanforgeError::new_err)?;
        Ok(Self::build(Arc::new(task)))
    }

    #[staticmethod]
    #[pyo3(signature = (path, restrict_task=false))]
    fn from_sas(path: PathBuf, restrict_task: bool) -> PyResult<Self> {
        let text = std::fs::read_to_string(&path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                PyFileNotFoundError::new_err(format!("{}: {e}", path.display()))
            }
            _ => ParseError::new_err(format!("failed to read {}: {e}", path.display())),
        })?;
        Self::from_sas_text(&text, restrict_task)
    }

    #[staticmethod]
    #[pyo3(signature = (domain, problem, restrict_task=false))]
    fn from_pddl(
        py: Python<'_>,
        domain: PathBuf,
        problem: PathBuf,
        restrict_task: bool,
    ) -> PyResult<Self> {
        let task = py
            .allow_threads(|| -> Result<NumericRootTask, String> {
                translate_pddl_to_task(&domain, &problem)
            })
            .map_err(TranslateError::new_err)?;
        let task = restrict_numeric_task(task, restrict_task).map_err(PlanforgeError::new_err)?;
        Ok(Self::build(Arc::new(task)))
    }

    #[getter]
    fn num_variables(&self) -> usize {
        self.task.variables().len()
    }

    #[getter]
    fn num_numeric_variables(&self) -> usize {
        self.task.numeric_variables().len()
    }

    #[getter]
    fn num_operators(&self) -> usize {
        self.task.get_operators().len()
    }

    #[getter]
    fn num_goals(&self) -> usize {
        self.task.get_num_goals()
    }

    #[getter]
    fn goals(&self) -> Vec<(usize, usize)> {
        (0..self.task.get_num_goals())
            .map(|i| {
                let f = self.task.get_goal_fact(i);
                (f.var(), f.value())
            })
            .collect()
    }

    #[getter]
    fn metric(&self) -> bool {
        self.task.metric().use_metric()
    }

    #[getter]
    fn variable_names(&self) -> Vec<String> {
        (0..self.task.variables().len())
            .map(|i| {
                self.task
                    .get_variable_name(i)
                    .expect("variable index came from task.variables()")
                    .to_string()
            })
            .collect()
    }

    #[getter]
    fn variable_domain_sizes(&self) -> Vec<usize> {
        self.task
            .variables()
            .iter()
            .map(|variable| variable.domain_size())
            .collect()
    }

    #[getter]
    fn numeric_variable_names(&self) -> Vec<String> {
        self.task
            .numeric_variables()
            .iter()
            .map(|variable| variable.name().to_string())
            .collect()
    }

    #[getter]
    fn numeric_variable_types(&self) -> Vec<String> {
        self.task
            .numeric_variables()
            .iter()
            .map(|variable| numeric_type_name(variable.get_type()).to_string())
            .collect()
    }

    #[getter]
    fn registered_states(&self) -> usize {
        self.registry.borrow().num_registered_states()
    }

    fn operators(&self, py: Python<'_>) -> Vec<Py<PyOperator>> {
        let task_id = self.registry.borrow().id();
        self.task
            .get_operators()
            .iter()
            .enumerate()
            .map(|(operator_id, operator)| {
                operator_to_py(
                    py,
                    operator,
                    Some(operator_id),
                    Some(task_id),
                    self.task.abstract_operator_cost(operator_id),
                )
                .expect("creating a Python Operator should not fail")
            })
            .collect()
    }

    fn initial_state(&self) -> State {
        let mut reg = self.registry.borrow_mut();
        let s = reg.get_initial_state();
        State::snapshot(&s, &reg)
    }

    fn is_goal(&self, state: &State) -> PyResult<bool> {
        let reg = self.registry.borrow();
        let cstate = self.lookup(state, &reg)?;
        let mut all = true;
        for i in 0..self.task.get_num_goals() {
            let g = self.task.get_goal_fact(i);
            if !g.is_hold(reg.view(&cstate)) {
                all = false;
                break;
            }
        }
        Ok(all)
    }

    /// Return every operator applicable in `state`.
    ///
    /// This is the Python-driven prototyping path: crossing the FFI boundary
    /// once per expansion is deliberate. Pure Rust search never calls it.
    fn applicable_operators(&self, py: Python<'_>, state: &State) -> PyResult<Vec<Py<PyOperator>>> {
        let reg = self.registry.borrow();
        let cstate = self.lookup(state, &reg)?;
        let ids = self.applicable_operator_ids(&cstate, &reg);
        let task_id = reg.id();
        ids.into_iter()
            .map(|operator_id| {
                let operator_id = operator_id as usize;
                let operator = self
                    .task
                    .get_operators()
                    .get(operator_id)
                    .unwrap_or_else(|| {
                        panic!("successor generator returned invalid operator id {operator_id}")
                    });
                operator_to_py(
                    py,
                    operator,
                    Some(operator_id),
                    Some(task_id),
                    self.task.abstract_operator_cost(operator_id),
                )
            })
            .collect()
    }

    /// Apply an applicable operator, including numeric effects and axiom closure.
    fn apply(&self, state: &State, operator: PyRef<'_, PyOperator>) -> PyResult<State> {
        self.apply_operator_snapshot(state, &operator)
            .map(|(successor, _cost)| successor)
    }

    /// Apply an operator and also return its state-dependent transition cost.
    fn apply_with_cost(
        &self,
        state: &State,
        operator: PyRef<'_, PyOperator>,
    ) -> PyResult<(State, f64)> {
        self.apply_operator_snapshot(state, &operator)
    }

    /// Enumerate the complete reachable graph in Rust and compute exact h*.
    ///
    /// Every bound is mandatory. Hitting one raises `EnumerationError`; no
    /// partial graph and no plausible-but-wrong h* array is returned.
    #[pyo3(signature = (*, max_states, max_transitions, max_time))]
    fn enumerate_state_space(
        &self,
        py: Python<'_>,
        max_states: usize,
        max_transitions: usize,
        max_time: f64,
    ) -> PyResult<PyStateSpace> {
        let limits = EnumerationLimits {
            max_states,
            max_transitions,
            max_time: checked_duration(max_time, "max_time")?,
        };
        let task = GilReleasedTask(self.task.clone());
        let graph = py
            .allow_threads(move || task.enumerate(limits))
            .map_err(|error| EnumerationError::new_err(error.to_string()))?;
        state_space_to_py(py, graph)
    }

    /// (operator, successor_state, transition_cost) for every applicable operator.
    fn successors(
        &self,
        py: Python<'_>,
        state: &State,
    ) -> PyResult<Vec<(Py<PyOperator>, State, f64)>> {
        let mut reg = self.registry.borrow_mut();
        let cstate = self.lookup(state, &reg)?;
        let ids = self.applicable_operator_ids(&cstate, &reg);
        let operators = self.task.get_operators();
        let task_id = reg.id();
        let mut out = Vec::with_capacity(ids.len());
        let (mut b1, mut b2) = (Vec::new(), Vec::new());
        for op_id in ids {
            let operator_id = op_id as usize;
            let op = operators.get(operator_id).unwrap_or_else(|| {
                panic!("successor generator returned invalid operator id {operator_id}")
            });
            let (succ, cost) = reg
                .get_successor_state_with_buffers_and_cost(&cstate, op, &mut b1, &mut b2)
                .map_err(|e| {
                    PlanforgeError::new_err(format!(
                        "successor generation failed for {}: {e:?}",
                        op.name()
                    ))
                })?;
            let py_op = operator_to_py(
                py,
                op,
                Some(operator_id),
                Some(task_id),
                self.task.abstract_operator_cost(operator_id),
            )?;
            let snap = State::snapshot(&succ, &reg);
            out.push((py_op, snap, cost));
        }
        Ok(out)
    }

    /// Full search reusing this parsed task; delegates to the same pipeline as
    /// the module-level `solve()`.
    #[pyo3(signature = (search=None, max_time=None, max_memory=None))]
    fn solve(
        &self,
        py: Python<'_>,
        search: Option<String>,
        max_time: Option<f64>,
        max_memory: Option<u64>,
    ) -> PyResult<PySearchResult> {
        let search = search.unwrap_or_else(|| "astar(blind())".to_string());
        let spec = planforge_searcher::parse_search_spec(&search).map_err(SpecError::new_err)?;
        let time_limit = max_time.map(Duration::from_secs_f64);
        let task = GilReleasedTask(self.task.clone());
        let result = py
            .allow_threads(move || task.solve(&spec, time_limit, max_memory))
            .map_err(|e| PlanforgeError::new_err(e.to_string()))?;
        Ok(search_result_to_py(py, result))
    }

    /// Run A* or greedy best-first search with a Python heuristic callback.
    ///
    /// The callback receives a value snapshot of each evaluated State for
    /// inspection via `.values` and `.numeric_values`. The snapshot belongs to
    /// the search's internal registry, so `task.successors(state)` rejects it;
    /// guidance heuristics should read state values, not re-explore.
    #[pyo3(signature = (heuristic, greedy=false, max_time=None, max_memory=None))]
    fn search_with_heuristic(
        &self,
        py: Python<'_>,
        heuristic: Py<PyAny>,
        greedy: bool,
        max_time: Option<f64>,
        max_memory: Option<u64>,
    ) -> PyResult<PySearchResult> {
        let registry = StateRegistry::for_task(self.task.clone());
        let error = Rc::new(RefCell::new(None));
        let heur: Box<dyn Heuristic> = Box::new(PyHeuristic {
            callable: heuristic.clone_ref(py),
            error: error.clone(),
            name: "python".to_string(),
        });
        let time_limit = max_time.map(Duration::from_secs_f64);
        // GIL is held for the whole search: the heuristic calls back into Python
        // once per evaluated state. This is intentionally NOT allow_threads.
        let result = if greedy {
            AStarSearch::new_gbfs(&*self.task, registry, Some(heur), time_limit, max_memory)
                .search()
        } else {
            AStarSearch::new(&*self.task, registry, Some(heur), time_limit, max_memory).search()
        };
        if let Some(err) = error.borrow_mut().take() {
            return Err(err);
        }
        let result =
            result.map_err(|error| PlanforgeError::new_err(format!("search failed: {error:#}")))?;
        Ok(search_result_to_py(py, result))
    }
}

impl Task {
    fn build(task: TaskRef<'static>) -> Self {
        let registry = StateRegistry::for_task(task.clone());
        let succ = SuccessorTree::new(&*task);
        Task {
            task,
            registry: RefCell::new(registry),
            succ,
        }
    }

    /// Resolve a State to a ConcreteState in this task's registry, asserting the
    /// state actually came from this task.
    fn lookup(&self, state: &State, reg: &StateRegistry) -> PyResult<ConcreteState> {
        if state.registry_id != reg.id() {
            return Err(PyValueError::new_err("State does not belong to this Task"));
        }
        reg.lookup_state(state.state_id)
            .map_err(|e| PlanforgeError::new_err(format!("state lookup failed: {e:?}")))
    }

    fn applicable_operator_ids(&self, state: &ConcreteState, registry: &StateRegistry) -> Vec<u32> {
        let mut values = Vec::new();
        state.fill_state(registry, &mut values);
        let mut ids = Vec::new();
        self.succ.get_applicable_operators(&values, &mut ids);
        ids
    }

    fn validate_operator(&self, operator: &PyOperator, task_id: usize) -> PyResult<usize> {
        if operator.task_id != Some(task_id) {
            return Err(PyValueError::new_err(
                "Operator does not belong to this Task",
            ));
        }
        let operator_id = operator.id.ok_or_else(|| {
            PyValueError::new_err("plan-result Operator cannot be applied to a Task")
        })?;
        if operator_id >= self.task.get_operators().len() {
            return Err(PyValueError::new_err(format!(
                "operator id {operator_id} is out of bounds for {} operators",
                self.task.get_operators().len()
            )));
        }
        Ok(operator_id)
    }

    fn apply_operator_snapshot(
        &self,
        state: &State,
        operator: &PyOperator,
    ) -> PyResult<(State, f64)> {
        let mut registry = self.registry.borrow_mut();
        let concrete = self.lookup(state, &registry)?;
        let operator_id = self.validate_operator(operator, registry.id())?;
        let applicable = self.applicable_operator_ids(&concrete, &registry);
        if !applicable.contains(&(operator_id as u32)) {
            return Err(PyValueError::new_err(format!(
                "operator {} ({}) is not applicable in this state",
                operator_id, operator.name
            )));
        }
        let rust_operator = &self.task.get_operators()[operator_id];
        let (mut numeric_buffer, mut cost_buffer) = (Vec::new(), Vec::new());
        let (successor, cost) = registry
            .get_successor_state_with_buffers_and_cost(
                &concrete,
                rust_operator,
                &mut numeric_buffer,
                &mut cost_buffer,
            )
            .map_err(|error| {
                PlanforgeError::new_err(format!(
                    "successor generation failed for {}: {error:?}",
                    rust_operator.name()
                ))
            })?;
        Ok((State::snapshot(&successor, &registry), cost))
    }
}

/// One-call solve: pick a source, pick a search, get a result.
///
/// The parameter list is the published Python signature -- PyO3 maps one
/// keyword to one Rust parameter, and `tests/test_smoke.py` calls all eight by
/// name -- so the argument count cannot be reduced by extracting a struct. The
/// two ways to shrink it are both worse than the lint: a `**kwargs` dict loses
/// the introspectable signature and turns a misspelled keyword from PyO3's
/// `TypeError` into hand-rolled validation, and folding the four source
/// keywords into one argument is a breaking API change. `Task.from_pddl`,
/// `Task.from_sas`, `Task.from_sas_text` and `Task.solve` are the decomposed
/// form for callers who want it; this stays the flat one.
#[pyfunction]
#[pyo3(signature = (*, domain=None, problem=None, sas=None, sas_text=None,
                    search=None, max_time=None, max_memory=None, restrict_task=false))]
#[allow(clippy::too_many_arguments)]
fn solve(
    py: Python<'_>,
    domain: Option<PathBuf>,
    problem: Option<PathBuf>,
    sas: Option<PathBuf>,
    sas_text: Option<String>,
    search: Option<String>,
    max_time: Option<f64>,
    max_memory: Option<u64>,
    restrict_task: bool,
) -> PyResult<PySearchResult> {
    let has_pddl = domain.is_some() && problem.is_some();
    let has_partial_pddl = domain.is_some() ^ problem.is_some();
    let source_count = has_pddl as u8 + sas.is_some() as u8 + sas_text.is_some() as u8;

    if has_partial_pddl {
        return Err(PyValueError::new_err(
            "domain and problem must be given together",
        ));
    }
    if source_count != 1 {
        return Err(PyValueError::new_err(
            "provide exactly one of: (domain and problem), sas, or sas_text",
        ));
    }

    let search = search.unwrap_or_else(|| "astar(blind())".to_string());
    let spec = planforge_searcher::parse_search_spec(&search).map_err(SpecError::new_err)?;

    let time_limit = max_time.map(Duration::from_secs_f64);
    let memory_limit = max_memory;

    let outcome: Result<SearchResult, SolveError> = py.allow_threads(|| {
        let task = if let (Some(domain), Some(problem)) = (&domain, &problem) {
            translate_pddl_to_task(domain, problem).map_err(SolveError::Translate)?
        } else {
            let sas_text: String = if let Some(path) = &sas {
                std::fs::read_to_string(path).map_err(|err| match err.kind() {
                    std::io::ErrorKind::NotFound => {
                        SolveError::FileNotFound(format!("{}: {err}", path.display()))
                    }
                    _ => SolveError::Parse(format!("failed to read {}: {err}", path.display())),
                })?
            } else {
                sas_text
                    .clone()
                    .expect("validated: exactly one source was provided")
            };
            NumericRootTask::try_from_str(&sas_text).map_err(SolveError::Parse)?
        };

        let task = restrict_numeric_task(task, restrict_task).map_err(SolveError::Restrict)?;
        let task: TaskRef<'static> = Arc::new(task);
        planforge_core::solve_task(task, &spec, time_limit, memory_limit)
            .map_err(|err| SolveError::Search(err.to_string()))
    });

    let result = outcome.map_err(|err| -> PyErr {
        match err {
            SolveError::Translate(message) => TranslateError::new_err(message),
            SolveError::Parse(message) => ParseError::new_err(message),
            SolveError::Restrict(message) => PlanforgeError::new_err(message),
            SolveError::Search(message) => PlanforgeError::new_err(message),
            SolveError::FileNotFound(message) => PyFileNotFoundError::new_err(message),
        }
    })?;

    Ok(search_result_to_py(py, result))
}

fn status_str(status: &SearchStatus) -> &'static str {
    match status {
        SearchStatus::Solved(_) => "solved",
        SearchStatus::Failed => "unsolvable",
        SearchStatus::Timeout => "timeout",
        SearchStatus::MemoryLimitReached => "memory_limit",
        SearchStatus::InProgress => "in_progress",
    }
}

fn search_result_to_py(py: Python<'_>, result: SearchResult) -> PySearchResult {
    let plan = result.plan.as_ref().map(|operators| {
        operators
            .iter()
            .map(|operator: &Operator| {
                operator_to_py(py, operator, None, None, operator.cost() as f64)
                    .expect("creating a Python Operator should not fail")
            })
            .collect()
    });

    PySearchResult {
        status: status_str(&result.status).to_string(),
        plan,
        cost: result.solution_cost,
        nodes_expanded: result.nodes_expanded,
        nodes_reopened: result.nodes_reopened,
        nodes_evaluated: result.nodes_evaluated,
        evaluations: result.evaluations,
        nodes_generated: result.nodes_generated,
        dead_ends: result.dead_ends,
        registered_states: result.registered_states,
        search_time: result.search_time.as_secs_f64(),
    }
}

#[pymodule]
fn planforge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyStateSpace>()?;
    m.add_class::<PyOperator>()?;
    m.add_class::<PyEffect>()?;
    m.add_class::<PyNumericEffect>()?;
    m.add_class::<Task>()?;
    m.add_class::<State>()?;
    let py = m.py();
    m.add("PlanforgeError", py.get_type::<PlanforgeError>())?;
    m.add("TranslateError", py.get_type::<TranslateError>())?;
    m.add("ParseError", py.get_type::<ParseError>())?;
    m.add("SpecError", py.get_type::<SpecError>())?;
    m.add("EnumerationError", py.get_type::<EnumerationError>())?;
    Ok(())
}
