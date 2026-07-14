use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;

use planforge_core;
use planforge_sas::numeric::numeric_task::{NumericRootTask, Operator, TaskRef};
use planforge_sas::numeric::state_registry::{ConcreteState, StateRegistry};
use planforge_search::numeric::evaluation::domain_abstractions::restricted_task::build_restricted_task;
use planforge_search::numeric::evaluation::{EvaluationError, EvaluationState, Heuristic};
use planforge_search::numeric::search::{AStarSearch, SearchEngine, SearchResult, SearchStatus};
use planforge_search::numeric::successor_generator::SuccessorTree;

create_exception!(planforge, PlanforgeError, PyException);
create_exception!(planforge, TranslateError, PlanforgeError);
create_exception!(planforge, ParseError, PlanforgeError);
create_exception!(planforge, SpecError, PyValueError);

/// Internal error carried out of the GIL-released closure. PyErr values are
/// constructed only after the GIL is reacquired.
enum SolveError {
    Translate(String),
    Parse(String),
    Restrict(String),
    Search(String),
    FileNotFound(String),
}

#[pyclass(name = "Operator", frozen, get_all)]
struct PyOperator {
    name: String,
    cost: f64,
}

#[pymethods]
impl PyOperator {
    fn __repr__(&self) -> String {
        format!("Operator({:?}, cost={})", self.name, self.cost)
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
            return Ok(0.0);
        }
        let registry = eval_state
            .state_registry()
            .expect("python heuristic needs the state registry");
        let snapshot = State::snapshot(eval_state.state(), registry);
        let value = Python::with_gil(|py| -> PyResult<f64> {
            let state_obj = Py::new(py, snapshot)?;
            let result = self.callable.call1(py, (state_obj,))?;
            result.extract::<f64>(py)
        });
        match value {
            Ok(h) => Ok(h),
            Err(err) => {
                // Capture the first error; return 0.0 so the (finite) search
                // still terminates, then `search_with_heuristic` re-raises it.
                *self.error.borrow_mut() = Some(err);
                Ok(0.0)
            }
        }
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
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
        let text = py
            .allow_threads(|| -> Result<String, String> {
                let raw = planforge_translator::translate_to_sas_string(
                    &domain.to_string_lossy(),
                    &problem.to_string_lossy(),
                )
                .map_err(|e| e.to_string())?;
                Ok(planforge_translate::preprocess::run_preprocess_to_string(
                    &raw,
                ))
            })
            .map_err(TranslateError::new_err)?;
        Self::from_sas_text(&text, restrict_task)
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
    fn registered_states(&self) -> usize {
        self.registry.borrow().num_registered_states()
    }

    fn operators(&self, py: Python<'_>) -> Vec<Py<PyOperator>> {
        self.task
            .get_operators()
            .iter()
            .map(|op| {
                Py::new(
                    py,
                    PyOperator {
                        name: op.name().to_string(),
                        cost: op.cost() as f64,
                    },
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
            if !g.is_hold(&cstate, &reg) {
                all = false;
                break;
            }
        }
        Ok(all)
    }

    /// (operator, successor_state, transition_cost) for every applicable operator.
    fn successors(
        &self,
        py: Python<'_>,
        state: &State,
    ) -> PyResult<Vec<(Py<PyOperator>, State, f64)>> {
        let mut reg = self.registry.borrow_mut();
        let cstate = self.lookup(state, &reg)?;
        let mut vals = Vec::new();
        cstate.fill_state(&reg, &mut vals);
        let mut ids: Vec<u32> = Vec::new();
        self.succ.get_applicable_operators(&vals, &mut ids);
        let operators = self.task.get_operators();
        let mut out = Vec::with_capacity(ids.len());
        let (mut b1, mut b2) = (Vec::new(), Vec::new());
        for op_id in ids {
            let op = &operators[op_id as usize];
            let (succ, cost) = reg
                .get_successor_state_with_buffers_and_cost(&cstate, op, &mut b1, &mut b2)
                .map_err(|e| {
                    PlanforgeError::new_err(format!(
                        "successor generation failed for {}: {e:?}",
                        op.name()
                    ))
                })?;
            let py_op = Py::new(
                py,
                PyOperator {
                    name: op.name().to_string(),
                    cost: op.cost() as f64,
                },
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
        let mut search = if greedy {
            AStarSearch::new_gbfs(
                self.task.clone(),
                registry,
                Some(heur),
                time_limit,
                max_memory,
            )
        } else {
            AStarSearch::new(
                self.task.clone(),
                registry,
                Some(heur),
                time_limit,
                max_memory,
            )
        };
        let result = search.search();
        if let Some(err) = error.borrow_mut().take() {
            return Err(err);
        }
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
}

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
        let sas_text: String = if let (Some(domain), Some(problem)) = (&domain, &problem) {
            let raw = planforge_translator::translate_to_sas_string(
                &domain.to_string_lossy(),
                &problem.to_string_lossy(),
            )
            .map_err(|err| SolveError::Translate(err.to_string()))?;
            planforge_translate::preprocess::run_preprocess_to_string(&raw)
        } else if let Some(path) = &sas {
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

        let task = NumericRootTask::try_from_str(&sas_text).map_err(SolveError::Parse)?;
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
                Py::new(
                    py,
                    PyOperator {
                        name: operator.name().to_string(),
                        cost: operator.cost() as f64,
                    },
                )
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
    m.add_class::<PyOperator>()?;
    m.add_class::<Task>()?;
    m.add_class::<State>()?;
    let py = m.py();
    m.add("PlanforgeError", py.get_type::<PlanforgeError>())?;
    m.add("TranslateError", py.get_type::<TranslateError>())?;
    m.add("ParseError", py.get_type::<ParseError>())?;
    m.add("SpecError", py.get_type::<SpecError>())?;
    Ok(())
}
