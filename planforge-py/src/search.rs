//! Search from Python: the `solve` function, searches with a Python
//! heuristic callback, and their results.

use super::*;

/// Internal error carried out of the GIL-released closure. PyErr values are
/// constructed only after the GIL is reacquired.
pub(crate) enum SolveError {
    Translate(String),
    Parse(String),
    Transform(String),
    Search(String),
    FileNotFound(String),
}

#[pyclass(name = "SearchResult", frozen, get_all)]
pub(crate) struct PySearchResult {
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
pub(crate) struct State {
    #[pyo3(get)]
    values: Vec<u32>,
    numeric_values: Vec<NumericValue>,
    pub(crate) registry_id: usize,
    pub(crate) state_id: usize,
}

#[pymethods]
impl State {
    #[getter]
    fn numeric_values(&self) -> Vec<f64> {
        self.numeric_values
            .iter()
            .map(NumericValue::value)
            .collect()
    }

    /// Read one finite-domain variable without copying the complete snapshot.
    fn value(&self, variable: usize) -> PyResult<u32> {
        self.values.get(variable).copied().ok_or_else(|| {
            PyIndexError::new_err(format!(
                "propositional variable {variable} is out of bounds for {} values",
                self.values.len()
            ))
        })
    }

    /// Read one numeric variable without copying the complete snapshot.
    fn numeric_value(&self, variable: usize) -> PyResult<f64> {
        self.numeric_values
            .get(variable)
            .map(NumericValue::value)
            .ok_or_else(|| {
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
        self.numeric_values.hash(&mut h);
        h.finish()
    }

    fn __eq__(&self, other: &State) -> bool {
        self.values == other.values && self.numeric_values == other.numeric_values
    }

    fn __repr__(&self) -> String {
        format!(
            "State(values={:?}, numeric_values={:?})",
            self.values,
            self.numeric_values()
        )
    }
}

impl State {
    pub(crate) fn snapshot(cstate: &ConcreteState, reg: &StateRegistry) -> State {
        State {
            values: cstate
                .propositional_values(reg)
                .iter()
                .map(|v| v.index() as u32)
                .collect(),
            numeric_values: cstate.regular_numeric_values(reg),
            registry_id: reg.id(),
            state_id: cstate.id(),
        }
    }
}

pub(crate) struct PyHeuristic {
    pub(crate) callable: Py<PyAny>,
    /// First error raised by the Python callable, re-raised after search.
    pub(crate) error: Arc<Mutex<Option<PyErr>>>,
    pub(crate) name: String,
}

impl Heuristic for PyHeuristic {
    fn evaluate(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<Estimate, EvaluationError> {
        // If a previous call already failed, stop doing work.
        let mut error = self.error.lock().expect("the error slot is never poisoned");
        if error.is_some() {
            return Err(EvaluationError::ComputationFailed(
                "Python heuristic callback previously failed".to_string(),
            ));
        }
        let registry = eval_state.state_registry();
        let snapshot = State::snapshot(eval_state.state(), registry);
        let value = Python::with_gil(|py| -> PyResult<f64> {
            let state_obj = Py::new(py, snapshot)?;
            let result = self.callable.call1(py, (state_obj,))?;
            let h = result.extract::<f64>(py)?;
            if h >= 0.0 {
                Ok(h)
            } else {
                Err(PyValueError::new_err(format!(
                    "a heuristic estimate is non-negative or infinity, got {h}"
                )))
            }
        });
        match value {
            Ok(h) => Ok(Estimate::new(h)),
            Err(err) => {
                // Capture the first error and abort the Rust search immediately;
                // `search_with_heuristic` re-raises the original Python error.
                *error = Some(err);
                Err(EvaluationError::ComputationFailed(
                    "Python heuristic callback failed".to_string(),
                ))
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// The default way from PDDL to a task: no SAS+ text in between. `from_sas` and
/// `from_sas_text` remain for a file produced earlier or by another tool.
pub(crate) fn translate_pddl_to_task(
    domain: &Path,
    problem: &Path,
) -> Result<planforge_sas::numeric_task::Task, String> {
    planforge_translate::translate_to_task(&domain.to_string_lossy(), &problem.to_string_lossy())
        .map_err(|err| err.to_string())
}

/// `task` transformed by the transformation specifications `transforms`,
/// in order, which must include the ones `requirements` names.
pub(crate) fn transform_numeric_task(
    task: planforge_sas::numeric_task::Task,
    transforms: &[String],
    requirements: &TaskRequirements,
) -> Result<planforge_sas::numeric_task::Task, String> {
    let features = planforge_heuristics::features();
    let transformations = transforms
        .iter()
        .map(|raw| features.parse_transformation(raw))
        .collect::<Result<Vec<_>, _>>()?;
    transform_task(task, &transformations, requirements).map_err(|err| err.to_string())
}

/// The message of a failed search: a heuristic that cannot be built speaks
/// for itself, anything else is prefixed.
pub(crate) fn search_failure(error: SearchError) -> String {
    match error {
        SearchError::Build(error) => error.to_string(),
        error => format!("search failed: {error}"),
    }
}

pub(crate) fn checked_duration(seconds: f64, name: &str) -> PyResult<Duration> {
    if !seconds.is_finite() || seconds < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be a finite non-negative number of seconds"
        )));
    }
    Ok(Duration::from_secs_f64(seconds))
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
                    search=None, max_time=None, max_memory=None, transforms=Vec::new()))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve(
    py: Python<'_>,
    domain: Option<PathBuf>,
    problem: Option<PathBuf>,
    sas: Option<PathBuf>,
    sas_text: Option<String>,
    search: Option<String>,
    max_time: Option<f64>,
    max_memory: Option<u64>,
    transforms: Vec<String>,
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
    let spec = planforge_heuristics::features()
        .parse_search(&search)
        .map_err(|error| SpecError::new_err(error.to_string()))?;

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
            planforge_sas::numeric_task::Task::try_from_str(&sas_text)
                .map_err(|error| SolveError::Parse(error.to_string()))?
        };

        let task = transform_numeric_task(task, &transforms, &spec.requirements())
            .map_err(SolveError::Transform)?;
        let task: Arc<planforge_sas::numeric_task::Task> = Arc::new(task);
        spec.solve(
            task,
            planforge_search::search::SearchLimits {
                time: time_limit,
                memory_bytes: memory_limit,
            },
        )
        .map_err(|error| SolveError::Search(search_failure(error)))
    });

    let result = outcome.map_err(|err| -> PyErr {
        match err {
            SolveError::Translate(message) => TranslateError::new_err(message),
            SolveError::Parse(message) => ParseError::new_err(message),
            SolveError::Transform(message) => PlanforgeError::new_err(message),
            SolveError::Search(message) => PlanforgeError::new_err(message),
            SolveError::FileNotFound(message) => PyFileNotFoundError::new_err(message),
        }
    })?;

    Ok(search_result_to_py(py, result))
}

pub(crate) fn search_result_to_py(py: Python<'_>, result: SearchResult) -> PySearchResult {
    let (status, plan, cost) = match &result.outcome {
        SearchOutcome::Solved { plan, cost } => {
            let plan = plan
                .iter()
                .map(|operator: &Operator| {
                    operator_to_py(py, operator, None, None, operator.cost().value() as f64)
                        .expect("creating a Python Operator should not fail")
                })
                .collect();
            ("solved", Some(plan), Some(*cost))
        }
        SearchOutcome::Unsolvable => ("unsolvable", None, None),
        SearchOutcome::Timeout => ("timeout", None, None),
        SearchOutcome::OutOfMemory => ("memory_limit", None, None),
    };
    let statistics = &result.statistics;
    PySearchResult {
        status: status.to_owned(),
        plan,
        cost,
        nodes_expanded: statistics.nodes.expanded,
        nodes_reopened: statistics.nodes.reopened,
        nodes_evaluated: statistics.nodes.evaluated,
        evaluations: statistics.evaluations,
        nodes_generated: statistics.nodes.generated,
        dead_ends: statistics.dead_ends,
        registered_states: statistics.registered_states,
        search_time: result.search_time.as_secs_f64(),
    }
}
