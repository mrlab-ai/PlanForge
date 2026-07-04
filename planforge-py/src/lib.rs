use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyFileNotFoundError, PyValueError};
use pyo3::prelude::*;

use planforge_core;
use planforge_sas::numeric::numeric_task::{NumericRootTask, Operator, TaskRef};
use planforge_search::numeric::search::{SearchResult, SearchStatus};

create_exception!(planforge, PlanforgeError, PyException);
create_exception!(planforge, TranslateError, PlanforgeError);
create_exception!(planforge, ParseError, PlanforgeError);
create_exception!(planforge, SpecError, PyValueError);

/// Internal error carried out of the GIL-released closure. PyErr values are
/// constructed only after the GIL is reacquired.
enum SolveError {
    Translate(String),
    Parse(String),
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

#[pyfunction]
#[pyo3(signature = (*, domain=None, problem=None, sas=None, sas_text=None,
                    search=None, max_time=None, max_memory=None))]
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
        let task: TaskRef<'static> = Arc::new(task);
        planforge_core::solve_task(task, &spec, time_limit, memory_limit)
            .map_err(|err| SolveError::Search(err.to_string()))
    });

    let result = outcome.map_err(|err| -> PyErr {
        match err {
            SolveError::Translate(message) => TranslateError::new_err(message),
            SolveError::Parse(message) => ParseError::new_err(message),
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
    let py = m.py();
    m.add("PlanforgeError", py.get_type::<PlanforgeError>())?;
    m.add("TranslateError", py.get_type::<TranslateError>())?;
    m.add("ParseError", py.get_type::<ParseError>())?;
    m.add("SpecError", py.get_type::<SpecError>())?;
    Ok(())
}
