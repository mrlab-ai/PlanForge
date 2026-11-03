//! Python bindings for the complete PlanForge pipeline.
//!
//! The extension translates PDDL into a SAS+ task and exposes both Rust-owned
//! search and Python-driven prototyping interfaces. Its supported user-facing
//! examples are Python programs in the package README rather than Rust calls
//! into the generated PyO3 module.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyFileNotFoundError, PyIndexError, PyValueError};
use pyo3::prelude::*;

use planforge_sas::numeric_task::{
    AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex, NumericType, NumericValue,
    Operator, OperatorIndex, VariableIndex,
};
use planforge_sas::state_registry::{ConcreteState, StateRegistry};
use planforge_search::evaluation::{Estimate, EvaluationError, EvaluationState, Heuristic};
use planforge_search::heuristic_features::TaskRequirements;
use planforge_search::search::best_first::{AStarSearch, GreedyBestFirstSearch, SearchParams};
use planforge_search::search::{
    SearchConfig, SearchError, SearchLimits, SearchOutcome, SearchResult,
};
use planforge_search::state_space::{
    EnumerationLimits, OwnedStateSpace,
    StateSpaceEnumerationError as RustStateSpaceEnumerationError, enumerate_state_space,
};
use planforge_search::successor_generator::{StandardGroundedTree, SuccessorGenerator};
use planforge_search::task_transformations::transform_task;

mod operators;
mod search;
mod state_space;
mod task;

use operators::*;
use search::*;
use state_space::*;
use task::*;

create_exception!(planforge, PlanforgeError, PyException);
create_exception!(planforge, TranslateError, PlanforgeError);
create_exception!(planforge, ParseError, PlanforgeError);
create_exception!(planforge, SpecError, PyValueError);
create_exception!(planforge, EnumerationError, PlanforgeError);

#[pymodule]
fn planforge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_class::<PySearchResult>()?;
    m.add_class::<PyEdge>()?;
    m.add_class::<PyStateSpace>()?;
    m.add_class::<PyAtom>()?;
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
