#![cfg(feature = "cplex")]

//! Minimal safe ownership and sparse-LP layer over the IBM ILOG CPLEX C API.
//!
//! This crate deliberately exposes only the operations required by PlanForge.
//! All unsafe code is isolated here; planning algorithms consume checked Rust
//! types and exhaustive solve statuses.
//! It is an optional backend at the search end of PlanForge's PDDL translation
//! → SAS+ task → search pipeline; the `cplex` feature must be enabled.
//!
//! A variable definition can be prepared without opening a solver model:
//!
//! ```
//! use planforge_cplex::Variable;
//!
//! let variable = Variable::new(0.0, 1.0, 2.0);
//! assert_eq!(variable.objective, 2.0);
//! ```

mod ffi;

use std::ffi::{CStr, CString, c_char, c_int};
use std::fmt;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::sync::OnceLock;

const ERROR_BUFFER_SIZE: usize = 4096;
const LICENSE_PROBE_COLUMNS: usize = 1001;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectiveSense {
    Minimize,
    Maximize,
}

impl ObjectiveSense {
    fn cplex_value(self) -> c_int {
        match self {
            Self::Minimize => ffi::CPX_MIN,
            Self::Maximize => ffi::CPX_MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Variable {
    pub lower: f64,
    pub upper: f64,
    pub objective: f64,
}

impl Variable {
    pub const fn new(lower: f64, upper: f64, objective: f64) -> Self {
        Self {
            lower,
            upper,
            objective,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    pub lower: f64,
    pub upper: f64,
    pub coefficients: Vec<(usize, f64)>,
}

impl Constraint {
    pub fn new(lower: f64, upper: f64, coefficients: Vec<(usize, f64)>) -> Self {
        Self {
            lower,
            upper,
            coefficients,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit,
    TimeLimit,
    ObjectiveLimit,
    UserAbort,
    DeterministicTimeLimit,
    Unknown(i32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    Api,
    InvalidModel,
    InvalidState,
    RestrictedLicense,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    kind: ErrorKind,
    operation: &'static str,
    code: i32,
    detail: String,
}

impl Error {
    fn api(
        environment: Option<NonNull<ffi::Environment>>,
        operation: &'static str,
        code: i32,
    ) -> Self {
        let kind = if code == ffi::CPXERR_RESTRICTED_VERSION {
            ErrorKind::RestrictedLicense
        } else {
            ErrorKind::Api
        };
        let detail = environment
            .and_then(|environment| cplex_error_string(environment, code))
            .unwrap_or_else(|| format!("CPLEX status {code}"));
        Self {
            kind,
            operation,
            code,
            detail,
        }
    }

    fn invalid_model(detail: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidModel,
            operation: "validate model",
            code: 0,
            detail: detail.into(),
        }
    }

    fn invalid_state(operation: &'static str, detail: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidState,
            operation,
            code: 0,
            detail: detail.into(),
        }
    }

    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    pub fn code(&self) -> i32 {
        self.code
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.code != 0 {
            format!(" with CPLEX status {}", self.code)
        } else {
            String::new()
        };
        write!(
            formatter,
            "{} failed{}: {}",
            self.operation, status, self.detail
        )
    }
}

impl std::error::Error for Error {}

fn cplex_error_string(environment: NonNull<ffi::Environment>, code: i32) -> Option<String> {
    let mut buffer = [0_u8; ERROR_BUFFER_SIZE];
    // SAFETY: `environment` is live and `buffer` is writable for its full
    // capacity. CPLEX documents CPXgeterrorstring as writing a NUL-terminated
    // string into a CPXMESSAGEBUFSIZE buffer.
    let result = unsafe {
        ffi::CPXgeterrorstring(
            environment.as_ptr(),
            code,
            buffer.as_mut_ptr().cast::<c_char>(),
        )
    };
    if result.is_null() {
        return None;
    }
    // SAFETY: a non-null result from CPXgeterrorstring points to the
    // NUL-terminated contents of `buffer`.
    Some(
        unsafe { CStr::from_ptr(result) }
            .to_string_lossy()
            .trim()
            .to_owned(),
    )
}

#[derive(Debug)]
struct Environment {
    raw: NonNull<ffi::Environment>,
}

impl Environment {
    fn open() -> Result<Self, Error> {
        let mut status = 0;
        // SAFETY: CPLEX initializes a fresh environment and writes `status`.
        let raw = unsafe { ffi::CPXopenCPLEX(&mut status) };
        let raw = NonNull::new(raw).ok_or_else(|| Error::api(None, "CPXopenCPLEX", status))?;
        let environment = Self { raw };
        environment.set_int_parameter(ffi::CPX_PARAM_THREADS, 1)?;
        environment.set_int_parameter(ffi::CPX_PARAM_SCRIND, ffi::CPX_OFF)?;
        Ok(environment)
    }

    fn set_int_parameter(&self, parameter: i32, value: i32) -> Result<(), Error> {
        // SAFETY: `self.raw` is a live CPLEX environment.
        let status = unsafe { ffi::CPXsetintparam(self.raw.as_ptr(), parameter, value) };
        self.check("CPXsetintparam", status)
    }

    fn set_double_parameter(&self, parameter: i32, value: f64) -> Result<(), Error> {
        // SAFETY: `self.raw` is a live CPLEX environment.
        let status = unsafe { ffi::CPXsetdblparam(self.raw.as_ptr(), parameter, value) };
        self.check("CPXsetdblparam", status)
    }

    fn int_parameter(&self, parameter: i32) -> Result<i32, Error> {
        let mut value = 0;
        // SAFETY: `self.raw` is live and `value` is writable.
        let status = unsafe { ffi::CPXgetintparam(self.raw.as_ptr(), parameter, &mut value) };
        self.check("CPXgetintparam", status)?;
        Ok(value)
    }

    fn check(&self, operation: &'static str, status: i32) -> Result<(), Error> {
        if status == 0 {
            Ok(())
        } else {
            Err(Error::api(Some(self.raw), operation, status))
        }
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        let mut raw = self.raw.as_ptr();
        // SAFETY: this is the unique owned environment pointer. Drop cannot
        // report cleanup errors, but it never converts them into algorithmic
        // success: all explicit operations were already checked.
        let _ = unsafe { ffi::CPXcloseCPLEX(&mut raw) };
    }
}

#[derive(Debug)]
pub struct Model {
    environment: Environment,
    problem: NonNull<ffi::Problem>,
    permanent_rows: usize,
    temporary_rows: usize,
    last_status: Option<SolveStatus>,
}

impl Model {
    pub fn new(name: &str) -> Result<Self, Error> {
        let environment = Environment::open()?;
        let name = CString::new(name)
            .map_err(|_| Error::invalid_model("model name contains an interior NUL byte"))?;
        let mut status = 0;
        // SAFETY: environment and name are live; CPLEX writes `status`.
        let problem =
            unsafe { ffi::CPXcreateprob(environment.raw.as_ptr(), &mut status, name.as_ptr()) };
        let problem = NonNull::new(problem)
            .ok_or_else(|| Error::api(Some(environment.raw), "CPXcreateprob", status))?;
        Ok(Self {
            environment,
            problem,
            permanent_rows: 0,
            temporary_rows: 0,
            last_status: None,
        })
    }

    pub const fn infinity() -> f64 {
        ffi::CPX_INFBOUND
    }

    pub fn load(
        &mut self,
        objective_sense: ObjectiveSense,
        variables: &[Variable],
        constraints: &[Constraint],
    ) -> Result<(), Error> {
        if self.num_columns() != 0 || self.num_rows() != 0 {
            return Err(Error::invalid_state(
                "load model",
                "a CPLEX model can only be loaded once; construct a new Model for a new matrix",
            ));
        }
        validate_model(variables, constraints)?;
        self.set_objective_sense(objective_sense)?;

        let objective: Vec<f64> = variables
            .iter()
            .map(|variable| variable.objective)
            .collect();
        let lower: Vec<f64> = variables.iter().map(|variable| variable.lower).collect();
        let upper: Vec<f64> = variables.iter().map(|variable| variable.upper).collect();
        let count = to_c_int(variables.len(), "column count")?;
        // SAFETY: all slices contain `count` elements and remain live for the
        // call. Null optional type/name arrays request continuous unnamed
        // columns.
        let status = unsafe {
            ffi::CPXnewcols(
                self.environment.raw.as_ptr(),
                self.problem.as_ptr(),
                count,
                objective.as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
                ptr::null(),
                ptr::null_mut(),
            )
        };
        self.environment.check("CPXnewcols", status)?;
        self.add_rows(constraints)?;
        self.permanent_rows = constraints.len();
        self.last_status = None;
        Ok(())
    }

    pub fn set_objective_sense(&mut self, sense: ObjectiveSense) -> Result<(), Error> {
        // SAFETY: model and environment are live.
        let status = unsafe {
            ffi::CPXchgobjsen(
                self.environment.raw.as_ptr(),
                self.problem.as_ptr(),
                sense.cplex_value(),
            )
        };
        self.environment.check("CPXchgobjsen", status)?;
        self.last_status = None;
        Ok(())
    }

    pub fn set_objective(&mut self, values: &[f64]) -> Result<(), Error> {
        let columns = self.num_columns();
        if values.len() != columns {
            return Err(Error::invalid_model(format!(
                "objective has {} coefficients but model has {columns} columns",
                values.len()
            )));
        }
        let indices = consecutive_indices(columns)?;
        // SAFETY: both slices contain `columns` entries.
        let status = unsafe {
            ffi::CPXchgobj(
                self.environment.raw.as_ptr(),
                self.problem.as_ptr(),
                to_c_int(columns, "objective length")?,
                indices.as_ptr(),
                values.as_ptr(),
            )
        };
        self.environment.check("CPXchgobj", status)?;
        self.last_status = None;
        Ok(())
    }

    pub fn set_column_bounds(
        &mut self,
        column: usize,
        lower: f64,
        upper: f64,
    ) -> Result<(), Error> {
        if column >= self.num_columns() {
            return Err(Error::invalid_model(format!(
                "column {column} is outside a {}-column model",
                self.num_columns()
            )));
        }
        validate_bounds(lower, upper, "column bounds")?;
        let index = to_c_int(column, "column index")?;
        let indices = [index, index];
        let kinds = [b'L' as c_char, b'U' as c_char];
        let values = [lower, upper];
        // SAFETY: the arrays each contain two entries.
        let status = unsafe {
            ffi::CPXchgbds(
                self.environment.raw.as_ptr(),
                self.problem.as_ptr(),
                2,
                indices.as_ptr(),
                kinds.as_ptr(),
                values.as_ptr(),
            )
        };
        self.environment.check("CPXchgbds", status)?;
        self.last_status = None;
        Ok(())
    }

    pub fn set_row_bounds(&mut self, row: usize, lower: f64, upper: f64) -> Result<(), Error> {
        if row >= self.num_rows() {
            return Err(Error::invalid_model(format!(
                "row {row} is outside a {}-row model",
                self.num_rows()
            )));
        }
        validate_bounds(lower, upper, "row bounds")?;
        let encoding = RowEncoding::new(lower, upper);
        let row = to_c_int(row, "row index")?;
        let indices = [row];
        let rhs = [encoding.rhs];
        let senses = [encoding.sense];
        let ranges = [encoding.range];

        // Change all three pieces while no solve is in progress. This method
        // is the atomic logical operation exposed to callers, preventing the
        // transient negative ranged-row intervals that affected the C++ OSI
        // implementation.
        // SAFETY: all arrays contain one entry for the validated row.
        let status = unsafe {
            ffi::CPXchgrhs(
                self.environment.raw.as_ptr(),
                self.problem.as_ptr(),
                1,
                indices.as_ptr(),
                rhs.as_ptr(),
            )
        };
        self.environment.check("CPXchgrhs", status)?;
        // SAFETY: as above.
        let status = unsafe {
            ffi::CPXchgsense(
                self.environment.raw.as_ptr(),
                self.problem.as_ptr(),
                1,
                indices.as_ptr(),
                senses.as_ptr(),
            )
        };
        self.environment.check("CPXchgsense", status)?;
        // CPLEX only reads range values for ranged rows, but setting zero for
        // the other senses keeps later transitions deterministic.
        // SAFETY: as above.
        let status = unsafe {
            ffi::CPXchgrngval(
                self.environment.raw.as_ptr(),
                self.problem.as_ptr(),
                1,
                indices.as_ptr(),
                ranges.as_ptr(),
            )
        };
        self.environment.check("CPXchgrngval", status)?;
        self.last_status = None;
        Ok(())
    }

    pub fn add_temporary_constraints(&mut self, constraints: &[Constraint]) -> Result<(), Error> {
        if self.temporary_rows != 0 {
            return Err(Error::invalid_state(
                "add temporary constraints",
                "temporary constraints are already active; clear them before adding another set",
            ));
        }
        validate_constraints(self.num_columns(), constraints)?;
        self.add_rows(constraints)?;
        self.temporary_rows = constraints.len();
        self.last_status = None;
        Ok(())
    }

    pub fn clear_temporary_constraints(&mut self) -> Result<(), Error> {
        if self.temporary_rows == 0 {
            return Ok(());
        }
        let first = to_c_int(self.permanent_rows, "first temporary row")?;
        let last = to_c_int(
            self.permanent_rows + self.temporary_rows - 1,
            "last temporary row",
        )?;
        // SAFETY: the tracked temporary suffix is present in the live model.
        let status = unsafe {
            ffi::CPXdelrows(
                self.environment.raw.as_ptr(),
                self.problem.as_ptr(),
                first,
                last,
            )
        };
        self.environment.check("CPXdelrows", status)?;
        self.temporary_rows = 0;
        self.last_status = None;
        Ok(())
    }

    pub fn set_time_limit(&mut self, seconds: Option<f64>) -> Result<(), Error> {
        let seconds = match seconds {
            Some(seconds) if seconds.is_finite() && seconds >= 0.0 => seconds,
            Some(seconds) => {
                return Err(Error::invalid_model(format!(
                    "time limit must be finite and non-negative, got {seconds}"
                )));
            }
            None => 1.0e75,
        };
        self.environment
            .set_double_parameter(ffi::CPX_PARAM_TILIM, seconds)
    }

    pub fn solve(&mut self) -> Result<SolveStatus, Error> {
        let status = self.optimize()?;
        let status = if status == SolveStatus::Unknown(ffi::CPX_STAT_INFORUNBD) {
            self.disambiguate_infeasible_or_unbounded()?
        } else {
            status
        };
        self.last_status = Some(status);
        Ok(status)
    }

    pub fn objective_value(&self) -> Result<f64, Error> {
        if self.last_status != Some(SolveStatus::Optimal) {
            return Err(Error::invalid_state(
                "extract objective",
                format!("last solve status is {:?}", self.last_status),
            ));
        }
        let mut value = 0.0;
        // SAFETY: model has a current optimal solution and value is writable.
        let status = unsafe {
            ffi::CPXgetobjval(
                self.environment.raw.as_ptr(),
                self.problem.as_ptr(),
                &mut value,
            )
        };
        self.environment.check("CPXgetobjval", status)?;
        Ok(value)
    }

    pub fn solution(&self) -> Result<Vec<f64>, Error> {
        if self.last_status != Some(SolveStatus::Optimal) {
            return Err(Error::invalid_state(
                "extract solution",
                format!("last solve status is {:?}", self.last_status),
            ));
        }
        let columns = self.num_columns();
        if columns == 0 {
            return Ok(Vec::new());
        }
        let mut values = vec![0.0; columns];
        // SAFETY: the model is optimal and `values` covers the requested
        // inclusive column range.
        let status = unsafe {
            ffi::CPXgetx(
                self.environment.raw.as_ptr(),
                self.problem.as_ptr(),
                values.as_mut_ptr(),
                0,
                to_c_int(columns - 1, "last solution column")?,
            )
        };
        self.environment.check("CPXgetx", status)?;
        Ok(values)
    }

    pub fn primal_ray(&self) -> Result<Vec<f64>, Error> {
        if self.last_status != Some(SolveStatus::Unbounded) {
            return Err(Error::invalid_state(
                "extract primal ray",
                format!("last solve status is {:?}", self.last_status),
            ));
        }
        let mut values = vec![0.0; self.num_columns()];
        // SAFETY: CPLEX has proven this LP unbounded and `values` is sized to
        // the number of columns.
        let status = unsafe {
            ffi::CPXgetray(
                self.environment.raw.as_ptr(),
                self.problem.as_ptr(),
                values.as_mut_ptr(),
            )
        };
        self.environment.check("CPXgetray", status)?;
        Ok(values)
    }

    pub fn write(&self, path: &Path) -> Result<(), Error> {
        let path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| Error::invalid_model("model path contains an interior NUL byte"))?;
        let file_type = c"LP";
        // SAFETY: model and C strings are live for the call.
        let status = unsafe {
            ffi::CPXwriteprob(
                self.environment.raw.as_ptr(),
                self.problem.as_ptr(),
                path.as_ptr(),
                file_type.as_ptr(),
            )
        };
        self.environment.check("CPXwriteprob", status)
    }

    pub fn num_columns(&self) -> usize {
        // SAFETY: model and environment are live. CPLEX returns a nonnegative
        // count for a valid LP.
        let count =
            unsafe { ffi::CPXgetnumcols(self.environment.raw.as_ptr(), self.problem.as_ptr()) };
        usize::try_from(count).expect("CPLEX returned a negative column count")
    }

    pub fn num_rows(&self) -> usize {
        // SAFETY: model and environment are live.
        let count =
            unsafe { ffi::CPXgetnumrows(self.environment.raw.as_ptr(), self.problem.as_ptr()) };
        usize::try_from(count).expect("CPLEX returned a negative row count")
    }

    fn optimize(&self) -> Result<SolveStatus, Error> {
        // SAFETY: model and environment are live.
        let status = unsafe { ffi::CPXlpopt(self.environment.raw.as_ptr(), self.problem.as_ptr()) };
        self.environment.check("CPXlpopt", status)?;
        // SAFETY: optimization completed and the model is live.
        let status =
            unsafe { ffi::CPXgetstat(self.environment.raw.as_ptr(), self.problem.as_ptr()) };
        Ok(map_solve_status(status))
    }

    fn disambiguate_infeasible_or_unbounded(&self) -> Result<SolveStatus, Error> {
        let previous_presolve = self.environment.int_parameter(ffi::CPX_PARAM_PREIND)?;
        let previous_reduce = self.environment.int_parameter(ffi::CPX_PARAM_REDUCE)?;
        self.environment
            .set_int_parameter(ffi::CPX_PARAM_PREIND, ffi::CPX_OFF)?;
        self.environment
            .set_int_parameter(ffi::CPX_PARAM_REDUCE, ffi::CPX_OFF)?;
        let solve_result = self.optimize();
        let restore_presolve = self
            .environment
            .set_int_parameter(ffi::CPX_PARAM_PREIND, previous_presolve);
        let restore_reduce = self
            .environment
            .set_int_parameter(ffi::CPX_PARAM_REDUCE, previous_reduce);
        let status = solve_result?;
        restore_presolve?;
        restore_reduce?;
        if status == SolveStatus::Unknown(ffi::CPX_STAT_INFORUNBD) {
            return Err(Error::invalid_state(
                "disambiguate CPLEX solve status",
                "CPLEX still reports infeasible-or-unbounded with presolve and reductions disabled",
            ));
        }
        Ok(status)
    }

    fn add_rows(&mut self, constraints: &[Constraint]) -> Result<(), Error> {
        if constraints.is_empty() {
            return Ok(());
        }
        let mut rhs = Vec::with_capacity(constraints.len());
        let mut senses = Vec::with_capacity(constraints.len());
        let mut ranges = Vec::with_capacity(constraints.len());
        let mut row_starts = Vec::with_capacity(constraints.len());
        let nonzero_count: usize = constraints
            .iter()
            .map(|constraint| constraint.coefficients.len())
            .sum();
        let mut column_indices = Vec::with_capacity(nonzero_count);
        let mut coefficients = Vec::with_capacity(nonzero_count);

        for constraint in constraints {
            row_starts.push(to_c_int(column_indices.len(), "row matrix offset")?);
            let encoding = RowEncoding::new(constraint.lower, constraint.upper);
            rhs.push(encoding.rhs);
            senses.push(encoding.sense);
            ranges.push(encoding.range);
            for &(column, coefficient) in &constraint.coefficients {
                column_indices.push(to_c_int(column, "constraint column")?);
                coefficients.push(coefficient);
            }
        }

        let first_new_row = self.num_rows();
        // SAFETY: CSR arrays have the documented lengths; no column names are
        // added and all referenced columns were validated.
        let status = unsafe {
            ffi::CPXaddrows(
                self.environment.raw.as_ptr(),
                self.problem.as_ptr(),
                0,
                to_c_int(constraints.len(), "row count")?,
                to_c_int(nonzero_count, "nonzero count")?,
                rhs.as_ptr(),
                senses.as_ptr(),
                row_starts.as_ptr(),
                column_indices.as_ptr(),
                coefficients.as_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        self.environment.check("CPXaddrows", status)?;

        let ranged_indices: Vec<c_int> = ranges
            .iter()
            .enumerate()
            .filter(|(_, range)| **range != 0.0)
            .map(|(offset, _)| to_c_int(first_new_row + offset, "ranged row index"))
            .collect::<Result<_, _>>()?;
        let ranged_values: Vec<f64> = ranges
            .iter()
            .copied()
            .filter(|range| *range != 0.0)
            .collect();
        if !ranged_indices.is_empty() {
            // SAFETY: index and value slices have equal nonzero length.
            let status = unsafe {
                ffi::CPXchgrngval(
                    self.environment.raw.as_ptr(),
                    self.problem.as_ptr(),
                    to_c_int(ranged_indices.len(), "ranged row count")?,
                    ranged_indices.as_ptr(),
                    ranged_values.as_ptr(),
                )
            };
            self.environment.check("CPXchgrngval", status)?;
        }
        Ok(())
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        let mut problem = self.problem.as_ptr();
        // SAFETY: this is the unique owned problem pointer and the environment
        // outlives it because fields are dropped after this implementation.
        let _ = unsafe { ffi::CPXfreeprob(self.environment.raw.as_ptr(), &mut problem) };
    }
}

#[derive(Debug, Clone, Copy)]
struct RowEncoding {
    rhs: f64,
    sense: c_char,
    range: f64,
}

impl RowEncoding {
    fn new(lower: f64, upper: f64) -> Self {
        let lower_finite = is_finite_bound(lower);
        let upper_finite = is_finite_bound(upper);
        match (lower_finite, upper_finite) {
            (true, true) if lower == upper => Self {
                rhs: lower,
                sense: b'E' as c_char,
                range: 0.0,
            },
            (true, true) => Self {
                rhs: lower,
                sense: b'R' as c_char,
                range: upper - lower,
            },
            (true, false) => Self {
                rhs: lower,
                sense: b'G' as c_char,
                range: 0.0,
            },
            (false, true) => Self {
                rhs: upper,
                sense: b'L' as c_char,
                range: 0.0,
            },
            (false, false) => unreachable!("free rows are rejected during model validation"),
        }
    }
}

fn is_finite_bound(value: f64) -> bool {
    value > -ffi::CPX_INFBOUND && value < ffi::CPX_INFBOUND
}

fn validate_model(variables: &[Variable], constraints: &[Constraint]) -> Result<(), Error> {
    for (column, variable) in variables.iter().enumerate() {
        validate_bounds(
            variable.lower,
            variable.upper,
            &format!("column {column} bounds"),
        )?;
        if !variable.objective.is_finite() {
            return Err(Error::invalid_model(format!(
                "column {column} has non-finite objective {}",
                variable.objective
            )));
        }
    }
    validate_constraints(variables.len(), constraints)
}

fn validate_constraints(columns: usize, constraints: &[Constraint]) -> Result<(), Error> {
    for (row, constraint) in constraints.iter().enumerate() {
        validate_bounds(
            constraint.lower,
            constraint.upper,
            &format!("row {row} bounds"),
        )?;
        if !is_finite_bound(constraint.lower) && !is_finite_bound(constraint.upper) {
            return Err(Error::invalid_model(format!(
                "row {row} is free; omit it instead of loading a twice-infinite ranged row"
            )));
        }
        let mut previous = None;
        for &(column, coefficient) in &constraint.coefficients {
            if column >= columns {
                return Err(Error::invalid_model(format!(
                    "row {row} references column {column}, but model has {columns} columns"
                )));
            }
            if !coefficient.is_finite() {
                return Err(Error::invalid_model(format!(
                    "row {row}, column {column} has non-finite coefficient {coefficient}"
                )));
            }
            if previous.is_some_and(|previous| previous >= column) {
                return Err(Error::invalid_model(format!(
                    "row {row} column indices must be strictly increasing"
                )));
            }
            previous = Some(column);
        }
    }
    Ok(())
}

fn validate_bounds(lower: f64, upper: f64, description: &str) -> Result<(), Error> {
    if lower.is_nan() || upper.is_nan() {
        return Err(Error::invalid_model(format!("{description} contain NaN")));
    }
    if lower > upper {
        return Err(Error::invalid_model(format!(
            "{description} are empty: [{lower}, {upper}]"
        )));
    }
    if lower < -ffi::CPX_INFBOUND || upper > ffi::CPX_INFBOUND {
        return Err(Error::invalid_model(format!(
            "{description} exceed CPLEX infinity {}: [{lower}, {upper}]",
            ffi::CPX_INFBOUND
        )));
    }
    Ok(())
}

fn to_c_int(value: usize, description: &str) -> Result<c_int, Error> {
    c_int::try_from(value)
        .map_err(|_| Error::invalid_model(format!("{description} {value} exceeds C int range")))
}

fn consecutive_indices(len: usize) -> Result<Vec<c_int>, Error> {
    (0..len)
        .map(|index| to_c_int(index, "column index"))
        .collect()
}

fn map_solve_status(status: i32) -> SolveStatus {
    match status {
        ffi::CPX_STAT_OPTIMAL => SolveStatus::Optimal,
        ffi::CPX_STAT_INFEASIBLE => SolveStatus::Infeasible,
        ffi::CPX_STAT_UNBOUNDED => SolveStatus::Unbounded,
        ffi::CPX_STAT_ABORT_IT_LIM => SolveStatus::IterationLimit,
        ffi::CPX_STAT_ABORT_TIME_LIM => SolveStatus::TimeLimit,
        ffi::CPX_STAT_ABORT_OBJ_LIM
        | ffi::CPX_STAT_ABORT_PRIM_OBJ_LIM
        | ffi::CPX_STAT_ABORT_DUAL_OBJ_LIM => SolveStatus::ObjectiveLimit,
        ffi::CPX_STAT_ABORT_USER => SolveStatus::UserAbort,
        ffi::CPX_STAT_ABORT_DETTIME_LIM => SolveStatus::DeterministicTimeLimit,
        other => SolveStatus::Unknown(other),
    }
}

static LICENSE_CHECK: OnceLock<Result<(), Error>> = OnceLock::new();

pub fn assert_unrestricted_license() -> Result<(), Error> {
    LICENSE_CHECK
        .get_or_init(run_unrestricted_license_probe)
        .clone()
}

fn run_unrestricted_license_probe() -> Result<(), Error> {
    let mut model = Model::new("planforge-cplex-license-check")?;
    let variables = vec![Variable::new(0.0, 1.0, 0.0); LICENSE_PROBE_COLUMNS];
    model.load(ObjectiveSense::Minimize, &variables, &[])?;
    match model.solve()? {
        SolveStatus::Optimal => Ok(()),
        status => Err(Error::invalid_state(
            "verify unrestricted CPLEX license",
            format!("1001-column probe returned {status:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64) {
        assert!((left - right).abs() <= 1e-9, "{left} != {right}");
    }

    #[test]
    fn solves_sparse_maximization_and_reuses_objective() {
        assert_unrestricted_license().unwrap();
        let mut model = Model::new("sparse-max").unwrap();
        let variables = [
            Variable::new(0.0, Model::infinity(), 3.0),
            Variable::new(0.0, Model::infinity(), 2.0),
        ];
        let constraints = [
            Constraint::new(-Model::infinity(), 4.0, vec![(0, 1.0), (1, 1.0)]),
            Constraint::new(-Model::infinity(), 2.0, vec![(0, 1.0)]),
            Constraint::new(-Model::infinity(), 3.0, vec![(1, 1.0)]),
        ];
        model
            .load(ObjectiveSense::Maximize, &variables, &constraints)
            .unwrap();
        assert_eq!(model.solve().unwrap(), SolveStatus::Optimal);
        close(model.objective_value().unwrap(), 10.0);
        close(model.solution().unwrap()[0], 2.0);

        model.set_objective(&[1.0, 4.0]).unwrap();
        assert_eq!(model.solve().unwrap(), SolveStatus::Optimal);
        close(model.objective_value().unwrap(), 13.0);
    }

    #[test]
    fn ranged_rows_and_atomic_bound_changes_are_exact() {
        let mut model = Model::new("ranges").unwrap();
        model
            .load(
                ObjectiveSense::Minimize,
                &[Variable::new(-Model::infinity(), Model::infinity(), 1.0)],
                &[Constraint::new(2.0, 5.0, vec![(0, 1.0)])],
            )
            .unwrap();
        assert_eq!(model.solve().unwrap(), SolveStatus::Optimal);
        close(model.objective_value().unwrap(), 2.0);

        model.set_row_bounds(0, 3.0, 4.0).unwrap();
        model.set_objective_sense(ObjectiveSense::Maximize).unwrap();
        assert_eq!(model.solve().unwrap(), SolveStatus::Optimal);
        close(model.objective_value().unwrap(), 4.0);
    }

    #[test]
    fn temporary_constraints_are_a_checked_suffix() {
        let mut model = Model::new("temporary").unwrap();
        model
            .load(
                ObjectiveSense::Maximize,
                &[Variable::new(0.0, 10.0, 1.0)],
                &[],
            )
            .unwrap();
        model
            .add_temporary_constraints(&[Constraint::new(-Model::infinity(), 3.0, vec![(0, 1.0)])])
            .unwrap();
        assert_eq!(model.num_rows(), 1);
        assert_eq!(model.solve().unwrap(), SolveStatus::Optimal);
        close(model.objective_value().unwrap(), 3.0);
        model.clear_temporary_constraints().unwrap();
        assert_eq!(model.num_rows(), 0);
        assert_eq!(model.solve().unwrap(), SolveStatus::Optimal);
        close(model.objective_value().unwrap(), 10.0);
    }

    #[test]
    fn distinguishes_infeasible_and_unbounded_and_extracts_ray() {
        let mut infeasible = Model::new("infeasible").unwrap();
        infeasible
            .load(
                ObjectiveSense::Minimize,
                &[Variable::new(0.0, 1.0, 1.0)],
                &[Constraint::new(2.0, Model::infinity(), vec![(0, 1.0)])],
            )
            .unwrap();
        assert_eq!(infeasible.solve().unwrap(), SolveStatus::Infeasible);

        let mut unbounded = Model::new("unbounded").unwrap();
        unbounded
            .load(
                ObjectiveSense::Maximize,
                &[Variable::new(0.0, Model::infinity(), 1.0)],
                &[],
            )
            .unwrap();
        assert_eq!(unbounded.solve().unwrap(), SolveStatus::Unbounded);
        let ray = unbounded.primal_ray().unwrap();
        assert_eq!(ray.len(), 1);
        assert!(ray[0] > 0.0);
    }

    #[test]
    fn invalid_models_fail_before_entering_cplex() {
        let mut model = Model::new("invalid").unwrap();
        let error = model
            .load(
                ObjectiveSense::Minimize,
                &[Variable::new(2.0, 1.0, 0.0)],
                &[],
            )
            .unwrap_err();
        assert_eq!(error.kind(), &ErrorKind::InvalidModel);
    }

    #[test]
    fn free_rows_and_nonfinite_time_limits_are_explicit_errors() {
        let mut model = Model::new("invalid-free-row").unwrap();
        let error = model
            .load(
                ObjectiveSense::Minimize,
                &[Variable::new(0.0, 1.0, 0.0)],
                &[Constraint::new(
                    -Model::infinity(),
                    Model::infinity(),
                    vec![(0, 1.0)],
                )],
            )
            .unwrap_err();
        assert_eq!(error.kind(), &ErrorKind::InvalidModel);
        assert!(error.to_string().contains("free"));

        let mut model = Model::new("invalid-time-limit").unwrap();
        let error = model.set_time_limit(Some(f64::INFINITY)).unwrap_err();
        assert_eq!(error.kind(), &ErrorKind::InvalidModel);
    }

    #[test]
    fn restricted_license_status_is_never_genericized() {
        let error = Error::api(None, "license probe", ffi::CPXERR_RESTRICTED_VERSION);
        assert_eq!(error.kind(), &ErrorKind::RestrictedLicense);
        assert_eq!(error.code(), ffi::CPXERR_RESTRICTED_VERSION);
    }
}
