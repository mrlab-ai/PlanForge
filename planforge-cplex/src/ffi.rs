use std::ffi::{c_char, c_double, c_int, c_void};

pub(crate) type Environment = c_void;
pub(crate) type Problem = c_void;

pub(crate) const CPX_MIN: c_int = 1;
pub(crate) const CPX_MAX: c_int = -1;
pub(crate) const CPX_INFBOUND: c_double = 1.0e20;

pub(crate) const CPX_PARAM_PREIND: c_int = 1030;
pub(crate) const CPX_PARAM_SCRIND: c_int = 1035;
pub(crate) const CPX_PARAM_TILIM: c_int = 1039;
pub(crate) const CPX_PARAM_REDUCE: c_int = 1057;
pub(crate) const CPX_PARAM_THREADS: c_int = 1067;

pub(crate) const CPX_OFF: c_int = 0;

pub(crate) const CPX_STAT_OPTIMAL: c_int = 1;
pub(crate) const CPX_STAT_UNBOUNDED: c_int = 2;
pub(crate) const CPX_STAT_INFEASIBLE: c_int = 3;
pub(crate) const CPX_STAT_INFORUNBD: c_int = 4;
pub(crate) const CPX_STAT_ABORT_IT_LIM: c_int = 10;
pub(crate) const CPX_STAT_ABORT_TIME_LIM: c_int = 11;
pub(crate) const CPX_STAT_ABORT_OBJ_LIM: c_int = 12;
pub(crate) const CPX_STAT_ABORT_USER: c_int = 13;
pub(crate) const CPX_STAT_ABORT_PRIM_OBJ_LIM: c_int = 21;
pub(crate) const CPX_STAT_ABORT_DUAL_OBJ_LIM: c_int = 22;
pub(crate) const CPX_STAT_ABORT_DETTIME_LIM: c_int = 25;

pub(crate) const CPXERR_RESTRICTED_VERSION: c_int = 1016;

unsafe extern "C" {
    pub(crate) fn CPXopenCPLEX(status: *mut c_int) -> *mut Environment;
    pub(crate) fn CPXcloseCPLEX(environment: *mut *mut Environment) -> c_int;
    pub(crate) fn CPXcreateprob(
        environment: *const Environment,
        status: *mut c_int,
        name: *const c_char,
    ) -> *mut Problem;
    pub(crate) fn CPXfreeprob(environment: *const Environment, problem: *mut *mut Problem)
    -> c_int;

    pub(crate) fn CPXsetintparam(
        environment: *mut Environment,
        parameter: c_int,
        value: c_int,
    ) -> c_int;
    pub(crate) fn CPXsetdblparam(
        environment: *mut Environment,
        parameter: c_int,
        value: c_double,
    ) -> c_int;
    pub(crate) fn CPXgetintparam(
        environment: *const Environment,
        parameter: c_int,
        value: *mut c_int,
    ) -> c_int;

    pub(crate) fn CPXnewcols(
        environment: *const Environment,
        problem: *mut Problem,
        count: c_int,
        objective: *const c_double,
        lower: *const c_double,
        upper: *const c_double,
        variable_types: *const c_char,
        names: *mut *mut c_char,
    ) -> c_int;
    pub(crate) fn CPXaddrows(
        environment: *const Environment,
        problem: *mut Problem,
        column_count: c_int,
        row_count: c_int,
        nonzero_count: c_int,
        right_hand_sides: *const c_double,
        senses: *const c_char,
        row_starts: *const c_int,
        column_indices: *const c_int,
        coefficients: *const c_double,
        column_names: *mut *mut c_char,
        row_names: *mut *mut c_char,
    ) -> c_int;
    pub(crate) fn CPXdelrows(
        environment: *const Environment,
        problem: *mut Problem,
        first: c_int,
        last: c_int,
    ) -> c_int;

    pub(crate) fn CPXchgobj(
        environment: *const Environment,
        problem: *mut Problem,
        count: c_int,
        indices: *const c_int,
        values: *const c_double,
    ) -> c_int;
    pub(crate) fn CPXchgobjsen(
        environment: *const Environment,
        problem: *mut Problem,
        sense: c_int,
    ) -> c_int;
    pub(crate) fn CPXchgbds(
        environment: *const Environment,
        problem: *mut Problem,
        count: c_int,
        indices: *const c_int,
        lower_or_upper: *const c_char,
        values: *const c_double,
    ) -> c_int;
    pub(crate) fn CPXchgrhs(
        environment: *const Environment,
        problem: *mut Problem,
        count: c_int,
        indices: *const c_int,
        values: *const c_double,
    ) -> c_int;
    pub(crate) fn CPXchgrngval(
        environment: *const Environment,
        problem: *mut Problem,
        count: c_int,
        indices: *const c_int,
        values: *const c_double,
    ) -> c_int;
    pub(crate) fn CPXchgsense(
        environment: *const Environment,
        problem: *mut Problem,
        count: c_int,
        indices: *const c_int,
        senses: *const c_char,
    ) -> c_int;

    pub(crate) fn CPXlpopt(environment: *const Environment, problem: *mut Problem) -> c_int;
    pub(crate) fn CPXgetstat(environment: *const Environment, problem: *const Problem) -> c_int;
    pub(crate) fn CPXgetobjval(
        environment: *const Environment,
        problem: *const Problem,
        value: *mut c_double,
    ) -> c_int;
    pub(crate) fn CPXgetx(
        environment: *const Environment,
        problem: *const Problem,
        values: *mut c_double,
        first: c_int,
        last: c_int,
    ) -> c_int;
    pub(crate) fn CPXgetray(
        environment: *const Environment,
        problem: *const Problem,
        values: *mut c_double,
    ) -> c_int;
    pub(crate) fn CPXgetnumcols(environment: *const Environment, problem: *const Problem) -> c_int;
    pub(crate) fn CPXgetnumrows(environment: *const Environment, problem: *const Problem) -> c_int;
    pub(crate) fn CPXwriteprob(
        environment: *const Environment,
        problem: *const Problem,
        path: *const c_char,
        file_type: *const c_char,
    ) -> c_int;
    pub(crate) fn CPXgeterrorstring(
        environment: *const Environment,
        error_code: c_int,
        buffer: *mut c_char,
    ) -> *const c_char;
}
