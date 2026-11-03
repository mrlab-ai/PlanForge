//! The task's operators, effects and atoms as Python sees them, and their
//! conversion from the task's own types.

use super::*;

pub(crate) type PyFact = (usize, usize);

#[derive(Clone)]
pub(crate) struct EffectData {
    conditions: Vec<PyFact>,
    variable: usize,
    precondition_value: Option<usize>,
    value: usize,
}

#[pyclass(name = "Effect", frozen, get_all)]
pub(crate) struct PyEffect {
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
pub(crate) struct NumericEffectData {
    conditions: Vec<PyFact>,
    affected_variable: usize,
    operation: String,
    source_variable: usize,
    conditional: bool,
}

#[pyclass(name = "NumericEffect", frozen, get_all)]
pub(crate) struct PyNumericEffect {
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

/// Split a translator fact name into its parts.
///
/// The translator writes `Atom pred(a, b)`, `NegatedAtom pred(a, b)`, `pred()`
/// for a nullary predicate, and `<none of those>` for the sentinel value of a
/// mutex group. A name that does not have this shape is reported as
/// unparsed - predicate `None` and no arguments - rather than guessed at, so a
/// caller can always fall back on the raw name.
pub(crate) fn split_fact_name(name: &str) -> (bool, Option<String>, Vec<String>) {
    let (negated, rest) = match name.strip_prefix("NegatedAtom ") {
        Some(rest) => (true, rest),
        None => (false, name.strip_prefix("Atom ").unwrap_or(name)),
    };
    let Some(open) = rest.find('(') else {
        return (negated, None, Vec::new());
    };
    if !rest.ends_with(')') {
        return (negated, None, Vec::new());
    }
    let predicate = rest[..open].trim().to_string();
    if predicate.is_empty() {
        return (negated, None, Vec::new());
    }
    let inner = &rest[open + 1..rest.len() - 1];
    let arguments = if inner.trim().is_empty() {
        Vec::new()
    } else {
        inner
            .split(',')
            .map(|argument| argument.trim().to_string())
            .collect()
    };
    (negated, Some(predicate), arguments)
}

/// Split a translator operator name into the action and its arguments.
///
/// The translator writes them whitespace separated, `drop item1 rooma bot1`.
pub(crate) fn split_operator_name(name: &str) -> (Option<String>, Vec<String>) {
    let mut parts = name.split_whitespace();
    let Some(action) = parts.next() else {
        return (None, Vec::new());
    };
    (
        Some(action.to_string()),
        parts.map(str::to_string).collect(),
    )
}

/// One ground atom of the task, as the SAS encoding represents it: a value of a
/// finite-domain variable.
///
/// A PDDL ground atom becomes a `(variable, value)` pair. Mutually exclusive
/// atoms are collapsed into one variable, so a variable holding `n` values
/// encodes `n` atoms, one of which is true in any state. Some values are the
/// `<none of those>` sentinel the translator adds when no atom of a mutex group
/// holds; they are reported as they are named rather than filtered out, because
/// which values are real atoms is the translator's business, not this API's.
#[pyclass(name = "Atom", frozen, get_all)]
pub(crate) struct PyAtom {
    /// Index of the finite-domain variable this atom belongs to.
    pub(crate) variable: u32,
    /// The variable's value that makes this atom true.
    pub(crate) value: u32,
    /// The translator's name for the atom, e.g. `Atom at(rover1, waypoint2)`.
    pub(crate) name: String,
    /// The predicate, or `None` when the name is not of the form `pred(args)`.
    pub(crate) predicate: Option<String>,
    /// The predicate's arguments, in order.
    pub(crate) arguments: Vec<String>,
    /// Whether this value encodes the *absence* of the predicate.
    pub(crate) negated: bool,
}

#[pymethods]
impl PyAtom {
    fn __repr__(&self) -> String {
        format!(
            "Atom(variable={}, value={}, name={:?})",
            self.variable, self.value, self.name
        )
    }
}

#[pyclass(name = "Operator", frozen)]
pub(crate) struct PyOperator {
    pub(crate) id: Option<usize>,
    pub(crate) name: String,
    cost: f64,
    preconditions: Vec<PyFact>,
    effects: Vec<EffectData>,
    numeric_effects: Vec<NumericEffectData>,
    pub(crate) task_id: Option<usize>,
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

    /// The action this is a ground instance of, or `None` for an unnamed
    /// operator.
    #[getter]
    fn action(&self) -> Option<String> {
        split_operator_name(&self.name).0
    }

    /// The action's arguments, in order.
    #[getter]
    fn arguments(&self) -> Vec<String> {
        split_operator_name(&self.name).1
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

pub(crate) fn fact_to_py(fact: &planforge_sas::numeric_task::ExplicitFact) -> PyFact {
    (fact.var(), fact.value())
}

pub(crate) fn operation_name(operation: &AssignmentOperation) -> &'static str {
    match operation {
        AssignmentOperation::Assign => "assign",
        AssignmentOperation::Plus => "increase",
        AssignmentOperation::Minus => "decrease",
        AssignmentOperation::Times => "scale_up",
        AssignmentOperation::Divide => "scale_down",
    }
}

pub(crate) fn numeric_type_name(numeric_type: NumericType) -> &'static str {
    match numeric_type {
        NumericType::Constant => "constant",
        NumericType::Derived => "derived",
        NumericType::Cost => "cost",
        NumericType::Regular => "regular",
    }
}

pub(crate) fn effect_data(effect: &Effect) -> EffectData {
    EffectData {
        conditions: effect.conditions().iter().map(fact_to_py).collect(),
        variable: effect.var_id().index(),
        precondition_value: effect.precondition_value().map(|v| v.index()),
        value: effect.value().index(),
    }
}

pub(crate) fn operator_to_py(
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
            affected_variable: effect.affected_var_id().index(),
            operation: operation_name(effect.operation()).to_string(),
            source_variable: effect.var_id().index(),
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
