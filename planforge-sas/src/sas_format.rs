//! What the SAS+ format *means*, without its syntax.
//!
//! A [`NumericRootTask`] is built two ways: parsed from the text format by
//! [`crate::numeric_parser`], or handed over directly by the translator, which
//! holds the same task in its own representation. Both must produce the same
//! task, so everything the format leaves implicit lives here rather than in
//! either producer: the token tables, the axiom default a derived variable
//! takes from the initial state, and the renumbering that moves the numeric
//! conditions to the end of the variable id space.
//!
//! [`crate::numeric_parser`] owns the text syntax and nothing else.

use crate::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use crate::numeric_task::{
    AssignmentOperation, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericType,
    NumericVariable, Operator,
};

/// The version of the format this crate reads and the translator writes.
pub const SAS_FILE_VERSION: u32 = 4;

/// The layer the format's `layer` field denotes. A negative layer — the writers
/// spell it `-1` — means no axiom derives the variable, which is a different
/// thing from layer zero.
pub fn axiom_layer_from_sas(layer: i32) -> Option<usize> {
    usize::try_from(layer).ok()
}

/// The value an effect requires of the variable it writes, or `None` when it
/// applies whatever that variable holds. The format spells the latter `-1`.
///
/// The same answer decides whether the effect contributes a precondition to its
/// operator, so both readers ask this one question rather than each testing the
/// field for itself.
pub fn effect_precondition_from_sas(precondition: i32) -> Option<usize> {
    usize::try_from(precondition).ok()
}

/// The cost the format carries for an operator.
///
/// Costs are real-valued inside the translation and integral in the file, so a
/// fractional one would be written out as a number the reader cannot take back.
/// Failing here rather than there is the difference between a loud translation
/// and a file that misparses.
pub fn operator_cost_from_sas(cost: f64) -> u64 {
    assert!(
        cost.is_finite() && cost >= 0.0 && cost.fract() == 0.0 && cost <= u64::MAX as f64,
        "operator cost {cost} is not a non-negative integer, which is all the SAS format carries"
    );
    cost as u64
}

impl NumericType {
    /// The letter the format spells this type with.
    pub fn as_sas(self) -> &'static str {
        match self {
            NumericType::Constant => "C",
            NumericType::Derived => "D",
            NumericType::Cost => "I",
            NumericType::Regular => "R",
        }
    }

    /// Inverse of [`Self::as_sas`].
    pub fn from_sas(letter: &str) -> Option<Self> {
        match letter {
            "C" => Some(NumericType::Constant),
            "D" => Some(NumericType::Derived),
            "I" => Some(NumericType::Cost),
            "R" => Some(NumericType::Regular),
            _ => None,
        }
    }
}

impl Metric {
    /// The metric as the format spells it: a direction, and the numeric
    /// variable that accumulates the plan's cost.
    ///
    /// Index `0` reads as "no metric variable". That is the format's own
    /// convention and not an encoding of variable zero, so a task whose metric
    /// variable ends up first is a task without a metric — for both ways in,
    /// which is what matters here.
    pub fn from_sas(direction: char, index: usize) -> Option<Self> {
        let is_min = match direction {
            '<' => true,
            '>' => false,
            _ => return None,
        };
        Some(Metric::new(is_min, (index > 0).then_some(index)))
    }
}

impl ComparisonOperator {
    pub fn from_sas(comparator: &str) -> Option<Self> {
        match comparator {
            "<" => Some(ComparisonOperator::LessThan),
            "<=" => Some(ComparisonOperator::LessThanOrEqual),
            "=" => Some(ComparisonOperator::Equal),
            ">=" => Some(ComparisonOperator::GreaterThanOrEqual),
            ">" => Some(ComparisonOperator::GreaterThan),
            "!=" => Some(ComparisonOperator::UnEqual),
            _ => None,
        }
    }
}

impl CalOperator {
    /// The four operators a numeric axiom combines its two operands with. The
    /// format's fifth assignment token, `=`, is not one of them: an axiom
    /// defines its variable *as* the combination, so there is nothing to assign.
    pub fn from_sas(operator: &str) -> Option<Self> {
        match operator {
            "+" => Some(CalOperator::Sum),
            "-" => Some(CalOperator::Difference),
            "*" => Some(CalOperator::Product),
            "/" => Some(CalOperator::Division),
            _ => None,
        }
    }
}

impl AssignmentOperation {
    pub fn from_sas(operator: &str) -> Option<Self> {
        match operator {
            "=" => Some(AssignmentOperation::Assign),
            "+" => Some(AssignmentOperation::Plus),
            "-" => Some(AssignmentOperation::Minus),
            "*" => Some(AssignmentOperation::Times),
            "/" => Some(AssignmentOperation::Divide),
            _ => None,
        }
    }
}

/// One `variable` block, as it stands in the file.
///
/// A derived variable's axiom default is *not* part of the block: the format
/// writes it into the initial-state block instead and lets the axiom closure
/// compute the real value on top of it. An [`ExplicitVariable`] can therefore
/// only be built once the initial state is known, which is what
/// [`NumericRootTask::from_sas_parts`] does.
pub struct SasVariable {
    pub domain_size: usize,
    pub name: String,
    pub fact_names: Vec<String>,
    pub axiom_layer: Option<usize>,
}

/// A whole task in the shape the SAS+ format carries it, section by section.
///
/// Handed to [`NumericRootTask::from_sas_parts`], which is the only way into a
/// root task from this shape. A second entry point that established one
/// invariant fewer would let the text path and the direct path drift apart
/// silently.
pub struct SasTaskParts {
    pub version: u32,
    pub metric: Metric,
    pub variables: Vec<SasVariable>,
    pub numeric_variables: Vec<NumericVariable>,
    pub mutexes: Vec<Vec<ExplicitFact>>,
    /// One entry per variable, in variable order. For a derived variable this
    /// is its axiom default rather than its initial value.
    pub state: Vec<usize>,
    pub numeric_state: Vec<f64>,
    pub goals: Vec<ExplicitFact>,
    pub operators: Vec<Operator>,
    pub axioms: Vec<PropositionalAxiom>,
    pub comparison_axioms: Vec<ComparisonAxiom>,
    pub assignment_axioms: Vec<AssignmentAxiom>,
    pub global_constraint: ExplicitFact,
}

impl NumericRootTask {
    /// The one way into a root task from the SAS+ shape.
    ///
    /// Two things the format states only implicitly are established here:
    /// a derived variable's axiom default, which it writes into the initial
    /// state, and the position of the numeric conditions in the variable id
    /// space, which it leaves interleaved with the genuine propositional
    /// variables. [`NumericRootTask::new`] takes care of the rest — fact
    /// namespaces and the axiom closure of the initial state — for every task,
    /// however built.
    pub fn from_sas_parts(parts: SasTaskParts) -> Self {
        let SasTaskParts {
            version,
            metric,
            variables,
            numeric_variables,
            mutexes,
            state,
            numeric_state,
            goals,
            operators,
            axioms,
            comparison_axioms,
            assignment_axioms,
            global_constraint,
        } = parts;

        let mut task = NumericRootTask::new(
            version,
            metric,
            build_variables(variables, &state),
            numeric_variables,
            goals,
            mutexes,
            state,
            numeric_state,
            operators,
            axioms,
            comparison_axioms,
            assignment_axioms,
            global_constraint,
        );
        // A task in the SAS+ shape owns its variable ids, unlike one derived
        // from another task, so this is the only place the renumbering may
        // happen.
        task.renumber_condition_variables_last();
        task
    }
}

/// Join the variable blocks with the initial state they were written against.
///
/// The initial-state entry of a derived variable is its axiom default — the
/// value it holds until an axiom proves something else — so this is where
/// [`ExplicitVariable`]'s axiom default comes from. Non-derived variables are
/// never reset, so their entry is simply their initial value and the field is
/// never read for them.
fn build_variables(variables: Vec<SasVariable>, state: &[usize]) -> Vec<ExplicitVariable> {
    assert_eq!(
        variables.len(),
        state.len(),
        "the SAS initial state must name every variable"
    );
    variables
        .into_iter()
        .zip(state)
        .map(|(variable, &initial_value)| {
            assert!(
                initial_value < variable.domain_size,
                "initial value {initial_value} of variable {} is outside its domain of size {}",
                variable.name,
                variable.domain_size
            );
            ExplicitVariable::new(
                variable.domain_size,
                variable.name,
                variable.fact_names,
                variable.axiom_layer,
                initial_value,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every letter the format spells a numeric type with reads back as the
    /// type it was written from; a table that disagreed with itself would
    /// retype a variable on a round trip through the file.
    #[test]
    fn a_numeric_type_survives_the_format_letter() {
        for numeric_type in [
            NumericType::Constant,
            NumericType::Derived,
            NumericType::Cost,
            NumericType::Regular,
        ] {
            assert_eq!(
                NumericType::from_sas(numeric_type.as_sas()),
                Some(numeric_type)
            );
        }
    }

    #[test]
    fn an_unknown_numeric_type_letter_is_rejected() {
        assert_eq!(NumericType::from_sas("X"), None);
    }

    /// `=` is an assignment operator but not one a numeric axiom can combine
    /// its operands with, and the two tables have to disagree about it.
    #[test]
    fn assignment_and_axiom_operators_differ_on_assignment() {
        assert!(AssignmentOperation::from_sas("=").is_some());
        assert!(CalOperator::from_sas("=").is_none());
    }

    #[test]
    fn a_metric_without_a_variable_is_spelled_zero() {
        let metric = Metric::from_sas('<', 0).expect("`<` is a direction");
        assert!(metric.is_min());
        assert!(!metric.use_metric());

        let metric = Metric::from_sas('>', 3).expect("`>` is a direction");
        assert!(!metric.is_min());
        assert_eq!(metric.var_id(), Some(3));

        assert!(Metric::from_sas('=', 1).is_none());
    }

    #[test]
    #[should_panic(expected = "is not a non-negative integer")]
    fn a_fractional_operator_cost_is_rejected() {
        operator_cost_from_sas(2.5);
    }

    #[test]
    fn an_integral_operator_cost_is_the_integer_it_spells() {
        assert_eq!(operator_cost_from_sas(0.0), 0);
        assert_eq!(operator_cost_from_sas(7.0), 7);
    }

    #[test]
    fn a_negative_axiom_layer_means_no_axiom_derives_the_variable() {
        assert_eq!(axiom_layer_from_sas(-1), None);
        assert_eq!(axiom_layer_from_sas(0), Some(0));
        assert_eq!(axiom_layer_from_sas(3), Some(3));
    }

    /// Layer zero and "no layer" are different answers, and an effect that
    /// requires value zero is not an effect that requires nothing. Both fields
    /// spell absence as `-1`, so the two easiest mistakes here are the same one.
    #[test]
    fn an_effect_precondition_of_zero_is_not_the_absence_of_one() {
        assert_eq!(effect_precondition_from_sas(-1), None);
        assert_eq!(effect_precondition_from_sas(0), Some(0));
    }
}
