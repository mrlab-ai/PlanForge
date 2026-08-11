//! What the SAS+ format *means*, without its syntax.
//!
//! Everything that goes in or out of the format goes through [`SasTaskParts`]:
//! the translator builds one from its own representation, [`crate::sas_writer`]
//! writes one out, [`crate::numeric_parser`] reads one back in, and
//! [`NumericRootTask::from_sas_parts`] turns one into the task the search runs.
//! So everything the format leaves implicit lives here rather than in any of
//! them: the token tables and their inverses, the way it spells an absent
//! value, the prevail/effect-precondition merge, and the axiom default a
//! derived variable takes from the initial state.
//!
//! [`crate::numeric_parser`] and [`crate::sas_writer`] own the text syntax and
//! nothing else.

use crate::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use crate::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
    NumericRootTask, NumericRootTaskParts, NumericType, NumericVariable, Operator,
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

/// How the format spells a field that may be absent: `-1`.
///
/// The inverse of both readers above, which ask the same question of two
/// different fields. A present zero stays `0`; the two easiest mistakes here are
/// confusing it with the absent case in either direction.
pub fn optional_value_to_sas(value: Option<usize>) -> i32 {
    match value {
        Some(value) => i32::try_from(value).expect("the SAS format carries this field as an i32"),
        None => -1,
    }
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

/// One token table of the format, in both directions.
///
/// The two directions are generated from one list, so they cannot disagree about
/// a token; a token the list does not name is rejected rather than guessed at.
macro_rules! sas_token_table {
    ($type:ident { $($variant:path => $token:literal,)+ }) => {
        impl $type {
            /// The token the format spells this value with.
            pub fn as_sas(&self) -> &'static str {
                match self {
                    $($variant => $token,)+
                }
            }

            /// Inverse of [`Self::as_sas`].
            pub fn from_sas(token: &str) -> Option<Self> {
                match token {
                    $($token => Some($variant),)+
                    _ => None,
                }
            }
        }
    };
}

sas_token_table!(NumericType {
    NumericType::Constant => "C",
    NumericType::Derived => "D",
    NumericType::Cost => "I",
    NumericType::Regular => "R",
});

sas_token_table!(ComparisonOperator {
    ComparisonOperator::LessThan => "<",
    ComparisonOperator::LessThanOrEqual => "<=",
    ComparisonOperator::Equal => "=",
    ComparisonOperator::GreaterThanOrEqual => ">=",
    ComparisonOperator::GreaterThan => ">",
    ComparisonOperator::UnEqual => "!=",
});

// The four operators a numeric axiom combines its two operands with. The
// format's fifth assignment token, `=`, is deliberately not one of them: an
// axiom defines its variable *as* the combination, so there is nothing to
// assign.
sas_token_table!(CalOperator {
    CalOperator::Sum => "+",
    CalOperator::Difference => "-",
    CalOperator::Product => "*",
    CalOperator::Division => "/",
});

sas_token_table!(AssignmentOperation {
    AssignmentOperation::Assign => "=",
    AssignmentOperation::Plus => "+",
    AssignmentOperation::Minus => "-",
    AssignmentOperation::Times => "*",
    AssignmentOperation::Divide => "/",
});

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

    /// Inverse of [`Self::from_sas`]: a task without a metric variable is
    /// written with index `0`.
    pub fn as_sas(&self) -> (char, usize) {
        let direction = if self.is_min() { '<' } else { '>' };
        (direction, self.var_id().unwrap_or(0))
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

/// One `operator` block, as it stands in the file.
///
/// The file states an operator's conditions in two places — the prevail block,
/// and the value an effect requires of the variable it writes — while an
/// [`Operator`] holds the single merged list the search checks. Keeping the two
/// apart until [`Self::into_operator`] is what lets a writer emit the block
/// without having to work out which merged precondition came from where.
pub struct SasOperator {
    pub name: String,
    pub prevail: Vec<ExplicitFact>,
    pub effects: Vec<Effect>,
    pub assignment_effects: Vec<AssignmentEffect>,
    pub cost: u64,
}

impl SasOperator {
    /// The operator the search runs.
    ///
    /// An effect that requires a value of the variable it writes contributes
    /// that requirement to the operator, after the prevail conditions and in
    /// the order the effects are listed in: an operator's preconditions are held
    /// in that order rather than sorted.
    fn into_operator(self) -> Operator {
        let SasOperator {
            name,
            mut prevail,
            effects,
            assignment_effects,
            cost,
        } = self;
        for effect in &effects {
            if let Some(precondition_value) = effect.precondition_value() {
                prevail.push(ExplicitFact::propositional(
                    effect.var_id(),
                    precondition_value,
                ));
            }
        }
        Operator::new(name, prevail, effects, assignment_effects, cost)
    }
}

/// A whole task in the shape the SAS+ format carries it, section by section.
///
/// The single intermediate of the format: [`crate::numeric_parser`] reads one,
/// [`crate::sas_writer`] writes one, the translator builds one, and
/// [`NumericRootTask::from_sas_parts`] is the only way from one into a root
/// task. A second entry point that established one invariant fewer would let
/// the text path and the direct path drift apart silently.
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
    pub operators: Vec<SasOperator>,
    pub axioms: Vec<PropositionalAxiom>,
    pub comparison_axioms: Vec<ComparisonAxiom>,
    pub assignment_axioms: Vec<AssignmentAxiom>,
    pub global_constraint: ExplicitFact,
}

impl NumericRootTask {
    /// The one way into a root task from the SAS+ shape.
    ///
    /// Two things the format states only implicitly are established here: a
    /// derived variable's axiom default, which it writes into the initial state
    /// rather than into the variable's own block, and the merge of an operator's
    /// two condition lists. [`NumericRootTask::new`] takes care of the rest —
    /// fact namespaces and the axiom closure of the initial state — for every
    /// task, however built.
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

        NumericRootTask::new(NumericRootTaskParts {
            version,
            metric,
            variables: build_variables(variables, &state),
            numeric_variables,
            goals,
            mutexes,
            state,
            numeric_state,
            operators: operators
                .into_iter()
                .map(SasOperator::into_operator)
                .collect(),
            axioms,
            comparison_axioms,
            assignment_axioms,
            global_constraint,
        })
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

    /// Every token table is written by the writer and read by the parser, so a
    /// table that disagreed with its own inverse would change an operator, a
    /// comparator or an axiom on the way through the file.
    #[test]
    fn every_operator_token_reads_back_as_what_it_was_written_from() {
        for comparator in [
            ComparisonOperator::LessThan,
            ComparisonOperator::LessThanOrEqual,
            ComparisonOperator::Equal,
            ComparisonOperator::GreaterThanOrEqual,
            ComparisonOperator::GreaterThan,
            ComparisonOperator::UnEqual,
        ] {
            assert_eq!(
                ComparisonOperator::from_sas(comparator.as_sas()),
                Some(comparator)
            );
        }
        for operator in [
            CalOperator::Sum,
            CalOperator::Difference,
            CalOperator::Product,
            CalOperator::Division,
        ] {
            assert_eq!(CalOperator::from_sas(operator.as_sas()), Some(operator));
        }
        for operation in [
            AssignmentOperation::Assign,
            AssignmentOperation::Plus,
            AssignmentOperation::Minus,
            AssignmentOperation::Times,
            AssignmentOperation::Divide,
        ] {
            assert_eq!(
                AssignmentOperation::from_sas(operation.as_sas()),
                Some(operation)
            );
        }
    }

    #[test]
    fn a_metric_without_a_variable_is_spelled_zero() {
        let metric = Metric::from_sas('<', 0).expect("`<` is a direction");
        assert!(metric.is_min());
        assert!(!metric.use_metric());
        assert_eq!(metric.as_sas(), ('<', 0));

        let metric = Metric::from_sas('>', 3).expect("`>` is a direction");
        assert!(!metric.is_min());
        assert_eq!(metric.var_id(), Some(3));
        assert_eq!(metric.as_sas(), ('>', 3));

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
    /// spell absence as `-1`, so the two easiest mistakes here are the same one
    /// -- in both directions, since the writer spells these fields too.
    #[test]
    fn an_effect_precondition_of_zero_is_not_the_absence_of_one() {
        assert_eq!(effect_precondition_from_sas(-1), None);
        assert_eq!(effect_precondition_from_sas(0), Some(0));
        assert_eq!(optional_value_to_sas(None), -1);
        assert_eq!(optional_value_to_sas(Some(0)), 0);
        assert_eq!(optional_value_to_sas(Some(3)), 3);
    }

    /// An operator's two condition lists are one list to the search, and the
    /// order they merge in is the order the file states them: the prevail
    /// conditions, then the effects' own requirements in effect order.
    #[test]
    fn an_operator_merges_its_prevail_and_effect_conditions_in_file_order() {
        let operator = SasOperator {
            name: "move".to_owned(),
            prevail: vec![ExplicitFact::propositional(7, 1)],
            effects: vec![
                Effect::new(vec![], 4, None, 1),
                Effect::new(vec![], 5, Some(0), 1),
                Effect::new(vec![], 6, Some(2), 0),
            ],
            assignment_effects: vec![],
            cost: 3,
        }
        .into_operator();

        let preconditions: Vec<(usize, usize)> = operator
            .preconditions()
            .iter()
            .map(|fact| (fact.var(), fact.value()))
            .collect();
        // Variable 4 contributes nothing: its effect applies whatever the
        // variable holds.
        assert_eq!(preconditions, [(7, 1), (5, 0), (6, 2)]);
        assert_eq!(operator.effects().len(), 3);
        assert_eq!(operator.cost(), 3);
    }
}
