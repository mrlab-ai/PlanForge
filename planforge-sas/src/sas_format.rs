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

#[cfg(test)]
mod tests;

use crate::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use crate::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex,
    ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts, NumericType, NumericValue,
    NumericVariable, Operator, OperatorCost, VariableIndex,
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
pub fn effect_precondition_from_sas(precondition: i32) -> Option<ExplicitValueIndex> {
    u32::try_from(precondition)
        .ok()
        .map(ExplicitValueIndex::new)
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
    pub fn from_sas(direction: char, index: Option<VariableIndex>) -> Option<Self> {
        let is_min = match direction {
            '<' => true,
            '>' => false,
            _ => return None,
        };
        Some(Metric::new(is_min, index))
    }

    /// Inverse of [`Self::from_sas`]: a task without a metric variable is
    /// written with index `0`.
    pub fn as_sas(&self) -> (char, Option<VariableIndex>) {
        let direction = if self.is_min() { '<' } else { '>' };
        (direction, self.var_id())
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
/// apart until `Self::into_operator` is what lets a writer emit the block
/// without having to work out which merged precondition came from where.
pub struct SasOperator {
    pub name: String,
    pub prevail: Vec<ExplicitFact>,
    pub effects: Vec<Effect>,
    pub assignment_effects: Vec<AssignmentEffect>,
    pub cost: OperatorCost,
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
                prevail.push(ExplicitFact::propositional_from_indexes(
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
    pub state: Vec<ExplicitValueIndex>,
    pub numeric_state: Vec<NumericValue>,
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
fn build_variables(
    variables: Vec<SasVariable>,
    state: &[ExplicitValueIndex],
) -> Vec<ExplicitVariable> {
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
                initial_value.index() < variable.domain_size,
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
