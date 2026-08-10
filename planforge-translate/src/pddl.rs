pub mod actions;
pub mod axioms;
pub mod conditions;
pub mod effects;
pub mod f_expression;
pub mod functions;
pub mod pddl_types;
pub mod predicates;
pub mod tasks;

/// A substitution of the variable names in an argument list.
///
/// The pipeline applies two of them, at opposite ends: normalization renames an
/// action's parameters apart from every other action's, and grounding binds them
/// to objects. Both walk the same argument lists in the same terms -- atoms,
/// primitive numeric expressions, assignments -- so the walk is written once
/// here and the substitution is what varies.
///
/// Implementors are chosen statically at every call site, so nothing is
/// dispatched dynamically on the grounding path.
pub trait Substitution {
    /// What `name` stands for under this substitution. A name the substitution
    /// does not mention stands for itself, which is what an object or any other
    /// constant does.
    fn resolve<'a>(&'a self, name: &'a str) -> &'a str;
}

/// A renaming maps the names it lists and leaves the rest alone.
impl Substitution for std::collections::HashMap<String, String> {
    fn resolve<'a>(&'a self, name: &'a str) -> &'a str {
        self.get(name).map_or(name, String::as_str)
    }
}

/// `args` with every name replaced by what `substitution` resolves it to.
pub fn substitute(args: &[String], substitution: &impl Substitution) -> Vec<String> {
    args.iter()
        .map(|name| substitution.resolve(name).to_owned())
        .collect()
}

// Re-export the types the translation pipeline names most often.
pub use actions::{Action, PropositionalAction};
pub use axioms::{Axiom, InstantiatedNumericAxiom, NumericAxiom, PropositionalAxiom};
pub use conditions::{
    Atom, Condition, Conjunction, Disjunction, ExistentialCondition, FunctionComparison,
    NegatedAtom, NegatedFunctionComparison, UniversalCondition,
};
pub use effects::{
    ConditionalEffect, ConjunctiveEffect, Effect, NumericEffect, SimpleEffect, UniversalEffect,
};
pub use f_expression::{
    AdditiveInverse, ArithmeticExpression, FunctionAssignment, FunctionalExpression,
    NumericConstant, PrimitiveNumericExpression,
};
pub use functions::Function;
pub use pddl_types::{Type, TypedObject};
pub use predicates::Predicate;
pub use tasks::{DerivedFunctionAdministrator, Requirements, Task};
