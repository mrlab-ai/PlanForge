//! The check that every fact names its variable in the right namespace.

use super::*;

/// Panic unless every fact `task` exposes names the namespace the task's own
/// numeric conditions put its variable in.
///
/// A mistagged fact is invisible later: it still denotes a well-formed
/// variable, just the wrong kind of one, so the search produces a wrong plan
/// and no crash. That is why this is an assertion and not a diagnostic.
pub fn assert_fact_namespaces(task: &Task) {
    let conditions = task.numeric_conditions();
    let check = |fact: &ExplicitFact, origin: &dyn fmt::Display| {
        assert_ne!(
            fact.namespace(),
            FactNamespace::NumericVariable,
            "{origin}: {fact:?} names a domain abstraction's private numeric id space, \
             which is not a variable of this task at all"
        );
        assert!(
            fact.var() < conditions.num_propositional_vars(),
            "{origin}: {fact:?} names variable {}, past the task's {} propositional variables",
            fact.var(),
            conditions.num_propositional_vars()
        );
        let expected = conditions.namespace_of(fact.var_index());
        assert_eq!(
            fact.namespace(),
            expected,
            "{origin}: {fact:?} is tagged {:?}, but variable {} belongs to {expected:?}",
            fact.namespace(),
            fact.var()
        );
    };

    for (operator_id, operator) in task.operators().iter().enumerate() {
        let origin = format_args!("operator {operator_id}").to_string();
        for precondition in operator.preconditions() {
            check(precondition, &origin);
        }
        for effect in operator.effects() {
            for condition in effect.conditions() {
                check(condition, &origin);
            }
        }
        for effect in operator.assignment_effects() {
            for condition in effect.conditions() {
                check(condition, &origin);
            }
        }
    }
    for (axiom_id, axiom) in task.axioms().iter().enumerate() {
        let origin = format_args!("axiom {axiom_id}").to_string();
        for condition in axiom.conditions() {
            check(condition, &origin);
        }
    }
    for goal in task.goals() {
        check(goal, &"goal");
    }
}

/// [`assert_fact_namespaces`] in debug builds only. The check walks every fact
/// of the task, which is linear in the task size and therefore too costly to
/// pay for on every release run.
pub fn debug_assert_fact_namespaces(task: &Task) {
    if cfg!(debug_assertions) {
        assert_fact_namespaces(task);
    }
}
