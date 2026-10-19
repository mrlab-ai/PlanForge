//! Invariant synthesis must merge mutually exclusive facts into one variable.
//!
//! The corpus tests translate through the fast path, which skips invariant
//! synthesis, so they never noticed when it stopped finding any invariant.

use planforge_sas::numeric_task::{AbstractNumericTask, ExplicitFact};

use crate::corpus::{assets, translate_in_memory};

/// Exactly one `timenow(?t)` holds in every state: `advance_time` deletes the
/// time point it moves away from. Without that invariant, every time point
/// becomes a binary variable of its own.
#[test]
fn hydropower_time_points_form_one_variable() {
    let dir = assets().join("numeric-pddl-files/hydropower");
    let task = translate_in_memory(&dir.join("domain.pddl"), &dir.join("pfile4.pddl"));

    let time_variables: Vec<usize> = (0..task.variables().len())
        .filter(|&var| {
            (0..task.variables()[var].domain_size()).any(|value| {
                task.get_fact_name(&ExplicitFact::propositional(var, value))
                    .starts_with("Atom timenow(")
            })
        })
        .collect();

    assert_eq!(
        time_variables.len(),
        1,
        "the timenow facts are spread over {} variables",
        time_variables.len()
    );
    assert!(task.variables()[time_variables[0]].domain_size() > 2);
}
