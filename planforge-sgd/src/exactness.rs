//! Exhaustive verification that the transcription is exact at integrality.
//!
//! The proposition the whole method rests on is that, at one-hot action and
//! state rows, zero residual is equivalent to valid SAS+ semantics. This module
//! checks that by brute force on randomly generated tiny tasks: it enumerates
//! *every* combination of a one-hot action row, a one-hot current state row and a
//! one-hot next state row, and compares the residuals against what
//! [`planforge_sas::state_registry::StateRegistry`] actually does.
//!
//! Comparing against the registry rather than a hand-written reference is the
//! point: the registry is the semantics the exact verifier uses, so this pins the
//! transcription to the real thing rather than to a second implementation that
//! could drift the same way.
//!
//! The generator deliberately produces the awkward cases: several effects on the
//! same variable within one operator (agreeing and conflicting), conditional
//! effects, and effects carrying `precondition_value`.

#![cfg(test)]

use std::sync::Arc;

use planforge_sas::numeric_task::{AbstractNumericTask, NumericRootTask};
use planforge_sas::state_registry::StateRegistry;

use crate::residuals::{Assignment, Masses, evaluate, masses_at};
use crate::testing::{Rng, random_task};
use crate::transcription::Transcription;

/// Every combination of values over the transcription's variables.
fn all_value_combinations(transcription: &Transcription) -> Vec<Vec<usize>> {
    let mut combinations = vec![Vec::new()];
    for &size in transcription.var_domain() {
        let mut next = Vec::new();
        for prefix in &combinations {
            for value in 0..size as usize {
                let mut extended = prefix.clone();
                extended.push(value);
                next.push(extended);
            }
        }
        combinations = next;
    }
    combinations
}

const TOLERANCE: f64 = 1e-9;

struct Counters {
    integral_points: usize,
    multi_effect_groups: usize,
    conditional_effects: usize,
}

/// Check all three exactness claims on one task. Returns the work done, so the
/// test can assert the awkward cases were actually exercised.
fn check_task(task: NumericRootTask, counters: &mut Counters) {
    let transcription = match Transcription::build(&task) {
        Ok(transcription) => transcription,
        // A random task can be provably unsolvable (contradictory goals); that
        // is a legitimate outcome and there is nothing to check.
        Err(_) => return,
    };
    counters.conditional_effects += transcription.cond_effect().len();
    for group in 0..transcription.num_groups() {
        if transcription.group_effects(group).len() > 1 {
            counters.multi_effect_groups += 1;
        }
    }

    let arc = Arc::new(task);
    let mut registry = StateRegistry::for_task(arc.clone());
    let combinations = all_value_combinations(&transcription);
    let num_variables = transcription.num_variables();
    let mut masses = Masses::zeros(&transcription);

    for current in &combinations {
        // Register the state. Values are indexed by *task* variable, so slot 0
        // is the derived variable; the registry's axiom pass fixes it.
        let mut values = vec![1u64];
        values.extend(current.iter().map(|&value| value as u64));
        let state = registry
            .register_state(values, Vec::new())
            .expect("registering an enumerated state failed");

        for action in 0..transcription.num_actions() {
            let expected: Vec<usize> = match transcription.action_source(action) {
                None => current.clone(),
                Some(op_index) => {
                    let operator = &arc.get_operators()[op_index];
                    let successor = registry
                        .get_successor_state(&state, operator)
                        .expect("applying an operator failed");
                    transcription
                        .primary_vars()
                        .iter()
                        .map(|&task_var| {
                            successor
                                .get_propositional_value(&registry, task_var)
                                .expect("variable is in range")
                        })
                        .collect()
                }
            };

            // Claim B: precondition residuals vanish exactly when the operator
            // is applicable in the current state.
            let applicable = match transcription.action_source(action) {
                None => true,
                Some(op_index) => arc.get_operators()[op_index].preconditions().iter().all(
                    |fact: &planforge_sas::numeric_task::ExplicitFact| {
                        state
                            .get_propositional_value(&registry, fact.var())
                            .expect("variable is in range")
                            == fact.value()
                    },
                ),
            };

            // Claim A: transition residuals vanish for exactly one next state,
            // namely the registry's successor.
            for candidate in &combinations {
                let mut assignment = Assignment::zeros(&transcription, 1);
                assignment.set_action_one_hot(0, action);
                assignment.set_state_one_hot(&transcription, 0, current);
                assignment.set_state_one_hot(&transcription, 1, candidate);

                let residuals = evaluate(&transcription, &assignment);
                let transition_zero = residuals
                    .transition
                    .iter()
                    .all(|family| family.iter().all(|&r| r <= TOLERANCE));
                assert_eq!(
                    transition_zero,
                    candidate == &expected,
                    "transition residuals disagree with the registry: \
                     action {action}, current {current:?}, candidate {candidate:?}, \
                     registry successor {expected:?}"
                );

                let precondition_zero = residuals.precondition.iter().all(|&r| r <= TOLERANCE);
                assert_eq!(
                    precondition_zero, applicable,
                    "precondition residuals disagree with applicability: \
                     action {action}, current {current:?}"
                );

                counters.integral_points += 1;
            }

            // Claim C: the telescoping identity, per group, which is what makes
            // D = Chg - Add non-negative.
            let mut assignment = Assignment::zeros(&transcription, 1);
            assignment.set_action_one_hot(0, action);
            assignment.set_state_one_hot(&transcription, 0, current);
            masses_at(&transcription, &assignment, 0, &mut masses);
            for var in 0..num_variables {
                let offset = transcription.var_offset()[var] as usize;
                let size = transcription.var_domain()[var] as usize;
                let add_sum: f64 = masses.add[offset..offset + size].iter().sum();
                assert!(
                    (add_sum - masses.chg[var]).abs() <= TOLERANCE,
                    "sum_d Add != Chg for variable {var}: {add_sum} vs {}",
                    masses.chg[var]
                );
                for fact in offset..offset + size {
                    assert!(
                        masses.delete(&transcription, fact) >= -TOLERANCE,
                        "D < 0 for fact {fact}"
                    );
                }
            }
        }
    }
}

#[test]
fn transcription_is_exact_at_integrality_against_the_state_registry() {
    let mut rng = Rng::new(0x5EED_1234_ABCD_0001);
    let mut counters = Counters {
        integral_points: 0,
        multi_effect_groups: 0,
        conditional_effects: 0,
    };

    for _ in 0..400 {
        let task = random_task(&mut rng);
        check_task(task, &mut counters);
    }

    // Guard against a generator that quietly stops producing the hard cases;
    // without these the test would still pass while checking nothing useful.
    assert!(
        counters.integral_points > 10_000,
        "too few integral points checked: {}",
        counters.integral_points
    );
    assert!(
        counters.multi_effect_groups > 100,
        "generator produced too few multi-effect groups: {}",
        counters.multi_effect_groups
    );
    assert!(
        counters.conditional_effects > 100,
        "generator produced too few conditional effects: {}",
        counters.conditional_effects
    );
    eprintln!(
        "checked {} integral points; {} multi-effect groups, {} conditional effects",
        counters.integral_points, counters.multi_effect_groups, counters.conditional_effects
    );
}

/// Fractional check of the telescoping identity: it must hold for *any* `φ` in
/// `[0,1]`, not only at integrality, since that is what keeps `D >= 0` during
/// optimization rather than merely at the solution.
#[test]
fn add_mass_sums_to_change_mass_for_fractional_assignments() {
    let mut rng = Rng::new(0xC0FF_EE00_1234_5678);

    for _ in 0..200 {
        let task = random_task(&mut rng);
        let Ok(transcription) = Transcription::build(&task) else {
            continue;
        };
        let mut masses = Masses::zeros(&transcription);
        let mut assignment = Assignment::zeros(&transcription, 1);

        // Random fractional action row (normalized) and random fractional state
        // rows (normalized per variable).
        let row = assignment.action_row_mut(0);
        let mut total = 0.0;
        for slot in row.iter_mut() {
            *slot = (rng.below(1000) as f64 + 1.0) / 1000.0;
            total += *slot;
        }
        for slot in row.iter_mut() {
            *slot /= total;
        }
        for t in 0..2 {
            for var in 0..transcription.num_variables() {
                let offset = transcription.var_offset()[var] as usize;
                let size = transcription.var_domain()[var] as usize;
                let row = assignment.state_row_mut(t);
                let mut total = 0.0;
                for slot in row[offset..offset + size].iter_mut() {
                    *slot = (rng.below(1000) as f64 + 1.0) / 1000.0;
                    total += *slot;
                }
                for slot in row[offset..offset + size].iter_mut() {
                    *slot /= total;
                }
            }
        }

        masses_at(&transcription, &assignment, 0, &mut masses);
        for var in 0..transcription.num_variables() {
            let offset = transcription.var_offset()[var] as usize;
            let size = transcription.var_domain()[var] as usize;
            let add_sum: f64 = masses.add[offset..offset + size].iter().sum();
            assert!(
                (add_sum - masses.chg[var]).abs() <= 1e-12,
                "fractional sum_d Add != Chg for variable {var}: {add_sum} vs {}",
                masses.chg[var]
            );
            assert!(
                masses.chg[var] <= 1.0 + 1e-12,
                "Chg exceeds one for variable {var}: {}",
                masses.chg[var]
            );
            for fact in offset..offset + size {
                assert!(
                    masses.delete(&transcription, fact) >= -1e-12,
                    "fractional D < 0 for fact {fact}"
                );
            }
        }
    }
}
