//! The initial split a restart of a standard collection starts CEGAR from,
//! as domain abstractions make it: a propositional variable is split into its
//! initial value and the rest, a goal condition at the initial values of its
//! numeric operands. A Cartesian abstraction applies the split to every
//! abstract state it divides.

use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::*;

/// Splits `working` on the encoded variable `var`; see the module
/// documentation. Blacklisted variables stay unsplit.
pub(super) fn apply_initial_split(
    working: &mut WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    config: &CartesianAbstractionConfig,
    initial_prop_values: &[ExplicitValueIndex],
    var: VariableIndex,
) -> Result<()> {
    let task = semantics.task();
    let num_prop_vars = task.variables().len();
    if var.index() >= num_prop_vars {
        // A numeric variable gets no split of its own, as in domain abstractions.
        return Ok(());
    }
    if task.numeric_conditions().is_condition_var(var) {
        let mut rng = SmallRng::seed_from_u64(
            config
                .cegar
                .random_seed
                .unwrap_or(crate::evaluation::DEFAULT_RANDOM_SEED),
        );
        let coordinates = semantics.cegar().coordinates();
        let initial_numeric = coordinates.initial_values();
        for goal_id in 0..task.get_num_goals() {
            if task.get_goal_fact(goal_id).var_index() != var {
                continue;
            }
            let condition = task
                .numeric_conditions()
                .for_var(var)
                .expect("a condition variable has its condition");
            for numeric_var in coordinates.refinement_dimensions(condition) {
                if config.blacklisted_numeric_var_ids.contains(&numeric_var)
                    || !coordinates.is_coordinate(numeric_var)
                {
                    continue;
                }
                let boundary = initial_numeric[numeric_var.index()];
                let lower_includes_boundary = rng.gen_range(0..2) == 0;
                let integer_lattice =
                    semantics.cegar().numeric_integer_lattice()[numeric_var.index()];
                split_every_state(working, semantics, |state_id, region| {
                    region.numeric[numeric_var.index()]
                        .can_split_at(boundary, lower_includes_boundary)
                        .then_some(Split::Numeric {
                            state_id,
                            var_id: numeric_var,
                            boundary,
                            lower_includes_boundary,
                            witness_value: boundary,
                            integer_lattice,
                        })
                })?;
            }
        }
        return Ok(());
    }
    if config.blacklisted_prop_var_ids.contains(&var) {
        return Ok(());
    }
    let initial = initial_prop_values[var.index()];
    split_every_state(working, semantics, |state_id, region| {
        let values = &region.propositions()[var.index()];
        (values.len() > 1 && values.binary_search(&initial).is_ok()).then(|| Split::Propositional {
            state_id,
            var_id: var,
            wanted: vec![initial],
            witness_value: initial,
        })
    })
}

/// Applies the split `split_of` returns for each abstract state that exists
/// before the first split.
fn split_every_state(
    working: &mut WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    split_of: impl Fn(usize, &CartesianSet) -> Option<Split>,
) -> Result<()> {
    for state_id in 0..working.states.len() {
        if let Some(split) = split_of(state_id, &working.states[state_id]) {
            working.apply_split(semantics, split)?;
        }
    }
    Ok(())
}
