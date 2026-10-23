//! The splits a restart of a standard collection starts CEGAR from, the same
//! for domain and Cartesian abstractions: a propositional variable is split
//! into its initial value and the rest; a condition variable's goal
//! conditions are split at the initial values of the coordinates they refine
//! along, so an operator can change them. A numeric variable gets no split of
//! its own: its restart starts from the trivial abstraction, as with the
//! `init_value` initial splits of the numeric Fast Downward collections this
//! ports.

use std::collections::HashSet;

use planforge_sas::numeric_task::{ExplicitValueIndex, NumericValue, VariableIndex};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::semantics::CegarSemantics;
use super::split::{Split, numeric_split_intervals};
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;

/// A variable a restart of a standard collection splits initially, and may
/// blacklist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InitialSplitVar {
    Propositional(VariableIndex),
    Numeric(VariableIndex),
}

/// Which variables refinement starts from and may not refine.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RefinementScope {
    /// The variable the trivial abstraction is split on before CEGAR; none
    /// for the trivial abstraction.
    pub initial_split_var: Option<InitialSplitVar>,
    pub blacklisted_prop_var_ids: HashSet<VariableIndex>,
    pub blacklisted_numeric_var_ids: HashSet<VariableIndex>,
}

/// One split of the initial abstraction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum InitialSplit {
    /// The initial value of `var_id` against its other values.
    Propositional {
        var_id: VariableIndex,
        initial_value: ExplicitValueIndex,
    },
    /// Numeric coordinate `var_id` at its initial value.
    Numeric {
        var_id: VariableIndex,
        boundary: NumericValue,
        lower_includes_boundary: bool,
        integer_lattice: bool,
    },
}

impl InitialSplit {
    /// The split of abstract state `state_id` with region `region`, if it
    /// divides the state.
    pub(crate) fn in_state(self, state_id: usize, region: &CartesianSet) -> Option<Split> {
        match self {
            Self::Propositional {
                var_id,
                initial_value,
            } => {
                let values = &region.propositions()[var_id.index()];
                (values.len() > 1 && values.binary_search(&initial_value).is_ok()).then(|| {
                    Split::Propositional {
                        state_id,
                        var_id,
                        wanted: vec![initial_value],
                        witness_value: initial_value,
                    }
                })
            }
            Self::Numeric {
                var_id,
                boundary,
                lower_includes_boundary,
                integer_lattice,
            } => numeric_split_intervals(
                region.numeric[var_id.index()],
                boundary,
                lower_includes_boundary,
                integer_lattice,
            )
            .map(|_| Split::Numeric {
                state_id,
                var_id,
                boundary,
                lower_includes_boundary,
                witness_value: boundary,
                integer_lattice,
            }),
        }
    }
}

/// The splits of the initial abstraction that `scope` asks for, in the
/// order to apply them: none without an initial split variable, for a
/// numeric one, and for a blacklisted propositional variable that is no
/// condition variable; a condition variable skips its blacklisted
/// coordinates instead. Which side of a numeric boundary
/// its own value falls on is drawn with `random_seed`, which every restart
/// with an initial split has.
pub(crate) fn initial_splits(
    semantics: &CegarSemantics<'_>,
    scope: &RefinementScope,
    random_seed: Option<u64>,
) -> Vec<InitialSplit> {
    let var = match scope.initial_split_var {
        None | Some(InitialSplitVar::Numeric(_)) => return Vec::new(),
        Some(InitialSplitVar::Propositional(var)) => var,
    };
    let task = semantics.task();
    if !task.numeric_conditions().is_condition_var(var) {
        if scope.blacklisted_prop_var_ids.contains(&var) {
            return Vec::new();
        }
        return vec![InitialSplit::Propositional {
            var_id: var,
            initial_value: task.get_initial_propositional_state_values()[var.index()],
        }];
    }
    assert!(
        task.goals().iter().any(|goal| goal.var_index() == var),
        "condition variable {var} is an initial split candidate only as a goal variable"
    );
    let coordinates = semantics.coordinates();
    let initial_values = coordinates.initial_values();
    let condition = task
        .numeric_conditions()
        .for_var(var)
        .expect("a condition variable has its condition");
    let mut rng = SmallRng::seed_from_u64(
        random_seed.expect("a restart with an initial split has its own seed"),
    );
    let mut splits = Vec::new();
    for numeric_var in coordinates.refinement_dimensions(condition) {
        assert!(
            coordinates.is_coordinate(numeric_var),
            "condition {var} refines along non-coordinate {numeric_var}"
        );
        if scope.blacklisted_numeric_var_ids.contains(&numeric_var) {
            continue;
        }
        splits.push(InitialSplit::Numeric {
            var_id: numeric_var,
            boundary: initial_values[numeric_var.index()],
            lower_includes_boundary: rng.gen_range(0..2) == 0,
            integer_lattice: semantics.numeric_integer_lattice()[numeric_var.index()],
        });
    }
    splits
}
