//! The task as CEGAR refines its abstractions: its concrete states, region
//! semantics, operator costs, which numeric dimensions live on an integer lattice, the constants
//! wanted splits split at, and how splits are chosen.

use std::cell::RefCell;

use anyhow::Result;
use planforge_sas::explicit_state::ExplicitStateSpace;
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentOperation, NumericTaskExt, NumericType, NumericValue,
    OperatorIndex,
};
use planforge_sas::utils::float_tolerance;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::split::{Split, SplitDimension, approximately_equal};
use super::{SplitKind, SplitSelection};
use crate::evaluation::abstractions::collections::strategy::{mix_seed, stable_text_seed};
use crate::evaluation::abstractions::common::coordinates::Coordinates;
use crate::evaluation::abstractions::common::regions::RegionSemantics;
use crate::evaluation::abstractions::common::task::validate_abstraction_operator;
use crate::evaluation::validate_abstractable_goal;

/// The seed of random split selection when none is configured.
const RANDOM_SPLIT_SEED: u64 = 2011;

/// How CEGAR places and chooses splits.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SplitOptions {
    pub(crate) kind: SplitKind,
    pub(crate) selection: SplitSelection,
    /// Choose the split at this rank among the tied best, ordered by a
    /// deterministic key, instead of by the seed.
    pub(crate) rank: Option<usize>,
    pub(crate) random_seed: Option<u64>,
}

pub(crate) struct CegarSemantics<'task> {
    states: ExplicitStateSpace<'task>,
    regions: RegionSemantics<'task>,
    operator_costs: Vec<NumericValue>,
    numeric_integer_lattice: Vec<bool>,
    wanted_split_boundaries: Vec<NumericValue>,
    split_options: SplitOptions,
    random_split_rng: Option<RefCell<SmallRng>>,
}

impl<'task> CegarSemantics<'task> {
    pub(crate) fn new(
        task: &'task dyn AbstractNumericTask,
        split_options: SplitOptions,
    ) -> Result<Self> {
        validate_abstractable_goal(task).map_err(anyhow::Error::msg)?;
        for (op_id, op) in task.get_operators().iter().enumerate() {
            validate_abstraction_operator(task, op, OperatorIndex::from_usize(op_id))?;
        }
        let regions = RegionSemantics::new(task)?;
        // Task views that move the initial state keep these costs pinned to
        // the base task, so the goal distances stay admissible there.
        let operator_costs = (0..task.get_operators().len())
            .map(|op_id| task.abstract_operator_cost(OperatorIndex::from_usize(op_id)))
            .collect();
        let mut wanted_split_boundaries = task
            .numeric_variables()
            .iter()
            .enumerate()
            .filter(|(_, variable)| variable.get_type() == &NumericType::Constant)
            .filter_map(|(numeric_var_id, _)| {
                task.get_initial_numeric_state_values()
                    .get(numeric_var_id)
                    .copied()
                    .filter(|value| value.is_finite())
            })
            .map(float_tolerance::canonicalize)
            .collect::<Vec<_>>();
        wanted_split_boundaries.sort_by(NumericValue::total_cmp);
        wanted_split_boundaries.dedup_by(|left, right| {
            float_tolerance::canonical_bits(*left) == float_tolerance::canonical_bits(*right)
        });
        let random_split_rng = (split_options.selection == SplitSelection::Random).then(|| {
            RefCell::new(SmallRng::seed_from_u64(
                split_options.random_seed.unwrap_or(RANDOM_SPLIT_SEED),
            ))
        });
        let numeric_integer_lattice = numeric_integer_lattice(task, regions.coordinates());
        Ok(Self {
            states: ExplicitStateSpace::new(task),
            regions,
            operator_costs,
            numeric_integer_lattice,
            wanted_split_boundaries,
            split_options,
            random_split_rng,
        })
    }

    pub(crate) fn task(&self) -> &'task dyn AbstractNumericTask {
        self.regions.task()
    }

    /// The concrete states that flaws are witnessed in.
    pub(crate) fn states(&self) -> &ExplicitStateSpace<'task> {
        &self.states
    }

    pub(crate) fn regions(&self) -> &RegionSemantics<'task> {
        &self.regions
    }

    pub(crate) fn coordinates(&self) -> &Coordinates<'task> {
        self.regions.coordinates()
    }

    pub(crate) fn operator_costs(&self) -> &[NumericValue] {
        &self.operator_costs
    }

    /// Per numeric dimension, whether every reachable value is an integer.
    pub(crate) fn numeric_integer_lattice(&self) -> &[bool] {
        &self.numeric_integer_lattice
    }

    /// The task's finite constants, sorted: where wanted splits split.
    pub(crate) fn wanted_split_boundaries(&self) -> &[NumericValue] {
        &self.wanted_split_boundaries
    }

    pub(crate) fn split_kind(&self) -> SplitKind {
        self.split_options.kind
    }

    pub(crate) fn split_selection(&self) -> SplitSelection {
        self.split_options.selection
    }

    /// The index of `keys` the seed picks for decision `tag`; the first one
    /// without a seed.
    pub(crate) fn choose_keyed_index(&self, keys: &[u64], tag: u64) -> usize {
        debug_assert!(
            !keys.is_empty(),
            "cannot choose from an empty candidate set"
        );
        let Some(seed) = self.split_options.random_seed else {
            return 0;
        };
        keys.iter()
            .enumerate()
            .min_by_key(|(_, key)| mix_seed(seed ^ tag ^ **key))
            .map(|(index, _)| index)
            .expect("nonempty key set has no minimum")
    }

    /// The candidate to refine among equally good `candidates`: the one at
    /// the configured rank in a deterministic order, or the one the seed picks.
    pub(crate) fn choose_split_index(&self, candidates: &[Split], tag: u64) -> usize {
        debug_assert!(
            !candidates.is_empty(),
            "cannot choose from an empty split set"
        );
        if let Some(rank) = self.split_options.rank {
            let mut indices = (0..candidates.len()).collect::<Vec<_>>();
            indices.sort_by_key(|&index| {
                let dimension = match candidates[index].dimension() {
                    SplitDimension::Propositional(var_id) => (0usize, var_id),
                    SplitDimension::Numeric(var_id) => (1usize, var_id),
                };
                (dimension, self.split_choice_key(&candidates[index]))
            });
            return indices[rank % indices.len()];
        }
        let keys = candidates
            .iter()
            .map(|split| self.split_choice_key(split))
            .collect::<Vec<_>>();
        self.choose_keyed_index(&keys, tag)
    }

    /// A uniformly random index below `candidate_count`.
    pub(crate) fn choose_random_split_index(&self, candidate_count: usize) -> usize {
        debug_assert!(candidate_count > 0, "cannot choose from an empty split set");
        self.random_split_rng
            .as_ref()
            .expect("random split selection must initialize its RNG")
            .borrow_mut()
            .gen_range(0..candidate_count)
    }

    /// A key that identifies a split independently of the order it was found in.
    pub(crate) fn split_choice_key(&self, split: &Split) -> u64 {
        match split {
            Split::Propositional { var_id, wanted, .. } => {
                wanted.iter().fold(var_id.index() as u64, |key, value| {
                    mix_seed(key ^ (value.index() as u64))
                })
            }
            Split::Numeric {
                var_id,
                boundary,
                lower_includes_boundary,
                ..
            } => {
                let variable_name = self.task().numeric_variables()[var_id.index()].name();
                numeric_split_choice_key(variable_name, *boundary, *lower_includes_boundary)
            }
        }
    }
}

pub(crate) fn numeric_split_choice_key(
    variable_name: &str,
    boundary: NumericValue,
    lower_closed: bool,
) -> u64 {
    mix_seed(stable_text_seed(variable_name) ^ boundary.to_bits()) ^ (u64::from(lower_closed) << 63)
}

/// Per numeric dimension, whether the initial value and every effect keep it
/// on the integers.
fn numeric_integer_lattice(
    task: &dyn AbstractNumericTask,
    coordinates: &Coordinates<'_>,
) -> Vec<bool> {
    let initial_numeric = task.get_initial_numeric_state_values();
    let mut lattice = initial_numeric
        .iter()
        .map(|&value| approximately_equal(value.value(), value.value().round()))
        .collect::<Vec<_>>();
    for op in task.get_operators() {
        for effect in op.assignment_effects() {
            let rhs = initial_numeric[effect.var_id().index()];
            let preserves_integers = match effect.operation() {
                AssignmentOperation::Plus
                | AssignmentOperation::Minus
                | AssignmentOperation::Assign
                | AssignmentOperation::Times => {
                    approximately_equal(rhs.value(), rhs.value().round())
                }
                AssignmentOperation::Divide => approximately_equal(rhs.value().abs(), 1.0),
            };
            lattice[effect.affected_var_id().index()] &= preserves_integers;
        }
    }
    for (numeric_var_id, view) in coordinates.views().iter().enumerate() {
        let Some(view) = view else {
            continue;
        };
        let initial_value = view.evaluate(initial_numeric);
        lattice[numeric_var_id] =
            approximately_equal(initial_value.value(), initial_value.value().round())
                && (0..task.get_operators().len()).all(|op_id| {
                    view.operator_delta(OperatorIndex::from_usize(op_id))
                        .is_ok_and(|delta| {
                            approximately_equal(delta.value(), delta.value().round())
                        })
                });
    }
    lattice
}
