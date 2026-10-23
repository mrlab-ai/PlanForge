//! The task as CEGAR refines its abstractions: its concrete states, region
//! semantics, operator costs, coordinate value lattices, the constants
//! wanted splits split at, and how splits are chosen.

use planforge_sas::utils::value_lattice::ValueLattice;
use std::cell::RefCell;
use std::sync::Arc;

use anyhow::{Result, ensure};
use planforge_sas::explicit_state::ExplicitStateSpace;
use planforge_sas::numeric_task::{
    AbstractNumericTask, NumericTaskExt, NumericType, NumericValue, OperatorIndex,
};
use planforge_sas::utils::float_tolerance;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::options::CegarOptions;
use super::regression::ensure_regressable;
use super::split::{Split, SplitDimension};
use super::{FlawKind, SplitKind, SplitSelection};
use crate::evaluation::abstractions::common::coordinates::Coordinates;
use crate::evaluation::abstractions::common::regions::RegionSemantics;
use crate::evaluation::abstractions::common::task::validate_abstraction_operator;
use crate::evaluation::seed::{RANDOM_SPLIT_SEED, mix_seed, stable_text_seed};
use crate::evaluation::validate_abstractable_goal;

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
    operator_costs: Arc<[f64]>,
    numeric_lattices: Vec<ValueLattice>,
    /// Per numeric variable, the seed of its name that split choice keys
    /// mix in.
    numeric_name_seeds: Vec<u64>,
    wanted_split_boundaries: Vec<NumericValue>,
    split_options: SplitOptions,
    random_split_rng: Option<RefCell<SmallRng>>,
}

impl<'task> CegarSemantics<'task> {
    /// The semantics a CEGAR run with `options` refines abstractions of
    /// `task` under. Fails if the run regresses and `task` is not
    /// regressable.
    pub(crate) fn for_options(
        task: &'task dyn AbstractNumericTask,
        options: &CegarOptions,
    ) -> Result<Self> {
        let semantics = Self::new(
            task,
            SplitOptions {
                kind: options.split_kind,
                selection: options.split_selection,
                rank: options.split_selection_rank,
                random_seed: options.random_seed,
            },
        )?;
        if options.flaw_kind == FlawKind::Regression {
            ensure_regressable(task, semantics.coordinates())?;
        }
        Ok(semantics)
    }

    pub(crate) fn new(
        task: &'task dyn AbstractNumericTask,
        split_options: SplitOptions,
    ) -> Result<Self> {
        validate_abstractable_goal(task).map_err(anyhow::Error::msg)?;
        for (op_id, op) in task.get_operators().iter().enumerate() {
            validate_abstraction_operator(task, op, OperatorIndex::from_usize(op_id))?;
        }
        let regions = RegionSemantics::new(task)?;
        // Costs within the search tolerance below zero are zero.
        let operator_costs = (0..task.get_operators().len())
            .map(|op_id| {
                let cost = task.abstract_operator_cost(OperatorIndex::from_usize(op_id));
                ensure!(
                    cost >= -float_tolerance::SEARCH_EPSILON,
                    "operator {op_id} has the negative cost {cost}"
                );
                Ok(cost.max(0.0))
            })
            .collect::<Result<_>>()?;
        let mut wanted_split_boundaries = task
            .numeric_variables()
            .iter()
            .enumerate()
            .filter(|(_, variable)| variable.get_type() == &NumericType::Constant)
            .map(|(numeric_var_id, _)| task.get_initial_numeric_state_values()[numeric_var_id])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        wanted_split_boundaries.sort();
        wanted_split_boundaries.dedup_by(|left, right| *left == *right);
        let random_split_rng = (split_options.selection == SplitSelection::Random).then(|| {
            RefCell::new(SmallRng::seed_from_u64(
                split_options.random_seed.unwrap_or(RANDOM_SPLIT_SEED),
            ))
        });
        let numeric_lattices = regions.coordinates().value_lattices();
        let numeric_name_seeds = task
            .numeric_variables()
            .iter()
            .map(|variable| stable_text_seed(variable.name()))
            .collect();
        Ok(Self {
            states: ExplicitStateSpace::new(task),
            regions,
            operator_costs,
            numeric_lattices,
            numeric_name_seeds,
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

    /// The nonnegative cost of every operator.
    pub(crate) fn operator_costs(&self) -> &[f64] {
        &self.operator_costs
    }

    /// The costs, shared with an abstraction that outlives the task borrow.
    pub(crate) fn shared_operator_costs(&self) -> Arc<[f64]> {
        Arc::clone(&self.operator_costs)
    }

    /// A lattice containing each numeric coordinate's reachable values.
    pub(crate) fn numeric_lattices(&self) -> &[ValueLattice] {
        &self.numeric_lattices
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
            } => numeric_split_choice_key(
                self.numeric_name_seeds[var_id.index()],
                *boundary,
                *lower_includes_boundary,
            ),
        }
    }
}

/// The choice key of a numeric split of the variable whose name has seed
/// `name_seed`.
pub(crate) fn numeric_split_choice_key(
    name_seed: u64,
    boundary: NumericValue,
    lower_closed: bool,
) -> u64 {
    mix_seed(name_seed ^ boundary.to_bits()) ^ (u64::from(lower_closed) << 63)
}
