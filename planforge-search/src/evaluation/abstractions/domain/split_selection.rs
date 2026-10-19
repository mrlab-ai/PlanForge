#[cfg(test)]
mod tests;

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

use planforge_sas::numeric_conditions::NumericConditions;
use planforge_sas::numeric_task::AbstractNumericTask;
use planforge_sas::utils::float_tolerance;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use super::flaws::{Flaw, NumericFlaw};

/// A flaw chosen for refinement: its index in the flaw list and the dependent
/// numeric flaws a propositional flaw is restricted to.
#[derive(Clone)]
pub struct FlawCandidate {
    pub(super) idx: usize,
    pub(super) restricted_dep: Option<Vec<NumericFlaw>>,
}
pub type ChosenFlaws = Vec<FlawCandidate>;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InitSplitMethod {
    GoalValue,
    GoalValueOrRandomIfNonGoal,
    InitValue,
    RandomValue,
    RandomPartition,
    RandomBinaryPartitionSeparatingInitGoal,
    Identity,
}

impl fmt::Display for InitSplitMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GoalValue => write!(f, "goal_value"),
            Self::GoalValueOrRandomIfNonGoal => write!(f, "goal_value_or_random_if_non_goal"),
            Self::InitValue => write!(f, "init_value"),
            Self::RandomValue => write!(f, "random_value"),
            Self::RandomPartition => write!(f, "random_partition"),
            Self::RandomBinaryPartitionSeparatingInitGoal => {
                write!(f, "random_binary_partition_separating_init_goal")
            }
            Self::Identity => write!(f, "identity"),
        }
    }
}

impl crate::config::FromOptionValue for InitSplitMethod {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "goal_value" => Ok(Self::GoalValue),
            "goal_value_or_random_if_non_goal" => Ok(Self::GoalValueOrRandomIfNonGoal),
            "init_value" => Ok(Self::InitValue),
            "random_value" => Ok(Self::RandomValue),
            "random_partition" => Ok(Self::RandomPartition),
            "random_binary_partition_separating_init_goal" => {
                Ok(Self::RandomBinaryPartitionSeparatingInitGoal)
            }
            "identity" => Ok(Self::Identity),
            other => Err(format!("invalid InitSplitMethod `{other}`")),
        }
    }
}

/// How a flaw treatment decides between equally good flaws.
pub enum TieBreak<'rng> {
    /// Shuffle them with the seeded generator.
    Random(&'rng mut SmallRng),
    /// Order them by a deterministic key and take the one at `rank % ties`.
    /// Treatments that refine every flaw have nothing to choose.
    Rank(usize),
}

fn flaw_atom_key(flaw: &Flaw) -> (u8, usize, usize, u64, bool) {
    match flaw {
        Flaw::Propositional(pf) => (0, pf.fact.var(), pf.fact.value(), 0, false),
        Flaw::Numeric(nf) => (
            1,
            nf.numeric_var_id.index(),
            0,
            float_tolerance::canonical_bits(nf.value),
            nf.include_in_lower,
        ),
    }
}

/// How `fix_flaws` chooses which flaws to refine.
///
/// This mirrors numeric-FD's `FlawTreatment` options, but our defaults aim to
/// stay deterministic.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FlawTreatmentVariants {
    RandomSingleAtom,
    MinGrowthSingleAtom,
}

impl fmt::Display for FlawTreatmentVariants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RandomSingleAtom => write!(f, "random_single_atom"),
            Self::MinGrowthSingleAtom => write!(f, "min_growth_single_atom"),
        }
    }
}

impl crate::config::FromOptionValue for FlawTreatmentVariants {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "random_single_atom" => Ok(Self::RandomSingleAtom),
            "min_growth_single_atom" => Ok(Self::MinGrowthSingleAtom),
            other => Err(format!(
                "invalid flaw_treatment `{other}`; expected random_single_atom or min_growth_single_atom"
            )),
        }
    }
}

impl FlawTreatmentVariants {
    /// The flaws to refine, best first: refinement takes the first one that
    /// still splits the abstraction. `tie_break` decides between equally good
    /// flaws; see [`TieBreak`].
    pub fn choose_flaws(
        &self,
        task: &dyn AbstractNumericTask,
        flaws: &[Flaw],
        domain_sizes: &[usize],
        numeric_domain_sizes: &[usize],
        tie_break: TieBreak<'_>,
    ) -> ChosenFlaws {
        let conditions = task.numeric_conditions();
        let rng = match tie_break {
            TieBreak::Rank(rank) => {
                return self.choose_ranked_flaws(
                    flaws,
                    conditions,
                    domain_sizes,
                    numeric_domain_sizes,
                    rank,
                );
            }
            TieBreak::Random(rng) => rng,
        };
        match self {
            Self::RandomSingleAtom => choose_single_random_flaw(flaws, rng),
            Self::MinGrowthSingleAtom => fix_single_flaw_min_growth(
                flaws,
                conditions,
                domain_sizes,
                numeric_domain_sizes,
                rng,
            ),
        }
    }
}

impl FlawTreatmentVariants {
    /// [`Self::choose_flaws`] with [`TieBreak::Rank`]: the treatment's
    /// preference order, ties broken by [`flaw_tie_key`], and the best tier
    /// rotated so the flaw at `rank % ties` comes first.
    fn choose_ranked_flaws(
        &self,
        flaws: &[Flaw],
        conditions: &NumericConditions,
        domain_sizes: &[usize],
        numeric_domain_sizes: &[usize],
        rank: usize,
    ) -> ChosenFlaws {
        match self {
            Self::RandomSingleAtom => rank_tied_best(
                flaws,
                unscored_candidates(flaws)
                    .map(|candidate| (candidate, ()))
                    .collect(),
                rank,
                |_, _| Ordering::Equal,
            ),
            Self::MinGrowthSingleAtom => rank_tied_best(
                flaws,
                min_growth_candidates(flaws, conditions, domain_sizes, numeric_domain_sizes),
                rank,
                |left, right| compare_growth_keys(*left, *right),
            ),
        }
    }
}

/// A key that orders flaws independently of the order the flaw search emitted
/// them in.
fn flaw_tie_key(flaw: &Flaw) -> ((u8, usize, usize, u64, bool), usize) {
    (flaw_atom_key(flaw), flaw.step())
}

fn unscored_candidates(flaws: &[Flaw]) -> impl Iterator<Item = FlawCandidate> + '_ {
    (0..flaws.len()).map(|idx| FlawCandidate {
        idx,
        restricted_dep: None,
    })
}

/// Order `scored` best first by `prefer`, break ties by [`flaw_tie_key`] and
/// flaw index, and rotate the best tier so that its candidate at
/// `rank % tier size` comes first. The other candidates keep their order.
fn rank_tied_best<S>(
    flaws: &[Flaw],
    mut scored: Vec<(FlawCandidate, S)>,
    rank: usize,
    prefer: impl Fn(&S, &S) -> Ordering,
) -> ChosenFlaws {
    scored.sort_by(|(left, left_score), (right, right_score)| {
        prefer(left_score, right_score)
            .then_with(|| flaw_tie_key(&flaws[left.idx]).cmp(&flaw_tie_key(&flaws[right.idx])))
            .then_with(|| left.idx.cmp(&right.idx))
    });
    if let Some((_, best)) = scored.first() {
        let tier_size = scored
            .iter()
            .take_while(|(_, score)| prefer(score, best).is_eq())
            .count();
        scored[..tier_size].rotate_left(rank % tier_size);
    }
    scored.into_iter().map(|(candidate, _)| candidate).collect()
}

pub(super) fn choose_single_random_flaw(flaws: &[Flaw], rng: &mut SmallRng) -> ChosenFlaws {
    if flaws.is_empty() {
        return vec![];
    }

    let mut candidates: Vec<FlawCandidate> = (0..flaws.len())
        .map(|i| FlawCandidate {
            idx: i,
            restricted_dep: None,
        })
        .collect();
    candidates.shuffle(rng);

    candidates
}

fn growth_key_for_domain_size(current_size: usize) -> (usize, usize) {
    assert!(current_size > 0, "abstract domain sizes are positive");
    (current_size + 1, current_size)
}

fn compare_growth_keys(left: (usize, usize), right: (usize, usize)) -> std::cmp::Ordering {
    let left_cross = (left.0 as u128) * (right.1 as u128);
    let right_cross = (right.0 as u128) * (left.1 as u128);
    left_cross.cmp(&right_cross)
}

fn best_min_growth_dependent_flaws(
    dependent_numeric_flaws: &[NumericFlaw],
    numeric_domain_sizes: &[usize],
) -> Option<(Vec<NumericFlaw>, (usize, usize))> {
    let mut by_partition_count: BTreeMap<usize, Vec<NumericFlaw>> = BTreeMap::new();
    for nf in dependent_numeric_flaws.iter().cloned() {
        let partitions = numeric_domain_sizes[nf.numeric_var_id.index()];
        by_partition_count.entry(partitions).or_default().push(nf);
    }
    let (&max_partitions, flaws) = by_partition_count.iter().next_back()?;
    Some((flaws.clone(), growth_key_for_domain_size(max_partitions)))
}

/// Every flaw with the growth (as a ratio `new / old` abstraction size) its
/// refinement causes.
fn min_growth_candidates(
    flaws: &[Flaw],
    conditions: &NumericConditions,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
) -> Vec<(FlawCandidate, (usize, usize))> {
    let mut candidates: Vec<(FlawCandidate, (usize, usize))> = Vec::with_capacity(flaws.len());
    for (idx, flaw) in flaws.iter().enumerate() {
        let mut restricted_dep = None;
        let growth = match flaw {
            Flaw::Numeric(nf) => {
                growth_key_for_domain_size(numeric_domain_sizes[nf.numeric_var_id.index()])
            }
            Flaw::Propositional(pf) => {
                let var_id = pf.fact.var_index();
                let prop_size = domain_sizes[var_id.index()];
                let prop_growth = if conditions.is_condition_var(var_id) && prop_size >= 2 {
                    (1, 1)
                } else {
                    growth_key_for_domain_size(prop_size)
                };
                if conditions.is_condition_var(var_id)
                    && !pf.dependent_numeric_flaws.is_empty()
                    && let Some((deps, dep_growth)) = best_min_growth_dependent_flaws(
                        &pf.dependent_numeric_flaws,
                        numeric_domain_sizes,
                    )
                {
                    restricted_dep = Some(deps);
                    (
                        prop_growth.0.saturating_mul(dep_growth.0),
                        prop_growth.1.saturating_mul(dep_growth.1),
                    )
                } else {
                    prop_growth
                }
            }
        };
        candidates.push((
            FlawCandidate {
                idx,
                restricted_dep,
            },
            growth,
        ));
    }
    candidates
}

pub(super) fn fix_single_flaw_min_growth(
    flaws: &[Flaw],
    conditions: &NumericConditions,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    rng: &mut SmallRng,
) -> ChosenFlaws {
    let mut candidates =
        min_growth_candidates(flaws, conditions, domain_sizes, numeric_domain_sizes);
    candidates.shuffle(rng);
    // `sort_by` is stable, so equal-growth candidates retain their seeded
    // shuffled order. Breaking ties by the original flaw index here would
    // silently undo the shuffle and make every collection member choose the
    // same variable.
    candidates.sort_by(|(_, left_growth), (_, right_growth)| {
        compare_growth_keys(*left_growth, *right_growth)
    });
    candidates
        .into_iter()
        .map(|(candidate, _)| candidate)
        .collect()
}
