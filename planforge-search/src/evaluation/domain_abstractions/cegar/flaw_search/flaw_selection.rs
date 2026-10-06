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

use super::{Flaw, NumericFlaw};
use crate::evaluation::domain_abstractions::cegar::{ChosenFlaws, FlawCandidate};

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

/// Trait that all flaw treatment variants must implement.
pub trait FlawTreatment {
    /// Return the ordered chosen flaws. The two domain-size vectors are what
    /// the scoring criteria read: how far each variable has already been
    /// refined, and therefore what a further split would cost.
    ///
    /// `tie_break` decides between equally good flaws; see [`TieBreak`].
    fn choose_flaws(
        &self,
        task: &dyn AbstractNumericTask,
        flaws: &[Flaw],
        domain_sizes: &[usize],
        numeric_domain_sizes: &[usize],
        tie_break: TieBreak<'_>,
        plan_length: usize,
    ) -> ChosenFlaws;

    /// Specify if all flaws should be refined or only one instead.
    fn refine_all(&self) -> bool;

    /// Function that specifies if a flaw should be refined based on another one.
    /// Note that this function is called with the last refined flaw as second
    /// parameter, and so `choose_flaws` must sort flaws by the criterion used
    /// to discriminate whether it should be refined to avoid refining multiple
    /// flaws for which this function return `false`.
    fn should_be_refined(&self, flaw: &Flaw, last_refined: &Flaw) -> bool;
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

fn flaw_variable_key(flaw: &Flaw) -> (u8, usize) {
    match flaw {
        Flaw::Propositional(pf) => (0, pf.fact.var()),
        Flaw::Numeric(nf) => (1, nf.numeric_var_id.index()),
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
    OneSplitPerAtom,
    OneSplitPerVariable,
    MaxRefinedSingleAtom,
    MinGrowthSingleAtom,
    MaxRefinedPreferringProp,
    ClosestToGoal,
    BalanceMaxRefinedAndClosestToGoal,
    BalanceMaxRefinedPreferringPropAndClosestToGoal,
}

impl fmt::Display for FlawTreatmentVariants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RandomSingleAtom => write!(f, "random_single_atom"),
            Self::OneSplitPerAtom => write!(f, "one_split_per_atom"),
            Self::OneSplitPerVariable => write!(f, "one_split_per_variable"),
            Self::MaxRefinedSingleAtom => write!(f, "max_refined_single_atom"),
            Self::MinGrowthSingleAtom => write!(f, "min_growth_single_atom"),
            Self::MaxRefinedPreferringProp => write!(f, "max_refined_preferring_prop"),
            Self::ClosestToGoal => write!(f, "closest_to_goal"),
            Self::BalanceMaxRefinedAndClosestToGoal => {
                write!(f, "balance_max_refined_and_closest_to_goal")
            }
            Self::BalanceMaxRefinedPreferringPropAndClosestToGoal => {
                write!(f, "balance_max_refined_preferring_prop_and_closest_to_goal")
            }
        }
    }
}

impl crate::config::FromOptionValue for FlawTreatmentVariants {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "random_single_atom" => Ok(Self::RandomSingleAtom),
            "one_split_per_atom" => Ok(Self::OneSplitPerAtom),
            "one_split_per_variable" => Ok(Self::OneSplitPerVariable),
            "max_refined_single_atom" => Ok(Self::MaxRefinedSingleAtom),
            "min_growth_single_atom" => Ok(Self::MinGrowthSingleAtom),
            "max_refined_preferring_prop" => Ok(Self::MaxRefinedPreferringProp),
            "closest_to_goal" => Ok(Self::ClosestToGoal),
            "balance_max_refined_and_closest_to_goal" => {
                Ok(Self::BalanceMaxRefinedAndClosestToGoal)
            }
            "balance_max_refined_preferring_prop_and_closest_to_goal" => {
                Ok(Self::BalanceMaxRefinedPreferringPropAndClosestToGoal)
            }
            other => Err(format!("invalid FlawTreatment `{other}`")),
        }
    }
}

impl FlawTreatment for FlawTreatmentVariants {
    fn choose_flaws(
        &self,
        task: &dyn AbstractNumericTask,
        flaws: &[Flaw],
        domain_sizes: &[usize],
        numeric_domain_sizes: &[usize],
        tie_break: TieBreak<'_>,
        plan_length: usize,
    ) -> ChosenFlaws {
        let conditions = task.numeric_conditions();
        let rng = match tie_break {
            TieBreak::Rank(rank) => {
                return self.choose_ranked_flaws(
                    flaws,
                    conditions,
                    domain_sizes,
                    numeric_domain_sizes,
                    plan_length,
                    rank,
                );
            }
            TieBreak::Random(rng) => rng,
        };
        match self {
            FlawTreatmentVariants::RandomSingleAtom => choose_single_random_flaw(flaws, rng),
            FlawTreatmentVariants::OneSplitPerAtom => choose_flaws_per_atom(flaws),
            FlawTreatmentVariants::OneSplitPerVariable => fix_flaws_per_variable(flaws),
            FlawTreatmentVariants::MaxRefinedSingleAtom => fix_single_flaw_max_refined(
                flaws,
                conditions,
                domain_sizes,
                numeric_domain_sizes,
                1,
                rng,
            ),
            FlawTreatmentVariants::MinGrowthSingleAtom => fix_single_flaw_min_growth(
                flaws,
                conditions,
                domain_sizes,
                numeric_domain_sizes,
                rng,
            ),
            Self::MaxRefinedPreferringProp => fix_single_flaw_max_refined(
                flaws,
                conditions,
                domain_sizes,
                numeric_domain_sizes,
                100,
                rng,
            ),
            FlawTreatmentVariants::ClosestToGoal => fix_closest_to_goal(flaws),
            Self::BalanceMaxRefinedAndClosestToGoal => fix_balance_max_refined_closest_to_goal(
                flaws,
                conditions,
                domain_sizes,
                numeric_domain_sizes,
                plan_length,
                1,
            ),
            Self::BalanceMaxRefinedPreferringPropAndClosestToGoal => {
                fix_balance_max_refined_closest_to_goal(
                    flaws,
                    conditions,
                    domain_sizes,
                    numeric_domain_sizes,
                    plan_length,
                    100,
                )
            }
        }
    }

    fn refine_all(&self) -> bool {
        match self {
            FlawTreatmentVariants::RandomSingleAtom => false,
            FlawTreatmentVariants::OneSplitPerAtom => true,
            FlawTreatmentVariants::OneSplitPerVariable => true,
            FlawTreatmentVariants::MaxRefinedSingleAtom => false,
            FlawTreatmentVariants::MinGrowthSingleAtom => false,
            FlawTreatmentVariants::MaxRefinedPreferringProp => false,
            FlawTreatmentVariants::ClosestToGoal => false,
            &FlawTreatmentVariants::BalanceMaxRefinedAndClosestToGoal => false,
            &FlawTreatmentVariants::BalanceMaxRefinedPreferringPropAndClosestToGoal => false,
        }
    }

    /// Function that specifies if a flaw should be refined based on another one.
    /// Note that this function is called with the last refined flaw as second
    /// parameter, and so `choose_flaws` must sort flaws by the criterion used
    /// to discriminate whether it should be refined to avoid refining multiple
    /// flaws for which this function return `false`.
    fn should_be_refined(&self, flaw: &Flaw, last_refined: &Flaw) -> bool {
        match self {
            FlawTreatmentVariants::RandomSingleAtom => false,
            FlawTreatmentVariants::OneSplitPerAtom => {
                flaw_atom_key(flaw) == flaw_atom_key(last_refined)
            }
            FlawTreatmentVariants::OneSplitPerVariable => {
                flaw_variable_key(flaw) == flaw_variable_key(last_refined)
            }
            FlawTreatmentVariants::MaxRefinedSingleAtom => false,
            FlawTreatmentVariants::MinGrowthSingleAtom => false,
            FlawTreatmentVariants::MaxRefinedPreferringProp => false,
            FlawTreatmentVariants::ClosestToGoal => false,
            FlawTreatmentVariants::BalanceMaxRefinedAndClosestToGoal => false,
            &FlawTreatmentVariants::BalanceMaxRefinedPreferringPropAndClosestToGoal => false,
        }
    }
}

impl FlawTreatmentVariants {
    /// [`FlawTreatment::choose_flaws`] with [`TieBreak::Rank`]: the
    /// treatment's preference order, ties broken by [`flaw_tie_key`], and the
    /// best tier rotated so the flaw at `rank % ties` comes first.
    fn choose_ranked_flaws(
        &self,
        flaws: &[Flaw],
        conditions: &NumericConditions,
        domain_sizes: &[usize],
        numeric_domain_sizes: &[usize],
        plan_length: usize,
        rank: usize,
    ) -> ChosenFlaws {
        let max_refined = |prop_multiplier| {
            compute_max_refined(
                flaws,
                conditions,
                domain_sizes,
                numeric_domain_sizes,
                prop_multiplier,
            )
        };
        let higher_first = |left: &usize, right: &usize| right.cmp(left);
        match self {
            Self::RandomSingleAtom => rank_tied_best(
                flaws,
                unscored_candidates(flaws)
                    .map(|candidate| (candidate, ()))
                    .collect(),
                rank,
                |_, _| Ordering::Equal,
            ),
            Self::MaxRefinedSingleAtom | Self::MaxRefinedPreferringProp => {
                let prop_multiplier = if *self == Self::MaxRefinedSingleAtom {
                    1
                } else {
                    100
                };
                let (candidates, _) = max_refined(prop_multiplier);
                rank_tied_best(
                    flaws,
                    candidates
                        .into_iter()
                        .map(|candidate| {
                            let score = candidate.score;
                            (candidate, score)
                        })
                        .collect(),
                    rank,
                    higher_first,
                )
            }
            Self::MinGrowthSingleAtom => rank_tied_best(
                flaws,
                min_growth_candidates(flaws, conditions, domain_sizes, numeric_domain_sizes),
                rank,
                |left, right| compare_growth_keys(*left, *right),
            ),
            Self::ClosestToGoal => rank_tied_best(
                flaws,
                unscored_candidates(flaws)
                    .map(|candidate| {
                        let step = flaws[candidate.idx].step();
                        (candidate, step)
                    })
                    .collect(),
                rank,
                higher_first,
            ),
            Self::BalanceMaxRefinedAndClosestToGoal
            | Self::BalanceMaxRefinedPreferringPropAndClosestToGoal => {
                let prop_multiplier = if *self == Self::BalanceMaxRefinedAndClosestToGoal {
                    1
                } else {
                    100
                };
                let (candidates, max_score) = max_refined(prop_multiplier);
                rank_tied_best(
                    flaws,
                    candidates
                        .into_iter()
                        .map(|candidate| {
                            let balance = balance_score(flaws, &candidate, max_score, plan_length);
                            (candidate, balance)
                        })
                        .collect(),
                    rank,
                    |left: &f64, right: &f64| right.total_cmp(left),
                )
            }
            Self::OneSplitPerAtom => choose_flaws_per_atom(flaws),
            Self::OneSplitPerVariable => fix_flaws_per_variable(flaws),
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
        score: 0,
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
            score: 0,
            restricted_dep: None,
        })
        .collect();
    candidates.shuffle(rng);

    candidates
}

pub(super) fn choose_flaws_per_atom(flaws: &[Flaw]) -> ChosenFlaws {
    let mut candidates: ChosenFlaws = (0..flaws.len())
        .map(|i| FlawCandidate {
            idx: i,
            score: 0,
            restricted_dep: None,
        })
        .collect();
    candidates.sort_by_key(|c| flaw_atom_key(&flaws[c.idx]));

    candidates
}

pub(super) fn fix_flaws_per_variable(flaws: &[Flaw]) -> ChosenFlaws {
    let mut candidates: ChosenFlaws = (0..flaws.len())
        .map(|i| FlawCandidate {
            idx: i,
            score: 0,
            restricted_dep: None,
        })
        .collect();
    candidates.sort_by_key(|c| flaw_variable_key(&flaws[c.idx]));

    candidates
}

fn compute_max_refined(
    flaws: &[Flaw],
    conditions: &NumericConditions,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    prop_multiplier: usize,
) -> (ChosenFlaws, usize) {
    let mut max_score = 0;
    let mut candidates: ChosenFlaws = Vec::with_capacity(flaws.len());
    for (idx, flaw) in flaws.iter().enumerate() {
        let mut restricted_dep: Option<Vec<NumericFlaw>> = None;
        let score: usize = match flaw {
            Flaw::Numeric(nf) => numeric_domain_sizes
                .get(nf.numeric_var_id.index())
                .copied()
                .unwrap_or(0),
            Flaw::Propositional(pf) => {
                let var_id = pf.fact.var_index();
                let base: usize =
                    domain_sizes.get(var_id.index()).copied().unwrap_or(0) * prop_multiplier;
                if conditions.is_condition_var(var_id) && !pf.dependent_numeric_flaws.is_empty() {
                    let mut by_partition_count: BTreeMap<usize, Vec<NumericFlaw>> = BTreeMap::new();
                    for nf in pf.dependent_numeric_flaws.iter().cloned() {
                        let partitions = numeric_domain_sizes
                            .get(nf.numeric_var_id.index())
                            .copied()
                            .unwrap_or(0);
                        by_partition_count.entry(partitions).or_default().push(nf);
                    }
                    if let Some((&max_partitions, vec)) = by_partition_count.iter().next_back() {
                        restricted_dep = Some(vec.clone());
                        base.saturating_add(max_partitions)
                    } else {
                        base
                    }
                } else {
                    base
                }
            }
        };
        candidates.push(FlawCandidate {
            idx,
            score,
            restricted_dep,
        });
        if score > max_score {
            max_score = score;
        }
    }

    (candidates, max_score)
}

pub(super) fn fix_single_flaw_max_refined(
    flaws: &[Flaw],
    conditions: &NumericConditions,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    prop_multiplier: usize,
    rng: &mut SmallRng,
) -> ChosenFlaws {
    if flaws.is_empty() {
        return vec![];
    }

    let (mut candidates, _max_score) = compute_max_refined(
        flaws,
        conditions,
        domain_sizes,
        numeric_domain_sizes,
        prop_multiplier,
    );
    // Match numeric-FD: highest score first, random order within an equal-score
    // tier. `sort_by_key` rather than `sort_unstable_by_key` because the tier
    // shuffle below relies on equal-score candidates keeping their input order.
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.score));
    let mut tier_start = 0;
    while tier_start < candidates.len() {
        let score = candidates[tier_start].score;
        let mut tier_end = tier_start + 1;
        while tier_end < candidates.len() && candidates[tier_end].score == score {
            tier_end += 1;
        }
        candidates[tier_start..tier_end].shuffle(rng);
        tier_start = tier_end;
    }

    candidates
}

fn growth_key_for_domain_size(current_size: usize) -> (usize, usize) {
    let current_size = current_size.max(1);
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
        let partitions = numeric_domain_sizes
            .get(nf.numeric_var_id.index())
            .copied()
            .unwrap_or(1)
            .max(1);
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
            Flaw::Numeric(nf) => growth_key_for_domain_size(
                numeric_domain_sizes
                    .get(nf.numeric_var_id.index())
                    .copied()
                    .unwrap_or(1),
            ),
            Flaw::Propositional(pf) => {
                let var_id = pf.fact.var_index();
                let prop_size = domain_sizes
                    .get(var_id.index())
                    .copied()
                    .unwrap_or(1)
                    .max(1);
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
                score: 0,
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

pub(super) fn fix_closest_to_goal(flaws: &[Flaw]) -> ChosenFlaws {
    let mut candidates: Vec<FlawCandidate> = (0..flaws.len())
        .map(|i| FlawCandidate {
            idx: i,
            score: flaws[i].step(),
            restricted_dep: None,
        })
        .collect();
    // `Reverse` orders them by step descending.
    candidates.sort_unstable_by_key(|candidate| std::cmp::Reverse(candidate.score));

    candidates
}

pub(super) fn fix_balance_max_refined_closest_to_goal(
    flaws: &[Flaw],
    conditions: &NumericConditions,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    plan_length: usize,
    prop_multiplier: usize,
) -> ChosenFlaws {
    let (mut candidates, max_score) = compute_max_refined(
        flaws,
        conditions,
        domain_sizes,
        numeric_domain_sizes,
        prop_multiplier,
    );
    candidates.sort_unstable_by(|a, b| {
        balance_score(flaws, b, max_score, plan_length)
            .partial_cmp(&balance_score(flaws, a, max_score, plan_length))
            .unwrap()
    });

    candidates
}

/// How refined a flaw's variable already is, relative to the most refined one,
/// minus how far along the plan the flaw sits.
fn balance_score(
    flaws: &[Flaw],
    candidate: &FlawCandidate,
    max_score: usize,
    plan_length: usize,
) -> f64 {
    let max_length = if plan_length > 0 {
        plan_length as f64
    } else {
        1.0
    };
    candidate.score as f64 / max_score as f64 - flaws[candidate.idx].step() as f64 / max_length
}
