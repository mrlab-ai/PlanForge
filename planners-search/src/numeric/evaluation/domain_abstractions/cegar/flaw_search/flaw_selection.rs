use std::collections::BTreeMap;
use std::{collections::HashSet, fmt};

use planners_sas::numeric::numeric_task::AbstractNumericTask;
use planners_sas::numeric::utils::float_tolerance;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use super::{Flaw, NumericFlaw};
use crate::numeric::evaluation::domain_abstractions::abstract_operator_generator::DomainMapping;
use crate::numeric::evaluation::domain_abstractions::cegar::{
    CegarConfig, ChosenFlaws, FlawCandidate,
};
use crate::numeric::evaluation::domain_abstractions::domain_abstraction::NumericPartitions;

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

/// Trait that all flaw treatment variants must implement.
pub trait FlawTreatment {
    /// Return the ordered chosen flaws.
    #[allow(clippy::too_many_arguments)]
    fn choose_flaws(
        &self,
        task: &dyn AbstractNumericTask,
        flaws: &[Flaw],
        config: &CegarConfig,
        comparison_var_ids: &HashSet<usize>,
        rng: &mut SmallRng,
        blacklisted_prop_var_ids: &mut HashSet<usize>,
        blacklisted_numeric_var_ids: &mut HashSet<usize>,
        domain_mapping: &mut DomainMapping,
        domain_sizes: &mut [usize],
        partitions: &mut NumericPartitions,
        numeric_domain_sizes: &mut [usize],
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

fn flaw_atom_key(flaw: &Flaw) -> (u8, usize, usize, u64, bool) {
    match flaw {
        Flaw::Propositional(pf) => (0, pf.fact.var, pf.fact.value, 0, false),
        Flaw::Numeric(nf) => (
            1,
            nf.numeric_var_id,
            0,
            float_tolerance::canonical_bits(nf.value),
            nf.include_in_lower,
        ),
    }
}

fn flaw_variable_key(flaw: &Flaw) -> (u8, usize) {
    match flaw {
        Flaw::Propositional(pf) => (0, pf.fact.var),
        Flaw::Numeric(nf) => (1, nf.numeric_var_id),
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

impl FlawTreatment for FlawTreatmentVariants {
    #[allow(clippy::too_many_arguments)]
    fn choose_flaws(
        &self,
        _task: &dyn AbstractNumericTask,
        flaws: &[Flaw],
        _config: &CegarConfig,
        comparison_var_ids: &HashSet<usize>,
        rng: &mut SmallRng,
        _blacklisted_prop_var_ids: &mut HashSet<usize>,
        _blacklisted_numeric_var_ids: &mut HashSet<usize>,
        _domain_mapping: &mut DomainMapping,
        domain_sizes: &mut [usize],
        _partitions: &mut NumericPartitions,
        numeric_domain_sizes: &mut [usize],
        plan_length: usize,
    ) -> ChosenFlaws {
        match self {
            FlawTreatmentVariants::RandomSingleAtom => choose_single_random_flaw(flaws, rng),
            FlawTreatmentVariants::OneSplitPerAtom => choose_flaws_per_atom(flaws),
            FlawTreatmentVariants::OneSplitPerVariable => fix_flaws_per_variable(flaws),
            FlawTreatmentVariants::MaxRefinedSingleAtom => fix_single_flaw_max_refined(
                flaws,
                comparison_var_ids,
                domain_sizes,
                numeric_domain_sizes,
                1,
                rng,
            ),
            FlawTreatmentVariants::MinGrowthSingleAtom => fix_single_flaw_min_growth(
                flaws,
                comparison_var_ids,
                domain_sizes,
                numeric_domain_sizes,
                rng,
            ),
            Self::MaxRefinedPreferringProp => fix_single_flaw_max_refined(
                flaws,
                comparison_var_ids,
                domain_sizes,
                numeric_domain_sizes,
                100,
                rng,
            ),
            FlawTreatmentVariants::ClosestToGoal => fix_closest_to_goal(flaws),
            Self::BalanceMaxRefinedAndClosestToGoal => fix_balance_max_refined_closest_to_goal(
                flaws,
                comparison_var_ids,
                domain_sizes,
                numeric_domain_sizes,
                plan_length,
                1,
            ),
            Self::BalanceMaxRefinedPreferringPropAndClosestToGoal => {
                fix_balance_max_refined_closest_to_goal(
                    flaws,
                    comparison_var_ids,
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
    comparison_var_ids: &HashSet<usize>,
    domain_sizes: &mut [usize],
    numeric_domain_sizes: &mut [usize],
    prop_multiplier: usize,
) -> (ChosenFlaws, usize) {
    let mut max_score = 0;
    let mut candidates: ChosenFlaws = Vec::with_capacity(flaws.len());
    for (idx, flaw) in flaws.iter().enumerate() {
        let mut restricted_dep: Option<Vec<NumericFlaw>> = None;
        let score: usize = match flaw {
            Flaw::Numeric(nf) => numeric_domain_sizes
                .get(nf.numeric_var_id)
                .copied()
                .unwrap_or(0),
            Flaw::Propositional(pf) => {
                let var_id = pf.fact.var;
                let base: usize = domain_sizes.get(var_id).copied().unwrap_or(0) * prop_multiplier;
                if comparison_var_ids.contains(&var_id) && !pf.dependent_numeric_flaws.is_empty() {
                    let mut by_partition_count: BTreeMap<usize, Vec<NumericFlaw>> = BTreeMap::new();
                    for nf in pf.dependent_numeric_flaws.iter().cloned() {
                        let partitions = numeric_domain_sizes
                            .get(nf.numeric_var_id)
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
    comparison_var_ids: &HashSet<usize>,
    domain_sizes: &mut [usize],
    numeric_domain_sizes: &mut [usize],
    prop_multiplier: usize,
    rng: &mut SmallRng,
) -> ChosenFlaws {
    if flaws.is_empty() {
        return vec![];
    }

    let (mut candidates, _max_score) = compute_max_refined(
        flaws,
        comparison_var_ids,
        domain_sizes,
        numeric_domain_sizes,
        prop_multiplier,
    );
    // Match numeric-FD: highest score first, random order within an equal-score tier.
    candidates.sort_by(|a, b| b.score.cmp(&a.score));
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
            .get(nf.numeric_var_id)
            .copied()
            .unwrap_or(1)
            .max(1);
        by_partition_count.entry(partitions).or_default().push(nf);
    }
    let (&max_partitions, flaws) = by_partition_count.iter().next_back()?;
    Some((flaws.clone(), growth_key_for_domain_size(max_partitions)))
}

pub(super) fn fix_single_flaw_min_growth(
    flaws: &[Flaw],
    comparison_var_ids: &HashSet<usize>,
    domain_sizes: &mut [usize],
    numeric_domain_sizes: &mut [usize],
    rng: &mut SmallRng,
) -> ChosenFlaws {
    let mut candidates: Vec<(FlawCandidate, (usize, usize))> = Vec::with_capacity(flaws.len());
    for (idx, flaw) in flaws.iter().enumerate() {
        let mut restricted_dep = None;
        let growth = match flaw {
            Flaw::Numeric(nf) => growth_key_for_domain_size(
                numeric_domain_sizes
                    .get(nf.numeric_var_id)
                    .copied()
                    .unwrap_or(1),
            ),
            Flaw::Propositional(pf) => {
                let var_id = pf.fact.var;
                let prop_size = domain_sizes.get(var_id).copied().unwrap_or(1).max(1);
                let prop_growth = if comparison_var_ids.contains(&var_id) && prop_size >= 2 {
                    (1, 1)
                } else {
                    growth_key_for_domain_size(prop_size)
                };
                if comparison_var_ids.contains(&var_id)
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
    candidates.shuffle(rng);
    candidates.sort_by(|(left, left_growth), (right, right_growth)| {
        compare_growth_keys(*left_growth, *right_growth).then_with(|| left.idx.cmp(&right.idx))
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
    // `b.cmp` is used instead of `a.cmp` to order them by step at reverse order.
    candidates.sort_unstable_by(|a, b| b.score.cmp(&a.score));

    candidates
}

pub(super) fn fix_balance_max_refined_closest_to_goal(
    flaws: &[Flaw],
    comparison_var_ids: &HashSet<usize>,
    domain_sizes: &mut [usize],
    numeric_domain_sizes: &mut [usize],
    plan_length: usize,
    prop_multiplier: usize,
) -> ChosenFlaws {
    let (mut candidates, max_score) = compute_max_refined(
        flaws,
        comparison_var_ids,
        domain_sizes,
        numeric_domain_sizes,
        prop_multiplier,
    );
    let max_score = max_score as f64;
    let max_length = if plan_length > 0 {
        plan_length as f64
    } else {
        1.0
    };
    candidates.sort_unstable_by(|a, b| {
        (b.score as f64 / max_score - flaws[b.idx].step() as f64 / max_length)
            .partial_cmp(&(a.score as f64 / max_score - flaws[a.idx].step() as f64 / max_length))
            .unwrap()
    });

    candidates
}

#[cfg(test)]
mod tests {
    use planners_sas::numeric::numeric_task::ExplicitFact;
    use rand::SeedableRng;

    use super::*;
    use crate::numeric::evaluation::domain_abstractions::cegar::flaw_search::PropFlaw;

    fn prop_flaw(var: usize, deps: Vec<NumericFlaw>) -> Flaw {
        Flaw::Propositional(PropFlaw {
            fact: ExplicitFact::new(var, 0),
            dependent_numeric_flaws: deps,
            step: 0,
        })
    }

    fn numeric_flaw(var: usize) -> NumericFlaw {
        NumericFlaw {
            numeric_var_id: var,
            value: 0.0,
            include_in_lower: true,
            step: 0,
        }
    }

    #[test]
    fn max_refined_scores_comparison_flaws_like_numeric_fd() {
        let flaws = vec![
            prop_flaw(1, Vec::new()),
            prop_flaw(0, vec![numeric_flaw(0)]),
        ];
        let comparison_var_ids = HashSet::from([0]);
        let mut domain_sizes = vec![1, 2];
        let mut numeric_domain_sizes = vec![1];
        let mut rng = SmallRng::seed_from_u64(1);

        let chosen = fix_single_flaw_max_refined(
            &flaws,
            &comparison_var_ids,
            &mut domain_sizes,
            &mut numeric_domain_sizes,
            1,
            &mut rng,
        );

        assert_eq!(chosen[0].idx, 0);
    }

    #[test]
    fn max_refined_continues_most_refined_dependent_numeric_view() {
        let flaws = vec![prop_flaw(0, vec![numeric_flaw(0), numeric_flaw(1)])];
        let comparison_var_ids = HashSet::from([0]);
        let mut domain_sizes = vec![2];
        let mut numeric_domain_sizes = vec![7, 2];

        let (chosen, _) = compute_max_refined(
            &flaws,
            &comparison_var_ids,
            &mut domain_sizes,
            &mut numeric_domain_sizes,
            1,
        );

        let restricted = chosen[0]
            .restricted_dep
            .as_ref()
            .expect("comparison flaw should restrict dependent numeric flaws");
        assert_eq!(restricted, &vec![numeric_flaw(0)]);
    }

    #[test]
    fn min_growth_continues_most_refined_dependent_numeric_view() {
        let flaws = vec![prop_flaw(0, vec![numeric_flaw(0), numeric_flaw(1)])];
        let comparison_var_ids = HashSet::from([0]);
        let mut domain_sizes = vec![2];
        let mut numeric_domain_sizes = vec![7, 2];
        let mut rng = SmallRng::seed_from_u64(1);

        let chosen = fix_single_flaw_min_growth(
            &flaws,
            &comparison_var_ids,
            &mut domain_sizes,
            &mut numeric_domain_sizes,
            &mut rng,
        );

        let restricted = chosen[0]
            .restricted_dep
            .as_ref()
            .expect("comparison flaw should restrict dependent numeric flaws");
        assert_eq!(restricted, &vec![numeric_flaw(0)]);
    }
}
