use std::collections::BTreeMap;
use std::{collections::HashSet, fmt};

use planners_sas::numeric::numeric_task::AbstractNumericTask;
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
            nf.value.to_bits(),
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
}

impl fmt::Display for FlawTreatmentVariants {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RandomSingleAtom => write!(f, "random_single_atom"),
            Self::OneSplitPerAtom => write!(f, "one_split_per_atom"),
            Self::OneSplitPerVariable => write!(f, "one_split_per_variable"),
            Self::MaxRefinedSingleAtom => write!(f, "max_refined_single_atom"),
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
            ),
        }
    }

    fn refine_all(&self) -> bool {
        match self {
            FlawTreatmentVariants::RandomSingleAtom => false,
            FlawTreatmentVariants::OneSplitPerAtom => true,
            FlawTreatmentVariants::OneSplitPerVariable => true,
            FlawTreatmentVariants::MaxRefinedSingleAtom => false,
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

pub(super) fn fix_single_flaw_max_refined(
    flaws: &[Flaw],
    comparison_var_ids: &HashSet<usize>,
    domain_sizes: &mut [usize],
    numeric_domain_sizes: &mut [usize],
) -> ChosenFlaws {
    if flaws.is_empty() {
        return vec![];
    }

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
                let base: usize = domain_sizes.get(var_id).copied().unwrap_or(0);
                if comparison_var_ids.contains(&var_id) && !pf.dependent_numeric_flaws.is_empty() {
                    let mut best: BTreeMap<usize, Vec<NumericFlaw>> = BTreeMap::new();
                    for nf in pf.dependent_numeric_flaws.iter().cloned() {
                        let partitions = numeric_domain_sizes
                            .get(nf.numeric_var_id)
                            .copied()
                            .unwrap_or(0);
                        best.entry(partitions).or_default().push(nf);
                    }
                    if let Some((&max_partitions, vec)) = best.iter().next_back() {
                        restricted_dep = Some(vec.clone());
                        base + (max_partitions)
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
    }

    // Highest score first; tie-break by stable atom key for determinism.
    candidates.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| flaw_atom_key(&flaws[a.idx]).cmp(&flaw_atom_key(&flaws[b.idx])))
    });

    candidates
}
