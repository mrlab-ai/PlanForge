use std::collections::BTreeMap;
use std::{collections::HashSet, fmt};

use anyhow::Result;
use planners_sas::numeric::numeric_task::AbstractNumericTask;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

use super::{Flaw, NumericFlaw, flaw_atom_key, flaw_variable_key};
use crate::numeric::evaluation::domain_abstractions::abstract_operator_generator::DomainMapping;
use crate::numeric::evaluation::domain_abstractions::cegar::{
    CegarConfig, DependentNumericRefinement, compute_abstraction_size, shuffle_indices_with_rng,
    try_refine_from_flaw,
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

/// How `fix_flaws` chooses which flaws to refine.
///
/// This mirrors numeric-fd's `FlawTreatment` options, but our defaults aim to
/// stay deterministic.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FlawTreatment {
    RandomSingleAtom,
    OneSplitPerAtom,
    OneSplitPerVariable,
    MaxRefinedSingleAtom,
}

impl fmt::Display for FlawTreatment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RandomSingleAtom => write!(f, "random_single_atom"),
            Self::OneSplitPerAtom => write!(f, "one_split_per_atom"),
            Self::OneSplitPerVariable => write!(f, "one_split_per_variable"),
            Self::MaxRefinedSingleAtom => write!(f, "max_refined_single_atom"),
        }
    }
}

impl FlawTreatment {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fix(
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
    ) -> Result<bool> {
        match self {
            FlawTreatment::RandomSingleAtom => fix_single_random_flaw(
                task,
                flaws,
                config,
                comparison_var_ids,
                rng,
                blacklisted_prop_var_ids,
                blacklisted_numeric_var_ids,
                domain_mapping,
                domain_sizes,
                partitions,
                numeric_domain_sizes,
            ),
            FlawTreatment::OneSplitPerAtom => fix_flaws_per_atom(
                task,
                flaws,
                config,
                comparison_var_ids,
                blacklisted_prop_var_ids,
                blacklisted_numeric_var_ids,
                domain_mapping,
                domain_sizes,
                partitions,
                numeric_domain_sizes,
            ),
            FlawTreatment::OneSplitPerVariable => fix_flaws_per_variable(
                task,
                flaws,
                config,
                comparison_var_ids,
                blacklisted_prop_var_ids,
                blacklisted_numeric_var_ids,
                domain_mapping,
                domain_sizes,
                partitions,
                numeric_domain_sizes,
            ),
            FlawTreatment::MaxRefinedSingleAtom => fix_single_flaw_max_refined(
                task,
                flaws,
                config,
                comparison_var_ids,
                blacklisted_prop_var_ids,
                blacklisted_numeric_var_ids,
                domain_mapping,
                domain_sizes,
                partitions,
                numeric_domain_sizes,
                compute_abstraction_size(domain_sizes, numeric_domain_sizes),
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fix_single_random_flaw(
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
) -> Result<bool> {
    if flaws.is_empty() {
        return Ok(false);
    }

    let mut indices: Vec<usize> = (0..flaws.len()).collect();
    shuffle_indices_with_rng(&mut indices, rng);

    for idx in indices {
        if try_refine_from_flaw(
            task,
            &flaws[idx],
            config,
            comparison_var_ids,
            blacklisted_prop_var_ids,
            blacklisted_numeric_var_ids,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            DependentNumericRefinement::One,
        )? {
            return Ok(true);
        }
    }

    Ok(false)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fix_flaws_per_atom(
    task: &dyn AbstractNumericTask,
    flaws: &[Flaw],
    config: &CegarConfig,
    comparison_var_ids: &HashSet<usize>,
    blacklisted_prop_var_ids: &mut HashSet<usize>,
    blacklisted_numeric_var_ids: &mut HashSet<usize>,
    domain_mapping: &mut DomainMapping,
    domain_sizes: &mut [usize],
    partitions: &mut NumericPartitions,
    numeric_domain_sizes: &mut [usize],
) -> Result<bool> {
    let mut ordered: Vec<&Flaw> = flaws.iter().collect();
    ordered.sort_by_key(|a| flaw_atom_key(a));

    let mut changed = false;
    let mut last: Option<(u8, usize, usize, u64, bool)> = None;
    for flaw in ordered {
        let key = flaw_atom_key(flaw);
        if last.as_ref() == Some(&key) {
            continue;
        }
        last = Some(key);
        let local_changed = try_refine_from_flaw(
            task,
            flaw,
            config,
            comparison_var_ids,
            blacklisted_prop_var_ids,
            blacklisted_numeric_var_ids,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            DependentNumericRefinement::All,
        )?;
        changed = changed || local_changed;
    }
    Ok(changed)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fix_flaws_per_variable(
    task: &dyn AbstractNumericTask,
    flaws: &[Flaw],
    config: &CegarConfig,
    comparison_var_ids: &HashSet<usize>,
    blacklisted_prop_var_ids: &mut HashSet<usize>,
    blacklisted_numeric_var_ids: &mut HashSet<usize>,
    domain_mapping: &mut DomainMapping,
    domain_sizes: &mut [usize],
    partitions: &mut NumericPartitions,
    numeric_domain_sizes: &mut [usize],
) -> Result<bool> {
    let mut ordered: Vec<&Flaw> = flaws.iter().collect();
    ordered.sort_by_key(|a| flaw_variable_key(a));

    let mut changed = false;
    let mut last: Option<(u8, usize)> = None;

    for flaw in ordered {
        let key = flaw_variable_key(flaw);
        if last.as_ref() == Some(&key) {
            continue;
        }
        last = Some(key);
        let local_changed = try_refine_from_flaw(
            task,
            flaw,
            config,
            comparison_var_ids,
            blacklisted_prop_var_ids,
            blacklisted_numeric_var_ids,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            DependentNumericRefinement::One,
        )?;
        changed = changed || local_changed;
    }
    Ok(changed)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fix_single_flaw_max_refined(
    task: &dyn AbstractNumericTask,
    flaws: &[Flaw],
    config: &CegarConfig,
    comparison_var_ids: &HashSet<usize>,
    blacklisted_prop_var_ids: &mut HashSet<usize>,
    blacklisted_numeric_var_ids: &mut HashSet<usize>,
    domain_mapping: &mut DomainMapping,
    domain_sizes: &mut [usize],
    partitions: &mut NumericPartitions,
    numeric_domain_sizes: &mut [usize],
    abstraction_size: usize,
) -> Result<bool> {
    if flaws.is_empty() {
        return Ok(false);
    }

    #[derive(Clone)]
    struct Candidate {
        idx: usize,
        score: usize,
        restricted_dep: Option<Vec<NumericFlaw>>,
    }

    let mut candidates: Vec<Candidate> = Vec::with_capacity(flaws.len());
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
        candidates.push(Candidate {
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

    for cand in candidates {
        let mut chosen = flaws[cand.idx].clone();
        if let (Flaw::Propositional(pf), Some(restricted)) = (&mut chosen, cand.restricted_dep) {
            pf.dependent_numeric_flaws = restricted;
        }

        if try_refine_from_flaw(
            task,
            &chosen,
            config,
            comparison_var_ids,
            blacklisted_prop_var_ids,
            blacklisted_numeric_var_ids,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            DependentNumericRefinement::One,
        )? {
            return Ok(true);
        }
    }

    let _ = abstraction_size;
    Ok(false)
}
