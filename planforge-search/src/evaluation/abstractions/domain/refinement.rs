//! Refinement of a domain abstraction: which flaws can still be refined and
//! the splits that refine them.

use std::collections::HashSet;

use anyhow::Result;
use planforge_sas::numeric_conditions::{ConditionValue, NumericConditions};
use planforge_sas::numeric_task::{AbstractNumericTask, ExplicitValueIndex, VariableIndex};
use rand::rngs::SmallRng;
use tracing::debug;

use super::abstract_operators::DomainMapping;
use super::abstraction::compute_abstraction_size_u128;
use super::config::CegarConfig;
use super::flaws::{Flaw, NumericFlaw, PropFlaw, can_split_numeric_var};
use super::partitions::NumericPartitions;
use super::split_selection::{ChosenFlaws, TieBreak};

/// The only abstract domain a refined condition variable ever has: the value
/// that establishes the condition in its own class, the value that does not in
/// the other. Indexed by [`ConditionValue`], so `True` maps to class 1 and
/// `False` to class 0 — the class an abstract operator's condition effect
/// targets and the one
/// [`DomainAbstractionFactory`](super::factory::DomainAbstractionFactory) clears a comparison digit
/// to before deriving its verdict.
const CONDITION_SPLIT_MAPPING: [ExplicitValueIndex; ConditionValue::DOMAIN_SIZE] =
    [ExplicitValueIndex::new(1), ExplicitValueIndex::new(0)];

/// What a refinement writes to: the abstraction's propositional domain mapping
/// with its per-variable domain sizes, its numeric partitions with theirs, and
/// the variables refinement has given up on.
///
/// Every landed split adds one value to one of the two halves and bumps the
/// matching size, which is how the CEGAR loop detects progress. The initial-split
/// functions only read the two blacklists; `fix_flaws` lets flaw selection extend
/// them.
pub struct RefinementState<'a> {
    pub domain_mapping: &'a mut DomainMapping,
    pub domain_sizes: &'a mut [usize],
    pub partitions: &'a mut NumericPartitions,
    pub numeric_domain_sizes: &'a mut [usize],
    pub blacklisted_prop_var_ids: &'a mut HashSet<VariableIndex>,
    pub blacklisted_numeric_var_ids: &'a mut HashSet<VariableIndex>,
}

impl RefinementState<'_> {
    /// Shorten the borrows so an inner call can rewrite the same abstraction.
    fn reborrow(&mut self) -> RefinementState<'_> {
        RefinementState {
            domain_mapping: &mut *self.domain_mapping,
            domain_sizes: &mut *self.domain_sizes,
            partitions: &mut *self.partitions,
            numeric_domain_sizes: &mut *self.numeric_domain_sizes,
            blacklisted_prop_var_ids: &mut *self.blacklisted_prop_var_ids,
            blacklisted_numeric_var_ids: &mut *self.blacklisted_numeric_var_ids,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RefinementSummary {
    pub refined_propositional_vars: HashSet<VariableIndex>,
    pub refined_numeric_vars: HashSet<VariableIndex>,
}

impl RefinementSummary {
    pub fn is_empty(&self) -> bool {
        self.refined_propositional_vars.is_empty() && self.refined_numeric_vars.is_empty()
    }

    fn mark_propositional(&mut self, var_id: VariableIndex) {
        self.refined_propositional_vars.insert(var_id);
    }

    fn mark_numeric(&mut self, var_id: VariableIndex) {
        self.refined_numeric_vars.insert(var_id);
    }
}

#[derive(Debug, Clone)]
pub(super) struct EligibleFlaws {
    pub(super) flaws: Vec<Flaw>,
    pub(super) filtered_stale: usize,
}

pub(super) fn filter_eligible_flaws(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    domain_sizes: &[usize],
    flaws: &[Flaw],
) -> EligibleFlaws {
    let mut eligible = Vec::with_capacity(flaws.len());
    let mut filtered_stale = 0usize;
    for flaw in flaws {
        match flaw {
            Flaw::Numeric(numeric) => {
                if numeric_flaw_is_refinable(partitions, numeric) {
                    eligible.push(Flaw::Numeric(numeric.clone()));
                } else {
                    filtered_stale += 1;
                }
            }
            Flaw::Propositional(prop) => {
                if !propositional_flaw_is_refinable(task, partitions, domain_sizes, prop) {
                    filtered_stale += 1 + prop.dependent_numeric_flaws.len();
                    continue;
                }
                let mut filtered_prop = prop.clone();
                let old_dep_count = filtered_prop.dependent_numeric_flaws.len();
                filtered_prop
                    .dependent_numeric_flaws
                    .retain(|numeric| numeric_flaw_is_refinable(partitions, numeric));
                filtered_stale += old_dep_count - filtered_prop.dependent_numeric_flaws.len();
                eligible.push(Flaw::Propositional(filtered_prop));
            }
        }
    }
    EligibleFlaws {
        flaws: eligible,
        filtered_stale,
    }
}

fn numeric_flaw_is_refinable(partitions: &NumericPartitions, flaw: &NumericFlaw) -> bool {
    can_split_numeric_var(
        partitions,
        flaw.numeric_var_id,
        flaw.value,
        flaw.include_in_lower,
    )
}

/// Is there a refinement left that this propositional flaw asks for?
///
/// For an ordinary variable that is one question: are there values still sharing
/// an abstract class? A condition variable is different. Splitting it is a
/// one-shot — [`ConditionValue`]'s two values, told apart — but the flaw it
/// raises is rarely about the split. It is about the numeric variables the
/// comparison reads, and those come along in `dependent_numeric_flaws`, so the
/// flaw stays refinable for as long as one of them is.
fn propositional_flaw_is_refinable(
    task: &dyn AbstractNumericTask,
    partitions: &NumericPartitions,
    domain_sizes: &[usize],
    flaw: &PropFlaw,
) -> bool {
    let var_id = flaw.fact.var_index();
    let current_size = *domain_sizes
        .get(var_id.index())
        .unwrap_or_else(|| panic!("flaw variable {var_id} is outside domain_sizes"));
    let true_size = task
        .get_variable_domain_size(var_id)
        .unwrap_or_else(|error| panic!("failed to get true domain size for var {var_id}: {error}"));
    assert!(
        flaw.fact.value() < true_size,
        "flaw fact value {} outside true domain size {} for var {}",
        flaw.fact.value(),
        true_size,
        var_id
    );
    if current_size < true_size {
        return true;
    }
    task.numeric_conditions().is_condition_var(var_id)
        && flaw
            .dependent_numeric_flaws
            .iter()
            .any(|numeric| numeric_flaw_is_refinable(partitions, numeric))
}

/// Whether growing one variable's abstract domain from `old_size` to
/// `new_size` keeps the abstraction within `max_abstraction_size`.
///
/// Sizes are compared exactly in `u128`. A size that does not fit `u128` is
/// above every `usize` limit, so the refinement does not fit: that is the
/// configured size limit, and the callers blacklist the variable for it.
fn refined_size_fits(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    old_size: usize,
    new_size: usize,
    max_abstraction_size: usize,
) -> bool {
    assert!(
        old_size > 0 && new_size > 0,
        "abstract domain sizes are positive, got {old_size} -> {new_size}"
    );
    let Some(total_size) = compute_abstraction_size_u128(domain_sizes, numeric_domain_sizes) else {
        // The current abstraction is already larger than u128 and thus than
        // any limit.
        return false;
    };
    (total_size / old_size as u128)
        .checked_mul(new_size as u128)
        .is_some_and(|candidate| candidate <= max_abstraction_size as u128)
}

pub(super) fn can_refine_propositional_variable(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    var_id: VariableIndex,
    new_domain_size: usize,
    max_abstraction_size: usize,
) -> bool {
    refined_size_fits(
        domain_sizes,
        numeric_domain_sizes,
        domain_sizes[var_id.index()],
        new_domain_size,
        max_abstraction_size,
    )
}

pub(super) fn can_refine_numeric_variable(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    numeric_var_id: VariableIndex,
    max_abstraction_size: usize,
) -> bool {
    let old_partition_count = numeric_domain_sizes[numeric_var_id.index()];
    refined_size_fits(
        domain_sizes,
        numeric_domain_sizes,
        old_partition_count,
        old_partition_count + 1,
        max_abstraction_size,
    )
}

/// Whether the variable may be refined. A variable whose refinement does
/// not fit the size limit never will, because abstract sizes only grow, so it
/// is blacklisted for the rest of the run: the configured size limit, made
/// permanent per variable.
fn can_refine_propositional_variable_with_blacklist(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    var_id: VariableIndex,
    new_domain_size: usize,
    max_abstraction_size: usize,
    conditions: &NumericConditions,
    blacklisted_prop_var_ids: &mut HashSet<VariableIndex>,
) -> bool {
    if blacklisted_prop_var_ids.contains(&var_id) {
        return false;
    }
    if conditions.is_condition_var(var_id) && domain_sizes[var_id.index()] >= 2 {
        return true;
    }
    if can_refine_propositional_variable(
        domain_sizes,
        numeric_domain_sizes,
        var_id,
        new_domain_size,
        max_abstraction_size,
    ) {
        true
    } else {
        blacklisted_prop_var_ids.insert(var_id);
        false
    }
}

/// Whether the variable may be refined. A variable whose refinement does
/// not fit the size limit never will, because abstract sizes only grow, so it
/// is blacklisted for the rest of the run: the configured size limit, made
/// permanent per variable.
fn can_refine_numeric_variable_with_blacklist(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    numeric_var_id: VariableIndex,
    max_abstraction_size: usize,
    blacklisted_numeric_var_ids: &mut HashSet<VariableIndex>,
) -> bool {
    if blacklisted_numeric_var_ids.contains(&numeric_var_id) {
        return false;
    }
    if can_refine_numeric_variable(
        domain_sizes,
        numeric_domain_sizes,
        numeric_var_id,
        max_abstraction_size,
    ) {
        true
    } else {
        blacklisted_numeric_var_ids.insert(numeric_var_id);
        false
    }
}

/// Port of numeric-FD's refinement step (`fix_flaws`).
///
/// Return the refined variable IDs.
pub fn fix_flaws(
    config: &CegarConfig,
    task: &dyn AbstractNumericTask,
    flaws: &[Flaw],
    mut state: RefinementState<'_>,
    rng: &mut SmallRng,
) -> Result<RefinementSummary> {
    let eligible_flaws = filter_eligible_flaws(task, state.partitions, state.domain_sizes, flaws);
    if eligible_flaws.filtered_stale > 0 {
        debug!(
            "filtered {} stale flaws (already-split values / fully refined vars)",
            eligible_flaws.filtered_stale
        );
    }
    let flaws = eligible_flaws.flaws;
    if flaws.is_empty() {
        return Ok(RefinementSummary::default());
    }

    let chosen_flaws: ChosenFlaws = config.flaw_treatment.choose_flaws(
        task,
        &flaws,
        state.domain_sizes,
        state.numeric_domain_sizes,
        match config.flaw_selection_rank {
            Some(rank) => TieBreak::Rank(rank),
            None => TieBreak::Random(rng),
        },
    );

    for candidate in chosen_flaws {
        let mut chosen = flaws[candidate.idx].clone();
        if let (Flaw::Propositional(pf), Some(restricted)) = (&mut chosen, candidate.restricted_dep)
        {
            pf.dependent_numeric_flaws = restricted;
        }
        if let Some(refined) = try_refine_from_flaw(task, &chosen, config, state.reborrow())? {
            return Ok(refined);
        }
    }
    Ok(RefinementSummary::default())
}

/// The partition count of a numeric variable that was just split.
fn split_partition_count(partitions: &NumericPartitions, numeric_var_id: VariableIndex) -> usize {
    partitions
        .partitions(numeric_var_id)
        .unwrap_or_else(|| panic!("split numeric variable {numeric_var_id} has no partitions"))
        .len()
}

fn try_refine_from_flaw(
    task: &dyn AbstractNumericTask,
    flaw: &Flaw,
    config: &CegarConfig,
    state: RefinementState<'_>,
) -> Result<Option<RefinementSummary>> {
    let RefinementState {
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
        blacklisted_prop_var_ids,
        blacklisted_numeric_var_ids,
    } = state;
    match flaw {
        Flaw::Numeric(nf) => {
            let var_id = nf.numeric_var_id;
            if !can_refine_numeric_variable_with_blacklist(
                domain_sizes,
                numeric_domain_sizes,
                var_id,
                config.max_abstraction_size,
                blacklisted_numeric_var_ids,
            ) {
                return Ok(None);
            }
            if partitions.split_at(var_id, nf.value, nf.include_in_lower) {
                numeric_domain_sizes[var_id.index()] = split_partition_count(partitions, var_id);
                let mut refined = RefinementSummary::default();
                refined.mark_numeric(var_id);
                return Ok(Some(refined));
            }
            Ok(None)
        }
        Flaw::Propositional(pf) => {
            let var_id = pf.fact.var_index();
            let value = pf.fact.value();

            // Flaws name facts of the task the abstraction is built for.
            assert!(
                var_id.index() < domain_mapping.len() && var_id.index() < domain_sizes.len(),
                "flaw variable {var_id} is outside the abstraction (mapping.len={}, domain_sizes.len={})",
                domain_mapping.len(),
                domain_sizes.len()
            );
            let concrete_size = task
                .get_variable_domain_size(var_id)
                .unwrap_or_else(|error| panic!("flaw variable {var_id} has no domain: {error}"));
            assert!(
                value < concrete_size,
                "flaw fact value {value} is outside the domain of variable {var_id} (size {concrete_size})"
            );
            assert_eq!(
                domain_mapping[var_id.index()].len(),
                concrete_size,
                "the domain mapping of variable {var_id} covers its concrete domain"
            );

            let mut changed = false;
            let mut prop_domain_size_changed = false;

            if task.numeric_conditions().is_condition_var(var_id) {
                if !can_refine_propositional_variable_with_blacklist(
                    domain_sizes,
                    numeric_domain_sizes,
                    var_id,
                    2,
                    config.max_abstraction_size,
                    task.numeric_conditions(),
                    blacklisted_prop_var_ids,
                ) {
                    return Ok(None);
                }
                // A condition variable has one split and only one: the two
                // values of its domain, told apart. Refining it therefore means
                // going straight to the full mapping.
                let old_size = domain_sizes[var_id.index()];
                if domain_sizes[var_id.index()] < ConditionValue::DOMAIN_SIZE {
                    domain_sizes[var_id.index()] = ConditionValue::DOMAIN_SIZE;
                    changed = true;
                    prop_domain_size_changed = true;
                }
                if domain_mapping[var_id.index()] != CONDITION_SPLIT_MAPPING {
                    domain_mapping[var_id.index()] = CONDITION_SPLIT_MAPPING.to_vec();
                    changed = true;
                }
                debug_assert!(domain_sizes[var_id.index()] >= old_size);
            } else {
                let abs_size = domain_sizes[var_id.index()];
                // If we've already fully refined this variable, nothing to do.
                if abs_size >= concrete_size {
                    return Ok(None);
                }
                // Only refine if the value is still mapped to the default class (0).
                if domain_mapping[var_id.index()][value] != ExplicitValueIndex::new(0) {
                    return Ok(None);
                }
                if !can_refine_propositional_variable_with_blacklist(
                    domain_sizes,
                    numeric_domain_sizes,
                    var_id,
                    abs_size + 1,
                    config.max_abstraction_size,
                    task.numeric_conditions(),
                    blacklisted_prop_var_ids,
                ) {
                    return Ok(None);
                }

                domain_mapping[var_id.index()][value] = ExplicitValueIndex::from_usize(abs_size);
                domain_sizes[var_id.index()] = abs_size + 1;
                changed = true;
                prop_domain_size_changed = true;
            }

            // Dependent numeric refinements, produced only for comparison
            // variables: split the first operand that can still be split.
            if !pf.dependent_numeric_flaws.is_empty() {
                let mut any_numeric_changed = false;
                let mut refined = RefinementSummary::default();
                if prop_domain_size_changed {
                    refined.mark_propositional(var_id);
                }
                for dep in &pf.dependent_numeric_flaws {
                    let num_id = dep.numeric_var_id;

                    if !can_refine_numeric_variable_with_blacklist(
                        domain_sizes,
                        numeric_domain_sizes,
                        num_id,
                        config.max_abstraction_size,
                        blacklisted_numeric_var_ids,
                    ) {
                        continue;
                    }

                    if partitions.split_at(num_id, dep.value, dep.include_in_lower) {
                        numeric_domain_sizes[num_id.index()] =
                            split_partition_count(partitions, num_id);
                        any_numeric_changed = true;
                        refined.mark_numeric(num_id);
                        break;
                    }
                }
                return Ok((any_numeric_changed || changed).then_some(refined));
            }

            if changed {
                let mut refined = RefinementSummary::default();
                if prop_domain_size_changed {
                    refined.mark_propositional(var_id);
                }
                Ok(Some(refined))
            } else {
                Ok(None)
            }
        }
    }
}
