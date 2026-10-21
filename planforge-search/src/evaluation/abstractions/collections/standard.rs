//! Standard collections (SoCS 2026): full-task abstractions by random
//! restarts, shared by every abstraction backend.
//!
//! Every restart refines the full task from the trivial abstraction split on
//! one initial variable, which the candidates take turns being, with a fresh
//! seed. A restart that repeats an earlier member is dropped. Once the time
//! budget is mostly spent, or once no new member appeared for the stagnation
//! limit, restarts also blacklist a random subset of the candidates, which
//! CEGAR then may not refine. The collection stops at its state or time
//! budget, at stagnation while blacklisting, at the memory limit, or at a
//! member that settles the task: its abstract plan is a concrete plan, or its
//! initial state is an abstract dead end, and no further member can help the
//! search.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use planforge_sas::numeric_task::{AbstractNumericTask, ExplicitFact, NumericType, VariableIndex};
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng, rngs::SmallRng};
use tracing::{debug, info};

use super::budget::{CollectionBudget, CollectionLimits, CollectionStopReason};
use super::config::{MemberConfig, SplitKinds};
use super::generator::AbstractionGenerator;
use super::member::{CollectionMember, MemberConstruction};
use crate::evaluation::abstractions::common::cegar::initial_splits::{
    InitialSplitVar, RefinementScope,
};
use crate::resource_limits;

/// The budgets and diversification of a standard collection.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StandardSchedule {
    pub(crate) limits: CollectionLimits,
    /// How long without a new member the collection stagnates after; never
    /// when `None`.
    pub(crate) stagnation_limit: Option<Duration>,
    /// How long after the start restarts blacklist; only at stagnation when
    /// `None`.
    pub(crate) blacklist_start: Option<Duration>,
    /// Start blacklisting at stagnation instead of stopping.
    pub(crate) enable_blacklist_on_stagnation: bool,
    /// The seed the restarts' seeds and blacklists are drawn with.
    pub(crate) random_seed: u64,
    /// The split kinds of the restarts, in order.
    pub(crate) split_kinds: SplitKinds,
    /// The members' configured rank among tied best splits.
    pub(crate) split_selection_rank: Option<usize>,
}

/// Builds the standard collection of `task` with members from `G`
/// configured by `member`.
pub(crate) fn build_standard_collection<G: AbstractionGenerator>(
    task: &dyn AbstractNumericTask,
    schedule: &StandardSchedule,
    member: &G::Config,
) -> Result<Vec<G::Abstraction>> {
    let mut rng = SmallRng::seed_from_u64(schedule.random_seed);
    let candidates = initial_split_candidates(task);

    let start = Instant::now();
    let mut budget = CollectionBudget::new(schedule.limits, start)?;
    let mut keys: HashSet<G::Key> = HashSet::new();
    let mut members: Vec<G::Abstraction> = Vec::new();
    let mut last_new_member = start;
    let mut blacklisting = false;
    let mut iteration = 1usize;
    let stop_reason = loop {
        let iteration_start = Instant::now();
        if !blacklisting
            && schedule
                .blacklist_start
                .is_some_and(|blacklist_start| iteration_start - start > blacklist_start)
        {
            blacklisting = true;
            last_new_member = iteration_start;
        }

        let max_states = budget.member_max_states();
        if max_states == 0 {
            break CollectionStopReason::SizeLimit;
        }
        let max_time = budget.member_max_time();
        // Numeric FD constructs one abstraction before checking the
        // collection's time limit: a collection is never empty.
        if max_time == Some(Duration::ZERO) && !members.is_empty() {
            break CollectionStopReason::TimeLimit;
        }

        let (blacklisted_prop_var_ids, blacklisted_numeric_var_ids) = if blacklisting {
            sample_blacklist(&candidates, &mut rng)
        } else {
            Default::default()
        };
        let construction = MemberConstruction {
            max_states,
            max_time,
            random_seed: Some(rng.next_u64()),
            split_kind: schedule.split_kinds.of(iteration - 1),
            split_selection_rank: schedule.split_selection_rank,
            scope: RefinementScope {
                initial_split_var: (!candidates.is_empty())
                    .then(|| candidates[iteration % candidates.len()]),
                blacklisted_prop_var_ids,
                blacklisted_numeric_var_ids,
            },
        };
        debug!(
            "{} collection: building restart {iteration}, max_time={max_time:?}, split_kind={}",
            G::KIND,
            construction.split_kind
        );
        let mut built = match G::new(member.for_member(&construction))
            .and_then(|generator| generator.generate_from_root(task, None, &construction.scope))
        {
            Ok(built) => built,
            Err(error) if resource_limits::is_deadline_exceeded(&error) => {
                info!(
                    "{} collection: generation deadline expired at restart {iteration}; keeping {} members",
                    G::KIND,
                    members.len()
                );
                break CollectionStopReason::GenerationDeadline;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to build {} collection restart {iteration}", G::KIND)
                });
            }
        };

        let settles = built.settles();
        let iteration_end = Instant::now();
        if keys.insert(G::key(&built)) {
            last_new_member = iteration_end;
            budget.charge(built.report().num_states);
            built.record(iteration, None);
            members.push(built);
            debug!(
                "{} collection: added restart {iteration}, remaining_collection_size={}, blacklisting={blacklisting}",
                G::KIND,
                budget.remaining_states()
            );
        }

        if let Some(reason) = settles {
            break reason;
        }
        let stagnated = schedule
            .stagnation_limit
            .is_some_and(|limit| iteration_end - last_new_member > limit);
        if budget.remaining_states() == 0 {
            break CollectionStopReason::SizeLimit;
        }
        if budget.deadline_passed() {
            break CollectionStopReason::TimeLimit;
        }
        if stagnated && (!schedule.enable_blacklist_on_stagnation || blacklisting) {
            break CollectionStopReason::StagnationLimit;
        }
        // Release the reserved padding before the process reaches its hard
        // limit, leaving enough memory for search to report a useful result.
        if !resource_limits::poll_and_release_if_exceeded() {
            info!(
                "{} collection: stopping at restart {iteration} because the memory padding was released (RSS limit reached)",
                G::KIND
            );
            break CollectionStopReason::MemoryLimit;
        }
        if stagnated && schedule.enable_blacklist_on_stagnation {
            blacklisting = true;
            last_new_member = iteration_start;
        }
        iteration += 1;
    };

    if members.is_empty() {
        bail!(
            "the {} collection failed to generate its mandatory first abstraction ({stop_reason})",
            G::KIND
        )
    }
    info!(
        "{} collection finished: abstractions={}, states={}, elapsed={:.3}s, stop_reason={stop_reason}",
        G::KIND,
        members.len(),
        budget.used_states(),
        start.elapsed().as_secs_f64(),
    );
    Ok(members)
}

/// The variables restarts split initially and blacklist: every propositional
/// variable no axiom writes, except condition variables that no goal
/// mentions, and then every regular numeric variable.
fn initial_split_candidates(task: &dyn AbstractNumericTask) -> Vec<InitialSplitVar> {
    let goal_var_ids: HashSet<VariableIndex> =
        task.goals().iter().map(ExplicitFact::var_index).collect();
    let axiom_var_ids: HashSet<VariableIndex> =
        task.axioms().iter().map(|axiom| axiom.var_id()).collect();
    let propositional = (0..task.variables().len())
        .map(VariableIndex::from_usize)
        .filter(|var_id| {
            !axiom_var_ids.contains(var_id)
                && (!task.numeric_conditions().is_condition_var(*var_id)
                    || goal_var_ids.contains(var_id))
        })
        .map(InitialSplitVar::Propositional);
    let numeric = task
        .numeric_variables()
        .iter()
        .enumerate()
        .filter(|(_, variable)| variable.get_type() == &NumericType::Regular)
        .map(|(numeric_var_id, _)| {
            InitialSplitVar::Numeric(VariableIndex::from_usize(numeric_var_id))
        });
    propositional.chain(numeric).collect()
}

/// A random non-empty subset of `candidates`, empty only without candidates,
/// as propositional and numeric variables.
fn sample_blacklist(
    candidates: &[InitialSplitVar],
    rng: &mut SmallRng,
) -> (HashSet<VariableIndex>, HashSet<VariableIndex>) {
    let mut propositional = HashSet::new();
    let mut numeric = HashSet::new();
    if candidates.is_empty() {
        return (propositional, numeric);
    }
    let size = rng.gen_range(1..=candidates.len());
    let mut shuffled = candidates.to_vec();
    shuffled.shuffle(rng);
    for var in shuffled.into_iter().take(size) {
        match var {
            InitialSplitVar::Propositional(var_id) => propositional.insert(var_id),
            InitialSplitVar::Numeric(var_id) => numeric.insert(var_id),
        };
    }
    (propositional, numeric)
}
