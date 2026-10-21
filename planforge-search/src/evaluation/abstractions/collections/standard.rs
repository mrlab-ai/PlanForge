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
use std::hash::Hash;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use planforge_sas::numeric_task::{AbstractNumericTask, ExplicitFact, NumericType, VariableIndex};
use rand::seq::SliceRandom;
use rand::{Rng, RngCore, SeedableRng, rngs::SmallRng};
use tracing::{debug, info};

use super::config::SplitKinds;
use super::member::CollectionMember;
use crate::evaluation::abstractions::common::cegar::SplitKind;
use crate::evaluation::abstractions::common::cegar::initial_splits::InitialSplitVar;
use crate::resource_limits;

/// The budgets and diversification of a standard collection.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StandardSchedule {
    pub(crate) max_states: usize,
    pub(crate) max_collection_size: usize,
    /// Seconds per member; infinite when unbounded.
    pub(crate) max_time: f64,
    /// Seconds for the collection; infinite when unbounded.
    pub(crate) total_max_time: f64,
    /// Seconds without a new member after which the collection stagnates.
    pub(crate) stagnation_limit: f64,
    /// The share of `total_max_time` after which restarts blacklist.
    pub(crate) blacklist_trigger_percentage: f64,
    /// Start blacklisting at stagnation instead of stopping.
    pub(crate) enable_blacklist_on_stagnation: bool,
    /// The seed the restarts' seeds, initial splits and blacklists are drawn
    /// with.
    pub(crate) random_seed: u64,
    pub(crate) split_kinds: SplitKinds,
}

impl StandardSchedule {
    /// The split kind of restart `iteration` (one-based).
    pub(crate) fn split_kind(&self, iteration: usize) -> SplitKind {
        assert!(iteration > 0, "restarts are one-based");
        self.split_kinds.of(iteration - 1)
    }
}

/// How one restart is built.
#[derive(Debug, Clone)]
pub struct StandardMember {
    pub(crate) max_states: usize,
    /// Seconds; infinite when unbounded.
    pub(crate) max_time: f64,
    pub(crate) seed: u64,
    pub(crate) split_kind: SplitKind,
    pub(crate) initial_split_var: Option<InitialSplitVar>,
    pub(crate) blacklisted_prop_var_ids: HashSet<VariableIndex>,
    pub(crate) blacklisted_numeric_var_ids: HashSet<VariableIndex>,
}

/// One abstraction backend of a standard collection.
pub(crate) trait StandardBackend {
    type Member: CollectionMember;
    /// Equal exactly for members that are the same abstraction.
    type Key: Eq + Hash;

    const NAME: &'static str;

    /// Builds the member; a deadline error ends the collection.
    fn build(
        &self,
        task: &dyn AbstractNumericTask,
        member: &StandardMember,
    ) -> Result<Self::Member>;

    fn key(&self, member: &Self::Member) -> Self::Key;
}

/// Builds the standard collection of `task` with `backend`.
pub(crate) fn build_standard_collection<B: StandardBackend>(
    task: &dyn AbstractNumericTask,
    schedule: &StandardSchedule,
    backend: &B,
) -> Result<Vec<B::Member>> {
    let mut rng = SmallRng::seed_from_u64(schedule.random_seed);
    let candidates = initial_split_candidates(task);

    let start = Instant::now();
    let mut remaining_collection_size = schedule.max_collection_size;
    let mut keys: HashSet<B::Key> = HashSet::new();
    let mut members: Vec<B::Member> = Vec::new();
    let mut time_point_of_last_new_member = 0.0f64;
    let mut blacklisting = false;
    let blacklist_start_time = schedule.total_max_time * schedule.blacklist_trigger_percentage;
    let mut iteration = 1usize;
    let stop_reason: &str;
    loop {
        let elapsed = start.elapsed().as_secs_f64();
        if !blacklisting && elapsed > blacklist_start_time {
            blacklisting = true;
            time_point_of_last_new_member = elapsed;
        }

        let remaining_total_time = if schedule.total_max_time.is_finite() {
            (schedule.total_max_time - elapsed).max(0.0)
        } else {
            f64::INFINITY
        };
        let max_time = schedule.max_time.min(remaining_total_time);
        let max_states = remaining_collection_size.min(schedule.max_states);
        if max_states == 0 {
            stop_reason = "collection size limit";
            break;
        }
        // Numeric FD constructs one abstraction before checking the
        // collection's time limit: a collection is never empty.
        if max_time <= 0.0 && !members.is_empty() {
            stop_reason = "collection time limit";
            break;
        }

        let mut blacklisted_prop_var_ids = HashSet::new();
        let mut blacklisted_numeric_var_ids = HashSet::new();
        if blacklisting {
            for var in sample_blacklist(&candidates, &mut rng) {
                match var {
                    InitialSplitVar::Propositional(var_id) => {
                        blacklisted_prop_var_ids.insert(var_id);
                    }
                    InitialSplitVar::Numeric(var_id) => {
                        blacklisted_numeric_var_ids.insert(var_id);
                    }
                }
            }
        }
        let member = StandardMember {
            max_states,
            max_time,
            seed: rng.next_u64(),
            split_kind: schedule.split_kind(iteration),
            initial_split_var: (!candidates.is_empty())
                .then(|| candidates[iteration % candidates.len()]),
            blacklisted_prop_var_ids,
            blacklisted_numeric_var_ids,
        };
        debug!(
            "{} collection: building restart {iteration}, max_time={max_time:.2}s, split_kind={}",
            B::NAME,
            member.split_kind
        );
        let mut built = match backend.build(task, &member) {
            Ok(built) => built,
            Err(error) if resource_limits::is_deadline_exceeded(&error) => {
                info!(
                    "{} collection: generation deadline expired at restart {iteration}; keeping {} members",
                    B::NAME,
                    members.len()
                );
                stop_reason = "abstraction generation deadline";
                break;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to build {} collection restart {iteration}", B::NAME)
                });
            }
        };

        let report = built.report();
        let elapsed_after_build = start.elapsed().as_secs_f64();
        if keys.insert(backend.key(&built)) {
            time_point_of_last_new_member = elapsed_after_build;
            let consumed = report.num_states;
            assert!(
                consumed <= max_states,
                "{} restart {iteration} has {consumed} states, over its limit of {max_states}",
                B::NAME
            );
            remaining_collection_size -= consumed;
            built.record(iteration, None);
            members.push(built);
            debug!(
                "{} collection: added restart {iteration}, states={consumed}, remaining_collection_size={remaining_collection_size}, blacklisting={blacklisting}",
                B::NAME
            );
        }

        if let Some(reason) = report.settles() {
            stop_reason = reason;
            break;
        }
        let stagnated =
            elapsed_after_build - time_point_of_last_new_member > schedule.stagnation_limit;
        if remaining_collection_size == 0 {
            stop_reason = "collection size limit";
            break;
        }
        if schedule.total_max_time.is_finite() && elapsed_after_build >= schedule.total_max_time {
            stop_reason = "collection time limit";
            break;
        }
        if stagnated && (!schedule.enable_blacklist_on_stagnation || blacklisting) {
            stop_reason = "stagnation limit";
            break;
        }
        // Release the reserved padding before the process reaches its hard
        // limit, leaving enough memory for search to report a useful result.
        if !resource_limits::poll_and_release_if_exceeded() {
            info!(
                "{} collection: stopping at restart {iteration} because the memory padding was released (RSS limit reached)",
                B::NAME
            );
            stop_reason = "memory limit";
            break;
        }
        if stagnated && schedule.enable_blacklist_on_stagnation {
            blacklisting = true;
            time_point_of_last_new_member = elapsed;
        }
        iteration += 1;
    }

    if members.is_empty() {
        bail!(
            "the {} collection failed to generate its mandatory first abstraction ({stop_reason})",
            B::NAME
        )
    }
    info!(
        "{} collection finished: abstractions={}, states={}, elapsed={:.3}s, stop_reason={stop_reason}",
        B::NAME,
        members.len(),
        members
            .iter()
            .map(|member| member.report().num_states)
            .sum::<usize>(),
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

/// A random non-empty subset of `candidates`, empty only without candidates.
fn sample_blacklist(candidates: &[InitialSplitVar], rng: &mut SmallRng) -> Vec<InitialSplitVar> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let size = rng.gen_range(1..=candidates.len());
    let mut shuffled = candidates.to_vec();
    shuffled.shuffle(rng);
    shuffled.truncate(size);
    shuffled
}
