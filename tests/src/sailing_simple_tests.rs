//! The hand-written `sailing-simple` corpus.
//!
//! `assets/numeric-pddl-files/sailing-simple/README.md` derives an optimum for
//! every instance by hand and states that those numbers must be machine-checked
//! with blind A* before any heuristic is pinned against them.
//! [`SAILING_SIMPLE_OPTIMA`] is that check, and it is compared set-wise against
//! the instances on disk so a new instance cannot be added without an optimum.
//!
//! Everything else here is an abstraction or cost-partitioning test that reads
//! its ground truth from that table.
//!
//! # Where the time goes
//!
//! This module used to be 95% of the whole test suite: 116.7s of a 122s
//! `cargo test --workspace` run, against 5.1s for the next largest crate group.
//! Measured per test in an unoptimized build, that was
//!
//! | test | before | after |
//! |---|---|---|
//! | `..._multiboat_scp_falls_short_of_the_additive_optimum` | 91.3s | 5.9s |
//! | `blind_astar_reproduces_every_..._optimum` | 72.2s | 3.0s + optimized-only |
//! | `sailing_simple_assignment_gap` | 55.0s | 5.9s |
//! | `sailing_simple_scp_online_admissible` | 37.2s | 1.3s |
//! | `sailing_simple_ratchet_equilibrium` | 25.1s | 3.3s |
//! | `..._complementary_collection_keeps_single_goal_solved_abstractions` | 18.9s | 2.1s |
//!
//! and almost none of it was the assertions. The collection generator was
//! stopped by a wall-clock budget it could never stagnate out of, so it ran the
//! clock out on every instance and the collection it handed back was whatever
//! the machine happened to build in that window -- 3 members in one run of the
//! same binary and 32 in the next, because each iteration that fits in the
//! budget derives a fresh CEGAR seed and a *deadline-truncated* abstraction is
//! a distinct abstraction. Region-CP table construction then paid for all of
//! them. Replacing that budget with the two clock-free stopping rules the
//! generator already has -- a collection-size limit and "stop at the first
//! duplicate" -- reproduces every asserted value from a 2-to-6-member
//! collection instead of a 2-to-71-member one, and takes the module from 116.7s
//! to 6.4s. See [`standard_round7_collection_config`].
//!
//! What is left that genuinely needs the time is blind A* on the three biggest
//! instances: an optimum can only be confirmed by searching for it, and those
//! three are 87% of the corpus's expansions. They run in an optimized build
//! only, where the same search is an order of magnitude cheaper -- see
//! [`BlindCost`]. CI covers them in the `release tests` job; locally,
//! `cargo test -p tests --release`.

use std::path::PathBuf;
use std::sync::Arc;

use planforge_sas::numeric_task::{AbstractNumericTask, NumericRootTask};
use planforge_sas::state_registry::StateRegistry;
use planforge_search::evaluation::abstraction_collections::portfolio::CollectionStrategy;
use planforge_search::evaluation::abstraction_collections::saturated_cost_partitioning_online_heuristic::{
    CostPartitioningMethod, SaturatedCostPartitioningOnlineHeuristic, ScpOnlineConfig,
};
use planforge_search::evaluation::cartesian_abstractions::{
    CartesianAbstractionConfig, CartesianAbstractionGenerator, CartesianAbstractionHeuristic,
};
use planforge_search::evaluation::domain_abstractions::cegar::{CegarConfig, FlawKind};
use planforge_search::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegar,
    DomainAbstractionCollectionGeneratorMultipleCegarConfig,
    FlawTreatmentVariants,
    InitSplitMethod,
    InitSplitQuantity,
    NumericSplitStrategy,
};
use planforge_search::evaluation::domain_abstractions::domain_abstraction_generator::{
    DomainAbstraction, DomainAbstractionGenerator,
};
use planforge_search::evaluation::domain_abstractions::domain_abstraction_heuristic::DomainAbstractionHeuristic;
use planforge_search::evaluation::evaluator::EvaluationState;
use planforge_search::evaluation::heuristic::Heuristic;
use planforge_search::search::{AStarSearch, SearchEngine, SearchStatus};
use planforge_search::task_restriction::build_restricted_task;

use crate::corpus::{
    self, Scratch, assert_fixture_set_is_pinned, blind_astar_cost, problem_file_names,
    translate_to_disk,
};

/// What blind A* costs on an instance, and therefore which build verifies it.
///
/// Measured with the compact numeric state representation using
/// `target/release/planforge --search 'astar(blind())'`, expanded states per
/// instance:
///
/// | instance | expanded | class |
/// |---|---|---|
/// | `prob_1b1p_x` | 363 | `Cheap` |
/// | `prob_1b1p_diag` | 380 | `Cheap` |
/// | `prob_1b2p_x` | 1_070 | `Cheap` |
/// | `prob_1b2p_diag` | 2_125 | `Cheap` |
/// | `prob_2b1p` | 21_843 | `Cheap` |
/// | `prob_1b1p_far` | 39_603 | `Cheap` |
/// | `prob_1b4p_axes` | 146_243 | `Costly` |
/// | `prob_2b2p_assign` | 216_257 | `Costly` |
/// | `prob_2b2p_x` | 520_975 | `Costly` |
///
/// The three `Costly` instances are 87% of the corpus's 883_859 expansions and
/// there is no cheaper way to confirm an optimum than to search for it, so they
/// are verified in an optimized build (where all nine cost 3.5s of search) and
/// the six `Cheap` ones in every build.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BlindCost {
    Cheap,
    Costly,
}

/// Hand-derived optima from the corpus README, machine-verified by
/// [`blind_astar_reproduces_every_cheap_sailing_simple_optimum`] and
/// [`blind_astar_reproduces_every_costly_sailing_simple_optimum`].
const SAILING_SIMPLE_OPTIMA: &[(&str, f64, BlindCost)] = &[
    ("prob_1b1p_x", 11.0, BlindCost::Cheap),
    ("prob_1b1p_diag", 11.0, BlindCost::Cheap),
    ("prob_1b1p_far", 101.0, BlindCost::Cheap),
    ("prob_1b2p_x", 17.0, BlindCost::Cheap),
    ("prob_1b2p_diag", 22.0, BlindCost::Cheap),
    ("prob_1b4p_axes", 74.0, BlindCost::Costly),
    ("prob_2b1p", 11.0, BlindCost::Cheap),
    ("prob_2b2p_x", 22.0, BlindCost::Costly),
    ("prob_2b2p_assign", 17.0, BlindCost::Costly),
];

fn instances_root() -> PathBuf {
    corpus::assets().join("numeric-pddl-files/sailing-simple")
}

fn sailing_task(instance: &str) -> (NumericRootTask, Scratch) {
    let root = instances_root();
    let scratch = Scratch::new(&format!("sailing_simple_{instance}"));
    let task = translate_to_disk(
        &root.join("domain.pddl"),
        &root.join(format!("{instance}.pddl")),
        &scratch,
    );
    (task, scratch)
}

fn optimum_of(instance: &str) -> f64 {
    SAILING_SIMPLE_OPTIMA
        .iter()
        .find(|(name, _, _)| *name == instance)
        .unwrap_or_else(|| panic!("{instance} is not in SAILING_SIMPLE_OPTIMA"))
        .1
}

/// Blind A* is optimal, so this both verifies one hand-derived optimum and pins
/// that `build_restricted_task` is plan-preserving: the restriction promotes
/// derived condition roots, and a restriction that changed reachability would
/// change the cost.
fn assert_optimum_survives_restriction(instance: &str, expected: f64) {
    let (task, _scratch) = sailing_task(instance);
    assert_eq!(
        blind_astar_cost(&task, instance),
        expected,
        "machine-verified h* changed for {instance}"
    );

    let restricted = build_restricted_task(&task)
        .expect("sailing-simple restricted task construction must not fail")
        .expect("sailing-simple instances have promotable derived roots")
        .into_task();
    assert_eq!(
        blind_astar_cost(&restricted, instance),
        expected,
        "the restricted task is not plan-preserving for {instance}"
    );
}

/// The six instances blind A* solves in well under 40_000 expansions, plus the
/// set-wise comparison that makes [`SAILING_SIMPLE_OPTIMA`] load-bearing. The
/// comparison lives here rather than in the optimized-only test below so that
/// an unpinned instance fails in every build.
#[test]
fn blind_astar_reproduces_every_cheap_sailing_simple_optimum() {
    let discovered: Vec<String> = problem_file_names(&instances_root())
        .iter()
        .map(|file| {
            file.strip_suffix(".pddl")
                .unwrap_or_else(|| panic!("{file} does not end in .pddl"))
                .to_owned()
        })
        .collect();
    let pinned: Vec<&str> = SAILING_SIMPLE_OPTIMA
        .iter()
        .map(|(name, _, _)| *name)
        .collect();
    assert_fixture_set_is_pinned("sailing-simple", &discovered, &pinned);

    for &(instance, expected, cost) in SAILING_SIMPLE_OPTIMA {
        if cost == BlindCost::Cheap {
            assert_optimum_survives_restriction(instance, expected);
        }
    }
}

/// The three instances that are 87% of the corpus's blind-A* expansions. See
/// [`BlindCost`] for the per-instance measurement.
#[cfg(not(debug_assertions))]
#[test]
fn blind_astar_reproduces_every_costly_sailing_simple_optimum() {
    for &(instance, expected, cost) in SAILING_SIMPLE_OPTIMA {
        if cost == BlindCost::Costly {
            assert_optimum_survives_restriction(instance, expected);
        }
    }
}

fn assert_exact_single_abstraction_search<H>(
    task: &NumericRootTask,
    heuristic: H,
    expected_cost: f64,
    backend: &str,
) where
    H: Heuristic,
{
    assert!(
        heuristic.proves_initial_state_optimal(),
        "unrestricted {backend} CEGAR must finish with a concrete plan"
    );
    let state_registry = StateRegistry::for_task(Arc::new(task));
    let mut search = AStarSearch::new(task, state_registry, Some(Box::new(heuristic)), None, None);
    let result = search
        .search()
        .expect("single-abstraction A* search failed");

    assert!(
        matches!(result.status, SearchStatus::Solved(_)),
        "A* with the unrestricted {backend} abstraction did not solve: {:?}",
        result.status
    );
    assert_eq!(
        result.solution_cost,
        Some(expected_cost),
        "A* with the unrestricted {backend} abstraction changed the optimal cost"
    );
    assert_eq!(
        result.nodes_expanded_until_last_jump, 0,
        "an unrestricted {backend} abstraction that proves h(init) = h* must start A* in its final f-layer"
    );
}

#[test]
fn unrestricted_single_abstractions_start_astar_in_final_f_layer() {
    let (task, _scratch) = sailing_task("prob_1b1p_x");
    let optimum = optimum_of("prob_1b1p_x");

    let domain_abstraction = DomainAbstractionGenerator::new(CegarConfig {
        max_iterations: usize::MAX,
        random_seed: Some(1),
        compute_operator_footprints: false,
        ..Default::default()
    })
    .expect("unrestricted domain abstraction generator should construct")
    .generate(&task)
    .expect("unrestricted domain abstraction should solve sailing-simple");
    assert!(
        domain_abstraction.metadata.solved_by_self,
        "unrestricted domain abstraction stopped without a real plan: metadata={:?}, prop_domains={:?}, numeric_domains={:?}",
        domain_abstraction.metadata,
        domain_abstraction.factory.domain_sizes(),
        domain_abstraction.factory.numeric_domain_sizes()
    );
    assert_exact_single_abstraction_search(
        &task,
        DomainAbstractionHeuristic::new(None, domain_abstraction),
        optimum,
        "domain",
    );

    let cartesian_abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        max_states: usize::MAX,
        compute_operator_footprints: false,
        ..Default::default()
    })
    .expect("unrestricted Cartesian abstraction generator should construct")
    .generate(&task)
    .expect("unrestricted Cartesian abstraction should solve sailing-simple");
    assert!(cartesian_abstraction.metadata.solved_by_self);
    assert_exact_single_abstraction_search(
        &task,
        CartesianAbstractionHeuristic::new(None, cartesian_abstraction),
        optimum,
        "Cartesian",
    );
}

/// The collection configuration every cost-partitioning test below shares.
///
/// This used to stop the generator on `total_max_time = 15s` with
/// `stagnation_limit` at 30s -- which is to say it could never stagnate, so it
/// ran the clock out on every instance, however small. Worse, what it handed
/// back was not a function of its inputs: the CEGAR seed is derived from the
/// iteration counter and an abstraction cut off by
/// `abstraction_generation_max_time` is a distinct abstraction, so every extra
/// iteration the machine squeezed into the window added a member. The same
/// binary on the same instance produced 3 members in one run and 32 in the next,
/// and region-CP table construction then paid for all of them -- 63s of it on
/// `prob_2b2p_x`.
///
/// The generator already has two stopping rules that do not read a clock, and
/// they are what bound it now:
///
/// * `max_collection_size` -- an abstract-state budget, decremented by each new
///   member. Kept equal to `max_abstraction_size` so the first member still gets
///   the full per-abstraction cap and is unaffected.
/// * `stagnation_limit = 0.0` with `enable_blacklist_on_stagnation = false` --
///   an iteration that produces an abstraction already in the collection ends
///   it. `time_point_of_last_new_abstraction` is assigned the same `f64` that is
///   compared against it, so a productive iteration compares `0.0 > 0.0` and a
///   duplicate compares a positive elapsed difference: the rule is "stop at the
///   first duplicate", not a duration.
///
/// `total_max_time` and `abstraction_generation_max_time` stay only as hang
/// guards, raised rather than lowered so that they cannot truncate an
/// abstraction and put the clock back into the result. Collection generation now
/// measures 0.06-0.25s against those 30s and 10s.
///
/// Every value the tests below assert is reproduced, from collections one order
/// of magnitude smaller. Measured (unoptimized, members `n`, initial h, and the
/// whole test):
///
/// | instance | n | h | before | after |
/// |---|---|---|---|---|
/// | `prob_1b1p_x` | 2 | 11 | 15.2s | 0.14s |
/// | `prob_1b2p_x` | 4 | 17 | 21.2s | 1.09s |
/// | `prob_2b1p` | 6 | 11 | 25.1s | 3.3s |
/// | `prob_2b2p_x` | 3 | 12 | 91.3s | 5.9s |
/// | `prob_2b2p_assign` | 2 | 7 | 55.0s | 5.9s |
///
/// and the h column is the same one the 2-to-71-member time-bounded collections
/// produced, so this is the identical claim bought for a fiftieth of the price.
fn standard_round7_collection_config(
    seed: u64,
) -> DomainAbstractionCollectionGeneratorMultipleCegarConfig {
    DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        max_abstraction_size: 10_000,
        max_collection_size: 10_000,
        abstraction_generation_max_time: 10.0,
        total_max_time: 30.0,
        stagnation_limit: 0.0,
        enable_blacklist_on_stagnation: false,
        blacklist_trigger_percentage: 1.0,
        init_split_quantity: InitSplitQuantity::All,
        init_split_method: InitSplitMethod::RandomValue,
        flaw_treatment: FlawTreatmentVariants::MaxRefinedSingleAtom,
        numeric_split_strategy: NumericSplitStrategy::Standard,
        use_wildcard_plans: true,
        combine_labels: false,
        flaw_kind: FlawKind::SequenceBidirectional,
        collection_strategy: CollectionStrategy::Complementary,
        random_seed: Some(seed),
        ..Default::default()
    }
}

fn scp_online_initial_h_with_config(
    instance: &str,
    collection_config: DomainAbstractionCollectionGeneratorMultipleCegarConfig,
) -> f64 {
    let (task, _scratch) = sailing_task(instance);
    let task = build_restricted_task(&task)
        .expect("sailing-simple restricted task construction must not fail")
        .expect("sailing-simple instances have promotable derived roots")
        .into_task();
    let generator =
        DomainAbstractionCollectionGeneratorMultipleCegar::new(collection_config.clone());
    let abstractions = generator
        .generate_collection(&task)
        .expect("scp_online domain abstractions should build");
    scp_online_initial_h_for_collection(&task, abstractions, collection_config)
}

/// Greedy order optimization is the third budget that is spent in full instead
/// of on convergence: at the 5s default, a single initial-state evaluation of
/// `prob_1b1p_x` costs 5.54s, of which 5.38s is the optimizer and 0.16s is
/// everything else. Initial h is unchanged at 1s and at 0s on every instance
/// asserted here (`prob_1b1p_x` 11, `prob_1b2p_x` 17, `prob_2b1p` 11,
/// `prob_2b2p_x` 12, `prob_2b2p_assign` 7), so 1s keeps the optimizer on the
/// hook for a regression while paying a fifth of the price.
const ORDER_OPTIMIZATION_MAX_TIME: f64 = 1.0;

fn scp_online_initial_h_for_collection(
    task: &NumericRootTask,
    abstractions: Vec<DomainAbstraction>,
    collection_config: DomainAbstractionCollectionGeneratorMultipleCegarConfig,
) -> f64 {
    let config = ScpOnlineConfig {
        max_time: 100.0,
        max_size: 10_000_000,
        interval: 100_000_000_000,
        table_construction_max_time: 100.0,
        order_optimization_max_time: ORDER_OPTIMIZATION_MAX_TIME,
        collection_config,
        use_numeric_pdbs: false,
        partitioning: CostPartitioningMethod::Region,
        ..Default::default()
    };
    let heuristic =
        SaturatedCostPartitioningOnlineHeuristic::new(None, abstractions, vec![], config, task)
            .expect("scp_online heuristic should construct");

    let mut state_registry = StateRegistry::for_task(Arc::new(task));
    let initial = state_registry.get_initial_state();
    let mut eval = EvaluationState::new_with_registry(
        &initial,
        0.0,
        false,
        task as &dyn AbstractNumericTask,
        &state_registry,
    );
    eval.set_is_goal(false);
    heuristic
        .compute_heuristic(&eval)
        .expect("scp_online initial evaluation should succeed")
}

/// Runs A* to completion with the online SCP heuristic and returns the cost.
///
/// The initial state is not enough to test this heuristic. It always rebuilds a
/// cost partitioning, which computes an abstract state id for *every*
/// abstraction. Only later states take the other branch, where ids are computed
/// for the abstractions in `required_lookup_ids` and left absent for the rest,
/// and that is the branch a missing id used to be read as a dead end on.
fn scp_online_search_cost(instance: &str) -> Option<f64> {
    let (task, _scratch) = sailing_task(instance);
    let task = build_restricted_task(&task)
        .expect("sailing-simple restricted task construction must not fail")
        .expect("sailing-simple instances have promotable derived roots")
        .into_task();
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(
        standard_round7_collection_config(1),
    );
    let abstractions = generator
        .generate_collection(&task)
        .expect("scp_online domain abstractions should build");
    let config = ScpOnlineConfig {
        max_time: 100.0,
        max_size: 10_000_000,
        interval: 100_000_000_000,
        table_construction_max_time: 100.0,
        order_optimization_max_time: ORDER_OPTIMIZATION_MAX_TIME,
        collection_config: standard_round7_collection_config(1),
        use_numeric_pdbs: false,
        partitioning: CostPartitioningMethod::Region,
        ..Default::default()
    };
    let heuristic =
        SaturatedCostPartitioningOnlineHeuristic::new(None, abstractions, vec![], config, &task)
            .expect("scp_online heuristic should construct");
    let state_registry = StateRegistry::for_task(Arc::new(&task));
    let mut search = AStarSearch::new(&task, state_registry, Some(Box::new(heuristic)), None, None);
    let result = search.search().expect("scp_online A* search failed");
    result.solution_cost
}

fn scp_online_initial_h(instance: &str) -> f64 {
    scp_online_initial_h_with_config(instance, standard_round7_collection_config(1))
}

fn numeric_var_id_by_name_parts(
    task: &dyn AbstractNumericTask,
    required_parts: &[&str],
) -> Option<usize> {
    task.numeric_variables()
        .iter()
        .position(|var| required_parts.iter().all(|part| var.name().contains(part)))
}

/// Under abstract-operator saturation one per-person abstraction contributes its
/// 10-move route plus save (11), and the other keeps only one distinct save cost
/// because rival-achiever route footprints consume overlapping move residuals.
/// The name says what the assertion says: this is the *shortfall* against the
/// additive 22 that disjoint near boats should give, pinned so a change in
/// either direction is visible.
///
/// This was the most expensive test in the workspace at 91.3s, and none of it
/// was needed: with the time-bounded collection, region-CP table construction
/// took 63s over 6 members to reach 12, and truncating *that* moved the pinned
/// value rather than merely costing precision (measured h = 12 at construction
/// budgets of 30s and 100s, 11 at 10s and below -- pinning 11 would be pinning a
/// truncation artifact that a faster machine crosses back over). The clock-free
/// bound on [`standard_round7_collection_config`] reaches the same 12 from 3
/// members in 5.9s, with no construction deadline in sight.
#[test]
fn sailing_simple_multiboat_scp_falls_short_of_the_additive_optimum() {
    let optimum = optimum_of("prob_2b2p_x");
    let h = scp_online_initial_h("prob_2b2p_x");
    assert!(h <= optimum, "prob_2b2p_x: h={h} must be admissible");
    assert_eq!(h, 12.0);
}

/// Both persons' nearest boat is b0. Per-person abstractions cannot encode that
/// one boat must perform the 10-move transfer between targets; with the current
/// residual saturation they retain 7 units at the initial state.
///
/// Same story as
/// [`sailing_simple_multiboat_scp_falls_short_of_the_additive_optimum`], and
/// sharper because the b1 decoy sits at x=100 and its chain is ~190 layers:
/// construction over the time-bounded collection took 78s to converge on 7 and
/// every budget of 10s or less yielded 6, while the clock-free bound reaches 7
/// from 2 members in 5.9s.
#[test]
fn sailing_simple_assignment_gap() {
    let optimum = optimum_of("prob_2b2p_assign");
    let h = scp_online_initial_h("prob_2b2p_assign");
    assert!(
        h <= optimum,
        "prob_2b2p_assign: h={h} must be admissible against h*={optimum}"
    );
    assert_eq!(h, 7.0);
}

#[test]
fn sailing_simple_ratchet_equilibrium() {
    let (task, _scratch) = sailing_task("prob_2b1p");
    let mut config = standard_round7_collection_config(1);
    config.max_abstraction_size = 1_000;
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config.clone());
    let abstractions = generator
        .generate_collection(&task)
        .expect("complementary collection should build on prob_2b1p");

    let saved_p0_group_counts = abstractions
        .iter()
        .filter_map(|abstraction| {
            if abstraction.metadata.full_goal_task != Some(false) {
                return None;
            }
            let factory_task = abstraction.task_for_factory(&task);
            let near_x = numeric_var_id_by_name_parts(
                factory_task,
                &["difference", "x(?boat_0)", "(b0, p0)"],
            )?;
            let far_x = numeric_var_id_by_name_parts(
                factory_task,
                &["difference", "x(?boat_0)", "(b1, p0)"],
            )?;
            let near_count = abstraction.factory.partitions().partitions(near_x)?.len();
            let far_count = abstraction.factory.partitions().partitions(far_x)?.len();
            (near_count > 1 && far_count > 1).then_some((near_count, far_count))
        })
        .collect::<Vec<_>>();
    // Teleport-trap invariant: any saved(p0) abstraction must refine BOTH
    // boats' x roots (near b0 and far b1). If it refined only one boat while
    // keeping saved(p0), the other boat's save precondition would evaluate
    // `unknown` over its unrefined interval and fire optimistically, letting
    // that boat teleport onto p0 for cost ~1 (see the teleport-theorem
    // analysis). The filter above requires both counts > 1, so a non-empty
    // result proves both roots are refined together.
    //
    // Note: we deliberately do NOT bound far-boat layers relative to near-boat
    // layers here. An earlier "ratchet" hypothesis (seed only the nearest
    // achiever's full chain, let CEGAR lay rival layers on demand) predicted
    // far <= ~near; measurement falsified it — on-demand refinement costs a
    // full CEGAR iteration per layer and collapsed initial h on prob_2_2
    // (72 -> 36) and prob_1_11 (101 -> 59), so upfront full-chain seeding was
    // restored. Under full-chain seeding the far boat is sized by its own
    // distance to the target (here ~182 vs ~22), which is expected, not a bug.
    assert!(
        !saved_p0_group_counts.is_empty(),
        "expected at least one saved(p0) abstraction refining both boat x roots"
    );

    let h = scp_online_initial_h_for_collection(&task, abstractions, config);
    assert!(h >= 10.0, "prob_2b1p: initial h={h} should stay >= 10");
}

#[test]
fn sailing_simple_scp_online_admissible() {
    for instance in ["prob_1b1p_x", "prob_1b2p_x"] {
        let optimum = optimum_of(instance);
        let h = scp_online_initial_h(instance);
        assert!(
            h <= optimum,
            "{instance}: h={h} must be admissible against h*={optimum}"
        );
        assert!(h > 1.0, "{instance}: h={h} should beat blind guidance");
    }
}

/// The online heuristic must not turn a solvable state into a dead end.
///
/// `sailing_simple_scp_online_admissible` above checks the initial state, and
/// that is not enough: the initial state always rebuilds a cost partitioning, so
/// it never reaches the branch that reads a state's abstract ids from a subset
/// of the abstractions. Reading an absent id there as "unreachable" made the
/// search record solvable successors as dead ends, and it reported no solution
/// for a task that has one. Running the search to completion is what covers it.
#[test]
fn sailing_simple_scp_online_solves_rather_than_reporting_dead_ends() {
    for instance in ["prob_1b1p_x", "prob_1b2p_x"] {
        let optimum = optimum_of(instance);
        let found = scp_online_search_cost(instance);
        assert_eq!(
            found,
            Some(optimum),
            "{instance}: scp_online must find the optimum, not report the task unsolvable"
        );
    }
}

#[test]
fn sailing_simple_complementary_collection_keeps_single_goal_solved_abstractions() {
    let (task, _scratch) = sailing_task("prob_1b4p_axes");
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        collection_strategy: CollectionStrategy::Complementary,
        random_seed: Some(1),
        max_abstraction_size: 10_000,
        // Bounded by abstract states rather than by the clock, for the reason
        // spelled out on `standard_round7_collection_config`: at 100_000 nothing
        // but `total_max_time` ever stopped this, so it ran the full 30s. This
        // test wants breadth rather than a heuristic value, so it keeps the
        // default stagnation behaviour instead of stopping at the first
        // duplicate -- that would stop at 3 members, and 8 are asserted.
        // Measured member counts: 11-13 at a 5_000 budget, 21-26 at 10_000,
        // 31-35 at 20_000. 10_000 keeps a factor of two and a half over the
        // assertion for 2.1s instead of 18.9s.
        max_collection_size: 10_000,
        abstraction_generation_max_time: 10.0,
        total_max_time: 30.0,
        ..Default::default()
    };
    let generator = DomainAbstractionCollectionGeneratorMultipleCegar::new(config);
    let abstractions = generator
        .generate_collection(&task)
        .expect("complementary collection should build on prob_1b4p_axes");

    assert!(
        abstractions.len() >= 8,
        "expected at least regression and progression single-goal abstractions per goal, got {}",
        abstractions.len()
    );
    assert!(
        abstractions
            .iter()
            .all(|abstraction| abstraction.metadata.full_goal_task == Some(false)),
        "multi-goal complementary collection must not generate full-goal abstractions"
    );
}
