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
use planforge_search::evaluation::evaluator::{EvaluationState, Evaluator};
use planforge_search::evaluation::heuristic::Heuristic;
use planforge_search::search::{AStarSearch, SearchEngine, SearchStatus};
use planforge_search::task_restriction::build_restricted_task;

use crate::corpus::{
    self, Scratch, assert_fixture_set_is_pinned, blind_astar_cost, problem_file_names,
    translate_to_disk,
};

/// Hand-derived optima from the corpus README, machine-verified by
/// [`blind_astar_reproduces_every_sailing_simple_optimum`].
const SAILING_SIMPLE_OPTIMA: &[(&str, f64)] = &[
    ("prob_1b1p_x", 11.0),
    ("prob_1b1p_diag", 11.0),
    ("prob_1b1p_far", 101.0),
    ("prob_1b2p_x", 17.0),
    ("prob_1b2p_diag", 22.0),
    ("prob_1b4p_axes", 74.0),
    ("prob_2b1p", 11.0),
    ("prob_2b2p_x", 22.0),
    ("prob_2b2p_assign", 17.0),
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
        .find(|(name, _)| *name == instance)
        .unwrap_or_else(|| panic!("{instance} is not in SAILING_SIMPLE_OPTIMA"))
        .1
}

/// Blind A* is optimal, so this both verifies the hand-derived optima and pins
/// that `build_restricted_task` is plan-preserving: the restriction promotes
/// derived condition roots, and a restriction that changed reachability would
/// change one of these costs.
#[test]
fn blind_astar_reproduces_every_sailing_simple_optimum() {
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
        .map(|(name, _)| *name)
        .collect();
    assert_fixture_set_is_pinned("sailing-simple", &discovered, &pinned);

    for &(instance, expected) in SAILING_SIMPLE_OPTIMA {
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
    let mut search = AStarSearch::new(
        Arc::new(task),
        state_registry,
        Some(Box::new(heuristic)),
        None,
        None,
    );
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

fn standard_round7_collection_config(
    seed: u64,
) -> DomainAbstractionCollectionGeneratorMultipleCegarConfig {
    DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        max_abstraction_size: 10_000,
        max_collection_size: 1_000_000,
        abstraction_generation_max_time: 5.0,
        total_max_time: 15.0,
        stagnation_limit: 30.0,
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
        .evaluate_state(&mut eval)
        .expect("scp_online initial evaluation should succeed")
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

#[test]
fn sailing_simple_complementary_collection_keeps_single_goal_solved_abstractions() {
    let (task, _scratch) = sailing_task("prob_1b4p_axes");
    let config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
        collection_strategy: CollectionStrategy::Complementary,
        random_seed: Some(1),
        max_abstraction_size: 10_000,
        max_collection_size: 100_000,
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
