//! Admissibility of goal-focused abstraction collections along optimal plans.
//!
//! Blind A* finds an optimal plan `s_0, ..., s_n`, and every suffix of an
//! optimal plan is optimal, so `h*(s_i)` is the cost of the suffix from `s_i`.
//! Every heuristic built from a goal-focused collection -- each member on its
//! own and the online saturated cost partitionings over domain, Cartesian, and
//! mixed collections, with label and region partitioning -- must stay finite
//! and at most `h*(s_i)` on every one of those states. Nothing is clamped or skipped: an infinite value on a solvable
//! state is as much a failure as an overestimate.

use std::sync::Arc;

use planforge_sas::numeric_task::{AbstractNumericTask, NumericRootTask};
use planforge_sas::state_registry::{ConcreteState, StateRegistry};
use planforge_search::evaluation::abstractions::cartesian::{
    CartesianAbstraction, CartesianAbstractionCollectionConfig,
    CartesianAbstractionCollectionGenerator, CartesianAbstractionConfig,
    CartesianAbstractionHeuristic,
};
use planforge_search::evaluation::abstractions::collections::portfolio::CollectionStrategy;
use planforge_search::evaluation::abstractions::cost_partitioning::saturated::{
    CostPartitioningMethod, SaturatedCostPartitioningHeuristic, ScpConfig, collection_components,
};
use planforge_search::evaluation::abstractions::domain::cegar::FlawKind;
use planforge_search::evaluation::abstractions::domain::collection_generator::{
    DomainAbstractionCollectionGeneratorMultipleCegar,
    DomainAbstractionCollectionGeneratorMultipleCegarConfig,
};
use planforge_search::evaluation::abstractions::domain::generator::DomainAbstraction;
use planforge_search::evaluation::abstractions::domain::heuristic::DomainAbstractionHeuristic;
use planforge_search::evaluation::evaluator::EvaluationState;
use planforge_search::evaluation::heuristic::Heuristic;
use planforge_search::search::{AStarSearch, SearchEngine, SearchStatus};
use planforge_search::task_restriction::build_restricted_task;

use crate::corpus::{self, Scratch, translate_to_disk};

/// The instance's task, restricted exactly as `--restrict-task` restricts it.
fn restricted_task(folder: &str, stem: &str) -> NumericRootTask {
    let root = corpus::assets().join("numeric-pddl-files").join(folder);
    let scratch = Scratch::new(&format!("goal_focused_admissibility_{folder}_{stem}"));
    let task = translate_to_disk(
        &root.join("domain.pddl"),
        &root.join(format!("{stem}.pddl")),
        &scratch,
    );
    match build_restricted_task(&task).expect("restricted task construction must not fail") {
        Some(restricted) => restricted.into_task(),
        None => task,
    }
}

/// The states of an optimal plan and `h*` of each of them.
fn optimal_plan_states(task: &NumericRootTask) -> (StateRegistry<'_>, Vec<(ConcreteState, f64)>) {
    let mut search = AStarSearch::new(
        task,
        StateRegistry::for_task(Arc::new(task)),
        None,
        None,
        None,
    );
    let result = search.search().expect("blind A* search failed");
    assert!(
        matches!(result.status, SearchStatus::Solved(_)),
        "blind A* must solve the instance: {:?}",
        result.status
    );
    let plan = result.plan.expect("a solved search returns its plan");
    let optimum = result.solution_cost.expect("a solved search has a cost");

    let mut registry = StateRegistry::for_task(Arc::new(task));
    let mut state = registry.get_initial_state();
    let mut prefix_costs = vec![0.0];
    let mut states = vec![state.clone()];
    let (mut successor_values, mut cost_values) = (Vec::new(), Vec::new());
    for operator in &plan {
        let (successor, cost) = registry
            .get_successor_state_with_buffers_and_cost(
                &state,
                operator,
                &mut successor_values,
                &mut cost_values,
            )
            .expect("an optimal plan's operators apply in sequence");
        prefix_costs.push(prefix_costs.last().unwrap() + cost.value());
        states.push(successor.clone());
        state = successor;
    }
    let total = *prefix_costs.last().unwrap();
    assert!(
        (total - optimum).abs() <= 1e-6,
        "replayed plan cost {total} differs from the search's optimum {optimum}"
    );
    let states = states
        .into_iter()
        .zip(prefix_costs)
        .map(|(state, prefix)| (state, total - prefix))
        .collect();
    (registry, states)
}

fn domain_collection(task: &NumericRootTask) -> Vec<DomainAbstraction> {
    DomainAbstractionCollectionGeneratorMultipleCegar::new(
        DomainAbstractionCollectionGeneratorMultipleCegarConfig {
            max_abstraction_size: 300,
            max_collection_size: 3_000,
            abstraction_generation_max_time: f64::INFINITY,
            total_max_time: f64::INFINITY,
            flaw_kind: FlawKind::ExecuteEntirePlan,
            collection_strategy: CollectionStrategy::Complementary,
            variants_per_goal: 4,
            progressive_goal_roots: true,
            random_seed: Some(1),
            ..Default::default()
        },
    )
    .generate_collection(task)
    .expect("the goal-focused domain collection must build")
}

fn cartesian_collection(task: &NumericRootTask) -> Vec<CartesianAbstraction> {
    CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
        abstraction: CartesianAbstractionConfig {
            max_states: 300,
            random_seed: Some(1),
            ..Default::default()
        },
        collection_strategy: CollectionStrategy::Complementary,
        variants_per_goal: 4,
        max_collection_states: 3_000,
        total_max_time: None,
        progressive_goal_roots: true,
        alternate_split_kinds: true,
        single_goal_tasks: true,
    })
    .expect("the Cartesian collection configuration is valid")
    .generate(task)
    .expect("the goal-focused Cartesian collection must build")
}

fn scp_config(partitioning: CostPartitioningMethod) -> ScpConfig {
    ScpConfig {
        max_time: 100.0,
        table_construction_max_time: 100.0,
        random_seed: Some(1),
        partitioning,
        ..Default::default()
    }
}

/// `h(s_i) <= h*(s_i)` and finite on every state of the optimal plan.
fn assert_admissible_along_plan(
    what: &str,
    task: &NumericRootTask,
    registry: &StateRegistry<'_>,
    states: &[(ConcreteState, f64)],
    heuristic: &dyn Heuristic,
) {
    for (step, (state, optimal_cost)) in states.iter().enumerate() {
        let mut eval = EvaluationState::new(state, task as &dyn AbstractNumericTask, registry);
        eval.set_is_goal(step + 1 == states.len());
        let h = heuristic
            .compute_heuristic(&eval)
            .unwrap_or_else(|error| panic!("{what}: evaluation failed at step {step}: {error:?}"));
        assert!(
            h.is_finite(),
            "{what}: h={h} on solvable plan state {step} (h*={optimal_cost})"
        );
        assert!(
            h <= optimal_cost + 1e-6,
            "{what}: h={h} overestimates h*={optimal_cost} at plan state {step}"
        );
    }
}

fn assert_goal_focused_collections_admissible(folder: &str, stem: &str) {
    {
        let task = restricted_task(folder, stem);
        let (registry, states) = optimal_plan_states(&task);
        let instance = format!("{folder}/{stem}");

        let domain = domain_collection(&task);
        let cartesian = cartesian_collection(&task);
        for (index, abstraction) in domain.iter().enumerate() {
            assert_admissible_along_plan(
                &format!("{instance} domain member {index}"),
                &task,
                &registry,
                &states,
                &DomainAbstractionHeuristic::new(None, abstraction.clone()),
            );
        }
        for (index, abstraction) in cartesian.iter().enumerate() {
            assert_admissible_along_plan(
                &format!("{instance} Cartesian member {index}"),
                &task,
                &registry,
                &states,
                &CartesianAbstractionHeuristic::new(None, abstraction.clone()),
            );
        }
        for partitioning in [
            CostPartitioningMethod::Label,
            CostPartitioningMethod::Region,
        ] {
            let heuristics = [
                (
                    "domain",
                    SaturatedCostPartitioningHeuristic::new(
                        None,
                        collection_components("scp", domain.clone(), Vec::new(), vec![]),
                        scp_config(partitioning),
                        &task,
                        Default::default(),
                    ),
                ),
                (
                    "Cartesian",
                    SaturatedCostPartitioningHeuristic::new(
                        None,
                        collection_components("scp", vec![], cartesian.clone(), vec![]),
                        scp_config(partitioning),
                        &task,
                        Default::default(),
                    ),
                ),
                (
                    "mixed",
                    SaturatedCostPartitioningHeuristic::new(
                        None,
                        collection_components("scp", domain.clone(), cartesian.clone(), vec![]),
                        scp_config(partitioning),
                        &task,
                        Default::default(),
                    ),
                ),
            ];
            for (collection, heuristic) in heuristics {
                let heuristic = heuristic.unwrap_or_else(|error| {
                    panic!("{instance} {collection} SCP must construct: {error:?}")
                });
                assert_admissible_along_plan(
                    &format!("{instance} {collection} SCP {partitioning:?}"),
                    &task,
                    &registry,
                    &states,
                    &heuristic,
                );
            }
        }
    }
}

/// One test per instance. Every instance is one blind A* solves in a few
/// milliseconds: measured with `target/release/planforge --restrict-task
/// --search 'astar(blind())'`,
///
/// | instance | expanded | search time |
/// |---|---|---|
/// | `sailing-simple/prob_1b1p_x` | 363 | 0.001s |
/// | `sailing-simple/prob_1b1p_diag` | 380 | 0.001s |
/// | `sailing-simple/prob_1b2p_x` | 1_070 | 0.002s |
/// | `sailing-simple/prob_1b2p_diag` | 2_125 | 0.004s |
/// | `sailing-simple/prob_2b1p` | 21_843 | 0.058s |
/// | `counters-sym/problem_2` | 16 | 0.0002s |
/// | `fn-counters-small_instances/problem_2` | 2 | 0.0001s |
/// | `plant-watering/prob_4_1_1` | 227 | 0.001s |
///
/// `h*` comes from that search with no time limit, so it is bounded by the
/// instance's state space alone. What the tests spend their time on is
/// building the collections and the cost partitionings in an unoptimized
/// build, which the member budgets (300 abstract states, 3_000 per
/// collection) keep to seconds.
macro_rules! admissibility_tests {
    ($($name:ident: $folder:literal / $stem:literal,)*) => {
        $(
            #[test]
            fn $name() {
                assert_goal_focused_collections_admissible($folder, $stem);
            }
        )*
    };
}

admissibility_tests! {
    sailing_simple_1b1p_x: "sailing-simple" / "prob_1b1p_x",
    sailing_simple_1b1p_diag: "sailing-simple" / "prob_1b1p_diag",
    sailing_simple_1b2p_x: "sailing-simple" / "prob_1b2p_x",
    sailing_simple_1b2p_diag: "sailing-simple" / "prob_1b2p_diag",
    sailing_simple_2b1p: "sailing-simple" / "prob_2b1p",
    counters_sym: "counters-sym" / "problem_2",
    fn_counters_small: "fn-counters-small_instances" / "problem_2",
    plant_watering: "plant-watering" / "prob_4_1_1",
}
