use crate::evaluation::abstractions::common::abstraction::CegarAbstraction;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitFact, ExplicitValueIndex, INF_VALUE, NEG_INF_VALUE,
    NumericRootTask, NumericType, NumericValue, VariableIndex, ZERO_VALUE,
};
use planforge_translate::translate_to_sas_to_path_fast;

use super::*;
use crate::evaluation::abstractions::cartesian::{
    CartesianAbstraction, CartesianAbstractionCollectionConfig,
    CartesianAbstractionCollectionGenerator, CartesianAbstractionConfig,
};
use crate::evaluation::abstractions::common::abstraction::{
    AbstractionMetadata, AbstractionUse, CollectionRecord,
};
use crate::evaluation::abstractions::common::cegar::CegarStopReason;
use crate::evaluation::abstractions::common::cegar::SplitKind;
use crate::evaluation::abstractions::common::cegar::refine::CegarOutcome;
use crate::evaluation::abstractions::common::task::SingleGoalTask;
use crate::evaluation::abstractions::domain::factory::DomainAbstractionFactory;
use crate::evaluation::abstractions::domain::partitions::NumericPartitions;
use crate::task_restriction::build_restricted_task;
use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::utils::interval::Interval;

#[test]
fn interval_stolen_costs_keep_disjoint_infinite_tails() {
    let region = |lower, upper, lower_closed, upper_closed| {
        CartesianSet::with_all_props_constrained(
            Vec::new(),
            vec![Interval::new(lower, upper, lower_closed, upper_closed)],
        )
    };
    let left_tail = region(NEG_INF_VALUE, ZERO_VALUE, false, true);
    let right_tail = region(ZERO_VALUE, INF_VALUE, false, false);
    let overlapping_tail = region(NumericValue::new(-1.0), INF_VALUE, true, false);

    assert_eq!(
        pairwise_interval_stolen_cost(&[&left_tail], 1.0, &[&right_tail], 1.0, 1.0),
        0.0
    );
    assert_eq!(
        pairwise_interval_stolen_cost(&[&left_tail], 1.0, &[&overlapping_tail], 1.0, 1.0),
        1.0
    );
}

#[test]
#[ignore = "oracle collection report over translated sailing benchmark"]
fn sailing_goal_focused_cartesian_collection_reaches_optimum_with_interval_scp() {
    let task = translated_sailing_2_2_task();
    let restricted_task = build_restricted_task(&task)
        .expect("sailing task should support restricted task")
        .expect("sailing task should have promotable roots")
        .into_task();
    let abstractions =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            member: CartesianAbstractionConfig {
                cegar: CegarOptions {
                    max_states: 1_000,
                    max_time: Some(Duration::from_secs(5)),
                    random_seed: Some(1),
                    debug: false,
                    ..CartesianAbstractionConfig::default().cegar
                },
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::GoalFocused,
            variants_per_goal: 4,
            max_collection_size: 10_000,
            total_max_time: Some(Duration::from_secs(15)),
            progress_states: false,
            alternate_split_kinds: Some(true),
            split_kind: None,
            random_seed: Some(1),
            single_goal_tasks: true,
            ..Default::default()
        })
        .expect("valid oracle Cartesian collection config")
        .generate(&restricted_task)
        .expect("failed to build oracle Cartesian collection");

    assert_eq!(abstractions.len(), 8);
    assert_eq!(
        abstractions
            .iter()
            .map(CegarAbstraction::num_states)
            .sum::<usize>(),
        8_000
    );
    for goal_id in 0..2 {
        let mut modes = abstractions
            .iter()
            .filter(|abstraction| {
                abstraction
                    .metadata
                    .collection_member
                    .map(|member| member.goal_id)
                    == Some(goal_id)
            })
            .map(|abstraction| {
                (
                    abstraction.metadata.split_kind,
                    abstraction.metadata.split_selection_rank,
                )
            })
            .collect::<Vec<_>>();
        modes.sort_by_key(|(direction, rank)| {
            (
                match direction {
                    SplitKind::Concrete => 0,
                    SplitKind::Wanted => 1,
                },
                *rank,
            )
        });
        assert_eq!(
            modes,
            vec![
                (SplitKind::Concrete, Some(0)),
                (SplitKind::Concrete, Some(1)),
                (SplitKind::Wanted, Some(0)),
                (SplitKind::Wanted, Some(1)),
            ]
        );
    }

    let standalone_max = abstractions
        .iter()
        .map(|abstraction| {
            abstraction.distance_table.distances[abstraction.distance_table.initial_state]
        })
        .fold(0.0f64, f64::max);
    let abstraction_h = abstractions
        .iter()
        .map(|abstraction| {
            abstraction.distance_table.distances[abstraction.distance_table.initial_state]
        })
        .collect::<Vec<_>>();
    let components = abstractions
        .iter()
        .cloned()
        .map(AbstractionComponent::from)
        .collect::<Vec<_>>();
    let goal_cover_order = goal_cover_order(
        &(0..abstractions.len()).collect::<Vec<_>>(),
        &components,
        &abstraction_h,
        true,
        GoalCoverOrderVariant::default(),
    )
    .expect("the goal-focused sailing collection must have a goal-cover order");
    assert_eq!(&goal_cover_order[..3], &[5, 0, 1]);
    let regional_h = initial_cartesian_scp_value(&restricted_task, abstractions, true);

    println!(
        "SAILING_PERFECT_COLLECTION standalone_max={standalone_max} regional_scp={regional_h}"
    );
    assert_eq!(standalone_max, 40.0);
    assert_eq!(regional_h, 76.0);
}

fn initial_cartesian_scp_value(
    task: &dyn AbstractNumericTask,
    abstractions: Vec<CartesianAbstraction>,
    partitioning: bool,
) -> f64 {
    let initial_prop = task.get_initial_propositional_state_values();
    let initial_numeric = task.get_initial_numeric_state_values();
    let abstract_state_ids = abstractions
        .iter()
        .map(|abstraction| {
            Some(
                abstraction
                    .hierarchy
                    .map_state(initial_prop, initial_numeric)
                    .expect("failed to map sailing initial state"),
            )
        })
        .collect::<Vec<_>>();
    let config = ScpConfig {
        table_construction_max_time: 30.0,
        saturator: Saturator::All,
        random_seed: Some(1),
        partitioning: if partitioning {
            CostPartitioningMethod::Region
        } else {
            CostPartitioningMethod::Label
        },
        ..Default::default()
    };
    let heuristic = SaturatedCostPartitioningHeuristic::new(
        None,
        collection_components(vec![], abstractions, vec![]),
        config,
        task,
        Default::default(),
    )
    .expect("failed to construct oracle SCP heuristic");
    let mut state = heuristic.state.borrow_mut();
    let mut partitions = heuristic
        .build_cost_partitionings(task, &mut state, &abstract_state_ids)
        .expect("oracle SCP construction failed");
    assert_eq!(partitions.len(), 1);
    partitions
        .pop()
        .expect("one oracle SCP partition")
        .compute_heuristic(&abstract_state_ids)
}

#[test]
#[ignore = "diagnostic full-task handcrafted sailing regional SCP report"]
fn sailing_handcrafted_abstractions_interval_scp_initial_h_report() {
    let task = translated_sailing_2_2_task();
    let restricted_task = build_restricted_task(&task)
        .expect("sailing task should support restricted task")
        .expect("sailing task should have promotable roots")
        .into_task();
    let transformed_task = &restricted_task;
    let specs = handcrafted_full_task_specs(transformed_task);
    assert_eq!(specs.len(), 4);

    let mut abstractions = Vec::new();
    for (index, spec) in specs.iter().enumerate() {
        let single_goal_task = SingleGoalTask::new(transformed_task, spec.goal);
        let mut abstraction = build_handcrafted_abstraction(&single_goal_task, spec)
            .unwrap_or_else(|error| panic!("failed to build {}: {error:#}", spec.name));
        abstraction.metadata.collection = Some(CollectionRecord {
            iteration: index + 1,
            full_task: false,
        });
        let states = abstraction.distance_table.distances.len();
        assert!(
            states <= 10_000,
            "{} has {states} states, expected at most 10000",
            spec.name
        );
        abstractions.push(abstraction);
    }

    let config = ScpConfig {
        max_time: 300.0,
        table_construction_max_time: 30.0,
        diversify: false,
        samples: 1_000,
        max_orders: usize::MAX,
        order_generator: OrderGenerator::Greedy,
        initial_order_generation_max_time: 0.0,
        saturator: Saturator::All,
        random_seed: Some(1),
        partitioning: CostPartitioningMethod::Region,
        ..Default::default()
    };

    let heuristic = SaturatedCostPartitioningHeuristic::new(
        None,
        collection_components(abstractions, Vec::new(), vec![]),
        config,
        transformed_task,
        Default::default(),
    )
    .expect("failed to construct SCP heuristic");
    let abstract_state_ids = initial_abstract_state_ids(&heuristic, transformed_task);
    {
        let mut state = heuristic.state.borrow_mut();
        let mut partitions = heuristic
            .build_cost_partitionings(transformed_task, &mut state, &abstract_state_ids)
            .expect("initial SCP construction failed");
        assert_eq!(partitions.len(), 1);
        SaturatedCostPartitioningHeuristic::retain_cp(&mut state, partitions.pop().unwrap());
    }

    let state = heuristic.state.borrow();
    let initial_h = SaturatedCostPartitioningHeuristic::compute_max_h(&state, &abstract_state_ids);
    let components = heuristic.components.borrow();

    println!("HANDCRAFTED_FULL_TASK_INITIAL_H {initial_h}");
    let mut contributions = vec![0.0; specs.len()];
    for cp in &state.cp_heuristics {
        for table in &cp.lookup_tables {
            let contribution = current_h_for_distances(
                table.abstraction_id,
                &table.distances,
                &abstract_state_ids,
            );
            if let Some(total) = contributions.get_mut(table.abstraction_id) {
                *total += contribution;
            }
        }
    }

    for (index, (spec, component)) in specs.iter().zip(components.iter()).enumerate() {
        let abstraction = component
            .as_domain()
            .expect("handcrafted diagnostic uses domain components");
        let abstraction_h = current_h_for_distances(
            index,
            &abstraction.distance_table.distances,
            &abstract_state_ids,
        );
        println!(
            "HANDCRAFTED_FULL_TASK_ABS index={index} name={} abstraction_h={abstraction_h} scp_contribution={} states={} abstract_ops={} views={}",
            spec.name,
            contributions[index],
            abstraction.distance_table.distances.len(),
            abstraction.abstract_operators.len(),
            partition_report(transformed_task, abstraction, &spec.view_ids),
        );
    }
    println!("HANDCRAFTED_FULL_TASK_CONTRIBUTIONS {contributions:?}");

    assert!(
        initial_h.is_finite(),
        "initial h must be finite, got {initial_h}"
    );
    assert!(initial_h > 0.0, "initial h should be positive");
    assert!(
        initial_h <= 76.0,
        "full-task diagnostic should not exceed the known optimal cost 76, got {initial_h}"
    );
}

#[derive(Debug, Clone)]
struct HandcraftedSpec {
    name: String,
    goal: ExplicitFact,
    view_ids: Vec<usize>,
    seed_splits: Vec<NumericSeed>,
}

/// A numeric split the handcrafted abstractions start from.
#[derive(Debug, Clone, PartialEq)]
struct NumericSeed {
    numeric_var_id: VariableIndex,
    value: NumericValue,
    include_in_lower: bool,
}

fn handcrafted_full_task_specs(task: &dyn AbstractNumericTask) -> Vec<HandcraftedSpec> {
    [
        ("p1-u", "p1", ViewKind::Sum),
        ("p1-v", "p1", ViewKind::Difference),
        ("p0-u", "p0", ViewKind::Sum),
        ("p0-v", "p0", ViewKind::Difference),
    ]
    .into_iter()
    .map(|(name, person, view_kind)| {
        let view_ids = ["b0", "b1"]
            .into_iter()
            .map(|boat| find_sailing_view(task, boat, person, view_kind))
            .collect::<Vec<_>>();
        let seed_splits = route_seed_splits(task, &view_ids);
        HandcraftedSpec {
            name: name.to_string(),
            goal: find_saved_goal_fact(task, person),
            view_ids,
            seed_splits,
        }
    })
    .collect()
}

fn build_handcrafted_abstraction(
    transformed_task: &dyn AbstractNumericTask,
    spec: &HandcraftedSpec,
) -> anyhow::Result<DomainAbstraction> {
    let mut partitions = NumericPartitions::trivial(transformed_task);
    for seed in &spec.seed_splits {
        partitions.split_at(
            seed.numeric_var_id,
            seed.value,
            seed.include_in_lower,
            false,
        );
    }

    let goal_vars = transformed_task
        .goals()
        .iter()
        .map(ExplicitFact::var)
        .collect::<HashSet<_>>();
    let domain_mapping = (0..transformed_task.get_num_variables())
        .map(|var_id| {
            let domain_size = transformed_task
                .get_variable_domain_size(VariableIndex::from_usize(var_id))
                .expect("valid transformed prop var id");
            if goal_vars.contains(&var_id) {
                (0..domain_size)
                    .map(ExplicitValueIndex::from_usize)
                    .collect::<Vec<_>>()
            } else {
                vec![ExplicitValueIndex::new(0); domain_size]
            }
        })
        .collect::<Vec<_>>();
    let domain_sizes = domain_mapping
        .iter()
        .map(|mapping| {
            mapping
                .iter()
                .copied()
                .max()
                .map_or(0, |value| value.index() + 1)
        })
        .collect::<Vec<_>>();
    let factory =
        DomainAbstractionFactory::new(transformed_task, domain_mapping, domain_sizes, partitions)?;
    let abstract_operators = factory.abstract_operators(transformed_task, false, Deadline::NONE)?;
    let abstract_source_sets =
        factory.build_abstract_source_sets(transformed_task, &abstract_operators)?;
    let distance_table =
        factory.build_distance_table(transformed_task, &abstract_operators, Deadline::NONE)?;
    let relevant_operator_ids =
        factory.relevant_operator_ids(transformed_task, &abstract_operators);
    Ok(DomainAbstraction {
        factory,
        distance_table,
        combine_labels: false,
        relevant_operator_ids,
        abstract_operators,
        transition_data_retained: true,
        abstract_source_sets,
        interval_transition_system: RefCell::new(None),
        metadata: AbstractionMetadata {
            abstraction_use: AbstractionUse::CollectionMember,
            ..AbstractionMetadata::standalone(
                CegarOutcome {
                    stop_reason: CegarStopReason::SizeLimit,
                    concrete_plan: None,
                    pending_flaw: None,
                    refinements: 0,
                },
                SplitKind::Concrete,
                None,
            )
        },
    })
}

fn initial_abstract_state_ids(
    heuristic: &SaturatedCostPartitioningHeuristic<'_>,
    task: &dyn AbstractNumericTask,
) -> Vec<Option<usize>> {
    let values = ExplicitState {
        propositional: task.get_initial_propositional_state_values().to_vec(),
        numeric: task.get_initial_numeric_state_values().to_vec(),
    };
    heuristic
        .components
        .borrow()
        .iter()
        .map(|component| {
            component
                .abstract_state_id(&values)
                .expect("failed to map the initial state")
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
enum ViewKind {
    Sum,
    Difference,
}

fn find_sailing_view(
    task: &dyn AbstractNumericTask,
    boat: &str,
    person: &str,
    view_kind: ViewKind,
) -> usize {
    let tuple = format!("({boat}, {boat}, {person})");
    let candidates = task
        .numeric_variables()
        .iter()
        .enumerate()
        .filter(|(_, variable)| variable.get_type() == &NumericType::Regular)
        .filter(|(_, variable)| {
            let name = variable.name();
            name.contains(&tuple)
                && !name.contains("25.0")
                && match view_kind {
                    ViewKind::Sum => name.contains("derived!sum_PNE x"),
                    ViewKind::Difference => name.contains("derived!difference_PNE y"),
                }
        })
        .map(|(id, _)| id)
        .collect::<Vec<_>>();
    assert_eq!(
        candidates.len(),
        1,
        "expected one {view_kind:?} view for {boat}/{person}, got {candidates:?}"
    );
    candidates[0]
}

fn find_saved_goal_fact(task: &dyn AbstractNumericTask, person: &str) -> ExplicitFact {
    let suffix = format!(" {person}");
    let mut candidates = task
        .get_operators()
        .iter()
        .filter(|operator| operator.name().starts_with("save_person "))
        .filter(|operator| operator.name().ends_with(&suffix))
        .flat_map(|operator| operator.effects().iter())
        .filter(|effect| effect.conditions().is_empty())
        .map(|effect| ExplicitFact::propositional(effect.var_id().index(), effect.value().index()))
        .collect::<Vec<_>>();
    candidates.sort();
    candidates.dedup();
    assert_eq!(
        candidates.len(),
        1,
        "expected one saved fact from save_person operators for {person}, got {candidates:?}"
    );
    candidates[0]
}

fn route_seed_splits(task: &dyn AbstractNumericTask, view_ids: &[usize]) -> Vec<NumericSeed> {
    let initial = task.get_initial_numeric_state_values();
    let mut seeds = Vec::new();
    for &view_id in view_ids {
        add_split(
            &mut seeds,
            VariableIndex::from_usize(view_id),
            initial[view_id],
            false,
        );
        add_split(
            &mut seeds,
            VariableIndex::from_usize(view_id),
            NumericValue::new(0.0),
            true,
        );
        add_split(
            &mut seeds,
            VariableIndex::from_usize(view_id),
            NumericValue::new(25.0),
            true,
        );
        add_route_grid_values(
            &mut seeds,
            VariableIndex::from_usize(view_id),
            initial[view_id],
            NumericValue::new(25.0),
            3.0,
        );
    }
    seeds.sort_by_key(seed_description);
    seeds.dedup();
    seeds
}

fn add_route_grid_values(
    seeds: &mut Vec<NumericSeed>,
    numeric_var_id: VariableIndex,
    start: NumericValue,
    end: NumericValue,
    step: f64,
) {
    assert!(start.is_finite() && end.is_finite() && step.is_finite() && step > 0.0);
    let direction = if start <= end { 1.0 } else { -1.0 };
    let mut value = start;
    while (end.value() - value.value()) * direction > step {
        value = NumericValue::new(value.value() + direction * step);
        add_split(seeds, numeric_var_id, value, true);
    }
}

fn add_split(
    seeds: &mut Vec<NumericSeed>,
    numeric_var_id: VariableIndex,
    value: NumericValue,
    include_in_lower: bool,
) {
    seeds.push(NumericSeed {
        numeric_var_id,
        value,
        include_in_lower,
    });
}

fn partition_report(
    task: &dyn AbstractNumericTask,
    abstraction: &DomainAbstraction,
    view_ids: &[usize],
) -> String {
    view_ids
        .iter()
        .map(|&view_id| {
            let name = task.numeric_variables()[view_id].name();
            let num_parts = abstraction
                .factory
                .partitions()
                .partitions(VariableIndex::from_usize(view_id))
                .len();
            format!("n{view_id}:{name} parts={num_parts}")
        })
        .collect::<Vec<_>>()
        .join(" || ")
}

fn seed_description(seed: &NumericSeed) -> String {
    format!(
        "n{}{}{}",
        seed.numeric_var_id,
        if seed.include_in_lower { "<=" } else { "<" },
        seed.value
    )
}

fn translated_sailing_2_2_task() -> NumericRootTask {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let domain = root.join("others/sailing/domain.pddl");
    let problem = root.join("others/sailing/prob_2_2_1229.pddl");
    assert!(domain.is_file(), "missing {}", domain.display());
    assert!(problem.is_file(), "missing {}", problem.display());
    let temp_dir = unique_temp_dir("sailing_handcrafted_full_task_scp")
        .expect("failed to create sailing diagnostic temp dir");
    let preprocessed = temp_dir.join("output");
    translate_to_sas_to_path_fast(
        domain.to_str().expect("non-utf8 sailing domain path"),
        problem.to_str().expect("non-utf8 sailing problem path"),
        &preprocessed,
    )
    .expect("sailing translation failed");
    NumericRootTask::from_file(&preprocessed)
}

fn unique_temp_dir(prefix: &str) -> std::io::Result<PathBuf> {
    let base = std::env::temp_dir().join("numeric_planneRS");
    std::fs::create_dir_all(&base)?;
    let dir = base.join(format!(
        "{prefix}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    std::fs::create_dir(&dir)?;
    Ok(dir)
}
