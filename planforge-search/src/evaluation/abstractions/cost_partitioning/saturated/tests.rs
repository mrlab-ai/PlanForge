use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

use super::config::{CostPartitioningMethod, OrderGenerator, Saturator, ScpConfig};
use super::diverse_orders::retain_if_improving_some_sample;
use super::goal_cover_orders::{
    GoalCoverOrderVariant, compact_goal_cover_variants, goal_cover_order,
};
use super::lookup::{
    CostPartitioningHeuristic, LookupTable, compute_lookup_mask, compute_max_h,
    current_h_for_distances,
};
use super::orders::max_heuristic_greedy_order;
use super::partition::is_zero_at_evaluated_state;
use super::sampling::random_walk_length;
use super::scoring::{pairwise_region_stolen_cost, region_stolen_costs};
use std::collections::{HashMap, HashSet};

use super::*;
use crate::evaluation::abstractions::common::abstraction::CegarAbstraction;
use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use std::time::Duration;

use crate::evaluation::abstractions::collections::strategy::CollectionStrategy;

use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, Operator, OperatorCost, VariableIndex,
};

use planforge_sas::state_registry::StateRegistry;

use crate::evaluation::abstractions::cartesian::{
    CartesianAbstraction, CartesianAbstractionCollectionConfig,
    CartesianAbstractionCollectionGenerator, CartesianAbstractionConfig,
    CartesianAbstractionGenerator,
};
use crate::evaluation::abstractions::domain::abstraction::DomainAbstraction;

use crate::evaluation::abstractions::domain::config::DomainAbstractionConfig;

use crate::evaluation::abstractions::domain::generator::DomainAbstractionGenerator;

use crate::evaluation::abstractions::pdb::pattern_database::PatternDatabase;

use crate::evaluation::abstractions::pdb::projected_task::{Pattern, ProjectedTask};

pub(super) fn binary_variable(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        None,
        ExplicitValueIndex::new(1),
    )
}

pub(super) fn independent_goals_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![binary_variable("p"), binary_variable("q")],
        numeric_variables: vec![],
        goals: vec![
            ExplicitFact::propositional(0, 1),
            ExplicitFact::propositional(1, 1),
        ],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![
            Operator::new(
                "set-p".to_string(),
                vec![],
                vec![Effect::new(
                    vec![],
                    VariableIndex::new(0),
                    Some(ExplicitValueIndex::new(0)),
                    ExplicitValueIndex::new(1),
                )],
                vec![],
                OperatorCost::new(2),
            ),
            Operator::new(
                "set-q".to_string(),
                vec![],
                vec![Effect::new(
                    vec![],
                    VariableIndex::new(1),
                    Some(ExplicitValueIndex::new(0)),
                    ExplicitValueIndex::new(1),
                )],
                vec![],
                OperatorCost::new(3),
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

/// A Cartesian abstraction of `task` with its source sets.
pub(super) fn cartesian_abstraction(task: &NumericRootTask) -> CartesianAbstraction {
    let mut abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        cegar: CegarOptions {
            max_states: 16,
            max_time: None.into(),
            combine_labels: false,
            random_seed: None,
            debug: false,
            ..CartesianAbstractionConfig::default().cegar
        },
    })
    .unwrap()
    .generate(task)
    .unwrap();
    abstraction
        .prepare_regional_saturation(&RegionSemantics::new(task).unwrap(), Deadline::NONE)
        .unwrap();
    abstraction
}

pub(super) fn scp_config(saturator: Saturator, abstract_operator: bool) -> ScpConfig {
    ScpConfig {
        max_time: Some(Duration::from_secs(10)).into(),
        table_construction_max_time: Some(Duration::from_secs(10)).into(),
        saturator,
        random_seed: 1,
        partitioning: if abstract_operator {
            CostPartitioningMethod::Region
        } else {
            CostPartitioningMethod::Label
        },
        ..ScpConfig::default()
    }
}

pub(super) fn evaluate_initial(
    task: &NumericRootTask,
    heuristic: &dyn Heuristic,
) -> Result<f64, EvaluationError> {
    let mut registry = StateRegistry::for_task(std::sync::Arc::new(task));
    let initial_state = registry.get_initial_state();
    let eval_state = EvaluationState::new(&initial_state, task, &registry);
    heuristic.compute_heuristic(&eval_state)
}

/// Offline diversified SCP over a goal-focused Cartesian collection of `task`,
/// with the goals its abstractions specialize in.
pub(super) fn goal_cover_scp<'task>(
    task: &'task std::sync::Arc<NumericRootTask>,
    max_orders: impl FnOnce(usize) -> usize,
) -> (SaturatedCostPartitioningHeuristic<'task>, HashSet<usize>) {
    let mut abstractions =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            member: CartesianAbstractionConfig {
                cegar: CegarOptions {
                    max_states: 16,
                    max_time: None.into(),
                    combine_labels: false,
                    random_seed: Some(1),
                    debug: false,
                    ..CartesianAbstractionConfig::default().cegar
                },
            },
            collection_strategy: CollectionStrategy::GoalFocused,
            variants_per_goal: 4,
            max_collection_size: 128,
            total_max_time: None.into(),
            progress_states: true,
            alternate_split_kinds: Some(true),
            split_kind: None,
            single_goal_tasks: true,
            ..Default::default()
        })
        .unwrap()
        .generate(&**task)
        .unwrap();
    for abstraction in &mut abstractions {
        abstraction
            .prepare_regional_saturation(&RegionSemantics::new(&**task).unwrap(), Deadline::NONE)
            .unwrap();
    }
    let expected_goals = abstractions
        .iter()
        .filter_map(|abstraction| {
            abstraction
                .metadata()
                .collection_membership()
                .map(|member| member.goal_id)
        })
        .collect::<HashSet<_>>();
    let components = abstractions
        .into_iter()
        .map(AbstractionComponent::from)
        .collect();
    let mut config = scp_config(Saturator::All, true);
    config.diversify = true;
    config.samples = 16;
    config.max_orders = max_orders(expected_goals.len());
    config.initial_order_generation_max_time = Some(Duration::from_secs(10)).into();
    let heuristic = SaturatedCostPartitioningHeuristic::new(
        None,
        components,
        config,
        &**task,
        ScpBuildOptions {
            sampling_task: Some(task.clone()),
            ..Default::default()
        },
    )
    .unwrap();
    (heuristic, expected_goals)
}

/// The cost partitionings `heuristic` retained at its first evaluation.
pub(super) fn retained_cost_partitionings<'a>(
    heuristic: &'a SaturatedCostPartitioningHeuristic<'_>,
) -> std::cell::Ref<'a, [CostPartitioningHeuristic]> {
    std::cell::Ref::map(heuristic.state.borrow(), |state| match &state.phase {
        ScpPhase::Built(retained) => retained.cp_heuristics.as_slice(),
        ScpPhase::Unbuilt(_) => panic!("the heuristic has not evaluated a state"),
    })
}

/// The cost partitionings `heuristic` builds at the first evaluated state,
/// whose abstract states are `abstract_state_ids`.
pub(super) fn build_cost_partitionings(
    heuristic: &SaturatedCostPartitioningHeuristic<'_>,
    abstract_state_ids: &[Option<usize>],
) -> Vec<CostPartitioningHeuristic> {
    let mut state = heuristic.state.borrow_mut();
    let ScpState { components, phase } = &mut *state;
    let ScpPhase::Unbuilt(selection) = phase else {
        panic!("the heuristic built its cost partitionings already");
    };
    heuristic
        .build_cost_partitionings(components, selection, abstract_state_ids)
        .expect("SCP construction failed")
}

pub(super) fn retained_goal_cover_orders_per_goal(
    heuristic: &SaturatedCostPartitioningHeuristic<'_>,
) -> HashMap<usize, usize> {
    let mut retained_per_goal = HashMap::<usize, usize>::new();
    for goal_id in retained_cost_partitionings(heuristic)
        .iter()
        .filter_map(|cp| cp.goal_cover_goal_id)
    {
        *retained_per_goal.entry(goal_id).or_default() += 1;
    }
    retained_per_goal
}

mod diverse_orders;
mod goal_cover_orders;
mod heuristic;
mod sailing;
mod sampling;
mod scoring;
