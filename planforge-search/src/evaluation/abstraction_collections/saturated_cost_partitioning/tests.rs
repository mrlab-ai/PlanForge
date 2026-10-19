use super::diverse_orders::*;

use super::orders::*;

use super::sampling::*;

use super::*;

use crate::evaluation::abstraction_collections::portfolio::CollectionStrategy;

use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, Operator, OperatorCost, VariableIndex,
};

use planforge_sas::state_registry::StateRegistry;

use crate::evaluation::cartesian_abstractions::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionCollectionGenerator,
    CartesianAbstractionConfig, CartesianAbstractionGenerator,
};

use crate::evaluation::domain_abstractions::cegar::CegarConfig;

use crate::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstractionGenerator;

use crate::evaluation::pattern_databases::pattern_database::PatternDatabase;

use crate::evaluation::pattern_databases::projected_task::{Pattern, ProjectedTask};

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

pub(super) fn cartesian_abstraction(task: &NumericRootTask) -> CartesianAbstraction {
    CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        max_states: 16,
        max_time: None,
        combine_labels: false,
        compute_source_sets: true,
        random_seed: None,
        debug: false,
        ..Default::default()
    })
    .unwrap()
    .generate(task)
    .unwrap()
}

pub(super) fn scp_config(saturator: Saturator, abstract_operator: bool) -> ScpConfig {
    ScpConfig {
        max_time: 10.0,
        table_construction_max_time: 10.0,
        interval: usize::MAX,
        order_optimization_max_time: 0.0,
        saturator,
        random_seed: Some(1),
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

/// Offline diversified SCP over a complementary Cartesian collection of `task`,
/// with the goals its abstractions specialize in.
pub(super) fn goal_cover_scp<'task>(
    task: &'task std::sync::Arc<NumericRootTask>,
    max_orders: impl FnOnce(usize) -> usize,
) -> (SaturatedCostPartitioningHeuristic<'task>, HashSet<usize>) {
    let abstractions =
        CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
            abstraction: CartesianAbstractionConfig {
                max_states: 16,
                max_time: None,
                combine_labels: false,
                compute_source_sets: true,
                random_seed: Some(1),
                debug: false,
                ..Default::default()
            },
            collection_strategy: CollectionStrategy::Complementary,
            variants_per_goal: 4,
            max_collection_states: 128,
            total_max_time: None,
            progressive_goal_roots: true,
            alternate_split_kinds: true,
            single_goal_tasks: true,
        })
        .unwrap()
        .generate(&**task)
        .unwrap();
    let expected_goals = abstractions
        .iter()
        .filter_map(|abstraction| {
            abstraction
                .metadata
                .collection_member
                .map(|member| member.goal_id)
        })
        .collect::<HashSet<_>>();
    let components = abstractions
        .into_iter()
        .map(|abstraction| AbstractionComponent::cartesian(None, abstraction))
        .collect();
    let mut config = scp_config(Saturator::All, true);
    config.online = false;
    config.diversify = true;
    config.samples = 16;
    config.max_orders = max_orders(expected_goals.len());
    config.initial_order_generation_max_time = 10.0;
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

pub(super) fn retained_goal_cover_orders_per_goal(
    heuristic: &SaturatedCostPartitioningHeuristic<'_>,
) -> HashMap<usize, usize> {
    let mut retained_per_goal = HashMap::<usize, usize>::new();
    for goal_id in heuristic
        .state
        .borrow()
        .cp_heuristics
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
mod saturation;
mod scoring;
