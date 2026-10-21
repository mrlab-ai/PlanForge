use super::diverse_orders::*;
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

use super::orders::*;

use super::sampling::*;

use super::*;
use crate::evaluation::abstractions::common::abstraction::CegarAbstraction;

use crate::evaluation::abstractions::collections::strategy::CollectionStrategy;

use planforge_sas::numeric_task::{
    Effect, ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, Operator, OperatorCost, VariableIndex,
};

use planforge_sas::state_registry::StateRegistry;

use crate::evaluation::abstractions::cartesian::{
    CartesianAbstractionCollectionConfig, CartesianAbstractionCollectionGenerator,
    CartesianAbstractionConfig, CartesianAbstractionGenerator,
};

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
            max_time: None,
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
        max_time: 10.0,
        table_construction_max_time: 10.0,
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

#[test]
fn scp_rejects_negative_and_nan_time_limits() {
    let valid = scp_config(Saturator::All, false);
    valid.validate(true).unwrap();
    for seconds in [-1.0, f64::NAN, f64::NEG_INFINITY] {
        let configs = [
            ScpConfig {
                max_time: seconds,
                ..valid.clone()
            },
            ScpConfig {
                table_construction_max_time: seconds,
                ..valid.clone()
            },
            ScpConfig {
                initial_order_generation_max_time: seconds,
                ..valid.clone()
            },
        ];
        for config in configs {
            let error = config.validate(true).unwrap_err();
            assert!(error.contains("non-negative number of seconds"), "{error}");
        }
    }
}

#[test]
fn capping_scp_construction_time_keeps_invalid_limits() {
    let mut config = ScpConfig {
        table_construction_max_time: f64::NAN,
        initial_order_generation_max_time: f64::INFINITY,
        ..scp_config(Saturator::All, false)
    };
    config.cap_construction_time(5.0);
    assert!(config.table_construction_max_time.is_nan());
    assert_eq!(config.initial_order_generation_max_time, 5.0);
    assert!(config.validate(true).is_err());
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
                    max_time: None,
                    combine_labels: false,
                    random_seed: Some(1),
                    debug: false,
                    ..CartesianAbstractionConfig::default().cegar
                },
            },
            collection_strategy: CollectionStrategy::GoalFocused,
            variants_per_goal: 4,
            max_collection_size: 128,
            total_max_time: None,
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
    config.initial_order_generation_max_time = 10.0;
    let heuristic = SaturatedCostPartitioningHeuristic::new(
        None,
        components,
        config,
        &**task,
        ScpBuildOptions {
            sampling_task: Some(task.clone()),
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
