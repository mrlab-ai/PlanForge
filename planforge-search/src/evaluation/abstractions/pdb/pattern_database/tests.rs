use planforge_sas::explicit_state::StateValues;
use std::sync::Arc;

use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::explicit_state::ExplicitState;
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex,
    ExplicitVariable, Metric, NumericType, NumericValue, NumericVariable, Operator, OperatorCost,
    OperatorIndex, Task, TaskParts, VariableIndex,
};
use planforge_sas::state_registry::StateRegistry;

use super::{PatternDatabase, PdbBuildConfig, PdbHeuristicConfig, PdbInternalHeuristic};
use crate::task_transformations::projection::{Pattern, ProjectedTask};

fn build(task: &Task, pattern: &Pattern, max_states: usize) -> PatternDatabase {
    PatternDatabase::new(ProjectedTask::new(task, pattern).unwrap(), max_states).unwrap()
}

fn build_with(
    task: &Task,
    pattern: &Pattern,
    max_states: usize,
    heuristics: PdbHeuristicConfig,
) -> PatternDatabase {
    PatternDatabase::build(
        ProjectedTask::new(task, pattern).unwrap(),
        PdbBuildConfig {
            max_states,
            heuristics,
            retain_transitions: true,
        },
    )
    .unwrap()
}

fn initial_state(task: &Task) -> ExplicitState {
    let mut registry = StateRegistry::for_task(Arc::new(task.clone()));
    let initial = registry.get_initial_state();
    let mut values = ExplicitState::default();
    registry.view(&initial).fill_explicit(&mut values).unwrap();
    values
}

fn value(pdb: &PatternDatabase, state: &ExplicitState) -> f64 {
    pdb.value(StateValues::new(&state.propositional, &state.numeric))
}

fn propositional(values: &[usize]) -> Vec<ExplicitValueIndex> {
    values
        .iter()
        .copied()
        .map(ExplicitValueIndex::from_usize)
        .collect()
}

fn regular_pattern(var: usize) -> Pattern {
    Pattern::new(vec![VariableIndex::from_usize(var)], vec![])
}

fn effect(var: usize, from: usize, to: usize) -> Effect {
    Effect::new(
        vec![],
        VariableIndex::from_usize(var),
        Some(ExplicitValueIndex::from_usize(from)),
        ExplicitValueIndex::from_usize(to),
    )
}

fn simple_var(name: &str, axiom_layer: Option<usize>) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        axiom_layer,
        ExplicitValueIndex::new(1),
    )
}

/// A variable carrying a numeric condition, with the three values the
/// translator gives one: true, false and "not derived yet".
fn condition_var(name: &str, axiom_layer: usize) -> ExplicitVariable {
    ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        name.to_string(),
        vec![
            format!("{name}=0"),
            format!("{name}=1"),
            format!("{name}=2"),
        ],
        Some(axiom_layer),
        ExplicitValueIndex::new(1),
    )
}

fn comparison_guarded_task() -> Task {
    Task::new(TaskParts {
        metric: Metric::new(true, None),
        variables: vec![condition_var("cmp", 0), simple_var("goal", None)],
        numeric_variables: vec![
            NumericVariable::new("threshold".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2), ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(0.0)],
        operators: vec![Operator::new(
            "advance".to_string(),
            vec![ExplicitFact::propositional(0, 0)],
            vec![planforge_sas::numeric_task::Effect::new(
                vec![],
                VariableIndex::new(1),
                Some(ExplicitValueIndex::new(0)),
                ExplicitValueIndex::new(1),
            )],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(1),
            VariableIndex::new(0),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    })
}

fn truncated_chain_task() -> Task {
    Task::new(TaskParts {
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            3,
            "p".to_string(),
            vec!["p=0".to_string(), "p=1".to_string(), "p=2".to_string()],
            None,
            ExplicitValueIndex::new(2),
        )],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 2)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![
            Operator::new(
                "step-1".to_string(),
                vec![ExplicitFact::propositional(0, 0)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    VariableIndex::new(0),
                    Some(ExplicitValueIndex::new(0)),
                    ExplicitValueIndex::new(1),
                )],
                vec![],
                OperatorCost::new(5),
            ),
            Operator::new(
                "step-2".to_string(),
                vec![ExplicitFact::propositional(0, 1)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    VariableIndex::new(0),
                    Some(ExplicitValueIndex::new(1)),
                    ExplicitValueIndex::new(2),
                )],
                vec![],
                OperatorCost::new(5),
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    })
}

fn relevance_precision_task() -> Task {
    Task::new(TaskParts {
        metric: Metric::new(true, None),
        variables: vec![simple_var("p", None)],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![
            Operator::new(
                "set-goal".to_string(),
                vec![ExplicitFact::propositional(0, 0)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    VariableIndex::new(0),
                    Some(ExplicitValueIndex::new(0)),
                    ExplicitValueIndex::new(1),
                )],
                vec![],
                OperatorCost::new(1),
            ),
            Operator::new(
                "goal-self-loop".to_string(),
                vec![ExplicitFact::propositional(0, 1)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    VariableIndex::new(0),
                    Some(ExplicitValueIndex::new(1)),
                    ExplicitValueIndex::new(1),
                )],
                vec![],
                OperatorCost::new(1),
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    })
}

fn cost_only_hidden_numeric_task() -> Task {
    Task::new(TaskParts {
        metric: Metric::new(true, Some(VariableIndex::from_usize(0))),
        variables: vec![simple_var("p", None)],
        numeric_variables: vec![
            NumericVariable::new("total-cost".to_string(), NumericType::Cost, None),
            NumericVariable::new("c1".to_string(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(1.0)],
        operators: vec![
            Operator::new(
                "wait".to_string(),
                vec![ExplicitFact::propositional(0, 0)],
                vec![],
                vec![AssignmentEffect::new(
                    VariableIndex::new(0),
                    AssignmentOperation::Plus,
                    VariableIndex::new(1),
                    false,
                    vec![],
                )],
                OperatorCost::new(1),
            ),
            Operator::new(
                "finish".to_string(),
                vec![ExplicitFact::propositional(0, 0)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    VariableIndex::new(0),
                    Some(ExplicitValueIndex::new(0)),
                    ExplicitValueIndex::new(1),
                )],
                vec![AssignmentEffect::new(
                    VariableIndex::new(0),
                    AssignmentOperation::Plus,
                    VariableIndex::new(1),
                    false,
                    vec![],
                )],
                OperatorCost::new(1),
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    })
}

fn zero_metric_cost_hidden_numeric_task() -> Task {
    Task::new(TaskParts {
        metric: Metric::new(true, Some(VariableIndex::from_usize(0))),
        variables: vec![simple_var("p", None)],
        numeric_variables: vec![
            NumericVariable::new("total-cost".to_string(), NumericType::Cost, None),
            NumericVariable::new("zero".to_string(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(0.0)],
        operators: vec![Operator::new(
            "finish".to_string(),
            vec![ExplicitFact::propositional(0, 0)],
            vec![planforge_sas::numeric_task::Effect::new(
                vec![],
                VariableIndex::new(0),
                Some(ExplicitValueIndex::new(0)),
                ExplicitValueIndex::new(1),
            )],
            vec![AssignmentEffect::new(
                VariableIndex::from_usize(0),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(1),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    })
}

fn failed_lookup_chain_task() -> Task {
    Task::new(TaskParts {
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            5,
            "p".to_string(),
            vec![
                "p=0".to_string(),
                "p=1".to_string(),
                "p=2".to_string(),
                "p=3".to_string(),
                "p=4".to_string(),
            ],
            None,
            ExplicitValueIndex::new(4),
        )],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 4)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![
            Operator::new(
                "to-1".to_string(),
                vec![ExplicitFact::propositional(0, 0)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    VariableIndex::new(0),
                    Some(ExplicitValueIndex::new(0)),
                    ExplicitValueIndex::new(1),
                )],
                vec![],
                OperatorCost::new(1),
            ),
            Operator::new(
                "to-2".to_string(),
                vec![ExplicitFact::propositional(0, 1)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    VariableIndex::new(0),
                    Some(ExplicitValueIndex::new(1)),
                    ExplicitValueIndex::new(2),
                )],
                vec![],
                OperatorCost::new(1),
            ),
            Operator::new(
                "to-3".to_string(),
                vec![ExplicitFact::propositional(0, 2)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    VariableIndex::new(0),
                    Some(ExplicitValueIndex::new(2)),
                    ExplicitValueIndex::new(3),
                )],
                vec![],
                OperatorCost::new(1),
            ),
            Operator::new(
                "to-4".to_string(),
                vec![ExplicitFact::propositional(0, 3)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    VariableIndex::new(0),
                    Some(ExplicitValueIndex::new(3)),
                    ExplicitValueIndex::new(4),
                )],
                vec![],
                OperatorCost::new(1),
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    })
}

/// `inc` raises `x` by one at cost 2 while `p` is 0, and `trap` sets `p` to
/// 1 at cost 1; the goal is `x >= 5`.
fn counter_task() -> Task {
    Task::new(TaskParts {
        metric: Metric::new(true, None),
        variables: vec![condition_var("x>=5", 0), simple_var("p", None)],
        numeric_variables: vec![
            NumericVariable::new("five".to_string(), NumericType::Constant, None),
            NumericVariable::new("one".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2), ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(5.0),
            NumericValue::new(1.0),
            NumericValue::new(0.0),
        ],
        operators: vec![
            Operator::new(
                "inc".to_string(),
                vec![ExplicitFact::propositional(1, 0)],
                vec![],
                vec![AssignmentEffect::new(
                    VariableIndex::new(2),
                    AssignmentOperation::Plus,
                    VariableIndex::new(1),
                    false,
                    vec![],
                )],
                OperatorCost::new(2),
            ),
            Operator::new(
                "trap".to_string(),
                vec![ExplicitFact::propositional(1, 0)],
                vec![effect(1, 0, 1)],
                vec![],
                OperatorCost::new(1),
            ),
        ],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(2),
            VariableIndex::new(0),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: Some(ExplicitFact::propositional(0, 0)),
    })
}

/// The base task state of `counter_task` with the given `x` and `p`.
fn counter_state(x: f64, p: usize) -> ExplicitState {
    let reached = if x >= 5.0 { 0 } else { 1 };
    ExplicitState {
        propositional: propositional(&[reached, p]),
        numeric: vec![
            NumericValue::new(5.0),
            NumericValue::new(1.0),
            NumericValue::new(x),
        ],
    }
}

fn counter_pattern() -> Pattern {
    Pattern::new(
        vec![VariableIndex::from_usize(1)],
        vec![VariableIndex::from_usize(2)],
    )
}

#[test]
fn propositional_projection_is_explored_in_full() {
    let task = truncated_chain_task();
    let pdb = build(&task, &regular_pattern(0), 1);

    assert!(pdb.is_exhausted());
    assert_eq!(pdb.num_states(), 3);
    assert_eq!(pdb.value(StateValues::new(&propositional(&[0]), &[])), 10.0);
    assert_eq!(pdb.value(StateValues::new(&propositional(&[1]), &[])), 5.0);
    assert_eq!(pdb.value(StateValues::new(&propositional(&[2]), &[])), 0.0);
}

#[test]
fn unreachable_state_of_an_exhausted_projection_is_a_dead_end() {
    let task = failed_lookup_chain_task();
    let pdb = build(&task, &regular_pattern(0), 1);

    assert!(pdb.is_exhausted());
    assert_eq!(pdb.value(StateValues::new(&propositional(&[0]), &[])), 4.0);
    assert_eq!(pdb.value(StateValues::new(&propositional(&[3]), &[])), 1.0);
}

#[test]
fn numeric_exploration_is_exact_when_the_goal_lies_within_the_limit() {
    let task = counter_task();
    let pdb = build(&task, &counter_pattern(), 64);

    assert!(!pdb.is_exhausted());
    assert_eq!(value(&pdb, &counter_state(0.0, 0)), 10.0);
    assert_eq!(value(&pdb, &counter_state(4.0, 0)), 2.0);
    assert_eq!(value(&pdb, &counter_state(5.0, 0)), 0.0);
    assert_eq!(value(&pdb, &initial_state(&task)), 10.0);
}

#[test]
fn truncated_exploration_seeds_the_frontier_with_the_cheapest_operator_cost() {
    let task = counter_task();
    // Expanding x=0 reaches x=1 and the trap; the trap, cheaper, is
    // expanded next and has no successor; x=1 then reaches x=2 and its trap,
    // which both lie on the frontier.
    let pdb = build(&task, &counter_pattern(), 3);

    assert!(!pdb.is_exhausted());
    assert_eq!(pdb.min_operator_cost(), 1.0);
    assert_eq!(value(&pdb, &counter_state(2.0, 0)), 1.0);
    assert_eq!(value(&pdb, &counter_state(1.0, 1)), 1.0);
    assert_eq!(value(&pdb, &counter_state(0.0, 0)), 4.0);
}

#[test]
fn stored_dead_end_is_final_in_a_truncated_exploration() {
    let task = counter_task();
    let pdb = build(&task, &counter_pattern(), 3);

    assert_eq!(value(&pdb, &counter_state(0.0, 1)), f64::INFINITY);
}

#[test]
fn failed_lookup_is_zero_at_an_abstract_goal_and_blind_elsewhere() {
    let task = counter_task();
    let pdb = build(&task, &counter_pattern(), 3);

    assert_eq!(
        pdb.abstract_state_id(StateValues::new(
            &counter_state(7.0, 0).propositional,
            &counter_state(7.0, 0).numeric
        )),
        None
    );
    assert_eq!(value(&pdb, &counter_state(7.0, 0)), 0.0);
    assert_eq!(value(&pdb, &counter_state(3.0, 0)), 1.0);
}

#[test]
fn failed_lookup_lmcut_is_admissible_and_more_informed_than_blind() {
    let task = counter_task();
    let lmcut = PdbHeuristicConfig {
        failed_lookup_heuristic: PdbInternalHeuristic::Lmcut,
        ..PdbHeuristicConfig::default()
    };
    let blind_pdb = build(&task, &counter_pattern(), 0);
    let lmcut_pdb = build_with(&task, &counter_pattern(), 0, lmcut);

    let state = counter_state(1.0, 0);
    let blind_value = value(&blind_pdb, &state);
    let lmcut_value = value(&lmcut_pdb, &state);
    assert_eq!(blind_value, 1.0);
    assert!(lmcut_value > blind_value);
    assert!(lmcut_value <= 8.0);
}

#[test]
fn lmcut_exploration_skips_its_dead_ends() {
    let task = counter_task();
    let lmcut = PdbHeuristicConfig {
        exploration_heuristic: PdbInternalHeuristic::Lmcut,
        ..PdbHeuristicConfig::default()
    };
    // Blind exploration spends states on the traps. LM-cut recognises them
    // as dead ends and leaves them closed, so its states all lie on the path
    // to the goal: it reaches x=4, whose frontier seed is the cheapest
    // operator cost.
    let blind_pdb = build(&task, &counter_pattern(), 8);
    let lmcut_pdb = build_with(&task, &counter_pattern(), 8, lmcut);

    assert_eq!(value(&blind_pdb, &counter_state(0.0, 0)), 8.0);
    assert_eq!(value(&lmcut_pdb, &counter_state(0.0, 0)), 9.0);
}

#[test]
fn lookups_project_the_base_task_state() {
    let task = comparison_guarded_task();
    let pattern = Pattern::new(
        vec![VariableIndex::from_usize(1)],
        vec![VariableIndex::from_usize(1)],
    );
    let pdb = build(&task, &pattern, 16);

    assert_eq!(value(&pdb, &initial_state(&task)), 1.0);
}

#[test]
fn hidden_cost_variables_do_not_split_abstract_states() {
    let task = cost_only_hidden_numeric_task();
    let pdb = build(&task, &regular_pattern(0), 64);

    assert_eq!(pdb.num_states(), 2);
    assert_eq!(value(&pdb, &initial_state(&task)), 1.0);
}

#[test]
fn zero_metric_costs_give_zero_distances() {
    let task = zero_metric_cost_hidden_numeric_task();
    let pdb = build(&task, &regular_pattern(0), 64);

    assert_eq!(pdb.min_operator_cost(), 0.0);
    assert_eq!(value(&pdb, &initial_state(&task)), 0.0);
}

#[test]
fn relevant_operators_of_an_exhausted_projection_label_a_transition() {
    let task = relevance_precision_task();
    let pdb = build(&task, &regular_pattern(0), 32);

    assert_eq!(pdb.relevant_operator_ids(), &[OperatorIndex::new(0)]);
}

#[test]
fn cost_partitioned_tables_saturate_the_operator_costs() {
    let task = truncated_chain_task();
    let pdb = build(&task, &regular_pattern(0), 32);
    let id = |p| {
        pdb.abstract_state_id(StateValues::new(&propositional(&[p]), &[]))
            .unwrap()
    };

    let (distances, saturated) = pdb.build_cost_partitioned_distance_table(&[3.0, 4.0]);
    assert_eq!(distances[id(0)], 7.0);
    assert_eq!(distances[id(1)], 4.0);
    assert_eq!(saturated, vec![3.0, 4.0]);

    let (distances, saturated) =
        pdb.build_cost_partitioned_distance_table_capped(&[3.0, 4.0], id(1));
    assert_eq!(saturated, vec![0.0, 4.0]);
    assert_eq!(distances[id(0)], 4.0);
}
