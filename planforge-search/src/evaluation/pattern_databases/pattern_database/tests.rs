use planforge_sas::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
    NumericType, NumericVariable, Operator,
};
use planforge_sas::state_registry::StateRegistry;

use crate::evaluation::pattern_databases::pattern_database::{
    PatternDatabase, PdbHeuristicConfig, PdbInternalHeuristic,
};
use crate::evaluation::pattern_databases::projected_task::{Pattern, ProjectedTask};

fn build_pdb_from_projected_task(
    task: ProjectedTask<'_>,
    max_states: usize,
) -> PatternDatabase<'_> {
    PatternDatabase::new(task, max_states).unwrap()
}

fn build_pdb_with_heuristics(
    task: ProjectedTask<'_>,
    max_states: usize,
    heuristic_config: PdbHeuristicConfig,
) -> PatternDatabase<'_> {
    PatternDatabase::with_heuristic_config(task, max_states, heuristic_config).unwrap()
}

fn simple_var(name: &str, axiom_layer: Option<usize>) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        axiom_layer,
        1,
    )
}

fn propositional_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("p", None)],
        vec![NumericVariable::new(
            "x".to_string(),
            NumericType::Regular,
            None,
        )],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![0.0],
        vec![Operator::new(
            "set-goal".to_string(),
            vec![ExplicitFact::new(0, 0)],
            vec![planforge_sas::numeric_task::Effect::new(
                vec![],
                0,
                Some(0),
                1,
            )],
            vec![],
            3,
        )],
        vec![],
        vec![],
        vec![AssignmentAxiom::new(0, CalOperator::Sum, 0, 0)],
        ExplicitFact::new(0, 0),
    )
}

fn comparison_guarded_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("cmp", Some(0)), simple_var("goal", None)],
        vec![
            NumericVariable::new("threshold".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![2, 0],
        vec![0.0, 0.0],
        vec![Operator::new(
            "advance".to_string(),
            vec![ExplicitFact::new(0, 0)],
            vec![planforge_sas::numeric_task::Effect::new(
                vec![],
                1,
                Some(0),
                1,
            )],
            vec![],
            1,
        )],
        vec![],
        vec![ComparisonAxiom::new(
            0,
            1,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn truncated_chain_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            3,
            "p".to_string(),
            vec!["p=0".to_string(), "p=1".to_string(), "p=2".to_string()],
            None,
            2,
        )],
        vec![],
        vec![ExplicitFact::new(0, 2)],
        vec![],
        vec![0],
        vec![],
        vec![
            Operator::new(
                "step-1".to_string(),
                vec![ExplicitFact::new(0, 0)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(0),
                    1,
                )],
                vec![],
                5,
            ),
            Operator::new(
                "step-2".to_string(),
                vec![ExplicitFact::new(0, 1)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(1),
                    2,
                )],
                vec![],
                5,
            ),
        ],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn relevance_precision_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("p", None)],
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![
            Operator::new(
                "set-goal".to_string(),
                vec![ExplicitFact::new(0, 0)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(0),
                    1,
                )],
                vec![],
                1,
            ),
            Operator::new(
                "goal-self-loop".to_string(),
                vec![ExplicitFact::new(0, 1)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(1),
                    1,
                )],
                vec![],
                1,
            ),
        ],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

#[test]
fn relevant_operators_are_exact_when_complete_and_sound_when_truncated() {
    let task = relevance_precision_task();
    let pattern = Pattern::new(vec![0], vec![]);
    let complete = PatternDatabase::new(ProjectedTask::new(&task, &pattern).unwrap(), 32).unwrap();
    let truncated = PatternDatabase::new(ProjectedTask::new(&task, &pattern).unwrap(), 1).unwrap();

    assert_eq!(complete.relevant_operator_ids(), vec![0]);
    assert_eq!(truncated.relevant_operator_ids(), vec![0, 1]);
}

fn truncation_gap_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
            4,
            "p".to_string(),
            vec![
                "p=0".to_string(),
                "p=1".to_string(),
                "p=2".to_string(),
                "p=3".to_string(),
            ],
            None,
            3,
        )],
        vec![],
        vec![ExplicitFact::new(0, 3)],
        vec![],
        vec![0],
        vec![],
        vec![
            Operator::new(
                "to-1".to_string(),
                vec![ExplicitFact::new(0, 0)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(0),
                    1,
                )],
                vec![],
                1,
            ),
            Operator::new(
                "to-2".to_string(),
                vec![ExplicitFact::new(0, 0)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(0),
                    2,
                )],
                vec![],
                1,
            ),
            Operator::new(
                "to-3".to_string(),
                vec![ExplicitFact::new(0, 0)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(0),
                    3,
                )],
                vec![],
                1,
            ),
        ],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn cost_only_hidden_numeric_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, Some(0)),
        vec![simple_var("p", None)],
        vec![
            NumericVariable::new("total-cost".to_string(), NumericType::Cost, None),
            NumericVariable::new("c1".to_string(), NumericType::Constant, None),
        ],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![0.0, 1.0],
        vec![
            Operator::new(
                "wait".to_string(),
                vec![ExplicitFact::new(0, 0)],
                vec![],
                vec![AssignmentEffect::new(
                    0,
                    AssignmentOperation::Plus,
                    1,
                    false,
                    vec![],
                )],
                1,
            ),
            Operator::new(
                "finish".to_string(),
                vec![ExplicitFact::new(0, 0)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(0),
                    1,
                )],
                vec![AssignmentEffect::new(
                    0,
                    AssignmentOperation::Plus,
                    1,
                    false,
                    vec![],
                )],
                1,
            ),
        ],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn zero_metric_cost_hidden_numeric_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, Some(0)),
        vec![simple_var("p", None)],
        vec![
            NumericVariable::new("total-cost".to_string(), NumericType::Cost, None),
            NumericVariable::new("zero".to_string(), NumericType::Constant, None),
        ],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![0.0, 0.0],
        vec![Operator::new(
            "finish".to_string(),
            vec![ExplicitFact::new(0, 0)],
            vec![planforge_sas::numeric_task::Effect::new(
                vec![],
                0,
                Some(0),
                1,
            )],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                1,
                false,
                vec![],
            )],
            1,
        )],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn numeric_pair_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("p", None)],
        vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
        ],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![3.0, 7.0],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn failed_lookup_chain_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![ExplicitVariable::new(
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
            4,
        )],
        vec![],
        vec![ExplicitFact::new(0, 4)],
        vec![],
        vec![0],
        vec![],
        vec![
            Operator::new(
                "to-1".to_string(),
                vec![ExplicitFact::new(0, 0)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(0),
                    1,
                )],
                vec![],
                1,
            ),
            Operator::new(
                "to-2".to_string(),
                vec![ExplicitFact::new(0, 1)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(1),
                    2,
                )],
                vec![],
                1,
            ),
            Operator::new(
                "to-3".to_string(),
                vec![ExplicitFact::new(0, 2)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(2),
                    3,
                )],
                vec![],
                1,
            ),
            Operator::new(
                "to-4".to_string(),
                vec![ExplicitFact::new(0, 3)],
                vec![planforge_sas::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(3),
                    4,
                )],
                vec![],
                1,
            ),
        ],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

#[test]
fn lookup_returns_distance_for_reached_state() {
    let task = propositional_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![0],
        },
    )
    .unwrap();
    let pdb = PatternDatabase::new(projected_task, 32).unwrap();

    assert_eq!(pdb.lookup(&[0], &[0.0]), Some(3.0));
    assert_eq!(pdb.lookup(&[1], &[0.0]), Some(0.0));
}

#[test]
fn pattern_database_accepts_numeric_abstract_task_boundary() {
    let task = propositional_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![0],
        },
    )
    .unwrap();

    let pdb = build_pdb_from_projected_task(projected_task, 32);

    assert_eq!(pdb.lookup(&[0], &[0.0]), Some(3.0));
}

#[test]
fn lookup_miss_returns_zero_for_goal_state() {
    let task = propositional_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![0],
        },
    )
    .unwrap();
    let pdb = PatternDatabase::new(projected_task, 0).unwrap();

    assert_eq!(pdb.lookup(&[1], &[0.0]), None);
    assert_eq!(pdb.lookup_or_fallback(&[1], &[0.0]), 0.0);
}

#[test]
fn lookup_miss_returns_min_operator_cost_for_non_goal_state() {
    let task = propositional_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![0],
        },
    )
    .unwrap();
    let pdb = PatternDatabase::new(projected_task, 1).unwrap();

    assert_eq!(pdb.lookup(&[0], &[42.0]), None);
    assert_eq!(pdb.lookup_or_fallback(&[0], &[42.0]), 3.0);
}

#[test]
fn direct_concrete_lookup_matches_projected_lookup_for_propositional_task() {
    let task = propositional_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![0],
        },
    )
    .unwrap();
    let pdb = PatternDatabase::new(projected_task, 32).unwrap();
    let mut state_registry = StateRegistry::for_task(std::sync::Arc::new(&task));
    let initial_state = state_registry.get_initial_state();

    assert_eq!(
        pdb.lookup_or_fallback_from_concrete_state(&initial_state, &state_registry)
            .unwrap(),
        pdb.lookup_or_fallback(&[0], &[0.0]),
    );
}

#[test]
fn pdb_build_expands_from_axiom_closed_initial_state() {
    let task = comparison_guarded_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![1],
            numeric: vec![1],
        },
    )
    .unwrap();

    let (initial_prop, initial_num) = projected_task.evaluated_initial_state_values().unwrap();
    assert_eq!(initial_prop, vec![0]);

    let pdb = PatternDatabase::new(projected_task, 16).unwrap();

    assert!(pdb.states.len() > 1);
    assert_eq!(pdb.lookup(&[0], &[0.0]), Some(1.0));
    assert_eq!(pdb.lookup(&initial_prop, &initial_num), Some(1.0));
    assert!(pdb.distances.contains(&0.0));
}

#[test]
fn direct_concrete_lookup_matches_projected_lookup_for_comparison_guarded_task() {
    let task = comparison_guarded_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![1],
            numeric: vec![1],
        },
    )
    .unwrap();
    let (initial_prop, initial_num) = projected_task.evaluated_initial_state_values().unwrap();
    let pdb = PatternDatabase::new(projected_task, 16).unwrap();
    let mut state_registry = StateRegistry::for_task(std::sync::Arc::new(&task));
    let initial_state = state_registry.get_initial_state();

    assert_eq!(
        pdb.lookup_or_fallback_from_concrete_state(&initial_state, &state_registry)
            .unwrap(),
        pdb.lookup_or_fallback(&initial_prop, &initial_num),
    );
}

#[test]
fn truncated_pdb_propagates_frontier_seed_costs() {
    let task = truncated_chain_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![],
        },
    )
    .unwrap();

    let pdb = PatternDatabase::new(projected_task, 2).unwrap();

    assert!(pdb.truncated);
    assert_eq!(pdb.reached_goal_states, 0);
    assert_eq!(pdb.frontier_states, vec![2]);
    assert_eq!(pdb.lookup(&[1], &[]), Some(5.0));
    assert_eq!(pdb.lookup(&[0], &[]), Some(10.0));
    assert_eq!(pdb.lookup_or_fallback(&[0], &[]), 10.0);
}

#[test]
fn truncated_pdb_handles_multiple_new_successors_after_hitting_limit() {
    let task = truncation_gap_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![],
        },
    )
    .unwrap();

    let pdb = PatternDatabase::new(projected_task, 2).unwrap();

    assert!(pdb.truncated);
    assert_eq!(pdb.frontier_states, vec![1, 2, 3]);
    assert_eq!(pdb.lookup(&[0], &[]), Some(1.0));
}

#[test]
fn pdb_collapses_hidden_cost_dimensions_outside_pattern() {
    let task = cost_only_hidden_numeric_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![],
        },
    )
    .unwrap();

    let pdb = PatternDatabase::new(projected_task, 64).unwrap();

    assert_eq!(pdb.states.len(), 2);
    assert_eq!(pdb.lookup(&[0], &[]), Some(1.0));
    assert_eq!(pdb.lookup(&[1], &[]), Some(0.0));
}

#[test]
fn direct_concrete_lookup_uses_compact_prop_table_for_pure_propositional_patterns() {
    let task = cost_only_hidden_numeric_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![],
        },
    )
    .unwrap();

    let pdb = PatternDatabase::new(projected_task, 64).unwrap();
    let mut state_registry = StateRegistry::for_task(std::sync::Arc::new(&task));
    let initial_state = state_registry.get_initial_state();

    assert_eq!(
        pdb.lookup_or_fallback_from_concrete_state(&initial_state, &state_registry)
            .unwrap(),
        1.0,
    );
}

#[test]
fn direct_concrete_numeric_lookup_keeps_slot_zero_for_prop_hash() {
    let task = numeric_pair_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![],
            numeric: vec![0, 1],
        },
    )
    .unwrap();

    let mut pdb = PatternDatabase::new(projected_task, 0).unwrap();
    pdb.states = vec![super::PdbState {
        propositional: vec![],
        numeric: vec![3.0, 7.0],
    }];
    pdb.distances = vec![11.0];
    pdb.rebuild_lookup_indexes();

    let mut state_registry = StateRegistry::for_task(std::sync::Arc::new(&task));
    let initial_state = state_registry.get_initial_state();

    assert_eq!(
        pdb.lookup_or_fallback_from_concrete_state(&initial_state, &state_registry)
            .unwrap(),
        11.0,
    );
}

#[test]
fn lookup_uses_min_distance_across_pattern_aliases() {
    let task = comparison_guarded_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![1],
            numeric: vec![1],
        },
    )
    .unwrap();

    let mut pdb = PatternDatabase::new(projected_task, 16).unwrap();
    pdb.states = vec![
        super::PdbState {
            propositional: vec![0],
            numeric: vec![0.0],
        },
        super::PdbState {
            propositional: vec![0],
            numeric: vec![0.0],
        },
    ];
    pdb.distances = vec![5.0, 1.0];
    pdb.rebuild_lookup_indexes();

    assert_eq!(pdb.lookup(&[0], &[0.0]), Some(1.0));
}

#[test]
fn projected_runtime_lookup_uses_pattern_min_across_full_projected_aliases() {
    let task = comparison_guarded_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![],
        },
    )
    .unwrap();

    let mut pdb = PatternDatabase::new(projected_task, 16).unwrap();
    pdb.states = vec![
        super::PdbState {
            propositional: vec![0],
            numeric: vec![1.0, 0.0],
        },
        super::PdbState {
            propositional: vec![0],
            numeric: vec![0.0, 0.0],
        },
    ];
    pdb.distances = vec![5.0, 1.0];
    pdb.rebuild_lookup_indexes();

    assert_eq!(
        pdb.lookup_pattern_or_fallback_in_projected_values(&[0], &[0.0, 0.0]),
        1.0
    );
}

#[test]
fn pdb_uses_metric_delta_costs_even_when_metric_var_is_hidden() {
    let task = zero_metric_cost_hidden_numeric_task();
    let projected_task = ProjectedTask::new(
        &task,
        &Pattern {
            regular: vec![0],
            numeric: vec![],
        },
    )
    .unwrap();

    let pdb = PatternDatabase::new(projected_task, 64).unwrap();

    assert_eq!(pdb.min_operator_cost(), 0.0);
    assert_eq!(pdb.lookup(&[0], &[]), Some(0.0));
    assert_eq!(pdb.lookup(&[1], &[]), Some(0.0));
}

#[test]
fn failed_lookup_lmcut_is_more_informed_than_blind() {
    let task = failed_lookup_chain_task();
    let pattern = Pattern {
        regular: vec![0],
        numeric: vec![],
    };
    let blind_projected_task = ProjectedTask::new(&task, &pattern).unwrap();
    let blind_pdb = build_pdb_with_heuristics(
        blind_projected_task,
        1,
        PdbHeuristicConfig {
            failed_lookup_heuristic: PdbInternalHeuristic::Blind,
            ..PdbHeuristicConfig::default()
        },
    );

    let lmcut_projected_task = ProjectedTask::new(&task, &pattern).unwrap();
    let lmcut_pdb = build_pdb_with_heuristics(
        lmcut_projected_task,
        1,
        PdbHeuristicConfig {
            failed_lookup_heuristic: PdbInternalHeuristic::Lmcut,
            ..PdbHeuristicConfig::default()
        },
    );

    let blind_value = blind_pdb.lookup_or_fallback(&[2], &[]);
    let lmcut_value = lmcut_pdb.lookup_or_fallback(&[2], &[]);

    assert_eq!(blind_value, 1.0);
    assert!(lmcut_value > blind_value);
}
