#[cfg(test)]
mod tests {
    use planners_sas::numeric::axioms::{
        AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator,
    };
    use planners_sas::numeric::numeric_task::{
        AssignmentEffect, AssignmentOperation, ExplicitVariable, Fact, Metric, NumericRootTask,
        NumericType, NumericVariable, Operator,
    };

    use crate::numeric::evaluation::pattern_databases::pattern_database::PatternDatabase;
    use crate::numeric::evaluation::pattern_databases::projected_task::{Pattern, ProjectedTask};

    fn build_pdb_from_projected_task(
        task: ProjectedTask<'_>,
        max_states: usize,
    ) -> PatternDatabase<'_> {
        PatternDatabase::new(task, max_states).unwrap()
    }

    fn simple_var(name: &str, axiom_layer: i32) -> ExplicitVariable {
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
            Metric::new(true, -1),
            vec![simple_var("p", -1)],
            vec![NumericVariable::new(
                "x".to_string(),
                NumericType::Regular,
                -1,
            )],
            vec![Fact::new(0, 1)],
            vec![],
            vec![0],
            vec![0.0],
            vec![Operator::new(
                "set-goal".to_string(),
                vec![Fact::new(0, 0)],
                vec![planners_sas::numeric::numeric_task::Effect::new(
                    vec![],
                    0,
                    0,
                    1,
                )],
                vec![],
                3,
            )],
            vec![],
            vec![],
            vec![AssignmentAxiom::new(0, CalOperator::Sum, 0, 0)],
            (0, 0),
        )
    }

    fn comparison_guarded_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![simple_var("cmp", 0), simple_var("goal", -1)],
            vec![
                NumericVariable::new("threshold".to_string(), NumericType::Constant, -1),
                NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            ],
            vec![Fact::new(1, 1)],
            vec![],
            vec![2, 0],
            vec![0.0, 0.0],
            vec![Operator::new(
                "advance".to_string(),
                vec![Fact::new(0, 0)],
                vec![planners_sas::numeric::numeric_task::Effect::new(
                    vec![],
                    1,
                    0,
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
            (0, 0),
        )
    }

    fn truncated_chain_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![ExplicitVariable::new(
                3,
                "p".to_string(),
                vec!["p=0".to_string(), "p=1".to_string(), "p=2".to_string()],
                -1,
                2,
            )],
            vec![],
            vec![Fact::new(0, 2)],
            vec![],
            vec![0],
            vec![],
            vec![
                Operator::new(
                    "step-1".to_string(),
                    vec![Fact::new(0, 0)],
                    vec![planners_sas::numeric::numeric_task::Effect::new(
                        vec![],
                        0,
                        0,
                        1,
                    )],
                    vec![],
                    5,
                ),
                Operator::new(
                    "step-2".to_string(),
                    vec![Fact::new(0, 1)],
                    vec![planners_sas::numeric::numeric_task::Effect::new(
                        vec![],
                        0,
                        1,
                        2,
                    )],
                    vec![],
                    5,
                ),
            ],
            vec![],
            vec![],
            vec![],
            (0, 0),
        )
    }

    fn truncation_gap_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, -1),
            vec![ExplicitVariable::new(
                4,
                "p".to_string(),
                vec![
                    "p=0".to_string(),
                    "p=1".to_string(),
                    "p=2".to_string(),
                    "p=3".to_string(),
                ],
                -1,
                3,
            )],
            vec![],
            vec![Fact::new(0, 3)],
            vec![],
            vec![0],
            vec![],
            vec![
                Operator::new(
                    "to-1".to_string(),
                    vec![Fact::new(0, 0)],
                    vec![planners_sas::numeric::numeric_task::Effect::new(vec![], 0, 0, 1)],
                    vec![],
                    1,
                ),
                Operator::new(
                    "to-2".to_string(),
                    vec![Fact::new(0, 0)],
                    vec![planners_sas::numeric::numeric_task::Effect::new(vec![], 0, 0, 2)],
                    vec![],
                    1,
                ),
                Operator::new(
                    "to-3".to_string(),
                    vec![Fact::new(0, 0)],
                    vec![planners_sas::numeric::numeric_task::Effect::new(vec![], 0, 0, 3)],
                    vec![],
                    1,
                ),
            ],
            vec![],
            vec![],
            vec![],
            (0, 0),
        )
    }

    fn cost_only_hidden_numeric_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, 0),
            vec![simple_var("p", -1)],
            vec![
                NumericVariable::new("total-cost".to_string(), NumericType::Cost, -1),
                NumericVariable::new("c1".to_string(), NumericType::Constant, -1),
            ],
            vec![Fact::new(0, 1)],
            vec![],
            vec![0],
            vec![0.0, 1.0],
            vec![
                Operator::new(
                    "wait".to_string(),
                    vec![Fact::new(0, 0)],
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
                    vec![Fact::new(0, 0)],
                    vec![planners_sas::numeric::numeric_task::Effect::new(
                        vec![],
                        0,
                        0,
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
            (0, 0),
        )
    }

    fn zero_metric_cost_hidden_numeric_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, 0),
            vec![simple_var("p", -1)],
            vec![
                NumericVariable::new("total-cost".to_string(), NumericType::Cost, -1),
                NumericVariable::new("zero".to_string(), NumericType::Constant, -1),
            ],
            vec![Fact::new(0, 1)],
            vec![],
            vec![0],
            vec![0.0, 0.0],
            vec![Operator::new(
                "finish".to_string(),
                vec![Fact::new(0, 0)],
                vec![planners_sas::numeric::numeric_task::Effect::new(
                    vec![],
                    0,
                    0,
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
            (0, 0),
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
        let pdb = PatternDatabase::new(projected_task, 1).unwrap();

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
        assert_eq!(initial_prop, vec![0, 0]);

        let pdb = PatternDatabase::new(projected_task, 16).unwrap();

        assert!(pdb.states.len() > 1);
        assert_eq!(pdb.lookup(&initial_prop, &initial_num), Some(1.0));
        assert!(pdb.distances.iter().any(|&distance| distance == 0.0));
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
        assert_eq!(pdb.frontier_states, vec![1]);
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
        assert_eq!(pdb.frontier_states, vec![0, 1]);
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
}
