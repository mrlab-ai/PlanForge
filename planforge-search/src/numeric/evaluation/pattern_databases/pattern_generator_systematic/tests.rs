use planforge_sas::numeric::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use planforge_sas::numeric::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
    NumericRootTask, NumericVariable, Operator,
};

use crate::numeric::evaluation::pattern_databases::projected_task::ProjectedTask;

use super::*;

fn simple_var(name: &str, axiom_layer: Option<usize>) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        axiom_layer,
        1,
    )
}

fn propositional_predecessor_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("q", None), simple_var("p", None)],
        vec![],
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![0, 0],
        vec![],
        vec![Operator::new(
            "set-goal".to_string(),
            vec![ExplicitFact::new(0, 1)],
            vec![planforge_sas::numeric::numeric_task::Effect::new(
                vec![],
                1,
                Some(0),
                1,
            )],
            vec![],
            1,
        )],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn numeric_goal_task() -> NumericRootTask {
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
        vec![0, 0],
        vec![1.0, 0.0],
        vec![],
        vec![PropositionalAxiom::new(
            vec![ExplicitFact::new(0, 0)],
            1,
            0,
            1,
        )],
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

fn eff_eff_goal_join_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("g1", None), simple_var("g2", None)],
        vec![],
        vec![ExplicitFact::new(0, 1), ExplicitFact::new(1, 1)],
        vec![],
        vec![0, 0],
        vec![],
        vec![Operator::new(
            "set-both".to_string(),
            vec![],
            vec![
                Effect::new(vec![], 0, Some(0), 1),
                Effect::new(vec![], 1, Some(0), 1),
            ],
            vec![],
            1,
        )],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    )
}

fn helper_goal_with_unsupported_numeric_effect_task() -> NumericRootTask {
    NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![simple_var("cmp", Some(0)), simple_var("goal", None)],
        vec![
            NumericVariable::new("const2".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
        ],
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![0, 0],
        vec![2.0, 1.0, 1.0, 2.0],
        vec![Operator::new(
            "scale-x".to_string(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                1,
                AssignmentOperation::Times,
                0,
                false,
                vec![],
            )],
            1,
        )],
        vec![PropositionalAxiom::new(
            vec![ExplicitFact::new(0, 0)],
            1,
            0,
            1,
        )],
        vec![ComparisonAxiom::new(
            0,
            3,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(3, CalOperator::Sum, 1, 2)],
        ExplicitFact::new(0, 0),
    )
}

#[test]
fn systematic_generator_includes_goal_singleton_and_predecessor_pair() {
    let task = propositional_predecessor_task();
    let collection = generate_systematic_patterns(
        &task,
        SystematicPatternGeneratorConfig {
            max_pattern_size: 2,
            ..SystematicPatternGeneratorConfig::default()
        },
    );

    assert!(collection.contains(&Pattern::new(vec![1], vec![])));
    assert!(collection.contains(&Pattern::new(vec![0, 1], vec![])));
}

#[test]
fn systematic_generator_returns_projectable_numeric_patterns() {
    let task = numeric_goal_task();
    let collection =
        generate_systematic_patterns(&task, SystematicPatternGeneratorConfig::default());

    assert!(collection.contains(&Pattern::new(vec![], vec![1])));
}

#[test]
fn systematic_generator_joins_disjoint_sga_patterns_via_connection_points() {
    let task = eff_eff_goal_join_task();
    let collection = generate_systematic_patterns(
        &task,
        SystematicPatternGeneratorConfig {
            max_pattern_size: 2,
            ..SystematicPatternGeneratorConfig::default()
        },
    );

    assert!(collection.contains(&Pattern::new(vec![0], vec![])));
    assert!(collection.contains(&Pattern::new(vec![1], vec![])));
    assert!(collection.contains(&Pattern::new(vec![0, 1], vec![])));
}

#[test]
fn systematic_generator_keeps_patterns_rejected_by_projected_task() {
    let task = helper_goal_with_unsupported_numeric_effect_task();
    let helper_var_id = task.numeric_variables().len();

    assert!(ProjectedTask::new(&task, &Pattern::new(vec![], vec![helper_var_id])).is_err());

    let collection = generate_systematic_patterns(
        &task,
        SystematicPatternGeneratorConfig {
            max_pattern_size: 2,
            ..SystematicPatternGeneratorConfig::default()
        },
    );

    assert!(collection.contains(&Pattern::new(vec![], vec![helper_var_id])));
}

#[test]
#[should_panic(expected = "not implemented: numeric systematic naive pattern generation")]
fn systematic_generator_rejects_naive_mode_like_cpp() {
    let task = propositional_predecessor_task();
    let _ = generate_systematic_patterns(
        &task,
        SystematicPatternGeneratorConfig {
            only_interesting_patterns: false,
            max_pattern_size: 2,
            ..SystematicPatternGeneratorConfig::default()
        },
    );
}
