use planforge_sas::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex,
    ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts, NumericValue, NumericVariable,
    Operator, OperatorCost, VariableIndex,
};

use super::*;

fn simple_var(name: &str, axiom_layer: Option<usize>) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        axiom_layer,
        ExplicitValueIndex::new(1),
    )
}

fn propositional_predecessor_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![simple_var("q", None), simple_var("p", None)],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![Operator::new(
            "set-goal".to_string(),
            vec![ExplicitFact::propositional(0, 1)],
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
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

/// The goal is the comparison `x >= threshold`, which is what a numeric goal
/// compiles to: the pattern generator has to reach the comparison's numeric
/// operand from it.
fn numeric_goal_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![simple_var("cmp", Some(0)), simple_var("reads-cmp", Some(1))],
        numeric_variables: vec![
            NumericVariable::new("threshold".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(1.0), NumericValue::new(0.0)],
        operators: vec![],
        axioms: vec![PropositionalAxiom::new(
            vec![ExplicitFact::propositional(0, 0)],
            VariableIndex::from_usize(1),
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(1),
        )],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(1),
            VariableIndex::new(0),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

fn eff_eff_goal_join_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![simple_var("g1", None), simple_var("g2", None)],
        numeric_variables: vec![],
        goals: vec![
            ExplicitFact::propositional(0, 1),
            ExplicitFact::propositional(1, 1),
        ],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![Operator::new(
            "set-both".to_string(),
            vec![],
            vec![
                Effect::new(
                    vec![],
                    VariableIndex::new(0),
                    Some(ExplicitValueIndex::new(0)),
                    ExplicitValueIndex::new(1),
                ),
                Effect::new(
                    vec![],
                    VariableIndex::new(1),
                    Some(ExplicitValueIndex::new(0)),
                    ExplicitValueIndex::new(1),
                ),
            ],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

fn helper_goal_with_unsupported_numeric_effect_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![simple_var("cmp", Some(1)), simple_var("goal", None)],
        numeric_variables: vec![
            NumericVariable::new("const2".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
        ],
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(2.0),
            NumericValue::new(1.0),
            NumericValue::new(1.0),
            NumericValue::new(2.0),
        ],
        operators: vec![Operator::new(
            "scale-x".to_string(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                VariableIndex::from_usize(1),
                AssignmentOperation::Times,
                VariableIndex::from_usize(0),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        )],
        axioms: vec![PropositionalAxiom::new(
            vec![ExplicitFact::propositional(0, 0)],
            VariableIndex::from_usize(1),
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(1),
        )],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(3),
            VariableIndex::new(0),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(3),
            CalOperator::Sum,
            VariableIndex::from_usize(1),
            VariableIndex::from_usize(2),
        )],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
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

    assert!(collection.contains(&Pattern::new(vec![VariableIndex::from_usize(1)], vec![])));
    assert!(collection.contains(&Pattern::new(
        vec![VariableIndex::from_usize(0), VariableIndex::from_usize(1)],
        vec![]
    )));
}

#[test]
fn systematic_generator_returns_projectable_numeric_patterns() {
    let task = numeric_goal_task();
    let collection =
        generate_systematic_patterns(&task, SystematicPatternGeneratorConfig::default());

    assert!(collection.contains(&Pattern::new(vec![], vec![VariableIndex::from_usize(1)])));
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

    assert!(collection.contains(&Pattern::new(vec![VariableIndex::from_usize(0)], vec![])));
    assert!(collection.contains(&Pattern::new(vec![VariableIndex::from_usize(1)], vec![])));
    assert!(collection.contains(&Pattern::new(
        vec![VariableIndex::from_usize(0), VariableIndex::from_usize(1)],
        vec![]
    )));
}

#[test]
#[should_panic(expected = "systematic PDB patterns require a restricted task")]
fn systematic_generator_rejects_unrestricted_tasks() {
    let task = helper_goal_with_unsupported_numeric_effect_task();
    let _ = generate_systematic_patterns(
        &task,
        SystematicPatternGeneratorConfig {
            max_pattern_size: 2,
            ..SystematicPatternGeneratorConfig::default()
        },
    );
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
