use super::*;
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

/// A goal on a derived variable is refused, by name.
///
/// The goal is `derived=default`: the derived variable at the value its axioms
/// leave it at when none of them fires. No abstract operator writes a derived
/// variable, so an abstraction has nothing to reach for such a goal -- it used to
/// substitute the axiom body and get the answer right only for the shapes it had
/// been taught. `validate_abstractable_goal` refuses the task instead; blind,
/// `ff` and `lmcutnumeric` still solve it.
#[test]
fn a_derived_goal_is_refused_by_name() {
    let variables = vec![
        ExplicitVariable::new(
            2,
            "base".into(),
            vec!["on".into(), "off".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
        ExplicitVariable::new(
            2,
            "derived".into(),
            vec!["active".into(), "default".into()],
            Some(0),
            ExplicitValueIndex::new(1),
        ),
    ];
    let turn_off = Operator::new(
        "turn-off".into(),
        vec![],
        vec![Effect::new(
            vec![],
            VariableIndex::from_usize(0),
            None,
            ExplicitValueIndex::new(1),
        )],
        vec![],
        OperatorCost::new(1),
    );
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables,
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)],
        numeric_state: vec![],
        operators: vec![turn_off],
        axioms: vec![PropositionalAxiom::new(
            vec![ExplicitFact::propositional(0, 0)],
            VariableIndex::from_usize(1),
            ExplicitValueIndex::new(1),
            ExplicitValueIndex::new(0),
        )],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let error = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        cegar: CegarOptions {
            max_states: 64,
            ..CartesianAbstractionConfig::default().cegar
        },
        ..Default::default()
    })
    .unwrap()
    .generate(&task)
    .expect_err("a derived goal is not an abstractable goal");
    let message = format!("{error:#}");
    assert!(
        message.contains("abstractions support conjunctive goals only")
            && message.contains("derived"),
        "the refusal has to name what it refuses, got: {message}"
    );
}

#[test]
fn state_limit_returns_an_admissible_partial_snp_abstraction() {
    let variables = vec![comparison_variable("sum-at-limit")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("sum".into(), NumericType::Derived, None),
        NumericVariable::new("limit".into(), NumericType::Constant, None),
    ];
    let increment = Operator::new(
        "increment".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::from_usize(0),
            AssignmentOperation::Plus,
            VariableIndex::from_usize(1),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );
    let task = numeric_goal_task(
        variables,
        numeric_variables,
        vec![
            NumericValue::new(0.0),
            NumericValue::new(1.0),
            NumericValue::new(0.0),
            NumericValue::new(3.0),
        ],
        vec![increment],
        vec![ComparisonAxiom::new(
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(2),
            VariableIndex::from_usize(3),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(
            VariableIndex::from_usize(2),
            CalOperator::Sum,
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(1),
        )],
    );

    let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        cegar: CegarOptions {
            max_states: 1,
            ..CartesianAbstractionConfig::default().cegar
        },
        ..Default::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();
    let h = abstraction.distance_table.distances[abstraction.distance_table.initial_state];
    assert!(h <= 2.0);
    assert!(!abstraction.metadata.solved_by_self());
    assert_eq!(abstraction.metadata.stop_reason, CegarStopReason::SizeLimit);
    assert!(abstraction.metadata.pending_flaw.is_some());
    assert!(abstraction.abstract_source_sets.is_empty());
}
