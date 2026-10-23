use super::*;
use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;

/// A goal on a comparison variable is refined until the comparison holds.
///
/// `x-at-limit` is the condition variable of `x >= limit`, which is what the
/// translator emits for the numeric goal `(>= (x) (limit))`, and reaching it
/// takes two increments. Nothing about the goal is propositional, so the
/// refinement has to come from the comparison's operands.
#[test]
fn refines_through_a_comparison_goal() {
    let variables = vec![ExplicitVariable::new(
        2,
        "x-at-limit".into(),
        vec!["true".into(), "false".into()],
        Some(0),
        ExplicitValueIndex::new(1),
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("limit".into(), NumericType::Constant, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
    ];
    let increment = Operator::new(
        "increment".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::from_usize(0),
            AssignmentOperation::Plus,
            VariableIndex::from_usize(2),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(1)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(2.0),
            NumericValue::new(1.0),
        ],
        operators: vec![increment],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(1),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        cegar: CegarOptions {
            max_states: 16,
            ..CartesianAbstractionConfig::default().cegar
        },
    })
    .unwrap()
    .generate(&task)
    .unwrap();

    assert_eq!(abstraction.initial_h(), 2.0);
    assert!(abstraction.metadata().solved_by_self());
    assert_eq!(
        abstraction.metadata().stop_reason,
        CegarStopReason::ConcretePlan
    );
}

#[test]
fn supports_snp_assignment_axiom_comparisons() {
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

    assert_solved_with_h(&task, 2.0);
}

#[test]
fn supports_nonlinear_snp_assignment_axiom_comparisons() {
    let variables = vec![comparison_variable("product-at-nine")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("product".into(), NumericType::Derived, None),
        NumericVariable::new("nine".into(), NumericType::Constant, None),
    ];
    let increment_both = Operator::new(
        "increment-both".into(),
        vec![],
        vec![],
        vec![
            AssignmentEffect::new(
                VariableIndex::new(0),
                AssignmentOperation::Plus,
                VariableIndex::new(2),
                false,
                vec![],
            ),
            AssignmentEffect::new(
                VariableIndex::new(1),
                AssignmentOperation::Plus,
                VariableIndex::new(2),
                false,
                vec![],
            ),
        ],
        OperatorCost::new(1),
    );
    let task = numeric_goal_task(
        variables,
        numeric_variables,
        vec![
            NumericValue::new(1.0),
            NumericValue::new(1.0),
            NumericValue::new(1.0),
            NumericValue::new(0.0),
            NumericValue::new(9.0),
        ],
        vec![increment_both],
        vec![ComparisonAxiom::new(
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(3),
            VariableIndex::from_usize(4),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(
            VariableIndex::from_usize(3),
            CalOperator::Product,
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(1),
        )],
    );

    assert_solved_with_h(&task, 2.0);
}

#[test]
fn supports_comparisons_between_regular_variables() {
    let variables = vec![comparison_variable("x-at-y")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
    ];
    let increment = Operator::new(
        "increment-x".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::from_usize(0),
            AssignmentOperation::Plus,
            VariableIndex::from_usize(2),
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
            NumericValue::new(2.0),
            NumericValue::new(1.0),
        ],
        vec![increment],
        vec![ComparisonAxiom::new(
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(1),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
    );

    assert_solved_with_h(&task, 2.0);
}

#[test]
fn supports_multiplicative_numeric_effects() {
    let variables = vec![comparison_variable("x-at-four")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("two".into(), NumericType::Constant, None),
        NumericVariable::new("four".into(), NumericType::Constant, None),
    ];
    let double = Operator::new(
        "double".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::from_usize(0),
            AssignmentOperation::Times,
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
            NumericValue::new(1.0),
            NumericValue::new(2.0),
            NumericValue::new(4.0),
        ],
        vec![double],
        vec![ComparisonAxiom::new(
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(2),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
    );

    assert_solved_with_h(&task, 2.0);
}

fn affine_effect_task(operation: AssignmentOperation, rhs: f64) -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(false, None),
        variables: vec![ExplicitVariable::new(
            1,
            "dummy".into(),
            vec!["value".into()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("rhs".into(), NumericType::Constant, None),
        ],
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(rhs)],
        operators: vec![Operator::new(
            "affine".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                VariableIndex::from_usize(0),
                operation,
                VariableIndex::from_usize(1),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

#[test]
fn canonical_preimages_respect_open_targets() {
    for (operation, operand, lower, upper, source_bounds) in [
        (AssignmentOperation::Plus, 1.0, 5.0, 10.0, [4.0, 9.0]),
        (AssignmentOperation::Times, -2.0, -10.0, -4.0, [2.0, 5.0]),
        (AssignmentOperation::Divide, -2.0, -5.0, -2.0, [4.0, 10.0]),
    ] {
        let task = affine_effect_task(operation.clone(), operand);
        let semantics =
            CartesianSemantics::new(&task, &CartesianAbstractionConfig::default()).unwrap();
        let target = Interval::new(
            NumericValue::new(lower),
            NumericValue::new(upper),
            false,
            true,
        );
        let preimage = semantics
            .cegar()
            .coordinates()
            .preimage(target, OperatorIndex::new(0), VariableIndex::new(0))
            .unwrap();
        for bound in source_bounds {
            for offset in [-2e-12, -1e-12, 0.0, 1e-12, 2e-12] {
                let source = NumericValue::new(bound + offset);
                assert_eq!(
                    preimage.contains(source),
                    target.contains(AssignmentOperation::apply(
                        source,
                        &operation,
                        NumericValue::new(operand)
                    )),
                    "{operation:?} {source}"
                );
            }
        }
    }
}

#[test]
fn assignment_preimage_is_universal_exactly_when_target_contains_rhs() {
    let task = affine_effect_task(AssignmentOperation::Assign, 3.0);
    let semantics = CartesianSemantics::new(&task, &CartesianAbstractionConfig::default()).unwrap();
    assert_eq!(
        semantics
            .cegar()
            .coordinates()
            .preimage(
                Interval::new(NumericValue::new(2.0), NumericValue::new(3.0), true, true),
                OperatorIndex::new(0),
                VariableIndex::new(0)
            )
            .unwrap(),
        Interval::unbounded()
    );
    assert!(
        semantics
            .cegar()
            .coordinates()
            .preimage(
                Interval::new(NumericValue::new(2.0), NumericValue::new(3.0), true, false),
                OperatorIndex::new(0),
                VariableIndex::new(0)
            )
            .is_none()
    );
}

#[test]
fn assignment_outside_target_is_not_a_cartesian_transition() {
    let task = affine_effect_task(AssignmentOperation::Assign, 3.0);
    let semantics = CartesianSemantics::new(&task, &CartesianAbstractionConfig::default()).unwrap();
    let source = semantics.regions().trivial_region().unwrap();
    let mut target = source.clone();
    Arc::make_mut(&mut target.numeric)[0] =
        Interval::new(NumericValue::new(2.0), NumericValue::new(3.0), true, false);

    assert!(
        !semantics
            .regions()
            .may_transition(&source, OperatorIndex::new(0), &target)
            .unwrap()
    );
    assert!(
        semantics
            .regions()
            .transition_source_set(&source, OperatorIndex::new(0), &target)
            .unwrap()
            .is_none()
    );
}
