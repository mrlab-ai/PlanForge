use super::*;

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
        max_states: 16,
        ..Default::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();

    assert_eq!(
        abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash],
        2.0
    );
    assert!(abstraction.metadata.solved_by_self);
    assert_eq!(
        abstraction.metadata.stop_reason,
        CartesianStopReason::ConcretePlan
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
fn desired_region_uses_additive_snp_coordinate() {
    let variables = vec![comparison_variable("sum-at-four")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("sum".into(), NumericType::Derived, None),
        NumericVariable::new("four".into(), NumericType::Constant, None),
    ];
    let increment = |name: &str, var_id| {
        Operator::new(
            name.into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                var_id,
                AssignmentOperation::Plus,
                VariableIndex::from_usize(2),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        )
    };
    let task = numeric_goal_task(
        variables,
        numeric_variables,
        // The stored value of the derived sum is intentionally stale. The
        // Cartesian initial-state hash must use the axiom-evaluated value 3.
        vec![
            NumericValue::new(2.0),
            NumericValue::new(1.0),
            NumericValue::new(1.0),
            NumericValue::new(0.0),
            NumericValue::new(4.0),
        ],
        vec![
            increment("increment-x", VariableIndex::from_usize(0)),
            increment("increment-y", VariableIndex::from_usize(1)),
        ],
        vec![ComparisonAxiom::new(
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(3),
            VariableIndex::from_usize(4),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(
            VariableIndex::from_usize(3),
            CalOperator::Sum,
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(1),
        )],
    );

    for abstract_plan_selection in [
        CartesianAbstractPlanSelection::BackwardShortestPath,
        CartesianAbstractPlanSelection::StableAStar,
    ] {
        let abstraction = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
            max_states: 32,
            random_seed: Some(1),
            refinement_direction: CartesianRefinementDirection::Regression,
            abstract_plan_selection,
            flaw_candidate_generation: CartesianFlawCandidateGeneration::DesiredRegion,
            split_selection: CartesianSplitSelection::Random,
            ..Default::default()
        })
        .unwrap()
        .generate(&task)
        .unwrap();

        assert_eq!(
            abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash],
            1.0
        );
        assert!(abstraction.metadata.solved_by_self);
    }
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
fn desired_region_rejects_non_additive_snp_coordinate() {
    let variables = vec![comparison_variable("product-at-nine")];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("product".into(), NumericType::Derived, None),
        NumericVariable::new("nine".into(), NumericType::Constant, None),
    ];
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
        vec![Operator::new(
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
        )],
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
    let error = CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
        max_states: 32,
        random_seed: Some(1),
        refinement_direction: CartesianRefinementDirection::Regression,
        abstract_plan_selection: CartesianAbstractPlanSelection::StableAStar,
        flaw_candidate_generation: CartesianFlawCandidateGeneration::DesiredRegion,
        split_selection: CartesianSplitSelection::Random,
        ..Default::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap_err();

    let diagnostic = format!("{error:#}");
    assert!(
        diagnostic.contains("exact additive coordinate"),
        "unexpected diagnostic: {diagnostic}"
    );
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
fn exact_affine_preimages_preserve_open_boundaries() {
    let plus = affine_effect_task(AssignmentOperation::Plus, 1.0);
    let plus_semantics =
        CartesianSemantics::new(&plus, &CartesianAbstractionConfig::default()).unwrap();
    assert_eq!(
        plus_semantics
            .numeric_effect_preimage(
                Interval::new(NumericValue::new(5.0), NumericValue::new(10.0), false, true),
                OperatorIndex::new(0),
                VariableIndex::new(0)
            )
            .unwrap()
            .unwrap(),
        Interval::new(NumericValue::new(4.0), NumericValue::new(9.0), false, true)
    );

    let times = affine_effect_task(AssignmentOperation::Times, -2.0);
    let times_semantics =
        CartesianSemantics::new(&times, &CartesianAbstractionConfig::default()).unwrap();
    assert_eq!(
        times_semantics
            .numeric_effect_preimage(
                Interval::new(
                    NumericValue::new(-10.0),
                    NumericValue::new(-4.0),
                    false,
                    true
                ),
                OperatorIndex::new(0),
                VariableIndex::new(0)
            )
            .unwrap()
            .unwrap(),
        Interval::new(NumericValue::new(2.0), NumericValue::new(5.0), true, false)
    );

    let divide = affine_effect_task(AssignmentOperation::Divide, -2.0);
    let divide_semantics =
        CartesianSemantics::new(&divide, &CartesianAbstractionConfig::default()).unwrap();
    assert_eq!(
        divide_semantics
            .numeric_effect_preimage(
                Interval::new(
                    NumericValue::new(-5.0),
                    NumericValue::new(-2.0),
                    false,
                    true
                ),
                OperatorIndex::new(0),
                VariableIndex::new(0)
            )
            .unwrap()
            .unwrap(),
        Interval::new(NumericValue::new(4.0), NumericValue::new(10.0), true, false)
    );
}

#[test]
fn assignment_preimage_is_universal_exactly_when_target_contains_rhs() {
    let task = affine_effect_task(AssignmentOperation::Assign, 3.0);
    let semantics = CartesianSemantics::new(&task, &CartesianAbstractionConfig::default()).unwrap();
    assert_eq!(
        semantics
            .numeric_effect_preimage(
                Interval::new(NumericValue::new(2.0), NumericValue::new(3.0), true, true),
                OperatorIndex::new(0),
                VariableIndex::new(0)
            )
            .unwrap()
            .unwrap(),
        Interval::unbounded()
    );
    assert!(
        semantics
            .numeric_effect_preimage(
                Interval::new(NumericValue::new(2.0), NumericValue::new(3.0), true, false),
                OperatorIndex::new(0),
                VariableIndex::new(0)
            )
            .unwrap()
            .is_none()
    );
}

#[test]
fn assignment_outside_target_is_not_a_cartesian_transition() {
    let task = affine_effect_task(AssignmentOperation::Assign, 3.0);
    let semantics = CartesianSemantics::new(&task, &CartesianAbstractionConfig::default()).unwrap();
    let source = semantics.trivial_region().unwrap();
    let mut target = source.clone();
    Arc::make_mut(&mut target.numeric)[0] =
        Interval::new(NumericValue::new(2.0), NumericValue::new(3.0), true, false);

    assert!(
        !semantics
            .may_transition(&source, OperatorIndex::new(0), &target)
            .unwrap()
    );
    assert!(
        semantics
            .source_set_source_for_transition(&source, OperatorIndex::new(0), &target)
            .unwrap()
            .is_none()
    );
}
