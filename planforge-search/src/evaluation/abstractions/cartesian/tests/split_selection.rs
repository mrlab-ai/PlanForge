use super::*;

#[test]
fn operator_bitsets_preserve_exact_membership_and_intersections() {
    let mut left = OperatorBitSet::empty(130);
    let mut right = OperatorBitSet::empty(130);
    for operator_id in [0, 1, 63, 64, 65, 129] {
        assert!(left.insert(OperatorIndex::new(operator_id)));
    }
    assert!(!left.insert(OperatorIndex::new(64)));
    for operator_id in [1, 64, 127, 129] {
        assert!(right.insert(OperatorIndex::new(operator_id)));
    }

    assert_eq!(
        left.intersection_iter(&right).collect::<Vec<_>>(),
        vec![
            OperatorIndex::new(1),
            OperatorIndex::new(64),
            OperatorIndex::new(129)
        ]
    );
    let difference = left.clone_without(&right);
    for operator_id in [0, 63, 65] {
        assert!(difference.contains(OperatorIndex::new(operator_id)));
    }
    for operator_id in [1, 64, 129] {
        assert!(!difference.contains(OperatorIndex::new(operator_id)));
    }
}

#[test]
fn numeric_split_keys_include_semantic_identity_and_boundary() {
    let key = numeric_split_choice_key("x(b0)", NumericValue::new(1.0), true);
    assert_ne!(
        key,
        numeric_split_choice_key("x(b1)", NumericValue::new(1.0), true)
    );
    assert_ne!(
        key,
        numeric_split_choice_key("x(b0)", NumericValue::new(2.0), true)
    );
    assert_ne!(
        key,
        numeric_split_choice_key("x(b0)", NumericValue::new(1.0), false)
    );
}

#[test]
fn whole_plan_candidates_deduplicate_identical_refinements() {
    let split = Split::Numeric {
        state_id: 3,
        var_id: VariableIndex::from_usize(1),
        boundary: NumericValue::new(2.0),
        lower_includes_boundary: true,
        witness_value: NumericValue::new(1.0),
        integer_lattice: false,
        description: "first witness".into(),
    };
    let mut candidates = Vec::new();
    let mut identities = HashSet::new();
    push_unique_split(&mut candidates, &mut identities, split.clone());
    let mut duplicate = split;
    let Split::Numeric { description, .. } = &mut duplicate else {
        unreachable!()
    };
    *description = "same refinement seen later".into();
    push_unique_split(&mut candidates, &mut identities, duplicate);

    assert_eq!(candidates.len(), 1);
}

#[test]
fn min_growth_uses_projected_transition_count() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            2,
            "goal".into(),
            vec!["true".into(), "false".into()],
            None,
            ExplicitValueIndex::new(1),
        )],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(1.0),
        ],
        operators: vec![Operator::new(
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
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 1),
    });
    let semantics = CartesianSemantics::new(&task, &CartesianAbstractionConfig::default()).unwrap();
    let mut working = WorkingAbstraction::new(
        CartesianSet::with_all_props_constrained(
            vec![vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)]],
            vec![
                Interval::unbounded(),
                Interval::unbounded(),
                Interval::singleton(NumericValue::new(1.0)),
            ],
        ),
        1,
    );
    working.add_transition(0, OperatorIndex::new(0), 0);
    let split = |var_id| Split::Numeric {
        state_id: 0,
        var_id,
        boundary: NumericValue::new(0.0),
        lower_includes_boundary: true,
        witness_value: NumericValue::new(0.0),
        integer_lattice: false,
        description: String::new(),
    };

    let mut candidates = vec![split(VariableIndex::new(0)), split(VariableIndex::new(1))];
    let paths = all_goal_paths(working.states().len());
    let view = CartesianView {
        working: &working,
        semantics: &semantics,
        shortest_paths: &paths,
    };
    retain_min_growth_splits(&view, &mut candidates, |candidate| candidate).unwrap();
    assert_eq!(candidates.len(), 1);
    assert!(matches!(
        candidates[0],
        Split::Numeric { var_id, .. } if var_id == VariableIndex::new(1)
    ));
}

#[test]
fn native_random_selector_is_seeded() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![
            ExplicitVariable::new(
                2,
                "left".into(),
                vec!["zero".into(), "one".into()],
                None,
                ExplicitValueIndex::new(0),
            ),
            ExplicitVariable::new(
                2,
                "right".into(),
                vec!["zero".into(), "one".into()],
                None,
                ExplicitValueIndex::new(0),
            ),
        ],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let split = |var_id| Split::Propositional {
        state_id: 0,
        var_id,
        wanted: vec![ExplicitValueIndex::new(1)],
        witness_value: ExplicitValueIndex::new(0),
        description: String::new(),
    };
    let candidates = vec![
        split(VariableIndex::from_usize(0)),
        split(VariableIndex::from_usize(1)),
    ];

    let random_config = CartesianAbstractionConfig {
        split_selection: SplitSelection::Random,
        random_seed: Some(7),
        ..CartesianAbstractionConfig::default()
    };
    let random_a = CartesianSemantics::new(&task, &random_config).unwrap();
    let random_b = CartesianSemantics::new(&task, &random_config).unwrap();
    let working = WorkingAbstraction::new(
        CartesianSet::with_all_props_constrained(
            vec![
                vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)],
                vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)],
            ],
            vec![],
        ),
        0,
    );
    let paths = all_goal_paths(working.states().len());
    let draw = |semantics: &CartesianSemantics<'_>| {
        let view = CartesianView {
            working: &working,
            semantics,
            shortest_paths: &paths,
        };
        (0..32)
            .map(|_| {
                select_refinement_split(&view, semantics.cegar(), candidates.clone(), 0)
                    .unwrap()
                    .dimension()
            })
            .collect::<Vec<_>>()
    };
    let sequence = draw(&random_a);
    assert_eq!(sequence, draw(&random_b));
    assert!(sequence.contains(&SplitDimension::Propositional(VariableIndex::from_usize(0))));
    assert!(sequence.contains(&SplitDimension::Propositional(VariableIndex::from_usize(1))));
}
