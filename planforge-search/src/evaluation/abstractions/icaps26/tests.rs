use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex,
    ExplicitVariable, Metric, NEG_INF_VALUE, NumericRootTask, NumericRootTaskParts, NumericType,
    NumericValue, NumericVariable, ONE_VALUE, Operator, OperatorCost, OperatorIndex, VariableIndex,
};

use planforge_sas::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;

use super::*;
use crate::evaluation::abstractions::common::abstraction::CegarAbstraction;

#[test]
fn icaps_rng_matches_std_mt19937_uniform_integer_distribution() {
    let mut rng = ArtifactMt19937::new(2011);
    let sampled = [2, 3, 5, 7, 11, 100, 2, 17].map(|bound| rng.uniform_index(bound));
    assert_eq!(sampled, [1, 0, 2, 5, 4, 50, 1, 8]);
}

#[test]
fn icaps_numeric_splits_preserve_integer_lattices_without_dropping_continuous_values() {
    let (integer_lower, integer_upper) =
        numeric_split_intervals(Interval::unbounded(), NumericValue::new(0.0), false, true)
            .unwrap();
    assert!(integer_lower.contains(NumericValue::new(-1.0)));
    assert!(integer_upper.contains(NumericValue::new(0.0)));
    assert!(!integer_lower.contains(NumericValue::new(-0.5)));
    assert!(!integer_upper.contains(NumericValue::new(-0.5)));

    let (continuous_lower, continuous_upper) =
        numeric_split_intervals(Interval::unbounded(), NumericValue::new(0.0), false, false)
            .unwrap();
    assert!(continuous_lower.contains(NumericValue::new(-0.5)));
    assert!(!continuous_lower.contains(NumericValue::new(0.0)));
    assert!(continuous_upper.contains(NumericValue::new(0.0)));

    let integer_parent = Interval::new(NEG_INF_VALUE, ONE_VALUE, false, false);
    assert!(
        numeric_split_intervals(integer_parent, ZERO_VALUE, true, true).is_err(),
        "an integer split must reject a child containing only values in (0, 1)"
    );
}

#[test]
fn icaps_prevail_conditions_fix_the_post_value() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            2,
            "position".into(),
            vec!["left".into(), "right".into()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![Operator::new(
            "prevail-left".into(),
            vec![ExplicitFact::propositional(0, 0)],
            vec![],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let source = CartesianSet::with_all_props_constrained(
        vec![vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)]],
        vec![],
    );
    let target =
        CartesianSet::with_all_props_constrained(vec![vec![ExplicitValueIndex::new(1)]], vec![]);

    let icaps = CartesianSemantics::new(
        &task,
        &Icaps26CartesianConfig {
            pick: Icaps26SplitSelection::MinUnwanted,
            ..Icaps26CartesianConfig::default()
        },
    )
    .unwrap();
    assert!(
        !icaps
            .may_transition(&source, OperatorIndex::new(0), &target)
            .unwrap()
    );
}

#[test]
fn icaps_split_preserves_artifact_loop_and_arc_order() {
    let operators = vec![
        Operator::new(
            "independent".into(),
            vec![],
            vec![],
            vec![],
            OperatorCost::new(1),
        ),
        Operator::new(
            "set-right".into(),
            vec![],
            vec![Effect::new(
                vec![],
                VariableIndex::from_usize(0),
                None,
                ExplicitValueIndex::new(1),
            )],
            vec![],
            OperatorCost::new(1),
        ),
        Operator::new(
            "set-left".into(),
            vec![],
            vec![Effect::new(
                vec![],
                VariableIndex::from_usize(0),
                None,
                ExplicitValueIndex::new(0),
            )],
            vec![],
            OperatorCost::new(1),
        ),
        Operator::new(
            "prevail-left".into(),
            vec![ExplicitFact::propositional(0, 0)],
            vec![],
            vec![],
            OperatorCost::new(1),
        ),
        Operator::new(
            "prevail-right".into(),
            vec![ExplicitFact::propositional(0, 1)],
            vec![],
            vec![],
            OperatorCost::new(1),
        ),
    ];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            2,
            "position".into(),
            vec!["left".into(), "right".into()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators,
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let semantics = CartesianSemantics::new(
        &task,
        &Icaps26CartesianConfig {
            pick: Icaps26SplitSelection::MinUnwanted,
            ..Icaps26CartesianConfig::default()
        },
    )
    .unwrap();
    let mut working = WorkingAbstraction::new_icaps26(
        CartesianSet::with_all_props_constrained(
            vec![vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)]],
            vec![],
        ),
        5,
    );
    for op_id in 0..5 {
        working.add_transition(0, OperatorIndex::new(op_id), 0);
    }

    working
        .apply_split(
            &semantics,
            Split::Propositional {
                state_id: 0,
                var_id: VariableIndex::from_usize(0),
                wanted: vec![ExplicitValueIndex::new(1)],
                witness_value: ExplicitValueIndex::new(0),
                description: String::new(),
            },
        )
        .unwrap();

    assert_eq!(
        working.icaps_self_loop_order.as_ref().unwrap(),
        &vec![
            vec![
                OperatorIndex::new(0),
                OperatorIndex::new(2),
                OperatorIndex::new(3)
            ],
            vec![
                OperatorIndex::new(0),
                OperatorIndex::new(1),
                OperatorIndex::new(4)
            ]
        ]
    );
    let arcs = working
        .active_transition_ids()
        .map(|id| {
            let transition = working.transition(id);
            (
                transition.source,
                transition.concrete_op_id,
                transition.target,
            )
        })
        .collect::<HashSet<_>>();
    assert_eq!(
        arcs,
        HashSet::from([(0, OperatorIndex::new(1), 1), (1, OperatorIndex::new(2), 0)])
    );
}

#[test]
fn icaps26_unwanted_score_counts_excluded_values_and_penalizes_open_desired_tails() {
    let working = WorkingAbstraction::new_icaps26(
        CartesianSet::with_all_props_constrained(
            vec![vec![
                ExplicitValueIndex::new(0),
                ExplicitValueIndex::new(1),
                ExplicitValueIndex::new(2),
                ExplicitValueIndex::new(3),
            ]],
            vec![Interval::new(
                NumericValue::new(-10.0),
                NumericValue::new(10.0),
                true,
                true,
            )],
        ),
        0,
    );
    let propositional = Split::Propositional {
        state_id: 0,
        var_id: VariableIndex::from_usize(0),
        wanted: vec![ExplicitValueIndex::new(2), ExplicitValueIndex::new(3)],
        witness_value: ExplicitValueIndex::new(0),
        description: String::new(),
    };
    assert_eq!(
        artifact_unwanted_score(&working, &propositional).unwrap(),
        2.0
    );

    let finite_numeric = Split::Numeric {
        state_id: 0,
        var_id: VariableIndex::new(0),
        boundary: NumericValue::new(4.0),
        lower_includes_boundary: false,
        witness_value: NumericValue::new(8.0),
        desired_contains_witness: false,
        integer_lattice: false,
        description: String::new(),
    };
    assert_eq!(
        artifact_unwanted_score(&working, &finite_numeric).unwrap(),
        7.0
    );
    let mut witness_is_desired = finite_numeric.clone();
    let Split::Numeric {
        desired_contains_witness,
        ..
    } = &mut witness_is_desired
    else {
        unreachable!()
    };
    *desired_contains_witness = true;
    assert_eq!(
        artifact_unwanted_score(&working, &witness_is_desired).unwrap(),
        14.0
    );

    let open_tail = WorkingAbstraction::new_icaps26(
        CartesianSet::with_all_props_constrained(
            vec![vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)]],
            vec![Interval::unbounded()],
        ),
        0,
    );
    let desired_open_tail = Split::Numeric {
        state_id: 0,
        var_id: VariableIndex::new(0),
        boundary: NumericValue::new(4.0),
        lower_includes_boundary: false,
        witness_value: NumericValue::new(8.0),
        desired_contains_witness: false,
        integer_lattice: false,
        description: String::new(),
    };
    assert!(
        artifact_unwanted_score(&open_tail, &desired_open_tail)
            .unwrap()
            .is_infinite()
    );

    let fractional = WorkingAbstraction::new_icaps26(
        CartesianSet::with_all_props_constrained(
            vec![vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)]],
            vec![Interval::new(
                NumericValue::new(0.0),
                NumericValue::new(0.5),
                true,
                true,
            )],
        ),
        0,
    );
    let fractional_split = Split::Numeric {
        state_id: 0,
        var_id: VariableIndex::new(0),
        boundary: NumericValue::new(0.25),
        lower_includes_boundary: true,
        witness_value: NumericValue::new(0.0),
        desired_contains_witness: true,
        integer_lattice: false,
        description: String::new(),
    };
    assert_eq!(
        artifact_unwanted_score(&fractional, &fractional_split).unwrap(),
        0.25
    );

    let open_integer_interval = WorkingAbstraction::new_icaps26(
        CartesianSet::with_all_props_constrained(
            vec![vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)]],
            vec![Interval::new(
                NumericValue::new(5.0),
                NumericValue::new(10.0),
                false,
                true,
            )],
        ),
        0,
    );
    let open_integer_split = Split::Numeric {
        state_id: 0,
        var_id: VariableIndex::new(0),
        boundary: NumericValue::new(7.0),
        lower_includes_boundary: false,
        witness_value: NumericValue::new(6.0),
        desired_contains_witness: false,
        integer_lattice: false,
        description: String::new(),
    };
    assert_eq!(
        artifact_unwanted_score(&open_integer_interval, &open_integer_split).unwrap(),
        1.0
    );
}

#[test]
fn icaps26_selector_uses_unwanted_values_without_native_growth_filtering() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            4,
            "location".into(),
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            None,
            ExplicitValueIndex::new(3),
        )],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 3),
    });
    let mut config = Icaps26CartesianConfig {
        pick: Icaps26SplitSelection::MinUnwanted,
        ..Icaps26CartesianConfig::default()
    };
    let mut semantics = CartesianSemantics::new(&task, &config).unwrap();
    let working = WorkingAbstraction::new_icaps26(
        CartesianSet::with_all_props_constrained(
            vec![vec![
                ExplicitValueIndex::new(0),
                ExplicitValueIndex::new(1),
                ExplicitValueIndex::new(2),
                ExplicitValueIndex::new(3),
            ]],
            semantics.trivial_region().unwrap().numeric.to_vec(),
        ),
        0,
    );
    let split = |wanted| Split::Propositional {
        state_id: 0,
        var_id: VariableIndex::from_usize(0),
        wanted,
        witness_value: ExplicitValueIndex::new(0),
        description: String::new(),
    };
    let candidates = vec![
        split(vec![ExplicitValueIndex::new(1)]),
        split(vec![
            ExplicitValueIndex::new(1),
            ExplicitValueIndex::new(2),
            ExplicitValueIndex::new(3),
        ]),
    ];
    let selected = select_refinement_split(&working, &semantics, candidates.clone(), 0).unwrap();
    assert!(matches!(selected, Split::Propositional { wanted, .. } if wanted.len() == 3));

    config.pick = Icaps26SplitSelection::MaxUnwanted;
    semantics.pick = config.pick;
    let selected = select_refinement_split(&working, &semantics, candidates, 0).unwrap();
    assert!(matches!(selected, Split::Propositional { wanted, .. } if wanted.len() == 1));

    config.pick = Icaps26SplitSelection::Random;
    config.random_seed = 2011;
    let random_a = CartesianSemantics::new(&task, &config).unwrap();
    let random_b = CartesianSemantics::new(&task, &config).unwrap();
    let draw = |semantics: &CartesianSemantics<'_>| {
        (0..32)
            .map(|_| {
                let selected = select_refinement_split(
                    &working,
                    semantics,
                    vec![
                        split(vec![ExplicitValueIndex::new(1)]),
                        split(vec![
                            ExplicitValueIndex::new(1),
                            ExplicitValueIndex::new(2),
                            ExplicitValueIndex::new(3),
                        ]),
                    ],
                    0,
                )
                .unwrap();
                match selected {
                    Split::Propositional { wanted, .. } => wanted.len(),
                    Split::Numeric { .. } => unreachable!(),
                }
            })
            .collect::<Vec<_>>()
    };
    let sequence = draw(&random_a);
    assert_eq!(sequence, draw(&random_b));
    assert!(sequence.contains(&1) && sequence.contains(&3));
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

    let abstraction = Icaps26CartesianGenerator::new(Icaps26CartesianConfig {
        max_states: 32,
        random_seed: 1,
        pick: Icaps26SplitSelection::Random,
        ..Icaps26CartesianConfig::default()
    })
    .unwrap()
    .generate(&task)
    .unwrap();

    assert_eq!(abstraction.initial_h(), 1.0);
    assert!(abstraction.metadata().solved_by_self());
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
    let error = Icaps26CartesianGenerator::new(Icaps26CartesianConfig {
        max_states: 32,
        random_seed: 1,
        pick: Icaps26SplitSelection::Random,
        ..Icaps26CartesianConfig::default()
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

fn comparison_variable(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        name.into(),
        vec!["true".into(), "false".into()],
        Some(0),
        ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
    )
}

fn numeric_goal_task(
    variables: Vec<ExplicitVariable>,
    numeric_variables: Vec<NumericVariable>,
    initial_numeric: Vec<NumericValue>,
    operators: Vec<Operator>,
    comparison_axioms: Vec<ComparisonAxiom>,
    assignment_axioms: Vec<AssignmentAxiom>,
) -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2)],
        numeric_state: initial_numeric,
        operators,
        axioms: vec![],
        comparison_axioms,
        assignment_axioms,
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}
