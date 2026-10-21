use super::*;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;
use crate::evaluation::abstractions::common::cegar::semantics::CegarSemantics;

use planforge_sas::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitVariable, INF_VALUE, Metric,
    NEG_INF_VALUE, NumericRootTask, NumericRootTaskParts, NumericType, NumericVariable, Operator,
    OperatorCost, VariableIndex,
};

use crate::evaluation::abstractions::domain::initial_splits::identity_domain_mapping_and_sizes;
use crate::evaluation::abstractions::domain::partitions::NumericPartitions;
use planforge_sas::utils::interval::Interval;

/// The abstraction of `task` that keeps every propositional value apart and
/// partitions numeric variables by `partitions`.
fn identity_factory(
    task: &NumericRootTask,
    partitions: NumericPartitions,
) -> DomainAbstractionFactory {
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(task).unwrap();
    DomainAbstractionFactory::new(
        &CegarSemantics::for_options(task, &CegarOptions::default()).unwrap(),
        domain_mapping,
        domain_sizes,
        partitions,
    )
    .unwrap()
}

/// Why the CEGAR semantics rejects `task`.
fn semantics_error(task: &NumericRootTask) -> anyhow::Error {
    match CegarSemantics::for_options(task, &CegarOptions::default()) {
        Ok(_) => panic!("the CEGAR semantics accepted the task"),
        Err(error) => error,
    }
}

/// The abstract operators of [`identity_factory`].
fn identity_operators(
    task: &NumericRootTask,
    partitions: NumericPartitions,
    combine_labels: bool,
) -> Result<Vec<AbstractOperator>> {
    identity_factory(task, partitions).abstract_operators(task, combine_labels, Deadline::NONE)
}

/// What identifies an abstract operator for a determinism comparison: its
/// regression offset plus its progression and regression preconditions.
type AbstractOperatorSignature = (isize, Vec<ExplicitFact>, Vec<ExplicitFact>);

fn global_constraint_variable() -> ExplicitVariable {
    ExplicitVariable::new(
        1,
        "global-constraint".into(),
        vec!["true".into()],
        None,
        ExplicitValueIndex::new(0),
    )
}

#[test]
fn propositional_effect_without_precondition_preserves_sibling_branches() {
    let variables = vec![
        ExplicitVariable::new(
            2,
            "changed".into(),
            vec!["c0".into(), "c1".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
        ExplicitVariable::new(
            2,
            "prevail".into(),
            vec!["p0".into(), "p1".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
    ];
    let op = Operator::new(
        "set_changed".into(),
        vec![ExplicitFact::propositional(1, 0)],
        vec![planforge_sas::numeric_task::Effect::new(
            vec![],
            VariableIndex::from_usize(0),
            None,
            ExplicitValueIndex::new(0),
        )],
        vec![],
        OperatorCost::new(1),
    );
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(1), ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let partitions = NumericPartitions::trivial(&task);
    let operators = identity_operators(&task, partitions, true).unwrap();

    assert!(
        operators.iter().any(|op| {
            op.preconditions
                == vec![
                    ExplicitFact::propositional(0, 1),
                    ExplicitFact::propositional(1, 0),
                ]
                && op.regression_preconditions
                    == vec![
                        ExplicitFact::propositional(0, 0),
                        ExplicitFact::propositional(1, 0),
                    ]
        }),
        "expected the branch changing changed=1 to changed=0 to survive"
    );
}

#[test]
fn numeric_partition_transitions_and_comparison_filtering() {
    // Propositional var 0 is the derived comparison var for (x0 < c10).
    let variables = vec![ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        "cmp".into(),
        vec!["true".into(), "false".into()],
        Some(0),
        ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
    )];

    // Numeric vars: x0 (regular), c10 (constant), c7 (constant)
    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c10".into(), NumericType::Constant, None),
        NumericVariable::new("c7".into(), NumericType::Constant, None),
    ];

    // Comparison axiom (derived propositional var 0): x0 < c10
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(0),
        VariableIndex::new(1),
        ComparisonOperator::LessThan,
    )];

    // Operator requires comparison to hold (var0 == 0) and applies x0 += c7.
    let op = Operator::new(
        "op".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::new(0),
            AssignmentOperation::Plus,
            VariableIndex::new(2),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(10.0),
            NumericValue::new(7.0),
        ],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms,
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    // Partitions for x0: (-inf,10) and [10,inf)
    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(NEG_INF_VALUE, NumericValue::new(10.0), false, false),
            Interval::new(NumericValue::new(10.0), INF_VALUE, true, false),
        ],
        vec![Interval::singleton(NumericValue::new(10.0))],
        vec![Interval::singleton(NumericValue::new(7.0))],
    ]);

    let abs_ops = identity_operators(&task, partitions, false).unwrap();

    // From x0 in (-inf,10), x0 += 7 can stay in partition 0 or move to 1.
    // From [10,inf), the precondition x0 < 10 is definitely contradicted and filtered.
    assert_eq!(abs_ops.len(), 2);

    // The comparison variable is never refined, so an operator either keeps
    // x0's partition or advances it, which is worth 1.
    let mut regression_offsets: Vec<isize> = abs_ops.iter().map(|o| o.regression_offset).collect();
    regression_offsets.sort();
    assert_eq!(regression_offsets, vec![-1, 0]);
}

#[test]
fn repeated_numeric_operator_generation_is_deterministic() {
    let variables = vec![ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        "cmp".into(),
        vec!["true".into(), "false".into()],
        Some(0),
        ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
    )];

    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c10".into(), NumericType::Constant, None),
        NumericVariable::new("c7".into(), NumericType::Constant, None),
    ];

    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(0),
        VariableIndex::new(1),
        ComparisonOperator::LessThan,
    )];

    let op = Operator::new(
        "op".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::new(0),
            AssignmentOperation::Plus,
            VariableIndex::new(2),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(10.0),
            NumericValue::new(7.0),
        ],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms,
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(NEG_INF_VALUE, NumericValue::new(10.0), false, false),
            Interval::new(NumericValue::new(10.0), INF_VALUE, true, false),
        ],
        vec![Interval::singleton(NumericValue::new(10.0))],
        vec![Interval::singleton(NumericValue::new(7.0))],
    ]);

    let mut signatures: Vec<Vec<AbstractOperatorSignature>> = Vec::new();
    for _ in 0..12 {
        let ops = identity_operators(&task, partitions.clone(), false).unwrap();
        signatures.push(
            ops.iter()
                .map(|op| {
                    (
                        op.regression_offset,
                        op.preconditions.clone(),
                        op.regression_preconditions.clone(),
                    )
                })
                .collect(),
        );
    }

    for sig in signatures.iter().skip(1) {
        assert_eq!(sig, &signatures[0]);
    }
}

#[test]
fn affected_numeric_var_stays_marked_changed_with_identity_partition_transition() {
    let variables = vec![global_constraint_variable()];

    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
    ];

    let op = Operator::new(
        "op".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::new(0),
            AssignmentOperation::Plus,
            VariableIndex::new(1),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(1.0)],
        operators: vec![op.clone()],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(NEG_INF_VALUE, NumericValue::new(10.0), false, false),
            Interval::new(NumericValue::new(10.0), INF_VALUE, true, false),
        ],
        vec![Interval::singleton(NumericValue::new(1.0))],
    ]);

    let factory = identity_factory(&task, partitions);
    let generator = AbstractOperatorGenerator::new(&factory, &task);

    let transitions = compute_hash_effects_with_preconditions(
        &task,
        &generator,
        OperatorIndex::new(0),
        &[],
        op.assignment_effects(),
        Deadline::NONE,
    )
    .unwrap();

    let identity_transition = transitions.iter().find(|trans| {
        trans.source_partition_facts == vec![ExplicitFact::numeric_variable(1, 0)]
            && trans.target_partition_facts == vec![ExplicitFact::numeric_variable(1, 0)]
    });
    assert!(identity_transition.is_some());
    assert_eq!(
        identity_transition.unwrap().changed_numeric_vars,
        vec![VariableIndex::from_usize(0)]
    );
}

#[test]
fn additive_derived_numeric_partitions_are_materialized_in_transitions() {
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
        NumericVariable::new("c5".into(), NumericType::Constant, None),
        NumericVariable::new("d1".into(), NumericType::Derived, None),
        NumericVariable::new("d2".into(), NumericType::Derived, None),
    ];

    let assignment_axioms = vec![
        AssignmentAxiom::new(
            VariableIndex::from_usize(3),
            CalOperator::Sum,
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(1),
        ),
        AssignmentAxiom::new(
            VariableIndex::from_usize(4),
            CalOperator::Difference,
            VariableIndex::from_usize(3),
            VariableIndex::from_usize(2),
        ),
    ];

    let op = Operator::new(
        "inc".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::new(0),
            AssignmentOperation::Plus,
            VariableIndex::new(1),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![global_constraint_variable()],
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(3.0),
            NumericValue::new(1.0),
            NumericValue::new(5.0),
            NumericValue::new(4.0),
            NumericValue::new(-1.0),
        ],
        operators: vec![op.clone()],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms,
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::singleton(NumericValue::new(3.0)),
            Interval::singleton(NumericValue::new(4.0)),
        ],
        vec![Interval::singleton(NumericValue::new(1.0))],
        vec![Interval::singleton(NumericValue::new(5.0))],
        vec![
            Interval::singleton(NumericValue::new(4.0)),
            Interval::singleton(NumericValue::new(5.0)),
        ],
        vec![
            Interval::singleton(NumericValue::new(-1.0)),
            Interval::singleton(NumericValue::new(0.0)),
        ],
    ]);

    let factory = identity_factory(&task, partitions);
    let generator = AbstractOperatorGenerator::new(&factory, &task);

    let transitions = compute_hash_effects_with_preconditions(
        &task,
        &generator,
        OperatorIndex::new(0),
        &[],
        op.assignment_effects(),
        Deadline::NONE,
    )
    .unwrap();

    assert!(transitions.iter().any(|trans| {
        trans
            .source_partition_facts
            .contains(&ExplicitFact::numeric_variable(1, 0))
            && trans
                .target_partition_facts
                .contains(&ExplicitFact::numeric_variable(1, 1))
    }));
    assert!(transitions.iter().all(|trans| {
        trans
            .source_partition_facts
            .iter()
            .any(|fact| fact.var() == 4)
            && trans
                .target_partition_facts
                .iter()
                .any(|fact| fact.var() == 4)
            && trans
                .source_partition_facts
                .iter()
                .any(|fact| fact.var() == 5)
            && trans
                .target_partition_facts
                .iter()
                .any(|fact| fact.var() == 5)
            && trans.changed_numeric_vars
                == vec![
                    VariableIndex::from_usize(0),
                    VariableIndex::from_usize(3),
                    VariableIndex::from_usize(4),
                ]
    }));
}

#[test]
fn multiply_out_unconditional_propositional_effects() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["0".into(), "1".into()],
        None,
        ExplicitValueIndex::new(0),
    )];

    let op = Operator::new(
        "set".into(),
        vec![],
        vec![planforge_sas::numeric_task::Effect::new(
            vec![],
            VariableIndex::new(0),
            Some(ExplicitValueIndex::new(0)),
            ExplicitValueIndex::new(1),
        )],
        vec![],
        OperatorCost::new(1),
    );

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables: vec![],
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::with_partitions(vec![]);

    let abs_ops = identity_operators(&task, partitions, false).unwrap();

    // multiply_out creates one operator for predecessor value 0 -> 1.
    assert_eq!(abs_ops.len(), 1);
    assert_eq!(abs_ops[0].regression_offset, -1);
    assert_eq!(
        abs_ops[0].preconditions,
        vec![ExplicitFact::propositional(0, 0)]
    );
    assert_eq!(
        abs_ops[0].regression_preconditions,
        vec![ExplicitFact::propositional(0, 1)]
    );
}

/// An operator that requires a comparison does not carry the verdict across:
/// the abstract operator's regression side clears the digit to the class of
/// `False`, so regression re-derives the comparison from the target partition.
#[test]
fn metric_tasks_use_metric_delta_for_abstract_operator_cost() {
    let variables = vec![global_constraint_variable()];
    let numeric_variables = vec![
        NumericVariable::new("fuel-used".into(), NumericType::Cost, None),
        NumericVariable::new("c5".into(), NumericType::Constant, None),
    ];

    let op = Operator::new(
        "fly".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::new(0),
            AssignmentOperation::Plus,
            VariableIndex::new(1),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, Some(VariableIndex::from_usize(0))),
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(5.0)],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::with_partitions(vec![
        vec![Interval::unbounded()],
        vec![Interval::singleton(NumericValue::new(5.0))],
    ]);

    let abs_ops = identity_operators(&task, partitions, false).unwrap();

    assert_eq!(abs_ops.len(), 1);
    assert_eq!(abs_ops[0].cost, NumericValue::new(5.0));
}

#[test]
fn derived_comparison_transition_is_skipped_when_the_target_cannot_be_decided() {
    let variables = vec![ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        "cmp".into(),
        vec!["true".into(), "false".into()],
        Some(0),
        ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
    )];

    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c0_5".into(), NumericType::Constant, None),
        NumericVariable::new("y".into(), NumericType::Derived, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
    ];

    let assignment_axioms = vec![AssignmentAxiom::new(
        VariableIndex::from_usize(2),
        CalOperator::Sum,
        VariableIndex::from_usize(0),
        VariableIndex::from_usize(1),
    )];
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(2),
        VariableIndex::new(3),
        ComparisonOperator::LessThan,
    )];

    let op = Operator::new(
        "inc-x".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::new(0),
            AssignmentOperation::Plus,
            VariableIndex::new(1),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(0.5),
            NumericValue::new(0.5),
            NumericValue::new(1.0),
        ],
        operators: vec![op.clone()],
        axioms: vec![],
        comparison_axioms,
        assignment_axioms,
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::singleton(NumericValue::new(0.0)),
            Interval::new(NumericValue::new(0.0), NumericValue::new(1.0), false, true),
        ],
        vec![Interval::singleton(NumericValue::new(0.5))],
        vec![
            Interval::singleton(NumericValue::new(0.5)),
            Interval::new(NumericValue::new(0.5), NumericValue::new(1.5), false, true),
        ],
        vec![Interval::singleton(NumericValue::new(1.0))],
    ]);

    let factory = identity_factory(&task, partitions);
    let generator = AbstractOperatorGenerator::new(&factory, &task);

    let transitions = compute_hash_effects_with_preconditions(
        &task,
        &generator,
        OperatorIndex::new(0),
        &[],
        op.assignment_effects(),
        Deadline::NONE,
    )
    .unwrap();

    assert!(
        transitions.iter().all(|trans| {
            trans
                .source_partition_facts
                .iter()
                .chain(trans.target_partition_facts.iter())
                .all(|fact| fact.var() != 0)
        }),
        "transitions={transitions:#?}"
    );
    let derived_abs_var = task.variables().len() + 2;
    assert!(
        transitions.iter().all(|transition| {
            transition
                .changed_numeric_vars
                .contains(&VariableIndex::from_usize(2))
                && transition
                    .source_partition_facts
                    .iter()
                    .any(|fact| fact.var() == derived_abs_var)
                && transition
                    .target_partition_facts
                    .iter()
                    .any(|fact| fact.var() == derived_abs_var)
        }),
        "the exact y=x+0.5 coordinate must receive its own +0.5 transitions: {transitions:#?}"
    );
}

#[test]
fn additive_view_filters_false_equality_precondition() {
    let variables = vec![ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        "at-target".into(),
        vec!["true".into(), "false".into()],
        Some(0),
        ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("target".into(), NumericType::Constant, None),
        NumericVariable::new("x-minus-target".into(), NumericType::Derived, None),
        NumericVariable::new("zero".into(), NumericType::Constant, None),
    ];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(10.0),
            NumericValue::new(-10.0),
            NumericValue::new(0.0),
        ],
        operators: vec![Operator::new(
            "save".into(),
            vec![ExplicitFact::propositional(0, 0)],
            vec![],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(2),
            VariableIndex::new(3),
            ComparisonOperator::Equal,
        )],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(2),
            CalOperator::Difference,
            VariableIndex::from_usize(0),
            VariableIndex::from_usize(1),
        )],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let partitions = NumericPartitions::with_partitions(vec![
        vec![Interval::unbounded()],
        vec![Interval::singleton(NumericValue::new(10.0))],
        vec![
            Interval::singleton(NumericValue::new(-9.0)),
            Interval::new(NumericValue::new(-9.0), INF_VALUE, false, false),
        ],
        vec![Interval::singleton(NumericValue::new(0.0))],
    ]);
    let factory = identity_factory(&task, partitions);
    let generator = AbstractOperatorGenerator::new(&factory, &task);

    let transitions = compute_hash_effects_with_preconditions(
        &task,
        &generator,
        OperatorIndex::new(0),
        &[ExplicitFact::propositional(0, 0)],
        &[],
        Deadline::NONE,
    )
    .unwrap();
    let derived_abs_var = task.variables().len() + 2;

    assert!(transitions.iter().all(|transition| {
        !transition
            .source_partition_facts
            .contains(&ExplicitFact::numeric_variable(derived_abs_var, 0))
    }));
}

#[test]
fn combo_interval_build_keeps_missing_derived_operand_unknown_during_propagation() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![global_constraint_variable()],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("c0".into(), NumericType::Constant, None),
            NumericVariable::new("ghost".into(), NumericType::Derived, None),
            NumericVariable::new("d".into(), NumericType::Derived, None),
        ],
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(5.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
        ],
        operators: vec![],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::from_usize(3),
            CalOperator::Product,
            VariableIndex::from_usize(2),
            VariableIndex::from_usize(1),
        )],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::singleton(NumericValue::new(5.0)),
            Interval::singleton(NumericValue::new(6.0)),
        ],
        vec![Interval::singleton(NumericValue::new(0.0))],
        vec![Interval::unbounded()],
        vec![Interval::unbounded()],
    ]);

    let factory = identity_factory(&task, partitions);
    let generator = AbstractOperatorGenerator::new(&factory, &task);

    let mut intervals = Vec::new();
    prepare_comparison_tree_inputs_for_combo_into(
        &task,
        &generator,
        &[(VariableIndex::from_usize(0), 0, 1)],
        &mut intervals,
    );

    assert_eq!(intervals[0], Interval::singleton(NumericValue::new(5.0)));
    assert_eq!(intervals[1], Interval::singleton(NumericValue::new(0.0)));
    assert_eq!(intervals[2], Interval::unbounded());
    assert_eq!(intervals[3], Interval::unbounded());
}

#[test]
fn duplicate_assignment_effects_are_rejected() {
    let variables = vec![global_constraint_variable()];

    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
        NumericVariable::new("c2".into(), NumericType::Constant, None),
    ];

    let op = Operator::new(
        "dup-ass".into(),
        vec![],
        vec![],
        vec![
            AssignmentEffect::new(
                VariableIndex::new(0),
                AssignmentOperation::Plus,
                VariableIndex::new(1),
                false,
                vec![],
            ),
            AssignmentEffect::new(
                VariableIndex::new(0),
                AssignmentOperation::Plus,
                VariableIndex::new(2),
                false,
                vec![],
            ),
        ],
        OperatorCost::new(1),
    );

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(1.0),
            NumericValue::new(2.0),
        ],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let error = semantics_error(&task);
    assert!(
        error
            .to_string()
            .contains("multiple numeric effects on variable 0")
    );
}

#[test]
fn conditional_propositional_effect_branches() {
    let variables = vec![
        ExplicitVariable::new(
            2,
            "c".into(),
            vec!["0".into(), "1".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
        ExplicitVariable::new(
            2,
            "u".into(),
            vec!["0".into(), "1".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
        ExplicitVariable::new(
            2,
            "v".into(),
            vec!["0".into(), "1".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
    ];

    let op = Operator::new(
        "op".into(),
        vec![
            ExplicitFact::propositional(1, 0),
            ExplicitFact::propositional(2, 0),
        ],
        vec![
            Effect::new(
                vec![],
                VariableIndex::new(1),
                Some(ExplicitValueIndex::new(0)),
                ExplicitValueIndex::new(1),
            ),
            Effect::new(
                vec![ExplicitFact::propositional(0, 1)],
                VariableIndex::from_usize(2),
                Some(ExplicitValueIndex::new(0)),
                ExplicitValueIndex::new(1),
            ),
        ],
        vec![],
        OperatorCost::new(1),
    );

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables: vec![],
        goals: vec![],
        mutexes: vec![],
        state: vec![
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(0),
        ],
        numeric_state: vec![],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let err = semantics_error(&task);
    assert!(crate::unsupported::is_unsupported_task(&err), "{err:#}");
    assert!(err.to_string().contains("has a conditional"), "{err:#}");
}

#[test]
fn conditional_assignment_effect_branches() {
    let variables = vec![
        ExplicitVariable::new(
            2,
            "c".into(),
            vec!["0".into(), "1".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
        ExplicitVariable::new(
            2,
            "p".into(),
            vec!["0".into(), "1".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
    ];

    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c1".into(), NumericType::Constant, None),
    ];

    let op = Operator::new(
        "op".into(),
        vec![ExplicitFact::propositional(1, 0)],
        vec![planforge_sas::numeric_task::Effect::new(
            vec![],
            VariableIndex::new(1),
            Some(ExplicitValueIndex::new(0)),
            ExplicitValueIndex::new(1),
        )],
        vec![AssignmentEffect::new(
            VariableIndex::new(0),
            AssignmentOperation::Plus,
            VariableIndex::new(1),
            true,
            vec![ExplicitFact::propositional(0, 1)],
        )],
        OperatorCost::new(1),
    );

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(1.0)],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let err = semantics_error(&task);
    assert!(crate::unsupported::is_unsupported_task(&err), "{err:#}");
    assert!(err.to_string().contains("has a conditional"), "{err:#}");
}

#[test]
fn variable_rhs_assignment_effect_is_rejected_for_parity() {
    let variables = vec![global_constraint_variable()];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
    ];

    let op = Operator::new(
        "assign-var".into(),
        vec![],
        vec![],
        vec![AssignmentEffect::new(
            VariableIndex::new(0),
            AssignmentOperation::Plus,
            VariableIndex::new(1),
            false,
            vec![],
        )],
        OperatorCost::new(1),
    );

    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(0.0)],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let err = semantics_error(&task);
    assert!(crate::unsupported::is_unsupported_task(&err), "{err:#}");
    assert!(err.to_string().contains("not a constant"), "{err:#}");
}
