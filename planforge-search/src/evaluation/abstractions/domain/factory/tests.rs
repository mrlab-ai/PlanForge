use super::*;
use ordered_float::NotNan;

use std::collections::BTreeSet;

use crate::evaluation::abstractions::common::cartesian_set::CartesianSet;
use crate::evaluation::abstractions::common::transition_system::{
    AbstractOperatorSourceSets, LabelSourceSet,
};
use crate::evaluation::abstractions::cost_partitioning::costs::{
    ExplicitSaturationGraph, RemainingCosts, saturate_labels, saturate_regions,
};
use crate::evaluation::abstractions::domain::initial_splits::identity_domain_mapping_and_sizes;
use planforge_sas::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::{ConditionNode, ConditionValue, NumericCondition};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitValueIndex,
    ExplicitVariable, INF_VALUE, Metric, NEG_INF_VALUE, NumericRootTask, NumericRootTaskParts,
    NumericType, NumericValue, NumericVariable, Operator, OperatorCost, OperatorIndex,
};

/// The propositional variable a comparison axiom writes, on axiom layer 0.
fn condition_variable(name: &str) -> ExplicitVariable {
    ExplicitVariable::new(
        ConditionValue::DOMAIN_SIZE,
        name.into(),
        vec!["true".into(), "false".into()],
        Some(0),
        ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
    )
}

/// Initial values of the constant numeric variables `condition` reads.
///
/// Every leaf id is a valid numeric variable: `NumericConditions::build`
/// rejects out-of-range references before a task is ever constructed.
fn constant_leaf_values(
    condition: &NumericCondition,
    task: &dyn AbstractNumericTask,
    initial_numeric_values: &[NumericValue],
) -> Vec<f64> {
    let numeric_variables = task.numeric_variables();
    let mut out: HashSet<u64> = HashSet::new();
    for node in condition.nodes() {
        let ConditionNode::Leaf { numeric_var_id } = node else {
            continue;
        };
        if numeric_variables[numeric_var_id.index()].get_type() != &NumericType::Constant {
            continue;
        }
        let value = initial_numeric_values[numeric_var_id.index()];
        if value.is_nan() {
            continue;
        }
        out.insert(value.to_bits());
    }
    out.into_iter().map(f64::from_bits).collect()
}

fn partitions_from_cutpoints(cutpoints: &[NumericValue]) -> Vec<Interval> {
    let mut cuts: Vec<NumericValue> = cutpoints
        .iter()
        .copied()
        .filter(|v| v.is_finite() && !v.is_nan())
        .collect();
    cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cuts.dedup_by(|a, b| a.to_bits() == b.to_bits());

    let mut out: Vec<Interval> = Vec::new();
    let mut prev = NEG_INF_VALUE;
    for &c in &cuts {
        out.push(Interval::new(prev, c, false, false));
        out.push(Interval::singleton(c));
        prev = c;
    }
    out.push(Interval::new(prev, INF_VALUE, false, false));
    out
}

fn cutpoint_partitions_for_task(task: &dyn AbstractNumericTask) -> Result<NumericPartitions> {
    let initial_numeric_values = task.get_initial_numeric_state_values();
    let num_numeric_vars = task.numeric_variables().len();

    let mut cutpoints_by_var: Vec<BTreeSet<NotNan<f64>>> = vec![BTreeSet::new(); num_numeric_vars];
    for condition in task.numeric_conditions().iter() {
        let constant_values = constant_leaf_values(condition, task, initial_numeric_values);
        if constant_values.is_empty() {
            continue;
        }

        for &dep in condition.regular_numeric_var_dependencies() {
            ensure!(
                dep.index() < cutpoints_by_var.len(),
                "numeric condition depends on numeric var {dep}, but only {} numeric vars exist",
                cutpoints_by_var.len()
            );
            for &value in &constant_values {
                let value = NotNan::new(value).map_err(|_| anyhow!("NaN cutpoint encountered"))?;
                if value.is_finite() {
                    cutpoints_by_var[dep.index()].insert(value);
                }
            }
        }
    }

    let mut partitions_by_numeric_var: Vec<Vec<Interval>> = Vec::with_capacity(num_numeric_vars);
    for (var_id, var) in task.numeric_variables().iter().enumerate() {
        let parts = match var.get_type() {
            NumericType::Constant => vec![Interval::singleton(initial_numeric_values[var_id])],
            NumericType::Regular => {
                let cuts: Vec<NumericValue> = cutpoints_by_var[var_id]
                    .iter()
                    .map(|v| NumericValue::new(v.into_inner()))
                    .collect();
                if cuts.is_empty() {
                    vec![Interval::unbounded()]
                } else {
                    partitions_from_cutpoints(&cuts)
                }
            }
            NumericType::Derived | NumericType::Cost => vec![Interval::unbounded()],
        };
        partitions_by_numeric_var.push(parts);
    }

    Ok(NumericPartitions::with_partitions(
        partitions_by_numeric_var,
    ))
}

fn factory_identity_cutpoints(task: &dyn AbstractNumericTask) -> Result<DomainAbstractionFactory> {
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(task)?;
    let partitions = cutpoint_partitions_for_task(task)?;
    DomainAbstractionFactory::new(task, domain_mapping, domain_sizes, partitions)
}

#[test]
fn explicit_transition_system_matches_implicit_distances_across_comparison_cascades() {
    let variables = vec![
        condition_variable("x-lt-ten"),
        ExplicitVariable::new(
            2,
            "goal".into(),
            vec!["false".into(), "true".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("ten".into(), NumericType::Constant, None),
        NumericVariable::new("twenty".into(), NumericType::Constant, None),
    ];
    let operators = vec![
        Operator::new(
            "increase-x".into(),
            vec![ExplicitFact::propositional(
                0,
                ConditionValue::True.as_usize(),
            )],
            vec![],
            vec![AssignmentEffect::new(
                VariableIndex::from_usize(0),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(2),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        ),
        Operator::new(
            "finish".into(),
            vec![ExplicitFact::propositional(
                0,
                ConditionValue::False.as_usize(),
            )],
            vec![Effect::new(
                vec![],
                VariableIndex::from_usize(1),
                Some(ExplicitValueIndex::new(0)),
                ExplicitValueIndex::new(1),
            )],
            vec![],
            OperatorCost::new(1),
        ),
    ];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: vec![],
        state: vec![
            ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
            ExplicitValueIndex::new(0),
        ],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(10.0),
            NumericValue::new(20.0),
        ],
        operators,
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(0),
            VariableIndex::new(1),
            ComparisonOperator::LessThan,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, ConditionValue::False.as_usize()),
    });
    let factory = factory_identity_cutpoints(&task).unwrap();
    let implicit = factory.build_distance_table_for_test(&task, false).unwrap();
    let abstract_operators = factory
        .abstract_operators(&task, false, Deadline::NONE)
        .unwrap();
    let transition_system = factory
        .build_transition_system(
            &task,
            &abstract_operators,
            StateRegions::Omit,
            Deadline::NONE,
        )
        .unwrap();
    let (explicit, _) = saturate_labels(
        &mut ExplicitSaturationGraph::new(&transition_system),
        &[1.0, 1.0],
        None,
        Deadline::NONE,
    )
    .unwrap();

    assert_eq!(explicit, implicit.distances);
    assert_eq!(explicit[implicit.initial_state], 2.0);
}

#[test]
fn precise_regional_table_charges_only_the_transition_source_partition() {
    let variables = vec![ExplicitVariable::new(
        2,
        "p".into(),
        vec!["p0".into(), "p1".into()],
        None,
        ExplicitValueIndex::new(0),
    )];
    let op = Operator::new(
        "move".into(),
        vec![ExplicitFact::propositional(0, 0)],
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
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 1),
    });
    let factory = factory_identity_cutpoints(&task).unwrap();
    let abstract_operators = factory
        .abstract_operators(&task, false, Deadline::NONE)
        .unwrap();
    assert_eq!(abstract_operators.len(), 1);
    let transition_system = factory
        .build_transition_system(
            &task,
            &abstract_operators,
            StateRegions::Omit,
            Deadline::NONE,
        )
        .unwrap();
    let source_sets = vec![AbstractOperatorSourceSets {
        labels: vec![LabelSourceSet {
            concrete_op_id: OperatorIndex::new(0),
            source: CartesianSet::with_all_props_constrained(
                vec![vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)]],
                Vec::new(),
            )
            .into(),
        }],
    }];
    let mut residuals = RemainingCosts::from_operator_costs(&[1.0]);

    let (distances, allocation) = saturate_regions(
        &transition_system,
        &source_sets,
        |transition, source_set| {
            source_set_within(&factory.state_region(transition.source_state), source_set)
        },
        &residuals,
        None,
        Deadline::NONE,
    )
    .unwrap();

    assert_eq!(distances[transition_system.initial_state], 1.0);
    assert_eq!(allocation.entries().len(), 1);
    assert_eq!(
        allocation.entries()[0].source_set.source.propositions()[0],
        vec![ExplicitValueIndex::new(0)]
    );
    residuals
        .reduce_by_source_set_allocation_with_deadline(&allocation, Deadline::NONE)
        .unwrap();

    let disjoint_source = LabelSourceSet {
        concrete_op_id: OperatorIndex::new(0),
        source: CartesianSet::with_all_props_constrained(
            vec![vec![ExplicitValueIndex::new(1)]],
            Vec::new(),
        )
        .into(),
    };
    assert_eq!(residuals.cost_for_source_set(&disjoint_source), 1.0);
}

#[test]
fn factory_splits_regular_var_at_constants_in_comparison_trees() {
    let variables = vec![condition_variable("cmp")];

    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c10".into(), NumericType::Constant, None),
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
        vec![],
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
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(10.0)],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms,
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let factory = factory_identity_cutpoints(&task).unwrap();
    assert_eq!(factory.numeric_domain_sizes(), &[3, 1]);

    let x0_parts = factory
        .partitions()
        .partitions(VariableIndex::from_usize(0));
    assert_eq!(x0_parts.len(), 3);
    assert_eq!(
        x0_parts[0],
        Interval::new(NEG_INF_VALUE, NumericValue::new(10.0), false, false)
    );
    assert_eq!(x0_parts[1], Interval::singleton(NumericValue::new(10.0)));
    assert_eq!(
        x0_parts[2],
        Interval::new(NumericValue::new(10.0), INF_VALUE, false, false)
    );

    let c10_parts = factory
        .partitions()
        .partitions(VariableIndex::from_usize(1));
    assert_eq!(c10_parts, &[Interval::singleton(NumericValue::new(10.0))]);
}

impl DomainAbstractionFactory {
    /// Goal distances of every abstract state under the task's own operator
    /// costs.
    fn build_distance_table_for_test(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
    ) -> Result<AbstractDistanceTable> {
        let operators = self.abstract_operators(task, combine_labels, Deadline::NONE)?;
        self.build_distance_table(task, &operators, Deadline::NONE)
    }
}

/// The concrete operators the first step of the optimal abstract plan from
/// the initial state may execute, as CEGAR's progression replay sees them.
fn first_plan_step(
    task: &NumericRootTask,
    combine_labels: bool,
    use_wildcard_plans: bool,
) -> Vec<OperatorIndex> {
    use crate::evaluation::abstractions::common::cegar::options::CegarOptions;
    use crate::evaluation::abstractions::common::cegar::refine::Refinable;
    use crate::evaluation::abstractions::common::cegar::view::AbstractionView;
    use crate::evaluation::abstractions::common::cegar::view::RefinementLimits;
    use crate::evaluation::abstractions::domain::config::DomainAbstractionConfig;
    use crate::evaluation::abstractions::domain::view::DomainRefinement;

    let config = DomainAbstractionConfig {
        cegar: CegarOptions {
            combine_labels,
            ..DomainAbstractionConfig::default().cegar
        },
        use_wildcard_plans,
        ..DomainAbstractionConfig::default()
    };
    let no_variables = std::collections::HashSet::new();
    let limits = RefinementLimits {
        max_states: usize::MAX,
        blacklisted_prop_var_ids: &no_variables,
        blacklisted_numeric_var_ids: &no_variables,
    };
    let factory = factory_identity_cutpoints(task).unwrap();
    let mut refinement = DomainRefinement::new(task, factory, limits, &config);
    let view = refinement.view(Deadline::NONE).unwrap();
    let initial = view
        .abstract_state(
            task.get_initial_propositional_state_values(),
            task.get_initial_numeric_state_values(),
        )
        .unwrap();
    view.step_operators(view.optimal_transition(initial).unwrap())
}

#[test]
fn wildcard_plan_collects_all_equivalent_concrete_ops() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        Some(0),
        ExplicitValueIndex::new(0),
    )];
    let numeric_variables: Vec<NumericVariable> = vec![];
    let goals = vec![ExplicitFact::propositional(0, 1)];
    let op0 = Operator::new(
        "set0".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![Effect::new(
            vec![],
            VariableIndex::from_usize(0),
            Some(ExplicitValueIndex::new(0)),
            ExplicitValueIndex::new(1),
        )],
        vec![],
        OperatorCost::new(1),
    );
    let op1 = Operator::new(
        "set1".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![Effect::new(
            vec![],
            VariableIndex::from_usize(0),
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
        numeric_variables,
        goals,
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![op0, op1],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    assert_eq!(
        first_plan_step(&task, true, true),
        [OperatorIndex::new(0), OperatorIndex::new(1)]
    );
}

#[test]
fn wildcard_plan_uses_first_matching_operator_group_when_labels_uncombined() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        Some(0),
        ExplicitValueIndex::new(0),
    )];
    let numeric_variables: Vec<NumericVariable> = vec![];
    let goals = vec![ExplicitFact::propositional(0, 1)];
    let op0 = Operator::new(
        "set0".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![Effect::new(
            vec![],
            VariableIndex::from_usize(0),
            Some(ExplicitValueIndex::new(0)),
            ExplicitValueIndex::new(1),
        )],
        vec![],
        OperatorCost::new(1),
    );
    let op1 = Operator::new(
        "set1".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![Effect::new(
            vec![],
            VariableIndex::from_usize(0),
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
        numeric_variables,
        goals,
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![op0, op1],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    assert_eq!(first_plan_step(&task, false, true), [OperatorIndex::new(0)]);
}

#[test]
fn singleton_plan_is_produced_when_wildcards_are_disabled() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        Some(0),
        ExplicitValueIndex::new(0),
    )];
    let numeric_variables: Vec<NumericVariable> = vec![];
    let goals = vec![ExplicitFact::propositional(0, 1)];
    let op0 = Operator::new(
        "set0".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![Effect::new(
            vec![],
            VariableIndex::new(0),
            Some(ExplicitValueIndex::new(0)),
            ExplicitValueIndex::new(1),
        )],
        vec![],
        OperatorCost::new(1),
    );
    let op1 = Operator::new(
        "set1".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![Effect::new(
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
        numeric_variables,
        goals,
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![op0, op1],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    assert_eq!(first_plan_step(&task, true, false), [OperatorIndex::new(0)]);
}

#[test]
fn initial_state_hash_is_unique() {
    // One comparison-axiom propositional variable.
    let variables = vec![condition_variable("cmp")];
    // Two regular numeric vars with no constants -> partitions are unbounded.
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(0),
        VariableIndex::new(1),
        ComparisonOperator::LessThan,
    )];
    let op = Operator::new("noop".into(), vec![], vec![], vec![], OperatorCost::new(1));
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        // The concrete initial state used by numeric-fd has comparisons evaluated.
        state: vec![ExplicitValueIndex::from_usize(
            ConditionValue::False.as_usize(),
        )],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(0.0)],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms,
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let factory = factory_identity_cutpoints(&task).unwrap();
    let table = factory.build_distance_table_for_test(&task, true).unwrap();

    assert_eq!(
        factory.domain_sizes(),
        [1],
        "condition variables keep their single abstract value"
    );
    assert!(table.initial_state < table.distances.len());
}

/// A goal on a variable the abstraction collapsed to one value is dropped.
///
/// Such a variable carries no information: every abstract state agrees on it, so
/// testing it would either accept everything or nothing. Dropping it is the
/// relaxation, and the goals that remain are the ones the abstraction can tell
/// apart.
#[test]
fn abstract_goals_drop_collapsed_variables() {
    let variables = vec![
        ExplicitVariable::new(
            1,
            "trivial".into(),
            vec!["only".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
        ExplicitVariable::new(
            2,
            "target".into(),
            vec!["off".into(), "on".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
    ];
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables: vec![],
        goals: vec![
            ExplicitFact::propositional(0, 0),
            ExplicitFact::propositional(1, 1),
        ],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![Operator::new(
            "noop".into(),
            vec![],
            vec![],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let factory = factory_identity_cutpoints(&task).unwrap();
    assert_eq!(
        factory.compute_abstract_goals(&task),
        vec![ExplicitFact::propositional(1, 1)],
        "the goal on the one-valued variable is the one that has to go"
    );
}

#[test]
fn factory_numeric_context_keeps_consistent_additive_derived_partition() {
    let variables = vec![condition_variable("cmp")];

    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c2".into(), NumericType::Constant, None),
        NumericVariable::new("d".into(), NumericType::Derived, None),
    ];
    let assignment_axioms = vec![AssignmentAxiom::new(
        VariableIndex::from_usize(2),
        CalOperator::Sum,
        VariableIndex::new(0),
        VariableIndex::new(1),
    )];
    let comparison_axioms = vec![ComparisonAxiom::new(
        VariableIndex::new(0),
        VariableIndex::new(2),
        VariableIndex::new(1),
        ComparisonOperator::GreaterThan,
    )];
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
            NumericValue::new(2.0),
            NumericValue::new(0.0),
        ],
        operators: vec![Operator::new(
            "noop".into(),
            vec![],
            vec![],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms,
        assignment_axioms,
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::singleton(NumericValue::new(0.0)),
            Interval::singleton(NumericValue::new(1.0)),
        ],
        vec![Interval::singleton(NumericValue::new(2.0))],
        vec![
            Interval::singleton(NumericValue::new(2.0)),
            Interval::singleton(NumericValue::new(100.0)),
        ],
    ]);
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory =
        DomainAbstractionFactory::new(&task, domain_mapping, domain_sizes, partitions).unwrap();
    let hash_multipliers = factory.layout().multipliers();

    let x_partition = 0;
    let c2_partition = 0;
    let derived_partition = 0;
    let state_hash = x_partition * hash_multipliers[1]
        + c2_partition * hash_multipliers[2]
        + derived_partition * hash_multipliers[3];

    let mut numeric_intervals = Vec::new();
    fill_state_intervals(
        &task,
        factory.numeric_conditions().all(),
        factory.partitions(),
        factory.layout(),
        state_hash,
        &mut numeric_intervals,
    )
    .unwrap();

    assert_eq!(
        numeric_intervals[0],
        Interval::singleton(NumericValue::new(0.0))
    );
    assert_eq!(
        numeric_intervals[1],
        Interval::singleton(NumericValue::new(2.0))
    );
    assert_eq!(
        numeric_intervals[2],
        Interval::singleton(NumericValue::new(2.0))
    );
}

fn additive_numeric_source_set_task() -> (NumericRootTask, DomainAbstractionFactory) {
    let variables = vec![ExplicitVariable::new(
        1,
        "p".into(),
        vec!["p0".into()],
        None,
        ExplicitValueIndex::new(0),
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
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
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(9.0), NumericValue::new(1.0)],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(NumericValue::new(9.0), NumericValue::new(10.0), true, true),
            Interval::new(NumericValue::new(10.0), INF_VALUE, true, false),
        ],
        vec![Interval::singleton(NumericValue::new(1.0))],
    ]);
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory =
        DomainAbstractionFactory::new(&task, domain_mapping, domain_sizes, partitions).unwrap();

    (task, factory)
}

fn additive_numeric_source_set_task_with_partitions(
    x_partitions: Vec<Interval>,
) -> (NumericRootTask, DomainAbstractionFactory) {
    let variables = vec![ExplicitVariable::new(
        1,
        "p".into(),
        vec!["p0".into()],
        None,
        ExplicitValueIndex::new(0),
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
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
        variables,
        numeric_variables,
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(1.0)],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let partitions = NumericPartitions::with_partitions(vec![
        x_partitions,
        vec![Interval::singleton(NumericValue::new(1.0))],
    ]);
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory =
        DomainAbstractionFactory::new(&task, domain_mapping, domain_sizes, partitions).unwrap();

    (task, factory)
}

#[test]
fn abstract_source_set_keeps_finite_source_when_target_reaches_tail() {
    let (task, factory) = additive_numeric_source_set_task();
    let x_abs_var = task.variables().len();
    let op = super::super::abstract_operators::AbstractOperator {
        concrete_op_ids: vec![OperatorIndex::new(0)],
        cost: NumericValue::new(1.0),
        regression_offset: 0,
        regression_preconditions: vec![ExplicitFact::numeric_variable(x_abs_var, 1)],
        preconditions: vec![ExplicitFact::numeric_variable(x_abs_var, 0)],
        changed_numeric_vars: vec![VariableIndex::from_usize(0)],
    };

    let source_sets = factory.build_abstract_source_sets(&task, &[op]).unwrap();
    let concrete = &source_sets[0].labels[0];

    assert_eq!(concrete.concrete_op_id, OperatorIndex::new(0));
    assert_eq!(
        concrete.source.numeric[0],
        Interval::new(NumericValue::new(9.0), NumericValue::new(10.0), true, true)
    );
    assert_eq!(
        concrete.source.numeric[1],
        Interval::new(NEG_INF_VALUE, INF_VALUE, false, false)
    );
}

#[test]
fn abstract_source_set_tightens_source_by_inverse_target_image() {
    let (task, factory) = additive_numeric_source_set_task_with_partitions(vec![
        Interval::new(NEG_INF_VALUE, NumericValue::new(5.0), false, true),
        Interval::new(NumericValue::new(5.0), NumericValue::new(10.0), false, true),
    ]);
    let x_abs_var = task.variables().len();
    let op = super::super::abstract_operators::AbstractOperator {
        concrete_op_ids: vec![OperatorIndex::new(0)],
        cost: NumericValue::new(1.0),
        regression_offset: 0,
        regression_preconditions: vec![ExplicitFact::numeric_variable(x_abs_var, 1)],
        preconditions: vec![ExplicitFact::numeric_variable(x_abs_var, 0)],
        changed_numeric_vars: vec![VariableIndex::from_usize(0)],
    };

    let source_sets = factory.build_abstract_source_sets(&task, &[op]).unwrap();
    let concrete = &source_sets[0].labels[0];

    assert_eq!(concrete.concrete_op_id, OperatorIndex::new(0));
    assert_eq!(
        concrete.source.numeric[0],
        Interval::new(NumericValue::new(4.0), NumericValue::new(5.0), false, true)
    );
}

#[test]
fn source_set_active_preimage_allows_boundary_charge() {
    let (task, factory) = additive_numeric_source_set_task_with_partitions(vec![
        Interval::new(NEG_INF_VALUE, NumericValue::new(5.0), false, true),
        Interval::new(NumericValue::new(5.0), NumericValue::new(6.0), false, true),
    ]);
    let x_abs_var = task.variables().len();
    let op = super::super::abstract_operators::AbstractOperator {
        concrete_op_ids: vec![OperatorIndex::new(0)],
        cost: NumericValue::new(1.0),
        regression_offset: 0,
        regression_preconditions: vec![ExplicitFact::numeric_variable(x_abs_var, 1)],
        preconditions: vec![ExplicitFact::numeric_variable(x_abs_var, 0)],
        changed_numeric_vars: vec![VariableIndex::from_usize(0)],
    };

    let source_sets = factory.build_abstract_source_sets(&task, &[op]).unwrap();
    let concrete = &source_sets[0].labels[0];

    assert_eq!(
        concrete.source.numeric[0],
        Interval::new(NumericValue::new(4.0), NumericValue::new(5.0), false, true)
    );
}

#[test]
fn abstract_source_set_rejects_empty_inverse_target_image() {
    let (task, factory) = additive_numeric_source_set_task_with_partitions(vec![
        Interval::new(NEG_INF_VALUE, NumericValue::new(5.0), false, true),
        Interval::new(NumericValue::new(7.0), NumericValue::new(10.0), false, true),
    ]);
    let x_abs_var = task.variables().len();
    let op = super::super::abstract_operators::AbstractOperator {
        concrete_op_ids: vec![OperatorIndex::new(0)],
        cost: NumericValue::new(1.0),
        regression_offset: 0,
        regression_preconditions: vec![ExplicitFact::numeric_variable(x_abs_var, 1)],
        preconditions: vec![ExplicitFact::numeric_variable(x_abs_var, 0)],
        changed_numeric_vars: vec![VariableIndex::from_usize(0)],
    };

    let error = factory
        .build_abstract_source_sets(&task, &[op])
        .unwrap_err();
    assert!(error.to_string().contains("empty source set"));
}

#[test]
fn abstract_source_set_allocates_unbounded_changed_tail() {
    let (task, factory) = additive_numeric_source_set_task();
    let x_abs_var = task.variables().len();
    let op = super::super::abstract_operators::AbstractOperator {
        concrete_op_ids: vec![OperatorIndex::new(0)],
        cost: NumericValue::new(1.0),
        regression_offset: 0,
        regression_preconditions: vec![ExplicitFact::numeric_variable(x_abs_var, 1)],
        preconditions: vec![ExplicitFact::numeric_variable(x_abs_var, 1)],
        changed_numeric_vars: vec![VariableIndex::from_usize(0)],
    };

    let source_sets = factory.build_abstract_source_sets(&task, &[op]).unwrap();
    let concrete = &source_sets[0].labels[0];

    assert_eq!(concrete.concrete_op_id, OperatorIndex::new(0));
    assert_eq!(
        concrete.source.numeric[0],
        Interval::new(NumericValue::new(10.0), INF_VALUE, true, false)
    );
}

#[test]
fn abstract_source_set_allocates_operator_without_numeric_effects() {
    let variables = vec![ExplicitVariable::new(
        2,
        "saved".into(),
        vec!["false".into(), "true".into()],
        None,
        ExplicitValueIndex::new(0),
    )];
    let op = Operator::new(
        "save".into(),
        vec![ExplicitFact::propositional(0, 0)],
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
        version: 4,
        metric: Metric::new(true, None),
        variables,
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory = DomainAbstractionFactory::new(
        &task,
        domain_mapping,
        domain_sizes,
        NumericPartitions::with_partitions(vec![]),
    )
    .unwrap();
    let op = super::super::abstract_operators::AbstractOperator {
        concrete_op_ids: vec![OperatorIndex::new(0)],
        cost: NumericValue::new(1.0),
        regression_offset: 0,
        regression_preconditions: vec![ExplicitFact::propositional(0, 1)],
        preconditions: vec![ExplicitFact::propositional(0, 0)],
        changed_numeric_vars: vec![],
    };

    let source_sets = factory.build_abstract_source_sets(&task, &[op]).unwrap();
    let concrete = &source_sets[0].labels[0];

    assert_eq!(
        concrete.source.propositions()[0],
        vec![ExplicitValueIndex::new(0)]
    );
}

#[test]
fn abstract_source_set_allows_one_finite_changed_source() {
    let variables = vec![ExplicitVariable::new(
        1,
        "p".into(),
        vec!["p0".into()],
        None,
        ExplicitValueIndex::new(0),
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
    ];
    let op = Operator::new(
        "inc_both".into(),
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
            NumericValue::new(0.0),
            NumericValue::new(1.0),
        ],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(NumericValue::new(0.0), NumericValue::new(1.0), true, true),
            Interval::new(NumericValue::new(1.0), NumericValue::new(2.0), false, true),
        ],
        vec![
            Interval::new(NEG_INF_VALUE, NumericValue::new(0.0), false, true),
            Interval::new(NumericValue::new(0.0), INF_VALUE, false, false),
        ],
        vec![Interval::singleton(NumericValue::new(1.0))],
    ]);
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory =
        DomainAbstractionFactory::new(&task, domain_mapping, domain_sizes, partitions).unwrap();
    let x_abs_var = task.variables().len();
    let y_abs_var = x_abs_var + 1;
    let op = super::super::abstract_operators::AbstractOperator {
        concrete_op_ids: vec![OperatorIndex::new(0)],
        cost: NumericValue::new(1.0),
        regression_offset: 0,
        regression_preconditions: vec![
            ExplicitFact::numeric_variable(x_abs_var, 1),
            ExplicitFact::numeric_variable(y_abs_var, 0),
        ],
        preconditions: vec![
            ExplicitFact::numeric_variable(x_abs_var, 0),
            ExplicitFact::numeric_variable(y_abs_var, 0),
        ],
        changed_numeric_vars: vec![VariableIndex::from_usize(0), VariableIndex::from_usize(1)],
    };

    let source_sets = factory.build_abstract_source_sets(&task, &[op]).unwrap();
    let concrete = &source_sets[0].labels[0];

    assert_eq!(
        concrete.source.numeric[0],
        Interval::new(NumericValue::new(0.0), NumericValue::new(1.0), false, true)
    );
    assert_eq!(
        concrete.source.numeric[1],
        Interval::new(NEG_INF_VALUE, NumericValue::new(-1.0), false, true)
    );
}

#[test]
fn source_set_one_finite_dim_suffices() {
    let variables = vec![
        condition_variable("x_gt_zero"),
        ExplicitVariable::new(
            2,
            "saved".into(),
            vec!["false".into(), "true".into()],
            None,
            ExplicitValueIndex::new(0),
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("half".into(), NumericType::Constant, None),
        NumericVariable::new("zero".into(), NumericType::Constant, None),
    ];
    let diagonal = Operator::new(
        "diagonal".into(),
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
    let save = Operator::new(
        "save".into(),
        vec![ExplicitFact::propositional(0, 0)],
        vec![Effect::new(
            vec![],
            VariableIndex::new(1),
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
        numeric_variables,
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(2), ExplicitValueIndex::new(0)],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(0.0),
            NumericValue::new(0.5),
            NumericValue::new(0.0),
        ],
        operators: vec![diagonal, save],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(0),
            VariableIndex::new(3),
            ComparisonOperator::GreaterThan,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::singleton(NumericValue::new(0.0)),
            Interval::new(NumericValue::new(0.0), INF_VALUE, false, false),
        ],
        vec![Interval::unbounded()],
        vec![Interval::singleton(NumericValue::new(0.5))],
        vec![Interval::singleton(NumericValue::new(0.0))],
    ]);
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory =
        DomainAbstractionFactory::new(&task, domain_mapping, domain_sizes, partitions).unwrap();

    let x_abs_var = task.variables().len();
    let y_abs_var = x_abs_var + 1;
    let op = super::super::abstract_operators::AbstractOperator {
        concrete_op_ids: vec![OperatorIndex::new(0)],
        cost: NumericValue::new(1.0),
        regression_offset: 0,
        regression_preconditions: vec![
            ExplicitFact::numeric_variable(x_abs_var, 1),
            ExplicitFact::numeric_variable(y_abs_var, 0),
        ],
        preconditions: vec![
            ExplicitFact::numeric_variable(x_abs_var, 0),
            ExplicitFact::numeric_variable(y_abs_var, 0),
        ],
        changed_numeric_vars: vec![VariableIndex::from_usize(0), VariableIndex::from_usize(1)],
    };
    let source_sets = factory.build_abstract_source_sets(&task, &[op]).unwrap();
    let concrete = &source_sets[0].labels[0];
    assert_eq!(
        concrete.source.numeric[0],
        Interval::singleton(NumericValue::new(0.0))
    );
    assert_eq!(concrete.source.numeric[1], Interval::unbounded());

    let table = factory.build_distance_table_for_test(&task, false).unwrap();
    assert_eq!(table.distances[table.initial_state], 2.0);
}

#[test]
fn abstract_source_set_ignores_zero_additive_effect_dimension() {
    let variables = vec![ExplicitVariable::new(
        1,
        "p".into(),
        vec!["p0".into()],
        None,
        ExplicitValueIndex::new(0),
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
        NumericVariable::new("zero".into(), NumericType::Constant, None),
    ];
    let op = Operator::new(
        "inc_x_keep_y".into(),
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
                VariableIndex::new(3),
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
            NumericValue::new(0.0),
            NumericValue::new(1.0),
            NumericValue::new(0.0),
        ],
        operators: vec![op],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });
    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(NumericValue::new(0.0), NumericValue::new(1.0), true, true),
            Interval::new(NumericValue::new(1.0), NumericValue::new(2.0), false, true),
        ],
        vec![
            Interval::new(NumericValue::new(0.0), NumericValue::new(1.0), true, true),
            Interval::new(NumericValue::new(1.0), NumericValue::new(2.0), false, true),
        ],
        vec![Interval::singleton(NumericValue::new(1.0))],
        vec![Interval::singleton(NumericValue::new(0.0))],
    ]);
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory =
        DomainAbstractionFactory::new(&task, domain_mapping, domain_sizes, partitions).unwrap();
    let x_abs_var = task.variables().len();
    let y_abs_var = x_abs_var + 1;
    let op = super::super::abstract_operators::AbstractOperator {
        concrete_op_ids: vec![OperatorIndex::new(0)],
        cost: NumericValue::new(1.0),
        regression_offset: 0,
        regression_preconditions: vec![
            ExplicitFact::numeric_variable(x_abs_var, 1),
            ExplicitFact::numeric_variable(y_abs_var, 0),
        ],
        preconditions: vec![
            ExplicitFact::numeric_variable(x_abs_var, 0),
            ExplicitFact::numeric_variable(y_abs_var, 0),
        ],
        changed_numeric_vars: vec![VariableIndex::from_usize(0), VariableIndex::from_usize(1)],
    };

    let source_sets = factory.build_abstract_source_sets(&task, &[op]).unwrap();
    let concrete = &source_sets[0].labels[0];

    assert_eq!(
        concrete.source.numeric[0],
        Interval::new(NumericValue::new(0.0), NumericValue::new(1.0), false, true)
    );
    // y has a zero-delta additive effect, so it is not in the affected-var loop
    // that intersects with the inverse target. But y *is* pinned by the
    // operator's `y_abs_var = 0` precondition, so the operator region reflects that
    // partition's interval instead of going unbounded — which is the tightest
    // admissible operator region and the source of the cost-partitioning precision
    // benefit on operators whose preconditions reference unaffected variables.
    assert_eq!(
        concrete.source.numeric[1],
        Interval::new(NumericValue::new(0.0), NumericValue::new(1.0), true, true)
    );
}

#[test]
fn source_set_width_does_not_change_valid_preimage() {
    // Same fixture as `abstract_source_set_tightens_source_by_inverse_target_image`:
    // the preimage source for the operator is `(4.0, 5.0]`, width 1.0.
    let (task, factory) = additive_numeric_source_set_task_with_partitions(vec![
        Interval::new(NEG_INF_VALUE, NumericValue::new(5.0), false, true),
        Interval::new(NumericValue::new(5.0), NumericValue::new(10.0), false, true),
    ]);
    let x_abs_var = task.variables().len();
    let op = super::super::abstract_operators::AbstractOperator {
        concrete_op_ids: vec![OperatorIndex::new(0)],
        cost: NumericValue::new(1.0),
        regression_offset: 0,
        regression_preconditions: vec![ExplicitFact::numeric_variable(x_abs_var, 1)],
        preconditions: vec![ExplicitFact::numeric_variable(x_abs_var, 0)],
        changed_numeric_vars: vec![VariableIndex::from_usize(0)],
    };

    let source_sets = factory.build_abstract_source_sets(&task, &[op]).unwrap();
    assert_eq!(
        source_sets[0].labels[0].source.numeric[0],
        Interval::new(NumericValue::new(4.0), NumericValue::new(5.0), false, true)
    );
}

#[test]
fn singleton_preimage_is_preserved_exactly() {
    // Partition `x` so that the operator `x += 1` maps singleton `{4.0}` to
    // singleton `{5.0}`. The preimage `{4.0} ∩ shift({5.0}, -1) = {4.0}` is a
    // singleton of width 0.
    let (task, factory) = additive_numeric_source_set_task_with_partitions(vec![
        Interval::singleton(NumericValue::new(4.0)),
        Interval::singleton(NumericValue::new(5.0)),
    ]);
    let x_abs_var = task.variables().len();
    let op = super::super::abstract_operators::AbstractOperator {
        concrete_op_ids: vec![OperatorIndex::new(0)],
        cost: NumericValue::new(1.0),
        regression_offset: 0,
        regression_preconditions: vec![ExplicitFact::numeric_variable(x_abs_var, 1)],
        preconditions: vec![ExplicitFact::numeric_variable(x_abs_var, 0)],
        changed_numeric_vars: vec![VariableIndex::from_usize(0)],
    };

    let source_sets = factory.build_abstract_source_sets(&task, &[op]).unwrap();
    let concrete = &source_sets[0].labels[0];
    assert_eq!(
        concrete.source.numeric[0],
        Interval::singleton(NumericValue::new(4.0))
    );
}

/// A refinement changes which propositional variables an abstract state can
/// constrain, so the factory's cached list has to move with it.
///
/// Regression test: the list used to be derived once in the constructor, which
/// runs before the CEGAR loop, while refinement mutated `domain_sizes` through
/// it afterwards. State regions were then stamped with a list naming only the
/// seed-split variables, so the sparse propositional overlap skipped a
/// dimension that had genuinely been narrowed, reported two disjoint regions as
/// overlapping, and let one operator's cost be claimed twice.
#[test]
fn refinement_updates_the_variables_a_cartesian_set_can_constrain() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            2,
            "p".into(),
            vec!["p0".into(), "p1".into()],
            None,
            ExplicitValueIndex::new(0),
        )],
        numeric_variables: vec![],
        goals: vec![ExplicitFact::propositional(0, 1)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![],
        operators: vec![Operator::new(
            "move".into(),
            vec![ExplicitFact::propositional(0, 0)],
            vec![Effect::new(
                vec![],
                VariableIndex::from_usize(0),
                None,
                ExplicitValueIndex::new(1),
            )],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 1),
    });

    // Start coarse: both concrete values share one abstract class, so the single
    // class covers the variable's whole domain and constrains nothing.
    let mut factory = DomainAbstractionFactory::new(
        &task,
        vec![vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(0)]],
        vec![1],
        NumericPartitions::with_partitions(vec![]),
    )
    .unwrap();

    let coarse = factory.state_region(0);
    assert!(
        coarse.constrained_props().is_empty(),
        "a one-class variable constrains nothing, got {:?}",
        coarse.constrained_props()
    );
    assert_eq!(
        coarse.propositions()[0],
        vec![ExplicitValueIndex::new(0), ExplicitValueIndex::new(1)]
    );

    // Refine: split value 1 into its own class, as CEGAR does.
    factory
        .apply_split(
            &task,
            &Split::Propositional {
                state_id: 0,
                var_id: VariableIndex::new(0),
                wanted: vec![ExplicitValueIndex::new(1)],
                witness_value: ExplicitValueIndex::new(1),
            },
        )
        .unwrap();

    for (state_hash, expected_values) in [
        (0usize, vec![ExplicitValueIndex::new(0u32)]),
        (1usize, vec![ExplicitValueIndex::new(1u32)]),
    ] {
        let refined = factory.state_region(state_hash);
        assert_eq!(
            refined.propositions()[0],
            expected_values,
            "state {state_hash} should hold only its own class after refinement"
        );
        assert_eq!(
            refined.constrained_props(),
            &[0],
            "variable 0 is split, so a state region constrains it"
        );
    }

    // The two classes are disjoint, so the regions must not overlap. This is the
    // comparison the stale list got wrong.
    let zero = factory.state_region(0);
    let one = factory.state_region(1);
    assert!(!zero.overlaps(&one));
}
