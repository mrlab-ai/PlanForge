use super::*;

use std::collections::BTreeSet;

use rand::{SeedableRng, rngs::SmallRng};

use crate::evaluation::abstraction_collections::cost_partitioning::{
    AbstractOperatorFootprint, ConcreteOperatorFootprint, StateRegion, TransitionResidualCosts,
    build_explicit_label_cost_partitioning_table,
};
use crate::evaluation::domain_abstractions::utils::identity_domain_mapping_and_sizes;
use planforge_sas::axioms::PropositionalAxiom;
use planforge_sas::axioms::{AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
    NumericRootTask, NumericType, NumericVariable, Operator,
};

fn constant_leaf_values(
    tree: &ComparisonTree,
    task: &dyn AbstractNumericTask,
    initial_numeric_values: &[f64],
) -> Vec<f64> {
    let num_numeric_vars = task.numeric_variables().len();
    let mut out: HashSet<u64> = HashSet::new();
    for node in &tree.nodes {
        let super::super::comparison_expression::ComparisonTreeNode::Leaf { numeric_var_id } = node
        else {
            continue;
        };
        if *numeric_var_id >= num_numeric_vars {
            continue;
        }
        if task.numeric_variables()[*numeric_var_id].get_type() != &NumericType::Constant {
            continue;
        }
        let v = initial_numeric_values[*numeric_var_id];
        if v.is_nan() {
            continue;
        }
        out.insert(v.to_bits());
    }
    out.into_iter().map(f64::from_bits).collect()
}

fn partitions_from_cutpoints(cutpoints: &[f64]) -> Vec<Interval> {
    let mut cuts: Vec<f64> = cutpoints
        .iter()
        .copied()
        .filter(|v| v.is_finite() && !v.is_nan())
        .collect();
    cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cuts.dedup_by(|a, b| a.to_bits() == b.to_bits());

    let mut out: Vec<Interval> = Vec::new();
    let mut prev = f64::NEG_INFINITY;
    for &c in &cuts {
        out.push(Interval::new(prev, c, false, false));
        out.push(Interval::singleton(c));
        prev = c;
    }
    out.push(Interval::new(prev, f64::INFINITY, false, false));
    out
}

fn cutpoint_partitions_for_task(
    task: &dyn AbstractNumericTask,
) -> Result<(NumericPartitions, Vec<usize>)> {
    let mut comparison_trees: Vec<ComparisonTree> =
        Vec::with_capacity(task.comparison_axioms().len());
    for comparison_axiom_id in 0..task.comparison_axioms().len() {
        let tree = ComparisonTree::from_task(task, comparison_axiom_id).map_err(|e| {
            anyhow!(
                "failed to build ComparisonTree for comparison axiom {comparison_axiom_id}: {e:?}"
            )
        })?;
        comparison_trees.push(tree);
    }

    let initial_numeric_values = task.get_initial_numeric_state_values();
    let num_numeric_vars = task.numeric_variables().len();

    let mut cutpoints_by_var: Vec<BTreeSet<NotNan<f64>>> = vec![BTreeSet::new(); num_numeric_vars];
    for tree in &comparison_trees {
        let constant_values = constant_leaf_values(tree, task, &initial_numeric_values);
        if constant_values.is_empty() {
            continue;
        }

        for dep in tree.regular_numeric_var_dependencies(task) {
            ensure!(
                dep < cutpoints_by_var.len(),
                "comparison tree depends on numeric var {dep}, but only {} numeric vars exist",
                cutpoints_by_var.len()
            );
            for &v in &constant_values {
                let v = NotNan::new(v).map_err(|_| anyhow!("NaN cutpoint encountered"))?;
                if v.is_finite() {
                    cutpoints_by_var[dep].insert(v);
                }
            }
        }
    }

    let mut partitions_by_numeric_var: Vec<Vec<Interval>> = Vec::with_capacity(num_numeric_vars);
    let mut numeric_domain_sizes: Vec<usize> = Vec::with_capacity(num_numeric_vars);
    for (var_id, var) in task.numeric_variables().iter().enumerate() {
        let parts = match var.get_type() {
            NumericType::Constant => vec![Interval::singleton(initial_numeric_values[var_id])],
            NumericType::Regular => {
                let cuts: Vec<f64> = cutpoints_by_var[var_id]
                    .iter()
                    .map(|v| v.into_inner())
                    .collect();
                if cuts.is_empty() {
                    vec![Interval::unbounded()]
                } else {
                    partitions_from_cutpoints(&cuts)
                }
            }
            NumericType::Derived | NumericType::Cost => vec![Interval::unbounded()],
        };
        numeric_domain_sizes.push(parts.len());
        partitions_by_numeric_var.push(parts);
    }

    Ok((
        NumericPartitions::with_partitions(partitions_by_numeric_var),
        numeric_domain_sizes,
    ))
}

fn factory_identity_cutpoints(task: &dyn AbstractNumericTask) -> Result<DomainAbstractionFactory> {
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(task)?;
    let (partitions, numeric_domain_sizes) = cutpoint_partitions_for_task(task)?;
    DomainAbstractionFactory::new(
        task,
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
    )
}

#[test]
fn transition_cost_partitioned_table_uses_abstract_transitions() {
    let variables = vec![ExplicitVariable::new(
        2,
        "p".into(),
        vec!["p0".into(), "p1".into()],
        None,
        0,
    )];
    let op = Operator::new(
        "move".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 0, None, 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 1),
    );
    let factory = factory_identity_cutpoints(&task).unwrap();
    let residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);

    let (table, tcf, transition_system) = factory
        .build_transition_cost_partitioned_distance_table(&task, false, &residuals, 0, None)
        .unwrap();

    assert_eq!(transition_system.transitions.len(), 1);
    assert_eq!(table.distances[table.initial_state_hash], 1.0);
    assert_eq!(tcf.transition_costs, vec![1.0]);
}

#[test]
fn explicit_transition_system_matches_implicit_distances_across_comparison_cascades() {
    let variables = vec![
        ExplicitVariable::new(
            3,
            "x-lt-ten".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            COMPARISON_UNKNOWN_VAL,
        ),
        ExplicitVariable::new(
            2,
            "goal".into(),
            vec!["false".into(), "true".into()],
            None,
            0,
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
            vec![ExplicitFact::new(0, COMPARISON_TRUE_VAL)],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                2,
                false,
                vec![],
            )],
            1,
        ),
        Operator::new(
            "finish".into(),
            vec![ExplicitFact::new(0, COMPARISON_FALSE_VAL)],
            vec![Effect::new(vec![], 1, Some(0), 1)],
            vec![],
            1,
        ),
    ];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![COMPARISON_UNKNOWN_VAL, 0],
        vec![0.0, 10.0, 20.0],
        operators,
        vec![],
        vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)],
        vec![],
        ExplicitFact::new(0, COMPARISON_UNKNOWN_VAL),
    );
    let factory = factory_identity_cutpoints(&task).unwrap();
    let implicit = factory
        .build_abstract_distance_table(&task, false, false)
        .unwrap();
    let mut generator = factory.make_operator_generator(&task, false).unwrap();
    let abstract_operators = generator.build_abstract_operators(&task).unwrap();
    let transition_system = factory
        .build_abstract_transition_system_from_operators_without_regions_with_deadline(
            &task,
            false,
            &abstract_operators,
            None,
        )
        .unwrap();
    let (explicit, _) =
        build_explicit_label_cost_partitioning_table(&transition_system, &[1.0, 1.0], None, None)
            .unwrap();

    assert_eq!(explicit, implicit.distances);
    assert_eq!(explicit[implicit.initial_state_hash], 2.0);
}

#[test]
fn precise_regional_table_charges_only_the_transition_source_partition() {
    let variables = vec![ExplicitVariable::new(
        2,
        "p".into(),
        vec!["p0".into(), "p1".into()],
        None,
        0,
    )];
    let op = Operator::new(
        "move".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 0, None, 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 1),
    );
    let factory = factory_identity_cutpoints(&task).unwrap();
    let mut operator_generator = factory.make_operator_generator(&task, false).unwrap();
    let abstract_operators = operator_generator.build_abstract_operators(&task).unwrap();
    assert_eq!(abstract_operators.len(), 1);
    let transition_system = factory
        .build_abstract_transition_system_from_operators_without_regions_with_deadline(
            &task,
            false,
            &abstract_operators,
            None,
        )
        .unwrap();
    let footprints = vec![AbstractOperatorFootprint {
        labels: vec![ConcreteOperatorFootprint {
            concrete_op_id: 0,
            source_region: StateRegion {
                propositions: vec![vec![0, 1]].into(),
                numeric: Vec::new().into(),
            }
            .into(),
        }],
    }];
    let mut residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);

    let (table, allocation) = factory
        .build_precise_regional_cost_partitioned_distance_table_with_deadline(
            &transition_system,
            &footprints,
            &residuals,
            0,
            None,
            None,
        )
        .unwrap();

    assert_eq!(table.distances[table.initial_state_hash], 1.0);
    assert_eq!(allocation.entries().len(), 1);
    assert_eq!(
        allocation.entries()[0].footprint.source_region.propositions[0],
        vec![0]
    );
    residuals
        .reduce_by_regional_allocation_with_deadline(&allocation, None)
        .unwrap();

    let disjoint_source = ConcreteOperatorFootprint {
        concrete_op_id: 0,
        source_region: StateRegion {
            propositions: vec![vec![1]].into(),
            numeric: Vec::new().into(),
        }
        .into(),
    };
    assert_eq!(
        residuals.cost_for_operator_footprint(1, 0, &disjoint_source),
        1.0
    );
}

#[test]
fn factory_splits_regular_var_at_constants_in_comparison_trees() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];

    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, None),
        NumericVariable::new("c10".into(), NumericType::Constant, None),
    ];

    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];

    let op = Operator::new(
        "op".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![],
        1,
    );

    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![0],
        vec![0.0, 10.0],
        vec![op],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    let factory = factory_identity_cutpoints(&task).unwrap();
    assert_eq!(factory.numeric_domain_sizes(), &[3, 1]);

    let x0_parts = factory.partitions().partitions(0).unwrap();
    assert_eq!(x0_parts.len(), 3);
    assert_eq!(
        x0_parts[0],
        Interval::new(f64::NEG_INFINITY, 10.0, false, false)
    );
    assert_eq!(x0_parts[1], Interval::singleton(10.0));
    assert_eq!(
        x0_parts[2],
        Interval::new(10.0, f64::INFINITY, false, false)
    );

    let c10_parts = factory.partitions().partitions(1).unwrap();
    assert_eq!(c10_parts, &[Interval::singleton(10.0)]);

    // Smoke-test that generator can be created.
    let _gen = factory.make_operator_generator(&task, false).unwrap();
}

#[test]
fn enumerate_states_branches_on_undecidable_comparison() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];
    let op = Operator::new("noop".into(), vec![], vec![], vec![], 1);
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![COMPARISON_UNKNOWN_VAL],
        vec![0.0, 0.0],
        vec![op],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    let factory = factory_identity_cutpoints(&task).unwrap();
    let generator = factory.make_operator_generator(&task, true).unwrap();
    let hash_multipliers = generator.hash_multipliers().to_vec();
    let init_hash = factory
        .compute_initial_state_hash_determined(
            &task,
            generator.numeric_domain_sizes(),
            &hash_multipliers,
            &[0],
        )
        .unwrap();

    // For enumeration we want the same numeric-partition assignment, but comparisons set back
    // to UNKNOWN so `enumerate_states_with_evaluated_comparisons` can branch.
    let base = factory
        .reset_comparison_vars_to_unknown_except(init_hash, &hash_multipliers, &[0], &[])
        .unwrap();
    let states = factory
        .enumerate_states_with_evaluated_comparisons(
            base,
            &task,
            generator.numeric_domain_sizes(),
            &hash_multipliers,
            &[0],
            &[],
        )
        .unwrap();
    assert_eq!(states.len(), 2);
}

#[test]
fn initial_state_hash_evaluates_derived_numeric_comparison_via_tree_inputs() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c2".into(), NumericType::Constant, None),
        NumericVariable::new("d".into(), NumericType::Derived, None),
    ];
    let assignment_axioms = vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)];
    let comparison_axioms = vec![ComparisonAxiom::new(
        0,
        2,
        1,
        ComparisonOperator::GreaterThan,
    )];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![COMPARISON_UNKNOWN_VAL],
        vec![1.0, 2.0, 0.0],
        vec![Operator::new("noop".into(), vec![], vec![], vec![], 1)],
        vec![],
        comparison_axioms,
        assignment_axioms,
        ExplicitFact::new(0, 0),
    );

    let factory = factory_identity_cutpoints(&task).unwrap();
    let generator = factory.make_operator_generator(&task, true).unwrap();
    let hash_multipliers = generator.hash_multipliers().to_vec();
    let init_hash = factory
        .compute_initial_state_hash_determined(
            &task,
            generator.numeric_domain_sizes(),
            &hash_multipliers,
            &[0],
        )
        .unwrap();

    let comparison_abs_value = init_hash / hash_multipliers[0];
    assert_eq!(comparison_abs_value, COMPARISON_TRUE_VAL);
}

#[test]
fn unknown_comparison_preconditions_are_not_treated_as_fixed() {
    let op = super::super::abstract_operator_generator::AbstractOperator {
        concrete_op_ids: vec![0],
        cost: 1.0,
        hash_effect: 0,
        regression_preconditions: vec![ExplicitFact::new(0, COMPARISON_UNKNOWN_VAL)],
        preconditions: vec![
            ExplicitFact::new(0, COMPARISON_UNKNOWN_VAL),
            ExplicitFact::new(1, 7),
        ],
        changed_numeric_vars: vec![],
    };

    let fixed = get_comparison_preconditions(&op, &[0]);
    assert!(fixed.is_empty());
}

#[test]
fn wildcard_plan_collects_all_equivalent_concrete_ops() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        Some(0),
        0,
    )];
    let numeric_variables: Vec<NumericVariable> = vec![];
    let goals = vec![ExplicitFact::new(0, 1)];
    let op0 = Operator::new(
        "set0".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 0, Some(0), 1)],
        vec![],
        1,
    );
    let op1 = Operator::new(
        "set1".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 0, Some(0), 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        goals,
        vec![],
        vec![0],
        vec![],
        vec![op0, op1],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let factory = factory_identity_cutpoints(&task).unwrap();
    let result = factory
        .compute_wildcard_plan(&task, true, false)
        .unwrap()
        .expect("plan exists");
    assert_eq!(result.wildcard_plan.len(), 1);
    let mut step = result.wildcard_plan[0].clone();
    step.sort_unstable();
    assert_eq!(step, vec![0, 1]);
}

#[test]
fn wildcard_plan_uses_first_matching_operator_group_when_labels_uncombined() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        Some(0),
        0,
    )];
    let numeric_variables: Vec<NumericVariable> = vec![];
    let goals = vec![ExplicitFact::new(0, 1)];
    let op0 = Operator::new(
        "set0".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 0, Some(0), 1)],
        vec![],
        1,
    );
    let op1 = Operator::new(
        "set1".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 0, Some(0), 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        goals,
        vec![],
        vec![0],
        vec![],
        vec![op0, op1],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let factory = factory_identity_cutpoints(&task).unwrap();
    let result = factory
        .compute_wildcard_plan(&task, false, false)
        .unwrap()
        .expect("plan exists");
    assert_eq!(result.wildcard_plan.len(), 1);
    assert_eq!(result.wildcard_plan[0], vec![0]);
}

#[test]
fn singleton_plan_is_produced_when_wildcards_are_disabled() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        Some(0),
        0,
    )];
    let numeric_variables: Vec<NumericVariable> = vec![];
    let goals = vec![ExplicitFact::new(0, 1)];
    let op0 = Operator::new(
        "set0".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 0, Some(0), 1)],
        vec![],
        1,
    );
    let op1 = Operator::new(
        "set1".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 0, Some(0), 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        goals,
        vec![],
        vec![0],
        vec![],
        vec![op0, op1],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let factory = factory_identity_cutpoints(&task).unwrap();
    let result = factory
        .compute_plan(&task, true, false, false)
        .unwrap()
        .expect("plan exists");
    assert_eq!(result.wildcard_plan.len(), 1);
    assert!(matches!(result.wildcard_plan[0].as_slice(), [0] | [1]));
}

#[test]
fn singleton_plan_selection_uses_seeded_rng() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        Some(0),
        0,
    )];
    let numeric_variables: Vec<NumericVariable> = vec![];
    let goals = vec![ExplicitFact::new(0, 1)];
    let op0 = Operator::new(
        "set0".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 0, Some(0), 1)],
        vec![],
        1,
    );
    let op1 = Operator::new(
        "set1".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 0, Some(0), 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        goals,
        vec![],
        vec![0],
        vec![],
        vec![op0, op1],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let factory = factory_identity_cutpoints(&task).unwrap();
    let mut rng = SmallRng::seed_from_u64(7);
    let result = factory
        .compute_plan_with_rng(&task, true, false, false, Some(&mut rng))
        .unwrap()
        .expect("plan exists");
    assert_eq!(result.wildcard_plan.len(), 1);
    assert!(matches!(result.wildcard_plan[0].as_slice(), [0] | [1]));
}

#[test]
fn match_tree_indexes_comparison_variables() {
    let operators = vec![
        super::super::abstract_operator_generator::AbstractOperator {
            concrete_op_ids: vec![0],
            cost: 1.0,
            hash_effect: 0,
            regression_preconditions: vec![ExplicitFact::new(0, COMPARISON_TRUE_VAL)],
            preconditions: vec![ExplicitFact::new(0, COMPARISON_TRUE_VAL)],
            changed_numeric_vars: vec![],
        },
        super::super::abstract_operator_generator::AbstractOperator {
            concrete_op_ids: vec![1],
            cost: 1.0,
            hash_effect: 0,
            regression_preconditions: vec![ExplicitFact::new(0, COMPARISON_FALSE_VAL)],
            preconditions: vec![ExplicitFact::new(0, COMPARISON_FALSE_VAL)],
            changed_numeric_vars: vec![],
        },
    ];

    let tree = MatchTree::build(&[3], &[], &[1], &operators, &[0]);
    let mut out = Vec::new();

    tree.get_applicable_operator_ids(COMPARISON_TRUE_VAL, &mut out);
    assert_eq!(out, vec![0]);

    tree.get_applicable_operator_ids(COMPARISON_FALSE_VAL, &mut out);
    assert_eq!(out, vec![1]);
}

#[test]
fn initial_state_is_unique_and_comparisons_are_determined() {
    // One comparison-axiom propositional variable.
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];
    // Two regular numeric vars with no constants -> partitions are unbounded.
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];
    let op = Operator::new("noop".into(), vec![], vec![], vec![], 1);
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        // The concrete initial state used by numeric-fd has comparisons evaluated.
        vec![COMPARISON_UNKNOWN_VAL],
        vec![0.0, 0.0],
        vec![op],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    let factory = factory_identity_cutpoints(&task).unwrap();
    let table = factory
        .build_abstract_distance_table(&task, true, false)
        .unwrap();

    let generator = factory.make_operator_generator(&task, true).unwrap();
    let hash_multipliers = generator.hash_multipliers().to_vec();
    let domain_sizes = generator.domain_sizes().to_vec();
    let numeric_domain_sizes = generator.numeric_domain_sizes().to_vec();
    let init_hash = table.initial_state_hash;

    let mut props: Vec<Vec<usize>> = Vec::new();
    let mut nums: Vec<Vec<usize>> = Vec::new();
    decode_state_to_vectors(
        init_hash,
        domain_sizes.len(),
        &domain_sizes,
        &numeric_domain_sizes,
        &hash_multipliers,
        &mut props,
        &mut nums,
    );
    assert_eq!(props.len(), 1);
    assert_eq!(props[0][0], COMPARISON_FALSE_VAL);
}

#[test]
fn abstract_goals_skip_trivial_goal_axiom_preconditions() {
    let variables = vec![
        ExplicitVariable::new(1, "trivial".into(), vec!["only".into()], Some(0), 0),
        ExplicitVariable::new(
            2,
            "goal".into(),
            vec!["off".into(), "on".into()],
            Some(1),
            0,
        ),
    ];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        vec![],
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![0, 0],
        vec![],
        vec![Operator::new("noop".into(), vec![], vec![], vec![], 1)],
        vec![PropositionalAxiom::new(
            vec![ExplicitFact::new(0, 0)],
            1,
            0,
            1,
        )],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let factory = factory_identity_cutpoints(&task).unwrap();
    let goals = factory.compute_abstract_goals(&task);
    assert!(goals.is_empty());
}

#[test]
fn comparison_enumeration_is_unsorted_and_goal_membership_still_works() {
    let variables = vec![
        ExplicitVariable::new(
            3,
            "cmp0".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
        ),
        ExplicitVariable::new(
            3,
            "cmp1".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("y".into(), NumericType::Regular, None),
        NumericVariable::new("z".into(), NumericType::Regular, None),
    ];
    let comparison_axioms = vec![
        ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan),
        ComparisonAxiom::new(1, 0, 2, ComparisonOperator::LessThan),
    ];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![
            ExplicitFact::new(0, COMPARISON_TRUE_VAL),
            ExplicitFact::new(1, COMPARISON_FALSE_VAL),
        ],
        vec![],
        vec![COMPARISON_UNKNOWN_VAL, COMPARISON_UNKNOWN_VAL],
        vec![0.0, 0.0, 0.0],
        vec![Operator::new("noop".into(), vec![], vec![], vec![], 1)],
        vec![],
        comparison_axioms,
        vec![],
        ExplicitFact::new(0, 0),
    );

    let factory = factory_identity_cutpoints(&task).unwrap();
    let generator = factory.make_operator_generator(&task, true).unwrap();
    let hash_multipliers = generator.hash_multipliers().to_vec();
    let comparison_var_ids = vec![0usize, 1usize];

    let unsorted_goal_hash = COMPARISON_TRUE_VAL + 3 * COMPARISON_FALSE_VAL;
    let states = factory
        .enumerate_states_with_evaluated_comparisons(
            unsorted_goal_hash,
            &task,
            generator.numeric_domain_sizes(),
            &hash_multipliers,
            &comparison_var_ids,
            &[],
        )
        .unwrap();

    assert_eq!(states, vec![0, 3, 1, 4]);
    assert!(states.contains(&unsorted_goal_hash));
    assert!(states.binary_search(&unsorted_goal_hash).is_err());

    let table = factory
        .build_abstract_distance_table(&task, true, false)
        .unwrap();
    assert_eq!(table.distances[unsorted_goal_hash], 0.0);
}

#[test]
fn factory_numeric_context_keeps_consistent_additive_derived_partition() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        Some(0),
        2,
    )];

    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("c2".into(), NumericType::Constant, None),
        NumericVariable::new("d".into(), NumericType::Derived, None),
    ];
    let assignment_axioms = vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)];
    let comparison_axioms = vec![ComparisonAxiom::new(
        0,
        2,
        1,
        ComparisonOperator::GreaterThan,
    )];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![0],
        vec![0.0, 2.0, 0.0],
        vec![Operator::new("noop".into(), vec![], vec![], vec![], 1)],
        vec![],
        comparison_axioms,
        assignment_axioms,
        ExplicitFact::new(0, 0),
    );

    let partitions = NumericPartitions::with_partitions(vec![
        vec![Interval::singleton(0.0), Interval::singleton(1.0)],
        vec![Interval::singleton(2.0)],
        vec![Interval::singleton(2.0), Interval::singleton(100.0)],
    ]);
    let numeric_domain_sizes = vec![2, 1, 2];
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory = DomainAbstractionFactory::new(
        &task,
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
    )
    .unwrap();
    let generator = factory.make_operator_generator(&task, true).unwrap();
    let hash_multipliers = generator.hash_multipliers().to_vec();

    let x_partition = 0;
    let c2_partition = 0;
    let derived_partition = 0;
    let state_hash = x_partition * hash_multipliers[1]
        + c2_partition * hash_multipliers[2]
        + derived_partition * hash_multipliers[3];

    let numeric_intervals = factory
        .build_numeric_intervals(
            state_hash,
            generator.numeric_domain_sizes(),
            &hash_multipliers,
            &task,
        )
        .unwrap();

    assert_eq!(numeric_intervals[0], Interval::singleton(0.0));
    assert_eq!(numeric_intervals[1], Interval::singleton(2.0));
    assert_eq!(numeric_intervals[2], Interval::singleton(2.0));
}

fn additive_numeric_footprint_task() -> (NumericRootTask, DomainAbstractionFactory) {
    let variables = vec![ExplicitVariable::new(
        1,
        "p".into(),
        vec!["p0".into()],
        None,
        0,
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
            0,
            AssignmentOperation::Plus,
            1,
            false,
            vec![],
        )],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![0],
        vec![9.0, 1.0],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(9.0, 10.0, true, true),
            Interval::new(10.0, f64::INFINITY, true, false),
        ],
        vec![Interval::singleton(1.0)],
    ]);
    let numeric_domain_sizes = vec![2, 1];
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory = DomainAbstractionFactory::new(
        &task,
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
    )
    .unwrap();

    (task, factory)
}

fn additive_numeric_footprint_task_with_partitions(
    x_partitions: Vec<Interval>,
) -> (NumericRootTask, DomainAbstractionFactory) {
    let variables = vec![ExplicitVariable::new(
        1,
        "p".into(),
        vec!["p0".into()],
        None,
        0,
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
            0,
            AssignmentOperation::Plus,
            1,
            false,
            vec![],
        )],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![0],
        vec![0.0, 1.0],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let numeric_domain_sizes = vec![x_partitions.len(), 1];
    let partitions =
        NumericPartitions::with_partitions(vec![x_partitions, vec![Interval::singleton(1.0)]]);
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory = DomainAbstractionFactory::new(
        &task,
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
    )
    .unwrap();

    (task, factory)
}

#[test]
fn abstract_operator_footprint_keeps_finite_source_when_target_reaches_tail() {
    let (task, factory) = additive_numeric_footprint_task();
    let x_abs_var = task.variables().len();
    let op = super::super::abstract_operator_generator::AbstractOperator {
        concrete_op_ids: vec![0],
        cost: 1.0,
        hash_effect: 0,
        regression_preconditions: vec![ExplicitFact::new(x_abs_var, 1)],
        preconditions: vec![ExplicitFact::new(x_abs_var, 0)],
        changed_numeric_vars: vec![0],
    };

    let footprints = factory
        .build_abstract_operator_footprints(&task, &[op])
        .unwrap();
    let concrete = &footprints[0].labels[0];

    assert_eq!(concrete.concrete_op_id, 0);
    assert_eq!(
        concrete.source_region.numeric[0],
        Interval::new(9.0, 10.0, true, true)
    );
    assert_eq!(
        concrete.source_region.numeric[1],
        Interval::new(f64::NEG_INFINITY, f64::INFINITY, false, false)
    );
}

#[test]
fn abstract_operator_footprint_tightens_source_by_inverse_target_image() {
    let (task, factory) = additive_numeric_footprint_task_with_partitions(vec![
        Interval::new(f64::NEG_INFINITY, 5.0, false, true),
        Interval::new(5.0, 10.0, false, true),
    ]);
    let x_abs_var = task.variables().len();
    let op = super::super::abstract_operator_generator::AbstractOperator {
        concrete_op_ids: vec![0],
        cost: 1.0,
        hash_effect: 0,
        regression_preconditions: vec![ExplicitFact::new(x_abs_var, 1)],
        preconditions: vec![ExplicitFact::new(x_abs_var, 0)],
        changed_numeric_vars: vec![0],
    };

    let footprints = factory
        .build_abstract_operator_footprints(&task, &[op])
        .unwrap();
    let concrete = &footprints[0].labels[0];

    assert_eq!(concrete.concrete_op_id, 0);
    assert_eq!(
        concrete.source_region.numeric[0],
        Interval::new(4.0, 5.0, false, true)
    );
}

#[test]
fn footprint_active_preimage_allows_boundary_charge() {
    let (task, factory) = additive_numeric_footprint_task_with_partitions(vec![
        Interval::new(f64::NEG_INFINITY, 5.0, false, true),
        Interval::new(5.0, 6.0, false, true),
    ]);
    let x_abs_var = task.variables().len();
    let op = super::super::abstract_operator_generator::AbstractOperator {
        concrete_op_ids: vec![0],
        cost: 1.0,
        hash_effect: 0,
        regression_preconditions: vec![ExplicitFact::new(x_abs_var, 1)],
        preconditions: vec![ExplicitFact::new(x_abs_var, 0)],
        changed_numeric_vars: vec![0],
    };

    let footprints = factory
        .build_abstract_operator_footprints(&task, &[op])
        .unwrap();
    let concrete = &footprints[0].labels[0];

    assert_eq!(
        concrete.source_region.numeric[0],
        Interval::new(4.0, 5.0, false, true)
    );
}

#[test]
fn abstract_operator_footprint_rejects_empty_inverse_target_image() {
    let (task, factory) = additive_numeric_footprint_task_with_partitions(vec![
        Interval::new(f64::NEG_INFINITY, 5.0, false, true),
        Interval::new(7.0, 10.0, false, true),
    ]);
    let x_abs_var = task.variables().len();
    let op = super::super::abstract_operator_generator::AbstractOperator {
        concrete_op_ids: vec![0],
        cost: 1.0,
        hash_effect: 0,
        regression_preconditions: vec![ExplicitFact::new(x_abs_var, 1)],
        preconditions: vec![ExplicitFact::new(x_abs_var, 0)],
        changed_numeric_vars: vec![0],
    };

    let error = factory
        .build_abstract_operator_footprints(&task, &[op])
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("empty regressed source footprint")
    );
}

#[test]
fn abstract_operator_footprint_allocates_unbounded_changed_tail() {
    let (task, factory) = additive_numeric_footprint_task();
    let x_abs_var = task.variables().len();
    let op = super::super::abstract_operator_generator::AbstractOperator {
        concrete_op_ids: vec![0],
        cost: 1.0,
        hash_effect: 0,
        regression_preconditions: vec![ExplicitFact::new(x_abs_var, 1)],
        preconditions: vec![ExplicitFact::new(x_abs_var, 1)],
        changed_numeric_vars: vec![0],
    };

    let footprints = factory
        .build_abstract_operator_footprints(&task, &[op])
        .unwrap();
    let concrete = &footprints[0].labels[0];

    assert_eq!(concrete.concrete_op_id, 0);
    assert_eq!(
        concrete.source_region.numeric[0],
        Interval::new(10.0, f64::INFINITY, true, false)
    );
}

#[test]
fn abstract_operator_footprint_allocates_operator_without_numeric_effects() {
    let variables = vec![ExplicitVariable::new(
        2,
        "saved".into(),
        vec!["false".into(), "true".into()],
        None,
        0,
    )];
    let op = Operator::new(
        "save".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 0, None, 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory = DomainAbstractionFactory::new(
        &task,
        domain_mapping,
        domain_sizes,
        NumericPartitions::with_partitions(vec![]),
        vec![],
    )
    .unwrap();
    let op = super::super::abstract_operator_generator::AbstractOperator {
        concrete_op_ids: vec![0],
        cost: 1.0,
        hash_effect: 0,
        regression_preconditions: vec![ExplicitFact::new(0, 1)],
        preconditions: vec![ExplicitFact::new(0, 0)],
        changed_numeric_vars: vec![],
    };

    let footprints = factory
        .build_abstract_operator_footprints(&task, &[op])
        .unwrap();
    let concrete = &footprints[0].labels[0];

    assert_eq!(concrete.source_region.propositions[0], vec![0]);
}

#[test]
fn abstract_operator_footprint_allows_one_finite_changed_source() {
    let variables = vec![ExplicitVariable::new(
        1,
        "p".into(),
        vec!["p0".into()],
        None,
        0,
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
            AssignmentEffect::new(0, AssignmentOperation::Plus, 2, false, vec![]),
            AssignmentEffect::new(1, AssignmentOperation::Plus, 2, false, vec![]),
        ],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![0],
        vec![0.0, 0.0, 1.0],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(0.0, 1.0, true, true),
            Interval::new(1.0, 2.0, false, true),
        ],
        vec![
            Interval::new(f64::NEG_INFINITY, 0.0, false, true),
            Interval::new(0.0, f64::INFINITY, false, false),
        ],
        vec![Interval::singleton(1.0)],
    ]);
    let numeric_domain_sizes = vec![2, 2, 1];
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory = DomainAbstractionFactory::new(
        &task,
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
    )
    .unwrap();
    let x_abs_var = task.variables().len();
    let y_abs_var = x_abs_var + 1;
    let op = super::super::abstract_operator_generator::AbstractOperator {
        concrete_op_ids: vec![0],
        cost: 1.0,
        hash_effect: 0,
        regression_preconditions: vec![
            ExplicitFact::new(x_abs_var, 1),
            ExplicitFact::new(y_abs_var, 0),
        ],
        preconditions: vec![
            ExplicitFact::new(x_abs_var, 0),
            ExplicitFact::new(y_abs_var, 0),
        ],
        changed_numeric_vars: vec![0, 1],
    };

    let footprints = factory
        .build_abstract_operator_footprints(&task, &[op])
        .unwrap();
    let concrete = &footprints[0].labels[0];

    assert_eq!(
        concrete.source_region.numeric[0],
        Interval::new(0.0, 1.0, false, true)
    );
    assert_eq!(
        concrete.source_region.numeric[1],
        Interval::new(f64::NEG_INFINITY, -1.0, false, true)
    );

    let residuals = TransitionResidualCosts::from_operator_costs(&[1.0]);
    let operator_costs =
        abstract_operator_costs_from_footprints(1, &footprints, &residuals, 0, None).unwrap();
    assert_eq!(operator_costs, vec![1.0]);
}

#[test]
fn footprint_one_finite_dim_suffices() {
    let variables = vec![
        ExplicitVariable::new(
            3,
            "x_gt_zero".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            Some(0),
            2,
        ),
        ExplicitVariable::new(
            2,
            "saved".into(),
            vec!["false".into(), "true".into()],
            None,
            0,
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
            AssignmentEffect::new(0, AssignmentOperation::Plus, 2, false, vec![]),
            AssignmentEffect::new(1, AssignmentOperation::Plus, 2, false, vec![]),
        ],
        1,
    );
    let save = Operator::new(
        "save".into(),
        vec![ExplicitFact::new(0, 0)],
        vec![Effect::new(vec![], 1, Some(0), 1)],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![2, 0],
        vec![0.0, 0.0, 0.5, 0.0],
        vec![diagonal, save],
        vec![],
        vec![ComparisonAxiom::new(
            0,
            0,
            3,
            ComparisonOperator::GreaterThan,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::singleton(0.0),
            Interval::new(0.0, f64::INFINITY, false, false),
        ],
        vec![Interval::unbounded()],
        vec![Interval::singleton(0.5)],
        vec![Interval::singleton(0.0)],
    ]);
    let numeric_domain_sizes = vec![2, 1, 1, 1];
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory = DomainAbstractionFactory::new(
        &task,
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
    )
    .unwrap();

    let x_abs_var = task.variables().len();
    let y_abs_var = x_abs_var + 1;
    let op = super::super::abstract_operator_generator::AbstractOperator {
        concrete_op_ids: vec![0],
        cost: 1.0,
        hash_effect: 0,
        regression_preconditions: vec![
            ExplicitFact::new(x_abs_var, 1),
            ExplicitFact::new(y_abs_var, 0),
        ],
        preconditions: vec![
            ExplicitFact::new(x_abs_var, 0),
            ExplicitFact::new(y_abs_var, 0),
        ],
        changed_numeric_vars: vec![0, 1],
    };
    let footprints = factory
        .build_abstract_operator_footprints(&task, &[op])
        .unwrap();
    let concrete = &footprints[0].labels[0];
    assert_eq!(concrete.source_region.numeric[0], Interval::singleton(0.0));
    assert_eq!(concrete.source_region.numeric[1], Interval::unbounded());

    let table = factory
        .build_abstract_distance_table(&task, false, false)
        .unwrap();
    assert_eq!(table.distances[table.initial_state_hash], 2.0);
}

#[test]
fn abstract_operator_footprint_ignores_zero_additive_effect_dimension() {
    let variables = vec![ExplicitVariable::new(
        1,
        "p".into(),
        vec!["p0".into()],
        None,
        0,
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
            AssignmentEffect::new(0, AssignmentOperation::Plus, 2, false, vec![]),
            AssignmentEffect::new(1, AssignmentOperation::Plus, 3, false, vec![]),
        ],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, None),
        variables,
        numeric_variables,
        vec![],
        vec![],
        vec![0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![op],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let partitions = NumericPartitions::with_partitions(vec![
        vec![
            Interval::new(0.0, 1.0, true, true),
            Interval::new(1.0, 2.0, false, true),
        ],
        vec![
            Interval::new(0.0, 1.0, true, true),
            Interval::new(1.0, 2.0, false, true),
        ],
        vec![Interval::singleton(1.0)],
        vec![Interval::singleton(0.0)],
    ]);
    let numeric_domain_sizes = vec![2, 2, 1, 1];
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory = DomainAbstractionFactory::new(
        &task,
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
    )
    .unwrap();
    let x_abs_var = task.variables().len();
    let y_abs_var = x_abs_var + 1;
    let op = super::super::abstract_operator_generator::AbstractOperator {
        concrete_op_ids: vec![0],
        cost: 1.0,
        hash_effect: 0,
        regression_preconditions: vec![
            ExplicitFact::new(x_abs_var, 1),
            ExplicitFact::new(y_abs_var, 0),
        ],
        preconditions: vec![
            ExplicitFact::new(x_abs_var, 0),
            ExplicitFact::new(y_abs_var, 0),
        ],
        changed_numeric_vars: vec![0, 1],
    };

    let footprints = factory
        .build_abstract_operator_footprints(&task, &[op])
        .unwrap();
    let concrete = &footprints[0].labels[0];

    assert_eq!(
        concrete.source_region.numeric[0],
        Interval::new(0.0, 1.0, false, true)
    );
    // y has a zero-delta additive effect, so it is not in the affected-var loop
    // that intersects with the inverse target. But y *is* pinned by the
    // operator's `y_abs_var = 0` precondition, so the footprint reflects that
    // partition's interval instead of going unbounded — which is the tightest
    // admissible footprint and the source of the cost-partitioning precision
    // benefit on operators whose preconditions reference unaffected variables.
    assert_eq!(
        concrete.source_region.numeric[1],
        Interval::new(0.0, 1.0, true, true)
    );
}

#[test]
fn footprint_width_does_not_change_valid_preimage() {
    // Same fixture as `abstract_operator_footprint_tightens_source_by_inverse_target_image`:
    // the preimage source for the operator is `(4.0, 5.0]`, width 1.0.
    let (task, factory) = additive_numeric_footprint_task_with_partitions(vec![
        Interval::new(f64::NEG_INFINITY, 5.0, false, true),
        Interval::new(5.0, 10.0, false, true),
    ]);
    let x_abs_var = task.variables().len();
    let op = super::super::abstract_operator_generator::AbstractOperator {
        concrete_op_ids: vec![0],
        cost: 1.0,
        hash_effect: 0,
        regression_preconditions: vec![ExplicitFact::new(x_abs_var, 1)],
        preconditions: vec![ExplicitFact::new(x_abs_var, 0)],
        changed_numeric_vars: vec![0],
    };

    let footprints = factory
        .build_abstract_operator_footprints(&task, &[op])
        .unwrap();
    assert_eq!(
        footprints[0].labels[0].source_region.numeric[0],
        Interval::new(4.0, 5.0, false, true)
    );
}

#[test]
fn singleton_preimage_is_preserved_exactly() {
    // Partition `x` so that the operator `x += 1` maps singleton `{4.0}` to
    // singleton `{5.0}`. The preimage `{4.0} ∩ shift({5.0}, -1) = {4.0}` is a
    // singleton of width 0.
    let (task, factory) = additive_numeric_footprint_task_with_partitions(vec![
        Interval::singleton(4.0),
        Interval::singleton(5.0),
    ]);
    let x_abs_var = task.variables().len();
    let op = super::super::abstract_operator_generator::AbstractOperator {
        concrete_op_ids: vec![0],
        cost: 1.0,
        hash_effect: 0,
        regression_preconditions: vec![ExplicitFact::new(x_abs_var, 1)],
        preconditions: vec![ExplicitFact::new(x_abs_var, 0)],
        changed_numeric_vars: vec![0],
    };

    let footprints = factory
        .build_abstract_operator_footprints(&task, &[op])
        .unwrap();
    let concrete = &footprints[0].labels[0];
    assert_eq!(concrete.source_region.numeric[0], Interval::singleton(4.0));
}
