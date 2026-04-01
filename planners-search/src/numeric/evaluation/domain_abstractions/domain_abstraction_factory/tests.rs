use super::*;

use std::collections::BTreeSet;

use planners_sas::numeric::axioms::PropositionalAxiom;
use planners_sas::numeric::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator,
};
use planners_sas::numeric::numeric_task::{
    ExplicitVariable, Fact, Metric, NumericRootTask, NumericType, NumericVariable, Operator,
};

fn identity_domain_mapping_and_sizes(
    task: &dyn AbstractNumericTask,
) -> Result<(DomainMapping, Vec<i32>)> {
    let num_vars = task.get_num_variables() as usize;
    let derived_prop: HashSet<u32> = task
        .comparison_axioms()
        .iter()
        .map(|ax| ax.get_affected_var_id() as u32)
        .collect();

    let mut domain_mapping: DomainMapping = Vec::with_capacity(num_vars);
    let mut domain_sizes: Vec<i32> = Vec::with_capacity(num_vars);
    for var_id in 0..num_vars {
        if derived_prop.contains(&(var_id as u32)) {
            domain_mapping.push(vec![0, 1, 2]);
            domain_sizes.push(3);
        } else {
            let size_i32 = task
                .get_variable_domain_size(var_id as i32)
                .map_err(|e| anyhow!(e.to_string()))
                .with_context(|| format!("failed to get domain size for variable {var_id}"))?;
            ensure!(
                size_i32 > 0,
                "non-positive domain size for variable {var_id}: {size_i32}"
            );
            let size = size_i32 as usize;
            domain_mapping.push((0..size as i32).collect());
            domain_sizes.push(size_i32);
        }
    }

    Ok((domain_mapping, domain_sizes))
}

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
        let Ok(idx) = usize::try_from(*numeric_var_id) else {
            continue;
        };
        if idx >= num_numeric_vars {
            continue;
        }
        if task.numeric_variables()[idx].get_type() != &NumericType::Constant {
            continue;
        }
        let v = initial_numeric_values[idx];
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
            let dep_idx = usize::try_from(dep).map_err(|_| {
                anyhow!("regular_numeric_var_dependencies returned non-usize index: {dep}")
            })?;
            ensure!(
                dep_idx < cutpoints_by_var.len(),
                "comparison tree depends on numeric var {dep_idx}, but only {} numeric vars exist",
                cutpoints_by_var.len()
            );
            for &v in &constant_values {
                let v = NotNan::new(v).map_err(|_| anyhow!("NaN cutpoint encountered"))?;
                if v.is_finite() {
                    cutpoints_by_var[dep_idx].insert(v);
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
fn factory_splits_regular_var_at_constants_in_comparison_trees() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        0,
        2,
    )];

    let numeric_variables = vec![
        NumericVariable::new("x0".into(), NumericType::Regular, -1),
        NumericVariable::new("c10".into(), NumericType::Constant, -1),
    ];

    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];

    let op = Operator::new("op".into(), vec![Fact::new(0, 0)], vec![], vec![], 1);

    let task = NumericRootTask::new(
        4,
        Metric::new(true, -1),
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
        (0, 0),
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
        0,
        2,
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, -1),
        NumericVariable::new("y".into(), NumericType::Regular, -1),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];
    let op = Operator::new("noop".into(), vec![], vec![], vec![], 1);
    let task = NumericRootTask::new(
        4,
        Metric::new(true, -1),
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
        (0, 0),
    );

    let factory = factory_identity_cutpoints(&task).unwrap();
    let mut generator = factory.make_operator_generator(&task, true).unwrap();
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
        0,
        2,
    )];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, -1),
        NumericVariable::new("c2".into(), NumericType::Constant, -1),
        NumericVariable::new("d".into(), NumericType::Derived, -1),
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
        Metric::new(true, -1),
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
        (0, 0),
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
        regression_preconditions: vec![Fact::new(0, COMPARISON_UNKNOWN_VAL)],
        preconditions: vec![Fact::new(0, COMPARISON_UNKNOWN_VAL), Fact::new(1, 7)],
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
        0,
        0,
    )];
    let numeric_variables: Vec<NumericVariable> = vec![];
    let goals = vec![Fact::new(0, 1)];
    let op0 = Operator::new(
        "set0".into(),
        vec![Fact::new(0, 0)],
        vec![planners_sas::numeric::numeric_task::Effect::new(
            vec![],
            0,
            0,
            1,
        )],
        vec![],
        1,
    );
    let op1 = Operator::new(
        "set1".into(),
        vec![Fact::new(0, 0)],
        vec![planners_sas::numeric::numeric_task::Effect::new(
            vec![],
            0,
            0,
            1,
        )],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, -1),
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
        (0, 0),
    );

    let factory = factory_identity_cutpoints(&task).unwrap();
    let result = factory
        .compute_wildcard_plan(&task, true, false)
        .unwrap()
        .expect("plan exists");
    assert_eq!(result.wildcard_plan.len(), 1);
    assert_eq!(result.wildcard_plan[0], vec![0, 1]);
}

#[test]
fn wildcard_plan_uses_first_matching_operator_group_when_labels_uncombined() {
    let variables = vec![ExplicitVariable::new(
        2,
        "v".into(),
        vec!["v0".into(), "v1".into()],
        0,
        0,
    )];
    let numeric_variables: Vec<NumericVariable> = vec![];
    let goals = vec![Fact::new(0, 1)];
    let op0 = Operator::new(
        "set0".into(),
        vec![Fact::new(0, 0)],
        vec![planners_sas::numeric::numeric_task::Effect::new(
            vec![],
            0,
            0,
            1,
        )],
        vec![],
        1,
    );
    let op1 = Operator::new(
        "set1".into(),
        vec![Fact::new(0, 0)],
        vec![planners_sas::numeric::numeric_task::Effect::new(
            vec![],
            0,
            0,
            1,
        )],
        vec![],
        1,
    );
    let task = NumericRootTask::new(
        4,
        Metric::new(true, -1),
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
        (0, 0),
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
fn match_tree_indexes_comparison_variables() {
    let operators = vec![
        super::super::abstract_operator_generator::AbstractOperator {
            concrete_op_ids: vec![0],
            cost: 1.0,
            hash_effect: 0,
            regression_preconditions: vec![Fact::new(0, COMPARISON_TRUE_VAL)],
            preconditions: vec![Fact::new(0, COMPARISON_TRUE_VAL)],
            changed_numeric_vars: vec![],
        },
        super::super::abstract_operator_generator::AbstractOperator {
            concrete_op_ids: vec![1],
            cost: 1.0,
            hash_effect: 0,
            regression_preconditions: vec![Fact::new(0, COMPARISON_FALSE_VAL)],
            preconditions: vec![Fact::new(0, COMPARISON_FALSE_VAL)],
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
        0,
        2,
    )];
    // Two regular numeric vars with no constants -> partitions are unbounded.
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, -1),
        NumericVariable::new("y".into(), NumericType::Regular, -1),
    ];
    let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];
    let op = Operator::new("noop".into(), vec![], vec![], vec![], 1);
    let task = NumericRootTask::new(
        4,
        Metric::new(true, -1),
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
        (0, 0),
    );

    let factory = factory_identity_cutpoints(&task).unwrap();
    let table = factory
        .build_abstract_distance_table(&task, true, false)
        .unwrap();

    let mut generator = factory.make_operator_generator(&task, true).unwrap();
    let hash_multipliers = generator.hash_multipliers().to_vec();
    let domain_sizes = generator.domain_sizes().to_vec();
    let numeric_domain_sizes = generator.numeric_domain_sizes().to_vec();
    let init_hash = table.initial_state_hash;

    let mut props: Vec<Vec<i32>> = Vec::new();
    let mut nums: Vec<Vec<i32>> = Vec::new();
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
        ExplicitVariable::new(1, "trivial".into(), vec!["only".into()], 0, 0),
        ExplicitVariable::new(2, "goal".into(), vec!["off".into(), "on".into()], 1, 0),
    ];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, -1),
        variables,
        vec![],
        vec![Fact::new(1, 1)],
        vec![],
        vec![0, 0],
        vec![],
        vec![Operator::new("noop".into(), vec![], vec![], vec![], 1)],
        vec![PropositionalAxiom::new(vec![Fact::new(0, 0)], 1, 0, 1)],
        vec![],
        vec![],
        (0, 0),
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
            0,
            2,
        ),
        ExplicitVariable::new(
            3,
            "cmp1".into(),
            vec!["true".into(), "false".into(), "unknown".into()],
            0,
            2,
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, -1),
        NumericVariable::new("y".into(), NumericType::Regular, -1),
        NumericVariable::new("z".into(), NumericType::Regular, -1),
    ];
    let comparison_axioms = vec![
        ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan),
        ComparisonAxiom::new(1, 0, 2, ComparisonOperator::LessThan),
    ];
    let task = NumericRootTask::new(
        4,
        Metric::new(true, -1),
        variables,
        numeric_variables,
        vec![
            Fact::new(0, COMPARISON_TRUE_VAL),
            Fact::new(1, COMPARISON_FALSE_VAL),
        ],
        vec![],
        vec![COMPARISON_UNKNOWN_VAL, COMPARISON_UNKNOWN_VAL],
        vec![0.0, 0.0, 0.0],
        vec![Operator::new("noop".into(), vec![], vec![], vec![], 1)],
        vec![],
        comparison_axioms,
        vec![],
        (0, 0),
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
    assert_eq!(table.distances[unsorted_goal_hash as usize], 0.0);
}

#[test]
fn factory_numeric_context_recomputes_tree_reachable_derived_intervals_from_regular_vars() {
    let variables = vec![ExplicitVariable::new(
        3,
        "cmp".into(),
        vec!["true".into(), "false".into(), "unknown".into()],
        0,
        2,
    )];

    let numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, -1),
        NumericVariable::new("c2".into(), NumericType::Constant, -1),
        NumericVariable::new("d".into(), NumericType::Derived, -1),
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
        Metric::new(true, -1),
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
        (0, 0),
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

    let x_partition = 0i32;
    let c2_partition = 0i32;
    let derived_partition = 1i32;
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
