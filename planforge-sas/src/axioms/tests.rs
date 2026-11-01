use super::*;

use crate::numeric_conditions::ConditionValue;
use crate::numeric_task::{
    ExplicitVariable, Metric, NumericType, NumericVariable, Task, TaskParts,
};
use crate::tests::*;

#[test]
fn assignment_axiom_reports_division_by_zero_distinctly() {
    let axiom = AssignmentAxiom::new(
        VariableIndex::new(2),
        CalOperator::Division,
        VariableIndex::new(0),
        VariableIndex::new(1),
    );
    let mut values = [
        NumericValue::new(4.0),
        NumericValue::new(0.0),
        NumericValue::new(99.0),
    ];

    let error = axiom
        .update_values(&mut values)
        .expect_err("division by zero must fail");

    assert!(
        format!("{error:?}").contains("DivisionByZero"),
        "unexpected error: {error:?}"
    );
}

/// A task whose condition "left + right > zero" reads the derived sum.
fn sum_condition_task_parts() -> TaskParts {
    TaskParts {
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            ConditionValue::DOMAIN_SIZE,
            "sum-exceeds-left".to_string(),
            vec![
                "sum-exceeds-left".to_string(),
                "not-sum-exceeds-left".to_string(),
            ],
            Some(1),
            ExplicitValueIndex::new(ConditionValue::False.as_u32()),
        )],
        numeric_variables: vec![
            NumericVariable::new("left".to_string(), NumericType::Regular, None),
            NumericVariable::new("right".to_string(), NumericType::Regular, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
            NumericVariable::new("zero".to_string(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::condition(0, ConditionValue::True.as_usize())],
        mutexes: Vec::new(),
        state: vec![ExplicitValueIndex::new(ConditionValue::False.as_u32())],
        numeric_state: vec![
            NumericValue::new(2.0),
            NumericValue::new(3.0),
            NumericValue::new(0.0),
            NumericValue::new(0.0),
        ],
        operators: Vec::new(),
        axioms: Vec::new(),
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(2),
            VariableIndex::new(3),
            ComparisonOperator::GreaterThan,
        )],
        assignment_axioms: vec![AssignmentAxiom::new(
            VariableIndex::new(2),
            CalOperator::Sum,
            VariableIndex::new(0),
            VariableIndex::new(1),
        )],
        global_constraint: Some(ExplicitFact::condition(0, ConditionValue::True.as_usize())),
    }
}

#[test]
#[should_panic(expected = "assignment axiom 2 names numeric variable 7")]
fn task_rejects_an_assignment_axiom_operand_out_of_range() {
    let mut parts = sum_condition_task_parts();
    parts.assignment_axioms[0] = AssignmentAxiom::new(
        VariableIndex::new(2),
        CalOperator::Sum,
        VariableIndex::new(0),
        VariableIndex::new(7),
    );
    Task::new(parts);
}

#[test]
#[should_panic(expected = "comparison axiom 0 names numeric variable 7")]
fn task_rejects_a_comparison_axiom_operand_out_of_range() {
    let mut parts = sum_condition_task_parts();
    parts.comparison_axioms[0].right_hand_side = VariableIndex::new(7);
    Task::new(parts);
}

#[test]
fn complete_evaluation_runs_arithmetic_before_comparisons() {
    let problem = std::sync::Arc::new(Task::new(sum_condition_task_parts()));
    let state_packer = std::sync::Arc::new(StatePacker::new(&[ConditionValue::DOMAIN_SIZE as u64]));
    let result_var = problem.comparison_axioms()[0].get_left_var_id();
    let mut numeric_state = problem.initial_numeric_values().to_vec();
    numeric_state[result_var.index()] = ZERO_VALUE;
    let axiom_evaluator = AxiomEvaluator::new(&problem, state_packer.clone());
    let mut buffer = vec![0; state_packer.num_bins()];
    state_packer.set(&mut buffer, 0, ConditionValue::False.as_usize() as u64);

    axiom_evaluator
        .evaluate(&mut buffer, &mut numeric_state)
        .unwrap();

    assert_eq!(numeric_state[result_var.index()], NumericValue::new(5.0));
    assert_eq!(
        state_packer.get(&buffer, 0),
        ConditionValue::True.as_usize() as u64
    );
}

#[test]
fn test_axiom_evaluator_creation() {
    let problem = std::sync::Arc::new(get_root_task());
    let mut domain_sizes = vec![];
    for var in problem.variables().iter() {
        domain_sizes.push(var.domain_size() as u64);
    }
    for _numeric_var in problem.numeric_variables().iter() {
        domain_sizes.push(u64::MAX);
    }

    let state_packer = std::sync::Arc::new(StatePacker::new(&domain_sizes));
    let axiom_evaluator = AxiomEvaluator::new(&problem, state_packer);

    // The task's initial state is already closed under its axioms: var0 is
    // derived by the unconditional propositional axiom, var1 keeps its value
    // and var2 holds the verdict of `1.0 > total_cost`, true initially.
    let init_state = problem.initial_propositional_values();
    assert_eq!(
        *init_state,
        vec![
            ExplicitValueIndex::new(1),
            ExplicitValueIndex::new(1),
            ExplicitValueIndex::new(0)
        ]
    );

    let mut buffer = vec![0; axiom_evaluator.state_packer.num_bins()];
    for (i, value) in init_state.iter().enumerate() {
        axiom_evaluator
            .state_packer
            .set(&mut buffer, i, value.index() as u64);
    }

    assert_eq!(axiom_evaluator.state_packer.get(&buffer, 0), 1);
    assert_eq!(axiom_evaluator.state_packer.get(&buffer, 1), 1);
    assert_eq!(axiom_evaluator.state_packer.get(&buffer, 2), 0);

    assert_eq!(problem.numeric_variables().len(), 2);
}

#[test]
fn test_example1_axiom_evaluation() {
    let problem = std::sync::Arc::new(get_root_task());

    // Set up state packer and axiom evaluator
    let mut domain_sizes = vec![];
    for var in problem.variables().iter() {
        domain_sizes.push(var.domain_size() as u64);
    }
    for _numeric_var in problem.numeric_variables().iter() {
        domain_sizes.push(u64::MAX);
    }

    let state_packer = std::sync::Arc::new(StatePacker::new(&domain_sizes));
    let axiom_evaluator = AxiomEvaluator::new(&problem, state_packer);

    // Verify axiom structure is set up correctly.
    assert!(
        axiom_evaluator.has_numeric_axioms(),
        "Should have numeric axioms"
    );
    assert!(
        axiom_evaluator.has_propositional_axioms(),
        "Should have propositional axioms"
    );
    assert_eq!(
        problem.comparison_axioms().len(),
        1,
        "Should have 1 comparison axioms"
    );
    assert_eq!(
        problem.axioms().len(),
        1,
        "Should have 1 propositional axioms"
    );
}

#[test]
fn assignment_axioms_put_derived_values_on_the_canonical_lattice() {
    // 0.1 + 0.2 is 0.30000000000000004 in f64; the derived difference must
    // be exactly zero so that `difference <= 0` holds.
    let axiom = AssignmentAxiom::new(
        VariableIndex::new(2),
        CalOperator::Difference,
        VariableIndex::new(0),
        VariableIndex::new(1),
    );
    let mut values = [
        NumericValue::new(0.1) + NumericValue::new(0.2),
        NumericValue::new(0.3),
        NumericValue::new(99.0),
    ];
    assert_eq!(values[0], values[1]);

    assert_eq!(axiom.update_values(&mut values).unwrap(), ZERO_VALUE);
    assert_eq!(values[2], ZERO_VALUE);
}
