use super::*;

use crate::numeric_conditions::ConditionValue;
use crate::numeric_task::{
    ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts, NumericType, NumericVariable,
};
use crate::tests::*;

#[test]
fn assignment_axiom_reports_division_by_zero_distinctly() {
    let axiom = AssignmentAxiom::new(2, CalOperator::Division, 0, 1);
    let mut values = [4.0, 0.0, 99.0];

    let error = axiom
        .update_values(&mut values)
        .expect_err("division by zero must fail");

    assert!(
        format!("{error:?}").contains("DivisionByZero"),
        "unexpected error: {error:?}"
    );
}

#[test]
fn assignment_axiom_reports_the_invalid_right_operand() {
    let axiom = AssignmentAxiom::new(1, CalOperator::Sum, 0, 3);
    let mut values = [4.0, 0.0];

    let error = axiom
        .update_values(&mut values)
        .expect_err("right operand is out of bounds");

    assert!(
        format!("{error:?}").contains("index: 3"),
        "unexpected error: {error:?}"
    );
}

#[test]
fn comparison_axiom_reports_the_invalid_right_operand() {
    let axiom = ComparisonAxiom::new(0, 0, 3, ComparisonOperator::Equal);

    let error = axiom
        .is_hold(&[4.0, 0.0])
        .expect_err("right operand is out of bounds");

    assert_eq!(error.index, 3);
}

#[test]
fn complete_evaluation_runs_arithmetic_before_comparisons() {
    let problem = std::sync::Arc::new(NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            ConditionValue::DOMAIN_SIZE,
            "sum-exceeds-left".to_string(),
            vec![
                "sum-exceeds-left".to_string(),
                "not-sum-exceeds-left".to_string(),
            ],
            Some(1),
            ConditionValue::False.as_usize(),
        )],
        numeric_variables: vec![
            NumericVariable::new("left".to_string(), NumericType::Constant, None),
            NumericVariable::new("right".to_string(), NumericType::Constant, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
        ],
        goals: vec![ExplicitFact::condition(0, ConditionValue::True.as_usize())],
        mutexes: Vec::new(),
        state: vec![ConditionValue::False.as_usize()],
        numeric_state: vec![2.0, 3.0, 0.0],
        operators: Vec::new(),
        axioms: Vec::new(),
        comparison_axioms: vec![ComparisonAxiom::new(
            0,
            2,
            0,
            ComparisonOperator::GreaterThan,
        )],
        assignment_axioms: vec![AssignmentAxiom::new(2, CalOperator::Sum, 0, 1)],
        global_constraint: ExplicitFact::condition(0, ConditionValue::True.as_usize()),
    }));
    let state_packer = std::sync::Arc::new(StatePacker::new(&[ConditionValue::DOMAIN_SIZE as u64]));
    let axiom_evaluator = AxiomEvaluator::new(problem, state_packer.clone());
    let mut buffer = vec![0; state_packer.num_bins()];
    state_packer.set(&mut buffer, 0, ConditionValue::False.as_usize() as u64);
    let mut numeric_state = vec![2.0, 3.0, 0.0];

    axiom_evaluator
        .evaluate(&mut buffer, &mut numeric_state)
        .unwrap();

    assert_eq!(numeric_state[2], 5.0);
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
    let axiom_evaluator = AxiomEvaluator::new(problem.clone(), state_packer);

    // The task's initial state is already closed under its axioms: var0 is
    // derived by the unconditional propositional axiom, var1 keeps its value
    // and var2 holds the verdict of `1.0 > total_cost`, true initially.
    let init_state = problem.get_initial_propositional_state_values();
    assert_eq!(*init_state, vec![1, 1, 0]);

    let mut buffer = vec![0; axiom_evaluator.state_packer.num_bins()];
    for (i, value) in init_state.iter().enumerate() {
        axiom_evaluator
            .state_packer
            .set(&mut buffer, i, *value as u64);
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
    let axiom_evaluator = AxiomEvaluator::new(problem.clone(), state_packer);

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
    // TODO: Build unknown example1.sas to replicate the behaviour of the old test.
    // assert_eq!(
    //     problem.comparison_axioms().len(),
    //     5,
    //     "Should have 5 comparison axioms"
    // );
    // assert_eq!(
    //     problem.axioms().len(),
    //     2,
    //     "Should have 2 propositional axioms"
    // );
    //
    // // Set up initial state buffer
    // let init_state = problem.get_initial_propositional_state_values();
    // let mut buffer = vec![0; axiom_evaluator.state_packer.num_bins() as usize];
    //
    // // Pack initial propositional state into buffer
    // for (i, value) in init_state.iter().enumerate() {
    //     axiom_evaluator
    //         .state_packer
    //         .set(&mut buffer, i as i32, *value as u64);
    // }
    //
    // // Test initial state before axiom evaluation
    // println!("=== Testing Example1 Axiom Evaluation ===");
    // println!("Initial buffer state:");
    // for i in 0..problem.variables().len() {
    //     let val = axiom_evaluator.state_packer.get(&buffer, i as i32);
    //     println!("  var {} = {}", i, val);
    // }
    //
    // // Set up initial numeric state
    // let mut numeric_state = problem.get_initial_numeric_state_values().clone();
    // println!("Initial numeric state:");
    // for (i, val) in numeric_state.iter().enumerate() {
    //     println!("  numeric_var_{} = {}", i, val);
    // }
    //
    // // Test arithmetic axiom evaluation
    // let result = axiom_evaluator.evaluate_arithmetic_axioms(&mut numeric_state);
    // assert!(result.is_ok(), "Arithmetic axiom evaluation should succeed");
    //
    // println!("After arithmetic axioms:");
    // for (i, val) in numeric_state.iter().enumerate() {
    //     println!("  numeric_var_{} = {}", i, val);
    // }
    //
    // // Test comparison axiom evaluation
    // let result = axiom_evaluator.evaluate_comparison_axioms(&mut buffer, &mut numeric_state);
    // assert!(result.is_ok(), "Comparison axiom evaluation should succeed");
    //
    // println!("After comparison axioms:");
    // for i in 0..problem.variables().len() {
    //     let val = axiom_evaluator.state_packer.get(&buffer, i as i32);
    //     println!("  var {} = {}", i, val);
    // }
    //
    // // Test propositional axiom evaluation
    // let result = axiom_evaluator.evaluate_propositional_axioms(&mut buffer);
    // assert!(
    //     result.is_ok(),
    //     "Propositional axiom evaluation should succeed"
    // );
    //
    // println!("After propositional axioms:");
    // for i in 0..problem.variables().len() {
    //     let val = axiom_evaluator.state_packer.get(&buffer, i as i32);
    //     println!("  var {} = {}", i, val);
    // }
    //
    // // Test complete axiom evaluation
    // let mut numeric_state_copy = problem.get_initial_numeric_state_values().clone();
    // let mut buffer_copy = vec![0; axiom_evaluator.state_packer.num_bins() as usize];
    // for (i, value) in init_state.iter().enumerate() {
    //     axiom_evaluator
    //         .state_packer
    //         .set(&mut buffer_copy, i as i32, *value as u64);
    // }
    //
    // let result = axiom_evaluator.evaluate(&mut buffer_copy, &mut numeric_state_copy);
    // assert!(result.is_ok(), "Complete axiom evaluation should succeed");
    //
    // println!("After complete evaluation:");
    // for i in 0..problem.variables().len() {
    //     let val = axiom_evaluator.state_packer.get(&buffer_copy, i as i32);
    //     println!("  var {} = {}", i, val);
    // }
    //
    // // Test specific axiom behavior based on example1.sas analysis
    // // The complete evaluation should actually reach the goal state!
    // let var5_value = axiom_evaluator.state_packer.get(&buffer_copy, 5);
    // println!("Variable 5 final value: {}", var5_value);
    //
    // let var4_value = axiom_evaluator.state_packer.get(&buffer_copy, 4);
    // println!("Variable 4 final value: {}", var4_value);
    // println!(
    //     "  numeric_var_16 = {}, numeric_var_2 = {}",
    //     numeric_state_copy[16], numeric_state_copy[2]
    // );
    // println!(
    //     "  Comparison result: {} >= {} = {}",
    //     numeric_state_copy[16],
    //     numeric_state_copy[2],
    //     numeric_state_copy[16] >= numeric_state_copy[2]
    // );
    //
    // // Variables 0,1,2,3 should all be 0 (comparison results should be true)
    // for i in 0..4 {
    //     let val = axiom_evaluator.state_packer.get(&buffer_copy, i);
    //     println!("Variable {} = {} (comparison axiom result)", i, val);
    // }
    //
    // // The complete evaluation actually reaches the goal state where:
    // // - Variable 4 becomes 0 (because numeric_var_16 becomes >= numeric_var_2)
    // // - Variable 5 becomes 0 (because all conditions var1=0, var2=0, var4=0 are met)
    // assert_eq!(
    //     var4_value, 0,
    //     "Variable 4 should be 0 after complete evaluation"
    // );
    // assert_eq!(
    //     var5_value, 0,
    //     "Variable 5 should be 0 after complete evaluation (goal reached!)"
    // );
    //
    // // Verify that the goal condition is actually satisfied
    // println!(
    //     "🎉 Goal state reached! Variable 5 = {} (required: 0)",
    //     var5_value
    // );
}
