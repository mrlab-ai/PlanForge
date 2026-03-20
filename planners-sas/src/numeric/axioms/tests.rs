use super::*;
use crate::numeric::numeric_parser;
use crate::numeric::numeric_task::NumericRootTask;

fn setup_problems() -> Vec<NumericRootTask> {
    let mut problems = vec![];
    for file in std::fs::read_dir("misc/numeric_sas").unwrap() {
        let file = file.unwrap();
        if !file.file_name().to_string_lossy().contains("example1") {
            continue;
        }
        if file.path().extension().unwrap() == "sas" {
            let input = std::fs::read_to_string(file.path()).unwrap();
            let (unconsumed_input, problem) =
                numeric_parser::parse_numeric_sas_output(&input).unwrap();
            assert!(
                unconsumed_input.is_empty(),
                "Unconsumed input: {}",
                unconsumed_input
            );
            problems.push(problem);
            println!("Parsed problem from {:?}", file.path());
            break;
        }
    }
    problems
}

#[test]
fn test_axiom_evaluator_creation() {
    let problems = setup_problems();
    assert!(!problems.is_empty());

    for problem in problems {
        let mut domain_sizes = vec![];
        for var in problem.variables().iter() {
            domain_sizes.push(var.domain_size() as u64);
        }
        for numeric_var in problem.numeric_variables().iter() {
            domain_sizes.push(u64::MAX);
        }

        let state_packer = IntDoublePacker::new(&domain_sizes);
        let axiom_evaluator = AxiomEvaluator::new(&problem, &state_packer);

        let init_state = problem.get_initial_propositional_state_values();
        let mut buffer = vec![0; axiom_evaluator.state_packer.num_bins() as usize];
        for (i, value) in init_state.iter().enumerate() {
            dbg!(i, value);
            axiom_evaluator
                .state_packer
                .set(&mut buffer, i as i32, *value as u64);
        }

        dbg!(axiom_evaluator.state_packer.get(&buffer, 0));

        dbg!(&buffer);
        dbg!(problem.numeric_variables().len());
    }
}

#[test]
fn test_example1_axiom_evaluation() {
    // Load specifically example1.sas
    let input = std::fs::read_to_string("misc/numeric_sas/example1.sas").unwrap();
    let (unconsumed_input, problem) = numeric_parser::parse_numeric_sas_output(&input).unwrap();
    assert!(unconsumed_input.is_empty());

    // Set up state packer and axiom evaluator
    let mut domain_sizes = vec![];
    for var in problem.variables().iter() {
        domain_sizes.push(var.domain_size() as u64);
    }
    for numeric_var in problem.numeric_variables().iter() {
        domain_sizes.push(u64::MAX);
    }

    let state_packer = IntDoublePacker::new(&domain_sizes);
    let axiom_evaluator = AxiomEvaluator::new(&problem, &state_packer);

    // Verify axiom structure is set up correctly
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
        5,
        "Should have 5 comparison axioms"
    );
    assert_eq!(
        problem.axioms().len(),
        2,
        "Should have 2 propositional axioms"
    );

    // Set up initial state buffer
    let init_state = problem.get_initial_propositional_state_values();
    let mut buffer = vec![0; axiom_evaluator.state_packer.num_bins() as usize];

    // Pack initial propositional state into buffer
    for (i, value) in init_state.iter().enumerate() {
        axiom_evaluator
            .state_packer
            .set(&mut buffer, i as i32, *value as u64);
    }

    // Test initial state before axiom evaluation
    println!("=== Testing Example1 Axiom Evaluation ===");
    println!("Initial buffer state:");
    for i in 0..problem.variables().len() {
        let val = axiom_evaluator.state_packer.get(&buffer, i as i32);
        println!("  var {} = {}", i, val);
    }

    // Set up initial numeric state
    let mut numeric_state = problem.get_initial_numeric_state_values().clone();
    println!("Initial numeric state:");
    for (i, val) in numeric_state.iter().enumerate() {
        println!("  numeric_var_{} = {}", i, val);
    }

    // Test arithmetic axiom evaluation
    let result = axiom_evaluator.evaluate_arithmetic_axioms(&mut numeric_state);
    assert!(result.is_ok(), "Arithmetic axiom evaluation should succeed");

    println!("After arithmetic axioms:");
    for (i, val) in numeric_state.iter().enumerate() {
        println!("  numeric_var_{} = {}", i, val);
    }

    // Test comparison axiom evaluation
    let result = axiom_evaluator.evaluate_comparison_axioms(&mut buffer, &mut numeric_state);
    assert!(result.is_ok(), "Comparison axiom evaluation should succeed");

    println!("After comparison axioms:");
    for i in 0..problem.variables().len() {
        let val = axiom_evaluator.state_packer.get(&buffer, i as i32);
        println!("  var {} = {}", i, val);
    }

    // Test propositional axiom evaluation
    let result = axiom_evaluator.evaluate_propositional_axioms(&mut buffer);
    assert!(
        result.is_ok(),
        "Propositional axiom evaluation should succeed"
    );

    println!("After propositional axioms:");
    for i in 0..problem.variables().len() {
        let val = axiom_evaluator.state_packer.get(&buffer, i as i32);
        println!("  var {} = {}", i, val);
    }

    // Test complete axiom evaluation
    let mut numeric_state_copy = problem.get_initial_numeric_state_values().clone();
    let mut buffer_copy = vec![0; axiom_evaluator.state_packer.num_bins() as usize];
    for (i, value) in init_state.iter().enumerate() {
        axiom_evaluator
            .state_packer
            .set(&mut buffer_copy, i as i32, *value as u64);
    }

    let result = axiom_evaluator.evaluate(&mut buffer_copy, &mut numeric_state_copy);
    assert!(result.is_ok(), "Complete axiom evaluation should succeed");

    println!("After complete evaluation:");
    for i in 0..problem.variables().len() {
        let val = axiom_evaluator.state_packer.get(&buffer_copy, i as i32);
        println!("  var {} = {}", i, val);
    }

    // Test specific axiom behavior based on example1.sas analysis
    // The complete evaluation should actually reach the goal state!
    let var5_value = axiom_evaluator.state_packer.get(&buffer_copy, 5);
    println!("Variable 5 final value: {}", var5_value);

    let var4_value = axiom_evaluator.state_packer.get(&buffer_copy, 4);
    println!("Variable 4 final value: {}", var4_value);
    println!(
        "  numeric_var_16 = {}, numeric_var_2 = {}",
        numeric_state_copy[16], numeric_state_copy[2]
    );
    println!(
        "  Comparison result: {} >= {} = {}",
        numeric_state_copy[16],
        numeric_state_copy[2],
        numeric_state_copy[16] >= numeric_state_copy[2]
    );

    // Variables 0,1,2,3 should all be 0 (comparison results should be true)
    for i in 0..4 {
        let val = axiom_evaluator.state_packer.get(&buffer_copy, i);
        println!("Variable {} = {} (comparison axiom result)", i, val);
    }

    // The complete evaluation actually reaches the goal state where:
    // - Variable 4 becomes 0 (because numeric_var_16 becomes >= numeric_var_2)
    // - Variable 5 becomes 0 (because all conditions var1=0, var2=0, var4=0 are met)
    assert_eq!(
        var4_value, 0,
        "Variable 4 should be 0 after complete evaluation"
    );
    assert_eq!(
        var5_value, 0,
        "Variable 5 should be 0 after complete evaluation (goal reached!)"
    );

    // Verify that the goal condition is actually satisfied
    println!(
        "🎉 Goal state reached! Variable 5 = {} (required: 0)",
        var5_value
    );
}
