use std::fs::File;
use std::io::Read;

use crate::axiom::{
    strip_axiom_functional_assignment, strip_axiom_functional_comparisons, strip_axiom_relationals,
    AxiomFunctionalComparison, AxiomNumericComputation, AxiomRelational,
};
use crate::causal_graph::{CausalGraph, G_DO_NOT_PRUNE_VARIABLES};
use crate::domain_transition_graph::{
    are_dtgs_strongly_connected, build_dtgs, DomainTransitionGraph,
};
use crate::helper_functions::{
    check_and_repair_empty_axiom_layers, generate_cpp_input, read_preprocessed_problem_description,
    GlobalConstraint, InputStream, Metric,
};
use crate::mutex_group::{strip_mutexes, MutexGroup};
use crate::operator::{strip_operators, Operator};
use crate::state::State;
use crate::variable::{NumericVariable, Variable};

pub fn run_preprocess(args: &[String]) {
    let mut metric = Metric {
        optimization_criterion: '<',
        index: 0,
    };
    let mut variables: Vec<*mut Variable> = Vec::new();
    let mut internal_variables: Vec<Variable> = Vec::new();
    let mut numeric_variables: Vec<*mut NumericVariable> = Vec::new();
    let mut internal_numeric_variables: Vec<NumericVariable> = Vec::new();
    let mut initial_state = State::new();
    let mut goals: Vec<(*mut Variable, i32)> = Vec::new();
    let mut mutexes: Vec<MutexGroup> = Vec::new();
    let mut operators: Vec<Operator> = Vec::new();
    let mut axioms_rel: Vec<AxiomRelational> = Vec::new();
    let mut axioms_numeric: Vec<AxiomNumericComputation> = Vec::new();
    let mut axioms_func_comp: Vec<AxiomFunctionalComparison> = Vec::new();
    let mut transition_graphs: Vec<DomainTransitionGraph> = Vec::new();
    let mut global_constraint = GlobalConstraint {
        var: std::ptr::null_mut(),
        val: 0,
    };

    let mut input = String::new();
    let mut argc = args.len();
    if args.len() == 2 {
        let mut file_content = File::open(&args[1]).expect("opening file");
        file_content.read_to_string(&mut input).expect("read file");
        argc -= 1;
    } else {
        std::io::stdin()
            .read_to_string(&mut input)
            .expect("read stdin");
    }

    if argc != 1 {
        println!("*** do not perform relevance analysis ***");
        unsafe { G_DO_NOT_PRUNE_VARIABLES = true };
    }

    let mut stream = InputStream::new(input);

    read_preprocessed_problem_description(
        &mut stream,
        &mut metric,
        &mut internal_variables,
        &mut variables,
        &mut internal_numeric_variables,
        &mut numeric_variables,
        &mut mutexes,
        &mut initial_state,
        &mut goals,
        &mut operators,
        &mut axioms_rel,
        &mut axioms_numeric,
        &mut axioms_func_comp,
        &mut global_constraint,
    );

    println!("Building causal graph...");
    let metric_var = numeric_variables[metric.index as usize];
    let (ordering, numeric_ordering, cg_acyclic, new_metric_index) = {
        let causal_graph = CausalGraph::new(
            &mut variables,
            &mut numeric_variables,
            &operators,
            &axioms_rel,
            &axioms_numeric,
            &axioms_func_comp,
            &goals,
            global_constraint,
            metric_var,
        );
        (
            causal_graph.get_variable_ordering().clone(),
            causal_graph.get_numeric_variable_ordering().clone(),
            causal_graph.is_acyclic(),
            causal_graph.get_metric_index(),
        )
    };

    let mut ordering_mut = ordering.clone();
    check_and_repair_empty_axiom_layers(&numeric_variables, &mut ordering_mut);

    let old_metric_index = metric.index;
    metric.index = new_metric_index;
    if crate::helper_functions::DEBUG {
        println!(
            "Metric index changed from {} to {}",
            old_metric_index, metric.index
        );
    }

    strip_mutexes(&mut mutexes);
    strip_operators(&mut operators);
    strip_axiom_relationals(&mut axioms_rel);
    strip_axiom_functional_comparisons(&mut axioms_func_comp);
    strip_axiom_functional_assignment(&mut axioms_numeric);

    println!("Building domain transition graphs...");
    build_dtgs(&ordering, &operators, &axioms_rel, &mut transition_graphs);

    let mut solveable_in_poly_time = false;
    if cg_acyclic {
        solveable_in_poly_time = are_dtgs_strongly_connected(&transition_graphs);
    }
    println!("solveable in poly time {}", solveable_in_poly_time);

    let mut facts = 0;
    let mut derived_vars = 0;
    for var in &ordering {
        let var_ref = unsafe { &**var };
        facts += var_ref.get_range();
        if var_ref.is_derived() {
            derived_vars += 1;
        }
    }
    println!("Preprocessor facts: {}", facts);
    println!("Preprocessor derived variables: {}", derived_vars);

    let mut task_size = ordering.len() as i32 + facts + goals.len() as i32;
    for mutex in &mutexes {
        task_size += mutex.get_encoding_size() as i32;
    }
    for op in &operators {
        task_size += op.get_encoding_size();
    }
    for axiom in &axioms_rel {
        task_size += axiom.get_encoding_size();
    }
    for axiom in &axioms_numeric {
        task_size += axiom.get_encoding_size();
    }
    for axiom in &axioms_func_comp {
        task_size += axiom.get_encoding_size();
    }
    println!("Preprocessor task size: {}", task_size);

    println!("Writing output...");
    generate_cpp_input(
        solveable_in_poly_time,
        &ordering,
        &numeric_ordering,
        &metric,
        &mutexes,
        &initial_state,
        &goals,
        &operators,
        &axioms_rel,
        &axioms_numeric,
        &axioms_func_comp,
        &global_constraint,
    );
    println!("done");
}
