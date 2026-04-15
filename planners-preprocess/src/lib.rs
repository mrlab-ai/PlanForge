pub mod axiom;
pub mod causal_graph;
pub mod domain_transition_graph;
pub mod fact;
pub mod helper_functions;
pub mod max_dag;
pub mod mutex_group;
pub mod operator;
pub mod scc;
pub mod state;
pub mod successor_generator;
pub mod variable;

use std::fs::File;
use std::io::Read;

use crate::causal_graph::CausalGraph;
use crate::domain_transition_graph::{are_dtgs_strongly_connected, build_dtgs};
use crate::fact::ExplicitFact;
use crate::helper_functions::{InputStream, read_preprocessed_problem_description, to_sas_at_path};

pub const SAS_FILE_VERSION: i32 = 4;
pub const PRE_FILE_VERSION: i32 = SAS_FILE_VERSION;

pub const DEBUG: bool = false;

#[derive(Debug, Clone)]
pub struct Metric {
    pub optimization_criterion: char,
    pub index: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct GlobalConstraint {
    pub var: usize,
    pub value: usize,
}
pub type Condition = Vec<ExplicitFact>;

pub fn run_preprocess(args: &[String]) {
    run_preprocess_to_output(args, std::path::Path::new("output"));
}

pub fn run_preprocess_to_output(args: &[String], output_path: &std::path::Path) {
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

    let prune_variables = if argc != 1 {
        println!("*** do not perform relevance analysis ***");
        false
    } else {
        true
    };

    let mut stream = InputStream::new(input);

    let (
        mut metric,
        variables,
        numeric_variables,
        mutexes,
        initial_state,
        goals,
        operators,
        axioms_rel,
        axioms_func_comp,
        axioms_numeric,
        global_constraint,
    ) = read_preprocessed_problem_description(&mut stream);

    println!("Building causal graph...");
    let old_metric_index = metric.index;
    let (
        orig_variables,
        orig_numeric_variables,
        ordered_variables,
        ordered_numeric_variables,
        operators,
        axioms_rel,
        axioms_numeric,
        axioms_func_comp,
        mutexes,
        goals,
        global_constraint,
        cg_acyclic,
        new_metric_index,
    ) = CausalGraph::new(
        variables,
        numeric_variables,
        operators,
        axioms_rel,
        axioms_numeric,
        axioms_func_comp,
        mutexes,
        goals,
        global_constraint,
        metric.index,
        prune_variables,
    )
    .finalize();

    metric.index = new_metric_index;
    if DEBUG {
        println!(
            "Metric index changed from {} to {}",
            old_metric_index, new_metric_index
        );
    }

    println!("Building domain transition graphs...");
    let transition_graphs =
        build_dtgs(&ordered_variables, &orig_variables, &operators, &axioms_rel);

    let mut solvable_in_poly_time = false;
    if cg_acyclic {
        solvable_in_poly_time = are_dtgs_strongly_connected(&transition_graphs);
    }
    println!("solvable in poly time {}", solvable_in_poly_time);

    let mut facts = 0;
    let mut derived_vars = 0;
    for var in &ordered_variables {
        facts += var.get_range();
        if var.is_derived() {
            derived_vars += 1;
        }
    }
    println!("Preprocessor facts: {}", facts);
    println!("Preprocessor derived variables: {}", derived_vars);

    let mut task_size =
        ordered_variables.len() + ordered_numeric_variables.len() + facts + goals.len();
    for mutex in &mutexes {
        task_size += mutex.get_encoding_size();
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
    to_sas_at_path(
        &orig_variables,
        &orig_numeric_variables,
        &ordered_variables,
        &ordered_numeric_variables,
        &metric,
        &mutexes,
        &initial_state,
        &goals,
        &operators,
        &axioms_rel,
        &axioms_numeric,
        &axioms_func_comp,
        &global_constraint,
        output_path,
    );
    println!("done");
}
