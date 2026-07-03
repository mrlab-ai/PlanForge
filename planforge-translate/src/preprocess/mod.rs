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
use std::io::{Read, Write};

use tracing::{debug, info};

use self::causal_graph::CausalGraph;
use self::domain_transition_graph::{are_dtgs_strongly_connected, build_dtgs};
use self::fact::ExplicitFact;
use self::helper_functions::{InputStream, read_preprocessed_problem_description, to_sas_writer};

pub const SAS_FILE_VERSION: i32 = 4;
pub const PRE_FILE_VERSION: i32 = SAS_FILE_VERSION;

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
    let mut outfile = File::create(output_path)
        .unwrap_or_else(|_| panic!("open output {}", output_path.display()));
    run_preprocess_args(args, &mut outfile);
}

/// In-memory entry point: take the translator's SAS+ text, return the
/// preprocessor's `output` binary as a `String`. Skips disk I/O entirely.
pub fn run_preprocess_to_string(sas_input: &str) -> String {
    let mut buf: Vec<u8> = Vec::new();
    run_preprocess_str_to_writer(sas_input, /* prune_variables = */ true, &mut buf);
    String::from_utf8(buf).expect("preprocessor output is valid UTF-8")
}

/// CLI-facing wrapper: read the SAS+ text (from a file argument or stdin),
/// run the same preprocessing pipeline as the standalone binary, and write
/// the result to `outfile`.
pub fn run_preprocess_args<W: Write>(args: &[String], outfile: &mut W) {
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
        info!("*** do not perform relevance analysis ***");
        false
    } else {
        true
    };

    run_preprocess_str_to_writer(&input, prune_variables, outfile);
}

/// Core preprocessor: parse SAS+ input from a string, run all preprocessing
/// stages, and serialize the preprocessed task into `outfile`.
pub fn run_preprocess_str_to_writer<W: Write>(input: &str, prune_variables: bool, outfile: &mut W) {
    let mut stream = InputStream::new(input.to_string());

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

    info!("Building causal graph...");
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
    debug!(
        "Metric index changed from {} to {}",
        old_metric_index, new_metric_index
    );

    info!("Building domain transition graphs...");
    let transition_graphs =
        build_dtgs(&ordered_variables, &orig_variables, &operators, &axioms_rel);

    let mut solvable_in_poly_time = false;
    if cg_acyclic {
        solvable_in_poly_time = are_dtgs_strongly_connected(&transition_graphs);
    }
    info!("solvable in poly time {}", solvable_in_poly_time);

    let mut facts = 0;
    let mut derived_vars = 0;
    for var in &ordered_variables {
        facts += var.get_range();
        if var.is_derived() {
            derived_vars += 1;
        }
    }
    info!("Preprocessor facts: {}", facts);
    info!("Preprocessor derived variables: {}", derived_vars);

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
    info!("Preprocessor task size: {}", task_size);

    info!("Writing output...");
    to_sas_writer(
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
        outfile,
    );
    info!("done");
}
