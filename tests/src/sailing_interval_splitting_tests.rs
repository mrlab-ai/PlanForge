use std::path::{Path, PathBuf};

use planners_preprocess::run_preprocess_to_output;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericRootTask, NumericType};
use planners_sas::numeric::state_registry::StateRegistry;
use planners_sas::numeric::utils::int_packer::IntDoublePacker;
use planners_search::numeric::evaluation::domain_abstractions::comparison_expression::Interval;
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction::NumericPartitions;
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_factory::DomainAbstractionFactory;
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_generator::{
    DomainAbstraction, compute_hash_multipliers,
};
use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_heuristic::DomainAbstractionHeuristic;
use planners_search::numeric::evaluation::evaluator::EvaluationState;
use planners_search::numeric::evaluation::heuristic::Heuristic;
use planners_translator::translate_to_sas_to_path_fast;

type DomainMapping = Vec<Vec<usize>>;

#[derive(Copy, Clone)]
struct SailingUvVarIds {
    u_b0: usize,
    v_b0: usize,
    u_b1: usize,
    v_b1: usize,
}

fn unique_temp_dir(prefix: &str) -> std::io::Result<PathBuf> {
    let base = std::env::temp_dir().join("numeric_planneRS");
    std::fs::create_dir_all(&base)?;

    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let dir = base.join(format!("{prefix}_{pid}_{nanos}"));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn preprocess_sailing_problem() -> PathBuf {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let domain = workspace_root.join("others/sailing/domain.pddl");
    let problem = workspace_root.join("others/sailing/prob_2_2_1229.pddl");

    let temp_dir = unique_temp_dir("sailing_interval_splitting")
        .unwrap_or_else(|e| panic!("failed to create temp dir: {e}"));
    let output_sas = temp_dir.join("output.sas");
    let preprocessed = temp_dir.join("output");

    translate_to_sas_to_path_fast(
        domain
            .to_str()
            .unwrap_or_else(|| panic!("non-utf8 domain path: {domain:?}")),
        problem
            .to_str()
            .unwrap_or_else(|| panic!("non-utf8 problem path: {problem:?}")),
        &output_sas,
    )
    .unwrap_or_else(|e| panic!("translate failed for sailing instance: {e}"));

    run_preprocess_to_output(
        &[
            "preprocess".to_string(),
            output_sas.to_string_lossy().to_string(),
        ],
        &preprocessed,
    );

    preprocessed
}

fn trivial_domain_mapping_and_sizes(task: &dyn AbstractNumericTask) -> (DomainMapping, Vec<usize>) {
    let mut domain_mapping: DomainMapping = Vec::with_capacity(task.get_num_variables());
    let mut domain_sizes: Vec<usize> = Vec::with_capacity(task.get_num_variables());

    for var_id in 0..task.get_num_variables() {
        let concrete_size = task
            .get_variable_domain_size(var_id)
            .unwrap_or_else(|e| panic!("get_variable_domain_size({var_id}) failed: {e}"));
        domain_mapping.push(vec![0; concrete_size]);
        domain_sizes.push(1);
    }

    (domain_mapping, domain_sizes)
}

fn find_numeric_var_id_named(task: &dyn AbstractNumericTask, name: &str) -> usize {
    task.numeric_variables()
        .iter()
        .position(|var| var.name() == name)
        .unwrap_or_else(|| panic!("could not find numeric variable named {name}"))
}

fn find_propositional_var_named(task: &dyn AbstractNumericTask, name: &str) -> usize {
    (0..task.get_num_variables())
        .find(|&var_id| {
            task.get_variable_name(var_id)
                .is_ok_and(|var_name| var_name == name)
        })
        .unwrap_or_else(|| panic!("could not find propositional variable named {name}"))
}

fn sailing_uv_var_ids(task: &dyn AbstractNumericTask) -> SailingUvVarIds {
    SailingUvVarIds {
        u_b0: find_numeric_var_id_named(
            task,
            "PNE derived!sum_PNE x(?boat_0)_PNE y(?boat_0)(b0, b0)",
        ),
        v_b0: find_numeric_var_id_named(
            task,
            "PNE derived!difference_PNE y(?boat_0)_PNE x(?boat_0)(b0, b0)",
        ),
        u_b1: find_numeric_var_id_named(
            task,
            "PNE derived!sum_PNE x(?boat_0)_PNE y(?boat_0)(b1, b1)",
        ),
        v_b1: find_numeric_var_id_named(
            task,
            "PNE derived!difference_PNE y(?boat_0)_PNE x(?boat_0)(b1, b1)",
        ),
    }
}

fn partitions_from_integer_upper_bounds(upper_bounds: &[i32]) -> Vec<Interval> {
    assert!(
        !upper_bounds.is_empty(),
        "expected at least one upper bound for interval construction"
    );

    let mut partitions = Vec::with_capacity(upper_bounds.len() + 1);
    let first_cut = f64::from(upper_bounds[0]) + 0.5;
    partitions.push(Interval::new(f64::NEG_INFINITY, first_cut, false, false));

    let mut previous_cut = first_cut;
    for upper in upper_bounds.iter().skip(1).copied() {
        let next_cut = f64::from(upper) + 0.5;
        partitions.push(Interval::new(previous_cut, next_cut, true, false));
        previous_cut = next_cut;
    }
    partitions.push(Interval::new(previous_cut, f64::INFINITY, true, false));

    partitions
}

fn sailing_46_partitions(
    task: &dyn AbstractNumericTask,
    var_ids: SailingUvVarIds,
) -> (NumericPartitions, Vec<usize>) {
    let initial_numeric_values = task.get_initial_numeric_state_values();
    let mut partitions_by_numeric_var: Vec<Vec<Interval>> = task
        .numeric_variables()
        .iter()
        .enumerate()
        .map(|(var_id, variable)| match variable.get_type() {
            NumericType::Constant => vec![Interval::singleton(initial_numeric_values[var_id])],
            _ => vec![Interval::unbounded()],
        })
        .collect();

    let plus_seven_intervals = partitions_from_integer_upper_bounds(&[
        7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 46, 58, 70, 82, 94, 109,
    ]);
    let minus_seven_intervals = partitions_from_integer_upper_bounds(&[
        -7, -4, -1, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 38, 44, 50, 56, 62, 68, 74, 80, 86,
        92, 98, 104, 107, 109,
    ]);

    partitions_by_numeric_var[var_ids.u_b0] = plus_seven_intervals.clone();
    partitions_by_numeric_var[var_ids.v_b1] = plus_seven_intervals;
    partitions_by_numeric_var[var_ids.v_b0] = minus_seven_intervals.clone();
    partitions_by_numeric_var[var_ids.u_b1] = minus_seven_intervals;

    let numeric_domain_sizes = partitions_by_numeric_var.iter().map(Vec::len).collect();
    (
        NumericPartitions::with_partitions(partitions_by_numeric_var),
        numeric_domain_sizes,
    )
}

#[test]
fn handcrafted_sailing_interval_splitting_reports_initial_h_45() {
    let preprocessed = preprocess_sailing_problem();
    let task = NumericRootTask::from_file(&preprocessed);
    let var_ids = sailing_uv_var_ids(&task);

    let (mut domain_mapping, mut domain_sizes) = trivial_domain_mapping_and_sizes(&task);
    let saved_p0_var_id = find_propositional_var_named(&task, "var1");
    let saved_p1_var_id = find_propositional_var_named(&task, "var2");
    for var_id in [saved_p0_var_id, saved_p1_var_id] {
        let concrete_size = task
            .get_variable_domain_size(var_id)
            .unwrap_or_else(|e| panic!("get_variable_domain_size({var_id}) failed: {e}"));
        domain_mapping[var_id] = (0..concrete_size).collect();
        domain_sizes[var_id] = concrete_size;
    }

    let (partitions, numeric_domain_sizes) = sailing_46_partitions(&task, var_ids);
    let abstract_size: u128 = domain_sizes
        .iter()
        .chain(numeric_domain_sizes.iter())
        .map(|&size| size as u128)
        .product();
    assert_eq!(abstract_size, 972_196);

    let factory = DomainAbstractionFactory::new(
        &task,
        domain_mapping,
        domain_sizes,
        partitions,
        numeric_domain_sizes,
    )
    .expect("manual sailing abstraction should build");
    let distance_table = factory
        .build_abstract_distance_table(&task, false, false)
        .expect("manual sailing abstraction distance table should build");

    let abstraction = DomainAbstraction {
        hash_multipliers: compute_hash_multipliers(
            factory.domain_sizes(),
            factory.numeric_domain_sizes(),
        )
        .expect("hash multipliers should fit"),
        factory,
        distance_table,
        combine_labels: false,
    };
    let heuristic = DomainAbstractionHeuristic::new(
        Some("sailing_interval_splitting_46".to_string()),
        abstraction,
    );

    let packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &packer);
    let mut registry = StateRegistry::new(&task, &packer, &axiom_evaluator);
    let initial_state = registry.get_initial_state();
    let eval_state =
        EvaluationState::new_with_registry(&initial_state, 0.0, false, &task, &registry);
    let initial_h = heuristic
        .compute_heuristic(&eval_state)
        .expect("initial heuristic evaluation should succeed");

    println!("handcrafted sailing interval-splitting initial_h={initial_h}");
    assert_eq!(initial_h, 45.0);

    let temp_dir = preprocessed
        .parent()
        .expect("preprocessed output should have parent directory");
    let _ = std::fs::remove_dir_all(temp_dir);
}
