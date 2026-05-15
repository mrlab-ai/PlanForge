use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use planforge_preprocess::run_preprocess_to_output;
use planforge_sas::numeric::axioms::AxiomEvaluator;
use planforge_sas::numeric::numeric_task::AbstractNumericTask;
use planforge_sas::numeric::numeric_task::NumericRootTask;
use planforge_sas::numeric::state_registry::StateRegistry;
use planforge_sas::numeric::utils::int_packer::IntDoublePacker;
use planforge_search::numeric::evaluation::evaluator::EvaluationState;
use planforge_search::numeric::evaluation::evaluator::Evaluator;
use planforge_search::numeric::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LandmarkCutNumericHeuristic;
use planforge_search::numeric::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig;
use planforge_search::numeric::evaluation::numeric_landmarks::numeric_lm_cut_landmarks::LandmarkCutLandmarks;
use planforge_search::numeric::search_engine::SearchStatus;
use planforge_search::numeric::search_engine::{AStarSearch, SearchEngine};
use planforge_search::numeric::successor_generator::GroundedSuccessorGenerator;
use planforge_translator::translate_to_sas_to_path_fast;

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

fn find_single_file(dir: &Path, predicate: impl Fn(&Path) -> bool) -> PathBuf {
    let mut matches: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir failed for {dir:?}: {e}"))
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && predicate(path))
        .collect();

    matches.sort();

    match matches.as_slice() {
        [only] => only.clone(),
        [] => panic!("no matching file in {dir:?}"),
        _ => panic!(
            "expected exactly 1 matching file in {dir:?}, got {}",
            matches.len()
        ),
    }
}

fn parse_fd_plan_cost_from_log(log_file: &Path) -> Option<f64> {
    let content = std::fs::read_to_string(log_file).ok()?;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("Plan cost:") {
            let value = rest.trim();
            if let Ok(cost) = value.parse::<f64>() {
                return Some(cost);
            }
        }
    }
    None
}

fn expected_plan_cost(stats_file: &Path) -> f64 {
    let content = std::fs::read_to_string(stats_file)
        .unwrap_or_else(|e| panic!("read_to_string failed for {stats_file:?}: {e}"));
    let json: serde_json::Value = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("invalid json {stats_file:?}: {e}"));

    let solution_found = json
        .get("stats")
        .and_then(|s| s.get("solution_found"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(
        solution_found,
        "expected solution_found=true in {stats_file:?}"
    );

    // If `stats.plan_cost` is missing (older captures), try reading it from
    // the referenced log.
    // Otherwise, fall back to plan_length (unit-cost tasks).
    if let Some(v) = json.get("stats").and_then(|s| s.get("plan_cost")) {
        if let Some(f) = v.as_f64() {
            return f;
        }
        if let Some(u) = v.as_u64() {
            return u as f64;
        }
    }

    if let Some(log_file) = json.get("log_file").and_then(|v| v.as_str()) {
        let log_path = {
            let p = Path::new(log_file);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                Path::new(env!("CARGO_MANIFEST_DIR")).join(p)
            }
        };
        if let Some(cost) = parse_fd_plan_cost_from_log(&log_path) {
            return cost;
        }
    }

    json.get("stats")
        .and_then(|s| s.get("plan_length"))
        .and_then(|v| v.as_u64())
        .map(|v| v as f64)
        .unwrap_or_else(|| {
            panic!("missing stats.plan_cost/log_file and stats.plan_length in {stats_file:?}")
        })
}

fn expected_plan_length(stats_file: &Path) -> u64 {
    let content = std::fs::read_to_string(stats_file)
        .unwrap_or_else(|e| panic!("read_to_string failed for {stats_file:?}: {e}"));
    let json: serde_json::Value = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("invalid json {stats_file:?}: {e}"));

    let solution_found = json
        .get("stats")
        .and_then(|s| s.get("solution_found"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(
        solution_found,
        "expected solution_found=true in {stats_file:?}"
    );

    json.get("stats")
        .and_then(|s| s.get("plan_length"))
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| panic!("missing stats.plan_length in {stats_file:?}"))
}

#[test]
fn fd_blind_plan_cost_matches_misc_benchmarks() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/numeric-pddl-files");

    let mut benchmark_dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("read_dir failed for {root:?}: {e}"))
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();

    benchmark_dirs.sort();
    assert_eq!(
        benchmark_dirs.len(),
        14,
        "expected 14 benchmark folders under {root:?}"
    );

    let mut mismatches: Vec<String> = Vec::new();

    for bench_dir in benchmark_dirs {
        let bench_name = bench_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("<unknown>")
            .to_string();

        if bench_name.starts_with("minecraft-") {
            eprintln!("Skipping {bench_name} (too slow in debug test mode)");
            continue;
        }

        let domain = bench_dir.join("domain.pddl");
        assert!(domain.is_file(), "missing domain.pddl in {bench_dir:?}");

        let problem = find_single_file(&bench_dir, |path| {
            path.extension()
                .is_some_and(|ext| ext == OsStr::new("pddl"))
                && path
                    .file_name()
                    .is_some_and(|name| name != OsStr::new("domain.pddl"))
        });

        let stats = find_single_file(&bench_dir, |path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".fd_blind.json"))
        });

        let expected_cost = expected_plan_cost(&stats);
        let expected_len = expected_plan_length(&stats);

        let temp_dir = unique_temp_dir(&format!("fd_blind_{bench_name}"))
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
        .unwrap_or_else(|e| panic!("translate failed for {bench_name}: {e}"));

        run_preprocess_to_output(
            &[
                "preprocess".to_string(),
                output_sas.to_string_lossy().to_string(),
            ],
            &preprocessed,
        );

        let (actual_cost, actual_len): (Option<f64>, Option<u64>) = {
            let task = NumericRootTask::from_file(&preprocessed);
            let state_packer = IntDoublePacker::from_task(&task);
            let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
            let state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);

            let result = {
                let task_ref: &dyn AbstractNumericTask = &task;
                let mut search = AStarSearch::new(task_ref, state_registry, None, None, None);
                search.search()
            };

            match (&result.status, &result.plan) {
                (SearchStatus::Solved(_), Some(plan)) => {
                    let len: u64 = plan.len() as u64;
                    let cost: Option<f64> = result
                        .solution_cost
                        .or_else(|| Some(plan.iter().map(|op| op.cost() as f64).sum()));
                    (cost, Some(len))
                }
                _ => (None, None),
            }
        };

        if let (Some(len), Some(cost)) = (actual_len, actual_cost) {
            eprintln!(
                "{bench_name}: len={len} cost={cost:.6} (expected len={expected_len} cost={expected_cost:.6})"
            );
        }

        match (actual_len, actual_cost) {
            (Some(len), Some(cost))
                if len == expected_len && (cost - expected_cost).abs() <= 1e-3 => {}
            (Some(len), Some(cost)) => mismatches.push(format!(
                "{bench_name}: expected (len={expected_len}, cost={expected_cost:.6}), got (len={len}, cost={cost:.6})"
            )),
            _ => mismatches.push(format!(
                "{bench_name}: expected (len={expected_len}, cost={expected_cost:.6}), but search did not return a solved plan"
            )),
        }

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    assert!(
        mismatches.is_empty(),
        "plan cost mismatches:\n{}",
        mismatches.join("\n")
    );
}

#[test]
fn plant_watering_lmcutnumeric_initial_state_is_finite_and_bounded_by_optimum() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/numeric-pddl-files/plant-watering");
    let domain = root.join("domain.pddl");
    let problem = root.join("prob_4_1_1.pddl");
    let expected_optimal_cost = 13.0;

    let temp_dir = unique_temp_dir("plant_watering_lmcut_initial")
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
    .unwrap_or_else(|e| panic!("translate failed for Plant Watering: {e}"));

    run_preprocess_to_output(
        &[
            "preprocess".to_string(),
            output_sas.to_string_lossy().to_string(),
        ],
        &preprocessed,
    );

    let (dead_end, total_cost) = {
        let task = NumericRootTask::from_file(&preprocessed);
        let state_packer = IntDoublePacker::from_task(&task);
        let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
        let mut state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);
        let initial_state = state_registry.get_initial_state();

        let mut propositional_values = Vec::new();
        let mut numeric_values = Vec::new();
        state_registry
            .fill_state_and_numeric_vars(
                &initial_state,
                &mut propositional_values,
                &mut numeric_values,
            )
            .expect("initial Plant Watering state should unpack successfully");

        let mut landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());
        let (dead_end, total_cost, _cuts) = landmarks
            .compute_landmarks(
                &propositional_values,
                initial_state.buffer(&state_registry).len(),
                &numeric_values,
                false,
            )
            .expect("Plant Watering initial LM-cut computation should finish");
        (dead_end, total_cost)
    };

    let _ = std::fs::remove_dir_all(&temp_dir);

    assert!(
        !dead_end,
        "Plant Watering initial state should not be a dead end for lmcutnumeric"
    );
    assert!(
        total_cost.is_finite(),
        "Plant Watering initial lmcutnumeric value should be finite, got {total_cost}"
    );
    assert!(
        total_cost <= expected_optimal_cost + 1e-6,
        "Plant Watering initial lmcutnumeric value should be <= {expected_optimal_cost}, got {total_cost}"
    );
}

#[test]
#[ignore = "local ipc2023 drone fixture repro"]
fn drone_pfile1_lmcutnumeric_initial_state_local_repro() {
    let domain = Path::new("/home/markus/data/ipc2023/drone/domain.pddl");
    let problem = Path::new("/home/markus/data/ipc2023/drone/pfile1.pddl");

    if !domain.is_file() || !problem.is_file() {
        eprintln!("Skipping local drone repro; fixture files are unavailable");
        return;
    }

    let temp_dir = unique_temp_dir("drone_pfile1_lmcut_initial_local")
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
    .unwrap_or_else(|e| panic!("translate failed for drone pfile1: {e}"));

    run_preprocess_to_output(
        &[
            "preprocess".to_string(),
            output_sas.to_string_lossy().to_string(),
        ],
        &preprocessed,
    );

    let task = NumericRootTask::from_file(&preprocessed);
    let state_packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
    let mut state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);
    let initial_state = state_registry.get_initial_state();

    let mut propositional_values = Vec::new();
    initial_state.fill_state(&state_registry, &mut propositional_values);
    let mut numeric_values = Vec::new();
    state_registry
        .fill_numeric_vars(&initial_state, &mut numeric_values)
        .unwrap_or_else(|err| panic!("failed to prepare drone numeric values: {err:?}"));

    let mut landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());
    let relaxed_operator_count = landmarks.relaxed_operators().len();
    let proposition_count = landmarks.propositions().len();
    let numeric_condition_count = landmarks
        .propositions()
        .iter()
        .filter(|proposition| proposition.is_numeric_condition)
        .count();
    let infinite_operator_count = landmarks
        .relaxed_operators()
        .iter()
        .filter(|operator| operator.infinite)
        .count();
    let sose_operator_count = landmarks
        .relaxed_operators()
        .iter()
        .filter(|operator| operator.original_op_id_1.is_some())
        .count();
    let (dead_end, total_cost, landmarks_vec) = landmarks
        .compute_landmarks(
            &propositional_values,
            initial_state.buffer(&state_registry).len(),
            &numeric_values,
            false,
        )
        .unwrap_or_else(|error| {
            panic!(
                "Drone initial LM-cut failed with: {error} | counts: propositions={proposition_count} numeric_conditions={numeric_condition_count} relaxed_operators={relaxed_operator_count} infinite={infinite_operator_count} sose={sose_operator_count}"
            )
        });

    assert!(
        !dead_end,
        "Drone initial LM-cut should not be a dead end; landmarks={landmarks_vec:?}"
    );
    assert!(
        (total_cost - 3.0).abs() <= 1e-6,
        "Drone initial LM-cut should equal 3 after the zero-cost-cut fix, got {total_cost}; landmarks={landmarks_vec:?}"
    );
}

#[test]
#[ignore = "missing `drone.output` file"]
fn drone_output_lmcutnumeric_initial_state_matches_fd_regression() {
    let preprocessed = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("test_outputs")
        .join("drone.output");

    let task = NumericRootTask::from_file(&preprocessed);
    let state_packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
    let mut state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);
    let initial_state = state_registry.get_initial_state();

    let mut propositional_values = Vec::new();
    let mut numeric_values = Vec::new();
    state_registry
        .fill_state_and_numeric_vars(
            &initial_state,
            &mut propositional_values,
            &mut numeric_values,
        )
        .expect("drone.output initial state should unpack successfully");

    let mut landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());
    let (dead_end, total_cost, _cuts) = landmarks
        .compute_landmarks(
            &propositional_values,
            initial_state.buffer(&state_registry).len(),
            &numeric_values,
            false,
        )
        .expect("drone.output initial LM-cut should succeed");

    assert!(
        !dead_end,
        "drone.output initial state should not be a dead end"
    );
    assert!(
        (total_cost - 3.0).abs() <= 1e-6,
        "drone.output initial LM-cut should equal 3, got {total_cost}"
    );
}

#[test]
fn plant_watering_lmcutnumeric_full_search_solves_without_dead_ends() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/numeric-pddl-files/plant-watering");
    let domain = root.join("domain.pddl");
    let problem = root.join("prob_4_1_1.pddl");
    let expected_optimal_cost = 13.0;

    let temp_dir = unique_temp_dir("plant_watering_lmcut_full_search")
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
    .unwrap_or_else(|e| panic!("translate failed for Plant Watering: {e}"));

    run_preprocess_to_output(
        &[
            "preprocess".to_string(),
            output_sas.to_string_lossy().to_string(),
        ],
        &preprocessed,
    );

    let result = {
        let task = NumericRootTask::from_file(&preprocessed);
        let state_packer = IntDoublePacker::from_task(&task);
        let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
        let state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);
        let task_ref: &dyn AbstractNumericTask = &task;
        let heuristic =
            LandmarkCutNumericHeuristic::from_config(task_ref, LmCutNumericConfig::default())
                .expect("default lmcutnumeric config should be supported");
        let mut search = AStarSearch::new(
            task_ref,
            state_registry,
            Some(Box::new(heuristic)),
            None,
            None,
        );
        search.search()
    };

    let _ = std::fs::remove_dir_all(&temp_dir);

    let plan = match (&result.status, &result.plan) {
        (SearchStatus::Solved(_), Some(plan)) => plan,
        _ => panic!(
            "Plant Watering full lmcutnumeric search should solve the task, got status {:?}",
            result.status
        ),
    };

    let solution_cost = result
        .solution_cost
        .unwrap_or_else(|| plan.iter().map(|op| op.cost() as f64).sum());

    assert_eq!(
        result.dead_ends, 0,
        "Plant Watering lmcutnumeric full search should not mark any state as dead end"
    );
    assert!(
        !plan.is_empty(),
        "Plant Watering lmcutnumeric full search should return a non-empty plan"
    );
    assert!(
        (solution_cost - expected_optimal_cost).abs() <= 1e-6,
        "Plant Watering lmcutnumeric should keep optimal cost {expected_optimal_cost}, got {solution_cost}"
    );
}

#[test]
#[ignore = "parity probe for remaining zero-cost plateau behavior on blind-only reachable states"]
fn plant_watering_lmcutnumeric_remains_finite_along_blind_solution() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/numeric-pddl-files/plant-watering");
    let domain = root.join("domain.pddl");
    let problem = root.join("prob_4_1_1.pddl");

    let temp_dir = unique_temp_dir("plant_watering_lmcut_blind_prefix")
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
    .unwrap_or_else(|e| panic!("translate failed for Plant Watering: {e}"));

    run_preprocess_to_output(
        &[
            "preprocess".to_string(),
            output_sas.to_string_lossy().to_string(),
        ],
        &preprocessed,
    );

    let blind_plan = {
        let task = NumericRootTask::from_file(&preprocessed);
        let state_packer = IntDoublePacker::from_task(&task);
        let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
        let state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);
        let task_ref: &dyn AbstractNumericTask = &task;
        let mut search = AStarSearch::new(task_ref, state_registry, None, None, None);
        let result = search.search();
        match result {
            planforge_search::numeric::search_engine::SearchResult {
                status: SearchStatus::Solved(_),
                plan: Some(plan),
                ..
            } => plan,
            other => panic!(
                "blind Plant Watering search should solve the task before LM-cut replay, got {:?}",
                other.status
            ),
        }
    };

    let task = NumericRootTask::from_file(&preprocessed);
    let state_packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
    let mut state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);
    let mut state = state_registry.get_initial_state();
    let mut landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());
    let mut propositional_values = Vec::new();
    let mut numeric_values = Vec::new();

    for (step, operator) in std::iter::once(None)
        .chain(blind_plan.iter().map(Some))
        .enumerate()
    {
        state_registry
            .fill_state_and_numeric_vars(&state, &mut propositional_values, &mut numeric_values)
            .unwrap_or_else(|e| {
                panic!("failed to unpack Plant Watering state at step {step}: {e:?}")
            });

        let (dead_end, total_cost, _cuts) = landmarks
            .compute_landmarks(
                &propositional_values,
                state.buffer(&state_registry).len(),
                &numeric_values,
                false,
            )
            .unwrap_or_else(|e| panic!("LM-cut evaluation failed at step {step}: {e}"));

        assert!(
            !dead_end,
            "Plant Watering blind-solution state at step {step} should be reachable for LM-cut; last operator: {:?}",
            operator.map(|op| op.name())
        );
        assert!(
            total_cost.is_finite(),
            "Plant Watering blind-solution state at step {step} should have finite LM-cut value; last operator: {:?}",
            operator.map(|op| op.name())
        );

        if let Some(operator) = operator {
            state = state_registry
                .get_successor_state(&state, operator)
                .unwrap_or_else(|e| {
                    panic!("failed to apply blind-plan operator at step {step}: {e:?}")
                });
        }
    }

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
#[ignore = "debug hydropower initial lmcut successor values"]
fn hydropower_output_lmcutnumeric_initial_successor_trace() {
    let task_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("test_outputs/hydropower.output");
    let task = NumericRootTask::from_file(&task_path);
    let state_packer = IntDoublePacker::from_task(&task);
    let axiom_evaluator = AxiomEvaluator::new(&task, &state_packer);
    let mut state_registry = StateRegistry::new(&task, &state_packer, &axiom_evaluator);
    let initial_state = state_registry.get_initial_state();
    let propositional_state = initial_state.get_state(&state_registry);
    let successor_generator = GroundedSuccessorGenerator::construct_node_from_task(&task);
    let mut applicable_operators = Vec::new();
    successor_generator.get_applicable_operators(&propositional_state, &mut applicable_operators);

    let task_ref: &dyn AbstractNumericTask = &task;
    let heuristic =
        LandmarkCutNumericHeuristic::from_config(task_ref, LmCutNumericConfig::default())
            .expect("default lmcutnumeric config should be supported");

    let mut initial_eval =
        EvaluationState::new_with_registry(&initial_state, 0.0, false, task_ref, &state_registry);
    initial_eval.set_is_goal(false);
    heuristic
        .evaluate_state(&mut initial_eval)
        .expect("initial LM-cut evaluation should succeed");
    let initial_result = initial_eval.into_result();
    println!(
        "TRACE initial-state h={} applicable_ops={}",
        initial_result.get_heuristic_value(&heuristic.name()),
        applicable_operators.len()
    );

    for (operator, operator_id) in applicable_operators {
        let succ_state = state_registry
            .get_successor_state(&initial_state, operator)
            .unwrap_or_else(|e| {
                panic!("successor generation failed for {}: {e:?}", operator.name())
            });
        let g_value = state_registry
            .metric_delta_applying_operator(&initial_state, operator)
            .unwrap_or_else(|_| task.get_operators()[operator_id].cost() as f64);
        let mut successor_eval = EvaluationState::new_with_registry(
            &succ_state,
            g_value,
            false,
            task_ref,
            &state_registry,
        );
        successor_eval.set_is_goal(false);
        heuristic
            .evaluate_state(&mut successor_eval)
            .unwrap_or_else(|e| panic!("LM-cut evaluation failed for {}: {e}", operator.name()));
        let result = successor_eval.into_result();
        println!(
            "TRACE initial-successor op={} g={} h={} f={} dead_end={} state_id={}",
            operator.name(),
            g_value,
            result.get_heuristic_value(&heuristic.name()),
            g_value + result.get_heuristic_value(&heuristic.name()),
            result.is_dead_end,
            succ_state.get_id()
        );
    }
}
