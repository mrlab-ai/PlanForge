use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use planforge_sas::numeric_task::AbstractNumericTask;
use planforge_sas::numeric_task::NumericRootTask;
use planforge_sas::state_registry::StateRegistry;
use planforge_search::evaluation::evaluator::EvaluationState;
use planforge_search::evaluation::evaluator::Evaluator;
use planforge_search::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LandmarkCutNumericHeuristic;
use planforge_search::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig;
use planforge_search::evaluation::numeric_landmarks::numeric_lm_cut_landmarks::LandmarkCutLandmarks;
use planforge_search::search::SearchStatus;
use planforge_search::search::{AStarSearch, SearchEngine};
use planforge_search::successor_generator::GroundedSuccessorGenerator;
use planforge_translate::preprocess::run_preprocess_to_output;
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

/// Reference solution of one benchmark folder under `assets/numeric-pddl-files`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Solution {
    cost: f64,
    length: u64,
}

impl Solution {
    /// Reads the blind-A* reference capture stored next to the problem file.
    fn from_recorded_stats(stats_file: &Path) -> Self {
        let content = std::fs::read_to_string(stats_file)
            .unwrap_or_else(|e| panic!("read_to_string failed for {stats_file:?}: {e}"));
        let json: serde_json::Value = serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("invalid json {stats_file:?}: {e}"));
        let stats = json
            .get("stats")
            .unwrap_or_else(|| panic!("missing `stats` object in {stats_file:?}"));

        assert_eq!(
            stats.get("solution_found").and_then(|v| v.as_bool()),
            Some(true),
            "expected stats.solution_found=true in {stats_file:?}"
        );

        Solution {
            cost: stats
                .get("plan_cost")
                .and_then(|v| v.as_f64())
                .unwrap_or_else(|| panic!("missing numeric stats.plan_cost in {stats_file:?}")),
            length: stats
                .get("plan_length")
                .and_then(|v| v.as_u64())
                .unwrap_or_else(|| panic!("missing integer stats.plan_length in {stats_file:?}")),
        }
    }

    fn matches(&self, other: &Solution) -> bool {
        self.length == other.length && (self.cost - other.cost).abs() <= 1e-3
    }
}

/// Known optima of every benchmark folder under `assets/numeric-pddl-files`,
/// measured with `planforge --search 'astar(blind())'`. Blind A* is optimal, so
/// `cost` is the true optimum of the task; `length` pins the plan realising it.
///
/// The hand-written `sailing-simple` corpus is covered by `sailing_simple_tests`
/// and deliberately absent here.
const BENCHMARK_OPTIMA: &[(&str, f64, u64)] = &[
    ("counters-sym", 5.0, 5),
    ("delivery", 22.0, 10),
    ("depots", 22.0, 10),
    ("depots-sym", 22.0, 10),
    ("drone", 10.0, 10),
    ("expedition", 26.0, 26),
    ("farmland", 78.0, 78),
    ("farmland2", 78.0, 78),
    ("fn-counters-small_instances", 1.0, 1),
    ("forestfire", 24.0, 24),
    ("hydropower", 35.0, 35),
    ("minecraft-pogo-advanced", 6.0, 6),
    ("minecraft-sword-advanced", 2.0, 2),
    ("mprime", 4.0, 4),
    ("onlycraft-opt", 5.0, 5),
    ("pathwaysmetric", 12.0, 12),
    ("plant-watering", 13.0, 13),
    ("rover-unit", 10.0, 10),
    ("sailing", 23.0, 23),
    ("satellite", 108.586, 11),
    ("zenotravel", 9.0, 9),
];

/// Benchmarks whose translation or search needs several seconds in an
/// unoptimized test build. Their recorded optima are still validated against
/// `BENCHMARK_OPTIMA`; only the search itself is skipped.
const TOO_SLOW_FOR_UNOPTIMIZED_SEARCH: &[&str] =
    &["minecraft-pogo-advanced", "minecraft-sword-advanced"];

fn benchmarks_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/numeric-pddl-files")
}

/// Discovers the benchmark folders and asserts that they are exactly the ones
/// listed in [`BENCHMARK_OPTIMA`], so a new fixture cannot slip in untested.
fn discover_benchmark_dirs() -> Vec<(&'static str, Solution, PathBuf)> {
    let root = benchmarks_root();
    let mut discovered: Vec<String> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("read_dir failed for {root:?}: {e}"))
        .map(|entry| entry.unwrap_or_else(|e| panic!("bad dir entry under {root:?}: {e}")))
        .filter(|entry| entry.path().is_dir())
        .map(|entry| {
            entry
                .file_name()
                .into_string()
                .unwrap_or_else(|name| panic!("non-utf8 benchmark folder {name:?}"))
        })
        .filter(|name| name != "sailing-simple")
        .collect();
    discovered.sort();

    let mut expected: Vec<&str> = BENCHMARK_OPTIMA.iter().map(|(name, _, _)| *name).collect();
    expected.sort_unstable();
    assert_eq!(
        discovered, expected,
        "benchmark folders under {root:?} do not match BENCHMARK_OPTIMA"
    );
    for slow in TOO_SLOW_FOR_UNOPTIMIZED_SEARCH {
        assert!(
            expected.contains(slow),
            "TOO_SLOW_FOR_UNOPTIMIZED_SEARCH lists unknown benchmark {slow:?}"
        );
    }

    BENCHMARK_OPTIMA
        .iter()
        .map(|&(name, cost, length)| {
            let bench_dir = root.join(name);
            assert!(
                bench_dir.join("domain.pddl").is_file(),
                "missing domain.pddl in {bench_dir:?}"
            );
            (name, Solution { cost, length }, bench_dir)
        })
        .collect()
}

fn problem_file(bench_dir: &Path) -> PathBuf {
    find_single_file(bench_dir, |path| {
        path.extension()
            .is_some_and(|ext| ext == OsStr::new("pddl"))
            && path
                .file_name()
                .is_some_and(|name| name != OsStr::new("domain.pddl"))
    })
}

fn recorded_stats_file(bench_dir: &Path) -> PathBuf {
    find_single_file(bench_dir, |path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".fd_blind.json"))
    })
}

/// Translates and preprocesses a PDDL pair inside `temp_dir` and returns the
/// path of the preprocessed task.
fn preprocess_task(domain: &Path, problem: &Path, temp_dir: &Path) -> PathBuf {
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
    .unwrap_or_else(|e| panic!("translate failed for {problem:?}: {e}"));

    run_preprocess_to_output(
        &[
            "preprocess".to_string(),
            output_sas.to_string_lossy().to_string(),
        ],
        &preprocessed,
    );

    preprocessed
}

/// Runs blind A* on a preprocessed task; `None` means the search returned no plan.
fn blind_astar_solution(preprocessed: &Path) -> Option<Solution> {
    let task = NumericRootTask::from_file(preprocessed);
    let state_registry = StateRegistry::for_task(std::sync::Arc::new(&task));
    let mut search = AStarSearch::new(std::sync::Arc::new(&task), state_registry, None, None, None);
    let result = search.search().expect("blind A* search failed");

    match (&result.status, &result.plan) {
        (SearchStatus::Solved(_), Some(plan)) => Some(Solution {
            cost: result
                .solution_cost
                .unwrap_or_else(|| plan.iter().map(|op| op.cost() as f64).sum()),
            length: plan.len() as u64,
        }),
        _ => None,
    }
}

/// Guards the recorded `.fd_blind.json` captures against silent drift: they must
/// agree with the optima pinned in this file.
#[test]
fn recorded_blind_stats_match_known_optima() {
    let mut mismatches: Vec<String> = Vec::new();

    for (bench_name, optimum, bench_dir) in discover_benchmark_dirs() {
        let recorded = Solution::from_recorded_stats(&recorded_stats_file(&bench_dir));
        if !recorded.matches(&optimum) {
            mismatches.push(format!(
                "{bench_name}: recorded {recorded:?}, expected {optimum:?}"
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "recorded blind-A* stats disagree with BENCHMARK_OPTIMA:\n{}",
        mismatches.join("\n")
    );
}

#[test]
fn fd_blind_plan_cost_matches_misc_benchmarks() {
    let mut mismatches: Vec<String> = Vec::new();

    for (bench_name, optimum, bench_dir) in discover_benchmark_dirs() {
        if TOO_SLOW_FOR_UNOPTIMIZED_SEARCH.contains(&bench_name) {
            eprintln!("Skipping {bench_name} (too slow in debug test mode)");
            continue;
        }

        let temp_dir = unique_temp_dir(&format!("fd_blind_{bench_name}"))
            .unwrap_or_else(|e| panic!("failed to create temp dir: {e}"));
        let preprocessed = preprocess_task(
            &bench_dir.join("domain.pddl"),
            &problem_file(&bench_dir),
            &temp_dir,
        );
        let actual = blind_astar_solution(&preprocessed);
        let _ = std::fs::remove_dir_all(&temp_dir);

        match actual {
            Some(actual) if actual.matches(&optimum) => {
                eprintln!("{bench_name}: {actual:?} (optimal)");
            }
            Some(actual) => mismatches.push(format!(
                "{bench_name}: expected {optimum:?}, got {actual:?}"
            )),
            None => mismatches.push(format!(
                "{bench_name}: expected {optimum:?}, but search did not return a solved plan"
            )),
        }
    }

    assert!(
        mismatches.is_empty(),
        "blind A* did not reproduce the known optima:\n{}",
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
    let preprocessed = preprocess_task(&domain, &problem, &temp_dir);

    let (dead_end, total_cost) = {
        let task = NumericRootTask::from_file(&preprocessed);
        let mut state_registry = StateRegistry::for_task(std::sync::Arc::new(&task));
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
    let preprocessed = preprocess_task(domain, problem, &temp_dir);

    let task = NumericRootTask::from_file(&preprocessed);
    let mut state_registry = StateRegistry::for_task(std::sync::Arc::new(&task));
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
    let mut state_registry = StateRegistry::for_task(std::sync::Arc::new(&task));
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
    let preprocessed = preprocess_task(&domain, &problem, &temp_dir);

    let result = {
        let task = NumericRootTask::from_file(&preprocessed);
        let state_registry = StateRegistry::for_task(std::sync::Arc::new(&task));
        let task_ref: &dyn AbstractNumericTask = &task;
        let heuristic =
            LandmarkCutNumericHeuristic::from_config(task_ref, LmCutNumericConfig::default())
                .expect("default lmcutnumeric config should be supported");
        let mut search = AStarSearch::new(
            std::sync::Arc::new(&task),
            state_registry,
            Some(Box::new(heuristic)),
            None,
            None,
        );
        search.search().expect("LM-cut A* search failed")
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
    let preprocessed = preprocess_task(&domain, &problem, &temp_dir);

    let blind_plan = {
        let task = NumericRootTask::from_file(&preprocessed);
        let state_registry = StateRegistry::for_task(std::sync::Arc::new(&task));
        let mut search =
            AStarSearch::new(std::sync::Arc::new(&task), state_registry, None, None, None);
        let result = search.search().expect("blind A* search failed");
        match result {
            planforge_search::search::SearchResult {
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
    let mut state_registry = StateRegistry::for_task(std::sync::Arc::new(&task));
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
    let mut state_registry = StateRegistry::for_task(std::sync::Arc::new(&task));
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

    for op_id in applicable_operators {
        let operator_id = op_id as usize;
        let operator = &task.get_operators()[operator_id];
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
