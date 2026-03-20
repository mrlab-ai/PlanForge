use super::*;

use std::path::{Path, PathBuf};
use std::ffi::OsStr;

use planners::preprocess_port::planner::run_preprocess_to_output;

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
        _ => panic!("expected exactly 1 matching file in {dir:?}, got {}", matches.len()),
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
    let json: serde_json::Value =
        serde_json::from_str(&content).unwrap_or_else(|e| panic!("invalid json {stats_file:?}: {e}"));

    let solution_found = json
        .get("stats")
        .and_then(|s| s.get("solution_found"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(solution_found, "expected solution_found=true in {stats_file:?}");

    // If stats.plan_cost is missing (older captures), try reading it from the referenced log.
    // Otherwise fall back to plan_length (unit-cost tasks).
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
        .unwrap_or_else(|| panic!("missing stats.plan_cost/log_file and stats.plan_length in {stats_file:?}"))
}

fn expected_plan_length(stats_file: &Path) -> u64 {
    let content = std::fs::read_to_string(stats_file)
        .unwrap_or_else(|e| panic!("read_to_string failed for {stats_file:?}: {e}"));
    let json: serde_json::Value =
        serde_json::from_str(&content).unwrap_or_else(|e| panic!("invalid json {stats_file:?}: {e}"));

    let solution_found = json
        .get("stats")
        .and_then(|s| s.get("solution_found"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    assert!(solution_found, "expected solution_found=true in {stats_file:?}");

    json.get("stats")
        .and_then(|s| s.get("plan_length"))
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| panic!("missing stats.plan_length in {stats_file:?}"))
}

#[test]
fn fd_blind_plan_cost_matches_misc_benchmarks() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("misc/numeric-pddl-files");

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
            &vec![
                "preprocess".to_string(),
                output_sas
                    .to_string_lossy()
                    .to_string(),
            ],
            &preprocessed,
        );

        let (actual_cost, actual_len): (Option<f64>, Option<u64>) = {
            let task = setup_numeric_task(&preprocessed);
            let state_packer = setup_state_packer(&task);
            let axiom_evaluator = setup_axiom_evaluator(&task, &state_packer);
            let state_registry = setup_state_registry(&task, &state_packer, &axiom_evaluator);

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
