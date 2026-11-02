//! Plan cost and search effort of every heuristic family on the numeric
//! benchmark corpus, pinned in `snapshots/search.txt`.
//!
//! This is the semantic gate: a refactoring must not move a single line, and
//! a behaviour change shows exactly which runs it moved in the snapshot's
//! diff. It needs an optimized build, so it is ignored by default:
//!
//! ```sh
//! cargo test -p tests --release -- --ignored search_snapshot
//! PLANFORGE_BLESS=1 cargo test -p tests --release -- --ignored search_snapshot
//! ```
//!
//! The second line rewrites the snapshot after an intended change. Every
//! configuration is bounded by states, iterations or collection sizes, never
//! by time, so the snapshot is a function of the code alone.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use planforge_sas::numeric_task::Task;
use planforge_search::search::{SearchLimits, SearchOutcome, parse_search_spec};
use planforge_search::task_transformations::{parse_transformation_spec, transform_task};

use crate::corpus::{self, problem_file_names, subdirectory_names, translate_in_memory};

/// The configurations, each with the transformation of the task it
/// searches, if any.
const CONFIGS: &[(Option<&str>, &str)] = &[
    (None, "astar(blind())"),
    (None, "astar(lmcutnumeric())"),
    (None, "gbfs(ff())"),
    (None, "astar(domain())"),
    (None, "astar(cartesian())"),
    (
        None,
        "astar(max([domain(max_states=500),cartesian(max_states=500)]))",
    ),
    (
        Some("restrict"),
        "astar(canonical(pdb(patterns=systematic(max_pdb_states=1000))))",
    ),
    (
        Some("restrict"),
        "astar(canonical(pdb(patterns=hillclimbing(max_pdb_states=1000,num_samples=100,collection_max_size=20000))))",
    ),
    (
        Some("restrict"),
        "astar(scp([domain_collection(total_max_time=infinity,max_collection_size=20000)]))",
    ),
    (
        Some("icaps26_restrict"),
        "astar(canonical(icaps26_cartesian(pick=min_unwanted,max_states=2000)))",
    ),
];

fn snapshot_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("snapshots/search.txt")
}

/// Every problem of every benchmark folder, as `(folder, problem file)`.
fn instances() -> Vec<(String, String)> {
    let root = corpus::assets().join("numeric-pddl-files");
    subdirectory_names(&root)
        .into_iter()
        .flat_map(|folder| {
            problem_file_names(&root.join(&folder))
                .into_iter()
                .map(move |problem| (folder.clone(), problem))
        })
        .collect()
}

/// One snapshot line: the run's outcome, or the error that ended it.
fn run(task: &Task, transformation: Option<&str>, spec: &str) -> String {
    let search = parse_search_spec(spec).unwrap_or_else(|error| panic!("{spec}: {error}"));
    let transformations: Vec<_> = transformation
        .map(|raw| parse_transformation_spec(raw).unwrap_or_else(|error| panic!("{raw}: {error}")))
        .into_iter()
        .collect();
    let task = match transform_task(task.clone(), &transformations, &search.requirements()) {
        Ok(task) => task,
        Err(error) => return format!("restriction error: {}", first_line(&error.to_string())),
    };
    match search.solve(Arc::new(task), SearchLimits::default()) {
        Ok(result) => {
            let nodes = &result.statistics.nodes;
            let outcome = match result.outcome {
                SearchOutcome::Solved { cost, .. } => format!("cost={cost}"),
                SearchOutcome::Unsolvable => "unsolvable".to_owned(),
                other => panic!("{spec}: unbounded run ended with {other:?}"),
            };
            format!(
                "{outcome} expanded={} generated={}",
                nodes.expanded, nodes.generated
            )
        }
        Err(error) => format!("error: {}", first_line(&error.to_string())),
    }
}

fn first_line(message: &str) -> &str {
    message.split_once('\n').map_or(message, |(first, _)| first)
}

/// The snapshot of `instances`, computed by a few worker threads.
fn snapshot(instances: &[(String, String)]) -> String {
    let root = corpus::assets().join("numeric-pddl-files");
    let next = AtomicUsize::new(0);
    let workers = std::thread::available_parallelism()
        .map_or(1, |cores| cores.get())
        .min(4);
    let mut lines: Vec<(usize, String)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut lines = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some((folder, problem)) = instances.get(index) else {
                            return lines;
                        };
                        let dir = root.join(folder);
                        let task =
                            translate_in_memory(&dir.join("domain.pddl"), &dir.join(problem));
                        for (transformation, spec) in CONFIGS {
                            lines.push((
                                index,
                                format!(
                                    "{folder}/{problem} | {spec} | {}",
                                    run(&task, *transformation, spec)
                                ),
                            ));
                        }
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("snapshot worker panicked"))
            .collect()
    });
    // Instances in corpus order, configurations in table order.
    lines.sort_by_key(|(index, _)| *index);
    lines.into_iter().map(|(_, line)| line + "\n").collect()
}

#[test]
#[ignore = "semantic gate; run with --release -- --ignored search_snapshot"]
fn search_snapshot() {
    planforge_search::heuristic_factory::HEURISTICS
        .ensure_registered(planforge_heuristics::heuristics);
    planforge_search::task_transformations::TRANSFORMATIONS
        .ensure_registered(planforge_heuristics::transformations);
    let actual = snapshot(&instances());
    let path = snapshot_path();
    if std::env::var_os("PLANFORGE_BLESS").is_some() {
        std::fs::create_dir_all(path.parent().expect("snapshot directory"))
            .expect("create the snapshot directory");
        std::fs::write(&path, &actual).expect("write the snapshot");
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!("no snapshot at {path:?} ({error}); create it with PLANFORGE_BLESS=1")
    });
    let changed: Vec<String> = expected
        .lines()
        .zip(actual.lines())
        .filter(|(before, after)| before != after)
        .map(|(before, after)| format!("- {before}\n+ {after}"))
        .collect();
    assert!(
        changed.is_empty() && expected.lines().count() == actual.lines().count(),
        "{} of {} runs differ from {path:?} (expected {} lines, got {}):\n{}",
        changed.len(),
        expected.lines().count(),
        expected.lines().count(),
        actual.lines().count(),
        changed.join("\n")
    );
}
