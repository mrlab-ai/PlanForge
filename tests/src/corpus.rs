//! The shared harness every integration test in this crate is built on.
//!
//! The suite is corpus-driven rather than bespoke: fixtures live under
//! `tests/assets`, every fixture folder is *discovered* from the filesystem, and
//! the discovered set is compared against a pinned table. A fixture therefore
//! cannot be added, renamed or deleted without a test failing, which is what
//! makes "the suite passes" mean something.
//!
//! Two translation paths are exposed on purpose, because they produce different
//! tasks and both are exercised elsewhere in the planner:
//!
//! * [`translate_to_disk`] asks for singleton fact groups, and goes through the
//!   SAS+ file: it is the write-then-read path other planners interoperate on.
//!   Every propositional variable ends up binary.
//! * [`translate_in_memory`] mirrors what the `planforge` binary does for a
//!   two-argument PDDL invocation, and builds genuine multi-valued variables
//!   without going through text at all.
//!
//! That the two ways of getting a task out of one translation agree is the
//! subject of `task_equivalence_tests`, not of the harness here.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use planforge_sas::numeric_task::NumericRootTask;
use planforge_sas::state_registry::StateRegistry;
use planforge_search::search::{AStarSearch, SearchEngine, SearchStatus};
use planforge_translator::{translate_to_sas_to_path_fast, translate_to_task};

/// Root of the checked-in fixture tree.
pub fn assets() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("assets")
}

/// A scratch directory that deletes itself, including when the test panics.
///
/// The previous `let _ = fs::remove_dir_all(dir)` at the end of each test leaked
/// the directory on every failure, which is exactly when it accumulates.
pub struct Scratch {
    path: PathBuf,
}

impl Scratch {
    pub fn new(prefix: &str) -> Self {
        let base = std::env::temp_dir().join("planforge-tests");
        std::fs::create_dir_all(&base)
            .unwrap_or_else(|e| panic!("cannot create scratch root {base:?}: {e}"));

        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is after the unix epoch")
            .as_nanos();
        let path = base.join(format!("{prefix}_{}_{nanos}", std::process::id()));
        std::fs::create_dir(&path)
            .unwrap_or_else(|e| panic!("cannot create scratch dir {path:?}: {e}"));
        Scratch { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // A leftover scratch directory is not worth failing a test over, but a
        // failure to remove one is worth knowing about.
        if let Err(e) = std::fs::remove_dir_all(&self.path) {
            eprintln!("could not remove scratch dir {:?}: {e}", self.path);
        }
    }
}

/// Translates through the on-disk `..._fast` path, which requests singleton
/// fact groups.
///
/// The returned [`Scratch`] owns the temporary files and must stay alive for as
/// long as the caller wants them.
pub fn translate_to_disk(domain: &Path, problem: &Path, scratch: &Scratch) -> NumericRootTask {
    let output = scratch.path().join("output.sas");

    translate_to_sas_to_path_fast(path_str(domain), path_str(problem), &output)
        .unwrap_or_else(|e| panic!("translate failed for {problem:?}: {e}"));

    NumericRootTask::from_file(&output)
}

/// Translates entirely in memory, the way the `planforge` binary does for a
/// two-argument PDDL invocation. Keeps multi-valued variables.
pub fn translate_in_memory(domain: &Path, problem: &Path) -> NumericRootTask {
    assert!(domain.is_file(), "missing fixture {domain:?}");
    assert!(problem.is_file(), "missing fixture {problem:?}");

    translate_to_task(path_str(domain), path_str(problem))
        .unwrap_or_else(|e| panic!("translate failed for {problem:?}: {e}"))
}

fn path_str(path: &Path) -> &str {
    path.to_str()
        .unwrap_or_else(|| panic!("non-utf8 fixture path: {path:?}"))
}

/// Cost and length of a plan. Costs are compared with a tolerance because task
/// metrics are real-valued; lengths must agree exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solution {
    pub cost: f64,
    pub length: u64,
}

impl Solution {
    pub fn matches(&self, other: &Solution) -> bool {
        self.length == other.length && (self.cost - other.cost).abs() <= 1e-3
    }
}

/// Everything a blind-A* run is determined by its task alone: the plan it
/// returns, and how much search it took to get there.
///
/// The counters are part of it because they are the sensitive half. Two runs can
/// agree on an optimal cost while disagreeing on which optimal plan they find and
/// on how many states they looked at, and it is the latter that moves when
/// anything in the pipeline depends on the iteration order of a hash map.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchRun {
    pub cost: f64,
    /// Operator names, in plan order.
    pub plan: Vec<String>,
    pub expanded: usize,
    pub generated: usize,
}

impl SearchRun {
    pub fn solution(&self) -> Solution {
        Solution {
            cost: self.cost,
            length: self.plan.len() as u64,
        }
    }
}

/// Runs blind A*, which is optimal, so the returned cost is the true optimum.
/// `None` means the search terminated without a plan.
pub fn blind_astar_run(task: &NumericRootTask) -> Option<SearchRun> {
    let registry = StateRegistry::for_task(Arc::new(task));
    let mut search = AStarSearch::new(Arc::new(task), registry, None, None, None);
    let result = search.search().expect("blind A* search failed");

    match (&result.status, &result.plan) {
        (SearchStatus::Solved(_), Some(plan)) => Some(SearchRun {
            cost: result
                .solution_cost
                .unwrap_or_else(|| plan.iter().map(|op| op.cost() as f64).sum()),
            plan: plan.iter().map(|op| op.name().to_owned()).collect(),
            expanded: result.nodes_expanded,
            generated: result.nodes_generated,
        }),
        _ => None,
    }
}

/// Cost and length of the plan blind A* finds, for the tests that pin an optimum
/// rather than a whole run.
pub fn blind_astar(task: &NumericRootTask) -> Option<Solution> {
    blind_astar_run(task).map(|run| run.solution())
}

/// Blind A* that insists on a plan, for fixtures whose solvability is pinned.
pub fn blind_astar_cost(task: &NumericRootTask, what: &str) -> f64 {
    blind_astar(task)
        .unwrap_or_else(|| panic!("blind A* returned no plan for {what}"))
        .cost
}

/// Names of the sub-directories of `dir`, sorted.
pub fn subdirectory_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir failed for {dir:?}: {e}"))
        .map(|entry| entry.unwrap_or_else(|e| panic!("bad dir entry under {dir:?}: {e}")))
        .filter(|entry| entry.path().is_dir())
        .map(|entry| {
            entry
                .file_name()
                .into_string()
                .unwrap_or_else(|name| panic!("non-utf8 folder name {name:?} under {dir:?}"))
        })
        .collect();
    names.sort();
    names
}

/// Asserts that a discovered set of fixture names is exactly the pinned one.
///
/// This is the mechanism that makes the tables in this crate load-bearing: an
/// unpinned fixture is a test failure, not a silently ignored file.
pub fn assert_fixture_set_is_pinned(what: &str, discovered: &[String], pinned: &[&str]) {
    let mut expected: Vec<&str> = pinned.to_vec();
    expected.sort_unstable();
    assert!(
        expected.windows(2).all(|pair| pair[0] != pair[1]),
        "{what}: the pinned table lists a name twice"
    );

    // Sorted here rather than assumed of the caller: the comparison is about the
    // two sets, and an unsorted argument would otherwise fail with a diff that
    // looks like a missing fixture.
    let mut discovered: Vec<&str> = discovered.iter().map(String::as_str).collect();
    discovered.sort_unstable();
    assert_eq!(
        discovered, expected,
        "{what}: the fixtures on disk do not match the pinned table"
    );
}

/// The single `.pddl` file in `dir` that is not the domain.
pub fn problem_file(dir: &Path) -> PathBuf {
    single_file(dir, is_problem_file)
}

/// Every `.pddl` file in `dir` that is not the domain, sorted by file name.
pub fn problem_file_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir failed for {dir:?}: {e}"))
        .map(|entry| entry.unwrap_or_else(|e| panic!("bad dir entry under {dir:?}: {e}")))
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && is_problem_file(path))
        .map(|path| {
            path.file_name()
                .expect("a file has a name")
                .to_str()
                .unwrap_or_else(|| panic!("non-utf8 problem file {path:?}"))
                .to_owned()
        })
        .collect();
    names.sort();
    names
}

fn is_problem_file(path: &Path) -> bool {
    path.extension() == Some(OsStr::new("pddl"))
        && path.file_name() != Some(OsStr::new("domain.pddl"))
}

/// The single file in `dir` matching `predicate`; more or fewer than one is a
/// broken fixture and fails immediately.
pub fn single_file(dir: &Path, predicate: impl Fn(&Path) -> bool) -> PathBuf {
    let mut matches: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read_dir failed for {dir:?}: {e}"))
        .map(|entry| entry.unwrap_or_else(|e| panic!("bad dir entry under {dir:?}: {e}")))
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && predicate(path))
        .collect();
    matches.sort();

    match matches.as_slice() {
        [only] => only.clone(),
        [] => panic!("no matching file in {dir:?}"),
        many => panic!("expected exactly one matching file in {dir:?}, got {many:?}"),
    }
}
