//! Two runs of the planner on the same input must report the same thing.
//!
//! `tests/src/determinism_tests.rs` replans inside one process, which exercises
//! every `HashMap` the pipeline creates because each one gets its own hash key.
//! What that cannot see is a dependence on something seeded once per *process* --
//! a hash key cached in a static, an address the loader chose, a clock read at
//! startup. So this spawns the binary instead, and compares what it says.
//!
//! The comparison is the whole log minus its timings, rather than a chosen few
//! numbers: a line that differs between two runs of the same input is a bug
//! whether or not anyone thought to pin it. Plan steps, plan cost and length, and
//! every state counter are all in there.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Fixtures planned in a fresh process each time. Small enough to plan three
/// times in a debug build, and between them they cover multi-valued variables
/// with mutex groups (depots) and numeric conditions (plant-watering).
const FIXTURES: &[(&str, &str)] = &[
    ("depots", "pfile1.pddl"),
    ("plant-watering", "prob_4_1_1.pddl"),
];

const RUNS: usize = 3;

fn fixture(domain_dir: &str, problem: &str) -> (PathBuf, PathBuf) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/assets/numeric-pddl-files")
        .join(domain_dir);
    let domain = dir.join("domain.pddl");
    let problem = dir.join(problem);
    assert!(domain.is_file(), "missing fixture {}", domain.display());
    assert!(problem.is_file(), "missing fixture {}", problem.display());
    (domain, problem)
}

/// The message part of a log line: everything after the level, which is what
/// drops the timestamp the line starts with. A line without a level -- a panic,
/// say -- is kept whole. Colour escapes go too, because they wrap both the
/// timestamp and the level and would otherwise be part of every message.
fn message_of(line: &str) -> String {
    const LEVELS: [&str; 5] = ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"];
    let plain = strip_ansi(line);
    for level in LEVELS {
        if let Some((_, message)) = plain.split_once(level) {
            return message.trim().to_owned();
        }
    }
    plain.trim().to_owned()
}

/// `line` without its SGR escapes (`ESC [ ... m`), which is all the subscriber
/// emits.
fn strip_ansi(line: &str) -> String {
    let mut plain = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(escape) = rest.find('\x1b') {
        plain.push_str(&rest[..escape]);
        match rest[escape..].find('m') {
            Some(end) => rest = &rest[escape + end + 1..],
            // An escape the subscriber never finished: keep it whole rather than
            // swallow the rest of the line, so the surprise shows up.
            None => {
                plain.push_str(&rest[escape..]);
                return plain;
            }
        }
    }
    plain.push_str(rest);
    plain
}

/// A reported message that says how long something took, and so cannot be
/// expected to repeat. Everything else is expected to, verbatim.
///
/// `t=` catches the progress lines, which pair an elapsed time with a
/// resident-set size; `time` the search and total times; ` in: ` the durations
/// the pipeline reports for a stage of its own.
fn reports_a_duration(message: &str) -> bool {
    message.contains("t=") || message.contains("time") || message.contains(" in: ")
}

/// Everything the planner reported that a rerun has to report again.
fn reported_lines(domain: &Path, problem: &Path) -> Vec<String> {
    let output = Command::new(env!("CARGO_BIN_EXE_planforge"))
        .arg(domain)
        .arg(problem)
        .output()
        .unwrap_or_else(|e| panic!("could not run the planner on {}: {e}", problem.display()));
    assert!(
        output.status.success(),
        "the planner failed on {}: {}",
        problem.display(),
        String::from_utf8_lossy(&output.stderr)
    );

    // The two streams are concatenated rather than interleaved: which of them a
    // line lands in is up to the subscriber, but the order within each is not.
    let mut text = String::from_utf8(output.stdout).expect("planner stdout is UTF-8");
    text.push_str(&String::from_utf8(output.stderr).expect("planner stderr is UTF-8"));

    text.lines()
        .map(message_of)
        .filter(|message| !message.is_empty() && !reports_a_duration(message))
        .collect()
}

#[test]
fn planning_in_a_fresh_process_reports_the_same_thing_every_time() {
    for &(domain_dir, problem_name) in FIXTURES {
        let (domain, problem) = fixture(domain_dir, problem_name);

        let first = reported_lines(&domain, &problem);
        assert!(
            first.iter().any(|line| line.starts_with("Plan cost:")),
            "{domain_dir}: the planner reported no plan cost, so there is nothing to compare"
        );
        for run_no in 2..=RUNS {
            let again = reported_lines(&domain, &problem);
            assert_eq!(
                first, again,
                "planning {domain_dir} is not reproducible: run 1 and run {run_no} differ"
            );
        }
    }
}
