use planforge_search::search::SearchOutcome;

use super::*;
use crate::limits::{
    format_time_limit, normalize_wrapped_exit, parse_memory_limit, parse_time_limit,
};
use crate::output::{EXIT_OUT_OF_MEMORY, EXIT_SUCCESS};

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

#[test]
fn parses_memory_limit_suffixes() {
    assert_eq!(parse_memory_limit("500M").unwrap(), 500 * 1024 * 1024);
    assert_eq!(parse_memory_limit("8g").unwrap(), 8 * 1024 * 1024 * 1024);
    assert_eq!(parse_memory_limit("1024").unwrap(), 1024);
}

#[test]
fn parses_time_limit_suffixes() {
    assert_eq!(parse_time_limit("60s").unwrap(), Duration::from_secs(60));
    assert_eq!(parse_time_limit("5m").unwrap(), Duration::from_secs(300));
    assert_eq!(
        parse_time_limit("250ms").unwrap(),
        Duration::from_millis(250)
    );
    assert_eq!(format_time_limit(Duration::from_secs(60)), "60s");
    assert_eq!(format_time_limit(Duration::from_millis(250)), "250ms");
}

#[test]
fn maps_search_outcomes_to_exit_codes() {
    let solved = SearchOutcome::Solved {
        plan: Vec::new(),
        cost: 0.0,
    };
    assert_eq!(exit_code_for_search_outcome(&solved), EXIT_SUCCESS);
    assert_eq!(
        exit_code_for_search_outcome(&SearchOutcome::Unsolvable),
        EXIT_SUCCESS
    );
    assert_eq!(
        exit_code_for_search_outcome(&SearchOutcome::Timeout),
        EXIT_TIMEOUT
    );
    assert_eq!(
        exit_code_for_search_outcome(&SearchOutcome::OutOfMemory),
        EXIT_OUT_OF_MEMORY
    );
}

#[cfg(unix)]
#[test]
fn normalizes_wrapped_timeout_signal() {
    let status = std::process::ExitStatus::from_raw(libc::SIGXCPU);
    assert_eq!(
        normalize_wrapped_exit(status, false, Some(Duration::from_secs(1))),
        EXIT_TIMEOUT
    );
}

#[cfg(unix)]
#[test]
fn only_the_memory_watchdog_reports_out_of_memory() {
    let killed = std::process::ExitStatus::from_raw(libc::SIGKILL);
    assert_eq!(
        normalize_wrapped_exit(killed, true, None),
        EXIT_OUT_OF_MEMORY
    );
    let crashed = std::process::ExitStatus::from_raw((128 + libc::SIGSEGV) << 8);
    assert_eq!(
        normalize_wrapped_exit(crashed, false, None),
        128 + libc::SIGSEGV
    );
}

#[cfg(unix)]
#[test]
fn normalizes_wrapped_timeout_exit_code() {
    let status = std::process::ExitStatus::from_raw((128 + libc::SIGXCPU) << 8);
    assert_eq!(
        normalize_wrapped_exit(status, false, Some(Duration::from_secs(1))),
        EXIT_TIMEOUT
    );
}

#[test]
fn cli_rejects_nested_config_errors_without_loading_pddl() {
    crate::register_planforge_features();
    for (spec, detail) in [
        (
            "astar(scp(domian(),bogus=1))",
            "scp.sources: unknown AbstractionSource `domian`",
        ),
        ("astar(scp(domain(),bogus=1))", "scp.bogus"),
        ("astar(domain(max_states=bogus))", "max_states"),
        (
            "astar(numeric_potential(bogus=1))",
            "numeric_potential.bogus",
        ),
    ] {
        let error = PlannersCli::try_parse_from([
            "planforge",
            "--search",
            spec,
            "missing-domain.pddl",
            "missing-problem.pddl",
        ])
        .expect_err("configuration must fail during CLI parsing");
        assert!(error.to_string().contains(detail), "{error}");
    }
}
