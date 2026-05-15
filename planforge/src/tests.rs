use planforge_search::numeric::search_engine::SearchStatus;

use super::*;

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
}

#[test]
fn maps_search_statuses_to_exit_codes() {
    assert_eq!(
        exit_code_for_search_status(&SearchStatus::Solved(0)),
        EXIT_SUCCESS
    );
    assert_eq!(
        exit_code_for_search_status(&SearchStatus::Failed),
        EXIT_SUCCESS
    );
    assert_eq!(
        exit_code_for_search_status(&SearchStatus::Timeout),
        EXIT_TIMEOUT
    );
    assert_eq!(
        exit_code_for_search_status(&SearchStatus::MemoryLimitReached),
        EXIT_OUT_OF_MEMORY
    );
}

#[cfg(unix)]
#[test]
fn normalizes_wrapped_timeout_signal() {
    let status = std::process::ExitStatus::from_raw(libc::SIGXCPU);
    assert_eq!(
        normalize_wrapped_exit(status, Some(Duration::from_secs(1)), None),
        EXIT_TIMEOUT
    );
}

#[cfg(unix)]
#[test]
fn normalizes_wrapped_memory_signal() {
    let status = std::process::ExitStatus::from_raw(libc::SIGSEGV);
    assert_eq!(
        normalize_wrapped_exit(status, None, Some(1024)),
        EXIT_OUT_OF_MEMORY
    );
}

#[cfg(unix)]
#[test]
fn normalizes_wrapped_timeout_exit_code() {
    let status = std::process::ExitStatus::from_raw((128 + libc::SIGXCPU) << 8);
    assert_eq!(
        normalize_wrapped_exit(status, Some(Duration::from_secs(1)), None),
        EXIT_TIMEOUT
    );
}
