use std::alloc::{GlobalAlloc, Layout};
use std::os::unix::process::ExitStatusExt;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Once, OnceLock};
use std::time::Duration;
use tracing::info;

pub const EXIT_SUCCESS: i32 = 0;
pub const EXIT_OUT_OF_MEMORY: i32 = 6;
pub const EXIT_TIMEOUT: i32 = 7;

#[cfg(unix)]
pub static OOM_REPORTED: AtomicBool = AtomicBool::new(false);

// `GlobalAlloc` wrapper that delegates to `mimalloc` and intercepts
// null returns to call `report_out_of_memory_and_exit` (graceful exit
// with status 6, peak-memory log, etc.) rather than letting Rust abort.
//
// We can't use `std::alloc::set_alloc_error_hook` for the OOM path
// because it's nightly-only (#51245), so wrapping the allocator at the
// `GlobalAlloc` layer is the only stable way to redirect allocation
// failures away from the default `intrinsics::abort`. The wrapper's
// null check inlines into a single predicted-not-taken branch per
// allocation — essentially free.
//
// mimalloc was chosen because, on tasks dominated by the
// successor-generator's hundreds of thousands of small allocations,
// it decommits free pages more aggressively than glibc's main arena
// (matching numeric-FD's ~500 MB RSS on minecraft 30x30_5 vs glibc's
// ~2 GB), and its small-allocation path is ~11% faster.
#[cfg(unix)]
pub struct ReportingAllocator;

#[cfg(unix)]
#[global_allocator]
pub static GLOBAL_ALLOCATOR: ReportingAllocator = ReportingAllocator;

#[cfg(unix)]
static MIMALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(unix)]
static OOM_RECOVERY: OnceLock<fn() -> bool> = OnceLock::new();

#[cfg(unix)]
pub fn install_oom_recovery(recovery: fn() -> bool) {
    OOM_RECOVERY
        .set(recovery)
        .expect("out-of-memory recovery hook must be installed once");
}

#[cfg(unix)]
fn recover_from_oom() -> bool {
    OOM_RECOVERY.get().is_some_and(|recovery| recovery())
}

#[cfg(unix)]
unsafe impl GlobalAlloc for ReportingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let mut ptr = unsafe { MIMALLOC.alloc(layout) };
        if ptr.is_null() && recover_from_oom() {
            ptr = unsafe { MIMALLOC.alloc(layout) };
        }
        if ptr.is_null() {
            unsafe { report_out_of_memory_and_exit() };
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let mut ptr = unsafe { MIMALLOC.alloc_zeroed(layout) };
        if ptr.is_null() && recover_from_oom() {
            ptr = unsafe { MIMALLOC.alloc_zeroed(layout) };
        }
        if ptr.is_null() {
            unsafe { report_out_of_memory_and_exit() };
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let mut new_ptr = unsafe { MIMALLOC.realloc(ptr, layout, new_size) };
        if new_ptr.is_null() && recover_from_oom() {
            new_ptr = unsafe { MIMALLOC.realloc(ptr, layout, new_size) };
        }
        if new_ptr.is_null() {
            unsafe { report_out_of_memory_and_exit() };
        }
        new_ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { MIMALLOC.dealloc(ptr, layout) }
    }
}

pub fn parse_suffixed_value(
    input: &str,
    default_multiplier: u64,
    units: &[(&str, u64)],
    kind: &str,
) -> Result<u64, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(format!("{} cannot be empty", kind));
    }

    let suffix_start = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    if suffix_start == 0 {
        return Err(format!("{} must start with a number: {}", kind, input));
    }

    let value = trimmed[..suffix_start]
        .parse::<u64>()
        .map_err(|_| format!("invalid {} value: {}", kind, input))?;
    let suffix = trimmed[suffix_start..].trim().to_ascii_lowercase();

    let multiplier = if suffix.is_empty() {
        default_multiplier
    } else {
        units
            .iter()
            .find_map(|(unit, factor)| (*unit == suffix).then_some(*factor))
            .ok_or_else(|| format!("invalid {} suffix '{}': {}", kind, suffix, input))?
    };

    value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("{} is too large: {}", kind, input))
}

pub fn parse_memory_limit(input: &str) -> Result<u64, String> {
    parse_suffixed_value(
        input,
        1,
        &[
            ("b", 1),
            ("k", 1024),
            ("kb", 1024),
            ("m", 1024 * 1024),
            ("mb", 1024 * 1024),
            ("g", 1024 * 1024 * 1024),
            ("gb", 1024 * 1024 * 1024),
            ("t", 1024_u64.pow(4)),
            ("tb", 1024_u64.pow(4)),
        ][..],
        "memory limit",
    )
}

#[cfg(unix)]
pub fn effective_rss_limit(memory_limit: u64) -> std::io::Result<u64> {
    let mut address_space_limit = std::mem::MaybeUninit::<libc::rlimit>::uninit();
    let result = unsafe { libc::getrlimit(libc::RLIMIT_AS, address_space_limit.as_mut_ptr()) };
    if result != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let address_space_limit = unsafe { address_space_limit.assume_init() };
    if address_space_limit.rlim_cur == libc::RLIM_INFINITY {
        Ok(memory_limit)
    } else {
        Ok(memory_limit.min(address_space_limit.rlim_cur))
    }
}

pub fn parse_time_limit(input: &str) -> Result<Duration, String> {
    let seconds = parse_suffixed_value(
        input,
        1,
        &[("ms", 0), ("s", 1), ("m", 60), ("h", 60 * 60)][..],
        "time limit",
    )?;

    if input.trim().to_ascii_lowercase().ends_with("ms") {
        let millis = input.trim()[..input.trim().len() - 2]
            .trim()
            .parse::<u64>()
            .map_err(|_| format!("invalid time limit value: {}", input))?;
        Ok(Duration::from_millis(millis))
    } else {
        Ok(Duration::from_secs(seconds))
    }
}

pub fn format_time_limit(limit: Duration) -> String {
    if limit.subsec_nanos() == 0 {
        format!("{}s", limit.as_secs())
    } else {
        assert_eq!(
            limit.subsec_nanos() % 1_000_000,
            0,
            "time limit must have millisecond precision"
        );
        format!("{}ms", limit.as_millis())
    }
}

#[cfg(target_os = "linux")]
pub fn peak_memory_kb() -> u64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(value) = line.strip_prefix("VmPeak:")
                && let Some(kb) = value
                    .split_whitespace()
                    .next()
                    .and_then(|part| part.parse::<u64>().ok())
            {
                return kb;
            }
        }
    }
    0
}

#[cfg(not(target_os = "linux"))]
pub fn peak_memory_kb() -> u64 {
    0
}

#[cfg(unix)]
pub fn register_event_handlers() {
    static INIT: Once = Once::new();

    // TODO: use signal-hook crate instead.
    #[allow(function_casts_as_integer)]
    INIT.call_once(|| unsafe {
        libc::signal(libc::SIGABRT, signal_handler as libc::sighandler_t);
        libc::signal(libc::SIGTERM, signal_handler as libc::sighandler_t);
        libc::signal(libc::SIGSEGV, signal_handler as libc::sighandler_t);
        libc::signal(libc::SIGINT, signal_handler as libc::sighandler_t);
        libc::signal(libc::SIGXCPU, signal_handler as libc::sighandler_t);
    });
}

#[cfg(not(unix))]
pub fn register_event_handlers() {}

#[cfg(unix)]
pub extern "C" fn signal_handler(signal_number: libc::c_int) {
    unsafe {
        print_peak_memory_reentrant(libc::STDOUT_FILENO);
        write_fd(libc::STDOUT_FILENO, b"caught signal ");
        write_number_fd(libc::STDOUT_FILENO, signal_number as u64);
        write_fd(libc::STDOUT_FILENO, b" -- exiting\n");
        libc::_exit(128 + signal_number);
    }
}

/// Report out of memory and exit.
///
/// # Safety
/// This uses `libc`.
#[cfg(unix)]
pub unsafe fn report_out_of_memory_and_exit() -> ! {
    if OOM_REPORTED.swap(true, Ordering::SeqCst) {
        unsafe { libc::_exit(6) };
    }

    unsafe { write_fd(libc::STDOUT_FILENO, b"Failed to allocate memory.\n") };
    unsafe { write_fd(libc::STDOUT_FILENO, b"Memory limit has been reached.\n") };
    unsafe { print_peak_memory_reentrant(libc::STDOUT_FILENO) };
    unsafe { libc::_exit(6) }
}

/// Print peak memory reentrant.
///
/// # Safety
/// This uses `libc`.
#[cfg(target_os = "linux")]
pub unsafe fn print_peak_memory_reentrant(fd: libc::c_int) {
    let proc_fd = unsafe { libc::open(c"/proc/self/status".as_ptr(), libc::O_RDONLY) };
    if proc_fd < 0 {
        return;
    }

    let magic = b"VmPeak:";
    let mut matched = 0usize;
    let mut found = false;
    let mut wrote_prefix = false;
    let mut buffer = [0u8; 4096];

    loop {
        let bytes_read = unsafe { libc::read(proc_fd, buffer.as_mut_ptr().cast(), buffer.len()) };
        if bytes_read <= 0 {
            break;
        }

        for &byte in &buffer[..bytes_read as usize] {
            if !found {
                if byte == magic[matched] {
                    matched += 1;
                    if matched == magic.len() {
                        found = true;
                    }
                } else {
                    matched = if byte == magic[0] { 1 } else { 0 };
                }
                continue;
            }

            if byte.is_ascii_digit() {
                if !wrote_prefix {
                    unsafe { write_fd(fd, b"Peak memory: ") };
                    wrote_prefix = true;
                }
                unsafe { write_fd(fd, std::slice::from_ref(&byte)) };
            } else if wrote_prefix {
                unsafe { write_fd(fd, b" KB\n") };
                let _ = unsafe { libc::close(proc_fd) };
                return;
            }
        }
    }

    let _ = unsafe { libc::close(proc_fd) };
}

/// Print peak memory reentrant.
///
/// # Safety
/// This uses `libc`.
#[cfg(all(unix, not(target_os = "linux")))]
pub unsafe fn print_peak_memory_reentrant(_fd: libc::c_int) {}

/// Write into a file descriptor.
///
/// # Safety
/// This uses `libc`.
#[cfg(unix)]
pub unsafe fn write_fd(fd: libc::c_int, mut bytes: &[u8]) {
    while !bytes.is_empty() {
        let written = unsafe { libc::write(fd, bytes.as_ptr().cast(), bytes.len()) };
        if written <= 0 {
            break;
        }
        bytes = &bytes[written as usize..];
    }
}

/// Write a number into a file descriptor.
///
/// # Safety
/// This uses `libc`.
#[cfg(unix)]
pub unsafe fn write_number_fd(fd: libc::c_int, value: u64) {
    let mut buffer = [0u8; 32];
    let mut index = buffer.len();
    let mut current = value;

    if current == 0 {
        unsafe { write_fd(fd, b"0") };
        return;
    }

    while current > 0 {
        index -= 1;
        buffer[index] = b'0' + (current % 10) as u8;
        current /= 10;
    }

    unsafe { write_fd(fd, &buffer[index..]) };
}

#[cfg(unix)]
pub fn wrapper_exit_code(status: std::process::ExitStatus) -> i32 {
    status
        .code()
        .unwrap_or_else(|| status.signal().map(|signal| 128 + signal).unwrap_or(1))
}

#[cfg(unix)]
pub fn normalize_wrapped_exit(
    status: std::process::ExitStatus,
    time_limit: Option<Duration>,
    memory_limit: Option<u64>,
) -> i32 {
    if let Some(signal) = status.signal() {
        if signal == libc::SIGXCPU && time_limit.is_some() {
            info!("Time limit reached. Abort search.");
            return EXIT_TIMEOUT;
        }

        if memory_limit.is_some()
            && (signal == libc::SIGABRT || signal == libc::SIGSEGV || signal == libc::SIGKILL)
        {
            info!("Failed to allocate memory.");
            info!("Memory limit has been reached.");
            return EXIT_OUT_OF_MEMORY;
        }
    }

    let exit_code = wrapper_exit_code(status);

    if time_limit.is_some() && exit_code == 128 + libc::SIGXCPU {
        info!("Time limit reached. Abort search.");
        return EXIT_TIMEOUT;
    }

    if memory_limit.is_some()
        && (exit_code == 128 + libc::SIGABRT
            || exit_code == 128 + libc::SIGSEGV
            || exit_code == 128 + libc::SIGKILL)
    {
        info!("Failed to allocate memory.");
        info!("Memory limit has been reached.");
        return EXIT_OUT_OF_MEMORY;
    }

    exit_code
}

#[cfg(unix)]
pub fn apply_process_limits(
    time_limit: Option<Duration>,
    memory_limit: Option<u64>,
) -> std::io::Result<()> {
    if let Some(time_limit) = time_limit {
        let mut soft_limit = time_limit.as_secs();
        if time_limit.subsec_nanos() > 0 {
            soft_limit = soft_limit.saturating_add(1);
        }
        let hard_limit = soft_limit.saturating_add(1);
        let cpu_limit = libc::rlimit {
            rlim_cur: soft_limit as libc::rlim_t,
            rlim_max: hard_limit as libc::rlim_t,
        };

        let result = unsafe { libc::setrlimit(libc::RLIMIT_CPU, &cpu_limit) };
        if result != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }

    if let Some(memory_limit) = memory_limit {
        // mimalloc reserves address space in arenas well ahead of committed
        // resident memory. The wrapper enforces the requested Linux limit
        // against VmRSS; keep RLIMIT_AS only as an emergency ceiling there.
        #[cfg(target_os = "linux")]
        let memory_limit = memory_limit.saturating_mul(4);
        let address_space_limit = libc::rlimit {
            rlim_cur: memory_limit as libc::rlim_t,
            rlim_max: memory_limit as libc::rlim_t,
        };

        let result = unsafe { libc::setrlimit(libc::RLIMIT_AS, &address_space_limit) };
        if result != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }

    Ok(())
}

#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProcessMemory {
    Resident(u64),
    ResidentUnavailable,
    Exited,
}

#[cfg(target_os = "linux")]
fn parse_process_memory(pid: u32, status: &str) -> std::io::Result<ProcessMemory> {
    let state = status
        .lines()
        .find_map(|line| line.strip_prefix("State:"))
        .and_then(|value| value.split_whitespace().next())
        .ok_or_else(|| std::io::Error::other(format!("State missing for process {pid}")))?;
    if state == "Z" || state == "X" {
        return Ok(ProcessMemory::Exited);
    }

    let Some(rss_kib) = status.lines().find_map(|line| {
        line.strip_prefix("VmRSS:").and_then(|value| {
            value
                .split_whitespace()
                .next()
                .and_then(|part| part.parse::<u64>().ok())
        })
    }) else {
        return Ok(ProcessMemory::ResidentUnavailable);
    };
    let rss_bytes = rss_kib
        .checked_mul(1024)
        .ok_or_else(|| std::io::Error::other(format!("VmRSS overflow for process {pid}")))?;
    Ok(ProcessMemory::Resident(rss_bytes))
}

#[cfg(target_os = "linux")]
fn process_memory(pid: u32) -> std::io::Result<ProcessMemory> {
    let status = std::fs::read_to_string(format!("/proc/{pid}/status"))?;
    match parse_process_memory(pid, &status)? {
        ProcessMemory::ResidentUnavailable => {
            let statm = std::fs::read_to_string(format!("/proc/{pid}/statm"))?;
            parse_process_statm(pid, &statm)
        }
        memory => Ok(memory),
    }
}

#[cfg(target_os = "linux")]
fn parse_process_statm(pid: u32, statm: &str) -> std::io::Result<ProcessMemory> {
    let resident_pages = statm
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| std::io::Error::other(format!("resident pages missing for process {pid}")))?
        .parse::<u64>()
        .map_err(|_| std::io::Error::other(format!("invalid resident pages for process {pid}")))?;
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(std::io::Error::last_os_error());
    }
    let rss_bytes = resident_pages
        .checked_mul(page_size as u64)
        .ok_or_else(|| {
            std::io::Error::other(format!("resident memory overflow for process {pid}"))
        })?;
    Ok(ProcessMemory::Resident(rss_bytes))
}

#[cfg(target_os = "linux")]
pub fn wait_with_memory_limit(
    child: &mut std::process::Child,
    memory_limit: u64,
) -> std::io::Result<std::process::ExitStatus> {
    const POLL_INTERVAL: Duration = Duration::from_millis(50);
    const MAX_STATUS_READ_FAILURES: usize = 20;
    let mut consecutive_status_read_failures = 0;

    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        match process_memory(child.id()) {
            Ok(ProcessMemory::Resident(rss)) if rss >= memory_limit => {
                child.kill()?;
                return child.wait();
            }
            Ok(ProcessMemory::Resident(_)) => consecutive_status_read_failures = 0,
            Ok(ProcessMemory::Exited) => return child.wait(),
            Ok(ProcessMemory::ResidentUnavailable) => {
                unreachable!("process_memory resolves unavailable VmRSS through statm")
            }
            Err(error) => {
                if let Some(status) = child.try_wait()? {
                    return Ok(status);
                }
                consecutive_status_read_failures += 1;
                if consecutive_status_read_failures >= MAX_STATUS_READ_FAILURES {
                    return Err(error);
                }
            }
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::{
        ProcessMemory, effective_rss_limit, parse_process_memory, parse_process_statm,
        process_memory, wait_with_memory_limit,
    };
    use std::os::unix::process::ExitStatusExt;
    use std::process::Command;

    #[test]
    fn reads_current_process_rss() {
        assert!(matches!(
            process_memory(std::process::id()).unwrap(),
            ProcessMemory::Resident(rss) if rss > 0
        ));
    }

    #[test]
    fn recognizes_zombie_without_rss_as_exited() {
        let status = "Name:\ttest\nState:\tZ (zombie)\nThreads:\t1\n";
        assert_eq!(
            parse_process_memory(42, status).unwrap(),
            ProcessMemory::Exited
        );
    }

    #[test]
    fn reports_live_process_without_rss_for_statm_fallback() {
        let status = "Name:\ttest\nState:\tR (running)\nThreads:\t1\n";
        assert_eq!(
            parse_process_memory(42, status).unwrap(),
            ProcessMemory::ResidentUnavailable
        );
    }

    #[test]
    fn parses_resident_pages_from_statm() {
        assert!(matches!(
            parse_process_statm(42, "100 25 0 0 0 0 0").unwrap(),
            ProcessMemory::Resident(bytes) if bytes > 0
        ));
    }

    #[test]
    fn effective_memory_limit_does_not_exceed_requested_limit() {
        assert!(effective_rss_limit(8 * 1024 * 1024 * 1024).unwrap() <= 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn terminates_child_over_resident_memory_limit() {
        let mut child = Command::new("sleep").arg("10").spawn().unwrap();
        let status = wait_with_memory_limit(&mut child, 1).unwrap();
        assert_eq!(status.signal(), Some(libc::SIGKILL));
    }

    #[test]
    fn handles_process_status_races_for_quickly_exiting_children() {
        for _ in 0..100 {
            let mut child = Command::new("true").spawn().unwrap();
            assert!(
                wait_with_memory_limit(&mut child, u64::MAX)
                    .unwrap()
                    .success()
            );
        }
    }
}
