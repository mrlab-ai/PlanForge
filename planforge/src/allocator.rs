//! The allocator, and what the process does when it runs out of memory or
//! catches a fatal signal.
//!
//! Both paths run with the heap already exhausted or the process already dying,
//! so everything below the reporting entry points is reentrant: raw `write`
//! calls into a file descriptor, no formatting, no allocation.

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Once, OnceLock};

#[cfg(unix)]
static OOM_REPORTED: AtomicBool = AtomicBool::new(false);

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
struct ReportingAllocator;

#[cfg(unix)]
#[global_allocator]
static GLOBAL_ALLOCATOR: ReportingAllocator = ReportingAllocator;

#[cfg(unix)]
static MIMALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[cfg(unix)]
static OOM_RECOVERY: OnceLock<fn() -> bool> = OnceLock::new();

#[cfg(unix)]
pub(crate) fn install_oom_recovery(recovery: fn() -> bool) {
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

#[cfg(unix)]
pub(crate) fn register_event_handlers() {
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
pub(crate) fn register_event_handlers() {}

#[cfg(unix)]
extern "C" fn signal_handler(signal_number: libc::c_int) {
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
unsafe fn report_out_of_memory_and_exit() -> ! {
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
unsafe fn print_peak_memory_reentrant(fd: libc::c_int) {
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
unsafe fn print_peak_memory_reentrant(_fd: libc::c_int) {}

/// Write into a file descriptor.
///
/// # Safety
/// This uses `libc`.
#[cfg(unix)]
unsafe fn write_fd(fd: libc::c_int, mut bytes: &[u8]) {
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
unsafe fn write_number_fd(fd: libc::c_int, value: u64) {
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
