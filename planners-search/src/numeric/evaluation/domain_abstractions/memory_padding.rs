//! Mirror of numeric-FD's `utils::reserve_extra_memory_padding` mechanism
//! (see `numeric-fd/src/search/domain_abstractions/cegar.cc:2544` and the
//! check at `cegar.cc:2845-2848`).
//!
//! C++ reserves a chunk of memory at startup and releases it when the
//! out-of-memory handler fires; CEGAR's outer loop then polls
//! `extra_memory_padding_is_reserved()` and stops cleanly. Rust's
//! `set_alloc_error_hook` runs *before* abort and cannot prevent it,
//! so we instead reserve a padding chunk and poll resident-set-size
//! (RSS) periodically. When usage approaches the configured ceiling we
//! release the padding, set a flag, and CEGAR's outer loop sees
//! `padding_is_reserved() == false` on its next iteration and stops.
//!
//! Configuration is via env var:
//!   * `DA_MEMORY_PADDING_MB` — size of the reservation (default 512 MB).
//!     Set to 0 to disable.
//!   * `DA_MEMORY_LIMIT_MB` — RSS ceiling that triggers release
//!     (default: padding * 16, i.e. ~8 GB headroom by default).

use std::sync::Mutex;
use std::sync::OnceLock;

static PADDING: OnceLock<Mutex<Option<Vec<u8>>>> = OnceLock::new();
static LIMIT_BYTES: OnceLock<u64> = OnceLock::new();

fn padding_cell() -> &'static Mutex<Option<Vec<u8>>> {
    PADDING.get_or_init(|| Mutex::new(None))
}

/// Reserves the memory padding at startup. Subsequent calls are no-ops.
/// `padding_mb=0` disables the mechanism.
pub fn reserve_memory_padding() {
    let padding_mb: usize = std::env::var("DA_MEMORY_PADDING_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    if padding_mb == 0 {
        return;
    }
    let limit_mb: u64 = std::env::var("DA_MEMORY_LIMIT_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(padding_mb as u64 * 16);
    LIMIT_BYTES
        .set(limit_mb.saturating_mul(1024 * 1024))
        .ok();
    let cell = padding_cell();
    let mut guard = cell.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_some() {
        return;
    }
    let bytes = padding_mb.saturating_mul(1024 * 1024);
    let mut buf = Vec::with_capacity(bytes);
    // Touch the pages so the OS actually maps them; otherwise the
    // reservation is virtual-only and won't push us toward the RSS limit.
    buf.resize(bytes, 0xA5_u8);
    *guard = Some(buf);
}

/// Returns true iff the padding is still held. The CEGAR collection loop
/// polls this; a `false` reading means we should stop adding abstractions.
pub fn padding_is_reserved() -> bool {
    let cell = padding_cell();
    cell.lock()
        .map(|g| g.is_some())
        .unwrap_or(false)
}

/// Releases the padding. After this call, `padding_is_reserved()` returns
/// false. Idempotent.
pub fn release_padding() {
    let cell = padding_cell();
    if let Ok(mut g) = cell.lock() {
        *g = None;
    }
}

/// Reads `/proc/self/status` on Linux and returns RSS in bytes.
/// On other platforms or on read failure, returns `None`.
fn current_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let content = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: u64 = rest
                    .split_whitespace()
                    .next()
                    .and_then(|n| n.parse().ok())?;
                return Some(kb * 1024);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Polls the RSS and releases the padding if the configured limit is
/// exceeded. Safe to call frequently; returns `true` if memory is still
/// within limits (padding still held), `false` if the limit has been hit
/// (padding released — caller should stop allocating).
pub fn poll_and_release_if_exceeded() -> bool {
    let Some(&limit_bytes) = LIMIT_BYTES.get() else {
        // Padding never reserved → mechanism disabled. Always "OK".
        return true;
    };
    if !padding_is_reserved() {
        return false;
    }
    if let Some(rss) = current_rss_bytes()
        && rss >= limit_bytes
    {
        tracing::warn!(
            "domain abstraction collection: memory limit reached (RSS={} MB >= {} MB); \
             releasing memory padding and stopping collection",
            rss / (1024 * 1024),
            limit_bytes / (1024 * 1024)
        );
        release_padding();
        return false;
    }
    true
}
