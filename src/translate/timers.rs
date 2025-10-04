//! Timing utilities for performance measurement
//! Port of python/translate/timers.py

use std::time::{Duration, Instant};
use std::fmt;

/// Timer that matches Python's Timer class behavior:
/// - Automatically starts timing when created
/// - Tracks wall-clock time (CPU time not available in safe Rust)
/// - Provides formatted string output
pub struct Timer {
    start_time: Instant,
    #[allow(dead_code)]
    start_clock: Instant, // Placeholder for CPU time (not implemented)
}

impl Timer {
    /// Create and start a new timer (matches Python Timer.__init__)
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            start_time: now,
            start_clock: now, // CPU time not available, use wall-clock as placeholder
        }
    }
    
    /// Get elapsed wall-clock time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// Get elapsed time as seconds (for compatibility)
    pub fn elapsed_secs(&self) -> f64 {
        self.elapsed().as_secs_f64()
    }
}

impl fmt::Display for Timer {
    /// Format timer output to match Python's "[%.3fs CPU, %.3fs wall-clock]"
    /// Note: CPU time not available in safe Rust, so we use wall-clock for both
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elapsed = self.elapsed_secs();
        write!(f, "[{:.3}s CPU, {:.3}s wall-clock]", elapsed, elapsed)
    }
}

/// Simple timing function for compatibility
/// Note: This is a simplified version - Python has a full context manager
pub fn timing<F, R>(text: &str, block: bool, f: F) -> R
where
    F: FnOnce() -> R,
{
    let timer = Timer::new();
    if block {
        println!("{}...", text);
    } else {
        print!("{}... ", text);
        use std::io::{self, Write};
        io::stdout().flush().unwrap();
    }
    
    let result = f();
    
    if block {
        println!("{}: {}", text, timer);
    } else {
        println!("{}", timer);
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_timer() {
        let timer = Timer::new();
        thread::sleep(std::time::Duration::from_millis(10));
        
        let elapsed = timer.elapsed();
        assert!(elapsed.as_millis() >= 10);
        
        // Test string formatting
        let timer_str = format!("{}", timer);
        assert!(timer_str.starts_with("["));
        assert!(timer_str.contains("CPU"));
        assert!(timer_str.contains("wall-clock"));
        assert!(timer_str.ends_with("]"));
    }
}
