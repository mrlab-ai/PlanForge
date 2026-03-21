use std::time::{Duration, Instant};

pub struct Timer {
    start_wall: Instant,
    start_cpu: f64,
}

impl Timer {
    pub fn new() -> Self {
        Self {
            start_wall: Instant::now(),
            start_cpu: cpu_seconds(),
        }
    }

    pub fn cpu_elapsed(&self) -> Duration {
        let elapsed = cpu_seconds() - self.start_cpu;
        if elapsed <= 0.0 {
            Duration::from_secs(0)
        } else {
            Duration::from_secs_f64(elapsed)
        }
    }

    pub fn wall_elapsed(&self) -> Duration {
        self.start_wall.elapsed()
    }
}

impl std::fmt::Display for Timer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{:.3}s CPU, {:.3}s wall-clock]",
            self.cpu_elapsed().as_secs_f64(),
            self.wall_elapsed().as_secs_f64()
        )
    }
}

#[cfg(unix)]
fn cpu_seconds() -> f64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let result = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if result != 0 {
        return 0.0;
    }
    let usage = unsafe { usage.assume_init() };
    let user = usage.ru_utime;
    let sys = usage.ru_stime;
    user.tv_sec as f64
        + user.tv_usec as f64 / 1_000_000.0
        + sys.tv_sec as f64
        + sys.tv_usec as f64 / 1_000_000.0
}

#[cfg(not(unix))]
fn cpu_seconds() -> f64 {
    0.0
}

pub fn timing<F, T>(text: &str, block: bool, f: F) -> T
where
    F: FnOnce() -> T,
{
    let timer = Timer::new();
    if block {
        print!("{}...\n", text);
    } else {
        print!("{}... ", text);
    }
    use std::io::Write;
    std::io::stdout().flush().ok();

    let result = f();

    if block {
        println!("{}: {}", text, timer);
    } else {
        println!("{}", timer);
    }
    std::io::stdout().flush().ok();
    result
}
