//! `--portfolio`: a two-stage sequential portfolio over this same binary.
//!
//! Stage 1 runs `astar(lmcutnumeric())` under a tight budget. LM-cut is
//! admissible and often finds optimal plans quickly on small-to-medium tasks.
//!
//! Stage 2 runs `astar(canonical_domain_abstractions(...))` with the user's
//! preferred CEGAR construction budget and, by default, *no* search-side time
//! limit -- canonical's stronger abstractions handle the cases LM-cut struggles
//! with, and they need the time.
//!
//! Each stage is a child process rather than an in-process call, because that
//! is what makes stage 1's memory and time limits enforceable: the limits are
//! per-process, and stage 2 has to start from a clean address space after
//! stage 1 was killed for exceeding them.

use std::ffi::OsString;
use std::path::Path;
use std::process::{Command, ExitStatus};
use std::time::{Duration, Instant};

use clap::Args;

use crate::PlannersCli;

#[derive(Args, Debug, Clone)]
pub struct PortfolioOptions {
    /// Run the two-stage portfolio (LM-cut, then canonical domain
    /// abstractions) instead of a single search.
    #[arg(long, conflicts_with_all = ["max_time", "max_memory", "search"])]
    pub portfolio: bool,

    /// LM-cut stage memory cap.
    #[arg(long, default_value = "7G", requires = "portfolio")]
    pub lmcut_memory: String,

    /// LM-cut stage wall-clock budget.
    #[arg(long, default_value = "5m", requires = "portfolio")]
    pub lmcut_time: String,

    /// CEGAR `total_max_time` for the canonical fallback, in seconds.
    #[arg(long, default_value = "300", requires = "portfolio")]
    pub canonical_construction_time: u64,

    /// Memory cap for the canonical fallback.
    #[arg(long, default_value = "8G", requires = "portfolio")]
    pub canonical_memory: String,

    /// Optional wall-clock cap for the canonical fallback. When unset,
    /// canonical runs until a plan is found, the process is killed for
    /// exceeding its memory cap, or the user interrupts.
    #[arg(long, requires = "portfolio")]
    pub canonical_time: Option<String>,
}

/// Run both stages and exit with the outcome. Returns only on an I/O error
/// spawning a stage; otherwise it exits the process, like
/// [`crate::run_wrapped_process`].
pub fn run_portfolio(cli: &PlannersCli) -> std::io::Result<()> {
    let options = &cli.portfolio;
    assert!(
        options.portfolio,
        "portfolio mode entered without --portfolio"
    );
    // The stages are this same executable, so there is nothing to look up and
    // no way to end up running a different planner than the one asked for.
    let executable = std::env::current_exe()?;

    tracing::info!(
        "[portfolio] stage 1: astar(lmcutnumeric())  --max-time {}  --max-memory {}",
        options.lmcut_time,
        options.lmcut_memory
    );
    let stage1 = run_stage(
        &executable,
        cli,
        "astar(lmcutnumeric())",
        Some(&options.lmcut_time),
        Some(&options.lmcut_memory),
    )?;
    if stage1.answered {
        tracing::info!(
            "[portfolio] stage 1 answered (exit {}, {})",
            stage1.status,
            describe_duration(stage1.elapsed)
        );
        std::process::exit(0);
    }
    tracing::info!(
        "[portfolio] stage 1 ran out of budget (exit {}, {})",
        stage1.status,
        describe_duration(stage1.elapsed)
    );

    let canonical_spec = format!(
        "astar(canonical_domain_abstractions(blacklist_trigger_percentage=0.6,total_max_time={},flaw_treatment=max_refined_single_atom,numeric_split_strategy=standard,flaw_kind=execute_entire_plan,use_wildcard_plans=true,combine_labels=true,max_abstraction_size=100000,max_collection_size=1000000))",
        options.canonical_construction_time,
    );
    tracing::info!(
        "[portfolio] stage 2: canonical_domain_abstractions(...)  --max-memory {}  {}",
        options.canonical_memory,
        match options.canonical_time.as_deref() {
            Some(limit) => format!("--max-time {limit}"),
            None => "(no time limit)".to_string(),
        }
    );
    let stage2 = run_stage(
        &executable,
        cli,
        &canonical_spec,
        options.canonical_time.as_deref(),
        Some(&options.canonical_memory),
    )?;
    if stage2.answered {
        tracing::info!(
            "[portfolio] stage 2 answered (exit {}, {})",
            stage2.status,
            describe_duration(stage2.elapsed)
        );
        std::process::exit(0);
    }
    tracing::info!(
        "[portfolio] stage 2 ran out of budget (exit {}, {})",
        stage2.status,
        describe_duration(stage2.elapsed)
    );
    // Propagate the canonical stage's exit code so callers can distinguish
    // "OOM in stage 2" from "translate failed in stage 2" etc.
    std::process::exit(stage2.status.code().unwrap_or(1));
}

struct StageResult {
    /// The stage terminated with an answer: either a plan, or a completed
    /// optimal search that proved there is none. Both mean the next stage has
    /// nothing left to contribute.
    answered: bool,
    status: ExitStatus,
    elapsed: Duration,
}

/// Spawn one stage as a full `planforge` run: it does its own translation, its
/// own resource-limit wrapping and its own plan writing.
fn run_stage(
    executable: &Path,
    cli: &PlannersCli,
    search: &str,
    time_limit: Option<&str>,
    memory_limit: Option<&str>,
) -> std::io::Result<StageResult> {
    let mut args: Vec<OsString> = vec![OsString::from("--search"), OsString::from(search)];
    if let Some(time) = time_limit {
        args.push(OsString::from("--max-time"));
        args.push(OsString::from(time));
    }
    if let Some(memory) = memory_limit {
        args.push(OsString::from("--max-memory"));
        args.push(OsString::from(memory));
    }
    if let Some(level) = cli.log_level {
        args.push(OsString::from("--log-level"));
        args.push(OsString::from(level.to_string()));
    }
    if cli.restrict_task {
        args.push(OsString::from("--restrict-task"));
    }
    args.extend(cli.inputs.iter().cloned().map(OsString::from));

    let mut command = Command::new(executable);
    command.args(args);
    command.stdout(std::process::Stdio::inherit());
    command.stderr(std::process::Stdio::inherit());

    let start = Instant::now();
    let status = command.status()?;
    let elapsed = start.elapsed();

    // `exit_code_for_search_status` gives 0 to a stage that ran to completion --
    // whether it found a plan or exhausted the state space proving there is
    // none -- and a distinct positive code to one that hit its time or memory
    // limit. A failed translation exits non-zero as well. So exit 0 is exactly
    // "there is nothing more for the next stage to try".
    Ok(StageResult {
        answered: status.success(),
        status,
        elapsed,
    })
}

fn describe_duration(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs_f64();
    if seconds < 60.0 {
        format!("{seconds:.1}s")
    } else if seconds < 3600.0 {
        format!(
            "{}m{:02}s",
            (seconds / 60.0).floor() as u64,
            (seconds as u64) % 60
        )
    } else {
        format!(
            "{}h{:02}m",
            (seconds / 3600.0).floor() as u64,
            ((seconds as u64) / 60) % 60
        )
    }
}
