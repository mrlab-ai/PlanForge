//! Two-stage portfolio runner.
//!
//! Stage 1: invoke `planforge --search 'astar(lmcutnumeric())'` with a tight
//! 5-minute / 7 GiB budget. LM-cut is admissible and often finds optimal plans
//! quickly on small-to-medium tasks.
//!
//! Stage 2: if stage 1 doesn't produce a plan (timeout / OOM / no solution
//! within the budget), fall back to
//! `astar(canonical_domain_abstractions(...))` with the user's preferred
//! CEGAR construction budget (300 s by default) and *no* search-side time
//! limit — keep running until a plan is found. Canonical's stronger
//! abstractions handle the cases LM-cut struggles with.
//!
//! Both stages call the same `planforge` binary as a child process, so the
//! portfolio inherits planforge's existing translate + preprocess pipeline.
//! Stage 1 produces an `output` SAS file in the working directory as a
//! side-effect; stage 2 reuses it to skip a second translate.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "planforge-portfolio",
    about = "Two-stage portfolio: LMcut (7G/5min) → canonical DA (no limit)"
)]
struct Cli {
    /// Path to the `planforge` binary. Defaults to looking in `$PATH` for
    /// `planforge`, then falling back to `target/release/planforge` relative
    /// to the current working directory.
    #[arg(long, value_name = "PATH")]
    planforge: Option<PathBuf>,

    /// LM-cut stage memory cap. Default 7 GiB.
    #[arg(long, default_value = "7G")]
    lmcut_memory: String,

    /// LM-cut stage wall-clock budget. Default 5m.
    #[arg(long, default_value = "5m")]
    lmcut_time: String,

    /// CEGAR `total_max_time` for the canonical fallback. Default 300 s.
    #[arg(long, default_value = "300")]
    canonical_construction_time: u64,

    /// Memory cap for the canonical fallback. Default 8 GiB.
    #[arg(long, default_value = "8G")]
    canonical_memory: String,

    /// Optional total wall-clock cap for the canonical fallback. When unset
    /// (the default) canonical runs until a plan is found, the process is
    /// OOM-killed, or the user interrupts.
    #[arg(long)]
    canonical_time: Option<String>,

    /// One PDDL pair (`domain.pddl problem.pddl`) or one pre-translated
    /// SAS file. Passed verbatim to both stages.
    #[arg(value_name = "INPUT", required = true, num_args = 1..=2)]
    inputs: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let planforge = resolve_planforge_binary(cli.planforge.as_deref())?;

    eprintln!(
        "[portfolio] stage 1: astar(lmcutnumeric())  --max-time {}  --max-memory {}",
        cli.lmcut_time, cli.lmcut_memory
    );
    let stage1 = run_stage(
        &planforge,
        "astar(lmcutnumeric())",
        Some(&cli.lmcut_time),
        Some(&cli.lmcut_memory),
        &cli.inputs,
    )?;
    if stage1.solved {
        eprintln!(
            "[portfolio] stage 1 solved (exit {}, {})",
            stage1.status,
            describe_duration(stage1.elapsed)
        );
        std::process::exit(0);
    }
    eprintln!(
        "[portfolio] stage 1 did not produce a plan (exit {}, {})",
        stage1.status,
        describe_duration(stage1.elapsed)
    );

    let canonical_spec = format!(
        "astar(canonical_domain_abstractions(blacklist_trigger_percentage=0.6,total_max_time={},flaw_treatment=max_refined_single_atom,numeric_split_strategy=standard,flaw_kind=execute_entire_plan,use_wildcard_plans=true,combine_labels=true,max_abstraction_size=100000,max_collection_size=1000000))",
        cli.canonical_construction_time,
    );

    // Reuse the SAS file produced by stage 1 if it exists; otherwise re-pass
    // the original inputs and let stage 2 translate again.
    let stage2_inputs: Vec<String> = if Path::new("output").exists() {
        eprintln!("[portfolio] reusing translated SAS `output` from stage 1");
        vec!["output".to_string()]
    } else {
        cli.inputs.clone()
    };

    eprintln!(
        "[portfolio] stage 2: {}  --max-memory {} {}",
        short_search_spec(&canonical_spec),
        cli.canonical_memory,
        cli.canonical_time
            .as_deref()
            .map(|t| format!("--max-time {t}"))
            .unwrap_or_else(|| "(no time limit)".to_string()),
    );
    let stage2 = run_stage(
        &planforge,
        &canonical_spec,
        cli.canonical_time.as_deref(),
        Some(&cli.canonical_memory),
        &stage2_inputs,
    )?;
    if stage2.solved {
        eprintln!(
            "[portfolio] stage 2 solved (exit {}, {})",
            stage2.status,
            describe_duration(stage2.elapsed)
        );
        std::process::exit(0);
    }
    eprintln!(
        "[portfolio] stage 2 did not produce a plan (exit {}, {})",
        stage2.status,
        describe_duration(stage2.elapsed)
    );
    // Propagate the canonical stage's exit code so callers can distinguish
    // "OOM in stage 2" from "translate failed in stage 2" etc.
    std::process::exit(stage2.status.code().unwrap_or(1));
}

struct StageResult {
    solved: bool,
    status: ExitStatus,
    elapsed: Duration,
}

fn run_stage(
    planforge: &Path,
    search: &str,
    time_limit: Option<&str>,
    memory_limit: Option<&str>,
    inputs: &[String],
) -> Result<StageResult> {
    let mut cmd = Command::new(planforge);
    cmd.arg("--search").arg(search);
    if let Some(time) = time_limit {
        cmd.arg("--max-time").arg(time);
    }
    if let Some(mem) = memory_limit {
        cmd.arg("--max-memory").arg(mem);
    }
    for input in inputs {
        cmd.arg(input);
    }
    cmd.stdout(std::process::Stdio::inherit());
    cmd.stderr(std::process::Stdio::inherit());

    let start = std::time::Instant::now();
    let status = cmd
        .status()
        .with_context(|| format!("failed to spawn {}", planforge.display()))?;
    let elapsed = start.elapsed();

    // planforge exits 0 on solved, non-zero on failure (timeout, OOM,
    // unsolvable, parse error). `exit_code_for_search_status` in the
    // searcher maps `SearchStatus::Solved` to 0 and everything else to a
    // distinct positive code. So success === exit 0.
    let solved = status.success();
    Ok(StageResult {
        solved,
        status,
        elapsed,
    })
}

fn resolve_planforge_binary(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if !path.exists() {
            bail!("`--planforge {}` does not exist", path.display());
        }
        return Ok(path.to_path_buf());
    }

    // Try `$PATH` first via `which`-style logic, falling back to
    // `target/release/planforge` and `target/debug/planforge` if running
    // from a workspace checkout.
    if let Ok(path) = std::env::var("PLANFORGE_BIN") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Ok(p);
        }
    }
    for candidate in [
        PathBuf::from("planforge"),
        PathBuf::from("target/release/planforge"),
        PathBuf::from("target/debug/planforge"),
    ] {
        if candidate.exists() {
            return Ok(candidate);
        }
        // Best-effort PATH lookup for the bare `planforge` name.
        if candidate.as_os_str() == "planforge"
            && let Some(path) = which_in_path("planforge")
        {
            return Ok(path);
        }
    }
    bail!(
        "could not locate the `planforge` binary. Pass --planforge PATH or set PLANFORGE_BIN, or run from a workspace checkout with `target/release/planforge` built."
    );
}

fn which_in_path(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn describe_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 60.0 {
        format!("{secs:.1}s")
    } else if secs < 3600.0 {
        format!("{}m{:02}s", (secs / 60.0).floor() as u64, (secs as u64) % 60)
    } else {
        format!(
            "{}h{:02}m",
            (secs / 3600.0).floor() as u64,
            ((secs as u64) / 60) % 60
        )
    }
}

fn short_search_spec(spec: &str) -> String {
    // Trim the search spec for log readability — full spec is shown by
    // planforge itself in stderr.
    if let Some(paren) = spec.find('(') {
        return format!("{}(...)", &spec[..paren]);
    }
    spec.to_string()
}
