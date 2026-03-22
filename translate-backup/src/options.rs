use clap::Parser;
use std::sync::OnceLock;

#[derive(Debug, Clone, clap::Parser)]
#[command(name = "translate", about = "Translator options")]
pub struct Options {
    /// path to domain pddl file
    pub domain: String,
    /// path to task pddl file
    pub task: String,
    /// output relaxed task (no delete effects)
    #[arg(long, default_value_t = false)]
    pub relaxed: bool,
    /// represent facts in multiple mutex groups in multiple variables
    #[arg(long = "full-encoding", default_value_t = false)]
    pub full_encoding: bool,
    /// max number of candidates for invariant generation
    #[arg(long = "invariant-generation-max-candidates", default_value_t = 100000)]
    pub invariant_generation_max_candidates: i32,
    /// max time for invariant generation (seconds)
    #[arg(long = "invariant-generation-max-time", default_value_t = 300)]
    pub invariant_generation_max_time: i32,
    /// infer additional preconditions
    #[arg(long = "add-implied-preconditions", default_value_t = false)]
    pub add_implied_preconditions: bool,
    /// keep facts that can't be reached from the initial state
    #[arg(long = "keep-unreachable-facts", default_value_t = false)]
    pub keep_unreachable_facts: bool,
    /// dump human-readable SAS+ representation of the task
    #[arg(long = "dump-task", default_value_t = false)]
    pub dump_task: bool,
}

static OPTIONS: OnceLock<Options> = OnceLock::new();

pub fn parse_args() -> Options {
    Options::parse()
}

pub fn setup() -> &'static Options {
    let opts = parse_args();
    let _ = OPTIONS.set(opts);
    OPTIONS.get().expect("options initialized")
}

pub fn get() -> Option<&'static Options> {
    OPTIONS.get()
}
