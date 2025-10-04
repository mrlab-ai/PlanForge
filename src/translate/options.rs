// options stub
pub fn options_placeholder() {}

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "translator")]
#[command(about = "Fast Downward PDDL to SAS+ translator")]
pub struct Options {
    /// Path to domain PDDL file
    #[arg(help = "path to domain pddl file")]
    pub domain: String,

    /// Path to task PDDL file (called "task" in Python, "problem" is common alias)
    #[arg(help = "path to task pddl file")]
    pub task: String,

    /// Output relaxed task (no delete effects)
    #[arg(long = "relaxed")]
    pub generate_relaxed_task: bool,

    /// Use full encoding instead of partial encoding
    /// Note: This inverts the use_partial_encoding default to match Python behavior
    #[arg(long = "full-encoding")]
    pub full_encoding: bool,

    /// Max number of candidates for invariant generation
    #[arg(long = "invariant-generation-max-candidates", default_value = "100000")]
    pub invariant_generation_max_candidates: i32,

    /// Max time for invariant generation
    #[arg(long = "invariant-generation-max-time", default_value = "300")]
    pub invariant_generation_max_time: i32,

    /// Infer additional preconditions
    #[arg(long = "add-implied-preconditions")]
    pub add_implied_preconditions: bool,

    /// Keep facts that can't be reached from the initial state
    #[arg(long = "keep-unreachable-facts")]
    pub keep_unreachable_facts: bool,

    /// Dump human-readable SAS+ representation of the task
    #[arg(long = "dump-task")]
    pub dump_task: bool,
}

impl Options {
    /// Get use_partial_encoding (inverted from full_encoding to match Python)
    pub fn use_partial_encoding(&self) -> bool {
        !self.full_encoding
    }

    /// Get filter_unreachable_facts (inverted from keep_unreachable_facts to match Python)
    pub fn filter_unreachable_facts(&self) -> bool {
        !self.keep_unreachable_facts
    }
}

impl Default for Options {
    fn default() -> Self {
        Self {
            domain: String::new(),
            task: String::new(),
            generate_relaxed_task: false,
            full_encoding: false,  // Default false means use_partial_encoding = true
            invariant_generation_max_candidates: 100000,
            invariant_generation_max_time: 300,
            add_implied_preconditions: false,
            keep_unreachable_facts: false,  // Default false means filter_unreachable_facts = true
            dump_task: false,
        }
    }
}

impl Options {
    pub fn parse_args() -> Self {
        Options::parse()
    }
}
