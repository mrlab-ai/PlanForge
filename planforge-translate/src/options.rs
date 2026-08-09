/// Configuration options for the translator.

/// Whether to use partial encoding (default: true)
pub const USE_PARTIAL_ENCODING: bool = true;

/// Maximum candidates for invariant generation (default: 100000)
pub const INVARIANT_GENERATION_MAX_CANDIDATES: usize = 100000;

/// Maximum time for invariant generation in seconds (default: 300)
pub const INVARIANT_GENERATION_MAX_TIME: u64 = 300;

/// Whether to add implied preconditions (default: false)
pub const ADD_IMPLIED_PRECONDITIONS: bool = false;

/// Whether to filter unreachable facts (default: true)
pub const FILTER_UNREACHABLE_FACTS: bool = true;

/// Whether to dump the task (default: false)
pub const DUMP_TASK: bool = false;

/// Whether to generate a relaxed task (default: false)
pub const GENERATE_RELAXED_TASK: bool = false;
