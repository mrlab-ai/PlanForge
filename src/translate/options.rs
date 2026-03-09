/// Port of options.py
/// Configuration options for the translator.

/// Whether to use partial encoding (default: true)
/// Python: argparser.add_argument("--full-encoding", dest="use_partial_encoding", action="store_false")
pub const USE_PARTIAL_ENCODING: bool = true;

/// Maximum candidates for invariant generation (default: 100000)
/// Python: argparser.add_argument("--invariant-generation-max-candidates", default=100000, type=int)
pub const INVARIANT_GENERATION_MAX_CANDIDATES: usize = 100000;

/// Maximum time for invariant generation in seconds (default: 300)
/// Python: argparser.add_argument("--invariant-generation-max-time", default=300, type=int)
pub const INVARIANT_GENERATION_MAX_TIME: u64 = 300;

/// Whether to add implied preconditions (default: false)
/// Python: argparser.add_argument("--add-implied-preconditions", action="store_true")
pub const ADD_IMPLIED_PRECONDITIONS: bool = false;

/// Whether to filter unreachable facts (default: true)
/// Python: argparser.add_argument("--keep-unreachable-facts", dest="filter_unreachable_facts", action="store_false")
pub const FILTER_UNREACHABLE_FACTS: bool = true;

/// Whether to dump the task (default: false)
/// Python: argparser.add_argument("--dump-task", action="store_true")
pub const DUMP_TASK: bool = false;

/// Whether to generate a relaxed task (default: false)
/// Python: argparser.add_argument("--relaxed", dest="generate_relaxed_task", action="store_true")
pub const GENERATE_RELAXED_TASK: bool = false;
