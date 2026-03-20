struct SearchAlgorithm {
    description: String,
    search_status: SearchStatus,
    search_result: SearchResult,
    solution_found: bool,
    plan: Vec<String>, // TODO: change that
}

struct SearchStatus {}

struct SearchResult {}
