pub fn search_algorithm_help() -> String {
    format!(
        "Search specification. Available algorithms: astar([heuristic], mpd=false, lazy=none), gbfs([heuristic])\n\n{}",
        crate::heuristic_factory::heuristic_help()
    )
}
