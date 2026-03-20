use super::*;

#[test]
fn test_search_status_enum() {
    // Test basic enum functionality
    assert_eq!(SearchStatus::InProgress, SearchStatus::InProgress);
    assert_ne!(SearchStatus::Solved(0), SearchStatus::Failed);
    assert_ne!(SearchStatus::MemoryLimitReached, SearchStatus::Timeout);
}

#[test]
fn test_search_result_creation() {
    let result = SearchResult {
        status: SearchStatus::Failed,
        plan: None,
        solution_cost: None,
        nodes_expanded: 0,
        nodes_generated: 0,
        search_time: Duration::from_millis(100),
    };

    assert_eq!(result.status, SearchStatus::Failed);
    assert!(result.plan.is_none());
    assert_eq!(result.nodes_expanded, 0);
}
