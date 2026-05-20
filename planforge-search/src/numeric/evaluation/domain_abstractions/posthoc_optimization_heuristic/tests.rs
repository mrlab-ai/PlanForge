use std::collections::BTreeSet;

use super::*;

#[test]
fn no_active_abstractions_returns_zero() {
    let heuristic = PostHocOptimizationHeuristic {
        name: "posthoc_test".to_string(),
        heuristics: Vec::new(),
        constraints: Vec::new(),
        state_value_cache: RefCell::new(Vec::new()),
        lookup_scratch: RefCell::new(DomainAbstractionLookupScratch::new()),
        diagnostics_logged: RefCell::new(false),
    };
    // empty h vector
    assert_eq!(heuristic.solve_dual(&[]), 0.0);
}

#[test]
fn single_abstraction_returns_its_h() {
    // One abstraction, with one constraint mentioning it.
    let heuristic = PostHocOptimizationHeuristic {
        name: "posthoc_test".to_string(),
        heuristics: Vec::new(),
        constraints: vec![vec![0]],
        state_value_cache: RefCell::new(Vec::new()),
        lookup_scratch: RefCell::new(DomainAbstractionLookupScratch::new()),
        diagnostics_logged: RefCell::new(false),
    };
    assert!((heuristic.solve_dual(&[7.0]) - 7.0).abs() < 1e-9);
}

#[test]
fn two_disjoint_abstractions_sum() {
    // Two abstractions, each with its own constraint. No shared operator
    // means both X_i can be 1, total = h_0 + h_1.
    let heuristic = PostHocOptimizationHeuristic {
        name: "posthoc_test".to_string(),
        heuristics: Vec::new(),
        constraints: vec![vec![0], vec![1]],
        state_value_cache: RefCell::new(Vec::new()),
        lookup_scratch: RefCell::new(DomainAbstractionLookupScratch::new()),
        diagnostics_logged: RefCell::new(false),
    };
    assert!((heuristic.solve_dual(&[3.0, 4.0]) - 7.0).abs() < 1e-9);
}

#[test]
fn two_competing_abstractions_take_max() {
    // Two abstractions with a shared operator: constraint forces
    // X_0 + X_1 <= 1, optimum picks the larger h.
    let heuristic = PostHocOptimizationHeuristic {
        name: "posthoc_test".to_string(),
        heuristics: Vec::new(),
        constraints: vec![vec![0, 1]],
        state_value_cache: RefCell::new(Vec::new()),
        lookup_scratch: RefCell::new(DomainAbstractionLookupScratch::new()),
        diagnostics_logged: RefCell::new(false),
    };
    let v = heuristic.solve_dual(&[3.0, 5.0]);
    assert!((v - 5.0).abs() < 1e-9, "got {v}");
}

#[test]
fn three_way_packing_picks_best() {
    // Abstractions 0,1,2; constraints {0,2} and {1,2}. Selecting X_2 = 1
    // forces X_0 = X_1 = 0 (objective 4). Selecting X_0 = X_1 = 1 yields 3+5 = 8.
    let heuristic = PostHocOptimizationHeuristic {
        name: "posthoc_test".to_string(),
        heuristics: Vec::new(),
        constraints: vec![vec![0, 2], vec![1, 2]],
        state_value_cache: RefCell::new(Vec::new()),
        lookup_scratch: RefCell::new(DomainAbstractionLookupScratch::new()),
        diagnostics_logged: RefCell::new(false),
    };
    let v = heuristic.solve_dual(&[3.0, 5.0, 4.0]);
    assert!((v - 8.0).abs() < 1e-9, "got {v}");
}

#[test]
fn build_constraints_drops_free_operators() {
    let mut rel0: BTreeSet<usize> = BTreeSet::new();
    rel0.insert(0);
    rel0.insert(1);
    let mut rel1: BTreeSet<usize> = BTreeSet::new();
    rel1.insert(0);

    // Operator 0 cost > 0, operator 1 cost = 0 → only operator 0 produces a
    // constraint, which mentions abstractions 0 and 1.
    let cons = build_constraints(&[rel0, rel1], &[2.0, 0.0], 2);
    assert_eq!(cons.len(), 1);
    assert_eq!(cons[0], vec![0, 1]);
}

#[test]
fn build_constraints_deduplicates() {
    let mut rel: BTreeSet<usize> = BTreeSet::new();
    rel.insert(0);
    rel.insert(1);

    // Two operators with the same relevance pattern produce only one constraint.
    let cons = build_constraints(&[rel.clone(), rel], &[1.0, 1.0], 2);
    // Wait — that's two abstractions both relevant for the same two ops,
    // not two operators sharing a pattern. Re-derive:
    // operator_to_abstractions: op 0 -> [0,1] (relevant for both abstractions),
    //                           op 1 -> [0,1]
    // After dedup these collapse to one constraint.
    assert_eq!(cons.len(), 1);
    assert_eq!(cons[0], vec![0, 1]);
}
