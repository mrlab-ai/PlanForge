use std::collections::BTreeSet;

use super::*;

fn test_heuristic(
    num_abstractions: usize,
    constraints: Vec<Vec<usize>>,
) -> PostHocOptimizationHeuristic<'static> {
    PostHocOptimizationHeuristic {
        name: "posthoc_test".to_string(),
        components: Vec::new(),
        lp_model: RefCell::new(build_lp_model(num_abstractions, &constraints).unwrap()),
        constraints,
        state_value_cache: RefCell::new(StateValueCache::default()),
        values: RefCell::default(),
        diagnostics_logged: RefCell::new(false),
    }
}

fn operators(ids: &[usize]) -> BTreeSet<OperatorIndex> {
    ids.iter()
        .map(|&id| OperatorIndex::from_usize(id))
        .collect()
}

#[test]
fn no_active_abstractions_returns_zero() {
    let heuristic = test_heuristic(2, vec![vec![0, 1]]);
    assert_eq!(heuristic.solve_dual(&[0.0, 0.0]).unwrap(), 0.0);
}

#[test]
fn single_abstraction_returns_its_h() {
    // One abstraction, with one constraint mentioning it.
    let heuristic = test_heuristic(1, vec![vec![0]]);
    assert!((heuristic.solve_dual(&[7.0]).unwrap() - 7.0).abs() < 1e-9);
}

#[test]
fn two_disjoint_abstractions_sum() {
    // Two abstractions, each with its own constraint. No shared operator
    // means both X_i can be 1, total = h_0 + h_1.
    let heuristic = test_heuristic(2, vec![vec![0], vec![1]]);
    assert!((heuristic.solve_dual(&[3.0, 4.0]).unwrap() - 7.0).abs() < 1e-9);
}

#[test]
fn two_competing_abstractions_take_max() {
    // Two abstractions with a shared operator: constraint forces
    // X_0 + X_1 <= 1, optimum picks the larger h.
    let heuristic = test_heuristic(2, vec![vec![0, 1]]);
    let v = heuristic.solve_dual(&[3.0, 5.0]).unwrap();
    assert!((v - 5.0).abs() < 1e-9, "got {v}");
}

#[test]
fn three_way_packing_picks_best() {
    // Abstractions 0,1,2; constraints {0,2} and {1,2}. Selecting X_2 = 1
    // forces X_0 = X_1 = 0 (objective 4). Selecting X_0 = X_1 = 1 yields 3+5 = 8.
    let heuristic = test_heuristic(3, vec![vec![0, 2], vec![1, 2]]);
    let v = heuristic.solve_dual(&[3.0, 5.0, 4.0]).unwrap();
    assert!((v - 8.0).abs() < 1e-9, "got {v}");
}

#[test]
fn build_constraints_drops_free_operators() {
    let rel0 = operators(&[0, 1]);
    let rel1 = operators(&[0]);

    // Operator 0 cost > 0, operator 1 cost = 0 → only operator 0 produces a
    // constraint, which mentions abstractions 0 and 1.
    let cons = build_constraints(&[rel0, rel1], &[2.0, 0.0]);
    assert_eq!(cons.len(), 1);
    assert_eq!(cons[0], vec![0, 1]);
}

#[test]
fn build_constraints_deduplicates() {
    // Operators 0 and 1 are relevant to both abstractions, so both give the
    // constraint over abstractions 0 and 1, kept once.
    let rel = operators(&[0, 1]);
    let cons = build_constraints(&[rel.clone(), rel], &[1.0, 1.0]);
    assert_eq!(cons.len(), 1);
    assert_eq!(cons[0], vec![0, 1]);
}
