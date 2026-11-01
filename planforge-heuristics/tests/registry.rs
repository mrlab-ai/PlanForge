//! The registry as a binary sees it once it registered this crate.

use planforge_search::heuristic_factory::{ensure_heuristics_registered, heuristic_names};

#[test]
fn heuristic_registry_names_are_unique_and_complete() {
    ensure_heuristics_registered(planforge_heuristics::heuristics);
    let names = heuristic_names().collect::<Vec<_>>();
    let unique = names
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        names.len(),
        unique.len(),
        "duplicate registry names: {names:?}"
    );
    for required in [
        "blind",
        "ff",
        "domain",
        "cartesian",
        "scp",
        "pdb",
        "lmcutnumeric",
        "numeric_potential",
    ] {
        assert!(
            unique.contains(required),
            "missing registry entry {required}"
        );
    }
}
