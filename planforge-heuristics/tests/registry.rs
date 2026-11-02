//! The features of this crate as a binary parses with them.

#[test]
fn heuristic_registry_names_are_unique_and_complete() {
    let features = planforge_heuristics::features();
    let names = features.heuristics.names().collect::<Vec<_>>();
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

/// Every heuristic's defaults, written as the option arguments the reference
/// lists, parse back: the reference documents the configuration grammar.
#[test]
fn every_heuristics_default_options_parse_back() {
    let features = planforge_heuristics::features();
    for entry in features.heuristics.entries() {
        let options = entry.options();
        // A combination has no default children; give it one.
        let arguments: Vec<String> = options
            .iter()
            .map(|option| {
                let value = if option.default == "[]" {
                    "[domain()]"
                } else {
                    &option.default
                };
                format!("{}={value}", option.name)
            })
            .collect();
        let spec = format!("{}({})", entry.name(), arguments.join(", "));
        let spec = if entry.name() == "check_admissible" {
            "check_admissible(blind())".to_string()
        } else {
            spec
        };
        features.parse_heuristic(&spec).unwrap_or_else(|error| {
            panic!(
                "the defaults of `{}` are rejected: {error}\n{spec}",
                entry.name()
            )
        });
    }
}
