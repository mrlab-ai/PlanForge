//! The registry as a binary sees it once it registered this crate.

use planforge_search::heuristic_features::HEURISTICS;

#[test]
fn heuristic_registry_names_are_unique_and_complete() {
    HEURISTICS.ensure_registered(planforge_heuristics::heuristics);
    let names = HEURISTICS.names().collect::<Vec<_>>();
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
    use planforge_search::config::parse_call;
    use planforge_search::heuristic_features::HeuristicConfig;

    HEURISTICS.ensure_registered(planforge_heuristics::heuristics);
    for entry in HEURISTICS.entries() {
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
        HeuristicConfig::from_spec(&parse_call(&spec).unwrap_or_else(|error| {
            panic!(
                "the reference of `{}` does not parse: {error}\n{spec}",
                entry.name()
            )
        }))
        .unwrap_or_else(|error| {
            panic!(
                "the defaults of `{}` are rejected: {error}\n{spec}",
                entry.name()
            )
        });
    }
}
