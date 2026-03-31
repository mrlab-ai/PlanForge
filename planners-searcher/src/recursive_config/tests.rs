use super::*;

#[test]
fn parses_astar_blind_with_or_without_unit_parens() {
    assert_eq!(
        parse_search_spec("astar(blind)").unwrap(),
        SearchSpec::Astar(HeuristicSpec::Blind)
    );
    assert_eq!(
        parse_search_spec("astar(blind())").unwrap(),
        SearchSpec::Astar(HeuristicSpec::Blind)
    );
}

#[test]
fn parses_astar_domain_abstraction_with_or_without_unit_parens() {
    assert_eq!(
        parse_search_spec("astar(domain_abstraction)").unwrap(),
        SearchSpec::Astar(HeuristicSpec::DomainAbstraction)
    );
    assert_eq!(
        parse_search_spec("astar(domain_abstraction())").unwrap(),
        SearchSpec::Astar(HeuristicSpec::DomainAbstraction)
    );
}

#[test]
fn trims_trailing_punctuation() {
    assert_eq!(
        parse_search_spec("astar(blind()).").unwrap(),
        SearchSpec::Astar(HeuristicSpec::Blind)
    );

    assert_eq!(
        parse_search_spec("astar(domain_abstraction());").unwrap(),
        SearchSpec::Astar(HeuristicSpec::DomainAbstraction)
    );

    assert_eq!(parse_search_spec("da_debug();").unwrap(), SearchSpec::DaDebug);
    assert_eq!(
        parse_search_spec("astar_da_debug();").unwrap(),
        SearchSpec::AstarDaDebug
    );
}

#[test]
fn parses_top_level_da_debug_with_or_without_unit_parens() {
    assert_eq!(parse_search_spec("da_debug").unwrap(), SearchSpec::DaDebug);
    assert_eq!(parse_search_spec("da_debug()").unwrap(), SearchSpec::DaDebug);
}

#[test]
fn parses_top_level_astar_da_debug_with_or_without_unit_parens() {
    assert_eq!(
        parse_search_spec("astar_da_debug").unwrap(),
        SearchSpec::AstarDaDebug
    );
    assert_eq!(
        parse_search_spec("astar_da_debug()").unwrap(),
        SearchSpec::AstarDaDebug
    );
}

#[test]
fn errors_are_human_readable() {
    let err = parse_search_spec("astar(").unwrap_err();
    assert!(err.to_lowercase().contains("invalid"));
}
