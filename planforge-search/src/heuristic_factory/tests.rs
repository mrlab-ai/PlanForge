use super::{BuildContext, EmptyConfig, HeuristicBuildError, HeuristicFeature, entry};
use super::{heuristic_names, register_external_heuristics};
use crate::evaluation::Heuristic;
use crate::evaluation::heuristic::{BlindFeature, BlindHeuristic};

#[test]
fn heuristic_registry_names_are_unique_and_complete() {
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

#[test]
fn external_registration_rejects_builtin_name_collision() {
    let error = register_external_heuristics(vec![entry::<BlindFeature>()]).unwrap_err();
    assert!(error.contains("collides with a built-in"), "got `{error}`");
}

#[test]
fn external_registration_rejects_duplicate_external_names() {
    let error =
        register_external_heuristics(vec![entry::<Duplicate>(), entry::<Duplicate>()]).unwrap_err();
    assert!(error.contains("registered more than once"), "got `{error}`");
}

struct Duplicate;
impl HeuristicFeature for Duplicate {
    const NAME: &'static str = "duplicate";
    type Config = EmptyConfig;
    fn build<'a>(
        _: &Self::Config,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        Ok(Box::new(BlindHeuristic::for_task(context.task)))
    }
}
