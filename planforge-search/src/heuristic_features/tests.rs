use super::{
    BuildContext, HeuristicBuildError, HeuristicFeature, TaskCheck, TaskRequirements, entry,
};
use crate::config::EmptyConfig;
use crate::evaluation::Heuristic;
use crate::evaluation::{BlindFeature, BlindHeuristic};
use crate::registry::Features;
use planforge_sas::numeric_task::Task;

#[test]
fn a_registry_rejects_a_second_entry_of_a_name() {
    let mut features = Features::builtin();
    let error = features
        .heuristics
        .add(entry::<BlindFeature>())
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("heuristic named `blind` already"),
        "got `{error}`"
    );
    features.heuristics.add(entry::<Duplicate>()).unwrap();
    assert!(features.heuristics.add(entry::<Duplicate>()).is_err());
}

#[test]
fn two_feature_sets_parse_side_by_side() {
    let builtin = Features::builtin();
    let mut extended = Features::builtin();
    extended.heuristics.add(entry::<Duplicate>()).unwrap();

    let search = extended.parse_search("astar(duplicate())").unwrap();
    assert_eq!(search.to_string(), "astar(duplicate())");
    let error = builtin.parse_search("astar(duplicate())").unwrap_err();
    assert!(
        error.to_string().contains("unknown heuristic `duplicate`"),
        "got `{error}`"
    );
    assert_eq!(
        builtin.parse_search("astar(blind())").unwrap(),
        extended.parse_search("astar(blind())").unwrap()
    );
}

struct Duplicate;
impl HeuristicFeature for Duplicate {
    const NAME: &'static str = "duplicate";
    type Config = EmptyConfig;
    fn build(
        _: &Self::Config,
        context: &BuildContext,
    ) -> Result<Box<dyn Heuristic>, HeuristicBuildError> {
        Ok(Box::new(BlindHeuristic::for_task(&context.task)))
    }
}

#[test]
fn a_union_of_requirements_keeps_each_check_and_transformation_once() {
    fn passes(_: &Task) -> Result<(), HeuristicBuildError> {
        Ok(())
    }
    let some = TaskRequirements {
        transformations: vec!["first"],
        checks: vec![TaskCheck {
            name: "property",
            check: passes,
        }],
        keep_unimportant_variables: false,
    };
    let other = TaskRequirements {
        transformations: vec!["second"],
        checks: Vec::new(),
        keep_unimportant_variables: true,
    };
    let union = some.clone().union(some).union(other);
    assert_eq!(union.transformations, ["first", "second"]);
    assert_eq!(union.checks.len(), 1);
    assert!(union.requires("property") && !union.requires("other"));
    assert!(union.keep_unimportant_variables);
}
