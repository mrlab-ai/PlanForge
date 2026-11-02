use super::HEURISTICS;
use super::{
    BuildContext, EmptyConfig, HeuristicBuildError, HeuristicFeature, TaskCheck, TaskRequirements,
    entry,
};
use crate::evaluation::Heuristic;
use crate::evaluation::{BlindFeature, BlindHeuristic};
use planforge_sas::numeric_task::Task;

#[test]
fn registration_rejects_builtin_name_collision() {
    let error = HEURISTICS
        .register(vec![entry::<BlindFeature>()])
        .unwrap_err();
    assert!(error.contains("collides with a built-in"), "got `{error}`");
}

#[test]
fn registration_rejects_duplicate_names() {
    let error = HEURISTICS
        .register(vec![entry::<Duplicate>(), entry::<Duplicate>()])
        .unwrap_err();
    assert!(error.contains("registered more than once"), "got `{error}`");
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
    };
    let other = TaskRequirements {
        transformations: vec!["second"],
        checks: Vec::new(),
    };
    let union = some.clone().union(some).union(other);
    assert_eq!(union.transformations, ["first", "second"]);
    assert_eq!(union.checks.len(), 1);
    assert!(union.requires("property") && !union.requires("other"));
}
