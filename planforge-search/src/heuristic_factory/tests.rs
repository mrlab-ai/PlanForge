use super::HEURISTICS;
use super::{BuildContext, EmptyConfig, HeuristicBuildError, HeuristicFeature, entry};
use crate::evaluation::Heuristic;
use crate::evaluation::heuristic::{BlindFeature, BlindHeuristic};

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
    fn build<'a>(
        _: &Self::Config,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        Ok(Box::new(BlindHeuristic::for_task(context.task)))
    }
}
