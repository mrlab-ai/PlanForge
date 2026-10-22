use std::sync::Arc;

use planforge_sas::numeric_task::{NumericRootTask, TaskRef};
use planforge_search::config::parse_call;
use planforge_search::evaluation::Heuristic;
use planforge_search::evaluation::heuristic::BlindHeuristic;
use planforge_search::heuristic_factory::{
    BuildContext, EmptyConfig, HeuristicBuildError, HeuristicConfig, HeuristicFeature, entry,
    register_external_heuristics,
};

#[test]
fn external_registration_after_heuristic_construction_is_rejected() {
    let spec = parse_call("blind()").unwrap();
    let task: TaskRef<'static> = Arc::new(NumericRootTask::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/assets/numeric_sas/example2.sas"
    )));
    HeuristicConfig::from_spec(&spec)
        .unwrap()
        .build(&BuildContext::new(&task))
        .unwrap();

    let error = register_external_heuristics(vec![entry::<TooLate>()]).unwrap_err();
    assert!(
        error.contains("already been registered or used"),
        "got `{error}`"
    );
}

struct TooLate;
impl HeuristicFeature for TooLate {
    const NAME: &'static str = "too_late";
    type Config = EmptyConfig;
    fn build<'a>(
        _: &Self::Config,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        Ok(Box::new(BlindHeuristic::for_task(&**context.task)))
    }
}
