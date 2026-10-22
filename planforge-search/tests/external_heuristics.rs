use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use planforge_sas::numeric_task::{NumericRootTask, TaskRef};
use planforge_search::config::parse_call;
use planforge_search::evaluation::Heuristic;
use planforge_search::heuristic_factory::{
    BuildContext, EmptyConfig, HeuristicBuildError, HeuristicConfig, HeuristicFeature,
    RequiredBackend, entry, heuristic_names, register_external_heuristics,
};

static BUILD_CALLED: AtomicBool = AtomicBool::new(false);

struct ExternalTest;
impl HeuristicFeature for ExternalTest {
    const NAME: &'static str = "external_test";
    type Config = EmptyConfig;
    fn build<'a>(
        _: &Self::Config,
        _: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        BUILD_CALLED.store(true, Ordering::SeqCst);
        Ok(Box::new(
            planforge_search::evaluation::heuristic::BlindHeuristic::new(None),
        ))
    }
}
struct ExternalCplexTest;
impl HeuristicFeature for ExternalCplexTest {
    const NAME: &'static str = "external_cplex_test";
    const BACKEND: RequiredBackend = RequiredBackend::Cplex;
    type Config = EmptyConfig;
    fn build<'a>(
        config: &Self::Config,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        ExternalTest::build(config, context)
    }
}

#[test]
fn external_heuristic_is_visible_to_every_registry_path() {
    register_external_heuristics(vec![entry::<ExternalTest>(), entry::<ExternalCplexTest>()])
        .unwrap();

    assert!(heuristic_names().any(|name| name == "external_test"));

    let config = |text: &str| HeuristicConfig::from_spec(&parse_call(text).unwrap());
    config("check_admissible(external_test())").unwrap();

    let external = config("external_test()").unwrap();
    external.preflight().unwrap();
    #[cfg(not(feature = "cplex"))]
    {
        let error = config("check_admissible(external_cplex_test())")
            .unwrap()
            .preflight()
            .unwrap_err();
        assert!(error.to_string().contains("requires unrestricted CPLEX"));
    }
    let task: TaskRef<'static> = Arc::new(NumericRootTask::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/assets/numeric_sas/example2.sas"
    )));
    let heuristic = external.build(&BuildContext::new(&task)).unwrap();
    assert_eq!(heuristic.heuristic_name(), "blind_heuristic");
    assert!(BUILD_CALLED.load(Ordering::SeqCst));

    let error = config("unknown()").unwrap_err().to_string();
    assert!(error.contains("blind"), "got `{error}`");
    assert!(error.contains("external_test"), "got `{error}`");

    let error = register_external_heuristics(Vec::new()).unwrap_err();
    assert!(error.contains("already been registered"), "got `{error}`");
}
