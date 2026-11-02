use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use planforge_sas::numeric_task::Task;
use planforge_search::config::parse_call;
use planforge_search::evaluation::Heuristic;
use planforge_search::heuristic_factory::{
    BuildContext, EmptyConfig, HEURISTICS, HeuristicBuildError, HeuristicConfig, HeuristicFeature,
    entry,
};

static BUILD_CALLED: AtomicBool = AtomicBool::new(false);

struct ExternalTest;
impl HeuristicFeature for ExternalTest {
    const NAME: &'static str = "external_test";
    type Config = EmptyConfig;
    fn build<'a>(
        _: &Self::Config,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        BUILD_CALLED.store(true, Ordering::SeqCst);
        Ok(Box::new(
            planforge_search::evaluation::BlindHeuristic::for_task(context.task),
        ))
    }
}
struct ExternalCplexTest;
impl HeuristicFeature for ExternalCplexTest {
    const NAME: &'static str = "external_cplex_test";
    type Config = EmptyConfig;
    fn preflight(_: &Self::Config) -> std::io::Result<()> {
        planforge_heuristics::requirements::require_cplex()
    }
    fn build<'a>(
        config: &Self::Config,
        context: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        ExternalTest::build(config, context)
    }
}

#[test]
fn external_heuristic_is_visible_to_every_registry_path() {
    let mut heuristics = planforge_heuristics::heuristics();
    heuristics.extend([entry::<ExternalTest>(), entry::<ExternalCplexTest>()]);
    HEURISTICS.register(heuristics).unwrap();

    assert!(HEURISTICS.names().any(|name| name == "external_test"));

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
    let task: Arc<Task> = Arc::new(Task::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/assets/numeric_sas/example2.sas"
    )));
    let heuristic = external.build(&BuildContext::new(&task)).unwrap();
    assert_eq!(heuristic.name(), "blind_heuristic");
    assert!(BUILD_CALLED.load(Ordering::SeqCst));

    let error = config("unknown()").unwrap_err().to_string();
    assert!(error.contains("blind"), "got `{error}`");
    assert!(error.contains("external_test"), "got `{error}`");

    let error = HEURISTICS.register(Vec::new()).unwrap_err();
    assert!(error.contains("already been registered"), "got `{error}`");
}
