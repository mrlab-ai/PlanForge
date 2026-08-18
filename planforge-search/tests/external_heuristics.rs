use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use planforge_sas::numeric_task::{AbstractNumericTask, NumericRootTask, TaskRef};
use planforge_search::config::{HeuristicSpec, parse_heuristic_spec};
use planforge_search::evaluation::Heuristic;
use planforge_search::heuristic_factory::{
    ExternalHeuristic, HeuristicBuildError, RequiredBackend, TaskRequirements,
    build_heuristic_from_spec, heuristic_names, preflight_required_backends,
    register_external_heuristics, validate_heuristic_spec,
};

static BUILD_CALLED: AtomicBool = AtomicBool::new(false);

fn any_task(_: &HeuristicSpec) -> Result<TaskRequirements, String> {
    Ok(TaskRequirements::ANY)
}

fn no_nested_heuristics(_: &HeuristicSpec) -> Result<Vec<HeuristicSpec>, String> {
    Ok(Vec::new())
}

fn build_test_heuristic<'a>(
    spec: &HeuristicSpec,
    _: &'a dyn AbstractNumericTask,
    _: TaskRef<'a>,
) -> Result<Option<Box<dyn Heuristic + 'a>>, HeuristicBuildError> {
    if !spec.args.is_empty() {
        return Err("`external_test` does not accept arguments"
            .to_string()
            .into());
    }
    BUILD_CALLED.store(true, Ordering::SeqCst);
    Ok(None)
}

#[test]
fn external_heuristic_is_visible_to_every_registry_path() {
    register_external_heuristics(vec![
        ExternalHeuristic {
            name: "external_test",
            backend: RequiredBackend::None,
            requirements: any_task,
            nested_heuristics: no_nested_heuristics,
            build: build_test_heuristic,
        },
        ExternalHeuristic {
            name: "external_cplex_test",
            backend: RequiredBackend::Cplex,
            requirements: any_task,
            nested_heuristics: no_nested_heuristics,
            build: build_test_heuristic,
        },
    ])
    .unwrap();

    assert!(heuristic_names().any(|name| name == "external_test"));

    let nested = parse_heuristic_spec("check_admissible(external_test())").unwrap();
    validate_heuristic_spec(&nested).unwrap();

    let spec = parse_heuristic_spec("external_test()").unwrap();
    preflight_required_backends(&spec).unwrap();
    #[cfg(not(feature = "cplex"))]
    {
        let cplex_spec = parse_heuristic_spec("external_cplex_test()").unwrap();
        let error = preflight_required_backends(&cplex_spec).unwrap_err();
        assert!(error.to_string().contains("requires unrestricted CPLEX"));
    }
    let task: TaskRef<'static> = Arc::new(NumericRootTask::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/assets/numeric_sas/example2.sas"
    )));
    let heuristic = build_heuristic_from_spec(&spec, &*task, task.clone()).unwrap();
    assert!(heuristic.is_none());
    assert!(BUILD_CALLED.load(Ordering::SeqCst));

    let unknown = parse_heuristic_spec("unknown()").unwrap();
    let error = validate_heuristic_spec(&unknown).unwrap_err();
    assert!(error.contains("blind"), "got `{error}`");
    assert!(error.contains("external_test"), "got `{error}`");

    let error = register_external_heuristics(Vec::new()).unwrap_err();
    assert!(error.contains("already been registered"), "got `{error}`");
}
