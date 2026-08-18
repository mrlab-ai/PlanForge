use std::sync::Arc;

use planforge_sas::numeric_task::{NumericRootTask, TaskRef};
use planforge_search::config::parse_heuristic_spec;
use planforge_search::heuristic_factory::{
    ExternalHeuristic, RequiredBackend, TaskRequirements, build_heuristic_from_spec,
    register_external_heuristics,
};

#[test]
fn external_registration_after_heuristic_construction_is_rejected() {
    let spec = parse_heuristic_spec("blind()").unwrap();
    let task: TaskRef<'static> = Arc::new(NumericRootTask::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/assets/numeric_sas/example2.sas"
    )));
    build_heuristic_from_spec(&spec, &*task, task.clone()).unwrap();

    let error = register_external_heuristics(vec![ExternalHeuristic {
        name: "too_late",
        backend: RequiredBackend::None,
        requirements: |_| Ok(TaskRequirements::ANY),
        nested_heuristics: |_| Ok(Vec::new()),
        build: |_, _, _| Ok(None),
    }])
    .unwrap_err();
    assert!(
        error.contains("already been registered or used"),
        "got `{error}`"
    );
}
