use std::sync::Arc;

use planforge_sas::numeric_task::{NumericRootTask, TaskRef};
use planforge_search::config::parse_call;
use planforge_search::heuristic_factory::{
    BuildContext, build_heuristic_from_spec, entry, register_external_heuristics,
};

#[test]
fn external_registration_after_heuristic_construction_is_rejected() {
    let spec = parse_call("blind()").unwrap();
    let task: TaskRef<'static> = Arc::new(NumericRootTask::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tests/assets/numeric_sas/example2.sas"
    )));
    build_heuristic_from_spec(&spec, &BuildContext::new(&task)).unwrap();

    let error = register_external_heuristics(vec![
        entry::<planforge_search::evaluation::heuristic::BlindFeature>().with_name("too_late"),
    ])
    .unwrap_err();
    assert!(
        error.contains("already been registered or used"),
        "got `{error}`"
    );
}
