use std::path::PathBuf;

use planforge_translate::{
    GroundingLimitError, GroundingLimitKind, GroundingLimits, LayerStrategy,
    translate_to_sas_to_path_with_limits,
};

#[test]
fn atom_limit_is_typed_and_never_writes_a_partial_sas_task() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/assets/strips-pddl-files/blocks-4-0");
    let domain = fixture.join("domain.pddl");
    let problem = fixture.join("probBLOCKS-4-0.pddl");
    let output = std::env::temp_dir().join(format!(
        "planforge-grounding-limit-test-{}.sas",
        std::process::id()
    ));
    if output.exists() {
        std::fs::remove_file(&output).expect("remove stale test output");
    }

    let error = translate_to_sas_to_path_with_limits(
        domain.to_str().expect("fixture path is UTF-8"),
        problem.to_str().expect("fixture path is UTF-8"),
        &output,
        LayerStrategy::default(),
        GroundingLimits {
            max_ground_actions: u64::MAX,
            max_ground_atoms: 0,
            max_grounding_memory: u64::MAX,
        },
    )
    .expect_err("the zero atom limit must stop grounding");

    let limit = error
        .downcast_ref::<GroundingLimitError>()
        .expect("the anyhow error must preserve the typed grounding error");
    assert_eq!(limit.kind, GroundingLimitKind::Atoms);
    assert_eq!(limit.value, 1);
    assert_eq!(limit.limit, 0);
    assert_eq!(limit.phase, "building the grounding model");
    assert_eq!(
        limit.to_string(),
        "grounding exceeded the atom limit: 1 ground atom (limit 0) while building the grounding \
         model; the task is likely too large to ground. Raise --max-ground-atoms to continue, or \
         use a smaller instance."
    );
    assert!(
        !output.exists(),
        "a failed grounding run must not create a SAS task"
    );
}
