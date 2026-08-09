//! The translator must be a function of its input alone.
//!
//! Every `HashMap` in the pipeline gets its own hash seed, so two translations
//! in the same process already visit the unordered containers in two different
//! orders. Any output that depends on such an order shows up here as a
//! difference between the runs.

use std::path::PathBuf;

fn fixture(domain_dir: &str, problem: &str) -> (String, String) {
    let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/assets/numeric-pddl-files")
        .join(domain_dir);
    let domain = assets.join("domain.pddl");
    let problem = assets.join(problem);
    assert!(domain.is_file(), "missing fixture {}", domain.display());
    assert!(problem.is_file(), "missing fixture {}", problem.display());
    (
        domain.to_str().expect("fixture path is UTF-8").to_owned(),
        problem.to_str().expect("fixture path is UTF-8").to_owned(),
    )
}

fn assert_translation_is_reproducible(domain_dir: &str, problem: &str) {
    let (domain, problem) = fixture(domain_dir, problem);
    let first = planforge_translator::translate_to_sas_string(&domain, &problem)
        .expect("translation failed");
    for run in 2..=4 {
        let again = planforge_translator::translate_to_sas_string(&domain, &problem)
            .expect("translation failed");
        assert_eq!(
            first, again,
            "translating {domain_dir} is not reproducible: run 1 and run {run} differ"
        );
    }
}

#[test]
fn translating_plant_watering_twice_gives_the_same_sas() {
    assert_translation_is_reproducible("plant-watering", "prob_4_1_1.pddl");
}

#[test]
fn translating_satellite_twice_gives_the_same_sas() {
    assert_translation_is_reproducible("satellite", "pfile1.pddl");
}

#[test]
fn translating_sailing_twice_gives_the_same_sas() {
    assert_translation_is_reproducible("sailing", "prob_1_1_1229.pddl");
}
