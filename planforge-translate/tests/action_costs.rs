//! Action costs as PDDL and numeric Fast Downward define them: under a metric
//! an action without a cost effect costs nothing; without a metric every
//! action costs one.

use planforge_sas::numeric_task::metric_operator_costs;

const DOMAIN: &str = "(define (domain errands)
  (:requirements :strips :action-costs)
  (:predicates (paid) (walked))
  (:functions (total-cost))
  (:action pay
    :parameters ()
    :precondition ()
    :effect (and (paid) (increase (total-cost) 5)))
  (:action walk
    :parameters ()
    :precondition ()
    :effect (walked)))";

const PROBLEM: &str = "(define (problem errands-1)
  (:domain errands)
  (:init (= (total-cost) 0))
  (:goal (and (paid) (walked)))
  (:metric minimize (total-cost)))";

fn operator_costs(name: &str, problem_text: &str) -> Vec<(String, f64)> {
    let dir = std::env::temp_dir().join(format!(
        "planforge-action-costs-{name}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create fixture directory");
    let domain = dir.join("domain.pddl");
    let problem = dir.join("problem.pddl");
    std::fs::write(&domain, DOMAIN).expect("write domain");
    std::fs::write(&problem, problem_text).expect("write problem");
    let task = planforge_translate::translate_to_task(
        domain.to_str().expect("UTF-8 path"),
        problem.to_str().expect("UTF-8 path"),
    )
    .expect("translation failed");
    std::fs::remove_dir_all(&dir).expect("remove fixture");

    let mut costs: Vec<(String, f64)> = task
        .operators()
        .iter()
        .map(|operator| operator.name().to_string())
        .zip(metric_operator_costs(&task))
        .collect();
    costs.sort_by(|a, b| a.0.cmp(&b.0));
    costs
}

#[test]
fn an_action_without_cost_effect_is_free_under_a_metric() {
    assert_eq!(
        operator_costs("metric", PROBLEM),
        [("pay".to_string(), 5.0), ("walk".to_string(), 0.0)]
    );
}

#[test]
fn every_action_costs_one_without_a_metric() {
    let problem = PROBLEM.replace("(:metric minimize (total-cost))", "");
    assert_eq!(
        operator_costs("no-metric", &problem),
        [("pay".to_string(), 1.0), ("walk".to_string(), 1.0)]
    );
}
