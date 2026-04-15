use super::*;
use planners_sas::numeric::numeric_task::{
    ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
};

fn simple_var(name: &str, values: &[&str], axiom_layer: Option<usize>) -> ExplicitVariable {
    ExplicitVariable::new(
        values.len(),
        name.to_string(),
        values.iter().map(|value| value.to_string()).collect(),
        axiom_layer,
        0,
    )
}

#[test]
fn lmcutnumeric_config_defaults_match_fd_parser_defaults() {
    let config = LmCutNumericConfig::default();
    assert!(!config.ceiling_less_than_one);
    assert!(!config.ignore_numeric);
    assert!(!config.random_pcf);
    assert!(!config.irmax);
    assert!(!config.disable_ma);
    assert!(!config.use_second_order_simple);
    assert!(!config.use_constant_assignment);
    assert_eq!(config.bound_iterations, 0);
    assert_eq!(config.precision, 0.000001);
    assert_eq!(config.epsilon, 0.0);
}

#[test]
fn from_config_accepts_second_order_simple_flag() {
    let task = NumericRootTask::new(
        3,
        Metric::new(true, None),
        vec![simple_var("v0", &["zero", "one"], None)],
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = LmCutNumericConfig {
        use_second_order_simple: true,
        ..Default::default()
    };

    let heuristic = LandmarkCutNumericHeuristic::from_config(&task, config)
        .expect("supported SOSE flag should construct the heuristic");
    assert!(heuristic.config().use_second_order_simple);
}

#[test]
fn from_config_accepts_irmax() {
    let task = NumericRootTask::new(
        3,
        Metric::new(true, None),
        vec![simple_var("v0", &["zero", "one"], None)],
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = LmCutNumericConfig {
        irmax: true,
        ..Default::default()
    };

    let heuristic = LandmarkCutNumericHeuristic::from_config(&task, config)
        .expect("irmax flag should construct the heuristic");

    assert!(heuristic.config().irmax);
}

#[test]
fn from_config_rejects_unimplemented_random_pcf() {
    let task = NumericRootTask::new(
        3,
        Metric::new(true, None),
        vec![simple_var("v0", &["zero", "one"], None)],
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = LmCutNumericConfig {
        random_pcf: true,
        ..Default::default()
    };

    let error = match LandmarkCutNumericHeuristic::from_config(&task, config) {
        Err(error) => error,
        Ok(_) => panic!("random_pcf should be rejected until it is implemented"),
    };

    assert!(error.contains("random_pcf=true"));
}

#[test]
fn from_config_accepts_disable_ma() {
    let task = NumericRootTask::new(
        3,
        Metric::new(true, None),
        vec![simple_var("v0", &["zero", "one"], None)],
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = LmCutNumericConfig {
        disable_ma: true,
        ..Default::default()
    };

    let heuristic = LandmarkCutNumericHeuristic::from_config(&task, config)
        .expect("disable_ma flag should construct the heuristic");

    assert!(heuristic.config().disable_ma);
}

#[test]
fn from_config_accepts_constant_assignment() {
    let task = NumericRootTask::new(
        3,
        Metric::new(true, None),
        vec![simple_var("v0", &["zero", "one"], None)],
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = LmCutNumericConfig {
        use_constant_assignment: true,
        ..Default::default()
    };

    let heuristic = LandmarkCutNumericHeuristic::from_config(&task, config)
        .expect("constant assignment flag should construct the heuristic");

    assert!(heuristic.config().use_constant_assignment);
}

#[test]
fn from_config_accepts_bound_iterations() {
    let task = NumericRootTask::new(
        3,
        Metric::new(true, None),
        vec![simple_var("v0", &["zero", "one"], None)],
        vec![],
        vec![ExplicitFact::new(0, 1)],
        vec![],
        vec![0],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        ExplicitFact::new(0, 0),
    );
    let config = LmCutNumericConfig {
        bound_iterations: 1,
        ..Default::default()
    };

    let heuristic = LandmarkCutNumericHeuristic::from_config(&task, config)
        .expect("bound iterations should construct the heuristic once bounds are implemented");

    assert_eq!(heuristic.config().bound_iterations, 1);
}
