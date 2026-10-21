use super::*;

#[test]
fn uniform_reduction_rejects_significant_underflow() {
    let mut remaining = RemainingCosts::from_operator_costs(&[1.0]);

    let error = remaining.reduce_operator_costs_uniform(&[1.5]).unwrap_err();
    assert!(format!("{error:#}").contains("underflow"));
}

#[test]
fn uniform_reduction_rejects_non_finite_saturated_costs() {
    let mut remaining = RemainingCosts::from_operator_costs(&[1.0]);

    let error = remaining
        .reduce_operator_costs_uniform(&[f64::NEG_INFINITY])
        .unwrap_err();
    assert!(error.to_string().contains("invalid saturated cost"));
}

#[test]
fn uniform_reduction_clamps_tiny_negative_roundoff() {
    let mut remaining = RemainingCosts::from_operator_costs(&[1.0]);

    remaining
        .reduce_operator_costs_uniform(&[1.0 + 1e-12])
        .unwrap();
    assert_eq!(remaining.operator_costs_for_label_cp(), vec![0.0]);
}
