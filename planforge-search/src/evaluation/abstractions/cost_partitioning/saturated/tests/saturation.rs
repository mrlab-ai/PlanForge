use super::*;

#[test]
fn reduce_costs_rejects_significant_underflow() {
    let mut remaining = vec![1.0];
    let saturated = vec![1.5];

    let err = reduce_costs(&mut remaining, &saturated).unwrap_err();
    assert!(format!("{err}").contains("underflow"));
}

#[test]
fn reduce_costs_rejects_non_finite_saturated_costs() {
    let mut remaining = vec![1.0];

    let error = reduce_costs(&mut remaining, &[f64::NEG_INFINITY]).unwrap_err();

    assert!(error.to_string().contains("must be finite"));
}

#[test]
fn reduce_costs_clamps_tiny_negative_roundoff() {
    let mut remaining = vec![1.0];
    let saturated = vec![1.0 + 1e-12];

    reduce_costs(&mut remaining, &saturated).unwrap();
    assert_eq!(remaining, vec![0.0]);
}
