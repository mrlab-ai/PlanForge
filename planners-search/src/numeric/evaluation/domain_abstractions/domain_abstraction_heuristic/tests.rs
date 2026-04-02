use super::*;

#[test]
fn comparison_projection_uses_concrete_value_mapping() {
    let mapping = vec![vec![0, 1, 2]];

    let abs_val = abstract_propositional_value(0, 1, &mapping).unwrap();

    assert_eq!(abs_val, 1);
}
