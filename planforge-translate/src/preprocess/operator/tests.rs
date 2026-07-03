use super::super::helper_functions::InputStream;
use super::Operator;

#[test]
fn from_stream_preserves_conditional_numeric_effects() {
    let input = "begin_operator\nop\n0\n0\n1\n1 1 0 0 + 1\n0\nend_operator\n".to_string();
    let mut stream = InputStream::new(input);

    let op = Operator::from_stream(&mut stream);
    let num_eff = &op.get_num_eff()[0];

    assert!(!num_eff.is_conditional_effect);
    assert!(num_eff.effect_conds.is_empty());
}
