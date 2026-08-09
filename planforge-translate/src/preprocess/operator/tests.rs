use super::super::helper_functions::InputStream;
use super::Operator;

/// `1 1 0 0 + 1` is one assignment effect guarded by one condition: if var1 has
/// value 0 then `var0 += var1`. The preprocessor used to read the condition and
/// then drop it, turning a guarded numeric effect into an unconditional one.
#[test]
fn from_stream_preserves_conditional_numeric_effects() {
    let input = "begin_operator\nop\n0\n0\n1\n1 1 0 0 + 1\n0\nend_operator\n".to_string();
    let mut stream = InputStream::new(input);

    let op = Operator::from_stream(&mut stream);

    let num_eff = &op.get_num_eff()[0];
    assert!(num_eff.is_conditional_effect);
    assert_eq!(num_eff.effect_conds.len(), 1);
    assert_eq!(num_eff.effect_conds[0].var, 1);
    assert_eq!(num_eff.effect_conds[0].cond, 0);
    assert_eq!(num_eff.var, 0);
    assert_eq!(num_eff.foperand, 1);
}

/// An unconditional assignment effect must stay unconditional.
#[test]
fn from_stream_keeps_unconditional_numeric_effects_unconditional() {
    let input = "begin_operator\nop\n0\n0\n1\n0 0 + 1\n0\nend_operator\n".to_string();
    let mut stream = InputStream::new(input);

    let op = Operator::from_stream(&mut stream);

    let num_eff = &op.get_num_eff()[0];
    assert!(!num_eff.is_conditional_effect);
    assert!(num_eff.effect_conds.is_empty());
    assert_eq!(num_eff.var, 0);
    assert_eq!(num_eff.foperand, 1);
}

/// A malformed cost must fail loudly rather than silently becoming zero, which
/// would make the operator free and break optimality downstream.
#[test]
#[should_panic(expected = "malformed cost")]
fn from_stream_rejects_a_malformed_cost() {
    let input = "begin_operator\nop\n0\n0\n0\nnot-a-number\nend_operator\n".to_string();
    let mut stream = InputStream::new(input);

    Operator::from_stream(&mut stream);
}
