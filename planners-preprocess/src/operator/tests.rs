use super::Operator;
use crate::helper_functions::InputStream;
use crate::variable::{NumericVariable, Variable};

#[test]
fn from_stream_preserves_conditional_numeric_effects() {
    let mut variable_stream = InputStream::new(
        "begin_variable\nv0\n-1\n2\na\nb\nend_variable\n\
begin_variable\nv1\n-1\n2\nc\nd\nend_variable\n"
            .to_string(),
    );
    let mut variables_storage = [
        Variable::from_stream(&mut variable_stream),
        Variable::from_stream(&mut variable_stream),
    ];
    let variables = variables_storage
        .iter_mut()
        .map(|var| var as *mut Variable)
        .collect::<Vec<_>>();

    let mut numeric_stream = InputStream::new("R -1 n0\nR -1 n1\n".to_string());
    let mut numeric_storage = [
        NumericVariable::from_stream(&mut numeric_stream),
        NumericVariable::from_stream(&mut numeric_stream),
    ];
    let numeric_variables = numeric_storage
        .iter_mut()
        .map(|var| var as *mut NumericVariable)
        .collect::<Vec<_>>();

    let input = "begin_operator\nop\n0\n0\n1\n1 1 0 0 + 1\n0\nend_operator\n".to_string();
    let mut stream = InputStream::new(input);

    let op = Operator::from_stream(&mut stream, &variables, &numeric_variables);
    let num_eff = &op.get_num_eff()[0];

    assert!(!num_eff.is_conditional_effect);
    assert!(num_eff.effect_conds.is_empty());
}
