use super::Operator;
use crate::sas_tasks::SASOperator;

fn operator_with_assign_effect(assign_effects: Vec<crate::sas_tasks::AssignEffect>) -> SASOperator {
    SASOperator::new(
        "(op)".to_string(),
        Vec::new(),
        Vec::new(),
        assign_effects,
        1.0,
    )
}

/// `var0 += var1 if var1 == 0`. The preprocessor used to read the condition and
/// then drop it, turning a guarded numeric effect into an unconditional one.
#[test]
fn from_sas_preserves_conditional_numeric_effects() {
    let sas_op = operator_with_assign_effect(vec![(0, "+".to_string(), 1, vec![(1, 0)])]);

    let op = Operator::from_sas(&sas_op);

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
fn from_sas_keeps_unconditional_numeric_effects_unconditional() {
    let sas_op = operator_with_assign_effect(vec![(0, "+".to_string(), 1, Vec::new())]);

    let op = Operator::from_sas(&sas_op);

    let num_eff = &op.get_num_eff()[0];
    assert!(!num_eff.is_conditional_effect);
    assert!(num_eff.effect_conds.is_empty());
    assert_eq!(num_eff.var, 0);
    assert_eq!(num_eff.foperand, 1);
}

/// The SAS file spells operator names without the PDDL parentheses, and the
/// causal graph reports on operators by that name.
#[test]
fn from_sas_strips_the_pddl_parentheses_from_the_name() {
    let sas_op = operator_with_assign_effect(Vec::new());

    let op = Operator::from_sas(&sas_op);

    assert_eq!(op.get_name(), "op");
}

/// An unknown assignment operator must fail loudly rather than being folded
/// into a default one, which would silently rewrite the effect.
#[test]
#[should_panic(expected = "Unknown assignment operator")]
fn from_sas_rejects_an_unknown_assignment_operator() {
    let sas_op = operator_with_assign_effect(vec![(0, "^".to_string(), 1, Vec::new())]);

    Operator::from_sas(&sas_op);
}
