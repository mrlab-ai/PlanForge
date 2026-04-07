use super::*;

#[test]
fn reverse_level_reverses_order() {
    let mut ids = vec![3, 1, 2];
    order_variable_ids(&mut ids, GreedyVariableOrderType::ReverseLevel, 0);
    assert_eq!(ids, vec![3, 2, 1]);
}

#[test]
fn random_order_is_deterministic_for_seed() {
    let mut lhs = vec![0, 1, 2, 3, 4];
    let mut rhs = vec![0, 1, 2, 3, 4];
    order_variable_ids(&mut lhs, GreedyVariableOrderType::Random, 7);
    order_variable_ids(&mut rhs, GreedyVariableOrderType::Random, 7);
    assert_eq!(lhs, rhs);
}
