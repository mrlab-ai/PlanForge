use super::*;

#[test]
fn test_literal_positive() {
    let lit = Literal::new("pred".to_string(), vec!["a".to_string()], true);
    let pos = lit.positive();
    assert!(!pos.negated);
    assert_eq!(pos.predicate, "pred");
}

#[test]
fn test_literal_negate() {
    let lit = Literal::new("pred".to_string(), vec!["a".to_string()], false);
    let neg = lit.negate();
    assert!(neg.negated);
}

#[test]
fn test_simplify_removes_duplicates() {
    let axiom = PropositionalAxiom::new(
        "test".to_string(),
        vec![
            Literal::new("p".to_string(), vec![], false),
            Literal::new("p".to_string(), vec![], false), // duplicate
        ],
        Literal::new("q".to_string(), vec![], false),
    );
    let simplified = simplify(&[axiom]);
    assert_eq!(simplified.len(), 1);
    assert_eq!(simplified[0].condition.len(), 1);
}

#[test]
fn test_empty_condition_dominates() {
    let axiom1 = PropositionalAxiom::new(
        "test".to_string(),
        vec![],
        Literal::new("q".to_string(), vec![], false),
    );
    let axiom2 = PropositionalAxiom::new(
        "test".to_string(),
        vec![Literal::new("p".to_string(), vec![], false)],
        Literal::new("q".to_string(), vec![], false),
    );
    let simplified = simplify(&[axiom1, axiom2]);
    assert_eq!(simplified.len(), 1);
    assert!(simplified[0].condition.is_empty());
}
