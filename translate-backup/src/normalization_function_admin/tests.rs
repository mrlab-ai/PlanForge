use super::*;

#[test]
fn test_constant() {
    let mut admin = NormalizationFunctionAdministrator::new();
    let constant = FunctionalExpression::Constant(NumericConstant::new(42.0));
    let pne = admin.get_derived_function(&constant);

    assert!(pne.symbol.contains("derived!"));
    assert!(pne.symbol.contains("42"));
    assert_eq!(pne.args.len(), 0);
    assert_eq!(admin.get_all_axioms().len(), 1);
}

#[test]
fn test_primitive_passthrough() {
    let mut admin = NormalizationFunctionAdministrator::new();
    let pne = PrimitiveNumericExpression::new("fuel".to_string(), vec!["?v".to_string()], 'S');
    let result = admin.get_derived_function(&FunctionalExpression::Primitive(pne.clone()));

    assert_eq!(result.symbol, "fuel");
    assert_eq!(result.args, vec!["?v"]);
    assert_eq!(admin.get_all_axioms().len(), 0); // No axiom created
}

#[test]
fn test_additive_inverse() {
    let mut admin = NormalizationFunctionAdministrator::new();
    let pne = PrimitiveNumericExpression::new("fuel".to_string(), vec!["?v".to_string()], 'S');
    let inv = AdditiveInverse::new(FunctionalExpression::Primitive(pne));
    let result = admin.get_derived_function(&FunctionalExpression::AdditiveInverse(inv));

    assert!(result.symbol.contains("derived!"));
    assert!(result.symbol.contains("difference"));
    assert_eq!(result.args, vec!["?v"]);
    assert_eq!(admin.get_all_axioms().len(), 1);
}

#[test]
fn test_arithmetic_sum() {
    let mut admin = NormalizationFunctionAdministrator::new();
    let pne1 = PrimitiveNumericExpression::new("fuel".to_string(), vec![], 'S');
    let pne2 = PrimitiveNumericExpression::new("distance".to_string(), vec![], 'S');

    let sum = ArithmeticExpression::new(
        "+".to_string(),
        vec![
            FunctionalExpression::Primitive(pne1),
            FunctionalExpression::Primitive(pne2),
        ],
    );

    let result = admin.get_derived_function(&FunctionalExpression::Arithmetic(sum));

    assert!(result.symbol.contains("derived!"));
    assert!(result.symbol.contains("sum"));
    assert_eq!(admin.get_all_axioms().len(), 1);
}

#[test]
fn test_caching() {
    let mut admin = NormalizationFunctionAdministrator::new();
    let constant = FunctionalExpression::Constant(NumericConstant::new(42.0));

    let pne1 = admin.get_derived_function(&constant);
    let pne2 = admin.get_derived_function(&constant);

    // Should return the same symbol (caching works)
    assert_eq!(pne1.symbol, pne2.symbol);
    assert_eq!(admin.get_all_axioms().len(), 1); // Only one axiom created
}

#[test]
fn test_commutative_canonicalization() {
    let mut admin = NormalizationFunctionAdministrator::new();
    let pne1 = PrimitiveNumericExpression::new("a".to_string(), vec![], 'S');
    let pne2 = PrimitiveNumericExpression::new("b".to_string(), vec![], 'S');

    // Create a + b
    let sum1 = ArithmeticExpression::new(
        "+".to_string(),
        vec![
            FunctionalExpression::Primitive(pne1.clone()),
            FunctionalExpression::Primitive(pne2.clone()),
        ],
    );

    // Create b + a (should be same due to sorting)
    let sum2 = ArithmeticExpression::new(
        "+".to_string(),
        vec![
            FunctionalExpression::Primitive(pne2),
            FunctionalExpression::Primitive(pne1),
        ],
    );

    let result1 = admin.get_derived_function(&FunctionalExpression::Arithmetic(sum1));
    let result2 = admin.get_derived_function(&FunctionalExpression::Arithmetic(sum2));

    // Should be the same symbol due to canonicalization
    assert_eq!(result1.symbol, result2.symbol);
    assert_eq!(admin.get_all_axioms().len(), 1);
}
