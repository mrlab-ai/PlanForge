use super::*;

/// Every letter the format spells a numeric type with reads back as the
/// type it was written from; a table that disagreed with itself would
/// retype a variable on a round trip through the file.
#[test]
fn a_numeric_type_survives_the_format_letter() {
    for numeric_type in [
        NumericType::Constant,
        NumericType::Derived,
        NumericType::Cost,
        NumericType::Regular,
    ] {
        assert_eq!(
            NumericType::from_sas(numeric_type.as_sas()),
            Some(numeric_type)
        );
    }
}

#[test]
fn an_unknown_numeric_type_letter_is_rejected() {
    assert_eq!(NumericType::from_sas("X"), None);
}

/// `=` is an assignment operator but not one a numeric axiom can combine
/// its operands with, and the two tables have to disagree about it.
#[test]
fn assignment_and_axiom_operators_differ_on_assignment() {
    assert!(AssignmentOperation::from_sas("=").is_some());
    assert!(CalOperator::from_sas("=").is_none());
}

/// Every token table is written by the writer and read by the parser, so a
/// table that disagreed with its own inverse would change an operator, a
/// comparator or an axiom on the way through the file.
#[test]
fn every_operator_token_reads_back_as_what_it_was_written_from() {
    for comparator in [
        ComparisonOperator::LessThan,
        ComparisonOperator::LessThanOrEqual,
        ComparisonOperator::Equal,
        ComparisonOperator::GreaterThanOrEqual,
        ComparisonOperator::GreaterThan,
        ComparisonOperator::UnEqual,
    ] {
        assert_eq!(
            ComparisonOperator::from_sas(comparator.as_sas()),
            Some(comparator)
        );
    }
    for operator in [
        CalOperator::Sum,
        CalOperator::Difference,
        CalOperator::Product,
        CalOperator::Division,
    ] {
        assert_eq!(CalOperator::from_sas(operator.as_sas()), Some(operator));
    }
    for operation in [
        AssignmentOperation::Assign,
        AssignmentOperation::Plus,
        AssignmentOperation::Minus,
        AssignmentOperation::Times,
        AssignmentOperation::Divide,
    ] {
        assert_eq!(
            AssignmentOperation::from_sas(operation.as_sas()),
            Some(operation)
        );
    }
}

#[test]
fn a_metric_without_a_variable_is_spelled_zero() {
    let metric = Metric::from_sas('<', None).expect("`<` is a direction");
    assert!(metric.is_min());
    assert!(!metric.use_metric());
    assert_eq!(metric.as_sas(), ('<', None));

    let metric = Metric::from_sas('>', Some(VariableIndex::new(3))).expect("`>` is a direction");
    assert!(!metric.is_min());
    assert_eq!(metric.var_id(), Some(VariableIndex::new(3)));
    assert_eq!(metric.as_sas(), ('>', Some(VariableIndex::new(3))));

    assert!(Metric::from_sas('=', Some(VariableIndex::new(1))).is_none());
}

#[test]
#[should_panic(expected = "is not a non-negative integer")]
fn a_fractional_operator_cost_is_rejected() {
    operator_cost_from_sas(2.5);
}

#[test]
fn an_integral_operator_cost_is_the_integer_it_spells() {
    assert_eq!(operator_cost_from_sas(0.0), 0);
    assert_eq!(operator_cost_from_sas(7.0), 7);
}

#[test]
fn a_negative_axiom_layer_means_no_axiom_derives_the_variable() {
    assert_eq!(axiom_layer_from_sas(-1), None);
    assert_eq!(axiom_layer_from_sas(0), Some(0));
    assert_eq!(axiom_layer_from_sas(3), Some(3));
}

/// Layer zero and "no layer" are different answers, and an effect that
/// requires value zero is not an effect that requires nothing. Both fields
/// spell absence as `-1`, so the two easiest mistakes here are the same one
/// -- in both directions, since the writer spells these fields too.
#[test]
fn an_effect_precondition_of_zero_is_not_the_absence_of_one() {
    assert_eq!(effect_precondition_from_sas(-1), None);
    assert_eq!(
        effect_precondition_from_sas(0),
        Some(ExplicitValueIndex::new(0))
    );
    assert_eq!(optional_value_to_sas(None), -1);
    assert_eq!(optional_value_to_sas(Some(0)), 0);
    assert_eq!(optional_value_to_sas(Some(3)), 3);
}

/// An operator's two condition lists are one list to the search, and the
/// order they merge in is the order the file states them: the prevail
/// conditions, then the effects' own requirements in effect order.
#[test]
fn an_operator_merges_its_prevail_and_effect_conditions_in_file_order() {
    let operator = SasOperator {
        name: "move".to_owned(),
        prevail: vec![ExplicitFact::propositional(7, 1)],
        effects: vec![
            Effect::new(
                vec![],
                VariableIndex::new(4),
                None,
                ExplicitValueIndex::new(1),
            ),
            Effect::new(
                vec![],
                VariableIndex::new(5),
                Some(ExplicitValueIndex::new(0)),
                ExplicitValueIndex::new(1),
            ),
            Effect::new(
                vec![],
                VariableIndex::new(6),
                Some(ExplicitValueIndex::new(2)),
                ExplicitValueIndex::new(0),
            ),
        ],
        assignment_effects: vec![],
        cost: OperatorCost::new(3),
    }
    .into_operator();

    let preconditions: Vec<(usize, usize)> = operator
        .preconditions()
        .iter()
        .map(|fact| (fact.var(), fact.value()))
        .collect();
    // Variable 4 contributes nothing: its effect applies whatever the
    // variable holds.
    assert_eq!(preconditions, [(7, 1), (5, 0), (6, 2)]);
    assert_eq!(operator.effects().len(), 3);
    assert_eq!(operator.cost(), OperatorCost::new(3));
}
