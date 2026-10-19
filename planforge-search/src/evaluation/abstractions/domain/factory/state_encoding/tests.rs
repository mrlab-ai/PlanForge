use super::*;

use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    ExplicitFact, ExplicitValueIndex, ExplicitVariable, Metric, NumericRootTask,
    NumericRootTaskParts, NumericType, NumericVariable, Operator, OperatorCost, VariableIndex,
    ZERO_VALUE,
};

use crate::evaluation::abstractions::domain::utils::identity_domain_mapping_and_sizes;

/// `x < y` over two unbounded numeric variables: undecided in the only
/// abstract numeric state, so enumeration branches on the comparison.
fn undecided_comparison_task() -> NumericRootTask {
    NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![ExplicitVariable::new(
            ConditionValue::DOMAIN_SIZE,
            "x<y".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
        )],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("y".into(), NumericType::Regular, None),
        ],
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::from_usize(
            ConditionValue::False.as_usize(),
        )],
        numeric_state: vec![ZERO_VALUE, ZERO_VALUE],
        operators: vec![Operator::new(
            "noop".into(),
            vec![],
            vec![],
            vec![],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            VariableIndex::new(0),
            VariableIndex::new(1),
            ComparisonOperator::LessThan,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

#[test]
fn comparison_enumeration_memo_verifies_keys_behind_colliding_signatures() {
    let task = undecided_comparison_task();
    let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    let factory = DomainAbstractionFactory::new(
        &task,
        domain_mapping,
        domain_sizes,
        NumericPartitions::trivial(&task),
        vec![1, 1],
    )
    .unwrap();
    let generator = factory.make_operator_generator(&task, true).unwrap();
    let comparison_var_ids = [VariableIndex::new(0)];
    let mut memo = ComparisonEnumerationMemo::new(ComparisonBranchingLayout {
        numeric_domain_sizes: generator.numeric_domain_sizes(),
        hash_multipliers: generator.hash_multipliers(),
        comparison_var_ids: &comparison_var_ids,
    });
    let base = 0;
    let fixed_true = [ExplicitFact::propositional(
        0,
        ConditionValue::True.as_usize(),
    )];

    // Plant an entry for a different key under the signature of `(base, [])`,
    // as a 64-bit collision would.
    memo.entries_by_signature.insert(
        comparison_enumeration_signature(base, &[]),
        vec![ComparisonEnumerationEntry {
            base_state_hash: base + 1,
            fixed_comparisons: Box::new([]),
            states: vec![usize::MAX],
        }],
    );
    let mut both = factory
        .enumerate_states_with_evaluated_comparisons_cached(base, &task, &[], &mut memo)
        .unwrap()
        .to_vec();
    both.sort_unstable();
    assert_eq!(both.len(), 2, "both verdicts of the undecided comparison");
    assert!(
        !both.contains(&usize::MAX),
        "the colliding entry must not answer"
    );
    assert_eq!(
        memo.entries_by_signature[&comparison_enumeration_signature(base, &[])].len(),
        2,
        "the real key is stored next to the colliding one"
    );

    // A hit returns the stored set; a fixed comparison is a different key.
    let again = factory
        .enumerate_states_with_evaluated_comparisons_cached(base, &task, &[], &mut memo)
        .unwrap();
    assert_eq!(again.len(), 2);
    let fixed = factory
        .enumerate_states_with_evaluated_comparisons_cached(base, &task, &fixed_true, &mut memo)
        .unwrap();
    assert_eq!(fixed.len(), 1, "a fixed comparison does not branch");
}
