use crate::{
    axioms::{ComparisonAxiom, PropositionalAxiom},
    numeric_task::{
        AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
        NumericRootTask, NumericType, NumericVariable, Operator,
    },
};

/// The variable the fixture's comparison axiom writes.
const CONDITION_VAR: usize = 2;

pub(crate) fn get_root_task() -> NumericRootTask {
    root_task_with_extra_preconditions(Vec::new())
}

/// The fixture task, optionally with extra preconditions on its single
/// operator. Tests that need a fact on a specific variable to travel through
/// `NumericRootTask::new` pass it here rather than rebuilding the task.
pub(crate) fn root_task_with_extra_preconditions(
    extra_preconditions: Vec<ExplicitFact>,
) -> NumericRootTask {
    let version = 4;
    let metric = Metric::new(true, Some(1));
    let variables = vec![
        ExplicitVariable::new(
            2,
            String::from("var13"),
            vec![String::from("new-axiom"), String::from("not-new-axiom")],
            Some(1),
            1,
        ),
        ExplicitVariable::new(
            7,
            String::from("var10"),
            vec![
                String::from("on(d, a)"),
                String::from("on(d, b)"),
                String::from("on(d, c)"),
                String::from("on(d, e)"),
                String::from("on(d, f)"),
                String::from("ontable(d)"),
            ],
            None,
            1,
        ),
        // The comparison axiom's target: a derived variable carrying the
        // verdict of `1.0 > total_cost`, three-valued like every numeric
        // condition and defaulting to `Unknown`.
        ExplicitVariable::new(
            3,
            String::from("var-cost-below-one"),
            vec![
                String::from("cost-below-one"),
                String::from("not-cost-below-one"),
                String::from("unknown-cost-below-one"),
            ],
            Some(0),
            2,
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new(String::from("derived!1.0()"), NumericType::Constant, None),
        NumericVariable::new(String::from("total_cost()"), NumericType::Cost, None),
    ];
    // `var10` on the table while the cost is still below one. Both goals go in
    // untagged, as the parser hands them over, so `NumericRootTask::new` has to
    // recognise that the second one names a condition variable.
    let goals = vec![
        ExplicitFact::propositional(1, 5),
        ExplicitFact::propositional(CONDITION_VAR, 0),
    ];
    let mutexes = Vec::new();
    let state = vec![1, 1, 2];
    let numeric_state = vec![1f64, 0f64];
    // `drop` bumps the cost counter by one, which flips the comparison below
    // and therefore distinguishes the successor from the initial state.
    let mut preconditions = vec![ExplicitFact::propositional(1, 1)];
    preconditions.extend(extra_preconditions);
    let operators = vec![Operator::new(
        String::from("drop"),
        preconditions,
        vec![Effect::new(Vec::new(), 1, Some(1), 5)],
        vec![AssignmentEffect::new(
            1,
            AssignmentOperation::Plus,
            0,
            false,
            vec![],
        )],
        1,
    )];
    let axioms = vec![PropositionalAxiom::new(vec![], 0, 0, 1)];
    let comparison_axioms = vec![ComparisonAxiom::new(
        2,
        0,
        1,
        crate::axioms::ComparisonOperator::GreaterThan,
    )];
    // Accumulating a variable into itself is an operator effect, not an
    // assignment axiom: axioms define a numeric variable from *other*
    // numeric variables, and a self-referential definition has no fixpoint.
    let assignment_axioms = Vec::new();
    let global_constraint = ExplicitFact::propositional(0, 0);
    NumericRootTask::new(
        version,
        metric,
        variables,
        numeric_variables,
        goals,
        mutexes,
        state,
        numeric_state,
        operators,
        axioms,
        comparison_axioms,
        assignment_axioms,
        global_constraint,
    )
}

#[cfg(test)]
mod fact_namespace {
    use super::{CONDITION_VAR, get_root_task};
    use crate::numeric_task::{
        AbstractNumericTask, ExplicitFact, FactNamespace, assert_fact_namespaces,
    };

    #[test]
    fn tag_survives_var_and_value() {
        let fact = ExplicitFact::condition(ExplicitFact::MAX_VAR_ID, u32::MAX as usize);
        assert_eq!(fact.namespace(), FactNamespace::Condition);
        assert_eq!(fact.var(), ExplicitFact::MAX_VAR_ID);
        assert_eq!(fact.value(), u32::MAX as usize);
    }

    #[test]
    #[should_panic(expected = "exceeds the 28 packed variable-id bits")]
    fn var_id_beyond_the_tag_boundary_is_rejected() {
        ExplicitFact::propositional(ExplicitFact::MAX_VAR_ID + 1, 0);
    }

    #[test]
    fn identity_and_order_ignore_the_tag() {
        let propositional = ExplicitFact::propositional(7, 1);
        let condition = ExplicitFact::condition(7, 1);
        let numeric = ExplicitFact::numeric_variable(7, 1);
        assert_eq!(propositional, condition);
        assert_eq!(propositional, numeric);
        assert_eq!(propositional.cmp(&condition), std::cmp::Ordering::Equal);
        assert_eq!(propositional.cmp(&numeric), std::cmp::Ordering::Equal);
        // Variable-major, which is what the successor generator sorts on.
        assert!(ExplicitFact::condition(6, 9) < ExplicitFact::propositional(7, 0));
        assert!(ExplicitFact::numeric_variable(6, 9) < ExplicitFact::propositional(7, 0));
    }

    /// Facts in a domain abstraction's own numeric id space are not facts of
    /// the task, and the tag is what says so.
    #[test]
    fn the_numeric_variable_namespace_round_trips() {
        let fact = ExplicitFact::numeric_variable(ExplicitFact::MAX_VAR_ID, 3);
        assert_eq!(fact.namespace(), FactNamespace::NumericVariable);
        assert!(!fact.is_condition());
        assert_eq!(fact.var(), ExplicitFact::MAX_VAR_ID);
        assert_eq!(fact.value(), 3);
        assert_eq!(format!("{fact:?}"), "Fact(num: 268435455, partition: 3)");
    }

    #[test]
    fn the_task_tags_facts_on_its_condition_variables() {
        let task = get_root_task();
        assert!(task.numeric_conditions().is_condition_var(CONDITION_VAR));
        assert_eq!(
            task.numeric_conditions().fact(CONDITION_VAR, 0).namespace(),
            FactNamespace::Condition
        );
        assert_eq!(
            task.numeric_conditions().fact(1, 0).namespace(),
            FactNamespace::Propositional
        );
        assert_fact_namespaces(&task);
    }

    /// The parser cannot know the namespaces, so it hands facts over untagged
    /// and `NumericRootTask::new` retags them. A fact on a condition variable
    /// must come back out tagged even though it went in as propositional.
    #[test]
    fn construction_retags_facts_the_parser_could_not_tag() {
        let task = super::root_task_with_extra_preconditions(vec![ExplicitFact::propositional(
            CONDITION_VAR,
            0,
        )]);
        let precondition = *task.get_operators()[0].preconditions().last().unwrap();
        assert_eq!(precondition.var(), CONDITION_VAR);
        assert_eq!(precondition.namespace(), FactNamespace::Condition);
        assert_fact_namespaces(&task);
    }
}
