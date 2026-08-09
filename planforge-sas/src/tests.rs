use crate::{
    axioms::{ComparisonAxiom, PropositionalAxiom},
    numeric_task::{
        AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, ExplicitVariable, Metric,
        NumericRootTask, NumericType, NumericVariable, Operator,
    },
};

pub(crate) fn get_root_task() -> NumericRootTask {
    let version = 4;
    let metric = Metric::new(true, Some(1));
    let variables = vec![
        ExplicitVariable::new(
            2,
            String::from("var13"),
            vec![String::from("new-axiom"), String::from("not-new-axiom")],
            Some(0),
            0,
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
            0,
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new(String::from("derived!1.0()"), NumericType::Constant, None),
        NumericVariable::new(String::from("total_cost()"), NumericType::Cost, None),
    ];
    let goals = vec![
        ExplicitFact::new(9, 4),
        ExplicitFact::new(10, 1),
        ExplicitFact::new(11, 2),
        ExplicitFact::new(12, 5),
        ExplicitFact::new(13, 4),
    ];
    let mutexes = Vec::new();
    let state = vec![1, 1];
    let numeric_state = vec![1f64, 0f64];
    // `drop` bumps the cost counter by one, which flips the comparison below
    // and therefore distinguishes the successor from the initial state.
    let operators = vec![Operator::new(
        String::from("drop"),
        vec![ExplicitFact::new(1, 1)],
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
        1,
        0,
        1,
        crate::axioms::ComparisonOperator::GreaterThan,
    )];
    // Accumulating a variable into itself is an operator effect, not an
    // assignment axiom: axioms define a numeric variable from *other*
    // numeric variables, and a self-referential definition has no fixpoint.
    let assignment_axioms = Vec::new();
    let global_constraint = ExplicitFact::new(0, 0);
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
