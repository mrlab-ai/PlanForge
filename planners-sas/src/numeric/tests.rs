use crate::numeric::{
    axioms::{AssignmentAxiom, AxiomEvaluator, ComparisonAxiom, PropositionalAxiom},
    numeric_task::{
        AbstractNumericTask, Effect, ExplicitVariable, Fact, Metric, NumericRootTask, NumericType,
        NumericVariable, Operator,
    },
    state_registry::StateRegistry,
    utils::int_packer::IntDoublePacker,
};

pub(crate) fn get_root_task() -> NumericRootTask {
    let version = 4;
    let metric = Metric::new(true, 1);
    let variables = vec![
        ExplicitVariable::new(
            2,
            String::from("var13"),
            vec![String::from("new-axiom"), String::from("not-new-axiom")],
            0,
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
            -1,
            0,
        ),
    ];
    let numeric_variables = vec![
        NumericVariable::new(String::from("derived!1.0()"), NumericType::Constant, -1),
        NumericVariable::new(String::from("total_cost()"), NumericType::Cost, -1),
    ];
    let goals = vec![
        Fact::new(9, 4),
        Fact::new(10, 1),
        Fact::new(11, 2),
        Fact::new(12, 5),
        Fact::new(13, 4),
    ];
    let mutexes = Vec::new();
    let state = vec![1, 1];
    let numeric_state = vec![1f64, 0f64];
    let operators = vec![Operator::new(
        String::from("drop"),
        vec![Fact::new(1, 1)],
        vec![Effect::new(Vec::new(), 1, 1, 5)],
        Vec::new(),
        1,
    )];
    let axioms = vec![PropositionalAxiom::new(vec![], 0, 0, 1)];
    let comparison_axioms = vec![ComparisonAxiom::new(
        1,
        0,
        1,
        crate::numeric::axioms::ComparisonOperator::Equal,
    )];
    let assignment_axioms = vec![AssignmentAxiom::new(
        1,
        crate::numeric::axioms::CalOperator::Sum,
        0,
        1,
    )];
    let global_constraint = (0, 0);
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

pub(crate) fn setup_state_packer(problem: &NumericRootTask) -> IntDoublePacker {
    let mut domain_sizes = vec![];
    for var in problem.variables().iter() {
        domain_sizes.push(var.domain_size() as u64);
    }
    for numeric_var in problem.numeric_variables().iter() {
        if numeric_var.get_type() == &NumericType::Regular {
            domain_sizes.push(u64::MAX);
        }
    }
    IntDoublePacker::new(&domain_sizes)
}

pub(crate) fn setup_axiom_evaluator<'a>(
    problem: &'a NumericRootTask,
    state_packer: &'a IntDoublePacker,
) -> AxiomEvaluator<'a> {
    let task: &'a dyn AbstractNumericTask = problem;
    let axiom_evaluator = AxiomEvaluator::new(task, state_packer);
    axiom_evaluator
}

pub(crate) fn setup_state_registry<'a>(
    problem: &'a NumericRootTask,
    state_packer: &'a IntDoublePacker,
    axiom_evaluator: &'a AxiomEvaluator<'a>,
) -> StateRegistry<'a> {
    StateRegistry::new(problem, state_packer, axiom_evaluator)
}
