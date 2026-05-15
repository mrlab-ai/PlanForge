use planforge_sas::numeric::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use planforge_sas::numeric::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitVariable, Metric, NumericRootTask,
    NumericType, NumericVariable, Operator,
};

use super::*;

fn simple_var(name: &str, axiom_layer: Option<usize>) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        axiom_layer,
        1,
    )
}

#[test]
fn causal_graph_collects_operator_and_axiom_dependencies() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![
            simple_var("pre", None),
            simple_var("goal", None),
            ExplicitVariable::new(
                3,
                "cmp".to_string(),
                vec!["t".to_string(), "f".to_string(), "u".to_string()],
                Some(0),
                2,
            ),
        ],
        vec![
            NumericVariable::new("c".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        vec![ExplicitFact::new(1, 1)],
        vec![],
        vec![0, 0, 2],
        vec![1.0, 0.0],
        vec![Operator::new(
            "advance".to_string(),
            vec![ExplicitFact::new(0, 1), ExplicitFact::new(2, 0)],
            vec![planforge_sas::numeric::numeric_task::Effect::new(
                vec![],
                1,
                Some(0),
                1,
            )],
            vec![AssignmentEffect::new(
                1,
                AssignmentOperation::Plus,
                0,
                false,
                vec![],
            )],
            1,
        )],
        vec![PropositionalAxiom::new(
            vec![ExplicitFact::new(0, 1)],
            1,
            0,
            1,
        )],
        vec![ComparisonAxiom::new(
            2,
            1,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![],
        ExplicitFact::new(0, 0),
    );

    let graph = MixedCausalGraph::new(&task);

    assert!(
        graph
            .predecessors_of(CausalGraphVariable::Regular(1))
            .collect::<Vec<_>>()
            .contains(&CausalGraphVariable::Regular(0))
    );
    assert_eq!(
        graph
            .predecessors_of(CausalGraphVariable::Regular(2))
            .collect::<Vec<_>>(),
        Vec::<CausalGraphVariable>::new()
    );
    assert_eq!(
        graph.goal_distance(CausalGraphVariable::Regular(1)),
        Some(0)
    );
    assert_eq!(
        graph.goal_distance(CausalGraphVariable::Regular(0)),
        Some(1)
    );
}

#[test]
fn causal_graph_bypasses_comparison_propositions_for_operator_preconditions() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![
            simple_var("goal", None),
            ExplicitVariable::new(
                3,
                "cmp".to_string(),
                vec!["t".to_string(), "f".to_string(), "u".to_string()],
                Some(0),
                2,
            ),
        ],
        vec![
            NumericVariable::new("c5".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("sum".to_string(), NumericType::Derived, Some(0)),
        ],
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![0, 2],
        vec![5.0, 0.0, 0.0, 0.0],
        vec![Operator::new(
            "achieve-goal".to_string(),
            vec![ExplicitFact::new(1, 0)],
            vec![planforge_sas::numeric::numeric_task::Effect::new(
                vec![],
                0,
                Some(1),
                0,
            )],
            vec![],
            1,
        )],
        vec![],
        vec![ComparisonAxiom::new(
            1,
            3,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![AssignmentAxiom::new(3, CalOperator::Sum, 1, 2)],
        ExplicitFact::new(0, 0),
    );

    let graph = MixedCausalGraph::new(&task);
    let helper_var_id = task.numeric_variables().len();
    let predecessors = graph
        .predecessors_of(CausalGraphVariable::Regular(0))
        .collect::<Vec<_>>();

    assert!(predecessors.contains(&CausalGraphVariable::Numeric(helper_var_id)));
    assert!(!predecessors.contains(&CausalGraphVariable::Regular(1)));
    assert!(
        graph
            .predecessors_of(CausalGraphVariable::Numeric(helper_var_id))
            .collect::<Vec<_>>()
            .is_empty()
    );
}

#[test]
fn causal_graph_flattens_helper_predecessors_to_regular_leaves() {
    let task = NumericRootTask::new(
        1,
        Metric::new(true, None),
        vec![
            simple_var("goal", None),
            ExplicitVariable::new(
                3,
                "cmp".to_string(),
                vec!["t".to_string(), "f".to_string(), "u".to_string()],
                Some(0),
                2,
            ),
        ],
        vec![
            NumericVariable::new("c5".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
            NumericVariable::new("z".to_string(), NumericType::Regular, None),
            NumericVariable::new("a".to_string(), NumericType::Derived, Some(0)),
            NumericVariable::new("b".to_string(), NumericType::Derived, Some(0)),
        ],
        vec![ExplicitFact::new(0, 0)],
        vec![],
        vec![0, 2],
        vec![5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![Operator::new(
            "achieve-goal".to_string(),
            vec![ExplicitFact::new(1, 0)],
            vec![planforge_sas::numeric::numeric_task::Effect::new(
                vec![],
                0,
                Some(1),
                0,
            )],
            vec![],
            1,
        )],
        vec![],
        vec![ComparisonAxiom::new(
            1,
            5,
            0,
            ComparisonOperator::GreaterThanOrEqual,
        )],
        vec![
            AssignmentAxiom::new(4, CalOperator::Sum, 1, 2),
            AssignmentAxiom::new(5, CalOperator::Sum, 4, 3),
        ],
        ExplicitFact::new(0, 0),
    );

    let graph = MixedCausalGraph::new(&task);
    let root_helper_id = task.numeric_variables().len() + 1;
    let intermediate_helper_id = task.numeric_variables().len();
    let predecessors = graph
        .predecessors_of(CausalGraphVariable::Numeric(root_helper_id))
        .collect::<Vec<_>>();

    assert!(predecessors.is_empty());
    assert!(!predecessors.contains(&CausalGraphVariable::Numeric(intermediate_helper_id)));
}
