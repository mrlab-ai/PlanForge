use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator, PropositionalAxiom};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitValueIndex, ExplicitVariable,
    Metric, NumericRootTask, NumericRootTaskParts, NumericType, NumericValue, NumericVariable,
    Operator, OperatorCost, VariableIndex,
};

use super::*;

fn simple_var(name: &str, axiom_layer: Option<usize>) -> ExplicitVariable {
    ExplicitVariable::new(
        2,
        name.to_string(),
        vec![format!("{name}=0"), format!("{name}=1")],
        axiom_layer,
        ExplicitValueIndex::new(1),
    )
}

#[test]
fn causal_graph_collects_operator_and_axiom_dependencies() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![
            simple_var("pre", None),
            simple_var("goal", None),
            ExplicitVariable::new(
                3,
                "cmp".to_string(),
                vec!["t".to_string(), "f".to_string(), "u".to_string()],
                Some(0),
                ExplicitValueIndex::new(2),
            ),
        ],
        numeric_variables: vec![
            NumericVariable::new("c".to_string(), NumericType::Constant, None),
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
        ],
        goals: vec![ExplicitFact::propositional(1, 1)],
        mutexes: vec![],
        state: vec![
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(2),
        ],
        numeric_state: vec![NumericValue::new(1.0), NumericValue::new(0.0)],
        operators: vec![Operator::new(
            "advance".to_string(),
            vec![
                ExplicitFact::propositional(0, 1),
                ExplicitFact::propositional(2, 0),
            ],
            vec![planforge_sas::numeric_task::Effect::new(
                vec![],
                VariableIndex::new(1),
                Some(ExplicitValueIndex::new(0)),
                ExplicitValueIndex::new(1),
            )],
            vec![AssignmentEffect::new(
                VariableIndex::from_usize(1),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(0),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        )],
        axioms: vec![PropositionalAxiom::new(
            vec![ExplicitFact::propositional(0, 1)],
            VariableIndex::from_usize(1),
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(1),
        )],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(2),
            VariableIndex::new(1),
            VariableIndex::new(0),
            ComparisonOperator::GreaterThanOrEqual,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let graph = CausalGraph::new(&task).unwrap();

    assert!(
        graph
            .predecessors_of(CausalGraphVariable::Propositional(
                VariableIndex::from_usize(1)
            ))
            .collect::<Vec<_>>()
            .contains(&CausalGraphVariable::Propositional(
                VariableIndex::from_usize(0)
            ))
    );
    assert_eq!(
        graph
            .predecessors_of(CausalGraphVariable::Propositional(
                VariableIndex::from_usize(2)
            ))
            .collect::<Vec<_>>(),
        Vec::<CausalGraphVariable>::new()
    );
}

#[test]
fn restricted_causal_graph_tracks_numeric_effect_sources() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        version: 1,
        metric: Metric::new(true, None),
        variables: vec![simple_var("p", None)],
        numeric_variables: vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, None),
            NumericVariable::new("y".to_string(), NumericType::Regular, None),
        ],
        goals: vec![],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::new(0)],
        numeric_state: vec![NumericValue::new(0.0), NumericValue::new(1.0)],
        operators: vec![Operator::new(
            "update-x".to_string(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                VariableIndex::from_usize(0),
                AssignmentOperation::Plus,
                VariableIndex::from_usize(1),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    });

    let graph = CausalGraph::new(&task).unwrap();
    let predecessors = graph
        .predecessors_of(CausalGraphVariable::Numeric(VariableIndex::from_usize(0)))
        .collect::<Vec<_>>();

    assert_eq!(
        predecessors,
        vec![CausalGraphVariable::Numeric(VariableIndex::from_usize(1))]
    );
}
