use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, NumericRootTask, NumericRootTaskParts, NumericTaskExt,
    metric_operator_cost_from_initial_values,
};

use super::*;
use crate::evaluation::abstractions::collections::goal_focused::RefinementRootDeadEnd;
use crate::evaluation::abstractions::common::concrete_state::ConcreteState;
use crate::evaluation::abstractions::domain::cegar::{CegarConfig, FlawKind};
use crate::evaluation::abstractions::domain::generator::DomainAbstractionGenerator;

const X: VariableIndex = VariableIndex::new(0);
const ONE: VariableIndex = VariableIndex::new(1);
const TWO: VariableIndex = VariableIndex::new(2);
const BOUND: VariableIndex = VariableIndex::new(3);
const COST: VariableIndex = VariableIndex::new(4);

/// A counter `x` from 0 with the goal `x >= bound` (or `x <= bound`), whose
/// `increment` also doubles the metric variable `cost` from 1: its metric cost
/// is 1 at the initial state but 2^k after k increments.
fn doubling_cost_task(bound: f64, at_most: bool) -> NumericRootTask {
    let (operator, holds) = if at_most {
        (ComparisonOperator::LessThanOrEqual, 0.0 <= bound)
    } else {
        (ComparisonOperator::GreaterThanOrEqual, 0.0 >= bound)
    };
    NumericRootTask::new(NumericRootTaskParts {
        version: 2,
        metric: Metric::new(true, Some(COST)),
        variables: vec![ExplicitVariable::new(
            ConditionValue::DOMAIN_SIZE,
            "goal".into(),
            vec!["true".into(), "false".into()],
            Some(0),
            ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
        )],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("two".into(), NumericType::Constant, None),
            NumericVariable::new("bound".into(), NumericType::Constant, None),
            NumericVariable::new("cost".into(), NumericType::Cost, None),
        ],
        goals: vec![ExplicitFact::propositional(0, 0)],
        mutexes: vec![],
        state: vec![ExplicitValueIndex::from_usize(usize::from(!holds))],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(1.0),
            NumericValue::new(2.0),
            NumericValue::new(bound),
            NumericValue::new(1.0),
        ],
        operators: vec![Operator::new(
            "increment".into(),
            vec![],
            vec![],
            vec![
                AssignmentEffect::new(X, AssignmentOperation::Plus, ONE, false, vec![]),
                AssignmentEffect::new(COST, AssignmentOperation::Times, TWO, false, vec![]),
            ],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms: vec![ComparisonAxiom::new(
            VariableIndex::new(0),
            X,
            BOUND,
            operator,
        )],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

fn root_after(task: &NumericRootTask, increments: usize) -> ConcreteState {
    ConcreteState::initial(task)
        .unwrap()
        .replay(task, &vec![OperatorIndex::new(0); increments])
        .unwrap()
}

#[test]
fn rooted_task_moves_only_the_initial_state() {
    let task = doubling_cost_task(4.0, false);
    let root = root_after(&task, 2);
    let rooted = RootedTask::new(&task, &root).unwrap();

    assert_eq!(
        rooted.get_initial_numeric_state_values(),
        &[2.0, 1.0, 2.0, 4.0, 4.0].map(NumericValue::new)
    );
    assert_eq!(
        rooted.get_initial_propositional_state_values(),
        &[ExplicitValueIndex::from_usize(
            ConditionValue::False.as_usize()
        )]
    );
    assert_eq!(
        rooted.evaluate_initial_state_values().unwrap(),
        (
            rooted.get_initial_propositional_state_values().to_vec(),
            rooted.get_initial_numeric_state_values().to_vec()
        )
    );
    assert_eq!(rooted.goals(), task.goals());
    assert_eq!(rooted.get_operators().len(), task.get_operators().len());
    // Evaluated at the root, the metric would price the increment at 4; the
    // abstraction cost hook keeps the base task's price.
    assert_eq!(
        metric_operator_cost_from_initial_values(&rooted, &task.get_operators()[0]),
        NumericValue::new(4.0)
    );
    assert_eq!(
        rooted.abstract_operator_cost(OperatorIndex::new(0)),
        NumericValue::new(1.0)
    );
    assert_eq!(
        task.abstract_operator_cost(OperatorIndex::new(0)),
        NumericValue::new(1.0)
    );
}

#[test]
fn rooted_task_rejects_a_root_that_changes_a_constant() {
    let task = doubling_cost_task(4.0, false);
    let mut root = root_after(&task, 1);
    root.numeric[BOUND.index()] = NumericValue::new(5.0);
    let error = RootedTask::new(&task, &root).err().unwrap();

    assert!(
        error.to_string().contains("constant numeric variable 3"),
        "{error:#}"
    );
}

fn cegar_config() -> CegarConfig {
    CegarConfig {
        max_abstraction_size: 1_000,
        flaw_kind: FlawKind::Progression,
        init_split_var_ids: Some(Default::default()),
        ..Default::default()
    }
}

#[test]
fn a_rooted_domain_abstraction_plans_from_the_root_and_reports_the_initial_state() {
    let task = doubling_cost_task(4.0, false);
    let root = root_after(&task, 2);
    let abstraction = DomainAbstractionGenerator::new(cegar_config())
        .unwrap()
        .generate_from_root(&task, Some(&root))
        .unwrap();

    assert!(abstraction.metadata.solved_by_self);
    assert_eq!(
        abstraction.metadata.concrete_plan_operator_ids,
        Some(vec![OperatorIndex::new(0); 2]),
        "the executed plan starts at the root x=2"
    );
    // h(s0) uses the base task's initial state and its pinned unit costs: the
    // abstraction never overestimates the four increments from x=0.
    let initial_h =
        abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash];
    assert!(initial_h <= 4.0, "h(s0)={initial_h} overestimates h*=4");
    assert!(initial_h.is_finite());
}

#[test]
fn a_dead_root_is_a_typed_error() {
    let task = doubling_cost_task(0.0, true);
    let root = root_after(&task, 1);
    let error = DomainAbstractionGenerator::new(cegar_config())
        .unwrap()
        .generate_from_root(&task, Some(&root))
        .unwrap_err();

    assert!(
        error.downcast_ref::<RefinementRootDeadEnd>().is_some(),
        "{error:#}"
    );
}
