use crate::evaluation::abstractions::collections::generator::AbstractionGenerator;
use crate::evaluation::abstractions::common::cegar::options::CegarOptions;
use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, NumericRootTask, NumericRootTaskParts,
};

use super::*;
use crate::evaluation::abstractions::common::cegar::{CegarStopReason, FlawKind};
use crate::evaluation::abstractions::domain::config::DomainAbstractionConfig;
use crate::evaluation::abstractions::domain::generator::DomainAbstractionGenerator;
use planforge_sas::explicit_state::{ExplicitState, ExplicitStateSpace};

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

fn root_after(task: &NumericRootTask, increments: usize) -> ExplicitState {
    let states = ExplicitStateSpace::new(task);
    states
        .replay(
            &states.initial_state().unwrap(),
            &vec![OperatorIndex::new(0); increments],
        )
        .unwrap()
}

fn cegar_config() -> DomainAbstractionConfig {
    DomainAbstractionConfig {
        cegar: CegarOptions {
            max_states: 1_000,
            flaw_kind: FlawKind::Progression,
            ..DomainAbstractionConfig::default().cegar
        },
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

    assert!(abstraction.metadata.solved_by_self());
    assert_eq!(
        abstraction.metadata.concrete_plan_operator_ids,
        Some(vec![OperatorIndex::new(0); 2]),
        "the executed plan starts at the root x=2"
    );
    // h(s0) uses the base task's initial state and its pinned unit costs: the
    // abstraction never overestimates the four increments from x=0.
    let initial_h = abstraction.distance_table.distances[abstraction.distance_table.initial_state];
    assert!(initial_h <= 4.0, "h(s0)={initial_h} overestimates h*=4");
    assert!(initial_h.is_finite());
}

#[test]
fn a_dead_root_stops_refinement_as_a_dead_end() {
    let task = doubling_cost_task(0.0, true);
    let root = root_after(&task, 1);
    let abstraction = DomainAbstractionGenerator::new(cegar_config())
        .unwrap()
        .generate_from_root(&task, Some(&root))
        .unwrap();

    assert_eq!(abstraction.metadata.stop_reason, CegarStopReason::DeadEnd);
    assert!(!abstraction.metadata.solved_by_self());
}
