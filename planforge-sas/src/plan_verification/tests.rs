use super::*;
use crate::axioms::PropositionalAxiom;
use crate::numeric_task::{
    Effect, ExplicitValueIndex, ExplicitVariable, Metric, NumericRootTask, NumericRootTaskParts,
    OperatorCost, VariableIndex,
};
use std::sync::Arc;

/// A three-position chain task.
///
/// * `var0` — derived global-constraint atom, true (value 0) via an
///   unconditional axiom, mirroring the translator's `new-axiom@0()`.
/// * `var1` — position, domain 3, initially 0, goal 2.
///
/// Operators: `move_ab` (0→1), `move_bc` (1→2), `reset` (2→0). `reset`
/// exists so a sequence can be padded past the goal.
fn chain_task() -> NumericRootTask {
    chain_task_with(
        ExplicitFact::propositional(0, 0),
        ExplicitValueIndex::new(0),
    )
}

/// `chain_task`, parameterized by the global constraint and the initial
/// position, so tests can vary exactly one thing.
fn chain_task_with(
    global_constraint: ExplicitFact,
    initial_position: ExplicitValueIndex,
) -> NumericRootTask {
    let variables = vec![
        ExplicitVariable::new(
            2,
            String::from("var0"),
            vec![String::from("gc"), String::from("not-gc")],
            Some(0),
            ExplicitValueIndex::new(1),
        ),
        ExplicitVariable::new(
            3,
            String::from("var1"),
            vec![
                String::from("at(a)"),
                String::from("at(b)"),
                String::from("at(c)"),
            ],
            None,
            ExplicitValueIndex::new(0),
        ),
    ];
    let operators = vec![
        Operator::new(
            String::from("move_ab"),
            vec![ExplicitFact::propositional(1, 0)],
            vec![Effect::new(
                Vec::new(),
                VariableIndex::new(1),
                Some(ExplicitValueIndex::new(0)),
                ExplicitValueIndex::new(1),
            )],
            Vec::new(),
            OperatorCost::new(1),
        ),
        Operator::new(
            String::from("move_bc"),
            vec![ExplicitFact::propositional(1, 1)],
            vec![Effect::new(
                Vec::new(),
                VariableIndex::new(1),
                Some(ExplicitValueIndex::new(1)),
                ExplicitValueIndex::new(2),
            )],
            Vec::new(),
            OperatorCost::new(1),
        ),
        Operator::new(
            String::from("reset"),
            vec![ExplicitFact::propositional(1, 2)],
            vec![Effect::new(
                Vec::new(),
                VariableIndex::new(1),
                Some(ExplicitValueIndex::new(2)),
                ExplicitValueIndex::new(0),
            )],
            Vec::new(),
            OperatorCost::new(1),
        ),
    ];
    NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        // No metric variable, so every transition costs 1.0.
        metric: Metric::new(true, None),
        variables,
        numeric_variables: Vec::new(),
        goals: vec![ExplicitFact::propositional(1, 2)],
        mutexes: Vec::new(),
        state: vec![ExplicitValueIndex::new(1), initial_position],
        numeric_state: Vec::new(),
        operators,
        // Unconditionally derive var0 = 0 ("gc" holds).
        axioms: vec![PropositionalAxiom::new(
            Vec::new(),
            VariableIndex::new(0),
            ExplicitValueIndex::new(1),
            ExplicitValueIndex::new(0),
        )],
        comparison_axioms: Vec::new(),
        assignment_axioms: Vec::new(),
        global_constraint,
    })
}

/// Replay `names` against a fresh registry over `task`.
fn replay(task: NumericRootTask, names: &[&str]) -> Replay {
    let arc: Arc<NumericRootTask> = Arc::new(task);
    let mut registry = StateRegistry::for_task(arc.clone());
    let operators: Vec<&Operator> = names
        .iter()
        .map(|name| {
            arc.get_operators()
                .iter()
                .find(|op| op.name() == *name)
                .unwrap_or_else(|| panic!("no operator named {name}"))
        })
        .collect();
    replay_plan(&*arc, &mut registry, arc.global_constraint(), &operators)
        .expect("replay machinery failed")
}

#[test]
fn valid_plan_is_verified_with_its_cost() {
    let result = replay(chain_task(), &["move_ab", "move_bc"]);
    assert_eq!(
        result.outcome,
        ReplayOutcome::Solved(VerifiedPlan {
            prefix_len: 2,
            cost: 2.0,
        })
    );
    // states[0] initial, states[1] after move_ab, states[2] after move_bc.
    assert_eq!(result.states.len(), 3);
    assert_eq!(result.applied, 2);
}

#[test]
fn padding_past_the_goal_returns_the_first_goal_reaching_prefix() {
    let result = replay(chain_task(), &["move_ab", "move_bc", "reset", "move_ab"]);
    // `reset` would leave the goal, but verification stops before it.
    assert_eq!(
        result.outcome,
        ReplayOutcome::Solved(VerifiedPlan {
            prefix_len: 2,
            cost: 2.0,
        })
    );
    assert_eq!(result.applied, 2, "must not apply operators past the goal");
}

#[test]
fn inapplicable_operator_reports_step_and_offending_fact() {
    let result = replay(chain_task(), &["move_bc"]);
    assert_eq!(
        result.outcome,
        ReplayOutcome::Rejected(PlanRejection::InapplicableOperator {
            step: 0,
            operator: String::from("move_bc"),
            fact: ExplicitFact::propositional(1, 1),
        })
    );
    assert!(!result.is_solved());
    assert_eq!(result.applied, 0);
}

#[test]
fn inapplicable_operator_mid_sequence_reports_the_earliest_failure() {
    // `move_ab` applies, then `move_ab` again does not.
    let result = replay(chain_task(), &["move_ab", "move_ab", "move_bc"]);
    assert_eq!(
        result.outcome,
        ReplayOutcome::Rejected(PlanRejection::InapplicableOperator {
            step: 1,
            operator: String::from("move_ab"),
            fact: ExplicitFact::propositional(1, 0),
        })
    );
    assert_eq!(result.applied, 1);
}

#[test]
fn applicable_sequence_missing_the_goal_is_rejected() {
    let result = replay(chain_task(), &["move_ab"]);
    assert_eq!(
        result.outcome,
        ReplayOutcome::Rejected(PlanRejection::GoalNotReached {
            unsatisfied: vec![ExplicitFact::propositional(1, 2)],
        })
    );
    assert_eq!(result.applied, 1);
}

#[test]
fn empty_sequence_is_rejected_when_the_initial_state_is_not_a_goal() {
    let result = replay(chain_task(), &[]);
    assert_eq!(
        result.outcome,
        ReplayOutcome::Rejected(PlanRejection::GoalNotReached {
            unsatisfied: vec![ExplicitFact::propositional(1, 2)],
        })
    );
    assert_eq!(result.states.len(), 1, "only the initial state is visited");
}

#[test]
fn empty_sequence_verifies_when_the_initial_state_is_already_a_goal() {
    // Start at the goal position.
    let result = replay(
        chain_task_with(
            ExplicitFact::propositional(0, 0),
            ExplicitValueIndex::new(2),
        ),
        &[],
    );
    assert_eq!(
        result.outcome,
        ReplayOutcome::Solved(VerifiedPlan {
            prefix_len: 0,
            cost: 0.0,
        })
    );
}

#[test]
fn violated_global_constraint_is_reported_separately() {
    // Demand the *negation* of the derived atom, which the axiom never
    // produces, so the constraint fails immediately in the initial state.
    let task = chain_task_with(
        ExplicitFact::propositional(0, 1),
        ExplicitValueIndex::new(0),
    );
    let result = replay(task, &["move_ab", "move_bc"]);
    assert_eq!(
        result.outcome,
        ReplayOutcome::Rejected(PlanRejection::GlobalConstraintViolated { step: 0 })
    );
}
