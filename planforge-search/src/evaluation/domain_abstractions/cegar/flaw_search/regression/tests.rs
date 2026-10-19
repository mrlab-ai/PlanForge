use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitValueIndex, ExplicitVariable, Metric,
    NumericRootTask, NumericRootTaskParts, NumericType, NumericVariable, Operator, OperatorCost,
};

use super::*;
use crate::evaluation::domain_abstractions::cegar::flaw_search::get_progression_flaws;
use crate::evaluation::domain_abstractions::cegar::flaw_search::single_switch_task;
use crate::evaluation::domain_abstractions::domain_abstraction_factory::DomainAbstractionFactory;
use crate::evaluation::domain_abstractions::utils::identity_domain_mapping_and_sizes;

const GOAL: usize = 0;
const JUMP_PRECONDITION: usize = 1;

/// `x` starts at 0 and the goal is `x >= 3`. `jump` increments `x` but
/// requires `x >= 1`. The plan `<jump>` fails its precondition at the
/// initial state, and its goal fails after `jump`, which regresses to
/// `x >= 2`.
fn jump_task() -> NumericRootTask {
    let comparison_variable = |name: &str| {
        ExplicitVariable::new(
            ConditionValue::DOMAIN_SIZE,
            name.into(),
            vec!["true".into(), "false".into()],
            Some(0),
            ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
        )
    };
    NumericRootTask::new(NumericRootTaskParts {
        version: 4,
        metric: Metric::new(true, None),
        variables: vec![
            comparison_variable("x-at-least-three"),
            comparison_variable("x-at-least-one"),
        ],
        numeric_variables: vec![
            NumericVariable::new("x".into(), NumericType::Regular, None),
            NumericVariable::new("one".into(), NumericType::Constant, None),
            NumericVariable::new("three".into(), NumericType::Constant, None),
        ],
        goals: vec![ExplicitFact::propositional(
            GOAL,
            ConditionValue::True.as_usize(),
        )],
        mutexes: vec![],
        state: vec![
            ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
            ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
        ],
        numeric_state: vec![
            NumericValue::new(0.0),
            NumericValue::new(1.0),
            NumericValue::new(3.0),
        ],
        operators: vec![Operator::new(
            "jump".into(),
            vec![ExplicitFact::propositional(
                JUMP_PRECONDITION,
                ConditionValue::True.as_usize(),
            )],
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
        comparison_axioms: vec![
            ComparisonAxiom::new(
                VariableIndex::from_usize(GOAL),
                VariableIndex::from_usize(0),
                VariableIndex::from_usize(2),
                ComparisonOperator::GreaterThanOrEqual,
            ),
            ComparisonAxiom::new(
                VariableIndex::from_usize(JUMP_PRECONDITION),
                VariableIndex::from_usize(0),
                VariableIndex::from_usize(1),
                ComparisonOperator::GreaterThanOrEqual,
            ),
        ],
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    })
}

/// The plan `<jump>` in the trivial abstraction, whose partitions all have
/// id 0.
fn jump_plan() -> WildcardPlanResult {
    WildcardPlanResult {
        wildcard_plan: vec![vec![OperatorIndex::new(0)]],
        abstract_state_hashes: vec![],
        abstract_prop_states: vec![],
        abstract_numeric_states: vec![vec![0; 3]; 2],
    }
}

fn regression_flaws(direction: SplitDirection) -> Vec<Flaw> {
    let task = jump_task();
    let (domain_mapping, _) = identity_domain_mapping_and_sizes(&task).unwrap();
    get_regression_flaws(
        &task,
        &NumericPartitions::trivial(&task),
        &domain_mapping,
        &jump_plan(),
        direction,
    )
    .unwrap()
    .flaws
}

#[test]
fn regression_regresses_the_goal_to_the_initial_state_at_its_concrete_value() {
    let flaws = regression_flaws(SplitDirection::Forward);
    assert_eq!(
        flaws,
        [Flaw::Propositional(PropFlaw {
            fact: ExplicitFact::propositional(GOAL, ConditionValue::True.as_usize()),
            dependent_numeric_flaws: vec![NumericFlaw {
                numeric_var_id: VariableIndex::new(0),
                value: NumericValue::new(0.0),
                include_in_lower: true,
                step: 1,
            }],
            step: 1,
        })],
        "x <= 0 progresses through jump to x <= 1, which violates the goal x >= 3"
    );
}

#[test]
fn regression_regresses_the_goal_to_the_initial_state_at_its_wanted_values() {
    let flaws = regression_flaws(SplitDirection::Backward);
    assert_eq!(
        flaws,
        [Flaw::Propositional(PropFlaw {
            fact: ExplicitFact::propositional(GOAL, ConditionValue::True.as_usize()),
            dependent_numeric_flaws: vec![NumericFlaw {
                numeric_var_id: VariableIndex::new(0),
                value: NumericValue::new(2.0),
                include_in_lower: false,
                step: 1,
            }],
            step: 1,
        })],
        "the goal x >= 3 regresses through jump to the wanted values x >= 2"
    );
}

#[test]
fn progression_splits_for_the_precondition_instead() {
    let task = jump_task();
    let flaws = get_progression_flaws(
        &task,
        &NumericPartitions::trivial(&task),
        &jump_plan(),
        SplitDirection::Forward,
    )
    .unwrap()
    .flaws;
    assert!(
        matches!(
            flaws.as_slice(),
            [Flaw::Propositional(PropFlaw { fact, .. })]
                if *fact == ExplicitFact::propositional(JUMP_PRECONDITION, ConditionValue::True.as_usize())
        ),
        "got {flaws:?}"
    );
}

#[test]
fn regression_equals_progression_when_only_the_goal_fails() {
    let task = single_switch_task(
        3,
        ExplicitValueIndex::new(2),
        vec![ExplicitValueIndex::new(0)],
    );
    let (mut domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
    // The abstraction cannot tell `v=1` from `v=2`, so `set` reaches an
    // abstract goal state although it sets `v=1`.
    domain_mapping[0] = vec![
        ExplicitValueIndex::new(0),
        ExplicitValueIndex::new(1),
        ExplicitValueIndex::new(1),
    ];
    let factory = DomainAbstractionFactory::new(
        &task,
        domain_mapping,
        domain_sizes,
        NumericPartitions::trivial(&task),
        vec![],
    )
    .unwrap();
    let plan = factory
        .compute_wildcard_plan(&task, true, false)
        .unwrap()
        .expect("plan exists");

    let regression = get_regression_flaws(
        &task,
        factory.partitions(),
        factory.domain_mapping(),
        &plan,
        SplitDirection::Forward,
    )
    .unwrap();
    let progression =
        get_progression_flaws(&task, factory.partitions(), &plan, SplitDirection::Forward).unwrap();
    assert_eq!(regression, progression);
    assert!(matches!(
        regression.flaws.as_slice(),
        [Flaw::Propositional(PropFlaw { fact, .. })] if *fact == ExplicitFact::propositional(0, 2)
    ));
}
