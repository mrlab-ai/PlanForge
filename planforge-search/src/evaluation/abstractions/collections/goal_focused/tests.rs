use std::cell::RefCell;

use planforge_sas::axioms::{ComparisonAxiom, ComparisonOperator};
use planforge_sas::numeric_conditions::ConditionValue;
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, ExplicitFact, ExplicitValueIndex, ExplicitVariable,
    Metric, NumericRootTask, NumericRootTaskParts, NumericType, NumericValue, NumericVariable,
    Operator, OperatorCost, VariableIndex,
};

use super::*;

/// One goal of the counter task: `x >= bound` or `x <= bound`.
#[derive(Debug, Clone, Copy)]
enum Bound {
    AtLeast(f64),
    AtMost(f64),
}

/// A counter `x`, starting at 0, with one `increment` operator and one goal
/// per bound.
fn counter_task(bounds: &[Bound]) -> NumericRootTask {
    NumericRootTask::new(counter_task_parts(bounds))
}

fn counter_task_parts(bounds: &[Bound]) -> NumericRootTaskParts {
    let variables = bounds
        .iter()
        .enumerate()
        .map(|(goal_id, _)| {
            ExplicitVariable::new(
                ConditionValue::DOMAIN_SIZE,
                format!("goal-{goal_id}"),
                vec!["true".into(), "false".into()],
                Some(0),
                ExplicitValueIndex::from_usize(ConditionValue::False.as_usize()),
            )
        })
        .collect::<Vec<_>>();
    let mut numeric_variables = vec![
        NumericVariable::new("x".into(), NumericType::Regular, None),
        NumericVariable::new("one".into(), NumericType::Constant, None),
    ];
    let mut numeric_state = vec![NumericValue::new(0.0), NumericValue::new(1.0)];
    let mut comparison_axioms = Vec::new();
    let mut state = Vec::new();
    for (goal_id, bound) in bounds.iter().enumerate() {
        let (value, operator, holds) = match *bound {
            Bound::AtLeast(value) => (value, ComparisonOperator::GreaterThanOrEqual, value <= 0.0),
            Bound::AtMost(value) => (value, ComparisonOperator::LessThanOrEqual, value >= 0.0),
        };
        numeric_variables.push(NumericVariable::new(
            format!("bound-{goal_id}"),
            NumericType::Constant,
            None,
        ));
        numeric_state.push(NumericValue::new(value));
        comparison_axioms.push(ComparisonAxiom::new(
            VariableIndex::from_usize(goal_id),
            VariableIndex::new(0),
            VariableIndex::from_usize(goal_id + 2),
            operator,
        ));
        state.push(ExplicitValueIndex::from_usize(usize::from(!holds)));
    }
    NumericRootTaskParts {
        version: 2,
        metric: Metric::new(true, None),
        variables,
        numeric_variables,
        goals: (0..bounds.len())
            .map(|goal_id| ExplicitFact::propositional(goal_id, 0))
            .collect(),
        mutexes: vec![],
        state,
        numeric_state,
        operators: vec![Operator::new(
            "increment".into(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                VariableIndex::new(0),
                AssignmentOperation::Plus,
                VariableIndex::new(1),
                false,
                vec![],
            )],
            OperatorCost::new(1),
        )],
        axioms: vec![],
        comparison_axioms,
        assignment_axioms: vec![],
        global_constraint: ExplicitFact::propositional(0, 0),
    }
}

/// What one `build` call was asked for.
#[derive(Debug, Clone, PartialEq)]
struct Call {
    goal_id: Option<usize>,
    variant_id: usize,
    kind: MemberKind,
    direction: Option<MemberDirection>,
    tie_rank: Option<usize>,
    seed: Option<u64>,
    /// `x` at the refinement root, `None` for the task initial state.
    root_x: Option<f64>,
    remaining_states: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct Member {
    call: Call,
    membership: Option<CollectionMembership>,
}

/// A backend that solves the counter goals exactly: the plan from root `x`
/// increments until the bound holds, `x <= bound` from above the bound is a
/// dead end, and `h(s0)` is the plan length from `x = 0` -- or a scripted
/// value per goal.
struct ScriptedBackend {
    bounds: Vec<Bound>,
    initial_h: Option<Vec<f64>>,
    states_per_member: usize,
    /// Calls (by index) that stop refining before finding a concrete plan.
    unsolved_calls: Vec<usize>,
    calls: RefCell<Vec<Call>>,
}

impl ScriptedBackend {
    fn new(bounds: &[Bound]) -> Self {
        Self {
            bounds: bounds.to_vec(),
            initial_h: None,
            states_per_member: 2,
            unsolved_calls: Vec::new(),
            calls: RefCell::new(Vec::new()),
        }
    }

    /// Increments from `x` to the goal, `None` for a dead end.
    fn plan_length(&self, goal_id: usize, x: f64) -> Option<usize> {
        match self.bounds[goal_id] {
            Bound::AtLeast(bound) => Some((bound - x).max(0.0) as usize),
            Bound::AtMost(bound) => (x <= bound).then_some(0),
        }
    }
}

impl GoalFocusedBackend for ScriptedBackend {
    type Member = Member;

    const NAME: &'static str = "scripted";

    fn build(
        &self,
        task: &dyn AbstractNumericTask,
        root: Option<&ConcreteState>,
        variant: &GoalVariant,
        budget: MemberBudget,
    ) -> Result<(Member, BuiltReport)> {
        let call = Call {
            goal_id: variant.goal_id,
            variant_id: variant.variant_id,
            kind: variant.kind,
            direction: variant.direction,
            tie_rank: variant.tie_rank,
            seed: variant.seed,
            root_x: root.map(|root| root.numeric[0].value()),
            remaining_states: budget.remaining_states,
        };
        let call_index = self.calls.borrow().len();
        self.calls.borrow_mut().push(call.clone());
        let Some(goal_id) = variant.goal_id else {
            assert_eq!(task.get_num_goals(), 0, "only a goal-free task has no goal");
            let report = BuiltReport {
                initial_h: 0.0,
                num_states: 1,
                concrete_plan: Some(Vec::new()),
                root_is_dead_end: false,
            };
            return Ok((
                Member {
                    call,
                    membership: None,
                },
                report,
            ));
        };
        let goals = if variant.full_task {
            assert_eq!(
                task.get_num_goals(),
                self.bounds.len(),
                "full-task members see every goal"
            );
            (0..self.bounds.len()).collect::<Vec<_>>()
        } else {
            assert_eq!(
                task.get_num_goals(),
                1,
                "members see their single-goal task"
            );
            vec![goal_id]
        };
        let plan_length = |x: f64| {
            goals
                .iter()
                .map(|&goal| self.plan_length(goal, x))
                .collect::<Option<Vec<_>>>()
                .map(|lengths| lengths.into_iter().max().expect("a member has a goal"))
        };
        let length = plan_length(call.root_x.unwrap_or(0.0));
        let initial_h = match &self.initial_h {
            Some(initial_h) => initial_h[goal_id],
            None => plan_length(0.0).map_or(f64::INFINITY, |length| length as f64),
        };
        let report = BuiltReport {
            initial_h,
            num_states: self.states_per_member.min(budget.remaining_states),
            concrete_plan: length
                .filter(|_| !self.unsolved_calls.contains(&call_index))
                .map(|length| vec![OperatorIndex::new(0); length]),
            root_is_dead_end: length.is_none(),
        };
        Ok((
            Member {
                call,
                membership: None,
            },
            report,
        ))
    }

    fn record_membership(&self, member: &mut Member, membership: Option<CollectionMembership>) {
        member.membership = membership;
    }
}

fn schedule(
    strategy: CollectionStrategy,
    variants_per_goal: usize,
    progressive_goal_roots: bool,
) -> GoalFocusedSchedule {
    GoalFocusedSchedule {
        strategy,
        variants_per_goal,
        random_seed: Some(5),
        max_collection_states: 1_000,
        total_max_time: None,
        progressive_goal_roots,
        alternate_split_kinds: true,
        single_goal_tasks: true,
    }
}

/// `(goal, variant, kind, root x, progressive root)` per member.
fn lanes(members: &[Member]) -> Vec<(usize, usize, MemberKind, Option<f64>, bool)> {
    members
        .iter()
        .map(|member| {
            let membership = member.membership.expect("goal tasks tag every member");
            assert_eq!(Some(membership.goal_id), member.call.goal_id);
            assert_eq!(membership.variant_id, member.call.variant_id);
            (
                membership.goal_id,
                membership.variant_id,
                member.call.kind,
                member.call.root_x,
                membership.progressive_root,
            )
        })
        .collect()
}

#[test]
fn goals_round_robin_before_focusing_the_strongest() {
    let mut counts = vec![0, 0, 0];
    let strengths = vec![2.0, 5.0, 3.0];
    let mut selected = Vec::new();
    for _ in 0..9 {
        let goal = select_next_collection_goal(&counts, &strengths, 3).unwrap();
        selected.push(goal);
        counts[goal] += 1;
    }

    assert_eq!(selected, vec![0, 1, 2, 0, 1, 2, 1, 2, 0]);
    assert_eq!(counts, vec![3, 3, 3]);
    assert_eq!(select_next_collection_goal(&counts, &strengths, 3), None);
}

#[test]
fn complementary_variants_alternate_direction_and_rank_ties_by_pairs() {
    let bounds = [
        Bound::AtLeast(1.0),
        Bound::AtLeast(1.0),
        Bound::AtLeast(1.0),
    ];
    let task = counter_task(&bounds);
    let mut backend = ScriptedBackend::new(&bounds);
    backend.initial_h = Some(vec![2.0, 5.0, 3.0]);
    let members = build_goal_focused_collection(
        &task,
        &schedule(CollectionStrategy::Complementary, 4, false),
        &backend,
    )
    .unwrap();

    let calls = members
        .iter()
        .map(|member| {
            (
                member.call.goal_id.unwrap(),
                member.call.variant_id,
                member.call.direction.unwrap(),
                member.call.tie_rank.unwrap(),
                member.call.seed,
            )
        })
        .collect::<Vec<_>>();
    let seed = |goal_id, variant_id: usize| Some(derive_variant_seed(5, goal_id, variant_id - 1));
    use MemberDirection::{Progression, Regression};
    assert_eq!(
        calls,
        vec![
            (0, 0, Progression, 0, None),
            (1, 0, Progression, 0, None),
            (2, 0, Progression, 0, None),
            (0, 1, Regression, 0, seed(0, 1)),
            (1, 1, Regression, 0, seed(1, 1)),
            (2, 1, Regression, 0, seed(2, 1)),
            (1, 2, Progression, 1, seed(1, 2)),
            (1, 3, Regression, 1, seed(1, 3)),
            (2, 2, Progression, 1, seed(2, 2)),
            (2, 3, Regression, 1, seed(2, 3)),
            (0, 2, Progression, 1, seed(0, 2)),
            (0, 3, Regression, 1, seed(0, 3)),
        ]
    );
    assert!(members.iter().all(|member| member.call.root_x.is_none()));
}

#[test]
fn standard_variants_vary_only_the_seed() {
    let bounds = [Bound::AtLeast(1.0)];
    let task = counter_task(&bounds);
    let backend = ScriptedBackend::new(&bounds);
    let members = build_goal_focused_collection(
        &task,
        &schedule(CollectionStrategy::Standard, 2, false),
        &backend,
    )
    .unwrap();

    assert_eq!(
        members
            .iter()
            .map(|member| (
                member.call.direction,
                member.call.tie_rank,
                member.call.seed
            ))
            .collect::<Vec<_>>(),
        vec![
            (None, None, Some(5)),
            (None, None, Some(derive_variant_seed(5, 0, 0))),
        ]
    );
}

#[test]
fn a_goal_free_task_gets_one_full_task_member() {
    let task = NumericRootTask::new(NumericRootTaskParts {
        goals: Vec::new(),
        ..counter_task_parts(&[Bound::AtLeast(1.0)])
    });
    let backend = ScriptedBackend::new(&[]);
    let members = build_goal_focused_collection(
        &task,
        &schedule(CollectionStrategy::Complementary, 3, true),
        &backend,
    )
    .unwrap();

    assert_eq!(
        members,
        vec![Member {
            call: Call {
                goal_id: None,
                variant_id: 0,
                kind: MemberKind::Scheduled,
                direction: None,
                tie_rank: None,
                seed: Some(5),
                root_x: None,
                remaining_states: 1_000,
            },
            membership: None,
        }]
    );
}

#[test]
fn lanes_advance_through_concrete_plans_and_add_initial_root_specialists() {
    let bounds = [Bound::AtLeast(2.0), Bound::AtLeast(4.0)];
    let task = counter_task(&bounds);
    let backend = ScriptedBackend::new(&bounds);
    let members = build_goal_focused_collection(
        &task,
        &schedule(CollectionStrategy::Standard, 1, true),
        &backend,
    )
    .unwrap();

    assert_eq!(
        lanes(&members),
        vec![
            (0, 0, MemberKind::Scheduled, Some(0.0), false),
            (1, 0, MemberKind::Scheduled, Some(2.0), true),
            (1, 1, MemberKind::InitialRootSpecialist, None, false),
        ]
    );
}

#[test]
fn a_lane_reaching_the_full_goal_becomes_terminal() {
    let bounds = [Bound::AtLeast(4.0), Bound::AtLeast(2.0)];
    let task = counter_task(&bounds);
    let backend = ScriptedBackend::new(&bounds);
    let members = build_goal_focused_collection(
        &task,
        &schedule(CollectionStrategy::Standard, 1, true),
        &backend,
    )
    .unwrap();

    assert_eq!(
        lanes(&members),
        vec![
            (0, 0, MemberKind::Scheduled, Some(0.0), false),
            (1, 0, MemberKind::Scheduled, None, false),
        ]
    );
}

#[test]
fn a_dead_lane_root_is_rebuilt_from_the_initial_state_and_ends_the_lane() {
    let bounds = [Bound::AtLeast(1.0), Bound::AtMost(0.0), Bound::AtLeast(3.0)];
    let task = counter_task(&bounds);
    let backend = ScriptedBackend::new(&bounds);
    let members = build_goal_focused_collection(
        &task,
        &schedule(CollectionStrategy::Standard, 1, true),
        &backend,
    )
    .unwrap();

    assert_eq!(
        lanes(&members),
        vec![
            (0, 0, MemberKind::Scheduled, Some(0.0), false),
            // Built from the dead root x=1, then rebuilt from the initial state.
            (1, 0, MemberKind::Scheduled, None, false),
            // The lane is terminal: later members start from the initial state.
            (2, 0, MemberKind::Scheduled, None, false),
        ]
    );
    assert_eq!(
        backend
            .calls
            .borrow()
            .iter()
            .map(|call| (call.goal_id, call.root_x))
            .collect::<Vec<_>>(),
        vec![
            (Some(0), Some(0.0)),
            (Some(1), Some(1.0)),
            (Some(1), None),
            (Some(2), None),
        ]
    );
}

#[test]
fn a_member_whose_initial_state_is_dead_is_kept() {
    // `x <= -1` is unreachable from x=0 by increments: the member proves the
    // task unsolvable, so the collection keeps it instead of rebuilding.
    let bounds = [Bound::AtLeast(1.0), Bound::AtMost(-1.0)];
    let task = counter_task(&bounds);
    let backend = ScriptedBackend::new(&bounds);
    let members = build_goal_focused_collection(
        &task,
        &schedule(CollectionStrategy::Standard, 1, false),
        &backend,
    )
    .unwrap();

    assert!(
        members.iter().any(|member| member.call.goal_id == Some(1)),
        "the dead goal's member is part of the collection"
    );
    assert!(
        backend
            .calls
            .borrow()
            .iter()
            .all(|call| call.root_x.is_none()),
        "a dead initial state is never rebuilt"
    );
}

#[test]
fn an_advanced_lane_retries_goals_it_reopened() {
    let bounds = [Bound::AtLeast(4.0), Bound::AtLeast(2.0)];
    let task = counter_task(&bounds);
    let mut backend = ScriptedBackend::new(&bounds);
    backend.unsolved_calls = vec![0];
    let members = build_goal_focused_collection(
        &task,
        &schedule(CollectionStrategy::Standard, 1, true),
        &backend,
    )
    .unwrap();

    assert_eq!(
        lanes(&members),
        vec![
            (0, 0, MemberKind::Scheduled, Some(0.0), false),
            (1, 0, MemberKind::Scheduled, Some(0.0), false),
            (0, 0, MemberKind::Continuation, Some(2.0), true),
        ]
    );
}

#[test]
fn an_invalid_concrete_plan_is_an_error() {
    let bounds = [Bound::AtLeast(2.0), Bound::AtLeast(3.0)];
    let task = counter_task(&bounds);

    /// Reports a plan one increment short of its goal.
    struct ShortPlans(ScriptedBackend);
    impl GoalFocusedBackend for ShortPlans {
        type Member = Member;
        const NAME: &'static str = "short";
        fn build(
            &self,
            task: &dyn AbstractNumericTask,
            root: Option<&ConcreteState>,
            variant: &GoalVariant,
            budget: MemberBudget,
        ) -> Result<(Member, BuiltReport)> {
            let (member, mut report) = self.0.build(task, root, variant, budget)?;
            report.concrete_plan.as_mut().unwrap().pop();
            Ok((member, report))
        }
        fn record_membership(&self, member: &mut Member, membership: Option<CollectionMembership>) {
            self.0.record_membership(member, membership);
        }
    }

    let error = build_goal_focused_collection(
        &task,
        &schedule(CollectionStrategy::Standard, 1, true),
        &ShortPlans(ScriptedBackend::new(&bounds)),
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("does not reach its goal"),
        "{error:#}"
    );
}

#[test]
fn members_share_the_collection_state_budget() {
    let bounds = [Bound::AtLeast(1.0), Bound::AtLeast(2.0)];
    let task = counter_task(&bounds);
    let mut backend = ScriptedBackend::new(&bounds);
    backend.states_per_member = 3;
    let members = build_goal_focused_collection(
        &task,
        &GoalFocusedSchedule {
            max_collection_states: 8,
            ..schedule(CollectionStrategy::Complementary, 3, false)
        },
        &backend,
    )
    .unwrap();

    // 8 -> 5 -> 2 -> stop below 2 after the member that used the last 2.
    assert_eq!(
        members
            .iter()
            .map(|member| member.call.remaining_states)
            .collect::<Vec<_>>(),
        vec![8, 5, 2]
    );
}

#[test]
fn an_exhausted_time_budget_still_builds_the_first_member() {
    let bounds = [Bound::AtLeast(1.0), Bound::AtLeast(2.0)];
    let task = counter_task(&bounds);
    let backend = ScriptedBackend::new(&bounds);
    let members = build_goal_focused_collection(
        &task,
        &GoalFocusedSchedule {
            total_max_time: Some(Duration::ZERO),
            ..schedule(CollectionStrategy::Complementary, 3, false)
        },
        &backend,
    )
    .unwrap();

    assert_eq!(members.len(), 1);
}

#[test]
fn concrete_states_replay_and_check_goals() {
    let task = counter_task(&[Bound::AtLeast(2.0), Bound::AtMost(1.0)]);
    let initial = ConcreteState::initial(&task).unwrap();
    assert_eq!(initial.satisfied_goal_count(&task).unwrap(), 1);
    assert!(initial.satisfies_goal(&task, 1).unwrap());

    let advanced = initial.replay(&task, &[OperatorIndex::new(0); 2]).unwrap();
    assert_eq!(advanced.numeric[0], NumericValue::new(2.0));
    assert!(advanced.satisfies_goal(&task, 0).unwrap());
    assert!(!advanced.satisfies_goal(&task, 1).unwrap());
    assert_eq!(
        advanced.propositional_values(&task).unwrap(),
        vec![
            ExplicitValueIndex::from_usize(ConditionValue::True.as_usize()),
            ExplicitValueIndex::from_usize(ConditionValue::False.as_usize())
        ]
    );
    assert!(
        initial.replay(&task, &[OperatorIndex::new(1)]).is_err(),
        "operator 1 does not exist"
    );
    assert!(
        initial.satisfies_goal(&task, 2).is_err(),
        "goal 2 does not exist"
    );
}

#[test]
fn without_alternation_every_variant_keeps_the_configured_direction() {
    let bounds = [Bound::AtLeast(1.0), Bound::AtLeast(2.0)];
    let task = counter_task(&bounds);
    let backend = ScriptedBackend::new(&bounds);
    let members = build_goal_focused_collection(
        &task,
        &GoalFocusedSchedule {
            alternate_split_kinds: false,
            ..schedule(CollectionStrategy::Complementary, 3, false)
        },
        &backend,
    )
    .unwrap();

    let seed = |goal_id, variant_id: usize| Some(derive_variant_seed(5, goal_id, variant_id - 1));
    assert_eq!(
        members
            .iter()
            .map(|member| (
                member.call.goal_id.unwrap(),
                member.call.variant_id,
                member.call.direction,
                member.call.tie_rank,
                member.call.seed
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 0, None, Some(0), None),
            (1, 0, None, Some(0), None),
            (0, 1, None, Some(0), seed(0, 1)),
            (1, 1, None, Some(0), seed(1, 1)),
            (1, 2, None, Some(1), seed(1, 2)),
            (0, 2, None, Some(1), seed(0, 2)),
        ]
    );
}

#[test]
fn full_task_members_are_scheduled_round_robin_without_membership() {
    let bounds = [
        Bound::AtLeast(1.0),
        Bound::AtLeast(3.0),
        Bound::AtLeast(2.0),
    ];
    let task = counter_task(&bounds);
    let mut backend = ScriptedBackend::new(&bounds);
    // Unequal scripted h(s0) values would reorder single-goal members after
    // the first two rounds; full-task members ignore them.
    backend.initial_h = Some(vec![2.0, 5.0, 3.0]);
    let members = build_goal_focused_collection(
        &task,
        &GoalFocusedSchedule {
            single_goal_tasks: false,
            ..schedule(CollectionStrategy::Complementary, 3, false)
        },
        &backend,
    )
    .unwrap();

    let seed = |goal_id, variant_id: usize| Some(derive_variant_seed(5, goal_id, variant_id - 1));
    use MemberDirection::{Progression, Regression};
    assert_eq!(
        members
            .iter()
            .map(|member| (
                member.call.goal_id.unwrap(),
                member.call.variant_id,
                member.call.direction.unwrap(),
                member.call.tie_rank.unwrap(),
                member.call.seed,
                member.membership,
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 0, Progression, 0, None, None),
            (1, 0, Progression, 0, None, None),
            (2, 0, Progression, 0, None, None),
            (0, 1, Regression, 0, seed(0, 1), None),
            (1, 1, Regression, 0, seed(1, 1), None),
            (2, 1, Regression, 0, seed(2, 1), None),
            (0, 2, Progression, 1, seed(0, 2), None),
            (1, 2, Progression, 1, seed(1, 2), None),
            (2, 2, Progression, 1, seed(2, 2), None),
        ]
    );
    assert!(members.iter().all(|member| member.call.root_x.is_none()));
}

#[test]
#[should_panic(expected = "full-task members require")]
fn full_task_members_reject_progressive_goal_roots() {
    let bounds = [Bound::AtLeast(1.0)];
    let task = counter_task(&bounds);
    let _ = build_goal_focused_collection(
        &task,
        &GoalFocusedSchedule {
            single_goal_tasks: false,
            ..schedule(CollectionStrategy::Complementary, 2, true)
        },
        &ScriptedBackend::new(&bounds),
    );
}
