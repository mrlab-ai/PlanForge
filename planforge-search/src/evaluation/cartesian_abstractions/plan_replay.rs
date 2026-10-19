use super::*;

#[derive(Debug)]
pub(super) struct ConcretePlan {
    operator_ids: Vec<OperatorIndex>,
    cost: f64,
}

impl ConcretePlan {
    pub(super) fn cost(&self) -> f64 {
        self.cost
    }

    pub(super) fn into_operator_ids(self) -> Vec<OperatorIndex> {
        self.operator_ids
    }
}

pub(super) enum PlanCheck {
    ConcretePlan(ConcretePlan),
    AbstractDeadEnd(usize),
    Refine(Split),
}

pub(super) fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-7 * left.abs().max(right.abs()).max(1.0)
}

fn concrete_is_goal(
    semantics: &CartesianSemantics<'_>,
    state_packer: &StatePacker,
    propositions: &[u64],
) -> bool {
    (0..semantics.task().get_num_goals()).all(|goal_id| {
        fact_is_held(
            semantics.task().get_goal_fact(goal_id),
            state_packer,
            propositions,
        )
    })
}

/// A state `s_i` of the concrete trace of an abstract plan, together with the
/// abstract state `α(s_i)` it maps to.
pub(super) struct TraceState {
    pub(super) abstract_state: usize,
    pub(super) propositions: Vec<u64>,
    pub(super) prop_values: Vec<ExplicitValueIndex>,
    pub(super) numeric: Vec<NumericValue>,
}

/// The first flaw of a concrete trace, exposed by its last state `s_k`.
pub(super) enum TraceFlaw<'task> {
    /// `α(s_k)` is an abstract goal state, but `s_k` violates these goal facts.
    Goal(Vec<&'task ExplicitFact>),
    /// `s_k` violates these preconditions of the plan's next transition.
    Precondition {
        transition: TransitionKey,
        failed: Vec<&'task ExplicitFact>,
    },
    /// The plan's next transition leads `s_k` outside its target state.
    Deviation {
        transition: TransitionKey,
        successor_prop_values: Vec<ExplicitValueIndex>,
        successor_numeric: Vec<NumericValue>,
    },
}

/// A concrete trace `s_0, ..., s_k` that fails at its last state.
pub(super) struct FailedTrace<'task> {
    /// The operators `o_1, ..., o_k` leading to `s_k`.
    pub(super) operator_ids: Vec<OperatorIndex>,
    /// The state `s_k`.
    pub(super) last: TraceState,
    pub(super) flaw: TraceFlaw<'task>,
}

pub(super) enum Trace<'task> {
    ConcretePlan(ConcretePlan),
    AbstractDeadEnd(usize),
    Failed(FailedTrace<'task>),
}

/// Replays the optimal abstract plan from `refinement_root` until its first
/// flaw. When `prefix` is given, it receives the states `s_0, ..., s_{k-1}`
/// that precede the failing state `s_k`.
#[allow(clippy::too_many_arguments)]
pub(super) fn trace_abstract_plan<'s>(
    working: &WorkingAbstraction,
    semantics: &'s CartesianSemantics<'_>,
    shortest_paths: &ShortestPaths,
    state_packer: &Arc<StatePacker>,
    axiom_evaluator: &AxiomEvaluator<'_>,
    refinement_root: &ConcreteState,
    selected_plan: Option<&[TransitionKey]>,
    mut prefix: Option<&mut Vec<TraceState>>,
) -> Result<Trace<'s>> {
    let mut propositions = refinement_root.propositions.clone();
    let mut numeric = refinement_root.numeric.clone();
    let mut prop_values = Vec::new();
    let mut successor_propositions = Vec::new();
    let mut successor_numeric = Vec::new();
    let mut successor_prop_values = Vec::new();
    semantics.concrete_prop_values(state_packer, &propositions, &mut prop_values);
    let initial_abstract_state = working.hierarchy().map_state(&prop_values, &numeric)?;
    if !shortest_paths.distance(initial_abstract_state).is_finite() {
        return Ok(Trace::AbstractDeadEnd(initial_abstract_state));
    }
    let abstract_plan_cost = shortest_paths.distance(initial_abstract_state);
    let mut operator_ids: Vec<OperatorIndex> = Vec::new();
    let mut concrete_cost = 0.0;
    let mut selected_plan_pos = 0usize;

    loop {
        semantics.concrete_prop_values(state_packer, &propositions, &mut prop_values);
        let abstract_state = working.hierarchy().map_state(&prop_values, &numeric)?;
        let abstract_distance = shortest_paths.distance(abstract_state);
        ensure!(
            approximately_equal(concrete_cost + abstract_distance, abstract_plan_cost),
            "concrete trace left optimal abstract path: g={concrete_cost} h={abstract_distance} initial_h={abstract_plan_cost}"
        );

        if shortest_paths.is_goal(abstract_state) {
            ensure!(
                selected_plan.is_none_or(|plan| selected_plan_pos == plan.len()),
                "selected Cartesian plan reaches an abstract goal before its final transition"
            );
            if concrete_is_goal(semantics, state_packer, &propositions) {
                return Ok(Trace::ConcretePlan(ConcretePlan {
                    operator_ids,
                    cost: concrete_cost,
                }));
            }
            let failed_goals = (0..semantics.task().get_num_goals())
                .map(|goal_id| semantics.task().get_goal_fact(goal_id))
                .filter(|goal| !fact_is_held(goal, state_packer, &propositions))
                .collect::<Vec<_>>();
            ensure!(
                !failed_goals.is_empty(),
                "abstract goal contains a concrete non-goal without a failed goal fact"
            );
            return Ok(Trace::Failed(FailedTrace {
                operator_ids,
                last: TraceState {
                    abstract_state,
                    propositions,
                    prop_values,
                    numeric,
                },
                flaw: TraceFlaw::Goal(failed_goals),
            }));
        }

        ensure!(
            operator_ids.len() <= working.states().len(),
            "Cartesian generating transitions contain a cycle"
        );
        let transition = if let Some(plan) = selected_plan {
            let transition = *plan.get(selected_plan_pos).with_context(|| {
                format!("selected Cartesian plan ends in non-goal abstract state {abstract_state}")
            })?;
            ensure!(
                transition.source == abstract_state,
                "selected Cartesian plan expects source {}, concrete trace maps to {abstract_state}",
                transition.source
            );
            transition
        } else {
            shortest_paths
                .generating_transition(abstract_state)
                .context(
                    "non-goal Cartesian state with finite distance has no generating transition",
                )?
        };
        ensure!(
            working.contains_transition(transition),
            "Cartesian shortest path references missing transition {transition:?}"
        );
        let op_id = transition.concrete_op_id;
        let op = &semantics.task().get_operators()[op_id.index()];
        let failed_preconditions = op
            .preconditions()
            .iter()
            .filter(|fact| !fact_is_held(fact, state_packer, &propositions))
            .collect::<Vec<_>>();
        if !failed_preconditions.is_empty() {
            return Ok(Trace::Failed(FailedTrace {
                operator_ids,
                last: TraceState {
                    abstract_state,
                    propositions,
                    prop_values,
                    numeric,
                },
                flaw: TraceFlaw::Precondition {
                    transition,
                    failed: failed_preconditions,
                },
            }));
        }

        successor_propositions.clone_from(&propositions);
        successor_numeric.clone_from(&numeric);
        progress_concrete_state(
            op,
            axiom_evaluator,
            state_packer,
            &mut successor_propositions,
            &mut successor_numeric,
        )?;
        semantics.concrete_prop_values(
            state_packer,
            &successor_propositions,
            &mut successor_prop_values,
        );
        let concrete_target = working
            .hierarchy
            .map_state(&successor_prop_values, &successor_numeric)?;
        if concrete_target != transition.target {
            return Ok(Trace::Failed(FailedTrace {
                operator_ids,
                last: TraceState {
                    abstract_state,
                    propositions,
                    prop_values,
                    numeric,
                },
                flaw: TraceFlaw::Deviation {
                    transition,
                    successor_prop_values,
                    successor_numeric,
                },
            }));
        }

        let op_cost = semantics.operator_costs()[op_id.index()];
        ensure!(
            approximately_equal(
                op_cost.value() + shortest_paths.distance(transition.target),
                abstract_distance
            ),
            "Cartesian generating transition is not distance preserving"
        );
        if let Some(prefix) = prefix.as_deref_mut() {
            prefix.push(TraceState {
                abstract_state,
                propositions: std::mem::take(&mut propositions),
                prop_values: prop_values.clone(),
                numeric: std::mem::take(&mut numeric),
            });
        }
        std::mem::swap(&mut propositions, &mut successor_propositions);
        std::mem::swap(&mut numeric, &mut successor_numeric);
        concrete_cost += op_cost.value();
        operator_ids.push(op_id);
        selected_plan_pos += usize::from(selected_plan.is_some());
    }
}

/// The split that refines the first flaw of a failed trace, as progression
/// CEGAR chooses it.
pub(super) fn progression_split(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    failed: &FailedTrace<'_>,
) -> Result<Split> {
    let TraceState {
        abstract_state,
        prop_values,
        numeric,
        ..
    } = &failed.last;
    let abstract_state = *abstract_state;
    let use_desired_region_candidates = matches!(
        semantics.flaw_candidate_generation(),
        CartesianFlawCandidateGeneration::DesiredRegion
    );
    match &failed.flaw {
        TraceFlaw::Goal(failed_goals) => {
            let candidates = if use_desired_region_candidates {
                splits_for_desired_facts(
                    working,
                    semantics,
                    abstract_state,
                    failed_goals,
                    prop_values,
                    numeric,
                    "failed goal",
                )?
            } else {
                failed_goals
                    .iter()
                    .map(|goal| {
                        split_failed_fact(
                            working,
                            semantics,
                            abstract_state,
                            goal,
                            prop_values,
                            numeric,
                            format!("goal {goal:?}"),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
            };
            select_refinement_split(working, semantics, candidates, 0x474F_414C)
        }
        TraceFlaw::Precondition { transition, failed } => {
            let op_id = transition.concrete_op_id;
            let op = &semantics.task().get_operators()[op_id.index()];
            let candidates = if use_desired_region_candidates {
                splits_for_desired_facts(
                    working,
                    semantics,
                    abstract_state,
                    failed,
                    prop_values,
                    numeric,
                    &format!("operator {op_id} ({}) preconditions {failed:?}", op.name()),
                )?
            } else {
                failed
                    .iter()
                    .map(|fact| {
                        split_failed_fact(
                            working,
                            semantics,
                            abstract_state,
                            fact,
                            prop_values,
                            numeric,
                            format!("operator {op_id} ({}) precondition {fact:?}", op.name()),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
            };
            select_refinement_split(working, semantics, candidates, 0x5052_4543)
        }
        TraceFlaw::Deviation {
            transition,
            successor_prop_values,
            successor_numeric,
        } => split_deviation(
            working,
            semantics,
            DeviationWitness::new(
                abstract_state,
                transition.target,
                transition.concrete_op_id,
                successor_prop_values,
                numeric,
                successor_numeric,
            ),
        ),
    }
}

/// Progression flaw search: the first flaw of the optimal abstract plan's
/// concrete trace.
#[allow(clippy::too_many_arguments)]
pub(super) fn find_progression_flaw(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    shortest_paths: &ShortestPaths,
    state_packer: &Arc<StatePacker>,
    axiom_evaluator: &AxiomEvaluator<'_>,
    refinement_root: &ConcreteState,
    selected_plan: Option<&[TransitionKey]>,
) -> Result<PlanCheck> {
    Ok(
        match trace_abstract_plan(
            working,
            semantics,
            shortest_paths,
            state_packer,
            axiom_evaluator,
            refinement_root,
            selected_plan,
            None,
        )? {
            Trace::ConcretePlan(plan) => PlanCheck::ConcretePlan(plan),
            Trace::AbstractDeadEnd(state_id) => PlanCheck::AbstractDeadEnd(state_id),
            Trace::Failed(failed) => {
                PlanCheck::Refine(progression_split(working, semantics, &failed)?)
            }
        },
    )
}

/// Replays the current optimal abstract policy against one concrete execution
/// and ranks every witnessed flaw together. After a deviation, replay resumes
/// from the abstract state containing the real concrete successor. This keeps
/// every split tied to a concrete witness and avoids inventing projected states.
pub(super) fn replay_entire_optimal_abstract_trace(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    shortest_paths: &ShortestPaths,
    state_packer: &Arc<StatePacker>,
    axiom_evaluator: &AxiomEvaluator<'_>,
    refinement_root: &ConcreteState,
) -> Result<PlanCheck> {
    let mut propositions = refinement_root.propositions.clone();
    let mut numeric = refinement_root.numeric.clone();
    let mut prop_values = Vec::new();
    let mut successor_prop_values = Vec::new();
    let mut operator_ids = Vec::new();
    let mut concrete_cost = 0.0;
    let mut candidates = Vec::new();
    let mut candidate_identities = HashSet::new();
    let mut visited_abstract_states = HashSet::new();

    loop {
        semantics.concrete_prop_values(state_packer, &propositions, &mut prop_values);
        let abstract_state = working.hierarchy().map_state(&prop_values, &numeric)?;
        if !visited_abstract_states.insert(abstract_state) {
            ensure!(
                !candidates.is_empty(),
                "Cartesian whole-plan replay cycled without witnessing a flaw"
            );
            return select_refinement(working, semantics, candidates);
        }
        let abstract_distance = shortest_paths.distance(abstract_state);
        if !abstract_distance.is_finite() {
            return if candidates.is_empty() {
                Ok(PlanCheck::AbstractDeadEnd(abstract_state))
            } else {
                select_refinement(working, semantics, candidates)
            };
        }

        if shortest_paths.is_goal(abstract_state) {
            if concrete_is_goal(semantics, state_packer, &propositions) {
                return if candidates.is_empty() {
                    Ok(PlanCheck::ConcretePlan(ConcretePlan {
                        operator_ids,
                        cost: concrete_cost,
                    }))
                } else {
                    select_refinement(working, semantics, candidates)
                };
            }
            for goal_id in 0..semantics.task().get_num_goals() {
                let goal = semantics.task().get_goal_fact(goal_id);
                if !fact_is_held(goal, state_packer, &propositions) {
                    let split = split_failed_fact(
                        working,
                        semantics,
                        abstract_state,
                        goal,
                        &prop_values,
                        &numeric,
                        format!("goal {goal:?}"),
                    )?;
                    push_unique_split(&mut candidates, &mut candidate_identities, split);
                }
            }
            ensure!(
                !candidates.is_empty(),
                "abstract goal contains a concrete non-goal without a refinable failed goal fact"
            );
            return select_refinement(working, semantics, candidates);
        }

        let transition = shortest_paths
            .generating_transition(abstract_state)
            .context(
                "non-goal Cartesian state with finite distance has no generating transition",
            )?;
        ensure!(
            working.contains_transition(transition),
            "Cartesian shortest path references missing transition {transition:?}"
        );
        ensure!(
            approximately_equal(
                semantics.operator_costs()[transition.concrete_op_id.index()].value()
                    + shortest_paths.distance(transition.target),
                abstract_distance
            ),
            "Cartesian generating transition is not distance preserving"
        );

        let op_id = transition.concrete_op_id;
        let op = &semantics.task().get_operators()[op_id.index()];
        for failed in op
            .preconditions()
            .iter()
            .filter(|fact| !fact_is_held(fact, state_packer, &propositions))
        {
            let split = split_failed_fact(
                working,
                semantics,
                abstract_state,
                failed,
                &prop_values,
                &numeric,
                format!("operator {op_id} ({}) precondition {failed:?}", op.name()),
            )?;
            push_unique_split(&mut candidates, &mut candidate_identities, split);
        }

        let source_numeric = numeric.clone();
        progress_concrete_state(
            op,
            axiom_evaluator,
            state_packer,
            &mut propositions,
            &mut numeric,
        )?;
        semantics.concrete_prop_values(state_packer, &propositions, &mut successor_prop_values);
        let concrete_target = working
            .hierarchy
            .map_state(&successor_prop_values, &numeric)?;
        if concrete_target != transition.target {
            for split in split_deviation_candidates(
                working,
                semantics,
                DeviationWitness::new(
                    abstract_state,
                    transition.target,
                    op_id,
                    &successor_prop_values,
                    &source_numeric,
                    &numeric,
                ),
            )? {
                push_unique_split(&mut candidates, &mut candidate_identities, split);
            }
        }

        concrete_cost += semantics.operator_costs()[op_id.index()].value();
        operator_ids.push(op_id);
    }
}

pub(super) fn validate_concrete_plan(
    semantics: &CartesianSemantics<'_>,
    state_packer: &Arc<StatePacker>,
    axiom_evaluator: &AxiomEvaluator<'_>,
    refinement_root: &ConcreteState,
    plan: &ConcretePlan,
) -> Result<()> {
    let mut propositions = refinement_root.propositions.clone();
    let mut numeric = refinement_root.numeric.clone();
    let mut cost = 0.0;
    for (step, &op_id) in plan.operator_ids.iter().enumerate() {
        let op = semantics
            .task()
            .get_operators()
            .get(op_id.index())
            .with_context(|| format!("concrete plan step {step} has invalid operator {op_id}"))?;
        for precondition in op.preconditions() {
            ensure!(
                fact_is_held(precondition, state_packer, &propositions),
                "concrete plan operator {op_id} ({}) has false precondition {precondition:?} at step {step}",
                op.name()
            );
        }
        progress_concrete_state(
            op,
            axiom_evaluator,
            state_packer,
            &mut propositions,
            &mut numeric,
        )?;
        cost += semantics.operator_costs()[op_id.index()].value();
    }
    ensure!(
        concrete_is_goal(semantics, state_packer, &propositions),
        "replayed Cartesian concrete plan does not satisfy the full goal"
    );
    ensure!(
        approximately_equal(cost, plan.cost),
        "replayed Cartesian concrete plan cost {cost} differs from recorded cost {}",
        plan.cost
    );
    Ok(())
}
