use super::*;

#[derive(Debug)]
pub(super) struct ConcretePlan {
    operator_ids: Vec<usize>,
    cost: f64,
}

impl ConcretePlan {
    pub(super) fn cost(&self) -> f64 {
        self.cost
    }

    pub(super) fn into_operator_ids(self) -> Vec<usize> {
        self.operator_ids
    }
}

pub(super) enum PlanCheck {
    ConcretePlan(ConcretePlan),
    AbstractDeadEnd(usize),
    Refine(Split),
}

#[derive(Debug)]
pub(super) struct RefinementRootDeadEnd {
    abstract_state_id: usize,
}

impl RefinementRootDeadEnd {
    pub(super) fn new(abstract_state_id: usize) -> Self {
        Self { abstract_state_id }
    }
}

impl std::fmt::Display for RefinementRootDeadEnd {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "concrete refinement root maps to abstract dead end {}",
            self.abstract_state_id
        )
    }
}

impl std::error::Error for RefinementRootDeadEnd {}

pub(super) fn approximately_equal(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-7 * left.abs().max(right.abs()).max(1.0)
}

fn concrete_is_goal(
    semantics: &CartesianSemantics<'_>,
    state_packer: &StatePacker,
    propositions: &[u64],
) -> bool {
    (0..semantics.task().get_num_goals()).all(|goal_id| {
        fact_is_hold(
            semantics.task().get_goal_fact(goal_id),
            state_packer,
            propositions,
        )
    })
}

pub(super) fn replay_optimal_abstract_trace(
    working: &WorkingAbstraction,
    semantics: &CartesianSemantics<'_>,
    shortest_paths: &ShortestPaths,
    state_packer: &Arc<StatePacker>,
    axiom_evaluator: &AxiomEvaluator<'_>,
    refinement_root: &CartesianConcreteState,
    selected_plan: Option<&[TransitionKey]>,
) -> Result<PlanCheck> {
    let use_desired_region_candidates = matches!(
        semantics.flaw_candidate_generation(),
        CartesianFlawCandidateGeneration::DesiredRegion
    );
    let mut propositions = refinement_root.propositions.clone();
    let mut numeric = refinement_root.numeric.clone();
    let mut prop_values = Vec::new();
    let mut successor_prop_values = Vec::new();
    semantics.concrete_prop_values(state_packer, &propositions, &mut prop_values);
    let initial_abstract_state = working.hierarchy().map_state(&prop_values, &numeric)?;
    if !shortest_paths.distance(initial_abstract_state).is_finite() {
        return Ok(PlanCheck::AbstractDeadEnd(initial_abstract_state));
    }
    let abstract_plan_cost = shortest_paths.distance(initial_abstract_state);
    let mut operator_ids = Vec::new();
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
                return Ok(PlanCheck::ConcretePlan(ConcretePlan {
                    operator_ids,
                    cost: concrete_cost,
                }));
            }
            let failed_goals = (0..semantics.task().get_num_goals())
                .map(|goal_id| semantics.task().get_goal_fact(goal_id))
                .filter(|goal| !fact_is_hold(goal, state_packer, &propositions))
                .collect::<Vec<_>>();
            ensure!(
                !failed_goals.is_empty(),
                "abstract goal contains a concrete non-goal without a failed goal fact"
            );
            let candidates = if use_desired_region_candidates {
                splits_for_desired_facts(
                    working,
                    semantics,
                    abstract_state,
                    &failed_goals,
                    &prop_values,
                    &numeric,
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
                            &prop_values,
                            &numeric,
                            format!("goal {goal:?}"),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
            };
            return Ok(PlanCheck::Refine(select_refinement_split(
                working,
                semantics,
                candidates,
                0x474F_414C,
            )?));
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
        let op = &semantics.task().get_operators()[op_id];
        let failed_preconditions = op
            .preconditions()
            .iter()
            .filter(|fact| !fact_is_hold(fact, state_packer, &propositions))
            .collect::<Vec<_>>();
        if !failed_preconditions.is_empty() {
            let candidates = if use_desired_region_candidates {
                splits_for_desired_facts(
                    working,
                    semantics,
                    abstract_state,
                    &failed_preconditions,
                    &prop_values,
                    &numeric,
                    &format!(
                        "operator {op_id} ({}) preconditions {failed_preconditions:?}",
                        op.name()
                    ),
                )?
            } else {
                failed_preconditions
                    .iter()
                    .map(|failed| {
                        split_failed_fact(
                            working,
                            semantics,
                            abstract_state,
                            failed,
                            &prop_values,
                            &numeric,
                            format!("operator {op_id} ({}) precondition {failed:?}", op.name()),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
            };
            return Ok(PlanCheck::Refine(select_refinement_split(
                working,
                semantics,
                candidates,
                0x5052_4543,
            )?));
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
            return Ok(PlanCheck::Refine(split_deviation(
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
            )?));
        }

        let op_cost = semantics.operator_costs()[op_id];
        ensure!(
            approximately_equal(
                op_cost + shortest_paths.distance(transition.target),
                abstract_distance
            ),
            "Cartesian generating transition is not distance preserving"
        );
        concrete_cost += op_cost;
        operator_ids.push(op_id);
        selected_plan_pos += usize::from(selected_plan.is_some());
    }
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
    refinement_root: &CartesianConcreteState,
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
                if !fact_is_hold(goal, state_packer, &propositions) {
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
                semantics.operator_costs()[transition.concrete_op_id]
                    + shortest_paths.distance(transition.target),
                abstract_distance
            ),
            "Cartesian generating transition is not distance preserving"
        );

        let op_id = transition.concrete_op_id;
        let op = &semantics.task().get_operators()[op_id];
        for failed in op
            .preconditions()
            .iter()
            .filter(|fact| !fact_is_hold(fact, state_packer, &propositions))
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

        concrete_cost += semantics.operator_costs()[op_id];
        operator_ids.push(op_id);
    }
}

pub(super) fn validate_concrete_plan(
    semantics: &CartesianSemantics<'_>,
    state_packer: &Arc<StatePacker>,
    axiom_evaluator: &AxiomEvaluator<'_>,
    refinement_root: &CartesianConcreteState,
    plan: &ConcretePlan,
) -> Result<()> {
    let mut propositions = refinement_root.propositions.clone();
    let mut numeric = refinement_root.numeric.clone();
    let mut cost = 0.0;
    for (step, &op_id) in plan.operator_ids.iter().enumerate() {
        let op = semantics
            .task()
            .get_operators()
            .get(op_id)
            .with_context(|| format!("concrete plan step {step} has invalid operator {op_id}"))?;
        for precondition in op.preconditions() {
            ensure!(
                fact_is_hold(precondition, state_packer, &propositions),
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
        cost += semantics.operator_costs()[op_id];
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
