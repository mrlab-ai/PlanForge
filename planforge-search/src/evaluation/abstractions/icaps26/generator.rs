use super::*;

pub struct Icaps26CartesianGenerator {
    config: Icaps26CartesianConfig,
}

impl Icaps26CartesianGenerator {
    pub fn new(config: Icaps26CartesianConfig) -> Result<Self> {
        ensure!(config.max_states > 0, "Cartesian max_states must be > 0");
        Ok(Self { config })
    }

    /// Refine from the task's initial state, as the artifact does.
    pub fn generate(&self, task: &dyn AbstractNumericTask) -> Result<CartesianAbstraction> {
        let start = Instant::now();
        let semantics = CartesianSemantics::new(task, &self.config)?;
        let initial_region = semantics.trivial_region()?;
        let operator_count = semantics.task.get_operators().len();
        let mut working = WorkingAbstraction::new_icaps26(initial_region, operator_count);
        for op_id in 0..task.get_operators().len() {
            if semantics.may_transition(
                &working.states[0],
                OperatorIndex::from_usize(op_id),
                &working.states[0],
            )? {
                working.add_transition(0, OperatorIndex::from_usize(op_id), 0);
            }
        }
        let state_packer = Arc::new(make_prop_state_packer(task));
        let axiom_evaluator = AxiomEvaluator::new(Arc::new(task), state_packer.clone());
        let (initial_propositions, initial_numeric) =
            get_initial_state(task, &state_packer, &axiom_evaluator)?;
        let refinement_root = ConcreteState {
            propositions: initial_propositions.clone(),
            numeric: initial_numeric.clone(),
        };
        let mut refinements: usize = 0;
        let mut shortest_paths =
            compute_shortest_paths_with_goals(&working, &semantics, vec![true])?;
        let mut abstract_search = StableAbstractSearch::trivial();
        let mut pending_flaw = None;
        let mut solved_plan = None;
        let run = CegarDriver::new(usize::MAX, None).run_from_zero_based(
            start,
            |_iteration, _deadline| {
                if !crate::resource_limits::poll_and_release_if_exceeded() {
                    return Ok(CegarIterationResult::Stop(CegarStopReason::MemoryLimit));
                }
                let selected_plan = {
                    let mut initial_prop_values = Vec::new();
                    semantics.concrete_prop_values(
                        &state_packer,
                        &refinement_root.propositions,
                        &mut initial_prop_values,
                    );
                    let initial_state = working
                        .hierarchy
                        .map_state(&initial_prop_values, &refinement_root.numeric)?;
                    let Some(plan) = abstract_search.find_plan(
                        &working,
                        &semantics,
                        initial_state,
                        shortest_paths.goal_flags(),
                    )?
                    else {
                        return Err(RefinementRootDeadEnd::new(format!(
                            "Cartesian abstract state {initial_state}"
                        ))
                        .into());
                    };
                    plan
                };
                let check = replay_optimal_abstract_trace(
                    &working,
                    &semantics,
                    &shortest_paths,
                    &state_packer,
                    &axiom_evaluator,
                    &refinement_root,
                    Some(&selected_plan),
                )?;
                match check {
                    PlanCheck::ConcretePlan(plan) => {
                        solved_plan = Some(plan);
                        Ok(CegarIterationResult::Stop(CegarStopReason::ConcretePlan))
                    }
                    PlanCheck::AbstractDeadEnd(abstract_state_id) => {
                        Err(RefinementRootDeadEnd::new(format!(
                            "Cartesian abstract state {abstract_state_id}"
                        ))
                        .into())
                    }
                    PlanCheck::Refine(split) => {
                        if working.states.len() >= self.config.max_states {
                            pending_flaw = Some(split.description().to_string());
                            return Ok(CegarIterationResult::Stop(CegarStopReason::SizeLimit));
                        }
                        if self
                            .config
                            .max_time
                            .is_some_and(|max_time| start.elapsed() >= max_time)
                        {
                            pending_flaw = Some(split.description().to_string());
                            return Ok(CegarIterationResult::Stop(CegarStopReason::TimeLimit));
                        }
                        if self.config.debug {
                            info!(
                                "Cartesian refinement {} at {} states: {split:?}",
                                refinements,
                                working.states.len()
                            );
                        }
                        let old_state_id = split.state_id();
                        let new_state_id = working.apply_split(&semantics, split)?;
                        update_shortest_paths_after_split(
                            &working,
                            &semantics,
                            &mut shortest_paths,
                            old_state_id,
                            new_state_id,
                        )?;
                        abstract_search.inherit_split_state(old_state_id, new_state_id);
                        refinements += 1;
                        Ok(CegarIterationResult::Continue)
                    }
                }
            },
        )?;
        let stop_reason = run.stop_reason;
        assert!(
            matches!(
                stop_reason,
                CegarStopReason::ConcretePlan
                    | CegarStopReason::SizeLimit
                    | CegarStopReason::TimeLimit
                    | CegarStopReason::MemoryLimit
            ),
            // The ICAPS26 loop reports a dead root as an error.
            "Cartesian CEGAR driver returned unsupported stop reason {stop_reason:?}"
        );

        let working_transition_count = working.active_transition_ids().count();
        let mut initial_prop_values = Vec::new();
        semantics.concrete_prop_values(
            &state_packer,
            &initial_propositions,
            &mut initial_prop_values,
        );
        let initial_abstract_state = working
            .hierarchy
            .map_state(&initial_prop_values, &initial_numeric)?;
        let (transitions, distance_table, relevant_operator_ids) = if self
            .config
            .retain_transition_system
        {
            let (transitions, distance_table, relevant_operator_ids) = finalize_abstraction(
                &working,
                &semantics,
                initial_abstract_state,
                self.config.combine_labels,
                self.config.compute_source_sets,
            )?;
            (Some(transitions), distance_table, relevant_operator_ids)
        } else {
            let (distance_table, relevant_operator_ids) =
                finalize_standalone_abstraction(&working, &shortest_paths, initial_abstract_state)?;
            (None, distance_table, relevant_operator_ids)
        };
        let transition_count = match &transitions {
            Some(transitions) => transitions.system.transitions.len(),
            None => working_transition_count,
        };
        if let Some(plan) = &solved_plan {
            validate_concrete_plan(
                &semantics,
                &state_packer,
                &axiom_evaluator,
                &refinement_root,
                plan,
            )?;
            let mut root_prop_values = Vec::new();
            semantics.concrete_prop_values(
                &state_packer,
                &refinement_root.propositions,
                &mut root_prop_values,
            );
            let root_state_id = working
                .hierarchy
                .map_state(&root_prop_values, &refinement_root.numeric)?;
            let h = distance_table.distances[root_state_id];
            ensure!(
                (plan.cost() - h).abs() <= 1e-7,
                "concrete Cartesian plan cost {} differs from abstract h(refinement root) {h}",
                plan.cost()
            );
        }
        info!(
            "Cartesian abstraction: states={}, transitions={}, refinements={}, h(init)={}, stop={stop_reason:?}, elapsed={:.3}s",
            distance_table.distances.len(),
            transition_count,
            refinements,
            distance_table.distances[distance_table.initial_state],
            start.elapsed().as_secs_f64()
        );
        // ICAPS26 abstractions read derived coordinates without additive views.
        let raw_values = vec![None; semantics.task().numeric_variables().len()].into();
        Ok(CartesianAbstraction::new(
            working.hierarchy,
            raw_values,
            AbstractionCore {
                distances: distance_table.distances,
                initial_state: distance_table.initial_state,
                relevant_operator_ids,
                num_abstract_operators: transition_count,
                metadata: AbstractionMetadata {
                    abstraction_use: AbstractionUse::Standalone,
                    stop_reason,
                    concrete_plan_operator_ids: solved_plan.map(ConcretePlan::into_operator_ids),
                    pending_flaw,
                    refinements,
                    split_kind: SplitKind::Wanted,
                    split_selection_rank: None,
                },
            },
            transitions,
        ))
    }
}
