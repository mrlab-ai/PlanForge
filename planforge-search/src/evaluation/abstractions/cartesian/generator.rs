use super::*;

pub struct CartesianAbstractionGenerator {
    pub(super) config: CartesianAbstractionConfig,
}

impl CartesianAbstractionGenerator {
    pub fn new(config: CartesianAbstractionConfig) -> Result<Self> {
        ensure!(config.max_states > 0, "Cartesian max_states must be > 0");
        validate_cartesian_flaw_kind(config.flaw_kind).map_err(anyhow::Error::msg)?;
        Ok(Self { config })
    }

    pub fn generate(&self, task: &dyn AbstractNumericTask) -> Result<CartesianAbstraction> {
        self.generate_from_root(task, None)
    }

    pub(super) fn generate_from_root(
        &self,
        task: &dyn AbstractNumericTask,
        refinement_root: Option<&ConcreteState>,
    ) -> Result<CartesianAbstraction> {
        let start = Instant::now();
        let semantics = CartesianSemantics::new(task, &self.config)?;
        let initial_region = semantics.trivial_region()?;
        let operator_count = semantics.task.get_operators().len();
        let mut working = WorkingAbstraction::new(initial_region, operator_count);
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
        let refinement_root = match refinement_root {
            Some(root) => {
                ensure!(
                    root.propositions.len() == state_packer.num_bins(),
                    "Cartesian refinement root has {} proposition bins, expected {}",
                    root.propositions.len(),
                    state_packer.num_bins()
                );
                ensure!(
                    root.numeric.len() == task.numeric_variables().len(),
                    "Cartesian refinement root has {} numeric values, expected {}",
                    root.numeric.len(),
                    task.numeric_variables().len()
                );
                root.clone()
            }
            None => ConcreteState {
                propositions: initial_propositions.clone(),
                numeric: initial_numeric.clone(),
            },
        };
        let mut refinements: usize = 0;
        let mut shortest_paths = compute_shortest_paths(&working, &semantics)?;
        let mut pending_flaw = None;
        let mut solved_plan = None;
        let run = CegarDriver::new(usize::MAX, None).run_from_zero_based(
            start,
            |_iteration, _deadline| {
                let check = match self.config.flaw_kind {
                    FlawKind::Progression => replay_optimal_abstract_trace(
                        &working,
                        &semantics,
                        &shortest_paths,
                        &state_packer,
                        &axiom_evaluator,
                        &refinement_root,
                    )?,
                    FlawKind::Regression => find_regression_flaw(
                        &working,
                        &semantics,
                        &shortest_paths,
                        &state_packer,
                        &refinement_root,
                    )?,
                    FlawKind::ExecuteEntirePlan => replay_entire_optimal_abstract_trace(
                        &working,
                        &semantics,
                        &shortest_paths,
                        &state_packer,
                        &axiom_evaluator,
                        &refinement_root,
                    )?,
                    _ => unreachable!(
                        "unsupported Cartesian flaw kind passed constructor validation"
                    ),
                };
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
                        refinements += 1;
                        Ok(CegarIterationResult::Continue)
                    }
                }
            },
        )?;
        let stop_reason = match run.stop_reason {
            CegarStopReason::ConcretePlan => CartesianStopReason::ConcretePlan,
            CegarStopReason::SizeLimit => CartesianStopReason::StateLimit,
            CegarStopReason::TimeLimit => CartesianStopReason::TimeLimit,
            CegarStopReason::MemoryLimit => CartesianStopReason::MemoryLimit,
            CegarStopReason::IterationLimit | CegarStopReason::NoRefinableFlaw => {
                unreachable!("Cartesian CEGAR driver returned unsupported stop reason")
            }
        };

        let working_transition_count = working.active_transition_ids().count();
        let mut initial_prop_values = Vec::new();
        semantics.concrete_prop_values(
            &state_packer,
            &initial_propositions,
            &mut initial_prop_values,
        );
        let initial_state_hash = working
            .hierarchy
            .map_state(&initial_prop_values, &initial_numeric)?;
        let (transition_system, distance_table, relevant_operator_ids, source_sets) =
            if self.config.retain_transition_system {
                finalize_abstraction(
                    &working,
                    &semantics,
                    initial_state_hash,
                    self.config.combine_labels,
                    self.config.compute_source_sets,
                )?
            } else {
                finalize_standalone_abstraction(
                    &working,
                    &semantics,
                    &shortest_paths,
                    initial_state_hash,
                )?
            };
        let transition_count = if self.config.retain_transition_system {
            transition_system.transitions.len()
        } else {
            working_transition_count
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
            distance_table.distances[distance_table.initial_state_hash],
            start.elapsed().as_secs_f64()
        );
        Ok(CartesianAbstraction {
            hierarchy: working.hierarchy,
            distance_table,
            transition_system,
            relevant_operator_ids,
            abstract_source_sets: source_sets,
            metadata: CartesianAbstractionMetadata {
                solved_by_self: solved_plan.is_some(),
                abstraction_use: AbstractionUse::Standalone,
                stop_reason,
                pending_flaw,
                refinements,
                collection_member: None,
                refinement_direction: self.config.refinement_direction,
                split_selection_rank: self.config.split_selection_rank,
                concrete_plan_operator_ids: solved_plan.map(ConcretePlan::into_operator_ids),
                transition_count,
            },
        })
    }
}
