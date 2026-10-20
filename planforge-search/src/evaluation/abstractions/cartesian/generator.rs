use super::*;

pub struct CartesianAbstractionGenerator {
    pub(super) config: CartesianAbstractionConfig,
}

impl CartesianAbstractionGenerator {
    pub fn new(config: CartesianAbstractionConfig) -> Result<Self> {
        ensure!(config.max_states > 0, "Cartesian max_states must be > 0");
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
        let initial_region = semantics.regions().trivial_region()?;
        let operator_count = semantics.task().get_operators().len();
        let mut working = WorkingAbstraction::new(initial_region, operator_count);
        for op_id in 0..task.get_operators().len() {
            if semantics.regions().may_transition(
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
        if let Some(var) = self.config.initial_split_var {
            let mut initial_prop_values = Vec::new();
            concrete_prop_values(
                task,
                &state_packer,
                &initial_propositions,
                &mut initial_prop_values,
            );
            apply_initial_split(
                &mut working,
                &semantics,
                &self.config,
                &initial_prop_values,
                var,
            )?;
        }
        let shortest_paths = compute_shortest_paths(&working, &semantics)?;
        let mut refinement = CartesianRefinement {
            working,
            shortest_paths,
            semantics: &semantics,
            limits: CartesianLimits {
                max_states: self.config.max_states,
                blacklisted_prop_var_ids: &self.config.blacklisted_prop_var_ids,
                blacklisted_numeric_var_ids: &self.config.blacklisted_numeric_var_ids,
            },
            debug: self.config.debug,
        };
        let outcome = refine(
            &mut refinement,
            &CegarRun {
                semantics: semantics.cegar(),
                flaw_kind: self.config.flaw_kind,
                flaw_scope: self.config.flaw_scope,
                state_packer: &state_packer,
                axiom_evaluator: &axiom_evaluator,
                refinement_root: &refinement_root,
                driver: CegarDriver::new(usize::MAX, self.config.max_time),
            },
            start,
        )?;
        let CartesianRefinement {
            working,
            shortest_paths,
            ..
        } = refinement;
        let stop_reason = outcome.stop_reason;
        let solved_plan = outcome.concrete_plan;

        let working_transition_count = working.active_transition_ids().count();
        let mut initial_prop_values = Vec::new();
        concrete_prop_values(
            semantics.task(),
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
                semantics.cegar(),
                &state_packer,
                &axiom_evaluator,
                &refinement_root,
                plan,
            )?;
            let mut root_prop_values = Vec::new();
            concrete_prop_values(
                semantics.task(),
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
            outcome.refinements,
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
                pending_flaw: outcome.pending_flaw,
                refinements: outcome.refinements,
                collection_member: None,
                split_kind: self.config.split_kind,
                split_selection_rank: self.config.split_selection_rank,
                concrete_plan_operator_ids: solved_plan.map(ConcretePlan::into_operator_ids),
                transition_count,
            },
        })
    }
}
