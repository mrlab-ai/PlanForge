use super::*;
use crate::evaluation::abstractions::common::abstraction::AbstractionMetadata;

pub struct CartesianAbstractionGenerator {
    pub(super) config: CartesianAbstractionConfig,
}

impl AbstractionGenerator for CartesianAbstractionGenerator {
    type Config = CartesianAbstractionConfig;
    type Abstraction = CartesianAbstraction;
    type Key = Vec<HierarchyKey>;

    const NAME: &'static str = "Cartesian";

    fn new(config: CartesianAbstractionConfig) -> Result<Self> {
        config.cegar.validate()?;
        Ok(Self { config })
    }

    fn key(abstraction: &CartesianAbstraction) -> Vec<HierarchyKey> {
        hierarchy_key(&abstraction.hierarchy)
    }

    fn generate_from_root(
        &self,
        task: &dyn AbstractNumericTask,
        refinement_root: Option<&ExplicitState>,
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
        let initial_state = semantics.cegar().states().initial_state()?;
        let refinement_root = refinement_root.unwrap_or(&initial_state);
        if let Some(var) = self.config.initial_split_var {
            apply_initial_split(
                &mut working,
                &semantics,
                &self.config,
                &initial_state.propositional,
                var,
            )?;
        }
        let shortest_paths = compute_shortest_paths(&working, &semantics)?;
        let mut refinement = CartesianRefinement {
            working,
            shortest_paths,
            semantics: &semantics,
            limits: CartesianLimits {
                max_states: self.config.cegar.max_states,
                blacklisted_prop_var_ids: &self.config.blacklisted_prop_var_ids,
                blacklisted_numeric_var_ids: &self.config.blacklisted_numeric_var_ids,
            },
            debug: self.config.cegar.debug,
        };
        let outcome = refine(
            &mut refinement,
            &CegarRun {
                semantics: semantics.cegar(),
                flaw_kind: self.config.cegar.flaw_kind,
                flaw_scope: self.config.cegar.flaw_scope,
                refinement_root,
                driver: CegarDriver::new(
                    self.config.cegar.max_iterations,
                    self.config.cegar.max_time,
                ),
            },
            start,
        )?;
        let CartesianRefinement {
            working,
            shortest_paths,
            ..
        } = refinement;
        let working_transition_count = working.active_transition_ids().count();
        let initial_state_hash = working
            .hierarchy
            .map_state(&initial_state.propositional, &initial_state.numeric)?;
        let (transition_system, distance_table, relevant_operator_ids, source_sets) =
            if self.config.retain_transition_system {
                finalize_abstraction(
                    &working,
                    &semantics,
                    initial_state_hash,
                    self.config.cegar.combine_labels,
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
        if let Some(plan) = &outcome.concrete_plan {
            let root_state_id = working
                .hierarchy
                .map_state(&refinement_root.propositional, &refinement_root.numeric)?;
            let h = distance_table.distances[root_state_id];
            ensure!(
                (plan.cost() - h).abs() <= 1e-7,
                "concrete Cartesian plan cost {} differs from abstract h(refinement root) {h}",
                plan.cost()
            );
        }
        info!(
            "Cartesian abstraction: states={}, transitions={}, refinements={}, h(init)={}, stop={:?}, elapsed={:.3}s",
            distance_table.distances.len(),
            transition_count,
            outcome.refinements,
            distance_table.distances[distance_table.initial_state_hash],
            outcome.stop_reason,
            start.elapsed().as_secs_f64()
        );
        Ok(CartesianAbstraction {
            hierarchy: working.hierarchy,
            distance_table,
            transition_system,
            relevant_operator_ids,
            abstract_source_sets: source_sets,
            metadata: AbstractionMetadata::standalone(
                outcome,
                self.config.cegar.split_kind,
                self.config.cegar.split_selection_rank,
            ),
            transition_count,
        })
    }
}
