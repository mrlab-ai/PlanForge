use super::*;
use crate::evaluation::abstractions::common::abstraction::AbstractionMetadata;
use crate::evaluation::abstractions::common::abstraction::finish_abstraction;

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
        apply_initial_splits(&mut working, &semantics, &self.config)?;
        let shortest_paths = compute_shortest_paths(&working, &semantics)?;
        let mut refinement = CartesianRefinement {
            working,
            shortest_paths,
            semantics: &semantics,
            limits: RefinementLimits::new(&self.config.cegar, &self.config.scope),
        };
        let outcome = refine_from_root(
            &mut refinement,
            semantics.cegar(),
            &self.config.cegar,
            refinement_root,
            start,
        )?;
        let CartesianRefinement {
            working,
            shortest_paths,
            ..
        } = refinement;
        let initial_state = semantics.cegar().states().initial_state()?;
        let initial_abstract_state = working
            .hierarchy
            .map_state(&initial_state.propositional, &initial_state.numeric)?;
        let abstraction = finalize_abstraction(
            working,
            &shortest_paths,
            initial_abstract_state,
            &self.config.cegar,
            AbstractionMetadata::standalone(
                outcome,
                self.config.cegar.split_kind,
                self.config.cegar.split_selection_rank,
            ),
        );
        finish_abstraction(&abstraction, start.elapsed())?;
        Ok(abstraction)
    }
}
