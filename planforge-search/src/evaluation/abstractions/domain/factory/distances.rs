use super::*;

/// Everything about one distance-table build that is not the abstraction
/// itself. The default is a plain build without budget that materializes
/// state regions.
#[derive(Clone, Copy)]
pub struct DistanceTableOptions<'a> {
    /// Override the task goals for this distance-table build.
    pub goal_facts: Option<&'a [ExplicitFact]>,
    /// Give up once the online-SCP budget is spent.
    pub deadline: Option<Instant>,
    /// State whose distance bounds perimeter saturation.
    pub cap_state_id: Option<usize>,
    /// Materialize concrete regions for abstract states.
    pub materialize_cartesian_sets: bool,
    /// Retain abstract self loops in an explicit transition system.
    pub include_self_loops: bool,
    /// Bound concrete label transitions in an OCP transition system.
    pub max_concrete_label_transitions: Option<usize>,
}

impl Default for DistanceTableOptions<'_> {
    fn default() -> Self {
        Self {
            goal_facts: None,
            deadline: None,
            cap_state_id: None,
            materialize_cartesian_sets: true,
            include_self_loops: false,
            max_concrete_label_transitions: None,
        }
    }
}

impl<'a> DistanceTableOptions<'a> {
    pub fn with_deadline(mut self, deadline: Option<Instant>) -> Self {
        self.deadline = deadline;
        self
    }

    pub fn with_goal_facts(mut self, goal_facts: &'a [ExplicitFact]) -> Self {
        self.goal_facts = Some(goal_facts);
        self
    }

    pub fn with_cap_state(mut self, cap_state_id: Option<usize>) -> Self {
        self.cap_state_id = cap_state_id;
        self
    }

    pub fn without_cartesian_sets(mut self) -> Self {
        self.materialize_cartesian_sets = false;
        self
    }

    pub fn with_concrete_label_cap(mut self, max_transitions: usize) -> Self {
        self.max_concrete_label_transitions = Some(max_transitions);
        self
    }
}

impl DomainAbstractionFactory {
    /// Goal distances of every abstract state under `operator_costs`, without
    /// saturated costs. Used by the order generator during diversification.
    pub fn build_goal_distances(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operator_costs: &[f64],
    ) -> Result<AbstractDistanceTable> {
        let goal_facts = self.compute_abstract_goals(task);
        self.build_goal_distances_for_goals(task, combine_labels, operator_costs, &goal_facts)
    }

    pub fn build_goal_distances_for_goals(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operator_costs: &[f64],
        goal_facts: &[ExplicitFact],
    ) -> Result<AbstractDistanceTable> {
        let mut operators = self.abstract_operators(task, combine_labels, None)?;
        apply_operator_costs(&mut operators, operator_costs)?;
        self.build_distance_table(
            task,
            &operators,
            DistanceTableOptions::default().with_goal_facts(goal_facts),
        )
    }

    /// Goal distances of every abstract state under `operators`, to the goals
    /// of `options` or else the task's abstract goals.
    pub(crate) fn build_distance_table(
        &self,
        task: &dyn AbstractNumericTask,
        operators: &[AbstractOperator],
        options: DistanceTableOptions<'_>,
    ) -> Result<AbstractDistanceTable> {
        let match_tree = MatchTree::new(&self.layout, operators);
        self.build_distance_table_with_match_tree(task, operators, &match_tree, options)
    }

    /// [`Self::build_distance_table`] with the match tree of `operators`.
    pub(super) fn build_distance_table_with_match_tree(
        &self,
        task: &dyn AbstractNumericTask,
        operators: &[AbstractOperator],
        match_tree: &MatchTree,
        options: DistanceTableOptions<'_>,
    ) -> Result<AbstractDistanceTable> {
        let deadline = options.deadline;
        ensure_online_scp_deadline(deadline)?;
        let computed_goal_facts;
        let goal_facts = match options.goal_facts {
            Some(goal_facts) => goal_facts,
            None => {
                computed_goal_facts = self.compute_abstract_goals(task);
                &computed_goal_facts
            }
        };
        let goal_states = self.goal_states(task, goal_facts, deadline)?;
        let num_states = self.layout.num_states();
        let mut generating_op_ids = vec![None; num_states];
        let distances = backward_goal_distances(
            &mut MatchTreeGraph {
                operators,
                match_tree,
                num_states,
                applicable: Vec::new(),
            },
            &goal_states,
            || ensure_online_scp_deadline(deadline),
            Some(&mut generating_op_ids),
        )?;
        Ok(AbstractDistanceTable {
            distances,
            generating_op_ids,
            initial_state_hash: self.initial_state(task)?,
            goal_facts: goal_facts.to_vec(),
        })
    }

    pub(super) fn build_distance_table_with_transition_costs(
        &self,
        transition_system: &AbstractTransitionSystem,
        transition_costs: &[f64],
    ) -> Result<AbstractDistanceTable> {
        ensure!(
            transition_system.transitions.len() == transition_costs.len(),
            "transition system/cost vector size mismatch: {} vs {}",
            transition_system.transitions.len(),
            transition_costs.len()
        );

        let mut generating_op_ids = vec![None; transition_system.backward.len()];
        let distances = backward_goal_distances(
            &mut CostedTransitionSystem {
                system: transition_system,
                transition_costs,
            },
            &transition_system.goal_state_hashes,
            || Ok(()),
            Some(&mut generating_op_ids),
        )?;

        Ok(AbstractDistanceTable {
            distances,
            generating_op_ids,
            initial_state_hash: transition_system.initial_state_hash,
            goal_facts: transition_system.goal_facts.clone(),
        })
    }
}

/// The implicit transition graph of a domain abstraction: every abstract
/// operator the match tree finds applicable in a state, indexed by its
/// effects, leads into it from the state its hash effect names.
pub(super) struct MatchTreeGraph<'a> {
    pub(super) operators: &'a [AbstractOperator],
    pub(super) match_tree: &'a MatchTree,
    pub(super) num_states: usize,
    pub(super) applicable: Vec<usize>,
}

impl BackwardGraph for MatchTreeGraph<'_> {
    type Label = usize;

    fn num_states(&self) -> usize {
        self.num_states
    }

    fn incoming(&mut self, target: usize, incoming: &mut Vec<(usize, usize, f64)>) -> Result<()> {
        incoming.clear();
        self.match_tree
            .get_applicable_operator_ids(target, &mut self.applicable);
        for &op_id in &self.applicable {
            let op = &self.operators[op_id];
            let source = target
                .checked_add_signed(op.regression_offset)
                .filter(|&source| source < self.num_states)
                .with_context(|| {
                    format!(
                        "abstract operator {op_id} regresses state {target} by {} out of the {} states",
                        op.regression_offset, self.num_states
                    )
                })?;
            incoming.push((source, op_id, op.cost.value()));
        }
        Ok(())
    }
}
