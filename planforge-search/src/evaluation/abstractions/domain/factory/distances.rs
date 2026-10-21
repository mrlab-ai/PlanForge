use super::*;

impl DomainAbstractionFactory {
    /// The goal distances of every abstract state under `operators`, to the
    /// task's abstract goals.
    pub(crate) fn build_distance_table(
        &self,
        task: &dyn AbstractNumericTask,
        operators: &[AbstractOperator],
        deadline: Deadline,
    ) -> Result<AbstractDistanceTable> {
        deadline.ensure(DISTANCE_TABLE)?;
        let goal_facts = self.compute_abstract_goals(task);
        let goal_states = self.goal_states(task, &goal_facts, deadline)?;
        let match_tree = MatchTree::new(&self.layout, operators);
        let num_states = self.layout.num_states();
        let mut generating_op_ids = vec![None; num_states];
        let distances = backward_goal_distances(
            &mut MatchTreeGraph::new(operators, &match_tree, num_states, &goal_states),
            &goal_states,
            || deadline.ensure(DISTANCE_TABLE),
            Some(&mut generating_op_ids),
        )?;
        Ok(AbstractDistanceTable {
            distances,
            generating_op_ids,
            initial_state_hash: self.initial_state(task)?,
            goal_facts,
        })
    }
}

/// The implicit transition graph of a domain abstraction: every abstract
/// operator the match tree finds applicable in a state, indexed by its
/// effects, leads into it from the state its regression offset names.
pub(crate) struct MatchTreeGraph<'a> {
    operators: &'a [AbstractOperator],
    match_tree: &'a MatchTree,
    num_states: usize,
    goal_states: &'a [usize],
    /// The cost of every abstract operator.
    costs: Vec<f64>,
    applicable: Vec<usize>,
}

impl<'a> MatchTreeGraph<'a> {
    /// The graph of `operators`, indexed by `match_tree`, over `num_states`
    /// states with goal states `goal_states`, under the operators' own costs.
    pub(crate) fn new(
        operators: &'a [AbstractOperator],
        match_tree: &'a MatchTree,
        num_states: usize,
        goal_states: &'a [usize],
    ) -> Self {
        Self {
            operators,
            match_tree,
            num_states,
            goal_states,
            costs: operators.iter().map(|op| op.cost.value()).collect(),
            applicable: Vec::new(),
        }
    }
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
            incoming.push((source, op_id, self.costs[op_id]));
        }
        Ok(())
    }
}

impl SaturationGraph for MatchTreeGraph<'_> {
    fn goal_states(&self) -> &[usize] {
        self.goal_states
    }

    fn set_operator_costs(&mut self, operator_costs: &[f64]) {
        for (cost, op) in self.costs.iter_mut().zip(self.operators) {
            *cost = cheapest_label(&op.concrete_op_ids, operator_costs);
        }
    }

    fn label_operators(&self, op_id: usize) -> &[OperatorIndex] {
        &self.operators[op_id].concrete_op_ids
    }
}
