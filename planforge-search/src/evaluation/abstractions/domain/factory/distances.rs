use super::*;

/// How abstract states are hashed: the numeric domain sizes and the
/// mixed-radix multipliers of every variable.
#[derive(Clone, Copy)]
pub(super) struct HashLayout<'a> {
    pub(super) numeric_domain_sizes: &'a [usize],
    pub(super) hash_multipliers: &'a [usize],
}

/// The abstract state space a backward goal-distance Dijkstra runs over: the
/// task the abstraction was built from, its abstract operators together with
/// the match tree indexing them, the abstract goal facts, the hash layout, and
/// how many states that layout spans.
#[derive(Clone, Copy)]
pub(super) struct AbstractGoalDistanceSpace<'a> {
    pub(super) task: &'a dyn AbstractNumericTask,
    pub(super) operators: &'a [AbstractOperator],
    pub(super) match_tree: &'a MatchTree,
    pub(super) goal_facts: &'a [ExplicitFact],
    pub(super) layout: HashLayout<'a>,
    pub(super) num_states: usize,
}

/// Everything about one distance-table build that is not the abstraction
/// itself. All three default to "plain build, no logging, no budget".
#[derive(Clone, Copy)]
pub struct DistanceTableOptions<'a> {
    /// Log the resulting distances.
    pub dump_distances: bool,
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
            dump_distances: false,
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

    pub fn with_dump_distances(mut self, dump_distances: bool) -> Self {
        self.dump_distances = dump_distances;
        self
    }
}

pub(super) fn compute_num_states(
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
) -> Result<usize> {
    let mut num: usize = 1;
    for (i, &s) in domain_sizes.iter().enumerate() {
        ensure!(s > 0, "domain size for var {i} must be > 0, got {s}");
        num = num
            .checked_mul(s)
            .context("abstract state space too large (overflow)")?;
    }
    for &s in numeric_domain_sizes.iter() {
        num = num
            .checked_mul(s)
            .context("abstract state space too large (overflow)")?;
    }
    Ok(num)
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
        let mut generator = self.make_operator_generator(task, combine_labels)?;
        let mut operators = generator.build_abstract_operators(task)?;
        apply_operator_costs(&mut operators, operator_costs)?;
        self.build_distance_table(
            task,
            &generator,
            &operators,
            DistanceTableOptions::default().with_goal_facts(goal_facts),
        )
    }

    /// Goal distances of every abstract state under `operators`, to the goals
    /// of `options` or else the task's abstract goals.
    pub(crate) fn build_distance_table(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        options: DistanceTableOptions<'_>,
    ) -> Result<AbstractDistanceTable> {
        let match_tree = MatchTree::new(generator, operators);
        self.build_distance_table_with_match_tree(task, generator, operators, &match_tree, options)
    }

    /// [`Self::build_distance_table`] with the match tree of `operators`.
    pub(super) fn build_distance_table_with_match_tree(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
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
        let hash_multipliers = generator.hash_multipliers();
        let numeric_domain_sizes = generator.numeric_domain_sizes();

        // Numeric-fd computes a *single* initial abstract state hash directly from the
        // concrete initial state (comparisons are evaluated, not enumerated).
        let init_hash = self.compute_initial_state_hash_determined(
            task,
            numeric_domain_sizes,
            hash_multipliers,
        )?;

        let num_states = compute_num_states(&self.domain_sizes, numeric_domain_sizes)?;
        let mut generating_op_ids = vec![None; num_states];
        let distances = self.compute_goal_distances(
            AbstractGoalDistanceSpace {
                task,
                operators,
                match_tree,
                goal_facts,
                layout: HashLayout {
                    numeric_domain_sizes,
                    hash_multipliers,
                },
                num_states,
            },
            Some(&mut generating_op_ids),
            deadline,
        )?;

        let table = AbstractDistanceTable {
            distances,
            generating_op_ids,
            initial_state_hash: init_hash,
            goal_facts: goal_facts.to_vec(),
            hash_multipliers: hash_multipliers.to_vec(),
            numeric_domain_sizes: numeric_domain_sizes.to_vec(),
        };

        if options.dump_distances {
            self.dump_distances(task, &table);
        }

        Ok(table)
    }

    pub(super) fn build_distance_table_with_transition_costs(
        &self,
        transition_system: &AbstractTransitionSystem,
        transition_costs: &[f64],
        hash_multipliers: &[usize],
        numeric_domain_sizes: &[usize],
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
            hash_multipliers: hash_multipliers.to_vec(),
            numeric_domain_sizes: numeric_domain_sizes.to_vec(),
        })
    }

    /// Prints a numeric-fd style table of core variables for all reachable abstract states.
    ///
    /// Core variables are:
    /// - all numeric variables with more than one partition,
    /// - all non-axiom propositional variables with abstract domain size > 1.
    pub fn dump_distances(&self, task: &dyn AbstractNumericTask, table: &AbstractDistanceTable) {
        dump_distances(self, task, table);
    }

    pub(super) fn compute_goal_distances(
        &self,
        space: AbstractGoalDistanceSpace<'_>,
        generating_op_ids: Option<&mut [Option<usize>]>,
        deadline: Option<Instant>,
    ) -> Result<Vec<f64>> {
        let AbstractGoalDistanceSpace {
            task,
            operators,
            match_tree,
            goal_facts,
            layout,
            num_states,
        } = space;
        let HashLayout {
            numeric_domain_sizes,
            hash_multipliers,
        } = layout;
        let mut goal_states = Vec::new();
        for state_hash in 0..num_states {
            if state_hash % 1024 == 0 {
                ensure_online_scp_deadline(deadline)?;
            }
            if self.is_goal_state(
                task,
                state_hash,
                goal_facts,
                numeric_domain_sizes,
                hash_multipliers,
            )? {
                goal_states.push(state_hash);
            }
        }
        backward_goal_distances(
            &mut MatchTreeGraph {
                operators,
                match_tree,
                num_states,
                applicable: Vec::new(),
            },
            &goal_states,
            || ensure_online_scp_deadline(deadline),
            generating_op_ids,
        )
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
                .checked_add_signed(op.hash_effect as isize)
                .filter(|&source| source < self.num_states)
                .with_context(|| {
                    format!(
                        "abstract operator {op_id} regresses state {target} by {} out of the {} states",
                        op.hash_effect, self.num_states
                    )
                })?;
            incoming.push((source, op_id, op.cost.value()));
        }
        Ok(())
    }
}
