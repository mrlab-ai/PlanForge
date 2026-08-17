use super::*;

#[derive(Clone, Copy)]
pub(super) struct ComparisonBranchingLayout<'a> {
    pub(super) numeric_domain_sizes: &'a [usize],
    pub(super) hash_multipliers: &'a [usize],
    pub(super) comparison_var_ids: &'a [usize],
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
    pub(super) layout: ComparisonBranchingLayout<'a>,
    pub(super) num_states: usize,
}

#[derive(Clone, Copy)]
pub(super) enum GoalDistanceStop {
    Exhaust,
    FirstReaching(usize),
}

pub(super) enum GoalDistanceResult {
    Exhausted(Vec<f64>),
    Reached(f64),
}

/// Everything about one distance-table build that is not the abstraction
/// itself. All three default to "plain build, no logging, no budget".
#[derive(Clone, Copy)]
pub struct DistanceTableOptions<'a> {
    /// Log the resulting distances.
    pub dump_distances: bool,
    /// Override the task goals for this distance-table build.
    pub goal_facts: Option<&'a [ExplicitFact]>,
    /// Reuse a match tree the caller already built for these operators.
    pub(super) prebuilt_match_tree: Option<&'a MatchTree>,
    /// Give up once the online-SCP budget is spent.
    pub deadline: Option<Instant>,
    /// State whose distance bounds perimeter saturation.
    pub cap_state_id: Option<usize>,
    /// Materialize concrete regions for abstract states.
    pub materialize_state_regions: bool,
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
            prebuilt_match_tree: None,
            deadline: None,
            cap_state_id: None,
            materialize_state_regions: true,
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

    pub fn without_state_regions(mut self) -> Self {
        self.materialize_state_regions = false;
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

#[derive(Debug, Clone)]
pub struct AbstractDistanceTable {
    pub distances: Vec<f64>,
    // Per-state operator leading to a goal along a shortest path.
    pub generating_op_ids: Vec<Option<usize>>,
    pub initial_state_hash: usize,
    pub goal_facts: Vec<ExplicitFact>,
    pub hash_multipliers: Vec<usize>,
    pub numeric_domain_sizes: Vec<usize>,
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
    /// Runs numeric-fd style implicit regression Dijkstra and returns distances-to-goal for
    /// all abstract states plus the generating operator per state.
    pub fn build_abstract_distance_table(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        dump_distances: bool,
    ) -> Result<AbstractDistanceTable> {
        let mut generator = self.make_operator_generator(task, combine_labels)?;
        let operators = generator.build_abstract_operators(task)?;
        self.build_distance_table_with_operators(task, &generator, &operators, dump_distances)
    }

    /// Builds goal distances using the supplied operator costs, without computing
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
        self.build_distance_table_with_operators_for_goals_inner(
            task,
            &generator,
            &operators,
            goal_facts,
            DistanceTableOptions::default(),
        )
    }

    pub(crate) fn build_distance_table_with_operators(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        dump_distances: bool,
    ) -> Result<AbstractDistanceTable> {
        let goal_facts = self.compute_abstract_goals(task);
        self.build_distance_table_with_operators_for_goals(
            task,
            generator,
            operators,
            dump_distances,
            &goal_facts,
        )
    }

    pub(super) fn zero_distance_table_for_generator(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
    ) -> Result<AbstractDistanceTable> {
        let hash_multipliers = generator.hash_multipliers();
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let comparison_var_ids = self.comparison_var_ids();
        let init_hash = self.compute_initial_state_hash_determined(
            task,
            numeric_domain_sizes,
            hash_multipliers,
            &comparison_var_ids,
        )?;
        let num_states = compute_num_states(&self.domain_sizes, numeric_domain_sizes)?;
        Ok(AbstractDistanceTable {
            distances: vec![0.0; num_states],
            generating_op_ids: vec![None; num_states],
            initial_state_hash: init_hash,
            goal_facts: self.compute_abstract_goals(task),
            hash_multipliers: hash_multipliers.to_vec(),
            numeric_domain_sizes: numeric_domain_sizes.to_vec(),
        })
    }

    pub(super) fn compute_distance_to_goal_state_with_operators(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        target_state_hash: usize,
        deadline: Option<Instant>,
    ) -> Result<f64> {
        let goal_facts = self.compute_abstract_goals(task);
        let hash_multipliers = generator.hash_multipliers();
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let num_states = compute_num_states(&self.domain_sizes, numeric_domain_sizes)?;
        ensure!(
            target_state_hash < num_states,
            "target abstract state {target_state_hash} is out of bounds for {num_states} states"
        );
        let comparison_var_ids = self.comparison_var_ids();
        let match_tree = MatchTree::build(
            &self.domain_sizes,
            numeric_domain_sizes,
            hash_multipliers,
            operators,
            &comparison_var_ids,
        );
        self.compute_distance_to_goal_state(
            AbstractGoalDistanceSpace {
                task,
                operators,
                match_tree: &match_tree,
                goal_facts: &goal_facts,
                layout: ComparisonBranchingLayout {
                    numeric_domain_sizes,
                    hash_multipliers,
                    comparison_var_ids: &comparison_var_ids,
                },
                num_states,
            },
            target_state_hash,
            deadline,
        )
    }

    pub(super) fn build_distance_table_with_operators_for_goals(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        dump_distances: bool,
        goal_facts: &[ExplicitFact],
    ) -> Result<AbstractDistanceTable> {
        self.build_distance_table_with_operators_for_goals_inner(
            task,
            generator,
            operators,
            goal_facts,
            DistanceTableOptions {
                dump_distances,
                ..DistanceTableOptions::default()
            },
        )
    }

    pub(super) fn build_distance_table_with_operators_for_goals_inner(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        goal_facts: &[ExplicitFact],
        options: DistanceTableOptions<'_>,
    ) -> Result<AbstractDistanceTable> {
        let DistanceTableOptions {
            dump_distances,
            prebuilt_match_tree,
            deadline,
            ..
        } = options;
        ensure_online_scp_deadline(deadline)?;
        let hash_multipliers = generator.hash_multipliers();
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let comparison_var_ids = self.comparison_var_ids();

        // Numeric-fd computes a *single* initial abstract state hash directly from the
        // concrete initial state (comparisons are evaluated, not enumerated).
        let init_hash = self.compute_initial_state_hash_determined(
            task,
            numeric_domain_sizes,
            hash_multipliers,
            &comparison_var_ids,
        )?;

        let num_states = compute_num_states(&self.domain_sizes, numeric_domain_sizes)?;

        let owned_match_tree = if prebuilt_match_tree.is_none() {
            Some(MatchTree::build(
                &self.domain_sizes,
                numeric_domain_sizes,
                hash_multipliers,
                operators,
                &comparison_var_ids,
            ))
        } else {
            None
        };
        let match_tree = prebuilt_match_tree.unwrap_or_else(|| owned_match_tree.as_ref().unwrap());
        let (distances, generating_op_ids) = self.compute_distances_and_generating_ops(
            AbstractGoalDistanceSpace {
                task,
                operators,
                match_tree,
                goal_facts,
                layout: ComparisonBranchingLayout {
                    numeric_domain_sizes,
                    hash_multipliers,
                    comparison_var_ids: &comparison_var_ids,
                },
                num_states,
            },
            deadline,
        )?;

        let goal_facts = goal_facts.to_vec();
        let table = AbstractDistanceTable {
            distances,
            generating_op_ids,
            initial_state_hash: init_hash,
            goal_facts,
            hash_multipliers: hash_multipliers.to_vec(),
            numeric_domain_sizes: numeric_domain_sizes.to_vec(),
        };

        if dump_distances {
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

        let num_states = transition_system.backward.len();
        let mut distances = vec![f64::INFINITY; num_states];
        let mut generating_op_ids = vec![None; num_states];
        let mut heap: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();

        for &state_hash in &transition_system.goal_state_hashes {
            ensure!(
                state_hash < num_states,
                "goal state hash out of range: {state_hash} >= {num_states}"
            );
            distances[state_hash] = 0.0;
            heap.push((Reverse(NotNan::new(0.0).unwrap()), state_hash));
        }

        while let Some((Reverse(d), target_hash)) = heap.pop() {
            let d = d.into_inner();
            if d > distances[target_hash] + float_tolerance::DIJKSTRA_EPSILON {
                continue;
            }
            for &transition_id in &transition_system.backward[target_hash] {
                let transition = &transition_system.transitions[transition_id];
                let transition_cost = transition_costs[transition_id];
                if !transition_cost.is_finite() {
                    continue;
                }
                ensure!(
                    transition_cost >= -1e-9,
                    "transition costs must be nonnegative, got {transition_cost}"
                );
                let transition_cost = transition_cost.max(0.0);
                let alternative_cost = d + transition_cost;
                if alternative_cost + float_tolerance::DIJKSTRA_EPSILON
                    < distances[transition.source_hash]
                {
                    distances[transition.source_hash] = alternative_cost;
                    generating_op_ids[transition.source_hash] = Some(transition.abstract_op_id);
                    heap.push((
                        Reverse(NotNan::new(alternative_cost).context("alternative cost is NaN")?),
                        transition.source_hash,
                    ));
                }
            }
        }

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
        utils::dump_distances(self, task, table);
    }

    pub(super) fn compute_distances_and_generating_ops(
        &self,
        space: AbstractGoalDistanceSpace<'_>,
        deadline: Option<Instant>,
    ) -> Result<(Vec<f64>, Vec<Option<usize>>)> {
        let mut generating_op_ids = vec![None; space.num_states];
        match self.compute_goal_distances(
            space,
            GoalDistanceStop::Exhaust,
            Some(&mut generating_op_ids),
            deadline,
        )? {
            GoalDistanceResult::Exhausted(distances) => Ok((distances, generating_op_ids)),
            GoalDistanceResult::Reached(_) => {
                unreachable!("exhaustive goal-distance search returned an early distance")
            }
        }
    }

    pub(super) fn compute_goal_distances(
        &self,
        space: AbstractGoalDistanceSpace<'_>,
        stop: GoalDistanceStop,
        mut generating_op_ids: Option<&mut [Option<usize>]>,
        deadline: Option<Instant>,
    ) -> Result<GoalDistanceResult> {
        let AbstractGoalDistanceSpace {
            task,
            operators,
            match_tree,
            goal_facts,
            layout,
            num_states,
        } = space;
        let ComparisonBranchingLayout {
            numeric_domain_sizes,
            hash_multipliers,
            comparison_var_ids,
        } = layout;
        ensure_online_scp_deadline(deadline)?;
        let mut distances: Vec<f64> = vec![f64::INFINITY; num_states];
        if let Some(generating) = &generating_op_ids {
            ensure!(
                generating.len() == num_states,
                "generating-operator output has {} entries for {num_states} abstract states",
                generating.len()
            );
        }
        let mut heap: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();
        let mut comparison_enumeration_memo = ComparisonEnumerationMemo::default();
        let comparison_branching = !comparison_var_ids.is_empty();

        for (state_hash, distance) in distances.iter_mut().enumerate() {
            if state_hash % 1024 == 0 {
                ensure_online_scp_deadline(deadline)?;
            }
            if !self.is_goal_state(
                state_hash,
                goal_facts,
                numeric_domain_sizes,
                hash_multipliers,
            ) {
                continue;
            }
            if comparison_branching {
                let alts = self.enumerate_states_with_evaluated_comparisons_cached(
                    state_hash,
                    task,
                    layout,
                    &[],
                    &mut comparison_enumeration_memo,
                )?;
                if !alts.contains(&state_hash) {
                    continue;
                }
            }
            *distance = 0.0;
            heap.push((Reverse(NotNan::new(0.0).unwrap()), state_hash));
        }

        let comparison_preconditions = if comparison_branching {
            comparison_preconditions_by_operator(operators, comparison_var_ids)
        } else {
            Vec::new()
        };
        let mut applicable_operator_ids: Vec<usize> = Vec::new();
        while let Some((Reverse(d), state_hash)) = heap.pop() {
            if state_hash % 1024 == 0 {
                ensure_online_scp_deadline(deadline)?;
            }
            let d = d.into_inner();
            if d > distances[state_hash] + float_tolerance::DIJKSTRA_EPSILON {
                continue;
            }
            if matches!(stop, GoalDistanceStop::FirstReaching(target) if state_hash == target) {
                return Ok(GoalDistanceResult::Reached(d));
            }

            let base_state = if comparison_branching {
                self.clear_comparison_vars_except(
                    state_hash,
                    hash_multipliers,
                    comparison_var_ids,
                    &[],
                )?
            } else {
                state_hash
            };
            match_tree.get_applicable_operator_ids(base_state, &mut applicable_operator_ids);
            for &op_id in &applicable_operator_ids {
                let op = &operators[op_id];
                ensure!(op.cost.is_finite(), "abstract operator cost must be finite");
                let alternative_cost = d + op.cost;
                let predecessor_i64 = base_state as i64 + op.hash_effect as i64;
                if predecessor_i64 < 0 || predecessor_i64 >= num_states as i64 {
                    continue;
                }
                if comparison_branching {
                    let possible_predecessors = self
                        .enumerate_states_with_evaluated_comparisons_cached(
                            predecessor_i64 as usize,
                            task,
                            layout,
                            &comparison_preconditions[op_id],
                            &mut comparison_enumeration_memo,
                        )?;

                    for pred in possible_predecessors.iter().copied() {
                        debug_assert!(pred < num_states, "predecessor hash does not fit usize");
                        if alternative_cost + float_tolerance::DIJKSTRA_EPSILON < distances[pred] {
                            distances[pred] = alternative_cost;
                            if let Some(generating) = generating_op_ids.as_deref_mut() {
                                generating[pred] = Some(op_id);
                            }
                            heap.push((
                                Reverse(
                                    NotNan::new(alternative_cost)
                                        .context("alternative cost is NaN")?,
                                ),
                                pred,
                            ));
                        }
                    }
                } else {
                    let pred = predecessor_i64 as usize;
                    debug_assert!(pred < num_states, "predecessor hash does not fit usize");
                    if alternative_cost + float_tolerance::DIJKSTRA_EPSILON < distances[pred] {
                        distances[pred] = alternative_cost;
                        if let Some(generating) = generating_op_ids.as_deref_mut() {
                            generating[pred] = Some(op_id);
                        }
                        heap.push((
                            Reverse(
                                NotNan::new(alternative_cost).context("alternative cost is NaN")?,
                            ),
                            pred,
                        ));
                    }
                }
            }
        }

        match stop {
            GoalDistanceStop::Exhaust => Ok(GoalDistanceResult::Exhausted(distances)),
            GoalDistanceStop::FirstReaching(_) => Ok(GoalDistanceResult::Reached(f64::INFINITY)),
        }
    }

    pub(super) fn compute_distance_to_goal_state(
        &self,
        space: AbstractGoalDistanceSpace<'_>,
        target_state_hash: usize,
        deadline: Option<Instant>,
    ) -> Result<f64> {
        ensure!(
            target_state_hash < space.num_states,
            "target abstract state {target_state_hash} is out of bounds for {} states",
            space.num_states
        );
        match self.compute_goal_distances(
            space,
            GoalDistanceStop::FirstReaching(target_state_hash),
            None,
            deadline,
        )? {
            GoalDistanceResult::Reached(distance) => Ok(distance),
            GoalDistanceResult::Exhausted(_) => {
                unreachable!("early goal-distance search exhausted in exhaustive mode")
            }
        }
    }
}
