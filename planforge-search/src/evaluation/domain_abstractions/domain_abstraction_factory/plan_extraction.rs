use super::*;

/// A solved abstraction a wildcard plan can be read back from: the task and the
/// generator that fixes its hash layout, the abstract operators with the match
/// tree indexing them, the goal-distance table the plan descends, and the
/// comparison variables that branch when a state's conditions are re-evaluated.
#[derive(Clone, Copy)]
pub(super) struct SolvedAbstraction<'a> {
    pub(super) task: &'a dyn AbstractNumericTask,
    pub(super) generator: &'a AbstractOperatorGenerator,
    pub(super) operators: &'a [AbstractOperator],
    pub(super) table: &'a AbstractDistanceTable,
    pub(super) match_tree: &'a MatchTree,
    pub(super) comparison_var_ids: &'a [usize],
}

#[derive(Debug, Clone)]
pub struct WildcardPlanResult {
    // Per-step set of concrete operator IDs.
    pub wildcard_plan: Vec<Vec<usize>>,
    // Path of abstract state hashes (`len = steps+1`).
    pub abstract_state_hashes: Vec<usize>,
    // Decoded propositional values along path.
    pub abstract_prop_states: Vec<Vec<usize>>,
    // Decoded numeric partitions along path.
    pub abstract_numeric_states: Vec<Vec<usize>>,
}

impl DomainAbstractionFactory {
    /// Computes an abstract wildcard plan (sequence of per-step concrete-op-ID sets) by:
    /// 1) Computing abstract goal distances with implicit regression Dijkstra.
    /// 2) Extracting a shortest-path abstract plan from the initial abstract state.
    /// 3) Collecting all cheapest realizations per step.
    pub fn compute_wildcard_plan(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        dump_distances: bool,
    ) -> Result<Option<WildcardPlanResult>> {
        self.compute_plan(task, combine_labels, dump_distances, true)
    }

    pub fn compute_plan(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        dump_distances: bool,
        use_wildcard_plans: bool,
    ) -> Result<Option<WildcardPlanResult>> {
        let mut local_rng = Some(SmallRng::seed_from_u64(
            crate::evaluation::DEFAULT_RANDOM_SEED,
        ));
        self.compute_plan_with_rng(
            task,
            combine_labels,
            use_wildcard_plans,
            local_rng.as_mut(),
            DistanceTableOptions {
                dump_distances,
                ..DistanceTableOptions::default()
            },
        )
    }

    pub(crate) fn compute_plan_with_rng(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        use_wildcard_plans: bool,
        plan_step_rng: Option<&mut SmallRng>,
        options: DistanceTableOptions<'_>,
    ) -> Result<Option<WildcardPlanResult>> {
        let deadline = options.deadline;
        ensure_online_scp_deadline(deadline)?;
        let start = Instant::now();
        let mut generator = self.make_operator_generator(task, combine_labels)?;
        debug!(
            "domain abstraction factory: operator generator prepared in {:.3}s",
            start.elapsed().as_secs_f64()
        );
        let operator_start = Instant::now();
        let operators = generator.build_abstract_operators_with_deadline(task, deadline)?;
        debug!(
            "domain abstraction factory: built {} abstract operators in {:.3}s",
            operators.len(),
            operator_start.elapsed().as_secs_f64()
        );
        ensure_online_scp_deadline(deadline)?;
        let table_start = Instant::now();
        let goal_facts = self.compute_abstract_goals(task);
        let table = self.build_distance_table_with_operators_for_goals_inner(
            task,
            &generator,
            &operators,
            &goal_facts,
            DistanceTableOptions {
                dump_distances: options.dump_distances,
                deadline,
                ..DistanceTableOptions::default()
            },
        )?;
        debug!(
            "domain abstraction factory: built abstract distance table with {} states in {:.3}s",
            table.distances.len(),
            table_start.elapsed().as_secs_f64()
        );
        ensure_online_scp_deadline(deadline)?;

        let comparison_var_ids = self.comparison_var_ids();
        let match_tree_start = Instant::now();
        let match_tree = MatchTree::build(
            generator.domain_sizes(),
            generator.numeric_domain_sizes(),
            generator.hash_multipliers(),
            &operators,
            &comparison_var_ids,
        );
        debug!(
            "domain abstraction factory: built match tree in {:.3}s",
            match_tree_start.elapsed().as_secs_f64()
        );
        ensure_online_scp_deadline(deadline)?;

        let plan_start = Instant::now();
        let plan = self.compute_wildcard_plan_from_table(
            SolvedAbstraction {
                task,
                generator: &generator,
                operators: &operators,
                table: &table,
                match_tree: &match_tree,
                comparison_var_ids: &comparison_var_ids,
            },
            use_wildcard_plans,
            plan_step_rng,
            deadline,
        );
        debug!(
            "domain abstraction factory: extracted wildcard plan in {:.3}s",
            plan_start.elapsed().as_secs_f64()
        );
        plan
    }

    pub(super) fn compute_wildcard_plan_from_table(
        &self,
        abstraction: SolvedAbstraction<'_>,
        use_wildcard_plans: bool,
        mut plan_step_rng: Option<&mut SmallRng>,
        deadline: Option<Instant>,
    ) -> Result<Option<WildcardPlanResult>> {
        let SolvedAbstraction {
            task,
            generator,
            operators,
            table,
            match_tree,
            comparison_var_ids,
        } = abstraction;
        ensure_online_scp_deadline(deadline)?;
        let domain_sizes = generator.domain_sizes();
        let hash_multipliers = generator.hash_multipliers();
        let num_props = domain_sizes.len();
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let comparison_branching = !comparison_var_ids.is_empty();

        let dist = &table.distances;
        let generating_op = &table.generating_op_ids;
        let comparison_preconditions = if comparison_branching {
            comparison_preconditions_by_operator(operators, comparison_var_ids)
        } else {
            Vec::new()
        };
        let mut comparison_enumeration_memo = ComparisonEnumerationMemo::default();

        let mut current_hash = table.initial_state_hash;
        if current_hash >= dist.len() || !dist[current_hash].is_finite() {
            return Ok(None);
        }

        let mut wildcard_plan: Vec<Vec<usize>> = Vec::new();
        let mut abstract_state_hashes: Vec<usize> = vec![current_hash];
        let mut seen_states: Vec<usize> = Vec::new();

        // For debugging / parity with numeric-fd deviation code.
        let mut abstract_prop_states: Vec<Vec<usize>> = Vec::new();
        let mut abstract_numeric_states: Vec<Vec<usize>> = Vec::new();
        decode_state_to_vectors(
            current_hash,
            num_props,
            domain_sizes,
            numeric_domain_sizes,
            hash_multipliers,
            &mut abstract_prop_states,
            &mut abstract_numeric_states,
        );

        let mut safety_steps = 0usize;
        while !self.is_goal_state(
            current_hash,
            &table.goal_facts,
            numeric_domain_sizes,
            hash_multipliers,
        ) {
            if safety_steps.is_multiple_of(64) {
                ensure_online_scp_deadline(deadline)?;
            }
            safety_steps += 1;
            if safety_steps > dist.len() + 1 {
                bail!("abstract plan extraction exceeded safety limit")
            }
            let Some(op_id) = generating_op.get(current_hash).copied().flatten() else {
                bail!("missing generating operator for state {current_hash} with finite distance");
            };
            let op = operators
                .get(op_id)
                .with_context(|| format!("generating op id out of bounds: {op_id}"))?;
            let candidate_hash_effect = op.hash_effect;
            let base_successor_i64 = current_hash as i64 - candidate_hash_effect as i64;
            ensure!(
                base_successor_i64 >= 0 && base_successor_i64 < dist.len() as i64,
                "plan-extraction base successor out of range for state {current_hash} with op {op_id}"
            );
            let base_successor = if comparison_branching {
                self.clear_comparison_vars_except(
                    base_successor_i64 as usize,
                    hash_multipliers,
                    comparison_var_ids,
                    &[],
                )?
            } else {
                base_successor_i64 as usize
            };
            let cur_d = dist[current_hash];
            ensure!(cur_d.is_finite(), "current distance must be finite");

            let mut chosen_successor: Option<usize> = None;
            let mut lowest_so_far = cur_d;
            let mut consider_successor = |cand: usize| {
                if cand == current_hash {
                    return;
                }
                if seen_states.contains(&cand) {
                    return;
                }
                let cd = dist[cand];
                if !cd.is_finite() {
                    return;
                }
                // Classify op cost with `float_tolerance::ABS_EPSILON` (1e-12) instead of strict
                // 0/!=0 so canonicalization-snapped near-zero costs (state_registry.rs:1661 grid)
                // don't fall through both branches. Mirrors numeric-fd's tolerant if/else
                // structure (domain_abstraction_factory.cc:1500/1524).
                let is_zero_cost = op.cost.abs() <= float_tolerance::ABS_EPSILON;
                let valid_progress = if is_zero_cost {
                    (cd - cur_d).abs() <= 1e-9
                } else {
                    cd < cur_d
                };
                if valid_progress && chosen_successor.is_none_or(|x| cand > x) {
                    chosen_successor = Some(cand);
                    lowest_so_far = cd;
                }
            };
            if comparison_branching {
                let possible_successors = self.enumerate_states_with_evaluated_comparisons_cached(
                    base_successor,
                    task,
                    ComparisonBranchingLayout {
                        numeric_domain_sizes,
                        hash_multipliers,
                        comparison_var_ids,
                    },
                    &[],
                    &mut comparison_enumeration_memo,
                )?;
                for cand in possible_successors.iter().copied() {
                    consider_successor(cand);
                }
            } else {
                consider_successor(base_successor);
            }
            let successor_hash = chosen_successor.with_context(|| {
                format!(
                    "plan-extraction: no successor satisfies dist equation for state {current_hash} with op {op_id} (cur_d={cur_d}, op.cost={})",
                    op.cost
                )
            })?;
            ensure!(
                successor_hash < dist.len(),
                "successor hash out of range: {successor_hash}"
            );
            ensure!(
                (lowest_so_far - cur_d + op.cost).abs() <= 1e-6,
                "chosen successor violates plan-extraction distance relation"
            );
            let required_cost = op.cost;

            let mut step: Vec<usize> = Vec::new();
            let mut applicable_operator_ids: Vec<usize> = Vec::new();
            match_tree.get_applicable_operator_ids(base_successor, &mut applicable_operator_ids);
            for &cand_op_id in &applicable_operator_ids {
                let cand_op = operators
                    .get(cand_op_id)
                    .with_context(|| format!("candidate op id out of bounds: {cand_op_id}"))?;
                if (cand_op.cost - required_cost).abs() > 1e-9 {
                    continue;
                }
                let cand_pred_i64 = base_successor as i64 + cand_op.hash_effect as i64;
                if cand_pred_i64 < 0 || cand_pred_i64 >= dist.len() as i64 {
                    continue;
                }
                let contains_current = if comparison_branching {
                    let possible_predecessors = self
                        .enumerate_states_with_evaluated_comparisons_cached(
                            cand_pred_i64 as usize,
                            task,
                            ComparisonBranchingLayout {
                                numeric_domain_sizes,
                                hash_multipliers,
                                comparison_var_ids,
                            },
                            &comparison_preconditions[cand_op_id],
                            &mut comparison_enumeration_memo,
                        )?;
                    possible_predecessors.contains(&current_hash)
                } else {
                    cand_pred_i64 as usize == current_hash
                };
                if contains_current {
                    step = cand_op.concrete_op_ids.clone();
                    step.sort_unstable();
                    step.dedup();
                    if use_wildcard_plans {
                        if let Some(rng) = plan_step_rng.as_deref_mut() {
                            step.shuffle(rng);
                        }
                    } else {
                        let selected_op = match plan_step_rng.as_deref_mut() {
                            Some(rng) => step.choose(rng).copied(),
                            None => step.first().copied(),
                        }
                        .with_context(|| {
                            format!(
                                "failed to choose a representative concrete operator for abstract state {current_hash}"
                            )
                        })?;
                        step.clear();
                        step.push(selected_op);
                    }
                    break;
                }
            }
            ensure!(
                !step.is_empty(),
                "failed to extract a concrete plan step for abstract state {current_hash}"
            );
            wildcard_plan.push(step);

            seen_states.push(current_hash);
            current_hash = successor_hash;
            abstract_state_hashes.push(current_hash);
            decode_state_to_vectors(
                current_hash,
                num_props,
                domain_sizes,
                numeric_domain_sizes,
                hash_multipliers,
                &mut abstract_prop_states,
                &mut abstract_numeric_states,
            );
        }

        Ok(Some(WildcardPlanResult {
            wildcard_plan,
            abstract_state_hashes,
            abstract_prop_states,
            abstract_numeric_states,
        }))
    }
}
