use super::*;

pub enum OcpTransitionSystemBuild {
    Complete(AbstractTransitionSystem),
    ConcreteLabelCapExceeded { required_at_least: usize },
}

impl OcpTransitionSystemBuild {
    fn into_complete(self) -> Result<AbstractTransitionSystem> {
        match self {
            Self::Complete(system) => Ok(system),
            Self::ConcreteLabelCapExceeded { .. } => {
                Err(anyhow!("transition cap triggered without a configured cap"))
            }
        }
    }
}

/// What a transition-system build keeps beyond the transitions themselves: the
/// per-state regions, the abstract self-loops that potential/abstraction OCP
/// needs for its `c(a) >= 0` constraints, and a cap on the concrete label
/// transitions it will record before reporting the cap exceeded.
#[derive(Clone, Copy)]
pub(super) struct TransitionSystemContents {
    materialize_state_regions: bool,
    include_self_loops: bool,
    max_concrete_label_transitions: Option<usize>,
}

impl DomainAbstractionFactory {
    pub fn build_abstract_transition_system(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        options: DistanceTableOptions<'_>,
    ) -> Result<AbstractTransitionSystem> {
        let mut generator = self.make_operator_generator(task, combine_labels)?;
        let operators = generator.build_abstract_operators(task)?;
        self.build_transition_system_with_operators(
            task,
            &generator,
            &operators,
            TransitionSystemContents {
                materialize_state_regions: options.materialize_state_regions,
                include_self_loops: options.include_self_loops,
                max_concrete_label_transitions: options.max_concrete_label_transitions,
            },
            options.deadline,
        )?
        .into_complete()
    }

    pub fn build_abstract_transition_system_from_operators(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operators: &[AbstractOperator],
        options: DistanceTableOptions<'_>,
    ) -> Result<AbstractTransitionSystem> {
        let generator = self.make_operator_generator(task, combine_labels)?;
        self.build_transition_system_with_operators(
            task,
            &generator,
            operators,
            TransitionSystemContents {
                materialize_state_regions: options.materialize_state_regions,
                include_self_loops: options.include_self_loops,
                max_concrete_label_transitions: options.max_concrete_label_transitions,
            },
            options.deadline,
        )?
        .into_complete()
    }

    /// Build the transition relation required by potential/abstraction OCP.
    /// Unlike shortest-path transition systems, this retains abstract
    /// self-loops: `d(s)-d(s) <= c(a)` is the necessary `c(a) >= 0`
    /// constraint for every concrete label that can stutter abstractly.
    pub fn build_ocp_transition_system_from_operators(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operators: &[AbstractOperator],
        options: DistanceTableOptions<'_>,
    ) -> Result<OcpTransitionSystemBuild> {
        let generator = self.make_operator_generator(task, combine_labels)?;
        self.build_transition_system_with_operators(
            task,
            &generator,
            operators,
            TransitionSystemContents {
                materialize_state_regions: false,
                include_self_loops: true,
                max_concrete_label_transitions: options.max_concrete_label_transitions,
            },
            options.deadline,
        )
    }

    pub fn relevant_operator_ids_from_operators(
        &self,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        operators: &[AbstractOperator],
        options: DistanceTableOptions<'_>,
    ) -> Result<Vec<usize>> {
        let generator = self.make_operator_generator(task, combine_labels)?;
        self.relevant_operator_ids_with_operators(task, &generator, operators, options.deadline)
    }

    pub(super) fn build_transition_system_with_operators(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        contents: TransitionSystemContents,
        deadline: Option<Instant>,
    ) -> Result<OcpTransitionSystemBuild> {
        let TransitionSystemContents {
            materialize_state_regions,
            include_self_loops,
            max_concrete_label_transitions,
        } = contents;
        ensure_online_scp_deadline(deadline)?;
        let hash_multipliers = generator.hash_multipliers();
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let comparison_var_ids = self.comparison_var_ids();
        let goal_facts = self.compute_abstract_goals(task);
        let init_hash = self.compute_initial_state_hash_determined(
            task,
            numeric_domain_sizes,
            hash_multipliers,
            &comparison_var_ids,
        )?;
        let num_states = compute_num_states(&self.domain_sizes, numeric_domain_sizes)?;
        let match_tree = MatchTree::build(
            &self.domain_sizes,
            numeric_domain_sizes,
            hash_multipliers,
            operators,
            &comparison_var_ids,
        );

        let mut transitions: Vec<AbstractTransition> = Vec::with_capacity(num_states);
        let mut concrete_label_transition_count = 0usize;
        let mut backward: Vec<Vec<usize>> = vec![Vec::new(); num_states];
        let mut forward: Vec<Vec<usize>> = vec![Vec::new(); num_states];
        let mut state_regions = Vec::new();
        if materialize_state_regions {
            state_regions.reserve(num_states);
            for state_hash in 0..num_states {
                state_regions.push(self.state_region_from_hash(
                    state_hash,
                    numeric_domain_sizes,
                    hash_multipliers,
                )?);
            }
        }
        let duplicate_transition_attempts = 0usize;
        let mut applicable_operator_ids: Vec<usize> = Vec::new();
        // Debug-only triple-uniqueness witness: every pushed AbstractTransition
        // must have a unique `(abstract_op_id, source_hash, target_hash)`.
        #[cfg(debug_assertions)]
        let mut seen_transition_triples: HashSet<(usize, usize, usize)> = HashSet::new();

        // Cascade-aware predecessor enumeration. When the abstraction has
        // refined comparison-axiom prop vars, an operator with `hash_effect=0`
        // on those vars can still transition between abstract states because
        // the comparison bit is evaluated from the post-update numeric state
        // (see `compute_distances_and_generating_ops`, which calls
        // `enumerate_states_with_evaluated_comparisons_cached` to find all
        // predecessors compatible with the operator's comparison preconditions).
        // Mirroring that here ensures the transition system records the same
        // edges Dijkstra walks; otherwise SCP's per-op saturated cost is
        // undercounted on cascade-only transitions, the residual stays high,
        // subsequent abstractions over-saturate the same operator, and the
        // sum exceeds the optimal — inadmissibility (plant-watering/prob_4_1_2
        // h=31 reproducer with `scp_online`).
        let comparison_branching = !comparison_var_ids.is_empty();
        let comparison_preconditions = if comparison_branching {
            comparison_preconditions_by_operator(operators, &comparison_var_ids)
        } else {
            Vec::new()
        };
        let mut comparison_enumeration_memo = ComparisonEnumerationMemo::default();

        for target_hash in 0..num_states {
            if target_hash % 64 == 0 {
                ensure_online_scp_deadline(deadline)?;
            }
            let base_target = if comparison_branching {
                let possible_targets = self.enumerate_states_with_evaluated_comparisons_cached(
                    target_hash,
                    task,
                    ComparisonBranchingLayout {
                        numeric_domain_sizes,
                        hash_multipliers,
                        comparison_var_ids: &comparison_var_ids,
                    },
                    &[],
                    &mut comparison_enumeration_memo,
                )?;
                if !possible_targets.contains(&target_hash) {
                    continue;
                }
                self.clear_comparison_vars_except(
                    target_hash,
                    hash_multipliers,
                    &comparison_var_ids,
                    &[],
                )?
            } else {
                target_hash
            };
            match_tree.get_applicable_operator_ids(base_target, &mut applicable_operator_ids);
            for &abstract_op_id in &applicable_operator_ids {
                let op = &operators[abstract_op_id];
                let predecessor_i64 = base_target as i64 + op.hash_effect as i64;
                if predecessor_i64 < 0 || predecessor_i64 >= num_states as i64 {
                    continue;
                }
                let base_predecessor = predecessor_i64 as usize;

                // Without comparison branching the predecessor is unique.
                // With comparison branching, the comparison var bits in the
                // predecessor hash are wildcarded — enumerate every state hash
                // whose comparison-axiom variables are consistent with the
                // operator's comparison preconditions.
                let push_source =
                    |source_hash: usize,
                     transitions: &mut Vec<AbstractTransition>,
                     concrete_label_transition_count: &mut usize,
                     backward: &mut Vec<Vec<usize>>,
                     forward: &mut Vec<Vec<usize>>,
                     #[cfg(debug_assertions)] seen: &mut HashSet<(usize, usize, usize)>|
                     -> bool {
                        if source_hash == target_hash && !include_self_loops {
                            return true;
                        }
                        let required = concrete_label_transition_count
                            .saturating_add(op.concrete_op_ids.len());
                        if max_concrete_label_transitions.is_some_and(|cap| required > cap) {
                            return false;
                        }
                        *concrete_label_transition_count = required;
                        #[cfg(debug_assertions)]
                        {
                            let triple = (abstract_op_id, source_hash, target_hash);
                            debug_assert!(
                                seen.insert(triple),
                                "duplicate AbstractTransition triple {:?}",
                                triple
                            );
                        }
                        let transition_id = transitions.len();
                        transitions.push(AbstractTransition {
                            transition_id,
                            abstract_op_id,
                            concrete_op_ids: op.concrete_op_ids.clone(),
                            source_hash,
                            target_hash,
                        });
                        backward[target_hash].push(transition_id);
                        forward[source_hash].push(transition_id);
                        true
                    };

                if comparison_branching {
                    let possible_predecessors = self
                        .enumerate_states_with_evaluated_comparisons_cached(
                            base_predecessor,
                            task,
                            ComparisonBranchingLayout {
                                numeric_domain_sizes,
                                hash_multipliers,
                                comparison_var_ids: &comparison_var_ids,
                            },
                            &comparison_preconditions[abstract_op_id],
                            &mut comparison_enumeration_memo,
                        )?;
                    for &source_hash in possible_predecessors.iter() {
                        if !push_source(
                            source_hash,
                            &mut transitions,
                            &mut concrete_label_transition_count,
                            &mut backward,
                            &mut forward,
                            #[cfg(debug_assertions)]
                            &mut seen_transition_triples,
                        ) {
                            return Ok(OcpTransitionSystemBuild::ConcreteLabelCapExceeded {
                                required_at_least: concrete_label_transition_count
                                    .saturating_add(op.concrete_op_ids.len()),
                            });
                        }
                    }
                } else {
                    if !push_source(
                        base_predecessor,
                        &mut transitions,
                        &mut concrete_label_transition_count,
                        &mut backward,
                        &mut forward,
                        #[cfg(debug_assertions)]
                        &mut seen_transition_triples,
                    ) {
                        return Ok(OcpTransitionSystemBuild::ConcreteLabelCapExceeded {
                            required_at_least: concrete_label_transition_count
                                .saturating_add(op.concrete_op_ids.len()),
                        });
                    }
                }
            }
        }

        // Tight invariant: within one abstraction, every transition sharing an
        // `abstract_op_id` must have identical numeric source and target regions.
        // The partition-fact enumeration in `abstract_operator_generator.rs`
        // (build_abstract_operators → enumerate_partition_combos) bakes the
        // numeric (source_partition, target_partition) pair into the abstract
        // operator's identity, so two transitions sharing the abstract op can
        // only differ in propositional wildcard dimensions of `source_hash` /
        // `target_hash`. This homogeneity is the property that lets the
        // finite-support cost-partitioning gate decide stealability per abstract
        // op rather than per individual transition.
        #[cfg(debug_assertions)]
        if materialize_state_regions {
            let mut representative_per_op: HashMap<usize, (usize, usize)> = HashMap::new();
            for transition in &transitions {
                match representative_per_op.get(&transition.abstract_op_id) {
                    Some(&(rep_src_hash, rep_tgt_hash)) => {
                        debug_assert_eq!(
                            state_regions[rep_src_hash].numeric,
                            state_regions[transition.source_hash].numeric,
                            "abstract_op_id {} has transitions with differing numeric source regions",
                            transition.abstract_op_id
                        );
                        debug_assert_eq!(
                            state_regions[rep_tgt_hash].numeric,
                            state_regions[transition.target_hash].numeric,
                            "abstract_op_id {} has transitions with differing numeric target regions",
                            transition.abstract_op_id
                        );
                    }
                    None => {
                        representative_per_op.insert(
                            transition.abstract_op_id,
                            (transition.source_hash, transition.target_hash),
                        );
                    }
                }
            }
        }

        // Goal states are simply those whose hash matches the goal facts.
        // The old self-consistency check via
        // `enumerate_states_with_evaluated_comparisons` filtered to states
        // whose comparison bits agreed with the (potentially ambiguous)
        // interval evaluation — that filtering is no longer needed because
        // operators only land transitions in states with the *optimistic*
        // comparison bit, and the initial state hash is computed with the
        // same optimistic semantics, so every reachable state has
        // self-consistent bits by construction.
        let mut goal_state_hashes = Vec::new();
        for state_hash in 0..num_states {
            if self.is_goal_state(
                state_hash,
                &goal_facts,
                numeric_domain_sizes,
                hash_multipliers,
            ) {
                if comparison_branching {
                    let possible_states = self.enumerate_states_with_evaluated_comparisons_cached(
                        state_hash,
                        task,
                        ComparisonBranchingLayout {
                            numeric_domain_sizes,
                            hash_multipliers,
                            comparison_var_ids: &comparison_var_ids,
                        },
                        &[],
                        &mut comparison_enumeration_memo,
                    )?;
                    if !possible_states.contains(&state_hash) {
                        continue;
                    }
                }
                goal_state_hashes.push(state_hash);
            }
        }

        Ok(OcpTransitionSystemBuild::Complete(
            AbstractTransitionSystem {
                transitions,
                duplicate_transition_attempts,
                backward,
                forward,
                goal_facts,
                goal_state_hashes,
                initial_state_hash: init_hash,
                hash_multipliers: hash_multipliers.to_vec(),
                numeric_domain_sizes: numeric_domain_sizes.to_vec(),
                state_regions: state_regions.into_iter().map(Arc::new).collect(),
            },
        ))
    }

    pub(super) fn relevant_operator_ids_with_operators(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        deadline: Option<Instant>,
    ) -> Result<Vec<usize>> {
        ensure_online_scp_deadline(deadline)?;
        let hash_multipliers = generator.hash_multipliers();
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let comparison_var_ids = self.comparison_var_ids();
        let num_states = compute_num_states(&self.domain_sizes, numeric_domain_sizes)?;
        let match_tree = MatchTree::build(
            &self.domain_sizes,
            numeric_domain_sizes,
            hash_multipliers,
            operators,
            &comparison_var_ids,
        );
        let mut seen_operator_ids = vec![false; task.get_operators().len()];
        let mut num_seen = 0usize;
        let mut applicable_operator_ids: Vec<usize> = Vec::new();

        for target_hash in 0..num_states {
            if target_hash % 64 == 0 {
                ensure_online_scp_deadline(deadline)?;
            }
            if num_seen == seen_operator_ids.len() {
                break;
            }
            match_tree.get_applicable_operator_ids(target_hash, &mut applicable_operator_ids);
            for &abstract_op_id in &applicable_operator_ids {
                let op = &operators[abstract_op_id];
                if op
                    .concrete_op_ids
                    .iter()
                    .all(|&op_id| seen_operator_ids.get(op_id).copied().unwrap_or(false))
                {
                    continue;
                }
                let predecessor_i64 = target_hash as i64 + op.hash_effect as i64;
                if predecessor_i64 < 0 || predecessor_i64 >= num_states as i64 {
                    continue;
                }
                let source_hash = predecessor_i64 as usize;
                if source_hash == target_hash {
                    continue;
                }
                for &op_id in &op.concrete_op_ids {
                    ensure!(
                        op_id < seen_operator_ids.len(),
                        "concrete operator id out of range: {op_id} >= {}",
                        seen_operator_ids.len()
                    );
                    if !seen_operator_ids[op_id] {
                        seen_operator_ids[op_id] = true;
                        num_seen += 1;
                    }
                }
            }
        }

        // Cascade-relevance: an operator is also relevant to this abstraction
        // if it modifies a numeric variable that feeds a comparison-axiom prop
        // var refined in this abstraction. The hash_effect-based check above
        // misses these because `compute_comparison_transition_facts` does not
        // bake cascade source/target facts into operator pre/eff. Mirrors
        // numeric-FD's `TaskInfo::operator_is_active`
        // (cost_saturation/projection.cc:421-425). Without this, canonical's
        // additive-subset check claims two abstractions disjoint when they
        // share a cascade-only operator, and summing their heuristics
        // double-counts that operator's cost — producing an inadmissible
        // canonical heuristic (sailing/plant-watering reproducers).
        if !comparison_var_ids.is_empty() {
            let mut cascade_numeric_deps: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            for &cmp_var_id in &comparison_var_ids {
                if let Some(condition) = self.numeric_conditions.for_var(cmp_var_id) {
                    cascade_numeric_deps
                        .extend(condition.regular_numeric_var_dependencies().iter().copied());
                }
            }
            if !cascade_numeric_deps.is_empty() {
                for (concrete_op_id, op) in task.get_operators().iter().enumerate() {
                    if seen_operator_ids
                        .get(concrete_op_id)
                        .copied()
                        .unwrap_or(false)
                    {
                        continue;
                    }
                    if op
                        .assignment_effects()
                        .iter()
                        .any(|eff| cascade_numeric_deps.contains(&eff.affected_var_id()))
                    {
                        seen_operator_ids[concrete_op_id] = true;
                    }
                }
            }
        }

        Ok(seen_operator_ids
            .into_iter()
            .enumerate()
            .filter_map(|(op_id, seen)| seen.then_some(op_id))
            .collect())
    }
}
