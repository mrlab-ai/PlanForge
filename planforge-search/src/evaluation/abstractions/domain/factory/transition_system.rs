use planforge_sas::numeric_task::OperatorIndex;

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
    materialize_cartesian_sets: bool,
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
                materialize_cartesian_sets: options.materialize_cartesian_sets,
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
                materialize_cartesian_sets: options.materialize_cartesian_sets,
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
                materialize_cartesian_sets: false,
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
    ) -> Result<Vec<OperatorIndex>> {
        let generator = self.make_operator_generator(task, combine_labels)?;
        self.relevant_operator_ids_with_operators(task, &generator, operators, options.deadline)
    }

    pub(super) fn build_transition_system_with_operators(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        contents: TransitionSystemContents,
        deadline: Deadline,
    ) -> Result<OcpTransitionSystemBuild> {
        let TransitionSystemContents {
            materialize_cartesian_sets,
            include_self_loops,
            max_concrete_label_transitions,
        } = contents;
        deadline.ensure(ONLINE_SCP)?;
        let hash_multipliers = generator.hash_multipliers();
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let goal_facts = self.compute_abstract_goals(task);
        let init_hash = self.compute_initial_state_hash_determined(
            task,
            numeric_domain_sizes,
            hash_multipliers,
        )?;
        let num_states = compute_num_states(&self.domain_sizes, numeric_domain_sizes)?;
        let match_tree = MatchTree::new(generator, operators);

        let mut transitions: Vec<AbstractTransition> = Vec::with_capacity(num_states);
        let mut concrete_label_transition_count = 0usize;
        let mut backward: Vec<Vec<usize>> = vec![Vec::new(); num_states];
        let mut forward: Vec<Vec<usize>> = vec![Vec::new(); num_states];
        let mut cartesian_sets = Vec::new();
        if materialize_cartesian_sets {
            cartesian_sets.reserve(num_states);
            for state_hash in 0..num_states {
                cartesian_sets.push(self.cartesian_set_from_hash(
                    state_hash,
                    numeric_domain_sizes,
                    hash_multipliers,
                )?);
            }
        }
        // Debug-only triple-uniqueness witness: every pushed AbstractTransition
        // must have a unique `(abstract_op_id, source_hash, target_hash)`.
        #[cfg(debug_assertions)]
        let mut seen_transition_triples: HashSet<(usize, usize, usize)> = HashSet::new();

        let mut graph = MatchTreeGraph {
            operators,
            match_tree: &match_tree,
            num_states,
            applicable: Vec::new(),
        };
        let mut incoming = Vec::new();
        for (target_hash, target_backward) in backward.iter_mut().enumerate() {
            if target_hash % 64 == 0 {
                deadline.ensure(ONLINE_SCP)?;
            }
            graph.incoming(target_hash, &mut incoming)?;
            for &(source_hash, abstract_op_id, _) in &incoming {
                if source_hash == target_hash && !include_self_loops {
                    continue;
                }
                let op = &operators[abstract_op_id];
                let required =
                    concrete_label_transition_count.saturating_add(op.concrete_op_ids.len());
                if max_concrete_label_transitions.is_some_and(|cap| required > cap) {
                    return Ok(OcpTransitionSystemBuild::ConcreteLabelCapExceeded {
                        required_at_least: required,
                    });
                }
                concrete_label_transition_count = required;
                #[cfg(debug_assertions)]
                {
                    let triple = (abstract_op_id, source_hash, target_hash);
                    debug_assert!(
                        seen_transition_triples.insert(triple),
                        "duplicate AbstractTransition triple {triple:?}"
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
                target_backward.push(transition_id);
                forward[source_hash].push(transition_id);
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
        if materialize_cartesian_sets {
            let mut representative_per_op: std::collections::HashMap<usize, (usize, usize)> =
                std::collections::HashMap::new();
            for transition in &transitions {
                match representative_per_op.get(&transition.abstract_op_id) {
                    Some(&(rep_src_hash, rep_tgt_hash)) => {
                        debug_assert_eq!(
                            cartesian_sets[rep_src_hash].numeric,
                            cartesian_sets[transition.source_hash].numeric,
                            "abstract_op_id {} has transitions with differing numeric source regions",
                            transition.abstract_op_id
                        );
                        debug_assert_eq!(
                            cartesian_sets[rep_tgt_hash].numeric,
                            cartesian_sets[transition.target_hash].numeric,
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

        // Goal states are those whose hash matches the goal facts.
        let mut goal_state_hashes = Vec::new();
        for state_hash in 0..num_states {
            if self.is_goal_state(
                task,
                state_hash,
                &goal_facts,
                numeric_domain_sizes,
                hash_multipliers,
            )? {
                goal_state_hashes.push(state_hash);
            }
        }

        Ok(OcpTransitionSystemBuild::Complete(
            AbstractTransitionSystem {
                transitions,
                backward,
                forward,
                goal_facts,
                goal_state_hashes,
                initial_state_hash: init_hash,
                hash_multipliers: hash_multipliers.to_vec(),
                numeric_domain_sizes: numeric_domain_sizes.to_vec(),
                cartesian_sets: cartesian_sets.into_iter().map(Arc::new).collect(),
            },
        ))
    }

    pub(super) fn relevant_operator_ids_with_operators(
        &self,
        task: &dyn AbstractNumericTask,
        generator: &AbstractOperatorGenerator,
        operators: &[AbstractOperator],
        deadline: Deadline,
    ) -> Result<Vec<OperatorIndex>> {
        deadline.ensure(ONLINE_SCP)?;
        let numeric_domain_sizes = generator.numeric_domain_sizes();
        let num_states = compute_num_states(&self.domain_sizes, numeric_domain_sizes)?;
        let match_tree = MatchTree::new(generator, operators);
        let mut seen_operator_ids = vec![false; task.get_operators().len()];
        let mut num_seen = 0usize;
        let mut applicable_operator_ids: Vec<usize> = Vec::new();

        for target_hash in 0..num_states {
            if target_hash % 64 == 0 {
                deadline.ensure(ONLINE_SCP)?;
            }
            if num_seen == seen_operator_ids.len() {
                break;
            }
            match_tree.get_applicable_operator_ids(target_hash, &mut applicable_operator_ids);
            for &abstract_op_id in &applicable_operator_ids {
                let op = &operators[abstract_op_id];
                if op.concrete_op_ids.iter().all(|&op_id| {
                    seen_operator_ids
                        .get(op_id.index())
                        .copied()
                        .unwrap_or(false)
                }) {
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
                        op_id.index() < seen_operator_ids.len(),
                        "concrete operator id out of range: {op_id} >= {}",
                        seen_operator_ids.len()
                    );
                    if !seen_operator_ids[op_id.index()] {
                        seen_operator_ids[op_id.index()] = true;
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

        Ok(seen_operator_ids
            .into_iter()
            .enumerate()
            .filter_map(|(op_id, seen)| seen.then_some(OperatorIndex::from_usize(op_id)))
            .collect())
    }
}
