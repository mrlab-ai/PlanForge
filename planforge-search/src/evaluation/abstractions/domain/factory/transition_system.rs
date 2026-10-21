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
        let operators = self.abstract_operators(task, combine_labels, options.deadline)?;
        self.build_abstract_transition_system_from_operators(task, &operators, options)
    }

    pub fn build_abstract_transition_system_from_operators(
        &self,
        task: &dyn AbstractNumericTask,
        operators: &[AbstractOperator],
        options: DistanceTableOptions<'_>,
    ) -> Result<AbstractTransitionSystem> {
        self.build_transition_system_with_operators(
            task,
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
        operators: &[AbstractOperator],
        options: DistanceTableOptions<'_>,
    ) -> Result<OcpTransitionSystemBuild> {
        self.build_transition_system_with_operators(
            task,
            operators,
            TransitionSystemContents {
                materialize_cartesian_sets: false,
                include_self_loops: true,
                max_concrete_label_transitions: options.max_concrete_label_transitions,
            },
            options.deadline,
        )
    }

    pub(super) fn build_transition_system_with_operators(
        &self,
        task: &dyn AbstractNumericTask,
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
        let goal_facts = self.compute_abstract_goals(task);
        let num_states = self.layout.num_states();
        let match_tree = MatchTree::new(&self.layout, operators);

        let mut transitions: Vec<AbstractTransition> = Vec::with_capacity(num_states);
        let mut concrete_label_transition_count = 0usize;
        let mut backward: Vec<Vec<usize>> = vec![Vec::new(); num_states];
        let cartesian_sets: Vec<Arc<CartesianSet>> = if materialize_cartesian_sets {
            (0..num_states)
                .map(|state| Arc::new(self.state_region(state)))
                .collect()
        } else {
            Vec::new()
        };
        // Debug-only triple-uniqueness witness: every pushed AbstractTransition
        // must have a unique `(abstract_op_id, source_hash, target_hash)`.
        #[cfg(debug_assertions)]
        let mut seen_transition_triples: HashSet<(usize, usize, usize)> = HashSet::new();

        let mut graph = MatchTreeGraph::new(operators, &match_tree, num_states, &[]);
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
                let required = concrete_label_transition_count + op.concrete_op_ids.len();
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
            }
        }

        // Tight invariant: within one abstraction, every transition sharing an
        // `abstract_op_id` must have identical numeric source and target regions.
        // The partition-fact enumeration in `abstract_operators.rs`
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

        let goal_state_hashes = self.goal_states(task, &goal_facts, deadline)?;
        Ok(OcpTransitionSystemBuild::Complete(
            AbstractTransitionSystem {
                transitions,
                backward,
                goal_facts,
                goal_state_hashes,
                initial_state_hash: self.initial_state(task)?,
                cartesian_sets,
            },
        ))
    }

    /// The concrete operators that label a transition between two distinct
    /// abstract states.
    pub fn relevant_operator_ids(
        &self,
        task: &dyn AbstractNumericTask,
        operators: &[AbstractOperator],
    ) -> Vec<OperatorIndex> {
        let match_tree = MatchTree::new(&self.layout, operators);
        let mut graph = MatchTreeGraph::new(operators, &match_tree, self.layout.num_states(), &[]);
        let mut relevant = vec![false; task.get_operators().len()];
        let mut num_relevant = 0usize;
        let mut incoming = Vec::new();
        for target in 0..self.layout.num_states() {
            if num_relevant == relevant.len() {
                break;
            }
            graph
                .incoming(target, &mut incoming)
                .expect("abstract operators regress within the abstraction");
            for &(source, abstract_op_id, _) in &incoming {
                if source == target {
                    continue;
                }
                for &op_id in &operators[abstract_op_id].concrete_op_ids {
                    if !relevant[op_id.index()] {
                        relevant[op_id.index()] = true;
                        num_relevant += 1;
                    }
                }
            }
        }
        relevant
            .into_iter()
            .enumerate()
            .filter_map(|(op_id, relevant)| relevant.then_some(OperatorIndex::from_usize(op_id)))
            .collect()
    }
}
