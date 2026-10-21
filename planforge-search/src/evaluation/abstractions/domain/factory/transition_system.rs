use super::*;

pub enum OcpTransitionSystemBuild {
    Complete(AbstractTransitionSystem),
    ConcreteLabelCapExceeded { required_at_least: usize },
}

/// What a transition-system build keeps beyond the transitions themselves:
/// the abstract self-loops that potential/abstraction OCP needs for its
/// `c(a) >= 0` constraints, and a cap on the concrete label transitions it
/// will record before reporting the cap exceeded.
#[derive(Clone, Copy)]
pub(super) struct TransitionSystemContents {
    include_self_loops: bool,
    max_concrete_label_transitions: Option<usize>,
}

impl DomainAbstractionFactory {
    /// The explicit transition system of `operators`, without self-loops.
    pub fn build_transition_system(
        &self,
        task: &dyn AbstractNumericTask,
        operators: &[AbstractOperator],
        deadline: Deadline,
    ) -> Result<AbstractTransitionSystem> {
        match self.build_transition_system_with_operators(
            task,
            operators,
            TransitionSystemContents {
                include_self_loops: false,
                max_concrete_label_transitions: None,
            },
            deadline,
        )? {
            OcpTransitionSystemBuild::Complete(system) => Ok(system),
            OcpTransitionSystemBuild::ConcreteLabelCapExceeded { .. } => {
                unreachable!("an uncapped transition system build exceeds no cap")
            }
        }
    }

    /// Build the transition relation required by potential/abstraction OCP.
    /// Unlike shortest-path transition systems, this retains abstract
    /// self-loops: `d(s)-d(s) <= c(a)` is the necessary `c(a) >= 0`
    /// constraint for every concrete label that can stutter abstractly.
    pub fn build_ocp_transition_system(
        &self,
        task: &dyn AbstractNumericTask,
        operators: &[AbstractOperator],
        max_concrete_label_transitions: usize,
        deadline: Deadline,
    ) -> Result<OcpTransitionSystemBuild> {
        self.build_transition_system_with_operators(
            task,
            operators,
            TransitionSystemContents {
                include_self_loops: true,
                max_concrete_label_transitions: Some(max_concrete_label_transitions),
            },
            deadline,
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
            include_self_loops,
            max_concrete_label_transitions,
        } = contents;
        deadline.ensure(TRANSITION_SYSTEM)?;
        let num_states = self.layout.num_states();
        let match_tree = MatchTree::new(&self.layout, operators);

        let mut transitions: Vec<AbstractTransition> = Vec::with_capacity(num_states);
        let mut concrete_label_transition_count = 0usize;
        let mut backward: Vec<Vec<usize>> = vec![Vec::new(); num_states];
        // Debug-only triple-uniqueness witness: every pushed AbstractTransition
        // must have a unique `(abstract_op_id, source_state, target_state)`.
        #[cfg(debug_assertions)]
        let mut seen_transition_triples: HashSet<(usize, usize, usize)> = HashSet::new();

        let mut graph = MatchTreeGraph::new(operators, &match_tree, num_states, &[]);
        let mut incoming = Vec::new();
        for (target_state, target_backward) in backward.iter_mut().enumerate() {
            if target_state % 64 == 0 {
                deadline.ensure(TRANSITION_SYSTEM)?;
            }
            graph.incoming(target_state, &mut incoming)?;
            for &(source_state, abstract_op_id, _) in &incoming {
                if source_state == target_state && !include_self_loops {
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
                    let triple = (abstract_op_id, source_state, target_state);
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
                    source_state,
                    target_state,
                });
                target_backward.push(transition_id);
            }
        }

        let goal_states = self.goal_states(task, &self.compute_abstract_goals(task), deadline)?;
        Ok(OcpTransitionSystemBuild::Complete(
            AbstractTransitionSystem {
                transitions,
                backward,
                goal_states,
            },
        ))
    }
}
