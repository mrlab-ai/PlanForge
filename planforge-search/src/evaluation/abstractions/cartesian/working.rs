use super::*;

#[derive(Debug, Clone)]
pub(super) struct OperatorBitSet {
    pub(super) words: Box<[u64]>,
    pub(super) operator_count: usize,
}

impl OperatorBitSet {
    pub(super) fn empty(operator_count: usize) -> Self {
        Self {
            words: vec![0; operator_count.div_ceil(u64::BITS as usize)].into_boxed_slice(),
            operator_count,
        }
    }

    pub(super) fn insert(&mut self, operator_id: OperatorIndex) -> bool {
        debug_assert!(
            operator_id.index() < self.operator_count,
            "operator {operator_id} exceeds Cartesian operator-set size {}",
            self.operator_count
        );
        let word = &mut self.words[operator_id.index() / u64::BITS as usize];
        let mask = 1_u64 << (operator_id.index() % u64::BITS as usize);
        if *word & mask != 0 {
            return false;
        }
        *word |= mask;
        true
    }

    pub(super) fn contains(&self, operator_id: OperatorIndex) -> bool {
        debug_assert!(
            operator_id.index() < self.operator_count,
            "operator {operator_id} exceeds Cartesian operator-set size {}",
            self.operator_count
        );
        self.words[operator_id.index() / u64::BITS as usize]
            & (1_u64 << (operator_id.index() % u64::BITS as usize))
            != 0
    }

    pub(super) fn intersection_iter<'a>(
        &'a self,
        other: &'a Self,
    ) -> OperatorBitSetIntersectionIter<'a> {
        debug_assert_eq!(
            self.operator_count, other.operator_count,
            "cannot intersect Cartesian operator sets of different sizes"
        );
        OperatorBitSetIntersectionIter {
            left: &self.words,
            right: &other.words,
            operator_count: self.operator_count,
            word_id: 0,
            remaining: self.words.first().copied().unwrap_or(0)
                & other.words.first().copied().unwrap_or(0),
        }
    }

    pub(super) fn clone_without(&self, excluded: &Self) -> Self {
        debug_assert_eq!(
            self.operator_count, excluded.operator_count,
            "cannot subtract Cartesian operator sets of different sizes"
        );
        Self {
            words: self
                .words
                .iter()
                .zip(excluded.words.iter())
                .map(|(&word, &excluded_word)| word & !excluded_word)
                .collect(),
            operator_count: self.operator_count,
        }
    }
}

pub(super) struct OperatorBitSetIntersectionIter<'a> {
    pub(super) left: &'a [u64],
    pub(super) right: &'a [u64],
    pub(super) operator_count: usize,
    pub(super) word_id: usize,
    pub(super) remaining: u64,
}

impl Iterator for OperatorBitSetIntersectionIter<'_> {
    type Item = OperatorIndex;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.remaining != 0 {
                let bit = self.remaining.trailing_zeros() as usize;
                self.remaining &= self.remaining - 1;
                let operator_id = self.word_id * u64::BITS as usize + bit;
                debug_assert!(
                    operator_id < self.operator_count,
                    "Cartesian operator intersection has a set padding bit"
                );
                return Some(OperatorIndex::from_usize(operator_id));
            }
            self.word_id += 1;
            self.remaining = *self.left.get(self.word_id)? & self.right[self.word_id];
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct WorkingAbstraction {
    pub(super) states: Vec<CartesianSet>,
    pub(super) leaf_node_ids: Vec<usize>,
    pub(super) hierarchy: CartesianRefinementHierarchy,
    pub(super) transitions: Vec<Option<TransitionKey>>,
    pub(super) free_transition_ids: Vec<usize>,
    pub(super) outgoing: Vec<Vec<usize>>,
    pub(super) incoming: Vec<Vec<usize>>,
    pub(super) self_loop_operator_ids: Vec<OperatorBitSet>,
    pub(super) transition_ids_by_key: HashMap<TransitionKey, usize>,
}

impl WorkingAbstraction {
    pub(super) fn states(&self) -> &[CartesianSet] {
        &self.states
    }

    pub(super) fn hierarchy(&self) -> &CartesianRefinementHierarchy {
        &self.hierarchy
    }

    pub(super) fn outgoing(&self) -> &[Vec<usize>] {
        &self.outgoing
    }

    pub(super) fn incoming(&self) -> &[Vec<usize>] {
        &self.incoming
    }

    pub(super) fn self_loop_operator_ids(&self) -> &[OperatorBitSet] {
        &self.self_loop_operator_ids
    }

    pub(super) fn new(initial_region: CartesianSet, operator_count: usize) -> Self {
        Self {
            states: vec![initial_region],
            leaf_node_ids: vec![0],
            hierarchy: CartesianRefinementHierarchy::trivial(),
            transitions: Vec::new(),
            free_transition_ids: Vec::new(),
            outgoing: vec![Vec::new()],
            incoming: vec![Vec::new()],
            self_loop_operator_ids: vec![OperatorBitSet::empty(operator_count)],
            transition_ids_by_key: HashMap::new(),
        }
    }

    pub(super) fn add_transition(&mut self, source: usize, op_id: OperatorIndex, target: usize) {
        if source == target {
            self.self_loop_operator_ids[source].insert(op_id);
            return;
        }
        let key = TransitionKey {
            source,
            concrete_op_id: op_id,
            target,
        };
        if self.transition_ids_by_key.contains_key(&key) {
            return;
        }
        let transition = key;
        let transition_id = if let Some(transition_id) = self.free_transition_ids.pop() {
            assert!(
                self.transitions[transition_id].is_none(),
                "reused Cartesian transition slot is occupied"
            );
            self.transitions[transition_id] = Some(transition);
            transition_id
        } else {
            let transition_id = self.transitions.len();
            self.transitions.push(Some(transition));
            transition_id
        };
        let old = self.transition_ids_by_key.insert(key, transition_id);
        assert!(old.is_none(), "duplicate Cartesian transition key");
        self.outgoing[source].push(transition_id);
        self.incoming[target].push(transition_id);
    }

    pub(super) fn remove_transition(&mut self, transition_id: usize) -> TransitionKey {
        let transition = self.transitions[transition_id]
            .take()
            .expect("Cartesian adjacency references a removed transition");
        let removed_id = self.transition_ids_by_key.remove(&transition);
        assert_eq!(
            removed_id,
            Some(transition_id),
            "active Cartesian transition key is missing or inconsistent"
        );
        self.free_transition_ids.push(transition_id);
        transition
    }

    /// The transitions into or out of `state_id`, ascending.
    pub(super) fn incident_transition_ids(&self, state_id: usize) -> Vec<usize> {
        let mut incident = self.outgoing[state_id].clone();
        incident.extend(self.incoming[state_id].iter().copied());
        incident.sort_unstable();
        incident.dedup();
        incident
    }

    pub(super) fn remove_incident_transitions(&mut self, state_id: usize) -> Vec<TransitionKey> {
        let incident = self.incident_transition_ids(state_id);

        let mut old_transitions = Vec::with_capacity(incident.len());
        let mut changed_outgoing = Vec::with_capacity(incident.len());
        let mut changed_incoming = Vec::with_capacity(incident.len());
        for transition_id in incident {
            let transition = self.remove_transition(transition_id);
            changed_outgoing.push(transition.source);
            changed_incoming.push(transition.target);
            old_transitions.push(transition);
        }
        changed_outgoing.sort_unstable();
        changed_outgoing.dedup();
        changed_incoming.sort_unstable();
        changed_incoming.dedup();

        let transitions = &self.transitions;
        for source in changed_outgoing {
            self.outgoing[source].retain(|&id| transitions[id].is_some());
        }
        for target in changed_incoming {
            self.incoming[target].retain(|&id| transitions[id].is_some());
        }
        old_transitions
    }

    pub(super) fn active_transition_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.transitions
            .iter()
            .enumerate()
            .filter_map(|(id, transition)| transition.as_ref().map(|_| id))
    }

    pub(super) fn transition(&self, transition_id: usize) -> &TransitionKey {
        self.transitions[transition_id]
            .as_ref()
            .expect("Cartesian adjacency references a removed transition")
    }

    pub(super) fn contains_transition(&self, key: TransitionKey) -> bool {
        if key.source == key.target {
            self.self_loop_operator_ids[key.source].contains(key.concrete_op_id)
        } else {
            self.transition_ids_by_key.contains_key(&key)
        }
    }
}
