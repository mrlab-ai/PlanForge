#[cfg(test)]
mod tests;

use std::{
    cell::{Ref, RefCell, RefMut},
    collections::HashSet,
};

use anyhow::Result;
use planners_sas::numeric::{
    axioms::AxiomEvaluator,
    numeric_task::{AbstractNumericTask, ExplicitFact, Operator},
};

use crate::numeric::evaluation::domain_abstractions::{
    abstract_operator_generator::DomainMapping,
    cegar::{
        CegarConfig, RefinementSummary,
        flaw_search::{DependentNumericRefinement, Flaw, NumericFlaw},
    },
    domain_abstraction::NumericPartitions,
    utils::{
        abstract_state_values_for_concrete_state, compute_abstraction_size_u128, get_goals,
        get_initial_state, get_post, make_prop_state_packer, partition_for_value,
    },
};

#[derive(Clone, Debug)]
pub struct Refinement {
    var: usize,
    value: usize,
    next_value: usize,
    n_states_before_refinement: usize,
    numeric: bool,
}

#[derive(Clone, Debug)]
pub struct Transition {
    op_id: usize,
    target: usize,
}

pub type Transitions = Vec<Transition>;
pub type Loops = Vec<usize>;

#[derive(Clone, Debug)]
pub struct TransitionSystem {
    initial_state_prop: Vec<usize>,
    initial_state_numeric: Vec<f64>,
    goals: Vec<ExplicitFact>,
    operators: Vec<Operator>,
    non_looping_transitions: u64,
    n_loops: u64,
    incoming_transitions: RefCell<Vec<Transitions>>,
    outgoing_transitions: RefCell<Vec<Transitions>>,
    loops: RefCell<Vec<Loops>>,
    distances: Vec<f64>,
    initial_abstract_state_hash: usize,
    goal_abstract_states_hashes: HashSet<usize>,
    domain_mapping: DomainMapping,
    domain_sizes: Vec<usize>,
    partitions: NumericPartitions,
    numeric_domain_sizes: Vec<usize>,
    applied_refinements: Vec<Refinement>,
}

impl TransitionSystem {
    pub fn trivial_abstraction(
        task: &dyn AbstractNumericTask,
        domain_mapping: DomainMapping,
        domain_sizes: Vec<usize>,
        partitions: NumericPartitions,
        numeric_domain_sizes: Vec<usize>,
    ) -> TransitionSystem {
        let state_packer = make_prop_state_packer(task);
        let axiom_evaluator = AxiomEvaluator::new(task, &state_packer);
        let (_initial_state_buffer, initial_state_numeric) =
            get_initial_state(task, &state_packer, &axiom_evaluator)
                .expect("Error getting the initial state to build the trivial abstraction.");
        let initial_state_prop = task.get_initial_propositional_state_values().clone();
        TransitionSystem {
            initial_state_prop,
            initial_state_numeric,
            goals: get_goals(task),
            operators: task.get_operators().clone(),
            non_looping_transitions: 0,
            n_loops: task.get_operators().len() as u64,
            incoming_transitions: RefCell::new(vec![]),
            outgoing_transitions: RefCell::new(vec![]),
            loops: RefCell::new(vec![
                task.get_operators()
                    .iter()
                    .enumerate()
                    .map(|(i, _o)| i)
                    .collect(),
            ]),
            distances: vec![0.0],
            initial_abstract_state_hash: 0,
            goal_abstract_states_hashes: HashSet::from_iter([0]),
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
            applied_refinements: vec![],
        }
    }

    pub fn non_looping_transitions(&self) -> u64 {
        self.non_looping_transitions
    }

    pub fn n_loops(&self) -> u64 {
        self.n_loops
    }

    pub fn incoming_transitions(&self) -> Ref<Vec<Transitions>> {
        self.incoming_transitions.borrow()
    }

    pub fn outgoing_transitions(&self) -> Ref<Vec<Transitions>> {
        self.outgoing_transitions.borrow()
    }

    pub fn loops(&self) -> Ref<Vec<Loops>> {
        self.loops.borrow()
    }

    pub fn initial_abstract_state_hash(&self) -> usize {
        self.initial_abstract_state_hash
    }

    pub fn goal_abstract_states_hashes(&self) -> &HashSet<usize> {
        &self.goal_abstract_states_hashes
    }

    pub fn domain_mapping(&self) -> &DomainMapping {
        &self.domain_mapping
    }

    pub fn domain_sizes(&self) -> &Vec<usize> {
        &self.domain_sizes
    }

    pub fn partitions(&self) -> &NumericPartitions {
        &self.partitions
    }

    pub fn numeric_domain_sizes(&self) -> &Vec<usize> {
        &self.numeric_domain_sizes
    }

    /// Get the domain size for the accumulated index var, i.e.,
    /// `domain_sizes[accumulated_index_var]` for propositional variables and
    /// `numeric_domain_sizes[accumulated_index_var - domain_sizes.len()]` for
    /// numeric variables.
    pub fn get_domain_size(&self, accumulated_index_var: usize) -> usize {
        if accumulated_index_var > self.domain_sizes.len() {
            self.numeric_domain_sizes[accumulated_index_var - self.domain_sizes.len()]
        } else {
            self.domain_sizes[accumulated_index_var]
        }
    }

    pub fn abstract_prop_value_in_var_for_hash(&self, var: usize, hash: usize) -> usize {
        for (i, refin) in self.applied_refinements.iter().enumerate().rev() {
            if hash >= refin.n_states_before_refinement {
                if refin.var == var && !refin.numeric {
                    return refin.next_value;
                }
                let offset = hash - refin.n_states_before_refinement;
                let mut traversed_offset = self.applied_refinements[i + 1]
                    .n_states_before_refinement
                    - refin.n_states_before_refinement;
                for j in (0..i).rev() {
                    if self.applied_refinements[j].var != refin.var
                        || self.applied_refinements[j].numeric
                    {
                        traversed_offset -= self.applied_refinements[j + 1]
                            .n_states_before_refinement
                            - self.applied_refinements[j].n_states_before_refinement;
                        if offset >= traversed_offset {
                            // The state is enclosed in this refinement.
                            if self.applied_refinements[j].var == var
                                && !self.applied_refinements[j].numeric
                            {
                                return self.applied_refinements[j].next_value;
                            } else {
                                // TODO: This function.
                            }
                        }
                    }
                }
            }
        }

        0
    }

    pub fn abstract_state_hash(&self, prop: &[usize], numeric: &[usize]) -> usize {
        let mut hash = 0;
        let mut var_added = vec![false; prop.len()];
        let mut num_var_added = vec![false; numeric.len()];

        for refin in self.applied_refinements.iter().rev() {
            if refin.numeric {
                if num_var_added[refin.var] {
                    hash -= refin.n_states_before_refinement;
                    num_var_added[refin.var] = true;
                }
                if numeric[refin.var] == refin.next_value {
                    hash += refin.n_states_before_refinement;
                }
            } else {
                if var_added[refin.var] {
                    hash -= refin.n_states_before_refinement;
                }
                if prop[refin.var] == refin.next_value {
                    hash += refin.n_states_before_refinement;
                    var_added[refin.var] = true;
                }
            }
        }

        hash
    }

    pub fn abstract_states_with_abstract_value(
        &self,
        var: usize,
        value: usize,
        numeric: bool,
    ) -> Vec<usize> {
        // Transitions and loops are stored by state.
        let domain_size = if numeric {
            self.numeric_domain_sizes[var]
        } else {
            self.domain_sizes[var]
        };
        let n_matching_states = self.loops.borrow().len() / domain_size;
        let mut matching_states = Vec::with_capacity(n_matching_states);

        let mut prop = vec![0; self.domain_sizes.len()];
        let mut num = vec![0; self.numeric_domain_sizes.len()];
        matching_states.push(self.abstract_state_hash(&prop, &num));
        if numeric {
            num[var] = value;
        } else {
            prop[var] = value;
        }
        for (i, refin) in self.applied_refinements.iter().enumerate() {
            if refin.var != var || refin.numeric != numeric {
                if i == 0 {
                    if refin.numeric {
                        num[refin.var] = refin.next_value;
                    } else {
                        prop[refin.var] = refin.next_value;
                    }
                    matching_states.push(self.abstract_state_hash(&prop, &num));
                } else {
                    for j in 0..i {
                        let prev_refin = &self.applied_refinements[j];
                        if (refin.var != var || refin.numeric != numeric)
                            && (prev_refin.var != refin.var || prev_refin.numeric != refin.numeric)
                        {
                            for v in [prev_refin.value, prev_refin.next_value] {
                                if prev_refin.numeric {
                                    num[prev_refin.var] = v;
                                } else {
                                    prop[prev_refin.var] = v;
                                }
                                matching_states.push(self.abstract_state_hash(&prop, &num));
                            }
                        }
                    }
                }
            }
        }

        matching_states
    }

    fn compute_goal_abstract_states_hashes(&self) -> HashSet<usize> {
        let mut goal_hashes = HashSet::new();

        let mut prop = vec![0; self.domain_sizes.len()];
        let mut num = vec![0; self.numeric_domain_sizes.len()];
        let mut goal_vars = HashSet::new();
        for goal in &self.goals {
            goal_vars.insert(goal.var);
            prop[goal.var] = goal.value;
        }
        goal_hashes.insert(self.abstract_state_hash(&prop, &num));
        for (i, refin) in self.applied_refinements.iter().enumerate() {
            if !goal_vars.contains(&refin.var) {
                if i == 0 {
                    if refin.numeric {
                        num[refin.var] = refin.next_value;
                    } else {
                        prop[refin.var] = refin.next_value;
                    }
                    goal_hashes.insert(self.abstract_state_hash(&prop, &num));
                } else {
                    for j in 0..i {
                        let prev_refin = &self.applied_refinements[j];
                        if !goal_vars.contains(&refin.var)
                            && (prev_refin.var != refin.var || prev_refin.numeric != refin.numeric)
                        {
                            for v in [prev_refin.value, prev_refin.next_value] {
                                if prev_refin.numeric {
                                    num[prev_refin.var] = v;
                                } else {
                                    prop[prev_refin.var] = v;
                                }
                                goal_hashes.insert(self.abstract_state_hash(&prop, &num));
                            }
                        }
                    }
                }
            }
        }

        goal_hashes
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_refine_from_flaw(
        &mut self,
        task: &dyn AbstractNumericTask,
        flaw: &Flaw,
        config: &CegarConfig,
        comparison_var_ids: &HashSet<usize>,
        blacklisted_prop_var_ids: &mut HashSet<usize>,
        blacklisted_numeric_var_ids: &mut HashSet<usize>,
        dependent_numeric_refinement: DependentNumericRefinement,
    ) -> Result<Option<RefinementSummary>> {
        match flaw {
            Flaw::Numeric(nf) => {
                let var_id = nf.numeric_var_id;
                if !Self::can_refine_numeric_variable_with_blacklist(
                    &self.domain_sizes,
                    &self.numeric_domain_sizes,
                    var_id,
                    config.max_abstraction_size,
                    blacklisted_numeric_var_ids,
                ) {
                    return Ok(None);
                }
                if self
                    .partitions
                    .split_at(var_id, nf.value, nf.include_in_lower)
                {
                    if let Some(parts) = self.partitions.partitions(var_id)
                        && let Some(slot) = self.numeric_domain_sizes.get_mut(var_id)
                    {
                        *slot = parts.len();
                    }
                    self.post_refine_abstract_fact(
                        var_id,
                        true,
                        partition_for_value(&self.partitions.partitions(var_id).unwrap(), nf.value)
                            .unwrap(),
                    );
                    let mut refined = RefinementSummary::default();
                    refined.mark_numeric(var_id);
                    return Ok(Some(refined));
                }
                Ok(None)
            }
            Flaw::Propositional(pf) => {
                let var_id = pf.fact.var;
                let value = pf.fact.value;

                // Bounds and conversion checks: these should hold in normal operation;
                // surface violations during debug builds but keep release behavior.
                if var_id >= self.domain_mapping.len() || var_id >= self.domain_sizes.len() {
                    debug_assert!(
                        false,
                        "try_refine_from_flaw: var_id out of bounds: {} (mapping.len={}, domain_sizes.len={})",
                        var_id,
                        self.domain_mapping.len(),
                        self.domain_sizes.len()
                    );
                    return Ok(None);
                }

                let concrete_size = match task.get_variable_domain_size(var_id) {
                    Ok(s) => s,
                    Err(e) => {
                        debug_assert!(
                            false,
                            "try_refine_from_flaw: get_variable_domain_size({}) failed: {}",
                            var_id, e
                        );
                        return Ok(None);
                    }
                };

                if value >= concrete_size {
                    debug_assert!(
                        false,
                        "try_refine_from_flaw: fact value {} out of range (concrete size {}) for var {}",
                        value, concrete_size, var_id
                    );
                    return Ok(None);
                }

                let mut changed = false;

                if comparison_var_ids.contains(&var_id) {
                    if !Self::can_refine_propositional_variable_with_blacklist(
                        &self.domain_sizes,
                        &self.numeric_domain_sizes,
                        var_id,
                        2,
                        config.max_abstraction_size,
                        comparison_var_ids,
                        blacklisted_prop_var_ids,
                    ) {
                        return Ok(None);
                    }
                    // Comparison axiom vars: split into {false/unknown} vs {true} like numeric-fd.
                    if self.domain_sizes[var_id] < 2 {
                        self.domain_sizes[var_id] = 2;
                        changed = true;
                    }
                    // Ensure mapping values are within the new abstract size.
                    if !self.domain_mapping[var_id].is_empty()
                        && self.domain_mapping[var_id][0] != 1
                    {
                        self.domain_mapping[var_id][0] = 1;
                        changed = true;
                    }
                    if self.domain_mapping[var_id].len() >= 2 && self.domain_mapping[var_id][1] != 0
                    {
                        self.domain_mapping[var_id][1] = 0;
                        changed = true;
                    }
                    if self.domain_mapping[var_id].len() >= 3 && self.domain_mapping[var_id][2] != 0
                    {
                        self.domain_mapping[var_id][2] = 0;
                        changed = true;
                    }
                } else {
                    let abs_size = self.domain_sizes[var_id];
                    // If we've already fully refined this variable, nothing to do.
                    if abs_size >= concrete_size {
                        return Ok(None);
                    }
                    // Only refine if the value is still mapped to the default class (0).
                    if self.domain_mapping[var_id].get(value).copied().unwrap_or(0) != 0 {
                        return Ok(None);
                    }
                    if !Self::can_refine_propositional_variable_with_blacklist(
                        &self.domain_sizes,
                        &self.numeric_domain_sizes,
                        var_id,
                        abs_size + 1,
                        config.max_abstraction_size,
                        comparison_var_ids,
                        blacklisted_prop_var_ids,
                    ) {
                        return Ok(None);
                    }

                    self.domain_mapping[var_id][value] = abs_size;
                    self.domain_sizes[var_id] = abs_size + 1;
                    changed = true;
                }

                self.post_refine_abstract_fact(var_id, false, 0);

                // Optional dependent numeric refinements (currently produced only for comparison vars).
                if dependent_numeric_refinement != DependentNumericRefinement::None
                    && !pf.dependent_numeric_flaws.is_empty()
                {
                    let mut any_numeric_changed = false;
                    let mut refined = RefinementSummary::default();
                    if changed {
                        refined.mark_propositional(var_id);
                    }
                    let iter: Box<dyn Iterator<Item = &NumericFlaw>> =
                        match dependent_numeric_refinement {
                            DependentNumericRefinement::None => Box::new(std::iter::empty()),
                            DependentNumericRefinement::All => {
                                Box::new(pf.dependent_numeric_flaws.iter())
                            }
                            DependentNumericRefinement::One => {
                                Box::new(pf.dependent_numeric_flaws.iter())
                            }
                        };

                    for dep in iter {
                        let num_id = dep.numeric_var_id;

                        if !Self::can_refine_numeric_variable_with_blacklist(
                            &self.domain_sizes,
                            &self.numeric_domain_sizes,
                            num_id,
                            config.max_abstraction_size,
                            blacklisted_numeric_var_ids,
                        ) {
                            continue;
                        }

                        if self
                            .partitions
                            .split_at(num_id, dep.value, dep.include_in_lower)
                        {
                            if let Some(parts) = self.partitions.partitions(num_id)
                                && let Some(slot) = self.numeric_domain_sizes.get_mut(num_id)
                            {
                                *slot = parts.len();
                            }
                            self.post_refine_abstract_fact(
                                num_id,
                                true,
                                partition_for_value(
                                    &self.partitions.partitions(num_id).unwrap(),
                                    dep.value,
                                )
                                .unwrap(),
                            );
                            any_numeric_changed = true;
                            refined.mark_numeric(num_id);
                            if dependent_numeric_refinement == DependentNumericRefinement::One {
                                break;
                            }
                        }
                    }
                    return Ok((any_numeric_changed || changed).then_some(refined));
                }

                if changed {
                    let mut refined = RefinementSummary::default();
                    refined.mark_propositional(var_id);
                    Ok(Some(refined))
                } else {
                    Ok(None)
                }
            }
        }
    }

    pub fn can_refine_propositional_variable(
        domain_sizes: &[usize],
        numeric_domain_sizes: &[usize],
        var_id: usize,
        new_domain_size: usize,
        max_abstraction_size: usize,
    ) -> bool {
        let Some(total_size) = compute_abstraction_size_u128(domain_sizes, numeric_domain_sizes)
        else {
            return false;
        };
        let Some(&old_domain_size) = domain_sizes.get(var_id) else {
            return false;
        };
        if old_domain_size == 0 || new_domain_size == 0 {
            return false;
        }
        let reduced = total_size / (old_domain_size as u128);
        reduced
            .checked_mul(new_domain_size as u128)
            .map(|candidate| candidate <= max_abstraction_size as u128)
            .unwrap_or(false)
    }

    pub fn can_refine_numeric_variable(
        domain_sizes: &[usize],
        numeric_domain_sizes: &[usize],
        numeric_var_id: usize,
        max_abstraction_size: usize,
    ) -> bool {
        let Some(total_size) = compute_abstraction_size_u128(domain_sizes, numeric_domain_sizes)
        else {
            return false;
        };
        let Some(&old_partition_count) = numeric_domain_sizes.get(numeric_var_id) else {
            return false;
        };
        if old_partition_count == 0 {
            return false;
        }
        let reduced = total_size / (old_partition_count as u128);
        reduced
            .checked_mul((old_partition_count as u128) + 1)
            .map(|candidate| candidate <= max_abstraction_size as u128)
            .unwrap_or(false)
    }

    pub fn can_refine_propositional_variable_with_blacklist(
        domain_sizes: &[usize],
        numeric_domain_sizes: &[usize],
        var_id: usize,
        new_domain_size: usize,
        max_abstraction_size: usize,
        comparison_var_ids: &HashSet<usize>,
        blacklisted_prop_var_ids: &mut HashSet<usize>,
    ) -> bool {
        if blacklisted_prop_var_ids.contains(&var_id) {
            return false;
        }
        if comparison_var_ids.contains(&var_id)
            && domain_sizes.get(var_id).copied().unwrap_or(0) >= 2
        {
            return true;
        }
        if Self::can_refine_propositional_variable(
            domain_sizes,
            numeric_domain_sizes,
            var_id,
            new_domain_size,
            max_abstraction_size,
        ) {
            true
        } else {
            blacklisted_prop_var_ids.insert(var_id);
            false
        }
    }

    pub fn can_refine_numeric_variable_with_blacklist(
        domain_sizes: &[usize],
        numeric_domain_sizes: &[usize],
        numeric_var_id: usize,
        max_abstraction_size: usize,
        blacklisted_numeric_var_ids: &mut HashSet<usize>,
    ) -> bool {
        if blacklisted_numeric_var_ids.contains(&numeric_var_id) {
            return false;
        }
        if Self::can_refine_numeric_variable(
            domain_sizes,
            numeric_domain_sizes,
            numeric_var_id,
            max_abstraction_size,
        ) {
            true
        } else {
            blacklisted_numeric_var_ids.insert(numeric_var_id);
            false
        }
    }

    fn post_refine_abstract_fact(&mut self, var: usize, numeric: bool, value: usize) {
        // Abstract domains, mappings, and partitions must be currently updated.
        let next_value = if numeric {
            value + 1
        } else {
            self.domain_sizes[var] - 1
        };
        let affected_states = self.abstract_states_with_abstract_value(var, value, numeric);
        // Clone all affected states in order at the end.
        for state in &affected_states {
            self.incoming_transitions
                .borrow_mut()
                .push(Vec::with_capacity(
                    self.incoming_transitions.borrow()[*state].len(),
                ));
            self.outgoing_transitions
                .borrow_mut()
                .push(Vec::with_capacity(
                    self.outgoing_transitions.borrow()[*state].len(),
                ));
            self.loops
                .borrow_mut()
                .push(Vec::with_capacity(self.loops.borrow()[*state].len()));
            // Set the same distance as for the parent, it will be updated during A*.
            self.distances.push(self.distances[*state]);
        }
        self.applied_refinements.push(Refinement {
            var,
            value,
            next_value,
            n_states_before_refinement: self.loops.borrow().len(),
            numeric,
        });

        let new_states = self.abstract_states_with_abstract_value(var, next_value, numeric);
        // Determine if abstract initial state and goals have changed.
        for (old_state, new_state) in affected_states.iter().zip(&new_states) {
            if self.initial_abstract_state_hash == *old_state {
                let (abstract_prop, abstract_num) = abstract_state_values_for_concrete_state(
                    &self.initial_state_prop,
                    &self.initial_state_numeric,
                    &self.domain_mapping,
                    &self.partitions,
                );
                if self.abstract_state_hash(&abstract_prop, &abstract_num) != *old_state {
                    debug_assert!(
                        self.abstract_state_hash(&abstract_prop, &abstract_num) == *new_state
                    );
                    self.initial_abstract_state_hash = *new_state;
                }
            }
            self.rewire(*old_state, *new_state, var, numeric);
            self.incoming_transitions.borrow_mut()[*old_state].shrink_to_fit();
            self.incoming_transitions.borrow_mut()[*new_state].shrink_to_fit();
            self.outgoing_transitions.borrow_mut()[*old_state].shrink_to_fit();
            self.outgoing_transitions.borrow_mut()[*new_state].shrink_to_fit();
            self.loops.borrow_mut()[*old_state].shrink_to_fit();
            self.loops.borrow_mut()[*new_state].shrink_to_fit();
        }
        self.goal_abstract_states_hashes = self.compute_goal_abstract_states_hashes();
    }

    fn add_transition(
        transitions: &mut RefMut<Vec<Transitions>>,
        src: usize,
        op_id: usize,
        target: usize,
    ) {
        transitions[src].push(Transition { op_id, target });
    }

    fn remove_transition(
        transitions: &mut RefMut<Vec<Transitions>>,
        src: usize,
        op_id: usize,
        target: usize,
    ) {
        for i in 0..transitions[src].len() {
            let tr = &transitions[src][i];
            if tr.op_id == op_id && tr.target == target {
                transitions[src].remove(i);
                break;
            }
        }
    }

    fn add_loop(loops: &mut RefMut<Vec<Loops>>, state: usize, op_id: usize) {
        loops[state].push(op_id);
    }

    fn remove_loop(loops: &mut RefMut<Vec<Loops>>, state: usize, op_id: usize) {
        for i in 0..loops[state].len() {
            let tr_op = &loops[state][i];
            if *tr_op == op_id {
                loops[state].remove(i);
                break;
            }
        }
    }

    fn rewire(&mut self, state_hash: usize, next_state_hash: usize, var: usize, numeric: bool) {
        if numeric {
            self.rewire_incoming_numeric(state_hash, next_state_hash, var);
            self.rewire_outgoing_numeric(state_hash, next_state_hash, var);
            self.rewire_loops_numeric(state_hash, next_state_hash, var);
        } else {
            self.rewire_incoming_prop(state_hash, next_state_hash, var);
            self.rewire_outgoing_prop(state_hash, next_state_hash, var);
            self.rewire_loops_prop(state_hash, next_state_hash, var);
        }
    }

    fn rewire_incoming_numeric(&self, state_hash: usize, next_state_hash: usize, var: usize) {
        // for (const Arc &arc : incoming_arcs) {
        //     int op_id = arc.op_id;
        //     NumericOperatorProxy op = numeric_task_proxy.get_operators()[op_id];
        //     AbstractState *u = arc.target;
        //     assert(u != this);
        //     Interval post = get_numeric_post(op, var, u->numeric_domains.get_interval(var));
        //     assert(post.defined());
        //
        //     if (v1->numeric_domains.test(var, post)) {
        //         u->add_arc(op_id, v1);
        //     }
        //     if (v2->numeric_domains.test(var, post)) {
        //         u->add_arc(op_id, v2);
        //     }
        //     assert((v1->numeric_domains.test(var, post) ||
        //             v2->numeric_domains.test(var, post)) &&
        //            "split_incoming_arcs_numeric: operator yields no arc after split");
        //     u->remove_outgoing_arc(op_id, this);
        // }
    }
    fn rewire_incoming_prop(&mut self, state_hash: usize, next_state_hash: usize, var: usize) {
        let mut outgoing = &mut self.outgoing_transitions.borrow_mut();
        let n_trs = &self.incoming_transitions.borrow()[state_hash].len();
        for i in 0..*n_trs {
            let tr = &self.incoming_transitions.borrow()[state_hash][i];
            let op = &self.operators[tr.op_id];
            let u = tr.target;
            let post = get_post(op, var);
            if post.is_none() {
                // `op` has no precondition and no effect on `var`.
                let abstract_value_for_u = self.abstract_prop_value_in_var_for_hash(var, u);
                let u_and_v1_intersect = abstract_value_for_u
                    == self.abstract_prop_value_in_var_for_hash(var, state_hash);
                if u_and_v1_intersect {
                    Self::add_transition(&mut outgoing, u, tr.op_id, state_hash);
                }
                /* If the domains of u and v1 don't intersect, we must add
                the other arc and can avoid an intersection test. */
                if !u_and_v1_intersect
                    || abstract_value_for_u
                        == self.abstract_prop_value_in_var_for_hash(var, next_state_hash)
                {
                    Self::add_transition(&mut outgoing, u, tr.op_id, next_state_hash);
                }
            } else if self.domain_mapping[var][post.unwrap()]
                == self.abstract_prop_value_in_var_for_hash(var, state_hash)
            {
                // `op` can only end in `state_hash`.
                Self::add_transition(&mut outgoing, u, tr.op_id, state_hash);
            } else {
                // `op` can only end in `next_state_hash`.
                Self::add_transition(&mut outgoing, u, tr.op_id, next_state_hash);
            }

            Self::remove_transition(&mut outgoing, u, tr.op_id, state_hash);
        }
    }

    fn rewire_outgoing_numeric(&mut self, state_hash: usize, next_state_hash: usize, var: usize) {}
    fn rewire_outgoing_prop(&mut self, state_hash: usize, next_state_hash: usize, var: usize) {}

    fn rewire_loops_numeric(&mut self, state_hash: usize, next_state_hash: usize, var: usize) {}
    fn rewire_loops_prop(&mut self, state_hash: usize, next_state_hash: usize, var: usize) {}
}
