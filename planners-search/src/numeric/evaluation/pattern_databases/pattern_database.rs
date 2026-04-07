#[cfg(test)]
mod tests;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use ordered_float::NotNan;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, ExplicitFact, Operator,
};
use planners_sas::numeric::utils::int_packer::IntDoublePacker;

use crate::numeric::successor_generator::{ApplicableOperator, GroundedSuccessorGenerator};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct AbstractStateKey {
    propositional: Vec<usize>,
    numeric: Vec<u64>,
}

#[derive(Debug, Clone)]
pub(super) struct PdbState {
    pub(super) propositional: Vec<usize>,
    pub(super) numeric: Vec<f64>,
}

impl PartialEq for PdbState {
    fn eq(&self, other: &Self) -> bool {
        self.propositional == other.propositional
            && self.numeric.len() == other.numeric.len()
            && self
                .numeric
                .iter()
                .zip(other.numeric.iter())
                .all(|(lhs, rhs)| lhs.to_bits() == rhs.to_bits())
    }
}

impl Eq for PdbState {}

impl Hash for PdbState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.propositional.hash(state);
        for value in &self.numeric {
            value.to_bits().hash(state);
        }
    }
}

pub struct PatternDatabase<T: AbstractNumericTask> {
    pub(super) task: T,
    pub(super) state_to_id: HashMap<AbstractStateKey, usize>,
    pub(super) states: Vec<PdbState>,
    pub(super) distances: Vec<f64>,
    pub(super) min_operator_cost: f64,
    pub(super) reached_goal_states: usize,
    pub(super) truncated: bool,
    pub(super) frontier_states: Vec<usize>,
}

impl<T: AbstractNumericTask> PatternDatabase<T> {
    pub fn new(task: T, max_states: usize) -> Result<Self, String> {
        let min_operator_cost = task.min_abstract_operator_cost();

        let mut pdb = Self {
            task,
            state_to_id: HashMap::new(),
            states: Vec::new(),
            distances: Vec::new(),
            min_operator_cost,
            reached_goal_states: 0,
            truncated: false,
            frontier_states: Vec::new(),
        };
        pdb.build(max_states)?;
        //super::utils::dump_distance_table(&pdb);
        Ok(pdb)
    }

    pub fn lookup(&self, propositional: &[usize], numeric: &[f64]) -> Option<f64> {
        let key = make_abstract_state_key_from_values(&self.task, propositional, numeric)?;
        self.state_to_id
            .get(&key)
            .and_then(|&state_id| self.distances.get(state_id))
            .copied()
    }

    pub fn lookup_or_fallback(&self, propositional: &[usize], numeric: &[f64]) -> f64 {
        match self.lookup(propositional, numeric) {
            Some(distance) if distance.is_finite() => distance,
            Some(_) if self.is_goal_state(propositional) => 0.0,
            Some(_) if self.truncated => self.min_operator_cost(),
            Some(distance) => distance,
            None => {
                if self.is_goal_state(propositional) {
                    0.0
                } else {
                    self.min_operator_cost()
                }
            }
        }
    }

    pub fn is_goal_state(&self, propositional: &[usize]) -> bool {
        (0..self.task.get_num_goals().max(0)).all(|goal_index| {
            let goal = self.task.get_goal_fact(goal_index);
            propositional.get(goal.var).copied() == Some(goal.value)
        })
    }

    pub fn min_operator_cost(&self) -> f64 {
        self.min_operator_cost
    }

    pub fn abstract_state_values(
        &self,
        propositional: &[usize],
        numeric: &[f64],
    ) -> Result<(Vec<usize>, Vec<f64>), String> {
        let (abstract_prop, abstract_num) =
            self.task.abstract_state_values(propositional, numeric)?;
        Ok((
            self.task
                .abstract_propositional_var_ids()
                .iter()
                .map(|&var_id| abstract_prop[var_id])
                .collect(),
            self.task
                .abstract_numeric_var_ids()
                .iter()
                .map(|&var_id| abstract_num[var_id])
                .collect(),
        ))
    }

    fn build(&mut self, max_states: usize) -> Result<(), String> {
        let mut predecessors: Vec<Vec<(usize, f64)>> = Vec::new();
        let mut frontier_seed_costs: HashMap<usize, f64> = HashMap::new();
        let packer = propositional_packer(&self.task);
        let axiom_evaluator = AxiomEvaluator::new(&self.task, &packer);
        let successor_generator = GroundedSuccessorGenerator::construct_node_from_task(&self.task);
        let mut applicable_operators: Vec<ApplicableOperator<'_>> = Vec::new();
        let (initial_propositional, initial_numeric) =
            self.task.evaluated_initial_abstract_state_values()?;
        let initial_state = PdbState {
            propositional: initial_propositional,
            numeric: initial_numeric,
        };
        self.state_to_id
            .insert(make_abstract_state_key(&self.task, &initial_state), 0);
        self.states.push(initial_state);
        predecessors.push(Vec::new());

        let mut queue = VecDeque::from([0usize]);
        while let Some(state_id) = queue.pop_front() {
            if self.states.len() >= max_states {
                self.truncated = true;
                frontier_seed_costs
                    .entry(state_id)
                    .and_modify(|cost| *cost = cost.min(self.min_operator_cost))
                    .or_insert(self.min_operator_cost);
                for queued_state_id in queue.iter().copied() {
                    frontier_seed_costs
                        .entry(queued_state_id)
                        .and_modify(|cost| *cost = cost.min(self.min_operator_cost))
                        .or_insert(self.min_operator_cost);
                }
                break;
            }
            let state = self.states[state_id].clone();

            applicable_operators.clear();
            successor_generator
                .get_applicable_operators(&state.propositional, &mut applicable_operators);

            for (operator, operator_id) in applicable_operators.iter().copied() {
                let operator_cost = self.task.abstract_operator_cost(operator_id);
                let successor = apply_operator(&packer, &axiom_evaluator, &state, operator)?;
                if successor == state {
                    continue;
                }

                let successor_key = make_abstract_state_key(&self.task, &successor);
                let next_id =
                    if let Some(existing_id) = self.state_to_id.get(&successor_key).copied() {
                        existing_id
                    } else {
                        if self.states.len() >= max_states {
                            self.truncated = true;
                            frontier_seed_costs
                                .entry(state_id)
                                .and_modify(|cost| *cost = cost.min(operator_cost))
                                .or_insert(operator_cost);
                            continue;
                        }
                        let new_id = self.states.len();
                        self.state_to_id.insert(successor_key, new_id);
                        self.states.push(successor);
                        predecessors.push(Vec::new());
                        queue.push_back(new_id);
                        new_id
                    };

                predecessors[next_id].push((state_id, operator_cost));
            }
        }

        self.distances = vec![f64::INFINITY; self.states.len()];
        let mut heap: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();

        self.reached_goal_states = 0;
        for (state_id, state) in self.states.iter().enumerate() {
            if self.is_goal_state(&state.propositional) {
                self.reached_goal_states += 1;
                self.distances[state_id] = 0.0;
                heap.push((Reverse(NotNan::new(0.0).unwrap()), state_id));
            }
        }

        if self.truncated {
            let mut frontier_states: Vec<usize> = frontier_seed_costs.keys().copied().collect();
            frontier_states.sort_unstable();
            frontier_states.dedup();
            self.frontier_states = frontier_states;

            for (&state_id, &seed_cost) in &frontier_seed_costs {
                if seed_cost + 1e-12 < self.distances[state_id] {
                    self.distances[state_id] = seed_cost;
                    heap.push((Reverse(NotNan::new(seed_cost).unwrap()), state_id));
                }
            }
        } else {
            self.frontier_states.clear();
        }

        while let Some((Reverse(distance), state_id)) = heap.pop() {
            let distance = distance.into_inner();
            if distance > self.distances[state_id] + 1e-12 {
                continue;
            }

            for &(parent_id, operator_cost) in &predecessors[state_id] {
                let alternative = distance + operator_cost;
                if alternative + 1e-12 < self.distances[parent_id] {
                    self.distances[parent_id] = alternative;
                    heap.push((Reverse(NotNan::new(alternative).unwrap()), parent_id));
                }
            }
        }

        Ok(())
    }
}

fn make_abstract_state_key<T: AbstractNumericTask>(task: &T, state: &PdbState) -> AbstractStateKey {
    AbstractStateKey {
        propositional: task
            .abstract_propositional_var_ids()
            .iter()
            .map(|&var_id| state.propositional[var_id])
            .collect(),
        numeric: task
            .abstract_numeric_var_ids()
            .iter()
            .map(|&var_id| state.numeric[var_id].to_bits())
            .collect(),
    }
}

fn make_abstract_state_key_from_values(
    task: &impl AbstractNumericTask,
    propositional: &[usize],
    numeric: &[f64],
) -> Option<AbstractStateKey> {
    let propositional_values = if propositional.len() == task.variables().len() {
        task.abstract_propositional_var_ids()
            .iter()
            .map(|&var_id| propositional.get(var_id).copied())
            .collect::<Option<Vec<_>>>()?
    } else {
        propositional.to_vec()
    };

    let numeric_values = if numeric.len() == task.numeric_variables().len() {
        task.abstract_numeric_var_ids()
            .iter()
            .map(|&var_id| numeric.get(var_id).copied().map(f64::to_bits))
            .collect::<Option<Vec<_>>>()?
    } else {
        numeric.iter().map(|value| value.to_bits()).collect()
    };

    Some(AbstractStateKey {
        propositional: propositional_values,
        numeric: numeric_values,
    })
}

fn facts_hold(propositional: &[usize], facts: &[ExplicitFact]) -> bool {
    facts
        .iter()
        .all(|fact| propositional.get(fact.var).copied() == Some(fact.value))
}

fn assignment_effect_holds(propositional: &[usize], effect: &AssignmentEffect) -> bool {
    !effect.is_conditional() || facts_hold(propositional, effect.conditions())
}

fn apply_operator(
    packer: &IntDoublePacker,
    axiom_evaluator: &AxiomEvaluator<'_>,
    state: &PdbState,
    operator: &Operator,
) -> Result<PdbState, String> {
    let mut propositional = state.propositional.clone();
    let mut numeric = state.numeric.clone();

    for effect in operator.effects() {
        if facts_hold(&state.propositional, effect.conditions())
            && let Some(slot) = propositional.get_mut(effect.var_id())
        {
            *slot = effect.value();
        }
    }

    for effect in operator.assignment_effects() {
        if !assignment_effect_holds(&state.propositional, effect) {
            continue;
        }
        let source = numeric
            .get(effect.var_id())
            .copied()
            .ok_or_else(|| format!("assignment source out of bounds: {}", effect.var_id()))?;
        let target = numeric
            .get(effect.affected_var_id())
            .copied()
            .ok_or_else(|| {
                format!(
                    "assignment target out of bounds: {}",
                    effect.affected_var_id()
                )
            })?;
        let result = AssignmentOperation::apply(target, effect.operation(), source);
        if let Some(slot) = numeric.get_mut(effect.affected_var_id()) {
            *slot = result;
        }
    }

    let mut buffer = vec![0u64; packer.num_bins()];
    for (var_id, value) in propositional.iter().enumerate() {
        packer.set(&mut buffer, var_id, *value as u64);
    }

    axiom_evaluator
        .evaluate_arithmetic_axioms(&mut numeric)
        .map_err(|err| format!("failed to evaluate arithmetic axioms: {err:?}"))?;
    axiom_evaluator
        .evaluate(&mut buffer, &mut numeric)
        .map_err(|err| format!("failed to evaluate axioms: {err:?}"))?;

    for (var_id, slot) in propositional.iter_mut().enumerate() {
        *slot = packer.get(&buffer, var_id) as usize;
    }

    Ok(PdbState {
        propositional,
        numeric,
    })
}

fn propositional_packer(task: &dyn AbstractNumericTask) -> IntDoublePacker {
    let ranges: Vec<u64> = task
        .variables()
        .iter()
        .map(|variable| variable.domain_size() as u64)
        .collect();
    IntDoublePacker::new(&ranges)
}
