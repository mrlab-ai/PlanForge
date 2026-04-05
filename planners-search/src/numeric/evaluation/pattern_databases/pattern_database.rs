use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};

use ordered_float::NotNan;
use planners_sas::numeric::axioms::AxiomEvaluator;
use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, ExplicitFact, Operator,
};
use planners_sas::numeric::utils::int_packer::IntDoublePacker;

use crate::numeric::successor_generator::{ApplicableOperator, GroundedSuccessorGenerator, Node};

use super::projected_task::ProjectedTask;

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

pub struct PatternDatabase<'task> {
    pub(super) task: ProjectedTask<'task>,
    pub(super) state_to_id: HashMap<PdbState, usize>,
    pub(super) states: Vec<PdbState>,
    pub(super) distances: Vec<f64>,
    pub(super) min_operator_cost: f64,
}

impl<'task> PatternDatabase<'task> {
    pub fn new(projected_task: ProjectedTask<'task>, max_states: usize) -> Result<Self, String> {
        let min_operator_cost = projected_task
            .get_operators()
            .iter()
            .map(|operator| operator.cost() as f64)
            .fold(f64::INFINITY, f64::min);
        let min_operator_cost = if min_operator_cost.is_finite() {
            min_operator_cost.max(0.0)
        } else {
            0.0
        };

        let mut pdb = Self {
            task: projected_task,
            state_to_id: HashMap::new(),
            states: Vec::new(),
            distances: Vec::new(),
            min_operator_cost,
        };
        pdb.build(max_states)?;
        super::utils::dump_distance_table(&pdb);
        Ok(pdb)
    }

    pub fn lookup(&self, propositional: &[usize], numeric: &[f64]) -> Option<f64> {
        let key = PdbState {
            propositional: propositional.to_vec(),
            numeric: numeric.to_vec(),
        };
        self.state_to_id
            .get(&key)
            .and_then(|&state_id| self.distances.get(state_id))
            .copied()
    }

    pub fn lookup_or_fallback(&self, propositional: &[usize], numeric: &[f64]) -> f64 {
        self.lookup(propositional, numeric).unwrap_or_else(|| {
            if self.is_goal_state(propositional) {
                0.0
            } else {
                self.min_operator_cost()
            }
        })
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

    pub fn project_state_values(
        &self,
        propositional: &[usize],
        numeric: &[f64],
    ) -> Result<(Vec<usize>, Vec<f64>), String> {
        self.task.project_state_values(propositional, numeric)
    }

    fn build(&mut self, max_states: usize) -> Result<(), String> {
        let mut predecessors: Vec<Vec<(usize, f64)>> = Vec::new();
        let packer = propositional_packer(&self.task);
        let axiom_evaluator = AxiomEvaluator::new(&self.task, &packer);
        let successor_generator = GroundedSuccessorGenerator::construct_node_from_task(&self.task);
        let mut applicable_operators: Vec<ApplicableOperator<'_>> = Vec::new();
        let initial_state = PdbState {
            propositional: self.task.get_initial_propositional_state_values().to_vec(),
            numeric: self.task.get_initial_numeric_state_values().to_vec(),
        };
        self.state_to_id.insert(initial_state.clone(), 0);
        self.states.push(initial_state);
        predecessors.push(Vec::new());

        let mut queue = VecDeque::from([0usize]);
        while let Some(state_id) = queue.pop_front() {
            if self.states.len() >= max_states {
                break;
            }
            let state = self.states[state_id].clone();

            applicable_operators.clear();
            successor_generator
                .get_applicable_operators(&state.propositional, &mut applicable_operators);

            for (operator, _) in applicable_operators.iter().copied() {
                let successor =
                    apply_operator(&self.task, &packer, &axiom_evaluator, &state, operator)?;
                if successor == state {
                    continue;
                }

                let next_id = if let Some(existing_id) = self.state_to_id.get(&successor).copied() {
                    existing_id
                } else {
                    if self.states.len() >= max_states {
                        continue;
                    }
                    let new_id = self.states.len();
                    self.state_to_id.insert(successor.clone(), new_id);
                    self.states.push(successor);
                    predecessors.push(Vec::new());
                    queue.push_back(new_id);
                    new_id
                };

                predecessors[next_id].push((state_id, operator.cost() as f64));
            }
        }

        self.distances = vec![f64::INFINITY; self.states.len()];
        let mut heap: BinaryHeap<(Reverse<NotNan<f64>>, usize)> = BinaryHeap::new();

        for (state_id, state) in self.states.iter().enumerate() {
            if self.is_goal_state(&state.propositional) {
                self.distances[state_id] = 0.0;
                heap.push((Reverse(NotNan::new(0.0).unwrap()), state_id));
            }
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

fn facts_hold(propositional: &[usize], facts: &[ExplicitFact]) -> bool {
    facts
        .iter()
        .all(|fact| propositional.get(fact.var).copied() == Some(fact.value))
}

fn assignment_effect_holds(propositional: &[usize], effect: &AssignmentEffect) -> bool {
    !effect.is_conditional() || facts_hold(propositional, effect.conditions())
}

fn apply_operator(
    task: &dyn AbstractNumericTask,
    packer: &IntDoublePacker,
    axiom_evaluator: &AxiomEvaluator<'_>,
    state: &PdbState,
    operator: &Operator,
) -> Result<PdbState, String> {
    let mut propositional = state.propositional.clone();
    let mut numeric = state.numeric.clone();

    for effect in operator.effects() {
        if facts_hold(&state.propositional, effect.conditions()) {
            if let Some(slot) = propositional.get_mut(effect.var_id()) {
                *slot = effect.value();
            }
        }
    }

    for effect in operator.assignment_effects() {
        if !assignment_effect_holds(&state.propositional, effect) {
            continue;
        }
        let source = numeric
            .get(effect.var_id() as usize)
            .copied()
            .ok_or_else(|| format!("assignment source out of bounds: {}", effect.var_id()))?;
        let target = numeric
            .get(effect.affected_var_id() as usize)
            .copied()
            .ok_or_else(|| {
                format!(
                    "assignment target out of bounds: {}",
                    effect.affected_var_id()
                )
            })?;
        let result = AssignmentOperation::apply(target, effect.operation(), source);
        if let Some(slot) = numeric.get_mut(effect.affected_var_id() as usize) {
            *slot = result;
        }
    }

    let mut buffer = vec![0u64; packer.num_bins() as usize];
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

#[cfg(test)]
mod tests {
    use planners_sas::numeric::axioms::AssignmentAxiom;
    use planners_sas::numeric::axioms::CalOperator;
    use planners_sas::numeric::numeric_task::{
        ExplicitVariable, Metric, NumericRootTask, NumericType, NumericVariable,
    };

    use super::*;
    use crate::numeric::evaluation::pattern_databases::projected_task::{Pattern, ProjectedTask};

    fn simple_var(name: &str, axiom_layer: Option<usize>) -> ExplicitVariable {
        ExplicitVariable::new(
            2,
            name.to_string(),
            vec![format!("{name}=0"), format!("{name}=1")],
            axiom_layer,
            1,
        )
    }

    fn propositional_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, None),
            vec![simple_var("p", None)],
            vec![NumericVariable::new(
                "x".to_string(),
                NumericType::Regular,
                None,
            )],
            vec![ExplicitFact::new(0, 1)],
            vec![],
            vec![0],
            vec![0.0],
            vec![Operator::new(
                "set-goal".to_string(),
                vec![ExplicitFact::new(0, 0)],
                vec![planners_sas::numeric::numeric_task::Effect::new(
                    vec![],
                    0,
                    Some(0),
                    1,
                )],
                vec![],
                3,
            )],
            vec![],
            vec![],
            vec![AssignmentAxiom::new(0, CalOperator::Sum, 0, 0)],
            ExplicitFact::new(0, 0),
        )
    }

    #[test]
    fn lookup_returns_distance_for_reached_state() {
        let task = propositional_task();
        let projected_task = ProjectedTask::new(
            &task,
            &Pattern {
                regular: vec![0],
                numeric: vec![0],
            },
        )
        .unwrap();
        let pdb = PatternDatabase::new(projected_task, 32).unwrap();

        assert_eq!(pdb.lookup(&[0], &[0.0]), Some(3.0));
        assert_eq!(pdb.lookup(&[1], &[0.0]), Some(0.0));
    }

    #[test]
    fn lookup_miss_returns_zero_for_goal_state() {
        let task = propositional_task();
        let projected_task = ProjectedTask::new(
            &task,
            &Pattern {
                regular: vec![0],
                numeric: vec![0],
            },
        )
        .unwrap();
        let pdb = PatternDatabase::new(projected_task, 1).unwrap();

        assert_eq!(pdb.lookup(&[1], &[0.0]), None);
        assert_eq!(pdb.lookup_or_fallback(&[1], &[0.0]), 0.0);
    }

    #[test]
    fn lookup_miss_returns_min_operator_cost_for_non_goal_state() {
        let task = propositional_task();
        let projected_task = ProjectedTask::new(
            &task,
            &Pattern {
                regular: vec![0],
                numeric: vec![0],
            },
        )
        .unwrap();
        let pdb = PatternDatabase::new(projected_task, 1).unwrap();

        assert_eq!(pdb.lookup(&[0], &[42.0]), None);
        assert_eq!(pdb.lookup_or_fallback(&[0], &[42.0]), 3.0);
    }
}
