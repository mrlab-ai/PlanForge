#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ops::Deref;

use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, NumericType,
};

use crate::task_restriction::validate_restricted_task;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CausalGraphVariable {
    Propositional(usize),
    Numeric(usize),
}

#[derive(Debug, Default, Clone)]
pub struct CausalGraph {
    eff_predecessors: BTreeMap<CausalGraphVariable, BTreeSet<CausalGraphVariable>>,
    predecessors: BTreeMap<CausalGraphVariable, BTreeSet<CausalGraphVariable>>,
    successors: BTreeMap<CausalGraphVariable, BTreeSet<CausalGraphVariable>>,
    goal_distances: BTreeMap<CausalGraphVariable, usize>,
    causal_levels: BTreeMap<CausalGraphVariable, usize>,
    comparison_numeric_vars: Vec<Option<usize>>,
}

impl CausalGraph {
    pub fn predecessors_of(
        &self,
        variable: CausalGraphVariable,
    ) -> impl Iterator<Item = CausalGraphVariable> + '_ {
        self.predecessors
            .get(&variable)
            .into_iter()
            .flat_map(|predecessors| predecessors.iter().copied())
    }

    pub fn successors_of(
        &self,
        variable: CausalGraphVariable,
    ) -> impl Iterator<Item = CausalGraphVariable> + '_ {
        self.successors
            .get(&variable)
            .into_iter()
            .flat_map(|successors| successors.iter().copied())
    }

    pub fn eff_pre_neighbors_of(
        &self,
        variable: CausalGraphVariable,
    ) -> impl Iterator<Item = CausalGraphVariable> + '_ {
        self.eff_predecessors
            .get(&variable)
            .into_iter()
            .flat_map(|predecessors| predecessors.iter().copied())
    }

    pub fn goal_distance(&self, variable: CausalGraphVariable) -> Option<usize> {
        self.goal_distances.get(&variable).copied()
    }

    pub fn causal_level(&self, variable: CausalGraphVariable) -> Option<usize> {
        self.causal_levels.get(&variable).copied()
    }

    pub fn predecessor_count(&self, variable: CausalGraphVariable) -> usize {
        self.predecessors
            .get(&variable)
            .map(BTreeSet::len)
            .unwrap_or(0)
    }

    pub fn comparison_numeric_var(&self, comparison_axiom_id: usize) -> Option<usize> {
        self.comparison_numeric_vars
            .get(comparison_axiom_id)
            .copied()
            .flatten()
    }

    fn build(task: &dyn AbstractNumericTask, support: &NumericGraphSupport) -> Self {
        let comparison_numeric_vars = (0..task.comparison_axioms().len())
            .map(|comparison_axiom_id| support.comparison_numeric_var(task, comparison_axiom_id))
            .collect();
        let mut graph = Self {
            comparison_numeric_vars,
            ..Self::default()
        };

        for var_id in 0..task.variables().len() {
            if is_regular_propositional_var(task, var_id) {
                graph.ensure_node(CausalGraphVariable::Propositional(var_id));
            }
        }
        for numeric_var_id in support.numeric_nodes(task) {
            graph.ensure_node(CausalGraphVariable::Numeric(numeric_var_id));
        }

        for operator in task.get_operators() {
            let precondition_sources = graph.precondition_sources(task, operator.preconditions());
            let propositional_effect_targets: Vec<_> = operator
                .effects()
                .iter()
                .filter_map(|effect| {
                    is_regular_propositional_var(task, effect.var_id())
                        .then_some(CausalGraphVariable::Propositional(effect.var_id()))
                })
                .collect();
            let numeric_effect_targets: Vec<_> = operator
                .assignment_effects()
                .iter()
                .filter_map(|effect| numeric_effect_target(task, effect))
                .collect();
            let effect_targets: Vec<_> = propositional_effect_targets
                .iter()
                .copied()
                .chain(numeric_effect_targets.iter().copied())
                .collect();

            for &target in &effect_targets {
                for &source in &precondition_sources {
                    graph.add_pre_eff_arc(source, target);
                }
            }
            for effect in operator.effects() {
                if !is_regular_propositional_var(task, effect.var_id()) {
                    continue;
                }
                let target = CausalGraphVariable::Propositional(effect.var_id());
                for source in graph.precondition_sources(task, effect.conditions()) {
                    graph.add_pre_eff_arc(source, target);
                }
            }
            for effect in operator.assignment_effects() {
                if let Some(target) = numeric_effect_target(task, effect) {
                    for source in graph.precondition_sources(task, effect.conditions()) {
                        graph.add_pre_eff_arc(source, target);
                    }
                    if let Some(source) = support.numeric_effect_source(task, effect) {
                        graph.add_pre_eff_arc(source, target);
                    }
                }
            }
            for effect_index in 0..effect_targets.len() {
                for other_index in (effect_index + 1)..effect_targets.len() {
                    graph.add_eff_eff_edge(
                        effect_targets[effect_index],
                        effect_targets[other_index],
                    );
                }
            }
        }

        graph.compute_goal_distances(task);
        graph.compute_causal_levels();
        graph
    }

    fn precondition_sources(
        &self,
        task: &dyn AbstractNumericTask,
        preconditions: &[planforge_sas::numeric_task::ExplicitFact],
    ) -> Vec<CausalGraphVariable> {
        let mut sources = BTreeSet::new();
        for fact in preconditions {
            if let Some(comparison_axiom_id) = comparison_axiom_id_for_var(task, fact.var()) {
                if let Some(numeric_var_id) = self.comparison_numeric_var(comparison_axiom_id) {
                    sources.insert(CausalGraphVariable::Numeric(numeric_var_id));
                }
            } else if is_regular_propositional_var(task, fact.var()) {
                sources.insert(CausalGraphVariable::Propositional(fact.var()));
            }
        }
        sources.into_iter().collect()
    }

    fn ensure_node(&mut self, variable: CausalGraphVariable) {
        self.eff_predecessors.entry(variable).or_default();
        self.predecessors.entry(variable).or_default();
        self.successors.entry(variable).or_default();
    }

    fn add_pre_eff_arc(&mut self, source: CausalGraphVariable, target: CausalGraphVariable) {
        self.ensure_node(source);
        self.ensure_node(target);
        if source == target {
            return;
        }
        self.eff_predecessors
            .entry(target)
            .or_default()
            .insert(source);
        self.successors.entry(source).or_default().insert(target);
        self.predecessors.entry(target).or_default().insert(source);
    }

    fn add_eff_eff_edge(&mut self, lhs: CausalGraphVariable, rhs: CausalGraphVariable) {
        self.ensure_node(lhs);
        self.ensure_node(rhs);
        if lhs == rhs {
            return;
        }
        self.successors.entry(lhs).or_default().insert(rhs);
        self.successors.entry(rhs).or_default().insert(lhs);
        self.predecessors.entry(lhs).or_default().insert(rhs);
        self.predecessors.entry(rhs).or_default().insert(lhs);
    }

    fn compute_goal_distances(&mut self, task: &dyn AbstractNumericTask) {
        let mut queue = VecDeque::new();
        for goal_index in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(goal_index);
            let goal_var = comparison_axiom_id_for_var(task, goal.var())
                .and_then(|id| self.comparison_numeric_var(id))
                .map(CausalGraphVariable::Numeric)
                .or_else(|| {
                    is_regular_propositional_var(task, goal.var())
                        .then_some(CausalGraphVariable::Propositional(goal.var()))
                });
            if let Some(goal_var) = goal_var
                && self.goal_distances.insert(goal_var, 0).is_none()
            {
                queue.push_back(goal_var);
            }
        }

        while let Some(variable) = queue.pop_front() {
            let distance = self.goal_distances[&variable];
            let predecessors: Vec<_> = self.predecessors_of(variable).collect();
            for predecessor in predecessors {
                if self.goal_distances.contains_key(&predecessor) {
                    continue;
                }
                self.goal_distances.insert(predecessor, distance + 1);
                queue.push_back(predecessor);
            }
        }
    }

    fn compute_causal_levels(&mut self) {
        let mut queue = VecDeque::new();
        for (&variable, predecessors) in &self.predecessors {
            if predecessors.is_empty() {
                self.causal_levels.insert(variable, 0);
                queue.push_back(variable);
            }
        }

        while let Some(variable) = queue.pop_front() {
            let level = self.causal_levels[&variable];
            let successors: Vec<_> = self.successors_of(variable).collect();
            for successor in successors {
                let next_level = level + 1;
                let should_enqueue = match self.causal_levels.get(&successor).copied() {
                    Some(existing_level) if existing_level <= next_level => false,
                    Some(_) => {
                        self.causal_levels.insert(successor, next_level);
                        true
                    }
                    None => {
                        self.causal_levels.insert(successor, next_level);
                        true
                    }
                };
                if should_enqueue {
                    queue.push_back(successor);
                }
            }
        }

        for &variable in self.predecessors.keys() {
            self.causal_levels.entry(variable).or_insert(usize::MAX / 2);
        }
    }
}

#[derive(Debug, Clone)]
pub struct RestrictedCausalGraph(CausalGraph);

impl RestrictedCausalGraph {
    pub fn new(task: &dyn AbstractNumericTask) -> Result<Self, String> {
        validate_restricted_task(task)?;
        Ok(Self(CausalGraph::build(
            task,
            &NumericGraphSupport::Restricted,
        )))
    }
}

impl Deref for RestrictedCausalGraph {
    type Target = CausalGraph;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct SnpCausalGraph(CausalGraph);

impl SnpCausalGraph {
    pub fn new(task: &dyn AbstractNumericTask) -> Result<Self, String> {
        let support = SnpNumericSupport::new(task)?;
        Ok(Self(CausalGraph::build(
            task,
            &NumericGraphSupport::Snp(support),
        )))
    }
}

impl Deref for SnpCausalGraph {
    type Target = CausalGraph;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

enum NumericGraphSupport {
    Restricted,
    Snp(SnpNumericSupport),
}

impl NumericGraphSupport {
    fn numeric_nodes(&self, task: &dyn AbstractNumericTask) -> Vec<usize> {
        let mut nodes: Vec<_> = task
            .numeric_variables()
            .iter()
            .enumerate()
            .filter_map(|(id, variable)| {
                (variable.get_type() == &NumericType::Regular).then_some(id)
            })
            .collect();
        if let Self::Snp(support) = self {
            nodes.extend(support.helper_ids());
        }
        nodes
    }

    fn comparison_numeric_var(
        &self,
        task: &dyn AbstractNumericTask,
        comparison_axiom_id: usize,
    ) -> Option<usize> {
        let comparison_axiom = task.comparison_axioms().get(comparison_axiom_id)?;
        let left = comparison_axiom.get_left_var_id();
        let right = comparison_axiom.get_right_var_id();
        match self {
            Self::Restricted => preferred_nonconstant_side(task, left, right)
                .and_then(|id| restricted_numeric_var(task, id)),
            Self::Snp(support) => {
                preferred_nonconstant_side_with(left, right, |id| support.is_nonconstant(task, id))
                    .and_then(|id| support.representative(task, id))
            }
        }
    }

    fn numeric_effect_source(
        &self,
        task: &dyn AbstractNumericTask,
        effect: &AssignmentEffect,
    ) -> Option<CausalGraphVariable> {
        let numeric_var_id = effect.var_id();
        match self {
            Self::Restricted => restricted_numeric_var(task, numeric_var_id),
            Self::Snp(support) => support.representative(task, numeric_var_id),
        }
        .map(CausalGraphVariable::Numeric)
    }
}

#[derive(Debug)]
struct SnpNumericSupport {
    assignment_by_affected: Vec<Option<usize>>,
    helper_by_derived: Vec<Option<usize>>,
    regular_dependencies: Vec<Option<BTreeSet<usize>>>,
}

impl SnpNumericSupport {
    fn new(task: &dyn AbstractNumericTask) -> Result<Self, String> {
        let mut assignment_by_affected = vec![None; task.numeric_variables().len()];
        for (axiom_id, axiom) in task.assignment_axioms().iter().enumerate() {
            let affected = axiom.get_affected_var_id();
            let slot = assignment_by_affected.get_mut(affected).ok_or_else(|| {
                format!("assignment axiom {axiom_id} has invalid target {affected}")
            })?;
            if slot.replace(axiom_id).is_some() {
                return Err(format!(
                    "multiple assignment axioms define numeric variable {affected}"
                ));
            }
        }

        let num_numeric = task.numeric_variables().len();
        let mut support = Self {
            assignment_by_affected,
            helper_by_derived: vec![None; num_numeric],
            regular_dependencies: vec![None; num_numeric],
        };
        let mut visiting = vec![false; num_numeric];
        let mut next_helper = num_numeric;
        for numeric_var_id in 0..num_numeric {
            if task.numeric_variables()[numeric_var_id].get_type() != &NumericType::Derived {
                continue;
            }
            let dependencies =
                support.collect_regular_dependencies(task, numeric_var_id, &mut visiting)?;
            if dependencies.len() > 1 {
                support.helper_by_derived[numeric_var_id] = Some(next_helper);
                next_helper += 1;
            }
        }
        Ok(support)
    }

    fn collect_regular_dependencies(
        &mut self,
        task: &dyn AbstractNumericTask,
        numeric_var_id: usize,
        visiting: &mut [bool],
    ) -> Result<BTreeSet<usize>, String> {
        if let Some(cached) = &self.regular_dependencies[numeric_var_id] {
            return Ok(cached.clone());
        }
        if std::mem::replace(&mut visiting[numeric_var_id], true) {
            return Err(format!(
                "cycle in numeric assignment axioms at variable {numeric_var_id}"
            ));
        }
        let dependencies = match task.numeric_variables()[numeric_var_id].get_type() {
            NumericType::Regular => BTreeSet::from([numeric_var_id]),
            NumericType::Constant | NumericType::Cost => BTreeSet::new(),
            NumericType::Derived => {
                let axiom_id = self.assignment_by_affected[numeric_var_id].ok_or_else(|| {
                    format!("derived numeric variable {numeric_var_id} has no assignment axiom")
                })?;
                let axiom = &task.assignment_axioms()[axiom_id];
                let mut dependencies =
                    self.collect_regular_dependencies(task, axiom.get_left_var_id(), visiting)?;
                dependencies.extend(self.collect_regular_dependencies(
                    task,
                    axiom.get_right_var_id(),
                    visiting,
                )?);
                dependencies
            }
        };
        visiting[numeric_var_id] = false;
        self.regular_dependencies[numeric_var_id] = Some(dependencies.clone());
        Ok(dependencies)
    }

    fn helper_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.helper_by_derived.iter().filter_map(|id| *id)
    }

    fn is_nonconstant(&self, task: &dyn AbstractNumericTask, numeric_var_id: usize) -> bool {
        match task.numeric_variables()[numeric_var_id].get_type() {
            NumericType::Regular => true,
            NumericType::Derived => self.regular_dependencies[numeric_var_id]
                .as_ref()
                .is_some_and(|dependencies| !dependencies.is_empty()),
            NumericType::Constant | NumericType::Cost => false,
        }
    }

    fn representative(
        &self,
        task: &dyn AbstractNumericTask,
        numeric_var_id: usize,
    ) -> Option<usize> {
        match task.numeric_variables()[numeric_var_id].get_type() {
            NumericType::Regular => Some(numeric_var_id),
            NumericType::Constant | NumericType::Cost => None,
            NumericType::Derived => self.helper_by_derived[numeric_var_id].or_else(|| {
                self.regular_dependencies[numeric_var_id]
                    .as_ref()
                    .and_then(|dependencies| {
                        (dependencies.len() == 1)
                            .then(|| *dependencies.first().expect("singleton dependency"))
                    })
            }),
        }
    }
}

fn preferred_nonconstant_side(
    task: &dyn AbstractNumericTask,
    left: usize,
    right: usize,
) -> Option<usize> {
    preferred_nonconstant_side_with(left, right, |id| {
        task.numeric_variables()[id].get_type() == &NumericType::Regular
    })
}

fn preferred_nonconstant_side_with(
    left: usize,
    right: usize,
    is_nonconstant: impl Fn(usize) -> bool,
) -> Option<usize> {
    let left_nonconstant = is_nonconstant(left);
    let right_nonconstant = is_nonconstant(right);
    if right_nonconstant {
        Some(right)
    } else if left_nonconstant {
        Some(left)
    } else {
        None
    }
}

fn restricted_numeric_var(task: &dyn AbstractNumericTask, numeric_var_id: usize) -> Option<usize> {
    (task.numeric_variables()[numeric_var_id].get_type() == &NumericType::Regular)
        .then_some(numeric_var_id)
}

fn numeric_effect_target(
    task: &dyn AbstractNumericTask,
    effect: &AssignmentEffect,
) -> Option<CausalGraphVariable> {
    if task
        .numeric_variables()
        .get(effect.affected_var_id())
        .is_none_or(|variable| variable.get_type() != &NumericType::Regular)
    {
        return None;
    }
    if matches!(
        effect.operation(),
        AssignmentOperation::Plus | AssignmentOperation::Minus
    ) && task
        .numeric_variables()
        .get(effect.var_id())
        .is_some_and(|variable| variable.get_type() == &NumericType::Constant)
        && task
            .get_initial_numeric_state_values()
            .get(effect.var_id())
            .copied()
            == Some(0.0)
    {
        return None;
    }
    Some(CausalGraphVariable::Numeric(effect.affected_var_id()))
}

fn comparison_axiom_id_for_var(task: &dyn AbstractNumericTask, var_id: usize) -> Option<usize> {
    task.comparison_axioms()
        .iter()
        .position(|axiom| axiom.get_affected_var_id() == var_id)
}

fn is_regular_propositional_var(task: &dyn AbstractNumericTask, var_id: usize) -> bool {
    task.get_variable_axiom_layer(var_id)
        .unwrap_or(None)
        .is_none()
        && comparison_axiom_id_for_var(task, var_id).is_none()
}
