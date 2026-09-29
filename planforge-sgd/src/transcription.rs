//! The static structure the optimizer runs on.
//!
//! A [`Transcription`] is built once per task and holds nothing that changes
//! during optimization: a flat `(variable, value)` fact layout, the action list
//! with an appended no-op, and the precondition/effect incidence as flat index
//! lists. Everything is stored as parallel `u32` vectors so the same arrays can
//! be handed to a tensor backend as index tensors without rebuilding them.
//!
//! Three normalizations happen here, all of which turn a whole class of runtime
//! special-casing into a one-off preprocessing step:
//!
//! * **Derived variables are constant-folded.** The classical-fragment check guarantees every
//!   axiom is unconditional, so every derived variable has one fixed value.
//!   Derived facts therefore never get a row in the transcription: in a
//!   precondition or effect condition a derived fact is either always satisfied
//!   (drop it) or never satisfied (the operator or effect is dead).
//! * **Effects are grouped by `(action, variable)` in `operator.effects()`
//!   order.** That order is significant: it is the order
//!   `StateRegistry::apply_propositional_effects` writes them in, so a later
//!   effect overwrites an earlier one, and the residual code relies on it to
//!   compute last-write-wins masses.
//! * **Duplicate preconditions are canonicalized per action.** SAS effect
//!   preconditions are hoisted onto the operator, so multiple effects can append
//!   the same literal more than once. A precondition is a Boolean obligation,
//!   not consumable demand: retaining duplicate incidences would make grouped
//!   causal demand exceed one even for an integral action.

#[cfg(test)]
mod tests;

use planforge_sas::numeric_task::{
    AbstractNumericTask, ExplicitValueIndex, NumericTaskExt, VariableIndex,
};

use crate::classical::{NotClassical, check_classical};

/// Why a task could not be turned into a [`Transcription`].
#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptionError {
    /// The task is outside the supported fragment.
    NotClassical(Vec<NotClassical>),
    /// The task has no plan, and we can see that without searching. Callers
    /// should report an unsolvable task, not an internal failure.
    ProvablyUnsolvable(Unsolvable),
    /// Evaluating the initial state's axiom closure failed.
    AxiomEvaluation(String),
}

/// A structural proof that no plan exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsolvable {
    /// A goal requires a derived fact that the axioms never make true. Since
    /// derived values are state-independent here, no state satisfies it.
    UnreachableDerivedGoal { var_id: usize, value: usize },
    /// Two goal facts demand different values of the same variable.
    ContradictoryGoals {
        var_id: usize,
        first: usize,
        second: usize,
    },
}

impl std::fmt::Display for Unsolvable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnreachableDerivedGoal { var_id, value } => write!(
                f,
                "goal requires derived fact var{var_id}={value}, which the axioms never establish"
            ),
            Self::ContradictoryGoals {
                var_id,
                first,
                second,
            } => write!(
                f,
                "goals require both var{var_id}={first} and var{var_id}={second}"
            ),
        }
    }
}

impl std::fmt::Display for TranscriptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotClassical(problems) => {
                write!(
                    f,
                    "task is not a classical task the sgd engine can transcribe:"
                )?;
                for problem in problems {
                    write!(f, "\n  - {problem}")?;
                }
                Ok(())
            }
            Self::ProvablyUnsolvable(reason) => write!(f, "task is unsolvable: {reason}"),
            Self::AxiomEvaluation(message) => {
                write!(f, "failed to evaluate the initial axiom closure: {message}")
            }
        }
    }
}

impl std::error::Error for TranscriptionError {}

/// The static transcription structure. Index vectors are parallel arrays; see
/// the field docs for which index space each one lives in.
#[derive(Debug, Clone)]
pub struct Transcription {
    /// Task variable id of each transcription variable.
    primary_vars: Vec<usize>,
    /// Flat index of each transcription variable's first fact.
    var_offset: Vec<u32>,
    /// Domain size of each transcription variable.
    var_domain: Vec<u32>,
    /// Transcription variable of each flat fact. Length is [`Self::num_facts`].
    var_of_fact: Vec<u32>,

    /// Task operator index of each action, or `None` for the appended no-op.
    action_source: Vec<Option<usize>>,
    /// Index of the appended no-op action.
    noop_action: u32,

    /// Flat fact holding each transcription variable's initial value.
    initial_fact: Vec<u32>,
    /// Goal facts, as flat fact indices.
    goal_facts: Vec<u32>,

    /// Precondition incidence: action and fact of each precondition.
    pre_action: Vec<u32>,
    pre_fact: Vec<u32>,

    /// `(action, variable)` groups: one entry per group.
    group_action: Vec<u32>,
    group_var: Vec<u32>,
    /// Effect list of each group, in CSR form over [`Self::group_effect_order`].
    /// Effects of one operator on *different* variables interleave in
    /// `operator.effects()`, so a group's effects are not contiguous in effect
    /// index space and this indirection is what orders them.
    group_effect_start: Vec<u32>,
    /// Effect indices ordered by group, and within a group in
    /// `operator.effects()` order — that is, last write wins.
    group_effect_order: Vec<u32>,
    /// Group of each effect.
    effect_group: Vec<u32>,
    effect_fact: Vec<u32>,
    /// Effect conditions, as a segmented list over effects.
    cond_effect: Vec<u32>,
    cond_fact: Vec<u32>,
    /// Largest number of effects in any single group. When this is 1 the
    /// last-write-wins arithmetic collapses to the identity.
    max_group_size: usize,

    /// Operators dropped because a folded derived precondition can never hold,
    /// so they are inapplicable in every state.
    dropped_operators: Vec<String>,
}

impl Transcription {
    pub fn num_variables(&self) -> usize {
        self.primary_vars.len()
    }
    pub fn num_facts(&self) -> usize {
        self.var_of_fact.len()
    }
    pub fn num_actions(&self) -> usize {
        self.action_source.len()
    }
    pub fn noop_action(&self) -> usize {
        self.noop_action as usize
    }
    pub fn num_groups(&self) -> usize {
        self.group_action.len()
    }
    pub fn num_effects(&self) -> usize {
        self.effect_group.len()
    }
    pub fn max_group_size(&self) -> usize {
        self.max_group_size
    }
    pub fn primary_vars(&self) -> &[usize] {
        &self.primary_vars
    }
    pub fn var_domain(&self) -> &[u32] {
        &self.var_domain
    }
    pub fn var_offset(&self) -> &[u32] {
        &self.var_offset
    }
    pub fn var_of_fact(&self) -> &[u32] {
        &self.var_of_fact
    }
    pub fn initial_fact(&self) -> &[u32] {
        &self.initial_fact
    }
    pub fn goal_facts(&self) -> &[u32] {
        &self.goal_facts
    }
    pub fn pre_action(&self) -> &[u32] {
        &self.pre_action
    }
    pub fn pre_fact(&self) -> &[u32] {
        &self.pre_fact
    }
    pub fn group_action(&self) -> &[u32] {
        &self.group_action
    }
    pub fn group_var(&self) -> &[u32] {
        &self.group_var
    }
    /// Effect indices of `group`, in last-write-wins order.
    pub fn group_effects(&self, group: usize) -> &[u32] {
        let start = self.group_effect_start[group] as usize;
        let end = self.group_effect_start[group + 1] as usize;
        &self.group_effect_order[start..end]
    }
    pub fn effect_group(&self) -> &[u32] {
        &self.effect_group
    }
    pub fn effect_fact(&self) -> &[u32] {
        &self.effect_fact
    }
    pub fn cond_effect(&self) -> &[u32] {
        &self.cond_effect
    }
    pub fn cond_fact(&self) -> &[u32] {
        &self.cond_fact
    }
    pub fn dropped_operators(&self) -> &[String] {
        &self.dropped_operators
    }
    /// Task operator index of `action`, or `None` for the no-op.
    pub fn action_source(&self, action: usize) -> Option<usize> {
        self.action_source[action]
    }

    /// Flat fact index of `(variable, value)` in transcription variable space.
    pub fn fact(&self, var: usize, value: usize) -> u32 {
        debug_assert!(value < self.var_domain[var] as usize);
        self.var_offset[var] + value as u32
    }

    /// Build the transcription for `task`, appending a no-op action.
    pub fn build<T: AbstractNumericTask + ?Sized>(
        task: &T,
    ) -> Result<Transcription, TranscriptionError> {
        check_classical(task).map_err(TranscriptionError::NotClassical)?;

        // Derived values are state-independent, so one axiom closure of the
        // initial state gives their value everywhere.
        let (folded_values, _) = task
            .evaluated_initial_abstract_state_values()
            .map_err(TranscriptionError::AxiomEvaluation)?;

        let num_task_vars = task.get_num_variables();
        let derived: Vec<bool> = (0..num_task_vars)
            .map(|v| {
                task.get_variable_axiom_layer(VariableIndex::from_usize(v))
                    .ok()
                    .flatten()
                    .is_some()
            })
            .collect();

        // Transcription variables are the primary ones, in task order.
        let mut primary_vars = Vec::new();
        let mut local_of_task_var = vec![u32::MAX; num_task_vars];
        for task_var in 0..num_task_vars {
            if !derived[task_var] {
                local_of_task_var[task_var] = primary_vars.len() as u32;
                primary_vars.push(task_var);
            }
        }

        let mut var_offset = Vec::with_capacity(primary_vars.len());
        let mut var_domain = Vec::with_capacity(primary_vars.len());
        let mut var_of_fact = Vec::new();
        for (local, &task_var) in primary_vars.iter().enumerate() {
            let size = task
                .get_variable_domain_size(VariableIndex::from_usize(task_var))
                .expect("the classical-fragment check validated variable ranges");
            var_offset.push(var_of_fact.len() as u32);
            var_domain.push(size as u32);
            var_of_fact.extend(std::iter::repeat_n(local as u32, size));
        }

        // `true` if a fact on a derived variable holds under the folded values.
        let derived_fact_holds = |var: usize, value: usize| -> bool {
            folded_values.get(var).copied() == Some(ExplicitValueIndex::from_usize(value))
        };

        let initial_fact = primary_vars
            .iter()
            .map(|&task_var| {
                let value = folded_values
                    .get(task_var)
                    .copied()
                    .expect("folded values cover every task variable");
                var_offset[local_of_task_var[task_var] as usize] + value.index() as u32
            })
            .collect();

        // Goals. A derived goal is decided now; a primary goal becomes a fact.
        let mut goal_facts: Vec<u32> = Vec::new();
        let mut goal_value_of_var: Vec<Option<usize>> = vec![None; num_task_vars];
        for index in 0..task.get_num_goals() {
            let goal = task.get_goal_fact(index);
            if derived[goal.var()] {
                if !derived_fact_holds(goal.var(), goal.value()) {
                    return Err(TranscriptionError::ProvablyUnsolvable(
                        Unsolvable::UnreachableDerivedGoal {
                            var_id: goal.var(),
                            value: goal.value(),
                        },
                    ));
                }
                continue;
            }
            match goal_value_of_var[goal.var()] {
                Some(existing) if existing != goal.value() => {
                    return Err(TranscriptionError::ProvablyUnsolvable(
                        Unsolvable::ContradictoryGoals {
                            var_id: goal.var(),
                            first: existing,
                            second: goal.value(),
                        },
                    ));
                }
                Some(_) => continue,
                None => goal_value_of_var[goal.var()] = Some(goal.value()),
            }
            let local = local_of_task_var[goal.var()] as usize;
            goal_facts.push(var_offset[local] + goal.value() as u32);
        }

        let mut pre_action = Vec::new();
        let mut pre_fact = Vec::new();
        let mut group_action = Vec::new();
        let mut group_var = Vec::new();
        let mut effect_group = Vec::new();
        let mut effect_fact = Vec::new();
        let mut cond_effect = Vec::new();
        let mut cond_fact = Vec::new();
        let mut action_source = Vec::new();
        let mut dropped_operators = Vec::new();
        // Generation stamps avoid clearing an O(num_facts) set for every
        // operator while preserving the first occurrence of each local fact.
        let mut precondition_seen_in_operator = vec![usize::MAX; var_of_fact.len()];

        for (op_index, operator) in task.get_operators().iter().enumerate() {
            // A precondition on a derived variable is decided up front: either
            // it always holds and carries no information, or the operator is
            // inapplicable in every state and can be removed.
            let dead = operator
                .preconditions()
                .iter()
                .any(|fact| derived[fact.var()] && !derived_fact_holds(fact.var(), fact.value()));
            if dead {
                dropped_operators.push(operator.name().to_string());
                continue;
            }

            let action = action_source.len() as u32;
            action_source.push(Some(op_index));

            for fact in operator.preconditions() {
                if derived[fact.var()] {
                    continue;
                }
                let local = local_of_task_var[fact.var()] as usize;
                let local_fact = var_offset[local] + fact.value() as u32;
                let seen_at = &mut precondition_seen_in_operator[local_fact as usize];
                if *seen_at == op_index {
                    continue;
                }
                *seen_at = op_index;
                pre_action.push(action);
                pre_fact.push(local_fact);
            }

            // Effects grouped by affected variable, preserving their original
            // relative order inside each group.
            let mut group_of_var: Vec<Option<u32>> = vec![None; num_task_vars];
            for effect in operator.effects() {
                // The classical-fragment check guarantees the affected variable is primary.
                debug_assert!(!derived[effect.var_id().index()]);

                // A condition on a derived variable is likewise decided now.
                let never_fires = effect.conditions().iter().any(|fact| {
                    derived[fact.var()] && !derived_fact_holds(fact.var(), fact.value())
                });
                if never_fires {
                    continue;
                }

                let group = match group_of_var[effect.var_id().index()] {
                    Some(group) => group,
                    None => {
                        let group = group_action.len() as u32;
                        group_action.push(action);
                        group_var.push(local_of_task_var[effect.var_id().index()]);
                        group_of_var[effect.var_id().index()] = Some(group);
                        group
                    }
                };

                let effect_index = effect_group.len() as u32;
                let local = local_of_task_var[effect.var_id().index()] as usize;
                effect_group.push(group);
                effect_fact.push(var_offset[local] + effect.value().index() as u32);

                for fact in effect.conditions() {
                    if derived[fact.var()] {
                        continue;
                    }
                    let local = local_of_task_var[fact.var()] as usize;
                    cond_effect.push(effect_index);
                    cond_fact.push(var_offset[local] + fact.value() as u32);
                }
            }
        }

        // The generic no-op: no preconditions, no effects. It lets a plan
        // shorter than the horizon occupy the remaining slots.
        let noop_action = action_source.len() as u32;
        action_source.push(None);

        // CSR-order the effects by group. A counting sort keeps each group's
        // effects in their original relative order, which is exactly the
        // last-write-wins order the residuals need.
        let num_groups = group_action.len();
        let mut group_effect_start = vec![0u32; num_groups + 1];
        for &group in &effect_group {
            group_effect_start[group as usize + 1] += 1;
        }
        let max_group_size = group_effect_start
            .iter()
            .skip(1)
            .copied()
            .max()
            .unwrap_or(0) as usize;
        for index in 0..num_groups {
            group_effect_start[index + 1] += group_effect_start[index];
        }
        let mut cursor = group_effect_start.clone();
        let mut group_effect_order = vec![0u32; effect_group.len()];
        for (effect, &group) in effect_group.iter().enumerate() {
            let slot = &mut cursor[group as usize];
            group_effect_order[*slot as usize] = effect as u32;
            *slot += 1;
        }

        Ok(Transcription {
            primary_vars,
            var_offset,
            var_domain,
            var_of_fact,
            action_source,
            noop_action,
            initial_fact,
            goal_facts,
            pre_action,
            pre_fact,
            group_action,
            group_var,
            group_effect_start,
            group_effect_order,
            effect_group,
            effect_fact,
            cond_effect,
            cond_fact,
            max_group_size,
            dropped_operators,
        })
    }
}
